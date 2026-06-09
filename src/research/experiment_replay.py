"""Safe replay smoke checks for Labs experiment artifacts."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.paths import PROJECT_ROOT
from src.research.experiment_artifact_validator import LABS_REPLAY_SCHEMA_VERSION

DEFAULT_METRIC_TOLERANCES: dict[str, float] = {
    "sharpe": 0.01,
    "cagr_pct": 0.1,
    "max_drawdown_pct": 0.1,
    "wfe": 0.01,
    "dsr": 0.001,
    "num_windows": 0.0,
    "config_count": 0.0,
}

ALLOWED_REPLAY_MODULE_PREFIXES = ("src.",)
ALLOWED_REPLAY_SCRIPT_SUFFIXES = {".py", ".sh"}
SHELL_CONTROL_FRAGMENTS = ("&&", "||", ";", "|", "&", "`", "$(", "<", ">")

__all__ = [
    "DEFAULT_METRIC_TOLERANCES",
    "ExperimentReplayResult",
    "MetricComparison",
    "ReplaySafetyError",
    "replay_experiment",
]


class ReplaySafetyError(RuntimeError):
    """Raised when a replay target is unsafe without explicit approval."""


@dataclass(frozen=True)
class MetricComparison:
    """Comparison for one replayed metric."""

    metric: str
    expected: float
    actual: float | None
    tolerance: float

    @property
    def delta(self) -> float | None:
        if self.actual is None:
            return None
        return self.actual - self.expected

    @property
    def passed(self) -> bool:
        if self.delta is None:
            return False
        return abs(self.delta) <= self.tolerance


@dataclass(frozen=True)
class ExperimentReplayResult:
    """Result of one Labs experiment replay smoke."""

    experiment_id: str
    artifact_path: str
    command: str
    provenance_status: str
    duration_seconds: float
    comparisons: tuple[MetricComparison, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and all(comparison.passed for comparison in self.comparisons)

    @property
    def status(self) -> str:
        return "passed" if self.passed else "failed"

    @property
    def metric_deltas(self) -> dict[str, float]:
        return {
            comparison.metric: comparison.delta
            for comparison in self.comparisons
            if comparison.delta is not None and math.isfinite(comparison.delta)
        }

    def as_labs_replay(self) -> dict[str, Any]:
        """Return a dashboard-compatible Labs replay artifact."""
        failed_metric_count = sum(0 if comparison.passed else 1 for comparison in self.comparisons)
        max_abs_delta = max((abs(delta) for delta in self.metric_deltas.values()), default=0.0)
        return {
            "schema_version": LABS_REPLAY_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "artifact_path": self.artifact_path,
            "status": self.status,
            "provenance_status": self.provenance_status,
            "passed": self.passed,
            "command": self.command,
            "duration_seconds": self.duration_seconds,
            "metric_deltas": self.metric_deltas,
            "metrics": {
                "metrics_compared": len(self.comparisons),
                "failed_metric_count": failed_metric_count,
                "max_abs_metric_delta": max_abs_delta,
                "duration_seconds": self.duration_seconds,
                "returncode": self.returncode,
            },
            "baseline_deltas": self.metric_deltas,
        }


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _coerce_metric_map(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping of numeric metrics")

    metrics: dict[str, float] = {}
    for key, metric_value in value.items():
        if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
            raise ValueError(f"{label}.{key} must be numeric")
        numeric_value = float(metric_value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"{label}.{key} must be finite")
        metrics[str(key)] = numeric_value
    return metrics


def _load_expected_metrics(target: Mapping[str, Any], artifact_path: Path) -> dict[str, float]:
    if "metrics" in target:
        return _coerce_metric_map(target["metrics"], label="target.metrics")

    with open(artifact_path) as f:
        artifact = json.load(f)

    if isinstance(artifact, Mapping):
        if isinstance(artifact.get("metrics"), Mapping):
            return _coerce_metric_map(artifact["metrics"], label="artifact.metrics")
        if isinstance(artifact.get("summary"), Mapping):
            return _coerce_metric_map(artifact["summary"], label="artifact.summary")

    raise ValueError("target metrics are required when artifact has no metrics or summary object")


def _command_args(command: Any) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    if isinstance(command, Sequence) and not isinstance(command, (bytes, bytearray)):
        return [str(part) for part in command]
    raise ValueError("target.command must be a command string or sequence")


def _is_python_executable(command: str) -> bool:
    executable = Path(command).name.lower()
    return executable in {"python", "python3"} or executable.startswith("python3.")


def _is_within_project(path: Path, project_root: Path) -> bool:
    try:
        path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return False
    return True


def _project_local_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _reject_shell_control(command_args: Sequence[str], experiment_id: str) -> None:
    for part in command_args:
        if any(fragment in part for fragment in SHELL_CONTROL_FRAGMENTS):
            raise ReplaySafetyError(f"{experiment_id} replay command contains shell control syntax")


def _assert_replay_command_allowed(command_args: Sequence[str], *, project_root: Path, experiment_id: str) -> None:
    """Fail closed unless the replay command is a narrow project-local target."""
    if not command_args:
        raise ValueError("target.command must not be empty")

    _reject_shell_control(command_args, experiment_id)
    executable = command_args[0]
    if _is_python_executable(executable):
        python_args = list(command_args[1:])
        while python_args and python_args[0] in {"-B", "-u"}:
            python_args.pop(0)

        if len(python_args) >= 2 and python_args[0] == "-m":
            module = python_args[1]
            if module.startswith(ALLOWED_REPLAY_MODULE_PREFIXES):
                return
            raise ReplaySafetyError(f"{experiment_id} replay module is not allowlisted: {module}")

        if not python_args or python_args[0].startswith("-"):
            raise ReplaySafetyError(f"{experiment_id} replay command is not allowlisted")

        script_path = _project_local_path(python_args[0], project_root)
        if not _is_within_project(script_path, project_root):
            raise ReplaySafetyError(f"{experiment_id} replay script is outside project root: {python_args[0]}")
        if script_path.suffix.lower() in ALLOWED_REPLAY_SCRIPT_SUFFIXES:
            return
        raise ReplaySafetyError(f"{experiment_id} replay script suffix is not allowlisted: {python_args[0]}")

    executable_path = _project_local_path(executable, project_root)
    if _is_within_project(executable_path, project_root) and executable_path.suffix.lower() in ALLOWED_REPLAY_SCRIPT_SUFFIXES:
        return

    raise ReplaySafetyError(f"{experiment_id} replay command is not allowlisted: {executable}")


def _parse_stdout_json(stdout: str) -> Mapping[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return payload
    raise ValueError("replay command did not emit a JSON object on stdout")


def _extract_actual_metrics(stdout: str) -> dict[str, float]:
    payload = _parse_stdout_json(stdout)
    if isinstance(payload.get("metrics"), Mapping):
        return _coerce_metric_map(payload["metrics"], label="stdout.metrics")
    return _coerce_metric_map(payload, label="stdout")


def _target_artifact_path(target: Mapping[str, Any]) -> str:
    value = target.get("artifact_path") or target.get("source_artifact_path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("target artifact_path or source_artifact_path is required")
    return value


def _build_comparisons(
    expected_metrics: Mapping[str, float],
    actual_metrics: Mapping[str, float],
    tolerances: Mapping[str, float],
) -> tuple[MetricComparison, ...]:
    comparisons: list[MetricComparison] = []
    for metric, expected_value in expected_metrics.items():
        tolerance = float(tolerances.get(metric, DEFAULT_METRIC_TOLERANCES.get(metric, 0.0)))
        actual_value = actual_metrics.get(metric)
        comparisons.append(
            MetricComparison(
                metric=metric,
                expected=expected_value,
                actual=actual_value,
                tolerance=tolerance,
            )
        )
    return tuple(comparisons)


def replay_experiment(
    target: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
    allow_market_data_fetch: bool = False,
    allow_expensive: bool = False,
    timeout_seconds: float = 30.0,
) -> ExperimentReplayResult:
    """Replay a registry row or provenance manifest and compare core metrics.

    Safe mode is the default: targets must be marked ``replay_safe`` and any
    target marked ``fetches_market_data`` requires an explicit fetch override.
    """
    experiment_id = target.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("target.experiment_id is required")

    if not _coerce_bool(target.get("replay_safe", False)) and not allow_expensive:
        raise ReplaySafetyError(f"{experiment_id} is not marked replay_safe; pass allow_expensive=True to run it")

    if _coerce_bool(target.get("fetches_market_data", False)) and not allow_market_data_fetch:
        raise ReplaySafetyError(
            f"{experiment_id} is marked as a market data fetch; pass allow_market_data_fetch=True to run it"
        )

    command = target.get("command")
    if command is None:
        raise ValueError("target.command is required")
    command_args = _command_args(command)
    cwd = Path(project_root or PROJECT_ROOT)
    _assert_replay_command_allowed(command_args, project_root=cwd, experiment_id=experiment_id)
    command_text = " ".join(shlex.quote(part) for part in command_args)

    artifact_path = _target_artifact_path(target)
    artifact_file = Path(artifact_path)
    if not artifact_file.is_absolute():
        artifact_file = Path(project_root or PROJECT_ROOT) / artifact_file
    expected_metrics = _load_expected_metrics(target, artifact_file)
    tolerances = _coerce_metric_map(target.get("metric_tolerances", {}), label="target.metric_tolerances")

    env = os.environ.copy()
    env.setdefault("PORTFOLIO_LAB_ENABLE_ML", "0")
    env["PORTFOLIO_LAB_REPLAY_ALLOW_MARKET_DATA_FETCH"] = "1" if allow_market_data_fetch else "0"

    start = time.perf_counter()
    completed = subprocess.run(
        command_args,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    duration_seconds = time.perf_counter() - start

    actual_metrics: dict[str, float] = {}
    if completed.returncode == 0:
        actual_metrics = _extract_actual_metrics(completed.stdout)

    return ExperimentReplayResult(
        experiment_id=experiment_id,
        artifact_path=artifact_path,
        command=command_text,
        provenance_status=str(target.get("provenance_status", "unknown")),
        duration_seconds=duration_seconds,
        comparisons=_build_comparisons(expected_metrics, actual_metrics, tolerances),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
