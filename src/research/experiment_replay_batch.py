"""Batch runner for safe Labs replay reports."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.paths import DATA_DIR, PROJECT_ROOT, PUBLIC_DATA_DIR
from src.research.experiment_artifact_validator import LABS_REPLAY_SCHEMA_VERSION, validate_artifact
from src.research.experiment_replay import ReplaySafetyError, replay_experiment

LABS_REPLAYS_FILENAME = "labs_replays.json"
LABS_REPLAY_TARGETS_FILENAME = "labs_replay_targets.json"
_TARGET_SOURCE_KEY = "_replay_target_source"
MAX_DIAGNOSTIC_MESSAGE_LENGTH = 240
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)\s*=\s*[^,\s;]+"
)

__all__ = [
    "LABS_REPLAYS_FILENAME",
    "LABS_REPLAY_TARGETS_FILENAME",
    "ReplayBatchResult",
    "load_replay_targets",
    "main",
    "publish_labs_replays",
    "run_replay_batch",
]


@dataclass(frozen=True)
class ReplayBatchResult:
    """Summary of one batch replay report run."""

    output_path: Path
    rows: tuple[dict[str, Any], ...]
    duplicate_targets: tuple[dict[str, str], ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        counts = {"passed": 0, "failed": 0, "warning": 0}
        for row in self.rows:
            status = row.get("status")
            if status in counts:
                counts[status] += 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "counts": self.counts,
            "rows": len(self.rows),
            "duplicate_targets": list(self.duplicate_targets),
        }


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _mapping_rows(value: Any, *, source: Path) -> list[dict[str, Any]]:
    if isinstance(value, Mapping) and isinstance(value.get("experiments"), list):
        rows = [dict(row) for row in value["experiments"] if isinstance(row, Mapping)]
        for row in rows:
            row[_TARGET_SOURCE_KEY] = str(source)
        return rows
    if isinstance(value, Mapping):
        row = dict(value)
        row[_TARGET_SOURCE_KEY] = str(source)
        return [row]
    if isinstance(value, list):
        rows = [dict(row) for row in value if isinstance(row, Mapping)]
        for row in rows:
            row[_TARGET_SOURCE_KEY] = str(source)
        return rows
    raise ValueError(f"{source} must contain an object, a registry object, or an array of objects")


def _load_replay_target_rows(input_paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for input_path in input_paths:
        path = Path(input_path)
        targets.extend(_mapping_rows(_load_json(path), source=path))
    return targets


def _dedup_key(target: Mapping[str, Any]) -> str | None:
    value = target.get("experiment_id")
    if isinstance(value, str) and value.strip():
        return value
    return None


def _target_source(target: Mapping[str, Any]) -> str:
    value = target.get(_TARGET_SOURCE_KEY)
    return str(value) if isinstance(value, str) else ""


def _duplicate_target_diagnostic(
    experiment_id: str,
    *,
    retained: Mapping[str, Any],
    duplicate: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "experiment_id": experiment_id,
        "retained_source": _target_source(retained),
        "duplicate_source": _target_source(duplicate),
        "retained_artifact_path": _artifact_path(retained),
        "duplicate_artifact_path": _artifact_path(duplicate),
    }


def _deduplicate_replay_targets(targets: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], tuple[dict[str, str], ...]]:
    retained_by_id: dict[str, dict[str, Any]] = {}
    deduped: list[dict[str, Any]] = []
    duplicate_targets: list[dict[str, str]] = []

    for target in targets:
        experiment_id = _dedup_key(target)
        if experiment_id is None:
            deduped.append(target)
            continue

        retained = retained_by_id.get(experiment_id)
        if retained is not None:
            duplicate_targets.append(
                _duplicate_target_diagnostic(experiment_id, retained=retained, duplicate=target)
            )
            continue

        retained_by_id[experiment_id] = target
        deduped.append(target)

    return deduped, tuple(duplicate_targets)


def _load_replay_targets_with_diagnostics(
    input_paths: Sequence[str | Path],
) -> tuple[list[dict[str, Any]], tuple[dict[str, str], ...]]:
    return _deduplicate_replay_targets(_load_replay_target_rows(input_paths))


def _without_internal_target_metadata(target: Mapping[str, Any]) -> dict[str, Any]:
    clean_target = dict(target)
    clean_target.pop(_TARGET_SOURCE_KEY, None)
    return clean_target


def load_replay_targets(input_paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load deduplicated replay targets from registry-like JSON files or provenance manifests."""
    targets, _duplicate_targets = _load_replay_targets_with_diagnostics(input_paths)
    return [_without_internal_target_metadata(target) for target in targets]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _target_id(target: Mapping[str, Any]) -> str:
    value = target.get("experiment_id")
    return str(value) if isinstance(value, str) and value.strip() else "unknown"


def _artifact_path(target: Mapping[str, Any]) -> str:
    value = target.get("artifact_path") or target.get("source_artifact_path")
    return str(value) if isinstance(value, str) else ""


def _command_text(target: Mapping[str, Any]) -> str | None:
    value = target.get("command")
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return " ".join(str(part) for part in value)
    return str(value)


def _base_replay_payload(target: Mapping[str, Any], *, status: str, passed: bool) -> dict[str, Any]:
    return {
        "schema_version": LABS_REPLAY_SCHEMA_VERSION,
        "experiment_id": _target_id(target),
        "generated_at": _now_iso(),
        "artifact_path": _artifact_path(target),
        "status": status,
        "provenance_status": str(target.get("provenance_status", "unknown")),
        "passed": passed,
        "command": _command_text(target),
        "duration_seconds": 0.0,
        "metric_deltas": {},
        "metrics": {},
        "baseline_deltas": {},
    }


def _sanitize_diagnostic_message(message: str) -> str:
    redacted = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=[redacted]", message)
    normalized = " ".join(redacted.split())
    if len(normalized) <= MAX_DIAGNOSTIC_MESSAGE_LENGTH:
        return normalized
    return normalized[: MAX_DIAGNOSTIC_MESSAGE_LENGTH - 3].rstrip() + "..."


def _diagnostic_fields(*, failure_reason: str, error_type: str, error_message: str) -> dict[str, str]:
    return {
        "failure_reason": failure_reason,
        "error_type": _sanitize_diagnostic_message(error_type),
        "error_message": _sanitize_diagnostic_message(error_message),
    }


def _validation_failure_message(result: Any) -> str:
    failed_metrics = sorted(
        comparison.metric
        for comparison in result.comparisons
        if not comparison.passed
    )
    if not failed_metrics:
        return "replay validation failed"
    return f"failed replay metrics: {', '.join(failed_metrics)}"


def _skipped_replay_payload(
    target: Mapping[str, Any],
    *,
    safety_skip: bool,
    error_message: str = "",
) -> dict[str, Any]:
    payload = _base_replay_payload(target, status="warning", passed=False)
    payload["metrics"] = {
        "skipped_count": 1,
        "safety_skip_count": 1 if safety_skip else 0,
        "failed_metric_count": 0,
        "max_abs_metric_delta": 0.0,
    }
    payload.update(
        _diagnostic_fields(
            failure_reason="safety_skip" if safety_skip else "unexpected_error",
            error_type="ReplaySafetyError" if safety_skip else "ReplaySkip",
            error_message=error_message or "replay target was skipped",
        )
    )
    return payload


def _failed_replay_payload(
    target: Mapping[str, Any],
    *,
    timeout: bool = False,
    failure_reason: str = "unexpected_error",
    error_type: str = "ReplayError",
    error_message: str = "replay failed",
) -> dict[str, Any]:
    payload = _base_replay_payload(target, status="failed", passed=False)
    payload["metrics"] = {
        "error_count": 1,
        "timeout_count": 1 if timeout else 0,
        "failed_metric_count": 1,
        "max_abs_metric_delta": 0.0,
    }
    payload.update(
        _diagnostic_fields(
            failure_reason=failure_reason,
            error_type=error_type,
            error_message=error_message,
        )
    )
    return payload


def _add_result_diagnostics(row: dict[str, Any], result: Any) -> dict[str, Any]:
    if result.passed:
        return row

    if result.returncode != 0:
        row.update(
            _diagnostic_fields(
                failure_reason="command_failure",
                error_type="ReplayCommandFailure",
                error_message=f"replay command exited with return code {result.returncode}",
            )
        )
        return row

    row.update(
        _diagnostic_fields(
            failure_reason="validation_failure",
            error_type="ReplayValidationFailure",
            error_message=_validation_failure_message(result),
        )
    )
    return row


def _validate_replay_row(row: Mapping[str, Any]) -> None:
    result = validate_artifact(row)
    if not result.valid:
        raise ValueError(f"generated replay row failed validation: {result.error_messages()}")


def _validate_concurrency(concurrency: int) -> int:
    if isinstance(concurrency, bool) or int(concurrency) < 1:
        raise ValueError("concurrency must be a positive integer")
    return int(concurrency)


def _run_replay_target(
    target: Mapping[str, Any],
    *,
    root: Path,
    allow_market_data_fetch: bool,
    allow_expensive: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        result = replay_experiment(
            target,
            project_root=root,
            allow_market_data_fetch=allow_market_data_fetch,
            allow_expensive=allow_expensive,
            timeout_seconds=timeout_seconds,
        )
        row = _add_result_diagnostics(result.as_labs_replay(), result)
    except ReplaySafetyError as exc:
        row = _skipped_replay_payload(target, safety_skip=True, error_message=str(exc))
    except subprocess.TimeoutExpired:
        row = _failed_replay_payload(
            target,
            timeout=True,
            failure_reason="timeout",
            error_type="TimeoutExpired",
            error_message=f"replay command timed out after {timeout_seconds:g} seconds",
        )
    except Exception as exc:
        row = _failed_replay_payload(
            target,
            failure_reason="unexpected_error",
            error_type=exc.__class__.__name__,
            error_message=str(exc),
        )
    _validate_replay_row(row)
    return row


def run_replay_batch(
    input_paths: Sequence[str | Path],
    *,
    output_path: str | Path = PUBLIC_DATA_DIR / LABS_REPLAYS_FILENAME,
    project_root: str | Path | None = None,
    allow_market_data_fetch: bool = False,
    allow_expensive: bool = False,
    timeout_seconds: float = 30.0,
    concurrency: int = 1,
) -> ReplayBatchResult:
    """Run safe replay checks for many targets and write a Labs replay report."""
    root = Path(project_root or PROJECT_ROOT)
    concurrency = _validate_concurrency(concurrency)
    targets, duplicate_targets = _load_replay_targets_with_diagnostics(input_paths)

    if concurrency == 1:
        rows = [
            _run_replay_target(
                target,
                root=root,
                allow_market_data_fetch=allow_market_data_fetch,
                allow_expensive=allow_expensive,
                timeout_seconds=timeout_seconds,
            )
            for target in targets
        ]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    _run_replay_target,
                    target,
                    root=root,
                    allow_market_data_fetch=allow_market_data_fetch,
                    allow_expensive=allow_expensive,
                    timeout_seconds=timeout_seconds,
                )
                for target in targets
            ]
            rows = [future.result() for future in futures]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from src.monitor.signal_authority import (
        is_ephemeral_write_path,
        serialize_json_payload,
    )

    path.write_text(
        serialize_json_payload(
            rows,
            output_path=path,
            public=not is_ephemeral_write_path(path),
        ),
        encoding="utf-8",
    )
    return ReplayBatchResult(output_path=path, rows=tuple(rows), duplicate_targets=duplicate_targets)


def _validate_replay_report_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("precomputed Labs replay report must be an array")

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ValueError(f"precomputed Labs replay row {index} must be an object")
        validation = validate_artifact(row)
        if not validation.valid:
            raise ValueError(f"precomputed Labs replay row {index} failed validation: {validation.error_messages()}")
        rows.append(dict(row))
    return rows


def _write_replay_rows(rows: Sequence[Mapping[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from src.monitor.signal_authority import (
        is_ephemeral_write_path,
        serialize_json_payload,
    )

    output_path.write_text(
        serialize_json_payload(
            [dict(row) for row in rows],
            output_path=output_path,
            public=not is_ephemeral_write_path(output_path),
        ),
        encoding="utf-8",
    )
    return output_path


def _dashboard_project_root(data_dir: Path, project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root)
    return data_dir.parent if data_dir.name == "data" else data_dir


def publish_labs_replays(
    *,
    data_dir: str | Path = DATA_DIR,
    public_dir: str | Path = PUBLIC_DATA_DIR,
    target_path: str | Path | None = None,
    precomputed_report_path: str | Path | None = None,
    output_path: str | Path | None = None,
    project_root: str | Path | None = None,
    allow_market_data_fetch: bool = False,
    allow_expensive: bool = False,
    timeout_seconds: float = 30.0,
    concurrency: int = 1,
) -> Path | None:
    """Publish an explicit Labs replay report for dashboard generation.

    The dashboard path is fail-closed: absent inputs are a no-op, explicit target
    files run through the existing safe replay batch runner, and precomputed
    reports are copied only after every row validates as ``labs-replay/v1``.
    """
    data_dir_path = Path(data_dir)
    target_file = Path(target_path) if target_path is not None else data_dir_path / LABS_REPLAY_TARGETS_FILENAME
    precomputed_file = (
        Path(precomputed_report_path)
        if precomputed_report_path is not None
        else data_dir_path / LABS_REPLAYS_FILENAME
    )
    output_file = Path(output_path) if output_path is not None else Path(public_dir) / LABS_REPLAYS_FILENAME

    if target_file.exists():
        return run_replay_batch(
            [target_file],
            output_path=output_file,
            project_root=_dashboard_project_root(data_dir_path, project_root),
            allow_market_data_fetch=allow_market_data_fetch,
            allow_expensive=allow_expensive,
            timeout_seconds=timeout_seconds,
            concurrency=concurrency,
        ).output_path

    if precomputed_file.exists():
        rows = _validate_replay_report_rows(_load_json(precomputed_file))
        return _write_replay_rows(rows, output_file)

    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run safe Labs replay checks in batch")
    parser.add_argument("inputs", nargs="+", help="Registry, target-list, or provenance manifest JSON files")
    parser.add_argument(
        "--output",
        default=str(PUBLIC_DATA_DIR / LABS_REPLAYS_FILENAME),
        help="Output labs_replays.json path",
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root for relative artifacts")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="Per-target replay timeout")
    parser.add_argument("--concurrency", type=int, default=1, help="Max replay targets to run concurrently")
    parser.add_argument("--allow-market-data-fetch", action="store_true", help="Allow targets marked as data-fetching")
    parser.add_argument("--allow-expensive", action="store_true", help="Allow targets not marked replay_safe")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for batch Labs replay reporting."""
    args = _build_parser().parse_args(argv)
    result = run_replay_batch(
        args.inputs,
        output_path=args.output,
        project_root=args.project_root,
        allow_market_data_fetch=args.allow_market_data_fetch,
        allow_expensive=args.allow_expensive,
        timeout_seconds=args.timeout_seconds,
        concurrency=args.concurrency,
    )
    sys.stdout.write(json.dumps(result.as_dict(), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
