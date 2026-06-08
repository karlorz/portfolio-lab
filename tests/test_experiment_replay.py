"""Tests for safe Labs experiment replay smoke checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.research.experiment_artifact_validator import validate_artifact
from src.research.experiment_replay import ReplaySafetyError, replay_experiment


EXPECTED_METRICS = {
    "sharpe": 0.95,
    "cagr_pct": 10.4,
    "max_drawdown_pct": -25.0,
    "wfe": 1.39,
    "dsr": 0.979,
    "num_windows": 15,
    "config_count": 109,
}


def _write_fixture_experiment(tmp_path: Path, metrics: dict[str, float], *, marker: Path | None = None) -> str:
    script_path = tmp_path / "fixture_experiment.py"
    script_path.write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                f"marker = {str(marker)!r}",
                "if marker:",
                "    Path(marker).write_text('ran')",
                f"print(json.dumps({{'metrics': {json.dumps(metrics)}}}))",
            ]
        )
    )
    return f"{sys.executable} {script_path}"


def _registry_target(tmp_path: Path, command: str, **overrides) -> dict:
    artifact_path = tmp_path / "saved_experiment.json"
    artifact_path.write_text(json.dumps({"metrics": EXPECTED_METRICS}))
    target = {
        "experiment_id": "fixture-gold-sweep",
        "artifact_path": str(artifact_path),
        "command": command,
        "replay_safe": True,
        "fetches_market_data": False,
        "metrics": EXPECTED_METRICS,
        "metric_tolerances": {
            "sharpe": 0.01,
            "cagr_pct": 0.1,
            "max_drawdown_pct": 0.1,
            "wfe": 0.01,
            "dsr": 0.001,
            "num_windows": 0.0,
            "config_count": 0.0,
        },
        "provenance_status": "present",
    }
    target.update(overrides)
    return target


def test_replay_experiment_emits_labs_payload_when_metrics_match(tmp_path: Path) -> None:
    """Fixture replay should compare core metrics and emit schema-compatible Labs replay JSON."""
    actual_metrics = {**EXPECTED_METRICS, "sharpe": 0.956, "cagr_pct": 10.45}
    target = _registry_target(tmp_path, _write_fixture_experiment(tmp_path, actual_metrics))

    result = replay_experiment(target, project_root=tmp_path)
    payload = result.as_labs_replay()

    assert result.passed is True
    assert payload["schema_version"] == "labs-replay/v1"
    assert payload["status"] == "passed"
    assert payload["passed"] is True
    assert payload["command"] == target["command"]
    assert payload["artifact_path"] == target["artifact_path"]
    assert payload["duration_seconds"] >= 0
    assert payload["metric_deltas"]["sharpe"] == pytest.approx(0.006)
    assert payload["metrics"]["metrics_compared"] == len(EXPECTED_METRICS)
    assert payload["metrics"]["max_abs_metric_delta"] == pytest.approx(0.05)
    assert validate_artifact(payload).valid is True


def test_replay_experiment_fails_when_metric_drift_exceeds_tolerance(tmp_path: Path) -> None:
    """Replay should fail without raising when a saved metric drifts beyond tolerance."""
    actual_metrics = {**EXPECTED_METRICS, "sharpe": 0.80}
    target = _registry_target(tmp_path, _write_fixture_experiment(tmp_path, actual_metrics))

    result = replay_experiment(target, project_root=tmp_path)
    payload = result.as_labs_replay()

    assert result.passed is False
    assert payload["status"] == "failed"
    assert payload["passed"] is False
    assert payload["metric_deltas"]["sharpe"] == pytest.approx(-0.15)
    assert payload["metrics"]["failed_metric_count"] == 1
    assert payload["metrics"]["max_abs_metric_delta"] == pytest.approx(0.15)
    assert validate_artifact(payload).valid is True


def test_replay_experiment_accepts_manifest_source_artifact_path(tmp_path: Path) -> None:
    """Manifest-style targets should load expected metrics from the source artifact."""
    artifact_path = tmp_path / "manifest_saved_experiment.json"
    artifact_path.write_text(json.dumps({"summary": EXPECTED_METRICS}))
    target = {
        "schema_version": "experiment-manifest/v1",
        "experiment_id": "fixture-manifest-replay",
        "source_artifact_path": str(artifact_path),
        "command": _write_fixture_experiment(tmp_path, EXPECTED_METRICS),
        "replay_safe": True,
        "provenance_status": "embedded",
    }

    result = replay_experiment(target, project_root=tmp_path)
    payload = result.as_labs_replay()

    assert result.passed is True
    assert payload["artifact_path"] == str(artifact_path)
    assert payload["provenance_status"] == "embedded"
    assert payload["metrics"]["metrics_compared"] == len(EXPECTED_METRICS)


def test_replay_rejects_market_data_fetch_without_explicit_flag(tmp_path: Path) -> None:
    """Replay-safe targets must still opt in before running commands marked as data-fetching."""
    marker = tmp_path / "command-ran.txt"
    target = _registry_target(
        tmp_path,
        _write_fixture_experiment(tmp_path, EXPECTED_METRICS, marker=marker),
        fetches_market_data=True,
    )

    with pytest.raises(ReplaySafetyError, match="market data fetch"):
        replay_experiment(target, project_root=tmp_path)

    assert marker.exists() is False

    result = replay_experiment(target, project_root=tmp_path, allow_market_data_fetch=True)

    assert result.passed is True
    assert marker.read_text() == "ran"


def test_replay_rejects_expensive_target_without_approval(tmp_path: Path) -> None:
    """Full or expensive experiments require explicit approval even when a command is present."""
    marker = tmp_path / "command-ran.txt"
    target = _registry_target(
        tmp_path,
        _write_fixture_experiment(tmp_path, EXPECTED_METRICS, marker=marker),
        replay_safe=False,
    )

    with pytest.raises(ReplaySafetyError, match="not marked replay_safe"):
        replay_experiment(target, project_root=tmp_path)

    assert marker.exists() is False

    result = replay_experiment(target, project_root=tmp_path, allow_expensive=True)

    assert result.passed is True
    assert marker.read_text() == "ran"
