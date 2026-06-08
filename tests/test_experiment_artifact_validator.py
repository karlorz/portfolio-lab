"""Tests for Labs experiment artifact schema validation."""

from __future__ import annotations

import json

from src.research.experiment_artifact_validator import (
    LABS_REGISTRY_SCHEMA_VERSION,
    LABS_REPLAY_SCHEMA_VERSION,
    LABS_SCORECARD_SCHEMA_VERSION,
    main,
    validate_artifact,
    validate_file,
    validate_paths,
)
from src.research.experiment_manifest import EXPERIMENT_MANIFEST_SCHEMA_VERSION


def _valid_provenance() -> dict:
    return {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "experiment_id": "gold-sweep",
        "generated_at": "2026-06-08T12:00:00+00:00",
        "source_artifact_path": "data/gold_allocation_sweep.json",
        "command": "python -m src.backtest.gold_allocation_sweep",
        "module": "src.backtest.gold_allocation_sweep",
        "git": {"commit": "abc123", "branch": "main", "dirty": False},
        "config_snapshot": {"min_gold_pct": 20},
        "environment": {"PORTFOLIO_LAB_ENABLE_ML": "0"},
        "input_file_hashes": {"data/prices.json": "a" * 64},
        "freeze_manifest": {
            "timestamp": "2026-06-08T12:00:00+00:00",
            "config": {},
            "file_hashes": {},
            "file_count": 0,
        },
    }


def _valid_registry() -> dict:
    return {
        "schema_version": LABS_REGISTRY_SCHEMA_VERSION,
        "generated_at": "2026-06-08T12:00:00+00:00",
        "experiments": [
            {
                "experiment_id": "gold-sweep",
                "artifact_path": "data/gold_allocation_sweep.json",
                "status": "validated",
                "provenance_status": "present",
                "metrics": {"sharpe": 0.95, "cagr_pct": 10.4, "max_drawdown_pct": -25.0},
                "baseline_deltas": {"sharpe": 0.04, "cagr_pct": 0.8},
            }
        ],
    }


def _valid_scorecard() -> dict:
    return {
        "schema_version": LABS_SCORECARD_SCHEMA_VERSION,
        "experiment_id": "gold-sweep",
        "generated_at": "2026-06-08T12:00:00+00:00",
        "status": "promote",
        "provenance_status": "present",
        "metrics": {"sharpe": 0.95, "cagr_pct": 10.4},
        "baseline_deltas": {"sharpe": 0.04, "max_drawdown_pct": 1.2},
    }


def _valid_replay() -> dict:
    return {
        "schema_version": LABS_REPLAY_SCHEMA_VERSION,
        "experiment_id": "gold-sweep",
        "generated_at": "2026-06-08T12:00:00+00:00",
        "artifact_path": "data/gold_allocation_sweep.json",
        "status": "passed",
        "provenance_status": "present",
        "metrics": {"rows_replayed": 109, "max_abs_metric_delta": 0.0},
        "baseline_deltas": {"sharpe": 0.0},
    }


def test_valid_labs_artifacts_pass_without_live_market_data() -> None:
    """Fixture registry, provenance, scorecard, and replay artifacts validate offline."""
    for artifact in (_valid_registry(), _valid_provenance(), _valid_scorecard(), _valid_replay()):
        result = validate_artifact(artifact)

        assert result.valid, result.error_messages()


def test_invalid_registry_reports_missing_required_field() -> None:
    artifact = _valid_registry()
    del artifact["experiments"][0]["artifact_path"]

    result = validate_artifact(artifact)

    assert not result.valid
    assert "$.experiments[0].artifact_path: missing required field" in result.error_messages()


def test_invalid_metrics_report_wrong_percentage_units() -> None:
    artifact = _valid_scorecard()
    artifact["metrics"]["cagr_pct"] = 240.0

    result = validate_artifact(artifact)

    assert not result.valid
    assert (
        "$.metrics.cagr_pct: expected percentage-point value between -100 and 100"
        in result.error_messages()
    )


def test_stale_schema_version_fails_with_actionable_message() -> None:
    artifact = _valid_registry()
    artifact["schema_version"] = "labs-registry/v0"

    result = validate_artifact(artifact)

    assert not result.valid
    assert (
        "$.schema_version: unsupported schema_version 'labs-registry/v0' "
        "(expected one of experiment-manifest/v1, labs-registry/v1, labs-replay/v1, labs-scorecard/v1)"
        in result.error_messages()
    )


def test_malformed_provenance_reports_nested_field() -> None:
    artifact = _valid_provenance()
    artifact["input_file_hashes"] = ["data/prices.json"]

    result = validate_artifact(artifact)

    assert not result.valid
    assert "$.input_file_hashes: expected object" in result.error_messages()


def test_validate_file_and_paths_read_existing_artifacts_without_running_experiments(tmp_path) -> None:
    registry_path = tmp_path / "labs_registry.json"
    registry_path.write_text(json.dumps(_valid_registry()))
    invalid_path = tmp_path / "invalid_scorecard.json"
    invalid = _valid_scorecard()
    invalid["status"] = "ship"
    invalid_path.write_text(json.dumps(invalid))

    file_result = validate_file(registry_path)
    path_results = validate_paths([registry_path, invalid_path])

    assert file_result.valid
    assert [result.path for result in path_results] == [registry_path, invalid_path]
    assert path_results[0].valid
    assert not path_results[1].valid
    assert "$.status: unsupported status 'ship'" in path_results[1].error_messages()


def test_cli_discover_defaults_allows_empty_artifact_set(tmp_path, monkeypatch, capsys) -> None:
    """Default discovery should be callable before Labs artifacts exist."""
    import src.research.experiment_artifact_validator as validator

    monkeypatch.setattr(validator, "DEFAULT_ARTIFACT_GLOBS", ((tmp_path, "*.json"),))

    exit_code = main(["--discover-defaults"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == {"results": []}
