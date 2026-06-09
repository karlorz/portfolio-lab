"""Tests for offline Labs experiment registry generation."""

from __future__ import annotations

import json
from pathlib import Path

from src.research.experiment_artifact_validator import validate_artifact
from src.research.experiment_manifest import EXPERIMENT_MANIFEST_SCHEMA_VERSION
from src.research.optimizer_labs_contract import build_optimizer_labs_output


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _rows_by_id(registry: dict) -> dict[str, dict]:
    return {row["experiment_id"]: row for row in registry["experiments"]}


def test_registry_includes_valid_nested_optimizer_registry(tmp_path: Path) -> None:
    """Optimizer Labs output should be scanned without rerunning optimization."""
    from src.research.experiment_registry import build_labs_registry

    data_dir = tmp_path / "data"
    optimizer_path = _write_json(
        data_dir / "optimized_weights.json",
        build_optimizer_labs_output(
            {
                "max_sharpe": {"weights": {"SPY": 0.44}, "sharpe": 0.96, "cagr": 10.2},
                "champion": {"weights": {"SPY": 0.46}, "sharpe": 0.79, "cagr": 10.6},
            },
            symbols=["SPY"],
            target_vol=0.11,
            generated_at="2026-06-08T00:00:00+00:00",
            artifact_path="data/optimized_weights.json",
        ),
    )

    registry = build_labs_registry(
        data_dirs=[data_dir],
        project_root=tmp_path,
        generated_at="2026-06-08T01:00:00+00:00",
    )

    assert registry["schema_version"] == "labs-registry/v1"
    assert registry["generated_at"] == "2026-06-08T01:00:00+00:00"
    assert validate_artifact(registry).valid is True
    rows = _rows_by_id(registry)
    assert rows["optimizer:max_sharpe"]["artifact_path"] == "data/optimized_weights.json#max_sharpe"
    assert rows["optimizer:max_sharpe"]["status"] == "candidate"
    assert rows["optimizer:max_sharpe"]["metrics"]["sharpe"] == 0.96
    assert registry["sources"] == [str(optimizer_path.relative_to(tmp_path))]


def test_registry_derives_rows_from_plain_result_artifacts_with_sidecar_provenance(tmp_path: Path) -> None:
    """Plain historical result JSON should become a registry row from fixture data only."""
    from src.research.experiment_registry import build_labs_registry

    result_path = _write_json(
        tmp_path / "data" / "backtest_results" / "walk_forward_report.json",
        {
            "sharpe_ratio": 1.12,
            "cagr": 9.4,
            "max_drawdown": -18.5,
            "baseline_sharpe": 0.95,
        },
    )
    _write_json(
        result_path.with_suffix(result_path.suffix + ".manifest.json"),
        {
            "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
            "experiment_id": "walk-forward-validation",
            "generated_at": "2026-06-08T00:00:00+00:00",
            "source_artifact_path": "data/backtest_results/walk_forward_report.json",
            "git": {},
            "config_snapshot": {},
            "environment": {},
            "input_file_hashes": {},
            "freeze_manifest": {},
        },
    )

    registry = build_labs_registry(
        data_dirs=[tmp_path / "data"],
        project_root=tmp_path,
        generated_at="2026-06-08T01:00:00+00:00",
    )

    row = _rows_by_id(registry)["walk-forward-validation"]
    assert row == {
        "experiment_id": "walk-forward-validation",
        "artifact_path": "data/backtest_results/walk_forward_report.json",
        "status": "candidate",
        "provenance_status": "sidecar",
        "metrics": {
            "sharpe": 1.12,
            "cagr_pct": 9.4,
            "max_drawdown_pct": -18.5,
        },
        "baseline_deltas": {"sharpe": 0.17},
    }
    assert validate_artifact(registry).valid is True


def test_registry_emits_warning_row_for_malformed_source_artifact(tmp_path: Path) -> None:
    """Malformed sources should not fail registry generation or dashboard indexing."""
    from src.research.experiment_registry import build_labs_registry

    bad_path = tmp_path / "data" / "backtest_results" / "bad_result.json"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{not-json")

    registry = build_labs_registry(
        data_dirs=[tmp_path / "data"],
        project_root=tmp_path,
        generated_at="2026-06-08T01:00:00+00:00",
    )

    row = _rows_by_id(registry)["artifact:bad_result"]
    assert row["status"] == "warning"
    assert row["provenance_status"] == "malformed"
    assert row["metrics"] == {}
    assert row["baseline_deltas"] == {}
    assert registry["warnings"][0]["artifact_path"] == "data/backtest_results/bad_result.json"
    assert "invalid JSON" in registry["warnings"][0]["error"]
    assert validate_artifact(registry).valid is True


def test_save_labs_registry_writes_valid_public_artifact(tmp_path: Path) -> None:
    from src.research.experiment_registry import save_labs_registry

    _write_json(
        tmp_path / "data" / "combined_regime_alloc_vol_target_results.json",
        {
            "_provenance": {
                "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
                "experiment_id": "combined-regime-alloc-vol-target",
                "generated_at": "2026-06-08T00:00:00+00:00",
                "source_artifact_path": "data/combined_regime_alloc_vol_target_results.json",
                "git": {},
                "config_snapshot": {},
                "environment": {},
                "input_file_hashes": {},
                "freeze_manifest": {},
            },
            "static_sharpe": 0.95,
            "combined_sharpe_delta": 0.05,
            "static_max_dd": -27.6,
        },
    )

    output_path = save_labs_registry(
        data_dirs=[tmp_path / "data"],
        public_dir=tmp_path / "public" / "data",
        project_root=tmp_path,
        generated_at="2026-06-08T01:00:00+00:00",
    )

    assert output_path == tmp_path / "public" / "data" / "labs_registry.json"
    registry = json.loads(output_path.read_text())
    assert _rows_by_id(registry)["combined-regime-alloc-vol-target"]["provenance_status"] == "embedded"
    assert validate_artifact(registry).valid is True
