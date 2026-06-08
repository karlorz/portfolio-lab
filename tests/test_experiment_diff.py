"""Tests for Labs experiment result diffing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixtures.labs import load_labs_fixture


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _missing_by_metric(diff: dict) -> dict[str, dict]:
    return {entry["metric"]: entry for entry in diff["missing_metrics"]}


def test_diff_reports_metric_deltas_missing_fields_config_and_provenance() -> None:
    from src.research.experiment_diff import diff_experiment_artifacts

    left = {
        "experiment_id": "champion",
        "metrics": {
            "sharpe": 0.95,
            "cagr_pct": 10.4,
            "max_drawdown_pct": -27.6,
            "dsr": 0.979,
        },
        "config_snapshot": {
            "target_vol": 0.09,
            "max_leverage": 1.5,
        },
        "provenance_status": "present",
    }
    right = {
        "experiment_id": "challenger",
        "metrics": {
            "sharpe": 0.99,
            "cagr_pct": 10.1,
            "max_drawdown_pct": -20.1,
        },
        "config_snapshot": {
            "target_vol": 0.11,
            "max_leverage": 1.5,
        },
        "provenance_status": "stale",
    }

    diff = diff_experiment_artifacts(left, right)

    assert diff["schema_version"] == "experiment-diff/v1"
    assert diff["left"]["experiment_id"] == "champion"
    assert diff["right"]["experiment_id"] == "challenger"
    assert diff["metric_deltas"]["sharpe"]["delta"] == pytest.approx(0.04)
    assert diff["metric_deltas"]["cagr_pct"]["delta"] == pytest.approx(-0.3)
    assert diff["metric_deltas"]["max_drawdown_pct"]["delta"] == pytest.approx(7.5)
    assert _missing_by_metric(diff)["dsr"]["missing_from"] == ["right"]
    assert diff["config_diffs"]["target_vol"] == {"left": 0.09, "right": 0.11}
    assert "max_leverage" not in diff["config_diffs"]
    assert diff["provenance"]["changed"] is True
    assert diff["provenance"]["left"] == "present"
    assert diff["provenance"]["right"] == "stale"


def test_diff_compares_registry_rows_without_rerunning_artifacts() -> None:
    from src.research.experiment_diff import diff_experiment_artifacts

    left = load_labs_fixture("valid_registry")["experiments"][0]
    right = {
        **left,
        "experiment_id": "gold-sweep-challenger",
        "artifact_path": "data/backtest_results/gold_sweep_challenger.json",
        "metrics": {
            **left["metrics"],
            "sharpe": 0.99,
            "wfe": 1.37,
        },
        "provenance_status": "sidecar",
    }

    diff = diff_experiment_artifacts(left, right)

    assert diff["left"]["artifact_path"] == "data/gold_allocation_sweep.json"
    assert diff["right"]["artifact_path"] == "data/backtest_results/gold_sweep_challenger.json"
    assert diff["left"]["artifact_type"] == "registry_row"
    assert diff["metric_deltas"]["sharpe"]["delta"] == pytest.approx(0.04)
    assert _missing_by_metric(diff)["wfe"]["missing_from"] == ["left"]
    assert diff["provenance"]["changed"] is True


def test_diff_cli_outputs_json_and_human_readable_text(tmp_path: Path, capsys) -> None:
    from src.research.experiment_diff import main

    left = _write_json(
        tmp_path / "left.json",
        {
            "experiment_id": "baseline",
            "metrics": {"sharpe": 0.8, "dsr": 0.9},
            "config": {"allocation": "46/38/16"},
            "provenance_status": "present",
        },
    )
    right = _write_json(
        tmp_path / "right.json",
        {
            "experiment_id": "candidate",
            "metrics": {"sharpe": 0.85},
            "config": {"allocation": "44/36/20"},
            "provenance_status": "present",
        },
    )

    exit_code = main([str(left), str(right), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["metric_deltas"]["sharpe"]["delta"] == pytest.approx(0.05)
    assert _missing_by_metric(payload)["dsr"]["missing_from"] == ["right"]

    exit_code = main([str(left), str(right), "--format", "text"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Experiment Diff: baseline -> candidate" in captured.out
    assert "sharpe" in captured.out
    assert "Missing metrics" in captured.out
    assert "allocation" in captured.out
