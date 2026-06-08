"""Tests for optimizer output serialization into Labs-compatible artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from src.research.experiment_artifact_validator import validate_artifact


OPTIMIZER_RESULTS = {
    "max_sharpe": {
        "weights": {"SPY": 0.44, "GLD": 0.36, "TLT": 0.20},
        "sharpe": 0.96,
        "cagr": 10.2,
        "volatility": 10.7,
    },
    "min_volatility": {
        "error": "solver infeasible",
    },
    "efficient_risk": {
        "weights": {"SPY": 0.50, "GLD": 0.30, "TLT": 0.20},
        "sharpe": 0.91,
        "cagr": 9.8,
        "volatility": 11.0,
        "target_vol": 0.11,
    },
    "hrp": {
        "weights": {"SPY": 0.29, "GLD": 0.28, "TLT": 0.43},
        "sharpe": 0.93,
        "cagr": 9.9,
        "volatility": 10.4,
    },
    "champion": {
        "weights": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "sharpe": 0.79,
        "cagr": 10.6,
        "volatility": 11.1,
        "note": "Grid search champion (2005-2026)",
    },
}


def _rows_by_method(payload: dict) -> dict[str, dict]:
    return {row["method"]: row for row in payload["optimizer_results"]}


def test_optimizer_labs_output_serializes_success_failure_and_reference_rows() -> None:
    from src.research.optimizer_labs_contract import build_optimizer_labs_output

    payload = build_optimizer_labs_output(
        OPTIMIZER_RESULTS,
        symbols=["SPY", "GLD", "TLT"],
        target_vol=0.11,
        generated_at="2026-06-08T00:00:00+00:00",
        artifact_path="data/optimized_weights.json",
    )

    assert payload["schema_version"] == "optimizer-labs-output/v1"
    assert payload["generated_at"] == "2026-06-08T00:00:00+00:00"
    assert payload["symbols"] == ["SPY", "GLD", "TLT"]
    assert payload["target_vol"] == 0.11

    rows = _rows_by_method(payload)
    assert rows["max_sharpe"]["status"] == "succeeded"
    assert rows["max_sharpe"]["metrics"] == {
        "sharpe": 0.96,
        "cagr_pct": 10.2,
        "volatility_pct": 10.7,
    }
    assert rows["max_sharpe"]["weights"] == {"SPY": 0.44, "GLD": 0.36, "TLT": 0.20}
    assert rows["efficient_risk"]["target_vol"] == 0.11
    assert rows["min_volatility"]["status"] == "failed"
    assert rows["min_volatility"]["error"] == "solver infeasible"
    assert rows["min_volatility"]["metrics"] == {}
    assert rows["champion"]["status"] == "reference"
    assert rows["champion"]["note"] == "Grid search champion (2005-2026)"

    registry = payload["registry"]
    assert registry["schema_version"] == "labs-registry/v1"
    assert validate_artifact(registry).valid is True
    registry_rows = {row["experiment_id"]: row for row in registry["experiments"]}
    assert registry_rows["optimizer:max_sharpe"]["status"] == "candidate"
    assert registry_rows["optimizer:max_sharpe"]["metrics"]["sharpe"] == 0.96
    assert registry_rows["optimizer:min_volatility"]["status"] == "warning"
    assert registry_rows["optimizer:min_volatility"]["metrics"] == {}
    assert registry_rows["optimizer:champion_reference"]["status"] == "validated"


def test_save_optimizer_labs_output_writes_predictable_schema(tmp_path: Path) -> None:
    from src.research.optimizer_labs_contract import save_optimizer_labs_output

    output_path = tmp_path / "optimized_weights.json"

    saved_path = save_optimizer_labs_output(
        OPTIMIZER_RESULTS,
        output_path=output_path,
        symbols=["SPY", "GLD", "TLT"],
        target_vol=0.11,
        generated_at="2026-06-08T00:00:00+00:00",
    )

    assert saved_path == output_path
    payload = json.loads(output_path.read_text())
    assert payload["schema_version"] == "optimizer-labs-output/v1"
    assert _rows_by_method(payload)["min_volatility"]["status"] == "failed"
    assert validate_artifact(payload["registry"]).valid is True


def test_optimizer_price_loader_respects_symbol_filter_fixture() -> None:
    from scripts.optimize_portfolio import load_prices

    raw_prices = {
        "SPY": [{"d": "2026-01-01", "p": 100}, {"d": "2026-01-02", "p": 101}],
        "GLD": [{"d": "2026-01-01", "p": 200}, {"d": "2026-01-02", "p": 201}],
        "QQQ": [{"d": "2026-01-01", "p": 300}, {"d": "2026-01-02", "p": 301}],
    }

    prices = load_prices(raw_prices, symbols=["SPY", "GLD"])

    assert list(prices.columns) == ["GLD", "SPY"]
    assert len(prices) == 2
    assert "QQQ" not in prices.columns
