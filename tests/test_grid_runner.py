#!/usr/bin/env python3
"""
Tests for the shared grid_runner (Item 32, A5 phase 1).

A4 output-equality contract: both pilot migrations were verified against
pre-migration captures on the SAME fixture (tests/fixtures/prices_snapshot
.json — 504 daily bars of SPY/GLD/TLT/IEF/SHY captured 2026-08-12). The
values pinned below ARE the pre-migration outputs (float policy:
pytest.approx(rel=1e-9)); any drift in grid_runner or pilot semantics
fails here. The only excluded field is the wall-clock timestamp.

Also covers the shared helper surface directly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.backtest import grid_runner

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "prices_snapshot.json"


def _fixture():
    return json.loads(FIXTURE_PATH.read_text())


def _approx(value):
    return pytest.approx(value, rel=1e-9)


# ---------------------------------------------------------------------------
# A4: pilot outputs pinned to the pre-migration captures
# ---------------------------------------------------------------------------
def test_a4_gold_sweep_matches_pre_migration_capture(tmp_path):
    """Gold sweep output identical to the pre-migration capture (Item 32 A4)."""
    from src.backtest.gold_allocation_sweep import run_gold_sweep

    out = tmp_path / "gold.json"
    with patch("src.data.price_cache.get_prices", return_value=_fixture()):
        run_gold_sweep(output=str(out))

    data = json.loads(out.read_text())
    assert data["baseline_sharpe"] == _approx(1.701)
    assert data["baseline_cagr"] == _approx(22.33)
    assert data["baseline_max_dd"] == _approx(-36.75)
    assert data["n_days"] == 504
    assert len(data["rows"]) == 109
    # Best-sharpe row (pre-migration pinned)
    assert data["best_sharpe_row"]["gld_pct"] == _approx(36.0)
    assert data["best_sharpe_row"]["spy_pct"] == _approx(48.0)
    assert data["best_sharpe_row"]["sharpe"] == _approx(1.792)


def test_a4_duration_backtest_matches_pre_migration_capture():
    """Duration backtest core output identical to the pre-migration capture."""
    from src.backtest.duration_yield_backtest import (
        load_price_data,
        load_yield_spread_history,
        run_backtest,
    )

    with patch("src.data.price_cache.get_prices", return_value=_fixture()):
        prices_df = load_price_data()
        regimes_df = load_yield_spread_history()
        result = run_backtest(prices_df, regimes_df)

    assert result.total_return == _approx(56.634873670469354)
    assert result.cagr == _approx(29.0067308179766)
    assert result.volatility == _approx(13.091357920594815)
    assert result.sharpe_ratio == _approx(1.6679463106936907)
    assert result.max_drawdown == _approx(-10.744085087413145)
    assert result.total_rebalances == 0
    e = result.extras
    assert e["static_sharpe"] == _approx(1.6603076836997195)
    assert e["dynamic_sharpe"] == _approx(1.6679463106936907)
    assert e["sharpe_delta"] == _approx(0.00763862699397122)
    assert e["total_days"] == 444


# ---------------------------------------------------------------------------
# Shared helper surface
# ---------------------------------------------------------------------------
def test_load_prices_returns_raw_payload():
    """load_prices delegates to price_cache.get_prices (patch seam live)."""
    with patch("src.data.price_cache.get_prices", return_value={"SPY": [{"d": "2026-01-01", "p": 1.0}]}):
        assert grid_runner.load_prices() == {"SPY": [{"d": "2026-01-01", "p": 1.0}]}


def test_prices_to_frame_columns_and_sort():
    """Dict payload → DataFrame with lowercase symbol columns, sorted dates."""
    import pandas as pd

    data = {
        "SPY": [{"d": "2026-01-02", "p": 2.0}, {"d": "2026-01-01", "p": 1.0}],
        "TLT": [{"d": "2026-01-01", "p": 10.0}, {"d": "2026-01-02", "p": 11.0}],
    }
    df = grid_runner.prices_to_frame(data)
    assert list(df.columns) == ["date", "spy", "tlt"]
    assert df.iloc[0]["date"] == pd.Timestamp("2026-01-01")
    assert df.iloc[1]["spy"] == 2.0


def test_simulate_portfolio_matches_manual_math():
    """simulate_portfolio reproduces the gold pilot math on a tiny series."""
    prices = {
        "SPY": [100.0, 101.0, 102.0],
        "GLD": [100.0, 100.0, 101.0],
        "TLT": [100.0, 99.0, 99.0],
        "IEF": [100.0, 100.0, 100.5],
    }
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    weights = {"spy": 0.46, "gld": 0.38, "tlt": 0.16, "ief": 0.0}
    cagr, vol, sharpe, max_dd, yearly = grid_runner.simulate_portfolio(
        prices, dates, weights
    )
    # Manual daily returns for the weighted portfolio
    spy_rets = [101 / 100 - 1, 102 / 101 - 1]
    gld_rets = [100 / 100 - 1, 101 / 100 - 1]
    tlt_rets = [99 / 100 - 1, 99 / 99 - 1]
    daily = [
        weights["spy"] * spy_rets[0] + weights["gld"] * gld_rets[0] + weights["tlt"] * tlt_rets[0],
        weights["spy"] * spy_rets[1] + weights["gld"] * gld_rets[1] + weights["tlt"] * tlt_rets[1],
    ]
    raw_cagr = np.mean(daily) * 252 * 100
    raw_vol = np.std(daily) * math.sqrt(252) * 100
    assert cagr == _approx(round(raw_cagr, 2))
    assert vol == _approx(round(raw_vol, 2))
    # Pilot math: sharpe computed from UNROUNDED cagr/vol, then rounded
    assert sharpe == _approx(round(raw_cagr / raw_vol, 3))
    assert isinstance(yearly, dict)


def test_calculate_metrics_delegation_consistency():
    """grid_runner metrics == the pilot module-level names (re-export)."""
    import src.backtest.duration_yield_backtest as dur

    assert dur.calculate_sharpe is grid_runner.calculate_sharpe
    assert dur.calculate_max_drawdown is grid_runner.calculate_max_drawdown
    assert dur.calculate_cagr is grid_runner.calculate_cagr

    import pandas as pd

    returns = pd.Series([0.001, -0.002, 0.003, 0.001] * 10)
    assert dur.calculate_cagr(returns) == _approx(grid_runner.calculate_cagr(returns))
    assert dur.calculate_max_drawdown(returns) == _approx(
        grid_runner.calculate_max_drawdown(returns)
    )


def test_run_grid_and_save_writes_artifact(tmp_path):
    """run_grid_and_save persists via save_results_json with manifest."""
    out = tmp_path / "grid.json"
    grid_runner.run_grid_and_save(
        {"a": 1},
        out,
        experiment_manifest={"experiment_id": "test", "input_paths": []},
    )
    data = json.loads(out.read_text())
    assert data["a"] == 1
    assert data["_provenance"]["experiment_id"] == "test"
