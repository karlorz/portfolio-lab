#!/usr/bin/env python3
"""
Tests for duration_yield_backtest.py — BacktestResult dataclass,
classify_regime_from_spread, calculate_returns/Sharpe/max_drawdown/CAGR,
run_backtest with synthetic data, save_results, print_results, and CLI.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.backtest.duration_yield_backtest import (
    STATIC_ALLOCATION,
    DYNAMIC_ALLOCATIONS,
    REGIME_EFFECTIVE_DURATION,
    EXPENSE_RATIOS,
    TRANSACTION_COST,
    BacktestResult,
    classify_regime_from_spread,
    calculate_returns,
    calculate_sharpe,
    calculate_max_drawdown,
    calculate_cagr,
    run_backtest,
    save_results,
    print_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(n_days=504, start="2010-01-01", seed=42):
    """Create synthetic prices_df and regimes_df for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start=start, periods=n_days, freq="B")

    # Synthetic prices with random walks
    prices_df = pd.DataFrame({"date": dates})
    for col, start_price, drift, vol in [
        ("tlt", 95, 0.0002, 0.012),
        ("ief", 100, 0.00015, 0.008),
        ("shy", 80, 0.0001, 0.002),
        ("spy", 110, 0.0004, 0.012),
        ("gld", 60, 0.0003, 0.014),
    ]:
        returns = rng.normal(drift, vol, n_days)
        prices_df[col] = start_price * np.cumprod(1 + returns)

    # Synthetic regimes: cycle through inverted/flat/steep
    regimes = []
    for i in range(n_days):
        cycle = (i // 100) % 3
        if cycle == 0:
            regimes.append({"date": dates[i], "regime": "flat", "spread": 0.30})
        elif cycle == 1:
            regimes.append({"date": dates[i], "regime": "steep", "spread": 1.20})
        else:
            regimes.append({"date": dates[i], "regime": "inverted", "spread": -0.40})
    regimes_df = pd.DataFrame(regimes)

    return prices_df, regimes_df


def _make_simple_prices_df(n_days=252, seed=42):
    """Create minimal prices DataFrame for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="B")
    df = pd.DataFrame({"date": dates})
    for col, start in [("tlt", 100), ("ief", 100), ("shy", 80), ("spy", 100), ("gld", 60)]:
        df[col] = start * np.cumprod(1 + rng.normal(0.0003, 0.01, n_days))
    return df


def _make_regimes_df(dates, default_regime="flat"):
    """Create regime DataFrame matching date range."""
    regimes = [{"date": d, "regime": default_regime, "spread": 0.30} for d in dates]
    return pd.DataFrame(regimes)


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    def test_static_allocation(self):
        assert STATIC_ALLOCATION["tlt"] == 0.16
        assert STATIC_ALLOCATION["ief"] == 0.15
        assert STATIC_ALLOCATION["shy"] == 0.05

    def test_dynamic_allocations_keys(self):
        assert "inverted" in DYNAMIC_ALLOCATIONS
        assert "flat" in DYNAMIC_ALLOCATIONS
        assert "steep" in DYNAMIC_ALLOCATIONS

    def test_dynamic_inverted_shorter_duration(self):
        assert DYNAMIC_ALLOCATIONS["inverted"]["tlt"] < STATIC_ALLOCATION["tlt"]

    def test_dynamic_steep_longer_duration(self):
        assert DYNAMIC_ALLOCATIONS["steep"]["tlt"] > STATIC_ALLOCATION["tlt"]

    def test_regime_effective_duration(self):
        assert REGIME_EFFECTIVE_DURATION["inverted"] < REGIME_EFFECTIVE_DURATION["flat"]
        assert REGIME_EFFECTIVE_DURATION["steep"] > REGIME_EFFECTIVE_DURATION["flat"]

    def test_expense_ratios(self):
        assert "tlt" in EXPENSE_RATIOS
        assert "spy" in EXPENSE_RATIOS

    def test_transaction_cost(self):
        assert TRANSACTION_COST == 0.0010


# ---------------------------------------------------------------------------
# BacktestResult Tests
# ---------------------------------------------------------------------------

class TestBacktestResult:

    def test_fields(self):
        r = BacktestResult(
            static_cagr=0.10, static_volatility=0.12, static_sharpe=0.80, static_max_dd=-0.25,
            dynamic_cagr=0.12, dynamic_volatility=0.13, dynamic_sharpe=0.85, dynamic_max_dd=-0.23,
            sharpe_delta=0.05, cagr_delta=0.02, max_dd_delta=0.02,
            crisis_2008_static=-0.12, crisis_2008_dynamic=-0.10,
            crisis_2020_static=-0.07, crisis_2020_dynamic=-0.05,
            crisis_2022_static=-0.14, crisis_2022_dynamic=-0.13,
            regime_days={"flat": 100, "inverted": 80}, regime_transitions=3,
            rebalancing_costs=0.002,
            start_date="2010-01-01", end_date="2020-12-31", total_days=2520,
            timestamp="2026-05-14",
        )
        assert r.sharpe_delta == 0.05
        assert r.cagr_delta == 0.02

    def test_negative_delta(self):
        r = BacktestResult(
            static_cagr=0.10, static_volatility=0.12, static_sharpe=0.80, static_max_dd=-0.25,
            dynamic_cagr=0.08, dynamic_volatility=0.14, dynamic_sharpe=0.70, dynamic_max_dd=-0.30,
            sharpe_delta=-0.10, cagr_delta=-0.02, max_dd_delta=-0.05,
            crisis_2008_static=-0.12, crisis_2008_dynamic=-0.15,
            crisis_2020_static=-0.07, crisis_2020_dynamic=-0.09,
            crisis_2022_static=-0.14, crisis_2022_dynamic=-0.16,
            regime_days={"flat": 100}, regime_transitions=0,
            rebalancing_costs=0.001,
            start_date="2010-01-01", end_date="2020-12-31", total_days=2520,
            timestamp="2026-05-14",
        )
        assert r.sharpe_delta < 0
        assert r.cagr_delta < 0


# ---------------------------------------------------------------------------
# classify_regime_from_spread Tests
# ---------------------------------------------------------------------------

class TestClassifyRegime:

    def test_inverted(self):
        assert classify_regime_from_spread(-0.50) == "inverted"
        assert classify_regime_from_spread(-0.30) == "inverted"

    def test_flat(self):
        assert classify_regime_from_spread(0.0) == "flat"
        assert classify_regime_from_spread(0.50) == "flat"
        assert classify_regime_from_spread(-0.20) == "flat"
        assert classify_regime_from_spread(0.70) == "flat"

    def test_steep(self):
        assert classify_regime_from_spread(0.80) == "steep"
        assert classify_regime_from_spread(1.50) == "steep"

    def test_boundary_inverted(self):
        # -0.25 → inverted (strictly less)
        assert classify_regime_from_spread(-0.26) == "inverted"
        assert classify_regime_from_spread(-0.25) == "flat"

    def test_boundary_steep(self):
        # 0.75 → flat (not strictly greater)
        assert classify_regime_from_spread(0.75) == "flat"
        assert classify_regime_from_spread(0.76) == "steep"


# ---------------------------------------------------------------------------
# calculate_returns Tests
# ---------------------------------------------------------------------------

class TestCalculateReturns:

    def test_returns_length(self):
        s = pd.Series([100, 102, 101, 105])
        rets = calculate_returns(s)
        assert len(rets) == 4

    def test_first_return_zero(self):
        s = pd.Series([100, 102])
        rets = calculate_returns(s)
        assert rets.iloc[0] == 0.0  # fillna(0)

    def test_positive_return(self):
        s = pd.Series([100, 102])
        rets = calculate_returns(s)
        assert rets.iloc[1] == pytest.approx(0.02)

    def test_negative_return(self):
        s = pd.Series([100, 95])
        rets = calculate_returns(s)
        assert rets.iloc[1] == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# calculate_sharpe Tests
# ---------------------------------------------------------------------------

class TestCalculateSharpe:

    def test_positive_sharpe(self):
        rng = np.random.RandomState(42)
        rets = pd.Series(rng.normal(0.001, 0.01, 252))
        sharpe = calculate_sharpe(rets)
        assert sharpe > 0

    def test_short_data(self):
        rets = pd.Series([0.01, 0.02])
        assert calculate_sharpe(rets) == 0.0

    def test_zero_vol(self):
        rets = pd.Series([0.001] * 100)
        assert calculate_sharpe(rets) == 0.0

    def test_with_risk_free_rate(self):
        rets = pd.Series([0.0005] * 252)  # ~13.4% annualized
        sharpe = calculate_sharpe(rets, risk_free_rate=0.04)
        # Excess = 0.0005 - 0.04/252 ≈ 0.000341
        assert sharpe > 0


# ---------------------------------------------------------------------------
# calculate_max_drawdown Tests
# ---------------------------------------------------------------------------

class TestCalculateMaxDrawdown:

    def test_negative(self):
        rets = pd.Series([0.02, -0.05, 0.01, -0.03, 0.02] * 10)
        mdd = calculate_max_drawdown(rets)
        assert mdd < 0

    def test_zero_drawdown(self):
        rets = pd.Series([0.01] * 100)
        mdd = calculate_max_drawdown(rets)
        assert mdd == 0.0

    def test_known_max_dd(self):
        # Single big drop
        rets = pd.Series([0.0] * 10 + [-0.50] + [0.0] * 10)
        mdd = calculate_max_drawdown(rets)
        assert mdd == pytest.approx(-0.50)


# ---------------------------------------------------------------------------
# calculate_cagr Tests
# ---------------------------------------------------------------------------

class TestCalculateCAGR:

    def test_positive_cagr(self):
        rng = np.random.RandomState(42)
        rets = pd.Series(rng.normal(0.0005, 0.01, 252))
        cagr = calculate_cagr(rets)
        assert cagr > 0

    def test_empty_returns(self):
        assert calculate_cagr(pd.Series([], dtype=float)) == 0.0

    def test_short_period(self):
        rets = pd.Series([0.01] * 5)
        assert calculate_cagr(rets) == 0.0  # < 0.1 years

    def test_negative_cagr(self):
        rets = pd.Series([-0.001] * 252)
        cagr = calculate_cagr(rets)
        assert cagr < 0


# ---------------------------------------------------------------------------
# run_backtest Tests
# ---------------------------------------------------------------------------

class TestRunBacktest:

    def test_returns_result(self):
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        assert result is not None
        assert isinstance(result, BacktestResult)

    def test_has_all_metrics(self):
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        assert result.static_sharpe != 0
        assert result.dynamic_sharpe != 0
        assert result.total_days > 0

    def test_regime_days_sum(self):
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        total = sum(result.regime_days.values())
        assert total == result.total_days

    def test_single_regime(self):
        dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
        rng = np.random.RandomState(42)
        prices_df = pd.DataFrame({"date": dates})
        for col, start in [("tlt", 100), ("ief", 100), ("shy", 80), ("spy", 100), ("gld", 60)]:
            prices_df[col] = start * np.cumprod(1 + rng.normal(0.0003, 0.01, 252))
        regimes_df = _make_regimes_df(dates, "flat")
        result = run_backtest(prices_df, regimes_df)
        assert result.regime_days.get("flat", 0) == result.total_days

    def test_result_properties(self):
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        assert -1 <= result.static_sharpe <= 5
        assert -1 <= result.dynamic_sharpe <= 5

    def test_crisis_returns_populated(self):
        prices_df, regimes_df = _make_synthetic_data(504)
        result = run_backtest(prices_df, regimes_df)
        assert isinstance(result.crisis_2008_static, float)
        assert isinstance(result.crisis_2008_dynamic, float)

    def test_date_filtering(self):
        prices_df, regimes_df = _make_synthetic_data(252, start="2010-06-01")
        result = run_backtest(prices_df, regimes_df,
                              start_date="2010-07-01", end_date="2010-12-31")
        assert result.total_days > 0

    def test_sharpe_delta_computed(self):
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        assert isinstance(result.sharpe_delta, float)

    def test_handles_missing_columns(self):
        """backtest should handle DataFrames without all expected columns."""
        dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
        prices_df = pd.DataFrame({"date": dates, "tlt": 100.0, "spy": 200.0, "gld": 60.0})
        regimes_df = _make_regimes_df(dates, "flat")
        result = run_backtest(prices_df, regimes_df)
        assert result is not None


# ---------------------------------------------------------------------------
# save_results Tests
# ---------------------------------------------------------------------------

class TestSaveResults:

    def test_creates_file(self, tmp_path):
        r = BacktestResult(
            static_cagr=0.10, static_volatility=0.12, static_sharpe=0.80, static_max_dd=-0.25,
            dynamic_cagr=0.12, dynamic_volatility=0.13, dynamic_sharpe=0.85, dynamic_max_dd=-0.23,
            sharpe_delta=0.05, cagr_delta=0.02, max_dd_delta=0.02,
            crisis_2008_static=-0.12, crisis_2008_dynamic=-0.10,
            crisis_2020_static=-0.07, crisis_2020_dynamic=-0.05,
            crisis_2022_static=-0.14, crisis_2022_dynamic=-0.13,
            regime_days={"flat": 100}, regime_transitions=0,
            rebalancing_costs=0.001,
            start_date="2010-01-01", end_date="2020-12-31", total_days=2520,
            timestamp="2026-05-14",
        )
        path = tmp_path / "results.json"
        with patch("src.backtest.duration_yield_backtest.OUTPUT_PATH", path):
            save_results(r)
        assert path.exists()

    def test_valid_json(self, tmp_path):
        r = BacktestResult(
            static_cagr=0.10, static_volatility=0.12, static_sharpe=0.80, static_max_dd=-0.25,
            dynamic_cagr=0.12, dynamic_volatility=0.13, dynamic_sharpe=0.85, dynamic_max_dd=-0.23,
            sharpe_delta=0.05, cagr_delta=0.02, max_dd_delta=0.02,
            crisis_2008_static=-0.12, crisis_2008_dynamic=-0.10,
            crisis_2020_static=-0.07, crisis_2020_dynamic=-0.05,
            crisis_2022_static=-0.14, crisis_2022_dynamic=-0.13,
            regime_days={"flat": 100}, regime_transitions=0,
            rebalancing_costs=0.001,
            start_date="2010-01-01", end_date="2020-12-31", total_days=2520,
            timestamp="2026-05-14",
        )
        path = tmp_path / "results.json"
        with patch("src.backtest.duration_yield_backtest.OUTPUT_PATH", path):
            save_results(r)
        with open(path) as f:
            data = json.load(f)
        assert "static_cagr" in data
        assert "dynamic_cagr" in data


# ---------------------------------------------------------------------------
# print_results Tests
# ---------------------------------------------------------------------------

class TestPrintResults:

    def test_prints_output(self, capsys):
        r = BacktestResult(
            static_cagr=0.10, static_volatility=0.12, static_sharpe=0.80, static_max_dd=-0.25,
            dynamic_cagr=0.12, dynamic_volatility=0.13, dynamic_sharpe=0.85, dynamic_max_dd=-0.23,
            sharpe_delta=0.05, cagr_delta=0.02, max_dd_delta=0.02,
            crisis_2008_static=-0.12, crisis_2008_dynamic=-0.10,
            crisis_2020_static=-0.07, crisis_2020_dynamic=-0.05,
            crisis_2022_static=-0.14, crisis_2022_dynamic=-0.13,
            regime_days={"flat": 100, "inverted": 50, "steep": 50},
            regime_transitions=3,
            rebalancing_costs=0.002,
            start_date="2010-01-01", end_date="2020-12-31", total_days=2520,
            timestamp="2026-05-14",
        )
        print_results(r)
        out = capsys.readouterr().out
        assert "DURATION-YIELD" in out
        assert "PERFORMANCE COMPARISON" in out
        assert "CRISIS PERFORMANCE" in out

    def test_shows_sharpe_delta(self, capsys):
        r = BacktestResult(
            static_cagr=0.10, static_volatility=0.12, static_sharpe=0.80, static_max_dd=-0.25,
            dynamic_cagr=0.12, dynamic_volatility=0.13, dynamic_sharpe=0.85, dynamic_max_dd=-0.23,
            sharpe_delta=0.05, cagr_delta=0.02, max_dd_delta=0.02,
            crisis_2008_static=-0.12, crisis_2008_dynamic=-0.10,
            crisis_2020_static=-0.07, crisis_2020_dynamic=-0.05,
            crisis_2022_static=-0.14, crisis_2022_dynamic=-0.13,
            regime_days={"flat": 100}, regime_transitions=0,
            rebalancing_costs=0.001,
            start_date="2010-01-01", end_date="2020-12-31", total_days=2520,
            timestamp="2026-05-14",
        )
        print_results(r)
        out = capsys.readouterr().out
        assert "+0.050" in out or "0.050" in out


# ---------------------------------------------------------------------------
# CLI (main) Tests
# ---------------------------------------------------------------------------

class TestCLI:

    def test_main_with_synthetic_data(self):
        """Test main with mocked data loading."""
        prices_df, regimes_df = _make_synthetic_data(252)
        with patch("src.backtest.duration_yield_backtest.load_price_data", return_value=prices_df), \
             patch("src.backtest.duration_yield_backtest.load_yield_spread_history", return_value=regimes_df), \
             patch("src.backtest.duration_yield_backtest.save_results"), \
             patch("src.backtest.duration_yield_backtest.print_results"):
            from src.backtest.duration_yield_backtest import main
            result = main()
            assert result in (0, 1)

    def test_main_no_result(self):
        """Test main when run_backtest returns None."""
        with patch("src.backtest.duration_yield_backtest.load_price_data", return_value=pd.DataFrame()), \
             patch("src.backtest.duration_yield_backtest.load_yield_spread_history", return_value=pd.DataFrame()), \
             patch("src.backtest.duration_yield_backtest.run_backtest", return_value=None):
            from src.backtest.duration_yield_backtest import main
            result = main()
            assert result == 1
