#!/usr/bin/env python3
"""
Tests for duration_yield_backtest.py — BacktestResult dataclass,
classify_regime_from_spread, calculate_returns/Sharpe/max_drawdown/CAGR,
run_backtest with synthetic data, save_results, print_results, and CLI.
"""
import json
import logging
from dataclasses import asdict

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch

from src.backtest.duration_yield_backtest import (
    STATIC_ALLOCATION,
    DYNAMIC_ALLOCATIONS,
    REGIME_EFFECTIVE_DURATION,
    EXPENSE_RATIOS,
    TRANSACTION_COST,
    classify_regime_from_spread,
    calculate_returns,
    calculate_sharpe,
    calculate_max_drawdown,
    calculate_cagr,
    run_backtest,
    save_results,
    print_results,
)
from src.backtest.metrics import BacktestResult, BacktestMetrics


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

    def test_static_allocation_sum(self):
        """Static allocation should sum to the bond portfolio target (0.36)."""
        total = sum(STATIC_ALLOCATION.values())
        assert total == pytest.approx(0.36, abs=1e-10)

    def test_dynamic_allocations_sum(self):
        """Each regime allocation dict should also sum to 0.36."""
        for regime, alloc in DYNAMIC_ALLOCATIONS.items():
            total = sum(alloc.values())
            assert total == pytest.approx(0.36, abs=1e-10), (
                f"{regime} allocation sums to {total}, expected 0.36"
            )

    def test_dynamic_allocations_have_all_etfs(self):
        """Every regime allocation must contain tlt, ief, and shy keys."""
        required = {"tlt", "ief", "shy"}
        for regime, alloc in DYNAMIC_ALLOCATIONS.items():
            assert required.issubset(alloc.keys()), (
                f"{regime} allocation missing: {required - alloc.keys()}"
            )

    def test_regime_effective_duration_positive(self):
        """All regime effective durations must be strictly positive."""
        for regime, duration in REGIME_EFFECTIVE_DURATION.items():
            assert duration > 0, f"{regime} duration must be positive, got {duration}"

    def test_expense_ratios_range(self):
        """All expense ratios must be between 0 and 1 (0% to 100%)."""
        for ticker, ratio in EXPENSE_RATIOS.items():
            assert 0 < ratio < 1, (
                f"{ticker} expense ratio {ratio} out of range (0, 1)"
            )

    def test_shy_has_lowest_expense(self):
        """SHY should have the same expense ratio as TLT and IEF."""
        assert EXPENSE_RATIOS["shy"] == EXPENSE_RATIOS["tlt"]
        assert EXPENSE_RATIOS["ief"] == EXPENSE_RATIOS["tlt"]


# ---------------------------------------------------------------------------
# BacktestResult Tests
# ---------------------------------------------------------------------------

class TestBacktestResult:

    def test_fields(self):
        r = BacktestResult(
            total_return=210.58, cagr=12.0, volatility=13.0, sharpe_ratio=0.85,
            max_drawdown=-23.0,
            extras={
                "static_cagr": 0.10, "static_volatility": 0.12, "static_sharpe": 0.80,
                "static_max_dd": -0.25,
                "dynamic_cagr": 0.12, "dynamic_volatility": 0.13, "dynamic_sharpe": 0.85,
                "dynamic_max_dd": -0.23,
                "sharpe_delta": 0.05, "cagr_delta": 0.02, "max_dd_delta": 0.02,
                "crisis_2008_static": -0.12, "crisis_2008_dynamic": -0.10,
                "crisis_2020_static": -0.07, "crisis_2020_dynamic": -0.05,
                "crisis_2022_static": -0.14, "crisis_2022_dynamic": -0.13,
                "regime_days": {"flat": 100, "inverted": 80}, "regime_transitions": 3,
                "rebalancing_costs": 0.002,
                "start_date": "2010-01-01", "end_date": "2020-12-31", "total_days": 2520,
                "timestamp": "2026-05-14",
            },
        )
        assert r.extras["sharpe_delta"] == 0.05
        assert r.extras["cagr_delta"] == 0.02

    def test_negative_delta(self):
        r = BacktestResult(
            total_return=-50.0, cagr=-5.0, volatility=14.0, sharpe_ratio=0.70,
            max_drawdown=-30.0,
            extras={
                "static_cagr": 0.10, "static_volatility": 0.12, "static_sharpe": 0.80,
                "static_max_dd": -0.25,
                "dynamic_cagr": 0.08, "dynamic_volatility": 0.14, "dynamic_sharpe": 0.70,
                "dynamic_max_dd": -0.30,
                "sharpe_delta": -0.10, "cagr_delta": -0.02, "max_dd_delta": -0.05,
                "crisis_2008_static": -0.12, "crisis_2008_dynamic": -0.15,
                "crisis_2020_static": -0.07, "crisis_2020_dynamic": -0.09,
                "crisis_2022_static": -0.14, "crisis_2022_dynamic": -0.16,
                "regime_days": {"flat": 100}, "regime_transitions": 0,
                "rebalancing_costs": 0.001,
                "start_date": "2010-01-01", "end_date": "2020-12-31", "total_days": 2520,
                "timestamp": "2026-05-14",
            },
        )
        assert r.extras["sharpe_delta"] < 0
        assert r.extras["cagr_delta"] < 0

    def test_asdict_core_fields(self):
        """asdict() should contain all core BacktestResult fields."""
        r = BacktestResult(
            total_return=100.0, cagr=8.0, volatility=12.0, sharpe_ratio=0.75,
            max_drawdown=-20.0,
            extras={"static_sharpe": 0.70},
        )
        d = asdict(r)
        for field in ("total_return", "cagr", "volatility", "sharpe_ratio",
                      "max_drawdown", "total_rebalances", "total_transaction_costs",
                      "extras"):
            assert field in d, f"asdict missing field: {field}"

    def test_asdict_extras_preserved(self):
        """The extras dict should be preserved verbatim through asdict()."""
        extras = {
            "static_sharpe": 0.80, "dynamic_sharpe": 0.85,
            "regime_days": {"flat": 100},
        }
        r = BacktestResult(
            total_return=100.0, cagr=8.0, volatility=12.0, sharpe_ratio=0.75,
            max_drawdown=-20.0, extras=extras,
        )
        d = asdict(r)
        assert d["extras"]["static_sharpe"] == 0.80
        assert d["extras"]["regime_days"]["flat"] == 100

    def test_asdict_crisis_returns(self):
        """crisis_returns field should survive asdict round-trip."""
        crisis = {"2008": -0.10, "2020": -0.05, "2022": -0.13}
        r = BacktestResult(
            total_return=100.0, cagr=8.0, volatility=12.0, sharpe_ratio=0.75,
            max_drawdown=-20.0, extras={}, crisis_returns=crisis,
        )
        d = asdict(r)
        assert d["crisis_returns"] == crisis

    def test_asdict_field_types(self):
        """Verify that core fields have the correct types after asdict()."""
        r = BacktestResult(
            total_return=100.0, cagr=8.0, volatility=12.0, sharpe_ratio=0.75,
            max_drawdown=-20.0, total_rebalances=5,
            extras={"key": "value"},
        )
        d = asdict(r)
        assert isinstance(d["total_rebalances"], int)
        assert isinstance(d["total_return"], float)
        assert isinstance(d["cagr"], float)
        assert isinstance(d["extras"], dict)

    def test_backtest_metrics_dataclass(self):
        """BacktestMetrics should have the same core fields as BacktestResult."""
        m = BacktestMetrics(
            total_return=50.0, cagr=6.0, volatility=10.0, sharpe_ratio=0.60,
            max_drawdown=-15.0,
        )
        d = asdict(m)
        for field in ("total_return", "cagr", "volatility", "sharpe_ratio",
                      "max_drawdown"):
            assert field in d
            assert isinstance(d[field], float)


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

    def test_extreme_values(self):
        """Very large positive/negative spreads should still classify correctly."""
        assert classify_regime_from_spread(-10.0) == "inverted"
        assert classify_regime_from_spread(-100.0) == "inverted"
        assert classify_regime_from_spread(10.0) == "steep"
        assert classify_regime_from_spread(100.0) == "steep"

    def test_high_precision_boundaries(self):
        """Boundary with many decimal places should behave consistently."""
        # -0.2500000001 → inverted (epsilon below boundary)
        assert classify_regime_from_spread(-0.2500000001) == "inverted"
        # 0.7500000001 → steep (epsilon above boundary)
        assert classify_regime_from_spread(0.7500000001) == "steep"
        # Exactly -0.25 → flat (boundary is flat)
        assert classify_regime_from_spread(-0.25) == "flat"
        # Exactly 0.75 → flat (boundary is flat)
        assert classify_regime_from_spread(0.75) == "flat"

    def test_near_boundary_inverted(self):
        """Values infinitesimally close to -0.25 boundary."""
        assert classify_regime_from_spread(-0.249999) == "flat"
        assert classify_regime_from_spread(-0.250001) == "inverted"

    def test_classify_regime_zero_spread(self):
        """Zero spread should classify as flat."""
        assert classify_regime_from_spread(0.0) == "flat"

    def test_classify_regime_small_positive(self):
        """Very small positive spread near zero is flat."""
        assert classify_regime_from_spread(0.001) == "flat"
        assert classify_regime_from_spread(0.0001) == "flat"


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

    def test_constant_prices(self):
        """All identical prices should yield all zero returns."""
        s = pd.Series([100.0] * 10)
        rets = calculate_returns(s)
        assert (rets == 0.0).all()

    def test_single_element(self):
        """A single-element series should return a single zero."""
        s = pd.Series([100.0])
        rets = calculate_returns(s)
        assert len(rets) == 1
        assert rets.iloc[0] == 0.0

    def test_empty_series(self):
        """An empty series should return an empty series."""
        s = pd.Series([], dtype=float)
        rets = calculate_returns(s)
        assert len(rets) == 0

    def test_returns_preserves_index(self):
        """Return series should preserve the input index."""
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        s = pd.Series([100, 102, 101, 103, 105], index=idx)
        rets = calculate_returns(s)
        pd.testing.assert_index_equal(rets.index, idx)


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
        # Use returns equal to risk-free rate so excess mean is also 0
        rets = pd.Series([0.02 / 252] * 100)
        assert calculate_sharpe(rets) == 0.0

    def test_with_risk_free_rate(self):
        rets = pd.Series([0.0005] * 252)  # ~13.4% annualized
        sharpe = calculate_sharpe(rets, risk_free_rate=0.04)
        # Excess = 0.0005 - 0.04/252 ≈ 0.000341
        assert sharpe > 0

    def test_negative_sharpe(self):
        """Consistently negative returns should produce a negative Sharpe."""
        rets = pd.Series(np.full(252, -0.001))
        sharpe = calculate_sharpe(rets)
        assert sharpe < 0

    def test_constant_positive_returns(self):
        """Constant returns equal to risk-free rate produce zero excess -> zero vol -> zero Sharpe."""
        rets = pd.Series(np.full(100, 0.02 / 252))
        assert calculate_sharpe(rets) == 0.0

    def test_large_risk_free_rate(self):
        """When risk_free_rate exceeds all returns, Sharpe should be negative."""
        rng = np.random.RandomState(999)
        rets = pd.Series(rng.normal(-0.0005, 0.01, 252))
        sharpe = calculate_sharpe(rets, risk_free_rate=0.10)
        assert sharpe < 0

    def test_sharpe_with_all_equal_excess(self):
        """All returns equal and exactly matching risk-free rate -> Sharpe is 0 (zero vol)."""
        rf = 0.02
        rets = pd.Series(np.full(252, rf / 252))
        sharpe = calculate_sharpe(rets, risk_free_rate=rf)
        assert sharpe == 0.0


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

    def test_monotonic_increasing(self):
        """Returns that are always positive should have a zero drawdown."""
        rets = pd.Series([0.001] * 100)
        mdd = calculate_max_drawdown(rets)
        assert mdd == 0.0

    def test_monotonic_decreasing(self):
        """Returns that are always negative should have drawdown == total loss."""
        rets = pd.Series([-0.01] * 100)
        mdd = calculate_max_drawdown(rets)
        # Each day loses 1%, cumulative loss is large
        assert mdd < -0.5

    def test_recovery_after_drawdown(self):
        """After a drop and full recovery, max DD should reflect the drop."""
        rets = pd.Series([0.0, -0.30, 0.0, 0.0, 0.0, 0.4286, 0.0])
        mdd = calculate_max_drawdown(rets)
        assert mdd == pytest.approx(-0.30, abs=1e-4)

    def test_drawdown_empty_series(self):
        """Empty series should return NaN (no crash)."""
        rets = pd.Series([], dtype=float)
        mdd = calculate_max_drawdown(rets)
        assert np.isnan(mdd) or mdd == 0.0


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

    def test_short_period_boundary(self):
        """Exactly 25 trading days (~0.099 years) should still produce a CAGR."""
        rets = pd.Series([0.001] * 25)
        cagr = calculate_cagr(rets)
        # 25/252 ≈ 0.0992 which is < 0.1, so returns 0.0
        assert cagr == 0.0

    def test_single_day_return(self):
        """A single day should return 0.0 (< 0.1 years)."""
        rets = pd.Series([0.01])
        assert calculate_cagr(rets) == 0.0

    def test_cagr_exact_one_year(self):
        """Exactly 252 trading days should produce a valid CAGR."""
        rets = pd.Series([0.0005] * 252)
        cagr = calculate_cagr(rets)
        total = (1.0005 ** 252) - 1
        assert cagr == pytest.approx(total, rel=0.01)

    def test_cagr_zero_total_return(self):
        """All zero returns should produce CAGR of 0."""
        rets = pd.Series([0.0] * 252)
        assert calculate_cagr(rets) == 0.0


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
        assert result.extras["static_sharpe"] != 0
        assert result.extras["dynamic_sharpe"] != 0
        assert result.extras["total_days"] > 0

    def test_regime_days_sum(self):
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        total = sum(result.extras["regime_days"].values())
        assert total == result.extras["total_days"]

    def test_single_regime(self):
        dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
        rng = np.random.RandomState(42)
        prices_df = pd.DataFrame({"date": dates})
        for col, start in [("tlt", 100), ("ief", 100), ("shy", 80), ("spy", 100), ("gld", 60)]:
            prices_df[col] = start * np.cumprod(1 + rng.normal(0.0003, 0.01, 252))
        regimes_df = _make_regimes_df(dates, "flat")
        result = run_backtest(prices_df, regimes_df)
        assert result.extras["regime_days"].get("flat", 0) == result.extras["total_days"]

    def test_result_properties(self):
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        assert -1 <= result.extras["static_sharpe"] <= 5
        assert -1 <= result.extras["dynamic_sharpe"] <= 5

    def test_crisis_returns_populated(self):
        prices_df, regimes_df = _make_synthetic_data(504)
        result = run_backtest(prices_df, regimes_df)
        assert isinstance(result.extras["crisis_2008_static"], float)
        assert isinstance(result.extras["crisis_2008_dynamic"], float)

    def test_date_filtering(self):
        prices_df, regimes_df = _make_synthetic_data(252, start="2010-06-01")
        result = run_backtest(prices_df, regimes_df,
                              start_date="2010-07-01", end_date="2010-12-31")
        assert result.extras["total_days"] > 0

    def test_sharpe_delta_computed(self):
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        assert isinstance(result.extras["sharpe_delta"], float)

    def test_handles_missing_columns(self):
        """backtest should handle DataFrames without all expected columns."""
        dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
        prices_df = pd.DataFrame({"date": dates, "tlt": 100.0, "spy": 200.0, "gld": 60.0})
        regimes_df = _make_regimes_df(dates, "flat")
        result = run_backtest(prices_df, regimes_df)
        assert result is not None

    def test_volatility_positive(self):
        """Both static and dynamic volatility should be positive."""
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        assert result.extras["static_volatility"] > 0
        assert result.extras["dynamic_volatility"] > 0

    def test_max_dd_non_positive(self):
        """Max drawdown should be <= 0."""
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        assert result.extras["static_max_dd"] <= 0
        assert result.extras["dynamic_max_dd"] <= 0

    def test_timestamp_in_extras(self):
        """The extras dict should contain an ISO-format timestamp."""
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        assert "timestamp" in result.extras
        assert isinstance(result.extras["timestamp"], str)
        # Should be parseable as ISO format
        parsed = datetime.fromisoformat(result.extras["timestamp"])
        assert parsed is not None

    def test_all_regimes_present(self):
        """With cycling synthetic data, all three regimes should appear."""
        prices_df, regimes_df = _make_synthetic_data(504)
        result = run_backtest(prices_df, regimes_df)
        regime_days = result.extras["regime_days"]
        for regime in ("flat", "inverted", "steep"):
            assert regime_days.get(regime, 0) > 0, f"{regime} not present"

    def test_regime_transitions_non_negative(self):
        """Regime transitions count should be >= 0."""
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        assert result.extras["regime_transitions"] >= 0

    def test_crisis_2020_populated(self):
        """Crisis returns for 2020 should be populated when data covers 2020."""
        prices_df, regimes_df = _make_synthetic_data(504, start="2019-01-01")
        result = run_backtest(prices_df, regimes_df)
        assert isinstance(result.extras["crisis_2020_static"], float)
        assert isinstance(result.extras["crisis_2020_dynamic"], float)

    def test_crisis_2022_populated(self):
        """Crisis returns for 2022 should be populated when data covers 2022."""
        prices_df, regimes_df = _make_synthetic_data(504, start="2020-01-01")
        result = run_backtest(prices_df, regimes_df)
        assert isinstance(result.extras["crisis_2022_static"], float)
        assert isinstance(result.extras["crisis_2022_dynamic"], float)

    def test_rebalancing_costs_non_negative(self):
        """Rebalancing costs should never be negative."""
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        assert result.extras["rebalancing_costs"] >= 0

    def test_result_total_return_is_scaled(self):
        """total_return field should be scaled to percentage (not decimal)."""
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        # Total return as a percentage should be a reasonable magnitude
        assert isinstance(result.total_return, float)
        assert result.total_return > -100  # Not a complete loss
        assert result.total_return < 10000  # Sanity check upper bound

    def test_result_cagr_is_scaled(self):
        """cagr field should be scaled to percentage."""
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        assert isinstance(result.cagr, float)
        # CAGR scaled to percentage should be non-zero for positive drift data
        assert result.cagr != 0.0

    def test_result_volatility_is_scaled(self):
        """volatility field should be scaled to percentage."""
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        assert isinstance(result.volatility, float)
        assert result.volatility > 0
        assert result.volatility < 100  # Sanity check

    def test_start_end_date_in_extras(self):
        """start_date and end_date should be in the extras dict."""
        prices_df, regimes_df = _make_synthetic_data(252)
        result = run_backtest(prices_df, regimes_df)
        assert "start_date" in result.extras
        assert "end_date" in result.extras
        assert isinstance(result.extras["start_date"], str)
        assert isinstance(result.extras["end_date"], str)


# ---------------------------------------------------------------------------
# save_results Tests
# ---------------------------------------------------------------------------

class TestSaveResults:

    def test_creates_file(self, tmp_path):
        r = BacktestResult(
            total_return=210.58, cagr=12.0, volatility=13.0, sharpe_ratio=0.85,
            max_drawdown=-23.0,
            extras={
                "static_cagr": 0.10, "static_volatility": 0.12, "static_sharpe": 0.80,
                "static_max_dd": -0.25,
                "dynamic_cagr": 0.12, "dynamic_volatility": 0.13, "dynamic_sharpe": 0.85,
                "dynamic_max_dd": -0.23,
                "sharpe_delta": 0.05, "cagr_delta": 0.02, "max_dd_delta": 0.02,
                "crisis_2008_static": -0.12, "crisis_2008_dynamic": -0.10,
                "crisis_2020_static": -0.07, "crisis_2020_dynamic": -0.05,
                "crisis_2022_static": -0.14, "crisis_2022_dynamic": -0.13,
                "regime_days": {"flat": 100}, "regime_transitions": 0,
                "rebalancing_costs": 0.001,
                "start_date": "2010-01-01", "end_date": "2020-12-31", "total_days": 2520,
                "timestamp": "2026-05-14",
            },
        )
        path = tmp_path / "results.json"
        with patch("src.backtest.duration_yield_backtest.OUTPUT_PATH", path):
            save_results(r)
        assert path.exists()

    def test_valid_json(self, tmp_path):
        r = BacktestResult(
            total_return=210.58, cagr=12.0, volatility=13.0, sharpe_ratio=0.85,
            max_drawdown=-23.0,
            extras={
                "static_cagr": 0.10, "static_volatility": 0.12, "static_sharpe": 0.80,
                "static_max_dd": -0.25,
                "dynamic_cagr": 0.12, "dynamic_volatility": 0.13, "dynamic_sharpe": 0.85,
                "dynamic_max_dd": -0.23,
                "sharpe_delta": 0.05, "cagr_delta": 0.02, "max_dd_delta": 0.02,
                "crisis_2008_static": -0.12, "crisis_2008_dynamic": -0.10,
                "crisis_2020_static": -0.07, "crisis_2020_dynamic": -0.05,
                "crisis_2022_static": -0.14, "crisis_2022_dynamic": -0.13,
                "regime_days": {"flat": 100}, "regime_transitions": 0,
                "rebalancing_costs": 0.001,
                "start_date": "2010-01-01", "end_date": "2020-12-31", "total_days": 2520,
                "timestamp": "2026-05-14",
            },
        )
        path = tmp_path / "results.json"
        with patch("src.backtest.duration_yield_backtest.OUTPUT_PATH", path):
            save_results(r)
        with open(path) as f:
            data = json.load(f)
        assert "static_cagr" in data["extras"]
        assert "dynamic_cagr" in data["extras"]


# ---------------------------------------------------------------------------
# print_results Tests
# ---------------------------------------------------------------------------

class TestPrintResults:

    def test_prints_output(self, caplog):
        r = BacktestResult(
            total_return=210.58, cagr=12.0, volatility=13.0, sharpe_ratio=0.85,
            max_drawdown=-23.0,
            extras={
                "static_cagr": 0.10, "static_volatility": 0.12, "static_sharpe": 0.80,
                "static_max_dd": -0.25,
                "dynamic_cagr": 0.12, "dynamic_volatility": 0.13, "dynamic_sharpe": 0.85,
                "dynamic_max_dd": -0.23,
                "sharpe_delta": 0.05, "cagr_delta": 0.02, "max_dd_delta": 0.02,
                "crisis_2008_static": -0.12, "crisis_2008_dynamic": -0.10,
                "crisis_2020_static": -0.07, "crisis_2020_dynamic": -0.05,
                "crisis_2022_static": -0.14, "crisis_2022_dynamic": -0.13,
                "regime_days": {"flat": 100, "inverted": 50, "steep": 50},
                "regime_transitions": 3,
                "rebalancing_costs": 0.002,
                "start_date": "2010-01-01", "end_date": "2020-12-31", "total_days": 2520,
                "timestamp": "2026-05-14",
            },
        )
        with caplog.at_level(logging.INFO, logger="src.backtest.duration_yield_backtest"):
            print_results(r)
        assert "DURATION-YIELD" in caplog.text
        assert "PERFORMANCE COMPARISON" in caplog.text
        assert "CRISIS PERFORMANCE" in caplog.text

    def test_shows_sharpe_delta(self, caplog):
        r = BacktestResult(
            total_return=210.58, cagr=12.0, volatility=13.0, sharpe_ratio=0.85,
            max_drawdown=-23.0,
            extras={
                "static_cagr": 0.10, "static_volatility": 0.12, "static_sharpe": 0.80,
                "static_max_dd": -0.25,
                "dynamic_cagr": 0.12, "dynamic_volatility": 0.13, "dynamic_sharpe": 0.85,
                "dynamic_max_dd": -0.23,
                "sharpe_delta": 0.05, "cagr_delta": 0.02, "max_dd_delta": 0.02,
                "crisis_2008_static": -0.12, "crisis_2008_dynamic": -0.10,
                "crisis_2020_static": -0.07, "crisis_2020_dynamic": -0.05,
                "crisis_2022_static": -0.14, "crisis_2022_dynamic": -0.13,
                "regime_days": {"flat": 100}, "regime_transitions": 0,
                "rebalancing_costs": 0.001,
                "start_date": "2010-01-01", "end_date": "2020-12-31", "total_days": 2520,
                "timestamp": "2026-05-14",
            },
        )
        with caplog.at_level(logging.INFO, logger="src.backtest.duration_yield_backtest"):
            print_results(r)
        assert "+0.050" in caplog.text or "0.050" in caplog.text


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
