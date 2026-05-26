#!/usr/bin/env python3
"""Tests for src/backtest/correlation_adaptive_backtest.py and vol_targeting_backtest.py."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


# ── Correlation Adaptive Tests ──────────────────────────────────────────────


class TestAdaptiveWeights:
    """Tests for _get_adaptive_weights."""

    def test_correlated_regime_shifts_to_ief(self):
        from src.backtest.correlation_adaptive_backtest import _get_adaptive_weights

        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.00}
        w = _get_adaptive_weights(0.30, base, max_ief_shift=0.50)
        assert w["IEF"] > 0.02  # meaningful IEF allocation
        assert w["TLT"] < 0.14  # TLT reduced
        assert abs(sum(w.values()) - 1.0) < 0.02  # near-sum to 1

    def test_diversifying_regime_keeps_tlt(self):
        from src.backtest.correlation_adaptive_backtest import _get_adaptive_weights

        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.00}
        w = _get_adaptive_weights(-0.30, base, max_ief_shift=0.50)
        assert w["IEF"] == 0.0
        assert w["TLT"] == 0.16

    def test_neutral_regime_partial_shift(self):
        from src.backtest.correlation_adaptive_backtest import _get_adaptive_weights

        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.00}
        w = _get_adaptive_weights(0.0, base, max_ief_shift=0.50)
        assert 0 < w["IEF"] < 0.10  # partial shift
        assert 0.08 < w["TLT"] < 0.16

    def test_preserves_ie_free_weights(self):
        from src.backtest.correlation_adaptive_backtest import _get_adaptive_weights

        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.00}
        w = _get_adaptive_weights(0.50, base)
        assert w["SPY"] == 0.46
        assert w["GLD"] == 0.38

    def test_max_shift_respected(self):
        from src.backtest.correlation_adaptive_backtest import _get_adaptive_weights

        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.00}
        w = _get_adaptive_weights(0.50, base, max_ief_shift=1.0)
        # At max correlation, all TLT should shift to IEF
        assert abs(w["TLT"] - 0.0) < 0.01
        assert abs(w["IEF"] - 0.16) < 0.01


class TestRollingCorrelation:
    """Tests for _compute_rolling_correlation."""

    def _make_prices(self, n=500):
        dates = pd.bdate_range("2015-01-01", periods=n)
        rng = np.random.default_rng(42)
        gld = 150 + np.cumsum(rng.standard_normal(n) * 0.3)
        tlt = 100 + np.cumsum(rng.standard_normal(n) * 0.2)
        spy = 300 + np.cumsum(rng.standard_normal(n) * 1.0)
        ief = 80 + np.cumsum(rng.standard_normal(n) * 0.15)
        return pd.DataFrame({"SPY": spy, "GLD": gld, "TLT": tlt, "IEF": ief}, index=dates)

    def test_correlation_bounded(self):
        from src.backtest.correlation_adaptive_backtest import _compute_rolling_correlation

        prices = self._make_prices()
        corr = _compute_rolling_correlation(prices, "GLD", "TLT", window=100)
        assert (corr >= -1.0).all() and (corr <= 1.0).all()

    def test_returns_series_with_data(self):
        from src.backtest.correlation_adaptive_backtest import _compute_rolling_correlation

        prices = self._make_prices()
        corr = _compute_rolling_correlation(prices, "GLD", "TLT", window=100)
        assert len(corr) > 0


class TestBacktestHelpers:
    """Tests for _run_portfolio_backtest."""

    def _make_prices_df(self, n=200):
        dates = pd.bdate_range("2020-01-01", periods=n)
        spy = 100 + np.cumsum(np.random.default_rng(42).standard_normal(n) * 0.5)
        gld = 150 + np.cumsum(np.random.default_rng(42).standard_normal(n) * 0.3)
        tlt = 100 + np.cumsum(np.random.default_rng(42).standard_normal(n) * 0.2)
        ief = 80 + np.cumsum(np.random.default_rng(42).standard_normal(n) * 0.15)
        return pd.DataFrame({"SPY": spy, "GLD": gld, "TLT": tlt, "IEF": ief}, index=dates)

    def test_backtest_runs(self):
        from src.backtest.correlation_adaptive_backtest import _run_portfolio_backtest

        prices = self._make_prices_df()
        weights = [{"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.00}] * len(prices)
        returns, values = _run_portfolio_backtest(prices, weights)
        assert len(returns) == len(prices) - 1
        assert len(values) == len(prices)

    def test_equal_allocations_same_result(self):
        from src.backtest.correlation_adaptive_backtest import _run_portfolio_backtest

        prices = self._make_prices_df()
        w1 = [{"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.0}] * len(prices)
        r1, v1 = _run_portfolio_backtest(prices, w1)
        r2, v2 = _run_portfolio_backtest(prices, w1)  # same weights
        assert v1[-1] == v2[-1]  # deterministic

    def test_all_in_one_asset(self):
        from src.backtest.correlation_adaptive_backtest import _run_portfolio_backtest

        prices = self._make_prices_df()
        w = [{"SPY": 1.0, "GLD": 0.0, "TLT": 0.0, "IEF": 0.0}] * len(prices)
        returns, values = _run_portfolio_backtest(prices, w)
        assert values[0] == 100000.0
        assert len(returns) > 0


# ── Vol Targeting Tests ─────────────────────────────────────────────────────


class TestVolTargetLeverage:
    """Tests for _compute_vol_target_leverage."""

    def test_low_vol_increases_leverage(self):
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        lev = _compute_vol_target_leverage(0.05, 0.11, max_leverage=2.0)
        assert lev > 1.5  # scale up significantly

    def test_high_vol_reduces_leverage(self):
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        lev = _compute_vol_target_leverage(0.25, 0.11, max_leverage=2.0)
        assert lev < 0.7  # scale down significantly (smoothed toward 1.0)

    def test_at_target_vol(self):
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        lev = _compute_vol_target_leverage(0.11, 0.11, max_leverage=2.0)
        assert 0.95 < lev < 1.05  # near 1.0

    def test_respects_max_leverage(self):
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        lev = _compute_vol_target_leverage(0.01, 0.11, max_leverage=1.5)
        assert lev <= 1.5

    def test_respects_min_leverage(self):
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        lev = _compute_vol_target_leverage(0.50, 0.11, max_leverage=2.0)
        assert lev >= 0.5  # max_leverage=2.0 → min = 1/2.0

    def test_smoothing_partial(self):
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        # With smoothing=0.67, first call from 1.0 should be partial
        lev = _compute_vol_target_leverage(
            0.05, 0.11, max_leverage=2.0, smoothing=0.67, prev_leverage=1.0,
        )
        # raw = 0.11/0.05 = 2.2; smoothed = 0.67*2.2 + 0.33*1.0 = 1.804; capped at 2.0
        expected = 0.67 * 2.2 + 0.33 * 1.0  # = 1.804
        assert abs(lev - expected) < 0.01

    def test_zero_vol_returns_one(self):
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        lev = _compute_vol_target_leverage(0.0, 0.11)
        assert lev == 1.0
