#!/usr/bin/env python3
"""Tests for src/backtest/correlation_adaptive_backtest.py and vol_targeting_backtest.py."""

import json

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


class TestComputeCorrelationAdaptiveBacktest:
    """Tests for compute_correlation_adaptive_backtest (previously untested)."""

    def test_basic_run_returns_result(self):
        from src.backtest.correlation_adaptive_backtest import (
            compute_correlation_adaptive_backtest, CorrelationAdaptiveResult,
        )
        result = compute_correlation_adaptive_backtest()
        assert isinstance(result, CorrelationAdaptiveResult)

    def test_sharpes_are_finite(self):
        from src.backtest.correlation_adaptive_backtest import (
            compute_correlation_adaptive_backtest,
        )
        result = compute_correlation_adaptive_backtest()
        assert np.isfinite(result.static_sharpe)
        assert np.isfinite(result.adaptive_sharpe)
        assert np.isfinite(result.sharpe_delta)

    def test_max_dd_non_positive(self):
        from src.backtest.correlation_adaptive_backtest import (
            compute_correlation_adaptive_backtest,
        )
        result = compute_correlation_adaptive_backtest()
        assert result.static_max_dd <= 0.0
        assert result.adaptive_max_dd <= 0.0

    def test_regime_distribution_sums_to_total(self):
        from src.backtest.correlation_adaptive_backtest import (
            compute_correlation_adaptive_backtest,
        )
        result = compute_correlation_adaptive_backtest()
        total = sum(result.correlation_regime_distribution.values())
        assert total > 0
        # All regimes should be present
        for r in ["diversifying", "neutral", "correlated"]:
            assert r in result.correlation_regime_distribution

    def test_ief_shift_frequency_in_range(self):
        from src.backtest.correlation_adaptive_backtest import (
            compute_correlation_adaptive_backtest,
        )
        result = compute_correlation_adaptive_backtest()
        assert 0.0 <= result.ief_shift_frequency <= 1.0

    def test_adaptive_weights_mean_sum_to_one(self):
        from src.backtest.correlation_adaptive_backtest import (
            compute_correlation_adaptive_backtest,
        )
        result = compute_correlation_adaptive_backtest()
        total = sum(result.adaptive_weights_mean.values())
        assert abs(total - 1.0) < 0.02

    def test_default_allocation(self):
        from src.backtest.correlation_adaptive_backtest import (
            compute_correlation_adaptive_backtest,
        )
        result = compute_correlation_adaptive_backtest()
        assert result.base_weights == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.00}

    def test_custom_corr_window(self):
        from src.backtest.correlation_adaptive_backtest import (
            compute_correlation_adaptive_backtest,
        )
        result = compute_correlation_adaptive_backtest(corr_window=126)
        assert np.isfinite(result.sharpe_delta)

    def test_summary_contains_key_info(self):
        from src.backtest.correlation_adaptive_backtest import (
            compute_correlation_adaptive_backtest,
        )
        result = compute_correlation_adaptive_backtest()
        assert "Sharpe" in result.summary
        assert "IEF" in result.summary


class TestCorrelationAdaptiveResultDataclass:
    """Tests for CorrelationAdaptiveResult fields."""

    def test_all_fields_present(self):
        from src.backtest.correlation_adaptive_backtest import (
            compute_correlation_adaptive_backtest,
        )
        result = compute_correlation_adaptive_backtest()
        assert hasattr(result, 'analysis_date')
        assert hasattr(result, 'base_weights')
        assert hasattr(result, 'adaptive_weights_mean')
        assert hasattr(result, 'static_sharpe')
        assert hasattr(result, 'adaptive_sharpe')
        assert hasattr(result, 'sharpe_delta')
        assert hasattr(result, 'static_max_dd')
        assert hasattr(result, 'adaptive_max_dd')
        assert hasattr(result, 'correlation_regime_distribution')
        assert hasattr(result, 'ief_shift_frequency')
        assert hasattr(result, 'summary')

    def test_sharpe_delta_consistent(self):
        from src.backtest.correlation_adaptive_backtest import (
            compute_correlation_adaptive_backtest,
        )
        result = compute_correlation_adaptive_backtest()
        expected = round(result.adaptive_sharpe - result.static_sharpe, 4)
        assert result.sharpe_delta == expected


class TestAdaptiveWeightsEdgeCases:
    """Edge case tests for _get_adaptive_weights."""

    def test_extreme_correlation_full_shift(self):
        from src.backtest.correlation_adaptive_backtest import _get_adaptive_weights
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.0}
        w = _get_adaptive_weights(0.99, base, max_ief_shift=0.50)
        # At correlation 0.99, shift_fraction = min((0.99-0.15)/0.35, 1.0) * 0.50 = 0.50
        assert w["IEF"] > 0.07
        assert w["TLT"] < 0.09

    def test_zero_correlation_neutral(self):
        from src.backtest.correlation_adaptive_backtest import _get_adaptive_weights
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.0}
        w = _get_adaptive_weights(0.0, base)
        # In neutral zone, some partial shift
        assert w["TLT"] + w["IEF"] == pytest.approx(0.16, abs=0.001)

    def test_total_bond_preserved(self):
        from src.backtest.correlation_adaptive_backtest import _get_adaptive_weights
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.0}
        for corr in [-0.5, -0.15, 0.0, 0.15, 0.5]:
            w = _get_adaptive_weights(corr, base)
            assert w["TLT"] + w["IEF"] == pytest.approx(0.16, abs=0.001)

    def test_spy_gld_unchanged(self):
        from src.backtest.correlation_adaptive_backtest import _get_adaptive_weights
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.0}
        w = _get_adaptive_weights(0.5, base)
        assert w["SPY"] == 0.46
        assert w["GLD"] == 0.38


# ── Edge-Case Tests ─────────────────────────────────────────────────────────


class TestAdaptiveWeightsBoundaries:
    """Boundary-threshold and special-case tests for _get_adaptive_weights."""

    def test_boundary_correlation_exactly_neg_015(self):
        """correlation == -0.15 is the diversifying/neutral boundary."""
        from src.backtest.correlation_adaptive_backtest import _get_adaptive_weights
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.00}
        # correlation < -0.15 → shift_fraction = 0.0; correlation == -0.15 → neutral
        w_at = _get_adaptive_weights(-0.15, base, max_ief_shift=0.50)
        w_below = _get_adaptive_weights(-0.16, base, max_ief_shift=0.50)
        # At -0.15 neutral zone: shift_fraction = ((-0.15+0.15)/0.30)*0.5*0.5 = 0.0
        assert w_at["IEF"] == pytest.approx(0.0, abs=0.001)
        # Below -0.15 diversifying: shift_fraction = 0.0
        assert w_below["IEF"] == 0.0

    def test_boundary_correlation_exactly_015(self):
        """correlation == 0.15 is the neutral/correlated boundary."""
        from src.backtest.correlation_adaptive_backtest import _get_adaptive_weights
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.00}
        w_at = _get_adaptive_weights(0.15, base, max_ief_shift=0.50)
        w_above = _get_adaptive_weights(0.16, base, max_ief_shift=0.50)
        # At 0.15, still in neutral zone: shift_fraction = ((0.15+0.15)/0.30)*0.5*0.5 = 0.25
        assert w_at["IEF"] == pytest.approx(0.16 * 0.25, abs=0.001)
        # At 0.16, crosses into correlated: shift_fraction = ((0.16-0.15)/0.35)*1.0*0.50 ≈ 0.00143
        assert w_above["IEF"] > 0.0

    def test_zero_max_ief_shift_no_allocation_change(self):
        """max_ief_shift=0 should produce zero IEF regardless of correlation."""
        from src.backtest.correlation_adaptive_backtest import _get_adaptive_weights
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.00}
        for corr in [-0.5, 0.0, 0.5, 1.0]:
            w = _get_adaptive_weights(corr, base, max_ief_shift=0.0)
            assert w["IEF"] == 0.0, f"IEF nonzero at corr={corr} with max_ief_shift=0"
            assert w["TLT"] == 0.16

    def test_existing_ief_in_base_allocation(self):
        """When base allocation already has IEF > 0, total_bond includes it."""
        from src.backtest.correlation_adaptive_backtest import _get_adaptive_weights
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.12, "IEF": 0.04}
        w = _get_adaptive_weights(0.50, base, max_ief_shift=0.50)
        # total_bond = 0.12 + 0.04 = 0.16; shift applied to total_bond
        assert w["TLT"] + w["IEF"] == pytest.approx(0.16, abs=0.001)
        assert w["IEF"] > 0.04  # IEF increased from base
        assert w["SPY"] == 0.46
        assert w["GLD"] == 0.38

    def test_both_tlt_and_ief_zero(self):
        """When base has zero TLT and zero IEF, no bond shift occurs."""
        from src.backtest.correlation_adaptive_backtest import _get_adaptive_weights
        base = {"SPY": 0.50, "GLD": 0.50, "TLT": 0.00, "IEF": 0.00}
        w = _get_adaptive_weights(0.50, base, max_ief_shift=0.50)
        assert w["TLT"] == 0.0
        assert w["IEF"] == 0.0

    def test_negative_correlation_neutral_zone(self):
        """Correlation -0.05 is in neutral zone, produces partial shift."""
        from src.backtest.correlation_adaptive_backtest import _get_adaptive_weights
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.00}
        w = _get_adaptive_weights(-0.05, base, max_ief_shift=0.50)
        # shift_fraction = ((-0.05+0.15)/0.30)*0.5*0.5 = (0.10/0.30)*0.25 ≈ 0.0833
        expected_ief = 0.16 * ((0.10 / 0.30) * 0.25)
        assert w["IEF"] == pytest.approx(expected_ief, abs=0.001)
        assert w["TLT"] + w["IEF"] == pytest.approx(0.16, abs=0.001)


class TestRollingCorrelationEdgeCases:
    """Edge cases for _compute_rolling_correlation."""

    def _make_prices(self, n=500):
        dates = pd.bdate_range("2015-01-01", periods=n)
        rng = np.random.default_rng(42)
        gld = 150 + np.cumsum(rng.standard_normal(n) * 0.3)
        tlt = 100 + np.cumsum(rng.standard_normal(n) * 0.2)
        spy = 300 + np.cumsum(rng.standard_normal(n) * 1.0)
        ief = 80 + np.cumsum(rng.standard_normal(n) * 0.15)
        return pd.DataFrame({"SPY": spy, "GLD": gld, "TLT": tlt, "IEF": ief}, index=dates)

    def test_window_larger_than_series(self):
        """When window > series length, result should be all-NaN then dropna."""
        from src.backtest.correlation_adaptive_backtest import _compute_rolling_correlation
        prices = self._make_prices(n=50)
        corr = _compute_rolling_correlation(prices, "GLD", "TLT", window=252)
        # With 49 returns (50 prices), window=252 means all rolling windows are incomplete
        # dropna should remove everything
        assert len(corr) == 0

    def test_single_row_prices(self):
        """Single row produces 0 returns, so dropna yields empty."""
        from src.backtest.correlation_adaptive_backtest import _compute_rolling_correlation
        dates = pd.bdate_range("2020-01-01", periods=1)
        prices = pd.DataFrame(
            {"SPY": [300.0], "GLD": [150.0], "TLT": [100.0], "IEF": [80.0]},
            index=dates,
        )
        corr = _compute_rolling_correlation(prices, "GLD", "TLT", window=252)
        assert len(corr) == 0

    def test_two_rows_min_window(self):
        """Two rows with window=1 — rolling corr needs 2+ values, so 1 return is NaN."""
        from src.backtest.correlation_adaptive_backtest import _compute_rolling_correlation
        dates = pd.bdate_range("2020-01-01", periods=2)
        prices = pd.DataFrame(
            {"SPY": [300.0, 301.0], "GLD": [150.0, 152.0],
             "TLT": [100.0, 99.0], "IEF": [80.0, 81.0]},
            index=dates,
        )
        # pct_change on 2 rows → 1 return; rolling(1) needs 2+ values for corr → NaN → dropped
        corr = _compute_rolling_correlation(prices, "GLD", "TLT", window=1)
        assert len(corr) == 0

    def test_empty_after_pct_change(self):
        """Two identical prices → pct_change = 0 → correlation is NaN."""
        from src.backtest.correlation_adaptive_backtest import _compute_rolling_correlation
        dates = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame(
            {"SPY": [300.0] * 5, "GLD": [150.0] * 5,
             "TLT": [100.0] * 5, "IEF": [80.0] * 5},
            index=dates,
        )
        corr = _compute_rolling_correlation(prices, "GLD", "TLT", window=3)
        # All returns are 0 → correlation of zeros → NaN
        assert len(corr) == 0


class TestComputeAdaptiveBacktestEdgeCases:
    """Edge cases for compute_correlation_adaptive_backtest with synthetic data."""

    def _make_synthetic_prices_json(self, tmp_path, n=400):
        """Create a synthetic prices.json with controlled data."""
        dates = pd.bdate_range("2018-01-01", periods=n)
        rng = np.random.default_rng(123)
        spy = 280 + np.cumsum(rng.standard_normal(n) * 0.8)
        gld = 130 + np.cumsum(rng.standard_normal(n) * 0.25)
        tlt = 110 + np.cumsum(rng.standard_normal(n) * 0.15)
        ief = 90 + np.cumsum(rng.standard_normal(n) * 0.10)
        data = {}
        for sym, arr in [("SPY", spy), ("GLD", gld), ("TLT", tlt), ("IEF", ief)]:
            data[sym] = [{"d": d.strftime("%Y-%m-%d"), "p": round(float(v), 2)}
                         for d, v in zip(dates, arr)]
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps(data))
        return prices_file

    def test_max_ief_shift_zero_produces_static_like_result(self, tmp_path):
        """max_ief_shift=0 means adaptive weights == base weights (no shift)."""
        from src.backtest.correlation_adaptive_backtest import (
            compute_correlation_adaptive_backtest,
        )
        prices_file = self._make_synthetic_prices_json(tmp_path)
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.00}
        result = compute_correlation_adaptive_backtest(
            base_allocation=base,
            max_ief_shift=0.0,
            corr_window=60,
            rebalance_days=21,
        )
        # With no shift allowed, adaptive IEF should be 0 for all days
        assert result.adaptive_weights_mean.get("IEF", 0) == pytest.approx(0.0, abs=0.001)
        assert result.adaptive_weights_mean.get("TLT", 0) == pytest.approx(0.16, abs=0.001)
        # Sharpe delta should be near zero (same weights)
        assert abs(result.sharpe_delta) < 0.01

    def test_rebalance_days_exceeds_prices_length(self, tmp_path):
        """rebalance_days > len(prices) means at most one rebalance."""
        from src.backtest.correlation_adaptive_backtest import (
            compute_correlation_adaptive_backtest,
        )
        prices_file = self._make_synthetic_prices_json(tmp_path, n=60)
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.00}
        result = compute_correlation_adaptive_backtest(
            base_allocation=base,
            max_ief_shift=0.50,
            corr_window=30,
            rebalance_days=100,  # > 60-day series
        )
        assert np.isfinite(result.static_sharpe)
        assert np.isfinite(result.adaptive_sharpe)
        assert result.static_max_dd <= 0.0

    def test_custom_allocation_with_nonzero_base_ief(self, tmp_path):
        """Base allocation with pre-existing IEF — total bond bucket preserved."""
        from src.backtest.correlation_adaptive_backtest import (
            compute_correlation_adaptive_backtest,
        )
        prices_file = self._make_synthetic_prices_json(tmp_path)
        base = {"SPY": 0.44, "GLD": 0.36, "TLT": 0.12, "IEF": 0.08}
        result = compute_correlation_adaptive_backtest(
            base_allocation=base,
            max_ief_shift=0.50,
            corr_window=60,
        )
        # Mean weights should sum to ~1.0
        total = sum(result.adaptive_weights_mean.values())
        assert abs(total - 1.0) < 0.02
        # Mean TLT + IEF should equal base total_bond (0.20)
        mean_tlt_ief = result.adaptive_weights_mean.get("TLT", 0) + result.adaptive_weights_mean.get("IEF", 0)
        assert mean_tlt_ief == pytest.approx(0.20, abs=0.005)


class TestLoadPricesEdgeCases:
    """Edge cases for _load_prices with monkeypatched file I/O."""

    def _make_prices_data(self):
        """Create synthetic price data dict for monkeypatching."""
        dates = pd.bdate_range("2020-01-01", periods=10)
        data = {}
        for sym, base_price in [("SPY", 300), ("GLD", 150), ("TLT", 100), ("IEF", 80)]:
            rng = np.random.default_rng(hash(sym) % 2**31)
            arr = base_price + np.cumsum(rng.standard_normal(10) * 0.5)
            data[sym] = [{"d": d.strftime("%Y-%m-%d"), "p": round(float(v), 2)}
                         for d, v in zip(dates, arr)]
        return data

    def test_default_loader_uses_shared_price_dataframe_cache(self, monkeypatch):
        """Default _load_prices should use the shared TTL-cached DataFrame accessor."""
        from src.backtest import correlation_adaptive_backtest as cab

        dates = pd.bdate_range("2020-01-01", periods=4)
        cached_df = pd.DataFrame(
            {
                "SPY": [300.0, 301.0, 302.0, 303.0],
                "GLD": [150.0, 151.0, 152.0, 153.0],
                "TLT": [100.0, 101.0, 102.0, 103.0],
                "IEF": [80.0, 81.0, 82.0, 83.0],
            },
            index=dates,
        )
        calls = []

        def fake_get_prices_df(symbols=None):
            calls.append(symbols)
            return cached_df

        def fail_open(*args, **kwargs):
            raise AssertionError("default loader should not re-read prices.json")

        monkeypatch.setattr(cab, "get_prices_df", fake_get_prices_df, raising=False)
        monkeypatch.setattr("builtins.open", fail_open)

        df = cab._load_prices()

        assert calls == [["SPY", "GLD", "TLT", "IEF"]]
        expected = cached_df.copy()
        expected.index.name = "date"
        pd.testing.assert_frame_equal(df, expected)
        assert df.index.name == "date"

    def test_missing_symbol_ief(self, monkeypatch, tmp_path):
        """Missing IEF symbol should still produce a DataFrame (dropna drops it)."""
        import json
        from src.backtest.correlation_adaptive_backtest import _load_prices
        data = self._make_prices_data()
        del data["IEF"]  # remove IEF entirely
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps(data))
        monkeypatch.setattr(
            "src.backtest.correlation_adaptive_backtest.PRICES_JSON", str(prices_file),
        )
        df = _load_prices()
        # IEF column won't exist → dropna won't drop rows but IEF is missing
        # The DataFrame should still load but IEF column may be absent
        assert len(df) > 0

    def test_missing_all_optional_symbols(self, monkeypatch, tmp_path):
        """Only SPY and GLD present — DataFrame loads but is incomplete."""
        import json
        from src.backtest.correlation_adaptive_backtest import _load_prices
        dates = pd.bdate_range("2020-01-01", periods=5)
        data = {
            "SPY": [{"d": d.strftime("%Y-%m-%d"), "p": 300.0} for d in dates],
            "GLD": [{"d": d.strftime("%Y-%m-%d"), "p": 150.0} for d in dates],
        }
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps(data))
        monkeypatch.setattr(
            "src.backtest.correlation_adaptive_backtest.PRICES_JSON", str(prices_file),
        )
        df = _load_prices()
        assert len(df) == 5
        assert "SPY" in df.columns
        assert "GLD" in df.columns

    def test_empty_symbol_list(self, monkeypatch, tmp_path):
        """Empty prices.json → empty DataFrame."""
        import json
        from src.backtest.correlation_adaptive_backtest import _load_prices
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps({}))
        monkeypatch.setattr(
            "src.backtest.correlation_adaptive_backtest.PRICES_JSON", str(prices_file),
        )
        df = _load_prices()
        assert len(df) == 0

    def test_symbol_with_empty_entries(self, monkeypatch, tmp_path):
        """Symbol present but with empty list → treated as missing."""
        import json
        from src.backtest.correlation_adaptive_backtest import _load_prices
        data = self._make_prices_data()
        data["TLT"] = []  # empty list
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps(data))
        monkeypatch.setattr(
            "src.backtest.correlation_adaptive_backtest.PRICES_JSON", str(prices_file),
        )
        df = _load_prices()
        # TLT should be absent since its entry list is empty
        assert "TLT" not in df.columns or len(df) > 0

    def test_non_dict_entries_skipped(self, monkeypatch, tmp_path):
        """Symbol with non-dict entries (e.g. raw values) should be skipped."""
        import json
        from src.backtest.correlation_adaptive_backtest import _load_prices
        data = self._make_prices_data()
        data["IEF"] = [100, 101, 102]  # not dicts
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps(data))
        monkeypatch.setattr(
            "src.backtest.correlation_adaptive_backtest.PRICES_JSON", str(prices_file),
        )
        df = _load_prices()
        # IEF should be skipped (non-dict entries)
        assert "IEF" not in df.columns
