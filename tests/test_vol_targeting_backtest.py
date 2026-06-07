#!/usr/bin/env python3
"""Tests for src/backtest/vol_targeting_backtest.py.

Coverage: _compute_vol_target_leverage (6), _classify_regime_from_vol (6),
compute_vol_target_backtest (8), compute_regime_conditional_vol_target_backtest (6),
VolTargetResult (2), RegimeVolTargetResult (2), edge cases (5).
"""

import numpy as np
import pytest


class TestLoadPrices:
    """Tests for cached price loading."""

    def test_load_prices_uses_shared_price_dataframe_cache(self, monkeypatch):
        import pandas as pd
        from src.backtest import vol_targeting_backtest as mod

        dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
        cached_df = pd.DataFrame(
            {
                "SPY": [100.0, 101.0],
                "GLD": [100.0, 99.5],
                "TLT": [100.0, 100.5],
                "IEF": [100.0, 100.1],
            },
            index=dates,
        )
        calls = []

        def fake_get_prices_df(symbols=None):
            calls.append(symbols)
            return cached_df

        def fail_open(*args, **kwargs):
            raise AssertionError("_load_prices should use get_prices_df")

        monkeypatch.setattr(mod, "get_prices_df", fake_get_prices_df, raising=False)
        monkeypatch.setattr("builtins.open", fail_open)

        result = mod._load_prices()

        assert calls == [["SPY", "GLD", "TLT", "IEF"]]
        assert result.index.name == "date"
        assert list(result.columns) == ["SPY", "GLD", "TLT", "IEF"]
        assert result.equals(cached_df.rename_axis("date"))


class TestVolTargetLeverage:
    """Tests for _compute_vol_target_leverage."""

    def test_basic_leverage(self):
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        # 10% realized vol, 10% target → 1.0x leverage
        lev = _compute_vol_target_leverage(0.10, 0.10, smoothing=1.0)
        assert abs(lev - 1.0) < 0.01

    def test_low_vol_increases_leverage(self):
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        # 5% realized vol, 10% target → ~2.0x leverage
        lev = _compute_vol_target_leverage(0.05, 0.10, smoothing=1.0)
        assert lev > 1.5

    def test_high_vol_decreases_leverage(self):
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        # 20% realized vol, 10% target → ~0.5x leverage
        lev = _compute_vol_target_leverage(0.20, 0.10, smoothing=1.0)
        assert lev < 0.8

    def test_max_leverage_cap(self):
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        # Very low vol, 1.5x max → capped at 1.5
        lev = _compute_vol_target_leverage(0.01, 0.10, max_leverage=1.5, smoothing=1.0)
        assert lev <= 1.5

    def test_zero_vol_returns_one(self):
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        lev = _compute_vol_target_leverage(0.0, 0.10, smoothing=1.0)
        assert lev == 1.0

    def test_smoothing_dampens_changes(self):
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        # Without smoothing: target 0.10 / realized 0.05 = 2.0
        raw = _compute_vol_target_leverage(0.05, 0.10, smoothing=1.0, prev_leverage=1.0)
        # With smoothing: 0.67 * 2.0 + 0.33 * 1.0 = 1.67
        smoothed = _compute_vol_target_leverage(0.05, 0.10, smoothing=0.67, prev_leverage=1.0)
        assert abs(raw - smoothed) > 0.1  # smoothing should make a difference
        assert 1.0 < smoothed < raw  # between prev and raw

    def test_scaling_exponent_dampens_low_vol_leverage(self):
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        linear = _compute_vol_target_leverage(
            0.05, 0.10, max_leverage=3.0, smoothing=1.0, scaling_exponent=1.0,
        )
        square_root = _compute_vol_target_leverage(
            0.05, 0.10, max_leverage=3.0, smoothing=1.0, scaling_exponent=0.5,
        )

        assert linear == pytest.approx(2.0)
        assert square_root == pytest.approx(np.sqrt(2.0), abs=0.0001)
        assert 1.0 < square_root < linear

    def test_scaling_exponent_dampens_high_vol_deleveraging(self):
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        linear = _compute_vol_target_leverage(
            0.20, 0.10, max_leverage=3.0, smoothing=1.0, scaling_exponent=1.0,
        )
        reduced = _compute_vol_target_leverage(
            0.20, 0.10, max_leverage=3.0, smoothing=1.0, scaling_exponent=0.5,
        )

        assert linear == pytest.approx(0.5)
        assert reduced == pytest.approx(np.sqrt(0.5), abs=0.0001)
        assert linear < reduced < 1.0


class TestRegimeVolScalingConfig:
    """Tests for regime-specific vol-targeting scaling configuration."""

    def test_default_scaling_exponents_cover_all_regimes(self):
        from src.backtest.vol_targeting_backtest import REGIME_VOL_SCALING_EXPONENTS

        expected = {"CRISIS", "HIGH_VOL", "NORMAL", "LOW_VOL", "RECOVERY"}
        assert set(REGIME_VOL_SCALING_EXPONENTS.keys()) == expected
        assert REGIME_VOL_SCALING_EXPONENTS["NORMAL"] == pytest.approx(1.0)
        assert REGIME_VOL_SCALING_EXPONENTS["LOW_VOL"] == pytest.approx(0.5)
        assert (
            REGIME_VOL_SCALING_EXPONENTS["CRISIS"]
            <= REGIME_VOL_SCALING_EXPONENTS["HIGH_VOL"]
        )

    def test_default_adaptive_lookbacks_cover_all_regimes(self):
        from src.backtest.vol_targeting_backtest import REGIME_VOL_LOOKBACKS

        expected = {"CRISIS", "HIGH_VOL", "NORMAL", "LOW_VOL", "RECOVERY"}
        assert set(REGIME_VOL_LOOKBACKS.keys()) == expected
        assert REGIME_VOL_LOOKBACKS["LOW_VOL"] == 20
        assert REGIME_VOL_LOOKBACKS["NORMAL"] == 63
        assert REGIME_VOL_LOOKBACKS["CRISIS"] == 252

    def test_unknown_regime_falls_back_to_default_scaling_and_lookback(self):
        from src.backtest.vol_targeting_backtest import (
            _get_regime_scaling_exponent,
            _get_regime_vol_lookback,
        )

        assert _get_regime_scaling_exponent("UNKNOWN") == pytest.approx(1.0)
        assert _get_regime_vol_lookback("UNKNOWN", default_lookback=63) == 63


class TestRegimeClassification:
    """Tests for _classify_regime_from_vol."""

    def test_crisis_at_high_ratio(self):
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol

        regime = _classify_regime_from_vol(0.20, median_vol=0.10)
        assert regime == "CRISIS"  # ratio = 2.0 > 1.7

    def test_high_vol_at_moderate_ratio(self):
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol

        regime = _classify_regime_from_vol(0.15, median_vol=0.10)
        assert regime == "HIGH_VOL"  # ratio = 1.5, between 1.25 and 1.7

    def test_normal_at_unit_ratio(self):
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol

        regime = _classify_regime_from_vol(0.10, median_vol=0.10)
        assert regime == "NORMAL"  # ratio = 1.0

    def test_low_vol_at_low_ratio(self):
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol

        regime = _classify_regime_from_vol(0.05, median_vol=0.10)
        assert regime == "LOW_VOL"  # ratio = 0.5 < 0.75

    def test_recovery_when_declining(self):
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol

        # vol is in normal range but was previously CRISIS and is declining
        regime = _classify_regime_from_vol(
            0.12, median_vol=0.10, prev_regime="CRISIS", vol_declining=True,
        )
        assert regime == "RECOVERY"

    def test_zero_median_returns_normal(self):
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol

        regime = _classify_regime_from_vol(0.20, median_vol=0.0)
        assert regime == "NORMAL"


class TestRegimeVolTargetBacktest:
    """Integration tests for regime-conditional vol targeting."""

    def test_backtest_runs_and_produces_regime_breakdown(self):
        from src.backtest.vol_targeting_backtest import (
            compute_regime_conditional_vol_target_backtest,
        )

        result = compute_regime_conditional_vol_target_backtest(
            vol_lookback=63,
            max_leverage=1.5,
        )

        assert result.static_sharpe > 0
        assert result.vol_target_sharpe > 0
        assert result.mean_leverage > 0
        assert len(result.regime_breakdown) == 5

        # All 5 regimes should appear
        for reg in ["CRISIS", "HIGH_VOL", "NORMAL", "LOW_VOL", "RECOVERY"]:
            assert reg in result.regime_breakdown

    def test_regime_targets_are_respected(self):
        from src.backtest.vol_targeting_backtest import (
            REGIME_VOL_TARGETS,
            compute_regime_conditional_vol_target_backtest,
        )

        result = compute_regime_conditional_vol_target_backtest(
            vol_lookback=63,
            max_leverage=1.5,
        )

        for reg, info in result.regime_breakdown.items():
            expected_target = REGIME_VOL_TARGETS.get(reg, 0.09)
            assert info["target_vol"] == expected_target

    def test_default_regime_targets_are_defensive(self):
        from src.backtest.vol_targeting_backtest import REGIME_VOL_TARGETS

        assert REGIME_VOL_TARGETS["CRISIS"] == pytest.approx(0.03)
        assert REGIME_VOL_TARGETS["HIGH_VOL"] == pytest.approx(0.05)
        assert REGIME_VOL_TARGETS["NORMAL"] == pytest.approx(0.08)
        assert REGIME_VOL_TARGETS["LOW_VOL"] == pytest.approx(0.10)
        assert REGIME_VOL_TARGETS["RECOVERY"] == pytest.approx(0.09)

    def test_default_regime_scaling_meets_risk_targets(self):
        from src.backtest.vol_targeting_backtest import (
            compute_regime_conditional_vol_target_backtest,
        )

        result = compute_regime_conditional_vol_target_backtest()

        assert result.vol_target_sharpe >= 1.0
        assert result.vol_target_max_dd >= -18.0

    def test_custom_regime_targets(self):
        from src.backtest.vol_targeting_backtest import (
            compute_regime_conditional_vol_target_backtest,
        )

        custom = {"CRISIS": 0.03, "NORMAL": 0.10}
        result = compute_regime_conditional_vol_target_backtest(
            regime_targets=custom,
            vol_lookback=63,
            max_leverage=1.5,
        )

        assert result.regime_breakdown["CRISIS"]["target_vol"] == 0.03
        assert result.regime_breakdown["NORMAL"]["target_vol"] == 0.10

    def test_sharpe_is_finite(self):
        from src.backtest.vol_targeting_backtest import (
            compute_regime_conditional_vol_target_backtest,
        )

        result = compute_regime_conditional_vol_target_backtest(
            vol_lookback=63,
            max_leverage=1.5,
        )

        assert np.isfinite(result.static_sharpe)
        assert np.isfinite(result.vol_target_sharpe)
        assert np.isfinite(result.sharpe_delta)
        assert np.isfinite(result.mean_leverage)
        assert np.isfinite(result.vol_target_max_dd)

    def test_regime_days_sum_to_total(self):
        from src.backtest.vol_targeting_backtest import (
            compute_regime_conditional_vol_target_backtest,
        )

        result = compute_regime_conditional_vol_target_backtest(
            vol_lookback=63,
            max_leverage=1.5,
        )

        total_pct = sum(info["pct_of_time"] for info in result.regime_breakdown.values())
        assert abs(total_pct - 1.0) < 0.01

    def test_result_to_dict(self):
        from src.backtest.vol_targeting_backtest import (
            compute_regime_conditional_vol_target_backtest,
        )

        result = compute_regime_conditional_vol_target_backtest(
            vol_lookback=63,
            max_leverage=1.5,
        )

        d = result.__dict__ if hasattr(result, '__dict__') else {}
        # Should have key fields
        assert "static_sharpe" in str(result) or hasattr(result, 'static_sharpe')


class TestComputeVolTargetBacktest:
    """Tests for compute_vol_target_backtest (previously untested)."""

    def test_basic_backtest_uses_precomputed_realized_vols(self, monkeypatch):
        import pandas as pd
        from src.backtest import vol_targeting_backtest as mod

        dates = pd.date_range("2020-01-02", periods=12, freq="B")
        prices = pd.DataFrame(
            {
                "SPY": np.linspace(100.0, 111.0, len(dates)),
                "GLD": np.linspace(100.0, 105.0, len(dates)),
                "TLT": np.linspace(100.0, 98.0, len(dates)),
                "IEF": np.linspace(100.0, 101.0, len(dates)),
            },
            index=dates,
        )
        prices.index.name = "date"
        calls = []
        original_precompute = mod._precompute_realized_vols

        def record_precompute(portfolio_returns, lookbacks, fallback_vol=0.15):
            calls.append(tuple(lookbacks))
            return original_precompute(portfolio_returns, lookbacks, fallback_vol)

        monkeypatch.setattr(mod, "_load_prices", lambda: prices)
        monkeypatch.setattr(mod, "_precompute_realized_vols", record_precompute)

        result = mod.compute_vol_target_backtest(vol_lookback=3, max_leverage=1.5)

        assert calls == [(3,)]
        assert np.isfinite(result.vol_target_sharpe)

    def test_basic_run_returns_vol_target_result(self):
        from src.backtest.vol_targeting_backtest import (
            compute_vol_target_backtest, VolTargetResult,
        )
        result = compute_vol_target_backtest(target_vol=0.11, max_leverage=1.5)
        assert isinstance(result, VolTargetResult)

    def test_sharpes_are_finite(self):
        from src.backtest.vol_targeting_backtest import compute_vol_target_backtest
        result = compute_vol_target_backtest(target_vol=0.11, max_leverage=1.5)
        assert np.isfinite(result.static_sharpe)
        assert np.isfinite(result.vol_target_sharpe)
        assert np.isfinite(result.sharpe_delta)

    def test_leverage_stats_in_range(self):
        from src.backtest.vol_targeting_backtest import compute_vol_target_backtest
        result = compute_vol_target_backtest(target_vol=0.09, max_leverage=1.5)
        assert result.mean_leverage >= 0.5
        assert result.mean_leverage <= 2.0
        assert result.max_leverage_reached <= 1.5 + 0.01  # within max_leverage cap
        assert 0.0 <= result.leverage_above_1_pct <= 1.0

    def test_max_dd_non_positive(self):
        from src.backtest.vol_targeting_backtest import compute_vol_target_backtest
        result = compute_vol_target_backtest(target_vol=0.11, max_leverage=1.5)
        assert result.static_max_dd <= 0.0
        assert result.vol_target_max_dd <= 0.0

    def test_cagr_non_negative(self):
        from src.backtest.vol_targeting_backtest import compute_vol_target_backtest
        result = compute_vol_target_backtest(target_vol=0.11, max_leverage=1.5)
        # Over 20 years of real data, CAGR should be positive
        assert result.static_cagr > 0
        assert result.vol_target_cagr > 0

    def test_summary_contains_key_info(self):
        from src.backtest.vol_targeting_backtest import compute_vol_target_backtest
        result = compute_vol_target_backtest(target_vol=0.11, max_leverage=1.5)
        assert "Sharpe" in result.summary
        assert "leverage" in result.summary.lower()

    def test_default_allocation_is_463816(self):
        from src.backtest.vol_targeting_backtest import compute_vol_target_backtest
        result = compute_vol_target_backtest()
        assert result.base_allocation == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.00}

    def test_custom_allocation(self):
        from src.backtest.vol_targeting_backtest import compute_vol_target_backtest
        custom = {"SPY": 0.60, "GLD": 0.40, "TLT": 0.0, "IEF": 0.0}
        result = compute_vol_target_backtest(base_allocation=custom)
        assert result.base_allocation == custom


class TestVolTargetResultDataclass:
    """Tests for VolTargetResult dataclass fields."""

    def test_all_fields_present(self):
        from src.backtest.vol_targeting_backtest import compute_vol_target_backtest
        result = compute_vol_target_backtest(target_vol=0.11, max_leverage=1.5)
        assert hasattr(result, 'analysis_date')
        assert hasattr(result, 'base_allocation')
        assert hasattr(result, 'target_vol')
        assert hasattr(result, 'vol_lookback')
        assert hasattr(result, 'max_leverage')
        assert hasattr(result, 'static_sharpe')
        assert hasattr(result, 'vol_target_sharpe')
        assert hasattr(result, 'sharpe_delta')
        assert hasattr(result, 'static_cagr')
        assert hasattr(result, 'vol_target_cagr')
        assert hasattr(result, 'static_max_dd')
        assert hasattr(result, 'vol_target_max_dd')
        assert hasattr(result, 'mean_leverage')
        assert hasattr(result, 'max_leverage_reached')
        assert hasattr(result, 'leverage_above_1_pct')
        assert hasattr(result, 'summary')

    def test_sharpe_delta_consistent(self):
        from src.backtest.vol_targeting_backtest import compute_vol_target_backtest
        result = compute_vol_target_backtest(target_vol=0.11, max_leverage=1.5)
        expected_delta = round(result.vol_target_sharpe - result.static_sharpe, 4)
        assert result.sharpe_delta == expected_delta


class TestRegimeVolTargetResultDataclass:
    """Tests for RegimeVolTargetResult dataclass fields."""

    def test_regime_breakdown_structure(self):
        from src.backtest.vol_targeting_backtest import (
            compute_regime_conditional_vol_target_backtest,
        )
        result = compute_regime_conditional_vol_target_backtest(
            vol_lookback=63, max_leverage=1.5,
        )
        for regime, info in result.regime_breakdown.items():
            assert "days" in info
            assert "pct_of_time" in info
            assert "mean_leverage" in info
            assert "target_vol" in info
            assert "scaling_exponent" in info
            assert "vol_lookback" in info

    def test_regime_targets_stored(self):
        from src.backtest.vol_targeting_backtest import (
            compute_regime_conditional_vol_target_backtest, REGIME_VOL_TARGETS,
        )
        result = compute_regime_conditional_vol_target_backtest(
            vol_lookback=63, max_leverage=1.5,
        )
        for regime, target in REGIME_VOL_TARGETS.items():
            assert regime in result.regime_targets

    def test_regime_scaling_and_lookbacks_stored(self):
        from src.backtest.vol_targeting_backtest import (
            REGIME_VOL_LOOKBACKS,
            REGIME_VOL_SCALING_EXPONENTS,
            compute_regime_conditional_vol_target_backtest,
        )

        result = compute_regime_conditional_vol_target_backtest(
            vol_lookback=63, max_leverage=1.5,
        )

        assert result.regime_scaling_exponents == REGIME_VOL_SCALING_EXPONENTS
        assert result.regime_lookbacks == REGIME_VOL_LOOKBACKS
        for regime, info in result.regime_breakdown.items():
            assert info["scaling_exponent"] == REGIME_VOL_SCALING_EXPONENTS[regime]
            assert info["vol_lookback"] == REGIME_VOL_LOOKBACKS[regime]


class TestRegimeClassificationEdgeCases:
    """Additional edge case tests for _classify_regime_from_vol."""

    def test_boundary_crisis_high_vol(self):
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol
        # Exactly 1.7x median → CRISIS
        assert _classify_regime_from_vol(0.17, median_vol=0.10) == "CRISIS"

    def test_boundary_high_vol_normal(self):
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol
        # Exactly 1.25x median → HIGH_VOL
        assert _classify_regime_from_vol(0.125, median_vol=0.10) == "HIGH_VOL"

    def test_boundary_normal_low_vol(self):
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol
        # Just above 0.75x → NORMAL
        assert _classify_regime_from_vol(0.076, median_vol=0.10) == "NORMAL"
        # Just below 0.75x → LOW_VOL
        assert _classify_regime_from_vol(0.074, median_vol=0.10) == "LOW_VOL"

    def test_recovery_only_from_crisis_or_high_vol(self):
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol
        # From NORMAL with declining → stays NORMAL (not RECOVERY)
        regime = _classify_regime_from_vol(
            0.10, median_vol=0.10, prev_regime="NORMAL", vol_declining=True,
        )
        assert regime == "NORMAL"

    def test_recovery_from_high_vol(self):
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol
        regime = _classify_regime_from_vol(
            0.10, median_vol=0.10, prev_regime="HIGH_VOL", vol_declining=True,
        )
        assert regime == "RECOVERY"


# ── Edge-case tests ────────────────────────────────────────────────────────

class TestVolTargetLeverageEdgeCases:
    """Edge cases for _compute_vol_target_leverage."""

    def test_zero_vol_returns_one_with_default_smoothing(self):
        """realized_vol=0 should short-circuit to 1.0 regardless of smoothing."""
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        lev = _compute_vol_target_leverage(0.0, 0.10)
        assert lev == 1.0

    def test_negative_vol_returns_one(self):
        """Negative realized_vol (data artifact) should short-circuit to 1.0."""
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        lev = _compute_vol_target_leverage(-0.05, 0.10, smoothing=1.0)
        assert lev == 1.0

    def test_smoothing_zero_ignores_raw(self):
        """smoothing=0 should keep leverage at prev_leverage unchanged."""
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        lev = _compute_vol_target_leverage(0.05, 0.10, smoothing=0.0, prev_leverage=1.0)
        # raw = 0.10/0.05 = 2.0, but smoothing=0 => 0*2.0 + 1*1.0 = 1.0
        assert abs(lev - 1.0) < 0.01

    def test_smoothing_one_uses_raw_directly(self):
        """smoothing=1 should use the raw leverage directly (no prev blending)."""
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        # raw = 0.10/0.05 = 2.0, smoothing=1 => 1*2.0 + 0*1.5 = 2.0
        lev = _compute_vol_target_leverage(0.05, 0.10, smoothing=1.0, prev_leverage=1.5)
        assert abs(lev - 2.0) < 0.01

    def test_max_leverage_floor_enforced(self):
        """The 1/max_leverage floor should prevent extreme deleveraging."""
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        # raw = 0.10/0.30 = 0.333, floor = 1/2.0 = 0.5
        lev = _compute_vol_target_leverage(0.30, 0.10, max_leverage=2.0, smoothing=1.0)
        assert lev >= 0.5 - 0.001  # floor at 1/max_leverage

    def test_leverage_clamped_to_max(self):
        """Raw leverage exceeding max_leverage should be clamped."""
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        # raw = 0.10/0.01 = 10.0, max_leverage=1.5 => capped at 1.5
        lev = _compute_vol_target_leverage(0.01, 0.10, max_leverage=1.5, smoothing=1.0)
        assert abs(lev - 1.5) < 0.01

    def test_leverage_rounded_to_four_decimals(self):
        """Result should be rounded to 4 decimal places."""
        from src.backtest.vol_targeting_backtest import _compute_vol_target_leverage

        lev = _compute_vol_target_leverage(0.08, 0.10, smoothing=0.5, prev_leverage=1.0)
        # Verify rounding: no more than 4 decimal places
        assert lev == round(lev, 4)


class TestRegimeClassificationEdgeCasesExtra:
    """Edge cases for _classify_regime_from_vol boundaries and transitions."""

    def test_ratio_clearly_above_075x_is_normal(self):
        """ratio clearly above 0.75 (but below 1.25) should be NORMAL.

        Note: 0.075/0.10 loses precision in float64, so we test with values
        that produce ratio=0.76, safely above the 0.75 threshold.
        """
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol

        # 0.076/0.10 = 0.76 > 0.75 => NORMAL
        assert _classify_regime_from_vol(0.076, median_vol=0.10) == "NORMAL"

    def test_just_below_075x_is_low_vol(self):
        """ratio just under 0.75 should classify as LOW_VOL."""
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol

        assert _classify_regime_from_vol(0.0749, median_vol=0.10) == "LOW_VOL"

    def test_exactly_125x_is_high_vol(self):
        """ratio == 1.25 should be HIGH_VOL (>= 1.25 branch)."""
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol

        assert _classify_regime_from_vol(0.125, median_vol=0.10) == "HIGH_VOL"

    def test_exactly_17x_is_crisis(self):
        """ratio == 1.7 should be CRISIS (>= 1.7 branch)."""
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol

        assert _classify_regime_from_vol(0.17, median_vol=0.10) == "CRISIS"

    def test_recovery_not_triggered_without_vol_declining(self):
        """From CRISIS in normal range without vol_declining => NORMAL, not RECOVERY."""
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol

        regime = _classify_regime_from_vol(
            0.10, median_vol=0.10, prev_regime="CRISIS", vol_declining=False,
        )
        assert regime == "NORMAL"

    def test_recovery_not_from_low_vol(self):
        """RECOVERY transition should not fire from LOW_VOL even with vol_declining."""
        from src.backtest.vol_targeting_backtest import _classify_regime_from_vol

        regime = _classify_regime_from_vol(
            0.10, median_vol=0.10, prev_regime="LOW_VOL", vol_declining=True,
        )
        assert regime == "NORMAL"


class TestRegimeVolTargetsCoverage:
    """Verify REGIME_VOL_TARGETs has all 5 regimes with reasonable values."""

    def test_all_five_regimes_defined(self):
        from src.backtest.vol_targeting_backtest import REGIME_VOL_TARGETS

        expected = {"CRISIS", "HIGH_VOL", "NORMAL", "LOW_VOL", "RECOVERY"}
        assert set(REGIME_VOL_TARGETS.keys()) == expected

    def test_all_targets_positive(self):
        from src.backtest.vol_targeting_backtest import REGIME_VOL_TARGETS

        for regime, target in REGIME_VOL_TARGETS.items():
            assert target > 0, f"{regime} target should be positive"

    def test_crisis_has_lowest_target(self):
        from src.backtest.vol_targeting_backtest import REGIME_VOL_TARGETS

        assert REGIME_VOL_TARGETS["CRISIS"] <= REGIME_VOL_TARGETS["HIGH_VOL"]
        assert REGIME_VOL_TARGETS["CRISIS"] <= REGIME_VOL_TARGETS["NORMAL"]

    def test_low_vol_has_highest_target(self):
        from src.backtest.vol_targeting_backtest import REGIME_VOL_TARGETS

        assert REGIME_VOL_TARGETS["LOW_VOL"] >= REGIME_VOL_TARGETS["NORMAL"]
        assert REGIME_VOL_TARGETS["LOW_VOL"] >= REGIME_VOL_TARGETS["HIGH_VOL"]


class TestRegimeBacktestEmptyAndSingleDay:
    """Edge cases for compute_regime_conditional_vol_target_backtest."""

    def test_regime_backtest_uses_precomputed_realized_vols(self, monkeypatch):
        """Regime backtest should not recompute realized vol inside the daily loop."""
        import pandas as pd
        from src.backtest.vol_targeting_backtest import (
            compute_regime_conditional_vol_target_backtest,
        )

        dates = pd.date_range("2020-01-02", periods=12, freq="B")
        prices = pd.DataFrame(
            {
                "SPY": np.linspace(100.0, 111.0, len(dates)),
                "GLD": np.linspace(100.0, 105.0, len(dates)),
                "TLT": np.linspace(100.0, 98.0, len(dates)),
                "IEF": np.linspace(100.0, 101.0, len(dates)),
            },
            index=dates,
        )
        prices.index.name = "date"

        def fail_if_called(*args, **kwargs):
            raise AssertionError("daily loop should use precomputed realized vols")

        monkeypatch.setattr(
            "src.backtest.vol_targeting_backtest._load_prices",
            lambda: prices,
        )
        monkeypatch.setattr(
            "src.backtest.vol_targeting_backtest._load_vix_term_structure_data",
            lambda: {},
        )
        monkeypatch.setattr(
            "src.backtest.vol_targeting_backtest._compute_realized_vol",
            fail_if_called,
        )

        result = compute_regime_conditional_vol_target_backtest(
            vol_lookback=3,
            regime_lookbacks={
                "CRISIS": 5,
                "HIGH_VOL": 4,
                "NORMAL": 3,
                "LOW_VOL": 2,
                "RECOVERY": 3,
            },
            max_leverage=1.5,
        )

        assert np.isfinite(result.vol_target_sharpe)

    def test_single_day_prices(self, monkeypatch):
        """Single-day price data should produce a valid result (no crash)."""
        import json
        from unittest.mock import patch
        from src.backtest.vol_targeting_backtest import (
            compute_regime_conditional_vol_target_backtest,
        )

        # Single day of prices for each symbol
        single_day = {
            "SPY": [{"d": "2020-01-02", "p": 100.0}],
            "GLD": [{"d": "2020-01-02", "p": 100.0}],
            "TLT": [{"d": "2020-01-02", "p": 100.0}],
            "IEF": [{"d": "2020-01-02", "p": 100.0}],
        }

        monkeypatch.setattr(
            "src.backtest.vol_targeting_backtest._load_prices",
            lambda: __import__("pandas").DataFrame(
                {k: [e["p"] for e in v] for k, v in single_day.items()},
                index=__import__("pandas").to_datetime([e["d"] for e in single_day["SPY"]]),
            ),
        )

        result = compute_regime_conditional_vol_target_backtest(
            vol_lookback=63, max_leverage=1.5,
        )
        # Should not crash; static and vol-target Sharpe should both be finite
        assert np.isfinite(result.static_sharpe)
        assert np.isfinite(result.vol_target_sharpe)
        assert result.static_max_dd <= 0.0

    def test_two_day_prices(self, monkeypatch):
        """Two-day price data should produce valid result with one return."""
        import pandas as pd
        from src.backtest.vol_targeting_backtest import (
            compute_regime_conditional_vol_target_backtest,
        )

        dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
        two_day = pd.DataFrame(
            {
                "SPY": [100.0, 102.0],
                "GLD": [100.0, 101.0],
                "TLT": [100.0, 99.0],
                "IEF": [100.0, 100.5],
            },
            index=dates,
        )
        two_day.index.name = "date"

        monkeypatch.setattr(
            "src.backtest.vol_targeting_backtest._load_prices",
            lambda: two_day,
        )

        result = compute_regime_conditional_vol_target_backtest(
            vol_lookback=63, max_leverage=1.5,
        )
        assert np.isfinite(result.static_sharpe)
        assert np.isfinite(result.vol_target_sharpe)
        # With 2 days, static and vol-target should be the same (no vol estimation yet)
        assert result.static_sharpe == result.vol_target_sharpe
