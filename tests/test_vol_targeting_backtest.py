#!/usr/bin/env python3
"""Tests for src/backtest/vol_targeting_backtest.py."""

import numpy as np
import pytest


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
