#!/usr/bin/env python3
"""Tests for VIX term structure integration with volatility targeting.

Coverage: _load_vix_term_structure_data (3), _get_vix_regime_for_date (4),
VIX_REGIME_VOL_BIAS constants (5), VIX-enhanced regime classification (6).
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock



class TestVIXDataLoading:
    """Tests for VIX term structure data loading."""

    def test_load_vix_data_empty_when_missing(self):
        from src.backtest.vol_targeting_backtest import _load_vix_term_structure_data
        
        with patch('src.backtest.vol_targeting_backtest.DATA_DIR') as mock_dir:
            mock_dir.__truediv__ = MagicMock(return_value=Path('/nonexistent/vix_term_structure.json'))
            data = _load_vix_term_structure_data()
            assert data == {}

    def test_load_vix_data_valid_json(self, tmp_path):
        from src.backtest.vol_targeting_backtest import _load_vix_term_structure_data
        
        vix_data = {
            "2024-01-05": {"vix_spot": 13.35, "regime": "backwardation"},
            "2024-01-06": {"vix_spot": 14.00, "regime": "contango"},
        }
        vix_file = tmp_path / 'vix_term_structure.json'
        with open(vix_file, 'w') as f:
            json.dump(vix_data, f)
        
        with patch('src.backtest.vol_targeting_backtest.DATA_DIR', tmp_path):
            data = _load_vix_term_structure_data()
            assert len(data) == 2
            assert "2024-01-05" in data

    def test_load_vix_data_invalid_json(self, tmp_path):
        from src.backtest.vol_targeting_backtest import _load_vix_term_structure_data
        
        vix_file = tmp_path / 'vix_term_structure.json'
        with open(vix_file, 'w') as f:
            f.write("invalid json {{{")
        
        with patch('src.backtest.vol_targeting_backtest.DATA_DIR', tmp_path):
            data = _load_vix_term_structure_data()
            assert data == {}


class TestVIXRegimeLookup:
    """Tests for VIX regime date lookup."""

    def test_exact_date_match(self):
        from src.backtest.vol_targeting_backtest import _get_vix_regime_for_date
        
        vix_data = {
            "2024-01-05": {"regime": "backwardation"},
            "2024-01-06": {"regime": "contango"},
        }
        regime = _get_vix_regime_for_date("2024-01-05", vix_data)
        assert regime == "backwardation"

    def test_closest_date_fallback(self):
        from src.backtest.vol_targeting_backtest import _get_vix_regime_for_date
        
        vix_data = {
            "2024-01-05": {"regime": "backwardation"},
            "2024-01-10": {"regime": "contango"},
        }
        # 2024-01-07 not in data, should use 2024-01-05
        regime = _get_vix_regime_for_date("2024-01-07", vix_data)
        assert regime == "backwardation"

    def test_empty_vix_data_returns_none(self):
        from src.backtest.vol_targeting_backtest import _get_vix_regime_for_date
        
        regime = _get_vix_regime_for_date("2024-01-05", {})
        assert regime is None

    def test_date_before_data_range(self):
        from src.backtest.vol_targeting_backtest import _get_vix_regime_for_date
        
        vix_data = {
            "2024-01-05": {"regime": "backwardation"},
        }
        regime = _get_vix_regime_for_date("2023-12-01", vix_data)
        assert regime is None

    def test_precomputed_dates_avoid_repeated_key_scans(self):
        from src.backtest.vol_targeting_backtest import _get_vix_regime_for_date

        class CountingVixData(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.keys_calls = 0

            def keys(self):
                self.keys_calls += 1
                return super().keys()

        vix_data = CountingVixData({
            "2024-01-02": {"regime": "backwardation"},
            "2024-01-05": {"regime": "flat"},
            "2024-01-10": {"regime": "contango"},
        })
        sorted_dates = sorted(vix_data.keys())
        vix_data.keys_calls = 0

        assert (
            _get_vix_regime_for_date("2024-01-04", vix_data, sorted_dates)
            == "backwardation"
        )
        assert (
            _get_vix_regime_for_date("2024-01-07", vix_data, sorted_dates)
            == "flat"
        )
        assert (
            _get_vix_regime_for_date("2024-01-11", vix_data, sorted_dates)
            == "contango"
        )
        assert vix_data.keys_calls == 0


class TestVIXRegimeVolBias:
    """Tests for VIX regime vol bias constants."""

    def test_extreme_contango_positive_bias(self):
        from src.backtest.vol_targeting_backtest import VIX_REGIME_VOL_BIAS
        assert VIX_REGIME_VOL_BIAS["extreme_contango"] == 0.02

    def test_contango_mild_positive_bias(self):
        from src.backtest.vol_targeting_backtest import VIX_REGIME_VOL_BIAS
        assert VIX_REGIME_VOL_BIAS["contango"] == 0.01

    def test_flat_neutral(self):
        from src.backtest.vol_targeting_backtest import VIX_REGIME_VOL_BIAS
        assert VIX_REGIME_VOL_BIAS["flat"] == 0.0

    def test_backwardation_negative_bias(self):
        from src.backtest.vol_targeting_backtest import VIX_REGIME_VOL_BIAS
        assert VIX_REGIME_VOL_BIAS["backwardation"] == -0.01

    def test_extreme_backwardation_strong_negative(self):
        from src.backtest.vol_targeting_backtest import VIX_REGIME_VOL_BIAS
        assert VIX_REGIME_VOL_BIAS["extreme_backwardation"] == -0.03

    def test_bias_direction_correct(self):
        from src.backtest.vol_targeting_backtest import VIX_REGIME_VOL_BIAS
        # Complacency should increase risk, crisis should decrease
        assert VIX_REGIME_VOL_BIAS["extreme_contango"] > VIX_REGIME_VOL_BIAS["extreme_backwardation"]


class TestVIXEnhancedRegimeClassification:
    """Tests for VIX-enhanced regime classification in backtest."""

    def test_vix_term_structure_import(self):
        """Verify VIXTermStructureSignalGenerator is imported."""
        from src.backtest.vol_targeting_backtest import VIXTermStructureSignalGenerator
        assert VIXTermStructureSignalGenerator is not None

    def test_vix_regime_enum_import(self):
        """Verify VIXRegime enum is imported."""
        from src.backtest.vol_targeting_backtest import VIXRegime
        assert hasattr(VIXRegime, 'EXTREME_CONTANGO')
        assert hasattr(VIXRegime, 'BACKWARDATION')

    def test_vix_regime_bias_in_range(self):
        """All VIX regime biases should be reasonable adjustments."""
        from src.backtest.vol_targeting_backtest import VIX_REGIME_VOL_BIAS
        for regime, bias in VIX_REGIME_VOL_BIAS.items():
            assert -0.05 <= bias <= 0.05, f"Bias for {regime} out of range: {bias}"

    def test_target_vol_bounds(self):
        """Target vol should be bounded after VIX adjustment."""
        from src.backtest.vol_targeting_backtest import VIX_REGIME_VOL_BIAS
        
        # Simulate: base target 0.09, extreme backwardation
        base_target = 0.09
        bias = VIX_REGIME_VOL_BIAS["extreme_backwardation"]  # -0.03
        adjusted = max(0.03, min(0.15, base_target + bias))
        assert adjusted == 0.06  # 0.09 - 0.03
        assert 0.03 <= adjusted <= 0.15

    def test_target_vol_floor_ceiling(self):
        """VIX adjustment should respect floor/ceiling."""
        from src.backtest.vol_targeting_backtest import VIX_REGIME_VOL_BIAS
        
        # Floor test: very low base + negative bias
        base_target = 0.04
        bias = VIX_REGIME_VOL_BIAS["extreme_backwardation"]  # -0.03
        adjusted = max(0.03, min(0.15, base_target + bias))
        assert adjusted == 0.03  # Floor at 0.03

        # Ceiling test: high base + positive bias
        base_target = 0.14
        bias = VIX_REGIME_VOL_BIAS["extreme_contango"]  # +0.02
        adjusted = max(0.03, min(0.15, base_target + bias))
        assert adjusted == 0.15  # Ceiling at 0.15


class TestVIXEnhancedBacktest:
    """Integration tests for VIX-enhanced regime-conditional vol targeting."""

    def test_backtest_runs_with_vix_data(self):
        """Backtest should complete with VIX data available."""
        from src.backtest.vol_targeting_backtest import compute_regime_conditional_vol_target_backtest
        
        # This should not raise even if VIX data is missing
        result = compute_regime_conditional_vol_target_backtest(
            save=False,
            vol_lookback=63,
            max_leverage=1.5,
        )
        assert result is not None
        assert result.static_sharpe > 0
        assert result.vol_target_sharpe > 0

    def test_backtest_result_has_expected_fields(self):
        """RegimeVolTargetResult should have all expected fields."""
        from src.backtest.vol_targeting_backtest import compute_regime_conditional_vol_target_backtest
        
        result = compute_regime_conditional_vol_target_backtest(save=False)
        assert hasattr(result, 'static_sharpe')
        assert hasattr(result, 'vol_target_sharpe')
        assert hasattr(result, 'sharpe_delta')
        assert hasattr(result, 'regime_breakdown')

    def test_vix_enhanced_does_not_break_backward_compat(self):
        """VIX integration should not break existing regime classification."""
        from src.backtest.vol_targeting_backtest import (
            _classify_regime_from_vol,
            REGIME_VOL_TARGETS,
        )
        
        # Existing regime classification should still work
        regime = _classify_regime_from_vol(0.20, median_vol=0.10)
        assert regime == "CRISIS"
        
        # Regime targets should still be accessible
        assert "CRISIS" in REGIME_VOL_TARGETS
        assert REGIME_VOL_TARGETS["CRISIS"] == 0.03
