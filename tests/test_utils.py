"""Tests for src.utils — shared utility functions."""

import pytest
from src.utils import classify_vix_regime


class TestClassifyVixRegime:
    """Tests for the shared VIX regime classifier."""

    def test_none_vix_returns_trend(self):
        """When VIX is None, return trend_regime."""
        assert classify_vix_regime(None, "crisis") == "crisis"
        assert classify_vix_regime(None, "normal") == "normal"
        assert classify_vix_regime(None) == "normal"

    def test_crisis_vix(self):
        """VIX > 25 returns crisis."""
        assert classify_vix_regime(30.0, "normal") == "crisis"
        assert classify_vix_regime(25.01, "normal") == "crisis"

    def test_vol_spike_vix(self):
        """VIX between 20-25 returns vol_spike."""
        assert classify_vix_regime(22.5, "normal") == "vol_spike"
        assert classify_vix_regime(20.01, "normal") == "vol_spike"

    def test_low_vol_vix(self):
        """VIX < 15 with non-crisis trend returns low_vol."""
        assert classify_vix_regime(12.0, "normal") == "low_vol"
        assert classify_vix_regime(14.99, "normal") == "low_vol"

    def test_normal_vix(self):
        """VIX between 15-20 returns trend_regime (which is 'normal' by default)."""
        assert classify_vix_regime(18.0, "normal") == "normal"

    def test_crisis_overrides_any_trend(self):
        """VIX crisis always wins."""
        assert classify_vix_regime(30.0, "low_vol") == "crisis"
        assert classify_vix_regime(30.0, "recovery") == "crisis"

    def test_vol_spike_overrides_any_trend(self):
        """VIX vol_spike always wins."""
        assert classify_vix_regime(22.0, "low_vol") == "vol_spike"

    def test_low_vol_with_crisis_trend_returns_crisis(self):
        """Low VIX but crisis trend → crisis (don't ignore risk)."""
        assert classify_vix_regime(12.0, "crisis") == "crisis"

    def test_low_vol_with_non_crisis_trend(self):
        """Low VIX + non-crisis trend → low_vol."""
        assert classify_vix_regime(12.0, "normal") == "low_vol"
        assert classify_vix_regime(12.0, "recovery") == "low_vol"

    def test_boundary_vix_25_exactly(self):
        """VIX=25.0 is vol_spike (25 > 25 is False)."""
        assert classify_vix_regime(25.0, "normal") == "vol_spike"

    def test_boundary_vix_20_exactly(self):
        """VIX=20.0 is normal (20 > 20 is False, 20 < 15 is False)."""
        assert classify_vix_regime(20.0, "normal") == "normal"

    def test_boundary_vix_15_exactly(self):
        """VIX=15.0 is normal (15 < 15 is False)."""
        assert classify_vix_regime(15.0, "normal") == "normal"

    def test_normal_vix_with_crisis_trend_returns_crisis(self):
        """VIX in normal range but trend is crisis → trend wins."""
        assert classify_vix_regime(18.0, "crisis") == "crisis"

    def test_vol_spike_with_crisis_trend_returns_vol_spike(self):
        """VIX vol_spike overrides even crisis trend."""
        assert classify_vix_regime(22.0, "crisis") == "vol_spike"
