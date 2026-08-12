"""Tests for src.utils — shared utility functions."""

import time


from src.utils import classify_vix_regime, signal_timeout


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


    def test_very_high_vix(self):
        """Extreme VIX values are still classified as crisis."""
        assert classify_vix_regime(80.0, "normal") == "crisis"
        assert classify_vix_regime(50.0, "low_vol") == "crisis"

    def test_very_low_vix(self):
        """Very low VIX values are low_vol."""
        assert classify_vix_regime(9.0, "normal") == "low_vol"
        assert classify_vix_regime(5.0, "recovery") == "low_vol"

    def test_recovery_trend_with_normal_vix(self):
        """VIX in normal range with recovery trend → recovery."""
        assert classify_vix_regime(18.0, "recovery") == "recovery"

    def test_high_vol_trend_with_normal_vix(self):
        """VIX in normal range with high_vol trend → high_vol."""
        assert classify_vix_regime(18.0, "high_vol") == "high_vol"

    def test_low_vol_trend_with_low_vix(self):
        """Low VIX with low_vol trend → low_vol."""
        assert classify_vix_regime(12.0, "low_vol") == "low_vol"

    def test_floating_point_boundary_crisis(self):
        """VIX=25.0001 is crisis (just above threshold)."""
        assert classify_vix_regime(25.0001, "normal") == "crisis"

    def test_floating_point_boundary_vol_spike(self):
        """VIX=20.0001 is vol_spike (just above threshold)."""
        assert classify_vix_regime(20.0001, "normal") == "vol_spike"

    def test_floating_point_boundary_low_vol(self):
        """VIX=14.9999 is low_vol (just below threshold)."""
        assert classify_vix_regime(14.9999, "normal") == "low_vol"


class TestSignalTimeout:
    """Tests for the signal_timeout decorator."""

    def test_fast_function_returns_result(self):
        """A function that completes within timeout returns its result."""
        @signal_timeout(default=None, seconds=5.0)
        def fast():
            return 42
        assert fast() == 42

    def test_slow_function_returns_default(self):
        """A function that exceeds timeout returns the default value."""
        @signal_timeout(default="fallback", seconds=0.1)
        def slow():
            time.sleep(5)
            return "never reached"
        assert slow() == "fallback"

    def test_exception_returns_default(self):
        """A function that raises returns the default value."""
        @signal_timeout(default=None, seconds=5.0)
        def broken():
            raise ValueError("boom")
        assert broken() is None

    def test_exception_with_custom_default(self):
        """Custom default is returned on exception."""
        @signal_timeout(default={"neutral": True}, seconds=5.0)
        def broken():
            raise RuntimeError("crash")
        assert broken() == {"neutral": True}

    def test_custom_signal_name_logged(self, caplog):
        """Custom signal_name appears in timeout log messages."""
        import logging
        @signal_timeout(default=None, seconds=0.1, signal_name="my_signal")
        def slow():
            time.sleep(5)
        with caplog.at_level(logging.WARNING, logger="src.utils"):
            slow()
        assert "my_signal" in caplog.text

    def test_default_signal_name_is_function_name(self, caplog):
        """When signal_name is omitted, function name is used in logs."""
        import logging
        @signal_timeout(default=None, seconds=0.1)
        def slow_signal():
            time.sleep(5)
        with caplog.at_level(logging.WARNING, logger="src.utils"):
            slow_signal()
        assert "slow_signal" in caplog.text

    def test_zero_timeout_returns_default(self):
        """Zero timeout means always return default."""
        @signal_timeout(default="instant", seconds=0.0)
        def anything():
            return "computed"
        # With 0s timeout, the thread may or may not complete before join
        # Either way the function should not raise
        result = anything()
        assert result in ("instant", "computed")

    def test_preserves_function_name(self):
        """Decorator preserves the wrapped function's name."""
        @signal_timeout(default=None, seconds=5.0)
        def my_signal_func():
            return 1
        assert my_signal_func.__name__ == "my_signal_func"

    def test_passes_args_and_kwargs(self):
        """Arguments are forwarded to the wrapped function."""
        @signal_timeout(default=None, seconds=5.0)
        def add(a, b, extra=0):
            return a + b + extra
        assert add(1, 2, extra=3) == 6
