"""Tests for online IC-based ensemble weight learning.

TDD red phase — defines behavior before implementation.
Non-ML alternative to the XGBoost stacking integrator.
"""

import pytest
import math
import numpy as np
from src.strategy.online_ic_weighter import OnlineICWeighter


class TestOnlineICWeighter:
    """Test suite for OnlineICWeighter."""

    def test_basic_construction(self):
        """Should initialize with default parameters."""
        weighter = OnlineICWeighter()
        assert weighter.half_life > 0
        assert weighter.min_weight >= 0
        assert weighter.max_weight <= 1.0

    def test_update_and_get_weights(self):
        """update() should accept IC values and get_weights() returns normalized weights."""
        weighter = OnlineICWeighter()
        weighter.update({"signal_a": 0.05, "signal_b": 0.02, "signal_c": -0.01})
        weights = weighter.get_weights()
        assert set(weights.keys()) == {"signal_a", "signal_b", "signal_c"}
        # Weights should sum to 1.0
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
        # Higher IC → higher weight
        assert weights["signal_a"] > weights["signal_b"] > weights["signal_c"]

    def test_softmax_conversion(self):
        """IC values should be converted to weights via softmax."""
        weighter = OnlineICWeighter()
        weighter.update({"a": 0.10, "b": 0.05, "c": 0.00})
        weights = weighter.get_weights()
        # Softmax of [0.10, 0.05, 0.00] scaled by temperature
        assert weights["a"] > weights["b"] > weights["c"]
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_exponential_moving_average(self):
        """Successive updates should use exponential moving average."""
        weighter = OnlineICWeighter(half_life=5)
        # First update: signal_a is good
        weighter.update({"signal_a": 0.10, "signal_b": 0.00})
        w1 = weighter.get_weights()
        assert w1["signal_a"] > w1["signal_b"]

        # Second update: signal_b becomes good, signal_a drops
        weighter.update({"signal_a": 0.00, "signal_b": 0.10})
        w2 = weighter.get_weights()

        # signal_b should have gained weight, but EMA means signal_a still
        # has residual weight from first update
        assert w2["signal_b"] > w1["signal_b"]

    def test_half_life_parameter(self):
        """Smaller half_life should give more weight to recent observations."""
        weighter_fast = OnlineICWeighter(half_life=3)
        weighter_slow = OnlineICWeighter(half_life=20)

        # Both start with same good signal
        for w in [weighter_fast, weighter_slow]:
            w.update({"a": 0.10, "b": 0.00})

        # Then signal flips
        for w in [weighter_fast, weighter_slow]:
            w.update({"a": 0.00, "b": 0.10})

        w_fast = weighter_fast.get_weights()
        w_slow = weighter_slow.get_weights()

        # Fast adapter should have shifted more toward signal_b
        assert w_fast["b"] > w_slow["b"]

    def test_min_weight_floor(self):
        """No signal should get zero weight (exploration floor)."""
        weighter = OnlineICWeighter(min_weight=0.05)
        weighter.update({"a": 0.20, "b": -0.20, "c": 0.00})
        weights = weighter.get_weights()
        for w in weights.values():
            assert w >= 0.05

    def test_max_weight_cap(self):
        """No signal should exceed max_weight."""
        weighter = OnlineICWeighter(max_weight=0.50)
        # Only one signal with positive IC
        weighter.update({"a": 0.20, "b": -0.20, "c": -0.20, "d": -0.20})
        weights = weighter.get_weights()
        assert weights["a"] <= 0.50

    def test_empty_update_raises(self):
        """Empty IC dict should raise ValueError."""
        weighter = OnlineICWeighter()
        with pytest.raises(ValueError, match="empty"):
            weighter.update({})

    def test_single_signal(self):
        """Single signal should get weight 1.0."""
        weighter = OnlineICWeighter()
        weighter.update({"only_signal": 0.05})
        weights = weighter.get_weights()
        assert weights["only_signal"] == pytest.approx(1.0, abs=1e-6)

    def test_blend_with_static_weights(self):
        """blend_with_static() should mix online weights with static regime weights."""
        weighter = OnlineICWeighter(blend_alpha=0.5)
        weighter.update({"a": 0.10, "b": 0.05})
        static = {"a": 0.60, "b": 0.40}
        blended = weighter.blend_with_static(static)
        # With alpha=0.5, blended = 0.5*online + 0.5*static
        assert sum(blended.values()) == pytest.approx(1.0, abs=1e-6)
        # Should be between online and static
        online = weighter.get_weights()
        assert min(online["a"], static["a"]) <= blended["a"] <= max(online["a"], static["a"])

    def test_blend_preserves_signal_set(self):
        """blend_with_static should handle signal set differences."""
        weighter = OnlineICWeighter(blend_alpha=0.5)
        weighter.update({"a": 0.10, "b": 0.05, "c": 0.02})
        static = {"a": 0.50, "b": 0.50}  # c not in static
        blended = weighter.blend_with_static(static)
        # All signals from both sets should be present
        assert "a" in blended
        assert "b" in blended
        assert "c" in blended
        assert sum(blended.values()) == pytest.approx(1.0, abs=1e-6)

    def test_temperature_parameter(self):
        """Higher temperature should make weights more uniform."""
        weighter_cold = OnlineICWeighter(temperature=0.01)
        weighter_hot = OnlineICWeighter(temperature=10.0)
        ic_values = {"a": 0.10, "b": 0.05, "c": 0.00}
        weighter_cold.update(ic_values)
        weighter_hot.update(ic_values)
        w_cold = weighter_cold.get_weights()
        w_hot = weighter_hot.get_weights()
        # Cold temperature: more concentrated (higher max weight)
        assert max(w_cold.values()) > max(w_hot.values())

    def test_state_persistence(self):
        """save_state/load_state should preserve EMA values."""
        weighter = OnlineICWeighter(half_life=10)
        weighter.update({"a": 0.10, "b": 0.05})
        state = weighter.get_state()
        assert "ema_values" in state
        assert "update_count" in state

        # Restore in new weighter
        weighter2 = OnlineICWeighter(half_life=10)
        weighter2.load_state(state)
        assert weighter2.get_weights() == weighter.get_weights()

    def test_ic_trend_adjustment(self):
        """Decaying IC signals should get penalized weights."""
        weighter = OnlineICWeighter(trend_penalty=0.5)
        # Build up history
        for _ in range(10):
            weighter.update({"a": 0.10, "b": 0.05})
        # Now report trend info
        weighter.update_trends({"a": "decaying", "b": "stable"})
        weights = weighter.get_weights()
        # signal_a should be penalized for decaying trend
        basic = OnlineICWeighter()
        for _ in range(10):
            basic.update({"a": 0.10, "b": 0.05})
        basic_weights = basic.get_weights()
        assert weights["a"] < basic_weights["a"]

    def test_negative_ic_penalty(self):
        """Signals with negative IC should get minimum weight."""
        weighter = OnlineICWeighter(min_weight=0.02)
        weighter.update({"good": 0.10, "bad": -0.05, "neutral": 0.00})
        weights = weighter.get_weights()
        assert weights["good"] > weights["neutral"] > weights["bad"]
        assert weights["bad"] >= 0.02

    def test_many_signals(self):
        """Should handle 8+ signals (current ensemble size)."""
        weighter = OnlineICWeighter()
        ic_values = {
            "alt_data": 0.08,
            "intl_mom": 0.06,
            "cross_rv": 0.04,
            "regime_arb": 0.03,
            "unified": 0.05,
            "mtf": 0.02,
            "google_trends": 0.01,
            "msm": -0.01,
        }
        weighter.update(ic_values)
        weights = weighter.get_weights()
        assert len(weights) == 8
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
        # alt_data has highest IC → highest weight
        assert weights["alt_data"] == max(weights.values())

    def test_get_weight_vector(self):
        """get_weight_vector() should return ordered array matching signal order."""
        weighter = OnlineICWeighter()
        weighter.update({"a": 0.10, "b": 0.05, "c": 0.02})
        vec, names = weighter.get_weight_vector()
        assert len(vec) == 3
        assert len(names) == 3
        assert sum(vec) == pytest.approx(1.0, abs=1e-6)
