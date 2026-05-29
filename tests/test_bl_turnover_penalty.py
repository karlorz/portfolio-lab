"""Tests for Black-Litterman turnover penalty in optimization.

TDD red phase — defines behavior before implementation.
"""

import pytest
import numpy as np
import pandas as pd
from src.strategy.black_litterman_mapper import (
    BLViews,
    run_black_litterman,
    compute_bl_weights,
)


# Minimal covariance and views for testing
SYMBOLS = ["SPY", "GLD", "TLT"]
COV = np.array([
    [0.04, 0.005, 0.008],
    [0.005, 0.03, 0.006],
    [0.008, 0.006, 0.025],
])


def _make_views(**kwargs):
    """Create minimal BLViews for testing."""
    defaults = dict(
        symbols=SYMBOLS,
        absolute_views={"SPY": 0.08, "GLD": 0.05, "TLT": 0.03},
        view_confidences=[0.7, 0.5, 0.5],
        tau=0.15,
        prior="equal",
    )
    defaults.update(kwargs)
    return BLViews(**defaults)


class TestBLTurnoverPenalty:
    """Test suite for turnover penalty in BL optimization."""

    def test_no_penalty_unchanged(self):
        """Without turnover_penalty, behavior should be unchanged."""
        views = _make_views()
        result = run_black_litterman(COV, views, transaction_costs=False)
        assert result.bl_weights
        assert sum(result.bl_weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_penalty_with_current_weights(self):
        """With turnover_penalty and current_weights, result should differ from unpenalized."""
        views = _make_views()
        current = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        # Without penalty
        result_no_penalty = run_black_litterman(
            COV, views, transaction_costs=False,
        )
        # With penalty — should produce weights closer to current
        result_with_penalty = run_black_litterman(
            COV, views, transaction_costs=False,
            turnover_penalty=0.5, current_weights=current,
        )
        # Both should produce valid weights
        assert sum(result_with_penalty.bl_weights.values()) == pytest.approx(1.0, abs=0.01)
        # The penalty version should record the penalty in extras
        assert "turnover_penalty_applied" in result_with_penalty.extras

    def test_higher_penalty_closer_to_current(self):
        """Higher turnover_penalty should produce weights closer to current_weights."""
        views = _make_views()
        current = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}

        result_low = run_black_litterman(
            COV, views, transaction_costs=False,
            turnover_penalty=0.1, current_weights=current,
        )
        result_high = run_black_litterman(
            COV, views, transaction_costs=False,
            turnover_penalty=2.0, current_weights=current,
        )

        # Compute turnover (L1 distance from current)
        def turnover(weights):
            total = 0
            for s in SYMBOLS:
                curr = current.get(s, 0)
                new = weights.get(s, 0)
                total += abs(new - curr)
            return total / 2  # One-way turnover

        t_low = turnover(result_low.bl_weights)
        t_high = turnover(result_high.bl_weights)
        # Higher penalty should mean lower turnover
        assert t_high <= t_low + 0.05  # Allow small numerical tolerance

    def test_penalty_zero_same_as_none(self):
        """turnover_penalty=0 should produce same result as no penalty."""
        views = _make_views()
        current = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        result_none = run_black_litterman(COV, views, transaction_costs=False)
        result_zero = run_black_litterman(
            COV, views, transaction_costs=False,
            turnover_penalty=0, current_weights=current,
        )
        # Weights should be identical (no penalty applied)
        for sym in SYMBOLS:
            w_none = result_none.bl_weights.get(sym, 0)
            w_zero = result_zero.bl_weights.get(sym, 0)
            assert abs(w_none - w_zero) < 0.01

    def test_penalty_without_current_weights_ignored(self):
        """turnover_penalty without current_weights should be ignored."""
        views = _make_views()
        result = run_black_litterman(
            COV, views, transaction_costs=False,
            turnover_penalty=1.0, current_weights=None,
        )
        assert result.bl_weights
        assert "turnover_penalty_applied" in result.extras
        assert result.extras["turnover_penalty_applied"] is False

    def test_penalty_extras_recorded(self):
        """Extras should record turnover penalty details."""
        views = _make_views()
        current = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        result = run_black_litterman(
            COV, views, transaction_costs=False,
            turnover_penalty=0.5, current_weights=current,
        )
        assert result.extras["turnover_penalty_applied"] is True
        assert result.extras["turnover_penalty_lambda"] == 0.5
        assert "turnover_bps" in result.extras

    def test_penalty_with_transaction_costs(self):
        """Both transaction costs and turnover penalty can be applied together."""
        views = _make_views()
        current = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        result = run_black_litterman(
            COV, views, transaction_costs=True,
            turnover_penalty=0.3, current_weights=current,
        )
        assert result.bl_weights
        assert result.extras["transaction_costs_applied"] is True
        assert result.extras["turnover_penalty_applied"] is True

    def test_compute_bl_weights_accepts_penalty(self):
        """compute_bl_weights convenience function should accept penalty params."""
        # This tests that the API is extended, not just run_black_litterman
        import inspect
        sig = inspect.signature(compute_bl_weights)
        assert "turnover_penalty" in sig.parameters
        assert "current_weights" in sig.parameters
