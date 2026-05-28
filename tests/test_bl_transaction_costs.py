"""
Tests for transaction cost adjustment in Black-Litterman optimizer.

When transaction_costs=True, round-trip costs are subtracted from
posterior expected returns before optimization, penalizing high-cost
assets and reducing churn-heavy rebalances.
"""
import pytest
import numpy as np

from src.strategy.black_litterman_mapper import (
    BLViews, BLResult,
    map_biases_to_views, run_black_litterman,
)


class TestTransactionCostAdjustment:
    """Tests for transaction_costs parameter in run_black_litterman()."""

    @pytest.fixture
    def sample_cov(self):
        return np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])

    def test_costs_default_enabled(self, sample_cov):
        """transaction_costs should default to True."""
        views = map_biases_to_views(0.3, 0.1, 0.2)
        result = run_black_litterman(sample_cov, views)
        assert result.extras.get("transaction_costs_applied") is True

    def test_costs_disabled(self, sample_cov):
        """With transaction_costs=False, no cost adjustment."""
        views = map_biases_to_views(0.3, 0.1, 0.2)
        result = run_black_litterman(sample_cov, views, transaction_costs=False)
        assert result.extras.get("transaction_costs_applied") is False

    def test_costs_reduce_posterior_returns(self, sample_cov):
        """Cost-adjusted posterior returns should be lower than raw."""
        views = map_biases_to_views(0.3, 0.1, 0.2)
        result_with = run_black_litterman(sample_cov, views, transaction_costs=True)
        result_without = run_black_litterman(sample_cov, views, transaction_costs=False)

        # Total posterior returns with costs should be <= without costs
        total_with = sum(result_with.posterior_returns.values())
        total_without = sum(result_without.posterior_returns.values())
        assert total_with <= total_without

    def test_costs_recorded_in_extras(self, sample_cov):
        """Extras should contain cost penalty information."""
        views = map_biases_to_views(0.3, 0.1, 0.2)
        result = run_black_litterman(sample_cov, views)
        assert "cost_penalties_bps" in result.extras
        # Should have penalties for each symbol
        penalties = result.extras["cost_penalties_bps"]
        assert isinstance(penalties, dict)
        assert len(penalties) == 3

    def test_spy_has_lowest_cost(self, sample_cov):
        """SPY should have the lowest cost penalty (most liquid)."""
        views = map_biases_to_views(0.3, 0.1, 0.2)
        result = run_black_litterman(sample_cov, views)
        penalties = result.extras["cost_penalties_bps"]
        assert penalties["SPY"] < penalties["TLT"]
        assert penalties["SPY"] < penalties["GLD"]

    def test_regime_adjusts_costs(self, sample_cov):
        """Crisis regime should increase cost penalties."""
        views = map_biases_to_views(0.3, 0.1, 0.2)
        result_normal = run_black_litterman(sample_cov, views, regime="normal")
        result_crisis = run_black_litterman(sample_cov, views, regime="crisis")

        normal_total = sum(result_normal.extras["cost_penalties_bps"].values())
        crisis_total = sum(result_crisis.extras["cost_penalties_bps"].values())
        assert crisis_total > normal_total

    def test_zero_views_with_costs_still_valid(self, sample_cov):
        """Zero views with costs should still produce valid weights."""
        views = map_biases_to_views(0.0, 0.0, 0.0)
        result = run_black_litterman(sample_cov, views, transaction_costs=True)
        # Check type by class name to avoid import-order isinstance issues
        assert type(result).__name__ == "BLResult"
        assert len(result.bl_weights) > 0

    def test_cost_adjustment_doesnt_break_cascade(self, sample_cov):
        """Cost adjustment should work through the HRP/EW fallback cascade."""
        views = map_biases_to_views(0.3, 0.1, 0.2)
        result = run_black_litterman(sample_cov, views)
        assert result.extras["optimization_method"] in (
            "bl_max_sharpe", "bl_hrp", "bl_equal_weight"
        )
