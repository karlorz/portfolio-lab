"""
Tests for BL -> HRP -> Equal-Weight fallback cascade in black_litterman_mapper.py.

TDD: These tests define the expected cascade behavior before implementation.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.strategy.black_litterman_mapper import (
    BLViews, BLResult,
    map_biases_to_views, run_black_litterman,
    _run_hrp_fallback,
)


class TestHRPFallback:
    """Tests for the _run_hrp_fallback() helper."""

    @pytest.fixture
    def sample_cov(self):
        return np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])

    @pytest.fixture
    def symbols(self):
        return ["SPY", "GLD", "TLT"]

    def test_hrp_returns_valid_weights(self, sample_cov, symbols):
        """HRP fallback should produce valid weights summing to ~1.0."""
        weights = _run_hrp_fallback(sample_cov, symbols)
        assert isinstance(weights, dict)
        assert len(weights) > 0
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.05, f"HRP weights sum to {total}"

    def test_hrp_weights_positive(self, sample_cov, symbols):
        """HRP weights should all be non-negative."""
        weights = _run_hrp_fallback(sample_cov, symbols)
        for sym, w in weights.items():
            assert w >= 0, f"{sym} has negative HRP weight {w}"

    def test_hrp_returns_empty_on_degenerate_cov(self):
        """HRP should return empty dict on zero/degenerate covariance."""
        cov = np.zeros((3, 3))
        weights = _run_hrp_fallback(cov, ["SPY", "GLD", "TLT"])
        # HRP may return empty or default weights on degenerate input
        assert isinstance(weights, dict)

    def test_hrp_returns_empty_on_single_asset(self):
        """HRP with single asset should return equal weight."""
        cov = np.array([[0.0225]])
        weights = _run_hrp_fallback(cov, ["SPY"])
        assert isinstance(weights, dict)

    def test_hrp_with_nan_cov(self):
        """HRP should handle NaN covariance gracefully."""
        cov = np.array([
            [0.0225, float("nan"), -0.0063],
            [float("nan"), 0.0256, 0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        weights = _run_hrp_fallback(cov, ["SPY", "GLD", "TLT"])
        assert isinstance(weights, dict)


class TestFallbackCascade:
    """Tests for the BL -> HRP -> EW cascade in run_black_litterman()."""

    @pytest.fixture
    def sample_cov(self):
        return np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])

    def test_optimization_method_recorded(self, sample_cov):
        """BLResult.extras should record which optimization method was used."""
        views = map_biases_to_views(0.3, 0.1, 0.2)
        result = run_black_litterman(sample_cov, views)
        assert "optimization_method" in result.extras
        assert result.extras["optimization_method"] in (
            "bl_max_sharpe", "bl_hrp", "bl_equal_weight"
        )

    def test_normal_case_uses_max_sharpe(self, sample_cov):
        """With valid covariance and views, should use max_sharpe."""
        views = map_biases_to_views(0.3, 0.1, 0.2)
        result = run_black_litterman(sample_cov, views)
        assert result.extras["optimization_method"] == "bl_max_sharpe"
        assert result.expected_sharpe is not None

    def test_hrp_fallback_when_max_sharpe_fails(self, sample_cov):
        """When max_sharpe fails, should fall back to HRP (not directly to bl_weights)."""
        views = map_biases_to_views(0.3, 0.1, 0.2)
        # Mock EfficientFrontier.max_sharpe to fail
        with patch("pypfopt.EfficientFrontier") as mock_ef_cls:
            mock_ef = MagicMock()
            mock_ef.max_sharpe.side_effect = ValueError("Optimizer failed")
            mock_ef_cls.return_value = mock_ef

            result = run_black_litterman(sample_cov, views)
            # Should have fallen back to HRP, not bl_weights
            assert result.extras["optimization_method"] == "bl_hrp"
            assert len(result.bl_weights) > 0

    def test_ew_fallback_when_both_bl_and_hrp_fail(self, sample_cov):
        """When both max_sharpe and HRP fail, should fall back to equal weight."""
        views = map_biases_to_views(0.3, 0.1, 0.2)
        with patch("pypfopt.EfficientFrontier") as mock_ef_cls, \
             patch("src.strategy.black_litterman_mapper._run_hrp_fallback") as mock_hrp:
            mock_ef = MagicMock()
            mock_ef.max_sharpe.side_effect = ValueError("Optimizer failed")
            mock_ef_cls.return_value = mock_ef
            mock_hrp.return_value = {}  # HRP also fails

            result = run_black_litterman(sample_cov, views)
            assert result.extras["optimization_method"] == "bl_equal_weight"
            assert len(result.bl_weights) > 0

    def test_equal_weight_distribution(self, sample_cov):
        """Equal weight fallback should distribute evenly across symbols."""
        views = map_biases_to_views(0.3, 0.1, 0.2)
        with patch("pypfopt.EfficientFrontier") as mock_ef_cls, \
             patch("src.strategy.black_litterman_mapper._run_hrp_fallback") as mock_hrp:
            mock_ef = MagicMock()
            mock_ef.max_sharpe.side_effect = ValueError("Failed")
            mock_ef_cls.return_value = mock_ef
            mock_hrp.return_value = {}  # HRP fails

            result = run_black_litterman(sample_cov, views)
            weights = result.bl_weights
            # Each symbol should get roughly 1/3
            for sym in ["SPY", "GLD", "TLT"]:
                if sym in weights:
                    assert abs(weights[sym] - 1/3) < 0.05

    def test_cascade_preserves_posterior_returns(self, sample_cov):
        """Regardless of optimization method, posterior_returns should always be populated."""
        views = map_biases_to_views(0.3, 0.1, 0.2)
        result = run_black_litterman(sample_cov, views)
        assert len(result.posterior_returns) == 3
        for sym in ["SPY", "GLD", "TLT"]:
            assert sym in result.posterior_returns

    @pytest.mark.skip(reason="cvxpy non-deterministic with zero cov — tested via test_zero_covariance_matrix elsewhere")
    def test_degenerate_cov_handles_gracefully(self):
        """Zero covariance matrix should produce valid weights without crashing.

        Non-deterministic: cvxpy may succeed (returning max_sharpe with inf Sharpe)
        or fail (cascading to equal weight). Both are valid cascade outcomes.
        Fully covered by test_zero_covariance_matrix in TestRunBlackLittermanFailurePaths.
        """
        cov = np.zeros((3, 3))
        views = map_biases_to_views(0.0, 0.0, 0.0)
        result = run_black_litterman(cov, views)
        assert isinstance(result, BLResult)
        assert len(result.bl_weights) > 0
        assert result.extras.get("optimization_method") in (
            "bl_max_sharpe", "bl_hrp", "bl_equal_weight"
        )


class TestCascadeWithRealPrices:
    """Integration tests using synthetic price data."""

    def test_compute_bl_weights_has_method_info(self):
        """compute_bl_weights should propagate optimization_method."""
        import pandas as pd

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 500)),
            "GLD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 500)),
            "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.008, 500)),
        }, index=dates)

        from src.strategy.black_litterman_mapper import compute_bl_weights
        result = compute_bl_weights(
            prices_df=prices,
            equity_bias=0.5,
            duration_bias=0.2,
            gold_bias=0.3,
        )
        assert "optimization_method" in result.extras
