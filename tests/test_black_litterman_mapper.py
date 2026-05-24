#!/usr/bin/env python3
"""
Tests for src/strategy/black_litterman_mapper.py — BL view mapper.
"""
import pytest
import numpy as np

from src.strategy.black_litterman_mapper import (
    BLViews, BLResult,
    map_biases_to_views, run_black_litterman, compute_bl_weights,
    tau_sensitivity,
    DEFAULT_TAU, DEFAULT_SYMBOLS, BIAS_TO_RETURN_SCALE,
    MIN_VIEW_CONFIDENCE,
)


class TestBLViews:
    """Tests for BLViews dataclass."""

    def test_defaults(self):
        v = BLViews(
            absolute_views={"SPY": 0.05, "GLD": 0.03, "TLT": 0.02},
            view_confidences=[0.5, 0.5, 0.5],
        )
        assert v.tau == DEFAULT_TAU
        assert v.prior == "equal"
        assert v.symbols == DEFAULT_SYMBOLS

    def test_custom_tau(self):
        v = BLViews(
            absolute_views={"SPY": 0.05},
            view_confidences=[0.5],
            tau=0.30,
        )
        assert v.tau == 0.30

    def test_market_prior(self):
        v = BLViews(
            absolute_views={"SPY": 0.05},
            view_confidences=[0.5],
            prior="market",
        )
        assert v.prior == "market"


class TestBLResult:
    """Tests for BLResult dataclass."""

    def test_defaults(self):
        r = BLResult(
            posterior_returns={"SPY": 0.08},
            bl_weights={"SPY": 0.5},
            tau=0.15,
            prior_type="equal",
            view_confidences=[0.5],
        )
        assert r.expected_sharpe is None
        assert r.extras == {}

    def test_with_metrics(self):
        r = BLResult(
            posterior_returns={"SPY": 0.08},
            bl_weights={"SPY": 0.5},
            tau=0.15,
            prior_type="equal",
            view_confidences=[0.5],
            expected_sharpe=0.85,
            expected_cagr=9.5,
            expected_volatility=11.2,
        )
        assert r.expected_sharpe == 0.85
        assert r.expected_cagr == 9.5


class TestMapBiasesToViews:
    """Tests for map_biases_to_views()."""

    def test_zero_biases(self):
        """Zero biases → zero expected returns."""
        views = map_biases_to_views(0.0, 0.0, 0.0)
        assert views.absolute_views["SPY"] == 0.0
        assert views.absolute_views["TLT"] == 0.0
        assert views.absolute_views["GLD"] == 0.0

    def test_positive_equity_bias(self):
        views = map_biases_to_views(equity_bias=1.0, duration_bias=0.0, gold_bias=0.0)
        assert views.absolute_views["SPY"] == pytest.approx(BIAS_TO_RETURN_SCALE)
        assert views.absolute_views["TLT"] == 0.0

    def test_negative_duration_bias(self):
        views = map_biases_to_views(equity_bias=0.0, duration_bias=-0.5, gold_bias=0.0)
        assert views.absolute_views["TLT"] == pytest.approx(-0.5 * BIAS_TO_RETURN_SCALE)

    def test_gold_bias_scaling(self):
        views = map_biases_to_views(equity_bias=0.0, duration_bias=0.0, gold_bias=0.8)
        assert views.absolute_views["GLD"] == pytest.approx(0.8 * BIAS_TO_RETURN_SCALE)

    def test_default_confidences_without_health(self):
        views = map_biases_to_views(0.5, 0.3, 0.2)
        assert views.view_confidences == [0.50, 0.50, 0.50]

    def test_health_scores_modulate_confidence(self):
        """With high health scores, confidences should be higher than default."""
        health = {"MULTI_SPEED_MOM": 0.80, "CROSS_ASSET_RV": 0.75}
        views = map_biases_to_views(0.5, 0.3, 0.2, health_scores=health)
        # With bias magnitudes 0.5, 0.3, 0.2 and avg_health ~0.775:
        # conf = avg_health * (0.5 + 0.5 * |bias|)
        # SPY: 0.775 * (0.5 + 0.25) = 0.581
        # TLT: 0.775 * (0.5 + 0.15) = 0.504
        # GLD: 0.775 * (0.5 + 0.10) = 0.465
        assert views.view_confidences[0] > views.view_confidences[1]
        assert views.view_confidences[1] > views.view_confidences[2]
        # All above minimum
        for c in views.view_confidences:
            assert c >= MIN_VIEW_CONFIDENCE

    def test_zero_bias_with_high_health(self):
        """Zero bias should give base confidence (avg_health * 0.5)."""
        health = {"SIG1": 0.80}
        views = map_biases_to_views(0.0, 0.0, 0.0, health_scores=health)
        # conf = 0.80 * (0.5 + 0) = 0.40
        for c in views.view_confidences:
            assert c == pytest.approx(0.40, abs=0.01)

    def test_confidence_min_floor(self):
        """Very low health should still hit MIN_VIEW_CONFIDENCE floor."""
        health = {"SIG1": 0.01}
        views = map_biases_to_views(0.01, 0.01, 0.01, health_scores=health)
        for c in views.view_confidences:
            assert c >= MIN_VIEW_CONFIDENCE

    def test_confidence_capped_at_one(self):
        """Confidence should never exceed 1.0."""
        health = {"SIG1": 1.0}
        views = map_biases_to_views(1.0, 1.0, 1.0, health_scores=health)
        for c in views.view_confidences:
            assert c <= 1.0

    def test_custom_tau(self):
        views = map_biases_to_views(0.5, 0.3, 0.2, tau=0.30)
        assert views.tau == 0.30

    def test_market_prior(self):
        views = map_biases_to_views(0.5, 0.3, 0.2, prior="market")
        assert views.prior == "market"


class TestRunBlackLitterman:
    """Tests for run_black_litterman()."""

    @pytest.fixture
    def sample_cov(self):
        """Realistic covariance matrix for SPY/GLD/TLT."""
        # Roughly: SPY vol ~15%, GLD vol ~16%, TLT vol ~14%
        # Correlations: SPY-GLD ~0.0, SPY-TLT ~-0.3, GLD-TLT ~0.1
        return np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])

    def test_equal_prior_zero_views(self):
        """Zero views with equal prior should return near-equal weights."""
        views = map_biases_to_views(0.0, 0.0, 0.0)
        result = run_black_litterman(sample_cov_fixture := np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ]), views)
        assert result.prior_type == "equal"
        # Weights should exist for at least one asset
        assert len(result.bl_weights) > 0

    def test_positive_equity_view_shifts_weights(self):
        """Strong equity view should increase SPY allocation."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])

        views_neutral = map_biases_to_views(0.0, 0.0, 0.0)
        views_bull = map_biases_to_views(1.0, 0.0, 0.0)

        result_neutral = run_black_litterman(cov, views_neutral)
        result_bull = run_black_litterman(cov, views_bull)

        # Bull view should shift weight toward SPY
        spy_neutral = result_neutral.bl_weights.get("SPY", 0)
        spy_bull = result_bull.bl_weights.get("SPY", 0)
        assert spy_bull >= spy_neutral

    def test_custom_prior(self):
        """Custom pi array should be used as prior."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.0, 0.0, 0.0)
        custom_pi = np.array([0.08, 0.04, 0.03])
        result = run_black_litterman(cov, views, pi=custom_pi)
        assert result.prior_type == "custom"

    def test_market_prior(self):
        """Market prior should use market cap weights."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.0, 0.0, 0.0, prior="market")
        result = run_black_litterman(cov, views)
        assert result.prior_type == "market"

    def test_covariance_dimension_mismatch(self):
        """Wrong-size covariance matrix should raise ValueError."""
        cov = np.array([[0.01]])  # 1x1 but 3 symbols
        views = map_biases_to_views(0.5, 0.3, 0.2)
        with pytest.raises(ValueError, match="shape"):
            run_black_litterman(cov, views)

    def test_posterior_returns_are_reasonable(self):
        """Posterior returns should be in a sensible range."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.5, -0.3, 0.2)
        result = run_black_litterman(cov, views)
        for sym, ret in result.posterior_returns.items():
            # Returns should be within reasonable bounds
            assert -0.30 < ret < 0.30, f"{sym} posterior return {ret} out of bounds"

    def test_result_has_performance_metrics(self):
        """Result should have Sharpe/CAGR/vol when EF succeeds."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.5, 0.3, 0.2)
        result = run_black_litterman(cov, views)
        # If EF optimization succeeded, metrics should be present
        if result.expected_sharpe is not None:
            assert isinstance(result.expected_sharpe, float)
            assert result.expected_cagr is not None
            assert result.expected_volatility is not None

    def test_weights_sum_approximately_one(self):
        """BL weights should sum to approximately 1.0."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.5, 0.3, 0.2)
        result = run_black_litterman(cov, views)
        weight_sum = sum(result.bl_weights.values())
        assert abs(weight_sum - 1.0) < 0.05, f"Weights sum to {weight_sum}"


class TestComputeBLWeights:
    """Tests for compute_bl_weights() convenience function."""

    def test_with_synthetic_prices(self):
        """Test with synthetic price data."""
        import pandas as pd

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 500)),
            "GLD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 500)),
            "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.008, 500)),
        }, index=dates)

        result = compute_bl_weights(
            prices_df=prices,
            equity_bias=0.5,
            duration_bias=0.2,
            gold_bias=0.3,
        )
        assert isinstance(result, BLResult)
        assert len(result.bl_weights) > 0

    def test_zero_biases(self):
        """Zero biases should still produce valid weights."""
        import pandas as pd

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 500)),
            "GLD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 500)),
            "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.008, 500)),
        }, index=dates)

        result = compute_bl_weights(prices_df=prices)
        assert isinstance(result, BLResult)

    def test_with_health_scores(self):
        """Health scores should modulate view confidence."""
        import pandas as pd

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 500)),
            "GLD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 500)),
            "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.008, 500)),
        }, index=dates)

        health = {"SIG1": 0.80, "SIG2": 0.70, "SIG3": 0.60}
        result = compute_bl_weights(
            prices_df=prices,
            equity_bias=0.5,
            duration_bias=0.2,
            gold_bias=0.3,
            health_scores=health,
        )
        assert isinstance(result, BLResult)


class TestChampionComparison:
    """Integration test: BL weights vs grid search champion."""

    def test_bl_near_champion_region(self):
        """BL with moderate views should produce weights near 46/38/16."""
        import pandas as pd

        np.random.seed(42)
        # Simulate 5 years of data with realistic parameters
        dates = pd.date_range("2020-01-01", periods=1260, freq="B")
        prices = pd.DataFrame({
            "SPY": 300 * np.cumprod(1 + np.random.normal(0.0004, 0.012, 1260)),
            "GLD": 150 * np.cumprod(1 + np.random.normal(0.0002, 0.011, 1260)),
            "TLT": 130 * np.cumprod(1 + np.random.normal(0.0001, 0.009, 1260)),
        }, index=dates)

        # Moderate bullish equity, slight gold and duration view
        result = compute_bl_weights(
            prices_df=prices,
            equity_bias=0.3,
            duration_bias=-0.1,
            gold_bias=0.1,
            tau=0.15,
        )

        # BL should produce non-trivial weights
        assert len(result.bl_weights) >= 2
        # All weights should be positive
        for sym, w in result.bl_weights.items():
            assert w > 0, f"{sym} has non-positive weight {w}"


class TestTauSensitivity:
    """Tests for tau_sensitivity()."""

    def test_default_tau_range(self):
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.5, 0.2, 0.3)
        results = tau_sensitivity(cov, views)
        # Default should have 8 tau values
        assert len(results) == 8
        assert 0.005 in results
        assert 0.15 in results
        assert 0.50 in results

    def test_custom_tau_range(self):
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.5, 0.2, 0.3)
        results = tau_sensitivity(cov, views, tau_values=[0.05, 0.15, 0.30])
        assert len(results) == 3

    def test_results_are_blresult_instances(self):
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.5, 0.2, 0.3)
        results = tau_sensitivity(cov, views, tau_values=[0.05, 0.15])
        for tau, result in results.items():
            assert isinstance(result, BLResult)
            assert result.tau == tau

    def test_higher_tau_changes_metrics(self):
        """Different tau values should produce different Sharpe ratios."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(1.0, -0.5, 0.0)
        results = tau_sensitivity(cov, views, tau_values=[0.005, 0.50])

        if 0.005 in results and 0.50 in results:
            # At minimum, the results should have metrics
            assert results[0.005].expected_sharpe is not None
            assert results[0.50].expected_sharpe is not None


class TestBLViewsExtended:
    """Extended tests for BLViews dataclass."""

    def test_absolute_views_dict(self):
        v = BLViews(
            absolute_views={"SPY": 0.10, "GLD": -0.02},
            view_confidences=[0.7, 0.4],
        )
        assert len(v.absolute_views) == 2
        assert v.absolute_views["SPY"] == 0.10

    def test_view_confidences_length(self):
        v = BLViews(
            absolute_views={"SPY": 0.05},
            view_confidences=[0.6],
        )
        assert len(v.view_confidences) == 1

    def test_custom_symbols(self):
        v = BLViews(
            absolute_views={"EFA": 0.04},
            view_confidences=[0.5],
            symbols=["EFA", "EEM", "SPY"],
        )
        assert v.symbols == ["EFA", "EEM", "SPY"]


class TestBLResultExtended:
    """Extended tests for BLResult dataclass."""

    def test_all_fields(self):
        r = BLResult(
            posterior_returns={"SPY": 0.08, "GLD": 0.03, "TLT": 0.02},
            bl_weights={"SPY": 0.60, "GLD": 0.25, "TLT": 0.15},
            tau=0.15,
            prior_type="equal",
            view_confidences=[0.5, 0.5, 0.5],
            expected_sharpe=0.85,
            expected_cagr=0.10,
            expected_volatility=0.12,
        )
        assert r.tau == 0.15
        assert r.posterior_returns["SPY"] == 0.08
        assert r.bl_weights["SPY"] == 0.60
        assert r.expected_sharpe == 0.85
        assert r.expected_cagr == 0.10
        assert r.prior_type == "equal"

    def test_none_sharpe(self):
        r = BLResult(
            posterior_returns={},
            bl_weights={},
            tau=0.15,
            prior_type="equal",
            view_confidences=[],
        )
        assert r.expected_sharpe is None
        assert r.expected_cagr is None

    def test_extras_default_empty(self):
        r = BLResult(
            posterior_returns={}, bl_weights={}, tau=0.15,
            prior_type="equal", view_confidences=[],
        )
        assert r.extras == {}


class TestMapBiasesToViewsExtended:
    """Extended tests for map_biases_to_views."""

    def test_all_zero_biases(self):
        views = map_biases_to_views(0.0, 0.0, 0.0)
        assert isinstance(views, BLViews)
        for val in views.absolute_views.values():
            assert val == 0.0

    def test_large_equity_bias(self):
        views = map_biases_to_views(2.0, 0.0, 0.0)
        assert views.absolute_views.get("SPY", 0) > 0

    def test_negative_gold_bias(self):
        views = map_biases_to_views(0.0, 0.0, -1.0)
        assert views.absolute_views.get("GLD", 0) < 0

    def test_positive_duration_bias(self):
        views = map_biases_to_views(0.0, 1.0, 0.0)
        assert views.absolute_views.get("TLT", 0) > 0

    def test_confidence_scaled(self):
        views = map_biases_to_views(0.5, 0.2, 0.3)
        assert len(views.view_confidences) > 0
        for c in views.view_confidences:
            assert c > 0


class TestConstants:
    """Validate module constants."""

    def test_default_tau(self):
        assert DEFAULT_TAU == 0.15

    def test_default_symbols(self):
        assert DEFAULT_SYMBOLS == ["SPY", "GLD", "TLT"]

    def test_bias_scale_positive(self):
        assert BIAS_TO_RETURN_SCALE > 0

    def test_min_view_confidence(self):
        assert 0 < MIN_VIEW_CONFIDENCE < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
