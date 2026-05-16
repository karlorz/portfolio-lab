"""
Tests for Bayesian Adaptive Volatility Model (v5.20)
"""

import pytest
import numpy as np
import math
from src.monitor.bayesian_vol import (
    BayesianVolModel, BayesianVolPipeline, BayesianVolEstimate,
    estimate_bayesian_vol,
)


class TestBayesianVolModel:
    @pytest.fixture
    def model(self):
        return BayesianVolModel(prior_window=252, update_window=20)

    def test_fit_prior_with_sufficient_data(self, model):
        rng = np.random.RandomState(42)
        vols = list(rng.normal(0.20, 0.03, 300))
        vols = [abs(v) for v in vols]  # Ensure positive
        prior_vol, prior_precision = model.fit_prior(vols)
        assert 0.10 < prior_vol < 0.40
        assert prior_precision > 0

    def test_fit_prior_insufficient_data(self, model):
        prior_vol, prior_precision = model.fit_prior([0.15, 0.18, 0.22])
        assert prior_vol == 0.20  # Default
        assert prior_precision == 10.0

    def test_bayesian_update_shrinks_toward_prior(self, model):
        """With few observations, posterior should be closer to prior."""
        prior_vol, prior_prec = 0.20, 30.0  # Strong prior
        recent = [0.40, 0.38, 0.42]  # Very different from prior
        result = model.update(prior_vol, prior_prec, recent)
        # Should shrink toward prior (0.20) rather than follow recent (0.40)
        assert result.posterior_vol < result.likelihood_vol
        assert result.shrinkage_factor > 0.5  # Strong shrinkage

    def test_bayesian_update_weak_prior(self, model):
        """With weak prior, posterior should follow data closely."""
        prior_vol, prior_prec = 0.20, 1.0  # Weak prior
        recent = [0.35] * 30  # Lots of recent data
        result = model.update(prior_vol, prior_prec, recent)
        assert abs(result.posterior_vol - result.likelihood_vol) < 0.05

    def test_regime_scale_reduces_shrinkage(self, model):
        """High regime scale should reduce prior weight."""
        prior_vol, prior_prec = 0.20, 30.0
        recent = [0.40, 0.42, 0.38, 0.41, 0.39]
        normal = model.update(prior_vol, prior_prec, recent, regime_scale=1.0)
        crisis = model.update(prior_vol, prior_prec, recent, regime_scale=3.0)
        # Crisis should be more responsive (less shrinkage)
        assert crisis.shrinkage_factor < normal.shrinkage_factor
        assert crisis.posterior_vol > normal.posterior_vol

    def test_empty_recent_returns_prior(self, model):
        result = model.update(0.20, 10.0, [])
        assert not result.is_valid
        assert result.posterior_vol == 0.20

    def test_credible_interval_contains_posterior(self, model):
        result = model.update(0.20, 10.0, [0.18, 0.22, 0.19, 0.21, 0.20])
        assert result.credible_interval_lower < result.posterior_vol
        assert result.credible_interval_upper > result.posterior_vol

    def test_regime_scale_from_normal_returns(self, model):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 100))
        scale = model.compute_regime_scale(returns)
        assert 0.5 <= scale <= 1.5  # Normal returns → scale near 1

    def test_regime_scale_from_fat_tails(self, model):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 60))
        # Add extreme events
        for i in [10, 20, 30, 40, 50]:
            returns[i] = -0.08
        scale = model.compute_regime_scale(returns)
        assert scale > 1.0  # Fat tails → elevated scale

    def test_excess_kurtosis_normal(self, model):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 1000)
        ek = model._excess_kurtosis(x)
        assert abs(ek) < 0.5  # Normal → ~0 excess

    def test_excess_kurtosis_fat_tail(self, model):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 1000)
        x[::20] = rng.normal(0, 5, 50)  # Fat tails every 20th
        ek = model._excess_kurtosis(x)
        assert ek > 1.0  # Should be elevated


class TestBayesianVolPipeline:
    @pytest.fixture
    def pipeline(self):
        return BayesianVolPipeline()

    def test_estimate_returns_result(self, pipeline):
        result = pipeline.estimate("SPY")
        assert isinstance(result, BayesianVolEstimate)
        assert result.symbol == "SPY"

    def test_convenience_function(self):
        result = estimate_bayesian_vol("SPY")
        assert isinstance(result, BayesianVolEstimate)

    def test_result_serializable(self, pipeline):
        result = pipeline.estimate("SPY")
        d = result.to_dict()
        assert "posterior_vol" in d
        assert "shrinkage_factor" in d
