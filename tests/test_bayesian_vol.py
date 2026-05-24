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


class TestBayesianVolModelExtended:
    """Extended coverage for BayesianVolModel edge cases."""

    @pytest.fixture
    def model(self):
        return BayesianVolModel(prior_window=252, update_window=20)

    def test_fit_prior_constant_vols(self, model):
        """Constant vol history should return that vol with high precision."""
        vols = [0.20] * 300
        prior_vol, prior_prec = model.fit_prior(vols)
        assert abs(prior_vol - 0.20) < 0.01
        assert prior_prec == 60.0  # Max precision (capped)

    def test_fit_prior_single_value(self, model):
        """Single value falls below prior_window threshold."""
        prior_vol, prior_prec = model.fit_prior([0.18])
        assert prior_vol == 0.20  # Default
        assert prior_prec == 10.0

    def test_update_single_observation(self, model):
        """Single observation should still produce valid estimate."""
        result = model.update(0.20, 10.0, [0.30])
        assert result.is_valid
        assert result.n_obs == 1
        assert result.likelihood_vol == 0.30

    def test_update_with_zero_regime_scale(self, model):
        """Regime scale=0 would cause division, but default is 1.0.
        Test very small regime_scale (strong prior dominance)."""
        result = model.update(0.20, 10.0, [0.40] * 5, regime_scale=0.5)
        # regime_scale < 1 strengthens prior → more shrinkage
        result_normal = model.update(0.20, 10.0, [0.40] * 5, regime_scale=1.0)
        assert result.shrinkage_factor > result_normal.shrinkage_factor

    def test_update_large_regime_scale_ignores_prior(self, model):
        """Very large regime_scale should make posterior follow likelihood."""
        result = model.update(0.20, 30.0, [0.40] * 5, regime_scale=100.0)
        assert abs(result.posterior_vol - result.likelihood_vol) < 0.01
        assert result.shrinkage_factor < 0.1

    def test_update_high_vol_regime_flag(self, model):
        """regime_scale > 1.5 should set is_high_vol_regime."""
        result = model.update(0.20, 10.0, [0.30] * 10, regime_scale=2.0)
        assert result.is_high_vol_regime

    def test_update_low_regime_not_flagged(self, model):
        """regime_scale <= 1.5 should NOT set is_high_vol_regime."""
        result = model.update(0.20, 10.0, [0.30] * 10, regime_scale=1.0)
        assert not result.is_high_vol_regime

    def test_credible_interval_width_varies_with_obs(self, model):
        """More observations should narrow the credible interval."""
        result_few = model.update(0.20, 10.0, [0.25] * 3)
        result_many = model.update(0.20, 10.0, [0.25] * 50)
        width_few = result_few.credible_interval_upper - result_few.credible_interval_lower
        width_many = result_many.credible_interval_upper - result_many.credible_interval_lower
        assert width_many < width_few

    def test_update_result_fields_populated(self, model):
        """All result fields should be populated and reasonable."""
        result = model.update(0.20, 10.0, [0.22, 0.18, 0.20, 0.19, 0.21])
        assert result.prior_vol == 0.20
        assert result.prior_precision == 10.0
        assert 0.15 < result.likelihood_vol < 0.25
        assert 0.10 < result.posterior_vol < 0.30
        assert 0 <= result.shrinkage_factor <= 1
        assert result.is_valid

    def test_compute_regime_scale_short_returns(self, model):
        """Fewer than 20 returns should default to scale 1.0."""
        scale = model.compute_regime_scale([0.01, -0.01, 0.02])
        assert scale == 1.0

    def test_compute_regime_scale_thresholds(self, model):
        """Test kurtosis-based scale thresholds."""
        rng = np.random.RandomState(99)
        # Moderate kurtosis → scale 1.5
        normal_rets = list(rng.normal(0, 0.01, 60))
        normal_rets[5] = -0.04  # Slight fat tail
        scale = model.compute_regime_scale(normal_rets)
        assert 1.0 <= scale <= 3.0  # Valid range


class TestBayesianVolPipelineExtended:
    """Extended pipeline coverage."""

    @pytest.fixture
    def pipeline(self):
        return BayesianVolPipeline()

    def test_estimate_with_explicit_vol_history(self, pipeline):
        """Passing vol_history directly should skip DB loading."""
        vol_history = [0.20 + 0.01 * i for i in range(50)]
        result = pipeline.estimate("TEST", vol_history=vol_history)
        assert result.symbol == "TEST"
        assert result.is_valid

    def test_estimate_with_recent_returns(self, pipeline):
        """Passing recent_returns should compute regime_scale."""
        vol_history = [0.18] * 100
        recent_returns = [0.01, -0.01, 0.02, -0.015, 0.005] * 10
        result = pipeline.estimate("TEST", vol_history=vol_history,
                                   recent_returns=recent_returns)
        assert result.regime_scale >= 1.0

    def test_estimate_short_vol_history(self, pipeline):
        """Short vol history (< update_window) should still work."""
        vol_history = [0.20, 0.22, 0.18]
        result = pipeline.estimate("TEST", vol_history=vol_history)
        assert result.is_valid

    def test_save_estimate(self, pipeline, tmp_path):
        """save_estimate should write JSON file."""
        vol_history = [0.20] * 100
        result = pipeline.estimate("TESTSAVE", vol_history=vol_history)
        # Override output dir for test
        pipeline.OUTPUT_DIR = tmp_path
        pipeline.save_estimate(result)
        out_file = tmp_path / "TESTSAVE_bayesian_vol.json"
        assert out_file.exists()
        import json
        with open(out_file) as f:
            data = json.load(f)
        assert data["symbol"] == "TESTSAVE"
        assert "posterior_vol" in data

    def test_to_dict_has_all_fields(self, pipeline):
        """to_dict should contain all dataclass fields."""
        vol_history = [0.20] * 50
        result = pipeline.estimate("SPY", vol_history=vol_history)
        d = result.to_dict()
        expected_keys = {
            "symbol", "timestamp", "window", "prior_vol", "prior_precision",
            "likelihood_vol", "likelihood_precision", "posterior_vol",
            "credible_interval_lower", "credible_interval_upper",
            "simple_mean_vol", "shrinkage_factor", "regime_scale",
            "is_high_vol_regime", "n_obs", "is_valid",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_estimate_different_symbols(self, pipeline):
        """Estimate should work for different symbols."""
        for sym in ["SPY", "GLD", "TLT"]:
            result = pipeline.estimate(sym)
            assert isinstance(result, BayesianVolEstimate)
            assert result.symbol == sym
