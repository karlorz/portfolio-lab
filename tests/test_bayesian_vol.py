"""
Tests for Bayesian Adaptive Volatility Model (v5.20)
"""

import pytest
import numpy as np
import math
import sys
import logging
from unittest.mock import patch
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


class TestBayesianVolModelMoreExtended:
    """Additional edge cases beyond the existing extended tests."""

    @pytest.fixture
    def model(self):
        return BayesianVolModel(prior_window=252, update_window=20)

    def test_fit_prior_wide_spread_vols(self, model):
        """Wide spread of vols should give lower precision."""
        rng = np.random.RandomState(42)
        vols = list(rng.uniform(0.05, 0.60, 300))
        _, prec = model.fit_prior(vols)
        # Wide spread → low precision (unless capped at 60)
        assert prec <= 60.0

    def test_update_extreme_likelihood(self, model):
        """Extreme likelihood vol should be shrunk toward prior."""
        result = model.update(0.20, 30.0, [0.80] * 5)
        # Posterior should be between prior (0.20) and likelihood (~0.80)
        assert result.posterior_vol < 0.80
        assert result.posterior_vol > 0.20

    def test_update_all_same_recent(self, model):
        """All identical recent values should give very precise likelihood."""
        result = model.update(0.20, 10.0, [0.25] * 20)
        assert result.likelihood_vol == pytest.approx(0.25, abs=0.01)

    def test_regime_scale_monotonic_with_kurtosis(self, model):
        """Higher kurtosis should generally produce higher regime_scale."""
        rng = np.random.RandomState(42)
        base = list(rng.normal(0, 0.01, 60))
        # Add more extreme events for second series
        extreme = base.copy()
        for i in [10, 20, 30, 40, 50]:
            extreme[i] = -0.08
        for i in [11, 21, 31, 41, 51]:
            extreme[i] = 0.08
        scale_base = model.compute_regime_scale(base)
        scale_extreme = model.compute_regime_scale(extreme)
        assert scale_extreme >= scale_base

    def test_excess_kurtosis_single_value(self, model):
        """Single value should return 0 (insufficient data)."""
        ek = model._excess_kurtosis([1.0])
        assert ek == 0.0

    def test_excess_kurtosis_two_values(self, model):
        """Two identical values → excess kurtosis = -2 (degenerate)."""
        ek = model._excess_kurtosis([1.0, 1.0])
        # Variance = 0, should return 0 per implementation
        assert ek == 0.0

    def test_update_zero_prior_precision(self, model):
        """Zero prior precision should make posterior follow likelihood."""
        result = model.update(0.20, 0.01, [0.40] * 10)
        # Very weak prior → posterior should be near likelihood
        assert abs(result.posterior_vol - result.likelihood_vol) < 0.05


class TestBayesianVolPipelineMoreExtended:
    """Additional pipeline edge cases."""

    @pytest.fixture
    def pipeline(self):
        return BayesianVolPipeline()

    def test_estimate_empty_vol_history(self, pipeline):
        """Empty vol history should still produce a valid estimate."""
        result = pipeline.estimate("TEST", vol_history=[])
        # Should use defaults
        assert isinstance(result, BayesianVolEstimate)

    def test_estimate_very_short_recent_returns(self, pipeline):
        """Very short recent_returns list should default scale to 1.0."""
        vol_history = [0.20] * 50
        recent_returns = [0.01]
        result = pipeline.estimate("TEST", vol_history=vol_history,
                                   recent_returns=recent_returns)
        assert result.regime_scale == 1.0

    def test_estimate_none_recent_returns(self, pipeline):
        """None recent_returns should use default regime_scale."""
        vol_history = [0.20] * 50
        result = pipeline.estimate("TEST", vol_history=vol_history,
                                   recent_returns=None)
        assert result.regime_scale == 1.0

    def test_save_estimate_to_existing_dir(self, pipeline, tmp_path):
        """save_estimate should write to an existing directory."""
        vol_history = [0.20] * 50
        result = pipeline.estimate("TESTDIR", vol_history=vol_history)
        pipeline.OUTPUT_DIR = tmp_path
        pipeline.save_estimate(result)
        assert (tmp_path / "TESTDIR_bayesian_vol.json").exists()


class TestExports:
    """Module __all__ exports validation."""

    def test_all_exports(self):
        import src.monitor.bayesian_vol as mod
        expected = {'BayesianVolEstimate', 'BayesianVolModel', 'BayesianVolPipeline', 'estimate_bayesian_vol'}
        assert expected.issubset(set(mod.__all__))

    def test_all_exports_importable(self):
        from src.monitor.bayesian_vol import (
            BayesianVolEstimate, BayesianVolModel,
            BayesianVolPipeline, estimate_bayesian_vol,
        )
        assert BayesianVolEstimate is not None
        assert BayesianVolModel is not None


class TestBayesianVolEstimateDataclass:
    """Comprehensive dataclass field validation."""

    def test_all_fields_in_to_dict(self):
        from dataclasses import fields
        est = BayesianVolEstimate(
            symbol="SPY", timestamp="2026-01-01", window=20,
            prior_vol=0.20, prior_precision=30.0,
            likelihood_vol=0.18, likelihood_precision=20.0,
            posterior_vol=0.19,
            credible_interval_lower=0.15, credible_interval_upper=0.23,
            simple_mean_vol=0.18, shrinkage_factor=0.6,
            regime_scale=1.0, is_high_vol_regime=False,
            n_obs=20, is_valid=True,
        )
        d = est.to_dict()
        for f in fields(BayesianVolEstimate):
            assert f.name in d, f"Missing field: {f.name}"

    def test_field_types(self):
        est = BayesianVolEstimate(
            symbol="SPY", timestamp="2026-01-01", window=20,
            prior_vol=0.20, prior_precision=30.0,
            likelihood_vol=0.18, likelihood_precision=20.0,
            posterior_vol=0.19,
            credible_interval_lower=0.15, credible_interval_upper=0.23,
            simple_mean_vol=0.18, shrinkage_factor=0.6,
            regime_scale=1.0, is_high_vol_regime=False,
            n_obs=20, is_valid=True,
        )
        assert isinstance(est.symbol, str)
        assert isinstance(est.window, int)
        assert isinstance(est.prior_vol, float)
        assert isinstance(est.is_high_vol_regime, bool)
        assert isinstance(est.is_valid, bool)

    def test_to_dict_json_serializable(self):
        est = BayesianVolEstimate(
            symbol="SPY", timestamp="2026-01-01", window=20,
            prior_vol=0.20, prior_precision=30.0,
            likelihood_vol=0.18, likelihood_precision=20.0,
            posterior_vol=0.19,
            credible_interval_lower=0.15, credible_interval_upper=0.23,
            simple_mean_vol=0.18, shrinkage_factor=0.6,
            regime_scale=1.0, is_high_vol_regime=False,
            n_obs=20, is_valid=True,
        )
        import json
        serialized = json.dumps(est.to_dict())
        assert isinstance(serialized, str)
        data = json.loads(serialized)
        assert data["symbol"] == "SPY"


class TestExcessKurtosis:
    """Test _excess_kurtosis static method."""

    def test_normal_distribution_kurtosis_near_zero(self):
        """Normal distribution should have excess kurtosis near 0."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 10000)
        kurt = BayesianVolModel._excess_kurtosis(x)
        assert abs(kurt) < 0.5  # Should be near 0

    def test_too_few_elements(self):
        """Less than 4 elements should return 0."""
        assert BayesianVolModel._excess_kurtosis(np.array([1.0, 2.0, 3.0])) == 0.0

    def test_constant_series(self):
        """All same values should return 0."""
        assert BayesianVolModel._excess_kurtosis(np.ones(100)) == 0.0

    def test_heavy_tailed_distribution(self):
        """Heavy-tailed data should have positive excess kurtosis."""
        np.random.seed(42)
        x = np.concatenate([np.random.normal(0, 1, 100), np.array([5.0, -5.0, 8.0, -8.0])])
        kurt = BayesianVolModel._excess_kurtosis(x)
        assert kurt > 0


class TestComputeRegimeScaleExtended:
    """Extended regime scale tests."""

    def test_crisis_regime(self):
        """Excess kurtosis > 5 should give scale 3.0."""
        model = BayesianVolModel()
        # Create heavy-tailed returns
        returns = [0.01] * 50 + [0.2, -0.2, 0.15, -0.15, 0.3, -0.3, 0.25, -0.25, 0.35, -0.35]
        scale = model.compute_regime_scale(returns)
        assert scale >= 2.0  # Crisis or high kurtosis

    def test_normal_regime(self):
        """Normal returns should give scale 1.0."""
        model = BayesianVolModel()
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.015, 100))
        scale = model.compute_regime_scale(returns)
        assert scale == 1.0


class TestBayesianUpdateEdgeCases:
    """Edge cases for Bayesian update."""

    def test_update_with_single_observation(self):
        model = BayesianVolModel()
        result = model.update(0.20, 30.0, [0.25])
        assert result.is_valid
        assert result.n_obs == 1
        assert result.likelihood_vol == 0.25

    def test_update_with_high_regime_scale(self):
        model = BayesianVolModel()
        result_normal = model.update(0.20, 30.0, [0.30] * 10, regime_scale=1.0)
        result_crisis = model.update(0.20, 30.0, [0.30] * 10, regime_scale=3.0)
        # Higher regime scale → less shrinkage toward prior
        assert result_crisis.shrinkage_factor < result_normal.shrinkage_factor

    def test_update_zero_recent_vols(self):
        """All-zero recent vols should still produce valid result."""
        model = BayesianVolModel()
        result = model.update(0.20, 30.0, [0.0] * 5)
        assert result.is_valid
        assert result.posterior_vol >= 0

    def test_posterior_between_prior_and_likelihood(self):
        """Posterior should be between prior and likelihood when regime_scale=1."""
        model = BayesianVolModel()
        result = model.update(0.20, 30.0, [0.30] * 20, regime_scale=1.0)
        # With non-trivial prior, posterior should be between prior and likelihood
        assert result.posterior_vol >= min(result.prior_vol, result.likelihood_vol) - 0.01
        assert result.posterior_vol <= max(result.prior_vol, result.likelihood_vol) + 0.01


class TestFitPriorEdgeCases:
    """Edge cases for fit_prior."""

    def test_short_history_returns_defaults(self):
        model = BayesianVolModel()
        prior_vol, precision = model.fit_prior([0.20] * 30)
        assert prior_vol == 0.20
        assert precision == 10.0

    def test_stable_vol_high_precision(self):
        """Stable vol history should give higher prior precision."""
        model = BayesianVolModel()
        stable = [0.20] * 300
        volatile = [0.10 + 0.15 * (i % 10) / 10 for i in range(300)]
        _, stable_alpha = model.fit_prior(stable)
        _, volatile_alpha = model.fit_prior(volatile)
        assert stable_alpha >= volatile_alpha

    def test_long_history_uses_prior_window(self):
        """History longer than prior_window should only use last prior_window values."""
        model = BayesianVolModel(prior_window=60)
        # First 200 entries at 10%, last 200 at 30%
        history = [0.10] * 200 + [0.30] * 200
        prior_vol, _ = model.fit_prior(history)
        # Should be around 30% since we use last 60 (all from the 30% range)
        assert prior_vol > 0.25


class TestConvenienceFunctions:
    """Test standalone convenience functions."""

    def test_estimate_bayesian_vol(self):
        from src.monitor.bayesian_vol import estimate_bayesian_vol
        with patch.object(BayesianVolPipeline, 'estimate') as mock_est:
            mock_est.return_value = BayesianVolEstimate(
                symbol="SPY", timestamp="2026-01-01", window=20,
                prior_vol=0.20, prior_precision=30.0,
                likelihood_vol=0.18, likelihood_precision=20.0,
                posterior_vol=0.19,
                credible_interval_lower=0.15, credible_interval_upper=0.23,
                simple_mean_vol=0.18, shrinkage_factor=0.6,
                regime_scale=1.0, is_high_vol_regime=False,
                n_obs=20, is_valid=True,
            )
            result = estimate_bayesian_vol("SPY")
            assert isinstance(result, BayesianVolEstimate)


class TestCLI:
    """CLI main() function tests."""

    def test_main_runs(self, caplog):
        from src.monitor.bayesian_vol import main
        with patch.object(BayesianVolPipeline, 'estimate') as mock_est:
            mock_est.return_value = BayesianVolEstimate(
                symbol="SPY", timestamp="2026-01-01", window=20,
                prior_vol=0.20, prior_precision=30.0,
                likelihood_vol=0.18, likelihood_precision=20.0,
                posterior_vol=0.19,
                credible_interval_lower=0.15, credible_interval_upper=0.23,
                simple_mean_vol=0.18, shrinkage_factor=0.6,
                regime_scale=1.0, is_high_vol_regime=False,
                n_obs=20, is_valid=True,
            )
            with patch.object(sys, 'argv', ['bayesian_vol.py', '--symbol', 'SPY']):
                with caplog.at_level(logging.INFO, logger="src.monitor.bayesian_vol"):
                    main()
        assert "BAYESIAN VOLATILITY" in caplog.text
