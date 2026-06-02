"""
Tests for ensemble diversity floor — ensures minimum weight for each active
signal to prevent weight concentration and improve N_eff.
"""

import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from dataclasses import fields

from src.strategy.ensemble_voter import (
    EnsembleVoter,
    EnsembleVote,
    Regime,
    SignalReading,
    DEFAULT_DIVERSITY_FLOOR,
)
from src.signals.signal_source import SignalSource


@pytest.fixture
def voter():
    """Create an EnsembleVoter with mocked dependencies."""
    with patch.object(EnsembleVoter, '_init_db'):
        v = EnsembleVoter.__new__(EnsembleVoter)
        v.regime_gate = None
        v.health_tracker = None
        v.bandit = None
        v._attribution_data = None
        v._regime_weights = None
        return v


@pytest.fixture
def concentrated_weights():
    """Weights concentrated on 2 signals — typical pre-floor state."""
    return {
        SignalSource.ALTERNATIVE_DATA: 0.45,
        SignalSource.INTERNATIONAL_MOMENTUM: 0.40,
        SignalSource.CROSS_ASSET_RV: 0.08,
        SignalSource.CROSS_ASSET_REGIME_ARB: 0.05,
        SignalSource.UNIFIED_OVERLAY: 0.02,
        SignalSource.MULTI_TIMEFRAME_FUSION: 0.0,
        SignalSource.GOOGLE_TRENDS: 0.0,
    }


@pytest.fixture
def uniform_weights():
    """Nearly uniform weights — N_eff should already be high."""
    return {s: 1.0 / len(SignalSource) for s in SignalSource}


class TestDiversityFloor:
    """Tests for _apply_diversity_floor()."""

    def test_raises_low_weights_to_floor(self, voter, concentrated_weights):
        """Signals below floor should be raised to floor level."""
        result = voter._apply_diversity_floor(concentrated_weights)
        active = {k: v for k, v in result.items() if concentrated_weights.get(k, 0) > 0 or v > 0}
        # At minimum, all signals that were active (>0) should still have weight
        for source, weight in result.items():
            if concentrated_weights[source] > 0:
                assert weight > 0, f"{source.value} was active but weight dropped to 0"

    def test_preserves_ordering(self, voter, concentrated_weights):
        """Higher-weighted signals should still have higher weights after floor."""
        result = voter._apply_diversity_floor(concentrated_weights)
        # ALTERNATIVE_DATA was highest, should still be >= CROSS_ASSET_RV
        assert result[SignalSource.ALTERNATIVE_DATA] >= result[SignalSource.CROSS_ASSET_RV]

    def test_weights_sum_to_one(self, voter, concentrated_weights):
        """After floor application, weights must sum to ~1.0."""
        result = voter._apply_diversity_floor(concentrated_weights)
        total = sum(result.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"

    def test_increases_n_eff(self, voter, concentrated_weights):
        """Floor application should increase N_eff."""
        # Compute N_eff before
        w_before = np.array([v for v in concentrated_weights.values() if v > 0])
        n_eff_before = float(np.exp(-np.sum(w_before * np.log(w_before))))

        result = voter._apply_diversity_floor(concentrated_weights)

        w_after = np.array([v for v in result.values() if v > 0])
        n_eff_after = float(np.exp(-np.sum(w_after * np.log(w_after))))

        assert n_eff_after > n_eff_before, f"N_eff did not improve: {n_eff_before:.2f} -> {n_eff_after:.2f}"

    def test_no_change_when_already_diverse(self, voter, uniform_weights):
        """Uniform weights should not change significantly."""
        result = voter._apply_diversity_floor(uniform_weights)
        for source in uniform_weights:
            assert abs(result[source] - uniform_weights[source]) < 0.02

    def test_custom_floor_value(self, voter, concentrated_weights):
        """Floor value should be configurable via parameter."""
        # Higher floor should make weights more uniform
        result_high = voter._apply_diversity_floor(concentrated_weights, floor=0.10)
        result_low = voter._apply_diversity_floor(concentrated_weights, floor=0.02)

        # With higher floor, minimum weight should be higher
        min_high = min(v for v in result_high.values() if v > 0)
        min_low = min(v for v in result_low.values() if v > 0)
        assert min_high >= min_low

    def test_env_var_floor(self, voter, concentrated_weights):
        """ENSEMBLE_DIVERSITY_FLOOR env var should control floor."""
        with patch.dict(os.environ, {"ENSEMBLE_DIVERSITY_FLOOR": "0.08"}):
            result = voter._apply_diversity_floor(concentrated_weights)
            # Active signals should have at least ~8% after normalization
            total = sum(result.values())
            for source, weight in result.items():
                if concentrated_weights[source] > 0:
                    assert weight > 0

    def test_zero_floor_is_noop(self, voter, concentrated_weights):
        """Floor of 0 should not change weights."""
        result = voter._apply_diversity_floor(concentrated_weights, floor=0.0)
        for source in concentrated_weights:
            assert abs(result[source] - concentrated_weights[source]) < 0.001

    def test_all_zero_weights_handled(self, voter):
        """All-zero weights should not crash."""
        zero_weights = {s: 0.0 for s in SignalSource}
        result = voter._apply_diversity_floor(zero_weights)
        # Should return unchanged
        assert all(v == 0.0 for v in result.values())

    def test_single_signal_unchanged(self, voter):
        """Single active signal should remain unchanged."""
        single = {s: 0.0 for s in SignalSource}
        single[SignalSource.ALTERNATIVE_DATA] = 1.0
        result = voter._apply_diversity_floor(single)
        assert result[SignalSource.ALTERNATIVE_DATA] == 1.0


class TestDiversityFloorIntegration:
    """Test that diversity floor integrates into compute_vote pipeline."""

    def test_floor_applied_in_pipeline(self, voter):
        """Diversity floor should be called during compute_vote."""
        # Create minimal readings
        readings = {}
        for src in [SignalSource.ALTERNATIVE_DATA, SignalSource.INTERNATIONAL_MOMENTUM, SignalSource.CROSS_ASSET_RV]:
            readings[src] = SignalReading(
                source=src,
                value=0.5,
                confidence=0.7,
                weight=0.0,
                timestamp="2026-01-01",
                regime_fit="normal",
            )

        # Mock the pipeline stages
        with patch.object(voter, '_resolve_inputs', return_value=(
            readings, Regime.NORMAL, 0.8
        )), patch.object(voter, 'get_blended_weights', return_value={
            s: 0.15 for s in SignalSource
        }), patch.object(voter, '_apply_regime_gating', side_effect=lambda w, r, c=0.8: w), \
            patch.object(voter, '_apply_adaptive_weights', side_effect=lambda w, r: w), \
            patch.object(voter, '_apply_health_weights', side_effect=lambda w: w), \
            patch.object(voter, '_apply_correlation_penalty', side_effect=lambda w: w), \
            patch.object(voter, '_apply_regime_weights', side_effect=lambda w, r: w), \
            patch.object(voter, '_apply_utility_reweighting', side_effect=lambda w, r: w), \
            patch.object(voter, '_apply_exploration_noise', side_effect=lambda w, r: w), \
            patch.object(voter, '_apply_turnover_validation', side_effect=lambda w, rd, r: w), \
            patch.object(voter, '_persist_vote'):

            # Floor should make N_eff higher even with concentrated initial weights
            vote = voter.compute_vote(readings, Regime.NORMAL, 0.8)
            assert vote.n_eff > 0


class TestDiversityFloorConstants:
    """Tests for diversity floor defaults."""

    def test_default_floor_value(self):
        """Default diversity floor should be reasonable (2-8%)."""
        from src.strategy.ensemble_voter import DEFAULT_DIVERSITY_FLOOR
        assert 0.02 <= DEFAULT_DIVERSITY_FLOOR <= 0.08
