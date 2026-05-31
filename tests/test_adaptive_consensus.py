"""Tests for adaptive consensus thresholds by regime.

TDD red phase — defines behavior before implementation.
"""

import pytest
from src.strategy.ensemble_voter import (
    EnsembleVoter,
    Regime,
    REGIME_CONSENSUS_THRESHOLDS,
)


class TestAdaptiveConsensus:
    """Test suite for regime-conditional consensus thresholds."""

    def test_thresholds_defined(self):
        """REGIME_CONSENSUS_THRESHOLDS should cover all regimes."""
        assert set(REGIME_CONSENSUS_THRESHOLDS.keys()) == {
            "NORMAL", "CRISIS", "HIGH_VOL", "LOW_VOL", "RECOVERY"
        }

    def test_crisis_lowest_threshold(self):
        """CRISIS should have the lowest threshold (act fast)."""
        assert REGIME_CONSENSUS_THRESHOLDS["CRISIS"] <= 0.55

    def test_normal_highest_threshold(self):
        """NORMAL should have the highest threshold (require consensus)."""
        assert REGIME_CONSENSUS_THRESHOLDS["NORMAL"] >= 0.70

    def test_thresholds_in_valid_range(self):
        """All thresholds should be between 0.3 and 0.9."""
        for regime, threshold in REGIME_CONSENSUS_THRESHOLDS.items():
            assert 0.3 <= threshold <= 0.9, f"{regime} threshold {threshold} out of range"

    def test_determine_action_uses_regime_threshold(self):
        """_determine_action should use regime-specific threshold."""
        # High agreement (0.65) — enough for HIGH_VOL (0.55) but not NORMAL (0.75)
        action_hv, _ = EnsembleVoter._determine_action(
            Regime.HIGH_VOL, 0.8, 0.4, 0.65,
        )
        action_normal, _ = EnsembleVoter._determine_action(
            Regime.NORMAL, 0.8, 0.4, 0.65,
        )
        # HIGH_VOL should act (lower threshold), NORMAL should not
        assert action_hv == "increase_equity"
        assert action_normal == "neutral"

    def test_determine_action_crisis_always_risk_off(self):
        """CRISIS regime always returns risk_off regardless of agreement."""
        action, _ = EnsembleVoter._determine_action(
            Regime.CRISIS, 0.9, 0.5, 0.9,
        )
        assert action == "risk_off"

    def test_low_vol_moderate_threshold(self):
        """LOW_VOL should have a moderate threshold."""
        threshold = REGIME_CONSENSUS_THRESHOLDS["LOW_VOL"]
        assert 0.60 <= threshold <= 0.75

    def test_recovery_lower_threshold(self):
        """RECOVERY should have a lower threshold than NORMAL (capitalize fast)."""
        assert REGIME_CONSENSUS_THRESHOLDS["RECOVERY"] < REGIME_CONSENSUS_THRESHOLDS["NORMAL"]

    def test_env_override_still_works(self):
        """ENSEMBLE_CONSENSUS_THRESHOLD env var should still be used as fallback."""
        from src.paths import ENSEMBLE_CONSENSUS_THRESHOLD
        assert isinstance(ENSEMBLE_CONSENSUS_THRESHOLD, float)
        assert 0 < ENSEMBLE_CONSENSUS_THRESHOLD < 1
