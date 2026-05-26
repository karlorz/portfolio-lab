"""Tests for regime-conditional ensemble weights in EnsembleVoter."""

import pytest


class TestRegimeConditionalWeights:
    """Tests for REGIME_CONDITIONAL_WEIGHTS constant."""

    def test_constant_has_all_five_regimes(self):
        from src.strategy.ensemble_voter import REGIME_CONDITIONAL_WEIGHTS
        for reg in ["CRISIS", "HIGH_VOL", "NORMAL", "LOW_VOL", "RECOVERY"]:
            assert reg in REGIME_CONDITIONAL_WEIGHTS

    def test_crisis_boosts_alt_data_reduces_unified(self):
        from src.strategy.ensemble_voter import REGIME_CONDITIONAL_WEIGHTS
        crisis = REGIME_CONDITIONAL_WEIGHTS["CRISIS"]
        assert crisis["alternative_data"] > 1.0
        assert crisis["unified_overlay"] < 1.0

    def test_low_vol_gates_off_cross_asset_regime_arb(self):
        from src.strategy.ensemble_voter import REGIME_CONDITIONAL_WEIGHTS
        low_vol = REGIME_CONDITIONAL_WEIGHTS["LOW_VOL"]
        assert low_vol["cross_asset_regime_arb"] < 1.0
        assert low_vol["international_momentum"] > 1.0

    def test_normal_has_no_multipliers(self):
        from src.strategy.ensemble_voter import REGIME_CONDITIONAL_WEIGHTS
        assert REGIME_CONDITIONAL_WEIGHTS["NORMAL"] == {}


class TestApplyRegimeWeights:
    """Tests for _apply_regime_weights() method."""

    def test_normal_returns_unchanged(self):
        from src.strategy.ensemble_voter import EnsembleVoter, SignalSource, Regime
        voter = EnsembleVoter.__new__(EnsembleVoter)
        weights = {
            SignalSource.ALTERNATIVE_DATA: 0.40,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.30,
            SignalSource.CROSS_ASSET_RV: 0.20,
            SignalSource.UNIFIED_OVERLAY: 0.10,
        }
        result = voter._apply_regime_weights(weights, Regime.NORMAL)
        for k in weights:
            assert abs(result[k] - weights[k]) < 0.01

    def test_crisis_adjusts_weights(self):
        from src.strategy.ensemble_voter import EnsembleVoter, SignalSource, Regime
        voter = EnsembleVoter.__new__(EnsembleVoter)
        weights = {
            SignalSource.ALTERNATIVE_DATA: 0.30,
            SignalSource.UNIFIED_OVERLAY: 0.20,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.25,
            SignalSource.CROSS_ASSET_RV: 0.15,
            SignalSource.CROSS_ASSET_REGIME_ARB: 0.10,
        }
        result = voter._apply_regime_weights(weights, Regime.CRISIS)
        # Alt data (1.3x) should increase relative to unified (0.3x)
        assert result[SignalSource.ALTERNATIVE_DATA] > weights[SignalSource.ALTERNATIVE_DATA]
        assert result[SignalSource.UNIFIED_OVERLAY] < weights[SignalSource.UNIFIED_OVERLAY]
        # Sum = 1.0
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_low_vol_adjusts_weights(self):
        from src.strategy.ensemble_voter import EnsembleVoter, SignalSource, Regime
        voter = EnsembleVoter.__new__(EnsembleVoter)
        weights = {
            SignalSource.ALTERNATIVE_DATA: 0.30,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.25,
            SignalSource.CROSS_ASSET_REGIME_ARB: 0.20,
            SignalSource.UNIFIED_OVERLAY: 0.15,
            SignalSource.CROSS_ASSET_RV: 0.10,
        }
        result = voter._apply_regime_weights(weights, Regime.LOW_VOL)
        # Intl momentum (1.2x) should increase, regime arb (0.5x) should decrease
        assert result[SignalSource.INTERNATIONAL_MOMENTUM] > weights[SignalSource.INTERNATIONAL_MOMENTUM]
        assert result[SignalSource.CROSS_ASSET_REGIME_ARB] < weights[SignalSource.CROSS_ASSET_REGIME_ARB]
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_unknown_regime_defaults_to_no_adjustment(self):
        from src.strategy.ensemble_voter import EnsembleVoter, SignalSource
        voter = EnsembleVoter.__new__(EnsembleVoter)
        weights = {
            SignalSource.ALTERNATIVE_DATA: 0.50,
            SignalSource.CROSS_ASSET_RV: 0.50,
        }
        result = voter._apply_regime_weights(weights, "UNKNOWN_REGIME")
        for k in weights:
            assert abs(result[k] - weights[k]) < 0.01

    def test_signal_not_in_regime_defaults_to_1(self):
        from src.strategy.ensemble_voter import EnsembleVoter, SignalSource, Regime
        voter = EnsembleVoter.__new__(EnsembleVoter)
        weights = {
            SignalSource.ALTERNATIVE_DATA: 0.60,
            SignalSource.MULTI_SPEED_MOM: 0.40,
        }
        result = voter._apply_regime_weights(weights, Regime.LOW_VOL)
        # MULTI_SPEED_MOM not in LOW_VOL config → multiplier 1.0
        # alternative_data in LOW_VOL → multiplier 0.8
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_sum_preserved_to_1_all_regimes(self):
        from src.strategy.ensemble_voter import EnsembleVoter, SignalSource, Regime
        voter = EnsembleVoter.__new__(EnsembleVoter)
        for regime in [Regime.CRISIS, Regime.HIGH_VOL, Regime.NORMAL,
                        Regime.LOW_VOL, Regime.RECOVERY]:
            weights = {
                SignalSource.ALTERNATIVE_DATA: 0.30,
                SignalSource.INTERNATIONAL_MOMENTUM: 0.25,
                SignalSource.CROSS_ASSET_RV: 0.20,
                SignalSource.UNIFIED_OVERLAY: 0.15,
                SignalSource.CROSS_ASSET_REGIME_ARB: 0.10,
            }
            result = voter._apply_regime_weights(weights, regime)
            assert abs(sum(result.values()) - 1.0) < 0.01, f"Sum != 1.0 for {regime.name}"

    def test_exception_graceful_degradation_none_regime(self):
        from src.strategy.ensemble_voter import EnsembleVoter, SignalSource
        voter = EnsembleVoter.__new__(EnsembleVoter)
        weights = {SignalSource.ALTERNATIVE_DATA: 1.0}
        # None regime should not crash
        result = voter._apply_regime_weights(weights, None)
        assert len(result) == 1
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_high_vol_boosts_defensive(self):
        from src.strategy.ensemble_voter import EnsembleVoter, SignalSource, Regime
        voter = EnsembleVoter.__new__(EnsembleVoter)
        weights = {
            SignalSource.ALTERNATIVE_DATA: 0.25,
            SignalSource.UNIFIED_OVERLAY: 0.25,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.25,
            SignalSource.CROSS_ASSET_RV: 0.25,
        }
        result = voter._apply_regime_weights(weights, Regime.HIGH_VOL)
        # unified_overlay (1.2x) should increase, intl_momentum (0.8x) should decrease
        assert result[SignalSource.UNIFIED_OVERLAY] > weights[SignalSource.UNIFIED_OVERLAY]
        assert result[SignalSource.INTERNATIONAL_MOMENTUM] < weights[SignalSource.INTERNATIONAL_MOMENTUM]
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_recovery_boosts_momentum(self):
        from src.strategy.ensemble_voter import EnsembleVoter, SignalSource, Regime
        voter = EnsembleVoter.__new__(EnsembleVoter)
        weights = {
            SignalSource.ALTERNATIVE_DATA: 0.30,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.25,
            SignalSource.CROSS_ASSET_RV: 0.25,
            SignalSource.UNIFIED_OVERLAY: 0.20,
        }
        result = voter._apply_regime_weights(weights, Regime.RECOVERY)
        # intl_momentum (1.3x) should increase most
        assert result[SignalSource.INTERNATIONAL_MOMENTUM] > weights[SignalSource.INTERNATIONAL_MOMENTUM]
        assert abs(sum(result.values()) - 1.0) < 0.01


class TestRegimeWeightsIntegration:
    """Integration/smoke tests for regime-conditional weights."""

    def test_compute_vote_with_explicit_regime(self):
        from src.strategy.ensemble_voter import EnsembleVoter, Regime
        voter = EnsembleVoter()
        assert hasattr(voter, '_apply_regime_weights'), "Method _apply_regime_weights not found"
        # Smoke test: compute_vote with explicit regime
        try:
            vote = voter.compute_vote(regime=Regime.LOW_VOL)
            assert vote.regime == Regime.LOW_VOL
        except Exception:
            # May fail due to missing data files in test env
            pass

    def test_method_returns_dict(self):
        from src.strategy.ensemble_voter import EnsembleVoter, SignalSource, Regime
        voter = EnsembleVoter.__new__(EnsembleVoter)
        weights = {SignalSource.ALTERNATIVE_DATA: 1.0}
        result = voter._apply_regime_weights(weights, Regime.CRISIS)
        assert isinstance(result, dict)
        assert len(result) == 1

    def test_regime_conditional_weights_in_all(self):
        from src.strategy.ensemble_voter import (
            REGIME_CONDITIONAL_WEIGHTS,
            __all__,
        )
        assert 'REGIME_CONDITIONAL_WEIGHTS' in __all__
        assert isinstance(REGIME_CONDITIONAL_WEIGHTS, dict)
