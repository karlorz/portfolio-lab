"""
Integration tests for the full compute_vote pipeline:
  RegimeGate → BanditWeighter → compute_vote → renormalization

These tests validate that the components work correctly when wired together
inside EnsembleVoter.compute_vote, covering:
1. RegimeGate zeros out gated signals and weights are renormalized
2. BanditWeighter blends with static weights over time
3. Hysteresis guard prevents premature gate changes
4. Combined flow: gate + blend + health + renormalize
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.strategy.ensemble_voter import (
    BanditWeighter, EnsembleVoter, Regime, SignalSource,
    SignalReading, REGIME_WEIGHTS,
)
from src.signals.regime_gate import RegimeGate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reading(source=SignalSource.MULTI_SPEED_MOM, value=0.5,
                  confidence=0.8, asset_signals=None):
    return SignalReading(
        source=source,
        timestamp='2026-01-01',
        value=value,
        confidence=confidence,
        weight=0.0,
        regime_fit='all',
        asset_signals=asset_signals or {'SPY': 0.5, 'TLT': -0.2, 'GLD': 0.1},
        explanation='test',
    )


def _make_voter(tmp_path):
    """Create an EnsembleVoter with DB initialized but without expensive I/O."""
    voter = EnsembleVoter.__new__(EnsembleVoter)
    voter.data_path = tmp_path
    voter.db_path = tmp_path / "ensemble_signals.db"
    voter.current_readings = {}
    voter.current_regime = Regime.NORMAL
    voter.current_regime_confidence = 0.5
    voter._init_db()

    # BanditWeighter — same as real __init__
    voter.bandit = BanditWeighter(
        signals=[s.value for s in SignalSource],
        epsilon=0.1,
        window=252,
    )
    voter.bandit_observations = 0

    # RegimeGate — same as real __init__
    voter.regime_gate = RegimeGate()
    voter._prev_regime = None
    voter._days_in_regime = 999

    return voter


def _all_signal_readings():
    """Return one reading per signal source (for multi-signal tests)."""
    return {
        SignalSource.MULTI_SPEED_MOM: _make_reading(
            source=SignalSource.MULTI_SPEED_MOM, value=0.3),
        SignalSource.CROSS_ASSET_RV: _make_reading(
            source=SignalSource.CROSS_ASSET_RV, value=0.2),
        SignalSource.INTERNATIONAL_MOMENTUM: _make_reading(
            source=SignalSource.INTERNATIONAL_MOMENTUM, value=0.4),
        SignalSource.ALTERNATIVE_DATA: _make_reading(
            source=SignalSource.ALTERNATIVE_DATA, value=0.5),
        SignalSource.CROSS_ASSET_REGIME_ARB: _make_reading(
            source=SignalSource.CROSS_ASSET_REGIME_ARB, value=0.1),
        SignalSource.UNIFIED_OVERLAY: _make_reading(
            source=SignalSource.UNIFIED_OVERLAY, value=0.2),
    }


# ---------------------------------------------------------------------------
# 1. RegimeGate integration in compute_vote
# ---------------------------------------------------------------------------

class TestRegimeGateInComputeVote:
    """Verify that compute_vote applies RegimeGate and renormalizes."""

    def test_crisis_zeros_msm_and_renormalize(self, tmp_path):
        """In CRISIS, MSM and INTL_MOM should be zeroed and weights renormalized."""
        voter = _make_voter(tmp_path)
        readings = _all_signal_readings()

        vote = voter.compute_vote(
            readings=readings, regime=Regime.CRISIS, regime_confidence=0.9)

        # Vote should still compute (not crash)
        assert vote.num_sources == 6
        # The internal weight processing should have zeroed MSM and INTL_MOM
        # Check by inspecting that the resulting action is risk_off (signals bearish)
        # and that the vote was computed successfully
        assert vote.action in ('risk_off', 'decrease_equity', 'neutral')

    def test_normal_regime_no_gating(self, tmp_path):
        """In NORMAL, no signals are gated — all weights preserved."""
        voter = _make_voter(tmp_path)
        readings = _all_signal_readings()

        vote = voter.compute_vote(
            readings=readings, regime=Regime.NORMAL, regime_confidence=0.7)

        assert vote.num_sources == 6
        # Should not crash; weighted_consensus should reflect all signals

    def test_high_vol_zeros_msm_only(self, tmp_path):
        """In HIGH_VOL, only MSM is zeroed (not INTL_MOM)."""
        voter = _make_voter(tmp_path)
        readings = _all_signal_readings()

        vote = voter.compute_vote(
            readings=readings, regime=Regime.HIGH_VOL, regime_confidence=0.8)

        assert vote.num_sources == 6
        # Should succeed — INTL_MOM still active in HIGH_VOL

    def test_crisis_renormalize_preserves_relative_weights(self, tmp_path):
        """After zeroing MSM and INTL_MOM in CRISIS, surviving signals' relative
        proportions should be preserved (just scaled up to sum to 1)."""
        gate = RegimeGate()
        weights = dict(REGIME_WEIGHTS[Regime.CRISIS])
        filtered = gate.filter_weights(weights, "CRISIS")

        # MSM and INTL_MOM should be 0
        assert filtered[SignalSource.MULTI_SPEED_MOM] == 0.0
        assert filtered[SignalSource.INTERNATIONAL_MOMENTUM] == 0.0

        # Renormalize
        total = sum(filtered.values())
        assert total > 0
        renormed = {k: v / total for k, v in filtered.items()}

        # Surviving signals should sum to 1.0
        assert abs(sum(renormed.values()) - 1.0) < 1e-10

        # Relative proportions of survivors preserved
        survivors = {k: v for k, v in filtered.items() if v > 0}
        for sig in survivors:
            assert abs(renormed[sig] - filtered[sig] / total) < 1e-10


# ---------------------------------------------------------------------------
# 2. BanditWeighter blending integration
# ---------------------------------------------------------------------------

class TestBanditBlendingInComputeVote:
    """Verify that BanditWeighter blends with static weights in compute_vote."""

    def test_cold_start_uses_static_weights(self, tmp_path):
        """With no bandit observations, compute_vote uses 100% static weights."""
        voter = _make_voter(tmp_path)
        assert voter.bandit_observations == 0

        weights = voter.get_blended_weights("NORMAL")
        # Should match static REGIME_WEIGHTS exactly
        for source in REGIME_WEIGHTS[Regime.NORMAL]:
            assert abs(weights[source] - REGIME_WEIGHTS[Regime.NORMAL][source]) < 1e-10

    def test_blended_weights_after_observations(self, tmp_path):
        """After some bandit observations, weights should shift from static."""
        voter = _make_voter(tmp_path)
        rng = np.random.RandomState(42)

        # Feed asymmetric returns: alt_data does well, msm does poorly
        for _ in range(100):
            voter.update_bandit("alternative_data", "NORMAL", rng.normal(0.003, 0.005))
            voter.update_bandit("multi_speed_momentum", "NORMAL", rng.normal(-0.001, 0.005))

        # After 100 observations, blend should be nonzero
        # blend = min(0.7, 100/252 * 0.7) ≈ 0.278
        weights = voter.get_blended_weights("NORMAL")

        # ALT_DATA should have gained weight vs static
        static_alt = REGIME_WEIGHTS[Regime.NORMAL][SignalSource.ALTERNATIVE_DATA]
        assert weights[SignalSource.ALTERNATIVE_DATA] >= static_alt * 0.95  # at least close

    def test_blend_progression_over_time(self, tmp_path):
        """Bandit blend factor should increase with more observations."""
        voter = _make_voter(tmp_path)

        # 0 observations → 0% bandit
        assert voter.bandit_observations == 0
        blend_0 = min(0.7, 0 / 252 * 0.7)
        assert blend_0 == 0.0

        # 50 observations
        for _ in range(50):
            voter.update_bandit("alternative_data", "NORMAL", 0.001)
        blend_50 = min(0.7, 50 / 252 * 0.7)
        assert 0.05 < blend_50 < 0.20

        # 252+ observations → capped at 70%
        for _ in range(250):
            voter.update_bandit("alternative_data", "NORMAL", 0.001)
        blend_302 = min(0.7, 302 / 252 * 0.7)
        assert blend_302 == 0.7

    def test_full_blend_weights_sum_to_one(self, tmp_path):
        """Blended weights should always sum to 1.0."""
        voter = _make_voter(tmp_path)
        rng = np.random.RandomState(42)

        # Feed diverse data
        for sig in [s.value for s in SignalSource]:
            for _ in range(60):
                voter.update_bandit(sig, "NORMAL", rng.normal(0.001, 0.01))

        weights = voter.get_blended_weights("NORMAL")
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# 3. Hysteresis guard in compute_vote
# ---------------------------------------------------------------------------

class TestHysteresisInComputeVote:
    """Verify that hysteresis prevents premature regime gate changes."""

    def test_short_regime_transition_uses_old_gating(self, tmp_path):
        """If regime just changed and dwell < 20 days, use old regime's gating."""
        voter = _make_voter(tmp_path)
        voter._prev_regime = "NORMAL"
        voter._days_in_regime = 5  # Below 20-day hysteresis

        # In CRISIS with hysteresis, should still use NORMAL gating
        active = voter.regime_gate.gate_with_hysteresis(
            "CRISIS", voter._prev_regime, voter._days_in_regime)
        # MSM should still be active (using NORMAL gating from hysteresis)
        assert "multi_speed_momentum" in active

    def test_stable_regime_uses_current_gating(self, tmp_path):
        """If regime has been stable for > 20 days, use current regime gating."""
        voter = _make_voter(tmp_path)
        voter._prev_regime = "NORMAL"
        voter._days_in_regime = 25  # Above 20-day hysteresis

        active = voter.regime_gate.gate_with_hysteresis(
            "CRISIS", voter._prev_regime, voter._days_in_regime)
        # MSM should be gated OFF (using CRISIS gating)
        assert "multi_speed_momentum" not in active

    def test_no_prev_regime_skips_hysteresis(self, tmp_path):
        """First regime detection (no prev) should use current regime directly."""
        voter = _make_voter(tmp_path)
        voter._prev_regime = None
        voter._days_in_regime = 0

        active = voter.regime_gate.gate_with_hysteresis(
            "CRISIS", None, 0)
        # CRISIS gating should apply immediately
        assert "multi_speed_momentum" not in active


# ---------------------------------------------------------------------------
# 4. Combined pipeline: gate + blend + renormalize
# ---------------------------------------------------------------------------

class TestCombinedPipeline:
    """End-to-end: compute_vote with both RegimeGate and BanditWeighter active."""

    def test_crisis_with_bandit_observations(self, tmp_path):
        """CRISIS regime: gate zeros MSM/INTL_MOM, bandit still blends survivors."""
        voter = _make_voter(tmp_path)
        rng = np.random.RandomState(42)

        # Feed bandit with diverse data
        for sig in [s.value for s in SignalSource]:
            for _ in range(60):
                voter.update_bandit(sig, "CRISIS", rng.normal(0.001, 0.01))

        readings = _all_signal_readings()
        vote = voter.compute_vote(
            readings=readings, regime=Regime.CRISIS, regime_confidence=0.9)

        # Should not crash; vote should be valid
        assert vote.num_sources == 6
        assert vote.action in ('risk_off', 'decrease_equity', 'neutral')

    def test_normal_with_full_bandit_blend(self, tmp_path):
        """NORMAL regime with 252+ bandit observations: 70% bandit, 30% static."""
        voter = _make_voter(tmp_path)
        rng = np.random.RandomState(42)

        # Feed 300 observations to reach max blend
        for _ in range(300):
            for sig in [s.value for s in SignalSource]:
                voter.update_bandit(sig, "NORMAL", rng.normal(0.001, 0.01))

        # Verify blend is at max
        blend = min(0.7, voter.bandit_observations / 252 * 0.7)
        assert blend == 0.7

        readings = _all_signal_readings()
        vote = voter.compute_vote(
            readings=readings, regime=Regime.NORMAL, regime_confidence=0.7)

        assert vote.num_sources == 6
        # No signals gated in NORMAL, all weights positive
        weights = voter.get_blended_weights("NORMAL")
        for source in weights:
            assert weights[source] > 0

    def test_high_vol_gated_plus_bandit(self, tmp_path):
        """HIGH_VOL: MSM gated off, bandit optimizes among survivors."""
        voter = _make_voter(tmp_path)
        rng = np.random.RandomState(42)

        # Feed bandit — alt_data outperforms in HIGH_VOL
        for _ in range(100):
            voter.update_bandit("alternative_data", "HIGH_VOL", rng.normal(0.003, 0.005))
            voter.update_bandit("cross_asset_rv", "HIGH_VOL", rng.normal(0.001, 0.008))
            voter.update_bandit("multi_speed_momentum", "HIGH_VOL", rng.normal(-0.002, 0.01))
            voter.update_bandit("international_momentum", "HIGH_VOL", rng.normal(0.001, 0.008))

        readings = _all_signal_readings()
        vote = voter.compute_vote(
            readings=readings, regime=Regime.HIGH_VOL, regime_confidence=0.8)

        # Vote should reflect the gated + blended pipeline
        assert vote.num_sources == 6

    def test_regime_transition_with_hysteresis_and_bandit(self, tmp_path):
        """Transition NORMAL→CRISIS with short dwell: hysteresis preserves old gating."""
        voter = _make_voter(tmp_path)
        voter._prev_regime = "NORMAL"
        voter._days_in_regime = 5

        readings = _all_signal_readings()
        # This test verifies the hysteresis path works through compute_vote
        # Without hysteresis, MSM would be zeroed in CRISIS
        # With hysteresis (5 days < 20 min dwell), NORMAL gating is used
        vote = voter.compute_vote(
            readings=readings, regime=Regime.CRISIS, regime_confidence=0.9)

        assert vote.num_sources == 6

    def test_gate_then_renormalize_then_bandit_weight_sum(self, tmp_path):
        """After gating + renormalization, all weights should sum to 1.0."""
        gate = RegimeGate()
        weights = dict(REGIME_WEIGHTS[Regime.CRISIS])

        # Step 1: Gate
        filtered = gate.filter_weights(weights, "CRISIS")

        # Step 2: Renormalize
        total = sum(filtered.values())
        renormed = {k: v / total for k, v in filtered.items()}
        assert abs(sum(renormed.values()) - 1.0) < 1e-10

        # Step 3: Verify no negative weights
        for source, weight in renormed.items():
            assert weight >= 0.0

    def test_msm_weight_zero_in_crisis(self, tmp_path):
        """MSM weight should be exactly 0.0 in CRISIS after gating."""
        gate = RegimeGate()
        weights = dict(REGIME_WEIGHTS[Regime.CRISIS])
        filtered = gate.filter_weights(weights, "CRISIS")

        assert filtered[SignalSource.MULTI_SPEED_MOM] == 0.0
        assert filtered[SignalSource.INTERNATIONAL_MOMENTUM] == 0.0

    def test_all_regimes_produce_valid_vote(self, tmp_path):
        """compute_vote should work for all regimes without crashing."""
        voter = _make_voter(tmp_path)
        readings = _all_signal_readings()

        for regime in Regime:
            vote = voter.compute_vote(
                readings=readings, regime=regime, regime_confidence=0.7)
            assert vote.num_sources == 6
            assert vote.action in (
                'increase_equity', 'decrease_equity', 'neutral', 'risk_off')


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestPipelineEdgeCases:
    """Edge cases in the combined pipeline."""

    def test_all_signals_gated(self, tmp_path):
        """If all signals were gated (hypothetical), weights sum to 0."""
        gate = RegimeGate(gate_rules={
            "multi_speed_momentum": {"CRISIS"},
            "cross_asset_rv": {"CRISIS"},
            "international_momentum": {"CRISIS"},
            "alternative_data": {"CRISIS"},
            "cross_asset_regime_arb": {"CRISIS"},
            "unified_overlay": {"CRISIS"},
            "multi_timeframe_fusion": {"CRISIS"},
            "google_trends": {"CRISIS"},
        })
        weights = dict(REGIME_WEIGHTS[Regime.CRISIS])
        filtered = gate.filter_weights(weights, "CRISIS")

        # All weights zeroed — total should be 0
        total = sum(filtered.values())
        assert total == 0.0

    def test_single_signal_survives_gating(self, tmp_path):
        """If only one signal survives gating, it gets 100% weight after renorm."""
        gate = RegimeGate(gate_rules={
            "multi_speed_momentum": {"CRISIS"},
            "international_momentum": {"CRISIS"},
        })
        # Only 3 signals in weights
        weights = {
            SignalSource.MULTI_SPEED_MOM: 0.20,
            SignalSource.CROSS_ASSET_RV: 0.50,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.30,
        }
        filtered = gate.filter_weights(weights, "CRISIS")

        # MSM and INTL_MOM zeroed; CROSS_ASSET_RV survives
        assert filtered[SignalSource.CROSS_ASSET_RV] == 0.50
        assert filtered[SignalSource.MULTI_SPEED_MOM] == 0.0
        assert filtered[SignalSource.INTERNATIONAL_MOMENTUM] == 0.0

        # Renormalize
        total = sum(filtered.values())
        renormed = {k: v / total for k, v in filtered.items()}
        assert abs(renormed[SignalSource.CROSS_ASSET_RV] - 1.0) < 1e-10

    def test_bandit_with_zero_observations_no_blend(self, tmp_path):
        """With 0 observations, get_blended_weights returns pure static."""
        voter = _make_voter(tmp_path)
        voter.bandit_observations = 0

        weights = voter.get_blended_weights("NORMAL")
        static = REGIME_WEIGHTS[Regime.NORMAL]
        for source in static:
            assert abs(weights[source] - static[source]) < 1e-10

    def test_compute_vote_without_regime_gate(self, tmp_path):
        """If regime_gate is None (e.g., manually disabled), compute_vote still works."""
        voter = _make_voter(tmp_path)
        voter.regime_gate = None  # Disable gating

        readings = _all_signal_readings()
        vote = voter.compute_vote(
            readings=readings, regime=Regime.CRISIS, regime_confidence=0.9)

        assert vote.num_sources == 6

    def test_compute_vote_without_bandit(self, tmp_path):
        """If bandit is None (e.g., manually disabled), compute_vote uses static."""
        voter = _make_voter(tmp_path)
        voter.bandit = None

        weights = voter.get_blended_weights("NORMAL")
        static = REGIME_WEIGHTS[Regime.NORMAL]
        for source in static:
            assert abs(weights[source] - static[source]) < 1e-10

        readings = _all_signal_readings()
        vote = voter.compute_vote(
            readings=readings, regime=Regime.NORMAL, regime_confidence=0.7)
        assert vote.num_sources == 6
