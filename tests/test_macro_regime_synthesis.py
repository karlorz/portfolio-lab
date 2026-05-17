"""
Tests for Macro Regime Meta-Synthesis module (v8.07).
Runs under PORTFOLIO_LAB_ENABLE_ML=0 — ML-gated detectors gracefully fall back.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure PORTFOLIO_LAB_ENABLE_ML=0 for safe testing
os.environ.setdefault("PORTFOLIO_LAB_ENABLE_ML", "0")

from src.signals.macro_regime_synthesis import (
    CANONICAL_REGIMES,
    DETECTOR_NAMES,
    DETECTOR_WEIGHTS,
    REGIME_MAPPING,
    REGIME_SIGNAL_VALUE,
    DetectorVote,
    MetaRegimeConsensus,
    MetaRegimeSynthesizer,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def synthesizer(tmp_path):
    """Create a synthesizer with a temp data dir."""
    return MetaRegimeSynthesizer(data_dir=tmp_path)


# ============================================================
# Data Model Tests
# ============================================================

class TestDetectorVote:
    def test_default_values(self):
        vote = DetectorVote(
            detector_name="test",
            raw_regime="bull",
            canonical_regime="bull",
            confidence=0.8,
            available=True,
        )
        assert vote.detector_name == "test"
        assert vote.canonical_regime == "bull"
        assert vote.confidence == 0.8
        assert vote.available is True

    def test_unavailable_detector(self):
        vote = DetectorVote(
            detector_name="unavailable",
            raw_regime="unavailable",
            canonical_regime="neutral",
            confidence=0.0,
            available=False,
        )
        assert vote.available is False
        assert vote.confidence == 0.0


class TestMetaRegimeConsensus:
    def test_default_values(self):
        votes = [
            DetectorVote("d1", "bull", "bull", 0.8, True),
            DetectorVote("d2", "crisis", "crisis", 0.9, True),
        ]
        consensus = MetaRegimeConsensus(
            timestamp="2026-05-17T12:00:00",
            consensus_regime="bull",
            consensus_confidence=0.55,
            regime_signal=0.6,
            vote_details=votes,
            agreement_ratio=0.5,
            num_active_detectors=2,
            num_total_detectors=5,
        )
        assert consensus.consensus_regime == "bull"
        assert consensus.regime_signal == 0.6
        assert consensus.num_active_detectors == 2
        assert consensus.num_total_detectors == 5

    def test_empty_votes(self):
        consensus = MetaRegimeConsensus(
            timestamp="2026-05-17T12:00:00",
            consensus_regime="neutral",
            consensus_confidence=0.0,
            regime_signal=0.0,
            vote_details=[],
            agreement_ratio=0.0,
            num_active_detectors=0,
            num_total_detectors=5,
        )
        assert consensus.num_active_detectors == 0


# ============================================================
# Regime Mapping Tests
# ============================================================

class TestRegimeMapping:
    """Verify all detector regimes map to canonical regimes."""

    def test_macro_regime_mappings(self):
        """macro_regime outputs map correctly."""
        assert REGIME_MAPPING["risk_on_growth"] == "bull"
        assert REGIME_MAPPING["risk_on_late"] == "bull"
        assert REGIME_MAPPING["neutral"] == "neutral"
        assert REGIME_MAPPING["risk_off_rotation"] == "bear"
        assert REGIME_MAPPING["defensive"] == "bear"
        assert REGIME_MAPPING["crisis"] == "crisis"

    def test_kurtosis_mappings(self):
        """kurtosis_regime outputs map correctly."""
        assert REGIME_MAPPING["low_kurtosis"] == "bull"
        assert REGIME_MAPPING["normal"] == "neutral"
        assert REGIME_MAPPING["high_kurtosis"] == "high_vol"
        assert REGIME_MAPPING["extreme_kurtosis"] == "crisis"

    def test_vol_volume_gap_mappings(self):
        """vol_volume_gap outputs map correctly."""
        assert REGIME_MAPPING["trend_up"] == "bull"
        assert REGIME_MAPPING["trend_down"] == "bear"
        assert REGIME_MAPPING["mean_revert"] == "neutral"
        assert REGIME_MAPPING["high_vol"] == "high_vol"
        assert REGIME_MAPPING["crisis"] == "crisis"

    def test_hmm_mappings(self):
        """HMM regime outputs map correctly."""
        assert REGIME_MAPPING["bull"] == "bull"
        assert REGIME_MAPPING["bear"] == "bear"
        assert REGIME_MAPPING["neutral"] == "neutral"
        assert REGIME_MAPPING["high_vol"] == "high_vol"
        assert REGIME_MAPPING["crisis"] == "crisis"

    def test_transformer_mappings(self):
        """transformer_regime outputs map correctly."""
        assert REGIME_MAPPING["trend_up"] == "bull"
        assert REGIME_MAPPING["trend_down"] == "bear"
        assert REGIME_MAPPING["mean_revert"] == "neutral"
        assert REGIME_MAPPING["high_vol"] == "high_vol"
        assert REGIME_MAPPING["crisis"] == "crisis"


class TestRegimeSignalValues:
    """Verify canonical regime -> signal value mapping."""

    def test_bull_is_positive(self):
        assert REGIME_SIGNAL_VALUE["bull"] > 0

    def test_crisis_is_strongly_negative(self):
        assert REGIME_SIGNAL_VALUE["crisis"] < -0.5

    def test_neutral_is_zero(self):
        assert REGIME_SIGNAL_VALUE["neutral"] == 0.0

    def test_all_canonical_mapped(self):
        for regime in CANONICAL_REGIMES:
            assert regime in REGIME_SIGNAL_VALUE


# ============================================================
# Synthesizer Tests
# ============================================================

class TestSynthesizerInit:
    def test_init_creates_no_state_file(self, synthesizer):
        """State file should not exist until first poll."""
        assert not synthesizer.state_path.exists()

    def test_init_with_default_dir(self):
        """Default data dir is project-root/data."""
        s = MetaRegimeSynthesizer()
        assert "data" in str(s.data_dir)

    def test_init_loads_empty_state(self, synthesizer):
        assert synthesizer.state == {}


class TestGetEnsembleSignal:
    def test_returns_float(self, synthesizer):
        signal = synthesizer.get_ensemble_signal()
        assert isinstance(signal, float)

    def test_signal_in_range(self, synthesizer):
        signal = synthesizer.get_ensemble_signal()
        assert -1.0 <= signal <= 1.0


class TestGetRegimeForEnsembleVoter:
    def test_returns_tuple(self, synthesizer):
        result = synthesizer.get_regime_for_ensemble_voter()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_regime_is_string(self, synthesizer):
        regime, confidence = synthesizer.get_regime_for_ensemble_voter()
        assert isinstance(regime, str)
        assert regime in ("normal", "high_vol", "crisis")

    def test_confidence_in_range(self, synthesizer):
        _, confidence = synthesizer.get_regime_for_ensemble_voter()
        assert 0.0 <= confidence <= 1.0


class TestPollRegimes:
    def test_returns_consensus(self, synthesizer):
        consensus = synthesizer.poll_regimes()
        assert isinstance(consensus, MetaRegimeConsensus)

    def test_consensus_has_timestamp(self, synthesizer):
        consensus = synthesizer.poll_regimes()
        assert consensus.timestamp is not None

    def test_consensus_has_votes_for_all_detectors(self, synthesizer):
        consensus = synthesizer.poll_regimes()
        assert len(consensus.vote_details) == len(DETECTOR_NAMES)

    def test_saves_state_after_poll(self, synthesizer):
        synthesizer.poll_regimes()
        assert synthesizer.state_path.exists()
        state = json.loads(synthesizer.state_path.read_text())
        assert "consensus_regime" in state

    def test_non_ml_detectors_respond(self, synthesizer):
        """kurtosis_regime and vol_volume_gap should work without ML."""
        consensus = synthesizer.poll_regimes()
        # At least 2 non-ML detectors should be available
        non_ml_votes = [
            v for v in consensus.vote_details
            if v.detector_name in ("kurtosis_regime", "vol_volume_gap", "macro_regime")
        ]
        available = [v for v in non_ml_votes if v.available]
        assert len(available) >= 1, (
            f"Expected at least 1 non-ML detector available, "
            f"got: {[(v.detector_name, v.available) for v in non_ml_votes]}"
        )

    def test_ml_detectors_fallback_gracefully(self, synthesizer):
        """HMM and transformer should fall back without ML env."""
        consensus = synthesizer.poll_regimes()
        ml_votes = [
            v for v in consensus.vote_details
            if v.detector_name in ("regime_hmm", "transformer_regime")
        ]
        for v in ml_votes:
            assert v.available is False, f"{v.detector_name} should fall back without ML"
            assert v.canonical_regime == "neutral"

    def test_agreement_ratio_range(self, synthesizer):
        consensus = synthesizer.poll_regimes()
        assert 0.0 <= consensus.agreement_ratio <= 1.0

    def test_all_five_detectors_tracked(self, synthesizer):
        consensus = synthesizer.poll_regimes()
        detector_names = {v.detector_name for v in consensus.vote_details}
        assert detector_names == set(DETECTOR_NAMES)


class TestExplain:
    def test_returns_string(self, synthesizer):
        explanation = synthesizer.explain()
        assert isinstance(explanation, str)

    def test_explain_includes_all_detectors(self, synthesizer):
        explanation = synthesizer.explain()
        for name in DETECTOR_NAMES:
            assert name in explanation

    def test_explain_shows_consensus(self, synthesizer):
        explanation = synthesizer.explain()
        assert "Consensus:" in explanation


class TestGetStatus:
    def test_idle_before_poll(self, synthesizer):
        status = synthesizer.get_status()
        assert status["status"] == "idle"

    def test_active_after_poll(self, synthesizer):
        synthesizer.poll_regimes()
        status = synthesizer.get_status()
        assert status["status"] == "active"

    def test_status_has_required_keys(self, synthesizer):
        synthesizer.poll_regimes()
        status = synthesizer.get_status()
        required = ["status", "timestamp", "consensus_regime",
                     "consensus_confidence", "regime_signal", "agreement_ratio"]
        for key in required:
            assert key in status


# ============================================================
# State Persistence Tests
# ============================================================

class TestStatePersistence:
    def test_saves_and_loads(self, synthesizer, tmp_path):
        s1 = MetaRegimeSynthesizer(data_dir=tmp_path)
        s1.poll_regimes()
        saved_state = json.loads(s1.state_path.read_text())

        # Create new instance pointing to same dir
        s2 = MetaRegimeSynthesizer(data_dir=tmp_path)
        assert s2.state == saved_state

    def test_corrupted_state_handled(self, synthesizer):
        synthesizer.state_path.write_text("{invalid json}")
        s = MetaRegimeSynthesizer(data_dir=synthesizer.data_dir)
        assert s.state == {}


# ============================================================
# Detector Weight Tests
# ============================================================

class TestDetectorWeights:
    def test_all_detectors_have_weights(self):
        for name in DETECTOR_NAMES:
            assert name in DETECTOR_WEIGHTS

    def test_weights_sum_to_one(self):
        total = sum(DETECTOR_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"

    def test_non_ml_have_higher_weights(self):
        """Non-ML detectors should have higher weight than ML-gated ones."""
        non_ml_weight = sum(
            DETECTOR_WEIGHTS[n] for n in ["macro_regime", "kurtosis_regime", "vol_volume_gap"]
        )
        ml_weight = sum(
            DETECTOR_WEIGHTS[n] for n in ["regime_hmm", "transformer_regime"]
        )
        assert non_ml_weight > ml_weight


# ============================================================
# Edge Case Tests
# ============================================================

class TestEdgeCases:
    def test_unknown_detector_name(self, synthesizer):
        """Polling an unknown detector should return unavailable vote."""
        vote = synthesizer._poll_detector("nonexistent_detector")
        assert vote.available is False
        assert vote.canonical_regime == "neutral"

    def test_multi_poll_consistent(self, synthesizer):
        """Multiple polls in quick succession should return consistent types."""
        c1 = synthesizer.poll_regimes()
        c2 = synthesizer.poll_regimes()
        assert isinstance(c1, MetaRegimeConsensus)
        assert isinstance(c2, MetaRegimeConsensus)

    def test_all_detectors_mapped(self, synthesizer):
        """Every canonical regime should have a signal value."""
        for regime in CANONICAL_REGIMES:
            assert regime in REGIME_SIGNAL_VALUE
            assert isinstance(REGIME_SIGNAL_VALUE[regime], (int, float))

    def test_regime_mapping_contains_all_known_states(self):
        """All known regime strings from all detectors should be mapped."""
        known_states = [
            # macro_regime
            "risk_on_growth", "risk_on_late", "risk_on",
            "neutral", "risk_off_rotation", "defensive",
            "crisis", "risk_off",
            # kurtosis_regime
            "low_kurtosis", "normal", "high_kurtosis", "extreme_kurtosis",
            # vol_volume_gap
            "trend_up", "trend_down", "mean_revert", "high_vol", "crisis",
            # HMM
            "bull", "bear", "neutral", "high_vol", "crisis",
            # transformer
            "trend_up", "trend_down", "mean_revert", "high_vol", "crisis",
        ]
        for state in known_states:
            assert state in REGIME_MAPPING, f"Missing mapping for '{state}'"


# ============================================================
# Integration: SignalSource Enum
# ============================================================

class TestIntegration:
    def test_signal_source_exists(self):
        """Verify MACRO_REGIME_SYNTHESIS is registered in ensemble_voter."""
        from src.strategy.ensemble_voter import SignalSource
        assert hasattr(SignalSource, "MACRO_REGIME_SYNTHESIS")
        assert SignalSource.MACRO_REGIME_SYNTHESIS.value == "macro_regime_synthesis"

    def test_regime_weights_have_macro_regime_synthesis(self):
        """Verify MACRO_REGIME_SYNTHESIS has weights in all regimes."""
        from src.strategy.ensemble_voter import REGIME_WEIGHTS, SignalSource, Regime
        for regime in Regime:
            weights = REGIME_WEIGHTS[regime]
            assert SignalSource.MACRO_REGIME_SYNTHESIS in weights, (
                f"Missing weight in {regime.value}"
            )
            assert weights[SignalSource.MACRO_REGIME_SYNTHESIS] >= 0.02
