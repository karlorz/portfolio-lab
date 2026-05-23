"""
Tests for Orchestrator-EnsembleVoter Bridge (v4.90).

Covers: _weight_to_signal threshold boundaries, UnifiedSignalReading
serialization (to_dict, to_signal_reading),
and compare_with_ensemble_source edge cases.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.strategy.orchestrator_ensemble_bridge import (
    UnifiedSignalReading,
    OrchestratorEnsembleBridge,
)
from src.strategy.ensemble_voter import SignalSource, SignalReading


# ── _weight_to_signal Tests ───────────────────────────────────────────────


class TestWeightToSignal:
    """Test the static _weight_to_signal threshold boundaries."""

    def test_strong_overweight(self):
        # delta > 0.05 → +1.0
        assert OrchestratorEnsembleBridge._weight_to_signal(0.12, 0.0) == 1.0
        assert OrchestratorEnsembleBridge._weight_to_signal(0.06, 0.0) == 1.0
        # Exact boundary: delta = 0.05 → NOT > 0.05, falls to moderate
        assert OrchestratorEnsembleBridge._weight_to_signal(0.05, 0.0) == 0.5

    def test_moderate_overweight(self):
        # delta 0.02 to 0.05 → +0.5
        assert OrchestratorEnsembleBridge._weight_to_signal(0.04, 0.0) == 0.5
        assert OrchestratorEnsembleBridge._weight_to_signal(0.021, 0.0) == 0.5
        # Exact boundary: delta = 0.02 → NOT > 0.02, falls to neutral
        assert OrchestratorEnsembleBridge._weight_to_signal(0.02, 0.0) == 0.0

    def test_neutral(self):
        # delta -0.02 to +0.02 → 0.0
        assert OrchestratorEnsembleBridge._weight_to_signal(0.01, 0.0) == 0.0
        assert OrchestratorEnsembleBridge._weight_to_signal(0.0, 0.0) == 0.0
        assert OrchestratorEnsembleBridge._weight_to_signal(-0.01, 0.0) == 0.0
        # Boundaries
        assert OrchestratorEnsembleBridge._weight_to_signal(0.02, 0.0) == 0.0
        assert OrchestratorEnsembleBridge._weight_to_signal(-0.02, 0.0) == -0.5

    def test_moderate_underweight(self):
        # delta -0.05 to -0.02 → -0.5
        assert OrchestratorEnsembleBridge._weight_to_signal(-0.03, 0.0) == -0.5
        assert OrchestratorEnsembleBridge._weight_to_signal(-0.04, 0.0) == -0.5
        # Exact boundary: delta = -0.05 → NOT > -0.05, falls to strong
        assert OrchestratorEnsembleBridge._weight_to_signal(-0.05, 0.0) == -1.0

    def test_strong_underweight(self):
        # delta < -0.05 → -1.0
        assert OrchestratorEnsembleBridge._weight_to_signal(-0.06, 0.0) == -1.0
        assert OrchestratorEnsembleBridge._weight_to_signal(-0.20, 0.0) == -1.0

    def test_edge_exact_boundaries(self):
        """Exact threshold boundaries for all 5 zones."""
        assert OrchestratorEnsembleBridge._weight_to_signal(0.051, 0.0) == 1.0
        assert OrchestratorEnsembleBridge._weight_to_signal(0.05, 0.0) == 0.5
        assert OrchestratorEnsembleBridge._weight_to_signal(0.021, 0.0) == 0.5
        assert OrchestratorEnsembleBridge._weight_to_signal(0.02, 0.0) == 0.0
        assert OrchestratorEnsembleBridge._weight_to_signal(0.0, 0.0) == 0.0
        assert OrchestratorEnsembleBridge._weight_to_signal(-0.02, 0.0) == -0.5
        assert OrchestratorEnsembleBridge._weight_to_signal(-0.021, 0.0) == -0.5
        assert OrchestratorEnsembleBridge._weight_to_signal(-0.05, 0.0) == -1.0
        assert OrchestratorEnsembleBridge._weight_to_signal(-0.051, 0.0) == -1.0

    def test_with_nonzero_baseline(self):
        """Delta is computed as current - baseline."""
        # current 0.46, baseline 0.46 → delta 0 → neutral
        assert OrchestratorEnsembleBridge._weight_to_signal(0.46, 0.46) == 0.0
        # current 0.51, baseline 0.46 → delta 0.05 → moderate overweight
        assert OrchestratorEnsembleBridge._weight_to_signal(0.51, 0.46) == 0.5
        # current 0.52, baseline 0.46 → delta 0.06 → strong overweight
        assert OrchestratorEnsembleBridge._weight_to_signal(0.52, 0.46) == 1.0
        # current 0.41, baseline 0.46 → delta -0.05 → strong underweight
        assert OrchestratorEnsembleBridge._weight_to_signal(0.41, 0.46) == -1.0


# ── UnifiedSignalReading Tests ─────────────────────────────────────────────


class TestUnifiedSignalReading:
    def test_create(self):
        signal = UnifiedSignalReading(
            timestamp="2026-05-23T00:00:00",
            source="unified_overlay",
            value=0.15,
            confidence=0.8,
            weight=0.20,
            spy_signal=0.2,
            gld_signal=0.0,
            tlt_signal=-0.1,
            ief_signal=0.0,
            shy_signal=0.0,
            btc_signal=0.0,
            eth_signal=0.0,
            risk_signal=0.0,
            execution_signal=0.5,
            explanation="Test signal",
            num_overlays_active=2,
            conflict_count=1,
        )
        assert signal.source == "unified_overlay"
        assert signal.value == 0.15
        assert signal.confidence == 0.8
        assert signal.spy_signal == 0.2
        assert signal.risk_signal == 0.0

    def test_to_dict(self):
        signal = UnifiedSignalReading(
            timestamp="2026-05-23T00:00:00",
            source="test",
            value=0.1,
            confidence=0.9,
            weight=0.20,
            spy_signal=0.0,
            gld_signal=0.0,
            tlt_signal=0.0,
            ief_signal=0.0,
            shy_signal=0.0,
            btc_signal=0.0,
            eth_signal=0.0,
            risk_signal=0.0,
            execution_signal=0.0,
            explanation="",
            num_overlays_active=0,
            conflict_count=0,
        )
        d = signal.to_dict()
        assert isinstance(d, dict)
        assert d["source"] == "test"
        assert d["value"] == 0.1
        assert d["confidence"] == 0.9

    def test_to_signal_reading(self):
        signal = UnifiedSignalReading(
            timestamp="2026-05-23T00:00:00",
            source="unified_overlay",
            value=0.3,
            confidence=0.75,
            weight=0.20,
            spy_signal=0.5,
            gld_signal=-0.2,
            tlt_signal=0.1,
            ief_signal=0.0,
            shy_signal=0.0,
            btc_signal=0.0,
            eth_signal=0.0,
            risk_signal=-0.3,
            execution_signal=0.8,
            explanation="Bridge test",
            num_overlays_active=3,
            conflict_count=2,
        )
        reading = signal.to_signal_reading()
        assert isinstance(reading, SignalReading)
        assert reading.source == SignalSource.UNIFIED_OVERLAY
        assert reading.value == 0.3
        assert reading.confidence == 0.75
        assert reading.weight == 0.20
        assert reading.regime_fit == "all"
        assert reading.explanation == "Bridge test"
        assert reading.asset_signals["SPY"] == 0.5
        assert reading.asset_signals["GLD"] == -0.2
        assert reading.asset_signals["TLT"] == 0.1

    def test_to_signal_reading_all_assets_present(self):
        signal = UnifiedSignalReading(
            timestamp="2026-05-23T00:00:00",
            source="unified_overlay",
            value=0.0,
            confidence=0.5,
            weight=0.20,
            spy_signal=0.0,
            gld_signal=0.0,
            tlt_signal=0.0,
            ief_signal=0.0,
            shy_signal=0.0,
            btc_signal=0.0,
            eth_signal=0.0,
            risk_signal=0.0,
            execution_signal=0.0,
            explanation="",
            num_overlays_active=0,
            conflict_count=0,
        )
        reading = signal.to_signal_reading()
        assert set(reading.asset_signals.keys()) == {"SPY", "GLD", "TLT", "IEF", "SHY", "BTC", "ETH"}


# ── OrchestratorEnsembleBridge Tests ──────────────────────────────────────


class TestBridgeInit:
    def test_recommended_weight(self):
        bridge = OrchestratorEnsembleBridge()
        assert bridge.RECOMMENDED_ENSEMBLE_WEIGHT == 0.20

    def test_output_path(self):
        bridge = OrchestratorEnsembleBridge()
        assert bridge.OUTPUT_PATH.name == "unified_ensemble_signal.json"


# ── compare_with_ensemble_source Tests ────────────────────────────────────


class TestCompareWithEnsembleSource:
    def test_returns_dict_with_keys(self, monkeypatch):
        """Test that compare_with_ensemble_source returns expected structure."""
        bridge = OrchestratorEnsembleBridge()

        # We need to mock generate_signal since it depends on UnifiedOrchestrator
        # which requires a database and live data.
        mock_signal = UnifiedSignalReading(
            timestamp="2026-05-23T00:00:00",
            source="unified_overlay",
            value=0.12,
            confidence=0.85,
            weight=0.20,
            spy_signal=0.3,
            gld_signal=0.1,
            tlt_signal=-0.1,
            ief_signal=0.0,
            shy_signal=0.0,
            btc_signal=0.0,
            eth_signal=0.0,
            risk_signal=0.0,
            execution_signal=0.5,
            explanation="Mock",
            num_overlays_active=2,
            conflict_count=1,
        )

        with patch.object(bridge, 'generate_signal', return_value=mock_signal):
            result = bridge.compare_with_ensemble_source("test_source")

        assert isinstance(result, dict)
        assert result["unified_value"] == 0.12
        assert result["unified_confidence"] == 0.85
        assert result["compared_source"] == "test_source"
        assert result["active_overlays"] == 2
        assert result["conflicts"] == 1
        assert result["recommendation"] == "integrate"

    def test_standalone_recommendation_when_few_active(self, monkeypatch):
        bridge = OrchestratorEnsembleBridge()

        mock_signal = UnifiedSignalReading(
            timestamp="2026-05-23T00:00:00",
            source="unified_overlay",
            value=0.0,
            confidence=0.5,
            weight=0.20,
            spy_signal=0.0, gld_signal=0.0, tlt_signal=0.0,
            ief_signal=0.0, shy_signal=0.0, btc_signal=0.0, eth_signal=0.0,
            risk_signal=0.0,
            execution_signal=0.0,
            explanation="Mock",
            num_overlays_active=1,
            conflict_count=0,
        )

        with patch.object(bridge, 'generate_signal', return_value=mock_signal):
            result = bridge.compare_with_ensemble_source("other")

        assert result["recommendation"] == "standalone"
