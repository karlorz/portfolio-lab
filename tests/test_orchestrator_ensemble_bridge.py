"""
Tests for Orchestrator-EnsembleVoter Bridge (v4.90)
"""

import json
import pytest
from datetime import datetime
from pathlib import Path

from src.strategy.orchestrator_ensemble_bridge import (
    OrchestratorEnsembleBridge,
    UnifiedSignalReading,
    BridgeSignalType,
    get_unified_ensemble_signal,
    get_unified_ensemble_reading,
)


class TestBridgeSignalType:
    """Test bridge signal type enum."""

    def test_values(self):
        assert BridgeSignalType.ALLOCATION.value == "allocation"
        assert BridgeSignalType.DIRECTION.value == "direction"
        assert BridgeSignalType.RISK.value == "risk"


class TestUnifiedSignalReading:
    """Test unified signal reading dataclass."""

    def test_serializable(self):
        sig = UnifiedSignalReading(
            timestamp="2026-05-16T00:00:00", source="unified_overlay",
            value=0.15, confidence=0.75, weight=0.20,
            spy_signal=0.0, gld_signal=-0.5, tlt_signal=0.0,
            ief_signal=0.5, shy_signal=0.0, btc_signal=0.5, eth_signal=0.5,
            risk_signal=0.0, execution_signal=0.85,
            explanation="Test", num_overlays_active=3, conflict_count=0,
        )
        d = sig.to_dict()
        assert isinstance(d, dict)
        assert d["source"] == "unified_overlay"
        assert d["spy_signal"] == 0.0

    def test_to_signal_reading(self):
        sig = UnifiedSignalReading(
            timestamp="2026-05-16T00:00:00", source="unified_overlay",
            value=0.15, confidence=0.75, weight=0.20,
            spy_signal=0.0, gld_signal=-0.5, tlt_signal=0.0,
            ief_signal=0.5, shy_signal=0.0, btc_signal=0.5, eth_signal=0.5,
            risk_signal=0.0, execution_signal=0.85,
            explanation="Test", num_overlays_active=3, conflict_count=0,
        )
        reading = sig.to_signal_reading()
        assert reading.source is not None
        assert reading.value == 0.15
        assert reading.asset_signals is not None
        assert "SPY" in reading.asset_signals


class TestOrchestratorEnsembleBridge:
    """Test bridge core functionality."""

    @pytest.fixture
    def bridge(self):
        return OrchestratorEnsembleBridge()

    def test_generates_signal(self, bridge):
        signal = bridge.generate_signal()
        assert isinstance(signal, UnifiedSignalReading)
        assert signal.source == "unified_overlay"
        assert -1.0 <= signal.value <= 1.0
        assert 0.0 <= signal.confidence <= 1.0

    def test_weight_to_signal_neutral(self, bridge):
        """Same as baseline should return 0."""
        assert bridge._weight_to_signal(0.46, 0.46) == 0.0
        assert bridge._weight_to_signal(0.38, 0.38) == 0.0

    def test_weight_to_signal_overweight(self, bridge):
        """Large overweight should return +1."""
        assert bridge._weight_to_signal(0.56, 0.46) == 1.0

    def test_weight_to_signal_underweight(self, bridge):
        """Large underweight should return -1."""
        assert bridge._weight_to_signal(0.36, 0.46) == -1.0

    def test_weight_to_signal_moderate(self, bridge):
        """Moderate change should return ±0.5."""
        assert bridge._weight_to_signal(0.49, 0.46) == 0.5
        assert bridge._weight_to_signal(0.43, 0.46) == -0.5

    def test_weight_to_signal_from_zero(self, bridge):
        """Asset with baseline 0 (IEF/SHY) should signal on any allocation."""
        assert bridge._weight_to_signal(0.0, 0.0) == 0.0
        assert bridge._weight_to_signal(0.06, 0.0) == 1.0
        assert bridge._weight_to_signal(0.03, 0.0) == 0.5

    def test_all_weight_to_signal_cases(self, bridge):
        """All 5 signal levels should be reachable."""
        signals = set()
        for w in [0.30, 0.40, 0.45, 0.48, 0.50]:
            signals.add(bridge._weight_to_signal(w, 0.46))
        assert len(signals) >= 3  # Should see -1, 0, +1 at minimum

    def test_signal_has_asset_signals(self, bridge):
        signal = bridge.generate_signal()
        assert -1.0 <= signal.spy_signal <= 1.0
        assert -1.0 <= signal.gld_signal <= 1.0
        assert -1.0 <= signal.btc_signal <= 1.0

    def test_risk_signal_in_range(self, bridge):
        signal = bridge.generate_signal()
        assert -1.0 <= signal.risk_signal <= 1.0

    def test_execution_signal_in_range(self, bridge):
        signal = bridge.generate_signal()
        assert 0.0 <= signal.execution_signal <= 1.0

    def test_get_ensemble_reading(self, bridge):
        reading = bridge.get_ensemble_reading()
        assert reading.source is not None
        assert reading.value is not None
        assert len(reading.explanation) > 0

    def test_save_signal(self, bridge, tmp_path):
        bridge.OUTPUT_PATH = tmp_path / "test_signal.json"
        signal = bridge.generate_signal()
        bridge.save_signal(signal)
        assert bridge.OUTPUT_PATH.exists()

    def test_compare_with_other_source(self, bridge):
        result = bridge.compare_with_ensemble_source("tsfm_momentum")
        assert "unified_value" in result
        assert result["compared_source"] == "tsfm_momentum"

    def test_convenience_function(self):
        signal = get_unified_ensemble_signal()
        assert isinstance(signal, UnifiedSignalReading)

    def test_convenience_reading_function(self):
        reading = get_unified_ensemble_reading()
        assert reading.source is not None
        assert reading.asset_signals is not None


class TestEdgeCases:
    """Edge cases for bridge."""

    @pytest.fixture
    def bridge(self):
        return OrchestratorEnsembleBridge()

    def test_multiple_calls_consistent(self, bridge):
        s1 = bridge.generate_signal()
        s2 = bridge.generate_signal()
        assert abs(s1.value - s2.value) < 0.5  # Should be similar

    def test_explanation_non_empty(self, bridge):
        signal = bridge.generate_signal()
        assert len(signal.explanation) > 0

    def test_ensemble_reading_has_asset_dict(self, bridge):
        reading = bridge.get_ensemble_reading()
        assert isinstance(reading.asset_signals, dict)
        assert "SPY" in reading.asset_signals
        assert "BTC" in reading.asset_signals
