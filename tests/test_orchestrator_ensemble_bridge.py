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


class TestWeightToSignalExtended:
    """Additional _weight_to_signal edge cases."""

    def test_large_positive_delta(self):
        """Very large positive delta should still return +1.0."""
        assert OrchestratorEnsembleBridge._weight_to_signal(1.0, 0.0) == 1.0

    def test_large_negative_delta(self):
        """Very large negative delta should still return -1.0."""
        assert OrchestratorEnsembleBridge._weight_to_signal(-1.0, 0.0) == -1.0

    def test_symmetry(self):
        """Positive and negative deltas of same magnitude should give symmetric results."""
        assert (OrchestratorEnsembleBridge._weight_to_signal(0.03, 0.0) ==
                -OrchestratorEnsembleBridge._weight_to_signal(-0.03, 0.0))

    def test_moderate_underweight_boundary(self):
        """Delta = -0.021 should give -0.5."""
        assert OrchestratorEnsembleBridge._weight_to_signal(-0.021, 0.0) == -0.5

    def test_moderate_overweight_boundary(self):
        """Delta = 0.049 should give +0.5."""
        assert OrchestratorEnsembleBridge._weight_to_signal(0.049, 0.0) == 0.5

    def test_zero_current_zero_baseline(self):
        """Both zero → delta 0 → neutral."""
        assert OrchestratorEnsembleBridge._weight_to_signal(0.0, 0.0) == 0.0


class TestUnifiedSignalReadingExtended:
    """Additional UnifiedSignalReading edge cases."""

    def test_to_dict_serializable(self):
        """to_dict should produce JSON-serializable output."""
        signal = UnifiedSignalReading(
            timestamp="2026-05-23T00:00:00", source="test", value=0.1,
            confidence=0.9, weight=0.20,
            spy_signal=0.0, gld_signal=0.0, tlt_signal=0.0,
            ief_signal=0.0, shy_signal=0.0, btc_signal=0.0, eth_signal=0.0,
            risk_signal=0.0, execution_signal=0.0,
            explanation="", num_overlays_active=0, conflict_count=0,
        )
        d = signal.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_to_signal_reading_source_mapping(self):
        """to_signal_reading should always map to UNIFIED_OVERLAY source."""
        signal = UnifiedSignalReading(
            timestamp="2026-05-23T00:00:00", source="custom_source",
            value=0.5, confidence=0.7, weight=0.20,
            spy_signal=0.1, gld_signal=-0.1, tlt_signal=0.0,
            ief_signal=0.0, shy_signal=0.0, btc_signal=0.0, eth_signal=0.0,
            risk_signal=0.0, execution_signal=0.0,
            explanation="Custom", num_overlays_active=1, conflict_count=0,
        )
        reading = signal.to_signal_reading()
        assert reading.source == SignalSource.UNIFIED_OVERLAY

    def test_negative_signals_in_asset_map(self):
        """Negative signals should be preserved in asset_signals."""
        signal = UnifiedSignalReading(
            timestamp="2026-05-23T00:00:00", source="test",
            value=-0.3, confidence=0.6, weight=0.20,
            spy_signal=-0.5, gld_signal=-0.2, tlt_signal=-0.1,
            ief_signal=-0.05, shy_signal=0.0, btc_signal=-0.1, eth_signal=-0.05,
            risk_signal=-0.4, execution_signal=-0.2,
            explanation="Bearish", num_overlays_active=3, conflict_count=0,
        )
        reading = signal.to_signal_reading()
        assert reading.asset_signals["SPY"] == -0.5
        assert reading.asset_signals["GLD"] == -0.2

    def test_all_signal_fields_present_in_dict(self):
        """to_dict should include all fields."""
        signal = UnifiedSignalReading(
            timestamp="2026-05-23T00:00:00", source="test",
            value=0.1, confidence=0.8, weight=0.20,
            spy_signal=0.3, gld_signal=0.1, tlt_signal=-0.05,
            ief_signal=0.0, shy_signal=0.0, btc_signal=0.1, eth_signal=0.05,
            risk_signal=-0.1, execution_signal=0.2,
            explanation="All fields", num_overlays_active=4, conflict_count=1,
        )
        d = signal.to_dict()
        expected_fields = {
            "timestamp", "source", "value", "confidence", "weight",
            "spy_signal", "gld_signal", "tlt_signal", "ief_signal",
            "shy_signal", "btc_signal", "eth_signal", "risk_signal",
            "execution_signal", "explanation", "num_overlays_active",
            "conflict_count",
        }
        assert expected_fields.issubset(set(d.keys()))


class TestBridgeConvenienceFunctions:
    """Test standalone convenience functions."""

    def test_get_unified_ensemble_signal(self):
        """get_unified_ensemble_signal should return UnifiedSignalReading."""
        from src.strategy.orchestrator_ensemble_bridge import get_unified_ensemble_signal
        with patch.object(OrchestratorEnsembleBridge, 'generate_signal') as mock_gen:
            mock_gen.return_value = UnifiedSignalReading(
                timestamp="2026-05-23T00:00:00", source="test",
                value=0.1, confidence=0.8, weight=0.20,
                spy_signal=0.0, gld_signal=0.0, tlt_signal=0.0,
                ief_signal=0.0, shy_signal=0.0, btc_signal=0.0, eth_signal=0.0,
                risk_signal=0.0, execution_signal=0.0,
                explanation="", num_overlays_active=0, conflict_count=0,
            )
            result = get_unified_ensemble_signal()
            assert isinstance(result, UnifiedSignalReading)

    def test_get_unified_ensemble_reading(self):
        """get_unified_ensemble_reading should return SignalReading."""
        from src.strategy.orchestrator_ensemble_bridge import get_unified_ensemble_reading
        with patch.object(OrchestratorEnsembleBridge, 'get_ensemble_reading') as mock_gen:
            mock_reading = MagicMock(spec=SignalReading)
            mock_gen.return_value = mock_reading
            result = get_unified_ensemble_reading()
            assert result is mock_reading


class TestCompareWithEnsembleSourceExtended:
    """Additional compare_with_ensemble_source edge cases."""

    def test_integrate_recommendation_when_active_overlays_gte_2(self):
        """Active overlays >= 2 should recommend 'integrate'."""
        bridge = OrchestratorEnsembleBridge()
        mock_signal = UnifiedSignalReading(
            timestamp="2026-05-23T00:00:00", source="test",
            value=0.1, confidence=0.7, weight=0.20,
            spy_signal=0.0, gld_signal=0.0, tlt_signal=0.0,
            ief_signal=0.0, shy_signal=0.0, btc_signal=0.0, eth_signal=0.0,
            risk_signal=0.0, execution_signal=0.0,
            explanation="", num_overlays_active=2, conflict_count=0,
        )
        with patch.object(bridge, 'generate_signal', return_value=mock_signal):
            result = bridge.compare_with_ensemble_source("other")
        assert result["recommendation"] == "integrate"

    def test_result_contains_expected_keys(self):
        """Result dict should contain all expected keys."""
        bridge = OrchestratorEnsembleBridge()
        mock_signal = UnifiedSignalReading(
            timestamp="2026-05-23T00:00:00", source="test",
            value=0.1, confidence=0.7, weight=0.20,
            spy_signal=0.0, gld_signal=0.0, tlt_signal=0.0,
            ief_signal=0.0, shy_signal=0.0, btc_signal=0.0, eth_signal=0.0,
            risk_signal=0.0, execution_signal=0.0,
            explanation="", num_overlays_active=2, conflict_count=0,
        )
        with patch.object(bridge, 'generate_signal', return_value=mock_signal):
            result = bridge.compare_with_ensemble_source("test_src")
        expected_keys = {
            "unified_value", "unified_confidence", "compared_source",
            "active_overlays", "conflicts", "recommendation",
        }
        assert expected_keys.issubset(set(result.keys()))


class TestExports:
    """Module __all__ exports validation."""

    def test_all_exports(self):
        import src.strategy.orchestrator_ensemble_bridge as mod
        expected = {'UnifiedSignalReading', 'OrchestratorEnsembleBridge',
                    'get_unified_ensemble_signal', 'get_unified_ensemble_reading'}
        assert expected.issubset(set(mod.__all__))

    def test_all_exports_importable(self):
        from src.strategy.orchestrator_ensemble_bridge import (
            UnifiedSignalReading, OrchestratorEnsembleBridge,
            get_unified_ensemble_signal, get_unified_ensemble_reading,
        )
        assert UnifiedSignalReading is not None
        assert OrchestratorEnsembleBridge is not None


class TestConstants:
    """Module-level constants validation."""

    def test_recommended_weight_range(self):
        assert 0.0 < OrchestratorEnsembleBridge.RECOMMENDED_ENSEMBLE_WEIGHT <= 1.0

    def test_output_path_is_json(self):
        assert str(OrchestratorEnsembleBridge.OUTPUT_PATH).endswith(".json")


class TestWeightToSignalBoundary:
    """Exhaustive boundary tests for _weight_to_signal."""

    @pytest.mark.parametrize("current,baseline,expected", [
        (0.060, 0.0, 1.0),    # delta +0.06 > +0.05
        (0.050, 0.0, 0.5),    # delta +0.05 = boundary → moderate
        (0.049, 0.0, 0.5),    # delta +0.049 → moderate
        (0.021, 0.0, 0.5),    # delta +0.021 → moderate
        (0.020, 0.0, 0.0),    # delta +0.020 = boundary → neutral
        (0.019, 0.0, 0.0),    # delta +0.019 → neutral
        (0.001, 0.0, 0.0),    # delta +0.001 → neutral
        (-0.001, 0.0, 0.0),   # delta -0.001 → neutral (delta > -0.02)
        (-0.019, 0.0, 0.0),   # delta -0.019 → neutral
        (-0.020, 0.0, -0.5),  # delta -0.020 = boundary → moderate under
        (-0.021, 0.0, -0.5),  # delta -0.021 → moderate under
        (-0.049, 0.0, -0.5),  # delta -0.049 → moderate under
        (-0.050, 0.0, -1.0),  # delta -0.050 = boundary → strong under
        (-0.051, 0.0, -1.0),  # delta -0.051 → strong under
    ])
    def test_boundary_values(self, current, baseline, expected):
        assert OrchestratorEnsembleBridge._weight_to_signal(current, baseline) == expected


class TestGenerateSignalExtended:
    """Extended tests for generate_signal method."""

    def _make_mock_rec(self, spy=0.46, gld=0.38, tlt=0.16, ief=0.0,
                       shy=0.0, btc=0.0, eth=0.0, baseline_spy=0.46,
                       baseline_gld=0.38, baseline_tlt=0.16,
                       conflict_count=0, total_spy_delta=0.0,
                       calendar_modifier=0.5, confidence=80.0,
                       estimated_sharpe=0.79, contributions=None):
        """Create a mock recommendation object."""
        rec = MagicMock()
        rec.spy = spy
        rec.gld = gld
        rec.tlt = tlt
        rec.ief = ief
        rec.shy = shy
        rec.btc = btc
        rec.eth = eth
        rec.baseline_spy = baseline_spy
        rec.baseline_gld = baseline_gld
        rec.baseline_tlt = baseline_tlt
        rec.conflict_count = conflict_count
        rec.total_spy_delta = total_spy_delta
        rec.calendar_modifier = calendar_modifier
        rec.confidence = confidence
        rec.estimated_sharpe = estimated_sharpe
        rec.contributions = contributions or []
        return rec

    def test_generate_signal_returns_reading(self):
        bridge = OrchestratorEnsembleBridge()
        mock_rec = self._make_mock_rec()
        with patch.object(bridge._orch, 'recommend', return_value=mock_rec):
            result = bridge.generate_signal()
        assert isinstance(result, UnifiedSignalReading)
        assert result.source == "unified_overlay"
        assert result.weight == 0.20

    def test_generate_signal_high_conflict_de_risk(self):
        bridge = OrchestratorEnsembleBridge()
        mock_rec = self._make_mock_rec(conflict_count=3)
        with patch.object(bridge._orch, 'recommend', return_value=mock_rec):
            result = bridge.generate_signal()
        assert result.risk_signal == -0.5

    def test_generate_signal_spy_reduction_cautious(self):
        bridge = OrchestratorEnsembleBridge()
        mock_rec = self._make_mock_rec(total_spy_delta=-0.04, conflict_count=0)
        with patch.object(bridge._orch, 'recommend', return_value=mock_rec):
            result = bridge.generate_signal()
        assert result.risk_signal == -0.3

    def test_generate_signal_spy_increase_risk_on(self):
        bridge = OrchestratorEnsembleBridge()
        mock_rec = self._make_mock_rec(total_spy_delta=0.03, conflict_count=0)
        with patch.object(bridge._orch, 'recommend', return_value=mock_rec):
            result = bridge.generate_signal()
        assert result.risk_signal == 0.3

    def test_generate_signal_neutral_risk(self):
        bridge = OrchestratorEnsembleBridge()
        mock_rec = self._make_mock_rec(total_spy_delta=0.0, conflict_count=0)
        with patch.object(bridge._orch, 'recommend', return_value=mock_rec):
            result = bridge.generate_signal()
        assert result.risk_signal == 0.0

    def test_generate_signal_active_count(self):
        bridge = OrchestratorEnsembleBridge()
        contrib = [MagicMock(status="active"), MagicMock(status="active"),
                   MagicMock(status="inactive")]
        mock_rec = self._make_mock_rec(contributions=contrib)
        with patch.object(bridge._orch, 'recommend', return_value=mock_rec):
            result = bridge.generate_signal()
        assert result.num_overlays_active == 2

    def test_generate_signal_execution_from_calendar(self):
        bridge = OrchestratorEnsembleBridge()
        mock_rec = self._make_mock_rec(calendar_modifier=0.8)
        with patch.object(bridge._orch, 'recommend', return_value=mock_rec):
            result = bridge.generate_signal()
        assert result.execution_signal == 0.8

    def test_generate_signal_explanation_contains_sharpe(self):
        bridge = OrchestratorEnsembleBridge()
        mock_rec = self._make_mock_rec(estimated_sharpe=0.95)
        with patch.object(bridge._orch, 'recommend', return_value=mock_rec):
            result = bridge.generate_signal()
        assert "0.950" in result.explanation


class TestSaveSignal:
    """Test save_signal method."""

    def test_save_signal_writes_json(self, tmp_path):
        bridge = OrchestratorEnsembleBridge()
        signal = UnifiedSignalReading(
            timestamp="2026-05-23T00:00:00", source="test",
            value=0.1, confidence=0.8, weight=0.20,
            spy_signal=0.0, gld_signal=0.0, tlt_signal=0.0,
            ief_signal=0.0, shy_signal=0.0, btc_signal=0.0, eth_signal=0.0,
            risk_signal=0.0, execution_signal=0.0,
            explanation="", num_overlays_active=0, conflict_count=0,
        )
        mock_path = tmp_path / "unified_ensemble_signal.json"
        with patch.object(OrchestratorEnsembleBridge, 'OUTPUT_PATH', mock_path):
            bridge.save_signal(signal)
        assert mock_path.exists()
        data = json.loads(mock_path.read_text())
        assert data["value"] == 0.1
        assert data["source"] == "test"


class TestCLI:
    """CLI main() function tests."""

    def test_main_runs(self, capsys):
        from src.strategy.orchestrator_ensemble_bridge import main
        bridge = OrchestratorEnsembleBridge()
        mock_signal = UnifiedSignalReading(
            timestamp="2026-05-23T00:00:00", source="test",
            value=0.1, confidence=0.8, weight=0.20,
            spy_signal=0.3, gld_signal=-0.2, tlt_signal=0.1,
            ief_signal=0.0, shy_signal=0.0, btc_signal=0.0, eth_signal=0.0,
            risk_signal=0.0, execution_signal=0.5,
            explanation="CLI test", num_overlays_active=2, conflict_count=1,
        )
        with patch.object(OrchestratorEnsembleBridge, 'generate_signal', return_value=mock_signal):
            main()
        captured = capsys.readouterr()
        assert "ORCHESTRATOR-ENSEMBLE VOTER BRIDGE" in captured.out
        assert "0.100" in captured.out
