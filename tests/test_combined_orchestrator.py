#!/usr/bin/env python3
"""
Tests for Combined Signal Orchestrator — constants, data classes,
conflict detection, signal resolution, and recommendation generation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.signals.combined_orchestrator import (
    SIGNAL_WEIGHTS, CONFLICT_THRESHOLD, HIGH_CONFIDENCE,
    SignalSource, CombinedRecommendation,
    CombinedSignalOrchestrator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal_source(source='tsmom', deltas=None, confidence=0.8, regime=None):
    return SignalSource(
        source=source,
        deltas=deltas or {'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
        confidence=confidence,
        regime=regime,
        notes='test',
    )


def _make_orchestrator(base=None):
    """Create orchestrator with mocked sub-modules."""
    with patch('src.signals.combined_orchestrator.TSMOMOverlay'), \
         patch('src.signals.combined_orchestrator.PortfolioRegimeManager'), \
         patch('src.signals.combined_orchestrator.FedPolicyOverlay'), \
         patch('src.signals.combined_orchestrator._AI_CONTROLLER_AVAILABLE', False):
        orch = CombinedSignalOrchestrator(base_allocation=base)
    return orch


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_signal_weights_sum_to_one(self):
        total = sum(SIGNAL_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_signal_weights_keys(self):
        assert 'tsmom' in SIGNAL_WEIGHTS
        assert 'hmm_regime' in SIGNAL_WEIGHTS
        assert 'fed_policy' in SIGNAL_WEIGHTS
        assert 'ai_agent' in SIGNAL_WEIGHTS
        assert 'base' in SIGNAL_WEIGHTS

    def test_tsmom_highest_weight(self):
        assert SIGNAL_WEIGHTS['tsmom'] == max(SIGNAL_WEIGHTS.values())

    def test_conflict_threshold(self):
        assert CONFLICT_THRESHOLD == 0.05

    def test_high_confidence(self):
        assert HIGH_CONFIDENCE == 0.75


# ---------------------------------------------------------------------------
# SignalSource tests
# ---------------------------------------------------------------------------

class TestSignalSource:
    def test_creation(self):
        s = _make_signal_source()
        assert s.source == 'tsmom'
        assert s.confidence == 0.8

    def test_deltas(self):
        s = _make_signal_source(deltas={'SPY': 0.05, 'GLD': -0.03})
        assert s.deltas['SPY'] == 0.05

    def test_optional_fields(self):
        s = _make_signal_source(regime='bull')
        assert s.regime == 'bull'


# ---------------------------------------------------------------------------
# CombinedRecommendation tests
# ---------------------------------------------------------------------------

class TestCombinedRecommendation:
    def _make_rec(self, **overrides):
        defaults = dict(
            timestamp='2026-01-01',
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            recommended_allocation={'SPY': 0.48, 'GLD': 0.36, 'TLT': 0.16},
            deltas={'SPY': 0.02, 'GLD': -0.02, 'TLT': 0.0},
            source_signals={},
            conflicts_detected=[],
            resolution_strategy='weighted_average',
            predicted_volatility=0.12,
            regime_dominant='neutral',
            confidence=0.7,
        )
        defaults.update(overrides)
        return CombinedRecommendation(**defaults)

    def test_creation(self):
        rec = self._make_rec()
        assert rec.timestamp == '2026-01-01'
        assert rec.confidence == 0.7

    def test_to_dict(self):
        rec = self._make_rec()
        d = rec.to_dict()
        assert 'timestamp' in d
        assert 'recommended_allocation' in d
        assert 'conflicts_detected' in d
        assert 'resolution_strategy' in d

    def test_to_dict_source_signals(self):
        signals = {'tsmom': _make_signal_source()}
        rec = self._make_rec(source_signals=signals)
        d = rec.to_dict()
        assert 'tsmom' in d['source_signals']
        assert d['source_signals']['tsmom']['confidence'] == 0.8


# ---------------------------------------------------------------------------
# CombinedSignalOrchestrator tests
# ---------------------------------------------------------------------------

class TestCombinedSignalOrchestrator:
    def test_init_default_base(self):
        orch = _make_orchestrator()
        assert orch.base_allocation == {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}

    def test_init_custom_base(self):
        base = {'SPY': 0.50, 'GLD': 0.30, 'TLT': 0.20}
        orch = _make_orchestrator(base=base)
        assert orch.base_allocation == base

    def test_init_weights(self):
        orch = _make_orchestrator()
        assert orch.weights == SIGNAL_WEIGHTS

    def test_detect_conflicts_no_conflict(self):
        orch = _make_orchestrator()
        signals = {
            'tsmom': _make_signal_source(deltas={'SPY': 0.05, 'GLD': 0.03, 'TLT': 0.0}),
            'fed_policy': _make_signal_source('fed_policy', deltas={'SPY': 0.03, 'GLD': 0.02, 'TLT': 0.0}),
        }
        conflicts = orch.detect_conflicts(signals)
        assert len(conflicts) == 0

    def test_detect_conflicts_opposite_direction(self):
        orch = _make_orchestrator()
        signals = {
            'tsmom': _make_signal_source(deltas={'SPY': 0.05, 'GLD': -0.05, 'TLT': 0.0}),
            'fed_policy': _make_signal_source('fed_policy', deltas={'SPY': -0.05, 'GLD': 0.05, 'TLT': 0.0}),
        }
        conflicts = orch.detect_conflicts(signals)
        assert len(conflicts) > 0
        assert any('TSMOM' in c and 'Fed' in c for c in conflicts)

    def test_detect_conflicts_low_confidence_ignored(self):
        orch = _make_orchestrator()
        signals = {
            'tsmom': _make_signal_source(confidence=0.3, deltas={'SPY': 0.05, 'GLD': -0.05, 'TLT': 0.0}),
            'fed_policy': _make_signal_source('fed_policy', confidence=0.3, deltas={'SPY': -0.05, 'GLD': 0.05, 'TLT': 0.0}),
        }
        conflicts = orch.detect_conflicts(signals)
        assert len(conflicts) == 0

    def test_resolve_signals_weighted(self):
        orch = _make_orchestrator()
        signals = {
            'tsmom': _make_signal_source(deltas={'SPY': 0.10, 'GLD': -0.05, 'TLT': 0.0}),
            'fed_policy': _make_signal_source('fed_policy', deltas={'SPY': 0.05, 'GLD': 0.03, 'TLT': 0.0}),
        }
        deltas, resolution = orch.resolve_signals(signals, [])
        assert 'SPY' in deltas
        assert 'weighted_average' in resolution

    def test_resolve_signals_conflict_caps_deltas(self):
        orch = _make_orchestrator()
        signals = {
            'tsmom': _make_signal_source(deltas={'SPY': 0.10, 'GLD': -0.10, 'TLT': 0.0}),
            'fed_policy': _make_signal_source('fed_policy', deltas={'SPY': -0.10, 'GLD': 0.10, 'TLT': 0.0}),
        }
        conflicts = ['SPY: TSMOM(+10%) vs Fed(-10%)']
        deltas_conflict, _ = orch.resolve_signals(signals, conflicts)
        deltas_no_conflict, _ = orch.resolve_signals(signals, [])
        # Conflict should reduce magnitude
        assert abs(deltas_conflict['SPY']) <= abs(deltas_no_conflict['SPY'])

    def test_resolve_signals_hmm_neutral_reduces(self):
        orch = _make_orchestrator()
        signals = {
            'tsmom': _make_signal_source(deltas={'SPY': 0.10, 'GLD': 0.0, 'TLT': 0.0}),
            'hmm_regime': _make_signal_source('hmm_regime', confidence=0.9, regime='neutral',
                                               deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0}),
        }
        deltas, resolution = orch.resolve_signals(signals, [])
        assert 'hmm_neutral_reduction' in resolution

    def test_resolve_signals_consensus_boost(self):
        orch = _make_orchestrator()
        signals = {
            'tsmom': _make_signal_source(deltas={'SPY': 0.05, 'GLD': 0.03, 'TLT': 0.01}),
            'fed_policy': _make_signal_source('fed_policy', deltas={'SPY': 0.03, 'GLD': 0.02, 'TLT': 0.01}),
        }
        deltas, resolution = orch.resolve_signals(signals, [])
        assert 'consensus_boost' in resolution

    def test_resolve_signals_zero_weight(self):
        orch = _make_orchestrator()
        # Only base source with zero delta
        signals = {
            'base': _make_signal_source('base', confidence=0.0, deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0}),
        }
        deltas, _ = orch.resolve_signals(signals, [])
        assert all(v == 0.0 for v in deltas.values())

    @patch.object(CombinedSignalOrchestrator, 'collect_signals')
    def test_generate_recommendation_returns_rec(self, mock_collect):
        orch = _make_orchestrator()
        mock_collect.return_value = {
            'tsmom': _make_signal_source(deltas={'SPY': 0.02, 'GLD': -0.01, 'TLT': 0.0}),
            'hmm_regime': _make_signal_source('hmm_regime', confidence=0.6, regime='neutral'),
            'fed_policy': _make_signal_source('fed_policy', confidence=0.5),
            'ai_agent': _make_signal_source('ai_agent', confidence=0.0),
            'base': _make_signal_source('base', confidence=0.6),
        }
        rec = orch.generate_recommendation()
        assert isinstance(rec, CombinedRecommendation)
        assert rec.timestamp is not None

    @patch.object(CombinedSignalOrchestrator, 'collect_signals')
    def test_generate_recommendation_sums_near_one(self, mock_collect):
        orch = _make_orchestrator()
        mock_collect.return_value = {
            'tsmom': _make_signal_source(deltas={'SPY': 0.02, 'GLD': -0.01, 'TLT': 0.0}),
            'hmm_regime': _make_signal_source('hmm_regime'),
            'fed_policy': _make_signal_source('fed_policy'),
            'ai_agent': _make_signal_source('ai_agent', confidence=0.0),
            'base': _make_signal_source('base'),
        }
        rec = orch.generate_recommendation()
        total = sum(rec.recommended_allocation.values())
        assert abs(total - 1.0) < 0.02

    @patch.object(CombinedSignalOrchestrator, 'collect_signals')
    def test_generate_recommendation_conflicts(self, mock_collect):
        orch = _make_orchestrator()
        mock_collect.return_value = {
            'tsmom': _make_signal_source(deltas={'SPY': 0.10, 'GLD': -0.10, 'TLT': 0.0}),
            'hmm_regime': _make_signal_source('hmm_regime'),
            'fed_policy': _make_signal_source('fed_policy', deltas={'SPY': -0.10, 'GLD': 0.10, 'TLT': 0.0}),
            'ai_agent': _make_signal_source('ai_agent', confidence=0.0),
            'base': _make_signal_source('base'),
        }
        rec = orch.generate_recommendation()
        assert len(rec.conflicts_detected) > 0

    @patch.object(CombinedSignalOrchestrator, 'collect_signals')
    def test_generate_recommendation_all_agree(self, mock_collect):
        orch = _make_orchestrator()
        mock_collect.return_value = {
            'tsmom': _make_signal_source(deltas={'SPY': 0.05, 'GLD': 0.03, 'TLT': 0.01}),
            'hmm_regime': _make_signal_source('hmm_regime', deltas={'SPY': 0.03, 'GLD': 0.02, 'TLT': 0.01}),
            'fed_policy': _make_signal_source('fed_policy', deltas={'SPY': 0.02, 'GLD': 0.01, 'TLT': 0.0}),
            'ai_agent': _make_signal_source('ai_agent', confidence=0.0),
            'base': _make_signal_source('base', deltas={'SPY': 0.01, 'GLD': 0.01, 'TLT': 0.0}),
        }
        rec = orch.generate_recommendation()
        assert 'consensus_boost' in rec.resolution_strategy

    def test_format_recommendation(self):
        orch = _make_orchestrator()
        rec = CombinedRecommendation(
            timestamp='2026-01-01',
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            recommended_allocation={'SPY': 0.48, 'GLD': 0.36, 'TLT': 0.16},
            deltas={'SPY': 0.02, 'GLD': -0.02, 'TLT': 0.0},
            source_signals={'tsmom': _make_signal_source()},
            conflicts_detected=[],
            resolution_strategy='weighted_average',
            predicted_volatility=0.12,
            regime_dominant='neutral',
            confidence=0.7,
        )
        text = orch.format_recommendation(rec)
        assert 'Combined Signal Orchestrator' in text
        assert 'SPY' in text
        assert 'GLD' in text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
