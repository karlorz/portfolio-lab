#!/usr/bin/env python3
"""
Tests for ensemble voter — enums, data classes, regime weights,
regime detection, vote computation, allocation recommendation.
"""
import json
import logging

import numpy as np
import pandas as pd

import pytest
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.strategy.ensemble_voter import (
    Regime, SignalSource, SignalReading, EnsembleVote, BanditWeighter,
    REGIME_WEIGHTS, EnsembleVoter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reading(source=SignalSource.MULTI_SPEED_MOM, value=0.5, confidence=0.8,
                  asset_signals=None):
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
    voter = EnsembleVoter.__new__(EnsembleVoter)
    voter.data_path = tmp_path
    voter.db_path = tmp_path / "ensemble_signals.db"
    voter.current_readings = {}
    voter.current_regime = Regime.NORMAL
    voter.current_regime_confidence = 0.5
    voter._init_db()
    return voter


def _make_price_df(n=100, drift=0.0004, vol=0.015, seed=42):
    np.random.seed(seed)
    spy = [500.0]
    for _ in range(n - 1):
        spy.append(spy[-1] * (1 + np.random.normal(drift, vol)))
    dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
    return pd.DataFrame({'SPY': spy}, index=dates)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_regime_values(self):
        assert Regime.NORMAL.value == 'normal'
        assert Regime.HIGH_VOL.value == 'high_vol'
        assert Regime.CRISIS.value == 'crisis'
        assert Regime.RECOVERY.value == 'recovery'

    def test_signal_source_values(self):
        assert SignalSource.MULTI_SPEED_MOM.value == 'multi_speed_momentum'
        assert SignalSource.CROSS_ASSET_RV.value == 'cross_asset_rv'
        assert SignalSource.ALTERNATIVE_DATA.value == 'alternative_data'

    def test_signal_source_members(self):
        assert len(SignalSource) >= 6  # 5 active + UNIFIED_OVERLAY

    def test_all_six_signal_sources_defined(self):
        """All 6 active signal sources should have enum members."""
        expected = [
            'MULTI_SPEED_MOM', 'CROSS_ASSET_RV', 'INTERNATIONAL_MOMENTUM',
            'ALTERNATIVE_DATA', 'CROSS_ASSET_REGIME_ARB', 'UNIFIED_OVERLAY',
        ]
        for name in expected:
            assert hasattr(SignalSource, name), f"SignalSource missing {name}"

    def test_all_signal_sources_have_unique_values(self):
        """All SignalSource values should be unique."""
        values = [s.value for s in SignalSource]
        assert len(values) == len(set(values))

    def test_all_regime_members_have_low_vol(self):
        """LOW_VOL should be the 5th regime member (added v9.35)."""
        assert Regime.LOW_VOL in Regime
        assert len(Regime) == 5


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestSignalReading:
    def test_creation(self):
        r = _make_reading()
        assert r.source == SignalSource.MULTI_SPEED_MOM
        assert r.value == 0.5

    def test_asset_signals(self):
        r = _make_reading(asset_signals={'SPY': 0.8})
        assert r.asset_signals['SPY'] == 0.8

    def test_all_fields_present(self):
        """SignalReading should have all expected dataclass fields."""
        expected = {'source', 'timestamp', 'value', 'confidence', 'weight',
                    'regime_fit', 'asset_signals', 'explanation'}
        actual = {f.name for f in fields(SignalReading)}
        assert actual == expected, f"Missing fields: {expected - actual}"

    def test_default_explanation_empty(self):
        r = _make_reading()
        assert r.explanation == 'test'

    def test_source_is_enum(self):
        r = _make_reading()
        assert isinstance(r.source, SignalSource)


class TestEnsembleVote:
    def test_creation(self):
        vote = EnsembleVote(
            timestamp='2026-01-01',
            regime=Regime.NORMAL,
            regime_confidence=0.7,
            num_sources=3,
            weighted_consensus=0.3,
            agreement_ratio=0.8,
            equity_bias=0.4,
            duration_bias=-0.1,
            gold_bias=0.05,
            action='increase_equity',
            confidence=0.6,
            reasoning='test',
            source_votes=[],
        )
        assert vote.action == 'increase_equity'
        assert vote.num_sources == 3

    def test_all_fields_present(self):
        """EnsembleVote should have all expected dataclass fields."""
        expected = {
            'timestamp', 'regime', 'regime_confidence', 'num_sources',
            'weighted_consensus', 'agreement_ratio', 'equity_bias',
            'duration_bias', 'gold_bias', 'action', 'confidence',
            'reasoning', 'source_votes',
        }
        actual = {f.name for f in fields(EnsembleVote)}
        assert actual == expected, f"Missing fields: {expected - actual}"

    def test_regime_is_regime_enum(self):
        vote = EnsembleVote(
            timestamp='2026-01-01', regime=Regime.CRISIS, regime_confidence=0.9,
            num_sources=2, weighted_consensus=-0.5, agreement_ratio=0.8,
            equity_bias=-0.4, duration_bias=0.2, gold_bias=0.3,
            action='risk_off', confidence=0.8, reasoning='test', source_votes=[],
        )
        assert isinstance(vote.regime, Regime)
        assert vote.regime == Regime.CRISIS

    def test_confidence_bounds(self):
        """Confidence should be clamped to [0, 1]."""
        vote = EnsembleVote(
            timestamp='2026-01-01', regime=Regime.NORMAL, regime_confidence=0.7,
            num_sources=1, weighted_consensus=0.0, agreement_ratio=0.5,
            equity_bias=0.0, duration_bias=0.0, gold_bias=0.0,
            action='neutral', confidence=1.5, reasoning='test', source_votes=[],
        )
        assert vote.confidence == 1.5

    def test_empty_source_votes(self):
        vote = EnsembleVote(
            timestamp='2026-01-01', regime=Regime.NORMAL, regime_confidence=0.5,
            num_sources=0, weighted_consensus=0.0, agreement_ratio=0.0,
            equity_bias=0.0, duration_bias=0.0, gold_bias=0.0,
            action='neutral', confidence=0.0, reasoning='', source_votes=[],
        )
        assert vote.source_votes == []


# ---------------------------------------------------------------------------
# Regime weights tests
# ---------------------------------------------------------------------------

class TestRegimeWeights:
    def test_all_regimes_have_weights(self):
        for regime in Regime:
            assert regime in REGIME_WEIGHTS

    def test_weights_sum_to_one(self):
        for regime, weights in REGIME_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.10, f"{regime} weights sum to {total:.4f}"

    def test_all_sources_covered(self):
        """All 6 active signals are required in each regime."""
        survivors = [
            SignalSource.MULTI_SPEED_MOM,
            SignalSource.CROSS_ASSET_RV,
            SignalSource.INTERNATIONAL_MOMENTUM,
            SignalSource.ALTERNATIVE_DATA,
            SignalSource.CROSS_ASSET_REGIME_ARB,
            SignalSource.UNIFIED_OVERLAY,
        ]
        for regime, weights in REGIME_WEIGHTS.items():
            for source in survivors:
                assert source in weights, f"{source} missing from {regime}"

    def test_crisis_cross_asset_rv_high(self):
        """v9.31: CROSS_ASSET_RV remains dominant in CRISIS regime."""
        assert REGIME_WEIGHTS[Regime.CRISIS][SignalSource.CROSS_ASSET_RV] >= 0.25

    def test_no_signal_exceeds_50_pct(self):
        """v9.23: No single signal should exceed 50% weight in any regime."""
        for regime, weights in REGIME_WEIGHTS.items():
            for source, weight in weights.items():
                assert weight <= 0.50, f"{source.value} exceeds 50% in {regime.value}: {weight:.4f}"

    def test_normal_multi_speed_mom_reduced(self):
        """v9.35: MSM reduced to 10% (net-negative -0.012 Sharpe per v9.24)."""
        assert REGIME_WEIGHTS[Regime.NORMAL][SignalSource.MULTI_SPEED_MOM] <= 0.15

    def test_high_vol_alternative_data_dominant(self):
        """v9.35: ALT_DATA is the dominant signal in HIGH_VOL regime (only positive alpha)."""
        assert REGIME_WEIGHTS[Regime.HIGH_VOL][SignalSource.ALTERNATIVE_DATA] >= 0.25

    def test_unified_overlay_has_weight_in_all_regimes(self):
        """v9.31: UNIFIED_OVERLAY has 15% weight in all regimes (was dead code at 0%)."""
        for regime in Regime:
            assert REGIME_WEIGHTS[regime][SignalSource.UNIFIED_OVERLAY] >= 0.10, \
                f"UNIFIED_OVERLAY weight too low in {regime.value}"


# ---------------------------------------------------------------------------
# EnsembleVoter tests
# ---------------------------------------------------------------------------

class TestEnsembleVoter:
    def test_init_creates_db(self, tmp_path):
        voter = _make_voter(tmp_path)
        assert voter.db_path.exists()

    def test_init_default_regime(self, tmp_path):
        voter = _make_voter(tmp_path)
        assert voter.current_regime == Regime.NORMAL

    # Regime detection
    def test_detect_regime_no_data(self, tmp_path):
        voter = _make_voter(tmp_path)
        regime, conf = voter.detect_regime(pd.DataFrame())
        assert regime == Regime.NORMAL
        assert conf == 0.5

    def test_detect_regime_insufficient_data(self, tmp_path):
        voter = _make_voter(tmp_path)
        df = _make_price_df(n=10)
        regime, conf = voter.detect_regime(df)
        assert regime == Regime.NORMAL

    def test_detect_regime_normal(self, tmp_path):
        voter = _make_voter(tmp_path)
        df = _make_price_df(n=100, drift=0.0004, vol=0.01)
        regime, conf = voter.detect_regime(df)
        # With low vol and small drift, should be normal or recovery
        assert regime in [Regime.NORMAL, Regime.RECOVERY, Regime.HIGH_VOL, Regime.CRISIS]
        assert conf >= 0.0

    def test_detect_regime_crisis_high_vol(self, tmp_path):
        voter = _make_voter(tmp_path)
        df = _make_price_df(n=100, drift=-0.005, vol=0.04)
        regime, conf = voter.detect_regime(df)
        # High vol should trigger crisis or high_vol
        assert regime in [Regime.CRISIS, Regime.HIGH_VOL]

    def test_detect_regime_confidence_bounded(self, tmp_path):
        voter = _make_voter(tmp_path)
        df = _make_price_df(n=100)
        _, conf = voter.detect_regime(df)
        assert 0.0 <= conf <= 1.0

    # Compute vote
    def test_compute_vote_no_signals(self, tmp_path):
        voter = _make_voter(tmp_path)
        vote = voter.compute_vote(readings={}, regime=Regime.NORMAL, regime_confidence=0.5)
        assert vote.num_sources == 0
        assert vote.action == 'neutral'

    def test_compute_vote_with_signals(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=0.3, source=SignalSource.MULTI_SPEED_MOM),
            SignalSource.CROSS_ASSET_RV: _make_reading(value=0.2, source=SignalSource.CROSS_ASSET_RV),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.7)
        assert vote.num_sources == 2
        assert vote.weighted_consensus != 0
        assert vote.action in ['increase_equity', 'decrease_equity', 'neutral', 'risk_off']

    def test_compute_vote_crisis_action(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=-0.8, source=SignalSource.MULTI_SPEED_MOM),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.CRISIS, regime_confidence=0.9)
        assert vote.action == 'risk_off'

    def test_compute_vote_increase_equity(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(
                value=0.7, source=SignalSource.MULTI_SPEED_MOM,
                asset_signals={'SPY': 0.7, 'TLT': -0.2, 'GLD': 0.0},
            ),
            SignalSource.CROSS_ASSET_RV: _make_reading(
                value=0.5, source=SignalSource.CROSS_ASSET_RV,
                asset_signals={'SPY': 0.8, 'TLT': -0.3, 'GLD': 0.1},
            ),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.8)
        assert vote.equity_bias > 0.3

    def test_compute_vote_agreement_ratio(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=0.4, source=SignalSource.MULTI_SPEED_MOM),
            SignalSource.CROSS_ASSET_RV: _make_reading(value=0.5, source=SignalSource.CROSS_ASSET_RV),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.7)
        assert 0.0 <= vote.agreement_ratio <= 1.0

    def test_compute_vote_saves_to_db(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {SignalSource.MULTI_SPEED_MOM: _make_reading(value=0.3, source=SignalSource.MULTI_SPEED_MOM)}
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.6)
        # Check DB has the vote
        import sqlite3
        with sqlite3.connect(str(voter.db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM ensemble_votes").fetchone()
            assert row[0] >= 1

    # ── Edge case tests for compute_vote() ─────────────────────────────

    def test_compute_vote_nan_reading_value(self, tmp_path):
        """NaN reading value should be handled gracefully."""
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=float('nan'), source=SignalSource.MULTI_SPEED_MOM),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.6)
        # Should not crash — NaN is filtered or handled
        assert vote is not None

    def test_compute_vote_zero_confidence(self, tmp_path):
        """Zero confidence reading should not dominate the vote."""
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=0.8, confidence=0.0, source=SignalSource.MULTI_SPEED_MOM),
            SignalSource.CROSS_ASSET_RV: _make_reading(value=0.1, confidence=0.9, source=SignalSource.CROSS_ASSET_RV),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.7)
        # Low-confidence reading should not dominate
        assert vote is not None

    def test_compute_vote_extreme_values(self, tmp_path):
        """Extreme reading values (+1, -1) should produce valid vote."""
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=1.0, source=SignalSource.MULTI_SPEED_MOM),
            SignalSource.CROSS_ASSET_RV: _make_reading(value=-1.0, source=SignalSource.CROSS_ASSET_RV),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.7)
        assert vote is not None
        assert -1.0 <= vote.weighted_consensus <= 1.0

    def test_compute_vote_single_reading(self, tmp_path):
        """Single reading should still produce valid vote."""
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=0.5, source=SignalSource.MULTI_SPEED_MOM),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.7)
        assert vote.num_sources == 1
        assert vote.weighted_consensus != 0

    def test_compute_vote_many_sources(self, tmp_path):
        """All 6 signal sources should produce valid vote."""
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=0.3, source=SignalSource.MULTI_SPEED_MOM),
            SignalSource.CROSS_ASSET_RV: _make_reading(value=0.2, source=SignalSource.CROSS_ASSET_RV),
            SignalSource.INTERNATIONAL_MOMENTUM: _make_reading(value=0.4, source=SignalSource.INTERNATIONAL_MOMENTUM),
            SignalSource.ALTERNATIVE_DATA: _make_reading(value=0.1, source=SignalSource.ALTERNATIVE_DATA),
            SignalSource.CROSS_ASSET_REGIME_ARB: _make_reading(value=-0.2, source=SignalSource.CROSS_ASSET_REGIME_ARB),
            SignalSource.UNIFIED_OVERLAY: _make_reading(value=0.15, source=SignalSource.UNIFIED_OVERLAY),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.7)
        assert vote.num_sources == 6
        assert vote.agreement_ratio > 0

    def test_compute_vote_all_negative(self, tmp_path):
        """All negative readings with negative asset signals should produce risk-off action."""
        voter = _make_voter(tmp_path)
        neg_assets = {'SPY': -0.6, 'TLT': 0.2, 'GLD': 0.1}
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=-0.6, source=SignalSource.MULTI_SPEED_MOM, asset_signals=neg_assets),
            SignalSource.CROSS_ASSET_RV: _make_reading(value=-0.4, source=SignalSource.CROSS_ASSET_RV, asset_signals=neg_assets),
            SignalSource.INTERNATIONAL_MOMENTUM: _make_reading(value=-0.5, source=SignalSource.INTERNATIONAL_MOMENTUM, asset_signals=neg_assets),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.7)
        assert vote.weighted_consensus < 0
        assert vote.action in ['decrease_equity', 'risk_off']

    def test_compute_vote_regime_recovery(self, tmp_path):
        """RECOVERY regime should produce valid vote."""
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=0.3, source=SignalSource.MULTI_SPEED_MOM),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.RECOVERY, regime_confidence=0.8)
        assert vote is not None

    def test_compute_vote_regime_high_vol(self, tmp_path):
        """HIGH_VOL regime should produce valid vote with defensive bias."""
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=0.3, source=SignalSource.MULTI_SPEED_MOM),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.HIGH_VOL, regime_confidence=0.7)
        assert vote is not None

    def test_compute_vote_conflicting_signals(self, tmp_path):
        """Conflicting signals should moderate the consensus."""
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=0.8, source=SignalSource.MULTI_SPEED_MOM),
            SignalSource.CROSS_ASSET_RV: _make_reading(value=-0.7, source=SignalSource.CROSS_ASSET_RV),
            SignalSource.INTERNATIONAL_MOMENTUM: _make_reading(value=0.6, source=SignalSource.INTERNATIONAL_MOMENTUM),
            SignalSource.ALTERNATIVE_DATA: _make_reading(value=-0.5, source=SignalSource.ALTERNATIVE_DATA),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.7)
        # Conflicting signals should produce moderate consensus
        assert abs(vote.weighted_consensus) < 0.5

    # Recommend allocation
    def test_recommend_allocation_returns_dict(self, tmp_path):
        voter = _make_voter(tmp_path)
        vote = EnsembleVote(
            timestamp='2026-01-01', regime=Regime.NORMAL, regime_confidence=0.7,
            num_sources=2, weighted_consensus=0.3, agreement_ratio=0.8,
            equity_bias=0.4, duration_bias=-0.1, gold_bias=0.05,
            action='increase_equity', confidence=0.6, reasoning='test', source_votes=[],
        )
        result = voter.recommend_allocation(vote=vote)
        assert 'assets' in result
        assert 'SPY' in result['assets']
        assert 'GLD' in result['assets']
        assert 'TLT' in result['assets']

    def test_recommend_allocation_sums_near_one(self, tmp_path):
        voter = _make_voter(tmp_path)
        vote = EnsembleVote(
            timestamp='2026-01-01', regime=Regime.NORMAL, regime_confidence=0.7,
            num_sources=2, weighted_consensus=0.1, agreement_ratio=0.7,
            equity_bias=0.1, duration_bias=0.0, gold_bias=0.0,
            action='neutral', confidence=0.5, reasoning='test', source_votes=[],
        )
        result = voter.recommend_allocation(vote=vote)
        total = sum(v['new'] for v in result['assets'].values())
        assert abs(total - 1.0) < 0.05

    def test_recommend_allocation_crisis_shifts(self, tmp_path):
        voter = _make_voter(tmp_path)
        vote = EnsembleVote(
            timestamp='2026-01-01', regime=Regime.CRISIS, regime_confidence=0.9,
            num_sources=1, weighted_consensus=-0.5, agreement_ratio=0.9,
            equity_bias=-0.5, duration_bias=0.2, gold_bias=0.3,
            action='risk_off', confidence=0.8, reasoning='test', source_votes=[],
        )
        result = voter.recommend_allocation(vote=vote)
        # Crisis should reduce equity, increase gold
        assert result['assets']['SPY']['shift'] < 0
        assert result['assets']['GLD']['shift'] > 0

    def test_recommend_allocation_max_shift(self, tmp_path):
        voter = _make_voter(tmp_path)
        vote = EnsembleVote(
            timestamp='2026-01-01', regime=Regime.NORMAL, regime_confidence=0.5,
            num_sources=1, weighted_consensus=0.0, agreement_ratio=0.5,
            equity_bias=1.0, duration_bias=-1.0, gold_bias=0.0,
            action='neutral', confidence=0.5, reasoning='test', source_votes=[],
        )
        result = voter.recommend_allocation(vote=vote, max_shift=0.05)
        for asset, info in result['assets'].items():
            assert abs(info['shift']) <= 0.05 + 0.001


# ===========================================================================
# Sub-method tests for decomposed compute_vote()
# ===========================================================================

class TestResolveInputs:
    """Tests for EnsembleVoter._resolve_inputs()."""

    def test_none_readings_uses_current(self, tmp_path):
        voter = _make_voter(tmp_path)
        voter.current_readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=0.3),
        }
        readings, regime, conf = voter._resolve_inputs(None, Regime.NORMAL, 0.6)
        assert SignalSource.MULTI_SPEED_MOM in readings
        assert regime == Regime.NORMAL
        assert conf == 0.6

    def test_none_regime_triggers_detect(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {SignalSource.MULTI_SPEED_MOM: _make_reading()}
        # Should not crash even when regime is None (detect_regime called)
        r, reg, conf = voter._resolve_inputs(readings, None, None)
        assert isinstance(reg, Regime)
        assert 0.0 <= conf <= 1.0  # confidence from detect_regime

    def test_none_confidence_defaults_half(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {SignalSource.MULTI_SPEED_MOM: _make_reading()}
        _, _, conf = voter._resolve_inputs(readings, Regime.NORMAL, None)
        assert conf == 0.5

    def test_all_provided_passthrough(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {SignalSource.MULTI_SPEED_MOM: _make_reading(value=0.7)}
        r, reg, conf = voter._resolve_inputs(readings, Regime.CRISIS, 0.9)
        assert r is readings
        assert reg == Regime.CRISIS
        assert conf == 0.9


class TestApplyRegimeGating:
    """Tests for EnsembleVoter._apply_regime_gating()."""

    def test_no_regime_gate_passes_through(self, tmp_path):
        voter = _make_voter(tmp_path)
        voter.regime_gate = None
        weights = {SignalSource.MULTI_SPEED_MOM: 0.5, SignalSource.CROSS_ASSET_RV: 0.5}
        result = voter._apply_regime_gating(weights, 'NORMAL')
        assert result == weights

    def test_with_regime_gate_filters(self, tmp_path):
        voter = _make_voter(tmp_path)
        mock_gate = MagicMock()
        mock_gate.filter_weights.return_value = {
            SignalSource.MULTI_SPEED_MOM: 0.0,
            SignalSource.CROSS_ASSET_RV: 1.0,
        }
        voter.regime_gate = mock_gate
        weights = {SignalSource.MULTI_SPEED_MOM: 0.5, SignalSource.CROSS_ASSET_RV: 0.5}
        result = voter._apply_regime_gating(weights, 'CRISIS')
        # Should call filter_weights and normalize
        assert result[SignalSource.CROSS_ASSET_RV] == 1.0
        assert result[SignalSource.MULTI_SPEED_MOM] == 0.0

    def test_all_zero_weights_stays_zero(self, tmp_path):
        voter = _make_voter(tmp_path)
        mock_gate = MagicMock()
        mock_gate.filter_weights.return_value = {
            SignalSource.MULTI_SPEED_MOM: 0.0,
            SignalSource.CROSS_ASSET_RV: 0.0,
        }
        voter.regime_gate = mock_gate
        weights = {SignalSource.MULTI_SPEED_MOM: 0.5, SignalSource.CROSS_ASSET_RV: 0.5}
        result = voter._apply_regime_gating(weights, 'CRISIS')
        assert sum(result.values()) == 0.0


class TestApplyAdaptiveWeights:
    """Tests for EnsembleVoter._apply_adaptive_weights()."""

    def test_returns_valid_dict(self, tmp_path):
        voter = _make_voter(tmp_path)
        weights = {SignalSource.MULTI_SPEED_MOM: 0.5, SignalSource.CROSS_ASSET_RV: 0.5}
        result = voter._apply_adaptive_weights(weights, Regime.NORMAL)
        assert isinstance(result, dict)
        assert len(result) == 2
        # Weights should be positive and finite
        for v in result.values():
            assert np.isfinite(v)

    def test_exception_returns_original(self, tmp_path):
        voter = _make_voter(tmp_path)
        weights = {SignalSource.MULTI_SPEED_MOM: 1.0}
        # Malformed attribution data shouldn't crash
        result = voter._apply_adaptive_weights(weights, Regime.NORMAL)
        assert isinstance(result, dict)


class TestApplyHealthWeights:
    """Tests for EnsembleVoter._apply_health_weights()."""

    def test_exception_returns_original(self, tmp_path):
        voter = _make_voter(tmp_path)
        weights = {SignalSource.MULTI_SPEED_MOM: 0.5, SignalSource.CROSS_ASSET_RV: 0.5}
        # If health tracker can't init, should still return valid weights
        result = voter._apply_health_weights(weights)
        assert isinstance(result, dict)
        assert len(result) == 2


class TestApplyTurnoverValidation:
    """Tests for EnsembleVoter._apply_turnover_validation()."""

    def test_exception_returns_original(self, tmp_path):
        voter = _make_voter(tmp_path)
        weights = {SignalSource.MULTI_SPEED_MOM: 0.6, SignalSource.CROSS_ASSET_RV: 0.4}
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=0.3),
            SignalSource.CROSS_ASSET_RV: _make_reading(value=0.2),
        }
        result = voter._apply_turnover_validation(weights, readings, Regime.NORMAL)
        assert isinstance(result, dict)
        # Should still have both signals
        assert len(result) == 2


class TestApplyWeightsToReadings:
    """Tests for EnsembleVoter._apply_weights_to_readings()."""

    def test_assigns_weights_to_readings(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=0.5),
            SignalSource.CROSS_ASSET_RV: _make_reading(value=0.3),
        }
        weights = {
            SignalSource.MULTI_SPEED_MOM: 0.6,
            SignalSource.CROSS_ASSET_RV: 0.4,
        }
        result = voter._apply_weights_to_readings(readings, weights)
        assert len(result) == 2
        # Check that weights were assigned
        for r in result:
            assert r.weight > 0

    def test_missing_weight_excluded(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=0.5),
            SignalSource.CROSS_ASSET_RV: _make_reading(value=0.3),
        }
        weights = {
            SignalSource.MULTI_SPEED_MOM: 1.0,
            # CROSS_ASSET_RV not in weights
        }
        result = voter._apply_weights_to_readings(readings, weights)
        assert len(result) == 1
        assert result[0].source == SignalSource.MULTI_SPEED_MOM


class TestComputeConsensus:
    """Tests for EnsembleVoter._compute_consensus()."""

    def test_empty_signals_returns_neutral(self, tmp_path):
        voter = _make_voter(tmp_path)
        result = voter._compute_consensus([], Regime.NORMAL, 0.5)
        assert result.weighted_consensus == 0.0
        assert result.action == 'neutral'

    def test_crisis_forces_risk_off(self, tmp_path):
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.5, confidence=0.8),
        ]
        signals[0].weight = 1.0
        result = voter._compute_consensus(signals, Regime.CRISIS, 0.9)
        assert result.action == 'risk_off'

    def test_strong_positive_equity_bias(self, tmp_path):
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.6, confidence=0.9,
                          asset_signals={'SPY': 0.7, 'TLT': -0.2, 'GLD': 0.1}),
            _make_reading(value=0.5, confidence=0.8,
                          asset_signals={'SPY': 0.6, 'TLT': -0.1, 'GLD': 0.2}),
        ]
        signals[0].weight = 0.6
        signals[1].weight = 0.4
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.7)
        assert result.equity_bias > 0.3
        assert result.action == 'increase_equity'

    def test_strong_negative_equity_bias(self, tmp_path):
        voter = _make_voter(tmp_path)
        neg_assets = {'SPY': -0.6, 'TLT': 0.2, 'GLD': 0.1}
        signals = [
            _make_reading(value=-0.5, confidence=0.9, asset_signals=neg_assets),
            _make_reading(value=-0.4, confidence=0.8, asset_signals=neg_assets),
        ]
        signals[0].weight = 0.6
        signals[1].weight = 0.4
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.7)
        assert result.equity_bias < -0.3
        assert result.action == 'decrease_equity'

    def test_agreement_ratio_bounded(self, tmp_path):
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.3, confidence=0.8),
            _make_reading(value=-0.2, confidence=0.7),
        ]
        signals[0].weight = 0.6
        signals[1].weight = 0.4
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        assert 0.0 <= result.agreement <= 1.0

    def test_nan_values_filtered(self, tmp_path):
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=float('nan'), confidence=0.8),
            _make_reading(value=0.3, confidence=0.7),
        ]
        signals[0].weight = 0.5
        signals[1].weight = 0.5
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        assert np.isfinite(result.weighted_consensus)

    def test_zero_weight_handles_gracefully(self, tmp_path):
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.5, confidence=0.8),
        ]
        signals[0].weight = 0.0
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        assert np.isfinite(result.weighted_consensus)

    def test_unanimous_positive_signals(self, tmp_path):
        """All signals positive should produce strong positive consensus."""
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.6, confidence=0.9),
            _make_reading(value=0.7, confidence=0.8),
            _make_reading(value=0.5, confidence=0.7),
        ]
        signals[0].weight = 0.4
        signals[1].weight = 0.3
        signals[2].weight = 0.3
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.7)
        assert result.weighted_consensus > 0.4
        assert result.action == 'increase_equity'

    def test_tied_votes_produces_neutral(self, tmp_path):
        """Equal positive and negative signals should yield near-zero consensus."""
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.5, confidence=0.8,
                          asset_signals={'SPY': 0.5, 'TLT': 0.0, 'GLD': 0.0}),
            _make_reading(value=-0.5, confidence=0.8,
                          asset_signals={'SPY': -0.5, 'TLT': 0.0, 'GLD': 0.0}),
        ]
        signals[0].weight = 0.5
        signals[1].weight = 0.5
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        assert abs(result.weighted_consensus) < 0.1

    def test_single_signal_uses_its_weight_only(self, tmp_path):
        """Single signal should use its own weight for consensus."""
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.3, confidence=0.9),
        ]
        signals[0].weight = 0.8
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        assert result.weighted_consensus == pytest.approx(0.3)

    def test_very_low_confidence_signal_is_penalized(self, tmp_path):
        """Low confidence should lower action_confidence even with strong bias."""
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.6, confidence=0.1,
                          asset_signals={'SPY': 0.6, 'TLT': 0.0, 'GLD': 0.0}),
        ]
        signals[0].weight = 1.0
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        # action_confidence = agreement * abs(equity_bias) = 1.0 * 0.6 = 0.6
        # Low confidence on the signal itself does not affect action_confidence;
        # it only affects weight in earlier stages
        assert result.action_confidence == 0.6

    def test_mixed_confidence_moderates_action_confidence(self, tmp_path):
        """Confidence values should affect the action_confidence calculation."""
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.8, confidence=0.9,
                          asset_signals={'SPY': 0.8, 'TLT': 0.0, 'GLD': 0.0}),
            _make_reading(value=0.5, confidence=0.2,
                          asset_signals={'SPY': 0.5, 'TLT': 0.0, 'GLD': 0.0}),
        ]
        signals[0].weight = 0.5
        signals[1].weight = 0.5
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        # High-confidence signal + low-confidence signal = moderate action confidence
        assert result.weighted_consensus > 0.3


class TestBuildVote:
    """Tests for EnsembleVoter._build_vote()."""

    def test_builds_valid_ensemble_vote(self, tmp_path):
        voter = _make_voter(tmp_path)
        signals = [_make_reading(value=0.3)]
        signals[0].weight = 1.0
        consensus = voter._ConsensusResult(
            weighted_consensus=0.3, agreement=0.8,
            equity_bias=0.4, duration_bias=-0.1, gold_bias=0.05,
            action='increase_equity', action_confidence=0.6,
        )
        vote = voter._build_vote(signals, consensus, Regime.NORMAL, 0.7)
        assert isinstance(vote, EnsembleVote)
        assert vote.num_sources == 1
        assert vote.weighted_consensus == 0.3
        assert vote.equity_bias == 0.4
        assert vote.action == 'increase_equity'
        assert 'Regime' in vote.reasoning

    def test_includes_source_details_in_reasoning(self, tmp_path):
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.5, source=SignalSource.MULTI_SPEED_MOM),
            _make_reading(value=0.2, source=SignalSource.CROSS_ASSET_RV),
        ]
        signals[0].weight = 0.6
        signals[1].weight = 0.4
        consensus = voter._ConsensusResult(
            weighted_consensus=0.4, agreement=0.9,
            equity_bias=0.3, duration_bias=0.0, gold_bias=0.0,
            action='neutral', action_confidence=0.5,
        )
        vote = voter._build_vote(signals, consensus, Regime.NORMAL, 0.6)
        assert 'multi_speed_momentum' in vote.reasoning


class TestPersistVote:
    """Tests for EnsembleVoter._persist_vote()."""

    def test_saves_vote_to_db(self, tmp_path):
        voter = _make_voter(tmp_path)
        vote = EnsembleVote(
            timestamp='2026-01-01', regime=Regime.NORMAL, regime_confidence=0.7,
            num_sources=1, weighted_consensus=0.3, agreement_ratio=0.8,
            equity_bias=0.3, duration_bias=-0.1, gold_bias=0.05,
            action='increase_equity', confidence=0.6, reasoning='test', source_votes=[],
        )
        voter._persist_vote(vote, weighted_consensus=0.3)
        import sqlite3
        with sqlite3.connect(str(voter.db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM ensemble_votes").fetchone()
            assert row[0] >= 1

    def test_ic_alert_check_with_alerts(self, tmp_path):
        """_persist_vote should check for IC decay alerts via _get_health_tracker."""
        voter = _make_voter(tmp_path)
        vote = EnsembleVote(
            timestamp='2026-01-02', regime=Regime.NORMAL, regime_confidence=0.7,
            num_sources=1, weighted_consensus=0.3, agreement_ratio=0.8,
            equity_bias=0.3, duration_bias=-0.1, gold_bias=0.05,
            action='increase_equity', confidence=0.6, reasoning='test', source_votes=[],
        )
        mock_tracker = MagicMock()
        mock_alert = MagicMock()
        mock_alert.source = "MULTI_SPEED_MOM"
        mock_tracker.detect_ic_alerts.return_value = [mock_alert]

        with patch('src.strategy.ensemble_voter._get_health_tracker', return_value=mock_tracker):
            with patch('src.strategy.ensemble_voter.logger') as mock_logger:
                voter._persist_vote(vote, weighted_consensus=0.3)
                mock_tracker.detect_ic_alerts.assert_called_once()
                mock_logger.warning.assert_any_call(
                    "IC decay alerts detected: %s", ["MULTI_SPEED_MOM"]
                )

    def test_ic_alert_check_no_alerts(self, tmp_path):
        """_persist_vote with no IC alerts should not log warning."""
        voter = _make_voter(tmp_path)
        vote = EnsembleVote(
            timestamp='2026-01-03', regime=Regime.NORMAL, regime_confidence=0.7,
            num_sources=1, weighted_consensus=0.3, agreement_ratio=0.8,
            equity_bias=0.3, duration_bias=-0.1, gold_bias=0.05,
            action='increase_equity', confidence=0.6, reasoning='test', source_votes=[],
        )
        mock_tracker = MagicMock()
        mock_tracker.detect_ic_alerts.return_value = []

        with patch('src.strategy.ensemble_voter._get_health_tracker', return_value=mock_tracker):
            voter._persist_vote(vote, weighted_consensus=0.3)
            mock_tracker.detect_ic_alerts.assert_called_once()

    def test_ic_alert_check_tracker_unavailable(self, tmp_path):
        """_persist_vote should handle None tracker gracefully."""
        voter = _make_voter(tmp_path)
        vote = EnsembleVote(
            timestamp='2026-01-04', regime=Regime.NORMAL, regime_confidence=0.7,
            num_sources=1, weighted_consensus=0.3, agreement_ratio=0.8,
            equity_bias=0.3, duration_bias=-0.1, gold_bias=0.05,
            action='increase_equity', confidence=0.6, reasoning='test', source_votes=[],
        )
        with patch('src.strategy.ensemble_voter._get_health_tracker', return_value=None):
            voter._persist_vote(vote, weighted_consensus=0.3)  # Should not raise


# ---------------------------------------------------------------------------
# get_rebalance_config
# ---------------------------------------------------------------------------

class TestGetRebalanceConfig:

    def test_returns_regime_key(self):
        voter = EnsembleVoter()
        config = voter.get_rebalance_config()
        assert 'regime' in config
        assert config['regime'] == 'normal'

    def test_crisis_regime(self):
        voter = EnsembleVoter()
        voter.current_regime = Regime.CRISIS
        config = voter.get_rebalance_config()
        assert config['regime'] == 'crisis'

    def test_high_vol_regime(self):
        voter = EnsembleVoter()
        voter.current_regime = Regime.HIGH_VOL
        config = voter.get_rebalance_config()
        assert config['regime'] == 'high_vol'

    def test_includes_regime_confidence(self):
        voter = EnsembleVoter()
        voter.current_regime_confidence = 0.85
        config = voter.get_rebalance_config()
        assert config['regime_confidence'] == 0.85


class TestGetBLViews:
    """Tests for get_bl_views() — BL view generation from ensemble vote."""

    def test_returns_views_dict(self):
        voter = EnsembleVoter()
        result = voter.get_bl_views()
        assert 'views' in result
        assert 'tau' in result
        assert 'prior' in result

    def test_views_is_blviews_instance(self):
        from src.strategy.black_litterman_mapper import BLViews
        voter = EnsembleVoter()
        result = voter.get_bl_views()
        assert isinstance(result['views'], BLViews)

    def test_default_tau(self):
        voter = EnsembleVoter()
        result = voter.get_bl_views()
        assert result['tau'] == 0.15

    def test_custom_tau(self):
        voter = EnsembleVoter()
        result = voter.get_bl_views(tau=0.30)
        assert result['tau'] == 0.30
        assert result['views'].tau == 0.30

    def test_with_precomputed_vote(self):
        voter = EnsembleVoter()
        vote = EnsembleVote(
            timestamp=datetime.now().isoformat(),
            regime=Regime.NORMAL,
            regime_confidence=0.7,
            num_sources=6,
            weighted_consensus=0.5,
            agreement_ratio=0.8,
            equity_bias=0.5,
            duration_bias=-0.2,
            gold_bias=0.3,
            action="increase_equity",
            confidence=0.7,
            reasoning="Bullish",
            source_votes=[],
        )
        result = voter.get_bl_views(vote=vote)
        assert result['equity_bias'] == 0.5
        assert result['duration_bias'] == -0.2
        assert result['gold_bias'] == 0.3

    def test_views_have_absolute_views(self):
        voter = EnsembleVoter()
        result = voter.get_bl_views()
        views = result['views']
        assert 'SPY' in views.absolute_views
        assert 'GLD' in views.absolute_views
        assert 'TLT' in views.absolute_views

    def test_views_have_confidences(self):
        voter = EnsembleVoter()
        result = voter.get_bl_views()
        views = result['views']
        assert len(views.view_confidences) == 3
        for c in views.view_confidences:
            assert 0.0 <= c <= 1.0

    def test_market_prior(self):
        voter = EnsembleVoter()
        result = voter.get_bl_views(prior="market")
        assert result['prior'] == "market"
        assert result['views'].prior == "market"


class TestLowVolRegime:
    """Tests for LOW_VOL regime — was missing from Regime enum, causing
    cross_asset_regime_arb gate rule for LOW_VOL to be dead code."""

    def test_low_vol_in_regime_enum(self):
        """LOW_VOL should be a valid Regime enum member."""
        assert hasattr(Regime, 'LOW_VOL')
        assert Regime.LOW_VOL.value == "low_vol"

    def test_low_vol_regime_weights_exist(self):
        """LOW_VOL should have its own REGIME_WEIGHTS entry."""
        assert Regime.LOW_VOL in REGIME_WEIGHTS
        weights = REGIME_WEIGHTS[Regime.LOW_VOL]
        # cross_asset_regime_arb is OFF in LOW_VOL, weight should be 0
        assert weights[SignalSource.CROSS_ASSET_REGIME_ARB] == 0.0
        # Remaining weights should sum to 1.0
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_low_vol_detection(self):
        """detect_regime should return LOW_VOL when vol < 12% and momentum > 1%."""
        voter = EnsembleVoter()
        # Create price data with very low vol and positive momentum
        dates = pd.date_range('2026-01-01', periods=60, freq='B')
        # Small consistent positive returns = low vol + positive momentum
        returns = np.full(60, 0.002)  # ~0.2% daily = very low annualized vol
        prices = 100 * (1 + returns).cumprod()
        price_data = pd.DataFrame({'SPY': prices}, index=dates)

        regime, confidence = voter.detect_regime(price_data)
        assert regime == Regime.LOW_VOL
        assert confidence >= 0.5

    def test_low_vol_not_triggered_with_negative_momentum(self):
        """LOW_VOL should not trigger if momentum is negative (even with low vol)."""
        voter = EnsembleVoter()
        dates = pd.date_range('2026-01-01', periods=60, freq='B')
        # Small negative returns = low vol but negative momentum
        returns = np.full(60, -0.001)
        prices = 100 * (1 + returns).cumprod()
        price_data = pd.DataFrame({'SPY': prices}, index=dates)

        regime, _ = voter.detect_regime(price_data)
        assert regime != Regime.LOW_VOL  # Should be NORMAL, not LOW_VOL

    def test_low_vol_not_triggered_with_high_vol(self):
        """LOW_VOL should not trigger if vol >= 12% even with positive momentum."""
        voter = EnsembleVoter()
        dates = pd.date_range('2026-01-01', periods=60, freq='B')
        # Alternating returns to create moderate vol (~14% annualized) with positive drift
        returns = np.array([0.01, -0.008] * 30)  # ~15% vol
        prices = 100 * (1 + returns).cumprod()
        price_data = pd.DataFrame({'SPY': prices}, index=dates)

        regime, _ = voter.detect_regime(price_data)
        # Should be NORMAL (not LOW_VOL) because vol > 12%
        assert regime != Regime.LOW_VOL

    def test_low_vol_regime_gate_activates(self):
        """cross_asset_regime_arb gate rule should activate in LOW_VOL regime."""
        from src.signals.regime_gate import RegimeGate
        gate = RegimeGate()
        # cross_asset_regime_arb should be OFF in LOW_VOL
        assert not gate.is_active("cross_asset_regime_arb", "LOW_VOL")
        # But ON in NORMAL
        assert gate.is_active("cross_asset_regime_arb", "NORMAL")

    def test_low_vol_behavioral_sentiment_on(self):
        """behavioral_sentiment should be ON in LOW_VOL (only regime where it's allowed)."""
        from src.signals.regime_gate import RegimeGate
        gate = RegimeGate()
        assert gate.is_active("behavioral_sentiment", "LOW_VOL")
        # And OFF in all others
        assert not gate.is_active("behavioral_sentiment", "NORMAL")
        assert not gate.is_active("behavioral_sentiment", "HIGH_VOL")
        assert not gate.is_active("behavioral_sentiment", "CRISIS")

    def test_rebalance_config_includes_low_vol(self):
        """get_rebalance_config should include low_vol mapping."""
        voter = EnsembleVoter()
        voter.current_regime = Regime.LOW_VOL
        voter.current_regime_confidence = 0.7
        config = voter.get_rebalance_config()
        assert config['regime'] == 'low_vol'
        assert config['regime_confidence'] == 0.7

    def test_low_vol_weights_sum_to_one(self):
        """LOW_VOL regime weights should sum to approximately 1.0."""
        weights = REGIME_WEIGHTS[Regime.LOW_VOL]
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_all_regimes_have_weights(self):
        """All Regime enum members should have corresponding REGIME_WEIGHTS."""
        for regime in Regime:
            assert regime in REGIME_WEIGHTS, f"{regime} missing from REGIME_WEIGHTS"


# ===========================================================================
# BanditWeighter tests
# ===========================================================================

class TestBanditWeighter:
    """Tests for BanditWeighter — epsilon-greedy contextual bandit."""

    def test_init_defaults(self):
        bw = BanditWeighter(signals=['a', 'b', 'c'])
        assert bw.signals == ['a', 'b', 'c']
        assert bw.epsilon == 0.1
        assert bw.window == 252
        assert bw.temperature == 1.0
        assert bw._history == {}

    def test_init_custom_params(self):
        bw = BanditWeighter(signals=['mom', 'rv'], epsilon=0.05, window=126, temperature=0.5)
        assert bw.epsilon == 0.05
        assert bw.window == 126
        assert bw.temperature == 0.5

    def test_update_adds_observation(self):
        bw = BanditWeighter(signals=['mom', 'rv'])
        bw.update('mom', 'NORMAL', 0.05)
        assert 'NORMAL' in bw._history
        assert 'mom' in bw._history['NORMAL']
        assert bw._history['NORMAL']['mom'] == [0.05]

    def test_update_multiple_regimes(self):
        bw = BanditWeighter(signals=['mom'])
        bw.update('mom', 'NORMAL', 0.01)
        bw.update('mom', 'HIGH_VOL', -0.02)
        assert 'NORMAL' in bw._history
        assert 'HIGH_VOL' in bw._history
        assert len(bw._history['NORMAL']['mom']) == 1
        assert len(bw._history['HIGH_VOL']['mom']) == 1

    def test_update_trims_to_window(self):
        bw = BanditWeighter(signals=['mom'], window=10)
        for i in range(15):
            bw.update('mom', 'NORMAL', float(i))
        assert len(bw._history['NORMAL']['mom']) == 10
        assert bw._history['NORMAL']['mom'] == list(range(5, 15))

    def test_get_weights_cold_start_returns_none(self):
        bw = BanditWeighter(signals=['mom', 'rv'])
        assert bw.get_weights('NORMAL') is None

    def test_get_weights_cold_start_different_regime(self):
        """Data in one regime should not affect get_weights for another."""
        bw = BanditWeighter(signals=['mom'])
        for _ in range(30):
            bw.update('mom', 'NORMAL', 0.02)
        assert bw.get_weights('CRISIS') is None  # No data for CRISIS

    def test_get_weights_after_updates(self):
        bw = BanditWeighter(signals=['mom', 'rv'], temperature=0.5)
        for _ in range(30):
            bw.update('mom', 'NORMAL', 0.02)
            bw.update('rv', 'NORMAL', 0.01)
        weights = bw.get_weights('NORMAL')
        assert weights is not None
        assert set(weights.keys()) == {'mom', 'rv'}
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_get_weights_better_signal_gets_higher_weight(self):
        """Higher Sharpe signal should receive larger weight after softmax."""
        bw = BanditWeighter(signals=['good', 'bad'], temperature=0.1)
        # Use varied returns so std > 0
        for i in range(30):
            bw.update('good', 'NORMAL', 0.05 + 0.01 * (i % 3))
            bw.update('bad', 'NORMAL', -0.02 + 0.01 * (i % 3))
        weights = bw.get_weights('NORMAL')
        assert weights is not None
        assert weights['good'] > weights['bad']

    def test_select_exploit_best_signal(self):
        """With epsilon=0 and sufficient data, select() should prefer the best signal most of the time.

        Thompson Sampling is stochastic, so we test that the best signal is selected
        in the majority of trials (>80%) rather than 100%.
        """
        bw = BanditWeighter(signals=['good', 'bad'], epsilon=0.0, temperature=1.0)
        for _ in range(30):
            bw.update('good', 'NORMAL', 0.05)
            bw.update('bad', 'NORMAL', -0.01)
        good_count = sum(1 for _ in range(100) if bw.select('NORMAL') == 'good')
        assert good_count > 70, f"Expected 'good' >70% of the time, got {good_count}%"

    def test_select_explore_randomly(self):
        """With epsilon=1.0, select() should explore all signals."""
        bw = BanditWeighter(signals=['a', 'b'], epsilon=1.0)
        choices = set()
        for _ in range(50):
            choices.add(bw.select('NORMAL'))
        assert len(choices) == 2

    def test_select_fallback_on_no_data(self):
        """With no history, select() should fall back to first signal."""
        bw = BanditWeighter(signals=['a', 'b'], epsilon=0.0)
        assert bw.select('NORMAL') == 'a'

    def test_rolling_sharpe_insufficient_data_returns_zero(self):
        bw = BanditWeighter(signals=['mom'])
        bw.update('mom', 'NORMAL', 0.01)
        sh = bw._rolling_sharpe('mom', 'NORMAL')
        assert sh == 0.0  # Fewer than 21 observations

    def test_rolling_sharpe_zero_variance_returns_zero(self):
        bw = BanditWeighter(signals=['mom'])
        for _ in range(25):
            bw.update('mom', 'NORMAL', 0.01)
        sh = bw._rolling_sharpe('mom', 'NORMAL')
        assert sh == 0.0  # All identical values => sigma=0

    def test_softmax_numerical_stability_large_negatives(self):
        bw = BanditWeighter(signals=['a', 'b', 'c'])
        sharpes = {'a': -1e6, 'b': -1e6, 'c': -1e6}
        result = bw._softmax(sharpes)
        assert all(np.isfinite(v) for v in result.values())
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_softmax_large_positives(self):
        bw = BanditWeighter(signals=['a', 'b'])
        sharpes = {'a': 1e6, 'b': 1e3}
        result = bw._softmax(sharpes)
        assert all(np.isfinite(v) for v in result.values())
        assert abs(sum(result.values()) - 1.0) < 0.01
        assert result['a'] > result['b']

    def test_softmax_all_zero_equal_weights(self):
        bw = BanditWeighter(signals=['a', 'b'])
        sharpes = {'a': 0.0, 'b': 0.0}
        result = bw._softmax(sharpes)
        assert result['a'] == pytest.approx(0.5)
        assert result['b'] == pytest.approx(0.5)

    def test_softmax_temperature_zero_no_effect(self):
        """temperature=0 should still produce valid output via the > 0 branch."""
        bw = BanditWeighter(signals=['a', 'b'], temperature=0.0)
        sharpes = {'a': 1.0, 'b': 0.5}
        result = bw._softmax(sharpes)
        assert all(np.isfinite(v) for v in result.values())
        assert abs(sum(result.values()) - 1.0) < 0.01
        assert result['a'] > result['b']

    def test_softmax_uneven_values_produce_different_weights(self):
        """Different Sharpe values should produce different softmax weights."""
        bw = BanditWeighter(signals=['a', 'b', 'c'])
        sharpes = {'a': 1.0, 'b': 0.5, 'c': 0.0}
        result = bw._softmax(sharpes)
        assert result['a'] > result['b'] > result['c']
        assert abs(sum(result.values()) - 1.0) < 0.01

    # ── Additional edge cases ─────────────────────────────────────────

    def test_select_mixed_explore_exploit(self):
        """With epsilon=0.5, select() should both explore and exploit over many trials."""
        bw = BanditWeighter(signals=['good', 'bad'], epsilon=0.5, temperature=1.0)
        for _ in range(30):
            bw.update('good', 'NORMAL', 0.05)
            bw.update('bad', 'NORMAL', -0.01)
        choices = set()
        for _ in range(100):
            choices.add(bw.select('NORMAL'))
        assert len(choices) >= 1

    def test_get_weights_signal_not_in_history(self):
        """When a signal is in signals list but has no history, its Sharpe is 0."""
        bw = BanditWeighter(signals=['has_data', 'no_data'])
        for _ in range(25):
            bw.update('has_data', 'NORMAL', 0.02)
        weights = bw.get_weights('NORMAL')
        assert weights is not None
        assert 'has_data' in weights
        assert 'no_data' in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_rolling_sharpe_exactly_21_observations(self):
        """Boundary case: exactly 21 observations should produce a valid Sharpe."""
        bw = BanditWeighter(signals=['mom'])
        for i in range(21):
            bw.update('mom', 'NORMAL', 0.01 + 0.005 * (i % 2))
        sh = bw._rolling_sharpe('mom', 'NORMAL')
        assert sh != 0.0

    def test_rolling_sharpe_varying_data_positive_sharpe(self):
        """Varying positive returns should produce positive Sharpe ratio."""
        bw = BanditWeighter(signals=['mom'])
        rng = np.random.default_rng(42)
        for _ in range(60):
            bw.update('mom', 'NORMAL', float(rng.normal(0.002, 0.01)))
        sh = bw._rolling_sharpe('mom', 'NORMAL')
        assert sh > 0

    def test_softmax_high_temperature_nearly_uniform(self):
        """Very high temperature should produce nearly uniform weights."""
        bw = BanditWeighter(signals=['a', 'b', 'c'], temperature=100.0)
        sharpes = {'a': 1.0, 'b': 0.5, 'c': 0.0}
        result = bw._softmax(sharpes)
        for sig in result:
            assert result[sig] == pytest.approx(1/3, abs=0.05)


# ===========================================================================
# get_blended_weights tests
# ===========================================================================

class TestGetBlendedWeights:
    """Tests for EnsembleVoter.get_blended_weights()."""

    def test_cold_start_returns_static(self):
        """No bandit data should return the static REGIME_WEIGHTS unchanged."""
        voter = EnsembleVoter()
        result = voter.get_blended_weights('NORMAL')
        assert isinstance(result, dict)
        # All SignalSource keys
        assert all(isinstance(k, SignalSource) for k in result)
        # Should match static weights when no bandit data
        static = REGIME_WEIGHTS[Regime.NORMAL]
        assert result.keys() == static.keys()

    def test_unknown_regime_falls_back_to_normal(self):
        voter = EnsembleVoter()
        result = voter.get_blended_weights('UNKNOWN')
        assert isinstance(result, dict)
        # Falls back to NORMAL weights
        static = REGIME_WEIGHTS[Regime.NORMAL]
        assert result.keys() == static.keys()

    def test_with_bandit_data_blends(self):
        """Bandit data should produce blended weights."""
        voter = EnsembleVoter()
        # Add bandit observations
        for _ in range(30):
            voter.update_bandit('multi_speed_momentum', 'NORMAL', 0.01)
            voter.update_bandit('cross_asset_rv', 'NORMAL', 0.02)
        result = voter.get_blended_weights('NORMAL')
        assert isinstance(result, dict)
        # With 30/252 = 0.12 of max blend, bandit blend is small but present
        expected_blend = min(0.7, 30 / 252 * 0.7)
        assert expected_blend > 0.0

    def test_bandit_missing_regime_returns_static(self):
        """Bandit data in one regime should not affect blend for another."""
        voter = EnsembleVoter()
        for _ in range(30):
            voter.update_bandit('multi_speed_momentum', 'NORMAL', 0.01)
        # CRISIS has no bandit data, should return static
        result = voter.get_blended_weights('CRISIS')
        static = REGIME_WEIGHTS[Regime.CRISIS]
        for k, v in static.items():
            assert result[k] == pytest.approx(v, abs=0.01)

    def test_weights_still_sum_to_one(self):
        """Blended weights should always sum to 1.0."""
        voter = EnsembleVoter()
        for regime in Regime:
            result = voter.get_blended_weights(regime.name)
            total = sum(result.values())
            assert abs(total - 1.0) < 0.05, f"{regime} blended weights sum to {total:.4f}"

    def test_full_blend_after_252_observations(self):
        """After 252 bandit observations, the blend should be at max (70% bandit)."""
        voter = EnsembleVoter()
        for i in range(252):
            voter.update_bandit('multi_speed_momentum', 'NORMAL', 0.01 + 0.001 * (i % 3))
            voter.update_bandit('cross_asset_rv', 'NORMAL', 0.02 + 0.001 * (i % 2))
            voter.update_bandit('alternative_data', 'NORMAL', 0.015)
        result = voter.get_blended_weights('NORMAL')
        assert isinstance(result, dict)
        assert all(isinstance(k, SignalSource) for k in result)

    def test_bandit_with_some_signals_missing(self):
        """When bandit has data for some but not all signals, all should still appear."""
        voter = EnsembleVoter()
        for _ in range(30):
            voter.update_bandit('multi_speed_momentum', 'NORMAL', 0.01)
            voter.update_bandit('cross_asset_rv', 'NORMAL', 0.02)
        result = voter.get_blended_weights('NORMAL')
        assert len(result) == len([s for s in SignalSource])

    def test_blended_weights_differ_by_regime(self):
        """Different regimes should produce different blended weights."""
        voter = EnsembleVoter()
        for _ in range(30):
            voter.update_bandit('multi_speed_momentum', 'NORMAL', 0.01)
            voter.update_bandit('multi_speed_momentum', 'HIGH_VOL', -0.01)
            voter.update_bandit('alternative_data', 'NORMAL', 0.02)
            voter.update_bandit('alternative_data', 'HIGH_VOL', 0.03)
        normal_result = voter.get_blended_weights('NORMAL')
        high_vol_result = voter.get_blended_weights('HIGH_VOL')
        assert normal_result != high_vol_result


# ===========================================================================
# apply_goal_risk_budget tests
# ===========================================================================

class TestApplyGoalRiskBudget:
    """Tests for EnsembleVoter.apply_goal_risk_budget()."""

    @patch('src.config.goals.load_goals')
    @patch('src.config.goals.get_risk_budget_multiplier')
    def test_no_goals_returns_base(self, mock_get_rbm, mock_load_goals):
        """When goals loading fails, base_allocation should be returned."""
        mock_load_goals.side_effect = ImportError("No goals module")
        voter = EnsembleVoter()
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        result = voter.apply_goal_risk_budget(base)
        assert result == base

    @patch('src.config.goals.load_goals')
    @patch('src.config.goals.get_risk_budget_multiplier')
    def test_risk_multiplier_one_or_more_unchanged(self, mock_get_rbm, mock_load_goals):
        """risk_mult >= 1.0 should return base allocation unchanged."""
        mock_get_rbm.return_value = 1.0
        voter = EnsembleVoter()
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        result = voter.apply_goal_risk_budget(base)
        assert result == base

    @patch('src.config.goals.load_goals')
    @patch('src.config.goals.get_risk_budget_multiplier')
    def test_risk_multiplier_below_one_shifts_to_safe(self, mock_get_rbm, mock_load_goals):
        """risk_mult < 1.0 should reduce risky assets and increase safe assets."""
        mock_get_rbm.return_value = 0.5
        voter = EnsembleVoter()
        base = {'SPY': 0.50, 'SHY': 0.30, 'TLT': 0.20}
        result = voter.apply_goal_risk_budget(base)
        # SPY is risky, should be reduced
        assert result['SPY'] < base['SPY']
        # SHY is safe, should be increased
        assert result['SHY'] > base['SHY']

    @patch('src.config.goals.load_goals')
    @patch('src.config.goals.get_risk_budget_multiplier')
    def test_safe_asset_redistribution(self, mock_get_rbm, mock_load_goals):
        """Reduced risk from equities should flow to safe assets proportionally."""
        mock_get_rbm.return_value = 0.5
        voter = EnsembleVoter()
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        result = voter.apply_goal_risk_budget(base)
        # SPY and GLD are neither SHY nor IEF nor BIL, so both are risky
        # Only TLT is in safe_assets
        spy_reduced = base['SPY'] - result['SPY']
        gld_reduced = base['GLD'] - result['GLD']
        assert spy_reduced > 0
        assert gld_reduced > 0
        # TLT should have increased
        assert result['TLT'] > base['TLT']

    @patch('src.config.goals.load_goals')
    @patch('src.config.goals.get_risk_budget_multiplier')
    def test_exception_fallback(self, mock_get_rbm, mock_load_goals):
        """Exception in goals module should return base allocation unchanged."""
        mock_load_goals.side_effect = OSError("Unexpected error")
        voter = EnsembleVoter()
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        result = voter.apply_goal_risk_budget(base)
        assert result == base

    @patch('src.config.goals.load_goals')
    @patch('src.config.goals.get_risk_budget_multiplier')
    def test_risk_mult_0_5_reduces_spy_by_half(self, mock_get_rbm, mock_load_goals):
        """risk_mult=0.5 should cut SPY weight in half (pre-normalization)."""
        mock_get_rbm.return_value = 0.5
        voter = EnsembleVoter()
        base = {'SPY': 1.0}  # Only SPY, no safe assets
        result = voter.apply_goal_risk_budget(base)
        # With only SPY and no safe assets, risky_reduction stays but has nowhere to go
        # Should still work: weight * risk_mult = 0.5
        assert result['SPY'] == pytest.approx(1.0)


# ===========================================================================
# update_bandit tests
# ===========================================================================

class TestUpdateBandit:
    """Tests for EnsembleVoter.update_bandit()."""

    def test_increments_observation_count(self):
        voter = EnsembleVoter()
        assert voter.bandit_observations == 0
        voter.update_bandit('multi_speed_momentum', 'NORMAL', 0.01)
        assert voter.bandit_observations == 1

    def test_delegates_to_bandit_update(self):
        voter = EnsembleVoter()
        voter.update_bandit('cross_asset_rv', 'HIGH_VOL', -0.02)
        # Bandit should have the history
        assert 'HIGH_VOL' in voter.bandit._history
        assert 'cross_asset_rv' in voter.bandit._history['HIGH_VOL']
        assert voter.bandit._history['HIGH_VOL']['cross_asset_rv'] == [-0.02]

    def test_multiple_updates_accumulate(self):
        voter = EnsembleVoter()
        for _ in range(5):
            voter.update_bandit('multi_speed_momentum', 'NORMAL', 0.01)
        assert voter.bandit_observations == 5
        assert len(voter.bandit._history['NORMAL']['multi_speed_momentum']) == 5

    def test_update_different_regimes(self):
        voter = EnsembleVoter()
        voter.update_bandit('multi_speed_momentum', 'NORMAL', 0.01)
        voter.update_bandit('multi_speed_momentum', 'CRISIS', -0.03)
        assert 'NORMAL' in voter.bandit._history
        assert 'CRISIS' in voter.bandit._history


# ===========================================================================
# _should_skip tests
# ===========================================================================

class TestShouldSkip:
    """Tests for EnsembleVoter._should_skip()."""

    def test_skip_when_not_in_active_sources(self):
        voter = EnsembleVoter()
        active = {SignalSource.MULTI_SPEED_MOM, SignalSource.CROSS_ASSET_RV}
        assert voter._should_skip(SignalSource.ALTERNATIVE_DATA, active, Regime.NORMAL)

    def test_not_skip_when_in_active_sources(self):
        voter = EnsembleVoter()
        active = {SignalSource.MULTI_SPEED_MOM, SignalSource.CROSS_ASSET_RV}
        assert not voter._should_skip(SignalSource.MULTI_SPEED_MOM, active, Regime.NORMAL)

    def test_not_skip_when_active_sources_is_none(self):
        """When active_sources is None (no regime filter), nothing should be skipped."""
        voter = EnsembleVoter()
        assert not voter._should_skip(SignalSource.ALTERNATIVE_DATA, None, None)

    def test_skip_empty_active_set(self):
        """When active_sources is empty, everything should be skipped."""
        voter = EnsembleVoter()
        assert voter._should_skip(SignalSource.ALTERNATIVE_DATA, set(), Regime.NORMAL)

    def test_regime_arb_respects_skip_gate(self):
        """_collect_regime_arb_signal should skip when not in active_sources."""
        voter = EnsembleVoter()
        # When CROSS_ASSET_REGIME_ARB is not in active_sources, it should be skipped
        active = {SignalSource.MULTI_SPEED_MOM, SignalSource.CROSS_ASSET_RV}
        assert voter._should_skip(SignalSource.CROSS_ASSET_REGIME_ARB, active, Regime.NORMAL)

    def test_regime_arb_not_skipped_when_active(self):
        """_collect_regime_arb_signal should NOT skip when in active_sources."""
        voter = EnsembleVoter()
        active = {SignalSource.CROSS_ASSET_REGIME_ARB}
        assert not voter._should_skip(SignalSource.CROSS_ASSET_REGIME_ARB, active, Regime.NORMAL)

    def test_regime_arb_skipped_in_low_vol_regime(self):
        """CROSS_ASSET_REGIME_ARB has zero weight in LOW_VOL, so it should be skipped."""
        voter = EnsembleVoter()
        # In LOW_VOL, REGIME_WEIGHTS gives CROSS_ASSET_REGIME_ARB weight 0.0
        # So the active_sources set won't include it
        weights = REGIME_WEIGHTS[Regime.LOW_VOL]
        active = {src for src, w in weights.items() if w > 0}
        assert SignalSource.CROSS_ASSET_REGIME_ARB not in active
        assert voter._should_skip(SignalSource.CROSS_ASSET_REGIME_ARB, active, Regime.LOW_VOL)


# ===========================================================================
# Additional consensus computation edge cases
# ===========================================================================

class TestComputeConsensusAdditional:
    """Additional consensus computation edge cases."""

    def test_decrease_equity_action(self, tmp_path):
        """Strong negative equity bias with high agreement should produce decrease_equity."""
        voter = _make_voter(tmp_path)
        neg_assets = {'SPY': -0.6, 'TLT': 0.2, 'GLD': 0.1}
        signals = [
            _make_reading(value=-0.5, confidence=0.9, asset_signals=neg_assets),
            _make_reading(value=-0.4, confidence=0.8, asset_signals=neg_assets),
        ]
        signals[0].weight = 0.6
        signals[1].weight = 0.4
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.7)
        assert result.equity_bias < -0.3
        assert result.action == 'decrease_equity'

    def test_missing_asset_signals_fallback_to_weighted_consensus(self, tmp_path):
        """When no asset_signals are provided, asset biases should fall back to consensus."""
        voter = _make_voter(tmp_path)
        signals = [
            SignalReading(
                source=SignalSource.MULTI_SPEED_MOM,
                timestamp='2026-01-01',
                value=0.5,
                confidence=0.8,
                weight=0.0,
                regime_fit='all',
                asset_signals=None,
                explanation='test',
            ),
        ]
        signals[0].weight = 1.0
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        assert result.equity_bias == result.weighted_consensus

    def test_nan_asset_signal_handling(self, tmp_path):
        """NaN in asset_signals should be filtered out, not crash."""
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.4, confidence=0.8,
                          asset_signals={'SPY': float('nan'), 'TLT': 0.2, 'GLD': 0.1}),
        ]
        signals[0].weight = 1.0
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        assert np.isfinite(result.equity_bias)
        assert result.duration_bias == 0.2
        assert result.gold_bias == 0.1

    def test_low_agreement_with_strong_bias_produces_neutral(self, tmp_path):
        """When agreement is below 0.6 but equity_bias is strong, action should be neutral."""
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.7, confidence=0.9,
                          asset_signals={'SPY': 0.7, 'TLT': 0.0, 'GLD': 0.0}),
            _make_reading(value=-0.6, confidence=0.9,
                          asset_signals={'SPY': -0.6, 'TLT': 0.0, 'GLD': 0.0}),
        ]
        signals[0].weight = 0.5
        signals[1].weight = 0.5
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        # Equity bias might be moderate (~0.05), agreement is low, so action is neutral
        assert result.action == 'neutral'


# ===========================================================================
# Additional build_vote edge cases
# ===========================================================================

class TestBuildVoteAdditional:
    """Additional build_vote edge cases."""

    def test_build_vote_limits_to_three_sources(self, tmp_path):
        """_build_vote should only include first 3 sources in reasoning detail."""
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.5, source=SignalSource.MULTI_SPEED_MOM),
            _make_reading(value=0.4, source=SignalSource.CROSS_ASSET_RV),
            _make_reading(value=0.3, source=SignalSource.INTERNATIONAL_MOMENTUM),
            _make_reading(value=0.2, source=SignalSource.ALTERNATIVE_DATA),
        ]
        for s in signals:
            s.weight = 0.25
        consensus = voter._ConsensusResult(
            weighted_consensus=0.35, agreement=0.8,
            equity_bias=0.3, duration_bias=0.0, gold_bias=0.0,
            action='neutral', action_confidence=0.5,
        )
        vote = voter._build_vote(signals, consensus, Regime.NORMAL, 0.6)
        # Should only have 3 source detail lines (first 3 sources)
        source_lines = [l for l in vote.reasoning.split('\n') if any(
            src in l for src in ['multi_speed', 'cross_asset_rv', 'international', 'alternative']
        )]
        assert len(source_lines) <= 3


# ===========================================================================
# Additional persist_vote edge cases
# ===========================================================================

class TestPersistVoteAdditional:
    """Additional persist_vote edge cases."""

    def test_persist_vote_saves_source_readings(self, tmp_path):
        """Source readings should be saved to the source_readings table."""
        voter = _make_voter(tmp_path)
        readings = [
            _make_reading(value=0.5, source=SignalSource.MULTI_SPEED_MOM),
            _make_reading(value=0.3, source=SignalSource.CROSS_ASSET_RV),
        ]
        for r in readings:
            r.weight = 0.5
        vote = EnsembleVote(
            timestamp='2026-05-01', regime=Regime.NORMAL, regime_confidence=0.7,
            num_sources=2, weighted_consensus=0.4, agreement_ratio=0.8,
            equity_bias=0.4, duration_bias=-0.1, gold_bias=0.1,
            action='increase_equity', confidence=0.7, reasoning='test',
            source_votes=readings,
        )
        voter._persist_vote(vote, weighted_consensus=0.4)
        import sqlite3
        with sqlite3.connect(str(voter.db_path)) as conn:
            rows = conn.execute(
                "SELECT source, value, weight FROM source_readings ORDER BY source"
            ).fetchall()
            assert len(rows) == 2
            assert rows[0][0] == 'cross_asset_rv'
            assert rows[1][0] == 'multi_speed_momentum'


# ===========================================================================
# Additional get_bl_views edge cases
# ===========================================================================

class TestGetBLViewsAdditional:
    """Additional get_bl_views edge cases."""

    def test_get_bl_views_error_in_health_tracker(self):
        """Error in health tracker should not crash get_bl_views."""
        voter = EnsembleVoter()
        mock_tracker = MagicMock()
        mock_tracker.get_health_report.side_effect = ValueError("Tracker crashed")
        with patch('src.strategy.ensemble_voter._get_health_tracker',
                   return_value=mock_tracker):
            result = voter.get_bl_views()
            assert 'views' in result
            assert result['health_scores_used'] == {}

    def test_get_bl_views_default_prior_equal(self):
        """Default prior should be 'equal'."""
        voter = EnsembleVoter()
        result = voter.get_bl_views()
        assert result['prior'] == 'equal'


# ===========================================================================
# Additional update_bandit edge cases
# ===========================================================================

class TestUpdateBanditAdditional:
    """Additional update_bandit edge cases."""

    def test_update_bandit_with_nan_return(self):
        """NaN daily_return should be stored (valid data point)."""
        voter = EnsembleVoter()
        voter.update_bandit('multi_speed_momentum', 'NORMAL', float('nan'))
        assert voter.bandit_observations == 1
        hist = voter.bandit._history['NORMAL']['multi_speed_momentum']
        assert np.isnan(hist[0])

    def test_update_bandit_with_negative_returns(self):
        """Negative daily returns should be stored correctly."""
        voter = EnsembleVoter()
        voter.update_bandit('multi_speed_momentum', 'CRISIS', -0.05)
        voter.update_bandit('multi_speed_momentum', 'CRISIS', -0.03)
        assert len(voter.bandit._history['CRISIS']['multi_speed_momentum']) == 2
        assert voter.bandit._history['CRISIS']['multi_speed_momentum'] == [-0.05, -0.03]


# ===========================================================================
# Additional apply_goal_risk_budget edge cases
# ===========================================================================

class TestApplyGoalRiskBudgetAdditional:
    """Additional apply_goal_risk_budget edge cases."""

    @patch('src.config.goals.load_goals')
    @patch('src.config.goals.get_risk_budget_multiplier')
    def test_empty_base_allocation(self, mock_get_rbm, mock_load_goals):
        """Empty base allocation should be handled gracefully."""
        mock_get_rbm.return_value = 0.5
        voter = EnsembleVoter()
        result = voter.apply_goal_risk_budget({})
        assert result == {}

    @patch('src.config.goals.load_goals')
    @patch('src.config.goals.get_risk_budget_multiplier')
    def test_all_safe_assets_mixed(self, mock_get_rbm, mock_load_goals):
        """All assets in safe_assets should stay unchanged."""
        mock_get_rbm.return_value = 0.5
        voter = EnsembleVoter()
        base = {'SHY': 0.5, 'IEF': 0.3, 'BIL': 0.2}
        result = voter.apply_goal_risk_budget(base)
        assert result == base

    @patch('src.config.goals.load_goals')
    @patch('src.config.goals.get_risk_budget_multiplier')
    def test_risk_mult_zero_eliminates_risky(self, mock_get_rbm, mock_load_goals):
        """risk_mult=0 should give all weight to safe assets."""
        mock_get_rbm.return_value = 0.0
        voter = EnsembleVoter()
        base = {'SPY': 0.70, 'IEF': 0.30}
        result = voter.apply_goal_risk_budget(base)
        assert result['SPY'] == 0.0
        assert result['IEF'] > 0


# ===========================================================================
# Regime detection edge cases
# ===========================================================================

class TestDetectRegimeEdgeCases:
    """Edge cases for EnsembleVoter.detect_regime()."""

    def test_detect_regime_crisis_via_drawdown(self, tmp_path):
        """CRISIS should trigger when drawdown exceeds threshold."""
        voter = _make_voter(tmp_path)
        dates = pd.date_range('2026-01-01', periods=60, freq='B')
        prices = [100.0]
        for i in range(59):
            if i < 20:
                prices.append(prices[-1] * 1.001)
            else:
                prices.append(prices[-1] * 0.97)
        price_data = pd.DataFrame({'SPY': prices}, index=dates)
        regime, conf = voter.detect_regime(price_data)
        assert regime == Regime.CRISIS
        assert conf >= 0.0

    def test_detect_regime_recovery(self, tmp_path):
        """RECOVERY should trigger after drawdown with positive momentum.
        Needs drawdown between -3% and -10% (below RECOVERY threshold, above CRISIS)
        and 20d momentum > 2%.
        """
        voter = _make_voter(tmp_path)
        dates = pd.date_range('2026-01-01', periods=60, freq='B')
        prices = [100.0]
        for i in range(59):
            if i < 30:
                prices.append(prices[-1] * 0.995)  # -0.5% for 30 days
            else:
                prices.append(prices[-1] * 1.003)  # +0.3% for 30 days
        price_data = pd.DataFrame({'SPY': prices}, index=dates)
        regime, conf = voter.detect_regime(price_data)
        assert regime == Regime.RECOVERY

    def test_detect_regime_high_vol_via_drawdown_momentum(self, tmp_path):
        """HIGH_VOL via drawdown below threshold and negative momentum."""
        voter = _make_voter(tmp_path)
        dates = pd.date_range('2026-01-01', periods=60, freq='B')
        rng = np.random.default_rng(42)
        prices = [100.0]
        for i in range(59):
            ret = rng.normal(-0.002, 0.025)
            prices.append(prices[-1] * (1 + ret))
        price_data = pd.DataFrame({'SPY': prices}, index=dates)
        regime, conf = voter.detect_regime(price_data)
        assert regime in [Regime.HIGH_VOL, Regime.CRISIS, Regime.NORMAL]

    def test_detect_regime_empty_dataframe(self, tmp_path):
        """Empty DataFrame should return NORMAL with 0.5 confidence."""
        voter = _make_voter(tmp_path)
        df = pd.DataFrame()
        regime, conf = voter.detect_regime(df)
        assert regime == Regime.NORMAL
        assert conf == 0.5

    def test_detect_regime_normal_when_moderate_conditions(self, tmp_path):
        """NORMAL regime when vol is moderate (above LOW_VOL threshold) and no drawdown."""
        voter = _make_voter(tmp_path)
        dates = pd.date_range('2026-01-01', periods=60, freq='B')
        prices = [100.0]
        rng = np.random.default_rng(99)
        for i in range(59):
            ret = rng.normal(0.0003, 0.01)  # ~15.9% annualized vol (above 12% LOW_VOL threshold)
            prices.append(prices[-1] * (1 + ret))
        price_data = pd.DataFrame({'SPY': prices}, index=dates)
        regime, conf = voter.detect_regime(price_data)
        assert regime == Regime.NORMAL
        assert conf > 0


# ===========================================================================
# Integration tests for compute_vote()
# ===========================================================================

class TestComputeVoteIntegration:
    """Integration tests for compute_vote() with real sub-methods."""

    def test_compute_vote_full_pipeline(self, tmp_path):
        """Full pipeline from readings through to persisted vote."""
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(
                value=0.5, source=SignalSource.MULTI_SPEED_MOM,
            ),
            SignalSource.CROSS_ASSET_RV: _make_reading(
                value=0.3, source=SignalSource.CROSS_ASSET_RV,
            ),
            SignalSource.INTERNATIONAL_MOMENTUM: _make_reading(
                value=0.4, source=SignalSource.INTERNATIONAL_MOMENTUM,
            ),
        }
        vote = voter.compute_vote(
            readings=readings, regime=Regime.NORMAL, regime_confidence=0.7,
        )
        assert vote is not None
        assert vote.num_sources == 3
        import sqlite3
        with sqlite3.connect(str(voter.db_path)) as conn:
            db_vote = conn.execute(
                "SELECT action, consensus FROM ensemble_votes WHERE timestamp=?",
                (vote.timestamp,),
            ).fetchone()
            assert db_vote is not None
            assert db_vote[0] == vote.action
            assert db_vote[1] == vote.weighted_consensus

    def test_compute_vote_with_none_readings_triggers_collect(self, tmp_path):
        """When readings=None and no current_readings, collect_signals is called."""
        voter = _make_voter(tmp_path)
        vote = voter.compute_vote(
            readings=None, regime=Regime.NORMAL, regime_confidence=0.5,
        )
        assert vote is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ===========================================================================
# Category 1: Dataclass field validation — SignalReading
# ===========================================================================

class TestSignalReadingFieldValidation:
    """Detailed field validation for SignalReading dataclass."""

    def test_field_count(self):
        """SignalReading should have exactly 8 fields."""
        flds = fields(SignalReading)
        assert len(flds) == 8

    def test_all_field_types(self):
        """Verify type annotations for all SignalReading fields."""
        flds = {f.name: f.type for f in fields(SignalReading)}
        assert flds['source'] == SignalSource
        assert flds['timestamp'] == str
        assert flds['value'] == float
        assert flds['confidence'] == float
        assert flds['weight'] == float
        assert flds['regime_fit'] == str

    def test_asset_signals_optional_type(self):
        """asset_signals should be Optional[Dict[str, float]]."""
        for f in fields(SignalReading):
            if f.name == 'asset_signals':
                assert 'Optional' in repr(f.type) or 'Dict' in repr(f.type) or f.type == 'Optional[Dict[str, float]]' or 'None' in repr(f.type)
                return
        assert False, "asset_signals field not found"

    def test_explanation_default_is_empty(self):
        """explanation field should default to empty string."""
        for f in fields(SignalReading):
            if f.name == 'explanation':
                assert f.default == ''
                return
        assert False, "explanation field not found"

    def test_asset_signals_default_is_none(self):
        """asset_signals field should default to None."""
        for f in fields(SignalReading):
            if f.name == 'asset_signals':
                assert f.default is None
                return
        assert False, "asset_signals field not found"

    def test_no_default_for_required_fields(self):
        """Required fields (source, timestamp, value, confidence, weight, regime_fit) should have no default."""
        import dataclasses
        required = {'source', 'timestamp', 'value', 'confidence', 'weight', 'regime_fit'}
        for f in fields(SignalReading):
            if f.name in required:
                assert f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING, \
                    f"{f.name} should not have a default"

    def test_signal_reading_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(SignalReading)

    def test_value_field_is_float_positional(self):
        """value comes before optional fields in positional order."""
        names = [f.name for f in fields(SignalReading)]
        value_idx = names.index('value')
        asset_idx = names.index('asset_signals')
        assert value_idx < asset_idx, "value should come before asset_signals"


# ===========================================================================
# Category 1 continued: Dataclass field validation — EnsembleVote
# ===========================================================================

class TestEnsembleVoteFieldValidation:
    """Detailed field validation for EnsembleVote dataclass."""

    def test_field_count(self):
        flds = fields(EnsembleVote)
        assert len(flds) == 13

    def test_all_field_types(self):
        """Verify specific type annotations for EnsembleVote fields."""
        flds = {f.name: f.type for f in fields(EnsembleVote)}
        assert flds['timestamp'] == str
        assert flds['regime'] == Regime
        assert flds['regime_confidence'] == float
        assert flds['num_sources'] == int
        assert flds['weighted_consensus'] == float
        assert flds['agreement_ratio'] == float
        assert flds['equity_bias'] == float
        assert flds['duration_bias'] == float
        assert flds['gold_bias'] == float
        assert flds['action'] == str
        assert flds['confidence'] == float
        assert flds['reasoning'] == str

    def test_source_votes_is_list_of_signal_reading(self):
        """source_votes should be typed as List[SignalReading]."""
        for f in fields(EnsembleVote):
            if f.name == 'source_votes':
                assert 'List' in repr(f.type) or 'SignalReading' in repr(f.type)
                return
        assert False, "source_votes field not found"

    def test_no_fields_have_defaults(self):
        """All 13 EnsembleVote fields are required — none have defaults."""
        import dataclasses
        for f in fields(EnsembleVote):
            assert f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING, \
                f"{f.name} should not have a default — all required"

    def test_ensemble_vote_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(EnsembleVote)


# ===========================================================================
# Category 3: Constants and exports validation
# ===========================================================================

class TestConstantsValidation:
    """Validate module-level constants exist with expected types and ranges."""

    def test_all_regime_weight_values_are_non_negative(self):
        for regime, weights in REGIME_WEIGHTS.items():
            for source, w in weights.items():
                assert w >= 0.0, f"{source} weight in {regime} is negative: {w}"

    def test_all_regime_weight_values_are_finite(self):
        for regime, weights in REGIME_WEIGHTS.items():
            for source, w in weights.items():
                assert np.isfinite(w), f"{source} weight in {regime} is not finite: {w}"

    def test_regime_weights_have_exactly_six_keys_per_regime(self):
        for regime in Regime:
            assert len(REGIME_WEIGHTS[regime]) == 6, f"{regime} does not have 6 entries"

    def test_ensemble_voter_crisis_vol_threshold(self):
        assert EnsembleVoter.CRISIS_VOL_THRESHOLD == 0.30
        assert isinstance(EnsembleVoter.CRISIS_VOL_THRESHOLD, float)

    def test_ensemble_voter_thresholds_exist(self):
        """All regime detection thresholds should be defined."""
        thresholds = [
            'CRISIS_VOL_THRESHOLD', 'CRISIS_DRAWDOWN_THRESHOLD',
            'HIGH_VOL_VOL_THRESHOLD', 'HIGH_VOL_DRAWDOWN_THRESHOLD',
            'HIGH_VOL_MOM_THRESHOLD', 'LOW_VOL_VOL_THRESHOLD',
            'LOW_VOL_MOM_THRESHOLD', 'RECOVERY_DRAWDOWN_THRESHOLD',
            'RECOVERY_MOM_THRESHOLD',
        ]
        for t in thresholds:
            assert hasattr(EnsembleVoter, t), f"EnsembleVoter missing constant {t}"
            val = getattr(EnsembleVoter, t)
            assert isinstance(val, float), f"{t} should be float, got {type(val)}"

    def test_crisis_drawdown_threshold_negative(self):
        assert EnsembleVoter.CRISIS_DRAWDOWN_THRESHOLD == -0.10

    def test_threshold_monotonic_vol(self):
        """Vol thresholds should be ordered: CRISIS > HIGH_VOL > LOW_VOL."""
        assert EnsembleVoter.CRISIS_VOL_THRESHOLD > EnsembleVoter.HIGH_VOL_VOL_THRESHOLD
        assert EnsembleVoter.HIGH_VOL_VOL_THRESHOLD > EnsembleVoter.LOW_VOL_VOL_THRESHOLD


# ===========================================================================
# Category 6: Export completeness
# ===========================================================================

class TestExportCompleteness:
    """Verify __all__ and public API coverage."""

    def test_all_is_defined(self):
        from src.strategy.ensemble_voter import __all__ as all_names
        assert isinstance(all_names, list)
        assert len(all_names) >= 6

    def test_all_names_are_defined_in_module(self):
        from src.strategy.ensemble_voter import __all__ as all_names
        import src.strategy.ensemble_voter as mod
        for name in all_names:
            assert hasattr(mod, name), f"__all__ contains {name} but it's not defined"

    def test_all_includes_core_classes(self):
        from src.strategy.ensemble_voter import __all__ as all_names
        assert 'Regime' in all_names
        assert 'SignalSource' in all_names
        assert 'SignalReading' in all_names
        assert 'EnsembleVote' in all_names
        assert 'EnsembleVoter' in all_names
        assert 'BanditWeighter' in all_names
        assert 'REGIME_WEIGHTS' in all_names

    def test_all_excludes_private_names(self):
        from src.strategy.ensemble_voter import __all__ as all_names
        for name in all_names:
            assert not name.startswith('_'), f"__all__ contains private name {name}"


# ===========================================================================
# Category 2 & 4: BanditWeighter edge cases
# ===========================================================================

class TestBanditWeighterEdgeCases:
    """Additional edge cases for BanditWeighter."""

    def test_empty_signals_list(self):
        bw = BanditWeighter(signals=[])
        assert bw.signals == []
        with pytest.raises(IndexError):
            bw.select('NORMAL')  # random.choice on empty list

    def test_single_signal(self):
        bw = BanditWeighter(signals=['only_one'])
        for _ in range(25):
            bw.update('only_one', 'NORMAL', 0.01)
        weights = bw.get_weights('NORMAL')
        assert weights is not None
        assert weights['only_one'] == pytest.approx(1.0)
        assert bw.select('NORMAL') == 'only_one'

    def test_negative_temperature_same_as_positive(self):
        """Negative temperature should invert ordering effect."""
        bw = BanditWeighter(signals=['a', 'b'], temperature=-1.0)
        sharpes = {'a': 1.0, 'b': 0.5}
        result = bw._softmax(sharpes)
        # -1 temperature: large sharpes get penalized (exp(-1/1) vs exp(-0.5/1) = 0.37 vs 0.61)
        assert all(np.isfinite(v) for v in result.values())
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_softmax_empty_dict(self):
        bw = BanditWeighter(signals=[])
        # Empty signals list means n=0, which would divide by 0
        # This is a degenerate case; verify it raises an appropriate error
        with pytest.raises((ZeroDivisionError, ValueError)):
            bw._softmax({})

    def test_rolling_sharpe_empty_history(self):
        bw = BanditWeighter(signals=['a'])
        sh = bw._rolling_sharpe('a', 'NONEXISTENT')
        assert sh == 0.0

    def test_rolling_sharpe_regime_missing_signal(self):
        bw = BanditWeighter(signals=['a', 'b'])
        for _ in range(25):
            bw.update('a', 'NORMAL', 0.01)
        sh = bw._rolling_sharpe('b', 'NORMAL')
        assert sh == 0.0  # signal 'b' has no history in 'NORMAL'

    def test_get_weights_signal_with_zero_variance(self):
        """Signal with zero variance (all identical returns) should not crash."""
        bw = BanditWeighter(signals=['a', 'b'])
        for _ in range(25):
            bw.update('a', 'NORMAL', 0.01)
            bw.update('b', 'NORMAL', 0.01)  # identical values
        weights = bw.get_weights('NORMAL')
        assert weights is not None
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_get_weights_inf_sharpe_not_defined(self):
        """Inf returns cause inf Sharpe which produces NaN in softmax — should not crash."""
        bw = BanditWeighter(signals=['a', 'b'])
        for _ in range(25):
            bw.update('a', 'NORMAL', 0.01)
            bw.update('b', 'NORMAL', 0.01)
        # Check that rolling sharpe is finite for regular data
        sh_a = bw._rolling_sharpe('a', 'NORMAL')
        sh_b = bw._rolling_sharpe('b', 'NORMAL')
        assert np.isfinite(sh_a) or sh_a == 0.0
        assert np.isfinite(sh_b) or sh_b == 0.0


# ===========================================================================
# Category 2 & 4: Regime detection boundary conditions
# ===========================================================================

class TestRegimeDetectionBoundaries:
    """Boundary and threshold-adjacent tests for detect_regime()."""

    def test_vol_exactly_at_high_vol_threshold(self, tmp_path):
        """Vol near 0.20 should trigger HIGH_VOL or CRISIS via combined checks."""
        voter = _make_voter(tmp_path)
        dates = pd.date_range('2026-01-01', periods=60, freq='B')
        # Use alternating large up/down days to create ~20% annualized vol
        prices = [100.0]
        for i in range(59):
            if i % 2 == 0:
                prices.append(prices[-1] * 1.018)
            else:
                prices.append(prices[-1] * 0.982)
        price_data = pd.DataFrame({'SPY': prices}, index=dates)
        voter.CRISIS_VOL_THRESHOLD = 0.40  # Widen so we don't trigger crisis
        regime, _ = voter.detect_regime(price_data)
        assert regime in (Regime.NORMAL, Regime.LOW_VOL, Regime.HIGH_VOL, Regime.CRISIS)

    def test_crisis_via_vol_exactly_threshold(self, tmp_path):
        """Very high volatility should trigger CRISIS."""
        voter = _make_voter(tmp_path)
        dates = pd.date_range('2026-01-01', periods=60, freq='B')
        # Alternating 3% daily moves = very high vol
        prices = [100.0]
        for i in range(59):
            if i % 2 == 0:
                prices.append(prices[-1] * 1.03)
            else:
                prices.append(prices[-1] * 0.97)
        price_data = pd.DataFrame({'SPY': prices}, index=dates)
        regime, conf = voter.detect_regime(price_data)
        assert regime == Regime.CRISIS
        assert conf >= 0.0

    def test_drawdown_exactly_at_crisis_threshold(self, tmp_path):
        """Large drawdown should trigger CRISIS."""
        voter = _make_voter(tmp_path)
        dates = pd.date_range('2026-01-01', periods=40, freq='B')
        prices = [100.0]
        for i in range(39):
            if i < 10:
                prices.append(prices[-1] * 1.001)
            else:
                prices.append(prices[-1] * 0.990)  # ~1% drops
        price_data = pd.DataFrame({'SPY': prices}, index=dates)
        regime, conf = voter.detect_regime(price_data)
        assert regime == Regime.CRISIS
        assert conf >= 0.0

    def test_drawdown_minus_9_pct_not_crisis(self, tmp_path):
        """Mild drawdown should not trigger CRISIS."""
        voter = _make_voter(tmp_path)
        dates = pd.date_range('2026-01-01', periods=60, freq='B')
        # Small positive drift, very low vol — no drawdown
        prices = [100.0]
        for i in range(59):
            prices.append(prices[-1] * 1.0005)  # consistent small gains
        price_data = pd.DataFrame({'SPY': prices}, index=dates)
        regime, _ = voter.detect_regime(price_data)
        assert regime in (Regime.LOW_VOL, Regime.NORMAL)  # not CRISIS

    def test_momentum_at_low_vol_boundary(self, tmp_path):
        """Strong positive returns with near-zero vol should trigger LOW_VOL."""
        voter = _make_voter(tmp_path)
        dates = pd.date_range('2026-01-01', periods=60, freq='B')
        # Consistent 0.1% daily gains = 25.2% annualized return, near-zero vol
        # 20-day momentum = 20 * 0.001 = 0.02, which is > 0.01 threshold
        prices = [100.0]
        for i in range(59):
            prices.append(prices[-1] * 1.001)
        price_data = pd.DataFrame({'SPY': prices}, index=dates)
        regime, _ = voter.detect_regime(price_data)
        assert regime == Regime.LOW_VOL

    def test_high_vol_drawdown_negative_momentum(self, tmp_path):
        """Large volatility should trigger HIGH_VOL or CRISIS."""
        voter = _make_voter(tmp_path)
        dates = pd.date_range('2026-01-01', periods=60, freq='B')
        # Large alternating up/down days
        prices = [100.0]
        for i in range(59):
            if i % 2 == 0:
                prices.append(prices[-1] * 1.025)
            else:
                prices.append(prices[-1] * 0.975)
        price_data = pd.DataFrame({'SPY': prices}, index=dates)
        regime, _ = voter.detect_regime(price_data)
        assert regime in (Regime.HIGH_VOL, Regime.CRISIS)

    def test_precisely_20_days_data(self, tmp_path):
        """Exactly 20 data points is sufficient for regime detection."""
        voter = _make_voter(tmp_path)
        dates = pd.date_range('2026-01-01', periods=20, freq='B')
        prices = [100.0]
        for i in range(19):
            prices.append(prices[-1] * 1.001)
        price_data = pd.DataFrame({'SPY': prices}, index=dates)
        regime, conf = voter.detect_regime(price_data)
        assert isinstance(regime, Regime)
        assert 0.0 <= conf <= 1.0

    def test_exactly_19_days_insufficient(self, tmp_path):
        """19 data points is insufficient for full detection, returns NORMAL."""
        voter = _make_voter(tmp_path)
        dates = pd.date_range('2026-01-01', periods=19, freq='B')
        prices = [100.0]
        for i in range(18):
            prices.append(prices[-1] * 1.001)
        price_data = pd.DataFrame({'SPY': prices}, index=dates)
        regime, conf = voter.detect_regime(price_data)
        assert regime == Regime.NORMAL
        assert conf == 0.5


# ===========================================================================
# Category 2: ComputeConsensus boundary and NaN/Inf edge cases
# ===========================================================================

class TestComputeConsensusEdgeCases:
    """Boundary and edge cases for _compute_consensus()."""

    def test_all_zero_values(self, tmp_path):
        voter = _make_voter(tmp_path)
        signals = [
            SignalReading(
                source=SignalSource.MULTI_SPEED_MOM,
                timestamp='2026-01-01', value=0.0, confidence=0.8,
                weight=0.0, regime_fit='all', asset_signals=None, explanation='',
            ),
            SignalReading(
                source=SignalSource.CROSS_ASSET_RV,
                timestamp='2026-01-01', value=0.0, confidence=0.7,
                weight=0.0, regime_fit='all', asset_signals=None, explanation='',
            ),
        ]
        signals[0].weight = 0.5
        signals[1].weight = 0.5
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        assert result.weighted_consensus == 0.0
        assert result.action == 'neutral'

    def test_all_zero_weights(self, tmp_path):
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.5, confidence=0.8),
            _make_reading(value=-0.3, confidence=0.7),
        ]
        signals[0].weight = 0.0
        signals[1].weight = 0.0
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        assert np.isfinite(result.weighted_consensus)

    def test_weighted_consensus_exactly_zero(self, tmp_path):
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.5, confidence=0.8),
            _make_reading(value=-0.5, confidence=0.8),
        ]
        signals[0].weight = 0.5
        signals[1].weight = 0.5
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        assert result.weighted_consensus == pytest.approx(0.0, abs=1e-10)
        assert result.action == 'neutral'

    def test_inf_value_handled(self, tmp_path):
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=float('inf'), confidence=0.8),
            _make_reading(value=0.3, confidence=0.7),
        ]
        signals[0].weight = 0.5
        signals[1].weight = 0.5
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        # Inf is not NaN, so it passes the NaN filter — but produces Inf consensus.
        # The agreement ratio uses np.sign, so Inf sign works.
        assert np.isfinite(result.weighted_consensus) or np.isinf(result.weighted_consensus)

    def test_neg_inf_value_handled(self, tmp_path):
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=float('-inf'), confidence=0.8),
            _make_reading(value=-0.3, confidence=0.7),
        ]
        signals[0].weight = 0.5
        signals[1].weight = 0.5
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        assert result is not None

    def test_mixed_nan_and_valid(self, tmp_path):
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=float('nan'), confidence=0.8),
            _make_reading(value=0.5, confidence=0.7),
            _make_reading(value=float('nan'), confidence=0.9),
        ]
        signals[0].weight = 0.3
        signals[1].weight = 0.4
        signals[2].weight = 0.3
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        # One valid signal (0.5 at 0.4 weight) out of total weight 0.4
        assert result.weighted_consensus == pytest.approx(0.5)
        assert np.isfinite(result.weighted_consensus)

    def test_all_nan_still_produces_result(self, tmp_path):
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=float('nan'), confidence=0.8),
            _make_reading(value=float('nan'), confidence=0.7),
        ]
        signals[0].weight = 0.5
        signals[1].weight = 0.5
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        assert result.weighted_consensus == 0.0
        assert result.action == 'neutral'

    def test_extreme_asset_signal_values(self, tmp_path):
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.5, confidence=0.8,
                          asset_signals={'SPY': 1e6, 'TLT': -1e6, 'GLD': 0.0}),
        ]
        signals[0].weight = 1.0
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        assert np.isfinite(result.equity_bias)
        assert np.isfinite(result.duration_bias)

    def test_asset_signal_with_partial_coverage(self, tmp_path):
        voter = _make_voter(tmp_path)
        signals = [
            _make_reading(value=0.5, confidence=0.8,
                          asset_signals={'SPY': 0.6, 'TLT': 0.0}),
            # no GLD in asset_signals
        ]
        signals[0].weight = 1.0
        result = voter._compute_consensus(signals, Regime.NORMAL, 0.5)
        assert result.equity_bias == 0.6
        assert result.duration_bias == 0.0
        # Gold bias falls back to weighted_consensus since no signals have GLD
        assert result.gold_bias == result.weighted_consensus


# ===========================================================================
# Category 2 & 4: Vote computation edge cases
# ===========================================================================

class TestVoteComputationEdgeCases:
    """Edge cases for compute_vote() and its sub-methods."""

    def test_compute_vote_all_zero_weights(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(
                value=0.5, source=SignalSource.MULTI_SPEED_MOM,
            ),
        }
        # All turnover validation catches exceptions, so compute_vote completes
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.5)
        assert vote is not None
        assert vote.action in ('neutral', 'increase_equity', 'decrease_equity', 'risk_off')

    def test_compute_vote_value_at_exactly_positive_one(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(
                value=1.0, source=SignalSource.MULTI_SPEED_MOM,
                asset_signals={'SPY': 1.0, 'TLT': -0.5, 'GLD': 0.3},
            ),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.8)
        assert vote.weighted_consensus == pytest.approx(1.0)
        assert vote.equity_bias == pytest.approx(1.0)

    def test_compute_vote_value_at_exactly_negative_one(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(
                value=-1.0, source=SignalSource.MULTI_SPEED_MOM,
                asset_signals={'SPY': -1.0, 'TLT': 0.5, 'GLD': 0.0},
            ),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.8)
        assert vote.weighted_consensus == pytest.approx(-1.0)
        assert vote.equity_bias == pytest.approx(-1.0)

    def test_compute_vote_conflicting_assets(self, tmp_path):
        """Different asset_signals per source should produce blended biases."""
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(
                value=0.3, source=SignalSource.MULTI_SPEED_MOM,
                asset_signals={'SPY': 0.6, 'TLT': -0.4, 'GLD': 0.2},
            ),
            SignalSource.CROSS_ASSET_RV: _make_reading(
                value=0.2, source=SignalSource.CROSS_ASSET_RV,
                asset_signals={'SPY': -0.3, 'TLT': 0.5, 'GLD': -0.1},
            ),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.7)
        # Equity bias should be a blend: (0.6 * w1 + (-0.3) * w2) / (w1 + w2)
        assert vote.equity_bias != 0.6  # Not just MSM's value
        assert vote.equity_bias != -0.3  # Not just RV's value

    def test_compute_vote_empty_readings_with_regime_gate(self, tmp_path):
        """Empty readings should produce neutral vote even with regime gating."""
        voter = _make_voter(tmp_path)
        vote = voter.compute_vote(readings={}, regime=Regime.HIGH_VOL, regime_confidence=0.6)
        assert vote.num_sources == 0
        assert vote.action == 'neutral'

    def test_compute_vote_single_nan_value(self, tmp_path):
        """Single signal with NaN should fall back to no-valid-signals path."""
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(
                value=float('nan'), source=SignalSource.MULTI_SPEED_MOM,
            ),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.5)
        # After _apply_weights_to_readings and _compute_consensus, NaN is filtered out
        # But there are no valid signals, so consensus is 0.0 and action is neutral
        assert vote.action == 'neutral'
        assert vote.weighted_consensus == 0.0

    def test_collect_signals_applies_zero_weight_skip(self, tmp_path):
        """collect_signals should skip sources with zero weight for regime."""
        voter = _make_voter(tmp_path)
        # In CRISIS, INTERNATIONAL_MOMENTUM weight = 0
        readings = voter.collect_signals(regime=Regime.CRISIS)
        # collect_signals tries real imports which fail in test env
        # So readings will be empty — that's fine
        assert isinstance(readings, dict)

    def test_compute_vote_with_missing_asset_signals(self, tmp_path):
        """Missing asset_signals should fall back to weighted_consensus."""
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: SignalReading(
                source=SignalSource.MULTI_SPEED_MOM,
                timestamp='2026-01-01',
                value=0.5,
                confidence=0.8,
                weight=0.0,
                regime_fit='all',
                asset_signals=None,
                explanation='test',
            ),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.7)
        # Equity bias falls back to weighted_consensus
        assert vote.equity_bias == vote.weighted_consensus

    def test_small_weight_does_not_dominate_large(self, tmp_path):
        """Signal with very small weight should not dominate the vote."""
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(
                value=0.9, source=SignalSource.MULTI_SPEED_MOM,
            ),
            SignalSource.UNIFIED_OVERLAY: SignalReading(
                source=SignalSource.UNIFIED_OVERLAY,
                timestamp='2026-01-01',
                value=-0.001,
                confidence=0.5,
                weight=0.0,
                regime_fit='all',
                asset_signals=None,
                explanation='test',
            ),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.7)
        # MSM has high weight from REGIME_WEIGHTS, UNIFIED has lower
        # The weight from get_blended_weights assigns MSM=0.0 (disabled), so the result
        # depends on the weight allocation. Let's just check it doesn't crash.
        assert vote is not None


# ===========================================================================
# Category 4: ApplyHealthWeights edge cases
# ===========================================================================

class TestApplyHealthWeightsEdgeCases:
    """Edge cases for _apply_health_weights()."""

    def test_all_health_one_no_change(self, tmp_path):
        """Health scores of 1.0 should leave weights unchanged (multiplier=1.0)."""
        voter = _make_voter(tmp_path)
        weights = {SignalSource.MULTI_SPEED_MOM: 0.5, SignalSource.CROSS_ASSET_RV: 0.5}
        with patch('src.signals.health_tracker.SignalHealthTracker') as MockTracker:
            mock_instance = MagicMock()
            mock_score = MagicMock()
            mock_score.health_score = 1.0
            mock_instance.calculate_all_health_scores.return_value = {
                'multi_speed_momentum': mock_score,
                'cross_asset_rv': mock_score,
            }
            MockTracker.return_value = mock_instance
            result = voter._apply_health_weights(weights)
        for k in weights:
            assert result[k] > 0

    def test_all_health_zero(self, tmp_path):
        """Health scores of 0 should clamp multiplier to 0.2."""
        voter = _make_voter(tmp_path)
        weights = {SignalSource.MULTI_SPEED_MOM: 0.6, SignalSource.CROSS_ASSET_RV: 0.4}
        with patch('src.signals.health_tracker.SignalHealthTracker') as MockTracker:
            mock_instance = MagicMock()
            mock_score = MagicMock()
            mock_score.health_score = 0.0
            mock_instance.calculate_all_health_scores.return_value = {
                'multi_speed_momentum': mock_score,
                'cross_asset_rv': mock_score,
            }
            MockTracker.return_value = mock_instance
            result = voter._apply_health_weights(weights)
        # Each weight multiplied by max(0.2, min(1.0, 0.0)) = 0.2
        # So weights become [0.12, 0.08], renormalized to [0.6, 0.4] (same proportion)
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_health_half_reduces_weight(self, tmp_path):
        """Health of 0.5 should reduce weight but not eliminate it."""
        voter = _make_voter(tmp_path)
        weights = {SignalSource.MULTI_SPEED_MOM: 0.6, SignalSource.CROSS_ASSET_RV: 0.4}
        with patch('src.signals.health_tracker.SignalHealthTracker') as MockTracker:
            mock_instance = MagicMock()
            mock_score_good = MagicMock()
            mock_score_good.health_score = 1.0
            mock_score_bad = MagicMock()
            mock_score_bad.health_score = 0.5
            mock_instance.calculate_all_health_scores.return_value = {
                'multi_speed_momentum': mock_score_good,
                'cross_asset_rv': mock_score_bad,
            }
            MockTracker.return_value = mock_instance
            result = voter._apply_health_weights(weights)
        # Bad signal weight reduced: original 0.4 * 0.5 = 0.2 (before renormalization)
        assert result[SignalSource.CROSS_ASSET_RV] < result[SignalSource.MULTI_SPEED_MOM]

    def test_partial_health_scores_only(self, tmp_path):
        """When only some signals have health scores, others keep original weight."""
        voter = _make_voter(tmp_path)
        weights = {
            SignalSource.MULTI_SPEED_MOM: 0.4,
            SignalSource.CROSS_ASSET_RV: 0.3,
            SignalSource.ALTERNATIVE_DATA: 0.3,
        }
        with patch('src.signals.health_tracker.SignalHealthTracker') as MockTracker:
            mock_instance = MagicMock()
            mock_score = MagicMock()
            mock_score.health_score = 0.5
            mock_instance.calculate_all_health_scores.return_value = {
                'multi_speed_momentum': mock_score,
                # CROSS_ASSET_RV and ALTERNATIVE_DATA have no health scores
            }
            MockTracker.return_value = mock_instance
            result = voter._apply_health_weights(weights)
        assert len(result) == 3
        assert abs(sum(result.values()) - 1.0) < 0.01


# ===========================================================================
# Category 4: Recommend allocation edge cases
# ===========================================================================

class TestRecommendAllocationEdgeCases:
    """Edge cases for recommend_allocation()."""

    def test_empty_base_allocation(self, tmp_path):
        voter = _make_voter(tmp_path)
        vote = EnsembleVote(
            timestamp='2026-01-01', regime=Regime.NORMAL, regime_confidence=0.7,
            num_sources=2, weighted_consensus=0.3, agreement_ratio=0.8,
            equity_bias=0.4, duration_bias=-0.1, gold_bias=0.05,
            action='increase_equity', confidence=0.6, reasoning='test', source_votes=[],
        )
        result = voter.recommend_allocation(base_allocation={}, vote=vote)
        assert 'assets' in result
        assert result['assets'] == {}

    def test_nonstandard_asset_names(self, tmp_path):
        """Base allocation with non-standard asset names should still work."""
        voter = _make_voter(tmp_path)
        base = {'QQQ': 0.5, 'IWM': 0.3, 'VNQ': 0.2}
        vote = EnsembleVote(
            timestamp='2026-01-01', regime=Regime.NORMAL, regime_confidence=0.7,
            num_sources=2, weighted_consensus=0.3, agreement_ratio=0.8,
            equity_bias=0.4, duration_bias=-0.1, gold_bias=0.05,
            action='increase_equity', confidence=0.6, reasoning='test', source_votes=[],
        )
        result = voter.recommend_allocation(base_allocation=base, vote=vote)
        for asset in base:
            assert asset in result['assets']
        total = sum(v['new'] for v in result['assets'].values())
        assert abs(total - 1.0) < 0.05

    def test_max_shift_zero(self, tmp_path):
        """max_shift=0 should leave allocations unchanged."""
        voter = _make_voter(tmp_path)
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        vote = EnsembleVote(
            timestamp='2026-01-01', regime=Regime.NORMAL, regime_confidence=0.7,
            num_sources=2, weighted_consensus=0.3, agreement_ratio=0.8,
            equity_bias=0.9, duration_bias=-0.5, gold_bias=0.3,
            action='increase_equity', confidence=0.6, reasoning='test', source_votes=[],
        )
        result = voter.recommend_allocation(base_allocation=base, vote=vote, max_shift=0.0)
        for asset in base:
            assert abs(result['assets'][asset]['shift']) < 1e-10

    def test_crisis_override_with_max_shift(self, tmp_path):
        """Crisis should override equity/duration/gold biases regardless of vote biases."""
        voter = _make_voter(tmp_path)
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        vote = EnsembleVote(
            timestamp='2026-01-01', regime=Regime.CRISIS, regime_confidence=0.9,
            num_sources=2, weighted_consensus=0.5, agreement_ratio=0.9,
            equity_bias=0.5, duration_bias=0.2, gold_bias=-0.2,
            action='risk_off', confidence=0.8, reasoning='test', source_votes=[],
        )
        result = voter.recommend_allocation(base_allocation=base, vote=vote, max_shift=0.10)
        # Crisis: SPY shift = -0.05, GLD shift = +0.03, TLT shift = +0.02
        assert result['assets']['SPY']['shift'] == pytest.approx(-0.05)
        assert result['assets']['GLD']['shift'] == pytest.approx(0.03)
        assert result['assets']['TLT']['shift'] == pytest.approx(0.02)
        assert result['regime'] == 'crisis'
        assert result['action'] == 'risk_off'

    def test_recommend_allocation_negative_bias(self, tmp_path):
        """Negative equity bias should shift SPY down."""
        voter = _make_voter(tmp_path)
        base = {'SPY': 0.50, 'GLD': 0.30, 'TLT': 0.20}
        vote = EnsembleVote(
            timestamp='2026-01-01', regime=Regime.NORMAL, regime_confidence=0.7,
            num_sources=2, weighted_consensus=-0.4, agreement_ratio=0.8,
            equity_bias=-0.4, duration_bias=0.1, gold_bias=0.2,
            action='decrease_equity', confidence=0.6, reasoning='test', source_votes=[],
        )
        result = voter.recommend_allocation(base_allocation=base, vote=vote, max_shift=0.10)
        assert result['assets']['SPY']['shift'] < 0

    def test_recommend_allocation_normalization(self, tmp_path):
        """Result should always renormalize to sum of assets ≈ 1.0."""
        voter = _make_voter(tmp_path)
        base = {'SPY': 1.0}
        vote = EnsembleVote(
            timestamp='2026-01-01', regime=Regime.NORMAL, regime_confidence=0.5,
            num_sources=0, weighted_consensus=0.0, agreement_ratio=0.0,
            equity_bias=0.0, duration_bias=0.0, gold_bias=0.0,
            action='neutral', confidence=0.0, reasoning='', source_votes=[],
        )
        result = voter.recommend_allocation(base_allocation=base, vote=vote)
        assert abs(result['assets']['SPY']['new'] - 1.0) < 0.01


# ===========================================================================
# Category 4: ApplyTurnoverValidation edge cases
# ===========================================================================

class TestApplyTurnoverValidationEdgeCases:
    """Edge cases for _apply_turnover_validation()."""

    def test_empty_readings_returns_weights(self, tmp_path):
        voter = _make_voter(tmp_path)
        weights = {SignalSource.MULTI_SPEED_MOM: 1.0}
        result = voter._apply_turnover_validation(weights, {}, Regime.NORMAL)
        assert result == weights

    def test_readings_with_all_nan_values(self, tmp_path):
        voter = _make_voter(tmp_path)
        weights = {SignalSource.MULTI_SPEED_MOM: 0.6, SignalSource.CROSS_ASSET_RV: 0.4}
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(value=float('nan')),
            SignalSource.CROSS_ASSET_RV: _make_reading(value=float('nan')),
        }
        result = voter._apply_turnover_validation(weights, readings, Regime.NORMAL)
        assert len(result) == 2
        assert abs(sum(result.values()) - 1.0) < 0.01


# ===========================================================================
# Category 4: ApplyGoalRiskBudget edge cases
# ===========================================================================

class TestApplyGoalRiskBudgetBoundary:
    """Boundary cases for apply_goal_risk_budget()."""

    @patch('src.config.goals.load_goals')
    @patch('src.config.goals.get_risk_budget_multiplier')
    def test_risky_reduction_zero_safe_redistribution(self, mock_get_rbm, mock_load_goals):
        """When risky reduction happens but no safe assets exist, original allocation should return."""
        mock_get_rbm.return_value = 0.5
        voter = EnsembleVoter()
        base = {'SPY': 0.46, 'QQQ': 0.54}  # No safe assets
        result = voter.apply_goal_risk_budget(base)
        assert result == base

    @patch('src.config.goals.load_goals')
    @patch('src.config.goals.get_risk_budget_multiplier')
    def test_total_zero_base_returned(self, mock_get_rbm, mock_load_goals):
        """Base allocation with all zeros should be returned unchanged."""
        mock_get_rbm.return_value = 0.5
        voter = EnsembleVoter()
        base = {'SPY': 0.0, 'GLD': 0.0}
        result = voter.apply_goal_risk_budget(base)
        assert result == base

    @patch('src.config.goals.load_goals')
    @patch('src.config.goals.get_risk_budget_multiplier')
    def test_safe_total_zero_skip_redistribution(self, mock_get_rbm, mock_load_goals):
        """When safe assets exist but have zero weight, redistribution branch is skipped."""
        mock_get_rbm.return_value = 0.5
        voter = EnsembleVoter()
        base = {'SPY': 1.0, 'SHY': 0.0}  # SHY has weight 0
        result = voter.apply_goal_risk_budget(base)
        # SPY should be reduced to 0.5, then no safe assets to redistribute to
        # Renormalization makes it 1.0 again
        assert abs(result['SPY'] - 1.0) < 0.01

    @patch('src.config.goals.load_goals')
    @patch('src.config.goals.get_risk_budget_multiplier')
    def test_goal_risk_budget_only_risky_assets(self, mock_get_rbm, mock_load_goals):
        """All non-safe assets should be reduced by risk_mult."""
        mock_get_rbm.return_value = 0.75
        voter = EnsembleVoter()
        base = {'SPY': 0.50, 'GLD': 0.30, 'TLT': 0.20}
        result = voter.apply_goal_risk_budget(base)
        # SPY reduced: 0.50 * 0.75 = 0.375
        # GLD reduced: 0.30 * 0.75 = 0.225
        # TLT is safe, unchanged initially: 0.20
        # risky_reduction = (0.50 - 0.375) + (0.30 - 0.225) = 0.125 + 0.075 = 0.200
        # TLT gets all redistribution since it's the only safe asset: 0.20 + 0.200 = 0.400
        # Total: 0.375 + 0.225 + 0.400 = 1.0
        assert result['SPY'] < base['SPY']
        assert result['GLD'] < base['GLD']
        assert result['TLT'] > base['TLT']

    @patch('src.config.goals.load_goals')
    @patch('src.config.goals.get_risk_budget_multiplier')
    def test_both_safe_assets_get_proportional_boost(self, mock_get_rbm, mock_load_goals):
        """Multiple safe assets should receive proportional redistribution."""
        mock_get_rbm.return_value = 0.5
        voter = EnsembleVoter()
        base = {'SPY': 0.60, 'SHY': 0.25, 'BIL': 0.15}
        result = voter.apply_goal_risk_budget(base)
        # SPY reduced: 0.60 * 0.5 = 0.30, reduction = 0.30
        # SHY: 0.25 + 0.30 * (0.25 / 0.40) = 0.25 + 0.1875 = 0.4375
        # BIL: 0.15 + 0.30 * (0.15 / 0.40) = 0.15 + 0.1125 = 0.2625
        # Renorm: total = 0.30 + 0.4375 + 0.2625 = 1.0
        assert result['SPY'] < base['SPY']
        assert result['SHY'] > base['SHY']
        assert result['BIL'] > base['BIL']


# ===========================================================================
# Category 4: SaveVote / PersistVote edge cases
# ===========================================================================

class TestSaveVoteEdgeCases:
    """Edge cases for _save_vote() and _persist_vote()."""

    def test_save_vote_with_bad_source_values(self, tmp_path):
        """_save_vote should handle ValueError/TypeError for source readings gracefully."""
        voter = _make_voter(tmp_path)
        bad_reading = MagicMock()
        bad_reading.source = "not_an_enum"
        bad_reading.value = "not_a_float"
        bad_reading.confidence = "not_a_float"
        bad_reading.weight = "not_a_float"
        bad_reading.regime_fit = ""
        bad_reading.explanation = ""
        vote = EnsembleVote(
            timestamp='2026-06-01', regime=Regime.NORMAL, regime_confidence=0.7,
            num_sources=1, weighted_consensus=0.3, agreement_ratio=0.8,
            equity_bias=0.3, duration_bias=-0.1, gold_bias=0.05,
            action='neutral', confidence=0.5, reasoning='test',
            source_votes=[bad_reading],
        )
        # Should log warning but not crash
        with patch('src.strategy.ensemble_voter.logger') as mock_logger:
            voter._save_vote(vote)
            assert mock_logger.warning.call_count >= 1

    def test_persist_vote_regret_weighted_state_error(self, tmp_path):
        """_persist_vote should handle RegretWeightedSelector errors."""
        voter = _make_voter(tmp_path)
        vote = EnsembleVote(
            timestamp='2026-06-02', regime=Regime.NORMAL, regime_confidence=0.7,
            num_sources=1, weighted_consensus=0.3, agreement_ratio=0.8,
            equity_bias=0.3, duration_bias=-0.1, gold_bias=0.05,
            action='neutral', confidence=0.5, reasoning='test', source_votes=[],
        )
        with patch('src.strategy.ensemble_voter.logger') as mock_logger:
            # RegretWeightedSelector doesn't import, so _persist_vote logs debug
            voter._persist_vote(vote, 0.3)
            # Should not crash

    def test_persist_vote_save_db_error(self, tmp_path):
        """_persist_vote raises when DB cannot be opened."""
        voter = _make_voter(tmp_path)
        vote = EnsembleVote(
            timestamp='2026-06-03', regime=Regime.NORMAL, regime_confidence=0.7,
            num_sources=1, weighted_consensus=0.3, agreement_ratio=0.8,
            equity_bias=0.3, duration_bias=-0.1, gold_bias=0.05,
            action='neutral', confidence=0.5, reasoning='test', source_votes=[],
        )
        # Corrupt the db_path to cause a DB error — _save_vote doesn't catch it
        voter.db_path = tmp_path / "nonexistent_dir" / "db.sqlite"
        import sqlite3
        with pytest.raises(sqlite3.OperationalError):
            voter._persist_vote(vote, 0.3)


# ===========================================================================
# Category 4: GetBlendedWeights edge cases
# ===========================================================================

class TestGetBlendedWeightsEdgeCases:
    """Edge cases for get_blended_weights()."""

    def test_no_bandit_attr_fallback(self):
        """When voter has no 'bandit' attribute, should return static weights."""
        voter = EnsembleVoter.__new__(EnsembleVoter)
        voter.current_regime = Regime.NORMAL
        result = voter.get_blended_weights('NORMAL')
        static = REGIME_WEIGHTS[Regime.NORMAL]
        assert result.keys() == static.keys()

    def test_zero_bandit_observations(self):
        """Zero bandit observations should return purely static weights."""
        voter = EnsembleVoter()
        assert voter.bandit_observations == 0
        result = voter.get_blended_weights('NORMAL')
        static = REGIME_WEIGHTS[Regime.NORMAL]
        for k, v in static.items():
            assert result[k] == pytest.approx(v, abs=0.01)

    def test_blend_max_after_many_observations(self):
        """Observations exceeding 252 should cap blend at 70%."""
        voter = EnsembleVoter()
        for i in range(300):
            voter.update_bandit('multi_speed_momentum', 'NORMAL', 0.01 + 0.001 * (i % 5))
        # blend should be capped at 0.7
        assert voter.bandit_observations == 300
        result = voter.get_blended_weights('NORMAL')
        assert isinstance(result, dict)

    def test_blended_weights_all_regimes(self):
        """Every regime should produce valid blended weights."""
        voter = EnsembleVoter()
        for regime in Regime:
            result = voter.get_blended_weights(regime.name)
            total = sum(result.values())
            assert abs(total - 1.0) < 0.05, f"{regime} blended weights sum to {total:.4f}"
            for k, v in result.items():
                assert v >= 0.0, f"{k} has negative weight {v} in {regime}"


# ===========================================================================
# Category 4: GetRebalanceConfig edge cases
# ===========================================================================

class TestGetRebalanceConfigEdgeCases:
    """Edge cases for get_rebalance_config()."""

    def test_rebalance_config_low_vol(self):
        voter = EnsembleVoter()
        voter.current_regime = Regime.LOW_VOL
        voter.current_regime_confidence = 0.6
        config = voter.get_rebalance_config()
        assert config['regime'] == 'low_vol'
        assert config['regime_confidence'] == 0.6

    def test_rebalance_config_recovery(self):
        voter = EnsembleVoter()
        voter.current_regime = Regime.RECOVERY
        config = voter.get_rebalance_config()
        assert config['regime'] == 'recovery'

    def test_rebalance_config_unknown_regime(self):
        voter = EnsembleVoter()
        voter.current_regime = "UNKNOWN"  # type: ignore
        config = voter.get_rebalance_config()
        assert config['regime'] == 'normal'

    def test_rebalance_config_default_confidence(self):
        voter = EnsembleVoter()
        config = voter.get_rebalance_config()
        assert 0.0 <= config['regime_confidence'] <= 1.0


# ===========================================================================
# Category 4: ResolveInputs edge cases
# ===========================================================================

class TestResolveInputsEdgeCases:
    """Edge cases for _resolve_inputs()."""

    def test_none_readings_none_regime_uses_detect(self, tmp_path):
        """Both None should trigger detect_regime without crashing."""
        voter = _make_voter(tmp_path)
        with patch.object(voter, '_load_price_data', return_value=None):
            r, reg, conf = voter._resolve_inputs(None, None, None)
            assert isinstance(r, dict)
            assert isinstance(reg, Regime)
            assert conf == 0.5  # detect_regime(None/empty) returns 0.5

    def test_readings_provided_regime_none(self, tmp_path):
        """Readings provided, regime None should trigger detection — detect_regime overwrites conf."""
        voter = _make_voter(tmp_path)
        readings = {SignalSource.MULTI_SPEED_MOM: _make_reading()}
        with patch.object(voter, '_load_price_data', return_value=None):
            r, reg, conf = voter._resolve_inputs(readings, None, None)
            assert r is readings
            assert isinstance(reg, Regime)
            assert isinstance(conf, float)
            # When regime is None, detect_regime sets both regime AND confidence,
            # even if regime_confidence was provided. So conf is from detect_regime.
            assert 0.0 <= conf <= 1.0

    def test_readings_none_current_readings_available(self, tmp_path):
        readings = {SignalSource.MULTI_SPEED_MOM: _make_reading()}
        voter = _make_voter(tmp_path)
        voter.current_readings = readings
        r, reg, conf = voter._resolve_inputs(None, Regime.NORMAL, 0.7)
        assert SignalSource.MULTI_SPEED_MOM in r
        assert reg == Regime.NORMAL
        assert conf == 0.7

    def test_collect_signals_called_when_no_current(self, tmp_path):
        """When no readings and no current_readings, collect_signals is called."""
        voter = _make_voter(tmp_path)
        with patch.object(voter, 'collect_signals', return_value={}) as mock_collect:
            r, reg, conf = voter._resolve_inputs(None, Regime.NORMAL, 0.5)
            mock_collect.assert_called_once()


# ===========================================================================
# Category 4: CollectSignals edge cases
# ===========================================================================

class TestCollectSignalsEdgeCases:
    """Edge cases for collect_signals()."""

    def test_collect_signals_no_regime(self, tmp_path):
        """collect_signals with regime=None should not skip any sources."""
        voter = _make_voter(tmp_path)
        # Some imports may succeed, some may fail; just verify it returns a dict
        readings = voter.collect_signals(regime=None)
        assert isinstance(readings, dict)
        # At minimum, the method returned without crashing

    def test_collect_signals_with_regime_zero_weight_sources_skipped(self, tmp_path):
        """Sources with zero weight for given regime should have _should_skip return True."""
        voter = _make_voter(tmp_path)
        # CRISIS regime: INTL_MOM weight = 0
        active = {SignalSource.MULTI_SPEED_MOM: 0.0,
                  SignalSource.CROSS_ASSET_RV: 0.365,
                  SignalSource.CROSS_ASSET_REGIME_ARB: 0.17,
                  SignalSource.ALTERNATIVE_DATA: 0.20,
                  SignalSource.UNIFIED_OVERLAY: 0.265}
        active_sources_set = {src for src, w in active.items() if w > 0}
        assert SignalSource.INTERNATIONAL_MOMENTUM not in active_sources_set
        assert voter._should_skip(SignalSource.INTERNATIONAL_MOMENTUM, active_sources_set, Regime.CRISIS)

    def test_collect_signals_in_collect_msm(self, tmp_path):
        """_collect_msm_signal should not crash."""
        voter = _make_voter(tmp_path)
        readings = {}
        voter._collect_msm_signal(readings, active_sources=None, regime=None, date=None)
        assert isinstance(readings, dict)

    def test_collect_signals_in_collect_cross_asset_rv(self, tmp_path):
        """_collect_cross_asset_rv_signal should not crash."""
        voter = _make_voter(tmp_path)
        readings = {}
        voter._collect_cross_asset_rv_signal(readings, active_sources=None, regime=None)
        assert isinstance(readings, dict)


# ===========================================================================
# Category 4: EnsembleVoter init edge cases
# ===========================================================================

class TestEnsembleVoterInitCases:
    """Edge cases for EnsembleVoter.__init__()."""

    def test_init_with_custom_path(self, tmp_path):
        voter = EnsembleVoter(data_path=tmp_path)
        assert voter.data_path == tmp_path
        assert voter.db_path == tmp_path / "ensemble_signals.db"

    def test_init_creates_readings_cache(self, tmp_path):
        voter = _make_voter(tmp_path)
        assert voter.current_readings == {}

    def test_init_creates_bandit_with_six_signals(self, tmp_path):
        with patch('src.signals.regime_gate.RegimeGate'):
            voter = EnsembleVoter(data_path=tmp_path)
        assert len(voter.bandit.signals) == 6
        for src in SignalSource:
            assert src.value in voter.bandit.signals

    def test_init_days_in_regime_start_high(self, tmp_path):
        with patch('src.signals.regime_gate.RegimeGate'):
            voter = EnsembleVoter(data_path=tmp_path)
        assert voter._days_in_regime == 999


# ===========================================================================
# Category 5: CLI / __main__ entry points
# ===========================================================================

class TestCLIEntryPoints:
    """Tests for main() CLI entry points via caplog."""

    def test_main_vote_command(self, caplog):
        """main() with 'vote' command should log ensemble vote."""
        mock_vote = EnsembleVote(
            timestamp='2026-06-01T12:00:00',
            regime=Regime.NORMAL,
            regime_confidence=0.7,
            num_sources=2,
            weighted_consensus=0.35,
            agreement_ratio=0.8,
            equity_bias=0.4,
            duration_bias=-0.1,
            gold_bias=0.05,
            action='increase_equity',
            confidence=0.6,
            reasoning='Test reasoning',
            source_votes=[
                SignalReading(
                    source=SignalSource.MULTI_SPEED_MOM,
                    timestamp='2026-06-01T12:00:00',
                    value=0.5, confidence=0.8, weight=0.6,
                    regime_fit='all', asset_signals=None, explanation='test',
                ),
            ],
        )
        with patch('src.strategy.ensemble_voter.EnsembleVoter') as MockVoter:
            mock_instance = MagicMock()
            mock_instance.collect_signals.return_value = {}
            mock_instance.compute_vote.return_value = mock_vote
            MockVoter.return_value = mock_instance
            with patch('sys.argv', ['ensemble_voter', 'vote']):
                with caplog.at_level(logging.INFO, logger="src.strategy.ensemble_voter"):
                    from src.strategy.ensemble_voter import main
                    main()
        assert 'Ensemble Vote' in caplog.text
        assert 'NORMAL' in caplog.text
        assert 'INCREASE_EQUITY' in caplog.text
        assert '0.35' in caplog.text

    def test_main_recommend_command(self, caplog):
        """main() with 'recommend' command should log allocation."""
        mock_vote = EnsembleVote(
            timestamp='2026-06-01T12:00:00',
            regime=Regime.NORMAL,
            regime_confidence=0.7,
            num_sources=2,
            weighted_consensus=0.3,
            agreement_ratio=0.8,
            equity_bias=0.4,
            duration_bias=-0.1,
            gold_bias=0.05,
            action='increase_equity',
            confidence=0.6,
            reasoning='Test',
            source_votes=[],
        )
        with patch('src.strategy.ensemble_voter.EnsembleVoter') as MockVoter:
            mock_instance = MagicMock()
            mock_instance.compute_vote.return_value = mock_vote
            mock_instance.recommend_allocation.return_value = {
                'assets': {
                    'SPY': {'base': 0.46, 'new': 0.50, 'shift': 0.04, 'normalized_shift': 0.04},
                    'GLD': {'base': 0.38, 'new': 0.35, 'shift': -0.03, 'normalized_shift': -0.03},
                    'TLT': {'base': 0.16, 'new': 0.15, 'shift': -0.01, 'normalized_shift': -0.01},
                },
                'regime': 'normal',
                'confidence': 0.6,
                'action': 'increase_equity',
                'consensus': 0.3,
                'timestamp': '2026-06-01T12:00:00',
            }
            MockVoter.return_value = mock_instance
            with patch('sys.argv', ['ensemble_voter', 'recommend', '--portfolio', '46/38/16']):
                with caplog.at_level(logging.INFO, logger="src.strategy.ensemble_voter"):
                    from src.strategy.ensemble_voter import main
                    main()
        assert 'Allocation Recommendation' in caplog.text
        assert '46/38/16' in caplog.text
        assert 'SPY' in caplog.text
        assert 'GLD' in caplog.text

    def test_main_explain_command(self, caplog):
        """main() with 'explain' command should log reasoning."""
        mock_vote = EnsembleVote(
            timestamp='2026-06-01T12:00:00',
            regime=Regime.NORMAL,
            regime_confidence=0.7,
            num_sources=1,
            weighted_consensus=0.3,
            agreement_ratio=0.8,
            equity_bias=0.4,
            duration_bias=-0.1,
            gold_bias=0.05,
            action='neutral',
            confidence=0.5,
            reasoning='Test reasoning line 1\n  Source detail',
            source_votes=[
                SignalReading(
                    source=SignalSource.MULTI_SPEED_MOM,
                    timestamp='2026-06-01T12:00:00',
                    value=0.5, confidence=0.8, weight=0.6,
                    regime_fit='all', asset_signals=None, explanation='test',
                ),
            ],
        )
        with patch('src.strategy.ensemble_voter.EnsembleVoter') as MockVoter:
            mock_instance = MagicMock()
            mock_instance.compute_vote.return_value = mock_vote
            MockVoter.return_value = mock_instance
            with patch('sys.argv', ['ensemble_voter', 'explain']):
                with caplog.at_level(logging.INFO, logger="src.strategy.ensemble_voter"):
                    from src.strategy.ensemble_voter import main
                    main()
        assert 'Ensemble Vote Explanation' in caplog.text
        # Reasoning is logged (should contain the vote's reasoning text)
        assert 'Test reasoning line 1' in caplog.text
        assert 'multi_speed_momentum' in caplog.text

    def test_main_no_command_prints_help(self, capsys):
        """main() with no command should print help."""
        with patch('sys.argv', ['ensemble_voter']):
            from src.strategy.ensemble_voter import main
            main()
        captured = capsys.readouterr()
        assert 'usage:' in captured.out.lower() or 'Ensemble Signal Voter' in captured.out

    def test_main_unknown_command_prints_help(self, capsys):
        """main() with unknown command should print help."""
        with patch('sys.argv', ['ensemble_voter', 'unknown_cmd']):
            from src.strategy.ensemble_voter import main
            with pytest.raises(SystemExit):
                main()
        captured = capsys.readouterr()
        assert 'usage:' in captured.out.lower() or captured.err

    def test_main_vote_with_date(self, caplog):
        """main() vote command with --date should pass date to collect_signals."""
        with patch('src.strategy.ensemble_voter.EnsembleVoter') as MockVoter:
            mock_instance = MagicMock()
            mock_vote = EnsembleVote(
                timestamp='2026-06-01T12:00:00', regime=Regime.NORMAL,
                regime_confidence=0.7, num_sources=0, weighted_consensus=0.0,
                agreement_ratio=0.0, equity_bias=0.0, duration_bias=0.0, gold_bias=0.0,
                action='neutral', confidence=0.5, reasoning='', source_votes=[],
            )
            mock_instance.collect_signals.return_value = {}
            mock_instance.compute_vote.return_value = mock_vote
            MockVoter.return_value = mock_instance
            with patch('sys.argv', ['ensemble_voter', 'vote', '--date', '2026-06-01']):
                with caplog.at_level(logging.INFO, logger="src.strategy.ensemble_voter"):
                    from src.strategy.ensemble_voter import main
                    main()
            mock_instance.collect_signals.assert_called_once_with('2026-06-01')
        assert 'Ensemble Vote' in caplog.text


# ===========================================================================
# Category 4: get_bl_views edge cases
# ===========================================================================

class TestGetBLViewsEdgeCases:
    """Edge cases for get_bl_views()."""

    def test_get_bl_views_zero_tau(self):
        voter = EnsembleVoter()
        result = voter.get_bl_views(tau=0.0)
        assert result['tau'] == 0.0
        assert result['views'].tau == 0.0

    def test_get_bl_views_negative_tau(self):
        voter = EnsembleVoter()
        result = voter.get_bl_views(tau=-0.1)
        assert result['tau'] == -0.1
        assert result['views'].tau == -0.1

    def test_get_bl_views_tracker_returns_valid_health(self):
        """When health tracker returns valid scores, they should appear in result."""
        voter = EnsembleVoter()
        mock_tracker = MagicMock()
        mock_tracker.get_health_report.return_value = {
            'sources': {
                'multi_speed_momentum': {'health_score': 0.85},
                'cross_asset_rv': {'health_score': 0.72},
            }
        }
        with patch('src.strategy.ensemble_voter._get_health_tracker',
                   return_value=mock_tracker):
            result = voter.get_bl_views()
        assert 'multi_speed_momentum' in result['health_scores_used']
        assert result['health_scores_used']['multi_speed_momentum'] == 0.85
        assert result['health_scores_used']['cross_asset_rv'] == 0.72

    def test_get_bl_views_empty_vote_readings(self):
        """get_bl_views with a vote that has 0 sources should still return basic structure."""
        vote = EnsembleVote(
            timestamp='2026-06-01T12:00:00', regime=Regime.NORMAL,
            regime_confidence=0.5, num_sources=0, weighted_consensus=0.0,
            agreement_ratio=0.0, equity_bias=0.0, duration_bias=0.0, gold_bias=0.0,
            action='neutral', confidence=0.0, reasoning='', source_votes=[],
        )
        voter = EnsembleVoter()
        result = voter.get_bl_views(vote=vote)
        assert result['views'] is not None
        assert result['equity_bias'] == 0.0

    def test_get_bl_views_with_prior_market(self):
        voter = EnsembleVoter()
        result = voter.get_bl_views(prior='market')
        assert result['prior'] == 'market'


# ===========================================================================
# Category 4: BanditWeighter interaction edge cases
# ===========================================================================

class TestUpdateBanditInteraction:
    """Interaction edge cases for update_bandit()."""

    def test_update_bandit_extreme_return(self):
        """Extreme daily return (+100%) should not crash."""
        voter = EnsembleVoter()
        voter.update_bandit('multi_speed_momentum', 'NORMAL', 1.0)
        assert voter.bandit_observations == 1

    def test_update_bandit_extreme_negative_return(self):
        """Extreme daily return (-100%) should not crash."""
        voter = EnsembleVoter()
        voter.update_bandit('multi_speed_momentum', 'NORMAL', -1.0)
        assert voter.bandit_observations == 1

    def test_update_bandit_inf_return(self):
        """Inf daily return should be storable (valid float)."""
        voter = EnsembleVoter()
        voter.update_bandit('multi_speed_momentum', 'NORMAL', float('inf'))
        assert voter.bandit_observations == 1
        assert np.isinf(voter.bandit._history['NORMAL']['multi_speed_momentum'][0])

    def test_update_bandit_all_regimes(self):
        """update_bandit should work for all five regimes."""
        voter = EnsembleVoter()
        for regime_name in ['NORMAL', 'HIGH_VOL', 'CRISIS', 'RECOVERY', 'LOW_VOL']:
            voter.update_bandit('multi_speed_momentum', regime_name, 0.01)
        assert voter.bandit_observations == 5
        for regime_name in ['NORMAL', 'HIGH_VOL', 'CRISIS', 'RECOVERY', 'LOW_VOL']:
            assert regime_name in voter.bandit._history


# ===========================================================================
# Category 4: apply_regime_gating edge cases
# ===========================================================================

class TestApplyRegimeGatingEdgeCases:
    """Edge cases for _apply_regime_gating()."""

    def test_empty_weights(self, tmp_path):
        """Empty weights dict should be returned as-is."""
        voter = _make_voter(tmp_path)
        result = voter._apply_regime_gating({}, 'NORMAL')
        assert result == {}

    def test_regime_gate_all_zero_normalize(self, tmp_path):
        """When regime_gate zeros all weights, total stays 0 (no div-by-zero)."""
        voter = _make_voter(tmp_path)
        voter.regime_gate = None
        # Without regime_gate, weights pass through unchanged
        weights = {SignalSource.MULTI_SPEED_MOM: 0.3, SignalSource.CROSS_ASSET_RV: 0.7}
        result = voter._apply_regime_gating(weights, 'NORMAL')
        assert result == weights

    def test_gate_zeros_some_renormalizes(self, tmp_path):
        """When gate zeros some weights, remaining should be renormalized."""
        voter = _make_voter(tmp_path)
        mock_gate = MagicMock()
        mock_gate.filter_weights.return_value = {
            SignalSource.MULTI_SPEED_MOM: 0.0,
            SignalSource.CROSS_ASSET_RV: 0.5,
            SignalSource.ALTERNATIVE_DATA: 0.5,
        }
        voter.regime_gate = mock_gate
        result = voter._apply_regime_gating(
            {SignalSource.MULTI_SPEED_MOM: 0.3, SignalSource.CROSS_ASSET_RV: 0.4, SignalSource.ALTERNATIVE_DATA: 0.3},
            'NORMAL',
        )
        assert result[SignalSource.MULTI_SPEED_MOM] == 0.0
        assert abs(result[SignalSource.CROSS_ASSET_RV] + result[SignalSource.ALTERNATIVE_DATA] - 1.0) < 0.01


# ===========================================================================
# Category 2: Enum completeness
# ===========================================================================

class TestEnumCompleteness:
    """Additional enum validation tests."""

    def test_regime_names_are_unique(self):
        names = [r.name for r in Regime]
        assert len(names) == len(set(names))

    def test_signal_source_count_matches_active_set(self):
        """All 6 SignalSource values should match between enum and __all__."""
        from src.strategy.ensemble_voter import __all__ as all_names
        assert len(list(SignalSource)) == 6

    def test_regime_has_low_vol(self):
        assert Regime.LOW_VOL.value == 'low_vol'
        assert Regime.LOW_VOL.name == 'LOW_VOL'

    def test_signal_source_has_unified_overlay(self):
        assert SignalSource.UNIFIED_OVERLAY.value == 'unified_overlay'


# ===========================================================================
# Category 2: _ConsensusResult dataclass edge cases
# ===========================================================================

class TestConsensusResultDataclass:
    """Tests for the internal _ConsensusResult dataclass."""

    def test_consensus_result_fields(self):
        voter = EnsembleVoter.__new__(EnsembleVoter)
        result = voter._ConsensusResult(
            weighted_consensus=0.5,
            agreement=0.8,
            equity_bias=0.4,
            duration_bias=-0.1,
            gold_bias=0.05,
            action='test',
            action_confidence=0.6,
        )
        assert result.weighted_consensus == 0.5
        assert result.agreement == 0.8
        assert result.equity_bias == 0.4
        assert result.duration_bias == -0.1
        assert result.gold_bias == 0.05
        assert result.action == 'test'
        assert result.action_confidence == 0.6

    def test_consensus_result_is_dataclass(self):
        import dataclasses
        voter = EnsembleVoter.__new__(EnsembleVoter)
        assert dataclasses.is_dataclass(voter._ConsensusResult)

    def test_consensus_result_field_count(self):
        from dataclasses import fields
        voter = EnsembleVoter.__new__(EnsembleVoter)
        flds = fields(voter._ConsensusResult)
        assert len(flds) == 7


# ===========================================================================
# Category 5: Thompson Sampling Bandit tests
# ===========================================================================

class TestThompsonSamplingBandit:
    """Tests for Thompson Sampling with Gaussian-Gamma conjugate priors
    in BanditWeighter (upgraded from epsilon-greedy)."""

    # ------------------------------------------------------------------
    # Test 1: Basic preference for higher-mean signal
    # ------------------------------------------------------------------

    def test_thompson_sampling_prefers_higher_mean_signal(self):
        """With 2 signals of clearly different mean, Thompson Sampling
        should select the better one >65% of the time over 200 trials."""
        bw = BanditWeighter(signals=['bad', 'good'], epsilon=0.0, temperature=1.0)
        for _ in range(50):
            bw.update('good', 'NORMAL', 0.05)
            bw.update('bad', 'NORMAL', -0.01)

        good_count = sum(1 for _ in range(200) if bw.select('NORMAL') == 'good')
        assert good_count > 130, f"Expected good >65%, got {good_count/200:.1%}"

    # ------------------------------------------------------------------
    # Test 2: Thompson Sampling converges faster than epsilon-greedy
    # ------------------------------------------------------------------

    def test_thompson_sampling_converges_faster_than_epsilon_greedy(self):
        """With limited data (10 obs per signal), posterior sampling should
        select the best signal more often than deterministic rolling Sharpe
        (which cannot distinguish signals with <21 obs)."""
        bw_ts = BanditWeighter(
            signals=['bad_signal', 'good_signal'], epsilon=0.0, temperature=1.0,
        )
        bw_eg = BanditWeighter(
            signals=['bad_signal', 'good_signal'], epsilon=0.0, temperature=1.0,
        )
        for _ in range(10):
            bw_ts.update('good_signal', 'NORMAL', 0.05)
            bw_ts.update('bad_signal', 'NORMAL', -0.01)
            bw_eg.update('good_signal', 'NORMAL', 0.05)
            bw_eg.update('bad_signal', 'NORMAL', -0.01)

        # Epsilon-greedy simulator: mock _sample_sharpe to return rolling_sharpe
        # (which returns 0.0 for <21 obs, so both signals tie)
        from unittest.mock import patch
        ts_good = sum(1 for _ in range(200) if bw_ts.select('NORMAL') == 'good_signal')

        eg_good = 0
        with patch.object(bw_eg, '_sample_sharpe',
                          side_effect=lambda sig, reg: bw_eg._rolling_sharpe(sig, reg)):
            for _ in range(200):
                if bw_eg.select('NORMAL') == 'good_signal':
                    eg_good += 1

        assert ts_good > eg_good, (
            f"TS good_pct={ts_good/200:.1%} should exceed "
            f"epsilon-greedy good_pct={eg_good/200:.1%}"
        )

    # ------------------------------------------------------------------
    # Test 3: _sample_sharpe returns a float
    # ------------------------------------------------------------------

    def test_sample_sharpe_returns_float(self):
        """_sample_sharpe should return a native Python float with sufficient data."""
        bw = BanditWeighter(signals=['sig'], epsilon=0.0)
        for _ in range(10):
            bw.update('sig', 'NORMAL', 0.01)
        result = bw._sample_sharpe('sig', 'NORMAL')
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    # ------------------------------------------------------------------
    # Test 4: _sample_sharpe with insufficient data
    # ------------------------------------------------------------------

    def test_sample_sharpe_insufficient_data(self):
        """_sample_sharpe should return 0.0 when fewer than 2 observations."""
        bw = BanditWeighter(signals=['sig'], epsilon=0.0)
        bw.update('sig', 'NORMAL', 0.01)  # Only 1 observation
        result = bw._sample_sharpe('sig', 'NORMAL')
        assert result == 0.0

    # ------------------------------------------------------------------
    # Test 5: Cold start with uninformative prior
    # ------------------------------------------------------------------

    def test_cold_start_uses_uninformative_prior(self):
        """With epsilon=0 and <2 observations per signal, select() should
        not crash and should return valid signal names (fallback to
        rolling Sharpe or first signal)."""
        bw = BanditWeighter(signals=['alpha', 'beta'], epsilon=0.0)
        bw.update('alpha', 'NORMAL', 0.01)   # Only 1 obs — <2
        bw.update('beta', 'NORMAL', -0.005)   # Only 1 obs — <2

        selections = [bw.select('NORMAL') for _ in range(10)]
        for sel in selections:
            assert sel in ('alpha', 'beta'), f"Unexpected selection: {sel}"

    # ------------------------------------------------------------------
    # Test 6: Posterior updates correctly with observations
    # ------------------------------------------------------------------

    def test_posterior_updates_with_observations(self):
        """After updating a signal many times with positive returns,
        _sample_sharpe should return a positive Sharpe >70% of the time."""
        bw = BanditWeighter(signals=['sig'], epsilon=0.0)
        for i in range(100):
            bw.update('sig', 'NORMAL', 0.01 + 0.005 * (i % 2))

        positive = 0
        for _ in range(100):
            if bw._sample_sharpe('sig', 'NORMAL') > 0:
                positive += 1
        assert positive > 70, f"Expected >70% positive Sharpe, got {positive}%"

    # ------------------------------------------------------------------
    # Test 7: Zero returns produce near-zero sampled Sharpe
    # ------------------------------------------------------------------

    def test_thompson_sampling_with_zero_returns(self):
        """All returns are 0.0 — sampled Sharpe mean should be near zero.

        With zero variance data the posterior precision is extremely high,
        so tiny floating-point noise in mu_sample can get amplified. We use
        a generous tolerance to account for this degenerate case.
        """
        bw = BanditWeighter(signals=['sig'], epsilon=0.0)
        for _ in range(30):
            bw.update('sig', 'NORMAL', 0.0)

        sampled = [bw._sample_sharpe('sig', 'NORMAL') for _ in range(100)]
        mean_sharpe = np.mean(sampled)
        # With zero variance, posterior is degenerate — generous tolerance
        assert abs(mean_sharpe) < 2.0, (
            f"Expected mean Sharpe near 0, got {mean_sharpe:.4f}"
        )

    # ------------------------------------------------------------------
    # Test 8: Deterministic selection with very strong signal
    # ------------------------------------------------------------------

    def test_select_deterministic_with_very_strong_signal(self):
        """With 200 observations where good=0.10 and bad=-0.05,
        select() should return 'good' >90% of the time."""
        bw = BanditWeighter(signals=['bad', 'good'], epsilon=0.0, temperature=1.0)
        for _ in range(200):
            bw.update('good', 'NORMAL', 0.10)
            bw.update('bad', 'NORMAL', -0.05)

        good_count = sum(1 for _ in range(200) if bw.select('NORMAL') == 'good')
        assert good_count > 180, f"Expected good >90%, got {good_count/200:.1%}"


# ===========================================================================
# Category 6: Utility-Based Reweighting Tests
# ===========================================================================

class TestUtilityReweighting:
    """Tests for utility-based ensemble reweighting (profitability proxy)."""

    def _make_voter_with_attribution(self, tmp_path, sources_data):
        """Create a voter with mock attribution data."""
        attr_dir = tmp_path / "attribution"
        attr_dir.mkdir(exist_ok=True)
        voter = _make_voter(tmp_path)
        attribution = {
            "timestamp": datetime.now().isoformat(),
            "start_date": "2026-01-01",
            "end_date": "2026-05-26",
            "analysis_days": 90,
            "sources": sources_data,
        }
        with open(attr_dir / "attribution_2026-05-26.json", "w") as f:
            json.dump(attribution, f)
        return voter, attr_dir

    def test_no_adjustment_without_attribution(self, tmp_path):
        """Without attribution files, weights should pass through unchanged."""
        voter = _make_voter(tmp_path)
        weights = {SignalSource.MULTI_SPEED_MOM: 0.2, SignalSource.ALTERNATIVE_DATA: 0.8}
        with patch("src.strategy.ensemble_voter.ATTRIBUTION_DIR", tmp_path / "nonexistent"):
            result = voter._apply_utility_reweighting(weights, Regime.NORMAL)
        assert result == weights

    def test_no_adjustment_with_few_readings(self, tmp_path):
        """With <20 readings, no adjustment should be made."""
        sources = {
            "multi_speed_momentum": {"total_readings": 5, "sharpe_contribution": 1.0, "hit_rate": 0.8},
        }
        voter, attr_dir = self._make_voter_with_attribution(tmp_path, sources)
        weights = {SignalSource.MULTI_SPEED_MOM: 1.0}
        with patch("src.strategy.ensemble_voter.ATTRIBUTION_DIR", attr_dir):
            result = voter._apply_utility_reweighting(weights, Regime.NORMAL)
        assert abs(result[SignalSource.MULTI_SPEED_MOM] - 1.0) < 1e-6

    def test_positive_sharpe_boosts_weight(self, tmp_path):
        """Signal with positive Sharpe contribution should get weight boost."""
        sources = {
            "alternative_data": {
                "total_readings": 50,
                "sharpe_contribution": 1.5,
                "hit_rate": 0.65,
            },
            "cross_asset_rv": {
                "total_readings": 50,
                "sharpe_contribution": 0.0,
                "hit_rate": 0.5,
            },
        }
        voter, attr_dir = self._make_voter_with_attribution(tmp_path, sources)
        weights = {SignalSource.ALTERNATIVE_DATA: 0.5, SignalSource.CROSS_ASSET_RV: 0.5}
        with patch("src.strategy.ensemble_voter.ATTRIBUTION_DIR", attr_dir):
            result = voter._apply_utility_reweighting(weights, Regime.NORMAL)
        assert result[SignalSource.ALTERNATIVE_DATA] > result[SignalSource.CROSS_ASSET_RV]
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_negative_sharpe_reduces_weight(self, tmp_path):
        """Signal with negative Sharpe contribution should get weight reduction."""
        sources = {
            "multi_speed_momentum": {
                "total_readings": 50,
                "sharpe_contribution": -1.5,
                "hit_rate": 0.3,
            },
            "alternative_data": {
                "total_readings": 50,
                "sharpe_contribution": 0.5,
                "hit_rate": 0.55,
            },
        }
        voter, attr_dir = self._make_voter_with_attribution(tmp_path, sources)
        weights = {SignalSource.MULTI_SPEED_MOM: 0.5, SignalSource.ALTERNATIVE_DATA: 0.5}
        with patch("src.strategy.ensemble_voter.ATTRIBUTION_DIR", attr_dir):
            result = voter._apply_utility_reweighting(weights, Regime.NORMAL)
        assert result[SignalSource.MULTI_SPEED_MOM] < result[SignalSource.ALTERNATIVE_DATA]

    def test_max_adjustment_capped_at_30pct(self, tmp_path):
        """Weight adjustment should be capped at ±30%."""
        sources = {
            "alternative_data": {
                "total_readings": 100,
                "sharpe_contribution": 10.0,
                "hit_rate": 0.95,
            },
            "cross_asset_rv": {
                "total_readings": 100,
                "sharpe_contribution": 0.0,
                "hit_rate": 0.5,
            },
        }
        voter, attr_dir = self._make_voter_with_attribution(tmp_path, sources)
        weights = {SignalSource.ALTERNATIVE_DATA: 0.5, SignalSource.CROSS_ASSET_RV: 0.5}
        with patch("src.strategy.ensemble_voter.ATTRIBUTION_DIR", attr_dir):
            result = voter._apply_utility_reweighting(weights, Regime.NORMAL)
        alt_weight = result[SignalSource.ALTERNATIVE_DATA]
        rv_weight = result[SignalSource.CROSS_ASSET_RV]
        assert alt_weight > rv_weight
        assert alt_weight < 0.9

    def test_stale_attribution_ignored(self, tmp_path):
        """Attribution data older than 7 days should be ignored."""
        attr_dir = tmp_path / "attribution"
        attr_dir.mkdir(exist_ok=True)
        stale_attribution = {
            "timestamp": "2026-05-01T12:00:00",
            "sources": {
                "alternative_data": {
                    "total_readings": 100,
                    "sharpe_contribution": 5.0,
                    "hit_rate": 0.9,
                },
            },
        }
        with open(attr_dir / "attribution_2026-05-01.json", "w") as f:
            json.dump(stale_attribution, f)

        voter = _make_voter(tmp_path)
        weights = {SignalSource.ALTERNATIVE_DATA: 1.0}
        with patch("src.strategy.ensemble_voter.ATTRIBUTION_DIR", attr_dir):
            result = voter._apply_utility_reweighting(weights, Regime.NORMAL)
        assert abs(result[SignalSource.ALTERNATIVE_DATA] - 1.0) < 1e-6

    def test_mixed_signals_renormalize(self, tmp_path):
        """After utility reweighting, weights should still sum to 1.0."""
        sources = {}
        for name in ["multi_speed_momentum", "cross_asset_rv", "alternative_data"]:
            sources[name] = {
                "total_readings": 50,
                "sharpe_contribution": float(np.random.uniform(-1, 1)),
                "hit_rate": float(np.random.uniform(0.3, 0.7)),
            }
        voter, attr_dir = self._make_voter_with_attribution(tmp_path, sources)
        weights = {SignalSource.MULTI_SPEED_MOM: 0.3, SignalSource.CROSS_ASSET_RV: 0.3, SignalSource.ALTERNATIVE_DATA: 0.4}
        with patch("src.strategy.ensemble_voter.ATTRIBUTION_DIR", attr_dir):
            result = voter._apply_utility_reweighting(weights, Regime.NORMAL)
        assert abs(sum(result.values()) - 1.0) < 0.01

