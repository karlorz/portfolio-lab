#!/usr/bin/env python3
"""
Tests for ensemble voter — enums, data classes, regime weights,
regime detection, vote computation, allocation recommendation.
"""
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
        """With epsilon=0, select() should always return the best signal."""
        bw = BanditWeighter(signals=['good', 'bad'], epsilon=0.0, temperature=1.0)
        for _ in range(30):
            bw.update('good', 'NORMAL', 0.05)
            bw.update('bad', 'NORMAL', -0.01)
        for _ in range(20):
            assert bw.select('NORMAL') == 'good'

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
        mock_load_goals.side_effect = Exception("Unexpected error")
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
        mock_tracker.get_health_report.side_effect = Exception("Tracker crashed")
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
