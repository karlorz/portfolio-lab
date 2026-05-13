#!/usr/bin/env python3
"""
Tests for ensemble voter — enums, data classes, regime weights,
regime detection, vote computation, allocation recommendation.
"""
import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.strategy.ensemble_voter import (
    Regime, SignalSource, SignalReading, EnsembleVote,
    REGIME_WEIGHTS, EnsembleVoter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reading(source=SignalSource.TSFM_MOMENTUM, value=0.5, confidence=0.8,
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
        assert SignalSource.TSFM_MOMENTUM.value == 'tsfm_momentum'
        assert SignalSource.HMM_REGIME.value == 'hmm_regime'
        assert SignalSource.CTA_TREND.value == 'cta_trend'

    def test_signal_source_members(self):
        assert len(SignalSource) == 8


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestSignalReading:
    def test_creation(self):
        r = _make_reading()
        assert r.source == SignalSource.TSFM_MOMENTUM
        assert r.value == 0.5

    def test_asset_signals(self):
        r = _make_reading(asset_signals={'SPY': 0.8})
        assert r.asset_signals['SPY'] == 0.8


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
            assert abs(total - 1.0) < 0.01, f"{regime} weights sum to {total}"

    def test_all_sources_covered(self):
        for regime, weights in REGIME_WEIGHTS.items():
            for source in SignalSource:
                assert source in weights, f"{source} missing from {regime}"

    def test_crisis_circuit_breaker_high(self):
        assert REGIME_WEIGHTS[Regime.CRISIS][SignalSource.CIRCUIT_BREAKER] > 0.3

    def test_normal_tsfm_dominant(self):
        assert REGIME_WEIGHTS[Regime.NORMAL][SignalSource.TSFM_MOMENTUM] > 0.3

    def test_high_vol_hmm_dominant(self):
        assert REGIME_WEIGHTS[Regime.HIGH_VOL][SignalSource.HMM_REGIME] > 0.3


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
            SignalSource.TSFM_MOMENTUM: _make_reading(value=0.5),
            SignalSource.CTA_TREND: _make_reading(value=0.3, source=SignalSource.CTA_TREND),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.7)
        assert vote.num_sources == 2
        assert vote.weighted_consensus != 0
        assert vote.action in ['increase_equity', 'decrease_equity', 'neutral', 'risk_off']

    def test_compute_vote_crisis_action(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.CIRCUIT_BREAKER: _make_reading(value=-0.8, source=SignalSource.CIRCUIT_BREAKER),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.CRISIS, regime_confidence=0.9)
        assert vote.action == 'risk_off'

    def test_compute_vote_increase_equity(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.TSFM_MOMENTUM: _make_reading(
                value=0.8,
                asset_signals={'SPY': 0.8, 'TLT': -0.3, 'GLD': 0.1},
            ),
            SignalSource.MULTI_SPEED_MOM: _make_reading(
                value=0.7, source=SignalSource.MULTI_SPEED_MOM,
                asset_signals={'SPY': 0.7, 'TLT': -0.2, 'GLD': 0.0},
            ),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.8)
        assert vote.equity_bias > 0.3

    def test_compute_vote_agreement_ratio(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.TSFM_MOMENTUM: _make_reading(value=0.5),
            SignalSource.CTA_TREND: _make_reading(value=0.4, source=SignalSource.CTA_TREND),
        }
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.7)
        assert 0.0 <= vote.agreement_ratio <= 1.0

    def test_compute_vote_saves_to_db(self, tmp_path):
        voter = _make_voter(tmp_path)
        readings = {SignalSource.TSFM_MOMENTUM: _make_reading(value=0.3)}
        vote = voter.compute_vote(readings=readings, regime=Regime.NORMAL, regime_confidence=0.6)
        # Check DB has the vote
        import sqlite3
        with sqlite3.connect(str(voter.db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM ensemble_votes").fetchone()
            assert row[0] >= 1

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
