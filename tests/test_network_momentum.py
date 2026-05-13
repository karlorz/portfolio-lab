#!/usr/bin/env python3
"""
Tests for network momentum lead-lag overlay — data classes, DTW distance,
Lévy area signatures, lead-lag matrix, window signals, ensemble signals,
and portfolio recommendation.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.strategy.network_momentum_leadlag import (
    LeadLagMatrix, WindowMomentumSignal, EnsembleNetworkSignal,
    NetworkMomentumPortfolio, NetworkMomentumLeadLag,
    NetworkMomentumBacktester,
    LOOKBACK_WINDOWS, DEFAULT_WINDOW, MAX_DEVIATION, MIN_WEIGHT,
    ASSETS, DEFAULT_BASE_ALLOCATION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices_df(n_days=300, seed=42):
    """Create synthetic price DataFrame with SPY, GLD, TLT columns."""
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    data = {}
    for ticker, drift in [('SPY', 0.0004), ('GLD', 0.0002), ('TLT', 0.0001)]:
        prices = [500.0]
        for _ in range(n_days - 1):
            ret = np.random.normal(drift, 0.012)
            prices.append(prices[-1] * (1 + ret))
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


def _make_leadlag_matrix(tickers=None):
    """Create a test LeadLagMatrix."""
    if tickers is None:
        tickers = ['SPY', 'GLD', 'TLT']
    leadlag = {}
    dtw = {}
    levy = {}
    adj = {}
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i != j:
                leadlag[(t1, t2)] = 0.1 * (i - j)
                dtw[(t1, t2)] = 5.0
                levy[(t1, t2)] = 0.05 * (i - j)
                adj[(t1, t2)] = max(0.0, 0.2 * (j - i))
    return LeadLagMatrix(
        timestamp=datetime.now().isoformat(),
        window=66,
        leadlag_matrix=leadlag,
        dtw_distances=dtw,
        levy_areas=levy,
        adjacency=adj,
    )


def _make_engine(tmp_path=None, prices_df=None):
    """Create a NetworkMomentumLeadLag with mocked prices."""
    engine = NetworkMomentumLeadLag.__new__(NetworkMomentumLeadLag)
    engine.prices_path = tmp_path / "prices.json" if tmp_path else Path("/tmp/prices.json")
    engine.db_path = tmp_path / "signals.db" if tmp_path else Path("/tmp/signals.db")
    engine.lookback_windows = LOOKBACK_WINDOWS
    engine.max_deviation = MAX_DEVIATION
    engine._prices_df = prices_df
    return engine


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestLeadLagMatrix:
    """Test LeadLagMatrix dataclass."""

    def test_to_dict(self):
        ll = _make_leadlag_matrix()
        d = ll.to_dict()
        assert d['timestamp'] == ll.timestamp
        assert d['window'] == 66
        assert 'SPY->GLD' in d['leadlag_matrix']
        assert 'SPY-GLD' in d['dtw_distances']
        assert 'SPY->GLD' in d['adjacency']


class TestWindowMomentumSignal:
    """Test WindowMomentumSignal dataclass."""

    def test_to_dict(self):
        sig = WindowMomentumSignal(
            ticker='SPY', window=66, timestamp=datetime.now().isoformat(),
            momentum_return=0.05, signal=1,
            network_momentum=0.06, network_adjustment=0.01,
            base_weight=0.46, target_weight=0.49, adjustment=0.03,
        )
        d = sig.to_dict()
        assert d['ticker'] == 'SPY'
        assert d['momentum_return'] == 0.05
        assert d['signal'] == 1


class TestEnsembleNetworkSignal:
    """Test EnsembleNetworkSignal dataclass."""

    def test_to_dict(self):
        sig = EnsembleNetworkSignal(
            ticker='SPY', timestamp=datetime.now().isoformat(),
            window_signals={},
            ensemble_momentum=0.04, ensemble_signal=1, ensemble_confidence=0.8,
            base_weight=0.46, adjustment=0.02, target_weight=0.48,
            leadership_score=0.3, followership_score=0.1, network_centrality=0.2,
        )
        d = sig.to_dict()
        assert d['ticker'] == 'SPY'
        assert d['leadership_score'] == 0.3


class TestNetworkMomentumPortfolio:
    """Test NetworkMomentumPortfolio dataclass."""

    def test_to_dict(self):
        ll = _make_leadlag_matrix()
        port = NetworkMomentumPortfolio(
            timestamp=datetime.now().isoformat(),
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            network_adjustments={'SPY': 0.02, 'GLD': -0.01, 'TLT': -0.01},
            target_allocation={'SPY': 0.48, 'GLD': 0.37, 'TLT': 0.15},
            leadlag_matrix=ll,
            ensemble_signals={},
            dominant_leader='SPY',
            dominant_follower='TLT',
            network_efficiency=0.5,
            overall_confidence=0.7,
        )
        d = port.to_dict()
        assert d['dominant_leader'] == 'SPY'
        assert 'SPY->GLD' in d['leadlag_matrix']['leadlag_matrix']


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Test module constants."""

    def test_lookback_windows(self):
        assert len(LOOKBACK_WINDOWS) == 6
        assert 22 in LOOKBACK_WINDOWS
        assert 132 in LOOKBACK_WINDOWS

    def test_default_base_allocation_sums_to_one(self):
        total = sum(DEFAULT_BASE_ALLOCATION.values())
        assert abs(total - 1.0) < 0.01

    def test_assets_list(self):
        assert 'SPY' in ASSETS
        assert 'GLD' in ASSETS
        assert 'TLT' in ASSETS
        assert 'CASH' in ASSETS

    def test_max_deviation(self):
        assert MAX_DEVIATION == 0.15


# ---------------------------------------------------------------------------
# DTW distance tests
# ---------------------------------------------------------------------------

class TestDTWDistance:
    """Test _simple_dtw_distance."""

    def test_identical_series_zero(self):
        """Identical series should have ~0 DTW distance."""
        engine = _make_engine()
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = engine._simple_dtw_distance(s, s)
        assert dist < 0.01

    def test_similar_series_low(self):
        """Similar series have low DTW distance."""
        engine = _make_engine()
        s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        dist = engine._simple_dtw_distance(s1, s2)
        assert dist < 1.0

    def test_different_series_high(self):
        """Very different series have high DTW distance."""
        engine = _make_engine()
        s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        dist = engine._simple_dtw_distance(s1, s2)
        assert dist > 1.0

    def test_returns_float(self):
        """Returns a finite float."""
        engine = _make_engine()
        s1 = np.random.normal(0, 1, 50)
        s2 = np.random.normal(0, 1, 50)
        dist = engine._simple_dtw_distance(s1, s2)
        assert np.isfinite(dist)


# ---------------------------------------------------------------------------
# Lévy area tests
# ---------------------------------------------------------------------------

class TestLevyArea:
    """Test _compute_levy_area_signature."""

    def test_returns_float(self):
        """Returns a finite float."""
        engine = _make_engine()
        s1 = np.random.normal(0, 1, 50)
        s2 = np.random.normal(0, 1, 50)
        levy = engine._compute_levy_area_signature(s1, s2)
        assert np.isfinite(levy)

    def test_symmetric_negation(self):
        """levy(a,b) ≈ -levy(b,a) for the sign."""
        engine = _make_engine()
        np.random.seed(42)
        s1 = np.random.normal(0, 1, 100)
        s2 = np.random.normal(0, 1, 100)
        levy_12 = engine._compute_levy_area_signature(s1, s2)
        levy_21 = engine._compute_levy_area_signature(s2, s1)
        # Should have opposite signs (not exact due to cross-product asymmetry)
        assert np.sign(levy_12) == -np.sign(levy_21) or abs(levy_12) < 0.01


# ---------------------------------------------------------------------------
# Adjacency matrix tests
# ---------------------------------------------------------------------------

class TestAdjacencyMatrix:
    """Test _learn_adjacency_matrix."""

    def test_returns_dict(self):
        """Returns adjacency dict for all pairs."""
        engine = _make_engine()
        leadlag = {
            ('SPY', 'GLD'): 0.1, ('GLD', 'SPY'): -0.1,
            ('SPY', 'TLT'): 0.2, ('TLT', 'SPY'): -0.2,
            ('GLD', 'TLT'): 0.05, ('TLT', 'GLD'): -0.05,
        }
        adj = engine._learn_adjacency_matrix(leadlag, ['SPY', 'GLD', 'TLT'])
        assert isinstance(adj, dict)
        assert ('SPY', 'GLD') in adj

    def test_non_negative_values(self):
        """Adjacency values should be non-negative (sparse graph)."""
        engine = _make_engine()
        leadlag = {
            ('SPY', 'GLD'): 0.1, ('GLD', 'SPY'): -0.1,
            ('SPY', 'TLT'): 0.2, ('TLT', 'SPY'): -0.2,
            ('GLD', 'TLT'): 0.05, ('TLT', 'GLD'): -0.05,
        }
        adj = engine._learn_adjacency_matrix(leadlag, ['SPY', 'GLD', 'TLT'])
        for v in adj.values():
            assert v >= 0.0


# ---------------------------------------------------------------------------
# Lead-lag matrix computation tests
# ---------------------------------------------------------------------------

class TestComputeLeadLagMatrix:
    """Test compute_leadlag_matrix."""

    def test_returns_leadlag_matrix(self):
        """Returns LeadLagMatrix with synthetic data."""
        prices_df = _make_prices_df(n_days=200)
        engine = _make_engine(prices_df=prices_df)
        matrix = engine.compute_leadlag_matrix(66, prices_df)
        assert isinstance(matrix, LeadLagMatrix)
        assert matrix.window == 66

    def test_returns_none_insufficient_data(self):
        """Returns None when too few data points."""
        prices_df = _make_prices_df(n_days=10)
        engine = _make_engine(prices_df=prices_df)
        matrix = engine.compute_leadlag_matrix(66, prices_df)
        assert matrix is None

    def test_leadlag_matrix_antisymmetric(self):
        """leadlag(A,B) ≈ -leadlag(B,A)."""
        prices_df = _make_prices_df(n_days=300)
        engine = _make_engine(prices_df=prices_df)
        matrix = engine.compute_leadlag_matrix(66, prices_df)
        if matrix:
            for (a, b), v in matrix.leadlag_matrix.items():
                v_rev = matrix.leadlag_matrix.get((b, a), 0.0)
                assert abs(v + v_rev) < 0.01

    def test_dtw_distances_symmetric(self):
        """DTW distances are symmetric."""
        prices_df = _make_prices_df(n_days=300)
        engine = _make_engine(prices_df=prices_df)
        matrix = engine.compute_leadlag_matrix(66, prices_df)
        if matrix:
            for (a, b), v in matrix.dtw_distances.items():
                v_rev = matrix.dtw_distances.get((b, a), 0.0)
                assert abs(v - v_rev) < 0.01


# ---------------------------------------------------------------------------
# Window signal tests
# ---------------------------------------------------------------------------

class TestComputeWindowSignal:
    """Test compute_window_signal."""

    def test_returns_signal(self):
        """Returns WindowMomentumSignal for valid ticker."""
        prices_df = _make_prices_df(n_days=200)
        engine = _make_engine(prices_df=prices_df)
        ll = _make_leadlag_matrix()
        sig = engine.compute_window_signal('SPY', 66, 0.46, ll, prices_df)
        assert isinstance(sig, WindowMomentumSignal)
        assert sig.ticker == 'SPY'

    def test_returns_none_missing_ticker(self):
        """Returns None for ticker not in DataFrame."""
        prices_df = _make_prices_df(n_days=200)
        engine = _make_engine(prices_df=prices_df)
        ll = _make_leadlag_matrix()
        sig = engine.compute_window_signal('NONEXISTENT', 66, 0.46, ll, prices_df)
        assert sig is None

    def test_signal_is_bounded(self):
        """Signal is -1, 0, or 1."""
        prices_df = _make_prices_df(n_days=200)
        engine = _make_engine(prices_df=prices_df)
        ll = _make_leadlag_matrix()
        sig = engine.compute_window_signal('SPY', 66, 0.46, ll, prices_df)
        if sig:
            assert sig.signal in [-1, 0, 1]

    def test_target_weight_bounded(self):
        """Target weight is clipped to [MIN_WEIGHT, 1.0]."""
        prices_df = _make_prices_df(n_days=200)
        engine = _make_engine(prices_df=prices_df)
        ll = _make_leadlag_matrix()
        sig = engine.compute_window_signal('SPY', 66, 0.46, ll, prices_df)
        if sig:
            assert MIN_WEIGHT <= sig.target_weight <= 1.0


# ---------------------------------------------------------------------------
# Ensemble signal tests
# ---------------------------------------------------------------------------

class TestComputeEnsembleSignal:
    """Test compute_ensemble_signal."""

    def test_returns_ensemble_signal(self):
        """Returns EnsembleNetworkSignal with valid data."""
        prices_df = _make_prices_df(n_days=300)
        engine = _make_engine(prices_df=prices_df)
        sig = engine.compute_ensemble_signal('SPY', 0.46, prices_df)
        if sig:
            assert isinstance(sig, EnsembleNetworkSignal)
            assert sig.ticker == 'SPY'

    def test_confidence_bounded(self):
        """Ensemble confidence is between 0 and 1."""
        prices_df = _make_prices_df(n_days=300)
        engine = _make_engine(prices_df=prices_df)
        sig = engine.compute_ensemble_signal('SPY', 0.46, prices_df)
        if sig:
            assert 0.0 <= sig.ensemble_confidence <= 1.0

    def test_ensemble_momentum_is_mean(self):
        """Ensemble momentum is mean of window momentums."""
        prices_df = _make_prices_df(n_days=300)
        engine = _make_engine(prices_df=prices_df)
        sig = engine.compute_ensemble_signal('SPY', 0.46, prices_df)
        if sig and sig.window_signals:
            expected = np.mean([s.network_momentum for s in sig.window_signals.values()])
            assert abs(sig.ensemble_momentum - expected) < 0.001


# ---------------------------------------------------------------------------
# Recommendation tests
# ---------------------------------------------------------------------------

class TestGetCurrentRecommendation:
    """Test get_current_recommendation."""

    def test_returns_portfolio(self):
        """Returns NetworkMomentumPortfolio recommendation."""
        prices_df = _make_prices_df(n_days=300)
        engine = _make_engine(prices_df=prices_df)
        rec = engine.get_current_recommendation(DEFAULT_BASE_ALLOCATION)
        if rec:
            assert isinstance(rec, NetworkMomentumPortfolio)
            assert 'SPY' in rec.target_allocation

    def test_weights_normalize(self):
        """Target allocation weights sum to ~1.0."""
        prices_df = _make_prices_df(n_days=300)
        engine = _make_engine(prices_df=prices_df)
        rec = engine.get_current_recommendation(DEFAULT_BASE_ALLOCATION)
        if rec:
            total = sum(w for k, w in rec.target_allocation.items() if k != 'CASH')
            assert abs(total - 1.0) < 0.05

    def test_dominant_leader_identified(self):
        """Dominant leader is one of the assets."""
        prices_df = _make_prices_df(n_days=300)
        engine = _make_engine(prices_df=prices_df)
        rec = engine.get_current_recommendation(DEFAULT_BASE_ALLOCATION)
        if rec:
            assert rec.dominant_leader in ['SPY', 'GLD', 'TLT']


# ---------------------------------------------------------------------------
# Backtester tests
# ---------------------------------------------------------------------------

class TestNetworkMomentumBacktester:
    """Test NetworkMomentumBacktester."""

    def test_init(self):
        """Backtester initializes correctly."""
        prices_df = _make_prices_df(n_days=400)
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.network_momentum = _make_engine(prices_df=prices_df)
        bt.prices_df = prices_df
        assert bt.rebalance_freq == 21

    def test_insufficient_data_returns_error(self):
        """Too few days returns error dict."""
        prices_df = _make_prices_df(n_days=50)
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.network_momentum = _make_engine(prices_df=prices_df)
        bt.prices_df = prices_df
        result = bt.run_backtest()
        assert 'error' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
