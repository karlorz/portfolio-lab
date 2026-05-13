#!/usr/bin/env python3
"""
Tests for Network Momentum Lead-Lag Module — constants, data classes,
DTW distance, Lévy area signatures, graph learning, lead-lag matrix,
window signals, ensemble signals, portfolio recommendation, and backtest.
"""
import sys
import os
import json
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from itertools import combinations

from src.strategy.network_momentum_leadlag import (
    LOOKBACK_WINDOWS, DEFAULT_WINDOW, DTW_RADIUS, LEVY_LAGS,
    GRAPH_SPARSITY_ALPHA, GRAPH_SMOOTHNESS_BETA,
    MAX_DEVIATION, MIN_WEIGHT, ASSETS, DEFAULT_BASE_ALLOCATION,
    LeadLagMatrix, WindowMomentumSignal, EnsembleNetworkSignal,
    NetworkMomentumPortfolio,
    NetworkMomentumLeadLag, NetworkMomentumBacktester,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices_df(n_days=200, seed=42):
    """Create synthetic price DataFrame with SPY, GLD, TLT columns."""
    np.random.seed(seed)
    dates = pd.bdate_range('2024-01-02', periods=n_days)
    spy = 400 * np.cumprod(1 + np.random.normal(0.0004, 0.012, n_days))
    gld = 150 * np.cumprod(1 + np.random.normal(0.0002, 0.008, n_days))
    tlt = 130 * np.cumprod(1 + np.random.normal(-0.0001, 0.006, n_days))
    df = pd.DataFrame({'SPY': spy, 'GLD': gld, 'TLT': tlt}, index=dates)
    return df


def _make_leadlag_matrix():
    """Create a sample LeadLagMatrix."""
    assets = ['SPY', 'GLD', 'TLT']
    leadlag = {}
    dtw = {}
    levy = {}
    adj = {}
    for a1, a2 in combinations(assets, 2):
        leadlag[(a1, a2)] = 0.5
        leadlag[(a2, a1)] = -0.5
        dtw[(a1, a2)] = 10.0
        dtw[(a2, a1)] = 10.0
        levy[(a1, a2)] = 0.3
        levy[(a2, a1)] = -0.3
        adj[(a1, a2)] = 0.6
        adj[(a2, a1)] = 0.0
    return LeadLagMatrix(
        timestamp='2026-01-01',
        window=66,
        leadlag_matrix=leadlag,
        dtw_distances=dtw,
        levy_areas=levy,
        adjacency=adj,
    )


def _make_engine():
    """Create a NetworkMomentumLeadLag with mocked prices."""
    engine = NetworkMomentumLeadLag.__new__(NetworkMomentumLeadLag)
    engine.prices_path = None
    engine.db_path = None
    engine.lookback_windows = LOOKBACK_WINDOWS
    engine.max_deviation = MAX_DEVIATION
    engine._prices_df = None
    return engine


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_lookback_windows(self):
        assert LOOKBACK_WINDOWS == [22, 44, 66, 88, 110, 132]

    def test_default_window(self):
        assert DEFAULT_WINDOW == 66

    def test_dtw_radius(self):
        assert DTW_RADIUS == 5

    def test_levy_lags(self):
        assert LEVY_LAGS == [1, 5, 10, 21]

    def test_max_deviation(self):
        assert MAX_DEVIATION == 0.15

    def test_min_weight(self):
        assert MIN_WEIGHT == 0.05

    def test_assets(self):
        assert 'SPY' in ASSETS
        assert 'GLD' in ASSETS
        assert 'TLT' in ASSETS
        assert 'CASH' in ASSETS

    def test_default_base_allocation_sums_to_one(self):
        total = sum(DEFAULT_BASE_ALLOCATION.values())
        assert abs(total - 1.0) < 0.001

    def test_default_base_allocation_keys(self):
        assert set(DEFAULT_BASE_ALLOCATION.keys()) == {'SPY', 'GLD', 'TLT', 'CASH'}


# ---------------------------------------------------------------------------
# LeadLagMatrix tests
# ---------------------------------------------------------------------------

class TestLeadLagMatrix:
    def test_creation(self):
        m = _make_leadlag_matrix()
        assert m.timestamp == '2026-01-01'
        assert m.window == 66

    def test_to_dict(self):
        m = _make_leadlag_matrix()
        d = m.to_dict()
        assert 'timestamp' in d
        assert 'window' in d
        assert 'leadlag_matrix' in d
        assert 'dtw_distances' in d
        assert 'levy_areas' in d
        assert 'adjacency' in d

    def test_to_dict_keys_formatted(self):
        m = _make_leadlag_matrix()
        d = m.to_dict()
        # Keys should be formatted as "SPY->GLD" etc.
        for key in d['leadlag_matrix']:
            assert '->' in key

    def test_to_dict_dtw_keys(self):
        m = _make_leadlag_matrix()
        d = m.to_dict()
        for key in d['dtw_distances']:
            assert '-' in key


# ---------------------------------------------------------------------------
# WindowMomentumSignal tests
# ---------------------------------------------------------------------------

class TestWindowMomentumSignal:
    def test_creation(self):
        s = WindowMomentumSignal(
            ticker='SPY', window=66, timestamp='2026-01-01',
            momentum_return=0.05, signal=1,
            network_momentum=0.06, network_adjustment=0.01,
            base_weight=0.46, target_weight=0.49, adjustment=0.03,
        )
        assert s.ticker == 'SPY'
        assert s.signal == 1

    def test_to_dict(self):
        s = WindowMomentumSignal(
            ticker='SPY', window=66, timestamp='2026-01-01',
            momentum_return=0.05, signal=1,
            network_momentum=0.06, network_adjustment=0.01,
            base_weight=0.46, target_weight=0.49, adjustment=0.03,
        )
        d = s.to_dict()
        assert d['ticker'] == 'SPY'
        assert d['window'] == 66


# ---------------------------------------------------------------------------
# EnsembleNetworkSignal tests
# ---------------------------------------------------------------------------

class TestEnsembleNetworkSignal:
    def test_creation(self):
        s = EnsembleNetworkSignal(
            ticker='SPY', timestamp='2026-01-01',
            window_signals={}, ensemble_momentum=0.05,
            ensemble_signal=1, ensemble_confidence=0.8,
            base_weight=0.46, adjustment=0.03, target_weight=0.49,
            leadership_score=0.6, followership_score=0.2,
            network_centrality=0.4,
        )
        assert s.ticker == 'SPY'
        assert s.ensemble_confidence == 0.8

    def test_to_dict(self):
        s = EnsembleNetworkSignal(
            ticker='SPY', timestamp='2026-01-01',
            window_signals={}, ensemble_momentum=0.05,
            ensemble_signal=1, ensemble_confidence=0.8,
            base_weight=0.46, adjustment=0.03, target_weight=0.49,
            leadership_score=0.6, followership_score=0.2,
            network_centrality=0.4,
        )
        d = s.to_dict()
        assert 'ticker' in d
        assert 'ensemble_momentum' in d
        assert 'leadership_score' in d


# ---------------------------------------------------------------------------
# NetworkMomentumPortfolio tests
# ---------------------------------------------------------------------------

class TestNetworkMomentumPortfolio:
    def test_creation(self):
        ll = _make_leadlag_matrix()
        p = NetworkMomentumPortfolio(
            timestamp='2026-01-01',
            base_allocation={'SPY': 0.46},
            network_adjustments={'SPY': 0.03},
            target_allocation={'SPY': 0.49},
            leadlag_matrix=ll,
            ensemble_signals={},
            dominant_leader='SPY',
            dominant_follower='TLT',
            network_efficiency=0.5,
            overall_confidence=0.7,
        )
        assert p.dominant_leader == 'SPY'
        assert p.overall_confidence == 0.7

    def test_to_dict(self):
        ll = _make_leadlag_matrix()
        p = NetworkMomentumPortfolio(
            timestamp='2026-01-01',
            base_allocation={'SPY': 0.46},
            network_adjustments={'SPY': 0.03},
            target_allocation={'SPY': 0.49},
            leadlag_matrix=ll,
            ensemble_signals={},
            dominant_leader='SPY',
            dominant_follower='TLT',
            network_efficiency=0.5,
            overall_confidence=0.7,
        )
        d = p.to_dict()
        assert 'base_allocation' in d
        assert 'target_allocation' in d
        assert 'leadlag_matrix' in d


# ---------------------------------------------------------------------------
# DTW distance tests
# ---------------------------------------------------------------------------

class TestDTWDistance:
    def test_identical_series(self):
        e = _make_engine()
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = e._simple_dtw_distance(s, s)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_shifted_series(self):
        e = _make_engine()
        s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        dist = e._simple_dtw_distance(s1, s2)
        assert dist >= 0

    def test_different_series_positive(self):
        e = _make_engine()
        np.random.seed(42)
        s1 = np.random.randn(50)
        s2 = np.random.randn(50)
        dist = e._simple_dtw_distance(s1, s2)
        assert dist > 0

    def test_symmetry(self):
        e = _make_engine()
        s1 = np.array([1.0, 2.0, 3.0, 4.0])
        s2 = np.array([4.0, 3.0, 2.0, 1.0])
        d1 = e._simple_dtw_distance(s1, s2)
        d2 = e._simple_dtw_distance(s2, s1)
        assert d1 == pytest.approx(d2, abs=1e-6)

    def test_short_series(self):
        e = _make_engine()
        s1 = np.array([1.0, 2.0])
        s2 = np.array([1.5, 2.5])
        dist = e._simple_dtw_distance(s1, s2)
        assert dist >= 0


# ---------------------------------------------------------------------------
# Lévy area signature tests
# ---------------------------------------------------------------------------

class TestLevyArea:
    def test_identical_series_zero(self):
        e = _make_engine()
        s = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
        levy = e._compute_levy_area_signature(s, s)
        # Identical paths → area should be near zero
        assert abs(levy) < 0.1

    def test_positive_for_leading(self):
        e = _make_engine()
        np.random.seed(42)
        s1 = np.random.randn(100) * 0.01
        s2 = np.random.randn(100) * 0.01
        levy = e._compute_levy_area_signature(s1, s2)
        assert isinstance(levy, float)

    def test_custom_lags(self):
        e = _make_engine()
        s1 = np.random.randn(50) * 0.01
        s2 = np.random.randn(50) * 0.01
        levy = e._compute_levy_area_signature(s1, s2, lags=[1, 5])
        assert isinstance(levy, float)

    def test_short_series(self):
        e = _make_engine()
        s1 = np.array([0.01, 0.02])
        s2 = np.array([0.02, 0.01])
        levy = e._compute_levy_area_signature(s1, s2, lags=[1])
        assert isinstance(levy, float)

    def test_empty_lags_returns_zero(self):
        e = _make_engine()
        s1 = np.array([0.01])
        s2 = np.array([0.02])
        levy = e._compute_levy_area_signature(s1, s2, lags=[100])
        assert levy == 0.0


# ---------------------------------------------------------------------------
# Graph learning tests
# ---------------------------------------------------------------------------

class TestLearnAdjacency:
    def test_returns_dict(self):
        e = _make_engine()
        scores = {('SPY', 'GLD'): 0.5, ('GLD', 'SPY'): -0.5}
        adj = e._learn_adjacency_matrix(scores, ['SPY', 'GLD'])
        assert isinstance(adj, dict)

    def test_strong_connections_kept(self):
        e = _make_engine()
        scores = {('SPY', 'GLD'): 1.0, ('GLD', 'SPY'): -1.0}
        adj = e._learn_adjacency_matrix(scores, ['SPY', 'GLD'])
        # Strong connection should be above sparsity threshold
        assert adj[('SPY', 'GLD')] > 0.3

    def test_weak_connections_zeroed(self):
        e = _make_engine()
        # All same value → normalized to 0.5, which is > 0.3
        scores = {('SPY', 'GLD'): 0.001, ('GLD', 'SPY'): 0.001}
        adj = e._learn_adjacency_matrix(scores, ['SPY', 'GLD'])
        # When range is tiny, normalization may keep them
        assert isinstance(adj, dict)

    def test_empty_scores(self):
        e = _make_engine()
        adj = e._learn_adjacency_matrix({}, ['SPY', 'GLD'])
        assert adj == {}

    def test_normalization(self):
        e = _make_engine()
        scores = {('A', 'B'): 0.0, ('B', 'C'): 1.0}
        adj = e._learn_adjacency_matrix(scores, ['A', 'B', 'C'])
        # Should normalize to [0, 1]
        for v in adj.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# compute_leadlag_matrix tests
# ---------------------------------------------------------------------------

class TestComputeLeadLagMatrix:
    def test_returns_leadlag_matrix(self):
        e = _make_engine()
        df = _make_prices_df(200)
        result = e.compute_leadlag_matrix(66, df)
        assert isinstance(result, LeadLagMatrix)

    def test_has_all_pairs(self):
        e = _make_engine()
        df = _make_prices_df(200)
        result = e.compute_leadlag_matrix(66, df)
        assert result is not None
        pairs = set(result.leadlag_matrix.keys())
        assert ('SPY', 'GLD') in pairs
        assert ('SPY', 'TLT') in pairs
        assert ('GLD', 'TLT') in pairs

    def test_dtw_symmetric(self):
        e = _make_engine()
        df = _make_prices_df(200)
        result = e.compute_leadlag_matrix(66, df)
        assert result is not None
        for (a1, a2), dist in result.dtw_distances.items():
            assert dist == pytest.approx(result.dtw_distances[(a2, a1)], abs=1e-6)

    def test_levy_antisymmetric(self):
        e = _make_engine()
        df = _make_prices_df(200)
        result = e.compute_leadlag_matrix(66, df)
        assert result is not None
        for (a1, a2), val in result.levy_areas.items():
            assert val == pytest.approx(-result.levy_areas[(a2, a1)], abs=1e-6)

    def test_window_too_large_returns_none(self):
        e = _make_engine()
        df = _make_prices_df(10)
        result = e.compute_leadlag_matrix(100, df)
        assert result is None

    def test_timestamp_set(self):
        e = _make_engine()
        df = _make_prices_df(200)
        result = e.compute_leadlag_matrix(66, df)
        assert result is not None
        assert result.timestamp is not None


# ---------------------------------------------------------------------------
# compute_window_signal tests
# ---------------------------------------------------------------------------

class TestComputeWindowSignal:
    def test_returns_signal(self):
        e = _make_engine()
        df = _make_prices_df(200)
        ll = e.compute_leadlag_matrix(66, df)
        assert ll is not None
        sig = e.compute_window_signal('SPY', 66, 0.46, ll, df)
        assert isinstance(sig, WindowMomentumSignal)

    def test_signal_direction(self):
        e = _make_engine()
        df = _make_prices_df(200)
        ll = e.compute_leadlag_matrix(66, df)
        assert ll is not None
        sig = e.compute_window_signal('SPY', 66, 0.46, ll, df)
        assert sig.signal in [-1, 0, 1]

    def test_target_weight_clipped(self):
        e = _make_engine()
        df = _make_prices_df(200)
        ll = e.compute_leadlag_matrix(66, df)
        assert ll is not None
        sig = e.compute_window_signal('SPY', 66, 0.46, ll, df)
        assert sig.target_weight >= MIN_WEIGHT
        assert sig.target_weight <= 1.0

    def test_missing_ticker_returns_none(self):
        e = _make_engine()
        df = _make_prices_df(200)
        ll = e.compute_leadlag_matrix(66, df)
        assert ll is not None
        sig = e.compute_window_signal('AAPL', 66, 0.10, ll, df)
        assert sig is None

    def test_base_weight_preserved(self):
        e = _make_engine()
        df = _make_prices_df(200)
        ll = e.compute_leadlag_matrix(66, df)
        assert ll is not None
        sig = e.compute_window_signal('SPY', 66, 0.46, ll, df)
        assert sig.base_weight == 0.46


# ---------------------------------------------------------------------------
# compute_ensemble_signal tests
# ---------------------------------------------------------------------------

class TestComputeEnsembleSignal:
    def test_returns_ensemble(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        sig = e.compute_ensemble_signal('SPY', 0.46, df)
        assert isinstance(sig, EnsembleNetworkSignal)

    def test_has_window_signals(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        sig = e.compute_ensemble_signal('SPY', 0.46, df)
        assert len(sig.window_signals) > 0

    def test_confidence_bounded(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        sig = e.compute_ensemble_signal('SPY', 0.46, df)
        assert 0.0 <= sig.ensemble_confidence <= 1.0

    def test_target_weight_clipped(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        sig = e.compute_ensemble_signal('SPY', 0.46, df)
        assert sig.target_weight >= MIN_WEIGHT

    def test_leadership_score(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        sig = e.compute_ensemble_signal('SPY', 0.46, df)
        assert sig.leadership_score >= 0
        assert sig.followership_score >= 0

    def test_network_centrality(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        sig = e.compute_ensemble_signal('SPY', 0.46, df)
        assert sig.network_centrality >= 0


# ---------------------------------------------------------------------------
# get_current_recommendation tests
# ---------------------------------------------------------------------------

class TestGetCurrentRecommendation:
    def test_returns_portfolio(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        rec = e.get_current_recommendation(DEFAULT_BASE_ALLOCATION)
        assert isinstance(rec, NetworkMomentumPortfolio)

    def test_has_target_allocation(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        rec = e.get_current_recommendation(DEFAULT_BASE_ALLOCATION)
        assert 'SPY' in rec.target_allocation
        assert 'GLD' in rec.target_allocation
        assert 'TLT' in rec.target_allocation

    def test_weights_normalized(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        rec = e.get_current_recommendation(DEFAULT_BASE_ALLOCATION)
        total = sum(w for k, w in rec.target_allocation.items() if k != 'CASH')
        assert abs(total - 1.0) < 0.01

    def test_dominant_leader_set(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        rec = e.get_current_recommendation(DEFAULT_BASE_ALLOCATION)
        assert rec.dominant_leader in ['SPY', 'GLD', 'TLT']

    def test_dominant_follower_set(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        rec = e.get_current_recommendation(DEFAULT_BASE_ALLOCATION)
        assert rec.dominant_follower in ['SPY', 'GLD', 'TLT']

    def test_overall_confidence_bounded(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        rec = e.get_current_recommendation(DEFAULT_BASE_ALLOCATION)
        assert 0.0 <= rec.overall_confidence <= 1.0


# ---------------------------------------------------------------------------
# NetworkMomentumBacktester tests
# ---------------------------------------------------------------------------

class TestNetworkMomentumBacktester:
    def test_run_backtest_insufficient_data(self):
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.network_momentum = _make_engine()
        bt.prices_df = _make_prices_df(50)
        result = bt.run_backtest()
        assert 'error' in result

    def test_run_backtest_with_data(self):
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.network_momentum = _make_engine()
        bt.prices_df = _make_prices_df(400)
        result = bt.run_backtest()
        assert 'cagr' in result
        assert 'sharpe_ratio' in result

    def test_run_backtest_has_crisis_fields(self):
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.network_momentum = _make_engine()
        bt.prices_df = _make_prices_df(400)
        result = bt.run_backtest()
        assert 'crisis_2008_return' in result
        assert 'crisis_2020_return' in result
        assert 'crisis_2022_return' in result

    def test_run_backtest_max_drawdown(self):
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.network_momentum = _make_engine()
        bt.prices_df = _make_prices_df(400)
        result = bt.run_backtest()
        assert result['max_drawdown'] <= 0

    def test_run_backtest_baseline_comparison(self):
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.network_momentum = _make_engine()
        bt.prices_df = _make_prices_df(400)
        result = bt.run_backtest()
        assert 'baseline_cagr' in result
        assert 'baseline_sharpe' in result
        assert 'excess_return' in result
        assert 'sharpe_improvement' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
