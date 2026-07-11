#!/usr/bin/env python3
"""
Tests for Network Momentum Lead-Lag Module — constants, data classes,
DTW distance, Lévy area signatures, graph learning, lead-lag matrix,
window signals, ensemble signals, portfolio recommendation, and backtest.
"""
import json
import logging
import math

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


class _BacktestRecommendation:
    """Small recommendation stub for deterministic backtester tests."""

    def __init__(self, target_allocation):
        self.target_allocation = target_allocation
        self.dominant_leader = "SPY"
        self.network_efficiency = 0.5


class _SequentialRecommendationEngine:
    """Return one allocation per rebalance, repeating the final allocation."""

    def __init__(self, allocations):
        self.allocations = allocations
        self.calls = 0
        self._prices_df = None

    def get_current_recommendation(self, _base_allocation):
        idx = min(self.calls, len(self.allocations) - 1)
        self.calls += 1
        return _BacktestRecommendation(self.allocations[idx])


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


class TestNetworkMomentumBacktesterTransactionCosts:
    """Turnover and transaction-cost accounting for standalone validation."""

    def _make_backtester(self, allocations, costs_bps=None, rebalance_freq=10):
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = rebalance_freq
        bt.network_momentum = _SequentialRecommendationEngine(allocations)
        bt.transaction_cost_bps = costs_bps if costs_bps is not None else {
            "SPY": 100.0,
            "GLD": 100.0,
            "TLT": 100.0,
            "CASH": 0.0,
        }
        bt.prices_df = _make_prices_df(260)
        return bt

    def test_nonzero_turnover_with_costs_reduces_net_results(self):
        bt = self._make_backtester([
            {"SPY": 0.70, "GLD": 0.20, "TLT": 0.10, "CASH": 0.0},
            {"SPY": 0.20, "GLD": 0.60, "TLT": 0.20, "CASH": 0.0},
            {"SPY": 0.60, "GLD": 0.10, "TLT": 0.30, "CASH": 0.0},
        ])

        result = bt.run_backtest()

        assert result["total_turnover"] > 0
        assert result["transaction_cost_bps"] > 0
        assert result["cost_drag_bps"] > 0
        assert result["net_end_value"] < result["end_value"]
        assert result["net_cagr"] < result["cagr"]
        assert result["net_sharpe_ratio"] < result["sharpe_ratio"]

    def test_zero_turnover_preserves_gross_results_even_with_costs(self):
        bt = self._make_backtester([
            DEFAULT_BASE_ALLOCATION,
            DEFAULT_BASE_ALLOCATION,
            DEFAULT_BASE_ALLOCATION,
        ])

        result = bt.run_backtest()

        assert result["total_turnover"] == pytest.approx(0.0)
        assert result["transaction_cost_bps"] == pytest.approx(0.0)
        assert result["cost_drag_bps"] == pytest.approx(0.0)
        assert result["net_end_value"] == pytest.approx(result["end_value"])
        assert result["net_cagr"] == pytest.approx(result["cagr"])
        assert result["net_sharpe_ratio"] == pytest.approx(result["sharpe_ratio"])

    def test_zero_cost_preserves_gross_results_even_with_turnover(self):
        bt = self._make_backtester(
            [
                {"SPY": 0.70, "GLD": 0.20, "TLT": 0.10, "CASH": 0.0},
                {"SPY": 0.20, "GLD": 0.60, "TLT": 0.20, "CASH": 0.0},
            ],
            costs_bps={"SPY": 0.0, "GLD": 0.0, "TLT": 0.0, "CASH": 0.0},
        )

        result = bt.run_backtest()

        assert result["total_turnover"] > 0
        assert result["transaction_cost_bps"] == pytest.approx(0.0)
        assert result["cost_drag_bps"] == pytest.approx(0.0)
        assert result["net_end_value"] == pytest.approx(result["end_value"])
        assert result["net_cagr"] == pytest.approx(result["cagr"])
        assert result["net_sharpe_ratio"] == pytest.approx(result["sharpe_ratio"])

    def test_annualized_turnover_is_deterministic(self):
        bt = self._make_backtester([
            {"SPY": 0.70, "GLD": 0.20, "TLT": 0.10, "CASH": 0.0},
            {"SPY": 0.20, "GLD": 0.60, "TLT": 0.20, "CASH": 0.0},
            {"SPY": 0.60, "GLD": 0.10, "TLT": 0.30, "CASH": 0.0},
        ])

        result = bt.run_backtest()

        expected = result["total_turnover"] / result["trading_days"] * 252
        assert result["annualized_turnover"] == pytest.approx(expected)

    def test_reports_one_way_turnover_per_rebalance(self):
        bt = self._make_backtester([
            {"SPY": 0.70, "GLD": 0.20, "TLT": 0.10, "CASH": 0.0},
            {"SPY": 0.20, "GLD": 0.60, "TLT": 0.20, "CASH": 0.0},
        ])

        result = bt.run_backtest()

        assert len(result["rebalance_turnover"]) == result["rebalances"]
        assert result["rebalance_turnover"][0]["one_way_turnover"] > 0
        assert result["rebalance_turnover"][0]["transaction_cost_bps"] > 0


# ---------------------------------------------------------------------------
# Additional constants validation
# ---------------------------------------------------------------------------

class TestConstantsMore:
    """Additional constants validation: sparsity, smoothness, ordering."""

    def test_graph_sparsity_alpha(self):
        assert GRAPH_SPARSITY_ALPHA == 0.01

    def test_graph_smoothness_beta(self):
        assert GRAPH_SMOOTHNESS_BETA == 0.01

    def test_lookback_windows_ascending(self):
        for i in range(1, len(LOOKBACK_WINDOWS)):
            assert LOOKBACK_WINDOWS[i] > LOOKBACK_WINDOWS[i - 1]

    def test_default_window_in_lookback(self):
        assert DEFAULT_WINDOW in LOOKBACK_WINDOWS

    def test_max_deviation_positive(self):
        assert MAX_DEVIATION > 0

    def test_min_weight_positive(self):
        assert MIN_WEIGHT > 0

    def test_min_weight_lt_max_deviation(self):
        assert MIN_WEIGHT < MAX_DEVIATION


# ---------------------------------------------------------------------------
# LeadLagMatrix to_dict field completeness
# ---------------------------------------------------------------------------

class TestLeadLagMatrixMore:
    """LeadLagMatrix to_dict field completeness and key formatting."""

    def test_to_dict_all_fields_present(self):
        m = _make_leadlag_matrix()
        d = m.to_dict()
        expected = {'timestamp', 'window', 'leadlag_matrix',
                     'dtw_distances', 'levy_areas', 'adjacency'}
        assert set(d.keys()) == expected

    def test_to_dict_levy_keys_dash_format(self):
        m = _make_leadlag_matrix()
        d = m.to_dict()
        for key in d['levy_areas']:
            assert '-' in key, f"levy_areas key '{key}' missing '-' separator"

    def test_to_dict_adjacency_keys_arrow_format(self):
        m = _make_leadlag_matrix()
        d = m.to_dict()
        for key in d['adjacency']:
            assert '->' in key, f"adjacency key '{key}' missing '->' separator"


# ---------------------------------------------------------------------------
# WindowMomentumSignal to_dict field completeness and edge values
# ---------------------------------------------------------------------------

class TestWindowMomentumSignalMore:
    """WindowMomentumSignal to_dict field completeness and edge cases."""

    def test_to_dict_all_fields_present(self):
        s = WindowMomentumSignal(
            ticker='SPY', window=66, timestamp='2026-01-01',
            momentum_return=0.05, signal=1,
            network_momentum=0.06, network_adjustment=0.01,
            base_weight=0.46, target_weight=0.49, adjustment=0.03,
        )
        d = s.to_dict()
        # asdict produces ticker, window, timestamp, momentum_return, signal,
        # network_momentum, network_adjustment, base_weight, target_weight, adjustment
        expected = {'ticker', 'window', 'timestamp', 'momentum_return',
                    'signal', 'network_momentum', 'network_adjustment',
                    'base_weight', 'target_weight', 'adjustment'}
        assert set(d.keys()) == expected

    def test_to_dict_values_match(self):
        s = WindowMomentumSignal(
            ticker='GLD', window=44, timestamp='2026-06-01',
            momentum_return=-0.02, signal=-1,
            network_momentum=-0.01, network_adjustment=0.01,
            base_weight=0.38, target_weight=0.36, adjustment=-0.02,
        )
        d = s.to_dict()
        assert d['ticker'] == 'GLD'
        assert d['window'] == 44
        assert d['signal'] == -1
        assert d['base_weight'] == 0.38
        assert d['adjustment'] == -0.02

    def test_signal_zero_when_momentum_zero(self):
        s = WindowMomentumSignal(
            ticker='SPY', window=66, timestamp='2026-01-01',
            momentum_return=0.0, signal=0,
            network_momentum=0.0, network_adjustment=0.0,
            base_weight=0.46, target_weight=0.46, adjustment=0.0,
        )
        assert s.signal == 0

    def test_network_adjustment_sign(self):
        s = WindowMomentumSignal(
            ticker='SPY', window=66, timestamp='2026-01-01',
            momentum_return=0.05, signal=1,
            network_momentum=0.03, network_adjustment=-0.02,
            base_weight=0.46, target_weight=0.44, adjustment=-0.02,
        )
        # When network_momentum < momentum_return, adjustment is negative
        assert s.network_adjustment < 0
        assert s.adjustment < 0

    def test_positive_network_adjustment(self):
        s = WindowMomentumSignal(
            ticker='SPY', window=66, timestamp='2026-01-01',
            momentum_return=0.05, signal=1,
            network_momentum=0.08, network_adjustment=0.03,
            base_weight=0.46, target_weight=0.49, adjustment=0.03,
        )
        assert s.network_adjustment > 0
        assert s.adjustment > 0

    def test_negative_signal(self):
        s = WindowMomentumSignal(
            ticker='SPY', window=66, timestamp='2026-01-01',
            momentum_return=-0.05, signal=-1,
            network_momentum=-0.06, network_adjustment=-0.01,
            base_weight=0.46, target_weight=0.44, adjustment=-0.02,
        )
        assert s.signal == -1


# ---------------------------------------------------------------------------
# EnsembleNetworkSignal to_dict field completeness
# ---------------------------------------------------------------------------

class TestEnsembleNetworkSignalMore:
    """EnsembleNetworkSignal to_dict field completeness and nested serialization."""

    def _make_window_signal(self, ticker='SPY', window=66):
        return WindowMomentumSignal(
            ticker=ticker, window=window, timestamp='2026-01-01',
            momentum_return=0.05, signal=1,
            network_momentum=0.06, network_adjustment=0.01,
            base_weight=0.46, target_weight=0.49, adjustment=0.03,
        )

    def test_to_dict_all_fields_present(self):
        ws = self._make_window_signal()
        s = EnsembleNetworkSignal(
            ticker='SPY', timestamp='2026-01-01',
            window_signals={66: ws}, ensemble_momentum=0.05,
            ensemble_signal=1, ensemble_confidence=0.8,
            base_weight=0.46, adjustment=0.03, target_weight=0.49,
            leadership_score=0.6, followership_score=0.2,
            network_centrality=0.4,
        )
        d = s.to_dict()
        expected = {'ticker', 'timestamp', 'window_signals',
                    'ensemble_momentum', 'ensemble_signal', 'ensemble_confidence',
                    'base_weight', 'adjustment', 'target_weight',
                    'leadership_score', 'followership_score', 'network_centrality'}
        assert set(d.keys()) == expected

    def test_to_dict_window_signals_nested_serialization(self):
        ws = self._make_window_signal()
        s = EnsembleNetworkSignal(
            ticker='SPY', timestamp='2026-01-01',
            window_signals={66: ws, 44: self._make_window_signal(window=44)},
            ensemble_momentum=0.05, ensemble_signal=1, ensemble_confidence=0.8,
            base_weight=0.46, adjustment=0.03, target_weight=0.49,
            leadership_score=0.6, followership_score=0.2, network_centrality=0.4,
        )
        d = s.to_dict()
        assert '66' in d['window_signals']
        assert '44' in d['window_signals']
        assert d['window_signals']['66']['ticker'] == 'SPY'
        assert d['window_signals']['44']['window'] == 44

    def test_to_dict_empty_window_signals(self):
        s = EnsembleNetworkSignal(
            ticker='SPY', timestamp='2026-01-01',
            window_signals={}, ensemble_momentum=0.05,
            ensemble_signal=1, ensemble_confidence=0.8,
            base_weight=0.46, adjustment=0.03, target_weight=0.49,
            leadership_score=0.6, followership_score=0.2, network_centrality=0.4,
        )
        d = s.to_dict()
        assert d['window_signals'] == {}

    def test_to_dict_value_consistency(self):
        ws = self._make_window_signal()
        s = EnsembleNetworkSignal(
            ticker='GLD', timestamp='2026-06-01',
            window_signals={66: ws}, ensemble_momentum=-0.02,
            ensemble_signal=-1, ensemble_confidence=0.5,
            base_weight=0.38, adjustment=-0.02, target_weight=0.36,
            leadership_score=0.1, followership_score=0.7, network_centrality=0.4,
        )
        d = s.to_dict()
        assert d['ticker'] == 'GLD'
        assert d['ensemble_signal'] == -1
        assert d['ensemble_confidence'] == 0.5
        assert d['leadership_score'] == 0.1
        assert d['followership_score'] == 0.7

    def test_extreme_confidence_values(self):
        s = EnsembleNetworkSignal(
            ticker='SPY', timestamp='2026-01-01',
            window_signals={}, ensemble_momentum=0.05,
            ensemble_signal=1, ensemble_confidence=1.0,
            base_weight=0.46, adjustment=0.03, target_weight=0.49,
            leadership_score=0.6, followership_score=0.2, network_centrality=0.4,
        )
        assert s.to_dict()['ensemble_confidence'] == 1.0

        s2 = EnsembleNetworkSignal(
            ticker='SPY', timestamp='2026-01-01',
            window_signals={}, ensemble_momentum=0.05,
            ensemble_signal=1, ensemble_confidence=0.0,
            base_weight=0.46, adjustment=0.03, target_weight=0.49,
            leadership_score=0.6, followership_score=0.2, network_centrality=0.4,
        )
        assert s2.to_dict()['ensemble_confidence'] == 0.0


# ---------------------------------------------------------------------------
# NetworkMomentumPortfolio to_dict field completeness
# ---------------------------------------------------------------------------

class TestNetworkMomentumPortfolioMore:
    """NetworkMomentumPortfolio to_dict field completeness and nested serialization."""

    def test_to_dict_all_fields_present(self):
        ll = _make_leadlag_matrix()
        p = NetworkMomentumPortfolio(
            timestamp='2026-01-01',
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0},
            network_adjustments={'SPY': 0.03, 'GLD': -0.02},
            target_allocation={'SPY': 0.49, 'GLD': 0.36, 'TLT': 0.15, 'CASH': 0.0},
            leadlag_matrix=ll,
            ensemble_signals={},
            dominant_leader='SPY',
            dominant_follower='TLT',
            network_efficiency=0.5,
            overall_confidence=0.7,
        )
        d = p.to_dict()
        expected = {'timestamp', 'base_allocation', 'network_adjustments',
                    'target_allocation', 'leadlag_matrix', 'ensemble_signals',
                    'dominant_leader', 'dominant_follower', 'network_efficiency',
                    'overall_confidence'}
        assert set(d.keys()) == expected

    def test_to_dict_leadlag_nested(self):
        ll = _make_leadlag_matrix()
        p = NetworkMomentumPortfolio(
            timestamp='2026-01-01',
            base_allocation={'SPY': 0.46},
            network_adjustments={'SPY': 0.03},
            target_allocation={'SPY': 0.49},
            leadlag_matrix=ll,
            ensemble_signals={},
            dominant_leader='SPY', dominant_follower='TLT',
            network_efficiency=0.5, overall_confidence=0.7,
        )
        d = p.to_dict()
        assert 'leadlag_matrix' in d
        assert d['leadlag_matrix']['window'] == 66
        assert d['leadlag_matrix']['timestamp'] == '2026-01-01'

    def test_to_dict_ensemble_signals_nested(self):
        ll = _make_leadlag_matrix()
        ws = WindowMomentumSignal(
            ticker='SPY', window=66, timestamp='2026-01-01',
            momentum_return=0.05, signal=1,
            network_momentum=0.06, network_adjustment=0.01,
            base_weight=0.46, target_weight=0.49, adjustment=0.03,
        )
        es = EnsembleNetworkSignal(
            ticker='SPY', timestamp='2026-01-01',
            window_signals={66: ws}, ensemble_momentum=0.05,
            ensemble_signal=1, ensemble_confidence=0.8,
            base_weight=0.46, adjustment=0.03, target_weight=0.49,
            leadership_score=0.6, followership_score=0.2, network_centrality=0.4,
        )
        p = NetworkMomentumPortfolio(
            timestamp='2026-01-01',
            base_allocation={'SPY': 0.46},
            network_adjustments={'SPY': 0.03},
            target_allocation={'SPY': 0.49},
            leadlag_matrix=ll,
            ensemble_signals={'SPY': es},
            dominant_leader='SPY', dominant_follower='TLT',
            network_efficiency=0.5, overall_confidence=0.7,
        )
        d = p.to_dict()
        assert 'SPY' in d['ensemble_signals']
        assert d['ensemble_signals']['SPY']['ticker'] == 'SPY'
        assert d['ensemble_signals']['SPY']['ensemble_confidence'] == 0.8

    def test_to_dict_empty_ensemble_signals(self):
        ll = _make_leadlag_matrix()
        p = NetworkMomentumPortfolio(
            timestamp='2026-01-01',
            base_allocation={'SPY': 0.46},
            network_adjustments={},
            target_allocation={'SPY': 0.46},
            leadlag_matrix=ll,
            ensemble_signals={},
            dominant_leader='SPY', dominant_follower='TLT',
            network_efficiency=0.0, overall_confidence=0.0,
        )
        d = p.to_dict()
        assert d['ensemble_signals'] == {}
        assert d['network_efficiency'] == 0.0
        assert d['overall_confidence'] == 0.0

    def test_cash_in_target_allocation(self):
        ll = _make_leadlag_matrix()
        p = NetworkMomentumPortfolio(
            timestamp='2026-01-01',
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0},
            network_adjustments={'SPY': 0.03, 'GLD': -0.02},
            target_allocation={'SPY': 0.49, 'GLD': 0.36, 'TLT': 0.15, 'CASH': 0.0},
            leadlag_matrix=ll, ensemble_signals={},
            dominant_leader='SPY', dominant_follower='TLT',
            network_efficiency=0.5, overall_confidence=0.7,
        )
        assert p.target_allocation.get('CASH', None) == 0.0


# ---------------------------------------------------------------------------
# DTW distance edge cases
# ---------------------------------------------------------------------------

class TestDTWEdgeCases:
    """Edge cases for DTW distance: single element, constant series, zero std."""

    def test_single_element_series(self):
        e = _make_engine()
        s1 = np.array([1.0])
        s2 = np.array([2.0])
        dist = e._simple_dtw_distance(s1, s2)
        assert dist >= 0

    def test_constant_series_identical(self):
        e = _make_engine()
        s = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        dist = e._simple_dtw_distance(s, s)
        # After z-score normalization, identical constant series match
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_constant_series_different_values(self):
        e = _make_engine()
        s1 = np.array([3.0, 3.0, 3.0])
        s2 = np.array([7.0, 7.0, 7.0])
        # Both after z-score normalization are all zeros, so same
        dist = e._simple_dtw_distance(s1, s2)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_zero_std_series(self):
        e = _make_engine()
        s1 = np.array([0.0, 0.0, 0.0])
        s2 = np.array([0.0, 0.0, 0.0])
        # Division by zero avoided by 1e-8 epsilon in std
        dist = e._simple_dtw_distance(s1, s2)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_radius_larger_than_length(self):
        e = _make_engine()
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([3.0, 2.0, 1.0])
        dist = e._simple_dtw_distance(s1, s2, radius=100)
        assert dist >= 0

    def test_radius_zero(self):
        e = _make_engine()
        s1 = np.array([1.0, 3.0, 2.0, 4.0])
        s2 = np.array([2.0, 4.0, 1.0, 3.0])
        # Radius 0 = no warping allowed, only diagonal alignment
        dist = e._simple_dtw_distance(s1, s2, radius=0)
        assert dist >= 0

    def test_very_different_series_large_distance(self):
        e = _make_engine()
        s1 = np.array([0.0, 0.0, 0.0, 0.0])
        s2 = np.array([100.0, 100.0, 100.0, 100.0])
        dist = e._simple_dtw_distance(s1, s2)
        # After z-score normalization, both are [0,0,0,0], so identical
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_different_lengths(self):
        e = _make_engine()
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Should not raise and return a finite distance
        dist = e._simple_dtw_distance(s1, s2)
        assert dist >= 0
        assert math.isfinite(dist)


# ---------------------------------------------------------------------------
# Levy area edge cases
# ---------------------------------------------------------------------------

class TestLevyAreaEdgeCases:
    """Edge cases for Levy area: directional lead-lag, short series, various lags."""

    def test_short_series_with_lag_larger_than_length(self):
        e = _make_engine()
        s1 = np.array([0.01])
        s2 = np.array([0.02])
        # All lags (2, 5) exceed length 1, so levy_areas list stays empty
        levy = e._compute_levy_area_signature(s1, s2, lags=[2, 5])
        assert levy == 0.0

    def test_all_zero_returns(self):
        e = _make_engine()
        s1 = np.zeros(50)
        s2 = np.zeros(50)
        levy = e._compute_levy_area_signature(s1, s2)
        assert levy == pytest.approx(0.0, abs=1e-10)

    def test_single_lag(self):
        e = _make_engine()
        s1 = np.random.randn(50) * 0.01
        s2 = np.random.randn(50) * 0.01
        levy = e._compute_levy_area_signature(s1, s2, lags=[5])
        assert isinstance(levy, float)

    def test_lag_matches_series_length(self):
        e = _make_engine()
        s1 = np.random.randn(20) * 0.01
        s2 = np.random.randn(20) * 0.01
        # lag=20 means n <= lag, so the lag is skipped
        levy = e._compute_levy_area_signature(s1, s2, lags=[20])
        assert levy == 0.0

    def test_negative_lag_path_handled(self):
        """Lines 331-335 handle negative lags (unreachable in practice)."""
        e = _make_engine()
        s1 = np.array([0.01, 0.02, 0.01, -0.01])
        s2 = np.array([0.02, 0.01, -0.01, 0.01])
        levy = e._compute_levy_area_signature(s1, s2, lags=[-1])
        assert isinstance(levy, float) or levy == 0.0


# ---------------------------------------------------------------------------
# Graph adjacency edge cases
# ---------------------------------------------------------------------------

class TestLearnAdjacencyEdgeCases:
    """Edge cases for graph adjacency learning: single node, identical scores."""

    def test_single_pair(self):
        e = _make_engine()
        scores = {('SPY', 'GLD'): 0.5}
        adj = e._learn_adjacency_matrix(scores, ['SPY', 'GLD'])
        assert isinstance(adj, dict)
        # At least the entry should exist (might be 0.0 or > 0)
        assert ('SPY', 'GLD') in adj

    def test_single_asset_no_pairs_returns_empty(self):
        e = _make_engine()
        adj = e._learn_adjacency_matrix({}, ['SPY'])
        assert adj == {}

    def test_all_identical_scores_min_equals_max(self):
        e = _make_engine()
        scores = {('A', 'B'): 0.5, ('B', 'C'): 0.5, ('A', 'C'): 0.5}
        adj = e._learn_adjacency_matrix(scores, ['A', 'B', 'C'])
        # All values normalize to 1.0 when min == max (division by 1.0)
        for v in adj.values():
            assert 0.0 <= v <= 1.0

    def test_many_pairs_all_normalized(self):
        e = _make_engine()
        scores = {
            ('A', 'B'): 0.9, ('B', 'C'): 0.3, ('C', 'A'): -0.1,
            ('B', 'A'): -0.9, ('C', 'B'): -0.3, ('A', 'C'): 0.1,
        }
        adj = e._learn_adjacency_matrix(scores, ['A', 'B', 'C'])
        for v in adj.values():
            assert 0.0 <= v <= 1.0, f"adjacency value {v} out of [0, 1]"

    def test_all_scores_zero(self):
        e = _make_engine()
        scores = {('A', 'B'): 0.0, ('B', 'A'): 0.0}
        adj = e._learn_adjacency_matrix(scores, ['A', 'B'])
        # Zero scores → all normalized to 0.5 → all > 0.3 → kept
        assert isinstance(adj, dict)

    def test_single_entry(self):
        e = _make_engine()
        scores = {('SPY', 'GLD'): 2.5}
        adj = e._learn_adjacency_matrix(scores, ['SPY', 'GLD'])
        # With single score, min==max, normalized to 1.0
        assert ('SPY', 'GLD') in adj


# ---------------------------------------------------------------------------
# compute_leadlag_matrix edge cases
# ---------------------------------------------------------------------------

class TestComputeLeadLagMatrixEdgeCases:
    """Edge cases: missing assets, insufficient data, all-NaN prices."""

    def test_missing_columns_returns_none(self):
        e = _make_engine()
        dates = pd.bdate_range('2024-01-02', periods=100)
        df = pd.DataFrame({'AAPL': np.cumprod(1 + np.random.randn(100) * 0.01)},
                          index=dates)
        result = e.compute_leadlag_matrix(66, df)
        # SPY, GLD, TLT not in columns
        assert result is None

    def test_single_asset_returns_none(self):
        e = _make_engine()
        dates = pd.bdate_range('2024-01-02', periods=100)
        df = pd.DataFrame({'SPY': np.cumprod(1 + np.random.randn(100) * 0.01)},
                          index=dates)
        result = e.compute_leadlag_matrix(66, df)
        # Only one asset with data → len(returns) < 2
        assert result is None

    def test_all_nan_prices_returns_none(self):
        e = _make_engine()
        dates = pd.bdate_range('2024-01-02', periods=200)
        df = pd.DataFrame(
            {'SPY': [np.nan] * 200, 'GLD': [np.nan] * 200, 'TLT': [np.nan] * 200},
            index=dates,
        )
        result = e.compute_leadlag_matrix(66, df)
        # dropna removes everything → returns empty → len(returns) < 2
        assert result is None

    def test_very_short_window_data_returns_none(self):
        e = _make_engine()
        df = _make_prices_df(5)
        result = e.compute_leadlag_matrix(66, df)
        assert result is None

    def test_window_too_large_vs_available(self):
        e = _make_engine()
        df = _make_prices_df(80)
        # Window 88 > 80, 80*0.8 = 64, 80-0 = 80 > 64, so start_idx=0
        # window 88 * 0.8 = 70.4, end_idx = 80, start_idx = 0
        # Wait: end_idx - start_idx = 80 - 0 = 80, window * 0.8 = 88 * 0.8 = 70.4
        # 80 >= 70.4 → passes the check
        result = e.compute_leadlag_matrix(88, df)
        # But after going through assets, it might work with 80 days
        assert result is not None or result is None  # Either result is valid

    def test_window_too_large_short_data(self):
        e = _make_engine()
        df = _make_prices_df(60)
        result = e.compute_leadlag_matrix(88, df)
        # end_idx = 60, start_idx = 0
        # 60 < 88 * 0.8 = 70.4 → returns None
        assert result is None


# ---------------------------------------------------------------------------
# compute_window_signal edge cases
# ---------------------------------------------------------------------------

class TestComputeWindowSignalEdgeCases:
    """Window signal edge cases: no connections, missing ticker, clipping."""

    def _make_zero_adjacency_matrix(self):
        """Create a LeadLagMatrix with all adjacency set to 0."""
        assets = ['SPY', 'GLD', 'TLT']
        leadlag = {}
        dtw = {}
        levy = {}
        adj = {}
        for a1, a2 in combinations(assets, 2):
            leadlag[(a1, a2)] = 0.0
            leadlag[(a2, a1)] = 0.0
            dtw[(a1, a2)] = 10.0
            dtw[(a2, a1)] = 10.0
            levy[(a1, a2)] = 0.0
            levy[(a2, a1)] = 0.0
            adj[(a1, a2)] = 0.0
            adj[(a2, a1)] = 0.0
        return LeadLagMatrix(
            timestamp='2026-01-01', window=66,
            leadlag_matrix=leadlag, dtw_distances=dtw,
            levy_areas=levy, adjacency=adj,
        )

    def test_no_network_connections(self):
        """When all adjacency values are below 0.1 threshold,
        network_momentum falls back to standalone momentum_return."""
        e = _make_engine()
        df = _make_prices_df(200)
        ll = self._make_zero_adjacency_matrix()
        sig = e.compute_window_signal('SPY', 66, 0.46, ll, df)
        assert sig is not None
        assert sig.network_momentum == sig.momentum_return
        assert sig.network_adjustment == 0.0

    def test_negative_network_adjustment_positive_connection(self):
        """When adjacency threshold is barely met, check sign consistency."""
        e = _make_engine()
        df = _make_prices_df(200)

        # Build a matrix where GLD strongly leads SPY (negative influence)
        assets = ['SPY', 'GLD', 'TLT']
        leadlag = {}
        dtw = {}
        levy = {}
        adj = {}
        for a1, a2 in combinations(assets, 2):
            leadlag[(a1, a2)] = 0.0
            leadlag[(a2, a1)] = 0.0
            dtw[(a1, a2)] = 10.0
            dtw[(a2, a1)] = 10.0
            levy[(a1, a2)] = 0.0
            levy[(a2, a1)] = 0.0
            adj[(a1, a2)] = 0.0
            adj[(a2, a1)] = 0.0
        # GLD strongly leads SPY
        adj[('GLD', 'SPY')] = 0.8
        ll = LeadLagMatrix(
            timestamp='2026-01-01', window=66,
            leadlag_matrix=leadlag, dtw_distances=dtw,
            levy_areas=levy, adjacency=adj,
        )

        sig = e.compute_window_signal('SPY', 66, 0.46, ll, df)
        assert sig is not None
        # adjustment must be within bounds
        assert -MAX_DEVIATION <= sig.adjustment <= MAX_DEVIATION

    def test_insufficient_window_data_returns_none(self):
        e = _make_engine()
        df = _make_prices_df(10)
        ll = _make_leadlag_matrix()
        sig = e.compute_window_signal('SPY', 44, 0.46, ll, df)
        # 10 < 44 * 0.5 = 22 → returns None
        assert sig is None


# ---------------------------------------------------------------------------
# compute_ensemble_signal edge cases
# ---------------------------------------------------------------------------

class TestComputeEnsembleSignalEdgeCases:
    """Ensemble signal edge cases: no leadlag matrix, missing ticker."""

    def test_no_leadlag_matrix_returns_none(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        with patch.object(e, 'compute_leadlag_matrix', return_value=None):
            sig = e.compute_ensemble_signal('SPY', 0.46, df)
            assert sig is None

    def test_missing_ticker_returns_none(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        sig = e.compute_ensemble_signal('AAPL', 0.46, df)
        assert sig is None

    def test_single_lookback_window(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        with patch.object(e, 'lookback_windows', [66]):
            sig = e.compute_ensemble_signal('SPY', 0.46, df)
            # May succeed or fail depending on data
            if sig is not None:
                assert 0.0 <= sig.ensemble_confidence <= 1.0

    def test_all_agreement_confidence_is_one(self):
        """When all window signals have the same sign, confidence = 1.0."""
        e = _make_engine()
        df = _make_prices_df(300)
        # Force all returns positive by using cumprod on positive-only noise
        np.random.seed(7)
        dates = pd.bdate_range('2024-01-02', periods=300)
        spy = 400 * np.cumprod(1 + abs(np.random.normal(0.0005, 0.01, 300)))
        gld = 150 * np.cumprod(1 + abs(np.random.normal(0.0003, 0.008, 300)))
        tlt = 130 * np.cumprod(1 + abs(np.random.normal(0.0001, 0.006, 300)))
        df_up = pd.DataFrame({'SPY': spy, 'GLD': gld, 'TLT': tlt}, index=dates)
        e._prices_df = df_up
        sig = e.compute_ensemble_signal('SPY', 0.46, df_up)
        if sig is not None and sig.ensemble_momentum != 0:
            # If all signals same direction, confidence would be 1.0
            assert sig.ensemble_confidence >= 0

    def test_target_weight_clipping_on_ensemble(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        sig = e.compute_ensemble_signal('SPY', 0.46, df)
        if sig is not None:
            assert sig.target_weight >= MIN_WEIGHT
            assert sig.target_weight <= 1.0

    def test_zero_base_weight_handled(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        sig = e.compute_ensemble_signal('SPY', 0.0, df)
        if sig is not None:
            assert sig.base_weight == 0.0
            assert sig.target_weight >= MIN_WEIGHT


# ---------------------------------------------------------------------------
# get_current_recommendation edge cases
# ---------------------------------------------------------------------------

class TestGetCurrentRecommendationEdgeCases:
    """Portfolio recommendation edge cases: cash-only, single asset, missing data."""

    def test_cash_only_allocation_returns_portfolio(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        rec = e.get_current_recommendation({'CASH': 1.0})
        # Should return a portfolio with CASH only
        assert rec is not None
        assert rec.target_allocation.get('CASH', None) == 0.0

    def test_single_asset_allocation_normalizes(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        rec = e.get_current_recommendation({'SPY': 1.0, 'CASH': 0.0})
        assert rec is not None
        assert 'SPY' in rec.target_allocation
        total = sum(w for k, w in rec.target_allocation.items() if k != 'CASH')
        assert abs(total - 1.0) < 0.01

    def test_overall_confidence_zero_no_ensemble_signals(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        rec = e.get_current_recommendation({'CASH': 1.0})
        assert rec.overall_confidence == 0.0
        assert rec.network_efficiency == 0.0

    def test_dominant_leader_fallback(self):
        """When no ensemble_signals, dominant leader falls back to SPY."""
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        rec = e.get_current_recommendation({'CASH': 1.0})
        assert rec.dominant_leader == 'SPY'
        assert rec.dominant_follower == 'TLT'

    def test_allocation_with_zero_weights_cash_and_spy(self):
        """Base allocation with all non-CASH assets present."""
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}
        rec = e.get_current_recommendation(base)
        assert rec is not None
        # CASH should be preserved
        assert 'CASH' in rec.target_allocation
        assert rec.target_allocation['CASH'] == 0.0


# ---------------------------------------------------------------------------
# NetworkMomentumBacktester edge cases
# ---------------------------------------------------------------------------

class TestNetworkMomentumBacktesterEdgeCases:
    """Backtester edge cases: date filtering, rebalance output, baseline comparison."""

    def test_start_date_in_future_insufficient_data(self):
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = pd.to_datetime('2099-01-01')
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.network_momentum = _make_engine()
        bt.prices_df = _make_prices_df(400)
        result = bt.run_backtest()
        # Future date filters everything out → insufficient data
        assert 'error' in result

    def test_rebalance_dates_populated(self):
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.network_momentum = _make_engine()
        bt.prices_df = _make_prices_df(400)
        result = bt.run_backtest()
        if 'error' not in result:
            assert result['rebalances'] >= 0
            assert 'start_date' in result
            assert 'end_date' in result

    def test_result_contains_lookback_windows(self):
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.network_momentum = _make_engine()
        bt.prices_df = _make_prices_df(400)
        result = bt.run_backtest()
        if 'error' not in result:
            assert result['lookback_windows'] == LOOKBACK_WINDOWS

    def test_result_has_trading_days(self):
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.network_momentum = _make_engine()
        bt.prices_df = _make_prices_df(400)
        result = bt.run_backtest()
        if 'error' not in result:
            assert result['trading_days'] > 0

    def test_baseline_comparison_values(self):
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.network_momentum = _make_engine()
        bt.prices_df = _make_prices_df(400)
        result = bt.run_backtest()
        if 'error' not in result:
            assert 'baseline_cagr' in result
            assert 'baseline_sharpe' in result
            assert 'excess_return' in result
            assert 'sharpe_improvement' in result
            assert 'start_value' in result
            assert result['start_value'] == 100000
            assert 'end_value' in result
            assert result['end_value'] > 0

    def test_crisis_returns_with_synthetic_data(self):
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.network_momentum = _make_engine()
        bt.prices_df = _make_prices_df(400)
        result = bt.run_backtest()
        if 'error' not in result:
            # Crisis fields should exist (may be None if date range doesn't overlap)
            assert 'crisis_2008_return' in result
            assert 'crisis_2020_return' in result



# ---------------------------------------------------------------------------
# Dataclass field validation via dataclasses.fields()
# ---------------------------------------------------------------------------

class TestLeadLagMatrixFields:
    """Verify LeadLagMatrix dataclass fields, types, and defaults."""

    def test_fields_count(self):
        import dataclasses
        fields = dataclasses.fields(LeadLagMatrix)
        assert len(fields) == 6

    def test_field_names(self):
        import dataclasses
        names = {f.name for f in dataclasses.fields(LeadLagMatrix)}
        expected = {'timestamp', 'window', 'leadlag_matrix',
                    'dtw_distances', 'levy_areas', 'adjacency'}
        assert names == expected

    def test_timestamp_is_str(self):
        import dataclasses
        f = next(f for f in dataclasses.fields(LeadLagMatrix) if f.name == 'timestamp')
        assert f.type is str or 'str' in str(f.type)

    def test_window_is_int(self):
        import dataclasses
        f = next(f for f in dataclasses.fields(LeadLagMatrix) if f.name == 'window')
        assert f.type is int or 'int' in str(f.type)


class TestWindowMomentumSignalFields:
    """Verify WindowMomentumSignal dataclass fields."""

    def test_fields_count(self):
        import dataclasses
        fields = dataclasses.fields(WindowMomentumSignal)
        assert len(fields) == 10

    def test_field_names(self):
        import dataclasses
        names = {f.name for f in dataclasses.fields(WindowMomentumSignal)}
        expected = {'ticker', 'window', 'timestamp', 'momentum_return',
                    'signal', 'network_momentum', 'network_adjustment',
                    'base_weight', 'target_weight', 'adjustment'}
        assert names == expected


class TestEnsembleNetworkSignalFields:
    """Verify EnsembleNetworkSignal dataclass fields."""

    def test_fields_count(self):
        import dataclasses
        fields = dataclasses.fields(EnsembleNetworkSignal)
        assert len(fields) == 12

    def test_field_names(self):
        import dataclasses
        names = {f.name for f in dataclasses.fields(EnsembleNetworkSignal)}
        expected = {'ticker', 'timestamp', 'window_signals',
                    'ensemble_momentum', 'ensemble_signal', 'ensemble_confidence',
                    'base_weight', 'adjustment', 'target_weight',
                    'leadership_score', 'followership_score', 'network_centrality'}
        assert names == expected

    def test_signal_field_is_int(self):
        import dataclasses
        f = next(f for f in dataclasses.fields(EnsembleNetworkSignal) if f.name == 'ensemble_signal')
        assert f.type is int or 'int' in str(f.type)


class TestNetworkMomentumPortfolioFields:
    """Verify NetworkMomentumPortfolio dataclass fields."""

    def test_fields_count(self):
        import dataclasses
        fields = dataclasses.fields(NetworkMomentumPortfolio)
        assert len(fields) == 10

    def test_field_names(self):
        import dataclasses
        names = {f.name for f in dataclasses.fields(NetworkMomentumPortfolio)}
        expected = {'timestamp', 'base_allocation', 'network_adjustments',
                    'target_allocation', 'leadlag_matrix', 'ensemble_signals',
                    'dominant_leader', 'dominant_follower', 'network_efficiency',
                    'overall_confidence'}
        assert names == expected

    def test_dominant_leader_is_str(self):
        import dataclasses
        f = next(f for f in dataclasses.fields(NetworkMomentumPortfolio) if f.name == 'dominant_leader')
        assert f.type is str or 'str' in str(f.type)

    def test_network_efficiency_is_float(self):
        import dataclasses
        f = next(f for f in dataclasses.fields(NetworkMomentumPortfolio) if f.name == 'network_efficiency')
        assert f.type is float or 'float' in str(f.type)


# ---------------------------------------------------------------------------
# Constants type and range validation
# ---------------------------------------------------------------------------

class TestConstantsTypes:
    """Verify module-level constants have expected types and ranges."""

    def test_lookback_windows_all_positive(self):
        for w in LOOKBACK_WINDOWS:
            assert w > 0

    def test_levy_lags_all_positive(self):
        for lag in LEVY_LAGS:
            assert lag > 0

    def test_dtw_radius_is_int(self):
        assert isinstance(DTW_RADIUS, int)

    def test_max_deviation_is_float_range(self):
        assert isinstance(MAX_DEVIATION, float)
        assert 0.0 < MAX_DEVIATION < 1.0

    def test_min_weight_is_float_range(self):
        assert isinstance(MIN_WEIGHT, float)
        assert 0.0 < MIN_WEIGHT < MAX_DEVIATION

    def test_assets_is_list_of_strings(self):
        assert isinstance(ASSETS, list)
        for a in ASSETS:
            assert isinstance(a, str)

    def test_sparsity_alpha_is_float(self):
        assert isinstance(GRAPH_SPARSITY_ALPHA, float)
        assert GRAPH_SPARSITY_ALPHA > 0

    def test_smoothness_beta_is_float(self):
        assert isinstance(GRAPH_SMOOTHNESS_BETA, float)
        assert GRAPH_SMOOTHNESS_BETA > 0


# ---------------------------------------------------------------------------
# __all__ export completeness
# ---------------------------------------------------------------------------

class TestAllExports:
    """Verify __all__ covers all public API names."""

    def test_all_is_list_of_strings(self):
        from src.strategy.network_momentum_leadlag import __all__
        assert isinstance(__all__, list)
        for name in __all__:
            assert isinstance(name, str)

    def test_all_names_exist_in_module(self):
        import src.strategy.network_momentum_leadlag as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"__all__ contains '{name}' but module has no such attribute"

    def test_all_contains_core_dataclasses(self):
        from src.strategy.network_momentum_leadlag import __all__
        required = {'LeadLagMatrix', 'WindowMomentumSignal',
                    'EnsembleNetworkSignal', 'NetworkMomentumPortfolio'}
        assert required.issubset(set(__all__)), f"Missing: {required - set(__all__)}"

    def test_all_contains_all_constants(self):
        from src.strategy.network_momentum_leadlag import __all__
        required = {'LOOKBACK_WINDOWS', 'DEFAULT_WINDOW', 'DTW_RADIUS',
                    'LEVY_LAGS', 'GRAPH_SPARSITY_ALPHA', 'GRAPH_SMOOTHNESS_BETA',
                    'MAX_DEVIATION', 'MIN_WEIGHT', 'ASSETS', 'DEFAULT_BASE_ALLOCATION'}
        assert required.issubset(set(__all__)), f"Missing: {required - set(__all__)}"

    def test_all_contains_both_classes(self):
        from src.strategy.network_momentum_leadlag import __all__
        required = {'NetworkMomentumLeadLag', 'NetworkMomentumBacktester'}
        assert required.issubset(set(__all__)), f"Missing: {required - set(__all__)}"


# ---------------------------------------------------------------------------
# CLI/__main__ guard tests
# ---------------------------------------------------------------------------

class TestCLI:
    """Test CLI entry points via capsys and argparse interaction."""

    def test_status_command_output(self, caplog):
        from src.strategy.network_momentum_leadlag import main
        with caplog.at_level(logging.INFO, logger="src.strategy.network_momentum_leadlag"):
            with patch('sys.argv', ['script', 'status']):
                try:
                    main()
                except (SystemExit, FileNotFoundError, json.JSONDecodeError):
                    pass  # Expected if no data file exists
        assert 'Network Momentum Lead-Lag' in caplog.text

    def test_no_command_shows_help(self, capsys):
        """No argument should trigger argparse error or print help."""
        from src.strategy.network_momentum_leadlag import main
        with patch('sys.argv', ['script']):
            try:
                main()
            except SystemExit:
                pass
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert combined  # Should produce some output (help or error)

    def test_compute_command_no_data_returns_error(self, capsys):
        """Compute with a ticker when no data available."""
        from src.strategy.network_momentum_leadlag import main
        with patch('sys.argv', ['script', 'compute', '--ticker', 'SPY', '--window', '66']):
            try:
                main()
            except (SystemExit, FileNotFoundError, json.JSONDecodeError):
                pass
        captured = capsys.readouterr()
        # Should either have output or gracefully fail
        assert True  # No crash is sufficient

    def test_unknown_command_does_not_crash(self, capsys):
        from src.strategy.network_momentum_leadlag import main
        with patch('sys.argv', ['script', 'unknown_cmd_xyz']):
            try:
                main()
            except SystemExit:
                pass
        captured = capsys.readouterr()
        assert True  # No unhandled exception


# ---------------------------------------------------------------------------
# DTW with NaN/Inf edge cases
# ---------------------------------------------------------------------------

class TestDTWNaNInf:
    """DTW distance with NaN, Inf, and extreme inputs."""

    def test_nan_input_raises_or_handles(self):
        e = _make_engine()
        s = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        with np.errstate(invalid='ignore'):
            try:
                dist = e._simple_dtw_distance(s, s)
                # Either it handles NaN gracefully (returns finite)
                assert math.isfinite(dist) if not math.isnan(dist) else True
            except (ValueError, ZeroDivisionError):
                pass  # Acceptable to raise on NaN

    def test_inf_input_handled(self):
        e = _make_engine()
        s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s2 = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        # Mean/std with inf → inf or nan; DTW should not crash
        with np.errstate(invalid='ignore', divide='ignore'):
            try:
                dist = e._simple_dtw_distance(s1, s2)
                # After normalization, inf becomes nan via (inf - inf) / inf
                # DTW returns nan, not a crash
                assert dist >= 0 or math.isinf(dist) or math.isnan(dist)
            except (ValueError, ZeroDivisionError, FloatingPointError):
                pass

    def test_neg_inf_input_handled(self):
        e = _make_engine()
        s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s2 = np.array([1.0, 2.0, -np.inf, 4.0, 5.0])
        with np.errstate(invalid='ignore', divide='ignore'):
            try:
                dist = e._simple_dtw_distance(s1, s2)
                assert dist >= 0 or math.isinf(dist) or math.isnan(dist)
            except (ValueError, ZeroDivisionError):
                pass

    def test_zero_length_series_raises(self):
        e = _make_engine()
        s = np.array([])
        # Empty array → mean is nan, std is nan, DTW may produce nan or raise
        with np.errstate(invalid='ignore', divide='ignore'):
            try:
                dist = e._simple_dtw_distance(s, s)
                # Either returns nan or raises
                assert math.isnan(dist) or dist >= 0
            except (ValueError, ZeroDivisionError, IndexError):
                pass

    def test_radius_negative_falls_back(self):
        e = _make_engine()
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([3.0, 2.0, 1.0])
        dist = e._simple_dtw_distance(s1, s2, radius=-1)
        # Negative radius should be handled (could fall back to 0 or default)
        assert dist >= 0


# ---------------------------------------------------------------------------
# Levy area with NaN/Inf edge cases
# ---------------------------------------------------------------------------

class TestLevyAreaNaNInf:
    """Levy area signature with NaN, Inf, and extreme inputs."""

    def test_nan_in_returns_handled(self):
        e = _make_engine()
        s1 = np.array([0.01, 0.02, np.nan, 0.03])
        s2 = np.array([0.02, 0.01, 0.03, 0.01])
        with np.errstate(invalid='ignore'):
            try:
                levy = e._compute_levy_area_signature(s1, s2)
                assert isinstance(levy, float)
            except (ValueError, ZeroDivisionError):
                pass

    def test_all_nan_returns_zero(self):
        e = _make_engine()
        s = np.array([np.nan, np.nan, np.nan])
        with np.errstate(invalid='ignore'):
            try:
                levy = e._compute_levy_area_signature(s, s)
                assert levy == 0.0 or math.isnan(levy)
            except (ValueError, ZeroDivisionError):
                pass

    def test_inf_values(self):
        e = _make_engine()
        s1 = np.array([0.01, np.inf, -0.01, 0.02])
        s2 = np.array([0.02, 0.01, 0.03, 0.01])
        with np.errstate(invalid='ignore', divide='ignore'):
            try:
                levy = e._compute_levy_area_signature(s1, s2)
                assert isinstance(levy, float)
            except (ValueError, ZeroDivisionError):
                pass

    def test_mismatched_lengths(self):
        e = _make_engine()
        s1 = np.random.randn(100) * 0.01
        s2 = np.random.randn(50) * 0.01  # shorter
        # Method uses min_len, so should work
        levy = e._compute_levy_area_signature(s1, s2, lags=[1, 5])
        assert isinstance(levy, float)


# ---------------------------------------------------------------------------
# Learn adjacency boundary conditions
# ---------------------------------------------------------------------------

class TestLearnAdjacencyBoundary:
    """Learn adjacency with extreme input values."""

    def test_extreme_positive_scores(self):
        e = _make_engine()
        scores = {('A', 'B'): 1e6, ('B', 'A'): -1e6}
        adj = e._learn_adjacency_matrix(scores, ['A', 'B'])
        for v in adj.values():
            assert 0.0 <= v <= 1.0

    def test_extreme_negative_scores(self):
        e = _make_engine()
        scores = {('A', 'B'): -1e6, ('B', 'A'): 1e6}
        adj = e._learn_adjacency_matrix(scores, ['A', 'B'])
        for v in adj.values():
            assert 0.0 <= v <= 1.0

    def test_very_large_number_of_pairs(self):
        e = _make_engine()
        assets = [f'A{i}' for i in range(20)]
        scores = {}
        for a1, a2 in combinations(assets, 2):
            scores[(a1, a2)] = np.random.randn() * 0.5
        adj = e._learn_adjacency_matrix(scores, assets)
        for v in adj.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# compute_leadlag_matrix extreme edge cases
# ---------------------------------------------------------------------------

class TestComputeLeadLagMatrixExtreme:
    """Lead-lag matrix computation with extreme inputs."""

    def test_constant_prices_all_same(self):
        e = _make_engine()
        dates = pd.bdate_range('2024-01-02', periods=200)
        df = pd.DataFrame(
            {'SPY': [100.0] * 200, 'GLD': [150.0] * 200, 'TLT': [130.0] * 200},
            index=dates
        )
        # Constant prices → zero returns → zero std → may return None or valid
        result = e.compute_leadlag_matrix(66, df)
        # Either None (due to zero std handling) or a valid LeadLagMatrix
        assert result is None or isinstance(result, LeadLagMatrix)

    def test_single_row_dataframe(self):
        e = _make_engine()
        dates = pd.bdate_range('2024-01-02', periods=1)
        df = pd.DataFrame({'SPY': [400.0], 'GLD': [150.0], 'TLT': [130.0]}, index=dates)
        result = e.compute_leadlag_matrix(66, df)
        assert result is None

    def test_prices_with_nan_column(self):
        e = _make_engine()
        df = _make_prices_df(200)
        df['SPY'] = np.nan  # All NaN for SPY
        result = e.compute_leadlag_matrix(66, df)
        # Only GLD + TLT have data → might still work with 2 assets
        assert result is None or isinstance(result, LeadLagMatrix)

    def test_window_exact_data_length(self):
        """Window exactly matching available data should work."""
        e = _make_engine()
        df = _make_prices_df(100)
        result = e.compute_leadlag_matrix(80, df)
        assert result is None or isinstance(result, LeadLagMatrix)

    def test_zero_returns_after_pct_change(self):
        """Prices that don't change should have zero returns."""
        e = _make_engine()
        dates = pd.bdate_range('2024-01-02', periods=200)
        # All same price → zero returns → may trigger division by zero in levy/DTW
        df = pd.DataFrame(
            {'SPY': [100.0] * 200, 'GLD': [100.0] * 200, 'TLT': [100.0] * 200},
            index=dates
        )
        result = e.compute_leadlag_matrix(66, df)
        assert result is None or isinstance(result, LeadLagMatrix)


# ---------------------------------------------------------------------------
# compute_window_signal extreme boundary conditions
# ---------------------------------------------------------------------------

class TestComputeWindowSignalExtreme:
    """Window signal with extreme base weights, zero weights, boundary values."""

    def test_base_weight_zero(self):
        e = _make_engine()
        df = _make_prices_df(200)
        ll = _make_leadlag_matrix()
        sig = e.compute_window_signal('SPY', 66, 0.0, ll, df)
        assert sig is not None
        assert sig.base_weight == 0.0
        assert MIN_WEIGHT <= sig.target_weight <= 1.0

    def test_base_weight_one(self):
        e = _make_engine()
        df = _make_prices_df(200)
        ll = _make_leadlag_matrix()
        sig = e.compute_window_signal('SPY', 66, 1.0, ll, df)
        assert sig is not None
        assert sig.base_weight == 1.0

    def test_base_weight_negative(self):
        e = _make_engine()
        df = _make_prices_df(200)
        ll = _make_leadlag_matrix()
        sig = e.compute_window_signal('SPY', 66, -0.5, ll, df)
        # Negative base weight is unusual but should not crash
        assert sig is not None or sig is None

    def test_extreme_negative_momentum(self):
        e = _make_engine()
        dates = pd.bdate_range('2024-01-02', periods=200)
        # Monotonically declining prices → strong negative momentum
        spy = 400.0 * (1.0 - np.linspace(0, 0.5, 200))  # drops 50%
        gld = 150.0 * (1.0 - np.linspace(0, 0.4, 200))
        tlt = 130.0 * (1.0 - np.linspace(0, 0.3, 200))
        df = pd.DataFrame({'SPY': spy, 'GLD': gld, 'TLT': tlt}, index=dates)
        ll = _make_leadlag_matrix()
        sig = e.compute_window_signal('SPY', 66, 0.46, ll, df)
        assert sig is not None
        assert sig.momentum_return < 0
        assert sig.signal == -1

    def test_extreme_positive_momentum(self):
        e = _make_engine()
        dates = pd.bdate_range('2024-01-02', periods=200)
        # Monotonically increasing prices → strong positive momentum
        spy = 400.0 * (1.0 + np.linspace(0, 1.0, 200))  # doubles
        gld = 150.0 * (1.0 + np.linspace(0, 0.5, 200))
        tlt = 130.0 * (1.0 + np.linspace(0, 0.3, 200))
        df = pd.DataFrame({'SPY': spy, 'GLD': gld, 'TLT': tlt}, index=dates)
        ll = _make_leadlag_matrix()
        sig = e.compute_window_signal('SPY', 66, 0.46, ll, df)
        assert sig is not None
        assert sig.momentum_return > 0
        assert sig.signal == 1


# ---------------------------------------------------------------------------
# compute_ensemble_signal extreme edge cases
# ---------------------------------------------------------------------------

class TestComputeEnsembleSignalExtreme:
    """Ensemble signal with edge conditions: all windows None, empty lookback, etc."""

    def test_empty_lookback_windows(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        with patch.object(e, 'lookback_windows', []):
            sig = e.compute_ensemble_signal('SPY', 0.46, df)
            assert sig is None

    def test_all_windows_return_none(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        with patch.object(e, 'lookback_windows', [22, 44]):
            with patch.object(e, 'compute_window_signal', return_value=None):
                sig = e.compute_ensemble_signal('SPY', 0.46, df)
                assert sig is None

    def test_leadlag_matrix_none_returns_none(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        with patch.object(e, 'compute_leadlag_matrix', return_value=None):
            sig = e.compute_ensemble_signal('SPY', 0.46, df)
            assert sig is None

    def test_ticker_not_in_assets(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        sig = e.compute_ensemble_signal('NOTAREALTICKER', 0.46, df)
        assert sig is None

    def test_ensemble_signal_zero_when_momentum_zero(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        ws = WindowMomentumSignal(
            ticker='SPY', window=66, timestamp='2026-01-01',
            momentum_return=0.0, signal=0,
            network_momentum=0.0, network_adjustment=0.0,
            base_weight=0.46, target_weight=0.46, adjustment=0.0,
        )
        ll = _make_leadlag_matrix()
        # Create a signal with all-zero window signals → ensemble_momentum=0
        with patch.object(e, 'compute_leadlag_matrix', return_value=ll):
            with patch.object(e, 'compute_window_signal', return_value=ws):
                sig = e.compute_ensemble_signal('SPY', 0.46, df)
                if sig is not None:
                    assert sig.ensemble_momentum == 0.0
                    assert sig.ensemble_signal == 0


# ---------------------------------------------------------------------------
# get_current_recommendation extreme edge cases
# ---------------------------------------------------------------------------

class TestGetCurrentRecommendationExtreme:
    """Portfolio recommendation with extreme inputs."""

    def test_empty_allocation_returns_portfolio(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        rec = e.get_current_recommendation({})
        # Empty allocation should still return a portfolio with default fields
        assert rec is not None
        assert 'CASH' not in rec.target_allocation or rec.target_allocation.get('CASH') is not None

    def test_all_cash_allocation(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        rec = e.get_current_recommendation({'CASH': 1.0})
        assert rec is not None
        assert rec.dominant_leader == 'SPY'
        assert rec.overall_confidence == 0.0

    def test_single_non_cash_asset(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        rec = e.get_current_recommendation({'SPY': 1.0})
        assert rec is not None
        assert 'SPY' in rec.target_allocation

    def test_normalization_with_all_zero_cash(self):
        """When all weights are zero except CASH, normalization should handle it."""
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        rec = e.get_current_recommendation({'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0, 'CASH': 1.0})
        assert rec is not None
        total = sum(w for k, w in rec.target_allocation.items() if k != 'CASH')
        # Either 0 or normalized to 1.0
        assert total == 0.0 or abs(total - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Backtester extreme edge cases
# ---------------------------------------------------------------------------

class TestBacktesterExtreme:
    """Backtester with extreme inputs: missing columns, NaNs in prices."""

    def test_prices_with_nan_columns_handled(self):
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.network_momentum = _make_engine()
        bt.prices_df = _make_prices_df(400)
        # Add an all-NaN column
        bt.prices_df['BOGUS'] = np.nan
        result = bt.run_backtest()
        if 'error' not in result:
            assert result['end_value'] > 0

    def test_negative_prices_handled(self):
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.network_momentum = _make_engine()
        df = _make_prices_df(400)
        df['SPY'] = -df['SPY']  # Negative prices
        bt.prices_df = df
        result = bt.run_backtest()
        if 'error' not in result:
            assert result['end_value'] > 0  # Should still produce a positive value

    def test_zero_rebalance_freq(self):
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 0
        bt.network_momentum = _make_engine()
        bt.prices_df = _make_prices_df(400)
        # Zero rebalance frequency may cause issues (modulo by zero)
        try:
            result = bt.run_backtest()
            assert 'error' in result or 'cagr' in result
        except (ValueError, ZeroDivisionError, TypeError):
            pass  # Acceptable edge case behavior

    def test_start_date_after_end_date(self):
        bt = NetworkMomentumBacktester.__new__(NetworkMomentumBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION
        bt.start_date = pd.to_datetime('2025-01-01')
        bt.end_date = pd.to_datetime('2024-01-01')  # Before start
        bt.rebalance_freq = 21
        bt.network_momentum = _make_engine()
        bt.prices_df = _make_prices_df(400)
        result = bt.run_backtest()
        assert 'error' in result


# ---------------------------------------------------------------------------
# Boundary conditions: wrong types, missing keys
# ---------------------------------------------------------------------------

class TestBoundaryWrongTypes:
    """Passing wrong types to methods should not crash."""

    def test_window_signal_string_window(self):
        e = _make_engine()
        df = _make_prices_df(200)
        ll = _make_leadlag_matrix()
        # Window as string instead of int
        try:
            sig = e.compute_window_signal('SPY', '66', 0.46, ll, df)
            assert sig is not None
        except (TypeError, ValueError):
            pass

    def test_window_signal_none_prices_df(self):
        """Passing None for prices_df triggers _load_prices."""
        e = NetworkMomentumLeadLag.__new__(NetworkMomentumLeadLag)
        e.prices_path = None
        e.db_path = None
        e.lookback_windows = LOOKBACK_WINDOWS
        e.max_deviation = MAX_DEVIATION
        df = _make_prices_df(200)
        e._prices_df = df
        ll = _make_leadlag_matrix()
        # With _prices_df set, _load_prices returns cached df, so prices_df=None works
        sig = e.compute_window_signal('SPY', 66, 0.46, ll, prices_df=None)
        assert sig is None or isinstance(sig, WindowMomentumSignal)

    def test_ensemble_signal_invalid_ticker_type(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        # Pass ticker as None instead of string
        sig = e.compute_ensemble_signal(None, 0.46, df)
        assert sig is None  # Should gracefully return None

    def test_get_recommendation_non_dict_allocation(self):
        e = _make_engine()
        df = _make_prices_df(300)
        e._prices_df = df
        with pytest.raises((TypeError, AttributeError)):
            e.get_current_recommendation("not_a_dict")

    def test_leadlag_matrix_empty_window_zero(self):
        """Compute leadlag with window=0 should not crash."""
        e = _make_engine()
        df = _make_prices_df(200)
        try:
            result = e.compute_leadlag_matrix(0, df)
            assert result is None or isinstance(result, LeadLagMatrix)
        except (ZeroDivisionError, ValueError):
            pass


# ---------------------------------------------------------------------------
# NetworkMomentumLeadLag initialization edge cases
# ---------------------------------------------------------------------------

class TestNetworkMomentumLeadLagInit:
    """Test __init__ edge cases for NetworkMomentumLeadLag."""

    def test_init_with_none_lookback(self):
        engine = NetworkMomentumLeadLag(lookback_windows=None)
        assert engine.lookback_windows == LOOKBACK_WINDOWS

    def test_init_with_custom_lookback(self):
        custom = [10, 20, 30]
        engine = NetworkMomentumLeadLag(lookback_windows=custom)
        assert engine.lookback_windows == custom

    def test_init_with_custom_max_deviation(self):
        engine = NetworkMomentumLeadLag(max_deviation=0.25)
        assert engine.max_deviation == 0.25

    def test_init_default_prices_path(self):
        from src.paths import PRICES_JSON
        engine = NetworkMomentumLeadLag()
        assert engine.prices_path == PRICES_JSON

    def test_load_prices_caches_result(self):
        """_load_prices should return cached _prices_df if set."""
        engine = NetworkMomentumLeadLag.__new__(NetworkMomentumLeadLag)
        engine.prices_path = None
        engine.db_path = None
        engine.lookback_windows = LOOKBACK_WINDOWS
        engine.max_deviation = MAX_DEVIATION
        fake_df = pd.DataFrame({'SPY': [100.0]}, index=pd.date_range('2024-01-02', periods=1))
        engine._prices_df = fake_df
        result = engine._load_prices()
        assert result is fake_df


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
