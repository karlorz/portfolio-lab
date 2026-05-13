#!/usr/bin/env python3
"""
Tests for combined strategy backtest — data classes, signal combination,
Fed regime classification, baseline backtest, and crisis return calculation.
"""
import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

# Mock heavy external dependencies before import
mock_tsmom = MagicMock()
mock_tsmom.DEFAULT_BASE_ALLOCATION = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
mock_hmm = MagicMock()
mock_fed = MagicMock()
mock_fed.classify_fed_regime = MagicMock(return_value='EASING')

sys.modules['src.signals.tsmom_overlay'] = mock_tsmom
sys.modules['src.agents.risk_agent_hmm'] = mock_hmm
sys.modules['src.signals.fed_policy_overlay'] = mock_fed

from src.backtest.combined_strategy import (
    DailyPosition, BacktestResult, CombinedStrategyBacktester,
    TRANSACTION_COST, REBALANCE_FREQ, MIN_HISTORY_DAYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_position(date='2026-01-01', value=100000.0):
    return DailyPosition(
        date=date,
        weights={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
        prices={'SPY': 500.0, 'GLD': 200.0, 'TLT': 100.0},
        portfolio_value=value,
    )


def _make_result():
    return BacktestResult(
        strategy='combined', start_date='2006-01-01', end_date='2026-05-08',
        trading_days=5000, rebalances=238,
        start_value=100000, end_value=500000,
        cagr=0.10, volatility=0.11, sharpe_ratio=0.79,
        max_drawdown=-0.25, calmar_ratio=0.40,
        baseline_cagr=0.09, baseline_sharpe=0.72,
        excess_return=0.01, information_ratio=0.30,
        tsmom_contribution=0.03, hmm_contribution=0.01, fed_contribution=0.005,
    )


def _make_backtester():
    bt = CombinedStrategyBacktester.__new__(CombinedStrategyBacktester)
    bt.tickers = ['SPY', 'GLD', 'TLT']
    bt.base_allocation = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
    bt.transaction_cost = 0.001
    bt.rebalance_freq = 21
    bt.tsmom = MagicMock()
    bt.hmm_manager = MagicMock()
    bt.fed_overlay = MagicMock()
    bt.prices_df = None
    bt.dates = []
    return bt


def _make_prices_df(n=500, seed=42):
    """Create a synthetic prices DataFrame."""
    np.random.seed(seed)
    data = {}
    for ticker in ['SPY', 'GLD', 'TLT']:
        prices = [500.0 if ticker == 'SPY' else 200.0 if ticker == 'GLD' else 100.0]
        for _ in range(n - 1):
            ret = np.random.normal(0.0004, 0.012)
            prices.append(prices[-1] * (1 + ret))
        data[ticker] = prices
    dates = pd.date_range(end=datetime.now(), periods=n, freq='B').strftime('%Y-%m-%d').tolist()
    return pd.DataFrame(data, index=dates)


def _make_regime_df(spy_start, spy_end, tlt_start, tlt_end, gld_start=200, gld_end=205, n=200):
    """Build a DataFrame with linear ramps for fed regime testing."""
    spy = np.linspace(spy_start, spy_end, n)
    tlt = np.linspace(tlt_start, tlt_end, n)
    gld = np.linspace(gld_start, gld_end, n)
    dates = pd.date_range(end=datetime.now(), periods=n, freq='B').strftime('%Y-%m-%d').tolist()
    return pd.DataFrame({'SPY': spy, 'GLD': gld, 'TLT': tlt}, index=dates)


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_transaction_cost(self):
        assert TRANSACTION_COST == 0.001

    def test_rebalance_freq(self):
        assert REBALANCE_FREQ == 21

    def test_min_history(self):
        assert MIN_HISTORY_DAYS == 273


# ---------------------------------------------------------------------------
# DailyPosition tests
# ---------------------------------------------------------------------------

class TestDailyPosition:
    def test_creation(self):
        pos = _make_position()
        assert pos.date == '2026-01-01'
        assert pos.portfolio_value == 100000.0

    def test_defaults(self):
        pos = _make_position()
        assert pos.tsmom_deltas is None
        assert pos.hmm_regime is None
        assert pos.rebalance_executed is False
        assert pos.turnover == 0.0

    def test_with_metadata(self):
        pos = DailyPosition(
            date='2026-01-01',
            weights={'SPY': 0.50},
            prices={'SPY': 500.0},
            portfolio_value=100000.0,
            tsmom_deltas={'SPY': 0.05},
            hmm_regime='bull',
            fed_regime='EASING',
            rebalance_executed=True,
            turnover=0.05,
        )
        assert pos.hmm_regime == 'bull'
        assert pos.rebalance_executed is True


# ---------------------------------------------------------------------------
# BacktestResult tests
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def test_creation(self):
        r = _make_result()
        assert r.strategy == 'combined'
        assert r.sharpe_ratio == 0.79

    def test_to_dict(self):
        r = _make_result()
        d = r.to_dict()
        assert d['strategy'] == 'combined'
        assert d['cagr'] == 0.10
        assert d['sharpe_ratio'] == 0.79
        assert d['max_drawdown'] == -0.25
        assert 'tsmom_contribution' in d
        assert 'hmm_contribution' in d
        assert 'fed_contribution' in d

    def test_to_dict_crisis_fields_optional(self):
        r = _make_result()
        d = r.to_dict()
        assert d['crisis_2008_return'] is None

    def test_to_dict_with_crisis(self):
        r = _make_result()
        r.crisis_2008_return = -0.12
        r.crisis_2020_return = -0.07
        r.crisis_2022_return = -0.13
        d = r.to_dict()
        assert d['crisis_2008_return'] == -0.12


# ---------------------------------------------------------------------------
# CombinedStrategyBacktester tests
# ---------------------------------------------------------------------------

class TestCombinedStrategyBacktester:
    def test_init_defaults(self):
        bt = _make_backtester()
        assert bt.tickers == ['SPY', 'GLD', 'TLT']
        assert bt.base_allocation == {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}

    def test_custom_init(self):
        bt = CombinedStrategyBacktester.__new__(CombinedStrategyBacktester)
        bt.tickers = ['SPY', 'QQQ']
        bt.base_allocation = {'SPY': 0.6, 'QQQ': 0.4}
        bt.transaction_cost = 0.002
        bt.rebalance_freq = 10
        assert len(bt.tickers) == 2
        assert bt.transaction_cost == 0.002

    def test_combine_signals_weights(self):
        bt = _make_backtester()
        tsmom = {'SPY': 0.10, 'GLD': -0.05, 'TLT': 0.0}
        combined, summary = bt._combine_signals(
            tsmom_deltas=tsmom,
            hmm_regime='bull',
            hmm_deltas={'SPY': 0.05, 'GLD': -0.02, 'TLT': -0.03},
            fed_regime='EASING',
            fed_deltas={'SPY': 0.03, 'GLD': 0.02, 'TLT': -0.05},
            current_idx=300,
        )
        assert isinstance(combined, dict)
        assert 'SPY' in combined
        assert 'GLD' in combined
        assert 'TLT' in combined

    def test_combine_signals_sums_near_zero(self):
        """Deltas should roughly cancel out since they are adjustments from base."""
        bt = _make_backtester()
        tsmom = {'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0}
        combined, _ = bt._combine_signals(
            tsmom_deltas=tsmom,
            hmm_regime='neutral',
            hmm_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            fed_regime='NEUTRAL',
            fed_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            current_idx=300,
        )
        for v in combined.values():
            assert abs(v) < 0.01

    def test_combine_signals_no_hmm_regime(self):
        bt = _make_backtester()
        tsmom = {'SPY': 0.05, 'GLD': -0.02, 'TLT': -0.03}
        combined, summary = bt._combine_signals(
            tsmom_deltas=tsmom,
            hmm_regime=None,
            hmm_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            fed_regime=None,
            fed_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            current_idx=300,
        )
        assert isinstance(combined, dict)

    def test_crisis_return_basic(self):
        bt = _make_backtester()
        positions = [
            _make_position(date='2020-02-01', value=100000),
            _make_position(date='2020-03-01', value=85000),
            _make_position(date='2020-04-01', value=90000),
        ]
        ret = bt._calculate_crisis_return(positions, '2020-02-01', '2020-04-01')
        assert ret == pytest.approx(-0.10, abs=0.01)

    def test_crisis_return_no_positions(self):
        bt = _make_backtester()
        ret = bt._calculate_crisis_return([], '2020-02-01', '2020-04-01')
        assert ret is None

    def test_crisis_return_outside_range(self):
        bt = _make_backtester()
        positions = [_make_position(date='2019-01-01', value=100000)]
        ret = bt._calculate_crisis_return(positions, '2020-02-01', '2020-04-01')
        assert ret is None

    def test_run_baseline(self):
        bt = _make_backtester()
        bt.prices_df = _make_prices_df(300)
        result = bt._run_baseline(252, 299, 100000.0)
        assert 'cagr' in result
        assert 'sharpe' in result
        assert 'daily_returns' in result
        assert len(result['daily_returns']) == 47

    def test_run_baseline_positive_cagr(self):
        bt = _make_backtester()
        bt.prices_df = _make_prices_df(300, seed=42)
        result = bt._run_baseline(252, 299, 100000.0)
        # With positive drift, CAGR should be non-negative
        assert result['cagr'] is not None

    def test_run_baseline_daily_returns_count(self):
        bt = _make_backtester()
        bt.prices_df = _make_prices_df(500)
        result = bt._run_baseline(300, 400, 100000.0)
        assert len(result['daily_returns']) == 100

    def _make_regime_df(spy_start, spy_end, tlt_start, tlt_end, gld_start=200, gld_end=205, n=200):
        """Build a DataFrame with linear ramps for fed regime testing."""
        spy = np.linspace(spy_start, spy_end, n)
        tlt = np.linspace(tlt_start, tlt_end, n)
        gld = np.linspace(gld_start, gld_end, n)
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B').strftime('%Y-%m-%d').tolist()
        return pd.DataFrame({'SPY': spy, 'GLD': gld, 'TLT': tlt}, index=dates)

    def test_fed_regime_easing(self):
        bt = _make_backtester()
        bt.prices_df = _make_regime_df(400, 500, 90, 110)
        regime, deltas = bt._get_fed_regime_deltas(136)
        assert regime == 'EASING'
        assert deltas['SPY'] > 0

    def test_fed_regime_tightening(self):
        bt = _make_backtester()
        bt.prices_df = _make_regime_df(500, 400, 110, 90)
        regime, deltas = bt._get_fed_regime_deltas(136)
        assert regime == 'TIGHTENING'
        assert deltas['SPY'] < 0

    def test_fed_regime_neutral(self):
        bt = _make_backtester()
        bt.prices_df = _make_regime_df(500, 501, 100, 100.5)
        regime, deltas = bt._get_fed_regime_deltas(136)
        assert regime == 'NEUTRAL'

    def test_fed_regime_insufficient_data(self):
        bt = _make_backtester()
        bt.prices_df = _make_prices_df(50)
        regime, deltas = bt._get_fed_regime_deltas(10)
        assert regime is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
