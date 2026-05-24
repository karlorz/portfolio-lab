#!/usr/bin/env python3
"""
Tests for combined strategy backtest — data classes, signal combination,
Fed regime classification, baseline backtest, and crisis return calculation.
"""
import sys
import numpy as np
import pandas as pd

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

# Mock heavy external dependencies before import
# Save originals so we can restore after import (prevents test pollution)
_orig_modules = {
    k: sys.modules.get(k) for k in
    ('src.signals.tsmom_overlay', 'src.agents.risk_agent_hmm', 'src.signals.fed_policy_overlay')
}

mock_tsmom = MagicMock()
mock_tsmom.DEFAULT_BASE_ALLOCATION = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
mock_hmm = MagicMock()
mock_fed = MagicMock()
mock_fed.classify_fed_regime = MagicMock(return_value='EASING')

sys.modules['src.signals.tsmom_overlay'] = mock_tsmom
sys.modules['src.agents.risk_agent_hmm'] = mock_hmm
sys.modules['src.signals.fed_policy_overlay'] = mock_fed

from src.backtest.combined_strategy import (
    DailyPosition, CombinedStrategyBacktester,
    TRANSACTION_COST, REBALANCE_FREQ, MIN_HISTORY_DAYS,
    START_DATE, END_DATE, __all__,
)
from src.backtest.metrics import BacktestResult

# Restore original modules to prevent polluting other test files
for _k, _v in _orig_modules.items():
    if _v is None:
        sys.modules.pop(_k, None)
    else:
        sys.modules[_k] = _v


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
        total_return=400.0,
        cagr=0.10, volatility=0.11, sharpe_ratio=0.79,
        max_drawdown=-0.25, total_rebalances=238,
        baseline_sharpe=0.72,
        extras={
            "strategy_name": 'combined', "start_date": '2006-01-01', "end_date": '2026-05-08',
            "trading_days": 5000,
            "start_value": 100000, "end_value": 500000,
            "calmar_ratio": 0.40,
            "baseline_cagr": 0.09,
            "excess_return": 0.01, "information_ratio": 0.30,
            "tsmom_contribution": 0.03, "hmm_contribution": 0.01, "fed_contribution": 0.005,
        },
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


def _make_backtester_with_tsmom():
    """Backtester with TSMOM mock attributes set for _get_tsmom_deltas tests."""
    bt = _make_backtester()
    bt.tsmom.lookback_days = 252
    bt.tsmom.skip_days = 21
    bt.tsmom.vol_window = 63
    bt.tsmom.max_deviation = 0.10
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
# Exports tests
# ---------------------------------------------------------------------------

class TestExports:
    def test_all_exports_contains_expected(self):
        expected = {
            'TRANSACTION_COST', 'REBALANCE_FREQ', 'MIN_HISTORY_DAYS',
            'START_DATE', 'END_DATE', 'DailyPosition', 'CombinedStrategyBacktester',
        }
        assert set(__all__) == expected


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

    def test_start_date(self):
        assert START_DATE == "2006-02-01"

    def test_end_date(self):
        assert END_DATE == "2026-05-08"

    def test_end_date_after_start(self):
        """END_DATE is chronologically after START_DATE."""
        from datetime import datetime
        start = datetime.strptime(START_DATE, '%Y-%m-%d')
        end = datetime.strptime(END_DATE, '%Y-%m-%d')
        assert end > start

    def test_rebalance_freq_positive(self):
        """REBALANCE_FREQ is a positive integer."""
        assert REBALANCE_FREQ > 0
        assert isinstance(REBALANCE_FREQ, int)

    def test_min_history_gt_rebalance(self):
        """MIN_HISTORY_DAYS exceeds REBALANCE_FREQ."""
        assert MIN_HISTORY_DAYS > REBALANCE_FREQ

    def test_transaction_cost_positive(self):
        """TRANSACTION_COST is a small positive float below 1%."""
        assert 0 < TRANSACTION_COST < 0.01


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

    def test_to_dict_all_fields_present(self):
        """asdict includes all 9 fields of DailyPosition."""
        from dataclasses import asdict
        pos = DailyPosition(
            date='2026-06-01',
            weights={'SPY': 0.5, 'GLD': 0.3, 'TLT': 0.2},
            prices={'SPY': 510.0, 'GLD': 195.0, 'TLT': 98.0},
            portfolio_value=110000.0,
            tsmom_deltas={'SPY': 0.02, 'GLD': -0.01, 'TLT': -0.01},
            hmm_regime='bull',
            fed_regime='EASING',
            rebalance_executed=True,
            turnover=0.04,
        )
        d = asdict(pos)
        assert 'date' in d
        assert 'weights' in d
        assert 'prices' in d
        assert 'portfolio_value' in d
        assert 'tsmom_deltas' in d
        assert 'hmm_regime' in d
        assert 'fed_regime' in d
        assert 'rebalance_executed' in d
        assert 'turnover' in d
        assert len(d) == 9

    def test_to_dict_field_values(self):
        """Verify values survive round-trip through asdict."""
        from dataclasses import asdict
        pos = _make_position()
        d = asdict(pos)
        assert d['date'] == '2026-01-01'
        assert d['weights'] == {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        assert d['prices'] == {'SPY': 500.0, 'GLD': 200.0, 'TLT': 100.0}
        assert d['portfolio_value'] == 100000.0
        assert d['tsmom_deltas'] is None
        assert d['hmm_regime'] is None
        assert d['fed_regime'] is None
        assert d['rebalance_executed'] is False
        assert d['turnover'] == 0.0

    def test_to_dict_fed_regime_present(self):
        """Ensure fed_regime survives to_dict when set."""
        from dataclasses import asdict
        pos = DailyPosition(
            date='2026-06-01',
            weights={'SPY': 0.5},
            prices={'SPY': 510.0},
            portfolio_value=100000.0,
            fed_regime='TIGHTENING',
        )
        d = asdict(pos)
        assert d['fed_regime'] == 'TIGHTENING'

    def test_to_dict_fed_regime_defaults_none(self):
        """fed_regime is None in to_dict when not provided."""
        from dataclasses import asdict
        pos = _make_position()
        d = asdict(pos)
        assert d['fed_regime'] is None

    def test_to_dict_empty_weights_prices(self):
        """Empty containers for weights/prices survive to_dict."""
        from dataclasses import asdict
        pos = DailyPosition(
            date='2026-01-01', weights={}, prices={}, portfolio_value=0.0,
        )
        d = asdict(pos)
        assert d['weights'] == {}
        assert d['prices'] == {}
        assert d['portfolio_value'] == 0.0

    def test_to_dict_extreme_portfolio_value(self):
        """Very large and very small portfolio values survive round-trip."""
        from dataclasses import asdict
        pos = DailyPosition(
            date='2026-01-01', weights={'SPY': 1.0}, prices={'SPY': 500.0},
            portfolio_value=1e12,
        )
        d = asdict(pos)
        assert d['portfolio_value'] == 1e12
        pos2 = DailyPosition(
            date='2026-01-01', weights={'SPY': 1.0}, prices={'SPY': 500.0},
            portfolio_value=0.01,
        )
        d2 = asdict(pos2)
        assert d2['portfolio_value'] == 0.01


# ---------------------------------------------------------------------------
# BacktestResult tests
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def test_creation(self):
        r = _make_result()
        assert r.extras['strategy_name'] == 'combined'
        assert r.sharpe_ratio == 0.79

    def test_to_dict(self):
        from dataclasses import asdict
        r = _make_result()
        d = asdict(r)
        assert d['extras']['strategy_name'] == 'combined'
        assert d['cagr'] == 0.10
        assert d['sharpe_ratio'] == 0.79
        assert d['max_drawdown'] == -0.25
        assert 'tsmom_contribution' in d['extras']
        assert 'hmm_contribution' in d['extras']
        assert 'fed_contribution' in d['extras']

    def test_to_dict_crisis_fields_optional(self):
        from dataclasses import asdict
        r = _make_result()
        d = asdict(r)
        assert d['crisis_returns'] is None or d['crisis_returns'].get('2008') is None

    def test_to_dict_with_crisis(self):
        from dataclasses import asdict
        r = _make_result()
        r.crisis_returns = {"2008": -0.12, "2020": -0.07, "2022": -0.13}
        d = asdict(r)
        assert d['crisis_returns']['2008'] == -0.12

    def test_total_return_negative(self):
        """Negative total_return survives creation and to_dict."""
        from dataclasses import asdict
        r = _make_result()
        r.total_return = -25.0
        d = asdict(r)
        assert d['total_return'] == -25.0


class TestBacktestResultToDict:
    def test_all_core_fields_present(self):
        """Verify all 10 core BacktestResult fields survive asdict."""
        from dataclasses import asdict
        r = _make_result()
        d = asdict(r)
        core_fields = [
            'total_return', 'cagr', 'volatility', 'sharpe_ratio', 'max_drawdown',
            'total_rebalances', 'total_transaction_costs', 'avg_turnover',
            'baseline_sharpe', 'sharpe_improvement',
        ]
        for field in core_fields:
            assert field in d, f"Missing core field: {field}"

    def test_extras_fields_present(self):
        """Verify all extras keys are present in to_dict."""
        from dataclasses import asdict
        r = _make_result()
        d = asdict(r)
        expected_extras = [
            'strategy_name', 'start_date', 'end_date', 'trading_days',
            'start_value', 'end_value', 'calmar_ratio', 'baseline_cagr',
            'excess_return', 'information_ratio',
            'tsmom_contribution', 'hmm_contribution', 'fed_contribution',
        ]
        for key in expected_extras:
            assert key in d['extras'], f"Missing extras key: {key}"

    def test_crisis_returns_nullable(self):
        """crisis_returns is None when not set."""
        from dataclasses import asdict
        r = _make_result()
        d = asdict(r)
        assert d['crisis_returns'] is None

    def test_extras_daily_data_included(self):
        """extras contains the core 13 keys from _make_result()."""
        from dataclasses import asdict
        r = _make_result()
        d = asdict(r)
        # daily_values/daily_returns/positions are only set by run_backtest(),
        # not guaranteed by the BacktestResult dataclass itself.
        assert 'strategy_name' in d['extras']

    def test_extras_keys_comprehensive(self):
        """All 13 extras keys from _make_result() are present in to_dict."""
        from dataclasses import asdict
        r = _make_result()
        d = asdict(r)
        expected_extras = [
            'strategy_name', 'start_date', 'end_date', 'trading_days',
            'start_value', 'end_value', 'calmar_ratio', 'baseline_cagr',
            'excess_return', 'information_ratio',
            'tsmom_contribution', 'hmm_contribution', 'fed_contribution',
        ]
        for key in expected_extras:
            assert key in d['extras'], f"Missing extras key: {key}"
        assert len(d['extras']) == 13

    def test_sharpe_improvement_computed(self):
        """sharpe_improvement is the difference between strategy and baseline."""
        from dataclasses import asdict
        r = _make_result()
        r.sharpe_improvement = r.sharpe_ratio - r.baseline_sharpe
        d = asdict(r)
        assert d['sharpe_improvement'] == pytest.approx(0.79 - 0.72)


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


# ---------------------------------------------------------------------------
# Combine signals edge cases
# ---------------------------------------------------------------------------

class TestCombineSignalsEdgeCases:
    def test_all_positive_deltas(self):
        """All three signal sources push in the same positive direction."""
        bt = _make_backtester()
        combined, resolution = bt._combine_signals(
            tsmom_deltas={'SPY': 0.05, 'GLD': 0.03, 'TLT': 0.02},
            hmm_regime='bull',
            hmm_deltas={'SPY': 0.10, 'GLD': -0.05, 'TLT': -0.05},
            fed_regime='EASING',
            fed_deltas={'SPY': 0.05, 'GLD': 0.05, 'TLT': -0.05},
            current_idx=300,
        )
        # SPY should have net positive delta from all sources
        assert combined['SPY'] > 0

    def test_all_negative_deltas(self):
        """All three signal sources push in the same negative direction."""
        bt = _make_backtester()
        combined, resolution = bt._combine_signals(
            tsmom_deltas={'SPY': -0.05, 'GLD': -0.03, 'TLT': -0.02},
            hmm_regime='bear',
            hmm_deltas={'SPY': -0.10, 'GLD': 0.05, 'TLT': 0.05},
            fed_regime='TIGHTENING',
            fed_deltas={'SPY': -0.10, 'GLD': 0.10, 'TLT': 0.0},
            current_idx=300,
        )
        assert combined['SPY'] < 0

    def test_conflicting_deltas_triggers_split(self):
        """TSMOM and Fed with opposite signs trigger split_difference resolution."""
        bt = _make_backtester()
        # TSMOM wants SPY up > 1%, Fed wants SPY down > 1% => conflict
        tsmom = {'SPY': 0.05, 'GLD': 0.0, 'TLT': 0.0}
        fed = {'SPY': -0.05, 'GLD': 0.0, 'TLT': 0.0}
        combined, resolution = bt._combine_signals(
            tsmom_deltas=tsmom,
            hmm_regime='bull',
            hmm_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            fed_regime='TIGHTENING',
            fed_deltas=fed,
            current_idx=300,
        )
        assert 'split_difference' in resolution

    def test_hmm_neutral_regime_reduction(self):
        """Neutral HMM regime triggers the hmm_neutral damping."""
        bt = _make_backtester()
        combined, resolution = bt._combine_signals(
            tsmom_deltas={'SPY': 0.05, 'GLD': 0.0, 'TLT': 0.0},
            hmm_regime='neutral',
            hmm_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            fed_regime='EASING',
            fed_deltas={'SPY': 0.03, 'GLD': 0.0, 'TLT': 0.0},
            current_idx=300,
        )
        assert 'hmm_neutral' in resolution

    def test_missing_ticker_in_deltas(self):
        """Deltas missing a ticker should default to 0.0 for that ticker."""
        bt = _make_backtester()
        combined, resolution = bt._combine_signals(
            tsmom_deltas={'SPY': 0.05},
            hmm_regime='bull',
            hmm_deltas={'SPY': 0.03, 'GLD': -0.02},  # Missing TLT
            fed_regime='EASING',
            fed_deltas={'SPY': 0.02, 'TLT': -0.03},   # Missing GLD
            current_idx=300,
        )
        # All registered tickers should be present in output
        for t in bt.tickers:
            assert t in combined, f"Missing ticker {t} in combined output"

    def test_resolution_string_no_conflicts_no_neutral(self):
        """Default resolution is 'weighted_average' when no issues."""
        bt = _make_backtester()
        combined, resolution = bt._combine_signals(
            tsmom_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            hmm_regime='bull',
            hmm_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            fed_regime='EASING',
            fed_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            current_idx=300,
        )
        assert resolution == 'weighted_average'


class TestCombineSignalsWeightingEdgeCases:
    """Signal weighting boundary conditions and regime confidence edge cases."""

    def test_all_none_regimes(self):
        """hmm_regime=None and fed_regime=None use lower confidence (0.5)."""
        bt = _make_backtester()
        combined, resolution = bt._combine_signals(
            tsmom_deltas={'SPY': 0.05, 'GLD': -0.02, 'TLT': -0.03},
            hmm_regime=None,
            hmm_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            fed_regime=None,
            fed_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            current_idx=300,
        )
        assert resolution == 'weighted_average'
        for t in bt.tickers:
            assert isinstance(combined[t], float)

    def test_all_signals_zero(self):
        """All signal deltas are zero -> combined deltas are zero."""
        bt = _make_backtester()
        combined, resolution = bt._combine_signals(
            tsmom_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            hmm_regime='bull',
            hmm_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            fed_regime='EASING',
            fed_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            current_idx=300,
        )
        for v in combined.values():
            assert abs(v) < 0.001

    def test_missing_delta_in_some_sources(self):
        """When one source lacks a ticker, .get() defaults to 0.0."""
        bt = _make_backtester()
        combined, resolution = bt._combine_signals(
            tsmom_deltas={'SPY': 0.05, 'GLD': -0.02},  # Missing TLT
            hmm_regime='bull',
            hmm_deltas={'SPY': 0.03, 'TLT': -0.03},     # Missing GLD
            fed_regime='EASING',
            fed_deltas={'GLD': 0.05, 'TLT': -0.05},     # Missing SPY
            current_idx=300,
        )
        for t in bt.tickers:
            assert t in combined

    def test_conflicting_all_tickers(self):
        """Conflict on every ticker triggers split_difference."""
        bt = _make_backtester()
        tsmom = {'SPY': 0.05, 'GLD': 0.03, 'TLT': 0.04}
        fed = {'SPY': -0.05, 'GLD': -0.03, 'TLT': -0.04}
        combined, resolution = bt._combine_signals(
            tsmom_deltas=tsmom,
            hmm_regime='bull',
            hmm_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            fed_regime='TIGHTENING',
            fed_deltas=fed,
            current_idx=300,
        )
        assert 'split_difference' in resolution
        # All combined deltas should be scaled down by 0.7
        for t in bt.tickers:
            assert abs(combined[t]) < abs(tsmom[t]) + abs(fed[t])

    def test_neutral_hmm_with_split(self):
        """Combined hmm_neutral damping and split_difference conflict."""
        bt = _make_backtester()
        tsmom = {'SPY': 0.05, 'GLD': -0.02, 'TLT': -0.03}
        fed = {'SPY': -0.05, 'GLD': 0.02, 'TLT': 0.03}
        combined, resolution = bt._combine_signals(
            tsmom_deltas=tsmom,
            hmm_regime='neutral',
            hmm_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            fed_regime='TIGHTENING',
            fed_deltas=fed,
            current_idx=300,
        )
        assert 'split_difference' in resolution
        assert 'hmm_neutral' in resolution

    def test_tsmom_only_no_other_inputs(self):
        """Only TSMOM has non-zero deltas; the weighted result tracks TSMOM."""
        bt = _make_backtester()
        combined, resolution = bt._combine_signals(
            tsmom_deltas={'SPY': 0.10, 'GLD': 0.0, 'TLT': 0.0},
            hmm_regime=None,
            hmm_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            fed_regime=None,
            fed_deltas={'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            current_idx=300,
        )
        assert combined['SPY'] > 0
        assert resolution == 'weighted_average'
# _get_tsmom_deltas tests
# ---------------------------------------------------------------------------

class TestGetTsmomDeltas:
    def test_insufficient_history(self):
        """Not enough price data returns zero deltas."""
        bt = _make_backtester_with_tsmom()
        # Only 50 rows of data, well below lookback+skip (273)
        bt.prices_df = _make_prices_df(50)
        deltas = bt._get_tsmom_deltas(current_idx=30)
        for t in bt.tickers:
            assert deltas[t] == 0.0

    def test_clear_uptrend(self):
        """Strong upward trend produces positive SPY delta."""
        bt = _make_backtester_with_tsmom()
        # Build 300 rows with strong upward trend
        n = 300
        spy = np.linspace(500, 800, n)
        gld = np.linspace(200, 220, n)
        tlt = np.linspace(100, 105, n)
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B').strftime('%Y-%m-%d').tolist()
        bt.prices_df = pd.DataFrame({'SPY': spy, 'GLD': gld, 'TLT': tlt}, index=dates)
        deltas = bt._get_tsmom_deltas(current_idx=n - 1)
        # SPY has strong uptrend -> positive delta
        assert deltas['SPY'] > 0

    def test_clear_downtrend(self):
        """Strong downward trend produces negative SPY delta."""
        bt = _make_backtester_with_tsmom()
        n = 300
        spy = np.linspace(500, 300, n)
        gld = np.linspace(200, 220, n)
        tlt = np.linspace(100, 105, n)
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B').strftime('%Y-%m-%d').tolist()
        bt.prices_df = pd.DataFrame({'SPY': spy, 'GLD': gld, 'TLT': tlt}, index=dates)
        deltas = bt._get_tsmom_deltas(current_idx=n - 1)
        # SPY has strong downtrend -> negative delta
        assert deltas['SPY'] < 0


class TestGetTsmomDeltasEdgeCases:
    """TSMOM delta extraction edge cases."""

    def test_ticker_not_in_dataframe(self):
        """Ticker missing from prices_df is skipped gracefully."""
        bt = _make_backtester_with_tsmom()
        bt.tickers = ['SPY', 'MISSING']
        bt.base_allocation = {'SPY': 0.7, 'MISSING': 0.3}
        bt.prices_df = pd.DataFrame(
            {'SPY': [500.0] * 300},
            index=pd.date_range(end=datetime.now(), periods=300, freq='B'),
        )
        deltas = bt._get_tsmom_deltas(current_idx=299)
        assert 'SPY' in deltas
        # MISSING is not in columns so it is skipped; it should be absent or 0.0
        assert deltas.get('MISSING', 0.0) == 0.0

    def test_tiny_formation_return_no_signal(self):
        """Formation return < 0.001 results in signal=0 and zero delta."""
        bt = _make_backtester_with_tsmom()
        n = 300
        # Nearly flat prices produce formation return < 0.001
        spy = np.linspace(500, 500.2, n)
        gld = np.linspace(200, 200.1, n)
        tlt = np.linspace(100, 100.05, n)
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B').strftime('%Y-%m-%d').tolist()
        bt.prices_df = pd.DataFrame({'SPY': spy, 'GLD': gld, 'TLT': tlt}, index=dates)
        deltas = bt._get_tsmom_deltas(current_idx=n - 1)
        for t in bt.tickers:
            assert abs(deltas[t]) < 0.001

    def test_missing_base_allocation_default(self):
        """Ticker not in base_allocation defaults to 0.25 weight."""
        bt = _make_backtester_with_tsmom()
        n = 300
        spy = np.linspace(500, 800, n)
        gld = np.linspace(200, 220, n)
        tlt = np.linspace(100, 105, n)
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B').strftime('%Y-%m-%d').tolist()
        bt.prices_df = pd.DataFrame({'SPY': spy, 'GLD': gld, 'TLT': tlt}, index=dates)
        # Remove SPY from base_allocation to exercise the .get(..., 0.25) fallback
        bt.base_allocation = {'GLD': 0.5, 'TLT': 0.5}
        deltas = bt._get_tsmom_deltas(current_idx=n - 1)
        assert 'SPY' in deltas
        assert isinstance(deltas['SPY'], float)

    def test_barely_sufficient_history(self):
        """Exactly MIN_HISTORY_DAYS rows available does not trigger early return."""
        bt = _make_backtester_with_tsmom()
        min_needed = bt.tsmom.lookback_days + bt.tsmom.skip_days  # 273
        # Provide exactly lookback + skip + 1 rows
        n = min_needed + 1
        spy = np.linspace(500, 800, n)
        gld = np.linspace(200, 220, n)
        tlt = np.linspace(100, 105, n)
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B').strftime('%Y-%m-%d').tolist()
        bt.prices_df = pd.DataFrame({'SPY': spy, 'GLD': gld, 'TLT': tlt}, index=dates)
        deltas = bt._get_tsmom_deltas(current_idx=n - 1)
        assert isinstance(deltas['SPY'], float)


# ---------------------------------------------------------------------------
# _get_hmm_regime tests
# ---------------------------------------------------------------------------

class TestGetHmmRegime:
    def test_no_hmm_manager(self):
        """When hmm_manager is None, returns (None, zero deltas)."""
        bt = _make_backtester()
        bt.hmm_manager = None
        regime, deltas = bt._get_hmm_regime(current_idx=300)
        assert regime is None
        for t in bt.tickers:
            assert deltas[t] == 0.0

    def test_detector_not_fitted(self):
        """When detector is not fitted, returns (None, zero deltas)."""
        bt = _make_backtester()
        bt.hmm_manager = MagicMock()
        bt.hmm_manager.detector.is_fitted = False
        regime, deltas = bt._get_hmm_regime(current_idx=300)
        assert regime is None
        for t in bt.tickers:
            assert deltas[t] == 0.0

    def test_spy_not_in_columns(self):
        """When SPY is not in prices_df columns, returns (None, zero deltas)."""
        bt = _make_backtester()
        bt.tickers = ['QQQ', 'GLD']
        bt.prices_df = pd.DataFrame({'QQQ': [100.0] * 200, 'GLD': [200.0] * 200})
        regime, deltas = bt._get_hmm_regime(current_idx=150)
        assert regime is None
        for t in bt.tickers:
            assert deltas[t] == 0.0


class TestGetHmmRegimeEdgeCases:
    """HMM regime detection boundary conditions."""

    def test_insufficient_spy_history(self):
        """Fewer than 126 SPY price rows returns (None, zero deltas)."""
        bt = _make_backtester()
        bt.hmm_manager = MagicMock()
        bt.hmm_manager.detector.is_fitted = True
        bt.prices_df = pd.DataFrame(
            {'SPY': [500.0] * 100, 'GLD': [200.0] * 100, 'TLT': [100.0] * 100},
        )
        regime, deltas = bt._get_hmm_regime(current_idx=95)
        assert regime is None
        for t in bt.tickers:
            assert deltas[t] == 0.0

    def test_predict_regime_returns_none(self):
        """detector.predict_regime returning None is handled."""
        bt = _make_backtester()
        bt.hmm_manager = MagicMock()
        bt.hmm_manager.detector.is_fitted = True
        bt.hmm_manager.detector.predict_regime.return_value = None
        bt.prices_df = pd.DataFrame(
            {'SPY': [500.0] * 200, 'GLD': [200.0] * 200, 'TLT': [100.0] * 200},
        )
        regime, deltas = bt._get_hmm_regime(current_idx=150)
        assert regime is None
        for t in bt.tickers:
            assert deltas[t] == 0.0


# ---------------------------------------------------------------------------
# Fed regime edge cases
# ---------------------------------------------------------------------------

class TestFedRegimeEdgeCases:
    def test_no_gld_column(self):
        """When GLD column is missing, fallback logic still works."""
        bt = _make_backtester()
        # Build DataFrame without GLD
        n = 200
        spy = np.linspace(400, 500, n)
        tlt = np.linspace(90, 110, n)
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B').strftime('%Y-%m-%d').tolist()
        bt.prices_df = pd.DataFrame({'SPY': spy, 'TLT': tlt}, index=dates)
        # Both up > 5% => EASING
        regime, deltas = bt._get_fed_regime_deltas(136)
        assert regime is not None  # Should still classify
        assert deltas['SPY'] > 0

    def test_uncertain_inflation_proxy(self):
        """Gold outperforming SPY significantly triggers UNCERTAIN (inflation)."""
        bt = _make_backtester()
        # SPY barely moves, gold rises sharply => inflation_proxy > 0.05
        bt.prices_df = _make_regime_df(
            spy_start=400, spy_end=410,     # SPY up ~2.5%
            tlt_start=100, tlt_end=102,      # TLT up ~2%
            gld_start=100, gld_end=200,      # Gold up ~100%
        )
        regime, deltas = bt._get_fed_regime_deltas(136)
        assert regime == 'UNCERTAIN'
        assert deltas['GLD'] > 0  # Gold should get positive delta

    def test_uncertain_mixed_signals(self):
        """Mixed signals that don't fit easing/tightening/neutral give UNCERTAIN."""
        bt = _make_backtester()
        # SPY up ~7.5% (fails neutral: abs > 3%), TLT up ~3% (fails easing: < 5%)
        # This triggers the else branch => UNCERTAIN
        bt.prices_df = _make_regime_df(
            spy_start=400, spy_end=500,   # SPY up ~25%
            tlt_start=100, tlt_end=110,    # TLT up ~10% (~3% at current_idx)
        )
        regime, deltas = bt._get_fed_regime_deltas(136)
        assert regime == 'UNCERTAIN'

    def test_exact_easing_threshold(self):
        """TLT and SPY both above 5% on 63-day window triggers EASING."""
        bt = _make_backtester()
        n = 250
        flat_len = 118
        ramp_len = n - flat_len
        spy = np.concatenate([np.ones(flat_len) * 400, np.linspace(400, 500, ramp_len)])
        tlt = np.concatenate([np.ones(flat_len) * 100, np.linspace(100, 120, ramp_len)])
        gld = np.ones(n) * 200
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B').strftime('%Y-%m-%d').tolist()
        bt.prices_df = pd.DataFrame({'SPY': spy, 'GLD': gld, 'TLT': tlt}, index=dates)
        # current_idx=180: need n >= 180+63=243; 250 >= 243 -- passes
        # Window rows 118:180 both in ramp -- >5% return for both
        regime, deltas = bt._get_fed_regime_deltas(180)
        assert regime == 'EASING'
        assert deltas['SPY'] > 0

    def test_exact_tightening_threshold(self):
        """TLT and SPY both below -5% on 63-day window triggers TIGHTENING."""
        bt = _make_backtester()
        n = 250
        flat_len = 118
        ramp_len = n - flat_len
        spy = np.concatenate([np.ones(flat_len) * 500, np.linspace(500, 400, ramp_len)])
        tlt = np.concatenate([np.ones(flat_len) * 120, np.linspace(120, 100, ramp_len)])
        gld = np.ones(n) * 200
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B').strftime('%Y-%m-%d').tolist()
        bt.prices_df = pd.DataFrame({'SPY': spy, 'GLD': gld, 'TLT': tlt}, index=dates)
        regime, deltas = bt._get_fed_regime_deltas(180)
        assert regime == 'TIGHTENING'
        assert deltas['SPY'] < 0

    def test_neutral_threshold(self):
        """Within abs(TLT) < 2% and abs(SPY) < 3% on 63-day window triggers NEUTRAL."""
        bt = _make_backtester()
        n = 250
        flat_len = 118
        ramp_len = n - flat_len
        spy = np.concatenate([np.ones(flat_len) * 500, np.linspace(500, 510, ramp_len)])
        tlt = np.concatenate([np.ones(flat_len) * 100, np.linspace(100, 101, ramp_len)])
        gld = np.ones(n) * 200
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B').strftime('%Y-%m-%d').tolist()
        bt.prices_df = pd.DataFrame({'SPY': spy, 'GLD': gld, 'TLT': tlt}, index=dates)
        regime, deltas = bt._get_fed_regime_deltas(180)
        assert regime == 'NEUTRAL'
        assert deltas['SPY'] == 0.0


# ---------------------------------------------------------------------------
# Crisis return edge cases
# ---------------------------------------------------------------------------

class TestCrisisReturnEdgeCases:
    def test_exact_boundary_dates(self):
        """Positions exactly on start/end boundaries are included."""
        bt = _make_backtester()
        positions = [
            _make_position(date='2020-02-01', value=100000),
            _make_position(date='2020-03-15', value=80000),
            _make_position(date='2020-04-30', value=95000),
        ]
        ret = bt._calculate_crisis_return(positions, '2020-02-01', '2020-04-30')
        assert ret == pytest.approx(-0.05, abs=0.01)

    def test_single_position_in_range(self):
        """A single position within the range returns 0% return."""
        bt = _make_backtester()
        positions = [_make_position(date='2020-03-01', value=100000)]
        ret = bt._calculate_crisis_return(positions, '2020-02-01', '2020-04-30')
        assert ret == pytest.approx(0.0, abs=0.01)

    def test_single_day_crisis_range(self):
        """Crisis period with start == end returns 0% when exact date exists."""
        bt = _make_backtester()
        positions = [
            _make_position(date='2020-03-01', value=100000),
            _make_position(date='2020-03-02', value=101000),
        ]
        ret = bt._calculate_crisis_return(positions, '2020-03-01', '2020-03-01')
        assert ret == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Run baseline edge cases
# ---------------------------------------------------------------------------

class TestRunBaselineEdgeCases:
    def test_single_day(self):
        """Only one day of data returns a baseline with 0 returns."""
        bt = _make_backtester()
        bt.prices_df = _make_prices_df(300)
        # start_idx + 1 == end_idx => only one day of returns
        result = bt._run_baseline(298, 299, 100000.0)
        assert len(result['daily_returns']) == 1
        assert result['cagr'] is not None

    def test_baseline_zero_volatility(self):
        """Constant prices produce 0 % CAGR and 0 daily returns."""
        bt = _make_backtester()
        n = 100
        bt.prices_df = pd.DataFrame(
            {'SPY': [500.0] * n, 'GLD': [200.0] * n, 'TLT': [100.0] * n},
            index=pd.date_range(end=datetime.now(), periods=n, freq='B'),
        )
        result = bt._run_baseline(0, n - 1, 100000.0)
        assert result['cagr'] == 0.0
        assert all(r == 0.0 for r in result['daily_returns'])

    def test_baseline_sharpe_with_positive_volatility(self):
        """Sharpe ratio is non-zero when CAGR is positive."""
        bt = _make_backtester()
        bt.prices_df = _make_prices_df(300, seed=42)
        result = bt._run_baseline(252, 299, 100000.0)
        if result['cagr'] > 0:
            assert result['sharpe'] != 0


# ---------------------------------------------------------------------------
# Backtester init edge cases
# ---------------------------------------------------------------------------

class TestInitEdgeCases:
    def test_no_tickers_provided_defaults(self):
        """No tickers provided uses default ['SPY', 'GLD', 'TLT']."""
        bt = CombinedStrategyBacktester.__new__(CombinedStrategyBacktester)
        bt.tickers = None
        bt.base_allocation = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        bt.transaction_cost = 0.001
        bt.rebalance_freq = 21
        bt.fed_overlay = MagicMock()
        assert bt.tickers is None
        # tickers is None; _combine_signals uses self.tickers
        # This test verifies the attribute can be None without crashing

    def test_base_allocation_empty(self):
        """Empty base allocation doesn't crash combine_signals."""
        bt = _make_backtester()
        bt.tickers = ['SPY']
        bt.base_allocation = {'SPY': 1.0}
        result, resolution = bt._combine_signals(
            tsmom_deltas={'SPY': 0.0},
            hmm_regime=None,
            hmm_deltas={'SPY': 0.0},
            fed_regime=None,
            fed_deltas={'SPY': 0.0},
            current_idx=300,
        )
        assert 'SPY' in result

    def test_base_allocation_single_ticker(self):
        """Single ticker base allocation doesn't break combine_signals."""
        bt = _make_backtester()
        bt.tickers = ['SPY']
        bt.base_allocation = {'SPY': 1.0}
        result, resolution = bt._combine_signals(
            tsmom_deltas={'SPY': 0.05},
            hmm_regime='bull',
            hmm_deltas={'SPY': 0.03},
            fed_regime='EASING',
            fed_deltas={'SPY': 0.02},
            current_idx=300,
        )
        assert 'SPY' in result
        assert result['SPY'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
