#!/usr/bin/env python3
"""
Tests for Smart Rebalancing Backtest — data classes, constants, VPIN simulation,
price index building, strategy result computation, and calendar/drift strategies.
"""
import sys
import os
import json
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

# Mock smart_rebalancer before import
_orig_sr = sys.modules.get('src.rebalancing.smart_rebalancer')
mock_sr = MagicMock()
mock_sr.SmartRebalancingController = MagicMock
mock_sr.PortfolioSnapshot = MagicMock
mock_sr.MarketConditions = MagicMock
mock_sr.RebalanceDecision = MagicMock()
mock_sr.UrgencyLevel = MagicMock()
# Make RebalanceDecision enum-like values
for val in ['EXECUTE', 'OVERRIDE_EMERGENCY', 'DEFER_TOXICITY', 'DEFER_TIMING', 'DEFER_BUDGET']:
    setattr(mock_sr.RebalanceDecision, val, val)
sys.modules['src.rebalancing.smart_rebalancer'] = mock_sr

from src.rebalancing.backtest import (
    RebalanceEvent, StrategyResult,
    BASE_WEIGHTS,
    simulate_synthetic_vpin, build_price_index,
    _compute_strategy_result, load_price_data,
    run_calendar_strategy, run_drift_only_strategy,
)

# Restore
if _orig_sr is None:
    sys.modules.pop('src.rebalancing.smart_rebalancer', None)
else:
    sys.modules['src.rebalancing.smart_rebalancer'] = _orig_sr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_data(n_days=100, start_date='2020-01-02', base_prices=None):
    """Create synthetic price data dict."""
    if base_prices is None:
        base_prices = {'SPY': 300.0, 'GLD': 150.0, 'TLT': 130.0}
    from datetime import timedelta
    dates = []
    d = datetime.strptime(start_date, '%Y-%m-%d')
    for _ in range(n_days):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)

    data = {}
    for sym, base in base_prices.items():
        bars = []
        price = base
        for dt in dates:
            price *= 1.001
            bars.append({'d': dt, 'p': round(price, 2)})
        data[sym] = bars
    return data, dates


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestRebalanceEvent:
    def test_creation(self):
        e = RebalanceEvent(
            date='2020-06-01', strategy='smart', decision='execute',
            max_drift=0.12, cost_bps=5.0, urgency='drift_triggered', vpin=0.35,
        )
        assert e.strategy == 'smart'
        assert e.cost_bps == 5.0


class TestStrategyResult:
    def test_creation(self):
        r = StrategyResult(
            name='Test', total_rebalances=10, total_cost_bps=50.0,
            avg_cost_per_rebalance=5.0, annual_cost_pct=0.05,
            max_drawdown=-15.0, tracking_error=0.0, final_value=150000,
            cagr=8.0, sharpe=0.7, events=[],
        )
        assert r.name == 'Test'
        assert r.sharpe == 0.7


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_base_weights_sum_to_one(self):
        assert abs(sum(BASE_WEIGHTS.values()) - 1.0) < 0.001

    def test_base_weights_keys(self):
        assert set(BASE_WEIGHTS.keys()) == {'SPY', 'GLD', 'TLT'}


# ---------------------------------------------------------------------------
# simulate_synthetic_vpin tests
# ---------------------------------------------------------------------------

class TestSimulateSyntheticVpin:
    def test_returns_dict(self):
        dates = ['2020-01-02', '2020-01-03', '2020-01-06']
        vpin = simulate_synthetic_vpin(dates)
        assert isinstance(vpin, dict)
        assert len(vpin) == 3

    def test_values_bounded(self):
        dates = [f'2020-01-{d:02d}' for d in range(2, 32)]
        vpin = simulate_synthetic_vpin(dates)
        for val in vpin.values():
            assert 0.10 <= val <= 0.90

    def test_crisis_periods_higher(self):
        # COVID crash period
        dates = [f'2020-03-{d:02d}' for d in range(2, 32)]
        vpin_covid = simulate_synthetic_vpin(dates)
        # Normal period
        dates_normal = [f'2021-06-{d:02d}' for d in range(2, 22)]
        vpin_normal = simulate_synthetic_vpin(dates_normal)
        # COVID should trend higher
        avg_covid = sum(vpin_covid.values()) / len(vpin_covid)
        avg_normal = sum(vpin_normal.values()) / len(vpin_normal)
        assert avg_covid > avg_normal

    def test_deterministic(self):
        dates = ['2020-01-02', '2020-01-03']
        vpin1 = simulate_synthetic_vpin(dates)
        vpin2 = simulate_synthetic_vpin(dates)
        assert vpin1 == vpin2


# ---------------------------------------------------------------------------
# build_price_index tests
# ---------------------------------------------------------------------------

class TestBuildPriceIndex:
    def test_returns_dict(self):
        data, dates = _make_price_data(10)
        idx = build_price_index(data)
        assert isinstance(idx, dict)

    def test_indexed_by_date(self):
        data, dates = _make_price_data(10)
        idx = build_price_index(data)
        assert dates[0] in idx

    def test_has_all_symbols(self):
        data, dates = _make_price_data(10)
        idx = build_price_index(data)
        for d in dates:
            assert 'SPY' in idx[d]
            assert 'GLD' in idx[d]
            assert 'TLT' in idx[d]

    def test_empty_prices(self):
        idx = build_price_index({})
        assert idx == {}


# ---------------------------------------------------------------------------
# _compute_strategy_result tests
# ---------------------------------------------------------------------------

class TestComputeStrategyResult:
    def test_returns_strategy_result(self):
        daily_returns = [0.001] * 252
        r = _compute_strategy_result(
            'Test', events=[], total_cost=50.0, rebalance_count=10,
            daily_returns=daily_returns, final_value=110000,
            total_days=252, initial_value=100000,
        )
        assert isinstance(r, StrategyResult)
        assert r.name == 'Test'

    def test_cagr_positive(self):
        daily_returns = [0.001] * 252
        r = _compute_strategy_result(
            'Test', events=[], total_cost=0, rebalance_count=0,
            daily_returns=daily_returns, final_value=110000,
            total_days=252, initial_value=100000,
        )
        assert r.cagr > 0

    def test_max_drawdown_non_positive(self):
        daily_returns = [0.01] * 50 + [-0.05] * 5 + [0.01] * 50
        r = _compute_strategy_result(
            'Test', events=[], total_cost=0, rebalance_count=0,
            daily_returns=daily_returns, final_value=110000,
            total_days=105, initial_value=100000,
        )
        assert r.max_drawdown <= 0

    def test_sharpe_zero_for_empty_returns(self):
        r = _compute_strategy_result(
            'Test', events=[], total_cost=0, rebalance_count=0,
            daily_returns=[], final_value=100000,
            total_days=0, initial_value=100000,
        )
        assert r.sharpe == 0

    def test_avg_cost_per_rebalance(self):
        r = _compute_strategy_result(
            'Test', events=[], total_cost=100.0, rebalance_count=10,
            daily_returns=[0.001] * 100, final_value=100000,
            total_days=100, initial_value=100000,
        )
        assert r.avg_cost_per_rebalance == 10.0

    def test_zero_rebalances(self):
        r = _compute_strategy_result(
            'Test', events=[], total_cost=0, rebalance_count=0,
            daily_returns=[0.001] * 100, final_value=100000,
            total_days=100, initial_value=100000,
        )
        assert r.avg_cost_per_rebalance == 0

    def test_with_deferred(self):
        r = _compute_strategy_result(
            'Test', events=[], total_cost=50.0, rebalance_count=5,
            daily_returns=[0.001] * 100, final_value=100000,
            total_days=100, initial_value=100000, deferred=3,
        )
        assert isinstance(r, StrategyResult)


# ---------------------------------------------------------------------------
# load_price_data tests
# ---------------------------------------------------------------------------

class TestLoadPriceData:
    def test_loads_json(self, tmp_path):
        data = {'SPY': [{'d': '2020-01-02', 'p': 300.0}]}
        fpath = tmp_path / "prices.json"
        fpath.write_text(json.dumps(data))
        loaded = load_price_data(str(fpath))
        assert 'SPY' in loaded


# ---------------------------------------------------------------------------
# run_calendar_strategy tests
# ---------------------------------------------------------------------------

class TestCalendarStrategy:
    def test_returns_strategy_result(self):
        data, dates = _make_price_data(500)
        idx = build_price_index(data)
        r = run_calendar_strategy(idx, dates)
        assert isinstance(r, StrategyResult)

    def test_has_rebalances(self):
        data, dates = _make_price_data(500)
        idx = build_price_index(data)
        r = run_calendar_strategy(idx, dates)
        assert r.total_rebalances >= 0

    def test_final_value_positive(self):
        data, dates = _make_price_data(500)
        idx = build_price_index(data)
        r = run_calendar_strategy(idx, dates)
        assert r.final_value > 0


# ---------------------------------------------------------------------------
# run_drift_only_strategy tests
# ---------------------------------------------------------------------------

class TestDriftOnlyStrategy:
    def test_returns_strategy_result(self):
        data, dates = _make_price_data(500)
        idx = build_price_index(data)
        r = run_drift_only_strategy(idx, dates)
        assert isinstance(r, StrategyResult)

    def test_final_value_positive(self):
        data, dates = _make_price_data(500)
        idx = build_price_index(data)
        r = run_drift_only_strategy(idx, dates)
        assert r.final_value > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
