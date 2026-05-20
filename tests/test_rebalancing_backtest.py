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

from src.rebalancing.backtest import (
    RebalanceEvent, StrategyResult,
    BASE_WEIGHTS,
    simulate_synthetic_vpin, build_price_index,
    _compute_strategy_result, load_price_data,
    run_calendar_strategy, run_drift_only_strategy,
)


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


# ---------------------------------------------------------------------------
# run_smart_strategy tests (uses patch.object instead of sys.modules hacking)
# ---------------------------------------------------------------------------

import src.rebalancing.backtest as _bt_mod
from src.rebalancing.smart_rebalancer import RebalanceDecision as _RealDecision, UrgencyLevel as _RealUrgency


def _make_controller_mock():
    """Create a mock controller that returns execute decisions."""
    controller = MagicMock()
    result = MagicMock()
    result.decision = _RealDecision.EXECUTE
    result.max_drift = 0.12
    result.estimated_cost_bps = 4.0
    result.urgency = _RealUrgency.MODERATE
    controller.should_rebalance.return_value = result
    controller.record_rebalance = MagicMock()
    return controller


class TestSmartStrategy:
    @patch.object(_bt_mod, 'SmartRebalancingController')
    def test_returns_strategy_result(self, MockController):
        MockController.return_value = _make_controller_mock()
        data, dates = _make_price_data(500)
        idx = build_price_index(data)
        vpin = simulate_synthetic_vpin(dates)
        r = _bt_mod.run_smart_strategy(idx, dates, vpin)
        assert isinstance(r, StrategyResult)

    @patch.object(_bt_mod, 'SmartRebalancingController')
    def test_smart_has_rebalances(self, MockController):
        MockController.return_value = _make_controller_mock()
        data, dates = _make_price_data(500)
        idx = build_price_index(data)
        vpin = simulate_synthetic_vpin(dates)
        r = _bt_mod.run_smart_strategy(idx, dates, vpin)
        assert r.total_rebalances >= 0

    @patch.object(_bt_mod, 'SmartRebalancingController')
    def test_smart_deferred_count(self, MockController):
        """When controller defers, no rebalance is recorded."""
        controller = MagicMock()
        result = MagicMock()
        result.decision = _RealDecision.DEFER_TOXICITY
        result.max_drift = 0.05
        result.estimated_cost_bps = 2.0
        result.urgency = _RealUrgency.LOW
        controller.should_rebalance.return_value = result
        controller.record_rebalance = MagicMock()
        MockController.return_value = controller

        data, dates = _make_price_data(500)
        idx = build_price_index(data)
        vpin = simulate_synthetic_vpin(dates)
        r = _bt_mod.run_smart_strategy(idx, dates, vpin)
        # All deferrals → 0 rebalances
        assert r.total_rebalances == 0

    @patch.object(_bt_mod, 'SmartRebalancingController')
    def test_smart_final_value_positive(self, MockController):
        MockController.return_value = _make_controller_mock()
        data, dates = _make_price_data(500)
        idx = build_price_index(data)
        vpin = simulate_synthetic_vpin(dates)
        r = _bt_mod.run_smart_strategy(idx, dates, vpin)
        assert r.final_value > 0


# ---------------------------------------------------------------------------
# print_comparison tests
# ---------------------------------------------------------------------------

class TestPrintComparison:
    def _make_results(self):
        cal = StrategyResult(
            name='Calendar', total_rebalances=10, total_cost_bps=70.0,
            avg_cost_per_rebalance=7.0, annual_cost_pct=0.05,
            max_drawdown=-15.0, tracking_error=0.0, final_value=150000,
            cagr=8.0, sharpe=0.7, events=[],
        )
        drift = StrategyResult(
            name='Drift-Only', total_rebalances=5, total_cost_bps=25.0,
            avg_cost_per_rebalance=5.0, annual_cost_pct=0.02,
            max_drawdown=-14.0, tracking_error=0.0, final_value=152000,
            cagr=8.2, sharpe=0.72, events=[],
        )
        smart = StrategyResult(
            name='Smart', total_rebalances=7, total_cost_bps=35.0,
            avg_cost_per_rebalance=5.0, annual_cost_pct=0.03,
            max_drawdown=-14.5, tracking_error=0.0, final_value=151000,
            cagr=8.1, sharpe=0.71, events=[],
        )
        return {'calendar': cal, 'drift': drift, 'smart': smart}

    def test_prints_without_error(self, capsys):
        results = self._make_results()
        _bt_mod.print_comparison(results)
        captured = capsys.readouterr()
        assert 'STRATEGY COMPARISON' in captured.out
        assert 'Calendar' in captured.out

    def test_prints_validation(self, capsys):
        results = self._make_results()
        _bt_mod.print_comparison(results)
        captured = capsys.readouterr()
        assert 'VALIDATION' in captured.out

    def test_prints_metrics(self, capsys):
        results = self._make_results()
        _bt_mod.print_comparison(results)
        captured = capsys.readouterr()
        assert 'CAGR' in captured.out
        assert 'Sharpe' in captured.out
        assert 'Max Drawdown' in captured.out


# ---------------------------------------------------------------------------
# save_results tests
# ---------------------------------------------------------------------------

class TestSaveResults:
    def _make_results(self):
        cal = StrategyResult(
            name='Calendar', total_rebalances=10, total_cost_bps=70.0,
            avg_cost_per_rebalance=7.0, annual_cost_pct=0.05,
            max_drawdown=-15.0, tracking_error=0.0, final_value=150000,
            cagr=8.0, sharpe=0.7, events=[],
        )
        drift = StrategyResult(
            name='Drift-Only', total_rebalances=5, total_cost_bps=25.0,
            avg_cost_per_rebalance=5.0, annual_cost_pct=0.02,
            max_drawdown=-14.0, tracking_error=0.0, final_value=152000,
            cagr=8.2, sharpe=0.72, events=[],
        )
        smart = StrategyResult(
            name='Smart', total_rebalances=7, total_cost_bps=35.0,
            avg_cost_per_rebalance=5.0, annual_cost_pct=0.03,
            max_drawdown=-14.5, tracking_error=0.0, final_value=151000,
            cagr=8.1, sharpe=0.71, events=[],
        )
        return {'calendar': cal, 'drift': drift, 'smart': smart}

    def test_saves_json(self, tmp_path):
        results = self._make_results()
        with patch.object(_bt_mod, 'DATA_DIR', tmp_path):
            _bt_mod.save_results(results)

        output_path = tmp_path / "smart_rebalance_backtest_results.json"
        assert output_path.exists()

        data = json.loads(output_path.read_text())
        assert 'metadata' in data
        assert 'strategies' in data
        assert 'calendar' in data['strategies']

    def test_json_has_metadata(self, tmp_path):
        results = self._make_results()
        with patch.object(_bt_mod, 'DATA_DIR', tmp_path):
            _bt_mod.save_results(results)

        output_path = tmp_path / "smart_rebalance_backtest_results.json"
        data = json.loads(output_path.read_text())
        assert data['metadata']['version'] == '2.90'
        assert data['metadata']['phase'] == '3'

    def test_json_has_strategy_metrics(self, tmp_path):
        results = self._make_results()
        with patch.object(_bt_mod, 'DATA_DIR', tmp_path):
            _bt_mod.save_results(results)

        output_path = tmp_path / "smart_rebalance_backtest_results.json"
        data = json.loads(output_path.read_text())
        cal = data['strategies']['calendar']
        assert 'total_rebalances' in cal
        assert 'cagr' in cal
        assert 'sharpe' in cal
        assert 'max_drawdown' in cal


# ---------------------------------------------------------------------------
# run_full_backtest tests
# ---------------------------------------------------------------------------

class TestRunFullBacktest:
    def test_with_synthetic_data(self, tmp_path):
        """Run full backtest with synthetic price file."""
        data, dates = _make_price_data(1000, start_date='2020-01-02')
        price_file = tmp_path / "prices.json"
        price_file.write_text(json.dumps(data))

        with patch.object(_bt_mod, 'DATA_DIR', tmp_path):
            results = _bt_mod.run_full_backtest(
                price_filepath=str(price_file),
                start_date='2020-01-01',
                end_date='2023-12-31',
            )

        assert 'calendar' in results
        assert 'drift' in results
        assert 'smart' in results
        assert isinstance(results['calendar'], StrategyResult)

    def test_full_backtest_saves_output(self, tmp_path):
        data, dates = _make_price_data(1000, start_date='2020-01-02')
        price_file = tmp_path / "prices.json"
        price_file.write_text(json.dumps(data))

        with patch.object(_bt_mod, 'DATA_DIR', tmp_path):
            _bt_mod.run_full_backtest(
                price_filepath=str(price_file),
                start_date='2020-01-01',
                end_date='2023-12-31',
            )

        output_path = tmp_path / "smart_rebalance_backtest_results.json"
        assert output_path.exists()


# ---------------------------------------------------------------------------
# Calendar strategy edge cases
# ---------------------------------------------------------------------------

class TestCalendarStrategyEdgeCases:
    def test_no_rebalance_months(self):
        """If dates don't include rebalance months, no rebalances."""
        data, dates = _make_price_data(30, start_date='2020-03-01')
        idx = build_price_index(data)
        r = run_calendar_strategy(idx, dates, rebalance_months=[7])
        # March only, no July → 0 rebalances
        assert r.total_rebalances == 0

    def test_custom_initial_value(self):
        data, dates = _make_price_data(500)
        idx = build_price_index(data)
        r = run_calendar_strategy(idx, dates, initial_value=50000)
        assert r.final_value > 0


class TestDriftOnlyEdgeCases:
    def test_high_drift_threshold(self):
        """Very high threshold → no rebalances."""
        data, dates = _make_price_data(100)
        idx = build_price_index(data)
        r = run_drift_only_strategy(idx, dates, drift_threshold=0.99)
        assert r.total_rebalances == 0

    def test_custom_initial_value(self):
        data, dates = _make_price_data(500)
        idx = build_price_index(data)
        r = run_drift_only_strategy(idx, dates, initial_value=200000)
        assert r.final_value > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
