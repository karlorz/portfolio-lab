#!/usr/bin/env python3
"""
Tests for src/backtest/metrics.py — shared backtest metrics module.
"""
import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from pathlib import Path

from src.backtest.metrics import (
    BacktestConfig, BacktestResult, DailyPrices,
    BacktestMetrics, OverlayMetrics, CrisisReturns,
    compute_metrics, compute_crisis_returns,
    print_metrics_report, save_results_json,
)


class TestBacktestConfig:
    """Tests for the canonical BacktestConfig dataclass."""

    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.start_date == "2006-01-01"
        assert cfg.end_date == "2026-05-15"
        assert cfg.initial_capital == 100000.0
        assert cfg.rebalance_frequency_days == 21
        assert cfg.rebalance_frequency == "monthly"
        assert cfg.transaction_cost_bps == 10.0
        assert cfg.extras == {}

    def test_base_weights_are_canonical(self):
        """Base weights must match BASE_ALLOCATION from src.paths."""
        cfg = BacktestConfig()
        from src.paths import BASE_ALLOCATION
        for key, val in BASE_ALLOCATION.items():
            assert cfg.base_weights[key] == val

    def test_custom_start_end(self):
        cfg = BacktestConfig(start_date="2020-01-01", end_date="2024-12-31")
        assert cfg.start_date == "2020-01-01"
        assert cfg.end_date == "2024-12-31"

    def test_custom_base_weights(self):
        cfg = BacktestConfig(base_weights={"SPY": 0.5, "GLD": 0.3, "TLT": 0.2})
        assert cfg.base_weights["SPY"] == 0.5
        assert cfg.base_weights["GLD"] == 0.3
        assert cfg.base_weights["TLT"] == 0.2

    def test_extras_dict(self):
        cfg = BacktestConfig(extras={"max_shift": 0.05, "lookback": 130})
        assert cfg.extras["max_shift"] == 0.05
        assert cfg.extras["lookback"] == 130

    def test_default_extras_is_empty_and_independent(self):
        """Each instance should get its own extras dict."""
        cfg1 = BacktestConfig()
        cfg2 = BacktestConfig()
        cfg1.extras["foo"] = "bar"
        assert "foo" not in cfg2.extras


class TestDailyPrices:
    """Tests for the canonical DailyPrices dataclass."""

    def test_required_fields(self):
        dp = DailyPrices(date="2024-01-01", spy=500.0, gld=180.0, tlt=130.0)
        assert dp.date == "2024-01-01"
        assert dp.spy == 500.0
        assert dp.gld == 180.0
        assert dp.tlt == 130.0
        assert dp.vix is None  # Optional

    def test_optional_vix(self):
        dp = DailyPrices(date="2024-01-01", spy=500, gld=180, tlt=130, vix=18.5)
        assert dp.vix == 18.5

    def test_optional_crypto(self):
        dp = DailyPrices(date="2024-01-01", spy=500, gld=180, tlt=130,
                         btc=42000.0, eth=2200.0)
        assert dp.btc == 42000.0
        assert dp.eth == 2200.0

    def test_optional_ief_shy(self):
        dp = DailyPrices(date="2024-01-01", spy=500, gld=180, tlt=130,
                         ief=95.0, shy=82.0)
        assert dp.ief == 95.0
        assert dp.shy == 82.0

    def test_extras_default_empty(self):
        dp = DailyPrices(date="2024-01-01", spy=500, gld=180, tlt=130)
        assert dp.extras == {}

    def test_extras_custom(self):
        dp = DailyPrices(date="2024-01-01", spy=500, gld=180, tlt=130,
                         extras={"DBC": 18.5, "VIXY": 12.3})
        assert dp.extras["DBC"] == 18.5

    def test_all_none_optional_fields(self):
        dp = DailyPrices(date="2024-01-01", spy=500, gld=180, tlt=130)
        assert dp.vix is None
        assert dp.ief is None
        assert dp.shy is None
        assert dp.btc is None
        assert dp.eth is None


class TestBacktestResult:
    """Tests for the canonical BacktestResult dataclass."""

    def test_required_fields_only(self):
        r = BacktestResult(total_return=10.0, cagr=5.0, volatility=12.0,
                           sharpe_ratio=0.5, max_drawdown=-15.0)
        assert r.total_rebalances == 0
        assert r.total_transaction_costs == 0.0
        assert r.avg_turnover == 0.0
        assert r.baseline_sharpe is None
        assert r.sharpe_improvement is None
        assert r.extras == {}
        assert r.crisis_returns is None

    def test_all_fields(self):
        r = BacktestResult(
            total_return=50.0, cagr=8.0, volatility=10.0,
            sharpe_ratio=0.8, max_drawdown=-20.0,
            total_rebalances=120, total_transaction_costs=500.0,
            avg_turnover=0.03,
            baseline_sharpe=0.7, sharpe_improvement=0.1,
            extras={"custom": "value"},
            crisis_returns={"2008": -12.0},
        )
        assert r.total_rebalances == 120
        assert r.baseline_sharpe == 0.7
        assert r.extras["custom"] == "value"
        assert r.crisis_returns["2008"] == -12.0

    def test_extras_independent_per_instance(self):
        r1 = BacktestResult(total_return=0, cagr=0, volatility=0,
                            sharpe_ratio=0, max_drawdown=0)
        r2 = BacktestResult(total_return=0, cagr=0, volatility=0,
                            sharpe_ratio=0, max_drawdown=0)
        r1.extras["key"] = "val"
        assert "key" not in r2.extras

    def test_crisis_returns_none_vs_empty(self):
        r_none = BacktestResult(total_return=0, cagr=0, volatility=0,
                                sharpe_ratio=0, max_drawdown=0)
        r_empty = BacktestResult(total_return=0, cagr=0, volatility=0,
                                 sharpe_ratio=0, max_drawdown=0,
                                 crisis_returns={})
        assert r_none.crisis_returns is None
        assert r_empty.crisis_returns == {}

    def test_negative_sharpe_improvement(self):
        r = BacktestResult(total_return=5.0, cagr=3.0, volatility=12.0,
                           sharpe_ratio=0.3, max_drawdown=-10.0,
                           baseline_sharpe=0.5, sharpe_improvement=-0.2)
        assert r.sharpe_improvement < 0


class TestBacktestMetrics:
    def test_creation_defaults(self):
        m = BacktestMetrics(total_return=10.0, cagr=8.0, volatility=12.0,
                            sharpe_ratio=0.67, max_drawdown=-15.0)
        assert m.total_rebalances == 0
        assert m.total_transaction_costs == 0.0

    def test_custom_fields(self):
        m = BacktestMetrics(total_return=10.0, cagr=8.0, volatility=12.0,
                            sharpe_ratio=0.67, max_drawdown=-15.0,
                            total_rebalances=24, total_transaction_costs=50.0)
        assert m.total_rebalances == 24


class TestOverlayMetrics:
    def test_creation(self):
        m = OverlayMetrics(baseline_sharpe=0.94, sharpe_improvement=0.015)
        assert m.overlay_active_count == 0
        assert m.overlay_active_pct == 0.0

    def test_with_active_stats(self):
        m = OverlayMetrics(baseline_sharpe=0.94, sharpe_improvement=-0.012,
                           overlay_active_count=120, overlay_active_pct=0.45)
        assert m.overlay_active_pct == 0.45


class TestCrisisReturns:
    def test_empty(self):
        c = CrisisReturns()
        assert c.get('2008') is None

    def test_with_returns(self):
        c = CrisisReturns(returns={'2008': -12.3, '2020': -7.1, '2022': -13.0})
        assert c.get('2008') == -12.3
        assert c.get('2023') is None


class TestComputeMetrics:
    def test_empty_curve(self):
        m = compute_metrics([], 100000)
        assert m.cagr == 0.0
        assert m.sharpe_ratio == 0.0

    def test_single_value(self):
        m = compute_metrics([100000], 100000)
        assert m.cagr == 0.0

    def test_positive_return(self):
        curve = [100000, 101000, 102000, 103000, 104000]
        m = compute_metrics(curve, 100000)
        assert m.total_return > 0
        assert m.cagr > 0
        assert m.max_drawdown == 0.0  # Monotonically increasing

    def test_negative_return(self):
        curve = [100000, 99000, 98000, 97000]
        m = compute_metrics(curve, 100000)
        assert m.total_return < 0
        assert m.cagr < 0

    def test_drawdown(self):
        curve = [100000, 110000, 95000, 105000]
        m = compute_metrics(curve, 100000)
        assert m.max_drawdown < 0  # Should have drawdown from 110k to 95k

    def test_zero_initial_capital(self):
        curve = [0, 1000, 2000]
        m = compute_metrics(curve, 0)
        assert m.cagr == 0.0

    def test_sharpe_positive(self):
        np.random.seed(42)
        curve = [100000]
        for _ in range(251):
            curve.append(curve[-1] * (1 + np.random.normal(0.0004, 0.01)))
        m = compute_metrics(curve, 100000)
        assert m.sharpe_ratio > 0

    def test_rounding(self):
        curve = [100000, 105000]
        m = compute_metrics(curve, 100000)
        # Values should be rounded to 2-4 decimal places
        assert isinstance(m.total_return, float)


class TestComputeCrisisReturns:
    def test_basic(self):
        prices = {
            '2008-01-02': {'SPY': 100, 'GLD': 80, 'TLT': 90},
            '2008-12-31': {'SPY': 70, 'GLD': 85, 'TLT': 110},
        }
        result = compute_crisis_returns(prices, ['2008-01-02', '2008-12-31'])
        assert '2008' in result
        assert result['2008'] < 0  # SPY dropped, GLD/TLT partial offset

    def test_missing_year(self):
        prices = {
            '2020-01-02': {'SPY': 100, 'GLD': 150, 'TLT': 130},
            '2020-12-31': {'SPY': 120, 'GLD': 170, 'TLT': 155},
        }
        result = compute_crisis_returns(prices, ['2020-01-02', '2020-12-31'],
                                         crisis_years=['2008'])
        assert '2008' not in result

    def test_custom_weights(self):
        prices = {
            '2022-01-03': {'SPY': 100, 'GLD': 160, 'TLT': 140},
            '2022-12-30': {'SPY': 80, 'GLD': 155, 'TLT': 115},
        }
        result = compute_crisis_returns(prices, ['2022-01-03', '2022-12-30'],
                                         base_weights={'SPY': 1.0, 'GLD': 0.0, 'TLT': 0.0})
        assert '2022' in result
        assert result['2022'] < -15  # SPY alone dropped 20%

    def test_insufficient_data(self):
        prices = {'2008-01-02': {'SPY': 100}}
        result = compute_crisis_returns(prices, ['2008-01-02'])
        assert '2008' not in result  # Need at least 2 days


class TestPrintMetricsReport:
    def test_prints_without_error(self, capsys):
        m = BacktestMetrics(total_return=10.5, cagr=8.2, volatility=12.1,
                            sharpe_ratio=0.68, max_drawdown=-15.3)
        print_metrics_report(m, title="Test Report")
        captured = capsys.readouterr()
        assert "Test Report" in captured.out
        assert "8.2" in captured.out


class TestSaveResultsJson:
    def test_saves_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_results.json')
            data = {'cagr': 8.5, 'sharpe': 0.68, 'nested': {'key': 'val'}}
            save_results_json(data, output_path=path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded['cagr'] == 8.5
            assert loaded['nested']['key'] == 'val'

    def test_saves_with_default_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {'cagr': 8.5}
            save_results_json(data, default_dir=Path(tmpdir))
            output = Path(tmpdir) / 'backtest_results.json'
            assert output.exists()

    def test_numpy_serialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'numpy_results.json')
            data = {'cagr': np.float64(8.5), 'count': np.int64(42), 'arr': np.array([1, 2, 3])}
            save_results_json(data, output_path=path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded['cagr'] == 8.5
            assert loaded['count'] == 42
            assert loaded['arr'] == [1, 2, 3]

    def test_no_path_no_dir(self):
        # Should not crash, just return
        save_results_json({'key': 'val'}, output_path=None, default_dir=None)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
