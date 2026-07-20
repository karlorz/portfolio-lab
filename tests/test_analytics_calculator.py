#!/usr/bin/env python3
"""
Tests for analytics calculator — data classes, drawdown series, rolling metrics,
benchmark comparison, and analytics report generation.
"""
import json

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from src.analytics.calculator import (
    DrawdownPoint, RollingMetrics, BenchmarkSeries, CrisisPeriod,
    AnalyticsCalculator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_perf_data(n_days=300, start_value=100000, daily_return=0.001,
                    start_date='2024-01-02'):
    """Create synthetic performance data as list of dicts."""
    data = []
    value = start_value
    d = datetime.strptime(start_date, '%Y-%m-%d')
    for i in range(n_days):
        # Skip weekends
        while d.weekday() >= 5:
            d += timedelta(days=1)
        data.append({
            'timestamp': d.strftime('%Y-%m-%dT15:30:00'),
            'total_value': round(value, 2),
        })
        value *= (1 + daily_return)
        d += timedelta(days=1)
    return data


def _make_perf_data_with_drawdown(n_days=300, start_value=100000,
                                   start_date='2024-01-02'):
    """Create perf data with a drawdown period in the middle."""
    data = []
    value = start_value
    d = datetime.strptime(start_date, '%Y-%m-%d')
    for i in range(n_days):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        data.append({
            'timestamp': d.strftime('%Y-%m-%dT15:30:00'),
            'total_value': round(value, 2),
        })
        if 100 <= i < 130:
            # Drawdown period: lose ~2% per day
            value *= 0.98
        elif 130 <= i < 160:
            # Recovery
            value *= 1.015
        else:
            value *= 1.001
        d += timedelta(days=1)
    return data


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestDrawdownPoint:
    def test_creation(self):
        dp = DrawdownPoint(
            date='2024-06-01', value=95000, peak=100000,
            drawdown=-0.05, drawdown_pct=-5.0, days_since_peak=10,
            is_recovery=False,
        )
        assert dp.drawdown == -0.05
        assert dp.drawdown_pct == -5.0
        assert dp.days_since_peak == 10

    def test_drawdown_negative(self):
        dp = DrawdownPoint(
            date='2024-06-01', value=90000, peak=100000,
            drawdown=-0.10, drawdown_pct=-10.0, days_since_peak=20,
            is_recovery=False,
        )
        assert dp.drawdown < 0


class TestRollingMetrics:
    def test_creation(self):
        rm = RollingMetrics(
            date='2024-06-01', sharpe_63d=1.2, sharpe_126d=0.9,
            sharpe_252d=0.7, volatility_63d=12.5, returns_63d=5.0,
        )
        assert rm.sharpe_63d == 1.2
        assert rm.volatility_63d == 12.5


class TestBenchmarkSeries:
    def test_creation(self):
        bs = BenchmarkSeries(
            symbol='SPY', dates=['2024-01-02'], values=[100.0],
            cagr=0.10, volatility=0.15, max_drawdown=-0.20,
        )
        assert bs.symbol == 'SPY'
        assert bs.cagr == 0.10


class TestCrisisPeriod:
    def test_creation(self):
        cp = CrisisPeriod(
            name='GFC 2008', start_date='2008-09-01', end_date='2009-03-31',
            description='Global Financial Crisis', spy_return=-0.47,
            portfolio_return=None,
        )
        assert cp.name == 'GFC 2008'
        assert cp.spy_return == -0.47


# ---------------------------------------------------------------------------
# AnalyticsCalculator tests
# ---------------------------------------------------------------------------

class TestAnalyticsCalculatorInit:
    def test_default_data_dir(self):
        calc = AnalyticsCalculator()
        assert 'portfolio-lab' in str(calc.data_dir) or 'data' in str(calc.data_dir)

    def test_custom_data_dir(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        assert calc.data_dir == tmp_path

    def test_performance_file_path(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        assert calc.performance_file == tmp_path / "performance.jsonl"


class TestLoadPerformanceData:
    def test_no_file(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        assert calc.load_performance_data() == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "performance.jsonl"
        f.write_text("")
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        assert calc.load_performance_data() == []

    def test_valid_data(self, tmp_path):
        f = tmp_path / "performance.jsonl"
        lines = [
            json.dumps({"timestamp": "2024-01-02", "total_value": 100000}),
            json.dumps({"timestamp": "2024-01-03", "total_value": 101000}),
        ]
        f.write_text("\n".join(lines))
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = calc.load_performance_data()
        assert len(data) == 2
        assert data[0]['total_value'] == 100000

    def test_invalid_json_skipped(self, tmp_path):
        f = tmp_path / "performance.jsonl"
        lines = [
            json.dumps({"timestamp": "2024-01-02", "total_value": 100000}),
            "not json",
            json.dumps({"timestamp": "2024-01-03", "total_value": 101000}),
        ]
        f.write_text("\n".join(lines))
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = calc.load_performance_data()
        assert len(data) == 2

    def test_mixed_key_formats(self, tmp_path):
        """Some entries use 'date' instead of 'timestamp'."""
        f = tmp_path / "performance.jsonl"
        lines = [
            json.dumps({"date": "2024-01-02", "total_value": 100000}),
            json.dumps({"timestamp": "2024-01-03", "total_value": 101000}),
        ]
        f.write_text("\n".join(lines))
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = calc.load_performance_data()
        assert len(data) == 2


# ---------------------------------------------------------------------------
# calculate_drawdown_series tests
# ---------------------------------------------------------------------------

class TestCalculateDrawdownSeries:
    def test_empty_data(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        assert calc.calculate_drawdown_series([]) == []

    def test_single_point(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = [{"timestamp": "2024-01-02T15:30:00", "total_value": 100000}]
        series = calc.calculate_drawdown_series(data)
        assert len(series) == 1
        assert series[0].is_recovery is True

    def test_steady_growth_no_drawdown(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=50, daily_return=0.001)
        series = calc.calculate_drawdown_series(data)
        # All points should be at peak (recovery) since value always rises
        for dp in series:
            assert dp.is_recovery is True

    def test_drawdown_detected(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data_with_drawdown(n_days=300)
        series = calc.calculate_drawdown_series(data)
        # Should have at least some points in drawdown
        in_dd = [dp for dp in series if dp.drawdown < -0.01]
        assert len(in_dd) > 0

    def test_drawdown_values_negative(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data_with_drawdown(n_days=300)
        series = calc.calculate_drawdown_series(data)
        dd_points = [dp for dp in series if dp.drawdown < 0]
        for dp in dd_points:
            assert dp.drawdown_pct < 0

    def test_peak_updates(self, tmp_path):
        """Peak should increase when value reaches new high."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=50, daily_return=0.002)
        series = calc.calculate_drawdown_series(data)
        # Peak should be non-decreasing for monotonically increasing data
        peaks = [dp.peak for dp in series]
        for i in range(1, len(peaks)):
            assert peaks[i] >= peaks[i - 1]

    def test_days_since_peak_resets(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data_with_drawdown(n_days=300)
        series = calc.calculate_drawdown_series(data)
        # After recovery, days_since_peak should reset
        last_dp = series[-1]
        assert last_dp.days_since_peak >= 0

    def test_uses_date_key_fallback(self, tmp_path):
        """Entries with 'date' instead of 'timestamp' should work."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = [
            {"date": "2024-01-02", "total_value": 100000},
            {"date": "2024-01-03", "total_value": 98000},
            {"date": "2024-01-04", "total_value": 99000},
        ]
        series = calc.calculate_drawdown_series(data)
        assert len(series) == 3


# ---------------------------------------------------------------------------
# calculate_max_drawdown tests
# ---------------------------------------------------------------------------

class TestCalculateMaxDrawdown:
    def test_empty_series(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        result = calc.calculate_max_drawdown([])
        assert result['max_drawdown'] == 0
        assert result['max_drawdown_date'] is None

    def test_no_drawdown(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=50, daily_return=0.001)
        series = calc.calculate_drawdown_series(data)
        result = calc.calculate_max_drawdown(series)
        # Max drawdown should be 0 or very small
        assert result['max_drawdown'] >= -0.1

    def test_drawdown_with_recovery(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data_with_drawdown(n_days=300)
        series = calc.calculate_drawdown_series(data)
        result = calc.calculate_max_drawdown(series)
        assert result['max_drawdown'] < -5  # At least -5%
        assert result['max_drawdown_date'] is not None
        assert result['peak_value'] > result['trough_value']

    def test_still_underwater(self, tmp_path):
        """If no recovery, underwater_days should be positive."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        # Data that declines and never recovers
        data = _make_perf_data(n_days=200, daily_return=-0.002)
        series = calc.calculate_drawdown_series(data)
        result = calc.calculate_max_drawdown(series)
        # First point is peak, so max_dd is from first to last
        if result['recovery_date'] is None:
            assert result['underwater_days'] > 0

    def test_has_peak_and_trough(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data_with_drawdown(n_days=300)
        series = calc.calculate_drawdown_series(data)
        result = calc.calculate_max_drawdown(series)
        assert result['peak_value'] > 0
        assert result['trough_value'] > 0


# ---------------------------------------------------------------------------
# calculate_rolling_sharpe tests
# ---------------------------------------------------------------------------

class TestCalculateRollingSharpe:
    def test_insufficient_data(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=10)
        result = calc.calculate_rolling_sharpe(window_days=63, performance_data=data)
        assert result == []

    def test_valid_data(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        result = calc.calculate_rolling_sharpe(window_days=63, performance_data=data)
        assert len(result) > 0
        for entry in result:
            assert 'date' in entry
            assert 'sharpe' in entry
            assert 'volatility' in entry

    def test_sharpe_bounded(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        # Use noisy data to avoid zero-volatility blowup
        np.random.seed(42)
        data = []
        value = 100000
        d = datetime(2024, 1, 2)
        for i in range(300):
            while d.weekday() >= 5:
                d += timedelta(days=1)
            data.append({
                'timestamp': d.strftime('%Y-%m-%dT15:30:00'),
                'total_value': round(value, 2),
            })
            value *= (1 + np.random.normal(0.001, 0.01))
            d += timedelta(days=1)
        result = calc.calculate_rolling_sharpe(window_days=63, performance_data=data)
        for entry in result:
            # Sharpe should be finite and reasonable
            assert abs(entry['sharpe']) < 50

    def test_negative_returns_negative_sharpe(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=-0.001)
        result = calc.calculate_rolling_sharpe(window_days=63, performance_data=data)
        for entry in result:
            assert entry['sharpe'] < 0

    def test_window_days_in_output(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        result = calc.calculate_rolling_sharpe(window_days=126, performance_data=data)
        for entry in result:
            assert entry['window_days'] == 126


# ---------------------------------------------------------------------------
# calculate_all_rolling_metrics tests
# ---------------------------------------------------------------------------

class TestCalculateAllRollingMetrics:
    def test_returns_three_windows(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        result = calc.calculate_all_rolling_metrics(performance_data=data)
        assert 'sharpe_63d' in result
        assert 'sharpe_126d' in result
        assert 'sharpe_252d' in result

    def test_insufficient_data_empty(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        result = calc.calculate_all_rolling_metrics(performance_data=[])
        for key, values in result.items():
            assert values == []

    def test_larger_window_fewer_points(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        result = calc.calculate_all_rolling_metrics(performance_data=data)
        # 63d window should have more points than 252d
        assert len(result['sharpe_63d']) >= len(result['sharpe_252d'])


# ---------------------------------------------------------------------------
# calculate_benchmark_comparison tests
# ---------------------------------------------------------------------------

class TestCalculateBenchmarkComparison:
    def test_empty_data(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        result = calc.calculate_benchmark_comparison(performance_data=[])
        assert result == {}

    def test_returns_portfolio_key(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=100, daily_return=0.001)
        result = calc.calculate_benchmark_comparison(performance_data=data)
        assert 'portfolio' in result

    def test_portfolio_metrics(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=100, daily_return=0.001)
        result = calc.calculate_benchmark_comparison(performance_data=data)
        p = result['portfolio']
        assert p['total_return'] > 0
        assert p['start_date'] is not None
        assert p['end_date'] is not None
        assert p['volatility'] >= 0


# ---------------------------------------------------------------------------
# generate_analytics_report tests
# ---------------------------------------------------------------------------

class TestGenerateAnalyticsReport:
    def test_no_data(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        report = calc.generate_analytics_report()
        assert report['status'] == 'no_data'

    def test_with_data(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        # Write perf data to file so report can load it
        f = tmp_path / "performance.jsonl"
        with open(f, 'w') as fh:
            for entry in data:
                fh.write(json.dumps(entry) + '\n')
        report = calc.generate_analytics_report()
        assert report['status'] == 'success'
        assert report['data_points'] == 300

    def test_report_has_drawdown(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        f = tmp_path / "performance.jsonl"
        with open(f, 'w') as fh:
            for entry in data:
                fh.write(json.dumps(entry) + '\n')
        report = calc.generate_analytics_report()
        assert 'drawdown' in report
        assert 'series' in report['drawdown']
        assert 'max_drawdown' in report['drawdown']

    def test_report_has_rolling_metrics(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        f = tmp_path / "performance.jsonl"
        with open(f, 'w') as fh:
            for entry in data:
                fh.write(json.dumps(entry) + '\n')
        report = calc.generate_analytics_report()
        assert 'rolling_metrics' in report

    def test_report_has_crisis_periods(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        f = tmp_path / "performance.jsonl"
        with open(f, 'w') as fh:
            for entry in data:
                fh.write(json.dumps(entry) + '\n')
        report = calc.generate_analytics_report()
        assert 'crisis_periods' in report
        assert len(report['crisis_periods']) == 3

    def test_all_null_crisis_portfolio_returns_emit_unavailable_metadata(self, tmp_path):
        """Global status may be success, but crisis comparison must not look complete."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        f = tmp_path / "performance.jsonl"
        with open(f, 'w') as fh:
            for entry in data:
                fh.write(json.dumps(entry) + '\n')
        report = calc.generate_analytics_report()
        assert report["status"] == "success"
        assert report["crisis_periods_status"] == "unavailable"
        assert report["crisis_periods_reason"] == "historical_simulation_unavailable"
        assert all(row["portfolio_return"] is None for row in report["crisis_periods"])
        assert all(row.get("portfolio_return_available") is False for row in report["crisis_periods"])

    def test_report_has_benchmark(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        f = tmp_path / "performance.jsonl"
        with open(f, 'w') as fh:
            for entry in data:
                fh.write(json.dumps(entry) + '\n')
        report = calc.generate_analytics_report()
        assert 'benchmark_comparison' in report


# ---------------------------------------------------------------------------
# CRISIS_PERIODS tests
# ---------------------------------------------------------------------------

class TestCrisisPeriods:
    def test_three_periods(self):
        assert len(AnalyticsCalculator.CRISIS_PERIODS) == 3

    def test_all_have_negative_spy_return(self):
        for cp in AnalyticsCalculator.CRISIS_PERIODS:
            assert cp.spy_return < 0

    def test_all_are_crisis_period_type(self):
        for cp in AnalyticsCalculator.CRISIS_PERIODS:
            assert isinstance(cp, CrisisPeriod)

    def test_date_ordering(self):
        for cp in AnalyticsCalculator.CRISIS_PERIODS:
            assert cp.start_date < cp.end_date


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_value_entries(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = [
            {"timestamp": "2024-01-02T15:30:00", "total_value": 0},
            {"timestamp": "2024-01-03T15:30:00", "total_value": 0},
        ]
        series = calc.calculate_drawdown_series(data)
        # Should not crash
        assert len(series) == 2

    def test_negative_value_entries(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = [
            {"timestamp": "2024-01-02T15:30:00", "total_value": -100},
        ]
        series = calc.calculate_drawdown_series(data)
        assert len(series) == 1

    def test_constant_value(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = [{"timestamp": f"2024-01-{d:02d}T15:30:00", "total_value": 100000}
                for d in range(2, 20)]
        series = calc.calculate_drawdown_series(data)
        # All drawdowns should be 0 for constant value
        for dp in series:
            assert dp.drawdown == 0

    def test_single_entry_rolling_sharpe(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = [{"timestamp": "2024-01-02T15:30:00", "total_value": 100000}]
        result = calc.calculate_rolling_sharpe(63, data)
        assert result == []

    def test_very_volatile_data(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.05)
        series = calc.calculate_drawdown_series(data)
        result = calc.calculate_max_drawdown(series)
        # Should not crash even with extreme volatility
        assert isinstance(result['max_drawdown'], float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_benchmark_comparison_portfolio_sharpe_not_null_with_returns():
    """Full-period portfolio.sharpe must be computed when NAV series has variance."""
    calc = AnalyticsCalculator()
    # Varying returns (not constant) so std > 0
    rng = np.random.RandomState(0)
    data = []
    value = 100000.0
    d = datetime(2024, 1, 2)
    for i in range(80):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        data.append({
            "timestamp": d.strftime("%Y-%m-%dT15:30:00"),
            "total_value": round(value, 2),
        })
        value *= 1.0 + 0.0008 + float(rng.normal(0, 0.005))
        d += timedelta(days=1)
    result = calc.calculate_benchmark_comparison(performance_data=data)
    port = result["portfolio"]
    assert port["sharpe"] is not None
    assert isinstance(port["sharpe"], (int, float))
    assert port["volatility"] > 0



def test_benchmark_comparison_sharpe_reason_when_flat():
    calc = AnalyticsCalculator()
    data = _make_perf_data(n_days=30, daily_return=0.0)
    result = calc.calculate_benchmark_comparison(performance_data=data)
    port = result["portfolio"]
    # zero variance → null sharpe with reason
    assert port.get("sharpe") is None
    assert port.get("sharpe_reason") in {
        "zero_return_variance",
        "insufficient_return_observations",
        "no_returns",
    }
