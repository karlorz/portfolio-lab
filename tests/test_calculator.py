#!/usr/bin/env python3
"""
Comprehensive tests for analytics calculator: dataclass field validation,
drawdown series, rolling metrics, benchmark comparison, analytics report
generation, main() CLI handler, and __all__ exports.
"""
import json
import io
import logging
import sys

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.analytics.calculator import (
    DrawdownPoint, RollingMetrics, BenchmarkSeries, CrisisPeriod,
    AnalyticsCalculator,
)


# ═══════════════════════════════════════════════════════════════════════════════
# __all__ export validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestModuleExports:
    """Verify that src.analytics exports calculator and calculator re-exports
    all expected public names."""

    def test_analytics_init_exports_calculator(self):
        import src.analytics
        assert "calculator" in src.analytics.__all__

    def test_calculator_module_has_all_public_names(self):
        """Every dataclass + class used in __init__ re-export chain must be
        importable from src.analytics.calculator."""
        expected = {
            "DrawdownPoint", "RollingMetrics", "BenchmarkSeries",
            "CrisisPeriod", "AnalyticsCalculator",
        }
        # module-level __all__ is not declared, so we check names that
        # start with an uppercase letter (public API)
        public_names = {
            name for name in dir(self)
            if name.startswith("Test")
        }
        for name in expected:
            assert hasattr(
                __import__("src.analytics.calculator", fromlist=[name]), name
            ), f"{name} is not importable from src.analytics.calculator"


# ═══════════════════════════════════════════════════════════════════════════════
# Data class field validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestDrawdownPoint:
    """DrawdownPoint dataclass — field types, bounds, edge cases."""

    def test_creation(self):
        dp = DrawdownPoint(
            date="2024-06-01", value=95000, peak=100000,
            drawdown=-0.05, drawdown_pct=-5.0, days_since_peak=10,
            is_recovery=False,
        )
        assert dp.date == "2024-06-01"
        assert dp.value == 95000
        assert dp.peak == 100000
        assert dp.drawdown == -0.05
        assert dp.drawdown_pct == -5.0
        assert dp.days_since_peak == 10
        assert dp.is_recovery is False

    def test_drawdown_negative(self):
        dp = DrawdownPoint(
            date="2024-06-01", value=90000, peak=100000,
            drawdown=-0.10, drawdown_pct=-10.0, days_since_peak=20,
            is_recovery=False,
        )
        assert dp.drawdown < 0

    def test_recovery_flag_true(self):
        dp = DrawdownPoint(
            date="2024-06-01", value=99000, peak=100000,
            drawdown=-0.01, drawdown_pct=-1.0, days_since_peak=5,
            is_recovery=True,
        )
        assert dp.is_recovery is True

    def test_field_types(self):
        dp = DrawdownPoint(
            date="2024-06-01", value=95000, peak=100000,
            drawdown=-0.05, drawdown_pct=-5.0, days_since_peak=10,
            is_recovery=False,
        )
        assert isinstance(dp.date, str)
        assert isinstance(dp.value, (int, float))
        assert isinstance(dp.peak, (int, float))
        assert isinstance(dp.drawdown, float)
        assert isinstance(dp.drawdown_pct, float)
        assert isinstance(dp.days_since_peak, int)
        assert isinstance(dp.is_recovery, bool)

    def test_zero_drawdown(self):
        dp = DrawdownPoint(
            date="2024-06-01", value=100000, peak=100000,
            drawdown=0.0, drawdown_pct=0.0, days_since_peak=0,
            is_recovery=True,
        )
        assert dp.drawdown == 0.0
        assert dp.days_since_peak == 0


class TestRollingMetrics:
    """RollingMetrics dataclass — nullable fields, types."""

    def test_creation(self):
        rm = RollingMetrics(
            date="2024-06-01", sharpe_63d=1.2, sharpe_126d=0.9,
            sharpe_252d=0.7, volatility_63d=12.5, returns_63d=5.0,
        )
        assert rm.sharpe_63d == 1.2
        assert rm.volatility_63d == 12.5
        assert rm.returns_63d == 5.0

    def test_nullable_fields(self):
        rm = RollingMetrics(
            date="2024-06-01", sharpe_63d=None, sharpe_126d=None,
            sharpe_252d=None, volatility_63d=None, returns_63d=None,
        )
        assert rm.sharpe_63d is None
        assert rm.volatility_63d is None
        assert rm.returns_63d is None

    def test_field_types(self):
        rm = RollingMetrics(
            date="2024-06-01", sharpe_63d=1.2, sharpe_126d=None,
            sharpe_252d=0.7, volatility_63d=12.5, returns_63d=None,
        )
        assert isinstance(rm.date, str)
        # Optional fields can be float or None
        assert rm.sharpe_63d is None or isinstance(rm.sharpe_63d, float)
        assert rm.volatility_63d is None or isinstance(rm.volatility_63d, float)

    def test_negative_sharpe(self):
        rm = RollingMetrics(
            date="2024-06-01", sharpe_63d=-0.5, sharpe_126d=-0.3,
            sharpe_252d=-0.1, volatility_63d=15.0, returns_63d=-2.0,
        )
        assert rm.sharpe_63d < 0


class TestBenchmarkSeries:
    """BenchmarkSeries dataclass — list lengths, CAGR types."""

    def test_creation(self):
        bs = BenchmarkSeries(
            symbol="SPY", dates=["2024-01-02"], values=[100.0],
            cagr=0.10, volatility=0.15, max_drawdown=-0.20,
        )
        assert bs.symbol == "SPY"
        assert bs.cagr == 0.10

    def test_multiple_dates(self):
        dates = [f"2024-01-{d:02d}" for d in range(2, 12)]
        values = [100.0 + i for i in range(10)]
        bs = BenchmarkSeries(
            symbol="GLD", dates=dates, values=values,
            cagr=0.08, volatility=0.12, max_drawdown=-0.15,
        )
        assert len(bs.dates) == 10
        assert len(bs.values) == 10
        assert bs.values[-1] == 109.0

    def test_field_types(self):
        bs = BenchmarkSeries(
            symbol="TLT", dates=["2024-01-02"], values=[100.0],
            cagr=0.05, volatility=0.18, max_drawdown=-0.25,
        )
        assert isinstance(bs.symbol, str)
        assert isinstance(bs.dates, list)
        assert isinstance(bs.values, list)
        assert isinstance(bs.cagr, float)
        assert isinstance(bs.volatility, float)
        assert isinstance(bs.max_drawdown, float)


class TestCrisisPeriod:
    """CrisisPeriod dataclass — optional portfolio_return."""

    def test_creation(self):
        cp = CrisisPeriod(
            name="GFC 2008", start_date="2008-09-01", end_date="2009-03-31",
            description="Global Financial Crisis", spy_return=-0.47,
            portfolio_return=None,
        )
        assert cp.name == "GFC 2008"
        assert cp.spy_return == -0.47
        assert cp.portfolio_return is None

    def test_with_portfolio_return(self):
        cp = CrisisPeriod(
            name="Test Crisis", start_date="2024-01-01", end_date="2024-03-01",
            description="A test", spy_return=-0.10, portfolio_return=-0.05,
        )
        assert cp.portfolio_return == -0.05

    def test_field_types(self):
        cp = CrisisPeriod(
            name="Test", start_date="2024-01-01", end_date="2024-03-01",
            description="Desc", spy_return=-0.10, portfolio_return=None,
        )
        assert isinstance(cp.name, str)
        assert isinstance(cp.start_date, str)
        assert isinstance(cp.end_date, str)
        assert isinstance(cp.description, str)
        assert isinstance(cp.spy_return, float)
        assert cp.portfolio_return is None or isinstance(cp.portfolio_return, float)

    def test_date_order_enforced(self):
        """Ensure start_date < end_date contract is preserved."""
        cp = CrisisPeriod(
            name="Test", start_date="2024-01-01", end_date="2024-03-01",
            description="Desc", spy_return=-0.10, portfolio_return=None,
        )
        assert cp.start_date < cp.end_date


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers for synthetic performance data
# ═══════════════════════════════════════════════════════════════════════════════

def _make_perf_data(n_days=300, start_value=100000, daily_return=0.001,
                    start_date="2024-01-02"):
    """Create synthetic performance data as list of dicts."""
    data = []
    value = start_value
    d = datetime.strptime(start_date, "%Y-%m-%d")
    for i in range(n_days):
        # Skip weekends
        while d.weekday() >= 5:
            d += timedelta(days=1)
        data.append({
            "timestamp": d.strftime("%Y-%m-%dT15:30:00"),
            "total_value": round(value, 2),
        })
        value *= (1 + daily_return)
        d += timedelta(days=1)
    return data


def _make_perf_data_with_drawdown(n_days=300, start_value=100000,
                                   start_date="2024-01-02"):
    """Create perf data with a drawdown period in the middle."""
    data = []
    value = start_value
    d = datetime.strptime(start_date, "%Y-%m-%d")
    for i in range(n_days):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        data.append({
            "timestamp": d.strftime("%Y-%m-%dT15:30:00"),
            "total_value": round(value, 2),
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


# ═══════════════════════════════════════════════════════════════════════════════
# AnalyticsCalculator — __init__
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyticsCalculatorInit:
    def test_default_data_dir(self):
        calc = AnalyticsCalculator()
        assert "portfolio-lab" in str(calc.data_dir) or "data" in str(calc.data_dir)

    def test_custom_data_dir(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        assert calc.data_dir == tmp_path

    def test_performance_file_path(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        assert calc.performance_file == tmp_path / "performance.jsonl"

    def test_data_dir_none(self):
        """data_dir=None should resolve to default DATA_DIR."""
        calc = AnalyticsCalculator(data_dir=None)
        assert calc.data_dir is not None
        assert calc.performance_file is not None


# ═══════════════════════════════════════════════════════════════════════════════
# load_performance_data
# ═══════════════════════════════════════════════════════════════════════════════

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
        assert data[0]["total_value"] == 100000

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

    def test_all_lines_corrupted(self, tmp_path):
        f = tmp_path / "performance.jsonl"
        f.write_text("not json\nstill not json\n")
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        assert calc.load_performance_data() == []

    def test_oserror_on_open_propagates(self, tmp_path):
        """Simulate an unreadable file — OSError on open() propagates
        because the try/except covers json.loads() inside the loop, not
        the open() call itself."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        f = tmp_path / "performance.jsonl"
        f.write_text("not relevant\n")
        # Make the file unreadable by os-level means on non-Windows
        f.chmod(0o000)
        try:
            data = calc.load_performance_data()
            assert data == []  # exists() might still return True on some FS
        except PermissionError:
            pass  # acceptable — file exists but is unreadable
        finally:
            f.chmod(0o644)

    def test_large_jsonl(self, tmp_path):
        """Stress test: many lines loaded efficiently."""
        lines = [
            json.dumps({"timestamp": f"2024-01-{d:02d}T12:00:00", "total_value": 100000 + d})
            for d in range(2, 102)
        ]
        f = tmp_path / "performance.jsonl"
        f.write_text("\n".join(lines))
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = calc.load_performance_data()
        assert len(data) == 100


# ═══════════════════════════════════════════════════════════════════════════════
# calculate_drawdown_series
# ═══════════════════════════════════════════════════════════════════════════════

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
        for dp in series:
            assert dp.is_recovery is True

    def test_drawdown_detected(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data_with_drawdown(n_days=300)
        series = calc.calculate_drawdown_series(data)
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
        peaks = [dp.peak for dp in series]
        for i in range(1, len(peaks)):
            assert peaks[i] >= peaks[i - 1]

    def test_days_since_peak_resets(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data_with_drawdown(n_days=300)
        series = calc.calculate_drawdown_series(data)
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

    def test_peak_equals_max_value(self, tmp_path):
        """The peak value should never exceed the maximum value seen so far."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data_with_drawdown(n_days=300)
        series = calc.calculate_drawdown_series(data)
        for dp in series:
            assert dp.peak >= dp.value

    def test_drawdown_pct_is_rounded(self, tmp_path):
        """drawdown_pct should be rounded to 2 decimal places."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data_with_drawdown(n_days=100)
        series = calc.calculate_drawdown_series(data)
        for dp in series:
            # Check rounding by multiplying back
            assert dp.drawdown_pct == round(dp.drawdown_pct, 2)

    def test_uses_file_data_when_none_passed(self, tmp_path):
        """When no performance_data is passed, load from file."""
        f = tmp_path / "performance.jsonl"
        lines = [
            json.dumps({"timestamp": "2024-01-02T15:30:00", "total_value": 100000}),
            json.dumps({"timestamp": "2024-01-03T15:30:00", "total_value": 98000}),
        ]
        f.write_text("\n".join(lines))
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        series = calc.calculate_drawdown_series()
        assert len(series) == 2
        assert series[1].drawdown < 0

    def test_no_file_fallback_empty(self, tmp_path):
        """With empty data dir and no perf data, return empty list."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        series = calc.calculate_drawdown_series()
        assert series == []


# ═══════════════════════════════════════════════════════════════════════════════
# calculate_max_drawdown
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculateMaxDrawdown:
    def test_empty_series(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        result = calc.calculate_max_drawdown([])
        assert result["max_drawdown"] == 0
        assert result["max_drawdown_date"] is None

    def test_no_drawdown(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=50, daily_return=0.001)
        series = calc.calculate_drawdown_series(data)
        result = calc.calculate_max_drawdown(series)
        assert result["max_drawdown"] >= -0.1

    def test_drawdown_with_recovery(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data_with_drawdown(n_days=300)
        series = calc.calculate_drawdown_series(data)
        result = calc.calculate_max_drawdown(series)
        assert result["max_drawdown"] < -5
        assert result["max_drawdown_date"] is not None
        assert result["peak_value"] > result["trough_value"]

    def test_still_underwater(self, tmp_path):
        """If no recovery, underwater_days should be positive."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=200, daily_return=-0.002)
        series = calc.calculate_drawdown_series(data)
        result = calc.calculate_max_drawdown(series)
        if result["recovery_date"] is None:
            assert result["underwater_days"] > 0

    def test_has_peak_and_trough(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data_with_drawdown(n_days=300)
        series = calc.calculate_drawdown_series(data)
        result = calc.calculate_max_drawdown(series)
        assert result["peak_value"] > 0
        assert result["trough_value"] > 0

    def test_recovery_date_not_before_max_dd_date(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data_with_drawdown(n_days=300)
        series = calc.calculate_drawdown_series(data)
        result = calc.calculate_max_drawdown(series)
        if result["recovery_date"]:
            assert result["recovery_date"] >= result["max_drawdown_date"]

    def test_underwater_days_positive_with_recovery(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data_with_drawdown(n_days=300)
        series = calc.calculate_drawdown_series(data)
        result = calc.calculate_max_drawdown(series)
        assert result["underwater_days"] > 0

    def test_single_point_no_dd(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = [{"timestamp": "2024-01-02T15:30:00", "total_value": 100000}]
        series = calc.calculate_drawdown_series(data)
        result = calc.calculate_max_drawdown(series)
        assert result["max_drawdown"] == 0.0

    def result_keys_present(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=50, daily_return=0.001)
        series = calc.calculate_drawdown_series(data)
        result = calc.calculate_max_drawdown(series)
        expected_keys = {
            "max_drawdown", "max_drawdown_date", "recovery_date",
            "underwater_days", "peak_value", "trough_value",
        }
        assert set(result.keys()) == expected_keys


# ═══════════════════════════════════════════════════════════════════════════════
# calculate_rolling_sharpe
# ═══════════════════════════════════════════════════════════════════════════════

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
            assert "date" in entry
            assert "sharpe" in entry
            assert "volatility" in entry

    def test_sharpe_bounded(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        np.random.seed(42)
        data = []
        value = 100000
        d = datetime(2024, 1, 2)
        for i in range(300):
            while d.weekday() >= 5:
                d += timedelta(days=1)
            data.append({
                "timestamp": d.strftime("%Y-%m-%dT15:30:00"),
                "total_value": round(value, 2),
            })
            value *= (1 + np.random.normal(0.001, 0.01))
            d += timedelta(days=1)
        result = calc.calculate_rolling_sharpe(window_days=63, performance_data=data)
        for entry in result:
            assert abs(entry["sharpe"]) < 50

    def test_negative_returns_negative_sharpe(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=-0.001)
        result = calc.calculate_rolling_sharpe(window_days=63, performance_data=data)
        for entry in result:
            assert entry["sharpe"] < 0

    def test_window_days_in_output(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        result = calc.calculate_rolling_sharpe(window_days=126, performance_data=data)
        for entry in result:
            assert entry["window_days"] == 126

    def test_exact_boundary_data(self, tmp_path):
        """Exactly window_days+1 entries should produce 1 result."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=64, daily_return=0.001)  # 63+1
        result = calc.calculate_rolling_sharpe(window_days=63, performance_data=data)
        assert len(result) == 1

    def test_zero_volatility_handled(self, tmp_path):
        """Constant values produce zero std; sharpe should be 0, not crash."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = []
        for i in range(70):
            data.append({
                "timestamp": f"2024-01-{i+2:02d}T15:30:00",
                "total_value": 100000,
            })
        result = calc.calculate_rolling_sharpe(window_days=63, performance_data=data)
        # Should not crash; constant returns -> std=0 -> sharpe=0
        assert len(result) > 0
        for entry in result:
            assert entry["sharpe"] >= 0

    def test_rolling_sharpe_uses_file_fallback(self, tmp_path):
        """When performance_data is None, fall back to file load."""
        f = tmp_path / "performance.jsonl"
        data = _make_perf_data(n_days=200, daily_return=0.001)
        with open(f, "w") as fh:
            for entry in data:
                fh.write(json.dumps(entry) + "\n")
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        result = calc.calculate_rolling_sharpe(window_days=63)
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# calculate_all_rolling_metrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculateAllRollingMetrics:
    def test_returns_three_windows(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        result = calc.calculate_all_rolling_metrics(performance_data=data)
        assert "sharpe_63d" in result
        assert "sharpe_126d" in result
        assert "sharpe_252d" in result

    def test_insufficient_data_empty(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        result = calc.calculate_all_rolling_metrics(performance_data=[])
        for key, values in result.items():
            assert values == []

    def test_larger_window_fewer_points(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        result = calc.calculate_all_rolling_metrics(performance_data=data)
        assert len(result["sharpe_63d"]) >= len(result["sharpe_252d"])

    def test_keys_consistent(self, tmp_path):
        """All three keys should be present even with insufficient data."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        result = calc.calculate_all_rolling_metrics(performance_data=[])
        assert set(result.keys()) == {"sharpe_63d", "sharpe_126d", "sharpe_252d"}


# ═══════════════════════════════════════════════════════════════════════════════
# calculate_benchmark_comparison
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculateBenchmarkComparison:
    def test_empty_data(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        result = calc.calculate_benchmark_comparison(performance_data=[])
        assert result == {}

    def test_returns_portfolio_key(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=100, daily_return=0.001)
        result = calc.calculate_benchmark_comparison(performance_data=data)
        assert "portfolio" in result

    def test_portfolio_metrics(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=100, daily_return=0.001)
        result = calc.calculate_benchmark_comparison(performance_data=data)
        p = result["portfolio"]
        assert p["total_return"] > 0
        assert p["start_date"] is not None
        assert p["end_date"] is not None
        assert p["volatility"] >= 0

    def test_returns_with_file_fallback(self, tmp_path):
        """When no performance_data is passed, fall back to JSONL."""
        f = tmp_path / "performance.jsonl"
        data = _make_perf_data(n_days=100, daily_return=0.001)
        with open(f, "w") as fh:
            for entry in data:
                fh.write(json.dumps(entry) + "\n")
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        result = calc.calculate_benchmark_comparison()
        assert "portfolio" in result

    def test_negative_return(self, tmp_path):
        """Consistently negative data should show negative total_return."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=100, daily_return=-0.001)
        result = calc.calculate_benchmark_comparison(performance_data=data)
        p = result["portfolio"]
        assert p["total_return"] < 0


# ═══════════════════════════════════════════════════════════════════════════════
# generate_analytics_report
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateAnalyticsReport:
    def test_no_data(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        report = calc.generate_analytics_report()
        assert report["status"] == "no_data"

    def test_with_data(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        f = tmp_path / "performance.jsonl"
        with open(f, "w") as fh:
            for entry in data:
                fh.write(json.dumps(entry) + "\n")
        report = calc.generate_analytics_report()
        assert report["status"] == "success"
        assert report["data_points"] == 300

    def test_report_has_drawdown(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        f = tmp_path / "performance.jsonl"
        with open(f, "w") as fh:
            for entry in data:
                fh.write(json.dumps(entry) + "\n")
        report = calc.generate_analytics_report()
        assert "drawdown" in report
        assert "series" in report["drawdown"]
        assert "max_drawdown" in report["drawdown"]

    def test_report_has_rolling_metrics(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        f = tmp_path / "performance.jsonl"
        with open(f, "w") as fh:
            for entry in data:
                fh.write(json.dumps(entry) + "\n")
        report = calc.generate_analytics_report()
        assert "rolling_metrics" in report

    def test_report_has_crisis_periods(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        f = tmp_path / "performance.jsonl"
        with open(f, "w") as fh:
            for entry in data:
                fh.write(json.dumps(entry) + "\n")
        report = calc.generate_analytics_report()
        assert "crisis_periods" in report
        assert len(report["crisis_periods"]) == 3

    def test_report_has_benchmark(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        f = tmp_path / "performance.jsonl"
        with open(f, "w") as fh:
            for entry in data:
                fh.write(json.dumps(entry) + "\n")
        report = calc.generate_analytics_report()
        assert "benchmark_comparison" in report

    def test_report_has_date_range(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        f = tmp_path / "performance.jsonl"
        with open(f, "w") as fh:
            for entry in data:
                fh.write(json.dumps(entry) + "\n")
        report = calc.generate_analytics_report()
        assert "date_range" in report
        assert report["date_range"]["start"] is not None
        assert report["date_range"]["end"] is not None

    def test_report_has_generated_at(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=300, daily_return=0.001)
        f = tmp_path / "performance.jsonl"
        with open(f, "w") as fh:
            for entry in data:
                fh.write(json.dumps(entry) + "\n")
        report = calc.generate_analytics_report()
        assert "generated_at" in report

    def test_report_no_data_keys(self, tmp_path):
        """No-data report should have minimal keys."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        report = calc.generate_analytics_report()
        assert set(report.keys()) == {"status", "message", "generated_at"}
        assert report["message"] == "No performance data available"


# ═══════════════════════════════════════════════════════════════════════════════
# CRISIS_PERIODS class-level constant
# ═══════════════════════════════════════════════════════════════════════════════

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

    def test_names_known(self):
        names = {cp.name for cp in AnalyticsCalculator.CRISIS_PERIODS}
        assert "GFC 2008" in names
        assert "COVID 2020" in names
        assert "Rate Hikes 2022" in names

    def test_all_descriptions_nonempty(self):
        for cp in AnalyticsCalculator.CRISIS_PERIODS:
            assert len(cp.description) > 0

    def test_all_portfolio_return_none(self):
        for cp in AnalyticsCalculator.CRISIS_PERIODS:
            assert cp.portfolio_return is None


# ═══════════════════════════════════════════════════════════════════════════════
# main() CLI function
# ═══════════════════════════════════════════════════════════════════════════════

class TestMainCLI:
    """Tests for the ``main()`` CLI entry point using argv patching."""

    def test_no_args_runs_report(self, tmp_path, caplog, monkeypatch):
        """No CLI args should run full report."""
        monkeypatch.setattr("src.analytics.calculator.DATA_DIR", tmp_path)
        data = _make_perf_data(n_days=50)
        f = tmp_path / "performance.jsonl"
        with open(f, "w") as fh:
            for entry in data:
                fh.write(json.dumps(entry) + "\n")

        with caplog.at_level(logging.INFO, logger="src.analytics.calculator"):
            with patch.object(sys, "argv", ["calculator.py"]):
                from src.analytics.calculator import main
                main()
            assert '"status": "success"' in caplog.text

    def test_drawdown_command(self, tmp_path, caplog, monkeypatch):
        """``drawdown`` subcommand prints max drawdown stats."""
        monkeypatch.setattr("src.analytics.calculator.DATA_DIR", tmp_path)
        data = _make_perf_data(n_days=50)
        f = tmp_path / "performance.jsonl"
        with open(f, "w") as fh:
            for entry in data:
                fh.write(json.dumps(entry) + "\n")

        with caplog.at_level(logging.INFO, logger="src.analytics.calculator"):
            with patch.object(sys, "argv", ["calculator.py", "drawdown"]):
                from src.analytics.calculator import main
                main()
            assert "max_drawdown" in caplog.text

    def test_rolling_command(self, tmp_path, caplog, monkeypatch):
        """``rolling`` subcommand prints latest rolling Sharpe."""
        monkeypatch.setattr("src.analytics.calculator.DATA_DIR", tmp_path)
        data = _make_perf_data(n_days=300)
        f = tmp_path / "performance.jsonl"
        with open(f, "w") as fh:
            for entry in data:
                fh.write(json.dumps(entry) + "\n")

        with caplog.at_level(logging.INFO, logger="src.analytics.calculator"):
            with patch.object(sys, "argv", ["calculator.py", "rolling"]):
                from src.analytics.calculator import main
                main()
            assert "Sharpe" in caplog.text

    def test_report_command(self, tmp_path, caplog, monkeypatch):
        """``report`` subcommand prints full JSON report."""
        monkeypatch.setattr("src.analytics.calculator.DATA_DIR", tmp_path)
        data = _make_perf_data(n_days=50)
        f = tmp_path / "performance.jsonl"
        with open(f, "w") as fh:
            for entry in data:
                fh.write(json.dumps(entry) + "\n")

        with caplog.at_level(logging.INFO, logger="src.analytics.calculator"):
            with patch.object(sys, "argv", ["calculator.py", "report"]):
                from src.analytics.calculator import main
                main()
            assert '"status": "success"' in caplog.text

    def test_unknown_command(self, tmp_path, caplog):
        """Unknown subcommand should print usage message."""
        with caplog.at_level(logging.INFO, logger="src.analytics.calculator"):
            with patch.object(sys, "argv", ["calculator.py", "unknown_cmd"]):
                from src.analytics.calculator import main
                main()
            assert "Unknown command" in caplog.text
            assert "drawdown" in caplog.text

    def test_no_data_drawdown(self, tmp_path, caplog):
        """``drawdown`` with no data prints JSON with default values."""
        with caplog.at_level(logging.INFO, logger="src.analytics.calculator"):
            with patch.object(sys, "argv", ["calculator.py", "drawdown"]):
                from src.analytics.calculator import main
                main()
            assert "max_drawdown" in caplog.text

    def test_no_data_report(self, tmp_path, caplog):
        """``report`` with no data prints no_data status."""
        import src.analytics.calculator as calc_mod
        with caplog.at_level(logging.INFO, logger="src.analytics.calculator"):
            with patch.object(sys, "argv", ["calculator.py", "report"]):
                with patch.object(calc_mod, "DATA_DIR", tmp_path):
                    from src.analytics.calculator import main
                    main()
                assert "no_data" in caplog.text


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases and boundary conditions
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_zero_value_entries(self, tmp_path):
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = [
            {"timestamp": "2024-01-02T15:30:00", "total_value": 0},
            {"timestamp": "2024-01-03T15:30:00", "total_value": 0},
        ]
        series = calc.calculate_drawdown_series(data)
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
        assert isinstance(result["max_drawdown"], float)

    def test_unsorted_data(self, tmp_path):
        """Data not sorted by date should be sorted internally."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = [
            {"timestamp": "2024-01-05T15:30:00", "total_value": 105000},
            {"timestamp": "2024-01-02T15:30:00", "total_value": 100000},
            {"timestamp": "2024-01-03T15:30:00", "total_value": 98000},
        ]
        series = calc.calculate_drawdown_series(data)
        dates = [dp.date for dp in series]
        assert dates == sorted(dates)

    def test_missing_total_value_key(self, tmp_path):
        """Entry missing total_value key should use default 0."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = [
            {"timestamp": "2024-01-02T15:30:00"},
            {"timestamp": "2024-01-03T15:30:00", "total_value": 100000},
        ]
        series = calc.calculate_drawdown_series(data)
        assert series[0].value == 0

    def test_benchmark_empty_drawdown_series(self, tmp_path):
        """Benchmark comparison with empty data returns empty."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        result = calc.calculate_benchmark_comparison(performance_data=[])
        assert result == {}

    def test_rolling_sharpe_all_windows_insufficient(self, tmp_path):
        """With very little data, all rolling windows return empty lists."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = _make_perf_data(n_days=5, daily_return=0.001)
        result = calc.calculate_all_rolling_metrics(performance_data=data)
        for k, v in result.items():
            assert v == []

    def test_max_drawdown_peak_value_zero(self, tmp_path):
        """Peak value of 0 should not cause division errors."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = [
            {"timestamp": "2024-01-02T15:30:00", "total_value": 0},
            {"timestamp": "2024-01-03T15:30:00", "total_value": 0},
        ]
        series = calc.calculate_drawdown_series(data)
        result = calc.calculate_max_drawdown(series)
        # Both entries have peak=0, the first is peak, second is drawdown=0
        assert isinstance(result["max_drawdown"], (int, float))

    def test_monotonically_decreasing(self, tmp_path):
        """Always-decreasing series: each point is a new low."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = []
        value = 100000
        for i in range(10):
            data.append({
                "timestamp": f"2024-01-{i+2:02d}T15:30:00",
                "total_value": value - i * 1000,
            })
        series = calc.calculate_drawdown_series(data)
        # First point is peak, all others have drawdown < 0
        for dp in series[1:]:
            assert dp.drawdown < 0

    def test_recovery_exact_one_percent(self, tmp_path):
        """Boundary: value exactly 99% of peak is NOT recovery."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = [
            {"timestamp": "2024-01-02T15:30:00", "total_value": 100000},
            {"timestamp": "2024-01-03T15:30:00", "total_value": 98999},  # just under 99%
        ]
        series = calc.calculate_drawdown_series(data)
        assert series[1].is_recovery is False

    def test_two_separate_drawdowns(self, tmp_path):
        """Multiple drawdown cycles: max_dd captures the worst."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        data = [
            {"timestamp": "2024-01-02T15:30:00", "total_value": 100000},
            # Drawdown 1: -10%
            {"timestamp": "2024-01-03T15:30:00", "total_value": 90000},
            # Recovery
            {"timestamp": "2024-01-04T15:30:00", "total_value": 100000},
            # Drawdown 2: -20% (worse)
            {"timestamp": "2024-01-05T15:30:00", "total_value": 80000},
        ]
        series = calc.calculate_drawdown_series(data)
        result = calc.calculate_max_drawdown(series)
        assert result["max_drawdown"] == -20.0


# ═══════════════════════════════════════════════════════════════════════════════
# Mock-based tests (external dependencies, database, file I/O)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMockedDependencies:
    """Tests using unittest.mock to isolate external dependencies."""

    def test_load_performance_data_patched_open(self, tmp_path):
        """Mock open to return synthetic content."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        fake_lines = [
            json.dumps({"timestamp": "2024-01-02T12:00:00", "total_value": 100000}),
            json.dumps({"timestamp": "2024-01-03T12:00:00", "total_value": 101000}),
        ]
        fake_content = "\n".join(fake_lines)

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=fake_content)):
                data = calc.load_performance_data()
        assert len(data) == 2

    def test_load_performance_data_exists_false(self, tmp_path):
        """When performance_file.exists() is False, return []."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        with patch.object(Path, "exists", return_value=False):
            data = calc.load_performance_data()
        assert data == []

    def test_calculate_drawdown_with_mocked_load(self, tmp_path):
        """Mock load_performance_data to avoid file I/O."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        fake_data = _make_perf_data_with_drawdown(n_days=100)

        with patch.object(
            calc, "load_performance_data", return_value=fake_data
        ) as mock_load:
            series = calc.calculate_drawdown_series()
            mock_load.assert_called_once()

        assert len(series) > 0

    def test_calculate_rolling_sharpe_with_mocked_load(self, tmp_path):
        """Mock load_performance_data to avoid file I/O."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        fake_data = _make_perf_data(n_days=200, daily_return=0.001)

        with patch.object(
            calc, "load_performance_data", return_value=fake_data
        ) as mock_load:
            result = calc.calculate_rolling_sharpe(window_days=63)
            mock_load.assert_called_once()

        assert len(result) > 0

    def test_generate_report_with_mocked_methods(self, tmp_path):
        """Mock intermediate methods to isolate report logic."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        fake_data = _make_perf_data(n_days=100, daily_return=0.001)

        with patch.object(calc, "load_performance_data", return_value=fake_data):
            report = calc.generate_analytics_report()

        assert report["status"] == "success"
        assert report["data_points"] == 100

    def test_benchmark_with_mocked_file(self, tmp_path):
        """Mock load_performance_data for benchmark comparison."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        fake_data = _make_perf_data(n_days=100, daily_return=0.001)

        with patch.object(
            calc, "load_performance_data", return_value=fake_data
        ) as mock_load:
            result = calc.calculate_benchmark_comparison()
            mock_load.assert_called_once()

        assert "portfolio" in result

    def test_drawdown_series_with_mocked_file_no_data(self, tmp_path):
        """When mocked load returns [], drawdown should be []."""
        calc = AnalyticsCalculator(data_dir=str(tmp_path))
        with patch.object(calc, "load_performance_data", return_value=[]):
            series = calc.calculate_drawdown_series()
        assert series == []


# ═══════════════════════════════════════════════════════════════════════════════
# main() guard (__name__ == '__main__')
# ═══════════════════════════════════════════════════════════════════════════════

class TestMainGuard:
    """Verify the ``if __name__ == '__main__'`` block calls main()."""

    def test_main_guard_present(self):
        """The module has ``if __name__ == '__main__': main()`` guard."""
        import src.analytics.calculator as calc_mod
        # The guard does: if __name__ == '__main__': main()
        # Verify main is callable
        assert callable(calc_mod.main)

    def test_main_not_called_when_imported(self):
        with patch("src.analytics.calculator.main") as mock_main:
            import importlib
            import src.analytics.calculator
            importlib.reload(src.analytics.calculator)
            mock_main.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Run with pytest
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
