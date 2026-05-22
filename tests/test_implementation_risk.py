#!/usr/bin/env python3
"""Tests for v8.05 Implementation Risk Quantification."""

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitor.implementation_risk import (
    METRICS_CONFIG,
    GRADE_THRESHOLDS,
    MetricGap,
    ImplementationRiskReport,
    _normalize_backtest,
    load_backtest_results,
    load_paper_trading_data,
    _extract_returns,
    _count_trading_days,
    bootstrap_confidence,
    compute_gap,
    assess_implementation_risk,
    _score_to_grade,
    _format_pct,
    _format_val,
)


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_backtest_file(tmp_path):
    """Create a sample backtest results JSON file."""
    data = {
        "strategy": "Test Strategy",
        "sharpe_ratio": 0.93,
        "cagr": 0.107,
        "max_drawdown": 0.257,
        "volatility": 0.115,
        "start_date": "2006-01-01",
        "end_date": "2026-05-08",
        "trading_days": 5097,
    }
    path = tmp_path / "backtest_results.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def sample_performance_log(tmp_path):
    """Create a sample performance.jsonl with ~30 days of daily returns."""
    path = tmp_path / "performance.jsonl"
    base = datetime.now(timezone.utc) - timedelta(days=35)

    with open(path, "w") as f:
        for i in range(30):
            ts = base + timedelta(days=i)
            ret = np.random.normal(0.0005, 0.008)  # realistic daily return
            entry = {
                "timestamp": ts.isoformat(),
                "total_value": 100000 * (1 + ret),
                "cash": 0.0,
                "daily_return": ret,
                "positions_count": 3,
                "mode": "paper",
            }
            f.write(json.dumps(entry) + "\n")
    return path


@pytest.fixture
def mock_data_dirs(tmp_path, sample_backtest_file, sample_performance_log):
    """Mock all data directories with test files."""
    # Create backtest_results directory
    bt_dir = tmp_path / "backtest_results"
    bt_dir.mkdir()
    shutil.copy(sample_backtest_file, bt_dir / "combined_backtest_results.json")
    
    return bt_dir, sample_performance_log


# ─── Tests: Helper Functions ────────────────────────────────────────────


class TestNormalizeBacktest:
    def test_sharpe_normalization(self):
        data = {"sharpe_ratio": 0.93}
        result = _normalize_backtest(data)
        assert result["sharpe"] == 0.93

    def test_cagr_normalization(self):
        data = {"cagr": 0.107}
        result = _normalize_backtest(data)
        assert result["cagr"] == 0.107

    def test_max_drawdown_normalization(self):
        data = {"max_dd": 0.257}
        result = _normalize_backtest(data)
        assert result["max_drawdown"] == 0.257

    def test_volatility_normalization(self):
        data = {"ann_vol": 0.115}
        result = _normalize_backtest(data)
        assert result["volatility"] == 0.115

    def test_all_names_map(self):
        data = {"sharpe": 0.9, "cagr": 0.1, "max_drawdown": 0.25, "volatility": 0.12}
        result = _normalize_backtest(data)
        assert result["sharpe"] == 0.9
        assert result["cagr"] == 0.1
        assert result["max_drawdown"] == 0.25
        assert result["volatility"] == 0.12

    def test_partial_data(self):
        data = {"sharpe_ratio": 0.8}
        result = _normalize_backtest(data)
        assert "sharpe" in result
        assert "cagr" not in result

    def test_empty_data(self):
        result = _normalize_backtest({})
        assert result == {}


class TestLoadBacktestResults:
    def test_loads_first_available(self, sample_backtest_file, tmp_path):
        """Should load the first available file from the list."""
        files = [sample_backtest_file]
        result = load_backtest_results(files)
        assert result is not None
        assert result["sharpe"] == 0.93

    def test_returns_none_when_no_files(self, tmp_path):
        """Should return None when no files exist."""
        files = [tmp_path / "nonexistent.json"]
        result = load_backtest_results(files)
        assert result is None

    def test_skips_corrupted_files(self, tmp_path):
        """Should skip bad JSON files."""
        bad_file = tmp_path / "bad.json"
        with open(bad_file, "w") as f:
            f.write("not json")
        good_file = tmp_path / "good.json"
        with open(good_file, "w") as f:
            json.dump({"sharpe_ratio": 0.8}, f)

        result = load_backtest_results([bad_file, good_file])
        assert result is not None
        assert result["sharpe"] == 0.8


class TestExtractReturns:
    def test_returns_empty_for_missing_file(self, tmp_path):
        """Should return empty array for missing file."""
        missing_path = tmp_path / "nonexistent.json"
        with patch("src.monitor.implementation_risk.PERFORMANCE_LOG", missing_path):
            result = _extract_returns(days=30)
        assert len(result) == 0

    def test_deduplicates_to_daily(self, tmp_path):
        """Should deduplicate to one return per day."""
        path = tmp_path / "perf.jsonl"
        base = datetime.now(timezone.utc) - timedelta(days=5)
        with open(path, "w") as f:
            for i in range(3):
                ts = base + timedelta(days=i)
                f.write(json.dumps({"timestamp": ts.isoformat(), "daily_return": 0.01}) + "\n")
                # Add duplicate intra-day entry
                f.write(json.dumps({"timestamp": ts.isoformat(), "daily_return": 0.02}) + "\n")

        with patch("src.monitor.implementation_risk.PERFORMANCE_LOG", path):
            result = _extract_returns(days=30)

        # Should have 3 unique days, each taking the last entry
        assert len(result) == 3

    def test_filters_by_days(self, tmp_path):
        """Should only include returns within the specified window."""
        path = tmp_path / "perf.jsonl"
        now = datetime.now(timezone.utc)
        with open(path, "w") as f:
            # An old entry
            f.write(json.dumps({
                "timestamp": (now - timedelta(days=60)).isoformat(),
                "daily_return": 0.01,
            }) + "\n")
            # Recent entries on different days (to avoid deduplication)
            f.write(json.dumps({
                "timestamp": (now - timedelta(days=2)).isoformat(),
                "daily_return": 0.02,
            }) + "\n")
            f.write(json.dumps({
                "timestamp": (now - timedelta(hours=1)).isoformat(),
                "daily_return": 0.03,
            }) + "\n")

        with patch("src.monitor.implementation_risk.PERFORMANCE_LOG", path):
            result = _extract_returns(days=30)

        assert len(result) == 2  # Only recent 2 entries (on different days)

    def test_handles_malformed_lines(self, tmp_path):
        """Should skip malformed JSON lines gracefully."""
        path = tmp_path / "perf.jsonl"
        with open(path, "w") as f:
            f.write("not json\n")
            f.write(json.dumps({"timestamp": "2026-05-17T00:00:00", "daily_return": 0.01}) + "\n")
            f.write("also not json\n")

        with patch("src.monitor.implementation_risk.PERFORMANCE_LOG", path):
            result = _extract_returns(days=30)

        assert len(result) == 1


class TestBootstrapConfidence:
    def test_returns_three_values(self):
        lower, upper, std = bootstrap_confidence(0.93)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert isinstance(std, float)

    def test_ci_contains_true_value(self):
        for _ in range(10):
            lower, upper, _ = bootstrap_confidence(0.93)
            assert lower <= 0.93 <= upper

    def test_ci_narrows_with_more_data(self):
        _, _, std_short = bootstrap_confidence(0.93, n_simulations=100, n_years=1.0)
        _, _, std_long = bootstrap_confidence(0.93, n_simulations=100, n_years=10.0)
        assert std_long < std_short

    def test_units_are_consistent(self):
        lower, upper, _ = bootstrap_confidence(0.10, n_years=5.0)
        assert lower < upper


class TestComputeGap:
    def test_zero_gap(self):
        gap, gap_pct, z, grade, detail, within = compute_gap(1.0, 1.0, 0.1)
        assert abs(gap) < 1e-10
        assert grade in ("A",)

    def test_small_gap(self):
        gap, gap_pct, z, grade, detail, within = compute_gap(1.0, 1.1, 0.5)
        assert within  # z=0.2 => grade A

    def test_large_gap(self):
        gap, gap_pct, z, grade, detail, within = compute_gap(1.0, 2.0, 0.1)
        assert not within  # z=10 => grade F
        assert grade == "F"

    def test_grade_b_for_moderate_gap(self):
        gap, gap_pct, z, grade, detail, within = compute_gap(1.0, 1.25, 0.3)
        # z = 0.833 => grade B (within 1 sigma)
        assert grade == "B"
        assert within

    def test_grade_c(self):
        gap, gap_pct, z, grade, detail, within = compute_gap(1.0, 1.4, 0.3)
        # z = 1.333 => grade C (within 1.5 sigma)
        assert grade == "C"
        assert not within


class TestScoreToGrade:
    def test_a_grade(self):
        assert _score_to_grade(3.8) == "A"

    def test_b_grade(self):
        assert _score_to_grade(3.0) == "B"

    def test_c_grade(self):
        assert _score_to_grade(2.0) == "C"

    def test_d_grade(self):
        assert _score_to_grade(1.0) == "D"

    def test_f_grade(self):
        assert _score_to_grade(0.2) == "F"


class TestFormatHelpers:
    def test_format_pct(self):
        assert _format_pct(0.10) == "10.00%"
        assert _format_pct(0.0) == "0.00%"
        assert _format_pct(1.0) == "100.00%"

    def test_format_val(self):
        # CAGR
        result = _format_val(0.107, "cagr")
        assert "%" in result
        # Sharpe
        result = _format_val(0.93, "sharpe")
        assert "%" not in result
        assert "0.9300" in result

    def test_format_val_max_drawdown(self):
        result = _format_val(0.257, "max_drawdown")
        assert "%" in result

    def test_format_val_volatility(self):
        result = _format_val(0.115, "volatility")
        assert "%" in result


# ─── Tests: End-to-End Assessment ───────────────────────────────────────


class TestMetricsConfig:
    def test_all_metrics_have_config(self):
        """All metrics should have a label and realistic_max."""
        for name, config in METRICS_CONFIG.items():
            assert "label" in config
            assert "realistic_max" in config
            assert config["realistic_max"] > 0

    def test_grade_thresholds_defined(self):
        """Grade thresholds should be properly defined."""
        assert len(GRADE_THRESHOLDS) >= 4
        # Last threshold should be inf
        assert GRADE_THRESHOLDS[-1][0] == float("inf")
        # First should be smallest
        for i in range(len(GRADE_THRESHOLDS) - 1):
            assert GRADE_THRESHOLDS[i][0] < GRADE_THRESHOLDS[i + 1][0]


class TestMetricGap:
    def test_dataclass_fields(self):
        """MetricGap should have all required fields."""
        m = MetricGap(
            name="sharpe",
            backtest_value=0.93,
            actual_value=0.5,
            gap=-0.43,
            gap_pct=0.462,
            bootstrap_std=0.1,
            z_score=-4.3,
            confidence_95_lower=0.75,
            confidence_95_upper=1.10,
            grade="D",
            grade_detail="Significant gap",
            within_expected=False,
        )
        assert m.name == "sharpe"
        assert m.z_score == -4.3


class TestImplementationRiskReport:
    def test_report_dataclass(self):
        """ImplementationRiskReport should have all required fields."""
        r = ImplementationRiskReport(
            timestamp="2026-05-17T12:00:00",
            trading_days=30,
            report_type="paper",
            metrics={},
            composite_grade="A",
            composite_gap_pct=0.05,
            alerts=[],
            recommendations=["Continue monitoring"],
            backtest_source="auto",
            data_quality={"days": 30},
        )
        assert r.report_type == "paper"
        assert r.composite_grade == "A"


# ─── Integration Tests ──────────────────────────────────────────────────


class TestAssessImplementationRisk:
    def test_assess_with_realistic_data(self, mock_data_dirs):
        """Should produce a valid report with mock data."""
        bt_dir, perf_log = mock_data_dirs

        with patch("src.monitor.implementation_risk.DATA_DIR", bt_dir.parent):
            with patch("src.monitor.implementation_risk.PERFORMANCE_LOG", perf_log):
                report = assess_implementation_risk()

        assert isinstance(report, ImplementationRiskReport)
        assert report.composite_grade in ("A", "B", "C", "D", "F", "N/A")
        assert len(report.metrics) > 0
        assert len(report.recommendations) > 0
        assert isinstance(report.alerts, list)

    def test_assess_contains_all_metrics(self, mock_data_dirs):
        """Report should contain all expected metrics."""
        bt_dir, perf_log = mock_data_dirs

        with patch("src.monitor.implementation_risk.DATA_DIR", bt_dir.parent):
            with patch("src.monitor.implementation_risk.PERFORMANCE_LOG", perf_log):
                report = assess_implementation_risk()

        for name in METRICS_CONFIG:
            assert name in report.metrics, f"Missing metric: {name}"

    def test_assess_with_no_backtest(self, tmp_path, sample_performance_log):
        """Should work with default benchmarks when no backtest file exists."""
        with patch("src.monitor.implementation_risk.DATA_DIR", tmp_path):
            with patch("src.monitor.implementation_risk.PERFORMANCE_LOG", sample_performance_log):
                with patch("src.monitor.implementation_risk.BACKTEST_RESULTS_DIR", tmp_path):
                    report = assess_implementation_risk(backtest_source="auto")

        assert isinstance(report, ImplementationRiskReport)
        # Should use default benchmark values since no backtest file exists
        for name, m in report.metrics.items():
            assert m.backtest_value == METRICS_CONFIG[name]["benchmark_value"]

    def test_assess_with_no_data(self, tmp_path):
        """Should handle absence of any data gracefully."""
        no_data = tmp_path / "no_data.jsonl"
        with open(no_data, "w") as f:
            f.write("")

        with patch("src.monitor.implementation_risk.PERFORMANCE_LOG", no_data):
            with patch("src.monitor.implementation_risk.BACKTEST_RESULTS_DIR", tmp_path):
                report = assess_implementation_risk()

        assert report.trading_days == 0
        assert report.composite_grade in ("A", "B", "C", "D", "F", "N/A")

    def test_state_persisted(self, mock_data_dirs, tmp_path):
        """Should save state to file."""
        bt_dir, perf_log = mock_data_dirs
        state_path = tmp_path / "implementation_risk_state.json"

        with patch("src.monitor.implementation_risk.DATA_DIR", bt_dir.parent):
            with patch("src.monitor.implementation_risk.PERFORMANCE_LOG", perf_log):
                with patch("src.monitor.implementation_risk.STATE_PATH", state_path):
                    assess_implementation_risk()

        assert state_path.exists()
        with open(state_path) as f:
            state = json.load(f)
        assert "timestamp" in state
        assert "composite_grade" in state
        assert "metrics" in state
        assert len(state["metrics"]) > 0


# ─── Edge Cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_load_paper_trading_empty(self, tmp_path):
        """Should return zeros for empty trading data."""
        path = tmp_path / "empty.jsonl"
        with open(path, "w") as f:
            f.write("")

        with patch("src.monitor.implementation_risk.PERFORMANCE_LOG", path):
            result = load_paper_trading_data()

        assert isinstance(result, dict)
        assert "sharpe" in result
        assert "cagr" in result
        assert result["sharpe"] == 0.0

    def test_load_paper_trading_single_returns(self, tmp_path):
        """Should handle single return (insufficient for vol)."""
        path = tmp_path / "single.jsonl"
        now = datetime.now(timezone.utc)
        with open(path, "w") as f:
            f.write(json.dumps({
                "timestamp": now.isoformat(),
                "daily_return": 0.001,
            }) + "\n")

        with patch("src.monitor.implementation_risk.PERFORMANCE_LOG", path):
            result = load_paper_trading_data()

        # Single return → insufficient data → zeros
        assert result["sharpe"] == 0.0
        assert result["cagr"] == 0.0

    def test_extract_returns_no_valid_lines(self, tmp_path):
        """Should return empty for log with no valid entries."""
        path = tmp_path / "invalid.jsonl"
        with open(path, "w") as f:
            f.write("not json\n")
            f.write("also not json\n")

        with patch("src.monitor.implementation_risk.PERFORMANCE_LOG", path):
            result = _extract_returns(days=30)

        assert len(result) == 0

    def test_compute_gap_zero_bootstrap_std(self):
        """Should handle zero bootstrap std gracefully."""
        gap, gap_pct, z, grade, detail, within = compute_gap(1.0, 1.5, 1e-10)
        # With near-zero std, z_score will be very large
        assert grade == "F"
        assert not within

    def test_bootstrap_consistency(self):
        """Bootstrap should produce consistent results across calls."""
        np.random.seed(42)
        l1, u1, s1 = bootstrap_confidence(0.93, n_simulations=1000, n_years=5.0)
        np.random.seed(42)
        l2, u2, s2 = bootstrap_confidence(0.93, n_simulations=1000, n_years=5.0)

        assert abs(l1 - l2) < 1e-10
        assert abs(u1 - u2) < 1e-10
        assert abs(s1 - s2) < 1e-10
