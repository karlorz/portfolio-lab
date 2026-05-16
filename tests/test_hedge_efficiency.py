"""
Tests for v7.04 Hedge Efficiency Monitor.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.monitor.hedge_efficiency import (
    HedgeEfficiencyMonitor,
    HedgeEfficiencyReport,
    DrawdownEvent,
    HedgeComparison,
)


class TestDrawdownDetection:
    """Test drawdown event detection from SPY returns."""

    def setup_method(self):
        self.monitor = HedgeEfficiencyMonitor()

    def test_no_drawdowns_in_flat_market(self):
        returns = [0.1] * 100  # Flat +0.1% daily
        events = self.monitor.detect_drawdowns(returns)
        assert len(events) == 0

    def test_detect_single_drawdown(self):
        # 20 days flat, 10 days down -2% each, recovery
        returns = [0.0] * 20 + [-2.0] * 10 + [0.5] * 30
        events = self.monitor.detect_drawdowns(returns)
        assert len(events) >= 1

    def test_detect_multiple_drawdowns(self):
        returns = (
            [0.0] * 10 + [-2.0] * 8 + [0.3] * 15 +   # DD1
            [0.0] * 10 + [-1.5] * 10 + [0.3] * 15     # DD2
        )
        events = self.monitor.detect_drawdowns(returns, threshold_pct=-3.0)
        assert len(events) >= 1

    def test_empty_returns(self):
        events = self.monitor.detect_drawdowns([])
        assert len(events) == 0


class TestEfficiencyComputation:
    """Test cost-benefit efficiency calculations."""

    def setup_method(self):
        self.monitor = HedgeEfficiencyMonitor()

    def test_event_efficiency_positive_for_good_hedge(self):
        """A drawdown where VIXY gains should produce positive efficiency."""
        eff = self.monitor.compute_event_efficiency(
            spy_drawdown_pct=-10.0,
            vixy_gain_pct=30.0,
            allocation_pct=5.0,
            annual_cost_bps=50.0,
            event_days=30,
        )
        assert eff > 0

    def test_running_efficiency_zero_with_no_benefit(self):
        eff = self.monitor.compute_running_efficiency(
            allocation_pct=3.0,
            vix_level=20.0,
            ytd_cost_bps=30.0,
            ytd_benefit_bps=0.0,
        )
        assert eff == 0.0

    def test_running_efficiency_zero_with_no_cost(self):
        eff = self.monitor.compute_running_efficiency(
            allocation_pct=3.0,
            vix_level=20.0,
            ytd_cost_bps=0.0,
            ytd_benefit_bps=100.0,
        )
        assert eff == 0.0


class TestStrategyComparison:
    """Test hedge strategy comparison rankings."""

    def setup_method(self):
        self.monitor = HedgeEfficiencyMonitor()

    def test_comparison_ranks_strategies(self):
        comparison = self.monitor.compare_strategies(vixy_efficiency=1.5, allocation_pct=3.0)
        assert "vixy" in comparison
        assert "collar" in comparison
        assert "trend_following" in comparison
        assert "cash" in comparison
        assert "vixy_rank" in comparison
        assert 1 <= comparison["vixy_rank"] <= 4

    def test_good_vixy_ranks_high(self):
        comparison = self.monitor.compare_strategies(vixy_efficiency=3.0, allocation_pct=5.0)
        assert comparison["vixy_rank"] == 1  # Should be best


class TestEfficiencyGrading:
    """Test letter grade assignment."""

    def test_grade_a_excellent(self):
        assert HedgeEfficiencyMonitor.grade_efficiency(2.5) == "A"
        assert HedgeEfficiencyMonitor.grade_efficiency(2.0) == "A"

    def test_grade_b_good(self):
        assert HedgeEfficiencyMonitor.grade_efficiency(1.7) == "B"

    def test_grade_c_marginal(self):
        assert HedgeEfficiencyMonitor.grade_efficiency(1.2) == "C"
        assert HedgeEfficiencyMonitor.grade_efficiency(1.0) == "C"

    def test_grade_d_poor(self):
        assert HedgeEfficiencyMonitor.grade_efficiency(0.7) == "D"

    def test_grade_f_failing(self):
        assert HedgeEfficiencyMonitor.grade_efficiency(0.3) == "F"
        assert HedgeEfficiencyMonitor.grade_efficiency(0.0) == "F"


class TestReportGeneration:
    """Test full report generation."""

    def setup_method(self):
        self.monitor = HedgeEfficiencyMonitor()

    def test_basic_report_has_all_fields(self):
        report = self.monitor.generate_report(
            allocation_pct=3.0,
            vix_level=22.0,
        )
        assert report.strategy == "VIXY Dynamic Hedge"
        assert report.current_allocation == 3.0
        assert report.efficiency_grade in ("A", "B", "C", "D", "F")
        assert len(report.recommendation) > 0

    def test_report_with_drawdowns(self):
        spy_returns = [0.1] * 30 + [-2.0] * 10 + [0.5] * 20
        vixy_returns = [0.05] * 30 + [3.0] * 10 + [-0.2] * 20
        dates = [f"2026-{i//30+1:02d}-{i%30+1:02d}" for i in range(60)]

        report = self.monitor.generate_report(
            allocation_pct=4.0,
            vix_level=25.0,
            ytd_cost_bps=45.0,
            ytd_benefit_bps=80.0,
            spy_returns=spy_returns,
            vixy_returns=vixy_returns,
            event_dates=dates,
        )
        assert report.running_efficiency > 0
        assert isinstance(report.recent_drawdowns, list)

    def test_recommendation_for_good_efficiency(self):
        report = self.monitor.generate_report(
            allocation_pct=4.0,
            vix_level=30.0,
            ytd_cost_bps=30.0,
            ytd_benefit_bps=90.0,  # 3x efficiency
        )
        assert "cost-effective" in report.recommendation.lower()

    def test_recommendation_for_poor_efficiency(self):
        report = self.monitor.generate_report(
            allocation_pct=2.0,
            vix_level=15.0,
            ytd_cost_bps=50.0,
            ytd_benefit_bps=10.0,  # Poor efficiency
        )
        assert "not cost-effective" in report.recommendation.lower()


class TestStatePersistence:
    """Test efficiency monitor state save/load."""

    def test_load_empty_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = HedgeEfficiencyMonitor(project_root=Path(tmpdir))
            state = monitor._load_state()
            assert state == {}

    def test_save_and_load_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = HedgeEfficiencyMonitor(project_root=Path(tmpdir))
            monitor._state_file = Path(tmpdir) / "data" / "hedge_efficiency_state.json"

            report = monitor.generate_report(allocation_pct=3.0, vix_level=20.0)
            monitor.save_state(report)

            # Verify file exists and is valid JSON
            assert monitor._state_file.exists()
            with open(monitor._state_file) as f:
                data = json.load(f)
            assert "strategy" in data
            assert "efficiency_grade" in data

    def test_corrupt_state_handled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "data" / "hedge_efficiency_state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text("not valid json{{{")

            monitor = HedgeEfficiencyMonitor(project_root=Path(tmpdir))
            monitor._state_file = state_file
            state = monitor._load_state()
            assert state == {}  # Should default gracefully


class TestDashboardStats:
    """Test dashboard stats output."""

    def setup_method(self):
        self.monitor = HedgeEfficiencyMonitor()

    def test_dashboard_stats_structure(self):
        stats = self.monitor.get_dashboard_stats()
        assert "current_allocation_pct" in stats
        assert "ytd_cost_bps" in stats
        assert "efficiency_score" in stats
        assert "efficiency_grade" in stats
