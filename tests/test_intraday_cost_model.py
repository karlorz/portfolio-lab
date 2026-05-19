"""Tests for src/execution/intraday_cost_model.py — Intraday Execution Cost Model."""

from datetime import time, datetime, timedelta
import pytest
from src.execution.intraday_cost_model import (
    IntradayCostEstimate,
    IntradayExecutionCostModel,
    RebalanceScheduler,
)


# ── IntradayCostEstimate ─────────────────────────────────────────────────

class TestCostEstimate:
    def test_dataclass_fields(self):
        est = IntradayCostEstimate(
            spread_cost_bps=1.0, impact_cost_bps=2.0, total_cost_bps=3.0,
            confidence="high", recommended_window="optimal",
        )
        assert est.spread_cost_bps == 1.0
        assert est.impact_cost_bps == 2.0
        assert est.total_cost_bps == 3.0
        assert est.confidence == "high"
        assert est.recommended_window == "optimal"


# ── IntradayExecutionCostModel ────────────────────────────────────────────

@pytest.fixture
def cost_model():
    return IntradayExecutionCostModel()


class TestBaselineSpread:
    def test_known_symbol(self, cost_model):
        assert cost_model.get_baseline_spread("SPY") == 0.5
        assert cost_model.get_baseline_spread("QQQ") == 0.7
        assert cost_model.get_baseline_spread("TLT") == 1.2
        assert cost_model.get_baseline_spread("EFA") == 2.5

    def test_unknown_symbol_defaults(self, cost_model):
        assert cost_model.get_baseline_spread("UNKNOWN") == 2.0

    def test_case_sensitive(self, cost_model):
        assert cost_model.get_baseline_spread("spy") == 2.0  # no match


class TestTimeMultiplier:
    def test_optimal_hours(self, cost_model):
        assert cost_model.get_time_multiplier(11) == 1.0
        assert cost_model.get_time_multiplier(12) == 1.0
        assert cost_model.get_time_multiplier(13) == 1.0

    def test_opening_hour(self, cost_model):
        assert cost_model.get_time_multiplier(9) == 3.0

    def test_close_hour(self, cost_model):
        assert cost_model.get_time_multiplier(16) == 2.5

    def test_unknown_hour_defaults(self, cost_model):
        assert cost_model.get_time_multiplier(5) == 1.5
        assert cost_model.get_time_multiplier(20) == 1.5


class TestEstimateCost:
    def test_optimal_window_high_confidence(self, cost_model):
        est = cost_model.estimate_cost("SPY", hour=12, size_dv_pct=0.01)
        assert est.total_cost_bps > 0
        assert est.confidence == "high"
        assert est.recommended_window == "optimal"
        # spread = 0.5 * 1.0 / 2 = 0.25, impact = 20 * sqrt(0.01) = 2.0
        # total = (0.25 + 2.0) * 1.0 = 2.25
        assert est.spread_cost_bps == pytest.approx(0.25)
        assert est.impact_cost_bps == pytest.approx(2.0)
        assert est.total_cost_bps == pytest.approx(2.25)

    def test_opening_window_avoid(self, cost_model):
        est = cost_model.estimate_cost("SPY", hour=9, size_dv_pct=0.01)
        assert est.recommended_window == "avoid"
        # spread = 0.5 * 3.0 / 2 = 0.75, impact = 2.0, total = 2.75
        assert est.spread_cost_bps == pytest.approx(0.75)
        assert est.total_cost_bps > 2.5

    def test_close_window_avoid(self, cost_model):
        est = cost_model.estimate_cost("SPY", hour=16, size_dv_pct=0.01)
        assert est.recommended_window == "avoid"

    def test_unknown_symbol_low_confidence(self, cost_model):
        est = cost_model.estimate_cost("MYSTERY", hour=12, size_dv_pct=0.01)
        assert est.confidence == "low"
        # default spread = 2.0
        assert est.spread_cost_bps == pytest.approx(1.0)  # 2.0 * 1.0 / 2

    def test_large_order_impact_capped(self, cost_model):
        est = cost_model.estimate_cost("SPY", hour=12, size_dv_pct=0.5)
        # sqrt(0.5) ≈ 0.707, 20 * 0.707 ≈ 14.14 < 100
        assert est.impact_cost_bps < 100
        # size_dv_pct=1.0 → sqrt(1.0)=1.0, 20*1.0=20 < 100
        est2 = cost_model.estimate_cost("SPY", hour=12, size_dv_pct=2.0)
        # sqrt(2.0) ≈ 1.414, 20 * 1.414 ≈ 28.28 < 100
        assert est2.impact_cost_bps < 100

    def test_urgency_multipliers(self, cost_model):
        base = cost_model.estimate_cost("SPY", hour=12, size_dv_pct=0.01, urgency="normal")
        low = cost_model.estimate_cost("SPY", hour=12, size_dv_pct=0.01, urgency="low")
        high = cost_model.estimate_cost("SPY", hour=12, size_dv_pct=0.01, urgency="high")
        urgent = cost_model.estimate_cost("SPY", hour=12, size_dv_pct=0.01, urgency="urgent")
        assert low.total_cost_bps < base.total_cost_bps
        assert high.total_cost_bps > base.total_cost_bps
        assert urgent.total_cost_bps > high.total_cost_bps

    def test_unknown_urgency_defaults_to_normal(self, cost_model):
        est = cost_model.estimate_cost("SPY", hour=12, size_dv_pct=0.01, urgency="unknown")
        normal = cost_model.estimate_cost("SPY", hour=12, size_dv_pct=0.01, urgency="normal")
        assert est.total_cost_bps == normal.total_cost_bps

    def test_acceptable_window_hour_10(self, cost_model):
        est = cost_model.estimate_cost("SPY", hour=10, size_dv_pct=0.01)
        assert est.recommended_window == "acceptable"
        assert est.confidence == "medium"

    def test_acceptable_window_hour_14(self, cost_model):
        est = cost_model.estimate_cost("SPY", hour=14, size_dv_pct=0.01)
        assert est.recommended_window == "acceptable"

    def test_tlt_higher_spread_than_spy(self, cost_model):
        est_spy = cost_model.estimate_cost("SPY", hour=12, size_dv_pct=0.01)
        est_tlt = cost_model.estimate_cost("TLT", hour=12, size_dv_pct=0.01)
        assert est_tlt.spread_cost_bps > est_spy.spread_cost_bps


class TestFindOptimalWindow:
    def test_returns_best_hour(self, cost_model):
        hour, est = cost_model.find_optimal_window("SPY", size_dv_pct=0.01)
        assert 9 <= hour <= 16
        assert est.total_cost_bps > 0
        # Optimal should be in 11-13 range
        assert hour in [11, 12, 13]

    def test_restricted_range(self, cost_model):
        hour, est = cost_model.find_optimal_window(
            "SPY", size_dv_pct=0.01, start_hour=14, end_hour=16
        )
        assert 14 <= hour <= 16

    def test_different_urgency(self, cost_model):
        hour_normal, est_normal = cost_model.find_optimal_window(
            "SPY", size_dv_pct=0.01, urgency="normal"
        )
        hour_urgent, est_urgent = cost_model.find_optimal_window(
            "SPY", size_dv_pct=0.01, urgency="urgent"
        )
        # Same optimal hour regardless of urgency (urgency scales cost, not hour)
        assert hour_normal == hour_urgent
        assert est_urgent.total_cost_bps > est_normal.total_cost_bps


class TestCompareWindows:
    def test_all_hours_present(self, cost_model):
        comparison = cost_model.compare_windows("SPY", size_dv_pct=0.01)
        assert len(comparison) == 8  # hours 9-16 inclusive (range(9,17))
        for h in range(9, 17):
            assert h in comparison
            assert "total_cost_bps" in comparison[h]
            assert "window" in comparison[h]

    def test_optimal_hours_cheapest(self, cost_model):
        comparison = cost_model.compare_windows("SPY", size_dv_pct=0.01)
        optimal_costs = [comparison[h]["total_cost_bps"] for h in [11, 12, 13]]
        other_costs = [comparison[h]["total_cost_bps"] for h in [9, 15, 16]]
        assert max(optimal_costs) < min(other_costs)


# ── RebalanceScheduler ────────────────────────────────────────────────────

@pytest.fixture
def scheduler():
    return RebalanceScheduler(urgency="normal")


class TestIsOptimalTime:
    def test_optimal_time(self, scheduler):
        assert scheduler.is_optimal_time(time(11, 30)) is True
        assert scheduler.is_optimal_time(time(12, 0)) is True

    def test_non_optimal_time(self, scheduler):
        assert scheduler.is_optimal_time(time(9, 30)) is False
        assert scheduler.is_optimal_time(time(15, 45)) is False

    def test_boundary_times(self, scheduler):
        assert scheduler.is_optimal_time(time(11, 0)) is True
        assert scheduler.is_optimal_time(time(13, 0)) is True
        assert scheduler.is_optimal_time(time(10, 30)) is True  # secondary start
        assert scheduler.is_optimal_time(time(14, 30)) is True  # secondary end


class TestShouldAvoid:
    def test_opening_to_avoid(self, scheduler):
        assert scheduler.should_avoid(time(9, 30)) is True
        assert scheduler.should_avoid(time(9, 45)) is True

    def test_close_to_avoid(self, scheduler):
        assert scheduler.should_avoid(time(15, 30)) is True
        assert scheduler.should_avoid(time(15, 59)) is True

    def test_midday_not_avoided(self, scheduler):
        assert scheduler.should_avoid(time(11, 0)) is False
        assert scheduler.should_avoid(time(14, 0)) is False

    def test_boundary_times(self, scheduler):
        # 9:30-10:00 avoid window is inclusive
        assert scheduler.should_avoid(time(9, 30)) is True
        assert scheduler.should_avoid(time(10, 0)) is True
        # 10:01 is past the opening avoid window
        assert scheduler.should_avoid(time(10, 1)) is False
        # 15:30-16:00 avoid window
        assert scheduler.should_avoid(time(15, 30)) is True
        assert scheduler.should_avoid(time(16, 0)) is True
        # 9:29 is before opening avoid window
        assert scheduler.should_avoid(time(9, 29)) is False


class TestSchedule:
    def test_urgent_executes_immediately(self):
        sched = RebalanceScheduler(urgency="urgent")
        target = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        result = sched.schedule(target, "SPY", 0.01)
        # urgent: max(target_time, now) — should be >= target
        assert result >= target

    def test_already_optimal_target(self, scheduler):
        now = datetime.now()
        # Find next optimal time today
        optimal_time = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if optimal_time < now:
            optimal_time += timedelta(days=1)
            # adjust to weekday
            while optimal_time.weekday() >= 5:
                optimal_time += timedelta(days=1)
        result = scheduler.schedule(optimal_time, "SPY", 0.01)
        # Should be near the optimal time
        assert abs((result - optimal_time).total_seconds()) < 86400  # within a day

    def test_returns_scheduled_time(self, scheduler):
        target = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        if target < datetime.now():
            target += timedelta(days=1)
        result = scheduler.schedule(target, "SPY", 0.01)
        assert isinstance(result, datetime)

    def test_low_urgency_has_longer_delay(self):
        low_sched = RebalanceScheduler(urgency="low")
        assert low_sched.max_delay_hours == 8

    def test_urgent_has_no_delay(self):
        urgent_sched = RebalanceScheduler(urgency="urgent")
        assert urgent_sched.max_delay_hours == 0


class TestGetScheduleRecommendation:
    def test_returns_all_fields(self, scheduler):
        target = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        if target < datetime.now():
            target += timedelta(days=1)
        rec = scheduler.get_schedule_recommendation("SPY", 0.01, target)
        assert "symbol" in rec
        assert rec["symbol"] == "SPY"
        assert "target_time" in rec
        assert "scheduled_time" in rec
        assert "urgency" in rec
        assert "target_cost_bps" in rec
        assert "scheduled_cost_bps" in rec
        assert "estimated_savings_bps" in rec
        assert "delay_hours" in rec
        assert "window_quality" in rec
        assert "confidence" in rec

    def test_savings_positive_or_zero(self, scheduler):
        target = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        if target < datetime.now():
            target += timedelta(days=1)
        rec = scheduler.get_schedule_recommendation("SPY", 0.01, target)
        # Scheduling should never increase cost
        assert rec["estimated_savings_bps"] >= -0.01  # allow tiny float error

    def test_evening_target(self, scheduler):
        """Target after market close should still get a schedule."""
        target = datetime.now().replace(hour=17, minute=0, second=0, microsecond=0)
        if target < datetime.now():
            target += timedelta(days=1)
        rec = scheduler.get_schedule_recommendation("SPY", 0.01, target)
        assert rec["scheduled_time"] is not None
