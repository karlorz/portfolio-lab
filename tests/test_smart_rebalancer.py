#!/usr/bin/env python3
"""
Tests for smart_rebalancer.py — enums, data classes, drift calculation, urgency
classification, cost estimation, rebalance decision engine, cost budget tracking,
and status reporting.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from unittest.mock import patch

from src.rebalancing.smart_rebalancer import (
    RebalanceDecision,
    UrgencyLevel,
    PortfolioSnapshot,
    MarketConditions,
    RebalanceDecisionResult,
    CostBudgetTracker,
    SmartRebalancingController,
    create_sample_portfolio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_portfolio(**overrides):
    defaults = dict(
        holdings={'SPY': 46000, 'GLD': 38000, 'TLT': 16000},
        targets={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
        total_value=100000,
        timestamp=datetime.now(),
    )
    defaults.update(overrides)
    return PortfolioSnapshot(**defaults)


def _make_market(**overrides):
    defaults = dict(vpin=0.30, vix=18.0, timestamp=datetime.now())
    defaults.update(overrides)
    return MarketConditions(**defaults)


def _drifted_portfolio(drift_pct=0.15):
    """Create a portfolio with ~drift_pct drift on SPY."""
    # Target SPY = 0.46, so drift_pct means current = 0.46 * (1 + drift_pct)
    spy_value = 100000 * 0.46 * (1 + drift_pct)
    remaining = 100000 - spy_value
    return _make_portfolio(
        holdings={'SPY': spy_value, 'GLD': remaining * 0.7, 'TLT': remaining * 0.3},
        total_value=100000,
    )


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------

class TestEnums:

    def test_rebalance_decision_values(self):
        assert RebalanceDecision.EXECUTE.value == "execute"
        assert RebalanceDecision.DEFER_TOXICITY.value == "defer_toxicity"
        assert RebalanceDecision.SKIP_LOW_DRIFT.value == "skip_low_drift"

    def test_urgency_level_values(self):
        assert UrgencyLevel.LOW.value == "low"
        assert UrgencyLevel.MODERATE.value == "moderate"
        assert UrgencyLevel.HIGH.value == "high"
        assert UrgencyLevel.EMERGENCY.value == "emergency"


# ---------------------------------------------------------------------------
# CostBudgetTracker Tests
# ---------------------------------------------------------------------------

class TestCostBudgetTracker:

    def test_initial_state(self):
        tracker = CostBudgetTracker()
        assert tracker.ytd_total_bps == 0
        assert tracker.ytd_total_pct == 0
        assert tracker.remaining_budget_pct == 0.005
        assert not tracker.is_over_budget()
        assert not tracker.is_warning()

    def test_add_cost(self):
        tracker = CostBudgetTracker()
        tracker.add_cost(5.0, "2026-05-14", ["SPY", "GLD"])
        assert tracker.ytd_total_bps == 5.0
        assert len(tracker.ytd_costs) == 1

    def test_cumulative_costs(self):
        tracker = CostBudgetTracker()
        tracker.add_cost(3.0, "2026-05-01", ["SPY"])
        tracker.add_cost(4.0, "2026-05-15", ["GLD"])
        assert tracker.ytd_total_bps == 7.0

    def test_is_warning(self):
        tracker = CostBudgetTracker()
        tracker.add_cost(41.0, "2026-05-14", ["SPY"])
        assert tracker.is_warning()
        assert not tracker.is_over_budget()

    def test_is_over_budget(self):
        tracker = CostBudgetTracker()
        tracker.add_cost(51.0, "2026-05-14", ["SPY"])
        assert tracker.is_over_budget()

    def test_remaining_budget(self):
        tracker = CostBudgetTracker()
        tracker.add_cost(20.0, "2026-05-14", ["SPY"])
        # 20 bps = 0.002, remaining = 0.005 - 0.002 = 0.003
        assert tracker.remaining_budget_pct == pytest.approx(0.003, abs=1e-6)

    def test_remaining_budget_clamped(self):
        tracker = CostBudgetTracker()
        tracker.add_cost(100.0, "2026-05-14", ["SPY"])
        assert tracker.remaining_budget_pct == 0


# ---------------------------------------------------------------------------
# PortfolioSnapshot Tests
# ---------------------------------------------------------------------------

class TestPortfolioSnapshot:

    def test_create(self):
        p = _make_portfolio()
        assert p.total_value == 100000
        assert 'SPY' in p.targets


# ---------------------------------------------------------------------------
# MarketConditions Tests
# ---------------------------------------------------------------------------

class TestMarketConditions:

    def test_create(self):
        m = _make_market(vpin=0.45)
        assert m.vpin == 0.45


# ---------------------------------------------------------------------------
# SmartRebalancingController — drift calculation
# ---------------------------------------------------------------------------

class TestCalculateDrift:

    def test_no_drift(self):
        ctrl = SmartRebalancingController()
        p = _make_portfolio()
        max_drift, details = ctrl.calculate_drift(p)
        assert max_drift == pytest.approx(0.0, abs=0.001)

    def test_symmetric_drift(self):
        ctrl = SmartRebalancingController()
        p = _make_portfolio(
            holdings={'SPY': 50000, 'GLD': 34000, 'TLT': 16000},
            total_value=100000,
        )
        max_drift, details = ctrl.calculate_drift(p)
        # SPY: 0.50 vs 0.46 → drift = |0.50 - 0.46| / 0.46 = 0.087
        assert details['SPY'] == pytest.approx(0.087, abs=0.01)

    def test_max_drift(self):
        ctrl = SmartRebalancingController()
        p = _make_portfolio(
            holdings={'SPY': 60000, 'GLD': 30000, 'TLT': 10000},
            total_value=100000,
        )
        max_drift, details = ctrl.calculate_drift(p)
        assert max_drift > 0.20

    def test_missing_symbol_zero_value(self):
        ctrl = SmartRebalancingController()
        p = _make_portfolio(
            holdings={'SPY': 100000, 'GLD': 0, 'TLT': 0},
            total_value=100000,
        )
        max_drift, details = ctrl.calculate_drift(p)
        assert details['GLD'] > 0  # Should show drift since target is 0.38

    def test_zero_total_value(self):
        ctrl = SmartRebalancingController()
        p = _make_portfolio(
            holdings={'SPY': 0, 'GLD': 0, 'TLT': 0},
            total_value=0,
        )
        max_drift, details = ctrl.calculate_drift(p)
        # With zero value, current_alloc = 0 for all → drift = |0 - target|/target = 1.0
        assert max_drift == 1.0


# ---------------------------------------------------------------------------
# SmartRebalancingController — urgency
# ---------------------------------------------------------------------------

class TestCalculateUrgency:

    def test_low(self):
        ctrl = SmartRebalancingController()
        assert ctrl.calculate_urgency(0.10) == UrgencyLevel.LOW

    def test_moderate(self):
        ctrl = SmartRebalancingController()
        assert ctrl.calculate_urgency(0.13) == UrgencyLevel.MODERATE

    def test_high(self):
        ctrl = SmartRebalancingController()
        assert ctrl.calculate_urgency(0.17) == UrgencyLevel.HIGH

    def test_emergency(self):
        ctrl = SmartRebalancingController()
        assert ctrl.calculate_urgency(0.25) == UrgencyLevel.EMERGENCY

    def test_boundary_low_moderate(self):
        ctrl = SmartRebalancingController()
        assert ctrl.calculate_urgency(0.12) == UrgencyLevel.LOW
        assert ctrl.calculate_urgency(0.121) == UrgencyLevel.MODERATE

    def test_boundary_moderate_high(self):
        ctrl = SmartRebalancingController()
        assert ctrl.calculate_urgency(0.15) == UrgencyLevel.MODERATE
        assert ctrl.calculate_urgency(0.151) == UrgencyLevel.HIGH

    def test_boundary_high_emergency(self):
        ctrl = SmartRebalancingController()
        assert ctrl.calculate_urgency(0.20) == UrgencyLevel.HIGH
        assert ctrl.calculate_urgency(0.201) == UrgencyLevel.EMERGENCY


# ---------------------------------------------------------------------------
# SmartRebalancingController — cost estimation
# ---------------------------------------------------------------------------

class TestEstimateCost:

    def test_base_cost_in_window(self):
        ctrl = SmartRebalancingController()
        cost = ctrl.estimate_cost_bps(vpin=0.30, in_optimal_window=True)
        # base_spread(0.0003) * vpin_mult(1.0) * time_mult(1.0) + fixed(0.0002) = 0.0005 → 5 bps
        assert cost == pytest.approx(5.0, abs=0.5)

    def test_high_vpin_increases_cost(self):
        ctrl = SmartRebalancingController()
        cost_low = ctrl.estimate_cost_bps(vpin=0.30, in_optimal_window=True)
        cost_high = ctrl.estimate_cost_bps(vpin=0.60, in_optimal_window=True)
        assert cost_high > cost_low

    def test_vpin_multiplier_capped(self):
        ctrl = SmartRebalancingController()
        cost = ctrl.estimate_cost_bps(vpin=1.0, in_optimal_window=True)
        assert cost < 15  # Should be bounded

    def test_outside_window_increases_cost(self):
        ctrl = SmartRebalancingController()
        cost_in = ctrl.estimate_cost_bps(vpin=0.30, in_optimal_window=True)
        cost_out = ctrl.estimate_cost_bps(vpin=0.30, in_optimal_window=False)
        assert cost_out >= cost_in


# ---------------------------------------------------------------------------
# SmartRebalancingController — optimal window
# ---------------------------------------------------------------------------

class TestOptimalWindow:

    def test_in_window(self):
        ctrl = SmartRebalancingController()
        noon = datetime(2026, 5, 14, 12, 0)
        assert ctrl._in_optimal_window(noon) is True

    def test_before_window(self):
        ctrl = SmartRebalancingController()
        morning = datetime(2026, 5, 14, 9, 0)
        assert ctrl._in_optimal_window(morning) is False

    def test_after_window(self):
        ctrl = SmartRebalancingController()
        afternoon = datetime(2026, 5, 14, 15, 0)
        assert ctrl._in_optimal_window(afternoon) is False

    def test_at_start(self):
        ctrl = SmartRebalancingController()
        at_start = datetime(2026, 5, 14, 11, 0)
        assert ctrl._in_optimal_window(at_start) is True

    def test_at_end(self):
        ctrl = SmartRebalancingController()
        at_end = datetime(2026, 5, 14, 14, 0)
        assert ctrl._in_optimal_window(at_end) is False


# ---------------------------------------------------------------------------
# SmartRebalancingController — should_rebalance decision engine
# ---------------------------------------------------------------------------

class TestShouldRebalance:

    def test_skip_low_drift(self):
        ctrl = SmartRebalancingController()
        p = _make_portfolio()  # No drift
        m = _make_market()
        result = ctrl.should_rebalance(p, m)
        assert result.decision == RebalanceDecision.SKIP_LOW_DRIFT

    def test_execute_in_window_low_vpin(self):
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.15)
        m = _make_market(vpin=0.30)
        now = datetime(2026, 5, 14, 12, 0)  # In window
        result = ctrl.should_rebalance(p, m, now=now)
        assert result.decision == RebalanceDecision.EXECUTE

    def test_defer_toxicity(self):
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.15)
        m = _make_market(vpin=0.60)  # High VPIN
        now = datetime(2026, 5, 14, 12, 0)
        result = ctrl.should_rebalance(p, m, now=now)
        assert result.decision == RebalanceDecision.DEFER_TOXICITY

    def test_defer_timing(self):
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.11)  # Low urgency
        m = _make_market(vpin=0.30)
        now = datetime(2026, 5, 14, 9, 30)  # Outside window
        result = ctrl.should_rebalance(p, m, now=now)
        assert result.decision == RebalanceDecision.DEFER_TIMING

    def test_emergency_override(self):
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.30)  # >25% drift
        m = _make_market(vpin=0.80)  # High VPIN
        now = datetime(2026, 5, 14, 9, 0)  # Outside window
        result = ctrl.should_rebalance(p, m, now=now)
        assert result.decision == RebalanceDecision.OVERRIDE_EMERGENCY
        assert result.urgency == UrgencyLevel.EMERGENCY

    def test_defer_budget(self):
        ctrl = SmartRebalancingController()
        # Exhaust budget
        ctrl.cost_tracker.add_cost(60, "2026-05-01", ["SPY"])
        p = _drifted_portfolio(0.15)
        m = _make_market(vpin=0.30)
        now = datetime(2026, 5, 14, 12, 0)
        result = ctrl.should_rebalance(p, m, now=now)
        assert result.decision == RebalanceDecision.DEFER_BUDGET

    def test_emergency_overrides_budget(self):
        ctrl = SmartRebalancingController()
        ctrl.cost_tracker.add_cost(60, "2026-05-01", ["SPY"])
        p = _drifted_portfolio(0.30)  # Emergency drift
        m = _make_market(vpin=0.30)
        now = datetime(2026, 5, 14, 12, 0)
        result = ctrl.should_rebalance(p, m, now=now)
        # Emergency should override budget deferral
        assert result.decision == RebalanceDecision.OVERRIDE_EMERGENCY

    def test_max_deferral_forces_execute(self):
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.15)
        m = _make_market(vpin=0.60)  # High VPIN
        now = datetime(2026, 5, 14, 12, 0)
        # Defer 5 times (max is 4)
        for _ in range(5):
            result = ctrl.should_rebalance(p, m, now=now)
        assert result.decision == RebalanceDecision.EXECUTE

    def test_vpin_resets_on_execute(self):
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.15)
        m_low = _make_market(vpin=0.30)
        m_high = _make_market(vpin=0.60)
        now = datetime(2026, 5, 14, 12, 0)
        # First: defer due to high VPIN
        ctrl.should_rebalance(p, m_high, now=now)
        # Then: execute with low VPIN resets counter
        result = ctrl.should_rebalance(p, m_low, now=now)
        assert result.decision == RebalanceDecision.EXECUTE


# ---------------------------------------------------------------------------
# SmartRebalancingController — record / status
# ---------------------------------------------------------------------------

class TestRecordAndStatus:

    def test_record_rebalance(self):
        ctrl = SmartRebalancingController()
        ctrl.record_rebalance(5.0, "2026-05-14", ["SPY", "GLD"])
        assert ctrl.cost_tracker.ytd_total_bps == 5.0
        assert ctrl.last_rebalance is not None

    def test_get_status(self):
        ctrl = SmartRebalancingController()
        status = ctrl.get_status()
        assert 'ytd_cost_bps' in status
        assert 'remaining_budget_pct' in status
        assert 'config' in status
        assert status['is_over_budget'] is False

    def test_status_after_costs(self):
        ctrl = SmartRebalancingController()
        ctrl.record_rebalance(10.0, "2026-05-14", ["SPY"])
        status = ctrl.get_status()
        assert status['ytd_cost_bps'] == 10.0


# ---------------------------------------------------------------------------
# create_sample_portfolio
# ---------------------------------------------------------------------------

class TestSamplePortfolio:

    def test_creates_valid_portfolio(self):
        p = create_sample_portfolio()
        assert isinstance(p, PortfolioSnapshot)
        assert p.total_value == 100000
        assert abs(sum(p.targets.values()) - 1.0) < 0.01
