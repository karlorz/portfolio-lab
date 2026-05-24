#!/usr/bin/env python3
"""
Tests for smart_rebalancer.py — enums, data classes, drift calculation, urgency
classification, cost estimation, rebalance decision engine, cost budget tracking,
and status reporting.
"""

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


# ---------------------------------------------------------------------------
# Per-ETF transaction cost constants
# ---------------------------------------------------------------------------

class TestPerETFTransactionCosts:

    def test_spy_is_cheapest_equity(self):
        assert SmartRebalancingController.ETF_TRANSACTION_COSTS_BPS['SPY'] == 2.0

    def test_tlt_more_expensive_than_spy(self):
        tlt = SmartRebalancingController.ETF_TRANSACTION_COSTS_BPS['TLT']
        spy = SmartRebalancingController.ETF_TRANSACTION_COSTS_BPS['SPY']
        assert tlt > spy

    def test_gld_between_spy_and_tlt(self):
        spy = SmartRebalancingController.ETF_TRANSACTION_COSTS_BPS['SPY']
        gld = SmartRebalancingController.ETF_TRANSACTION_COSTS_BPS['GLD']
        tlt = SmartRebalancingController.ETF_TRANSACTION_COSTS_BPS['TLT']
        assert spy < gld < tlt

    def test_known_symbols_count(self):
        costs = SmartRebalancingController.ETF_TRANSACTION_COSTS_BPS
        assert len(costs) >= 10

    def test_default_cost_constant(self):
        assert SmartRebalancingController.DEFAULT_COST_BPS == 5.0

    def test_all_costs_positive(self):
        for sym, cost in SmartRebalancingController.ETF_TRANSACTION_COSTS_BPS.items():
            assert cost > 0, f"{sym} has non-positive cost {cost}"

    def test_all_costs_reasonable_range(self):
        for sym, cost in SmartRebalancingController.ETF_TRANSACTION_COSTS_BPS.items():
            assert 1.0 <= cost <= 15.0, f"{sym} cost {cost} out of range"


# ---------------------------------------------------------------------------
# estimate_per_symbol_cost_bps
# ---------------------------------------------------------------------------

class TestEstimatePerSymbolCost:

    def test_spy_cheaper_than_tlt(self):
        ctrl = SmartRebalancingController()
        spy_cost = ctrl.estimate_per_symbol_cost_bps('SPY', 0.30, True)
        tlt_cost = ctrl.estimate_per_symbol_cost_bps('TLT', 0.30, True)
        assert spy_cost < tlt_cost

    def test_unknown_symbol_uses_default(self):
        ctrl = SmartRebalancingController()
        cost = ctrl.estimate_per_symbol_cost_bps('UNKNOWN', 0.30, True)
        # Should use DEFAULT_COST_BPS (5.0) as base, result > 0
        assert cost > 0

    def test_high_vpin_increases_cost(self):
        ctrl = SmartRebalancingController()
        low_vpin = ctrl.estimate_per_symbol_cost_bps('SPY', 0.20, True)
        high_vpin = ctrl.estimate_per_symbol_cost_bps('SPY', 0.60, True)
        assert high_vpin >= low_vpin

    def test_optimal_window_reduces_cost(self):
        ctrl = SmartRebalancingController()
        with patch('src.rebalancing.smart_rebalancer.datetime') as mock_dt:
            # Outside optimal window (8am)
            mock_dt.now.return_value = datetime(2026, 5, 13, 8, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            outside = ctrl.estimate_per_symbol_cost_bps('SPY', 0.30, False)
        inside = ctrl.estimate_per_symbol_cost_bps('SPY', 0.30, True)
        assert inside <= outside

    def test_returns_bps_value(self):
        ctrl = SmartRebalancingController()
        cost = ctrl.estimate_per_symbol_cost_bps('GLD', 0.30, True)
        assert isinstance(cost, float)
        assert cost > 0
        # Should be in reasonable range (1-30 bps)
        assert 1.0 <= cost <= 30.0


# ---------------------------------------------------------------------------
# Regime-adaptive drift thresholds
# ---------------------------------------------------------------------------

class TestRegimeAdaptiveDriftThresholds:

    def test_config_has_regime_thresholds(self):
        ctrl = SmartRebalancingController()
        thresholds = ctrl.config['drift_threshold_by_regime']
        assert 'low_vol' in thresholds
        assert 'normal' in thresholds
        assert 'high_vol' in thresholds
        assert 'crisis' in thresholds

    def test_regime_thresholds_ordering(self):
        """Crisis threshold should be tighter than normal, normal tighter than low_vol."""
        thresholds = SmartRebalancingController.DEFAULT_CONFIG['drift_threshold_by_regime']
        assert thresholds['crisis'] < thresholds['high_vol']
        assert thresholds['high_vol'] < thresholds['normal']
        assert thresholds['normal'] < thresholds['low_vol']

    def test_normal_equals_default(self):
        thresholds = SmartRebalancingController.DEFAULT_CONFIG['drift_threshold_by_regime']
        default = SmartRebalancingController.DEFAULT_CONFIG['drift_threshold']
        assert thresholds['normal'] == default

    def test_crisis_regime_triggers_sooner(self):
        """With crisis regime, a 6% drift triggers rebalance that would be skipped normally."""
        ctrl = SmartRebalancingController()
        # 6% drift on SPY: holdings slightly off
        portfolio = _drifted_portfolio(0.06)
        market = _make_market(vpin=0.30)

        # Without regime — default 10% threshold, 6% drift should skip
        result_normal = ctrl.should_rebalance(portfolio, market, regime=None)
        assert result_normal.decision == RebalanceDecision.SKIP_LOW_DRIFT

        # With crisis regime — 5% threshold, 6% drift should trigger
        result_crisis = ctrl.should_rebalance(portfolio, market, regime='crisis')
        assert result_crisis.decision != RebalanceDecision.SKIP_LOW_DRIFT

    def test_low_vol_regime_allows_more_drift(self):
        """With low_vol regime, a 12% drift triggers rebalance that would normally trigger."""
        ctrl = SmartRebalancingController()
        # 12% drift — triggers with default 10% threshold
        portfolio = _drifted_portfolio(0.12)
        market = _make_market(vpin=0.30)
        now = datetime(2026, 5, 13, 12, 0)

        result_low = ctrl.should_rebalance(portfolio, market, now=now, regime='low_vol')
        # low_vol threshold is 15%, so 12% drift should be skipped
        assert result_low.decision == RebalanceDecision.SKIP_LOW_DRIFT

    def test_unknown_regime_falls_back_to_default(self):
        ctrl = SmartRebalancingController()
        portfolio = _drifted_portfolio(0.08)
        market = _make_market(vpin=0.30)
        result = ctrl.should_rebalance(portfolio, market, regime='unknown_regime')
        # Default 10% threshold, 8% drift → skip
        assert result.decision == RebalanceDecision.SKIP_LOW_DRIFT


# ---------------------------------------------------------------------------
# estimate_total_cost_bps (per-ETF integration)
# ---------------------------------------------------------------------------

class TestEstimateTotalCostBps:

    def test_uses_per_symbol_costs(self):
        ctrl = SmartRebalancingController()
        drift_details = {'SPY': 0.05, 'GLD': 0.03}
        total = ctrl.estimate_total_cost_bps(drift_details, 0.30, True)
        assert total > 0

    def test_tlt_costs_more_than_spy_for_same_drift(self):
        ctrl = SmartRebalancingController()
        spy_drift = {'SPY': 0.10}
        tlt_drift = {'TLT': 0.10}
        spy_cost = ctrl.estimate_total_cost_bps(spy_drift, 0.30, True)
        tlt_cost = ctrl.estimate_total_cost_bps(tlt_drift, 0.30, True)
        assert tlt_cost > spy_cost

    def test_empty_drift_falls_back_to_flat(self):
        ctrl = SmartRebalancingController()
        total = ctrl.estimate_total_cost_bps({}, 0.30, True)
        flat = ctrl.estimate_cost_bps(0.30, True)
        assert total == flat

    def test_larger_drift_costs_more(self):
        ctrl = SmartRebalancingController()
        small_drift = {'SPY': 0.05}
        large_drift = {'SPY': 0.15}
        small_cost = ctrl.estimate_total_cost_bps(small_drift, 0.30, True)
        large_cost = ctrl.estimate_total_cost_bps(large_drift, 0.30, True)
        assert large_cost > small_cost

    def test_rebalance_result_uses_per_symbol_costs(self):
        """should_rebalance now uses per-ETF cost estimation."""
        ctrl = SmartRebalancingController()
        portfolio = _drifted_portfolio(0.15)
        market = _make_market(vpin=0.30)
        now = datetime(2026, 5, 13, 12, 0)
        result = ctrl.should_rebalance(portfolio, market, now=now)
        if result.decision == RebalanceDecision.EXECUTE:
            # Cost should reflect per-ETF pricing, not flat rate
            assert result.estimated_cost_bps > 0


# ---------------------------------------------------------------------------
# Per-symbol cost breakdown in metadata
# ---------------------------------------------------------------------------

class TestPerSymbolCostBreakdown:

    def test_execute_includes_per_symbol_costs(self):
        ctrl = SmartRebalancingController()
        portfolio = _drifted_portfolio(0.15)
        market = _make_market(vpin=0.30)
        now = datetime(2026, 5, 13, 12, 0)
        result = ctrl.should_rebalance(portfolio, market, now=now)
        if result.decision == RebalanceDecision.EXECUTE:
            assert 'per_symbol_cost_bps' in result.metadata
            per_sym = result.metadata['per_symbol_cost_bps']
            assert isinstance(per_sym, dict)
            # At least one symbol should have a cost entry
            assert len(per_sym) > 0

    def test_skip_has_no_per_symbol_costs(self):
        ctrl = SmartRebalancingController()
        portfolio = _make_portfolio()  # No drift
        market = _make_market(vpin=0.30)
        result = ctrl.should_rebalance(portfolio, market)
        assert result.decision == RebalanceDecision.SKIP_LOW_DRIFT
        assert 'per_symbol_cost_bps' not in result.metadata
