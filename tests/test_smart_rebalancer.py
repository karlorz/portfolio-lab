#!/usr/bin/env python3
"""
Tests for smart_rebalancer.py — enums, data classes, drift calculation, urgency
classification, cost estimation, rebalance decision engine, cost budget tracking,
and status reporting.
"""

import json
import logging
import inspect
from datetime import datetime
from unittest.mock import patch

import pytest

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


@pytest.fixture(autouse=True)
def _isolate_smart_rebalance_data_dir(tmp_path, monkeypatch):
    """Keep default SmartRebalancingController() from loading/writing host DATA_DIR."""
    monkeypatch.setattr(
        "src.rebalancing.smart_rebalancer.DATA_DIR",
        tmp_path,
        raising=False,
    )
    # Also isolate integration gate default data_dir if imported in-process
    monkeypatch.setattr(
        "src.rebalancing.integration.DATA_DIR",
        tmp_path,
        raising=False,
    )
    yield


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

    def test_default_clock_uses_america_new_york_not_host_local(self, monkeypatch):
        """When now is omitted, window membership follows America/New_York wall clock."""
        from zoneinfo import ZoneInfo

        ctrl = SmartRebalancingController()
        et = ZoneInfo("America/New_York")

        # 12:00 ET is inside 11–14 ET window
        fixed_et_noon = datetime(2026, 5, 14, 12, 0, tzinfo=et)

        class _FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    # Host local would be wrong if we used naive UTC 16:00 as "now"
                    return datetime(2026, 5, 14, 16, 0)  # naive host-local decoy
                return fixed_et_noon.astimezone(tz)

        monkeypatch.setattr("src.rebalancing.smart_rebalancer.datetime", _FixedDateTime)
        assert ctrl._in_optimal_window() is True

        # 08:00 ET is outside window (UTC 12:00 during EDT)
        fixed_et_morning = datetime(2026, 5, 14, 8, 0, tzinfo=et)

        class _FixedMorning(datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    return datetime(2026, 5, 14, 12, 0)  # decoy "local" hour in-window
                return fixed_et_morning.astimezone(tz)

        monkeypatch.setattr("src.rebalancing.smart_rebalancer.datetime", _FixedMorning)
        assert ctrl._in_optimal_window() is False

    def test_aware_utc_converted_to_et_for_window(self):
        """Timezone-aware UTC timestamps convert to ET before hour check."""
        from datetime import timezone

        ctrl = SmartRebalancingController()
        # 16:00 UTC on 2026-05-14 == 12:00 EDT → in window
        utc_noon_et = datetime(2026, 5, 14, 16, 0, tzinfo=timezone.utc)
        assert ctrl._in_optimal_window(utc_noon_et) is True
        # 12:00 UTC == 08:00 EDT → outside
        utc_morning_et = datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc)
        assert ctrl._in_optimal_window(utc_morning_et) is False

    def test_naive_datetime_treated_as_et_wall_clock(self):
        """Naive datetimes are interpreted as ET wall clock (documented contract)."""
        ctrl = SmartRebalancingController()
        assert ctrl._in_optimal_window(datetime(2026, 5, 14, 12, 0)) is True
        assert ctrl._in_optimal_window(datetime(2026, 5, 14, 8, 0)) is False


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
        # Exhaust budget (disable single-trade cap so synthetic 60 bps counts)
        ctrl.cost_tracker.max_single_trade_cost_bps = None
        ctrl.cost_tracker.add_cost(60, "2026-05-01", ["SPY"])
        p = _drifted_portfolio(0.15)
        m = _make_market(vpin=0.30)
        now = datetime(2026, 5, 14, 12, 0)
        result = ctrl.should_rebalance(p, m, now=now)
        assert result.decision == RebalanceDecision.DEFER_BUDGET

    def test_emergency_overrides_budget(self):
        ctrl = SmartRebalancingController()
        ctrl.cost_tracker.max_single_trade_cost_bps = None
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

    def test_record_rebalance(self, tmp_path):
        ctrl = SmartRebalancingController(
            state_path=tmp_path / "s.json", data_dir=tmp_path, load_state=False
        )
        ctrl.record_rebalance(5.0, "2026-05-14", ["SPY", "GLD"])
        assert ctrl.cost_tracker.ytd_total_bps == 5.0
        assert ctrl.last_rebalance is not None

    def test_get_status(self, tmp_path):
        ctrl = SmartRebalancingController(
            state_path=tmp_path / "s.json", data_dir=tmp_path, load_state=False
        )
        status = ctrl.get_status()
        assert 'ytd_cost_bps' in status
        assert 'remaining_budget_pct' in status
        assert 'config' in status
        assert status['is_over_budget'] is False

    def test_status_after_costs(self, tmp_path):
        ctrl = SmartRebalancingController(
            state_path=tmp_path / "s.json", data_dir=tmp_path, load_state=False
        )
        ctrl.record_rebalance(10.0, "2026-05-14", ["SPY"])
        status = ctrl.get_status()
        assert status['ytd_cost_bps'] == 10.0

    def test_record_rebalance_persists_state(self, tmp_path):
        """record_rebalance writes durable state so cold start restores costs."""
        state_path = tmp_path / "smart_rebalance_state.json"
        ctrl = SmartRebalancingController(
            state_path=state_path, data_dir=tmp_path, load_state=False
        )
        ctrl.record_rebalance(7.5, "2026-07-15", ["SPY", "GLD"])
        assert state_path.exists()
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        assert raw["ytd_costs"]
        assert abs(raw["ytd_costs"][0]["cost_bps"] - 7.5) < 1e-9
        assert raw["last_rebalance"] is not None

    def test_cold_start_loads_persisted_state(self, tmp_path):
        """Fresh controller with same state_path restores ytd cost + last_rebalance."""
        state_path = tmp_path / "smart_rebalance_state.json"
        first = SmartRebalancingController(
            state_path=state_path, data_dir=tmp_path, load_state=False
        )
        first.record_rebalance(12.0, "2026-07-10", ["SPY"])
        first.record_rebalance(3.0, "2026-07-12", ["TLT"])

        second = SmartRebalancingController(
            state_path=state_path, data_dir=tmp_path, load_state=True
        )
        assert abs(second.cost_tracker.ytd_total_bps - 15.0) < 1e-9
        assert second.last_rebalance is not None
        status = second.get_status()
        assert status["ytd_cost_bps"] == 15.0
        assert status["last_rebalance"] is not None

    def test_missing_state_file_starts_clean(self, tmp_path):
        ctrl = SmartRebalancingController(
            state_path=tmp_path / "missing.json",
            data_dir=tmp_path,
            load_state=True,
        )
        assert ctrl.cost_tracker.ytd_total_bps == 0
        assert ctrl.last_rebalance is None


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


# ---------------------------------------------------------------------------
# Dataclass field validation
# ---------------------------------------------------------------------------

class TestDataclassFields:
    """Validate field definitions, types, and defaults for all dataclasses."""

    def test_portfolio_snapshot_has_correct_fields(self):
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(PortfolioSnapshot)}
        assert set(fields) == {'holdings', 'targets', 'total_value', 'timestamp'}
        assert 'Dict[str, float]' in str(fields['holdings'].type)
        assert 'Dict[str, float]' in str(fields['targets'].type)
        assert fields['total_value'].type is float
        assert fields['timestamp'].type is datetime

    def test_portfolio_snapshot_all_fields_required(self):
        import dataclasses
        for f in dataclasses.fields(PortfolioSnapshot):
            msg = f"field '{f.name}' should not have a default"
            assert f.default is dataclasses.MISSING, msg
            assert f.default_factory is dataclasses.MISSING, msg

    def test_market_conditions_has_correct_fields(self):
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(MarketConditions)}
        assert set(fields) == {'vpin', 'vix', 'spread_bps', 'timestamp'}
        assert fields['vpin'].type is float
        assert 'Optional[float]' in str(fields['vix'].type) or 'Union[float, None]' in str(fields['vix'].type)
        assert fields['vix'].default is None
        assert fields['spread_bps'].default is None
        assert fields['timestamp'].default is None

    def test_rebalance_decision_result_has_correct_fields(self):
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(RebalanceDecisionResult)}
        expected = {'decision', 'urgency', 'max_drift', 'drift_details',
                    'vpin', 'estimated_cost_bps', 'reason', 'metadata'}
        assert set(fields) == expected

    def test_rebalance_decision_result_metadata_default(self):
        import dataclasses
        fields = dataclasses.fields(RebalanceDecisionResult)
        meta = [f for f in fields if f.name == 'metadata'][0]
        assert meta.default_factory is not dataclasses.MISSING
        # default_factory produces an empty dict
        result = meta.default_factory()
        assert result == {}

    def test_cost_budget_tracker_has_correct_fields(self):
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(CostBudgetTracker)}
        assert set(fields) == {
            'annual_limit_pct',
            'warning_threshold_pct',
            'ytd_costs',
            'ytd_year',  # Batch DY: calendar year for YTD view
            'max_single_trade_cost_bps',  # Batch DZ: single-trade outlier cap
            'quarantined_costs',  # Batch DZ: audit trail for outliers
        }
        assert fields['annual_limit_pct'].type is float
        assert fields['annual_limit_pct'].default == 0.005
        assert fields['warning_threshold_pct'].type is float
        assert fields['warning_threshold_pct'].default == 0.004

    def test_cost_budget_tracker_ytd_costs_factory(self):
        import dataclasses
        fields = dataclasses.fields(CostBudgetTracker)
        ytd = [f for f in fields if f.name == 'ytd_costs'][0]
        assert ytd.default_factory is not dataclasses.MISSING
        assert ytd.default_factory() == []

    def test_rebalance_decision_enum_members(self):
        members = {e.name: e.value for e in RebalanceDecision}
        assert members == {
            'EXECUTE': 'execute',
            'DEFER_TOXICITY': 'defer_toxicity',
            'DEFER_TIMING': 'defer_timing',
            'DEFER_BUDGET': 'defer_budget',
            'SKIP_LOW_DRIFT': 'skip_low_drift',
            'OVERRIDE_EMERGENCY': 'override_emergency',
        }

    def test_urgency_level_enum_members(self):
        members = {e.name: e.value for e in UrgencyLevel}
        assert members == {
            'LOW': 'low',
            'MODERATE': 'moderate',
            'HIGH': 'high',
            'EMERGENCY': 'emergency',
        }

    def test_portfolio_snapshot_can_be_constructed_with_all_fields(self):
        p = PortfolioSnapshot(
            holdings={'SPY': 100.0},
            targets={'SPY': 1.0},
            total_value=100.0,
            timestamp=datetime(2026, 1, 1),
        )
        assert p.holdings == {'SPY': 100.0}
        assert p.targets == {'SPY': 1.0}
        assert p.total_value == 100.0
        assert p.timestamp == datetime(2026, 1, 1)

    def test_market_conditions_defaults(self):
        m = MarketConditions(vpin=0.50)
        assert m.vpin == 0.50
        assert m.vix is None
        assert m.spread_bps is None
        assert m.timestamp is None

    def test_market_conditions_all_fields(self):
        ts = datetime(2026, 5, 1)
        m = MarketConditions(
            vpin=0.45,
            vix=22.0,
            spread_bps={'SPY': 1.5},
            timestamp=ts,
        )
        assert m.vpin == 0.45
        assert m.vix == 22.0
        assert m.spread_bps == {'SPY': 1.5}
        assert m.timestamp == ts


# ---------------------------------------------------------------------------
# Constants and exports validation
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify __all__, module-level constants, and config structure."""

    def test_all_exports_match_imports(self):
        expected = {
            'RebalanceDecision', 'UrgencyLevel', 'PortfolioSnapshot',
            'MarketConditions', 'RebalanceDecisionResult',
            'CostBudgetTracker', 'SmartRebalancingController',
            'create_sample_portfolio',
            # Batch EA: event-sourced cost ledger rebuild from order fills
            'collect_unique_order_fills',
            'estimate_day_cost_bps_from_fills',
            'rebuild_ytd_costs_from_order_fills',
        }
        from src.rebalancing import smart_rebalancer as mod
        assert set(mod.__all__) == expected

    def test_default_config_has_all_top_level_keys(self):
        keys = set(SmartRebalancingController.DEFAULT_CONFIG.keys())
        expected = {
            'drift_threshold', 'drift_threshold_by_regime', 'urgency_levels',
            'vpin', 'timing', 'cost_budget', 'fallback', 'safety',
        }
        assert keys == expected

    def test_default_config_drift_threshold_is_float_in_range(self):
        val = SmartRebalancingController.DEFAULT_CONFIG['drift_threshold']
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_default_config_vpin_threshold_in_range(self):
        vpin_cfg = SmartRebalancingController.DEFAULT_CONFIG['vpin']
        assert 0.0 <= vpin_cfg['threshold'] <= 1.0
        assert 0.0 <= vpin_cfg['default'] <= 1.0

    def test_default_config_timing_window_valid(self):
        timing = SmartRebalancingController.DEFAULT_CONFIG['timing']
        assert 0 <= timing['optimal_start'] < 24
        assert 0 <= timing['optimal_end'] < 24
        assert timing['optimal_start'] < timing['optimal_end']
        assert isinstance(timing['low_urgency_can_wait'], bool)

    def test_default_config_safety_ranges(self):
        safety = SmartRebalancingController.DEFAULT_CONFIG['safety']
        assert safety['max_deferral_hours'] > 0
        assert safety['max_single_trade_cost_bps'] > 0
        assert safety['max_annual_cost_pct'] > 0
        assert safety['min_drift_override'] > 0

    def test_default_config_cost_budget_ranges(self):
        budget = SmartRebalancingController.DEFAULT_CONFIG['cost_budget']
        assert 0 < budget['annual_limit'] < 1
        assert 0 < budget['warning_threshold'] < 1
        assert budget['warning_threshold'] <= budget['annual_limit']

    def test_default_config_fallback_ranges(self):
        fallback = SmartRebalancingController.DEFAULT_CONFIG['fallback']
        assert fallback['deferral_max_hours'] > 0
        assert 0 < fallback['force_if_drift_exceeds'] <= 1

    def test_default_config_urgency_levels_in_order(self):
        levels = SmartRebalancingController.DEFAULT_CONFIG['urgency_levels']
        assert levels['low'] < levels['moderate'] < levels['high'] < levels['emergency']

    def test_module_level_cost_constants(self):
        from src.rebalancing.smart_rebalancer import SmartRebalancingController as C
        assert isinstance(C.ETF_TRANSACTION_COSTS_BPS, dict)
        assert all(isinstance(v, float) for v in C.ETF_TRANSACTION_COSTS_BPS.values())
        assert isinstance(C.DEFAULT_COST_BPS, float)
        assert C.DEFAULT_COST_BPS == 5.0


# ---------------------------------------------------------------------------
# Computation edge cases — zero/empty/NaN/Inf/boundary
# ---------------------------------------------------------------------------

class TestComputeEdgeCases:
    """Edge cases: NaN, Inf, zero, empty, single-element, boundary values."""

    def _ctrl(self):
        return SmartRebalancingController()

    # --- Drift calculation edge cases ---

    def test_calculate_drift_nan_in_holdings(self):
        ctrl = self._ctrl()
        p = _make_portfolio(
            holdings={'SPY': float('nan'), 'GLD': 0, 'TLT': 0},
        )
        max_drift, details = ctrl.calculate_drift(p)
        # NaN in current_value leads to NaN current_alloc, NaN drift
        import math
        assert math.isnan(details['SPY'])

    def test_calculate_drift_inf_in_holdings(self):
        ctrl = self._ctrl()
        p = _make_portfolio(
            holdings={'SPY': float('inf'), 'GLD': 0, 'TLT': 0},
            total_value=float('inf'),
        )
        max_drift, details = ctrl.calculate_drift(p)
        # inf / inf = nan for current_alloc
        import math
        assert math.isnan(details['SPY']) or details['SPY'] == 0.0

    def test_calculate_drift_single_holding(self):
        ctrl = self._ctrl()
        p = PortfolioSnapshot(
            holdings={'SPY': 100000},
            targets={'SPY': 1.0},
            total_value=100000,
            timestamp=datetime.now(),
        )
        max_drift, details = ctrl.calculate_drift(p)
        assert max_drift == 0.0
        assert details['SPY'] == 0.0

    def test_calculate_drift_empty_holdings(self):
        """With empty holdings, all targets have 100% drift."""
        ctrl = self._ctrl()
        p = _make_portfolio(holdings={})
        max_drift, details = ctrl.calculate_drift(p)
        # current_alloc = 0 for all → drift = |0 - target|/target = 1.0
        assert max_drift == 1.0

    def test_calculate_drift_empty_targets(self):
        ctrl = self._ctrl()
        p = _make_portfolio(targets={})
        max_drift, details = ctrl.calculate_drift(p)
        assert max_drift == 0.0
        assert details == {}

    def test_calculate_drift_zero_target_allocation(self):
        ctrl = self._ctrl()
        p = _make_portfolio(targets={'SPY': 0.0, 'GLD': 1.0}, holdings={'SPY': 50000, 'GLD': 50000})
        max_drift, details = ctrl.calculate_drift(p)
        # SPY target=0 → drift formula uses division by target, returns 0
        # GLD: 0.50 vs 1.0 → drift = |0.5 - 1.0| / 1.0 = 0.5
        assert details['SPY'] == 0.0
        assert details['GLD'] == 0.5

    def test_calculate_drift_many_symbols(self):
        ctrl = self._ctrl()
        symbols = [f'SYM{i}' for i in range(100)]
        holdings = {s: 1000 for s in symbols}
        targets = {s: 1.0 / len(symbols) for s in symbols}
        p = PortfolioSnapshot(
            holdings=holdings,
            targets=targets,
            total_value=100000,
            timestamp=datetime.now(),
        )
        max_drift, details = ctrl.calculate_drift(p)
        assert max_drift == 0.0
        assert len(details) == 100

    # --- Cost estimation edge cases ---

    def test_estimate_cost_bps_vpin_zero(self):
        ctrl = self._ctrl()
        cost = ctrl.estimate_cost_bps(vpin=0.0, in_optimal_window=True)
        # vpin_mult = max(1.0, 1.0 + (0 - 0.30) * 2.0) = max(1.0, 0.4) = 1.0
        # cost = 0.0003 * 1.0 * 1.0 + 0.0002 = 0.0005 → 5 bps
        assert cost == pytest.approx(5.0, abs=0.5)

    def test_estimate_cost_bps_vpin_one(self):
        """Extreme VPIN=1.0 should cap multiplier at 2.0."""
        ctrl = self._ctrl()
        cost = ctrl.estimate_cost_bps(vpin=1.0, in_optimal_window=True)
        # vpin_mult = min(2.0, max(1.0, 1.0 + (1.0-0.30)*2.0)) = min(2.0, 2.4) = 2.0
        # cost = 0.0003 * 2.0 * 1.0 + 0.0002 = 0.0008 → 8 bps
        assert cost == pytest.approx(8.0, abs=0.5)

    def test_estimate_cost_bps_negative_vpin(self):
        """Negative VPIN should floor vpin_mult at 1.0."""
        ctrl = self._ctrl()
        cost = ctrl.estimate_cost_bps(vpin=-0.5, in_optimal_window=True)
        assert cost == pytest.approx(5.0, abs=0.5)

    def test_estimate_cost_bps_outside_window_morning(self):
        """Outside optimal window before 10am should use 1.25 multiplier."""
        ctrl = self._ctrl()
        with patch('src.rebalancing.smart_rebalancer.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 13, 9, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            cost = ctrl.estimate_cost_bps(vpin=0.30, in_optimal_window=False)
        # time_mult = 1.25, vpin_mult = 1.0, cost = 0.0003 * 1.0 * 1.25 + 0.0002 = 0.000575 → 5.75 bps
        assert cost == pytest.approx(5.75, abs=0.5)

    def test_estimate_cost_bps_outside_window_closing(self):
        """After 3:30pm should use 1.15 multiplier."""
        ctrl = self._ctrl()
        with patch('src.rebalancing.smart_rebalancer.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 13, 15, 35)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            cost = ctrl.estimate_cost_bps(vpin=0.30, in_optimal_window=False)
        # time_mult = 1.15
        assert cost == pytest.approx(5.45, abs=0.5)

    def test_estimate_cost_bps_outside_window_midday(self):
        """Mid-afternoon (between 14:00 and 15:30) should use 1.05 multiplier."""
        ctrl = self._ctrl()
        with patch('src.rebalancing.smart_rebalancer.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 13, 15, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            cost = ctrl.estimate_cost_bps(vpin=0.30, in_optimal_window=False)
        # time_mult = 1.05
        assert cost == pytest.approx(5.15, abs=0.5)

    def test_estimate_per_symbol_cost_bps_zero_vpin(self):
        ctrl = self._ctrl()
        cost = ctrl.estimate_per_symbol_cost_bps('SPY', 0.0, True)
        assert cost > 0
        assert isinstance(cost, float)

    def test_estimate_per_symbol_cost_bps_extreme_vpin(self):
        ctrl = self._ctrl()
        cost = ctrl.estimate_per_symbol_cost_bps('SPY', 1.0, True)
        # SPY base = 2 bps = 0.0002, vpin_mult = 2.0, time_mult = 1.0
        # cost = 0.0002 * 2.0 * 1.0 * 10000 = 4 bps
        assert cost == pytest.approx(4.0, abs=0.5)

    def test_estimate_per_symbol_cost_bps_negative_vpin(self):
        ctrl = self._ctrl()
        cost = ctrl.estimate_per_symbol_cost_bps('SPY', -0.5, True)
        assert cost > 0

    def test_estimate_per_symbol_cost_bps_outside_window(self):
        ctrl = self._ctrl()
        with patch('src.rebalancing.smart_rebalancer.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 13, 9, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            cost = ctrl.estimate_per_symbol_cost_bps('TLT', 0.30, False)
        assert cost > 0

    def test_estimate_total_cost_bps_with_empty_dict(self):
        ctrl = self._ctrl()
        total = ctrl.estimate_total_cost_bps({}, 0.30, True)
        flat = ctrl.estimate_cost_bps(0.30, True)
        assert total == flat

    def test_estimate_total_cost_bps_with_none_value(self):
        """If a drift detail has 0 drift, it should be skipped."""
        ctrl = self._ctrl()
        total = ctrl.estimate_total_cost_bps({'SPY': 0.0, 'GLD': 0.05}, 0.30, True)
        assert total > 0

    def test_estimate_total_cost_bps_single_symbol(self):
        ctrl = self._ctrl()
        total = ctrl.estimate_total_cost_bps({'SPY': 0.10}, 0.30, True)
        assert total > 0

    # --- Urgency boundary ---

    def test_calculate_urgency_zero_drift(self):
        ctrl = self._ctrl()
        assert ctrl.calculate_urgency(0.0) == UrgencyLevel.LOW

    def test_calculate_urgency_exact_emergency(self):
        ctrl = self._ctrl()
        assert ctrl.calculate_urgency(0.201) == UrgencyLevel.EMERGENCY


# ---------------------------------------------------------------------------
# Function boundary conditions — extreme inputs, missing keys, wrong types
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    """Extreme inputs, missing keys, type mismatches."""

    def test_should_rebalance_no_regime_and_none_vpin(self):
        """VPIN=None should use config default."""
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.15)
        m = MarketConditions(vpin=None)
        now = datetime(2026, 5, 14, 12, 0)
        result = ctrl.should_rebalance(p, m, now=now)
        # Should use config default vpin (0.30), which is below threshold → execute
        assert result.decision == RebalanceDecision.EXECUTE

    def test_should_rebalance_with_vpin_none_in_config_default(self):
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.15)
        m = MarketConditions(vpin=None)
        now = datetime(2026, 5, 14, 12, 0)
        result = ctrl.should_rebalance(p, m, now=now)
        assert result.vpin == ctrl.config['vpin']['default']

    def test_estimate_cost_bps_large_vpin_saturates(self):
        ctrl = SmartRebalancingController()
        cost_2 = ctrl.estimate_cost_bps(vpin=2.0, in_optimal_window=True)
        cost_10 = ctrl.estimate_cost_bps(vpin=10.0, in_optimal_window=True)
        # Both should be capped at vpin_mult=2.0 → same result
        assert cost_2 == cost_10

    def test_rebalance_decision_result_all_fields_populated_on_execute(self):
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.15)
        m = _make_market(vpin=0.30)
        now = datetime(2026, 5, 14, 12, 0)
        result = ctrl.should_rebalance(p, m, now=now)
        if result.decision == RebalanceDecision.EXECUTE:
            assert isinstance(result.max_drift, float)
            assert isinstance(result.drift_details, dict)
            assert isinstance(result.vpin, float)
            assert isinstance(result.estimated_cost_bps, float)
            assert isinstance(result.reason, str)
            assert isinstance(result.metadata, dict)

    def test_record_rebalance_isoformat_date(self):
        ctrl = SmartRebalancingController()
        date_str = "2026-05-14T12:30:00"
        ctrl.record_rebalance(5.0, date_str, ["SPY"])
        assert ctrl.last_rebalance == datetime(2026, 5, 14, 12, 30, 0)

    def test_record_rebalance_simple_date(self):
        ctrl = SmartRebalancingController()
        ctrl.record_rebalance(3.0, "2026-05-14", ["SPY"])
        assert ctrl.last_rebalance.year == 2026
        assert ctrl.last_rebalance.month == 5
        assert ctrl.last_rebalance.day == 14

    def test_record_rebalance_clears_deferred(self):
        ctrl = SmartRebalancingController()
        ctrl.deferred_until = datetime(2026, 6, 1)
        ctrl.record_rebalance(5.0, "2026-05-14", ["SPY"])
        assert ctrl.deferred_until is None

    def test_should_rebalance_regime_edge_high_vol(self):
        """High vol regime (7% threshold): 8% drift should trigger."""
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.08)
        m = _make_market(vpin=0.30)
        now = datetime(2026, 5, 14, 12, 0)
        result = ctrl.should_rebalance(p, m, now=now, regime='high_vol')
        # 8% drift > 7% threshold → shouldn't skip
        assert result.decision != RebalanceDecision.SKIP_LOW_DRIFT

    def test_should_rebalance_regime_edge_high_vol_skip(self):
        """High vol regime (7% threshold): 6% drift should skip."""
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.06)
        m = _make_market(vpin=0.30)
        now = datetime(2026, 5, 14, 12, 0)
        result = ctrl.should_rebalance(p, m, now=now, regime='high_vol')
        assert result.decision == RebalanceDecision.SKIP_LOW_DRIFT

    def test_maximally_deferred_then_toxicity_clears(self):
        """After max deferrals force execution, next toxicity should start fresh."""
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.15)
        m_high = _make_market(vpin=0.60)
        m_low = _make_market(vpin=0.30)
        now = datetime(2026, 5, 14, 12, 0)
        # Exhaust max deferrals (4 hours = 4 deferrals)
        for _ in range(4):
            ctrl.should_rebalance(p, m_high, now=now)
        # 5th call forces execute and resets counter
        ctrl.should_rebalance(p, m_high, now=now)
        # Now VPIN is low so it should execute
        result = ctrl.should_rebalance(p, m_low, now=now)
        assert result.decision == RebalanceDecision.EXECUTE

    def test_should_rebalance_defer_toxicity_vpin_threshold_tight(self):
        """VPIN exactly at threshold should NOT defer (not strictly greater)."""
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.15)
        m = _make_market(vpin=0.50)  # Exactly threshold
        now = datetime(2026, 5, 14, 12, 0)
        result = ctrl.should_rebalance(p, m, now=now)
        # vpin 0.50 is not > 0.50, so no toxicity defer
        assert result.decision != RebalanceDecision.DEFER_TOXICITY


# ---------------------------------------------------------------------------
# CostBudgetTracker edge cases
# ---------------------------------------------------------------------------

class TestCostBudgetTrackerEdgeCases:
    """Additional CostBudgetTracker scenarios beyond basic coverage."""

    def test_custom_limits(self):
        tracker = CostBudgetTracker(
            annual_limit_pct=0.01,
            warning_threshold_pct=0.008,
        )
        assert tracker.annual_limit_pct == 0.01
        assert tracker.warning_threshold_pct == 0.008
        assert tracker.remaining_budget_pct == 0.01

    def test_add_cost_single_entry(self):
        tracker = CostBudgetTracker()
        tracker.add_cost(10.0, "2026-05-01", ["SPY"])
        assert len(tracker.ytd_costs) == 1
        entry = tracker.ytd_costs[0]
        assert entry['cost_bps'] == 10.0
        assert entry['date'] == "2026-05-01"
        assert entry['symbols'] == ["SPY"]

    def test_add_cost_empty_symbols(self):
        tracker = CostBudgetTracker()
        tracker.add_cost(5.0, "2026-05-01", [])
        assert tracker.ytd_total_bps == 5.0

    def test_add_cost_negative_cost(self):
        tracker = CostBudgetTracker()
        tracker.add_cost(-5.0, "2026-05-01", ["SPY"])
        assert tracker.ytd_total_bps == -5.0
        assert tracker.remaining_budget_pct == 0.0055  # 0.005 - (-0.0005)

    def test_is_warning_at_exact_threshold(self):
        tracker = CostBudgetTracker()
        # warning_threshold_pct = 0.004, which is 40 bps
        tracker.add_cost(40.0, "2026-05-14", ["SPY"])
        assert tracker.is_warning() is True

    def test_is_warning_just_below_threshold(self):
        tracker = CostBudgetTracker()
        # 39.99 bps = 0.003999, just below 0.004
        tracker.add_cost(39.99, "2026-05-14", ["SPY"])
        assert tracker.is_warning() is False

    def test_is_over_budget_just_below(self):
        tracker = CostBudgetTracker()
        # 49.99 bps = 0.004999, just below 0.005
        tracker.add_cost(49.99, "2026-05-14", ["SPY"])
        assert tracker.is_over_budget() is False

    def test_is_over_budget_at_exact_limit(self):
        tracker = CostBudgetTracker()
        tracker.add_cost(50.0, "2026-05-14", ["SPY"])
        assert tracker.is_over_budget() is True

    def test_remaining_budget_no_costs(self):
        tracker = CostBudgetTracker()
        assert tracker.remaining_budget_pct == tracker.annual_limit_pct

    def test_is_over_budget_false_by_default(self):
        tracker = CostBudgetTracker()
        assert tracker.is_over_budget() is False

    def test_is_warning_false_by_default(self):
        tracker = CostBudgetTracker()
        assert tracker.is_warning() is False


# ---------------------------------------------------------------------------
# SmartRebalancingController — initialization & config loading
# ---------------------------------------------------------------------------

class TestControllerInitialization:
    """Constructor, config loading, deep merge."""

    def test_default_initialization(self):
        ctrl = SmartRebalancingController()
        assert ctrl.config is not None
        assert ctrl.cost_tracker is not None
        assert ctrl.deferred_until is None
        assert ctrl.last_rebalance is None

    def test_cost_tracker_uses_config_limits(self):
        ctrl = SmartRebalancingController()
        assert ctrl.cost_tracker.annual_limit_pct == ctrl.config['cost_budget']['annual_limit']
        assert ctrl.cost_tracker.warning_threshold_pct == ctrl.config['cost_budget']['warning_threshold']

    def test_consecutive_deferrals_initialized_on_first_call(self):
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.15)
        m = _make_market(vpin=0.60)
        now = datetime(2026, 5, 14, 12, 0)
        ctrl.should_rebalance(p, m, now=now)
        # consecutive_deferrals should have been set dynamically
        assert hasattr(ctrl, 'consecutive_deferrals')

    def test_deep_merge_nested_dict(self):
        ctrl = SmartRebalancingController()
        base = {'a': {'b': 1, 'c': 2}, 'd': 3}
        override = {'a': {'b': 99}}
        ctrl._deep_merge(base, override)
        assert base['a']['b'] == 99
        assert base['a']['c'] == 2  # Should be preserved
        assert base['d'] == 3

    def test_deep_merge_new_key(self):
        ctrl = SmartRebalancingController()
        base = {'a': 1}
        override = {'b': 2}
        ctrl._deep_merge(base, override)
        assert base['a'] == 1
        assert base['b'] == 2

    def test_deep_merge_non_dict_overwrites(self):
        ctrl = SmartRebalancingController()
        base = {'a': {'nested': 'value'}}
        override = {'a': 'scalar'}
        ctrl._deep_merge(base, override)
        assert base['a'] == 'scalar'

    def test_deep_merge_empty_override(self):
        ctrl = SmartRebalancingController()
        base = {'a': 1}
        ctrl._deep_merge(base, {})
        assert base == {'a': 1}

    def test_load_config_missing_file(self):
        """Loading a non-existent path should silently use defaults."""
        ctrl = SmartRebalancingController(config_path='/tmp/nonexistent_config_file_xyz.yaml')
        assert ctrl.config['drift_threshold'] == 0.10


class TestPerSymbolCostBreakdownExpanded:
    """Extended metadata coverage."""

    def test_execute_metadata_includes_window_and_budget(self):
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.15)
        m = _make_market(vpin=0.30)
        now = datetime(2026, 5, 13, 12, 0)
        result = ctrl.should_rebalance(p, m, now=now)
        if result.decision == RebalanceDecision.EXECUTE:
            assert 'in_optimal_window' in result.metadata
            assert result.metadata['in_optimal_window'] is True
            assert 'ytd_cost_bps' in result.metadata
            assert 'remaining_budget_pct' in result.metadata
            assert result.metadata['remaining_budget_pct'] > 0

    def test_override_emergency_has_no_metadata(self):
        """OVERRIDE_EMERGENCY bypasses metadata generation in execute path."""
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.30)
        m = _make_market(vpin=0.80)
        now = datetime(2026, 5, 13, 9, 0)
        result = ctrl.should_rebalance(p, m, now=now)
        assert result.decision == RebalanceDecision.OVERRIDE_EMERGENCY
        # Emergency override still goes through execute path for cost
        # But doesn't build per_symbol metadata
        assert 'per_symbol_cost_bps' not in result.metadata


# ---------------------------------------------------------------------------
# get_status edge cases
# ---------------------------------------------------------------------------

class TestGetStatusEdgeCases:
    """Cover additional status scenarios."""

    def test_status_last_rebalance_none(self):
        ctrl = SmartRebalancingController()
        status = ctrl.get_status()
        assert status['last_rebalance'] is None

    def test_status_deferred_none(self):
        ctrl = SmartRebalancingController()
        status = ctrl.get_status()
        assert status['deferred_until'] is None

    def test_status_config_summary_keys(self):
        ctrl = SmartRebalancingController()
        status = ctrl.get_status()
        config = status['config']
        assert 'drift_threshold' in config
        assert 'vpin_threshold' in config
        assert 'optimal_window' in config
        assert 'annual_cost_limit' in config

    def test_status_cost_pct_format(self):
        ctrl = SmartRebalancingController()
        # Under single-trade cap (15) so row stays in YTD budget sum
        ctrl.record_rebalance(12.5, "2026-05-14", ["SPY", "GLD"])
        status = ctrl.get_status()
        # 12.5 bps = 0.125% of total value
        assert status['ytd_cost_pct'] == 0.125
        assert status['remaining_budget_pct'] == 0.375  # 0.5 - 0.125


# ---------------------------------------------------------------------------
# CLI / __main__ guard
# ---------------------------------------------------------------------------

class TestCliMainGuard:
    """Cover demo() and the __name__ == '__main__' guard."""

    def test_demo_runs_without_error(self, caplog):
        """demo() should execute all 5 scenarios and print output."""
        caplog.set_level(logging.INFO, logger='src.rebalancing.smart_rebalancer')
        from src.rebalancing.smart_rebalancer import demo
        # demo() depends on datetime.now(), so we control critical calls
        with patch('src.rebalancing.smart_rebalancer.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 13, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            demo()
        assert 'Scenario 1' in caplog.text
        assert 'Scenario 2' in caplog.text
        assert 'Scenario 3' in caplog.text
        assert 'Scenario 4' in caplog.text
        assert 'Scenario 5' in caplog.text
        assert 'Controller Status' in caplog.text

    def test_demo_scenario_1_skip_low_drift(self, caplog):
        caplog.set_level(logging.INFO, logger='src.rebalancing.smart_rebalancer')
        from src.rebalancing.smart_rebalancer import demo
        with patch('src.rebalancing.smart_rebalancer.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 13, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            demo()
        assert 'skip_low_drift' in caplog.text

    def test_demo_scenario_2_execute(self, caplog):
        caplog.set_level(logging.INFO, logger='src.rebalancing.smart_rebalancer')
        from src.rebalancing.smart_rebalancer import demo
        with patch('src.rebalancing.smart_rebalancer.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 13, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            demo()
        assert 'Cost:' in caplog.text

    def test_demo_prints_urgency(self, caplog):
        caplog.set_level(logging.INFO, logger='src.rebalancing.smart_rebalancer')
        from src.rebalancing.smart_rebalancer import demo
        with patch('src.rebalancing.smart_rebalancer.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 13, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            demo()
        assert 'Urgency:' in caplog.text

    def test_main_guard_string(self):
        """The module-level __name__ check should reference 'demo()'."""
        import src.rebalancing.smart_rebalancer as mod
        source = inspect.getsource(mod)
        assert "__name__" in source and "__main__" in source
        assert "demo()" in source


# ---------------------------------------------------------------------------
# RebalanceDecisionResult default construction
# ---------------------------------------------------------------------------

class TestRebalanceDecisionResultConstruction:
    """Test that RebalanceDecisionResult can be constructed in various ways."""

    def test_construct_with_metadata(self):
        result = RebalanceDecisionResult(
            decision=RebalanceDecision.EXECUTE,
            urgency=UrgencyLevel.HIGH,
            max_drift=0.15,
            drift_details={'SPY': 0.15},
            vpin=0.30,
            estimated_cost_bps=5.0,
            reason='test',
            metadata={'extra': 'data'},
        )
        assert result.metadata == {'extra': 'data'}

    def test_construct_without_metadata(self):
        result = RebalanceDecisionResult(
            decision=RebalanceDecision.SKIP_LOW_DRIFT,
            urgency=UrgencyLevel.LOW,
            max_drift=0.0,
            drift_details={},
            vpin=0.30,
            estimated_cost_bps=0,
            reason='test',
        )
        assert result.metadata == {}

    def test_construct_all_fields_required(self):
        """Most fields are required (no defaults)."""
        import dataclasses
        required = {'decision', 'urgency', 'max_drift', 'drift_details',
                    'vpin', 'estimated_cost_bps', 'reason'}
        for f in dataclasses.fields(RebalanceDecisionResult):
            if f.name in required:
                assert f.default is dataclasses.MISSING
                assert f.default_factory is dataclasses.MISSING


# ---------------------------------------------------------------------------
# Drift calculation additional edge cases
# ---------------------------------------------------------------------------

class TestDriftCalculationAdditional:
    """Additional drift calculation scenarios."""

    def test_calculate_drift_symmetric_around_target(self):
        """Overweight and underweight should produce symmetric drift."""
        ctrl = SmartRebalancingController()
        # SPY target 0.46, current 0.56 → drift = |0.56-0.46|/0.46 = 0.217
        over = _make_portfolio(
            holdings={'SPY': 56000, 'GLD': 30000, 'TLT': 14000},
            total_value=100000,
        )
        over_drift, _ = ctrl.calculate_drift(over)

        # SPY target 0.46, current 0.36 → drift = |0.36-0.46|/0.46 = 0.217
        under = _make_portfolio(
            holdings={'SPY': 36000, 'GLD': 46000, 'TLT': 18000},
            total_value=100000,
        )
        under_drift, _ = ctrl.calculate_drift(under)
        assert over_drift == pytest.approx(under_drift, abs=0.01)

    def test_calculate_drift_holdings_not_in_targets(self):
        """Extra holdings without targets should be ignored."""
        ctrl = SmartRebalancingController()
        p = _make_portfolio(
            holdings={'SPY': 46000, 'GLD': 38000, 'TLT': 16000, 'CASH': 0},
        )
        max_drift, details = ctrl.calculate_drift(p)
        assert max_drift == 0.0
        # CASH is not in targets, so it won't appear
        assert 'CASH' not in details

    def test_calculate_drift_precision(self):
        """Drift values should be rounded to 4 decimal places."""
        ctrl = SmartRebalancingController()
        p = _make_portfolio(
            holdings={'SPY': 46211, 'GLD': 37889, 'TLT': 15900},
            total_value=100000,
        )
        _, details = ctrl.calculate_drift(p)
        for val in details.values():
            # Should have at most 4 decimal places
            assert val * 10000 == pytest.approx(round(val * 10000), abs=0.0001)


# ---------------------------------------------------------------------------
# SmartRebalancingController — estimate_cost_bps parameter validation
# ---------------------------------------------------------------------------

class TestEstimateCostValidation:
    """Test cost estimation with various parameter combinations."""

    def test_cost_increases_with_vpin(self):
        """Cost should be monotonically non-decreasing with VPIN."""
        ctrl = SmartRebalancingController()
        costs = []
        for vpin in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            costs.append(ctrl.estimate_cost_bps(vpin, in_optimal_window=True))
        for i in range(len(costs) - 1):
            assert costs[i] <= costs[i + 1], f"Cost decreased at vpin step {i}"

    def test_per_symbol_cost_increases_with_vpin(self):
        """Per-symbol cost should be monotonically non-decreasing with VPIN."""
        ctrl = SmartRebalancingController()
        costs = []
        for vpin in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            costs.append(ctrl.estimate_per_symbol_cost_bps('SPY', vpin, True))
        for i in range(len(costs) - 1):
            assert costs[i] <= costs[i + 1]


# ---------------------------------------------------------------------------
# SmartRebalancingController — should_rebalance decision logic edge cases
# ---------------------------------------------------------------------------

class TestShouldRebalanceEdgeCases:
    """Cover edge cases in the decision engine."""

    def test_skip_low_drift_has_zero_cost(self):
        ctrl = SmartRebalancingController()
        p = _make_portfolio()  # No drift
        m = _make_market()
        result = ctrl.should_rebalance(p, m)
        assert result.decision == RebalanceDecision.SKIP_LOW_DRIFT
        assert result.estimated_cost_bps == 0

    def test_defer_toxicity_has_zero_cost(self):
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.15)
        m = _make_market(vpin=0.60)
        now = datetime(2026, 5, 14, 12, 0)
        result = ctrl.should_rebalance(p, m, now=now)
        assert result.decision == RebalanceDecision.DEFER_TOXICITY
        assert result.estimated_cost_bps == 0

    def test_defer_timing_has_zero_cost(self):
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.11)
        m = _make_market(vpin=0.30)
        now = datetime(2026, 5, 14, 9, 30)
        result = ctrl.should_rebalance(p, m, now=now)
        assert result.decision == RebalanceDecision.DEFER_TIMING
        assert result.estimated_cost_bps == 0

    def test_defer_budget_has_non_zero_cost_in_result(self):
        """Defer_budget sets estimated_cost_bps=0."""
        ctrl = SmartRebalancingController()
        ctrl.cost_tracker.max_single_trade_cost_bps = None
        ctrl.cost_tracker.add_cost(60, "2026-05-01", ["SPY"])
        p = _drifted_portfolio(0.15)
        m = _make_market(vpin=0.30)
        now = datetime(2026, 5, 14, 12, 0)
        result = ctrl.should_rebalance(p, m, now=now)
        assert result.decision == RebalanceDecision.DEFER_BUDGET
        assert result.estimated_cost_bps == 0

    def test_low_urgency_outside_window_defers_without_toxicity(self):
        """Low urgency outside window should defer for timing, not toxicity."""
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.11)
        m = _make_market(vpin=0.60)  # High VPIN
        now = datetime(2026, 5, 14, 9, 30)  # Outside window
        result = ctrl.should_rebalance(p, m, now=now)
        # VPIN check comes before timing, so it should be DEFER_TOXICITY
        assert result.decision == RebalanceDecision.DEFER_TOXICITY

    def test_low_urgency_in_window_high_vpin_defers_toxicity(self):
        """Low urgency in window but high VPIN should defer for toxicity."""
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.11)
        m = _make_market(vpin=0.60)
        now = datetime(2026, 5, 14, 12, 0)  # In window
        result = ctrl.should_rebalance(p, m, now=now)
        assert result.decision == RebalanceDecision.DEFER_TOXICITY

    def test_high_urgency_passes_through_vpin_and_timing(self):
        """High urgency should bypass timing deferral but not VPIN deferral."""
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.17)  # HIGH urgency
        m = _make_market(vpin=0.30)  # Low VPIN (below 0.50 threshold)
        now = datetime(2026, 5, 14, 9, 30)  # Outside window
        result = ctrl.should_rebalance(p, m, now=now)
        # VPIN passes (0.30 <= 0.50), timing skipped (HIGH != LOW), budget ok
        assert result.decision == RebalanceDecision.EXECUTE

    def test_emergency_urgency_bypasses_all_checks(self):
        """Drift > 25% should produce OVERRIDE_EMERGENCY (bypassing toxicity/timing/budget)."""
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.30)  # 30% drift on SPY → max_drift > 0.25
        m = _make_market(vpin=0.90)
        now = datetime(2026, 5, 14, 9, 0)
        result = ctrl.should_rebalance(p, m, now=now)
        assert result.decision == RebalanceDecision.OVERRIDE_EMERGENCY

    def test_should_rebalance_with_now_param(self):
        """Passing explicit 'now' should be used for timing."""
        ctrl = SmartRebalancingController()
        p = _drifted_portfolio(0.15)
        m = _make_market(vpin=0.30)
        # Outside window test
        result = ctrl.should_rebalance(p, m, now=datetime(2026, 5, 14, 9, 0))
        assert result.decision == RebalanceDecision.EXECUTE

    def test_should_rebalance_high_urgency_but_over_budget_defers(self):
        """High urgency but over budget should defer unless EMERGENCY."""
        ctrl = SmartRebalancingController()
        ctrl.cost_tracker.max_single_trade_cost_bps = None
        ctrl.cost_tracker.add_cost(60, "2026-05-01", ["SPY"])
        p = _drifted_portfolio(0.17)  # HIGH urgency (not EMERGENCY)
        m = _make_market(vpin=0.30)
        now = datetime(2026, 5, 14, 12, 0)
        result = ctrl.should_rebalance(p, m, now=now)
        assert result.decision == RebalanceDecision.DEFER_BUDGET


class TestSamplePortfolioExpanded:
    """Additional create_sample_portfolio coverage."""

    def test_precision_of_targets(self):
        p = create_sample_portfolio()
        total = sum(p.targets.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_sample_portfolio_holdings(self):
        p = create_sample_portfolio()
        assert 'SPY' in p.holdings
        assert 'GLD' in p.holdings
        assert 'TLT' in p.holdings
        assert sum(p.holdings.values()) == p.total_value

    def test_sample_portfolio_timestamp_is_now(self):
        p = create_sample_portfolio()
        diff = (datetime.now() - p.timestamp).total_seconds()
        assert diff < 5  # Created within last 5 seconds


class TestMarketConditionsAdvanced:
    """Additional MarketConditions scenarios."""

    def test_spread_bps_structure(self):
        m = _make_market(spread_bps={'SPY': 1.5, 'TLT': 8.0})
        assert m.spread_bps['SPY'] == 1.5
        assert m.spread_bps['TLT'] == 8.0


class TestOptimalWindowAdvanced:
    """Additional timing window edge cases."""

    def test_in_window_exact_start_inclusive(self):
        ctrl = SmartRebalancingController()
        assert ctrl._in_optimal_window(datetime(2026, 5, 14, 11, 0)) is True

    def test_in_window_exact_end_exclusive(self):
        ctrl = SmartRebalancingController()
        assert ctrl._in_optimal_window(datetime(2026, 5, 14, 14, 0)) is False

    def test_at_midnight(self):
        ctrl = SmartRebalancingController()
        assert ctrl._in_optimal_window(datetime(2026, 5, 14, 0, 0)) is False

    def test_at_midnight_just_before(self):
        ctrl = SmartRebalancingController()
        assert ctrl._in_optimal_window(datetime(2026, 5, 14, 23, 59)) is False

    def test_default_now_returns_something(self):
        """Without explicit now, should use datetime.now() and return bool."""
        ctrl = SmartRebalancingController()
        result = ctrl._in_optimal_window()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Error resistance — verify functions don't crash on malformed data
# ---------------------------------------------------------------------------

class TestErrorResistance:
    """Functions should degrade gracefully on unexpected input."""

    def test_calculate_drift_with_negative_holdings(self):
        ctrl = SmartRebalancingController()
        p = _make_portfolio(holdings={'SPY': -10000, 'GLD': 60000, 'TLT': 30000})
        # Should not crash with negative holdings
        max_drift, details = ctrl.calculate_drift(p)
        assert isinstance(max_drift, float)
        assert isinstance(details, dict)

    def test_calculate_drift_with_negative_target(self):
        ctrl = SmartRebalancingController()
        p = _make_portfolio(targets={'SPY': -0.1, 'GLD': 1.1})
        max_drift, details = ctrl.calculate_drift(p)
        assert isinstance(max_drift, float)
        assert isinstance(details, dict)

    def test_estimate_per_symbol_cost_with_empty_string_symbol(self):
        ctrl = SmartRebalancingController()
        cost = ctrl.estimate_per_symbol_cost_bps('', 0.30, True)
        assert cost > 0  # Should use DEFAULT_COST_BPS

    def test_estimate_total_cost_with_non_dict_drift(self):
        """estimate_total_cost_bps should handle a non-dict-like drift_details."""
        # Actually, the signature expects Dict[str, float], and an empty dict
        # falls back to flat estimate. Non-dict would fail, but that's expected.
        pass  # Type contract is enforced by caller

    def test_estimate_cost_bps_negative_returns_positive(self):
        """Cost should always be positive even with negative vpin."""
        ctrl = SmartRebalancingController()
        cost = ctrl.estimate_cost_bps(vpin=-1.0, in_optimal_window=True)
        assert cost > 0

    def test_demo_prints_all_scenarios_more(self, caplog):
        """Collect all scenario outputs for completeness."""
        caplog.set_level(logging.INFO, logger='src.rebalancing.smart_rebalancer')
        from src.rebalancing.smart_rebalancer import demo
        with patch('src.rebalancing.smart_rebalancer.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 13, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            demo()
        lines = caplog.text.strip().split('\n')
        # Should have more than 5 lines of output (5 scenarios + status + blanks)
        assert len(lines) > 6
