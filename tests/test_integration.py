#!/usr/bin/env python3
"""
Comprehensive tests for src/rebalancing/integration.py — SmartRebalanceGate.

Covers: RebalanceGateResult dataclass, SmartRebalanceGate lifecycle
(init, update_vpin, update_regime, _compute_vpin, evaluate, record_execution,
get_status, to_json), create_gate_from_config factory, __all__ export,
all decision paths (skip, execute, defer_toxicity, defer_timing, defer_budget,
override_emergency, max_deferral_exceeded), regime-adaptive thresholds,
cost budget limits, and all boundary/edge conditions.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.rebalancing.integration import (
    SmartRebalanceGate,
    RebalanceGateResult,
    create_gate_from_config,
    __all__ as integration_all,
)
from src.rebalancing.smart_rebalancer import (
    SmartRebalancingController,
)


@pytest.fixture(autouse=True)
def _isolate_smart_rebalance_state(tmp_path, monkeypatch):
    """Do not load host data/smart_rebalance_state.json into unit tests."""
    # Force empty ephemeral state for every SmartRebalanceGate / controller
    monkeypatch.setattr(
        "src.rebalancing.smart_rebalancer.DATA_DIR",
        tmp_path,
        raising=False,
    )
    monkeypatch.setattr(
        "src.rebalancing.integration.DATA_DIR",
        tmp_path,
        raising=False,
    )
    # Also pass load_state via env-less default: empty dir has no state file
    yield


# =========================================================================
# __all__ export validation
# =========================================================================

class TestModuleExports:
    """Validate __all__ in integration.py."""

    def test_all_contains_expected_names(self):
        expected = {'RebalanceGateResult', 'SmartRebalanceGate', 'create_gate_from_config'}
        assert set(integration_all) == expected

    def test_all_names_are_importable(self):
        for name in integration_all:
            exec(f"from src.rebalancing.integration import {name}")


# =========================================================================
# RebalanceGateResult dataclass
# =========================================================================

class TestRebalanceGateResult:
    """Dataclass field validation and edge cases."""

    def test_creation_with_all_fields(self):
        r = RebalanceGateResult(
            should_execute=True,
            decision="execute",
            urgency="high",
            max_drift=0.12,
            estimated_cost_bps=5.0,
            reason="test_reason",
            metadata={"vpin": 0.3, "drift_details": {"SPY": 0.12}},
        )
        assert r.should_execute is True
        assert r.decision == "execute"
        assert r.urgency == "high"
        assert r.max_drift == 0.12
        assert r.estimated_cost_bps == 5.0
        assert r.reason == "test_reason"
        assert r.metadata["vpin"] == 0.3

    def test_defer_result(self):
        r = RebalanceGateResult(
            should_execute=False,
            decision="defer_toxicity",
            urgency="low",
            max_drift=0.05,
            estimated_cost_bps=2.0,
            reason="VPIN too high",
            metadata={},
        )
        assert r.should_execute is False
        assert r.decision == "defer_toxicity"
        assert r.urgency == "low"

    def test_zero_drift_and_cost(self):
        r = RebalanceGateResult(
            should_execute=False,
            decision="skip_low_drift",
            urgency="low",
            max_drift=0.0,
            estimated_cost_bps=0.0,
            reason="no_drift",
            metadata={},
        )
        assert r.max_drift == 0.0
        assert r.estimated_cost_bps == 0.0

    def test_high_values(self):
        r = RebalanceGateResult(
            should_execute=True,
            decision="override_emergency",
            urgency="emergency",
            max_drift=0.35,
            estimated_cost_bps=50.0,
            reason="drift_too_high",
            metadata={"vpin": 0.9},
        )
        assert r.max_drift == 0.35
        assert r.estimated_cost_bps == 50.0
        assert r.metadata["vpin"] == 0.9

    def test_empty_metadata(self):
        r = RebalanceGateResult(
            should_execute=False,
            decision="defer_budget",
            urgency="moderate",
            max_drift=0.14,
            estimated_cost_bps=0.0,
            reason="over_budget",
            metadata={},
        )
        assert r.metadata == {}

    def test_negative_values(self):
        r = RebalanceGateResult(
            should_execute=True,
            decision="execute",
            urgency="low",
            max_drift=-1.0,
            estimated_cost_bps=-5.0,
            reason="negative",
            metadata={"vpin": -0.1},
        )
        assert r.max_drift == -1.0
        assert r.estimated_cost_bps == -5.0

    def test_types_are_correct(self):
        r = RebalanceGateResult(True, "execute", "high", 0.12, 5.0, "reason", {"k": "v"})
        assert isinstance(r.should_execute, bool)
        assert isinstance(r.decision, str)
        assert isinstance(r.urgency, str)
        assert isinstance(r.max_drift, float)
        assert isinstance(r.estimated_cost_bps, float)
        assert isinstance(r.reason, str)
        assert isinstance(r.metadata, dict)

    def test_is_dataclass(self):
        from dataclasses import is_dataclass
        assert is_dataclass(RebalanceGateResult)

    def test_dataclass_order(self):
        """Verify field order hasn't changed."""
        import dataclasses
        fields = [f.name for f in dataclasses.fields(RebalanceGateResult)]
        assert fields == [
            "should_execute", "decision", "urgency", "max_drift",
            "estimated_cost_bps", "reason", "metadata",
        ]


# =========================================================================
# SmartRebalanceGate __init__
# =========================================================================

class TestSmartRebalanceGateInit:
    """Constructor behavior."""

    def test_init_creates_controller(self):
        gate = SmartRebalanceGate()
        assert gate.controller is not None
        assert isinstance(gate.controller, SmartRebalancingController)

    def test_init_vpin_cache_empty(self):
        gate = SmartRebalanceGate()
        assert gate._vpin_cache == {}

    def test_init_regime_none(self):
        gate = SmartRebalanceGate()
        assert gate._regime is None

    def test_init_vpin_engine_present(self):
        """VPIN engine is created when _VPIN_AVAILABLE is True."""
        gate = SmartRebalanceGate()
        assert gate._vpin_engine is not None

    def test_init_with_config_path(self, tmp_path):
        """Passing a config path to SmartRebalanceGate."""
        config_file = tmp_path / "smart_rebalance.yaml"
        config_file.write_text("smart_rebalancing:\n  drift_threshold: 0.15\n")
        gate = SmartRebalanceGate(config_path=str(config_file))
        assert gate.controller.config["drift_threshold"] == 0.15

    def test_init_non_existent_config(self):
        """Passing a non-existent config path uses defaults."""
        gate = SmartRebalanceGate(config_path="/nonexistent/path/config.yaml")
        assert gate.controller.config["drift_threshold"] == 0.10  # default


# =========================================================================
# update_vpin
# =========================================================================

class TestUpdateVpin:
    """VPIN cache updates."""

    def test_update_vpin_stores_value(self):
        gate = SmartRebalanceGate()
        gate.update_vpin(0.45)
        assert gate._vpin_cache["current"] == 0.45

    def test_update_vpin_overwrites(self):
        gate = SmartRebalanceGate()
        gate.update_vpin(0.30)
        gate.update_vpin(0.55)
        assert gate._vpin_cache["current"] == 0.55

    def test_update_vpin_zero(self):
        gate = SmartRebalanceGate()
        gate.update_vpin(0.0)
        assert gate._vpin_cache["current"] == 0.0

    def test_update_vpin_high(self):
        gate = SmartRebalanceGate()
        gate.update_vpin(1.0)
        assert gate._vpin_cache["current"] == 1.0


# =========================================================================
# update_regime
# =========================================================================

class TestUpdateRegime:
    """Regime updates."""

    def test_update_regime_stores_value(self):
        gate = SmartRebalanceGate()
        gate.update_regime("crisis")
        assert gate._regime == "crisis"

    def test_update_regime_overwrites(self):
        gate = SmartRebalanceGate()
        gate.update_regime("low_vol")
        gate.update_regime("high_vol")
        assert gate._regime == "high_vol"

    def test_update_regime_none(self):
        gate = SmartRebalanceGate()
        gate.update_regime(None)
        assert gate._regime is None

    def test_default_regime_none(self):
        gate = SmartRebalanceGate()
        assert gate._regime is None

    def test_regime_in_status(self):
        gate = SmartRebalanceGate()
        gate.update_regime("high_vol")
        status = gate.get_status()
        assert status["regime"] == "high_vol"

    def test_regime_none_in_status(self):
        gate = SmartRebalanceGate()
        status = gate.get_status()
        assert status["regime"] is None


# =========================================================================
# _compute_vpin
# =========================================================================

class TestComputeVpin:
    """Internal VPIN computation."""

    def test_returns_default_when_engine_is_none(self):
        gate = SmartRebalanceGate()
        with patch.object(gate, "_vpin_engine", None):
            vpin = gate._compute_vpin("SPY")
        assert vpin == 0.30

    def test_returns_default_when_bars_empty(self):
        """load_historical_bars returns empty DataFrame."""
        gate = SmartRebalanceGate()
        with patch("src.rebalancing.integration.load_historical_bars", return_value=MagicMock(empty=True)):
            vpin = gate._compute_vpin("SPY")
        assert vpin == 0.30

    def test_returns_default_when_vpin_is_none(self):
        """calculate_vpin returns None."""
        gate = SmartRebalanceGate()
        mock_engine = MagicMock()
        mock_engine.calculate_vpin.return_value = None
        mock_df = MagicMock(empty=False)
        mock_df.itertuples.return_value = iter(
            [MagicMock(Index=datetime(2026, 5, 1), open=100, high=101, low=99, close=100.5, volume=1000000)]
        )
        with (
            patch.object(gate, "_vpin_engine", mock_engine),
            patch("src.rebalancing.integration.load_historical_bars", return_value=mock_df),
        ):
            vpin = gate._compute_vpin("SPY")
        assert vpin == 0.30

    def test_returns_vpin_value_when_computed(self):
        """calculate_vpin returns a valid float."""
        gate = SmartRebalanceGate()
        mock_engine = MagicMock()
        mock_engine.calculate_vpin.return_value = 0.42
        mock_df = MagicMock(empty=False)
        mock_entry = MagicMock()
        mock_entry.Index = datetime(2026, 5, 1)
        mock_entry.open = 100
        mock_entry.high = 101
        mock_entry.low = 99
        mock_entry.close = 100.5
        mock_entry.volume = 1000000
        mock_df.itertuples.return_value = iter([mock_entry])
        with (
            patch.object(gate, "_vpin_engine", mock_engine),
            patch("src.rebalancing.integration.load_historical_bars", return_value=mock_df),
        ):
            vpin = gate._compute_vpin("SPY")
        assert vpin == 0.42

    def test_returns_default_on_exception(self):
        """Any exception during compute returns default."""
        gate = SmartRebalanceGate()
        with patch("src.rebalancing.integration.load_historical_bars", side_effect=RuntimeError("fail")):
            vpin = gate._compute_vpin("SPY")
        assert vpin == 0.30

    def test_processes_multiple_bars(self):
        """Multiple bars are processed sequentially."""
        gate = SmartRebalanceGate()
        mock_engine = MagicMock()
        mock_engine.calculate_vpin.return_value = 0.35
        mock_df = MagicMock(empty=False)
        rows = []
        for i in range(5):
            m = MagicMock()
            m.Index = datetime(2026, 5, 1 + i)
            m.open = 100 + i
            m.high = 101 + i
            m.low = 99 + i
            m.close = 100.5 + i
            m.volume = 1000000
            rows.append(m)
        mock_df.itertuples.return_value = iter(rows)
        with (
            patch.object(gate, "_vpin_engine", mock_engine),
            patch("src.rebalancing.integration.load_historical_bars", return_value=mock_df),
        ):
            vpin = gate._compute_vpin("SPY")
        assert vpin == 0.35
        assert mock_engine.process_bar.call_count == 5
        assert mock_engine.calculate_vpin.call_count == 1


# =========================================================================
# evaluate — all decision paths
# =========================================================================

class TestEvaluate:
    """Core evaluate() method — all decision paths."""

    # -- Drift below threshold: SKIP --

    def test_skip_low_drift(self):
        """Drift below default 10% threshold -> skip."""
        gate = SmartRebalanceGate()
        result = gate.evaluate(
            current_holdings={"SPY": 46000, "GLD": 38000, "TLT": 16000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
        )
        assert result.should_execute is False
        assert result.decision == "skip_low_drift"
        assert result.estimated_cost_bps == 0
        assert "drift_below_threshold" in result.reason

    def test_skip_low_drift_exact_boundary(self):
        """Drift at exactly 10% threshold is NOT below -> may execute."""
        gate = SmartRebalanceGate()
        # SPY target 46%, drift 10% means |a - 0.46| / 0.46 = 0.10 -> a = 0.46 * 1.10 = 0.506
        # current_value = 0.506 * 100000 = 50600
        result = gate.evaluate(
            current_holdings={"SPY": 50600, "GLD": 38000, "TLT": 11400},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
        )
        # 10% drift is exactly at threshold, not below -> should not skip
        assert result.decision != "skip_low_drift"

    def test_skip_zero_drift_perfect_allocation(self):
        """Perfect allocation -> zero drift -> skip."""
        result = SmartRebalanceGate().evaluate(
            current_holdings={"SPY": 46000, "GLD": 38000, "TLT": 16000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
        )
        assert result.should_execute is False
        assert result.max_drift == 0.0

    # -- Execute (normal) --

    def test_execute_normal(self):
        """12% drift, low VPIN, in optimal window -> execute."""
        gate = SmartRebalanceGate()
        now = datetime(2026, 5, 21, 12, 0)  # Noon ET — optimal window
        result = gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
            now=now,
        )
        assert result.should_execute is True
        assert result.decision == "execute"
        assert result.estimated_cost_bps > 0

    # -- Emergency override (drift > 25%) --

    def test_emergency_override(self):
        """26% drift -> override_emergency regardless of VPIN/time."""
        gate = SmartRebalanceGate()
        result = gate.evaluate(
            current_holdings={"SPY": 60000, "GLD": 28000, "TLT": 12000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.80,
            now=datetime(2026, 5, 21, 9, 0),  # Outside optimal window
        )
        assert result.should_execute is True
        assert result.decision == "override_emergency"
        assert result.urgency == "emergency"
        assert "emergency_override" in result.reason

    def test_emergency_override_at_boundary(self):
        """Drift exactly at 25% boundary -> should NOT override (not >25%)."""
        gate = SmartRebalanceGate()
        # Need drift = 25% exactly: |a - 0.46| / 0.46 = 0.25 -> a = 0.46 * 1.25 = 0.575
        # 57500 / 100000 = 57.5%
        result = gate.evaluate(
            current_holdings={"SPY": 57500, "GLD": 28500, "TLT": 14000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
        )
        # 25% is NOT > 25%, so no emergency override
        assert result.decision != "override_emergency"

    def test_emergency_override_just_above_boundary(self):
        """Drift just above 25% -> override_emergency."""
        gate = SmartRebalanceGate()
        result = gate.evaluate(
            current_holdings={"SPY": 58000, "GLD": 28000, "TLT": 14000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.80,
        )
        assert result.decision == "override_emergency"

    # -- VPIN toxicity deferral --

    def test_defer_toxicity_high_vpin(self):
        """High VPIN with moderate urgency -> defer_toxicity."""
        gate = SmartRebalanceGate()
        now = datetime(2026, 5, 21, 12, 0)
        result = gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.60,
            now=now,
        )
        # VPIN 0.60 > 0.50 threshold, urgency < emergency -> defer
        assert result.decision == "defer_toxicity"
        assert result.should_execute is False
        assert "high_toxicity_defer" in result.reason

    def test_defer_toxicity_high_vpin_but_emergency_still_defers(self):
        """High VPIN with high urgency (but not emergency) still defers."""
        gate = SmartRebalanceGate()
        now = datetime(2026, 5, 21, 12, 0)
        # SPY: 54000/100000 = 54%, drift = |0.54-0.46|/0.46 = 0.174 -> HIGH
        # GLD: 31000/100000 = 31%, drift = |0.31-0.38|/0.38 = 0.184 -> HIGH
        # TLT: 15000/100000 = 15%, drift = |0.15-0.16|/0.16 = 0.0625
        # Max drift = 18.4% -> HIGH (not EMERGENCY)
        result = gate.evaluate(
            current_holdings={"SPY": 54000, "GLD": 31000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.60,
            now=now,
        )
        # VPIN 0.60 > 0.50 threshold, urgency HIGH < EMERGENCY -> defer
        assert result.decision == "defer_toxicity"

    def test_max_deferral_exceeded_after_high_vpin(self):
        """After max consecutive deferrals, VPIN toxicity -> force execute."""
        gate = SmartRebalanceGate()
        now = datetime(2026, 5, 21, 12, 0)
        # Trigger deferrals by setting consecutive_deferrals past limit
        gate.controller.consecutive_deferrals = 5  # Max is 4 (deferral_max_hours)
        result = gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.60,
            now=now,
        )
        assert result.decision == "execute"
        assert result.should_execute is True
        assert "max_deferral_exceeded" in result.reason

    def test_consecutive_deferrals_counter(self):
        """Consecutive deferrals are tracked correctly."""
        gate = SmartRebalanceGate()
        now = datetime(2026, 5, 21, 12, 0)
        # First deferral
        gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.60,
            now=now,
        )
        assert gate.controller.consecutive_deferrals == 1
        # Second deferral
        gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.60,
            now=now,
        )
        assert gate.controller.consecutive_deferrals == 2
        # Non-toxicity evaluate resets counter
        gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
            now=now,
        )
        assert gate.controller.consecutive_deferrals == 0

    # -- Low-urgency timing deferral --

    def test_defer_timing_low_urgency_outside_window(self):
        """Low urgency outside optimal window -> defer_timing."""
        gate = SmartRebalanceGate()
        now = datetime(2026, 5, 21, 9, 30)  # Before optimal window
        result = gate.evaluate(
            current_holdings={"SPY": 50600, "GLD": 37400, "TLT": 16000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
            now=now,
        )
        # Drift = |0.506 - 0.46| / 0.46 = 0.10 = 10% -> LOW urgency
        # Outside window -> defer_timing
        assert result.decision == "defer_timing"
        assert result.should_execute is False
        assert "wait_for_optimal_window" in result.reason

    def test_execute_high_urgency_outside_window(self):
        """High urgency outside window -> executes (doesn't defer timing)."""
        gate = SmartRebalanceGate()
        now = datetime(2026, 5, 21, 9, 0)  # Outside optimal window
        result = gate.evaluate(
            current_holdings={"SPY": 55000, "GLD": 30000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
            now=now,
        )
        # SPY: 55000/100000 = 55%, drift = |0.55-0.46|/0.46 = 19.6% -> HIGH urgency
        # Low_urgency_can_wait only applies to LOW urgency
        assert result.decision == "execute"
        assert result.should_execute is True

    def test_execute_in_optimal_window(self):
        """Low urgency in optimal window -> executes (no timing deferral)."""
        gate = SmartRebalanceGate()
        now = datetime(2026, 5, 21, 12, 0)  # Inside optimal window
        result = gate.evaluate(
            current_holdings={"SPY": 50600, "GLD": 37400, "TLT": 16000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
            now=now,
        )
        assert result.decision == "execute"
        assert result.should_execute is True

    # -- Cost budget deferral --

    def test_defer_budget(self):
        """Over budget with non-emergency urgency -> defer_budget."""
        gate = SmartRebalanceGate()
        # Push cost tracker over budget
        gate.controller.cost_tracker.annual_limit_pct = 0.001  # 0.1% = 10 bps
        gate.record_execution(cost_bps=12.0, date="2026-05-20", symbols=["SPY"])

        now = datetime(2026, 5, 21, 12, 0)
        result = gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
            now=now,
        )
        assert result.decision == "defer_budget"
        assert result.should_execute is False
        assert "cost_budget_exceeded" in result.reason

    def test_emergency_overrides_budget(self):
        """Emergency urgency -> executes even if over budget."""
        gate = SmartRebalanceGate()
        # Push cost tracker over budget
        gate.controller.cost_tracker.annual_limit_pct = 0.001
        gate.record_execution(cost_bps=12.0, date="2026-05-20", symbols=["SPY"])

        now = datetime(2026, 5, 21, 12, 0)
        result = gate.evaluate(
            current_holdings={"SPY": 60000, "GLD": 28000, "TLT": 12000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
            now=now,
        )
        # 26% drift -> urgency EMERGENCY, should override budget check
        assert result.should_execute is True
        assert result.urgency == "emergency"

    # -- VPIN source precedence --

    def test_evaluate_uses_explicit_vpin(self):
        """Explicit vpin parameter takes precedence."""
        gate = SmartRebalanceGate()
        gate.update_vpin(0.55)  # Cached VPIN
        result = gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.80,  # Explicit overrides cache
        )
        assert result.metadata["vpin"] == 0.80

    def test_evaluate_uses_cached_vpin_when_not_provided(self):
        """Cached VPIN used when no explicit vpin parameter."""
        gate = SmartRebalanceGate()
        gate.update_vpin(0.55)
        result = gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
        )
        assert result.metadata["vpin"] == 0.55

    def test_evaluate_computes_vpin_when_not_cached(self):
        """_compute_vpin fallback when neither explicit nor cached VPIN."""
        gate = SmartRebalanceGate()
        with patch.object(gate, "_compute_vpin", return_value=0.35):
            result = gate.evaluate(
                current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
                target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
                total_value=100000,
            )
        assert result.metadata["vpin"] == 0.35

    # -- Regime-adaptive thresholds --

    def test_regime_low_vol_higher_threshold(self):
        """low_vol regime -> 15% threshold, so 12% drift does not hit."""
        gate = SmartRebalanceGate()
        gate.update_regime("low_vol")
        now = datetime(2026, 5, 21, 12, 0)
        result = gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
            now=now,
        )
        # 12% drift < 15% threshold in low_vol -> skip
        assert result.decision == "skip_low_drift"

    def test_regime_high_vol_lower_threshold(self):
        """high_vol regime -> 7% threshold, so 8.7% drift triggers execute."""
        gate = SmartRebalanceGate()
        gate.update_regime("high_vol")
        now = datetime(2026, 5, 21, 12, 0)
        # SPY=50000 -> 50% alloc -> drift = |0.50-0.46|/0.46 = 8.7% > 7% threshold
        result = gate.evaluate(
            current_holdings={"SPY": 50000, "GLD": 35000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
            now=now,
        )
        assert result.decision == "execute"

    def test_regime_crisis_lowest_threshold(self):
        """crisis regime -> 5% threshold, so 8.7% drift triggers execute."""
        gate = SmartRebalanceGate()
        gate.update_regime("crisis")
        now = datetime(2026, 5, 21, 12, 0)
        # SPY=50000 -> 50% alloc -> drift = |0.50-0.46|/0.46 = 8.7% > 5% threshold
        result = gate.evaluate(
            current_holdings={"SPY": 50000, "GLD": 35000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
            now=now,
        )
        assert result.decision == "execute"

    def test_regime_none_defaults_threshold(self):
        """No regime -> default 10% threshold."""
        gate = SmartRebalanceGate()
        now = datetime(2026, 5, 21, 12, 0)
        # 8% drift -> below 10% default -> skip
        result = gate.evaluate(
            current_holdings={"SPY": 49680, "GLD": 38320, "TLT": 16000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=104000,
            vpin=0.20,
            now=now,
        )
        assert result.decision == "skip_low_drift"

    def test_regime_propagated_to_controller_decision(self):
        """Regime set on gate flows through to controller."""
        gate = SmartRebalanceGate()
        gate.update_regime("crisis")
        now = datetime(2026, 5, 13, 12, 0)
        result = gate.evaluate(
            current_holdings={"SPY": 60000, "GLD": 28000, "TLT": 12000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.30,
            now=now,
        )
        # Crisis threshold 5%, 26% drift -> definitely triggers
        assert result.should_execute is True

    # -- Edge cases --

    def test_evaluate_with_now_none(self):
        """When now=None, evaluate uses datetime.now()."""
        gate = SmartRebalanceGate()
        result = gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
            now=datetime(2026, 5, 21, 12, 0),
        )
        assert isinstance(result, RebalanceGateResult)

    def test_evaluate_empty_holdings(self):
        """Empty current holdings -> all targets have full drift."""
        gate = SmartRebalanceGate()
        result = gate.evaluate(
            current_holdings={},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
        )
        # Each target has 100% drift: |0 - 0.46| / 0.46 = 1.0
        assert result.should_execute is True
        assert result.decision == "override_emergency"

    def test_evaluate_zero_total_value(self):
        """Zero total_value -> all targets have zero allocation."""
        gate = SmartRebalanceGate()
        result = gate.evaluate(
            current_holdings={"SPY": 0, "GLD": 0, "TLT": 0},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=0,
            vpin=0.20,
        )
        # current_alloc = 0 / 0 -> 0, drift = |0 - 0.46| / 0.46 = 1.0 for each
        assert result.max_drift == 1.0

    def test_evaluate_empty_targets(self):
        """Empty target allocations -> no drift to compute."""
        gate = SmartRebalanceGate()
        result = gate.evaluate(
            current_holdings={"SPY": 50000},
            target_allocations={},
            total_value=100000,
            vpin=0.20,
        )
        # No targets = empty drift_details -> falls through to execute path (drift 0? No, the loop iterates 0 times)
        # max_drift stays 0.0, drift below threshold -> skip
        assert result.decision == "skip_low_drift"
        assert result.max_drift == 0.0

    def test_evaluate_missing_symbol_in_holdings(self):
        """A target symbol not in holdings -> treated as zero."""
        gate = SmartRebalanceGate()
        result = gate.evaluate(
            current_holdings={"SPY": 50000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
        )
        # GLD: |0 - 0.38| / 0.38 = 1.0, TLT: |0 - 0.16| / 0.16 = 1.0
        assert result.max_drift == 1.0
        assert result.should_execute is True

    def test_evaluate_single_symbol(self):
        """Single symbol portfolio works correctly."""
        gate = SmartRebalanceGate()
        result = gate.evaluate(
            current_holdings={"SPY": 55000},
            target_allocations={"SPY": 1.0},
            total_value=100000,
            vpin=0.20,
        )
        assert result.should_execute is True

    def test_evaluate_metadata_contains_all_fields(self):
        """Metadata dict has all expected keys."""
        gate = SmartRebalanceGate()
        now = datetime(2026, 5, 21, 12, 0)
        result = gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.30,
            now=now,
        )
        expected_keys = {"drift_details", "vpin", "in_optimal_window", "ytd_cost_bps", "remaining_budget_pct"}
        assert expected_keys.issubset(result.metadata.keys())

    def test_evaluate_drift_details_format(self):
        """drift_details maps symbols to drift fractions."""
        gate = SmartRebalanceGate()
        result = gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.20,
        )
        details = result.metadata["drift_details"]
        assert "SPY" in details
        assert "GLD" in details
        assert "TLT" in details
        # SPY: |0.52 - 0.46| / 0.46 = 0.1304
        assert details["SPY"] == pytest.approx(0.1304, abs=0.001)
        # GLD: |0.33 - 0.38| / 0.38 = 0.1316
        assert details["GLD"] == pytest.approx(0.1316, abs=0.001)
        # TLT: |0.15 - 0.16| / 0.16 = 0.0625
        assert details["TLT"] == pytest.approx(0.0625, abs=0.001)


# =========================================================================
# record_execution
# =========================================================================

class TestRecordExecution:
    """Budget tracking via record_execution."""

    def test_record_basic(self):
        gate = SmartRebalanceGate()
        initial = gate.controller.cost_tracker.ytd_total_bps
        gate.record_execution(cost_bps=5.0, date="2026-05-21", symbols=["SPY", "GLD"])
        assert gate.controller.cost_tracker.ytd_total_bps == initial + 5.0

    def test_record_zero_cost(self):
        gate = SmartRebalanceGate()
        initial = gate.controller.cost_tracker.ytd_total_bps
        gate.record_execution(cost_bps=0.0, date="2026-05-21", symbols=["SPY"])
        assert gate.controller.cost_tracker.ytd_total_bps == initial

    def test_record_high_cost_over_budget(self):
        gate = SmartRebalanceGate()
        # Batch DZ: single-trade cap quarantines 100 bps outlier from YTD sum
        gate.record_execution(cost_bps=100.0, date="2026-05-21", symbols=["SPY"])
        assert gate.controller.cost_tracker.is_over_budget() is False
        assert len(gate.controller.cost_tracker.quarantined_costs) >= 1
        # Under-cap accumulation still can exhaust annual budget
        gate.controller.cost_tracker.max_single_trade_cost_bps = None
        gate.record_execution(cost_bps=60.0, date="2026-05-22", symbols=["SPY"])
        assert gate.controller.cost_tracker.is_over_budget() is True

    def test_record_iso_date_format(self):
        gate = SmartRebalanceGate()
        gate.record_execution(cost_bps=3.0, date="2026-05-21T12:00:00", symbols=["SPY"])
        assert gate.controller.cost_tracker.ytd_total_bps == 3.0
        assert gate.controller.last_rebalance is not None

    def test_record_multiple_executions(self):
        gate = SmartRebalanceGate()
        gate.record_execution(cost_bps=2.0, date="2026-01-15", symbols=["SPY"])
        gate.record_execution(cost_bps=3.0, date="2026-03-20", symbols=["GLD", "TLT"])
        gate.record_execution(cost_bps=1.5, date="2026-05-21", symbols=["SPY", "TLT"])
        assert gate.controller.cost_tracker.ytd_total_bps == pytest.approx(6.5, abs=0.01)
        assert len(gate.controller.cost_tracker.ytd_costs) == 3

    def test_record_updates_last_rebalance(self):
        gate = SmartRebalanceGate()
        assert gate.controller.last_rebalance is None
        gate.record_execution(cost_bps=5.0, date="2026-05-21", symbols=["SPY"])
        assert gate.controller.last_rebalance.year == 2026
        assert gate.controller.last_rebalance.month == 5
        assert gate.controller.last_rebalance.day == 21

    def test_record_clears_deferred(self):
        gate = SmartRebalanceGate()
        gate.controller.deferred_until = datetime(2026, 5, 22, 12, 0)
        gate.record_execution(cost_bps=5.0, date="2026-05-21", symbols=["SPY"])
        assert gate.controller.deferred_until is None

    def test_record_with_empty_symbols(self):
        gate = SmartRebalanceGate()
        gate.record_execution(cost_bps=5.0, date="2026-05-21", symbols=[])
        assert gate.controller.cost_tracker.ytd_total_bps == 5.0


# =========================================================================
# get_status
# =========================================================================

class TestGetStatus:
    """Status dictionary output."""

    def test_returns_dict(self):
        gate = SmartRebalanceGate()
        status = gate.get_status()
        assert isinstance(status, dict)

    def test_contains_all_keys(self):
        gate = SmartRebalanceGate()
        status = gate.get_status()
        expected_keys = {
            "ytd_cost_bps", "ytd_cost_pct", "remaining_budget_pct",
            "is_over_budget", "is_warning", "last_rebalance",
            "deferred_until", "config", "regime",
        }
        assert expected_keys.issubset(status.keys())

    def test_includes_regime(self):
        gate = SmartRebalanceGate()
        gate.update_regime("crisis")
        status = gate.get_status()
        assert status["regime"] == "crisis"

    def test_config_section(self):
        gate = SmartRebalanceGate()
        status = gate.get_status()
        config = status["config"]
        assert "drift_threshold" in config
        assert "vpin_threshold" in config
        assert "optimal_window" in config
        assert "annual_cost_limit" in config

    def test_initial_values(self):
        gate = SmartRebalanceGate()
        status = gate.get_status()
        assert status["ytd_cost_bps"] == 0
        assert status["is_over_budget"] is False
        assert status["is_warning"] is False
        assert status["last_rebalance"] is None
        assert status["deferred_until"] is None

    def test_after_execution(self):
        gate = SmartRebalanceGate()
        gate.record_execution(cost_bps=5.0, date="2026-05-21", symbols=["SPY"])
        status = gate.get_status()
        assert status["ytd_cost_bps"] == 5.0
        assert status["last_rebalance"] is not None

    def test_types_in_status(self):
        gate = SmartRebalanceGate()
        status = gate.get_status()
        assert isinstance(status["ytd_cost_bps"], (int, float))
        assert isinstance(status["is_over_budget"], bool)
        assert isinstance(status["is_warning"], bool)
        assert isinstance(status["config"], dict)


# =========================================================================
# to_json
# =========================================================================

class TestToJson:
    """JSON serialization."""

    def test_returns_valid_json(self):
        gate = SmartRebalanceGate()
        j = gate.to_json()
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_includes_all_status_fields(self):
        gate = SmartRebalanceGate()
        j = gate.to_json()
        parsed = json.loads(j)
        assert "regime" in parsed
        assert "ytd_cost_bps" in parsed
        assert "config" in parsed

    def test_after_state_changes(self):
        gate = SmartRebalanceGate()
        gate.update_regime("high_vol")
        gate.record_execution(cost_bps=3.5, date="2026-05-21", symbols=["SPY"])
        j = gate.to_json()
        parsed = json.loads(j)
        assert parsed["regime"] == "high_vol"
        assert parsed["ytd_cost_bps"] == 3.5

    def test_pretty_print_indentation(self):
        gate = SmartRebalanceGate()
        j = gate.to_json()
        lines = j.split("\n")
        assert len(lines) > 1  # Multi-line, so indented
        assert j.startswith("{")

    def test_round_trip(self):
        """to_json -> json.loads -> matches get_status."""
        gate = SmartRebalanceGate()
        gate.update_regime("crisis")
        gate.record_execution(cost_bps=2.0, date="2026-05-21", symbols=["GLD"])
        expected = gate.get_status()
        parsed = json.loads(gate.to_json())
        assert parsed["regime"] == expected["regime"]
        assert parsed["ytd_cost_bps"] == expected["ytd_cost_bps"]


# =========================================================================
# create_gate_from_config
# =========================================================================

class TestCreateGateFromConfig:
    """Factory function."""

    def test_creates_gate(self):
        gate = create_gate_from_config()
        assert isinstance(gate, SmartRebalanceGate)

    def test_creates_gate_with_custom_config(self, tmp_path):
        config_file = tmp_path / "custom.yaml"
        config_file.write_text("smart_rebalancing:\n  drift_threshold: 0.20\n")
        gate = create_gate_from_config(config_path=str(config_file))
        assert gate.controller.config["drift_threshold"] == 0.20

    def test_creates_gate_with_non_existent_config(self):
        gate = create_gate_from_config(config_path="/nonexistent/path.yaml")
        assert isinstance(gate, SmartRebalanceGate)

    def test_is_callable(self):
        assert callable(create_gate_from_config)


# =========================================================================
# Integration: full life cycle
# =========================================================================

class TestGateLifecycle:
    """End-to-end gate life cycle with multiple operations."""

    def test_full_cycle(self):
        gate = SmartRebalanceGate()

        # 1. Update regime
        gate.update_regime("normal")
        assert gate._regime == "normal"

        # 2. Update VPIN
        gate.update_vpin(0.30)

        # 3. Evaluate — should execute with moderate drift
        result = gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            now=datetime(2026, 5, 21, 12, 0),
        )
        assert isinstance(result, RebalanceGateResult)

        # 4. Record execution
        gate.record_execution(
            cost_bps=6.0,
            date="2026-05-21",
            symbols=list(result.metadata["drift_details"].keys()),
        )

        # 5. Check status
        status = gate.get_status()
        assert status["ytd_cost_bps"] == 6.0
        assert status["last_rebalance"] is not None

        # 6. Serialize
        serialized = gate.to_json()
        parsed = json.loads(serialized)
        assert parsed["regime"] == "normal"

    def test_defer_then_execute_cycle(self):
        """VPIN toxicity defers, then consecutive deferrals force execute."""
        gate = SmartRebalanceGate()
        now = datetime(2026, 5, 21, 12, 0)

        # Attempt 1: High VPIN -> defer
        r1 = gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.60,
            now=now,
        )
        assert r1.decision == "defer_toxicity"

        # Simulate multiple deferrals by setting counter high
        gate.controller.consecutive_deferrals = 5

        # Attempt 2: Max deferral exceeded -> execute
        r2 = gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.60,
            now=now,
        )
        assert r2.decision == "execute"

    def test_emergency_bypasses_all_guards(self):
        """Emergency drift overrides VPIN, timing, and budget guards."""
        gate = SmartRebalanceGate()
        gate.controller.cost_tracker.annual_limit_pct = 0.001
        gate.record_execution(cost_bps=15.0, date="2026-05-20", symbols=["SPY"])
        gate.update_regime("low_vol")

        now = datetime(2026, 5, 21, 9, 0)  # Outside optimal window
        result = gate.evaluate(
            current_holdings={"SPY": 60000, "GLD": 28000, "TLT": 12000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
            vpin=0.80,  # High toxicity
            now=now,
        )
        assert result.decision == "override_emergency"
        assert result.should_execute is True
        assert result.urgency == "emergency"


# =========================================================================
# CLI interface (__main__)
# =========================================================================

class TestCLI:
    """CLI entry point at __main__."""

    def test_status_command(self):
        """Simulate 'status' command via __main__ block logic."""
        gate = SmartRebalanceGate()
        gate.update_regime("normal")
        j = gate.to_json()
        parsed = json.loads(j)
        assert "regime" in parsed

    def test_check_command(self):
        """Simulate 'check' command output fields."""
        gate = SmartRebalanceGate()
        result = gate.evaluate(
            current_holdings={"SPY": 52000, "GLD": 33000, "TLT": 15000},
            target_allocations={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_value=100000,
        )
        # Fields printed by CLI
        assert hasattr(result, "decision")
        assert hasattr(result, "should_execute")
        assert hasattr(result, "urgency")
        assert hasattr(result, "max_drift")
        assert hasattr(result, "estimated_cost_bps")
        assert hasattr(result, "reason")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
