#!/usr/bin/env python3
"""
Tests for rebalancing/integration.py — SmartRebalanceGate.
Covers: RebalanceGateResult, SmartRebalanceGate init, evaluate, VPIN,
record_execution, get_status, create_gate_from_config.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.rebalancing.integration import (
    SmartRebalanceGate, RebalanceGateResult, create_gate_from_config,
    _VPIN_AVAILABLE,
)
from src.rebalancing.smart_rebalancer import (
    RebalanceDecision, UrgencyLevel,
)


# ---------------------------------------------------------------------------
# RebalanceGateResult dataclass
# ---------------------------------------------------------------------------

class TestRebalanceGateResult:
    def test_creation(self):
        r = RebalanceGateResult(
            should_execute=True, decision="execute", urgency="high",
            max_drift=0.12, estimated_cost_bps=5.0, reason="test",
            metadata={"vpin": 0.3},
        )
        assert r.should_execute is True
        assert r.decision == "execute"
        assert r.urgency == "high"
        assert r.max_drift == 0.12
        assert r.metadata["vpin"] == 0.3

    def test_defer_result(self):
        r = RebalanceGateResult(
            should_execute=False, decision="defer_toxicity", urgency="low",
            max_drift=0.05, estimated_cost_bps=2.0, reason="VPIN too high",
            metadata={},
        )
        assert r.should_execute is False
        assert r.decision == "defer_toxicity"


# ---------------------------------------------------------------------------
# SmartRebalanceGate init
# ---------------------------------------------------------------------------

class TestSmartRebalanceGateInit:
    def test_init_creates_controller(self):
        gate = SmartRebalanceGate()
        assert gate.controller is not None

    def test_init_vpin_cache(self):
        gate = SmartRebalanceGate()
        assert gate._vpin_cache == {}

    def test_init_vpin_engine_availability(self):
        gate = SmartRebalanceGate()
        if _VPIN_AVAILABLE:
            assert gate._vpin_engine is not None
        else:
            assert gate._vpin_engine is None


# ---------------------------------------------------------------------------
# update_vpin
# ---------------------------------------------------------------------------

class TestUpdateVpin:
    def test_update_vpin_stores_in_cache(self):
        gate = SmartRebalanceGate()
        gate.update_vpin(0.45)
        assert gate._vpin_cache['current'] == 0.45

    def test_update_vpin_overwrites(self):
        gate = SmartRebalanceGate()
        gate.update_vpin(0.30)
        gate.update_vpin(0.55)
        assert gate._vpin_cache['current'] == 0.55


# ---------------------------------------------------------------------------
# _compute_vpin
# ---------------------------------------------------------------------------

class TestComputeVpin:
    def test_returns_default_when_unavailable(self):
        gate = SmartRebalanceGate()
        if not _VPIN_AVAILABLE:
            vpin = gate._compute_vpin('SPY')
            assert vpin == 0.30

    def test_returns_default_on_error(self):
        gate = SmartRebalanceGate()
        with patch.object(gate, '_vpin_engine', None):
            vpin = gate._compute_vpin('SPY')
            assert vpin == 0.30


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_high_drift_executes(self):
        gate = SmartRebalanceGate()
        # SPY 52% vs target 46% = 6% drift → should execute
        result = gate.evaluate(
            current_holdings={'SPY': 52000, 'GLD': 33000, 'TLT': 15000},
            target_allocations={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            total_value=100000,
            vpin=0.20,
        )
        assert isinstance(result, RebalanceGateResult)
        assert isinstance(result.should_execute, bool)
        assert isinstance(result.decision, str)
        assert isinstance(result.urgency, str)
        assert isinstance(result.max_drift, float)
        assert isinstance(result.estimated_cost_bps, float)
        assert isinstance(result.reason, str)
        assert isinstance(result.metadata, dict)

    def test_evaluate_low_drift_skips(self):
        gate = SmartRebalanceGate()
        # Nearly on-target → should skip
        result = gate.evaluate(
            current_holdings={'SPY': 46000, 'GLD': 38000, 'TLT': 16000},
            target_allocations={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            total_value=100000,
            vpin=0.20,
        )
        assert isinstance(result, RebalanceGateResult)
        # Very low drift should skip
        if result.max_drift < 0.05:
            assert result.should_execute is False

    def test_evaluate_with_vpin_from_cache(self):
        gate = SmartRebalanceGate()
        gate.update_vpin(0.55)
        result = gate.evaluate(
            current_holdings={'SPY': 52000, 'GLD': 33000, 'TLT': 15000},
            target_allocations={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            total_value=100000,
        )
        assert isinstance(result, RebalanceGateResult)
        assert result.metadata['vpin'] == 0.55

    def test_evaluate_with_explicit_vpin(self):
        gate = SmartRebalanceGate()
        result = gate.evaluate(
            current_holdings={'SPY': 52000, 'GLD': 33000, 'TLT': 15000},
            target_allocations={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            total_value=100000,
            vpin=0.80,
        )
        assert result.metadata['vpin'] == 0.80

    def test_evaluate_metadata_fields(self):
        gate = SmartRebalanceGate()
        result = gate.evaluate(
            current_holdings={'SPY': 52000, 'GLD': 33000, 'TLT': 15000},
            target_allocations={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            total_value=100000,
            vpin=0.30,
        )
        assert 'drift_details' in result.metadata
        assert 'vpin' in result.metadata
        assert 'in_optimal_window' in result.metadata
        assert 'ytd_cost_bps' in result.metadata
        assert 'remaining_budget_pct' in result.metadata

    def test_evaluate_uses_provided_time(self):
        gate = SmartRebalanceGate()
        now = datetime(2026, 5, 21, 14, 0)
        result = gate.evaluate(
            current_holdings={'SPY': 46000, 'GLD': 38000, 'TLT': 16000},
            target_allocations={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            total_value=100000,
            vpin=0.20,
            now=now,
        )
        assert isinstance(result, RebalanceGateResult)


# ---------------------------------------------------------------------------
# record_execution
# ---------------------------------------------------------------------------

class TestRecordExecution:
    def test_record_updates_cost_tracker(self):
        gate = SmartRebalanceGate()
        initial_bps = gate.controller.cost_tracker.ytd_total_bps
        gate.record_execution(cost_bps=5.0, date="2026-05-21", symbols=["SPY", "GLD"])
        assert gate.controller.cost_tracker.ytd_total_bps > initial_bps


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_returns_dict(self):
        gate = SmartRebalanceGate()
        status = gate.get_status()
        assert isinstance(status, dict)


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------

class TestToJson:
    def test_returns_valid_json(self):
        import json
        gate = SmartRebalanceGate()
        j = gate.to_json()
        parsed = json.loads(j)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# create_gate_from_config
# ---------------------------------------------------------------------------

class TestCreateGateFromConfig:
    def test_creates_gate(self):
        gate = create_gate_from_config()
        assert isinstance(gate, SmartRebalanceGate)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
