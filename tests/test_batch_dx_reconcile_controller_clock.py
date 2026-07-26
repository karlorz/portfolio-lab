"""Batch DX: advance smart-rebalance controller last_rebalance from order events.

Live friction after DW: rebalance_controller_clock_lag_days≈51 (controller
2026-05-21 vs order event 2026-07-11). record_execution was never called on
broker fills, so durable state lagged the event log. Event-sourcing practice:
derive last_rebalance from RebalanceCompleted / fill event time; do not invent
YTD cost rows during clock reconcile (budget SLI stays honest).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.rebalancing.smart_rebalancer import SmartRebalancingController


def test_reconcile_advances_stale_controller_clock(tmp_path: Path) -> None:
    state_path = tmp_path / "smart_rebalance_state.json"
    ctrl = SmartRebalancingController(
        state_path=state_path, data_dir=tmp_path, load_state=False
    )
    # Stale May clock + polluted costs stay untouched for cost side
    ctrl.last_rebalance = datetime(2026, 5, 21, tzinfo=timezone.utc)
    ctrl.cost_tracker.add_cost(100.0, "2026-05-21", ["SPY"])
    before_bps = ctrl.cost_tracker.ytd_total_bps

    result = ctrl.reconcile_last_rebalance_from_event(
        "2026-07-11T00:20:02.531326+00:00",
        source="order_event_timestamp",
        persist=True,
    )
    assert result["reconciled"] is True
    assert result["advanced"] is True
    assert ctrl.last_rebalance is not None
    assert ctrl.last_rebalance.year == 2026
    assert ctrl.last_rebalance.month == 7
    assert ctrl.last_rebalance.day == 11
    # Costs not invented
    assert abs(ctrl.cost_tracker.ytd_total_bps - before_bps) < 1e-9
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert "2026-07-11" in str(raw["last_rebalance"])
    assert raw.get("last_rebalance_clock_source") == "order_event_timestamp"
    assert raw.get("last_rebalance_reconciled") is True


def test_reconcile_no_op_when_controller_already_ahead(tmp_path: Path) -> None:
    ctrl = SmartRebalancingController(
        state_path=tmp_path / "s.json", data_dir=tmp_path, load_state=False
    )
    ctrl.last_rebalance = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
    result = ctrl.reconcile_last_rebalance_from_event(
        "2026-07-11T00:20:02+00:00",
        source="order_event_timestamp",
        persist=False,
    )
    assert result["reconciled"] is False
    assert result["advanced"] is False
    assert ctrl.last_rebalance.day == 15


def test_reconcile_sets_clock_when_missing(tmp_path: Path) -> None:
    ctrl = SmartRebalancingController(
        state_path=tmp_path / "s.json", data_dir=tmp_path, load_state=False
    )
    assert ctrl.last_rebalance is None
    result = ctrl.reconcile_last_rebalance_from_event(
        "2026-07-11T00:20:02+00:00",
        source="order_event_timestamp",
        persist=False,
    )
    assert result["advanced"] is True
    assert ctrl.last_rebalance is not None
    assert ctrl.last_rebalance.month == 7


def test_get_status_discloses_reconcile_meta(tmp_path: Path) -> None:
    ctrl = SmartRebalancingController(
        state_path=tmp_path / "s.json", data_dir=tmp_path, load_state=False
    )
    ctrl.reconcile_last_rebalance_from_event(
        "2026-07-11T00:20:02+00:00",
        source="order_event_timestamp",
        persist=False,
    )
    status = ctrl.get_status()
    assert status["last_rebalance"] is not None
    assert "2026-07-11" in status["last_rebalance"]
    assert status.get("last_rebalance_clock_source") == "order_event_timestamp"
    assert status.get("last_rebalance_reconciled") is True


def test_gate_reconciles_from_rebalance_health(tmp_path: Path, monkeypatch) -> None:
    from src.rebalancing.integration import SmartRebalanceGate

    state_path = tmp_path / "smart_rebalance_state.json"
    # Seed stale controller state
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "smart-rebalance-state/v1",
                "ytd_costs": [
                    {"cost_bps": 10.0, "date": "2026-05-21", "symbols": ["SPY"]}
                ],
                "last_rebalance": "2026-05-21T00:00:00",
                "deferred_until": None,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "rebalance_health.json").write_text(
        json.dumps(
            {
                "next_rebalance": {
                    "last_execution_at": "2026-07-11T00:20:02.531326+00:00",
                    "last_execution_clock": "order_event_timestamp",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.rebalancing.smart_rebalancer.DATA_DIR", tmp_path, raising=False
    )
    monkeypatch.setattr(
        "src.rebalancing.integration.DATA_DIR", tmp_path, raising=False
    )

    gate = SmartRebalanceGate(
        state_path=state_path, data_dir=tmp_path, load_state=True
    )
    status = gate.get_status()
    assert "2026-07-11" in str(status.get("last_rebalance"))
    assert status.get("last_rebalance_reconciled") is True
    assert abs(status.get("ytd_cost_bps") - 10.0) < 1e-9  # costs unchanged
