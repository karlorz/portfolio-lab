"""Batch DW: smart-rebalance cost-budget + dual-clock lag SLI on compact health.

Live friction: smart_rebalance.status shows ytd_cost_bps=214 (4.3× 0.5% annual
limit) and last_rebalance=2026-05-21 while rebalance_health.next_rebalance
last_execution_at=2026-07-11 (order event time). Nested panel only — compact
health had no budget / clock-lag keys, so ops dashboards greenwashed overruns.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.dashboard.generator import project_smart_rebalance_budget_onto_health
from src.monitor.health_check import refresh_signals_health_kill_fields


def test_project_over_budget_and_clock_lag() -> None:
    health: dict = {"status": "ok"}
    smart = {
        "ytd_cost_bps": 214.0,
        "remaining_budget_pct": 0.0,
        "annual_cost_limit_pct": 0.5,
        "status": {
            "ytd_cost_bps": 214.0,
            "ytd_cost_pct": 2.14,
            "is_over_budget": True,
            "is_warning": True,
            "remaining_budget_pct": 0,
            "last_rebalance": "2026-05-21T00:00:00",
            "config": {"annual_cost_limit": "0.5%"},
        },
    }
    rh = {
        "next_rebalance": {
            "last_execution_at": "2026-07-11T00:20:02.531326+00:00",
            "last_execution_clock": "order_event_timestamp",
            "status": "scheduled",
        }
    }
    out = project_smart_rebalance_budget_onto_health(health, smart, rh)
    assert out["rebalance_is_over_budget"] is True
    assert out["rebalance_is_warning"] is True
    assert out["rebalance_ytd_cost_bps"] == 214.0
    assert out["rebalance_annual_cost_limit_pct"] == 0.5
    assert out["rebalance_budget_status"] == "over_budget"
    assert out["rebalance_controller_last_rebalance"] == "2026-05-21T00:00:00"
    assert out["rebalance_last_execution_at"].startswith("2026-07-11")
    assert out["rebalance_last_execution_clock"] == "order_event_timestamp"
    assert out["rebalance_controller_clock_lag_days"] is not None
    assert out["rebalance_controller_clock_lag_days"] >= 50
    assert out["rebalance_controller_clock_lagging"] is True
    assert out["status"] == "warning"


def test_project_ok_when_under_budget_and_clocks_aligned() -> None:
    health: dict = {"status": "ok"}
    smart = {
        "ytd_cost_bps": 12.0,
        "remaining_budget_pct": 0.38,
        "annual_cost_limit_pct": 0.5,
        "status": {
            "ytd_cost_bps": 12.0,
            "is_over_budget": False,
            "is_warning": False,
            "remaining_budget_pct": 0.38,
            "last_rebalance": "2026-07-11T00:20:00+00:00",
        },
    }
    rh = {
        "next_rebalance": {
            "last_execution_at": "2026-07-11T00:20:02+00:00",
            "last_execution_clock": "order_event_timestamp",
        }
    }
    out = project_smart_rebalance_budget_onto_health(health, smart, rh)
    assert out["rebalance_is_over_budget"] is False
    assert out["rebalance_budget_status"] == "ok"
    assert out["rebalance_controller_clock_lagging"] is False
    assert out["status"] == "ok"


def test_partial_health_refresh_reprojects_budget_sli(
    tmp_path: Path, monkeypatch
) -> None:
    public = tmp_path / "public"
    private = tmp_path / "data"
    public.mkdir()
    private.mkdir()
    signals = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "generated_at": "2026-07-22T05:00:00+00:00",
        "generator_git_sha": "deadbeef",
        "generator_git_sha_status": "full_generate",
        "smart_rebalance": {
            "ytd_cost_bps": 214.0,
            "remaining_budget_pct": 0.0,
            "annual_cost_limit_pct": 0.5,
            "status": {
                "ytd_cost_bps": 214.0,
                "is_over_budget": True,
                "is_warning": True,
                "last_rebalance": "2026-05-21T00:00:00",
            },
        },
        "rebalance_health": {
            "next_rebalance": {
                "last_execution_at": "2026-07-11T00:20:02.531326+00:00",
                "last_execution_clock": "order_event_timestamp",
            }
        },
        "health": {"status": "ok"},
    }
    (public / "signals.json").write_text(json.dumps(signals), encoding="utf-8")
    (private / "signals.json").write_text(json.dumps(signals), encoding="utf-8")
    (private / "kill_switch.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8"
    )
    (private / "incidents.json").write_text(
        json.dumps({"open_count": 0, "status": "ok"}), encoding="utf-8"
    )

    monkeypatch.setattr(
        "src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False
    )
    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", private, raising=False)

    report = {
        "status": "ok",
        "system_status": "ok",
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0, "status": "ok"},
    }
    refresh_signals_health_kill_fields(
        report, public_dir=public, data_dir=private
    )

    out = json.loads((public / "signals.json").read_text(encoding="utf-8"))
    h = out.get("health") or {}
    assert h.get("rebalance_is_over_budget") is True
    assert h.get("rebalance_budget_status") == "over_budget"
    assert h.get("rebalance_ytd_cost_bps") == 214.0
    assert h.get("rebalance_controller_clock_lagging") is True
    assert h.get("status") == "warning"
