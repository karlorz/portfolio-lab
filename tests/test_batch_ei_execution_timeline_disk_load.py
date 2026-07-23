"""Batch EI: load rebalance_health from disk on partial signals health patch.

Live friction (c400): Batch EG projectors only read sticky
``signals.json.rebalance_health``, which full generate rarely embeds.
Health cron partial patches left ``rebalance_execution_timeline_status``
absent while ``data/rebalance_health.json`` already had unique=4 raw=96.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.monitor.health_check import refresh_signals_health_kill_fields


def test_partial_patch_loads_rebalance_health_from_disk(tmp_path, monkeypatch):
    public = tmp_path / "public"
    private = tmp_path / "private"
    public.mkdir()
    private.mkdir()

    # No sticky rebalance_health on signals — matches live partial-patch shape
    signals = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "ensemble_voting": {
            "active_weights": {"cross_asset_rv": 0.5, "google_trends": 0.5},
            "max_active_weight": 0.5,
            "per_signal_active_weight_cap": 0.50,
            "ensemble_concentration_ok": True,
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
    # Disk panel with rewrite inflation (post-EG shape)
    (private / "rebalance_health.json").write_text(
        json.dumps(
            {
                "canonical_execution_days": 4,
                "total_executions": 4,
                "raw_history_entries": 96,
                "snapshot_rewrite_files": 55,
                "execution_timeline_policy": (
                    "canonical_event_day; raw rewrites forensic only"
                ),
                "next_rebalance": {
                    "last_execution_at": "2026-07-11T00:20:02+00:00",
                    "overdue": False,
                },
            }
        ),
        encoding="utf-8",
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
    assert h.get("rebalance_execution_timeline_status") == "rewrite_inflated"
    assert h.get("rebalance_unique_execution_days") == 4
    assert h.get("rebalance_raw_history_entries") == 96
    assert h.get("rebalance_snapshot_rewrite_files") == 55
    assert "unique=4" in (h.get("rebalance_execution_timeline_badge") or "")
    assert h.get("rebalance_health_source") == "disk_or_sticky"


def test_partial_patch_timeline_unknown_when_panel_missing(tmp_path, monkeypatch):
    public = tmp_path / "public"
    private = tmp_path / "private"
    public.mkdir()
    private.mkdir()

    signals = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
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

    refresh_signals_health_kill_fields(
        {"status": "ok", "kill_switch": {"enabled": False}},
        public_dir=public,
        data_dir=private,
    )
    h = json.loads((public / "signals.json").read_text(encoding="utf-8")).get(
        "health"
    ) or {}
    assert h.get("rebalance_execution_timeline_status") == "unknown"
    assert h.get("rebalance_health_source") == "missing"
