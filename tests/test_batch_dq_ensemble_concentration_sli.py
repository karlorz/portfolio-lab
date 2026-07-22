"""Batch DQ: ensemble concentration SLI on health + partial-patch re-project."""

from __future__ import annotations

import json
from pathlib import Path

from src.dashboard.generator import DashboardGenerator
from src.monitor.health_check import refresh_signals_health_kill_fields


def test_rollup_exposes_cap_fields() -> None:
    statuses = [
        {
            "source": "cross_asset_rv",
            "contributing": True,
            "active_weight": 0.85,
            "effective_weight": 0.68,
            "configured_weight": 0.11,
        },
        {
            "source": "google_trends",
            "contributing": True,
            "active_weight": 0.15,
            "effective_weight": 0.12,
            "configured_weight": 0.05,
        },
    ]
    rollup = DashboardGenerator._ensemble_active_weights_rollup(statuses)
    assert rollup["per_signal_active_weight_cap"] == 0.50
    assert rollup["per_signal_active_weight_cap_applied"] is True
    assert max(rollup["active_weights"].values()) <= 0.50 + 1e-6


def test_partial_health_refresh_reprojects_concentration(
    tmp_path: Path, monkeypatch
) -> None:
    """Sticky pre-DP CAR=0.85 must surface ensemble_concentration_ok=false."""
    public = tmp_path / "public"
    private = tmp_path / "data"
    public.mkdir()
    private.mkdir()
    signals = {
        "generated_at": "2026-07-22T04:00:00+00:00",
        "generator_git_sha": "deadbeef",
        "generator_git_sha_status": "full_generate",
        "ensemble_voting": {
            "active_weights": {
                "cross_asset_rv": 0.85042,
                "multi_timeframe_fusion": 0.04476,
                "google_trends": 0.06114,
                "vix_term_structure": 0.04367,
            },
            "n_eff": 1.79,
            "per_signal_active_weight_cap": 0.50,
            "ensemble_concentration_ok": False,
            "max_active_weight": 0.85042,
        },
        "health": {"status": "ok", "signal_health_healthy": 1},
    }
    (public / "signals.json").write_text(json.dumps(signals), encoding="utf-8")
    (private / "signals.json").write_text(json.dumps(signals), encoding="utf-8")
    # kill_switch absent → refresh should still run
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
    assert h.get("ensemble_max_active_weight", 0) >= 0.85 - 1e-6
    assert h.get("ensemble_concentration_ok") is False
    assert h.get("ensemble_concentration_status") == "concentrated"
    assert h.get("status") == "warning"
    assert h.get("ensemble_may_lag_full_generate") is True
    assert out.get("generator_git_sha_status") == "partial_patch"
