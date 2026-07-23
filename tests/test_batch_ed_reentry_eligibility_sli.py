"""Batch ED: multi-horizon reentry eligibility SLI on compact health.

Live friction after EC: soft_floor_dominant vote mass, but nested reentry
shows MSM/INTL/VIXTS multi-horizon eligible while ALT/CARA blocked — no compact
keys. Research: hysteresis reentry criteria; disclose eligible sleepers without
force-wake (policy multi_horizon_hysteresis_no_force_wake).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.dashboard.generator import project_reentry_eligibility_onto_health
from src.monitor.health_check import refresh_signals_health_kill_fields


def _row(source, status, contributing=False, active_weight=0.0, reentry_eligible=None, blocked=None):
    hm = {}
    if reentry_eligible is not None:
        hm["reentry"] = {
            "reentry_eligible": reentry_eligible,
            "reentry_blocked_reason": blocked,
            "policy": "multi_horizon_hysteresis_no_force_wake",
        }
        hm["reentry_eligible"] = reentry_eligible
    return {
        "source": source,
        "status": status,
        "contributing": contributing,
        "active_weight": active_weight,
        "health_metrics": hm or None,
    }


def test_live_shape_eligible_sleepers() -> None:
    health: dict = {"status": "ok"}
    ev = {
        "configured_source_status": [
            _row("multi_speed_momentum", "zero_baseline", reentry_eligible=True),
            _row("cross_asset_rv", "active_soft_floor", True, 0.5),
            _row(
                "alternative_data",
                "health_sleep",
                reentry_eligible=False,
                blocked="negative_ic_horizon(ic_60d,ic_90d)",
            ),
            _row("international_momentum", "inactive_signal", reentry_eligible=True),
            _row(
                "cross_asset_regime_arb",
                "health_sleep",
                reentry_eligible=False,
                blocked="negative_ic_horizon(ic_30d,ic_60d,ic_90d)",
            ),
            _row("vix_term_structure", "health_sleep", reentry_eligible=True),
            _row("google_trends", "active_soft_floor", True, 0.29),
        ],
        "health_gate_slept": {
            "alternative_data": "x",
            "cross_asset_regime_arb": "y",
            "vix_term_structure": "z",
        },
    }
    out = project_reentry_eligibility_onto_health(health, ev)
    assert out["ensemble_reentry_eligible_count"] == 3
    assert set(out["ensemble_reentry_eligible_sources"].split(",")) == {
        "international_momentum",
        "multi_speed_momentum",
        "vix_term_structure",
    }
    assert out["ensemble_reentry_blocked_count"] == 2
    assert out["ensemble_reentry_status"] == "eligible_pending"
    # Soft info only — do not force wake; ok status stays ok unless already warning
    assert out["status"] == "ok"
    assert "no_force_wake" in out["ensemble_reentry_policy"]


def test_none_eligible_ok() -> None:
    health: dict = {"status": "ok"}
    ev = {
        "configured_source_status": [
            _row(
                "alternative_data",
                "health_sleep",
                reentry_eligible=False,
                blocked="negative_ic",
            ),
        ],
        "health_gate_slept": {"alternative_data": "x"},
    }
    out = project_reentry_eligibility_onto_health(health, ev)
    assert out["ensemble_reentry_eligible_count"] == 0
    assert out["ensemble_reentry_status"] == "none_eligible"
    assert out["status"] == "ok"


def test_partial_refresh_reprojects_reentry(
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
        "ensemble_voting": {
            "configured_source_status": [
                _row("vix_term_structure", "health_sleep", reentry_eligible=True),
                _row(
                    "alternative_data",
                    "health_sleep",
                    reentry_eligible=False,
                    blocked="neg",
                ),
            ],
            "health_gate_slept": {
                "vix_term_structure": "z",
                "alternative_data": "x",
            },
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
    assert h.get("ensemble_reentry_eligible_count") == 1
    assert "vix_term_structure" in (h.get("ensemble_reentry_eligible_sources") or "")
    assert h.get("ensemble_reentry_status") == "eligible_pending"
