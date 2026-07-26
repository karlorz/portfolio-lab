"""Batch EC: ensemble voting-mass quality SLI on compact health.

Live friction: signal_health badge 1/9 healthy (MSM) but MSM is zero_baseline
non-voting — 100% of active_weights sit on soft_floor (CAR/MTF/GT). Compact
health only exposed source counts, not voting-mass quality. Research: soft
floor is graduated degradation; portfolio SLI = % of ensemble weight on soft
floor vs healthy vote mass.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.dashboard.generator import project_voting_mass_quality_onto_health
from src.monitor.health_check import refresh_signals_health_kill_fields


def test_soft_floor_dominant_live_shape() -> None:
    health: dict = {"status": "ok"}
    ev = {
        "active_weights": {
            "cross_asset_rv": 0.5,
            "multi_timeframe_fusion": 0.21,
            "google_trends": 0.29,
        },
        "health_gate_soft_floor": {
            "cross_asset_rv": "degraded_soft_floor",
            "multi_timeframe_fusion": "unhealthy_soft_floor",
            "google_trends": "degraded_soft_floor",
        },
        "health_gate_slept": {
            "alternative_data": "degraded_negative_ic",
            "vix_term_structure": "unhealthy_weak_ic",
        },
        "configured_source_status": [
            {
                "source": "multi_speed_momentum",
                "status": "zero_baseline",
                "contributing": False,
                "active_weight": 0.0,
                "health_metrics": {"status": "healthy", "ic": 0.026},
            },
            {
                "source": "cross_asset_rv",
                "status": "active_soft_floor",
                "contributing": True,
                "active_weight": 0.5,
                "health_metrics": {"status": "degraded", "ic": 0.147},
            },
            {
                "source": "multi_timeframe_fusion",
                "status": "active_soft_floor",
                "contributing": True,
                "active_weight": 0.21,
                "health_metrics": {"status": "unhealthy", "ic": 0.122},
            },
            {
                "source": "google_trends",
                "status": "active_soft_floor",
                "contributing": True,
                "active_weight": 0.29,
                "health_metrics": {"status": "degraded", "ic": 0.288},
            },
        ],
        "contributing_source_count": 3,
        "max_active_weight": 0.5,
        "n_eff": 2.81,
    }
    out = project_voting_mass_quality_onto_health(health, ev)
    assert out["ensemble_voting_soft_floor_mass"] == 1.0
    assert out["ensemble_voting_soft_floor_count"] == 3
    assert out["ensemble_voting_healthy_mass"] == 0.0
    assert out["ensemble_voting_quality_status"] == "soft_floor_dominant"
    assert out["ensemble_voting_healthy_contributors"] == 0
    assert out["status"] == "warning"


def test_healthy_vote_mass_ok() -> None:
    health: dict = {"status": "ok"}
    ev = {
        "active_weights": {"alpha": 0.6, "beta": 0.4},
        "health_gate_soft_floor": {},
        "configured_source_status": [
            {
                "source": "alpha",
                "status": "active",
                "contributing": True,
                "active_weight": 0.6,
                "health_metrics": {"status": "healthy"},
            },
            {
                "source": "beta",
                "status": "active",
                "contributing": True,
                "active_weight": 0.4,
                "health_metrics": {"status": "healthy"},
            },
        ],
        "contributing_source_count": 2,
    }
    out = project_voting_mass_quality_onto_health(health, ev)
    assert out["ensemble_voting_soft_floor_mass"] == 0.0
    assert abs(out["ensemble_voting_healthy_mass"] - 1.0) < 1e-9
    assert out["ensemble_voting_quality_status"] == "ok"
    assert out["status"] == "ok"


def test_partial_refresh_reprojects_voting_quality(
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
            "active_weights": {"cross_asset_rv": 0.7, "google_trends": 0.3},
            "health_gate_soft_floor": {
                "cross_asset_rv": "degraded_soft_floor",
                "google_trends": "degraded_soft_floor",
            },
            "configured_source_status": [
                {
                    "source": "cross_asset_rv",
                    "status": "active_soft_floor",
                    "contributing": True,
                    "active_weight": 0.7,
                    "health_metrics": {"status": "degraded"},
                },
                {
                    "source": "google_trends",
                    "status": "active_soft_floor",
                    "contributing": True,
                    "active_weight": 0.3,
                    "health_metrics": {"status": "degraded"},
                },
            ],
            "contributing_source_count": 2,
            "max_active_weight": 0.7,
            "n_eff": 1.7,
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
    assert h.get("ensemble_voting_soft_floor_mass") == 1.0
    assert h.get("ensemble_voting_quality_status") == "soft_floor_dominant"
    assert h.get("status") == "warning"
