"""Batch CR: sticky ensemble freeze clears on SH recovery + kill-refresh re-project."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from src.dashboard.generator import _compact_health_summary


def test_compact_projects_freeze_false_and_stale_when_healthy_gt_zero() -> None:
    report = {
        "system_status": "degraded",
        "signal_health": {
            "status": "degraded",
            "overall_health": "degraded",
            "summary": {
                "healthy": 3,
                "degraded": 5,
                "unhealthy": 1,
                "total_tracked": 9,
                "quality_badge": "3/9 healthy sources",
                "zero_healthy_sources": False,
                "ensemble_weight_freeze_active": False,
                "ensemble_weights_age_days": 46.2,
                "ensemble_weights_file_stale": True,
            },
            "quality_disclosure": {
                "badge": "3/9 healthy sources",
                "ensemble_weight_freeze": {
                    "weight_freeze_active": False,
                    "weight_file_stale": True,
                    "ensemble_weights_age_days": 46.2,
                },
            },
        },
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0},
    }
    compact = _compact_health_summary(report)
    assert compact.get("ensemble_weight_freeze_active") is False
    assert compact.get("ensemble_weights_file_stale") is True
    assert compact.get("ensemble_weights_age_days") == 46.2
    assert compact.get("signal_health_zero_healthy") is False
    assert compact.get("signal_health_healthy") == 3


def test_compact_freeze_true_only_when_zero_healthy() -> None:
    report = {
        "system_status": "degraded",
        "signal_health": {
            "status": "degraded",
            "summary": {
                "healthy": 0,
                "degraded": 8,
                "unhealthy": 1,
                "total_tracked": 9,
                "zero_healthy_sources": True,
                "ensemble_weight_freeze_active": True,
                "ensemble_weights_age_days": 40.0,
            },
        },
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0},
    }
    compact = _compact_health_summary(report)
    assert compact.get("ensemble_weight_freeze_active") is True
    assert compact.get("signal_health_zero_healthy") is True


def test_kill_refresh_clears_sticky_freeze_on_signals(tmp_path: Path) -> None:
    from src.monitor.health_check import refresh_signals_health_kill_fields

    public = tmp_path / "public"
    public.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    sticky = {
        "generated_at": "2026-07-21T22:15:00+00:00",
        "health": {
            "status": "degraded",
            "signal_health_healthy": 3,
            "signal_health_total_tracked": 9,
            "signal_health_quality_badge": "3/9 healthy sources",
            # pre-CQ sticky false freeze
            "ensemble_weight_freeze_active": True,
            "ensemble_weights_age_days": 46.23,
            "signal_health_zero_healthy": True,
        },
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
    }
    (public / "signals.json").write_text(json.dumps(sticky), encoding="utf-8")
    (data / "signals.json").write_text(json.dumps(sticky), encoding="utf-8")

    report = {
        "status": "degraded",
        "timestamp": "2026-07-21T22:40:00+00:00",
        "signal_health": {
            "status": "degraded",
            "overall_health": "degraded",
            "summary": {
                "healthy": 3,
                "degraded": 5,
                "unhealthy": 1,
                "total_tracked": 9,
                "quality_badge": "3/9 healthy sources",
                "zero_healthy_sources": False,
                "ensemble_weight_freeze_active": False,
                "ensemble_weights_age_days": 46.24,
                "ensemble_weights_file_stale": True,
            },
            "quality_disclosure": {
                "badge": "3/9 healthy sources",
                "ensemble_weight_freeze": {
                    "weight_freeze_active": False,
                    "weight_file_stale": True,
                    "ensemble_weights_age_days": 46.24,
                },
            },
        },
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0, "status": "ok"},
    }

    with patch(
        "src.monitor.health_check._disk_kill_and_open_incidents",
        return_value=(
            {"enabled": False, "level": None},
            {"open_count": 0, "status": "ok"},
        ),
    ):
        refresh_signals_health_kill_fields(
            report, public_dir=public, data_dir=data
        )

    pub = json.loads((public / "signals.json").read_text(encoding="utf-8"))
    priv = json.loads((data / "signals.json").read_text(encoding="utf-8"))
    for payload in (pub, priv):
        h = payload["health"]
        assert h.get("ensemble_weight_freeze_active") is False
        assert h.get("ensemble_weights_file_stale") is True
        assert h.get("signal_health_healthy") == 3
        assert h.get("signal_health_zero_healthy") is False
