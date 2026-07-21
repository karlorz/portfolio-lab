"""Batch BN residual honesty: signal_health 0/N folds into system_status + compact."""

from __future__ import annotations

from src.dashboard.generator import _compact_health_summary
from src.dashboard.health_report import derive_system_status


def test_live_shaped_zero_healthy_nine_tracked_not_system_healthy() -> None:
    """Mirrors WWW health.json cycle 314/BN: healthy=0 degraded=7 unhealthy=2."""
    signal_health = {
        "status": "degraded",
        "overall_health": "degraded",
        "summary": {
            "healthy": 0,
            "degraded": 7,
            "unhealthy": 2,
            "total_tracked": 9,
            "pending_predictions": 5875,
        },
    }
    status = derive_system_status(
        current="healthy",
        scheduler_status="ok",
        slo_status=None,
        failed_jobs=0,
        stale_count=0,
        signal_health=signal_health,
    )
    assert status == "degraded"


def test_compact_health_summary_includes_signal_health_rollup() -> None:
    report = {
        "system_status": "degraded",
        "signal_health": {
            "status": "degraded",
            "overall_health": "degraded",
            "summary": {
                "healthy": 0,
                "degraded": 7,
                "unhealthy": 2,
                "total_tracked": 9,
            },
        },
        "cron_jobs": [],
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0},
    }
    compact = _compact_health_summary(report)
    assert compact["status"] == "degraded"
    assert compact.get("signal_health_healthy") == 0
    assert compact.get("signal_health_total_tracked") == 9
    assert compact.get("signal_health_status") == "degraded"


def test_generator_passes_signal_health_into_derive() -> None:
    src = open("src/dashboard/generator.py", encoding="utf-8").read()
    assert "signal_health=health_data.get(\"signal_health\")" in src
    assert "signal_health_healthy" in src
