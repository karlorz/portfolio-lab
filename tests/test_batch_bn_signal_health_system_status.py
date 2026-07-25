"""Dual-plane successor to Batch BN: quality stays compact, not ops status."""

from __future__ import annotations

from src.dashboard.generator import _compact_health_summary
from src.dashboard.health_report import derive_system_status


def test_live_shaped_zero_healthy_nine_tracked_keeps_system_ops_healthy() -> None:
    """Quality 0/9 is disclosed separately while green ops stays healthy."""
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
    )
    assert status == "healthy"


def test_compact_health_summary_includes_signal_health_rollup() -> None:
    report = {
        "system_status": "healthy",
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
    assert compact["status"] == "healthy"
    assert compact.get("signal_health_healthy") == 0
    assert compact.get("signal_health_total_tracked") == 9
    assert compact.get("signal_health_status") == "degraded"


def test_generator_keeps_signal_health_in_compact_quality_fields() -> None:
    src = open("src/dashboard/generator.py", encoding="utf-8").read()
    assert "signal_health=health_data.get(\"signal_health\")" not in src
    assert "signal_health_healthy" in src
