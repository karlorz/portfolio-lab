"""Tests for health_report pure helpers."""

from __future__ import annotations

from src.dashboard.health_report import (
    build_symbol_freshness_entry,
    classify_market_data_freshness,
    derive_system_status,
    signal_health_status_contribution,
    summarize_stale_symbol_count,
)


def test_classify_market_data_freshness() -> None:
    assert classify_market_data_freshness(0) == "fresh"
    assert classify_market_data_freshness(1) == "fresh"
    assert classify_market_data_freshness(2) == "stale"
    assert classify_market_data_freshness(3) == "stale"
    assert classify_market_data_freshness(4) == "critical"


def test_summarize_stale_symbol_count() -> None:
    freshness = {
        "SPY": {"status": "fresh"},
        "GLD": {"status": "stale"},
        "TLT": {"status": "critical"},
    }
    assert summarize_stale_symbol_count(freshness) == 2


def test_derive_system_status_escalation() -> None:
    assert derive_system_status() == "healthy"
    assert derive_system_status(backend_error=True) == "degraded"
    assert derive_system_status(failed_jobs=1, stale_count=6) == "warning"
    assert derive_system_status(slo_status="critical", failed_jobs=3) == "critical"


def test_signal_health_zero_healthy_stays_on_quality_plane() -> None:
    """0/N remains quality-degraded without demoting the ops badge."""
    sh = {
        "status": "degraded",
        "overall_health": "degraded",
        "summary": {
            "healthy": 0,
            "degraded": 7,
            "unhealthy": 2,
            "total_tracked": 9,
        },
    }
    assert signal_health_status_contribution(sh) == "degraded"
    assert derive_system_status() == "healthy"
    assert derive_system_status(current="healthy") == "healthy"
    # A real ops-critical SLO still wins independently of signal quality.
    assert (
        derive_system_status(slo_status="critical") == "critical"
    )


def test_signal_health_all_unhealthy_is_quality_critical_only() -> None:
    sh = {
        "summary": {
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 5,
            "total_tracked": 5,
        },
    }
    assert signal_health_status_contribution(sh) == "critical"
    assert derive_system_status() == "healthy"


def test_signal_health_absent_does_not_demote() -> None:
    assert signal_health_status_contribution(None) is None
    assert signal_health_status_contribution({}) is None
    assert derive_system_status() == "healthy"


def test_build_symbol_freshness_entry() -> None:
    row = build_symbol_freshness_entry(
        last_date="2026-06-28",
        days_stale=2,
        market_lag_days=2,
        latest_available_market_date="2026-06-30",
    )
    assert row["status"] == "stale"
    assert row["market_lag_days"] == 2
    assert row["status_basis"] == "market_lag"
    assert row["calendar_age_days"] == 2


def test_fresh_status_with_wall_clock_age_disclosed() -> None:
    """Friday last bar on Monday: market_lag=0 → fresh; calendar age still 3."""
    row = build_symbol_freshness_entry(
        last_date="2026-07-17",
        days_stale=3,
        market_lag_days=0,
        latest_available_market_date="2026-07-17",
    )
    assert row["status"] == "fresh"
    assert row["status_basis"] == "market_lag"
    assert row["days_stale"] == 3
    assert row["calendar_age_days"] == 3
    assert row["calendar_note"] is not None
    assert "calendar_age_days=3" in row["calendar_note"]
    assert "market_lag" in row["calendar_note"]
