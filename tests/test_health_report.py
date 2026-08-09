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


# ── Task 4: session-aware freshness ────────────────────────────────────

def test_compute_session_freshness_friday_bar_on_sunday_zero_missed():
    """Friday daily bars on Sunday have 0 missed sessions before Monday close."""
    from datetime import datetime, timezone
    from src.dashboard.health_report import compute_session_freshness

    result = compute_session_freshness(
        last_bar_date="2026-08-07",
        as_of=datetime(2026, 8, 9, 20, 0, tzinfo=timezone.utc),  # Sunday 16:00 ET
    )
    assert result["missed_market_sessions"] == 0
    assert result["last_expected_completed_session"] == "2026-08-07"
    assert result["last_update_session"] == "2026-08-07"
    assert result["status"] == "fresh"


def test_compute_session_freshness_pre_close_does_not_demand_todays_bar():
    """A run before today's close must not demand today's daily bar."""
    from datetime import datetime, timezone
    from src.dashboard.health_report import compute_session_freshness

    # Friday 14:00 ET: Friday session has not completed yet.
    result = compute_session_freshness(
        last_bar_date="2026-08-06",
        as_of=datetime(2026, 8, 7, 18, 0, tzinfo=timezone.utc),  # Fri 14:00 ET
    )
    assert result["last_expected_completed_session"] == "2026-08-06"
    assert result["missed_market_sessions"] == 0
    assert result["status"] == "fresh"


def test_compute_session_freshness_genuine_lag_escalates():
    """Genuinely missed completed sessions escalate in session units."""
    from datetime import datetime, timezone
    from src.dashboard.health_report import compute_session_freshness

    # Last bar Monday 2026-08-03, as-of Friday 2026-08-07 17:00 ET: Tue/Wed/Thu/Fri
    # sessions completed → 4 missed sessions → critical.
    result = compute_session_freshness(
        last_bar_date="2026-08-03",
        as_of=datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc),  # Fri 17:00 ET
    )
    assert result["missed_market_sessions"] == 4
    assert result["status"] == "critical"

    # One missed session is stale (conservative translation), not fresh.
    result2 = compute_session_freshness(
        last_bar_date="2026-08-06",
        as_of=datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc),  # Fri 17:00 ET, bar Thu
    )
    assert result2["missed_market_sessions"] == 1
    assert result2["status"] == "stale"


def test_compute_session_freshness_holiday_uses_official_calendar():
    """Holiday dates are not expected sessions (July 4th 2026 is a Saturday)."""
    from datetime import datetime, timezone
    from src.dashboard.health_report import compute_session_freshness

    # 2026-07-03 (Friday) is the observed July 4th holiday → not a session.
    result = compute_session_freshness(
        last_bar_date="2026-07-02",
        as_of=datetime(2026, 7, 3, 20, 0, tzinfo=timezone.utc),  # Fri 16:00 ET holiday
    )
    assert result["last_expected_completed_session"] == "2026-07-02"
    assert result["missed_market_sessions"] == 0
    assert result["status"] == "fresh"
