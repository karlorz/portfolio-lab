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


# ── F4b: NYSE early-close sessions (13:00 ET close) ──────────────────────

def test_compute_session_freshness_early_close_day_completes_at_13h():
    """On an early-close day the session is complete after 13:00 ET.

    2026-12-24 (Thu) is a trading day with a 13:00 ET close; a run at
    14:30 ET with the previous day's bar must count Dec 24 as a completed,
    missed session (stale), not pretend the session is still open.
    """
    from datetime import datetime, timezone
    from src.dashboard.health_report import compute_session_freshness

    result = compute_session_freshness(
        last_bar_date="2026-12-23",
        as_of=datetime(2026, 12, 24, 19, 30, tzinfo=timezone.utc),  # 14:30 ET
    )
    assert result["last_expected_completed_session"] == "2026-12-24"
    assert result["missed_market_sessions"] == 1
    assert result["status"] == "stale"


def test_compute_session_freshness_early_close_before_13h_not_due():
    """Before the 13:00 ET early close, today's session is not yet complete."""
    from datetime import datetime, timezone
    from src.dashboard.health_report import compute_session_freshness

    result = compute_session_freshness(
        last_bar_date="2026-12-23",
        as_of=datetime(2026, 12, 24, 16, 0, tzinfo=timezone.utc),  # 11:00 ET
    )
    assert result["last_expected_completed_session"] == "2026-12-23"
    assert result["missed_market_sessions"] == 0
    assert result["status"] == "fresh"


def test_early_close_date_rules_and_session_close():
    """F4b rule model: recurring NYSE early-close dates close at 13:00 ET."""
    from datetime import date, time
    from src.dashboard.health_report import (
        is_early_close_date,
        session_close_et,
    )

    assert is_early_close_date(date(2026, 11, 27)) is True  # day after Thanksgiving
    assert is_early_close_date(date(2026, 12, 24)) is True  # Christmas Eve
    assert is_early_close_date(date(2026, 12, 31)) is True  # New Year's Eve
    assert is_early_close_date(date(2025, 7, 3)) is True    # July 3 (trading day)
    assert is_early_close_date(date(2026, 8, 10)) is False  # regular Monday
    assert is_early_close_date(date(2026, 11, 26)) is False  # Thanksgiving (holiday)
    assert session_close_et(date(2026, 12, 24)) == time(13, 0)
    assert session_close_et(date(2026, 8, 10)) == time(16, 0)


def test_early_close_rule_model_full_year_2026_2027():
    """Full-year enumeration lock: exactly the rule-model early-close days
    close at 13:00 ET, every other trading day closes at 16:00 ET, for all
    of 2026 and 2027 against the repo NYSECalendar.

    Expected early-close trading days: 2026-11-27 (Fri), 2026-12-24 (Thu),
    2026-12-31 (Thu), 2027-11-26 (Fri), 2027-12-31 (Fri). Non-issues the
    guard handles: 2026-07-03 (Fri) is the observed July 4 holiday (not a
    trading day); 2027-07-03 is a Saturday (weekend guard); 2027-12-24
    (Fri) is the observed Christmas holiday (Dec 25 2027 is a Saturday),
    so there is no session to early-close. Ad-hoc announced closes are
    still not enumerated (documented model limitation).
    """
    from datetime import date, timedelta, time

    from src.dashboard.health_report import is_early_close_date, session_close_et
    from src.signals.calendar_seasonality import NYSECalendar

    expected_early = {
        date(2026, 11, 27),
        date(2026, 12, 24),
        date(2026, 12, 31),
        date(2027, 11, 26),
        date(2027, 12, 31),
    }
    observed_early: set[date] = set()
    for year in (2026, 2027):
        cal = NYSECalendar(year=year)
        d = date(year, 1, 1)
        while d.year == year:
            if cal.is_trading_day(d):
                close = session_close_et(d)
                assert close in (time(13, 0), time(16, 0)), f"{d}: bad close {close}"
                if close == time(13, 0):
                    observed_early.add(d)
                assert (close == time(13, 0)) == is_early_close_date(d), (
                    f"{d}: session_close_et and is_early_close_date disagree"
                )
            d += timedelta(days=1)
    assert observed_early == expected_early, (
        f"early-close set drifted: {sorted(observed_early)} vs {sorted(expected_early)}"
    )
