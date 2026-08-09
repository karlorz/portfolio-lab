"""Pure helpers for dashboard health.json assembly (extracted from generator)."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Mapping

__all__ = [
    "classify_market_data_freshness",
    "compute_session_freshness",
    "derive_system_status",
    "signal_health_status_contribution",
    "summarize_stale_symbol_count",
    "build_symbol_freshness_entry",
]

# NYSE regular-session close in America/New_York. Early-close days remain a
# trading session in the repo calendar (the official calendar does not model
# special hours), so they count as one completed session.
SESSION_CLOSE_ET = time(16, 0)
ET_TZ_NAME = "America/New_York"


class _YearAwareNYSE:
    """Thin year-safe wrapper over the repo NYSECalendar.

    NYSECalendar computes holidays for one year; this wrapper resolves the
    calendar per queried date so windows crossing a year boundary (e.g. late
    December → early January) stay correct without a second holiday list.
    """

    def is_trading_day(self, d: date) -> bool:
        from src.signals.calendar_seasonality import NYSECalendar

        return NYSECalendar(year=d.year).is_trading_day(d)

    def previous_trading_day(self, d: date) -> date:
        from src.signals.calendar_seasonality import NYSECalendar

        candidate = d - timedelta(days=1)
        while not NYSECalendar(year=candidate.year).is_trading_day(candidate):
            candidate -= timedelta(days=1)
        return candidate

    def trading_days_between(self, start: date, end: date) -> list[date]:
        from src.signals.calendar_seasonality import NYSECalendar

        days: list[date] = []
        current = start
        while current <= end:
            if NYSECalendar(year=current.year).is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days


class _SevenDayCalendar:
    """Seven-day calendar policy for crypto: every day is a session."""

    def is_trading_day(self, d: date) -> bool:
        return True

    def previous_trading_day(self, d: date) -> date:
        return d - timedelta(days=1)

    def trading_days_between(self, start: date, end: date) -> list[date]:
        days: list[date] = []
        current = start
        while current <= end:
            days.append(current)
            current += timedelta(days=1)
        return days


def _as_et(value: datetime) -> datetime:
    """Normalize an injected as-of time to America/New_York (naive → UTC)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        return value.astimezone(ZoneInfo(ET_TZ_NAME))
    except Exception:  # pragma: no cover - zoneinfo unavailable fallback
        return value.astimezone(timezone.utc)


def compute_session_freshness(
    *,
    last_bar_date: str | date | None,
    as_of: datetime,
    calendar: Any | None = None,
    crypto: bool = False,
) -> dict[str, Any]:
    """Session-aware freshness for one symbol (Task 4).

    ``missed_market_sessions`` counts expected completed sessions on the
    symbol's calendar that have no bar: 0 on weekends/holidays before the next
    close, 0 for a pre-close run that must not demand today's daily bar, and
    ≥1 for genuinely missed completed sessions. Calendar age is disclosure
    only; the status basis is missed sessions.

    Threshold translation (conservative, documented): the legacy day-lag
    thresholds (fresh ≤1d, stale ≤3d, critical >3d) map to session units as
    fresh = 0 missed sessions, stale = 1 missed session, critical = ≥2 missed
    sessions. This can only flag symbols that genuinely missed a completed
    session — never calendar age alone.
    """
    if last_bar_date is None:
        return {
            "last_expected_completed_session": None,
            "last_update_session": None,
            "missed_market_sessions": None,
            "calendar_age_days": None,
            "status": "unknown",
            "status_basis": "missed_sessions",
        }
    if isinstance(last_bar_date, str):
        try:
            last_bar = date.fromisoformat(last_bar_date)
        except ValueError:
            return {
                "last_expected_completed_session": None,
                "last_update_session": None,
                "missed_market_sessions": None,
                "calendar_age_days": None,
                "status": "unknown",
                "status_basis": "missed_sessions",
            }
    else:
        last_bar = last_bar_date

    cal = calendar or (_SevenDayCalendar() if crypto else _YearAwareNYSE())
    as_of_et = _as_et(as_of)
    as_of_date = as_of_et.date()

    if crypto:
        # Crypto sessions complete at day end; today's session is not yet done.
        last_expected_completed = as_of_date - timedelta(days=1)
    elif cal.is_trading_day(as_of_date) and as_of_et.time() >= SESSION_CLOSE_ET:
        last_expected_completed = as_of_date
    else:
        last_expected_completed = cal.previous_trading_day(as_of_date)

    last_update_session = (
        last_bar if cal.is_trading_day(last_bar) else cal.previous_trading_day(last_bar)
    )
    if last_update_session >= last_expected_completed:
        missed = 0
    else:
        missed = len(
            cal.trading_days_between(last_update_session + timedelta(days=1), last_expected_completed)
        )
    if missed == 0:
        status = "fresh"
    elif missed == 1:
        status = "stale"
    else:
        status = "critical"
    return {
        "last_expected_completed_session": last_expected_completed.isoformat(),
        "last_update_session": last_update_session.isoformat(),
        "missed_market_sessions": missed,
        "calendar_age_days": max(0, (as_of_date - last_bar).days),
        "status": status,
        "status_basis": "missed_sessions",
    }


def classify_market_data_freshness(market_lag_days: int) -> str:
    """Classify a symbol by lag versus the provider's latest available date."""
    if market_lag_days <= 1:
        return "fresh"
    if market_lag_days <= 3:
        return "stale"
    return "critical"


def summarize_stale_symbol_count(data_freshness: Mapping[str, Any]) -> int:
    return sum(
        1
        for item in data_freshness.values()
        if isinstance(item, dict) and item.get("status") != "fresh"
    )


def _severity_rank(status: str) -> int:
    return {
        "healthy": 0,
        "ok": 0,
        "good": 0,
        "warning": 1,
        "warn": 1,
        "degraded": 2,
        "unhealthy": 3,
        "critical": 4,
        "error": 4,
    }.get(str(status or "").lower(), 0)


def _max_status(a: str, b: str) -> str:
    """Return the worse of two status labels (max-severity rollup)."""
    return a if _severity_rank(a) >= _severity_rank(b) else b


def signal_health_status_contribution(
    signal_health: Mapping[str, Any] | None,
) -> str | None:
    """Map signal_health section to a quality-plane severity contribution.

    This helper remains the SSOT for quality-sensitive consumers such as the
    graduation circuit breaker and ensemble freeze policy. It must not be
    folded into the operator-facing ``system_status`` ops badge. Returns None
    when signal_health is absent/empty (do not invent demotion).
    """
    if not isinstance(signal_health, Mapping):
        return None
    summary = signal_health.get("summary")
    if not isinstance(summary, Mapping):
        # Fall back to top-level overall_health / status only
        overall = str(
            signal_health.get("overall_health")
            or signal_health.get("status")
            or ""
        ).lower()
        if overall in {"degraded", "unhealthy", "critical", "warning"}:
            return "degraded" if overall in {"degraded", "unhealthy"} else overall
        return None

    try:
        healthy = int(summary.get("healthy") or 0)
        total = int(
            summary.get("total_tracked")
            or summary.get("total")
            or 0
        )
        unhealthy = int(summary.get("unhealthy") or 0)
        degraded = int(summary.get("degraded") or 0)
    except (TypeError, ValueError):
        return None

    if total <= 0:
        return None

    # 0 healthy of N tracked → cannot claim green system path
    if healthy == 0:
        if unhealthy >= total:
            return "critical"
        if unhealthy > 0 or degraded > 0:
            return "degraded"
        return "degraded"

    # Majority unhealthy → warning even if a few healthy remain
    if unhealthy > 0 and unhealthy >= max(1, total // 2):
        return "warning"

    overall = str(
        signal_health.get("overall_health") or signal_health.get("status") or ""
    ).lower()
    if overall == "degraded" and healthy < total:
        return "warning"
    return None


def derive_system_status(
    *,
    current: str = "healthy",
    backend_error: bool = False,
    scheduler_status: str | None = None,
    slo_status: str | None = None,
    failed_jobs: int = 0,
    stale_count: int = 0,
) -> str:
    """Merge operational dimensions into the ``system_status`` ops badge.

    Signal predictive health is intentionally excluded. Thin or unhealthy
    sleeves remain disclosed by the ``signal_health`` block and
    ``signal_quality`` alerts; they do not make serving/data plumbing look
    warning or critical when the ops plane is green.
    """
    status = current
    if status not in {"warning", "critical", "degraded"}:
        status = "healthy"
    if backend_error:
        # Backend error is hard degraded; still allow critical from other dims
        status = "degraded"
    if (
        scheduler_status in {"degraded", "warning", "unavailable"}
        or slo_status == "warning"
        or failed_jobs > 0
        or stale_count > 5
    ):
        status = _max_status(status, "warning")
    if slo_status == "critical" or failed_jobs > 2 or stale_count > 10:
        status = _max_status(status, "critical")

    return status


def build_symbol_freshness_entry(
    *,
    last_date: str,
    days_stale: int,
    market_lag_days: int,
    latest_available_market_date: str | None,
) -> dict[str, Any]:
    """Build per-symbol freshness with explicit dual-clock disclosure.

    ``status`` is **market-relative** (lag vs provider latest bar), not wall-clock.
    ``days_stale`` / ``calendar_age_days`` is wall-clock age of last_update.
    """
    market_status = classify_market_data_freshness(market_lag_days)
    return {
        "last_update": last_date,
        "days_stale": days_stale,
        "calendar_age_days": days_stale,
        "market_lag_days": market_lag_days,
        "latest_available_market_date": latest_available_market_date,
        "status": market_status,
        "status_basis": "market_lag",
        "calendar_note": (
            f"calendar_age_days={days_stale} (wall-clock); "
            f"status={market_status} uses market_lag_days only"
            if days_stale > market_lag_days
            else None
        ),
    }
