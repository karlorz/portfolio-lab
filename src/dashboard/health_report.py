"""Pure helpers for dashboard health.json assembly (extracted from generator)."""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "classify_market_data_freshness",
    "derive_system_status",
    "signal_health_status_contribution",
    "summarize_stale_symbol_count",
    "build_symbol_freshness_entry",
]


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
    """Map signal_health section → system_status contribution (Batch BN).

    When zero sources are healthy among a non-empty tracked set, the overall
    system must not present as fully healthy (SRE max-severity: all-arms
    degraded is at least ``degraded``). Returns None when signal_health is
    absent/empty (do not invent demotion).
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
    signal_health: Mapping[str, Any] | None = None,
) -> str:
    """Merge cron, freshness, scheduler, SLO, and signal_health into system_status.

    Batch BN: fold signal_health (0 healthy of N tracked → at least degraded)
    so dashboard cannot show overall healthy while every tracked sleeve is
    degraded/unhealthy.
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

    sh_contrib = signal_health_status_contribution(signal_health)
    if sh_contrib:
        status = _max_status(status, sh_contrib)

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