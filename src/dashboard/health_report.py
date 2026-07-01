"""Pure helpers for dashboard health.json assembly (extracted from generator)."""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "classify_market_data_freshness",
    "derive_system_status",
    "summarize_stale_symbol_count",
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


def derive_system_status(
    *,
    current: str = "healthy",
    backend_error: bool = False,
    scheduler_status: str | None = None,
    slo_status: str | None = None,
    failed_jobs: int = 0,
    stale_count: int = 0,
) -> str:
    """Merge cron, freshness, scheduler, and SLO signals into one system_status."""
    status = current
    if status not in {"warning", "critical", "degraded"}:
        status = "healthy"
    if backend_error:
        return "degraded"
    if (
        scheduler_status in {"degraded", "warning", "unavailable"}
        or slo_status == "warning"
        or failed_jobs > 0
        or stale_count > 5
    ):
        status = "warning"
    if slo_status == "critical" or failed_jobs > 2 or stale_count > 10:
        status = "critical"
    return status


def build_symbol_freshness_entry(
    *,
    last_date: str,
    days_stale: int,
    market_lag_days: int,
    latest_available_market_date: str | None,
) -> dict[str, Any]:
    return {
        "last_update": last_date,
        "days_stale": days_stale,
        "market_lag_days": market_lag_days,
        "latest_available_market_date": latest_available_market_date,
        "status": classify_market_data_freshness(market_lag_days),
    }