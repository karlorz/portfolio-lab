"""Tests for health_report pure helpers."""

from __future__ import annotations

from src.dashboard.health_report import (
    build_symbol_freshness_entry,
    classify_market_data_freshness,
    derive_system_status,
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


def test_build_symbol_freshness_entry() -> None:
    row = build_symbol_freshness_entry(
        last_date="2026-06-28",
        days_stale=2,
        market_lag_days=2,
        latest_available_market_date="2026-06-30",
    )
    assert row["status"] == "stale"
    assert row["market_lag_days"] == 2