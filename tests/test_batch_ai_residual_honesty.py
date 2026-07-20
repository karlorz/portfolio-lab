"""Batch AI residual honesty: convexity stale futures cache not status=ok."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_convexity_stale_futures_cache_is_degraded_not_ok(monkeypatch):
    from src.strategy.convexity_harvest import ConvexityHarvestStrategy, ConvexityPosition

    strategy = ConvexityHarvestStrategy.__new__(ConvexityHarvestStrategy)
    strategy.vix_manager = MagicMock()
    strategy.position_history = []
    strategy.consecutive_backwardation_days = 0
    strategy.last_vix_level = None
    strategy.last_allocation = 0.0
    strategy._last_resolve_meta = {
        "asof": "2026-05-22",
        "vix_source": "futures_cache_last_available",
        "requested_date": "2026-07-20",
    }

    pos = ConvexityPosition(
        date="2026-07-20",
        allocation_pct=3.8,
        position_type="short_vix",
        vix_level=16.76,
        contango_pct=19.5,
        expected_roll_yield=100.0,
        risk_score=0.2,
        exit_triggered=False,
        exit_reason=None,
    )
    monkeypatch.setattr(strategy, "generate_signal", lambda date: pos)

    payload = strategy.get_current_signal()
    assert payload["status"] == "degraded"
    assert payload["runtime_status"] == "stale_futures_cache"
    assert payload["freshness_status"] == "stale"
    assert payload.get("asof") == "2026-05-22"
    assert payload.get("asof_lag_days") == 59  # May 22 → Jul 20
    assert "status_reason" in payload
    # Must not look like a calibrated same-day ok signal
    assert payload["status"] != "ok"


def test_convexity_same_day_cache_is_ok(monkeypatch):
    from src.strategy.convexity_harvest import ConvexityHarvestStrategy, ConvexityPosition

    strategy = ConvexityHarvestStrategy.__new__(ConvexityHarvestStrategy)
    strategy.vix_manager = MagicMock()
    strategy.position_history = []
    strategy.consecutive_backwardation_days = 0
    strategy.last_vix_level = None
    strategy.last_allocation = 0.0
    strategy._last_resolve_meta = {
        "asof": "2026-07-20",
        "vix_source": "futures_cache",
        "requested_date": "2026-07-20",
    }
    pos = ConvexityPosition(
        date="2026-07-20",
        allocation_pct=2.0,
        position_type="short_vix",
        vix_level=17.0,
        contango_pct=8.0,
        expected_roll_yield=50.0,
        risk_score=0.5,
        exit_triggered=False,
        exit_reason=None,
    )
    monkeypatch.setattr(strategy, "generate_signal", lambda date: pos)
    payload = strategy.get_current_signal()
    assert payload["status"] == "ok"
    assert payload.get("freshness_status") == "fresh"
