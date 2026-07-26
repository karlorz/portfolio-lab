"""Batch CP: scheme-aware health status thresholds under 90/60 window collapse."""

from __future__ import annotations

from src.signals.health_tracker import (
    HEALTH_THRESHOLD_HEALTHY_COLLAPSED,
    HEALTH_THRESHOLD_HEALTHY_FULL,
    classify_health_status,
    status_thresholds_for_scheme,
)


def test_full_scheme_thresholds():
    healthy_min, degraded_min = status_thresholds_for_scheme("full_50_30_20")
    assert healthy_min == HEALTH_THRESHOLD_HEALTHY_FULL == 0.70
    assert degraded_min == 0.50
    assert classify_health_status(0.70, weight_scheme="full_50_30_20") == "healthy"
    assert classify_health_status(0.69, weight_scheme="full_50_30_20") == "degraded"
    assert classify_health_status(0.49, weight_scheme="full_50_30_20") == "unhealthy"


def test_collapsed_scheme_thresholds_reach_healthy():
    """c328: max fleet ~0.58 under collapse; 0.55 band allows healthy."""
    healthy_min, degraded_min = status_thresholds_for_scheme(
        "collapsed_recency_40_60"
    )
    assert healthy_min == HEALTH_THRESHOLD_HEALTHY_COLLAPSED == 0.55
    assert degraded_min == 0.48
    # Live google_trends ~0.5775 / cross_asset_rv ~0.5745
    assert (
        classify_health_status(0.5775, weight_scheme="collapsed_recency_40_60")
        == "healthy"
    )
    assert (
        classify_health_status(0.5538, weight_scheme="collapsed_recency_40_60")
        == "healthy"
    )
    assert (
        classify_health_status(0.5148, weight_scheme="collapsed_recency_40_60")
        == "degraded"
    )
    assert (
        classify_health_status(0.4648, weight_scheme="collapsed_recency_40_60")
        == "unhealthy"
    )


def test_collapsed_still_stricter_than_coin_flip_healthy():
    """Collapsed healthy band is below full 0.7 but above pure 0.5 coin-flip."""
    assert HEALTH_THRESHOLD_HEALTHY_COLLAPSED < HEALTH_THRESHOLD_HEALTHY_FULL
    assert HEALTH_THRESHOLD_HEALTHY_COLLAPSED > 0.50
