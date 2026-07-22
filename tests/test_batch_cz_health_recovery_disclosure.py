"""Batch CZ: health_sleep recovery metrics + operator recovery hints."""

from __future__ import annotations

from src.dashboard.generator import DashboardGenerator


def test_recovery_hint_negative_ic_with_decay() -> None:
    hint = DashboardGenerator._health_recovery_hint(
        status="degraded",
        ic=-0.07,
        acc30=0.43,
        acc60=0.63,
        health_score=0.51,
        half_life=14.0,
    )
    assert "decay" in hint.lower() or "negative ic" in hint.lower()
    assert "ic" in hint.lower()


def test_recovery_hint_deep_negative_ic() -> None:
    hint = DashboardGenerator._health_recovery_hint(
        status="degraded",
        ic=-0.3,
        acc30=0.5,
        acc60=0.5,
        health_score=0.5,
        half_life=77.0,
    )
    assert "deeply" in hint.lower() or "negative" in hint.lower()


def test_health_sleep_entry_attaches_metrics() -> None:
    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        source_breakdown=[
            {"source": "alternative_data", "weight": 0.0, "is_active": True},
            {"source": "cross_asset_rv", "weight": 0.5, "is_active": True},
        ],
        health_gate_slept={"alternative_data": "degraded_negative_ic(-0.071)"},
        health_metrics={
            "alternative_data": {
                "status": "degraded",
                "health_score": 0.5148,
                "ic": -0.0709,
                "accuracy_30d": 0.4353,
                "accuracy_60d": 0.634,
                "ic_half_life_days": 13.7,
                "window_collapse_90_60": True,
                "recovery_hint": "Negative IC with recent accuracy decay vs 60d.",
            }
        },
    )
    alt = next(r for r in statuses if r["source"] == "alternative_data")
    assert alt["status"] == "health_sleep"
    assert "health_metrics" in alt
    assert alt["health_metrics"]["ic"] == -0.0709
    assert alt.get("recovery_hint")
    assert "recovery:" in alt["reason"]


def test_active_without_sleep_skips_metric_load_when_empty_maps() -> None:
    # No sleep map and no metrics → no crash; active rows ok
    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        source_breakdown=[{"source": "cross_asset_rv", "weight": 1.0, "is_active": True}],
        health_gate_slept={},
        health_metrics={},
    )
    assert any(r["status"] == "active" for r in statuses)
