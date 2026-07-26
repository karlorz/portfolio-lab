"""Batch BU: honest multi-window health when 90d collapses onto 60d."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.signals.health_tracker import HealthScore, SignalHealthTracker


def test_resolve_weights_full_when_independent_counts():
    w = SignalHealthTracker.resolve_health_window_weights(
        {"90d": 100, "60d": 70, "30d": 40},
        {"90d": 0.8, "60d": 0.75, "30d": 0.7},
    )
    assert w["window_collapse_90_60"] is False
    assert w["weight_scheme"] == "full_50_30_20"
    assert w["weight_90d"] == 0.5
    assert w["weight_60d"] == 0.3
    assert w["weight_30d"] == 0.2


def test_resolve_weights_collapsed_when_same_count():
    w = SignalHealthTracker.resolve_health_window_weights(
        {"90d": 50, "60d": 50, "30d": 20},
        {"90d": 0.75, "60d": 0.75, "30d": 0.37},
    )
    assert w["window_collapse_90_60"] is True
    assert w["weight_scheme"] == "collapsed_recency_40_60"
    assert w["weight_90d"] == 0.0
    assert w["weight_60d"] == 0.4
    assert w["weight_30d"] == 0.6


def test_resolve_weights_not_collapsed_when_counts_differ_same_accuracy():
    """Extra 90d rows keep full weights even if hit-rate matches 60d by chance."""
    w = SignalHealthTracker.resolve_health_window_weights(
        {"90d": 100, "60d": 99, "30d": 40},
        {"90d": 0.6, "60d": 0.6, "30d": 0.4},
    )
    assert w["window_collapse_90_60"] is False
    assert w["weight_scheme"] == "full_50_30_20"


def test_collapsed_score_surfaces_recent_decay(tmp_path):
    """With only ~60d history, old formula masked a30=0.37 under a90=a60=0.75.

    Full weights: 0.8*0.75 + 0.2*0.37 = 0.674 (near healthy band).
    Collapsed recency: 0.4*0.75 + 0.6*0.37 = 0.522 (honest degraded).
    """
    db = tmp_path / "health.db"
    tracker = SignalHealthTracker(db_path=db)
    today = datetime.now()

    # 60 calendar days of perfect mid-window accuracy (bull signal + bull actual)
    for i in range(60):
        day = today - timedelta(days=i)
        # Recent 30d: half wrong (bear signal while market bull) → low a30
        if i < 30:
            # alternate wrong predictions
            val = -0.5 if (i % 2 == 0) else 0.5
        else:
            val = 0.5
        ts = day.strftime("%Y-%m-%dT10:00:00")
        tracker.log_prediction_simple(
            source="multi_speed_momentum",
            signal_value=val,
            confidence=0.8,
            timestamp=ts,
        )
        tracker.update_actual_directions({"SPY": 0.01}, day.strftime("%Y-%m-%d"))

    result = tracker.calculate_health_score("multi_speed_momentum")
    assert result is not None
    assert result.window_collapse_90_60 is True
    assert result.weight_scheme == "collapsed_recency_40_60"
    # Recency-weighted score must be below the naive 50/30/20 double-count score
    naive = (
        result.accuracy_90d * 0.5
        + result.accuracy_60d * 0.3
        + result.accuracy_30d * 0.2
    )
    assert result.health_score < naive - 0.05
    assert result.health_score == pytest.approx(
        result.accuracy_60d * 0.4 + result.accuracy_30d * 0.6,
        abs=0.02,
    )


def test_full_history_keeps_classic_weights(tmp_path):
    """90d history with extra labeled rows beyond 60d keeps 50/30/20."""
    db = tmp_path / "health.db"
    tracker = SignalHealthTracker(db_path=db)
    today = datetime.now()
    for i in range(100):
        day = today - timedelta(days=i)
        ts = day.strftime("%Y-%m-%dT10:00:00")
        tracker.log_prediction_simple(
            source="cta",
            signal_value=0.5,
            confidence=0.8,
            timestamp=ts,
        )
        tracker.update_actual_directions({"SPY": 0.01}, day.strftime("%Y-%m-%d"))

    result = tracker.calculate_health_score("cta")
    assert result is not None
    assert result.window_collapse_90_60 is False
    assert result.weight_scheme == "full_50_30_20"
    assert result.health_score == pytest.approx(1.0, abs=0.01)


def test_health_report_includes_collapse_policy(tmp_path):
    db = tmp_path / "health.db"
    tracker = SignalHealthTracker(db_path=db)
    report = tracker.get_health_report()
    assert "health_score_policy" in report
    assert report["health_score_policy"]["live_authoritative"] is False
    assert "window_collapse_90_60_count" in report["summary"]


def test_health_score_to_dict_includes_collapse_fields():
    hs = HealthScore(
        source="x",
        timestamp="2026-07-21",
        health_score=0.55,
        accuracy_30d=0.4,
        accuracy_60d=0.7,
        accuracy_90d=0.7,
        decay_rate=-0.01,
        predictions_count=50,
        status="degraded",
        window_collapse_90_60=True,
        weight_scheme="collapsed_recency_40_60",
    )
    d = hs.to_dict()
    assert d["window_collapse_90_60"] is True
    assert d["weight_scheme"] == "collapsed_recency_40_60"
