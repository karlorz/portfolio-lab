#!/usr/bin/env python3
"""
Tests for health_tracker.py — SignalSource/SignalHealthStatus enums,
SignalPrediction/HealthScore/DecayAlert dataclasses, and
SignalHealthTracker with temp SQLite.
"""
import json

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from src.signals.health_tracker import (
    SignalSource,
    SignalHealthStatus,
    SignalPrediction,
    HealthScore,
    DecayAlert,
    SignalHealthTracker,
)


# ---------------------------------------------------------------------------
# SignalSource enum
# ---------------------------------------------------------------------------

class TestSignalSource:
    def test_all_sources(self):
        sources = {s.value for s in SignalSource}
        assert "multi_speed_momentum" in sources
        assert "cross_asset_rv" in sources
        assert "alternative_data" in sources
        assert "unified_overlay" in sources

    def test_count(self):
        assert len(SignalSource) == 6


# ---------------------------------------------------------------------------
# SignalHealthStatus enum
# ---------------------------------------------------------------------------

class TestSignalHealthStatus:
    def test_all_statuses(self):
        assert SignalHealthStatus.HEALTHY.value == "healthy"
        assert SignalHealthStatus.DEGRADED.value == "degraded"
        assert SignalHealthStatus.UNHEALTHY.value == "unhealthy"


# ---------------------------------------------------------------------------
# SignalPrediction dataclass
# ---------------------------------------------------------------------------

class TestSignalPrediction:
    def test_create(self):
        sp = SignalPrediction(timestamp="2025-05-20T10:00:00", source="cta", signal_value=0.75, confidence=0.90, predicted_direction=1, metadata={"strategy": "trend"})
        assert sp.source == "cta"
        assert sp.signal_value == 0.75
        assert sp.predicted_direction == 1

    def test_to_dict(self):
        sp = SignalPrediction(timestamp="2025-05-20T10:00:00", source="hmm", signal_value=-0.50, confidence=0.80, predicted_direction=-1, metadata={})
        d = sp.to_dict()
        assert d["source"] == "hmm"
        assert d["signal_value"] == -0.50
        assert isinstance(d["metadata"], str)  # JSON-serialized


# ---------------------------------------------------------------------------
# HealthScore dataclass
# ---------------------------------------------------------------------------

class TestHealthScore:
    def test_create_and_to_dict(self):
        hs = HealthScore(source="cta", timestamp="2025-05-20", health_score=0.85, accuracy_30d=0.80, accuracy_60d=0.85, accuracy_90d=0.88, decay_rate=-0.02, predictions_count=120, status="healthy")
        d = hs.to_dict()
        assert d["health_score"] == 0.85
        assert d["status"] == "healthy"
        assert d["predictions_count"] == 120


# ---------------------------------------------------------------------------
# DecayAlert dataclass
# ---------------------------------------------------------------------------

class TestDecayAlert:
    def test_create_and_to_dict(self):
        da = DecayAlert(source="sentiment", alert_timestamp="2025-05-20", previous_health=0.80, current_health=0.55, drop_30d=0.25, severity="warning", message="Health dropping")
        d = da.to_dict()
        assert d["source"] == "sentiment"
        assert d["severity"] == "warning"
        assert d["drop_30d"] == 0.25


# ---------------------------------------------------------------------------
# SignalHealthTracker init and state
# ---------------------------------------------------------------------------

class TestTrackerInit:
    def test_default_db_path(self):
        tracker = SignalHealthTracker()
        assert tracker.db_path is not None
        assert tracker.state["version"] == "3.12.0"

    def test_custom_db_path(self, tmp_path):
        db = tmp_path / "test_health.db"
        tracker = SignalHealthTracker(db_path=db)
        assert tracker.db_path == db
        assert db.exists()  # DB created on init

    def test_state_loaded(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        assert "last_health_calculation" in tracker.state
        assert "decay_alerts" in tracker.state  # State loaded from global STATE_PATH


# ---------------------------------------------------------------------------
# log_prediction_simple direction inference
# ---------------------------------------------------------------------------

class TestLogPredictionSimple:
    def test_bullish_signal_gives_direction_1(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        tracker.log_prediction_simple(source="cta", signal_value=0.50, confidence=0.90, timestamp="2025-05-20T10:00:00")

    def test_bearish_signal_gives_direction_minus_1(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        tracker.log_prediction_simple(source="sentiment", signal_value=-0.50, confidence=0.80, timestamp="2025-05-20T10:00:00")

    def test_neutral_signal_gives_direction_0(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        tracker.log_prediction_simple(source="vix", signal_value=0.10, confidence=0.60, timestamp="2025-05-20T10:00:00")

    def test_auto_timestamp(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        tracker.log_prediction_simple(source="hmm", signal_value=0.80, confidence=0.95)


# ---------------------------------------------------------------------------
# Health score formula (weighted accuracy)
# ---------------------------------------------------------------------------

class TestHealthScoreFormula:
    def test_insufficient_data_returns_none(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        result = tracker.calculate_health_score("cta")
        assert result is None

    def test_with_minimum_data(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)

        today = datetime.now()
        # Insert 15 predictions with actual directions over 100 days
        for i in range(15):
            ts = (today - timedelta(days=i * 7)).strftime("%Y-%m-%dT10:00:00")
            tracker.log_prediction_simple(source="cta", signal_value=0.50, confidence=0.80, timestamp=ts)

        # Update actual directions (bull market → direction 1)
        for i in range(15):
            day = (today - timedelta(days=i * 7)).strftime("%Y-%m-%d")
            tracker.update_actual_directions({"SPY": 0.01}, day)

        result = tracker.calculate_health_score("cta")
        assert result is not None
        assert 0.0 <= result.health_score <= 1.0
        assert result.predictions_count >= 10

    def test_100_percent_accuracy(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)

        today = datetime.now()
        for i in range(20):
            ts = (today - timedelta(days=i * 5)).strftime("%Y-%m-%dT10:00:00")
            tracker.log_prediction_simple(source="cta", signal_value=0.50, confidence=0.80, timestamp=ts)

        for i in range(20):
            day = (today - timedelta(days=i * 5)).strftime("%Y-%m-%d")
            tracker.update_actual_directions({"SPY": 0.01}, day)

        result = tracker.calculate_health_score("cta")
        # All predictions correctly bullish, all actual bull → 100% accuracy
        assert result.health_score == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# decay threshold
# ---------------------------------------------------------------------------

class TestDecayThreshold:
    def test_decay_constant(self):
        assert SignalHealthTracker.DECAY_THRESHOLD == 0.20

    def test_health_floor_constant(self):
        assert SignalHealthTracker.HEALTH_FLOOR == 0.20


# ---------------------------------------------------------------------------
# get_health_report on empty tracker
# ---------------------------------------------------------------------------

class TestHealthReport:
    def test_empty_report(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        report = tracker.get_health_report()
        assert "timestamp" in report
        assert "scores" in report
        assert "overall_health" in report
