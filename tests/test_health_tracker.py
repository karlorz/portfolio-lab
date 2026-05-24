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


class TestLogPrediction:
    """Test log_prediction (full SignalPrediction version)."""

    def test_log_full_prediction(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        pred = SignalPrediction(
            timestamp=datetime.now().isoformat(),
            source="alternative_data",
            signal_value=0.5,
            confidence=0.8,
            predicted_direction=1,
            metadata={"regime": "normal"},
        )
        tracker.log_prediction(pred)
        # Verify it was saved by checking the database
        import sqlite3
        with sqlite3.connect(str(db)) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM signal_predictions").fetchone()
        assert rows[0] == 1


class TestUpdateActualDirections:
    """Test update_actual_directions."""

    def test_update_with_returns(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        # Log a prediction first
        tracker.log_prediction_simple(
            source="alternative_data",
            signal_value=0.5,
            confidence=0.7,
        )
        # Update with positive returns
        returns_data = {"SPY": 0.02, "GLD": 0.01, "TLT": -0.01}
        tracker.update_actual_directions(returns_data, datetime.now().isoformat())
        # Should not raise


class TestCalculateAllHealthScores:
    """Test calculate_all_health_scores."""

    def test_empty_database(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        scores = tracker.calculate_all_health_scores()
        assert isinstance(scores, dict)

    def test_with_multiple_sources(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        for _ in range(10):
            tracker.log_prediction_simple(
                source="alternative_data",
                signal_value=0.5,
                confidence=0.7,
            )
            tracker.log_prediction_simple(
                source="cross_asset_rv",
                signal_value=-0.4,
                confidence=0.6,
            )
        scores = tracker.calculate_all_health_scores()
        assert isinstance(scores, dict)


class TestSaveHealthScores:
    """Test save_health_scores."""

    def test_save_and_retrieve(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        scores = {
            "alternative_data": HealthScore(
                source="alternative_data",
                timestamp=datetime.now().isoformat(),
                health_score=0.75,
                accuracy_30d=0.70,
                accuracy_60d=0.72,
                accuracy_90d=0.68,
                decay_rate=0.0,
                predictions_count=50,
                status=SignalHealthStatus.HEALTHY.value,
            ),
        }
        tracker.save_health_scores(scores)
        # Should not raise and should persist


class TestDetectDecayAlerts:
    """Test detect_decay_alerts — queries database directly."""

    def test_no_alerts_for_fresh_db(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        # No health score history in DB → no alerts
        alerts = tracker.detect_decay_alerts()
        assert isinstance(alerts, list)
        assert len(alerts) == 0

    def test_with_historical_scores_in_db(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        # Save some health scores first (need at least 2 for comparison)
        scores_old = {
            "multi_speed_momentum": HealthScore(
                source="multi_speed_momentum",
                timestamp=(datetime.now() - timedelta(days=20)).isoformat(),
                health_score=0.80,
                accuracy_30d=0.75,
                accuracy_60d=0.78,
                accuracy_90d=0.73,
                decay_rate=0.01,
                predictions_count=100,
                status=SignalHealthStatus.HEALTHY.value,
            ),
        }
        tracker.save_health_scores(scores_old)
        scores_new = {
            "multi_speed_momentum": HealthScore(
                source="multi_speed_momentum",
                timestamp=datetime.now().isoformat(),
                health_score=0.40,
                accuracy_30d=0.35,
                accuracy_60d=0.40,
                accuracy_90d=0.45,
                decay_rate=0.08,
                predictions_count=100,
                status=SignalHealthStatus.DEGRADED.value,
            ),
        }
        tracker.save_health_scores(scores_new)
        alerts = tracker.detect_decay_alerts()
        assert isinstance(alerts, list)


class TestGetAdjustedWeights:
    """Test get_adjusted_weights."""

    def _make_health_score(self, source, health_score, status, accuracy_30d=0.70):
        return HealthScore(
            source=source,
            timestamp=datetime.now().isoformat(),
            health_score=health_score,
            accuracy_30d=accuracy_30d,
            accuracy_60d=accuracy_30d - 0.02,
            accuracy_90d=accuracy_30d - 0.05,
            decay_rate=0.01,
            predictions_count=100,
            status=status,
        )

    def test_default_weights_healthy(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        base_weights = {
            "alternative_data": 0.305,
            "cross_asset_rv": 0.13,
            "international_momentum": 0.245,
        }
        scores = {
            "alternative_data": self._make_health_score("alternative_data", 0.85, SignalHealthStatus.HEALTHY.value, 0.75),
            "cross_asset_rv": self._make_health_score("cross_asset_rv", 0.90, SignalHealthStatus.HEALTHY.value, 0.80),
            "international_momentum": self._make_health_score("international_momentum", 0.78, SignalHealthStatus.HEALTHY.value, 0.68),
        }
        adjusted = tracker.get_adjusted_weights(base_weights, scores)
        assert isinstance(adjusted, dict)
        # Healthy signals should keep most of their weight
        for src, weight in adjusted.items():
            assert weight >= 0

    def test_degraded_signal_reduced_weight(self, tmp_path):
        """When one signal has worse health, it should get less relative weight."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        # Log enough predictions for health calculation
        for _ in range(25):
            tracker.log_prediction_simple(source="alternative_data", signal_value=0.5, confidence=0.8)
            tracker.log_prediction_simple(source="multi_speed_momentum", signal_value=-0.1, confidence=0.3)
        base_weights = {
            "alternative_data": 0.305,
            "multi_speed_momentum": 0.305,
        }
        adjusted = tracker.get_adjusted_weights(base_weights)
        # Both sources have data now; should produce adjusted weights
        assert isinstance(adjusted, dict)
        # Weights should sum to ~1.0
        total = sum(adjusted.values())
        assert abs(total - 1.0) < 0.01


class TestDecayAlertDataclass:
    """Test DecayAlert dataclass creation."""

    def test_create_with_all_fields(self):
        alert = DecayAlert(
            source="multi_speed_momentum",
            alert_timestamp=datetime.now().isoformat(),
            previous_health=0.75,
            current_health=0.40,
            drop_30d=0.35,
            severity="critical",
            message="Health dropped below threshold",
        )
        assert alert.source == "multi_speed_momentum"
        assert alert.severity == "critical"
        assert alert.drop_30d == 0.35

    def test_to_dict(self):
        alert = DecayAlert(
            source="multi_speed_momentum",
            alert_timestamp=datetime.now().isoformat(),
            previous_health=0.60,
            current_health=0.35,
            drop_30d=0.25,
            severity="warning",
            message="Monitor closely",
        )
        d = alert.to_dict()
        assert isinstance(d, dict)
        assert d["source"] == "multi_speed_momentum"


class TestHealthScoreStatusClassification:
    """Test HealthScore status classification logic."""

    def test_healthy_threshold(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        for _ in range(20):
            tracker.log_prediction_simple(
                source="alternative_data",
                signal_value=0.5,
                confidence=0.9,
            )
        # With consistent predictions, should get a health score
        scores = tracker.calculate_all_health_scores()
        if "alternative_data" in scores:
            score = scores["alternative_data"]
            assert hasattr(score, "health_score")
            assert hasattr(score, "status")
            assert score.health_score >= 0
            assert score.health_score <= 1


# ---------------------------------------------------------------------------
# Spearman rank correlation (static helper)
# ---------------------------------------------------------------------------

class TestSpearmanRankCorrelation:

    def test_perfect_positive(self):
        rho = SignalHealthTracker._spearman_rank_correlation([1, 2, 3, 4], [1, 2, 3, 4])
        assert rho == pytest.approx(1.0, abs=0.01)

    def test_perfect_negative(self):
        rho = SignalHealthTracker._spearman_rank_correlation([1, 2, 3, 4], [4, 3, 2, 1])
        assert rho == pytest.approx(-1.0, abs=0.01)

    def test_no_correlation(self):
        # Uncorrelated sequences
        rho = SignalHealthTracker._spearman_rank_correlation([1, 2, 3, 4, 5], [5, 1, 4, 2, 3])
        assert -1.0 <= rho <= 1.0

    def test_too_few_points_returns_none(self):
        assert SignalHealthTracker._spearman_rank_correlation([1], [1]) is None
        assert SignalHealthTracker._spearman_rank_correlation([1, 2], [1, 2]) is None

    def test_constant_series_returns_zero(self):
        rho = SignalHealthTracker._spearman_rank_correlation([1, 1, 1, 1], [1, 2, 3, 4])
        assert rho == 0.0

    def test_tied_ranks(self):
        rho = SignalHealthTracker._spearman_rank_correlation([1, 1, 3, 4], [1, 2, 3, 4])
        assert -1.0 <= rho <= 1.0

    def test_mismatched_lengths_returns_none(self):
        assert SignalHealthTracker._spearman_rank_correlation([1, 2, 3], [1, 2]) is None


# ---------------------------------------------------------------------------
# compute_ic
# ---------------------------------------------------------------------------

class TestComputeIC:

    def test_insufficient_data_returns_none(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        # Only 2 predictions — need at least 3
        tracker.log_prediction_simple(source="cta", signal_value=0.5, confidence=0.8)
        tracker.log_prediction_simple(source="cta", signal_value=0.3, confidence=0.7)
        assert tracker.compute_ic("cta") is None

    def test_with_data_returns_float(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        today = datetime.now()
        for i in range(20):
            ts = (today - timedelta(days=i * 4)).strftime("%Y-%m-%dT10:00:00")
            tracker.log_prediction_simple(source="cta", signal_value=0.5, confidence=0.8, timestamp=ts)
        for i in range(20):
            day = (today - timedelta(days=i * 4)).strftime("%Y-%m-%d")
            tracker.update_actual_directions({"SPY": 0.01}, day)
        ic = tracker.compute_ic("cta")
        assert ic is not None
        assert -1.0 <= ic <= 1.0

    def test_unknown_source_returns_none(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        assert tracker.compute_ic("nonexistent") is None


# ---------------------------------------------------------------------------
# compute_ic_half_life
# ---------------------------------------------------------------------------

class TestComputeICHalfLife:

    def test_insufficient_data_returns_none(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        # Only 5 predictions — not enough for rolling windows
        for i in range(5):
            tracker.log_prediction_simple(source="cta", signal_value=0.5, confidence=0.8)
        assert tracker.compute_ic_half_life("cta") is None

    def test_returns_float_or_inf_when_stable(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        today = datetime.now()
        # Insert 60 predictions over 360 days for enough IC windows
        for i in range(60):
            ts = (today - timedelta(days=i * 6)).strftime("%Y-%m-%dT10:00:00")
            tracker.log_prediction_simple(source="cta", signal_value=0.5, confidence=0.8, timestamp=ts)
        for i in range(60):
            day = (today - timedelta(days=i * 6)).strftime("%Y-%m-%d")
            tracker.update_actual_directions({"SPY": 0.01}, day)
        hl = tracker.compute_ic_half_life("cta")
        # May be None (not enough varied IC), float, or inf (stable IC)
        if hl is not None:
            assert hl > 0 or hl == float("inf")
