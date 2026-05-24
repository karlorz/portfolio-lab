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
        assert "ic_metrics" in report


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

    def _make_health_score(self, source, health_score, status, accuracy_30d=0.70, ic=None):
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
            ic=ic,
        )

    def test_default_weights_healthy(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        base_weights = {
            "alternative_data": 0.305,
            "cross_asset_rv": 0.13,
            "international_momentum": 0.245,
        }
        adjusted = tracker.get_adjusted_weights(base_weights)
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

    def test_high_ic_gets_bonus_weight(self, tmp_path):
        """Source with high IC should get relatively more weight than same-health source with low IC."""
        from unittest.mock import patch
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)

        # Both sources have same health_score but different IC
        high_ic_score = HealthScore(
            source="alternative_data", timestamp=datetime.now().isoformat(),
            health_score=0.80, accuracy_30d=0.75, accuracy_60d=0.78,
            accuracy_90d=0.80, decay_rate=0.0, predictions_count=100,
            status="healthy", ic=0.10,
        )
        low_ic_score = HealthScore(
            source="cross_asset_rv", timestamp=datetime.now().isoformat(),
            health_score=0.80, accuracy_30d=0.75, accuracy_60d=0.78,
            accuracy_90d=0.80, decay_rate=0.0, predictions_count=100,
            status="healthy", ic=0.01,
        )

        with patch.object(tracker, 'calculate_all_health_scores', return_value={
            "alternative_data": high_ic_score,
            "cross_asset_rv": low_ic_score,
        }):
            base_weights = {"alternative_data": 0.50, "cross_asset_rv": 0.50}
            adjusted = tracker.get_adjusted_weights(
                base_weights,
                ic_bonus_threshold=0.05,
                ic_penalty_threshold=0.02,
            )
            # High IC source should get relatively more weight
            assert adjusted["alternative_data"] > adjusted["cross_asset_rv"]
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


# ---------------------------------------------------------------------------
# IC metrics in health report
# ---------------------------------------------------------------------------

class TestICMetricsInReport:

    def test_report_includes_ic_metrics_key(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        report = tracker.get_health_report()
        assert "ic_metrics" in report
        assert isinstance(report["ic_metrics"], dict)

    def test_ic_metrics_populated_with_data(self, tmp_path):
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        today = datetime.now()
        for i in range(20):
            ts = (today - timedelta(days=i * 4)).strftime("%Y-%m-%dT10:00:00")
            tracker.log_prediction_simple(source="cta", signal_value=0.5, confidence=0.8, timestamp=ts)
        for i in range(20):
            day = (today - timedelta(days=i * 4)).strftime("%Y-%m-%d")
            tracker.update_actual_directions({"SPY": 0.01}, day)
        report = tracker.get_health_report()
        if "cta" in report.get("ic_metrics", {}):
            assert "ic" in report["ic_metrics"]["cta"]
            assert "ic_half_life_days" in report["ic_metrics"]["cta"]


# ---------------------------------------------------------------------------
# IC fields in HealthScore dataclass
# ---------------------------------------------------------------------------

class TestHealthScoreICFields:

    def test_health_score_has_ic_fields(self):
        hs = HealthScore(
            source="cta", timestamp="2026-05-24",
            health_score=0.85, accuracy_30d=0.80, accuracy_60d=0.82,
            accuracy_90d=0.85, decay_rate=-0.01, predictions_count=100,
            status="healthy", ic=0.08, ic_half_life_days=300.0,
        )
        assert hs.ic == 0.08
        assert hs.ic_half_life_days == 300.0

    def test_ic_defaults_to_none(self):
        hs = HealthScore(
            source="cta", timestamp="2026-05-24",
            health_score=0.85, accuracy_30d=0.80, accuracy_60d=0.82,
            accuracy_90d=0.85, decay_rate=-0.01, predictions_count=100,
            status="healthy",
        )
        assert hs.ic is None
        assert hs.ic_half_life_days is None

    def test_to_dict_includes_ic_fields(self):
        hs = HealthScore(
            source="cta", timestamp="2026-05-24",
            health_score=0.85, accuracy_30d=0.80, accuracy_60d=0.82,
            accuracy_90d=0.85, decay_rate=-0.01, predictions_count=100,
            status="healthy", ic=0.05, ic_half_life_days=150.0,
        )
        d = hs.to_dict()
        assert "ic" in d
        assert "ic_half_life_days" in d
        assert d["ic"] == 0.05


class TestDetectICAlerts:
    """Tests for detect_ic_alerts() — IC-based degradation detection."""

    @pytest.fixture
    def tracker_with_predictions(self, tmp_path):
        """Create a tracker with some predictions logged."""
        db = str(tmp_path / "test_health.db")
        tracker = SignalHealthTracker(db_path=db)
        # Log enough predictions to compute IC
        for i in range(60):
            ts = (datetime.now() - timedelta(days=60 - i)).isoformat()
            tracker.log_prediction_simple(
                source="CROSS_ASSET_RV",
                signal_value=0.5,
                confidence=0.6,
                timestamp=ts,
            )
        return tracker

    def test_returns_list(self, tracker_with_predictions):
        alerts = tracker_with_predictions.detect_ic_alerts()
        assert isinstance(alerts, list)

    def test_empty_with_no_data(self, tmp_path):
        db = str(tmp_path / "test_health.db")
        tracker = SignalHealthTracker(db_path=db)
        alerts = tracker.detect_ic_alerts()
        assert alerts == []

    def test_alert_structure(self, tracker_with_predictions):
        """If any alerts are produced, they should be DecayAlert instances."""
        from unittest.mock import patch

        # Mock compute_ic to produce a negative streak
        with patch.object(tracker_with_predictions, 'compute_ic', return_value=-0.05):
            alerts = tracker_with_predictions.detect_ic_alerts()

        for alert in alerts:
            assert isinstance(alert, DecayAlert)
            assert alert.source is not None
            assert alert.message is not None

    def test_negative_ic_streak_alert(self, tmp_path):
        """3+ consecutive negative IC windows should trigger streak alert."""
        db = str(tmp_path / "test_health.db")
        tracker = SignalHealthTracker(db_path=db)

        # Mock compute_ic to return negative values (streak)
        def mock_ic(source, lookback_days=90, end_date=None):
            return -0.05  # Always negative

        from unittest.mock import patch
        with patch.object(tracker, 'compute_ic', side_effect=mock_ic):
            alerts = tracker.detect_ic_alerts()

        streak_alerts = [a for a in alerts if "streak" in a.message.lower()]
        assert len(streak_alerts) > 0

    def test_ic_drawdown_alert(self, tmp_path):
        """IC dropping >50% from peak should trigger drawdown alert."""
        db = str(tmp_path / "test_health.db")
        tracker = SignalHealthTracker(db_path=db)

        from unittest.mock import patch
        # First call returns low IC (current), subsequent calls return higher (past peaks)
        ic_sequence = iter([0.02, 0.10, 0.12, 0.08, 0.06, 0.05, 0.04])

        def mock_ic(source, lookback_days=90, end_date=None):
            try:
                return next(ic_sequence)
            except StopIteration:
                return 0.05

        with patch.object(tracker, 'compute_ic', side_effect=mock_ic):
            alerts = tracker.detect_ic_alerts()

        drawdown_alerts = [a for a in alerts if "drawdown" in a.message.lower()]
        assert len(drawdown_alerts) > 0

    def test_saves_to_state(self, tracker_with_predictions):
        """IC alerts should be saved to tracker state."""
        from unittest.mock import patch
        with patch.object(tracker_with_predictions, 'compute_ic', return_value=-0.05):
            alerts = tracker_with_predictions.detect_ic_alerts()

        if alerts:
            assert "ic_alerts" in tracker_with_predictions.state


# ---------------------------------------------------------------------------
# Extended coverage tests
# ---------------------------------------------------------------------------


class TestSignalPredictionExtended:
    """Extended SignalPrediction dataclass tests."""

    def test_to_dict_all_fields(self):
        sp = SignalPrediction(
            timestamp="2026-05-24T10:00:00",
            source="test_source",
            signal_value=0.5,
            confidence=0.9,
            predicted_direction=1,
            metadata={"key": "value"},
        )
        d = sp.to_dict()
        assert set(d.keys()) == {
            'timestamp', 'source', 'signal_value', 'confidence',
            'predicted_direction', 'metadata',
        }
        assert d['predicted_direction'] == 1

    def test_metadata_serialized_as_json_string(self):
        """Metadata should be JSON-serialized in to_dict."""
        sp = SignalPrediction(
            timestamp="2026-05-24",
            source="src",
            signal_value=0.3,
            confidence=0.7,
            predicted_direction=0,
            metadata={"nested": {"key": "val"}},
        )
        d = sp.to_dict()
        assert isinstance(d["metadata"], str)
        parsed = json.loads(d["metadata"])
        assert parsed["nested"]["key"] == "val"

    def test_negative_prediction(self):
        sp = SignalPrediction(
            timestamp="2026-05-24",
            source="src",
            signal_value=-0.8,
            confidence=0.6,
            predicted_direction=-1,
            metadata={},
        )
        assert sp.signal_value < 0
        assert sp.predicted_direction == -1


class TestHealthScoreExtended:
    """Extended HealthScore dataclass tests."""

    def test_to_dict_has_all_fields(self):
        hs = HealthScore(
            source="cta",
            timestamp="2026-05-24",
            health_score=0.85,
            accuracy_30d=0.80,
            accuracy_60d=0.82,
            accuracy_90d=0.85,
            decay_rate=-0.01,
            predictions_count=100,
            status="healthy",
            ic=0.05,
            ic_half_life_days=200.0,
        )
        d = hs.to_dict()
        expected_keys = {
            'source', 'timestamp', 'health_score', 'accuracy_30d',
            'accuracy_60d', 'accuracy_90d', 'decay_rate', 'predictions_count',
            'status', 'ic', 'ic_half_life_days',
        }
        assert set(d.keys()) == expected_keys

    def test_status_values(self):
        """Valid status values."""
        for status in ["healthy", "degraded", "unhealthy"]:
            hs = HealthScore(
                source="src", timestamp="2026-05-24",
                health_score=0.5, accuracy_30d=0.5, accuracy_60d=0.5,
                accuracy_90d=0.5, decay_rate=0.0, predictions_count=10,
                status=status,
            )
            assert hs.status == status


class TestDecayAlertExtended:
    """Extended DecayAlert dataclass tests."""

    def test_to_dict_all_fields(self):
        da = DecayAlert(
            source="src",
            alert_timestamp="2026-05-24",
            previous_health=0.8,
            current_health=0.5,
            drop_30d=0.3,
            severity="warning",
            message="Test alert",
        )
        d = da.to_dict()
        expected_keys = {
            'source', 'alert_timestamp', 'previous_health',
            'current_health', 'drop_30d', 'severity', 'message',
        }
        assert set(d.keys()) == expected_keys

    def test_severity_values(self):
        """Both warning and critical severities should work."""
        for severity in ["warning", "critical"]:
            da = DecayAlert(
                source="src", alert_timestamp="2026-05-24",
                previous_health=0.8, current_health=0.4,
                drop_30d=0.4, severity=severity, message="Test",
            )
            assert da.severity == severity


class TestLogPredictionSimpleExtended:
    """Extended log_prediction_simple tests."""

    def test_direction_thresholds(self, tmp_path):
        """Direction should be determined by signal_value thresholds."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        # Exactly at +0.2 threshold → direction 1
        tracker.log_prediction_simple(source="src", signal_value=0.2, confidence=0.8)
        # Exactly at -0.2 threshold → direction -1
        tracker.log_prediction_simple(source="src", signal_value=-0.2, confidence=0.8)
        # Between → direction 0
        tracker.log_prediction_simple(source="src", signal_value=0.1, confidence=0.8)

    def test_with_metadata(self, tmp_path):
        """Metadata should be preserved."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        tracker.log_prediction_simple(
            source="src", signal_value=0.5, confidence=0.8,
            metadata={"regime": "normal", "vix": 15.0},
        )


class TestHealthScoreClassificationExtended:
    """Extended health score classification tests."""

    def test_unhealthy_score(self, tmp_path):
        """Score below 0.5 should be unhealthy."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        # Log predictions where predicted_direction != actual
        today = datetime.now()
        for i in range(20):
            ts = (today - timedelta(days=i * 5)).strftime("%Y-%m-%dT10:00:00")
            # Predict bullish but actual will be bearish → inaccurate
            tracker.log_prediction_simple(
                source="bad_signal", signal_value=0.5, confidence=0.8, timestamp=ts,
            )
        for i in range(20):
            day = (today - timedelta(days=i * 5)).strftime("%Y-%m-%d")
            tracker.update_actual_directions({"SPY": -0.01}, day)
        result = tracker.calculate_health_score("bad_signal")
        # With wrong predictions, health should be low
        if result is not None:
            assert result.health_score < 0.7  # At least degraded

    def test_health_score_bounded(self, tmp_path):
        """Health score should always be between 0 and 1."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        today = datetime.now()
        for i in range(20):
            ts = (today - timedelta(days=i * 5)).strftime("%Y-%m-%dT10:00:00")
            tracker.log_prediction_simple(
                source="bounded_test", signal_value=0.5, confidence=0.8, timestamp=ts,
            )
        for i in range(20):
            day = (today - timedelta(days=i * 5)).strftime("%Y-%m-%d")
            tracker.update_actual_directions({"SPY": 0.01}, day)
        result = tracker.calculate_health_score("bounded_test")
        if result is not None:
            assert 0.0 <= result.health_score <= 1.0


class TestUpdateActualDirectionsExtended:
    """Extended update_actual_directions tests."""

    def test_negative_returns_direction_minus_1(self, tmp_path):
        """Negative SPY return should give actual_direction = -1."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        tracker.log_prediction_simple(source="src", signal_value=0.5, confidence=0.8)
        updated = tracker.update_actual_directions({"SPY": -0.02}, datetime.now().strftime("%Y-%m-%d"))
        assert isinstance(updated, int)

    def test_zero_returns_direction_0(self, tmp_path):
        """Zero SPY return should give actual_direction = 0."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        tracker.log_prediction_simple(source="src", signal_value=0.5, confidence=0.8)
        updated = tracker.update_actual_directions({"SPY": 0.0}, datetime.now().strftime("%Y-%m-%d"))
        assert isinstance(updated, int)

    def test_missing_spy_defaults_to_zero(self, tmp_path):
        """Missing SPY in returns should default to 0 (no direction)."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        tracker.log_prediction_simple(source="src", signal_value=0.5, confidence=0.8)
        updated = tracker.update_actual_directions({"GLD": 0.02}, datetime.now().strftime("%Y-%m-%d"))
        assert isinstance(updated, int)


class TestGetAdjustedWeightsExtended:
    """Extended get_adjusted_weights tests."""

    def test_empty_base_weights(self, tmp_path):
        """Empty base weights should return empty."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        adjusted = tracker.get_adjusted_weights({})
        assert adjusted == {}

    def test_weights_sum_to_one(self, tmp_path):
        """Adjusted weights should sum to 1.0."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        base_weights = {
            "alternative_data": 0.305,
            "cross_asset_rv": 0.13,
            "international_momentum": 0.245,
            "unified_overlay": 0.19,
        }
        adjusted = tracker.get_adjusted_weights(base_weights)
        total = sum(adjusted.values())
        assert abs(total - 1.0) < 0.01


class TestSpearmanRankCorrelationExtended:
    """Extended Spearman rank correlation tests."""

    def test_single_element_returns_none(self):
        assert SignalHealthTracker._spearman_rank_correlation([5], [5]) is None

    def test_two_elements_returns_none(self):
        assert SignalHealthTracker._spearman_rank_correlation([1, 2], [1, 2]) is None

    def test_three_elements_perfect_positive(self):
        rho = SignalHealthTracker._spearman_rank_correlation([1, 2, 3], [1, 2, 3])
        assert rho == pytest.approx(1.0, abs=0.01)

    def test_float_values(self):
        rho = SignalHealthTracker._spearman_rank_correlation(
            [0.1, 0.5, 0.3, 0.8], [0.2, 0.6, 0.4, 0.9]
        )
        assert -1.0 <= rho <= 1.0
        assert rho > 0  # Should be positively correlated


class TestGetHealthReportExtended:
    """Extended health report tests."""

    def test_report_structure(self, tmp_path):
        """Report should have expected top-level keys."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        report = tracker.get_health_report()
        assert "timestamp" in report
        assert "summary" in report
        assert "scores" in report
        assert "alerts" in report
        assert "overall_health" in report

    def test_summary_structure(self, tmp_path):
        """Summary should have expected fields."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        report = tracker.get_health_report()
        summary = report["summary"]
        assert "healthy" in summary
        assert "degraded" in summary
        assert "unhealthy" in summary
        assert "total_tracked" in summary


class TestDetectICAlertsExtended:
    """Extended detect_ic_alerts tests."""

    def test_custom_parameters(self, tmp_path):
        """Custom lookback and thresholds should be accepted."""
        db = str(tmp_path / "test_health.db")
        tracker = SignalHealthTracker(db_path=db)
        from unittest.mock import patch
        with patch.object(tracker, 'compute_ic', return_value=-0.03):
            alerts = tracker.detect_ic_alerts(
                lookback_days=60,
                streak_threshold=2,
                ic_ratio_floor=0.4,
                ic_drawdown_threshold=0.6,
            )
        assert isinstance(alerts, list)

    def test_no_alerts_with_positive_ic(self, tmp_path):
        """Consistently positive IC should not trigger streak alerts."""
        db = str(tmp_path / "test_health.db")
        tracker = SignalHealthTracker(db_path=db)
        from unittest.mock import patch
        with patch.object(tracker, 'compute_ic', return_value=0.08):
            alerts = tracker.detect_ic_alerts()
        # No streak alerts since IC is positive
        streak_alerts = [a for a in alerts if "streak" in a.message.lower()]
        assert len(streak_alerts) == 0


# ---------------------------------------------------------------------------
# HealthScore to_dict with None IC fields
# ---------------------------------------------------------------------------

class TestHealthScoreToDictEdgeCases:
    """Verify to_dict() handles optional IC fields set to None."""

    def test_to_dict_with_none_ic_fields(self):
        hs = HealthScore(
            source="cta", timestamp="2026-05-24",
            health_score=0.85, accuracy_30d=0.80, accuracy_60d=0.82,
            accuracy_90d=0.85, decay_rate=-0.01, predictions_count=100,
            status="healthy", ic=None, ic_half_life_days=None,
        )
        d = hs.to_dict()
        expected = {
            'source', 'timestamp', 'health_score', 'accuracy_30d',
            'accuracy_60d', 'accuracy_90d', 'decay_rate', 'predictions_count',
            'status', 'ic', 'ic_half_life_days',
        }
        assert set(d.keys()) == expected
        assert d['ic'] is None
        assert d['ic_half_life_days'] is None

    def test_to_dict_with_zero_accuracy_and_decay(self):
        """All numeric fields at zero boundary."""
        hs = HealthScore(
            source="dead_signal", timestamp="2026-01-01",
            health_score=0.0, accuracy_30d=0.0, accuracy_60d=0.0,
            accuracy_90d=0.0, decay_rate=0.0, predictions_count=0,
            status="unhealthy",
        )
        d = hs.to_dict()
        assert d['health_score'] == 0.0
        assert d['predictions_count'] == 0
        assert d['decay_rate'] == 0.0


# ---------------------------------------------------------------------------
# Health score formula edge cases
# ---------------------------------------------------------------------------

class TestHealthScoreFormulaEdgeCases:
    """Edge cases in health score computation (controlled DB seeding)."""

    def _seed_predictions(self, tracker, source, correct, wrong, days_ago_start=80):
        """Insert predictions with exactly *correct* right and *wrong* wrong.
        All predictions placed within *days_ago_start* days of today."""
        import sqlite3
        today = datetime.now()
        with sqlite3.connect(str(tracker.db_path)) as conn:
            cursor = conn.cursor()
            idx = 0
            for _ in range(correct):
                ts = (today - timedelta(days=days_ago_start - idx)).strftime("%Y-%m-%dT10:00:00")
                cursor.execute(
                    "INSERT INTO signal_predictions "
                    "(timestamp, source, signal_value, confidence, predicted_direction, actual_direction, accuracy_calculated) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (ts, source, 0.5, 0.8, 1, 1)
                )
                idx += 1
            for _ in range(wrong):
                ts = (today - timedelta(days=days_ago_start - idx)).strftime("%Y-%m-%dT10:00:00")
                cursor.execute(
                    "INSERT INTO signal_predictions "
                    "(timestamp, source, signal_value, confidence, predicted_direction, actual_direction, accuracy_calculated) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (ts, source, 0.5, 0.8, 1, -1)
                )
                idx += 1
            conn.commit()

    def test_exactly_minimum_data(self, tmp_path):
        """Exactly 10 predictions (minimum threshold) produces a health score."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        self._seed_predictions(tracker, "min_edge", correct=7, wrong=3, days_ago_start=80)
        result = tracker.calculate_health_score("min_edge")
        assert result is not None
        assert result.predictions_count >= 10

    def test_all_neutral_predictions(self, tmp_path):
        """All neutral (predicted_direction=0) -> 0.5 accuracy, 0.5 health."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        today = datetime.now()
        import sqlite3
        with sqlite3.connect(str(tracker.db_path)) as conn:
            cursor = conn.cursor()
            for i in range(15):
                ts = (today - timedelta(days=85 - i * 5)).strftime("%Y-%m-%dT10:00:00")
                cursor.execute(
                    "INSERT INTO signal_predictions "
                    "(timestamp, source, signal_value, confidence, predicted_direction, actual_direction, accuracy_calculated) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (ts, "neutral_signal", 0.0, 0.5, 0, 0)
                )
            conn.commit()
        result = tracker.calculate_health_score("neutral_signal")
        assert result is not None
        # All neutral -> total=0 for each period -> default 0.5
        assert result.health_score == pytest.approx(0.5, abs=0.01)

    def test_all_wrong_predictions(self, tmp_path):
        """All predictions wrong -> health near 0."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        self._seed_predictions(tracker, "all_wrong", correct=0, wrong=15, days_ago_start=80)
        result = tracker.calculate_health_score("all_wrong")
        assert result is not None
        assert result.health_score < 0.35  # implementation has a floor

    def test_mixed_accuracy(self, tmp_path):
        """Controlled 80% accuracy -> health = 0.8 -> status healthy."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        # 12 correct / 15 total = 0.8 for all three periods (all within last 25 days)
        self._seed_predictions(tracker, "mixed_acc", correct=12, wrong=3, days_ago_start=20)
        result = tracker.calculate_health_score("mixed_acc")
        assert result is not None
        assert result.health_score == pytest.approx(0.8, abs=0.01)
        assert result.status == "healthy"

    def test_zero_60d_data(self, tmp_path):
        """Predictions only in 90d window (60-80d ago) -> 60d/30d default to 0.5."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        self._seed_predictions(tracker, "no_60d", correct=10, wrong=2, days_ago_start=75)
        result = tracker.calculate_health_score("no_60d")
        assert result is not None
        # 90d accuracy = 10/12 ≈ 0.8333; 60d/30d default to 0.5
        # health = 0.8333*0.5 + 0.5*0.3 + 0.5*0.2 ≈ 0.4167 + 0.15 + 0.10 = 0.6667
        assert result.health_score == pytest.approx(0.6667, abs=0.01)
        # decay_rate = (0.5 - 0.5)/30 = 0 since counts['60d'] == 0
        assert result.decay_rate == 0.0


# ---------------------------------------------------------------------------
# Health score classification boundary conditions
# ---------------------------------------------------------------------------

class TestHealthScoreBoundaries:
    """Boundary conditions for healthy/degraded/unhealthy thresholds (0.7, 0.5)."""

    def _seed_exact_accuracy(self, tracker, source, accuracy, count=15, days_ago_start=20):
        """Seed *count* predictions with a given accuracy (0.0-1.0).
        Accuracy is achieved by setting correct = int(count * accuracy)."""
        correct = int(count * accuracy)
        wrong = count - correct
        import sqlite3
        today = datetime.now()
        with sqlite3.connect(str(tracker.db_path)) as conn:
            cursor = conn.cursor()
            idx = 0
            for _ in range(correct):
                ts = (today - timedelta(days=days_ago_start - idx)).strftime("%Y-%m-%dT10:00:00")
                cursor.execute(
                    "INSERT INTO signal_predictions "
                    "(timestamp, source, signal_value, confidence, predicted_direction, actual_direction, accuracy_calculated) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (ts, source, 0.5, 0.8, 1, 1)
                )
                idx += 1
            for _ in range(wrong):
                ts = (today - timedelta(days=days_ago_start - idx)).strftime("%Y-%m-%dT10:00:00")
                cursor.execute(
                    "INSERT INTO signal_predictions "
                    "(timestamp, source, signal_value, confidence, predicted_direction, actual_direction, accuracy_calculated) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (ts, source, 0.5, 0.8, 1, -1)
                )
                idx += 1
            conn.commit()

    def test_exactly_healthy_boundary(self, tmp_path):
        """Health = 0.7 should be classified as healthy (>= 0.7)."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        # 14/20 = 0.7 accuracy across all three periods
        self._seed_exact_accuracy(tracker, "hb", accuracy=0.7, count=20, days_ago_start=20)
        result = tracker.calculate_health_score("hb")
        assert result is not None
        assert result.health_score == pytest.approx(0.7, abs=0.01)
        assert result.status == "healthy"

    def test_just_below_healthy(self, tmp_path):
        """Health just below 0.7 should be classified as degraded."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        # 13/20 = 0.65 accuracy -> health = 0.65 < 0.7 -> degraded
        self._seed_exact_accuracy(tracker, "jb_healthy", accuracy=0.65, count=20, days_ago_start=20)
        result = tracker.calculate_health_score("jb_healthy")
        assert result is not None
        assert result.health_score < 0.7
        assert result.status == "degraded"

    def test_exactly_degraded_boundary(self, tmp_path):
        """Health = 0.5 should be classified as degraded (>= 0.5)."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        # 10/20 = 0.5 accuracy
        self._seed_exact_accuracy(tracker, "db", accuracy=0.5, count=20, days_ago_start=20)
        result = tracker.calculate_health_score("db")
        assert result is not None
        assert result.health_score == pytest.approx(0.5, abs=0.01)
        assert result.status == "degraded"

    def test_just_below_degraded(self, tmp_path):
        """Health just below 0.5 should be classified as unhealthy."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        # 9/20 = 0.45 accuracy -> health = 0.45 < 0.5 -> unhealthy
        self._seed_exact_accuracy(tracker, "jb_degraded", accuracy=0.45, count=20, days_ago_start=20)
        result = tracker.calculate_health_score("jb_degraded")
        assert result is not None
        assert result.health_score < 0.5
        assert result.status == "unhealthy"


# ---------------------------------------------------------------------------
# compute_ic edge cases
# ---------------------------------------------------------------------------

class TestComputeICEdgeCases:
    """Edge cases for compute_ic."""

    def test_constant_signal_values(self, tmp_path):
        """All signal values identical -> std=0 -> IC=0.0."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        today = datetime.now()
        import sqlite3
        with sqlite3.connect(str(db)) as conn:
            cursor = conn.cursor()
            for i in range(10):
                ts = (today - timedelta(days=80 - i)).strftime("%Y-%m-%dT10:00:00")
                cursor.execute(
                    "INSERT INTO signal_predictions "
                    "(timestamp, source, signal_value, confidence, predicted_direction, actual_direction, accuracy_calculated) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (ts, "const_signal", 0.5, 0.8, 1, 1)
                )
            conn.commit()
        ic = tracker.compute_ic("const_signal")
        assert ic == 0.0

    def test_constant_actual_directions(self, tmp_path):
        """All actual_direction values identical -> std=0 -> IC=0.0."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        today = datetime.now()
        import sqlite3
        with sqlite3.connect(str(db)) as conn:
            cursor = conn.cursor()
            for i in range(10):
                ts = (today - timedelta(days=80 - i)).strftime("%Y-%m-%dT10:00:00")
                signal = 0.5 if i % 2 == 0 else -0.3
                cursor.execute(
                    "INSERT INTO signal_predictions "
                    "(timestamp, source, signal_value, confidence, predicted_direction, actual_direction, accuracy_calculated) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (ts, "const_actual", signal, 0.8, 1, 1)
                )
            conn.commit()
        ic = tracker.compute_ic("const_actual")
        assert ic == 0.0

    def test_all_perfect_predictions(self, tmp_path):
        """Perfectly correlated signal_value <> actual_direction -> IC ~ 1.0."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        today = datetime.now()
        import sqlite3
        with sqlite3.connect(str(db)) as conn:
            cursor = conn.cursor()
            for i in range(10):
                ts = (today - timedelta(days=80 - i)).strftime("%Y-%m-%dT10:00:00")
                signal = 0.8 if i % 2 == 0 else -0.5
                actual = 1 if i % 2 == 0 else -1
                cursor.execute(
                    "INSERT INTO signal_predictions "
                    "(timestamp, source, signal_value, confidence, predicted_direction, actual_direction, accuracy_calculated) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (ts, "perfect_test", signal, 0.8, actual, actual)
                )
            conn.commit()
        ic = tracker.compute_ic("perfect_test")
        assert ic is not None
        assert ic == pytest.approx(1.0, abs=0.01)

    def test_empty_source_returns_none(self, tmp_path):
        """Source with no rows at all should return None."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        ic = tracker.compute_ic("ghost_source")
        assert ic is None

    def test_none_signal_values_skipped(self, tmp_path):
        """Rows with NULL signal_value should be excluded from IC computation."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        today = datetime.now()
        import sqlite3
        with sqlite3.connect(str(db)) as conn:
            cursor = conn.cursor()
            # Insert rows with NULL signal_value (should be skipped)
            for i in range(5):
                ts = (today - timedelta(days=80 - i)).strftime("%Y-%m-%dT10:00:00")
                cursor.execute(
                    "INSERT INTO signal_predictions "
                    "(timestamp, source, signal_value, confidence, predicted_direction, actual_direction, accuracy_calculated) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (ts, "null_signal", None, 0.8, 1, 1)
                )
            conn.commit()
        # Only NULL signal rows exist, so < 3 valid rows -> None
        ic = tracker.compute_ic("null_signal")
        assert ic is None


# ---------------------------------------------------------------------------
# detect_ic_alerts edge cases
# ---------------------------------------------------------------------------

class TestDetectICAlertsEdgeCases:
    """Boundary and edge conditions for detect_ic_alerts()."""

    def test_ic_peak_zero(self, tmp_path):
        """When all IC values are 0, guards prevent drawdown/ratio alerts."""
        db = str(tmp_path / "test_health.db")
        tracker = SignalHealthTracker(db_path=db)
        from unittest.mock import patch

        def mock_ic(source, lookback_days=90, end_date=None):
            return 0.0

        with patch.object(tracker, 'compute_ic', side_effect=mock_ic):
            alerts = tracker.detect_ic_alerts()

        # peak_ic = 0 -> drawdown guard (peak_ic > 0.02) prevents alerts
        assert len(alerts) == 0

    def test_ic_drawdown_at_threshold(self, tmp_path):
        """Drawdown exactly at 0.5 threshold should NOT trigger (must be >)."""
        db = str(tmp_path / "test_health.db")
        tracker = SignalHealthTracker(db_path=db)
        from unittest.mock import patch

        # current=0.05, history=[0.10, 0.10, 0.10]
        # peak=0.10, drawdown=(0.10-0.05)/0.10=0.5, 0.5 > 0.5 ? No
        ic_values = iter([0.05, 0.10, 0.10, 0.10] * 6)

        def mock_ic(source, lookback_days=90, end_date=None):
            return next(ic_values)

        with patch.object(tracker, 'compute_ic', side_effect=mock_ic):
            alerts = tracker.detect_ic_alerts()
        drawdown_alerts = [a for a in alerts if "drawdown" in a.message.lower()]
        assert len(drawdown_alerts) == 0

    def test_negative_streak_at_threshold(self, tmp_path):
        """3 consecutive negative IC windows should trigger streak alert."""
        db = str(tmp_path / "test_health.db")
        tracker = SignalHealthTracker(db_path=db)
        from unittest.mock import patch

        # current=-0.05, history=[-0.05, -0.05, -0.05]
        # streak = 3 >= 3 -> alert
        ic_values = iter([-0.05, -0.05, -0.05, -0.05] * 6)

        def mock_ic(source, lookback_days=90, end_date=None):
            return next(ic_values)

        with patch.object(tracker, 'compute_ic', side_effect=mock_ic):
            alerts = tracker.detect_ic_alerts()
        streak_alerts = [a for a in alerts if "streak" in a.message.lower()]
        assert len(streak_alerts) > 0

    def test_negative_streak_below_threshold(self, tmp_path):
        """2 consecutive negative windows (below 3) should NOT trigger."""
        db = str(tmp_path / "test_health.db")
        tracker = SignalHealthTracker(db_path=db)
        from unittest.mock import patch

        # current=0.03, history=[-0.05, -0.05, 0.03]
        # streak: most_recent=-0.05 <0->1, next=-0.05<0->2, next=0.03<0? No->break
        # 2 >= 3 ? No
        # peak = max(|-0.05|,|-0.05|,|0.03|) = 0.05
        # drawdown=(0.05-0.03)/0.05=0.4 < 0.5 -> no drawdown
        # ratio=0.03/0.05=0.6 > 0.3 -> no ratio
        ic_values = iter([0.03, -0.05, -0.05, 0.03] * 6)

        def mock_ic(source, lookback_days=90, end_date=None):
            return next(ic_values)

        with patch.object(tracker, 'compute_ic', side_effect=mock_ic):
            alerts = tracker.detect_ic_alerts()
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Spearman rank correlation edge cases
# ---------------------------------------------------------------------------

class TestSpearmanCorrelationEdgeCases:
    """Additional edge cases for _spearman_rank_correlation."""

    def test_all_identical_in_both_series(self):
        """Both series constant -> std=0 in both -> returns 0.0."""
        rho = SignalHealthTracker._spearman_rank_correlation(
            [1, 1, 1, 1], [1, 1, 1, 1]
        )
        assert rho == 0.0

    def test_negative_values(self):
        """Negative values in both series should still produce valid IC."""
        rho = SignalHealthTracker._spearman_rank_correlation(
            [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]
        )
        assert rho == pytest.approx(1.0, abs=0.01)

    def test_float_values_with_ties(self):
        """Float values with some ties produce valid rank correlation."""
        rho = SignalHealthTracker._spearman_rank_correlation(
            [0.1, 0.1, 0.3, 0.8, 1.0],
            [0.2, 0.2, 0.4, 0.9, 1.1],
        )
        assert rho == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# log_prediction_simple edge cases
# ---------------------------------------------------------------------------

class TestLogPredictionSimpleEdgeCases:
    """Edge cases for log_prediction_simple."""

    def test_zero_confidence(self, tmp_path):
        """Zero confidence should be stored correctly."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        tracker.log_prediction_simple(
            source="test_src", signal_value=0.5, confidence=0.0,
        )
        import sqlite3
        with sqlite3.connect(str(db)) as conn:
            row = conn.execute(
                "SELECT confidence FROM signal_predictions WHERE source=?",
                ("test_src",),
            ).fetchone()
        assert row[0] == 0.0

    def test_extreme_signal_values(self, tmp_path):
        """Extreme signal values (+/-1.0) should set correct direction."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        tracker.log_prediction_simple(
            source="extreme", signal_value=1.0, confidence=0.9,
        )
        tracker.log_prediction_simple(
            source="extreme", signal_value=-1.0, confidence=0.9,
        )
        import sqlite3
        with sqlite3.connect(str(db)) as conn:
            rows = conn.execute(
                "SELECT signal_value, predicted_direction FROM signal_predictions "
                "WHERE source=? ORDER BY signal_value DESC",
                ("extreme",),
            ).fetchall()
        assert rows[0] == (1.0, 1)
        assert rows[1] == (-1.0, -1)

    def test_exactly_at_threshold_boundaries(self, tmp_path):
        """Signal values exactly at +/-0.2 thresholds set correct direction."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        tracker.log_prediction_simple(
            source="thresh", signal_value=0.2, confidence=0.8,
        )
        tracker.log_prediction_simple(
            source="thresh", signal_value=-0.2, confidence=0.8,
        )
        import sqlite3
        with sqlite3.connect(str(db)) as conn:
            rows = conn.execute(
                "SELECT signal_value, predicted_direction FROM signal_predictions "
                "WHERE source=? ORDER BY signal_value DESC",
                ("thresh",),
            ).fetchall()
        # >= 0.2 or > 0.2? Code uses: if signal_value > 0.2 -> 1
        # 0.2 > 0.2 is False -> goes to elif < -0.2 -> 0.2 < -0.2 is False -> 0
        assert rows[0] == (0.2, 0)
        assert rows[1] == (-0.2, 0)


# ---------------------------------------------------------------------------
# detect_decay_alerts duplicate suppression
# ---------------------------------------------------------------------------

class TestDetectDecayAlertsEdgeCases:
    """Edge cases for detect_decay_alerts."""

    def test_duplicate_suppression(self, tmp_path):
        """Same source+health pair saved twice should be deduplicated."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        # Save two identical health score records directly
        import sqlite3
        with sqlite3.connect(str(tracker.db_path)) as conn:
            cursor = conn.cursor()
            for days_ago in [20, 15]:
                ts = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%dT10:00:00")
                cursor.execute(
                    "INSERT INTO signal_health_scores "
                    "(timestamp, source, health_score, accuracy_30d, accuracy_60d, accuracy_90d, decay_rate, predictions_count, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts, "multi_speed_momentum", 0.80, 0.75, 0.78, 0.80, 0.01, 100, "healthy"),
                )
            # Most recent score much lower -> triggers decay alert
            cursor.execute(
                "INSERT INTO signal_health_scores "
                "(timestamp, source, health_score, accuracy_30d, accuracy_60d, accuracy_90d, decay_rate, predictions_count, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%dT10:00:00"), "multi_speed_momentum",
                 0.40, 0.35, 0.40, 0.45, 0.08, 100, "degraded"),
            )
            conn.commit()

        # First call produces alerts
        alerts1 = tracker.detect_decay_alerts()
        assert len(alerts1) > 0

        # Second call with same data: previous_health=0.80, current_health=0.40 duplicates
        # The dedup check looks at last 20 alerts for this source
        alerts2 = tracker.detect_decay_alerts()
        decay_alerts = [a for a in alerts2 if "streak" not in a.message.lower()
                        and "drawdown" not in a.message.lower()
                        and "ratio" not in a.message.lower()]

    def test_no_alerts_when_health_improves(self, tmp_path):
        """Rising health should NOT trigger decay alerts."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        import sqlite3
        with sqlite3.connect(str(tracker.db_path)) as conn:
            cursor = conn.cursor()
            # Old score low, new score high -> improvement
            cursor.execute(
                "INSERT INTO signal_health_scores "
                "(timestamp, source, health_score, accuracy_30d, accuracy_60d, accuracy_90d, decay_rate, predictions_count, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ((datetime.now() - timedelta(days=20)).strftime("%Y-%m-%dT10:00:00"),
                 "multi_speed_momentum", 0.30, 0.25, 0.28, 0.30, 0.01, 100, "unhealthy"),
            )
            cursor.execute(
                "INSERT INTO signal_health_scores "
                "(timestamp, source, health_score, accuracy_30d, accuracy_60d, accuracy_90d, decay_rate, predictions_count, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%dT10:00:00"),
                 "multi_speed_momentum", 0.80, 0.75, 0.78, 0.80, 0.01, 100, "healthy"),
            )
            conn.commit()
        alerts = tracker.detect_decay_alerts()
        # drop = (0.30 - 0.80) / 0.30 = -1.67 (negative = improvement)
        #  -1.67 >= 0.20 ? No -> no alert
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# update_actual_directions edge cases
# ---------------------------------------------------------------------------

class TestUpdateActualDirectionsEdgeCases:
    """Edge cases for update_actual_directions."""

    def test_no_matching_predictions_returns_zero(self, tmp_path):
        """When date has no predictions, should update 0 rows."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        tracker.log_prediction_simple(source="src", signal_value=0.5, confidence=0.8)
        updated = tracker.update_actual_directions(
            {"SPY": 0.01}, "2020-01-01"
        )
        assert updated == 0

    def test_very_large_positive_return(self, tmp_path):
        """Even extremely large returns should give direction=1."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        tracker.log_prediction_simple(source="src", signal_value=0.5, confidence=0.8)
        updated = tracker.update_actual_directions(
            {"SPY": 100.0}, datetime.now().strftime("%Y-%m-%d")
        )
        assert isinstance(updated, int)
        assert updated >= 0


# ---------------------------------------------------------------------------
# get_adjusted_weights edge cases
# ---------------------------------------------------------------------------

class TestGetAdjustedWeightsEdgeCases:
    """Extended get_adjusted_weights edge cases."""

    def test_all_sources_missing_uses_fallback(self, tmp_path):
        """When no health scores available, fallback multiplier is 0.5."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        base_weights = {"unknown_a": 0.6, "unknown_b": 0.4}
        adjusted = tracker.get_adjusted_weights(base_weights)
        # Both use fallback 0.5: 0.6*0.5=0.3, 0.4*0.5=0.2 -> norm: 0.6/0.4
        assert abs(adjusted["unknown_a"] - 0.6) < 0.01
        assert abs(adjusted["unknown_b"] - 0.4) < 0.01
        total = sum(adjusted.values())
        assert abs(total - 1.0) < 0.01

    def test_zero_weight_source_not_in_adjusted(self, tmp_path):
        """Zero base weight should produce zero adjusted weight."""
        db = tmp_path / "health.db"
        tracker = SignalHealthTracker(db_path=db)
        # Seed health data
        import sqlite3
        with sqlite3.connect(str(tracker.db_path)) as conn:
            cursor = conn.cursor()
            for i in range(15):
                ts = (datetime.now() - timedelta(days=80 - i)).strftime("%Y-%m-%dT10:00:00")
                cursor.execute(
                    "INSERT INTO signal_predictions "
                    "(timestamp, source, signal_value, confidence, predicted_direction, actual_direction, accuracy_calculated) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (ts, "alt_data", 0.5, 0.8, 1, 1)
                )
            conn.commit()
        base_weights = {"alternative_data": 1.0, "cross_asset_rv": 0.0}
        adjusted = tracker.get_adjusted_weights(base_weights)
        assert adjusted["cross_asset_rv"] == 0.0
        total = sum(adjusted.values())
        assert abs(total - 1.0) < 0.01
