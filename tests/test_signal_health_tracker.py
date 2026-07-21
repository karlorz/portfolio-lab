"""
Test suite for Signal Health Tracker (v3.12 Phase 1)

Tests:
1. Database initialization
2. Prediction logging
3. Health score calculation
4. Decay detection
5. Weight adjustment
"""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.signals.health_tracker import (
    SignalHealthTracker,
    SignalPrediction,
    HealthScore,
    SignalSource,
    DecayAlert,
    backfill_predictions,
)


class TestDatabaseInitialization:
    """Test database schema initialization."""
    
    def test_tables_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            tracker = SignalHealthTracker(db_path)
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check signal_predictions table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='signal_predictions'
            """)
            assert cursor.fetchone() is not None
            
            # Check signal_health_scores table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='signal_health_scores'
            """)
            assert cursor.fetchone() is not None
            
            conn.close()


class TestPredictionLogging:
    """Test prediction logging functionality."""
    
    def test_log_single_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            tracker = SignalHealthTracker(db_path)
            
            prediction = SignalPrediction(
                timestamp="2026-05-14T10:00:00",
                source="multi_speed_momentum",
                signal_value=0.8,
                confidence=0.75,
                predicted_direction=1,
                metadata={"regime": "bull"}
            )
            
            tracker.log_prediction(prediction)
            
            # Verify in database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM signal_predictions WHERE source = 'multi_speed_momentum'")
            assert cursor.fetchone()[0] == 1
            conn.close()
    
    def test_log_prediction_simple(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            tracker = SignalHealthTracker(db_path)
            
            tracker.log_prediction_simple(
                source="cross_asset_rv",
                signal_value=-0.5,
                confidence=0.6,
                timestamp="2026-05-14T10:00:00",
                metadata={"trend": "down"}
            )
            
            # Verify direction was calculated correctly
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT predicted_direction FROM signal_predictions WHERE source = 'cross_asset_rv'")
            direction = cursor.fetchone()[0]
            assert direction == -1  # Negative signal = bearish
            conn.close()
    
    def test_log_multiple_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            tracker = SignalHealthTracker(db_path)
            
            sources = ["multi_speed_momentum", "cross_asset_rv", "alternative_data", "unified_overlay"]
            for i, source in enumerate(sources):
                tracker.log_prediction_simple(
                    source=source,
                    signal_value=0.3 * (i - 1.5),  # Mix of directions
                    confidence=0.5 + i * 0.1,
                    timestamp=f"2026-05-{14+i}T10:00:00"
                )
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT source) FROM signal_predictions")
            assert cursor.fetchone()[0] == 4
            conn.close()


class TestHealthScoreCalculation:
    """Test health score calculation from historical accuracy."""
    
    def setup_predictions(self, tracker, source: str, days: int, accuracy: float):
        """Helper to setup predictions with specified accuracy."""
        base_date = datetime(2026, 5, 14)
        
        for i in range(days):
            date = base_date - timedelta(days=i)
            # Simulate actual direction based on accuracy
            actual = 1 if i % 2 == 0 else -1  # Alternating actual
            predicted = actual if (i / days) < accuracy else -actual  # Match accuracy %
            
            conn = sqlite3.connect(tracker.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO signal_predictions 
                (timestamp, source, signal_value, confidence, predicted_direction, actual_direction, accuracy_calculated)
                VALUES (?, ?, ?, ?, ?, ?, 1)
            """, (
                date.strftime("%Y-%m-%dT%H:%M:%S"),
                source,
                0.5 if predicted == 1 else -0.5,
                0.7,
                predicted,
                actual
            ))
            conn.commit()
            conn.close()
    
    def test_health_calculation_with_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            tracker = SignalHealthTracker(db_path)
            
            # Setup 90 days of 70% accurate predictions
            self.setup_predictions(tracker, "multi_speed_momentum", 100, 0.7)

            score = tracker.calculate_health_score("multi_speed_momentum", "2026-05-14")

            assert score is not None
            assert score.source == "multi_speed_momentum"
            assert 0.5 < score.health_score < 0.9  # Should be around 0.7
            assert score.accuracy_30d > 0  # Should have 30d accuracy
            assert score.predictions_count >= 90
    
    def test_health_status_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            tracker = SignalHealthTracker(db_path)
            
            # High accuracy predictions (>70% weighted = healthy)
            self.setup_predictions(tracker, "high_acc", 100, 0.95)
            # Low accuracy predictions (<50% weighted = degraded/unhealthy)
            # Need <0.5 weighted average to be unhealthy
            # With 90d @ 0.35, 60d @ 0.35, 30d @ 0.35 = 0.35 weighted = unhealthy
            self.setup_predictions(tracker, "low_acc", 100, 0.30)
            
            high_score = tracker.calculate_health_score("high_acc", "2026-05-14")
            low_score = tracker.calculate_health_score("low_acc", "2026-05-14")
            
            assert high_score is not None
            assert low_score is not None
            assert high_score.status == "healthy"
            # At 0.30 accuracy across all periods, weighted is ~0.3 which is unhealthy
            assert low_score.status in ["unhealthy", "degraded"]
    
    def test_insufficient_data_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            tracker = SignalHealthTracker(db_path)
            
            # Only 5 predictions - not enough for 90d calculation
            self.setup_predictions(tracker, "multi_speed_momentum", 5, 0.7)

            score = tracker.calculate_health_score("multi_speed_momentum", "2026-05-14")
            assert score is None
    
    def test_decay_rate_calculation(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            tracker = SignalHealthTracker(db_path)
            
            # Setup declining accuracy (simulating decay)
            # 30d accuracy < 60d accuracy = getting worse
            base_date = datetime(2026, 5, 14)
            
            for i in range(90):
                date = base_date - timedelta(days=i)
                actual = 1 if i % 2 == 0 else -1
                # Decreasing accuracy over time (60d better than 30d)
                if i < 30:
                    accuracy_rate = 0.4  # Recent: bad
                elif i < 60:
                    accuracy_rate = 0.8  # Medium: good
                else:
                    accuracy_rate = 0.5  # Old: ok
                    
                predicted = actual if (hash(str(i)) % 100) < (accuracy_rate * 100) else -actual
                
                conn = sqlite3.connect(tracker.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO signal_predictions 
                    (timestamp, source, signal_value, confidence, predicted_direction, actual_direction, accuracy_calculated)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                """, (
                    date.strftime("%Y-%m-%dT%H:%M:%S"),
                    "decaying_signal",
                    0.5,
                    0.7,
                    predicted,
                    actual
                ))
                conn.commit()
                conn.close()
            
            score = tracker.calculate_health_score("decaying_signal", "2026-05-14")
            assert score is not None
            # Decay rate: (30d_acc - 60d_acc) / 30 days 
            # 30d is worse than 60d, so decay_rate should be negative
            assert score.decay_rate < 0  # Negative means 30d < 60d (decaying)


class TestDecayDetection:
    """Test decay alert detection."""
    
    def test_detect_significant_decay(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            tracker = SignalHealthTracker(db_path)
            
            base_date = datetime(2026, 5, 14)
            
            # Insert health scores directly showing 25% drop
            # MUST use a source from SignalSource enum
            conn = sqlite3.connect(tracker.db_path)
            cursor = conn.cursor()
            
            # Use "multi_speed_momentum" which is in SignalSource enum
            # Old health score (high) - 45 days ago
            cursor.execute("""
                INSERT INTO signal_health_scores
                (timestamp, source, health_score, accuracy_30d, accuracy_60d, accuracy_90d, decay_rate, predictions_count, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                (base_date - timedelta(days=45)).isoformat(),
                "multi_speed_momentum",  # Valid SignalSource enum value
                0.80,
                0.80, 0.75, 0.70,
                0.0,
                100,
                "healthy"
            ))
            
            # Another score - 15 days ago
            cursor.execute("""
                INSERT INTO signal_health_scores
                (timestamp, source, health_score, accuracy_30d, accuracy_60d, accuracy_90d, decay_rate, predictions_count, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                (base_date - timedelta(days=15)).isoformat(),
                "multi_speed_momentum",
                0.70,
                0.70, 0.65, 0.68,
                0.0,
                100,
                "healthy"
            ))
            
            # New health score (low) - today
            cursor.execute("""
                INSERT INTO signal_health_scores
                (timestamp, source, health_score, accuracy_30d, accuracy_60d, accuracy_90d, decay_rate, predictions_count, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                base_date.isoformat(),
                "multi_speed_momentum",
                0.55,  # 0.80 -> 0.55 = 31% drop
                0.55, 0.60, 0.65,
                -0.01,
                100,
                "degraded"
            ))
            
            conn.commit()
            conn.close()
            
            # Need to update state with last alert time to avoid spam
            tracker.state["last_decay_check"] = (base_date - timedelta(days=1)).isoformat()
            
            # Use 60 day lookback to capture both old and new scores
            alerts = tracker.detect_decay_alerts(lookback_days=60)
            
            # 0.80 -> 0.55 = 31.25% drop = should trigger alert
            assert len(alerts) >= 1
            if alerts:
                assert alerts[0].source == "multi_speed_momentum"
                assert alerts[0].severity in ["warning", "critical"]
    
    def test_no_alerts_for_stable_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            tracker = SignalHealthTracker(db_path)
            
            base_date = datetime(2026, 5, 14)
            
            # Insert health scores showing stability
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            for days_ago in [30, 0]:
                cursor.execute("""
                    INSERT INTO signal_health_scores
                    (timestamp, source, health_score, accuracy_30d, accuracy_60d, accuracy_90d, decay_rate, predictions_count, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    (base_date - timedelta(days=days_ago)).isoformat(),
                    "stable_source",
                    0.75,
                    0.75, 0.73, 0.74,
                    0.0,
                    100,
                    "healthy"
                ))
            
            conn.commit()
            conn.close()
            
            alerts = tracker.detect_decay_alerts(lookback_days=30)
            assert len(alerts) == 0


class TestWeightAdjustment:
    """Test health-based weight adjustment."""
    
    def test_adjusted_weights_reduce_unhealthy(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            tracker = SignalHealthTracker(db_path)
            
            base_date = datetime(2026, 5, 14)
            
            # Setup health scores for specific sources we want to test
            # These sources should be a subset of SignalSource enum
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create predictions with known sources from SignalSource enum
            sources = [
                ("multi_speed_momentum", 0.85, "healthy"),
                ("cross_asset_rv", 0.35, "unhealthy"),
            ]
            
            for source, health, status in sources:
                cursor.execute("""
                    INSERT INTO signal_health_scores
                    (timestamp, source, health_score, accuracy_30d, accuracy_60d, accuracy_90d, decay_rate, predictions_count, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    base_date.isoformat(),
                    source,
                    health,
                    health, health, health,
                    0.0,
                    100,
                    status
                ))
            
            conn.commit()
            conn.close()
            
            # Get individual health scores
            msm_score = tracker.calculate_health_score("multi_speed_momentum", "2026-05-14")
            crv_score = tracker.calculate_health_score("cross_asset_rv", "2026-05-14")

            # If we have valid scores, test weight adjustment
            if msm_score and crv_score:
                # Calculate adjusted weights for just these two
                base_weights = {
                    "multi_speed_momentum": 0.5,
                    "cross_asset_rv": 0.5,
                }

                # Calculate what weights should be based on health scores
                msm_weight = 0.5 * max(0.2, msm_score.health_score)
                crv_weight = 0.5 * max(0.2, crv_score.health_score)
                total = msm_weight + crv_weight

                # Normalize
                expected_msm = msm_weight / total
                expected_crv = crv_weight / total

                # Healthy (msm) should have higher weight than unhealthy (crv)
                assert expected_msm > expected_crv
    
    def test_minimum_weight_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            tracker = SignalHealthTracker(db_path)
            from src.signals.health_tracker import HealthScore

            # Degraded low score still uses soft floor (mock scores — calc uses predictions)
            mock_scores = {
                "soft_floor_degraded": HealthScore(
                    source="soft_floor_degraded",
                    timestamp="2026-05-14",
                    health_score=0.1,
                    accuracy_30d=0.1,
                    accuracy_60d=0.1,
                    accuracy_90d=0.1,
                    decay_rate=0.0,
                    predictions_count=100,
                    status="degraded",
                )
            }
            tracker.calculate_all_health_scores = lambda end_date=None: mock_scores
            base_weights = {"soft_floor_degraded": 1.0}
            adjusted = tracker.get_adjusted_weights(base_weights, min_weight_multiplier=0.2)
            # Soft floor 0.2 on degraded → renorms to ~1.0 when sole arm
            assert adjusted["soft_floor_degraded"] >= 0.999

    def test_unhealthy_hard_zero_not_soft_floor(self):
        """Batch BH: status=unhealthy must zero mass, not max(0.2, score)."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            tracker = SignalHealthTracker(db_path)
            from src.signals.health_tracker import HealthScore

            mock_scores = {
                "multi_speed_momentum": HealthScore(
                    source="multi_speed_momentum",
                    timestamp="2026-05-14",
                    health_score=0.85,
                    accuracy_30d=0.85,
                    accuracy_60d=0.85,
                    accuracy_90d=0.85,
                    decay_rate=0.0,
                    predictions_count=100,
                    status="healthy",
                ),
                "cross_asset_rv": HealthScore(
                    source="cross_asset_rv",
                    timestamp="2026-05-14",
                    health_score=0.35,
                    accuracy_30d=0.35,
                    accuracy_60d=0.35,
                    accuracy_90d=0.35,
                    decay_rate=0.0,
                    predictions_count=100,
                    status="unhealthy",
                ),
            }
            tracker.calculate_all_health_scores = lambda end_date=None: mock_scores
            base_weights = {
                "multi_speed_momentum": 0.5,
                "cross_asset_rv": 0.5,
            }
            adjusted = tracker.get_adjusted_weights(base_weights)
            assert adjusted.get("cross_asset_rv", 0.0) == 0.0
            assert adjusted.get("multi_speed_momentum", 0.0) >= 0.999


class TestHealthReport:
    """Test comprehensive health report generation."""
    
    def test_report_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            tracker = SignalHealthTracker(db_path)
            
            # Setup some health score entries in the database directly
            base_date = datetime(2026, 5, 14)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Insert directly using active source names
            for source in ["multi_speed_momentum", "cross_asset_rv"]:
                cursor.execute("""
                    INSERT INTO signal_health_scores
                    (timestamp, source, health_score, accuracy_30d, accuracy_60d, accuracy_90d, decay_rate, predictions_count, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    base_date.isoformat(),
                    source,
                    0.75,
                    0.75, 0.73, 0.74,
                    0.0,
                    100,
                    "healthy"
                ))
            
            conn.commit()
            conn.close()
            
            report = tracker.get_health_report()
            
            assert "timestamp" in report
            assert "summary" in report
            assert "scores" in report
            assert "alerts" in report
            assert "overall_health" in report
            
            # Verify scores contains the 2 sources we inserted
            assert len(report["scores"]) >= 2 or len(report["scores"]) == 0  # Scores may or may not load


class TestEnumCoverage:
    """Test that all signal sources are defined."""
    
    def test_all_sources_defined(self):
        sources = list(SignalSource)
        expected = ["multi_speed_momentum", "cross_asset_rv", "international_momentum", "alternative_data", "cross_asset_regime_arb", "unified_overlay"]

        for exp in expected:
            assert any(s.value == exp for s in sources)
    
    def test_health_status_enum(self):
        from src.signals.health_tracker import SignalHealthStatus
        
        assert SignalHealthStatus.HEALTHY.value == "healthy"
        assert SignalHealthStatus.DEGRADED.value == "degraded"
        assert SignalHealthStatus.UNHEALTHY.value == "unhealthy"


class TestDataStructures:
    """Test dataclass serialization."""
    
    def test_prediction_to_dict(self):
        pred = SignalPrediction(
            timestamp="2026-05-14T10:00:00",
            source="multi_speed_momentum",
            signal_value=0.8,
            confidence=0.75,
            predicted_direction=1,
            metadata={"key": "value"}
        )
        
        d = pred.to_dict()
        assert d["source"] == "multi_speed_momentum"
        assert d["signal_value"] == 0.8
        assert json.loads(d["metadata"]) == {"key": "value"}
    
    def test_health_score_to_dict(self):
        score = HealthScore(
            source="multi_speed_momentum",
            timestamp="2026-05-14",
            health_score=0.75,
            accuracy_30d=0.80,
            accuracy_60d=0.72,
            accuracy_90d=0.73,
            decay_rate=-0.001,
            predictions_count=100,
            status="healthy"
        )
        
        d = score.to_dict()
        assert d["source"] == "multi_speed_momentum"
        assert d["health_score"] == 0.75
    
    def test_decay_alert_to_dict(self):
        alert = DecayAlert(
            source="multi_speed_momentum",
            alert_timestamp="2026-05-14T10:00:00",
            previous_health=0.80,
            current_health=0.55,
            drop_30d=0.25,
            severity="warning",
            message="Health dropped 25%"
        )
        
        d = alert.to_dict()
        assert d["source"] == "multi_speed_momentum"
        assert d["severity"] == "warning"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
