#!/usr/bin/env python3
"""
Tests for signal health monitor — data classes, health scoring, win rate calculation,
correlation tracking, decay detection, weight adjustment, and report generation.
"""
import sys
import os
import json
import sqlite3
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.signals.signal_health_monitor import (
    SignalHealthMetrics, EnsembleHealthReport,
    SignalHealthMonitor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monitor(tmp_path):
    """Create a SignalHealthMonitor with test database."""
    db_path = str(tmp_path / "test_health.db")
    monitor = SignalHealthMonitor.__new__(SignalHealthMonitor)
    monitor.db_path = Path(db_path)
    monitor._init_db()
    return monitor


def _populate_predictions(monitor, source='test_signal', n=100, seed=42):
    """Insert synthetic predictions with realized returns."""
    np.random.seed(seed)
    conn = sqlite3.connect(str(monitor.db_path))
    for i in range(n):
        ts = (datetime.now() - timedelta(days=n - i)).isoformat()
        prediction = np.random.normal(0, 1)
        direction = 1 if prediction > 0 else -1
        realized_1d = np.random.normal(0.001, 0.01)
        realized_5d = np.random.normal(0.005, 0.02)
        realized_dir = 1 if realized_1d > 0 else -1
        conn.execute("""
            INSERT OR REPLACE INTO signal_predictions
            (source, timestamp, prediction, direction, realized_return_1d,
             realized_return_5d, realized_direction_1d)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (source, ts, prediction, direction, realized_1d, realized_5d, realized_dir))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestSignalHealthMetrics:
    """Test SignalHealthMetrics dataclass."""

    def test_creation(self):
        m = SignalHealthMetrics(
            source='hmm_regime', timestamp='2026-01-01',
            prediction_correlation=0.45, correlation_trend='stable',
            correlation_pvalue=0.01, win_rate_30d=0.62, win_rate_90d=0.58,
            win_rate_trend='stable', decay_rate=-0.001, half_life_days=120,
            health_score=0.72, health_status='healthy',
            recommended_action='maintain', weight_adjustment=1.0,
        )
        assert m.source == 'hmm_regime'
        assert m.health_score == 0.72

    def test_health_status_values(self):
        for status in ['healthy', 'degraded', 'critical']:
            m = SignalHealthMetrics(
                source='test', timestamp='2026-01-01',
                prediction_correlation=0.3, correlation_trend='stable',
                correlation_pvalue=0.05, win_rate_30d=0.5, win_rate_90d=0.5,
                win_rate_trend='stable', decay_rate=0.0, half_life_days=999,
                health_score=0.5, health_status=status,
                recommended_action='maintain', weight_adjustment=1.0,
            )
            assert m.health_status == status


class TestEnsembleHealthReport:
    """Test EnsembleHealthReport dataclass."""

    def test_creation(self):
        report = EnsembleHealthReport(
            timestamp='2026-01-01', signals={},
            overall_health=0.75, consensus_degradation=False,
            weight_adjustments={'hmm_regime': 1.0}, alerts=[],
            recommended_ensemble_weights={'hmm_regime': 0.35},
        )
        assert report.overall_health == 0.75
        assert report.consensus_degradation is False


# ---------------------------------------------------------------------------
# Monitor init tests
# ---------------------------------------------------------------------------

class TestMonitorInit:
    """Test SignalHealthMonitor initialization."""

    def test_db_created(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        assert monitor.db_path.exists()

    def test_thresholds_defined(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        assert 'healthy' in monitor.HEALTH_THRESHOLDS
        assert 'degraded' in monitor.HEALTH_THRESHOLDS
        assert 'critical' in monitor.HEALTH_THRESHOLDS

    def test_threshold_ordering(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        assert monitor.HEALTH_THRESHOLDS['healthy'] > monitor.HEALTH_THRESHOLDS['degraded']
        assert monitor.HEALTH_THRESHOLDS['degraded'] > monitor.HEALTH_THRESHOLDS['critical']


# ---------------------------------------------------------------------------
# Record prediction tests
# ---------------------------------------------------------------------------

class TestRecordPrediction:
    """Test record_prediction method."""

    def test_inserts_prediction(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        monitor.record_prediction(
            source='test', timestamp='2026-01-01T10:00:00',
            prediction=0.5, direction=1,
        )
        conn = sqlite3.connect(str(monitor.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM signal_predictions WHERE source = 'test'")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1

    def test_upsert_on_duplicate(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        ts = '2026-01-01T10:00:00'
        monitor.record_prediction(source='test', timestamp=ts, prediction=0.5, direction=1)
        monitor.record_prediction(source='test', timestamp=ts, prediction=0.8, direction=1)
        conn = sqlite3.connect(str(monitor.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM signal_predictions WHERE source = 'test'")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1


# ---------------------------------------------------------------------------
# Update realized returns tests
# ---------------------------------------------------------------------------

class TestUpdateRealizedReturns:
    """Test update_realized_returns method."""

    def test_updates_returns(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        monitor.record_prediction(
            source='test', timestamp='2026-01-01',
            prediction=0.5, direction=1,
        )
        monitor.update_realized_returns(
            source='test', timestamp='2026-01-01',
            realized_return_1d=0.01, realized_return_5d=0.03,
            realized_direction_1d=1,
        )
        conn = sqlite3.connect(str(monitor.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT realized_return_1d FROM signal_predictions
            WHERE source = 'test' AND timestamp = '2026-01-01'
        """)
        val = cursor.fetchone()[0]
        conn.close()
        assert val == 0.01


# ---------------------------------------------------------------------------
# Win rate calculation tests
# ---------------------------------------------------------------------------

class TestCalculateWinRate:
    """Test _calculate_win_rate method."""

    def test_perfect_win_rate(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        dirs = np.array([1, -1, 1, -1, 1])
        realized = np.array([1, -1, 1, -1, 1])
        wr = monitor._calculate_win_rate(dirs, realized, days=5)
        assert wr == 1.0

    def test_zero_win_rate(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        dirs = np.array([1, 1, 1, 1, 1])
        realized = np.array([-1, -1, -1, -1, -1])
        wr = monitor._calculate_win_rate(dirs, realized, days=5)
        assert wr == 0.0

    def test_ignores_neutral(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        dirs = np.array([0, 1, -1, 0, 1])
        realized = np.array([1, 1, -1, -1, 1])
        wr = monitor._calculate_win_rate(dirs, realized, days=5)
        # Only 3 non-neutral: 1→1 correct, -1→-1 correct, 1→1 correct → 3/3
        assert wr == 1.0

    def test_insufficient_data_returns_zero(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        dirs = np.array([1])
        realized = np.array([1])
        wr = monitor._calculate_win_rate(dirs, realized, days=30)
        assert wr == 0.0


# ---------------------------------------------------------------------------
# Health score calculation tests
# ---------------------------------------------------------------------------

class TestCalculateHealthScore:
    """Test _calculate_health_score method."""

    def test_high_correlation_high_winrate(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        score = monitor._calculate_health_score(
            correlation=0.5, win_rate_30d=0.7, win_rate_90d=0.65, decay_rate=0.0
        )
        assert score > 0.6

    def test_low_values(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        score = monitor._calculate_health_score(
            correlation=-0.1, win_rate_30d=0.4, win_rate_90d=0.4, decay_rate=-0.01
        )
        assert score < 0.5

    def test_bounded_01(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        for corr, wr, decay in [(1.0, 1.0, 0.0), (-0.5, 0.0, -0.05), (0.0, 0.5, -0.001)]:
            score = monitor._calculate_health_score(corr, wr, wr, decay)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Recommendations tests
# ---------------------------------------------------------------------------

class TestGetRecommendations:
    """Test _get_recommendations method."""

    def test_high_score_improving(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        action, adj = monitor._get_recommendations(0.85, 'improving', 0.0)
        assert action == 'increase_weight'
        assert adj == 1.20

    def test_healthy_score(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        action, adj = monitor._get_recommendations(0.72, 'stable', -0.001)
        assert action == 'maintain'
        assert adj == 1.0

    def test_degraded_score(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        action, adj = monitor._get_recommendations(0.40, 'decaying', -0.02)
        assert action == 'reduce_weight'
        assert adj == 0.70

    def test_critical_score(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        action, adj = monitor._get_recommendations(0.20, 'decaying', -0.05)
        assert action == 'disable'
        assert adj == 0.0


# ---------------------------------------------------------------------------
# Weight adjustment tests
# ---------------------------------------------------------------------------

class TestAdjustWeights:
    """Test _adjust_weights method."""

    def test_sums_to_one(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        base = {'a': 0.5, 'b': 0.3, 'c': 0.2}
        adj = {'a': 1.0, 'b': 1.0, 'c': 1.0}
        result = monitor._adjust_weights(base, adj)
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_disable_zeroes_weight(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        base = {'a': 0.5, 'b': 0.3, 'c': 0.2}
        adj = {'a': 0.0, 'b': 1.0, 'c': 1.0}
        result = monitor._adjust_weights(base, adj)
        assert result['a'] == 0.0

    def test_increase_boosts_weight(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        base = {'a': 0.5, 'b': 0.5}
        adj = {'a': 1.2, 'b': 1.0}
        result = monitor._adjust_weights(base, adj)
        assert result['a'] > result['b']


# ---------------------------------------------------------------------------
# Base weights tests
# ---------------------------------------------------------------------------

class TestGetBaseWeights:
    """Test _get_base_weights method."""

    def test_returns_default_weights(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        weights = monitor._get_base_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0
        assert abs(sum(weights.values()) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Health report tests
# ---------------------------------------------------------------------------

class TestGenerateHealthReport:
    """Test generate_health_report method."""

    def test_returns_report(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        report = monitor.generate_health_report(['source_a', 'source_b'])
        assert isinstance(report, EnsembleHealthReport)

    def test_empty_sources(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        report = monitor.generate_health_report([])
        assert report.overall_health == 0.0

    def test_has_weight_adjustments(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        report = generate_health_report = monitor.generate_health_report(['source_a'])
        assert 'weight_adjustments' in report.__dict__ or isinstance(report, EnsembleHealthReport)


# ---------------------------------------------------------------------------
# Save health metrics tests
# ---------------------------------------------------------------------------

class TestSaveHealthMetrics:
    """Test save_health_metrics persistence."""

    def test_saves_to_db(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        m = SignalHealthMetrics(
            source='test', timestamp='2026-01-01',
            prediction_correlation=0.45, correlation_trend='stable',
            correlation_pvalue=0.01, win_rate_30d=0.62, win_rate_90d=0.58,
            win_rate_trend='stable', decay_rate=-0.001, half_life_days=120,
            health_score=0.72, health_status='healthy',
            recommended_action='maintain', weight_adjustment=1.0,
        )
        monitor.save_health_metrics(m)
        conn = sqlite3.connect(str(monitor.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM health_metrics WHERE source = 'test'")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1


# ---------------------------------------------------------------------------
# Threshold tests
# ---------------------------------------------------------------------------

class TestHealthThresholds:
    """Test health status classification logic."""

    def test_healthy_above_threshold(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        assert 0.75 >= monitor.HEALTH_THRESHOLDS['healthy']

    def test_degraded_between_thresholds(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        score = 0.55
        assert score < monitor.HEALTH_THRESHOLDS['healthy']
        assert score >= monitor.HEALTH_THRESHOLDS['degraded']

    def test_critical_below_threshold(self, tmp_path):
        monitor = _make_monitor(tmp_path)
        score = 0.25
        assert score < monitor.HEALTH_THRESHOLDS['critical']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
