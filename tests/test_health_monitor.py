#!/usr/bin/env python3
"""
Tests for signal health monitor — data classes, health scoring, status thresholds,
weight adjustment, health reports.
"""
import sys
import os
import json
import sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.signals.health_monitor import (
    HealthMetrics, HealthScore, HealthReport,
    SignalHealthMonitor, init_health_database,
    HEALTH_THRESHOLD_WARNING, HEALTH_THRESHOLD_CRITICAL,
    HEALTH_THRESHOLD_RECOVERY, DEFAULT_BASE_WEIGHTS,
    MONITORED_SOURCES, DB_PATH,
)


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestDataClasses:
    """Test dataclass serialization."""

    def test_health_metrics_to_dict(self):
        m = HealthMetrics(
            source="tsmom", timestamp=datetime.now().isoformat(),
            rolling_correlation_30d=0.45, win_rate_30d=0.62,
            current_volatility_regime="neutral", samples_30d=30,
        )
        d = m.to_dict()
        assert d["source"] == "tsmom"
        assert d["rolling_correlation_30d"] == 0.45
        assert d["win_rate_30d"] == 0.62

    def test_health_score_to_dict(self):
        s = HealthScore(
            source="tsmom", timestamp=datetime.now().isoformat(),
            correlation_score=0.7, accuracy_score=0.65, stability_score=0.6,
            overall=0.65, status="healthy", trend="stable",
        )
        d = s.to_dict()
        assert d["source"] == "tsmom"
        assert d["overall"] == 0.65
        assert d["status"] == "healthy"

    def test_health_report_to_dict(self):
        r = HealthReport(
            timestamp=datetime.now().isoformat(),
            composite_health=0.7,
            degraded_signals=["macro"],
            critical_signals=[],
        )
        d = r.to_dict()
        assert d["composite_health"] == 0.7
        assert "macro" in d["alerts"]["degraded"]
        assert d["summary"]["degraded"] == 1

    def test_health_report_defaults(self):
        r = HealthReport(timestamp=datetime.now().isoformat())
        assert r.composite_health == 0.5
        assert r.degraded_signals == []
        assert r.critical_signals == []


# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

class TestThresholds:
    """Test threshold constants are set."""

    def test_thresholds_ordered(self):
        assert HEALTH_THRESHOLD_CRITICAL < HEALTH_THRESHOLD_WARNING < HEALTH_THRESHOLD_RECOVERY

    def test_default_weights_sum_near_one(self):
        total = sum(DEFAULT_BASE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.05  # Allow small rounding

    def test_monitored_sources_exist(self):
        assert len(MONITORED_SOURCES) > 0
        assert "tsmom" in MONITORED_SOURCES
        assert "momentum" in MONITORED_SOURCES


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------

class TestDatabase:
    """Test health database initialization."""

    def test_init_creates_tables(self, tmp_path):
        with patch("src.signals.health_monitor.DB_PATH", tmp_path / "health.db"):
            init_health_database()
        conn = sqlite3.connect(str(tmp_path / "health.db"))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "signal_health" in tables


# ---------------------------------------------------------------------------
# Health score calculation tests
# ---------------------------------------------------------------------------

class TestHealthScoreCalculation:
    """Test calculate_health_score logic."""

    def _make_monitor(self, tmp_path):
        """Create a monitor with a test database."""
        db_path = tmp_path / "health.db"
        with patch("src.signals.health_monitor.DB_PATH", db_path):
            init_health_database()
        monitor = SignalHealthMonitor(db_path=db_path)
        return monitor

    def test_high_correlation_high_winrate_healthy(self, tmp_path):
        """High correlation + high win rate → healthy status."""
        monitor = self._make_monitor(tmp_path)
        metrics = HealthMetrics(
            source="tsmom", timestamp=datetime.now().isoformat(),
            rolling_correlation_30d=0.5, win_rate_30d=0.65,
            current_volatility_regime="neutral", regime_stability=0.8,
            samples_30d=30,
        )
        score = monitor.calculate_health_score("tsmom", metrics)
        assert score.status == "healthy"
        assert score.weight_multiplier == 1.0
        assert score.overall >= HEALTH_THRESHOLD_RECOVERY

    def test_low_correlation_critical(self, tmp_path):
        """Very low correlation + low win rate + high vol → critical status."""
        monitor = self._make_monitor(tmp_path)
        metrics = HealthMetrics(
            source="macro", timestamp=datetime.now().isoformat(),
            rolling_correlation_30d=-0.3, win_rate_30d=0.30,
            current_volatility_regime="high", regime_stability=0.3,
            samples_30d=30,
        )
        score = monitor.calculate_health_score("macro", metrics)
        assert score.status == "critical"
        assert score.weight_multiplier == 0.0

    def test_medium_correlation_degraded(self, tmp_path):
        """Low correlation + mediocre win rate → degraded status."""
        monitor = self._make_monitor(tmp_path)
        metrics = HealthMetrics(
            source="value", timestamp=datetime.now().isoformat(),
            rolling_correlation_30d=0.0, win_rate_30d=0.45,
            current_volatility_regime="high", regime_stability=0.5,
            samples_30d=30,
        )
        score = monitor.calculate_health_score("value", metrics)
        assert score.status == "degraded"

    def test_no_data_neutral(self, tmp_path):
        """No correlation data → neutral score near 0.5."""
        monitor = self._make_monitor(tmp_path)
        metrics = HealthMetrics(
            source="ai_agent", timestamp=datetime.now().isoformat(),
            rolling_correlation_30d=None, win_rate_30d=None,
            current_volatility_regime="neutral", regime_stability=1.0,
            samples_30d=0,
        )
        score = monitor.calculate_health_score("ai_agent", metrics)
        assert 0.45 <= score.overall <= 0.65

    def test_score_components(self, tmp_path):
        """Score has correlation, accuracy, and stability components."""
        monitor = self._make_monitor(tmp_path)
        metrics = HealthMetrics(
            source="tsmom", timestamp=datetime.now().isoformat(),
            rolling_correlation_30d=0.3, win_rate_30d=0.60,
            current_volatility_regime="low", regime_stability=0.9,
            samples_30d=30,
        )
        score = monitor.calculate_health_score("tsmom", metrics)
        assert 0.0 <= score.correlation_score <= 1.0
        assert 0.0 <= score.accuracy_score <= 1.0
        assert 0.0 <= score.stability_score <= 1.0

    def test_recovering_status(self, tmp_path):
        """Score between warning and recovery → recovering."""
        monitor = self._make_monitor(tmp_path)
        # Correlation 0.2 → corr_score = 0.5 + 0.2 = 0.7
        # Win rate 0.45 → acc_score = 0.45 * 1.5 - 0.25 = 0.425
        # Neutral regime → stab_score = 0.6 * 0.7 + 0.7 * 0.3 = 0.63
        # overall = 0.7*0.4 + 0.425*0.35 + 0.63*0.25 = 0.28 + 0.149 + 0.158 = 0.587
        metrics = HealthMetrics(
            source="sentiment", timestamp=datetime.now().isoformat(),
            rolling_correlation_30d=0.2, win_rate_30d=0.45,
            current_volatility_regime="neutral", regime_stability=0.7,
            samples_30d=30,
        )
        score = monitor.calculate_health_score("sentiment", metrics)
        # This should be between WARNING (0.5) and RECOVERY (0.6)
        assert score.status == "recovering"
        assert score.weight_multiplier == 0.75


# ---------------------------------------------------------------------------
# Weight adjustment tests
# ---------------------------------------------------------------------------

class TestWeightAdjustment:
    """Test calculate_adjusted_weights."""

    def _make_monitor_with_scores(self, tmp_path, source_scores):
        """Create monitor with pre-set health scores."""
        db_path = tmp_path / "health.db"
        with patch("src.signals.health_monitor.DB_PATH", db_path):
            init_health_database()
        monitor = SignalHealthMonitor(db_path=db_path)

        # Mock get_all_health_scores
        scores = {}
        for source, (overall, weight_mult) in source_scores.items():
            scores[source] = HealthScore(
                source=source, timestamp=datetime.now().isoformat(),
                overall=overall, weight_multiplier=weight_mult,
                status="healthy" if overall >= 0.6 else "degraded",
            )
        monitor.get_all_health_scores = lambda: scores
        return monitor

    def test_healthy_weights_unchanged(self, tmp_path):
        """All healthy → weights proportional to base."""
        monitor = self._make_monitor_with_scores(tmp_path, {
            "momentum": (0.8, 1.0),
            "value": (0.7, 1.0),
        })
        adjusted = monitor.calculate_adjusted_weights(
            {"momentum": 0.5, "value": 0.5}
        )
        assert adjusted["momentum"] == pytest.approx(0.5, abs=0.01)
        assert adjusted["value"] == pytest.approx(0.5, abs=0.01)

    def test_degraded_weight_reduced(self, tmp_path):
        """Degraded source → weight reduced."""
        monitor = self._make_monitor_with_scores(tmp_path, {
            "momentum": (0.8, 1.0),
            "value": (0.3, 0.0),  # critical → disabled
        })
        adjusted = monitor.calculate_adjusted_weights(
            {"momentum": 0.5, "value": 0.5}
        )
        # Value should have near-zero weight
        assert adjusted["value"] < 0.1
        assert adjusted["momentum"] > 0.9

    def test_adjusted_weights_normalize(self, tmp_path):
        """Adjusted weights always sum to 1.0."""
        monitor = self._make_monitor_with_scores(tmp_path, {
            "momentum": (0.8, 1.0),
            "value": (0.3, 0.0),
            "macro": (0.7, 1.0),
        })
        adjusted = monitor.calculate_adjusted_weights(
            {"momentum": 0.4, "value": 0.3, "macro": 0.3}
        )
        assert abs(sum(adjusted.values()) - 1.0) < 0.01

    def test_unknown_source_keeps_base_weight(self, tmp_path):
        """Source without health data → keeps base weight."""
        monitor = self._make_monitor_with_scores(tmp_path, {
            "momentum": (0.8, 1.0),
        })
        adjusted = monitor.calculate_adjusted_weights(
            {"momentum": 0.5, "unknown": 0.5}
        )
        assert "unknown" in adjusted
        assert adjusted["unknown"] > 0


# ---------------------------------------------------------------------------
# Health report tests
# ---------------------------------------------------------------------------

class TestHealthReport:
    """Test generate_health_report."""

    def _make_monitor(self, tmp_path):
        db_path = tmp_path / "health.db"
        with patch("src.signals.health_monitor.DB_PATH", db_path):
            init_health_database()
        return SignalHealthMonitor(db_path=db_path)

    def test_report_structure(self, tmp_path):
        """Report has expected structure."""
        monitor = self._make_monitor(tmp_path)
        monitor.get_all_health_scores = lambda: {}
        # Mock calculate_health_metrics to avoid signal_history table
        monitor.calculate_health_metrics = lambda src: HealthMetrics(
            source=src, timestamp=datetime.now().isoformat()
        )
        report = monitor.generate_health_report()
        assert isinstance(report, HealthReport)
        assert report.composite_health == 0.5

    def test_report_with_degraded_source(self, tmp_path):
        """Degraded source appears in degraded_signals list."""
        monitor = self._make_monitor(tmp_path)
        monitor.get_all_health_scores = lambda: {
            "macro": HealthScore(
                source="macro", timestamp=datetime.now().isoformat(),
                overall=0.35, status="degraded", weight_multiplier=0.5,
            ),
        }
        monitor.calculate_health_metrics = lambda src: HealthMetrics(
            source=src, timestamp=datetime.now().isoformat()
        )
        report = monitor.generate_health_report()
        assert "macro" in report.degraded_signals

    def test_report_with_critical_source(self, tmp_path):
        """Critical source appears in critical_signals list."""
        monitor = self._make_monitor(tmp_path)
        monitor.get_all_health_scores = lambda: {
            "value": HealthScore(
                source="value", timestamp=datetime.now().isoformat(),
                overall=0.2, status="critical", weight_multiplier=0.0,
            ),
        }
        monitor.calculate_health_metrics = lambda src: HealthMetrics(
            source=src, timestamp=datetime.now().isoformat()
        )
        report = monitor.generate_health_report()
        assert "value" in report.critical_signals


# ---------------------------------------------------------------------------
# Volatility regime tests
# ---------------------------------------------------------------------------

class TestVolatilityRegime:
    """Test detect_volatility_regime."""

    def _create_spy_db(self, tmp_path, daily_vol, num_days=90):
        """Create market.db with SPY price data at a given daily volatility.

        Args:
            daily_vol: daily return standard deviation (e.g. 0.005 for ~8% ann vol)
        """
        import numpy as np
        market_db = tmp_path / "market.db"
        conn = sqlite3.connect(str(market_db))
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        base_date = datetime.now()
        price = 500.0
        for i in range(num_days):
            d = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            ret = np.random.normal(0.0003, daily_vol)
            price *= (1 + ret)
            conn.execute("INSERT INTO prices VALUES ('SPY', ?, ?)", (d, round(price, 2)))
        conn.commit()
        conn.close()
        return market_db

    def test_low_volatility(self, tmp_path):
        """SPY with ~5% daily ann vol → low regime."""
        import numpy as np
        np.random.seed(42)
        market_db = self._create_spy_db(tmp_path, daily_vol=0.003)  # ~4.8% ann vol
        health_db = tmp_path / "health.db"
        with patch("src.signals.health_monitor.DB_PATH", health_db), \
             patch("src.signals.health_monitor.MARKET_DB_PATH", market_db):
            init_health_database()
            monitor = SignalHealthMonitor(db_path=health_db)
            regime, stability = monitor.detect_volatility_regime()
        assert regime == "low"

    def test_neutral_volatility(self, tmp_path):
        """SPY with ~15% daily ann vol → neutral regime."""
        import numpy as np
        np.random.seed(42)
        market_db = self._create_spy_db(tmp_path, daily_vol=0.0095)  # ~15% ann vol
        health_db = tmp_path / "health.db"
        with patch("src.signals.health_monitor.DB_PATH", health_db), \
             patch("src.signals.health_monitor.MARKET_DB_PATH", market_db):
            init_health_database()
            monitor = SignalHealthMonitor(db_path=health_db)
            regime, stability = monitor.detect_volatility_regime()
        assert regime == "neutral"

    def test_high_volatility(self, tmp_path):
        """SPY with ~25% daily ann vol → high regime."""
        import numpy as np
        np.random.seed(42)
        market_db = self._create_spy_db(tmp_path, daily_vol=0.016)  # ~25% ann vol
        health_db = tmp_path / "health.db"
        with patch("src.signals.health_monitor.DB_PATH", health_db), \
             patch("src.signals.health_monitor.MARKET_DB_PATH", market_db):
            init_health_database()
            monitor = SignalHealthMonitor(db_path=health_db)
            regime, stability = monitor.detect_volatility_regime()
        assert regime == "high"

    def test_no_spy_data(self, tmp_path):
        """No SPY data → neutral regime."""
        market_db = tmp_path / "market.db"
        conn = sqlite3.connect(str(market_db))
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        conn.commit()
        conn.close()

        health_db = tmp_path / "health.db"
        with patch("src.signals.health_monitor.DB_PATH", health_db), \
             patch("src.signals.health_monitor.MARKET_DB_PATH", market_db):
            init_health_database()
            monitor = SignalHealthMonitor(db_path=health_db)
            regime, stability = monitor.detect_volatility_regime()
        assert regime == "neutral"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
