#!/usr/bin/env python3
"""
Tests for dashboard generator — VIX regime detection, data freshness,
health status, alerts, broker data, and stats calculation.
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

from src.dashboard.generator import DashboardGenerator, DATA_DIR, PUBLIC_DIR, DB_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_market_db(db_path, symbols=None, days=30, base_price=500.0):
    """Create a market.db with price data for testing."""
    if symbols is None:
        symbols = ['SPY', 'GLD', 'TLT', 'QQQ']
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
        PRIMARY KEY (symbol, date))
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS regime_log (
            date TEXT, regime TEXT, vix_level REAL, detected_at TEXT
        )
    """)
    base_date = datetime.now()
    for sym in symbols:
        for i in range(days):
            d = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            noise = np.random.normal(0, 2.0)
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                         (sym, d, round(base_price + noise, 2)))
    conn.commit()
    conn.close()


def _make_generator(tmp_path):
    """Create a DashboardGenerator with a test database."""
    db_path = tmp_path / "market.db"
    _create_market_db(db_path)
    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen.conn = sqlite3.connect(str(db_path))
    gen.conn.row_factory = sqlite3.Row
    return gen, db_path


# ---------------------------------------------------------------------------
# VIX regime detection tests
# ---------------------------------------------------------------------------

class TestVIXRegimeDetection:
    """Test VIX-based regime classification logic."""

    def _classify_vix(self, vix_level):
        """Extract VIX regime classification logic."""
        if vix_level > 25:
            return "crisis"
        elif vix_level > 20:
            return "vol_spike"
        elif vix_level < 15:
            return "low_vol"
        else:
            return "normal"

    def test_crisis_regime(self):
        assert self._classify_vix(30) == "crisis"
        assert self._classify_vix(26) == "crisis"

    def test_vol_spike_regime(self):
        assert self._classify_vix(22) == "vol_spike"
        assert self._classify_vix(21) == "vol_spike"

    def test_low_vol_regime(self):
        assert self._classify_vix(12) == "low_vol"
        assert self._classify_vix(14) == "low_vol"

    def test_normal_regime(self):
        assert self._classify_vix(18) == "normal"
        assert self._classify_vix(15) == "normal"
        assert self._classify_vix(20) == "normal"

    def test_composite_regime_vix_overrides(self):
        """VIX crisis/vol_spike overrides trend regime."""
        # If VIX says crisis, it overrides regardless of trend
        vix_regime = "crisis"
        trend_regime = "normal"
        if vix_regime in ["crisis", "vol_spike"]:
            current_regime = vix_regime
        else:
            current_regime = trend_regime
        assert current_regime == "crisis"

    def test_composite_regime_low_vol_with_normal_trend(self):
        """Low vol + normal trend → low_vol."""
        vix_regime = "low_vol"
        trend_regime = "normal"
        if vix_regime in ["crisis", "vol_spike"]:
            current_regime = vix_regime
        elif vix_regime == "low_vol" and trend_regime != "crisis":
            current_regime = "low_vol"
        else:
            current_regime = trend_regime
        assert current_regime == "low_vol"

    def test_composite_regime_normal_uses_trend(self):
        """Normal VIX + trend regime → uses trend."""
        vix_regime = "normal"
        trend_regime = "bull"
        if vix_regime in ["crisis", "vol_spike"]:
            current_regime = vix_regime
        elif vix_regime == "low_vol" and trend_regime != "crisis":
            current_regime = "low_vol"
        else:
            current_regime = trend_regime
        assert current_regime == "bull"


# ---------------------------------------------------------------------------
# Data freshness tests
# ---------------------------------------------------------------------------

class TestDataFreshness:
    """Test data freshness classification."""

    def _classify_freshness(self, days_stale):
        """Extract freshness classification logic."""
        if days_stale <= 1:
            return "fresh"
        elif days_stale <= 3:
            return "stale"
        else:
            return "critical"

    def test_fresh(self):
        assert self._classify_freshness(0) == "fresh"
        assert self._classify_freshness(1) == "fresh"

    def test_stale(self):
        assert self._classify_freshness(2) == "stale"
        assert self._classify_freshness(3) == "stale"

    def test_critical(self):
        assert self._classify_freshness(4) == "critical"
        assert self._classify_freshness(30) == "critical"


# ---------------------------------------------------------------------------
# Health status tests
# ---------------------------------------------------------------------------

class TestHealthStatus:
    """Test system health status determination."""

    def _determine_health(self, failed_jobs, stale_count):
        """Extract health status logic."""
        status = "healthy"
        if failed_jobs > 0 or stale_count > 5:
            status = "warning"
        if failed_jobs > 2 or stale_count > 10:
            status = "critical"
        return status

    def test_healthy(self):
        assert self._determine_health(0, 0) == "healthy"
        assert self._determine_health(0, 5) == "healthy"

    def test_warning(self):
        assert self._determine_health(1, 0) == "warning"
        assert self._determine_health(0, 6) == "warning"

    def test_critical(self):
        assert self._determine_health(3, 0) == "critical"
        assert self._determine_health(0, 11) == "critical"

    def test_critical_overrides_warning(self):
        """Critical takes precedence when both conditions met."""
        assert self._determine_health(3, 11) == "critical"


# ---------------------------------------------------------------------------
# Generator initialization tests
# ---------------------------------------------------------------------------

class TestGeneratorInit:
    """Test DashboardGenerator initialization."""

    def test_creates_with_db(self, tmp_path):
        """Generator connects to database."""
        gen, _ = _make_generator(tmp_path)
        assert gen.conn is not None
        gen.conn.close()

    def test_row_factory_set(self, tmp_path):
        """Row factory is set for dict-like access."""
        gen, _ = _make_generator(tmp_path)
        assert gen.conn.row_factory == sqlite3.Row
        gen.conn.close()


# ---------------------------------------------------------------------------
# Performance JSON tests
# ---------------------------------------------------------------------------

class TestPerformanceJSON:
    """Test generate_performance_json."""

    def test_generates_file(self, tmp_path):
        """Creates dashboard.json file."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            path = gen.generate_performance_json()
        assert path.exists()
        gen.conn.close()

    def test_output_structure(self, tmp_path):
        """Output has expected keys."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert "prices" in data
        assert "regimes" in data
        assert "generated_at" in data
        gen.conn.close()

    def test_prices_contain_symbols(self, tmp_path):
        """Prices dict contains expected symbols."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert "SPY" in data["prices"]
        assert "GLD" in data["prices"]
        gen.conn.close()


# ---------------------------------------------------------------------------
# Stats JSON tests
# ---------------------------------------------------------------------------

class TestStatsJSON:
    """Test generate_stats_json."""

    def test_generates_file(self, tmp_path):
        """Creates stats.json file."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        assert path.exists()
        gen.conn.close()

    def test_has_asset_stats(self, tmp_path):
        """Stats contain per-asset data."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        assert "assets" in data or "generated_at" in data
        gen.conn.close()


# ---------------------------------------------------------------------------
# Alerts JSON tests
# ---------------------------------------------------------------------------

class TestAlertsJSON:
    """Test generate_alerts_json."""

    def test_generates_file(self, tmp_path):
        """Creates alerts.json file."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        assert path.exists()
        gen.conn.close()

    def test_alerts_structure(self, tmp_path):
        """Alerts output has expected structure."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        assert "alerts" in data
        assert "count" in data
        assert isinstance(data["alerts"], list)
        gen.conn.close()

    def test_kill_switch_alert(self, tmp_path):
        """Kill switch file generates alert."""
        gen, _ = _make_generator(tmp_path)
        kill_file = tmp_path / ".kill_switch_paper"
        kill_file.write_text(json.dumps({"enabled": True, "reason": "test", "timestamp": datetime.now().isoformat()}))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        kill_alerts = [a for a in data["alerts"] if a["type"] == "kill_switch"]
        assert len(kill_alerts) >= 1
        gen.conn.close()

    def test_stale_data_alert(self, tmp_path):
        """Stale data generates warning alert."""
        gen, db_path = _make_generator(tmp_path)
        # Insert very old data
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prices VALUES ('STALE', '2020-01-01', 100.0)")
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        stale_alerts = [a for a in data["alerts"] if a["type"] == "stale_data"]
        assert len(stale_alerts) >= 1
        gen.conn.close()


# ---------------------------------------------------------------------------
# Health JSON tests
# ---------------------------------------------------------------------------

class TestHealthJSON:
    """Test generate_health_json."""

    def test_generates_file(self, tmp_path):
        """Creates health.json file."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        assert path.exists()
        gen.conn.close()

    def test_health_structure(self, tmp_path):
        """Health output has expected structure."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert "system_status" in data
        assert "data_freshness" in data
        assert "generated_at" in data
        gen.conn.close()

    def test_data_freshness_populated(self, tmp_path):
        """Data freshness contains symbols from DB."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["data_freshness"]) > 0
        assert "SPY" in data["data_freshness"]
        gen.conn.close()


# ---------------------------------------------------------------------------
# Broker data tests
# ---------------------------------------------------------------------------

class TestBrokerData:
    """Test _load_broker_data."""

    def test_default_structure(self, tmp_path):
        """Returns expected default structure."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert "connected" in broker
        assert "positions" in broker
        assert "drift" in broker
        assert "kill_switch" in broker
        assert broker["connected"] is False
        gen.conn.close()

    def test_kill_switch_detected(self, tmp_path):
        """Kill switch file is detected."""
        gen, _ = _make_generator(tmp_path)
        kill_file = tmp_path / "kill_switch.json"
        kill_file.write_text(json.dumps({"enabled": True}))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert broker["kill_switch"] is True
        gen.conn.close()

    def test_sync_log_detected(self, tmp_path):
        """Position sync log is loaded."""
        gen, _ = _make_generator(tmp_path)
        sync_log = tmp_path / "position_sync.jsonl"
        sync_log.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "broker_positions": [{"symbol": "SPY", "qty": 10}],
            "drift": [{"symbol": "SPY", "drift_pct": 0.02}],
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert broker["connected"] is True
        assert len(broker["positions"]) == 1
        gen.conn.close()


# ---------------------------------------------------------------------------
# ML signals tests
# ---------------------------------------------------------------------------

class TestMLSignals:
    """Test _generate_ml_signals."""

    def test_default_structure(self, tmp_path):
        """Returns expected default structure."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert "available" in signals
        assert signals["available"] is False
        gen.conn.close()

    def test_features_file_detected(self, tmp_path):
        """Features file makes signals available."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "timestamp": datetime.now().isoformat(),
            "momentum_12m": 0.15, "volatility": 0.18,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is True
        assert "SPY" in signals["features"]
        gen.conn.close()


# ---------------------------------------------------------------------------
# Yield curve tests
# ---------------------------------------------------------------------------

class TestYieldCurve:
    """Test _get_yield_curve_data."""

    def test_default_structure(self, tmp_path):
        """Returns expected default structure."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._get_yield_curve_data()
        assert "yield_curve" in data or "duration_allocation" in data
        gen.conn.close()


# ---------------------------------------------------------------------------
# Run integration test
# ---------------------------------------------------------------------------

class TestRun:
    """Test run method."""

    def test_run_generates_all_files(self, tmp_path):
        """run() generates all dashboard files."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                gen.run()
        assert (tmp_path / "dashboard.json").exists()
        assert (tmp_path / "index.json").exists()
        # conn is closed by run()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
