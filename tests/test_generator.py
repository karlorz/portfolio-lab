#!/usr/bin/env python3
"""
Tests for dashboard generator — VIX regime detection, data freshness,
health status, alerts, broker data, and stats calculation.
"""
import json
import sqlite3
import numpy as np

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


# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------

class TestConstants:
    """Test module-level constants."""

    def test_base_allocation_has_symbols(self):
        """BASE_ALLOCATION contains SPY, GLD, TLT."""
        from src.paths import BASE_ALLOCATION
        assert "SPY" in BASE_ALLOCATION
        assert "GLD" in BASE_ALLOCATION
        assert "TLT" in BASE_ALLOCATION

    def test_public_dir_is_path_instance(self):
        """PUBLIC_DIR is a Path instance."""
        from src.dashboard.generator import PUBLIC_DIR
        assert isinstance(PUBLIC_DIR, Path)

    def test_base_allocation_weights_sum_to_one(self):
        """BASE_ALLOCATION weights sum to 1.0."""
        from src.paths import BASE_ALLOCATION
        total = sum(BASE_ALLOCATION.values())
        assert abs(total - 1.0) < 0.01


# ---------------------------------------------------------------------------
# GARCH-CVaR data edge cases
# ---------------------------------------------------------------------------

class TestGarchCvarData:
    """Test _load_garch_cvar_data edge cases."""

    def test_defaults_no_health_file(self, tmp_path):
        """Returns expected defaults when no health file exists."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == -0.0179
        assert data["cvar_95_garch"] == -0.0215
        assert data["var_95"] == -0.0127
        assert data["garch_active"] is True
        assert data["volatility_clustering"] == "elevated"
        gen.conn.close()

    def test_flat_format_normalizes_percentages(self, tmp_path):
        """Values >1 in health report are divided by 100."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "garch_filtered": True,
            "cvar_95": 2.5,
            "var_95": 1.8,
            "cvar_ratio": 1.5,
            "filter_active": True,
            "conditional_volatility_current": 1.2,
            "garch_persistence": 0.97,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == 0.025
        assert data["var_95"] == pytest.approx(0.018)
        assert data["garch_active"] is True
        gen.conn.close()

    def test_flat_format_keeps_decimal_values(self, tmp_path):
        """Values <=1 are kept as-is."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "garch_filtered": True,
            "cvar_95": -0.0179,
            "var_95": -0.0127,
            "cvar_ratio": 1.51,
            "filter_active": True,
            "conditional_volatility_current": 1.5,
            "garch_persistence": 0.88,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == -0.0179
        assert data["var_95"] == -0.0127
        gen.conn.close()

    def test_legacy_nested_format(self, tmp_path):
        """Parses legacy nested check format from health report."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "checks": {
                "cvar_metrics": {
                    "garch_filtered": True,
                    "cvar_95": -0.0250,
                    "var_95": -0.0150,
                    "cvar_ratio": 1.75,
                    "garch_active": True,
                }
            }
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == -0.0250
        assert data["cvar_ratio"] == 1.75
        gen.conn.close()

    def test_volatility_clustering_boundaries(self, tmp_path):
        """Tests all persistence thresholds for clustering label."""
        gen, _ = _make_generator(tmp_path)
        for persistence, expected in [(0.96, "high"), (0.90, "elevated"), (0.80, "normal")]:
            health_file = tmp_path / ".health_report.json"
            health_file.write_text(json.dumps({
                "garch_filtered": True,
                "cvar_95": -0.0179,
                "var_95": -0.0127,
                "cvar_ratio": 1.51,
                "filter_active": True,
                "conditional_volatility_current": 1.5,
                "garch_persistence": persistence,
            }))
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                data = gen._load_garch_cvar_data()
            assert data["volatility_clustering"] == expected, (
                f"Persistence {persistence} should be {expected}"
            )
        gen.conn.close()


# ---------------------------------------------------------------------------
# Entropy data edge cases
# ---------------------------------------------------------------------------

class TestEntropyData:
    """Test _load_entropy_data edge cases."""

    def test_defaults_no_health_file(self, tmp_path):
        """Returns expected defaults when no health file exists."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_entropy_data()
        assert data["shannon_entropy"] == 1.02
        assert data["effective_n"] == 2.77
        assert data["concentration_risk"] == "good"
        assert data["hhi_index"] == 0.38
        gen.conn.close()

    def test_concentration_risk_all_levels(self, tmp_path):
        """All score thresholds map to correct risk labels."""
        gen, _ = _make_generator(tmp_path)
        for score, expected in [(92, "good"), (80, "low"), (60, "medium"),
                                 (40, "high"), (20, "critical")]:
            health_file = tmp_path / ".health_report.json"
            health_file.write_text(json.dumps({
                "checks": {
                    "portfolio_entropy": {
                        "metrics": {
                            "shannon_entropy": 1.02,
                            "effective_n": 2.77,
                            "normalized_score": score,
                            "hhi_index": 0.38,
                        }
                    }
                }
            }))
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                data = gen._load_entropy_data()
            assert data["concentration_risk"] == expected, (
                f"Score {score} should be {expected}"
            )
        gen.conn.close()

    def test_loads_from_health_file_metrics(self, tmp_path):
        """All fields are populated from health file metrics."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "checks": {
                "portfolio_entropy": {
                    "metrics": {
                        "shannon_entropy": 0.85,
                        "effective_n": 2.1,
                        "normalized_score": 77.0,
                        "hhi_index": 0.45,
                    }
                }
            }
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_entropy_data()
        assert data["shannon_entropy"] == 0.85
        assert data["effective_n"] == 2.1
        assert data["normalized_score"] == 77.0
        assert data["hhi_index"] == 0.45
        gen.conn.close()

    def test_missing_metrics_section(self, tmp_path):
        """Missing metrics section returns defaults unchanged."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "checks": {
                "portfolio_entropy": {}
            }
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_entropy_data()
        assert data["shannon_entropy"] == 1.02
        assert data["effective_n"] == 2.77
        gen.conn.close()


# ---------------------------------------------------------------------------
# ML signals edge cases
# ---------------------------------------------------------------------------

class TestMlSignalsEdgeCases:
    """Test _generate_ml_signals edge cases."""

    def test_no_features_file(self, tmp_path):
        """Returns available=False when features file does not exist."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is False
        assert "features" in signals
        assert "predictions" in signals
        gen.conn.close()

    def test_empty_features_file(self, tmp_path):
        """Empty features file returns available=False."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text("")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is False
        gen.conn.close()

    def test_malformed_line_skipped(self, tmp_path):
        """Malformed JSON line in features file is skipped."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(
            "not valid json\n"
            + json.dumps({"symbol": "SPY", "vix_level": 15, "timestamp": "2026-01-01"})
            + "\n"
        )
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is True
        assert "SPY" in signals["features"]
        gen.conn.close()

    def test_vix_crisis_predictions(self, tmp_path):
        """VIX >25 yields bearish prediction."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 30, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["predicted_regime"] == "bear"
        assert pred["confidence"] == 0.5
        assert pred["probabilities"]["bear"] == 0.5
        assert pred["probabilities"]["bull"] == 0.2
        gen.conn.close()

    def test_trend_bull_predictions(self, tmp_path):
        """Positive trend and price above SMA yields bullish prediction."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 15, "trend_direction": 1,
            "price_vs_sma20": 0.05, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["predicted_regime"] == "bull"
        assert pred["probabilities"]["bull"] == 0.6
        gen.conn.close()

    def test_trend_bear_predictions(self, tmp_path):
        """Negative trend yields bearish-leaning prediction."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 15, "trend_direction": -1,
            "price_vs_sma20": -0.03, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["probabilities"]["bear"] == 0.4
        assert pred["probabilities"]["neutral"] == 0.4
        assert pred["probabilities"]["bull"] == 0.2
        gen.conn.close()

    def test_default_predictions(self, tmp_path):
        """Default probabilities when vix <=20 and no trend signal."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 18, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["predicted_regime"] == "neutral"
        assert pred["probabilities"]["bear"] == 0.2
        assert pred["probabilities"]["neutral"] == 0.6
        assert pred["probabilities"]["bull"] == 0.2
        gen.conn.close()

    def test_grid_search_results(self, tmp_path):
        """Grid search results loaded from JSONL file."""
        gen, _ = _make_generator(tmp_path)
        grid_file = tmp_path / "grid_search_results.jsonl"
        grid_file.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:00",
            "allocations": {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2},
            "sharpe": 0.85,
            "volatility": 0.12,
        }) + "\n")
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 15, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        gs = signals["grid_search"]
        assert gs["available"] is True
        assert gs["sharpe"] == 0.85
        assert gs["top_allocation"] == {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2}
        gen.conn.close()

    def test_multi_symbol_features(self, tmp_path):
        """Multiple symbols produce separate predictions."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(
            json.dumps({"symbol": "SPY", "vix_level": 30, "trend_direction": 0,
                        "price_vs_sma20": 0, "timestamp": "2026-01-01T00:00:00"})
            + "\n"
            + json.dumps({"symbol": "GLD", "vix_level": 15, "trend_direction": 1,
                          "price_vs_sma20": 0.02, "timestamp": "2026-01-01T00:00:00"})
            + "\n"
        )
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert "SPY" in signals["predictions"]
        assert "GLD" in signals["predictions"]
        assert signals["predictions"]["SPY"]["predicted_regime"] == "bear"
        assert signals["predictions"]["GLD"]["predicted_regime"] == "bull"
        gen.conn.close()


# ---------------------------------------------------------------------------
# Yield curve edge cases
# ---------------------------------------------------------------------------

class TestYieldCurveEdgeCases:
    """Test _get_yield_curve_data edge cases."""

    def test_no_file_returns_empty(self, tmp_path):
        """Returns empty result when yields file does not exist."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.YIELDS_JSON", tmp_path / "nonexistent.json"):
            data = gen._get_yield_curve_data()
        assert data["yield_curve"] is None
        assert data["duration_allocation"] is None
        gen.conn.close()

    def test_empty_list_returns_empty(self, tmp_path):
        """Empty yields list returns empty result."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text("[]")
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        assert data["yield_curve"] is None
        gen.conn.close()

    def test_spread_classification_steep(self, tmp_path):
        """Spread > 100 bps classifies as steep."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_data = [{"spread2s10s": 150, "dgs2": 4.0, "dgs10": 5.5} for _ in range(35)]
        yields_path.write_text(json.dumps(yields_data))
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        assert data["yield_curve"]["duration_regime"] == "steep"
        gen.conn.close()

    def test_spread_classification_inverted(self, tmp_path):
        """Spread <= 0 bps classifies as inverted."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_data = [{"spread2s10s": -25, "dgs2": 5.0, "dgs10": 4.75} for _ in range(35)]
        yields_path.write_text(json.dumps(yields_data))
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        assert data["yield_curve"]["duration_regime"] == "inverted"
        gen.conn.close()

    def test_spread_boundary_values(self, tmp_path):
        """Boundary spread values map to correct regimes."""
        gen, _ = _make_generator(tmp_path)
        cases = [(100, "normal"), (50, "flat"), (1, "flat"), (0, "inverted")]
        for spread, expected in cases:
            yields_path = tmp_path / "yields.json"
            yields_data = [{"spread2s10s": spread, "dgs2": 4.0, "dgs10": 5.0} for _ in range(35)]
            yields_path.write_text(json.dumps(yields_data))
            with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                data = gen._get_yield_curve_data()
            assert data["yield_curve"]["duration_regime"] == expected, (
                f"Spread {spread} should be {expected}, got {data['yield_curve']['duration_regime']}"
            )
        gen.conn.close()

    def test_duration_allocation_by_regime(self, tmp_path):
        """Each regime maps to correct duration allocation."""
        gen, _ = _make_generator(tmp_path)
        allocations = {
            "steep": {"tlt": 0.70, "ief": 0.25, "shy": 0.05, "bil": 0.00},
            "inverted": {"tlt": 0.15, "ief": 0.25, "shy": 0.35, "bil": 0.25},
        }
        for regime, spread_val in [("steep", 150), ("inverted", -25)]:
            yields_path = tmp_path / "yields.json"
            yields_data = [{"spread2s10s": spread_val, "dgs2": 4.0, "dgs10": 5.0} for _ in range(35)]
            yields_path.write_text(json.dumps(yields_data))
            with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                data = gen._get_yield_curve_data()
            expected = allocations[regime]
            for k, v in expected.items():
                assert data["duration_allocation"][k] == v, (
                    f"Regime {regime}: {k} expected {v}, got {data['duration_allocation'][k]}"
                )
        gen.conn.close()

    def test_spread_history_length(self, tmp_path):
        """Spread history contains up to 30 entries."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_data = [{"spread2s10s": i * 5, "dgs2": 4.0, "dgs10": 5.0} for i in range(40)]
        yields_path.write_text(json.dumps(yields_data))
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        assert len(data["yield_curve"]["spread_history"]) == 30
        gen.conn.close()


# ---------------------------------------------------------------------------
# Broker data edge cases
# ---------------------------------------------------------------------------

class TestBrokerDataEdgeCases:
    """Test _load_broker_data edge cases."""

    def test_empty_sync_log(self, tmp_path):
        """Empty sync log file returns default structure."""
        gen, _ = _make_generator(tmp_path)
        sync_log = tmp_path / "position_sync.jsonl"
        sync_log.write_text("")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert broker["connected"] is False
        gen.conn.close()

    def test_malformed_sync_log(self, tmp_path):
        """Malformed JSON in sync log is caught gracefully."""
        gen, _ = _make_generator(tmp_path)
        sync_log = tmp_path / "position_sync.jsonl"
        sync_log.write_text("not valid json\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        # Exception is caught, broker stays in default state
        assert broker["connected"] is False
        gen.conn.close()

    def test_kill_switch_disabled(self, tmp_path):
        """Kill switch with enabled=False reports no kill switch."""
        gen, _ = _make_generator(tmp_path)
        kill_file = tmp_path / "kill_switch.json"
        kill_file.write_text(json.dumps({"enabled": False}))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert broker["kill_switch"] is False
        gen.conn.close()

    def test_broker_orders_loaded(self, tmp_path):
        """Broker orders log is loaded into recent_orders."""
        gen, _ = _make_generator(tmp_path)
        orders_log = tmp_path / "broker_orders.jsonl"
        orders_log.write_text(
            json.dumps({"order_id": 1, "symbol": "SPY", "side": "buy", "qty": 10})
            + "\n"
            + json.dumps({"order_id": 2, "symbol": "GLD", "side": "sell", "qty": 5})
            + "\n"
        )
        # Also need sync log so connected=True
        sync_log = tmp_path / "position_sync.jsonl"
        sync_log.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:00",
            "broker_positions": [],
            "drift": [],
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert len(broker["recent_orders"]) == 2
        assert broker["recent_orders"][0]["order_id"] == 2  # Reversed order
        gen.conn.close()

    def test_malformed_orders_line(self, tmp_path):
        """Malformed line in broker orders is caught gracefully."""
        gen, _ = _make_generator(tmp_path)
        orders_log = tmp_path / "broker_orders.jsonl"
        orders_log.write_text("not valid json\n")
        # Also need sync log
        sync_log = tmp_path / "position_sync.jsonl"
        sync_log.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:00",
            "broker_positions": [],
            "drift": [],
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        # Exception is caught, default recent_orders returned
        assert broker["recent_orders"] == []
        gen.conn.close()

    def test_malformed_kill_switch(self, tmp_path):
        """Malformed kill_switch.json returns default state."""
        gen, _ = _make_generator(tmp_path)
        kill_file = tmp_path / "kill_switch.json"
        kill_file.write_text("not valid json")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert broker["kill_switch"] is False
        gen.conn.close()


# ---------------------------------------------------------------------------
# Performance JSON edge cases
# ---------------------------------------------------------------------------

class TestPerformanceJSONEdgeCases:
    """Test generate_performance_json edge cases."""

    def test_no_perf_log_empty_paper_portfolio(self, tmp_path):
        """No performance.jsonl gives empty paper_portfolio."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert data["paper_portfolio"] == []
        gen.conn.close()

    def test_malformed_perf_entry_skipped(self, tmp_path):
        """Malformed entries in performance.jsonl are skipped."""
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        perf_log.write_text(
            "not valid json\n"
            + json.dumps({"timestamp": "2026-01-01", "total_value": 100000, "daily_return": 0.01})
            + "\n"
        )
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["paper_portfolio"]) == 1
        gen.conn.close()

    def test_prices_contain_correct_keys(self, tmp_path):
        """Each price entry has d and p keys."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        for sym, entries in data["prices"].items():
            assert len(entries) > 0
            assert "d" in entries[0]
            assert "p" in entries[0]
        gen.conn.close()

    def test_generated_at_isoformat(self, tmp_path):
        """generated_at is a valid ISO format string."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        # Verify it parses
        dt = datetime.fromisoformat(data["generated_at"])
        assert isinstance(dt, datetime)
        gen.conn.close()


# ---------------------------------------------------------------------------
# Signals JSON edge cases
# ---------------------------------------------------------------------------

class TestSignalsJSONEdgeCases:
    """Test generate_signals_json edge cases."""

    def test_missing_vix_handled(self, tmp_path):
        """Missing VIX symbol defaults vix to None and falls back to trend."""
        gen, _ = _make_generator(tmp_path)
        gen.conn.execute("DELETE FROM prices WHERE symbol = '^VIX'")
        gen.conn.commit()
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps([{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert data["regime"]["vix"] is None
        assert "regime" in data["regime"]
        gen.conn.close()

    def test_output_structure(self, tmp_path):
        """signals.json contains all expected top-level keys."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps([{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        required_keys = {"timestamp", "regime", "target_allocations", "current_positions",
                         "cash", "total_value", "latest_prices", "ml_signals",
                         "yield_curve", "broker"}
        assert required_keys.issubset(set(data.keys()))
        gen.conn.close()

    def test_default_values_when_no_state(self, tmp_path):
        """No portfolio_paper.json uses default cash and total value."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps([{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert data["cash"] == 100000.0
        assert data["total_value"] == 100000.0
        gen.conn.close()


# ---------------------------------------------------------------------------
# Health JSON edge cases
# ---------------------------------------------------------------------------

class TestHealthJSONEdgeCases:
    """Test generate_health_json edge cases."""

    def test_cron_fallback(self, tmp_path):
        """No cron_status.json uses fallback cron jobs."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["cron_jobs"]) == 7
        assert all(j["status"] == "unknown" for j in data["cron_jobs"])
        gen.conn.close()

    def test_cron_error_degraded(self, tmp_path):
        """Corrupted cron_status.json returns degraded status."""
        gen, _ = _make_generator(tmp_path)
        cron_file = tmp_path / "cron_status.json"
        cron_file.write_text("not valid json")
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert data["system_status"] == "degraded"
        gen.conn.close()

    def test_stale_data_warning_threshold(self, tmp_path):
        """stale_count > 5 sets system_status to warning."""
        gen, db_path = _make_generator(tmp_path)
        # Add 6 stale symbols to exceed warning threshold (>5) but not critical (>10)
        conn = sqlite3.connect(str(db_path))
        for i in range(6):
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                         (f"STALE{i}", "2020-01-01", 100.0))
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        # 6 stale + 4 fresh = 10 data_freshness entries, stale_count = 6
        assert data["system_status"] == "warning"
        gen.conn.close()

    def test_stale_data_critical_threshold(self, tmp_path):
        """stale_count > 10 sets system_status to critical."""
        gen, db_path = _make_generator(tmp_path)
        # Add 11 stale symbols to exceed critical threshold (>10)
        conn = sqlite3.connect(str(db_path))
        for i in range(11):
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                         (f"OLD{i}", "2020-01-01", 100.0))
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert data["system_status"] == "critical"
        gen.conn.close()

    def test_healthy_when_all_fresh(self, tmp_path):
        """Fresh data and no cron errors gives healthy status."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert data["system_status"] == "healthy"
        assert len(data["data_freshness"]) == 4
        gen.conn.close()


# ---------------------------------------------------------------------------
# Stats JSON edge cases
# ---------------------------------------------------------------------------

class TestStatsJSONEdgeCases:
    """Test generate_stats_json edge cases."""

    def test_no_perf_log_returns_basic_stats(self, tmp_path):
        """No performance.jsonl returns asset stats without paper metrics."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        assert "asset_stats" in data
        assert len(data["asset_stats"]) > 0
        assert data["paper_portfolio"] == {}
        assert data["spy_comparison"] is None
        gen.conn.close()

    def test_stats_have_expected_fields(self, tmp_path):
        """Each asset stat has 30d_return, volatility, and current."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        for sym, stat in data["asset_stats"].items():
            assert "30d_return" in stat
            assert "volatility" in stat
            assert "current" in stat
        gen.conn.close()

    def test_single_price_point_returns_empty_stats(self, tmp_path):
        """Only one price entry per symbol produces empty stats."""
        gen, db_path = _make_generator(tmp_path)
        # Create new DB with single price point
        conn = sqlite3.connect(str(db_path))
        conn.execute("DROP TABLE IF EXISTS prices")
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        today = datetime.now().strftime("%Y-%m-%d")
        for sym in ["SPY", "GLD"]:
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                         (sym, today, 100.0))
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        # Both symbols have < 2 prices, so no stats generated
        assert len(data["asset_stats"]) == 0
        gen.conn.close()


# ---------------------------------------------------------------------------
# Alerts JSON edge cases
# ---------------------------------------------------------------------------

class TestAlertsJSONEdgeCases:
    """Test generate_alerts_json edge cases."""

    def test_empty_db_no_alerts(self, tmp_path):
        """Empty prices table produces no stale data alerts."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM prices")
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        assert data["count"] == 0
        gen.conn.close()

    def test_alerts_sorted_by_timestamp(self, tmp_path):
        """Alerts are sorted by timestamp descending."""
        gen, _ = _make_generator(tmp_path)
        # Create two kill switch files with different timestamps
        kill_1 = tmp_path / ".kill_switch_paper"
        kill_1.write_text(json.dumps({
            "enabled": True, "reason": "first",
            "timestamp": "2026-01-02T00:00:00"
        }))
        kill_2 = tmp_path / ".kill_switch_live"
        kill_2.write_text(json.dumps({
            "enabled": True, "reason": "second",
            "timestamp": "2026-01-01T00:00:00"
        }))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        timestamps = [a.get("timestamp", "") for a in data["alerts"] if a.get("timestamp")]
        assert timestamps == sorted(timestamps, reverse=True), (
            f"Alerts not sorted descending: {timestamps}"
        )
        gen.conn.close()

    def test_regime_change_alert(self, tmp_path):
        """.regime_trigger file generates regime_change alert."""
        gen, _ = _make_generator(tmp_path)
        regime_file = tmp_path / ".regime_trigger"
        regime_file.write_text(json.dumps({
            "regime": "crisis", "vix": 30,
            "timestamp": "2026-01-01T00:00:00"
        }))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        regime_alerts = [a for a in data["alerts"] if a["type"] == "regime_change"]
        assert len(regime_alerts) >= 1
        assert regime_alerts[0]["level"] == "warning"
        gen.conn.close()

    def test_promote_trigger_alert(self, tmp_path):
        """.promote_to_live file generates graduation_candidate alert."""
        gen, _ = _make_generator(tmp_path)
        promote_file = tmp_path / ".promote_to_live"
        promote_file.write_text(json.dumps({
            "metrics": {"sharpe": 0.85},
            "timestamp": "2026-01-01T00:00:00"
        }))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        promote_alerts = [a for a in data["alerts"] if a["type"] == "graduation_candidate"]
        assert len(promote_alerts) >= 1
        assert promote_alerts[0]["level"] == "success"
        gen.conn.close()


# ---------------------------------------------------------------------------
# Run edge cases
# ---------------------------------------------------------------------------

class TestRunEdgeCases:
    """Test run() edge cases."""

    def test_run_closes_connection(self, tmp_path):
        """Connection is closed after run()."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                gen.run()
        # Should not raise - connection already closed
        gen.conn.close()

    def test_run_creates_index_json(self, tmp_path):
        """run() creates index.json with files list."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                gen.run()
        with open(tmp_path / "index.json") as f:
            index = json.load(f)
        assert "files" in index
        assert len(index["files"]) >= 6  # At least 6 dashboard files
        assert "generated_at" in index
        gen.conn.close()

    def test_generated_at_populated_in_all_files(self, tmp_path):
        """All non-index JSON files have generated_at field."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps([{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    gen.run()
        json_files = ["dashboard.json", "stats.json",
                      "alerts.json", "health.json"]
        for name in json_files:
            fpath = tmp_path / name
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                assert "generated_at" in data, f"{name} missing generated_at"
        # signals.json uses "timestamp" instead of "generated_at"
        signals_path = tmp_path / "signals.json"
        if signals_path.exists():
            with open(signals_path) as f:
                signals_data = json.load(f)
            assert "timestamp" in signals_data
        gen.conn.close()


# ---------------------------------------------------------------------------
# Analytics JSON tests
# ---------------------------------------------------------------------------

class TestGenerateAnalyticsJSON:
    """Test generate_analytics_json."""

    def test_generates_file(self, tmp_path):
        """Creates analytics.json even via fallback when dependencies unavailable."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_analytics_json()
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "status" in data
        gen.conn.close()

    def test_fallback_has_generated_at(self, tmp_path):
        """Fallback error report includes generated_at."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_analytics_json()
        with open(path) as f:
            data = json.load(f)
        assert "generated_at" in data
        gen.conn.close()


# ---------------------------------------------------------------------------
# Graduation JSON tests
# ---------------------------------------------------------------------------

class TestGenerateGraduationJSON:
    """Test generate_graduation_json."""

    def test_generates_file_with_graduation_data(self, tmp_path):
        """Creates graduation.json with expected top-level keys."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_graduation_json()
        assert path is not None
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "readiness_score" in data
        assert "is_graduation_ready" in data
        assert "criteria" in data
        assert "generated_at" in data
        gen.conn.close()

    def test_graduation_has_criteria_items(self, tmp_path):
        """Each criterion has name, passed, value, required, description."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_graduation_json()
        with open(path) as f:
            data = json.load(f)
        for criterion in data["criteria"]:
            assert "name" in criterion
            assert "passed" in criterion
            assert "value" in criterion
            assert "required" in criterion
            assert "description" in criterion
        gen.conn.close()


# ---------------------------------------------------------------------------
# Overlay JSON tests
# ---------------------------------------------------------------------------

class TestGenerateOverlayJSON:
    """Test generate_overlay_json."""

    def test_returns_path_or_none(self, tmp_path):
        """Returns a Path when overlay generator succeeds, or None otherwise."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                result = gen.generate_overlay_json()
        # Either a Path (success) or None (graceful failure) is acceptable
        assert result is None or isinstance(result, Path)
        gen.conn.close()


# ---------------------------------------------------------------------------
# GARCH-CVaR edge cases (continued)
# ---------------------------------------------------------------------------

class TestGarchCvarEdgeCases:
    """Additional _load_garch_cvar_data edge cases."""

    def test_flat_format_empty_dict(self, tmp_path):
        """Empty health report file returns all defaults."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text("{}")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == -0.0179
        assert data["garch_active"] is True
        assert data["volatility_clustering"] == "elevated"
        gen.conn.close()

    def test_flat_format_zero_values(self, tmp_path):
        """Zero values in health report are handled correctly."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "garch_filtered": True,
            "cvar_95": 0.0,
            "var_95": 0.0,
            "cvar_ratio": 0.0,
            "filter_active": False,
            "conditional_volatility_current": 0.0,
            "garch_persistence": 0.0,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == 0.0
        assert data["var_95"] == 0.0
        assert data["cvar_ratio"] == 0.0
        assert data["garch_active"] is False
        assert data["volatility_clustering"] == "normal"
        gen.conn.close()


# ---------------------------------------------------------------------------
# Entropy edge cases (continued)
# ---------------------------------------------------------------------------

class TestEntropyEdgeCases:
    """Additional _load_entropy_data edge cases."""

    def test_concentration_risk_exact_boundaries(self, tmp_path):
        """Normalized score at exact boundaries maps to correct risk labels."""
        gen, _ = _make_generator(tmp_path)
        boundaries = [(91, "good"), (71, "low"), (51, "medium"), (31, "high"), (0, "critical")]
        for score, expected in boundaries:
            health_file = tmp_path / ".health_report.json"
            health_file.write_text(json.dumps({
                "checks": {
                    "portfolio_entropy": {
                        "metrics": {
                            "shannon_entropy": 1.0,
                            "effective_n": 2.5,
                            "normalized_score": score,
                            "hhi_index": 0.38,
                        }
                    }
                }
            }))
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                data = gen._load_entropy_data()
            assert data["concentration_risk"] == expected, (
                f"Score {score} should be {expected}, got {data['concentration_risk']}"
            )
        gen.conn.close()

    def test_empty_health_file_returns_defaults(self, tmp_path):
        """Empty JSON health file returns default entropy values."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text("{}")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_entropy_data()
        assert data["shannon_entropy"] == 1.02
        assert data["concentration_risk"] == "good"
        gen.conn.close()


# ---------------------------------------------------------------------------
# Signals JSON edge cases (continued)
# ---------------------------------------------------------------------------

class TestSignalsJSONAdditionalEdgeCases:
    """Additional generate_signals_json edge cases."""

    def test_vix_at_low_vol_boundary(self, tmp_path):
        """VIX at 14 (just below 15) classifies as low_vol."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prices VALUES ('^VIX', ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), 14.0))
        conn.commit()
        conn.close()
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert data["regime"]["regime"] == "low_vol"
        gen.conn.close()

    def test_vix_at_crisis_boundary(self, tmp_path):
        """VIX at 26 (just above 25) classifies as crisis."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prices VALUES ('^VIX', ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), 26.0))
        conn.commit()
        conn.close()
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert data["regime"]["regime"] == "crisis"
        gen.conn.close()

    def test_empty_positions_in_portfolio_state(self, tmp_path):
        """Portfolio state with empty positions generates valid output."""
        gen, _ = _make_generator(tmp_path)
        state_file = tmp_path / "portfolio_paper.json"
        state_file.write_text(json.dumps({
            "positions": {},
            "cash": 50000.0
        }))
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert data["cash"] == 50000.0
        assert data["total_value"] == 50000.0
        assert data["current_positions"] == []
        gen.conn.close()

    def test_all_optional_keys_present(self, tmp_path):
        """signals.json output contains all optional section keys."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        all_keys = {
            "timestamp", "regime", "target_allocations", "current_positions",
            "cash", "total_value", "latest_prices", "recent_orders", "ml_signals",
            "factor_rotation", "yield_curve", "duration_allocation",
            "convexity_harvest", "volatility_parity", "llm_sentiment",
            "ensemble_voting", "sector_rotation", "alternative_data",
            "behavioral_sentiment", "stacking_ensemble", "factor_rotation_dashboard",
            "smart_rebalance", "broker", "garch_cvar", "entropy", "bond_momentum",
        }
        assert all_keys.issubset(set(data.keys())), (
            f"Missing keys: {all_keys - set(data.keys())}"
        )
        gen.conn.close()


# ---------------------------------------------------------------------------
# ML signals edge cases (continued)
# ---------------------------------------------------------------------------

class TestMlSignalsAdditionalEdgeCases:
    """Additional _generate_ml_signals edge cases."""

    def test_grid_search_empty_file(self, tmp_path):
        """Empty grid search results file returns empty grid_search dict."""
        gen, _ = _make_generator(tmp_path)
        grid_file = tmp_path / "grid_search_results.jsonl"
        grid_file.write_text("")
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 15, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is True
        assert signals["grid_search"] == {}
        gen.conn.close()

    def test_vix_at_vol_spike_boundary_ml_prediction(self, tmp_path):
        """VIX at 21 (just above 20) triggers vol_spike probability distribution."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 21, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["probabilities"]["bear"] == 0.3
        assert pred["probabilities"]["neutral"] == 0.5
        assert pred["probabilities"]["bull"] == 0.2
        gen.conn.close()


# ---------------------------------------------------------------------------
# Stats JSON edge cases (continued)
# ---------------------------------------------------------------------------

class TestStatsJSONAdditionalEdgeCases:
    """Additional generate_stats_json edge cases."""

    def test_no_vix_in_prices(self, tmp_path):
        """Missing VIX symbol in prices does not crash stats generation."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM prices WHERE symbol = '^VIX'")
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        assert "asset_stats" in data
        gen.conn.close()


# ---------------------------------------------------------------------------
# Constants validation (continued)
# ---------------------------------------------------------------------------

class TestConstantsAdditional:
    """Additional module-level constant validation."""

    def test_data_dir_is_path_instance(self):
        """DATA_DIR is a Path instance."""
        from src.dashboard.generator import DATA_DIR
        assert isinstance(DATA_DIR, Path)

    def test_db_path_is_path_instance(self):
        """DB_PATH is a Path instance."""
        from src.dashboard.generator import DB_PATH
        assert isinstance(DB_PATH, Path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
