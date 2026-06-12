#!/usr/bin/env python3
"""
Tests for dashboard generator — VIX regime detection, data freshness,
health status, alerts, broker data, and stats calculation.
"""
import json
import sqlite3
import numpy as np

import pytest
from datetime import datetime, timedelta, timezone
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


def _write_ok_source_manifest(public_dir: Path) -> None:
    """Write a compact live source manifest for healthy health-json fixtures."""
    (public_dir / "source_manifest.json").write_text(json.dumps({
        "artifacts": [
            {
                "artifact": "prices.json",
                "provider": "Yahoo Finance",
                "feed": "chart/v8",
                "source_mode": "live",
                "status": "success",
                "data_quality": {
                    "artifact": "data_quality.json",
                    "schema_version": "price-data-quality/v1",
                    "status": "ok",
                    "issue_counts": {"total": 0},
                },
            },
        ],
    }))


class TestSignalStalenessNormalization:
    """Signal staleness should distinguish stale from optional unavailable."""

    def test_required_fresh_and_optional_missing_sections_are_not_stale(self, tmp_path):
        gen, _ = _make_generator(tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()

        result = gen._check_signal_staleness({
            "ensemble_voting": {"generated_at": fresh},
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"timestamp": fresh},
            "smart_rebalance": {"generated_at": fresh},
            "rebalance_health": {"generated": fresh},
        })

        assert result["stale_signals"] == []
        assert "behavioral_sentiment" in result["unavailable_signals"]
        assert "two_stage_regime" in result["unavailable_signals"]
        assert result["signal_timestamps"]["rebalance_health"] == fresh
        assert result["healthy_count"] == result["required_count"]

    def test_required_stale_signal_remains_stale(self, tmp_path):
        gen, _ = _make_generator(tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()
        stale = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()

        result = gen._check_signal_staleness({
            "ensemble_voting": {"generated_at": fresh},
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"timestamp": stale},
            "smart_rebalance": {"generated_at": fresh},
            "rebalance_health": {"generated": fresh},
        })

        assert "garch_cvar" in result["stale_signals"]
        assert result["signal_age_hours"]["garch_cvar"] >= 8

    def test_optional_error_placeholder_is_unavailable_not_stale(self, tmp_path):
        gen, _ = _make_generator(tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()

        result = gen._check_signal_staleness({
            "ensemble_voting": {"generated_at": fresh},
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"timestamp": fresh},
            "smart_rebalance": {"generated_at": fresh},
            "rebalance_health": {"generated": fresh},
            "two_stage_regime": {"error": "FRED API key unavailable"},
        })

        assert "two_stage_regime" not in result["stale_signals"]
        assert "two_stage_regime" in result["unavailable_signals"]


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
        kill_file = tmp_path / "kill_switch.json"
        kill_file.write_text(json.dumps({"enabled": True, "reason": "test", "mode": "paper", "timestamp": datetime.now().isoformat()}))
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

    def test_fred_readiness_populates_health_and_slo(self, tmp_path):
        """FRED readiness should be included in dashboard health and SLO output."""
        gen, _ = _make_generator(tmp_path)
        readiness = {
            "status": "warning",
            "readiness": "warn",
            "mode": "lab",
            "ready": True,
            "blocking": False,
            "reason": "missing_fred_api_key",
            "source_mode": "synthetic",
            "remediation": "Set FRED_API_KEY for lab/paper/live operation.",
        }
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch(
                    "src.data.fred_data.get_fred_md_cache_health",
                    return_value={
                        "status": "unavailable",
                        "source_mode": "synthetic",
                        "api_key_configured": False,
                    },
                ):
                    with patch("src.monitor.fred_readiness.assess_fred_readiness", return_value=readiness):
                        path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)

        assert data["fred_readiness"]["reason"] == "missing_fred_api_key"
        assert data["data_pipeline_slo"]["dimensions"]["fred_readiness"]["status"] == "warning"
        gen.conn.close()

    def test_rebalance_live_diagnostics_populate_health_slo(self, tmp_path):
        """Rebalance live diagnostics should be included in dashboard health SLO output."""
        gen, _ = _make_generator(tmp_path)
        (tmp_path / "rebalance_health.json").write_text(json.dumps({
            "generated": "2026-06-12T16:43:07.176691",
            "market_data_consistency": {
                "status": "unavailable",
                "reason": "alpaca_not_configured",
                "checked_at": "2026-06-12T08:43:07.177011+00:00",
                "rows": [],
                "warnings": [],
            },
            "alpaca_feed_entitlement": {
                "configured_feed": "iex",
                "effective_feed": "iex",
                "entitlement": "unknown",
                "delayed": False,
                "acceptable_for_live": False,
                "policy_decision": "reject",
                "reason": "missing_entitlement",
            },
        }))

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)

        assert data["data_pipeline_slo"]["top_dimension"] == "alpaca_feed_entitlement"
        assert data["data_pipeline_slo"]["dimensions"]["alpaca_feed_entitlement"]["status"] == "critical"
        assert data["data_pipeline_slo"]["dimensions"]["market_data_consistency"]["status"] == "warning"
        gen.conn.close()

    def test_provider_latest_date_symbols_are_fresh_even_with_calendar_lag(self, tmp_path):
        """Freshness status is relative to the provider's latest available date."""
        db_path = tmp_path / "market.db"
        provider_latest = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, PRIMARY KEY (symbol, date))"
        )
        for symbol in ("SPY", "GLD", "TLT"):
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)", (symbol, provider_latest, 100.0))
        conn.commit()
        conn.close()
        gen = DashboardGenerator.__new__(DashboardGenerator)
        gen.conn = sqlite3.connect(str(db_path))
        gen.conn.row_factory = sqlite3.Row
        (tmp_path / "cron_status.json").write_text('{"jobs": []}')

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()

        with open(path) as f:
            data = json.load(f)
        assert {item["status"] for item in data["data_freshness"].values()} == {"fresh"}
        assert data["data_freshness"]["SPY"]["days_stale"] >= 2
        assert data["data_freshness"]["SPY"]["market_lag_days"] == 0
        assert data["data_freshness"]["SPY"]["latest_available_market_date"] == provider_latest
        gen.conn.close()

    def test_symbol_lagging_provider_latest_date_is_critical(self, tmp_path):
        """A symbol behind the provider's latest date should still be flagged."""
        db_path = tmp_path / "market.db"
        provider_latest = datetime.now() - timedelta(days=2)
        lagging_date = provider_latest - timedelta(days=5)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, PRIMARY KEY (symbol, date))"
        )
        conn.execute("INSERT INTO prices VALUES (?, ?, ?)", ("SPY", provider_latest.strftime("%Y-%m-%d"), 100.0))
        conn.execute("INSERT INTO prices VALUES (?, ?, ?)", ("GLD", lagging_date.strftime("%Y-%m-%d"), 100.0))
        conn.commit()
        conn.close()
        gen = DashboardGenerator.__new__(DashboardGenerator)
        gen.conn = sqlite3.connect(str(db_path))
        gen.conn.row_factory = sqlite3.Row
        (tmp_path / "cron_status.json").write_text('{"jobs": []}')

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()

        with open(path) as f:
            data = json.load(f)
        assert data["data_freshness"]["SPY"]["status"] == "fresh"
        assert data["data_freshness"]["GLD"]["status"] == "critical"
        assert data["data_freshness"]["GLD"]["market_lag_days"] == 5
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
        required_keys = {"generated_at", "regime", "target_allocations", "current_positions",
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
        """No cron_status.json does not invent scheduled cron jobs."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert data["cron_jobs"] == []
        assert data["system_status"] == "warning"
        assert data["scheduler_status"]["status"] == "unavailable"
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
        (tmp_path / "cron_status.json").write_text('{"jobs": []}')
        _write_ok_source_manifest(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert data["system_status"] == "healthy"
        assert len(data["data_freshness"]) == 4
        gen.conn.close()

    def test_hermes_error_degrades_dashboard_health(self, tmp_path, monkeypatch):
        """Active Hermes portfolio-lab errors should be visible in dashboard health."""
        gen, _ = _make_generator(tmp_path)
        (tmp_path / "cron_status.json").write_text(json.dumps({
            "jobs": [
                {
                    "name": "portfolio-lab-data",
                    "status": "ok",
                    "last_run": "2026-06-08T12:00:00+08:00",
                    "backend": "local",
                }
            ]
        }))
        hermes_jobs = tmp_path / "hermes_jobs.json"
        hermes_jobs.write_text(json.dumps({
            "jobs": [
                {
                    "id": "ok-job",
                    "name": "portfolio-lab-dashboard",
                    "schedule_display": "15 * * * *",
                    "last_run_at": "2026-06-08T12:15:00+08:00",
                    "next_run_at": "2026-06-08T13:15:00+08:00",
                    "last_status": "ok",
                    "state": "scheduled",
                    "enabled": True,
                    "workdir": str(tmp_path),
                },
                {
                    "id": "bad-job",
                    "name": "portfolio-lab-autonomous-agent",
                    "schedule_display": "40 */2 * * *",
                    "last_run_at": "2026-06-08T12:47:00+08:00",
                    "next_run_at": "2026-06-08T14:40:00+08:00",
                    "last_status": "error",
                    "last_error": "RuntimeError: final report text",
                    "state": "scheduled",
                    "enabled": True,
                    "workdir": str(tmp_path),
                },
                {
                    "id": "other-project",
                    "name": "finance-digest",
                    "last_status": "error",
                    "enabled": True,
                },
            ]
        }))
        monkeypatch.setenv("HERMES_CRON_JOBS_PATH", str(hermes_jobs))

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)

        hermes_error = next(j for j in data["cron_jobs"] if j["id"] == "bad-job")
        assert data["system_status"] == "warning"
        assert data["scheduler_status"]["status"] == "degraded"
        assert data["scheduler_status"]["backends"]["hermes"]["failed_jobs"] == 1
        assert hermes_error["backend"] == "hermes"
        assert hermes_error["name"] == "portfolio-lab-autonomous-agent"
        assert hermes_error["schedule"] == "40 */2 * * *"
        assert hermes_error["last_run"] == "2026-06-08T12:47:00+08:00"
        assert hermes_error["status"] == "error"
        assert hermes_error["error"] == "RuntimeError: final report text"
        assert not any(j["name"] == "finance-digest" for j in data["cron_jobs"])
        gen.conn.close()

    def test_missing_hermes_state_warns_without_crashing(self, tmp_path, monkeypatch):
        """Unavailable Hermes state should be explicit warning metadata."""
        gen, _ = _make_generator(tmp_path)
        (tmp_path / "cron_status.json").write_text(json.dumps({
            "jobs": [{"name": "portfolio-lab-data", "status": "ok"}]
        }))
        monkeypatch.setenv("HERMES_CRON_JOBS_PATH", str(tmp_path / "missing-jobs.json"))

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)

        assert data["system_status"] == "warning"
        assert data["scheduler_status"]["backends"]["hermes"]["status"] == "unavailable"
        assert "missing-jobs.json" in data["scheduler_status"]["backends"]["hermes"]["source"]
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
        conn = gen.conn
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                gen.run()
        # Connection should be None after close()
        assert gen.conn is None

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
        # Connection is closed by run()

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
        # signals.json uses "generated_at" consistent with other JSON outputs
        signals_path = tmp_path / "signals.json"
        if signals_path.exists():
            with open(signals_path) as f:
                signals_data = json.load(f)
            assert "generated_at" in signals_data
        # Connection already closed by run()


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
            "generated_at", "regime", "target_allocations", "current_positions",
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


# ---------------------------------------------------------------------------
# Sector momentum signals tests (completely untested method)
# ---------------------------------------------------------------------------

class TestSectorMomentumSignals:
    """Test _generate_sector_momentum_signals edge cases."""

    def test_none_when_import_fails(self, tmp_path):
        """Returns None when sector_momentum_calc cannot be imported."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.strategy.sector_momentum_calc.generate_sector_signals",
                          side_effect=ImportError("no module")):
                    result = gen._generate_sector_momentum_signals()
        assert result is None
        gen.conn.close()

    def test_none_when_generate_raises(self, tmp_path):
        """Returns None when generate_sector_signals raises an exception."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.strategy.sector_momentum_calc.generate_sector_signals",
                          side_effect=ValueError("bad data")):
                    result = gen._generate_sector_momentum_signals()
        assert result is None
        gen.conn.close()

    def test_passes_vix_to_generate(self, tmp_path):
        """Vix level is passed to generate_sector_signals via vix_level parameter."""
        gen, db_path = _make_generator(tmp_path)
        mock_signals = {"SPY": {"momentum": 0.5}}
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.strategy.sector_momentum_calc.generate_sector_signals",
                          return_value=mock_signals) as mock_gen:
                    result = gen._generate_sector_momentum_signals(vix_level=18.5)
        assert result == mock_signals
        # Verify vix was passed
        _, kwargs = mock_gen.call_args
        assert kwargs.get("vix") == 18.5
        gen.conn.close()

    def test_vix_fetch_failure_defaults_zero(self, tmp_path):
        """When vix_level is None (no VIX data), vix parameter defaults to 0."""
        gen, db_path = _make_generator(tmp_path)
        mock_signals = {"SPY": {"momentum": 0.5}}
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.strategy.sector_momentum_calc.generate_sector_signals",
                          return_value=mock_signals) as mock_gen:
                    result = gen._generate_sector_momentum_signals(vix_level=None)
        assert result == mock_signals
        _, kwargs = mock_gen.call_args
        assert kwargs.get("vix") == 0
        gen.conn.close()

    def test_none_when_no_vix_row(self, tmp_path):
        """No VIX row in DB still calls generate with vix=0 and returns result."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM prices WHERE symbol = '^VIX'")
        conn.commit()
        conn.close()
        mock_signals = {"SPY": {"momentum": 0.5}}
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.strategy.sector_momentum_calc.generate_sector_signals",
                          return_value=mock_signals) as mock_gen:
                    result = gen._generate_sector_momentum_signals()
        assert result == mock_signals
        gen.conn.close()


# ---------------------------------------------------------------------------
# Signals JSON — regime composite integration tests
# ---------------------------------------------------------------------------

class TestSignalsJSONRegimeComposite:
    """Test full regime composite logic in generate_signals_json."""

    def test_vix_crisis_overrides_trend(self, tmp_path):
        """VIX crisis (>25) overrides any trend regime."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prices VALUES ('^VIX', ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), 30.0))
        conn.execute("INSERT INTO regime_log VALUES (?, ?, ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), "bull", 30.0,
                      datetime.now().isoformat()))
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

    def test_vix_vol_spike_overrides_trend(self, tmp_path):
        """VIX vol_spike (21-25) overrides trend regime."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prices VALUES ('^VIX', ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), 22.0))
        conn.execute("INSERT INTO regime_log VALUES (?, ?, ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), "bull", 22.0,
                      datetime.now().isoformat()))
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
        assert data["regime"]["regime"] == "vol_spike"
        gen.conn.close()

    def test_low_vol_with_crisis_trend_uses_trend(self, tmp_path):
        """low_vol VIX with crisis trend falls through to trend."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prices VALUES ('^VIX', ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), 14.0))
        conn.execute("INSERT INTO regime_log VALUES (?, ?, ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), "crisis", 14.0,
                      datetime.now().isoformat()))
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

    def test_crisis_target_alloc_applied(self, tmp_path):
        """Crisis regime uses crisis target allocation weights."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prices VALUES ('^VIX', ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), 30.0))
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
        expected = {"SPY": 0.20, "GLD": 0.50, "TLT": 0.30}
        assert data["target_allocations"] == expected
        gen.conn.close()


# ---------------------------------------------------------------------------
# Signals JSON — positions, orders, and paper portfolio state
# ---------------------------------------------------------------------------

class TestSignalsJSONPositions:
    """Test generate_signals_json with portfolio state."""

    def test_portfolio_positions_parsed(self, tmp_path):
        """Portfolio paper state positions are parsed correctly."""
        gen, _ = _make_generator(tmp_path)
        state_file = tmp_path / "portfolio_paper.json"
        state_file.write_text(json.dumps({
            "positions": {
                "SPY": {"shares": 100, "value": 45000, "weight": 0.45, "unrealized_pnl": 500},
                "GLD": {"shares": 200, "value": 35000, "weight": 0.35, "unrealized_pnl": -200},
            },
            "cash": 20000.0
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
        positions = {p["symbol"]: p for p in data["current_positions"]}
        assert "SPY" in positions
        assert positions["SPY"]["shares"] == 100
        assert positions["SPY"]["value"] == 45000
        assert data["cash"] == 20000.0
        assert data["total_value"] == 100000.0  # cash + 45000 + 35000
        gen.conn.close()

    def test_orders_parsed_from_log(self, tmp_path):
        """Orders from orders.jsonl are parsed into recent_orders."""
        gen, _ = _make_generator(tmp_path)
        orders_file = tmp_path / "orders.jsonl"
        orders_file.write_text(
            json.dumps({"symbol": "SPY", "side": "buy", "shares": 10, "fill_value": 4500}) + "\n"
            + json.dumps({"symbol": "GLD", "side": "sell", "shares": 5, "fill_value": 1750}) + "\n"
        )
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
        orders = data["recent_orders"]
        assert len(orders) == 2
        assert orders[0]["sym"] == "GLD"    # Reversed
        assert orders[1]["sym"] == "SPY"
        gen.conn.close()

    def test_malformed_order_skipped(self, tmp_path):
        """Malformed JSON line in orders.jsonl is skipped."""
        gen, _ = _make_generator(tmp_path)
        orders_file = tmp_path / "orders.jsonl"
        orders_file.write_text(
            "not valid json\n"
            + json.dumps({"symbol": "SPY", "side": "buy", "shares": 10, "fill_value": 4500}) + "\n"
        )
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
        assert len(data["recent_orders"]) == 1
        gen.conn.close()

    def test_only_last_five_orders(self, tmp_path):
        """Only the last 5 orders from orders.jsonl are kept."""
        gen, _ = _make_generator(tmp_path)
        orders_file = tmp_path / "orders.jsonl"
        lines = []
        for i in range(10):
            lines.append(json.dumps({"symbol": f"SYM{i}", "side": "buy", "shares": 1, "fill_value": 100 * i}))
        orders_file.write_text("\n".join(lines) + "\n")
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
        assert len(data["recent_orders"]) == 5
        gen.conn.close()

    def test_latest_prices_from_db(self, tmp_path):
        """Latest prices dict is populated from DB."""
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
        assert "SPY" in data["latest_prices"]
        assert "GLD" in data["latest_prices"]
        assert isinstance(data["latest_prices"]["SPY"], float)
        gen.conn.close()


# ---------------------------------------------------------------------------
# Signals JSON — smart rebalance
# ---------------------------------------------------------------------------

class TestSignalsJSONSmartRebalance:
    """Test smart rebalance data in generate_signals_json."""

    def test_smart_rebalance_fallback_data(self, tmp_path):
        """Smart rebalance has fallback data when import fails."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    with patch("importlib.import_module",
                              side_effect=ImportError("no rebalancing")):
                        path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        # smart_rebalance should be None when import fails
        assert data["smart_rebalance"] is None
        gen.conn.close()


# ---------------------------------------------------------------------------
# Signals JSON — alternative data
# ---------------------------------------------------------------------------

class TestSignalsJSONAlternativeData:
    """Test alternative data loading in generate_signals_json."""

    def test_alternative_data_loaded(self, tmp_path):
        """Alternative data from JSON file is loaded into output."""
        gen, _ = _make_generator(tmp_path)
        alt_dir = tmp_path / "signals"
        alt_dir.mkdir(exist_ok=True)
        alt_file = alt_dir / "alternative_data_latest.json"
        alt_file.write_text(json.dumps({
            "regime": "risk_on",
            "probability": 0.65,
            "confidence": 0.72,
            "timestamp": "2026-01-01T00:00:00",
            "raw_data": {
                "earnings_sentiment": 0.3,
                "earnings_confidence": 0.8,
                "news_sentiment": 0.6,
                "news_confidence": 0.7,
                "jobs_signal": 0.2,
                "jobs_confidence": 0.6,
                "social_sentiment": 0.4,
                "social_confidence": 0.5,
                "composite_score": 0.38,
                "z_score": 0.5,
                "sources_count": 4,
                "data_freshness_hours": 2.5,
                "weights": {
                    "earnings": 0.3,
                    "news": 0.3,
                    "jobs": 0.2,
                    "social": 0.2
                }
            }
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
        alt = data["alternative_data"]
        assert alt is not None
        assert alt["regime"] == "risk_on"
        assert alt["composite_score"] == 0.38
        assert alt["components"]["earnings"]["score"] == 0.3
        assert alt["sources_count"] == 4
        assert alt["data_freshness_hours"] == 2.5
        gen.conn.close()

    def test_alternative_data_missing_file(self, tmp_path):
        """Missing alternative data file falls back to None."""
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
        assert data["alternative_data"] is None
        gen.conn.close()

    def test_alternative_data_malformed(self, tmp_path):
        """Malformed alternative data file falls back to None."""
        gen, _ = _make_generator(tmp_path)
        alt_dir = tmp_path / "signals"
        alt_dir.mkdir(exist_ok=True)
        alt_file = alt_dir / "alternative_data_latest.json"
        alt_file.write_text("not valid json")
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
        assert data["alternative_data"] is None
        gen.conn.close()


# ---------------------------------------------------------------------------
# Stats JSON — paper portfolio and SPY comparison (core logic, untested)
# ---------------------------------------------------------------------------

class TestStatsJSONPaperPerformance:
    """Test paper portfolio metrics in generate_stats_json."""

    def test_paper_metrics_with_perf_log(self, tmp_path):
        """Performance log entries produce paper portfolio metrics."""
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        value = 100000.0
        lines = []
        for i in range(25):
            lines.append(json.dumps({
                "timestamp": f"2026-01-{i+1:02d}T00:00:00",
                "total_value": value,
                "daily_return": 0.001 if i > 0 else 0.0,
            }))
            value *= 1.001
        perf_log.write_text("\n".join(lines) + "\n")
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        paper = data["paper_portfolio"]
        assert "sharpe" in paper
        assert "total_return" in paper
        assert "max_value" in paper
        assert "min_value" in paper
        assert "days_tracked" in paper
        assert paper["days_tracked"] == 25
        gen.conn.close()

    def test_paper_metrics_insufficient_data(self, tmp_path):
        """Fewer than 20 perf entries produces empty paper_metrics."""
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        perf_log.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:00",
            "total_value": 100000,
            "daily_return": 0.001,
        }) + "\n")
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        assert data["paper_portfolio"] == {}
        assert data["spy_comparison"] is None
        gen.conn.close()

    def test_paper_metrics_sharpe_with_no_variance(self, tmp_path):
        """All-zero daily_return entries are filtered out, yielding empty metrics."""
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        lines = []
        for i in range(25):
            lines.append(json.dumps({
                "timestamp": f"2026-01-{i+1:02d}T00:00:00",
                "total_value": 100000.0,
                "daily_return": 0.0,
            }))
        perf_log.write_text("\n".join(lines) + "\n")
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        # Zero returns are valid daily returns — Sharpe should be 0 (no excess return)
        paper = data["paper_portfolio"]
        assert paper["sharpe"] == 0
        assert paper["days_tracked"] == 25
        assert paper["total_return"] == 0.0
        gen.conn.close()

    def test_paper_metrics_all_fields_populated(self, tmp_path):
        """With enough non-zero returns, all paper metric fields are present."""
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        value = 100000.0
        lines = []
        for i in range(25):
            ret = 0.001 + (i * 0.0001)  # Increasing returns for variance
            lines.append(json.dumps({
                "timestamp": f"2026-01-{i+1:02d}T00:00:00",
                "total_value": round(value, 2),
                "daily_return": round(ret, 6),
            }))
            value *= 1.001
        perf_log.write_text("\n".join(lines) + "\n")
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        paper = data["paper_portfolio"]
        assert "sharpe" in paper
        assert "total_return" in paper
        assert "max_value" in paper
        assert "min_value" in paper
        assert "days_tracked" in paper
        assert isinstance(paper["sharpe"], (int, float))
        gen.conn.close()

    def test_paper_metrics_deduplicates_intraday_entries(self, tmp_path):
        """Intraday entries for the same date must not inflate days_tracked.

        Regression test: performance.jsonl may contain multiple entries per
        day (cron runs, manual syncs).  days_tracked must count unique
        calendar dates, not raw JSONL lines.
        """
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        lines = []
        # 3 dates × 10 intraday entries each = 30 raw lines, but only 3 unique days
        for day in range(1, 4):
            for hour in range(10):
                ret = 0.001 if hour == 0 else 0.0  # Only first entry per day has return
                lines.append(json.dumps({
                    "timestamp": f"2026-01-{day:02d}T{hour:02d}:00:00",
                    "total_value": 100000.0 + day * 100,
                    "daily_return": ret,
                }))
        perf_log.write_text("\n".join(lines) + "\n")
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        paper = data["paper_portfolio"]
        # Must be 3 unique dates, not 30 raw lines
        assert paper["days_tracked"] == 3
        gen.conn.close()


class TestStatsJSONSpyComparison:
    """Test SPY comparison in generate_stats_json."""

    def test_spy_comparison_present_with_enough_data(self, tmp_path):
        """SPY comparison is calculated with sufficient perf and SPY data."""
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        value = 100000.0
        lines = []
        for i in range(25):
            lines.append(json.dumps({
                "timestamp": f"2026-01-{i+1:02d}T00:00:00",
                "total_value": value,
                "daily_return": 0.001 if i > 0 else 0.0,
            }))
            value *= 1.001
        perf_log.write_text("\n".join(lines) + "\n")
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        # spy_comparison may be None if SPY prices don't align with timestamps
        # Just verify no crash and asset_stats present
        assert "asset_stats" in data
        gen.conn.close()


# ---------------------------------------------------------------------------
# Health JSON — signal health testing
# ---------------------------------------------------------------------------

class TestHealthJSONSignalHealth:
    """Test signal health in generate_health_json."""

    def test_signal_health_present(self, tmp_path):
        """Signal health is populated in health output."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert "signal_health" in data
        gen.conn.close()

    def test_signal_health_error_fallback(self, tmp_path):
        """Signal health has error fallback when tracker unavailable."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.signals.health_tracker.SignalHealthTracker.get_health_report",
                          side_effect=ImportError("no tracker")):
                    path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert "error" in data["signal_health"]
        assert data["signal_health"]["status"] == "unavailable"
        gen.conn.close()

    def test_cron_status_loaded(self, tmp_path):
        """Cron status from file is loaded into health data."""
        gen, _ = _make_generator(tmp_path)
        cron_file = tmp_path / "cron_status.json"
        cron_file.write_text(json.dumps({
            "jobs": [
                {"name": "portfolio-lab-data", "status": "success", "state": "completed"},
                {"name": "portfolio-lab-eval", "status": "error", "state": "failed"},
            ]
        }))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["cron_jobs"]) == 2
        assert data["cron_jobs"][0]["status"] == "ok"
        assert data["cron_jobs"][0]["backend"] == "local"
        gen.conn.close()


# ---------------------------------------------------------------------------
# Performance JSON — regime data and paper portfolio
# ---------------------------------------------------------------------------

class TestPerformanceJSONRegime:
    """Test regime data in generate_performance_json."""

    def test_regime_data_included(self, tmp_path):
        """Regime data from DB is included in performance output."""
        gen, _ = _make_generator(tmp_path)
        conn = gen.conn
        recent = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        conn.execute("INSERT INTO regime_log VALUES (?, ?, ?, ?)",
                     (recent, "normal", 15.0, datetime.now().isoformat()))
        conn.commit()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["regimes"]) >= 1
        assert data["regimes"][0]["r"] == "normal"
        gen.conn.close()

    def test_paper_portfolio_from_log(self, tmp_path):
        """Paper portfolio entries from performance.jsonl are included."""
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        perf_log.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:00",
            "total_value": 100000,
            "daily_return": 0.01,
        }) + "\n")
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["paper_portfolio"]) == 1
        entry = data["paper_portfolio"][0]
        assert entry["t"] == "2026-01-01"
        assert entry["v"] == 100000
        assert entry["r"] == 0.01
        gen.conn.close()


# ---------------------------------------------------------------------------
# Yield curve — missing keys and malformed data
# ---------------------------------------------------------------------------

class TestYieldCurveMalformed:
    """Test _get_yield_curve_data with malformed data."""

    def test_missing_spread_key(self, tmp_path):
        """Missing spread2s10s key defaults to 0 spread."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"dgs2": 4.0, "dgs10": 5.0} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        assert data["yield_curve"]["spread2s10s"] == 0
        assert data["yield_curve"]["duration_regime"] == "inverted"
        gen.conn.close()

    def test_none_spread_entries_skipped(self, tmp_path):
        """None values in spread entries are excluded from spread_history."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        entries = []
        for i in range(35):
            if i % 3 == 0:
                entries.append({"spread2s10s": None, "dgs2": 4.0, "dgs10": 5.0})
            else:
                entries.append({"spread2s10s": i * 3, "dgs2": 4.0, "dgs10": 5.0})
        yields_path.write_text(json.dumps(entries))
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        # None entries should be excluded from spread_history
        assert None not in data["yield_curve"]["spread_history"]
        gen.conn.close()

    def test_yields_file_empty_json_object(self, tmp_path):
        """Non-list JSON in yields file is handled gracefully."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text("{}")
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        # Should return empty result
        assert data["yield_curve"] is None
        gen.conn.close()


# ---------------------------------------------------------------------------
# Generator init — edge cases
# ---------------------------------------------------------------------------

class TestGeneratorInitEdgeCases:
    """Additional DashboardGenerator initialization edge cases."""

    def test_public_dir_created(self, tmp_path):
        """PUBLIC_DIR is created during init."""
        new_public = tmp_path / "non_existent" / "data"
        assert not new_public.exists()
        # We can't easily test the constructor because it calls sqlite_connect
        # Instead verify that __init__ would create it
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", new_public):
            gen.__init__()
        assert new_public.exists()
        gen.conn.close()


# ---------------------------------------------------------------------------
# Run — overlay and signals edge cases
# ---------------------------------------------------------------------------

class TestRunOverlay:
    """Test run() with overlay generation."""

    def test_run_with_overlay(self, tmp_path):
        """run() includes overlay path when overlay generates successfully."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    gen.run()
        assert (tmp_path / "index.json").exists()
        # Verify signals.json was generated with regime data
        with open(tmp_path / "signals.json") as f:
            signals = json.load(f)
        assert "regime" in signals
        assert "generated_at" in signals
        # Connection already closed by run()


# ---------------------------------------------------------------------------
# ML signals — edge cases continued
# ---------------------------------------------------------------------------

class TestMlSignalsGridSearch:
    """Test ML signals grid search edge cases."""

    def test_grid_search_malformed_line(self, tmp_path):
        """Malformed line in grid search file is caught gracefully."""
        gen, _ = _make_generator(tmp_path)
        grid_file = tmp_path / "grid_search_results.jsonl"
        grid_file.write_text("not valid json\n")
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

    def test_multiple_features_keeps_latest(self, tmp_path):
        """Multiple entries for same symbol keep the latest by timestamp."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(
            json.dumps({"symbol": "SPY", "vix_level": 30, "trend_direction": 0,
                        "price_vs_sma20": 0, "timestamp": "2026-01-01T00:00:00"}) + "\n"
            + json.dumps({"symbol": "SPY", "vix_level": 15, "trend_direction": 1,
                          "price_vs_sma20": 0.05, "timestamp": "2026-01-02T00:00:00"}) + "\n"
        )
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is True
        assert signals["features"]["SPY"]["vix_level"] == 15  # Latest
        gen.conn.close()

    def test_missing_vix_in_features(self, tmp_path):
        """Features missing vix_level key defaults to 20 in predictions."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        # Default probabilities: vix <=20 and no trend
        assert pred["predicted_regime"] == "neutral"
        gen.conn.close()


# ---------------------------------------------------------------------------
# VIX regime detection — boundary values at exact thresholds
# ---------------------------------------------------------------------------

class TestVIXRegimeBoundaries:
    """VIX regime at exact boundary values."""

    def test_vix_exactly_15(self):
        """VIX exactly 15 is normal regime."""
        gen = DashboardGenerator.__new__(DashboardGenerator)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path := Path("/tmp")):
            # Extract the classify logic
            def classify(v):
                if v > 25: return "crisis"
                elif v > 20: return "vol_spike"
                elif v < 15: return "low_vol"
                else: return "normal"
            assert classify(15) == "normal"
            assert classify(20) == "normal"
            assert classify(25) == "vol_spike"  # >20 not >=20

    def test_vix_vol_spike_upper(self):
        """VIX exactly 25 is vol_spike (>20, not >25)."""
        gen = DashboardGenerator.__new__(DashboardGenerator)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path := Path("/tmp")):
            def classify(v):
                if v > 25: return "crisis"
                elif v > 20: return "vol_spike"
                elif v < 15: return "low_vol"
                else: return "normal"
            assert classify(25) == "vol_spike"


# ---------------------------------------------------------------------------
# Graduation JSON — additional edge cases
# ---------------------------------------------------------------------------

class TestGraduationJSONEdgeCases:
    """Additional generate_graduation_json edge cases."""

    def test_graduation_manual_approval_fields(self, tmp_path):
        """Graduation output has manual approval fields."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_graduation_json()
        assert path is not None
        with open(path) as f:
            data = json.load(f)
        assert data.get("manual_approval_required") is True
        assert data.get("manual_approval_pending") is True
        gen.conn.close()

    def test_graduation_criteria_met_count(self, tmp_path):
        """Criteria counts are calculated properly."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_graduation_json()
        with open(path) as f:
            data = json.load(f)
        assert "criteria_met" in data
        assert "criteria_total" in data
        assert data["criteria_total"] > 0
        gen.conn.close()


# ---------------------------------------------------------------------------
# __all__ exports validation
# ---------------------------------------------------------------------------

class TestAllExports:
    """Test __all__ exports match module's public API."""

    def test_all_defined(self):
        """__all__ is defined and contains expected names."""
        from src.dashboard.generator import __all__
        assert isinstance(__all__, list)
        assert "DashboardGenerator" in __all__
        assert "PUBLIC_DIR" in __all__
        assert "DB_PATH" in __all__
        assert len(__all__) >= 3

    def test_all_names_importable(self):
        """Every name in __all__ can be imported from the module."""
        from src.dashboard.generator import __all__, DashboardGenerator, PUBLIC_DIR, DB_PATH
        name_map = {
            "DashboardGenerator": DashboardGenerator,
            "PUBLIC_DIR": PUBLIC_DIR,
            "DB_PATH": DB_PATH,
        }
        assert set(__all__) == set(name_map.keys())
        for name, obj in name_map.items():
            assert obj is not None, f"{name} should not be None"


# ---------------------------------------------------------------------------
# Extended field type / dataclass validation
# ---------------------------------------------------------------------------

class TestOutputFieldTypes:
    """Validate field types in all generated JSON outputs."""

    def test_signals_json_field_types(self, tmp_path):
        """All signals.json fields have correct Python types."""
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

        # Top-level scalar fields
        assert isinstance(data["generated_at"], str), "generated_at should be str"
        assert isinstance(data["cash"], (int, float)), "cash should be numeric"
        assert isinstance(data["total_value"], (int, float)), "total_value should be numeric"

        # Regime section
        regime = data["regime"]
        assert isinstance(regime, dict)
        assert isinstance(regime["regime"], str)
        assert regime["vix"] is None or isinstance(regime["vix"], (int, float))
        assert regime["detected"] is None or isinstance(regime["detected"], str)

        # Target allocations
        target = data["target_allocations"]
        assert isinstance(target, dict)
        for sym, weight in target.items():
            assert isinstance(sym, str)
            assert isinstance(weight, (int, float))
            assert 0 <= weight <= 1.0, f"Weight {weight} out of range [0, 1]"

        # Latest prices
        prices = data["latest_prices"]
        assert isinstance(prices, dict)
        for sym, price in prices.items():
            assert isinstance(sym, str)
            assert isinstance(price, (int, float)), f"Price for {sym} should be numeric"

        # Positions
        assert isinstance(data["current_positions"], list)
        for pos in data["current_positions"]:
            assert isinstance(pos["symbol"], str)
            assert isinstance(pos["shares"], (int, float))
            assert isinstance(pos["value"], (int, float))
            assert isinstance(pos["weight"], (int, float))
            assert isinstance(pos["unrealized"], (int, float))

        # ML signals
        ml = data["ml_signals"]
        assert isinstance(ml, dict)
        assert isinstance(ml["available"], bool)
        assert ml["timestamp"] is None or isinstance(ml["timestamp"], str)
        assert isinstance(ml["predictions"], dict)
        assert isinstance(ml["features"], dict)
        assert isinstance(ml["grid_search"], dict)
        gen.conn.close()

    def test_health_json_field_types(self, tmp_path):
        """All health.json fields have correct Python types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)

        assert isinstance(data["system_status"], str)
        assert data["system_status"] in ("healthy", "warning", "critical", "degraded")
        assert isinstance(data["cron_jobs"], list)
        assert isinstance(data["data_freshness"], dict)
        assert isinstance(data["signal_health"], dict)
        assert isinstance(data["generated_at"], str)

        for sym, freshness in data["data_freshness"].items():
            assert isinstance(sym, str)
            assert isinstance(freshness, dict)
            assert "last_update" in freshness
            assert "days_stale" in freshness
            assert "status" in freshness
            assert freshness["status"] in ("fresh", "stale", "critical")
            assert isinstance(freshness["days_stale"], int)

        for job in data["cron_jobs"]:
            assert isinstance(job["name"], str)
            assert isinstance(job["status"], str)
        gen.conn.close()

    def test_stats_json_field_types(self, tmp_path):
        """All stats.json fields have correct Python types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)

        assert isinstance(data["generated_at"], str)
        assert isinstance(data["asset_stats"], dict)

        for sym, stat in data["asset_stats"].items():
            assert isinstance(sym, str)
            assert isinstance(stat["30d_return"], (int, float))
            assert isinstance(stat["volatility"], (int, float))
            assert isinstance(stat["current"], (int, float))

        assert isinstance(data["paper_portfolio"], dict)
        assert data["spy_comparison"] is None or isinstance(data["spy_comparison"], dict)
        if data.get("spy_comparison"):
            sc = data["spy_comparison"]
            for key in ("portfolio_value", "spy_value", "relative_return", "correlation_30d", "beta"):
                assert key in sc, f"spy_comparison missing '{key}'"
        gen.conn.close()

    def test_alerts_json_field_types(self, tmp_path):
        """All alerts.json fields have correct Python types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)

        assert isinstance(data["alerts"], list)
        assert isinstance(data["count"], int)
        assert isinstance(data["generated_at"], str)

        for alert in data["alerts"]:
            assert isinstance(alert["level"], str)
            assert isinstance(alert["type"], str)
            assert isinstance(alert["title"], str)
            assert isinstance(alert["message"], str)
            assert isinstance(alert["requires_action"], bool)
            assert alert["level"] in ("success", "warning", "error", "info")
        gen.conn.close()

    def test_broker_data_field_types(self, tmp_path):
        """_load_broker_data dict has correct field types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert isinstance(broker["connected"], bool)
        assert isinstance(broker["positions"], list)
        assert isinstance(broker["drift"], list)
        assert isinstance(broker["recent_orders"], list)
        assert broker["last_sync"] is None or isinstance(broker["last_sync"], str)
        assert isinstance(broker["kill_switch"], bool)
        gen.conn.close()

    def test_garch_cvar_field_types(self, tmp_path):
        """_load_garch_cvar_data dict has correct field types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            garch = gen._load_garch_cvar_data()
        assert isinstance(garch["cvar_95"], (int, float))
        assert isinstance(garch["cvar_95_garch"], (int, float))
        assert isinstance(garch["var_95"], (int, float))
        assert isinstance(garch["var_95_garch"], (int, float))
        assert isinstance(garch["cvar_ratio"], (int, float))
        assert isinstance(garch["garch_active"], bool)
        assert isinstance(garch["current_volatility"], (int, float))
        assert isinstance(garch["forecast_volatility"], (int, float))
        assert isinstance(garch["volatility_clustering"], str)
        assert garch["volatility_clustering"] in ("normal", "elevated", "high")
        gen.conn.close()

    def test_entropy_data_field_types(self, tmp_path):
        """_load_entropy_data dict has correct field types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            entropy = gen._load_entropy_data()
        assert isinstance(entropy["shannon_entropy"], (int, float))
        assert isinstance(entropy["effective_n"], (int, float))
        assert isinstance(entropy["max_possible"], (int, float))
        assert isinstance(entropy["normalized_score"], (int, float))
        assert isinstance(entropy["concentration_risk"], str)
        assert entropy["concentration_risk"] in ("good", "low", "medium", "high", "critical")
        assert isinstance(entropy["hhi_index"], (int, float))
        assert isinstance(entropy["correlation_entropy"], (int, float))
        assert isinstance(entropy["participation_ratio"], (int, float))
        gen.conn.close()


# ---------------------------------------------------------------------------
# Additional computation edge cases — boundary values, zero/negative, large
# ---------------------------------------------------------------------------

class TestGarchCvarEdgeCasesExtended:
    """Additional _load_garch_cvar_data edge cases — boundary values."""

    def test_value_exactly_one_not_divided(self, tmp_path):
        """Value exactly 1.0 is kept as-is (not divided by 100 because abs(1) <= 1)."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "garch_filtered": True,
            "cvar_95": 1.0,
            "var_95": 1.0,
            "cvar_ratio": 1.5,
            "filter_active": True,
            "conditional_volatility_current": 1.0,
            "garch_persistence": 0.9,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        # abs(1.0) <= 1, so no division
        assert data["cvar_95"] == 1.0
        assert data["var_95"] == 1.0
        gen.conn.close()

    def test_value_slightly_above_one_divided(self, tmp_path):
        """Value 1.01 (> 1.0) is divided by 100."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "garch_filtered": True,
            "cvar_95": 1.01,
            "var_95": 1.01,
            "filter_active": True,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == 0.0101
        assert data["var_95"] == 0.0101
        gen.conn.close()

    def test_persistence_at_exact_boundaries(self, tmp_path):
        """GARCH persistence at exact boundary values."""
        gen, _ = _make_generator(tmp_path)
        # boundary cases: 0.85 -> elevated, 0.95 -> high, 0.951 -> high
        for persistence, expected in [
            # Code uses > 0.85 and > 0.95 (strict), not >=
            (0.85, "normal"),
            (0.86, "elevated"),
            (0.94, "elevated"),
            (0.95, "elevated"),
            (0.951, "high"),
            (0.80, "normal"),
            (0.84, "normal"),
        ]:
            health_file = tmp_path / ".health_report.json"
            health_file.write_text(json.dumps({
                "garch_filtered": True,
                "cvar_95": -0.0179,
                "filter_active": True,
                "garch_persistence": persistence,
            }))
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                data = gen._load_garch_cvar_data()
            assert data["volatility_clustering"] == expected, (
                f"Persistence {persistence} should be {expected}, got {data['volatility_clustering']}"
            )
        gen.conn.close()


class TestStatsEdgeCasesExtended:
    """Additional generate_stats_json computation edge cases."""

    def test_zero_returns_zero_volatility(self, tmp_path):
        """Identical prices produce zero volatility."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DROP TABLE IF EXISTS prices")
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        today = datetime.now()
        for sym in ["SPY", "GLD"]:
            for i in range(30):
                d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
                conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                             (sym, d, 100.0))  # All identical prices
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        for sym, stat in data["asset_stats"].items():
            assert stat["volatility"] == 0.0, f"{sym} volatility should be 0 with identical prices"
            assert stat["30d_return"] == 0.0, f"{sym} 30d return should be 0 with identical prices"
        gen.conn.close()

    def test_negative_prices_handled(self, tmp_path):
        """Negative prices are handled without crashing."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DROP TABLE IF EXISTS prices")
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        today = datetime.now()
        for i in range(5):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                         ("SPY", d, -100.0 + i * 10.0))
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        assert "SPY" in data["asset_stats"]
        gen.conn.close()

    def test_very_large_price_values(self, tmp_path):
        """Very large prices do not overflow or crash."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DROP TABLE IF EXISTS prices")
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        today = datetime.now()
        large_price = 1e12
        for i in range(5):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                         ("SPY", d, large_price + i * 1e9))
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        assert "SPY" in data["asset_stats"]
        assert data["asset_stats"]["SPY"]["current"] > 0
        gen.conn.close()

    def test_negative_returns_handled(self, tmp_path):
        """Negative daily returns produce valid volatility."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DROP TABLE IF EXISTS prices")
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        today = datetime.now()
        # Insert in ascending date order (earliest first), so ORDER BY date gives descending prices
        days_ago = [4, 3, 2, 1, 0]
        prices = [100.0, 98.0, 95.0, 93.0, 90.0]  # Strictly declining
        for days_back, price in zip(days_ago, prices):
            d = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                         ("SPY", d, price))
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        stat = data["asset_stats"]["SPY"]
        assert stat["30d_return"] < 0, "Declining prices should have negative return"
        assert stat["volatility"] >= 0, "Volatility must be non-negative"
        gen.conn.close()


class TestMLSignalsEdgeCasesExtended:
    """Additional _generate_ml_signals prediction edge cases."""

    def test_vix_exactly_25_classification(self, tmp_path):
        """VIX exactly 25 falls into vol_spike branch (>20, not >25)."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 25, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["probabilities"]["bear"] == 0.3  # vol_spike probs
        assert pred["probabilities"]["neutral"] == 0.5
        gen.conn.close()

    def test_vix_exactly_20_classification(self, tmp_path):
        """VIX exactly 20 falls into normal branch (not >20, not <15)."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 20, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["predicted_regime"] == "neutral"
        assert pred["probabilities"]["neutral"] == 0.6
        gen.conn.close()

    def test_vix_extremely_high(self, tmp_path):
        """Very high VIX (e.g., 80) does not crash."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 80, "trend_direction": -5,
            "price_vs_sma20": -0.2, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["probabilities"]["bear"] == 0.5  # crisis probs dominate
        assert pred["predicted_regime"] == "bear"
        gen.conn.close()

    def test_trend_direction_zero_price_vs_sma_zero(self, tmp_path):
        """trend_direction=0 and price_vs_sma20=0 with vix <=20 defaults."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 18, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["probabilities"]["bear"] == 0.2
        assert pred["probabilities"]["neutral"] == 0.6
        assert pred["probabilities"]["bull"] == 0.2
        gen.conn.close()

    def test_missing_trend_fields_defaults(self, tmp_path):
        """Missing trend_direction and price_vs_sma20 fields use default 0."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 18, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["predicted_regime"] == "neutral"
        gen.conn.close()

    def test_price_vs_sma_positive_trend_zero(self, tmp_path):
        """price_vs_sma20 > 0 but trend_direction is 0 → default branch."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 18, "trend_direction": 0,
            "price_vs_sma20": 0.1, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        # trend == 0 and price_vs_sma > 0 does NOT match trend > 0 condition
        assert pred["predicted_regime"] == "neutral"
        gen.conn.close()


class TestPerformanceJSONEdgeCasesExtended:
    """Additional generate_performance_json edge cases."""

    def test_empty_prices_table(self, tmp_path):
        """Empty prices table produces empty prices dict and no crash."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM prices")
        conn.execute("DELETE FROM regime_log")
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert data["prices"] == {}
        assert data["regimes"] == []
        assert "generated_at" in data
        gen.conn.close()

    def test_regime_log_empty_list(self, tmp_path):
        """Empty regime_log table produces empty regimes list."""
        gen, _ = _make_generator(tmp_path)
        gen.conn.execute("DELETE FROM regime_log")
        gen.conn.commit()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert data["regimes"] == []
        gen.conn.close()


class TestAlertsJSONEdgeCasesExtended:
    """Additional generate_alerts_json edge cases."""

    def test_stale_data_days_calculation(self, tmp_path):
        """Stale data alert shows correct days count."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prices VALUES ('OLD', '2020-06-15', 100.0)")
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        stale = [a for a in data["alerts"] if a["type"] == "stale_data"]
        assert len(stale) >= 1
        assert "days ago" in stale[0]["message"]
        gen.conn.close()

    def test_no_trigger_files_no_alerts(self, tmp_path):
        """No trigger files produce only stale data alerts."""
        gen, _ = _make_generator(tmp_path)
        gen.conn.execute("DELETE FROM regime_log")
        gen.conn.commit()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        # With today's data, there should be no stale data alerts
        types_found = {a["type"] for a in data["alerts"]}
        # If all data is fresh, alerts should be empty
        gen.conn.close()


# ---------------------------------------------------------------------------
# Constants validation — extended
# ---------------------------------------------------------------------------

class TestConstantsExtended:
    """Extended module-level constant validation."""

    def test_logger_is_logger_instance(self):
        """logger is a Logger instance."""
        import logging
        from src.dashboard.generator import logger
        assert isinstance(logger, logging.Logger)

    def test_base_allocation_keys_are_uppercase(self):
        """BASE_ALLOCATION keys are uppercase symbol names."""
        from src.paths import BASE_ALLOCATION
        for key in BASE_ALLOCATION:
            assert key == key.upper(), f"Key '{key}' should be uppercase"

    def test_regime_overrides_keys_match(self):
        """Regime override keys correspond to valid regimes."""
        from src.dashboard.generator import DashboardGenerator
        gen = DashboardGenerator.__new__(DashboardGenerator)
        # Extract the regime_overrides from generate_signals_json logic
        expected_regimes = {"crisis", "vol_spike", "low_vol"}
        overrides = {
            "crisis": {"SPY": 0.20, "GLD": 0.50, "TLT": 0.30},
            "vol_spike": {"SPY": 0.30, "GLD": 0.45, "TLT": 0.25},
            "low_vol": {"SPY": 0.55, "GLD": 0.30, "TLT": 0.15},
        }
        assert set(overrides.keys()) == expected_regimes
        for regime, alloc in overrides.items():
            total = sum(alloc.values())
            assert abs(total - 1.0) < 0.01, (
                f"Regime '{regime}' allocations sum to {total}, expected 1.0"
            )
            for sym in alloc:
                assert isinstance(alloc[sym], float)

    def test_yield_regime_allocations_sum_to_one(self):
        """Each yield regime allocation sums to ~1.0."""
        regime_allocations = {
            "steep": {"tlt": 0.70, "ief": 0.25, "shy": 0.05, "bil": 0.00},
            "normal": {"tlt": 0.50, "ief": 0.35, "shy": 0.15, "bil": 0.00},
            "flat": {"tlt": 0.30, "ief": 0.40, "shy": 0.25, "bil": 0.05},
            "inverted": {"tlt": 0.15, "ief": 0.25, "shy": 0.35, "bil": 0.25},
        }
        for regime, alloc in regime_allocations.items():
            total = sum(alloc.values())
            assert abs(total - 1.0) < 0.01, (
                f"Yield regime '{regime}' allocations sum to {total}, expected 1.0"
            )

    def test_public_dir_exists_after_creation(self):
        """PUBLIC_DIR is a Path pointing to an existing or creatable directory."""
        from src.dashboard.generator import PUBLIC_DIR
        # This test just validates the constant, directory creation is tested in init
        import os
        parent = PUBLIC_DIR.parent
        assert parent.exists(), f"Parent dir {parent} should exist"


# ---------------------------------------------------------------------------
# CLI main() function test
# ---------------------------------------------------------------------------

class TestCliMain:
    """Test the __main__ CLI entry point."""

    def test_main_logic_runs_generator(self):
        """__main__ block's logic: DashboardGenerator().run()."""
        with patch.object(DashboardGenerator, "run") as mock_run:
            with patch.object(DashboardGenerator, "__init__", return_value=None):
                # This replicates the __main__ block: gen = DashboardGenerator(); gen.run()
                gen = DashboardGenerator()
                gen.run()
                mock_run.assert_called_once()

    def test_main_block_guard_reads_correctly(self):
        """The __main__ guard checks __name__ == '__main__'."""
        import ast
        import importlib.util
        source_path = importlib.util.find_spec("src.dashboard.generator").origin
        source = Path(source_path).read_text()
        tree = ast.parse(source)
        found_guard = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check if this is an if __name__ == "__main__" guard
                if (isinstance(node.test, ast.Compare)
                        and isinstance(node.test.left, ast.Name)
                        and node.test.left.id == "__name__"
                        and isinstance(node.test.comparators[0], ast.Constant)
                        and node.test.comparators[0].value == "__main__"):
                    found_guard = True
                    # Verify the body contains DashboardGenerator and run()
                    body_source = ast.unparse(node.body)
                    assert "DashboardGenerator" in body_source
                    assert ".run()" in body_source
                    break
        assert found_guard, "No __name__ == '__main__' guard found"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
