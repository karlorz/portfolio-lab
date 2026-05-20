#!/usr/bin/env python3
"""
Tests for health.py — HealthMonitor init, kill_switches, portfolio health,
cron execution, and data freshness checks.
"""
import sys
import os
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch

from src.monitor.health import HealthMonitor, DATA_DIR


# ---------------------------------------------------------------------------
# HealthMonitor init
# ---------------------------------------------------------------------------

class TestHealthMonitorInit:
    def test_init_defaults(self):
        hm = HealthMonitor()
        assert hm.checks == []
        assert hm.status == "healthy"
        assert hm.alerts == []


# ---------------------------------------------------------------------------
# check_kill_switches
# ---------------------------------------------------------------------------

class TestKillSwitches:
    def test_no_switches_ok(self, monkeypatch, tmp_path):
        hm = HealthMonitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        ok = hm.check_kill_switches()
        assert ok is True
        assert len(hm.alerts) == 0

    def test_active_switch_critical(self, monkeypatch, tmp_path):
        hm = HealthMonitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        (tmp_path / ".kill_switch_paper").write_text(json.dumps({"reason": "drawdown exceeded"}))
        ok = hm.check_kill_switches()
        assert ok is False
        assert hm.status == "critical"
        assert any("KILL SWITCHES ACTIVE" in a for a in hm.alerts)

    def test_multiple_switches(self, monkeypatch, tmp_path):
        hm = HealthMonitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        (tmp_path / ".kill_switch_paper").write_text(json.dumps({"reason": "DD"}))
        (tmp_path / ".kill_switch_live").write_text(json.dumps({"reason": "circuit breaker"}))
        ok = hm.check_kill_switches()
        assert ok is False
        assert hm.status == "critical"


# ---------------------------------------------------------------------------
# check_portfolio_health
# ---------------------------------------------------------------------------

class TestPortfolioHealth:
    def test_no_file_not_initialized(self, monkeypatch, tmp_path):
        hm = HealthMonitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        ok = hm.check_portfolio_health()
        assert ok is True  # Not an error, just not started

    def test_balanced_portfolio(self, monkeypatch, tmp_path):
        hm = HealthMonitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        (tmp_path / "portfolio_paper.json").write_text(json.dumps({"cash": 10000, "positions": {"SPY": {"value": 46000}, "GLD": {"value": 38000}, "TLT": {"value": 16000}}, "history": []}))
        ok = hm.check_portfolio_health()
        assert ok is True  # Balanced 46/38/16 with reasonable cash

    def test_high_cash_ratio(self, monkeypatch, tmp_path):
        hm = HealthMonitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        (tmp_path / "portfolio_paper.json").write_text(json.dumps({"cash": 90000, "positions": {"SPY": {"value": 10000}}, "history": []}))
        ok = hm.check_portfolio_health()
        assert ok is False
        assert any("high cash" in a for a in hm.alerts)

    def test_drawdown_detected(self, monkeypatch, tmp_path):
        hm = HealthMonitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        history = [{"total_value": 100000 - i * 1000} for i in range(30)]
        (tmp_path / "portfolio_paper.json").write_text(json.dumps({"cash": 10000, "positions": {"SPY": {"value": 60000}}, "history": history}))
        ok = hm.check_portfolio_health()
        # Drawdown from 100k to 71k = ~29%
        assert ok is False
        assert any("drawdown" in a for a in hm.alerts)


# ---------------------------------------------------------------------------
# check_cron_execution
# ---------------------------------------------------------------------------

class TestCronExecution:
    def test_no_logs_stale(self, monkeypatch, tmp_path):
        hm = HealthMonitor()
        monkeypatch.setattr("src.monitor.health.LOG_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        ok = hm.check_cron_execution()
        assert ok is False
        assert any("Cron jobs need attention" in a for a in hm.alerts)

    def test_recent_logs_ok(self, monkeypatch, tmp_path):
        hm = HealthMonitor()
        monkeypatch.setattr("src.monitor.health.LOG_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        now = datetime.now()
        for job in ["cron.log", "eval.log", "research.log", "dashboard.log", "wiki_sync.log"]:
            p = tmp_path / job
            p.write_text("ok")
            # Set mtime to now (recent)
            os.utime(str(p), (now.timestamp(), now.timestamp()))
        ok = hm.check_cron_execution()
        assert ok is True


# ---------------------------------------------------------------------------
# check_data_freshness
# ---------------------------------------------------------------------------

class TestDataFreshness:
    def test_missing_db(self, monkeypatch, tmp_path):
        hm = HealthMonitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        ok = hm.check_data_freshness()
        assert ok is False
        assert any("database" in c["name"] for c in hm.checks)

    def test_empty_db_ok(self, monkeypatch, tmp_path):
        import sqlite3
        hm = HealthMonitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        db = tmp_path / "market.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, count INTEGER)")
        conn.commit()
        conn.close()
        ok = hm.check_data_freshness()
        assert ok is True  # Empty table = nothing stale


# ---------------------------------------------------------------------------
# check_circuit_breaker missing module
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_module_not_available(self, monkeypatch, tmp_path):
        hm = HealthMonitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        # Circuit breaker module not importable → graceful failure
        with patch("src.monitor.health.sys.path", []):
            ok = hm.check_circuit_breaker()
            assert ok is True  # Graceful degradation
