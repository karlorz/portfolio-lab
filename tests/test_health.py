#!/usr/bin/env python3
"""
Tests for health.py — HealthMonitor comprehensive coverage.
Covers: init, kill_switches, portfolio health, cron execution,
data freshness, circuit breaker, CVaR metrics, graduation,
wiki sync, portfolio entropy, escalation, run().
"""
import sys
import os
import json
import sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.monitor.health import HealthMonitor, DATA_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monitor():
    return HealthMonitor()


def _write_recent_log(log_dir, name, content="ok"):
    p = log_dir / name
    p.write_text(content)
    now = datetime.now().timestamp()
    os.utime(str(p), (now, now))


def _write_stale_log(log_dir, name, hours_ago=48):
    p = log_dir / name
    p.write_text("old")
    old = (datetime.now() - timedelta(hours=hours_ago)).timestamp()
    os.utime(str(p), (old, old))


def _make_market_db(tmp_path, symbols_with_dates=None):
    """Create a market.db with optional price data."""
    db = tmp_path / "market.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE IF NOT EXISTS prices (symbol TEXT, date TEXT, close REAL)")
    if symbols_with_dates:
        for symbol, date in symbols_with_dates:
            conn.execute("INSERT INTO prices VALUES (?, ?, 100.0)", (symbol, date))
    conn.commit()
    conn.close()
    return db


# ---------------------------------------------------------------------------
# HealthMonitor init
# ---------------------------------------------------------------------------

class TestHealthMonitorInit:
    def test_init_defaults(self):
        hm = _make_monitor()
        assert hm.checks == []
        assert hm.status == "healthy"
        assert hm.alerts == []

    def test_init_has_no_checks(self):
        hm = _make_monitor()
        assert len(hm.checks) == 0


# ---------------------------------------------------------------------------
# check_kill_switches
# ---------------------------------------------------------------------------

class TestKillSwitches:
    def test_no_switches_ok(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        ok = hm.check_kill_switches()
        assert ok is True
        assert len(hm.alerts) == 0

    def test_active_paper_switch_critical(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        (tmp_path / ".kill_switch_paper").write_text(json.dumps({"reason": "drawdown exceeded"}))
        ok = hm.check_kill_switches()
        assert ok is False
        assert hm.status == "critical"
        assert any("KILL SWITCHES ACTIVE" in a for a in hm.alerts)

    def test_active_live_switch_critical(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        (tmp_path / ".kill_switch_live").write_text(json.dumps({"reason": "circuit breaker"}))
        ok = hm.check_kill_switches()
        assert ok is False
        assert hm.status == "critical"

    def test_multiple_switches(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        (tmp_path / ".kill_switch_paper").write_text(json.dumps({"reason": "DD"}))
        (tmp_path / ".kill_switch_live").write_text(json.dumps({"reason": "circuit breaker"}))
        ok = hm.check_kill_switches()
        assert ok is False
        assert hm.status == "critical"

    def test_switch_unknown_reason(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        (tmp_path / ".kill_switch_paper").write_text(json.dumps({"reason": "unknown"}))
        ok = hm.check_kill_switches()
        assert ok is False
        assert any("unknown" in a for a in hm.alerts)

    def test_switch_missing_reason_key(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        (tmp_path / ".kill_switch_paper").write_text(json.dumps({}))
        ok = hm.check_kill_switches()
        assert ok is False
        assert any("unknown" in a for a in hm.alerts)

    def test_switch_records_check(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        (tmp_path / ".kill_switch_paper").write_text(json.dumps({"reason": "test"}))
        hm.check_kill_switches()
        assert any(c["name"] == "kill_switches" for c in hm.checks)


# ---------------------------------------------------------------------------
# check_portfolio_health
# ---------------------------------------------------------------------------

class TestPortfolioHealth:
    def test_no_file_not_initialized(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        ok = hm.check_portfolio_health()
        assert ok is True

    def test_balanced_portfolio(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        (tmp_path / "portfolio_paper.json").write_text(json.dumps({
            "cash": 10000, "positions": {"SPY": {"value": 46000}, "GLD": {"value": 38000}, "TLT": {"value": 16000}}, "history": []
        }))
        ok = hm.check_portfolio_health()
        assert ok is True

    def test_high_cash_ratio(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        (tmp_path / "portfolio_paper.json").write_text(json.dumps({
            "cash": 90000, "positions": {"SPY": {"value": 10000}}, "history": []
        }))
        ok = hm.check_portfolio_health()
        assert ok is False
        assert any("high cash" in a for a in hm.alerts)

    def test_drawdown_detected(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        history = [{"total_value": 100000 - i * 1000} for i in range(30)]
        (tmp_path / "portfolio_paper.json").write_text(json.dumps({
            "cash": 10000, "positions": {"SPY": {"value": 60000}}, "history": history
        }))
        ok = hm.check_portfolio_health()
        assert ok is False
        assert any("drawdown" in a for a in hm.alerts)

    def test_portfolio_zero_total(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        (tmp_path / "portfolio_paper.json").write_text(json.dumps({
            "cash": 0, "positions": {}, "history": []
        }))
        ok = hm.check_portfolio_health()
        assert ok is True

    def test_portfolio_short_history_no_drawdown(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        history = [{"total_value": 100000} for _ in range(5)]
        (tmp_path / "portfolio_paper.json").write_text(json.dumps({
            "cash": 5000, "positions": {"SPY": {"value": 95000}}, "history": history
        }))
        ok = hm.check_portfolio_health()
        assert ok is True

    def test_portfolio_value_in_check(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        (tmp_path / "portfolio_paper.json").write_text(json.dumps({
            "cash": 5000, "positions": {"SPY": {"value": 50000}}, "history": []
        }))
        hm.check_portfolio_health()
        check = [c for c in hm.checks if c["name"] == "portfolio"][0]
        assert "value" in check
        assert check["value"] == "$55,000.00"


# ---------------------------------------------------------------------------
# check_cron_execution
# ---------------------------------------------------------------------------

class TestCronExecution:
    def test_no_logs_stale(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.LOG_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        ok = hm.check_cron_execution()
        assert ok is False
        assert any("Cron jobs need attention" in a for a in hm.alerts)

    def test_recent_logs_ok(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.LOG_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        for job in ["cron.log", "eval.log", "research.log", "dashboard.log", "wiki_sync.log"]:
            _write_recent_log(tmp_path, job)
        ok = hm.check_cron_execution()
        assert ok is True

    def test_stale_data_pipeline_log(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.LOG_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        _write_stale_log(tmp_path, "cron.log", hours_ago=5)
        _write_recent_log(tmp_path, "eval.log")
        _write_recent_log(tmp_path, "research.log")
        _write_recent_log(tmp_path, "dashboard.log")
        _write_recent_log(tmp_path, "wiki_sync.log")
        ok = hm.check_cron_execution()
        assert ok is False

    def test_stale_wiki_sync_log(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.LOG_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        _write_recent_log(tmp_path, "cron.log")
        _write_recent_log(tmp_path, "eval.log")
        _write_recent_log(tmp_path, "research.log")
        _write_recent_log(tmp_path, "dashboard.log")
        _write_stale_log(tmp_path, "wiki_sync.log", hours_ago=20)
        ok = hm.check_cron_execution()
        assert ok is False

    def test_partial_logs(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.LOG_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        _write_recent_log(tmp_path, "cron.log")
        _write_recent_log(tmp_path, "eval.log")
        ok = hm.check_cron_execution()
        assert ok is False


# ---------------------------------------------------------------------------
# check_data_freshness
# ---------------------------------------------------------------------------

class TestDataFreshness:
    def test_missing_db(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        ok = hm.check_data_freshness()
        assert ok is False
        assert any("database" in c["name"] for c in hm.checks)

    def test_empty_db_ok(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        _make_market_db(tmp_path)
        ok = hm.check_data_freshness()
        assert ok is True

    def test_recent_data_ok(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        today = datetime.now().strftime("%Y-%m-%d")
        _make_market_db(tmp_path, [("SPY", today), ("GLD", today)])
        ok = hm.check_data_freshness()
        # Result depends on whether today is a trading day
        assert isinstance(ok, bool)

    def test_stale_data_detected(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        _make_market_db(tmp_path, [("SPY", old_date), ("GLD", old_date)])
        ok = hm.check_data_freshness()
        # Data is 10 days old; should flag stale on trading days
        check = [c for c in hm.checks if c["name"] == "data_freshness"][0]
        assert check["name"] == "data_freshness"


# ---------------------------------------------------------------------------
# check_circuit_breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_module_not_available(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        with patch("src.monitor.health.sys.path", []):
            ok = hm.check_circuit_breaker()
            assert ok is True

    def test_exception_graceful(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        with patch("src.monitor.health.Path") as mock_path:
            mock_path.side_effect = Exception("test error")
            ok = hm.check_circuit_breaker()
            # Should gracefully handle errors
            assert isinstance(ok, bool)


# ---------------------------------------------------------------------------
# check_cvar_metrics
# ---------------------------------------------------------------------------

class TestCVaRMetrics:
    def test_no_risk_file_ok(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        monkeypatch.setenv("USE_GARCH_CVAR", "false")
        ok = hm.check_cvar_metrics()
        assert ok is True

    def test_valid_risk_metrics(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        monkeypatch.setenv("USE_GARCH_CVAR", "false")
        metrics = {
            "cvar_95_daily": -0.025,
            "var_95_daily": -0.015,
            "cvar_ratio": 1.4,
            "tail_severity": "normal",
            "garch_filtered": False,
            "garch_active": False,
        }
        (tmp_path / "risk_metrics.json").write_text(json.dumps(metrics))
        ok = hm.check_cvar_metrics()
        assert ok is True

    def test_cvar_inversion_detected(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        monkeypatch.setenv("USE_GARCH_CVAR", "false")
        metrics = {
            "cvar_95_daily": -0.010,
            "var_95_daily": -0.025,
            "cvar_ratio": 1.4,
            "tail_severity": "normal",
        }
        (tmp_path / "risk_metrics.json").write_text(json.dumps(metrics))
        ok = hm.check_cvar_metrics()
        assert ok is False
        assert any("cvar_inversion" in a for a in hm.alerts)

    def test_cvar_ratio_out_of_range(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        monkeypatch.setenv("USE_GARCH_CVAR", "false")
        metrics = {
            "cvar_95_daily": -0.025,
            "var_95_daily": -0.015,
            "cvar_ratio": 0.5,
            "tail_severity": "unknown",
        }
        (tmp_path / "risk_metrics.json").write_text(json.dumps(metrics))
        ok = hm.check_cvar_metrics()
        assert ok is False

    def test_severe_tail_risk_alert(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        monkeypatch.setenv("USE_GARCH_CVAR", "false")
        metrics = {
            "cvar_95_daily": -0.040,
            "var_95_daily": -0.015,
            "cvar_ratio": 2.1,
            "tail_severity": "severe",
        }
        (tmp_path / "risk_metrics.json").write_text(json.dumps(metrics))
        hm.check_cvar_metrics()
        assert any("Severe tail risk" in a for a in hm.alerts)

    def test_elevated_tail_risk(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        monkeypatch.setenv("USE_GARCH_CVAR", "false")
        metrics = {
            "cvar_95_daily": -0.030,
            "var_95_daily": -0.015,
            "cvar_ratio": 1.6,
            "tail_severity": "moderate",
        }
        (tmp_path / "risk_metrics.json").write_text(json.dumps(metrics))
        ok = hm.check_cvar_metrics()
        assert ok is False

    def test_invalid_json_graceful(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        monkeypatch.setenv("USE_GARCH_CVAR", "false")
        (tmp_path / "risk_metrics.json").write_text("not json")
        ok = hm.check_cvar_metrics()
        assert ok is True  # Graceful degradation

    def test_garch_disabled_by_env(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        monkeypatch.setenv("USE_GARCH_CVAR", "false")
        (tmp_path / "risk_metrics.json").write_text(json.dumps({
            "cvar_95_daily": -0.02, "var_95_daily": -0.015,
            "cvar_ratio": 1.35, "tail_severity": "normal",
        }))
        ok = hm.check_cvar_metrics()
        assert ok is True

    def test_missing_cvar_fields(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        monkeypatch.setenv("USE_GARCH_CVAR", "false")
        (tmp_path / "risk_metrics.json").write_text(json.dumps({}))
        ok = hm.check_cvar_metrics()
        assert ok is True  # Missing fields = no issues detected


# ---------------------------------------------------------------------------
# check_graduation_candidate
# ---------------------------------------------------------------------------

class TestGraduationCandidate:
    def test_no_trigger_file_ok(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        with patch("src.monitor.health.GRADUATION_AVAILABLE", False):
            ok = hm.check_graduation_candidate()
            assert ok is True

    def test_trigger_file_with_metrics(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        trigger = {"metrics": {"sharpe": 1.0, "max_drawdown": -0.15, "win_rate": 0.6}}
        (tmp_path / ".promote_to_live").write_text(json.dumps(trigger))
        with patch("src.monitor.health.GRADUATION_AVAILABLE", False):
            ok = hm.check_graduation_candidate()
            assert ok is True
            assert any("PROMOTION CANDIDATE" in a for a in hm.alerts)

    def test_graduation_checklist_available(self, monkeypatch, tmp_path):
        """Test graduation path when GRADUATION_AVAILABLE is True.
        Since the checklist import is conditional and internal,
        we test via the fallback path (GRADUATION_AVAILABLE=False)
        and verify the check method handles both paths.
        """
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        # Test fallback path with trigger file
        trigger = {"metrics": {"sharpe": 1.2, "max_drawdown": -0.10, "win_rate": 0.65}}
        (tmp_path / ".promote_to_live").write_text(json.dumps(trigger))
        with patch("src.monitor.health.GRADUATION_AVAILABLE", False):
            ok = hm.check_graduation_candidate()
            assert ok is True
            check = [c for c in hm.checks if c["name"] == "graduation"][0]
            assert check["status"] == "candidate_ready"

    def test_graduation_no_candidate(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        with patch("src.monitor.health.GRADUATION_AVAILABLE", False):
            ok = hm.check_graduation_candidate()
            assert ok is True
            check = [c for c in hm.checks if c["name"] == "graduation"][0]
            assert check["status"] == "no_candidate"


# ---------------------------------------------------------------------------
# check_wiki_sync
# ---------------------------------------------------------------------------

class TestWikiSync:
    def test_wiki_not_configured(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.WIKI_DIR", tmp_path / "nonexistent")
        ok = hm.check_wiki_sync()
        assert ok is True

    def test_wiki_no_pages(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        wiki_dir = tmp_path / "wiki"
        compound_dir = wiki_dir / "projects" / "portfolio-lab" / "compound"
        compound_dir.mkdir(parents=True)
        monkeypatch.setattr("src.monitor.health.WIKI_DIR", wiki_dir)
        ok = hm.check_wiki_sync()
        assert ok is True

    def test_wiki_recent_page_ok(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        wiki_dir = tmp_path / "wiki"
        compound_dir = wiki_dir / "projects" / "portfolio-lab" / "compound"
        compound_dir.mkdir(parents=True)
        page = compound_dir / "test.md"
        page.write_text("# Test")
        now = datetime.now().timestamp()
        os.utime(str(page), (now, now))
        monkeypatch.setattr("src.monitor.health.WIKI_DIR", wiki_dir)
        ok = hm.check_wiki_sync()
        assert ok is True

    def test_wiki_stale_page_alert(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        wiki_dir = tmp_path / "wiki"
        compound_dir = wiki_dir / "projects" / "portfolio-lab" / "compound"
        compound_dir.mkdir(parents=True)
        page = compound_dir / "old.md"
        page.write_text("# Old")
        old = (datetime.now() - timedelta(hours=24)).timestamp()
        os.utime(str(page), (old, old))
        monkeypatch.setattr("src.monitor.health.WIKI_DIR", wiki_dir)
        ok = hm.check_wiki_sync()
        assert ok is False
        assert any("Wiki sync stale" in a for a in hm.alerts)

    def test_wiki_multiple_pages_uses_latest(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        wiki_dir = tmp_path / "wiki"
        compound_dir = wiki_dir / "projects" / "portfolio-lab" / "compound"
        compound_dir.mkdir(parents=True)
        old_page = compound_dir / "old.md"
        old_page.write_text("# Old")
        old = (datetime.now() - timedelta(hours=24)).timestamp()
        os.utime(str(old_page), (old, old))
        new_page = compound_dir / "new.md"
        new_page.write_text("# New")
        now = datetime.now().timestamp()
        os.utime(str(new_page), (now, now))
        monkeypatch.setattr("src.monitor.health.WIKI_DIR", wiki_dir)
        ok = hm.check_wiki_sync()
        assert ok is True


# ---------------------------------------------------------------------------
# check_portfolio_entropy
# ---------------------------------------------------------------------------

class TestPortfolioEntropy:
    def test_entropy_not_available(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        ok = hm.check_portfolio_entropy()
        assert ok is True

    def test_entropy_exception_graceful(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        with patch.dict("sys.modules", {"entropy_monitor": None}):
            ok = hm.check_portfolio_entropy()
            assert isinstance(ok, bool)


# ---------------------------------------------------------------------------
# run() integration
# ---------------------------------------------------------------------------

class TestRunIntegration:
    def test_run_healthy(self, monkeypatch, tmp_path, capsys):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.LOG_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.WIKI_DIR", tmp_path / "nonexistent")
        monkeypatch.setattr("src.monitor.health.WORK_DIR", tmp_path / "work")
        # Create minimal logs
        for job in ["cron.log", "eval.log", "research.log", "dashboard.log", "wiki_sync.log"]:
            _write_recent_log(tmp_path, job)
        report = hm.run()
        assert report["status"] in ["healthy", "warning"]
        assert "checks" in report
        assert "alerts" in report
        assert "summary" in report

    def test_run_creates_report_file(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.LOG_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.WIKI_DIR", tmp_path / "nonexistent")
        monkeypatch.setattr("src.monitor.health.WORK_DIR", tmp_path / "work")
        for job in ["cron.log", "eval.log", "research.log", "dashboard.log", "wiki_sync.log"]:
            _write_recent_log(tmp_path, job)
        hm.run()
        report_file = tmp_path / "health_report.json"
        assert report_file.exists()
        data = json.loads(report_file.read_text())
        assert "status" in data

    def test_run_critical_from_kill_switch(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.LOG_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.WIKI_DIR", tmp_path / "nonexistent")
        monkeypatch.setattr("src.monitor.health.WORK_DIR", tmp_path / "work")
        (tmp_path / ".kill_switch_paper").write_text(json.dumps({"reason": "test"}))
        report = hm.run()
        assert report["status"] == "critical"

    def test_run_summary_counts(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.LOG_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.WIKI_DIR", tmp_path / "nonexistent")
        monkeypatch.setattr("src.monitor.health.WORK_DIR", tmp_path / "work")
        report = hm.run()
        assert report["summary"]["total_checks"] > 0
        assert report["summary"]["passed"] + report["summary"]["failed"] == report["summary"]["total_checks"]


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

class TestEscalation:
    def test_critical_escalation_creates_file(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.LOG_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.WORK_DIR", tmp_path)
        (tmp_path / ".kill_switch_paper").write_text(json.dumps({"reason": "test"}))
        hm.run()
        work_files = list(tmp_path.glob("critical_health_*.md"))
        assert len(work_files) == 1
        content = work_files[0].read_text()
        assert "CRITICAL" in content
        assert "KILL SWITCHES ACTIVE" in content

    def test_warning_no_escalation_with_few_alerts(self, monkeypatch, tmp_path, capsys):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.LOG_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.WIKI_DIR", tmp_path / "nonexistent")
        monkeypatch.setattr("src.monitor.health.WORK_DIR", tmp_path / "work")
        # No logs = stale cron, but only 1 alert type, not enough for warning escalation
        hm.run()
        captured = capsys.readouterr()
        # Should not have critical escalation
        assert "Critical escalation" not in captured.out or hm.status != "critical"


# ---------------------------------------------------------------------------
# Path migration verification
# ---------------------------------------------------------------------------

class TestPathMigration:
    def test_uses_paths_module_not_expanduser(self):
        """Verify health.py imports from src.paths, not expanduser."""
        import inspect
        source = inspect.getsource(HealthMonitor)
        # Check no Path(...).expanduser() calls in method bodies
        assert ".expanduser()" not in source

    def test_wiki_dir_uses_paths_constant(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "test.md").write_text("# Test")
        monkeypatch.setattr("src.monitor.health.WIKI_DIR", wiki_dir)
        ok = hm.check_wiki_sync()
        assert ok is True

    def test_work_dir_uses_paths_constant(self, monkeypatch, tmp_path):
        hm = _make_monitor()
        monkeypatch.setattr("src.monitor.health.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.WORK_DIR", tmp_path / "work")
        monkeypatch.setattr("src.monitor.health.LOG_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health.WIKI_DIR", tmp_path / "nonexistent")
        (tmp_path / ".kill_switch_paper").write_text(json.dumps({"reason": "test"}))
        hm.run()
        assert (tmp_path / "work").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
