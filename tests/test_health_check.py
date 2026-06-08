"""Tests for health check module."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.monitor.health_check import (
    run_health_check,
    _check_data_freshness,
    _check_circuit_breaker,
    _compute_system_status,
    HEALTH_PATH,
)


class TestCheckDataFreshness:
    """Test data freshness checks."""

    def test_missing_prices(self, tmp_path, monkeypatch):
        """Missing prices.json should report missing status."""
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", tmp_path)
        freshness = _check_data_freshness()
        assert freshness["prices"]["status"] == "missing"

    def test_fresh_prices(self, tmp_path, monkeypatch):
        """Recently modified prices.json should report ok."""
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", tmp_path)
        (tmp_path / "prices.json").write_text("{}")
        freshness = _check_data_freshness()
        assert freshness["prices"]["status"] == "ok"
        assert freshness["prices"]["age_hours"] < 1

    def test_stale_prices(self, tmp_path, monkeypatch):
        """Old prices.json should report stale."""
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", tmp_path)
        prices_file = tmp_path / "prices.json"
        prices_file.write_text("{}")
        # Set mtime to 48 hours ago
        old_time = time.time() - 48 * 3600
        import os
        os.utime(prices_file, (old_time, old_time))
        freshness = _check_data_freshness()
        assert freshness["prices"]["status"] == "stale"

    def test_missing_signals(self, tmp_path, monkeypatch):
        """Missing signals.json should report missing."""
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", tmp_path)
        freshness = _check_data_freshness()
        assert freshness["signals"]["status"] == "missing"

    def test_hermes_cron_error_degrades_cron_check(self, tmp_path, monkeypatch):
        """Hermes scheduler errors should be included in health cron checks."""
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", tmp_path)
        (tmp_path / "cron_status.json").write_text(json.dumps({
            "jobs": [{"name": "portfolio-lab-data", "status": "ok"}]
        }))
        hermes_jobs = tmp_path / "hermes_jobs.json"
        hermes_jobs.write_text(json.dumps({
            "jobs": [
                {
                    "id": "bad-job",
                    "name": "portfolio-lab-health",
                    "last_status": "error",
                    "last_error": "Script exited with code 1",
                    "state": "scheduled",
                    "enabled": True,
                }
            ]
        }))
        monkeypatch.setenv("HERMES_CRON_JOBS_PATH", str(hermes_jobs))

        freshness = _check_data_freshness()

        assert freshness["cron"]["status"] == "degraded"
        assert freshness["cron"]["failed_jobs"] == 1
        assert freshness["cron"]["backends"]["hermes"]["failed_jobs"] == 1
        assert freshness["cron"]["jobs"][0]["backend"] == "local"
        assert freshness["cron"]["jobs"][1]["backend"] == "hermes"

    def test_missing_hermes_cron_state_warns(self, tmp_path, monkeypatch):
        """Missing configured Hermes state should be a warning, not a crash."""
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", tmp_path)
        (tmp_path / "cron_status.json").write_text(json.dumps({
            "jobs": [{"name": "portfolio-lab-data", "status": "ok"}]
        }))
        monkeypatch.setenv("HERMES_CRON_JOBS_PATH", str(tmp_path / "missing-jobs.json"))

        freshness = _check_data_freshness()

        assert freshness["cron"]["status"] == "warning"
        assert freshness["cron"]["backends"]["hermes"]["status"] == "unavailable"
        assert "missing-jobs.json" in freshness["cron"]["backends"]["hermes"]["source"]


class TestCheckCircuitBreaker:
    """Test circuit breaker state check."""

    def test_circuit_breaker_ok(self):
        """Closed circuit breaker should report ok."""
        result = _check_circuit_breaker()
        assert result["status"] in ("ok", "unavailable")
        if result["status"] == "ok":
            assert result["state"] == "closed"


class TestComputeSystemStatus:
    """Test overall status derivation."""

    def test_all_ok(self):
        """All ok checks → ok status."""
        checks = {
            "prices": {"status": "ok"},
            "signals": {"status": "ok"},
        }
        circuit = {"status": "ok"}
        assert _compute_system_status(checks, circuit) == "ok"

    def test_stale_data(self):
        """Stale data → warning status."""
        checks = {
            "prices": {"status": "stale"},
            "signals": {"status": "ok"},
        }
        circuit = {"status": "ok"}
        assert _compute_system_status(checks, circuit) == "warning"

    def test_missing_data(self):
        """Missing data → degraded status."""
        checks = {
            "prices": {"status": "missing"},
            "signals": {"status": "ok"},
        }
        circuit = {"status": "ok"}
        assert _compute_system_status(checks, circuit) == "degraded"

    def test_circuit_breaker_degraded(self):
        """Open circuit breaker → warning status."""
        checks = {
            "prices": {"status": "ok"},
            "signals": {"status": "ok"},
        }
        circuit = {"status": "degraded"}
        assert _compute_system_status(checks, circuit) == "warning"

    def test_warning_component(self):
        """Explicit warning components should produce warning system status."""
        checks = {
            "prices": {"status": "ok"},
            "signals": {"status": "ok"},
            "cron": {"status": "warning"},
        }
        circuit = {"status": "ok"}
        assert _compute_system_status(checks, circuit) == "warning"


class TestRunHealthCheck:
    """Test full health check execution."""

    def test_produces_report(self, tmp_path, monkeypatch):
        """run_health_check should return a structured report."""
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.HEALTH_PATH", tmp_path / "health.json")

        # Create minimal data files
        (tmp_path / "prices.json").write_text("{}")
        (tmp_path / "signals.json").write_text("{}")
        (tmp_path / "cron_status.json").write_text('{"jobs": []}')

        report = run_health_check()
        assert "status" in report
        assert "timestamp" in report
        assert "checks" in report
        assert "service" in report
        assert report["service"] == "portfolio-lab"

    def test_writes_health_json(self, tmp_path, monkeypatch):
        """run_health_check should write health.json to disk."""
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
        health_path = tmp_path / "health.json"
        monkeypatch.setattr("src.monitor.health_check.HEALTH_PATH", health_path)

        (tmp_path / "prices.json").write_text("{}")
        (tmp_path / "signals.json").write_text("{}")
        (tmp_path / "cron_status.json").write_text('{"jobs": []}')

        run_health_check()
        assert health_path.exists()
        data = json.loads(health_path.read_text())
        assert "status" in data
