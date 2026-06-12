"""Tests for health check module."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.monitor.hermes_cron import load_hermes_portfolio_cron_jobs
from src.monitor.health_check import (
    run_health_check,
    _check_data_freshness,
    _check_circuit_breaker,
    _check_fred_md_cache,
    _compute_system_status,
    HEALTH_PATH,
)
from src.monitor.fred_readiness import assess_fred_readiness, resolve_fred_operating_mode


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

    def test_unreadable_hermes_cron_state_warns_without_crashing(self):
        """Permission-denied Hermes state should degrade health, not crash CI."""
        unreadable = MagicMock(spec=Path)
        unreadable.__str__.return_value = "/root/.hermes/cron/jobs.json"
        unreadable.exists.side_effect = PermissionError("denied")

        jobs, backend = load_hermes_portfolio_cron_jobs(unreadable)

        assert jobs == []
        assert backend["status"] == "unavailable"
        assert "not readable" in backend["reason"]


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

    def test_empty_cache_component_warns(self):
        """Empty cache components should produce warning system status."""
        checks = {
            "prices": {"status": "ok"},
            "signals": {"status": "ok"},
            "fred_md_cache": {"status": "empty"},
        }
        circuit = {"status": "ok"}
        assert _compute_system_status(checks, circuit) == "warning"


class TestRunHealthCheck:
    """Test full health check execution."""

    def test_produces_report(self, tmp_path, monkeypatch):
        """run_health_check should return a structured report."""
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.HEALTH_PATH", tmp_path / "health.json")
        monkeypatch.setattr(
            "src.monitor.health_check._check_fred_md_cache",
            lambda: {"status": "ok", "row_count": 1, "latest_fetched_at": "2026-06-11T00:00:00+00:00"},
        )

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
        assert report["checks"]["data_freshness"]["fred_md_cache"]["status"] == "ok"

    def test_fred_md_cache_check_degrades_without_dependency_crash(self):
        """FRED-MD cache helper should always return a status dictionary."""
        result = _check_fred_md_cache()

        assert "status" in result
        assert "row_count" in result

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

    def test_fred_readiness_reports_live_failure_without_leaking_secret(self, tmp_path, monkeypatch):
        """Live health should fail readiness when FRED is synthetic without exposing key values."""
        monkeypatch.setenv("PORTFOLIO_LAB_MODE", "live")
        monkeypatch.setenv("FRED_API_KEY", "super-secret-fred-key")
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.HEALTH_PATH", tmp_path / "health.json")
        monkeypatch.setattr(
            "src.monitor.health_check._check_fred_md_cache",
            lambda: {
                "status": "unavailable",
                "source_mode": "synthetic",
                "api_key_configured": False,
                "reason": "missing_fred_api_key",
            },
        )

        report = run_health_check()

        readiness = report["checks"]["data_freshness"]["fred_readiness"]
        assert readiness["status"] == "critical"
        assert readiness["ready"] is False
        assert readiness["mode"] == "live"
        assert readiness["reason"] == "missing_fred_api_key"
        assert "FRED_API_KEY" in readiness["remediation"]
        assert "super-secret-fred-key" not in json.dumps(report)


class TestFredReadiness:
    """Test FRED credential readiness policy without live provider calls."""

    def test_resolves_local_mode_as_default_when_no_runtime_mode_is_configured(self, monkeypatch):
        monkeypatch.delenv("PORTFOLIO_LAB_MODE", raising=False)
        monkeypatch.delenv("ALPHALAB_MODE", raising=False)
        monkeypatch.delenv("APP_MODE", raising=False)
        monkeypatch.delenv("CRON_BACKEND", raising=False)

        assert resolve_fred_operating_mode() == "local"

    @pytest.mark.parametrize("mode", ["local", "test"])
    def test_missing_key_is_permissive_degraded_in_local_and_test_modes(self, mode):
        readiness = assess_fred_readiness(
            {"source_mode": "synthetic", "api_key_configured": False, "status": "unavailable"},
            mode=mode,
        )

        assert readiness["status"] == "warning"
        assert readiness["readiness"] == "pass"
        assert readiness["ready"] is True
        assert readiness["enforcement"] == "permissive"
        assert readiness["reason"] == "missing_fred_api_key"

    @pytest.mark.parametrize("mode", ["lab", "paper"])
    def test_missing_key_warns_in_lab_and_paper_modes(self, mode):
        readiness = assess_fred_readiness(
            {"source_mode": "cached", "api_key_configured": False, "status": "ok"},
            mode=mode,
        )

        assert readiness["status"] == "warning"
        assert readiness["readiness"] == "warn"
        assert readiness["ready"] is True
        assert readiness["blocking"] is False
        assert readiness["reason"] == "missing_fred_api_key"

    def test_missing_key_fails_live_mode(self):
        readiness = assess_fred_readiness(
            {"source_mode": "cached", "api_key_configured": False, "status": "ok"},
            mode="live",
        )

        assert readiness["status"] == "critical"
        assert readiness["readiness"] == "fail"
        assert readiness["ready"] is False
        assert readiness["blocking"] is True
        assert readiness["reason"] == "missing_fred_api_key"

    def test_invalid_configured_key_fails_live_mode_without_leaking_secret(self):
        readiness = assess_fred_readiness(
            {
                "source_mode": "synthetic",
                "api_key_configured": True,
                "status": "unavailable",
                "reason": "bad_credentials",
            },
            mode="live",
            env={"FRED_API_KEY": "super-secret-fred-key"},
        )

        assert readiness["status"] == "critical"
        assert readiness["readiness"] == "fail"
        assert readiness["ready"] is False
        assert readiness["reason"] == "invalid_fred_api_key"
        assert "super-secret-fred-key" not in json.dumps(readiness)

    def test_ready_state_redacts_configured_key_value(self):
        readiness = assess_fred_readiness(
            {
                "source_mode": "cached",
                "api_key_configured": True,
                "status": "ok",
                "latest_fetched_at": "2026-06-12T00:00:00+00:00",
            },
            mode="paper",
            env={"FRED_API_KEY": "super-secret-fred-key"},
        )

        assert readiness["status"] == "ok"
        assert readiness["ready"] is True
        assert "super-secret-fred-key" not in json.dumps(readiness)
