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
    check_scheduler_drift,
    _check_data_freshness,
    _check_circuit_breaker,
    _check_fred_md_cache,
    _compute_system_status,
    HEALTH_PATH,
)
from src.monitor.alerting import AlertChannel, AlertLevel
from src.monitor.fred_readiness import assess_fred_readiness, resolve_fred_operating_mode


class TestCheckDataFreshness:
    """Test data freshness checks."""

    def test_missing_prices(self, tmp_path, monkeypatch):
        """Missing prices.json should report missing status."""
        public = tmp_path / "public"
        private = tmp_path / "private"
        public.mkdir()
        private.mkdir()
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", public)
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", private)
        freshness = _check_data_freshness()
        assert freshness["prices"]["status"] == "missing"

    def test_fresh_prices(self, tmp_path, monkeypatch):
        """Recently modified prices.json should report ok."""
        public = tmp_path / "public"
        private = tmp_path / "private"
        public.mkdir()
        private.mkdir()
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", public)
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", private)
        (public / "prices.json").write_text("{}")
        freshness = _check_data_freshness()
        assert freshness["prices"]["status"] == "ok"
        assert freshness["prices"]["age_hours"] < 1

    def test_stale_prices(self, tmp_path, monkeypatch):
        """Old prices.json should report stale."""
        public = tmp_path / "public"
        private = tmp_path / "private"
        public.mkdir()
        private.mkdir()
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", public)
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", private)
        prices_file = public / "prices.json"
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
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path / "private")
        (tmp_path / "private").mkdir()
        freshness = _check_data_freshness()
        assert freshness["signals"]["status"] == "missing"

    def test_signals_private_twin_ok_when_public_missing(self, tmp_path, monkeypatch):
        """Batch HX: private multi-dest twin prevents false signals:missing."""
        public = tmp_path / "public"
        private = tmp_path / "private"
        public.mkdir()
        private.mkdir()
        (private / "signals.json").write_text(
            json.dumps({"target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}})
        )
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", public)
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", private)
        # Outside pytest path would use live fallback; force production-like
        # candidate walk by clearing PYTEST only for the helper's private twin.
        freshness = _check_data_freshness()
        assert freshness["signals"]["status"] == "ok"
        assert freshness["signals"]["age_hours"] is not None
        assert "signals.json" in str(freshness["signals"].get("path") or "")

    def test_signals_ephemeral_public_falls_back_outside_pytest(
        self, tmp_path, monkeypatch
    ):
        """Batch HX: outside pytest, ephemeral PUBLIC_DATA_DIR is not probed."""
        from src.monitor import health_check as hc

        ephemeral = tmp_path / "plab-pytest-public.hx" / "data"
        ephemeral.mkdir(parents=True)
        # No signals in ephemeral — live-like root has TA
        live = tmp_path / "var" / "www" / "portfolio-lab" / "data"
        live.mkdir(parents=True)
        (live / "signals.json").write_text(
            json.dumps({"target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}})
        )
        (live / "prices.json").write_text("{}")
        private = tmp_path / "private"
        private.mkdir()

        monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", ephemeral)
        monkeypatch.setattr(hc, "DATA_DIR", private)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        import src.paths as paths_mod

        monkeypatch.setattr(paths_mod, "DEFAULT_LIVE_PUBLIC_DATA_DIR", live)
        # Treat only plab isolation as ephemeral so pytest-of-root live fixture
        # is still a valid first-root probe (production live WWW is non-tmp).
        monkeypatch.setattr(
            "src.monitor.signal_authority.is_ephemeral_write_path",
            lambda p: "plab-pytest" in str(p or ""),
        )

        freshness = hc._check_data_freshness()
        assert freshness["signals"]["status"] == "ok"
        assert "plab-pytest" not in str(freshness["signals"].get("path") or "")
        assert freshness["prices"]["status"] == "ok"

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
                    # Non-self job: health self-errors are excluded from rollup.
                    "name": "portfolio-lab-eval",
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

    def test_health_self_job_error_does_not_sticky_degrade(self, tmp_path, monkeypatch):
        """Prior portfolio-lab-health error must not fail the next health rollup.

        Tasker/cron_update stamp the health job's own exit into cron_status.json.
        Counting that row as failed_jobs made make health exit 1 forever.
        """
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", tmp_path)
        (tmp_path / "cron_status.json").write_text(json.dumps({
            "backend": "tasker",
            "jobs": [
                {
                    "name": "portfolio-lab-health",
                    "status": "error",
                    "state": "scheduled",
                    "enabled": True,
                    "manual_only": False,
                    "last_run": "2026-07-17T20:30:02.962012+00:00",
                    "backend": "tasker",
                },
                {
                    "name": "portfolio-lab-data",
                    "status": "ok",
                    "state": "scheduled",
                    "enabled": True,
                    "manual_only": False,
                    "backend": "tasker",
                },
            ],
        }))

        freshness = _check_data_freshness()

        assert freshness["cron"]["failed_jobs"] == 0
        assert freshness["cron"]["status"] == "ok"
        assert freshness["cron"]["backends"]["tasker"]["status"] == "ok"
        assert freshness["cron"]["backends"]["tasker"]["failed_jobs"] == 0
        # Raw row still visible for operators before in-process stamp.
        health_row = next(
            j for j in freshness["cron"]["jobs"] if j["name"] == "portfolio-lab-health"
        )
        assert health_row["status"] == "error"

    def test_stamp_self_job_overwrites_prior_error(self, tmp_path, monkeypatch):
        """Successful health run must not publish prior self-error in report jobs."""
        from src.monitor.health_check import _stamp_health_self_job_running_success

        freshness = {
            "cron": {
                "status": "ok",
                "failed_jobs": 0,
                "jobs": [
                    {
                        "name": "portfolio-lab-health",
                        "status": "error",
                        "last_run": "2026-07-20T14:00:00+00:00",
                    },
                    {"name": "portfolio-lab-data", "status": "success"},
                ],
            }
        }
        _stamp_health_self_job_running_success(freshness)
        health_row = next(
            j for j in freshness["cron"]["jobs"] if j["name"] == "portfolio-lab-health"
        )
        assert health_row["status"] == "ok"
        assert health_row.get("self_observation") == "in_process_success_stamp"
        assert health_row.get("prior_status_before_stamp") == "error"

    def test_non_self_job_error_still_degrades_cron(self, tmp_path, monkeypatch):
        """Sibling job errors continue to degrade the cron rollup."""
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", tmp_path)
        (tmp_path / "cron_status.json").write_text(json.dumps({
            "backend": "tasker",
            "jobs": [
                {
                    "name": "portfolio-lab-health",
                    "status": "error",
                    "state": "scheduled",
                    "enabled": True,
                    "backend": "tasker",
                },
                {
                    "name": "portfolio-lab-data",
                    "status": "error",
                    "state": "scheduled",
                    "enabled": True,
                    "backend": "tasker",
                },
            ],
        }))

        freshness = _check_data_freshness()

        assert freshness["cron"]["failed_jobs"] == 1
        assert freshness["cron"]["status"] == "degraded"
        assert freshness["cron"]["backends"]["tasker"]["failed_jobs"] == 1

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

    def test_manual_only_tasker_rows_do_not_degrade_cron_health(self, tmp_path, monkeypatch):
        """Disabled/manual-only tasker rows remain explicit but do not imply active unknown jobs."""
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", tmp_path)
        (tmp_path / "cron_status.json").write_text(json.dumps({
            "backend": "tasker",
            "jobs": [
                {
                    "name": "portfolio-lab-build",
                    "status": "disabled",
                    "state": "manual_only",
                    "enabled": False,
                    "manual_only": True,
                    "last_run": None,
                    "backend": "tasker",
                }
            ],
        }))

        freshness = _check_data_freshness()

        job = freshness["cron"]["jobs"][0]
        assert job["state"] == "manual_only"
        assert job["status"] == "disabled"
        assert freshness["cron"]["status"] == "ok"
        assert freshness["cron"]["backends"]["tasker"]["status"] == "ok"

    def test_enabled_unknown_tasker_rows_degrade_cron_health(self, tmp_path, monkeypatch):
        """Enabled scheduled rows with unmapped status still degrade (true unknown)."""
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", tmp_path)
        (tmp_path / "cron_status.json").write_text(json.dumps({
            "backend": "tasker",
            "jobs": [
                {
                    "name": "portfolio-lab-health",
                    "status": "mystery_state",
                    "state": "scheduled",
                    "enabled": True,
                    "manual_only": False,
                    "last_run": None,
                    "backend": "tasker",
                }
            ],
        }))

        freshness = _check_data_freshness()

        assert freshness["cron"]["jobs"][0]["status"] == "unknown"
        assert freshness["cron"]["status"] in {"warning", "degraded"}
        assert freshness["cron"]["backends"]["tasker"]["status"] == "degraded"

    def test_pending_never_run_weekly_job_does_not_degrade_cron(self, tmp_path, monkeypatch):
        """Batch CI: pending weekly fetch-trends must not force cron degraded."""
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", tmp_path)
        # Batch DT: hermes_cron also searches DATA_DIR for trends artifacts
        import src.monitor.hermes_cron as hc

        monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
        monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", tmp_path)
        (tmp_path / "cron_status.json").write_text(json.dumps({
            "backend": "tasker",
            "jobs": [
                {
                    "name": "portfolio-lab-health",
                    "status": "success",
                    "state": "scheduled",
                    "enabled": True,
                    "manual_only": False,
                    "last_run": "2026-07-21T19:30:00+00:00",
                    "backend": "tasker",
                },
                {
                    "name": "portfolio-lab-fetch-trends",
                    "status": "pending",
                    "state": "scheduled",
                    "enabled": True,
                    "manual_only": False,
                    "last_run": None,
                    "schedule": "20 4 * * 0",
                    "backend": "tasker",
                },
            ],
        }))

        freshness = _check_data_freshness()

        trends = next(
            j for j in freshness["cron"]["jobs"] if j["name"] == "portfolio-lab-fetch-trends"
        )
        assert trends["status"] == "pending"
        tasker = freshness["cron"]["backends"]["tasker"]
        assert tasker.get("unknown_active_jobs") in (None, 0)
        assert tasker.get("pending_never_run_jobs") == 1
        assert tasker["status"] == "ok"
        assert freshness["cron"]["status"] == "ok"

    def test_unreadable_hermes_cron_state_warns_without_crashing(self):
        """Permission-denied Hermes state should degrade health, not crash CI."""
        unreadable = MagicMock(spec=Path)
        unreadable.__str__.return_value = "/root/.hermes/cron/jobs.json"
        unreadable.exists.side_effect = PermissionError("denied")

        jobs, backend = load_hermes_portfolio_cron_jobs(unreadable)

        assert jobs == []
        assert backend["status"] == "unavailable"
        assert "not readable" in backend["reason"]

    def test_scheduler_drift_single_mismatch_warns_without_alert(self, tmp_path):
        """One backend disagreement should persist drift state without paging yet."""
        backends = {
            "tasker": {"status": "ok", "backend": "tasker", "total_jobs": 10},
            "hermes": {"status": "error", "backend": "hermes", "total_jobs": 2},
        }

        with patch("src.monitor.health_check.send_alert") as mock_send:
            drift = check_scheduler_drift(
                backends,
                state_path=tmp_path / "scheduler_drift_state.json",
            )

        assert drift["status"] == "warning"
        assert drift["mismatch"] is True
        assert drift["consecutive_mismatches"] == 1
        assert drift["threshold"] == 2
        assert drift["backend_statuses"] == {"tasker": "ok", "hermes": "error"}
        mock_send.assert_not_called()

    def test_scheduler_drift_second_consecutive_mismatch_sends_halt(self, tmp_path):
        """Two consecutive backend disagreements should fire a CRON_FAILURE HALT."""
        state_path = tmp_path / "scheduler_drift_state.json"
        backends = {
            "tasker": {"status": "ok", "backend": "tasker", "total_jobs": 10},
            "hermes": {"status": "degraded", "backend": "hermes", "total_jobs": 2},
        }

        check_scheduler_drift(backends, state_path=state_path)
        with patch("src.monitor.health_check.send_alert") as mock_send:
            drift = check_scheduler_drift(backends, state_path=state_path)

        assert drift["status"] == "critical"
        assert drift["consecutive_mismatches"] == 2
        mock_send.assert_called_once()
        assert mock_send.call_args.args[0] == AlertChannel.CRON_FAILURE
        assert mock_send.call_args.args[1] == AlertLevel.HALT
        assert "Scheduler backend drift" in mock_send.call_args.args[2]
        assert mock_send.call_args.kwargs["details"]["consecutive_mismatches"] == 2
        assert mock_send.call_args.kwargs["details"]["backend_statuses"] == {
            "tasker": "ok",
            "hermes": "degraded",
        }

    def test_scheduler_drift_agreement_resets_state_and_sends_pass(self, tmp_path):
        """Resolved backend disagreement should reset count and close the incident."""
        state_path = tmp_path / "scheduler_drift_state.json"
        mismatch = {
            "tasker": {"status": "ok", "backend": "tasker", "total_jobs": 10},
            "hermes": {"status": "error", "backend": "hermes", "total_jobs": 3},
        }
        recovered = {
            "tasker": {"status": "ok", "backend": "tasker", "total_jobs": 10},
            "hermes": {"status": "ok", "backend": "hermes", "total_jobs": 3},
        }

        check_scheduler_drift(mismatch, state_path=state_path)
        with patch("src.monitor.health_check.send_alert") as mock_send:
            drift = check_scheduler_drift(recovered, state_path=state_path)

        assert drift["status"] == "ok"
        assert drift["mismatch"] is False
        assert drift["consecutive_mismatches"] == 0
        mock_send.assert_called_once()
        assert mock_send.call_args.args[0] == AlertChannel.CRON_FAILURE
        assert mock_send.call_args.args[1] == AlertLevel.PASS
        assert "Scheduler backends agree" in mock_send.call_args.args[2]

    def test_scheduler_drift_ignores_empty_idle_backend(self, tmp_path):
        """tasker=degraded + hermes=ok with zero jobs is not dual-backend drift."""
        backends = {
            "tasker": {
                "status": "degraded",
                "backend": "tasker",
                "total_jobs": 16,
                "failed_jobs": 3,
            },
            "hermes": {
                "status": "ok",
                "backend": "hermes",
                "total_jobs": 0,
                "failed_jobs": 0,
            },
        }
        state_path = tmp_path / "scheduler_drift_state.json"
        with patch("src.monitor.health_check.send_alert") as mock_send:
            first = check_scheduler_drift(backends, state_path=state_path)
            second = check_scheduler_drift(backends, state_path=state_path)

        assert first["mismatch"] is False
        assert first["status"] == "ok"
        assert second["status"] == "ok"
        assert second["consecutive_mismatches"] == 0
        assert first["compared_backend_statuses"] == {"tasker": "degraded"}
        mock_send.assert_not_called()

    def test_data_freshness_includes_scheduler_drift_summary(self, tmp_path, monkeypatch):
        """Cron health output should expose scheduler drift metadata for dashboards."""
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", tmp_path)
        (tmp_path / "cron_status.json").write_text(json.dumps({
            "backend": "tasker",
            "jobs": [{"name": "portfolio-lab-data", "status": "ok"}],
        }))
        hermes_jobs = tmp_path / "hermes_jobs.json"
        hermes_jobs.write_text(json.dumps({
            "jobs": [
                {
                    "id": "bad-job",
                    "name": "portfolio-lab-data",
                    "last_status": "error",
                    "state": "scheduled",
                    "enabled": True,
                }
            ]
        }))
        monkeypatch.setenv("HERMES_CRON_JOBS_PATH", str(hermes_jobs))

        freshness = _check_data_freshness()

        drift = freshness["cron"]["scheduler_drift"]
        assert drift["status"] == "warning"
        assert drift["consecutive_mismatches"] == 1
        assert drift["backend_statuses"] == {"tasker": "ok", "hermes": "degraded"}


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

    def test_empty_cache_without_fred_key_is_non_blocking(self):
        """Empty FRED cache with no API key is a lab advisory, not overall warning."""
        checks = {
            "prices": {"status": "ok"},
            "signals": {"status": "ok"},
            "fred_md_cache": {"status": "empty", "api_key_configured": False},
            "fred_readiness": {
                "status": "warning",
                "ready": True,
                "blocking": False,
                "reason": "missing_fred_api_key",
            },
        }
        circuit = {"status": "ok"}
        assert _compute_system_status(checks, circuit) == "ok"

    def test_empty_cache_with_key_configured_still_warns(self):
        """Empty cache despite a configured key remains an operator warning."""
        checks = {
            "prices": {"status": "ok"},
            "signals": {"status": "ok"},
            "fred_md_cache": {"status": "empty", "api_key_configured": True},
        }
        circuit = {"status": "ok"}
        assert _compute_system_status(checks, circuit) == "warning"

    def test_blocking_fred_readiness_warns(self):
        checks = {
            "prices": {"status": "ok"},
            "signals": {"status": "ok"},
            "fred_readiness": {
                "status": "warning",
                "ready": False,
                "blocking": True,
                "reason": "invalid_fred_api_key",
            },
        }
        circuit = {"status": "ok"}
        assert _compute_system_status(checks, circuit) == "warning"

    def test_critical_component_elevates_to_critical(self):
        """Active critical components (e.g. HALT kill switch) must not understate as warning."""
        checks = {
            "prices": {"status": "stale"},
            "signals": {"status": "ok"},
            "kill_switch": {"status": "critical"},
        }
        circuit = {"status": "ok"}
        assert _compute_system_status(checks, circuit) == "critical"


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
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", tmp_path / "public")

        (tmp_path / "prices.json").write_text("{}")
        (tmp_path / "signals.json").write_text("{}")
        (tmp_path / "cron_status.json").write_text('{"jobs": []}')

        run_health_check()
        assert health_path.exists()
        data = json.loads(health_path.read_text())
        assert "status" in data

    def test_publishes_health_ops_under_public_data_dir(self, tmp_path, monkeypatch):
        """Monitor report is dual-written to PUBLIC_DATA_DIR/health_ops.json."""
        public = tmp_path / "public"
        public.mkdir()
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", public)
        monkeypatch.setattr("src.monitor.health_check.HEALTH_PATH", tmp_path / "health.json")
        monkeypatch.setattr(
            "src.monitor.health_check._check_fred_md_cache",
            lambda: {"status": "ok", "row_count": 1, "latest_fetched_at": "2026-06-11T00:00:00+00:00"},
        )
        (tmp_path / "prices.json").write_text("{}")
        (tmp_path / "signals.json").write_text("{}")
        (tmp_path / "cron_status.json").write_text('{"jobs": []}')
        (tmp_path / "kill_switch.json").write_text(
            json.dumps(
                {
                    "enabled": True,
                    "level": "halt",
                    "reason": "unresolved_incident:signal_staleness",
                    "source": "incident_lifecycle",
                    "message": "optional unavailable",
                }
            )
        )

        run_health_check()
        ops = public / "health_ops.json"
        assert ops.exists()
        body = json.loads(ops.read_text())
        assert body["checks"]["kill_switch"]["enabled"] is True
        assert body["scope"] == "operational_readiness"

    def test_merges_kill_into_existing_public_dashboard_health(self, tmp_path, monkeypatch):
        """Dashboard health.json must reflect kill halt after health cron."""
        public = tmp_path / "public"
        public.mkdir()
        (public / "health.json").write_text(
            json.dumps(
                {
                    "system_status": "healthy",
                    "generated_at": "2026-07-01T00:00:00",
                    "kill_switch": {"enabled": False, "status": "ok"},
                    "cron_jobs": [],
                }
            )
        )
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", public)
        monkeypatch.setattr("src.monitor.health_check.HEALTH_PATH", tmp_path / "health.json")
        monkeypatch.setattr(
            "src.monitor.health_check._check_fred_md_cache",
            lambda: {"status": "ok", "row_count": 1, "latest_fetched_at": "2026-06-11T00:00:00+00:00"},
        )
        (tmp_path / "prices.json").write_text("{}")
        (tmp_path / "signals.json").write_text("{}")
        (tmp_path / "cron_status.json").write_text('{"jobs": []}')
        (tmp_path / "kill_switch.json").write_text(
            json.dumps(
                {
                    "enabled": True,
                    "level": "halt",
                    "reason": "unresolved_incident:signal_staleness",
                    "source": "incident_lifecycle",
                    "incident_id": "inc-1",
                    "message": "halt active",
                }
            )
        )

        run_health_check()
        public_health = json.loads((public / "health.json").read_text())
        assert public_health["kill_switch"]["enabled"] is True
        assert public_health["kill_switch"]["level"] == "halt"
        assert public_health["system_status"] == "critical"
        assert public_health["ops_health_source"] == "monitor.health_check"
        # Dashboard-only fields preserved
        assert public_health["cron_jobs"] == []
        assert "generated_at" in public_health

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

    def test_active_kill_switch_halt_elevates_status_while_freshness_is_warning(
        self, tmp_path, monkeypatch
    ):
        """Active incident HALT must not leave data/health.json as warning-only."""
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.HEALTH_PATH", tmp_path / "health.json")
        monkeypatch.setattr(
            "src.monitor.health_check._check_fred_md_cache",
            lambda: {"status": "ok", "row_count": 1, "latest_fetched_at": "2026-06-11T00:00:00+00:00"},
        )
        # Fresh prices/signals so only kill-switch/incident drive severity
        (tmp_path / "prices.json").write_text("{}")
        (tmp_path / "signals.json").write_text("{}")
        (tmp_path / "cron_status.json").write_text('{"jobs": []}')
        (tmp_path / "kill_switch.json").write_text(json.dumps({
            "enabled": True,
            "level": "halt",
            "mode": "paper",
            "reason": "unresolved_incident:signal_staleness",
            "source": "incident_lifecycle",
            "message": "1/23 signals stale: alternative_data",
            "timestamp": "2026-07-06T18:00:00+00:00",
            "incident_id": "inc-1",
        }))
        (tmp_path / "incidents.json").write_text(json.dumps({
            "open_count": 1,
            "incidents": [{
                "incident_id": "inc-1",
                "channel": "signal_staleness",
                "severity": "p2",
                "state": "firing",
                "message": "1/23 signals stale: alternative_data",
                "kill_switch_level": "halt",
            }],
        }))

        report = run_health_check()

        assert report["status"] == "critical"
        assert "kill_switch" in report["checks"]
        assert "open_incidents" in report["checks"]
        ks = report["checks"]["kill_switch"]
        assert ks["status"] == "critical"
        assert ks["enabled"] is True
        assert ks["level"] == "halt"
        assert ks["reason"] == "unresolved_incident:signal_staleness"
        oi = report["checks"]["open_incidents"]
        assert oi["status"] == "critical"
        assert oi["open_count"] == 1
        assert oi["incidents"][0]["incident_id"] == "inc-1"
        assert oi["incidents"][0]["kill_switch_level"] == "halt"

    def test_no_kill_switch_or_incidents_reports_ok_dimensions(self, tmp_path, monkeypatch):
        """Absent safety controls should not invent critical dimensions."""
        monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", tmp_path)
        monkeypatch.setattr("src.monitor.health_check.HEALTH_PATH", tmp_path / "health.json")
        monkeypatch.setattr(
            "src.monitor.health_check._check_fred_md_cache",
            lambda: {"status": "ok", "row_count": 1, "latest_fetched_at": "2026-06-11T00:00:00+00:00"},
        )
        (tmp_path / "prices.json").write_text("{}")
        (tmp_path / "signals.json").write_text("{}")
        (tmp_path / "cron_status.json").write_text('{"jobs": []}')

        report = run_health_check()

        assert report["checks"]["kill_switch"]["status"] == "ok"
        assert report["checks"]["kill_switch"]["enabled"] is False
        assert report["checks"]["open_incidents"]["status"] == "ok"
        assert report["checks"]["open_incidents"]["open_count"] == 0


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


class TestGraduationCircuitBreakerProducer:
    """consecutive_ok producer for graduation circuit_breaker_confidence."""

    def test_healthy_closed_increments_streak(self, tmp_path):
        from src.monitor.health_check import update_graduation_circuit_breaker_state

        p1 = update_graduation_circuit_breaker_state(
            system_status="healthy",
            broker_circuit={"state": "closed", "fail_count": 0},
            data_dir=tmp_path,
        )
        assert p1["consecutive_ok"] == 1
        assert p1["status"] == "green"
        path = tmp_path / ".circuit_breaker.json"
        assert path.exists()

        p2 = update_graduation_circuit_breaker_state(
            system_status="ok",
            broker_circuit={"state": "closed", "fail_count": 0},
            data_dir=tmp_path,
        )
        assert p2["consecutive_ok"] == 2

    def test_open_broker_resets_streak(self, tmp_path):
        from src.monitor.health_check import update_graduation_circuit_breaker_state

        update_graduation_circuit_breaker_state(
            system_status="healthy",
            broker_circuit={"state": "closed", "fail_count": 0},
            data_dir=tmp_path,
        )
        update_graduation_circuit_breaker_state(
            system_status="healthy",
            broker_circuit={"state": "closed", "fail_count": 0},
            data_dir=tmp_path,
        )
        bad = update_graduation_circuit_breaker_state(
            system_status="healthy",
            broker_circuit={"state": "open", "fail_count": 3},
            data_dir=tmp_path,
        )
        assert bad["consecutive_ok"] == 0
        assert bad["status"] == "red"

    def test_missing_file_starts_at_one_when_green(self, tmp_path):
        from src.monitor.health_check import update_graduation_circuit_breaker_state

        assert not (tmp_path / ".circuit_breaker.json").exists()
        p = update_graduation_circuit_breaker_state(
            system_status="healthy",
            broker_circuit={"state": "closed", "fail_count": 0},
            data_dir=tmp_path,
        )
        assert p["consecutive_ok"] == 1


    def test_signal_health_zero_of_n_blocks_streak_increment(self, tmp_path):
        """Batch CB: SH 0/N must not climb consecutive_ok even when ops status ok."""
        from src.monitor.health_check import update_graduation_circuit_breaker_state

        # seed streak
        seed = update_graduation_circuit_breaker_state(
            system_status="healthy",
            broker_circuit={"state": "closed", "fail_count": 0},
            data_dir=tmp_path,
        )
        assert seed["consecutive_ok"] == 1

        sh = {
            "status": "degraded",
            "summary": {
                "healthy": 0,
                "degraded": 8,
                "unhealthy": 1,
                "total_tracked": 9,
            },
        }
        blocked = update_graduation_circuit_breaker_state(
            system_status="ok",  # ops monitor schema often ok without SH
            broker_circuit={"state": "closed", "fail_count": 0},
            data_dir=tmp_path,
            signal_health=sh,
        )
        # Hold streak (do not climb); status yellow for quality outage
        assert blocked["consecutive_ok"] == 1
        assert blocked.get("signal_health_blocked") is True
        assert blocked["status"] == "yellow"

        # Hold also when ops is warning (lab FRED/cron) — SH gate freezes climb
        warn_hold = update_graduation_circuit_breaker_state(
            system_status="warning",
            broker_circuit={"state": "closed", "fail_count": 0},
            data_dir=tmp_path,
            signal_health=sh,
        )
        assert warn_hold["consecutive_ok"] == 1
        assert warn_hold.get("signal_health_blocked") is True

        # Clear SH path can resume climb
        resume = update_graduation_circuit_breaker_state(
            system_status="healthy",
            broker_circuit={"state": "closed", "fail_count": 0},
            data_dir=tmp_path,
            signal_health={
                "status": "healthy",
                "summary": {
                    "healthy": 5,
                    "degraded": 2,
                    "unhealthy": 0,
                    "total_tracked": 7,
                },
            },
        )
        assert resume["consecutive_ok"] == 2
        assert resume.get("signal_health_blocked") is not True

    def test_signal_health_warning_does_not_freeze_streak(self, tmp_path):
        """Batch EL: SH contribution warning (1/N healthy) must not freeze CB.

        Live fleet often has healthy=1, degraded=6, unhealthy=2 → contribution
        ``warning``. Freezing consecutive_ok on warning left graduation CB at
        0 forever despite ops status ok and broker closed.
        """
        from src.monitor.health_check import update_graduation_circuit_breaker_state

        seed = update_graduation_circuit_breaker_state(
            system_status="ok",
            broker_circuit={"state": "closed", "fail_count": 0},
            data_dir=tmp_path,
        )
        assert seed["consecutive_ok"] == 1

        sh_warn = {
            "status": "degraded",
            "summary": {
                "healthy": 1,
                "degraded": 6,
                "unhealthy": 2,
                "total_tracked": 9,
            },
        }
        climbed = update_graduation_circuit_breaker_state(
            system_status="ok",
            broker_circuit={"state": "closed", "fail_count": 0},
            data_dir=tmp_path,
            signal_health=sh_warn,
        )
        assert climbed["consecutive_ok"] == 2
        assert climbed.get("signal_health_blocked") is not True
        assert climbed["status"] == "green"

        # 0/N still freezes (degraded contribution)
        sh_zero = {
            "status": "degraded",
            "summary": {
                "healthy": 0,
                "degraded": 7,
                "unhealthy": 2,
                "total_tracked": 9,
            },
        }
        blocked = update_graduation_circuit_breaker_state(
            system_status="ok",
            broker_circuit={"state": "closed", "fail_count": 0},
            data_dir=tmp_path,
            signal_health=sh_zero,
        )
        assert blocked["consecutive_ok"] == 2  # hold, not climb
        assert blocked.get("signal_health_blocked") is True
        assert blocked["status"] == "yellow"

    def test_absent_signal_health_preserves_legacy_ops_only_climb(self, tmp_path):
        """When SH not passed, keep pre-CB ops+broker behavior (backward compat)."""
        from src.monitor.health_check import update_graduation_circuit_breaker_state

        p = update_graduation_circuit_breaker_state(
            system_status="healthy",
            broker_circuit={"state": "closed", "fail_count": 0},
            data_dir=tmp_path,
        )
        assert p["consecutive_ok"] == 1
