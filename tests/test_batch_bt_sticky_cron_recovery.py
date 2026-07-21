"""Batch BT: sticky cron error recovery via fresher producer artifacts."""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.monitor.hermes_cron import (
    cron_job_artifact_recovery_evidence,
    is_sticky_cron_error_recovered,
    rollup_failed_cron_jobs,
    summarize_backend,
)


def _touch(path: Path, epoch: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    # (atime, mtime)
    import os

    os.utime(path, (epoch, epoch))


def test_sticky_dashboard_error_recovered_by_fresher_signals(tmp_path: Path):
    now = time.time()
    last_run = now - 3600  # failed 1h ago
    # signals written after failure (recovery regenerate)
    _touch(tmp_path / "signals.json", now - 60)

    job = {
        "name": "portfolio-lab-dashboard",
        "status": "error",
        "last_run": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(last_run)),
    }
    assert is_sticky_cron_error_recovered(job, data_dirs=[tmp_path], now=now)
    evidence = cron_job_artifact_recovery_evidence(
        job, data_dirs=[tmp_path], now=now
    )
    assert evidence is not None
    assert evidence["artifact"] in {"signals.json", "health.json", "dashboard.json"}
    assert evidence["live_authoritative"] is False

    failed = rollup_failed_cron_jobs([job], data_dirs=[tmp_path], now=now)
    assert failed == []


def test_sticky_data_error_recovered_by_same_run_prices_within_grace(tmp_path: Path):
    """Fetch writes prices ~12s before job finished_at on generator failure."""
    now = time.time()
    last_run = now - 120
    # prices 12s before last_run, still fresh
    _touch(tmp_path / "prices.json", last_run - 12)

    job = {
        "name": "portfolio-lab-data",
        "status": "error",
        "last_run": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(last_run)),
    }
    assert is_sticky_cron_error_recovered(
        job, data_dirs=[tmp_path], now=now,  # type: ignore[call-arg]
    ) or is_sticky_cron_error_recovered(job, data_dirs=[tmp_path], now=now)
    # explicit: within default 180s grace
    assert is_sticky_cron_error_recovered(job, data_dirs=[tmp_path], now=now)
    assert rollup_failed_cron_jobs([job], data_dirs=[tmp_path], now=now) == []


def test_stale_artifact_does_not_recover(tmp_path: Path):
    now = time.time()
    last_run = now - 100
    # ancient prices (8h old) even if "after" last_run in a weird clock case:
    # make mtime older than max age
    _touch(tmp_path / "prices.json", now - 8 * 3600)

    job = {
        "name": "portfolio-lab-data",
        "status": "error",
        "last_run": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(last_run)),
    }
    assert not is_sticky_cron_error_recovered(job, data_dirs=[tmp_path], now=now)
    assert len(rollup_failed_cron_jobs([job], data_dirs=[tmp_path], now=now)) == 1


def test_unknown_job_not_auto_recovered(tmp_path: Path):
    now = time.time()
    _touch(tmp_path / "foo.json", now)
    job = {
        "name": "portfolio-lab-mystery",
        "status": "error",
        "last_run": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now - 10)),
    }
    assert not is_sticky_cron_error_recovered(job, data_dirs=[tmp_path], now=now)
    assert len(rollup_failed_cron_jobs([job], data_dirs=[tmp_path], now=now)) == 1


def test_health_self_job_still_excluded():
    job = {"name": "portfolio-lab-health", "status": "error", "last_run": "2026-07-21T00:00:00+00:00"}
    assert rollup_failed_cron_jobs([job]) == []


def test_summarize_backend_reports_recovered_count(tmp_path: Path):
    now = time.time()
    last_run = now - 1800
    _touch(tmp_path / "dashboard.json", now - 30)
    jobs = [
        {
            "name": "portfolio-lab-dashboard",
            "status": "error",
            "last_run": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(last_run)),
            "enabled": True,
        },
        {
            "name": "portfolio-lab-eval",
            "status": "ok",
            "last_run": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now)),
            "enabled": True,
        },
    ]
    summary = summarize_backend(
        backend="tasker",
        source="test",
        jobs=jobs,
        data_dirs=[tmp_path],
        now=now,
    )
    assert summary["failed_jobs"] == 0
    assert summary["status"] == "ok"
    assert summary["recovered_sticky_errors"] == 1
    assert "portfolio-lab-dashboard" in summary["recovered_sticky_job_names"]
