"""Unit tests for src.monitor.hermes_cron (Item Q54).

Tests cover:
- is_health_self_job (self job row vs other names vs invalid types)
- _parse_iso_to_utc_epoch (ISO parsing with Z, timezone offsets, naive timestamps, invalid formats)
- recovery_data_dirs (extra_dirs with/without defaults)
- cron_job_artifact_recovery_evidence & is_sticky_cron_error_recovered
- pending_job_artifact_evidence (Batch DT soft-success via fresh artifact)
- normalize_cron_status & normalize_cron_state
- rollup_failed_cron_jobs (self-job exclusion, timeout hard fail, recovered error filtering)
- load_local_cron_jobs (missing file, corrupt JSON, valid tasker file)
- combine_scheduler_backends (all ok, error/degraded, unavailable, warning mixed)
- resolve_hermes_cron_jobs_path (explicit env override, test isolation dir, default dir)
- estimate_schedule_period_seconds (weekly, step-hour, daily, hourly)
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.monitor import hermes_cron as hc


def test_is_health_self_job() -> None:
    assert hc.is_health_self_job({"name": "portfolio-lab-health"}) is True
    assert hc.is_health_self_job({"name": "portfolio-lab-data"}) is False
    assert hc.is_health_self_job({}) is False
    assert hc.is_health_self_job("not_a_dict") is False  # type: ignore


def test_parse_iso_to_utc_epoch() -> None:
    # ISO with Z
    ts_z = hc._parse_iso_to_utc_epoch("2026-08-17T08:00:00Z")
    assert ts_z is not None
    # ISO with offset
    ts_offset = hc._parse_iso_to_utc_epoch("2026-08-17T08:00:00+00:00")
    assert ts_offset == ts_z

    # Naive timestamp treated as UTC
    ts_naive = hc._parse_iso_to_utc_epoch("2026-08-17T08:00:00")
    assert ts_naive == ts_z

    # Invalid / empty / None
    assert hc._parse_iso_to_utc_epoch("") is None
    assert hc._parse_iso_to_utc_epoch("not-a-date") is None
    assert hc._parse_iso_to_utc_epoch(None) is None


def test_recovery_data_dirs(tmp_path: Path) -> None:
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()

    # Hermetic only (include_defaults=False)
    dirs_hermetic = hc.recovery_data_dirs([d1, d2], include_defaults=False)
    assert dirs_hermetic == [d1, d2]

    # With defaults
    dirs_default = hc.recovery_data_dirs([d1], include_defaults=True)
    assert d1 in dirs_default


def test_cron_job_artifact_recovery_evidence(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    now_ts = time.time()

    # 1. Non-error job returns None
    ok_job = {"name": "portfolio-lab-data", "status": "ok", "last_run": "2026-08-17T08:00:00Z"}
    assert hc.cron_job_artifact_recovery_evidence(ok_job, data_dirs=[data_dir], now=now_ts) is None

    # 2. Unknown job name (not in JOB_RECOVERY_ARTIFACTS) returns None
    unknown_job = {"name": "custom-task", "status": "error", "last_run": "2026-08-17T08:00:00Z"}
    assert hc.cron_job_artifact_recovery_evidence(unknown_job, data_dirs=[data_dir], now=now_ts) is None

    # 3. Error job with fresh proving artifact returns recovery evidence
    error_job = {
        "name": "portfolio-lab-data",
        "status": "error",
        "last_run": datetime.fromtimestamp(now_ts - 60, tz=timezone.utc).isoformat(),
    }
    # Create fresh prices.json artifact
    prices_file = data_dir / "prices.json"
    prices_file.write_text("{}", encoding="utf-8")

    evidence = hc.cron_job_artifact_recovery_evidence(error_job, data_dirs=[data_dir], now=now_ts)
    assert evidence is not None
    assert evidence["job"] == "portfolio-lab-data"
    assert evidence["artifact"] == "prices.json"
    assert hc.is_sticky_cron_error_recovered(error_job, data_dirs=[data_dir], now=now_ts) is True

    # 4. If artifact is older than max_age_hours, returns None
    evidence_stale = hc.cron_job_artifact_recovery_evidence(
        error_job,
        data_dirs=[data_dir],
        now=now_ts + (10 * 3600),
        max_age_hours=6.0,
    )
    assert evidence_stale is None


def test_pending_job_artifact_evidence(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    now_ts = time.time()

    # Create trends cache with _meta.fetched_at
    trends_file = data_dir / "google_trends.json"
    trends_file.write_text(
        json.dumps({
            "_meta": {"fetched_at": "2026-08-17T00:00:00Z"},
            "terms": ["inflation", "recession"],
        }),
        encoding="utf-8",
    )
    # Pin the artifact mtime to the supplied now_ts: pending evidence rejects
    # future artifacts (age < 0), while content keeps the fixed _meta.fetched_at.
    os.utime(trends_file, (now_ts, now_ts))

    pending_job = {
        "name": "portfolio-lab-fetch-trends",
        "status": "pending",
        "last_run": None,
    }
    evidence = hc.pending_job_artifact_evidence(pending_job, data_dirs=[data_dir], now=now_ts)
    assert evidence is not None
    assert evidence["job"] == "portfolio-lab-fetch-trends"
    assert evidence["artifact"] == "google_trends.json"
    assert evidence["meta_fetched_at"] == "2026-08-17T00:00:00Z"

    # Non-pending job returns None
    assert hc.pending_job_artifact_evidence(
        {"name": "portfolio-lab-fetch-trends", "status": "ok", "last_run": None},
        data_dirs=[data_dir],
        now=now_ts,
    ) is None


def test_normalize_cron_status_and_state() -> None:
    # Statuses
    assert hc.normalize_cron_status("success") == "ok"
    assert hc.normalize_cron_status("pass") == "ok"
    assert hc.normalize_cron_status("failed") == "error"
    assert hc.normalize_cron_status("timeout") == "error"
    assert hc.normalize_cron_status("paused") == "disabled"
    assert hc.normalize_cron_status("pending") == "pending"
    assert hc.normalize_cron_status("blocked") == "pending"
    assert hc.normalize_cron_status("unrecognized_status") == "unknown"

    # States
    assert hc.normalize_cron_state("scheduled") == "scheduled"
    assert hc.normalize_cron_state("running") == "running"
    assert hc.normalize_cron_state("anything", enabled=False) == "paused"
    assert hc.normalize_cron_state("anything", manual_only=True) == "manual_only"


def test_rollup_failed_cron_jobs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    now_ts = time.time()

    # Create fresh artifact for recoverable data job
    (data_dir / "prices.json").write_text("{}", encoding="utf-8")

    jobs = [
        # Self-job error excluded
        {"name": "portfolio-lab-health", "status": "error"},
        # Hard timeout kept
        {"name": "portfolio-lab-eval", "status": "timeout"},
        # Recoverable error excluded
        {
            "name": "portfolio-lab-data",
            "status": "error",
            "last_run": datetime.fromtimestamp(now_ts - 60, tz=timezone.utc).isoformat(),
        },
        # Unrecoverable error kept
        {"name": "unmapped-job", "status": "error"},
        # Ok job ignored
        {"name": "other-job", "status": "ok"},
    ]

    failed = hc.rollup_failed_cron_jobs(jobs, data_dirs=[data_dir], now=now_ts)
    failed_names = [j["name"] for j in failed]
    assert "portfolio-lab-eval" in failed_names
    assert "unmapped-job" in failed_names
    assert "portfolio-lab-health" not in failed_names
    assert "portfolio-lab-data" not in failed_names


def test_load_local_cron_jobs(tmp_path: Path) -> None:
    status_file = tmp_path / "cron_status.json"

    # 1. Missing file
    jobs_missing, summary_missing = hc.load_local_cron_jobs(status_file)
    assert jobs_missing == []
    assert summary_missing["status"] == "unavailable"

    # 2. Corrupt JSON
    status_file.write_text("{corrupt", encoding="utf-8")
    jobs_corrupt, summary_corrupt = hc.load_local_cron_jobs(status_file)
    assert jobs_corrupt == []
    assert summary_corrupt["status"] == "error"

    # 3. Valid status file; last_run must be current or the fixed-August date
    # makes the schedule-aware heartbeat overdue and the summary degraded.
    now_ts = time.time()
    status_file.write_text(
        json.dumps({
            "backend": "tasker",
            "jobs": [
                {
                    "name": "job-1",
                    "status": "success",
                    "schedule": "0 * * * *",
                    "last_run": datetime.fromtimestamp(now_ts - 600, tz=timezone.utc).isoformat(),
                }
            ],
        }),
        encoding="utf-8",
    )
    jobs_valid, summary_valid = hc.load_local_cron_jobs(status_file)
    assert len(jobs_valid) == 1
    assert jobs_valid[0]["name"] == "job-1"
    assert jobs_valid[0]["status"] == "ok"
    assert summary_valid["status"] == "ok"
    assert summary_valid["total_jobs"] == 1


def test_combine_scheduler_backends() -> None:
    # All ok
    assert hc.combine_scheduler_backends({"local": {"status": "ok"}})["status"] == "ok"

    # Degraded backend
    assert hc.combine_scheduler_backends({
        "local": {"status": "ok"},
        "hermes": {"status": "degraded"},
    })["status"] == "degraded"

    # All unavailable
    assert hc.combine_scheduler_backends({
        "local": {"status": "unavailable"},
        "hermes": {"status": "unavailable"},
    })["status"] == "unavailable"

    # Mixed ok and unavailable -> warning
    assert hc.combine_scheduler_backends({
        "local": {"status": "ok"},
        "hermes": {"status": "unavailable"},
    })["status"] == "warning"


def test_resolve_hermes_cron_jobs_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Explicit env variable
    custom_path = tmp_path / "custom" / "jobs.json"
    monkeypatch.setenv("HERMES_CRON_JOBS_PATH", str(custom_path))
    assert hc.resolve_hermes_cron_jobs_path(
        current_data_dir=tmp_path / "data",
        default_data_dir=tmp_path / "default_data",
    ) == custom_path

    # Without explicit env and non-default data dir -> returns None
    monkeypatch.delenv("HERMES_CRON_JOBS_PATH", raising=False)
    assert hc.resolve_hermes_cron_jobs_path(
        current_data_dir=tmp_path / "data",
        default_data_dir=tmp_path / "default_data",
    ) is None

    # Matching default data dir -> returns HERMES_HOME / cron / jobs.json
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    res = hc.resolve_hermes_cron_jobs_path(
        current_data_dir=tmp_path / "data",
        default_data_dir=tmp_path / "data",
    )
    assert res == tmp_path / ".hermes" / "cron" / "jobs.json"


def test_estimate_schedule_period_seconds() -> None:
    # Weekly (Sunday)
    assert hc.estimate_schedule_period_seconds("20 4 * * 0") == 7 * 86400

    # Step hour (every 3 hours)
    assert hc.estimate_schedule_period_seconds("0 */3 * * *") == 3 * 3600

    # Fixed hour list (every 6 hours)
    assert hc.estimate_schedule_period_seconds("0 0,6,12,18 * * *") == 6 * 3600

    # Daily (specific hour)
    assert hc.estimate_schedule_period_seconds("0 18 * * *") == 86400

    # Hourly
    assert hc.estimate_schedule_period_seconds("0 * * * *") == 3600
    assert hc.estimate_schedule_period_seconds("*/15 * * * *") == 15 * 60

    # Invalid / empty
    assert hc.estimate_schedule_period_seconds("") is None
    assert hc.estimate_schedule_period_seconds("invalid cron") is None


def test_schedule_aware_last_success_heartbeat() -> None:
    now_ts = time.time()

    # Inactive job (disabled / manual_only)
    hb_inactive = hc.schedule_aware_last_success_heartbeat({"enabled": False}, now=now_ts)
    assert hb_inactive["heartbeat_state"] == "inactive"

    # Pending never run job
    hb_pending = hc.schedule_aware_last_success_heartbeat(
        {"enabled": True, "status": "pending", "last_run": None},
        now=now_ts,
    )
    assert hb_pending["heartbeat_state"] == "pending_never_run"
    assert hb_pending["overdue"] is False

    # Fresh ok run
    hb_ok = hc.schedule_aware_last_success_heartbeat(
        {
            "enabled": True,
            "status": "ok",
            "schedule": "0 * * * *",
            "last_run": datetime.fromtimestamp(now_ts - 600, tz=timezone.utc).isoformat(),
        },
        now=now_ts,
    )
    assert hb_ok["heartbeat_state"] == "ok"
    assert hb_ok["overdue"] is False

    # Overdue hourly run (last run 2 hours ago > period 3600s + grace 3600s = 7200s)
    hb_overdue = hc.schedule_aware_last_success_heartbeat(
        {
            "enabled": True,
            "status": "ok",
            "schedule": "0 * * * *",
            "last_run": datetime.fromtimestamp(now_ts - 8000, tz=timezone.utc).isoformat(),
        },
        now=now_ts,
    )
    assert hb_overdue["heartbeat_state"] == "overdue"
    assert hb_overdue["overdue"] is True

