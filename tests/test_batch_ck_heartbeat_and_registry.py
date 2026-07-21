"""Batch CK: schedule-aware last_success heartbeat + decision_registry private dual-write."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.monitor.hermes_cron import (
    estimate_schedule_period_seconds,
    normalize_cron_job,
    schedule_aware_last_success_heartbeat,
    summarize_backend,
)


def test_estimate_schedule_period_weekly_hourly():
    assert estimate_schedule_period_seconds("20 4 * * 0") == 7 * 86400
    assert estimate_schedule_period_seconds("0,30 * * * *") == 30 * 60
    assert estimate_schedule_period_seconds("5 * * * *") == 3600
    assert estimate_schedule_period_seconds("0 9 * * *") == 86400
    assert estimate_schedule_period_seconds("") is None


def test_pending_never_run_weekly_not_overdue():
    job = {
        "name": "portfolio-lab-fetch-trends",
        "schedule": "20 4 * * 0",
        "status": "pending",
        "enabled": True,
        "manual_only": False,
        "state": "scheduled",
        "last_run": None,
    }
    hb = schedule_aware_last_success_heartbeat(job)
    assert hb["heartbeat_state"] == "pending_never_run"
    assert hb["overdue"] is False
    assert hb["last_success_age_seconds"] is None
    assert hb["schedule_period_seconds"] == 7 * 86400


def test_hourly_job_overdue_when_age_exceeds_period_plus_grace():
    # last success ~3h ago; period 3600; grace max(360, 3600)=3600 → threshold 7200
    # so 3h is still ok; use 3h period job that is old
    now = datetime(2026, 7, 21, 20, 0, tzinfo=timezone.utc).timestamp()
    last = datetime(2026, 7, 21, 10, 0, tzinfo=timezone.utc).timestamp()  # 10h ago
    job = {
        "schedule": "5 * * * *",  # hourly
        "status": "ok",
        "enabled": True,
        "last_run": datetime.fromtimestamp(last, tz=timezone.utc).isoformat(),
        "state": "scheduled",
        "manual_only": False,
    }
    hb = schedule_aware_last_success_heartbeat(job, now=now, min_grace_seconds=600)
    assert hb["schedule_period_seconds"] == 3600
    assert hb["last_success_age_seconds"] == 10 * 3600
    assert hb["overdue"] is True
    assert hb["heartbeat_state"] == "overdue"


def test_normalize_cron_job_attaches_heartbeat_fields():
    job = normalize_cron_job(
        {
            "name": "portfolio-lab-fetch-trends",
            "schedule": "20 4 * * 0",
            "status": "pending",
            "enabled": True,
            "state": "scheduled",
            "last_run": None,
        },
        backend="tasker",
        source="/tmp/cron_status.json",
    )
    assert job["status"] == "pending"
    assert job["heartbeat_state"] == "pending_never_run"
    assert job["heartbeat_overdue"] is False
    assert job["schedule_period_seconds"] == 7 * 86400
    assert job["last_success_age_seconds"] is None


def test_summarize_backend_pending_never_run_still_ok_with_heartbeat():
    jobs = [
        normalize_cron_job(
            {
                "name": "portfolio-lab-data",
                "schedule": "5 * * * *",
                "status": "success",
                "enabled": True,
                "state": "scheduled",
                "last_run": "2026-07-21T19:05:00+00:00",
            },
            backend="tasker",
            source="x",
            now=datetime(2026, 7, 21, 19, 30, tzinfo=timezone.utc).timestamp(),
        ),
        normalize_cron_job(
            {
                "name": "portfolio-lab-fetch-trends",
                "schedule": "20 4 * * 0",
                "status": "pending",
                "enabled": True,
                "state": "scheduled",
                "last_run": None,
            },
            backend="tasker",
            source="x",
        ),
    ]
    summary = summarize_backend(backend="tasker", source="x", jobs=jobs)
    assert summary["status"] == "ok"
    assert summary.get("pending_never_run_jobs") == 1
    assert summary.get("heartbeat_overdue_jobs") is None


def test_publish_decision_registry_dual_writes_private(tmp_path, monkeypatch):
    from src.monitor import decision_registry as dr

    public = tmp_path / "www"
    private = tmp_path / "data"
    public.mkdir()
    private.mkdir()
    reg = dr.DecisionRegistry(db_path=private / "decision_registry.db")
    monkeypatch.setattr(dr, "DATA_DIR", private)

    path = dr.publish_decision_registry_json(public_dir=public, registry=reg)
    assert path == public / "decision_registry.json"
    assert path.exists()
    assert (private / "decision_registry.json").exists()
    pub = json.loads(path.read_text(encoding="utf-8"))
    priv = json.loads((private / "decision_registry.json").read_text(encoding="utf-8"))
    assert pub == priv
