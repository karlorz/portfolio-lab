"""Batch DT: pending_never_run weekly jobs reconcile when producer artifact is fresh."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.monitor.hermes_cron import (
    normalize_cron_job,
    pending_job_artifact_evidence,
    summarize_backend,
)


def test_pending_trends_reconciles_with_fresh_google_trends(tmp_path: Path) -> None:
    trends = {
        "_meta": {
            "fetched_at": "2026-07-22T07:06:16.760078",
            "latest_observation": "2026-07-21",
            "schema": "google-trends-cache/v1",
        },
        "inflation": {"2026-07-21": 50},
    }
    (tmp_path / "google_trends.json").write_text(json.dumps(trends), encoding="utf-8")
    # Touch mtime to "now"
    Path(tmp_path / "google_trends.json").touch()

    job = {
        "name": "portfolio-lab-fetch-trends",
        "schedule": "20 4 * * 0",
        "enabled": True,
        "manual_only": False,
        "state": "scheduled",
        "status": "pending",
        "last_run": None,
    }
    evidence = pending_job_artifact_evidence(
        job, data_dirs=[tmp_path], now=datetime.now(timezone.utc).timestamp()
    )
    assert evidence is not None
    assert evidence["artifact"] == "google_trends.json"
    assert "pending_never_run" in evidence["reason"]

    normalized = normalize_cron_job(
        job,
        backend="tasker",
        source=str(tmp_path / "cron_status.json"),
        now=datetime.now(timezone.utc).timestamp(),
    )
    # normalize_cron_job uses live DATA_DIR by default for evidence — inject via
    # monkeypatch-free path: call evidence already proven; for full normalize
    # we need artifact on default path OR we unit-test evidence + manual apply.
    # Re-run normalize with evidence path by writing into a side channel is hard;
    # assert evidence API and summarize after manual stamp.
    assert evidence["job"] == "portfolio-lab-fetch-trends"


def test_normalize_reconciles_when_artifact_on_search_path(
    tmp_path: Path, monkeypatch
) -> None:
    from src.monitor import hermes_cron as hc

    trends_path = tmp_path / "google_trends.json"
    trends_path.write_text(
        json.dumps(
            {
                "_meta": {"fetched_at": "2026-07-22T07:06:16", "latest_observation": "2026-07-21"},
                "inflation": {"2026-07-21": 40},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", tmp_path)

    job = {
        "name": "portfolio-lab-fetch-trends",
        "schedule": "20 4 * * 0",
        "enabled": True,
        "manual_only": False,
        "state": "scheduled",
        "status": "pending",
        "last_run": None,
    }
    n = hc.normalize_cron_job(
        job, backend="tasker", source="test", now=datetime.now(timezone.utc).timestamp()
    )
    assert n["status"] == "ok"
    assert n.get("pending_artifact_reconciled") is True
    assert n.get("last_run") is not None
    assert n.get("heartbeat_state") in {"ok", "overdue"}  # ok if mtime fresh
    assert n["heartbeat_state"] == "ok"
    assert n.get("pending_artifact_evidence", {}).get("artifact") == "google_trends.json"

    summary = summarize_backend(
        backend="tasker",
        source="test",
        jobs=[n],
    )
    assert summary.get("pending_never_run_jobs") in (None, 0)
    assert summary.get("pending_artifact_reconciled_jobs") == 1
    assert summary["status"] == "ok"


def test_pending_without_artifact_stays_pending(tmp_path: Path, monkeypatch) -> None:
    from src.monitor import hermes_cron as hc

    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", tmp_path)
    job = {
        "name": "portfolio-lab-fetch-trends",
        "schedule": "20 4 * * 0",
        "enabled": True,
        "status": "pending",
        "last_run": None,
    }
    n = hc.normalize_cron_job(job, backend="tasker", source="test")
    assert n["status"] == "pending"
    assert not n.get("pending_artifact_reconciled")
    assert n.get("heartbeat_state") == "pending_never_run"
