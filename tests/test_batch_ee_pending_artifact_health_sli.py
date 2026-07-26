"""Batch EE: pending-artifact reconcile projected onto compact health.

Live friction: raw cron_status shows portfolio-lab-fetch-trends status=pending
while DT already soft-oks via fresh google_trends.json (age ~0.3d). Compact
health only had cron_job_count / failed_cron_jobs — no dual-signal disclosure.
Research: dual-signal (scheduler pending + artifact freshness).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.dashboard.generator import project_pending_artifact_cron_onto_health
from src.monitor.health_check import refresh_signals_health_kill_fields


def test_project_reconciled_and_true_pending() -> None:
    health: dict = {"status": "ok"}
    jobs = [
        {
            "name": "portfolio-lab-fetch-trends",
            "status": "ok",
            "pending_artifact_reconciled": True,
            "pending_artifact_evidence": {
                "artifact": "google_trends.json",
                "reason": "producer_artifact_fresh_while_tasker_pending_never_run",
            },
            "heartbeat_disclosure": "Batch DT: pending_never_run reconciled",
        },
        {
            "name": "other-weekly",
            "status": "pending",
            "last_run": None,
            "enabled": True,
            "manual_only": False,
        },
        {"name": "daily-data", "status": "ok", "enabled": True},
    ]
    out = project_pending_artifact_cron_onto_health(health, jobs)
    assert out["cron_pending_artifact_reconciled_jobs"] == 1
    assert out["cron_pending_never_run_jobs"] == 1
    assert "portfolio-lab-fetch-trends" in (
        out.get("cron_pending_artifact_reconciled_names") or ""
    )
    assert out["cron_pending_artifact_status"] == "mixed"
    assert "google_trends.json" in (out.get("cron_pending_artifact_sample") or "")
    assert out["status"] == "ok"  # dual-signal: do not warn for true pending alone


def test_all_reconciled_ok() -> None:
    health: dict = {"status": "ok"}
    jobs = [
        {
            "name": "portfolio-lab-fetch-trends",
            "status": "ok",
            "pending_artifact_reconciled": True,
            "pending_artifact_evidence": {"artifact": "google_trends.json"},
        }
    ]
    out = project_pending_artifact_cron_onto_health(health, jobs)
    assert out["cron_pending_artifact_reconciled_jobs"] == 1
    assert out["cron_pending_never_run_jobs"] == 0
    assert out["cron_pending_artifact_status"] == "reconciled"
    assert out["status"] == "ok"


def test_partial_refresh_reprojects_from_cron_status(
    tmp_path: Path, monkeypatch
) -> None:
    public = tmp_path / "public"
    private = tmp_path / "data"
    public.mkdir()
    private.mkdir()
    # Fresh trends artifact
    (private / "google_trends.json").write_text(
        json.dumps(
            {
                "_meta": {
                    "fetched_at": "2026-07-22T07:06:16",
                    "latest_observation": "2026-07-21",
                },
                "inflation": {"2026-07-21": 50},
            }
        ),
        encoding="utf-8",
    )
    (private / "google_trends.json").touch()
    (private / "cron_status.json").write_text(
        json.dumps(
            {
                "backend": "tasker",
                "jobs": [
                    {
                        "name": "portfolio-lab-fetch-trends",
                        "schedule": "20 4 * * 0",
                        "enabled": True,
                        "manual_only": False,
                        "state": "scheduled",
                        "status": "pending",
                        "last_run": None,
                        "backend": "tasker",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    signals = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "generated_at": "2026-07-22T05:00:00+00:00",
        "generator_git_sha": "deadbeef",
        "generator_git_sha_status": "full_generate",
        "health": {"status": "ok"},
    }
    (public / "signals.json").write_text(json.dumps(signals), encoding="utf-8")
    (private / "signals.json").write_text(json.dumps(signals), encoding="utf-8")
    (private / "kill_switch.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8"
    )
    (private / "incidents.json").write_text(
        json.dumps({"open_count": 0, "status": "ok"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False
    )
    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", private, raising=False)
    # hermes_cron DATA_DIR for normalize evidence
    monkeypatch.setattr("src.monitor.hermes_cron.DATA_DIR", private, raising=False)
    monkeypatch.setattr(
        "src.monitor.hermes_cron.PUBLIC_DATA_DIR", private, raising=False
    )

    report = {
        "status": "ok",
        "system_status": "ok",
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0, "status": "ok"},
        "cron_jobs": [
            {
                "name": "portfolio-lab-fetch-trends",
                "status": "ok",
                "pending_artifact_reconciled": True,
                "pending_artifact_evidence": {"artifact": "google_trends.json"},
            }
        ],
    }
    refresh_signals_health_kill_fields(
        report, public_dir=public, data_dir=private
    )
    out = json.loads((public / "signals.json").read_text(encoding="utf-8"))
    h = out.get("health") or {}
    assert h.get("cron_pending_artifact_reconciled_jobs") == 1
    assert h.get("cron_pending_artifact_status") in ("reconciled", "mixed")
    assert "fetch-trends" in (h.get("cron_pending_artifact_reconciled_names") or "")
