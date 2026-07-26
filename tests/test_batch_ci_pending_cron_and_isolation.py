"""Batch CI: pending never-run cron + pytest public isolation vs live private SSOT."""

from __future__ import annotations

import json
from pathlib import Path

from src.monitor.hermes_cron import normalize_cron_status, summarize_backend


def test_normalize_cron_status_preserves_pending():
    assert normalize_cron_status("pending") == "pending"
    assert normalize_cron_status("queued") == "pending"
    assert normalize_cron_status("waiting") == "pending"
    assert normalize_cron_status("never_run") == "pending"
    assert normalize_cron_status("success") == "ok"
    assert normalize_cron_status("mystery") == "unknown"


def test_summarize_backend_pending_never_run_does_not_degrade():
    jobs = [
        {
            "name": "portfolio-lab-data",
            "status": "ok",
            "enabled": True,
            "manual_only": False,
            "state": "scheduled",
            "last_run": "2026-07-21T19:00:00+00:00",
        },
        {
            "name": "portfolio-lab-fetch-trends",
            "status": "pending",
            "enabled": True,
            "manual_only": False,
            "state": "scheduled",
            "last_run": None,
        },
    ]
    summary = summarize_backend(
        backend="tasker",
        source="/tmp/cron_status.json",
        jobs=jobs,
    )
    assert summary["status"] == "ok"
    assert summary.get("unknown_active_jobs") is None
    assert summary.get("pending_never_run_jobs") == 1


def test_summarize_backend_true_unknown_still_degrades():
    jobs = [
        {
            "name": "weird-job",
            "status": "unknown",
            "enabled": True,
            "manual_only": False,
            "state": "scheduled",
            "last_run": None,
        },
    ]
    summary = summarize_backend(
        backend="tasker",
        source="/tmp/cron_status.json",
        jobs=jobs,
    )
    assert summary["status"] == "degraded"
    assert summary.get("unknown_active_jobs") == 1


def test_incident_write_summary_skips_pytest_public_when_private_live(
    tmp_path, monkeypatch
):
    """Live private summary must not embed plab-pytest public_path."""
    from src.monitor import incident_manager as im

    # Private is "live-shaped" (no plab-pytest in path); public is isolation.
    live_private = tmp_path / "repo_data"
    live_private.mkdir()
    pytest_public = tmp_path / "plab-pytest-public.fake" / "data"
    pytest_public.mkdir(parents=True)

    monkeypatch.setattr(im, "PUBLIC_DATA_DIR", pytest_public)
    # No live WWW under this tmp tree — dual-write should skip isolation
    mgr = im.IncidentManager(
        log_path=live_private / "incidents.jsonl",
        summary_path=live_private / "incidents.json",
        kill_switch_path=live_private / "kill_switch.json",
        escalation_enabled=False,
    )
    summary = mgr.write_summary()
    body = json.loads((live_private / "incidents.json").read_text(encoding="utf-8"))
    pc = body.get("provenance_completeness") or summary.get("provenance_completeness") or {}
    public_path = str(pc.get("public_path") or "")
    assert "plab-pytest" not in public_path
    # Isolation public must not receive a dual-write copy
    assert not (pytest_public / "incidents.json").exists()
    # Must not clobber live operator tree from a non-live private path
    assert "/var/www/portfolio-lab" not in public_path
    assert pc.get("dual_write_attempted") is False or pc.get("paths_identical") is True


def test_incident_write_summary_still_dual_writes_when_both_isolated(
    tmp_path, monkeypatch
):
    """Both sides under isolation (normal pytest) still dual-write."""
    from src.monitor import incident_manager as im

    private = tmp_path / "plab-pytest-private" / "data"
    public = tmp_path / "plab-pytest-public" / "data"
    private.mkdir(parents=True)
    public.mkdir(parents=True)
    monkeypatch.setattr(im, "PUBLIC_DATA_DIR", public)

    mgr = im.IncidentManager(
        log_path=private / "incidents.jsonl",
        summary_path=private / "incidents.json",
        kill_switch_path=private / "kill_switch.json",
        escalation_enabled=False,
    )
    summary = mgr.write_summary()
    assert (public / "incidents.json").exists()
    assert summary.get("open_count") == 0
    pc = summary["provenance_completeness"]
    assert pc["dual_write_ok"] is True
    assert "plab-pytest" in str(pc.get("public_path") or "")
