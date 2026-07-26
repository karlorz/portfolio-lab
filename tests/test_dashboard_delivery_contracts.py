from pathlib import Path
from src.dashboard.generator import _attach_signal_metadata, _compact_health_summary


def test_attach_signal_metadata_sets_timestamp_alias_from_generated_at():
    payload = _attach_signal_metadata({"regime": {"regime": "normal"}}, generated_at="2026-06-11T12:00:00")

    assert payload["generated_at"] == "2026-06-11T12:00:00"
    assert payload["timestamp"] == "2026-06-11T12:00:00"
    assert payload["regime"] == {"regime": "normal"}


def test_compact_health_summary_drops_unbounded_sections_and_counts_statuses():
    summary = _compact_health_summary(
        {
            "system_status": "critical",
            "generated_at": "2026-06-11T12:00:00",
            "cron_jobs": [
                {"name": "portfolio-lab-health", "status": "error"},
                {"name": "portfolio-lab-dashboard", "status": "ok"},
            ],
            "data_freshness": {
                "SPY": {"status": "critical"},
                "GLD": {"status": "fresh"},
                "TLT": {"status": "stale"},
            },
            "scheduler_status": {"status": "degraded"},
            "data_pipeline_slo": {
                "status": "critical",
                "top_dimension": "data_quality",
                "runbook": {"top_cause": {"code": "stale_prices"}},
            },
        }
    )

    assert summary == {
        "status": "critical",
        "generated_at": "2026-06-11T12:00:00",
        "cron_job_count": 2,
        # portfolio-lab-health self-errors are excluded from failed rollup so
        # sticky tasker mirrors of prior health exits cannot inflate counts.
        "failed_cron_jobs": 0,
        # Batch EE: dual-signal pending vs artifact-fresh keys always projected
        # when cron_jobs is present (zeros / none when no pending rows).
        "cron_pending_artifact_reconciled_jobs": 0,
        "cron_pending_never_run_jobs": 0,
        "cron_pending_artifact_reconciled_names": None,
        "cron_pending_never_run_names": None,
        "cron_pending_artifact_sample": None,
        "cron_pending_artifact_status": "none",
        "cron_pending_artifact_badge": "artifact_ok=0 pending_never_run=0",
        "stale_data_count": 2,
        "scheduler_status": "degraded",
        "data_pipeline_slo_status": "critical",
        "top_slo_dimension": "data_quality",
        "top_slo_cause_code": "stale_prices",
    }


def test_compact_health_summary_preserves_error_without_full_payload():
    summary = _compact_health_summary({"status": "error", "error": "health subsystem unavailable"})

    assert summary == {
        "status": "error",
        "error": "health subsystem unavailable",
    }

def test_finalize_signal_metadata_stamps_generator_git_sha(monkeypatch):
    from src.dashboard import generator as gen

    monkeypatch.setattr(gen, "_generator_git_sha_short", lambda: "abc1234dead")
    out = gen._finalize_signal_metadata({"target_allocations": {"SPY": 0.46}})
    assert out["generator_git_sha"] == "abc1234dead"
    assert "generated_at" in out
    assert "timestamp" in out


def test_ops_regen_makefile_target_exists():
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "ops-regen:" in makefile
    assert "make ops-regen" in makefile or "ops-regen" in makefile
    assert "wiki-sync" in makefile
    # target should invoke dashboard, wiki-sync, health
    assert "ops-regen" in makefile
    body = makefile.split("ops-regen:")[1].split("# ──")[0]
    assert "dashboard" in body
    assert "wiki-sync" in body
    assert "health" in body
