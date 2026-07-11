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
        "failed_cron_jobs": 1,
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
