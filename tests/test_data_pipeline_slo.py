"""Tests for data pipeline SLO summary derivation."""

from src.monitor.data_pipeline_slo import build_data_pipeline_slo


def _health(status: str = "fresh") -> dict:
    return {
        "cron_jobs": [{"id": "data", "status": "ok"}],
        "scheduler_status": {"status": "ok", "backends": {}},
        "data_freshness": {
            "SPY": {"status": status, "last_update": "2026-06-11", "days_stale": 0},
        },
    }


def _source_manifest(source_mode: str = "live", status: str = "success") -> dict:
    return {
        "artifacts": [
            {
                "artifact": "prices.json",
                "provider": "Yahoo Finance",
                "feed": "chart/v8",
                "source_mode": source_mode,
                "status": status,
            }
        ]
    }


def test_slo_healthy_when_all_dimensions_ok() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["status"] == "ok"
    assert slo["top_dimension"] is None


def test_slo_warns_on_provider_fallback() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(source_mode="last_good", status="failed"),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["status"] == "warning"
    assert slo["top_dimension"] == "provider"
    assert slo["dimensions"]["provider"]["degraded_artifacts"] == ["prices.json"]


def test_slo_reports_scheduler_failure_as_top_dimension() -> None:
    health = _health()
    health["cron_jobs"] = [{"id": "data", "status": "error"}]

    slo = build_data_pipeline_slo(
        health_data=health,
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["status"] == "warning"
    assert slo["top_dimension"] == "scheduler"


def test_slo_warns_on_artifact_staleness() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(status="stale"),
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["status"] == "warning"
    assert slo["top_dimension"] == "artifact"


def test_slo_warns_on_stale_required_signals_without_penalizing_unavailable_optional() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={
            "stale_signals": ["garch_cvar"],
            "unavailable_signals": ["two_stage_regime"],
        },
    )

    assert slo["status"] == "warning"
    assert slo["top_dimension"] == "signal"
    assert slo["dimensions"]["signal"]["stale_count"] == 1
    assert slo["dimensions"]["signal"]["unavailable_count"] == 1
