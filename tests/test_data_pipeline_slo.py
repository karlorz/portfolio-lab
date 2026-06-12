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


def test_slo_surfaces_fred_source_manifest_failure_reasons() -> None:
    source_manifest = {
        "artifacts": [
            {
                "artifact": "yields.json",
                "provider": "FRED",
                "feed": "series/observations",
                "source_mode": "stale_cached",
                "status": "degraded",
                "failure_reason": "cache_stale",
                "fallback_reason": "rate_limited",
            }
        ]
    }

    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=source_manifest,
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["status"] == "warning"
    assert slo["top_dimension"] == "provider"
    assert slo["dimensions"]["provider"]["degraded_artifacts"] == ["yields.json"]
    assert slo["dimensions"]["provider"]["degraded_reasons"]["yields.json"] == {
        "source_mode": "stale_cached",
        "status": "degraded",
        "failure_reason": "cache_stale",
        "fallback_reason": "rate_limited",
    }
    assert "cache_stale" in slo["dimensions"]["provider"]["message"]


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


def test_slo_distinguishes_provider_reconciliation_divergence_from_outage() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
        provider_reconciliation={
            "status": "warning",
            "failure_type": "provider_divergence",
            "issue_counts": {"adjusted_close_divergence": 1},
            "message": "1 provider divergence detected",
            "top_offenders": [{"symbol": "SPY", "issue": "adjusted_close_divergence"}],
        },
    )

    assert slo["status"] == "warning"
    assert slo["top_dimension"] == "provider_reconciliation"
    assert slo["dimensions"]["provider_reconciliation"]["failure_type"] == "provider_divergence"
    assert slo["dimensions"]["provider_reconciliation"]["outage_provider"] is None
    assert slo["dimensions"]["provider_reconciliation"]["top_offenders"] == [
        {"symbol": "SPY", "issue": "adjusted_close_divergence"}
    ]


def test_slo_classifies_provider_reconciliation_outage_as_critical() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
        provider_reconciliation={
            "status": "unavailable",
            "failure_type": "provider_outage",
            "outage_provider": "Yahoo Fixture",
            "issue_counts": {"provider_outage": 1},
            "message": "Yahoo Fixture returned no rows",
            "top_offenders": [],
        },
    )

    assert slo["status"] == "critical"
    assert slo["top_dimension"] == "provider_reconciliation"
    assert slo["dimensions"]["provider_reconciliation"]["failure_type"] == "provider_outage"
    assert slo["dimensions"]["provider_reconciliation"]["outage_provider"] == "Yahoo Fixture"


def test_slo_surfaces_fred_readiness_warning_from_health_data() -> None:
    health = _health()
    health["data_freshness"]["fred_readiness"] = {
        "status": "warning",
        "readiness": "warn",
        "reason": "missing_fred_api_key",
        "remediation": "Set FRED_API_KEY for lab/paper/live operation.",
    }

    slo = build_data_pipeline_slo(
        health_data=health,
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["status"] == "warning"
    assert slo["top_dimension"] == "fred_readiness"
    assert slo["dimensions"]["fred_readiness"]["reason"] == "missing_fred_api_key"
    assert "FRED_API_KEY" in slo["dimensions"]["fred_readiness"]["message"]


def test_slo_classifies_live_fred_readiness_failure_as_critical() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
        fred_readiness={
            "status": "critical",
            "readiness": "fail",
            "reason": "missing_fred_api_key",
            "remediation": "Set FRED_API_KEY before live operation.",
        },
    )

    assert slo["status"] == "critical"
    assert slo["top_dimension"] == "fred_readiness"
    assert slo["dimensions"]["fred_readiness"]["status"] == "critical"
