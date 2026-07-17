"""Tests for data pipeline SLO summary derivation."""

import pytest

from src.monitor.data_pipeline_slo import build_data_pipeline_slo


def _health(status: str = "fresh") -> dict:
    return {
        "cron_jobs": [{"id": "data", "status": "ok"}],
        "scheduler_status": {"status": "ok", "backends": {}},
        "data_freshness": {
            "SPY": {"status": status, "last_update": "2026-06-11", "days_stale": 0},
        },
    }


_QUALITY_ISSUE_KEYS = (
    "duplicate_dates",
    "empty_symbols",
    "extreme_returns",
    "internal_gaps",
    "invalid_dates",
    "invalid_prices",
    "missing_required_keys",
    "non_monotonic_rows",
    "non_object_records",
    "split_like_returns",
    "stale_latest_dates",
)


def _quality_counts(**overrides: int) -> dict:
    counts = {key: 0 for key in _QUALITY_ISSUE_KEYS}
    counts.update(overrides)
    counts["total"] = overrides.get(
        "total",
        sum(value for key, value in counts.items() if key != "total"),
    )
    return counts


def _quality_summary(status: str = "ok", **issue_counts: int) -> dict:
    return {
        "artifact": "data_quality.json",
        "schema_version": "price-data-quality/v1",
        "generated_at": "2026-06-12T09:00:00Z",
        "status": status,
        "issue_counts": _quality_counts(**issue_counts),
    }


def _source_manifest(
    source_mode: str = "live",
    status: str = "success",
    data_quality: dict | None = None,
    include_data_quality: bool = True,
) -> dict:
    row = {
        "artifact": "prices.json",
        "provider": "Yahoo Finance",
        "feed": "chart/v8",
        "source_mode": source_mode,
        "status": status,
        "symbols": ["SPY", "GLD", "TLT"],
    }
    if include_data_quality:
        row["data_quality"] = data_quality or _quality_summary()
    return {
        "artifacts": [
            row
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


def test_slo_includes_data_quality_ok_dimension() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["status"] == "ok"
    assert slo["dimensions"]["data_quality"]["status"] == "ok"
    assert slo["dimensions"]["data_quality"]["quality_status"] == "ok"
    assert slo["dimensions"]["data_quality"]["artifact"] == "data_quality.json"
    assert slo["dimensions"]["data_quality"]["issue_counts"]["total"] == 0


def test_slo_warns_on_data_quality_anomalous_returns() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(
            data_quality=_quality_summary("warn", split_like_returns=2),
        ),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["status"] == "warning"
    assert slo["top_dimension"] == "data_quality"
    assert slo["dimensions"]["data_quality"]["top_issue"] == "split_like_returns"
    assert slo["dimensions"]["data_quality"]["affected_issue_count"] == 2
    assert slo["dimensions"]["data_quality"]["affected_symbol_count"] == 3
    assert slo["runbook"]["top_cause"]["code"] == "price_quality_anomalous_returns"
    assert "split-like or extreme return" in slo["runbook"]["top_cause"]["action"]


def test_slo_classifies_data_quality_duplicate_dates_as_critical() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(
            data_quality=_quality_summary("fail", duplicate_dates=1),
        ),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["status"] == "critical"
    assert slo["top_dimension"] == "data_quality"
    assert slo["dimensions"]["data_quality"]["status"] == "critical"
    assert slo["dimensions"]["data_quality"]["top_issue"] == "duplicate_dates"
    assert slo["runbook"]["top_cause"]["code"] == "price_quality_duplicate_dates"
    assert "duplicate" in slo["runbook"]["top_cause"]["action"]


def test_slo_warns_when_data_quality_report_is_missing() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(include_data_quality=False),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["status"] == "warning"
    assert slo["top_dimension"] == "data_quality"
    assert slo["dimensions"]["data_quality"]["status"] == "warning"
    assert slo["dimensions"]["data_quality"]["quality_status"] == "missing"
    assert slo["runbook"]["top_cause"]["code"] == "price_quality_report_missing"


@pytest.mark.parametrize(
    ("issue_counts", "expected_code", "expected_action"),
    [
        ({"stale_latest_dates": 2}, "price_quality_stale_cross_section", "stale cross-section"),
        ({"internal_gaps": 3}, "price_quality_internal_gaps", "missing trading dates"),
        ({"extreme_returns": 1}, "price_quality_anomalous_returns", "split-like or extreme return"),
    ],
)
def test_slo_runbook_maps_data_quality_issue_actions(
    issue_counts: dict[str, int],
    expected_code: str,
    expected_action: str,
) -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(
            data_quality=_quality_summary("warn", **issue_counts),
        ),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["runbook"]["top_cause"]["code"] == expected_code
    assert expected_action in slo["runbook"]["top_cause"]["action"]


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
    health["cron_jobs"] = [{"id": "data", "name": "portfolio-lab-data", "status": "error"}]

    slo = build_data_pipeline_slo(
        health_data=health,
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["status"] == "warning"
    assert slo["top_dimension"] == "scheduler"


def test_slo_ignores_health_self_job_error_in_scheduler_dimension() -> None:
    """Sticky portfolio-lab-health errors must not keep scheduler SLO degraded."""
    health = _health()
    health["cron_jobs"] = [
        {
            "id": "tasker:portfolio-lab-health",
            "name": "portfolio-lab-health",
            "status": "error",
            "backend": "tasker",
        },
        {
            "id": "tasker:portfolio-lab-data",
            "name": "portfolio-lab-data",
            "status": "ok",
            "backend": "tasker",
        },
    ]
    health["scheduler_status"] = {"status": "degraded", "backends": {}}

    slo = build_data_pipeline_slo(
        health_data=health,
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    sched = slo["dimensions"]["scheduler"]
    assert sched["failed_jobs"] == 0
    assert sched["status"] == "ok"
    assert sched["scheduler_status"] == "ok"
    assert slo["top_dimension"] != "scheduler"


def test_slo_warns_on_artifact_staleness() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(status="stale"),
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["status"] == "warning"
    assert slo["top_dimension"] == "artifact"


def test_slo_one_critical_data_freshness_child_rolls_up_to_artifact_critical() -> None:
    """Any critical data_freshness child must not silently yield artifact warning only."""
    slo = build_data_pipeline_slo(
        health_data={
            "cron_jobs": [{"id": "data", "status": "ok"}],
            "scheduler_status": {"status": "ok", "backends": {}},
            "data_freshness": {
                "^VIX3M": {
                    "status": "critical",
                    "last_update": "2026-06-26",
                    "market_lag_days": 6,
                    "latest_available_market_date": "2026-07-02",
                },
            },
        },
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    artifact = slo["dimensions"]["artifact"]
    assert artifact["status"] == "critical"
    assert artifact["critical_count"] == 1
    assert "^VIX3M" in artifact.get("critical_artifacts", [])
    assert "1 critical" in artifact["message"]
    # Must not look like the old silent-downgrade shape
    assert not (
        artifact["status"] == "warning"
        and artifact["critical_count"] == 1
        and "critical_count_threshold" not in artifact
    )
    assert slo["status"] == "critical"
    assert slo["top_dimension"] == "artifact"
    top_cause = slo["runbook"]["top_cause"]
    assert top_cause is not None
    assert top_cause["code"] == "critical_data_freshness"
    assert top_cause["severity"] == "critical"
    assert "^VIX3M" in (top_cause.get("action") or "")


def test_slo_warns_when_source_manifest_is_newer_than_public_index() -> None:
    source_manifest = {
        **_source_manifest(),
        "generated_at": "2026-06-12T09:05:25.028Z",
    }
    public_index = {
        "generated_at": "2026-06-12T03:12:34.220521+00:00",
        "entries": [],
    }

    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=source_manifest,
        public_index=public_index,
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["status"] == "warning"
    assert slo["top_dimension"] == "artifact"
    assert slo["dimensions"]["artifact"]["source_manifest_index_status"] == "stale_index"
    assert "index.json" in slo["dimensions"]["artifact"]["message"]
    assert slo["runbook"]["top_cause"]["code"] == "stale_public_data_index"


def test_slo_accepts_public_index_at_least_as_new_as_source_manifest() -> None:
    source_manifest = {
        **_source_manifest(),
        "generated_at": "2026-06-12T09:05:25.028Z",
    }
    public_index = {
        "generated_at": "2026-06-12T09:06:00+00:00",
        "entries": [],
    }

    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=source_manifest,
        public_index=public_index,
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["status"] == "ok"
    assert slo["dimensions"]["artifact"]["source_manifest_index_status"] == "ok"


def test_slo_warns_without_crashing_on_malformed_index_timestamps() -> None:
    source_manifest = {
        **_source_manifest(),
        "generated_at": "not-a-date",
    }
    public_index = {
        "generated_at": "2026-06-12T09:06:00+00:00",
        "entries": [],
    }

    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=source_manifest,
        public_index=public_index,
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["status"] == "warning"
    assert slo["top_dimension"] == "artifact"
    assert slo["dimensions"]["artifact"]["source_manifest_index_status"] == "unknown_timestamp"
    assert slo["runbook"]["top_cause"]["code"] == "public_data_timestamp_unparseable"


def test_slo_warns_on_stale_required_signals_and_counts_unavailable() -> None:
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
    assert "stale" in slo["dimensions"]["signal"]["message"]
    assert "unavailable" in slo["dimensions"]["signal"]["message"]


def test_slo_signal_dim_warns_on_unavailable_without_stale() -> None:
    """Unavailable-only must not report OK / 'required signals fresh'."""
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={
            "stale_signals": [],
            "unavailable_signals": [
                "collar",
                "bond_momentum",
                "kurtosis_regime",
                "calendar_seasonality",
            ],
        },
    )

    signal = slo["dimensions"]["signal"]
    assert signal["status"] == "warning"
    assert signal["unavailable_count"] == 4
    assert signal["stale_count"] == 0
    assert "unavailable" in signal["message"]
    assert "required signals fresh" not in signal["message"]
    assert "collar" in signal.get("unavailable_signals", [])


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


def test_slo_classifies_rejected_alpaca_feed_entitlement_as_critical(
    monkeypatch,
) -> None:
    """Live mode still fail-closes on missing Alpaca feed entitlement."""
    monkeypatch.setenv("PORTFOLIO_LAB_MODE", "live")
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
        alpaca_feed_entitlement={
            "configured_feed": "iex",
            "effective_feed": "iex",
            "entitlement": "unknown",
            "delayed": False,
            "acceptable_for_live": False,
            "policy_decision": "reject",
            "reason": "missing_entitlement",
        },
    )

    assert slo["status"] == "critical"
    assert slo["top_dimension"] == "alpaca_feed_entitlement"
    assert slo["dimensions"]["alpaca_feed_entitlement"]["acceptable_for_live"] is False
    assert slo["dimensions"]["alpaca_feed_entitlement"]["reason"] == "missing_entitlement"
    assert slo["runbook"]["top_cause"]["code"] == "alpaca_feed_entitlement_rejected"
    assert "entitlement" in slo["runbook"]["top_cause"]["action"]


def test_slo_missing_alpaca_entitlement_is_warning_in_local_lab(
    monkeypatch,
) -> None:
    """Research hosts without ALPACA_FEED_ENTITLEMENT must not critical the SLO."""
    monkeypatch.setenv("PORTFOLIO_LAB_MODE", "local")
    monkeypatch.delenv("CRON_BACKEND", raising=False)
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
        alpaca_feed_entitlement={
            "configured_feed": "iex",
            "effective_feed": "iex",
            "entitlement": "unknown",
            "delayed": False,
            "acceptable_for_live": False,
            "policy_decision": "reject",
            "reason": "missing_entitlement",
        },
    )

    dim = slo["dimensions"]["alpaca_feed_entitlement"]
    assert dim["status"] == "warning"
    assert dim["intentional_lab_gap"] is True
    assert dim["blocking"] is False
    assert dim["reason"] == "missing_entitlement"
    # Overall must not be critical solely due to this lab gap.
    assert slo["status"] != "critical"
    assert slo["status"] in {"ok", "warning", "degraded", "unknown"}


def test_slo_warns_on_unavailable_market_data_consistency() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
        market_data_consistency={
            "status": "unavailable",
            "reason": "alpaca_not_configured",
            "checked_at": "2026-06-12T08:43:07.177011+00:00",
            "rows": [],
            "warnings": [],
        },
    )

    assert slo["status"] == "warning"
    assert slo["top_dimension"] == "market_data_consistency"
    assert slo["dimensions"]["market_data_consistency"]["status"] == "warning"
    assert slo["dimensions"]["market_data_consistency"]["reason"] == "alpaca_not_configured"
    assert slo["runbook"]["top_cause"]["code"] == "market_data_consistency_unavailable"


def test_slo_accepts_live_diagnostics_ok_cases() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
        alpaca_feed_entitlement={
            "configured_feed": "sip",
            "effective_feed": "sip",
            "entitlement": "sip",
            "delayed": False,
            "acceptable_for_live": True,
            "policy_decision": "accept",
        },
        market_data_consistency={
            "status": "ok",
            "rows": [{"symbol": "SPY", "status": "ok"}],
            "warnings": [],
        },
    )

    assert slo["status"] == "ok"
    assert slo["dimensions"]["alpaca_feed_entitlement"]["status"] == "ok"
    assert slo["dimensions"]["market_data_consistency"]["status"] == "ok"


def test_slo_runbook_maps_yahoo_provider_failure_without_leaking_fallback_details() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest={
            "artifacts": [
                {
                    "artifact": "prices.json",
                    "provider": "Yahoo Finance",
                    "source_mode": "last_good",
                    "status": "failed",
                    "failure_reason": "rate_limited",
                    "fallback_reason": "https://query.example.test/?token=super-secret-token",
                }
            ]
        },
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    runbook = slo["runbook"]

    assert runbook["top_cause"]["code"] == "yahoo_provider_failure"
    assert runbook["top_cause"]["dimension"] == "provider"
    assert runbook["top_cause"]["artifact"] == "prices.json"
    assert "Yahoo" in runbook["top_cause"]["action"]
    assert "super-secret-token" not in str(runbook)


def test_slo_runbook_maps_fred_missing_key_readiness() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
        fred_readiness={
            "status": "critical",
            "readiness": "fail",
            "reason": "missing_fred_api_key",
            "remediation": "Set FRED_API_KEY before live operation; current secret is super-secret-fred-key.",
        },
    )

    runbook = slo["runbook"]

    assert runbook["top_cause"]["code"] == "fred_missing_api_key"
    assert "FRED_API_KEY" in runbook["top_cause"]["action"]
    assert "super-secret-fred-key" not in str(runbook)


def test_slo_runbook_maps_synthetic_fred_fallback() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest={
            "artifacts": [
                {
                    "artifact": "yields.json",
                    "provider": "FRED",
                    "source_mode": "synthetic",
                    "status": "degraded",
                    "failure_reason": "missing_api_key",
                    "fallback_reason": "missing_api_key",
                }
            ]
        },
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["runbook"]["top_cause"]["code"] == "fred_synthetic_fallback"
    assert "synthetic" in slo["runbook"]["top_cause"]["action"]


def test_slo_runbook_maps_stale_quote_artifact() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(status="stale"),
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["runbook"]["top_cause"]["code"] == "stale_quote"
    assert slo["runbook"]["top_cause"]["dimension"] == "artifact"
    assert "prices" in slo["runbook"]["top_cause"]["action"]


def test_slo_runbook_maps_provider_divergence() -> None:
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

    assert slo["runbook"]["top_cause"]["code"] == "provider_divergence"
    assert "SPY" in slo["runbook"]["top_cause"]["action"]


def test_slo_runbook_maps_stale_source_manifest() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(),
        source_manifest={
            "status": "stale",
            "failure_reason": "stale_manifest",
            "artifacts": [
                {
                    "artifact": "prices.json",
                    "provider": "Yahoo Finance",
                    "source_mode": "live",
                    "status": "success",
                }
            ],
        },
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
    )

    assert slo["runbook"]["top_cause"]["code"] == "stale_source_manifest"
    assert "source_manifest.json" in slo["runbook"]["top_cause"]["action"]


def test_slo_runbook_top_cause_prefers_critical_over_warning() -> None:
    slo = build_data_pipeline_slo(
        health_data=_health(status="stale"),
        source_manifest=_source_manifest(),
        public_index={"entries": []},
        signal_staleness={"stale_signals": [], "unavailable_signals": []},
        provider_reconciliation={
            "status": "unavailable",
            "failure_type": "provider_outage",
            "outage_provider": "Yahoo Finance",
            "issue_counts": {"provider_outage": 1},
            "message": "Yahoo Finance returned no rows",
            "top_offenders": [],
        },
    )

    assert slo["runbook"]["top_cause"]["code"] == "provider_outage"
    assert slo["runbook"]["top_cause"]["severity"] == "critical"
