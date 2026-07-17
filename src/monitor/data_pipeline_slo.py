"""Data pipeline SLO summary for dashboard health output."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DATA_PIPELINE_SLO_SCHEMA_VERSION = "data-pipeline-slo/v1"
_STATUS_RANK = {"ok": 0, "unknown": 1, "warning": 2, "critical": 3}
_SAFE_REASON_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
_DATA_QUALITY_ISSUE_KEYS = (
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
    "total",
)
_DATA_QUALITY_ISSUE_PRIORITY = (
    "duplicate_dates",
    "invalid_prices",
    "invalid_dates",
    "missing_required_keys",
    "non_monotonic_rows",
    "non_object_records",
    "empty_symbols",
    "stale_latest_dates",
    "internal_gaps",
    "extreme_returns",
    "split_like_returns",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_source_manifest(public_dir: Path) -> dict[str, Any]:
    return _load_json(public_dir / "source_manifest.json")


def load_data_quality_report(public_dir: Path) -> dict[str, Any]:
    return _load_json(public_dir / "data_quality.json")


def load_public_index(public_dir: Path) -> dict[str, Any]:
    return _load_json(public_dir / "index.json")


def load_signal_staleness(public_dir: Path) -> dict[str, Any]:
    signals = _load_json(public_dir / "signals.json")
    staleness = signals.get("staleness")
    return staleness if isinstance(staleness, dict) else {}


def load_rebalance_health(public_dir: Path) -> dict[str, Any]:
    return _load_json(public_dir / "rebalance_health.json")


def _scheduler_dimension(health_data: Mapping[str, Any]) -> dict[str, Any]:
    from src.monitor.hermes_cron import is_health_self_job, rollup_failed_cron_jobs

    scheduler = health_data.get("scheduler_status")
    scheduler_status = scheduler.get("status") if isinstance(scheduler, Mapping) else "unknown"
    cron_jobs = health_data.get("cron_jobs")
    jobs = cron_jobs if isinstance(cron_jobs, list) else []
    # Exclude portfolio-lab-health self-errors (sticky tasker mirror of prior exits).
    failed_jobs = rollup_failed_cron_jobs(jobs)

    # If scheduler_status is degraded solely because of the self-job, treat as ok
    # for the dimension severity when no other failures remain.
    self_only_error = (
        not failed_jobs
        and any(
            isinstance(job, Mapping)
            and job.get("status") == "error"
            and is_health_self_job(job)
            for job in jobs
        )
    )
    effective_scheduler_status = scheduler_status
    if self_only_error and scheduler_status in {"degraded", "warning"}:
        effective_scheduler_status = "ok"

    if len(failed_jobs) > 2:
        status = "critical"
    elif failed_jobs or effective_scheduler_status in {
        "degraded",
        "error",
        "warning",
        "unavailable",
    }:
        status = "warning"
    elif effective_scheduler_status == "unknown":
        status = "unknown"
    else:
        status = "ok"
    return {
        "status": status,
        "scheduler_status": effective_scheduler_status,
        "failed_jobs": len(failed_jobs),
        "message": (
            f"{len(failed_jobs)} scheduler job(s) failed"
            if failed_jobs
            else f"scheduler {effective_scheduler_status}"
        ),
    }


def _is_price_quality_warn_only_row(row: Mapping[str, Any]) -> bool:
    """True when live fetch succeeded and only nested price-quality is warn.

    fetch-data marks prices.json / prices_compact.json status=degraded when
    overall_status=warn. That is an advisory quality signal already covered by
    the data_quality SLO dimension — not a provider outage/fallback.
    """
    if str(row.get("source_mode") or "") != "live":
        return False
    if str(row.get("status") or "") not in {"degraded", "warning", "warn"}:
        return False
    if row.get("failure_reason") not in (None, "", "null"):
        return False
    if row.get("fallback_reason") not in (None, "", "null"):
        return False
    quality = row.get("data_quality")
    if not isinstance(quality, Mapping):
        return False
    return str(quality.get("status") or "").lower() in {"warn", "warning"}


def _is_intentional_lab_provider_gap_row(row: Mapping[str, Any]) -> bool:
    """True for known non-outage lab gaps that have their own SLO dimensions.

    - Live prices degraded solely from quality warn → data_quality dimension
    - yields.json synthetic + missing_api_key → fred_readiness / FRED lab gap
    """
    if _is_price_quality_warn_only_row(row):
        return True
    artifact = str(row.get("artifact") or "")
    if artifact != "yields.json":
        return False
    if str(row.get("source_mode") or "") != "synthetic":
        return False
    reason = str(row.get("failure_reason") or row.get("fallback_reason") or "")
    return reason in {"missing_api_key", "missing_fred_api_key"}


def _provider_dimension(source_manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    artifacts = source_manifest.get("artifacts") if isinstance(source_manifest, Mapping) else None
    rows = [row for row in artifacts if isinstance(row, Mapping)] if isinstance(artifacts, list) else []
    manifest_status = str(source_manifest.get("status", "ok")) if isinstance(source_manifest, Mapping) else "unknown"
    manifest_failure_reason = source_manifest.get("failure_reason") if isinstance(source_manifest, Mapping) else None
    manifest_degraded = manifest_status in {"stale", "degraded", "failed", "error"} or manifest_failure_reason == "stale_manifest"
    if not rows:
        return {
            "status": "warning" if manifest_degraded else "unknown",
            "manifest_status": manifest_status,
            "manifest_failure_reason": manifest_failure_reason,
            "degraded_artifacts": [],
            "message": "source manifest stale" if manifest_degraded else "source manifest missing or empty",
        }
    degraded_rows = [
        row
        for row in rows
        if (row.get("status") != "success" or row.get("source_mode") != "live")
        and not _is_intentional_lab_provider_gap_row(row)
    ]
    quality_warn_only = [
        str(row.get("artifact"))
        for row in rows
        if _is_price_quality_warn_only_row(row)
    ]
    lab_gap_artifacts = [
        str(row.get("artifact"))
        for row in rows
        if _is_intentional_lab_provider_gap_row(row)
    ]
    degraded = [str(row.get("artifact")) for row in degraded_rows]
    degraded_reasons = {
        str(row.get("artifact")): {
            "source_mode": row.get("source_mode"),
            "status": row.get("status"),
            "failure_reason": row.get("failure_reason"),
            "fallback_reason": row.get("fallback_reason"),
        }
        for row in degraded_rows
    }
    reason_parts = [
        f"{artifact}: {details.get('failure_reason') or details.get('fallback_reason') or details.get('source_mode')}"
        for artifact, details in degraded_reasons.items()
    ]
    status = "warning" if degraded or manifest_degraded else "ok"
    payload: dict[str, Any] = {
        "status": status,
        "manifest_status": manifest_status,
        "manifest_failure_reason": manifest_failure_reason,
        "degraded_artifacts": degraded,
        "degraded_reasons": degraded_reasons,
        "message": (
            "source manifest stale"
            if manifest_degraded and not degraded
            else
            f"provider degraded for {', '.join(degraded)} ({'; '.join(reason_parts)})"
            if degraded
            else "providers live"
        ),
    }
    if quality_warn_only:
        payload["quality_warn_only_artifacts"] = quality_warn_only
    if lab_gap_artifacts:
        payload["intentional_lab_gap_artifacts"] = lab_gap_artifacts
    return payload


def _parse_generated_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _source_manifest_index_freshness(
    source_manifest: Mapping[str, Any] | None,
    public_index: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_generated = source_manifest.get("generated_at") if isinstance(source_manifest, Mapping) else None
    index_generated = public_index.get("generated_at") if isinstance(public_index, Mapping) else None
    if not source_generated and not index_generated:
        return {
            "status": "ok",
            "source_manifest_index_status": "not_checked",
            "message": "source manifest/index freshness not checked",
        }

    source_dt = _parse_generated_at(source_generated)
    index_dt = _parse_generated_at(index_generated)
    if source_dt is None or index_dt is None:
        return {
            "status": "warning",
            "source_manifest_index_status": "unknown_timestamp",
            "source_manifest_generated_at": source_generated,
            "index_generated_at": index_generated,
            "message": "could not compare source_manifest.json and index.json generated_at timestamps",
        }
    if source_dt > index_dt:
        return {
            "status": "warning",
            "source_manifest_index_status": "stale_index",
            "source_manifest_generated_at": source_generated,
            "index_generated_at": index_generated,
            "message": "index.json is older than source_manifest.json",
        }
    return {
        "status": "ok",
        "source_manifest_index_status": "ok",
        "source_manifest_generated_at": source_generated,
        "index_generated_at": index_generated,
        "message": "index.json is current with source_manifest.json",
    }


def _artifact_dimension(
    health_data: Mapping[str, Any],
    public_index: Mapping[str, Any] | None,
    source_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    data_freshness = health_data.get("data_freshness")
    freshness = data_freshness if isinstance(data_freshness, Mapping) else {}
    critical = [
        name
        for name, row in freshness.items()
        if isinstance(row, Mapping) and row.get("status") == "critical"
    ]
    stale = [
        name
        for name, row in freshness.items()
        if isinstance(row, Mapping) and row.get("status") == "stale"
    ]

    entries = public_index.get("entries") if isinstance(public_index, Mapping) else None
    missing_market_entries = [
        entry.get("filename")
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("category") == "market_data"
        and entry.get("status") == "missing"
    ] if isinstance(entries, list) else []

    source_index_freshness = _source_manifest_index_freshness(source_manifest, public_index)
    source_index_status = source_index_freshness.get("status")

    # Highest-severity rollup: any critical data_freshness child makes the
    # artifact dimension critical (no silent count-threshold downgrade).
    if critical or missing_market_entries:
        status = "critical"
    elif stale or source_index_status == "warning":
        status = "warning"
    else:
        status = "ok"
    message = (
        source_index_freshness["message"]
        if source_index_status == "warning" and not critical and not missing_market_entries
        else f"{len(critical)} critical, {len(stale)} stale artifacts"
        if critical or stale
        else "artifacts fresh"
        if not missing_market_entries
        else f"{len(missing_market_entries)} missing market-data index entries"
    )
    return {
        **source_index_freshness,
        "status": status,
        "critical_count": len(critical),
        "critical_artifacts": critical[:10],
        "stale_count": len(stale),
        "stale_artifacts": stale[:10],
        "missing_market_entries": missing_market_entries,
        "message": message,
    }


def _actionable_unavailable_signals(
    signal_staleness: Mapping[str, Any],
    unavailable_signals: list[str],
) -> tuple[list[str], int]:
    """Split unavailable signals into actionable vs intentional lab gaps.

    Matches ``classify_signal_staleness``: FRED-unconfigured / ML-off gaps must
    not keep the signal SLO dimension in permanent warning.
    """
    ownership = signal_staleness.get("unavailable_ownership")
    if not (isinstance(ownership, list) and ownership):
        try:
            from src.monitor.signal_ownership import annotate_unavailable_signals

            ownership = annotate_unavailable_signals(unavailable_signals)
        except ImportError:
            ownership = []

    if isinstance(ownership, list) and ownership:
        actionable = [
            str(row.get("signal"))
            for row in ownership
            if isinstance(row, Mapping)
            and not (
                row.get("intentional_lab_gap")
                or row.get("intentional_when_ml_off")
            )
        ]
        intentional_count = max(0, len(unavailable_signals) - len(actionable))
        return actionable, intentional_count

    # No ownership metadata: treat full list as actionable (fail-closed).
    return list(unavailable_signals), 0


def _signal_dimension(signal_staleness: Mapping[str, Any] | None) -> dict[str, Any]:
    stale = signal_staleness.get("stale_signals") if isinstance(signal_staleness, Mapping) else None
    unavailable = signal_staleness.get("unavailable_signals") if isinstance(signal_staleness, Mapping) else None
    stale_signals = [str(item) for item in stale] if isinstance(stale, list) else []
    unavailable_signals = [str(item) for item in unavailable] if isinstance(unavailable, list) else []

    actionable_unavailable: list[str] = []
    intentional_lab_gap_count = 0
    if unavailable_signals and isinstance(signal_staleness, Mapping):
        actionable_unavailable, intentional_lab_gap_count = _actionable_unavailable_signals(
            signal_staleness, unavailable_signals
        )
    elif unavailable_signals:
        actionable_unavailable = list(unavailable_signals)

    # Stale or actionable unavailable → warning. Intentional lab gaps alone → ok.
    if stale_signals or actionable_unavailable:
        status = "warning"
    else:
        status = "ok"

    if stale_signals and actionable_unavailable:
        message = (
            f"{len(stale_signals)} stale signal(s); "
            f"{len(actionable_unavailable)} unavailable signal(s)"
        )
    elif stale_signals:
        message = f"{len(stale_signals)} stale required signal(s)"
    elif actionable_unavailable:
        message = f"{len(actionable_unavailable)} unavailable signal(s) (not all-fresh)"
    elif intentional_lab_gap_count:
        message = (
            f"required signals fresh "
            f"({intentional_lab_gap_count} intentional lab gaps skipped)"
        )
    else:
        message = "required signals fresh"

    payload: dict[str, Any] = {
        "status": status,
        "stale_count": len(stale_signals),
        "unavailable_count": len(unavailable_signals),
        "actionable_unavailable_count": len(actionable_unavailable),
        "intentional_lab_gap_count": intentional_lab_gap_count,
        "stale_signals": stale_signals[:10],
        "unavailable_signals": unavailable_signals[:10],
        "actionable_unavailable_signals": actionable_unavailable[:10],
        "message": message,
    }
    if intentional_lab_gap_count and not actionable_unavailable and not stale_signals:
        payload["intentional_lab_gaps_only"] = True
    return payload


def _price_manifest_rows(source_manifest: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    artifacts = source_manifest.get("artifacts") if isinstance(source_manifest, Mapping) else None
    rows = [row for row in artifacts if isinstance(row, Mapping)] if isinstance(artifacts, list) else []
    price_rows = [row for row in rows if str(row.get("artifact", "")).startswith("prices")]
    return price_rows or rows


def _select_data_quality_row(
    source_manifest: Mapping[str, Any] | None,
    data_quality_report: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    rows = _price_manifest_rows(source_manifest)
    preferred_row: Mapping[str, Any] | None = None
    if isinstance(data_quality_report, Mapping) and data_quality_report:
        for artifact_name in ("prices.json", "prices_compact.json"):
            for row in rows:
                if row.get("artifact") == artifact_name:
                    return row, data_quality_report
        return (rows[0] if rows else None), data_quality_report
    for artifact_name in ("prices.json", "prices_compact.json"):
        for row in rows:
            if row.get("artifact") != artifact_name:
                continue
            preferred_row = preferred_row or row
            quality = row.get("data_quality")
            if isinstance(quality, Mapping):
                return row, quality
    for row in rows:
        quality = row.get("data_quality")
        if isinstance(quality, Mapping):
            return row, quality
    return (preferred_row or rows[0], None) if rows else (None, None)


def _data_quality_status(quality_status: str) -> str:
    if quality_status == "ok":
        return "ok"
    if quality_status in {"warn", "warning"}:
        return "warning"
    if quality_status in {"fail", "failed", "critical"}:
        return "critical"
    if quality_status in {"missing", "unavailable"}:
        return "warning"
    return "unknown"


def _data_quality_issue_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    counts = {
        key: count
        for key in _DATA_QUALITY_ISSUE_KEYS
        if isinstance((count := value.get(key)), int) and not isinstance(count, bool) and count >= 0
    }
    if counts and "total" not in counts:
        counts["total"] = sum(count for key, count in counts.items() if key != "total")
    return counts


def _top_data_quality_issue(issue_counts: Mapping[str, int]) -> str | None:
    for issue in _DATA_QUALITY_ISSUE_PRIORITY:
        if issue_counts.get(issue, 0) > 0:
            return issue
    for issue, count in issue_counts.items():
        if issue != "total" and count > 0:
            return issue
    return None


def _manifest_symbol_count(row: Mapping[str, Any] | None) -> int:
    symbols = row.get("symbols") if isinstance(row, Mapping) else None
    if isinstance(symbols, list):
        return len([symbol for symbol in symbols if isinstance(symbol, str) and symbol])
    return 0


def _data_quality_dimension(
    source_manifest: Mapping[str, Any] | None,
    data_quality_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row, quality = _select_data_quality_row(source_manifest, data_quality_report)
    quality_status = (
        str(quality.get("status") or quality.get("overall_status") or "missing").lower()
        if isinstance(quality, Mapping)
        else "missing"
    )
    status = _data_quality_status(quality_status)
    issue_counts = _data_quality_issue_counts(quality.get("issue_counts") if isinstance(quality, Mapping) else None)
    top_issue = _top_data_quality_issue(issue_counts)
    affected_issue_count = issue_counts.get(top_issue, 0) if top_issue else 0
    affected_symbol_count = _manifest_symbol_count(row) if status != "ok" else 0
    artifact = quality.get("artifact") if isinstance(quality, Mapping) else None
    artifact_name = str(artifact or "data_quality.json")
    source_artifact = row.get("artifact") if isinstance(row, Mapping) else None

    if quality_status == "missing":
        message = f"price data quality report missing for {affected_symbol_count} tracked symbol(s)"
    elif quality_status == "unavailable":
        message = f"price data quality report unavailable for {affected_symbol_count} tracked symbol(s)"
    elif status == "ok":
        message = "price data quality ok"
    elif top_issue:
        message = (
            f"price data quality {quality_status}: {top_issue}={affected_issue_count} "
            f"across {affected_symbol_count} tracked symbol(s)"
        )
    else:
        message = f"price data quality {quality_status}"

    return {
        "status": status,
        "quality_status": quality_status,
        "artifact": artifact_name,
        "source_artifact": str(source_artifact) if source_artifact else None,
        "schema_version": quality.get("schema_version") if isinstance(quality, Mapping) else None,
        "generated_at": quality.get("generated_at") if isinstance(quality, Mapping) else None,
        "issue_counts": issue_counts,
        "top_issue": top_issue,
        "affected_issue_count": affected_issue_count,
        "affected_symbol_count": affected_symbol_count,
        "message": message,
    }


def _provider_reconciliation_dimension(provider_reconciliation: Mapping[str, Any]) -> dict[str, Any]:
    reconciliation_status = str(provider_reconciliation.get("status", "unknown"))
    failure_type = provider_reconciliation.get("failure_type")
    if failure_type == "provider_outage" or reconciliation_status in {"critical", "unavailable"}:
        status = "critical"
    elif reconciliation_status in {"warning", "degraded"} or failure_type == "provider_divergence":
        status = "warning"
    elif reconciliation_status == "ok":
        status = "ok"
    else:
        status = "unknown"

    offenders = provider_reconciliation.get("top_offenders")
    top_offenders = [item for item in offenders if isinstance(item, Mapping)] if isinstance(offenders, list) else []
    issue_counts = provider_reconciliation.get("issue_counts")
    return {
        "status": status,
        "failure_type": failure_type,
        "outage_provider": provider_reconciliation.get("outage_provider"),
        "issue_counts": issue_counts if isinstance(issue_counts, Mapping) else {},
        "top_offenders": top_offenders[:5],
        "message": provider_reconciliation.get("message", "provider reconciliation unavailable"),
    }


def _fred_readiness_dimension(fred_readiness: Mapping[str, Any]) -> dict[str, Any]:
    readiness_status = str(fred_readiness.get("status", "unknown"))
    readiness = str(fred_readiness.get("readiness", "unknown"))
    if readiness == "fail":
        status = "critical"
    elif readiness == "warn":
        status = "warning"
    elif readiness == "pass":
        status = "ok"
    elif readiness_status == "critical":
        status = "critical"
    elif readiness_status in {"warning", "degraded"}:
        status = "warning"
    elif readiness_status == "ok":
        status = "ok"
    else:
        status = "unknown"

    message = fred_readiness.get("remediation") or fred_readiness.get("message") or "FRED readiness unavailable"
    return {
        "status": status,
        "readiness": fred_readiness.get("readiness"),
        "mode": fred_readiness.get("mode"),
        "ready": fred_readiness.get("ready"),
        "blocking": fred_readiness.get("blocking"),
        "reason": fred_readiness.get("reason"),
        "source_mode": fred_readiness.get("source_mode"),
        "message": str(message),
    }


def _alpaca_feed_entitlement_dimension(feed_entitlement: Mapping[str, Any]) -> dict[str, Any]:
    """Map feed entitlement into an SLO dimension.

    ``missing_entitlement`` is an intentional lab gap in local/lab/paper modes
    (same posture as missing FRED_API_KEY): warn, do not fail the overall SLO
    as critical. Live mode and delayed/insufficient feeds stay fail-closed.
    """
    policy_decision = str(feed_entitlement.get("policy_decision", "unknown"))
    acceptable_for_live = feed_entitlement.get("acceptable_for_live")
    delayed = bool(feed_entitlement.get("delayed", False))
    reason = feed_entitlement.get("reason")
    safe_reason = _safe_reason(reason)

    # Shared portfolio operating-mode resolver (FRED readiness uses the same keys).
    try:
        from src.monitor.fred_readiness import resolve_fred_operating_mode

        operating_mode = resolve_fred_operating_mode()
    except ImportError:
        operating_mode = "local"

    # Modes where live broker feed entitlement is not a hard gate.
    _LAB_GAP_MODES = {"local", "test", "lab", "paper", "staging", "dev", "development"}
    intentional_lab_gap = False

    if policy_decision in {"accept", "allow"} or acceptable_for_live is True:
        status = "ok"
    elif (
        safe_reason == "missing_entitlement"
        and operating_mode in _LAB_GAP_MODES
        and not delayed
    ):
        # IEX configured without ALPACA_FEED_ENTITLEMENT — expected on research hosts.
        status = "warning"
        intentional_lab_gap = True
    elif policy_decision == "reject" or acceptable_for_live is False or delayed:
        status = "critical"
    elif policy_decision in {"warn", "warning"}:
        status = "warning"
    else:
        status = "unknown"

    if status == "ok":
        message = "Alpaca feed entitlement acceptable for live operation"
    elif intentional_lab_gap:
        message = (
            f"Alpaca feed entitlement not declared for {operating_mode} mode "
            f"({safe_reason}); set ALPACA_FEED_ENTITLEMENT before live operation"
        )
    else:
        message = (
            f"Alpaca feed entitlement {policy_decision}: "
            f"{safe_reason or 'review required'}"
        )

    payload: dict[str, Any] = {
        "status": status,
        "configured_feed": feed_entitlement.get("configured_feed"),
        "effective_feed": feed_entitlement.get("effective_feed"),
        "entitlement": feed_entitlement.get("entitlement"),
        "delayed": delayed,
        "acceptable_for_live": acceptable_for_live,
        "policy_decision": policy_decision,
        "reason": safe_reason,
        "operating_mode": operating_mode,
        "message": message,
    }
    if intentional_lab_gap:
        payload["intentional_lab_gap"] = True
        payload["blocking"] = False
    return payload


def _market_data_consistency_dimension(market_data_consistency: Mapping[str, Any]) -> dict[str, Any]:
    consistency_status = str(market_data_consistency.get("status", "unknown"))
    if consistency_status in {"critical", "error", "failed"}:
        status = "critical"
    elif consistency_status in {"warning", "degraded", "unavailable"}:
        status = "warning"
    elif consistency_status == "ok":
        status = "ok"
    else:
        status = "unknown"

    warnings = market_data_consistency.get("warnings")
    warning_rows = [str(item) for item in warnings] if isinstance(warnings, list) else []
    rows = market_data_consistency.get("rows")
    checked_rows = [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []
    reason = market_data_consistency.get("reason")
    safe_reason = _safe_reason(reason)
    return {
        "status": status,
        "consistency_status": consistency_status,
        "reason": safe_reason,
        "checked_at": market_data_consistency.get("checked_at"),
        "row_count": len(checked_rows),
        "warning_count": len(warning_rows),
        "message": (
            "broker/local market data consistency ok"
            if status == "ok"
            else f"broker/local market data consistency {consistency_status}: {safe_reason or 'review required'}"
        ),
    }


def _overall_status(dimensions: Mapping[str, Mapping[str, Any]]) -> str:
    ranked = sorted(
        (str(row.get("status", "unknown")) for row in dimensions.values()),
        key=lambda status: _STATUS_RANK.get(status, 1),
        reverse=True,
    )
    return ranked[0] if ranked else "unknown"


def _top_dimension(dimensions: Mapping[str, Mapping[str, Any]]) -> str | None:
    failing = [
        (name, str(row.get("status", "unknown")))
        for name, row in dimensions.items()
        if row.get("status") not in {"ok", None}
    ]
    if not failing:
        return None
    return sorted(failing, key=lambda item: _STATUS_RANK.get(item[1], 1), reverse=True)[0][0]


def _safe_reason(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if len(value) > 80 or "://" in value or "=" in value:
        return "redacted"
    return value if all(char in _SAFE_REASON_CHARS for char in value) else "redacted"


def _known_provider_label(value: Any) -> str:
    provider = str(value or "").lower()
    if "yahoo" in provider:
        return "Yahoo Finance"
    if "fred" in provider:
        return "FRED"
    return "market data provider"


def _runbook_entry(
    *,
    dimension: str,
    code: str,
    severity: str,
    action: str,
    artifact: Any = None,
    provider: Any = None,
    reason: Any = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "dimension": dimension,
        "code": code,
        "severity": severity if severity in _STATUS_RANK else "unknown",
        "action": action,
    }
    if artifact:
        entry["artifact"] = str(artifact)
    if provider:
        entry["provider"] = _known_provider_label(provider)
    safe_reason = _safe_reason(reason)
    if safe_reason:
        entry["reason"] = safe_reason
    return entry


def _provider_runbook_entries(provider_dimension: Mapping[str, Any]) -> list[dict[str, Any]]:
    severity = str(provider_dimension.get("status", "unknown"))
    entries: list[dict[str, Any]] = []
    if provider_dimension.get("manifest_failure_reason") == "stale_manifest" or provider_dimension.get("manifest_status") == "stale":
        entries.append(_runbook_entry(
            dimension="provider",
            code="stale_source_manifest",
            severity=severity,
            action="Regenerate public/data/source_manifest.json by rerunning the market-data fetch; verify generated_at and latest_observation before promotion.",
            artifact="source_manifest.json",
            reason=provider_dimension.get("manifest_failure_reason") or provider_dimension.get("manifest_status"),
        ))

    # Intentional lab gaps are excluded from degraded_artifacts / provider status
    # but still need operator runbook hints (e.g. synthetic FRED without key).
    lab_gaps = provider_dimension.get("intentional_lab_gap_artifacts")
    if isinstance(lab_gaps, list):
        for artifact in lab_gaps:
            artifact_name = str(artifact)
            if artifact_name == "yields.json":
                entries.append(_runbook_entry(
                    dimension="provider",
                    code="fred_synthetic_fallback",
                    severity="warning",
                    action=(
                        "Set or verify FRED_API_KEY, rerun the data fetch, and treat "
                        "yield data as synthetic until FRED returns live or cached "
                        "observations."
                    ),
                    artifact=artifact_name,
                    provider="FRED",
                    reason="missing_api_key",
                ))
            elif artifact_name.startswith("prices"):
                entries.append(_runbook_entry(
                    dimension="data_quality",
                    code="price_quality_advisory",
                    severity="warning",
                    action=(
                        "Review data_quality.json advisory issues (internal gaps / "
                        "split-like returns); provider fetch is live — no Yahoo outage."
                    ),
                    artifact=artifact_name,
                    provider="Yahoo Finance",
                    reason="price_quality_warn",
                ))

    degraded_reasons = provider_dimension.get("degraded_reasons")
    if not isinstance(degraded_reasons, Mapping):
        return entries
    for artifact, details in degraded_reasons.items():
        if not isinstance(details, Mapping):
            continue
        source_mode = details.get("source_mode")
        failure_reason = details.get("failure_reason")
        fallback_reason = details.get("fallback_reason")
        reason = failure_reason or fallback_reason or source_mode
        artifact_name = str(artifact)
        if artifact_name == "yields.json" and source_mode == "synthetic":
            entries.append(_runbook_entry(
                dimension="provider",
                code="fred_synthetic_fallback",
                severity=severity,
                action="Set or verify FRED_API_KEY, rerun the data fetch, and treat yield data as synthetic until FRED returns live or cached observations.",
                artifact=artifact_name,
                provider="FRED",
                reason=reason,
            ))
        elif artifact_name.startswith("prices"):
            entries.append(_runbook_entry(
                dimension="provider",
                code="yahoo_provider_failure",
                severity=severity,
                action="Check Yahoo Finance reachability and rate limits, rerun fetch-data, and keep last-good price artifacts until live rows return.",
                artifact=artifact_name,
                provider="Yahoo Finance",
                reason=reason,
            ))
        else:
            entries.append(_runbook_entry(
                dimension="provider",
                code="provider_fallback",
                severity=severity,
                action="Inspect source_manifest.json for the degraded artifact, rerun the provider fetch, and keep last-good data until source_mode returns to live.",
                artifact=artifact_name,
                reason=reason,
            ))
    return entries


def _artifact_runbook_entries(artifact_dimension: Mapping[str, Any]) -> list[dict[str, Any]]:
    severity = str(artifact_dimension.get("status", "unknown"))
    if artifact_dimension.get("source_manifest_index_status") == "stale_index" and not (
        artifact_dimension.get("critical_count") or artifact_dimension.get("critical_artifacts")
    ):
        return [_runbook_entry(
            dimension="artifact",
            code="stale_public_data_index",
            severity=severity,
            action="Regenerate public/data/index.json after source_manifest.json changes, then verify generated_at ordering before publishing.",
            artifact="index.json",
            reason="stale_index",
        )]
    if artifact_dimension.get("source_manifest_index_status") == "unknown_timestamp" and not (
        artifact_dimension.get("critical_count") or artifact_dimension.get("critical_artifacts")
    ):
        return [_runbook_entry(
            dimension="artifact",
            code="public_data_timestamp_unparseable",
            severity=severity,
            action="Regenerate source_manifest.json and public/data/index.json so generated_at timestamps are parseable before publishing.",
            artifact="index.json",
            reason="unknown_timestamp",
        )]
    critical_artifacts = artifact_dimension.get("critical_artifacts")
    critical_names = (
        [str(name) for name in critical_artifacts]
        if isinstance(critical_artifacts, list)
        else []
    )
    critical_count = int(artifact_dimension.get("critical_count") or 0)
    if critical_count > 0 or critical_names:
        sample = ", ".join(critical_names[:5]) if critical_names else "critical symbols"
        if critical_names and len(critical_names) > 5:
            sample = f"{sample} (+{len(critical_names) - 5} more)"
        return [_runbook_entry(
            dimension="artifact",
            code="critical_data_freshness",
            severity="critical" if severity == "critical" else severity,
            action=(
                f"Refresh market data for critical freshness symbols ({sample}); "
                "verify market_lag_days and regenerate public price artifacts before live routing."
            ),
            artifact=critical_names[0] if critical_names else "data_freshness",
            reason="critical_freshness",
        )]
    if artifact_dimension.get("stale_count", 0):
        return [_runbook_entry(
            dimension="artifact",
            code="stale_quote",
            severity=severity,
            action="Run the market-data fetch and verify public price artifacts before order sizing; check scheduler health if prices remain stale.",
            artifact="prices.json",
            reason="stale",
        )]
    if artifact_dimension.get("missing_market_entries"):
        return [_runbook_entry(
            dimension="artifact",
            code="missing_market_artifact",
            severity=severity,
            action="Regenerate public market-data artifacts and public/data/index.json before publishing the dashboard.",
            artifact="public/data/index.json",
            reason="missing",
        )]
    return []


def _provider_reconciliation_runbook_entries(reconciliation_dimension: Mapping[str, Any]) -> list[dict[str, Any]]:
    severity = str(reconciliation_dimension.get("status", "unknown"))
    failure_type = reconciliation_dimension.get("failure_type")
    if failure_type == "provider_outage":
        provider = reconciliation_dimension.get("outage_provider")
        return [_runbook_entry(
            dimension="provider_reconciliation",
            code="provider_outage",
            severity=severity,
            action=f"Check {_known_provider_label(provider)} availability, credentials, and rate limits; keep last-good artifacts until provider rows recover.",
            provider=provider,
            reason=failure_type,
        )]
    if failure_type == "provider_divergence":
        offenders = reconciliation_dimension.get("top_offenders")
        top_offenders = [row for row in offenders if isinstance(row, Mapping)] if isinstance(offenders, list) else []
        offender = top_offenders[0] if top_offenders else {}
        symbol = _safe_reason(offender.get("symbol")) or "top offender"
        issue = _safe_reason(offender.get("issue")) or "price divergence"
        return [_runbook_entry(
            dimension="provider_reconciliation",
            code="provider_divergence",
            severity=severity,
            action=f"Inspect reconciliation top offender {symbol} ({issue}); hold promotion until provider prices converge or the trusted source is explicitly selected.",
            reason=failure_type,
        )]
    return []


def _fred_readiness_runbook_entries(fred_dimension: Mapping[str, Any]) -> list[dict[str, Any]]:
    severity = str(fred_dimension.get("status", "unknown"))
    reason = fred_dimension.get("reason")
    if reason == "missing_fred_api_key":
        code = "fred_missing_api_key"
        action = "Set FRED_API_KEY in the deployment environment, rerun the data fetch, and confirm FRED readiness returns pass before paper or live operation."
    else:
        code = "fred_readiness_failure"
        action = "Verify fredapi availability, FRED_API_KEY validity, and the local FRED cache before relying on macro or yield signals."
    return [_runbook_entry(
        dimension="fred_readiness",
        code=code,
        severity=severity,
        action=action,
        provider="FRED",
        reason=reason,
    )]


def _alpaca_feed_entitlement_runbook_entries(feed_dimension: Mapping[str, Any]) -> list[dict[str, Any]]:
    severity = str(feed_dimension.get("status", "unknown"))
    if severity == "ok":
        return []
    return [_runbook_entry(
        dimension="alpaca_feed_entitlement",
        code="alpaca_feed_entitlement_rejected",
        severity=severity,
        action="Verify Alpaca market-data feed entitlement, confirm delayed feeds are not used for live order sizing, and rerun rebalance health before trading.",
        provider="Alpaca",
        reason=feed_dimension.get("reason") or feed_dimension.get("policy_decision"),
    )]


def _market_data_consistency_runbook_entries(consistency_dimension: Mapping[str, Any]) -> list[dict[str, Any]]:
    severity = str(consistency_dimension.get("status", "unknown"))
    if severity == "ok":
        return []
    code = (
        "market_data_consistency_unavailable"
        if consistency_dimension.get("consistency_status") == "unavailable"
        else "market_data_consistency_degraded"
    )
    return [_runbook_entry(
        dimension="market_data_consistency",
        code=code,
        severity=severity,
        action="Run rebalance health after broker data access is configured; compare broker quotes against local market data before live order sizing.",
        provider="Alpaca",
        reason=consistency_dimension.get("reason") or consistency_dimension.get("consistency_status"),
    )]


def _data_quality_runbook_entries(data_quality_dimension: Mapping[str, Any]) -> list[dict[str, Any]]:
    severity = str(data_quality_dimension.get("status", "unknown"))
    if severity == "ok":
        return []

    top_issue = data_quality_dimension.get("top_issue")
    quality_status = data_quality_dimension.get("quality_status")
    artifact = data_quality_dimension.get("artifact") or "data_quality.json"
    issue_count = data_quality_dimension.get("affected_issue_count")
    symbol_count = data_quality_dimension.get("affected_symbol_count")

    if quality_status in {"missing", "unavailable"}:
        code = "price_quality_report_missing"
        action = "Regenerate data_quality.json from the public price artifacts and keep the SLO warning until the quality report is available in source_manifest.json."
        reason = quality_status
    elif top_issue == "duplicate_dates":
        code = "price_quality_duplicate_dates"
        action = f"Reject duplicate price dates, rerun the market-data fetch, and verify data_quality.json before promotion ({issue_count} duplicate date issue(s), {symbol_count} tracked symbol(s))."
        reason = top_issue
    elif top_issue == "stale_latest_dates":
        code = "price_quality_stale_cross_section"
        action = f"Refresh the stale cross-section, verify every tracked symbol has the latest market date, and regenerate data_quality.json ({issue_count} stale symbol issue(s))."
        reason = top_issue
    elif top_issue == "internal_gaps":
        code = "price_quality_internal_gaps"
        action = f"Inspect missing trading dates against the reference calendar, refill or document source gaps, and rerun the quality audit ({issue_count} gap issue(s))."
        reason = top_issue
    elif top_issue in {"extreme_returns", "split_like_returns"}:
        code = "price_quality_anomalous_returns"
        action = f"Verify split-like or extreme return observations against corporate actions before publishing price artifacts ({issue_count} anomaly issue(s))."
        reason = top_issue
    elif top_issue in {
        "empty_symbols",
        "invalid_dates",
        "invalid_prices",
        "missing_required_keys",
        "non_monotonic_rows",
        "non_object_records",
    }:
        code = "price_quality_invalid_rows"
        action = f"Block promotion, repair malformed price rows, and rerun data_quality.json before the dashboard consumes the artifacts ({issue_count} row issue(s))."
        reason = top_issue
    else:
        code = "price_quality_degraded"
        action = "Inspect data_quality.json issue_counts, repair the public price artifact, and rerun the market-data fetch before promotion."
        reason = quality_status or top_issue

    return [_runbook_entry(
        dimension="data_quality",
        code=code,
        severity=severity,
        action=action,
        artifact=artifact,
        provider="Yahoo Finance",
        reason=reason,
    )]


def _build_runbook(dimensions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    provider = dimensions.get("provider")
    if isinstance(provider, Mapping):
        actions.extend(_provider_runbook_entries(provider))
    artifact = dimensions.get("artifact")
    if isinstance(artifact, Mapping):
        actions.extend(_artifact_runbook_entries(artifact))
    data_quality = dimensions.get("data_quality")
    if isinstance(data_quality, Mapping):
        actions.extend(_data_quality_runbook_entries(data_quality))
    reconciliation = dimensions.get("provider_reconciliation")
    if isinstance(reconciliation, Mapping):
        actions.extend(_provider_reconciliation_runbook_entries(reconciliation))
    fred_readiness = dimensions.get("fred_readiness")
    if isinstance(fred_readiness, Mapping):
        actions.extend(_fred_readiness_runbook_entries(fred_readiness))
    feed_entitlement = dimensions.get("alpaca_feed_entitlement")
    if isinstance(feed_entitlement, Mapping):
        actions.extend(_alpaca_feed_entitlement_runbook_entries(feed_entitlement))
    market_consistency = dimensions.get("market_data_consistency")
    if isinstance(market_consistency, Mapping):
        actions.extend(_market_data_consistency_runbook_entries(market_consistency))

    active_actions = [action for action in actions if action.get("severity") not in {"ok", None}]
    top_cause = None
    if active_actions:
        top_cause = sorted(
            active_actions,
            key=lambda action: _STATUS_RANK.get(str(action.get("severity", "unknown")), 1),
            reverse=True,
        )[0]
    return {
        "status": str(top_cause.get("severity")) if isinstance(top_cause, Mapping) else "ok",
        "top_cause": top_cause,
        "actions": active_actions[:6],
    }


def build_data_pipeline_slo(
    *,
    health_data: Mapping[str, Any],
    source_manifest: Mapping[str, Any] | None = None,
    data_quality_report: Mapping[str, Any] | None = None,
    public_index: Mapping[str, Any] | None = None,
    signal_staleness: Mapping[str, Any] | None = None,
    provider_reconciliation: Mapping[str, Any] | None = None,
    fred_readiness: Mapping[str, Any] | None = None,
    alpaca_feed_entitlement: Mapping[str, Any] | None = None,
    market_data_consistency: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact SLO summary from already-generated dashboard artifacts."""
    dimensions = {
        "scheduler": _scheduler_dimension(health_data),
        "provider": _provider_dimension(source_manifest),
        "artifact": _artifact_dimension(health_data, public_index, source_manifest),
        "signal": _signal_dimension(signal_staleness),
        "data_quality": _data_quality_dimension(source_manifest, data_quality_report),
    }
    reconciliation = provider_reconciliation
    if reconciliation is None:
        health_reconciliation = health_data.get("provider_reconciliation")
        reconciliation = health_reconciliation if isinstance(health_reconciliation, Mapping) else None
    if isinstance(reconciliation, Mapping):
        dimensions["provider_reconciliation"] = _provider_reconciliation_dimension(reconciliation)
    readiness = fred_readiness
    if readiness is None:
        health_readiness = health_data.get("fred_readiness")
        if not isinstance(health_readiness, Mapping):
            data_freshness = health_data.get("data_freshness")
            if isinstance(data_freshness, Mapping):
                health_readiness = data_freshness.get("fred_readiness")
        readiness = health_readiness if isinstance(health_readiness, Mapping) else None
    if isinstance(readiness, Mapping):
        dimensions["fred_readiness"] = _fred_readiness_dimension(readiness)
    feed_entitlement = alpaca_feed_entitlement
    if feed_entitlement is None:
        health_feed_entitlement = health_data.get("alpaca_feed_entitlement")
        feed_entitlement = health_feed_entitlement if isinstance(health_feed_entitlement, Mapping) else None
    if isinstance(feed_entitlement, Mapping):
        dimensions["alpaca_feed_entitlement"] = _alpaca_feed_entitlement_dimension(feed_entitlement)
    consistency = market_data_consistency
    if consistency is None:
        health_consistency = health_data.get("market_data_consistency")
        consistency = health_consistency if isinstance(health_consistency, Mapping) else None
    if isinstance(consistency, Mapping):
        dimensions["market_data_consistency"] = _market_data_consistency_dimension(consistency)
    return {
        "schema_version": DATA_PIPELINE_SLO_SCHEMA_VERSION,
        "status": _overall_status(dimensions),
        "top_dimension": _top_dimension(dimensions),
        "dimensions": dimensions,
        "runbook": _build_runbook(dimensions),
    }
