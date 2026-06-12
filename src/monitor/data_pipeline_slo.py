"""Data pipeline SLO summary for dashboard health output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


DATA_PIPELINE_SLO_SCHEMA_VERSION = "data-pipeline-slo/v1"
_STATUS_RANK = {"ok": 0, "unknown": 1, "warning": 2, "critical": 3}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_source_manifest(public_dir: Path) -> dict[str, Any]:
    return _load_json(public_dir / "source_manifest.json")


def load_public_index(public_dir: Path) -> dict[str, Any]:
    return _load_json(public_dir / "index.json")


def load_signal_staleness(public_dir: Path) -> dict[str, Any]:
    signals = _load_json(public_dir / "signals.json")
    staleness = signals.get("staleness")
    return staleness if isinstance(staleness, dict) else {}


def _scheduler_dimension(health_data: Mapping[str, Any]) -> dict[str, Any]:
    scheduler = health_data.get("scheduler_status")
    scheduler_status = scheduler.get("status") if isinstance(scheduler, Mapping) else "unknown"
    cron_jobs = health_data.get("cron_jobs")
    jobs = cron_jobs if isinstance(cron_jobs, list) else []
    failed_jobs = [job for job in jobs if isinstance(job, Mapping) and job.get("status") == "error"]
    if len(failed_jobs) > 2:
        status = "critical"
    elif failed_jobs or scheduler_status in {"degraded", "error", "warning", "unavailable"}:
        status = "warning"
    elif scheduler_status == "unknown":
        status = "unknown"
    else:
        status = "ok"
    return {
        "status": status,
        "scheduler_status": scheduler_status,
        "failed_jobs": len(failed_jobs),
        "message": (
            f"{len(failed_jobs)} scheduler job(s) failed"
            if failed_jobs
            else f"scheduler {scheduler_status}"
        ),
    }


def _provider_dimension(source_manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    artifacts = source_manifest.get("artifacts") if isinstance(source_manifest, Mapping) else None
    rows = [row for row in artifacts if isinstance(row, Mapping)] if isinstance(artifacts, list) else []
    if not rows:
        return {
            "status": "unknown",
            "degraded_artifacts": [],
            "message": "source manifest missing or empty",
        }
    degraded_rows = [
        row for row in rows
        if row.get("status") != "success" or row.get("source_mode") != "live"
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
    status = "warning" if degraded else "ok"
    return {
        "status": status,
        "degraded_artifacts": degraded,
        "degraded_reasons": degraded_reasons,
        "message": (
            f"provider degraded for {', '.join(degraded)} ({'; '.join(reason_parts)})"
            if degraded
            else "providers live"
        ),
    }


def _artifact_dimension(
    health_data: Mapping[str, Any],
    public_index: Mapping[str, Any] | None,
) -> dict[str, Any]:
    data_freshness = health_data.get("data_freshness")
    freshness = data_freshness if isinstance(data_freshness, Mapping) else {}
    critical = [name for name, row in freshness.items() if isinstance(row, Mapping) and row.get("status") == "critical"]
    stale = [name for name, row in freshness.items() if isinstance(row, Mapping) and row.get("status") == "stale"]

    entries = public_index.get("entries") if isinstance(public_index, Mapping) else None
    missing_market_entries = [
        entry.get("filename")
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("category") == "market_data"
        and entry.get("status") == "missing"
    ] if isinstance(entries, list) else []

    if len(critical) > 10 or missing_market_entries:
        status = "critical"
    elif stale:
        status = "warning"
    elif critical:
        status = "warning"
    else:
        status = "ok"
    return {
        "status": status,
        "critical_count": len(critical),
        "stale_count": len(stale),
        "missing_market_entries": missing_market_entries,
        "message": (
            f"{len(critical)} critical, {len(stale)} stale artifacts"
            if critical or stale
            else "artifacts fresh"
        ),
    }


def _signal_dimension(signal_staleness: Mapping[str, Any] | None) -> dict[str, Any]:
    stale = signal_staleness.get("stale_signals") if isinstance(signal_staleness, Mapping) else None
    unavailable = signal_staleness.get("unavailable_signals") if isinstance(signal_staleness, Mapping) else None
    stale_signals = [str(item) for item in stale] if isinstance(stale, list) else []
    unavailable_signals = [str(item) for item in unavailable] if isinstance(unavailable, list) else []
    status = "warning" if stale_signals else "ok"
    return {
        "status": status,
        "stale_count": len(stale_signals),
        "unavailable_count": len(unavailable_signals),
        "stale_signals": stale_signals[:10],
        "message": (
            f"{len(stale_signals)} stale required signal(s)"
            if stale_signals
            else "required signals fresh"
        ),
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


def build_data_pipeline_slo(
    *,
    health_data: Mapping[str, Any],
    source_manifest: Mapping[str, Any] | None = None,
    public_index: Mapping[str, Any] | None = None,
    signal_staleness: Mapping[str, Any] | None = None,
    provider_reconciliation: Mapping[str, Any] | None = None,
    fred_readiness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact SLO summary from already-generated dashboard artifacts."""
    dimensions = {
        "scheduler": _scheduler_dimension(health_data),
        "provider": _provider_dimension(source_manifest),
        "artifact": _artifact_dimension(health_data, public_index),
        "signal": _signal_dimension(signal_staleness),
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
    return {
        "schema_version": DATA_PIPELINE_SLO_SCHEMA_VERSION,
        "status": _overall_status(dimensions),
        "top_dimension": _top_dimension(dimensions),
        "dimensions": dimensions,
    }
