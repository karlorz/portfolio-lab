"""Pure projection of health.json / data_pipeline_slo into operator alerts."""

from __future__ import annotations

from typing import Any, Mapping

HEALTH_SLO_ALERT_TYPE = "health_slo"

__all__ = [
    "HEALTH_SLO_ALERT_TYPE",
    "build_health_slo_alerts",
    "critical_health_requires_alert",
    "warning_health_requires_alert",
]


def _slo_status(health: Mapping[str, Any]) -> str:
    slo = health.get("data_pipeline_slo")
    slo = slo if isinstance(slo, Mapping) else {}
    return str(slo.get("status") or "").lower()


def critical_health_requires_alert(health: Mapping[str, Any] | None) -> bool:
    """True when system_status or data_pipeline_slo.status is critical."""
    if not isinstance(health, Mapping):
        return False
    system_status = str(health.get("system_status") or "").lower()
    return system_status == "critical" or _slo_status(health) == "critical"


def warning_health_requires_alert(health: Mapping[str, Any] | None) -> bool:
    """True when ops is warning/degraded but not critical (critical wins)."""
    if not isinstance(health, Mapping):
        return False
    if critical_health_requires_alert(health):
        return False
    system_status = str(health.get("system_status") or "").lower()
    slo_status = _slo_status(health)
    return system_status in {"warning", "degraded"} or slo_status in {
        "warning",
        "degraded",
    }


def build_health_slo_alerts(health: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Build operator alerts for critical or warning health / SLO state.

    Critical → level error. Warning/degraded (without critical) → level warning.
    Empty list when healthy/ok. Structured fields carry triage context.
    """
    if not isinstance(health, Mapping):
        return []

    is_critical = critical_health_requires_alert(health)
    is_warning = warning_health_requires_alert(health)
    if not is_critical and not is_warning:
        return []

    system_status = str(health.get("system_status") or "").lower() or None
    slo = health.get("data_pipeline_slo")
    slo = slo if isinstance(slo, Mapping) else {}
    slo_status = str(slo.get("status") or "").lower() or None

    top_dimension = slo.get("top_dimension")
    top_dim_str = str(top_dimension) if top_dimension else None
    dim_label = top_dim_str or "ops"

    dimensions = slo.get("dimensions") if isinstance(slo.get("dimensions"), Mapping) else {}
    dim_payload = dimensions.get(top_dim_str) if top_dim_str and isinstance(dimensions, Mapping) else None
    dim_payload = dim_payload if isinstance(dim_payload, Mapping) else {}

    runbook = slo.get("runbook") if isinstance(slo.get("runbook"), Mapping) else {}
    top_cause = runbook.get("top_cause") if isinstance(runbook.get("top_cause"), Mapping) else {}

    # Surface scheduler / signal_health summaries for warning triage
    scheduler_status = health.get("scheduler_status")
    if isinstance(scheduler_status, Mapping):
        scheduler_label = str(
            scheduler_status.get("status") or scheduler_status.get("state") or ""
        ).lower() or None
    else:
        scheduler_label = str(scheduler_status or "").lower() or None

    signal_health = health.get("signal_health")
    signal_summary = None
    if isinstance(signal_health, Mapping):
        signal_summary = (
            signal_health.get("status")
            or signal_health.get("summary")
            or signal_health.get("overall")
        )
        if signal_summary is not None:
            signal_summary = str(signal_summary)

    if is_critical:
        reason = (
            dim_payload.get("reason")
            or top_cause.get("reason")
            or top_cause.get("code")
            or dim_payload.get("message")
            or "critical_health_slo"
        )
        policy_decision = dim_payload.get("policy_decision") or top_cause.get("policy_decision")
        action = top_cause.get("action") or dim_payload.get("action")
        message = f"Critical health/SLO: {dim_label} ({reason})"
        if policy_decision is not None:
            message = f"{message}; policy={policy_decision}"
        alert: dict[str, Any] = {
            "level": "error",
            "type": HEALTH_SLO_ALERT_TYPE,
            "title": f"Critical Health/SLO: {dim_label}",
            "message": message,
            "timestamp": health.get("generated_at"),
            "requires_action": True,
            "top_dimension": top_dim_str,
            "reason": reason,
            "system_status": system_status,
            "data_pipeline_slo_status": slo_status,
        }
        if policy_decision is not None:
            alert["policy_decision"] = policy_decision
        if action:
            alert["runbook_action"] = action
        return [alert]

    # Warning / degraded path
    reason = (
        dim_payload.get("reason")
        or top_cause.get("reason")
        or top_cause.get("code")
        or dim_payload.get("message")
        or "system_status_warning"
    )
    parts = [f"system_status={system_status or 'n/a'}"]
    if scheduler_label:
        parts.append(f"scheduler={scheduler_label}")
    if signal_summary:
        parts.append(f"signal_health={signal_summary}")
    if slo_status and slo_status not in {"ok", "healthy"}:
        parts.append(f"slo={slo_status}")
    message = f"Health warning: {'; '.join(parts)} ({reason})"

    alert = {
        "level": "warning",
        "type": HEALTH_SLO_ALERT_TYPE,
        "title": f"Health Warning: {dim_label}",
        "message": message,
        "timestamp": health.get("generated_at"),
        "requires_action": False,
        "top_dimension": top_dim_str,
        "reason": reason,
        "system_status": system_status,
        "data_pipeline_slo_status": slo_status,
    }
    if scheduler_label:
        alert["scheduler_status"] = scheduler_label
    if signal_summary:
        alert["signal_health_status"] = signal_summary
    return [alert]
