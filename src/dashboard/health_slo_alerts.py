"""Pure projection of critical health.json / data_pipeline_slo into operator alerts."""

from __future__ import annotations

from typing import Any, Mapping

HEALTH_SLO_ALERT_TYPE = "health_slo"

__all__ = [
    "HEALTH_SLO_ALERT_TYPE",
    "build_health_slo_alerts",
    "critical_health_requires_alert",
]


def critical_health_requires_alert(health: Mapping[str, Any] | None) -> bool:
    """True when system_status or data_pipeline_slo.status is critical."""
    if not isinstance(health, Mapping):
        return False
    system_status = str(health.get("system_status") or "").lower()
    slo = health.get("data_pipeline_slo")
    slo = slo if isinstance(slo, Mapping) else {}
    slo_status = str(slo.get("status") or "").lower()
    return system_status == "critical" or slo_status == "critical"


def build_health_slo_alerts(health: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Build operator alerts for critical health / data_pipeline_slo state.

    Returns a single ``health_slo`` alert when either ``system_status`` or
    ``data_pipeline_slo.status`` is critical. Empty list otherwise.
    Structured fields carry triage context; ``message`` stays short.
    """
    if not critical_health_requires_alert(health):
        return []
    assert isinstance(health, Mapping)

    system_status = str(health.get("system_status") or "").lower() or None
    slo = health.get("data_pipeline_slo")
    slo = slo if isinstance(slo, Mapping) else {}
    slo_status = str(slo.get("status") or "").lower() or None

    top_dimension = slo.get("top_dimension")
    top_dim_str = str(top_dimension) if top_dimension else None
    dim_label = top_dim_str or "unknown"

    dimensions = slo.get("dimensions") if isinstance(slo.get("dimensions"), Mapping) else {}
    dim_payload = dimensions.get(top_dim_str) if top_dim_str and isinstance(dimensions, Mapping) else None
    dim_payload = dim_payload if isinstance(dim_payload, Mapping) else {}

    runbook = slo.get("runbook") if isinstance(slo.get("runbook"), Mapping) else {}
    top_cause = runbook.get("top_cause") if isinstance(runbook.get("top_cause"), Mapping) else {}

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
