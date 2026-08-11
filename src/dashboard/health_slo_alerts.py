"""Pure projection of health.json / data_pipeline_slo into operator alerts."""

from __future__ import annotations

from typing import Any, Mapping

from src.dashboard.health_report import signal_health_status_contribution

HEALTH_SLO_ALERT_TYPE = "health_slo"
SIGNAL_QUALITY_ALERT_TYPE = "signal_quality"

__all__ = [
    "HEALTH_SLO_ALERT_TYPE",
    "SIGNAL_QUALITY_ALERT_TYPE",
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


def _signal_health_quality_context(
    health: Mapping[str, Any],
) -> tuple[str | None, str | None, bool]:
    """Return (status_label, quality_badge, zero_healthy) from signal_health block."""
    signal_health = health.get("signal_health")
    if not isinstance(signal_health, Mapping):
        return None, None, False
    summary = signal_health.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    status = (
        signal_health.get("status")
        or signal_health.get("overall_health")
        or summary.get("status")
        or signal_health.get("overall")
    )
    status_label = str(status).lower() if status is not None else None
    badge = summary.get("quality_badge") or signal_health.get("quality_badge")
    badge_s = str(badge) if badge is not None else None
    zero = bool(
        summary.get("zero_healthy_sources")
        or signal_health.get("zero_healthy_sources")
    )
    if not zero and isinstance(summary.get("healthy"), (int, float)):
        total = summary.get("total_tracked")
        try:
            if int(summary.get("healthy") or 0) == 0 and int(total or 0) > 0:
                zero = True
        except (TypeError, ValueError):
            pass
    return status_label, badge_s, zero


def _ops_dimensions_ok(health: Mapping[str, Any], *, slo_status: str | None) -> bool:
    """True when kill/open/scheduler/data_pipeline look operationally fine.

    The live monitor restamps ``ops_health_status`` on every health publish,
    so a present non-green value is operational evidence. Mirror lag is also
    an ops SLI and must prevent quality-only ownership.
    """
    if slo_status and slo_status not in {"ok", "healthy", ""}:
        return False
    ops_status = str(health.get("ops_health_status") or "").lower()
    if ops_status and ops_status not in {"ok", "healthy", "green", "success"}:
        return False
    kill = health.get("kill_switch")
    if isinstance(kill, Mapping) and kill.get("enabled"):
        return False
    open_inc = health.get("open_incidents")
    if isinstance(open_inc, Mapping) and int(open_inc.get("open_count") or 0) > 0:
        return False
    scheduler = health.get("scheduler_status")
    if isinstance(scheduler, Mapping):
        sched = str(scheduler.get("status") or scheduler.get("state") or "").lower()
    else:
        sched = str(scheduler or "").lower()
    if sched and sched not in {"ok", "healthy", "success"}:
        return False
    lag_status = str(health.get("repo_public_mirror_lag_status") or "").lower()
    if lag_status and lag_status not in {"ok", "healthy", "green"}:
        return False
    try:
        if int(health.get("repo_public_mirror_lagging_count") or 0) > 0:
            return False
    except (TypeError, ValueError):
        return False
    return True


def build_health_slo_alerts(health: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Build operator alerts for critical or warning health / SLO state.

    Critical → level error. Warning/degraded (without critical) → level warning.
    Empty list when healthy/ok. Structured fields carry triage context.

    Batch JL JH1a: when demotion is signal_health-only (ops dimensions ok),
    emit ``signal_quality`` type and never title **"Health Warning: ops"**.
    """
    if not isinstance(health, Mapping):
        return []

    signal_status, quality_badge, zero_healthy = _signal_health_quality_context(health)
    signal_health = health.get("signal_health")
    quality_warning = bool(
        signal_health_status_contribution(
            signal_health if isinstance(signal_health, Mapping) else None
        )
    )
    is_critical = critical_health_requires_alert(health)
    is_warning = warning_health_requires_alert(health)
    if not is_critical and not is_warning and not quality_warning:
        return []

    system_status = str(health.get("system_status") or "").lower() or None
    slo = health.get("data_pipeline_slo")
    slo = slo if isinstance(slo, Mapping) else {}
    slo_status = str(slo.get("status") or "").lower() or None

    top_dimension = slo.get("top_dimension")
    top_dim_str = str(top_dimension) if top_dimension else None

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

    signal_summary = signal_status
    if quality_badge and (zero_healthy or signal_status in {"degraded", "warning", "unhealthy"}):
        signal_summary = quality_badge

    ops_ok = _ops_dimensions_ok(health, slo_status=slo_status)
    sh_only = bool(
        ops_ok
        and quality_warning
        and (not top_dim_str or top_dim_str in {"signal_health", "signal_quality"})
    )

    # JH1a: never default dim_label to bare "ops" for SH-only demotion
    if sh_only and not is_critical:
        dim_label = quality_badge or "signal_quality"
        alert_type = SIGNAL_QUALITY_ALERT_TYPE
    else:
        dim_label = top_dim_str or ("signal_quality" if sh_only else "ops")
        alert_type = HEALTH_SLO_ALERT_TYPE

    if is_critical:
        # N1 (2026-08-11 session B): dedupe G1-derived criticals. When the
        # kill-switch halt / open-incident complex is the ONLY critical driver
        # (SLO ok, scheduler ok) and the dedicated kill_switch critical alert
        # (incident_id + halt level) already pages it, do not emit a second
        # requires_action health_slo alert — the by-design halt pages once,
        # not twice. Observed: recurring critical_health_slo error every
        # :00/:30 since 09:15:13Z duplicating kill_switch for incident
        # 8115a9c1 (halt by design, operator decision pending).
        kill = health.get("kill_switch")
        kill_on = isinstance(kill, Mapping) and bool(kill.get("enabled"))
        kill_level = (
            str(kill.get("level") or "").lower()
            if isinstance(kill, Mapping)
            else ""
        )
        # Suppress only for the by-design HALT complex (kill_switch alert
        # covers it). Restrict/warning kills and open-incident-only criticals
        # keep their health_slo page — no dedicated alert exists for them.
        g1_only_critical = (
            kill_on
            and kill_level == "halt"
            and slo_status not in {"critical", "error"}
            and scheduler_label in {None, "", "ok", "healthy", "success"}
        )
        if g1_only_critical:
            return []

        reason = (
            dim_payload.get("reason")
            or top_cause.get("reason")
            or top_cause.get("code")
            or dim_payload.get("message")
            or "critical_health_slo"
        )
        policy_decision = dim_payload.get("policy_decision") or top_cause.get("policy_decision")
        action = top_cause.get("action") or dim_payload.get("action")
        crit_label = top_dim_str or dim_label or "ops"
        message = f"Critical health/SLO: {crit_label} ({reason})"
        if policy_decision is not None:
            message = f"{message}; policy={policy_decision}"
        alert: dict[str, Any] = {
            "level": "error",
            "type": HEALTH_SLO_ALERT_TYPE,
            "title": f"Critical Health/SLO: {crit_label}",
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
        or ("signal_quality_degraded" if sh_only else "system_status_warning")
    )
    parts = [f"system_status={system_status or 'n/a'}"]
    if scheduler_label:
        parts.append(f"scheduler={scheduler_label}")
    if signal_summary:
        parts.append(f"signal_health={signal_summary}")
    if slo_status and slo_status not in {"ok", "healthy"}:
        parts.append(f"slo={slo_status}")

    if sh_only:
        badge_bit = quality_badge or signal_summary or "signal quality degraded"
        title = f"Signal quality degraded ({badge_bit})"
        message = f"Signal quality warning: {'; '.join(parts)} ({reason})"
    else:
        title = f"Health Warning: {dim_label}"
        message = f"Health warning: {'; '.join(parts)} ({reason})"

    alert = {
        "level": "warning",
        "type": alert_type,
        "title": title,
        "message": message,
        "timestamp": health.get("generated_at"),
        "requires_action": False,
        "top_dimension": top_dim_str if not sh_only else "signal_health",
        "reason": reason,
        "system_status": system_status,
        "data_pipeline_slo_status": slo_status,
    }
    if scheduler_label:
        alert["scheduler_status"] = scheduler_label
    if signal_status or signal_summary:
        alert["signal_health_status"] = signal_status or str(signal_summary)
    if quality_badge:
        alert["signal_quality_badge"] = quality_badge
    if zero_healthy:
        alert["zero_healthy_sources"] = True
    return [alert]
