"""Kill-switch / incident authority projection for multi-surface honesty.

SSOT for operator-facing kill identity is ``data/kill_switch.json`` plus
``data/incidents.json``. Dashboard projectors copy the same incident_id,
level, reason, mode, and human message into public health, alerts, signals
compact health, and allocation surface roles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "load_kill_switch_payload",
    "load_open_incidents_summary",
    "project_kill_switch_fields",
    "project_compact_kill_fields",
    "elevate_system_status_for_kill",
    "build_kill_switch_alert",
    "allocation_roles_under_kill",
    "kill_identity_tuple",
]


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_kill_switch_payload(data_dir: str | Path) -> dict[str, Any] | None:
    """Load authority kill_switch.json; return None when absent/invalid."""
    return _load_json(Path(data_dir) / "kill_switch.json")


def load_open_incidents_summary(data_dir: str | Path) -> dict[str, Any]:
    """Bounded open-incident summary for public health projection."""
    payload = _load_json(Path(data_dir) / "incidents.json")
    if not payload:
        return {"status": "ok", "open_count": 0, "incidents": []}

    raw = payload.get("incidents", payload.get("open_incidents", []))
    rows = raw if isinstance(raw, list) else []
    open_incidents: list[dict[str, Any]] = []
    has_halt = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        state = str(row.get("state") or row.get("status") or "open").lower()
        if state in {"closed", "resolved", "pass"}:
            continue
        kill_level = str(row.get("kill_switch_level") or "").lower() or None
        if kill_level == "halt":
            has_halt = True
        open_incidents.append(
            {
                "incident_id": row.get("incident_id") or row.get("id"),
                "channel": row.get("channel"),
                "severity": row.get("severity"),
                "state": state,
                "message": row.get("message"),
                "kill_switch_level": kill_level,
            }
        )

    open_count = int(payload.get("open_count") or len(open_incidents) or 0)
    if open_count == 0 and open_incidents:
        open_count = len(open_incidents)

    if has_halt:
        status = "critical"
    elif open_count > 0 or open_incidents:
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "open_count": open_count,
        "incidents": open_incidents[:10],
    }


def project_kill_switch_fields(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize kill_switch authority into a public/monitor-shaped block."""
    if not payload:
        return {
            "status": "ok",
            "enabled": False,
            "level": None,
            "reason": None,
            "source": None,
            "message": None,
            "timestamp": None,
            "incident_id": None,
            "mode": None,
            "channel": None,
        }

    enabled = bool(payload.get("enabled"))
    level = str(payload.get("level") or "").lower() or None
    if enabled and level == "halt":
        status = "critical"
    elif enabled:
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "enabled": enabled,
        "level": level,
        "reason": payload.get("reason"),
        "source": payload.get("source"),
        "message": payload.get("message"),
        "timestamp": payload.get("timestamp"),
        "incident_id": payload.get("incident_id"),
        "mode": payload.get("mode"),
        "channel": payload.get("channel"),
    }


def project_compact_kill_fields(report: dict[str, Any]) -> dict[str, Any]:
    """Extract kill/incident keys for signals.health compact summary.

    Accepts either monitor health (checks.kill_switch) or public health
    (top-level kill_switch).
    """
    out: dict[str, Any] = {}
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    kill = report.get("kill_switch")
    if not isinstance(kill, dict):
        kill = checks.get("kill_switch") if isinstance(checks.get("kill_switch"), dict) else None
    if isinstance(kill, dict):
        if kill.get("enabled") is not None:
            out["kill_switch_enabled"] = bool(kill.get("enabled"))
        if kill.get("level") is not None:
            out["kill_switch_level"] = kill.get("level")
        if kill.get("reason") is not None:
            out["kill_switch_reason"] = kill.get("reason")
        if kill.get("status") is not None:
            out["kill_switch_status"] = kill.get("status")
        if kill.get("incident_id") is not None:
            out["kill_switch_incident_id"] = kill.get("incident_id")
        if kill.get("message") is not None:
            out["kill_switch_message"] = kill.get("message")
        if kill.get("mode") is not None:
            out["kill_switch_mode"] = kill.get("mode")

    open_inc = report.get("open_incidents")
    if not isinstance(open_inc, dict):
        open_inc = (
            checks.get("open_incidents")
            if isinstance(checks.get("open_incidents"), dict)
            else None
        )
    if isinstance(open_inc, dict):
        if open_inc.get("status") is not None:
            out["open_incidents_status"] = open_inc.get("status")
        if open_inc.get("open_count") is not None:
            out["open_incidents_count"] = open_inc.get("open_count")

    return out


def elevate_system_status_for_kill(
    current: str,
    kill: dict[str, Any] | None,
    open_incidents: dict[str, Any] | None = None,
) -> str:
    """Raise system_status when kill HALT/restrict or open-incident HALT is active."""
    status = current or "healthy"
    rank = {"healthy": 0, "ok": 0, "warning": 1, "degraded": 2, "critical": 3, "error": 3}
    target = rank.get(status, 0)

    if isinstance(kill, dict) and kill.get("enabled"):
        level = str(kill.get("level") or "").lower()
        kill_status = str(kill.get("status") or "").lower()
        if level == "halt" or kill_status == "critical":
            target = max(target, 3)
        else:
            target = max(target, 1)

    if isinstance(open_incidents, dict):
        oi_status = str(open_incidents.get("status") or "").lower()
        if oi_status == "critical":
            target = max(target, 3)
        elif oi_status == "warning":
            target = max(target, 1)

    for name, value in (("critical", 3), ("degraded", 2), ("warning", 1), ("healthy", 0)):
        if target >= value and name in rank:
            if target == 3:
                return "critical"
            if target == 2:
                return "degraded"
            if target == 1:
                return "warning"
            return "healthy" if status in {"healthy", "ok", ""} else status
    return status


def build_kill_switch_alert(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Build alerts.json kill row preferring human message over opaque reason."""
    if not payload or not payload.get("enabled"):
        return None
    mode = str(payload.get("mode") or "unknown")
    reason = payload.get("reason")
    message = payload.get("message")
    human = message if isinstance(message, str) and message.strip() else None
    display = human or (str(reason) if reason is not None else "Kill switch enabled")
    alert: dict[str, Any] = {
        "level": "error",
        "type": "kill_switch",
        "title": f"{mode.upper()} Kill Switch Triggered",
        "message": display,
        "timestamp": payload.get("timestamp"),
        "requires_action": True,
    }
    if reason is not None:
        alert["reason"] = reason
    if payload.get("incident_id") is not None:
        alert["incident_id"] = payload.get("incident_id")
    if payload.get("channel") is not None:
        alert["channel"] = payload.get("channel")
    if payload.get("level") is not None:
        alert["kill_switch_level"] = payload.get("level")
    if payload.get("source") is not None:
        alert["source"] = payload.get("source")
    return alert


def allocation_roles_under_kill(
    roles: dict[str, Any],
    *,
    kill_enabled: bool,
    kill_level: str | None = None,
) -> dict[str, Any]:
    """Mark execution-routed surface as blocked when kill switch is active."""
    if not kill_enabled or not isinstance(roles, dict):
        return roles
    surfaces = roles.get("surfaces")
    if not isinstance(surfaces, dict):
        return roles
    target = surfaces.get("target_allocations")
    if not isinstance(target, dict):
        return roles
    level = (kill_level or "").lower() or None
    target = dict(target)
    target["live_authoritative"] = False
    target["execution_blocked"] = True
    target["kill_switch_enabled"] = True
    if level:
        target["kill_switch_level"] = level
    target["role"] = "execution_blocked"
    # Keep routed=True (still the routing surface) but disclose block.
    desc = target.get("description") or ""
    halt_note = "Order routing blocked by active kill switch"
    if level:
        halt_note += f" (level={level})"
    target["description"] = f"{halt_note}. {desc}".strip()
    surfaces = dict(surfaces)
    surfaces["target_allocations"] = target
    out = dict(roles)
    out["surfaces"] = surfaces
    out["execution_blocked"] = True
    out["kill_switch_enabled"] = True
    if level:
        out["kill_switch_level"] = level
    return out


def kill_identity_tuple(payload: dict[str, Any] | None) -> tuple[Any, Any, Any, Any]:
    """Comparable identity for multi-surface consistency gates."""
    if not payload:
        return (None, None, None, None)
    return (
        payload.get("incident_id"),
        str(payload.get("level") or "").lower() or None,
        payload.get("reason"),
        payload.get("mode"),
    )
