"""Persistent incident lifecycle for production alert transitions."""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.paths import DATA_DIR, PUBLIC_DATA_DIR

logger = logging.getLogger(__name__)

DEFAULT_INCIDENT_LOG_PATH = DATA_DIR / "incidents.jsonl"
DEFAULT_INCIDENT_SUMMARY_PATH = DATA_DIR / "incidents.json"
DEFAULT_KILL_SWITCH_PATH = DATA_DIR / "kill_switch.json"

_KILL_SWITCH_LEVEL_RANK = {
    "none": 0,
    "warning": 1,
    "restrict": 2,
    "halt": 3,
    "liquidate": 4,
}
_KILL_SWITCH_REDUCTION = {
    "warning": 0.25,
    "restrict": 0.50,
    "halt": 1.0,
}


class IncidentState(str, Enum):
    """Incident lifecycle state."""

    FIRING = "firing"
    ACKNOWLEDGED = "acknowledged"
    RESOLVING = "resolving"
    RESOLVED = "resolved"


@dataclass
class Incident:
    """Latest known state for one incident."""

    incident_id: str
    channel: str
    severity: str
    state: IncidentState
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    resolved_at: str | None = None
    resolution_notes: str | None = None
    mttr_seconds: float | None = None
    alert_count: int = 1
    kill_switch_level: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Incident":
        data = dict(payload)
        data["state"] = IncidentState(data["state"])
        return cls(**data)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _severity_for_level(level: str) -> str | None:
    if level == "halt":
        return "p0"
    if level == "warn":
        return "p2"
    return None


def _normalise(value: Any) -> str:
    return str(getattr(value, "value", value))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


class IncidentManager:
    """Append-only incident event log plus queryable summary writer."""

    def __init__(
        self,
        log_path: str | Path = DEFAULT_INCIDENT_LOG_PATH,
        summary_path: str | Path = DEFAULT_INCIDENT_SUMMARY_PATH,
        kill_switch_path: str | Path | None = None,
        escalation_cycles: int | None = None,
        escalation_enabled: bool | None = None,
    ):
        self.log_path = Path(log_path)
        self.summary_path = Path(summary_path)
        self.kill_switch_path = (
            Path(kill_switch_path)
            if kill_switch_path is not None
            else self.log_path.parent / DEFAULT_KILL_SWITCH_PATH.name
        )
        cycles = escalation_cycles
        if cycles is None:
            cycles = _env_int("INCIDENT_KILL_SWITCH_ESCALATION_CYCLES", 3)
        self.escalation_cycles = max(1, cycles)
        self.escalation_enabled = (
            _env_bool("INCIDENT_KILL_SWITCH_ESCALATION_ENABLED", True)
            if escalation_enabled is None
            else escalation_enabled
        )

    def record_alert(
        self,
        *,
        channel: Any,
        level: Any,
        message: str,
        details: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> Incident | None:
        """Record an alert transition as an incident lifecycle event."""
        channel_value = _normalise(channel)
        level_value = _normalise(level)
        event_time = now or _utc_now()

        if level_value == "pass":
            incident = self._resolve(channel_value, message, event_time)
            if incident is not None:
                self._clear_matching_incident_kill_switch(incident)
        else:
            severity = _severity_for_level(level_value)
            if severity is None:
                return None
            incident = self._open_or_update(
                channel=channel_value,
                severity=severity,
                message=message,
                details=details or {},
                now=event_time,
            )
            self._write_incident_kill_switch(incident)

        self.write_summary()
        return incident

    def open_incidents(self) -> list[Incident]:
        return [
            incident
            for incident in self._latest_incidents().values()
            if incident.state != IncidentState.RESOLVED
        ]

    def metrics(self) -> dict[str, Any]:
        events = self._read_events()
        opened = [event for event in events if event.get("event") == "opened"]
        resolved = [
            event
            for event in events
            if event.get("event") == "resolved" and event.get("mttr_seconds") is not None
        ]
        mean_mttr = None
        if resolved:
            mean_mttr = sum(float(event["mttr_seconds"]) for event in resolved) / len(resolved)

        return {
            "incident_frequency": len(opened),
            "open_count": len(self.open_incidents()),
            "resolved_count": len(resolved),
            "mean_mttr_seconds": mean_mttr,
        }

    def write_summary(self) -> dict[str, Any]:
        """Write current open incidents and metrics to incidents.json.

        Dual-writes to ``PUBLIC_DATA_DIR/incidents.json`` when that tree is
        distinct from the private summary path so operators never see open_count
        split-brain (private firing vs public zero) after PASS resolve without
        a full dashboard cycle.
        """
        incidents = sorted(
            self.open_incidents(),
            key=lambda incident: (incident.created_at, incident.incident_id),
        )
        summary = {
            "schema_version": "incident-lifecycle/v1",
            "generated_at": _iso(_utc_now()),
            "open_count": len(incidents),
            "incidents": [incident.to_dict() for incident in incidents],
            "metrics": self.metrics(),
        }
        try:
            from src.dashboard.generator import _stamp_generator_git_sha

            summary = _stamp_generator_git_sha(summary)
        except Exception:  # noqa: BLE001 — never block incident SSOT write
            pass

        public_summary = Path(PUBLIC_DATA_DIR) / "incidents.json"
        dual_attempted = False
        dual_ok: bool | None = None
        paths_identical = False
        try:
            paths_identical = public_summary.resolve() == self.summary_path.resolve()
        except OSError:
            paths_identical = False

        # Stamp completeness *before* private write so both trees share metadata
        try:
            from src.dashboard.generator import _attach_dual_write_provenance

            # dual_write_ok filled after public attempt; first pass records intent
            summary = _attach_dual_write_provenance(
                summary,
                private_path=self.summary_path,
                public_path=public_summary,
                dual_write_attempted=not paths_identical,
                dual_write_ok=None if not paths_identical else True,
                paths_identical=paths_identical,
            )
        except Exception:  # noqa: BLE001
            pass

        body = json.dumps(summary, indent=2)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(body, encoding="utf-8")
        # Atomic dual-write to live WWW SSOT when configured
        try:
            if not paths_identical:
                dual_attempted = True
                public_summary.parent.mkdir(parents=True, exist_ok=True)
                tmp = public_summary.with_suffix(".json.tmp")
                # Refresh dual_write_ok=True into body for public tree
                try:
                    from src.dashboard.generator import _attach_dual_write_provenance

                    summary = _attach_dual_write_provenance(
                        summary,
                        private_path=self.summary_path,
                        public_path=public_summary,
                        dual_write_attempted=True,
                        dual_write_ok=True,
                        paths_identical=False,
                    )
                    body = json.dumps(summary, indent=2)
                    # Keep private tree in sync with final completeness block
                    self.summary_path.write_text(body, encoding="utf-8")
                except Exception:  # noqa: BLE001
                    pass
                tmp.write_text(body, encoding="utf-8")
                tmp.replace(public_summary)
                dual_ok = True
        except OSError as exc:
            dual_ok = False
            logger.warning("Public incidents dual-write failed: %s", exc)
            try:
                from src.dashboard.generator import _attach_dual_write_provenance

                summary = _attach_dual_write_provenance(
                    summary,
                    private_path=self.summary_path,
                    public_path=public_summary,
                    dual_write_attempted=True,
                    dual_write_ok=False,
                    paths_identical=False,
                    note=str(exc),
                )
                self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
        return summary

    def _open_or_update(
        self,
        *,
        channel: str,
        severity: str,
        message: str,
        details: dict[str, Any],
        now: datetime,
    ) -> Incident:
        existing = self._find_open_by_channel(channel)
        timestamp = _iso(now)

        if existing is None:
            incident = Incident(
                incident_id=str(uuid.uuid4()),
                channel=channel,
                severity=severity,
                state=IncidentState.FIRING,
                message=message,
                details=details,
                created_at=timestamp,
                updated_at=timestamp,
            )
            incident.kill_switch_level = self._kill_switch_level_for_count(
                incident.alert_count, severity=severity
            )
            self._append_event("opened", incident)
            return incident

        incident = Incident(
            incident_id=existing.incident_id,
            channel=channel,
            severity=severity,
            state=existing.state,
            message=message,
            details=details,
            created_at=existing.created_at,
            updated_at=timestamp,
            alert_count=existing.alert_count + 1,
        )
        incident.kill_switch_level = self._kill_switch_level_for_count(
            incident.alert_count, severity=severity
        )
        self._append_event("updated", incident)
        return incident

    def _resolve(self, channel: str, message: str, now: datetime) -> Incident | None:
        existing = self._find_open_by_channel(channel)
        if existing is None:
            return None

        resolved_at = _iso(now)
        created = datetime.fromisoformat(existing.created_at)
        mttr_seconds = (now.astimezone(timezone.utc) - created.astimezone(timezone.utc)).total_seconds()
        incident = Incident(
            incident_id=existing.incident_id,
            channel=existing.channel,
            severity=existing.severity,
            state=IncidentState.RESOLVED,
            message=message,
            details=existing.details,
            created_at=existing.created_at,
            updated_at=resolved_at,
            resolved_at=resolved_at,
            resolution_notes=message,
            mttr_seconds=mttr_seconds,
            alert_count=existing.alert_count,
            kill_switch_level=existing.kill_switch_level,
        )
        self._append_event("resolved", incident)
        return incident

    def _kill_switch_level_for_count(
        self, alert_count: int, *, severity: str | None = None
    ) -> str | None:
        """Map alert_count to kill stage, capped by incident severity.

        p0 (classifier HALT) may escalate warning → restrict → halt.
        Lower severities (e.g. p2 WARN) max out at ``warning`` so sustained
        optional/sheddable alerts cannot ratchet paper trading into halt.
        """
        if not self.escalation_enabled:
            return None
        if alert_count < self.escalation_cycles:
            return None
        stage = min(alert_count // self.escalation_cycles, 3)
        # Only p0 may progress past advisory warning.
        if severity != "p0":
            stage = min(stage, 1)
        return {
            1: "warning",
            2: "restrict",
            3: "halt",
        }[stage]

    def _write_incident_kill_switch(self, incident: Incident) -> None:
        level = incident.kill_switch_level
        if level is None:
            return
        if not self._should_write_incident_kill_switch(level):
            return

        payload = {
            "enabled": True,
            "level": level,
            "reason": f"unresolved_incident:{incident.channel}",
            "mode": os.environ.get("ALPHALAB_MODE", "paper"),
            "timestamp": incident.updated_at,
            "position_reduction": _KILL_SWITCH_REDUCTION[level],
            "source": "incident_lifecycle",
            "incident_id": incident.incident_id,
            "incident_channel": incident.channel,
            "incident_severity": incident.severity,
            "incident_alert_count": incident.alert_count,
            "message": incident.message,
        }
        try:
            self.kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
            self.kill_switch_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to write incident kill switch: %s", exc)

    def _write_most_restrictive_open_incident_kill_switch(self) -> None:
        candidates = [
            incident
            for incident in self.open_incidents()
            if incident.kill_switch_level is not None
        ]
        if not candidates:
            return
        incident = sorted(
            candidates,
            key=lambda item: (
                _KILL_SWITCH_LEVEL_RANK.get(item.kill_switch_level or "none", 0),
                item.updated_at,
                item.incident_id,
            ),
        )[-1]
        self._write_incident_kill_switch(incident)

    def _should_write_incident_kill_switch(self, level: str) -> bool:
        new_rank = _KILL_SWITCH_LEVEL_RANK[level]
        try:
            payload = json.loads(self.kill_switch_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return True
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Existing kill switch is unreadable; preserving it for safety: %s", exc)
            return False

        existing_level = str(payload.get("level", "none")).lower()
        existing_rank = _KILL_SWITCH_LEVEL_RANK.get(existing_level, _KILL_SWITCH_LEVEL_RANK["halt"])
        if existing_rank > new_rank:
            return False
        if existing_rank == new_rank and payload.get("source") != "incident_lifecycle":
            return False
        return True

    def _clear_matching_incident_kill_switch(self, incident: Incident) -> None:
        try:
            payload = json.loads(self.kill_switch_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Existing kill switch is unreadable; preserving it for safety: %s", exc)
            return

        if (
            payload.get("source") == "incident_lifecycle"
            and payload.get("incident_id") == incident.incident_id
        ):
            try:
                self.kill_switch_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to clear resolved incident kill switch: %s", exc)
                return
            self._write_most_restrictive_open_incident_kill_switch()

    def _find_open_by_channel(self, channel: str) -> Incident | None:
        candidates = [
            incident
            for incident in self.open_incidents()
            if incident.channel == channel
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda incident: incident.updated_at)[-1]

    def _latest_incidents(self) -> dict[str, Incident]:
        latest: dict[str, Incident] = {}
        for event in self._read_events():
            incident_id = event.get("incident_id")
            if not incident_id:
                continue
            try:
                latest[str(incident_id)] = Incident.from_dict({
                    key: value
                    for key, value in event.items()
                    if key not in {"event", "event_timestamp"}
                })
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed incident event in %s: %s", self.log_path, exc)
        return latest

    def _read_events(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []

        events: list[dict[str, Any]] = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed incident event in %s: %s", self.log_path, exc)
        return events

    def _append_event(self, event: str, incident: Incident) -> None:
        payload = {
            "event": event,
            "event_timestamp": incident.updated_at,
            **incident.to_dict(),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
