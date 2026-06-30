"""Persistent incident lifecycle for production alert transitions."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.paths import DATA_DIR

logger = logging.getLogger(__name__)

DEFAULT_INCIDENT_LOG_PATH = DATA_DIR / "incidents.jsonl"
DEFAULT_INCIDENT_SUMMARY_PATH = DATA_DIR / "incidents.json"


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


class IncidentManager:
    """Append-only incident event log plus queryable summary writer."""

    def __init__(
        self,
        log_path: str | Path = DEFAULT_INCIDENT_LOG_PATH,
        summary_path: str | Path = DEFAULT_INCIDENT_SUMMARY_PATH,
    ):
        self.log_path = Path(log_path)
        self.summary_path = Path(summary_path)

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
        """Write current open incidents and metrics to incidents.json."""
        incidents = sorted(
            self.open_incidents(),
            key=lambda incident: (incident.created_at, incident.incident_id),
        )
        summary = {
            "generated_at": _iso(_utc_now()),
            "open_count": len(incidents),
            "incidents": [incident.to_dict() for incident in incidents],
            "metrics": self.metrics(),
        }
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
        )
        self._append_event("resolved", incident)
        return incident

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
