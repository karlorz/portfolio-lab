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


def _is_test_isolation_path(path: Path | str | None) -> bool:
    """True when path is under portfolio-lab pytest dual-write isolation (Batch AX / CI).

    Live producers must never embed ``plab-pytest-public.*`` into production
    ``data/incidents.json`` provenance when a test process rebinds PUBLIC_DATA_DIR
    but still targets the real private summary path.

    Only the deliberate isolation prefix is matched — generic ``/tmp/pytest-*``
    (pytest's own tmp root) is not treated as PUBLIC isolation by itself.
    """
    if path is None:
        return False
    return "plab-pytest" in str(path)


# Repo-private operator SSOT (not monkeypatchable via DATA_DIR rebind in tests).
# src/monitor/incident_manager.py → parents[2] = project root.
_OPERATOR_PRIVATE_DATA = Path(__file__).resolve().parents[2] / "data"
_OPERATOR_WWW_DATA = Path("/var/www/portfolio-lab/data")


def _pytest_blocks_live_incident_write(path: Path | str | None) -> bool:
    """Refuse live incident/kill SSOT writes while a pytest test is running.

    Batch JG TI1: H16 isolates PUBLIC_DATA_DIR, but private DATA_DIR stays live
    unless each test rebinds IncidentManager paths. Under ``PYTEST_CURRENT_TEST``,
    default managers must not open/resolve/arm kill on the **operator** tree.

    Compare against fixed project/www paths only — never the import-time
    ``DATA_DIR`` alias, which tests monkeypatch to tmp and would false-block
    hermetic managers.

    Opt-in: ``PORTFOLIO_LAB_ALLOW_LIVE_INCIDENTS=1`` (explicit live-path tests only).
    """
    if path is None:
        return False
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if os.environ.get("PORTFOLIO_LAB_ALLOW_LIVE_INCIDENTS", "0") == "1":
        return False
    try:
        target = Path(path).resolve()
    except OSError:
        return False
    live_files: set[Path] = set()
    for root in (_OPERATOR_PRIVATE_DATA, _OPERATOR_WWW_DATA):
        try:
            live_files.add((root / "incidents.json").resolve())
            live_files.add((root / "incidents.jsonl").resolve())
            live_files.add((root / "kill_switch.json").resolve())
        except OSError:
            continue
    # Also block live DATA_DIR only when it still points at the real operator tree
    # (not a test monkeypatch to tmp).
    try:
        data_dir_resolved = Path(DATA_DIR).resolve()
        if data_dir_resolved == _OPERATOR_PRIVATE_DATA.resolve():
            live_files.add((data_dir_resolved / "incidents.json").resolve())
            live_files.add((data_dir_resolved / "incidents.jsonl").resolve())
            live_files.add((data_dir_resolved / "kill_switch.json").resolve())
    except OSError:
        pass
    return target in live_files

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
    # Task 2A: incidents on evidence-correction channels (ic_decay) require
    # explicit operator review; PASS alerts never auto-resolve them.
    manual_review_required: bool = False
    manual_review_reason: str | None = None

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
        # Batch IM DN: write-through kill/open onto mon/ops/public so operators
        # never wait for :00/:30 health cron after arm or clear.
        self._project_disk_kill_open_surfaces()
        return incident

    def resolve_operator(
        self,
        incident_id: str,
        message: str,
        now: datetime | None = None,
    ) -> Incident | None:
        """Explicit operator resolution (Task 2A path) by incident id.

        The operator's own action — bypasses the PASS-only manual-review hold
        (``_resolve`` :614-645) by design, but is never invoked by alert
        transitions. Mirrors the ``handle_alert`` resolution branch
        (journal append + ``_clear_matching_incident_kill_switch`` +
        ``write_summary`` + ``_project_disk_kill_open_surfaces``).

        Returns the resolved ``Incident``, or ``None`` when no OPEN incident
        matches (already resolved, or unknown id — callers distinguish via
        ``incident_state``).
        """
        event_time = now or _utc_now()
        existing = next(
            (
                incident
                for incident in self.open_incidents()
                if incident.incident_id == incident_id
            ),
            None,
        )
        if existing is None:
            return None

        resolved_at = _iso(event_time)
        created = datetime.fromisoformat(existing.created_at)
        mttr_seconds = (
            event_time.astimezone(timezone.utc)
            - created.astimezone(timezone.utc)
        ).total_seconds()
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
        self._clear_matching_incident_kill_switch(incident)
        self.write_summary()
        self._project_disk_kill_open_surfaces()
        logger.info(
            "Operator resolved incident %s (%s): %s",
            incident.incident_id,
            incident.channel,
            message,
        )
        return incident

    def incident_state(self, incident_id: str) -> str | None:
        """Latest lifecycle state for an incident id (``None`` if unknown)."""
        latest = self._latest_incidents().get(incident_id)
        return latest.state.value if latest is not None else None

    def _project_disk_kill_open_surfaces(self) -> None:
        """Best-effort fan-out of kill_switch.json + incidents onto health surfaces."""
        try:
            from src.monitor.health_check import project_disk_kill_open_to_all_surfaces

            # Prefer the directory that holds this manager's kill/summary SSOT
            # (tmp_path in tests; DATA_DIR in production).
            data_dir = self.kill_switch_path.parent
            project_disk_kill_open_to_all_surfaces(data_dir=data_dir)
        except Exception as exc:  # noqa: BLE001 — never block lifecycle
            logger.warning("Kill/open surface fan-out skipped: %s", exc)

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

        Batch BI residual honesty:
        - Private ``summary_path`` (DATA_DIR) is the open-set SSOT from this
          process's event log.
        - Public dual-write copies that private body atomically — never rebuilds
          public open set from a different process view that would clobber
          channels (e.g. evaluator_error replacing signal_staleness).
        - After public write, refresh index digests so sha/size stay honest.
        """
        if _pytest_blocks_live_incident_write(self.summary_path):
            logger.error(
                "TI1: refusing live incidents.json write under pytest (%s); "
                "set PORTFOLIO_LAB_ALLOW_LIVE_INCIDENTS=1 to override",
                self.summary_path,
            )
            return {
                "schema_version": "incident-lifecycle/v1",
                "generated_at": _iso(_utc_now()),
                "open_count": 0,
                "incidents": [],
                "metrics": {},
                "open_set_ssot": "blocked_live_under_pytest",
            }
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
            "open_set_ssot": "private_summary_path",
        }
        try:
            from src.dashboard.generator import _stamp_generator_git_sha

            summary = _stamp_generator_git_sha(summary)
        except Exception:  # noqa: BLE001 — never block incident SSOT write
            pass

        public_summary = Path(PUBLIC_DATA_DIR) / "incidents.json"
        paths_identical = False
        skip_dual_for_isolation = False
        try:
            paths_identical = public_summary.resolve() == self.summary_path.resolve()
        except OSError:
            paths_identical = False

        # Batch CI: never contaminate production private SSOT with pytest public
        # isolation paths (plab-pytest-public.*). Tests that dual-write both
        # sides under tmp still work (private also under tmp / isolation).
        #
        # Batch JG TI1: under pytest, never rebound dual-write to live WWW when
        # private is live DATA_DIR — that was the pollution amplifier. Skip dual
        # entirely (private write already blocked by TI1 if summary is live).
        private_is_live_ssot = False
        try:
            private_is_live_ssot = (
                self.summary_path.resolve()
                == (Path(DATA_DIR) / "incidents.json").resolve()
            )
        except OSError:
            private_is_live_ssot = False

        under_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
        allow_live_inc = os.environ.get("PORTFOLIO_LAB_ALLOW_LIVE_INCIDENTS", "0") == "1"

        if (
            not paths_identical
            and _is_test_isolation_path(public_summary)
            and private_is_live_ssot
        ):
            skip_dual_for_isolation = True
            if under_pytest and not allow_live_inc:
                paths_identical = True  # TI1: no live WWW dual-write under suite
                logger.warning(
                    "Incidents dual-write skipped under pytest (TI1): isolation "
                    "PUBLIC (%s) vs live private (%s)",
                    PUBLIC_DATA_DIR,
                    self.summary_path,
                )
            else:
                live_candidate = Path("/var/www/portfolio-lab/data") / "incidents.json"
                try:
                    if live_candidate.parent.is_dir() and not _is_test_isolation_path(
                        live_candidate
                    ):
                        public_summary = live_candidate
                        skip_dual_for_isolation = False
                        try:
                            paths_identical = (
                                public_summary.resolve() == self.summary_path.resolve()
                            )
                        except OSError:
                            paths_identical = False
                        logger.warning(
                            "Incidents dual-write: rebound pytest PUBLIC_DATA_DIR to "
                            "live operator tree %s (private is live DATA_DIR SSOT)",
                            public_summary,
                        )
                    else:
                        paths_identical = True  # skip dual; stamp local-only
                        logger.warning(
                            "Incidents dual-write skipped: PUBLIC_DATA_DIR is test "
                            "isolation (%s) while private summary is live (%s)",
                            PUBLIC_DATA_DIR,
                            self.summary_path,
                        )
                except OSError:
                    paths_identical = True
                    skip_dual_for_isolation = True
        elif (
            not paths_identical
            and _is_test_isolation_path(public_summary)
            and not private_is_live_ssot
            and not _is_test_isolation_path(self.summary_path)
        ):
            # Private is neither live DATA_DIR nor isolation-named (odd lab
            # path) — refuse dual-write into pytest public to avoid embedding
            # plab-pytest into any non-test private body.
            skip_dual_for_isolation = True
            paths_identical = True
            logger.warning(
                "Incidents dual-write skipped: pytest public (%s) vs non-live "
                "private (%s)",
                public_summary,
                self.summary_path,
            )

        # Stamp completeness *before* private write so both trees share metadata
        try:
            from src.dashboard.generator import _attach_dual_write_provenance

            # dual_write_ok filled after public attempt; first pass records intent
            summary = _attach_dual_write_provenance(
                summary,
                private_path=self.summary_path,
                public_path=public_summary if not skip_dual_for_isolation else self.summary_path,
                dual_write_attempted=not paths_identical and not skip_dual_for_isolation,
                dual_write_ok=None if (not paths_identical and not skip_dual_for_isolation) else True,
                paths_identical=paths_identical or skip_dual_for_isolation,
                note=(
                    "skipped dual-write: pytest PUBLIC_DATA_DIR isolation vs live private"
                    if skip_dual_for_isolation
                    else None
                ),
            )
        except Exception:  # noqa: BLE001
            pass

        body = json.dumps(summary, indent=2)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        # Task 5A: private incidents SSOT uses the canonical atomic writer so
        # an interrupted write can never leave a partial incidents.json.
        try:
            from src.monitor.signal_authority import _atomic_write_text

            _atomic_write_text(self.summary_path, body, mode=0o644)
        except Exception:  # noqa: BLE001 - fall back to plain write only on tooling failure
            self.summary_path.write_text(body, encoding="utf-8")
        # Atomic dual-write: public is a byte-copy of private SSOT (not a second
        # open-set derivation). Prevents secondary writers with a partial
        # process view from inventing a divergent public open set.
        try:
            if not paths_identical and not skip_dual_for_isolation:
                public_summary.parent.mkdir(parents=True, exist_ok=True)
                tmp = public_summary.with_suffix(".json.tmp")
                from src.monitor.signal_authority import (
                    is_ephemeral_write_path,
                    serialize_json_payload,
                )

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
                        note=(
                            "public incidents is dual-write copy of private "
                            "summary_path SSOT (Batch BI)"
                        ),
                    )
                    body = json.dumps(summary, indent=2)
                    # Keep private tree in sync with final completeness block
                    self.summary_path.write_text(body, encoding="utf-8")
                except Exception:  # noqa: BLE001
                    pass
                public_body = serialize_json_payload(
                    summary,
                    output_path=public_summary,
                    public=not is_ephemeral_write_path(public_summary),
                )
                tmp.write_text(public_body, encoding="utf-8")
                tmp.replace(public_summary)
                _ = True
                # Batch CJ: post-sync lag/hash so sticky dual_write_lag_stale clears
                try:
                    from src.dashboard.generator import (
                        finalize_dual_write_provenance_after_sync,
                    )

                    summary = finalize_dual_write_provenance_after_sync(
                        summary,
                        private_path=self.summary_path,
                        public_path=public_summary,
                        dual_write_ok=True,
                        note=(
                            "post_sync incidents dual-write (Batch CJ); public is "
                            "copy of private summary_path SSOT (Batch BI)"
                        ),
                    )
                except Exception:  # noqa: BLE001
                    pass
                # Content-addressed catalog must update digests after partial write
                try:
                    from src.dashboard.public_data_index import (
                        refresh_public_data_index_after_partial_write,
                    )

                    refresh_public_data_index_after_partial_write(
                        public_dir=public_summary.parent,
                        extra_paths=[public_summary],
                        reason="incidents_dual_write",
                    )
                except Exception:  # noqa: BLE001
                    pass
        except OSError as exc:
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
            # Task 2A: evidence-correction channels require explicit operator
            # review; a green producer run must never close them.
            # (Channel value mirrors AlertChannel.IC_DECAY = "ic_decay";
            # imported as a literal to avoid a module cycle with alerting.py.)
            if channel == "ic_decay":
                incident.manual_review_required = True
                incident.manual_review_reason = "ic_evidence_correction"
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
            manual_review_required=existing.manual_review_required,
            manual_review_reason=existing.manual_review_reason,
        )
        incident.kill_switch_level = self._kill_switch_level_for_count(
            incident.alert_count, severity=severity
        )
        # Task 2A: evidence-correction channels require explicit operator
        # review; a green producer run must never close them. Applied on both
        # open and update so pre-existing ic_decay incidents (e.g. live
        # incident 8115a9c1) become manual-review-required on their next
        # update without rewriting their history.
        # (Channel value mirrors AlertChannel.IC_DECAY = "ic_decay";
        # imported as a literal to avoid a module cycle with alerting.py.)
        if channel == "ic_decay":
            incident.manual_review_required = True
            incident.manual_review_reason = "ic_evidence_correction"
        self._append_event("updated", incident)
        return incident

    def _resolve(self, channel: str, message: str, now: datetime) -> Incident | None:
        existing = self._find_open_by_channel(channel)
        if existing is None:
            return None

        if existing.manual_review_required or channel == "ic_decay":
            # Task 2A: evidence-correction channels (ic_decay) require explicit
            # operator review; PASS alerts never auto-resolve them. The channel
            # check is the guarantee — it also covers incidents opened before
            # the flag existed (e.g. live incident 8115a9c1), whose persisted
            # flag may still be False until their next non-PASS update. Record
            # the PASS attempt without fabricating a healthy resolution event.
            hold_reason = existing.manual_review_reason or "ic_evidence_correction"
            logger.warning(
                "Incident %s on channel %s held for manual review (%s); PASS not resolved",
                existing.incident_id,
                channel,
                hold_reason,
            )
            self._append_event(
                "pass_held_for_manual_review",
                Incident(
                    incident_id=existing.incident_id,
                    channel=existing.channel,
                    severity=existing.severity,
                    state=existing.state,
                    message=message,
                    details=existing.details,
                    created_at=existing.created_at,
                    updated_at=_iso(now),
                    alert_count=existing.alert_count,
                    kill_switch_level=existing.kill_switch_level,
                    manual_review_required=True,
                    manual_review_reason=hold_reason,
                ),
            )
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
        if _pytest_blocks_live_incident_write(self.kill_switch_path):
            logger.error(
                "TI1: refusing live kill_switch.json write under pytest (%s)",
                self.kill_switch_path,
            )
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
        if _pytest_blocks_live_incident_write(self.kill_switch_path):
            logger.error(
                "TI1: refusing live kill_switch clear under pytest (%s)",
                self.kill_switch_path,
            )
            return
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
        if _pytest_blocks_live_incident_write(self.log_path):
            logger.error(
                "TI1: refusing live incidents.jsonl append under pytest (%s)",
                self.log_path,
            )
            return
        payload = {
            "event": event,
            "event_timestamp": incident.updated_at,
            **incident.to_dict(),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
