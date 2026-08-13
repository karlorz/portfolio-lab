"""portfolio-lab-prod-ideas — hybrid prod→dev capture (machine channel SSOT).

Scans live ops SSOT and maintains ``data/prod_idea_channels.json`` with
deduped channel records. Promote is badge-only (``promote_candidate``);
never creates SkillWiki planned work items.

Pure path: ``scan_ssot`` / ``apply_channel_delta`` / promote helpers.
I/O path: ``load_ssot_snapshot`` / ``write_channel_ssot`` / ``run_once``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.paths import DATA_DIR, PROJECT_ROOT, resolve_runtime_public_data_dir

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "prod-idea-channels/v1"
CHANNEL_SSOT_NAME = "prod_idea_channels.json"

# Promote badge thresholds (tunable)
PROMOTE_OPEN_HOURS = 6.0
PROMOTE_CONSECUTIVE_FAILURES = 3
PROMOTE_REARM_HOURS = 24.0

# Cron/tasker statuses that count as failure (enabled jobs only)
_FAIL_STATUSES = frozenset(
    {
        "error",
        "failed",
        "failure",
        "timeout",
        "oom",
        "crash",
        "exception",
    }
)

# Alert types / levels that open health-class channels
_HEALTH_ALERT_TYPES = frozenset({"health_slo", "system_status", "scheduler"})
_HEALTH_ALERT_LEVELS = frozenset({"error", "critical", "fatal"})

# Hold / unavailability reasons that must NOT open channels alone
_IGNORE_REBALANCE_HOLD_REASONS = frozenset(
    {
        "budget",
        "budget_exceeded",
        "budget_overage",
        "budget_hold",
        "drift_only",
        "drift",
        "cost_budget",
    }
)
_IGNORE_FRED_REASONS = frozenset(
    {
        "missing_fred_api_key",
        "fred_unavailable",
        "no_fred_key",
        "ml_off",
        "ml_disabled",
    }
)

# Evidence keys allowed in channel records (structured ops only)
_EVIDENCE_ALLOW = frozenset(
    {
        "job_name",
        "task_id",
        "status",
        "state",
        "exit_code",
        "last_exit_code",
        "level",
        "reason",
        "message",
        "incident_id",
        "incident_channel",
        "severity",
        "alert_type",
        "alert_level",
        "system_status",
        "scheduler_status",
        "failed_jobs",
        "consecutive_failures",
        "unavailable_names",
        "log_path",
        "timestamp",
        "source",
        "backend",
        "effects",
        "channel",
        "kill_level",
        "rebalance_kill_block",
    }
)


def _ts(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fingerprint(parts: dict[str, Any]) -> str:
    """Stable short fingerprint from sorted JSON of material fields."""
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _sanitize_evidence(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in raw.items():
        if key not in _EVIDENCE_ALLOW:
            continue
        if key == "unavailable_names" and isinstance(val, (list, tuple, set)):
            out[key] = sorted(str(x) for x in val)
            continue
        if key == "log_path" and val is not None:
            out[key] = _relative_log_path(str(val))
            continue
        if key == "message" and isinstance(val, str) and len(val) > 240:
            out[key] = val[:240]
            continue
        if key in {"reason", "message"} and isinstance(val, str):
            out[key] = val[:240]
            continue
        out[key] = val
    return out


def _relative_log_path(path: str) -> str:
    """Prefer ``data/tasker_logs/...`` relative form; never embed log bodies."""
    text = path.strip()
    if not text:
        return text
    # Already relative under data/
    if text.startswith("data/"):
        return text
    try:
        p = Path(text)
        if p.is_absolute():
            try:
                rel = p.relative_to(PROJECT_ROOT)
                return str(rel).replace("\\", "/")
            except ValueError:
                # keep basename under conventional prefix when outside repo
                return f"data/tasker_logs/{p.name}"
    except (OSError, TypeError, ValueError):
        pass
    if "/" in text or "\\" in text:
        name = Path(text).name
        return f"data/tasker_logs/{name}"
    return f"data/tasker_logs/{text}"


def _observation(
    *,
    key: str,
    channel: str,
    fingerprint_parts: dict[str, Any],
    evidence: dict[str, Any],
    consecutive_failures: int = 0,
    effects: list[str] | None = None,
) -> dict[str, Any]:
    ev = _sanitize_evidence(evidence)
    if effects:
        ev["effects"] = list(effects)
    fp = _fingerprint(fingerprint_parts)
    return {
        "key": key,
        "channel": channel,
        "fingerprint": fp,
        "evidence": ev,
        "consecutive_failures": int(consecutive_failures),
    }


# ── Pure scan ────────────────────────────────────────────────────────────────


def scan_ssot(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Scan a loaded SSOT snapshot into current open-condition observations.

    Intake filter (open only for):
      - kill enabled
      - firing open incidents (keyed by incident channel)
      - enabled cron/tasker failures (keyed by job/task id)
      - health critical/error or scheduler failed_jobs > 0 / health_slo-class

    Explicit non-open:
      - budget/drift-only rebalance holds
      - GARCH-only tail warnings
      - intentional FRED / ML-off unavailability alone
    """
    if not isinstance(snapshot, dict):
        return []

    observations: list[dict[str, Any]] = []
    kill = snapshot.get("kill_switch") if isinstance(snapshot.get("kill_switch"), dict) else {}
    incidents_block = (
        snapshot.get("incidents") if isinstance(snapshot.get("incidents"), dict) else {}
    )
    cron = snapshot.get("cron_status") if isinstance(snapshot.get("cron_status"), dict) else {}
    health = snapshot.get("health") if isinstance(snapshot.get("health"), dict) else {}
    alerts_block = snapshot.get("alerts") if isinstance(snapshot.get("alerts"), dict) else {}
    tasker_state = snapshot.get("tasker_state") if isinstance(snapshot.get("tasker_state"), list) else []

    # Kill switch
    if bool(kill.get("enabled")):
        level = str(kill.get("level") or "").lower() or None
        reason = kill.get("reason")
        effects: list[str] = []
        # rebalance kill-block is an effect, not a separate channel
        if level in {"restrict", "halt", "warning"}:
            effects.append("rebalance_kill_block")
        observations.append(
            _observation(
                key="kill_switch",
                channel="kill_switch",
                fingerprint_parts={
                    "kind": "kill",
                    "level": level,
                    "reason": reason,
                    "incident_id": kill.get("incident_id"),
                    "source": kill.get("source"),
                },
                evidence={
                    "level": level,
                    "reason": reason,
                    "message": kill.get("message"),
                    "timestamp": kill.get("timestamp"),
                    "incident_id": kill.get("incident_id"),
                    "source": kill.get("source"),
                    "channel": kill.get("channel") or kill.get("incident_channel"),
                    "rebalance_kill_block": True if effects else False,
                },
                effects=effects or None,
            )
        )

    # Open incidents
    raw_incidents = incidents_block.get("incidents") or incidents_block.get("open_incidents") or []
    if isinstance(raw_incidents, list):
        for inc in raw_incidents:
            if not isinstance(inc, dict):
                continue
            state = str(inc.get("state") or inc.get("status") or "open").lower()
            if state in {"closed", "resolved", "pass", "cleared"}:
                continue
            ch = str(inc.get("channel") or inc.get("incident_channel") or "unknown")
            inc_id = inc.get("incident_id") or inc.get("id")
            observations.append(
                _observation(
                    key=f"incident:{ch}",
                    channel="incident",
                    fingerprint_parts={
                        "kind": "incident",
                        "channel": ch,
                        "severity": inc.get("severity"),
                        "kill_switch_level": inc.get("kill_switch_level"),
                        "incident_id": inc_id,
                    },
                    evidence={
                        "incident_id": inc_id,
                        "incident_channel": ch,
                        "severity": inc.get("severity"),
                        "message": inc.get("message"),
                        "kill_level": inc.get("kill_switch_level"),
                        "status": state,
                    },
                )
            )

    # Cron job failures (enabled, non-manual-only)
    jobs = cron.get("jobs") if isinstance(cron.get("jobs"), list) else []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        name = str(job.get("name") or job.get("id") or "").strip()
        if not name:
            continue
        if job.get("manual_only") is True:
            continue
        if job.get("enabled") is False:
            continue
        status = str(job.get("status") or "").lower()
        state = str(job.get("state") or "").lower()
        if status not in _FAIL_STATUSES:
            continue
        observations.append(
            _observation(
                key=f"cron_failed:{name}",
                channel="cron_failed",
                fingerprint_parts={
                    "kind": "cron_failed",
                    "job": name,
                    "status": status,
                    "state": state,
                },
                evidence={
                    "job_name": name,
                    "status": status,
                    "state": state or None,
                    "timestamp": job.get("last_run"),
                    "backend": job.get("backend"),
                    "exit_code": job.get("exit_code") or job.get("last_exit_code"),
                },
            )
        )

    # Tasker consecutive failures
    for row in tasker_state:
        if not isinstance(row, dict):
            continue
        task_id = str(row.get("task_id") or row.get("id") or "").strip()
        if not task_id:
            continue
        if row.get("enabled") is False:
            continue
        consecutive = int(row.get("consecutive_failures") or 0)
        last_status = str(row.get("last_status") or "").lower()
        if consecutive < 1 and last_status not in _FAIL_STATUSES:
            continue
        if consecutive < 1:
            continue
        log_path = row.get("log_path")
        if not log_path and row.get("last_run_id"):
            log_path = f"data/tasker_logs/{row['last_run_id']}.log"
        observations.append(
            _observation(
                key=f"tasker_failed:{task_id}",
                channel="tasker_failed",
                fingerprint_parts={
                    "kind": "tasker_failed",
                    "task_id": task_id,
                    "last_status": last_status,
                    "exit_code": row.get("last_exit_code") or row.get("exit_code"),
                },
                evidence={
                    "task_id": task_id,
                    "status": last_status or None,
                    "consecutive_failures": consecutive,
                    "last_exit_code": row.get("last_exit_code") or row.get("exit_code"),
                    "log_path": log_path,
                    "timestamp": row.get("last_finished_at") or row.get("updated_at"),
                },
                consecutive_failures=consecutive,
            )
        )

    # Health critical / scheduler failed_jobs / health_slo-class alerts
    system_status = health.get("system_status")
    if isinstance(system_status, dict):
        system_status = system_status.get("status") or system_status.get("level")
    system_status_s = str(system_status or "").lower()

    sched = health.get("scheduler_status") if isinstance(health.get("scheduler_status"), dict) else {}
    sched_status = str(sched.get("status") or "").lower()
    failed_jobs = 0
    backends = sched.get("backends") if isinstance(sched.get("backends"), dict) else {}
    for _name, backend in backends.items():
        if isinstance(backend, dict):
            try:
                failed_jobs = max(failed_jobs, int(backend.get("failed_jobs") or 0))
            except (TypeError, ValueError):
                pass
    if failed_jobs == 0:
        try:
            failed_jobs = int(sched.get("failed_jobs") or 0)
        except (TypeError, ValueError):
            failed_jobs = 0

    unavail = health.get("unavailable_names")
    if not isinstance(unavail, list):
        unavail = []

    # Do not open for FRED-only / ML-off alone when system is merely degraded
    fred = health.get("fred_readiness") if isinstance(health.get("fred_readiness"), dict) else {}
    if not fred and isinstance(snapshot.get("fred_readiness"), dict):
        fred = snapshot["fred_readiness"]  # type: ignore[assignment]
    fred_reason = str(fred.get("reason") or "").lower()

    if system_status_s in {"critical", "error", "fatal"}:
        observations.append(
            _observation(
                key="health_critical",
                channel="health_critical",
                fingerprint_parts={
                    "kind": "health_critical",
                    "system_status": system_status_s,
                    "unavailable": sorted(str(x) for x in unavail),
                },
                evidence={
                    "system_status": system_status_s,
                    "unavailable_names": unavail,
                    "scheduler_status": sched_status or None,
                },
            )
        )

    if failed_jobs > 0 or sched_status in {"error", "critical", "failed"}:
        observations.append(
            _observation(
                key="scheduler_failed",
                channel="scheduler_failed",
                fingerprint_parts={
                    "kind": "scheduler_failed",
                    "failed_jobs": failed_jobs,
                    "status": sched_status,
                },
                evidence={
                    "failed_jobs": failed_jobs,
                    "scheduler_status": sched_status or None,
                    "status": sched_status or None,
                },
            )
        )

    alerts = alerts_block.get("alerts") if isinstance(alerts_block.get("alerts"), list) else []
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        a_type = str(alert.get("type") or "").lower()
        a_level = str(alert.get("level") or "").lower()
        # Skip GARCH-only tail warnings
        if a_type in {"garch_tail", "garch_warning", "garch"} and a_level not in _HEALTH_ALERT_LEVELS:
            continue
        if a_type in _HEALTH_ALERT_TYPES and a_level in _HEALTH_ALERT_LEVELS:
            # Avoid double-open when we already have health_critical from system_status
            if system_status_s in {"critical", "error", "fatal"} and a_type == "health_slo":
                # still fingerprint into health_critical evidence only — skip separate alert key
                continue
            observations.append(
                _observation(
                    key=f"alert:{a_type}",
                    channel="alert",
                    fingerprint_parts={
                        "kind": "alert",
                        "type": a_type,
                        "level": a_level,
                        "reason": alert.get("reason"),
                    },
                    evidence={
                        "alert_type": a_type,
                        "alert_level": a_level,
                        "message": alert.get("message") or alert.get("title"),
                        "system_status": alert.get("system_status"),
                        "scheduler_status": alert.get("scheduler_status"),
                        "reason": alert.get("reason"),
                    },
                )
            )
        elif a_level in _HEALTH_ALERT_LEVELS and a_type not in {
            "garch_tail",
            "garch_warning",
            "garch",
            "fred",
            "ml_unavailable",
        }:
            observations.append(
                _observation(
                    key=f"alert:{a_type or 'ops'}",
                    channel="alert",
                    fingerprint_parts={
                        "kind": "alert",
                        "type": a_type,
                        "level": a_level,
                    },
                    evidence={
                        "alert_type": a_type or None,
                        "alert_level": a_level,
                        "message": alert.get("message") or alert.get("title"),
                    },
                )
            )

    # Explicit non-goals: rebalance budget/drift hold alone never opens
    rebalance = snapshot.get("rebalance") if isinstance(snapshot.get("rebalance"), dict) else {}
    hold_reason = str(rebalance.get("hold_reason") or rebalance.get("reason") or "").lower()
    if hold_reason in _IGNORE_REBALANCE_HOLD_REASONS:
        pass  # intentionally ignored

    # FRED / ML-off alone: if the only "signal" would be fred readiness, do not open
    if fred_reason in _IGNORE_FRED_REASONS and not observations:
        return []

    # Dedupe by key (last wins — same scan shouldn't emit duplicates)
    by_key: dict[str, dict[str, Any]] = {}
    for obs in observations:
        by_key[obs["key"]] = obs
    return list(by_key.values())


# ── Pure apply / lifecycle ───────────────────────────────────────────────────


def _compute_promote(
    row: dict[str, Any],
    *,
    now: datetime,
    consecutive_failures: int = 0,
    rearmed: bool = False,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    first = _parse_ts(row.get("first_seen"))
    if first is not None and row.get("status") == "open":
        open_hours = (now - first).total_seconds() / 3600.0
        if open_hours >= PROMOTE_OPEN_HOURS:
            reasons.append("open_ge_6h")
    if consecutive_failures >= PROMOTE_CONSECUTIVE_FAILURES:
        reasons.append("consecutive_failures_ge_3")
    if rearmed:
        reasons.append("kill_rearm_within_24h")
    # preserve prior reasons that still apply from sticky badge? recompute each time
    return (bool(reasons), reasons)


def apply_channel_delta(
    prior_state: dict[str, Any] | None,
    observations: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Merge scan observations into channel SSOT state.

    Returns (new_state, lifecycle_events). Events types:
      first_open | observation | fingerprint_change | cleared | promote_badge_flip | reopened
    Never creates planned work items.
    """
    now = now or _now()
    now_s = _ts(now)
    prior_channels = []
    if isinstance(prior_state, dict):
        prior_channels = prior_state.get("channels") or []
        if not isinstance(prior_channels, list):
            prior_channels = []

    by_key: dict[str, dict[str, Any]] = {}
    for ch in prior_channels:
        if isinstance(ch, dict) and ch.get("key"):
            by_key[str(ch["key"])] = dict(ch)

    events: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for obs in observations:
        key = str(obs["key"])
        seen_keys.add(key)
        existing = by_key.get(key)
        consecutive = int(obs.get("consecutive_failures") or 0)
        if existing is None:
            row = {
                "key": key,
                "channel": obs["channel"],
                "fingerprint": obs["fingerprint"],
                "status": "open",
                "first_seen": now_s,
                "last_seen": now_s,
                "cleared_at": None,
                "observation_count": 1,
                "promote_candidate": False,
                "promote_reasons": [],
                "evidence": dict(obs.get("evidence") or {}),
                "consecutive_failures": consecutive,
                "last_cleared_at": None,
            }
            promote, reasons = _compute_promote(
                row, now=now, consecutive_failures=consecutive, rearmed=False
            )
            row["promote_candidate"] = promote
            row["promote_reasons"] = reasons
            by_key[key] = row
            events.append({"type": "first_open", "key": key, "at": now_s})
            if promote:
                events.append({"type": "promote_badge_flip", "key": key, "at": now_s})
            continue

        prev_status = str(existing.get("status") or "open")
        prev_fp = existing.get("fingerprint")
        prev_promote = bool(existing.get("promote_candidate"))
        rearmed = False

        if prev_status == "cleared":
            # reopen — check kill re-arm within 24h for kill_switch key
            cleared_at = _parse_ts(existing.get("cleared_at") or existing.get("last_cleared_at"))
            if key == "kill_switch" and cleared_at is not None:
                hours = (now - cleared_at).total_seconds() / 3600.0
                if hours <= PROMOTE_REARM_HOURS:
                    rearmed = True
            existing["status"] = "open"
            existing["cleared_at"] = None
            existing["first_seen"] = now_s  # new open episode
            existing["observation_count"] = 1
            events.append({"type": "reopened", "key": key, "at": now_s})
        else:
            existing["observation_count"] = int(existing.get("observation_count") or 0) + 1

        existing["last_seen"] = now_s
        existing["channel"] = obs["channel"]
        existing["evidence"] = dict(obs.get("evidence") or {})
        existing["consecutive_failures"] = consecutive

        if prev_fp != obs["fingerprint"]:
            existing["fingerprint"] = obs["fingerprint"]
            events.append({"type": "fingerprint_change", "key": key, "at": now_s})
        else:
            events.append({"type": "observation", "key": key, "at": now_s})

        # For sticky open episode, keep original first_seen when not reopened
        if prev_status != "cleared" and not existing.get("first_seen"):
            existing["first_seen"] = now_s

        promote, reasons = _compute_promote(
            existing, now=now, consecutive_failures=consecutive, rearmed=rearmed
        )
        existing["promote_candidate"] = promote
        existing["promote_reasons"] = reasons
        if promote and not prev_promote:
            events.append({"type": "promote_badge_flip", "key": key, "at": now_s})
        by_key[key] = existing

    # Clear conditions no longer observed
    for key, row in list(by_key.items()):
        if key in seen_keys:
            continue
        if str(row.get("status") or "open") == "open":
            row["status"] = "cleared"
            row["cleared_at"] = now_s
            row["last_cleared_at"] = now_s
            row["last_seen"] = now_s
            # badge stays as historical signal but is not auto-promoted
            events.append({"type": "cleared", "key": key, "at": now_s})
            by_key[key] = row

    channels = sorted(by_key.values(), key=lambda r: str(r.get("key") or ""))
    state = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_s,
        "channels": channels,
        "open_count": sum(1 for c in channels if c.get("status") == "open"),
        "promote_candidates": [
            c["key"] for c in channels if c.get("promote_candidate") and c.get("status") == "open"
        ],
    }
    return state, events


# ── I/O loaders / writers ────────────────────────────────────────────────────


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_tasker_consecutive_failures(
    tasker_db: Path | None,
    *,
    data_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Read consecutive_failures from tasker.db task_state when available."""
    if tasker_db is None:
        root = data_dir or DATA_DIR
        tasker_db = Path(root) / "tasker.db"
    path = Path(tasker_db)
    if not path.is_file():
        return []
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        logger.warning("tasker.db unreadable: %s", exc)
        return []
    try:
        try:
            rows = con.execute(
                """
                SELECT task_id, consecutive_failures, last_status, last_exit_code,
                       last_run_id, last_finished_at, updated_at
                FROM task_state
                """
            ).fetchall()
        except sqlite3.Error as exc:
            logger.warning("task_state query failed: %s", exc)
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            task_id = row["task_id"]
            last_run_id = row["last_run_id"]
            log_path = None
            if last_run_id:
                log_path = f"data/tasker_logs/{last_run_id}.log"
            out.append(
                {
                    "task_id": task_id,
                    "enabled": True,  # disabled tasks still in state; cron scan gates enabled
                    "consecutive_failures": int(row["consecutive_failures"] or 0),
                    "last_status": row["last_status"],
                    "last_exit_code": row["last_exit_code"],
                    "last_run_id": last_run_id,
                    "log_path": log_path,
                    "last_finished_at": row["last_finished_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return out
    finally:
        con.close()


def load_ssot_snapshot(
    *,
    data_dir: Path | str | None = None,
    public_data_dir: Path | str | None = None,
    tasker_db: Path | str | None = None,
    now: datetime | None = None,  # noqa: ARG001 — reserved for clock injection
) -> dict[str, Any]:
    """Load cron/kill/incidents/health/alerts/tasker into a scan snapshot."""
    data = Path(data_dir) if data_dir is not None else DATA_DIR
    if public_data_dir is not None:
        public = Path(public_data_dir)
    else:
        public = resolve_runtime_public_data_dir()

    kill = _read_json(data / "kill_switch.json") or {}
    # health may embed kill_switch projection when file absent
    health = _read_json(public / "health.json") or _read_json(data / "health.json") or {}
    if not kill and isinstance(health.get("kill_switch"), dict):
        kill = health["kill_switch"]

    incidents = _read_json(data / "incidents.json") or {}
    if not incidents and isinstance(health.get("open_incidents"), dict):
        incidents = health["open_incidents"]

    cron = _read_json(data / "cron_status.json") or {}
    alerts = _read_json(public / "alerts.json") or _read_json(data / "alerts.json") or {}

    rebalance = (
        _read_json(public / "rebalance_health.json")
        or _read_json(data / "rebalance_health.json")
        or {}
    )
    # surface hold_reason if nested under next_rebalance / smart state
    if "hold_reason" not in rebalance:
        smart = _read_json(data / "smart_rebalance_state.json") or {}
        if isinstance(smart, dict) and smart.get("hold_reason"):
            rebalance = {**rebalance, "hold_reason": smart.get("hold_reason")}

    tdb = Path(tasker_db) if tasker_db is not None else data / "tasker.db"
    tasker_state = load_tasker_consecutive_failures(tdb if tdb.is_file() else None, data_dir=data)

    # Cross-check enabled flags from cron jobs onto tasker rows
    jobs = cron.get("jobs") if isinstance(cron.get("jobs"), list) else []
    enabled_map = {
        str(j.get("name")): bool(j.get("enabled", True)) and not bool(j.get("manual_only"))
        for j in jobs
        if isinstance(j, dict) and j.get("name")
    }
    for row in tasker_state:
        tid = row.get("task_id")
        if tid in enabled_map:
            row["enabled"] = enabled_map[tid]

    return {
        "kill_switch": kill,
        "incidents": incidents,
        "cron_status": cron,
        "health": health,
        "alerts": alerts,
        "tasker_state": tasker_state,
        "rebalance": rebalance,
        "fred_readiness": health.get("fred_readiness")
        if isinstance(health.get("fred_readiness"), dict)
        else {},
    }


def read_channel_ssot(path: Path | str) -> dict[str, Any]:
    payload = _read_json(Path(path))
    if not payload:
        return {"schema_version": SCHEMA_VERSION, "channels": [], "open_count": 0}
    return payload


def write_channel_ssot(
    path: Path | str,
    state: dict[str, Any],
    *,
    now: datetime | None = None,
) -> Path:
    now = now or _now()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["schema_version"] = SCHEMA_VERSION
    payload["generated_at"] = _ts(now)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out


def maybe_write_sparse_vault_note(
    events: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    vault_dir: Path | str | None,
    now: datetime | None = None,
) -> list[Path]:
    """Write sparse raw/transcripts notes on first_open / fingerprint_change / promote_badge_flip.

    Non-fatal when vault unavailable. Structured ops fields only.
    """
    now = now or _now()
    if vault_dir is None:
        return []
    root = Path(vault_dir)
    interesting = {
        e["key"]
        for e in events
        if e.get("type") in {"first_open", "fingerprint_change", "promote_badge_flip"}
    }
    if not interesting:
        return []

    channels = {
        c["key"]: c
        for c in (state.get("channels") or [])
        if isinstance(c, dict) and c.get("key") in interesting
    }
    if not channels:
        return []

    transcripts = root / "raw" / "transcripts"
    try:
        transcripts.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("vault transcripts dir unavailable: %s", exc)
        return []

    day = now.strftime("%Y-%m-%d")
    written: list[Path] = []
    for key, row in sorted(channels.items()):
        safe = re_sub_key(key)
        path = transcripts / f"{day}-prod-ideas-{safe}.md"
        # one file per key per day — overwrite with latest structured snapshot
        body = _vault_note_body(row, now=now)
        try:
            path.write_text(body, encoding="utf-8")
            written.append(path)
        except OSError as exc:
            logger.warning("vault note write failed for %s: %s", key, exc)
    return written


def re_sub_key(key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in key)[:80]


def _vault_note_body(row: dict[str, Any], *, now: datetime) -> str:
    evidence = row.get("evidence") or {}
    # structured ops only
    safe_ev = {k: evidence[k] for k in sorted(evidence) if k in _EVIDENCE_ALLOW}
    lines = [
        "---",
        f"title: prod-ideas {row.get('key')}",
        f"date: {now.strftime('%Y-%m-%d')}",
        "project: portfolio-lab",
        "source: portfolio-lab-prod-ideas",
        f"channel_key: {row.get('key')}",
        f"status: {row.get('status')}",
        f"promote_candidate: {bool(row.get('promote_candidate'))}",
        "---",
        "",
        f"# Prod idea channel: `{row.get('key')}`",
        "",
        f"- channel: `{row.get('channel')}`",
        f"- fingerprint: `{row.get('fingerprint')}`",
        f"- status: `{row.get('status')}`",
        f"- observation_count: {row.get('observation_count')}",
        f"- promote_candidate: {row.get('promote_candidate')}",
        f"- promote_reasons: {row.get('promote_reasons')}",
        f"- first_seen: {row.get('first_seen')}",
        f"- last_seen: {row.get('last_seen')}",
        "",
        "## Evidence (structured ops only)",
        "",
        "```json",
        json.dumps(safe_ev, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def run_once(
    *,
    data_dir: Path | str | None = None,
    public_data_dir: Path | str | None = None,
    tasker_db: Path | str | None = None,
    vault_dir: Path | str | None = None,
    skip_vault: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Load SSOT → scan → apply → write machine JSON; optional sparse vault notes.

    Machine JSON always updates. Vault notes are best-effort and never fail the job.
    When ``vault_dir`` is None and ``skip_vault`` is False, attempts ``WIKI_DIR``.
    """
    now = now or _now()
    data = Path(data_dir) if data_dir is not None else DATA_DIR
    out_path = data / CHANNEL_SSOT_NAME

    snap = load_ssot_snapshot(
        data_dir=data,
        public_data_dir=public_data_dir,
        tasker_db=tasker_db,
        now=now,
    )
    obs = scan_ssot(snap)
    prior = read_channel_ssot(out_path)
    state, events = apply_channel_delta(prior, obs, now=now)
    write_channel_ssot(out_path, state, now=now)

    vault_written: list[Path] = []
    vault_error: str | None = None
    if not skip_vault:
        try:
            resolved_vault = vault_dir
            if resolved_vault is None:
                try:
                    from src.paths import WIKI_DIR

                    resolved_vault = WIKI_DIR
                except Exception:  # noqa: BLE001 — vault optional
                    resolved_vault = None
            if resolved_vault is not None:
                vault_written = maybe_write_sparse_vault_note(
                    events, state, vault_dir=resolved_vault, now=now
                )
        except OSError as exc:
            vault_error = str(exc)
            logger.warning("sparse vault write skipped: %s", exc)

    return {
        "ok": True,
        "path": str(out_path),
        "channels_open": state.get("open_count", 0),
        "observations": len(obs),
        "events": events,
        "promote_candidates": state.get("promote_candidates") or [],
        "vault_notes_written": len(vault_written),
        "vault_error": vault_error,
        "generated_at": state.get("generated_at"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan ops SSOT into data/prod_idea_channels.json (badge-only promote)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data/ directory (default: src.paths.DATA_DIR)",
    )
    parser.add_argument(
        "--public-data-dir",
        type=Path,
        default=None,
        help="Override public data dir (default: resolve_runtime_public_data_dir)",
    )
    parser.add_argument(
        "--tasker-db",
        type=Path,
        default=None,
        help="Optional path to tasker.db for consecutive_failures",
    )
    parser.add_argument(
        "--vault-dir",
        type=Path,
        default=None,
        help="Optional SkillWiki vault root for sparse raw/transcripts notes",
    )
    parser.add_argument(
        "--no-vault",
        action="store_true",
        help="Skip vault notes even if WIKI_DIR is available",
    )
    parser.add_argument("--json", action="store_true", help="Print result JSON to stdout")
    args = parser.parse_args(argv)

    # ML off by convention for this ops job
    os.environ.setdefault("PORTFOLIO_LAB_ENABLE_ML", "0")

    result = run_once(
        data_dir=args.data_dir,
        public_data_dir=args.public_data_dir,
        tasker_db=args.tasker_db,
        vault_dir=args.vault_dir,
        skip_vault=bool(args.no_vault),
    )

    open_n = result.get("channels_open", 0)
    print(
        f"prod-ideas: open={open_n} observations={result.get('observations')} "
        f"promote={len(result.get('promote_candidates') or [])} "
        f"vault_notes={result.get('vault_notes_written')} path={result.get('path')}"
    )
    if args.json:
        summary = {k: v for k, v in result.items() if k != "events"}
        summary["event_types"] = [e.get("type") for e in (result.get("events") or [])]
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    from src.utils.log_config import configure_logging

    configure_logging()
    sys.exit(main())
