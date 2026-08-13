"""Health-check kill-switch / incidents / system rollup cluster
(extracted from src/monitor/health_check.py, Item 5 s1).
"""

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from src.paths import DATA_DIR, PUBLIC_DATA_DIR
from src.monitor.hermes_cron import (
    is_health_self_job,
)

logger = logging.getLogger(__name__)

def _check_fred_md_cache() -> dict:
    """Check FRED-MD cache availability without making live provider calls."""
    try:
        from src.data.fred_data import get_fred_md_cache_health

        return get_fred_md_cache_health()
    except ImportError as exc:
        return {
            "status": "unavailable",
            "row_count": 0,
            "latest_fetched_at": None,
            "age_hours": None,
            "reason": str(exc),
        }


def _load_json_file(path: Path) -> dict[str, Any] | None:
    """Load a JSON object from disk; return None when missing or invalid."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _check_kill_switch(data_dir: Path | None = None) -> dict[str, Any]:
    """Bounded kill-switch dimension for operational readiness."""
    root = data_dir or DATA_DIR
    payload = _load_json_file(root / "kill_switch.json")
    if not payload:
        return {
            "status": "ok",
            "enabled": False,
            "level": None,
            "reason": None,
            "source": None,
            "message": None,
            "timestamp": None,
        }

    enabled = bool(payload.get("enabled"))
    level = str(payload.get("level") or "").lower() or None
    reason = payload.get("reason")
    source = payload.get("source")
    message = payload.get("message")
    timestamp = payload.get("timestamp")

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
        "reason": reason,
        "source": source,
        "message": message,
        "timestamp": timestamp,
        "incident_id": payload.get("incident_id"),
        "mode": payload.get("mode"),
    }


def _check_open_incidents(data_dir: Path | None = None) -> dict[str, Any]:
    """Bounded open-incident dimension for operational readiness."""
    root = data_dir or DATA_DIR
    payload = _load_json_file(root / "incidents.json")
    if not payload:
        return {
            "status": "ok",
            "open_count": 0,
            "incidents": [],
        }

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
        open_incidents.append({
            "incident_id": row.get("incident_id") or row.get("id"),
            "channel": row.get("channel"),
            "severity": row.get("severity"),
            "state": state,
            "message": row.get("message"),
            "kill_switch_level": kill_level,
        })

    open_count = int(payload.get("open_count") or len(open_incidents) or 0)
    if open_count == 0 and open_incidents:
        open_count = len(open_incidents)

    if has_halt or any(
        str(i.get("kill_switch_level") or "").lower() == "halt" for i in open_incidents
    ):
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


def _status_for_system_rollup(name: str, check: dict) -> str:
    """Map a component status into the overall rollup severity ladder.

    Nested report fields keep their raw status for operators. For rollup /
    process exit, non-blocking lab FRED advisories (ready, not blocking, or
    empty cache without a key) must not force overall ``warning`` — that made
    ``make health`` exit 1 every cycle and sticky tasker ``error`` rows.
    """
    status = str(check.get("status", "unknown"))
    if name == "fred_readiness":
        if check.get("ready") is True and check.get("blocking") is False:
            return "ok"
    if name == "fred_md_cache":
        if status == "empty" and not check.get("api_key_configured"):
            return "ok"
    return status


def _compute_system_status(checks: dict, circuit: dict) -> str:
    """Derive overall system status from component checks.

    Severity order (highest first): critical > degraded > warning > ok.
    Active kill-switch HALT / open-incident HALT use status ``critical`` so
    they cannot be understated by lower-severity freshness warnings.
    """
    statuses = []

    for name, check in checks.items():
        if isinstance(check, dict):
            statuses.append(_status_for_system_rollup(str(name), check))

    if isinstance(circuit, dict):
        statuses.append(str(circuit.get("status", "unknown")))

    if "critical" in statuses:
        return "critical"
    if "error" in statuses or "missing" in statuses:
        return "degraded"
    if (
        "stale" in statuses
        or "empty" in statuses
        or "degraded" in statuses
        or "warning" in statuses
        or "unavailable" in statuses
    ):
        return "warning"
    if all(s == "ok" for s in statuses):
        return "ok"
    return "unknown"


def attach_shared_freshness_slis_to_ops_report(
    report: dict[str, Any] | None,
    *,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Attach mirror-lag + execution-timeline SLIs to monitor health_ops (Batch EK).

    Deep-research: ops health report and compact dashboard health must share the
    same freshness metrics so operators do not see split-brain (signals.health
    shows rewrite_inflated / lagging while health_ops is silent).
    """
    from src.monitor.health_freshness_cb import project_graduation_cb_onto_report
    if not isinstance(report, dict):
        report = {}

    # --- repo public/data mirror lag (same probe as compact signals.health) ---
    try:
        from src.dashboard.generator import project_repo_public_mirror_lag_onto_health
        from src.monitor.repo_public_mirror_lag import summarize_repo_public_mirror_lag

        lag_summary = summarize_repo_public_mirror_lag()
        # Batch FX / EP: project onto the real report so soft-elevate of
        # top-level status=ok → warning under lagging/critical is not a dead path.
        report = project_repo_public_mirror_lag_onto_health(report, lag_summary)
        report["repo_public_mirror_lag"] = {
            "lagging_count": report.get("repo_public_mirror_lagging_count"),
            "total": report.get("repo_public_mirror_total"),
            "status": report.get("repo_public_mirror_lag_status"),
            "badge": report.get("repo_public_mirror_lag_badge"),
            "paths": report.get("repo_public_mirror_lagging_paths"),
            "source": report.get("repo_public_mirror_source"),
            "dest": report.get("repo_public_mirror_dest"),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("ops report mirror lag SLI skipped: %s", exc)
        report.setdefault("repo_public_mirror_lag_status", "unknown")

    # --- graduation CB SSOT re-projection (Batch EM) ---
    try:
        root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        report = project_graduation_cb_onto_report(report, data_dir=root)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ops report graduation CB SLI skipped: %s", exc)

    # --- rebalance execution timeline (disk panel, same as Batch EI) ---
    try:
        from src.dashboard.generator import project_execution_timeline_onto_health

        root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        rebalance_health_panel: dict[str, Any] | None = None
        for path in (
            root / "rebalance_health.json",
            Path(PUBLIC_DATA_DIR) / "rebalance_health.json",
        ):
            if not path.is_file():
                continue
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(loaded, dict):
                rebalance_health_panel = loaded
                break
        projected_tl: dict[str, Any] = {}
        projected_tl = project_execution_timeline_onto_health(
            projected_tl, rebalance_health_panel
        )
        for key in (
            "rebalance_execution_timeline_status",
            "rebalance_execution_timeline_badge",
            "rebalance_unique_execution_days",
            "rebalance_raw_history_entries",
            "rebalance_snapshot_rewrite_files",
            "rebalance_execution_timeline_policy",
        ):
            if key in projected_tl:
                report[key] = projected_tl[key]
        report["rebalance_execution_timeline"] = {
            "status": projected_tl.get("rebalance_execution_timeline_status"),
            "badge": projected_tl.get("rebalance_execution_timeline_badge"),
            "unique_days": projected_tl.get("rebalance_unique_execution_days"),
            "raw_entries": projected_tl.get("rebalance_raw_history_entries"),
            "rewrite_files": projected_tl.get("rebalance_snapshot_rewrite_files"),
            "source": "disk" if rebalance_health_panel is not None else "missing",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("ops report execution timeline SLI skipped: %s", exc)
        report.setdefault("rebalance_execution_timeline_status", "unknown")

    return report


def _stamp_health_self_job_running_success(freshness: dict[str, Any]) -> None:
    """Overwrite portfolio-lab-health row so a successful run does not publish prior error.

    Tasker writes cron_status *after* the job body. During ``run_health_check`` the
    self job still shows the previous terminal status. When that was ``error``, the
    successful run's own report freezes a false error row into health_ops/public
    until a later writer fixes it. Stamp in-process success for honesty; rollup
    already excludes self-errors from failed_jobs.
    """
    if not isinstance(freshness, dict):
        return
    cron = freshness.get("cron")
    if not isinstance(cron, dict):
        return
    jobs = cron.get("jobs")
    if not isinstance(jobs, list):
        return
    now = datetime.now(timezone.utc).isoformat()
    stamped = False
    for job in jobs:
        if not isinstance(job, dict):
            continue
        if not is_health_self_job(job):
            continue
        prev = job.get("status")
        job["status"] = "ok"
        job["last_run"] = now
        job["self_observation"] = "in_process_success_stamp"
        if prev not in (None, "ok", "success"):
            job["prior_status_before_stamp"] = prev
        stamped = True
    if stamped:
        # Keep embedded failed_jobs consistent with rollup (already excludes self)
        try:
            from src.monitor.hermes_cron import rollup_failed_cron_jobs

            cron["failed_jobs"] = len(rollup_failed_cron_jobs(jobs))
        except Exception:  # noqa: BLE001
            pass
