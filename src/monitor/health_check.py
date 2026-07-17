"""
Health check for portfolio-lab system.

Produces a JSON health report that can be served by the dashboard
or polled by uptime monitoring tools.

Usage::

    python -m src.monitor.health_check

Environment variables
---------------------
HEALTH_CHECK_PATH : str
    Output path for monitor health.json (default: DATA_DIR/health.json)
HEALTH_OPS_PATH : str
    Optional explicit path for PUBLIC health_ops.json (default:
    PUBLIC_DATA_DIR/health_ops.json)
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.paths import DATA_DIR, PUBLIC_DATA_DIR
from src.monitor.alerting import AlertChannel, AlertLevel, send_alert
from src.monitor.hermes_cron import (
    combine_scheduler_backends,
    load_hermes_portfolio_cron_jobs,
    load_local_cron_jobs,
    resolve_hermes_cron_jobs_path,
)

logger = logging.getLogger(__name__)

__all__ = ["run_health_check", "check_scheduler_drift", "publish_ops_health_surfaces"]

HEALTH_PATH = Path(os.environ.get("HEALTH_CHECK_PATH", str(DATA_DIR / "health.json")))
_DEFAULT_DATA_DIR = DATA_DIR
SCHEDULER_DRIFT_THRESHOLD = 2


def health_ops_path() -> Path:
    """Operator-facing monitor health under PUBLIC_DATA_DIR (dual-doc SSOT)."""
    override = os.environ.get("HEALTH_OPS_PATH")
    if override and override.strip():
        return Path(override.strip())
    return Path(PUBLIC_DATA_DIR) / "health_ops.json"


def _project_public_kill_fields(report: dict[str, Any]) -> dict[str, Any]:
    """Map monitor report kill/open-incident checks into dashboard-shaped fields."""
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    kill = checks.get("kill_switch") if isinstance(checks.get("kill_switch"), dict) else {}
    open_inc = (
        checks.get("open_incidents")
        if isinstance(checks.get("open_incidents"), dict)
        else {}
    )
    status = str(report.get("status") or "ok")
    return {
        "kill_switch": kill,
        "open_incidents": open_inc,
        "ops_health_status": status,
        "ops_health_timestamp": report.get("timestamp"),
        "ops_health_scope": report.get("scope") or "operational_readiness",
    }


def _elevate_public_system_status(current: Any, ops_status: str) -> str:
    """Raise dashboard system_status when ops monitor is more severe."""
    rank = {
        "healthy": 0,
        "ok": 0,
        "warning": 1,
        "degraded": 2,
        "critical": 3,
        "error": 3,
    }
    cur = str(current or "healthy")
    target = max(rank.get(cur, 0), rank.get(ops_status, 0))
    for name, value in rank.items():
        if value == target and name not in {"ok", "error"}:
            return name
    if target >= 3:
        return "critical"
    if target >= 2:
        return "degraded"
    if target >= 1:
        return "warning"
    return cur if cur else "healthy"


def publish_ops_health_surfaces(report: dict[str, Any]) -> None:
    """Write monitor health to PUBLIC_DATA_DIR and merge kill into public health.json.

    Dual-path honesty:
    - Always write ``health_ops.json`` (monitor schema) under PUBLIC_DATA_DIR.
    - If dashboard ``health.json`` already exists, merge kill_switch /
      open_incidents / elevated system_status so operators see halt without
      waiting for the dashboard generator cycle.
    """
    projected = _project_public_kill_fields(report)
    ops_path = health_ops_path()
    try:
        ops_path.parent.mkdir(parents=True, exist_ok=True)
        ops_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Ops health written to %s", ops_path)
    except OSError as exc:
        logger.warning("Failed to write ops health at %s: %s", ops_path, exc)

    public_health = Path(PUBLIC_DATA_DIR) / "health.json"
    if not public_health.exists():
        return
    try:
        payload = json.loads(public_health.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Public health.json unreadable; skip merge: %s", exc)
        return
    if not isinstance(payload, dict):
        return

    # Do not replace full dashboard schema; merge authority dimensions only.
    if isinstance(projected.get("kill_switch"), dict) and projected["kill_switch"]:
        payload["kill_switch"] = projected["kill_switch"]
    if isinstance(projected.get("open_incidents"), dict) and projected["open_incidents"]:
        payload["open_incidents"] = projected["open_incidents"]
    ops_status = str(projected.get("ops_health_status") or report.get("status") or "ok")
    if "system_status" in payload:
        payload["system_status"] = _elevate_public_system_status(
            payload.get("system_status"), ops_status
        )
    payload["ops_health_status"] = ops_status
    payload["ops_health_timestamp"] = projected.get("ops_health_timestamp")
    payload["ops_health_source"] = "monitor.health_check"
    try:
        public_health.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Merged ops kill authority into %s", public_health)
    except OSError as exc:
        logger.warning("Failed to merge ops health into %s: %s", public_health, exc)


def _should_include_hermes_audit(local_backend: dict) -> bool:
    """Return true when Hermes should be surfaced alongside tasker health."""
    if os.environ.get("TASKER_INCLUDE_HERMES_AUDIT") == "1":
        return True
    if local_backend.get("backend") == "tasker" and os.environ.get("CRON_BACKEND") == "tasker":
        return False
    return True


def _scheduler_drift_state_path() -> Path:
    """Resolve scheduler drift state relative to the active data directory."""
    return DATA_DIR / "scheduler_drift_state.json"


def _load_scheduler_drift_state(path: Path) -> dict[str, Any]:
    """Load prior scheduler drift state, tolerating missing or malformed state."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read scheduler drift state: %s", exc)
        return {}
    return data if isinstance(data, dict) else {}


def _save_scheduler_drift_state(path: Path, state: dict[str, Any]) -> None:
    """Persist scheduler drift state without blocking health report generation."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
    except OSError as exc:
        logger.warning("Failed to write scheduler drift state: %s", exc)


def _backend_participates_in_drift(backend: dict[str, Any]) -> bool:
    """Return True when a backend should count toward multi-backend drift.

    Idle empty schedulers (e.g. Hermes with zero portfolio-lab jobs after
    migration to tasker) report status=ok while the active backend may be
    degraded due to failed jobs. That is job-health noise, not dual-backend
    schedule drift — exclude zero-job backends from the comparison set.
    """
    try:
        total_jobs = int(backend.get("total_jobs") or 0)
    except (TypeError, ValueError):
        total_jobs = 0
    if total_jobs > 0:
        return True
    # Non-empty backends with explicit error still participate (misconfig).
    status = str(backend.get("status", "unknown")).lower()
    return status in {"error", "unavailable"} and bool(backend.get("reason"))


def check_scheduler_drift(
    backends: dict[str, dict[str, Any]],
    *,
    state_path: Path | None = None,
    threshold: int = SCHEDULER_DRIFT_THRESHOLD,
) -> dict[str, Any]:
    """Detect persistent disagreement between scheduler backend health states."""
    path = state_path or _scheduler_drift_state_path()
    backend_statuses = {
        str(name): str(backend.get("status", "unknown"))
        for name, backend in backends.items()
        if isinstance(backend, dict)
    }
    compared_statuses = {
        str(name): str(backend.get("status", "unknown"))
        for name, backend in backends.items()
        if isinstance(backend, dict) and _backend_participates_in_drift(backend)
    }
    # Drift requires ≥2 active/participating backends with disagreeing status.
    mismatch = len(compared_statuses) >= 2 and len(set(compared_statuses.values())) > 1
    previous_state = _load_scheduler_drift_state(path)
    previous_count = int(previous_state.get("consecutive_mismatches") or 0)
    consecutive_mismatches = previous_count + 1 if mismatch else 0
    status = "critical" if mismatch and consecutive_mismatches >= threshold else "warning" if mismatch else "ok"
    details = {
        "status": status,
        "mismatch": mismatch,
        "consecutive_mismatches": consecutive_mismatches,
        "threshold": threshold,
        "backend_statuses": backend_statuses,
        "compared_backend_statuses": compared_statuses,
    }

    if mismatch and consecutive_mismatches >= threshold:
        send_alert(
            AlertChannel.CRON_FAILURE,
            AlertLevel.HALT,
            f"Scheduler backend drift persisted for {consecutive_mismatches} checks",
            details=details,
        )
    elif not mismatch and previous_count > 0:
        send_alert(
            AlertChannel.CRON_FAILURE,
            AlertLevel.PASS,
            "Scheduler backends agree after drift",
            details=details,
        )

    if mismatch or previous_count > 0:
        _save_scheduler_drift_state(
            path,
            {
                **details,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    return details


def _check_data_freshness() -> dict:
    """Check how fresh the price data and signal data are."""
    checks = {}
    now = datetime.now(timezone.utc)

    # Price data freshness
    prices_path = PUBLIC_DATA_DIR / "prices.json"
    if prices_path.exists():
        mtime = datetime.fromtimestamp(prices_path.stat().st_mtime, tz=timezone.utc)
        age_hours = (now - mtime).total_seconds() / 3600
        checks["prices"] = {
            "status": "ok" if age_hours < 24 else "stale",
            "age_hours": round(age_hours, 1),
            "last_updated": mtime.isoformat(),
        }
    else:
        checks["prices"] = {"status": "missing", "age_hours": None, "last_updated": None}

    # Signal data freshness
    signals_path = PUBLIC_DATA_DIR / "signals.json"
    if signals_path.exists():
        mtime = datetime.fromtimestamp(signals_path.stat().st_mtime, tz=timezone.utc)
        age_hours = (now - mtime).total_seconds() / 3600
        checks["signals"] = {
            "status": "ok" if age_hours < 4 else "stale",
            "age_hours": round(age_hours, 1),
            "last_updated": mtime.isoformat(),
        }
    else:
        checks["signals"] = {"status": "missing", "age_hours": None, "last_updated": None}

    # Cron status
    cron_path = DATA_DIR / "cron_status.json"
    local_jobs, local_backend = load_local_cron_jobs(cron_path)
    scheduler_backends = {str(local_backend.get("backend", "local")): local_backend}
    jobs = list(local_jobs)

    hermes_path = None
    if _should_include_hermes_audit(local_backend):
        hermes_path = resolve_hermes_cron_jobs_path(
            current_data_dir=DATA_DIR,
            default_data_dir=_DEFAULT_DATA_DIR,
        )
    if hermes_path is not None:
        hermes_jobs, hermes_backend = load_hermes_portfolio_cron_jobs(hermes_path)
        scheduler_backends["hermes"] = hermes_backend
        jobs.extend(hermes_jobs)

    scheduler_status = combine_scheduler_backends(scheduler_backends)
    scheduler_drift = check_scheduler_drift(scheduler_status["backends"])
    failed = [job for job in jobs if job.get("status") == "error"]
    backend_error = any(backend.get("status") == "error" for backend in scheduler_backends.values())
    if scheduler_drift["status"] == "critical":
        cron_status = "error"
    elif backend_error:
        cron_status = "error"
    elif local_backend.get("status") == "unavailable" and len(scheduler_backends) == 1:
        cron_status = "missing"
    elif scheduler_status["status"] in {"unavailable", "warning"}:
        cron_status = "warning"
    else:
        cron_status = scheduler_status["status"]
    checks["cron"] = {
        "status": cron_status,
        "total_jobs": len(jobs),
        "failed_jobs": len(failed),
        "backends": scheduler_status["backends"],
        "jobs": jobs,
        "scheduler_drift": scheduler_drift,
    }

    return checks


def _check_circuit_breaker() -> dict:
    """Check broker circuit breaker state."""
    try:
        from src.broker.circuit_breaker import get_circuit_state
        state = get_circuit_state()
        return {
            "status": "ok" if state["state"] == "closed" else "degraded",
            "state": state["state"],
            "fail_count": state["fail_count"],
            "reset_timeout": state["reset_timeout"],
        }
    except ImportError:
        return {"status": "unavailable", "state": None, "fail_count": None, "reset_timeout": None}


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


def run_health_check() -> dict:
    """Run all health checks and return a structured report."""
    freshness = _check_data_freshness()
    circuit = _check_circuit_breaker()
    kill_switch = _check_kill_switch()
    open_incidents = _check_open_incidents()
    fred_md_cache = _check_fred_md_cache()
    freshness["fred_md_cache"] = fred_md_cache
    try:
        from src.monitor.fred_readiness import assess_fred_readiness

        freshness["fred_readiness"] = assess_fred_readiness(fred_md_cache)
    except ImportError as exc:
        freshness["fred_readiness"] = {
            "status": "warning",
            "readiness": "unknown",
            "ready": True,
            "blocking": False,
            "reason": "readiness_check_unavailable",
            "message": f"FRED readiness check unavailable: {exc}",
            "remediation": "Verify src.monitor.fred_readiness is importable.",
        }
    # Flatten nested freshness statuses for rollup while keeping nested shape
    # in the report. Critical/warning FRED readiness must still elevate.
    rollup_checks = {
        **{k: v for k, v in freshness.items() if isinstance(v, dict) and "status" in v},
        "kill_switch": kill_switch,
        "open_incidents": open_incidents,
    }
    system_status = _compute_system_status(rollup_checks, circuit)

    report = {
        "status": system_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "data_freshness": freshness,
            "circuit_breaker": circuit,
            "kill_switch": kill_switch,
            "open_incidents": open_incidents,
        },
        "service": "portfolio-lab",
        "scope": "operational_readiness",
    }

    # Always persist full checks (including kill_switch / open_incidents) so
    # on-disk data/health.json matches live run_health_check() output.
    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(HEALTH_PATH, "w") as f:
            json.dump(report, f, indent=2)
            f.flush()
        # Post-write integrity: re-read and confirm kill dimension survived.
        try:
            on_disk = json.loads(HEALTH_PATH.read_text(encoding="utf-8"))
            disk_checks = on_disk.get("checks") if isinstance(on_disk, dict) else None
            if not isinstance(disk_checks, dict) or "kill_switch" not in disk_checks:
                logger.error(
                    "Health check write missing kill_switch checks at %s", HEALTH_PATH
                )
            elif kill_switch.get("enabled") and not disk_checks.get("kill_switch", {}).get("enabled"):
                logger.error(
                    "Health check write lost kill_switch.enabled at %s", HEALTH_PATH
                )
        except (OSError, json.JSONDecodeError) as verify_exc:
            logger.error("Health check post-write verify failed: %s", verify_exc)
        logger.info("Health check: %s (written to %s)", system_status, HEALTH_PATH)
    except OSError as e:
        logger.error("Failed to write health check: %s", e)

    # Dual-path: also publish to PUBLIC_DATA_DIR so operator WWW is not stuck on
    # a stale dashboard health.json timestamp for kill authority.
    try:
        publish_ops_health_surfaces(report)
    except Exception as exc:  # noqa: BLE001 — never fail health job on public side publish
        logger.warning("Ops health surface publish failed: %s", exc)

    return report


def main():
    from src.utils.log_config import configure_logging
    configure_logging()
    report = run_health_check()
    logger.info("Health check: %s", json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    exit(main())
