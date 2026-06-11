"""
Health check for portfolio-lab system.

Produces a JSON health report that can be served by the dashboard
or polled by uptime monitoring tools.

Usage::

    python -m src.monitor.health_check

Environment variables
---------------------
HEALTH_CHECK_PATH : str
    Output path for health.json (default: DATA_DIR/health.json)
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from src.paths import DATA_DIR, PUBLIC_DATA_DIR
from src.monitor.hermes_cron import (
    combine_scheduler_backends,
    load_hermes_portfolio_cron_jobs,
    load_local_cron_jobs,
    resolve_hermes_cron_jobs_path,
)

logger = logging.getLogger(__name__)

__all__ = ["run_health_check"]

HEALTH_PATH = Path(os.environ.get("HEALTH_CHECK_PATH", str(DATA_DIR / "health.json")))
_DEFAULT_DATA_DIR = DATA_DIR


def _should_include_hermes_audit(local_backend: dict) -> bool:
    """Return true when Hermes should be surfaced alongside tasker health."""
    if os.environ.get("TASKER_INCLUDE_HERMES_AUDIT") == "1":
        return True
    if local_backend.get("backend") == "tasker" and os.environ.get("CRON_BACKEND") == "tasker":
        return False
    return True


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
    failed = [job for job in jobs if job.get("status") == "error"]
    backend_error = any(backend.get("status") == "error" for backend in scheduler_backends.values())
    if backend_error:
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


def _compute_system_status(checks: dict, circuit: dict) -> str:
    """Derive overall system status from component checks."""
    statuses = []

    for name, check in checks.items():
        statuses.append(check.get("status", "unknown"))

    statuses.append(circuit.get("status", "unknown"))

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
    fred_md_cache = _check_fred_md_cache()
    freshness["fred_md_cache"] = fred_md_cache
    system_status = _compute_system_status(freshness, circuit)

    report = {
        "status": system_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "data_freshness": freshness,
            "circuit_breaker": circuit,
        },
        "service": "portfolio-lab",
    }

    # Write to disk
    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(HEALTH_PATH, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Health check: %s (written to %s)", system_status, HEALTH_PATH)
    except OSError as e:
        logger.error("Failed to write health check: %s", e)

    return report


def main():
    from src.utils.log_config import configure_logging
    configure_logging()
    report = run_health_check()
    logger.info("Health check: %s", json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    exit(main())
