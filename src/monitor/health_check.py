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

logger = logging.getLogger(__name__)

__all__ = ["run_health_check"]

HEALTH_PATH = Path(os.environ.get("HEALTH_CHECK_PATH", str(DATA_DIR / "health.json")))


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
    if cron_path.exists():
        try:
            with open(cron_path) as f:
                cron_data = json.load(f)
            jobs = cron_data.get("jobs", [])
            failed = [j for j in jobs if j.get("status") == "error"]
            checks["cron"] = {
                "status": "ok" if not failed else "degraded",
                "total_jobs": len(jobs),
                "failed_jobs": len(failed),
            }
        except (json.JSONDecodeError, OSError):
            checks["cron"] = {"status": "error", "total_jobs": 0, "failed_jobs": 0}
    else:
        checks["cron"] = {"status": "missing", "total_jobs": 0, "failed_jobs": 0}

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


def _compute_system_status(checks: dict, circuit: dict) -> str:
    """Derive overall system status from component checks."""
    statuses = []

    for name, check in checks.items():
        statuses.append(check.get("status", "unknown"))

    statuses.append(circuit.get("status", "unknown"))

    if "error" in statuses or "missing" in statuses:
        return "degraded"
    if "stale" in statuses or "degraded" in statuses:
        return "warning"
    if all(s == "ok" for s in statuses):
        return "ok"
    return "unknown"


def run_health_check() -> dict:
    """Run all health checks and return a structured report."""
    freshness = _check_data_freshness()
    circuit = _check_circuit_breaker()
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
