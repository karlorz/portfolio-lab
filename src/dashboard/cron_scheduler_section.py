"""Cron scheduler section for health.json."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CRON_SCHEDULER_EXCEPTIONS",
    "build_cron_scheduler_section",
    "cron_scheduler_unavailable_payload",
    "refresh_public_health_cron_section",
]

CRON_SCHEDULER_EXCEPTIONS = (
    ImportError,
    OSError,
    ValueError,
    TypeError,
    RuntimeError,
)


def cron_scheduler_unavailable_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "cron_jobs": [],
        "scheduler_status": {
            "status": "unavailable",
            "backends": {},
            "error": f"Failed to load cron scheduler state: {exc}",
        },
    }


def build_cron_scheduler_section(
    *,
    cron_status_file: Path,
    resolve_hermes_path: Callable[[], Path | None] | None = None,
    log_error: Callable[[str, Exception], None] | None = None,
) -> dict[str, Any]:
    """Load cron job status from local status file and Hermes, when available.

    Aggregates jobs from both backends and combines their statuses into a
    single scheduler_status block. Loaders handle their own read errors
    gracefully (returning unavailable/error backends), so this orchestrator
    only catches import/environment failures.
    """
    try:
        from src.monitor.hermes_cron import (
            combine_scheduler_backends,
            load_hermes_portfolio_cron_jobs,
            load_local_cron_jobs,
        )

        scheduler_backends: dict[str, dict[str, Any]] = {}
        cron_jobs: list[dict[str, Any]] = []

        local_jobs, local_backend = load_local_cron_jobs(cron_status_file)
        cron_jobs.extend(local_jobs)
        scheduler_backends["local"] = local_backend

        hermes_jobs_path = resolve_hermes_path() if resolve_hermes_path else None
        if hermes_jobs_path is not None:
            hermes_jobs, hermes_backend = load_hermes_portfolio_cron_jobs(hermes_jobs_path)
            cron_jobs.extend(hermes_jobs)
            scheduler_backends["hermes"] = hermes_backend

        return {
            "cron_jobs": cron_jobs,
            "scheduler_status": combine_scheduler_backends(scheduler_backends),
        }
    except CRON_SCHEDULER_EXCEPTIONS as exc:
        if log_error:
            log_error("cron_scheduler", exc)
        else:
            logger.warning("Cron scheduler section not available: %s", exc)
        return cron_scheduler_unavailable_payload(exc)


def refresh_public_health_cron_section(
    *,
    public_health_path: Path | None = None,
    cron_status_file: Path | None = None,
    resolve_hermes_path: Callable[[], Path | None] | None = None,
) -> bool:
    """Re-read cron_status into existing public health.json after cron_update stamps.

    Data pipeline writes dashboard health *before* Makefile stamps
    portfolio-lab-data success into cron_status.json. Without a post-stamp
    refresh, health.cron_jobs keeps the previous hour's data last_run even
    though the data job just finished successfully.
    """
    import json
    from datetime import datetime, timezone

    try:
        from src.paths import DATA_DIR, PUBLIC_DATA_DIR
    except ImportError:
        return False

    health_path = Path(public_health_path) if public_health_path else Path(PUBLIC_DATA_DIR) / "health.json"
    status_file = Path(cron_status_file) if cron_status_file else Path(DATA_DIR) / "cron_status.json"
    if not health_path.exists():
        return False

    try:
        payload = json.loads(health_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False

    section = build_cron_scheduler_section(
        cron_status_file=status_file,
        resolve_hermes_path=resolve_hermes_path,
    )
    payload["cron_jobs"] = section.get("cron_jobs") or []
    payload["scheduler_status"] = section.get("scheduler_status") or {}

    # Keep compact failed_cron consistency with rollup (excludes health self-job).
    try:
        from src.monitor.hermes_cron import rollup_failed_cron_jobs

        failed = len(rollup_failed_cron_jobs(payload["cron_jobs"]))
        # Compact fields sometimes live only on signals.health; keep top-level
        # failed_cron_jobs if already present for consumers that peek health.json.
        if "failed_cron_jobs" in payload:
            payload["failed_cron_jobs"] = failed
    except Exception:  # noqa: BLE001
        pass

    # Re-derive system_status when only self-job errors remain (scheduler ok).
    try:
        from src.dashboard.health_report import derive_system_status
        from src.monitor.hermes_cron import rollup_failed_cron_jobs

        stale_count = 0
        freshness = payload.get("data_freshness")
        if isinstance(freshness, dict):
            stale_count = sum(
                1
                for item in freshness.values()
                if isinstance(item, dict) and item.get("status") not in {"fresh", "ok"}
            )
        failed_jobs = len(rollup_failed_cron_jobs(payload.get("cron_jobs") or []))
        scheduler_status = (payload.get("scheduler_status") or {}).get("status")
        slo_status = None
        slo = payload.get("data_pipeline_slo")
        if isinstance(slo, dict):
            slo_status = slo.get("status")
        backend_error = any(
            isinstance(b, dict) and b.get("status") == "error"
            for b in (payload.get("scheduler_status") or {}).get("backends", {}).values()
        )
        payload["system_status"] = derive_system_status(
            current="healthy",
            backend_error=backend_error,
            scheduler_status=scheduler_status,
            slo_status=slo_status,
            failed_jobs=failed_jobs,
            stale_count=stale_count,
        )
    except Exception:  # noqa: BLE001 — leave prior system_status
        pass

    payload["cron_section_refreshed_at"] = datetime.now(timezone.utc).isoformat()
    try:
        health_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to refresh public health cron section: %s", exc)
        return False
    logger.info("Refreshed public health cron section at %s", health_path)

    # Compact signals.health must not keep sticky scheduler_status=degraded /
    # failed_cron mismatch after post-stamp health refresh.
    try:
        from src.monitor.hermes_cron import rollup_failed_cron_jobs

        signals_path = Path(PUBLIC_DATA_DIR) / "signals.json"
        if signals_path.exists():
            signals = json.loads(signals_path.read_text(encoding="utf-8"))
            if isinstance(signals, dict):
                health = signals.get("health")
                if not isinstance(health, dict):
                    health = {}
                    signals["health"] = health
                failed = len(rollup_failed_cron_jobs(payload.get("cron_jobs") or []))
                health["failed_cron_jobs"] = failed
                new_count = len(payload.get("cron_jobs") or [])
                # Refuse inventory collapse (suite pollution / partial loader):
                # prior ≥10 → new ≤3 without a full generate is not a real shrink.
                prev_count = health.get("cron_job_count")
                if (
                    isinstance(prev_count, int)
                    and prev_count >= 10
                    and new_count <= 3
                ):
                    logger.warning(
                        "signals.health cron_job_count refuse drop %s→%s "
                        "(need full generate or honest full inventory)",
                        prev_count,
                        new_count,
                    )
                    health["cron_job_count_drop_refused"] = {
                        "previous": prev_count,
                        "attempted": new_count,
                    }
                else:
                    health["cron_job_count"] = new_count
                    health.pop("cron_job_count_drop_refused", None)
                sched = payload.get("scheduler_status") or {}
                if isinstance(sched, dict) and sched.get("status"):
                    health["scheduler_status"] = sched.get("status")
                if payload.get("system_status"):
                    health["status"] = payload.get("system_status")
                # Max-severity honesty: never leave compact healthy when
                # scheduler degraded / failed jobs / kill sticky (Batch BH).
                health = _elevate_compact_health_status(health)
                health["cron_section_refreshed_at"] = payload["cron_section_refreshed_at"]
                signals_path.write_text(json.dumps(signals, indent=2) + "\n", encoding="utf-8")
                logger.info("Refreshed signals.health cron compact fields at %s", signals_path)
    except Exception as exc:  # noqa: BLE001 — never fail health refresh on signals patch
        logger.warning("signals.health cron compact refresh failed: %s", exc)

    return True


def _elevate_compact_health_status(health: dict[str, Any]) -> dict[str, Any]:
    """Never report compact status=healthy when critical subsystems are worse.

    Max-severity rollup (SRE dashboard pattern): kill, scheduler degraded,
    or failed_cron_jobs must demote healthy/ok → warning/degraded.
    """
    if not isinstance(health, dict):
        return health
    status = str(health.get("status") or "unknown").lower()
    severity = {
        "healthy": 0,
        "ok": 0,
        "good": 0,
        "warning": 1,
        "warn": 1,
        "degraded": 2,
        "unhealthy": 3,
        "critical": 4,
        "error": 4,
    }
    rank = severity.get(status, 0)
    target = status if status in severity else "unknown"

    failed = int(health.get("failed_cron_jobs") or 0)
    sched = str(health.get("scheduler_status") or "").lower()
    kill_enabled = bool(health.get("kill_switch_enabled")) or (
        isinstance(health.get("kill_switch"), dict)
        and bool(health["kill_switch"].get("enabled"))
    )
    open_count = int(health.get("open_incidents_count") or 0)
    if isinstance(health.get("open_incidents"), dict):
        open_count = max(open_count, int(health["open_incidents"].get("open_count") or 0))

    if failed > 0 and rank < 1:
        target, rank = "warning", 1
    if sched in {"degraded", "warning", "warn"} and rank < 2:
        target, rank = "degraded", 2
    if sched in {"error", "critical", "unavailable"} and rank < 3:
        target, rank = "unhealthy", 3
    if (kill_enabled or open_count > 0) and rank < 1:
        target, rank = "warning", 1

    if target != status and severity.get(status, 0) < rank:
        health["status"] = target
        health["status_elevated_from"] = status
        health["status_elevate_reason"] = (
            f"max_severity(failed_cron={failed}, scheduler={sched or 'n/a'}, "
            f"kill={kill_enabled}, open_incidents={open_count})"
        )
    return health
