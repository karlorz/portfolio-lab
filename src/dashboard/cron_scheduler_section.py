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
        # Recompute from the ops dimensions on a healthy base. The independent
        # signal_health quality disclosure remains unchanged in the payload.
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
        from src.monitor.signal_authority import serialize_json_payload

        health_path.write_text(
            serialize_json_payload(payload, output_path=health_path, public=True),
            encoding="utf-8",
        )
        # Batch IA: public health dual-write must stay Caddy-readable
        try:
            import os

            os.chmod(health_path, 0o644)
        except OSError:
            pass
    except OSError as exc:
        logger.warning("Failed to refresh public health cron section: %s", exc)
        return False
    logger.info("Refreshed public health cron section at %s", health_path)

    # Compact signals.health must not keep sticky scheduler_status=degraded /
    # failed_cron mismatch after post-stamp health refresh.
    # Batch IA: fan-out same bytes to private + repo soft-mirror (public-only
    # write_text left priv≠www nested health after full multi-dest generate).
    try:
        from src.monitor.hermes_cron import rollup_failed_cron_jobs

        signals_path = Path(PUBLIC_DATA_DIR) / "signals.json"
        private_path = Path(DATA_DIR) / "signals.json"
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
                signals["health"] = health
                # Prefer multi-dest serialize-once when authority present.
                wrote_via_multi = False
                try:
                    from src.monitor.signal_authority import (
                        AuthorityValidationError,
                        try_write_signals_multi_dest,
                        validate_authority_payload,
                    )

                    validate_authority_payload(signals)
                    private_dest = (
                        private_path
                        if (private_path.exists() or private_path.parent.is_dir())
                        else None
                    )
                    result = try_write_signals_multi_dest(
                        signals,
                        public_path=signals_path,
                        private_path=private_dest,
                        soft_mirror_repo=True,
                    )
                    wrote_via_multi = bool(
                        result.wrote_public or result.wrote_private or result.wrote_repo
                    )
                    if result.skipped_reason:
                        logger.warning(
                            "signals cron compact multi-dest partial skip: %s",
                            result.skipped_reason,
                        )
                    if wrote_via_multi:
                        logger.info(
                            "Refreshed signals.health cron compact (multi-dest) at %s",
                            signals_path,
                        )
                except AuthorityValidationError:
                    wrote_via_multi = False
                except Exception as multi_exc:  # noqa: BLE001
                    logger.warning(
                        "signals cron compact multi-dest failed (%s); fallback write",
                        multi_exc,
                    )
                    wrote_via_multi = False
                if not wrote_via_multi:
                    # Fallback: public-only with 0644 (tests without TA fixture).
                    import os

                    from src.monitor.signal_authority import serialize_json_payload

                    signals_path.write_text(
                        serialize_json_payload(
                            signals,
                            output_path=signals_path,
                            public=True,
                        ),
                        encoding="utf-8",
                    )
                    try:
                        os.chmod(signals_path, 0o644)
                    except OSError:
                        pass
                    logger.info(
                        "Refreshed signals.health cron compact fields at %s",
                        signals_path,
                    )
    except Exception as exc:  # noqa: BLE001 — never fail health refresh on signals patch
        logger.warning("signals.health cron compact refresh failed: %s", exc)

    # Batch BI: health + signals partial patches must update index digests
    try:
        from src.dashboard.public_data_index import (
            refresh_public_data_index_after_partial_write,
        )

        refresh_public_data_index_after_partial_write(
            public_dir=health_path.parent,
            extra_paths=[health_path, Path(PUBLIC_DATA_DIR) / "signals.json"],
            reason="cron_section_refresh",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("public index refresh after cron section failed: %s", exc)

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
        reason_parts = [
            f"failed_cron={failed}",
            f"scheduler={sched or 'n/a'}",
            f"kill={kill_enabled}",
            f"open_incidents={open_count}",
        ]
        health["status_elevate_reason"] = f"max_severity({', '.join(reason_parts)})"
    return health
