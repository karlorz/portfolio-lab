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
