"""Helpers for merging Hermes cron scheduler state into health output."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.paths import PROJECT_ROOT


def normalize_cron_status(value: Any) -> str:
    """Map scheduler-specific status strings to dashboard status values."""
    status = str(value or "").strip().lower()
    if status in {"ok", "success", "succeeded", "completed", "pass", "passed"}:
        return "ok"
    if status in {"error", "failed", "fail", "failure", "timeout", "oom", "critical"}:
        return "error"
    return "unknown"


def normalize_cron_state(value: Any, *, enabled: Any = True) -> str:
    """Map scheduler state strings to dashboard state values."""
    if enabled is False:
        return "paused"
    state = str(value or "").strip().lower()
    if state in {"scheduled", "paused", "running"}:
        return state
    return "scheduled"


def _schedule_display(job: dict[str, Any]) -> str:
    schedule = job.get("schedule_display") or job.get("schedule") or ""
    if isinstance(schedule, dict):
        return str(schedule.get("display") or schedule.get("expr") or "")
    return str(schedule)


def normalize_cron_job(job: dict[str, Any], *, backend: str, source: str, index: int = 0) -> dict[str, Any]:
    """Normalize a local or Hermes cron row for dashboard health consumers."""
    name = str(job.get("name") or job.get("id") or f"job-{index}")
    normalized = {
        "id": str(job.get("id") or f"{backend}:{name}"),
        "name": name,
        "schedule": _schedule_display(job),
        "last_run": job.get("last_run") or job.get("last_run_at"),
        "next_run": job.get("next_run") or job.get("next_run_at"),
        "status": normalize_cron_status(job.get("status", job.get("last_status"))),
        "state": normalize_cron_state(job.get("state"), enabled=job.get("enabled", True)),
        "backend": backend,
        "source": source,
    }
    error = job.get("error") or job.get("last_error")
    if error:
        normalized["error"] = str(error)
    if "duration_seconds" in job:
        normalized["duration_seconds"] = job["duration_seconds"]
    return normalized


def summarize_backend(
    *,
    backend: str,
    source: str,
    jobs: list[dict[str, Any]],
    status: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build backend-level scheduler health metadata."""
    failed_jobs = sum(1 for job in jobs if job.get("status") == "error")
    backend_status = status or ("degraded" if failed_jobs else "ok")
    summary = {
        "backend": backend,
        "status": backend_status,
        "source": source,
        "total_jobs": len(jobs),
        "failed_jobs": failed_jobs,
    }
    if reason:
        summary["reason"] = reason
    return summary


def combine_scheduler_backends(backends: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize all scheduler backends into one status block."""
    statuses = [backend.get("status", "unknown") for backend in backends.values()]
    if any(status in {"error", "degraded"} for status in statuses):
        status = "degraded"
    elif statuses and all(status == "unavailable" for status in statuses):
        status = "unavailable"
    elif any(status == "unavailable" for status in statuses):
        status = "warning"
    elif statuses and all(status == "ok" for status in statuses):
        status = "ok"
    else:
        status = "unknown"
    return {"status": status, "backends": backends}


def resolve_hermes_cron_jobs_path(*, current_data_dir: Path, default_data_dir: Path) -> Path | None:
    """Resolve Hermes jobs.json when explicitly configured or in real project runs."""
    configured = os.environ.get("HERMES_CRON_JOBS_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    if current_data_dir != default_data_dir:
        return None
    hermes_home = Path(os.environ.get("HERMES_HOME", "/root/.hermes")).expanduser()
    return hermes_home / "cron" / "jobs.json"


def load_local_cron_jobs(status_file: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read project-local cron_status.json and normalize its rows."""
    source = str(status_file)
    if not status_file.exists():
        return [], summarize_backend(
            backend="local",
            source=source,
            jobs=[],
            status="unavailable",
            reason="cron_status.json missing; scheduler state not verified",
        )

    try:
        with status_file.open(encoding="utf-8") as handle:
            cron_data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        return [], summarize_backend(
            backend="local",
            source=source,
            jobs=[],
            status="error",
            reason=f"failed to read cron_status.json: {exc}",
        )

    jobs = [
        normalize_cron_job(job, backend="local", source=source, index=index)
        for index, job in enumerate(cron_data.get("jobs", []))
        if isinstance(job, dict)
    ]
    return jobs, summarize_backend(backend="local", source=source, jobs=jobs)


def load_hermes_portfolio_cron_jobs(
    jobs_path: Path,
    *,
    project_dir: Path = PROJECT_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read Hermes cron jobs.json and return active portfolio-lab jobs."""
    source = str(jobs_path)
    if not jobs_path.exists():
        return [], summarize_backend(
            backend="hermes",
            source=source,
            jobs=[],
            status="unavailable",
            reason="Hermes cron jobs file missing",
        )

    try:
        with jobs_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        return [], summarize_backend(
            backend="hermes",
            source=source,
            jobs=[],
            status="error",
            reason=f"failed to read Hermes cron jobs: {exc}",
        )

    jobs = []
    for index, job in enumerate(data.get("jobs", [])):
        if not isinstance(job, dict):
            continue
        name = str(job.get("name") or "")
        workdir = str(job.get("workdir") or "")
        is_portfolio_job = name.startswith("portfolio-lab") or workdir == str(project_dir)
        if not is_portfolio_job or job.get("enabled") is False:
            continue
        jobs.append(normalize_cron_job(job, backend="hermes", source=source, index=index))

    return jobs, summarize_backend(backend="hermes", source=source, jobs=jobs)
