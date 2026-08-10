"""Helpers for merging Hermes cron scheduler state into health output."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.paths import DATA_DIR, PROJECT_ROOT, PUBLIC_DATA_DIR

CRON_FIELD_PREVIEW_CHARS = int(os.environ.get("CRON_FIELD_PREVIEW_CHARS", "4096"))

# The health cron job records its own exit into cron_status.json (make health →
# cron_update + tasker mirrors). Counting that row as a "failed job" makes the
# next health/dashboard SLO run self-degrade forever. Exclude from rollups only.
HEALTH_SELF_JOB_NAME = "portfolio-lab-health"

# Batch BT: sticky last_status=error after successful recovery (manual CLI,
# nested-step partial success, or out-of-band producer). SRE practice: roll
# current health from output freshness, not a single sticky failure flag.
# Grace covers producer writes before the job stamps finished_at.
RECOVERY_MTIME_GRACE_SECONDS = float(
    os.environ.get("PORTFOLIO_LAB_CRON_RECOVERY_GRACE_SECONDS", "180")
)
# Max age of proving artifact (hours) so ancient same-run files cannot clear
# a fresh error forever.
RECOVERY_MAX_ARTIFACT_AGE_HOURS = float(
    os.environ.get("PORTFOLIO_LAB_CRON_RECOVERY_MAX_AGE_HOURS", "6")
)

# job name → relative artifact basenames under data / public data dirs
JOB_RECOVERY_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "portfolio-lab-dashboard": ("health.json", "signals.json", "dashboard.json"),
    "portfolio-lab-data": ("prices.json", "prices_compact.json"),
    # Batch DT: weekly trends may be filled by manual/ops stamp before first
    # Sunday tasker fire — google_trends.json proves producer health.
    "portfolio-lab-fetch-trends": ("google_trends.json",),
}

# Max age for pending_never_run → ok when artifact proves success (weekly jobs).
# Default 8 days covers weekly schedule + weekend lag.
PENDING_ARTIFACT_MAX_AGE_HOURS = float(
    os.environ.get("PORTFOLIO_LAB_PENDING_ARTIFACT_MAX_AGE_HOURS", "192")
)


def is_health_self_job(job: Any) -> bool:
    """True when the row is the health job reporting on itself."""
    if not isinstance(job, dict):
        return False
    return str(job.get("name") or "") == HEALTH_SELF_JOB_NAME


def _parse_iso_to_utc_epoch(value: Any) -> Optional[float]:
    """Parse ISO timestamps (with/without Z, naive→UTC) to epoch seconds."""
    if value is None:
        return None
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
    return dt.timestamp()


def recovery_data_dirs(
    extra_dirs: Optional[Sequence[Path]] = None,
    *,
    include_defaults: bool = True,
) -> list[Path]:
    """Search private DATA_DIR then PUBLIC_DATA_DIR for recovery artifacts.

    When ``extra_dirs`` is provided (hermetic tests), defaults are skipped
    (``include_defaults=False``) to prevent live WWW/DATA leakage.
    """
    dirs: list[Path] = []
    if extra_dirs is not None:
        for d in extra_dirs:
            p = Path(d)
            if p not in dirs:
                dirs.append(p)
        if not include_defaults:
            return dirs
    if include_defaults:
        for candidate in (DATA_DIR, PUBLIC_DATA_DIR):
            try:
                p = Path(candidate)
            except TypeError:
                continue
            if p not in dirs:
                dirs.append(p)
    return dirs


def cron_job_artifact_recovery_evidence(
    job: Mapping[str, Any],
    *,
    data_dirs: Optional[Sequence[Path]] = None,
    now: Optional[float] = None,
    grace_seconds: Optional[float] = None,
    max_age_hours: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Return recovery evidence when sticky error is contradicted by fresh output.

    Batch BT: a job with ``status=error`` is treated as recovered for *health
    rollups* when a known producer artifact is:

    1. not older than ``last_run`` by more than ``grace_seconds`` (same-run
       write before fail, or later successful regenerate), and
    2. still fresh vs *now* (``max_age_hours``) so stale archives cannot clear
       a real outage.

    Returns None when the job is not error, has no mapping, or evidence fails.
    Never invents success for unknown jobs.
    """
    if not isinstance(job, Mapping):
        return None
    status = normalize_cron_status(job.get("status", job.get("last_status")))
    if status != "error":
        return None
    name = str(job.get("name") or "")
    artifacts = JOB_RECOVERY_ARTIFACTS.get(name)
    if not artifacts:
        return None

    last_run_ts = _parse_iso_to_utc_epoch(
        job.get("last_run") or job.get("last_run_at") or job.get("last_finished_at")
    )
    if last_run_ts is None:
        return None

    grace = (
        float(grace_seconds)
        if grace_seconds is not None
        else float(RECOVERY_MTIME_GRACE_SECONDS)
    )
    max_age_h = (
        float(max_age_hours)
        if max_age_hours is not None
        else float(RECOVERY_MAX_ARTIFACT_AGE_HOURS)
    )
    now_ts = float(now) if now is not None else datetime.now(timezone.utc).timestamp()
    max_age_s = max(max_age_h, 0.0) * 3600.0
    threshold = last_run_ts - max(grace, 0.0)

    # Explicit data_dirs → hermetic-only roots (no live WWW/DATA leak)
    if data_dirs is not None:
        roots = recovery_data_dirs(data_dirs, include_defaults=False)
    else:
        roots = recovery_data_dirs(include_defaults=True)
    for root in roots:
        for rel in artifacts:
            path = root / rel
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime < threshold:
                continue
            age_s = now_ts - mtime
            if age_s > max_age_s:
                continue
            return {
                "job": name,
                "artifact": rel,
                "artifact_path": str(path),
                "artifact_mtime": datetime.fromtimestamp(
                    mtime, tz=timezone.utc
                ).isoformat(),
                "last_run": job.get("last_run") or job.get("last_run_at"),
                "grace_seconds": grace,
                "max_age_hours": max_age_h,
                "live_authoritative": False,
                "reason": "producer_artifact_fresh_after_sticky_error",
            }
    return None


def is_sticky_cron_error_recovered(
    job: Any,
    *,
    data_dirs: Optional[Sequence[Path]] = None,
    now: Optional[float] = None,
) -> bool:
    """True when sticky status=error should not count as a current failure."""
    return cron_job_artifact_recovery_evidence(
        job if isinstance(job, Mapping) else {},
        data_dirs=data_dirs,
        now=now,
    ) is not None


def pending_job_artifact_evidence(
    job: Mapping[str, Any],
    *,
    data_dirs: Optional[Sequence[Path]] = None,
    now: Optional[float] = None,
    max_age_hours: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Batch DT: pending_never_run + fresh producer artifact → soft-success.

    Weekly jobs can sit as tasker ``status=pending`` / ``last_run=null`` while
    ops or an out-of-band fetch already wrote a valid cache (e.g. google_trends
    with _meta.fetched_at). Operators then see a false "never run" while the
    signal is live. When a mapped artifact is fresher than ``max_age_hours``
    (default 8d for weekly), return evidence to reconcile heartbeat to ok.
    """
    if not isinstance(job, Mapping):
        return None
    status = normalize_cron_status(job.get("status", job.get("last_status")))
    last_run = job.get("last_run") or job.get("last_run_at") or job.get("last_finished_at")
    if status != "pending" or last_run:
        return None
    name = str(job.get("name") or "")
    artifacts = JOB_RECOVERY_ARTIFACTS.get(name)
    if not artifacts:
        return None

    max_age_h = (
        float(max_age_hours)
        if max_age_hours is not None
        else float(PENDING_ARTIFACT_MAX_AGE_HOURS)
    )
    now_ts = float(now) if now is not None else datetime.now(timezone.utc).timestamp()
    max_age_s = max(max_age_h, 0.0) * 3600.0

    if data_dirs is not None:
        roots = recovery_data_dirs(data_dirs, include_defaults=False)
    else:
        roots = recovery_data_dirs(include_defaults=True)

    for root in roots:
        for rel in artifacts:
            path = root / rel
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            age_s = now_ts - mtime
            if age_s < 0 or age_s > max_age_s:
                continue
            # Prefer _meta.fetched_at / latest_observation when present (trends)
            meta_fetched = None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    meta = payload.get("_meta")
                    if isinstance(meta, dict):
                        meta_fetched = meta.get("fetched_at") or meta.get(
                            "latest_observation"
                        )
            except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError):
                meta_fetched = None
            return {
                "job": name,
                "artifact": rel,
                "artifact_path": str(path),
                "artifact_mtime": datetime.fromtimestamp(
                    mtime, tz=timezone.utc
                ).isoformat(),
                "artifact_age_seconds": round(age_s, 1),
                "meta_fetched_at": meta_fetched,
                "max_age_hours": max_age_h,
                "live_authoritative": False,
                "reason": "producer_artifact_fresh_while_tasker_pending_never_run",
                "policy": "Batch DT pending artifact reconcile",
            }
    return None


def rollup_failed_cron_jobs(
    jobs: list[Any],
    *,
    data_dirs: Optional[Sequence[Path]] = None,
    now: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Failed jobs that should affect health/SLO exit.

    Excludes:
    - the portfolio-lab-health self-job (sticky self-degrade loop)
    - sticky errors with fresher producer artifacts (Batch BT recovery)
    """
    failed: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        if job.get("status") != "error":
            continue
        if is_health_self_job(job):
            continue
        if is_sticky_cron_error_recovered(job, data_dirs=data_dirs, now=now):
            continue
        failed.append(job)
    return failed


def normalize_cron_status(value: Any) -> str:
    """Map scheduler-specific status strings to dashboard status values.

    Batch CI: preserve ``pending`` / ``queued`` / ``waiting`` so never-run
    weekly jobs (e.g. portfolio-lab-fetch-trends before first Sunday fire) are
    not collapsed to ``unknown``. ``unknown`` is reserved for unmapped or
    truly missing evidence — only that tier degrades backend unknown_active.
    """
    status = str(value or "").strip().lower()
    if status in {"ok", "success", "succeeded", "completed", "pass", "passed"}:
        return "ok"
    if status in {"error", "failed", "fail", "failure", "timeout", "oom", "critical"}:
        return "error"
    if status in {"disabled", "paused"}:
        return "disabled"
    # Never-run / not-yet-fired scheduled work (tasker seeds status=pending).
    # ``blocked`` = intentional no-op under kill authority (tasker
    # RUN_BLOCKED, src/tasker/models.py:16; INTENTIONAL_BLOCK_TASK_IDS) — a
    # legitimate terminal state, not missing evidence. Normalized to pending
    # so it never degrades unknown_active; heartbeat still goes overdue if a
    # blocked job stops running entirely (real failure stays visible).
    if status in {"pending", "queued", "waiting", "never_run", "not_run", "blocked"}:
        return "pending"
    return "unknown"


def normalize_cron_state(value: Any, *, enabled: Any = True, manual_only: Any = False) -> str:
    """Map scheduler state strings to dashboard state values."""
    state = str(value or "").strip().lower()
    if manual_only is True or state == "manual_only":
        return "manual_only"
    if enabled is False:
        return "paused"
    if state in {"scheduled", "paused", "running", "manual"}:
        return state
    return "scheduled"


def _schedule_display(job: dict[str, Any]) -> str:
    schedule = job.get("schedule_display") or job.get("schedule") or ""
    if isinstance(schedule, dict):
        return str(schedule.get("display") or schedule.get("expr") or "")
    return str(schedule)


def _add_bounded_text_field(normalized: dict[str, Any], field: str, value: Any) -> None:
    if value is None:
        return
    text = str(value)
    if not text:
        return
    original_length = len(text)
    truncated = original_length > CRON_FIELD_PREVIEW_CHARS
    normalized[field] = text[:CRON_FIELD_PREVIEW_CHARS] if truncated else text
    if truncated:
        normalized[f"{field}_truncated"] = True
        normalized[f"{field}_original_length"] = original_length


def estimate_schedule_period_seconds(schedule: Any) -> Optional[float]:
    """Best-effort period (seconds) from a 5-field cron expr without croniter.

    Batch CK / deep-research: schedule-aware last_success age needs a period
    so weekly jobs do not share hourly thresholds. Returns None when unknown.
    """
    text = str(schedule or "").strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) != 5:
        return None
    minute, hour, dom, _month, dow = parts

    def _star(v: str) -> bool:
        return v in {"*", "?"}

    # Weekly: day-of-week constrained (e.g. ``20 4 * * 0``)
    if not _star(dow) and _star(dom):
        return float(7 * 86400)
    # Step-hour (P3): ``*/N`` in the hour field → every N hours. Must precede
    # the daily branch or ``0 */3 * * *`` is misread as a daily schedule.
    if hour.startswith("*/") and hour[2:].isdigit():
        return float(int(hour[2:]) * 3600)
    # Fixed-hour list: ``0 0,6,12,18 * * *`` → minimum hour gap.
    if "," in hour:
        hours = sorted(int(x) for x in hour.split(",") if x.isdigit())
        if len(hours) >= 2:
            gaps = [hours[i + 1] - hours[i] for i in range(len(hours) - 1)]
            gaps.append(24 - hours[-1] + hours[0])
            return float(min(g for g in gaps if g > 0) * 3600)
    # Daily: specific hour(s), any day-of-month/week
    if not _star(hour) and _star(dom) and _star(dow):
        return float(86400)
    # Intra-hour / hourly
    if _star(hour):
        if minute.isdigit():
            return 3600.0
        if minute.startswith("*/") and minute[2:].isdigit():
            return float(int(minute[2:]) * 60)
        if "," in minute:
            mins = sorted(int(x) for x in minute.split(",") if x.isdigit())
            if len(mins) >= 2:
                gaps = [mins[i + 1] - mins[i] for i in range(len(mins) - 1)]
                gaps.append(60 - mins[-1] + mins[0])
                return float(min(g for g in gaps if g > 0) * 60)
        return 3600.0
    # Fallback: treat as daily when hour fixed
    if not _star(hour):
        return float(86400)
    return None


def schedule_aware_last_success_heartbeat(
    job: Mapping[str, Any],
    *,
    now: Optional[float] = None,
    grace_fraction: float = 0.1,
    min_grace_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Dead-man's-switch style last_success age vs schedule period (Batch CK).

    States (SRE schedule-aware SLI):
    - ``inactive`` — disabled / manual_only / paused
    - ``pending_never_run`` — no last_run yet; **not** overdue (new weekly jobs)
    - ``ok`` — last success within period + grace
    - ``overdue`` — last success older than period + grace (or error aged out)
    - ``error`` — last terminal error still inside period window
    - ``unknown`` — insufficient schedule/evidence

    ``last_success_age_seconds`` is null for pending_never_run (does not burn
    error budget); otherwise ``now - last_run`` when last_run is known.
    """
    now_ts = float(now) if now is not None else datetime.now(timezone.utc).timestamp()
    enabled = bool(job.get("enabled", True))
    manual_only = bool(job.get("manual_only", False))
    state = str(job.get("state") or "").lower()
    status = normalize_cron_status(job.get("status", job.get("last_status")))
    schedule = job.get("schedule") or job.get("schedule_display") or ""
    period = estimate_schedule_period_seconds(schedule)
    last_run_ts = _parse_iso_to_utc_epoch(
        job.get("last_run") or job.get("last_run_at") or job.get("last_finished_at")
    )

    grace = min_grace_seconds
    if period is not None:
        grace = max(float(period) * float(grace_fraction), float(min_grace_seconds))

    out: dict[str, Any] = {
        "schedule_period_seconds": period,
        "grace_seconds": round(grace, 1),
        "last_success_age_seconds": None,
        "heartbeat_state": "unknown",
        "overdue": False,
        "disclosure": (
            "schedule-aware last_success age (Batch CK): pending_never_run does "
            "not burn budget; overdue when age > period + grace"
        ),
    }

    if manual_only or not enabled or state in {"manual_only", "paused", "disabled"}:
        out["heartbeat_state"] = "inactive"
        return out

    if last_run_ts is None:
        # Never fired — weekly/sparse jobs stay pending without false overdue
        if status in {"pending", "unknown"} or status == "ok":
            out["heartbeat_state"] = "pending_never_run"
            out["overdue"] = False
            return out
        out["heartbeat_state"] = "never_run"
        return out

    age = max(0.0, now_ts - float(last_run_ts))
    out["last_success_age_seconds"] = round(age, 1)
    threshold = (float(period) + grace) if period is not None else None

    if status == "error":
        out["heartbeat_state"] = "error"
        if threshold is not None and age > threshold:
            out["overdue"] = True
            out["heartbeat_state"] = "overdue"
        return out

    if status in {"ok", "pending"}:
        if threshold is not None and age > threshold:
            out["heartbeat_state"] = "overdue"
            out["overdue"] = True
        else:
            out["heartbeat_state"] = "ok"
        return out

    if threshold is not None and age > threshold:
        out["heartbeat_state"] = "overdue"
        out["overdue"] = True
    else:
        out["heartbeat_state"] = status or "unknown"
    return out


def normalize_cron_job(
    job: dict[str, Any],
    *,
    backend: str,
    source: str,
    index: int = 0,
    now: Optional[float] = None,
) -> dict[str, Any]:
    """Normalize a local or Hermes cron row for dashboard health consumers."""
    name = str(job.get("name") or job.get("id") or f"job-{index}")
    normalized = {
        "id": str(job.get("id") or f"{backend}:{name}"),
        "name": name,
        "schedule": _schedule_display(job),
        "last_run": job.get("last_run") or job.get("last_run_at"),
        "next_run": job.get("next_run") or job.get("next_run_at"),
        "status": normalize_cron_status(job.get("status", job.get("last_status"))),
        "state": normalize_cron_state(
            job.get("state"),
            enabled=job.get("enabled", True),
            manual_only=job.get("manual_only", False),
        ),
        "enabled": bool(job.get("enabled", True)),
        "manual_only": bool(job.get("manual_only", False)),
        "backend": backend,
        "source": source,
    }
    _add_bounded_text_field(normalized, "error", job.get("error") or job.get("last_error"))
    _add_bounded_text_field(normalized, "stdout", job.get("stdout"))
    _add_bounded_text_field(normalized, "stderr", job.get("stderr"))
    if "duration_seconds" in job:
        normalized["duration_seconds"] = job["duration_seconds"]
    # Batch CK: schedule-aware last_success heartbeat SLI (primitive for dead-man)
    try:
        hb = schedule_aware_last_success_heartbeat(normalized, now=now)
        normalized["schedule_period_seconds"] = hb.get("schedule_period_seconds")
        normalized["last_success_age_seconds"] = hb.get("last_success_age_seconds")
        normalized["heartbeat_state"] = hb.get("heartbeat_state")
        normalized["heartbeat_overdue"] = bool(hb.get("overdue"))
        normalized["heartbeat_grace_seconds"] = hb.get("grace_seconds")
    except Exception:  # noqa: BLE001 — never break job normalize on SLI
        pass

    # Batch DT: pending_never_run + fresh producer artifact → soft ok
    try:
        if (
            normalized.get("status") == "pending"
            and not normalized.get("last_run")
            and normalized.get("enabled", True)
            and not normalized.get("manual_only", False)
        ):
            evidence = pending_job_artifact_evidence(normalized, now=now)
            if evidence:
                normalized["status"] = "ok"
                normalized["pending_artifact_reconciled"] = True
                normalized["pending_artifact_evidence"] = {
                    "artifact": evidence.get("artifact"),
                    "artifact_mtime": evidence.get("artifact_mtime"),
                    "meta_fetched_at": evidence.get("meta_fetched_at"),
                    "reason": evidence.get("reason"),
                }
                # Heartbeat: use artifact mtime as synthetic last success
                mtime_iso = evidence.get("artifact_mtime")
                if mtime_iso:
                    normalized["last_run"] = mtime_iso
                    normalized["last_run_source"] = "producer_artifact_mtime"
                hb2 = schedule_aware_last_success_heartbeat(normalized, now=now)
                normalized["last_success_age_seconds"] = hb2.get(
                    "last_success_age_seconds"
                )
                normalized["heartbeat_state"] = hb2.get("heartbeat_state")
                normalized["heartbeat_overdue"] = bool(hb2.get("overdue"))
                normalized["heartbeat_grace_seconds"] = hb2.get("grace_seconds")
                normalized["heartbeat_disclosure"] = (
                    "Batch DT: pending_never_run reconciled to ok via fresh "
                    f"artifact {evidence.get('artifact')}"
                )
    except Exception:  # noqa: BLE001
        pass

    # Batch IU DT1: wall-clock timeout / sticky error + fresher producer
    # artifact → soft-ok for scheduler SLO (false-timeout after successful
    # partial publish, e.g. dashboard killed at outer 120s while signals
    # already dual-written). Widen grace to the observed run duration so
    # artifacts written early in a long run still count (data: prices at
    # :05 then hang on post-steps until outer kill at :10).
    try:
        if normalized.get("status") == "error":
            grace_s: Optional[float] = None
            dur = normalized.get("duration_seconds")
            try:
                if dur is not None:
                    grace_s = float(dur) + float(RECOVERY_MTIME_GRACE_SECONDS)
            except (TypeError, ValueError):
                grace_s = None
            if grace_s is None:
                try:
                    from src.cron_compat import CRON_EXPECTED_DURATIONS

                    expected = CRON_EXPECTED_DURATIONS.get(name)
                    if expected is not None:
                        grace_s = float(expected) + float(RECOVERY_MTIME_GRACE_SECONDS)
                except Exception:  # noqa: BLE001
                    grace_s = None
            recovery = cron_job_artifact_recovery_evidence(
                normalized, now=now, grace_seconds=grace_s
            )
            if recovery:
                normalized["status"] = "ok"
                normalized["timeout_artifact_reconciled"] = True
                normalized["timeout_artifact_evidence"] = {
                    "artifact": recovery.get("artifact"),
                    "artifact_mtime": recovery.get("artifact_mtime"),
                    "reason": recovery.get("reason"),
                    "policy": "Batch IU DT1 timeout/soft-ok via fresh producer artifact",
                }
                # Prefer artifact mtime as last_success for heartbeat honesty
                mtime_iso = recovery.get("artifact_mtime")
                if mtime_iso:
                    normalized["last_run"] = mtime_iso
                    normalized["last_run_source"] = "producer_artifact_mtime_timeout_soft_ok"
                hb3 = schedule_aware_last_success_heartbeat(normalized, now=now)
                normalized["last_success_age_seconds"] = hb3.get(
                    "last_success_age_seconds"
                )
                normalized["heartbeat_state"] = hb3.get("heartbeat_state")
                normalized["heartbeat_overdue"] = bool(hb3.get("overdue"))
                normalized["heartbeat_grace_seconds"] = hb3.get("grace_seconds")
                normalized["heartbeat_disclosure"] = (
                    "Batch IU DT1: sticky timeout/error reconciled to ok via fresh "
                    f"artifact {recovery.get('artifact')}"
                )
    except Exception:  # noqa: BLE001
        pass
    return normalized


def summarize_backend(
    *,
    backend: str,
    source: str,
    jobs: list[dict[str, Any]],
    status: str | None = None,
    reason: str | None = None,
    data_dirs: Optional[Sequence[Path]] = None,
    now: Optional[float] = None,
) -> dict[str, Any]:
    """Build backend-level scheduler health metadata.

    ``failed_jobs`` excludes the portfolio-lab-health self-job so dashboard
    ``scheduler_status.backends.*.failed_jobs`` matches ``rollup_failed_cron_jobs``
    / signals.health ``failed_cron_jobs``. Counting the health job's own exit
    as a scheduler failure creates a sticky degraded loop after a single
    non-ok health run.

    Batch BT: also excludes sticky errors recovered by fresher producer
    artifacts (same logic as ``rollup_failed_cron_jobs``).
    """
    rollup_failed = rollup_failed_cron_jobs(jobs, data_dirs=data_dirs, now=now)
    failed_jobs = len(rollup_failed)
    recovered = [
        job
        for job in jobs
        if isinstance(job, dict)
        and job.get("status") == "error"
        and not is_health_self_job(job)
        and is_sticky_cron_error_recovered(job, data_dirs=data_dirs, now=now)
    ]
    # True unknown (unmapped status) still degrades. Pending never-run does not
    # — weekly / sparse jobs sit as pending until first fire without meaning
    # scheduler failure (Batch CI / deep-research last_success-age guidance).
    active_unknown_jobs = sum(
        1
        for job in jobs
        if job.get("enabled", True)
        and not job.get("manual_only", False)
        and job.get("state") not in {"manual_only", "paused"}
        and job.get("status") == "unknown"
    )
    active_pending_never_run = sum(
        1
        for job in jobs
        if job.get("enabled", True)
        and not job.get("manual_only", False)
        and job.get("state") not in {"manual_only", "paused"}
        and job.get("status") == "pending"
        and not job.get("last_run")
        # Batch DT: already-reconciled rows are status=ok with last_run set
        and not job.get("pending_artifact_reconciled")
    )
    pending_artifact_reconciled = sum(
        1
        for job in jobs
        if isinstance(job, dict) and job.get("pending_artifact_reconciled")
    )
    backend_status = status or ("degraded" if failed_jobs or active_unknown_jobs else "ok")
    summary = {
        "backend": backend,
        "status": backend_status,
        "source": source,
        "total_jobs": len(jobs),
        "failed_jobs": failed_jobs,
    }
    if recovered:
        summary["recovered_sticky_errors"] = len(recovered)
        summary["recovered_sticky_job_names"] = [
            str(j.get("name") or "") for j in recovered
        ]
    if active_unknown_jobs:
        summary["unknown_active_jobs"] = active_unknown_jobs
    if active_pending_never_run:
        summary["pending_never_run_jobs"] = active_pending_never_run
    if pending_artifact_reconciled:
        summary["pending_artifact_reconciled_jobs"] = pending_artifact_reconciled
    # Batch CK: roll up schedule-aware heartbeats (do not degrade on pending_never_run)
    overdue_jobs = [
        job
        for job in jobs
        if isinstance(job, dict)
        and job.get("enabled", True)
        and not job.get("manual_only", False)
        and job.get("state") not in {"manual_only", "paused"}
        and (
            job.get("heartbeat_overdue") is True
            or job.get("heartbeat_state") == "overdue"
        )
    ]
    if overdue_jobs:
        summary["heartbeat_overdue_jobs"] = len(overdue_jobs)
        summary["heartbeat_overdue_job_names"] = [
            str(j.get("name") or "") for j in overdue_jobs
        ]
        # Soft signal only: unknown/error still own hard degrade; overdue is advisory
        # unless there are zero failed/unknown — then surface warning-tier degrade.
        if not failed_jobs and not active_unknown_jobs and summary["status"] == "ok":
            summary["status"] = "degraded"
            summary["reason"] = summary.get("reason") or "heartbeat_overdue"
    ages = [
        float(j["last_success_age_seconds"])
        for j in jobs
        if isinstance(j, dict) and j.get("last_success_age_seconds") is not None
    ]
    if ages:
        summary["max_last_success_age_seconds"] = round(max(ages), 1)
    if reason and not summary.get("reason"):
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

    file_backend = str(cron_data.get("backend") or "").strip().lower()
    backend_name = "tasker" if file_backend == "tasker" else "local"

    jobs = [
        normalize_cron_job(job, backend=str(job.get("backend") or backend_name), source=source, index=index)
        for index, job in enumerate(cron_data.get("jobs", []))
        if isinstance(job, dict)
    ]
    return jobs, summarize_backend(backend=backend_name, source=source, jobs=jobs)


def load_hermes_portfolio_cron_jobs(
    jobs_path: Path,
    *,
    project_dir: Path = PROJECT_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read Hermes cron jobs.json and return active portfolio-lab jobs."""
    source = str(jobs_path)
    try:
        path_exists = jobs_path.exists()
    except PermissionError as exc:
        return [], summarize_backend(
            backend="hermes",
            source=source,
            jobs=[],
            status="unavailable",
            reason=f"Hermes cron jobs file is not readable: {exc}",
        )

    if not path_exists:
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
    except PermissionError as exc:
        return [], summarize_backend(
            backend="hermes",
            source=source,
            jobs=[],
            status="unavailable",
            reason=f"Hermes cron jobs file is not readable: {exc}",
        )
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
