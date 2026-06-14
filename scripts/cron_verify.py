#!/usr/bin/env python3
"""Verify cron_status.json integrity for the active scheduler backend."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.cron_compat import CRON_TARGETS  # noqa: E402
from src.tasker.registry import load_task_registry  # noqa: E402


CRONTAB_TARGET_ALIASES = {
    "portfolio-lab-position-sync": "sync",
}
CRONTAB_COVERAGE_EXEMPT_TARGETS = {
    # mark-to-market is tracked in cron_status, but runs as the daily-pnl
    # prerequisite rather than as a standalone crontab entry.
    "portfolio-lab-mark-to-market",
}


def expected_jobs_for_backend(backend: str) -> set[str]:
    """Return the expected job IDs for a scheduler backend."""
    if backend == "tasker":
        return {task.id for task in load_task_registry().tasks}
    return set(CRON_TARGETS)


def _cron_target_to_make_target(job_id: str) -> str:
    """Map a portfolio-lab cron job ID to the Makefile target in crontab."""
    if job_id in CRONTAB_TARGET_ALIASES:
        return CRONTAB_TARGET_ALIASES[job_id]
    prefix = "portfolio-lab-"
    return job_id.removeprefix(prefix)


def expected_crontab_make_targets(
    cron_targets: list[str] | None = None,
) -> set[str]:
    """Return Makefile targets that should appear in the checked-in crontab."""
    source_targets = CRON_TARGETS if cron_targets is None else cron_targets
    return {
        _cron_target_to_make_target(target)
        for target in source_targets
        if target not in CRONTAB_COVERAGE_EXEMPT_TARGETS
    }


def _crontab_make_targets(crontab_text: str) -> set[str]:
    """Extract Makefile targets referenced by crontab lines or fallback comments."""
    pattern = re.compile(r"\bmake(?:\s+-C\s+\S+)?\s+([A-Za-z0-9_.-]+)")
    return {match.group(1) for match in pattern.finditer(crontab_text)}


def verify_crontab_targets(crontab_file: Path) -> int:
    """Verify checked-in crontab coverage against CRON_TARGETS."""
    if not crontab_file.exists():
        print(f"MISSING: {crontab_file}")
        return 1

    expected = expected_crontab_make_targets()
    actual = _crontab_make_targets(crontab_file.read_text(encoding="utf-8"))
    missing = expected - actual

    if missing:
        print(f"FAIL: Missing Makefile targets in {crontab_file.name}: {sorted(missing)}")
        return 1

    print(
        f"OK: {len(actual)} targets referenced, "
        f"all {len(expected)} expected crontab targets present"
    )
    return 0


def verify_status(status_file: Path) -> int:
    if not status_file.exists():
        print(f"MISSING: {status_file} - run 'make cron-reset' first")
        return 1

    data = json.loads(status_file.read_text(encoding="utf-8"))
    backend = str(data.get("backend") or os.environ.get("CRON_BACKEND", "hermes"))
    jobs = data.get("jobs", [])
    names = [str(job["name"]) for job in jobs if isinstance(job, dict) and "name" in job]
    expected = expected_jobs_for_backend(backend)
    missing = expected - set(names)
    extra = set(names) - expected

    if missing:
        print(f"FAIL: Missing jobs in {status_file.name}: {missing}")
        return 1
    if extra:
        print(f"WARN: Extra jobs in {status_file.name}: {extra}")

    print(f"OK: {len(names)} jobs tracked, all {len(expected)} expected {backend} targets present")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify cron scheduler status and crontab coverage."
    )
    parser.add_argument(
        "--crontab",
        type=Path,
        help="Verify checked-in crontab Makefile target coverage against CRON_TARGETS.",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help="Verify cron_status.json integrity; defaults to CRON_STATUS_FILE or data/cron_status.json.",
    )
    args = parser.parse_args(argv)

    if args.crontab is not None:
        return verify_crontab_targets(args.crontab)

    status_file = Path(
        args.status_file
        or os.environ.get(
            "CRON_STATUS_FILE",
            str(PROJECT_ROOT / "data" / "cron_status.json"),
        )
    )
    return verify_status(status_file)


if __name__ == "__main__":
    raise SystemExit(main())
