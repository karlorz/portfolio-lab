#!/usr/bin/env python3
"""Detect duplicate ownership among tasker, system crontab, and Hermes cron."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CRON_STATUS = PROJECT_ROOT / "data" / "cron_status.json"

MAKE_TO_JOB = {
    "data": "portfolio-lab-data",
    "dashboard": "portfolio-lab-dashboard",
    "health": "portfolio-lab-health",
    "unified-dashboard": "portfolio-lab-unified-dashboard",
    "eval": "portfolio-lab-eval",
    "research": "portfolio-lab-research",
    "wiki-sync": "portfolio-lab-wiki-sync",
    "sync": "portfolio-lab-position-sync",
    "overlay-signals": "portfolio-lab-overlay-signals",
    "overlay-dashboard": "portfolio-lab-overlay-dashboard",
    "attribution": "portfolio-lab-attribution",
    "daily-pnl": "portfolio-lab-daily-pnl",
    "garch-risk": "portfolio-lab-garch-risk",
    "build": "portfolio-lab-build",
    "prune-logs": "portfolio-lab-prune-logs",
}


def crontab_jobs_from_text(text: str) -> set[str]:
    """Return active portfolio-lab jobs from crontab text."""
    jobs: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.search(r"\bmake\s+-C\s+\S+\s+(\w[\w-]*)", line)
        if match and match.group(1) in MAKE_TO_JOB:
            jobs.add(MAKE_TO_JOB[match.group(1)])
    return jobs


def hermes_jobs_from_text(text: str) -> set[str]:
    """Return active portfolio-lab jobs from `hermes cron list` text."""
    jobs: set[str] = set()
    in_active_block = False
    for line in text.splitlines():
        block = re.match(r"\s+[a-f0-9]{12} \[(\w+)\]", line)
        if block:
            in_active_block = block.group(1) == "active"
            continue
        if not in_active_block:
            continue
        match = re.match(r"\s+Name:\s+(portfolio-lab-\S+)", line)
        if match:
            jobs.add(match.group(1))
    return jobs


def tasker_jobs_from_cron_status(path: Path | str) -> set[str]:
    """Return enabled, non-manual tasker-owned jobs from cron_status.json."""
    status_path = Path(path)
    if not status_path.exists():
        return set()
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return set()

    jobs_raw = payload.get("jobs", [])
    if not isinstance(jobs_raw, list):
        return set()

    jobs: set[str] = set()
    for row in jobs_raw:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name.startswith("portfolio-lab-"):
            continue
        if not row.get("enabled", False):
            continue
        if row.get("manual_only", False):
            continue
        # Treat missing backend as tasker when file backend is tasker (common).
        backend = str(row.get("backend") or payload.get("backend") or "tasker").lower()
        if backend != "tasker":
            continue
        jobs.add(name)
    return jobs


def find_overlap(crontab_text: str, hermes_text: str) -> set[str]:
    """Return jobs actively owned by both crontab and Hermes (legacy helper)."""
    return crontab_jobs_from_text(crontab_text) & hermes_jobs_from_text(hermes_text)


def find_multi_backend_overlaps(
    *,
    crontab_jobs: set[str],
    hermes_jobs: set[str],
    tasker_jobs: set[str],
) -> dict[str, set[str]]:
    """Pairwise ownership intersections across the three schedulers."""
    return {
        "crontab∩hermes": crontab_jobs & hermes_jobs,
        "crontab∩tasker": crontab_jobs & tasker_jobs,
        "tasker∩hermes": tasker_jobs & hermes_jobs,
    }


def any_overlap(multi: dict[str, set[str]]) -> bool:
    return any(multi.values())


def _run_command(command: Iterable[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(list(command), capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(list(command), 127, "", str(exc))


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cron_status_path = DEFAULT_CRON_STATUS
    if "--cron-status" in args:
        idx = args.index("--cron-status")
        if idx + 1 < len(args):
            cron_status_path = Path(args[idx + 1])

    crontab_result = _run_command(["crontab", "-l"])
    crontab_text = crontab_result.stdout if crontab_result.returncode == 0 else ""

    hermes_result = _run_command(["hermes", "cron", "list"])
    hermes_available = hermes_result.returncode == 0
    if not hermes_available:
        print("WARN: hermes cron list unavailable; Hermes ownership treated as empty")

    crontab_jobs = crontab_jobs_from_text(crontab_text)
    hermes_jobs = hermes_jobs_from_text(hermes_result.stdout) if hermes_available else set()
    tasker_jobs = tasker_jobs_from_cron_status(cron_status_path)

    multi = find_multi_backend_overlaps(
        crontab_jobs=crontab_jobs,
        hermes_jobs=hermes_jobs,
        tasker_jobs=tasker_jobs,
    )

    if any_overlap(multi):
        for label, jobs in multi.items():
            if jobs:
                print(f"ERROR: Overlapping cron jobs ({label}): {sorted(jobs)}")
        print(
            "Each portfolio-lab job must have exactly one active backend "
            "(tasker | crontab | hermes)."
        )
        print(
            f"Owners — crontab: {sorted(crontab_jobs) or 'none'}; "
            f"tasker: {sorted(tasker_jobs) or 'none'}; "
            f"hermes: {sorted(hermes_jobs) or 'none'}."
        )
        return 1

    print(
        "OK: No multi-backend overlap. "
        f"Crontab owns {sorted(crontab_jobs) or 'none'}, "
        f"tasker owns {sorted(tasker_jobs) or 'none'}, "
        f"Hermes owns {sorted(hermes_jobs) or 'none'}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
