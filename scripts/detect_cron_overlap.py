#!/usr/bin/env python3
"""Detect duplicate ownership between live Hermes cron and system crontab."""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Iterable


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
}


def crontab_jobs_from_text(text: str) -> set[str]:
    """Return active portfolio-lab jobs from crontab text."""
    jobs: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.search(r"make -C /root/projects/portfolio-lab (\w[\w-]*)", line)
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


def find_overlap(crontab_text: str, hermes_text: str) -> set[str]:
    """Return jobs actively owned by both schedulers."""
    return crontab_jobs_from_text(crontab_text) & hermes_jobs_from_text(hermes_text)


def _run_command(command: Iterable[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(list(command), capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(list(command), 127, "", str(exc))


def main() -> int:
    crontab_result = _run_command(["crontab", "-l"])
    crontab_text = crontab_result.stdout if crontab_result.returncode == 0 else ""

    hermes_result = _run_command(["hermes", "cron", "list"])
    if hermes_result.returncode != 0:
        print("WARN: hermes cron list unavailable; skipped live Hermes overlap check")
        return 0

    crontab_jobs = crontab_jobs_from_text(crontab_text)
    hermes_jobs = hermes_jobs_from_text(hermes_result.stdout)
    overlap = crontab_jobs & hermes_jobs

    if overlap:
        print(f"ERROR: Overlapping cron jobs detected: {sorted(overlap)}")
        print("System crontab and Hermes cron both own these jobs.")
        return 1

    print(
        "OK: No overlap. "
        f"Crontab owns {sorted(crontab_jobs) or 'none'}, "
        f"Hermes owns {sorted(hermes_jobs) or 'none'}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
