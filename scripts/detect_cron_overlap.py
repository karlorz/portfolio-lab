#!/usr/bin/env python3
"""Detect overlapping cron jobs between Hermes and system crontab.

This script checks that no portfolio-lab job is scheduled by both Hermes cron
and system crontab. It reads:
  - `crontab -l` (system crontab)
  - `hermes cron list` (human-readable output)  [fallback: cron_status.json]
  - `data/cron_status.json` (backend field)

It returns exit code 0 if no overlap, 1 if overlap detected.
"""
import subprocess
import re
import json
import sys
from pathlib import Path
from typing import Set

# Mapping from Make target to expected job name (as used in Hermes cron)
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

def get_crontab_jobs() -> Set[str]:
    """Extract active (non-commented) portfolio-lab job names from crontab."""
    try:
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return set()
    jobs = set()
    for line in result.stdout.splitlines():
        # Skip comments and empty lines
        if line.strip().startswith("#") or not line.strip():
            continue
        # Look for make -C /root/projects/portfolio-lab <target>
        match = re.search(r"make -C /root/projects/portfolio-lab (\w[\w-]*)", line)
        if match:
            target = match.group(1)
            if target in MAKE_TO_JOB:
                jobs.add(MAKE_TO_JOB[target])
    return jobs

def get_hermes_jobs() -> Set[str]:
    """Extract portfolio-lab job names from hermes cron list (human-readable)."""
    try:
        result = subprocess.run(["hermes", "cron", "list"], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to cron_status.json backend field
        return get_hermes_jobs_from_status()
    jobs = set()
    for line in result.stdout.splitlines():
        # Lines like "    Name:      portfolio-lab-data"
        match = re.match(r"\s+Name:\s+(portfolio-lab-\S+)", line)
        if match:
            jobs.add(match.group(1))
    # Filter only portfolio-lab jobs
    return {j for j in jobs if j.startswith("portfolio-lab-")}

def get_hermes_jobs_from_status() -> Set[str]:
    """Fallback: infer Hermes jobs from cron_status.json backend='hermes'."""
    status_path = Path("data/cron_status.json")
    if not status_path.exists():
        return set()
    with open(status_path) as f:
        data = json.load(f)
    jobs = set()
    for entry in data.get("jobs", []):
        if entry.get("backend") == "hermes":
            jobs.add(entry["name"])
    return jobs

def main():
    crontab_jobs = get_crontab_jobs()
    hermes_jobs = get_hermes_jobs()
    overlap = crontab_jobs & hermes_jobs
    if overlap:
        print(f"ERROR: Overlapping cron jobs detected: {overlap}")
        print("System crontab and Hermes cron both own these jobs.")
        print("Fix: comment out the crontab entries or pause Hermes jobs.")
        return 1
    else:
        print(f"OK: No overlap. Crontab owns {crontab_jobs or 'none'}, Hermes owns {hermes_jobs or 'none'}.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
