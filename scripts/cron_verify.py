#!/usr/bin/env python3
"""Verify cron_status.json integrity for the active scheduler backend."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.cron_compat import CRON_TARGETS  # noqa: E402
from src.tasker.registry import load_task_registry  # noqa: E402


def expected_jobs_for_backend(backend: str) -> set[str]:
    """Return the expected job IDs for a scheduler backend."""
    if backend == "tasker":
        return {task.id for task in load_task_registry().tasks}
    return set(CRON_TARGETS)


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


def main() -> int:
    status_file = Path(
        os.environ.get(
            "CRON_STATUS_FILE",
            str(PROJECT_ROOT / "data" / "cron_status.json"),
        )
    )
    return verify_status(status_file)


if __name__ == "__main__":
    raise SystemExit(main())
