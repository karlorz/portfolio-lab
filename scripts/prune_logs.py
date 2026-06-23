#!/usr/bin/env python3
"""Prune per-run tasker logs in ``data/tasker_logs/`` to a per-task retention.

Bounds run-log growth by keeping the newest ``--keep`` runs per task and
deleting older run-log files + their ``task_runs`` DB rows in lockstep.
Also optionally removes the dead Hermes-era ``data/health.log`` (1.5 GB
artifact that stopped growing at the 2026-06-10 tasker cutover).

Hygiene, not a release gate: pruning never raises on a missing file.

Usage:
    python scripts/prune_logs.py                          # keep 20/task
    python scripts/prune_logs.py --keep 50
    python scripts/prune_logs.py --dry-run                 # plan only
    python scripts/prune_logs.py --delete-dead-health-log
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.paths import DATA_DIR  # noqa: E402
from src.tasker.store import TaskerStore  # noqa: E402

HEALTH_LOG = DATA_DIR / "health.log"
# health.log is the dead Hermes-era artifact (written by
# scripts/cron/portfolio-lab-health-monitor.sh via tee -a). The tasker
# backend calls `make health` directly, whose Makefile target does NOT tee,
# so the file stopped growing at the ~2026-06-10 cutover. Treat it as dead
# only if it has not been modified in STALE_DAYS — robust to the exact
# cutover minute and to any future re-enablement of the .sh wrapper.
STALE_DAYS = 7


def _should_delete_health_log(path: Path) -> tuple[bool, str]:
    """Return (should_delete, reason). Only delete if confirmed stale/dead."""
    if not path.exists():
        return False, "absent"
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = datetime.now(timezone.utc) - mtime
    if age < timedelta(days=STALE_DAYS):
        return False, f"mtime {mtime.isoformat()} is {age.days}d old (< {STALE_DAYS}d) — may be live"
    return True, f"mtime {mtime.isoformat()} is {age.days}d old (>= {STALE_DAYS}d stale, dead Hermes-era artifact)"


def _delete_health_log(path: Path, dry_run: bool) -> dict:
    should, reason = _should_delete_health_log(path)
    size = path.stat().st_size if path.exists() else 0
    if not should:
        return {"path": str(path), "action": "skip", "bytes": 0, "reason": reason}
    if dry_run:
        return {"path": str(path), "action": "would_delete", "bytes": size, "reason": reason}
    path.unlink(missing_ok=True)
    return {"path": str(path), "action": "deleted", "bytes": size, "reason": reason}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", type=int, default=20, help="run-log files to keep per task (default: 20)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without deleting anything")
    parser.add_argument(
        "--delete-dead-health-log",
        action="store_true",
        help="also remove data/health.log if it predates the 2026-06-10 tasker cutover",
    )
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    store = TaskerStore()  # production DATA_DIR / TASKER_DB / tasker_logs
    summary = store.prune_runs(keep_per_task=args.keep, dry_run=args.dry_run)

    print(f"=== Prune Logs {'(DRY RUN)' if args.dry_run else ''}: {started.isoformat()} ===")
    print(
        f"tasker_logs: keep_per_task={args.keep} | "
        f"deleted_files={summary['deleted_files']} deleted_rows={summary['deleted_rows']} "
        f"kept_files={summary['kept_files']} bytes_freed={summary['bytes_freed']} "
        f"errors={len(summary['errors'])}"
    )
    for err in summary["errors"]:
        print(f"  skipped orphan run: {err['run_id']} ({err['reason']})")
    if args.dry_run and summary["plan"]:
        print(f"  would delete {len(summary['plan'])} run-log files:")
        for entry in summary["plan"][:10]:
            print(f"    {entry['run_id']}  {entry['bytes']}B  {entry['log_path']}")
        if len(summary["plan"]) > 10:
            print(f"    ... and {len(summary['plan']) - 10} more")

    health = None
    if args.delete_dead_health_log:
        health = _delete_health_log(HEALTH_LOG, args.dry_run)
        print(f"health.log: {health['action']} ({health['bytes']} bytes) — {health['reason']}")

    # mirror status into cron_status.json so the job is observable
    _record_cron_status(args, summary, health, started)
    return 0


def _record_cron_status(args, summary, health, started) -> None:
    duration = (datetime.now(timezone.utc) - started).total_seconds()
    backend = os.environ.get("CRON_BACKEND", "tasker")
    # errors are non-fatal — pruning is hygiene. Only a crash makes this "error".
    status = "ok"
    try:
        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "cron_update.py"),
                "portfolio-lab-prune-logs",
                status,
                f"{duration:.3f}",
                backend,
            ],
            check=False,
            cwd=str(PROJECT_ROOT),
        )
    except Exception:
        # cron_status mirroring is best-effort; never fail the prune on it
        pass


if __name__ == "__main__":
    raise SystemExit(main())
