#!/usr/bin/env python3
"""Prune per-run tasker logs in ``data/tasker_logs/`` to a per-task retention.

Bounds run-log growth by keeping the newest ``--keep`` runs per task and
deleting older run-log files + their ``task_runs`` DB rows in lockstep.

Also bounds the tee-appended ``data/*.log`` files (cron/dashboard/eval/...)
via copytruncate rotation once they exceed a size threshold, and removes
dead frozen logs (Hermes-era ``health.log`` + pre-tasker
``unified_dashboard/overlay_*/daily_pnl`` artifacts) when they are stale.

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
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.paths import DATA_DIR  # noqa: E402
from src.tasker.store import TaskerStore  # noqa: E402

# Tee-appended logs (Makefile `tee -a $(DATA_DIR)/*.log` sites): the only
# writers are per-run tee processes. Rotation is size-based only — files that
# never cross the threshold (eval/research/build/wiki_sync/position_sync/
# daily_brief.log) are covered automatically when they do; no special-casing.
ROTATE_LOG_NAMES = (
    "cron.log",
    "dashboard.log",
    "eval.log",
    "research.log",
    "wiki_sync.log",
    "build.log",
    "position_sync.log",
    "attribution.log",
    "adaptive_weights.log",
    "daily_brief.log",
)
ROTATE_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50MB; first run halves cron/dashboard.log
ROTATE_KEEP = 3  # .1 newest ... .3 oldest; oldest copy dropped each rotation

# Dead frozen logs: writers stopped (Hermes-era health.log at the 2026-06-10
# tasker cutover; unified_dashboard/overlay_*/daily_pnl frozen since
# 2026-07-18, output now captured to tasker_logs). The guard stays
# mtime-based (STALE_DAYS) — never delete by name alone: the underlying jobs
# still run cf=0, and a future Makefile change that re-tees to these files
# must be skipped until the file goes stale again.
STALE_DAYS = 7
DEAD_LOG_NAMES = (
    "health.log",
    "unified_dashboard.log",
    "overlay_dashboard.log",
    "overlay_signals.log",
    "daily_pnl.log",
)


def _rotate_oversized_log(path: Path, dry_run: bool) -> dict:
    """Rotate one tee-appended log once it exceeds ROTATE_THRESHOLD_BYTES.

    Copytruncate, not rename: tee -a writers hold per-run open fds and a
    long-running job (eval up to 600s) can overlap the 03:03Z prune run —
    renaming would orphan the open fd and lose that run's tail. Copying the
    current file to ``.1`` and truncating in place is fd-agnostic; worst case
    is ms-level loss between copy and truncate (~1x/50 days/file at ~1MB/day).
    """
    if not path.exists():
        return {"path": str(path), "action": "skip", "bytes": 0, "reason": "absent"}
    size = path.stat().st_size
    if size <= ROTATE_THRESHOLD_BYTES:
        return {
            "path": str(path),
            "action": "skip",
            "bytes": size,
            "reason": f"{size}B <= {ROTATE_THRESHOLD_BYTES}B threshold",
        }
    if dry_run:
        return {
            "path": str(path),
            "action": "would_rotate",
            "bytes": size,
            "reason": f">{ROTATE_THRESHOLD_BYTES}B -> copy to .1, truncate in place (keep {ROTATE_KEEP})",
        }
    # Cap the chain at ROTATE_KEEP copies: drop the oldest, shift the rest.
    path.with_name(f"{path.name}.{ROTATE_KEEP}").unlink(missing_ok=True)
    for i in range(ROTATE_KEEP - 1, 0, -1):
        src = path.with_name(f"{path.name}.{i}")
        if src.exists():
            src.replace(path.with_name(f"{path.name}.{i + 1}"))
    shutil.copyfile(path, path.with_name(f"{path.name}.1"))
    path.write_bytes(b"")  # truncate in place (copytruncate)
    return {
        "path": str(path),
        "action": "rotated",
        "bytes": size,
        "reason": f"copytruncate -> {path.name}.1, original truncated (keep {ROTATE_KEEP})",
    }


def _should_delete_dead_log(path: Path) -> tuple[bool, str]:
    """Return (should_delete, reason). Only delete if confirmed stale/dead."""
    if not path.exists():
        return False, "absent"
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = datetime.now(timezone.utc) - mtime
    if age < timedelta(days=STALE_DAYS):
        return False, f"mtime {mtime.isoformat()} is {age.days}d old (< {STALE_DAYS}d) — may be live"
    return True, f"mtime {mtime.isoformat()} is {age.days}d old (>= {STALE_DAYS}d stale, dead frozen log)"


def _delete_dead_log(path: Path, dry_run: bool) -> dict:
    should, reason = _should_delete_dead_log(path)
    present = path.exists()
    size = path.stat().st_size if present else 0
    if not should:
        return {"path": str(path), "action": "skip", "bytes": 0, "reason": reason, "present": present}
    if dry_run:
        return {"path": str(path), "action": "would_delete", "bytes": size, "reason": reason, "present": present}
    path.unlink(missing_ok=True)
    return {"path": str(path), "action": "deleted", "bytes": size, "reason": reason, "present": present}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", type=int, default=20, help="run-log files to keep per task (default: 20)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without deleting anything")
    parser.add_argument(
        "--delete-dead-health-log",
        action="store_true",
        help="also remove stale dead logs (health.log, unified_dashboard/overlay_*/daily_pnl)",
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

    # tee-appended log rotation — separate concern from tasker_logs/ +
    # task_runs rows (TaskerStore.prune_runs stays the sole owner of those).
    for entry in (_rotate_oversized_log(DATA_DIR / name, args.dry_run) for name in ROTATE_LOG_NAMES):
        if entry["action"] != "skip":
            print(f"rotation {Path(entry['path']).name}: {entry['action']} ({entry['bytes']} bytes) — {entry['reason']}")

    dead_entries = []
    if args.delete_dead_health_log:
        for name in DEAD_LOG_NAMES:
            entry = _delete_dead_log(DATA_DIR / name, args.dry_run)
            if entry["present"]:
                print(f"{Path(entry['path']).name}: {entry['action']} ({entry['bytes']} bytes) — {entry['reason']}")
            dead_entries.append(entry)

    # mirror status into cron_status.json so the job is observable
    _record_cron_status(args, summary, dead_entries, started)
    return 0


def _record_cron_status(args, summary, dead, started) -> None:
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
