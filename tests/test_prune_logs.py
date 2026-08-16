#!/usr/bin/env python3
"""Tests for the prune_logs CLI script."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


from scripts import prune_logs


def _seed_tasker_logs(tmp_path: Path, task_id: str = "portfolio-lab-health", n: int = 25) -> list[Path]:
    """Create n fake run-log files + a tasker.db with matching rows."""
    import sqlite3

    db_path = tmp_path / "tasker.db"
    log_dir = tmp_path / "tasker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS task_state (
            task_id TEXT PRIMARY KEY, paused INTEGER DEFAULT 0, pause_reason TEXT,
            failure_count INTEGER DEFAULT 0, consecutive_failures INTEGER DEFAULT 0,
            last_run_id TEXT, last_status TEXT, last_started_at TEXT, last_finished_at TEXT,
            last_duration_seconds REAL, last_exit_code INTEGER,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS task_runs (
            run_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, command_json TEXT NOT NULL,
            trigger TEXT NOT NULL, retry_of TEXT, status TEXT NOT NULL, pid INTEGER,
            started_at TEXT, finished_at TEXT, duration_seconds REAL, exit_code INTEGER,
            error TEXT, log_path TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        """
    )
    base = datetime(2026, 6, 20, 12, 0, 0, tzinfo=timezone.utc)
    paths = []
    for i in range(n):
        ts = (base + timedelta(minutes=i)).isoformat()
        run_id = f"run-{i:04d}"
        log_path = log_dir / f"{run_id}.log"
        log_path.write_bytes(b"x" * 100)
        conn.execute(
            "INSERT INTO task_runs (run_id, task_id, command_json, trigger, status, log_path, created_at, updated_at) "
            "VALUES (?, ?, '[]', 'manual', 'success', ?, ?, ?)",
            (run_id, task_id, str(log_path), ts, ts),
        )
        paths.append(log_path)
    conn.commit()
    conn.close()
    return paths


def _patched_store(tmp_path: Path):
    """Patch prune_logs to use a tmp TaskerStore instead of production DATA_DIR."""
    from src.tasker.store import TaskerStore

    return patch.object(prune_logs, "TaskerStore", lambda: TaskerStore(
        db_path=tmp_path / "tasker.db",
        public_status_path=tmp_path / "public" / "tasker_status.json",
        cron_status_path=tmp_path / "data" / "cron_status.json",
        log_dir=tmp_path / "tasker_logs",
    ))


def test_cli_dry_run_deletes_nothing(tmp_path, capsys):
    _seed_tasker_logs(tmp_path, n=25)
    cron_dir = tmp_path / "data"
    cron_dir.mkdir(parents=True, exist_ok=True)

    with patch.object(prune_logs, "DATA_DIR", tmp_path), _patched_store(tmp_path), \
         patch.object(prune_logs, "_record_cron_status"):
        with patch.object(sys, "argv", ["prune_logs", "--dry-run"]):
            prune_logs.main()

    out = capsys.readouterr().out
    assert "DRY RUN" in out
    # all 25 files still present, only 5 beyond keep=20 planned
    assert len(list((tmp_path / "tasker_logs").glob("*.log"))) == 25
    assert "would delete 5 run-log files" in out


def test_cli_default_keep_20(tmp_path, capsys):
    _seed_tasker_logs(tmp_path, n=25)

    with patch.object(prune_logs, "DATA_DIR", tmp_path), _patched_store(tmp_path), \
         patch.object(prune_logs, "_record_cron_status"):
        with patch.object(sys, "argv", ["prune_logs"]):
            prune_logs.main()

    out = capsys.readouterr().out
    assert "keep_per_task=20" in out
    assert len(list((tmp_path / "tasker_logs").glob("*.log"))) == 20


def test_cli_writes_cron_status(tmp_path, capsys):
    _seed_tasker_logs(tmp_path, n=25)
    cron_status = tmp_path / "data" / "cron_status.json"
    cron_status.parent.mkdir(parents=True, exist_ok=True)

    # real cron_update call is subprocess; point it at the tmp status file via DATA_DIR patch
    # but cron_update.py writes to PROJECT_ROOT/data — patch the subprocess call instead.
    recorded = {}

    def fake_record(args, summary, health, started):
        recorded["called"] = True
        recorded["status"] = "ok"

    with patch.object(prune_logs, "DATA_DIR", tmp_path), _patched_store(tmp_path), \
         patch.object(prune_logs, "_record_cron_status", side_effect=fake_record):
        with patch.object(sys, "argv", ["prune_logs"]):
            prune_logs.main()

    assert recorded.get("called") is True
    assert recorded["status"] == "ok"


def test_cli_deletes_dead_health_log(tmp_path, capsys):
    # create a dead health.log predating the 2026-06-10 cutover
    health = tmp_path / "health.log"
    health.write_bytes(b"x" * 2048)
    old_mtime = datetime(2026, 6, 5, tzinfo=timezone.utc).timestamp()
    os.utime(health, (old_mtime, old_mtime))

    with patch.object(prune_logs, "DATA_DIR", tmp_path), \
         patch.object(prune_logs, "_record_cron_status"), \
         patch.object(prune_logs, "TaskerStore", lambda: _no_op_store()):
        with patch.object(sys, "argv", ["prune_logs", "--delete-dead-health-log"]):
            prune_logs.main()

    out = capsys.readouterr().out
    assert "deleted" in out
    assert not health.exists()


def test_cli_refuses_live_health_log(tmp_path, capsys):
    # health.log with a recent mtime -> must NOT be deleted
    health = tmp_path / "health.log"
    health.write_bytes(b"x" * 2048)
    recent_mtime = (datetime.now(timezone.utc) - timedelta(days=prune_logs.STALE_DAYS - 1)).timestamp()
    os.utime(health, (recent_mtime, recent_mtime))

    with patch.object(prune_logs, "DATA_DIR", tmp_path), \
         patch.object(prune_logs, "_record_cron_status"), \
         patch.object(prune_logs, "TaskerStore", lambda: _no_op_store()):
        with patch.object(sys, "argv", ["prune_logs", "--delete-dead-health-log"]):
            prune_logs.main()

    out = capsys.readouterr().out
    assert "skip" in out
    assert health.exists()  # not deleted


def test_rotation_truncates_oversized_log(tmp_path, capsys):
    # oversized tee-appended log -> original truncated in place, .1 carries content
    log = tmp_path / "cron.log"
    log.write_bytes(b"x" * 2000)

    with patch.object(prune_logs, "DATA_DIR", tmp_path), \
         patch.object(prune_logs, "ROTATE_THRESHOLD_BYTES", 1000), \
         patch.object(prune_logs, "TaskerStore", lambda: _no_op_store()), \
         patch.object(prune_logs, "_record_cron_status"):
        with patch.object(sys, "argv", ["prune_logs"]):
            prune_logs.main()

    out = capsys.readouterr().out
    assert "rotation cron.log: rotated" in out
    assert log.read_bytes() == b""  # truncated, not renamed (copytruncate)
    assert (tmp_path / "cron.log.1").read_bytes() == b"x" * 2000


def test_rotation_keep_chain_capped(tmp_path, capsys):
    # keep=3 chain: .1/.2 shift down, oldest .3 content dropped
    log = tmp_path / "cron.log"
    log.write_bytes(b"A" * 2000)
    (tmp_path / "cron.log.1").write_bytes(b"B" * 100)
    (tmp_path / "cron.log.2").write_bytes(b"C" * 50)
    (tmp_path / "cron.log.3").write_bytes(b"D" * 20)

    with patch.object(prune_logs, "DATA_DIR", tmp_path), \
         patch.object(prune_logs, "ROTATE_THRESHOLD_BYTES", 1000), \
         patch.object(prune_logs, "TaskerStore", lambda: _no_op_store()), \
         patch.object(prune_logs, "_record_cron_status"):
        with patch.object(sys, "argv", ["prune_logs"]):
            prune_logs.main()

    assert (tmp_path / "cron.log.1").read_bytes() == b"A" * 2000
    assert (tmp_path / "cron.log.2").read_bytes() == b"B" * 100
    assert (tmp_path / "cron.log.3").read_bytes() == b"C" * 50  # old .3 dropped
    assert log.read_bytes() == b""


def test_rotation_skips_sub_threshold(tmp_path, capsys):
    # sub-threshold files are untouched and never listed
    log = tmp_path / "dashboard.log"
    payload = b"y" * 500
    log.write_bytes(payload)

    with patch.object(prune_logs, "DATA_DIR", tmp_path), \
         patch.object(prune_logs, "ROTATE_THRESHOLD_BYTES", 1000), \
         patch.object(prune_logs, "TaskerStore", lambda: _no_op_store()), \
         patch.object(prune_logs, "_record_cron_status"):
        with patch.object(sys, "argv", ["prune_logs"]):
            prune_logs.main()

    out = capsys.readouterr().out
    assert "rotation" not in out
    assert log.read_bytes() == payload
    assert not (tmp_path / "dashboard.log.1").exists()


def test_rotation_dry_run_plans_without_mutating(tmp_path, capsys):
    log = tmp_path / "cron.log"
    payload = b"x" * 2000
    log.write_bytes(payload)

    with patch.object(prune_logs, "DATA_DIR", tmp_path), \
         patch.object(prune_logs, "ROTATE_THRESHOLD_BYTES", 1000), \
         patch.object(prune_logs, "TaskerStore", lambda: _no_op_store()), \
         patch.object(prune_logs, "_record_cron_status"):
        with patch.object(sys, "argv", ["prune_logs", "--dry-run"]):
            prune_logs.main()

    out = capsys.readouterr().out
    assert "rotation cron.log: would_rotate" in out
    assert log.read_bytes() == payload  # untouched
    assert not (tmp_path / "cron.log.1").exists()


def test_dead_log_list_stale_deleted_fresh_kept(tmp_path, capsys):
    # dead-log list: stale frozen file deleted, fresh file skipped
    stale = tmp_path / "unified_dashboard.log"
    stale.write_bytes(b"u" * 1024)
    os.utime(stale, (datetime(2026, 7, 18, tzinfo=timezone.utc).timestamp(),) * 2)
    fresh = tmp_path / "daily_pnl.log"
    fresh.write_bytes(b"p" * 1024)
    os.utime(fresh, (datetime.now(timezone.utc).timestamp(),) * 2)

    with patch.object(prune_logs, "DATA_DIR", tmp_path), \
         patch.object(prune_logs, "_record_cron_status"), \
         patch.object(prune_logs, "TaskerStore", lambda: _no_op_store()):
        with patch.object(sys, "argv", ["prune_logs", "--delete-dead-health-log"]):
            prune_logs.main()

    out = capsys.readouterr().out
    assert "unified_dashboard.log: deleted" in out
    assert "daily_pnl.log: skip" in out
    assert not stale.exists()
    assert fresh.exists()


def test_dead_log_dry_run_lists_present_only(tmp_path, capsys):
    # absent dead logs (e.g. health.log on prod) must not appear in the plan
    frozen = tmp_path / "overlay_signals.log"
    frozen.write_bytes(b"s" * 1024)
    os.utime(frozen, (datetime(2026, 7, 18, tzinfo=timezone.utc).timestamp(),) * 2)

    with patch.object(prune_logs, "DATA_DIR", tmp_path), \
         patch.object(prune_logs, "_record_cron_status"), \
         patch.object(prune_logs, "TaskerStore", lambda: _no_op_store()):
        with patch.object(sys, "argv", ["prune_logs", "--delete-dead-health-log", "--dry-run"]):
            prune_logs.main()

    out = capsys.readouterr().out
    assert "overlay_signals.log: would_delete" in out
    assert "health.log" not in out
    assert frozen.exists()  # dry run never deletes


def _no_op_store():
    """A TaskerStore stand-in whose prune_runs is a no-op (for log-only tests)."""
    class _Stub:
        log_dir = Path("/nonexistent")
        def prune_runs(self, keep_per_task=20, dry_run=False):
            return {"deleted_files": 0, "deleted_rows": 0, "kept_files": 0,
                    "bytes_freed": 0, "errors": [], "plan": []}
    return _Stub()
