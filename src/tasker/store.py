"""SQLite state store for Portfolio Lab tasker."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.paths import DATA_DIR, PUBLIC_DATA_DIR, TASKER_DB, sqlite_connect
from src.tasker.models import RUN_CANCELLED, RUN_ERROR, RUN_SUCCESS, RUN_TIMEOUT
from src.tasker.registry import TaskRegistry


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskerStore:
    """Durable task state and run history."""

    def __init__(
        self,
        db_path: str | Path = TASKER_DB,
        public_status_path: str | Path = PUBLIC_DATA_DIR / "tasker_status.json",
        cron_status_path: str | Path = DATA_DIR / "cron_status.json",
        log_dir: str | Path = DATA_DIR / "tasker_logs",
    ):
        self.db_path = Path(db_path)
        self.public_status_path = Path(public_status_path)
        self.cron_status_path = Path(cron_status_path)
        self.log_dir = Path(log_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def sync_registry(self, registry: TaskRegistry) -> None:
        now = _utc_now()
        with self._connect() as conn:
            for task in registry.tasks:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO task_state (
                        task_id, paused, failure_count, consecutive_failures, created_at, updated_at
                    )
                    VALUES (?, 0, 0, 0, ?, ?)
                    """,
                    (task.id, now, now),
                )

    def get_task(self, task_id: str, registry: TaskRegistry) -> dict[str, Any]:
        definition = registry.get(task_id)
        self.sync_registry(registry)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM task_state WHERE task_id = ?", (task_id,)).fetchone()
        return {
            "id": definition.id,
            "label": definition.label,
            "definition": definition.to_dict(),
            "state": self._state_row_to_dict(row),
        }

    def list_tasks(self, registry: TaskRegistry) -> list[dict[str, Any]]:
        self.sync_registry(registry)
        return [self.get_task(task.id, registry) for task in registry.tasks]

    def set_task_paused(self, task_id: str, paused: bool, reason: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE task_state
                SET paused = ?, pause_reason = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (1 if paused else 0, reason if paused else None, _utc_now(), task_id),
            )

    def create_run(
        self,
        task_id: str,
        command: list[str],
        trigger: str,
        retry_of: str | None = None,
    ) -> dict[str, Any]:
        run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        now = _utc_now()
        log_path = str(self.log_dir / f"{run_id}.log")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_runs (
                    run_id, task_id, command_json, trigger, retry_of, status, log_path, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (run_id, task_id, json.dumps(command), trigger, retry_of, log_path, now, now),
            )
        return self.get_run(run_id)

    def mark_run_running(self, run_id: str, pid: int) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE task_runs
                SET status = 'running', pid = ?, started_at = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (pid, now, now, run_id),
            )

    def finish_run(
        self,
        run_id: str,
        status: str,
        exit_code: int | None,
        duration_seconds: float,
        error: str | None = None,
    ) -> None:
        finished_at = _utc_now()
        with self._connect() as conn:
            run = conn.execute("SELECT * FROM task_runs WHERE run_id = ?", (run_id,)).fetchone()
            if run is None:
                raise KeyError(f"Unknown run: {run_id}")
            conn.execute(
                """
                UPDATE task_runs
                SET status = ?, exit_code = ?, duration_seconds = ?, error = ?, finished_at = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (status, exit_code, duration_seconds, error, finished_at, finished_at, run_id),
            )
            self._update_task_health(conn, run["task_id"], run_id, status, exit_code, duration_seconds, finished_at)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM task_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown run: {run_id}")
        return self._run_row_to_dict(row)

    def list_runs(self, task_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            if task_id:
                rows = conn.execute(
                    "SELECT * FROM task_runs WHERE task_id = ? ORDER BY created_at DESC LIMIT ?",
                    (task_id, limit),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM task_runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._run_row_to_dict(row) for row in rows]

    def read_run_logs(self, run_id: str, tail: int = 200) -> str:
        run = self.get_run(run_id)
        log_path = Path(run["log_path"])
        if not log_path.exists():
            return ""
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max(1, min(int(tail), 5000)):])

    def prune_runs(self, keep_per_task: int = 20, dry_run: bool = False) -> dict[str, Any]:
        """Bound run-log growth: keep the newest ``keep_per_task`` runs per task.

        Deletes older per-run ``.log`` files and their ``task_runs`` rows in
        lockstep, so ``list_runs()`` never returns dangling references. Orphan
        rows (file already missing) are dropped and reported in ``errors``.
        Hygiene, not a release gate — pruning never raises on a missing file.

        Returns a summary dict:
            deleted_files, deleted_rows, kept_files, bytes_freed, errors, plan.
        ``plan`` is populated only when ``dry_run=True`` (run_ids that would be
        deleted); a real run leaves ``plan`` empty.
        """
        keep = max(0, int(keep_per_task))
        summary: dict[str, Any] = {
            "deleted_files": 0,
            "deleted_rows": 0,
            "kept_files": 0,
            "bytes_freed": 0,
            "errors": [],
            "plan": [],
        }
        with self._connect() as conn:
            total_before = conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0]
            # run_ids older than the keep window, per task (newest first)
            prune_rows = conn.execute(
                """
                SELECT run_id, task_id, log_path
                FROM (
                    SELECT run_id, task_id, log_path, created_at,
                           ROW_NUMBER() OVER (
                               PARTITION BY task_id ORDER BY created_at DESC
                           ) AS rn
                    FROM task_runs
                ) WHERE rn > ?
                ORDER BY created_at ASC
                """,
                (keep,),
            ).fetchall()
            ids_to_delete: list[str] = []
            for row in prune_rows:
                run_id = row["run_id"]
                log_path = Path(row["log_path"])
                try:
                    size = log_path.stat().st_size
                except FileNotFoundError:
                    size = 0
                    summary["errors"].append({"run_id": run_id, "reason": "log file missing"})
                summary["bytes_freed"] += size
                if dry_run:
                    summary["plan"].append(
                        {"run_id": run_id, "task_id": row["task_id"], "log_path": str(log_path), "bytes": size}
                    )
                    continue
                if size:
                    log_path.unlink(missing_ok=True)
                    summary["deleted_files"] += 1
                ids_to_delete.append(run_id)
            if ids_to_delete and not dry_run:
                placeholders = ",".join("?" for _ in ids_to_delete)
                conn.execute(
                    f"DELETE FROM task_runs WHERE run_id IN ({placeholders})",
                    ids_to_delete,
                )
                summary["deleted_rows"] = len(ids_to_delete)
            summary["kept_files"] = total_before - summary["deleted_rows"]
        return summary

    def status_payload(self, registry: TaskRegistry) -> dict[str, Any]:
        tasks = self.list_tasks(registry)
        return {
            "service": "portfolio-lab-tasker",
            "backend": "tasker",
            "timestamp": _utc_now(),
            "tasks": [self._flatten_task(task) for task in tasks],
            "recent_runs": self.list_runs(limit=20),
        }

    def write_status_mirrors(self, registry: TaskRegistry) -> dict[str, Any]:
        payload = self.status_payload(registry)
        self.public_status_path.parent.mkdir(parents=True, exist_ok=True)
        self.cron_status_path.parent.mkdir(parents=True, exist_ok=True)
        self.public_status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.cron_status_path.write_text(json.dumps(self._cron_payload(payload), indent=2), encoding="utf-8")
        return payload

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite_connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS task_state (
                    task_id TEXT PRIMARY KEY,
                    paused INTEGER NOT NULL DEFAULT 0,
                    pause_reason TEXT,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    last_run_id TEXT,
                    last_status TEXT,
                    last_started_at TEXT,
                    last_finished_at TEXT,
                    last_duration_seconds REAL,
                    last_exit_code INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS task_runs (
                    run_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    command_json TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    retry_of TEXT,
                    status TEXT NOT NULL,
                    pid INTEGER,
                    started_at TEXT,
                    finished_at TEXT,
                    duration_seconds REAL,
                    exit_code INTEGER,
                    error TEXT,
                    log_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_task_runs_task_created
                    ON task_runs (task_id, created_at DESC);
                """
            )

    def _update_task_health(
        self,
        conn: sqlite3.Connection,
        task_id: str,
        run_id: str,
        status: str,
        exit_code: int | None,
        duration_seconds: float,
        finished_at: str,
    ) -> None:
        state = conn.execute("SELECT * FROM task_state WHERE task_id = ?", (task_id,)).fetchone()
        failure_count = int(state["failure_count"]) if state else 0
        consecutive = int(state["consecutive_failures"]) if state else 0
        if status in {RUN_ERROR, RUN_TIMEOUT}:
            failure_count += 1
            consecutive += 1
        elif status == RUN_SUCCESS:
            consecutive = 0
        elif status == RUN_CANCELLED:
            pass

        conn.execute(
            """
            UPDATE task_state
            SET failure_count = ?,
                consecutive_failures = ?,
                last_run_id = ?,
                last_status = ?,
                last_finished_at = ?,
                last_duration_seconds = ?,
                last_exit_code = ?,
                updated_at = ?
            WHERE task_id = ?
            """,
            (failure_count, consecutive, run_id, status, finished_at, duration_seconds, exit_code, finished_at, task_id),
        )

    def _state_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "paused": bool(row["paused"]),
            "pause_reason": row["pause_reason"],
            "failure_count": int(row["failure_count"]),
            "consecutive_failures": int(row["consecutive_failures"]),
            "last_run_id": row["last_run_id"],
            "last_status": row["last_status"],
            "last_started_at": row["last_started_at"],
            "last_finished_at": row["last_finished_at"],
            "last_duration_seconds": row["last_duration_seconds"],
            "last_exit_code": row["last_exit_code"],
            "updated_at": row["updated_at"],
        }

    def _run_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "task_id": row["task_id"],
            "command": json.loads(row["command_json"]),
            "trigger": row["trigger"],
            "retry_of": row["retry_of"],
            "status": row["status"],
            "pid": row["pid"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "duration_seconds": row["duration_seconds"],
            "exit_code": row["exit_code"],
            "error": row["error"],
            "log_path": row["log_path"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _flatten_task(self, task: dict[str, Any]) -> dict[str, Any]:
        definition = task["definition"]
        state = task["state"]
        return {
            "id": definition["id"],
            "label": definition["label"],
            "command": definition["command"],
            "schedule": definition["schedule"],
            "enabled": definition["enabled"],
            "manual_only": definition["manual_only"],
            "timeout_seconds": definition["timeout_seconds"],
            "paused": state["paused"],
            "pause_reason": state["pause_reason"],
            "last_status": state["last_status"],
            "last_run_id": state["last_run_id"],
            "last_finished_at": state["last_finished_at"],
            "last_duration_seconds": state["last_duration_seconds"],
            "failure_count": state["failure_count"],
            "consecutive_failures": state["consecutive_failures"],
        }

    def _cron_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "backend": "tasker",
            "timestamp": payload["timestamp"],
            "jobs": [
                {
                    "name": task["id"],
                    "status": task["last_status"] or "pending",
                    "last_run": task["last_finished_at"],
                    "duration_seconds": task["last_duration_seconds"],
                    "backend": "tasker",
                }
                for task in payload["tasks"]
            ],
        }
