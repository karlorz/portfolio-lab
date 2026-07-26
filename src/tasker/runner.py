"""Serialized subprocess runner for registered tasker tasks."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from src.paths import PROJECT_ROOT
from src.tasker.models import (
    EXIT_CODE_BLOCKED,
    INTENTIONAL_BLOCK_TASK_IDS,
    RUN_BLOCKED,
    RUN_CANCELLED,
    RUN_ERROR,
    RUN_SUCCESS,
    RUN_TIMEOUT,
)
from src.tasker.registry import TaskRegistry
from src.tasker.store import TaskerStore


class TaskRunner:
    """Run one registered task at a time."""

    def __init__(
        self,
        registry: TaskRegistry,
        store: TaskerStore,
        project_root: str | Path = PROJECT_ROOT,
        base_env: dict[str, str] | None = None,
    ):
        self.registry = registry
        self.store = store
        self.project_root = Path(project_root)
        self.base_env = dict(base_env or {})
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._cancelled: set[str] = set()

    def start_task(self, task_id: str, trigger: str = "manual", retry_of: str | None = None) -> dict[str, Any]:
        task = self.registry.get(task_id)
        if trigger == "scheduled" and (not task.enabled or task.manual_only):
            raise RuntimeError(f"Task is not scheduled: {task_id}")
        state = self.store.get_task(task_id, self.registry)["state"]
        if state["paused"]:
            raise RuntimeError(f"Task is paused: {task_id}")

        with self._lock:
            if any(thread.is_alive() for thread in self._threads.values()):
                raise RuntimeError("tasker worker is busy")
            run = self.store.create_run(task_id, task.command, trigger=trigger, retry_of=retry_of)
            thread = threading.Thread(target=self._run_subprocess, args=(run["run_id"],), daemon=True)
            self._threads[run["run_id"]] = thread
            thread.start()
            return run

    def cancel_run(self, run_id: str, grace_seconds: float = 5.0) -> bool:
        process = self._processes.get(run_id)
        self._cancelled.add(run_id)
        if process is None:
            return False
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return True

    def wait_for_run(self, run_id: str, timeout_seconds: float = 60.0) -> dict[str, Any]:
        thread = self._threads.get(run_id)
        if thread is not None:
            thread.join(timeout_seconds)
            if thread.is_alive():
                raise TimeoutError(f"Task run did not finish within {timeout_seconds}s: {run_id}")
        return self.store.get_run(run_id)

    def _run_subprocess(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        command = run["command"]
        started = time.monotonic()
        process: subprocess.Popen[Any] | None = None
        try:
            log_path = Path(run["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(self.base_env)
            env.update(
                {
                    "TASKER_RUN_ID": run_id,
                    "CRON_RUN_ID": run_id,
                    "CRON_BACKEND": "tasker",
                    "PORTFOLIO_LAB_ENABLE_ML": "0",
                }
            )
            with log_path.open("ab") as log:
                process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self._processes[run_id] = process
                self.store.mark_run_running(run_id, pid=process.pid)
                try:
                    process.wait(timeout=self.registry.get(run["task_id"]).timeout_seconds)
                except subprocess.TimeoutExpired:
                    self._cancelled.add(run_id)
                    self._terminate_process_group(process)
                    duration = time.monotonic() - started
                    self.store.finish_run(
                        run_id,
                        status=RUN_TIMEOUT,
                        exit_code=process.returncode,
                        duration_seconds=duration,
                        error="timeout",
                    )
                    return

            duration = time.monotonic() - started
            error_msg: str | None = None
            if run_id in self._cancelled:
                status = RUN_CANCELLED
            elif process.returncode == 0:
                status = RUN_SUCCESS
            elif (
                process.returncode == EXIT_CODE_BLOCKED
                and run["task_id"] in INTENTIONAL_BLOCK_TASK_IDS
            ):
                # Evaluator/control-loop intentional skip under kill — not a hard error
                status = RUN_BLOCKED
            elif process.returncode == EXIT_CODE_BLOCKED:
                # Batch CE: bare exit 2 from make (recipe parse / missing separator)
                # must not look like intentional blocked. Count as error + hint.
                status = RUN_ERROR
                error_msg = (
                    "exit_code=2 from non-intentional-block task "
                    f"{run['task_id']!r}; treat as make/recipe failure "
                    f"(only {sorted(INTENTIONAL_BLOCK_TASK_IDS)} may use "
                    "EXIT_BLOCKED). Check tasker log for 'missing separator' "
                    "or other make parse errors."
                )
            else:
                status = RUN_ERROR
            self.store.finish_run(
                run_id,
                status=status,
                exit_code=process.returncode,
                duration_seconds=duration,
                error=error_msg,
            )
        except Exception as exc:  # pragma: no cover - defensive runtime path
            duration = time.monotonic() - started
            self.store.finish_run(run_id, status=RUN_ERROR, exit_code=None, duration_seconds=duration, error=str(exc))
        finally:
            self._processes.pop(run_id, None)
            self._cancelled.discard(run_id)
            self._threads.pop(run_id, None)

    def _terminate_process_group(self, process: subprocess.Popen[Any]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
