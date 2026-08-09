"""Serialized subprocess runner for registered tasker tasks."""

from __future__ import annotations

import errno
import logging
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
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

logger = logging.getLogger(__name__)


# A service restart must not immediately declare a process lost: the child can
# be between exec/initialisation and the first durable heartbeat. Keep this
# bounded so a dead row cannot hold the single worker forever.
ORPHAN_RUN_GRACE_SECONDS = 60.0
MAX_ORPHAN_RUN_GRACE_SECONDS = 300.0


class TaskRunner:
    """Run one registered task at a time."""

    def __init__(
        self,
        registry: TaskRegistry,
        store: TaskerStore,
        project_root: str | Path = PROJECT_ROOT,
        base_env: dict[str, str] | None = None,
        draining: threading.Event | None = None,
    ):
        self.registry = registry
        self.store = store
        self.project_root = Path(project_root)
        self.base_env = dict(base_env or {})
        # Shared drain signal: when set, the runner refuses new starts so a
        # service drain can stop scheduling without racing an in-flight run.
        self._draining = draining if draining is not None else threading.Event()
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._cancelled: set[str] = set()
        # Per-run named termination cause recorded during drain/cancel so the
        # finishing thread persists it with the terminal transition.
        self._termination_causes: dict[str, tuple[str | None, str | None]] = {}

    @property
    def draining(self) -> threading.Event:
        return self._draining

    def start_task(self, task_id: str, trigger: str = "manual", retry_of: str | None = None) -> dict[str, Any]:
        if self._draining.is_set():
            raise RuntimeError(f"tasker is draining: {task_id} not started")
        task = self.registry.get(task_id)
        if trigger == "scheduled" and (not task.enabled or task.manual_only):
            raise RuntimeError(f"Task is not scheduled: {task_id}")
        state = self.store.get_task(task_id, self.registry)["state"]
        if state["paused"]:
            raise RuntimeError(f"Task is paused: {task_id}")

        # A fresh service process has no in-memory thread map. Reconcile only
        # rows that are old enough and provably dead before accepting work;
        # otherwise a durable orphan would be silently bypassed.
        self.reconcile_orphaned_runs()
        with self._lock:
            if any(thread.is_alive() for thread in self._threads.values()):
                raise RuntimeError("tasker worker is busy")
            if self.store.list_running_runs():
                raise RuntimeError("tasker worker is busy")
            run = self.store.create_run(task_id, task.command, trigger=trigger, retry_of=retry_of)
            thread = threading.Thread(target=self._run_subprocess, args=(run["run_id"],), daemon=True)
            self._threads[run["run_id"]] = thread
            thread.start()
            return run

    def reconcile_orphaned_runs(
        self,
        *,
        now: datetime | None = None,
        grace_seconds: float = ORPHAN_RUN_GRACE_SECONDS,
    ) -> list[str]:
        """Finalize durable RUNNING rows whose worker process is gone.

        This is deliberately conservative. A row is eligible only when its
        start age exceeds a bounded grace period, it is not owned by a local
        runner thread/process, and its PID is no longer alive. Live or
        ambiguous PIDs are preserved so PID reuse or a service restart cannot
        turn an active task into a false failure.
        """
        observed_at = now or datetime.now(timezone.utc)
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        observed_at = observed_at.astimezone(timezone.utc)
        try:
            requested_grace = float(grace_seconds)
        except (TypeError, ValueError):
            requested_grace = ORPHAN_RUN_GRACE_SECONDS
        grace = max(0.0, min(requested_grace, MAX_ORPHAN_RUN_GRACE_SECONDS))

        reconciled: list[str] = []
        for run in self.store.list_running_runs():
            run_id = run["run_id"]
            if self._local_runner_owns_run(run_id, run.get("pid")):
                continue
            age_seconds = self._run_age_seconds(run, observed_at)
            if age_seconds is None or age_seconds < grace:
                continue
            if self._process_is_alive(run.get("pid")):
                # The ownership probe is intentionally observed even when the
                # process is not ours: a live PID is never safe to finalize.
                owned = self._process_belongs_to_run(run.get("pid"), run_id)
                logger.debug(
                    "Preserving live tasker run %s (pid=%s, owned=%s, age=%.1fs)",
                    run_id,
                    run.get("pid"),
                    owned,
                    age_seconds,
                )
                continue

            error = (
                "orphaned_run: tasker process is no longer alive after "
                f"{age_seconds:.1f}s (pid={run.get('pid')!r}, grace={grace:.1f}s)"
            )
            try:
                transitioned = self.store.finish_run(
                    run_id,
                    status=RUN_ERROR,
                    exit_code=None,
                    duration_seconds=age_seconds,
                    error=error,
                )
            except KeyError:
                logger.info("Tasker run %s disappeared during orphan reconciliation", run_id)
                continue
            if transitioned:
                reconciled.append(run_id)
                logger.warning("Finalized orphaned tasker run %s: %s", run_id, error)
        return reconciled

    def _run_age_seconds(self, run: dict[str, Any], now: datetime) -> float | None:
        started_at = run.get("started_at")
        if not started_at:
            return None
        try:
            started = datetime.fromisoformat(str(started_at))
        except (TypeError, ValueError):
            logger.warning(
                "Cannot reconcile tasker run %s with invalid started_at=%r",
                run.get("run_id"),
                started_at,
            )
            return None
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        return max(0.0, (now - started.astimezone(timezone.utc)).total_seconds())

    def _local_runner_owns_run(self, run_id: str, pid: int | None) -> bool:
        with self._lock:
            process = self._processes.get(run_id)
            thread = self._threads.get(run_id)
        if process is not None and (pid is None or process.pid == pid):
            return True
        # A child may have exited while its runner thread is still recording
        # the terminal result. Do not race that thread with reconciliation.
        return thread is not None and thread.is_alive()

    def _process_is_alive(self, pid: int | None) -> bool:
        try:
            normalized_pid = int(pid)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            # Missing or malformed ownership is ambiguous, not proof of a
            # dead worker. Preserve the durable row for an operator/repair
            # path rather than falsely finalizing it.
            return True
        if normalized_pid <= 0:
            return True
        try:
            os.kill(normalized_pid, 0)
        except ProcessLookupError:
            return False
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return False
            # EPERM and unknown OS errors are treated as alive/ambiguous. A
            # false preserve is safer than finalizing a process we cannot
            # inspect.
            return True
        return True

    def _process_belongs_to_run(self, pid: int | None, run_id: str) -> bool:
        """Best-effort Linux ownership check used only for diagnostics.

        The tasker injects TASKER_RUN_ID into every child environment. A live
        process without that marker is still preserved; this method never
        converts an ownership uncertainty into a terminal result.
        """
        try:
            normalized_pid = int(pid)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        if normalized_pid <= 0:
            return False
        try:
            environ = Path(f"/proc/{normalized_pid}/environ").read_bytes().split(b"\0")
        except (OSError, ValueError):
            return False
        marker = f"TASKER_RUN_ID={run_id}".encode()
        return marker in environ

    def cancel_run(
        self,
        run_id: str,
        grace_seconds: float = 5.0,
        termination_cause: str | None = None,
        termination_detail: str | None = None,
    ) -> bool:
        process = self._processes.get(run_id)
        self._cancelled.add(run_id)
        if termination_cause is not None:
            self._termination_causes[run_id] = (termination_cause, termination_detail)
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

    def drain_active_runs(
        self,
        *,
        termination_cause: str = "service_restart",
        termination_detail: str | None = None,
        grace_seconds: float = 10.0,
    ) -> list[str]:
        """Cancel all locally active runs with a named planned cause (Task 3B).

        Called during service drain. Every active child receives SIGTERM (with
        a bounded SIGKILL escalation), and the finishing thread persists the
        named cause so the terminal transition never looks like an unplanned
        failure. Returns the run IDs that were active.
        """
        with self._lock:
            run_ids = list(self._processes.keys())
        drained: list[str] = []
        for run_id in run_ids:
            try:
                if self.cancel_run(
                    run_id,
                    grace_seconds=grace_seconds,
                    termination_cause=termination_cause,
                    termination_detail=termination_detail,
                ):
                    drained.append(run_id)
            except Exception as exc:  # noqa: BLE001 - drain must not wedge
                logger.exception("Tasker drain failed to cancel run %s: %s", run_id, exc)
        for run_id in drained:
            try:
                self.wait_for_run(run_id, timeout_seconds=max(grace_seconds, 10.0))
            except TimeoutError:
                logger.warning("Tasker drain: run %s did not finalize in time", run_id)
        return drained

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
            cause, detail = self._termination_causes.pop(run_id, (None, None))
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
                termination_cause=cause,
                termination_detail=detail,
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
