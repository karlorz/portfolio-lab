"""Long-running tasker service entry point."""

from __future__ import annotations

import fcntl
import logging
import os
import signal
import threading
import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.paths import DATA_DIR
from src.tasker.api import create_app
from src.tasker.registry import TaskRegistry, load_task_registry
from src.tasker.runner import TaskRunner
from src.tasker.store import TaskerStore
from src.utils.log_config import configure_logging
import waitress

logger = logging.getLogger(__name__)

# TASKER-HARDENING s1 (2026-08-14): single-instance guard. Only one tasker
# service may hold the scheduler + serialized runner on the shared durable
# store; a second instance (e.g. a manual `python -m src.tasker.service`
# started while the systemd unit is up) previously duplicated every scheduled
# run. flock auto-releases on process death (incl. SIGKILL), so there is no
# stale-lock window across systemd RestartSec=10 restarts.
TASKER_LOCK_PATH = DATA_DIR / "tasker.lock"
_SINGLETON_LOCK_FD: object | None = None  # held for the process lifetime


def acquire_singleton_lock(lock_path: Path | None = None) -> None:
    """Take an exclusive flock so only one tasker service instance runs.

    The lock file also records the holder PID for diagnostics. Raises
    SystemExit(1) (``sys.exit(1)`` semantics) when another instance already
    holds the lock; the message is logged via the tasker logger so
    systemd/journald capture it. ``--once`` mirror-refresh runs deliberately
    skip the guard (short-lived helper; must run alongside the service).

    Args:
        lock_path: Override for the lock file (tests use tmp dirs).
    """
    global _SINGLETON_LOCK_FD
    path = Path(lock_path) if lock_path is not None else TASKER_LOCK_PATH
    fd = path.open("a+")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = ""
        try:
            fd.seek(0)
            holder = fd.read().strip()
        except OSError:  # pragma: no cover - diagnostics only
            pass
        fd.close()
        logger.error(
            "tasker singleton lock already held (pid %s): refusing to start a second scheduler instance",
            holder or "unknown",
        )
        raise SystemExit(1)
    try:
        fd.seek(0)
        fd.truncate()
        fd.write(str(os.getpid()))
        fd.flush()
    except OSError:  # pragma: no cover - the lock is held even without the note
        pass
    # Keep the fd referenced for the process lifetime: closing it (or GC)
    # would release the flock and let a second instance in.
    _SINGLETON_LOCK_FD = fd


class TaskerService:
    """Internal scheduler loop plus serialized runner."""

    def __init__(self, registry: TaskRegistry, store: TaskerStore, runner: TaskRunner):
        self.registry = registry
        self.store = store
        self.runner = runner
        self._stop = threading.Event()
        self._draining = threading.Event()
        self._last_fired: set[tuple[str, str]] = set()
        # TASKER-HARDENING s2: tasks whose start was blocked by a busy worker
        # warn at most once per due minute (bounded like _last_fired below).
        self._busy_warned: set[tuple[str, str]] = set()

    @property
    def is_draining(self) -> bool:
        return self._draining.is_set()

    def start_background_scheduler(self) -> threading.Thread:
        thread = threading.Thread(target=self.run_scheduler_loop, name="portfolio-lab-tasker-scheduler", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()

    def drain(self, termination_cause: str = "service_restart", termination_detail: str | None = None) -> None:
        """Enter draining: stop scheduling, reject starts, finalize active runs.

        The drain event also makes the API reject mutating actions and the
        runner refuse new starts. Active children are cancelled with the named
        cause and bounded to a terminal state before the service exits, so a
        replacement service never waits through orphan grace for them.
        """
        if self._draining.is_set():
            return
        self._draining.set()
        self._stop.set()
        logger.info("Tasker draining (cause=%s): stopping scheduler", termination_cause)
        try:
            active = self.runner.drain_active_runs(
                termination_cause=termination_cause,
                termination_detail=termination_detail,
            )
            if active:
                logger.info(
                    "Tasker drain finalized %d active run(s): %s",
                    len(active),
                    ", ".join(active),
                )
        except Exception as exc:  # noqa: BLE001 - drain must not wedge shutdown
            logger.exception("Tasker drain failed while finalizing runs: %s", exc)

    def run_scheduler_loop(self) -> None:
        while not self._stop.is_set() and not self._draining.is_set():
            self.tick(datetime.now(timezone.utc))
            self._stop.wait(15)

    def tick(self, now: datetime) -> None:
        if self._draining.is_set():
            return
        self.reconcile_orphaned_runs(now=now)
        minute_key = now.replace(second=0, microsecond=0).isoformat()
        for task in self.registry.due_tasks(now):
            fired_key = (task.id, minute_key)
            if fired_key in self._last_fired:
                continue
            try:
                state = self.store.get_task(task.id, self.registry)["state"]
                if state["paused"]:
                    continue
                if self._own_run_in_flight(task.id):
                    # TASKER-HARDENING s2: the task's own previous run is
                    # still active (a long run spanning its next due minute);
                    # the busy skip is expected, not a fault — keep it at
                    # DEBUG so long runs do not spam the journal with
                    # warnings on every 15s tick.
                    logger.debug("Tasker skipped %s: own in-flight run still active", task.id)
                    self._last_fired.add(fired_key)
                    continue
                try:
                    self.runner.start_task(task.id, trigger="scheduled")
                    self._last_fired.add(fired_key)
                except RuntimeError as exc:
                    if fired_key not in self._busy_warned:
                        logger.warning("Tasker skipped %s: %s", task.id, exc)
                        self._busy_warned.add(fired_key)
            except Exception as exc:  # noqa: BLE001 - isolate per-task failures
                # A single task's DB/read error must not kill the scheduler
                # loop (which would silently skip every later job until a
                # service restart). Log and keep ticking.
                logger.exception("Tasker tick failed for %s: %s", task.id, exc)
        if len(self._last_fired) > 5000:
            self._last_fired = set(list(self._last_fired)[-1000:])
        if len(self._busy_warned) > 5000:
            self._busy_warned = set(list(self._busy_warned)[-1000:])
        try:
            self.store.write_status_mirrors(self.registry)
        except Exception as exc:  # noqa: BLE001 - mirror writes must not kill the loop
            logger.exception("Tasker status mirror write failed: %s", exc)

    def _own_run_in_flight(self, task_id: str) -> bool:
        """True when task_id's own previous run still claims to be active.

        The single worker is serialized via the shared RUNNING row, so a
        task whose own run is still going is expected to be busy; callers
        treat that as a quiet skip rather than a warning.
        """
        list_running = getattr(self.store, "list_running_runs", None)
        if list_running is None:
            return False
        try:
            running = list_running()
        except Exception:  # noqa: BLE001 - a read failure must not block scheduling
            return False
        return any(run.get("task_id") == task_id for run in running)

    def reconcile_orphaned_runs(self, *, now: datetime | None = None) -> list[str]:
        reconciler = getattr(self.runner, "reconcile_orphaned_runs", None)
        if reconciler is None:
            return []
        try:
            return list(reconciler(now=now))
        except Exception as exc:  # noqa: BLE001 - reconciliation must not kill scheduler
            logger.exception("Tasker orphan reconciliation failed: %s", exc)
            return []


def build_service() -> tuple[TaskerService, object]:
    registry = load_task_registry()
    store = TaskerStore()
    store.sync_registry(registry)
    runner = TaskRunner(registry=registry, store=store)
    service = TaskerService(registry=registry, store=store, runner=runner)
    # Reconcile before the API is exposed so a restart immediately repairs the
    # durable state and mirrors even when the scheduler is disabled.
    service.reconcile_orphaned_runs()
    try:
        store.write_status_mirrors(registry)
    except Exception as exc:  # noqa: BLE001 - startup mirrors are best effort
        logger.exception("Tasker startup status mirror write failed: %s", exc)
    app = create_app(registry=registry, store=store, runner=runner, draining=service._draining)
    return service, app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Portfolio Lab tasker service.")
    parser.add_argument("--once", action="store_true", help="Write status mirrors and exit without starting the API server.")
    parser.add_argument("--host", default=os.environ.get("TASKER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TASKER_PORT", "8000")))
    parser.add_argument("--no-scheduler", action="store_true", help="Start the API server without the scheduler loop.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging()
    if not args.once:
        # TASKER-HARDENING s1: the single-instance guard must run before
        # build_service() — a second instance must not write mirrors,
        # reconcile runs, or bind the API.
        try:
            acquire_singleton_lock()
        except SystemExit:
            return 1
    service, app = build_service()

    if args.once:
        service.store.write_status_mirrors(service.registry)
        return 0

    if not args.no_scheduler and os.environ.get("TASKER_DISABLE_SCHEDULER") != "1":
        service.start_background_scheduler()

    # Bounded graceful drain (Task 3B): SIGTERM/SIGINT enter draining, finalize
    # active runs with a named cause, then the process exits cleanly inside
    # systemd's TimeoutStopSec so no control-group kill races the finalization.
    shutdown = threading.Event()

    def _handle_signal(signum, _frame):
        logger.info("Tasker received signal %s: entering drain", signum)
        shutdown.set()

    previous_term = signal.signal(signal.SIGTERM, _handle_signal)
    previous_int = signal.signal(signal.SIGINT, _handle_signal)
    try:
        # Waitress (ops follow-up decision #5): one process, four request
        # threads, bounded limits; explicit server so SIGTERM can stop
        # accepting requests before the bounded drain finalizes runs.
        server = waitress.create_server(
            app,
            host=args.host,
            port=args.port,
            threads=4,
            connection_limit=50,
            channel_timeout=120,
            max_request_header_size=16384,
            max_request_body_size=1048576,
            expose_tracebacks=False,
        )
        server_thread = threading.Thread(
            target=server.run,
            name="portfolio-lab-tasker-api",
            daemon=True,
        )
        server_thread.start()
        shutdown.wait()
        logger.info("Tasker shutdown requested: stopping API and draining before exit")
        server.close()
        service.drain(termination_cause="service_restart", termination_detail="service shutdown signal")
        try:
            service.store.write_status_mirrors(service.registry)
        except Exception as exc:  # noqa: BLE001 - final mirrors are best effort
            logger.exception("Tasker final status mirror write failed: %s", exc)
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


if __name__ == "__main__":
    raise SystemExit(main())
