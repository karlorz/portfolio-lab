"""Long-running tasker service entry point."""

from __future__ import annotations

import logging
import os
import signal
import threading
import argparse
from datetime import datetime, timezone

from src.tasker.api import create_app
from src.tasker.registry import TaskRegistry, load_task_registry
from src.tasker.runner import TaskRunner
from src.tasker.store import TaskerStore
from src.utils.log_config import configure_logging

logger = logging.getLogger(__name__)


class TaskerService:
    """Internal scheduler loop plus serialized runner."""

    def __init__(self, registry: TaskRegistry, store: TaskerStore, runner: TaskRunner):
        self.registry = registry
        self.store = store
        self.runner = runner
        self._stop = threading.Event()
        self._draining = threading.Event()
        self._last_fired: set[tuple[str, str]] = set()

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
                try:
                    self.runner.start_task(task.id, trigger="scheduled")
                    self._last_fired.add(fired_key)
                except RuntimeError as exc:
                    logger.warning("Tasker skipped %s: %s", task.id, exc)
            except Exception as exc:  # noqa: BLE001 - isolate per-task failures
                # A single task's DB/read error must not kill the scheduler
                # loop (which would silently skip every later job until a
                # service restart). Log and keep ticking.
                logger.exception("Tasker tick failed for %s: %s", task.id, exc)
        if len(self._last_fired) > 5000:
            self._last_fired = set(list(self._last_fired)[-1000:])
        try:
            self.store.write_status_mirrors(self.registry)
        except Exception as exc:  # noqa: BLE001 - mirror writes must not kill the loop
            logger.exception("Tasker status mirror write failed: %s", exc)

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
    parser.add_argument("--once", action="store_true", help="Write status mirrors and exit without starting Flask.")
    parser.add_argument("--host", default=os.environ.get("TASKER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TASKER_PORT", "8000")))
    parser.add_argument("--no-scheduler", action="store_true", help="Start the API server without the scheduler loop.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging()
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
        server_thread = threading.Thread(
            target=app.run,
            kwargs={"host": args.host, "port": args.port, "use_reloader": False},
            name="portfolio-lab-tasker-api",
            daemon=True,
        )
        server_thread.start()
        shutdown.wait()
        logger.info("Tasker shutdown requested: draining before exit")
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
