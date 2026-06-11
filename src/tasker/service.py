"""Long-running tasker service entry point."""

from __future__ import annotations

import logging
import os
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
        self._last_fired: set[tuple[str, str]] = set()

    def start_background_scheduler(self) -> threading.Thread:
        thread = threading.Thread(target=self.run_scheduler_loop, name="portfolio-lab-tasker-scheduler", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()

    def run_scheduler_loop(self) -> None:
        while not self._stop.is_set():
            self.tick(datetime.now(timezone.utc))
            self._stop.wait(15)

    def tick(self, now: datetime) -> None:
        minute_key = now.replace(second=0, microsecond=0).isoformat()
        for task in self.registry.due_tasks(now):
            fired_key = (task.id, minute_key)
            if fired_key in self._last_fired:
                continue
            state = self.store.get_task(task.id, self.registry)["state"]
            if state["paused"]:
                continue
            try:
                self.runner.start_task(task.id, trigger="scheduled")
                self._last_fired.add(fired_key)
            except RuntimeError as exc:
                logger.warning("Tasker skipped %s: %s", task.id, exc)
        if len(self._last_fired) > 5000:
            self._last_fired = set(list(self._last_fired)[-1000:])
        self.store.write_status_mirrors(self.registry)


def build_service() -> tuple[TaskerService, object]:
    registry = load_task_registry()
    store = TaskerStore()
    store.sync_registry(registry)
    runner = TaskRunner(registry=registry, store=store)
    service = TaskerService(registry=registry, store=store, runner=runner)
    app = create_app(registry=registry, store=store, runner=runner)
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
    app.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
