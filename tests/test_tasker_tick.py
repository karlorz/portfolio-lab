"""TASKER-HARDENING s2: busy-skip warning suppression in the scheduler tick."""

import logging
from datetime import datetime, timezone

from src.tasker import service


class _Def:
    def __init__(self, task_id):
        self.id = task_id
        self.enabled = True
        self.manual_only = False
        self.schedule = "* * * * *"


class _Store:
    def __init__(self, running_tasks=()):
        self.running_tasks = list(running_tasks)

    def get_task(self, task_id, registry):
        return {"state": {"paused": False}}

    def write_status_mirrors(self, registry):
        return None

    def list_running_runs(self):
        return [{"task_id": t} for t in self.running_tasks]


class _Registry:
    def __init__(self, tasks):
        self.tasks = tasks

    def due_tasks(self, now):
        return self.tasks


class _Runner:
    def __init__(self, busy_with=None):
        self.started = []
        self.busy_with = busy_with  # task id occupying the single worker

    def start_task(self, task_id, trigger="scheduled"):
        if self.busy_with is not None and self.busy_with != task_id:
            raise RuntimeError("tasker worker is busy")
        self.started.append(task_id)
        return {"run_id": f"run-{task_id}"}

    def reconcile_orphaned_runs(self, *, now=None):
        return []


def _tick(svc, minute="2026-08-14T10:00:00"):
    svc.tick(datetime.fromisoformat(minute).replace(tzinfo=timezone.utc))


def test_own_in_flight_run_skips_quietly(caplog):
    """A task whose own run is in flight skips at DEBUG, never WARNING."""
    reg = _Registry([_Def("portfolio-lab-data")])
    store = _Store(running_tasks=["portfolio-lab-data"])
    runner = _Runner()
    svc = service.TaskerService(registry=reg, store=store, runner=runner)

    with caplog.at_level(logging.DEBUG, logger="src.tasker.service"):
        _tick(svc)
        _tick(svc)  # same due minute: no repeat

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == []
    assert runner.started == []
    assert "own in-flight run still active" in caplog.text
    # Skipped once per due minute, not once per 15s tick.
    assert caplog.text.count("own in-flight run still active") == 1


def test_busy_other_task_warns_at_most_once_per_due_minute(caplog):
    """A worker busy with a DIFFERENT task warns once, then stays quiet."""
    reg = _Registry([_Def("portfolio-lab-garch-risk")])
    store = _Store(running_tasks=[])
    runner = _Runner(busy_with="portfolio-lab-health")  # other task owns worker
    svc = service.TaskerService(registry=reg, store=store, runner=runner)

    with caplog.at_level(logging.WARNING, logger="src.tasker.service"):
        _tick(svc)
        _tick(svc)
        _tick(svc)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "tasker worker is busy" in warnings[0].message
    assert runner.started == []  # still attempting every tick (worker may free)


def test_busy_other_task_starts_when_worker_frees(caplog):
    """Blocked task still starts later in the minute when the worker frees."""
    reg = _Registry([_Def("portfolio-lab-attribution")])
    store = _Store(running_tasks=[])
    runner = _Runner(busy_with="portfolio-lab-health")
    svc = service.TaskerService(registry=reg, store=store, runner=runner)

    with caplog.at_level(logging.WARNING, logger="src.tasker.service"):
        _tick(svc)
    assert runner.started == []
    runner.busy_with = None
    with caplog.at_level(logging.WARNING, logger="src.tasker.service"):
        _tick(svc)
    assert runner.started == ["portfolio-lab-attribution"]
    # No second warning after the successful start.
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1


def test_unblocked_task_fires_normally(caplog):
    """No running rows, free worker: unchanged fire path, no warnings."""
    reg = _Registry([_Def("portfolio-lab-health")])
    store = _Store(running_tasks=[])
    runner = _Runner()
    svc = service.TaskerService(registry=reg, store=store, runner=runner)

    with caplog.at_level(logging.WARNING, logger="src.tasker.service"):
        _tick(svc)
    assert runner.started == ["portfolio-lab-health"]
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
