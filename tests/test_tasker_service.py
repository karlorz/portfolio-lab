import sqlite3

from src.tasker import service


class FakeStore:
    def __init__(self):
        self.wrote = False

    def write_status_mirrors(self, registry):
        self.wrote = True
        return {"backend": "tasker", "tasks": []}


class FakeService:
    def __init__(self):
        self.registry = object()
        self.store = FakeStore()
        self.scheduler_started = False

    def start_background_scheduler(self):
        self.scheduler_started = True
        return None


class FakeApp:
    def __init__(self):
        self.ran = False

    def run(self, host, port):
        self.ran = True


def test_service_once_mode_writes_status_mirrors_without_starting_server(monkeypatch):
    fake_service = FakeService()
    fake_app = FakeApp()
    monkeypatch.setattr(service, "build_service", lambda: (fake_service, fake_app))
    monkeypatch.setattr(service, "configure_logging", lambda: None)

    exit_code = service.main(["--once"])

    assert exit_code == 0
    assert fake_service.store.wrote is True
    assert fake_service.scheduler_started is False
    assert fake_app.ran is False


class _Def:
    def __init__(self, task_id):
        self.id = task_id
        self.enabled = True
        self.manual_only = False
        self.schedule = "* * * * *"


class _RaisingStore:
    """Store whose get_task raises on the first task, succeeds on the second."""

    def __init__(self):
        self.calls = 0
        self.wrote_mirrors = False
        self.started = []

    def get_task(self, task_id, registry):
        self.calls += 1
        if task_id == "boom-task":
            raise sqlite3.OperationalError("database is locked")
        return {"state": {"paused": False}}

    def write_status_mirrors(self, registry):
        self.wrote_mirrors = True


class _RecordingRunner:
    def __init__(self):
        self.started = []
        self.reconciled = []

    def start_task(self, task_id, trigger="scheduled"):
        self.started.append(task_id)
        return {"run_id": f"run-{task_id}"}

    def reconcile_orphaned_runs(self, *, now=None):
        self.reconciled.append(now)
        return []


class _StubRegistry:
    def __init__(self, tasks):
        self.tasks = tasks

    def due_tasks(self, now):
        return self.tasks


def test_tick_swallows_store_error_so_scheduler_thread_survives():
    """A single task's DB error must not kill the scheduler loop.

    On 2026-07-19 the scheduler stopped firing after a store error during the
    prune-logs window; every subsequent job (including weekly fetch-trends) was
    silently skipped until a manual restart. tick() must isolate per-task
    failures and always finish by writing status mirrors.
    """
    from datetime import datetime, timezone

    reg = _StubRegistry([_Def("boom-task"), _Def("ok-task")])
    store = _RaisingStore()
    runner = _RecordingRunner()
    svc = service.TaskerService(registry=reg, store=store, runner=runner)

    svc.tick(datetime.now(timezone.utc))

    # boom-task raised in get_task; ok-task still fired.
    assert "ok-task" in runner.started
    # status mirrors still written even though one task errored.
    assert store.wrote_mirrors is True


def test_tick_reconciles_orphaned_runs_before_scheduling_and_mirroring():
    from datetime import datetime, timezone

    reg = _StubRegistry([])
    store = _RaisingStore()
    runner = _RecordingRunner()
    svc = service.TaskerService(registry=reg, store=store, runner=runner)
    now = datetime.now(timezone.utc)

    svc.tick(now)

    assert runner.reconciled == [now]
    assert store.wrote_mirrors is True
