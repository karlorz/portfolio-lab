import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest

from src.tasker.models import TaskDefinition
from src.tasker.registry import TaskRegistry
from src.tasker.runner import TaskRunner
from src.tasker.store import TaskerStore


def _store(tmp_path) -> TaskerStore:
    return TaskerStore(
        db_path=tmp_path / "tasker.db",
        public_status_path=tmp_path / "public" / "tasker_status.json",
        cron_status_path=tmp_path / "data" / "cron_status.json",
        log_dir=tmp_path / "logs",
    )


def _registry(command: list[str], timeout_seconds: int = 5) -> TaskRegistry:
    return TaskRegistry(
        [
            TaskDefinition(
                id="env-task",
                label="Env Task",
                command=command,
                schedule=None,
                timeout_seconds=timeout_seconds,
            )
        ],
        validate_commands=False,
    )


def test_runner_executes_registered_command_with_tasker_trace_environment(tmp_path):
    output_path = tmp_path / "env.json"
    command = [
        sys.executable,
        "-c",
        (
            "import json, os, pathlib; "
            "keys=['TASKER_RUN_ID','CRON_RUN_ID','CRON_BACKEND','PORTFOLIO_LAB_ENABLE_ML']; "
            "pathlib.Path(os.environ['TASKER_ENV_OUT']).write_text("
            "json.dumps({k: os.environ.get(k) for k in keys}), encoding='utf-8')"
        ),
    ]
    registry = _registry(command)
    store = _store(tmp_path)
    store.sync_registry(registry)
    runner = TaskRunner(registry=registry, store=store, project_root=tmp_path, base_env={"TASKER_ENV_OUT": str(output_path)})

    run = runner.start_task("env-task", trigger="manual")
    completed = runner.wait_for_run(run["run_id"], timeout_seconds=5)

    assert completed["status"] == "success"
    env = json.loads(output_path.read_text(encoding="utf-8"))
    assert env["TASKER_RUN_ID"] == run["run_id"]
    assert env["CRON_RUN_ID"] == run["run_id"]
    assert env["CRON_BACKEND"] == "tasker"
    assert env["PORTFOLIO_LAB_ENABLE_ML"] == "0"


def test_runner_serializes_execution_to_one_active_run(tmp_path):
    command = [sys.executable, "-c", "import time; time.sleep(0.4)"]
    registry = _registry(command)
    store = _store(tmp_path)
    store.sync_registry(registry)
    runner = TaskRunner(registry=registry, store=store, project_root=tmp_path)

    first = runner.start_task("env-task", trigger="manual")
    try:
        with pytest.raises(RuntimeError, match="worker is busy"):
            runner.start_task("env-task", trigger="manual")
    finally:
        runner.wait_for_run(first["run_id"], timeout_seconds=5)


def test_runner_cancels_active_process_without_incrementing_failure_health(tmp_path):
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    registry = _registry(command, timeout_seconds=60)
    store = _store(tmp_path)
    store.sync_registry(registry)
    runner = TaskRunner(registry=registry, store=store, project_root=tmp_path)

    run = runner.start_task("env-task", trigger="manual")
    deadline = time.time() + 5
    while time.time() < deadline and store.get_run(run["run_id"])["status"] != "running":
        time.sleep(0.01)

    assert runner.cancel_run(run["run_id"], grace_seconds=0.1) is True
    completed = runner.wait_for_run(run["run_id"], timeout_seconds=5)
    task = store.get_task("env-task", registry)

    assert completed["status"] == "cancelled"
    assert task["state"]["failure_count"] == 0
    assert task["state"]["consecutive_failures"] == 0


def test_runner_exit_2_is_error_for_non_eval_tasks(tmp_path):
    """Batch CE: make exit 2 (recipe failure) must not map to RUN_BLOCKED."""
    command = [sys.executable, "-c", "import sys; sys.exit(2)"]
    registry = _registry(command)
    # override task id to look like health (not intentional block list)
    registry = TaskRegistry(
        [
            TaskDefinition(
                id="portfolio-lab-health",
                label="Health",
                command=command,
                schedule=None,
                timeout_seconds=5,
            )
        ],
        validate_commands=False,
    )
    store = _store(tmp_path)
    store.sync_registry(registry)
    runner = TaskRunner(registry=registry, store=store, project_root=tmp_path)

    run = runner.start_task("portfolio-lab-health", trigger="manual")
    completed = runner.wait_for_run(run["run_id"], timeout_seconds=5)
    assert completed["status"] == "error"
    assert completed["exit_code"] == 2
    assert completed.get("error")
    assert "non-intentional-block" in (completed.get("error") or "")


def test_runner_exit_2_is_blocked_for_eval_task(tmp_path):
    """Eval may intentionally return EXIT_BLOCKED=2 under kill authority."""
    command = [sys.executable, "-c", "import sys; sys.exit(2)"]
    registry = TaskRegistry(
        [
            TaskDefinition(
                id="portfolio-lab-eval",
                label="Eval",
                command=command,
                schedule=None,
                timeout_seconds=5,
            )
        ],
        validate_commands=False,
    )
    store = _store(tmp_path)
    store.sync_registry(registry)
    runner = TaskRunner(registry=registry, store=store, project_root=tmp_path)

    run = runner.start_task("portfolio-lab-eval", trigger="manual")
    completed = runner.wait_for_run(run["run_id"], timeout_seconds=5)
    assert completed["status"] == "blocked"
    assert completed["exit_code"] == 2


def test_runner_clears_thread_after_completion(tmp_path):
    """Completed runs must not leak entries in _threads.

    Without cleanup, a finished-but-not-removed thread leaves stale state. If
    a later subprocess hangs (thread stays is_alive), the worker can report
    "busy" forever - the condition that froze the scheduler on 2026-07-19 and
    silently skipped the weekly fetch-trends job for weeks.
    """
    command = [sys.executable, "-c", "pass"]
    registry = _registry(command)
    store = _store(tmp_path)
    store.sync_registry(registry)
    runner = TaskRunner(registry=registry, store=store, project_root=tmp_path)

    run = runner.start_task("env-task", trigger="manual")
    runner.wait_for_run(run["run_id"], timeout_seconds=5)

    assert run["run_id"] not in runner._threads


def test_runner_clears_thread_after_timeout_kill(tmp_path):
    """A timed-out run must also clear its thread so a hung subprocess cannot
    permanently block the single worker slot."""
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    registry = _registry(command, timeout_seconds=1)
    store = _store(tmp_path)
    store.sync_registry(registry)
    runner = TaskRunner(registry=registry, store=store, project_root=tmp_path)

    run = runner.start_task("env-task", trigger="manual")
    completed = runner.wait_for_run(run["run_id"], timeout_seconds=5)
    assert completed["status"] == "timeout"
    assert run["run_id"] not in runner._threads


def test_runner_reconciles_only_an_old_dead_pid(tmp_path):
    registry = _registry([sys.executable, "-c", "pass"])
    store = _store(tmp_path)
    store.sync_registry(registry)
    runner = TaskRunner(registry=registry, store=store, project_root=tmp_path)

    run = store.create_run("env-task", registry.get("env-task").command, trigger="service-recovery")
    started_at = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    store.mark_run_running(run["run_id"], pid=99999999, started_at=started_at)

    reconciled = runner.reconcile_orphaned_runs(grace_seconds=30)
    completed = store.get_run(run["run_id"])
    task = store.get_task("env-task", registry)["state"]

    assert reconciled == [run["run_id"]]
    assert completed["status"] == "error"
    assert completed["finished_at"] is not None
    assert completed["duration_seconds"] >= 300
    assert "orphaned_run" in (completed["error"] or "")
    assert task["last_status"] == "error"
    assert task["last_run_id"] == run["run_id"]
    assert task["failure_count"] == 1
    assert task["consecutive_failures"] == 1
    assert runner.reconcile_orphaned_runs(grace_seconds=30) == []
    assert store.get_task("env-task", registry)["state"]["failure_count"] == 1


def test_runner_preserves_a_live_pid_even_after_grace(tmp_path):
    registry = _registry([sys.executable, "-c", "pass"])
    store = _store(tmp_path)
    store.sync_registry(registry)
    runner = TaskRunner(registry=registry, store=store, project_root=tmp_path)

    run = store.create_run("env-task", registry.get("env-task").command, trigger="service-recovery")
    started_at = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    store.mark_run_running(run["run_id"], pid=os.getpid(), started_at=started_at)

    assert runner.reconcile_orphaned_runs(grace_seconds=0) == []
    assert store.get_run(run["run_id"])["status"] == "running"


def test_runner_preserves_a_dead_pid_during_bounded_grace(tmp_path):
    registry = _registry([sys.executable, "-c", "pass"])
    store = _store(tmp_path)
    store.sync_registry(registry)
    runner = TaskRunner(registry=registry, store=store, project_root=tmp_path)

    run = store.create_run("env-task", registry.get("env-task").command, trigger="service-recovery")
    store.mark_run_running(run["run_id"], pid=99999999)

    assert runner.reconcile_orphaned_runs(grace_seconds=60) == []
    assert store.get_run(run["run_id"])["status"] == "running"
