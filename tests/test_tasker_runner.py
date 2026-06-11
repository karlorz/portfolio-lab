import json
import sys
import time

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
