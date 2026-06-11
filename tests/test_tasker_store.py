import json

from src.tasker.models import TaskDefinition
from src.tasker.registry import TaskRegistry
from src.tasker.store import TaskerStore


def _registry() -> TaskRegistry:
    return TaskRegistry(
        [
            TaskDefinition(
                id="portfolio-lab-health",
                label="Health",
                command=["make", "health"],
                schedule="0,30 * * * *",
                timeout_seconds=60,
            ),
            TaskDefinition(
                id="portfolio-lab-build",
                label="Build",
                command=["make", "build"],
                schedule=None,
                enabled=False,
                manual_only=True,
                timeout_seconds=600,
            ),
        ]
    )


def _store(tmp_path) -> TaskerStore:
    return TaskerStore(
        db_path=tmp_path / "tasker.db",
        public_status_path=tmp_path / "public" / "tasker_status.json",
        cron_status_path=tmp_path / "data" / "cron_status.json",
        log_dir=tmp_path / "logs",
    )


def test_store_syncs_registry_and_persists_pause_state(tmp_path):
    registry = _registry()
    store = _store(tmp_path)

    store.sync_registry(registry)
    task = store.get_task("portfolio-lab-health", registry)
    assert task["state"]["paused"] is False
    assert task["definition"]["enabled"] is True

    store.set_task_paused("portfolio-lab-health", paused=True, reason="migration")
    paused = store.get_task("portfolio-lab-health", registry)
    assert paused["state"]["paused"] is True
    assert paused["state"]["pause_reason"] == "migration"

    store.set_task_paused("portfolio-lab-health", paused=False)
    resumed = store.get_task("portfolio-lab-health", registry)
    assert resumed["state"]["paused"] is False
    assert resumed["state"]["pause_reason"] is None


def test_store_tracks_run_lifecycle_and_health_failures(tmp_path):
    registry = _registry()
    store = _store(tmp_path)
    store.sync_registry(registry)

    failed = store.create_run("portfolio-lab-health", ["make", "health"], trigger="manual")
    store.mark_run_running(failed["run_id"], pid=123)
    store.finish_run(failed["run_id"], status="error", exit_code=1, duration_seconds=2.5, error="boom")

    after_failure = store.get_task("portfolio-lab-health", registry)
    assert after_failure["state"]["last_status"] == "error"
    assert after_failure["state"]["failure_count"] == 1
    assert after_failure["state"]["consecutive_failures"] == 1

    cancelled = store.create_run("portfolio-lab-health", ["make", "health"], trigger="manual")
    store.mark_run_running(cancelled["run_id"], pid=456)
    store.finish_run(cancelled["run_id"], status="cancelled", exit_code=-15, duration_seconds=0.2)

    after_cancel = store.get_task("portfolio-lab-health", registry)
    assert after_cancel["state"]["last_status"] == "cancelled"
    assert after_cancel["state"]["failure_count"] == 1
    assert after_cancel["state"]["consecutive_failures"] == 1

    recovered = store.create_run("portfolio-lab-health", ["make", "health"], trigger="manual")
    store.mark_run_running(recovered["run_id"], pid=789)
    store.finish_run(recovered["run_id"], status="success", exit_code=0, duration_seconds=0.1)

    after_success = store.get_task("portfolio-lab-health", registry)
    assert after_success["state"]["last_status"] == "success"
    assert after_success["state"]["failure_count"] == 1
    assert after_success["state"]["consecutive_failures"] == 0


def test_store_lists_recent_runs_with_limit(tmp_path):
    store = _store(tmp_path)
    registry = _registry()
    store.sync_registry(registry)

    run_ids = []
    for _ in range(5):
        run = store.create_run("portfolio-lab-health", ["make", "health"], trigger="manual")
        store.finish_run(run["run_id"], status="success", exit_code=0, duration_seconds=0.1)
        run_ids.append(run["run_id"])

    recent = store.list_runs(limit=2)

    assert [run["run_id"] for run in recent] == list(reversed(run_ids[-2:]))


def test_store_writes_tasker_and_cron_compatibility_mirrors(tmp_path):
    registry = _registry()
    store = _store(tmp_path)
    store.sync_registry(registry)
    run = store.create_run("portfolio-lab-health", ["make", "health"], trigger="manual")
    store.finish_run(run["run_id"], status="success", exit_code=0, duration_seconds=0.1)

    status = store.write_status_mirrors(registry)

    assert status["backend"] == "tasker"
    assert status["tasks"][0]["id"] == "portfolio-lab-health"
    assert (tmp_path / "public" / "tasker_status.json").exists()
    assert (tmp_path / "data" / "cron_status.json").exists()

    tasker_status = json.loads((tmp_path / "public" / "tasker_status.json").read_text(encoding="utf-8"))
    cron_status = json.loads((tmp_path / "data" / "cron_status.json").read_text(encoding="utf-8"))
    assert tasker_status["service"] == "portfolio-lab-tasker"
    assert cron_status["jobs"][0]["name"] == "portfolio-lab-health"
    assert cron_status["jobs"][0]["backend"] == "tasker"
