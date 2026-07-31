import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.tasker.models import RUN_ERROR, RUN_SUCCESS, TaskDefinition
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


def test_store_finish_run_is_idempotent_for_terminal_rows(tmp_path):
    registry = _registry()
    store = _store(tmp_path)
    store.sync_registry(registry)

    run = store.create_run("portfolio-lab-health", ["make", "health"], trigger="service-recovery")
    store.mark_run_running(run["run_id"], pid=99999999)

    assert store.finish_run(
        run["run_id"],
        status=RUN_ERROR,
        exit_code=None,
        duration_seconds=61.5,
        error="orphaned_run: process is no longer alive",
    ) is True
    first = store.get_run(run["run_id"])
    first_state = store.get_task("portfolio-lab-health", registry)["state"]

    # A runner thread and the service reconciler may race. The first terminal
    # transition wins, and a repeated call must not overwrite the result or
    # increment task health a second time.
    assert store.finish_run(
        run["run_id"],
        status=RUN_SUCCESS,
        exit_code=0,
        duration_seconds=0.1,
    ) is False
    second = store.get_run(run["run_id"])
    second_state = store.get_task("portfolio-lab-health", registry)["state"]

    assert second["status"] == RUN_ERROR
    assert second["error"] == first["error"]
    assert second["finished_at"] == first["finished_at"]
    assert second_state["failure_count"] == first_state["failure_count"] == 1
    assert second_state["consecutive_failures"] == first_state["consecutive_failures"] == 1


def test_store_status_mirrors_expose_terminal_reconciled_run(tmp_path):
    registry = _registry()
    store = _store(tmp_path)
    store.sync_registry(registry)
    run = store.create_run("portfolio-lab-health", ["make", "health"], trigger="service-recovery")
    old_started_at = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    store.mark_run_running(run["run_id"], pid=99999999, started_at=old_started_at)
    store.finish_run(
        run["run_id"],
        status=RUN_ERROR,
        exit_code=None,
        duration_seconds=300.0,
        error="orphaned_run: dead pid after grace",
    )

    payload = store.write_status_mirrors(registry)
    task = next(item for item in payload["tasks"] if item["id"] == "portfolio-lab-health")
    recent = next(item for item in payload["recent_runs"] if item["run_id"] == run["run_id"])
    cron = json.loads((tmp_path / "data" / "cron_status.json").read_text(encoding="utf-8"))
    cron_task = next(item for item in cron["jobs"] if item["name"] == "portfolio-lab-health")

    assert task["last_status"] == RUN_ERROR
    assert task["last_run_id"] == run["run_id"]
    assert recent["status"] == RUN_ERROR
    assert recent["finished_at"] is not None
    assert cron_task["status"] == RUN_ERROR
    assert cron_task["last_run"] == recent["finished_at"]


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
    by_name = {job["name"]: job for job in cron_status["jobs"]}
    assert by_name["portfolio-lab-build"]["enabled"] is False
    assert by_name["portfolio-lab-build"]["manual_only"] is True
    assert by_name["portfolio-lab-build"]["state"] == "manual_only"
    assert by_name["portfolio-lab-build"]["status"] == "disabled"
    assert by_name["portfolio-lab-build"]["last_run"] is None


# ── prune_runs() — per-task run-log retention ─────────────────────────


def _seed_runs(store, task_id, n, *, write_log_bytes=128):
    """Create n finished runs for task_id, each with a real .log file on disk."""
    run_ids = []
    for _ in range(n):
        run = store.create_run(task_id, ["make", task_id.split("-")[-1]], trigger="manual")
        store.finish_run(run["run_id"], status="success", exit_code=0, duration_seconds=0.1)
        # write real bytes so bytes_freed and unlink are exercised
        Path(run["log_path"]).write_bytes(b"x" * write_log_bytes)
        run_ids.append(run["run_id"])
    return run_ids


def test_prune_runs_keeps_last_n_per_task(tmp_path):
    store = _store(tmp_path)
    registry = _registry()
    store.sync_registry(registry)

    run_ids = _seed_runs(store, "portfolio-lab-health", 30)

    summary = store.prune_runs(keep_per_task=20)

    remaining_files = list((tmp_path / "logs").glob("*.log"))
    assert len(remaining_files) == 20
    remaining_rows = store.list_runs("portfolio-lab-health", limit=500)
    assert [r["run_id"] for r in remaining_rows] == list(reversed(run_ids[-20:]))
    assert summary["deleted_files"] == 10
    assert summary["deleted_rows"] == 10
    assert summary["kept_files"] == 20


def test_prune_runs_preserves_newest(tmp_path):
    store = _store(tmp_path)
    store.sync_registry(_registry())

    run_ids = _seed_runs(store, "portfolio-lab-health", 25)

    store.prune_runs(keep_per_task=20)

    remaining = [r["run_id"] for r in store.list_runs("portfolio-lab-health", limit=500)]
    # newest 20 = last 20 created, in DESC order
    assert remaining == list(reversed(run_ids[-20:]))
    # oldest 5 are gone (files)
    for old_id in run_ids[:5]:
        assert not (store.log_dir / f"{old_id}.log").exists()


def test_prune_runs_respects_keep_per_task_boundary(tmp_path):
    store = _store(tmp_path)
    store.sync_registry(_registry())

    # 25 runs, keep 20 -> 20 remain
    _seed_runs(store, "portfolio-lab-health", 25)
    store.prune_runs(keep_per_task=20)
    assert len(list((tmp_path / "logs").glob("*.log"))) == 20
    assert len(store.list_runs("portfolio-lab-health", limit=500)) == 20

    # fresh store with 15 runs, keep 20 -> 15 remain (no over-deletion)
    store2 = _store(tmp_path / "second")
    store2.sync_registry(_registry())
    _seed_runs(store2, "portfolio-lab-health", 15)
    summary = store2.prune_runs(keep_per_task=20)
    assert len(list((tmp_path / "second" / "logs").glob("*.log"))) == 15
    assert summary["deleted_files"] == 0
    assert summary["kept_files"] == 15


def test_prune_runs_across_multiple_tasks(tmp_path):
    store = _store(tmp_path)
    store.sync_registry(_registry())

    _seed_runs(store, "portfolio-lab-health", 30)
    _seed_runs(store, "portfolio-lab-build", 10)

    store.prune_runs(keep_per_task=20)

    # per-task, not global: health keeps 20, build keeps all 10
    assert len(store.list_runs("portfolio-lab-health", limit=500)) == 20
    assert len(store.list_runs("portfolio-lab-build", limit=500)) == 10
    assert len(list((tmp_path / "logs").glob("*.log"))) == 30


def test_prune_run_skips_missing_log_file(tmp_path):
    store = _store(tmp_path)
    store.sync_registry(_registry())

    run_ids = _seed_runs(store, "portfolio-lab-health", 25)
    # delete one log file manually to create an orphan DB row
    orphan_id = run_ids[0]
    Path(store.log_dir / f"{orphan_id}.log").unlink()

    summary = store.prune_runs(keep_per_task=20)

    # must not raise; orphan row deleted and reported in errors
    assert orphan_id not in [r["run_id"] for r in store.list_runs("portfolio-lab-health", limit=500)]
    assert summary["deleted_rows"] == 5  # 4 normal + 1 orphan
    assert summary["deleted_files"] == 4  # orphan file already gone
    assert len(summary["errors"]) == 1
    assert orphan_id in summary["errors"][0]["run_id"]


def test_prune_runs_dry_run_deletes_nothing(tmp_path):
    store = _store(tmp_path)
    store.sync_registry(_registry())

    _seed_runs(store, "portfolio-lab-health", 30)

    summary = store.prune_runs(keep_per_task=20, dry_run=True)

    # plan returned, but nothing touched
    assert len(list((tmp_path / "logs").glob("*.log"))) == 30
    assert len(store.list_runs("portfolio-lab-health", limit=500)) == 30
    assert summary["deleted_files"] == 0
    assert summary["deleted_rows"] == 0
    assert summary["kept_files"] == 30
    assert len(summary["plan"]) == 10  # the 10 that WOULD be deleted
    assert summary["bytes_freed"] > 0


def test_prune_runs_returns_summary(tmp_path):
    store = _store(tmp_path)
    store.sync_registry(_registry())

    _seed_runs(store, "portfolio-lab-health", 25, write_log_bytes=256)

    summary = store.prune_runs(keep_per_task=20)

    assert set(summary.keys()) == {
        "deleted_files",
        "deleted_rows",
        "kept_files",
        "bytes_freed",
        "errors",
        "plan",
    }
    assert summary["deleted_files"] == 5
    assert summary["deleted_rows"] == 5
    assert summary["kept_files"] == 20
    assert summary["bytes_freed"] == 5 * 256
    assert summary["errors"] == []
    assert summary["plan"] == []  # real run, not dry-run -> no plan entries


def test_prune_runs_never_touches_state_files(tmp_path):
    store = _store(tmp_path)
    registry = _registry()
    store.sync_registry(registry)

    _seed_runs(store, "portfolio-lab-health", 25)
    # write the state mirrors so they exist on disk
    store.write_status_mirrors(registry)
    cron_before = (tmp_path / "data" / "cron_status.json").read_bytes()
    status_before = (tmp_path / "public" / "tasker_status.json").read_bytes()

    store.prune_runs(keep_per_task=20)

    # cron_status.json + tasker_status.json untouched by pruning
    assert (tmp_path / "data" / "cron_status.json").read_bytes() == cron_before
    assert (tmp_path / "public" / "tasker_status.json").read_bytes() == status_before
    # no json/jsonl anywhere in logs dir
    assert list((tmp_path / "logs").glob("*.json")) == []
    assert list((tmp_path / "logs").glob("*.jsonl")) == []
