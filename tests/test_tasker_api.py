from src.tasker.api import create_app
from src.tasker.models import TaskDefinition
from src.tasker.registry import TaskRegistry
from src.tasker.store import TaskerStore


class StubRunner:
    def __init__(self):
        self.started = []
        self.cancelled = []

    def start_task(self, task_id: str, trigger: str = "manual", retry_of: str | None = None):
        self.started.append({"task_id": task_id, "trigger": trigger, "retry_of": retry_of})
        return {"run_id": f"run-{len(self.started)}", "task_id": task_id, "status": "pending"}

    def cancel_run(self, run_id: str):
        self.cancelled.append(run_id)
        return True


def _registry() -> TaskRegistry:
    return TaskRegistry(
        [
            TaskDefinition(
                id="portfolio-lab-health",
                label="Health",
                command=["make", "health"],
                schedule="0,30 * * * *",
                timeout_seconds=60,
            )
        ]
    )


def _store(tmp_path) -> TaskerStore:
    return TaskerStore(
        db_path=tmp_path / "tasker.db",
        public_status_path=tmp_path / "public" / "tasker_status.json",
        cron_status_path=tmp_path / "data" / "cron_status.json",
        log_dir=tmp_path / "logs",
    )


def _client(tmp_path):
    registry = _registry()
    store = _store(tmp_path)
    store.sync_registry(registry)
    runner = StubRunner()
    app = create_app(registry=registry, store=store, runner=runner, admin_token="secret")
    return app.test_client(), store, runner


def test_api_serves_tasker_status_tasks_and_task_detail(tmp_path):
    client, _, _ = _client(tmp_path)

    status = client.get("/api/tasker/status")
    tasks = client.get("/api/tasks")
    task = client.get("/api/tasks/portfolio-lab-health")

    assert status.status_code == 200
    assert status.get_json()["backend"] == "tasker"
    assert tasks.status_code == 200
    assert tasks.get_json()["tasks"][0]["id"] == "portfolio-lab-health"
    assert task.status_code == 200
    assert task.get_json()["definition"]["command"] == ["make", "health"]


def test_api_projects_private_run_paths_to_logical_references(tmp_path):
    client, store, _ = _client(tmp_path)
    run = store.create_run("portfolio-lab-health", ["make", "health"], trigger="manual")

    status = client.get("/api/tasker/status")
    runs = client.get("/api/runs")
    detail = client.get(f"/api/runs/{run['run_id']}")

    for response in (status, runs, detail):
        assert response.status_code == 200
        assert str(tmp_path) not in response.get_data(as_text=True)

    assert runs.get_json()["runs"][0]["log_path"] == f"internal/{run['run_id']}.log"
    assert detail.get_json()["log_path"] == f"internal/{run['run_id']}.log"


def test_api_rejects_mutations_without_admin_token(tmp_path):
    client, _, _ = _client(tmp_path)

    assert client.post("/api/tasks/portfolio-lab-health/run").status_code == 403
    assert client.post("/api/tasks/portfolio-lab-health/pause").status_code == 403
    assert client.post("/api/runs/run-1/cancel").status_code == 403


def test_api_accepts_token_for_run_pause_resume_cancel_and_retry(tmp_path):
    client, store, runner = _client(tmp_path)
    headers = {"X-Tasker-Token": "secret"}

    run_response = client.post("/api/tasks/portfolio-lab-health/run", json={"command": "rm -rf /"}, headers=headers)
    pause_response = client.post("/api/tasks/portfolio-lab-health/pause", json={"reason": "maintenance"}, headers=headers)
    paused_state = store.get_task("portfolio-lab-health", _registry())["state"]["paused"]
    resume_response = client.post("/api/tasks/portfolio-lab-health/resume", headers=headers)
    cancel_response = client.post("/api/runs/run-1/cancel", headers=headers)

    existing = store.create_run("portfolio-lab-health", ["make", "health"], trigger="manual")
    store.finish_run(existing["run_id"], status="error", exit_code=1, duration_seconds=0.1)
    retry_response = client.post(f"/api/runs/{existing['run_id']}/retry", headers={"Authorization": "Bearer secret"})

    assert run_response.status_code == 202
    assert runner.started[0] == {"task_id": "portfolio-lab-health", "trigger": "manual", "retry_of": None}
    assert "command" not in run_response.get_json()
    assert pause_response.status_code == 200
    assert paused_state is True
    assert resume_response.status_code == 200
    assert store.get_task("portfolio-lab-health", _registry())["state"]["paused"] is False
    assert cancel_response.status_code == 202
    assert runner.cancelled == ["run-1"]
    assert retry_response.status_code == 202
    assert runner.started[-1] == {
        "task_id": "portfolio-lab-health",
        "trigger": "retry",
        "retry_of": existing["run_id"],
    }


def test_api_has_no_arbitrary_command_endpoint(tmp_path):
    client, _, _ = _client(tmp_path)

    assert client.post("/api/tasker/command", json={"command": "make health"}, headers={"X-Tasker-Token": "secret"}).status_code == 404
