"""Flask API for Portfolio Lab tasker."""

from __future__ import annotations

import os
from typing import Any

from flask import Flask, abort, jsonify, request

from src.tasker.registry import TaskRegistry, load_task_registry
from src.tasker.runner import TaskRunner
from src.tasker.store import TaskerStore


def create_app(
    registry: TaskRegistry | None = None,
    store: TaskerStore | None = None,
    runner: Any | None = None,
    admin_token: str | None = None,
) -> Flask:
    registry = registry or load_task_registry()
    store = store or TaskerStore()
    store.sync_registry(registry)
    runner = runner or TaskRunner(registry=registry, store=store)
    admin_token = admin_token if admin_token is not None else os.environ.get("TASKER_ADMIN_TOKEN")

    app = Flask(__name__)

    def require_admin() -> None:
        token = request.headers.get("X-Tasker-Token")
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth.removeprefix("Bearer ").strip()
        if not admin_token or token != admin_token:
            abort(403)

    @app.get("/api/tasker/status")
    def tasker_status():
        return jsonify(store.status_payload(registry))

    @app.get("/api/tasks")
    def tasks():
        return jsonify({"tasks": store.status_payload(registry)["tasks"]})

    @app.get("/api/tasks/<task_id>")
    def task_detail(task_id: str):
        return jsonify(store.get_task(task_id, registry))

    @app.get("/api/runs")
    def runs():
        task_id = request.args.get("task_id")
        limit = int(request.args.get("limit", "50"))
        return jsonify({"runs": store.list_runs(task_id=task_id, limit=limit)})

    @app.get("/api/runs/<run_id>")
    def run_detail(run_id: str):
        return jsonify(store.get_run(run_id))

    @app.get("/api/runs/<run_id>/logs")
    def run_logs(run_id: str):
        tail = int(request.args.get("tail", "200"))
        return jsonify({"run_id": run_id, "logs": store.read_run_logs(run_id, tail=tail)})

    @app.post("/api/tasks/<task_id>/run")
    def run_task(task_id: str):
        require_admin()
        run = runner.start_task(task_id, trigger="manual")
        return jsonify(run), 202

    @app.post("/api/tasks/<task_id>/pause")
    def pause_task(task_id: str):
        require_admin()
        payload = request.get_json(silent=True) or {}
        store.set_task_paused(task_id, paused=True, reason=payload.get("reason"))
        store.write_status_mirrors(registry)
        return jsonify(store.get_task(task_id, registry))

    @app.post("/api/tasks/<task_id>/resume")
    def resume_task(task_id: str):
        require_admin()
        store.set_task_paused(task_id, paused=False)
        store.write_status_mirrors(registry)
        return jsonify(store.get_task(task_id, registry))

    @app.post("/api/runs/<run_id>/cancel")
    def cancel_run(run_id: str):
        require_admin()
        runner.cancel_run(run_id)
        return jsonify({"run_id": run_id, "cancel_requested": True}), 202

    @app.post("/api/runs/<run_id>/retry")
    def retry_run(run_id: str):
        require_admin()
        existing = store.get_run(run_id)
        run = runner.start_task(existing["task_id"], trigger="retry", retry_of=run_id)
        return jsonify(run), 202

    return app
