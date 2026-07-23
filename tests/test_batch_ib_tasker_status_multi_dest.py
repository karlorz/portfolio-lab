"""Batch IB: tasker_status.json multi-dest soft-mirror + 0o644 atomic writes.

Session A residual after Batch IA:
- Live WWW tasker_status advances on every tasker poll while repo
  ``public/data/tasker_status.json`` only refreshes on satellite
  ``mirror-repo-public-data`` soft-gates → continuous content churn / lag.
- ``TaskerStore.write_status_mirrors`` used bare ``write_text`` (no fchmod /
  no private twin / no repo soft-mirror).

Authority: never touches ``signals.json.target_allocations`` / order_router.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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
        ]
    )


def _store(
    tmp_path: Path,
    *,
    repo_status_path: Path | None = None,
    private_status_path: Path | None = None,
) -> TaskerStore:
    return TaskerStore(
        db_path=tmp_path / "tasker.db",
        public_status_path=tmp_path / "public" / "tasker_status.json",
        private_status_path=private_status_path
        if private_status_path is not None
        else tmp_path / "data" / "tasker_status.json",
        repo_status_path=repo_status_path
        if repo_status_path is not None
        else tmp_path / "repo_public" / "tasker_status.json",
        cron_status_path=tmp_path / "data" / "cron_status.json",
        log_dir=tmp_path / "logs",
    )


def test_tasker_status_multi_dest_same_bytes_and_0644(tmp_path: Path) -> None:
    """Case DN: public + private + repo soft-mirror share bytes and 0o644."""
    registry = _registry()
    store = _store(tmp_path)
    store.sync_registry(registry)
    run = store.create_run("portfolio-lab-health", ["make", "health"], trigger="manual")
    store.finish_run(run["run_id"], status="success", exit_code=0, duration_seconds=0.1)

    payload = store.write_status_mirrors(registry)

    public = tmp_path / "public" / "tasker_status.json"
    private = tmp_path / "data" / "tasker_status.json"
    repo = tmp_path / "repo_public" / "tasker_status.json"
    assert public.is_file()
    assert private.is_file()
    assert repo.is_file()

    pub_bytes = public.read_bytes()
    assert private.read_bytes() == pub_bytes
    assert repo.read_bytes() == pub_bytes
    assert (public.stat().st_mode & 0o777) == 0o644
    assert (private.stat().st_mode & 0o777) == 0o644
    assert (repo.stat().st_mode & 0o777) == 0o644

    body = json.loads(pub_bytes.decode("utf-8"))
    assert body["service"] == "portfolio-lab-tasker"
    assert body["backend"] == "tasker"
    assert body["timestamp"] == payload["timestamp"]
    assert body["tasks"][0]["id"] == "portfolio-lab-health"


def test_tasker_status_cron_mirror_stays_0644(tmp_path: Path) -> None:
    """Case DO: cron_status.json compatibility mirror is 0o644 atomic."""
    registry = _registry()
    store = _store(tmp_path)
    store.sync_registry(registry)
    store.write_status_mirrors(registry)

    cron = tmp_path / "data" / "cron_status.json"
    assert cron.is_file()
    assert (cron.stat().st_mode & 0o777) == 0o644
    jobs = json.loads(cron.read_text(encoding="utf-8"))["jobs"]
    assert jobs[0]["name"] == "portfolio-lab-health"
    assert jobs[0]["backend"] == "tasker"


def test_tasker_status_does_not_touch_target_allocations(tmp_path: Path) -> None:
    """Case DP: tasker multi-dest must not rewrite signals authority surface."""
    registry = _registry()
    signals = tmp_path / "public" / "signals.json"
    signals.parent.mkdir(parents=True, exist_ok=True)
    champion = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    signals.write_text(
        json.dumps({"target_allocations": champion, "health": {"status": "ok"}}),
        encoding="utf-8",
    )
    before = signals.read_bytes()

    store = _store(tmp_path)
    store.sync_registry(registry)
    store.write_status_mirrors(registry)

    assert signals.read_bytes() == before
    assert json.loads(before)["target_allocations"] == champion


def test_tasker_status_skips_production_ssot_under_pytest(tmp_path, monkeypatch) -> None:
    """Case DQ: under pytest, refuse live production paths without allow flag."""
    from src.monitor.signal_authority import is_production_ssot_path

    live = Path("/var/www/portfolio-lab/data/tasker_status.json")
    if not is_production_ssot_path(live):
        # Host without live www layout — still exercise guard helper path.
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_batch_ib::dq")
        registry = _registry()
        store = _store(tmp_path)
        store.sync_registry(registry)
        store.write_status_mirrors(registry)
        assert (tmp_path / "public" / "tasker_status.json").is_file()
        return

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_batch_ib::dq")
    monkeypatch.delenv("PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC", raising=False)
    before = live.read_bytes() if live.is_file() else None

    registry = _registry()
    store = TaskerStore(
        db_path=tmp_path / "tasker.db",
        public_status_path=live,
        private_status_path=tmp_path / "data" / "tasker_status.json",
        repo_status_path=tmp_path / "repo_public" / "tasker_status.json",
        cron_status_path=tmp_path / "data" / "cron_status.json",
        log_dir=tmp_path / "logs",
    )
    store.sync_registry(registry)
    store.write_status_mirrors(registry)

    if before is not None:
        assert live.read_bytes() == before
    # Non-production dests still land
    assert (tmp_path / "data" / "tasker_status.json").is_file()
    assert (tmp_path / "repo_public" / "tasker_status.json").is_file()
