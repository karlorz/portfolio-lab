"""TASKER-HARDENING s1: single-instance flock guard for the tasker service."""

import os
from pathlib import Path

import pytest

from src.tasker import service


class _FakeStore:
    def __init__(self):
        self.wrote = False

    def write_status_mirrors(self, registry):
        self.wrote = True
        return {"backend": "tasker", "tasks": []}


class _FakeService:
    def __init__(self):
        self.registry = object()
        self.store = _FakeStore()


class _FakeApp:
    def run(self, host, port):
        raise AssertionError("app must never run when the lock guard fires")


def test_second_acquire_raises_system_exit_while_first_holds(tmp_path):
    """Only one instance may hold the singleton lock (flock LOCK_EX|LOCK_NB)."""
    lock = tmp_path / "tasker.lock"
    service.acquire_singleton_lock(lock_path=lock)

    with pytest.raises(SystemExit) as exc:
        service.acquire_singleton_lock(lock_path=lock)
    assert exc.value.code == 1


def test_lock_records_holder_pid(tmp_path):
    lock = tmp_path / "tasker.lock"
    service.acquire_singleton_lock(lock_path=lock)
    assert lock.read_text().strip() == str(os.getpid())


def test_lock_auto_releases_when_fd_closes(tmp_path):
    """flock releases on fd close — simulates process death (incl. SIGKILL)."""
    lock = tmp_path / "tasker.lock"
    service.acquire_singleton_lock(lock_path=lock)
    service._SINGLETON_LOCK_FD.close()
    service._SINGLETON_LOCK_FD = None
    # A restarted instance can re-acquire immediately: no stale-lock window.
    service.acquire_singleton_lock(lock_path=lock)


def test_main_exits_1_without_build_service_when_lock_held(monkeypatch, tmp_path):
    """A second service exits 1 before build_service (no mirrors/reconcile)."""
    monkeypatch.setattr(service, "TASKER_LOCK_PATH", tmp_path / "tasker.lock")
    monkeypatch.setattr(service, "configure_logging", lambda: None)
    service.acquire_singleton_lock()

    def _boom():
        raise AssertionError("build_service must not run when the lock is held")

    monkeypatch.setattr(service, "build_service", _boom)
    assert service.main([]) == 1


def test_main_once_mode_ignores_singleton_lock(monkeypatch, tmp_path):
    """--once mirror-refresh helpers must run alongside the live service."""
    monkeypatch.setattr(service, "TASKER_LOCK_PATH", tmp_path / "tasker.lock")
    monkeypatch.setattr(service, "configure_logging", lambda: None)
    service.acquire_singleton_lock()

    fake_service = _FakeService()
    fake_app = _FakeApp()
    monkeypatch.setattr(service, "build_service", lambda: (fake_service, fake_app))
    assert service.main(["--once"]) == 0


def test_no_scheduler_uses_same_lock_as_scheduler(tmp_path, monkeypatch):
    """API-only and scheduler processes share data/tasker.lock."""
    monkeypatch.setattr(service, "TASKER_LOCK_PATH", tmp_path / "tasker.lock")
    sched = tmp_path / "tasker.lock"
    service.acquire_singleton_lock()
    assert sched.is_file()
    assert not (tmp_path / "tasker-side.lock").exists()

    service._SINGLETON_LOCK_FD.close()
    service._SINGLETON_LOCK_FD = None
    service.acquire_singleton_lock()
    assert sched.read_text().strip() == str(os.getpid())
    assert not (tmp_path / "tasker-side.lock").exists()


def test_second_service_contends_for_the_same_lock(tmp_path, monkeypatch):
    """A second service in this checkout must not open the same TASKER_DB."""
    monkeypatch.setattr(service, "TASKER_LOCK_PATH", tmp_path / "tasker.lock")
    service.acquire_singleton_lock()
    held = service._SINGLETON_LOCK_FD
    service._SINGLETON_LOCK_FD = None
    with pytest.raises(SystemExit):
        service.acquire_singleton_lock()
    held.close()
    assert not (tmp_path / "tasker-side.lock").exists()


def test_env_tasker_lock_path_overrides_default(tmp_path, monkeypatch):
    custom = tmp_path / "custom.lock"
    monkeypatch.setenv("TASKER_LOCK_PATH", str(custom))
    monkeypatch.setattr(service, "TASKER_LOCK_PATH", tmp_path / "tasker.lock")
    assert service.resolve_tasker_lock_path() == custom


def test_main_refuses_prod_sidecar_without_override(monkeypatch, tmp_path):
    """API-only Tasker must not start from the production app dir."""
    monkeypatch.setattr(service, "PROJECT_ROOT", Path("/home/box/.local/share/portfolio-lab/app"))
    monkeypatch.setattr(service, "is_production_app_root", lambda root=None: True)
    monkeypatch.setattr(service, "configure_logging", lambda: None)

    def _boom():
        raise AssertionError("build_service must not run for a prod sidecar")

    monkeypatch.setattr(service, "build_service", _boom)
    monkeypatch.delenv("PORTFOLIO_LAB_ALLOW_PROD_SIDECAR", raising=False)
    assert service.main(["--no-scheduler"]) == 1


def test_scheduler_disabled_reads_flag_and_env(monkeypatch):
    monkeypatch.delenv("TASKER_DISABLE_SCHEDULER", raising=False)
    args = service._parse_args(["--no-scheduler"])
    assert service.scheduler_disabled(args) is True
    args = service._parse_args([])
    assert service.scheduler_disabled(args) is False
    monkeypatch.setenv("TASKER_DISABLE_SCHEDULER", "1")
    assert service.scheduler_disabled(args) is True


def test_is_production_app_root_matches_known_app_dirs():
    assert service.is_production_app_root(Path("/home/box/.local/share/portfolio-lab/app"))
    assert service.is_production_app_root(Path("/root/projects/portfolio-lab"))
    assert service.is_production_app_root(Path("/root/projects/portfolio-lab/src"))
    assert not service.is_production_app_root(Path("/root/projects/portfolio-lab-preview"))
    assert not service.is_production_app_root(Path("/home/box/code/portfolio-lab"))
