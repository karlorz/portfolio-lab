"""TASKER-HARDENING s1: single-instance flock guard for the tasker service."""

import os

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
