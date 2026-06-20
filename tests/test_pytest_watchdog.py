"""Regression tests for the sg01 pytest watchdog helper."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WATCHDOG_PATH = PROJECT_ROOT / "scripts" / "pytest_watchdog.py"


def _load_watchdog():
    spec = importlib.util.spec_from_file_location("pytest_watchdog", WATCHDOG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_small_pytest_tmp_tree_with_low_inode_pressure_is_silent(tmp_path: Path) -> None:
    """A few thousand pytest temp files should not trigger noisy cleanup alerts."""
    wd = _load_watchdog()

    assert wd.should_cleanup_pytest_tmp(inode_count=9_494, inode_pct=1, min_inodes=50_000, pressure_pct=80) is False


def test_cleanup_removes_only_stale_pytest_dirs_when_threshold_is_hit(tmp_path: Path, monkeypatch) -> None:
    """Cleanup should avoid deleting active/recent pytest temp directories."""
    wd = _load_watchdog()
    root = tmp_path / "pytest-of-root"
    root.mkdir()

    stale = root / "pytest-1"
    stale.mkdir()
    (stale / "old.txt").write_text("old")

    recent = root / "pytest-2"
    recent.mkdir()
    (recent / "new.txt").write_text("new")

    current = root / "pytest-current"
    current.symlink_to(recent, target_is_directory=True)

    old_time = time.time() - 3 * 3600
    os.utime(stale, (old_time, old_time))

    monkeypatch.setattr(wd, "count_tree_entries", lambda *_args, **_kwargs: 75_000)
    monkeypatch.setattr(wd, "inode_usage_pct", lambda _path: 4)

    result = wd.cleanup_pytest_tmp(root, min_inodes=50_000, pressure_pct=80, stale_seconds=3600)

    assert result.removed == [stale]
    assert not stale.exists()
    assert recent.exists()
    assert current.is_symlink()


def test_xdist_like_worker_without_pytest_ancestor_is_not_killed() -> None:
    """The worker pattern is broad, so it must be scoped to pytest ancestry."""
    wd = _load_watchdog()
    worker = wd.ProcessInfo(
        pid=222,
        ppid=1,
        etime=900,
        cpu=95.0,
        rss_kb=100_000,
        args="/venv/bin/python -c import sys; exec(eval(sys.stdin.read()))",
    )

    assert wd.select_runaway_processes([worker], max_run_sec=300, cpu_threshold=30, rss_threshold_kb=500_000) == []


def test_pytest_xdist_worker_with_pytest_ancestor_is_selected() -> None:
    """A resource-heavy xdist worker should be killable when pytest spawned it."""
    wd = _load_watchdog()
    parent = wd.ProcessInfo(
        pid=100,
        ppid=1,
        etime=901,
        cpu=5.0,
        rss_kb=100_000,
        args="/repo/.venv/bin/python -m pytest tests/ -n auto",
    )
    worker = wd.ProcessInfo(
        pid=101,
        ppid=100,
        etime=901,
        cpu=95.0,
        rss_kb=100_000,
        args="/repo/.venv/bin/python -c import sys; exec(eval(sys.stdin.read()))",
    )

    decisions = wd.select_runaway_processes(
        [parent, worker],
        max_run_sec=300,
        cpu_threshold=30,
        rss_threshold_kb=500_000,
    )

    assert any(decision.process == worker and decision.role == "worker" for decision in decisions)


def test_portfolio_lab_pytest_parent_is_not_killed_for_normal_full_suite_runtime() -> None:
    """Long portfolio-lab test runs are bounded by make test's own timeout."""
    wd = _load_watchdog()
    parent = wd.ProcessInfo(
        pid=100,
        ppid=1,
        etime=900,
        cpu=80.0,
        rss_kb=120_000,
        args="/repo/.venv/bin/python -m pytest tests/",
        cwd="/root/projects/portfolio-lab",
    )

    decisions = wd.select_runaway_processes([parent], max_run_sec=300, cpu_threshold=30, rss_threshold_kb=500_000)

    assert decisions == []


def test_wrong_repo_pytest_parent_is_selected_when_cpu_heavy() -> None:
    """Wrong-repo pytest parents should still be killable when they burn CPU."""
    wd = _load_watchdog()
    parent = wd.ProcessInfo(
        pid=100,
        ppid=1,
        etime=901,
        cpu=80.0,
        rss_kb=120_000,
        args="/root/.hermes/hermes-agent/.venv/bin/python -m pytest tests/",
        cwd="/root/.hermes/hermes-agent",
    )

    decisions = wd.select_runaway_processes([parent], max_run_sec=300, cpu_threshold=30, rss_threshold_kb=500_000)

    assert len(decisions) == 1
    assert decisions[0].process == parent
    assert decisions[0].role == "parent"
    assert "runtime 901s + 80% CPU" in decisions[0].reason


def test_wrong_repo_pytest_parent_is_selected_when_memory_heavy() -> None:
    """Wrong-repo pytest parents should still be killable when they exceed RSS limits."""
    wd = _load_watchdog()
    parent = wd.ProcessInfo(
        pid=100,
        ppid=1,
        etime=901,
        cpu=4.0,
        rss_kb=600_000,
        args="/root/.hermes/hermes-agent/.venv/bin/python -m pytest tests/",
        cwd="/root/.hermes/hermes-agent",
    )

    decisions = wd.select_runaway_processes([parent], max_run_sec=300, cpu_threshold=30, rss_threshold_kb=500_000)

    assert len(decisions) == 1
    assert decisions[0].process == parent
    assert decisions[0].role == "parent"
    assert "600000KB RSS > 500000KB" in decisions[0].reason
