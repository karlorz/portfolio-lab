#!/usr/bin/env python3
"""Pytest process and temporary-directory watchdog.

This is the repo-backed version of the sg01 Hermes ``pytest-watchdog`` helper.
It is intentionally conservative: process kills are scoped to pytest ancestry,
and pytest tmp cleanup is quiet until either inode pressure is high or the
pytest tmp tree is genuinely large.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_MAX_RUN_SEC = 300
DEFAULT_CPU_THRESHOLD = 30.0
DEFAULT_RSS_THRESHOLD_KB = 500_000
DEFAULT_TMP_DIR = Path("/tmp/pytest-of-root")
DEFAULT_MIN_TMP_INODES = 50_000
DEFAULT_INODE_PRESSURE_PCT = 80
DEFAULT_STALE_TMP_SECONDS = 3600
DEFAULT_LOG_PATH = Path("/root/.hermes/logs/python-monitor.log")
DEFAULT_LOG_MAX_BYTES = 5_000_000


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    etime: int
    cpu: float
    rss_kb: int
    args: str


@dataclass(frozen=True)
class KillDecision:
    process: ProcessInfo
    role: str
    reason: str


@dataclass
class CleanupResult:
    path: Path
    inode_count: int
    inode_pct: int
    triggered: bool
    removed: list[Path] = field(default_factory=list)


def env_int(name: str, default: int) -> int:
    """Read a positive integer env var with a safe fallback."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def is_pytest_parent(process: ProcessInfo) -> bool:
    """Return True for normal pytest parent commands."""
    args = f" {process.args} "
    return " -m pytest " in args or "/pytest " in args or " pytest " in args


def is_xdist_like_worker(process: ProcessInfo) -> bool:
    """Return True for pytest-xdist worker commands spawned via python -c."""
    args = process.args
    return "python" in args and "-c" in args and "exec" in args and "sys.stdin" in args


def has_pytest_ancestor(process: ProcessInfo, process_map: dict[int, ProcessInfo]) -> bool:
    """Return True when a process descends from a pytest parent."""
    seen: set[int] = set()
    current = process
    while current.ppid and current.ppid not in seen:
        seen.add(current.ppid)
        parent = process_map.get(current.ppid)
        if parent is None:
            return False
        if is_pytest_parent(parent):
            return True
        current = parent
    return False


def select_runaway_processes(
    processes: list[ProcessInfo],
    *,
    max_run_sec: int = DEFAULT_MAX_RUN_SEC,
    cpu_threshold: float = DEFAULT_CPU_THRESHOLD,
    rss_threshold_kb: int = DEFAULT_RSS_THRESHOLD_KB,
) -> list[KillDecision]:
    """Choose pytest-related processes that exceed runtime/resource limits."""
    process_map = {process.pid: process for process in processes}
    decisions: list[KillDecision] = []

    for process in processes:
        if is_pytest_parent(process) and process.etime > max_run_sec:
            decisions.append(
                KillDecision(
                    process=process,
                    role="parent",
                    reason=f"runtime {process.etime}s > {max_run_sec}s",
                )
            )
            continue

        if not is_xdist_like_worker(process):
            continue
        if not has_pytest_ancestor(process, process_map):
            continue

        reason = ""
        if process.etime > max_run_sec and process.cpu > cpu_threshold:
            reason = f"runtime {process.etime}s + {process.cpu:.0f}% CPU"
        elif process.rss_kb > rss_threshold_kb:
            reason = f"{process.rss_kb}KB RSS > {rss_threshold_kb}KB"

        if reason:
            decisions.append(KillDecision(process=process, role="worker", reason=reason))

    return decisions


def read_processes() -> list[ProcessInfo]:
    """Read a Linux process table snapshot."""
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,etimes=,pcpu=,rss=,args="],
        capture_output=True,
        text=True,
        check=False,
    )
    processes: list[ProcessInfo] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 5)
        if len(parts) < 6:
            continue
        pid_s, ppid_s, etime_s, cpu_s, rss_s, args = parts
        try:
            process = ProcessInfo(
                pid=int(pid_s),
                ppid=int(ppid_s),
                etime=int(float(etime_s)),
                cpu=float(cpu_s),
                rss_kb=int(float(rss_s)),
                args=args,
            )
        except ValueError:
            continue
        processes.append(process)
    return processes


def rotate_log(log_path: Path, max_bytes: int = DEFAULT_LOG_MAX_BYTES) -> None:
    """Rotate the monitor log before appending a new process snapshot."""
    if not log_path.exists():
        return
    try:
        if log_path.stat().st_size <= max_bytes:
            return
        old_path = log_path.with_suffix(log_path.suffix + ".old")
        log_path.replace(old_path)
    except OSError:
        return


def append_process_log(processes: list[ProcessInfo], log_path: Path = DEFAULT_LOG_PATH) -> None:
    """Append a lightweight Python-process snapshot for later diagnosis."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rotate_log(log_path)
        with log_path.open("a") as handle:
            handle.write(f"=== {time.strftime('%Y-%m-%d_%H:%M:%S')} ===\n")
            for process in processes:
                if not _looks_python_related(process.args):
                    continue
                rss_mb = process.rss_kb // 1024
                handle.write(
                    f"  {process.pid:<6} PPID={process.ppid:<6} "
                    f"{process.etime:>5}s {process.cpu:>5.1f}% CPU "
                    f"{rss_mb:>5}MB {process.args[:160]}\n"
                )
    except OSError:
        return


def _looks_python_related(args: str) -> bool:
    lowered = args.lower()
    return any(token in lowered for token in ("python", "pytest", "hermes", "uv run"))


def terminate_processes(decisions: list[KillDecision], *, dry_run: bool = False) -> list[KillDecision]:
    """Terminate selected processes, escalating to SIGKILL after a short grace period."""
    terminated: list[KillDecision] = []
    for decision in decisions:
        pid = decision.process.pid
        if dry_run:
            terminated.append(decision)
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            terminated.append(decision)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue

    if dry_run or not terminated:
        return terminated

    time.sleep(2)
    for decision in terminated:
        pid = decision.process.pid
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            continue
    return terminated


def count_tree_entries(root: Path, *, limit: int | None = None) -> int:
    """Count entries under a tree without following symlinks."""
    if not root.exists():
        return 0
    count = 1
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    count += 1
                    if limit is not None and count >= limit:
                        return count
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
        except OSError:
            continue
    return count


def inode_usage_pct(path: Path) -> int:
    """Return inode-use percentage for the filesystem containing path."""
    result = subprocess.run(["df", "-Pi", str(path)], capture_output=True, text=True, check=False)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return 0
    fields = lines[-1].split()
    if len(fields) < 5:
        return 0
    try:
        return int(fields[4].rstrip("%"))
    except ValueError:
        return 0


def should_cleanup_pytest_tmp(
    *,
    inode_count: int,
    inode_pct: int,
    min_inodes: int = DEFAULT_MIN_TMP_INODES,
    pressure_pct: int = DEFAULT_INODE_PRESSURE_PCT,
) -> bool:
    """Return True when pytest tmp cleanup is justified."""
    return inode_pct >= pressure_pct or inode_count >= min_inodes


def cleanup_pytest_tmp(
    path: Path = DEFAULT_TMP_DIR,
    *,
    min_inodes: int = DEFAULT_MIN_TMP_INODES,
    pressure_pct: int = DEFAULT_INODE_PRESSURE_PCT,
    stale_seconds: int = DEFAULT_STALE_TMP_SECONDS,
    dry_run: bool = False,
) -> CleanupResult:
    """Remove stale pytest tmp children when cleanup thresholds are crossed."""
    if not path.exists():
        return CleanupResult(path=path, inode_count=0, inode_pct=0, triggered=False)

    inode_count = count_tree_entries(path, limit=max(min_inodes, 1) + 1)
    inode_pct = inode_usage_pct(path)
    triggered = should_cleanup_pytest_tmp(
        inode_count=inode_count,
        inode_pct=inode_pct,
        min_inodes=min_inodes,
        pressure_pct=pressure_pct,
    )
    result = CleanupResult(path=path, inode_count=inode_count, inode_pct=inode_pct, triggered=triggered)
    if not triggered:
        return result

    now = time.time()
    try:
        children = sorted(path.iterdir())
    except OSError:
        return result

    for child in children:
        try:
            if child.is_symlink():
                continue
            if not child.is_dir():
                continue
            if now - child.stat().st_mtime < stale_seconds:
                continue
        except OSError:
            continue
        result.removed.append(child)
        if not dry_run:
            remove_tree(child)
    return result


def remove_tree(root: Path) -> None:
    """Remove a directory tree without following symlinks."""
    for current, dirs, files in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        for name in files:
            try:
                (current_path / name).unlink()
            except OSError:
                pass
        for name in dirs:
            path = current_path / name
            try:
                if path.is_symlink():
                    path.unlink()
                else:
                    path.rmdir()
            except OSError:
                pass
    try:
        root.rmdir()
    except OSError:
        pass


def format_report(killed: list[KillDecision], cleanup: CleanupResult) -> str:
    """Format a no-agent cron message. Empty string means silent tick."""
    lines: list[str] = []
    for decision in killed:
        process = decision.process
        lines.append(f"  - PID {process.pid} ({decision.role}, {decision.reason})")
    for removed in cleanup.removed:
        lines.append(
            f"  - Cleaned {removed} "
            f"({cleanup.inode_count} entries, {cleanup.inode_pct}% inode use)"
        )
    if not lines:
        return ""
    return "Pytest watchdog acted on %d item(s):\n%s" % (len(lines), "\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    dry_run = "--dry-run" in argv
    processes = read_processes()
    append_process_log(processes, Path(os.environ.get("PYTEST_WATCHDOG_LOG", str(DEFAULT_LOG_PATH))))

    decisions = select_runaway_processes(
        processes,
        max_run_sec=env_int("PYTEST_WATCHDOG_MAX_RUN_SEC", DEFAULT_MAX_RUN_SEC),
        cpu_threshold=float(env_int("PYTEST_WATCHDOG_CPU_THRESHOLD", int(DEFAULT_CPU_THRESHOLD))),
        rss_threshold_kb=env_int("PYTEST_WATCHDOG_RSS_THRESHOLD_KB", DEFAULT_RSS_THRESHOLD_KB),
    )
    killed = terminate_processes(decisions, dry_run=dry_run)
    cleanup = cleanup_pytest_tmp(
        Path(os.environ.get("PYTEST_WATCHDOG_TMP_DIR", str(DEFAULT_TMP_DIR))),
        min_inodes=env_int("PYTEST_WATCHDOG_MIN_TMP_INODES", DEFAULT_MIN_TMP_INODES),
        pressure_pct=env_int("PYTEST_WATCHDOG_INODE_PRESSURE_PCT", DEFAULT_INODE_PRESSURE_PCT),
        stale_seconds=env_int("PYTEST_WATCHDOG_STALE_TMP_SECONDS", DEFAULT_STALE_TMP_SECONDS),
        dry_run=dry_run,
    )
    report = format_report(killed, cleanup)
    if report:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
