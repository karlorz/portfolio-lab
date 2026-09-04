#!/usr/bin/env python3
"""Native static SPA lifecycle controller and ensure installer for Portfolio Lab on cursor-box.

Task 2.3 of the sg01 -> cursor-box migration: a focused, stdlib-only
controller for the static SPA origin lifecycle on ``box``.

Actions::

  preflight   --mode candidate|production --web-root PATH --service-name NAME
  status      (same)
  start       (same)
  stop        (same)
  ensure      (same)
  install-ensure-hook (same; requires [--ensure-script PATH])

All lifecycle actions except ``install-ensure-hook`` emit exactly one compact
JSON object on stdout satisfying the schema ``portfolio-lab-static-persist/v1``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_STATUS = "portfolio-lab-static-persist/v1"
SCHEMA_STATE = "portfolio-lab-static-persist/state/v1"
SCHEMA_INSTALL = "portfolio-lab-static-persist/install/v1"

PRODUCTION_ROOT = Path("/home/box/.local/share/portfolio-lab")
ORIGIN_EXECUTABLE_DEFAULT = Path("/home/box/.local/bin/portfolio-lab-static-origin")
CONTROLLER_INSTALL_PATH = "/home/box/.local/bin/portfolio-lab-static-persist"
ENSURE_SCRIPT_DEFAULT = "/home/box/.local/share/box-persist/ensure.sh"
DEFAULT_PATH = "/home/box/.local/bin:/usr/local/bin:/usr/bin:/bin"
DEFAULT_START_TIMEOUT = 15.0
DEFAULT_STOP_TIMEOUT = 10.0
DEFAULT_KILL_TIMEOUT = 5.0
STATIC_HOST = "127.0.0.1"
STATIC_PORT = 8001
MAX_INFLIGHT = 16

BLOCK_BEGIN = "# BEGIN portfolio-lab static managed"
BLOCK_END = "# END portfolio-lab static managed"
SERVICE_NAME_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")

# Cloudflare Ingress Rules Contract
# API route precedes static catch-all on the same host; final global catch-all remains outside.
CLOUDFLARE_INGRESS_RULES = [
    {
        "hostname": "lab.termolo.com",
        "path": r"^/api(?:/.*)?$",
        "service": "http://127.0.0.1:8000",
    },
    {
        "hostname": "lab.termolo.com",
        "service": "http://127.0.0.1:8001",
    },
]

_SPAWNED: dict[int, subprocess.Popen[str]] = {}


def match_cloudflare_route(hostname: str, path: str) -> str | None:
    """Evaluate Cloudflare first-match ingress contract for a given hostname and path."""
    for rule in CLOUDFLARE_INGRESS_RULES:
        if rule["hostname"] != hostname:
            continue
        rule_path = rule.get("path")
        if rule_path is not None:
            if re.match(rule_path, path):
                return rule["service"]
        else:
            return rule["service"]
    return None


def die(message: str, exc: BaseException | None = None) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    err = SystemExit(1)
    if exc is not None:
        err.__cause__ = exc
    raise err


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        die(f"{name} must be a numeric timeout in seconds; got {raw!r}")
    if value < 0:
        die(f"{name} must be nonnegative; got {value!r}")
    return value


def proc_root() -> Path:
    raw = os.environ.get("PLSP_PROC_ROOT")
    if raw is None:
        return Path("/proc")
    if not os.path.isabs(raw):
        die("PLSP_PROC_ROOT must be an absolute path")
    return Path(raw)


def static_persist_root() -> Path:
    raw = os.environ.get("PLSP_ROOT")
    if raw is None:
        return PRODUCTION_ROOT
    if not os.path.isabs(raw):
        die("PLSP_ROOT must be an absolute path")
    return Path(raw)


def origin_executable() -> Path:
    raw = os.environ.get("PLSP_ORIGIN_EXECUTABLE")
    if raw is None:
        return ORIGIN_EXECUTABLE_DEFAULT
    if not os.path.isabs(raw):
        die("PLSP_ORIGIN_EXECUTABLE must be an absolute path")
    return Path(raw)


def python_executable() -> Path:
    raw = os.environ.get("PLSP_PYTHON_EXECUTABLE")
    if raw is None:
        return Path(sys.executable).resolve()
    if not os.path.isabs(raw):
        die("PLSP_PYTHON_EXECUTABLE must be an absolute path")
    return Path(raw).resolve()


def configured_port() -> int:
    raw = os.environ.get("PLSP_PORT")
    if raw is None:
        return STATIC_PORT
    try:
        p = int(raw)
        if 1 <= p <= 65535:
            return p
    except ValueError:
        pass
    die(f"invalid PLSP_PORT: {raw!r}")
    return STATIC_PORT


def controlled_path() -> str:
    return os.environ.get("PLSP_PATH", DEFAULT_PATH)


def validate_service_name(name: str) -> None:
    if not SERVICE_NAME_RE.fullmatch(name):
        die(f"invalid service name {name!r}; must match [A-Za-z0-9_.@-]+")


def validate_web_root(mode: str, web_raw: str) -> Path:
    raw_root = static_persist_root()
    if not raw_root.is_dir():
        die(f"static-persist root is not a directory: {raw_root}")
    root = raw_root.resolve()
    web = Path(web_raw)
    if not web.is_absolute():
        die(f"--web-root must be an absolute path; got {web_raw!r}")
    if web.is_symlink():
        die(f"refusing symlinked --web-root: {web}")
    web_r = web.resolve()
    if not web_r.is_relative_to(root):
        die(f"--web-root must be beneath static-persist root {root}; got {web_r}")

    expected_name = "www-candidate" if mode == "candidate" else "www"
    if web_r != (root / expected_name):
        die(f"--web-root for mode {mode} must resolve exactly to {root / expected_name}; got {web_r}")

    if not web_r.is_dir():
        die(f"web-root is not a directory: {web_r}")
    idx = web_r / "index.html"
    if not idx.is_file() or idx.is_symlink():
        die(f"web-root must contain regular non-symlink index.html: {idx}")
    dt = web_r / "data"
    if not dt.is_dir() or dt.is_symlink():
        die(f"web-root must contain regular non-symlink data/ directory: {dt}")

    return web_r


def pid_file_path(root: Path, mode: str) -> Path:
    return root / "run" / f"static-{mode}.pid"


def state_file_path(root: Path, mode: str) -> Path:
    return root / "run" / f"static-{mode}-state.json"


def log_file_path(root: Path, mode: str) -> Path:
    return root / "run" / f"static-{mode}.log"


def expected_argv(web_r: Path, port: int) -> list[str]:
    return [
        str(python_executable().resolve()),
        str(origin_executable().resolve()),
        "--web-root",
        str(web_r),
        "--host",
        STATIC_HOST,
        "--port",
        str(port),
        "--max-inflight",
        str(MAX_INFLIGHT),
    ]


def atomic_write(path: Path, data: bytes, mode: int) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".plsp-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except Exception as exc:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise RuntimeError(f"failed to write {path}: {exc}") from exc


def remove_stale_files(root: Path, mode: str) -> None:
    pid_file_path(root, mode).unlink(missing_ok=True)
    state_file_path(root, mode).unlink(missing_ok=True)


# ── /proc Inspection and Identity ─────────────────────────────────────────


def read_proc_status(pid: int) -> dict[str, str]:
    path = Path(proc_root(), str(pid), "status")
    text = path.read_text(encoding="utf-8", errors="replace")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def read_proc_cmdline(pid: int) -> list[str]:
    data = Path(proc_root(), str(pid), "cmdline").read_bytes()
    parts = data.split(b"\x00")
    if parts and parts[-1] == b"":
        parts.pop()
    return [part.decode("utf-8", "replace") for part in parts]


def read_proc_link(pid: int, name: str) -> str:
    return os.readlink(str(Path(proc_root(), str(pid), name)))


def is_zombie(status: dict[str, str]) -> bool:
    return status.get("State", "").startswith("Z")


def parse_local_ip(table: str, raw: str) -> str:
    if table == "tcp":
        if len(raw) != 8:
            raise ValueError(raw)
        return str(socket.inet_ntoa(bytes.fromhex(raw)[::-1]))
    if len(raw) != 32:
        raise ValueError(raw)
    words = bytes.fromhex(raw)
    packed = b"".join(words[i : i + 4][::-1] for i in range(0, 16, 4))
    return str(socket.inet_ntop(socket.AF_INET6, packed))


def net_listening_rows() -> dict[int, tuple[str, int]]:
    """{inode: (ip, port)} for LISTEN rows on loopback only."""
    rows: dict[int, tuple[str, int]] = {}
    for table in ("tcp", "tcp6"):
        path = Path(proc_root(), "net", table)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines()[1:]:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            inode_raw = fields[9]
            if not inode_raw.isdigit():
                continue
            addr, _, port_raw = fields[1].partition(":")
            try:
                ip = parse_local_ip(table, addr)
            except (ValueError, OSError):
                continue
            if ip not in ("127.0.0.1", "::1"):
                continue
            rows[int(inode_raw)] = (ip, int(port_raw, 16))
    return rows


def proc_socket_inodes(pid: int) -> set[int]:
    fd_dir = Path(proc_root(), str(pid), "fd")
    try:
        entries = os.listdir(fd_dir)
    except OSError:
        return set()
    inodes: set[int] = set()
    for entry in entries:
        try:
            target = os.readlink(str(fd_dir / entry))
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            try:
                inodes.add(int(target[len("socket:[") : -1]))
            except ValueError:
                continue
    return inodes


def has_listening_loopback_socket(pid: int, port: int) -> bool:
    rows = net_listening_rows()
    for inode in proc_socket_inodes(pid):
        bound = rows.get(inode)
        if bound is not None and bound[1] == port:
            return True
    return False


def probe_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def reap(pid: int) -> None:
    handle = _SPAWNED.get(pid)
    if handle is not None:
        handle.poll()


def process_identity(
    pid: int,
    *,
    web_r: Path,
    port: int,
) -> tuple[bool, str]:
    """Exact Linux identity check:
    - positive live non-zombie PID
    - /proc/PID/exe equal to resolved python executable
    - cwd equal to web root
    - NUL-split argv equal to exact expected command
    - PID-owned LISTEN socket on loopback port matched by inode
    Fail-closed: if /proc does not exist, never claim exact identity.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        reap(pid)
        return False, f"pid {pid} is not alive"
    except (PermissionError, OSError) as exc:
        return False, f"pid {pid} is not probeable: {exc}"

    if not proc_root().exists():
        return False, f"proc root does not exist: {proc_root()}"

    try:
        status = read_proc_status(pid)
    except FileNotFoundError:
        reap(pid)
        return False, f"no proc status for pid {pid}"
    except OSError as exc:
        return False, f"proc status unreadable for pid {pid}: {exc}"

    if is_zombie(status):
        reap(pid)
        return False, f"pid {pid} is a zombie"

    try:
        exe = os.path.realpath(read_proc_link(pid, "exe"))
    except OSError as exc:
        return False, f"proc exe unreadable for pid {pid}: {exc}"
    expected_python = str(python_executable().resolve())
    if exe != expected_python:
        return False, f"pid {pid} executable mismatch ({exe!r} != {expected_python!r})"

    try:
        cwd = os.path.realpath(read_proc_link(pid, "cwd"))
    except OSError as exc:
        return False, f"proc cwd unreadable for pid {pid}: {exc}"
    if cwd != str(web_r):
        return False, f"pid {pid} cwd mismatch ({cwd!r} != {str(web_r)!r})"

    try:
        cmdline = read_proc_cmdline(pid)
    except OSError as exc:
        return False, f"proc cmdline unreadable for pid {pid}: {exc}"
    expected = expected_argv(web_r, port)
    if cmdline != expected:
        return False, f"pid {pid} argv mismatch ({cmdline!r} != {expected!r})"

    if not has_listening_loopback_socket(pid, port):
        return False, f"pid {pid} owns no LISTEN socket on {STATIC_HOST}:{port}"

    return True, ""


def proc_uids(status: dict[str, str]) -> frozenset[int] | None:
    raw = status.get("Uid")
    if not raw:
        return None
    try:
        uids = frozenset(int(part) for part in raw.split())
    except ValueError:
        return None
    return uids or None


def scan_origin_processes(web_r: Path, port: int) -> tuple[list[int], list[int]]:
    """Scan proc entries for exact or ambiguous static-origin processes for this host.
    Returns (exact_pids, conflicting_pids).
    """
    if hasattr(os, "getresuid"):
        own_uids = set(os.getresuid())
    else:
        own_uids = {os.getuid(), os.geteuid()}

    exact_pids: list[int] = []
    conflicts: list[int] = []

    try:
        entries = os.listdir(proc_root())
    except FileNotFoundError:
        return exact_pids, conflicts
    except OSError as exc:
        die(f"cannot list {proc_root()}: {exc}; failing closed")

    expected_script = str(origin_executable().resolve())

    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == os.getpid():
            continue
        try:
            status = read_proc_status(pid)
        except FileNotFoundError:
            continue
        except OSError as exc:
            die(f"cannot inspect process {pid} under {proc_root()}: {exc}; failing closed")
        if is_zombie(status):
            continue
        uids = proc_uids(status)
        if uids is not None and not (uids & own_uids):
            continue

        try:
            cmdline = read_proc_cmdline(pid)
        except FileNotFoundError:
            continue
        except OSError as exc:
            die(f"cannot inspect process {pid} under {proc_root()}: {exc}; failing closed")

        # Structured position-aware argv matching for shipped invocation:
        # Interpreter is argv[0]; the script argument is argv[1].
        # Allow exact configured resolved script path or exact configured basename at argv[1].
        if len(cmdline) < 2:
            continue
        origin_basename = os.path.basename(expected_script)
        script_arg = cmdline[1]
        if script_arg != expected_script and script_arg != origin_basename:
            continue

        conflicts.append(pid)

        # Check if exact
        exact, _ = process_identity(pid, web_r=web_r, port=port)
        if exact:
            exact_pids.append(pid)

    return exact_pids, conflicts


def port_free(host: str = STATIC_HOST, port: int = STATIC_PORT) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
    except OSError:
        return False
    return True


def read_pid_file(root: Path, mode: str, *, cleanup: bool = True) -> int | None:
    path = pid_file_path(root, mode)
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        die(f"cannot stat pid file {path}: {exc}")
    if stat.S_ISLNK(st.st_mode):
        die(f"pid file must not be a symlink: {path}")
    if not stat.S_ISREG(st.st_mode):
        die(f"pid file must be a regular file: {path}")
    if stat.S_IMODE(st.st_mode) != 0o600:
        die(f"pid file must be exactly mode 0600: {path} (got {oct(stat.S_IMODE(st.st_mode))})")
    try:
        data = path.read_text(encoding="utf-8")
    except OSError as exc:
        die(f"cannot read pid file {path}: {exc}")
    try:
        pid = int(data.strip())
    except ValueError:
        if cleanup:
            remove_stale_files(root, mode)
        return None
    if pid <= 0:
        if cleanup:
            remove_stale_files(root, mode)
        return None
    return pid


def status_payload(
    mode: str,
    web_r: Path,
    service_name: str,
    *,
    state: str,
    identity_exact: bool,
    pid: int | None,
    host: str = STATIC_HOST,
    port: int = STATIC_PORT,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA_STATUS,
        "state": state,
        "identity_exact": identity_exact,
        "pid": pid,
        "mode": mode,
        "service_name": service_name,
        "web_root": str(web_r),
        "host": host,
        "port": port,
    }
    if extra:
        payload.update(extra)
    return payload


def inspect(
    mode: str,
    web_r: Path,
    service_name: str,
    port: int,
    *,
    cleanup_stale: bool = True,
) -> dict[str, Any]:
    root = static_persist_root()
    exact_pids, conflicts = scan_origin_processes(web_r, port)
    pid = read_pid_file(root, mode, cleanup=cleanup_stale)
    if pid is not None:
        exact, reason = process_identity(pid, web_r=web_r, port=port)
        if exact:
            return status_payload(
                mode, web_r, service_name,
                state="active", identity_exact=True, pid=pid, port=port
            )
        # Check if dead or zombie
        if reason and ("zombie" in reason or "not alive" in reason):
            if cleanup_stale:
                remove_stale_files(root, mode)
            pid = None
        else:
            # Active but inexact identity (permission-denied or mismatch)
            return status_payload(
                mode, web_r, service_name,
                state="active", identity_exact=False, pid=pid, port=port
            )

    # PID file absent, but origin process exists
    if conflicts:
        return status_payload(
            mode, web_r, service_name,
            state="active", identity_exact=False, pid=conflicts[0], port=port
        )

    return status_payload(
        mode, web_r, service_name,
        state="inactive", identity_exact=True, pid=None, port=port
    )


# ── Termination ───────────────────────────────────────────────────────────


def proc_stopped(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        reap(pid)
        return True
    except (PermissionError, OSError):
        return False
    if not proc_root().exists():
        return False
    try:
        status = read_proc_status(pid)
    except FileNotFoundError:
        reap(pid)
        return True
    except OSError:
        return False
    if is_zombie(status):
        reap(pid)
        return True
    try:
        cmdline = read_proc_cmdline(pid)
    except FileNotFoundError:
        reap(pid)
        return True
    except OSError:
        return False
    if not cmdline:
        return True
    return False


def wait_stopped(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() <= deadline:
        if proc_stopped(pid):
            return True
        time.sleep(0.1)
    return False


def terminate_exact(
    pid: int,
    *,
    web_r: Path,
    port: int,
    stop_timeout: float,
    kill_timeout: float,
) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        reap(pid)
        return
    if wait_stopped(pid, stop_timeout):
        return
    reap(pid)
    if proc_stopped(pid):
        return
    exact, why = process_identity(pid, web_r=web_r, port=port)
    if proc_stopped(pid):
        return
    if not exact:
        die(f"refusing SIGKILL: identity of pid {pid} changed during stop ({why!r}); not signaling")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        reap(pid)
        return
    if not wait_stopped(pid, kill_timeout):
        die(f"pid {pid} did not stop even after SIGKILL")


def rollback_spawned_pid(
    pid: int,
    *,
    web_r: Path,
    port: int,
    mode: str,
    root: Path,
    deadline_seconds: float | None = None,
) -> None:
    # Bounded wait for exact identity or process exit.
    # _SPAWNED proves Popen created this PID, but does not override exact Linux identity.
    # Never signal an inexact process.
    if deadline_seconds is None:
        deadline_seconds = min(1.0, _env_float("PLSP_STOP_TIMEOUT", DEFAULT_STOP_TIMEOUT))
    deadline = time.monotonic() + deadline_seconds
    exact_now = False
    why = ""
    while time.monotonic() <= deadline:
        if not probe_alive(pid):
            break
        exact_now, why = process_identity(pid, web_r=web_r, port=port)
        if exact_now:
            break
        time.sleep(0.05)

    if exact_now:
        stop_timeout = _env_float("PLSP_STOP_TIMEOUT", DEFAULT_STOP_TIMEOUT)
        kill_timeout = _env_float("PLSP_KILL_TIMEOUT", DEFAULT_KILL_TIMEOUT)
        terminate_exact(pid, web_r=web_r, port=port, stop_timeout=stop_timeout, kill_timeout=kill_timeout)
    else:
        if probe_alive(pid):
            print(
                f"WARNING: rollback retaining inexact process {pid} alive without signaling; last check: {why!r}",
                file=sys.stderr,
            )

    handle = _SPAWNED.get(pid)
    if handle is not None:
        handle.poll()
    reap(pid)
    remove_stale_files(root, mode)


# ── Actions ───────────────────────────────────────────────────────────────


def preflight_checks(mode: str, web_r: Path, service_name: str, port: int) -> None:
    exe = origin_executable()
    if not exe.is_file() or exe.is_symlink():
        die(f"origin executable is not a regular non-symlink file: {exe}")
    if not os.access(exe, os.X_OK):
        die(f"origin executable is not executable: {exe}")
    py_exe = python_executable()
    if not py_exe.is_file() or py_exe.is_symlink():
        die(f"python executable is not a regular non-symlink file: {py_exe}")
    if not os.access(py_exe, os.X_OK):
        die(f"python executable is not executable: {py_exe}")

    # Check unmanaged or other-mode origin processes
    _, conflicts = scan_origin_processes(web_r, port)
    if conflicts:
        die(f"cannot proceed: conflicting static-origin process(es) active: {conflicts}")

    # Other mode must not be active on the shared port
    other_mode = "production" if mode == "candidate" else "candidate"
    root = static_persist_root()
    other_status = inspect(other_mode, root / ("www" if other_mode == "production" else "www-candidate"), service_name, port, cleanup_stale=False)
    if other_status["state"] == "active":
        die(f"cannot proceed: {other_mode} is active on shared port {port}")
    if not port_free(STATIC_HOST, port):
        die(f"port {port} is occupied on {STATIC_HOST}; refusing to proceed")


def action_preflight(mode: str, web_r: Path, service_name: str, port: int) -> dict[str, Any]:
    preflight_checks(mode, web_r, service_name, port)
    payload = inspect(mode, web_r, service_name, port, cleanup_stale=False)
    payload = dict(payload)
    payload["preflight"] = {"ok": True, "port": port}
    return payload


def action_status(mode: str, web_r: Path, service_name: str, port: int) -> dict[str, Any]:
    return inspect(mode, web_r, service_name, port)


def ensure_run_dir(root: Path) -> Path:
    run_dir = root / "run"
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Enforce 0700 permissions on existing or newly created run directory
    os.chmod(run_dir, 0o700)
    return run_dir


def open_mode_log(log_path: Path) -> int:
    try:
        st = os.lstat(log_path)
        if stat.S_ISLNK(st.st_mode):
            die(f"refusing symlinked log file: {log_path}")
        if not stat.S_ISREG(st.st_mode):
            die(f"log file must be a regular file: {log_path}")
    except FileNotFoundError:
        pass
    except OSError as exc:
        die(f"cannot stat log file {log_path}: {exc}")

    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    log_fd = os.open(log_path, flags, 0o600)
    os.fchmod(log_fd, 0o600)
    return log_fd


def spawn(mode: str, web_r: Path, service_name: str, port: int) -> dict[str, Any]:
    preflight_checks(mode, web_r, service_name, port)
    root = static_persist_root()
    ensure_run_dir(root)

    argv = expected_argv(web_r, port)
    env = {
        "PATH": controlled_path(),
        "HOME": os.environ.get("HOME", "/home/box"),
        "USER": os.environ.get("USER", "box"),
    }
    # Pass test-only fake proc knobs
    for k, v in os.environ.items():
        if k.startswith("PLSP_FAKE_"):
            env[k] = v

    log_path = log_file_path(root, mode)
    log_fd = open_mode_log(log_path)
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(web_r),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=log_fd,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(log_fd)

    pid = proc.pid
    _SPAWNED[pid] = proc
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        atomic_write(pid_file_path(root, mode), f"{pid}\n".encode(), 0o600)
        atomic_write(
            state_file_path(root, mode),
            json.dumps(
                {
                    "schema": SCHEMA_STATE,
                    "pid": pid,
                    "mode": mode,
                    "service_name": service_name,
                    "web_root": str(web_r),
                    "started_at": started_at,
                    "host": STATIC_HOST,
                    "port": port,
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode(),
            0o600,
        )
    except Exception as exc:
        rollback_spawned_pid(pid, web_r=web_r, port=port, mode=mode, root=root)
        die(f"failed to record {mode} ownership for pid {pid}: {exc}", exc=exc)

    deadline = time.monotonic() + _env_float("PLSP_START_TIMEOUT", DEFAULT_START_TIMEOUT)
    accepted: dict[str, Any] | None = None
    while time.monotonic() <= deadline:
        reap(pid)
        current = inspect(mode, web_r, service_name, port)
        if current["state"] == "active" and current["identity_exact"] is True and current["pid"] == pid:
            accepted = current
            break
        time.sleep(0.1)

    if accepted is not None:
        return accepted

    # Failed startup
    _, why = process_identity(pid, web_r=web_r, port=port)
    rollback_spawned_pid(pid, web_r=web_r, port=port, mode=mode, root=root)
    die(
        f"{mode} static process {pid} did not reach exact healthy identity within "
        f"PLSP_START_TIMEOUT; last check: {why!r}"
    )


def action_start(mode: str, web_r: Path, service_name: str, port: int) -> dict[str, Any]:
    current = inspect(mode, web_r, service_name, port)
    if current["state"] == "active":
        if current["identity_exact"] is True:
            return current
        die(f"refusing start: active process with inexact identity (pid {current['pid']}); not signaling")
    return spawn(mode, web_r, service_name, port)


def action_stop(mode: str, web_r: Path, service_name: str, port: int) -> dict[str, Any]:
    current = inspect(mode, web_r, service_name, port)
    if current["state"] == "inactive":
        return current
    if current["identity_exact"] is False:
        die(f"refusing stop: pid {current['pid']} has inexact identity; not signaling and not removing PID file")
    pid = current["pid"]
    root = static_persist_root()
    stop_timeout = _env_float("PLSP_STOP_TIMEOUT", DEFAULT_STOP_TIMEOUT)
    kill_timeout = _env_float("PLSP_KILL_TIMEOUT", DEFAULT_KILL_TIMEOUT)
    terminate_exact(pid, web_r=web_r, port=port, stop_timeout=stop_timeout, kill_timeout=kill_timeout)
    remove_stale_files(root, mode)
    return status_payload(
        mode, web_r, service_name,
        state="inactive", identity_exact=True, pid=None, port=port
    )


def action_ensure(mode: str, web_r: Path, service_name: str, port: int) -> dict[str, Any]:
    current = inspect(mode, web_r, service_name, port)
    if current["state"] == "active":
        if current["identity_exact"] is True:
            return current
        die(f"refusing ensure: active process with inexact identity (pid {current['pid']}); not signaling")
    return spawn(mode, web_r, service_name, port)


def install_ensure_hook(
    mode: str,
    web_r: Path,
    service_name: str,
    ensure_script_raw: str,
) -> dict[str, Any]:
    path = Path(ensure_script_raw)
    if not path.is_absolute():
        die(f"--ensure-script must be an absolute path; got {ensure_script_raw!r}")
    if path.is_symlink():
        die(f"ensure script must not be a symlink: {path}")
    try:
        st = path.stat()
    except OSError:
        die(f"ensure script missing at {path}")
    if not stat.S_ISREG(st.st_mode):
        die(f"ensure script must be a regular file: {path}")
    if st.st_mode & 0o022:
        die(f"ensure script must not be group/world-writable: {path}")
    if not st.st_mode & 0o111:
        die(f"ensure script must be executable: {path}")
    try:
        body = path.read_bytes()
        body.decode("utf-8")
    except OSError as exc:
        die(f"ensure script unreadable: {path} ({exc})")
    except UnicodeDecodeError:
        die(f"ensure script must be UTF-8 text: {path}")

    line = (
        f"PATH={DEFAULT_PATH} {CONTROLLER_INSTALL_PATH} ensure "
        + " ".join(
            f"{flag} {shlex.quote(value)}"
            for flag, value in (
                ("--mode", mode),
                ("--web-root", str(web_r)),
                ("--service-name", service_name),
            )
        )
    )
    block = f"{BLOCK_BEGIN}\n{line}\n{BLOCK_END}\n".encode("utf-8")
    begin = body.find(BLOCK_BEGIN.encode("utf-8"))
    end = body.find(BLOCK_END.encode("utf-8"))
    if begin != -1:
        if end == -1 or end < begin:
            die(f"malformed managed block in {path}: BEGIN without END")
        if body.find(BLOCK_BEGIN.encode("utf-8"), begin + 1) != -1 and (
            end == -1 or body.find(BLOCK_BEGIN.encode("utf-8"), begin + 1) < end
        ):
            die(f"malformed managed block in {path}: nested BEGIN marker")
        if body.find(BLOCK_END.encode("utf-8"), end + 1) != -1:
            die(f"malformed managed block in {path}: duplicate END marker")
        line_end = body.find(b"\n", end)
        if line_end == -1:
            line_end = len(body)
        new_body = body[:begin] + block + body[line_end + 1 :]
    else:
        if end != -1:
            die(f"malformed managed block in {path}: END without BEGIN")
        prefix = b"" if not body or body.endswith(b"\n") else b"\n"
        new_body = body + prefix + block

    changed = new_body != body
    if changed:
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".plsp-ensure-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(new_body)
            os.chmod(tmp_name, stat.S_IMODE(st.st_mode))
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    return {
        "schema": SCHEMA_INSTALL,
        "changed": changed,
        "path": str(path),
        "service_name": service_name,
        "mode": mode,
        "web_root": str(web_r),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio Lab Static Lifecycle Controller")
    parser.add_argument(
        "action",
        choices=["preflight", "status", "start", "stop", "ensure", "install-ensure-hook"],
        help="Lifecycle action",
    )
    parser.add_argument("--mode", required=True, choices=["candidate", "production"], help="Lifecycle mode")
    parser.add_argument("--web-root", required=True, help="Absolute path to web root")
    parser.add_argument("--service-name", required=True, help="Service name")
    parser.add_argument("--ensure-script", default=ENSURE_SCRIPT_DEFAULT, help="Path to ensure.sh script")
    args = parser.parse_args()

    validate_service_name(args.service_name)
    web_r = validate_web_root(args.mode, args.web_root)
    port = configured_port()

    if args.action == "install-ensure-hook":
        res = install_ensure_hook(args.mode, web_r, args.service_name, args.ensure_script)
        emit(res)
        return

    if args.action == "preflight":
        res = action_preflight(args.mode, web_r, args.service_name, port)
    elif args.action == "status":
        res = action_status(args.mode, web_r, args.service_name, port)
    elif args.action == "start":
        res = action_start(args.mode, web_r, args.service_name, port)
    elif args.action == "stop":
        res = action_stop(args.mode, web_r, args.service_name, port)
    elif args.action == "ensure":
        res = action_ensure(args.mode, web_r, args.service_name, port)
    else:
        die(f"unknown action {args.action}")

    emit(res)


if __name__ == "__main__":
    main()
