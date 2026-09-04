#!/usr/bin/env python3
"""Native box-persist lifecycle controller for Portfolio Lab on cursor-box.

Task 2.2 of the sg01 -> cursor-box migration: a focused, stdlib-only
controller for the user-owned (non-systemd) Tasker lifecycle on ``box``.

Actions (argparse-validated)::

  preflight   --mode candidate|production --app-dir PATH --web-root PATH \\
              --service-name NAME
  status      (same identity args)
  start-candidate (same; requires --mode candidate)
  stop        (same)
  ensure      (same)
  activate    (same; requires --former-authority-confirmed-stopped LABEL)
  install-ensure-hook (same; requires --ensure-script PATH)

All lifecycle actions except ``install-ensure-hook`` emit exactly one compact
JSON object on stdout satisfying the recovery schema
``portfolio-lab-box-persist/v1`` (state, scheduler_mode, identity_exact,
scheduler_instances, pid, plus the exact service_name/mode/app_dir/web_root
echo). The installer emits one non-secret JSON line with changed/ensure path.

Env-file contract: strict ``KEY=VALUE`` lines (blank/comment lines and simple
single/double-quoted values allowed; malformed keys/lines/NUL rejected),
never executed, never printed (diagnostics report only the file and 1-based
line number), 0600 no group/other permissions, separate per mode. The child
environment is built from a minimal safe-basics allowlist plus the selected
mode file; controlled runtime fields always override file values.

PID records used for live ownership must be non-symlink regular files with
exactly mode 0600; unsafe records fail closed without being cleaned or
signaled.

Process identity is exact (Linux /proc; tests override with absolute
``PLBP_PROC_ROOT``): PID alive non-zombie, /proc/PID/exe resolves to the
resolved ``app/.venv/bin/python``, cwd resolves to app, NUL-split cmdline
equals the exact expected argv, environ contains the exact controlled markers
(candidate also needs ``TASKER_DISABLE_SCHEDULER=1``, production must carry
neither disable control), and the PID owns a LISTEN socket on 127.0.0.1:8000
(127.0.0.1 or ::1 only; wildcard/non-loopback rejected).

Scheduler instances are counted by scanning numeric proc entries: exact
production argv + app cwd + no disable controls. Reads are ordered by
relevance (status, other-user UID skip, cmdline, then environ/cwd only for
relevant same-user Tasker candidates); unreadable/ambiguous candidates that
could matter fail closed.

``stop`` SIGTERMs only the exact PID with a bounded graceful wait, re-reads
exact identity before any SIGKILL escalation (PID-reuse protection), never
uses killpg/pkill/killall/shell/name matching, and cleans stale PID/state.

``activate`` starts production only from an inactive exact pre-state with no
conflicts and stores a 0600 ``run/production-activation.json`` marker binding
service/app/web/command identity with a SHA-256 of the former-authority proof
label (no plaintext label). Production ``ensure`` restarts from inactive only
when that marker matches exactly.

``install-ensure-hook`` atomically installs/replaces exactly one managed
block into the supplied ensure script, preserving mode and unrelated bytes.

Overrides (tests): PLBP_ROOT (absolute), PLBP_PROC_ROOT (absolute), PLBP_PATH,
PLBP_START_TIMEOUT / PLBP_STOP_TIMEOUT / PLBP_KILL_TIMEOUT (numeric seconds).
"""

from __future__ import annotations

import argparse
import hashlib
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

SCHEMA_STATUS = "portfolio-lab-box-persist/v1"
SCHEMA_STATE = "portfolio-lab-box-persist/state/v1"
SCHEMA_ACTIVATION = "portfolio-lab-production-activation/v1"
SCHEMA_INSTALL = "portfolio-lab-box-persist/install/v1"
PRODUCTION_ROOT = Path("/home/box/.local/share/portfolio-lab")
CONTROLLER_INSTALL_PATH = "/home/box/.local/bin/portfolio-lab-box-persist"
ENSURE_SCRIPT_DEFAULT = "/home/box/.local/share/box-persist/ensure.sh"
DEFAULT_PATH = "/home/box/.local/bin:/usr/local/bin:/usr/bin:/bin"
DEFAULT_START_TIMEOUT = 15.0
DEFAULT_STOP_TIMEOUT = 30.0
DEFAULT_KILL_TIMEOUT = 5.0
API_HOST = "127.0.0.1"
API_PORT = 8000
BLOCK_BEGIN = "# BEGIN portfolio-lab managed"
BLOCK_END = "# END portfolio-lab managed"
SERVICE_NAME_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ACTIONS = (
    "preflight",
    "status",
    "start-candidate",
    "stop",
    "ensure",
    "activate",
    "install-ensure-hook",
)

# Child Popen handles owned by this CLI invocation (reaped on poll).
_SPAWNED: dict[int, subprocess.Popen[str]] = {}


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
    raw = os.environ.get("PLBP_PROC_ROOT")
    if raw is None:
        return Path("/proc")
    if not os.path.isabs(raw):
        die("PLBP_PROC_ROOT must be an absolute path")
    return Path(raw)


def box_persist_root() -> Path:
    raw = os.environ.get("PLBP_ROOT")
    if raw is None:
        return PRODUCTION_ROOT
    if not os.path.isabs(raw):
        die("PLBP_ROOT must be an absolute path")
    return Path(raw)


def controlled_path() -> str:
    return os.environ.get("PLBP_PATH", DEFAULT_PATH)


# ── paths and targets ──────────────────────────────────────────────────────


def validate_targets(app_raw: str, web_raw: str) -> tuple[Path, Path]:
    """Absolute, non-symlink, beneath the resolved root, distinct and
    non-overlapping. Shared by every action before any dispatch."""
    raw_root = box_persist_root()
    if not raw_root.is_dir():
        die(f"box-persist root is not a directory: {raw_root}")
    root = raw_root.resolve()
    app = Path(app_raw)
    web = Path(web_raw)
    for raw, role in ((app, "--app-dir"), (web, "--web-root")):
        if not raw.is_absolute():
            die(f"{role} must be an absolute path; got {str(raw)!r}")
        if raw.is_symlink():
            die(f"refusing symlinked {role}: {raw}")
        resolved = raw.resolve()
        if not resolved.is_relative_to(root):
            die(f"{role} must be beneath box-persist root {root}; got {resolved}")
    app_r, web_r = app.resolve(), web.resolve()
    if app_r == web_r:
        die("--app-dir and --web-root must be distinct paths")
    if app_r.is_relative_to(web_r) or web_r.is_relative_to(app_r):
        die("--app-dir and --web-root must not overlap")
    return app_r, web_r


def validate_service_name(name: str) -> None:
    if not SERVICE_NAME_RE.fullmatch(name):
        die(f"invalid service name {name!r}; must match [A-Za-z0-9_.@-]+")


def pid_file_path(root: Path, mode: str) -> Path:
    return root / "run" / f"tasker-{mode}.pid"


def state_file_path(root: Path, mode: str) -> Path:
    return root / "run" / f"tasker-{mode}-state.json"


def log_file_path(root: Path, mode: str) -> Path:
    return root / "run" / f"tasker-{mode}.log"


def env_file_path(root: Path, mode: str) -> Path:
    return root / "runtime" / f"{mode}.env"


def activation_file_path(root: Path) -> Path:
    return root / "run" / "production-activation.json"


def python_exe(app_r: Path) -> Path:
    return app_r / ".venv" / "bin" / "python"


def expected_argv(app_r: Path, mode: str) -> list[str]:
    argv = [
        str(python_exe(app_r)),
        "-m",
        "src.tasker.service",
        "--host",
        API_HOST,
        "--port",
        str(API_PORT),
    ]
    if mode == "candidate":
        argv.append("--no-scheduler")
    return argv


def controlled_markers(
    app_r: Path,
    web_r: Path,
    service_name: str,
    mode: str,
    path_value: str,
) -> list[str]:
    markers = [
        f"PATH={path_value}",
        "PORTFOLIO_LAB_ENABLE_ML=0",
        "PORTFOLIO_LAB_MODE=lab",
        "CRON_BACKEND=tasker",
        f"PORTFOLIO_LAB_PROJECT_DIR={app_r}",
        f"PUBLIC_DATA_DIR={web_r / 'data'}",
        f"PYTHONPATH={app_r}",
        "LOG_LEVEL=INFO",
        "JSON_LOGS=1",
        f"TASKER_HOST={API_HOST}",
        f"TASKER_PORT={API_PORT}",
        f"PORTFOLIO_LAB_BOX_PERSIST_MODE={mode}",
        f"PORTFOLIO_LAB_BOX_PERSIST_SERVICE={service_name}",
    ]
    if mode == "candidate":
        markers.append("TASKER_DISABLE_SCHEDULER=1")
    return markers


def expected_python_exe(app_r: Path) -> str:
    return str(python_exe(app_r).resolve())


# ── env file parsing (strict, no shell, never printed) ────────────────────


def read_env_file(root: Path, mode: str) -> dict[str, str]:
    path = env_file_path(root, mode)
    what = f"{mode}.env configuration"
    if path.is_symlink():
        die(f"{what} must not be a symlink: {path}")
    try:
        st = path.stat()
    except OSError:
        die(f"{what} unreadable at {path}")
    if not stat.S_ISREG(st.st_mode):
        die(f"{what} must be a regular file: {path}")
    if st.st_mode & 0o077:
        die(f"{what} must have no group/other permissions: {path}")
    try:
        data = path.read_bytes()
    except OSError:
        die(f"{what} unreadable at {path}")
    if b"\x00" in data:
        die(f"{what} contains NUL bytes; refusing to parse {path}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        die(f"{what} is not valid UTF-8; refusing to parse {path}")
    values: dict[str, str] = {}
    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, eq, value = stripped.partition("=")
        if not eq:
            die(f"{what} line {index}: missing '='")
        key = key.strip()
        if not _ENV_KEY_RE.fullmatch(key):
            die(f"{what} line {index}: malformed key")
        value = value.strip()
        if len(value) >= 2 and value[0] in ("'", '"'):
            if value[-1] != value[0]:
                die(f"{what} line {index}: unterminated quoted value")
            value = value[1:-1]
        if any(ord(c) < 32 or ord(c) == 127 for c in value):
            die(f"{what} line {index}: value contains control characters")
        values[key] = value
    return values


# Safe process-basics the child may inherit from the controller environment.
# Everything else must come from the operator-provisioned mode file.
CHILD_ENV_ALLOWLIST = frozenset(
    {
        "HOME",
        "USER",
        "LOGNAME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    }
)


def build_child_env(
    root: Path,
    mode: str,
    app_r: Path,
    web_r: Path,
    service_name: str,
) -> dict[str, str]:
    path_value = controlled_path()
    env = {name: os.environ[name] for name in CHILD_ENV_ALLOWLIST if name in os.environ}
    # Test-only fake-proc controls are explicitly forced so test doubles can
    # mirror /proc; no other PLBP_* controls reach the child.
    for name, value in os.environ.items():
        if name.startswith("PLBP_FAKE_"):
            env[name] = value
    env.update(read_env_file(root, mode))
    env.update(
        {
            "PATH": path_value,
            "PORTFOLIO_LAB_ENABLE_ML": "0",
            "PORTFOLIO_LAB_MODE": "lab",
            "CRON_BACKEND": "tasker",
            "PORTFOLIO_LAB_PROJECT_DIR": str(app_r),
            "PUBLIC_DATA_DIR": str(web_r / "data"),
            "PYTHONPATH": str(app_r),
            "LOG_LEVEL": "INFO",
            "JSON_LOGS": "1",
            "TASKER_HOST": API_HOST,
            "TASKER_PORT": str(API_PORT),
            "PORTFOLIO_LAB_BOX_PERSIST_MODE": mode,
            "PORTFOLIO_LAB_BOX_PERSIST_SERVICE": service_name,
        }
    )
    if mode == "candidate":
        env["TASKER_DISABLE_SCHEDULER"] = "1"
    else:
        env.pop("TASKER_DISABLE_SCHEDULER", None)
    return env


# ── /proc readers (small and testable; PLBP_PROC_ROOT mirrors these) ───────


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


def read_proc_environ(pid: int) -> list[str]:
    data = Path(proc_root(), str(pid), "environ").read_bytes()
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
    # /proc/net/tcp6 stores the address as four little-endian 32-bit words.
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


def has_listening_loopback_socket(pid: int, port: int = API_PORT) -> bool:
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
    app_r: Path,
    web_r: Path,
    service_name: str,
    mode: str,
) -> tuple[bool, str]:
    """The seven exact-identity requirements; returns (exact=False, reason) for
    dead, zombie, or any mismatch. Never signals or mutates anything."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        reap(pid)
        return False, f"pid {pid} is not alive"
    except (PermissionError, OSError) as exc:
        # Exists but cannot be probed: never treat as stale or signal it.
        return False, f"pid {pid} is not probeable: {exc}"
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
    if exe != expected_python_exe(app_r):
        return False, f"pid {pid} executable mismatch ({exe!r})"
    try:
        cwd = os.path.realpath(read_proc_link(pid, "cwd"))
    except OSError as exc:
        return False, f"proc cwd unreadable for pid {pid}: {exc}"
    if cwd != str(app_r):
        return False, f"pid {pid} cwd mismatch ({cwd!r})"
    try:
        cmdline = read_proc_cmdline(pid)
    except OSError as exc:
        return False, f"proc cmdline unreadable for pid {pid}: {exc}"
    if cmdline != expected_argv(app_r, mode):
        return False, f"pid {pid} argv mismatch ({cmdline!r})"
    try:
        environ = read_proc_environ(pid)
    except OSError as exc:
        return False, f"proc environ unreadable for pid {pid}: {exc}"
    missing = [
        marker
        for marker in controlled_markers(app_r, web_r, service_name, mode, controlled_path())
        if marker not in environ
    ]
    if missing:
        return False, f"pid {pid} missing controlled env markers: {missing!r}"
    if mode == "production":
        carried = [entry for entry in environ if entry == "TASKER_DISABLE_SCHEDULER=1"]
        if carried:
            return False, f"pid {pid} carries scheduler-disable env in production: {carried!r}"
    if not has_listening_loopback_socket(pid):
        return False, f"pid {pid} owns no LISTEN socket on {API_HOST}:{API_PORT}"
    return True, ""


# ── scheduler instance scan (fail closed) ──────────────────────────────────


def proc_uids(status: dict[str, str]) -> frozenset[int] | None:
    """Real/effective/saved/fs UIDs from a status file, or None if unusable."""
    raw = status.get("Uid")
    if not raw:
        return None
    try:
        uids = frozenset(int(part) for part in raw.split())
    except ValueError:
        return None
    return uids or None


def scan_proc_entries(
    app_r: Path,
) -> tuple[list[int], list[int]]:
    """Return (scheduler_pids, conflicting_tasker_pids) for this app.

    A scheduler instance is a live, non-zombie process whose NUL-split
    cmdline equals the exact production argv, whose cwd resolves to app, and
    whose environ has the app marker without either disable control. A
    conflicting Tasker process is any live non-zombie process under this root
    with the tasker CLI prefix whose cwd or app marker matches the app.

    Reads are ordered by relevance: status first, then a UID-based skip (a
    process owned by another user cannot be this app's workload), then
    cmdline (clearly non-Tasker argv is skipped without environ/cwd reads),
    and only then environ/cwd for relevant same-user Tasker candidates.
    Unreadable or ambiguous entries that could matter fail closed rather
    than being counted as zero."""
    expected = expected_argv(app_r, "production")
    app_marker = f"PORTFOLIO_LAB_PROJECT_DIR={app_r}"
    if hasattr(os, "getresuid"):  # Linux: real/effective/saved UIDs
        own_uids = set(os.getresuid())
    else:  # portable fallback (macOS tests): real + effective
        own_uids = {os.getuid(), os.geteuid()}
    scheduler_pids: list[int] = []
    conflicts: list[int] = []
    try:
        entries = os.listdir(proc_root())
    except FileNotFoundError:
        return scheduler_pids, conflicts  # empty proc tree: nothing to count
    except OSError as exc:
        die(f"cannot list {proc_root()}: {exc}; failing closed")
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == os.getpid():
            continue
        try:
            status = read_proc_status(pid)
        except FileNotFoundError:
            continue  # process vanished mid-scan; nothing to count
        except OSError as exc:
            die(f"cannot inspect process {pid} under {proc_root()}: {exc}; failing closed")
        if is_zombie(status):
            continue
        uids = proc_uids(status)
        if uids is not None and not (uids & own_uids):
            # Another user's process: cannot be our workload; skip without
            # reading its cmdline/environ/cwd.
            continue
        try:
            cmdline = read_proc_cmdline(pid)
        except FileNotFoundError:
            continue
        except OSError as exc:
            die(f"cannot inspect process {pid} under {proc_root()}: {exc}; failing closed")
        if "-m" not in cmdline or "src.tasker.service" not in cmdline:
            continue  # clearly not a Tasker CLI; no environ/cwd reads needed
        try:
            environ = read_proc_environ(pid)
            cwd = os.path.realpath(read_proc_link(pid, "cwd"))
        except FileNotFoundError:
            continue  # process vanished mid-scan
        except OSError as exc:
            die(f"cannot inspect process {pid} under {proc_root()}: {exc}; failing closed")
        app_match = cwd == str(app_r) or app_marker in environ
        if not app_match:
            continue
        conflicts.append(pid)
        if (
            cmdline == expected
            and "TASKER_DISABLE_SCHEDULER=1" not in environ
            and app_marker in environ
        ):
            scheduler_pids.append(pid)
    return scheduler_pids, conflicts


# ── status evaluation ──────────────────────────────────────────────────────


def status_payload(
    mode: str,
    app_r: Path,
    web_r: Path,
    service_name: str,
    *,
    state: str,
    scheduler_mode: str,
    identity_exact: bool,
    scheduler_instances: int,
    pid: int | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA_STATUS,
        "state": state,
        "scheduler_mode": scheduler_mode,
        "identity_exact": identity_exact,
        "scheduler_instances": scheduler_instances,
        "pid": pid,
        "service_name": service_name,
        "mode": mode,
        "app_dir": str(app_r),
        "web_root": str(web_r),
    }
    if extra:
        payload.update(extra)
    return payload


def read_pid_file(root: Path, mode: str, *, cleanup: bool = True) -> int | None:
    """Read a live-ownership PID record.

    The record must be a non-symlink regular file with exactly mode 0600;
    unsafe records fail closed (never silently cleaned, never deleted, never
    signaled). Invalid content (non-positive or non-numeric) is treated as
    stale: removed when ``cleanup`` is true, otherwise left untouched."""
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
        die(
            f"pid file must be exactly mode 0600: {path} "
            f"(got {oct(stat.S_IMODE(st.st_mode))})"
        )
    try:
        data = path.read_text(encoding="utf-8")
    except OSError as exc:
        die(f"cannot read pid file {path}: {exc}")
    try:
        pid = int(data.strip())
    except ValueError:
        # Not a valid PID record.
        if cleanup:
            remove_stale_files(root, mode)
        return None
    if pid <= 0:
        if cleanup:
            remove_stale_files(root, mode)
        return None
    return pid


def remove_stale_files(root: Path, mode: str) -> None:
    pid_file_path(root, mode).unlink(missing_ok=True)
    state_file_path(root, mode).unlink(missing_ok=True)


def inspect(
    mode: str,
    app_r: Path,
    web_r: Path,
    service_name: str,
    *,
    cleanup_stale: bool = True,
) -> tuple[dict[str, Any], list[int], list[int]]:
    """Evaluate status; returns (status_payload, scheduler_pids, conflicts)."""
    scheduler_pids, conflicts = scan_proc_entries(app_r)
    instances = len(scheduler_pids)
    pid = read_pid_file(box_persist_root(), mode, cleanup=cleanup_stale)
    if pid is not None:
        exact, _reason = process_identity(
            pid, app_r=app_r, web_r=web_r, service_name=service_name, mode=mode
        )
        if exact:
            scheduler_mode = "enabled" if instances else "disabled"
            return (
                status_payload(
                    mode, app_r, web_r, service_name,
                    state="active", scheduler_mode=scheduler_mode,
                    identity_exact=True, scheduler_instances=instances, pid=pid,
                ),
                scheduler_pids,
                conflicts,
            )
        if _reason and ("zombie" in _reason or "not alive" in _reason):
            if cleanup_stale:
                remove_stale_files(box_persist_root(), mode)
            pid = None
        else:
            # Alive but identity differs: report active with inexact identity;
            # never delete the PID file and never signal the process.
            scheduler_mode = "enabled" if instances else "disabled"
            return (
                status_payload(
                    mode, app_r, web_r, service_name,
                    state="active", scheduler_mode=scheduler_mode,
                    identity_exact=False, scheduler_instances=instances, pid=pid,
                ),
                scheduler_pids,
                conflicts,
            )
    if pid is None and scheduler_pids:
        # A matching scheduler exists without the expected PID file: never a
        # preflight-acceptable inactive/disabled/exact/zero result.
        return (
            status_payload(
                mode, app_r, web_r, service_name,
                state="active", scheduler_mode="enabled",
                identity_exact=False,
                scheduler_instances=instances, pid=scheduler_pids[0],
                extra={"scheduler_without_pid_file": True},
            ),
            scheduler_pids,
            conflicts,
        )
    if conflicts:
        return (
            status_payload(
                mode, app_r, web_r, service_name,
                state="inactive", scheduler_mode="disabled",
                identity_exact=False, scheduler_instances=0, pid=None,
                extra={"conflict_pids": conflicts},
            ),
            scheduler_pids,
            conflicts,
        )
    return (
        status_payload(
            mode, app_r, web_r, service_name,
            state="inactive", scheduler_mode="disabled",
            identity_exact=True, scheduler_instances=0, pid=None,
        ),
        scheduler_pids,
        conflicts,
    )


# ── preflight ──────────────────────────────────────────────────────────────


def port_free(host: str = API_HOST, port: int = API_PORT) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
    except OSError:
        return False
    return True


def preflight_checks(
    mode: str,
    app_r: Path,
    web_r: Path,
    service_name: str,
) -> None:
    if not app_r.is_dir():
        die(f"app dir is not a real directory: {app_r}")
    service_module = app_r / "src" / "tasker" / "service.py"
    if not service_module.is_file():
        die(f"app dir must contain src/tasker/service.py: {app_r}")
    if not web_r.is_dir():
        die(f"web root is not a real directory: {web_r}")
    if not (web_r / "data").is_dir():
        die(f"web_root/data is not a real directory: {web_r / 'data'}")
    venv_python = python_exe(app_r)
    if not venv_python.is_file():
        die(f"app venv python is not a regular executable file: {venv_python}")
    if not os.access(venv_python, os.X_OK):
        die(f"app venv python is not executable: {venv_python}")
    read_env_file(box_persist_root(), mode)
    if not port_free():
        die(f"API port {API_PORT} is occupied on {API_HOST}; refusing to proceed")
    _ = service_name  # validated at the CLI layer


# ── spawn ──────────────────────────────────────────────────────────────────


def ensure_run_dirs(root: Path) -> None:
    (root / "runtime").mkdir(mode=0o700, parents=True, exist_ok=True)
    (root / "run").mkdir(mode=0o700, exist_ok=True)


def atomic_write(path: Path, data: bytes, mode: int) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".plbp-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def rollback_spawned_pid(
    pid: int,
    *,
    app_r: Path,
    web_r: Path,
    service_name: str,
    mode: str,
    root: Path,
    deadline_seconds: float | None = None,
) -> None:
    """Roll back a newly spawned child PID: wait boundedly for exact identity if
    needed, terminate exact identity boundedly, never kill an inexact process,
    and clean stale PID/state records."""
    if deadline_seconds is None:
        deadline_seconds = _env_float("PLBP_STOP_TIMEOUT", DEFAULT_STOP_TIMEOUT)
    deadline = time.monotonic() + deadline_seconds
    exact_now = False
    while time.monotonic() <= deadline:
        exact_now, _ = process_identity(
            pid, app_r=app_r, web_r=web_r, service_name=service_name, mode=mode
        )
        if exact_now:
            break
        if not probe_alive(pid):
            break
        time.sleep(0.05)
    if exact_now:
        terminate_exact(
            pid,
            app_r=app_r,
            web_r=web_r,
            service_name=service_name,
            mode=mode,
            stop_timeout=_env_float("PLBP_STOP_TIMEOUT", DEFAULT_STOP_TIMEOUT),
            kill_timeout=_env_float("PLBP_KILL_TIMEOUT", DEFAULT_KILL_TIMEOUT),
        )
    remove_stale_files(root, mode)


def spawn(
    mode: str,
    app_r: Path,
    web_r: Path,
    service_name: str,
    scheduler_pids: list[int],
    conflicts: list[int],
) -> dict[str, Any]:
    preflight_checks(mode, app_r, web_r, service_name)
    if scheduler_pids or conflicts:
        detail = (
            f"scheduler instance(s) {scheduler_pids}"
            if scheduler_pids
            else f"conflicting Tasker process(es) {conflicts}"
        )
        die(f"refusing to start {mode}: {detail} still active")
    if not port_free():
        die(f"API port {API_PORT} is occupied on {API_HOST}; refusing to start")
    root = box_persist_root()
    ensure_run_dirs(root)
    argv = expected_argv(app_r, mode)
    env = build_child_env(root, mode, app_r, web_r, service_name)
    log_path = log_file_path(root, mode)
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(log_fd, 0o600)
        proc = subprocess.Popen(
            argv,
            cwd=str(app_r),
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
                    "app_dir": str(app_r),
                    "web_root": str(web_r),
                    "started_at": started_at,
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode(),
            0o600,
        )
    except BaseException as exc:
        rollback_spawned_pid(
            pid,
            app_r=app_r,
            web_r=web_r,
            service_name=service_name,
            mode=mode,
            root=root,
        )
        die(f"failed to record {mode} ownership for pid {pid}: {exc}", exc=exc)
    deadline = time.monotonic() + _env_float("PLBP_START_TIMEOUT", DEFAULT_START_TIMEOUT)
    accepted: dict[str, Any] | None = None
    while time.monotonic() <= deadline:
        reap(pid)
        current, _schedulers, _conflicts = inspect(
            mode, app_r, web_r, service_name
        )
        if (
            current["state"] == "active"
            and current["identity_exact"] is True
            and current["pid"] == pid
            and (
                (mode == "candidate" and current["scheduler_instances"] == 0)
                or (mode == "production" and current["scheduler_instances"] == 1)
            )
        ):
            accepted = current
            break
        if time.monotonic() < deadline:
            time.sleep(0.2)
    if accepted is not None:
        return accepted
    # Failed start: terminate only the newly-spawned PID when its exact
    # identity can be proven; never kill an inexact process. Clean PID/state.
    _, _why = process_identity(
        pid, app_r=app_r, web_r=web_r, service_name=service_name, mode=mode
    )
    rollback_spawned_pid(
        pid,
        app_r=app_r,
        web_r=web_r,
        service_name=service_name,
        mode=mode,
        root=root,
    )
    die(
        f"{mode} process {pid} did not reach exact healthy identity within "
        f"PLBP_START_TIMEOUT ({_env_float('PLBP_START_TIMEOUT', DEFAULT_START_TIMEOUT)}s); "
        f"last identity check: {_why!r}"
    )


# ── termination ────────────────────────────────────────────────────────────


def proc_stopped(pid: int) -> bool:
    """Gone (real process or proc entry), zombie, or empty cmdline counts as
    stopped workload. Permission-denied or ambiguous non-ProcessLookup liveness
    must never count as safely stopped."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        reap(pid)
        return True
    except (PermissionError, OSError):
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
        if time.monotonic() < deadline:
            time.sleep(0.2)
    return False


def terminate_exact(
    pid: int,
    *,
    app_r: Path,
    web_r: Path,
    service_name: str,
    mode: str,
    stop_timeout: float,
    kill_timeout: float,
) -> None:
    """SIGTERM with bounded graceful wait; before SIGKILL escalation the exact
    identity is re-read to prevent PID-reuse killing."""
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
    exact, _why = process_identity(
        pid, app_r=app_r, web_r=web_r, service_name=service_name, mode=mode
    )
    if proc_stopped(pid):
        return  # stopped during the identity re-read
    if not exact:
        die(
            f"refusing SIGKILL: identity of pid {pid} changed during stop "
            f"({_why!r}); not signaling"
        )
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        reap(pid)
        return
    if not wait_stopped(pid, kill_timeout):
        die(f"pid {pid} did not stop even after SIGKILL")


# ── actions ────────────────────────────────────────────────────────────────


def action_status(mode: str, app_r: Path, web_r: Path, service_name: str) -> dict[str, Any]:
    payload, _schedulers, _conflicts = inspect(mode, app_r, web_r, service_name)
    return payload


def action_preflight(mode: str, app_r: Path, web_r: Path, service_name: str) -> dict[str, Any]:
    preflight_checks(mode, app_r, web_r, service_name)
    # Read-only evaluation: no stale cleanup (preflight must not mutate).
    payload, _schedulers, _conflicts = inspect(
        mode, app_r, web_r, service_name, cleanup_stale=False
    )
    payload = dict(payload)
    payload["preflight"] = {"ok": True, "config": f"{mode}.env", "port": API_PORT}
    return payload


def action_start_candidate(
    mode: str, app_r: Path, web_r: Path, service_name: str
) -> dict[str, Any]:
    if mode != "candidate":
        die(f"start-candidate requires --mode candidate; refusing --mode {mode}")
    current, scheduler_pids, conflicts = inspect(mode, app_r, web_r, service_name)
    if current["state"] == "active":
        if current["identity_exact"] is True:
            return current  # already exact: idempotent, same PID, no second start
        die(
            f"refusing start-candidate: active process with inexact identity "
            f"(pid {current['pid']}, scheduler_instances="
            f"{current['scheduler_instances']}); not signaling"
        )
    # Inactive: an exact-active process may be returned idempotently even when
    # the fixed port has since become occupied, but a fresh start must pass
    # the full preflight (including the occupied-port refusal).
    preflight_checks(mode, app_r, web_r, service_name)
    return spawn(mode, app_r, web_r, service_name, scheduler_pids, conflicts)


def action_stop(mode: str, app_r: Path, web_r: Path, service_name: str) -> dict[str, Any]:
    current, _schedulers, _conflicts = inspect(mode, app_r, web_r, service_name)
    if current["state"] == "inactive":
        if current["identity_exact"] is True:
            return current  # idempotent
        die(
            "refusing stop: ambiguous Tasker process(es) present without an owned "
            "PID file; not signaling"
        )
    if current["identity_exact"] is False:
        die(
            f"refusing stop: pid {current['pid']} has inexact identity; "
            "not signaling and not removing PID files"
        )
    pid = current["pid"]
    root = box_persist_root()
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        reap(pid)
        remove_stale_files(root, mode)
        return status_payload(
            mode, app_r, web_r, service_name,
            state="inactive", scheduler_mode="disabled",
            identity_exact=True, scheduler_instances=0, pid=None,
        )
    stop_timeout = _env_float("PLBP_STOP_TIMEOUT", DEFAULT_STOP_TIMEOUT)
    stopped = wait_stopped(pid, stop_timeout)
    if not stopped:
        # Grace expired: re-read exact identity before any escalation so a
        # reused PID is never killed.
        reap(pid)
        stopped = proc_stopped(pid)
        exact, _why = process_identity(
            pid, app_r=app_r, web_r=web_r, service_name=service_name, mode=mode
        )
        stopped = stopped or proc_stopped(pid)
    if stopped:
        remove_stale_files(root, mode)
        return status_payload(
            mode, app_r, web_r, service_name,
            state="inactive", scheduler_mode="disabled",
            identity_exact=True, scheduler_instances=0, pid=None,
        )
    if not exact:
        die(
            f"refusing SIGKILL: identity of pid {pid} changed during stop "
            f"({_why!r}); not signaling"
        )
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        reap(pid)
    else:
        if not wait_stopped(pid, _env_float("PLBP_KILL_TIMEOUT", DEFAULT_KILL_TIMEOUT)):
            die(f"pid {pid} did not stop even after SIGKILL")
    remove_stale_files(root, mode)
    return status_payload(
        mode, app_r, web_r, service_name,
        state="inactive", scheduler_mode="disabled",
        identity_exact=True, scheduler_instances=0, pid=None,
    )


def read_activation_marker(
    root: Path,
    mode: str,
    app_r: Path,
    web_r: Path,
    service_name: str,
) -> None:
    path = activation_file_path(root)
    if not path.exists():
        die(f"production activation marker missing at {path}; refusing production ensure")
    if path.is_symlink():
        die(f"production activation marker must not be a symlink: {path}")
    try:
        st = path.stat()
    except OSError as exc:
        die(f"production activation marker unreadable: {path} ({exc})")
    if not stat.S_ISREG(st.st_mode):
        die(f"production activation marker must be a regular file: {path}")
    if stat.S_IMODE(st.st_mode) != 0o600:
        die(f"production activation marker must be mode 0600: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        die(f"production activation marker is not valid JSON: {path} ({exc})")
    if not isinstance(payload, dict):
        die(f"production activation marker must be a JSON object: {path}")
    wants = [
        ("service_name", service_name, str),
        ("mode", "production", str),
        ("app_dir", str(app_r), str),
        ("web_root", str(web_r), str),
        ("argv", expected_argv(app_r, "production"), list),
    ]
    for field, expected, kind in wants:
        value = payload.get(field)
        if not isinstance(value, kind) or value != expected:
            die(
                f"production activation marker {field} mismatch "
                f"({value!r} != {expected!r}); refusing production ensure"
            )


def action_ensure(mode: str, app_r: Path, web_r: Path, service_name: str) -> dict[str, Any]:
    current, scheduler_pids, conflicts = inspect(mode, app_r, web_r, service_name)
    if current["state"] == "active":
        if current["identity_exact"] is True:
            return current  # idempotent
        die(
            f"refusing ensure: active process with inexact identity "
            f"(pid {current['pid']}); not signaling"
        )
    if scheduler_pids or conflicts:
        die(f"refusing ensure: conflicting Tasker process(es) active: {conflicts or scheduler_pids}")
    if mode == "production":
        read_activation_marker(box_persist_root(), mode, app_r, web_r, service_name)
    return spawn(mode, app_r, web_r, service_name, scheduler_pids, conflicts)


def action_activate(
    mode: str, app_r: Path, web_r: Path, service_name: str, label: str
) -> dict[str, Any]:
    if mode != "production":
        die("activate requires --mode production")
    if not label.strip():
        die("activate requires a non-whitespace --former-authority-confirmed-stopped label")
    current, scheduler_pids, conflicts = inspect(mode, app_r, web_r, service_name)
    if current["state"] == "active":
        if (
            current["identity_exact"] is True
            and current["scheduler_instances"] == 1
            and current["pid"] is not None
        ):
            return current  # repeated direct activate: never a second process
        die(
            f"refusing activate: active process state is not exactly one owned "
            f"scheduler (state={current['state']}, identity_exact="
            f"{current['identity_exact']}, scheduler_instances="
            f"{current['scheduler_instances']}, pid={current['pid']})"
        )
    if scheduler_pids or conflicts:
        detail = (
            f"scheduler instance(s) {scheduler_pids}"
            if scheduler_pids
            else f"conflicting Tasker process(es) {conflicts}"
        )
        die(f"refusing activate: {detail} still active")
    result = spawn(mode, app_r, web_r, service_name, scheduler_pids, conflicts)
    # Marker is written only after successful identity validation.
    marker = {
        "schema": SCHEMA_ACTIVATION,
        "service_name": service_name,
        "mode": mode,
        "app_dir": str(app_r),
        "web_root": str(web_r),
        "argv": expected_argv(app_r, mode),
        "former_authority_sha256": hashlib.sha256(label.encode("utf-8")).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        atomic_write(
            activation_file_path(box_persist_root()),
            json.dumps(marker, separators=(",", ":"), sort_keys=True).encode("utf-8"),
            0o600,
        )
    except BaseException as exc:
        spawned_pid = result.get("pid")
        if isinstance(spawned_pid, int) and spawned_pid > 0:
            rollback_spawned_pid(
                spawned_pid,
                app_r=app_r,
                web_r=web_r,
                service_name=service_name,
                mode=mode,
                root=box_persist_root(),
            )
        activation_file_path(box_persist_root()).unlink(missing_ok=True)
        die(
            f"failed to write production activation marker: {exc}",
            exc=exc,
        )
    return result


def install_ensure_hook(
    mode: str,
    app_r: Path,
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
                ("--app-dir", str(app_r)),
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
        atomic_write(path, new_body, stat.S_IMODE(st.st_mode))
    return {
        "schema": SCHEMA_INSTALL,
        "action": "install-ensure-hook",
        "changed": changed,
        "ensure_script": str(path),
    }


# ── CLI ────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Native box-persist lifecycle controller for Portfolio Lab."
    )
    parser.add_argument(
        "action",
        choices=_ACTIONS,
        help=f"one of: {', '.join(_ACTIONS)}",
    )
    parser.add_argument("--mode", required=True, choices=("candidate", "production"))
    parser.add_argument("--app-dir", required=True)
    parser.add_argument("--web-root", required=True)
    parser.add_argument("--service-name", required=True)
    parser.add_argument("--former-authority-confirmed-stopped", default=None)
    parser.add_argument("--ensure-script", default=ENSURE_SCRIPT_DEFAULT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_service_name(args.service_name)
    app_r, web_r = validate_targets(args.app_dir, args.web_root)
    if args.action == "install-ensure-hook":
        emit(
            install_ensure_hook(
                args.mode, app_r, web_r, args.service_name, args.ensure_script
            )
        )
        return 0
    if args.action == "preflight":
        emit(action_preflight(args.mode, app_r, web_r, args.service_name))
        return 0
    if args.action == "status":
        emit(action_status(args.mode, app_r, web_r, args.service_name))
        return 0
    if args.action == "start-candidate":
        emit(action_start_candidate(args.mode, app_r, web_r, args.service_name))
        return 0
    if args.action == "stop":
        emit(action_stop(args.mode, app_r, web_r, args.service_name))
        return 0
    if args.action == "ensure":
        emit(action_ensure(args.mode, app_r, web_r, args.service_name))
        return 0
    if args.action == "activate":
        if args.former_authority_confirmed_stopped is None:
            parser.error(
                "activate requires --former-authority-confirmed-stopped LABEL"
            )
        emit(
            action_activate(
                args.mode,
                app_r,
                web_r,
                args.service_name,
                args.former_authority_confirmed_stopped,
            )
        )
        return 0
    die(f"unknown action: {args.action}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())