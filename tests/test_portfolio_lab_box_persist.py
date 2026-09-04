"""Strict-TDD tests for the native box-persist lifecycle controller (Task 2.2).

Every test exercises the shipped CLI as ``sys.executable SCRIPT ...`` with an
isolated ``PLBP_ROOT``, a fake executable ``app/.venv/bin/python`` helper that
records received argv/environment (no secrets) and creates/removes fake
``PLBP_PROC_ROOT`` entries, and real child PIDs/signals where useful. Start/
stop/kill deadlines are lowered through the named environment variables
(PLBP_START_TIMEOUT / PLBP_STOP_TIMEOUT / PLBP_KILL_TIMEOUT); no sleep-heavy
tests.

The fake process identity mirrors /proc: ``<proc>/<pid>/{status,cmdline,
environ,exe,cwd,fd}`` plus ``<proc>/net/tcp`` (and ``tcp6``). The helper's
own PID is a real child PID in every case.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BP_SCRIPT = PROJECT_ROOT / "scripts" / "portfolio_lab_box_persist.py"
SCHEMA = "portfolio-lab-box-persist/v1"
STATE_SCHEMA = "portfolio-lab-box-persist/state/v1"
ACTIVATION_SCHEMA = "portfolio-lab-production-activation/v1"
INSTALL_SCHEMA = "portfolio-lab-box-persist/install/v1"
SERVICE = "portfolio-lab-tasker"
CONTROLLER_PATH = "/home/box/.local/bin/portfolio-lab-box-persist"
ENSURE_DEFAULT = "/home/box/.local/share/box-persist/ensure.sh"
PROD_PATH = "/home/box/.local/bin:/usr/local/bin:/usr/bin:/bin"
START_TIMEOUT = "5"
STOP_TIMEOUT = "3"
KILL_TIMEOUT = "3"

# The fake .venv/bin/python: records argv and an allowlisted env subset (no
# secrets), creates fake proc entries for its own real PID, and handles
# SIGTERM (exit / ignore / identity-switch knobs).
HELPER = r'''#!{python}
import json
import os
import shutil
import signal
import sys
import time

LOG = os.environ.get("PLBP_FAKE_HELPER_LOG", "")
PROC = os.environ.get("PLBP_FAKE_PROC_ROOT", "")
APP = os.environ.get("PLBP_FAKE_APP", "")
INODE = 100000 + os.getpid()
_ALLOW = {
    "PATH", "PYTHONPATH", "PUBLIC_DATA_DIR", "CRON_BACKEND", "LOG_LEVEL", "JSON_LOGS",
    "HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
    "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
}


def log(entry):
    if LOG:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


def recorded_env():
    env = {}
    for key, value in os.environ.items():
        if key in _ALLOW or key.startswith(("TASKER_", "PORTFOLIO_LAB_", "EXTRA_", "PROBE_")):
            env[key] = value
    return env


def cmdline_for():
    mode = os.environ.get("PLBP_FAKE_CMDLINE", "own")
    python = os.path.join(APP, ".venv", "bin", "python")
    if mode == "own":
        return sys.argv[:]
    if mode == "candidate":
        return [
            python, "-m", "src.tasker.service",
            "--host", "127.0.0.1", "--port", "8000", "--no-scheduler",
        ]
    if mode == "production":
        return [
            python, "-m", "src.tasker.service",
            "--host", "127.0.0.1", "--port", "8000",
        ]
    if mode == "once":
        return [python, "-m", "src.tasker.service", "--once"]
    return sys.argv[:]


def write_proc():
    pid = os.getpid()
    base = os.path.join(PROC, str(pid))
    os.makedirs(os.path.join(base, "fd"), exist_ok=True)
    with open(os.path.join(base, "status"), "w", encoding="utf-8") as fh:
        uid = os.getuid()
        fh.write(
            "Name:\tpython\nState:\tS (sleeping)\nPid:\t%d\nUid:\t%d\t%d\t%d\t%d\n"
            % (pid, uid, uid, uid, uid)
        )
    exe_target = os.environ.get("PLBP_FAKE_EXE_TARGET") or os.path.realpath(sys.argv[0])
    os.symlink(exe_target, os.path.join(base, "exe"))
    cwd_target = os.environ.get("PLBP_FAKE_CWD_TARGET") or os.getcwd()
    os.symlink(cwd_target, os.path.join(base, "cwd"))
    cmd = cmdline_for()
    with open(os.path.join(base, "cmdline"), "wb") as fh:
        for part in cmd:
            fh.write(part.encode("utf-8") + b"\x00")
    entries = ["%s=%s" % (k, v) for k, v in os.environ.items()]
    extra = os.environ.get("PLBP_FAKE_ENV_EXTRA")
    if extra:
        entries.append(extra)
    with open(os.path.join(base, "environ"), "wb") as fh:
        for part in entries:
            fh.write(part.encode("utf-8") + b"\x00")
    os.symlink("socket:[%d]" % INODE, os.path.join(base, "fd", "3"))
    os.makedirs(os.path.join(PROC, "net"), exist_ok=True)
    addr = os.environ.get("PLBP_FAKE_NET_ADDR", "0100007F:1F40")
    tcp = os.path.join(PROC, "net", "tcp")
    row = (
        "  %d: %s 00000000:0000 0A 00000000:00000000 00:00000000 "
        "00000000 1000 0 %d 1 0000000000000000 100 0 0 10 0\n"
        % (pid, addr, INODE)
    )
    if os.path.exists(tcp):
        with open(tcp, "a", encoding="utf-8") as fh:
            fh.write(row)
    else:
        with open(tcp, "w", encoding="utf-8") as fh:
            fh.write(
                "  sl  local_address rem_address   st tx_queue rx_queue "
                "tr tm->when retrnsmt   uid  timeout inode\n"
            )
            fh.write(row)


def remove_proc():
    base = os.path.join(PROC, str(os.getpid()))
    shutil.rmtree(os.path.join(base, "fd"), ignore_errors=True)
    for name in ("cmdline", "environ", "status", "exe", "cwd"):
        try:
            os.remove(os.path.join(base, name))
        except OSError:
            pass
    try:
        os.rmdir(base)
    except OSError:
        pass


def handle_term(_signum, _frame):
    log({"event": "term", "pid": os.getpid()})
    if os.environ.get("PLBP_FAKE_TERM_SWITCH") == "1":
        # Simulate the target's identity changing mid-stop (PID reuse guard):
        # rewrite cmdline to an unrelated argv and keep running.
        base = os.path.join(PROC, str(os.getpid()))
        with open(os.path.join(base, "cmdline"), "wb") as fh:
            fh.write(b"/bin/false\x00--replaced\x00")
        return
    if os.environ.get("PLBP_FAKE_IGNORE_TERM") == "1":
        return
    remove_proc()
    sys.exit(0)


def main():
    nonfake_plbp = sorted(
        key for key in os.environ
        if key.startswith("PLBP_") and not key.startswith("PLBP_FAKE_")
    )
    log({
        "event": "start", "pid": os.getpid(), "argv": sys.argv,
        "env": recorded_env(), "plbp_controls": nonfake_plbp,
    })
    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)
    if os.environ.get("PLBP_FAKE_SKIP_PROC") != "1":
        write_proc()
    while True:
        time.sleep(3600)


main()
'''


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def write_env_file(path: Path, content: str) -> Path:
    _write(path, content)
    os.chmod(path, 0o600)
    return path


def make_app(app: Path) -> Path:
    _write(app / "src/tasker/service.py", "def main() -> int:\n    return 0\n")
    return app


def make_web(web: Path) -> Path:
    _write(web / "index.html", "<html></html>\n")
    _write(web / "data/index.json", "{}\n")
    return web


def controlled_child_env(bp: SimpleNamespace, mode: str) -> dict[str, str]:
    """Replicate the CLI's controlled environment for direct helper spawns."""
    env = dict(bp.env)
    env.update(
        {
            "PORTFOLIO_LAB_ENABLE_ML": "0",
            "PORTFOLIO_LAB_MODE": "lab",
            "CRON_BACKEND": "tasker",
            "PORTFOLIO_LAB_PROJECT_DIR": str(bp.app),
            "PUBLIC_DATA_DIR": str(bp.web / "data"),
            "PYTHONPATH": str(bp.app),
            "LOG_LEVEL": "INFO",
            "JSON_LOGS": "1",
            "TASKER_HOST": "127.0.0.1",
            "TASKER_PORT": "8000",
            "PORTFOLIO_LAB_BOX_PERSIST_MODE": mode,
            "PORTFOLIO_LAB_BOX_PERSIST_SERVICE": SERVICE,
            "PATH": bp.env["PLBP_PATH"],
        }
    )
    if mode == "candidate":
        env["TASKER_DISABLE_SCHEDULER"] = "1"
    else:
        env.pop("TASKER_DISABLE_SCHEDULER", None)
    return env


def make_bp(tmp_path: Path, root_name: str = "plbp-root") -> SimpleNamespace:
    root = tmp_path / root_name
    root.mkdir(parents=True, exist_ok=True)
    app = make_app(root / "app")
    web = make_web(root / "www")
    proc_root = tmp_path / "proc-root"
    helper_log = tmp_path / "helpers.jsonl"
    helper = app / ".venv" / "bin" / "python"
    _write(helper, HELPER.replace("{python}", sys.executable))
    helper.chmod(0o755)
    env = {
        "PLBP_ROOT": str(root),
        "PLBP_PROC_ROOT": str(proc_root),
        "PLBP_START_TIMEOUT": START_TIMEOUT,
        "PLBP_STOP_TIMEOUT": STOP_TIMEOUT,
        "PLBP_KILL_TIMEOUT": KILL_TIMEOUT,
        "PLBP_PATH": "/usr/bin:/bin",
        "PLBP_FAKE_PROC_ROOT": str(proc_root),
        "PLBP_FAKE_HELPER_LOG": str(helper_log),
        "PLBP_FAKE_APP": str(app),
    }
    write_env_file(root / "runtime" / "candidate.env", "TASKER_EXTRA=hello\n")
    write_env_file(root / "runtime" / "production.env", "TASKER_EXTRA=hello\n")
    return SimpleNamespace(
        tmp=tmp_path,
        root=root,
        app=app,
        web=web,
        proc=proc_root,
        env=env,
        helper=helper,
        helper_log=helper_log,
        spawned=[],
    )


def cleanup_bp(bp: SimpleNamespace) -> None:
    pids: set[int] = set()
    for pid_file in (
        bp.root / "run" / "tasker-candidate.pid",
        bp.root / "run" / "tasker-production.pid",
    ):
        if pid_file.exists():
            try:
                pids.add(int(pid_file.read_text(encoding="utf-8").strip()))
            except ValueError:
                pass
    for proc in bp.spawned:
        pids.add(proc.pid)
    if bp.helper_log.exists():
        for line in bp.helper_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("event") == "start":
                pids.add(entry["pid"])
    for pid in pids:
        if pid <= 0:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for proc in bp.spawned:
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


@pytest.fixture
def bp(tmp_path: Path):
    ns = make_bp(tmp_path)
    yield ns
    cleanup_bp(ns)


@pytest.fixture
def occupied_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 8000))
    sock.listen(1)
    yield sock
    sock.close()


# ── runner helpers ─────────────────────────────────────────────────────────


def run_cli(
    bp: SimpleNamespace,
    *args: str,
    env_extra: dict[str, str] | None = None,
    timeout: int = 90,
) -> subprocess.CompletedProcess[str]:
    env = {**bp.env}
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(BP_SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def bp_args(
    bp: SimpleNamespace,
    action: str,
    mode: str = "candidate",
    service: str = SERVICE,
    app: Path | None = None,
    web: Path | None = None,
    **flags: str,
) -> list[str]:
    args = [
        action,
        "--mode",
        mode,
        "--app-dir",
        str(app or bp.app),
        "--web-root",
        str(web or bp.web),
        "--service-name",
        service,
    ]
    for key, value in flags.items():
        args += ["--" + key.replace("_", "-"), value]
    return args


def ok_cli(
    bp: SimpleNamespace,
    *args: str,
    env_extra: dict[str, str] | None = None,
) -> tuple[dict, subprocess.CompletedProcess[str]]:
    res = run_cli(bp, *args, env_extra=env_extra)
    assert res.returncode == 0, f"rc={res.returncode} stderr={res.stderr!r}"
    lines = [line for line in res.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, f"expected exactly one stdout line, got {lines!r}"
    payload = json.loads(lines[0])
    assert isinstance(payload, dict)
    return payload, res


def fail_cli(bp: SimpleNamespace, *args: str, env_extra: dict[str, str] | None = None):
    res = run_cli(bp, *args, env_extra=env_extra)
    assert res.returncode != 0, f"expected failure, got rc=0 stdout={res.stdout!r}"
    assert res.stdout.strip() == "", f"failure must not emit stdout: {res.stdout!r}"
    return res


def assert_status_schema(payload: dict, *, mode: str) -> None:
    assert payload["schema"] == SCHEMA
    assert payload["state"] in ("active", "inactive")
    assert payload["scheduler_mode"] in ("enabled", "disabled")
    assert isinstance(payload["identity_exact"], bool)
    assert isinstance(payload["scheduler_instances"], int)
    assert payload["scheduler_instances"] >= 0
    assert payload["pid"] is None or (isinstance(payload["pid"], int) and payload["pid"] > 0)
    assert payload["service_name"] == SERVICE
    assert payload["mode"] == mode


def helper_lines(bp: SimpleNamespace) -> list[dict]:
    if not bp.helper_log.exists():
        return []
    out = []
    for line in bp.helper_log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def spawn_entries(bp: SimpleNamespace) -> list[dict]:
    return [entry for entry in helper_lines(bp) if entry.get("event") == "start"]


def child_env(bp: SimpleNamespace) -> dict[str, str]:
    starts = spawn_entries(bp)
    assert starts, "helper never started"
    return dict(starts[-1]["env"])


def spawn_direct(
    bp: SimpleNamespace,
    args: list[str],
    mode: str = "candidate",
    knobs: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.Popen:
    env = controlled_child_env(bp, mode)
    if knobs:
        env.update(knobs)
    proc = subprocess.Popen(
        [str(bp.helper), *args],
        env=env,
        cwd=str(cwd or bp.app),
        stdin=subprocess.DEVNULL,
    )
    bp.spawned.append(proc)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (bp.proc / str(proc.pid) / "status").exists():
            break
        time.sleep(0.05)
    return proc


def proc_dir(bp: SimpleNamespace, pid: int) -> Path:
    return bp.proc / str(pid)


def write_pid_file(bp: SimpleNamespace, mode: str, pid: int) -> Path:
    path = bp.root / "run" / f"tasker-{mode}.pid"
    _write(path, f"{pid}\n")
    os.chmod(path, 0o600)
    return path


def rewrite_environ(bp: SimpleNamespace, pid: int, entries: list[str]) -> None:
    """Replace the fake proc environ with explicit KEY=VALUE entries."""
    data = b"".join(entry.encode("utf-8") + b"\x00" for entry in entries)
    (proc_dir(bp, pid) / "environ").write_bytes(data)


def environ_entries(bp: SimpleNamespace, pid: int) -> list[str]:
    data = (proc_dir(bp, pid) / "environ").read_bytes()
    parts = data.split(b"\x00")
    if parts and parts[-1] == b"":
        parts.pop()
    return [part.decode("utf-8", "replace") for part in parts]


def rewrite_cmdline(bp: SimpleNamespace, pid: int, argv: list[str]) -> None:
    (proc_dir(bp, pid) / "cmdline").write_bytes(
        b"".join(part.encode("utf-8") + b"\x00" for part in argv)
    )


def rewrite_status(bp: SimpleNamespace, pid: int, state_line: str = "Z (zombie)") -> None:
    _write(proc_dir(bp, pid) / "status", f"Name:\tpython\nState:\t{state_line}\nPid:\t{pid}\n")


def replace_symlink(path: Path, target: str) -> None:
    path.unlink()
    os.symlink(target, path)


def write_net_tcp(bp: SimpleNamespace, rows: list[tuple[str, int]]) -> None:
    """Rewrite proc/net/tcp with LISTEN rows; rows are (hex_addr, hex_port)."""
    header = (
        "  sl  local_address rem_address   st tx_queue rx_queue "
        "tr tm->when retrnsmt   uid  timeout inode\n"
    )
    lines = [header]
    for index, (addr, port) in enumerate(rows):
        lines.append(
            f"  {index}: {addr}:{port} 00000000:0000 0A 00000000:00000000 "
            "00:00000000 00000000 1000 0 500001 1 0000000000000000 100 0 0 10 0\n"
        )
    (bp.proc / "net" / "tcp").write_text("".join(lines), encoding="utf-8")


def write_net_tcp6(bp: SimpleNamespace, rows: list[tuple[str, str]]) -> None:
    header = (
        "  sl  local_address                         remote_address                        "
        "st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
    )
    lines = [header]
    for index, (addr, port) in enumerate(rows):
        lines.append(
            f"  {index}: {addr}:{port} 00000000000000000000000000000000:0000 0A "
            "00000000:00000000 00:00000000 00000000 1000 0 500002 1 "
            "0000000000000000 100 0 0 10 0\n"
        )
    (bp.proc / "net" / "tcp6").write_text("".join(lines), encoding="utf-8")


def set_fd_socket(bp: SimpleNamespace, pid: int, inode: int) -> None:
    fd_dir = proc_dir(bp, pid) / "fd"
    for entry in fd_dir.iterdir():
        entry.unlink()
    os.symlink(f"socket:[{inode}]", fd_dir / "3")


def imode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def wait_gone(pid: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f"pid {pid} still alive")


# ── 1. parser/action/mode restrictions + exact normalized schema/echo ──────


def test_unknown_action_fails_through_argparse(bp):
    res = run_cli(bp, "frobnicate", "--mode", "candidate", "--app-dir", str(bp.app),
                  "--web-root", str(bp.web), "--service-name", SERVICE)
    assert res.returncode != 0
    assert "invalid choice" in res.stderr


def test_unknown_option_fails_through_argparse(bp):
    res = run_cli(bp, "status", "--bogus", "x")
    assert res.returncode != 0
    assert "error:" in res.stderr


def test_invalid_mode_choice_fails(bp):
    res = run_cli(bp, "status", "--mode", "staging", "--app-dir", str(bp.app),
                  "--web-root", str(bp.web), "--service-name", SERVICE)
    assert res.returncode != 0
    assert "invalid choice" in res.stderr


def test_start_candidate_requires_candidate_mode(bp):
    res = fail_cli(bp, *bp_args(bp, "start-candidate", mode="production"))
    assert "production" in res.stderr


def test_activate_requires_production_and_label(bp):
    res = fail_cli(bp, *bp_args(bp, "activate", mode="candidate"))
    assert "production" in res.stderr
    res = run_cli(bp, *bp_args(bp, "activate", mode="production"))
    assert res.returncode != 0
    assert "former-authority" in res.stderr


def test_activate_rejects_whitespace_label(bp):
    res = fail_cli(bp, *bp_args(bp, "activate", mode="production",
                                former_authority_confirmed_stopped="   "))
    assert "former-authority" in res.stderr


def test_service_name_validation(bp):
    res = fail_cli(bp, *bp_args(bp, "status", service="bad/name"))
    assert "service" in res.stderr
    res = fail_cli(bp, *bp_args(bp, "status", service=""))
    assert res.returncode != 0
    res = fail_cli(bp, *bp_args(bp, "status", service="bad%name"))
    assert "service" in res.stderr
    payload, _ = ok_cli(bp, *bp_args(bp, "status", service="ok.name_9@x-y"))
    assert payload["service_name"] == "ok.name_9@x-y"


def test_status_normalized_schema_and_echo(bp):
    payload, res = ok_cli(bp, *bp_args(bp, "status"))
    assert_status_schema(payload, mode="candidate")
    assert payload == {
        "schema": SCHEMA,
        "state": "inactive",
        "scheduler_mode": "disabled",
        "identity_exact": True,
        "scheduler_instances": 0,
        "pid": None,
        "service_name": SERVICE,
        "mode": "candidate",
        "app_dir": str(bp.app),
        "web_root": str(bp.web),
    }


# ── 2. paths must be absolute, beneath root, non-symlink, distinct ──────────


def test_relative_paths_rejected(bp):
    res = fail_cli(bp, "status", "--mode", "candidate", "--app-dir", "relative/app",
                   "--web-root", str(bp.web), "--service-name", SERVICE)
    assert "absolute" in res.stderr
    res = fail_cli(bp, "status", "--mode", "candidate", "--app-dir", str(bp.app),
                   "--web-root", "relative/www", "--service-name", SERVICE)
    assert "absolute" in res.stderr


def test_paths_outside_root_rejected(bp):
    outside = bp.tmp / "outside"
    outside.mkdir()
    res = fail_cli(bp, "status", "--mode", "candidate", "--app-dir", str(outside),
                   "--web-root", str(bp.web), "--service-name", SERVICE)
    assert "beneath" in res.stderr
    res = fail_cli(bp, "status", "--mode", "candidate", "--app-dir", str(bp.app),
                   "--web-root", str(outside), "--service-name", SERVICE)
    assert "beneath" in res.stderr


def test_paths_must_be_distinct_and_non_overlapping(bp):
    res = fail_cli(bp, "status", "--mode", "candidate", "--app-dir", str(bp.app),
                   "--web-root", str(bp.app), "--service-name", SERVICE)
    assert "distinct" in res.stderr
    nested = bp.app / "www"
    nested.mkdir()
    res = fail_cli(bp, "status", "--mode", "candidate", "--app-dir", str(bp.app),
                   "--web-root", str(nested), "--service-name", SERVICE)
    assert "overlap" in res.stderr


def test_symlinked_paths_rejected(bp):
    link = bp.tmp / "app-link"
    os.symlink(bp.app, link)
    res = fail_cli(bp, "status", "--mode", "candidate", "--app-dir", str(link),
                   "--web-root", str(bp.web), "--service-name", SERVICE)
    assert "symlink" in res.stderr
    web_link = bp.tmp / "www-link"
    os.symlink(bp.web, web_link)
    res = fail_cli(bp, "status", "--mode", "candidate", "--app-dir", str(bp.app),
                   "--web-root", str(web_link), "--service-name", SERVICE)
    assert "symlink" in res.stderr


def test_non_absolute_plbp_root_rejected(bp):
    res = run_cli(bp, "status", *bp_args(bp, "status")[1:],
                  env_extra={"PLBP_ROOT": "relative/root"})
    assert res.returncode != 0
    assert "absolute" in res.stderr


# ── 3. status absent/stale/zombie; live inexact never removed or signaled ──


def test_status_absent_pid_file_is_inactive_exact(bp):
    payload, _ = ok_cli(bp, *bp_args(bp, "status"))
    assert payload["state"] == "inactive"
    assert payload["scheduler_mode"] == "disabled"
    assert payload["identity_exact"] is True
    assert payload["scheduler_instances"] == 0
    assert payload["pid"] is None
    assert not (bp.root / "run").exists()


def test_status_stale_pid_cleaned(bp):
    write_pid_file(bp, "candidate", 999999)
    state = bp.root / "run" / "tasker-candidate-state.json"
    _write(state, '{"pid": 999999}\n')
    payload, _ = ok_cli(bp, *bp_args(bp, "status"))
    assert payload["state"] == "inactive"
    assert not (bp.root / "run" / "tasker-candidate.pid").exists()
    assert not state.exists()


def test_status_garbage_pid_file_cleaned(bp):
    path = write_pid_file(bp, "candidate", 0)
    path.unlink()
    _write(path, "not-a-pid\n")
    os.chmod(path, 0o600)
    payload, _ = ok_cli(bp, *bp_args(bp, "status"))
    assert payload["state"] == "inactive"
    assert not path.exists()


def test_preflight_leaves_garbage_pid_and_state_unchanged(bp):
    pid_file = bp.root / "run" / "tasker-candidate.pid"
    _write(pid_file, "not-a-pid\n")
    os.chmod(pid_file, 0o600)
    state_file = bp.root / "run" / "tasker-candidate-state.json"
    _write(state_file, '{"pid": "garbage"}\n')
    payload, _ = ok_cli(bp, *bp_args(bp, "preflight"))
    assert payload["state"] == "inactive"
    assert payload["identity_exact"] is True
    assert pid_file.read_text(encoding="utf-8") == "not-a-pid\n"
    assert state_file.read_text(encoding="utf-8") == '{"pid": "garbage"}\n'


def test_preflight_leaves_zero_pid_record_unchanged(bp):
    pid_file = bp.root / "run" / "tasker-candidate.pid"
    _write(pid_file, "0\n")
    os.chmod(pid_file, 0o600)
    payload, _ = ok_cli(bp, *bp_args(bp, "preflight"))
    assert payload["state"] == "inactive"
    assert payload["identity_exact"] is True
    assert pid_file.read_text(encoding="utf-8") == "0\n"


def test_status_zombie_cleaned_without_signal(bp):
    proc = spawn_direct(bp, ["-m", "src.tasker.service", "--no-scheduler"])
    write_pid_file(bp, "candidate", proc.pid)
    rewrite_status(bp, proc.pid)
    payload, _ = ok_cli(bp, *bp_args(bp, "status"))
    assert payload == {
        "schema": SCHEMA,
        "state": "inactive",
        "scheduler_mode": "disabled",
        "identity_exact": True,
        "scheduler_instances": 0,
        "pid": None,
        "service_name": SERVICE,
        "mode": "candidate",
        "app_dir": str(bp.app),
        "web_root": str(bp.web),
    }
    assert not (bp.root / "run" / "tasker-candidate.pid").exists()
    assert not (bp.root / "run" / "tasker-candidate-state.json").exists()
    # The real process was never signaled (no term event) and is still alive.
    assert all(e.get("event") != "term" for e in helper_lines(bp))
    os.kill(proc.pid, 0)


def test_status_live_inexact_pid_never_removed_or_signaled(bp):
    proc = spawn_direct(bp, ["-m", "src.tasker.service", "--host", "127.0.0.1",
                             "--port", "8000", "--no-scheduler"])
    write_pid_file(bp, "candidate", proc.pid)
    rewrite_cmdline(bp, proc.pid, [str(bp.helper), "-m", "src.tasker.service",
                                   "--host", "127.0.0.1", "--port", "9999"])
    payload, _ = ok_cli(bp, *bp_args(bp, "status"))
    assert payload["state"] == "active"
    assert payload["identity_exact"] is False
    assert payload["pid"] == proc.pid
    assert (bp.root / "run" / "tasker-candidate.pid").exists()
    assert all(e.get("event") != "term" for e in helper_lines(bp))
    os.kill(proc.pid, 0)


# ── PID record hardening: ownership records must be 0600 non-symlink ───────


def test_symlinked_pid_record_fails_closed_without_signal_or_delete(bp):
    target = bp.tmp / "victim.pid"
    _write(target, "424242\n")
    os.chmod(target, 0o600)
    (bp.root / "run").mkdir()
    record = bp.root / "run" / "tasker-candidate.pid"
    os.symlink(target, record)
    res = fail_cli(bp, *bp_args(bp, "status"))
    assert "pid file" in res.stderr.lower() or "symlink" in res.stderr.lower()
    assert os.path.islink(record)
    assert target.read_text(encoding="utf-8") == "424242\n"
    assert all(e.get("event") != "term" for e in helper_lines(bp))


def test_group_readable_pid_record_fails_closed_without_cleanup(bp):
    pid_file = bp.root / "run" / "tasker-candidate.pid"
    _write(pid_file, "424242\n")
    os.chmod(pid_file, 0o644)
    res = fail_cli(bp, *bp_args(bp, "status"))
    assert "0600" in res.stderr
    assert pid_file.exists()
    assert pid_file.read_text(encoding="utf-8") == "424242\n"
    state = bp.root / "run" / "tasker-candidate-state.json"
    _write(state, '{"pid": 424242}\n')
    res = fail_cli(bp, *bp_args(bp, "status"))
    assert state.exists()
    assert state.read_text(encoding="utf-8") == '{"pid": 424242}\n'


# ── 4. exact identity rejection matrix ─────────────────────────────────────


@pytest.mark.parametrize(
    "mutation",
    [
        "exe",
        "cwd",
        "argv",
        "app_marker",
        "no_socket",
        "wildcard_bind",
        "non_loopback_bind",
        "wrong_port",
        "candidate_missing_env_disable",
        "candidate_missing_argv_disable",
    ],
)
def test_candidate_identity_rejections(bp, mutation):
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"))
    assert payload["identity_exact"] is True
    pid = payload["pid"]
    entries = environ_entries(bp, pid)
    if mutation == "exe":
        replace_symlink(proc_dir(bp, pid) / "exe", str(bp.web / "index.html"))
    elif mutation == "cwd":
        replace_symlink(proc_dir(bp, pid) / "cwd", str(bp.web))
    elif mutation == "argv":
        rewrite_cmdline(bp, pid, [str(bp.helper), "-m", "src.tasker.service",
                                  "--host", "127.0.0.1", "--port", "8000", "--port", "8001"])
    elif mutation == "app_marker":
        rewrite_environ(bp, pid, [e for e in entries
                                  if not e.startswith("PORTFOLIO_LAB_PROJECT_DIR=")])
    elif mutation == "no_socket":
        set_fd_socket(bp, pid, 1)  # inode 1 has no LISTEN row
    elif mutation == "wildcard_bind":
        write_net_tcp(bp, [("00000000", "1F40")])
        set_fd_socket(bp, pid, 500001)
    elif mutation == "non_loopback_bind":
        write_net_tcp(bp, [("0A000001", "1F40")])
        set_fd_socket(bp, pid, 500001)
    elif mutation == "wrong_port":
        write_net_tcp(bp, [("0100007F", "1F41")])
        set_fd_socket(bp, pid, 500001)
    elif mutation == "candidate_missing_env_disable":
        rewrite_environ(bp, pid, [e for e in entries
                                  if e != "TASKER_DISABLE_SCHEDULER=1"])
    elif mutation == "candidate_missing_argv_disable":
        rewrite_cmdline(bp, pid, [str(bp.helper), "-m", "src.tasker.service",
                                  "--host", "127.0.0.1", "--port", "8000"])
    payload, _ = ok_cli(bp, *bp_args(bp, "status"))
    assert payload["state"] == "active"
    assert payload["identity_exact"] is False
    assert payload["pid"] == pid
    assert (bp.root / "run" / "tasker-candidate.pid").exists()


@pytest.mark.parametrize("mutation", ["env_disable", "argv_disable"])
def test_production_identity_rejections(bp, mutation):
    payload, _ = ok_cli(bp, *bp_args(
        bp, "activate", mode="production",
        former_authority_confirmed_stopped="sg01 stopped 2026-09-03"))
    assert payload["identity_exact"] is True
    pid = payload["pid"]
    if mutation == "env_disable":
        entries = environ_entries(bp, pid)
        rewrite_environ(bp, pid, entries + ["TASKER_DISABLE_SCHEDULER=1"])
    else:
        argv = [str(bp.helper), "-m", "src.tasker.service",
                "--host", "127.0.0.1", "--port", "8000", "--no-scheduler"]
        rewrite_cmdline(bp, pid, argv)
    payload, _ = ok_cli(bp, *bp_args(bp, "status", mode="production"))
    assert payload["state"] == "active"
    assert payload["identity_exact"] is False
    assert payload["pid"] == pid


def test_ipv6_loopback_binding_accepted(bp):
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"))
    pid = payload["pid"]
    write_net_tcp6(bp, [("00000000000000000000000001000000", "1F40")])
    set_fd_socket(bp, pid, 500002)
    payload, _ = ok_cli(bp, *bp_args(bp, "status"))
    assert payload["state"] == "active"
    assert payload["identity_exact"] is True


def test_port_ownership_is_by_inode(bp):
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"))
    pid = payload["pid"]
    # A LISTEN row exists for a different inode: this PID owns no loopback
    # 8000 socket.
    set_fd_socket(bp, pid, 777777)
    payload, _ = ok_cli(bp, *bp_args(bp, "status"))
    assert payload["identity_exact"] is False


# ── 5. candidate config separation, parsing, permissions, secrets ──────────


def test_configs_are_separate(bp):
    (bp.root / "runtime" / "candidate.env").unlink()
    payload, _ = ok_cli(bp, *bp_args(bp, "status"))
    assert payload["state"] == "inactive"
    res = fail_cli(bp, *bp_args(bp, "start-candidate"))
    assert "candidate.env" in res.stderr


def test_config_parses_quotes_comments_and_blanks(bp):
    write_env_file(
        bp.root / "runtime" / "candidate.env",
        "# comment line\n\nEXTRA_SETTING=plain\nTASKER_EXTRA='quoted value'\n"
        'TASKER_QUOTED="double quotes"\n',
    )
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"))
    assert payload["identity_exact"] is True
    env = child_env(bp)
    assert env["EXTRA_SETTING"] == "plain"
    assert env["TASKER_EXTRA"] == "quoted value"
    assert env["TASKER_QUOTED"] == "double quotes"


@pytest.mark.parametrize(
    "bad_content",
    [
        "NOEQUALS\n",
        "BAD KEY=x\n",
        "1BAD=x\n",
        "K='unterminated\n",
        "K=va\tlue\n",
    ],
)
def test_config_malformed_lines_rejected_without_mutation(bp, bad_content):
    write_env_file(bp.root / "runtime" / "candidate.env", bad_content)
    res = fail_cli(bp, *bp_args(bp, "preflight"))
    assert "candidate.env" in res.stderr or "malformed" in res.stderr
    assert not (bp.root / "run").exists()


def test_config_nul_byte_rejected(bp):
    path = bp.root / "runtime" / "candidate.env"
    path.write_bytes(b"K=value\x00with-nul\n")
    os.chmod(path, 0o600)
    os.chmod(path, 0o600)
    res = fail_cli(bp, *bp_args(bp, "preflight"))
    assert "NUL" in res.stderr


@pytest.mark.parametrize(
    "bad_content",
    [
        "SENTINEL_9f8d77_missing_eq\n",
        "BAD SENTINEL_9f8d77_key=x\n",
        "K='SENTINEL_9f8d77_unterm\n",
        "K=va\tSENTINEL_9f8d77_ctrl\n",
    ],
)
def test_malformed_config_diagnostics_never_include_lines_or_values(bp, bad_content):
    write_env_file(bp.root / "runtime" / "candidate.env", bad_content)
    res = fail_cli(bp, *bp_args(bp, "preflight"))
    assert "SENTINEL_9f8d77" not in res.stderr
    assert "line 1" in res.stderr
    assert not (bp.root / "run").exists()


def test_malformed_config_nul_diagnostics_never_include_lines_or_values(bp):
    path = bp.root / "runtime" / "candidate.env"
    path.write_bytes(b"K=SENTINEL_9f8d77_nul\x00x\n")
    os.chmod(path, 0o600)
    res = fail_cli(bp, *bp_args(bp, "preflight"))
    assert "SENTINEL_9f8d77_nul" not in res.stderr
    assert "NUL" in res.stderr
    assert not (bp.root / "run").exists()


def test_config_permissions_required(bp):
    os.chmod(bp.root / "runtime" / "candidate.env", 0o644)
    res = fail_cli(bp, *bp_args(bp, "start-candidate"))
    assert "permissions" in res.stderr
    os.chmod(bp.root / "runtime" / "candidate.env", 0o600)
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"))
    assert payload["identity_exact"] is True


def test_config_symlink_rejected(bp):
    (bp.root / "runtime" / "candidate.env").unlink()
    os.symlink(bp.tmp / "elsewhere.env", bp.root / "runtime" / "candidate.env")
    res = fail_cli(bp, *bp_args(bp, "start-candidate"))
    assert "symlink" in res.stderr


def test_config_secrets_never_printed_or_logged(bp):
    secret_values = ["sk-secret123", "hunter2segredo"]
    write_env_file(
        bp.root / "runtime" / "candidate.env",
        "API_KEY=sk-secret123\nDB_PASSWORD=hunter2segredo\nTASKER_TOKEN=zzz\n",
    )
    payload, res = ok_cli(bp, *bp_args(bp, "start-candidate"))
    assert payload["identity_exact"] is True
    blob = res.stdout + res.stderr + bp.helper_log.read_text(encoding="utf-8")
    for secret in secret_values:
        assert secret not in blob
    state_file = bp.root / "run" / "tasker-candidate-state.json"
    assert all(secret not in state_file.read_text(encoding="utf-8") for secret in secret_values)
    pid_file = bp.root / "run" / "tasker-candidate.pid"
    assert all(secret not in pid_file.read_text(encoding="utf-8") for secret in secret_values)
    env = child_env(bp)
    assert "API_KEY" not in env
    assert "DB_PASSWORD" not in env


def test_controlled_fields_always_override_config(bp):
    write_env_file(
        bp.root / "runtime" / "candidate.env",
        "PORTFOLIO_LAB_ENABLE_ML=1\nPORTFOLIO_LAB_MODE=weird\nCRON_BACKEND=systemd\n"
        "PORTFOLIO_LAB_PROJECT_DIR=/evil\nPUBLIC_DATA_DIR=/evil\nPYTHONPATH=/evil\n"
        "LOG_LEVEL=DEBUG\nJSON_LOGS=0\nTASKER_HOST=0.0.0.0\nTASKER_PORT=9999\n"
        "PORTFOLIO_LAB_BOX_PERSIST_MODE=production\n"
        "PORTFOLIO_LAB_BOX_PERSIST_SERVICE=other\nPATH=/evil\n"
        "TASKER_DISABLE_SCHEDULER=0\n",
    )
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"))
    assert payload["identity_exact"] is True
    env = child_env(bp)
    assert env["PORTFOLIO_LAB_ENABLE_ML"] == "0"
    assert env["PORTFOLIO_LAB_MODE"] == "lab"
    assert env["CRON_BACKEND"] == "tasker"
    assert env["PORTFOLIO_LAB_PROJECT_DIR"] == str(bp.app)
    assert env["PUBLIC_DATA_DIR"] == str(bp.web / "data")
    assert env["PYTHONPATH"] == str(bp.app)
    assert env["LOG_LEVEL"] == "INFO"
    assert env["JSON_LOGS"] == "1"
    assert env["TASKER_HOST"] == "127.0.0.1"
    assert env["TASKER_PORT"] == "8000"
    assert env["PORTFOLIO_LAB_BOX_PERSIST_MODE"] == "candidate"
    assert env["PORTFOLIO_LAB_BOX_PERSIST_SERVICE"] == SERVICE
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["TASKER_DISABLE_SCHEDULER"] == "1"


def test_child_env_does_not_inherit_controller_secrets(bp):
    sentinel = "SENTINEL_9f8d77"
    payload, res = ok_cli(bp, *bp_args(bp, "start-candidate"),
                          env_extra={"PROBE_UNRELATED_SECRET": sentinel})
    assert payload["identity_exact"] is True
    env = child_env(bp)
    assert "PROBE_UNRELATED_SECRET" not in env
    controls = spawn_entries(bp)[-1]["plbp_controls"]
    assert "PLBP_ROOT" not in controls
    assert "PLBP_PROC_ROOT" not in controls
    assert "PLBP_START_TIMEOUT" not in controls
    blob = res.stdout + res.stderr + bp.helper_log.read_text(encoding="utf-8")
    assert sentinel not in blob
    state_file = bp.root / "run" / "tasker-candidate-state.json"
    assert sentinel not in state_file.read_text(encoding="utf-8")


def test_child_env_allowlist_passes_safe_basics_and_filters_others(bp):
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"),
                        env_extra={"LANG": "C.UTF-8", "TZ": "UTC",
                                   "PROBE_IRRELEVANT_SETTING": "zzz9"})
    assert payload["identity_exact"] is True
    env = child_env(bp)
    assert env["LANG"] == "C.UTF-8"
    assert env["TZ"] == "UTC"
    assert "PROBE_IRRELEVANT_SETTING" not in env
    assert "zzz9" not in bp.helper_log.read_text(encoding="utf-8")


def test_env_file_secrets_still_reach_child(bp):
    write_env_file(bp.root / "runtime" / "candidate.env",
                   "TASKER_OPERATOR_TOKEN=vault-secret-abc\n")
    payload, res = ok_cli(bp, *bp_args(bp, "start-candidate"))
    assert payload["identity_exact"] is True
    assert child_env(bp)["TASKER_OPERATOR_TOKEN"] == "vault-secret-abc"
    assert "vault-secret-abc" not in res.stdout + res.stderr
    state_file = bp.root / "run" / "tasker-candidate-state.json"
    assert "vault-secret-abc" not in state_file.read_text(encoding="utf-8")


# ── 6. preflight validation without mutation ───────────────────────────────


def test_preflight_happy_path_without_mutation(bp):
    payload, _ = ok_cli(bp, *bp_args(bp, "preflight"))
    assert_status_schema(payload, mode="candidate")
    assert payload["state"] == "inactive"
    assert payload["preflight"]["ok"] is True
    assert not (bp.root / "run").exists()
    assert sorted(p.name for p in (bp.root / "runtime").iterdir()) == [
        "candidate.env",
        "production.env",
    ]


def test_preflight_validates_app_layout(bp):
    (bp.app / "src" / "tasker" / "service.py").unlink()
    res = fail_cli(bp, *bp_args(bp, "preflight"))
    assert "service.py" in res.stderr


def test_preflight_validates_venv_python(bp):
    bp.helper.unlink()
    res = fail_cli(bp, *bp_args(bp, "preflight"))
    assert "python" in res.stderr


def test_preflight_validates_web_data_dir(bp):
    shutil.rmtree(bp.web / "data")
    res = fail_cli(bp, *bp_args(bp, "preflight"))
    assert "data" in res.stderr


def test_preflight_rejects_occupied_port(bp, occupied_port):
    res = fail_cli(bp, *bp_args(bp, "preflight"))
    assert "8000" in res.stderr


def test_status_works_without_config(bp):
    (bp.root / "runtime" / "candidate.env").unlink()
    (bp.root / "runtime" / "production.env").unlink()
    payload, _ = ok_cli(bp, *bp_args(bp, "status", mode="production"))
    assert payload["state"] == "inactive"


# ── 7. start-candidate: controls, loopback-only, exact, idempotent ─────────


def test_start_candidate_exact_status_and_files(bp):
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"))
    assert payload["state"] == "active"
    assert payload["scheduler_mode"] == "disabled"
    assert payload["identity_exact"] is True
    assert payload["scheduler_instances"] == 0
    assert payload["pid"] > 0
    pid = payload["pid"]
    pid_file = bp.root / "run" / "tasker-candidate.pid"
    assert int(pid_file.read_text(encoding="utf-8").strip()) == pid
    assert imode(pid_file) == 0o600
    state_file = bp.root / "run" / "tasker-candidate-state.json"
    assert imode(state_file) == 0o600
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["schema"] == STATE_SCHEMA
    assert state["pid"] == pid
    assert state["mode"] == "candidate"
    assert state["service_name"] == SERVICE
    log_file = bp.root / "run" / "tasker-candidate.log"
    assert imode(log_file) == 0o600
    start = spawn_entries(bp)[-1]
    assert start["argv"] == [
        str(bp.helper), "-m", "src.tasker.service",
        "--host", "127.0.0.1", "--port", "8000", "--no-scheduler",
    ]
    env = start["env"]
    assert env["TASKER_DISABLE_SCHEDULER"] == "1"
    assert env["TASKER_HOST"] == "127.0.0.1"
    assert env["TASKER_PORT"] == "8000"
    assert env["PORTFOLIO_LAB_BOX_PERSIST_MODE"] == "candidate"
    assert env["PORTFOLIO_LAB_BOX_PERSIST_SERVICE"] == SERVICE
    payload, _ = ok_cli(bp, *bp_args(bp, "status"))
    assert payload["pid"] == pid


def test_start_candidate_idempotent_same_pid(bp):
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"))
    first = payload["pid"]
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"))
    assert payload["pid"] == first
    assert len(spawn_entries(bp)) == 1


def test_start_candidate_writes_env_file_values_passthrough(bp):
    write_env_file(bp.root / "runtime" / "candidate.env", "EXTRA_SETTING=from-file\n")
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"))
    assert payload["identity_exact"] is True
    assert child_env(bp)["EXTRA_SETTING"] == "from-file"


# ── 8. candidate start refusals ────────────────────────────────────────────


def test_start_candidate_refuses_active_production(bp):
    payload, _ = ok_cli(
        bp, *bp_args(bp, "activate", mode="production",
                     former_authority_confirmed_stopped="proof"))
    prod_pid = payload["pid"]
    res = fail_cli(bp, *bp_args(bp, "start-candidate"))
    assert "conflict" in res.stderr.lower() or "refus" in res.stderr.lower()
    payload, _ = ok_cli(bp, *bp_args(bp, "status", mode="production"))
    assert payload["pid"] == prod_pid
    os.kill(prod_pid, 0)


def test_start_candidate_refuses_conflicting_tasker_process(bp):
    proc = spawn_direct(bp, ["-m", "src.tasker.service", "--once"])
    res = fail_cli(bp, *bp_args(bp, "start-candidate"))
    assert "conflict" in res.stderr.lower()
    assert not (bp.root / "run" / "tasker-candidate.pid").exists()
    os.kill(proc.pid, 0)


def test_start_candidate_refuses_scheduler_instance(bp):
    proc = spawn_direct(bp, ["-m", "src.tasker.service",
                             "--host", "127.0.0.1", "--port", "8000"],
                        mode="production")
    res = fail_cli(bp, *bp_args(bp, "start-candidate"))
    assert "scheduler" in res.stderr
    os.kill(proc.pid, 0)


def test_start_candidate_refuses_occupied_port(bp, occupied_port):
    res = fail_cli(bp, *bp_args(bp, "start-candidate"))
    assert "8000" in res.stderr
    assert not (bp.root / "run").exists()


def test_start_candidate_idempotent_with_occupied_port(bp):
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"))
    first = payload["pid"]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 8000))
    sock.listen(1)
    try:
        payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"))
        assert payload["pid"] == first
        assert len(spawn_entries(bp)) == 1
    finally:
        sock.close()
    payload, _ = ok_cli(bp, *bp_args(bp, "status"))
    assert payload["identity_exact"] is True


# ── 9. stop: SIGTERM, bounded SIGKILL escalation, stale, no-kill rules ─────


def test_stop_sigterm_success(bp):
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"))
    pid = payload["pid"]
    payload, _ = ok_cli(bp, *bp_args(bp, "stop"))
    assert payload["state"] == "inactive"
    assert payload["scheduler_mode"] == "disabled"
    assert payload["identity_exact"] is True
    assert payload["pid"] is None
    assert not (bp.root / "run" / "tasker-candidate.pid").exists()
    assert not (bp.root / "run" / "tasker-candidate-state.json").exists()
    events = [e["event"] for e in helper_lines(bp)]
    assert events.count("term") == 1
    wait_gone(pid)
    payload, _ = ok_cli(bp, *bp_args(bp, "status"))
    assert payload["state"] == "inactive"


def test_stop_idempotent_when_inactive(bp):
    payload, _ = ok_cli(bp, *bp_args(bp, "stop"))
    assert payload["state"] == "inactive"
    payload, _ = ok_cli(bp, *bp_args(bp, "stop"))
    assert payload["state"] == "inactive"


def test_stop_stale_cleanup(bp):
    write_pid_file(bp, "candidate", 999999)
    payload, _ = ok_cli(bp, *bp_args(bp, "stop"))
    assert payload["state"] == "inactive"
    assert not (bp.root / "run" / "tasker-candidate.pid").exists()


def test_stop_sigkill_escalation(bp):
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"),
                        env_extra={"PLBP_FAKE_IGNORE_TERM": "1"})
    pid = payload["pid"]
    payload, _ = ok_cli(bp, *bp_args(bp, "stop"),
                        env_extra={"PLBP_STOP_TIMEOUT": "1"})
    assert payload["state"] == "inactive"
    events = [e["event"] for e in helper_lines(bp)]
    assert events.count("term") == 1
    wait_gone(pid)


def test_stop_no_kill_when_identity_changes_before_escalation(bp):
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"),
                        env_extra={"PLBP_FAKE_TERM_SWITCH": "1"})
    pid = payload["pid"]
    res = run_cli(bp, *bp_args(bp, "stop"), env_extra={"PLBP_STOP_TIMEOUT": "1"})
    assert res.returncode != 0
    assert "identity" in res.stderr.lower()
    # SIGTERM was delivered but the PID was never SIGKILLed: process still alive.
    events = [e["event"] for e in helper_lines(bp)]
    assert events.count("term") == 1
    os.kill(pid, 0)
    assert (bp.root / "run" / "tasker-candidate.pid").exists()


def test_stop_inexact_identity_fails_without_signal(bp):
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"))
    pid = payload["pid"]
    rewrite_cmdline(bp, pid, [str(bp.helper), "-m", "src.tasker.service",
                              "--host", "127.0.0.1", "--port", "8000"])
    res = fail_cli(bp, *bp_args(bp, "stop"))
    assert "identity" in res.stderr.lower()
    assert all(e.get("event") != "term" for e in helper_lines(bp))
    os.kill(pid, 0)
    assert (bp.root / "run" / "tasker-candidate.pid").exists()


# ── 10. ensure (candidate) ─────────────────────────────────────────────────


def test_ensure_candidate_starts_and_is_idempotent(bp):
    payload, _ = ok_cli(bp, *bp_args(bp, "ensure"))
    assert payload["state"] == "active"
    assert payload["scheduler_mode"] == "disabled"
    assert payload["identity_exact"] is True
    assert payload["scheduler_instances"] == 0
    first = payload["pid"]
    payload, _ = ok_cli(bp, *bp_args(bp, "ensure"))
    assert payload["pid"] == first
    assert len(spawn_entries(bp)) == 1


def test_ensure_candidate_restarts_after_stop(bp):
    payload, _ = ok_cli(bp, *bp_args(bp, "ensure"))
    first = payload["pid"]
    ok_cli(bp, *bp_args(bp, "stop"))
    payload, _ = ok_cli(bp, *bp_args(bp, "ensure"))
    assert payload["state"] == "active"
    assert payload["pid"] != first
    assert len(spawn_entries(bp)) == 2


def test_ensure_candidate_refuses_inexact_active(bp):
    payload, _ = ok_cli(bp, *bp_args(bp, "start-candidate"))
    pid = payload["pid"]
    rewrite_cmdline(bp, pid, [str(bp.helper), "-m", "src.tasker.service"])
    res = fail_cli(bp, *bp_args(bp, "ensure"))
    assert "identity" in res.stderr.lower()
    assert len(spawn_entries(bp)) == 1


# ── 11. activate ───────────────────────────────────────────────────────────


def test_activate_refuses_active_candidate(bp):
    ok_cli(bp, *bp_args(bp, "start-candidate"))
    res = fail_cli(bp, *bp_args(
        bp, "activate", mode="production",
        former_authority_confirmed_stopped="proof"))
    assert "conflict" in res.stderr.lower()
    assert not (bp.root / "run" / "production-activation.json").exists()


def test_activate_refuses_second_scheduler(bp):
    ok_cli(bp, *bp_args(
        bp, "activate", mode="production",
        former_authority_confirmed_stopped="proof"))
    spawn_direct(bp, ["-m", "src.tasker.service",
                      "--host", "127.0.0.1", "--port", "8000"],
                 mode="production")
    res = fail_cli(bp, *bp_args(
        bp, "activate", mode="production",
        former_authority_confirmed_stopped="second-proof"))
    assert "scheduler" in res.stderr


def test_activate_starts_one_scheduler_and_writes_marker(bp):
    label = "sg01 tasker stopped 2026-09-03T12:00:00Z after drain"
    payload, _ = ok_cli(bp, *bp_args(
        bp, "activate", mode="production",
        former_authority_confirmed_stopped=label))
    assert payload["state"] == "active"
    assert payload["scheduler_mode"] == "enabled"
    assert payload["identity_exact"] is True
    assert payload["scheduler_instances"] == 1
    assert payload["pid"] > 0
    marker = bp.root / "run" / "production-activation.json"
    assert marker.exists()
    assert imode(marker) == 0o600
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data["schema"] == ACTIVATION_SCHEMA
    assert data["service_name"] == SERVICE
    assert data["mode"] == "production"
    assert data["app_dir"] == str(bp.app)
    assert data["web_root"] == str(bp.web)
    assert data["argv"] == [str(bp.helper), "-m", "src.tasker.service",
                            "--host", "127.0.0.1", "--port", "8000"]
    assert data["former_authority_sha256"] == hashlib.sha256(label.encode()).hexdigest()
    assert label not in marker.read_text(encoding="utf-8")
    assert imode(bp.root / "run" / "tasker-production.pid") == 0o600
    assert imode(bp.root / "run" / "tasker-production-state.json") == 0o600
    start = spawn_entries(bp)[-1]
    assert "--no-scheduler" not in start["argv"]
    assert start["env"].get("TASKER_DISABLE_SCHEDULER") is None
    assert start["env"]["PORTFOLIO_LAB_BOX_PERSIST_MODE"] == "production"


def test_activate_repeated_creates_no_second_process(bp):
    payload, _ = ok_cli(bp, *bp_args(
        bp, "activate", mode="production",
        former_authority_confirmed_stopped="proof"))
    first = payload["pid"]
    payload, _ = ok_cli(bp, *bp_args(
        bp, "activate", mode="production",
        former_authority_confirmed_stopped="proof-again"))
    assert payload["pid"] == first
    assert len(spawn_entries(bp)) == 1


def test_activate_failure_writes_no_marker(bp):
    res = run_cli(bp, *bp_args(
        bp, "activate", mode="production",
        former_authority_confirmed_stopped="proof"),
        env_extra={"PLBP_FAKE_SKIP_PROC": "1"})
    assert res.returncode != 0
    assert not (bp.root / "run" / "production-activation.json").exists()


# ── 12. ensure (production) ────────────────────────────────────────────────


def test_ensure_production_refuses_without_marker(bp):
    res = fail_cli(bp, *bp_args(bp, "ensure", mode="production"))
    assert "activation" in res.stderr
    assert len(spawn_entries(bp)) == 0


def test_ensure_production_refuses_mismatched_marker(bp):
    marker = bp.root / "run" / "production-activation.json"
    _write(marker, json.dumps({
        "schema": ACTIVATION_SCHEMA,
        "service_name": "someone-else",
        "mode": "production",
        "app_dir": str(bp.app),
        "web_root": str(bp.web),
        "argv": [str(bp.helper), "-m", "src.tasker.service",
                 "--host", "127.0.0.1", "--port", "8000"],
        "former_authority_sha256": "0" * 64,
    }))
    os.chmod(marker, 0o600)
    res = fail_cli(bp, *bp_args(bp, "ensure", mode="production"))
    assert "activation" in res.stderr


def test_ensure_production_refuses_bad_marker_file(bp):
    marker = bp.root / "run" / "production-activation.json"
    _write(marker, json.dumps({
        "schema": ACTIVATION_SCHEMA,
        "service_name": SERVICE,
        "mode": "production",
        "app_dir": str(bp.app),
        "web_root": str(bp.web),
        "argv": [str(bp.helper), "-m", "src.tasker.service",
                 "--host", "127.0.0.1", "--port", "8000"],
        "former_authority_sha256": "0" * 64,
    }))
    os.chmod(marker, 0o644)
    res = fail_cli(bp, *bp_args(bp, "ensure", mode="production"))
    assert "activation" in res.stderr


def test_ensure_production_restarts_only_with_valid_marker(bp):
    ok_cli(bp, *bp_args(bp, "activate", mode="production",
                        former_authority_confirmed_stopped="proof"))
    payload, _ = ok_cli(bp, *bp_args(bp, "stop", mode="production"))
    assert payload["state"] == "inactive"
    payload, _ = ok_cli(bp, *bp_args(bp, "ensure", mode="production"))
    assert payload["state"] == "active"
    assert payload["scheduler_mode"] == "enabled"
    assert payload["identity_exact"] is True
    assert payload["scheduler_instances"] == 1
    first = payload["pid"]
    payload, _ = ok_cli(bp, *bp_args(bp, "ensure", mode="production"))
    assert payload["pid"] == first
    assert len(spawn_entries(bp)) == 2


# ── 13. scheduler instance counting ────────────────────────────────────────


def test_scheduler_count_excludes_candidate_once_and_unrelated(bp):
    prod = spawn_direct(bp, ["-m", "src.tasker.service",
                             "--host", "127.0.0.1", "--port", "8000"],
                        mode="production")
    payload, _ = ok_cli(bp, *bp_args(bp, "status", mode="production"))
    assert payload["scheduler_instances"] == 1

    spawn_direct(bp, ["-m", "src.tasker.service", "--host", "127.0.0.1",
                      "--port", "8000", "--no-scheduler"])
    payload, _ = ok_cli(bp, *bp_args(bp, "status", mode="production"))
    assert payload["scheduler_instances"] == 1

    spawn_direct(bp, ["-m", "src.tasker.service", "--once"])
    payload, _ = ok_cli(bp, *bp_args(bp, "status", mode="production"))
    assert payload["scheduler_instances"] == 1

    spawn_direct(bp, ["--idle-unrelated"], knobs={"PLBP_FAKE_CMDLINE": "own"})
    payload, _ = ok_cli(bp, *bp_args(bp, "status", mode="production"))
    assert payload["scheduler_instances"] == 1

    payload, _ = ok_cli(bp, *bp_args(bp, "status", mode="production"))
    assert payload["state"] == "active"
    assert payload["identity_exact"] is False
    assert payload["pid"] == prod.pid


def test_scheduler_count_reports_two_instances(bp):
    spawn_direct(bp, ["-m", "src.tasker.service",
                      "--host", "127.0.0.1", "--port", "8000"],
                 mode="production")
    spawn_direct(bp, ["-m", "src.tasker.service",
                      "--host", "127.0.0.1", "--port", "8000"],
                 mode="production")
    payload, _ = ok_cli(bp, *bp_args(bp, "status", mode="production"))
    assert payload["state"] == "active"
    assert payload["identity_exact"] is False
    assert payload["scheduler_instances"] == 2


def test_scheduler_scan_fails_closed_on_unreadable_entry(bp):
    other = bp.proc / "424242"
    other.mkdir(parents=True)
    _write(other / "status", "Name:\tpython\nState:\tS (sleeping)\nPid:\t424242\n")
    _write(other / "cmdline", "-m\x00src.tasker.service\x00")
    os.chmod(other / "cmdline", 0o000)
    res = fail_cli(bp, *bp_args(bp, "status"))
    assert "424242" in res.stderr or "fail" in res.stderr.lower()


def test_scan_skips_other_user_entries_before_sensitive_reads(bp):
    other = bp.proc / "424242"
    other.mkdir(parents=True)
    _write(other / "status",
           "Name:\trootproc\nState:\tS (sleeping)\nPid:\t424242\nUid:\t0\t0\t0\t0\n")
    _write(other / "cmdline", "python\x00-m\x00src.tasker.service\x00")
    os.chmod(other / "cmdline", 0o000)  # would be fatal if probed
    payload, _ = ok_cli(bp, *bp_args(bp, "status"))
    assert payload["state"] == "inactive"
    assert payload["identity_exact"] is True
    assert payload["scheduler_instances"] == 0


def test_scan_skips_non_tasker_entries_without_env_cwd_reads(bp):
    uid = os.getuid()
    entry = bp.proc / "424243"
    entry.mkdir(parents=True)
    _write(entry / "status",
           f"Name:\tsleep\nState:\tS (sleeping)\nPid:\t424243\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
    _write(entry / "cmdline", "sleep\x00100\x00")
    _write(entry / "environ", "A=B\x00")
    os.chmod(entry / "environ", 0o000)  # would be fatal if probed
    payload, _ = ok_cli(bp, *bp_args(bp, "status"))
    assert payload["state"] == "inactive"
    assert payload["identity_exact"] is True
    assert payload["scheduler_instances"] == 0


def test_scan_skips_non_tasker_entries_with_missing_env_and_cwd(bp):
    uid = os.getuid()
    entry = bp.proc / "424244"
    entry.mkdir(parents=True)
    _write(entry / "status",
           f"Name:\ttop\nState:\tS (sleeping)\nPid:\t424244\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
    _write(entry / "cmdline", "top\x00-b\x00")
    payload, _ = ok_cli(bp, *bp_args(bp, "status"))
    assert payload["state"] == "inactive"
    assert payload["scheduler_instances"] == 0


def test_scan_fails_closed_on_same_user_tasker_ambiguity(bp):
    uid = os.getuid()
    entry = bp.proc / "424245"
    entry.mkdir(parents=True)
    _write(entry / "status",
           f"Name:\tpython\nState:\tS (sleeping)\nPid:\t424245\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
    _write(entry / "cmdline",
           "python\x00-m\x00src.tasker.service\x00--host\x00127.0.0.1\x00")
    _write(entry / "environ", "PORTFOLIO_LAB_PROJECT_DIR=/x\x00")
    os.chmod(entry / "environ", 0o000)
    res = fail_cli(bp, *bp_args(bp, "status"))
    assert "424245" in res.stderr or "fail" in res.stderr.lower()


# ── 14. ensure hook installer ──────────────────────────────────────────────


def ensure_script(bp: SimpleNamespace, name: str = "ensure.sh", body: str | None = None) -> Path:
    path = bp.tmp / name
    _write(path, body if body is not None else "#!/bin/sh\nset -e\nexit 0\n")
    os.chmod(path, 0o755)
    return path


def assert_one_block(content: str, mode: str) -> None:
    begin = "# BEGIN portfolio-lab managed\n"
    end = "# END portfolio-lab managed\n"
    starts = content.count(begin)
    ends = content.count(end)
    assert starts == 1, content
    assert ends == 1, content
    block = content[content.index(begin):]
    assert end in block
    assert block.count("portfolio-lab-box-persist ensure") == 1
    assert f"--mode {mode}" in block
    line = block[len(begin):block.index(end)]
    assert line.startswith(f"PATH={PROD_PATH} {CONTROLLER_PATH} ensure --mode {mode} --app-dir ")
    assert "--app-dir " in line and "--web-root " in line and "--service-name " in line


def test_install_hook_preserves_content_mode_and_is_idempotent(bp):
    original = "#!/bin/sh\nset -e\n# keep me\nexit 0\n"
    path = ensure_script(bp, body=original)
    payload, _ = ok_cli(bp, *bp_args(bp, "install-ensure-hook",
                                     ensure_script=str(path)))
    assert payload["schema"] == INSTALL_SCHEMA
    assert payload["changed"] is True
    assert payload["ensure_script"] == str(path)
    assert imode(path) == 0o755
    content = path.read_text(encoding="utf-8")
    assert content.startswith(original)
    assert_one_block(content, "candidate")
    assert "API_KEY" not in content and "=" + "secret" not in content
    payload, _ = ok_cli(bp, *bp_args(bp, "install-ensure-hook",
                                     ensure_script=str(path)))
    assert payload["changed"] is False
    assert path.read_bytes() == content.encode()
    assert_one_block(path.read_text(encoding="utf-8"), "candidate")


def test_install_hook_replaces_candidate_with_production(bp):
    path = ensure_script(bp)
    ok_cli(bp, *bp_args(bp, "install-ensure-hook", ensure_script=str(path)))
    payload, _ = ok_cli(bp, *bp_args(bp, "install-ensure-hook", mode="production",
                                     ensure_script=str(path)))
    assert payload["changed"] is True
    content = path.read_text(encoding="utf-8")
    assert_one_block(content, "production")
    assert "--mode candidate" not in content
    payload, _ = ok_cli(bp, *bp_args(bp, "install-ensure-hook", mode="production",
                                     ensure_script=str(path)))
    assert payload["changed"] is False
    assert path.read_bytes() == content.encode()


def test_install_hook_preserves_append_and_exec_only_mode(bp):
    path = ensure_script(bp)
    os.chmod(path, 0o555)
    payload, _ = ok_cli(bp, *bp_args(bp, "install-ensure-hook",
                                     ensure_script=str(path)))
    assert payload["changed"] is True
    assert imode(path) == 0o555
    assert path.read_text(encoding="utf-8").endswith("# END portfolio-lab managed\n")


@pytest.mark.parametrize(
    "label",
    ["missing", "symlink", "group_writable", "non_executable"],
)
def test_install_hook_rejects_unsafe_scripts(bp, label):
    path = bp.tmp / "ensure.sh"
    _write(path, "#!/bin/sh\nexit 0\n")
    os.chmod(path, 0o755)
    if label == "missing":
        path.unlink()
    elif label == "symlink":
        path.unlink()
        real = bp.tmp / "real.sh"
        _write(real, "#!/bin/sh\n")
        os.chmod(real, 0o755)
        os.symlink(real, path)
    elif label == "group_writable":
        os.chmod(path, 0o664)
    else:
        os.chmod(path, 0o644)
    res = fail_cli(bp, *bp_args(bp, "install-ensure-hook", ensure_script=str(path)))
    if label == "missing":
        assert "missing" in res.stderr
    elif label == "symlink":
        assert "symlink" in res.stderr
    elif label == "group_writable":
        assert "writable" in res.stderr
    else:
        assert "executable" in res.stderr


def test_install_hook_rejects_relative_path(bp):
    path = bp.tmp / "ensure.sh"
    _write(path, "#!/bin/sh\n")
    res = run_cli(bp, "install-ensure-hook", "--mode", "candidate",
                  "--app-dir", str(bp.app), "--web-root", str(bp.web),
                  "--service-name", SERVICE, "--ensure-script", "relative/ensure.sh")
    assert res.returncode != 0
    assert "absolute" in res.stderr


def test_install_hook_rejects_malformed_blocks(bp):
    path = bp.tmp / "ensure.sh"
    _write(path, "# END portfolio-lab managed\n")
    os.chmod(path, 0o755)
    res = fail_cli(bp, *bp_args(bp, "install-ensure-hook", ensure_script=str(path)))
    assert "BEGIN" in res.stderr or "END" in res.stderr
    _write(path, "# BEGIN portfolio-lab managed\nwhatever\n")
    os.chmod(path, 0o755)
    res = fail_cli(bp, *bp_args(bp, "install-ensure-hook", ensure_script=str(path)))
    assert "BEGIN" in res.stderr or "END" in res.stderr


def test_installed_block_passes_sh_syntax(bp):
    path = ensure_script(bp)
    ok_cli(bp, *bp_args(bp, "install-ensure-hook", ensure_script=str(path)))
    res = subprocess.run(["/bin/sh", "-n", str(path)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


# ── 15. no broad process matching / no shell execution (behavioral) ────────


def test_lifecycle_works_with_empty_path_no_external_commands(bp):
    empty = bp.tmp / "empty-bin"
    empty.mkdir()
    args = bp_args(bp, "start-candidate")
    payload, _ = ok_cli(bp, *args, env_extra={"PATH": str(empty)})
    assert payload["identity_exact"] is True
    pid = payload["pid"]
    ok_cli(bp, *bp_args(bp, "status"), env_extra={"PATH": str(empty)})
    # argv-array semantics: the fake cmdline is NUL-separated, not a shell line.
    cmdline = (bp.proc / str(pid) / "cmdline").read_bytes()
    assert b"\x00" in cmdline
    parts = cmdline.split(b"\x00")
    assert parts[-1] == b""
    assert parts[0].decode() == str(bp.helper)
    payload, _ = ok_cli(bp, *bp_args(bp, "stop"), env_extra={"PATH": str(empty)})
    assert payload["state"] == "inactive"


def test_paths_with_spaces_keep_exact_identity(tmp_path):
    ns = make_bp(tmp_path, root_name="plbp root dir")
    try:
        payload, _ = ok_cli(ns, *bp_args(ns, "start-candidate"))
        assert payload["identity_exact"] is True
        assert payload["state"] == "active"
        payload, _ = ok_cli(ns, *bp_args(ns, "status"))
        assert payload["identity_exact"] is True
        ok_cli(ns, *bp_args(ns, "stop"))
    finally:
        cleanup_bp(ns)


# ── Fix Round 2: ownership/liveness gap proofs ────────────────────────────


def test_probe_alive_permission_denied_never_counts_as_stopped(bp, monkeypatch):
    """Gap 1: probe_alive must not return False for PermissionError or generic
    OSError, and proc_stopped / inspect must not treat unprobeable live PIDs as
    stopped workload or discard ownership records.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import portfolio_lab_box_persist as bp_mod

    # Direct unit checks on probe_alive
    def raise_perm(pid, sig):
        raise PermissionError(1, "Operation not permitted")

    def raise_lookup(pid, sig):
        raise ProcessLookupError(3, "No such process")

    def raise_oserror(pid, sig):
        raise OSError(5, "Input/output error")

    # ProcessLookupError is the only safe stopped condition
    monkeypatch.setattr(os, "kill", raise_lookup)
    assert bp_mod.probe_alive(1234) is False
    assert bp_mod.proc_stopped(1234) is True

    # PermissionError and OSError must NOT count as dead/stopped
    monkeypatch.setattr(os, "kill", raise_perm)
    assert bp_mod.probe_alive(1234) is True
    assert bp_mod.proc_stopped(1234) is False

    monkeypatch.setattr(os, "kill", raise_oserror)
    assert bp_mod.probe_alive(1234) is True
    assert bp_mod.proc_stopped(1234) is False

    # Through production inspect boundary: unprobeable PID must retain its records
    # and report active with identity_exact=False, NEVER inactive/cleaned.
    pid = 99999
    pid_file = write_pid_file(bp, "candidate", pid)
    state_file = bp.root / "run" / "tasker-candidate-state.json"
    _write(state_file, json.dumps({"pid": pid, "mode": "candidate"}))
    os.chmod(state_file, 0o600)

    monkeypatch.setenv("PLBP_ROOT", str(bp.root))
    monkeypatch.setenv("PLBP_PROC_ROOT", str(bp.proc))
    monkeypatch.setattr(os, "kill", raise_perm)

    status_out, _, _ = bp_mod.inspect(
        "candidate", bp.app.resolve(), bp.web.resolve(), SERVICE, cleanup_stale=True
    )
    assert status_out["state"] == "active"
    assert status_out["identity_exact"] is False
    assert status_out["pid"] == pid
    assert pid_file.exists(), "stale cleanup must not remove unprobeable PID file"
    assert state_file.exists(), "stale cleanup must not remove unprobeable state file"

    # action_stop on unprobeable PID must refuse without signaling or removing files
    with pytest.raises(SystemExit) as excinfo:
        bp_mod.action_stop("candidate", bp.app.resolve(), bp.web.resolve(), SERVICE)
    assert excinfo.value.code != 0
    assert pid_file.exists(), "action_stop must not remove unprobeable PID file"
    assert state_file.exists(), "action_stop must not remove unprobeable state file"

    # Clean up artificial pid file before fixture teardown
    pid_file.unlink(missing_ok=True)
    state_file.unlink(missing_ok=True)


def test_spawn_rolls_back_exact_child_when_pid_write_fails(bp, monkeypatch):
    """Gap 2: spawn must bound-terminate the exact newly spawned child and clean
    PID/state records if the PID write fails.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import portfolio_lab_box_persist as bp_mod

    for k, v in bp.env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("PLBP_STOP_TIMEOUT", "2")
    monkeypatch.setenv("PLBP_KILL_TIMEOUT", "2")

    real_atomic_write = bp_mod.atomic_write
    spawned_pids: list[int] = []

    def failing_atomic_write(path: Path, data: bytes, mode: int) -> None:
        if path.name.endswith(".pid"):
            pid = int(data.strip())
            spawned_pids.append(pid)
            raise OSError("simulated disk full during pid write")
        return real_atomic_write(path, data, mode)

    monkeypatch.setattr(bp_mod, "atomic_write", failing_atomic_write)

    with pytest.raises(SystemExit) as excinfo:
        bp_mod.spawn("candidate", bp.app.resolve(), bp.web.resolve(), SERVICE, [], [])

    assert excinfo.value.code != 0
    assert len(spawned_pids) == 1
    child_pid = spawned_pids[0]

    # Verify the child was terminated boundedly
    wait_gone(child_pid, timeout=5.0)

    # Verify no pid or state files remain
    assert not (bp.root / "run" / "tasker-candidate.pid").exists()
    assert not (bp.root / "run" / "tasker-candidate-state.json").exists()

    # Verify original write failure is preserved in exception chain
    assert isinstance(excinfo.value.__cause__, OSError)
    assert "simulated disk full during pid write" in str(excinfo.value.__cause__)


def test_spawn_rolls_back_exact_child_when_state_write_fails(bp, monkeypatch):
    """Gap 2: spawn must bound-terminate the exact newly spawned child and clean
    PID/state records if the state write fails after the PID file was written.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import portfolio_lab_box_persist as bp_mod

    for k, v in bp.env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("PLBP_STOP_TIMEOUT", "2")
    monkeypatch.setenv("PLBP_KILL_TIMEOUT", "2")

    real_atomic_write = bp_mod.atomic_write
    spawned_pids: list[int] = []

    def failing_atomic_write(path: Path, data: bytes, mode: int) -> None:
        if path.name.endswith(".pid"):
            pid = int(data.strip())
            spawned_pids.append(pid)
            return real_atomic_write(path, data, mode)
        if path.name.endswith("-state.json"):
            raise OSError("simulated disk full during state write")
        return real_atomic_write(path, data, mode)

    monkeypatch.setattr(bp_mod, "atomic_write", failing_atomic_write)

    with pytest.raises(SystemExit) as excinfo:
        bp_mod.spawn("candidate", bp.app.resolve(), bp.web.resolve(), SERVICE, [], [])

    assert excinfo.value.code != 0
    assert len(spawned_pids) == 1
    child_pid = spawned_pids[0]

    # Child must be terminated boundedly
    wait_gone(child_pid, timeout=5.0)

    # Records must be cleaned
    assert not (bp.root / "run" / "tasker-candidate.pid").exists()
    assert not (bp.root / "run" / "tasker-candidate-state.json").exists()

    # Exception chain preserves original failure
    assert isinstance(excinfo.value.__cause__, OSError)
    assert "simulated disk full during state write" in str(excinfo.value.__cause__)


def test_spawn_rollback_does_not_signal_inexact_child(bp, monkeypatch):
    """Gap 2: spawn rollback must never signal an inexact child process."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import portfolio_lab_box_persist as bp_mod

    for k, v in bp.env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("PLBP_STOP_TIMEOUT", "1")
    monkeypatch.setenv("PLBP_KILL_TIMEOUT", "1")
    monkeypatch.setenv("PLBP_FAKE_EXE_TARGET", "/usr/bin/false")

    real_atomic_write = bp_mod.atomic_write
    spawned_pids: list[int] = []

    def failing_atomic_write(path: Path, data: bytes, mode: int) -> None:
        if path.name.endswith(".pid"):
            pid = int(data.strip())
            spawned_pids.append(pid)
            raise OSError("simulated disk full during pid write")
        return real_atomic_write(path, data, mode)

    monkeypatch.setattr(bp_mod, "atomic_write", failing_atomic_write)

    with pytest.raises(SystemExit) as excinfo:
        bp_mod.spawn("candidate", bp.app.resolve(), bp.web.resolve(), SERVICE, [], [])

    assert excinfo.value.code != 0
    assert len(spawned_pids) == 1
    child_pid = spawned_pids[0]

    # Inexact process must NOT have been signaled!
    assert all(e.get("event") != "term" for e in helper_lines(bp))
    # Still alive
    os.kill(child_pid, 0)
    try:
        os.kill(child_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_activate_rolls_back_production_scheduler_when_marker_write_fails(bp, monkeypatch):
    """Gap 3: action_activate must terminate the newly spawned production scheduler
    and clean PID/state records if marker creation fails.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import portfolio_lab_box_persist as bp_mod

    for k, v in bp.env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("PLBP_STOP_TIMEOUT", "2")
    monkeypatch.setenv("PLBP_KILL_TIMEOUT", "2")

    real_atomic_write = bp_mod.atomic_write

    def failing_marker_write(path: Path, data: bytes, mode: int) -> None:
        if path.name == "production-activation.json":
            raise OSError("simulated disk full during marker write")
        return real_atomic_write(path, data, mode)

    monkeypatch.setattr(bp_mod, "atomic_write", failing_marker_write)

    # Must raise or die when marker write fails
    with pytest.raises(SystemExit) as excinfo:
        bp_mod.action_activate(
            "production",
            bp.app.resolve(),
            bp.web.resolve(),
            SERVICE,
            "former-authority-confirmed-stopped-v1",
        )

    assert excinfo.value.code != 0

    # No marker must remain
    marker_path = bp.root / "run" / "production-activation.json"
    assert not marker_path.exists()

    # No PID or state files remain
    assert not (bp.root / "run" / "tasker-production.pid").exists()
    assert not (bp.root / "run" / "tasker-production-state.json").exists()

    # Check through inspect: no scheduler remains active!
    status_out, sched_pids, conflicts = bp_mod.inspect(
        "production", bp.app.resolve(), bp.web.resolve(), SERVICE
    )
    assert status_out["state"] == "inactive"
    assert status_out["scheduler_instances"] == 0
    assert sched_pids == []

    # Original write failure preserved in exception chain
    assert isinstance(excinfo.value.__cause__, OSError)
    assert "simulated disk full during marker write" in str(excinfo.value.__cause__)
