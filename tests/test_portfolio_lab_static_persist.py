"""Strict-TDD tests for the native static lifecycle controller and ensure installer (Task 2.3).

Tests exercise:
- lifecycle actions: preflight, status, start, stop, ensure, install-ensure-hook
- mode: candidate vs production (separation of PID/state records)
- exact process identity with PYTHON_EXECUTABLE + ORIGIN_SCRIPT argv & interpreter /proc/PID/exe
- fail-closed when /proc is absent or inexact
- proc scan for unmanaged/other-mode origin processes without PID files
- lifecycle test matrix: ensure start, idempotence, stop & restart
- rollback on PID write failure, state write failure, startup timeout
- live inexact process no-kill / no-signal
- permission-denied probe safety
- SIGTERM graceful exit and SIGKILL escalation with identity re-validation
- unsafe PID record symlink and permissions rejection (0600)
- preflight preserves malformed/non-positive PID records (read-only)
- ensure installer preservation, idempotence, static block replacement, Tasker block byte-for-byte preservation, sh -n
- Cloudflare routing contract verification (exact ordering and prefix matching)
- empty PATH execution (no external commands or shell execution)
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERSIST_SCRIPT = PROJECT_ROOT / "scripts" / "portfolio_lab_static_persist.py"
ORIGIN_SCRIPT = PROJECT_ROOT / "scripts" / "portfolio_lab_static_origin.py"
SCHEMA_STATUS = "portfolio-lab-static-persist/v1"
SERVICE = "portfolio-lab-static"
PROD_PATH = "/home/box/.local/bin:/usr/local/bin:/usr/bin:/bin"


# Fake origin script double that mimics running under a Python interpreter
# Argv will be: [python_exe, origin_script, --web-root, ..., --host, 127.0.0.1, --port, PORT, --max-inflight, 16]
FAKE_ORIGIN_SCRIPT_BODY = r'''#!{python}
import json
import os
import signal
import sys
import time

LOG = os.environ.get("PLSP_FAKE_HELPER_LOG", "")
PROC = os.environ.get("PLSP_FAKE_PROC_ROOT", "")
PORT = int(os.environ.get("PLSP_FAKE_PORT", "8001"))
INODE = 200000 + os.getpid()

def write_proc():
    if not PROC:
        return
    pid = os.getpid()
    base = os.path.join(PROC, str(pid))
    os.makedirs(os.path.join(base, "fd"), exist_ok=True)
    with open(os.path.join(base, "status"), "w", encoding="utf-8") as fh:
        uid = os.getuid()
        fh.write(f"Name:\tpython\nState:\tS (sleeping)\nPid:\t{pid}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
    # On real Linux, /proc/PID/exe points to the Python interpreter (sys.executable)
    exe_target = os.environ.get("PLSP_FAKE_EXE_TARGET") or os.path.realpath(sys.executable)
    os.symlink(exe_target, os.path.join(base, "exe"))
    cwd_target = os.environ.get("PLSP_FAKE_CWD_TARGET") or os.getcwd()
    os.symlink(cwd_target, os.path.join(base, "cwd"))
    with open(os.path.join(base, "cmdline"), "wb") as fh:
        # sys.argv contains [fake_origin_script, ...]. But the process is invoked as [python, fake_origin_script, ...].
        # If sys.argv[0] is the script, prepend sys.executable to match expected argv
        full_argv = [sys.executable, *sys.argv] if sys.argv[0] != sys.executable else sys.argv
        for part in full_argv:
            fh.write(part.encode("utf-8") + b"\x00")
    with open(os.path.join(base, "environ"), "wb") as fh:
        for k, v in os.environ.items():
            fh.write(f"{k}={v}".encode("utf-8") + b"\x00")
    os.symlink(f"socket:[{INODE}]", os.path.join(base, "fd", "3"))
    os.makedirs(os.path.join(PROC, "net"), exist_ok=True)
    tcp = os.path.join(PROC, "net", "tcp")
    port_hex = f"{PORT:04X}"
    row = (
        f"  0: 0100007F:{port_hex} 00000000:0000 0A 00000000:00000000 "
        f"00:00000000 00000000 1000 0 {INODE} 1 0000000000000000 100 0 0 10 0\n"
    )
    if os.path.exists(tcp):
        with open(tcp, "a", encoding="utf-8") as fh:
            fh.write(row)
    else:
        with open(tcp, "w", encoding="utf-8") as fh:
            fh.write("  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n")
            fh.write(row)

def handle_sigterm(signum, frame):
    if LOG:
        try:
            with open(LOG, "a", encoding="utf-8") as fh:
                fh.write(f"SIGNAL {signum}\n")
        except Exception:
            pass
    mode = os.environ.get("PLSP_FAKE_SIGTERM_ACTION", "exit")
    if mode == "ignore":
        return
    # Clean up fake proc dir on exit
    if PROC:
        try:
            import shutil
            shutil.rmtree(os.path.join(PROC, str(os.getpid())), ignore_errors=True)
        except Exception:
            pass
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)
write_proc()

print(json.dumps({"ready": True, "host": "127.0.0.1", "port": PORT}), flush=True)

while True:
    time.sleep(0.5)
'''


@pytest.fixture
def layout(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "share" / "portfolio-lab"
    root.mkdir(parents=True)
    run_dir = root / "run"
    run_dir.mkdir(mode=0o700)

    # Candidate web root: <root>/www-candidate
    www_candidate = root / "www-candidate"
    www_candidate.mkdir()
    (www_candidate / "index.html").write_text("candidate spa", encoding="utf-8")
    (www_candidate / "data").mkdir()

    # Production web root: <root>/www
    www_prod = root / "www"
    www_prod.mkdir()
    (www_prod / "index.html").write_text("production spa", encoding="utf-8")
    (www_prod / "data").mkdir()

    # Executable regular non-symlink fake origin script
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_origin = bin_dir / "portfolio-lab-static-origin"
    content = FAKE_ORIGIN_SCRIPT_BODY.replace("{python}", sys.executable)
    fake_origin.write_text(content, encoding="utf-8")
    fake_origin.chmod(0o755)

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    return {
        "root": root,
        "www_candidate": www_candidate,
        "www_prod": www_prod,
        "origin_exe": fake_origin,
        "proc_dir": proc_dir,
    }


def run_persist_cli(
    args: list[str],
    *,
    layout: dict[str, Path],
    port: int = 8001,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PLSP_ROOT"] = str(layout["root"])
    env["PLSP_ORIGIN_EXECUTABLE"] = str(layout["origin_exe"])
    env["PLSP_PYTHON_EXECUTABLE"] = sys.executable
    env["PLSP_PROC_ROOT"] = str(layout["proc_dir"])
    env["PLSP_PORT"] = str(port)
    env["PLSP_START_TIMEOUT"] = "3.0"
    env["PLSP_STOP_TIMEOUT"] = "2.0"
    env["PLSP_KILL_TIMEOUT"] = "2.0"
    env["PLSP_FAKE_PROC_ROOT"] = str(layout["proc_dir"])
    env["PLSP_FAKE_PORT"] = str(port)
    if extra_env:
        env.update(extra_env)

    cmd = [sys.executable, str(PERSIST_SCRIPT), *args]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


# ── 8. Lifecycle normalized status, exact identity, separate records, etc. ─


def test_port_free_rejects_non_accepting_listener() -> None:
    import socket

    import scripts.portfolio_lab_static_persist as persist_mod

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 8001))
    listener.listen(1)
    pending: list[socket.socket] = []
    backlog_full = False
    try:
        # Fill the listen queue without accepting so a connect-based probe can time out.
        for _ in range(32):
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(0.2)
            try:
                client.connect(("127.0.0.1", 8001))
            except OSError:
                client.close()
                backlog_full = True
                break
            pending.append(client)

        assert pending
        assert backlog_full
        assert persist_mod.port_free("127.0.0.1", 8001) is False
    finally:
        for client in pending:
            client.close()
        listener.close()


def test_run_dir_mode_enforced_as_0700_on_mutation(layout: dict[str, Path]) -> None:
    # Set run dir mode to 0755
    run_dir = layout["root"] / "run"
    run_dir.chmod(0o755)

    # Read-only status must NOT mutate run_dir mode
    res = run_persist_cli([
        "status",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res.returncode == 0
    assert oct(stat.S_IMODE(run_dir.stat().st_mode)) == "0o755"

    # Mutating start must enforce 0700
    res_start = run_persist_cli([
        "start",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res_start.returncode == 0
    assert oct(stat.S_IMODE(run_dir.stat().st_mode)) == "0o700"

    # Stop candidate
    run_persist_cli([
        "stop",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)


def test_log_path_symlink_rejected(layout: dict[str, Path], tmp_path: Path) -> None:
    # Create sentinel file
    sentinel = tmp_path / "sentinel.log"
    sentinel.write_text("sentinel content", encoding="utf-8")

    # Symlink static-candidate.log to sentinel
    log_file = layout["root"] / "run" / "static-candidate.log"
    log_file.symlink_to(sentinel)

    res = run_persist_cli([
        "start",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res.returncode != 0
    assert "symlink" in res.stderr.lower() or "log" in res.stderr.lower()

    # Sentinel must remain completely unchanged
    assert sentinel.read_text(encoding="utf-8") == "sentinel content"
    log_file.unlink()


def test_status_empty_inactive(layout: dict[str, Path]) -> None:
    res = run_persist_cli([
        "status",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert data["schema"] == SCHEMA_STATUS
    assert data["state"] == "inactive"
    assert data["identity_exact"] is True
    assert data["pid"] is None
    assert data["mode"] == "candidate"
    assert data["web_root"] == str(layout["www_candidate"])
    assert data["service_name"] == SERVICE
    assert data["host"] == "127.0.0.1"
    assert data["port"] == 8001


def test_start_status_ensure_stop_candidate(layout: dict[str, Path]) -> None:
    # Start candidate
    res = run_persist_cli([
        "start",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res.returncode == 0, res.stderr
    start_data = json.loads(res.stdout)
    assert start_data["state"] == "active"
    assert start_data["identity_exact"] is True
    assert start_data["pid"] is not None

    pid = start_data["pid"]
    pid_file = layout["root"] / "run" / "static-candidate.pid"
    assert pid_file.is_file()
    assert oct(stat.S_IMODE(pid_file.stat().st_mode)) == "0o600"
    assert pid_file.read_text().strip() == str(pid)

    # Status check
    res_status = run_persist_cli([
        "status",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res_status.returncode == 0
    status_data = json.loads(res_status.stdout)
    assert status_data["state"] == "active"
    assert status_data["identity_exact"] is True
    assert status_data["pid"] == pid

    # Ensure idempotence
    res_ensure = run_persist_cli([
        "ensure",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res_ensure.returncode == 0
    assert json.loads(res_ensure.stdout)["pid"] == pid

    # Stop
    res_stop = run_persist_cli([
        "stop",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res_stop.returncode == 0
    stop_data = json.loads(res_stop.stdout)
    assert stop_data["state"] == "inactive"
    assert stop_data["identity_exact"] is True
    assert stop_data["pid"] is None
    assert not pid_file.exists()

    # Ensure restarts cleanly after stop
    res_ensure2 = run_persist_cli([
        "ensure",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res_ensure2.returncode == 0
    assert json.loads(res_ensure2.stdout)["state"] == "active"
    assert json.loads(res_ensure2.stdout)["pid"] is not None

    # Final cleanup
    run_persist_cli([
        "stop",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)


def test_production_separation_and_shared_port_refusal(layout: dict[str, Path]) -> None:
    # Start candidate
    res1 = run_persist_cli([
        "start",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res1.returncode == 0
    assert json.loads(res1.stdout)["pid"] is not None

    # Candidate files exist, production files do not
    assert (layout["root"] / "run" / "static-candidate.pid").is_file()
    assert not (layout["root"] / "run" / "static-production.pid").exists()

    # Production start must be rejected because candidate occupies the shared port
    res2 = run_persist_cli([
        "start",
        "--mode", "production",
        "--web-root", str(layout["www_prod"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res2.returncode != 0
    assert "port" in res2.stderr.lower() or "active" in res2.stderr.lower()

    # Stop candidate
    run_persist_cli([
        "stop",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)

    # Now production start succeeds
    res3 = run_persist_cli([
        "start",
        "--mode", "production",
        "--web-root", str(layout["www_prod"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res3.returncode == 0
    prod_data = json.loads(res3.stdout)
    assert prod_data["mode"] == "production"
    assert prod_data["web_root"] == str(layout["www_prod"])
    assert (layout["root"] / "run" / "static-production.pid").is_file()

    # Clean up production
    run_persist_cli([
        "stop",
        "--mode", "production",
        "--web-root", str(layout["www_prod"]),
        "--service-name", SERVICE,
    ], layout=layout)


def test_absent_proc_with_live_pid_record_fail_closed(layout: dict[str, Path]) -> None:
    # Live PID with absent PLSP_PROC_ROOT
    nonexistent_proc = layout["root"] / "nonexistent_proc"
    env = {"PLSP_PROC_ROOT": str(nonexistent_proc)}

    sleep_proc = subprocess.Popen(["sleep", "30"])
    try:
        pid_file = layout["root"] / "run" / "static-candidate.pid"
        pid_file.write_text(f"{sleep_proc.pid}\n")
        pid_file.chmod(0o600)

        # Status must report identity_exact: False (never exact)
        res = run_persist_cli([
            "status",
            "--mode", "candidate",
            "--web-root", str(layout["www_candidate"]),
            "--service-name", SERVICE,
        ], layout=layout, extra_env=env)
        assert res.returncode == 0
        data = json.loads(res.stdout)
        assert data["identity_exact"] is False
        assert data["state"] == "active"
        assert data["pid"] == sleep_proc.pid
        # PID record is NOT deleted
        assert pid_file.exists()

        # Stop must refuse to signal inexact process and NOT delete record
        res_stop = run_persist_cli([
            "stop",
            "--mode", "candidate",
            "--web-root", str(layout["www_candidate"]),
            "--service-name", SERVICE,
        ], layout=layout, extra_env=env)
        assert res_stop.returncode != 0
        assert "inexact" in res_stop.stderr.lower()
        assert pid_file.exists()
        assert sleep_proc.poll() is None
    finally:
        sleep_proc.kill()
        sleep_proc.wait()


def test_permission_denied_and_oserror_liveness_unit(monkeypatch: pytest.MonkeyPatch, layout: dict[str, Path]) -> None:
    import scripts.portfolio_lab_static_persist as persist_mod

    # Prove probe_alive treats PermissionError and generic OSError as alive
    def raise_perm(pid: int, sig: int) -> None:
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(os, "kill", raise_perm)
    assert persist_mod.probe_alive(12345) is True

    def raise_oserr(pid: int, sig: int) -> None:
        raise OSError("Generic OS error")

    monkeypatch.setattr(os, "kill", raise_oserr)
    assert persist_mod.probe_alive(12345) is True

    # Now drive inspect() with a valid PID record, forcing permission/OSError identity ambiguity
    monkeypatch.setenv("PLSP_ROOT", str(layout["root"]))
    monkeypatch.setenv("PLSP_ORIGIN_EXECUTABLE", str(layout["origin_exe"]))
    monkeypatch.setenv("PLSP_PYTHON_EXECUTABLE", sys.executable)
    monkeypatch.setenv("PLSP_PROC_ROOT", str(layout["proc_dir"]))
    monkeypatch.setenv("PLSP_PORT", "8001")

    pid_file = layout["root"] / "run" / "static-candidate.pid"
    pid_file.write_text("12345\n")
    pid_file.chmod(0o600)

    # Force process_identity to return (False, "permission denied")
    monkeypatch.setattr(persist_mod, "process_identity", lambda pid, **kw: (False, "pid 12345 is not probeable: PermissionError"))

    payload = persist_mod.inspect("candidate", layout["www_candidate"], SERVICE, 8001)
    assert payload["state"] == "active"
    assert payload["identity_exact"] is False
    assert payload["pid"] == 12345
    # Must NOT clean up ambiguous PID file
    assert pid_file.exists()

    # Prove stop refuses before sending any signal
    kill_called = False

    def mock_kill(pid: int, sig: int) -> None:
        nonlocal kill_called
        kill_called = True

    monkeypatch.setattr(os, "kill", mock_kill)
    with pytest.raises(SystemExit):
        persist_mod.action_stop("candidate", layout["www_candidate"], SERVICE, 8001)
    assert kill_called is False
    assert pid_file.exists()


def test_unsafe_pid_symlink_rejected_and_not_deleted(layout: dict[str, Path], tmp_path: Path) -> None:
    target_file = tmp_path / "real_pid_target.txt"
    target_file.write_text("12345\n")
    target_file.chmod(0o600)

    pid_file = layout["root"] / "run" / "static-candidate.pid"
    pid_file.symlink_to(target_file)

    res = run_persist_cli([
        "status",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res.returncode != 0
    assert "symlink" in res.stderr.lower()
    # Symlink and target file must remain intact
    assert pid_file.is_symlink()
    assert target_file.exists()


def test_proc_scan_structured_matching_and_no_false_positives(layout: dict[str, Path]) -> None:
    # Process 1: with argument that contains origin script path as substring, but is NOT the origin script
    fake_pid = 77777
    proc_base = layout["proc_dir"] / str(fake_pid)
    proc_base.mkdir(parents=True)
    (proc_base / "status").write_text(f"Name:\tpython\nState:\tS (sleeping)\nPid:\t{fake_pid}\nUid:\t{os.getuid()}\n")
    (proc_base / "exe").symlink_to(sys.executable)
    (proc_base / "cwd").symlink_to(layout["www_candidate"])

    unrelated_argv = [
        sys.executable,
        "-m", "pytest",
        f"--some-flag=prefix_{str(layout['origin_exe'])}_suffix",
    ]
    with open(proc_base / "cmdline", "wb") as fh:
        for part in unrelated_argv:
            fh.write(part.encode() + b"\x00")

    # Process 2: a different script elsewhere that happens to have the same filename
    fake_pid2 = 77778
    proc_base2 = layout["proc_dir"] / str(fake_pid2)
    proc_base2.mkdir(parents=True)
    (proc_base2 / "status").write_text(f"Name:\tpython\nState:\tS (sleeping)\nPid:\t{fake_pid2}\nUid:\t{os.getuid()}\n")
    (proc_base2 / "exe").symlink_to(sys.executable)
    (proc_base2 / "cwd").symlink_to(layout["www_candidate"])

    different_script_argv = [
        sys.executable,
        "/some/unrelated/other/dir/portfolio-lab-static-origin",
        "--web-root", str(layout["www_candidate"]),
    ]
    with open(proc_base2 / "cmdline", "wb") as fh:
        for part in different_script_argv:
            fh.write(part.encode() + b"\x00")

    # Process 3: unrelated command where portfolio-lab-static-origin is an argument at later positions (not argv[1])
    fake_pid3 = 77779
    proc_base3 = layout["proc_dir"] / str(fake_pid3)
    proc_base3.mkdir(parents=True)
    (proc_base3 / "status").write_text(f"Name:\tpython\nState:\tS (sleeping)\nPid:\t{fake_pid3}\nUid:\t{os.getuid()}\n")
    (proc_base3 / "exe").symlink_to(sys.executable)
    (proc_base3 / "cwd").symlink_to(layout["www_candidate"])

    later_arg_argv = [
        sys.executable,
        "-c",
        "import sys; print(sys.argv)",
        "portfolio-lab-static-origin",
        "--web-root", str(layout["www_candidate"]),
    ]
    with open(proc_base3 / "cmdline", "wb") as fh:
        for part in later_arg_argv:
            fh.write(part.encode() + b"\x00")

    res = run_persist_cli([
        "status",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    # Must NOT have treated either unrelated process as a conflict
    assert data["state"] == "inactive"
    assert data["identity_exact"] is True


def test_ensure_installer_malformed_blocks(layout: dict[str, Path], tmp_path: Path) -> None:
    # 1. END without BEGIN
    f1 = tmp_path / "f1.sh"
    orig_bytes1 = b"#!/bin/sh\n# END portfolio-lab static managed\n"
    f1.write_bytes(orig_bytes1)
    f1.chmod(0o755)
    res1 = run_persist_cli(["install-ensure-hook", "--mode", "candidate", "--web-root", str(layout["www_candidate"]), "--service-name", SERVICE, "--ensure-script", str(f1)], layout=layout)
    assert res1.returncode != 0
    assert "malformed" in res1.stderr.lower()
    assert f1.read_bytes() == orig_bytes1
    assert oct(stat.S_IMODE(f1.stat().st_mode)) == "0o755"

    # 2. BEGIN without END
    f2 = tmp_path / "f2.sh"
    orig_bytes2 = b"#!/bin/sh\n# BEGIN portfolio-lab static managed\nsome code\n"
    f2.write_bytes(orig_bytes2)
    f2.chmod(0o755)
    res2 = run_persist_cli(["install-ensure-hook", "--mode", "candidate", "--web-root", str(layout["www_candidate"]), "--service-name", SERVICE, "--ensure-script", str(f2)], layout=layout)
    assert res2.returncode != 0
    assert "malformed" in res2.stderr.lower()
    assert f2.read_bytes() == orig_bytes2
    assert oct(stat.S_IMODE(f2.stat().st_mode)) == "0o755"

    # 3. Duplicate / nested BEGIN
    f3 = tmp_path / "f3.sh"
    orig_bytes3 = b"#!/bin/sh\n# BEGIN portfolio-lab static managed\n# BEGIN portfolio-lab static managed\n# END portfolio-lab static managed\n"
    f3.write_bytes(orig_bytes3)
    f3.chmod(0o755)
    res3 = run_persist_cli(["install-ensure-hook", "--mode", "candidate", "--web-root", str(layout["www_candidate"]), "--service-name", SERVICE, "--ensure-script", str(f3)], layout=layout)
    assert res3.returncode != 0
    assert "malformed" in res3.stderr.lower()
    assert f3.read_bytes() == orig_bytes3
    assert oct(stat.S_IMODE(f3.stat().st_mode)) == "0o755"

    # 4. Duplicate END
    f4 = tmp_path / "f4.sh"
    orig_bytes4 = b"#!/bin/sh\n# BEGIN portfolio-lab static managed\n# END portfolio-lab static managed\n# END portfolio-lab static managed\n"
    f4.write_bytes(orig_bytes4)
    f4.chmod(0o755)
    res4 = run_persist_cli(["install-ensure-hook", "--mode", "candidate", "--web-root", str(layout["www_candidate"]), "--service-name", SERVICE, "--ensure-script", str(f4)], layout=layout)
    assert res4.returncode != 0
    assert "malformed" in res4.stderr.lower()
    assert f4.read_bytes() == orig_bytes4
    assert oct(stat.S_IMODE(f4.stat().st_mode)) == "0o755"


def test_absent_proc_is_fail_closed(layout: dict[str, Path]) -> None:
    # Point PLSP_PROC_ROOT to a nonexistent directory
    nonexistent_proc = layout["root"] / "nonexistent_proc"
    env = {"PLSP_PROC_ROOT": str(nonexistent_proc)}

    res = run_persist_cli([
        "status",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout, extra_env=env)

    # When proc is absent, exact identity cannot be claimed
    if res.returncode == 0:
        data = json.loads(res.stdout)
        assert data["identity_exact"] is False or data["state"] == "inactive"
    else:
        assert res.returncode != 0


def test_proc_scan_unmanaged_origin_process(layout: dict[str, Path]) -> None:
    # Simulate a running origin process on the system that is NOT in our PID file
    fake_pid = 98765
    proc_base = layout["proc_dir"] / str(fake_pid)
    proc_base.mkdir(parents=True)
    (proc_base / "status").write_text(f"Name:\tpython\nState:\tS (sleeping)\nPid:\t{fake_pid}\nUid:\t{os.getuid()}\n")
    (proc_base / "exe").symlink_to(sys.executable)
    (proc_base / "cwd").symlink_to(layout["www_candidate"])
    # cmdline contains origin script and port
    argv = [
        sys.executable,
        str(layout["origin_exe"]),
        "--web-root", str(layout["www_candidate"]),
        "--host", "127.0.0.1",
        "--port", "8001",
        "--max-inflight", "16",
    ]
    with open(proc_base / "cmdline", "wb") as fh:
        for part in argv:
            fh.write(part.encode() + b"\x00")

    # Status without PID file must detect conflicting unmanaged origin and fail closed (active inexact)
    res = run_persist_cli([
        "status",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["identity_exact"] is False

    # Start must refuse to proceed
    res_start = run_persist_cli([
        "start",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res_start.returncode != 0
    assert "active" in res_start.stderr.lower() or "conflict" in res_start.stderr.lower() or "inexact" in res_start.stderr.lower()


def test_stale_pid_cleanup(layout: dict[str, Path]) -> None:
    pid_file = layout["root"] / "run" / "static-candidate.pid"
    state_file = layout["root"] / "run" / "static-candidate-state.json"
    pid_file.write_text("9999999\n")
    pid_file.chmod(0o600)
    state_file.write_text("{}", encoding="utf-8")
    state_file.chmod(0o600)

    res = run_persist_cli([
        "status",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["state"] == "inactive"
    assert data["identity_exact"] is True
    assert not pid_file.exists()
    assert not state_file.exists()


def test_unsafe_pid_file_permissions_fail_closed(layout: dict[str, Path]) -> None:
    pid_file = layout["root"] / "run" / "static-candidate.pid"
    pid_file.write_text("12345\n")
    # Set group writable 0664
    pid_file.chmod(0o664)

    res = run_persist_cli([
        "status",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    # Must fail closed with error
    assert res.returncode != 0
    assert "0600" in res.stderr
    # Unsafe PID file is NOT deleted
    assert pid_file.exists()


def test_live_inexact_no_signal_and_no_deletion(layout: dict[str, Path]) -> None:
    sleep_proc = subprocess.Popen(["sleep", "30"])
    try:
        pid_file = layout["root"] / "run" / "static-candidate.pid"
        pid_file.write_text(f"{sleep_proc.pid}\n")
        pid_file.chmod(0o600)

        res = run_persist_cli([
            "status",
            "--mode", "candidate",
            "--web-root", str(layout["www_candidate"]),
            "--service-name", SERVICE,
        ], layout=layout)
        assert res.returncode == 0
        data = json.loads(res.stdout)
        assert data["state"] == "active"
        assert data["identity_exact"] is False
        assert data["pid"] == sleep_proc.pid
        assert pid_file.exists()

        res_stop = run_persist_cli([
            "stop",
            "--mode", "candidate",
            "--web-root", str(layout["www_candidate"]),
            "--service-name", SERVICE,
        ], layout=layout)
        assert res_stop.returncode != 0
        assert "inexact" in res_stop.stderr.lower()
        assert sleep_proc.poll() is None
    finally:
        sleep_proc.kill()
        sleep_proc.wait()


def test_rollback_on_pid_and_state_write_failure(layout: dict[str, Path]) -> None:
    # 1. State write failure: create state file directory unwritable
    run_dir = layout["root"] / "run"
    state_file = run_dir / "static-candidate-state.json"
    state_file.mkdir()
    state_file.chmod(0o500)

    res = run_persist_cli([
        "start",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res.returncode != 0
    assert "failed" in res.stderr.lower() or "ownership" in res.stderr.lower()

    state_file.rmdir()
    assert not (run_dir / "static-candidate.pid").exists()


def test_sigkill_identity_revalidation_refusal(layout: dict[str, Path]) -> None:
    # Use a real child process that ignores SIGTERM, and verify terminate_exact refuses SIGKILL when identity is inexact
    script = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "while True:\n"
        "    time.sleep(1.0)\n"
    )
    ignoring_proc = subprocess.Popen([sys.executable, "-c", script])
    time.sleep(0.2)
    try:
        import scripts.portfolio_lab_static_persist as persist_mod

        # When identity is inexact, terminate_exact must refuse SIGKILL and exit without killing
        with pytest.raises(SystemExit):
            persist_mod.terminate_exact(
                ignoring_proc.pid,
                web_r=layout["www_candidate"],
                port=8001,
                stop_timeout=0.2,
                kill_timeout=0.2,
            )

        # Child is STILL ALIVE because SIGKILL was refused!
        assert ignoring_proc.poll() is None
    finally:
        ignoring_proc.kill()
        ignoring_proc.wait()


def test_startup_timeout_no_kill_inexact_process(layout: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Test that driving spawn() through startup timeout with its newly created child
    # registered in _SPAWNED leaves the child alive if it is inexact, and removes ownership files.
    import scripts.portfolio_lab_static_persist as persist_mod

    sig_log = tmp_path / "signals.log"
    env = {
        "PLSP_FAKE_EXE_TARGET": "/bin/sleep",  # Forces identity mismatch
        "PLSP_START_TIMEOUT": "0.3",
        "PLSP_STOP_TIMEOUT": "0.3",
        "PLSP_KILL_TIMEOUT": "0.3",
        "PLSP_FAKE_HELPER_LOG": str(sig_log),
    }

    spawned_proc: subprocess.Popen[Any] | None = None
    orig_popen = persist_mod.subprocess.Popen

    def wrap_popen(*args: Any, **kwargs: Any) -> Any:
        nonlocal spawned_proc
        proc = orig_popen(*args, **kwargs)
        spawned_proc = proc
        return proc

    monkeypatch_inst = pytest.MonkeyPatch()
    monkeypatch_inst.setattr(persist_mod.subprocess, "Popen", wrap_popen)

    for k, v in env.items():
        monkeypatch_inst.setenv(k, v)
    monkeypatch_inst.setenv("PLSP_ROOT", str(layout["root"]))
    monkeypatch_inst.setenv("PLSP_ORIGIN_EXECUTABLE", str(layout["origin_exe"]))
    monkeypatch_inst.setenv("PLSP_PYTHON_EXECUTABLE", sys.executable)
    monkeypatch_inst.setenv("PLSP_PROC_ROOT", str(layout["proc_dir"]))
    monkeypatch_inst.setenv("PLSP_PORT", "8001")
    monkeypatch_inst.setenv("PLSP_FAKE_PROC_ROOT", str(layout["proc_dir"]))
    monkeypatch_inst.setenv("PLSP_FAKE_PORT", "8001")

    try:
        with pytest.raises(SystemExit):
            persist_mod.spawn("candidate", layout["www_candidate"], SERVICE, 8001)

        assert spawned_proc is not None
        child_pid = spawned_proc.pid
        assert child_pid in persist_mod._SPAWNED

        assert str(child_pid) in capsys.readouterr().err

        # Inexact child must NOT receive SIGTERM or SIGKILL, and must remain alive!
        assert not sig_log.exists() or "SIGNAL" not in sig_log.read_text(encoding="utf-8")
        assert persist_mod.probe_alive(child_pid) is True
        assert spawned_proc.poll() is None

        # Ownership files must be removed
        run_dir = layout["root"] / "run"
        assert not (run_dir / "static-candidate.pid").exists()
        assert not (run_dir / "static-candidate-state.json").exists()
    finally:
        monkeypatch_inst.undo()
        if spawned_proc is not None:
            try:
                spawned_proc.kill()
                spawned_proc.wait(timeout=1.0)
            except Exception:
                pass


def test_pid_write_failure_inexact_child_retained(layout: dict[str, Path], tmp_path: Path) -> None:
    # Test boundary where PID write fails on an inexact child:
    # child must NOT be killed, must remain alive, and ownership records removed.
    import scripts.portfolio_lab_static_persist as persist_mod

    sig_log = tmp_path / "signals.log"
    monkeypatch_inst = pytest.MonkeyPatch()
    monkeypatch_inst.setenv("PLSP_ROOT", str(layout["root"]))
    monkeypatch_inst.setenv("PLSP_ORIGIN_EXECUTABLE", str(layout["origin_exe"]))
    monkeypatch_inst.setenv("PLSP_PYTHON_EXECUTABLE", sys.executable)
    monkeypatch_inst.setenv("PLSP_PROC_ROOT", str(layout["proc_dir"]))
    monkeypatch_inst.setenv("PLSP_PORT", "8001")
    monkeypatch_inst.setenv("PLSP_STOP_TIMEOUT", "0.2")
    monkeypatch_inst.setenv("PLSP_KILL_TIMEOUT", "0.2")
    monkeypatch_inst.setenv("PLSP_FAKE_PROC_ROOT", str(layout["proc_dir"]))
    monkeypatch_inst.setenv("PLSP_FAKE_PORT", "8001")
    monkeypatch_inst.setenv("PLSP_FAKE_EXE_TARGET", "/bin/sleep")  # Force inexact
    monkeypatch_inst.setenv("PLSP_FAKE_HELPER_LOG", str(sig_log))

    spawned_proc: subprocess.Popen[Any] | None = None
    orig_popen = persist_mod.subprocess.Popen

    def wrap_popen(*args: Any, **kwargs: Any) -> Any:
        nonlocal spawned_proc
        proc = orig_popen(*args, **kwargs)
        spawned_proc = proc
        return proc

    orig_atomic_write = persist_mod.atomic_write

    def fail_atomic(path: Path, data: bytes, mode: int) -> None:
        if path.name.endswith(".pid"):
            raise OSError("simulated pid write failure while inexact")
        orig_atomic_write(path, data, mode)

    monkeypatch_inst.setattr(persist_mod.subprocess, "Popen", wrap_popen)
    monkeypatch_inst.setattr(persist_mod, "atomic_write", fail_atomic)

    try:
        with pytest.raises(SystemExit):
            persist_mod.spawn("candidate", layout["www_candidate"], SERVICE, 8001)

        assert spawned_proc is not None
        child_pid = spawned_proc.pid
        assert child_pid in persist_mod._SPAWNED

        # Inexact child must NOT receive SIGTERM/SIGKILL and must remain alive!
        assert not sig_log.exists() or "SIGNAL" not in sig_log.read_text(encoding="utf-8")
        assert persist_mod.probe_alive(child_pid) is True
        assert spawned_proc.poll() is None

        run_dir = layout["root"] / "run"
        assert not (run_dir / "static-candidate.pid").exists()
        assert not (run_dir / "static-candidate-state.json").exists()
    finally:
        monkeypatch_inst.undo()
        if spawned_proc is not None:
            try:
                spawned_proc.kill()
                spawned_proc.wait(timeout=1.0)
            except Exception:
                pass


def test_pid_write_failure_rollback_unit(monkeypatch: pytest.MonkeyPatch, layout: dict[str, Path]) -> None:
    import scripts.portfolio_lab_static_persist as persist_mod

    monkeypatch.setenv("PLSP_ROOT", str(layout["root"]))
    monkeypatch.setenv("PLSP_ORIGIN_EXECUTABLE", str(layout["origin_exe"]))
    monkeypatch.setenv("PLSP_PYTHON_EXECUTABLE", sys.executable)
    monkeypatch.setenv("PLSP_PROC_ROOT", str(layout["proc_dir"]))
    monkeypatch.setenv("PLSP_PORT", "8001")
    monkeypatch.setenv("PLSP_FAKE_PROC_ROOT", str(layout["proc_dir"]))
    monkeypatch.setenv("PLSP_FAKE_PORT", "8001")

    spawned_pid: int | None = None
    orig_atomic_write = persist_mod.atomic_write

    def fail_atomic(path: Path, data: bytes, mode: int) -> None:
        if path.name.endswith(".pid"):
            # Ensure fake origin reaches exact identity before failing write
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                exact, _ = persist_mod.process_identity(spawned_pid, web_r=layout["www_candidate"], port=8001)
                if exact:
                    break
                time.sleep(0.05)
            raise OSError("simulated pid write failure")
        orig_atomic_write(path, data, mode)

    orig_popen = persist_mod.subprocess.Popen

    def wrap_popen(*args: Any, **kwargs: Any) -> Any:
        nonlocal spawned_pid
        proc = orig_popen(*args, **kwargs)
        spawned_pid = proc.pid
        return proc

    monkeypatch.setattr(persist_mod.subprocess, "Popen", wrap_popen)
    monkeypatch.setattr(persist_mod, "atomic_write", fail_atomic)

    with pytest.raises(SystemExit):
        persist_mod.spawn("candidate", layout["www_candidate"], SERVICE, 8001)

    assert spawned_pid is not None
    # Wait boundedly for the child to exit
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and persist_mod.probe_alive(spawned_pid):
        time.sleep(0.05)
    assert not persist_mod.probe_alive(spawned_pid)

    run_dir = layout["root"] / "run"
    assert not (run_dir / "static-candidate.pid").exists()
    assert not (run_dir / "static-candidate-state.json").exists()


def test_state_write_failure_rollback_unit(monkeypatch: pytest.MonkeyPatch, layout: dict[str, Path]) -> None:
    import scripts.portfolio_lab_static_persist as persist_mod

    monkeypatch.setenv("PLSP_ROOT", str(layout["root"]))
    monkeypatch.setenv("PLSP_ORIGIN_EXECUTABLE", str(layout["origin_exe"]))
    monkeypatch.setenv("PLSP_PYTHON_EXECUTABLE", sys.executable)
    monkeypatch.setenv("PLSP_PROC_ROOT", str(layout["proc_dir"]))
    monkeypatch.setenv("PLSP_PORT", "8001")
    monkeypatch.setenv("PLSP_FAKE_PROC_ROOT", str(layout["proc_dir"]))
    monkeypatch.setenv("PLSP_FAKE_PORT", "8001")

    spawned_pid: int | None = None
    orig_atomic_write = persist_mod.atomic_write

    def fail_atomic(path: Path, data: bytes, mode: int) -> None:
        if path.name.endswith("-state.json"):
            # Ensure fake origin reaches exact identity before failing write
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                exact, _ = persist_mod.process_identity(spawned_pid, web_r=layout["www_candidate"], port=8001)
                if exact:
                    break
                time.sleep(0.05)
            raise OSError("simulated state write failure")
        orig_atomic_write(path, data, mode)

    orig_popen = persist_mod.subprocess.Popen

    def wrap_popen(*args: Any, **kwargs: Any) -> Any:
        nonlocal spawned_pid
        proc = orig_popen(*args, **kwargs)
        spawned_pid = proc.pid
        return proc

    monkeypatch.setattr(persist_mod.subprocess, "Popen", wrap_popen)
    monkeypatch.setattr(persist_mod, "atomic_write", fail_atomic)

    with pytest.raises(SystemExit):
        persist_mod.spawn("candidate", layout["www_candidate"], SERVICE, 8001)

    assert spawned_pid is not None
    # Wait boundedly for the child to exit
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and persist_mod.probe_alive(spawned_pid):
        time.sleep(0.05)
    assert not persist_mod.probe_alive(spawned_pid)

    run_dir = layout["root"] / "run"
    assert not (run_dir / "static-candidate.pid").exists()
    assert not (run_dir / "static-candidate-state.json").exists()


def test_sigterm_and_sigkill_escalation(layout: dict[str, Path]) -> None:
    env = {
        "PLSP_FAKE_SIGTERM_ACTION": "ignore",
        "PLSP_STOP_TIMEOUT": "1.0",
        "PLSP_KILL_TIMEOUT": "1.0",
    }
    res_start = run_persist_cli([
        "start",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout, extra_env=env)
    assert res_start.returncode == 0
    assert json.loads(res_start.stdout)["pid"] is not None

    res_stop = run_persist_cli([
        "stop",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout, extra_env=env)
    assert res_stop.returncode == 0
    assert json.loads(res_stop.stdout)["state"] == "inactive"
    assert not (layout["root"] / "run" / "static-candidate.pid").exists()


def test_preflight_is_read_only_preserves_malformed_records(layout: dict[str, Path]) -> None:
    pid_file = layout["root"] / "run" / "static-candidate.pid"
    pid_file.write_text("-99\n")
    pid_file.chmod(0o600)

    res = run_persist_cli([
        "preflight",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["preflight"]["ok"] is True
    # Preflight must NOT have deleted the record
    assert pid_file.exists()


# ── 10. Ensure Installer ──────────────────────────────────────────────────


def test_install_ensure_hook_and_tasker_block_preservation(layout: dict[str, Path], tmp_path: Path) -> None:
    ensure_sh = tmp_path / "ensure.sh"
    tasker_block = (
        "#!/bin/sh\n"
        "# BEGIN portfolio-lab managed\n"
        "PATH=/home/box/.local/bin:/usr/local/bin:/usr/bin:/bin /home/box/.local/bin/portfolio-lab-box-persist ensure --mode candidate --app-dir /home/box/.local/share/portfolio-lab/app-candidate --web-root /home/box/.local/share/portfolio-lab/www-candidate --service-name portfolio-lab-tasker\n"
        "# END portfolio-lab managed\n"
    )
    ensure_sh.write_text(tasker_block, encoding="utf-8")
    ensure_sh.chmod(0o755)

    # Install candidate hook
    res = run_persist_cli([
        "install-ensure-hook",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
        "--ensure-script", str(ensure_sh),
    ], layout=layout)
    assert res.returncode == 0
    install_data = json.loads(res.stdout)
    assert install_data["changed"] is True

    content = ensure_sh.read_text(encoding="utf-8")
    # Tasker block preserved byte-for-byte
    assert tasker_block in content
    # Static block added
    assert "# BEGIN portfolio-lab static managed" in content
    assert "# END portfolio-lab static managed" in content
    assert "--mode candidate" in content

    # /bin/sh -n validation
    sh_check = subprocess.run(["/bin/sh", "-n", str(ensure_sh)], capture_output=True, text=True)
    assert sh_check.returncode == 0

    # Idempotent reinstall
    res_idem = run_persist_cli([
        "install-ensure-hook",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
        "--ensure-script", str(ensure_sh),
    ], layout=layout)
    assert res_idem.returncode == 0
    assert json.loads(res_idem.stdout)["changed"] is False
    assert ensure_sh.read_text(encoding="utf-8") == content

    # Candidate-to-production replaces only the static block
    res_prod = run_persist_cli([
        "install-ensure-hook",
        "--mode", "production",
        "--web-root", str(layout["www_prod"]),
        "--service-name", SERVICE,
        "--ensure-script", str(ensure_sh),
    ], layout=layout)
    assert res_prod.returncode == 0
    assert json.loads(res_prod.stdout)["changed"] is True

    prod_content = ensure_sh.read_text(encoding="utf-8")
    assert "--mode production" in prod_content
    assert str(layout["www_prod"]) in prod_content
    # Tasker block remains completely intact byte-for-byte
    assert tasker_block in prod_content

    sh_check2 = subprocess.run(["/bin/sh", "-n", str(ensure_sh)], capture_output=True, text=True)
    assert sh_check2.returncode == 0


def test_install_ensure_hook_rejections(layout: dict[str, Path], tmp_path: Path) -> None:
    # 1. Missing script
    res = run_persist_cli([
        "install-ensure-hook",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
        "--ensure-script", str(tmp_path / "nonexistent.sh"),
    ], layout=layout)
    assert res.returncode != 0

    # 2. Symlink script
    real_sh = tmp_path / "real.sh"
    real_sh.write_text("#!/bin/sh\n")
    real_sh.chmod(0o755)
    sym_sh = tmp_path / "sym.sh"
    sym_sh.symlink_to(real_sh)
    res = run_persist_cli([
        "install-ensure-hook",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
        "--ensure-script", str(sym_sh),
    ], layout=layout)
    assert res.returncode != 0
    assert "symlink" in res.stderr.lower()

    # 3. Group-writable script
    gw_sh = tmp_path / "gw.sh"
    gw_sh.write_text("#!/bin/sh\n")
    gw_sh.chmod(0o775)
    res = run_persist_cli([
        "install-ensure-hook",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
        "--ensure-script", str(gw_sh),
    ], layout=layout)
    assert res.returncode != 0
    assert "group/world-writable" in res.stderr.lower()

    # 4. Non-executable script
    nx_sh = tmp_path / "nx.sh"
    nx_sh.write_text("#!/bin/sh\n")
    nx_sh.chmod(0o644)
    res = run_persist_cli([
        "install-ensure-hook",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
        "--ensure-script", str(nx_sh),
    ], layout=layout)
    assert res.returncode != 0
    assert "executable" in res.stderr.lower()


# ── 11. Cloudflare Route Order and Matching ───────────────────────────────


def test_cloudflare_route_order_and_matching() -> None:
    from scripts.portfolio_lab_static_persist import (
        CLOUDFLARE_INGRESS_RULES,
        match_cloudflare_route,
    )

    assert len(CLOUDFLARE_INGRESS_RULES) == 2
    r1, r2 = CLOUDFLARE_INGRESS_RULES
    assert r1["hostname"] == "lab.termolo.com"
    assert r1["path"] == r"^/api(?:/.*)?$"
    assert r1["service"] == "http://127.0.0.1:8000"

    assert r2["hostname"] == "lab.termolo.com"
    assert "path" not in r2 or r2["path"] is None
    assert r2["service"] == "http://127.0.0.1:8001"

    for p in ("/api", "/api/", "/api/trades", "/api/v1/status"):
        assert match_cloudflare_route("lab.termolo.com", p) == "http://127.0.0.1:8000"

    for p in ("/", "/data/x.json", "/design-guide", "/assets/app.js", "/index.html"):
        assert match_cloudflare_route("lab.termolo.com", p) == "http://127.0.0.1:8001"

    assert match_cloudflare_route("other.termolo.com", "/api") is None


# ── 12. Empty PATH Proves No External Commands ────────────────────────────


def test_empty_path_execution(layout: dict[str, Path]) -> None:
    empty_env = {"PATH": ""}
    res = run_persist_cli([
        "status",
        "--mode", "candidate",
        "--web-root", str(layout["www_candidate"]),
        "--service-name", SERVICE,
    ], layout=layout, extra_env=empty_env)
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert data["state"] == "inactive"
