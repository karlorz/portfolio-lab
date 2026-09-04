"""Strict-TDD tests for the cursor-box daily operational evidence collector.

Every test drives the shipped CLI (``scripts/portfolio_lab_daily_evidence.py``)
as ``sys.executable SCRIPT ...`` and the shipped wrapper
(``scripts/cron/portfolio-lab-cursor-box-daily-evidence.sh``) as
``/bin/sh WRAPPER`` against isolated temp roots/outputs with fake
controllers, a fake flock (the dev host has no flock(1); production uses the
cursor-box BusyBox flock), and a loopback HTTP server. No network, SSH,
Docker, or service mutation.

Coverage: pass/warning/fail aggregation and exit codes; redaction (archive
log paths/keys/credentials, API command/log-path fields, former-authority
source path); evidence permissions, per-file size bound and deterministic
same-day replacement; invalid inputs (relative paths, unsafe output
placement, bad URLs/timeout/freshness/now); timeouts and malformed
controller/HTTP payloads; former-authority pass/warn/fail states; the
wrapper once-per-UTC-day, stamp-on-success-only, lock-held-skip, and
credentials-free contract. Pure decision functions are unit-tested by
importing the shipped script as a module.

Production defaults are asserted as module constants so accidental drift of
the wrapper/CLI production wiring is caught here.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "scripts" / "portfolio_lab_daily_evidence.py"
WRAPPER = PROJECT_ROOT / "scripts" / "cron" / "portfolio-lab-cursor-box-daily-evidence.sh"

SCHEMA = "portfolio-lab-daily-evidence/v1"
BOX_PERSIST_SCHEMA = "portfolio-lab-box-persist/v1"
STATIC_PERSIST_SCHEMA = "portfolio-lab-static-persist/v1"
PROOF_SCHEMA = "portfolio-lab-former-authority-proof/v1"
DEFAULT_EXPECTED_HOST = "sg01"

DEFAULT_ROOT = Path("/home/box/.local/share/portfolio-lab")
DEFAULT_OUTPUT_REL = "evidence"
DEFAULT_TASKER_CONTROLLER = "/home/box/.local/bin/portfolio-lab-box-persist"
DEFAULT_STATIC_CONTROLLER = "/home/box/.local/bin/portfolio-lab-static-persist"
DEFAULT_API_URL = "http://127.0.0.1:8000/api/tasker/status"
DEFAULT_STATIC_URL = "http://127.0.0.1:8001/"
DEFAULT_TIMEOUT = 10.0
DEFAULT_FRESHNESS_MAX_AGE = 21600
WRAPPER_PYTHON_DEFAULT = "/home/box/.local/bin/python3"
WRAPPER_SCRIPT_REL = "app/scripts/portfolio_lab_daily_evidence.py"
WRAPPER_LOCK_REL = "run/portfolio-lab-daily-evidence.lock"
WRAPPER_STAMP_REL = "run/portfolio-lab-daily-evidence-last-utc-day"
LOCK_NAME = WRAPPER_LOCK_REL.split("/")[1]
STAMP_NAME = WRAPPER_STAMP_REL.split("/")[1]

TASKER_SERVICE = "portfolio-lab-tasker"
STATIC_SERVICE = "portfolio-lab-static"
EVIDENCE_FILES = (
    "tasker.json",
    "jobs.json",
    "freshness.json",
    "archive.json",
    "resources.json",
    "authority.json",
    "summary.json",
)

NOW = "2026-09-05T04:00:00Z"
NOW_DT = datetime(2026, 9, 5, 4, 0, 0, tzinfo=timezone.utc)
TODAY = "2026-09-05"
GIB = 1024**3


# ── fake executables (env-driven, no secrets) ─────────────────────────────


def _write_exec(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def write_fake_controller(path: Path) -> Path:
    """One env-driven fake for both persist controllers: the tasker role is
    detected by the presence of --app-dir in argv (FT_* env), otherwise the
    static role applies (FS_* env): delay, exit code, noise lines, argv log,
    payload."""
    source = f"""#!{sys.executable}
import json, os, sys, time
tasker_role = "--app-dir" in sys.argv
prefix = "FT" if tasker_role else "FS"
delay = float(os.environ.get(prefix + "_DELAY", "") or "0")
if delay > 0:
    time.sleep(delay)
exit_code = os.environ.get(prefix + "_EXIT", "")
if exit_code:
    sys.exit(int(exit_code))
stderr_blob = os.environ.get(prefix + "_STDERR", "")
if stderr_blob:
    sys.stderr.write(stderr_blob)
payload = os.environ.get(prefix + "_PAYLOAD", "")
argv_log = os.environ.get(prefix + "_ARGV_LOG", "")
if argv_log:
    with open(argv_log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(sys.argv[1:]) + "\\n")
extra = int(os.environ.get(prefix + "_EXTRA_LINES", "") or "0")
for _ in range(extra):
    print("noise line")
print(payload)
"""
    return _write_exec(path, source)


def write_fake_flock(path: Path, *, mode: str = "normal") -> Path:
    """Mirror of flock(1) 'flock -n FD': non-blocking exclusive lock on the
    numeric file descriptor inherited from the caller. Mode 'error' emulates
    a broken flock(1) failing outside lock contention (usage/exit 2)."""
    if mode == "error":
        source = f"""#!{sys.executable}
import sys
sys.stderr.write("flock: invalid option\\n")
sys.exit(2)
"""
    else:
        source = f"""#!{sys.executable}
import fcntl, sys
fd = None
for arg in sys.argv[1:]:
    if arg.isdigit():
        fd = int(arg)
if fd is None:
    sys.exit(2)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit(1)
sys.exit(0)
"""
    return _write_exec(path, source)


def write_fake_cli(path: Path) -> Path:
    """Fake collector used for pure wrapper-contract tests: logs argv + the
    PORTFOLIO_LAB_ENABLE_ML env and exits with FAKE_CLI_EXIT (default 0)."""
    source = f"""#!{sys.executable}
import json, os, sys
log = os.environ.get("FAKE_CLI_LOG", "")
entry = {{
    "argv": sys.argv[1:],
    "ml": os.environ.get("PORTFOLIO_LAB_ENABLE_ML"),
    "root": os.environ.get("PLDE_ROOT"),
}}
if log:
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\\n")
print(json.dumps({{"schema": "portfolio-lab-daily-evidence/v1", "category": "summary",
    "utc_day": "2026-09-05", "overall": "pass", "notify_grok_bot": False,
    "categories": {{like: "pass" for like in ("tasker", "jobs", "freshness", "archive", "resources", "authority")}}}}))
sys.exit(int(os.environ.get("FAKE_CLI_EXIT", "0")))
"""
    return _write_exec(path, source)


# ── fixtures and helpers ──────────────────────────────────────────────────


@pytest.fixture
def box(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "root"
    app = root / "app"
    www = root / "www"
    (app / "data").mkdir(parents=True)
    (www / "data").mkdir(parents=True)
    run = root / "run"
    run.mkdir()
    tasker_controller = write_fake_controller(tmp_path / "fake-tasker-controller")
    static_controller = write_fake_controller(tmp_path / "fake-static-controller")
    fake_flock = write_fake_flock(tmp_path / "fake-flock")
    fake_cli = write_fake_cli(tmp_path / "fake-cli")
    proof = run / "former-authority-proof.json"
    return SimpleNamespace(
        root=root,
        app=app,
        www=www,
        run=run,
        data=app / "data",
        public=www / "data",
        out=tmp_path / "out",
        tasker_controller=tasker_controller,
        static_controller=static_controller,
        fake_flock=fake_flock,
        fake_cli=fake_cli,
        proof=proof,
    )


def run_cli(
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(CLI)]
    if args:
        cmd.extend(args)
    run_env = dict(os.environ)
    run_env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    if env:
        run_env.update(env)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=run_env,
        timeout=120,
    )


def run_wrapper(
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    run_env = dict(os.environ)
    run_env.update(env)
    return subprocess.run(
        ["/bin/sh", str(WRAPPER)],
        capture_output=True,
        text=True,
        env=run_env,
        timeout=120,
    )


def tasker_payload(box: SimpleNamespace, **over: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": BOX_PERSIST_SCHEMA,
        "state": "active",
        "scheduler_mode": "enabled",
        "identity_exact": True,
        "scheduler_instances": 1,
        "pid": 1234,
        "service_name": TASKER_SERVICE,
        "mode": "production",
        "app_dir": str(box.app.resolve()),
        "web_root": str(box.www.resolve()),
    }
    payload.update(over)
    return payload


def static_payload(box: SimpleNamespace, **over: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": STATIC_PERSIST_SCHEMA,
        "state": "active",
        "identity_exact": True,
        "pid": 4321,
        "mode": "production",
        "service_name": STATIC_SERVICE,
        "web_root": str(box.www.resolve()),
    }
    payload.update(over)
    return payload


def api_payload(
    box: SimpleNamespace,
    *,
    tasks: list[dict[str, object]] | None = None,
    runs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if tasks is None:
        tasks = [
            {
                "id": "portfolio-lab-data",
                "label": "Data pipeline",
                "command": "make fetch-secrets",
                "schedule": "5 * * * *",
                "enabled": True,
                "manual_only": False,
                "timeout_seconds": 3600,
                "paused": False,
                "pause_reason": None,
                "last_status": "success",
                "last_run_id": "run-20260905t035500-abcdef12",
                "last_finished_at": "2026-09-05T04:00:00Z",
                "last_duration_seconds": 42.5,
                "failure_count": 0,
                "consecutive_failures": 0,
            }
        ]
    if runs is None:
        runs = [
            {
                "run_id": "run-20260905t035500-abcdef12",
                "task_id": "portfolio-lab-data",
                "command": ["make", "data"],
                "trigger": "schedule",
                "retry_of": None,
                "status": "success",
                "pid": 9999,
                "started_at": "2026-09-05T03:55:00Z",
                "finished_at": "2026-09-05T04:00:00Z",
                "duration_seconds": 42.5,
                "exit_code": 0,
                "error": None,
                "termination_cause": None,
                "termination_detail": None,
                "log_path": "run-20260905t035500-abcdef12.log",
                "created_at": "2026-09-05T03:55:00Z",
                "updated_at": "2026-09-05T04:00:00Z",
            }
        ]
    return {
        "service": TASKER_SERVICE,
        "backend": "tasker",
        "timestamp": "2026-09-05T04:00:00Z",
        "tasks": tasks,
        "recent_runs": runs,
    }


def proof_payload(**over: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": PROOF_SCHEMA,
        "host_label": "sg01",
        "collected_at": "2026-09-05T03:30:00Z",
        "tasker": {"active": False, "enabled": False},
        "archive_timer": {"active": False, "enabled": False},
    }
    payload.update(over)
    return payload


def write_proof(box: SimpleNamespace, content: object) -> None:
    """Write the proof with the exact production mode 0600 so mode/ownership
    checks are opt-in (tests that vary them flip the mode explicitly)."""
    text = content if isinstance(content, str) else json.dumps(content)
    box.proof.write_text(text, encoding="utf-8")
    box.proof.chmod(0o600)


SECRET = "SUPERSECRETVALUE42"
S3_URL = "s3://portfolio-lab-archives/daily/2026/09/05/portfolio-lab-data-20260905_035500Z.tar"
SHA = "a" * 64


def archive_log_line(ts: str, sha: str = SHA, url: str | None = None) -> str:
    return f"[{ts}] published {url or S3_URL} sha256={sha}"


def prepare_pass_data(box: SimpleNamespace, *, ref: datetime | None = None) -> None:
    """Freshness files (fresh), proof (fresh), archive stamp today + log.

    ``ref`` pins every timestamp; the default is the fixed test NOW so CLI
    tests stay deterministic. Wrapper end-to-end tests pass the real UTC now
    so wrapper time (real ``date -u``) and CLI time agree."""
    ref = ref or NOW_DT
    day = ref.strftime("%Y%m%d")
    for rel in ("app/data/signals.json", "app/data/tasker_status.json",
                "www/data/signals.json", "www/data/tasker_status.json"):
        path = box.root / rel
        path.write_text(json.dumps({"allocation": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}}), encoding="utf-8")
        os.utime(path, (ref.timestamp() - 600, ref.timestamp() - 600))
    collected = (ref - timedelta(minutes=30)).isoformat()
    write_proof(box, proof_payload(collected_at=collected))
    (box.run / "s3-archive-last-utc-day").write_text(day + "\n", encoding="utf-8")
    log_ts = ref.strftime("%Y-%m-%dT%H:%M:%SZ")
    (box.run / "s3-archive.log").write_text(
        archive_log_line(log_ts) + "\n", encoding="utf-8"
    )


def full_env(
    box: SimpleNamespace,
    api_url: str,
    static_url: str,
    **extra: str,
) -> dict[str, str]:
    env: dict[str, str] = {
        "PLDE_ROOT": str(box.root),
        "PLDE_OUTPUT_ROOT": str(box.out),
        "PLDE_TASKER_CONTROLLER": str(box.tasker_controller),
        "PLDE_STATIC_CONTROLLER": str(box.static_controller),
        "PLDE_API_URL": api_url,
        "PLDE_STATIC_URL": static_url,
        "PLDE_AUTHORITY_PROOF": str(box.proof),
        "FT_PAYLOAD": json.dumps(tasker_payload(box)),
        "FS_PAYLOAD": json.dumps(static_payload(box)),
    }
    env.update(extra)
    return env


def base_args(**over: str) -> list[str]:
    args = ["--now", NOW, "--timeout", "10", "--freshness-max-age", "3600"]
    for flag, value in over.items():
        args += [f"--{flag.replace('_', '-')}", value]
    return args


def read_evidence(box: SimpleNamespace, day: str = TODAY) -> dict[str, dict]:
    day_dir = box.out / day
    assert day_dir.is_dir(), f"evidence day dir missing: {day_dir}"
    files: dict[str, dict] = {}
    for name in EVIDENCE_FILES:
        path = day_dir / name
        assert path.is_file(), f"evidence file missing: {path}"
        files[Path(name).stem] = json.loads(path.read_text(encoding="utf-8"))
    return files


def assert_envelope(payload: dict, category: str, status: str) -> None:
    assert payload["schema"] == SCHEMA
    assert payload["category"] == category
    assert payload["status"] == status
    assert payload["collected_at"] == "2026-09-05T04:00:00+00:00"
    assert isinstance(payload["details"], dict)


def assert_tree_clean(*texts: str) -> None:
    """None of the collected text may leak secrets, source paths, or URLs."""
    blob = "\n".join(texts)
    for forbidden in (SECRET, "supersecret", "log_path", "run-20260905t035500-abcdef12.log",
                      "make fetch-secrets", "s3://", "daily/2026", "aws_secret",
                      "AWS_SECRET", "boom "):
        assert forbidden not in blob, f"leaked {forbidden!r}"


@contextmanager
def http_pair(
    *,
    api_body: bytes | None = None,
    api_status: int = 200,
    api_ctype: str = "application/json",
    api_delay: float = 0.0,
    api_headers: dict[str, str] | None = None,
    static_body: bytes = b"<html>ok</html>",
    static_status: int = 200,
    static_delay: float = 0.0,
) -> tuple[str, str]:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (stdlib name)
            time.sleep(_Handler.delays.get(self.path, 0.0))
            parts = _Handler.routes.get(
                self.path, (200, static_body if self.path == "/" else b"{}", "text/html")
            )
            status, body, ctype = parts[:3]
            extra_headers = parts[3] if len(parts) > 3 else {}
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for key, value in extra_headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    _Handler.routes = {}
    _Handler.delays = {}
    if api_body is not None:
        _Handler.routes["/api/tasker/status"] = (
            api_status, api_body, api_ctype, api_headers or {}
        )
    _Handler.routes["/"] = (static_status, static_body, "text/html")
    _Handler.delays = {"/api/tasker/status": api_delay, "/": static_delay}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}/api/tasker/status", f"http://127.0.0.1:{port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("plde_under_test", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── production defaults contract ──────────────────────────────────────────


def test_production_defaults_match_wrapper_wiring():
    mod = _load_cli_module()
    assert mod.DEFAULT_ROOT == DEFAULT_ROOT
    assert mod.DEFAULT_OUTPUT_ROOT == DEFAULT_ROOT / DEFAULT_OUTPUT_REL
    assert mod.DEFAULT_TASKER_CONTROLLER == DEFAULT_TASKER_CONTROLLER
    assert mod.DEFAULT_STATIC_CONTROLLER == DEFAULT_STATIC_CONTROLLER
    assert mod.DEFAULT_API_URL == DEFAULT_API_URL
    assert mod.DEFAULT_STATIC_URL == DEFAULT_STATIC_URL
    assert mod.DEFAULT_TIMEOUT == DEFAULT_TIMEOUT
    assert mod.DEFAULT_FRESHNESS_MAX_AGE == DEFAULT_FRESHNESS_MAX_AGE
    wrapper_text = WRAPPER.read_text(encoding="utf-8")
    assert wrapper_text.startswith("#!/bin/sh\n")
    assert WRAPPER_PYTHON_DEFAULT in wrapper_text
    assert WRAPPER_SCRIPT_REL in wrapper_text
    assert WRAPPER_LOCK_REL in wrapper_text
    assert WRAPPER_STAMP_REL in wrapper_text


# ── wrapper contract ──────────────────────────────────────────────────────


def test_wrapper_is_posix_and_service_free():
    text = WRAPPER.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\n")
    assert "PORTFOLIO_LAB_ENABLE_ML=0" in text
    assert "flock" in text
    assert "exec 9>" in text
    assert "--now" in text
    for forbidden in ("ensure", "activate", "start-candidate", " stop ", "rclone",
                      "source ", "credential", "AWS_SECRET", "aws_access"):
        assert forbidden not in text, f"wrapper must not contain {forbidden!r}"
    result = subprocess.run(
        ["/bin/sh", "-n", str(WRAPPER)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_wrapper_runs_cli_once_per_day_and_stamps_only_on_success(box):
    env = {
        "PLDE_ROOT": str(box.root),
        "PLDE_PYTHON": sys.executable,
        "PLDE_SCRIPT": str(box.fake_cli),
        "PLDE_FLOCK": str(box.fake_flock),
        "FAKE_CLI_LOG": str(box.run / "calls.jsonl"),
        "AWS_SECRET_ACCESS_KEY": SECRET,
    }
    first = run_wrapper(env)
    assert first.returncode == 0, first.stderr
    calls = json.loads((box.run / "calls.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert calls["ml"] == "0"
    assert calls["root"] == str(box.root)
    # The wrapper pins one RFC3339 UTC timestamp and passes it as --now, so
    # the evidence day can never straddle midnight against the stamp day.
    argv = calls["argv"]
    assert len(argv) == 2
    assert argv[0] == "--now"
    parsed_now = datetime.strptime(argv[1], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert abs((parsed_now - datetime.now(timezone.utc)).total_seconds()) < 300
    stamp = (box.run / STAMP_NAME).read_text(encoding="utf-8").strip()
    assert stamp == parsed_now.strftime("%Y-%m-%d")
    assert SECRET not in first.stdout + first.stderr

    second = run_wrapper(env)  # already collected today: no re-invocation
    assert second.returncode == 0
    assert len((box.run / "calls.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert second.stdout == ""


def test_wrapper_does_not_stamp_on_collector_failure(box):
    env = {
        "PLDE_ROOT": str(box.root),
        "PLDE_PYTHON": sys.executable,
        "PLDE_SCRIPT": str(box.fake_cli),
        "PLDE_FLOCK": str(box.fake_flock),
        "FAKE_CLI_LOG": str(box.run / "calls.jsonl"),
        "FAKE_CLI_EXIT": "2",
    }
    failed = run_wrapper(env)
    assert failed.returncode == 2
    assert not (box.run / STAMP_NAME).exists()

    # A later ensure cycle may recheck: exit 1 also leaves no stamp.
    env["FAKE_CLI_EXIT"] = "1"
    broken = run_wrapper(env)
    assert broken.returncode == 1
    assert not (box.run / STAMP_NAME).exists()

    # Recovery on the next cycle succeeds and then stamps.
    env["FAKE_CLI_EXIT"] = "0"
    recovered = run_wrapper(env)
    assert recovered.returncode == 0
    assert (box.run / STAMP_NAME).exists()
    calls = (box.run / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3


def test_wrapper_skips_when_lock_held(box):
    lock_path = box.root / "run" / LOCK_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    env = {
        "PLDE_ROOT": str(box.root),
        "PLDE_PYTHON": sys.executable,
        "PLDE_SCRIPT": str(box.fake_cli),
        "PLDE_FLOCK": str(box.fake_flock),
        "FAKE_CLI_LOG": str(box.run / "calls.jsonl"),
    }
    with open(lock_path, "a+", encoding="utf-8") as holder_fd:
        fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held = run_wrapper(env)
        assert held.returncode == 0, held.stderr
        assert not (box.run / "calls.jsonl").exists()
        assert not (box.run / STAMP_NAME).exists()
        assert held.stdout == ""
        fcntl.flock(holder_fd, fcntl.LOCK_UN)

    released = run_wrapper(env)  # lock free again: collector runs and stamps
    assert released.returncode == 0, released.stderr
    assert (box.run / "calls.jsonl").exists()
    assert (box.run / STAMP_NAME).exists()


def test_wrapper_flock_missing_is_static_error_without_leak(box):
    env = {
        "PLDE_ROOT": str(box.root),
        "PLDE_PYTHON": sys.executable,
        "PLDE_SCRIPT": str(box.fake_cli),
        "PLDE_FLOCK": str(box.root / "no-such-flock"),
        "FAKE_CLI_LOG": str(box.run / "calls.jsonl"),
    }
    result = run_wrapper(env)
    assert result.returncode == 1
    assert result.stdout == ""
    assert not (box.run / "calls.jsonl").exists()
    assert not (box.run / STAMP_NAME).exists()
    assert not (box.root / "run" / LOCK_NAME).exists(), "no lock-file leak"
    assert "flock" in result.stderr.lower()


def test_wrapper_flock_error_is_static_error(box):
    broken = write_fake_flock(box.root.parent / "broken-flock", mode="error")
    env = {
        "PLDE_ROOT": str(box.root),
        "PLDE_PYTHON": sys.executable,
        "PLDE_SCRIPT": str(box.fake_cli),
        "PLDE_FLOCK": str(broken),
        "FAKE_CLI_LOG": str(box.run / "calls.jsonl"),
    }
    result = run_wrapper(env)
    assert result.returncode == 1
    assert result.stdout == ""
    assert not (box.run / "calls.jsonl").exists()
    assert not (box.run / STAMP_NAME).exists()
    assert "lock held" not in result.stderr  # contention message only for exit 1
    assert "flock" in result.stderr.lower()


def test_wrapper_end_to_end_with_real_cli(box):
    real_now = datetime.now(timezone.utc)
    prepare_pass_data(box, ref=real_now)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env.update(
            {
                "PLDE_ROOT": str(box.root),
                "PLDE_PYTHON": sys.executable,
                "PLDE_SCRIPT": str(CLI),
                "PLDE_FLOCK": str(box.fake_flock),
                "PLDE_TIMEOUT": "10",
                "PLDE_FRESHNESS_MAX_AGE": "3600",
                "PLDE_AUTHORITY_PROOF": str(box.proof),
            }
        )
        result = run_wrapper(env)
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    summary = json.loads(lines[0])
    assert summary["overall"] == "pass"
    real_utc_day = real_now.strftime("%Y-%m-%d")
    day_dir = box.out / real_utc_day
    for name in EVIDENCE_FILES:
        assert (day_dir / name).is_file()
    stamp = (box.run / STAMP_NAME).read_text(encoding="utf-8").strip()
    assert stamp == real_utc_day
    assert summary["utc_day"] == stamp  # --now pin: evidence day == stamp day


def test_wrapper_fail_exit_code_with_real_cli(box):
    real_now = datetime.now(timezone.utc)
    prepare_pass_data(box, ref=real_now)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env.update(
            {
                "PLDE_ROOT": str(box.root),
                "PLDE_PYTHON": sys.executable,
                "PLDE_SCRIPT": str(CLI),
                "PLDE_FLOCK": str(box.fake_flock),
                "PLDE_TIMEOUT": "10",
                "PLDE_FRESHNESS_MAX_AGE": "3600",
                "FT_PAYLOAD": json.dumps(tasker_payload(box, scheduler_instances=2)),
            }
        )
        result = run_wrapper(env)
    assert result.returncode == 2
    real_utc_day = real_now.strftime("%Y-%m-%d")
    assert not (box.run / STAMP_NAME).exists()
    assert (box.out / real_utc_day / "summary.json").is_file()
    assert json.loads(result.stdout.splitlines()[0])["overall"] == "fail"


# ── CLI: full pass, aggregation, exit codes ───────────────────────────────


def test_full_pass_end_to_end(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, "exactly one stdout line"
    summary = json.loads(lines[0])
    assert summary["schema"] == SCHEMA
    assert summary["utc_day"] == TODAY
    assert summary["overall"] == "pass"
    assert summary["notify_grok_bot"] is False
    assert summary["categories"] == {
        "tasker": "pass",
        "jobs": "pass",
        "freshness": "pass",
        "archive": "pass",
        "resources": "pass",
        "authority": "pass",
    }
    assert summary["output_dir"] == str(box.out / TODAY)
    assert result.stderr == ""

    files = read_evidence(box)
    assert set(files) == {Path(name).stem for name in EVIDENCE_FILES}
    for name, cat in zip(EVIDENCE_FILES, ("tasker", "jobs", "freshness", "archive",
                                          "resources", "authority", "summary")):
        assert_envelope(files[Path(name).stem], cat, "pass")
    tasker_details = files["tasker"]["details"]
    assert tasker_details["tasker_controller"]["state"] == "active"
    assert tasker_details["tasker_controller"]["identity_exact"] is True
    assert tasker_details["tasker_controller"]["scheduler_instances"] == 1
    assert tasker_details["tasker_controller"]["service_name_exact"] is True
    assert tasker_details["tasker_controller"]["mode"] == "production"
    assert tasker_details["static_controller"]["state"] == "active"
    assert tasker_details["static_controller"]["identity_exact"] is True
    jobs = files["jobs"]["details"]
    assert jobs["api"]["http_status"] == 200
    assert jobs["api"]["status"] == "pass"
    assert jobs["static_root"]["status"] == "pass"
    assert files["summary"]["details"]["overall"] == "pass"
    assert files["summary"]["details"]["notify_grok_bot"] is False
    assert files["summary"]["details"]["categories"] == summary["categories"]


def test_warning_aggregation_and_exit_zero(box):
    prepare_pass_data(box)
    stale = box.root / "app/data/signals.json"
    os.utime(stale, (NOW_DT.timestamp() - 7200, NOW_DT.timestamp() - 7200))  # > max age
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.splitlines()[0])
    assert summary["overall"] == "warning"
    assert summary["notify_grok_bot"] is False
    assert summary["categories"]["freshness"] == "warning"
    freshness = read_evidence(box)["freshness"]
    assert_envelope(freshness, "freshness", "warning")
    by_path = {entry["path"]: entry for entry in freshness["details"]["files"]}
    assert set(by_path) == {
        "app/data/signals.json",
        "app/data/tasker_status.json",
        "www/data/signals.json",
        "www/data/tasker_status.json",
    }
    assert by_path["app/data/signals.json"]["status"] == "warning"
    assert abs(by_path["app/data/signals.json"]["age_seconds"] - 7200) <= 2
    assert by_path["app/data/signals.json"]["size_bytes"] > 0
    assert by_path["app/data/tasker_status.json"]["status"] == "pass"
    assert abs(by_path["app/data/tasker_status.json"]["age_seconds"] - 600) <= 2
    assert by_path["www/data/tasker_status.json"]["present"] is True


def test_fail_aggregation_notify_and_exit_two(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env["FT_PAYLOAD"] = json.dumps(tasker_payload(box, identity_exact=False))
        result = run_cli(base_args(), env=env)
    assert result.returncode == 2
    summary = json.loads(result.stdout.splitlines()[0])
    assert summary["overall"] == "fail"
    assert summary["notify_grok_bot"] is True
    assert summary["categories"]["tasker"] == "fail"
    files = read_evidence(box)
    assert_envelope(files["tasker"], "tasker", "fail")
    assert files["tasker"]["details"]["tasker_controller"]["status"] == "fail"
    assert files["tasker"]["details"]["tasker_controller"]["identity_exact"] is False
    assert_envelope(files["summary"], "summary", "fail")
    assert files["summary"]["details"]["notify_grok_bot"] is True


def test_same_day_rerun_is_deterministic_replacement(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        first = run_cli(base_args(), env=env)
        day_dir = box.out / TODAY
        first_bytes = {name: (day_dir / name).read_bytes() for name in EVIDENCE_FILES}
        second = run_cli(base_args(), env=env)
    assert first.returncode == 0 and second.returncode == 0
    entries = sorted(p.name for p in day_dir.iterdir())
    assert entries == sorted(EVIDENCE_FILES), "bounded directory"
    # All observation-derived files are byte-identical under the same inputs;
    # resources.json is a live snapshot (statvfs//proc drift between runs), so
    # it is checked structure-wise instead of byte-wise.
    for name in EVIDENCE_FILES:
        if name == "resources.json":
            continue
        assert (day_dir / name).read_bytes() == first_bytes[name], name
    first_resources = json.loads(first_bytes["resources.json"])
    second_resources = json.loads((day_dir / "resources.json").read_text(encoding="utf-8"))
    assert first_resources.keys() == second_resources.keys()
    assert first_resources["status"] == second_resources["status"] == "pass"


def test_same_day_non_directory_or_symlink_day_path_rejected(box):
    """A same-day target must be a real non-symlink directory: a regular
    file or a symlink at the day path is a clean exit-1 rejection."""
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        first = run_cli(base_args(), env=env)
    assert first.returncode == 0
    day_dir = box.out / TODAY

    shutil.rmtree(day_dir)
    day_dir.write_text("not a directory", encoding="utf-8")
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        res = run_cli(base_args(), env=env)
    assert res.returncode == 1
    assert res.stdout == ""
    assert "Traceback" not in res.stderr

    day_dir.unlink()
    elsewhere = box.out / "elsewhere"
    elsewhere.mkdir()
    day_dir.symlink_to(elsewhere)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        res = run_cli(base_args(), env=env)
    assert res.returncode == 1
    assert "Traceback" not in res.stderr
    assert day_dir.is_symlink(), "rejected symlink day path must be left in place"


def test_same_day_unexpected_entries_rejected_and_never_deleted(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        first = run_cli(base_args(), env=env)
    assert first.returncode == 0
    day_dir = box.out / TODAY

    extra = day_dir / "evil.json"
    extra.write_text("{}", encoding="utf-8")
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        res = run_cli(base_args(), env=env)
    assert res.returncode == 1, "unexpected entry must be rejected"
    assert res.stdout == ""
    assert "Traceback" not in res.stderr
    assert extra.read_text(encoding="utf-8") == "{}", "unexpected entry must not be deleted"
    assert (day_dir / "summary.json").exists(), "known files must be left untouched"

    extra.unlink()
    (day_dir / "summary.json").unlink()  # missing expected entry is also unexpected
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        res = run_cli(base_args(), env=env)
    assert res.returncode == 1
    assert "Traceback" not in res.stderr


def test_same_day_symlink_or_non_regular_expected_file_rejected(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        first = run_cli(base_args(), env=env)
    assert first.returncode == 0
    day_dir = box.out / TODAY

    summary = day_dir / "summary.json"
    summary.unlink()
    summary.symlink_to(box.data / "signals.json")
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        res = run_cli(base_args(), env=env)
    assert res.returncode == 1
    assert "Traceback" not in res.stderr
    assert summary.is_symlink(), "symlinked expected file must not be replaced"

    summary.unlink()
    summary.mkdir()  # non-regular expected file
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        res = run_cli(base_args(), env=env)
    assert res.returncode == 1
    assert "Traceback" not in res.stderr
    assert summary.is_dir(), "non-regular expected file must not be replaced"


def test_same_day_rerun_hardens_day_dir_mode(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        first = run_cli(base_args(), env=env)
    assert first.returncode == 0
    day_dir = box.out / TODAY
    os.chmod(day_dir, 0o755)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        second = run_cli(base_args(), env=env)
    assert second.returncode == 0, second.stderr
    assert stat.S_IMODE(day_dir.stat().st_mode) == 0o700


def test_unit_controller_poll_sleeps_while_child_alive_at_eof(tmp_path, monkeypatch):
    """The bounded poll loop must keep sleeping while the child is alive even
    after both pipes hit EOF; otherwise it busy-spins (zero sleeps) until the
    deadline once EOF is observed."""
    mod = _load_cli_module()
    child = tmp_path / "hold.py"
    child.write_text(
        f"#!{sys.executable}\n"
        "import os, sys, time\n"
        "sys.stdout.write('ready\\n')\n"
        "sys.stdout.flush()\n"
        "os.close(1)\n"
        "os.close(2)\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    child.chmod(0o755)
    sleeps: list[float] = []
    real_sleep = mod.time.sleep

    def recording_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        real_sleep(seconds)

    monkeypatch.setattr(mod.time, "sleep", recording_sleep)
    res = mod.run_controller(child, [], 3.0)
    # The child drains 'ready' then EOFs both pipes while staying alive for
    # 5s: the loop must sleep through the remaining ~3s at 10ms per sleep
    # (expect ~300 sleeps), not spin hot (a handful of pre-EOF sleeps only).
    assert len(sleeps) >= 100, f"poll loop busy-spun: only {len(sleeps)} sleeps"
    assert res["status"] == "fail"
    assert "timeout" in res["reason"]


def test_output_permissions(box):
    prepare_pass_data(box)
    # Pre-create the output root with loose mode: the collector must harden it.
    box.out.mkdir(parents=True, exist_ok=True)
    os.chmod(box.out, 0o755)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0, result.stderr
    assert stat.S_IMODE(box.out.stat().st_mode) == 0o700
    day_dir = box.out / TODAY
    assert stat.S_IMODE(day_dir.stat().st_mode) == 0o700
    for name in EVIDENCE_FILES:
        path = day_dir / name
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, name
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema"] == SCHEMA
        assert payload["status"] in ("pass", "warning", "fail")


def test_stale_temp_dir_does_not_break_evidence(tmp_path, monkeypatch):
    """A stale/colliding .tmp-* sibling (same process pid reused) must not
    crash the writer: temp dirs are unique and stale ones are left alone."""
    mod = _load_cli_module()
    out_root = tmp_path / "out"
    out_root.mkdir()
    day = "2026-09-05"
    stale = out_root / f".tmp-{day}-424242"
    stale.mkdir()
    (stale / "junk.json").write_text("junk", encoding="utf-8")
    monkeypatch.setattr(mod.os, "getpid", lambda: 424242)
    payloads = {
        "summary": {
            "schema": mod.SCHEMA,
            "category": "summary",
            "collected_at": "2026-09-05T04:00:00+00:00",
            "status": "pass",
            "details": {},
        }
    }
    target = mod.write_evidence(out_root, day, payloads)
    assert target.is_dir()
    assert sorted(p.name for p in target.iterdir()) == ["summary.json"]
    assert stale.is_dir()  # stale sibling untouched, no traceback


def test_write_failure_is_clean_exit_one(box):
    prepare_pass_data(box)
    parent = box.out.parent
    os.chmod(parent, 0o500)  # block evidence creation
    try:
        with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
            env = full_env(box, api_url, static_url)
            result = run_cli(base_args(), env=env)
    finally:
        os.chmod(parent, 0o700)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    assert "write" in result.stderr.lower()


# ── CLI: invalid configuration (exit 1) ───────────────────────────────────


def test_relative_root_rejected(box):
    with http_pair() as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env["PLDE_ROOT"] = "relative/root"
        result = run_cli([], env=env)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "ROOT" in result.stderr


def test_relative_output_root_rejected(box):
    with http_pair() as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env["PLDE_OUTPUT_ROOT"] = "relative/out"
        result = run_cli([], env=env)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "OUTPUT_ROOT" in result.stderr


def test_relative_controller_and_proof_rejected(box):
    with http_pair() as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env["PLDE_TASKER_CONTROLLER"] = "relative-controller"
        assert run_cli([], env=env).returncode == 1
        env2 = full_env(box, api_url, static_url)
        env2["PLDE_STATIC_CONTROLLER"] = "relative-controller"
        assert run_cli([], env=env2).returncode == 1
        env3 = full_env(box, api_url, static_url)
        env3["PLDE_AUTHORITY_PROOF"] = "relative-proof.json"
        assert run_cli([], env=env3).returncode == 1
    assert not (box.out / TODAY).exists()


def test_symlinked_controller_rejected(box):
    link = box.root.parent / "controller-link"
    link.symlink_to(box.tasker_controller)
    with http_pair() as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env["PLDE_TASKER_CONTROLLER"] = str(link)
        result = run_cli([], env=env)
    assert result.returncode == 1
    assert result.stdout == ""
    assert not (box.out / TODAY).exists()


def test_unsafe_output_placement_rejected(box):
    with http_pair() as (api_url, static_url):
        for unsafe in (str(box.root), str(box.www), str(box.run), str(box.data)):
            env = full_env(box, api_url, static_url)
            env["PLDE_OUTPUT_ROOT"] = unsafe
            result = run_cli([], env=env)
            assert result.returncode == 1, unsafe
            assert result.stdout == ""
            assert "OUTPUT_ROOT" in result.stderr
    assert not (box.out / TODAY).exists()


def test_missing_root_is_config_error(box):
    with http_pair() as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env["PLDE_ROOT"] = str(box.root.parent / "does-not-exist")
        result = run_cli([], env=env)
    assert result.returncode == 1
    assert result.stdout == ""


def test_invalid_now_and_timeout_and_max_age_rejected(box):
    for args in (["--now", "not-a-date"], ["--timeout", "-1"], ["--timeout", "abc"],
                 ["--freshness-max-age", "0"]):
        result = run_cli(args)
        assert result.returncode == 1, args
        assert result.stdout == ""
        assert result.stderr
    # Non-finite values must be rejected at config time with exit 1, an
    # empty stdout, and no traceback (a valid root isolates the config
    # rejection as the only failure path).
    for flag in ("--timeout", "--freshness-max-age"):
        for value in ("nan", "inf", "-inf", "1e309"):
            result = run_cli([flag, value], env={"PLDE_ROOT": str(box.root)})
            assert result.returncode == 1, (flag, value)
            assert result.stdout == "", (flag, value)
            assert "Traceback" not in result.stderr, (flag, value)
            assert result.stderr, (flag, value)


def test_unknown_flag_exits_one_with_empty_stdout():
    result = run_cli(["--definitely-unknown-flag"])
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr


def test_missing_flag_value_exits_one_with_empty_stdout():
    result = run_cli(["--now"])
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr


def test_non_loopback_and_unsafe_urls_rejected():
    for url in ("http://example.com/api/tasker/status",
                "http://127.0.0.1:8000/api/tasker/status?token=abc",
                "http://user:pass@127.0.0.1:8000/x",
                "ftp://127.0.0.1/x",
                "https://127.0.0.1:8000/x"):
        result = run_cli([], env={"PLDE_API_URL": url})
        assert result.returncode == 1, url
        assert result.stdout == ""
        assert "API_URL" in result.stderr
    result = run_cli([], env={"PLDE_STATIC_URL": "http://192.168.1.1/"})
    assert result.returncode == 1
    assert "STATIC_URL" in result.stderr


def test_now_with_utc_offset_deterministic_day(box):
    prepare_pass_data(box)
    (box.run / "s3-archive-last-utc-day").write_text("20260904\n", encoding="utf-8")
    (box.run / "s3-archive.log").write_text(
        archive_log_line("2026-09-04T03:55:00Z") + "\n", encoding="utf-8")
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(
            ["--now", "2026-09-05T01:00:00+02:00", "--timeout", "10",
             "--freshness-max-age", "3600"],
            env=env,
        )
    # 2026-09-05T01:00:00+02:00 is 2026-09-04T23:00:00Z: the evidence day is UTC.
    assert result.returncode == 0, result.stderr
    assert (box.out / "2026-09-04").is_dir()
    assert not (box.out / "2026-09-05").exists()
    assert json.loads(result.stdout.splitlines()[0])["utc_day"] == "2026-09-04"


def test_now_offset_evidences_utc_day(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(["--now", "2026-09-05T05:30:00+02:00", "--timeout", "10",
                          "--freshness-max-age", "3600"], env=env)
    assert result.returncode == 0, result.stderr
    assert (box.out / "2026-09-05").is_dir()  # 05:30+02:00 == 03:30Z same day


# ── CLI: tasker controller checks ─────────────────────────────────────────


def test_controller_argv_contract(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env["FT_ARGV_LOG"] = str(box.run / "tasker-argv.jsonl")
        env["FS_ARGV_LOG"] = str(box.run / "static-argv.jsonl")
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0, result.stderr
    tasker_argv = json.loads((box.run / "tasker-argv.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert tasker_argv == [
        "status", "--read-only", "--mode", "production",
        "--app-dir", str(box.app.resolve()),
        "--web-root", str(box.www.resolve()),
        "--service-name", TASKER_SERVICE,
    ]
    static_argv = json.loads((box.run / "static-argv.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert static_argv == [
        "status", "--read-only", "--mode", "production",
        "--web-root", str(box.www.resolve()),
        "--service-name", STATIC_SERVICE,
    ]


@pytest.mark.parametrize(
    ("over", "expect_fail_field"),
    [
        ({"state": "inactive"}, "state"),
        ({"identity_exact": False}, "identity"),
        ({"scheduler_instances": 2}, "scheduler"),
        ({"service_name": "other-service"}, "service"),
        ({"mode": "candidate"}, "mode"),
        ({"app_dir": "/somewhere/else"}, "path"),
        ({"web_root": "/somewhere/else/www"}, "path"),
    ],
)
def test_tasker_controller_identity_failures(box, over, expect_fail_field):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env["FT_PAYLOAD"] = json.dumps(tasker_payload(box, **over))
        result = run_cli(base_args(), env=env)
    assert result.returncode == 2, over
    summary = json.loads(result.stdout.splitlines()[0])
    assert summary["categories"]["tasker"] == "fail"
    tasker_details = read_evidence(box)["tasker"]["details"]["tasker_controller"]
    assert tasker_details["status"] == "fail"
    assert expect_fail_field in tasker_details["reason"]


def test_tasker_controller_bad_schema_and_malformed_output(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env["FT_PAYLOAD"] = json.dumps(tasker_payload(box, schema="portfolio-lab-migration-evidence/v1"))
        result = run_cli(base_args(), env=env)
    assert result.returncode == 2
    assert json.loads(result.stdout.splitlines()[0])["categories"]["tasker"] == "fail"

    env2 = full_env(box, api_url, static_url)
    env2["FT_PAYLOAD"] = "not json at all"
    env2["FT_EXTRA_LINES"] = "2"
    result2 = run_cli(base_args(), env=env2)
    assert result2.returncode == 2
    assert json.loads(result2.stdout.splitlines()[0])["categories"]["tasker"] == "fail"


def test_tasker_controller_nonzero_exit_and_missing_binary(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env["FT_EXIT"] = "3"
        result = run_cli(base_args(), env=env)
    assert result.returncode == 2
    assert json.loads(result.stdout.splitlines()[0])["categories"]["tasker"] == "fail"

    env2 = full_env(box, api_url, static_url)
    env2["PLDE_TASKER_CONTROLLER"] = str(box.root / "no-such-controller")
    result2 = run_cli(base_args(), env=env2)
    assert result2.returncode == 2
    assert json.loads(result2.stdout.splitlines()[0])["categories"]["tasker"] == "fail"


def test_controller_timeout_is_bounded(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env["FT_DELAY"] = "30"
        env["FT_PAYLOAD"] = json.dumps(tasker_payload(box))
        started = time.monotonic()
        result = run_cli(["--now", NOW, "--timeout", "1", "--freshness-max-age", "3600"], env=env)
        elapsed = time.monotonic() - started
    assert result.returncode == 2
    assert elapsed < 15, "timeout must actually bound the controller"
    assert json.loads(result.stdout.splitlines()[0])["categories"]["tasker"] == "fail"
    reason = read_evidence(box)["tasker"]["details"]["tasker_controller"]["reason"]
    assert "timeout" in reason


def test_controller_stdout_exceeding_bound_fails_static_reason(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env["FT_PAYLOAD"] = json.dumps(tasker_payload(box, padding="x" * 70000))
        started = time.monotonic()
        result = run_cli(base_args(), env=env)
        elapsed = time.monotonic() - started
    assert result.returncode == 2
    assert elapsed < 15, "bound must stop the read promptly"
    assert json.loads(result.stdout.splitlines()[0])["categories"]["tasker"] == "fail"
    tasker = read_evidence(box)["tasker"]["details"]["tasker_controller"]
    assert tasker["status"] == "fail"
    assert "bound" in tasker["reason"]


def test_controller_stderr_exceeding_bound_is_secret_free(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env["FT_STDERR"] = SECRET + "Z" * 70000
        env["FT_PAYLOAD"] = json.dumps(tasker_payload(box))
        started = time.monotonic()
        result = run_cli(base_args(), env=env)
        elapsed = time.monotonic() - started
    assert result.returncode == 2
    assert elapsed < 15, "bound must stop the read promptly"
    tasker = read_evidence(box)["tasker"]["details"]["tasker_controller"]
    assert tasker["status"] == "fail"
    assert "bound" in tasker["reason"]
    assert_tree_clean(json.dumps(read_evidence(box)), result.stdout, result.stderr)


def test_controller_failure_reason_is_static_and_secret_free(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env["FT_EXIT"] = "5"
        env["FT_STDERR"] = "boom " + SECRET
        env["FT_PAYLOAD"] = json.dumps(tasker_payload(box))
        result = run_cli(base_args(), env=env)
    assert result.returncode == 2
    tasker = read_evidence(box)["tasker"]["details"]["tasker_controller"]
    assert tasker["status"] == "fail"
    assert tasker["reason"] == "controller exited with code 5"
    assert_tree_clean(json.dumps(read_evidence(box)), result.stdout, result.stderr)


def test_static_controller_failures(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        for over in ({"state": "inactive"}, {"identity_exact": False},
                     {"service_name": "wrong"}, {"mode": "candidate"},
                     {"web_root": "/somewhere/else/www"}):
            env = full_env(box, api_url, static_url)
            env["FS_PAYLOAD"] = json.dumps(static_payload(box, **over))
            result = run_cli(base_args(), env=env)
            assert result.returncode == 2, over
            assert json.loads(result.stdout.splitlines()[0])["categories"]["tasker"] == "fail"
            static_details = read_evidence(box)["tasker"]["details"]["static_controller"]
            assert static_details["status"] == "fail"
            assert static_details["reason"]


# ── CLI: jobs (HTTP API + static root) ────────────────────────────────────


def test_jobs_compact_retention_and_redaction(box):
    prepare_pass_data(box)
    api = api_payload(box)
    api["tasks"][0]["command"] = "make fetch-secrets"
    api["tasks"][0]["pause_reason"] = "hold for " + SECRET
    api["recent_runs"][0]["log_path"] = "run-20260905t035500-abcdef12.log"
    api["recent_runs"][0]["command"] = ["make", "fetch-secrets"]
    api["recent_runs"][0]["error"] = "boom " + SECRET
    api["recent_runs"][0]["termination_detail"] = SECRET
    with http_pair(api_body=json.dumps(api).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0, result.stderr
    jobs = read_evidence(box)["jobs"]
    assert_envelope(jobs, "jobs", "pass")
    task = jobs["details"]["api"]["tasks"][0]
    assert task["id"] == "portfolio-lab-data"
    assert task["label"] == "Data pipeline"
    assert task["last_status"] == "success"
    for forbidden_key in ("command", "pause_reason", "log_path", "error",
                          "termination_detail", "pid", "timeout_seconds",
                          "created_at", "updated_at", "retry_of"):
        assert forbidden_key not in task, forbidden_key
    run_ = jobs["details"]["api"]["recent_runs"][0]
    assert run_["run_id"] == "run-20260905t035500-abcdef12"
    assert run_["status"] == "success"
    assert run_["exit_code"] == 0
    for forbidden_key in ("command", "log_path", "pid", "error", "termination_cause",
                          "termination_detail", "created_at", "updated_at"):
        assert forbidden_key not in run_, forbidden_key
    blob = json.dumps(jobs) + result.stdout + result.stderr
    assert SECRET not in blob
    assert "fetch-secrets" not in blob


def test_jobs_api_http_errors_and_bad_json(box):
    prepare_pass_data(box)
    with http_pair(api_body=b"not json", static_status=200) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
        assert result.returncode == 2
        jobs = read_evidence(box)["jobs"]
        assert jobs["details"]["api"]["status"] == "fail"
        assert "json" in jobs["details"]["api"]["reason"].lower()
    with http_pair(api_body=b"{}", api_status=500, static_status=200) as (api_url2, static_url2):
        env2 = full_env(box, api_url2, static_url2)
        result2 = run_cli(base_args(), env=env2)
        assert result2.returncode == 2
        jobs2 = read_evidence(box)["jobs"]
        assert jobs2["details"]["api"]["status"] == "fail"
        assert jobs2["details"]["api"]["http_status"] == 500


def test_jobs_static_root_non_200_fails(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode(),
                   static_status=503) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 2
    jobs = read_evidence(box)["jobs"]
    assert jobs["details"]["static_root"]["status"] == "fail"
    assert jobs["details"]["static_root"]["http_status"] == 503


def test_jobs_connection_refused_fails(box):
    prepare_pass_data(box)
    port = free_port()
    env = full_env(box, f"http://127.0.0.1:{port}/api/tasker/status",
                   f"http://127.0.0.1:{port}/")
    result = run_cli(base_args(), env=env)
    assert result.returncode == 2
    jobs = read_evidence(box)["jobs"]
    assert jobs["details"]["api"]["status"] == "fail"
    assert jobs["details"]["static_root"]["status"] == "fail"


def test_jobs_redirect_is_not_followed(box):
    prepare_pass_data(box)
    with http_pair(
        api_body=json.dumps(api_payload(box)).encode(),
        api_status=302,
        api_headers={"Location": "http://example.com/api/tasker/status"},
    ) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 2
    jobs = read_evidence(box)["jobs"]
    assert jobs["details"]["api"]["status"] == "fail"
    assert jobs["details"]["api"]["http_status"] == 302


def test_jobs_ignores_proxy_environment(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY"):
            env[name] = "http://127.0.0.1:1"  # a live proxy here would fail the probe
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[0])["overall"] == "pass"


def test_jobs_http_timeout_is_bounded(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode(),
                   api_delay=30.0) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        started = time.monotonic()
        result = run_cli(["--now", NOW, "--timeout", "1", "--freshness-max-age", "3600"], env=env)
        elapsed = time.monotonic() - started
    assert result.returncode == 2
    assert elapsed < 15
    jobs = read_evidence(box)["jobs"]
    assert jobs["details"]["api"]["status"] == "fail"
    assert "timeout" in jobs["details"]["api"]["reason"]


def test_jobs_response_exceeding_fetch_bound_fails(box):
    prepare_pass_data(box)
    big = json.dumps(
        {"tasks": [{"id": f"t{i}", "label": "x" * 300, "schedule": "* * * * *",
                    "enabled": True, "paused": False, "last_status": "success",
                    "last_finished_at": None, "last_duration_seconds": None,
                    "consecutive_failures": 0, "failure_count": 0}
                   for i in range(6000)], "recent_runs": []}
    ).encode()  # ~1.9 MiB: beyond the 1 MiB fetch cap
    assert len(big) > 1048576
    with http_pair(api_body=big) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 2
    jobs = read_evidence(box)["jobs"]
    assert jobs["details"]["api"]["status"] == "fail"
    assert "bound" in jobs["details"]["api"]["reason"]


def test_jobs_serialized_size_bound_is_write_failure(box):
    prepare_pass_data(box)
    big = json.dumps(
        {"tasks": [{"id": f"t{i}", "label": "y" * 250, "schedule": None,
                    "enabled": True, "paused": False, "last_status": "success",
                    "last_finished_at": None, "last_duration_seconds": None,
                    "consecutive_failures": 0, "failure_count": 0}
                   for i in range(1500)], "recent_runs": []}
    ).encode()  # ~380 KiB: passes the fetch cap, exceeds the per-file cap
    assert 262144 < len(big) < 1048576
    with http_pair(api_body=big) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 1
    assert result.stdout == ""
    assert not (box.out / TODAY).exists(), "no partial evidence on write failure"
    assert "size" in result.stderr.lower()


# ── CLI: freshness ────────────────────────────────────────────────────────


def test_freshness_missing_and_stale_are_warnings_with_no_content_leak(box):
    # signals.json missing in both roots; tasker_status.json stale in app/data
    # and fresh in www/data. File contents must never reach evidence.
    stale = box.data / "tasker_status.json"
    stale.write_text(SECRET, encoding="utf-8")
    os.utime(stale, (NOW_DT.timestamp() - 7200, NOW_DT.timestamp() - 7200))
    fresh = box.public / "tasker_status.json"
    fresh.write_text("x", encoding="utf-8")
    os.utime(fresh, (NOW_DT.timestamp() - 60, NOW_DT.timestamp() - 60))
    assert not (box.data / "signals.json").exists()
    assert not (box.public / "signals.json").exists()
    write_proof(box, proof_payload())
    (box.run / "s3-archive-last-utc-day").write_text("20260905\n", encoding="utf-8")
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[0])["overall"] == "warning"
    freshness = read_evidence(box)["freshness"]
    assert_envelope(freshness, "freshness", "warning")
    by_path = {entry["path"]: entry for entry in freshness["details"]["files"]}
    assert by_path["app/data/signals.json"]["present"] is False
    assert by_path["app/data/signals.json"]["status"] == "warning"
    assert "missing" in by_path["app/data/signals.json"]["reason"]
    assert by_path["app/data/tasker_status.json"]["status"] == "warning"
    assert abs(by_path["app/data/tasker_status.json"]["age_seconds"] - 7200) <= 2
    assert by_path["www/data/tasker_status.json"]["status"] == "pass"
    assert by_path["www/data/tasker_status.json"]["size_bytes"] == 1
    assert "mtime" in by_path["www/data/tasker_status.json"]
    assert_tree_clean(json.dumps(freshness), result.stdout, result.stderr)


# ── CLI: archive evidence ─────────────────────────────────────────────────


def test_archive_stamp_tiers(box):
    api = json.dumps(api_payload(box)).encode()
    for stamp, expected in (("20260905", "pass"), ("20260904", "warning"), ("20260903", "fail")):
        prepare_pass_data(box)
        (box.run / "s3-archive-last-utc-day").write_text(stamp + "\n", encoding="utf-8")
        (box.run / "s3-archive.log").write_text(
            archive_log_line("2026-09-04T04:18:31Z") + "\n", encoding="utf-8")
        with http_pair(api_body=api) as (api_url, static_url):
            env = full_env(box, api_url, static_url)
            result = run_cli(base_args(), env=env)
        assert result.returncode == (0 if expected != "fail" else 2), stamp
        archive = read_evidence(box)["archive"]
        assert_envelope(archive, "archive", expected)
        assert archive["details"]["utc_day"] == {
            "20260905": "2026-09-05", "20260904": "2026-09-04", "20260903": "2026-09-03",
        }[stamp]


def test_archive_missing_and_malformed_stamp_fail(box):
    api = json.dumps(api_payload(box)).encode()
    for stamp in (None, "garbage", "202609"):
        prepare_pass_data(box)
        stamp_path = box.run / "s3-archive-last-utc-day"
        if stamp is None:
            stamp_path.unlink()
        else:
            stamp_path.write_text(stamp + "\n", encoding="utf-8")
        with http_pair(api_body=api) as (api_url, static_url):
            env = full_env(box, api_url, static_url)
            result = run_cli(base_args(), env=env)
        assert result.returncode == 2, stamp
        archive = read_evidence(box)["archive"]
        assert archive["status"] == "fail"
        assert archive["details"]["utc_day"] is None


def test_archive_oversized_stamp_fails_bounded(box):
    api = json.dumps(api_payload(box)).encode()
    prepare_pass_data(box)
    # Whitespace-only padding: an unbounded read would strip and parse this
    # as the valid current day; the collector must bound the read instead.
    (box.run / "s3-archive-last-utc-day").write_text(
        "20260905" + " " * 100000, encoding="utf-8")
    with http_pair(api_body=api) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 2
    archive = read_evidence(box)["archive"]
    assert archive["status"] == "fail"
    assert archive["details"]["utc_day"] is None
    assert "malformed" in archive["details"]["reason"]


def test_archive_calendar_invalid_stamp_fails_cleanly_not_traceback(box):
    """Eight-digit but calendar-invalid stamps (month > 12, impossible day,
    zero date) are archive fail evidence with exit 2 — never a traceback
    with exit 1 while formatting the day."""
    api = json.dumps(api_payload(box)).encode()
    for stamp in ("20261301", "20260230", "20260000"):
        prepare_pass_data(box)
        (box.run / "s3-archive-last-utc-day").write_text(stamp + "\n", encoding="utf-8")
        with http_pair(api_body=api) as (api_url, static_url):
            env = full_env(box, api_url, static_url)
            result = run_cli(base_args(), env=env)
        assert result.returncode == 2, stamp
        assert "Traceback" not in result.stderr, stamp
        archive = read_evidence(box)["archive"]
        assert archive["status"] == "fail", stamp
        assert archive["details"]["utc_day"] is None, stamp
        assert "malformed" in archive["details"]["reason"], stamp


def test_archive_log_redaction_and_latest_success(box):
    prepare_pass_data(box)
    log = "\n".join([
        "[2026-09-05T03:55:00Z] published s3://portfolio-lab-archives/daily/2026/09/05/portfolio-lab-data-20260905_035500Z.tar sha256=" + SHA,
        "AWS_SECRET_ACCESS_KEY=" + SECRET,
        "credentials file: /home/box/.config/portfolio-lab/s3-credentials.env",
        "[2026-09-05T03:56:00Z] published s3://portfolio-lab-archives/daily/2026/09/05/other-key-20260905_035600Z.tar sha256=" + "b" * 64,
        "rclone: ERROR : failed to copy: access denied",
    ])
    (box.run / "s3-archive.log").write_text(log + "\n", encoding="utf-8")
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0, result.stderr
    archive = read_evidence(box)["archive"]
    assert archive["status"] == "pass"
    details = archive["details"]
    assert set(details) == {"utc_day", "latest_success", "sha256"}
    assert details["utc_day"] == "2026-09-05"
    assert details["latest_success"] == "2026-09-05T03:56:00Z"
    assert details["sha256"] == "b" * 64
    assert_tree_clean(json.dumps(archive), result.stdout, result.stderr)
    assert "s3-credentials.env" not in (json.dumps(archive) + result.stdout + result.stderr)


def test_archive_log_unparseable_leaves_detail_null(box):
    prepare_pass_data(box)
    (box.run / "s3-archive.log").write_text(
        "rclone ERROR: access denied\nanother arbitrary line\n", encoding="utf-8")
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0, result.stderr
    archive = read_evidence(box)["archive"]
    assert archive["status"] == "pass"  # stamp drives the tier; log is provenance
    assert archive["details"]["latest_success"] is None
    assert archive["details"]["sha256"] is None
    assert archive["details"]["utc_day"] == "2026-09-05"


# ── CLI: resources ────────────────────────────────────────────────────────


def test_resources_reported(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0, result.stderr
    resources = read_evidence(box)["resources"]
    assert_envelope(resources, "resources", "pass")
    disk = resources["details"]["disk"]
    assert disk["status"] == "pass"
    assert disk["total_bytes"] > 0
    assert disk["used_bytes"] >= 0
    assert disk["free_bytes"] > 0
    assert 0.0 <= disk["percent"] <= 100.0
    memory = resources["details"]["memory"]
    if os.path.exists("/proc/meminfo"):
        assert memory["mem_total_kb"] > 0
    else:
        assert memory == "unavailable"


# ── CLI: former-authority proof ───────────────────────────────────────────


def test_authority_pass_when_fresh_all_false(box):
    prepare_pass_data(box)
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0, result.stderr
    authority = read_evidence(box)["authority"]
    assert_envelope(authority, "authority", "pass")
    details = authority["details"]
    assert details["host_label"] == "sg01"
    assert details["collected_at"] == "2026-09-05T03:30:00+00:00"
    assert details["tasker_active"] is False
    assert details["tasker_enabled"] is False
    assert details["archive_timer_active"] is False
    assert details["archive_timer_enabled"] is False
    assert details["present"] is True
    for forbidden in (str(box.proof), str(box.root), "former-authority", "proof"):
        assert forbidden not in json.dumps(authority), forbidden


@pytest.mark.parametrize(
    "over",
    [
        {"tasker": {"active": True, "enabled": False}},
        {"tasker": {"active": False, "enabled": True}},
        {"archive_timer": {"active": True, "enabled": False}},
        {"archive_timer": {"active": False, "enabled": True}},
    ],
)
def test_authority_active_or_enabled_fails(box, over):
    prepare_pass_data(box)
    write_proof(box, proof_payload(**over))
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 2, over
    authority = read_evidence(box)["authority"]
    assert authority["status"] == "fail"
    all_false = [authority["details"][key] for key in
                 ("tasker_active", "tasker_enabled", "archive_timer_active", "archive_timer_enabled")]
    assert any(value is True for value in all_false)
    reason = authority["details"]["reason"]
    assert "active" in reason or "enabled" in reason


def test_authority_absent_stale_and_malformed_warn(box):
    api = json.dumps(api_payload(box)).encode()
    # absent
    prepare_pass_data(box)
    box.proof.unlink()
    with http_pair(api_body=api) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0
    authority = read_evidence(box)["authority"]
    assert authority["status"] == "warning"
    assert authority["details"]["present"] is False
    assert "absent" in authority["details"]["reason"]

    # stale (valid, all false, but collected_at older than max age)
    prepare_pass_data(box)
    stale_proof = proof_payload(collected_at="2026-09-04T03:30:00Z")
    write_proof(box, stale_proof)
    with http_pair(api_body=api) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0
    authority = read_evidence(box)["authority"]
    assert authority["status"] == "warning"
    assert "stale" in authority["details"]["reason"]

    # malformed JSON / wrong types / oversized
    for content in ("not json", json.dumps({"host_label": 42}),
                    json.dumps(proof_payload(tasker={"active": "yes", "enabled": 0})),
                    json.dumps({"pad": "x" * 70000})):
        prepare_pass_data(box)
        write_proof(box, content)
        with http_pair(api_body=api) as (api_url, static_url):
            env = full_env(box, api_url, static_url)
            result = run_cli(base_args(), env=env)
        assert result.returncode == 0, content
        assert read_evidence(box)["authority"]["status"] == "warning"


def test_authority_wrong_schema_and_host_warn(box):
    api = json.dumps(api_payload(box)).encode()
    for over, keyword in (
        ({"schema": "portfolio-lab-migration-evidence/v1"}, "schema"),
        ({"host_label": "sg02"}, "host"),
    ):
        prepare_pass_data(box)
        write_proof(box, proof_payload(**over))
        with http_pair(api_body=api) as (api_url, static_url):
            env = full_env(box, api_url, static_url)
            result = run_cli(base_args(), env=env)
        assert result.returncode == 0, over
        authority = read_evidence(box)["authority"]
        assert authority["status"] == "warning", over
        assert keyword in authority["details"]["reason"].lower(), over


def test_authority_expected_host_override(box):
    prepare_pass_data(box)
    write_proof(box, proof_payload(host_label="box1"))
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        env["PLDE_EXPECTED_HOST"] = "box1"
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0, result.stderr
    authority = read_evidence(box)["authority"]
    assert authority["status"] == "pass"
    assert authority["details"]["host_label"] == "box1"


def test_authority_future_dated_warns_and_skew_passes(box):
    api = json.dumps(api_payload(box)).encode()
    # +1 day is beyond the fixed clock-skew allowance: warning, not fail.
    prepare_pass_data(box)
    write_proof(box, proof_payload(collected_at="2026-09-06T03:30:00Z"))
    with http_pair(api_body=api) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0
    authority = read_evidence(box)["authority"]
    assert authority["status"] == "warning"
    assert "future" in authority["details"]["reason"]

    # +60 seconds is inside the small fixed skew allowance: pass.
    prepare_pass_data(box)
    write_proof(box, proof_payload(collected_at="2026-09-05T04:01:00Z"))
    with http_pair(api_body=api) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0
    assert read_evidence(box)["authority"]["status"] == "pass"


def test_authority_mode_symlink_and_ownership_warn(box, monkeypatch):
    api = json.dumps(api_payload(box)).encode()

    # mode must be exactly 0600
    prepare_pass_data(box)
    box.proof.chmod(0o644)
    with http_pair(api_body=api) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 0
    authority = read_evidence(box)["authority"]
    assert authority["status"] == "warning"
    assert "0600" in authority["details"]["reason"]

    # symlink proof: rejected at config time, never followed, target untouched
    prepare_pass_data(box)
    victim = box.run / "victim-proof.json"
    victim.write_text(json.dumps(proof_payload()), encoding="utf-8")
    victim.chmod(0o600)
    box.proof.unlink()
    box.proof.symlink_to(victim)
    with http_pair(api_body=api) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    assert box.proof.is_symlink(), "rejected symlink must be left in place"
    assert victim.read_text(encoding="utf-8") == json.dumps(proof_payload())
    assert "sg01" not in (result.stdout + result.stderr), "symlink target must never be read"

    # ownership mismatch (module-level unit; cannot chown in tests)
    mod = _load_cli_module()
    owner_path = box.run / "owner-proof.json"
    owner_path.write_text(json.dumps(proof_payload()), encoding="utf-8")
    owner_path.chmod(0o600)
    args = mod.parse_args(["--now", NOW, "--timeout", "10",
                           "--freshness-max-age", "3600",
                           "--authority-proof", str(owner_path)])
    cfg = mod.Config(args)
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
    status, details = mod.collect_authority(cfg)
    assert status == "warning"
    assert "own" in details["reason"].lower()


def test_authority_active_or_enabled_still_fails_with_wrong_schema(box):
    """Fail precedence: active/enabled remains fail even when the schema is
    wrong or the file layout is off (warning-only conditions)."""
    prepare_pass_data(box)
    write_proof(box, proof_payload(
        schema="portfolio-lab-migration-evidence/v1",
        tasker={"active": True, "enabled": False},
    ))
    with http_pair(api_body=json.dumps(api_payload(box)).encode()) as (api_url, static_url):
        env = full_env(box, api_url, static_url)
        result = run_cli(base_args(), env=env)
    assert result.returncode == 2
    authority = read_evidence(box)["authority"]
    assert authority["status"] == "fail"
    assert "active" in authority["details"]["reason"]


# ── pure decision functions (module import) ───────────────────────────────


def test_unit_archive_status_tiers():
    mod = _load_cli_module()
    assert mod.archive_status("20260905", "2026-09-05") == "pass"
    assert mod.archive_status("20260904", "2026-09-05") == "warning"
    assert mod.archive_status("20260903", "2026-09-05") == "fail"
    assert mod.archive_status(None, "2026-09-05") == "fail"
    assert mod.archive_status("garbage", "2026-09-05") == "fail"
    assert mod.archive_status("20261301", "2026-09-05") == "fail"
    assert mod.archive_status("20260230", "2026-09-05") == "fail"
    assert mod.archive_status("20260905", "2026-09-06") == "warning"


def test_unit_disk_thresholds():
    mod = _load_cli_module()
    assert mod.evaluate_disk(100 * GIB, 50 * GIB, 50 * GIB) == "pass"
    assert mod.evaluate_disk(1000 * GIB, 920 * GIB, 80 * GIB) == "warning"  # percent >= 90
    assert mod.evaluate_disk(1000 * GIB, 400 * GIB, 10 * GIB) == "warning"  # free < 15 GiB
    assert mod.evaluate_disk(100 * GIB, 50 * GIB, 4 * GIB) == "fail"  # free < 5 GiB
    assert mod.evaluate_disk(100 * GIB, 98 * GIB, 2 * GIB) == "fail"


def test_unit_meminfo_parse():
    mod = _load_cli_module()
    parsed = mod.parse_meminfo("MemTotal:       16384000 kB\nMemAvailable:  8000000 kB\n")
    assert parsed == {"mem_total_kb": 16384000, "mem_available_kb": 8000000}
    assert mod.parse_meminfo("MemTotal: nope\n") is None
    assert mod.parse_meminfo("") is None


def test_unit_overall_status():
    mod = _load_cli_module()
    assert mod.overall_status({
        "tasker": "pass", "jobs": "pass", "freshness": "pass",
        "archive": "pass", "resources": "pass", "authority": "pass",
    }) == ("pass", False)
    assert mod.overall_status({
        "tasker": "pass", "jobs": "pass", "freshness": "warning",
        "archive": "pass", "resources": "pass", "authority": "pass",
    }) == ("warning", False)
    assert mod.overall_status({
        "tasker": "pass", "jobs": "fail", "freshness": "warning",
        "archive": "pass", "resources": "pass", "authority": "pass",
    }) == ("fail", True)