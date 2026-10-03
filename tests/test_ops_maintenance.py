"""Tests for S1 maintenance playbook entrypoint and runbooks."""

from __future__ import annotations

import os
import socket
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "ops" / "maintenance" / "run.sh"
CHECK_TASKER_HTTP = PROJECT_ROOT / "ops" / "maintenance" / "check-tasker-http.sh"
DAILY_RUNBOOK = PROJECT_ROOT / "ops" / "maintenance" / "daily.md"
WEEKLY_RUNBOOK = PROJECT_ROOT / "ops" / "maintenance" / "weekly.md"

TASKER_STATUS_OK = (
    b'{"backend":"tasker","service":"portfolio-lab-tasker",'
    b'"tasks":{},"timestamp":"2026-10-03T00:00:00Z"}'
)


def test_daily_runbook_must_not_rails_and_contracts():
    assert DAILY_RUNBOOK.is_file(), f"Missing {DAILY_RUNBOOK}"
    content = DAILY_RUNBOOK.read_text(encoding="utf-8")

    # MUST NOT rail checks
    assert "MUST NOT touch" in content
    assert "/home/box/.local/share/portfolio-lab/app" in content
    assert "MUST NOT call broker trade APIs or invoke `unlock_trade`" in content
    assert "MUST NOT drop stashes or delete branches without" in content
    assert "MUST NOT execute `pkill -f` or `killall` patterns" in content
    assert "MUST NOT start any second Tasker scheduler" in content
    assert "PORTFOLIO_LAB_ENABLE_ML" in content
    assert "must stay `0`" in content or "stays 0" in content

    # Autofix class registry and gate
    assert "Autofix Classes" in content
    assert "OPS_MAINT_AUTOFIX" in content
    assert "log-hygiene" in content
    assert "stale-lock-cleanup" in content
    assert "artifact-regen" in content

    # Output contract checks
    assert "data/ops-maintenance/last-daily.json" in content
    assert "host_health" in content
    assert "checks" in content
    assert "drift" in content
    assert "actions" in content
    assert "escalations" in content
    assert "ESCALATION:" in content
    assert "wiki_capture" in content


def test_weekly_runbook_must_not_rails_and_contracts():
    assert WEEKLY_RUNBOOK.is_file(), f"Missing {WEEKLY_RUNBOOK}"
    content = WEEKLY_RUNBOOK.read_text(encoding="utf-8")

    # MUST NOT rail checks
    assert "MUST NOT touch" in content
    assert "/home/box/.local/share/portfolio-lab/app" in content
    assert "MUST NOT call broker trade APIs or invoke `unlock_trade`" in content
    assert "MUST NOT drop stashes or delete branches without" in content
    assert "MUST NOT execute `pkill -f` or `killall` patterns" in content
    assert "MUST NOT start any second Tasker scheduler" in content
    assert "PORTFOLIO_LAB_ENABLE_ML" in content
    assert "archive/..." in content or "archive/" in content

    # Deep weekly audits
    assert "uv pip list --outdated" in content
    assert "make test-gate" in content
    assert "grok plugin update" in content

    # Autofix class registry and gate
    assert "Autofix Classes" in content
    assert "OPS_MAINT_AUTOFIX" in content
    assert "plugin-cache-refresh" in content
    assert "artifact-regen" in content

    # Output contract checks
    assert "data/ops-maintenance/last-weekly.json" in content
    assert "host_health" in content
    assert "checks" in content
    assert "drift" in content
    assert "actions" in content
    assert "escalations" in content
    assert "ESCALATION:" in content
    assert "wiki_capture" in content


def test_run_script_syntax():
    assert RUN_SCRIPT.is_file(), f"Missing {RUN_SCRIPT}"
    assert os.access(RUN_SCRIPT, os.X_OK), f"{RUN_SCRIPT} is not executable"

    res = subprocess.run(
        ["bash", "-n", str(RUN_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"bash -n failed: {res.stderr}"


def test_run_script_unknown_cycle_exits_nonzero():
    res = subprocess.run(
        [str(RUN_SCRIPT), "monthly"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0
    assert "Usage:" in res.stderr
    assert "daily|weekly" in res.stderr


def test_run_script_no_args_exits_nonzero():
    res = subprocess.run(
        [str(RUN_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0
    assert "Usage:" in res.stderr


def test_run_script_dry_run_daily(tmp_path):
    # Ensure a dummy grok binary exists if not already present
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_grok = bin_dir / "grok"
    fake_grok.write_text("#!/bin/sh\nexit 0\n")
    fake_grok.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"

    res = subprocess.run(
        [str(RUN_SCRIPT), "daily", "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert res.returncode == 0, f"run.sh failed: {res.stderr}"
    assert "--agent minimal" in res.stdout
    assert "--model flash-max" in res.stdout
    assert str(DAILY_RUNBOOK) in res.stdout
    assert "ops-maint-daily" in res.stdout
    assert "last-daily.json" in res.stdout
    assert "report.py" in res.stdout


def test_run_script_dry_run_weekly(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_grok = bin_dir / "grok"
    fake_grok.write_text("#!/bin/sh\nexit 0\n")
    fake_grok.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"

    res = subprocess.run(
        [str(RUN_SCRIPT), "weekly", "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert res.returncode == 0, f"run.sh failed: {res.stderr}"
    assert "--agent minimal" in res.stdout
    assert "--model flash-max" in res.stdout
    assert str(WEEKLY_RUNBOOK) in res.stdout
    assert "ops-maint-weekly" in res.stdout
    assert "last-weekly.json" in res.stdout
    assert "report.py" in res.stdout


def test_run_script_fails_closed_when_grok_absent(tmp_path):
    empty_bin = tmp_path / "empty_bin"
    empty_bin.mkdir()

    env = dict(os.environ)
    env["PATH"] = str(empty_bin)
    env["OPS_SKIP_PATH_PREPEND"] = "1"

    res = subprocess.run(
        [str(RUN_SCRIPT), "daily", "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert res.returncode != 0
    assert "grok" in res.stderr
    assert "not found" in res.stderr


def test_run_script_fails_closed_when_runbook_missing(tmp_path):
    empty_ops = tmp_path / "empty_ops"
    empty_ops.mkdir()

    # Supply dummy grok so grok check passes
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_grok = bin_dir / "grok"
    fake_grok.write_text("#!/bin/sh\nexit 0\n")
    fake_grok.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["OPS_MAINT_DIR"] = str(empty_ops)

    res = subprocess.run(
        [str(RUN_SCRIPT), "daily", "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert res.returncode != 0
    assert "Runbook file not found" in res.stderr
    assert str(empty_ops / "daily.md") in res.stderr


def test_run_script_dry_run_reports_env_file_status(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_grok = bin_dir / "grok"
    fake_grok.write_text("#!/bin/sh\nexit 0\n")
    fake_grok.chmod(0o755)

    env_file = tmp_path / "ops-maintenance.env"
    env_file.write_text("SKILLWIKI_MCP_TOKEN=dummy-test-token\n")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["OPS_MAINT_ENV_FILE"] = str(env_file)

    res = subprocess.run(
        [str(RUN_SCRIPT), "daily", "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert res.returncode == 0, f"run.sh failed: {res.stderr}"
    assert f"Escalation Env: present ({env_file})" in res.stdout
    # The token value must never be printed
    assert "dummy-test-token" not in res.stdout
    assert "dummy-test-token" not in res.stderr

    env_absent = dict(env)
    env_absent["OPS_MAINT_ENV_FILE"] = str(tmp_path / "missing.env")
    res_absent = subprocess.run(
        [str(RUN_SCRIPT), "daily", "--dry-run"],
        capture_output=True,
        text=True,
        env=env_absent,
        check=False,
    )
    assert res_absent.returncode == 0, f"run.sh failed: {res_absent.stderr}"
    assert "Escalation Env: absent" in res_absent.stdout


def test_daily_runbook_tasker_status_probe_contract():
    content = DAILY_RUNBOOK.read_text(encoding="utf-8")
    assert "/api/tasker/status" in content
    assert '"backend":"tasker"' in content
    assert "portfolio-lab-tasker" in content
    assert "observational" in content.lower()
    assert "NOT fail" in content
    assert "MUST NOT restart prod app" in content
    assert "/home/box/.local/share/portfolio-lab/app" in content
    assert "http://127.0.0.1:8001/" in content
    assert "Must be HTTP 200" in content
    # Stale canary: / on :8000 is observational; it is not the fail criterion.
    assert "http://127.0.0.1:8000/` -> Must be HTTP 200" not in content
    assert "http://127.0.0.1:8000/ -> Must be HTTP 200" not in content


def test_weekly_runbook_tasker_status_probe_contract():
    content = WEEKLY_RUNBOOK.read_text(encoding="utf-8")
    assert "/api/tasker/status" in content
    assert '"backend":"tasker"' in content or "backend" in content
    assert "observational" in content.lower()
    assert "NOT fail" in content
    assert ":8001/" in content
    assert "App HTTP 200 on `:8000`" not in content


def test_check_tasker_http_script_syntax_and_executable():
    assert CHECK_TASKER_HTTP.is_file(), f"Missing {CHECK_TASKER_HTTP}"
    assert os.access(CHECK_TASKER_HTTP, os.X_OK), f"{CHECK_TASKER_HTTP} is not executable"
    res = subprocess.run(
        ["sh", "-n", str(CHECK_TASKER_HTTP)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"sh -n failed: {res.stderr}"


@contextmanager
def _http_routes(routes: dict[str, tuple[int, bytes, str]]):
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            status, body, ctype = type(self).routes.get(
                self.path, (404, b"", "text/plain")
            )
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    _Handler.routes = routes
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_check(tasker_base: str, static_base: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["TASKER_HTTP_BASE"] = tasker_base
    env["STATIC_HTTP_BASE"] = static_base
    return subprocess.run(
        [str(CHECK_TASKER_HTTP)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=20,
    )


def test_check_tasker_http_ok_when_root_is_404():
    with _http_routes(
        {
            "/api/tasker/status": (200, TASKER_STATUS_OK, "application/json"),
            "/": (404, b"Not Found", "text/plain"),
        }
    ) as tasker_base, _http_routes(
        {"/": (200, b"ok", "text/html")}
    ) as static_base:
        res = _run_check(tasker_base, static_base)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "tasker-http: ok" in res.stdout
    assert "path=/api/tasker/status http=200 class=ok-tasker-status" in res.stdout
    assert "path=/ http=404 class=observe-root" in res.stdout
    assert "path=/ http=200 class=ok-static" in res.stdout
    assert "fail-tasker-status" not in res.stdout
    assert TASKER_STATUS_OK.decode() not in res.stdout
    assert TASKER_STATUS_OK.decode() not in res.stderr


def test_check_tasker_http_fails_when_status_not_200():
    with _http_routes(
        {
            "/api/tasker/status": (500, b"{}", "application/json"),
            "/": (200, b"ok", "text/html"),
        }
    ) as tasker_base, _http_routes(
        {"/": (200, b"ok", "text/html")}
    ) as static_base:
        res = _run_check(tasker_base, static_base)
    assert res.returncode != 0
    assert "tasker-http: fail" in res.stdout
    assert "class=fail-tasker-status" in res.stdout
    assert "class=observe-root" in res.stdout
    assert "class=ok-static" in res.stdout


def test_check_tasker_http_fails_when_status_json_lacks_identity():
    with _http_routes(
        {
            "/api/tasker/status": (
                200,
                b'{"backend":"other","service":"nope"}',
                "application/json",
            ),
            "/": (404, b"missing", "text/plain"),
        }
    ) as tasker_base, _http_routes(
        {"/": (200, b"ok", "text/html")}
    ) as static_base:
        res = _run_check(tasker_base, static_base)
    assert res.returncode != 0
    assert "tasker-http: fail" in res.stdout
    assert "http=200 class=fail-tasker-status" in res.stdout


def test_check_tasker_http_fails_when_static_not_200():
    with _http_routes(
        {
            "/api/tasker/status": (200, TASKER_STATUS_OK, "application/json"),
            "/": (404, b"missing", "text/plain"),
        }
    ) as tasker_base, _http_routes(
        {"/": (500, b"err", "text/plain")}
    ) as static_base:
        res = _run_check(tasker_base, static_base)
    assert res.returncode != 0
    assert "tasker-http: fail" in res.stdout
    assert "class=ok-tasker-status" in res.stdout
    assert "class=fail-static" in res.stdout
    assert "class=observe-root" in res.stdout


def test_check_tasker_http_fails_when_tasker_refused():
    refused = f"http://127.0.0.1:{_free_port()}"
    with _http_routes({"/": (200, b"ok", "text/html")}) as static_base:
        res = _run_check(refused, static_base)
    assert res.returncode != 0
    assert "tasker-http: fail" in res.stdout
    assert "http=000 class=fail-tasker-status" in res.stdout
    assert "class=observe-root" in res.stdout
    assert TASKER_STATUS_OK.decode() not in res.stdout
    assert TASKER_STATUS_OK.decode() not in res.stderr
