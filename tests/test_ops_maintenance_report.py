"""Tests for S3 ops-maintenance reporter (last-<cycle>.json + fail-only capture)."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PY = PROJECT_ROOT / "ops" / "maintenance" / "report.py"
RUN_SCRIPT = PROJECT_ROOT / "ops" / "maintenance" / "run.sh"

GROK_401 = (
    '{"type":"error","message":"Internal error: \\"Unauthorized (401) from '
    "https://cli-chat-proxy.grok.com/v1/responses: Invalid or expired credentials "
    '(auth_kind=none, x_xai_token_auth=xai-grok-cli)\\""}\n'
    "Error: Internal error: Unauthorized (401) auth_kind=none "
    "Authorization: Bearer super-secret-token-value\n"
)

OK_REPORT = {
    "host_health": {
        "status": "ok",
        "loadavg": "0.10, 0.20, 0.30",
        "memory_free_mb": 2048,
        "disk_usage_pct": 40,
    },
    "checks": [{"name": "api", "status": "ok", "evidence": "HTTP 200"}],
    "drift": {"repos": []},
    "actions": [],
    "escalations": [],
}


def load_report_mod():
    spec = importlib.util.spec_from_file_location("ops_maint_report", REPORT_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def report_mod():
    return load_report_mod()


@pytest.fixture(autouse=True)
def isolate_capture_env(monkeypatch):
    # The parent session may have live MCP tokens; never let tests inherit them.
    monkeypatch.delenv("SKILLWIKI_MCP_TOKEN", raising=False)
    monkeypatch.delenv("OPS_MAINT_CAPTURE_TOKEN", raising=False)
    monkeypatch.delenv("OPS_MAINT_CAPTURE_URL", raising=False)
    monkeypatch.delenv("SKILLWIKI_MCP_URL", raising=False)
    monkeypatch.delenv("OPS_MAINT_DISABLE_CAPTURE", raising=False)


def _bash4() -> str:
    candidates = [
        "/opt/homebrew/bin/bash",
        "/usr/local/bin/bash",
        shutil.which("bash") or "",
    ]
    for cand in candidates:
        if not cand or not Path(cand).is_file():
            continue
        probe = subprocess.run(
            [cand, "-c", "printf '%s' \"$BASH_VERSINFO\""],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            if int(probe.stdout.strip() or "0") >= 4:
                return cand
        except ValueError:
            continue
    pytest.skip("bash 4+ required for cron_guard fd redirect")


def _clean_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    for key in (
        "SKILLWIKI_MCP_TOKEN",
        "OPS_MAINT_CAPTURE_TOKEN",
        "OPS_MAINT_CAPTURE_URL",
        "SKILLWIKI_MCP_URL",
        "OPS_MAINT_DISABLE_CAPTURE",
    ):
        env.pop(key, None)
    env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
    return env


def test_extract_report_from_wrapped_result(report_mod):
    wrapped = json.dumps({"type": "result", "result": json.dumps(OK_REPORT)})
    extracted = report_mod.extract_report(wrapped)
    assert extracted is not None
    assert extracted["host_health"]["status"] == "ok"


def test_synthesize_fail_from_grok_401(report_mod, tmp_path):
    transcript = tmp_path / "run-daily-401.json"
    transcript.write_text(GROK_401, encoding="utf-8")
    summary = report_mod.write_and_capture(
        cycle="daily",
        transcript_path=transcript,
        output_dir=tmp_path,
        agent_exit=1,
        capture=True,
    )
    assert summary["overall"] == "fail"
    assert summary["capture"] == "skipped"
    assert summary["capture_reason"] == "missing_token"
    last = json.loads((tmp_path / "last-daily.json").read_text(encoding="utf-8"))
    assert last["host_health"]["status"] == "fail"
    assert last["checks"][0]["name"] == "grok-cli-auth"
    assert last["escalations"][0]["urgency"] == "high"
    blob = json.dumps(last)
    assert "super-secret-token-value" not in blob
    assert "[REDACTED:credential]" in last["checks"][0]["evidence"]
    assert (tmp_path / "last-daily.json").stat().st_mode & 0o777 == 0o600


def test_ok_report_is_silent_no_capture(report_mod, tmp_path, monkeypatch):
    called = {"n": 0}

    def boom(**kwargs):
        called["n"] += 1
        raise AssertionError("capture must not run on ok")

    transcript = tmp_path / "run-daily-ok.json"
    transcript.write_text(json.dumps(OK_REPORT), encoding="utf-8")
    monkeypatch.setenv("OPS_MAINT_CAPTURE_TOKEN", "dummy-token")
    summary = report_mod.write_and_capture(
        cycle="daily",
        transcript_path=transcript,
        output_dir=tmp_path,
        agent_exit=0,
        capture=True,
        capture_fn=boom,
    )
    assert summary["overall"] == "ok"
    assert summary["capture"] == "skipped"
    assert called["n"] == 0
    last = json.loads((tmp_path / "last-daily.json").read_text(encoding="utf-8"))
    assert last["actions"] == []


def test_fail_report_posts_escalation_title(report_mod, tmp_path, monkeypatch):
    posted: dict[str, Any] = {}

    def fake_capture(**kwargs):
        posted.update(kwargs)
        return {"status": "ok", "path": "raw/transcripts/2026-10-01-note-escalation.md"}

    monkeypatch.setenv("OPS_MAINT_CAPTURE_TOKEN", "dummy-token")

    fail_report = {
        "host_health": {"status": "ok", "loadavg": "", "memory_free_mb": 1, "disk_usage_pct": 1},
        "checks": [{"name": "broker", "status": "fail", "evidence": "HTTP 000"}],
        "drift": {"repos": []},
        "actions": [],
        "escalations": [
            {
                "issue": "IBKR TWS API refused — attended TWS re-login needed",
                "urgency": "high",
                "recommended_action": "Re-login TWS on the operator desktop.",
            }
        ],
    }
    transcript = tmp_path / "run-daily-fail.json"
    transcript.write_text(json.dumps(fail_report), encoding="utf-8")
    summary = report_mod.write_and_capture(
        cycle="daily",
        transcript_path=transcript,
        output_dir=tmp_path,
        agent_exit=1,
        capture=True,
        capture_fn=fake_capture,
    )
    assert summary["overall"] == "fail"
    assert summary["capture"] == "ok"
    assert summary["capture_path"].endswith("note-escalation.md")
    assert posted["kind"] == "note"
    assert posted["project"] == "portfolio-lab"
    assert posted["title"].startswith("ESCALATION: ops-maint-daily:")
    assert "IBKR TWS API refused" in posted["title"]
    last = json.loads((tmp_path / "last-daily.json").read_text(encoding="utf-8"))
    assert last["actions"][-1]["action"] == "fail-only wiki_capture"
    assert last["actions"][-1]["result"] == "ok"


def test_cli_exit_codes(tmp_path):
    env = _clean_env()
    ok_path = tmp_path / "ok.json"
    ok_path.write_text(json.dumps(OK_REPORT), encoding="utf-8")
    ok = subprocess.run(
        [
            sys.executable,
            str(REPORT_PY),
            "--cycle",
            "daily",
            "--transcript",
            str(ok_path),
            "--output-dir",
            str(tmp_path / "ok-out"),
            "--agent-exit",
            "0",
            "--no-capture",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr
    fail_path = tmp_path / "fail.json"
    fail_path.write_text(GROK_401, encoding="utf-8")
    fail = subprocess.run(
        [
            sys.executable,
            str(REPORT_PY),
            "--cycle",
            "daily",
            "--transcript",
            str(fail_path),
            "--output-dir",
            str(tmp_path / "fail-out"),
            "--agent-exit",
            "1",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert fail.returncode == 1, fail.stderr
    summary = json.loads(fail.stdout)
    assert summary["overall"] == "fail"
    assert "super-secret-token-value" not in fail.stdout
    assert "super-secret-token-value" not in fail.stderr


def test_wiki_capture_jsonrpc_against_local_server(report_mod, tmp_path, monkeypatch):
    received: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            received.append(payload)
            auth = self.headers.get("Authorization") or ""
            assert auth == "Bearer test-token"
            method = payload.get("method")
            if method == "initialize":
                body = json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": payload.get("id"),
                        "result": {"protocolVersion": "2025-03-26", "capabilities": {}},
                    }
                ).encode("utf-8")
            elif method == "notifications/initialized":
                self.send_response(202)
                self.end_headers()
                return
            elif method == "tools/call":
                assert payload["params"]["name"] == "wiki_capture"
                args = payload["params"]["arguments"]
                assert args["title"].startswith("ESCALATION:")
                body = json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": payload.get("id"),
                        "result": {
                            "path": "raw/transcripts/2026-10-01-note-escalation.md"
                        },
                    }
                ).encode("utf-8")
            else:
                self.send_response(400)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        url = f"http://127.0.0.1:{port}/mcp"
        transcript = tmp_path / "run-daily-401.json"
        transcript.write_text(GROK_401, encoding="utf-8")
        monkeypatch.setenv("OPS_MAINT_CAPTURE_TOKEN", "test-token")
        monkeypatch.setenv("OPS_MAINT_CAPTURE_URL", url)
        summary = report_mod.write_and_capture(
            cycle="daily",
            transcript_path=transcript,
            output_dir=tmp_path,
            agent_exit=1,
            capture=True,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert summary["capture"] == "ok"
    methods = [item.get("method") for item in received]
    assert "initialize" in methods
    assert "tools/call" in methods


def test_run_sh_dry_run_mentions_reporter(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_grok = bin_dir / "grok"
    fake_grok.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_grok.chmod(0o755)
    env = _clean_env()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["OPS_SKIP_PATH_PREPEND"] = "1"
    res = subprocess.run(
        [str(RUN_SCRIPT), "daily", "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "last-daily.json" in res.stdout
    assert "report.py" in res.stdout


def test_run_sh_401_writes_last_daily_json(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_grok = bin_dir / "grok"
    fake_grok.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' "
        "'{\"type\":\"error\",\"message\":\"Unauthorized (401) auth_kind=none "
        "Authorization: Bearer leaked-token\"}'\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_grok.chmod(0o755)
    data_dir = tmp_path / "ops-data"
    lock_dir = tmp_path / "locks"
    data_dir.mkdir()
    lock_dir.mkdir()
    env = _clean_env()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["OPS_SKIP_PATH_PREPEND"] = "1"
    env["OPS_MAINT_DATA_DIR"] = str(data_dir)
    env["CRON_GUARD_LOCK_DIR"] = str(lock_dir)
    env["PORTFOLIO_LAB_PROJECT_DIR"] = str(PROJECT_ROOT)
    env["OPS_MAINT_PYTHON"] = sys.executable
    res = subprocess.run(
        [_bash4(), str(RUN_SCRIPT), "daily"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=30,
    )
    assert res.returncode != 0
    last = data_dir / "last-daily.json"
    assert last.is_file(), res.stdout + res.stderr
    payload = json.loads(last.read_text(encoding="utf-8"))
    assert payload["host_health"]["status"] == "fail"
    blob = last.read_text(encoding="utf-8") + res.stdout + res.stderr
    assert "leaked-token" not in blob
    transcripts = list(data_dir.glob("run-daily-*.json"))
    assert transcripts
    assert last.stat().st_mode & stat.S_IRWXU == 0o600
