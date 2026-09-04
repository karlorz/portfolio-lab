#!/home/box/.local/bin/python3
"""Daily operational evidence collector for the live cursor-box Portfolio Lab.

A bounded, read-only collector (Task 2 of the sg01 -> cursor-box cutover):
observes the production lab each UTC day and writes one compact evidence
directory without mutating any service. The migration evidence schema
(``portfolio-lab-migration-evidence/v1``) is comparison-oriented and requires
unrelated recovery fields; this collector uses its own lightweight schema
``portfolio-lab-daily-evidence/v1``.

Read-only guarantees
    * Never starts/stops/ensures services, never touches cron/systemd, never
      invokes a shell (controllers and HTTP servers are probed with explicit
      argv, bounded timeouts, and capped stdout/stderr reads; uncooperative
      children are killed and reaped).
    * Never sources or echoes credentials; evidencing retains only compact
      summaries, ages/sizes, a UTC day, one timestamp and one 64-hex SHA-256.
      Failure reasons are static and never echo captured output.
    * Bounded: every HTTP/log/response read has a byte cap and every evidence
      file has a conservative per-file serialized size cap enforced before
      any directory is created.

Output
    OUTPUT_ROOT/YYYY-MM-DD/{tasker,jobs,freshness,archive,resources,authority,
    summary}.json  (one bounded directory; dirs 0700, files 0600; each file
    is ``{schema, category, collected_at, status, details}`` with
    status in pass|warning|fail). Same-day reruns replace files
    deterministically (atomic per-file replace; observation-derived files are
    byte-identical under the same inputs, resources.json is a live snapshot).
    summary.json carries per-category statuses, the overall status
    (fail if any fail, else warning if any warning, else pass) and
    ``notify_grok_bot`` = true only on fail.

Exit codes
    0  overall pass or warning
    2  overall fail (evidence is still written)
    1  invalid CLI/config, unsafe path placement, or write/size failure

stdout/stderr
    Exactly one compact, secret-free JSON summary line on stdout; static
    diagnostics on stderr never echo secret-bearing input values.

Overrides (test-safe; flag > env > production default)
    --root                PLDE_ROOT               /home/box/.local/share/portfolio-lab
    --output-root         PLDE_OUTPUT_ROOT        <root>/evidence
    --tasker-controller   PLDE_TASKER_CONTROLLER  /home/box/.local/bin/portfolio-lab-box-persist
    --static-controller   PLDE_STATIC_CONTROLLER  /home/box/.local/bin/portfolio-lab-static-persist
    --api-url             PLDE_API_URL            http://127.0.0.1:8000/api/tasker/status
    --static-url          PLDE_STATIC_URL         http://127.0.0.1:8001/
    --authority-proof     PLDE_AUTHORITY_PROOF    <root>/run/former-authority-proof.json
    --expected-host       PLDE_EXPECTED_HOST      sg01
    --timeout             PLDE_TIMEOUT            10.0        (seconds, per check)
    --freshness-max-age   PLDE_FRESHNESS_MAX_AGE  21600       (seconds)
    --now                 PLDE_NOW                (actual time; ISO-8601, any zone -> UTC)

All path overrides must be absolute. Relative paths, symlinked controllers,
an output root equal to the lab root (or inside app/www/run/data), non-http
or non-loopback URLs, and non-positive timeouts are rejected as config
errors. ``--now`` pins every time decision (UTC day of the evidence
directory, freshness ages, archive tiering, proof freshness) so reruns are
deterministic.

Archive evidence semantics (deterministic under ``--now``)
    s3-archive-last-utc-day (run dir, ``%Y%m%d``) drives the tier: current
    UTC day pass, previous day warning (the expected pre-window state for a
    once-per-day archive), older or missing fail. s3-archive.log is parsed
    for the latest ``published ... sha256=`` success only; evidence retains
    just the UTC day, the latest success timestamp when safely parsed, and
    the 64-hex SHA-256. Object keys, s3 URLs, archive paths, credentials and
    arbitrary log lines are never retained.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import http.client
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "portfolio-lab-daily-evidence/v1"
BOX_PERSIST_SCHEMA = "portfolio-lab-box-persist/v1"
STATIC_PERSIST_SCHEMA = "portfolio-lab-static-persist/v1"
PROOF_SCHEMA = "portfolio-lab-former-authority-proof/v1"
DEFAULT_EXPECTED_HOST = "sg01"
PROOF_CLOCK_SKEW_ALLOWANCE = 300.0  # fixed small skew allowance (seconds)

DEFAULT_ROOT = Path("/home/box/.local/share/portfolio-lab")
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "evidence"
DEFAULT_TASKER_CONTROLLER = "/home/box/.local/bin/portfolio-lab-box-persist"
DEFAULT_STATIC_CONTROLLER = "/home/box/.local/bin/portfolio-lab-static-persist"
DEFAULT_API_URL = "http://127.0.0.1:8000/api/tasker/status"
DEFAULT_STATIC_URL = "http://127.0.0.1:8001/"
DEFAULT_TIMEOUT = 10.0
DEFAULT_FRESHNESS_MAX_AGE = 21600
DEFAULT_AUTHORITY_PROOF_REL = "run/former-authority-proof.json"

TASKER_SERVICE = "portfolio-lab-tasker"
STATIC_SERVICE = "portfolio-lab-static"
ARCHIVE_STAMP_REL = "run/s3-archive-last-utc-day"
ARCHIVE_LOG_REL = "run/s3-archive.log"
FRESHNESS_FILES = (
    "app/data/signals.json",
    "app/data/tasker_status.json",
    "www/data/signals.json",
    "www/data/tasker_status.json",
)

MAX_RESPONSE_BYTES = 1048576  # per HTTP body
MAX_LOG_BYTES = 1048576  # per archive log read
MAX_PROOF_BYTES = 65536  # per former-authority proof read
MAX_FILE_BYTES = 262144  # conservative per-evidence-file serialized cap
MAX_CONTROLLER_BYTES = 65536  # per controller stdout/stderr stream
MAX_STAMP_BYTES = 64  # archive stamp file read bound
RUNS_CAP = 50
GIB = 1024**3
DISK_WARNING_PERCENT = 90.0
DISK_WARNING_FREE = 15 * GIB
DISK_FAIL_FREE = 5 * GIB
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
STAMP_RE = re.compile(r"^\d{8}$")
ARCHIVE_LOG_LINE_RE = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\][^\n]*"
    r"published[^\n]*sha256=([0-9a-f]{64})",
    re.MULTILINE,
)

# Compact retained keys only: no commands, env, logs, paths, or free text.
TASK_KEYS = (
    "id", "label", "schedule", "enabled", "paused", "last_status",
    "last_run_id", "last_finished_at", "last_duration_seconds",
    "failure_count", "consecutive_failures",
)
RUN_KEYS = (
    "run_id", "task_id", "trigger", "status", "started_at",
    "finished_at", "duration_seconds", "exit_code",
)

_ENV_PREFIX = "PLDE_"


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


# ── config ────────────────────────────────────────────────────────────────


def _opt(args_value: str | None, env_name: str, default: str) -> str:
    if args_value is not None:
        return args_value
    raw = os.environ.get(f"{_ENV_PREFIX}{env_name}")
    if raw is not None and raw.strip() != "":
        return raw
    return default


def _abs(path_raw: str, what: str) -> Path:
    if not path_raw:
        die(f"PLDE_{what} must be an absolute path")
    path = Path(path_raw)
    if not path.is_absolute():
        die(f"PLDE_{what} must be an absolute path; got a relative path")
    if path.is_symlink():
        die(f"PLDE_{what} must not be a symlink")
    return path


def _positive_float(raw: str, what: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        die(f"PLDE_{what} must be a numeric value in seconds")
    if not math.isfinite(value):
        die(f"PLDE_{what} must be a finite numeric value in seconds")
    if value <= 0:
        die(f"PLDE_{what} must be positive")
    return value


def _parse_now(raw: str) -> datetime:
    text = raw.strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        die("PLDE_NOW must be an ISO-8601 timestamp with a UTC offset or Z")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _loopback_url(raw: str, what: str) -> str:
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme != "http":
        die(f"PLDE_{what} must be an http loopback URL")
    if parsed.hostname not in LOOPBACK_HOSTS:
        die(f"PLDE_{what} must target a loopback host")
    if parsed.username is not None or parsed.password is not None:
        die(f"PLDE_{what} must not carry userinfo")
    if parsed.query or parsed.fragment:
        die(f"PLDE_{what} must not carry a query or fragment")
    return raw


class Config:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = _abs(_opt(args.root, "ROOT", str(DEFAULT_ROOT)), "ROOT")
        self.app = self.root.resolve() / "app"
        self.www = self.root.resolve() / "www"
        self.output_root = _abs(
            _opt(args.output_root, "OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)), "OUTPUT_ROOT"
        )
        self.tasker_controller = _abs(
            _opt(args.tasker_controller, "TASKER_CONTROLLER", DEFAULT_TASKER_CONTROLLER),
            "TASKER_CONTROLLER",
        )
        self.static_controller = _abs(
            _opt(args.static_controller, "STATIC_CONTROLLER", DEFAULT_STATIC_CONTROLLER),
            "STATIC_CONTROLLER",
        )
        self.api_url = _loopback_url(
            _opt(args.api_url, "API_URL", DEFAULT_API_URL), "API_URL"
        )
        self.static_url = _loopback_url(
            _opt(args.static_url, "STATIC_URL", DEFAULT_STATIC_URL), "STATIC_URL"
        )
        self.authority_proof = _abs(
            _opt(
                args.authority_proof,
                "AUTHORITY_PROOF",
                str(self.root / DEFAULT_AUTHORITY_PROOF_REL),
            ),
            "AUTHORITY_PROOF",
        )
        self.expected_host = _opt(
            args.expected_host, "EXPECTED_HOST", DEFAULT_EXPECTED_HOST
        )
        self.timeout = _positive_float(_opt(args.timeout, "TIMEOUT", str(DEFAULT_TIMEOUT)), "TIMEOUT")
        self.freshness_max_age = _positive_float(
            _opt(args.freshness_max_age, "FRESHNESS_MAX_AGE", str(DEFAULT_FRESHNESS_MAX_AGE)),
            "FRESHNESS_MAX_AGE",
        )
        self.now = _parse_now(_opt(args.now, "NOW", datetime.now(timezone.utc).isoformat()))
        self.utc_day = self.now.strftime("%Y-%m-%d")

    def validate(self) -> None:
        if not self.root.is_dir():
            die("PLDE_ROOT must be an existing directory")
        out = self.output_root.resolve()
        if out == Path("/") or out == self.root.resolve():
            die("PLDE_OUTPUT_ROOT must not be the filesystem root or the lab root")
        for guarded in (self.app, self.www, self.root / "run", self.root / "data"):
            if out == guarded or out.is_relative_to(guarded):
                die("PLDE_OUTPUT_ROOT must sit outside app/www/run/data")


class _ArgumentParser(argparse.ArgumentParser):
    """argparse exits 2 on usage errors by default; the collector contract
    is exit 1 for invalid CLI usage (stdout stays empty)."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _ArgumentParser(
        prog="portfolio_lab_daily_evidence.py",
        description="Read-only daily operational evidence for cursor-box Portfolio Lab.",
    )
    parser.add_argument("--root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--tasker-controller", default=None)
    parser.add_argument("--static-controller", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--static-url", default=None)
    parser.add_argument("--authority-proof", default=None)
    parser.add_argument("--expected-host", default=None)
    parser.add_argument("--timeout", default=None)
    parser.add_argument("--freshness-max-age", default=None)
    parser.add_argument("--now", default=None)
    return parser.parse_args(argv)


# ── pure decision helpers (unit-tested) ───────────────────────────────────


def archive_status(stamp_day: str | None, now_day: str) -> str:
    """Tier archive evidence: current UTC day pass, previous day warning,
    older/missing/malformed fail."""
    if stamp_day is None:
        return "fail"
    try:
        stamp = datetime.strptime(stamp_day, "%Y%m%d").date()
        today = datetime.strptime(now_day, "%Y-%m-%d").date()
    except ValueError:
        return "fail"
    delta = (today - stamp).days
    if delta == 0:
        return "pass"
    if delta == 1:
        return "warning"
    return "fail"


def evaluate_disk(total_bytes: int, used_bytes: int, free_bytes: int) -> str:
    """Disk tier: fail only when free < 5 GiB; warning when percent >= 90 or
    free < 15 GiB; otherwise pass."""
    if free_bytes < DISK_FAIL_FREE:
        return "fail"
    if total_bytes > 0 and used_bytes / total_bytes >= DISK_WARNING_PERCENT / 100.0:
        return "warning"
    if free_bytes < DISK_WARNING_FREE:
        return "warning"
    return "pass"


def parse_meminfo(text: str) -> dict[str, int] | None:
    """Parse /proc/meminfo totals; None when MemTotal is absent/unparseable."""
    result: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("MemTotal:", "MemAvailable:"):
            key = "mem_total_kb" if parts[0] == "MemTotal:" else "mem_available_kb"
            try:
                result[key] = int(parts[1])
            except ValueError:
                return None
    if "mem_total_kb" not in result:
        return None
    result.setdefault("mem_available_kb", 0)
    return result


def overall_status(categories: dict[str, str]) -> tuple[str, bool]:
    """fail if any fail, else warning if any warning, else pass; notify only
    on fail."""
    if any(status == "fail" for status in categories.values()):
        return "fail", True
    if any(status == "warning" for status in categories.values()):
        return "warning", False
    return "pass", False


# ── bounded probes ────────────────────────────────────────────────────────


def run_controller(controller: Path, argv: list[str], timeout: float) -> dict[str, Any]:
    """Probe one controller with bounded stdout/stderr reads and a hard
    deadline. On timeout or bound exceed the child is SIGKILLed and reaped.
    Failure reasons are static and never echo controller output.

    Pipes are non-blocking: EOF after the child closes a pipe is delivered
    as an empty read deterministically (select(2) on macOS can miss the
    post-drain EOF event and stall until the deadline)."""
    try:
        proc = subprocess.Popen(
            [str(controller), *argv],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return {"status": "fail", "reason": "controller not executable"}
    for stream in (proc.stdout, proc.stderr):
        flags = fcntl.fcntl(stream.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(stream.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
    deadline = time.monotonic() + timeout
    totals = {"out": 0, "err": 0}
    chunks: dict[str, list[bytes]] = {"out": [], "err": []}
    eof = {"out": False, "err": False}
    exceeded = False
    streams = (("out", proc.stdout), ("err", proc.stderr))
    while not exceeded:
        if time.monotonic() >= deadline:
            break
        if proc.poll() is not None and all(eof.values()):
            break
        drained = True
        for name, stream in streams:
            if eof[name]:
                continue
            try:
                chunk = os.read(stream.fileno(), 65536)
            except BlockingIOError:
                continue  # nothing buffered yet
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                chunk = b""  # broken stream: treat as closed
            if chunk:
                drained = False
                totals[name] += len(chunk)
                chunks[name].append(chunk)
                if totals[name] > MAX_CONTROLLER_BYTES:
                    exceeded = True
            else:
                eof[name] = True
        if not exceeded and drained and proc.poll() is None:
            # Sleep whenever the child is alive and no bytes were drained —
            # including after both pipes hit EOF — so an idle child never
            # busy-spins the poll loop until the deadline.
            time.sleep(0.01)
    still_running = proc.poll() is None
    if still_running:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.poll()
    proc.stdout.close()
    proc.stderr.close()
    if exceeded:
        return {"status": "fail", "reason": f"controller output exceeded the {MAX_CONTROLLER_BYTES} byte bound"}
    if still_running:
        return {"status": "fail", "reason": f"controller timeout after {timeout:g}s"}
    if proc.returncode != 0:
        return {"status": "fail", "reason": f"controller exited with code {proc.returncode}"}
    return {"status": "ok", "stdout": b"".join(chunks["out"]).decode("utf-8", "replace")}


def bounded_read(response: Any, max_bytes: int) -> tuple[bytes, bool]:
    """One capped read: HTTPResponse.read(n)/readinto accumulates until n
    bytes or EOF (BufferedReader loops across raw reads), so a single call
    returns the same bytes as chunked accumulation."""
    data = response.read(max_bytes + 1)
    return data, len(data) > max_bytes


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Loopback probes must never follow redirects off-loopback."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


# Probes are always direct (no env/system proxies) and never follow
# redirects; a 3xx is evidence of a fail, not a license to re-request.
_NO_PROXY_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), _NoRedirect()
)


def fetch(url: str, timeout: float) -> dict[str, Any]:
    try:
        with _NO_PROXY_OPENER.open(url, timeout=timeout) as response:
            status = response.status
            ctype = response.headers.get("Content-Type", "")[:200]
            data, exceeded = bounded_read(response, MAX_RESPONSE_BYTES)
    except urllib.error.HTTPError as exc:
        return {"status": "fail", "reason": f"HTTP {exc.code}", "http_status": exc.code}
    except TimeoutError:
        # socket.timeout can escape urlopen/read unwrapped on body reads.
        return {"status": "fail", "reason": f"request timeout after {timeout:g}s"}
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        return {"status": "fail", "reason": "connection failed"}
    if exceeded:
        return {"status": "fail", "reason": f"response exceeded the {MAX_RESPONSE_BYTES} byte bound"}
    return {"status": "ok", "http_status": status, "content_type": ctype, "data": data}


# ── per-category collection ───────────────────────────────────────────────


def _report_failure(reason: str) -> dict[str, Any]:
    return {"status": "fail", "reason": reason}


def controller_status(
    cfg: Config,
    *,
    controller: Path,
    schema: str,
    expected_service: str,
    argv: list[str],
    app_r: Path | None,
    web_r: Path,
) -> dict[str, Any]:
    run = run_controller(controller, argv, cfg.timeout)
    if run["status"] != "ok":
        return run
    try:
        obj = json.loads(run["stdout"])
    except json.JSONDecodeError:
        return _report_failure("controller output is not a single JSON object")
    if not isinstance(obj, dict):
        return _report_failure("controller output is not a single JSON object")
    partial: dict[str, Any] = {
        "status": "pass",
        "state": obj.get("state"),
        "mode": obj.get("mode"),
        "identity_exact": obj.get("identity_exact"),
        "service_name_exact": obj.get("service_name") == expected_service,
    }
    if app_r is not None:
        partial["scheduler_instances"] = obj.get("scheduler_instances")
    problems: list[str] = []
    if obj.get("schema") != schema:
        problems.append("unexpected schema")
    if partial["state"] != "active":
        problems.append("state is not active")
    if partial["mode"] != "production":
        problems.append("mode is not production")
    if partial["identity_exact"] is not True:
        problems.append("identity not exact")
    if app_r is not None and (
        not isinstance(obj.get("scheduler_instances"), int)
        or obj["scheduler_instances"] != 1
    ):
        problems.append("scheduler instance count is not exactly 1")
    if not partial["service_name_exact"]:
        problems.append("service name mismatch")
    app_ok = app_r is None or (isinstance(obj.get("app_dir"), str) and Path(obj["app_dir"]).resolve() == app_r)
    web_ok = isinstance(obj.get("web_root"), str) and Path(obj["web_root"]).resolve() == web_r
    partial["paths_exact"] = app_ok and web_ok
    if not partial["paths_exact"]:
        problems.append("app/web path mismatch")
    if problems:
        partial["status"] = "fail"
        partial["reason"] = "; ".join(problems)
    return partial


def collect_tasker(cfg: Config) -> tuple[str, dict[str, Any]]:
    tasker = controller_status(
        cfg,
        controller=cfg.tasker_controller,
        schema=BOX_PERSIST_SCHEMA,
        expected_service=TASKER_SERVICE,
        argv=[
            "status", "--read-only", "--mode", "production",
            "--app-dir", str(cfg.app),
            "--web-root", str(cfg.www),
            "--service-name", TASKER_SERVICE,
        ],
        app_r=cfg.app,
        web_r=cfg.www,
    )
    static = controller_status(
        cfg,
        controller=cfg.static_controller,
        schema=STATIC_PERSIST_SCHEMA,
        expected_service=STATIC_SERVICE,
        argv=[
            "status", "--read-only", "--mode", "production",
            "--web-root", str(cfg.www),
            "--service-name", STATIC_SERVICE,
        ],
        app_r=None,
        web_r=cfg.www,
    )
    status = "fail" if tasker["status"] == "fail" or static["status"] == "fail" else "pass"
    return status, {"tasker_controller": tasker, "static_controller": static}


def _compact(tasks: list[Any], runs: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Retained tasks are bounded by the response fetch cap; runs are capped.
    compact_tasks = [
        {key: task[key] for key in TASK_KEYS if isinstance(task, dict) and key in task}
        for task in tasks
    ]
    compact_runs = [
        {key: run[key] for key in RUN_KEYS if isinstance(run, dict) and key in run}
        for run in runs
    ][:RUNS_CAP]
    return compact_tasks, compact_runs


def collect_jobs(cfg: Config) -> tuple[str, dict[str, Any]]:
    api = fetch(cfg.api_url, cfg.timeout)
    if api["status"] == "ok":
        try:
            obj = json.loads(api["data"].decode("utf-8", "replace"))
        except json.JSONDecodeError:
            api = _report_failure("API response is not a JSON object")
        else:
            if not isinstance(obj, dict):
                api = _report_failure("API response is not a JSON object")
            elif not isinstance(obj.get("tasks"), list) or not isinstance(obj.get("recent_runs"), list):
                api = _report_failure("API response lacks tasks/recent_runs lists")
            else:
                compact_tasks, compact_runs = _compact(obj["tasks"], obj["recent_runs"])
                api = {
                    "status": "pass",
                    "http_status": api["http_status"],
                    "tasks_retained": len(compact_tasks),
                    "runs_retained": len(compact_runs),
                    "tasks": compact_tasks,
                    "recent_runs": compact_runs,
                }
    static = fetch(cfg.static_url, cfg.timeout)
    if static["status"] == "ok":
        static = {
            "status": "pass",
            "http_status": static["http_status"],
            "content_type": static["content_type"],
            "bytes": len(static["data"]),
        }
    status = "fail" if api["status"] == "fail" or static["status"] == "fail" else "pass"
    return status, {"api": api, "static_root": static}


def collect_freshness(cfg: Config) -> tuple[str, dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    worst = "pass"
    for rel in FRESHNESS_FILES:
        path = cfg.root / rel
        try:
            st = os.stat(path)
        except OSError:
            entries.append(
                {"path": rel, "status": "warning", "present": False, "reason": "missing"}
            )
            worst = "warning"
            continue
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
        age = (cfg.now - mtime).total_seconds()
        entry: dict[str, Any] = {
            "path": rel,
            "status": "pass" if age <= cfg.freshness_max_age else "warning",
            "present": True,
            "age_seconds": int(round(age)),
            "mtime": mtime.isoformat(),
            "size_bytes": st.st_size,
        }
        entries.append(entry)
        if entry["status"] == "warning":
            worst = "warning"
    return worst, {"files": entries}


def read_bounded_bytes(path: Path, cap: int) -> bytes | None:
    """Read at most ``cap`` bytes; None when the file is unreadable."""
    try:
        with open(path, "rb") as fh:
            return fh.read(cap + 1)
    except OSError:
        return None


def _parse_archive_log(path: Path) -> tuple[str | None, str | None]:
    data = read_bounded_bytes(path, MAX_LOG_BYTES)
    if not data:
        return None, None
    if len(data) > MAX_LOG_BYTES:
        data = data[:MAX_LOG_BYTES]
    best: tuple[datetime, str, str] | None = None
    for match in ARCHIVE_LOG_LINE_RE.finditer(data.decode("utf-8", "replace")):
        raw_ts = match.group(1)
        try:
            ts_dt = datetime.strptime(raw_ts, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
        if best is None or ts_dt > best[0]:
            best = (ts_dt, raw_ts, match.group(2))
    if best is None:
        return None, None
    return best[1], best[2]


def collect_archive(cfg: Config) -> tuple[str, dict[str, Any]]:
    stamp_day: str | None = None
    stamp_data = read_bounded_bytes(cfg.root / ARCHIVE_STAMP_REL, MAX_STAMP_BYTES)
    if stamp_data is not None and len(stamp_data) <= MAX_STAMP_BYTES:
        raw = stamp_data.decode("utf-8", "replace").strip()
    else:
        raw = ""  # unreadable or oversized: treated as missing/malformed
    if STAMP_RE.fullmatch(raw):
        stamp_day = raw
    utc_day: str | None = None
    if stamp_day is not None:
        try:
            utc_day = datetime.strptime(stamp_day, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            # Calendar-invalid eight-digit stamp (e.g. month 13): malformed.
            stamp_day = None
    details: dict[str, Any] = {
        "utc_day": utc_day,
        "latest_success": None,
        "sha256": None,
    }
    latest, sha = _parse_archive_log(cfg.root / ARCHIVE_LOG_REL)
    details["latest_success"] = latest
    details["sha256"] = sha
    status = archive_status(stamp_day, cfg.utc_day)
    if status == "fail":
        if stamp_day is None:
            details["reason"] = "archive stamp missing or malformed"
        else:
            details["reason"] = "archive stamp is older than one UTC day"
    return status, details


def collect_resources(cfg: Config) -> tuple[str, dict[str, Any]]:
    try:
        st = os.statvfs(str(cfg.root))
        total = st.f_blocks * st.f_frsize
        free = st.f_bavail * st.f_frsize
        used = total - free
        percent = round(100.0 * used / total, 1) if total > 0 else 0.0
        disk: dict[str, Any] = {
            "status": evaluate_disk(total, used, free),
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "percent": percent,
        }
    except OSError:
        disk = {"status": "warning", "reason": "disk info unavailable"}
    meminfo_path = Path("/proc/meminfo")
    try:
        meminfo_text = meminfo_path.read_text(encoding="utf-8", errors="replace")
        memory: dict[str, Any] | str = (
            parse_meminfo(meminfo_text) or "unavailable"
        )
    except OSError:
        memory = "unavailable"
    return disk["status"], {"disk": disk, "memory": memory}


def _strict_bool(obj: Any, key: str) -> bool | None:
    value = obj.get(key) if isinstance(obj, dict) else None
    return value if isinstance(value, bool) else None


def _authority_result(
    *,
    status: str,
    present: bool,
    host_label: str | None,
    collected_at: str | None,
    tasker_active: bool | None,
    tasker_enabled: bool | None,
    archive_timer_active: bool | None,
    archive_timer_enabled: bool | None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Shared builder for every authority outcome (no duplicate literals)."""
    result: dict[str, Any] = {
        "status": status,
        "present": present,
        "host_label": host_label,
        "collected_at": collected_at,
        "tasker_active": tasker_active,
        "tasker_enabled": tasker_enabled,
        "archive_timer_active": archive_timer_active,
        "archive_timer_enabled": archive_timer_enabled,
    }
    if reason is not None:
        result["reason"] = reason
    return result


def collect_authority(cfg: Config) -> tuple[str, dict[str, Any]]:
    """Evaluate the former-authority proof (v1 schema).

    Pass requires: non-symlink regular file owned by the collector uid with
    exactly mode 0600, bounded contents, valid v1 schema, expected host
    label, timestamp not future-dated beyond a small fixed clock-skew
    allowance, fresh under max age, and all four active/enabled booleans
    false. Wrong schema/host/ownership/mode/future timestamp is a warning;
    active/enabled stays fail. Reasons are static and never include the
    source path."""
    none = dict(
        host_label=None,
        collected_at=None,
        tasker_active=None,
        tasker_enabled=None,
        archive_timer_active=None,
        archive_timer_enabled=None,
    )
    path = cfg.authority_proof
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return "warning", _authority_result(
            status="warning", present=False, reason="proof file absent", **none
        )
    except OSError:
        return "warning", _authority_result(
            status="warning", present=False, reason="proof file unreadable", **none
        )
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        return "warning", _authority_result(
            status="warning", present=True,
            reason="source file must be a regular non-symlink file", **none
        )
    if st.st_uid != os.getuid():
        return "warning", _authority_result(
            status="warning", present=True, reason="source file ownership mismatch", **none
        )
    if stat.S_IMODE(st.st_mode) != 0o600:
        return "warning", _authority_result(
            status="warning", present=True, reason="source file must be exactly mode 0600", **none
        )
    data = read_bounded_bytes(path, MAX_PROOF_BYTES)
    if data is None or len(data) > MAX_PROOF_BYTES:
        return "warning", _authority_result(
            status="warning", present=True, reason="malformed proof", **none
        )
    try:
        obj = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "warning", _authority_result(
            status="warning", present=True, reason="malformed proof", **none
        )
    host_label = obj.get("host_label") if isinstance(obj, dict) else None
    collected_raw = obj.get("collected_at") if isinstance(obj, dict) else None
    tasker = obj.get("tasker") if isinstance(obj, dict) else None
    archive_timer = obj.get("archive_timer") if isinstance(obj, dict) else None
    tasker_active = _strict_bool(tasker, "active")
    tasker_enabled = _strict_bool(tasker, "enabled")
    archive_active = _strict_bool(archive_timer, "active")
    archive_enabled = _strict_bool(archive_timer, "enabled")
    collected_dt: datetime | None = None
    if isinstance(collected_raw, str):
        try:
            collected_dt = datetime.fromisoformat(collected_raw.replace("Z", "+00:00"))
            if collected_dt.tzinfo is None:
                collected_dt = collected_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            collected_dt = None

    invalid = _authority_result(
        status="warning", present=True, reason="malformed proof", **none
    )
    if not isinstance(host_label, str) or not host_label:
        return "warning", invalid
    if collected_dt is None:
        return "warning", invalid
    if (
        tasker_active is None or tasker_enabled is None
        or archive_active is None or archive_enabled is None
    ):
        return "warning", invalid
    if any((tasker_active, tasker_enabled, archive_active, archive_enabled)):
        return "fail", _authority_result(
            status="fail",
            present=True,
            host_label=host_label,
            collected_at=collected_raw,
            tasker_active=tasker_active,
            tasker_enabled=tasker_enabled,
            archive_timer_active=archive_active,
            archive_timer_enabled=archive_enabled,
            reason="former authority still active or enabled",
        )
    if obj.get("schema") != PROOF_SCHEMA:
        return "warning", _authority_result(
            status="warning", present=True, reason="unexpected source schema", **none
        )
    if host_label != cfg.expected_host:
        return "warning", _authority_result(
            status="warning", present=True, reason="host label mismatch", **none
        )
    if (collected_dt - cfg.now).total_seconds() > PROOF_CLOCK_SKEW_ALLOWANCE:
        return "warning", _authority_result(
            status="warning", present=True, reason="source timestamp is future-dated", **none
        )
    if (cfg.now - collected_dt).total_seconds() > cfg.freshness_max_age:
        return "warning", _authority_result(
            status="warning", present=True,
            host_label=host_label,
            collected_at=collected_raw,
            tasker_active=tasker_active,
            tasker_enabled=tasker_enabled,
            archive_timer_active=archive_active,
            archive_timer_enabled=archive_enabled,
            reason="proof is stale",
        )
    return "pass", _authority_result(
        status="pass",
        present=True,
        host_label=host_label,
        collected_at=collected_raw,
        tasker_active=tasker_active,
        tasker_enabled=tasker_enabled,
        archive_timer_active=archive_active,
        archive_timer_enabled=archive_enabled,
    )


# ── evidence writing ──────────────────────────────────────────────────────


def atomic_replace(path: Path, data: bytes, mode: int) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".plde-")
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


def write_evidence(
    out_root: Path,
    day: str,
    payloads: dict[str, dict[str, Any]],
) -> Path:
    """Serialize with a per-file bound, then commit atomically: a fresh day
    dir appears via one rename (temp dir populated once); a same-day rerun
    performs only per-file atomic replacements with no temp-dir I/O. A
    same-day target must be a non-symlink directory chmod'ed to 0700 with
    exactly the expected entry set; unexpected entries are rejected, never
    deleted, and expected files must be regular non-symlinks before
    replacement. Temp dirs are unique (mkdtemp), so stale/colliding .tmp-*
    siblings are never an error; every failure here is a clean exit-1
    diagnostic without a traceback."""
    serialized: dict[str, bytes] = {}
    for category, payload in payloads.items():
        file_name = f"{category}.json"
        data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        if len(data) > MAX_FILE_BYTES:
            die(f"serialized size for {file_name} exceeds the {MAX_FILE_BYTES} byte bound")
        serialized[file_name] = data
    try:
        out_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(out_root, 0o700)  # harden a pre-existing output root too
        target = out_root / day
        try:
            target_stat = target.lstat()
        except FileNotFoundError:
            target_stat = None
        if target_stat is None:
            tmp_dir = Path(tempfile.mkdtemp(dir=str(out_root), prefix=f".tmp-{day}-"))
            try:
                for file_name, data in serialized.items():
                    path = tmp_dir / file_name
                    path.write_bytes(data)
                    os.chmod(path, 0o600)
                os.replace(tmp_dir, target)
            except BaseException:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise
        else:
            if stat.S_ISLNK(target_stat.st_mode) or not stat.S_ISDIR(target_stat.st_mode):
                die(f"evidence day path must be a non-symlink directory: {day}")
            entries = set(os.listdir(target))
            if entries != set(serialized):
                die(
                    "evidence day directory contains unexpected entries "
                    f"(expected files: {', '.join(sorted(serialized))}); "
                    f"refusing to rewrite {day}"
                )
            os.chmod(target, 0o700)
            for file_name in serialized:
                path = target / file_name
                try:
                    file_stat = path.lstat()
                except FileNotFoundError:
                    file_stat = None
                if file_stat is not None and (
                    stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode)
                ):
                    die(
                        f"evidence file must be a regular non-symlink file "
                        f"before rewrite: {file_name}"
                    )
                atomic_replace(path, serialized[file_name], 0o600)
    except OSError as exc:
        die(f"failed to write evidence: {exc}")
    return target


# ── main ──────────────────────────────────────────────────────────────────


def envelope(category: str, status: str, details: dict[str, Any], collected_at: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "category": category,
        "collected_at": collected_at,
        "status": status,
        "details": details,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config(args)
    cfg.validate()

    collected_at = cfg.now.isoformat()
    categories: dict[str, str] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for category, collector in (
        ("tasker", collect_tasker),
        ("jobs", collect_jobs),
        ("freshness", collect_freshness),
        ("archive", collect_archive),
        ("resources", collect_resources),
        ("authority", collect_authority),
    ):
        status, details = collector(cfg)
        categories[category] = status
        payloads[category] = envelope(category, status, details, collected_at)

    overall, notify = overall_status(categories)
    summary_details = {
        "utc_day": cfg.utc_day,
        "overall": overall,
        "notify_grok_bot": notify,
        "categories": categories,
    }
    payloads["summary"] = envelope("summary", overall, summary_details, collected_at)

    day_dir = write_evidence(cfg.output_root, cfg.utc_day, payloads)
    emit(
        {
            "schema": SCHEMA,
            "category": "summary",
            "utc_day": cfg.utc_day,
            "overall": overall,
            "notify_grok_bot": notify,
            "categories": categories,
            "output_dir": str(day_dir),
        }
    )
    return 0 if overall in ("pass", "warning") else 2


if __name__ == "__main__":
    sys.exit(main())