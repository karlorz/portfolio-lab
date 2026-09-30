#!/usr/bin/env python3
"""Mechanical S3 reporter for ops/maintenance runs.

Always writes ``data/ops-maintenance/last-<cycle>.json`` with the five-section
contract. On any ``fail`` status, posts a fail-only SkillWiki ``wiki_capture``
titled ``ESCALATION: ...``. Never prints credentials, tokens, or bearer values.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

REPORT_KEYS = ("host_health", "checks", "drift", "actions", "escalations")
STATUSES = {"ok", "warn", "fail"}
URGENCIES = {"high", "medium", "low"}
DEFAULT_MCP_URL = "https://wiki.karldigi.dev/mcp"
TRANSCRIPT_CAP = 2_000_000
CAPTURE_TIMEOUT_SECS = 15
MAX_EVIDENCE_CHARS = 2000

_SECRET_RE = re.compile(
    r"(?is)"
    r"(authorization\s*[:=]\s*[^\n]+)"
    r"|(bearer\s+\S+)"
    r"|(api[_-]?key\s*[:=]\s*\S+)"
    r"|((?:sk|xai|gsk)-[a-z0-9\-_.]+)"
    r"|((?:SKILLWIKI_MCP_TOKEN|XAI_API_KEY|GROK_API_KEY|CURSOR_BOX_MCP_TOKEN)"
    r"\s*[:=]\s*\S+)"
)


class _NoRedirect(urllib.request.HTTPErrorProcessor):
    def http_response(self, request, response):  # type: ignore[override]
        if 300 <= response.status < 400:
            raise RuntimeError("redirect_refused")
        return response

    https_response = http_response


def redact(text: str) -> str:
    if not text:
        return ""
    redacted = text
    for _ in range(3):
        nxt = _SECRET_RE.sub("[REDACTED:credential]", redacted)
        if nxt == redacted:
            break
        redacted = nxt
    return redacted


def _truncate(text: str, limit: int = MAX_EVIDENCE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def is_report(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    return all(key in obj for key in REPORT_KEYS)


def overall_status(report: Mapping[str, Any]) -> str:
    statuses: list[str] = []
    host = report.get("host_health")
    if isinstance(host, dict):
        statuses.append(str(host.get("status") or ""))
    checks = report.get("checks")
    if isinstance(checks, list):
        for item in checks:
            if isinstance(item, dict):
                statuses.append(str(item.get("status") or ""))
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "ok"


def _iter_json_objects(text: str) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        brace = text.find("{", i)
        if brace < 0:
            return
        try:
            obj, end = decoder.raw_decode(text, brace)
        except json.JSONDecodeError:
            i = brace + 1
            continue
        if isinstance(obj, dict):
            yield obj
        i = max(end, brace + 1)


def extract_report(transcript_text: str) -> dict[str, Any] | None:
    snippet = transcript_text[:TRANSCRIPT_CAP]
    for obj in _iter_json_objects(snippet):
        if is_report(obj):
            return obj
        for key in ("result", "message", "content", "text"):
            value = obj.get(key)
            if isinstance(value, dict) and is_report(value):
                return value
            if isinstance(value, str):
                for inner in _iter_json_objects(value):
                    if is_report(inner):
                        return inner
    return None


def _empty_drift(cycle: str) -> dict[str, Any]:
    if cycle == "weekly":
        return {"dependencies": [], "repos": []}
    return {"repos": []}


def synthesize_fail(
    *,
    cycle: str,
    agent_exit: int,
    transcript_text: str,
) -> dict[str, Any]:
    redacted = _truncate(redact(transcript_text.strip() or "(empty transcript)"))
    lowered = redacted.lower()
    if "unauthorized (401)" in lowered or "auth_kind=none" in lowered:
        issue = "headless grok CLI unauthorized (401); no auth context"
        recommended = (
            "Operator: provision non-interactive grok CLI credentials on "
            "cursor-box (do not paste tokens into chat). Re-run "
            "ops/maintenance/run.sh daily after auth works."
        )
        check_name = "grok-cli-auth"
        urgency = "high"
    elif not transcript_text.strip():
        issue = f"ops-maint-{cycle} produced an empty transcript (agent exit {agent_exit})"
        recommended = "Inspect cron_guard logs and grok binary PATH on cursor-box."
        check_name = "grok-agent-transcript"
        urgency = "high"
    else:
        issue = f"ops-maint-{cycle} grok run failed (exit {agent_exit}) without a 5-section report"
        recommended = (
            "Inspect data/ops-maintenance/run-*.json on cursor-box and fix the "
            "agent invocation or runbook output."
        )
        check_name = "grok-agent"
        urgency = "high"
    return {
        "host_health": {
            "status": "fail",
            "loadavg": "",
            "memory_free_mb": None,
            "disk_usage_pct": None,
        },
        "checks": [
            {
                "name": check_name,
                "status": "fail",
                "evidence": redacted,
            }
        ],
        "drift": _empty_drift(cycle),
        "actions": [],
        "escalations": [
            {
                "issue": issue,
                "urgency": urgency,
                "recommended_action": recommended,
            }
        ],
    }


def normalize_report(report: Mapping[str, Any], *, cycle: str, agent_exit: int) -> dict[str, Any]:
    host = report.get("host_health")
    if not isinstance(host, dict):
        host = {"status": "fail"}
    host_status = str(host.get("status") or "fail")
    if host_status not in STATUSES:
        host_status = "fail"
    host_out = {
        "status": host_status,
        "loadavg": host.get("loadavg", ""),
        "memory_free_mb": host.get("memory_free_mb"),
        "disk_usage_pct": host.get("disk_usage_pct"),
    }

    checks_out: list[dict[str, Any]] = []
    checks = report.get("checks")
    if isinstance(checks, list):
        for item in checks:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "fail")
            if status not in STATUSES:
                status = "fail"
            checks_out.append(
                {
                    "name": str(item.get("name") or "unnamed-check"),
                    "status": status,
                    "evidence": _truncate(redact(str(item.get("evidence") or ""))),
                }
            )

    drift = report.get("drift")
    if not isinstance(drift, dict):
        drift_out = _empty_drift(cycle)
    else:
        drift_out = json.loads(redact(json.dumps(drift, default=str)))

    actions_out: list[dict[str, Any]] = []
    actions = report.get("actions")
    if isinstance(actions, list):
        for item in actions:
            if not isinstance(item, dict):
                continue
            actions_out.append(
                {
                    "action": str(item.get("action") or ""),
                    "result": str(item.get("result") or ""),
                    "detail": _truncate(redact(str(item.get("detail") or ""))),
                }
            )

    escalations_out: list[dict[str, Any]] = []
    escalations = report.get("escalations")
    if isinstance(escalations, list):
        for item in escalations:
            if not isinstance(item, dict):
                continue
            urgency = str(item.get("urgency") or "medium")
            if urgency not in URGENCIES:
                urgency = "medium"
            escalations_out.append(
                {
                    "issue": _truncate(redact(str(item.get("issue") or "")), 400),
                    "urgency": urgency,
                    "recommended_action": _truncate(
                        redact(str(item.get("recommended_action") or "")), 800
                    ),
                }
            )

    normalized = {
        "host_health": host_out,
        "checks": checks_out,
        "drift": drift_out,
        "actions": actions_out,
        "escalations": escalations_out,
    }
    if agent_exit != 0 and overall_status(normalized) != "fail":
        normalized["checks"].append(
            {
                "name": "grok-agent-exit",
                "status": "fail",
                "evidence": f"grok process exited {agent_exit} after a non-fail report",
            }
        )
        normalized["escalations"].append(
            {
                "issue": f"ops-maint-{cycle} grok exit {agent_exit}",
                "urgency": "high",
                "recommended_action": "Inspect the run transcript and cron_guard logs.",
            }
        )
    return normalized


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _parse_mcp_body(raw: bytes, content_type: str) -> dict[str, Any] | None:
    text = raw.decode("utf-8", "replace").strip()
    if not text:
        return None
    if "text/event-stream" in content_type or text.startswith("event:") or "data:" in text.splitlines()[0]:
        for line in text.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    parsed = json.loads(payload)
                    if isinstance(parsed, dict):
                        return parsed
        return None
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        return parsed
    raise ValueError("mcp_response_not_object")


def _mcp_rpc(
    url: str,
    headers: dict[str, str],
    payload: Mapping[str, Any],
    *,
    timeout: float = CAPTURE_TIMEOUT_SECS,
) -> dict[str, Any] | None:
    opener = urllib.request.build_opener(_NoRedirect)
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with opener.open(request, timeout=timeout) as response:
            session = response.headers.get("Mcp-Session-Id")
            if session:
                headers["Mcp-Session-Id"] = session
            raw = response.read(262144)
            content_type = response.headers.get("Content-Type") or ""
    except urllib.error.HTTPError as exc:
        detail = redact(exc.read(2048).decode("utf-8", "replace") if exc.fp else "")
        raise RuntimeError(f"mcp_http_{exc.code}:{detail[:200]}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"mcp_url_error:{type(exc.reason).__name__}") from None
    return _parse_mcp_body(raw, content_type)


def wiki_capture(
    *,
    kind: str,
    project: str,
    title: str,
    content: str,
    url: str,
    token: str,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    init = _mcp_rpc(
        url,
        headers,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "ops-maintenance-report", "version": "1"},
            },
        },
    )
    if isinstance(init, dict) and init.get("error"):
        raise RuntimeError("mcp_initialize_error")
    try:
        _mcp_rpc(
            url,
            headers,
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        )
    except RuntimeError:
        pass
    result = _mcp_rpc(
        url,
        headers,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "wiki_capture",
                "arguments": {
                    "kind": kind,
                    "project": project,
                    "title": title,
                    "content": content,
                    "agent_role": "grok-bot",
                },
            },
        },
    )
    if not isinstance(result, dict):
        raise RuntimeError("mcp_empty_result")
    if result.get("error"):
        raise RuntimeError("mcp_tools_call_error")
    inner = result.get("result")
    path = None
    if isinstance(inner, dict):
        path = inner.get("path")
        if path is None and isinstance(inner.get("content"), list):
            for block in inner["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    try:
                        parsed = json.loads(str(block.get("text") or ""))
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict) and parsed.get("path"):
                        path = parsed["path"]
                        break
    return {"status": "ok", "path": path}


def first_escalation_title(cycle: str, report: Mapping[str, Any]) -> str:
    escalations = report.get("escalations")
    if isinstance(escalations, list):
        for item in escalations:
            if isinstance(item, dict) and item.get("issue"):
                issue = str(item["issue"]).strip()
                return _truncate(f"ESCALATION: ops-maint-{cycle}: {issue}", 160)
    checks = report.get("checks")
    if isinstance(checks, list):
        for item in checks:
            if isinstance(item, dict) and item.get("status") == "fail":
                name = str(item.get("name") or "check")
                return f"ESCALATION: ops-maint-{cycle}: {name}"
    return f"ESCALATION: ops-maint-{cycle}: checks failed"


def maybe_capture(
    *,
    cycle: str,
    report: Mapping[str, Any],
    capture_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if overall_status(report) != "fail":
        return {"status": "skipped", "reason": "no_fail"}
    if os.environ.get("OPS_MAINT_DISABLE_CAPTURE") == "1":
        return {"status": "skipped", "reason": "disabled"}
    token = (
        os.environ.get("OPS_MAINT_CAPTURE_TOKEN")
        or os.environ.get("SKILLWIKI_MCP_TOKEN")
        or ""
    ).strip()
    if not token:
        return {"status": "skipped", "reason": "missing_token"}
    url = (
        os.environ.get("OPS_MAINT_CAPTURE_URL")
        or os.environ.get("SKILLWIKI_MCP_URL")
        or DEFAULT_MCP_URL
    ).strip()
    title = first_escalation_title(cycle, report)
    content = redact(
        "Fail-only ops-maintenance escalation.\n\n"
        f"cycle: {cycle}\n"
        f"overall: fail\n\n"
        "```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n"
    )
    poster = capture_fn or wiki_capture
    try:
        result = poster(
            kind="note",
            project="portfolio-lab",
            title=title,
            content=content,
            url=url,
            token=token,
        )
    except Exception as exc:
        return {"status": "failed", "reason": redact(str(exc))[:200]}
    status = str(result.get("status") or "ok")
    path = result.get("path")
    out = {"status": status}
    if isinstance(path, str) and path:
        out["path"] = path
    return out


def build_report(
    *,
    cycle: str,
    transcript_text: str,
    agent_exit: int,
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    extracted = extract_report(transcript_text)
    if extracted is None and existing and is_report(existing):
        extracted = dict(existing)
    if extracted is None:
        extracted = synthesize_fail(
            cycle=cycle, agent_exit=agent_exit, transcript_text=transcript_text
        )
    return normalize_report(extracted, cycle=cycle, agent_exit=agent_exit)


def write_and_capture(
    *,
    cycle: str,
    transcript_path: Path,
    output_dir: Path,
    agent_exit: int,
    capture: bool = True,
    capture_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    text = ""
    if transcript_path.is_file():
        text = transcript_path.read_text(encoding="utf-8", errors="replace")[:TRANSCRIPT_CAP]
    existing = None
    last_path = output_dir / f"last-{cycle}.json"
    if last_path.is_file():
        try:
            loaded = json.loads(last_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if is_report(loaded):
            existing = loaded
    report = build_report(
        cycle=cycle,
        transcript_text=text,
        agent_exit=agent_exit,
        existing=existing,
    )
    atomic_write_json(last_path, report)
    capture_result = {"status": "skipped", "reason": "no_fail"}
    if capture:
        capture_result = maybe_capture(cycle=cycle, report=report, capture_fn=capture_fn)
        if overall_status(report) == "fail":
            report["actions"].append(
                {
                    "action": "fail-only wiki_capture",
                    "result": str(capture_result.get("status")),
                    "detail": _truncate(
                        redact(
                            str(
                                capture_result.get("path")
                                or capture_result.get("reason")
                                or ""
                            )
                        )
                    ),
                }
            )
            atomic_write_json(last_path, report)
    summary = {
        "cycle": cycle,
        "overall": overall_status(report),
        "report": str(last_path),
        "capture": capture_result.get("status"),
    }
    if capture_result.get("reason"):
        summary["capture_reason"] = capture_result["reason"]
    if capture_result.get("path"):
        summary["capture_path"] = capture_result["path"]
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write last-<cycle>.json and fail-only wiki_capture")
    parser.add_argument("--cycle", choices=("daily", "weekly"), required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--agent-exit", type=int, default=0)
    parser.add_argument("--no-capture", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = write_and_capture(
            cycle=args.cycle,
            transcript_path=Path(args.transcript),
            output_dir=Path(args.output_dir),
            agent_exit=args.agent_exit,
            capture=not args.no_capture,
        )
    except OSError:
        print("ERROR: failed to write ops-maintenance report", file=sys.stderr)
        return 2
    print(json.dumps(summary, separators=(",", ":")))
    if summary.get("overall") == "fail":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
