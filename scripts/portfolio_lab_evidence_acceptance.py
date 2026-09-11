#!/usr/bin/env python3
"""Bounded seven-day acceptance checker for cursor-box daily evidence.

A read-only acceptance gate over the last seven consecutive UTC day
directories produced by ``portfolio_lab_daily_evidence.py`` under
``--evidence-root`` (schema ``portfolio-lab-daily-evidence/v1``), plus the
optional attended recycle proof ``recycle.json`` (schema
``portfolio-lab-recycle-proof/v1``). Emits one compact JSON report (schema
``portfolio-lab-evidence-acceptance/v1``). Never mutates evidence: no
network, no subprocesses, no controller/service/SSH/shell/Cloudflare access,
no restore, and no writes other than the optional atomic ``--output-json``
report (mode 0600, outside the evidence root, no Markdown).

Day acceptance
    The window is exactly the seven calendar days ending at ``--end-day``
    (default: current UTC day). ``--evidence-root`` must be an absolute
    non-symlink directory (a symlink or non-directory is rejected as input);
    its mode is validated (0700) but never mutated. Date-named directories
    outside the selected window (older or newer evidence days) are tolerated
    only as real non-symlink directories with exactly mode 0700 (contents are
    not validated because they are outside the window); date-named
    files/symlinks/bad modes and any non-date unexpected entry are
    investigated. Each day directory must be a non-symlink directory with
    exactly mode 0700 containing exactly the seven evidence files (tasker,
    jobs, freshness, archive, resources, authority, summary), each a
    non-symlink regular file with exactly mode 0600 and at most 262144
    bytes; extra entries are rejected. Each file must be valid JSON with the
    daily-evidence v1 schema, a category matching the file name, a UTC-offset
    ``collected_at`` whose UTC day matches the directory, and a status in
    pass|warning|fail. The summary must match the six category files
    (category map, overall envelope/details, notify-on-fail rule, utc day).
    A day passes when every category is pass and every acceptance criterion
    holds: tasker/jobs/freshness/resources/authority pass; archive pass with
    the same UTC day, a 64-hex sha256, and a ``latest_success`` timestamp on
    that UTC day; both controllers pass with state active, mode production,
    ``identity_exact``/``service_name_exact``/``paths_exact`` true, and
    exactly one tasker scheduler instance (strict integer; bool/float
    rejected); API and static endpoints pass with HTTP 200; freshness
    observes exactly the producer's path set
    (app/data/signals.json, app/data/tasker_status.json, www/data/signals.json,
    www/data/tasker_status.json) with no duplicates and every entry pass;
    disk pass with free space >= 15 GiB (strict integer; bool/float
    rejected); authority proof nested pass, present, host sg01, ``collected_at``
    on the evidence day and not future-dated beyond the 300 s collection
    clock skew (staleness is the daily producer's tier; the checker
    independently validates the same-day timestamp), and all four
    former-authority booleans strictly false. Archive sha256 values must be
    unique across the window. Report taxonomy keeps fail-grade ``problems``
    and warning-grade ``notices`` separate per day: warning notices never
    appear in ``blockers``.

Verdicts (report ``verdict`` and exit codes 0/2/1)
    accept                every day passes and the recycle proof is valid
    ready_except_recycle  every day passes but recycle.json is absent (an
                          attended step, never assumed) with
                          --require-recycle-proof off
    extend                any day directory missing or any daily/archive
                          category warning: keep observing. Warnings never
                          count as failures (they never escalate to
                          investigate); ``--require-recycle-proof`` with an
                          absent proof also extends: it blocks acceptance
                          without implying evidence failure
    investigate           any malformed/security/invariant issue, any
                          category fail, a duplicate archive sha256, an
                          invalid recycle proof, or an unexpected entry in
                          the evidence root
    1                     invalid CLI input or report write failure (stdout
                          stays empty; the report file is written before
                          stdout emission)

Recycle proof (attended, optional producer)
    ``<evidence-root>/recycle.json``: one JSON object with schema
    ``portfolio-lab-recycle-proof/v1``, category ``recycle``, status
    ``pass``, and details carrying ``performed_at`` (ISO-8601 with a UTC
    offset, within the seven-day window), plus ``pre_recycle`` and
    ``post_recycle`` phases each with ``scheduler_instances`` strictly the
    integer 1 (bool/float rejected) and ``tasker``/``static``/``tunnel``/
    ``api`` all ``pass``. The envelope ``collected_at`` must be present, a
    UTC-offset ISO-8601 timestamp inside the seven-day window, and on the
    same UTC day as ``performed_at``. Absent -> unproven; present but
    invalid -> investigate.

Diagnostics are static, secret-free reason strings (no evidence contents,
paths, or captured output are echoed).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA_DAILY = "portfolio-lab-daily-evidence/v1"
SCHEMA_RECYCLE = "portfolio-lab-recycle-proof/v1"
SCHEMA_REPORT = "portfolio-lab-evidence-acceptance/v1"
EXPECTED_HOST = "sg01"
WINDOW_DAYS = 7
MAX_EVIDENCE_BYTES = 262144
GIB = 1024**3
DISK_FREE_MIN = 15 * GIB
CATEGORIES = ("tasker", "jobs", "freshness", "archive", "resources", "authority")
EXPECTED_FILES = tuple(f"{name}.json" for name in (*CATEGORIES, "summary"))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RECYCLE_COMPONENTS = ("tasker", "static", "tunnel", "api")
AUTHORITY_CLOCK_SKEW_ALLOWANCE = 300.0  # matches the daily producer's proof skew allowance
FRESHNESS_PATHS = (
    "app/data/signals.json",
    "app/data/tasker_status.json",
    "www/data/signals.json",
    "www/data/tasker_status.json",
)


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


class _ArgumentParser(argparse.ArgumentParser):
    """argparse exits 2 on usage errors by default; the checker contract is
    exit 1 for invalid CLI usage (stdout stays empty)."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _ArgumentParser(
        prog="portfolio_lab_evidence_acceptance.py",
        description="Read-only seven-day acceptance gate for cursor-box daily evidence.",
    )
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--end-day", default=None)
    parser.add_argument("--days", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--require-recycle-proof", action="store_true")
    return parser.parse_args(argv)


@dataclass(frozen=True)
class Config:
    evidence_root: Path
    end_day: str
    start_day: str
    days: int
    output_json: Path | None
    require_recycle_proof: bool


def _is_calendar_day(name: str) -> bool:
    """Strict YYYY-MM-DD calendar name (day directories and dates)."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", name):
        return False
    try:
        datetime.strptime(name, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _parse_day(raw: str, what: str) -> str:
    text = raw.strip()
    if not _is_calendar_day(text):
        die(f"{what} must be a YYYY-MM-DD calendar date")
    return text


def build_config(args: argparse.Namespace) -> Config:
    root = Path(args.evidence_root)
    if not root.is_absolute():
        die("PLAE_EVIDENCE_ROOT must be an absolute path")
    end_day = _parse_day(
        args.end_day or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "PLAE_END_DAY",
    )
    try:
        days = WINDOW_DAYS if args.days is None else int(args.days)
    except ValueError:
        die("PLAE_DAYS must be an integer")
    if days != WINDOW_DAYS:
        die(f"PLAE_DAYS must be exactly {WINDOW_DAYS}")
    start = datetime.strptime(end_day, "%Y-%m-%d").date() - timedelta(days=WINDOW_DAYS - 1)
    output_json: Path | None = None
    if args.output_json is not None:
        output_json = Path(args.output_json)
        if not output_json.is_absolute():
            die("PLAE_OUTPUT_JSON must be an absolute path")
        if output_json.resolve().is_relative_to(root.resolve()):
            die("PLAE_OUTPUT_JSON must sit outside the evidence root")
    return Config(root, end_day, start.strftime("%Y-%m-%d"), WINDOW_DAYS, output_json, args.require_recycle_proof)


# ── shared parse helpers ──────────────────────────────────────────────────


def _parse_aware_utc(text: str) -> datetime | None:
    """ISO-8601 with a UTC offset (or Z); naive timestamps are rejected."""
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def _overall(statuses: dict[str, str]) -> str:
    """fail if any fail, else warning if any warning, else pass."""
    if any(status == "fail" for status in statuses.values()):
        return "fail"
    if any(status == "warning" for status in statuses.values()):
        return "warning"
    return "pass"


def _read_bounded(path: Path, cap: int) -> tuple[bytes, bool, str | None]:
    """(data, exceeded, error): at most ``cap`` bytes; error is static."""
    try:
        with open(path, "rb") as handle:
            data = handle.read(cap + 1)
    except OSError:
        return b"", False, "file unreadable"
    return data, len(data) > cap, None


# ── per-day validation ────────────────────────────────────────────────────


def _envelope_problems(obj: dict[str, Any], file_name: str, day: str) -> list[str]:
    problems: list[str] = []
    if obj.get("schema") != SCHEMA_DAILY:
        problems.append("unexpected schema")
    if obj.get("category") != file_name[: -len(".json")]:
        problems.append("category does not match file name")
    collected_raw = obj.get("collected_at")
    collected = _parse_aware_utc(collected_raw) if isinstance(collected_raw, str) else None
    if collected is None:
        problems.append("collected_at must be an ISO-8601 timestamp with a UTC offset")
    elif collected.strftime("%Y-%m-%d") != day:
        problems.append("collected_at UTC day does not match directory day")
    if obj.get("status") not in ("pass", "warning", "fail"):
        problems.append("status must be pass|warning|fail")
    return problems


def _validate_evidence_file(path: Path, file_name: str, day: str) -> tuple[dict[str, Any] | None, list[str]]:
    """One evidence file: lstat checks, bounded read, envelope checks.
    Returns (payload, reasons); payload is None when the file is unusable."""
    reasons: list[str] = []
    try:
        st = path.lstat()
    except FileNotFoundError:
        return None, ["missing evidence file"]
    except OSError:
        return None, ["evidence file unreadable"]
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        return None, ["evidence file must be a regular non-symlink file"]
    if stat.S_IMODE(st.st_mode) != 0o600:
        reasons.append("evidence file must be exactly mode 0600")
    data, exceeded, error = _read_bounded(path, MAX_EVIDENCE_BYTES)
    if error is not None:
        reasons.append(error)
        return None, reasons
    if exceeded:
        reasons.append(f"evidence file exceeds the {MAX_EVIDENCE_BYTES} byte bound")
        return None, reasons
    try:
        obj = json.loads(data.decode("utf-8"))
    except UnicodeDecodeError:
        reasons.append("evidence file must be valid utf-8")
        return None, reasons
    except json.JSONDecodeError:
        reasons.append("evidence file is not valid JSON")
        return None, reasons
    if not isinstance(obj, dict):
        reasons.append("evidence file is not a JSON object")
        return None, reasons
    problems = _envelope_problems(obj, file_name, day)
    if problems:
        reasons.extend(problems)
        return None, reasons
    return obj, reasons


def _summary_problems(payload: dict[str, Any], categories: dict[str, str], day: str) -> list[str]:
    problems: list[str] = []
    details = payload.get("details")
    if not isinstance(details, dict):
        return ["summary details missing"]
    if details.get("utc_day") != day:
        problems.append("summary utc day does not match directory day")
    if details.get("categories") != categories:
        problems.append("summary category map does not match category files")
    overall = _overall(categories)
    if details.get("overall") != overall:
        problems.append("summary overall does not match category statuses")
    if payload.get("status") != overall:
        problems.append("summary envelope status does not match category statuses")
    if details.get("notify_grok_bot") != (overall == "fail"):
        problems.append("summary notify flag does not match overall status")
    return problems


def _criterion_problems(parsed: dict[str, dict[str, Any]], day: str) -> list[str]:
    """Acceptance criteria over valid envelopes; a category is only
    criterion-checked when its status is pass (warnings/fails already carry
    their own reasons)."""
    problems: list[str] = []
    statuses = {file_name: payload["status"] for file_name, payload in parsed.items()}

    def active(category: str) -> bool:
        return statuses.get(f"{category}.json") == "pass"

    def _controller_problems(ctrl: Any, label: str) -> list[str]:
        if not isinstance(ctrl, dict):
            return [f"{label} controller details missing"]
        found: list[str] = []
        if (
            ctrl.get("status") != "pass"
            or ctrl.get("state") != "active"
            or ctrl.get("mode") != "production"
            or ctrl.get("identity_exact") is not True
            or ctrl.get("service_name_exact") is not True
            or ctrl.get("paths_exact") is not True
        ):
            found.append(f"{label} controller must be pass, active, production with exact identity and exact paths")
        if label == "tasker":
            instances = ctrl.get("scheduler_instances")
            if type(instances) is not int or instances != 1:
                found.append("tasker scheduler instance count is not exactly 1")
        return found

    if active("tasker"):
        details = parsed["tasker.json"]["details"]
        tasker_ctrl = details.get("tasker_controller") if isinstance(details, dict) else None
        static_ctrl = details.get("static_controller") if isinstance(details, dict) else None
        problems.extend(_controller_problems(tasker_ctrl, "tasker"))
        problems.extend(_controller_problems(static_ctrl, "static"))
    if active("jobs"):
        details = parsed["jobs.json"]["details"]
        api = details.get("api") if isinstance(details, dict) else None
        static_root = details.get("static_root") if isinstance(details, dict) else None
        if not isinstance(api, dict) or api.get("status") != "pass" or api.get("http_status") != 200:
            problems.append("api endpoint must pass with HTTP 200")
        if not isinstance(static_root, dict) or static_root.get("status") != "pass" or static_root.get("http_status") != 200:
            problems.append("static endpoint must pass with HTTP 200")
    if active("freshness"):
        details = parsed["freshness.json"]["details"]
        files = details.get("files") if isinstance(details, dict) else None
        if not isinstance(files, list) or not files:
            problems.append("freshness must observe the exact producer path set with every entry pass")
        else:
            if any(not isinstance(entry, dict) or entry.get("status") != "pass" for entry in files):
                problems.append("every freshness entry must be pass")
            observed = [
                entry["path"]
                for entry in files
                if isinstance(entry, dict) and isinstance(entry.get("path"), str)
            ]
            if (
                len(observed) != len(files)
                or set(observed) != set(FRESHNESS_PATHS)
                or len(set(observed)) != len(observed)
            ):
                problems.append("freshness must observe the exact producer path set with no duplicates")
    if active("archive"):
        details = parsed["archive.json"]["details"]
        archive = details if isinstance(details, dict) else {}
        if archive.get("utc_day") != day:
            problems.append("archive utc day does not match directory day")
        if not isinstance(archive.get("sha256"), str) or not SHA256_RE.fullmatch(archive["sha256"]):
            problems.append("archive sha256 must be a 64-hex string")
        latest_raw = archive.get("latest_success")
        latest = _parse_aware_utc(latest_raw) if isinstance(latest_raw, str) else None
        if latest is None or latest.strftime("%Y-%m-%d") != day:
            problems.append("archive latest success must be a timestamp on the directory day")
    if active("resources"):
        details = parsed["resources.json"]["details"]
        disk = details.get("disk") if isinstance(details, dict) else None
        free_bytes = disk.get("free_bytes") if isinstance(disk, dict) else None
        if (
            not isinstance(disk, dict)
            or disk.get("status") != "pass"
            or type(free_bytes) is not int
            or free_bytes < DISK_FREE_MIN
        ):
            problems.append("disk must pass with free space >= 15 GiB")
    if active("authority"):
        details = parsed["authority.json"]["details"]
        authority = details if isinstance(details, dict) else {}
        if authority.get("status") != "pass":
            problems.append("authority proof status must be pass")
        if authority.get("present") is not True:
            problems.append("authority proof must be present")
        if authority.get("host_label") != EXPECTED_HOST:
            problems.append("authority host label must be sg01")
        collected_raw = authority.get("collected_at")
        collected = _parse_aware_utc(collected_raw) if isinstance(collected_raw, str) else None
        envelope_raw = parsed["authority.json"].get("collected_at")
        envelope = _parse_aware_utc(envelope_raw) if isinstance(envelope_raw, str) else None
        if collected is None:
            problems.append("authority collected_at must be an ISO-8601 timestamp with a UTC offset")
        else:
            if collected.strftime("%Y-%m-%d") != day:
                problems.append("authority collected_at must be on the evidence day")
            if envelope is not None and (
                collected - envelope
            ).total_seconds() > AUTHORITY_CLOCK_SKEW_ALLOWANCE:
                problems.append("authority collected_at must not be future-dated beyond the collection clock skew")
        if any(
            authority.get(key) is not False
            for key in ("tasker_active", "tasker_enabled", "archive_timer_active", "archive_timer_enabled")
        ):
            problems.append("authority former flags must all be false")
    return problems


def validate_day(root: Path, day: str) -> dict[str, Any]:
    """Structural, schema, summary and criterion checks for one day
    directory; read-only with lstat and bounded opens only."""
    day_path = root / day
    categories: dict[str, str] = {}
    problems: list[str] = []  # fail-grade reasons
    notices: list[str] = []  # warning-grade category notices
    try:
        st = day_path.lstat()
    except FileNotFoundError:
        return {"day": day, "status": "missing", "problems": [], "warnings": [], "categories": categories, "archive_sha": None}
    except OSError:
        return {"day": day, "status": "fail", "problems": ["day directory unreadable"], "warnings": [], "categories": categories, "archive_sha": None}
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        return {
            "day": day, "status": "fail",
            "problems": ["day directory must be a non-symlink directory"],
            "warnings": [], "categories": categories, "archive_sha": None,
        }
    if stat.S_IMODE(st.st_mode) != 0o700:
        problems.append("day directory must be exactly mode 0700")
    try:
        entries = set(os.listdir(day_path))
    except OSError:
        return {"day": day, "status": "fail", "problems": ["day directory unreadable"], "warnings": [], "categories": categories, "archive_sha": None}
    if entries != set(EXPECTED_FILES):
        problems.append("day directory must contain exactly the seven expected evidence files")

    parsed: dict[str, dict[str, Any]] = {}
    for file_name in sorted(EXPECTED_FILES):
        payload, file_reasons = _validate_evidence_file(day_path / file_name, file_name, day)
        problems.extend(file_reasons)
        if payload is None:
            continue
        parsed[file_name] = payload
        if file_name != "summary.json":
            categories[payload["category"]] = payload["status"]
            if payload["status"] == "fail":
                problems.append(f"{payload['category']} category is fail")
            elif payload["status"] == "warning":
                notices.append(f"{payload['category']} category is warning")

    if set(parsed) == set(EXPECTED_FILES):
        problems.extend(_summary_problems(parsed["summary.json"], categories, day))
        problems.extend(_criterion_problems(parsed, day))

    archive_sha: str | None = None
    archive_payload = parsed.get("archive.json")
    if archive_payload is not None:
        archive_details = archive_payload.get("details")
        sha_value = archive_details.get("sha256") if isinstance(archive_details, dict) else None
        if isinstance(sha_value, str) and SHA256_RE.fullmatch(sha_value):
            archive_sha = sha_value

    if problems:
        status = "fail"
    elif notices:
        status = "warning"
    else:
        status = "pass"
    return {
        "day": day, "status": status,
        "problems": problems, "warnings": notices,
        "categories": categories, "archive_sha": archive_sha,
    }


# ── recycle proof ─────────────────────────────────────────────────────────


def _recycle_problems(obj: dict[str, Any], window_start: str, window_end: str) -> list[str]:
    problems: list[str] = []
    if obj.get("schema") != SCHEMA_RECYCLE:
        problems.append("recycle proof schema mismatch")
    if obj.get("category") != "recycle":
        problems.append("recycle proof category mismatch")
    if obj.get("status") != "pass":
        problems.append("recycle proof status is not pass")
    details = obj.get("details")
    if not isinstance(details, dict):
        problems.append("recycle proof details missing")
        return problems
    start_dt = datetime.strptime(window_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(window_end, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    collected_raw = obj.get("collected_at")
    collected = _parse_aware_utc(collected_raw) if isinstance(collected_raw, str) else None
    if collected is None:
        problems.append("recycle collected_at must be an ISO-8601 timestamp with a UTC offset")
    elif not (start_dt <= collected < end_dt):
        problems.append("recycle collected_at is outside the seven-day window")
    performed_raw = details.get("performed_at")
    performed = _parse_aware_utc(performed_raw) if isinstance(performed_raw, str) else None
    if performed is None:
        problems.append("recycle performed_at must be an ISO-8601 timestamp with a UTC offset")
    elif not (start_dt <= performed < end_dt):
        problems.append("recycle performed_at is outside the seven-day window")
    elif collected is not None and collected.strftime("%Y-%m-%d") != performed.strftime("%Y-%m-%d"):
        problems.append("recycle collected_at must be on the same UTC day as performed_at")
    for phase in ("pre_recycle", "post_recycle"):
        phase_details = details.get(phase)
        if not isinstance(phase_details, dict):
            problems.append("recycle phase details missing")
            continue
        instances = phase_details.get("scheduler_instances")
        if type(instances) is not int or instances != 1:
            problems.append("recycle scheduler instance count is not exactly 1")
        if any(phase_details.get(component) != "pass" for component in RECYCLE_COMPONENTS):
            problems.append("recycle component must pass")
    return problems


def validate_recycle(root: Path, window_start: str, window_end: str) -> dict[str, Any]:
    """Optional attended recycle proof: proven | unproven | invalid."""
    path = root / "recycle.json"
    reasons: list[str] = []
    try:
        st = path.lstat()
    except FileNotFoundError:
        return {"present": False, "status": "unproven", "reasons": reasons}
    except OSError:
        return {"present": True, "status": "invalid", "reasons": ["recycle proof unreadable"]}
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        return {
            "present": True,
            "status": "invalid",
            "reasons": ["recycle proof must be a regular non-symlink file"],
        }
    if stat.S_IMODE(st.st_mode) != 0o600:
        reasons.append("recycle proof must be exactly mode 0600")
    data, exceeded, error = _read_bounded(path, MAX_EVIDENCE_BYTES)
    if error is not None:
        reasons.append(error)
    elif exceeded:
        reasons.append(f"recycle proof exceeds the {MAX_EVIDENCE_BYTES} byte bound")
    else:
        try:
            obj = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            reasons.append("recycle proof is not valid JSON")
        else:
            if not isinstance(obj, dict):
                reasons.append("recycle proof is not a JSON object")
            else:
                reasons.extend(_recycle_problems(obj, window_start, window_end))
    if reasons:
        return {"present": True, "status": "invalid", "reasons": reasons}
    return {"present": True, "status": "proven", "reasons": reasons}


# ── report assembly ───────────────────────────────────────────────────────


def _build_continuity(per_day: list[dict[str, Any]]) -> dict[str, Any]:
    days: list[dict[str, str]] = [
        {"day": day_report["day"], "sha256": day_report["archive_sha"]}
        for day_report in per_day
        if day_report["archive_sha"] is not None
    ]
    seen: set[str] = set()
    duplicate = False
    for entry in days:
        if entry["sha256"] in seen:
            duplicate = True
        seen.add(entry["sha256"])
    return {"days": days, "duplicate_sha": duplicate}


def assemble_report(
    cfg: Config,
    per_day: list[dict[str, Any]],
    recycle: dict[str, Any],
    root_issues: list[str],
    root_blockers: list[str],
) -> dict[str, Any]:
    blockers: list[str] = []
    extend_reasons: list[str] = []
    for day_report in per_day:
        day, status = day_report["day"], day_report["status"]
        if status == "fail":
            blockers.extend(f"day {day}: {problem}" for problem in day_report["problems"])
        elif status == "missing":
            extend_reasons.append(f"day {day} directory missing")
        if day_report["warnings"]:
            extend_reasons.extend(f"day {day}: {notice}" for notice in day_report["warnings"])
    blockers.extend(root_issues)
    blockers.extend(root_blockers)
    continuity = _build_continuity(per_day)
    if continuity["duplicate_sha"]:
        blockers.append("duplicate archive sha256 across the window")
    if recycle["status"] == "invalid":
        blockers.extend(f"recycle: {reason}" for reason in recycle["reasons"])

    notes: list[str] = []
    attended: list[str] = []
    if recycle["status"] == "proven":
        attended.append(
            "recycle persistence accepted as attended proof"
            " (pre/post scheduler_instances=1 and tasker/static/tunnel/api pass)"
        )
    elif cfg.require_recycle_proof:
        notes.append(
            "attended recycle proof required (--require-recycle-proof) but unproven; acceptance blocked"
        )
        attended.append("attended recycle proof is required before acceptance")
    else:
        notes.append("recycle proof absent: recycle persistence remains an attended unproven gate")
        attended.append("recycle persistence is an attended unproven gate; acceptance is ready_except_recycle")
    if recycle["status"] == "invalid":
        attended.append("recycle proof present but invalid: attended recycle exercise remains unproven")

    if blockers:
        verdict = "investigate"
    elif extend_reasons:
        verdict = "extend"
    elif recycle["status"] == "proven":
        verdict = "accept"
    elif cfg.require_recycle_proof:
        verdict = "extend"
    else:
        verdict = "ready_except_recycle"

    return {
        "schema": SCHEMA_REPORT,
        "window": {"start_day": cfg.start_day, "end_day": cfg.end_day, "days": WINDOW_DAYS},
        "days_present": sum(1 for day_report in per_day if day_report["status"] != "missing"),
        "per_day": [
            {
                "day": day_report["day"],
                "status": day_report["status"],
                "problems": day_report["problems"],
                "warnings": day_report["warnings"],
                "categories": day_report["categories"],
            }
            for day_report in per_day
        ],
        "archive_continuity": continuity,
        "recycle": recycle,
        "verdict": verdict,
        "blockers": blockers,
        "warnings": extend_reasons + notes,
        "attended_decisions": attended,
    }


# ── output ────────────────────────────────────────────────────────────────


def atomic_write(path: Path, data: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".plae-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_report(output_json: Path | None, report: dict[str, Any]) -> None:
    if output_json is None:
        return
    data = json.dumps(report, separators=(",", ":"), sort_keys=True).encode("utf-8")
    try:
        atomic_write(output_json, data)
    except OSError:
        die("failed to write report")  # static diagnostic; never echoes the path


# ── main ──────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    end_date = datetime.strptime(cfg.end_day, "%Y-%m-%d").date()
    day_names = [
        (end_date - timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range(WINDOW_DAYS - 1, -1, -1)
    ]

    root_blockers: list[str] = []
    try:
        root_st = cfg.evidence_root.lstat()
    except OSError:
        root_st = None  # not collected yet: day directories report missing -> extend
    if root_st is not None:
        if stat.S_ISLNK(root_st.st_mode):
            die("PLAE_EVIDENCE_ROOT must not be a symlink")
        if not stat.S_ISDIR(root_st.st_mode):
            die("PLAE_EVIDENCE_ROOT must be an existing directory")
        if stat.S_IMODE(root_st.st_mode) != 0o700:
            root_blockers.append("evidence root must be exactly mode 0700")  # validated, never mutated

    per_day = [validate_day(cfg.evidence_root, day) for day in day_names]

    root_issues: list[str] = []
    try:
        entries = set(os.listdir(cfg.evidence_root))
    except OSError:
        entries = set()
    for entry in sorted(entries):
        if entry == "recycle.json" or entry in set(day_names):
            continue
        if _is_calendar_day(entry):
            # Date-named dirs outside the window are historical evidence:
            # tolerated only as real non-symlink 0700 directories; contents
            # are not validated because they are outside the window.
            historical_path = cfg.evidence_root / entry
            try:
                historical_st = historical_path.lstat()
            except OSError:
                root_issues.append("historical day entry must be a regular non-symlink directory")
                continue
            if stat.S_ISLNK(historical_st.st_mode) or not stat.S_ISDIR(historical_st.st_mode):
                root_issues.append("historical day entry must be a regular non-symlink directory")
            elif stat.S_IMODE(historical_st.st_mode) != 0o700:
                root_issues.append("historical day entry must be exactly mode 0700")
            continue
        root_issues.append("unexpected entry in evidence root")

    recycle = validate_recycle(cfg.evidence_root, cfg.start_day, cfg.end_day)
    report = assemble_report(cfg, per_day, recycle, root_issues, root_blockers)
    write_report(cfg.output_json, report)
    emit(report)
    return 0 if report["verdict"] in ("accept", "ready_except_recycle") else 2


if __name__ == "__main__":
    sys.exit(main())