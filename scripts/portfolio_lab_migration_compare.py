#!/usr/bin/env python3
"""Portfolio Lab migration comparison: deterministic, redacted comparison CLI.

Consumes immutable redacted evidence manifests from source (sg01) and
candidate (cursor-box), validates them strictly, compares them deterministically,
and emits stable JSON and Markdown reports. Standard library only.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA_EVIDENCE = "portfolio-lab-migration-evidence/v1"
SCHEMA_EXPLANATIONS = "portfolio-lab-migration-explanations/v1"
SCHEMA_COMPARISON = "portfolio-lab-migration-comparison/v1"
SCHEMA_RELEASE = "portfolio-lab-static-release/v1"

HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
PATH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
MEDIA_TYPE_RE = re.compile(r"^[A-Za-z0-9!#$%&'*+.^_`|~-]+/[A-Za-z0-9!#$%&'*+.^_`|~-]+$")

MAX_NAME_LEN = 200
MAX_CHECK_ID_LEN = 300
MAX_RESPONSE_BYTES = 1048576
MAX_FRESHNESS_DELTA_BOUND = 86400.0

FORBIDDEN_KEY_PATTERNS = (
    "password",
    "token",
    "secret",
    "cookie",
    "credential",
    "private_key",
)


def check_no_sensitive_key(key: str) -> None:
    """Reject sensitive keywords in keys."""
    if not isinstance(key, str):
        die_diagnostic("invalid evidence key")
    key_lower = key.lower()
    for s in FORBIDDEN_KEY_PATTERNS:
        if s in key_lower:
            die_diagnostic("sensitive key detected in evidence")
    if "auth" in key_lower:
        cleaned = key_lower.replace("authoritative", "").replace("authority", "")
        if "auth" in cleaned:
            die_diagnostic("sensitive key detected in evidence")


def check_all_keys_recursive(obj: Any) -> None:
    """Recursively ensure no sensitive keys exist anywhere in the structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            check_no_sensitive_key(k)
            check_all_keys_recursive(v)
    elif isinstance(obj, list):
        for item in obj:
            check_all_keys_recursive(item)

UNEXPLAINABLE_PREFIXES = (
    "recovery.",
    "sqlite.integrity.",
    "allocation.",
    "git.",
    "digests.",
    "release.",
    "tasker.scheduler_mode",
    "tasker.scheduler_instances",
    "tasker.scheduler_env_disabled",
    "tasker.scheduler_arg_disabled",
    "tasker.scheduled_starts_observed",
    "authority.authoritative",
    "authority.healthy",
    "authority.public_origin_loopback_only",
    "authority.access_protected",
)


def is_unexplainable(check_id: str) -> bool:
    """Check if check_id is in the unexplainable category."""
    if check_id.startswith("endpoints.") and check_id.endswith(".status"):
        return True
    return any(check_id.startswith(p) for p in UNEXPLAINABLE_PREFIXES)

CHAMPION_ALLOCATION = {
    "SPY": 0.46,
    "GLD": 0.38,
    "TLT": 0.16,
}

DIMENSIONS = (
    "git",
    "recovery",
    "sqlite",
    "digests",
    "release",
    "allocation",
    "safety",
    "tasker",
    "schemas",
    "freshness",
    "endpoints",
    "authority",
    "explanations",
)


def die_diagnostic(msg: str) -> None:
    """Exit 1 with a bounded diagnostic and no traceback."""
    sys.stderr.write(f"error: {msg}\n")
    sys.exit(1)


def strict_json_loads(text: str) -> Any:
    """Parse JSON without duplicate keys or non-standard numeric constants."""
    def reject_constant(value: str) -> Any:
        raise ValueError("non-finite JSON constant")

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_pairs,
    )


def canonical_json_bytes(val: Any) -> bytes:
    return json.dumps(val, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_fingerprint(val: Any) -> str:
    """Compute SHA-256 fingerprint of canonical JSON serialization."""
    return hashlib.sha256(canonical_json_bytes(val)).hexdigest()





def _validate_text(value: Any, max_bytes: int, what: str) -> str:
    """Validate bounded UTF-8 text and reject C0/C1/DEL controls."""
    if not isinstance(value, str) or not value:
        die_diagnostic(f"invalid {what}")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError:
        die_diagnostic(f"invalid UTF-8 {what}")
    if len(encoded) > max_bytes:
        die_diagnostic(f"excessive {what} length")
    if any(ord(ch) < 32 or 127 <= ord(ch) <= 159 for ch in value):
        die_diagnostic(f"control character in {what}")
    return value


def validate_simple_identifier(value: Any, what: str = "logical name") -> str:
    """Validate one safe non-path identifier."""
    text = _validate_text(value, MAX_NAME_LEN, what)
    if text in {".", ".."} or ".." in text:
        die_diagnostic(f"unsafe {what}")
    if not SAFE_IDENTIFIER_RE.fullmatch(text):
        die_diagnostic(f"unsafe {what}")
    check_no_sensitive_key(text)
    return text


def _valid_path_component(component: str) -> bool:
    """Return whether a logical path component is nonempty and safe."""
    if not component or component in {".", ".."}:
        return False
    forbidden = {"/", "\\", ":", "@", "=", "?", "#"}
    return not any(ch in forbidden or ch.isspace() or ord(ch) < 32 or 127 <= ord(ch) <= 159 for ch in component)


def validate_relative_logical_path(value: Any, what: str = "logical path") -> str:
    """Validate a bounded UTF-8 relative logical path used in evidence maps."""
    text = _validate_text(value, MAX_NAME_LEN, what)
    if text.startswith("/") or text.endswith("/") or "\\" in text:
        die_diagnostic(f"unsafe {what}")
    components = text.split("/")
    if any(not _valid_path_component(component) for component in components):
        die_diagnostic(f"unsafe {what}")
    check_no_sensitive_key(text)
    return text


def validate_endpoint_path(value: Any, what: str = "endpoint path") -> str:
    """Validate an absolute URL path without authority, query, or fragment."""
    text = _validate_text(value, MAX_NAME_LEN, what)
    if text == "/":
        return text
    if not text.startswith("/") or text.startswith("//"):
        die_diagnostic(f"unsafe {what}")
    if any(ch in text for ch in ("\\", "?", "#", "@", "=", ":")):
        die_diagnostic(f"unsafe {what}")
    components = text[1:].split("/")
    if any(not PATH_COMPONENT_RE.fullmatch(component) or component in {".", ".."} for component in components):
        die_diagnostic(f"unsafe {what}")
    return text


def validate_schema_string(value: Any, what: str = "schema string") -> str:
    """Validate a schema identifier made of nonempty relative path components."""
    return validate_relative_logical_path(value, what)


def validate_check_id(value: Any) -> str:
    """Validate a generated comparison check identifier, not an arbitrary path."""
    text = _validate_text(value, MAX_CHECK_ID_LEN, "check ID")
    if text.startswith("/") or "\\" in text or any(ch in text for ch in ("?", "#", "@", "=", ":")):
        die_diagnostic("unsafe check ID")
    if ".." in text or "://" in text:
        die_diagnostic("unsafe check ID")
    if any(ch.isspace() for ch in text):
        die_diagnostic("unsafe check ID")

    def known_simple(prefix: str, fields: set[str]) -> bool:
        return text.startswith(prefix) and text[len(prefix):] in fields

    if known_simple("git.", {"commit", "bundle_source_commit"}):
        return text
    if known_simple("recovery.", {"archive_sha256", "sidecar_ok", "archive_verified", "bundle_verified"}):
        return text
    if known_simple("release.", {"schema_version", "source_git_sha", "manifest_sha256"}):
        return text
    if known_simple("allocation.", {"SPY", "GLD", "TLT"}):
        return text
    if known_simple("safety.kill_switch.", {"enabled", "level", "incident_id"}):
        return text
    if text == "safety.open_incidents":
        return text
    if known_simple("tasker.", {
        "registry_sha256", "scheduler_mode", "scheduler_instances",
        "scheduler_env_disabled", "scheduler_arg_disabled",
        "scheduled_starts_observed", "status_schema",
    }):
        return text
    if known_simple("authority.", {
        "authoritative", "healthy", "access_protected", "public_origin_loopback_only",
    }):
        return text

    for prefix in (
        "sqlite.integrity.",
        "sqlite.counts.",
        "schemas.",
        "digests.static.",
        "digests.runtime.",
    ):
        if text.startswith(prefix):
            suffix = text[len(prefix):]
            if prefix.startswith("sqlite."):
                validate_simple_identifier(suffix, "check ID field")
            else:
                validate_relative_logical_path(suffix, "check ID path")
            return text

    if text.startswith("freshness."):
        suffix = text[len("freshness."):]
        try:
            path, field = suffix.rsplit(".", 1)
        except ValueError:
            die_diagnostic("unknown check ID")
        if field not in {"generated_at", "age_seconds", "max_age_seconds"}:
            die_diagnostic("unknown check ID")
        validate_relative_logical_path(path, "check ID path")
        return text

    if text.startswith("endpoints."):
        suffix = text[len("endpoints."):]
        try:
            route, field = suffix.rsplit(".", 1)
        except ValueError:
            die_diagnostic("unknown check ID")
        if field not in {"status", "content_type", "schema_version", "body_sha256"}:
            die_diagnostic("unknown check ID")
        validate_endpoint_path(route, "check ID endpoint")
        return text

    die_diagnostic("unknown check ID")
    return text


def check_safe_name(name: Any, allow_slash: bool = False) -> None:
    """Compatibility wrapper for simple logical names in fixed scalar fields."""
    if allow_slash:
        validate_relative_logical_path(name)
    else:
        validate_simple_identifier(name)


def validate_iso8601_tz(dt_str: Any) -> datetime.datetime:
    """Validate timezone-aware ISO-8601 timestamp."""
    if not isinstance(dt_str, str) or not dt_str or len(dt_str) > 64:
        die_diagnostic("invalid timestamp format")
    try:
        dt = datetime.datetime.fromisoformat(dt_str)
    except Exception:
        die_diagnostic("invalid ISO-8601 timestamp")
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        die_diagnostic("timestamp must be timezone-aware")
    return dt


def validate_evidence(ev: Any, expected_role: str, expected_host: str) -> dict[str, Any]:
    """Strictly validate evidence dictionary structure and types."""
    if not isinstance(ev, dict):
        die_diagnostic("evidence must be a JSON object")

    expected_top_keys = {
        "schema_version",
        "role",
        "host",
        "collected_at",
        "git",
        "recovery",
        "sqlite",
        "digests",
        "release",
        "allocation",
        "safety",
        "tasker",
        "schemas",
        "freshness",
        "endpoints",
        "authority",
    }
    if set(ev.keys()) != expected_top_keys:
        die_diagnostic("unexpected or missing top-level evidence keys")

    check_all_keys_recursive(ev)

    if ev["schema_version"] != SCHEMA_EVIDENCE:
        die_diagnostic("invalid evidence schema_version")
    if not isinstance(ev["role"], str) or ev["role"] != expected_role:
        die_diagnostic("invalid evidence role")
    if not isinstance(ev["host"], str) or ev["host"] != expected_host:
        die_diagnostic("invalid evidence host")

    validate_iso8601_tz(ev["collected_at"])

    # git
    git = ev["git"]
    if not isinstance(git, dict) or set(git.keys()) != {"commit", "bundle_source_commit"}:
        die_diagnostic("invalid git structure")
    if not isinstance(git["commit"], str) or not HEX_40_RE.match(git["commit"]):
        die_diagnostic("invalid git commit hex")
    if not isinstance(git["bundle_source_commit"], str) or not HEX_40_RE.match(git["bundle_source_commit"]):
        die_diagnostic("invalid git bundle_source_commit hex")
    if git["commit"] != git["bundle_source_commit"]:
        die_diagnostic("git commit and bundle_source_commit must match within host")

    # recovery
    rec = ev["recovery"]
    if not isinstance(rec, dict) or set(rec.keys()) != {"archive_sha256", "sidecar_ok", "archive_verified", "bundle_verified"}:
        die_diagnostic("invalid recovery structure")
    if not isinstance(rec["archive_sha256"], str) or not HEX_64_RE.match(rec["archive_sha256"]):
        die_diagnostic("invalid recovery archive_sha256 hex")
    for b in ("sidecar_ok", "archive_verified", "bundle_verified"):
        if type(rec[b]) is not bool:
            die_diagnostic(f"recovery {b} must be boolean")

    # sqlite
    sql = ev["sqlite"]
    if not isinstance(sql, dict) or set(sql.keys()) != {"integrity", "counts"}:
        die_diagnostic("invalid sqlite structure")
    if not isinstance(sql["integrity"], dict) or not sql["integrity"] or len(sql["integrity"]) > 100:
        die_diagnostic("invalid sqlite integrity map")
    for k, v in sql["integrity"].items():
        validate_simple_identifier(k, "sqlite logical name")
        if not isinstance(v, str) or len(v) > 50:
            die_diagnostic("invalid sqlite integrity value")
    if not isinstance(sql["counts"], dict) or not sql["counts"] or len(sql["counts"]) > 100:
        die_diagnostic("invalid sqlite counts map")
    for k, v in sql["counts"].items():
        validate_simple_identifier(k, "sqlite logical name")
        if type(v) is not int or v < 0:
            die_diagnostic("sqlite count must be non-negative integer")

    # digests
    dig = ev["digests"]
    if not isinstance(dig, dict) or set(dig.keys()) != {"static", "runtime"}:
        die_diagnostic("invalid digests structure")
    for sub in ("static", "runtime"):
        m = dig[sub]
        if not isinstance(m, dict) or not m or len(m) > 500:
            die_diagnostic(f"invalid digests {sub} map")
        for k, v in m.items():
            validate_relative_logical_path(k, f"{sub} digest name")
            if not isinstance(v, str) or not HEX_64_RE.fullmatch(v):
                die_diagnostic(f"invalid digest sha256 for {sub}")

    # release
    rel = ev["release"]
    if not isinstance(rel, dict) or set(rel.keys()) != {"schema_version", "source_git_sha", "manifest_sha256"}:
        die_diagnostic("invalid release structure")
    if rel["schema_version"] != SCHEMA_RELEASE:
        die_diagnostic("invalid release schema_version")
    if not isinstance(rel["source_git_sha"], str) or not HEX_40_RE.match(rel["source_git_sha"]):
        die_diagnostic("invalid release source_git_sha")
    if rel["source_git_sha"] != git["commit"]:
        die_diagnostic("release source_git_sha must match git commit")
    if not isinstance(rel["manifest_sha256"], str) or not HEX_64_RE.match(rel["manifest_sha256"]):
        die_diagnostic("invalid release manifest_sha256")

    # allocation
    alloc = ev["allocation"]
    if not isinstance(alloc, dict) or set(alloc.keys()) != {"SPY", "GLD", "TLT"}:
        die_diagnostic("allocation must contain exactly SPY, GLD, TLT")
    for k, v in alloc.items():
        if type(v) not in (int, float) or type(v) is bool or not math.isfinite(v):
            die_diagnostic("allocation values must be finite numeric")

    # safety
    safe = ev["safety"]
    if not isinstance(safe, dict) or set(safe.keys()) != {"kill_switch", "open_incidents"}:
        die_diagnostic("invalid safety structure")
    ks = safe["kill_switch"]
    if not isinstance(ks, dict) or set(ks.keys()) != {"enabled", "level", "incident_id"}:
        die_diagnostic("invalid kill_switch structure")
    if type(ks["enabled"]) is not bool:
        die_diagnostic("kill_switch enabled must be boolean")
    if not isinstance(ks["level"], str) or len(ks["level"]) > 64:
        die_diagnostic("invalid kill_switch level")
    check_safe_name(ks["level"])
    if ks["incident_id"] is not None:
        check_safe_name(ks["incident_id"])
    incidents = safe["open_incidents"]
    if not isinstance(incidents, list) or len(incidents) > 100:
        die_diagnostic("open_incidents must be a bounded list")
    for inc in incidents:
        if not isinstance(inc, dict) or set(inc.keys()) != {"id", "channel", "severity", "state"}:
            die_diagnostic("invalid incident item structure")
        for f in ("id", "channel", "severity", "state"):
            check_safe_name(inc[f])

    # tasker
    tsk = ev["tasker"]
    tasker_keys = {
        "registry_sha256",
        "scheduler_mode",
        "scheduler_instances",
        "scheduler_env_disabled",
        "scheduler_arg_disabled",
        "scheduled_starts_observed",
        "status_schema",
    }
    if not isinstance(tsk, dict) or set(tsk.keys()) != tasker_keys:
        die_diagnostic("invalid tasker structure")
    if not isinstance(tsk["registry_sha256"], str) or not HEX_64_RE.match(tsk["registry_sha256"]):
        die_diagnostic("invalid tasker registry_sha256")
    if not isinstance(tsk["scheduler_mode"], str) or len(tsk["scheduler_mode"]) > 64:
        die_diagnostic("invalid tasker scheduler_mode")
    check_safe_name(tsk["scheduler_mode"])
    if type(tsk["scheduler_instances"]) is not int or tsk["scheduler_instances"] < 0:
        die_diagnostic("tasker scheduler_instances must be nonnegative int")
    if type(tsk["scheduler_env_disabled"]) is not bool:
        die_diagnostic("tasker scheduler_env_disabled must be bool")
    if type(tsk["scheduler_arg_disabled"]) is not bool:
        die_diagnostic("tasker scheduler_arg_disabled must be bool")
    if type(tsk["scheduled_starts_observed"]) is not int or tsk["scheduled_starts_observed"] < 0:
        die_diagnostic("tasker scheduled_starts_observed must be nonnegative int")
    if not isinstance(tsk["status_schema"], str) or len(tsk["status_schema"]) > 100:
        die_diagnostic("invalid tasker status_schema")
    validate_schema_string(tsk["status_schema"], "tasker status_schema")

    # schemas
    sch = ev["schemas"]
    if not isinstance(sch, dict) or not sch or len(sch) > 100:
        die_diagnostic("schemas must be non-empty bounded dict")
    for k, v in sch.items():
        validate_relative_logical_path(k, "schema artifact name")
        validate_schema_string(v)

    # freshness
    fresh = ev["freshness"]
    if not isinstance(fresh, dict) or not fresh or len(fresh) > 100:
        die_diagnostic("freshness must be non-empty bounded dict")
    for k, v in fresh.items():
        validate_relative_logical_path(k, "freshness artifact name")
        if not isinstance(v, dict) or set(v.keys()) != {"generated_at", "age_seconds", "max_age_seconds"}:
            die_diagnostic("invalid freshness item structure")
        validate_iso8601_tz(v["generated_at"])
        for age_key in ("age_seconds", "max_age_seconds"):
            val = v[age_key]
            if type(val) not in (int, float) or type(val) is bool or not math.isfinite(val) or val < 0:
                die_diagnostic(f"freshness {age_key} must be finite nonnegative number")
        if v["age_seconds"] > v["max_age_seconds"]:
            die_diagnostic("freshness age_seconds cannot exceed max_age_seconds")

    # endpoints
    ep = ev["endpoints"]
    if not isinstance(ep, dict) or not ep or len(ep) > 100:
        die_diagnostic("endpoints must be a non-empty bounded map")
    for k, v in ep.items():
        validate_endpoint_path(k)
        if not isinstance(v, dict) or set(v.keys()) != {"status", "content_type", "schema_version", "body_sha256"}:
            die_diagnostic("invalid endpoint item structure")
        if type(v["status"]) is not int:
            die_diagnostic("endpoint status must be integer")
        if not isinstance(v["content_type"], str):
            die_diagnostic("invalid endpoint content_type")
        if ";" in v["content_type"] or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in v["content_type"]):
            die_diagnostic("invalid endpoint content_type")
        if not MEDIA_TYPE_RE.fullmatch(v["content_type"]):
            die_diagnostic("invalid endpoint content_type")
        if len(v["content_type"]) > 100:
            die_diagnostic("invalid endpoint content_type")
        is_html = v["content_type"].lower() == "text/html"
        if not is_html and v["schema_version"] is None:
            die_diagnostic("endpoint schema_version required for non-HTML endpoints")
        if v["schema_version"] is not None:
            if not isinstance(v["schema_version"], str):
                die_diagnostic("endpoint schema_version must be string or null")
            validate_schema_string(v["schema_version"], "endpoint schema_version")
        if k != "/api/tasker/status" and v["body_sha256"] is None:
            die_diagnostic("endpoint body_sha256 cannot be null")
        if v["body_sha256"] is not None:
            if not isinstance(v["body_sha256"], str) or not HEX_64_RE.fullmatch(v["body_sha256"]):
                die_diagnostic("invalid endpoint body_sha256")
        if k == "/api/tasker/status":
            if v["body_sha256"] is not None:
                die_diagnostic("/api/tasker/status body_sha256 must be null")

    # authority
    auth = ev["authority"]
    auth_keys = {"authoritative", "healthy", "access_protected", "public_origin_loopback_only"}
    if not isinstance(auth, dict) or set(auth.keys()) != auth_keys:
        die_diagnostic("invalid authority structure")
    for k in auth_keys:
        if type(auth[k]) is not bool:
            die_diagnostic(f"authority {k} must be boolean")

    return ev


def valid_json_media_type(value: str) -> bool:
    """Accept only application/json media type, case-insensitively, with parameters after semicolon."""
    media_type = value.split(";", 1)[0].strip()
    return media_type.lower() == "application/json"


def load_from_url(url: str) -> dict[str, Any]:
    """Fetch JSON from a strictly validated HTTPS URL."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        die_diagnostic("malformed URL")

    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        die_diagnostic("invalid HTTPS URL")
    if parsed.path == "" or parsed.path.endswith("/"):
        die_diagnostic("invalid HTTPS URL")
    if not parsed.hostname or parsed.params:
        die_diagnostic("invalid HTTPS URL")

    envelope: dict[str, Any]
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "portfolio-lab-migration-compare/1.0", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read(MAX_RESPONSE_BYTES + 1)
            envelope = {
                "final_url": resp.geturl(),
                "content_type": resp.headers.get("Content-Type"),
                "body": raw,
            }
    except Exception:
        die_diagnostic("failed to fetch evidence from URL")

    if not isinstance(envelope["final_url"], str):
        die_diagnostic("fetch response envelope invalid")
    if not isinstance(envelope["content_type"], str):
        die_diagnostic("fetch response envelope invalid")
    if not isinstance(envelope["body"], bytes):
        die_diagnostic("fetch response envelope invalid")
    if envelope["final_url"] != url:
        die_diagnostic("URL redirect not permitted")
    if not valid_json_media_type(envelope["content_type"]):
        die_diagnostic("URL content-type must be application/json")
    body = envelope["body"]
    if len(body) > MAX_RESPONSE_BYTES:
        die_diagnostic("URL response exceeded 1 MiB limit")
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeError:
        die_diagnostic("URL response is not valid UTF-8")
    try:
        data = strict_json_loads(text)
    except Exception:
        die_diagnostic("URL response is not valid JSON")
    return data


def load_evidence(source_spec: str, role: str, host: str) -> dict[str, Any]:
    """Load and validate evidence from directory, direct file, or HTTPS URL."""
    if source_spec.startswith("https://") or source_spec.startswith("http://"):
        raw_data = load_from_url(source_spec)
        return validate_evidence(raw_data, role, host)

    path = Path(source_spec)
    if path.is_dir():
        target = path / "evidence.json"
        if not target.is_file():
            die_diagnostic("evidence.json not found in directory")
        try:
            raw_data = strict_json_loads(target.read_text(encoding="utf-8"))
        except Exception:
            die_diagnostic("failed to read or parse evidence.json")
        return validate_evidence(raw_data, role, host)

    if path.is_file():
        try:
            raw_data = strict_json_loads(path.read_text(encoding="utf-8"))
        except Exception:
            die_diagnostic("failed to read or parse evidence file")
        return validate_evidence(raw_data, role, host)

    die_diagnostic("source/candidate path does not exist or is invalid")


def validate_explanations(path_str: str | None) -> dict[str, dict[str, str]]:
    """Validate and index explanations file."""
    if not path_str:
        return {}
    p = Path(path_str)
    if not p.is_file():
        die_diagnostic("explanations file missing or not a regular file")
    try:
        data = strict_json_loads(p.read_text(encoding="utf-8"))
    except Exception:
        die_diagnostic("failed to read or parse explanations file")

    if isinstance(data, dict) and set(data.keys()) == {"schema_version", "entries"}:
        check_all_keys_recursive(data)
    else:
        die_diagnostic("invalid explanations structure")
    if data["schema_version"] != SCHEMA_EXPLANATIONS:
        die_diagnostic("invalid explanations schema_version")
    entries = data["entries"]
    if not isinstance(entries, list) or len(entries) > 100:
        die_diagnostic("explanations entries must be a bounded list")

    indexed: dict[str, dict[str, str]] = {}
    entry_keys = {"check_id", "source_fingerprint", "candidate_fingerprint", "reason"}

    for item in entries:
        if not isinstance(item, dict) or set(item.keys()) != entry_keys:
            die_diagnostic("invalid explanation entry keys")
        check_all_keys_recursive(item)
        cid = item["check_id"]
        validate_check_id(cid)
        if cid in indexed:
            die_diagnostic("duplicate check_id in explanations")

        s_fp = item["source_fingerprint"]
        c_fp = item["candidate_fingerprint"]
        if not isinstance(s_fp, str) or not HEX_64_RE.fullmatch(s_fp):
            die_diagnostic("invalid source_fingerprint in explanation")
        if not isinstance(c_fp, str) or not HEX_64_RE.fullmatch(c_fp):
            die_diagnostic("invalid candidate_fingerprint in explanation")

        reason = item["reason"]
        if not isinstance(reason, str) or not (1 <= len(reason) <= 200):
            die_diagnostic("explanation reason must be 1-200 characters")
        try:
            encoded_reason = reason.encode("utf-8", errors="strict")
        except UnicodeError:
            die_diagnostic("invalid UTF-8 explanation reason")
        if len(encoded_reason) > 200:
            die_diagnostic("explanation reason must be 1-200 characters")
        if any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in reason):
            die_diagnostic("control characters in explanation reason")
        if any(c in reason for c in ("=", "\\", ":", "@")):
            die_diagnostic("forbidden characters in explanation reason")
        if "http://" in reason or "https://" in reason or "/" in reason:
            die_diagnostic("URLs or paths forbidden in explanation reason")
        check_no_sensitive_key(reason)

        indexed[cid] = {
            "source_fingerprint": s_fp,
            "candidate_fingerprint": c_fp,
            "reason": reason,
        }
    return indexed


def _validate_output_target(target: Path) -> tuple[bool, int | None]:
    """Validate one output parent and destination without following symlinks."""
    parent = target.parent
    try:
        parent_stat = parent.lstat()
    except OSError:
        die_diagnostic("target output directory does not exist")
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        die_diagnostic("target output directory does not exist")
    try:
        target_stat = target.lstat()
    except FileNotFoundError:
        return False, None
    except OSError:
        die_diagnostic("output path cannot be inspected")
    if stat.S_ISLNK(target_stat.st_mode):
        die_diagnostic("output path cannot be a symlink")
    if not stat.S_ISREG(target_stat.st_mode):
        die_diagnostic("output path must be a regular file")
    return True, stat.S_IMODE(target_stat.st_mode)


def _safe_unlink(path: Path | None) -> bool:
    """Remove a staging artifact when present."""
    if path is None:
        return True
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _cleanup_paths(paths: list[Path | None]) -> bool:
    """Remove all known staging artifacts."""
    ok = True
    for path in paths:
        ok = _safe_unlink(path) and ok
    return ok


def write_outputs_atomically(
    json_target: Path, json_content: str, md_target: Path, md_content: str
) -> None:
    """Publish both reports with rollback-safe sibling staging and backups."""
    targets = (json_target, md_target)
    contents = (json_content, md_content)

    if json_target.resolve(strict=False) == md_target.resolve(strict=False):
        die_diagnostic("output-json and output-markdown paths must be distinct")

    prior = [_validate_output_target(target) for target in targets]
    temp_files: list[Path | None] = [None, None]
    backup_files: list[Path | None] = [None, None]
    installed = [False, False]

    try:
        for index, (target, content) in enumerate(zip(targets, contents)):
            fd, temp_name = tempfile.mkstemp(prefix=".tmp-", dir=target.parent)
            temp_path = Path(temp_name)
            temp_files[index] = temp_path
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(content)
                    handle.flush()
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise

            existed, old_mode = prior[index]
            if existed:
                backup_fd, backup_name = tempfile.mkstemp(prefix=".bak-", dir=target.parent)
                backup_path = Path(backup_name)
                backup_files[index] = backup_path
                os.close(backup_fd)
                shutil.copyfile(target, backup_path, follow_symlinks=False)
                os.chmod(backup_path, old_mode if old_mode is not None else 0o600)

        for index, target in enumerate(targets):
            os.replace(temp_files[index], target)
            installed[index] = True
    except Exception:
        rollback_ok = True
        for index in range(len(targets) - 1, -1, -1):
            target = targets[index]
            existed, _ = prior[index]
            if installed[index]:
                if existed:
                    backup = backup_files[index]
                    if backup is None:
                        rollback_ok = False
                    else:
                        try:
                            os.replace(backup, target)
                            backup_files[index] = None
                        except OSError:
                            rollback_ok = False
                elif not _safe_unlink(target):
                    rollback_ok = False
        cleanup_ok = _cleanup_paths(temp_files + backup_files)
        if not rollback_ok:
            die_diagnostic("failed to roll back output files")
        if not cleanup_ok:
            die_diagnostic("failed to clean output staging")
        die_diagnostic("failed to publish output files atomically")
    else:
        if not _cleanup_paths(temp_files + backup_files):
            die_diagnostic("failed to clean output staging")


def markdown_escape(text: str) -> str:
    """Escape Markdown special characters to prevent table or heading injection."""
    text = text.replace("|", "\\|")
    text = text.replace("#", "\\#")
    text = text.replace("`", "\\`")
    return text


def safe_expected_scheduler_pair(src: dict[str, Any], cand: dict[str, Any]) -> bool:
    """Require all scheduler safety fields to be exact safe states."""
    source_tasker = src["tasker"]
    candidate_tasker = cand["tasker"]
    return (
        source_tasker["scheduler_mode"] == "enabled"
        and source_tasker["scheduler_instances"] == 1
        and source_tasker["scheduler_env_disabled"] is False
        and source_tasker["scheduler_arg_disabled"] is False
        and candidate_tasker["scheduler_mode"] == "disabled"
        and candidate_tasker["scheduler_instances"] == 0
        and candidate_tasker["scheduler_env_disabled"] is True
        and candidate_tasker["scheduler_arg_disabled"] is True
        and candidate_tasker["scheduled_starts_observed"] == 0
    )


def safe_expected_authority_pair(src: dict[str, Any], cand: dict[str, Any]) -> bool:
    """Require all authority safety fields to be exact safe states."""
    source_authority = src["authority"]
    candidate_authority = cand["authority"]
    return (
        source_authority["authoritative"] is True
        and source_authority["healthy"] is True
        and source_authority["access_protected"] is False
        and source_authority["public_origin_loopback_only"] is True
        and candidate_authority["authoritative"] is False
        and candidate_authority["healthy"] is True
        and candidate_authority["access_protected"] is True
        and candidate_authority["public_origin_loopback_only"] is True
    )


def _record_blocking_state(
    checks: list[dict[str, Any]],
    differences: list[dict[str, Any]],
    cid: str,
    dim: str,
    s_val: Any,
    c_val: Any,
    reason: str,
) -> None:
    """Record a redacted blocking state for a safety-gated asymmetry."""
    checks.append({"check_id": cid, "dimension": dim, "status": "different"})
    differences.append({
        "check_id": cid,
        "dimension": dim,
        "classification": "blocking",
        "source_fingerprint": canonical_fingerprint(s_val),
        "candidate_fingerprint": canonical_fingerprint(c_val),
        "reason": reason,
    })


def run_comparison(
    src: dict[str, Any],
    cand: dict[str, Any],
    explanations: dict[str, dict[str, str]],
    max_freshness_delta: float,
) -> tuple[dict[str, Any], str, int]:
    """Perform comparison and build JSON and Markdown representations."""
    checks: list[dict[str, Any]] = []
    differences: list[dict[str, Any]] = []
    used_explanations: set[str] = set()

    def record_check(
        cid: str,
        dim: str,
        s_val: Any,
        c_val: Any,
        expected_diff_reason: str | None = None,
    ) -> None:
        s_fp = canonical_fingerprint(s_val)
        c_fp = canonical_fingerprint(c_val)

        # Exact match is safe for ordinary equality checks only.
        if expected_diff_reason is None and (
            s_val == c_val
            or (isinstance(s_val, float) and isinstance(c_val, float) and math.isclose(s_val, c_val, abs_tol=1e-9))
        ):
            checks.append({"check_id": cid, "dimension": dim, "status": "match"})
            return

        if expected_diff_reason is not None:
            checks.append({"check_id": cid, "dimension": dim, "status": "different"})
            differences.append({
                "check_id": cid,
                "dimension": dim,
                "classification": "expected",
                "source_fingerprint": s_fp,
                "candidate_fingerprint": c_fp,
                "reason": expected_diff_reason,
            })
            return

        # Check if an explanation applies
        if cid in explanations and not is_unexplainable(cid):
            exp = explanations[cid]
            if exp["source_fingerprint"] == s_fp and exp["candidate_fingerprint"] == c_fp:
                used_explanations.add(cid)
                checks.append({"check_id": cid, "dimension": dim, "status": "different"})
                differences.append({
                    "check_id": cid,
                    "dimension": dim,
                    "classification": "explained",
                    "source_fingerprint": s_fp,
                    "candidate_fingerprint": c_fp,
                    "reason": exp["reason"],
                })
                return

        # Blocking difference
        checks.append({"check_id": cid, "dimension": dim, "status": "different"})
        differences.append({
            "check_id": cid,
            "dimension": dim,
            "classification": "blocking",
            "source_fingerprint": s_fp,
            "candidate_fingerprint": c_fp,
            "reason": f"value mismatch in {cid}",
        })

    def record_asymmetry(
        cid: str,
        dim: str,
        s_val: Any,
        c_val: Any,
        s_expected: Any,
        c_expected: Any,
        expected_reason: str,
        is_safe_pair: bool,
    ) -> None:
        """Record expected asymmetry only for the exact expected state pair; all other states block."""
        if is_safe_pair and s_val == s_expected and c_val == c_expected and s_val != c_val:
            s_fp = canonical_fingerprint(s_val)
            c_fp = canonical_fingerprint(c_val)
            checks.append({"check_id": cid, "dimension": dim, "status": "different"})
            differences.append({
                "check_id": cid,
                "dimension": dim,
                "classification": "expected",
                "source_fingerprint": s_fp,
                "candidate_fingerprint": c_fp,
                "reason": expected_reason,
            })
        elif s_val != c_val:
            record_check(cid, dim, s_val, c_val)
        else:
            _record_blocking_state(
                checks,
                differences,
                cid,
                dim,
                s_val,
                c_val,
                f"unexpected state for {cid}",
            )

    # 1. git
    record_check("git.commit", "git", src["git"]["commit"], cand["git"]["commit"])
    record_check("git.bundle_source_commit", "git", src["git"]["bundle_source_commit"], cand["git"]["bundle_source_commit"])

    # 2. recovery
    for k in ("archive_sha256", "sidecar_ok", "archive_verified", "bundle_verified"):
        cid = f"recovery.{k}"
        s_val = src["recovery"][k]
        c_val = cand["recovery"][k]
        # In recovery, both must be true for booleans
        if k != "archive_sha256" and (not s_val or not c_val):
            s_fp = canonical_fingerprint(s_val)
            c_fp = canonical_fingerprint(c_val)
            checks.append({"check_id": cid, "dimension": "recovery", "status": "different"})
            differences.append({
                "check_id": cid,
                "dimension": "recovery",
                "classification": "blocking",
                "source_fingerprint": s_fp,
                "candidate_fingerprint": c_fp,
                "reason": f"recovery verification failed on {cid}",
            })
        else:
            record_check(cid, "recovery", s_val, c_val)

    # 3. sqlite
    # Integrity
    s_keys = set(src["sqlite"]["integrity"].keys())
    c_keys = set(cand["sqlite"]["integrity"].keys())
    for k in sorted(s_keys | c_keys):
        cid = f"sqlite.integrity.{k}"
        s_val = src["sqlite"]["integrity"].get(k)
        c_val = cand["sqlite"]["integrity"].get(k)
        if s_val is None or c_val is None:
            checks.append({"check_id": cid, "dimension": "sqlite", "status": "missing"})
            differences.append({
                "check_id": cid,
                "dimension": "sqlite",
                "classification": "unavailable",
                "source_fingerprint": canonical_fingerprint(s_val),
                "candidate_fingerprint": canonical_fingerprint(c_val),
                "reason": f"integrity entry missing for {k}",
            })
        elif s_val != "ok" or c_val != "ok":
            checks.append({"check_id": cid, "dimension": "sqlite", "status": "different"})
            differences.append({
                "check_id": cid,
                "dimension": "sqlite",
                "classification": "blocking",
                "source_fingerprint": canonical_fingerprint(s_val),
                "candidate_fingerprint": canonical_fingerprint(c_val),
                "reason": f"sqlite integrity failure on {k}",
            })
        else:
            record_check(cid, "sqlite", s_val, c_val)

    # Counts
    s_cnt_keys = set(src["sqlite"]["counts"].keys())
    c_cnt_keys = set(cand["sqlite"]["counts"].keys())
    for k in sorted(s_cnt_keys | c_cnt_keys):
        cid = f"sqlite.counts.{k}"
        s_val = src["sqlite"]["counts"].get(k)
        c_val = cand["sqlite"]["counts"].get(k)
        if s_val is None or c_val is None:
            checks.append({"check_id": cid, "dimension": "sqlite", "status": "missing"})
            differences.append({
                "check_id": cid,
                "dimension": "sqlite",
                "classification": "unavailable",
                "source_fingerprint": canonical_fingerprint(s_val),
                "candidate_fingerprint": canonical_fingerprint(c_val),
                "reason": f"count entry missing for {k}",
            })
        else:
            record_check(cid, "sqlite", s_val, c_val)

    # 4. digests
    for sub in ("static", "runtime"):
        s_sub = src["digests"][sub]
        c_sub = cand["digests"][sub]
        all_d = sorted(set(s_sub.keys()) | set(c_sub.keys()))
        for k in all_d:
            cid = f"digests.{sub}.{k}"
            s_val = s_sub.get(k)
            c_val = c_sub.get(k)
            if s_val is None or c_val is None:
                checks.append({"check_id": cid, "dimension": "digests", "status": "missing"})
                differences.append({
                    "check_id": cid,
                    "dimension": "digests",
                    "classification": "unavailable",
                    "source_fingerprint": canonical_fingerprint(s_val),
                    "candidate_fingerprint": canonical_fingerprint(c_val),
                    "reason": f"digest entry missing for {k}",
                })
            else:
                record_check(cid, "digests", s_val, c_val)

    # 5. release
    for k in ("schema_version", "source_git_sha", "manifest_sha256"):
        record_check(f"release.{k}", "release", src["release"][k], cand["release"][k])

    # 6. allocation
    for sym in ("SPY", "GLD", "TLT"):
        cid = f"allocation.{sym}"
        s_val = src["allocation"][sym]
        c_val = cand["allocation"][sym]
        expected_val = CHAMPION_ALLOCATION[sym]
        if not math.isclose(s_val, expected_val, abs_tol=1e-9) or not math.isclose(c_val, expected_val, abs_tol=1e-9):
            checks.append({"check_id": cid, "dimension": "allocation", "status": "different"})
            differences.append({
                "check_id": cid,
                "dimension": "allocation",
                "classification": "blocking",
                "source_fingerprint": canonical_fingerprint(s_val),
                "candidate_fingerprint": canonical_fingerprint(c_val),
                "reason": f"champion allocation {sym} mismatch",
            })
        else:
            record_check(cid, "allocation", s_val, c_val)

    # 7. safety
    # kill_switch
    for k in ("enabled", "level", "incident_id"):
        record_check(f"safety.kill_switch.{k}", "safety", src["safety"]["kill_switch"][k], cand["safety"]["kill_switch"][k])
    # open_incidents: sort by (id, channel, severity, state)
    def inc_sort_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
        return (item["id"], item["channel"], item["severity"], item["state"])
    s_inc = sorted(src["safety"]["open_incidents"], key=inc_sort_key)
    c_inc = sorted(cand["safety"]["open_incidents"], key=inc_sort_key)
    record_check("safety.open_incidents", "safety", s_inc, c_inc)

    # 8. tasker
    record_check("tasker.registry_sha256", "tasker", src["tasker"]["registry_sha256"], cand["tasker"]["registry_sha256"])
    record_check("tasker.status_schema", "tasker", src["tasker"]["status_schema"], cand["tasker"]["status_schema"])

    # Scheduler asymmetries are valid only when scheduler safety fields are safe.
    safe_scheduler_pair = safe_expected_scheduler_pair(src, cand)
    # Source expected: scheduler_mode enabled, instances 1, env false, arg false
    # Candidate expected: scheduler_mode disabled, instances 0, env true, arg true, starts 0
    record_asymmetry(
        "tasker.scheduler_mode",
        "tasker",
        src["tasker"]["scheduler_mode"],
        cand["tasker"]["scheduler_mode"],
        "enabled",
        "disabled",
        "candidate scheduler is intentionally disabled during shadow comparison",
        safe_scheduler_pair,
    )
    record_asymmetry(
        "tasker.scheduler_instances",
        "tasker",
        src["tasker"]["scheduler_instances"],
        cand["tasker"]["scheduler_instances"],
        1,
        0,
        "candidate scheduler instance count is zero during shadow comparison",
        safe_scheduler_pair,
    )
    record_asymmetry(
        "tasker.scheduler_env_disabled",
        "tasker",
        src["tasker"]["scheduler_env_disabled"],
        cand["tasker"]["scheduler_env_disabled"],
        False,
        True,
        "candidate scheduler env disable is true during shadow comparison",
        safe_scheduler_pair,
    )
    record_asymmetry(
        "tasker.scheduler_arg_disabled",
        "tasker",
        src["tasker"]["scheduler_arg_disabled"],
        cand["tasker"]["scheduler_arg_disabled"],
        False,
        True,
        "candidate scheduler arg disable is true during shadow comparison",
        safe_scheduler_pair,
    )

    # Scheduled starts observed: candidate MUST be 0
    cid_starts = "tasker.scheduled_starts_observed"
    s_starts = src["tasker"]["scheduled_starts_observed"]
    c_starts = cand["tasker"]["scheduled_starts_observed"]
    if c_starts != 0:
        s_fp = canonical_fingerprint(s_starts)
        c_fp = canonical_fingerprint(c_starts)
        checks.append({"check_id": cid_starts, "dimension": "tasker", "status": "different"})
        differences.append({
            "check_id": cid_starts,
            "dimension": "tasker",
            "classification": "blocking",
            "source_fingerprint": s_fp,
            "candidate_fingerprint": c_fp,
            "reason": "candidate observed non-zero scheduled starts",
        })
    elif s_starts > 0:
        s_fp = canonical_fingerprint(s_starts)
        c_fp = canonical_fingerprint(c_starts)
        checks.append({"check_id": cid_starts, "dimension": "tasker", "status": "different"})
        differences.append({
            "check_id": cid_starts,
            "dimension": "tasker",
            "classification": "expected",
            "source_fingerprint": s_fp,
            "candidate_fingerprint": c_fp,
            "reason": "source scheduler is authoritative; candidate scheduler has zero observed starts",
        })
    else:
        record_check(cid_starts, "tasker", s_starts, c_starts)

    # 9. schemas
    s_sch_keys = set(src["schemas"].keys())
    c_sch_keys = set(cand["schemas"].keys())
    for k in sorted(s_sch_keys | c_sch_keys):
        cid = f"schemas.{k}"
        s_val = src["schemas"].get(k)
        c_val = cand["schemas"].get(k)
        if s_val is None or c_val is None:
            checks.append({"check_id": cid, "dimension": "schemas", "status": "missing"})
            differences.append({
                "check_id": cid,
                "dimension": "schemas",
                "classification": "unavailable",
                "source_fingerprint": canonical_fingerprint(s_val),
                "candidate_fingerprint": canonical_fingerprint(c_val),
                "reason": f"schema entry missing for {k}",
            })
        else:
            record_check(cid, "schemas", s_val, c_val)

    # 10. freshness
    s_f_keys = set(src["freshness"].keys())
    c_f_keys = set(cand["freshness"].keys())
    for k in sorted(s_f_keys | c_f_keys):
        s_item = src["freshness"].get(k)
        c_item = cand["freshness"].get(k)
        if s_item is None or c_item is None:
            cid = f"freshness.{k}"
            checks.append({"check_id": cid, "dimension": "freshness", "status": "missing"})
            differences.append({
                "check_id": cid,
                "dimension": "freshness",
                "classification": "unavailable",
                "source_fingerprint": canonical_fingerprint(s_item),
                "candidate_fingerprint": canonical_fingerprint(c_item),
                "reason": f"freshness entry missing for {k}",
            })
            continue

        # max_age_seconds must match
        cid_max = f"freshness.{k}.max_age_seconds"
        if s_item["max_age_seconds"] != c_item["max_age_seconds"]:
            checks.append({"check_id": cid_max, "dimension": "freshness", "status": "different"})
            differences.append({
                "check_id": cid_max,
                "dimension": "freshness",
                "classification": "blocking",
                "source_fingerprint": canonical_fingerprint(s_item["max_age_seconds"]),
                "candidate_fingerprint": canonical_fingerprint(c_item["max_age_seconds"]),
                "reason": "per-artifact max_age_seconds differs",
            })
        else:
            record_check(cid_max, "freshness", s_item["max_age_seconds"], c_item["max_age_seconds"])

        # Compare generated_at & age_seconds
        cid_gen = f"freshness.{k}.generated_at"
        s_dt = validate_iso8601_tz(s_item["generated_at"])
        c_dt = validate_iso8601_tz(c_item["generated_at"])
        dt_delta = abs((s_dt - c_dt).total_seconds())

        if s_item["generated_at"] == c_item["generated_at"]:
            record_check(cid_gen, "freshness", s_item["generated_at"], c_item["generated_at"])
        elif dt_delta <= max_freshness_delta:
            if cid_gen in explanations:
                record_check(cid_gen, "freshness", s_item["generated_at"], c_item["generated_at"])
            else:
                record_check(
                    cid_gen,
                    "freshness",
                    s_item["generated_at"],
                    c_item["generated_at"],
                    expected_diff_reason="generated_at collection delta within configured maximum",
                )
        else:
            record_check(cid_gen, "freshness", s_item["generated_at"], c_item["generated_at"])

        cid_age = f"freshness.{k}.age_seconds"
        age_delta = abs(s_item["age_seconds"] - c_item["age_seconds"])
        if s_item["age_seconds"] == c_item["age_seconds"]:
            record_check(cid_age, "freshness", s_item["age_seconds"], c_item["age_seconds"])
        elif age_delta <= max_freshness_delta:
            if cid_age in explanations:
                record_check(cid_age, "freshness", s_item["age_seconds"], c_item["age_seconds"])
            else:
                record_check(
                    cid_age,
                    "freshness",
                    s_item["age_seconds"],
                    c_item["age_seconds"],
                    expected_diff_reason="age_seconds collection delta within configured maximum",
                )
        else:
            record_check(cid_age, "freshness", s_item["age_seconds"], c_item["age_seconds"])

    # 11. endpoints
    required_eps = {"/", "/_release.json", "/data/index.json", "/data/signals.json", "/api/tasker/status"}
    for k in sorted(required_eps):
        s_ep = src["endpoints"].get(k)
        c_ep = cand["endpoints"].get(k)
        if s_ep is None or c_ep is None:
            cid = f"endpoints.{k}"
            checks.append({"check_id": cid, "dimension": "endpoints", "status": "missing"})
            differences.append({
                "check_id": cid,
                "dimension": "endpoints",
                "classification": "unavailable",
                "source_fingerprint": canonical_fingerprint(s_ep),
                "candidate_fingerprint": canonical_fingerprint(c_ep),
                "reason": f"required endpoint missing for {k}",
            })

    for k in sorted(set(src["endpoints"].keys()) | set(cand["endpoints"].keys())):
        s_ep = src["endpoints"].get(k)
        c_ep = cand["endpoints"].get(k)
        if s_ep is None or c_ep is None:
            cid = f"endpoints.{k}"
            if k in required_eps:
                continue  # already recorded above
            checks.append({"check_id": cid, "dimension": "endpoints", "status": "missing"})
            differences.append({
                "check_id": cid,
                "dimension": "endpoints",
                "classification": "unavailable",
                "source_fingerprint": canonical_fingerprint(s_ep),
                "candidate_fingerprint": canonical_fingerprint(c_ep),
                "reason": f"endpoint missing for {k}",
            })
            continue

        # Status must be exactly 200 on both sides (unexplainable)
        cid_status = f"endpoints.{k}.status"
        if s_ep["status"] == 200 and c_ep["status"] == 200:
            checks.append({"check_id": cid_status, "dimension": "endpoints", "status": "match"})
        else:
            s_fp = canonical_fingerprint(s_ep["status"])
            c_fp = canonical_fingerprint(c_ep["status"])
            checks.append({"check_id": cid_status, "dimension": "endpoints", "status": "different"})
            differences.append({
                "check_id": cid_status,
                "dimension": "endpoints",
                "classification": "blocking",
                "source_fingerprint": s_fp,
                "candidate_fingerprint": c_fp,
                "reason": f"endpoint {k} status is not 200 on both sides",
            })

        record_check(f"endpoints.{k}.content_type", "endpoints", s_ep["content_type"], c_ep["content_type"])
        record_check(f"endpoints.{k}.schema_version", "endpoints", s_ep["schema_version"], c_ep["schema_version"])

        cid_body = f"endpoints.{k}.body_sha256"
        if k == "/api/tasker/status":
            record_check(cid_body, "endpoints", s_ep["body_sha256"], c_ep["body_sha256"])
        elif s_ep["body_sha256"] is None or c_ep["body_sha256"] is None:
            checks.append({"check_id": cid_body, "dimension": "endpoints", "status": "different"})
            differences.append({
                "check_id": cid_body,
                "dimension": "endpoints",
                "classification": "blocking",
                "source_fingerprint": canonical_fingerprint(s_ep["body_sha256"]),
                "candidate_fingerprint": canonical_fingerprint(c_ep["body_sha256"]),
                "reason": f"endpoint {k} body digest missing",
            })
        else:
            record_check(cid_body, "endpoints", s_ep["body_sha256"], c_ep["body_sha256"])

    # 12. authority
    # Loopback and health are safety invariants: true on both sides, even if equal.
    for cid, key in (
        ("authority.public_origin_loopback_only", "public_origin_loopback_only"),
        ("authority.healthy", "healthy"),
    ):
        s_val = src["authority"][key]
        c_val = cand["authority"][key]
        if s_val is True and c_val is True:
            checks.append({"check_id": cid, "dimension": "authority", "status": "match"})
        else:
            checks.append({"check_id": cid, "dimension": "authority", "status": "different"})
            differences.append({
                "check_id": cid,
                "dimension": "authority",
                "classification": "blocking",
                "source_fingerprint": canonical_fingerprint(s_val),
                "candidate_fingerprint": canonical_fingerprint(c_val),
                "reason": f"{key} must be true on both sides",
            })

    # Authority and Access asymmetry
    safe_authority_pair = safe_expected_authority_pair(src, cand)
    record_asymmetry(
        "authority.authoritative",
        "authority",
        src["authority"]["authoritative"],
        cand["authority"]["authoritative"],
        True,
        False,
        "source is authoritative; candidate is shadow non-authoritative",
        safe_authority_pair,
    )
    record_asymmetry(
        "authority.access_protected",
        "authority",
        src["authority"]["access_protected"],
        cand["authority"]["access_protected"],
        False,
        True,
        "candidate is Cloudflare Access protected during shadow dry run",
        safe_authority_pair,
    )

    # Check for stale or unmatched explanations
    for exp_cid, exp_val in sorted(explanations.items()):
        if exp_cid not in used_explanations:
            checks.append({"check_id": f"explanation.{exp_cid}", "dimension": "explanations", "status": "stale"})
            differences.append({
                "check_id": f"explanation.{exp_cid}",
                "dimension": "explanations",
                "classification": "unavailable",
                "source_fingerprint": exp_val["source_fingerprint"],
                "candidate_fingerprint": exp_val["candidate_fingerprint"],
                "reason": f"unused or fingerprint-mismatched explanation for {exp_cid}",
            })

    # Sort checks by check_id
    checks.sort(key=lambda x: x["check_id"])

    # Sort differences by (classification_order, dimension, check_id)
    class_order = {"expected": 0, "explained": 1, "blocking": 2, "unavailable": 3}
    differences.sort(key=lambda x: (class_order.get(x["classification"], 99), x["dimension"], x["check_id"]))

    counts = {
        "expected": sum(1 for d in differences if d["classification"] == "expected"),
        "explained": sum(1 for d in differences if d["classification"] == "explained"),
        "blocking": sum(1 for d in differences if d["classification"] == "blocking"),
        "unavailable": sum(1 for d in differences if d["classification"] == "unavailable"),
    }

    if counts["blocking"] == 0 and counts["unavailable"] == 0:
        verdict = "pass"
        terminal_stmt = "Dry run passed; cutover approval required."
        exit_code = 0
    else:
        verdict = "blocked"
        exit_code = 2

    # Assemble JSON report
    report_json: dict[str, Any] = {
        "schema_version": SCHEMA_COMPARISON,
        "attribution": {
            "source": {
                "host": src["host"],
                "role": src["role"],
                "collected_at": src["collected_at"],
                "commit": src["git"]["commit"],
            },
            "candidate": {
                "host": cand["host"],
                "role": cand["role"],
                "collected_at": cand["collected_at"],
                "commit": cand["git"]["commit"],
            },
        },
        "summary": {
            "verdict": verdict,
            "counts": counts,
        },
        "differences": differences,
        "checks": checks,
    }
    if verdict == "pass":
        report_json["terminal_statement"] = terminal_stmt
    else:
        failed_ids = [d["check_id"] for d in differences if d["classification"] in ("blocking", "unavailable")]
        report_json["terminal_statement"] = (
            f"Dry run blocked ({', '.join(failed_ids)}). "
            "Retained safe state: sg01 remains authoritative; cursor-box scheduler remains disabled."
        )

    # Build Markdown report
    md_lines: list[str] = []
    md_lines.append("# Portfolio Lab Migration Comparison Report\n")
    md_lines.append("## Attribution\n")
    md_lines.append(f"- **Source Host:** `{markdown_escape(src['host'])}` (`{markdown_escape(src['role'])}`)")
    md_lines.append(f"- **Source Commit:** `{markdown_escape(src['git']['commit'])}`")
    md_lines.append(f"- **Source Collected At:** `{markdown_escape(src['collected_at'])}`")
    md_lines.append(f"- **Candidate Host:** `{markdown_escape(cand['host'])}` (`{markdown_escape(cand['role'])}`)")
    md_lines.append(f"- **Candidate Commit:** `{markdown_escape(cand['git']['commit'])}`")
    md_lines.append(f"- **Candidate Collected At:** `{markdown_escape(cand['collected_at'])}`\n")

    md_lines.append("## Summary\n")
    md_lines.append(f"- **Verdict:** `{verdict.upper()}`")
    md_lines.append(f"- **Expected Differences:** {counts['expected']}")
    md_lines.append(f"- **Explained Differences:** {counts['explained']}")
    md_lines.append(f"- **Blocking Differences:** {counts['blocking']}")
    md_lines.append(f"- **Unavailable Differences:** {counts['unavailable']}\n")

    md_lines.append("## Dimensions\n")
    for dim in DIMENSIONS:
        dim_checks = [c for c in checks if c["dimension"] == dim]
        match_count = sum(1 for c in dim_checks if c["status"] == "match")
        total_dim = len(dim_checks)
        md_lines.append(f"- **{dim}:** {match_count}/{total_dim} checks matched")
    md_lines.append("")

    md_lines.append("## Differences\n")
    if not differences:
        md_lines.append("No differences observed.\n")
    else:
        md_lines.append("| Classification | Dimension | Check ID | Reason |")
        md_lines.append("| --- | --- | --- | --- |")
        for d in differences:
            c_esc = markdown_escape(d["classification"])
            dim_esc = markdown_escape(d["dimension"])
            cid_esc = markdown_escape(d["check_id"])
            rsn_esc = markdown_escape(d["reason"])
            md_lines.append(f"| {c_esc} | {dim_esc} | `{cid_esc}` | {rsn_esc} |")
        md_lines.append("")

    md_lines.append("## Terminal Status\n")
    if verdict == "pass":
        md_lines.append("Dry run passed; cutover approval required.")
    else:
        failed_ids = [d["check_id"] for d in differences if d["classification"] in ("blocking", "unavailable")]
        md_lines.append("Dry run blocked")
        for fid in failed_ids:
            md_lines.append(f"- `{markdown_escape(fid)}`")
        md_lines.append("")
        md_lines.append("Retained safe state: sg01 remains authoritative; cursor-box scheduler remains disabled.")

    md_content = "\n".join(md_lines) + "\n"
    return report_json, md_content, exit_code


class RedactingArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that exits 1 on usage/type errors with static diagnostic, never echoing tokens."""

    def error(self, message: str) -> None:
        die_diagnostic("invalid command-line arguments")


def main() -> int:
    parser = RedactingArgumentParser(
        description="Deterministic, redacted comparison CLI for Portfolio Lab migration evidence."
    )
    parser.add_argument("--source", required=True, help="Path to source evidence directory, JSON file, or HTTPS URL")
    parser.add_argument("--candidate", required=True, help="Path to candidate evidence directory, JSON file, or HTTPS URL")
    parser.add_argument("--output-json", required=True, help="Destination path for comparison report JSON")
    parser.add_argument("--output-markdown", required=True, help="Destination path for comparison report Markdown")
    parser.add_argument("--explanations", default=None, help="Optional path to local explanations JSON file")
    parser.add_argument(
        "--max-freshness-delta-seconds",
        type=float,
        default=300.0,
        help="Maximum allowable delta in seconds for freshness timestamps (default 300.0)",
    )

    args = parser.parse_args()
    if not math.isfinite(args.max_freshness_delta_seconds) or not 0.0 <= args.max_freshness_delta_seconds <= MAX_FRESHNESS_DELTA_BOUND:
        die_diagnostic("invalid freshness delta")

    # Load evidence
    src_data = load_evidence(args.source, role="source", host="sg01")
    cand_data = load_evidence(args.candidate, role="candidate", host="cursor-box")

    # Load explanations
    explanations = validate_explanations(args.explanations)

    report_dict, report_md, exit_code = run_comparison(
        src=src_data,
        cand=cand_data,
        explanations=explanations,
        max_freshness_delta=args.max_freshness_delta_seconds,
    )

    # Serialize JSON canonically
    json_bytes = json.dumps(report_dict, indent=2, sort_keys=True, allow_nan=False) + "\n"

    # Write files atomically (mode 0600, sibling temp files, os.replace)
    write_outputs_atomically(Path(args.output_json), json_bytes, Path(args.output_markdown), report_md)

    # Print compact JSON to stdout (no local paths)
    stdout_summary = {
        "verdict": report_dict["summary"]["verdict"],
        "counts": report_dict["summary"]["counts"],
        "terminal_statement": report_dict["terminal_statement"],
    }
    sys.stdout.write(json.dumps(stdout_summary, sort_keys=True) + "\n")
    return exit_code


def run_cli() -> None:
    """Entry point with bounded failures: never leak a traceback or raw input."""
    code: int | None = None
    try:
        code = main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code == 2:
            # argparse usage errors use exit 2; the contract reserves exit 2
            # for successfully written blocked reports.
            code = 1
    except Exception:
        die_diagnostic("internal error")
        return
    sys.exit(code)


if __name__ == "__main__":
    run_cli()
