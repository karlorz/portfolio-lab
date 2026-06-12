"""Mode-sensitive FRED credential readiness checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Mapping, Sequence


FRED_READINESS_SCHEMA_VERSION = "fred-readiness/v1"

_MODE_ALIASES = {
    "ci": "test",
    "tests": "test",
    "dev": "local",
    "development": "local",
    "prod": "live",
    "production": "live",
    "paper-trading": "paper",
    "paper_trading": "paper",
}
_PERMISSIVE_MODES = {"local", "test"}
_WARNING_MODES = {"lab", "paper", "staging"}
_FAIL_HARD_MODES = {"live"}


def _normalize_mode(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    return _MODE_ALIASES.get(normalized, normalized)


def resolve_fred_operating_mode(
    mode: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the runtime mode used for FRED readiness policy."""
    if (resolved := _normalize_mode(mode)) is not None:
        return resolved

    runtime_env = env if env is not None else os.environ
    for key in ("PORTFOLIO_LAB_MODE", "ALPHALAB_MODE", "APP_MODE"):
        if (resolved := _normalize_mode(runtime_env.get(key))) is not None:
            return resolved

    if _normalize_mode(runtime_env.get("CRON_BACKEND")) == "tasker":
        return "lab"

    return "local"


def _classify_fred_issue(fred_health: Mapping[str, Any], api_key_configured: bool) -> str | None:
    source_mode = str(fred_health.get("source_mode") or "unknown").lower()
    health_status = str(fred_health.get("status") or "unknown").lower()
    reason = str(fred_health.get("reason") or "").lower()

    if not api_key_configured:
        return "missing_fred_api_key"
    if any(token in reason for token in ("invalid", "auth", "unauthorized", "api_key", "credential")):
        return "invalid_fred_api_key"
    if source_mode in {"synthetic", "unavailable"} or health_status in {"unavailable", "empty"}:
        return "fred_data_unavailable"
    return None


def assess_fred_readiness(
    fred_health: Mapping[str, Any] | None,
    *,
    mode: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Assess whether FRED credentials are acceptable for the resolved mode.

    The returned payload is intentionally safe for logs and public health
    artifacts: it records whether a key is configured, but never includes the
    key value.
    """
    health = fred_health if isinstance(fred_health, Mapping) else {}
    resolved_mode = resolve_fred_operating_mode(mode, env=env)
    env_mapping = env if env is not None else os.environ
    api_key_configured = bool(
        health.get("api_key_configured")
        if "api_key_configured" in health
        else env_mapping.get("FRED_API_KEY")
    )
    issue = _classify_fred_issue(health, api_key_configured)

    source_mode = str(health.get("source_mode") or "unknown")
    cache_status = str(health.get("status") or "unknown")

    if issue is None:
        return {
            "schema_version": FRED_READINESS_SCHEMA_VERSION,
            "status": "ok",
            "readiness": "pass",
            "ready": True,
            "blocking": False,
            "enforcement": "required",
            "mode": resolved_mode,
            "api_key_configured": api_key_configured,
            "source_mode": source_mode,
            "fred_cache_status": cache_status,
            "reason": None,
            "message": f"FRED credential readiness ok for {resolved_mode} mode.",
            "remediation": None,
        }

    if resolved_mode in _FAIL_HARD_MODES:
        status = "critical"
        readiness = "fail"
        ready = False
        blocking = True
        enforcement = "required"
    elif resolved_mode in _WARNING_MODES:
        status = "warning"
        readiness = "warn"
        ready = True
        blocking = False
        enforcement = "monitored"
    else:
        status = "warning"
        readiness = "pass"
        ready = True
        blocking = False
        enforcement = "permissive" if resolved_mode in _PERMISSIVE_MODES else "monitored"

    if issue == "missing_fred_api_key":
        message = (
            f"FRED_API_KEY is not configured for {resolved_mode} mode; "
            f"FRED data is using {source_mode} fallback."
        )
        remediation = "Set FRED_API_KEY in the deployment environment before paper/live operation."
    elif issue == "invalid_fred_api_key":
        message = f"FRED_API_KEY is configured but FRED data is unavailable in {resolved_mode} mode."
        remediation = "Verify or rotate FRED_API_KEY; do not publish the key value in logs or artifacts."
    else:
        message = f"FRED data is unavailable in {resolved_mode} mode."
        remediation = "Verify fredapi availability, FRED_API_KEY, and the local FRED cache."

    return {
        "schema_version": FRED_READINESS_SCHEMA_VERSION,
        "status": status,
        "readiness": readiness,
        "ready": ready,
        "blocking": blocking,
        "enforcement": enforcement,
        "mode": resolved_mode,
        "api_key_configured": api_key_configured,
        "source_mode": source_mode,
        "fred_cache_status": cache_status,
        "reason": issue,
        "message": message,
        "remediation": remediation,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check FRED credential readiness without live provider calls.")
    parser.add_argument("--mode", default=None, help="Override PORTFOLIO_LAB_MODE/ALPHALAB_MODE/APP_MODE.")
    args = parser.parse_args(argv)

    from src.data.fred_data import get_fred_md_cache_health

    readiness = assess_fred_readiness(get_fred_md_cache_health(), mode=args.mode)
    sys.stdout.write(json.dumps(readiness, indent=2, sort_keys=True) + "\n")
    return 2 if readiness["status"] == "critical" else 0


if __name__ == "__main__":
    raise SystemExit(main())
