#!/usr/bin/env python3
"""Deploy-time consistency checks for public dashboard data artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.monitor.market_data_consistency import reconcile_compact_prices_with_market_db


REQUIRED_DATA_FILES = ("source_manifest.json", "index.json", "health.json")
IGNORED_UNMANAGED_PUBLIC_JSON = {
    ".public_data_index_hash_cache.json",
    *REQUIRED_DATA_FILES,
}

# Operator/dashboard JSON that producers stamp with generator_git_sha (Batch AO/AP).
# Pure market blobs (prices/historical/yields) are excluded — content digests live
# in source_manifest, not code provenance stamps.
PROVENANCE_CONTRACT_FILES = (
    "index.json",
    "source_manifest.json",
    "data_quality.json",
    "health.json",
    "health_ops.json",
    "alerts.json",
    "incidents.json",
    "decision_registry.json",
    "stats.json",
    "analytics.json",
    "graduation.json",
    "signals.json",
    "adaptive_sizing.json",
    "risk_decomposition.json",
    "overlay_dashboard.json",
    "rebalance_health.json",
    "labs_registry.json",
)
# Status values that claim a successful full stamp without null sha
_PROVENANCE_FULL_STATUSES = frozenset({"full", "full_generate"})

# Artifacts that emit provenance_completeness dual-write blocks (Batch AS/AT)
DUAL_WRITE_PROVENANCE_FILES = (
    "incidents.json",
    "health_ops.json",
    "unified_dashboard.json",
    "rebalance_health.json",
    "garch_cvar.json",
)


@dataclass(frozen=True)
class ConsistencyResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


def _parse_generated_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        errors.append(f"{path} is missing")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path} is not valid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{path} must contain a JSON object")
        return None
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_generated_at(public_index: dict[str, Any]) -> Any:
    if public_index.get("generated_at"):
        return public_index.get("generated_at")
    entries = public_index.get("entries")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("filename") == "source_manifest.json" or entry.get("path") == "source_manifest.json":
            return entry.get("generated_at")
    return None


def _check_timestamp_order(
    source_manifest: dict[str, Any] | None,
    public_index: dict[str, Any] | None,
    errors: list[str],
) -> None:
    if source_manifest is None or public_index is None:
        return
    source_generated_at = source_manifest.get("generated_at")
    index_generated_at = _index_generated_at(public_index)
    source_dt = _parse_generated_at(source_generated_at)
    index_dt = _parse_generated_at(index_generated_at)
    if source_dt is None:
        errors.append("public/data/source_manifest.json generated_at is missing or unparseable")
        return
    if index_dt is None:
        errors.append("public/data/index.json generated_at is missing or unparseable")
        return
    if source_dt > index_dt:
        errors.append(
            "public/data/index.json is older than source_manifest.json "
            f"(source={source_generated_at}, index={index_generated_at})"
        )


def _check_source_manifest_identity(
    source_manifest_path: Path,
    source_manifest: dict[str, Any] | None,
    public_index: dict[str, Any] | None,
    errors: list[str],
) -> None:
    if source_manifest is None or public_index is None:
        return
    identity = public_index.get("source_manifest")
    if not isinstance(identity, dict):
        errors.append("public/data/index.json source_manifest metadata is missing")
        return
    if identity.get("path") != "source_manifest.json":
        errors.append("public/data/index.json source_manifest.path must be source_manifest.json")

    source_schema = source_manifest.get("schema_version")
    if source_schema and identity.get("schema_version") != source_schema:
        errors.append("public/data/index.json source_manifest.schema_version does not match source_manifest.json")

    source_generated_at = source_manifest.get("generated_at")
    if source_generated_at and identity.get("generated_at") != source_generated_at:
        errors.append("public/data/index.json source_manifest.generated_at does not match source_manifest.json")

    expected_hash = _sha256(source_manifest_path)
    if identity.get("sha256") != expected_hash:
        errors.append("public/data/index.json source_manifest.sha256 does not match public/data/source_manifest.json")


def _check_present_index_entries_resolve(
    public_data: Path,
    public_index: dict[str, Any] | None,
    errors: list[str],
) -> None:
    if public_index is None:
        return
    entries = public_index.get("entries")
    if not isinstance(entries, list):
        errors.append("public/data/index.json entries must be an array")
        return

    public_root = public_data.resolve()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("status") != "present":
            continue
        filename = entry.get("filename")
        path_value = entry.get("path")
        label = filename if isinstance(filename, str) and filename else f"entries[{index}]"
        if not isinstance(path_value, str) or not path_value:
            errors.append(f"public/data/index.json entry {label} is marked present but path is missing")
            continue
        path = Path(path_value)
        if path.is_absolute() or ".." in path.parts:
            errors.append(
                f"public/data/index.json entry {label} is marked present but path {path_value} escapes public/data"
            )
            continue
        resolved = (public_data / path).resolve()
        try:
            resolved.relative_to(public_root)
        except ValueError:
            errors.append(
                f"public/data/index.json entry {label} is marked present but path {path_value} escapes public/data"
            )
            continue
        if not resolved.exists():
            errors.append(
                f"public/data/index.json entry {label} is marked present but public/data/{path_value} is missing"
            )


def _indexed_public_paths(public_index: dict[str, Any] | None) -> set[str]:
    if public_index is None:
        return set()
    indexed: set[str] = set()
    entries = public_index.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            filename = entry.get("filename")
            path = entry.get("path")
            if isinstance(filename, str) and filename:
                indexed.add(filename)
            if isinstance(path, str) and path:
                indexed.add(path)
            pagination = entry.get("pagination")
            pages = pagination.get("pages") if isinstance(pagination, dict) else None
            if isinstance(pages, list):
                for page in pages:
                    if not isinstance(page, dict):
                        continue
                    page_path = page.get("path")
                    if isinstance(page_path, str) and page_path:
                        indexed.add(page_path)
    files = public_index.get("files")
    if isinstance(files, list):
        indexed.update(file for file in files if isinstance(file, str) and file)
    return indexed


def _check_source_manifest_quality_artifacts_are_indexed(
    source_manifest: dict[str, Any] | None,
    public_index: dict[str, Any] | None,
    errors: list[str],
) -> None:
    if source_manifest is None or public_index is None:
        return
    artifacts = source_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return
    indexed = _indexed_public_paths(public_index)
    checked_quality_artifacts: set[str] = set()
    for row in artifacts:
        if not isinstance(row, dict):
            continue
        quality = row.get("data_quality")
        if not isinstance(quality, dict):
            continue
        quality_artifact = quality.get("artifact")
        if not isinstance(quality_artifact, str) or not quality_artifact:
            continue
        if quality_artifact in checked_quality_artifacts:
            continue
        checked_quality_artifacts.add(quality_artifact)
        if quality_artifact not in indexed:
            errors.append(
                "public/data/source_manifest.json references "
                f"{quality_artifact} but public/data/index.json has no entry for it"
            )


def _check_public_json_artifacts_are_indexed(
    public_data: Path,
    public_index: dict[str, Any] | None,
    errors: list[str],
) -> None:
    if public_index is None or not public_data.exists():
        return
    indexed = _indexed_public_paths(public_index)
    for path in sorted(public_data.rglob("*.json")):
        try:
            relative_path = path.relative_to(public_data).as_posix()
        except ValueError:
            continue
        if relative_path in IGNORED_UNMANAGED_PUBLIC_JSON or path.name in IGNORED_UNMANAGED_PUBLIC_JSON:
            continue
        if relative_path not in indexed and path.name not in indexed:
            errors.append(f"public/data/{relative_path} exists but is absent from public/data/index.json")


def _check_dist_matches_public(app_dir: Path, errors: list[str]) -> None:
    public_data = app_dir / "public" / "data"
    dist_data = app_dir / "dist" / "data"
    for filename in REQUIRED_DATA_FILES:
        public_file = public_data / filename
        dist_file = dist_data / filename
        public_exists = public_file.exists()
        dist_exists = dist_file.exists()
        if not public_exists:
            errors.append(f"public/data/{filename} is missing")
        if not dist_exists:
            errors.append(f"dist/data/{filename} is missing")
        if public_exists and dist_exists and _sha256(public_file) != _sha256(dist_file):
            errors.append(f"dist/data/{filename} does not match public/data/{filename}")


def _check_compact_prices_match_market_db(app_dir: Path, errors: list[str]) -> None:
    prices_path = app_dir / "public" / "data" / "prices.json"
    if not prices_path.exists():
        return
    report = reconcile_compact_prices_with_market_db(
        prices_path=prices_path,
        db_path=app_dir / "data" / "market.db",
    )
    if report["status"] == "ok":
        return
    if report["status"] == "unavailable" and report["failure_type"] == "compact_prices_unavailable":
        return
    remediation = report.get("remediation_command")
    for offender in report.get("top_offenders", []):
        message = offender.get("message")
        if isinstance(message, str) and message:
            if remediation:
                message = f"{message}. Remediation: {remediation}"
            errors.append(message)
    if not report.get("top_offenders"):
        message = str(report.get("message") or "market.db did not reconcile with public/data/prices.json")
        if remediation:
            message = f"{message}. Remediation: {remediation}"
        errors.append(message)


def _check_critical_health_has_slo_alert(
    public_data: Path,
    health: dict[str, Any] | None,
    errors: list[str],
) -> None:
    """Critical health/SLO must project into alerts.json (or alerts must exist).

    Kill-driven critical system_status is covered by type=kill_switch alerts
    (enforced separately). health_slo is required when data_pipeline_slo is
    critical, or when system_status is critical without an active kill block.
    """
    if health is None:
        return
    try:
        from src.dashboard.health_slo_alerts import (
            HEALTH_SLO_ALERT_TYPE,
            critical_health_requires_alert,
        )
    except ImportError:
        return
    if not critical_health_requires_alert(health):
        return

    slo = health.get("data_pipeline_slo") if isinstance(health.get("data_pipeline_slo"), dict) else {}
    slo_status = str(slo.get("status") or "").lower()
    kill = health.get("kill_switch") if isinstance(health.get("kill_switch"), dict) else {}
    kill_enabled = bool(kill.get("enabled"))
    # Kill-only critical: kill_switch alert is the operator projection.
    if slo_status != "critical" and kill_enabled:
        return

    alerts_path = public_data / "alerts.json"
    if not alerts_path.exists():
        errors.append(
            "public/data/health.json is critical but public/data/alerts.json is missing"
        )
        return
    alerts_payload = _load_json(alerts_path, errors)
    if alerts_payload is None:
        return
    raw_alerts = alerts_payload.get("alerts")
    alerts = raw_alerts if isinstance(raw_alerts, list) else []
    has_health_slo = any(
        isinstance(a, dict) and a.get("type") == HEALTH_SLO_ALERT_TYPE for a in alerts
    )
    if not has_health_slo:
        system_status = health.get("system_status")
        errors.append(
            "public/data/health.json is critical "
            f"(system_status={system_status!r}, data_pipeline_slo.status={slo_status!r}) "
            f"but public/data/alerts.json has no type={HEALTH_SLO_ALERT_TYPE!r} alert"
        )


def _alert_rows(public_data: Path, errors: list[str]) -> list[dict[str, Any]]:
    alerts_path = public_data / "alerts.json"
    if not alerts_path.exists():
        return []
    payload = _load_json(alerts_path, errors)
    if payload is None:
        return []
    raw = payload.get("alerts")
    return [a for a in raw if isinstance(a, dict)] if isinstance(raw, list) else []


def _check_kill_and_graduation_alerts(
    app_dir: Path,
    public_data: Path,
    errors: list[str],
) -> None:
    """Require kill_switch / graduation_candidate alerts when authority files demand them."""
    data_dir = app_dir / "data"
    kill_path = data_dir / "kill_switch.json"
    promote_path = data_dir / ".promote_to_live"

    kill_enabled = False
    kill_identity: dict[str, Any] | None = None
    if kill_path.exists():
        kill_payload = _load_json(kill_path, errors)
        if isinstance(kill_payload, dict) and kill_payload.get("enabled"):
            kill_enabled = True
            kill_identity = kill_payload

    # Active candidacy only — match DashboardGenerator._is_active_promote_candidacy:
    # promote_blocked_* tombstones must not require graduation_candidate alerts.
    promote_requires_alert = False
    if promote_path.exists():
        promote_payload = _load_json(promote_path, errors)
        if isinstance(promote_payload, dict):
            action = promote_payload.get("action")
            if action is None:
                # Legacy markers omit action; treat as candidacy.
                promote_requires_alert = True
            elif isinstance(action, str) and action == "promote_to_live":
                promote_requires_alert = True
            elif isinstance(action, str) and action.startswith("promote_blocked"):
                promote_requires_alert = False
            else:
                promote_requires_alert = False
        else:
            # Unreadable/non-dict file still present: fail closed for candidacy gate.
            promote_requires_alert = True

    if not kill_enabled and not promote_requires_alert:
        return

    alerts = _alert_rows(public_data, errors)
    types = {a.get("type") for a in alerts}

    if kill_enabled:
        kill_alerts = [a for a in alerts if a.get("type") == "kill_switch"]
        if not kill_alerts:
            if not (public_data / "alerts.json").exists():
                errors.append(
                    "data/kill_switch.json is enabled but public/data/alerts.json is missing"
                )
            else:
                errors.append(
                    "data/kill_switch.json is enabled but public/data/alerts.json has no type='kill_switch' alert"
                )
        elif kill_identity is not None:
            # Multi-surface identity: alert must carry and match authority fields.
            # Missing identity fields are failures (stale reason-only alerts).
            alert = kill_alerts[0]
            auth_incident = kill_identity.get("incident_id")
            auth_level = kill_identity.get("level")
            auth_reason = kill_identity.get("reason")
            auth_mode = kill_identity.get("mode")

            alert_incident = alert.get("incident_id")
            if auth_incident is not None:
                if alert_incident is None:
                    errors.append(
                        "public/data/alerts.json kill_switch is missing incident_id "
                        f"required by data/kill_switch.json incident_id={auth_incident!r}"
                    )
                elif alert_incident != auth_incident:
                    errors.append(
                        "public/data/alerts.json kill_switch incident_id diverges from data/kill_switch.json "
                        f"(alert={alert_incident!r}, authority={auth_incident!r})"
                    )

            alert_level = alert.get("kill_switch_level")
            if auth_level is not None:
                if alert_level is None:
                    errors.append(
                        "public/data/alerts.json kill_switch is missing kill_switch_level "
                        f"required by data/kill_switch.json level={auth_level!r}"
                    )
                elif str(alert_level).lower() != str(auth_level).lower():
                    errors.append(
                        "public/data/alerts.json kill_switch level diverges from data/kill_switch.json "
                        f"(alert={alert_level!r}, authority={auth_level!r})"
                    )

            alert_reason = alert.get("reason")
            if auth_reason is not None:
                if alert_reason is None:
                    errors.append(
                        "public/data/alerts.json kill_switch is missing reason "
                        f"required by data/kill_switch.json reason={auth_reason!r}"
                    )
                elif alert_reason != auth_reason:
                    errors.append(
                        "public/data/alerts.json kill_switch reason diverges from data/kill_switch.json "
                        f"(alert={alert_reason!r}, authority={auth_reason!r})"
                    )

            if auth_mode is not None:
                title = str(alert.get("title") or "")
                if str(auth_mode).upper() not in title.upper():
                    errors.append(
                        "public/data/alerts.json kill_switch title mode does not match "
                        f"data/kill_switch.json mode={auth_mode!r}"
                    )

        # Public health must project kill_switch when authority kill is enabled.
        health_path = public_data / "health.json"
        health = _load_json(health_path, errors) if health_path.exists() else None
        if health is None:
            # _load_json already recorded missing/invalid when path exists; if absent
            # and not already required by REQUIRED_DATA_FILES path, still require projection.
            if not health_path.exists():
                errors.append(
                    "data/kill_switch.json is enabled but public/data/health.json is missing kill_switch projection"
                )
        elif kill_identity is not None:
            pub_kill = health.get("kill_switch")
            if not isinstance(pub_kill, dict):
                errors.append(
                    "data/kill_switch.json is enabled but public/data/health.json is missing kill_switch block"
                )
            else:
                for field in ("incident_id", "level", "reason", "mode", "enabled"):
                    auth_val = kill_identity.get(field)
                    if field == "enabled":
                        auth_val = True
                    pub_val = pub_kill.get(field)
                    if auth_val is None:
                        continue
                    if pub_val is None:
                        errors.append(
                            f"public/data/health.json kill_switch is missing {field} "
                            f"required by data/kill_switch.json {field}={auth_val!r}"
                        )
                    elif field == "level" and str(pub_val).lower() != str(auth_val).lower():
                        errors.append(
                            "public/data/health.json kill_switch.level diverges from data/kill_switch.json "
                            f"(public={pub_val!r}, authority={auth_val!r})"
                        )
                    elif field == "enabled" and bool(pub_val) is not True:
                        errors.append(
                            "public/data/health.json kill_switch.enabled must be true when data/kill_switch.json is enabled"
                        )
                    elif field not in {"level", "enabled"} and pub_val != auth_val:
                        errors.append(
                            f"public/data/health.json kill_switch.{field} diverges from data/kill_switch.json "
                            f"(public={pub_val!r}, authority={auth_val!r})"
                        )

    if promote_requires_alert:
        if "graduation_candidate" not in types:
            if not (public_data / "alerts.json").exists():
                errors.append(
                    "data/.promote_to_live is present but public/data/alerts.json is missing"
                )
            else:
                errors.append(
                    "data/.promote_to_live is present but public/data/alerts.json has no type='graduation_candidate' alert"
                )



def _check_generator_git_sha_provenance(
    public_data: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Canary: contract JSON has generator_git_sha or explicit unavailable status.

    - Missing stamp on present contract files → warning (stale tree until regen).
    - Dishonest stamp (status claims full* but sha null/absent) → error (fail-closed).
    """
    for filename in PROVENANCE_CONTRACT_FILES:
        path = public_data / filename
        if not path.exists():
            continue
        local_errors: list[str] = []
        payload = _load_json(path, local_errors)
        if payload is None:
            # Malformed JSON already surfaces elsewhere; avoid double-counting
            # hard errors when file is optional and unreadable.
            if local_errors and filename in REQUIRED_DATA_FILES:
                continue
            for msg in local_errors:
                warnings.append(f"provenance canary skipped unreadable {filename}: {msg}")
            continue

        sha = payload.get("generator_git_sha")
        status = payload.get("generator_git_sha_status")
        status_s = str(status).strip().lower() if status is not None else ""

        if status_s in _PROVENANCE_FULL_STATUSES and not sha:
            errors.append(
                f"public/data/{filename} claims generator_git_sha_status={status!r} "
                "but generator_git_sha is missing/null (dishonest provenance)"
            )
            continue

        if sha:
            continue

        if status_s in {"unavailable", "partial_patch", "partial"}:
            # Explicit honesty — acceptable without live sha
            continue

        # Present operator artifact without stamp: warn (do not block deploy on
        # pre-stamp trees until producers re-run).
        warnings.append(
            f"public/data/{filename} missing generator_git_sha "
            "(regenerate producer or stamp unavailable status)"
        )



def _check_dual_write_provenance_completeness(
    public_data: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Warn when dual-write was attempted but dual_write_ok is false (H11)."""
    for filename in DUAL_WRITE_PROVENANCE_FILES:
        path = public_data / filename
        if not path.exists():
            continue
        local: list[str] = []
        payload = _load_json(path, local)
        if payload is None:
            continue
        pc = payload.get("provenance_completeness")
        if not isinstance(pc, dict):
            continue
        if pc.get("dual_write_attempted") and pc.get("dual_write_ok") is False:
            warnings.append(
                f"public/data/{filename} dual_write_attempted but dual_write_ok=false "
                f"(note={pc.get('note')!r}; check private vs public split-brain)"
            )
        if pc.get("dual_write_lag_stale") is True:
            lag = pc.get("dual_write_lag_seconds")
            thr = pc.get("dual_write_lag_threshold_seconds")
            warnings.append(
                f"public/data/{filename} dual_write_lag_stale "
                f"(lag_seconds={lag!r} threshold={thr!r}; public mtime behind private)"
            )


def check_public_data_consistency(
    app_dir: str | Path,
    *,
    public_dir: str | Path | None = None,
    allow_repo_public_data: bool = False,
    env: dict[str, str] | None = None,
    live_public_data_dir: str | Path | None = None,
) -> ConsistencyResult:
    """Return deploy-blocking public data consistency errors for an app checkout.

    When ``public_dir`` is unset and live WWW public data exists, requires
    ``PUBLIC_DATA_DIR`` / ``--public-dir`` (or ``allow_repo_public_data``) so
    agents do not audit a multi-day-stale repo ``public/data`` tree.
    """
    from src.paths import resolve_ops_public_data_dir

    root = Path(app_dir)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        public_data = resolve_ops_public_data_dir(
            root,
            public_dir,
            env=env,
            live_public_data_dir=live_public_data_dir,
            allow_repo_public_data=allow_repo_public_data,
        )
    except ValueError as exc:
        return ConsistencyResult(ok=False, errors=[str(exc)], warnings=[])

    source_manifest_path = public_data / "source_manifest.json"
    source_manifest = _load_json(source_manifest_path, errors)
    public_index = _load_json(public_data / "index.json", errors)
    health = _load_json(public_data / "health.json", errors)
    _check_timestamp_order(source_manifest, public_index, errors)
    _check_source_manifest_identity(source_manifest_path, source_manifest, public_index, errors)
    _check_present_index_entries_resolve(public_data, public_index, errors)
    _check_source_manifest_quality_artifacts_are_indexed(source_manifest, public_index, errors)
    _check_public_json_artifacts_are_indexed(public_data, public_index, errors)
    # dist/ vs public/ only applies when auditing the checkout public tree
    repo_public = (root / "public" / "data").resolve()
    try:
        auditing_repo_tree = public_data.resolve() == repo_public
    except OSError:
        auditing_repo_tree = False
    if auditing_repo_tree:
        _check_dist_matches_public(root, errors)
        _check_compact_prices_match_market_db(root, errors)
    _check_critical_health_has_slo_alert(public_data, health, errors)
    _check_kill_and_graduation_alerts(root, public_data, errors)
    _check_generator_git_sha_provenance(public_data, errors, warnings)

    return ConsistencyResult(ok=not errors, errors=errors, warnings=warnings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, default=Path.cwd(), help="Portfolio Lab checkout directory")
    parser.add_argument(
        "--public-dir",
        type=Path,
        default=None,
        help="Public data tree to audit (default: PUBLIC_DATA_DIR or app-dir/public/data)",
    )
    parser.add_argument(
        "--allow-repo-public-data",
        action="store_true",
        help="Allow auditing app-dir/public/data even when live WWW public data exists",
    )
    args = parser.parse_args(argv)

    result = check_public_data_consistency(
        args.app_dir,
        public_dir=args.public_dir,
        allow_repo_public_data=args.allow_repo_public_data,
    )
    for warning in result.warnings:
        print(f"WARN: {warning}", file=sys.stderr)
    if result.ok:
        print("public data consistency check passed")
        return 0
    for error in result.errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
