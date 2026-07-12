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
    """Critical health/SLO must project into alerts.json (or alerts must exist)."""
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
        slo = health.get("data_pipeline_slo") if isinstance(health.get("data_pipeline_slo"), dict) else {}
        slo_status = slo.get("status")
        errors.append(
            "public/data/health.json is critical "
            f"(system_status={system_status!r}, data_pipeline_slo.status={slo_status!r}) "
            f"but public/data/alerts.json has no type={HEALTH_SLO_ALERT_TYPE!r} alert"
        )


def check_public_data_consistency(app_dir: str | Path) -> ConsistencyResult:
    """Return deploy-blocking public data consistency errors for an app checkout."""
    root = Path(app_dir)
    errors: list[str] = []
    warnings: list[str] = []
    public_data = root / "public" / "data"

    source_manifest_path = public_data / "source_manifest.json"
    source_manifest = _load_json(source_manifest_path, errors)
    public_index = _load_json(public_data / "index.json", errors)
    health = _load_json(public_data / "health.json", errors)
    _check_timestamp_order(source_manifest, public_index, errors)
    _check_source_manifest_identity(source_manifest_path, source_manifest, public_index, errors)
    _check_present_index_entries_resolve(public_data, public_index, errors)
    _check_source_manifest_quality_artifacts_are_indexed(source_manifest, public_index, errors)
    _check_public_json_artifacts_are_indexed(public_data, public_index, errors)
    _check_dist_matches_public(root, errors)
    _check_compact_prices_match_market_db(root, errors)
    _check_critical_health_has_slo_alert(public_data, health, errors)

    return ConsistencyResult(ok=not errors, errors=errors, warnings=warnings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, default=Path.cwd(), help="Portfolio Lab checkout directory")
    args = parser.parse_args(argv)

    result = check_public_data_consistency(args.app_dir)
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
