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


REQUIRED_DATA_FILES = ("source_manifest.json", "index.json", "health.json")


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


def check_public_data_consistency(app_dir: str | Path) -> ConsistencyResult:
    """Return deploy-blocking public data consistency errors for an app checkout."""
    root = Path(app_dir)
    errors: list[str] = []
    warnings: list[str] = []
    public_data = root / "public" / "data"

    source_manifest_path = public_data / "source_manifest.json"
    source_manifest = _load_json(source_manifest_path, errors)
    public_index = _load_json(public_data / "index.json", errors)
    _load_json(public_data / "health.json", errors)
    _check_timestamp_order(source_manifest, public_index, errors)
    _check_source_manifest_identity(source_manifest_path, source_manifest, public_index, errors)
    _check_dist_matches_public(root, errors)

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
