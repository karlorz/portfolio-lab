#!/usr/bin/env python3
"""Verify Portfolio Lab immutable static release bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "portfolio-lab-static-release/v1"
MANIFEST_NAME = "_release.json"
DATA_PREFIX = "data/"


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    errors: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_deployable_assets(release_dir: Path) -> list[Path]:
    assets: list[Path] = []
    for path in release_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(release_dir).as_posix()
        if rel == MANIFEST_NAME or rel.startswith(DATA_PREFIX):
            continue
        assets.append(path)
    return sorted(assets, key=lambda item: item.relative_to(release_dir).as_posix())


def load_manifest(release_dir: Path) -> dict[str, Any]:
    manifest_path = release_dir / MANIFEST_NAME
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _manifest_asset_map(manifest: dict[str, Any], errors: list[str]) -> dict[str, dict[str, Any]]:
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        errors.append("manifest assets must be a list")
        return {}

    mapped: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(assets):
        if not isinstance(row, dict):
            errors.append(f"manifest assets[{index}] must be an object")
            continue
        path = row.get("path")
        sha = row.get("sha256")
        size = row.get("bytes")
        if not isinstance(path, str) or not path:
            errors.append(f"manifest assets[{index}].path must be a non-empty string")
            continue
        if path == MANIFEST_NAME:
            errors.append("manifest must not list _release.json as a deployable asset")
        if path.startswith(DATA_PREFIX):
            errors.append(f"manifest must not include mutable data asset: {path}")
        if path.startswith("/") or ".." in Path(path).parts:
            errors.append(f"manifest asset path escapes release root: {path}")
        if not isinstance(sha, str) or len(sha) != 64:
            errors.append(f"manifest asset {path} has invalid sha256")
        if not isinstance(size, int) or size < 0:
            errors.append(f"manifest asset {path} has invalid byte size")
        if path in mapped:
            errors.append(f"manifest lists duplicate asset: {path}")
        mapped[path] = row
    return mapped


def verify_release(
    release_dir: Path,
    *,
    expected_source_sha: str | None = None,
    repo_dir: Path | None = None,
) -> VerificationResult:
    release_dir = release_dir.resolve()
    errors: list[str] = []
    manifest_path = release_dir / MANIFEST_NAME

    if not release_dir.is_dir():
        return VerificationResult(False, [f"release dir does not exist: {release_dir}"])
    if not manifest_path.is_file():
        return VerificationResult(False, [f"missing manifest: {manifest_path}"])

    try:
        manifest = load_manifest(release_dir)
    except json.JSONDecodeError as exc:
        return VerificationResult(False, [f"manifest is not valid JSON: {exc}"])

    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"manifest schema_version must be {SCHEMA_VERSION!r}")

    source_sha = manifest.get("source_git_sha")
    if not isinstance(source_sha, str) or len(source_sha) != 40:
        errors.append("manifest source_git_sha must be a full 40-character SHA")
    if expected_source_sha and source_sha != expected_source_sha:
        errors.append(
            f"manifest source_git_sha {source_sha!r} does not match expected {expected_source_sha!r}"
        )

    for key in ("build_time_utc", "build_command", "bun_version"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            errors.append(f"manifest {key} must be a non-empty string")

    lockfile = manifest.get("lockfile")
    if not isinstance(lockfile, dict):
        errors.append("manifest lockfile must be an object")
    else:
        lock_path = lockfile.get("path")
        lock_sha = lockfile.get("sha256")
        if not isinstance(lock_path, str) or not lock_path:
            errors.append("manifest lockfile.path must be a non-empty string")
        if not isinstance(lock_sha, str) or len(lock_sha) != 64:
            errors.append("manifest lockfile.sha256 must be a sha256 hex digest")
        if repo_dir and isinstance(lock_path, str) and isinstance(lock_sha, str):
            actual_lock = (repo_dir / lock_path).resolve()
            try:
                actual_lock.relative_to(repo_dir.resolve())
            except ValueError:
                errors.append(f"manifest lockfile path escapes repo root: {lock_path}")
            if not actual_lock.is_file():
                errors.append(f"manifest lockfile is missing from repo: {lock_path}")
            elif sha256_file(actual_lock) != lock_sha:
                errors.append(f"manifest lockfile digest mismatch: {lock_path}")

    policy = manifest.get("policy")
    if not isinstance(policy, dict) or policy.get("mutable_data_excluded") is not True:
        errors.append("manifest policy.mutable_data_excluded must be true")

    manifest_assets = _manifest_asset_map(manifest, errors)
    actual_assets = {
        path.relative_to(release_dir).as_posix(): path for path in iter_deployable_assets(release_dir)
    }

    missing = sorted(set(manifest_assets) - set(actual_assets))
    for rel in missing:
        errors.append(f"manifest asset is missing from release tree: {rel}")

    extra = sorted(set(actual_assets) - set(manifest_assets))
    for rel in extra:
        errors.append(f"release tree has unmanifested deployable asset: {rel}")

    for rel in sorted(set(manifest_assets) & set(actual_assets)):
        path = actual_assets[rel]
        row = manifest_assets[rel]
        actual_sha = sha256_file(path)
        actual_size = path.stat().st_size
        if row.get("sha256") != actual_sha:
            errors.append(f"asset digest mismatch: {rel}")
        if row.get("bytes") != actual_size:
            errors.append(f"asset byte size mismatch: {rel}")

    return VerificationResult(not errors, errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--expected-source-sha")
    parser.add_argument("--repo-dir", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable result")
    args = parser.parse_args(argv)

    result = verify_release(
        args.release_dir,
        expected_source_sha=args.expected_source_sha,
        repo_dir=args.repo_dir,
    )
    if args.json:
        print(json.dumps({"ok": result.ok, "errors": result.errors}, sort_keys=True))
    elif result.ok:
        print(f"release verification ok: {args.release_dir}")
    else:
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
