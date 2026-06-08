"""Public data index manifest generation for dashboard artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from src.paths import PUBLIC_DATA_DIR
from src.research.experiment_artifact_validator import (
    LABS_REGISTRY_SCHEMA_VERSION,
    LABS_REPLAY_SCHEMA_VERSION,
    LABS_SCORECARD_SCHEMA_VERSION,
    validate_artifact,
    validate_file,
)

PUBLIC_DATA_INDEX_SCHEMA_VERSION = "public-data-index/v1"
LABS_VALIDATION_SCHEMA_VERSION = "labs-validation/v1"

_PUBLIC_DATA_CONTRACT: dict[str, tuple[str, str]] = {
    "dashboard.json": ("dashboard", "dashboard/v1"),
    "signals.json": ("signals", "signals/v1"),
    "stats.json": ("dashboard", "stats/v1"),
    "alerts.json": ("monitoring", "alerts/v1"),
    "health.json": ("monitoring", "health/v1"),
    "analytics.json": ("analytics", "analytics/v1"),
    "graduation.json": ("paper_trading", "graduation/v1"),
    "adaptive_sizing.json": ("strategy", "adaptive-sizing/v1"),
    "vixy_hedge.json": ("strategy", "vixy-hedge/v1"),
    "black_litterman.json": ("strategy", "black-litterman/v1"),
    "turnover_validator.json": ("strategy", "turnover-validator/v1"),
    "regime_gate.json": ("strategy", "regime-gate/v1"),
    "tsmom.json": ("signals", "tsmom/v1"),
    "cross_asset_rv.json": ("signals", "cross-asset-rv/v1"),
    "explainability_latest.json": ("explainability", "explainability/v1"),
    "risk_decomposition.json": ("risk", "risk-decomposition/v1"),
    "overlay_dashboard.json": ("overlay", "overlay-dashboard/v1"),
    "labs_registry.json": ("labs", LABS_REGISTRY_SCHEMA_VERSION),
    "labs_scorecards.json": ("labs", LABS_SCORECARD_SCHEMA_VERSION),
    "labs_replays.json": ("labs", LABS_REPLAY_SCHEMA_VERSION),
    "labs_validation.json": ("labs", LABS_VALIDATION_SCHEMA_VERSION),
}

_OPTIONAL_PUBLIC_DATA_FILES = (
    "labs_registry.json",
    "labs_scorecards.json",
    "labs_replays.json",
    "labs_validation.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_for_filename(filename: str) -> tuple[str, str]:
    if filename in _PUBLIC_DATA_CONTRACT:
        return _PUBLIC_DATA_CONTRACT[filename]
    if "labs" in filename or "scorecard" in filename or "replay" in filename:
        return "labs", "unknown"
    return "unknown", "unknown"


def _generated_at_for_file(path: Path, fallback: str) -> str:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return fallback
    if isinstance(data, dict) and isinstance(data.get("generated_at"), str):
        return data["generated_at"]
    return fallback


def _format_collection_error(index: int, error: str) -> str:
    if error.startswith("$"):
        return f"$[{index}]{error[1:]}"
    return f"$[{index}].{error}"


def _validate_labs_collection(path: Path, expected_schema_version: str) -> tuple[str, str, list[str]]:
    try:
        with open(path) as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        return expected_schema_version, "invalid", [f"$: invalid JSON: {exc.msg}"]
    except OSError as exc:
        return expected_schema_version, "invalid", [f"$: could not read file: {exc}"]

    if not isinstance(payload, list):
        return expected_schema_version, "invalid", ["$: expected array"]

    errors: list[str] = []
    schema_version = expected_schema_version
    for index, item in enumerate(payload):
        result = validate_artifact(item, path=path)
        if result.schema_version:
            schema_version = result.schema_version
        errors.extend(_format_collection_error(index, error) for error in result.error_messages())
    return schema_version, "valid" if not errors else "invalid", errors


def _validate_labs_validation_report(path: Path) -> tuple[str, str, list[str]]:
    try:
        with open(path) as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        return LABS_VALIDATION_SCHEMA_VERSION, "invalid", [f"$: invalid JSON: {exc.msg}"]
    except OSError as exc:
        return LABS_VALIDATION_SCHEMA_VERSION, "invalid", [f"$: could not read file: {exc}"]

    if not isinstance(payload, dict):
        return LABS_VALIDATION_SCHEMA_VERSION, "invalid", ["$: expected object"]
    results = payload.get("results")
    if not isinstance(results, list):
        return LABS_VALIDATION_SCHEMA_VERSION, "invalid", ["$.results: expected array"]
    return LABS_VALIDATION_SCHEMA_VERSION, "valid", []


def _validate_public_data_entry(path: Path, filename: str, schema_version: str) -> tuple[str, str, list[str]]:
    if filename == "labs_scorecards.json":
        return _validate_labs_collection(path, LABS_SCORECARD_SCHEMA_VERSION)
    if filename == "labs_replays.json":
        return _validate_labs_collection(path, LABS_REPLAY_SCHEMA_VERSION)
    if filename == "labs_validation.json":
        return _validate_labs_validation_report(path)
    if filename == "labs_registry.json" or "labs" in filename or "scorecard" in filename or "replay" in filename:
        result = validate_file(path)
        return (
            result.schema_version or schema_version,
            "valid" if result.valid else "invalid",
            result.error_messages(),
        )
    return schema_version, "not_applicable", []


def _discover_labs_public_paths(public_dir: Path) -> list[Path]:
    paths: set[Path] = set()
    for pattern in ("*labs*.json", "*scorecard*.json", "*replay*.json"):
        paths.update(public_dir.glob(pattern))
    return sorted(path for path in paths if path.name != "index.json")


def _public_data_entry(path: Path | None, filename: str, generated_at: str, public_dir: Path) -> dict[str, Any]:
    category, schema_version = _contract_for_filename(filename)
    if path is None or not path.exists():
        return {
            "filename": filename,
            "path": filename,
            "category": category,
            "schema_version": schema_version,
            "status": "missing",
            "validation_status": "missing",
            "validation_errors": [],
            "size_bytes": None,
            "sha256": None,
            "generated_at": generated_at,
        }

    schema_version, validation_status, validation_errors = _validate_public_data_entry(path, filename, schema_version)
    try:
        relative_path = str(path.relative_to(public_dir))
    except ValueError:
        relative_path = filename

    return {
        "filename": filename,
        "path": relative_path,
        "category": category,
        "schema_version": schema_version,
        "status": "present",
        "validation_status": validation_status,
        "validation_errors": validation_errors,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "generated_at": _generated_at_for_file(path, generated_at),
    }


def build_public_data_index(
    paths: Iterable[Path | None],
    *,
    public_dir: Path = PUBLIC_DATA_DIR,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the public data manifest while preserving the legacy files list."""
    generated_at = generated_at or datetime.now().isoformat()
    path_map: dict[str, Path] = {}
    ordered_filenames: list[str] = []

    for path in paths:
        if path is None:
            continue
        path = Path(path)
        filename = path.name
        path_map[filename] = path
        if filename not in ordered_filenames:
            ordered_filenames.append(filename)

    for path in _discover_labs_public_paths(public_dir):
        path_map.setdefault(path.name, path)
        if path.name not in ordered_filenames:
            ordered_filenames.append(path.name)

    for filename in _OPTIONAL_PUBLIC_DATA_FILES:
        if filename not in ordered_filenames:
            ordered_filenames.append(filename)

    entries = [
        _public_data_entry(path_map.get(filename), filename, generated_at, public_dir)
        for filename in ordered_filenames
    ]

    return {
        "schema_version": PUBLIC_DATA_INDEX_SCHEMA_VERSION,
        "files": [entry["filename"] for entry in entries if entry["status"] == "present"],
        "entries": entries,
        "generated_at": generated_at,
    }
