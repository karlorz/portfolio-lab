"""Public data index manifest generation for dashboard artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.paths import PUBLIC_DATA_DIR
from src.dashboard.public_data_size_budget import (
    measure_public_data_size_budget,
    missing_public_data_size_budget,
)
from src.research.experiment_artifact_validator import (
    EXPERIMENT_DIFF_SCHEMA_VERSION,
    LABS_VALIDATION_SCHEMA_VERSION,
    LABS_REGISTRY_SCHEMA_VERSION,
    LABS_REPLAY_SCHEMA_VERSION,
    LABS_SCORECARD_SCHEMA_VERSION,
    validate_artifact,
    validate_file,
)

PUBLIC_DATA_INDEX_SCHEMA_VERSION = "public-data-index/v1"
PUBLIC_DATA_HASH_CACHE_SCHEMA_VERSION = "public-data-index-hash-cache/v1"
DEFAULT_HASH_CACHE_FILENAME = ".public_data_index_hash_cache.json"

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
    "prices.json": ("market_data", "prices/compact-v1"),
    "prices_compact.json": ("market_data", "prices/compact-v1"),
    "historical.json": ("market_data", "historical/v1"),
    "yields.json": ("market_data", "yields/v1"),
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

_LABS_OBJECT_PAGINATION_ROW_KEYS = {
    "labs_registry.json": "experiments",
    "labs_validation.json": "results",
}
_LABS_LIST_PAGINATION_FILES = {
    "labs_scorecards.json",
    "labs_replays.json",
}
_DEFAULT_LABS_PAGE_SIZE = 1000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_cache_key(path: Path, public_dir: Path) -> str:
    try:
        return path.resolve().relative_to(public_dir.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_hash_cache(cache_path: Path | None) -> dict[str, dict[str, Any]]:
    if cache_path is None or not cache_path.exists():
        return {}
    try:
        with open(cache_path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    files = payload.get("files")
    if not isinstance(files, dict):
        return {}
    return {str(key): value for key, value in files.items() if isinstance(value, dict)}


def _write_hash_cache(cache_path: Path | None, files: dict[str, dict[str, Any]] | None) -> None:
    if cache_path is None or files is None:
        return
    payload = {
        "schema_version": PUBLIC_DATA_HASH_CACHE_SCHEMA_VERSION,
        "files": files,
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        tmp_path.replace(cache_path)
    except OSError:
        return


def _cached_sha256_file(
    path: Path,
    *,
    public_dir: Path,
    hash_cache: dict[str, dict[str, Any]] | None,
    hash_cache_updates: dict[str, dict[str, Any]] | None,
) -> str:
    if hash_cache is None or hash_cache_updates is None:
        return _sha256_file(path)

    stat = path.stat()
    cache_key = _hash_cache_key(path, public_dir)
    cached = hash_cache.get(cache_key)
    if (
        isinstance(cached, dict)
        and cached.get("size_bytes") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(cached.get("sha256"), str)
        and len(cached["sha256"]) == 64
    ):
        return cached["sha256"]

    digest = _sha256_file(path)
    hash_cache_updates[cache_key] = {
        "path": cache_key,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }
    return digest


def _contract_for_filename(filename: str) -> tuple[str, str]:
    if filename in _PUBLIC_DATA_CONTRACT:
        return _PUBLIC_DATA_CONTRACT[filename]
    if "experiment_diff" in filename or "experiment-diff" in filename:
        return "labs", EXPERIMENT_DIFF_SCHEMA_VERSION
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
    if isinstance(data, list):
        generated_values = [
            item.get("generated_at")
            for item in data
            if isinstance(item, dict) and isinstance(item.get("generated_at"), str)
        ]
        if generated_values:
            return generated_values[0]
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
    errors: list[str] = []
    if payload.get("schema_version") != LABS_VALIDATION_SCHEMA_VERSION:
        errors.append(f"$.schema_version: expected {LABS_VALIDATION_SCHEMA_VERSION}")
    if not isinstance(payload.get("generated_at"), str):
        errors.append("$.generated_at: expected string")
    results = payload.get("results")
    if not isinstance(results, list):
        errors.append("$.results: expected array")
    return LABS_VALIDATION_SCHEMA_VERSION, "valid" if not errors else "invalid", errors


def _validate_public_data_entry(path: Path, filename: str, schema_version: str) -> tuple[str, str, list[str]]:
    if filename == "labs_scorecards.json":
        return _validate_labs_collection(path, LABS_SCORECARD_SCHEMA_VERSION)
    if filename == "labs_replays.json":
        return _validate_labs_collection(path, LABS_REPLAY_SCHEMA_VERSION)
    if filename == "labs_validation.json":
        return _validate_labs_validation_report(path)
    if (
        schema_version == EXPERIMENT_DIFF_SCHEMA_VERSION
        or filename == "labs_registry.json"
        or "labs" in filename
        or "scorecard" in filename
        or "replay" in filename
    ):
        result = validate_file(path)
        return (
            result.schema_version or schema_version,
            "valid" if result.valid else "invalid",
            result.error_messages(),
        )
    return schema_version, "not_applicable", []


def _discover_labs_public_paths(public_dir: Path) -> list[Path]:
    paths: set[Path] = set()
    for pattern in (
        "*labs*.json",
        "*scorecard*.json",
        "*replay*.json",
        "*experiment_diff*.json",
        "*experiment-diff*.json",
    ):
        paths.update(public_dir.glob(pattern))
    return sorted(path for path in paths if path.name != "index.json" and not _is_labs_page_shard_path(path))


def _discover_market_data_public_paths(public_dir: Path) -> list[Path]:
    filenames = ("prices.json", "prices_compact.json", "historical.json", "yields.json")
    return sorted(path for filename in filenames if (path := public_dir / filename).exists())


def _relative_public_path(path: Path, public_dir: Path, fallback: str) -> str:
    try:
        return path.relative_to(public_dir).as_posix()
    except ValueError:
        return fallback


def _is_labs_page_shard_path(path: Path) -> bool:
    return ".page-" in path.name and path.suffix == ".json"


def _labs_page_shard_name(filename: str, page: int) -> str:
    artifact_path = Path(filename)
    return f"{artifact_path.stem}.page-{page}{artifact_path.suffix}"


def _labs_page_shard_glob(filename: str) -> str:
    artifact_path = Path(filename)
    return f"{artifact_path.stem}.page-*{artifact_path.suffix}"


def _remove_labs_page_shards(public_dir: Path, filename: str, keep_filenames: set[str] | None = None) -> None:
    keep_filenames = keep_filenames or set()
    for shard_path in public_dir.glob(_labs_page_shard_glob(filename)):
        if shard_path.name in keep_filenames:
            continue
        try:
            shard_path.unlink()
        except FileNotFoundError:
            continue


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _labs_paginated_rows(payload: Any, filename: str) -> tuple[str | None, list[Any]] | None:
    row_key = _LABS_OBJECT_PAGINATION_ROW_KEYS.get(filename)
    if row_key is not None:
        if isinstance(payload, Mapping) and isinstance(payload.get(row_key), list):
            return row_key, list(payload[row_key])
        return None
    if filename in _LABS_LIST_PAGINATION_FILES and isinstance(payload, list):
        return None, list(payload)
    return None


def _labs_shard_payload(payload: Any, row_key: str | None, rows: list[Any]) -> Any:
    if row_key is None:
        return rows
    if not isinstance(payload, Mapping):
        return rows
    shard_payload = dict(payload)
    shard_payload[row_key] = rows
    return shard_payload


def _write_json_file(path: Path, payload: Any) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_labs_pagination_shards(
    path: Path,
    filename: str,
    public_dir: Path,
    size_budget: Mapping[str, Any],
    validation_status: str,
) -> dict[str, Any] | None:
    if filename not in _LABS_OBJECT_PAGINATION_ROW_KEYS and filename not in _LABS_LIST_PAGINATION_FILES:
        return None
    if size_budget.get("render_strategy") != "paginate" or validation_status != "valid":
        _remove_labs_page_shards(public_dir, filename)
        return None

    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        _remove_labs_page_shards(public_dir, filename)
        return None

    row_data = _labs_paginated_rows(payload, filename)
    if row_data is None:
        _remove_labs_page_shards(public_dir, filename)
        return None

    row_key, rows = row_data
    total_rows = len(rows)
    if total_rows == 0:
        _remove_labs_page_shards(public_dir, filename)
        return None

    page_size = _positive_int(size_budget.get("max_rows")) or _DEFAULT_LABS_PAGE_SIZE
    pages: list[dict[str, Any]] = []
    keep_filenames: set[str] = set()

    for page_index, start in enumerate(range(0, total_rows, page_size), start=1):
        page_rows = rows[start : start + page_size]
        shard_name = _labs_page_shard_name(filename, page_index)
        shard_path = public_dir / shard_name
        _write_json_file(shard_path, _labs_shard_payload(payload, row_key, page_rows))
        keep_filenames.add(shard_name)
        pages.append(
            {
                "page": page_index,
                "path": _relative_public_path(shard_path, public_dir, shard_name),
                "row_count": len(page_rows),
            }
        )

    _remove_labs_page_shards(public_dir, filename, keep_filenames)
    return {
        "total_rows": total_rows,
        "page_size": page_size,
        "page_count": len(pages),
        "pages": pages,
    }


def _public_data_entry(
    path: Path | None,
    filename: str,
    generated_at: str,
    public_dir: Path,
    *,
    hash_cache: dict[str, dict[str, Any]] | None = None,
    hash_cache_updates: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    category, schema_version = _contract_for_filename(filename)
    if path is None or not path.exists():
        _remove_labs_page_shards(public_dir, filename)
        return {
            "filename": filename,
            "path": filename,
            "category": category,
            "schema_version": schema_version,
            "status": "missing",
            "validation_status": "missing",
            "validation_errors": [],
            "size_bytes": None,
            "size_budget": missing_public_data_size_budget(),
            "sha256": None,
            "generated_at": generated_at,
        }

    schema_version, validation_status, validation_errors = _validate_public_data_entry(path, filename, schema_version)
    relative_path = _relative_public_path(path, public_dir, filename)
    size_budget = measure_public_data_size_budget(path)
    pagination = _write_labs_pagination_shards(path, filename, public_dir, size_budget, validation_status)

    entry = {
        "filename": filename,
        "path": relative_path,
        "category": category,
        "schema_version": schema_version,
        "status": "present",
        "validation_status": validation_status,
        "validation_errors": validation_errors,
        "size_bytes": path.stat().st_size,
        "size_budget": size_budget,
        "sha256": _cached_sha256_file(
            path,
            public_dir=public_dir,
            hash_cache=hash_cache,
            hash_cache_updates=hash_cache_updates,
        ),
        "generated_at": _generated_at_for_file(path, generated_at),
    }
    if pagination is not None:
        entry["pagination"] = pagination
    return entry


def build_public_data_index(
    paths: Iterable[Path | None],
    *,
    public_dir: Path = PUBLIC_DATA_DIR,
    generated_at: str | None = None,
    hash_cache_path: Path | None = None,
    use_hash_cache: bool = True,
) -> dict[str, Any]:
    """Build the public data manifest while preserving the legacy files list."""
    public_dir = Path(public_dir)
    generated_at = generated_at or datetime.now().isoformat()
    resolved_cache_path = (
        Path(hash_cache_path) if hash_cache_path is not None else public_dir / DEFAULT_HASH_CACHE_FILENAME
    )
    hash_cache = _load_hash_cache(resolved_cache_path) if use_hash_cache else None
    hash_cache_updates = dict(hash_cache) if hash_cache is not None else None
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

    for path in _discover_market_data_public_paths(public_dir):
        path_map.setdefault(path.name, path)
        if path.name not in ordered_filenames:
            ordered_filenames.append(path.name)

    for filename in _OPTIONAL_PUBLIC_DATA_FILES:
        if filename not in ordered_filenames:
            ordered_filenames.append(filename)

    entries = [
        _public_data_entry(
            path_map.get(filename),
            filename,
            generated_at,
            public_dir,
            hash_cache=hash_cache,
            hash_cache_updates=hash_cache_updates,
        )
        for filename in ordered_filenames
    ]
    _write_hash_cache(resolved_cache_path if use_hash_cache else None, hash_cache_updates)

    return {
        "schema_version": PUBLIC_DATA_INDEX_SCHEMA_VERSION,
        "files": [entry["filename"] for entry in entries if entry["status"] == "present"],
        "entries": entries,
        "generated_at": generated_at,
    }
