"""Public data index manifest generation for dashboard artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.paths import PUBLIC_DATA_DIR
from src.monitor.decision_registry import DECISION_REGISTRY_SCHEMA_VERSION
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
    "incidents.json": ("monitoring", "incident-lifecycle/v1"),
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
    "data_quality.json": ("market_data", "price-data-quality/v1"),
    "source_manifest.json": ("market_data", "market-data-source-manifest/v1"),
    "rebalance_health.json": ("operations", "rebalance-health/v1"),
    "tasker_status.json": ("operations", "tasker-status/v1"),
    "duration-sweep-results.json": ("research_archive", "duration-sweep-results/v1"),
    "labs_registry.json": ("labs", LABS_REGISTRY_SCHEMA_VERSION),
    "labs_scorecards.json": ("labs", LABS_SCORECARD_SCHEMA_VERSION),
    "labs_replays.json": ("labs", LABS_REPLAY_SCHEMA_VERSION),
    "labs_validation.json": ("labs", LABS_VALIDATION_SCHEMA_VERSION),
    "decision_registry.json": ("monitoring", DECISION_REGISTRY_SCHEMA_VERSION),
}

_OPTIONAL_PUBLIC_DATA_FILES = (
    "data_quality.json",
    "rebalance_health.json",
    "tasker_status.json",
    "duration-sweep-results.json",
    "labs_registry.json",
    "labs_scorecards.json",
    "labs_replays.json",
    "labs_validation.json",
    "decision_registry.json",
    "incidents.json",
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
_REDISTRIBUTION_MODES = {"public_summary", "provider_derived", "restricted", "internal_only"}
_LICENSE_SCOPES = {"project_generated", "public_domain", "provider_terms", "licensed_provider", "internal"}
_PROVIDER_DERIVED_MARKET_FILES = {"prices.json", "prices_compact.json", "historical.json"}
_ARCHIVED_RESEARCH_FILES = {"duration-sweep-results.json"}


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
    if isinstance(data, dict):
        for key in ("generated_at", "generated", "timestamp"):
            if isinstance(data.get(key), str):
                return data[key]
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
    try:
        with open(path) as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        return schema_version, "invalid", [f"$: invalid JSON: {exc.msg}"]
    except OSError as exc:
        return schema_version, "invalid", [f"$: could not read file: {exc}"]

    if isinstance(payload, Mapping) and isinstance(payload.get("schema_version"), str):
        payload_schema_version = payload["schema_version"]
        if schema_version != "unknown" and payload_schema_version != schema_version:
            return (
                payload_schema_version,
                "invalid",
                [f"$.schema_version: expected {schema_version}, got {payload_schema_version}"],
            )
        return payload_schema_version, "valid", []
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
    filenames = (
        "prices.json",
        "prices_compact.json",
        "historical.json",
        "yields.json",
        "data_quality.json",
        "source_manifest.json",
    )
    return sorted(path for filename in filenames if (path := public_dir / filename).exists())


def _discover_governed_public_paths(public_dir: Path) -> list[Path]:
    filenames = (
        "rebalance_health.json",
        "tasker_status.json",
        "duration-sweep-results.json",
    )
    return sorted(path for filename in filenames if (path := public_dir / filename).exists())


def _load_source_manifest(public_dir: Path) -> dict[str, Any] | None:
    manifest_path = public_dir / "source_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    return {
        "path": "source_manifest.json",
        "schema_version": payload.get("schema_version"),
        "generated_at": payload.get("generated_at"),
        "artifacts": {
            str(row.get("artifact")): row
            for row in artifacts
            if isinstance(row, dict) and isinstance(row.get("artifact"), str)
        },
    }


_QUALITY_ISSUE_COUNT_KEYS = (
    "duplicate_dates",
    "empty_symbols",
    "extreme_returns",
    "internal_gaps",
    "invalid_dates",
    "invalid_prices",
    "missing_required_keys",
    "non_monotonic_rows",
    "non_object_records",
    "split_like_returns",
    "stale_latest_dates",
    "total",
)


def _source_quality_metadata(row: Mapping[str, Any]) -> dict[str, Any] | None:
    quality = row.get("data_quality")
    if not isinstance(quality, Mapping):
        return None

    metadata = {
        key: quality.get(key)
        for key in ("artifact", "schema_version", "generated_at", "status")
        if key in quality
    }
    issue_counts = quality.get("issue_counts")
    if isinstance(issue_counts, Mapping):
        counts = {
            key: value
            for key in _QUALITY_ISSUE_COUNT_KEYS
            if isinstance((value := issue_counts.get(key)), int) and not isinstance(value, bool)
        }
        if counts:
            metadata["issue_counts"] = counts
    return metadata or None


def _source_metadata_for(filename: str, source_manifest: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if filename == "source_manifest.json" or source_manifest is None:
        return None
    artifacts = source_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    row = artifacts.get(filename)
    if not isinstance(row, Mapping):
        return None
    metadata = {
        key: row.get(key)
        for key in (
            "provider",
            "feed",
            "source_mode",
            "status",
            "fetched_at",
            "latest_observation",
            "row_count",
            "failure_reason",
            "fallback_reason",
        )
        if key in row
    }
    data_quality = _source_quality_metadata(row)
    if data_quality is not None:
        metadata["data_quality"] = data_quality
    return metadata


def _source_manifest_row_for(filename: str, source_manifest: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if filename == "source_manifest.json" or source_manifest is None:
        return None
    artifacts = source_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    row = artifacts.get(filename)
    return row if isinstance(row, Mapping) else None


def _source_manifest_quality_artifacts(source_manifest: Mapping[str, Any] | None) -> set[str]:
    if source_manifest is None:
        return set()
    artifacts = source_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return set()
    quality_artifacts: set[str] = set()
    for row in artifacts.values():
        if not isinstance(row, Mapping):
            continue
        quality = row.get("data_quality")
        if not isinstance(quality, Mapping):
            continue
        artifact = quality.get("artifact")
        if isinstance(artifact, str) and artifact:
            quality_artifacts.add(artifact)
    return quality_artifacts


def _lineage_status_for(
    filename: str,
    category: str,
    source_manifest_row: Mapping[str, Any] | None,
    source_manifest: Mapping[str, Any] | None,
    source_manifest_quality_artifacts: set[str],
) -> str | None:
    if filename == "source_manifest.json":
        return "self_describing"
    if source_manifest_row is not None:
        return "source_manifest_row"
    if filename in source_manifest_quality_artifacts:
        return "referenced_by_source_manifest"
    if filename in _PROVIDER_DERIVED_MARKET_FILES and source_manifest is not None:
        return "missing_source_manifest_row"
    if category == "research_archive":
        return "frozen_archive"
    return None


def _missing_source_metadata_for(
    filename: str,
    source_manifest: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if filename not in _PROVIDER_DERIVED_MARKET_FILES or source_manifest is None:
        return None
    return {
        "status": "skipped",
        "failure_reason": "missing_source_manifest_row",
    }


def _source_manifest_identity(
    source_manifest: Mapping[str, Any] | None,
    entries: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if source_manifest is None:
        return None
    source_entry = next(
        (
            entry
            for entry in entries
            if entry.get("filename") == "source_manifest.json" and entry.get("status") == "present"
        ),
        None,
    )
    if source_entry is None:
        return None
    sha256 = source_entry.get("sha256")
    if not isinstance(sha256, str):
        return None
    return {
        "path": source_entry.get("path", source_manifest.get("path", "source_manifest.json")),
        "schema_version": source_manifest.get("schema_version") or source_entry.get("schema_version"),
        "generated_at": source_manifest.get("generated_at") or source_entry.get("generated_at"),
        "sha256": sha256,
    }


def _policy_value(row: Mapping[str, Any] | None, key: str, allowed: set[str], default: str) -> str:
    value = row.get(key) if row is not None else None
    if isinstance(value, str) and value in allowed:
        return value
    return default


def _public_safe_value(row: Mapping[str, Any] | None, default: bool) -> bool:
    value = row.get("public_safe") if row is not None else None
    return value if isinstance(value, bool) else default


def _public_data_policy_for(filename: str, category: str, row: Mapping[str, Any] | None) -> dict[str, Any]:
    provider = str(row.get("provider", "")).lower() if row is not None else ""
    if filename == "yields.json" or provider == "fred":
        default_mode = "public_summary"
        default_scope = "public_domain"
        default_safe = True
        default_notes = "Public macro data or derived summary suitable for dashboard publication."
    elif filename in _PROVIDER_DERIVED_MARKET_FILES:
        default_mode = "provider_derived"
        default_scope = "provider_terms"
        default_safe = True
        default_notes = "Provider-derived market data; redistribution depends on provider terms."
    elif category == "unknown":
        default_mode = "internal_only"
        default_scope = "internal"
        default_safe = False
        default_notes = "Unknown artifact contract; treat as internal until classified."
    else:
        default_mode = "public_summary"
        default_scope = "project_generated"
        default_safe = True
        default_notes = "Project-generated summary artifact suitable for dashboard publication."

    redistribution_mode = _policy_value(row, "redistribution_mode", _REDISTRIBUTION_MODES, default_mode)
    license_scope = _policy_value(row, "license_scope", _LICENSE_SCOPES, default_scope)
    public_safe = _public_safe_value(row, default_safe and redistribution_mode not in {"restricted", "internal_only"})

    return {
        "redistribution_mode": redistribution_mode,
        "license_scope": license_scope,
        "public_safe": public_safe,
        "licensing_notes": default_notes,
    }


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
    source_manifest: Mapping[str, Any] | None = None,
    source_manifest_quality_artifacts: set[str] | None = None,
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
    source_manifest_row = _source_manifest_row_for(filename, source_manifest)
    if filename == "source_manifest.json":
        sha256 = _sha256_file(path)
        if hash_cache_updates is not None:
            stat = path.stat()
            cache_key = _hash_cache_key(path, public_dir)
            hash_cache_updates[cache_key] = {
                "path": cache_key,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256,
            }
    else:
        sha256 = _cached_sha256_file(
            path,
            public_dir=public_dir,
            hash_cache=hash_cache,
            hash_cache_updates=hash_cache_updates,
        )

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
        "sha256": sha256,
        "generated_at": _generated_at_for_file(path, generated_at),
        **_public_data_policy_for(filename, category, source_manifest_row),
    }
    if pagination is not None:
        entry["pagination"] = pagination
    source_metadata = _source_metadata_for(filename, source_manifest)
    lineage_status = _lineage_status_for(
        filename,
        category,
        source_manifest_row,
        source_manifest,
        source_manifest_quality_artifacts or set(),
    )
    if lineage_status is not None:
        entry["lineage_status"] = lineage_status
    if filename in _ARCHIVED_RESEARCH_FILES:
        entry["archive_status"] = "frozen_research_artifact"
    if source_metadata is not None:
        entry["source_manifest_path"] = source_manifest.get("path", "source_manifest.json")
        entry["source_metadata"] = source_metadata
    else:
        missing_source_metadata = _missing_source_metadata_for(filename, source_manifest)
        if missing_source_metadata is not None:
            entry["source_manifest_path"] = source_manifest.get("path", "source_manifest.json")
            entry["source_metadata"] = missing_source_metadata
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
    source_manifest = _load_source_manifest(public_dir)
    source_manifest_quality_artifacts = _source_manifest_quality_artifacts(source_manifest)
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

    for path in _discover_governed_public_paths(public_dir):
        path_map.setdefault(path.name, path)
        if path.name not in ordered_filenames:
            ordered_filenames.append(path.name)

    for filename in _OPTIONAL_PUBLIC_DATA_FILES:
        path = public_dir / filename
        if filename not in path_map and path.exists():
            path_map[filename] = path
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
            source_manifest=source_manifest,
            source_manifest_quality_artifacts=source_manifest_quality_artifacts,
        )
        for filename in ordered_filenames
    ]
    _write_hash_cache(resolved_cache_path if use_hash_cache else None, hash_cache_updates)

    index = {
        "schema_version": PUBLIC_DATA_INDEX_SCHEMA_VERSION,
        "files": [entry["path"] for entry in entries if entry["status"] == "present"],
        "entries": entries,
        "generated_at": generated_at,
    }
    source_manifest_identity = _source_manifest_identity(source_manifest, entries)
    if source_manifest_identity is not None:
        index["source_manifest"] = source_manifest_identity
    return index
