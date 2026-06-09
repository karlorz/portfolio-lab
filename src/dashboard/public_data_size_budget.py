"""Public JSON payload size-budget helpers for dashboard artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

PUBLIC_DATA_SIZE_BUDGET_SCHEMA_VERSION = "public-data-size-budget/v1"

DEFAULT_WARNING_BYTES = 512 * 1024
DEFAULT_MAX_BYTES = 1024 * 1024
DEFAULT_WARNING_ROWS = 500
DEFAULT_MAX_ROWS = 1000
JSON_PARSE_BYTES_PER_MS = 50_000

__all__ = [
    "PUBLIC_DATA_SIZE_BUDGET_SCHEMA_VERSION",
    "measure_public_data_size_budget",
    "missing_public_data_size_budget",
]


def _row_count(payload: Any) -> int | None:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, Mapping):
        for key in ("experiments", "results", "rows", "entries", "files"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
        list_lengths = [len(value) for value in payload.values() if isinstance(value, list)]
        if list_lengths:
            return sum(list_lengths)
    return None


def _render_strategy(status: str, requires_pagination: bool, requires_downsampling: bool) -> str:
    if status == "missing":
        return "missing"
    if requires_pagination:
        return "paginate"
    if requires_downsampling:
        return "summarize"
    return "direct"


def _int_field(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _truncation_metadata(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    truncation = payload.get("truncation")
    if not isinstance(truncation, Mapping):
        return {}

    omitted_row_count = _int_field(truncation.get("omitted_result_count"))
    omitted_error_count = _int_field(truncation.get("omitted_error_count"))
    total_row_count = _int_field(truncation.get("total_result_count"))
    max_results = _int_field(truncation.get("max_results"))
    max_errors_per_result = _int_field(truncation.get("max_errors_per_result"))
    truncated = bool((omitted_row_count or 0) > 0 or (omitted_error_count or 0) > 0)
    return {
        "truncated": truncated,
        "total_row_count": total_row_count,
        "omitted_row_count": omitted_row_count,
        "omitted_error_count": omitted_error_count,
        "max_results": max_results,
        "max_errors_per_result": max_errors_per_result,
    }


def missing_public_data_size_budget() -> dict[str, Any]:
    """Return size-budget metadata for an absent optional public data file."""
    return {
        "schema_version": PUBLIC_DATA_SIZE_BUDGET_SCHEMA_VERSION,
        "status": "missing",
        "size_bytes": None,
        "row_count": None,
        "estimated_parse_ms": None,
        "warning_bytes": DEFAULT_WARNING_BYTES,
        "max_bytes": DEFAULT_MAX_BYTES,
        "warning_rows": DEFAULT_WARNING_ROWS,
        "max_rows": DEFAULT_MAX_ROWS,
        "requires_downsampling": False,
        "requires_pagination": False,
        "render_strategy": "missing",
    }


def measure_public_data_size_budget(
    path: str | Path,
    *,
    warning_bytes: int = DEFAULT_WARNING_BYTES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    warning_rows: int = DEFAULT_WARNING_ROWS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
    """Measure file size, row count, and dashboard render-risk metadata."""
    path = Path(path)
    size_bytes = path.stat().st_size
    row_count: int | None = None
    status = "within_budget"

    try:
        with open(path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        payload = None
        status = "invalid"
    else:
        row_count = _row_count(payload)

    size_warning = size_bytes > warning_bytes
    size_oversized = size_bytes > max_bytes
    row_warning = row_count is not None and row_count > warning_rows
    row_oversized = row_count is not None and row_count > max_rows

    if status != "invalid":
        if size_oversized or row_oversized:
            status = "oversized"
        elif size_warning or row_warning:
            status = "warning"

    requires_downsampling = status in {"warning", "oversized"} and (row_warning or row_oversized)
    requires_pagination = status == "oversized"

    return {
        "schema_version": PUBLIC_DATA_SIZE_BUDGET_SCHEMA_VERSION,
        "status": status,
        "size_bytes": size_bytes,
        "row_count": row_count,
        "estimated_parse_ms": round(max(size_bytes / JSON_PARSE_BYTES_PER_MS, 0.001), 3),
        "warning_bytes": warning_bytes,
        "max_bytes": max_bytes,
        "warning_rows": warning_rows,
        "max_rows": max_rows,
        "requires_downsampling": requires_downsampling,
        "requires_pagination": requires_pagination,
        "render_strategy": _render_strategy(status, requires_pagination, requires_downsampling),
        **_truncation_metadata(payload),
    }
