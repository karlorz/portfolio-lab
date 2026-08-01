"""Generate the public Labs validation report artifact."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.paths import DATA_DIR, PUBLIC_DATA_DIR
from src.research.experiment_artifact_validator import (
    LABS_VALIDATION_SCHEMA_VERSION,
    ArtifactValidationResult,
    validate_artifact,
    validate_file,
)

LABS_VALIDATION_FILENAME = "labs_validation.json"
DEFAULT_MAX_VALIDATION_RESULTS = int(os.getenv("LABS_VALIDATION_MAX_RESULTS", "500"))
DEFAULT_MAX_VALIDATION_ERRORS_PER_RESULT = int(os.getenv("LABS_VALIDATION_MAX_ERRORS_PER_RESULT", "20"))

__all__ = [
    "DEFAULT_MAX_VALIDATION_ERRORS_PER_RESULT",
    "DEFAULT_MAX_VALIDATION_RESULTS",
    "LABS_VALIDATION_FILENAME",
    "build_labs_validation_report",
    "discover_labs_validation_artifacts",
    "save_labs_validation_report",
]


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        # Public validation rows may originate in the live WWW tree or another
        # worker checkout. Keep the row useful without exposing that host path.
        from src.dashboard.public_projection import logical_reference

        return logical_reference(path)


def _generated_at(value: datetime | str | None) -> str:
    if isinstance(value, str):
        return value
    return (value or datetime.now(timezone.utc)).isoformat()


def _is_validation_candidate(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name in {LABS_VALIDATION_FILENAME, "index.json"}:
        return False
    return True


def discover_labs_validation_artifacts(
    *,
    data_dirs: Sequence[str | Path] = (DATA_DIR,),
    public_dir: str | Path = PUBLIC_DATA_DIR,
) -> list[Path]:
    """Discover existing Labs artifacts without treating missing optional files as errors."""
    candidates: set[Path] = set()
    patterns = ("*labs*.json", "*scorecard*.json", "*replay*.json", "*.manifest.json")
    roots = [Path(root) for root in data_dirs] + [Path(public_dir)]
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.update(root.glob(pattern))
        candidates.update((root / "backtest_results").glob("*.manifest.json"))
    return sorted(path for path in candidates if _is_validation_candidate(path))


def _identity_fields(payload: Any) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        return {}

    fields: dict[str, str] = {}
    experiment_id = payload.get("experiment_id")
    if isinstance(experiment_id, str) and experiment_id.strip():
        fields["experiment_id"] = experiment_id

    artifact_path = payload.get("artifact_path")
    if not isinstance(artifact_path, str):
        artifact_path = payload.get("source_artifact_path")
    if isinstance(artifact_path, str) and artifact_path.strip():
        fields["artifact_path"] = artifact_path

    return fields


def _result_row(
    result: ArtifactValidationResult,
    path: str,
    *,
    payload: Any = None,
) -> dict[str, Any]:
    row = {
        "path": path,
        "artifact_type": result.artifact_type,
        "schema_version": result.schema_version,
        "valid": result.valid,
        "errors": result.error_messages(),
    }
    row.update(_identity_fields(payload))
    return row


def _rows_for_path(path: Path, project_root: Path) -> list[dict[str, Any]]:
    display_path = _display_path(path, project_root)
    try:
        with open(path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return [_result_row(validate_file(path), display_path)]

    if isinstance(payload, list):
        return [
            _result_row(validate_artifact(item), f"{display_path}[{index}]", payload=item)
            for index, item in enumerate(payload)
        ]
    return [_result_row(validate_artifact(payload), display_path, payload=payload)]


def _limit(value: int | None, default: int) -> int | None:
    if value is None:
        return None
    return max(int(value), 0)


def _error_count(row: dict[str, Any]) -> int:
    errors = row.get("errors")
    return len(errors) if isinstance(errors, list) else 0


def _apply_report_caps(
    rows: list[dict[str, Any]],
    *,
    max_results: int | None,
    max_errors_per_result: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    result_limit = _limit(max_results, DEFAULT_MAX_VALIDATION_RESULTS)
    error_limit = _limit(max_errors_per_result, DEFAULT_MAX_VALIDATION_ERRORS_PER_RESULT)
    total_result_count = len(rows)
    total_error_count = sum(_error_count(row) for row in rows)
    capped_rows = rows if result_limit is None else rows[:result_limit]
    retained_rows: list[dict[str, Any]] = []

    for row in capped_rows:
        next_row = dict(row)
        errors = list(next_row.get("errors", []))
        if error_limit is not None and len(errors) > error_limit:
            next_row["errors"] = errors[:error_limit]
            next_row["omitted_error_count"] = len(errors) - error_limit
        retained_rows.append(next_row)

    returned_error_count = sum(_error_count(row) for row in retained_rows)
    omitted_result_count = total_result_count - len(retained_rows)
    omitted_error_count = total_error_count - returned_error_count
    if omitted_result_count == 0 and omitted_error_count == 0:
        return retained_rows, None

    return retained_rows, {
        "max_results": result_limit if result_limit is not None else total_result_count,
        "max_errors_per_result": error_limit if error_limit is not None else total_error_count,
        "total_result_count": total_result_count,
        "returned_result_count": len(retained_rows),
        "omitted_result_count": omitted_result_count,
        "omitted_error_count": omitted_error_count,
    }


def build_labs_validation_report(
    *,
    paths: Iterable[str | Path] | None = None,
    data_dirs: Sequence[str | Path] = (DATA_DIR,),
    public_dir: str | Path = PUBLIC_DATA_DIR,
    project_root: str | Path = DATA_DIR.parent,
    generated_at: datetime | str | None = None,
    max_results: int | None = DEFAULT_MAX_VALIDATION_RESULTS,
    max_errors_per_result: int | None = DEFAULT_MAX_VALIDATION_ERRORS_PER_RESULT,
) -> dict[str, Any]:
    """Validate Labs artifacts and return a versioned public report payload."""
    project_root_path = Path(project_root)
    artifact_paths = (
        [Path(path) for path in paths]
        if paths is not None
        else discover_labs_validation_artifacts(data_dirs=data_dirs, public_dir=public_dir)
    )
    rows = [
        row
        for path in sorted(artifact_paths, key=lambda item: _display_path(Path(item), project_root_path))
        for row in _rows_for_path(Path(path), project_root_path)
    ]
    rows.sort(key=lambda row: row["path"])
    rows, truncation = _apply_report_caps(
        rows,
        max_results=max_results,
        max_errors_per_result=max_errors_per_result,
    )
    report = {
        "schema_version": LABS_VALIDATION_SCHEMA_VERSION,
        "generated_at": _generated_at(generated_at),
        "results": rows,
    }
    if truncation is not None:
        report["truncation"] = truncation
    return report


def save_labs_validation_report(
    *,
    paths: Iterable[str | Path] | None = None,
    data_dirs: Sequence[str | Path] = (DATA_DIR,),
    public_dir: str | Path = PUBLIC_DATA_DIR,
    project_root: str | Path = DATA_DIR.parent,
    output_path: str | Path | None = None,
    generated_at: datetime | str | None = None,
    max_results: int | None = DEFAULT_MAX_VALIDATION_RESULTS,
    max_errors_per_result: int | None = DEFAULT_MAX_VALIDATION_ERRORS_PER_RESULT,
) -> Path:
    """Write `labs_validation.json` and return its path."""
    public_dir_path = Path(public_dir)
    target_path = Path(output_path) if output_path is not None else public_dir_path / LABS_VALIDATION_FILENAME
    report = build_labs_validation_report(
        paths=paths,
        data_dirs=data_dirs,
        public_dir=public_dir_path,
        project_root=project_root,
        generated_at=generated_at,
        max_results=max_results,
        max_errors_per_result=max_errors_per_result,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    from src.monitor.signal_authority import (
        is_ephemeral_write_path,
        serialize_json_payload,
    )

    target_path.write_text(
        serialize_json_payload(
            report,
            output_path=target_path,
            public=not is_ephemeral_write_path(target_path),
        ),
        encoding="utf-8",
    )
    return target_path
