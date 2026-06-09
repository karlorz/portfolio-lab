"""Schema validation for Labs experiment artifacts.

The validator is intentionally dependency-free so it can run in cron and
dashboard paths without installing JSON Schema tooling.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.paths import BACKTEST_RESULTS_DIR, DATA_DIR, PROJECT_ROOT, PUBLIC_DATA_DIR
from src.research.experiment_diff import EXPERIMENT_DIFF_SCHEMA_VERSION
from src.research.experiment_manifest import EXPERIMENT_MANIFEST_SCHEMA_VERSION, file_sha256

LABS_REGISTRY_SCHEMA_VERSION = "labs-registry/v1"
LABS_REPLAY_SCHEMA_VERSION = "labs-replay/v1"
LABS_SCORECARD_SCHEMA_VERSION = "labs-scorecard/v1"
LABS_VALIDATION_SCHEMA_VERSION = "labs-validation/v1"

SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = (
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    LABS_REGISTRY_SCHEMA_VERSION,
    LABS_REPLAY_SCHEMA_VERSION,
    LABS_SCORECARD_SCHEMA_VERSION,
    LABS_VALIDATION_SCHEMA_VERSION,
    EXPERIMENT_DIFF_SCHEMA_VERSION,
)

REGISTRY_STATUS_LABELS = {"candidate", "validated", "warning", "rejected", "archived"}
SCORECARD_STATUS_LABELS = {"promote", "watch", "reject"}
SCORECARD_POLICY_THRESHOLD_FIELDS = {
    "min_promote_sharpe",
    "min_promote_sharpe_delta",
    "min_promote_dsr",
    "min_promote_wfe",
}
REPLAY_STATUS_LABELS = {"passed", "failed", "warning"}
REPLAY_FAILURE_REASON_LABELS = {
    "safety_skip",
    "timeout",
    "validation_failure",
    "command_failure",
    "unexpected_error",
}
PROVENANCE_STATUS_LABELS = {"present", "embedded", "sidecar", "missing", "stale", "malformed", "unknown"}

DEFAULT_ARTIFACT_GLOBS: tuple[tuple[Path, str], ...] = (
    (DATA_DIR, "*labs*.json"),
    (DATA_DIR, "*scorecard*.json"),
    (DATA_DIR, "*replay*.json"),
    (DATA_DIR, "*.manifest.json"),
    (BACKTEST_RESULTS_DIR, "*.manifest.json"),
    (PUBLIC_DATA_DIR, "*labs*.json"),
    (PUBLIC_DATA_DIR, "*scorecard*.json"),
    (PUBLIC_DATA_DIR, "*replay*.json"),
    (PUBLIC_DATA_DIR, "*experiment_diff*.json"),
    (PUBLIC_DATA_DIR, "*experiment-diff*.json"),
)

__all__ = [
    "ArtifactValidationResult",
    "LABS_REGISTRY_SCHEMA_VERSION",
    "LABS_REPLAY_SCHEMA_VERSION",
    "LABS_SCORECARD_SCHEMA_VERSION",
    "LABS_VALIDATION_SCHEMA_VERSION",
    "EXPERIMENT_DIFF_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "ValidationIssue",
    "discover_default_artifacts",
    "validate_artifact",
    "validate_file",
    "validate_paths",
]


@dataclass(frozen=True)
class ValidationIssue:
    """One actionable validation issue."""

    path: str
    message: str
    code: str = "schema"

    def format(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass
class ArtifactValidationResult:
    """Validation result for one artifact payload or file."""

    artifact_type: str
    schema_version: str | None
    path: Path | None = None
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues

    def error_messages(self) -> list[str]:
        return [issue.format() for issue in self.issues]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path is not None else None,
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "valid": self.valid,
            "errors": self.error_messages(),
        }


def _issue(issues: list[ValidationIssue], path: str, message: str, code: str = "schema") -> None:
    issues.append(ValidationIssue(path=path, message=message, code=code))


def _is_object(value: Any) -> bool:
    return isinstance(value, Mapping)


def _require_field(
    obj: Mapping[str, Any],
    field_name: str,
    path: str,
    issues: list[ValidationIssue],
) -> Any:
    if field_name not in obj:
        _issue(issues, f"{path}.{field_name}", "missing required field")
        return None
    return obj[field_name]


def _require_string(
    obj: Mapping[str, Any],
    field_name: str,
    path: str,
    issues: list[ValidationIssue],
    *,
    allow_empty: bool = False,
) -> str | None:
    value = _require_field(obj, field_name, path, issues)
    if value is None:
        return None
    if not isinstance(value, str):
        _issue(issues, f"{path}.{field_name}", "expected string")
        return None
    if not allow_empty and not value.strip():
        _issue(issues, f"{path}.{field_name}", "expected non-empty string")
        return None
    return value


def _optional_string(
    obj: Mapping[str, Any],
    field_name: str,
    path: str,
    issues: list[ValidationIssue],
    *,
    allow_empty: bool = False,
) -> str | None:
    if field_name not in obj:
        return None
    value = obj[field_name]
    if not isinstance(value, str):
        _issue(issues, f"{path}.{field_name}", "expected string")
        return None
    if not allow_empty and not value.strip():
        _issue(issues, f"{path}.{field_name}", "expected non-empty string")
        return None
    return value


def _require_non_negative_int(
    obj: Mapping[str, Any],
    field_name: str,
    path: str,
    issues: list[ValidationIssue],
) -> int | None:
    value = _require_field(obj, field_name, path, issues)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        _issue(issues, f"{path}.{field_name}", "expected integer")
        return None
    if value < 0:
        _issue(issues, f"{path}.{field_name}", "expected non-negative integer")
        return None
    return value


def _optional_non_negative_int(
    obj: Mapping[str, Any],
    field_name: str,
    path: str,
    issues: list[ValidationIssue],
) -> int | None:
    if field_name not in obj:
        return None
    value = obj[field_name]
    if not isinstance(value, int) or isinstance(value, bool):
        _issue(issues, f"{path}.{field_name}", "expected integer")
        return None
    if value < 0:
        _issue(issues, f"{path}.{field_name}", "expected non-negative integer")
        return None
    return value


def _require_object(
    obj: Mapping[str, Any],
    field_name: str,
    path: str,
    issues: list[ValidationIssue],
) -> Mapping[str, Any] | None:
    value = _require_field(obj, field_name, path, issues)
    if value is None:
        return None
    if not _is_object(value):
        _issue(issues, f"{path}.{field_name}", "expected object")
        return None
    return value


def _validate_generated_at(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not isinstance(value, str):
        _issue(issues, path, "expected ISO timestamp string")
        return
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _issue(issues, path, "expected ISO timestamp string")


def _validate_metric_map(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not _is_object(value):
        _issue(issues, path, "expected object")
        return
    for key, metric_value in value.items():
        metric_path = f"{path}.{key}"
        if not isinstance(metric_value, (int, float)) or isinstance(metric_value, bool):
            _issue(issues, metric_path, "expected numeric value")
            continue
        if not math.isfinite(float(metric_value)):
            _issue(issues, metric_path, "expected finite numeric value")
            continue
        if key.endswith("_pct") and not -100 <= float(metric_value) <= 100:
            _issue(issues, metric_path, "expected percentage-point value between -100 and 100")


def _validate_status(
    value: Any,
    path: str,
    labels: set[str],
    issues: list[ValidationIssue],
) -> None:
    if not isinstance(value, str):
        _issue(issues, path, "expected string")
        return
    if value not in labels:
        _issue(issues, path, f"unsupported status '{value}'")


def _validate_provenance_status(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not isinstance(value, str):
        _issue(issues, path, "expected string")
        return
    if value not in PROVENANCE_STATUS_LABELS:
        _issue(issues, path, f"unsupported provenance_status '{value}'")


def _artifact_type_for_schema(schema_version: str | None) -> str:
    if schema_version == EXPERIMENT_MANIFEST_SCHEMA_VERSION:
        return "provenance"
    if schema_version == LABS_REGISTRY_SCHEMA_VERSION:
        return "registry"
    if schema_version == LABS_SCORECARD_SCHEMA_VERSION:
        return "scorecard"
    if schema_version == LABS_REPLAY_SCHEMA_VERSION:
        return "replay"
    if schema_version == LABS_VALIDATION_SCHEMA_VERSION:
        return "validation"
    if schema_version == EXPERIMENT_DIFF_SCHEMA_VERSION:
        return "experiment_diff"
    return "unknown"


def _validate_common_generated_at(
    artifact: Mapping[str, Any],
    issues: list[ValidationIssue],
    *,
    path: str = "$",
) -> None:
    generated_at = _require_field(artifact, "generated_at", path, issues)
    if generated_at is not None:
        _validate_generated_at(generated_at, f"{path}.generated_at", issues)


def _validate_provenance(artifact: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    _require_string(artifact, "experiment_id", "$", issues)
    _validate_common_generated_at(artifact, issues)
    _require_string(artifact, "source_artifact_path", "$", issues)
    _require_object(artifact, "git", "$", issues)
    _require_object(artifact, "config_snapshot", "$", issues)
    _require_object(artifact, "environment", "$", issues)
    _require_object(artifact, "input_file_hashes", "$", issues)
    _require_object(artifact, "freeze_manifest", "$", issues)


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _candidate_file_paths(path_text: str, manifest_path: Path | None) -> list[Path]:
    path = Path(path_text)
    if path.is_absolute():
        return [path]

    candidates: list[Path] = []
    if manifest_path is not None:
        candidates.append(manifest_path.parent / path)
    candidates.extend((PROJECT_ROOT / path, path))
    return _unique_paths(candidates)


def _existing_file_path(path_text: str, manifest_path: Path | None) -> Path | None:
    for candidate in _candidate_file_paths(path_text, manifest_path):
        if candidate.exists():
            return candidate
    return None


def _same_file_reference(left: str, right: str, manifest_path: Path | None) -> bool:
    if left == right:
        return True
    left_candidates = _candidate_file_paths(left, manifest_path)
    right_candidates = _candidate_file_paths(right, manifest_path)
    left_resolved = {candidate.resolve() for candidate in left_candidates}
    return any(candidate.resolve() in left_resolved for candidate in right_candidates)


def _source_hash_entry(
    artifact: Mapping[str, Any],
    input_hashes: Mapping[str, Any],
    manifest_path: Path | None,
) -> tuple[str, str] | None:
    source_artifact_path = artifact.get("source_artifact_path")
    if not isinstance(source_artifact_path, str):
        return None
    for input_path, expected_hash in input_hashes.items():
        if not isinstance(input_path, str) or not isinstance(expected_hash, str):
            continue
        if _same_file_reference(input_path, source_artifact_path, manifest_path):
            return input_path, expected_hash
    return None


def _validate_provenance_current_hashes(
    artifact: Mapping[str, Any],
    issues: list[ValidationIssue],
    *,
    manifest_path: Path | None,
) -> None:
    source_artifact_path = artifact.get("source_artifact_path")
    input_hashes = artifact.get("input_file_hashes")
    if not isinstance(input_hashes, Mapping):
        return

    source_hash_entry = _source_hash_entry(artifact, input_hashes, manifest_path)
    source_input_path = source_hash_entry[0] if source_hash_entry is not None else None
    if isinstance(source_artifact_path, str):
        source_path = _existing_file_path(source_artifact_path, manifest_path)
        if source_path is None:
            _issue(
                issues,
                "$.source_artifact_path",
                f"source artifact is missing: {source_artifact_path}",
                "provenance_stale",
            )
        elif source_hash_entry is not None and file_sha256(source_path) != source_hash_entry[1]:
            _issue(
                issues,
                "$.source_artifact_path",
                "source artifact hash mismatch",
                "provenance_stale",
            )

    for input_path, expected_hash in input_hashes.items():
        if not isinstance(input_path, str) or not isinstance(expected_hash, str):
            continue
        if input_path == source_input_path:
            continue
        current_path = _existing_file_path(input_path, manifest_path)
        issue_path = f"$.input_file_hashes[{input_path!r}]"
        if current_path is None:
            _issue(
                issues,
                issue_path,
                f"recorded input file is missing: {input_path}",
                "provenance_stale",
            )
            continue
        if file_sha256(current_path) != expected_hash:
            _issue(
                issues,
                issue_path,
                f"recorded input file hash mismatch: {input_path}",
                "provenance_stale",
            )


def _validate_registry_entry(
    entry: Any,
    path: str,
    issues: list[ValidationIssue],
) -> None:
    if not _is_object(entry):
        _issue(issues, path, "expected object")
        return
    _require_string(entry, "experiment_id", path, issues)
    _require_string(entry, "artifact_path", path, issues)
    status = _require_field(entry, "status", path, issues)
    if status is not None:
        _validate_status(status, f"{path}.status", REGISTRY_STATUS_LABELS, issues)
    provenance_status = _require_field(entry, "provenance_status", path, issues)
    if provenance_status is not None:
        _validate_provenance_status(provenance_status, f"{path}.provenance_status", issues)
    metrics = _require_field(entry, "metrics", path, issues)
    if metrics is not None:
        _validate_metric_map(metrics, f"{path}.metrics", issues)
    baseline_deltas = _require_field(entry, "baseline_deltas", path, issues)
    if baseline_deltas is not None:
        _validate_metric_map(baseline_deltas, f"{path}.baseline_deltas", issues)


def _validate_registry(artifact: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    _validate_common_generated_at(artifact, issues)
    experiments = _require_field(artifact, "experiments", "$", issues)
    if experiments is None:
        return
    if not isinstance(experiments, list):
        _issue(issues, "$.experiments", "expected array")
        return
    for index, entry in enumerate(experiments):
        _validate_registry_entry(entry, f"$.experiments[{index}]", issues)


def _validate_experiment_summary(
    artifact: Mapping[str, Any],
    issues: list[ValidationIssue],
    *,
    status_labels: set[str],
    require_artifact_path: bool,
) -> None:
    _require_string(artifact, "experiment_id", "$", issues)
    _validate_common_generated_at(artifact, issues)
    if require_artifact_path:
        _require_string(artifact, "artifact_path", "$", issues)
    status = _require_field(artifact, "status", "$", issues)
    if status is not None:
        _validate_status(status, "$.status", status_labels, issues)
    provenance_status = _require_field(artifact, "provenance_status", "$", issues)
    if provenance_status is not None:
        _validate_provenance_status(provenance_status, "$.provenance_status", issues)
    metrics = _require_field(artifact, "metrics", "$", issues)
    if metrics is not None:
        _validate_metric_map(metrics, "$.metrics", issues)
    baseline_deltas = _require_field(artifact, "baseline_deltas", "$", issues)
    if baseline_deltas is not None:
        _validate_metric_map(baseline_deltas, "$.baseline_deltas", issues)


def _validate_non_negative_number(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _issue(issues, path, "expected numeric value")
        return
    if not math.isfinite(float(value)) or value < 0:
        _issue(issues, path, "expected non-negative finite numeric value")


def _validate_scorecard_policy(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not _is_object(value):
        _issue(issues, path, "expected object")
        return
    _require_string(value, "version", path, issues)
    thresholds = _require_object(value, "thresholds", path, issues)
    if thresholds is None:
        return
    for field_name in SCORECARD_POLICY_THRESHOLD_FIELDS:
        threshold = _require_field(thresholds, field_name, f"{path}.thresholds", issues)
        if threshold is not None:
            _validate_non_negative_number(threshold, f"{path}.thresholds.{field_name}", issues)


def _validate_scorecard(artifact: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    _validate_experiment_summary(
        artifact,
        issues,
        status_labels=SCORECARD_STATUS_LABELS,
        require_artifact_path=False,
    )
    if "policy" in artifact:
        _validate_scorecard_policy(artifact["policy"], "$.policy", issues)


def _validate_replay(artifact: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    _validate_experiment_summary(
        artifact,
        issues,
        status_labels=REPLAY_STATUS_LABELS,
        require_artifact_path=True,
    )
    failure_reason = _optional_string(artifact, "failure_reason", "$", issues)
    if failure_reason is not None and failure_reason not in REPLAY_FAILURE_REASON_LABELS:
        _issue(issues, "$.failure_reason", f"unsupported failure_reason '{failure_reason}'")
    _optional_string(artifact, "error_type", "$", issues)
    _optional_string(artifact, "error_message", "$", issues)


def _validate_validation_result(row: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not _is_object(row):
        _issue(issues, path, "expected object")
        return
    _require_string(row, "path", path, issues)
    _require_string(row, "artifact_type", path, issues)
    _require_string(row, "schema_version", path, issues, allow_empty=True)
    valid = _require_field(row, "valid", path, issues)
    if valid is not None and not isinstance(valid, bool):
        _issue(issues, f"{path}.valid", "expected boolean")
    errors = _require_field(row, "errors", path, issues)
    if errors is None:
        return
    if not isinstance(errors, list):
        _issue(issues, f"{path}.errors", "expected array")
        return
    for index, error in enumerate(errors):
        if not isinstance(error, str):
            _issue(issues, f"{path}.errors[{index}]", "expected string")
    _optional_non_negative_int(row, "omitted_error_count", path, issues)
    _optional_string(row, "experiment_id", path, issues)
    _optional_string(row, "artifact_path", path, issues)


def _validate_validation_truncation(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not _is_object(value):
        _issue(issues, path, "expected object")
        return
    for field_name in (
        "max_results",
        "max_errors_per_result",
        "total_result_count",
        "returned_result_count",
        "omitted_result_count",
        "omitted_error_count",
    ):
        _require_non_negative_int(value, field_name, path, issues)


def _validate_validation_report(artifact: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    _validate_common_generated_at(artifact, issues)
    results = _require_field(artifact, "results", "$", issues)
    if results is None:
        return
    if not isinstance(results, list):
        _issue(issues, "$.results", "expected array")
        return
    for index, row in enumerate(results):
        _validate_validation_result(row, f"$.results[{index}]", issues)
    if "truncation" in artifact:
        _validate_validation_truncation(artifact["truncation"], "$.truncation", issues)


def _validate_experiment_diff_side(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not _is_object(value):
        _issue(issues, path, "expected object")
        return
    _require_string(value, "label", path, issues)
    for field_name in ("experiment_id", "artifact_path"):
        if field_name in value and value[field_name] is not None and not isinstance(value[field_name], str):
            _issue(issues, f"{path}.{field_name}", "expected string or null")
    _require_string(value, "artifact_type", path, issues)


def _validate_metric_delta(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not _is_object(value):
        _issue(issues, path, "expected object")
        return
    for field_name in ("left", "right", "delta"):
        metric_value = _require_field(value, field_name, path, issues)
        if metric_value is None:
            continue
        if not isinstance(metric_value, (int, float)) or isinstance(metric_value, bool):
            _issue(issues, f"{path}.{field_name}", "expected numeric value")
        elif not math.isfinite(float(metric_value)):
            _issue(issues, f"{path}.{field_name}", "expected finite numeric value")


def _validate_metric_delta_map(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not _is_object(value):
        _issue(issues, path, "expected object")
        return
    for metric, delta in value.items():
        _validate_metric_delta(delta, f"{path}.{metric}", issues)


def _validate_missing_metric(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not _is_object(value):
        _issue(issues, path, "expected object")
        return
    _require_string(value, "metric", path, issues)
    missing_from = _require_field(value, "missing_from", path, issues)
    if missing_from is None:
        return
    if not isinstance(missing_from, list):
        _issue(issues, f"{path}.missing_from", "expected array")
        return
    for index, side in enumerate(missing_from):
        if side not in {"left", "right"}:
            _issue(issues, f"{path}.missing_from[{index}]", "expected 'left' or 'right'")


def _validate_config_diff(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not _is_object(value):
        _issue(issues, path, "expected object")
        return
    _require_field(value, "left", path, issues)
    _require_field(value, "right", path, issues)


def _validate_config_diff_map(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not _is_object(value):
        _issue(issues, path, "expected object")
        return
    for key, diff in value.items():
        _validate_config_diff(diff, f"{path}.{key}", issues)


def _validate_experiment_diff_provenance(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not _is_object(value):
        _issue(issues, path, "expected object")
        return
    _require_string(value, "left", path, issues)
    _require_string(value, "right", path, issues)
    changed = _require_field(value, "changed", path, issues)
    if changed is not None and not isinstance(changed, bool):
        _issue(issues, f"{path}.changed", "expected boolean")


def _validate_experiment_diff(artifact: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    _validate_common_generated_at(artifact, issues)
    left = _require_field(artifact, "left", "$", issues)
    if left is not None:
        _validate_experiment_diff_side(left, "$.left", issues)
    right = _require_field(artifact, "right", "$", issues)
    if right is not None:
        _validate_experiment_diff_side(right, "$.right", issues)
    metric_deltas = _require_field(artifact, "metric_deltas", "$", issues)
    if metric_deltas is not None:
        _validate_metric_delta_map(metric_deltas, "$.metric_deltas", issues)
    missing_metrics = _require_field(artifact, "missing_metrics", "$", issues)
    if missing_metrics is not None:
        if not isinstance(missing_metrics, list):
            _issue(issues, "$.missing_metrics", "expected array")
        else:
            for index, item in enumerate(missing_metrics):
                _validate_missing_metric(item, f"$.missing_metrics[{index}]", issues)
    config_diffs = _require_field(artifact, "config_diffs", "$", issues)
    if config_diffs is not None:
        _validate_config_diff_map(config_diffs, "$.config_diffs", issues)
    provenance = _require_field(artifact, "provenance", "$", issues)
    if provenance is not None:
        _validate_experiment_diff_provenance(provenance, "$.provenance", issues)


def validate_artifact(
    artifact: Mapping[str, Any] | Any,
    *,
    path: str | Path | None = None,
    check_current_files: bool = False,
) -> ArtifactValidationResult:
    """Validate one Labs artifact payload."""
    issues: list[ValidationIssue] = []

    if not _is_object(artifact):
        _issue(issues, "$", "expected object")
        return ArtifactValidationResult(
            artifact_type="unknown",
            schema_version=None,
            path=Path(path) if path is not None else None,
            issues=issues,
        )

    schema_version = artifact.get("schema_version")
    if not isinstance(schema_version, str):
        _issue(issues, "$.schema_version", "missing required field")
        artifact_type = "unknown"
    elif schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        expected = ", ".join(SUPPORTED_SCHEMA_VERSIONS)
        _issue(
            issues,
            "$.schema_version",
            f"unsupported schema_version '{schema_version}' (expected one of {expected})",
        )
        artifact_type = "unknown"
    else:
        artifact_type = _artifact_type_for_schema(schema_version)

    if artifact_type == "provenance":
        _validate_provenance(artifact, issues)
        if check_current_files:
            _validate_provenance_current_hashes(
                artifact,
                issues,
                manifest_path=Path(path) if path is not None else None,
            )
    elif artifact_type == "registry":
        _validate_registry(artifact, issues)
    elif artifact_type == "scorecard":
        _validate_scorecard(artifact, issues)
    elif artifact_type == "replay":
        _validate_replay(artifact, issues)
    elif artifact_type == "validation":
        _validate_validation_report(artifact, issues)
    elif artifact_type == "experiment_diff":
        _validate_experiment_diff(artifact, issues)

    return ArtifactValidationResult(
        artifact_type=artifact_type,
        schema_version=schema_version if isinstance(schema_version, str) else None,
        path=Path(path) if path is not None else None,
        issues=issues,
    )


def _prefix_collection_issue(issue: ValidationIssue, index: int) -> ValidationIssue:
    if issue.path == "$":
        path = f"$[{index}]"
    elif issue.path.startswith("$."):
        path = f"$[{index}]{issue.path[1:]}"
    elif issue.path.startswith("$["):
        path = f"$[{index}]{issue.path[1:]}"
    else:
        path = f"$[{index}].{issue.path}"
    return ValidationIssue(path=path, message=issue.message, code=issue.code)


def _schema_version_from_collection_path(path: Path) -> str | None:
    name = path.name
    if "scorecard" in name:
        return LABS_SCORECARD_SCHEMA_VERSION
    if "replay" in name:
        return LABS_REPLAY_SCHEMA_VERSION
    return None


def _validate_artifact_collection(artifacts: Sequence[Any], path: Path) -> ArtifactValidationResult:
    issues: list[ValidationIssue] = []
    schema_versions: list[str] = []
    artifact_types: list[str] = []

    for index, artifact in enumerate(artifacts):
        item_result = validate_artifact(artifact, path=path)
        issues.extend(_prefix_collection_issue(issue, index) for issue in item_result.issues)
        if item_result.schema_version is not None:
            schema_versions.append(item_result.schema_version)
        if item_result.artifact_type != "unknown":
            artifact_types.append(item_result.artifact_type)

    distinct_schema_versions = sorted(set(schema_versions))
    distinct_artifact_types = sorted(set(artifact_types))
    if len(distinct_schema_versions) > 1:
        _issue(
            issues,
            "$",
            f"mixed schema_version collection: {', '.join(distinct_schema_versions)}",
        )
        return ArtifactValidationResult(
            artifact_type="unknown",
            schema_version=None,
            path=path,
            issues=issues,
        )

    schema_version = (
        distinct_schema_versions[0]
        if distinct_schema_versions
        else _schema_version_from_collection_path(path)
    )
    artifact_type = (
        distinct_artifact_types[0]
        if len(distinct_artifact_types) == 1
        else _artifact_type_for_schema(schema_version)
    )

    return ArtifactValidationResult(
        artifact_type=artifact_type,
        schema_version=schema_version,
        path=path,
        issues=issues,
    )


def validate_file(path: str | Path) -> ArtifactValidationResult:
    """Validate one JSON artifact file without running any experiment."""
    artifact_path = Path(path)
    try:
        with open(artifact_path) as f:
            artifact = json.load(f)
    except json.JSONDecodeError as exc:
        return ArtifactValidationResult(
            artifact_type="unknown",
            schema_version=None,
            path=artifact_path,
            issues=[ValidationIssue("$", f"invalid JSON: {exc.msg}", "json")],
        )
    except OSError as exc:
        return ArtifactValidationResult(
            artifact_type="unknown",
            schema_version=None,
            path=artifact_path,
            issues=[ValidationIssue("$", f"could not read file: {exc}", "io")],
        )
    if isinstance(artifact, list):
        return _validate_artifact_collection(artifact, artifact_path)
    return validate_artifact(artifact, path=artifact_path, check_current_files=True)


def validate_paths(paths: Iterable[str | Path]) -> list[ArtifactValidationResult]:
    """Validate a sequence of artifact paths."""
    return [validate_file(path) for path in paths]


def discover_default_artifacts() -> list[Path]:
    """Find existing Labs-like artifacts in data and public data directories."""
    artifacts: set[Path] = set()
    for root, pattern in DEFAULT_ARTIFACT_GLOBS:
        if root.exists():
            artifacts.update(root.glob(pattern))
    return sorted(artifacts)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for offline artifact validation."""
    parser = argparse.ArgumentParser(description="Validate Labs experiment artifacts")
    parser.add_argument("paths", nargs="*", help="JSON artifact paths to validate")
    parser.add_argument(
        "--discover-defaults",
        action="store_true",
        help="Validate existing Labs-like artifacts under data/ and public/data/",
    )
    args = parser.parse_args(argv)

    paths = [Path(path) for path in args.paths]
    if args.discover_defaults:
        paths.extend(discover_default_artifacts())
    if not paths:
        if args.discover_defaults:
            sys.stdout.write(json.dumps({"results": []}, indent=2) + "\n")
            return 0
        parser.error("provide artifact paths or --discover-defaults")

    results = validate_paths(paths)
    sys.stdout.write(json.dumps({"results": [result.as_dict() for result in results]}, indent=2) + "\n")
    return 0 if all(result.valid for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
