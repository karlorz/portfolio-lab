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

from src.paths import BACKTEST_RESULTS_DIR, DATA_DIR, PUBLIC_DATA_DIR
from src.research.experiment_manifest import EXPERIMENT_MANIFEST_SCHEMA_VERSION

LABS_REGISTRY_SCHEMA_VERSION = "labs-registry/v1"
LABS_REPLAY_SCHEMA_VERSION = "labs-replay/v1"
LABS_SCORECARD_SCHEMA_VERSION = "labs-scorecard/v1"

SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = (
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    LABS_REGISTRY_SCHEMA_VERSION,
    LABS_REPLAY_SCHEMA_VERSION,
    LABS_SCORECARD_SCHEMA_VERSION,
)

REGISTRY_STATUS_LABELS = {"candidate", "validated", "warning", "rejected", "archived"}
SCORECARD_STATUS_LABELS = {"promote", "watch", "reject"}
REPLAY_STATUS_LABELS = {"passed", "failed", "warning"}
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
)

__all__ = [
    "ArtifactValidationResult",
    "LABS_REGISTRY_SCHEMA_VERSION",
    "LABS_REPLAY_SCHEMA_VERSION",
    "LABS_SCORECARD_SCHEMA_VERSION",
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


def _validate_scorecard(artifact: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    _validate_experiment_summary(
        artifact,
        issues,
        status_labels=SCORECARD_STATUS_LABELS,
        require_artifact_path=False,
    )


def _validate_replay(artifact: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    _validate_experiment_summary(
        artifact,
        issues,
        status_labels=REPLAY_STATUS_LABELS,
        require_artifact_path=True,
    )


def validate_artifact(
    artifact: Mapping[str, Any] | Any,
    *,
    path: str | Path | None = None,
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
    elif artifact_type == "registry":
        _validate_registry(artifact, issues)
    elif artifact_type == "scorecard":
        _validate_scorecard(artifact, issues)
    elif artifact_type == "replay":
        _validate_replay(artifact, issues)

    return ArtifactValidationResult(
        artifact_type=artifact_type,
        schema_version=schema_version if isinstance(schema_version, str) else None,
        path=Path(path) if path is not None else None,
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
    return validate_artifact(artifact, path=artifact_path)


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
