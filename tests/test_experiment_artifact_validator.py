"""Tests for Labs experiment artifact schema validation."""

from __future__ import annotations

import json
from pathlib import Path

from src.research.experiment_artifact_validator import (
    main,
    validate_artifact,
    validate_file,
    validate_paths,
)
from src.research.experiment_manifest import (
    build_experiment_manifest,
    manifest_sidecar_path,
)
from tests.fixtures.labs import load_labs_fixture

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = PROJECT_ROOT / "Makefile"


def _stub_freeze_manifest(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.research.experiment_manifest.freeze_manifest.create_manifest",
        lambda project_root=None: {
            "timestamp": "2026-06-08T00:00:00+00:00",
            "git": {"commit": "abc123", "branch": "main", "dirty": False, "tag": None},
            "config": {},
            "file_hashes": {},
            "file_count": 0,
        },
    )


def test_valid_labs_artifacts_pass_without_live_market_data() -> None:
    """Fixture Labs artifacts validate offline."""
    fixture_names = (
        "valid_registry",
        "valid_provenance",
        "valid_scorecard",
        "valid_replay_pass",
        "validation_report",
        "valid_experiment_diff",
    )
    for fixture_name in fixture_names:
        artifact = load_labs_fixture(fixture_name)
        result = validate_artifact(artifact)

        assert result.valid, result.error_messages()


def test_invalid_registry_reports_missing_required_field() -> None:
    artifact = load_labs_fixture("valid_registry")
    del artifact["experiments"][0]["artifact_path"]

    result = validate_artifact(artifact)

    assert not result.valid
    assert "$.experiments[0].artifact_path: missing required field" in result.error_messages()


def test_invalid_metrics_report_wrong_percentage_units() -> None:
    artifact = load_labs_fixture("valid_scorecard")
    artifact["metrics"]["cagr_pct"] = 240.0

    result = validate_artifact(artifact)

    assert not result.valid
    assert "$.metrics.cagr_pct: expected percentage-point value between -100 and 100" in result.error_messages()


def test_stale_schema_version_fails_with_actionable_message() -> None:
    artifact = load_labs_fixture("valid_registry")
    artifact["schema_version"] = "labs-registry/v0"

    result = validate_artifact(artifact)

    assert not result.valid
    assert (
        "$.schema_version: unsupported schema_version 'labs-registry/v0' "
        "(expected one of experiment-manifest/v1, labs-registry/v1, labs-replay/v1, "
        "labs-scorecard/v1, labs-validation/v1, experiment-diff/v1)" in result.error_messages()
    )


def test_invalid_experiment_diff_reports_nested_field() -> None:
    artifact = load_labs_fixture("valid_experiment_diff")
    del artifact["metric_deltas"]["sharpe"]["delta"]

    result = validate_artifact(artifact)

    assert not result.valid
    assert "$.metric_deltas.sharpe.delta: missing required field" in result.error_messages()


def test_invalid_labs_validation_report_requires_result_rows() -> None:
    artifact = load_labs_fixture("validation_report")
    del artifact["results"][0]["path"]

    result = validate_artifact(artifact)

    assert not result.valid
    assert "$.results[0].path: missing required field" in result.error_messages()


def test_malformed_provenance_reports_nested_field() -> None:
    artifact = load_labs_fixture("valid_provenance")
    artifact["input_file_hashes"] = ["data/prices.json"]

    result = validate_artifact(artifact)

    assert not result.valid
    assert "$.input_file_hashes: expected object" in result.error_messages()


def test_validate_file_reports_missing_recorded_input_file(tmp_path, monkeypatch) -> None:
    """Sidecar validation should diagnose recorded input files that disappear."""
    artifact = tmp_path / "walk_forward_report.json"
    input_path = tmp_path / "prices.json"
    artifact.write_text('{"summary": {"sharpe": 1.1}}')
    input_path.write_text('{"SPY": [100, 101]}')
    _stub_freeze_manifest(monkeypatch)
    manifest = build_experiment_manifest(
        experiment_id="labs:walk_forward_report",
        source_artifact_path=artifact,
        input_paths=[input_path],
    )
    sidecar_path = manifest_sidecar_path(artifact)
    sidecar_path.write_text(json.dumps(manifest))
    input_path.unlink()

    result = validate_file(sidecar_path)

    assert not result.valid
    assert any(
        message.startswith("$.input_file_hashes")
        and "recorded input file is missing" in message
        and str(input_path) in message
        for message in result.error_messages()
    )


def test_validate_file_reports_changed_recorded_input_file(tmp_path, monkeypatch) -> None:
    """Sidecar validation should diagnose recorded input files whose hashes drift."""
    artifact = tmp_path / "walk_forward_report.json"
    input_path = tmp_path / "prices.json"
    artifact.write_text('{"summary": {"sharpe": 1.1}}')
    input_path.write_text('{"SPY": [100, 101]}')
    _stub_freeze_manifest(monkeypatch)
    manifest = build_experiment_manifest(
        experiment_id="labs:walk_forward_report",
        source_artifact_path=artifact,
        input_paths=[input_path],
    )
    sidecar_path = manifest_sidecar_path(artifact)
    sidecar_path.write_text(json.dumps(manifest))
    input_path.write_text('{"SPY": [100, 102]}')

    result = validate_file(sidecar_path)

    assert not result.valid
    assert any(
        message.startswith("$.input_file_hashes")
        and "recorded input file hash mismatch" in message
        and str(input_path) in message
        for message in result.error_messages()
    )


def test_validate_file_and_paths_read_existing_artifacts_without_running_experiments(tmp_path) -> None:
    registry_path = tmp_path / "labs_registry.json"
    registry_path.write_text(json.dumps(load_labs_fixture("valid_registry")))
    invalid_path = tmp_path / "invalid_scorecard.json"
    invalid = load_labs_fixture("valid_scorecard")
    invalid["status"] = "ship"
    invalid_path.write_text(json.dumps(invalid))

    file_result = validate_file(registry_path)
    path_results = validate_paths([registry_path, invalid_path])

    assert file_result.valid
    assert [result.path for result in path_results] == [registry_path, invalid_path]
    assert path_results[0].valid
    assert not path_results[1].valid
    assert "$.status: unsupported status 'ship'" in path_results[1].error_messages()


def test_validate_file_accepts_static_labs_scorecard_collection(tmp_path) -> None:
    """Public dashboard scorecard artifacts are stored as top-level arrays."""
    scorecards_path = tmp_path / "labs_scorecards.json"
    scorecards_path.write_text(json.dumps([load_labs_fixture("valid_scorecard")]))

    result = validate_file(scorecards_path)

    assert result.valid
    assert result.artifact_type == "scorecard"
    assert result.schema_version == "labs-scorecard/v1"


def test_validate_file_reports_indexed_labs_collection_errors(tmp_path) -> None:
    scorecards_path = tmp_path / "labs_scorecards.json"
    invalid = load_labs_fixture("valid_scorecard")
    invalid["status"] = "ship"
    scorecards_path.write_text(json.dumps([invalid]))

    result = validate_file(scorecards_path)

    assert not result.valid
    assert "$[0].status: unsupported status 'ship'" in result.error_messages()


def test_cli_discover_defaults_allows_empty_artifact_set(tmp_path, monkeypatch, capsys) -> None:
    """Default discovery should be callable before Labs artifacts exist."""
    import src.research.experiment_artifact_validator as validator

    monkeypatch.setattr(validator, "DEFAULT_ARTIFACT_GLOBS", ((tmp_path, "*.json"),))

    exit_code = main(["--discover-defaults"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == {"results": []}


def test_makefile_exposes_labs_validation_target() -> None:
    """Labs validation should be discoverable and use the shared Python runtime."""
    text = MAKEFILE.read_text()

    assert "make labs-validate" in text
    assert ".PHONY: labs-validate" in text
    assert "labs-validate:" in text
    assert "$(PYTHON_RUNTIME) -m src.research.experiment_artifact_validator --discover-defaults" in text
