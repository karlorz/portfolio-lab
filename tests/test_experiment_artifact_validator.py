"""Tests for Labs experiment artifact schema validation."""

from __future__ import annotations

import json

from src.research.experiment_artifact_validator import (
    main,
    validate_artifact,
    validate_file,
    validate_paths,
)
from tests.fixtures.labs import load_labs_fixture


def test_valid_labs_artifacts_pass_without_live_market_data() -> None:
    """Fixture registry, provenance, scorecard, and replay artifacts validate offline."""
    fixture_names = ("valid_registry", "valid_provenance", "valid_scorecard", "valid_replay_pass")
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
    assert (
        "$.metrics.cagr_pct: expected percentage-point value between -100 and 100"
        in result.error_messages()
    )


def test_stale_schema_version_fails_with_actionable_message() -> None:
    artifact = load_labs_fixture("valid_registry")
    artifact["schema_version"] = "labs-registry/v0"

    result = validate_artifact(artifact)

    assert not result.valid
    assert (
        "$.schema_version: unsupported schema_version 'labs-registry/v0' "
        "(expected one of experiment-manifest/v1, labs-registry/v1, labs-replay/v1, labs-scorecard/v1)"
        in result.error_messages()
    )


def test_malformed_provenance_reports_nested_field() -> None:
    artifact = load_labs_fixture("valid_provenance")
    artifact["input_file_hashes"] = ["data/prices.json"]

    result = validate_artifact(artifact)

    assert not result.valid
    assert "$.input_file_hashes: expected object" in result.error_messages()


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


def test_cli_discover_defaults_allows_empty_artifact_set(tmp_path, monkeypatch, capsys) -> None:
    """Default discovery should be callable before Labs artifacts exist."""
    import src.research.experiment_artifact_validator as validator

    monkeypatch.setattr(validator, "DEFAULT_ARTIFACT_GLOBS", ((tmp_path, "*.json"),))

    exit_code = main(["--discover-defaults"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == {"results": []}
