"""Tests for shared Labs experiment fixture artifacts."""

from __future__ import annotations

from src.research.experiment_artifact_validator import validate_artifact
from tests.fixtures.labs import (
    LABS_FIXTURE_NAMES,
    build_labs_fixture,
    iter_labs_fixture_paths,
    labs_fixture_path,
    load_labs_fixture,
)


def test_labs_fixture_files_exist_and_stay_small() -> None:
    expected = {
        "valid_registry",
        "valid_provenance",
        "valid_scorecard",
        "valid_replay_pass",
        "valid_replay_fail",
        "validation_report",
        "invalid_missing_metrics",
        "invalid_mixed_units",
        "stale_schema",
        "dirty_provenance",
        "valid_experiment_diff",
    }

    assert set(LABS_FIXTURE_NAMES) == expected
    for path in iter_labs_fixture_paths():
        assert path.exists(), path
        assert path.stat().st_size < 4096, f"{path.name} is too large for a fast fixture"


def test_valid_labs_fixture_files_validate_without_live_data() -> None:
    fixture_names = [
        "valid_registry",
        "valid_provenance",
        "valid_scorecard",
        "valid_replay_pass",
        "valid_replay_fail",
        "dirty_provenance",
    ]

    for name in fixture_names:
        result = validate_artifact(load_labs_fixture(name), path=labs_fixture_path(name))
        assert result.valid, result.error_messages()


def test_invalid_labs_fixture_files_capture_expected_failure_modes() -> None:
    invalid_cases = {
        "invalid_missing_metrics": "$.experiments[0].metrics: missing required field",
        "invalid_mixed_units": "$.metrics.cagr_pct: expected percentage-point value between -100 and 100",
        "stale_schema": "$.schema_version: unsupported schema_version 'labs-registry/v0'",
    }

    for name, expected_message in invalid_cases.items():
        result = validate_artifact(load_labs_fixture(name), path=labs_fixture_path(name))
        assert not result.valid
        assert any(expected_message in message for message in result.error_messages())


def test_labs_fixture_builders_produce_named_variants() -> None:
    stale_registry = build_labs_fixture("registry", variant="stale_schema")
    missing_metrics = build_labs_fixture("registry", variant="missing_metrics")
    mixed_units = build_labs_fixture("scorecard", variant="mixed_units")
    dirty_provenance = build_labs_fixture("provenance", variant="dirty")
    replay_drift = build_labs_fixture("replay", variant="drift_fail")

    assert stale_registry["schema_version"] == "labs-registry/v0"
    assert "metrics" not in missing_metrics["experiments"][0]
    assert mixed_units["metrics"]["cagr_pct"] == 240.0
    assert dirty_provenance["git"]["dirty"] is True
    assert replay_drift["status"] == "failed"
    assert replay_drift["metrics"]["max_abs_metric_delta"] > 0


def test_validation_report_fixture_describes_validator_errors() -> None:
    report = load_labs_fixture("validation_report")

    assert report["results"][0]["artifact_type"] == "registry"
    assert report["results"][0]["valid"] is False
    assert report["results"][0]["errors"]
