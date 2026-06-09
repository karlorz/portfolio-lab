"""Tests for Labs validation report generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tests.fixtures.labs import build_labs_fixture, load_labs_fixture


GENERATED_AT = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _results_by_path(report: dict) -> dict[str, dict]:
    return {entry["path"]: entry for entry in report["results"]}


def test_labs_validation_report_adds_identity_keys_for_rows_that_have_them(
    tmp_path: Path,
) -> None:
    from src.research.labs_validation_report import build_labs_validation_report

    public_dir = tmp_path / "public" / "data"
    data_dir = tmp_path / "data"
    registry_row = {
        **load_labs_fixture("valid_registry")["experiments"][0],
        "experiment_id": "registry-row-target",
        "artifact_path": "data/registry_row_target.json",
    }
    replay = {
        **load_labs_fixture("valid_replay_pass"),
        "experiment_id": "replay-target",
        "artifact_path": "data/replay_target.json",
    }
    scorecard = {
        **load_labs_fixture("valid_scorecard"),
        "experiment_id": "scorecard-target",
        "metrics": {"cagr_pct": 240.0},
    }
    provenance = {
        **load_labs_fixture("valid_provenance"),
        "experiment_id": "manifest-target",
        "source_artifact_path": "data/backtest_results/manifest_target.json",
    }
    registry_rows = _write_json(public_dir / "labs_registry_rows.json", [registry_row])
    replays = _write_json(public_dir / "labs_replays.json", [replay])
    scorecards = _write_json(public_dir / "labs_scorecards.json", [scorecard])
    manifest = _write_json(data_dir / "backtest_results" / "manifest_target.manifest.json", provenance)

    report = build_labs_validation_report(
        paths=[registry_rows, replays, scorecards, manifest],
        project_root=tmp_path,
        generated_at=GENERATED_AT,
    )

    results = _results_by_path(report)
    assert results["public/data/labs_registry_rows.json[0]"]["experiment_id"] == "registry-row-target"
    assert results["public/data/labs_registry_rows.json[0]"]["artifact_path"] == "data/registry_row_target.json"
    assert results["public/data/labs_replays.json[0]"]["experiment_id"] == "replay-target"
    assert results["public/data/labs_replays.json[0]"]["artifact_path"] == "data/replay_target.json"
    assert results["public/data/labs_scorecards.json[0]"]["experiment_id"] == "scorecard-target"
    assert "artifact_path" not in results["public/data/labs_scorecards.json[0]"]
    assert results["data/backtest_results/manifest_target.manifest.json"]["experiment_id"] == "manifest-target"
    assert (
        results["data/backtest_results/manifest_target.manifest.json"]["artifact_path"]
        == "data/backtest_results/manifest_target.json"
    )


def test_build_labs_validation_report_validates_registry_scorecard_replay_and_provenance(
    tmp_path: Path,
) -> None:
    from src.research.labs_validation_report import build_labs_validation_report

    public_dir = tmp_path / "public" / "data"
    data_dir = tmp_path / "data"
    registry = _write_json(public_dir / "labs_registry.json", load_labs_fixture("valid_registry"))
    scorecards = _write_json(public_dir / "labs_scorecards.json", [load_labs_fixture("valid_scorecard")])
    replays = _write_json(public_dir / "labs_replays.json", [load_labs_fixture("valid_replay_pass")])
    provenance = _write_json(
        data_dir / "backtest_results" / "gold_sweep.manifest.json",
        load_labs_fixture("valid_provenance"),
    )

    report = build_labs_validation_report(
        paths=[replays, scorecards, registry, provenance],
        project_root=tmp_path,
        generated_at=GENERATED_AT,
    )

    assert report["schema_version"] == "labs-validation/v1"
    assert report["generated_at"] == "2026-06-08T12:00:00+00:00"
    assert [entry["path"] for entry in report["results"]] == [
        "data/backtest_results/gold_sweep.manifest.json",
        "public/data/labs_registry.json",
        "public/data/labs_replays.json[0]",
        "public/data/labs_scorecards.json[0]",
    ]
    assert {entry["artifact_type"] for entry in report["results"]} == {
        "provenance",
        "registry",
        "replay",
        "scorecard",
    }
    assert all(entry["valid"] is True for entry in report["results"])
    assert all(entry["errors"] == [] for entry in report["results"])


def test_labs_validation_report_captures_stable_errors_without_raising(tmp_path: Path) -> None:
    from src.research.labs_validation_report import save_labs_validation_report

    public_dir = tmp_path / "public" / "data"
    invalid_registry = _write_json(public_dir / "labs_registry.json", load_labs_fixture("invalid_missing_metrics"))
    invalid_scorecards = _write_json(
        public_dir / "labs_scorecards.json",
        [build_labs_fixture("scorecard", "mixed_units")],
    )

    output_path = save_labs_validation_report(
        paths=[invalid_scorecards, invalid_registry],
        public_dir=public_dir,
        project_root=tmp_path,
        generated_at=GENERATED_AT,
    )

    report = json.loads(output_path.read_text())
    results = _results_by_path(report)
    assert output_path == public_dir / "labs_validation.json"
    assert report["schema_version"] == "labs-validation/v1"
    assert results["public/data/labs_registry.json"]["valid"] is False
    assert "$.experiments[0].metrics: missing required field" in results["public/data/labs_registry.json"]["errors"]
    assert results["public/data/labs_scorecards.json[0]"]["valid"] is False
    assert (
        "$.metrics.cagr_pct: expected percentage-point value between -100 and 100"
        in results["public/data/labs_scorecards.json[0]"]["errors"]
    )


def test_labs_validation_report_caps_rows_and_errors_with_truncation_metadata(tmp_path: Path) -> None:
    from src.research.labs_validation_report import build_labs_validation_report

    public_dir = tmp_path / "public" / "data"
    invalid_scorecards = _write_json(
        public_dir / "labs_scorecards.json",
        [
            {
                "schema_version": "labs-scorecard/v1",
                "experiment_id": "",
                "generated_at": "not-a-date",
                "status": "ship",
                "provenance_status": "lost",
                "metrics": {"cagr_pct": 240.0, "sharpe": "bad"},
                "baseline_deltas": [],
            }
            for _ in range(4)
        ],
    )

    report = build_labs_validation_report(
        paths=[invalid_scorecards],
        project_root=tmp_path,
        generated_at=GENERATED_AT,
        max_results=2,
        max_errors_per_result=2,
    )

    assert [entry["path"] for entry in report["results"]] == [
        "public/data/labs_scorecards.json[0]",
        "public/data/labs_scorecards.json[1]",
    ]
    assert all(len(entry["errors"]) == 2 for entry in report["results"])
    assert all(entry["valid"] is False for entry in report["results"])
    assert all(entry["omitted_error_count"] == 5 for entry in report["results"])
    assert report["truncation"] == {
        "max_results": 2,
        "max_errors_per_result": 2,
        "total_result_count": 4,
        "returned_result_count": 2,
        "omitted_result_count": 2,
        "omitted_error_count": 24,
    }


def test_labs_validation_report_discovery_allows_missing_optional_artifacts(tmp_path: Path) -> None:
    from src.research.labs_validation_report import save_labs_validation_report

    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public" / "data"

    output_path = save_labs_validation_report(
        data_dirs=(data_dir,),
        public_dir=public_dir,
        project_root=tmp_path,
        generated_at=GENERATED_AT,
    )

    report = json.loads(output_path.read_text())
    assert report == {
        "schema_version": "labs-validation/v1",
        "generated_at": "2026-06-08T12:00:00+00:00",
        "results": [],
    }
