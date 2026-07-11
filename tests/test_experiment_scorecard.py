"""Tests for deterministic Labs experiment scorecard generation."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from src.dashboard.public_data_index import build_public_data_index
from src.research.experiment_artifact_validator import validate_artifact


def _scorecard_module():
    try:
        return importlib.import_module("src.research.experiment_scorecard")
    except ModuleNotFoundError:
        pytest.fail("src.research.experiment_scorecard is missing")


def _registry(rows: list[dict]) -> dict:
    return {
        "schema_version": "labs-registry/v1",
        "generated_at": "2026-06-09T00:00:00+00:00",
        "experiments": rows,
    }


def _registry_row(
    experiment_id: str,
    *,
    status: str = "candidate",
    provenance_status: str = "sidecar",
    metrics: dict[str, float] | None = None,
    baseline_deltas: dict[str, float] | None = None,
) -> dict:
    return {
        "experiment_id": experiment_id,
        "artifact_path": f"data/{experiment_id}.json",
        "status": status,
        "provenance_status": provenance_status,
        "metrics": metrics or {},
        "baseline_deltas": baseline_deltas or {},
    }


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_scorecard_generator_classifies_registry_rows_conservatively() -> None:
    module = _scorecard_module()

    scorecards = module.build_labs_scorecards(
        registry=_registry(
            [
                _registry_row(
                    "promote-me",
                    status="validated",
                    provenance_status="sidecar",
                    metrics={
                        "sharpe": 1.02,
                        "wfe": 1.18,
                        "dsr": 0.97,
                        "max_drawdown_pct": -18.0,
                    },
                    baseline_deltas={"sharpe": 0.06, "max_drawdown_pct": 2.5},
                ),
                _registry_row(
                    "watch-me",
                    status="candidate",
                    provenance_status="missing",
                    metrics={"sharpe": 0.88},
                    baseline_deltas={"sharpe": 0.03},
                ),
                _registry_row(
                    "reject-me",
                    status="warning",
                    provenance_status="malformed",
                    metrics={"sharpe": 0.42, "max_drawdown_pct": -34.0},
                    baseline_deltas={"sharpe": -0.12, "max_drawdown_pct": -5.0},
                ),
            ]
        ),
        generated_at="2026-06-09T01:00:00+00:00",
    )

    by_id = {row["experiment_id"]: row for row in scorecards}
    assert by_id["promote-me"]["status"] == "promote"
    assert by_id["watch-me"]["status"] == "watch"
    assert by_id["reject-me"]["status"] == "reject"
    assert by_id["promote-me"]["generated_at"] == "2026-06-09T01:00:00+00:00"
    assert by_id["promote-me"]["metrics"]["wfe"] == 1.18
    assert by_id["promote-me"]["baseline_deltas"]["sharpe"] == 0.06
    assert by_id["promote-me"]["policy"]["version"] == "default-v1"
    assert by_id["promote-me"]["policy"]["thresholds"]["min_promote_sharpe"] == 0.9
    assert all(validate_artifact(row).valid for row in scorecards)


def test_scorecard_discloses_missing_provenance_governance_block_for_strong_metrics() -> None:
    module = _scorecard_module()

    scorecards = module.build_labs_scorecards(
        registry=_registry(
            [
                _registry_row(
                    "metric-only",
                    status="candidate",
                    provenance_status="missing",
                    metrics={"sharpe": 1.21, "dsr": 0.98, "wfe": 1.1},
                    baseline_deltas={"sharpe": 0.08, "max_drawdown_pct": 2.0},
                )
            ]
        ),
        generated_at="2026-06-09T01:00:00+00:00",
    )

    assert scorecards[0]["status"] == "watch"
    assert scorecards[0]["governance_state"] == "governance_blocked"
    assert scorecards[0]["governance_reasons"] == ["provenance_missing"]
    assert scorecards[0]["promotion_governance"]["recommended_status"] == "candidate"
    assert scorecards[0]["promotion_governance"]["failures"] == ["provenance_missing"]
    assert validate_artifact(scorecards[0]).valid is True


def test_scorecard_rejects_invalid_validation_or_failed_replay_inputs() -> None:
    module = _scorecard_module()

    scorecards = module.build_labs_scorecards(
        registry=_registry(
            [
                _registry_row(
                    "fragile",
                    status="validated",
                    provenance_status="embedded",
                    metrics={"sharpe": 1.1, "dsr": 0.96},
                    baseline_deltas={"sharpe": 0.08},
                )
            ]
        ),
        validation_report={
            "schema_version": "labs-validation/v1",
            "generated_at": "2026-06-09T00:30:00+00:00",
            "results": [
                {
                    "path": "public/data/labs_registry.json",
                    "artifact_type": "registry",
                    "schema_version": "labs-registry/v1",
                    "valid": False,
                    "errors": ["registry row failed validation"],
                    "experiment_id": "fragile",
                }
            ],
        },
        replays=[
            {
                "schema_version": "labs-replay/v1",
                "experiment_id": "fragile",
                "generated_at": "2026-06-09T00:45:00+00:00",
                "artifact_path": "data/fragile.json",
                "status": "failed",
                "provenance_status": "embedded",
                "passed": False,
                "metrics": {},
                "baseline_deltas": {},
            }
        ],
        generated_at="2026-06-09T01:00:00+00:00",
    )

    assert len(scorecards) == 1
    assert scorecards[0]["experiment_id"] == "fragile"
    assert scorecards[0]["status"] == "reject"
    assert validate_artifact(scorecards[0]).valid is True


def test_scorecard_policy_override_changes_classification(tmp_path: Path) -> None:
    module = _scorecard_module()
    policy_path = _write_json(
        tmp_path / "scorecard-policy.json",
        {
            "version": "loose-research-v1",
            "thresholds": {
                "min_promote_sharpe": 0.8,
                "min_promote_sharpe_delta": 0.01,
                "min_promote_dsr": 0.9,
                "min_promote_wfe": 0.75,
            },
        },
    )

    scorecards = module.build_labs_scorecards(
        registry=_registry(
            [
                _registry_row(
                    "policy-sensitive",
                    status="validated",
                    provenance_status="sidecar",
                    metrics={"sharpe": 0.82, "dsr": 0.91, "wfe": 0.8},
                    baseline_deltas={"sharpe": 0.02, "max_drawdown_pct": 0.5},
                )
            ]
        ),
        policy_path=policy_path,
        generated_at="2026-06-09T01:00:00+00:00",
    )

    assert scorecards[0]["status"] == "promote"
    assert scorecards[0]["policy"] == {
        "version": "loose-research-v1",
        "thresholds": {
            "min_promote_sharpe": 0.8,
            "min_promote_sharpe_delta": 0.01,
            "min_promote_dsr": 0.9,
            "min_promote_wfe": 0.75,
        },
    }
    assert validate_artifact(scorecards[0]).valid is True


def test_invalid_scorecard_policy_override_falls_back_to_defaults(tmp_path: Path) -> None:
    module = _scorecard_module()
    policy_path = _write_json(
        tmp_path / "bad-policy.json",
        {
            "version": "bad-policy",
            "thresholds": {
                "min_promote_sharpe": -1.0,
                "min_promote_sharpe_delta": 0.0,
                "min_promote_dsr": 0.0,
                "min_promote_wfe": 0.0,
            },
        },
    )

    scorecards = module.build_labs_scorecards(
        registry=_registry(
            [
                _registry_row(
                    "would-promote-under-bad-policy",
                    status="validated",
                    provenance_status="sidecar",
                    metrics={"sharpe": 0.82, "dsr": 0.91, "wfe": 0.8},
                    baseline_deltas={"sharpe": 0.02, "max_drawdown_pct": 0.5},
                )
            ]
        ),
        policy_path=policy_path,
        generated_at="2026-06-09T01:00:00+00:00",
    )

    assert scorecards[0]["status"] == "watch"
    assert scorecards[0]["policy"]["version"] == "default-v1"
    assert scorecards[0]["policy"]["thresholds"]["min_promote_sharpe_delta"] == 0.04
    assert validate_artifact(scorecards[0]).valid is True


def test_save_scorecards_writes_valid_public_artifact_and_index_entry(tmp_path: Path) -> None:
    module = _scorecard_module()
    public_dir = tmp_path / "public" / "data"
    registry_path = _write_json(
        public_dir / "labs_registry.json",
        _registry(
            [
                _registry_row(
                    "scorecard-target",
                    status="validated",
                    provenance_status="sidecar",
                    metrics={"sharpe": 1.04, "dsr": 0.98},
                    baseline_deltas={"sharpe": 0.07},
                )
            ]
        ),
    )

    output_path = module.save_labs_scorecards(
        registry_path=registry_path,
        public_dir=public_dir,
        generated_at="2026-06-09T01:00:00+00:00",
    )

    assert output_path == public_dir / "labs_scorecards.json"
    scorecards = json.loads(output_path.read_text())
    assert scorecards[0]["experiment_id"] == "scorecard-target"
    assert scorecards[0]["status"] == "promote"
    assert validate_artifact(scorecards[0]).valid is True

    index = build_public_data_index([output_path], public_dir=public_dir, generated_at="2026-06-09T01:30:00")
    scorecard_entry = {entry["filename"]: entry for entry in index["entries"]}["labs_scorecards.json"]
    assert scorecard_entry["schema_version"] == "labs-scorecard/v1"
    assert scorecard_entry["status"] == "present"
    assert scorecard_entry["validation_status"] == "valid"
    assert scorecard_entry["size_budget"]["render_strategy"] == "direct"
