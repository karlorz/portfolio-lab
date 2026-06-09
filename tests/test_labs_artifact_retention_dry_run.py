"""Tests for Labs artifact archive dry-run planning."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


FIXED_NOW = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _set_age(path: Path, days_old: int) -> None:
    timestamp = (FIXED_NOW - timedelta(days=days_old)).timestamp()
    os.utime(path, (timestamp, timestamp))


def test_archive_dry_run_plan_is_deterministic_and_does_not_move_files(tmp_path: Path) -> None:
    from src.research.artifact_retention import build_archive_dry_run_plan

    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public" / "data"
    archive_candidate = _write_json(data_dir / "backtest_results" / "old_result.json", {"sharpe": 1.1})
    active_result = _write_json(data_dir / "backtest_results" / "active_result.json", {"sharpe": 1.2})
    public_dashboard = _write_json(public_dir / "dashboard.json", {"generated_at": "2026-06-08T00:00:00"})
    for path, age in ((archive_candidate, 240), (active_result, 20), (public_dashboard, 240)):
        _set_age(path, age)

    plan = build_archive_dry_run_plan(
        data_dir=data_dir,
        public_data_dir=public_dir,
        project_root=tmp_path,
        reference_roots=[],
        now=FIXED_NOW,
    )

    assert plan["schema_version"] == "labs-artifact-archive-plan/v1"
    assert plan["source_report_schema_version"] == "artifact-retention-report/v1"
    assert plan["generated_at"] == "2026-06-08T12:00:00+00:00"
    assert plan["dry_run"] is True
    assert plan["move_enabled"] is False
    assert plan["guardrails"] == {
        "destructive_actions_allowed": False,
        "requires_explicit_move_opt_in": True,
        "move_opt_in_flag": "--execute-move",
    }
    assert plan["counts"] == {
        "move_candidates": 1,
        "protected": 2,
        "source_entries": 3,
    }
    assert plan["move_candidates"] == [
        {
            "source_path": "data/backtest_results/old_result.json",
            "planned_archive_path": "data/archive/data/backtest_results/old_result.json",
            "category": "experiment_result",
            "age_days": 240,
            "size_bytes": archive_candidate.stat().st_size,
            "reason_codes": ["archive_candidate", "age_gte_180_days"],
        }
    ]
    protected_by_path = {entry["source_path"]: entry for entry in plan["protected"]}
    assert protected_by_path["data/backtest_results/active_result.json"]["reason_codes"] == [
        "active_retention_window",
    ]
    assert protected_by_path["public/data/dashboard.json"]["reason_codes"] == [
        "recent_dashboard_output",
    ]
    assert archive_candidate.exists()
    assert active_result.exists()
    assert public_dashboard.exists()
    assert not (data_dir / "archive").exists()


def test_archive_dry_run_plan_protects_referenced_and_public_index_artifacts(tmp_path: Path) -> None:
    from src.research.artifact_retention import build_archive_dry_run_plan

    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public" / "data"
    registry_target = _write_json(data_dir / "backtest_results" / "registry_target.json", {"metric": 1})
    manifest_target = _write_json(data_dir / "backtest_results" / "manifest_target.json", {"metric": 2})
    wiki_target = _write_json(data_dir / "backtest_results" / "wiki_target.json", {"metric": 3})
    public_index_target = _write_json(public_dir / "labs_replays.json", [{"schema_version": "labs-replay/v1"}])
    for path in (registry_target, manifest_target, wiki_target, public_index_target):
        path.write_text(path.read_text() + (" " * 2048))
        _set_age(path, 365)

    _write_json(
        data_dir / "labs_registry.json",
        {
            "schema_version": "labs-registry/v1",
            "generated_at": "2026-06-08T00:00:00",
            "experiments": [
                {
                    "experiment_id": "registry-target",
                    "artifact_path": "data/backtest_results/registry_target.json",
                    "status": "validated",
                    "provenance_status": "present",
                    "metrics": {},
                    "baseline_deltas": {},
                }
            ],
        },
    )
    _write_json(
        data_dir / "backtest_results" / "manifest_target.json.manifest.json",
        {
            "schema_version": "experiment-manifest/v1",
            "experiment_id": "manifest-target",
            "generated_at": "2026-06-08T00:00:00",
            "source_artifact_path": "data/backtest_results/manifest_target.json",
            "git": {},
            "config_snapshot": {},
            "environment": {},
            "input_file_hashes": {},
            "freeze_manifest": {"config": {}, "file_hashes": {}, "file_count": 0},
        },
    )
    reference_root = tmp_path / "wiki"
    _write_text(reference_root / "compound.md", "Reviewed data/backtest_results/wiki_target.json\n")
    _write_json(
        public_dir / "index.json",
        {
            "schema_version": "public-data-index/v1",
            "files": ["labs_replays.json"],
            "entries": [
                {
                    "filename": "labs_replays.json",
                    "path": "labs_replays.json",
                    "category": "labs",
                    "schema_version": "labs-replay/v1",
                    "status": "present",
                    "validation_status": "valid",
                    "validation_errors": [],
                    "size_bytes": public_index_target.stat().st_size,
                    "size_budget": {"render_strategy": "direct"},
                    "sha256": "0" * 64,
                    "generated_at": "2026-06-08T00:00:00",
                }
            ],
            "generated_at": "2026-06-08T00:00:00",
        },
    )

    plan = build_archive_dry_run_plan(
        data_dir=data_dir,
        public_data_dir=public_dir,
        project_root=tmp_path,
        reference_roots=[reference_root],
        now=FIXED_NOW,
    )

    assert plan["move_candidates"] == []
    protected_by_path = {entry["source_path"]: entry for entry in plan["protected"]}
    for source_path in (
        "data/backtest_results/registry_target.json",
        "data/backtest_results/manifest_target.json",
        "data/backtest_results/wiki_target.json",
    ):
        entry = protected_by_path[source_path]
        assert "protected_reference" in entry["reason_codes"]
        assert entry["referenced_by"]

    public_entry = protected_by_path["public/data/labs_replays.json"]
    assert public_entry["reason_codes"] == [
        "protected_reference",
        "public_index_reference",
    ]
    assert "public/data/index.json:entries[0].path" in public_entry["referenced_by"]
    assert registry_target.exists()
    assert manifest_target.exists()
    assert wiki_target.exists()
    assert public_index_target.exists()


def test_archive_plan_cli_outputs_plan_and_refuses_move_execution(tmp_path: Path, capsys) -> None:
    from scripts.report_artifact_retention import main

    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public" / "data"
    archive_candidate = _write_json(data_dir / "backtest_results" / "old_result.json", {"sharpe": 1.1})
    _set_age(archive_candidate, 240)

    exit_code = main(
        [
            "--archive-plan",
            "--data-dir",
            str(data_dir),
            "--public-data-dir",
            str(public_dir),
            "--project-root",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    plan = json.loads(captured.out)
    assert exit_code == 0
    assert plan["schema_version"] == "labs-artifact-archive-plan/v1"
    assert plan["move_enabled"] is False
    assert plan["move_candidates"][0]["source_path"] == "data/backtest_results/old_result.json"
    assert archive_candidate.exists()

    refusal_code = main(
        [
            "--archive-plan",
            "--execute-move",
            "--data-dir",
            str(data_dir),
            "--public-data-dir",
            str(public_dir),
            "--project-root",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert refusal_code == 2
    assert "Move execution is not implemented" in captured.err
    assert captured.out == ""
    assert archive_candidate.exists()
