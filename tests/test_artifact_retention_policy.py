"""Tests for Labs artifact retention dry-run reporting."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _set_age(path: Path, days_old: int) -> None:
    timestamp = (datetime.now(timezone.utc) - timedelta(days=days_old)).timestamp()
    os.utime(path, (timestamp, timestamp))


def _entries_by_name(report: dict) -> dict[str, dict]:
    return {Path(entry["path"]).name: entry for entry in report["entries"]}


def test_retention_report_classifies_representative_artifact_families(tmp_path: Path) -> None:
    from src.research.artifact_retention import build_retention_report

    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public" / "data"
    files = [
        _write_json(data_dir / "backtest_results" / "old_result.json", {"summary": {"sharpe": 1.2}}),
        _write_json(data_dir / "historical_orders" / "orders_2026.json", {"orders": []}),
        _write_json(data_dir / "attribution" / "attribution_20260608.json", {"sources": {}}),
        _write_json(data_dir / "llm_costs" / "costs_20260608.json", {"total_usd": 1.23}),
        _write_json(public_dir / "dashboard.json", {"generated_at": "2026-06-08T00:00:00"}),
        _write_text(data_dir / "logs" / "old_cron.log", "cron line\n"),
    ]
    for path in files:
        _set_age(path, 240)

    report = build_retention_report(data_dir=data_dir, public_data_dir=public_dir, project_root=tmp_path)

    assert report["schema_version"] == "artifact-retention-report/v1"
    assert report["dry_run"] is True
    assert report["archive_root"] == "data/archive"

    entries = _entries_by_name(report)
    assert entries["old_result.json"]["category"] == "experiment_result"
    assert entries["old_result.json"]["recommendation"] == "archive"
    assert entries["orders_2026.json"]["category"] == "operational_history"
    assert entries["orders_2026.json"]["recommendation"] == "archive"
    assert entries["attribution_20260608.json"]["category"] == "attribution_snapshot"
    assert entries["attribution_20260608.json"]["recommendation"] == "archive"
    assert entries["costs_20260608.json"]["category"] == "cost_history"
    assert entries["costs_20260608.json"]["recommendation"] == "archive"
    assert entries["dashboard.json"]["category"] == "dashboard_state"
    assert entries["dashboard.json"]["recommendation"] == "keep"
    assert entries["old_cron.log"]["category"] == "raw_log"
    assert entries["old_cron.log"]["recommendation"] == "prune"

    for path in files:
        assert path.exists(), "retention reporting must not delete or move files"


def test_referenced_registry_and_manifest_artifacts_are_protected(tmp_path: Path) -> None:
    from src.research.artifact_retention import build_retention_report

    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public" / "data"
    registry_target = _write_json(data_dir / "backtest_results" / "registry_target.json", {"metric": 1})
    manifest_target = _write_json(data_dir / "backtest_results" / "manifest_target.json", {"metric": 2})
    wiki_target = _write_json(data_dir / "backtest_results" / "wiki_target.json", {"metric": 3})
    for path in (registry_target, manifest_target, wiki_target):
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

    report = build_retention_report(
        data_dir=data_dir,
        public_data_dir=public_dir,
        project_root=tmp_path,
        reference_roots=[reference_root],
    )

    entries = _entries_by_name(report)
    for filename in ("registry_target.json", "manifest_target.json", "wiki_target.json"):
        assert entries[filename]["recommendation"] == "keep"
        assert entries[filename]["protected"] is True
        assert entries[filename]["retention_tier"] == "protected_reference"
        assert entries[filename]["referenced_by"]


def test_text_reference_scan_checks_each_text_file_once_for_many_artifacts(tmp_path: Path, monkeypatch) -> None:
    from src.research import artifact_retention

    class CountingText(str):
        contains_calls = 0

        def __contains__(self, item: object) -> bool:
            CountingText.contains_calls += 1
            return super().__contains__(item)

    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public" / "data"
    target_name = "artifact_042.json"
    target_reference = f"data/backtest_results/{target_name}"
    for index in range(80):
        path = data_dir / "backtest_results" / f"artifact_{index:03d}.json"
        _write_json(path, {"metric": index})
        _set_age(path, 365)

    reference_root = tmp_path / "wiki"
    reference_doc = _write_text(reference_root / "compound.md", f"Reviewed {target_reference}\n")
    original_read_text = Path.read_text

    def read_counting_text(path: Path, *args, **kwargs) -> str:
        text = original_read_text(path, *args, **kwargs)
        if path == reference_doc:
            return CountingText(text)
        return text

    monkeypatch.setattr(Path, "read_text", read_counting_text)

    report = artifact_retention.build_retention_report(
        data_dir=data_dir,
        public_data_dir=public_dir,
        project_root=tmp_path,
        reference_roots=[reference_root],
    )

    entries = _entries_by_name(report)
    assert entries[target_name]["referenced_by"] == ["wiki/compound.md"]
    assert CountingText.contains_calls == 0


def test_report_cli_outputs_json_and_stays_report_only(tmp_path: Path, capsys) -> None:
    from scripts.report_artifact_retention import main

    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public" / "data"
    stale_log = _write_text(data_dir / "logs" / "stale.log", "old\n")
    _set_age(stale_log, 365)

    exit_code = main([
        "--data-dir",
        str(data_dir),
        "--public-data-dir",
        str(public_dir),
        "--project-root",
        str(tmp_path),
    ])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert report["dry_run"] is True
    assert report["counts"]["prune"] == 1
    assert stale_log.exists()
