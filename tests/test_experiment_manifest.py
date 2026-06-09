"""Tests for experiment artifact provenance manifests."""

import json
from pathlib import Path

from src.research.experiment_manifest import (
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    backfill_experiment_manifests,
    build_experiment_manifest,
    file_sha256,
    main,
    manifest_sidecar_path,
    save_experiment_result_json,
)
from src.research.experiment_artifact_validator import validate_artifact
from src.research.experiment_artifact_validator import validate_file


def _stub_freeze_manifest(monkeypatch):
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


def test_file_sha256_is_deterministic(tmp_path):
    """File hashes should be stable and content-derived."""
    path = tmp_path / "prices.json"
    path.write_text('{"SPY": [100, 101]}')

    first = file_sha256(path)
    second = file_sha256(path)

    assert first == second
    assert len(first) == 64


def test_build_experiment_manifest_captures_required_fields(tmp_path, monkeypatch):
    """Manifest should capture git/config provenance and input file hashes."""
    source = tmp_path / "result.json"
    source.write_text("{}")
    prices = tmp_path / "prices.json"
    prices.write_text('{"SPY": [100, 101]}')

    def fake_freeze_manifest(project_root=None):
        return {
            "timestamp": "2026-06-08T00:00:00+00:00",
            "git": {"commit": "abc123", "branch": "main", "dirty": True, "tag": "abc123"},
            "config": {"LOG_LEVEL": "DEBUG"},
            "file_hashes": {"src/example.py": "hash"},
            "file_count": 1,
        }

    monkeypatch.setattr(
        "src.research.experiment_manifest.freeze_manifest.create_manifest",
        fake_freeze_manifest,
    )

    manifest = build_experiment_manifest(
        experiment_id="gold-sweep",
        source_artifact_path=source,
        command="python -m src.backtest.gold_allocation_sweep run",
        config_snapshot={"min_gold": 0.2},
        env_keys=("LOG_LEVEL",),
        input_paths=[prices],
    )

    assert manifest["schema_version"] == EXPERIMENT_MANIFEST_SCHEMA_VERSION
    assert manifest["experiment_id"] == "gold-sweep"
    assert manifest["source_artifact_path"].endswith("result.json")
    assert manifest["command"] == "python -m src.backtest.gold_allocation_sweep run"
    assert manifest["config_snapshot"] == {"min_gold": 0.2}
    assert manifest["environment"] == {"LOG_LEVEL": "DEBUG"}
    assert manifest["git"] == {"commit": "abc123", "branch": "main", "dirty": True, "tag": "abc123"}
    assert manifest["freeze_manifest"]["file_count"] == 1
    assert manifest["input_file_hashes"][str(prices)] == file_sha256(prices)


def test_save_experiment_result_json_embeds_provenance(tmp_path, monkeypatch):
    """Embedded mode should write a registry-readable provenance block."""
    output = tmp_path / "artifact.json"

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

    save_experiment_result_json(
        {"rows": [{"sharpe": 1.0}]},
        output,
        experiment_id="combined-regime",
        manifest_mode="embedded",
        command="pytest fixture",
    )

    loaded = json.loads(output.read_text())
    assert loaded["rows"][0]["sharpe"] == 1.0
    assert loaded["_provenance"]["experiment_id"] == "combined-regime"
    assert loaded["_provenance"]["source_artifact_path"].endswith("artifact.json")


def test_save_experiment_result_json_writes_sidecar_without_changing_payload(tmp_path, monkeypatch):
    """Sidecar mode should preserve legacy artifact shape."""
    output = tmp_path / "legacy.json"

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

    manifest_path = save_experiment_result_json(
        {"summary": {"sharpe": 0.95}},
        output,
        experiment_id="legacy-artifact",
        manifest_mode="sidecar",
    )

    assert json.loads(output.read_text()) == {"summary": {"sharpe": 0.95}}
    manifest = json.loads(manifest_path.read_text())
    assert manifest["experiment_id"] == "legacy-artifact"
    assert manifest["source_artifact_path"].endswith("legacy.json")


def test_backfill_experiment_manifests_writes_sidecars_for_existing_artifacts(tmp_path, monkeypatch):
    """Backfill should add sidecars for existing artifacts and skip missing paths."""
    artifact = tmp_path / "walk_forward_report.json"
    missing = tmp_path / "missing.json"
    artifact.write_text('{"summary": {"sharpe": 1.1}}')

    _stub_freeze_manifest(monkeypatch)

    written = backfill_experiment_manifests([artifact, missing], experiment_id_prefix="labs")

    assert len(written) == 1
    assert written[0].name == "walk_forward_report.json.manifest.json"
    manifest = json.loads(written[0].read_text())
    assert manifest["experiment_id"] == "labs:walk_forward_report"
    assert manifest["input_file_hashes"][str(artifact)] == file_sha256(artifact)
    assert json.loads(artifact.read_text()) == {"summary": {"sharpe": 1.1}}


def test_validate_file_keeps_unchanged_sidecar_manifest_valid(tmp_path, monkeypatch):
    """Sidecar provenance should stay valid when recorded file hashes still match."""
    artifact = tmp_path / "walk_forward_report.json"
    artifact.write_text('{"summary": {"sharpe": 1.1}}')
    _stub_freeze_manifest(monkeypatch)
    manifest = build_experiment_manifest(
        experiment_id="labs:walk_forward_report",
        source_artifact_path=artifact,
        input_paths=[artifact],
    )
    sidecar_path = manifest_sidecar_path(artifact)
    sidecar_path.write_text(json.dumps(manifest))

    result = validate_file(sidecar_path)

    assert result.valid, result.error_messages()


def test_validate_file_reports_stale_sidecar_when_source_artifact_changes(tmp_path, monkeypatch):
    """Sidecar provenance should diagnose a source artifact hash mismatch."""
    artifact = tmp_path / "walk_forward_report.json"
    artifact.write_text('{"summary": {"sharpe": 1.1}}')
    _stub_freeze_manifest(monkeypatch)
    manifest = build_experiment_manifest(
        experiment_id="labs:walk_forward_report",
        source_artifact_path=artifact,
        input_paths=[artifact],
    )
    sidecar_path = manifest_sidecar_path(artifact)
    sidecar_path.write_text(json.dumps(manifest))
    artifact.write_text('{"summary": {"sharpe": 0.9}}')

    result = validate_file(sidecar_path)

    assert not result.valid
    assert "$.source_artifact_path: source artifact hash mismatch" in result.error_messages()


def test_backfill_cli_dry_run_lists_targets_without_writing_sidecars(tmp_path, monkeypatch, capsys):
    """Dry-run should report target artifacts and sidecars without writing files."""
    artifact = tmp_path / "walk_forward_report.json"
    artifact.write_text('{"summary": {"sharpe": 1.1}}')
    _stub_freeze_manifest(monkeypatch)

    exit_code = main(["backfill", "--dry-run", str(artifact)])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary["dry_run"] is True
    assert summary["targeted"] is True
    assert summary["targets"] == 1
    assert summary["written"] == []
    assert summary["artifacts"][0]["artifact_path"] == str(artifact)
    assert summary["artifacts"][0]["sidecar_path"] == str(manifest_sidecar_path(artifact))
    assert summary["artifacts"][0]["status"] == "dry_run"
    assert not manifest_sidecar_path(artifact).exists()


def test_backfill_cli_targets_globs_and_writes_valid_sidecars(tmp_path, monkeypatch, capsys):
    """Targeted glob backfill should write sidecars that pass Labs provenance validation."""
    artifact_a = tmp_path / "walk_forward_report.json"
    artifact_b = tmp_path / "gold_allocation_sweep.json"
    artifact_a.write_text('{"summary": {"sharpe": 1.1}}')
    artifact_b.write_text('{"summary": {"sharpe": 0.9}}')
    _stub_freeze_manifest(monkeypatch)

    exit_code = main(["backfill", "--experiment-id-prefix", "labs", str(tmp_path / "*.json")])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    written = [item["sidecar_path"] for item in summary["artifacts"] if item["status"] == "written"]
    assert exit_code == 0
    assert summary["dry_run"] is False
    assert summary["targeted"] is True
    assert summary["targets"] == 2
    assert len(written) == 2
    for sidecar_path in written:
        manifest = json.loads(Path(sidecar_path).read_text())
        assert manifest["experiment_id"].startswith("labs:")
        assert validate_artifact(manifest).valid


def test_backfill_cli_reports_missing_target_artifacts(tmp_path, monkeypatch, capsys):
    """Explicit missing targets should be reported instead of silently succeeding."""
    missing = tmp_path / "missing.json"
    _stub_freeze_manifest(monkeypatch)

    exit_code = main(["backfill", str(missing)])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 1
    assert summary["missing"] == 1
    assert summary["artifacts"] == [
        {
            "artifact_path": str(missing),
            "sidecar_path": str(manifest_sidecar_path(missing)),
            "status": "missing",
            "errors": ["artifact does not exist"],
        }
    ]
    assert not manifest_sidecar_path(missing).exists()


def test_backfill_cli_preserves_legacy_backfill_default_discovery(tmp_path, monkeypatch, capsys):
    """The legacy `backfill` command should still use default artifact discovery."""
    artifact = tmp_path / "walk_forward_report.json"
    backtest_results_dir = tmp_path / "backtest_results"
    backtest_results_dir.mkdir()
    artifact.write_text('{"summary": {"sharpe": 1.1}}')
    _stub_freeze_manifest(monkeypatch)
    monkeypatch.setattr("src.research.experiment_manifest.DEFAULT_EXPERIMENT_ARTIFACTS", (artifact,))
    monkeypatch.setattr("src.research.experiment_manifest.BACKTEST_RESULTS_DIR", backtest_results_dir)

    exit_code = main(["backfill"])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary["targeted"] is False
    assert summary["written"] == [str(manifest_sidecar_path(artifact))]
    assert manifest_sidecar_path(artifact).exists()


def test_backfill_cli_blocks_invalid_generated_manifests(tmp_path, monkeypatch, capsys):
    """Generated manifests should validate before the CLI writes sidecars."""
    artifact = tmp_path / "walk_forward_report.json"
    artifact.write_text('{"summary": {"sharpe": 1.1}}')
    _stub_freeze_manifest(monkeypatch)
    monkeypatch.setattr(
        "src.research.experiment_manifest.build_experiment_manifest",
        lambda **kwargs: {"schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION},
    )

    exit_code = main(["backfill", str(artifact)])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 1
    assert summary["invalid"] == 1
    assert summary["artifacts"][0]["status"] == "invalid"
    assert "$.experiment_id: missing required field" in summary["artifacts"][0]["errors"]
    assert not manifest_sidecar_path(artifact).exists()
