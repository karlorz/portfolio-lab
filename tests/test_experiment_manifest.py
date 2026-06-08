"""Tests for experiment artifact provenance manifests."""

import json

from src.research.experiment_manifest import (
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    backfill_experiment_manifests,
    build_experiment_manifest,
    file_sha256,
    save_experiment_result_json,
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

    written = backfill_experiment_manifests([artifact, missing], experiment_id_prefix="labs")

    assert len(written) == 1
    assert written[0].name == "walk_forward_report.json.manifest.json"
    manifest = json.loads(written[0].read_text())
    assert manifest["experiment_id"] == "labs:walk_forward_report"
    assert manifest["input_file_hashes"][str(artifact)] == file_sha256(artifact)
    assert json.loads(artifact.read_text()) == {"summary": {"sharpe": 1.1}}
