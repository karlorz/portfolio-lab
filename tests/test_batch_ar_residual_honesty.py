"""Batch AR residual honesty: public-data generator_git_sha provenance canary."""

from __future__ import annotations

import json
from pathlib import Path



def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_provenance_canary_warns_on_missing_sha(tmp_path):
    from scripts.check_public_data_consistency import (
        PROVENANCE_CONTRACT_FILES,
        _check_generator_git_sha_provenance,
    )

    public = tmp_path / "public" / "data"
    # Minimal stamped-required operator file without sha
    _write(
        public / "alerts.json",
        {"alerts": [], "count": 0, "generated_at": "2026-07-21T00:00:00+00:00"},
    )
    errors: list[str] = []
    warnings: list[str] = []
    _check_generator_git_sha_provenance(public, errors, warnings)
    assert errors == []
    assert any("alerts.json" in w and "missing generator_git_sha" in w for w in warnings)
    assert "alerts.json" in PROVENANCE_CONTRACT_FILES


def test_provenance_canary_accepts_unavailable_status(tmp_path):
    from scripts.check_public_data_consistency import _check_generator_git_sha_provenance

    public = tmp_path / "public" / "data"
    _write(
        public / "decision_registry.json",
        {
            "schema_version": "decision-registry/v1",
            "generated_at": "2026-07-21T00:00:00+00:00",
            "generator_git_sha": None,
            "generator_git_sha_status": "unavailable",
            "counts": {},
        },
    )
    errors: list[str] = []
    warnings: list[str] = []
    _check_generator_git_sha_provenance(public, errors, warnings)
    assert errors == []
    assert warnings == []


def test_provenance_canary_errors_on_dishonest_full_status(tmp_path):
    from scripts.check_public_data_consistency import _check_generator_git_sha_provenance

    public = tmp_path / "public" / "data"
    _write(
        public / "stats.json",
        {
            "generated_at": "2026-07-21T00:00:00+00:00",
            "generator_git_sha": None,
            "generator_git_sha_status": "full_generate",
            "asset_stats": {},
        },
    )
    errors: list[str] = []
    warnings: list[str] = []
    _check_generator_git_sha_provenance(public, errors, warnings)
    assert any("dishonest provenance" in e for e in errors)
    assert warnings == []


def test_provenance_canary_ok_when_sha_present(tmp_path):
    from scripts.check_public_data_consistency import _check_generator_git_sha_provenance

    public = tmp_path / "public" / "data"
    _write(
        public / "index.json",
        {
            "schema_version": "public-data-index/v1",
            "files": [],
            "entries": [],
            "generated_at": "2026-07-21T00:00:00+00:00",
            "generator_git_sha": "abc123def456",
            "generator_git_sha_status": "full_generate",
        },
    )
    errors: list[str] = []
    warnings: list[str] = []
    _check_generator_git_sha_provenance(public, errors, warnings)
    assert errors == []
    assert warnings == []


def test_check_public_data_consistency_includes_provenance_canary(tmp_path, monkeypatch):
    """Integration: canary runs inside check_public_data_consistency."""
    from scripts import check_public_data_consistency as cpc

    app = tmp_path / "app"
    public = app / "public" / "data"
    public.mkdir(parents=True)
    # Required files for resolution path
    _write(
        public / "source_manifest.json",
        {
            "schema_version": "market-data-source-manifest/v1",
            "generated_at": "2026-07-21T12:00:00+00:00",
            "symbol_universe": {},
            "artifacts": [],
            "generator_git_sha": "deadbeefcafe",
            "generator_git_sha_status": "full_generate",
        },
    )
    _write(
        public / "index.json",
        {
            "schema_version": "public-data-index/v1",
            "files": ["source_manifest.json", "health.json"],
            "entries": [
                {
                    "filename": "source_manifest.json",
                    "path": "source_manifest.json",
                    "status": "present",
                    "generated_at": "2026-07-21T12:00:00+00:00",
                },
                {
                    "filename": "health.json",
                    "path": "health.json",
                    "status": "present",
                    "generated_at": "2026-07-21T12:00:00+00:00",
                },
            ],
            "generated_at": "2026-07-21T12:00:01+00:00",
            "generator_git_sha": "deadbeefcafe",
            "generator_git_sha_status": "full_generate",
            "source_manifest": {
                "path": "source_manifest.json",
                "schema_version": "market-data-source-manifest/v1",
                "generated_at": "2026-07-21T12:00:00+00:00",
                "sha256": __import__("hashlib")
                .sha256((public / "source_manifest.json").read_bytes())
                .hexdigest(),
            },
        },
    )
    _write(
        public / "health.json",
        {
            "generated_at": "2026-07-21T12:00:00+00:00",
            "system_status": "healthy",
            "generator_git_sha": "deadbeefcafe",
            "generator_git_sha_status": "full_generate",
        },
    )
    # Unstamped optional contract file → warning only
    _write(
        public / "alerts.json",
        {"alerts": [], "count": 0, "generated_at": "2026-07-21T12:00:00+00:00"},
    )

    result = cpc.check_public_data_consistency(
        app,
        public_dir=public,
        allow_repo_public_data=True,
        env={"PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA": "1"},
    )
    # May have other warnings/errors depending on host rules; provenance warn required
    assert any("alerts.json" in w and "missing generator_git_sha" in w for w in result.warnings)


def test_batch_ar_source_contract():
    src = Path("scripts/check_public_data_consistency.py").read_text(encoding="utf-8")
    assert "PROVENANCE_CONTRACT_FILES" in src
    assert "_check_generator_git_sha_provenance" in src
    assert "dishonest provenance" in src
