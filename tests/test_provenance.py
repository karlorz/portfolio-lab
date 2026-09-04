"""Unit tests for src.dashboard.provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dashboard.provenance import (
    PUBLIC_DATA_DIST_MIRROR_FILES,
    _apply_partial_patch_git_sha_honesty,
    _attach_dual_write_provenance,
    _attach_signal_metadata,
    _canonical_file_content_hash,
    _dist_data_dir_for_public_dir,
    _enrich_duration_allocation_provenance,
    _first_known_value,
    _mirror_public_data_contract_files_to_dist,
    _source_manifest_row_for,
    _stamp_generator_git_sha,
    _yield_source_provenance,
    finalize_dual_write_provenance_after_sync,
)


class TestPublicDataDistMirrorFiles:
    """Test PUBLIC_DATA_DIST_MIRROR_FILES contract."""

    def test_tuple_contains_required_files(self):
        assert isinstance(PUBLIC_DATA_DIST_MIRROR_FILES, tuple)
        assert "source_manifest.json" in PUBLIC_DATA_DIST_MIRROR_FILES
        assert "index.json" in PUBLIC_DATA_DIST_MIRROR_FILES
        assert "health.json" in PUBLIC_DATA_DIST_MIRROR_FILES


class TestPartialPatchGitShaHonesty:
    """Test _apply_partial_patch_git_sha_honesty behavior."""

    def test_archives_prior_sha_and_clears_current(self):
        payload = {"generator_git_sha": "abc1234567"}
        _apply_partial_patch_git_sha_honesty(payload, patch_source="test_patch")
        assert payload["last_full_generator_git_sha"] == "abc1234567"
        assert payload["generator_git_sha"] is None
        assert payload["generator_git_sha_status"] == "partial_patch"
        assert "test_patch" in payload["generator_git_sha_reason"]

    def test_empty_prior_sha(self):
        payload = {}
        _apply_partial_patch_git_sha_honesty(payload, patch_source="rebalance")
        assert "last_full_generator_git_sha" not in payload
        assert payload["generator_git_sha"] is None
        assert payload["generator_git_sha_status"] == "partial_patch"


class TestEnrichDurationAllocationProvenance:
    """Test _enrich_duration_allocation_provenance."""

    def test_none_or_empty_returns_input(self):
        assert _enrich_duration_allocation_provenance(None) is None
        assert _enrich_duration_allocation_provenance({}) == {}

    def test_flat_weights_nested_and_enriched(self):
        flat = {"tlt": 0.4, "ief": 0.3, "shy": 0.3, "bil": 0.0}
        res = _enrich_duration_allocation_provenance(flat)
        assert res is not None
        assert "weights" in res
        assert res["weights"]["tlt"] == 0.4
        assert res["sum"] == pytest.approx(1.0)
        assert res["live_authoritative"] is False
        assert res["role"] == "advisory_sleeve"
        assert res["unit"] == "portfolio_weight_fraction"


class TestSourceManifestAndYieldProvenance:
    """Test _source_manifest_row_for and _yield_source_provenance."""

    def test_source_manifest_missing_or_corrupt_returns_none(self, tmp_path: Path):
        assert _source_manifest_row_for(tmp_path, "yields.json") is None

        f = tmp_path / "source_manifest.json"
        f.write_text("corrupted", encoding="utf-8")
        assert _source_manifest_row_for(tmp_path, "yields.json") is None

    def test_source_manifest_row_lookup(self, tmp_path: Path):
        f = tmp_path / "source_manifest.json"
        manifest = {
            "artifacts": [
                {"artifact": "prices.json", "status": "ok"},
                {"filename": "yields.json", "provider": "fred", "source_mode": "api", "status": "ok"},
            ]
        }
        f.write_text(json.dumps(manifest), encoding="utf-8")

        row = _source_manifest_row_for(tmp_path, "yields.json")
        assert row is not None
        assert row["provider"] == "fred"

        yield_meta = _yield_source_provenance(tmp_path)
        assert yield_meta["source_provider"] == "fred"
        assert yield_meta["source_mode"] == "api"


class TestFirstKnownValue:
    """Test _first_known_value helper."""

    def test_skips_none_and_unknown(self):
        assert _first_known_value(None, "", "unknown", "UNKNOWN", "valid_value", "extra") == "valid_value"
        assert _first_known_value(None, "UNKNOWN", default="fallback") == "fallback"


class TestStampGeneratorGitSha:
    """Test _stamp_generator_git_sha and _attach_signal_metadata."""

    def test_attach_signal_metadata(self):
        out = _attach_signal_metadata({"signals": []}, generated_at="2026-08-17T12:00:00Z")
        assert out["generated_at"] == "2026-08-17T12:00:00Z"
        assert out["timestamp"] == "2026-08-17T12:00:00Z"

    def test_stamp_generator_git_sha(self, monkeypatch):
        from src.dashboard import generator as _gen

        monkeypatch.setattr(_gen, "_generator_git_sha_short", lambda: "1234567")
        stamped = _stamp_generator_git_sha({}, status="full_generate")
        assert stamped["generator_git_sha"] == "1234567"
        assert stamped["last_full_generator_git_sha"] == "1234567"
        assert stamped["generator_git_sha_status"] == "full_generate"

        # Update with new tip
        monkeypatch.setattr(_gen, "_generator_git_sha_short", lambda: "9999999")
        updated = _stamp_generator_git_sha(stamped, status="full_generate")
        assert updated["generator_git_sha"] == "9999999"
        assert updated["last_full_generator_git_sha"] == "1234567"


class TestCanonicalFileContentHash:
    """Test _canonical_file_content_hash."""

    def test_strips_trailing_newlines_for_consistent_hash(self, tmp_path: Path):
        f1 = tmp_path / "f1.json"
        f2 = tmp_path / "f2.json"
        f1.write_bytes(b'{"a": 1}\n')
        f2.write_bytes(b'{"a": 1}\n\n\n')

        h1 = _canonical_file_content_hash(f1)
        h2 = _canonical_file_content_hash(f2)
        assert h1 is not None
        assert h1 == h2

    def test_missing_file_returns_none(self, tmp_path: Path):
        assert _canonical_file_content_hash(tmp_path / "missing.json") is None


class TestDualWriteProvenance:
    """Test _attach_dual_write_provenance and finalize_dual_write_provenance_after_sync."""

    def test_identical_paths(self, tmp_path: Path):
        f = tmp_path / "same.json"
        f.write_text('{"data": 1}', encoding="utf-8")
        stamped = _attach_dual_write_provenance({}, private_path=f, public_path=f)
        prov = stamped["provenance_completeness"]
        assert prov["paths_identical"] is True
        assert prov["dual_write_lag_seconds"] == 0.0

    def test_content_hash_identical_clears_lag(self, tmp_path: Path):
        priv = tmp_path / "priv.json"
        pub = tmp_path / "pub.json"
        priv.write_text('{"data": 1}\n', encoding="utf-8")
        pub.write_text('{"data": 1}\n\n', encoding="utf-8")

        stamped = _attach_dual_write_provenance({}, private_path=priv, public_path=pub)
        prov = stamped["provenance_completeness"]
        assert prov["content_hash_identical"] is True
        assert prov["dual_write_lag_seconds"] == 0.0
        assert prov["dual_write_lag_stale"] is False

    def test_finalize_sync_without_write(self, tmp_path: Path):
        priv = tmp_path / "priv.json"
        pub = tmp_path / "pub.json"
        priv.write_text('{"x": 1}', encoding="utf-8")
        pub.write_text('{"x": 1}', encoding="utf-8")

        res = finalize_dual_write_provenance_after_sync(
            {"payload": "val"},
            private_path=priv,
            public_path=pub,
            write_json=False,
        )
        assert "provenance_completeness" in res


class TestDistMirroring:
    """Test _dist_data_dir_for_public_dir and _mirror_public_data_contract_files_to_dist."""

    def test_dist_data_dir_resolution(self, tmp_path: Path):
        pub = tmp_path / "public" / "data"
        dist = _dist_data_dir_for_public_dir(pub)
        assert dist == tmp_path / "dist" / "data"

    def test_mirror_public_data_contract_files_to_dist(self, tmp_path: Path):
        pub = tmp_path / "public" / "data"
        pub.mkdir(parents=True)
        (pub / "index.json").write_text('{"files": []}', encoding="utf-8")
        (pub / "health.json").write_text('{"status": "ok"}', encoding="utf-8")

        _mirror_public_data_contract_files_to_dist(pub)
        dist = tmp_path / "dist" / "data"
        assert dist.is_dir()
        assert (dist / "index.json").is_file()
        assert (dist / "health.json").is_file()
