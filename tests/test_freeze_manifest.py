"""Tests for freeze manifest module — config drift detection."""

import json
from pathlib import Path

import pytest

from src.monitor.freeze_manifest import (
    create_manifest,
    diff_manifests,
    load_manifest,
    save_manifest,
)


class TestCreateManifest:
    """Test freeze manifest creation."""

    def test_manifest_has_required_fields(self):
        """Manifest should contain timestamp, git, config, file_hashes."""
        manifest = create_manifest()
        assert "timestamp" in manifest
        assert "git" in manifest
        assert "config" in manifest
        assert "file_hashes" in manifest
        assert "file_count" in manifest

    def test_manifest_timestamp_is_iso_format(self):
        """Timestamp should be ISO format."""
        manifest = create_manifest()
        ts = manifest["timestamp"]
        assert "T" in ts  # ISO format has T separator

    def test_manifest_file_count_matches_hashes(self):
        """file_count should equal number of file_hashes."""
        manifest = create_manifest()
        assert manifest["file_count"] == len(manifest["file_hashes"])

    def test_config_captures_env_vars(self):
        """Config section should capture known env vars."""
        manifest = create_manifest()
        config = manifest["config"]
        assert "ALPHALAB_MODE" in config
        assert "JSON_LOGS" in config
        assert "LOG_LEVEL" in config

    def test_git_state_structure(self):
        """Git state should have commit, branch, dirty, tag keys."""
        manifest = create_manifest()
        git = manifest["git"]
        assert "commit" in git
        assert "branch" in git
        assert "dirty" in git
        assert "tag" in git

    def test_git_commit_is_short_hash(self):
        """Git commit should be a short hash (12 chars)."""
        manifest = create_manifest()
        commit = manifest["git"]["commit"]
        if commit is not None:  # Git may not be available
            assert len(commit) == 12

    def test_file_hashes_include_pyproject(self):
        """File hashes should include pyproject.toml."""
        manifest = create_manifest()
        assert any("pyproject.toml" in f for f in manifest["file_hashes"])

    def test_file_hashes_include_source_files(self):
        """File hashes should include .py source files."""
        manifest = create_manifest()
        py_files = [f for f in manifest["file_hashes"] if f.endswith(".py")]
        assert len(py_files) > 10  # Should have many source files


class TestSaveLoadManifest:
    """Test manifest save/load round-trip."""

    def test_save_and_load(self, tmp_path):
        """Save then load should produce identical manifest."""
        manifest = create_manifest()
        path = tmp_path / "freeze_manifest.json"
        save_manifest(manifest, path=path)

        loaded = load_manifest(path=path)
        assert loaded is not None
        assert loaded["timestamp"] == manifest["timestamp"]
        assert loaded["file_count"] == manifest["file_count"]
        assert loaded["git"] == manifest["git"]

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Loading a nonexistent file should return None."""
        result = load_manifest(path=tmp_path / "nonexistent.json")
        assert result is None

    def test_load_corrupt_json_returns_none(self, tmp_path):
        """Loading a corrupt JSON file should return None."""
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json{{{")
        result = load_manifest(path=path)
        assert result is None

    def test_save_creates_parent_dirs(self, tmp_path):
        """Save should create parent directories if needed."""
        path = tmp_path / "sub" / "dir" / "manifest.json"
        save_manifest({"test": True}, path=path)
        assert path.exists()


class TestDiffManifests:
    """Test manifest diffing for drift detection."""

    def test_no_drift(self):
        """Identical manifests should show no drift."""
        manifest = create_manifest()
        diff = diff_manifests(manifest, manifest)
        assert diff["drifted"] is False
        assert diff["git_changed"] is False
        assert len(diff["config_drift"]) == 0

    def test_git_commit_drift(self):
        """Different git commits should be detected."""
        baseline = {"git": {"commit": "abc12345678", "branch": "main", "dirty": False, "tag": None}, "config": {}, "file_hashes": {}}
        current = {"git": {"commit": "def789012345", "branch": "main", "dirty": False, "tag": None}, "config": {}, "file_hashes": {}}
        diff = diff_manifests(baseline, current)
        assert diff["drifted"] is True
        assert diff["git_changed"] is True

    def test_config_drift_detected(self):
        """Changed env var should be detected."""
        baseline = {"git": {}, "config": {"LOG_LEVEL": "INFO"}, "file_hashes": {}}
        current = {"git": {}, "config": {"LOG_LEVEL": "DEBUG"}, "file_hashes": {}}
        diff = diff_manifests(baseline, current)
        assert diff["drifted"] is True
        assert "LOG_LEVEL" in diff["config_drift"]
        assert diff["config_drift"]["LOG_LEVEL"]["from"] == "INFO"
        assert diff["config_drift"]["LOG_LEVEL"]["to"] == "DEBUG"

    def test_file_added_detected(self):
        """New files should be detected."""
        baseline = {"git": {}, "config": {}, "file_hashes": {"src/a.py": "hash1"}}
        current = {"git": {}, "config": {}, "file_hashes": {"src/a.py": "hash1", "src/b.py": "hash2"}}
        diff = diff_manifests(baseline, current)
        assert diff["drifted"] is True
        assert "src/b.py" in diff["file_changes"]["added"]

    def test_file_removed_detected(self):
        """Removed files should be detected."""
        baseline = {"git": {}, "config": {}, "file_hashes": {"src/a.py": "hash1", "src/b.py": "hash2"}}
        current = {"git": {}, "config": {}, "file_hashes": {"src/a.py": "hash1"}}
        diff = diff_manifests(baseline, current)
        assert diff["drifted"] is True
        assert "src/b.py" in diff["file_changes"]["removed"]

    def test_file_modified_detected(self):
        """Modified files should be detected."""
        baseline = {"git": {}, "config": {}, "file_hashes": {"src/a.py": "hash1"}}
        current = {"git": {}, "config": {}, "file_hashes": {"src/a.py": "hash2"}}
        diff = diff_manifests(baseline, current)
        assert diff["drifted"] is True
        assert "src/a.py" in diff["file_changes"]["modified"]

    def test_dirty_flag_drift(self):
        """Git dirty flag change should be detected."""
        baseline = {"git": {"commit": "abc", "branch": "main", "dirty": False, "tag": None}, "config": {}, "file_hashes": {}}
        current = {"git": {"commit": "abc", "branch": "main", "dirty": True, "tag": None}, "config": {}, "file_hashes": {}}
        diff = diff_manifests(baseline, current)
        assert diff["drifted"] is True
        assert diff["git_changed"] is True
