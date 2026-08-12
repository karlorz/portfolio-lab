"""Committed mutable-data generation contract (Task 5)."""

from __future__ import annotations

import hashlib
import json

import pytest

_CONTRACT_FILES = (
    "prices.json",
    "prices_compact.json",
    "historical.json",
    "yields.json",
    "data_quality.json",
    "source_manifest.json",
)


@pytest.fixture(autouse=True)
def _ephemeral_public(tmp_path, monkeypatch):
    """Keep every test fully inside the temp tree."""
    monkeypatch.setenv("PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC", "1")


def test_health_generation_fields_stamped(tmp_path, monkeypatch):
    """Health-owned outputs carry producer_run_id, generation_id, and sha."""
    from src.monitor import health_check

    out_path = tmp_path / "health.json"
    monkeypatch.setenv("TASKER_RUN_ID", "run-20260809123045-ab12cd34")
    payload = {"status": "critical", "checks": {"kill_switch": {"enabled": True}}}
    health_check.write_health_generation(payload, path=out_path, producer_sha="abc1234")

    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["producer_run_id"] == "run-20260809123045-ab12cd34"
    assert on_disk["generation_id"]
    assert on_disk["producer_git_sha"] == "abc1234"


def test_private_health_write_is_atomic_under_interruption(tmp_path, monkeypatch):
    """An injected failure mid-write leaves the prior committed file intact."""
    from src.monitor import health_check
    from src.monitor import signal_authority

    out_path = tmp_path / "health.json"
    prior = {"status": "ok", "generation_id": "gen-old"}
    out_path.write_text(json.dumps(prior), encoding="utf-8")

    def _boom_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(signal_authority, "_atomic_write_text", _boom_write)
    with pytest.raises(OSError):
        health_check.write_health_generation(
            {"status": "critical"}, path=out_path, producer_sha="x"
        )
    # Prior committed file untouched; no partial bytes.
    assert json.loads(out_path.read_text(encoding="utf-8")) == prior


def test_index_replaced_last_and_matches_served_bytes(tmp_path, monkeypatch):
    """Content first, index last; every core entry hash/size matches bytes."""
    from src.monitor import health_check
    from src.dashboard import public_data_index

    public_dir = tmp_path / "public"
    public_dir.mkdir()
    for name in _CONTRACT_FILES:
        (public_dir / name).write_text(f'{{"file": "{name}"}}', encoding="utf-8")
    (public_dir / "health.json").write_text('{"status": "critical"}', encoding="utf-8")

    paths = [public_dir / name for name in _CONTRACT_FILES]
    paths.append(public_dir / "health.json")
    entries = public_data_index.build_public_data_index(paths, public_dir=public_dir)
    index_path = public_dir / "index.json"
    health_check.commit_public_index(entries, index_path=index_path, generation_id="gen-1")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["generation_id"] == "gen-1"
    core = [e for e in index["entries"] if e.get("category") == "market_data"]
    assert len(core) >= 1
    for entry in core:
        path = public_dir / entry["filename"]
        bytes_ = path.read_bytes()
        assert entry["sha256"] == hashlib.sha256(bytes_).hexdigest()
        assert entry["size_bytes"] == len(bytes_)


def test_failed_generation_leaves_prior_index_committed(tmp_path, monkeypatch):
    """Failure before index replacement keeps the previous index untouched."""
    from src.monitor import health_check
    from src.monitor import signal_authority

    index_path = tmp_path / "index.json"
    prior = {"generation_id": "gen-old", "entries": []}
    index_path.write_text(json.dumps(prior), encoding="utf-8")

    def _boom_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(signal_authority, "_atomic_write_text", _boom_write)
    with pytest.raises(OSError):
        health_check.commit_public_index(
            {
                "schema_version": "public-data-index/v1",
                "entries": [{"filename": "prices.json", "sha256": "x", "size_bytes": 1}],
            },
            index_path=index_path,
            generation_id="gen-new",
        )
    assert json.loads(index_path.read_text(encoding="utf-8")) == prior
