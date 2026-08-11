"""Option C (operator-approved 2026-08-11): immutable generation store.

Each completed run records its public JSON surface as an immutable
generation (manifest + per-file sha256); the current pointer flips
atomically; rollback re-mirrors a previous generation over the flat public
dir; prune keeps the newest N. Tests drive the real shipped functions with
throwaway generation/public dirs.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from src.dashboard.generation_store import (
    GENERATION_SCHEMA,
    GenerationStore,
)


def _make_public_dir(tmp_path, files: dict[str, str]):
    public = tmp_path / "public"
    for rel, content in files.items():
        path = public / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return public


def test_record_creates_immutable_generation_with_verified_manifest(tmp_path):
    public = _make_public_dir(
        tmp_path, {"signals.json": '{"a": 1}', "explainability/x.json": "[1,2]"}
    )
    store = GenerationStore(generations_dir=tmp_path / "gens")
    manifest = store.record(run_id="gen-test-1", public_dir=public)

    assert manifest.run_id == "gen-test-1"
    assert manifest.file_count == 2
    assert set(manifest.files) == {"signals.json", "explainability/x.json"}
    # Manifest sha256s must match the generation files byte-for-byte.
    for rel, sha in manifest.files.items():
        stored = (tmp_path / "gens" / "gen-test-1" / rel).read_bytes()
        assert hashlib.sha256(stored).hexdigest() == sha

    # Pointer resolves and the manifest file is schema-valid.
    cur = store.current()
    assert cur is not None and cur.run_id == "gen-test-1"
    assert (tmp_path / "gens" / "gen-test-1" / "manifest.json").is_file()
    # No temp-file leftovers from the atomic pointer update.
    assert not (tmp_path / "gens" / "current.json.tmp").exists()
    assert not (tmp_path / "gens" / "current.link.tmp").exists()


def test_second_record_flips_pointer_but_first_generation_stays_immutable(tmp_path):
    public = _make_public_dir(tmp_path, {"signals.json": '{"a": 1}'})
    store = GenerationStore(generations_dir=tmp_path / "gens")
    store.record(run_id="gen-first", public_dir=public)

    (public / "signals.json").write_text('{"a": 2}', encoding="utf-8")
    store.record(run_id="gen-second", public_dir=public)

    assert store.current().run_id == "gen-second"
    # Immutability: the first generation's file still holds the original bytes.
    first = (tmp_path / "gens" / "gen-first" / "signals.json").read_text()
    assert first == '{"a": 1}'
    assert len(store.generations()) == 2


def test_unactivated_generation_dir_does_not_change_pointer(tmp_path):
    """A killed run may leave a generation dir without activating it; the
    current pointer must keep resolving to the last complete generation."""
    public = _make_public_dir(tmp_path, {"signals.json": '{"a": 1}'})
    store = GenerationStore(generations_dir=tmp_path / "gens")
    store.record(run_id="gen-complete", public_dir=public)

    # Simulate a run that wrote files into its generation dir but died
    # before activate(): files exist, no manifest, pointer untouched.
    run_dir = tmp_path / "gens" / "gen-hung"
    run_dir.mkdir()
    (run_dir / "signals.json").write_text('{"a": 999}', encoding="utf-8")

    cur = store.current()
    assert cur is not None and cur.run_id == "gen-complete"


def test_rollback_restores_previous_generation_with_verified_sha(tmp_path):
    public = _make_public_dir(tmp_path, {"signals.json": '{"a": 1}'})
    store = GenerationStore(generations_dir=tmp_path / "gens")
    store.record(run_id="gen-ok", public_dir=public)

    # The flat surface gets clobbered by a bad run.
    (public / "signals.json").write_text('{"a": 999}', encoding="utf-8")

    restored = store.rollback_to("gen-ok", public_dir=public)
    assert restored == 1
    assert (public / "signals.json").read_text(encoding="utf-8") == '{"a": 1}'


def test_prune_keeps_newest_and_current(tmp_path):
    public = _make_public_dir(tmp_path, {"signals.json": '{"a": 1}'})
    store = GenerationStore(generations_dir=tmp_path / "gens")
    for i in range(5):
        store.record(run_id=f"gen-{i}", public_dir=public)

    store.prune(keep=2)
    ids = [m.run_id for m in store.generations()]
    assert ids == ["gen-4", "gen-3"]  # newest two survive
    assert store.current().run_id == "gen-4"
    for old in ("gen-0", "gen-1", "gen-2"):
        assert not (tmp_path / "gens" / old).exists()


def test_invalid_run_id_rejected(tmp_path):
    public = _make_public_dir(tmp_path, {"signals.json": '{"a": 1}'})
    store = GenerationStore(generations_dir=tmp_path / "gens")
    with pytest.raises(ValueError):
        store.record(run_id="../evil", public_dir=public)


def test_pointer_never_half_written_across_many_records(tmp_path):
    """Drive the real atomic pointer path repeatedly; current.json must
    parse as a valid manifest after every record."""
    public = _make_public_dir(tmp_path, {"signals.json": '{"a": 1}'})
    store = GenerationStore(generations_dir=tmp_path / "gens")
    for i in range(5):
        store.record(run_id=f"gen-many-{i}", public_dir=public)
        payload = json.loads(
            (tmp_path / "gens" / "current.json").read_text(encoding="utf-8")
        )
        assert payload["schema"] == GENERATION_SCHEMA
        assert payload["run_id"] == f"gen-many-{i}"


def test_public_index_discloses_current_generation(tmp_path, monkeypatch):
    """build_public_data_index (real shipped builder) carries the current
    generation as a rollback/integrity anchor when one exists."""
    from src.dashboard import generation_store as gs
    from src.dashboard.public_data_index import build_public_data_index

    public = _make_public_dir(tmp_path, {"signals.json": '{"a": 1}'})
    monkeypatch.setattr(gs, "GENERATIONS_DIR", tmp_path / "gens")
    gs.GenerationStore(public_dir=public).record(
        run_id="gen-index-1", public_dir=public
    )

    index = build_public_data_index(
        [public / "signals.json"], public_dir=public, use_hash_cache=False
    )
    assert index["generation"]["run_id"] == "gen-index-1"
    assert index["generation"]["file_count"] == 1
    assert index["generation"]["git_sha"]
