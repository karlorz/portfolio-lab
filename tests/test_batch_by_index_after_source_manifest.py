"""Batch BY: index.json refresh after source_manifest market-data writes."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.dashboard.public_data_index import (
    _PARTIAL_INDEX_CORE_FILES,
    refresh_public_data_index_after_partial_write,
)


def test_partial_core_includes_market_data_artifacts():
    for name in (
        "source_manifest.json",
        "data_quality.json",
        "prices.json",
        "prices_compact.json",
        "yields.json",
    ):
        assert name in _PARTIAL_INDEX_CORE_FILES


def test_refresh_after_source_manifest_updates_generated_at(tmp_path: Path):
    """Index generated_at must advance past older source_manifest stamp."""
    # Prior index older than source_manifest (stale_index scenario)
    old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    new = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()

    (tmp_path / "source_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "market-data-source-manifest/v1",
                "generated_at": new,
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "prices.json").write_text("{}", encoding="utf-8")
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "schema_version": "public-data-index/v1",
                "generated_at": old,
                "files": ["prices.json"],
                "entries": [
                    {
                        "filename": "prices.json",
                        "path": "prices.json",
                        "status": "present",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    out = refresh_public_data_index_after_partial_write(
        public_dir=tmp_path,
        reason="source_manifest",
    )
    assert out is not None
    assert out["content_patch_source"] == "index_refresh:source_manifest"
    # Fresh stamp
    gen = datetime.fromisoformat(out["generated_at"].replace("Z", "+00:00"))
    sm = datetime.fromisoformat(new.replace("Z", "+00:00"))
    assert gen >= sm.replace(tzinfo=timezone.utc) or gen.timestamp() >= sm.timestamp() - 1

    on_disk = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert on_disk["generated_at"] == out["generated_at"]


def test_fetch_data_wires_index_refresh():
    src = Path("scripts/fetch-data.ts").read_text(encoding="utf-8")
    assert "refresh_public_data_index.py" in src
    assert "source_manifest" in src
    assert "Batch BY" in src


def test_refresh_cli_script_exists():
    script = Path("scripts/refresh_public_data_index.py")
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "refresh_public_data_index_after_partial_write" in text
