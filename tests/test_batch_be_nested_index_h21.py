"""Batch BE / H21: nested explainability (+ attribution) public index discovery."""

from __future__ import annotations

import json
from pathlib import Path


def test_discover_explainability_public_paths_source():
    src = Path("src/dashboard/public_data_index.py").read_text(encoding="utf-8")
    assert "_discover_explainability_public_paths" in src
    assert "explainability/explainability_latest.json" in src or "explainability_latest.json" in src


def test_build_index_discovers_explainability_subdir(tmp_path):
    from src.dashboard.public_data_index import build_public_data_index

    expl = tmp_path / "explainability"
    expl.mkdir()
    latest = expl / "explainability_latest.json"
    latest.write_text(
        json.dumps(
            {
                "generated_at": "2026-07-21T12:00:00+00:00",
                "status": "unavailable",
                "reason": "test",
            }
        )
    )

    index = build_public_data_index(
        [], public_dir=tmp_path, generated_at="2026-07-21T12:00:00+00:00"
    )
    files = index.get("files") or []
    assert "explainability/explainability_latest.json" in files
    by_name = {e["filename"]: e for e in index["entries"]}
    assert by_name["explainability_latest.json"]["path"] == (
        "explainability/explainability_latest.json"
    )
    assert by_name["explainability_latest.json"]["category"] == "explainability"


def test_build_index_discovers_attribution_and_explainability_together(tmp_path):
    from src.dashboard.public_data_index import build_public_data_index

    (tmp_path / "attribution").mkdir()
    (tmp_path / "explainability").mkdir()
    (tmp_path / "attribution" / "latest.json").write_text(
        json.dumps({"timestamp": "2026-07-21T00:00:00+00:00", "status": "no_data", "sources": {}})
    )
    (tmp_path / "explainability" / "explainability_latest.json").write_text(
        json.dumps({"generated_at": "2026-07-21T00:00:00+00:00", "status": "unavailable"})
    )

    index = build_public_data_index([], public_dir=tmp_path)
    files = set(index.get("files") or [])
    assert "attribution/latest.json" in files
    assert "explainability/explainability_latest.json" in files


def test_batch_be_source_contracts():
    src = Path("src/dashboard/public_data_index.py").read_text(encoding="utf-8")
    assert "_discover_attribution_public_paths" in src
    assert "_discover_explainability_public_paths" in src
    # Both wired into build_public_data_index
    assert src.count("_discover_explainability_public_paths") >= 2
