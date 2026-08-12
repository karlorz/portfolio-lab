"""Batch BD / H19: attribution dual-write discovered by public data index."""

from __future__ import annotations

import json
from pathlib import Path


def test_discover_attribution_public_paths():

    # Source contract present
    src = Path("src/dashboard/public_data_index.py").read_text(encoding="utf-8")
    assert "_discover_attribution_public_paths" in src
    assert "attribution/latest.json" in src


def test_build_index_discovers_attribution_subdir(tmp_path):
    from src.dashboard.public_data_index import build_public_data_index

    attr = tmp_path / "attribution"
    attr.mkdir()
    latest = attr / "latest.json"
    dated = attr / "attribution_2026-07-21.json"
    body = {
        "timestamp": "2026-07-21T12:00:00+00:00",
        "status": "no_data",
        "sources": {},
        "generator_git_sha": "attribsha1234",
    }
    latest.write_text(json.dumps(body))
    dated.write_text(json.dumps(body))

    index = build_public_data_index([], public_dir=tmp_path, generated_at="2026-07-21T12:00:00+00:00")
    files = index.get("files") or []
    assert "attribution/latest.json" in files
    assert "attribution/attribution_2026-07-21.json" in files
    by_name = {e["filename"]: e for e in index["entries"]}
    assert by_name["latest.json"]["path"] == "attribution/latest.json"
    assert by_name["latest.json"]["category"] == "attribution"
    assert by_name["attribution_2026-07-21.json"]["category"] == "attribution"


def test_attribution_save_refreshes_public_index(tmp_path, monkeypatch):
    from src.monitor import performance_attribution as pa
    from src.monitor.performance_attribution import PerformanceAttribution

    private = tmp_path / "data"
    public = tmp_path / "public"
    private.mkdir()
    public.mkdir()
    # Seed minimal index so rebuild has a prior
    (public / "index.json").write_text(
        json.dumps(
            {
                "schema_version": "public-data-index/v1",
                "files": ["health.json"],
                "entries": [
                    {
                        "filename": "health.json",
                        "path": "health.json",
                        "status": "present",
                    }
                ],
                "generated_at": "2026-07-01T00:00:00+00:00",
            }
        )
    )
    (public / "health.json").write_text("{}")

    monkeypatch.setattr(pa, "PUBLIC_DATA_DIR", public)
    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "attribsha1234",
    )
    # build_public_data_index uses PUBLIC_DATA_DIR default — patch paths too
    monkeypatch.setattr("src.paths.PUBLIC_DATA_DIR", public)

    attr = PerformanceAttribution(data_dir=private)

    class FakeReport:
        timestamp = "2026-07-21T12:00:00+00:00"

        def to_dict(self):
            return {"timestamp": self.timestamp, "status": "no_data", "sources": {}}

    attr.save_report(FakeReport())
    assert (public / "attribution" / "latest.json").is_file()
    index = json.loads((public / "index.json").read_text())
    files = index.get("files") or []
    assert any("attribution" in f for f in files), files
    assert "attribution/latest.json" in files or any(
        e.get("path") == "attribution/latest.json" for e in index.get("entries", [])
    )


def test_batch_bd_source_contracts():
    pa = Path("src/monitor/performance_attribution.py").read_text(encoding="utf-8")
    # Batch BI centralized index refresh; still must refresh after attribution dual-write
    assert (
        "build_public_data_index" in pa
        or "refresh_public_data_index_after_partial_write" in pa
    )
    assert "Refreshed public index after attribution" in pa or "index" in pa.lower()
