"""Batch AK residual honesty: stats.json generator_git_sha provenance stamp."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_stats_json_stamps_generator_git_sha(tmp_path, monkeypatch):
    """stats.json carries generator_git_sha for code-vs-artifact lag detection."""
    from src.dashboard.generator import DashboardGenerator

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    public = tmp_path / "public"
    public.mkdir()

    monkeypatch.setattr("src.dashboard.generator.DATA_DIR", data_dir)
    monkeypatch.setattr("src.dashboard.generator.PUBLIC_DIR", public)

    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen.conn = MagicMock()
    cur = gen.conn.cursor.return_value
    cur.fetchall.return_value = []
    cur.execute = MagicMock()

    with patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="deadbeef12",
    ):
        out = DashboardGenerator.generate_stats_json(gen)

    payload = json.loads(Path(out).read_text())
    assert payload["generator_git_sha"] == "deadbeef12"
    assert payload["generator_git_sha_status"] == "full_generate"
    assert "generated_at" in payload


def test_stats_json_omits_sha_when_unavailable(tmp_path, monkeypatch):
    """When git SHA probe fails, do not invent a stamp."""
    from src.dashboard.generator import DashboardGenerator

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    public = tmp_path / "public"
    public.mkdir()

    monkeypatch.setattr("src.dashboard.generator.DATA_DIR", data_dir)
    monkeypatch.setattr("src.dashboard.generator.PUBLIC_DIR", public)

    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen.conn = MagicMock()
    cur = gen.conn.cursor.return_value
    cur.fetchall.return_value = []
    cur.execute = MagicMock()

    with patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value=None,
    ):
        out = DashboardGenerator.generate_stats_json(gen)

    payload = json.loads(Path(out).read_text())
    assert "generator_git_sha" not in payload or payload.get("generator_git_sha") is None
