"""Batch BW: mirror live public/data → repo public/data (H22b)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.mirror_repo_public_data import (
    lag_report,
    mirror_repo_public_data,
    resolve_mirror_paths,
)


def test_resolve_rejects_parent_escape():
    try:
        resolve_mirror_paths("../etc/passwd", Path("/a"), Path("/b"))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_mirror_copies_new_and_skips_unchanged(tmp_path: Path):
    src = tmp_path / "live"
    dst = tmp_path / "repo"
    src.mkdir()
    (src / "signals.json").write_text(
        json.dumps({"generator_git_sha": "abc123", "x": 1}),
        encoding="utf-8",
    )
    (src / "health.json").write_text(
        json.dumps({"generator_git_sha": "abc123", "status": "ok"}),
        encoding="utf-8",
    )
    # nested
    (src / "attribution").mkdir()
    (src / "attribution" / "latest.json").write_text(
        json.dumps({"generator_git_sha": "abc123"}),
        encoding="utf-8",
    )

    report = mirror_repo_public_data(
        source_root=src,
        dest_root=dst,
        files=("signals.json", "health.json", "attribution/latest.json", "missing.json"),
        dry_run=False,
    )
    assert "signals.json" in report.copied
    assert "health.json" in report.copied
    assert "attribution/latest.json" in report.copied
    assert "missing.json" in report.skipped_missing
    assert json.loads((dst / "signals.json").read_text())["generator_git_sha"] == "abc123"

    # second pass unchanged
    report2 = mirror_repo_public_data(
        source_root=src,
        dest_root=dst,
        files=("signals.json",),
        dry_run=False,
    )
    assert report2.skipped_unchanged == ["signals.json"]
    assert report2.copied == []


def test_mirror_updates_stale_dest(tmp_path: Path):
    src = tmp_path / "live"
    dst = tmp_path / "repo"
    src.mkdir()
    dst.mkdir()
    (src / "dashboard.json").write_text(
        json.dumps({"generator_git_sha": "newsha", "v": 2}),
        encoding="utf-8",
    )
    (dst / "dashboard.json").write_text(
        json.dumps({"generator_git_sha": "oldsha", "v": 1}),
        encoding="utf-8",
    )
    report = mirror_repo_public_data(
        source_root=src,
        dest_root=dst,
        files=("dashboard.json",),
    )
    assert report.copied == ["dashboard.json"]
    body = json.loads((dst / "dashboard.json").read_text())
    assert body["generator_git_sha"] == "newsha"
    assert body["v"] == 2


def test_lag_report_detects_mismatch(tmp_path: Path):
    src = tmp_path / "live"
    dst = tmp_path / "repo"
    src.mkdir()
    dst.mkdir()
    (src / "stats.json").write_text(
        json.dumps({"generator_git_sha": "aaa"}), encoding="utf-8"
    )
    (dst / "stats.json").write_text(
        json.dumps({"generator_git_sha": "bbb"}), encoding="utf-8"
    )
    rows = lag_report(src, dst, files=("stats.json",))
    assert len(rows) == 1
    assert rows[0]["lagging"] is True
    assert rows[0]["source_sha"] == "aaa"
    assert rows[0]["dest_sha"] == "bbb"


def test_dry_run_does_not_write(tmp_path: Path):
    src = tmp_path / "live"
    dst = tmp_path / "repo"
    src.mkdir()
    (src / "analytics.json").write_text("{}", encoding="utf-8")
    report = mirror_repo_public_data(
        source_root=src,
        dest_root=dst,
        files=("analytics.json",),
        dry_run=True,
    )
    assert report.copied == ["analytics.json"]
    assert not (dst / "analytics.json").exists()
