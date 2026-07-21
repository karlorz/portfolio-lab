"""Batch AU residual honesty: dual_write_lag_seconds mtime forensics."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def test_dual_write_lag_seconds_public_behind_private(tmp_path):
    from src.dashboard.generator import _attach_dual_write_provenance

    private = tmp_path / "private.json"
    public = tmp_path / "public.json"
    # Different content so content-hash identity does not clear lag
    private.write_text('{"v":1}')
    public.write_text('{"v":0,"stale":true}')
    # Make public older than private by ~5s
    now = time.time()
    os.utime(public, (now - 10, now - 10))
    os.utime(private, (now, now))

    out = _attach_dual_write_provenance(
        {"generator_git_sha": "abc"},
        private_path=private,
        public_path=public,
        dual_write_attempted=True,
        dual_write_ok=True,
        paths_identical=False,
        lag_threshold_seconds=5.0,
    )
    pc = out["provenance_completeness"]
    assert pc["dual_write_lag_seconds"] is not None
    assert pc["dual_write_lag_seconds"] < -5.0
    assert pc["dual_write_lag_stale"] is True
    assert pc["content_hash_identical"] is False
    assert pc["private_mtime"] is not None
    assert pc["public_mtime"] is not None
    assert pc["public_mtime"] < pc["private_mtime"]


def test_dual_write_lag_not_stale_when_in_sync(tmp_path):
    from src.dashboard.generator import _attach_dual_write_provenance

    private = tmp_path / "private.json"
    public = tmp_path / "public.json"
    private.write_text("{}")
    public.write_text("{}")
    now = time.time()
    os.utime(private, (now, now))
    os.utime(public, (now, now))

    out = _attach_dual_write_provenance(
        {"generator_git_sha": "abc"},
        private_path=private,
        public_path=public,
        dual_write_attempted=True,
        dual_write_ok=True,
        paths_identical=False,
        lag_threshold_seconds=120.0,
    )
    pc = out["provenance_completeness"]
    assert abs(pc["dual_write_lag_seconds"]) < 1.0
    assert pc["dual_write_lag_stale"] is False


def test_dual_write_lag_null_when_paths_identical(tmp_path):
    from src.dashboard.generator import _attach_dual_write_provenance

    path = tmp_path / "same.json"
    path.write_text("{}")
    out = _attach_dual_write_provenance(
        {},
        private_path=path,
        public_path=path,
        dual_write_attempted=False,
        dual_write_ok=True,
        paths_identical=True,
    )
    pc = out["provenance_completeness"]
    # Same path / content-hash identical → lag not stale; lag seconds may be
    # 0.0 (resolved identical) rather than null.
    assert pc["dual_write_lag_stale"] is False
    assert pc["paths_identical"] is True
    assert pc["dual_write_lag_seconds"] in (None, 0.0)


def test_canary_warns_on_dual_write_lag_stale(tmp_path):
    from scripts.check_public_data_consistency import (
        _check_dual_write_provenance_completeness,
    )

    public = tmp_path / "public" / "data"
    public.mkdir(parents=True)
    (public / "incidents.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-07-21T00:00:00+00:00",
                "provenance_completeness": {
                    "dual_write_attempted": True,
                    "dual_write_ok": True,
                    "dual_write_lag_stale": True,
                    "dual_write_lag_seconds": -300.0,
                    "dual_write_lag_threshold_seconds": 120.0,
                },
            }
        ),
        encoding="utf-8",
    )
    errors: list[str] = []
    warnings: list[str] = []
    _check_dual_write_provenance_completeness(public, errors, warnings)
    assert errors == []
    assert any("dual_write_lag_stale" in w for w in warnings)


def test_batch_au_source_contracts():
    gen = Path("src/dashboard/generator.py").read_text(encoding="utf-8")
    assert "dual_write_lag_seconds" in gen
    assert "dual_write_lag_stale" in gen

    cpc = Path("scripts/check_public_data_consistency.py").read_text(encoding="utf-8")
    assert "dual_write_lag_stale" in cpc
