"""Batch CJ: post-sync dual-write provenance clears sticky lag after public write."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def test_finalize_after_sync_clears_sticky_prewrite_lag(tmp_path):
    from src.dashboard.generator import (
        _attach_dual_write_provenance,
        finalize_dual_write_provenance_after_sync,
    )

    private = tmp_path / "private" / "unified_dashboard.json"
    public = tmp_path / "public" / "unified_dashboard.json"
    private.parent.mkdir(parents=True)
    public.parent.mkdir(parents=True)

    # Pre-existing public is old (simulates prior cycle)
    public.write_text(json.dumps({"old": True}) + "\n", encoding="utf-8")
    now = time.time()
    os.utime(public, (now - 600, now - 600))

    payload = {"dashboard_version": "test", "generated_at": "2026-07-21T20:00:00+00:00"}
    private.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.utime(private, (now, now))

    # Pre-write stamp freezes sticky lag (the bug pattern)
    pre = _attach_dual_write_provenance(
        payload,
        private_path=private,
        public_path=public,
        dual_write_attempted=True,
        dual_write_ok=True,
        paths_identical=False,
        lag_threshold_seconds=120.0,
    )
    assert pre["provenance_completeness"]["dual_write_lag_stale"] is True

    # Successful dual-write: public becomes content-equal to private
    body = json.dumps(pre, indent=2) + "\n"
    private.write_text(body, encoding="utf-8")
    public.write_text(body, encoding="utf-8")
    # Align mtimes as a healthy dual-write would after replace
    t = time.time()
    os.utime(private, (t, t))
    os.utime(public, (t, t))

    final = finalize_dual_write_provenance_after_sync(
        pre,
        private_path=private,
        public_path=public,
        dual_write_ok=True,
        lag_threshold_seconds=120.0,
    )
    pc = final["provenance_completeness"]
    assert pc["dual_write_lag_stale"] is False
    assert pc["content_hash_identical"] is True
    # Disk rewritten
    disk = json.loads(private.read_text(encoding="utf-8"))
    assert disk["provenance_completeness"]["dual_write_lag_stale"] is False
    pub_disk = json.loads(public.read_text(encoding="utf-8"))
    assert pub_disk["provenance_completeness"]["dual_write_lag_stale"] is False


def test_unified_save_post_sync_not_stale(tmp_path, monkeypatch):
    """End-to-end: _save_unified_dashboard clears lag_stale after dual-write."""
    from src.monitor import unified_dashboard as ud

    private = tmp_path / "data" / "unified_dashboard.json"
    public = tmp_path / "www" / "unified_dashboard.json"
    private.parent.mkdir(parents=True)
    public.parent.mkdir(parents=True)

    # Stale public pre-exists
    public.write_text(json.dumps({"dashboard_version": "old"}) + "\n", encoding="utf-8")
    now = time.time()
    os.utime(public, (now - 900, now - 900))

    monkeypatch.setattr(ud, "DATA_DIR", private.parent)
    monkeypatch.setattr(ud, "PUBLIC_DATA_DIR", public.parent)

    # Sections need available=True for section_score dual-write gate
    dashboard = {
        "dashboard_version": "v1",
        "generated_at": "2026-07-21T20:10:00+00:00",
        "health": {"available": True, "status": "ok"},
        "portfolio": {"available": True, "daily_return": 0.01},
        "risk": {"available": True},
        "regime": {"available": True},
        "attribution": {"available": True},
        "cron": {"available": True},
    }

    written = ud._save_unified_dashboard(dashboard)
    assert private.exists()
    assert public.exists()
    assert any(Path(p) == public or Path(p).name == "unified_dashboard.json" for p in written)
    body = json.loads(private.read_text(encoding="utf-8"))
    pc = body.get("provenance_completeness") or {}
    assert pc.get("dual_write_lag_stale") is False
    assert pc.get("dual_write_ok") is True


def test_batch_cj_source_contracts():
    gen = Path("src/dashboard/generator.py").read_text(encoding="utf-8")
    assert "finalize_dual_write_provenance_after_sync" in gen
    assert "post_sync" in gen

    unified = Path("src/monitor/unified_dashboard.py").read_text(encoding="utf-8")
    assert "finalize_dual_write_provenance_after_sync" in unified

    rebal = Path("src/monitor/rebalance_health.py").read_text(encoding="utf-8")
    assert "finalize_dual_write_provenance_after_sync" in rebal
