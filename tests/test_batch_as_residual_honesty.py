"""Batch AS residual honesty: dual-write provenance_completeness blocks."""

from __future__ import annotations

import json
from pathlib import Path


def test_attach_dual_write_provenance_block_shape():
    from src.dashboard.generator import _attach_dual_write_provenance

    payload = {"generated_at": "2026-07-21T00:00:00+00:00", "generator_git_sha": "abc123"}
    out = _attach_dual_write_provenance(
        payload,
        private_path="/tmp/private.json",
        public_path="/tmp/public.json",
        dual_write_attempted=True,
        dual_write_ok=True,
        paths_identical=False,
    )
    block = out["provenance_completeness"]
    assert block["generator_git_sha_present"] is True
    assert block["dual_write_attempted"] is True
    assert block["dual_write_ok"] is True
    assert block["paths_identical"] is False
    assert "dual_write_lag_seconds" in block
    assert block["dual_write_lag_unit"] == "seconds_public_mtime_minus_private"
    assert "advisory" in block["disclosure"].lower() or "split-brain" in block["disclosure"].lower()


def test_incident_write_summary_includes_provenance_completeness(tmp_path, monkeypatch):
    from src.monitor.incident_manager import IncidentManager

    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "incsha123456",
    )
    # Force distinct public path under tmp so dual-write runs
    public = tmp_path / "public"
    public.mkdir()
    monkeypatch.setattr("src.monitor.incident_manager.PUBLIC_DATA_DIR", public)

    mgr = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
    )
    summary = mgr.write_summary()
    assert "provenance_completeness" in summary
    pc = summary["provenance_completeness"]
    assert pc["generator_git_sha_present"] is True
    assert pc["dual_write_attempted"] is True
    assert pc["dual_write_ok"] is True
    # Path strings differ; content-hash post-sync may mark paths_identical for lag
    assert "public" in str(pc.get("public_path") or "")
    assert pc.get("dual_write_lag_stale") is False
    # Public dual-write materialised
    public_body = json.loads((public / "incidents.json").read_text(encoding="utf-8"))
    assert public_body["provenance_completeness"]["dual_write_ok"] is True


def test_incident_paths_identical_skips_dual_attempt(tmp_path, monkeypatch):
    from src.monitor.incident_manager import IncidentManager

    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "samesha12345",
    )
    # PUBLIC_DATA_DIR == summary parent so paths resolve identical for incidents.json
    monkeypatch.setattr("src.monitor.incident_manager.PUBLIC_DATA_DIR", tmp_path)

    mgr = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
    )
    summary = mgr.write_summary()
    pc = summary["provenance_completeness"]
    assert pc["paths_identical"] is True
    assert pc["dual_write_attempted"] is False
    assert pc["dual_write_ok"] is True


def test_health_ops_publish_attaches_provenance(tmp_path, monkeypatch):
    from src.monitor import health_check as hc

    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(hc, "HEALTH_PATH", tmp_path / "health.json")
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", tmp_path / "public")
    (tmp_path / "public").mkdir(parents=True)

    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "healthsha1234",
    )

    report = {
        "status": "ok",
        "timestamp": "2026-07-21T00:00:00+00:00",
        "checks": {},
        "service": "portfolio-lab",
        "scope": "operational_readiness",
        "generator_git_sha": "healthsha1234",
        "generator_git_sha_status": "full_generate",
    }
    # Avoid signals kill refresh side effects
    monkeypatch.setattr(hc, "refresh_signals_health_kill_fields", lambda report: None)
    hc.publish_ops_health_surfaces(report)
    ops = json.loads((tmp_path / "public" / "health_ops.json").read_text(encoding="utf-8"))
    assert "provenance_completeness" in ops
    assert ops["provenance_completeness"]["dual_write_attempted"] is True
    assert ops["provenance_completeness"]["dual_write_ok"] is True


def test_batch_as_source_contracts():
    provenance = Path("src/dashboard/provenance.py").read_text(encoding="utf-8")
    assert "def _attach_dual_write_provenance" in provenance
    assert "provenance_completeness" in provenance

    incidents = Path("src/monitor/incident_manager.py").read_text(encoding="utf-8")
    assert "_attach_dual_write_provenance" in incidents

    health = Path("src/monitor/health_check.py").read_text(encoding="utf-8")
    assert "_attach_dual_write_provenance" in health


def test_attach_dual_write_content_hash_clears_lag_when_payloads_match(tmp_path):
    """Trailing-newline-only pair → content-hash identical, lag not sticky."""
    from src.dashboard.generator import _attach_dual_write_provenance

    priv = tmp_path / "private.json"
    pub = tmp_path / "public.json"
    body = b'{"status": "ok", "value": 1}'
    priv.write_bytes(body + b"\n")
    pub.write_bytes(body)  # no trailing newline — path-different trees, same content
    # Stagger mtimes so naive lag would look stale
    import os
    import time
    now = time.time()
    os.utime(priv, (now, now))
    os.utime(pub, (now - 600, now - 600))  # public 10 min older

    out = _attach_dual_write_provenance(
        {"generator_git_sha": "abc"},
        private_path=priv,
        public_path=pub,
        dual_write_attempted=True,
        dual_write_ok=True,
        paths_identical=False,  # resolve differs
        lag_threshold_seconds=120.0,
    )
    block = out["provenance_completeness"]
    assert block["content_hash_identical"] is True
    # Path identity stays false when resolves differ; content hash clears lag
    assert block["paths_identical"] is False
    assert block["dual_write_lag_stale"] is False
    assert block["dual_write_lag_seconds"] in (None, 0.0)
    assert block["private_content_hash"] == block["public_content_hash"]
