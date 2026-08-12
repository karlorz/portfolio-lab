"""Batch BI residual honesty: public index touch after partial write + incidents SSOT copy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path



def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_refresh_public_data_index_updates_sha_after_partial_write(tmp_path):
    from src.dashboard.public_data_index import (
        build_public_data_index,
        refresh_public_data_index_after_partial_write,
    )

    public = tmp_path / "public"
    public.mkdir()
    health = public / "health.json"
    signals = public / "signals.json"
    health.write_text(json.dumps({"status": "ok", "v": 1}), encoding="utf-8")
    signals.write_text(json.dumps({"health": {"status": "ok"}, "v": 1}), encoding="utf-8")

    index = build_public_data_index([health, signals], public_dir=public)
    index_path = public / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    def entry_sha(name: str) -> str | None:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        for e in payload.get("entries") or []:
            if e.get("path") == name or e.get("filename") == name:
                return e.get("sha256") or e.get("hash")
        return None

    old_health_sha = entry_sha("health.json")
    assert old_health_sha is not None

    # Partial patch changes bytes
    health.write_text(
        json.dumps({"status": "ok", "v": 2, "content_patched_at": "2026-07-21T05:00:00Z"}),
        encoding="utf-8",
    )
    live_sha = _sha256(health)
    assert live_sha != old_health_sha

    out = refresh_public_data_index_after_partial_write(
        public_dir=public,
        extra_paths=[health],
        reason="test_partial",
    )
    assert out is not None
    assert out.get("content_patch_source") == "index_refresh:test_partial"
    new_health_sha = entry_sha("health.json")
    assert new_health_sha == live_sha


def test_incident_write_summary_public_is_byte_copy_of_private(tmp_path, monkeypatch):
    """Public dual-write must copy private open set (SSOT), not invent another."""
    from src.monitor import incident_manager as im

    private = tmp_path / "private"
    public = tmp_path / "public"
    private.mkdir()
    public.mkdir()
    monkeypatch.setattr(im, "PUBLIC_DATA_DIR", public)
    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "biincidentsha",
    )

    mgr = im.IncidentManager(
        log_path=private / "incidents.jsonl",
        summary_path=private / "incidents.json",
    )
    # Seed a firing incident via open path
    mgr.record_alert(
        channel="signal_staleness",
        level="halt",
        message="convexity ownership stale",
        details={"symbols": ["convexity"]},
    )
    private_body = (private / "incidents.json").read_text(encoding="utf-8")
    public_body = (public / "incidents.json").read_text(encoding="utf-8")
    priv = json.loads(private_body)
    pub = json.loads(public_body)
    assert priv["open_count"] == pub["open_count"] == 1
    assert priv["incidents"][0]["channel"] == "signal_staleness"
    assert pub["incidents"][0]["channel"] == "signal_staleness"
    assert priv.get("open_set_ssot") == "private_summary_path"
    # Same open set ids
    assert priv["incidents"][0]["incident_id"] == pub["incidents"][0]["incident_id"]


def test_incident_write_summary_does_not_keep_divergent_public_open_set(
    tmp_path, monkeypatch
):
    """Stale public evaluator_error must be replaced by private signal_staleness SSOT."""
    from src.monitor import incident_manager as im

    private = tmp_path / "private"
    public = tmp_path / "public"
    private.mkdir()
    public.mkdir()
    monkeypatch.setattr(im, "PUBLIC_DATA_DIR", public)
    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "biclobbersha1",
    )

    # Poison public with wrong open channel
    (public / "incidents.json").write_text(
        json.dumps(
            {
                "schema_version": "incident-lifecycle/v1",
                "open_count": 1,
                "incidents": [
                    {
                        "incident_id": "stale-eval",
                        "channel": "evaluator_error",
                        "severity": "p0",
                        "state": "firing",
                        "message": "evaluator crashed",
                        "details": {},
                        "created_at": "2026-07-20T00:00:00+00:00",
                        "updated_at": "2026-07-20T00:00:00+00:00",
                        "alert_count": 1,
                    }
                ],
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )

    mgr = im.IncidentManager(
        log_path=private / "incidents.jsonl",
        summary_path=private / "incidents.json",
    )
    mgr.record_alert(
        channel="signal_staleness",
        level="halt",
        message="ownership stale",
        details={},
    )
    pub = json.loads((public / "incidents.json").read_text(encoding="utf-8"))
    assert pub["open_count"] == 1
    assert pub["incidents"][0]["channel"] == "signal_staleness"
    assert pub["incidents"][0]["incident_id"] != "stale-eval"


def test_refresh_index_helper_exported():
    from src.dashboard import public_data_index as pdi

    assert callable(pdi.refresh_public_data_index_after_partial_write)
