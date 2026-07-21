"""Batch AT residual honesty: unified dual-write provenance + dual_write_ok canary."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


def _rich_dashboard() -> dict:
    """Minimal non-hollow unified payload (section_score >= 1)."""
    return {
        "dashboard_version": "v6.08",
        "generated_at": "2026-07-21T00:00:00+00:00",
        "generator_git_sha": "unifysha1234",
        "generator_git_sha_status": "full_generate",
        "health": {"available": True, "status": "ok"},
        "portfolio": {"available": False},
        "risk": {"available": False},
        "regime": {"available": False},
        "cron": {"available": False},
        "attribution": {"available": False},
        "risk_history": {"available": False},
        "tca": {"available": False},
        "overlays": {"available": False},
        "adaptive_weights": {"available": False},
    }


def test_unified_save_dual_write_stamps_provenance(tmp_path, monkeypatch):
    from src.monitor import unified_dashboard as ud

    private = tmp_path / "data"
    public = tmp_path / "public"
    private.mkdir()
    public.mkdir()
    monkeypatch.setattr(ud, "DATA_DIR", private)
    monkeypatch.setattr(ud, "PUBLIC_DATA_DIR", public)

    dash = _rich_dashboard()
    written = ud._save_unified_dashboard(dash)
    assert any(p.name == "unified_dashboard.json" for p in written)
    private_body = json.loads((private / "unified_dashboard.json").read_text())
    public_body = json.loads((public / "unified_dashboard.json").read_text())
    for body in (private_body, public_body):
        pc = body["provenance_completeness"]
        assert pc["dual_write_attempted"] is True
        assert pc["dual_write_ok"] is True
        # After dual-write, content-hash identity may set paths_identical True
        # for lag purposes (Batch content-hash / CJ post-sync). Path strings still differ.
        assert pc["private_path"] != pc["public_path"]
        assert pc.get("dual_write_lag_stale") is False
        assert pc["section_score"] >= 1


def test_unified_save_hollow_skips_public_and_records(tmp_path, monkeypatch):
    from src.monitor import unified_dashboard as ud

    private = tmp_path / "data"
    public = tmp_path / "public"
    private.mkdir()
    public.mkdir()
    monkeypatch.setattr(ud, "DATA_DIR", private)
    monkeypatch.setattr(ud, "PUBLIC_DATA_DIR", public)

    hollow = {
        "generated_at": "2026-07-21T00:00:00+00:00",
        "health": {"available": False},
        "portfolio": {"available": False},
        "risk": {"available": False},
        "regime": {"available": False},
        "cron": {"available": False},
        "attribution": {"available": False},
        "risk_history": {"available": False},
    }
    written = ud._save_unified_dashboard(hollow)
    assert len(written) == 1
    assert not (public / "unified_dashboard.json").exists()
    body = json.loads((private / "unified_dashboard.json").read_text())
    pc = body["provenance_completeness"]
    assert pc["dual_write_attempted"] is False
    assert pc["dual_write_ok"] is False
    assert "hollow" in (pc.get("note") or "")


def test_dual_write_ok_false_canary_warns(tmp_path):
    from scripts.check_public_data_consistency import (
        _check_dual_write_provenance_completeness,
    )

    public = tmp_path / "public" / "data"
    public.mkdir(parents=True)
    (public / "unified_dashboard.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-07-21T00:00:00+00:00",
                "provenance_completeness": {
                    "dual_write_attempted": True,
                    "dual_write_ok": False,
                    "note": "simulated_failure",
                },
            }
        ),
        encoding="utf-8",
    )
    errors: list[str] = []
    warnings: list[str] = []
    _check_dual_write_provenance_completeness(public, errors, warnings)
    assert errors == []
    assert any("dual_write_ok=false" in w for w in warnings)


def test_batch_at_source_contracts():
    unified = Path("src/monitor/unified_dashboard.py").read_text(encoding="utf-8")
    assert "_attach_dual_write_provenance" in unified
    assert "section_score" in unified

    cpc = Path("scripts/check_public_data_consistency.py").read_text(encoding="utf-8")
    assert "DUAL_WRITE_PROVENANCE_FILES" in cpc
    assert "_check_dual_write_provenance_completeness" in cpc
