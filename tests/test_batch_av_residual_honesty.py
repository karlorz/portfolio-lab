"""Batch AV residual honesty: rebalance_health + garch dual-write provenance."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_rebalance_health_main_dual_write_provenance(tmp_path, monkeypatch):
    from src.monitor import rebalance_health as rh

    private = tmp_path / "data"
    public = tmp_path / "public"
    private.mkdir()
    public.mkdir()

    monkeypatch.setattr(rh, "DATA_DIR", private)
    monkeypatch.setattr(rh, "PUBLIC_DATA_DIR", public)
    monkeypatch.setattr(rh, "OUTPUT_PATH", private / "rebalance_health.json")
    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "rebalsha12345",
    )

    payload = {
        "total_executions": 0,
        "next_rebalance": {"date": "2026-08-01", "days_until": 10},
        "schedule_compliance": {"compliance_pct": 100},
        "generated": "2026-07-21T00:00:00+00:00",
    }
    monkeypatch.setattr(rh, "generate", lambda: dict(payload))

    rh.main()

    priv = json.loads((private / "rebalance_health.json").read_text())
    pub = json.loads((public / "rebalance_health.json").read_text())
    for body in (priv, pub):
        assert body.get("generator_git_sha") == "rebalsha12345"
        pc = body["provenance_completeness"]
        assert pc["dual_write_attempted"] is True
        assert pc["dual_write_ok"] is True
        assert pc["paths_identical"] is False


def test_garch_public_payload_source_has_dual_write_provenance():
    src = Path("scripts/compute_garch_risk.py").read_text(encoding="utf-8")
    assert "_attach_dual_write_provenance" in src
    assert "garch_cvar.json" in src
    assert "private_health_report" in src


def test_dual_write_canary_includes_rebalance_and_garch():
    from scripts.check_public_data_consistency import DUAL_WRITE_PROVENANCE_FILES

    assert "rebalance_health.json" in DUAL_WRITE_PROVENANCE_FILES
    assert "garch_cvar.json" in DUAL_WRITE_PROVENANCE_FILES


def test_batch_av_source_contracts():
    rebal = Path("src/monitor/rebalance_health.py").read_text(encoding="utf-8")
    assert "_attach_dual_write_provenance" in rebal
