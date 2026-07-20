"""Batch AA residual honesty: public dual-write contracts."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_garch_risk_script_writes_public_path():
    """compute_garch_risk dual-writes PUBLIC_DATA_DIR/garch_cvar.json."""
    src = Path("scripts/compute_garch_risk.py").read_text(encoding="utf-8")
    assert "garch_cvar.json" in src
    assert "PUBLIC_DATA_DIR" in src


def test_attribution_save_report_dual_writes_public(tmp_path, monkeypatch):
    from src.monitor import performance_attribution as pa
    from src.monitor.performance_attribution import PerformanceAttribution

    monkeypatch.setattr(pa, "PUBLIC_DATA_DIR", tmp_path / "public")
    attr = PerformanceAttribution(data_dir=tmp_path)

    class FakeReport:
        timestamp = "2026-07-20T12:00:00"

        def to_dict(self):
            return {"timestamp": self.timestamp, "status": "no_data", "sources": {}}

    path = attr.save_report(FakeReport())
    assert path.exists()
    public_latest = tmp_path / "public" / "attribution" / "latest.json"
    assert public_latest.exists()
    payload = json.loads(public_latest.read_text())
    assert payload["status"] == "no_data"
    assert (tmp_path / "public" / "attribution" / "attribution_2026-07-20.json").exists()
