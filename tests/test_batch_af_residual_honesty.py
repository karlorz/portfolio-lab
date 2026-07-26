"""Batch AF residual honesty: unified public dual-write, SPC zero-var, garch active stamp."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_unified_save_dual_writes_public(tmp_path, monkeypatch):
    from src.monitor import unified_dashboard as ud

    monkeypatch.setattr(ud, "DATA_DIR", tmp_path / "private")
    monkeypatch.setattr(ud, "PUBLIC_DATA_DIR", tmp_path / "public")
    (tmp_path / "private").mkdir()
    (tmp_path / "public").mkdir()

    payload = {
        "dashboard_version": "v6.08",
        "generated_at": "2026-07-20T18:00:00+00:00",
        "risk": {"available": True, "max_drawdown": -10.5, "garch_active": False},
    }
    written = ud._save_unified_dashboard(payload)
    assert len(written) >= 1
    private = tmp_path / "private" / "unified_dashboard.json"
    public = tmp_path / "public" / "unified_dashboard.json"
    assert private.exists()
    assert public.exists()
    pub = json.loads(public.read_text())
    assert pub["risk"]["garch_active"] is False
    assert pub["risk"]["max_drawdown"] == pytest.approx(-10.5)


def test_spc_zero_variance_limits_unavailable():
    from src.monitor.spc_monitor import SPCMonitor

    mon = SPCMonitor(window_size=10, consecutive_breach_limit=3)
    for _ in range(8):
        mon.record("flat_signal", 0.5)
    # Different value would breach if limits collapsed to mean — must not count
    mon.record("flat_signal", 0.9)
    mon.record("flat_signal", 0.1)

    status = mon.get_signal_status("flat_signal")
    assert status is not None
    assert status["std"] == 0.0 or status.get("limits_status") == "unavailable_zero_variance"
    # After constant baseline, first ref has std 0
    # Re-record constants to lock zero-var reference
    mon2 = SPCMonitor(window_size=10, consecutive_breach_limit=3)
    for _ in range(10):
        mon2.record("const", -0.5)
    st = mon2.get_signal_status("const")
    assert st["std"] == 0.0
    assert st.get("ucl") is None
    assert st.get("lcl") is None
    assert st.get("limits_status") == "unavailable_zero_variance"
    assert st["is_flagged"] is False
    assert mon2.check_flags() == []


def test_compute_garch_script_always_stamps_garch_active_on_health():
    src = Path("scripts/compute_garch_risk.py").read_text(encoding="utf-8")
    assert 'report["garch_active"] = bool(risk_payload.get("garch_active"' in src
    assert "garch_active_reason" in src
