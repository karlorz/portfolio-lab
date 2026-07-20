"""Batch AG residual honesty: hollow unified dual-write guard, vol/duration provenance."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_unified_skips_hollow_public_dual_write(tmp_path, monkeypatch):
    from src.monitor import unified_dashboard as ud

    private = tmp_path / "private"
    public = tmp_path / "public"
    private.mkdir()
    public.mkdir()
    monkeypatch.setattr(ud, "DATA_DIR", private)
    monkeypatch.setattr(ud, "PUBLIC_DATA_DIR", public)

    # Seed a rich public file
    rich = {
        "dashboard_version": "v6.08",
        "health": {"available": True, "status": "healthy"},
        "portfolio": {"available": True},
        "risk": {"available": True, "max_drawdown": -10.5},
        "regime": {"available": True},
        "cron": {"available": True},
        "attribution": {"available": False},
        "risk_history": {"available": True, "data_points": 3},
    }
    (public / "unified_dashboard.json").write_text(json.dumps(rich))

    hollow = {
        "dashboard_version": "v6.08",
        "health": {"available": False, "status": "unknown"},
        "portfolio": {"available": False},
        "risk": {"available": False},
        "regime": {"available": False},
        "cron": {"available": False},
        "attribution": {"available": False},
        "risk_history": {"available": False},
    }
    written = ud._save_unified_dashboard(hollow)
    assert private / "unified_dashboard.json" in written or any(
        p.name == "unified_dashboard.json" for p in written
    )
    # Public must remain rich
    pub = json.loads((public / "unified_dashboard.json").read_text())
    assert pub["risk"]["available"] is True
    assert pub["risk"]["max_drawdown"] == pytest.approx(-10.5)


def test_unified_writes_public_when_payload_has_sections(tmp_path, monkeypatch):
    from src.monitor import unified_dashboard as ud

    private = tmp_path / "private"
    public = tmp_path / "public"
    private.mkdir()
    public.mkdir()
    monkeypatch.setattr(ud, "DATA_DIR", private)
    monkeypatch.setattr(ud, "PUBLIC_DATA_DIR", public)

    payload = {
        "dashboard_version": "v6.08",
        "health": {"available": True},
        "portfolio": {"available": True},
        "risk": {"available": True, "garch_active": False, "max_drawdown": -8.0},
        "regime": {"available": False},
        "cron": {"available": True},
        "attribution": {"available": False},
        "risk_history": {"available": True},
    }
    ud._save_unified_dashboard(payload)
    pub = json.loads((public / "unified_dashboard.json").read_text())
    assert pub["risk"]["garch_active"] is False
    assert pub["risk"]["max_drawdown"] == pytest.approx(-8.0)


def test_enrich_duration_allocation_provenance_on_bare_weights():
    from src.dashboard.generator import _enrich_duration_allocation_provenance

    bare = {"tlt": 0.3, "ief": 0.4, "shy": 0.25, "bil": 0.05}
    out = _enrich_duration_allocation_provenance(bare)
    assert out["role"] == "advisory_sleeve"
    assert out["live_authoritative"] is False
    assert out["unit"] == "portfolio_weight_fraction"
    assert out["weights"]["tlt"] == 0.3
    assert out["sum"] == pytest.approx(1.0)


def test_vol_parity_to_dict_has_provenance():
    from src.strategy.vol_parity_allocator import VolParityAllocation

    alloc = VolParityAllocation(
        date="2026-07-20",
        target_volatility=10.0,
        spy_pct=36.8,
        gld_pct=30.4,
        tlt_pct=12.8,
        core_vol_contribution=11.0,
        vix_short_pct=3.8,
        vix_tail_pct=1.5,
        vix_vol_contribution=60.0,
        cash_pct=14.0,
        expected_portfolio_vol=10.5,
        expected_max_dd=15.0,
        rebalance_triggered=False,
        rebalance_reason=None,
    )
    d = alloc.to_dict()
    assert d["weight_unit"] == "percent_of_portfolio_0_100"
    assert d["live_authoritative"] is False
    assert d["role"] == "advisory_research_sleeve"
