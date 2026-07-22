"""Batch EH: public dual-write for risk_metrics.json (was private-only).

Investigate residual (batch AR→BT): risk_metrics public missing while garch_cvar
was dual-written. Deep-research: measure lag after both writes; content-hash
equality clears sticky dual_write_lag_stale.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_compute_garch_risk():
    path = Path("scripts/compute_garch_risk.py").resolve()
    spec = importlib.util.spec_from_file_location("compute_garch_risk_batch_eh", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_compute_garch_risk_dual_writes_public_risk_metrics(tmp_path, monkeypatch):
    from src.monitor.garch_cvar import GARCHCVaRMetrics

    cgr = _load_compute_garch_risk()

    rng = np.random.default_rng(1)
    rets = rng.normal(0.0005, 0.01, size=120)

    fake_metrics = GARCHCVaRMetrics(
        timestamp="2026-07-22T08:00:00",
        var_95=-1.1,
        cvar_95=-1.7,
        cvar_ratio=1.55,
        tail_severity="elevated",
        max_drawdown=-15.0,
        current_drawdown=0.0,
        volatility_annual=11.0,
        garch_filtered=True,
        garch_omega=1e-6,
        garch_alpha=0.1,
        garch_beta=0.8,
        garch_persistence=0.9,
        conditional_volatility_current=0.7,
        historical_volatility=11.0,
        filter_active=True,
        filter_reason=None,
    )

    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public"
    data_dir.mkdir()
    public_dir.mkdir()

    monkeypatch.setattr(cgr, "DATA_DIR", data_dir)
    monkeypatch.setattr(cgr, "PUBLIC_DATA_DIR", public_dir)
    monkeypatch.setattr(cgr, "MARKET_DB", tmp_path / "missing.db")
    monkeypatch.setattr(cgr, "compute_portfolio_returns", lambda *a, **k: rets)
    monkeypatch.setattr(cgr, "calculate_garch_cvar", lambda **kw: fake_metrics)
    monkeypatch.setattr(
        sys,
        "argv",
        ["compute_garch_risk.py", "--window", "60", "--days", "120"],
    )

    cgr.main()

    private_risk = data_dir / "risk_metrics.json"
    public_risk = public_dir / "risk_metrics.json"
    public_garch = public_dir / "garch_cvar.json"

    assert private_risk.is_file()
    assert public_risk.is_file(), "Batch EH: risk_metrics must dual-write to public"
    assert public_garch.is_file()

    priv = json.loads(private_risk.read_text(encoding="utf-8"))
    pub = json.loads(public_risk.read_text(encoding="utf-8"))
    garch = json.loads(public_garch.read_text(encoding="utf-8"))

    # Twin body preserves risk schema fields
    assert priv.get("var_95_daily") is not None
    assert pub.get("var_95_daily") == priv.get("var_95_daily")
    assert pub.get("cvar_95_daily") == priv.get("cvar_95_daily")
    assert pub.get("source") == "compute_garch_risk"

    # Post-sync provenance on risk twin
    pc = pub.get("provenance_completeness") or {}
    assert pc.get("dual_write_attempted") is True
    assert pc.get("dual_write_ok") is True
    # Content hash should clear sticky lag when twin payloads match
    if pc.get("content_hash_identical") is True:
        assert pc.get("dual_write_lag_stale") is False
        assert pc.get("dual_write_lag_seconds") == 0.0

    # garch_cvar remains enriched panel (schema_version) and present
    assert garch.get("schema_version") == "garch-cvar/v1"
    assert garch.get("var_95_daily") is not None


def test_public_data_index_catalogs_risk_metrics():
    from src.dashboard.public_data_index import (
        _OPTIONAL_PUBLIC_DATA_FILES,
        _PUBLIC_DATA_CONTRACT,
    )

    assert "risk_metrics.json" in _PUBLIC_DATA_CONTRACT
    assert "garch_cvar.json" in _PUBLIC_DATA_CONTRACT
    assert "risk_metrics.json" in _OPTIONAL_PUBLIC_DATA_FILES
    assert "garch_cvar.json" in _OPTIONAL_PUBLIC_DATA_FILES
