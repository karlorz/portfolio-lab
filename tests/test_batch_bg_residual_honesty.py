"""Batch BG residual honesty: GARCH coverage demote status + VIX split_like skip."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.data.market_db_sync import (
    audit_prices_payload,
    is_volatility_index_symbol,
)


def _load_compute_garch_risk():
    path = Path("scripts/compute_garch_risk.py").resolve()
    spec = importlib.util.spec_from_file_location("compute_garch_risk_batch_bg", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_is_volatility_index_symbol_vix_family():
    assert is_volatility_index_symbol("^VIX")
    assert is_volatility_index_symbol("^VIX3M")
    assert is_volatility_index_symbol("VIX3M")
    assert is_volatility_index_symbol("vix")
    assert not is_volatility_index_symbol("SPY")
    assert not is_volatility_index_symbol("GLD")


def test_audit_skips_split_like_on_vix3m_crisis_jumps():
    """Volmageddon-style jumps on ^VIX3M must not warn overall_status."""
    prices = {
        "SPY": [
            {"d": "2018-02-02", "p": 100.0},
            {"d": "2018-02-05", "p": 98.0},
        ],
        "^VIX3M": [
            {"d": "2018-02-02", "p": 17.0},
            {"d": "2018-02-05", "p": 37.0},  # ~+118% regime jump
        ],
    }
    report = audit_prices_payload(prices)
    assert report["status"] == "ok"
    assert report["issue_counts"]["split_like_returns"] == 0
    assert report["issue_counts"]["extreme_returns"] == 0
    vix_row = next(s for s in report["symbols"] if s["symbol"] == "^VIX3M")
    assert vix_row["status"] == "ok"
    assert vix_row.get("return_anomaly_gates") == "skipped_volatility_index"


def test_audit_still_flags_equity_split_like():
    prices = {
        "SPY": [
            {"d": "2026-06-10", "p": 100.0},
            {"d": "2026-06-11", "p": 45.0},  # -55% split-like
        ],
    }
    report = audit_prices_payload(prices)
    assert report["status"] == "warn"
    assert report["issue_counts"]["split_like_returns"] == 1


def test_garch_coverage_demote_sets_status_degraded_not_healthy(tmp_path, monkeypatch):
    """coverage_pass=false → garch_active false AND status ≠ healthy."""
    from src.monitor.garch_cvar import GARCHCVaRMetrics

    cgr = _load_compute_garch_risk()

    rng = np.random.default_rng(1)
    rets = rng.normal(0.0005, 0.01, size=120)

    fake_metrics = GARCHCVaRMetrics(
        timestamp="2026-07-21T12:00:00",
        var_95=-1.0,
        cvar_95=-1.5,
        cvar_ratio=1.5,
        tail_severity="normal",
        max_drawdown=-15.0,
        current_drawdown=0.0,
        volatility_annual=12.0,
        garch_filtered=True,
        garch_omega=1e-6,
        garch_alpha=0.1,
        garch_beta=0.8,
        garch_persistence=0.9,
        conditional_volatility_current=0.8,
        historical_volatility=12.0,
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

    failed_coverage = {
        "coverage_pass": False,
        "kupiec_p_value": 0.003,
        "exceedance_rate": 0.024,
        "expected_rate": 0.05,
    }

    with patch(
        "src.monitor.conformal_risk.conformal_coverage_diagnostics",
        return_value=failed_coverage,
    ), patch(
        "src.monitor.conformal_risk.conformal_var",
        return_value=-0.02,
    ):
        cgr.main()

    report = json.loads((data_dir / ".health_report.json").read_text())
    assert report.get("garch_active") is False
    assert report.get("runtime_role") == "advisory_degraded"
    assert report.get("status") != "healthy"
    assert report.get("status") in {"degraded", "unhealthy", "warning"}
    assert report.get("summary", {}).get("passed") == 0

    risk = json.loads((data_dir / "risk_metrics.json").read_text())
    assert risk.get("garch_active") is False
    assert risk.get("status") == report.get("status")

    public = json.loads((public_dir / "garch_cvar.json").read_text())
    assert public.get("garch_active") is False
    assert public.get("status") == report.get("status")


def test_compute_garch_src_has_coverage_status_honesty():
    src = Path("scripts/compute_garch_risk.py").read_text(encoding="utf-8")
    assert "Batch BG" in src or "coverage demote must revise" in src
    assert 'report["status"] = "degraded"' in src
