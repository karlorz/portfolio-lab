"""Batch AC residual honesty: GARCH cron DD, rebalance overdue, analytics UTC, entropy max_possible."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def _load_compute_garch_risk():
    path = Path("scripts/compute_garch_risk.py").resolve()
    spec = importlib.util.spec_from_file_location("compute_garch_risk_batch_ac", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_compute_garch_risk_script_has_policy_vs_measured_fields():
    src = Path("scripts/compute_garch_risk.py").read_text(encoding="utf-8")
    assert "max_drawdown_limit" in src
    assert "measured_max_drawdown" in src
    assert "drawdown_field_semantics" in src
    assert "max_possible" in src
    assert "policy_max_dd" in src
    # Cron producer must tag non-ops inventory so graduation multi-day SSOT
    # is not blocked by summary.total_checks=1 (GARCH residual restamp).
    assert 'inventory_role' in src
    assert "garch_risk" in src


def test_compute_garch_risk_writes_measured_not_policy_as_max_drawdown(tmp_path, monkeypatch):
    """Cron path: risk_metrics.max_drawdown is measured NAV DD, not -15 policy."""
    from src.monitor.garch_cvar import GARCHCVaRMetrics

    cgr = _load_compute_garch_risk()

    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, size=120)
    rets[50:70] = -0.02  # force a real drawdown

    fake_metrics = GARCHCVaRMetrics(
        timestamp="2026-07-20T12:00:00",
        var_95=-1.2,
        cvar_95=-1.9,
        cvar_ratio=1.58,
        tail_severity="elevated",
        max_drawdown=-15.0,  # policy echo from calculator
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

    cgr.main()

    report = json.loads((data_dir / ".health_report.json").read_text())
    assert "max_drawdown_limit" in report
    assert report.get("measured_max_drawdown") is not None
    assert report.get("measured_max_drawdown") != -15.0
    ent = report["checks"]["portfolio_entropy"]["metrics"]
    assert ent.get("max_possible") is not None
    assert ent["max_possible"] > 0

    risk = json.loads((data_dir / "risk_metrics.json").read_text())
    assert risk.get("measured_max_drawdown") is not None
    assert risk.get("max_drawdown_limit") is not None
    assert "drawdown_field_semantics" in risk

    public = json.loads((public_dir / "garch_cvar.json").read_text())
    assert public.get("measured_max_drawdown") is not None
    assert "max_drawdown_limit" in public


def test_rebalance_next_overdue_when_days_until_negative():
    import src.monitor.rebalance_health as rh

    original_dir = rh.ORDERS_DIR
    original_data = rh.DATA_DIR
    try:
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            rh.DATA_DIR = data_dir
            rh.ORDERS_DIR = data_dir / "historical_orders"
            rh.ORDERS_DIR.mkdir()
            orders = [
                {
                    "symbol": "SPY",
                    "side": "buy",
                    "estimated_value": 1000,
                    "reason": "rebalance",
                }
            ]
            path = rh.ORDERS_DIR / "order_history_20260510_120000_aaa.json"
            path.write_text(json.dumps(orders))
            result = rh.generate()
    finally:
        rh.ORDERS_DIR = original_dir
        rh.DATA_DIR = original_data

    nxt = result["next_rebalance"]
    assert nxt["days_until"] < 0
    assert nxt["status"] == "overdue"
    assert nxt["overdue"] is True
    assert nxt.get("status_reason")


def test_analytics_generated_at_is_timezone_aware_utc():
    from src.analytics.calculator import AnalyticsCalculator

    calc = AnalyticsCalculator(data_dir="/tmp/nonexistent-analytics-batch-ac")
    report = calc.generate_analytics_report()
    gen = report["generated_at"]
    assert "+00:00" in gen or gen.endswith("Z")
    dt = datetime.fromisoformat(gen.replace("Z", "+00:00"))
    assert dt.tzinfo is not None
    assert dt.utcoffset().total_seconds() == 0
