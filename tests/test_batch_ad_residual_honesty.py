"""Batch AD residual honesty: unified risk DD/demote, GARCH UTC, two_stage nulls."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest


def test_unified_risk_prefers_measured_drawdown_and_demote(tmp_path):
    """health_report policy -15 must not beat measured DD + coverage demote."""
    from src.monitor.unified_dashboard import _get_risk_section

    health = {
        "timestamp": "2026-07-20T12:00:00",
        "var_95": -1.2,
        "cvar_95": -1.9,
        "cvar_ratio": 1.59,
        "tail_severity": "elevated",
        "max_drawdown_limit": -15.0,
        "measured_max_drawdown": -10.5,
        "measured_current_drawdown": -7.3,
        "current_drawdown": -7.3,
        "filter_active": True,
        "garch_filtered": True,
        "volatility_annual": 13.0,
        "drawdown_field_semantics": "limit=policy; measured=NAV",
    }
    risk = {
        "timestamp": "2026-07-20T12:00:00",
        "var_95_daily": -1.2,
        "cvar_95_daily": -1.9,
        "cvar_ratio": 1.59,
        "tail_severity": "elevated",
        "max_drawdown": -10.5,
        "measured_max_drawdown": -10.5,
        "max_drawdown_limit": -15.0,
        "current_drawdown": -7.3,
        "garch_active": False,
        "garch_filtered": True,
        "runtime_role": "advisory_degraded",
        "garch_active_reason": "coverage_pass=false",
        "volatility_annual": 13.0,
    }
    (tmp_path / ".health_report.json").write_text(json.dumps(health))
    (tmp_path / "risk_metrics.json").write_text(json.dumps(risk))

    with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
        section = _get_risk_section()

    assert section["available"] is True
    assert section["max_drawdown"] == pytest.approx(-10.5)
    assert section["max_drawdown"] != pytest.approx(-15.0)
    assert section["garch_active"] is False
    assert section.get("runtime_role") == "advisory_degraded"
    assert section.get("measured_max_drawdown") == pytest.approx(-10.5)
    assert section.get("max_drawdown_limit") == pytest.approx(-15.0)


def test_unified_risk_demotes_when_only_risk_metrics_has_false(tmp_path):
    """Sibling risk_metrics demote must override health filter_active=true."""
    from src.monitor.unified_dashboard import _get_risk_section

    health = {
        "timestamp": "2026-07-20T12:00:01",
        "var_95": -1.2,
        "cvar_95": -1.9,
        "cvar_ratio": 1.5,
        "max_drawdown": -15.0,  # legacy policy echo
        "current_drawdown": 0.0,
        "filter_active": True,
        "garch_filtered": True,
    }
    risk = {
        "timestamp": "2026-07-20T12:00:00",  # slightly older
        "var_95_daily": -1.2,
        "cvar_95_daily": -1.9,
        "garch_active": False,
        "runtime_role": "advisory_degraded",
        "garch_active_reason": "coverage_pass=false",
        "measured_max_drawdown": -8.0,
        "max_drawdown_limit": -15.0,
    }
    (tmp_path / ".health_report.json").write_text(json.dumps(health))
    (tmp_path / "risk_metrics.json").write_text(json.dumps(risk))

    with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
        section = _get_risk_section()

    # Health wins by timestamp but demote fields merge from sibling
    assert section["garch_active"] is False
    assert section["max_drawdown"] == pytest.approx(-8.0)


def test_garch_cvar_timestamp_is_utc_aware():
    from src.monitor.garch_cvar import GARCHFilteredCVaR
    import numpy as np

    calc = GARCHFilteredCVaR(window=60)
    rets = np.random.default_rng(1).normal(0, 0.01, size=80)
    metrics = calc.compute(rets, current_drawdown=-0.01, max_drawdown=-0.15)
    ts = metrics.timestamp
    assert "+00:00" in ts or ts.endswith("Z")
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert dt.tzinfo is not None
    assert dt.utcoffset().total_seconds() == 0


def test_two_stage_unavailable_uses_null_metrics_not_zeros():
    # Source contract: unavailable two_stage uses null metric slots
    src = Path("src/dashboard/generator.py").read_text(encoding="utf-8")
    assert "generator_returned_none" in src
    assert '"confidence": None' in src
    assert '"regime": None' in src
    # Old dishonest zeros should not remain in the unavailable block
    assert '"confidence": 0.0' not in src.split("generator_returned_none")[0][-400:]
