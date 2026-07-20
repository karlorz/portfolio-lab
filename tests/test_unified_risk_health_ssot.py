"""Unified risk section prefers fresher GARCH .health_report over stale risk_metrics."""
import json
from datetime import datetime

import pytest


def test_prefers_fresher_health_report(tmp_path, monkeypatch):
    import src.monitor.unified_dashboard as ud
    monkeypatch.setattr(ud, "DATA_DIR", tmp_path)
    (tmp_path / "risk_metrics.json").write_text(json.dumps({
        "timestamp": "2026-05-21T17:22:01",
        "var_95_daily": -1.41,
        "cvar_95_daily": -2.02,
        "cvar_ratio": 1.43,
        "tail_severity": "moderate",
        "garch_active": False,
        "garch_filtered": False,
    }))
    (tmp_path / ".health_report.json").write_text(json.dumps({
        "timestamp": "2026-07-20T21:57:05",
        "var_95": -1.30,
        "cvar_95": -2.07,
        "cvar_ratio": 1.58,
        "tail_severity": "elevated",
        "filter_active": True,
        "garch_filtered": True,
        "max_drawdown": -0.12,
        "current_drawdown": -0.01,
        "volatility_annual": 13.0,
    }))
    section = ud._get_risk_section()
    assert section["available"] is True
    assert section["source"] == "health_report"
    assert abs(float(section["var_95_daily"]) - (-1.30)) < 1e-9
    assert section["tail_severity"] == "elevated"
    assert section["garch_active"] is True


def test_falls_back_to_risk_metrics_when_no_health(tmp_path, monkeypatch):
    import src.monitor.unified_dashboard as ud
    monkeypatch.setattr(ud, "DATA_DIR", tmp_path)
    (tmp_path / "risk_metrics.json").write_text(json.dumps({
        "timestamp": "2026-05-21T17:22:01",
        "var_95_daily": -1.41,
        "cvar_95_daily": -2.02,
        "cvar_ratio": 1.43,
        "tail_severity": "moderate",
        "garch_active": False,
        "garch_filtered": False,
    }))
    section = ud._get_risk_section()
    assert section["available"] is True
    assert section["source"] == "risk_metrics"
    assert abs(float(section["var_95_daily"]) - (-1.41)) < 1e-9
