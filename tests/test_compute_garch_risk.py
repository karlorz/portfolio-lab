"""Focused persistence tests for the standalone GARCH risk job."""

from __future__ import annotations

import json

from scripts.compute_garch_risk import (
    append_garch_log,
    append_risk_metrics_history,
)


def _risk_payload() -> dict:
    return {
        "timestamp": "2026-07-29T00:00:00+00:00",
        "var_95_daily": -1.2,
        "cvar_95_daily": -1.9,
        "cvar_ratio": 1.58,
        "tail_severity": "elevated",
        "garch_active": False,
        "coverage_diagnostics": {
            "kupiec_p_value": 0.0029,
            "kupiec_pass": False,
            "conditional_coverage_p_value": 0.0065,
            "conditional_coverage_pass": False,
            "exceedance_rate": 0.024,
            "coverage_rate": 0.976,
            "exceedance_bias": "under",
            "coverage_efficiency_warning": True,
        },
    }


def test_history_append_persists_coverage_and_preserves_fields(tmp_path) -> None:
    path = tmp_path / "risk_metrics_history.json"

    append_risk_metrics_history(path, _risk_payload())

    row = json.loads(path.read_text(encoding="utf-8"))[-1]
    assert row["var_95"] == -1.2
    assert row["cvar_95"] == -1.9
    assert row["source"] == "compute_garch_risk"
    assert row["coverage_diagnostics"]["kupiec_p_value"] == 0.0029
    assert row["coverage_diagnostics"]["conditional_coverage_p_value"] == 0.0065


def test_history_retention_stays_bounded(tmp_path) -> None:
    path = tmp_path / "risk_metrics_history.json"
    path.write_text(json.dumps([{"timestamp": str(i)} for i in range(720)]))

    append_risk_metrics_history(path, _risk_payload())

    rows = json.loads(path.read_text(encoding="utf-8"))
    assert len(rows) == 720
    assert rows[-1]["coverage_diagnostics"]["kupiec_pass"] is False


def test_history_records_null_when_coverage_is_unavailable(tmp_path) -> None:
    path = tmp_path / "risk_metrics_history.json"
    payload = _risk_payload()
    payload["coverage_diagnostics"] = None

    append_risk_metrics_history(path, payload)

    row = json.loads(path.read_text(encoding="utf-8"))[-1]
    assert "coverage_diagnostics" in row
    assert row["coverage_diagnostics"] is None


def test_direct_log_append_works_without_make_wrapper(tmp_path) -> None:
    path = tmp_path / "garch.log"

    append_garch_log(path, _risk_payload())

    text = path.read_text(encoding="utf-8")
    assert "GARCH-CVaR Risk" in text
    assert "Kupiec p-value: 0.0029" in text
