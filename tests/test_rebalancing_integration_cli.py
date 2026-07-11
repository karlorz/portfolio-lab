"""CLI output visibility tests for ``python -m src.rebalancing.integration``."""

from __future__ import annotations

import runpy
import sys
import warnings

import pandas as pd


def _sample_ohlcv() -> pd.DataFrame:
    index = pd.date_range("2026-07-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "open": [500.0, 501.0, 502.0, 503.0, 504.0, 505.0],
            "high": [502.0, 503.0, 504.0, 505.0, 506.0, 507.0],
            "low": [499.0, 500.0, 501.0, 502.0, 503.0, 504.0],
            "close": [501.0, 502.0, 503.0, 504.0, 505.0, 506.0],
            "volume": [200_000.0] * 6,
        },
        index=index,
    )


def _run_rebalancing_module(monkeypatch, *args: str) -> None:
    import src.signals.vpin_bvc as vpin_module

    monkeypatch.setattr(sys, "argv", ["python -m src.rebalancing.integration", *args])
    monkeypatch.setattr(
        vpin_module,
        "load_historical_bars",
        lambda symbol, days=60: _sample_ohlcv(),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=".*found in sys.modules.*",
        )
        runpy.run_module("src.rebalancing.integration", run_name="__main__")


def test_status_command_emits_visible_status_fields(monkeypatch, capsys):
    _run_rebalancing_module(monkeypatch, "status")

    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert "ytd_cost_bps" in combined
    assert "remaining_budget_pct" in combined
    assert "drift_threshold" in combined


def test_check_command_emits_visible_decision_fields(monkeypatch, capsys):
    _run_rebalancing_module(monkeypatch, "check")

    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert "Decision:" in combined
    assert "Should execute:" in combined
    assert "Max drift:" in combined
    assert "Reason:" in combined
