"""CLI output visibility tests for ``python -m src.signals.fed_policy_overlay``."""

from __future__ import annotations

import sys

import pandas as pd

from src.signals import fed_policy_overlay
from src.signals.fed_policy_overlay import FedPolicyRegime


def _sample_data(*, string_dates: bool = False, string_timestamps: bool = False) -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2026-01-01", periods=14, freq="MS")
    if string_timestamps:
        date_values = [d.strftime("%Y-%m-%d 00:00:00") for d in dates]
    elif string_dates:
        date_values = [d.strftime("%Y-%m-%d") for d in dates]
    else:
        date_values = dates
    return {
        "FEDFUNDS": pd.DataFrame({"date": date_values, "value": [5.0] * len(dates)}),
        "CPIAUCSL": pd.DataFrame({"date": date_values, "value": [300.0 + i for i in range(len(dates))]}),
        "DFII10": pd.DataFrame({"date": date_values, "value": [1.5] * len(dates)}),
        "DGS10": pd.DataFrame({"date": date_values, "value": [4.4] * len(dates)}),
        "DGS2": pd.DataFrame({"date": date_values, "value": [4.0] * len(dates)}),
        "T10YIE": pd.DataFrame({"date": date_values, "value": [2.3] * len(dates)}),
    }


def _sample_regime() -> FedPolicyRegime:
    return FedPolicyRegime(
        timestamp="2026-07-05T00:00:00",
        regime="NEUTRAL",
        fed_funds_rate=5.0,
        inflation_yoy=2.3,
        real_rate_10y=1.5,
        real_rate_short=2.7,
        breakeven_10y=2.3,
        yield_curve_10y2y=0.4,
        confidence=0.75,
        regime_factors={"real_rate_level": 2.7},
    )


def _patch_overlay_data(monkeypatch, *, string_dates: bool = False, string_timestamps: bool = False) -> None:
    def fake_fetch_data(self, force_refresh: bool = False):
        self.data = _sample_data(string_dates=string_dates, string_timestamps=string_timestamps)
        return self.data

    def fake_detect_regime(self, timestamp: str | None = None):
        self.current_regime = _sample_regime()
        return self.current_regime

    monkeypatch.setattr(fed_policy_overlay.FedPolicyOverlay, "fetch_data", fake_fetch_data)
    monkeypatch.setattr(fed_policy_overlay.FedPolicyOverlay, "detect_regime", fake_detect_regime)


def _run_cli(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["python -m src.signals.fed_policy_overlay", *args],
    )
    fed_policy_overlay.main()


def test_status_command_emits_visible_output(monkeypatch, capsys):
    _patch_overlay_data(monkeypatch)

    _run_cli(monkeypatch, "status")

    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert "Fed Policy Overlay v2.54 - Status" in combined
    assert "FEDFUNDS" in combined


def test_status_command_handles_cached_string_dates(monkeypatch, capsys):
    _patch_overlay_data(monkeypatch, string_dates=True)

    _run_cli(monkeypatch, "status")

    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert "Fed Policy Overlay v2.54 - Status" in combined
    assert "FEDFUNDS: 2027-02-01 = 5.00" in combined


def test_status_command_normalizes_cached_string_timestamps(monkeypatch, capsys):
    _patch_overlay_data(monkeypatch, string_timestamps=True)

    _run_cli(monkeypatch, "status")

    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert "FEDFUNDS: 2027-02-01 = 5.00" in combined
    assert "FEDFUNDS: 2027-02-01 00:00:00 = 5.00" not in combined


def test_allocate_command_emits_visible_recommendation(monkeypatch, capsys):
    _patch_overlay_data(monkeypatch)

    _run_cli(monkeypatch, "allocate", "--portfolio", "46/38/16")

    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert "Fed Policy Overlay v2.54" in combined
    assert "recommended_allocation" in combined
    assert "NEUTRAL" in combined


def test_backtest_stub_emits_visible_not_implemented_message(monkeypatch, capsys):
    _run_cli(monkeypatch, "backtest", "--start", "2005-01-01")

    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert "Backtest functionality" in combined
    assert "TBD" in combined
