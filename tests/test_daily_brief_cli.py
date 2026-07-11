"""CLI output visibility tests for ``python -m src.monitor.daily_brief``."""

import runpy
import sys
import warnings


def _sample_dashboard() -> dict:
    return {
        "health": {"available": True, "status": "healthy", "alerts": []},
        "portfolio": {
            "available": True,
            "total_value": 250000,
            "positions": [
                {"symbol": "SPY", "weight": 44.0, "value": 110000},
                {"symbol": "GLD", "weight": 36.0, "value": 90000},
                {"symbol": "TLT", "weight": 18.0, "value": 45000},
            ],
        },
        "risk": {
            "available": True,
            "current_drawdown": -8.5,
            "var_95_daily": -1.2,
            "volatility_annual": 11.5,
        },
        "overlays": {
            "collar": {"active": True},
            "bond_duration": {"active": True},
            "_meta": {"active_count": 2, "total_count": 2},
        },
        "regime": {
            "available": True,
            "classifier": {"current_regime": "normal", "confidence": 0.75},
        },
        "tca": {
            "available": True,
            "scorecard": {"avg_slippage_bps": 5.2, "total_orders": 3},
        },
        "attribution": {
            "available": True,
            "sources": [
                {"name": "TSFM Momentum", "total_return_bps": 12.5},
                {"name": "Risk Budget", "total_return_bps": -3.2},
            ],
        },
    }


def _run_daily_brief_module(monkeypatch, tmp_path, *args) -> None:
    import src.monitor.unified_dashboard as dashboard_module
    import src.paths as paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        dashboard_module,
        "generate_unified_dashboard",
        lambda: _sample_dashboard(),
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["python -m src.monitor.daily_brief", *args],
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=".*found in sys.modules.*",
        )
        runpy.run_module("src.monitor.daily_brief", run_name="__main__")


def test_no_narrative_command_emits_visible_brief(monkeypatch, tmp_path, capsys):
    _run_daily_brief_module(monkeypatch, tmp_path, "--no-narrative")

    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert "PORTFOLIO-LAB DAILY BRIEF" in combined
    assert "PORTFOLIO SNAPSHOT" in combined


def test_default_command_emits_visible_brief_after_narrative_fallback(
    monkeypatch, tmp_path, capsys
):
    _run_daily_brief_module(monkeypatch, tmp_path)

    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert "PORTFOLIO-LAB DAILY BRIEF" in combined
    assert "ACTION ITEMS" in combined


def test_save_command_emits_visible_confirmation(monkeypatch, tmp_path, capsys):
    _run_daily_brief_module(monkeypatch, tmp_path, "--save", "--no-narrative")

    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert "PORTFOLIO-LAB DAILY BRIEF" in combined
    assert "Saved to" in combined
    assert (tmp_path / "daily_brief.json").exists()
