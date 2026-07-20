"""Overlay collar must prefer collar_signal.json SSOT over hardcoded 550 spot."""
import json
from pathlib import Path

import pytest


def test_get_collar_data_loads_saved_signal_file(tmp_path, monkeypatch):
    from src.dashboard import overlay_dashboard as od
    from src.dashboard.overlay_dashboard import OverlayDashboardGenerator

    signals_dir = tmp_path / "signals"
    signals_dir.mkdir()
    payload = {
        "timestamp": "2026-07-20T21:40:04",
        "signal_state": "collared",
        "call_strike": 767.06,
        "put_strike": 714.62,
        "underlying_price": 743.29,
        "expected_monthly_yield": 1.2,
        "max_upside_pct": 3.2,
        "max_downside_pct": 3.9,
        "vix_level": 16.76,
        "regime": "normal",
        "strikes": {
            "call_strike": 767.06,
            "put_strike": 714.62,
            "net_premium": 0.5,
            "is_cashless": False,
            "collar_cost_pct": 0.001,
        },
        "collar_notional_pct": 0.46,
        "spy_shift": 0.0,
        "confidence": 72.0,
        "is_valid": True,
        "reason": "live fixture",
    }
    (signals_dir / "collar_signal.json").write_text(json.dumps(payload))

    import src.paths as paths
    monkeypatch.setattr(paths, "SIGNALS_DIR", signals_dir)
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr("src.signals.collar_signal.CollarSignalGenerator.OUTPUT_PATH", signals_dir / "collar_signal.json")

    gen = OverlayDashboardGenerator()
    # Force load path via data_dir if supported
    if hasattr(gen, "data_dir"):
        gen.data_dir = tmp_path
    data = gen._get_collar_data()
    assert data.get("call_strike") == 767.06
    assert data.get("put_strike") == 714.62
    assert abs(float(data.get("vix_level", 0)) - 16.76) < 1e-6
    assert data.get("active") is True


def test_get_collar_data_does_not_hardcode_550_when_no_file(tmp_path, monkeypatch):
    """Without SSOT file, generate_collar_signal must be called without spot=550."""
    from src.dashboard.overlay_dashboard import OverlayDashboardGenerator
    import src.paths as paths

    calls = {}

    def fake_generate(spot=None, vix=None):
        calls["spot"] = spot
        calls["vix"] = vix
        class S:
            is_valid = True
            regime = "normal"
            call_strike = 700.0
            put_strike = 650.0
            max_upside_pct = 1.0
            max_downside_pct = 1.0
            vix_level = vix or 15.0
            confidence = 50.0
            timestamp = "2026-07-20T00:00:00"
            class strikes:
                net_premium = 0.0
                is_cashless = True
        return S()

    empty = tmp_path / "empty_signals"
    empty.mkdir()
    monkeypatch.setattr(paths, "SIGNALS_DIR", empty)
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        "src.signals.collar_signal.generate_collar_signal",
        fake_generate,
    )
    monkeypatch.setattr(
        "src.signals.collar_signal.CollarSignalGenerator.OUTPUT_PATH",
        empty / "collar_signal.json",
    )
    gen = OverlayDashboardGenerator()
    data = gen._get_collar_data()
    assert calls.get("spot") is None  # no hardcode 550
    assert data.get("call_strike") == 700.0
