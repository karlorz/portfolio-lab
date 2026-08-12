"""Collar underlying refreshes from live SPY marks."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def test_collar_refreshes_when_spy_mark_drifts(tmp_path, monkeypatch):
    from src.dashboard.overlay_dashboard import OverlayDashboardGenerator

    # Stale saved collar
    saved = {
        "is_valid": True,
        "regime": "normal",
        "call_strike": 780,
        "put_strike": 700,
        "underlying_price": 745.91,
        "vix_level": 15,
        "confidence": 80,
        "timestamp": "2026-07-22T10:00:00+00:00",
        "strikes": {"call_strike": 780, "put_strike": 700, "net_premium": 0.1, "is_cashless": True},
    }
    (tmp_path / "signals").mkdir()
    (tmp_path / "signals" / "collar_signal.json").write_text(json.dumps(saved))
    prices = {"SPY": {"close": 748.50}}
    (tmp_path / "prices.json").write_text(json.dumps(prices))

    dash = OverlayDashboardGenerator()
    monkeypatch.setattr("src.dashboard.overlay_dashboard.DATA_DIR", tmp_path)

    # Patch load + live mark + generate
    mock_signal = MagicMock()
    mock_signal.is_valid = True
    mock_signal.regime = "normal"
    mock_signal.call_strike = 790.0
    mock_signal.put_strike = 710.0
    mock_signal.strikes.net_premium = 0.1
    mock_signal.strikes.is_cashless = True
    mock_signal.max_upside_pct = 5.0
    mock_signal.max_downside_pct = 5.0
    mock_signal.vix_level = 15.0
    mock_signal.confidence = 85.0
    mock_signal.underlying_price = 748.50
    mock_signal.timestamp = "2026-07-22T12:00:00+00:00"

    with patch.object(dash, "_load_collar_signal_file", return_value=saved):
        with patch.object(dash, "_live_spy_mark", return_value=748.50):
            with patch(
                "src.signals.collar_signal.CollarSignalGenerator"
            ) as Gen:
                Gen.return_value.generate_signal.return_value = mock_signal
                data = dash._get_collar_data()

    assert data.get("underlying_price") == 748.50
    assert data.get("source") == "collar_refresh_from_spy_marks"
    assert abs(data["underlying_price"] - data["spy_mark"]) < 0.01
