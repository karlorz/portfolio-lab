"""MTM previous_total uses prior session only."""

from __future__ import annotations

from scripts.mark_to_market import mark_to_market


def test_same_day_peak_does_not_poison_previous_total():
    portfolio = {
        "mode": "paper",
        "cash": 10000,
        "positions": {
            "SPY": {"shares": 10, "avg_price": 100, "current_price": 100, "value": 1000}
        },
        "history": [
            {
                "timestamp": "2026-07-20T16:00:00",
                "session_date": "2026-07-20",
                "total_value": 100000,
                "daily_return": 0.0,
            },
            # same-session intermediate peak (should be ignored as previous_total)
            {
                "timestamp": "2026-07-21T10:00:00",
                "session_date": "2026-07-21",
                "total_value": 110000,
                "daily_return": 0.1,
            },
        ],
    }
    prices = {"SPY": 100.0}  # position still 1000; cash 10000 → total 11000
    # Force session date to 2026-07-21 via monkeypatch in test body
    from unittest.mock import patch

    with patch("scripts.mark_to_market.us_cash_session_date", return_value="2026-07-21"):
        updated = mark_to_market(portfolio, prices)
    hist = updated["history"]
    last = hist[-1]
    assert last["session_date"] == "2026-07-21"
    # previous_total must be 100000 (prior session), not 110000 peak
    assert last["previous_total"] == 100000
    # total_value = 10000 + 10*100 = 11000
    assert last["total_value"] == 11000
    expected_ret = (11000 - 100000) / 100000
    assert abs(last["daily_return"] - expected_ret) < 1e-6
