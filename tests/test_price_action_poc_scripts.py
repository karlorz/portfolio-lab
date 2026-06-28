import pandas as pd

from scripts.faber_sma_gate import backtest_gate, compute_gate
from scripts.vol_adjusted_momentum import backtest_signal


def test_faber_gate_backtest_uses_prior_day_gate_for_returns():
    prices = pd.DataFrame(
        {"SPY": [100.0, 100.0, 120.0], "GLD": [100.0, 100.0, 100.0], "TLT": [100.0, 100.0, 100.0]},
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
    )
    gate = compute_gate(prices, window=2)

    result = backtest_gate(prices, gate, "SPY")

    assert result["on_count"] == 0
    assert result["off_count"] == 2


def test_vol_adjusted_momentum_backtest_uses_prior_day_signal_for_returns():
    prices = pd.DataFrame(
        {"SPY": [100.0, 120.0], "GLD": [100.0, 100.0], "TLT": [100.0, 100.0]},
        index=pd.date_range("2026-01-01", periods=2, freq="D"),
    )
    signal = pd.DataFrame(
        {"SPY": [0.0, 1.0], "GLD": [0.0, 0.0], "TLT": [0.0, 0.0]},
        index=prices.index,
    )

    result = backtest_signal(prices, signal, "same-day signal")

    assert result["cagr"] == 0.0
    assert result["sharpe"] == 0.0
