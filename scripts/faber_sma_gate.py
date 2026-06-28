"""
Faber 10-Month SMA Gate — Proof of Concept

Computes 200-day SMA for SPY/GLD/TLT, generates binary gate signal
(price > SMA → risk-on), and measures hit rate, average return,
Sharpe ratio, and max drawdown.

Usage: uv run python scripts/faber_sma_gate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent / "public" / "data" / "prices.json"
SMA_WINDOW = 200  # ~10 months of trading days
TICKERS = ["SPY", "GLD", "TLT"]


def load_prices() -> pd.DataFrame:
    with open(DATA_FILE) as f:
        raw = json.load(f)
    rows = []
    for sym, data in raw.items():
        if sym not in TICKERS:
            continue
        for rec in data:
            rows.append({"date": rec["d"], "sym": sym, "close": rec["p"]})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="sym", values="close").sort_index().dropna()


def compute_gate(prices: pd.DataFrame, window: int = SMA_WINDOW) -> pd.DataFrame:
    """Binary gate: 1 if price > SMA, else 0."""
    sma = prices.rolling(window).mean()
    return (prices > sma).astype(int)


def backtest_gate(prices: pd.DataFrame, gate: pd.DataFrame, ticker: str) -> dict:
    """Compute returns when gate is on vs off."""
    returns = prices[ticker].pct_change().dropna()
    gate_aligned = gate[ticker].reindex(returns.index).fillna(0)
    gate_shifted = gate_aligned.shift(1).fillna(0)

    on_returns = returns[gate_shifted == 1]
    off_returns = returns[gate_shifted == 0]

    def sharpe(r):
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return r.mean() / r.std() * np.sqrt(252)

    def max_dd(r):
        cum = (1 + r).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd.min()

    return {
        "ticker": ticker,
        "gate_on_pct": gate_shifted.mean(),
        "on_count": len(on_returns),
        "off_count": len(off_returns),
        "on_mean_annual": on_returns.mean() * 252,
        "off_mean_annual": off_returns.mean() * 252,
        "on_sharpe": sharpe(on_returns),
        "off_sharpe": sharpe(off_returns),
        "on_max_dd": max_dd(on_returns),
        "off_max_dd": max_dd(off_returns),
    }


def main():
    prices = load_prices()
    gate = compute_gate(prices)

    print("=" * 70)
    print("FABER 10-MONTH SMA GATE — PROOF OF CONCEPT")
    print("=" * 70)
    print(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Tickers: {TICKERS}")
    print(f"SMA window: {SMA_WINDOW} days (~10 months)")
    print()

    for ticker in TICKERS:
        r = backtest_gate(prices, gate, ticker)
        print(f"--- {ticker} ---")
        print(f"  Gate ON:  {r['gate_on_pct']:.1%} of days ({r['on_count']} obs)")
        print(f"  Gate OFF: {1 - r['gate_on_pct']:.1%} of days ({r['off_count']} obs)")
        print(f"  ON  → annual return: {r['on_mean_annual']:+.2%}, Sharpe: {r['on_sharpe']:.2f}, MaxDD: {r['on_max_dd']:.2%}")
        print(f"  OFF → annual return: {r['off_mean_annual']:+.2%}, Sharpe: {r['off_sharpe']:.2f}, MaxDD: {r['off_max_dd']:.2%}")
        delta = r['on_sharpe'] - r['off_sharpe']
        print(f"  Sharpe delta (ON - OFF): {delta:+.2f}")
        print()

    # Equal-weight portfolio when all gates are on
    all_returns = prices.pct_change().dropna()
    gate_all = gate.reindex(all_returns.index).fillna(0)
    gate_all_shifted = gate_all.shift(1).fillna(0)
    portfolio_on = (all_returns * gate_all_shifted).mean(axis=1)
    portfolio_off = (all_returns * (1 - gate_all_shifted)).mean(axis=1)

    def sharpe(r):
        return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0

    def max_dd(r):
        cum = (1 + r).cumprod()
        return ((cum - cum.cummax()) / cum.cummax()).min()

    print("--- EQUAL-WEIGHT PORTFOLIO (all 3 tickers) ---")
    print(f"  Gate ALL-ON:  Sharpe {sharpe(portfolio_on):.2f}, MaxDD {max_dd(portfolio_on):.2%}")
    print(f"  Gate ANY-OFF: Sharpe {sharpe(portfolio_off):.2f}, MaxDD {max_dd(portfolio_off):.2%}")

    # Buy-and-hold comparison
    bh_returns = all_returns.mean(axis=1)
    print(f"  Buy-and-Hold: Sharpe {sharpe(bh_returns):.2f}, MaxDD {max_dd(bh_returns):.2%}")
    print()
    print("INTERPRETATION:")
    print("  If gate ON Sharpe >> gate OFF Sharpe, the SMA filter adds value.")
    print("  If gate ON Sharpe ≈ buy-and-hold, the filter mainly reduces drawdown.")


if __name__ == "__main__":
    main()
