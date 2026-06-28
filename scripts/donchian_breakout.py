"""
Donchian Channel Breakout Ensemble — Proof of Concept

Implements dual 20/55-day Donchian channel on SPY with 200-day SMA filter.
Computes signal hit rate, average return, max drawdown, and compares
single-parameter vs ensemble approach.

Reference: Zarattini & Antonacci (2025), "A Century of Profitable Industry Trends"
  https://concretumgroup.com/a-century-of-profitable-industry-trends/

Usage: uv run python scripts/donchian_breakout.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent / "public" / "data" / "prices.json"
TICKERS = ["SPY"]  # Focus on SPY for clarity
CHANNEL_PERIODS = [20, 55, 252]  # Short, medium, long
SMA_FILTER = 200  # Only trade long when price > 200-day SMA


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


def donchian_signal(prices: pd.Series, period: int, sma_filter: int = SMA_FILTER) -> pd.Series:
    """
    Donchian channel breakout signal.
    Long when price breaks above N-day high AND price > SMA filter.
    Exit when price breaks below N-day low.

    Uses shifted channel (previous day's high/low) to avoid lookahead bias.
    """
    # Shift channel by 1 day to avoid lookahead (use prior day's high/low)
    high = prices.rolling(period).max().shift(1)
    low = prices.rolling(period).min().shift(1)
    sma = prices.rolling(sma_filter).mean()

    signal = pd.Series(0, index=prices.index)
    position = 0

    for i in range(max(period, sma_filter) + 1, len(prices)):
        price = prices.iloc[i]

        if position == 0:
            # Entry: price breaks above prior day's channel high AND above SMA
            if price > high.iloc[i] and price > sma.iloc[i]:
                position = 1
        elif position == 1:
            # Exit: price breaks below prior day's channel low
            if price < low.iloc[i]:
                position = 0

        signal.iloc[i] = position

    return signal


def backtest_signal(prices: pd.Series, signal: pd.Series, label: str) -> dict:
    """Compute returns for a given signal."""
    returns = prices.pct_change().dropna()
    signal_aligned = signal.reindex(returns.index).fillna(0)

    # Signal returns (shifted by 1 to avoid lookahead)
    signal_shifted = signal_aligned.shift(1).fillna(0)
    strategy_returns = returns * signal_shifted

    def sharpe(r):
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return r.mean() / r.std() * np.sqrt(252)

    def max_dd(r):
        cum = (1 + r).cumprod()
        peak = cum.cummax()
        return ((cum - peak) / peak).min()

    def cagr(r):
        total = (1 + r).prod()
        years = len(r) / 252
        return total ** (1 / years) - 1 if years > 0 else 0.0

    # Hit rate: % of trades that are profitable
    trade_entries = signal_shifted.diff().abs()
    trade_returns = strategy_returns[trade_entries > 0]
    hit_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0.0

    # Trade count
    n_trades = int(trade_entries.sum() / 2)  # entry + exit = 1 trade

    return {
        "label": label,
        "cagr": cagr(strategy_returns),
        "sharpe": sharpe(strategy_returns),
        "max_dd": max_dd(strategy_returns),
        "vol": strategy_returns.std() * np.sqrt(252),
        "hit_rate": hit_rate,
        "n_trades": n_trades,
        "pct_invested": signal_shifted.mean(),
        "bh_sharpe": sharpe(returns),
        "bh_max_dd": max_dd(returns),
        "bh_cagr": cagr(returns),
    }


def main():
    prices = load_prices()
    spy = prices["SPY"]

    print("=" * 70)
    print("DONCHIAN CHANNEL BREAKOUT ENSEMBLE — PROOF OF CONCEPT")
    print("=" * 70)
    print(f"Data: {spy.index[0].date()} to {spy.index[-1].date()}")
    print(f"Ticker: SPY")
    print(f"Channel periods: {CHANNEL_PERIODS}")
    print(f"SMA filter: {SMA_FILTER} days")
    print()

    # Single-parameter Donchian
    results = []
    for period in CHANNEL_PERIODS:
        signal = donchian_signal(spy, period)
        r = backtest_signal(spy, signal, f"Donchian {period}d")
        results.append(r)
        print(f"--- {r['label']} ---")
        print(f"  CAGR: {r['cagr']:+.2%}, Sharpe: {r['sharpe']:.2f}, MaxDD: {r['max_dd']:.2%}")
        print(f"  Hit rate: {r['hit_rate']:.1%}, Trades: {r['n_trades']}, Invested: {r['pct_invested']:.1%}")
        print()

    # Ensemble: average signal across all periods
    signals = [donchian_signal(spy, p) for p in CHANNEL_PERIODS]
    ensemble_signal = pd.concat(signals, axis=1).mean(axis=1)
    # Round to binary (majority vote)
    ensemble_binary = (ensemble_signal >= 0.5).astype(int)

    r_ens = backtest_signal(spy, ensemble_binary, "Ensemble (20/55/252d)")
    results.append(r_ens)
    print(f"--- {r_ens['label']} ---")
    print(f"  CAGR: {r_ens['cagr']:+.2%}, Sharpe: {r_ens['sharpe']:.2f}, MaxDD: {r_ens['max_dd']:.2%}")
    print(f"  Hit rate: {r_ens['hit_rate']:.1%}, Trades: {r_ens['n_trades']}, Invested: {r_ens['pct_invested']:.1%}")
    print()

    # Buy-and-hold comparison
    returns = spy.pct_change().dropna()
    def sharpe(r):
        return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    def max_dd(r):
        cum = (1 + r).cumprod()
        return ((cum - cum.cummax()) / cum.cummax()).min()
    def cagr(r):
        total = (1 + r).prod()
        years = len(r) / 252
        return total ** (1 / years) - 1 if years > 0 else 0.0

    print(f"--- BUY-AND-HOLD SPY ---")
    print(f"  CAGR: {cagr(returns):+.2%}, Sharpe: {sharpe(returns):.2f}, MaxDD: {max_dd(returns):.2%}")
    print()

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<25} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Hit%':>8} {'Trades':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<25} {r['cagr']:>+7.2%} {r['sharpe']:>8.2f} {r['max_dd']:>8.2%} {r['hit_rate']:>7.1%} {r['n_trades']:>8}")
    print(f"{'Buy-and-Hold':<25} {cagr(returns):>+7.2%} {sharpe(returns):>8.2f} {max_dd(returns):>8.2%} {'':>8} {'':>8}")
    print()
    print("INTERPRETATION:")
    print("  Donchian often underperforms buy-and-hold on CAGR but cuts drawdowns.")
    print("  Ensemble of multiple horizons should be more robust than any single one.")
    print("  If MaxDD reduction is significant, Donchian could serve as a risk filter.")


if __name__ == "__main__":
    main()
