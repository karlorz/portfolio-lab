"""
Vol-Adjusted Momentum (Barroso/Santa-Clara 2015) — Proof of Concept

Scales momentum signal inversely to 6-month realized volatility,
targeting 12% annualized vol. Compares raw momentum vs vol-adjusted
momentum on SPY/GLD/TLT 2005-2026 data.

Reference: Barroso & Santa-Clara (2015), "Momentum Has Its Moments"
  J. Financial Economics 115(3):499-513

Usage: uv run python scripts/vol_adjusted_momentum.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent / "public" / "data" / "prices.json"
TICKERS = ["SPY", "GLD", "TLT"]
VOL_WINDOW = 126  # ~6 months of trading days
TARGET_VOL = 0.12  # 12% annualized target
MOM_WINDOW = 252  # 12-month momentum lookback


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


def compute_raw_momentum(prices: pd.DataFrame, window: int = MOM_WINDOW) -> pd.DataFrame:
    """Raw momentum: sign of trailing return."""
    # Skip last month (1m reversal effect)
    skip = 21
    trailing_ret_shifted = prices.pct_change(window).shift(skip)
    return np.sign(trailing_ret_shifted)


def compute_vol_adjusted_momentum(
    prices: pd.DataFrame,
    vol_window: int = VOL_WINDOW,
    target_vol: float = TARGET_VOL,
) -> pd.DataFrame:
    """Vol-adjusted momentum: scale by target_vol / realized_vol."""
    returns = prices.pct_change()
    realized_vol = returns.rolling(vol_window).std() * np.sqrt(252)
    raw_mom = compute_raw_momentum(prices)

    # Scale: target_vol / realized_vol (capped at 3x to avoid extreme leverage)
    vol_scale = (target_vol / realized_vol).clip(0, 3)
    return raw_mom * vol_scale


def backtest_signal(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    label: str,
) -> dict:
    """Backtest a momentum signal as an equal-weight overlay."""
    returns = prices.pct_change().dropna()
    signal_aligned = signal.reindex(returns.index).fillna(0)
    signal_shifted = signal_aligned.shift(1).fillna(0)

    # Signal-weighted returns (signal acts as position sizing)
    signal_returns = (returns * signal_shifted).mean(axis=1)

    # Equal-weight buy-and-hold
    bh_returns = returns.mean(axis=1)

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

    return {
        "label": label,
        "cagr": cagr(signal_returns),
        "sharpe": sharpe(signal_returns),
        "max_dd": max_dd(signal_returns),
        "vol": signal_returns.std() * np.sqrt(252),
        "bh_sharpe": sharpe(bh_returns),
        "bh_max_dd": max_dd(bh_returns),
        "mean_signal": signal_shifted.mean().mean(),
        "signal_std": signal_shifted.stack().std(),
    }


def main():
    prices = load_prices()

    print("=" * 70)
    print("VOL-ADJUSTED MOMENTUM (BARROSO/SANTA-CLARA 2015) — PROOF OF CONCEPT")
    print("=" * 70)
    print(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Tickers: {TICKERS}")
    print(f"Vol window: {VOL_WINDOW} days (~6 months)")
    print(f"Target vol: {TARGET_VOL:.0%} annualized")
    print(f"Momentum window: {MOM_WINDOW} days (~12 months)")
    print()

    raw_mom = compute_raw_momentum(prices)
    vol_adj_mom = compute_vol_adjusted_momentum(prices)

    r_raw = backtest_signal(prices, raw_mom, "Raw Momentum")
    r_vol = backtest_signal(prices, vol_adj_mom, "Vol-Adjusted Momentum")

    print("--- RAW MOMENTUM ---")
    print(f"  CAGR: {r_raw['cagr']:+.2%}")
    print(f"  Sharpe: {r_raw['sharpe']:.2f}")
    print(f"  Max DD: {r_raw['max_dd']:.2%}")
    print(f"  Vol: {r_raw['vol']:.2%}")
    print(f"  Mean signal: {r_raw['mean_signal']:.3f}, Signal std: {r_raw['signal_std']:.3f}")
    print()

    print("--- VOL-ADJUSTED MOMENTUM ---")
    print(f"  CAGR: {r_vol['cagr']:+.2%}")
    print(f"  Sharpe: {r_vol['sharpe']:.2f}")
    print(f"  Max DD: {r_vol['max_dd']:.2%}")
    print(f"  Vol: {r_vol['vol']:.2%}")
    print(f"  Mean signal: {r_vol['mean_signal']:.3f}, Signal std: {r_vol['signal_std']:.3f}")
    print()

    print("--- BUY-AND-HOLD (equal-weight) ---")
    print(f"  Sharpe: {r_raw['bh_sharpe']:.2f}")
    print(f"  Max DD: {r_raw['bh_max_dd']:.2%}")
    print()

    delta = r_vol['sharpe'] - r_raw['sharpe']
    print(f"Sharpe delta (vol-adj - raw): {delta:+.2f}")
    print()

    print("INTERPRETATION:")
    print("  Barroso/Santa-Clara found Sharpe 0.53 → 0.97 (nearly doubles).")
    print("  If vol-adjusted Sharpe > raw Sharpe, vol-scaling helps.")
    print("  If vol-adjusted Sharpe ≈ raw Sharpe, the effect is marginal here.")

    # Per-ticker breakdown
    print()
    print("--- PER-TICKER VOL-SCALING EFFECT ---")
    returns = prices.pct_change().dropna()
    realized_vol = returns.rolling(VOL_WINDOW).std() * np.sqrt(252)
    for ticker in TICKERS:
        avg_vol = realized_vol[ticker].mean()
        scale = TARGET_VOL / avg_vol if avg_vol > 0 else 0
        print(f"  {ticker}: avg realized vol = {avg_vol:.1%}, scale factor = {scale:.2f}x")


if __name__ == "__main__":
    main()
