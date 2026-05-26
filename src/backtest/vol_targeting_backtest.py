#!/usr/bin/env python3
"""
Volatility Targeting Overlay Backtest.

Tests whether scaling portfolio positions to maintain a constant volatility
target improves risk-adjusted returns for the champion allocation.

Academic basis: Moreira & Muir (2017) show +0.30-0.50 Sharpe for equities;
Lohre et al. (2020) show +0.02-0.05 for multi-asset portfolios.

Methodology:
- Compute portfolio realized volatility over a rolling window
- When vol < target: scale up positions (up to max_leverage)
- When vol > target: scale down positions (deleverage)
- Rebalance monthly

Usage:
    python -m src.backtest.vol_targeting_backtest
    python -m src.backtest.vol_targeting_backtest --target-vol 0.11 --save
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.paths import DATA_DIR, PRICES_JSON
from src.backtest.metrics import (
    BacktestMetrics,
    compute_metrics,
    save_results_json,
)

logger = logging.getLogger(__name__)

__all__ = [
    "VolTargetResult",
    "compute_vol_target_backtest",
    "run_backtest",
]


@dataclass
class VolTargetResult:
    """Result from volatility targeting backtest."""
    analysis_date: str
    base_allocation: Dict[str, float]
    target_vol: float
    vol_lookback: int
    max_leverage: float
    static_sharpe: float
    vol_target_sharpe: float
    sharpe_delta: float
    static_cagr: float
    vol_target_cagr: float
    static_max_dd: float
    vol_target_max_dd: float
    mean_leverage: float
    max_leverage_reached: float
    leverage_above_1_pct: float  # % of days with leverage > 1
    summary: str


def _load_prices() -> pd.DataFrame:
    """Load price data from prices.json."""
    with open(PRICES_JSON) as f:
        raw = json.load(f)

    frames = {}
    for sym in ["SPY", "GLD", "TLT", "IEF"]:
        entries = raw.get(sym, [])
        if isinstance(entries, list) and len(entries) > 0 and isinstance(entries[0], dict):
            dates = [e["d"] for e in entries]
            prices = [e["p"] for e in entries]
            frames[sym] = pd.Series(prices, index=pd.to_datetime(dates), name=sym)

    df = pd.DataFrame(frames).dropna()
    df.index.name = "date"
    return df


def _compute_vol_target_leverage(
    realized_vol: float,
    target_vol: float,
    max_leverage: float = 2.0,
    smoothing: float = 0.67,
    prev_leverage: float = 1.0,
) -> float:
    """Compute leverage factor for volatility targeting.

    Args:
        realized_vol: Current realized volatility (annualized)
        target_vol: Target volatility (annualized)
        max_leverage: Maximum allowed leverage
        smoothing: Smoothing coefficient (0-1). Lower = more responsive.
        prev_leverage: Previous period's leverage factor

    Returns:
        Leverage factor (1.0 = no scaling)
    """
    if realized_vol <= 0:
        return 1.0

    raw_leverage = target_vol / realized_vol
    # Smooth to prevent excessive turnover
    smoothed = smoothing * raw_leverage + (1 - smoothing) * prev_leverage
    return round(max(1.0 / max_leverage, min(max_leverage, smoothed)), 4)


def compute_vol_target_backtest(
    base_allocation: Optional[Dict[str, float]] = None,
    target_vol: float = 0.11,
    vol_lookback: int = 63,
    max_leverage: float = 2.0,
    smoothing: float = 0.67,
    rebalance_days: int = 21,
    transaction_cost_bps: float = 10.0,
    save: bool = False,
) -> VolTargetResult:
    """Run volatility targeting overlay backtest.

    Args:
        base_allocation: Base allocation (default: 46/38/16)
        target_vol: Target annualized volatility (default: 0.11)
        vol_lookback: Volatility estimation window in days (default: 63)
        max_leverage: Maximum leverage multiple (default: 2.0)
        smoothing: Leverage smoothing coefficient (default: 0.67)
        rebalance_days: Rebalance frequency (default: 21 = monthly)
        transaction_cost_bps: Transaction cost in basis points
        save: Whether to save results

    Returns:
        VolTargetResult
    """
    if base_allocation is None:
        base_allocation = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "IEF": 0.00}

    logger.info("Loading prices for volatility targeting backtest")
    prices = _load_prices()
    logger.info("Loaded %d days of price data", len(prices))

    symbols = ["SPY", "GLD", "TLT", "IEF"]
    n_days = len(prices)

    # Buy-and-hold returns for the portfolio at each step
    portfolio_returns = np.zeros(n_days)
    portfolio_returns[0] = 0.0

    for i in range(1, n_days):
        ret = 0.0
        for sym in symbols:
            w = base_allocation.get(sym, 0)
            if w > 0:
                sym_ret = prices[sym].iloc[i] / prices[sym].iloc[i - 1] - 1
                ret += w * sym_ret
        portfolio_returns[i] = ret

    # Static backtest
    static_equity = [100000.0]
    for i in range(1, n_days):
        static_equity.append(static_equity[-1] * (1 + portfolio_returns[i]))

    # Vol-targeted backtest
    capital = 100000.0
    vol_target_equity = [capital]
    leverage_history = [1.0]
    prev_leverage = 1.0
    cash = 0.0

    for i in range(1, n_days):
        # Compute realized vol over lookback
        if i >= vol_lookback:
            hist_returns = portfolio_returns[i - vol_lookback : i]
            realized_vol = float(np.std(hist_returns) * np.sqrt(252))
        else:
            # Insufficient history: use expanding window
            hist_returns = portfolio_returns[1:i] if i > 1 else np.array([0.0])
            realized_vol = float(np.std(hist_returns) * np.sqrt(252)) if len(hist_returns) > 2 else 0.15

        # Compute target leverage
        leverage = _compute_vol_target_leverage(
            realized_vol, target_vol, max_leverage, smoothing, prev_leverage,
        )

        # Apply leverage to daily return
        # Leverage > 1 means borrow at risk-free rate to scale up
        # Leverage < 1 means hold cash for the remainder
        excess_leverage = leverage - 1.0
        if excess_leverage > 0:
            # Borrow cash to invest more
            borrowed = capital * excess_leverage
            invested_return = (capital + borrowed) * portfolio_returns[i]
            # Borrowing cost: risk-free rate (4.5% annualized)
            daily_rf = 0.045 / 252
            borrowing_cost = borrowed * daily_rf
            daily_pnl = invested_return - borrowing_cost
        else:
            # Hold cash: (1-leverage) fraction in cash earning risk-free
            invested_return = capital * leverage * portfolio_returns[i]
            cash_return = capital * (1 - leverage) * (0.045 / 252)
            daily_pnl = invested_return + cash_return

        capital += daily_pnl

        # Rebalance cost: transaction cost on leverage change
        if abs(leverage - prev_leverage) > 0.01:
            trade_value = abs(leverage - prev_leverage) * capital
            capital -= trade_value * transaction_cost_bps / 10000

        vol_target_equity.append(capital)
        leverage_history.append(leverage)
        prev_leverage = leverage

    # Compute metrics
    static_metrics = compute_metrics(
        equity_curve=static_equity,
        initial_capital=100000.0,
    )
    vol_metrics = compute_metrics(
        equity_curve=vol_target_equity,
        initial_capital=100000.0,
    )

    static_sharpe = static_metrics.sharpe_ratio
    vol_target_sharpe = vol_metrics.sharpe_ratio
    sharpe_delta = round(vol_target_sharpe - static_sharpe, 4)

    mean_leverage = round(float(np.mean(leverage_history)), 4)
    max_lev = round(float(np.max(leverage_history)), 4)
    above_1_pct = round(
        float(sum(1 for l in leverage_history if l > 1.01) / len(leverage_history)), 4
    )

    # Summary
    if sharpe_delta > 0.03:
        verdict = "VOL TARGET STRONG — significant Sharpe improvement"
    elif sharpe_delta > 0.01:
        verdict = "VOL TARGET MODEST — mild improvement, reduced tail risk"
    elif sharpe_delta > -0.01:
        verdict = "VOL TARGET NEUTRAL — similar risk-adjusted returns"
    else:
        verdict = "VOL TARGET NEGATIVE — reduces returns without offsetting benefit"

    summary = (
        f"Vol targeting ({target_vol:.0%} target, {vol_lookback}d lookback): "
        f"Sharpe {vol_target_sharpe:.4f} vs static {static_sharpe:.4f} "
        f"(delta {sharpe_delta:+.4f}). "
        f"Mean leverage: {mean_leverage:.2f}x, max: {max_lev:.2f}x, "
        f"above 1x: {above_1_pct:.1%}. {verdict}"
    )

    result = VolTargetResult(
        analysis_date=datetime.now().isoformat(),
        base_allocation=base_allocation,
        target_vol=target_vol,
        vol_lookback=vol_lookback,
        max_leverage=max_leverage,
        static_sharpe=round(static_sharpe, 4),
        vol_target_sharpe=round(vol_target_sharpe, 4),
        sharpe_delta=sharpe_delta,
        static_cagr=round(static_metrics.cagr, 4),
        vol_target_cagr=round(vol_metrics.cagr, 4),
        static_max_dd=round(static_metrics.max_drawdown, 4),
        vol_target_max_dd=round(vol_metrics.max_drawdown, 4),
        mean_leverage=mean_leverage,
        max_leverage_reached=max_lev,
        leverage_above_1_pct=above_1_pct,
        summary=summary,
    )

    if save:
        output_path = DATA_DIR / "vol_targeting_backtest.json"
        save_results_json(asdict(result), output_path=str(output_path))
        logger.info("Saved results to %s", output_path)

    logger.info("Vol Targeting Backtest Results:")
    logger.info("  Static Sharpe:     %.4f (CAGR: %.4f, DD: %.4f)",
                static_sharpe, static_metrics.cagr, static_metrics.max_drawdown)
    logger.info("  Vol Target Sharpe: %.4f (CAGR: %.4f, DD: %.4f)",
                vol_target_sharpe, vol_metrics.cagr, vol_metrics.max_drawdown)
    logger.info("  Delta:             %+.4f", sharpe_delta)
    logger.info("  Leverage:          mean=%.2fx, max=%.2fx, >1x=%.1f%%",
                mean_leverage, max_lev, above_1_pct * 100)

    return result


def main():
    """CLI entry point."""
    import argparse
    from src.utils.log_config import configure_logging
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Volatility Targeting Overlay Backtest"
    )
    parser.add_argument("--target-vol", type=float, default=0.11,
                        help="Target annualized volatility (default: 0.11)")
    parser.add_argument("--vol-lookback", type=int, default=63,
                        help="Volatility estimation window in days (default: 63)")
    parser.add_argument("--max-leverage", type=float, default=2.0,
                        help="Maximum leverage multiple (default: 2.0)")
    parser.add_argument("--smoothing", type=float, default=0.67,
                        help="Leverage smoothing coefficient (default: 0.67)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to JSON")
    args = parser.parse_args()

    result = compute_vol_target_backtest(
        target_vol=args.target_vol,
        vol_lookback=args.vol_lookback,
        max_leverage=args.max_leverage,
        smoothing=args.smoothing,
        save=args.save,
    )

    print(f"\n{'='*60}")
    print("VOLATILITY TARGETING OVERLAY BACKTEST")
    print(f"{'='*60}")
    print(f"  Static Sharpe:      {result.static_sharpe:.4f}")
    print(f"  Vol Target Sharpe:  {result.vol_target_sharpe:.4f}")
    print(f"  Delta:              {result.sharpe_delta:+.4f}")
    print(f"  Static CAGR:        {result.static_cagr:.4f}")
    print(f"  Vol Target CAGR:    {result.vol_target_cagr:.4f}")
    print(f"  Static Max DD:      {result.static_max_dd:.4f}")
    print(f"  Vol Target Max DD:  {result.vol_target_max_dd:.4f}")
    print(f"  Mean Leverage:      {result.mean_leverage:.2f}x")
    print(f"  Max Leverage:       {result.max_leverage_reached:.2f}x")
    print(f"  Leverage > 1x:      {result.leverage_above_1_pct:.1%}")
    print(f"\n  {result.summary}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
