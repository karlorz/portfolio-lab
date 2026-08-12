#!/usr/bin/env python3
"""
Correlation-Regime-Adaptive Allocation Backtest.

Tests whether dynamically adjusting the TLT/IEF split based on the GLD-TLT
correlation regime improves the champion portfolio's risk-adjusted returns.

Key hypothesis: When GLD-TLT correlation is positive (correlated regime),
shifting from TLT to IEF preserves the diversification benefit that the
46/38/16 allocation relies on.

Methodology:
- Compute rolling 252-day GLD-TLT correlation
- When correlation > 0.15 (correlated): shift TLT allocation toward IEF
- When correlation < -0.15 (diversifying): keep full TLT allocation
- In between (neutral): blend proportionally
- Compare against static 46/38/16 and 44/36/20 benchmarks

Usage:
    python -m src.backtest.correlation_adaptive_backtest
    python -m src.backtest.correlation_adaptive_backtest --save
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.paths import BASE_ALLOCATION, DATA_DIR, PRICES_JSON
from src.backtest.metrics import (
    compute_metrics,
    save_results_json,
)
from src.data.price_cache import get_prices_df

logger = logging.getLogger(__name__)

_PRICE_SYMBOLS = ["SPY", "GLD", "TLT", "IEF"]
_DEFAULT_PRICES_JSON = PRICES_JSON

__all__ = [
    "CorrelationAdaptiveResult",
    "compute_correlation_adaptive_backtest",
    "run_backtest",
]


@dataclass
class CorrelationAdaptiveResult:
    """Result from correlation-adaptive allocation backtest."""
    analysis_date: str
    base_weights: Dict[str, float]
    adaptive_weights_mean: Dict[str, float]
    static_sharpe: float
    adaptive_sharpe: float
    sharpe_delta: float
    static_max_dd: float
    adaptive_max_dd: float
    correlation_regime_distribution: Dict[str, int]
    ief_shift_frequency: float  # % of days where IEF > 0
    summary: str


def _load_prices_from_json(prices_path: Path | str) -> pd.DataFrame:
    """Load price data from an explicit prices.json path."""
    with open(prices_path) as f:
        raw = json.load(f)

    frames = {}
    for sym in _PRICE_SYMBOLS:
        entries = raw.get(sym, [])
        if isinstance(entries, list) and len(entries) > 0 and isinstance(entries[0], dict):
            dates = [e["d"] for e in entries]
            prices = [e["p"] for e in entries]
            frames[sym] = pd.Series(prices, index=pd.to_datetime(dates), name=sym)

    df = pd.DataFrame(frames).dropna()
    df.index.name = "date"
    return df


def _load_prices() -> pd.DataFrame:
    """Load price data through the shared cache unless PRICES_JSON is overridden."""
    if Path(PRICES_JSON) != Path(_DEFAULT_PRICES_JSON):
        return _load_prices_from_json(PRICES_JSON)

    df = get_prices_df(symbols=_PRICE_SYMBOLS).dropna()
    df.index.name = "date"
    return df


def _compute_rolling_correlation(
    prices: pd.DataFrame,
    sym_a: str = "GLD",
    sym_b: str = "TLT",
    window: int = 252,
) -> pd.Series:
    """Compute rolling correlation between two symbols."""
    returns = prices[[sym_a, sym_b]].pct_change().dropna()
    return returns[sym_a].rolling(window).corr(returns[sym_b]).dropna()


def _get_adaptive_weights(
    correlation: float,
    base_allocation: Dict[str, float],
    max_ief_shift: float = 0.50,
) -> Dict[str, float]:
    """Compute adaptive allocation based on GLD-TLT correlation.

    When correlation > 0.15 (correlated): shift up to max_ief_shift of
    TLT allocation to IEF proportionally.

    When correlation < -0.15 (diversifying): keep full TLT.

    In neutral zone (-0.15 to 0.15): proportional blend.
    """
    tlt_weight = base_allocation.get("TLT", BASE_ALLOCATION["TLT"])
    ief_weight = base_allocation.get("IEF", 0.0)
    total_bond = tlt_weight + ief_weight

    if correlation > 0.15:
        # Correlated regime: shift toward IEF proportionally
        shift_fraction = min((correlation - 0.15) / 0.35, 1.0) * max_ief_shift
    elif correlation < -0.15:
        # Diversifying regime: keep full TLT
        shift_fraction = 0.0
    else:
        # Neutral zone: partial shift
        shift_fraction = ((correlation + 0.15) / 0.30) * max_ief_shift * 0.5

    ief_allocation = total_bond * shift_fraction
    tlt_allocation = total_bond - ief_allocation

    weights = dict(base_allocation)
    weights["TLT"] = round(tlt_allocation, 4)
    weights["IEF"] = round(ief_allocation, 4)

    return weights


def _run_portfolio_backtest(
    prices: pd.DataFrame,
    weights_list: List[Dict[str, float]],
    rebalance_days: int = 21,
    transaction_cost_bps: float = 10.0,
) -> Tuple[List[float], List[float]]:
    """Run buy-and-hold backtest with periodic rebalancing.

    Args:
        prices: DataFrame with price columns
        weights_list: List of weight dicts (one per day, for adaptive)
        rebalance_days: Rebalance frequency
        transaction_cost_bps: Transaction cost in basis points

    Returns:
        Tuple of (daily_returns, portfolio_values)
    """
    symbols = ["SPY", "GLD", "TLT", "IEF"]
    n_days = len(prices)
    capital = 100000.0
    values = [capital]
    returns = []

    # Track positions (number of shares)
    current_weights = weights_list[0]
    positions = {s: 0.0 for s in symbols}

    # Initial purchase
    for sym in symbols:
        w = current_weights.get(sym, 0.0)
        price = prices[sym].iloc[0]
        if price > 0 and w > 0:
            positions[sym] = (capital * w) / price

    days_since_rebalance = 0

    for i in range(1, n_days):
        # Compute portfolio value
        portfolio_value = sum(
            positions.get(s, 0) * prices[s].iloc[i]
            for s in symbols
        )

        daily_return = (portfolio_value / values[-1]) - 1 if values[-1] > 0 else 0.0
        returns.append(daily_return)
        values.append(portfolio_value)

        days_since_rebalance += 1

        # Rebalance check
        if days_since_rebalance >= rebalance_days and i < len(weights_list):
            target_weights = weights_list[i]
            total_value = portfolio_value

            if total_value > 0:
                for sym in symbols:
                    w = target_weights.get(sym, 0.0)
                    price = prices[sym].iloc[i]
                    if price > 0:
                        target_shares = (total_value * w) / price
                        current_value = positions.get(sym, 0) * price
                        target_value = total_value * w
                        trade_value = abs(target_value - current_value)

                        # Transaction cost
                        cost = trade_value * transaction_cost_bps / 10000
                        total_value -= cost

                        positions[sym] = target_shares

                days_since_rebalance = 0

    return returns, values


def compute_correlation_adaptive_backtest(
    base_allocation: Optional[Dict[str, float]] = None,
    corr_window: int = 252,
    max_ief_shift: float = 0.50,
    rebalance_days: int = 21,
    save: bool = False,
) -> CorrelationAdaptiveResult:
    """Run correlation-regime-adaptive allocation backtest.

    Args:
        base_allocation: Base allocation dict (default: 46/38/16)
        corr_window: Rolling correlation window (default: 252)
        max_ief_shift: Maximum fraction of TLT to shift to IEF (default: 0.50)
        rebalance_days: Rebalance frequency (default: 21 = monthly)
        save: Whether to save results

    Returns:
        CorrelationAdaptiveResult
    """
    if base_allocation is None:
        base_allocation = dict(BASE_ALLOCATION, IEF=0.00)

    logger.info("Loading prices for correlation-adaptive backtest")
    prices = _load_prices()
    logger.info("Loaded %d days of price data", len(prices))

    # Compute rolling correlation
    rolling_corr = _compute_rolling_correlation(prices, "GLD", "TLT", corr_window)

    # Align prices with correlation data
    common_idx = prices.index.intersection(rolling_corr.index)
    prices_aligned = prices.loc[common_idx]
    corr_aligned = rolling_corr.loc[common_idx]

    logger.info("Aligned: %d days with correlation data", len(common_idx))

    # Build adaptive weights for each day
    adaptive_weights_list = []
    regime_counts = {"diversifying": 0, "neutral": 0, "correlated": 0}
    ief_shift_days = 0

    for i in range(len(corr_aligned)):
        corr = corr_aligned.iloc[i]
        adaptive_w = _get_adaptive_weights(corr, base_allocation, max_ief_shift)
        adaptive_weights_list.append(adaptive_w)

        if corr < -0.15:
            regime_counts["diversifying"] += 1
        elif corr > 0.15:
            regime_counts["correlated"] += 1
        else:
            regime_counts["neutral"] += 1

        if adaptive_w.get("IEF", 0) > 0.001:
            ief_shift_days += 1

    # Static weights list (same weights every day)
    static_weights_list = [dict(base_allocation)] * len(prices_aligned)

    # Run backtests
    logger.info("Running static backtest (46/38/16)")
    static_returns, static_values = _run_portfolio_backtest(
        prices_aligned, static_weights_list, rebalance_days,
    )

    logger.info("Running adaptive backtest (correlation-regime)")
    adaptive_returns, adaptive_values = _run_portfolio_backtest(
        prices_aligned, adaptive_weights_list, rebalance_days,
    )

    # Compute metrics
    static_metrics = compute_metrics(
        equity_curve=static_values,
        initial_capital=100000.0,
    )
    adaptive_metrics = compute_metrics(
        equity_curve=adaptive_values,
        initial_capital=100000.0,
    )

    static_sharpe = static_metrics.sharpe_ratio
    adaptive_sharpe = adaptive_metrics.sharpe_ratio
    sharpe_delta = round(adaptive_sharpe - static_sharpe, 4)

    # Mean adaptive weights
    mean_weights = {}
    for sym in ["SPY", "GLD", "TLT", "IEF"]:
        vals = [w.get(sym, 0) for w in adaptive_weights_list]
        mean_weights[sym] = round(float(np.mean(vals)), 4)

    ief_frequency = round(ief_shift_days / len(corr_aligned), 4) if len(corr_aligned) > 0 else 0.0

    # Summary
    if sharpe_delta > 0.02:
        verdict = "ADAPTIVE IMPROVES — correlation-regime switching adds value"
    elif sharpe_delta > 0:
        verdict = "MARGINAL — adaptive slightly improves, may not justify complexity"
    elif sharpe_delta > -0.02:
        verdict = "NEUTRAL — adaptive and static are equivalent"
    else:
        verdict = "ADAPTIVE HURTS — correlation switching reduces returns"

    summary = (
        f"Correlation-adaptive allocation: Sharpe {adaptive_sharpe:.4f} vs "
        f"static {static_sharpe:.4f} (delta {sharpe_delta:+.4f}). "
        f"IEF shift frequency: {ief_frequency:.1%}. "
        f"Regime distribution: {regime_counts}. {verdict}"
    )

    result = CorrelationAdaptiveResult(
        analysis_date=datetime.now().isoformat(),
        base_weights=base_allocation,
        adaptive_weights_mean=mean_weights,
        static_sharpe=round(static_sharpe, 4),
        adaptive_sharpe=round(adaptive_sharpe, 4),
        sharpe_delta=sharpe_delta,
        static_max_dd=round(static_metrics.max_drawdown, 4),
        adaptive_max_dd=round(adaptive_metrics.max_drawdown, 4),
        correlation_regime_distribution=regime_counts,
        ief_shift_frequency=ief_frequency,
        summary=summary,
    )

    if save:
        output_path = DATA_DIR / "correlation_adaptive_backtest.json"
        save_results_json(asdict(result), output_path=str(output_path))
        logger.info("Saved results to %s", output_path)

    logger.info("Correlation-Adaptive Backtest Results:")
    logger.info("  Static Sharpe:   %.4f", static_sharpe)
    logger.info("  Adaptive Sharpe: %.4f", adaptive_sharpe)
    logger.info("  Delta:           %+.4f", sharpe_delta)
    logger.info("  Static Max DD:   %.4f", static_metrics.max_drawdown)
    logger.info("  Adaptive Max DD: %.4f", adaptive_metrics.max_drawdown)
    logger.info("  IEF shift freq:  %.1f%%", ief_frequency * 100)
    logger.info("  Regimes:         %s", regime_counts)

    return result


def main():
    """CLI entry point."""
    import argparse
    from src.utils.log_config import configure_logging
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Correlation-Regime-Adaptive Allocation Backtest"
    )
    parser.add_argument("--max-ief-shift", type=float, default=0.50,
                        help="Max fraction of TLT to shift to IEF (0-1)")
    parser.add_argument("--corr-window", type=int, default=252,
                        help="Rolling correlation window (days)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to JSON")
    args = parser.parse_args()

    result = compute_correlation_adaptive_backtest(
        max_ief_shift=args.max_ief_shift,
        corr_window=args.corr_window,
        save=args.save,
    )

    logger.info("\n%s", "=" * 60)
    logger.info("CORRELATION-ADAPTIVE ALLOCATION BACKTEST")
    logger.info("%s", "=" * 60)
    logger.info("  Static Sharpe:     %.4f", result.static_sharpe)
    logger.info("  Adaptive Sharpe:   %.4f", result.adaptive_sharpe)
    logger.info("  Delta:             %+.4f", result.sharpe_delta)
    logger.info("  Static Max DD:     %.4f", result.static_max_dd)
    logger.info("  Adaptive Max DD:   %.4f", result.adaptive_max_dd)
    logger.info("  IEF Shift Freq:    %.1f%%", result.ief_shift_frequency * 100)
    logger.info("  Regime Dist:       %s", result.correlation_regime_distribution)
    logger.info("  Mean Weights:      %s", result.adaptive_weights_mean)
    logger.info("\n  %s", result.summary)
    logger.info("%s", "=" * 60)


if __name__ == "__main__":
    main()
