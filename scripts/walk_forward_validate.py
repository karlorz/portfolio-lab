#!/usr/bin/env python3
"""
Walk-Forward Validation for Champion Portfolio — v1.0

Runs rolling train/test windows to produce out-of-sample (OOS) Sharpe
estimates. This is the gold standard for validating whether the grid-search
champion Sharpe (0.79) is real or inflated by in-sample optimization.

Method:
- Train window: 2 years (504 trading days)
- Test window: 6 months (126 trading days)
- Step: 6 months (non-overlapping test periods)
- Produces ~30+ OOS Sharpe estimates across 2005-2026

Also computes the Deflated Sharpe Ratio (DSR) to adjust for multiple
testing across 94 grid-search configurations.

Usage:
    uv run python scripts/walk_forward_validate.py
    uv run python scripts/walk_forward_validate.py --save
    uv run python scripts/walk_forward_validate.py --train 3 --test 1
"""

import json
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Ensure project root is on sys.path for src.* imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.paths import PRICES_JSON, DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


def load_prices(path: Path) -> pd.DataFrame:
    """Load prices.json into a wide DataFrame (dates x symbols)."""
    with open(path) as f:
        raw = json.load(f)

    records = []
    for symbol, bars in raw.items():
        for bar in bars:
            records.append({
                "date": bar.get("d", ""),
                "symbol": symbol,
                "price": float(bar.get("p", bar.get("close", 0))),
            })

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot(index="date", columns="symbol", values="price")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index().dropna()
    return pivot


def compute_sharpe(returns: np.ndarray, annualize: bool = True) -> float:
    """Compute Sharpe ratio from daily returns."""
    if len(returns) < 10:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret == 0:
        return 0.0
    sharpe = mean_ret / std_ret
    if annualize:
        sharpe *= np.sqrt(TRADING_DAYS_PER_YEAR)
    return sharpe


def compute_max_drawdown(equity: np.ndarray) -> float:
    """Compute maximum drawdown as a negative percentage."""
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(np.min(drawdown)) * 100


def portfolio_daily_returns(
    prices: pd.DataFrame,
    weights: dict[str, float],
) -> np.ndarray:
    """Compute daily portfolio returns from price DataFrame and weights."""
    symbols = [s for s in weights if s in prices.columns]
    if not symbols:
        return np.array([])

    price_matrix = prices[symbols].values
    if price_matrix.shape[0] < 2:
        return np.array([])

    daily_returns = np.diff(price_matrix, axis=0) / price_matrix[:-1]
    w = np.array([weights[s] for s in symbols])
    return daily_returns @ w


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_tests: int,
    sample_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute Deflated Sharpe Ratio (DSR).

    Adjusts for multiple testing by computing the probability that the
    observed Sharpe exceeds the expected maximum Sharpe under the null.

    Reference: Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"

    Args:
        observed_sharpe: The observed Sharpe ratio (annualized).
        n_tests: Number of independent tests (e.g., 94 grid configs).
        sample_length: Number of observations used to compute Sharpe.
        skewness: Skewness of returns.
        kurtosis: Excess kurtosis of returns.

    Returns:
        DSR probability (0-1). Values > 0.95 suggest genuine skill.
    """
    from scipy.stats import norm

    # Expected maximum Sharpe under multiple testing
    # E[max(SR)] ≈ (1 - gamma) * phi^{-1}(1 - 1/N) + gamma * phi^{-1}(1 - 1/(N*e))
    # Simplified: use the first term (conservative)
    if n_tests < 1:
        n_tests = 1

    # Expected maximum SR under null (Bailey & Lopez de Prado Eq. 11)
    z_alpha = norm.ppf(1 - 1.0 / n_tests)
    expected_max_sr = z_alpha

    # SE of Sharpe ratio estimate (approximate)
    # SE(SR) ≈ sqrt((1 - skewness*SR + (kurtosis-1)/4 * SR^2) / (T-1))
    sr = observed_sharpe / np.sqrt(TRADING_DAYS_PER_YEAR)  # De-annualize
    se_sr = np.sqrt(
        (1 - skewness * sr + (kurtosis - 1) / 4 * sr ** 2)
        / max(sample_length - 1, 1)
    ) * np.sqrt(TRADING_DAYS_PER_YEAR)  # Re-annualize SE

    if se_sr == 0:
        return 0.0

    # DSR = Prob(SR* > E[max(SR)])
    # = CDF((observed - expected_max) / SE)
    dsr = norm.cdf((observed_sharpe - expected_max_sr) / se_sr)
    return dsr


def run_walk_forward(
    prices: pd.DataFrame,
    weights: dict[str, float],
    train_years: float = 2.0,
    test_years: float = 0.5,
    step_years: float = 0.5,
) -> list[dict]:
    """Run walk-forward validation with rolling train/test windows.

    Returns list of dicts with train/test metrics per window.
    """
    train_days = int(train_years * TRADING_DAYS_PER_YEAR)
    test_days = int(test_years * TRADING_DAYS_PER_YEAR)
    step_days = int(step_years * TRADING_DAYS_PER_YEAR)

    n_days = len(prices)
    results = []

    start = 0
    window_id = 0

    while start + train_days + test_days <= n_days:
        train_end = start + train_days
        test_end = train_end + test_days

        train_prices = prices.iloc[start:train_end]
        test_prices = prices.iloc[train_end:test_end]

        # In-sample metrics
        train_returns = portfolio_daily_returns(train_prices, weights)
        train_sharpe = compute_sharpe(train_returns)

        # Out-of-sample metrics
        test_returns = portfolio_daily_returns(test_prices, weights)
        test_sharpe = compute_sharpe(test_returns)

        # Cumulative equity for drawdown
        test_equity = np.cumprod(1 + test_returns) * 100000
        test_max_dd = compute_max_drawdown(test_equity)

        results.append({
            "window_id": window_id,
            "train_start": str(prices.index[start].date()),
            "train_end": str(prices.index[train_end - 1].date()),
            "test_start": str(prices.index[train_end].date()),
            "test_end": str(prices.index[test_end - 1].date()),
            "train_sharpe": round(train_sharpe, 4),
            "test_sharpe": round(test_sharpe, 4),
            "test_max_drawdown": round(test_max_dd, 2),
            "test_total_return": round((test_equity[-1] / test_equity[0] - 1) * 100, 2),
            "n_test_days": len(test_returns),
        })

        window_id += 1
        start += step_days

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward Validation for Champion Portfolio"
    )
    parser.add_argument(
        "--train", type=float, default=2.0,
        help="Training window in years (default: 2.0)"
    )
    parser.add_argument(
        "--test", type=float, default=0.5,
        help="Test window in years (default: 0.5)"
    )
    parser.add_argument(
        "--step", type=float, default=0.5,
        help="Step size in years (default: 0.5)"
    )
    parser.add_argument(
        "--weights", type=str, default="champion",
        help="Portfolio weights: 'champion' (46/38/16), 'max_sharpe' (40/34/26), or JSON"
    )
    parser.add_argument(
        "--n-tests", type=int, default=94,
        help="Number of grid-search configs for DSR (default: 94)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save results to JSON"
    )
    args = parser.parse_args()

    # Resolve weights
    if args.weights == "champion":
        weights = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    elif args.weights == "max_sharpe":
        weights = {"SPY": 0.3985, "GLD": 0.3428, "TLT": 0.2587}
    else:
        weights = json.loads(args.weights)

    if not PRICES_JSON.exists():
        logger.error("Prices file not found: %s", PRICES_JSON)
        return

    prices = load_prices(PRICES_JSON)
    if prices.empty:
        logger.error("No price data loaded")
        return

    # Filter to symbols in weights
    symbols = [s for s in weights if s in prices.columns]
    prices = prices[symbols].dropna()
    logger.info(
        "Loaded %d trading days for %d symbols: %s",
        len(prices), len(symbols), symbols
    )

    # Run walk-forward
    logger.info(
        "Running walk-forward: train=%.1fyr, test=%.1fyr, step=%.1fyr",
        args.train, args.test, args.step,
    )
    results = run_walk_forward(
        prices, weights,
        train_years=args.train,
        test_years=args.test,
        step_years=args.step,
    )

    if not results:
        logger.error("No walk-forward windows produced — data insufficient")
        return

    # Aggregate OOS metrics
    test_sharpes = [r["test_sharpe"] for r in results]
    test_dds = [r["test_max_drawdown"] for r in results]
    test_returns = [r["test_total_return"] for r in results]

    mean_oos_sharpe = float(np.mean(test_sharpes))
    median_oos_sharpe = float(np.median(test_sharpes))
    oos_sharpe_std = float(np.std(test_sharpes, ddof=1))
    worst_oos_sharpe = float(np.min(test_sharpes))
    best_oos_sharpe = float(np.max(test_sharpes))
    pct_positive_sharpe = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes) * 100
    avg_max_dd = float(np.mean(test_dds))

    # Deflated Sharpe Ratio
    total_test_days = int(args.test * TRADING_DAYS_PER_YEAR)
    dsr = deflated_sharpe_ratio(
        observed_sharpe=mean_oos_sharpe,
        n_tests=args.n_tests,
        sample_length=total_test_days,
    )

    # Print results
    print(f"\n{'='*70}")
    print("  Walk-Forward Validation Results")
    print(f"{'='*70}")
    print(f"  Portfolio: {', '.join(f'{s} {w:.0%}' for s, w in weights.items())}")
    print(f"  Train: {args.train}yr | Test: {args.test}yr | Step: {args.step}yr")
    print(f"  Windows: {len(results)}")
    print(f"  Period: {results[0]['train_start']} to {results[-1]['test_end']}")

    print(f"\n  --- Out-of-Sample Summary ---")
    print(f"  Mean OOS Sharpe:     {mean_oos_sharpe:.4f}")
    print(f"  Median OOS Sharpe:   {median_oos_sharpe:.4f}")
    print(f"  OOS Sharpe Std:      {oos_sharpe_std:.4f}")
    print(f"  Worst OOS Sharpe:    {worst_oos_sharpe:.4f}")
    print(f"  Best OOS Sharpe:     {best_oos_sharpe:.4f}")
    print(f"  % Positive Sharpe:   {pct_positive_sharpe:.1f}%")
    print(f"  Avg Max Drawdown:    {avg_max_dd:.2f}%")
    print(f"\n  --- Deflated Sharpe Ratio ---")
    print(f"  Grid configs tested: {args.n_tests}")
    print(f"  DSR probability:     {dsr:.4f}")
    print(f"  DSR > 0.95 (genuine): {'YES' if dsr > 0.95 else 'NO'}")

    print(f"\n  --- Per-Window Results ---")
    print(f"  {'Window':<7} {'Train Period':<24} {'Test Period':<24} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'OOS DD%':>8} {'OOS Ret%':>9}")
    print(f"  {'-'*7} {'-'*24} {'-'*24} {'-'*10} {'-'*11} {'-'*8} {'-'*9}")
    for r in results:
        print(
            f"  {r['window_id']:<7} "
            f"{r['train_start']} to {r['train_end']:<12} "
            f"{r['test_start']} to {r['test_end']:<12} "
            f"{r['train_sharpe']:>10.4f} "
            f"{r['test_sharpe']:>11.4f} "
            f"{r['test_max_drawdown']:>8.2f} "
            f"{r['test_total_return']:>9.2f}"
        )

    if args.save:
        output = {
            "weights": weights,
            "config": {
                "train_years": args.train,
                "test_years": args.test,
                "step_years": args.step,
                "n_grid_configs": args.n_tests,
            },
            "summary": {
                "mean_oos_sharpe": round(mean_oos_sharpe, 4),
                "median_oos_sharpe": round(median_oos_sharpe, 4),
                "oos_sharpe_std": round(oos_sharpe_std, 4),
                "worst_oos_sharpe": round(worst_oos_sharpe, 4),
                "best_oos_sharpe": round(best_oos_sharpe, 4),
                "pct_positive_sharpe": round(pct_positive_sharpe, 1),
                "avg_max_drawdown": round(avg_max_dd, 2),
                "deflated_sharpe_ratio": round(dsr, 4),
                "n_windows": len(results),
            },
            "windows": results,
        }
        output_path = DATA_DIR / "walk_forward_validation.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
