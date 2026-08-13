#!/usr/bin/env python3
"""
Gold Allocation Sensitivity Sweep — 2005 to 2026

Tests GLD allocations from 18% to 40% (2% steps) across SPY/GLD/TLT and
SPY/GLD/IEF portfolios to validate or challenge the current champion
SPY/GLD/TLT 46/38/16 (Sharpe 0.79).

Usage:
    python scripts/gold_allocation_sweep.py

Output:
    data/gold_allocation_sweep_2026.json  — full results
    stdout summary table sorted by Sharpe
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to sys.path so we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.metrics import (  # noqa: E402  # deliberate placement (bootstrap/sys.path ordering)
    BacktestMetrics,
    compute_crisis_returns,
    compute_metrics,
    save_results_json,
)
from src.paths import DATA_DIR, PRICES_JSON  # noqa: E402  # deliberate placement (bootstrap/sys.path ordering)
from src.backtest.metrics import TRADING_DAYS_PER_YEAR  # noqa: E402  # deliberate placement (bootstrap/sys.path ordering)
from src.utils.log_config import configure_logging  # noqa: E402  # deliberate placement (bootstrap/sys.path ordering)

logger = logging.getLogger("gold_sweep")

# ── Constants ──────────────────────────────────────────────────────────
PRICES_PATH = PRICES_JSON
OUTPUT_PATH = DATA_DIR / "gold_allocation_sweep_2026.json"

# GLD sweep range
GLD_MIN = 18
GLD_MAX = 40
GLD_STEP = 2

# SPY sweep range: SPY from (100-GLD-20)% to (100-GLD)% in 2% steps
SPY_STEP = 2

# Rebalancing
ANNUAL_DAYS = TRADING_DAYS_PER_YEAR  # 252

# Crisis years to evaluate
CRISIS_YEARS = ["2008", "2020", "2022"]


def load_prices(path: Path) -> Dict[str, np.ndarray]:
    """Load price data from prices.json and return aligned arrays per symbol."""
    logger.info("Loading prices from %s", path)
    with open(path) as f:
        raw = json.load(f)

    symbols_needed = ["SPY", "GLD", "TLT", "IEF"]
    data: Dict[str, List[Dict]] = {}
    for sym in symbols_needed:
        if sym not in raw:
            raise ValueError(f"Symbol {sym} not found in prices.json")
        data[sym] = raw[sym]

    # Find common date range across all symbols
    all_dates: List[str] = sorted(set(v["d"] for values in data.values() for v in values))

    # Build aligned price arrays
    prices: Dict[str, np.ndarray] = {}
    for sym in symbols_needed:
        lookup = {item["d"]: item["p"] for item in data[sym]}
        prices[sym] = np.array([lookup[d] for d in all_dates])

    logger.info(
        "Loaded %d symbols, %d trading days, %s to %s",
        len(prices),
        len(all_dates),
        all_dates[0],
        all_dates[-1],
    )
    return prices, all_dates


def compute_daily_returns(prices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute daily simple returns for each symbol."""
    returns: Dict[str, np.ndarray] = {}
    for sym, arr in prices.items():
        returns[sym] = arr[1:] / arr[:-1] - 1.0
    return returns


def run_single_backtest(
    daily_returns: Dict[str, np.ndarray],
    dates: List[str],
    spy_w: float,
    gld_w: float,
    bond_w: float,
    bond_symbol: str,
    initial_capital: float = 1.0,
) -> BacktestMetrics:
    """Run a buy-and-hold backtest with annual rebalancing.

    Args:
        daily_returns: Daily returns for each symbol (aligned arrays).
        dates: Trading dates (len = len(returns) + 1).
        spy_w: SPY target weight (0-1).
        gld_w: GLD target weight (0-1).
        bond_w: Bond target weight (0-1).
        bond_symbol: "TLT" or "IEF".
        initial_capital: Starting capital (default 1.0 for normalized).

    Returns:
        BacktestMetrics with CAGR, Sharpe, max drawdown, etc.
    """
    n = len(daily_returns["SPY"])  # Number of return days
    weights = np.array([spy_w, gld_w, bond_w])
    returns_matrix = np.column_stack([
        daily_returns["SPY"],
        daily_returns["GLD"],
        daily_returns[bond_symbol],
    ])

    # Annual rebalancing: reset to target weights every 252 trading days
    # In between, let weights drift naturally
    daily_port_returns = np.zeros(n)
    current_weights = weights.copy()

    # We need to track the actual returns of each asset to compute drift
    # But for simplicity, we rebalance at fixed intervals by resetting weights
    for i in range(n):
        # Check if rebalance needed
        if i > 0 and i % ANNUAL_DAYS == 0:
            current_weights = weights.copy()

        # Portfolio return for this day is weighted sum of asset returns
        daily_port_returns[i] = np.dot(current_weights, returns_matrix[i])

        # Drift the weights by the asset returns (the weights grow/shrink with the assets)
        # But since we reset on the next rebalance, we don't need to track intra-period weights
        # for the annual rebalance case. The dot product approach above is an approximation.
        # For a proper implementation, we need to track actual weight drift.

    # Let me re-do this as a proper implementation
    # Track weight drift properly: after day i, weight of asset j changes by:
    # w_j_new = w_j * (1 + r_j) / (1 + sum(w_k * r_k))
    equity_curve = [initial_capital]
    current_weights = weights.copy()
    port_value = initial_capital

    for i in range(n):
        # Compute portfolio return using current drifted weights
        port_ret = np.dot(current_weights, returns_matrix[i])
        port_value *= (1.0 + port_ret)
        equity_curve.append(port_value)

        # Update drifted weights
        # w_j' = w_j * (1 + r_j) / (1 + port_ret)
        current_weights = current_weights * (1.0 + returns_matrix[i])
        current_weights /= current_weights.sum()  # Normalize

        # Rebalance back to target weights at year boundaries
        if (i + 1) % ANNUAL_DAYS == 0:
            current_weights = weights.copy()

    metrics = compute_metrics(
        equity_curve=equity_curve,
        initial_capital=initial_capital,
    )

    return metrics


def compute_crisis_returns_for_config(
    daily_returns: Dict[str, np.ndarray],
    dates: List[str],
    spy_w: float,
    gld_w: float,
    bond_w: float,
    bond_symbol: str,
) -> Dict[str, float]:
    """Compute crisis year returns for a given allocation.

    Returns the total return for each crisis year.
    """
    weights = np.array([spy_w, gld_w, bond_w])
    returns_matrix = np.column_stack([
        daily_returns["SPY"],
        daily_returns["GLD"],
        daily_returns[bond_symbol],
    ])
    n = len(returns_matrix)

    # Rebuild equity curve (same logic as run_single_backtest)
    equity_curve = [1.0]
    current_weights = weights.copy()
    port_value = 1.0

    for i in range(n):
        port_ret = np.dot(current_weights, returns_matrix[i])
        port_value *= (1.0 + port_ret)
        equity_curve.append(port_value)
        current_weights = current_weights * (1.0 + returns_matrix[i])
        current_weights /= current_weights.sum()
        if (i + 1) % ANNUAL_DAYS == 0:
            current_weights = weights.copy()

    # Compute crisis returns using the equity curve
    crisis = compute_crisis_returns(
        prices={},  # Not used when equity_curve is provided
        trading_days=dates,
        crisis_years=CRISIS_YEARS,
        equity_curve=equity_curve,
    )
    return crisis


def generate_allocations() -> List[Tuple[str, str, float, float, float]]:
    """Generate all allocation configurations to test.

    Returns:
        List of (label, bond_symbol, spy_w, gld_w, bond_w) tuples.
    """
    configs: List[Tuple[str, str, float, float, float]] = []

    for gld in range(GLD_MIN, GLD_MAX + 1, GLD_STEP):
        gld_w = gld / 100.0
        spymin = 100 - gld - 20
        spymax = 100 - gld
        for spy in range(spymin, spymax + 1, SPY_STEP):
            spy_w = spy / 100.0
            bond_w = 1.0 - spy_w - gld_w
            if bond_w < 0:
                continue

            # SPY/GLD/TLT variant
            label_tlt = f"SPY/GLD/TLT {spy}/{gld}/{int(round(bond_w * 100))}"
            configs.append((label_tlt, "TLT", spy_w, gld_w, bond_w))

            # SPY/GLD/IEF variant
            label_ief = f"SPY/GLD/IEF {spy}/{gld}/{int(round(bond_w * 100))}"
            configs.append((label_ief, "IEF", spy_w, gld_w, bond_w))

    return configs


def main():
    configure_logging()
    logger.info("=" * 70)
    logger.info("GOLD ALLOCATION SENSITIVITY SWEEP — 2005 to 2026")
    logger.info("=" * 70)

    t0 = time.time()

    # Load data
    prices, dates = load_prices(PRICES_PATH)
    daily_returns = compute_daily_returns(prices)
    trading_dates = dates[1:]  # dates[0] has no preceding day for returns

    logger.info("Daily returns computed: %d trading days", len(trading_dates))

    # Generate all allocation configs
    configs = generate_allocations()
    logger.info("Testing %d allocation configs", len(configs))

    # Run backtests
    results = []
    for idx, (label, bond_sym, spy_w, gld_w, bond_w) in enumerate(configs):
        if (idx + 1) % 50 == 0:
            logger.info("Progress: %d / %d configs", idx + 1, len(configs))

        metrics = run_single_backtest(daily_returns, trading_dates, spy_w, gld_w, bond_w, bond_sym)
        crisis = compute_crisis_returns_for_config(daily_returns, trading_dates, spy_w, gld_w, bond_w, bond_sym)

        results.append({
            "label": label,
            "bond_symbol": bond_sym,
            "weights": {
                "SPY": round(spy_w, 4),
                "GLD": round(gld_w, 4),
                bond_sym: round(bond_w, 4),
            },
            "sharpe": metrics.sharpe_ratio,
            "cagr": metrics.cagr,
            "volatility": metrics.volatility,
            "max_drawdown": metrics.max_drawdown,
            "crisis_2008": crisis.get("2008", 0),
            "crisis_2020": crisis.get("2020", 0),
            "crisis_2022": crisis.get("2022", 0),
        })

    elapsed = time.time() - t0
    logger.info("All backtests completed in %.1f seconds", elapsed)

    # Sort by Sharpe descending
    results.sort(key=lambda r: r["sharpe"], reverse=True)

    # Print summary table
    print()
    print(f"{'Rank':<5} {'Label':<38} {'Sharpe':<8} {'CAGR%':<8} {'Vol%':<8} {'MaxDD%':<8} {'2008%':<8} {'2020%':<8} {'2022%':<8}")
    print("-" * 95)
    for rank, r in enumerate(results, 1):
        print(
            f"{rank:<5} {r['label']:<38} {r['sharpe']:<8.4f} {r['cagr']:<7.2f}% "
            f"{r['volatility']:<7.2f}% {r['max_drawdown']:<7.2f}% "
            f"{r['crisis_2008']:<7.2f}% {r['crisis_2020']:<7.2f}% {r['crisis_2022']:<7.2f}%"
        )

    # Print champion comparison
    print()
    print("=" * 70)
    print("CHAMPION COMPARISON")
    print("=" * 70)
    champion_label = "SPY/GLD/TLT 46/38/16"
    champion_found = [r for r in results if r["label"] == champion_label]
    if champion_found:
        champ = champion_found[0]
        champ_rank = next(i + 1 for i, r in enumerate(results) if r["label"] == champion_label)
        print(f"Champion {champion_label}: Rank {champ_rank}, Sharpe {champ['sharpe']:.4f}")

    # Print top 20
    print()
    print("TOP 20 CONFIGURATIONS")
    print("-" * 95)
    print(f"{'Rank':<5} {'Label':<38} {'Sharpe':<8} {'CAGR%':<8} {'Vol%':<8} {'MaxDD%':<8} {'2008%':<8} {'2020%':<8} {'2022%':<8}")
    print("-" * 95)
    for rank, r in enumerate(results[:20], 1):
        print(
            f"{rank:<5} {r['label']:<38} {r['sharpe']:<8.4f} {r['cagr']:<7.2f}% "
            f"{r['volatility']:<7.2f}% {r['max_drawdown']:<7.2f}% "
            f"{r['crisis_2008']:<7.2f}% {r['crisis_2020']:<7.2f}% {r['crisis_2022']:<7.2f}%"
        )

    # Save results
    output = {
        "metadata": {
            "description": "Gold Allocation Sensitivity Sweep — 2005 to 2026",
            "gld_range": f"{GLD_MIN}-{GLD_MAX}% step {GLD_STEP}%",
            "spy_step": f"{SPY_STEP}%",
            "bond_variants": ["TLT", "IEF"],
            "total_configs": len(results),
            "elapsed_seconds": round(elapsed, 1),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "champion": champion_found[0] if champion_found else None,
        "top_20": results[:20],
        "all_results": results,
    }

    save_results_json(output, output_path=str(OUTPUT_PATH))
    logger.info("Results saved to %s", OUTPUT_PATH)

    # Analysis summary
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Best by bond variant
    best_tlt = next(r for r in results if r["bond_symbol"] == "TLT")
    best_ief = next(r for r in results if r["bond_symbol"] == "IEF")
    print(f"Best TLT variant: {best_tlt['label']} — Sharpe {best_tlt['sharpe']:.4f}")
    print(f"Best IEF variant: {best_ief['label']} — Sharpe {best_ief['sharpe']:.4f}")

    # All results with 38% GLD
    gld38_results = [r for r in results if abs(r["weights"]["GLD"] - 0.38) < 0.01]
    if gld38_results:
        avg_sharpe_38 = np.mean([r["sharpe"] for r in gld38_results])
        print(f"\n38% GLD configs: {len(gld38_results)} found, avg Sharpe {avg_sharpe_38:.4f}")

    # All results with 18-25% GLD
    gld_low = [r for r in results if 0.18 <= r["weights"]["GLD"] <= 0.25]
    if gld_low:
        avg_sharpe_low = np.mean([r["sharpe"] for r in gld_low])
        print(f"18-25% GLD configs: {len(gld_low)} found, avg Sharpe {avg_sharpe_low:.4f}")

    print()


if __name__ == "__main__":
    main()
