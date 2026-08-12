#!/usr/bin/env python3
"""
Walk-Forward Validation for Portfolio-Lab

Expanding-window walk-forward analysis using scikit-learn TimeSeriesSplit.
Validates the grid search champion (SPY/GLD/TLT 46/38/16, Sharpe 0.79)
by producing 20 out-of-sample Sharpe estimates, Walk-Forward Efficiency,
and Deflated Sharpe Ratio.

Architecture:
  - Uses scikit-learn TimeSeriesSplit (already installed, no ML gate)
  - Expanding window: train on [start, T], test on [T+gap, T+gap+test_size]
  - Reports WFE, OOS Sharpe distribution, DSR, and crisis sub-metrics
  - Reuses compute_metrics() from src/backtest/metrics.py

Usage:
    uv run python scripts/walk_forward_validation.py
    uv run python scripts/walk_forward_validation.py --save
    uv run python scripts/walk_forward_validation.py --n-splits 15 --test-size 126
"""

import json
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Ensure project root is on sys.path for src.* imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.paths import PRICES_JSON, DATA_DIR
from src.backtest.metrics import (
    BacktestMetrics, compute_metrics, compute_crisis_returns,
    compute_deflated_sharpe_ratio, DEFAULT_CRISIS_YEARS, save_results_json,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

WALK_FORWARD_SCHEMA_VERSION = "walk-forward-validation/v1"
CANONICAL_WINDOW_MODE = "expanding"
CANONICAL_WALK_FORWARD_ARTIFACT = Path("data") / "walk_forward_report.json"
METRICS_SOURCE = "src.backtest.metrics"


# ── Grid Search Configurations ────────────────────────────────────────


def _with_canonical_contract(result: dict) -> dict:
    """Attach the stable walk-forward artifact contract."""
    contracted = dict(result)
    contracted.update({
        "schema_version": WALK_FORWARD_SCHEMA_VERSION,
        "analysis_type": "walk_forward_validation",
        "window_mode": CANONICAL_WINDOW_MODE,
        "artifact_path": str(CANONICAL_WALK_FORWARD_ARTIFACT),
        "metrics_source": METRICS_SOURCE,
    })
    return contracted


def generate_grid_configs() -> list[dict[str, float]]:
    """Generate the 53 grid search configurations for SPY/GLD/TLT.

    The Python walk-forward engine is SPY/GLD/TLT-only (load_prices drops
    symbols outside that universe), so it mirrors the SPY/GLD/TLT subset of
    the TypeScript grid (src/backtest/grid-search.ts) — regions 1, 2, and 8
    — NOT the full 94-config grid. TS regions 3-7/9 (IEF, trend-following,
    quarterly rebalance, VTI/VBR, vol-target) are intentionally excluded
    (engine limitation; parity pinned by tests/test_walk_forward_grid_parity.py).

    TS source-of-truth regions mirrored here:
    - Region 1: SPY/GLD sweep (40-70% SPY, 5% step) — grid-search.ts:52-58
    - Region 2: SPY/GLD/TLT sweep (TLT 5-20%, SPY 50-65%, 5% step,
      gld 10-60) — grid-search.ts:62-71
    - Region 8: Fine sweep around champion (SPY 46-54%, TLT 10-20%, 2%
      step, gld 25-45) — grid-search.ts:129-136
    """
    configs = []

    # Region 1: SPY/GLD sweep
    for spy in range(40, 71, 5):
        gld = 100 - spy
        configs.append({"SPY": spy / 100, "GLD": gld / 100, "TLT": 0.0})

    # Region 2: SPY/GLD/TLT sweep (5% step)
    for tlt in range(5, 21, 5):
        for spy in range(50, 66, 5):
            gld = 100 - spy - tlt
            if 10 <= gld <= 60:
                configs.append({"SPY": spy / 100, "GLD": gld / 100, "TLT": tlt / 100})

    # Region 8: Fine sweep around champion (2% step)
    for spy in range(46, 55, 2):
        for tlt in range(10, 21, 2):
            gld = 100 - spy - tlt
            if 25 <= gld <= 45:
                configs.append({"SPY": spy / 100, "GLD": gld / 100, "TLT": tlt / 100})

    return configs


GRID_CONFIGS = generate_grid_configs()


# ── Price Loading ─────────────────────────────────────────────────────

def load_prices(path: Path, symbols: list[str] | None = None) -> pd.DataFrame:
    """Load prices.json into a wide DataFrame (dates x symbols)."""
    with open(path) as f:
        raw = json.load(f)

    records = []
    for symbol, bars in raw.items():
        if symbols and symbol not in symbols:
            continue
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
    pivot = pivot.sort_index()
    pivot = pivot.dropna(subset=["SPY", "GLD", "TLT"])
    return pivot


# ── Walk-Forward Engine ───────────────────────────────────────────────

def run_single_window(
    prices: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    configs: list[dict[str, float]],
) -> dict:
    """Run grid search on training period, evaluate champion on test period.

    Returns dict with IS metrics, OOS metrics, and champion weights.
    """
    train_prices = prices.iloc[train_idx]
    test_prices = prices.iloc[test_idx]

    # Compute returns for each period
    train_returns = train_prices.pct_change().dropna()
    test_returns = test_prices.pct_change().dropna()

    if len(train_returns) < 50 or len(test_returns) < 20:
        return None

    # Grid search on training period: find best Sharpe
    best_sharpe = -999
    best_config = None
    best_is_metrics = None

    for config in configs:
        # Portfolio returns = weighted sum of asset returns
        # Only use assets present in both config and prices
        weights = np.array([config.get(s, 0) for s in prices.columns])
        port_returns = train_returns.values @ weights

        # Skip if all returns are zero (degenerate config)
        if np.all(port_returns == 0):
            continue

        # Compute equity curve
        initial = 100000.0
        equity = [initial]
        for r in port_returns:
            equity.append(equity[-1] * (1 + r))

        metrics = compute_metrics(equity, initial)
        if metrics.sharpe_ratio > best_sharpe:
            best_sharpe = metrics.sharpe_ratio
            best_config = config
            best_is_metrics = metrics

    if best_config is None:
        return None

    # Evaluate champion on test period (OOS)
    weights = np.array([best_config.get(s, 0) for s in prices.columns])
    oos_returns = test_returns.values @ weights

    initial = 100000.0
    oos_equity = [initial]
    for r in oos_returns:
        oos_equity.append(oos_equity[-1] * (1 + r))

    oos_metrics = compute_metrics(oos_equity, initial)

    # Crisis sub-metrics (if test period overlaps crisis years)
    crisis_years_in_test = []
    test_dates = test_prices.index
    for year in DEFAULT_CRISIS_YEARS:
        if any(d.strftime("%Y") == year for d in test_dates):
            crisis_years_in_test.append(year)

    crisis_returns = {}
    if crisis_years_in_test:
        # Build price dict for crisis computation
        prices_dict = {}
        trading_days = []
        for date, row in test_prices.iterrows():
            date_str = date.strftime("%Y-%m-%d")
            prices_dict[date_str] = {s: row[s] for s in prices.columns if s in row}
            trading_days.append(date_str)

        crisis_returns = compute_crisis_returns(
            prices_dict, trading_days,
            crisis_years=crisis_years_in_test,
            base_weights=best_config,
            equity_curve=oos_equity[1:],  # skip initial capital
        )

    return {
        "is_sharpe": best_is_metrics.sharpe_ratio,
        "is_cagr": best_is_metrics.cagr,
        "is_max_dd": best_is_metrics.max_drawdown,
        "oos_sharpe": oos_metrics.sharpe_ratio,
        "oos_cagr": oos_metrics.cagr,
        "oos_max_dd": oos_metrics.max_drawdown,
        "oos_volatility": oos_metrics.volatility,
        "champion_weights": best_config,
        "train_start": train_prices.index[0].strftime("%Y-%m-%d"),
        "train_end": train_prices.index[-1].strftime("%Y-%m-%d"),
        "test_start": test_prices.index[0].strftime("%Y-%m-%d"),
        "test_end": test_prices.index[-1].strftime("%Y-%m-%d"),
        "train_days": len(train_returns),
        "test_days": len(test_returns),
        "crisis_returns": crisis_returns,
    }


def run_walk_forward(
    prices: pd.DataFrame,
    n_splits: int = 20,
    test_size: int = 252,
    gap: int = 21,
    configs: list[dict[str, float]] | None = None,
) -> dict:
    """Run expanding-window walk-forward validation.

    Args:
        prices: DataFrame with SPY/GLD/TLT columns, datetime index.
        n_splits: Number of walk-forward windows (default 20).
        test_size: Test period length in trading days (default 252 = 1yr).
        gap: Embargo period between train and test (default 21 = 1mo).
        configs: Grid search configs (default: 53-config SPY/GLD/TLT grid =
            TS regions 1+2+8 subset; see generate_grid_configs).

    Returns:
        Dict with per-window results, aggregate metrics, WFE, and DSR.
    """
    if configs is None:
        configs = GRID_CONFIGS

    # Create time series splits
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

    window_results = []
    all_train_idx = []
    all_test_idx = []

    for split_idx, (train_idx, test_idx) in enumerate(tscv.split(prices)):
        logger.info(
            "Window %2d/%d: train=%s to %s (%d days), test=%s to %s (%d days)",
            split_idx + 1, n_splits,
            prices.index[train_idx[0]].strftime("%Y-%m-%d"),
            prices.index[train_idx[-1]].strftime("%Y-%m-%d"),
            len(train_idx),
            prices.index[test_idx[0]].strftime("%Y-%m-%d"),
            prices.index[test_idx[-1]].strftime("%Y-%m-%d"),
            len(test_idx),
        )

        result = run_single_window(prices, train_idx, test_idx, configs)
        if result is not None:
            result["window"] = split_idx + 1
            window_results.append(result)

    if not window_results:
        return _with_canonical_contract({"error": "No valid walk-forward windows produced"})

    # Aggregate metrics
    is_sharpes = [r["is_sharpe"] for r in window_results]
    oos_sharpes = [r["oos_sharpe"] for r in window_results]

    mean_is_sharpe = np.mean(is_sharpes)
    mean_oos_sharpe = np.mean(oos_sharpes)

    # Walk-Forward Efficiency
    wfe = mean_oos_sharpe / mean_is_sharpe if mean_is_sharpe > 0 else 0.0

    # Deflated Sharpe Ratio from OOS distribution
    n_trials = len(configs)  # Use full grid search count
    n_obs = int(np.mean([r["test_days"] for r in window_results]))
    champion_oos_sharpe = max(oos_sharpes)  # Best OOS Sharpe across windows
    mean_oos_sharpe_for_dsr = mean_oos_sharpe  # Average OOS Sharpe

    dsr_champion = compute_deflated_sharpe_ratio(
        champion_oos_sharpe, n_trials=n_trials, n_observations=n_obs,
    )
    dsr_average = compute_deflated_sharpe_ratio(
        mean_oos_sharpe_for_dsr, n_trials=n_trials, n_observations=n_obs,
    )

    # Crisis sub-metrics aggregation
    crisis_by_year = {}
    for r in window_results:
        for year, ret in r.get("crisis_returns", {}).items():
            crisis_by_year.setdefault(year, []).append(ret)

    crisis_summary = {}
    for year, returns in crisis_by_year.items():
        crisis_summary[year] = {
            "mean": round(float(np.mean(returns)), 2),
            "worst": round(float(np.min(returns)), 2),
            "count": len(returns),
        }

    # Champion weight consistency
    weight_keys = ["SPY", "GLD", "TLT"]
    weight_stats = {}
    for key in weight_keys:
        vals = [r["champion_weights"].get(key, 0) for r in window_results]
        weight_stats[key] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }

    return _with_canonical_contract({
        "n_windows": len(window_results),
        "n_configs": len(configs),
        "walk_forward_efficiency": round(float(wfe), 4),
        "is_sharpe": {
            "mean": round(float(mean_is_sharpe), 4),
            "std": round(float(np.std(is_sharpes)), 4),
            "min": round(float(np.min(is_sharpes)), 4),
            "max": round(float(np.max(is_sharpes)), 4),
        },
        "oos_sharpe": {
            "mean": round(float(mean_oos_sharpe), 4),
            "std": round(float(np.std(oos_sharpes)), 4),
            "min": round(float(np.min(oos_sharpes)), 4),
            "max": round(float(np.max(oos_sharpes)), 4),
        },
        "dsr_champion_oos": round(float(dsr_champion), 4),
        "dsr_average_oos": round(float(dsr_average), 4),
        "crisis_summary": crisis_summary,
        "champion_weight_consistency": weight_stats,
        "windows": window_results,
    })


def print_report(result: dict):
    """Print a formatted walk-forward validation report."""
    print(f"\n{'='*70}")
    print("Walk-Forward Validation Report")
    print(f"{'='*70}")
    print(f"Windows: {result['n_windows']}")
    print(f"Grid configs per window: {result['n_configs']}")

    print(f"\n--- In-Sample Performance ---")
    is_s = result["is_sharpe"]
    print(f"  Sharpe: mean={is_s['mean']:.4f}, std={is_s['std']:.4f}, "
          f"range=[{is_s['min']:.4f}, {is_s['max']:.4f}]")

    print(f"\n--- Out-of-Sample Performance ---")
    oos = result["oos_sharpe"]
    print(f"  Sharpe: mean={oos['mean']:.4f}, std={oos['std']:.4f}, "
          f"range=[{oos['min']:.4f}, {oos['max']:.4f}]")

    print(f"\n--- Walk-Forward Efficiency ---")
    wfe = result["walk_forward_efficiency"]
    print(f"  WFE: {wfe:.4f}", end="")
    if wfe > 0.60:
        print(" (GOOD — genuine predictive power)")
    elif wfe > 0.40:
        print(" (BORDERLINE — mild overfitting)")
    else:
        print(" (POOR — likely overfit)")

    print(f"\n--- Deflated Sharpe Ratio ---")
    print(f"  DSR (champion OOS): {result['dsr_champion_oos']:.4f}")
    print(f"  DSR (average OOS):  {result['dsr_average_oos']:.4f}")

    if result.get("crisis_summary"):
        print(f"\n--- Crisis Period Returns ---")
        for year, stats in sorted(result["crisis_summary"].items()):
            print(f"  {year}: mean={stats['mean']:.2f}%, "
                  f"worst={stats['worst']:.2f}% ({stats['count']} windows)")

    print(f"\n--- Champion Weight Consistency ---")
    for sym, stats in result.get("champion_weight_consistency", {}).items():
        print(f"  {sym}: mean={stats['mean']:.2%}, std={stats['std']:.2%}, "
              f"range=[{stats['min']:.2%}, {stats['max']:.2%}]")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward validation for portfolio-lab grid search"
    )
    parser.add_argument("--save", action="store_true",
                        help="Save results to data/walk_forward_report.json")
    parser.add_argument("--n-splits", type=int, default=20,
                        help="Number of walk-forward windows (default: 20)")
    parser.add_argument("--test-size", type=int, default=252,
                        help="Test period in trading days (default: 252 = 1yr)")
    parser.add_argument("--gap", type=int, default=21,
                        help="Embargo period in trading days (default: 21 = 1mo)")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "GLD", "TLT"],
                        help="Symbols to include (default: SPY GLD TLT)")
    args = parser.parse_args()

    if not PRICES_JSON.exists():
        logger.error("Prices file not found: %s", PRICES_JSON)
        return

    prices = load_prices(PRICES_JSON, symbols=args.symbols)
    if prices.empty:
        logger.error("No price data loaded")
        return

    logger.info("Loaded %d trading days for %d symbols: %s",
                len(prices), len(prices.columns), list(prices.columns))
    logger.info("Date range: %s to %s",
                prices.index[0].strftime("%Y-%m-%d"),
                prices.index[-1].strftime("%Y-%m-%d"))
    logger.info("Grid configs: %d", len(GRID_CONFIGS))
    logger.info("Walk-forward: %d windows, %d-day test, %d-day gap\n",
                args.n_splits, args.test_size, args.gap)

    result = run_walk_forward(
        prices,
        n_splits=args.n_splits,
        test_size=args.test_size,
        gap=args.gap,
    )
    result = _with_canonical_contract(result)

    if "error" in result:
        logger.error("Walk-forward failed: %s", result["error"])
        return

    print_report(result)

    if args.save:
        output_path = DATA_DIR / CANONICAL_WALK_FORWARD_ARTIFACT.name
        save_results_json(
            result,
            output_path=str(output_path),
            experiment_manifest={
                "experiment_id": "walk-forward-validation",
                "manifest_mode": "sidecar",
                "module": __name__,
                "command": (
                    "python scripts/walk_forward_validation.py --save "
                    f"--n-splits {args.n_splits} --test-size {args.test_size} --gap {args.gap}"
                ),
                "config_snapshot": {
                    "n_splits": args.n_splits,
                    "test_size": args.test_size,
                    "gap": args.gap,
                    "schema_version": WALK_FORWARD_SCHEMA_VERSION,
                    "window_mode": CANONICAL_WINDOW_MODE,
                },
                "input_paths": [PRICES_JSON],
            },
        )
        logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
