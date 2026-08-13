#!/usr/bin/env python3
"""
Portfolio optimization via PyPortfolioOpt.

Loads prices.json and runs multiple optimization strategies:
- Max Sharpe (tangency portfolio)
- Min Volatility
- Efficient Risk (target 10% vol)
- Hierarchical Risk Parity (HRP)

Outputs Labs-compatible optimizer rows to data/optimized_weights.json for
comparison with the champion SPY/GLD/TLT 46/38/16 (Sharpe 0.79).

Usage:
    uv run python scripts/optimize_portfolio.py
    uv run python scripts/optimize_portfolio.py --save
    uv run python scripts/optimize_portfolio.py --symbols SPY GLD TLT
"""

import json
import sys
import argparse
import logging
from pathlib import Path

# Ensure project root is on sys.path for src.* imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from src.paths import PRICES_JSON, DATA_DIR
from src.research.optimizer_labs_contract import save_optimizer_labs_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_prices(path_or_data: Path | dict, symbols: list[str] | None = None) -> pd.DataFrame:
    """Load prices.json into a wide DataFrame (dates x symbols).

    Args:
        path_or_data: Either a Path to prices.json, or a pre-loaded dict
            (e.g., from src.data.price_cache.get_prices()).
        symbols: Optional symbol filter.
    """
    if isinstance(path_or_data, dict):
        raw = path_or_data
    else:
        with open(path_or_data) as f:
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
    # Drop rows with any NaN (incomplete cross-section)
    pivot = pivot.dropna()
    return pivot


def run_max_sharpe(prices: pd.DataFrame) -> dict:
    """Find the tangency portfolio (maximum Sharpe ratio)."""
    from pypfopt import EfficientFrontier, risk_models, expected_returns

    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S)

    try:
        _ = ef.max_sharpe()
        cleaned = ef.clean_weights()
        perf = ef.portfolio_performance()
        return {
            "weights": {k: round(v, 4) for k, v in cleaned.items() if v > 0.001},
            "sharpe": round(perf[2], 4),
            "cagr": round(perf[0] * 100, 2),
            "volatility": round(perf[1] * 100, 2),
        }
    except Exception as e:
        logger.warning("Max Sharpe optimization failed: %s", e)
        return {"error": str(e)}


def run_min_volatility(prices: pd.DataFrame) -> dict:
    """Find the minimum volatility portfolio."""
    from pypfopt import EfficientFrontier, risk_models, expected_returns

    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S)

    try:
        _ = ef.min_volatility()
        cleaned = ef.clean_weights()
        perf = ef.portfolio_performance()
        return {
            "weights": {k: round(v, 4) for k, v in cleaned.items() if v > 0.001},
            "sharpe": round(perf[2], 4),
            "cagr": round(perf[0] * 100, 2) if perf[0] is not None else None,
            "volatility": round(perf[1] * 100, 2),
        }
    except Exception as e:
        logger.warning("Min Volatility optimization failed: %s", e)
        return {"error": str(e)}


def run_efficient_risk(prices: pd.DataFrame, target_vol: float = 0.10) -> dict:
    """Find the max-return portfolio for a given target volatility."""
    from pypfopt import EfficientFrontier, risk_models, expected_returns

    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S)

    try:
        _ = ef.efficient_risk(target_vol)
        cleaned = ef.clean_weights()
        perf = ef.portfolio_performance()
        return {
            "weights": {k: round(v, 4) for k, v in cleaned.items() if v > 0.001},
            "sharpe": round(perf[2], 4),
            "cagr": round(perf[0] * 100, 2),
            "volatility": round(perf[1] * 100, 2),
            "target_vol": target_vol,
        }
    except Exception as e:
        logger.warning("Efficient Risk optimization failed: %s", e)
        return {"error": str(e)}


def run_hrp(prices: pd.DataFrame) -> dict:
    """Hierarchical Risk Parity (De Prado 2016)."""
    from pypfopt import HRPOpt

    returns = prices.pct_change().dropna()
    hrp = HRPOpt(returns)

    try:
        _ = hrp.optimize()
        cleaned = hrp.clean_weights()
        perf = hrp.portfolio_performance()
        return {
            "weights": {k: round(v, 4) for k, v in cleaned.items() if v > 0.001},
            "sharpe": round(perf[2], 4),
            "cagr": round(perf[0] * 100, 2),
            "volatility": round(perf[1] * 100, 2),
        }
    except Exception as e:
        logger.warning("HRP optimization failed: %s", e)
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Portfolio optimization via PyPortfolioOpt")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to optimize (default: all in prices.json)")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument("--target-vol", type=float, default=0.10,
                        help="Target volatility for efficient_risk (default: 0.10)")
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

    results = {}

    # Max Sharpe
    logger.info("Running Max Sharpe optimization...")
    results["max_sharpe"] = run_max_sharpe(prices)

    # Min Volatility
    logger.info("Running Min Volatility optimization...")
    results["min_volatility"] = run_min_volatility(prices)

    # Efficient Risk
    logger.info("Running Efficient Risk (target vol=%.0f%%)...", args.target_vol * 100)
    results["efficient_risk"] = run_efficient_risk(prices, target_vol=args.target_vol)

    # HRP
    logger.info("Running HRP optimization...")
    results["hrp"] = run_hrp(prices)

    # Champion reference
    results["champion"] = {
        "weights": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "sharpe": 0.79,
        "cagr": 10.6,
        "volatility": 11.1,
        "note": "Grid search champion (2005-2026)",
    }

    # Print results
    print(f"\n{'='*70}")
    print("Portfolio Optimization Results (PyPortfolioOpt)")
    print(f"{'='*70}")
    print(f"Period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Symbols: {list(prices.columns)}")

    for name, result in results.items():
        print(f"\n--- {name.upper()} ---")
        if "error" in result:
            print(f"  Error: {result['error']}")
            continue
        print(f"  Weights: {result.get('weights', {})}")
        print(f"  CAGR:    {result.get('cagr', 'N/A')}%")
        print(f"  Vol:     {result.get('volatility', 'N/A')}%")
        print(f"  Sharpe:  {result.get('sharpe', 'N/A')}")

    if args.save:
        output_path = DATA_DIR / "optimized_weights.json"
        save_optimizer_labs_output(
            results,
            output_path=output_path,
            symbols=list(prices.columns),
            target_vol=args.target_vol,
        )
        logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
