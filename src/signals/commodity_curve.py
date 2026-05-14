#!/usr/bin/env python3
"""
Portfolio-Lab v3.20: Commodity Curve Overlay

Uses futures curve shape (contango/backwardation) to gate commodity ETF
allocation. Re-evaluates the v2.80 DBC rejection with curve-aware entry/exit.

Research: contango → -12% annualized, backwardation → +8% annualized.
Gate: allow DBC only during backwardation, reduce to 0% during contango.

Usage:
    python -m src.signals.commodity_curve fetch --ticker DBC
    python -m src.signals.commodity_curve regime --ticker DBC
    python -m src.signals.commodity_curve status
"""

import numpy as np
import pandas as pd
from enum import IntEnum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
from pathlib import Path
import json
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
COMMODITY_ETFS = {
    "DBC": {"name": "Invesco DB Commodity Index", "type": "broad"},
    "GSG": {"name": "iShares S&P GSCI Commodity", "type": "broad"},
    "USO": {"name": "United States Oil Fund", "type": "energy"},
    "UNG": {"name": "United States Natural Gas", "type": "energy"},
    "DBA": {"name": "Invesco DB Agriculture", "type": "agriculture"},
}

# Futures month codes for nearby contracts
FUTURES_MONTH_CODES = "FGHJKMNQUVXZ"

# Curve classification thresholds
CONTANGO_THRESHOLD = -1.0   # spread < -1% = contango
BACKWARDATION_THRESHOLD = 0.5  # spread > 0.5% = backwardation
# Between these = flat

# ETF ticker → Yahoo Finance futures proxy ticker (if direct futures unavailable)
ETF_FUTURES_PROXY = {
    "DBC": "DBC",  # DBC tracks its own index, use spot price + deferred proxy
    "GSG": "GSG",
    "USO": "USO",
    "UNG": "UNG",
    "DBA": "DBA",
}

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
PRICES_PATH = Path("~/projects/portfolio-lab/public/data/prices.json").expanduser()


class CurveRegime(IntEnum):
    CONTANGO = -1
    FLAT = 0
    BACKWARDATION = 1


@dataclass
class CommodityCurveSignal:
    ticker: str
    front_month_price: float
    deferred_month_price: float
    regime: CurveRegime
    spread_pct: float
    timestamp: datetime


def compute_curve_spread(
    front_price: float, deferred_price: float
) -> Tuple[CurveRegime, float]:
    """Compute futures curve spread and classify regime.

    Args:
        front_price: Front-month (nearby) futures/ETF price
        deferred_price: Deferred-month (next) futures/ETF price

    Returns:
        (CurveRegime, spread_pct) where spread_pct = (front - deferred) / front * 100
    """
    if front_price == 0:
        return CurveRegime.FLAT, 0.0

    spread_pct = (front_price - deferred_price) / abs(front_price) * 100

    if spread_pct > BACKWARDATION_THRESHOLD:
        regime = CurveRegime.BACKWARDATION
    elif spread_pct < CONTANGO_THRESHOLD:
        regime = CurveRegime.CONTANGO
    else:
        regime = CurveRegime.FLAT

    return regime, round(spread_pct, 4)


def fetch_curve_signal(
    ticker: str,
    prices_path: Optional[str] = None,
    deferred_offset_days: int = 21
) -> CommodityCurveSignal:
    """Fetch front-month vs deferred-month price spread from prices.json.

    Uses spot ETF price as front-month proxy and price from ~1 month ago
    as deferred-month proxy (since commodity ETFs roll monthly, the price
    difference approximates the roll yield direction).

    Args:
        ticker: ETF ticker (DBC, GSG, USO, etc.)
        prices_path: Path to prices.json (default: project public/data/prices.json)
        deferred_offset_days: Lookback days for deferred-month proxy (default: 21)

    Returns:
        CommodityCurveSignal with regime and spread

    Raises:
        ValueError: If ticker not found or insufficient data
    """
    if prices_path is None:
        prices_path = str(PRICES_PATH)

    path = Path(prices_path).expanduser()
    if not path.exists():
        raise ValueError(f"No price data at {prices_path}")

    with open(path) as f:
        data = json.load(f)

    if ticker not in data:
        raise ValueError(f"No price data for ticker: {ticker}")

    prices = data[ticker]
    if len(prices) < 2:
        raise ValueError(f"Insufficient data for {ticker}: need >=2 data points")

    # Sort by date ascending
    sorted_prices = sorted(prices, key=lambda x: x["d"])

    # Front-month = most recent close (field 'p' = adjusted close)
    front_price = sorted_prices[-1]["p"]

    # Deferred-month proxy = close from ~1 month ago
    target_date = sorted_prices[-1]["d"]
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    cutoff_dt = target_dt - timedelta(days=deferred_offset_days)
    cutoff_str = cutoff_dt.strftime("%Y-%m-%d")

    deferred_candidates = [
        p for p in sorted_prices if p["d"] <= cutoff_str
    ]

    if not deferred_candidates:
        # Fall back to oldest available price
        deferred_price = sorted_prices[0]["p"]
    else:
        deferred_price = deferred_candidates[-1]["p"]

    regime, spread = compute_curve_spread(front_price, deferred_price)

    return CommodityCurveSignal(
        ticker=ticker,
        front_month_price=front_price,
        deferred_month_price=deferred_price,
        regime=regime,
        spread_pct=spread,
        timestamp=datetime.now()
    )


def fetch_all_curves(
    prices_path: Optional[str] = None,
    tickers: Optional[list] = None
) -> Dict[str, CommodityCurveSignal]:
    """Fetch curve signals for multiple commodity ETFs.

    Args:
        prices_path: Path to prices.json
        tickers: List of tickers to fetch (default: all COMMODITY_ETFS keys)

    Returns:
        Dict[ticker, CommodityCurveSignal]
    """
    if tickers is None:
        tickers = list(COMMODITY_ETFS.keys())

    results = {}
    for ticker in tickers:
        try:
            signal = fetch_curve_signal(ticker, prices_path=prices_path)
            results[ticker] = signal
        except (ValueError, KeyError) as e:
            logger.warning(f"Skipping {ticker}: {e}")

    return results


def get_curve_summary(
    signals: Dict[str, CommodityCurveSignal]
) -> dict:
    """Summarize curve regimes across commodities.

    Returns:
        Dict with total, backwardation, contango, flat counts, and
        allocation_allowed flag (True if any commodity in backwardation).
    """
    summary = {
        "total": len(signals),
        "backwardation": sum(1 for s in signals.values() if s.regime == CurveRegime.BACKWARDATION),
        "contango": sum(1 for s in signals.values() if s.regime == CurveRegime.CONTANGO),
        "flat": sum(1 for s in signals.values() if s.regime == CurveRegime.FLAT),
    }
    summary["allocation_allowed"] = summary["backwardation"] > 0
    return summary


def get_commodity_allocation(
    signal: Optional[CommodityCurveSignal],
    base_weight: float = 5.0
) -> float:
    """Compute curve-gated commodity allocation.

    Args:
        signal: Curve signal for the commodity ETF (None = no allocation)
        base_weight: Maximum allocation weight when fully allowed

    Returns:
        Allocation weight (0.0 to base_weight)
    """
    if signal is None:
        return 0.0

    if signal.regime == CurveRegime.BACKWARDATION:
        return base_weight
    elif signal.regime == CurveRegime.CONTANGO:
        return 0.0
    else:
        # Flat: allocate half weight as neutral stance
        return base_weight * 0.5


def compute_commodity_allocation(
    signals: Dict[str, CommodityCurveSignal],
    max_weight: float = 5.0
) -> dict:
    """Compute portfolio-level commodity allocation from curve signals.

    Uses DBC as primary broad commodity proxy. If DBC is in backwardation,
    allocate up to max_weight. During contango, zero out commodities entirely.

    Args:
        signals: Dict of curve signals from fetch_all_curves()
        max_weight: Maximum commodity allocation (default: 5%)

    Returns:
        Dict with dbc_weight, allocation_allowed, summary
    """
    summary = get_curve_summary(signals)

    # Primary signal: DBC as broad commodity proxy
    dbc_signal = signals.get("DBC")
    dbc_weight = get_commodity_allocation(dbc_signal, base_weight=max_weight)

    return {
        "dbc_weight": dbc_weight,
        "allocation_allowed": summary["allocation_allowed"],
        "regime_summary": summary,
        "signals": {
            ticker: {
                "regime": s.regime.name,
                "spread_pct": s.spread_pct
            }
            for ticker, s in signals.items()
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Commodity Curve Overlay v3.20"
    )
    sub = parser.add_subparsers(dest="command")

    fetch_p = sub.add_parser("fetch", help="Fetch curve signal for a ticker")
    fetch_p.add_argument("--ticker", default="DBC", help="ETF ticker")
    fetch_p.add_argument("--prices", help="Path to prices.json")

    status_p = sub.add_parser("status", help="Show all commodity curve regimes")
    status_p.add_argument("--prices", help="Path to prices.json")

    regime_p = sub.add_parser("regime", help="Show regime classification only")
    regime_p.add_argument("--ticker", default="DBC", help="ETF ticker")
    regime_p.add_argument("--prices", help="Path to prices.json")

    args = parser.parse_args()

    if args.command == "fetch":
        signal = fetch_curve_signal(args.ticker, prices_path=args.prices)
        print(f"{signal.ticker}: regime={signal.regime.name}, "
              f"spread={signal.spread_pct:+.2f}%, "
              f"front={signal.front_month_price:.2f}, "
              f"deferred={signal.deferred_month_price:.2f}")

    elif args.command == "regime":
        signal = fetch_curve_signal(args.ticker, prices_path=args.prices)
        print(f"{signal.ticker}: {signal.regime.name} ({signal.spread_pct:+.2f}%)")

    elif args.command == "status":
        results = fetch_all_curves(prices_path=args.prices)
        alloc = compute_commodity_allocation(results)
        print("=== Commodity Curve Status (v3.20) ===")
        print(f"Allocation allowed: {alloc['allocation_allowed']}")
        print(f"DBC weight: {alloc['dbc_weight']:.1f}%")
        print(f"Regimes: {alloc['regime_summary']}")
        for ticker, signal in results.items():
            print(f"  {ticker}: {signal.regime.name} ({signal.spread_pct:+.2f}%)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
