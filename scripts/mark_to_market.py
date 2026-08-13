#!/usr/bin/env python3
"""Mark portfolio positions to market using latest prices from prices.json.

Reads portfolio_paper.json (or portfolio_live.json), refreshes current_price
for each position from the latest available market data, recalculates values,
and saves the updated portfolio. Also appends a daily history entry.

Idempotent: safe to run multiple times per day — only updates the most recent
history entry for the current date.

Usage:
    python scripts/mark_to_market.py                  # paper mode (default)
    python scripts/mark_to_market.py --mode live       # live mode
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import DATA_DIR, resolve_runtime_public_data_dir  # noqa: E402  # deliberate placement (bootstrap/sys.path ordering)

INITIAL_CAPITAL = 100000  # Default from evaluator PAPER_CONFIG

_ET = ZoneInfo("America/New_York")


def us_cash_session_date(now: Optional[datetime] = None) -> str:
    """America/New_York calendar date for MTM history keys (match capture_daily_pnl)."""
    if now is None:
        now = datetime.now(tz=_ET)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo("UTC")).astimezone(_ET)
    else:
        now = now.astimezone(_ET)
    return now.strftime("%Y-%m-%d")


def get_prices_file() -> Path:
    """Resolve prices.json SSOT from live PUBLIC_DATA_DIR / operator WWW.

    Matches ``src.paths.resolve_runtime_public_data_dir`` so unset env on a
    host with tasker WWW does not silently mark from stale repo public/data.
    """
    return resolve_runtime_public_data_dir() / "prices.json"


# Back-compat name: prefer get_prices_file() / load_prices() at call sites.
PRICES_FILE = get_prices_file()


def load_prices(
    prices_file: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """Load latest closing prices from the operator prices SSOT.

    Args:
        prices_file: Optional explicit path. When omitted, uses
            ``PUBLIC_DATA_DIR/prices.json`` (same contract as ``PRICES_JSON``).
    """
    path = Path(prices_file) if prices_file is not None else get_prices_file()
    with open(path) as f:
        data = json.load(f)

    prices: Dict[str, float] = {}
    for symbol, series in data.items():
        if isinstance(series, list) and series:
            last_entry = series[-1]
            prices[symbol] = last_entry.get("p", 0.0)
    return prices


def load_portfolio(mode: str = "paper") -> Optional[Dict[str, Any]]:
    """Load portfolio state from JSON file."""
    path = DATA_DIR / f"portfolio_{mode}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_portfolio(portfolio: Dict[str, Any], mode: str = "paper") -> None:
    """Save updated portfolio state."""
    path = DATA_DIR / f"portfolio_{mode}.json"
    with open(path, "w") as f:
        json.dump(portfolio, f, indent=2, default=str)


def get_latest_portfolio_date(portfolio: Dict[str, Any]) -> Optional[str]:
    """Get the date of the most recent history entry."""
    history = portfolio.get("history", [])
    for entry in reversed(history):
        date_str = entry.get("timestamp", "")
        if date_str:
            return date_str[:10]  # YYYY-MM-DD
    return None


def mark_to_market(portfolio: Dict[str, Any], prices: Dict[str, float]) -> Dict[str, Any]:
    """Update portfolio positions with current market prices."""
    positions = portfolio.get("positions", {})
    cash = portfolio.get("cash", 0)
    mode = portfolio.get("mode", "paper")
    history = portfolio.get("history", [])

    today = us_cash_session_date()
    previous_total = INITIAL_CAPITAL

    # previous_total = last history entry with date *strictly before* session date.
    # Same-day intermediate peaks must not poison the DoD baseline (Batch AO).
    for entry in reversed(history):
        ts = str(entry.get("timestamp") or entry.get("date") or "")
        entry_date = ts[:10] if len(ts) >= 10 else ""
        if entry_date and entry_date >= today:
            continue  # skip same-session / future rows
        tv = entry.get("total_value", 0)
        try:
            tv_f = float(tv)
        except (TypeError, ValueError):
            continue
        if tv_f > 0:
            previous_total = tv_f
            break

    # Update each position with current market price
    for symbol, pos in positions.items():
        market_price = prices.get(symbol)
        if market_price and market_price > 0:
            shares = pos.get("shares", 0)
            avg_price = pos.get("avg_price", 0)

            pos["current_price"] = round(market_price, 2)
            pos["value"] = round(shares * market_price, 2)
            pos["unrealized_pnl"] = round((market_price - avg_price) * shares, 2)

    # Recalculate total value
    total_value = cash + sum(p.get("value", 0) for p in positions.values())

    # Update weights
    for pos in positions.values():
        pos["weight"] = round(pos["value"] / total_value, 4) if total_value > 0 else 0

    # Compute daily return vs prior session close only
    daily_return = (total_value - previous_total) / previous_total if previous_total > 0 else 0

    # Update or append today's history entry
    today_entry = {
        "timestamp": datetime.now().isoformat(),
        "date": today,
        "total_value": round(total_value, 2),
        "cash": round(cash, 2),
        "daily_return": round(daily_return, 6),
        "positions_count": len(positions),
        "mode": mode,
        "previous_total": round(previous_total, 2),
        "session_date": today,
    }

    # Remove any existing entry for today (idempotent) — match session date key
    def _entry_date(h: Dict[str, Any]) -> str:
        if h.get("session_date"):
            return str(h["session_date"])[:10]
        if h.get("date"):
            return str(h["date"])[:10]
        ts = str(h.get("timestamp") or "")
        return ts[:10] if len(ts) >= 10 else ""

    history = [h for h in history if _entry_date(h) != today]
    history.append(today_entry)

    # Keep last 100 history entries
    history = history[-100:]

    portfolio["cash"] = cash
    portfolio["history"] = history
    portfolio["updated"] = datetime.now().isoformat()

    return portfolio


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mark portfolio to market")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"],
                        help="Portfolio mode")
    parser.add_argument(
        "--prices",
        default=None,
        help="Optional prices.json path (default: PUBLIC_DATA_DIR/prices.json)",
    )
    args = parser.parse_args()

    print(f"Mark-to-Market — {datetime.now().isoformat()}")
    print(f"  Mode: {args.mode}")
    prices_path = Path(args.prices) if args.prices else get_prices_file()
    print(f"  Prices SSOT: {prices_path}")

    # Load prices
    prices = load_prices(prices_file=prices_path)
    print(f"  Prices loaded: {len(prices)} symbols")

    # Show latest prices for key assets
    for sym in ["SPY", "GLD", "TLT"]:
        if sym in prices:
            print(f"    {sym}: ${prices[sym]:.2f}")

    # Load portfolio
    portfolio = load_portfolio(args.mode)
    if not portfolio:
        print(f"  ERROR: No portfolio_{args.mode}.json found")
        sys.exit(1)

    old_total = 0
    cash = portfolio.get("cash", 0)
    old_positions = portfolio.get("positions", {})
    old_total = cash + sum(p.get("value", 0) for p in old_positions.values())
    print(f"  Portfolio value (before): ${old_total:,.2f}")

    # Mark to market
    portfolio = mark_to_market(portfolio, prices)
    save_portfolio(portfolio, args.mode)

    # Show new totals
    new_positions = portfolio.get("positions", {})
    new_total = portfolio.get("cash", 0) + sum(p.get("value", 0) for p in new_positions.values())
    daily_return = (new_total - old_total) / old_total if old_total > 0 else 0
    print(f"  Portfolio value (after):  ${new_total:,.2f}")
    print(f"  Change:                   ${new_total - old_total:+,.2f} ({daily_return:+.4%})")

    # Show position details
    print(f"\n  Positions ({len(new_positions)}):")
    for sym, pos in new_positions.items():
        print(f"    {sym}: {pos['shares']:.4f} sh × ${pos['current_price']:.2f} = ${pos['value']:,.2f} "
              f"(avg ${pos['avg_price']:.2f}, P&L ${pos['unrealized_pnl']:+,.2f})")

    print(f"\n  Portfolio saved to portfolio_{args.mode}.json")
    print(f"  History entries: {len(portfolio.get('history', []))}")


if __name__ == "__main__":
    main()
