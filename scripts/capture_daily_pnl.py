#!/usr/bin/env python3
"""Capture daily P&L snapshot from portfolio state.

Reads portfolio_paper.json (or portfolio_live.json), computes daily P&L
metrics, and appends a snapshot to data/daily_pnl.jsonl. Also writes
the latest snapshot to data/daily_pnl_latest.json for dashboard consumption.

Designed as a cron job — idempotent within a single day (won't duplicate
entries for the same date if run multiple times).

Usage:
    python scripts/capture_daily_pnl.py
    python scripts/capture_daily_pnl.py --mode live
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import DATA_DIR


def load_portfolio(mode: str = "paper") -> Optional[Dict[str, Any]]:
    """Load portfolio state from JSON file."""
    path = DATA_DIR / f"portfolio_{mode}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compute_pnl_snapshot(portfolio: Dict[str, Any]) -> Dict[str, Any]:
    """Compute P&L snapshot from portfolio state."""
    positions = portfolio.get("positions", {})
    cash = portfolio.get("cash", 0)
    mode = portfolio.get("mode", "paper")

    total_value = cash + sum(p.get("value", 0) for p in positions.values())

    # Position-level P&L
    position_pnl = {}
    for sym, p in positions.items():
        shares = p.get("shares", 0)
        avg_price = p.get("avg_price", 0)
        current_price = p.get("current_price", 0)
        value = p.get("value", 0)
        unrealized = p.get("unrealized_pnl", 0)
        weight = value / total_value if total_value > 0 else 0
        position_pnl[sym] = {
            "shares": round(shares, 4),
            "avg_price": round(avg_price, 2),
            "current_price": round(current_price, 2),
            "value": round(value, 2),
            "weight": round(weight, 4),
            "unrealized_pnl": round(unrealized, 2),
        }

    # Compute daily return from portfolio history
    history = portfolio.get("history", [])
    daily_return = 0.0
    if history:
        last = history[-1]
        daily_return = last.get("daily_return", 0.0)

    # Compute cumulative P&L
    initial_capital = 100000  # Default from PAPER_CONFIG
    total_pnl = total_value - initial_capital
    total_pnl_pct = total_pnl / initial_capital if initial_capital > 0 else 0

    # Drawdown calculation
    max_value = initial_capital
    for h in history:
        v = h.get("total_value", 0)
        if v > max_value:
            max_value = v
    drawdown = (total_value - max_value) / max_value if max_value > 0 else 0

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "total_value": round(total_value, 2),
        "cash": round(cash, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 6),
        "daily_return": round(daily_return, 6),
        "drawdown": round(drawdown, 6),
        "positions_count": len(positions),
        "positions": position_pnl,
    }


def save_snapshot(snapshot: Dict[str, Any], append_path: Path, latest_path: Path) -> bool:
    """Save P&L snapshot. Idempotent within same date — replaces if duplicate."""
    date = snapshot["date"]

    # Read existing entries, remove any from same date (idempotent)
    existing = []
    if append_path.exists():
        with open(append_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        if entry.get("date") != date:
                            existing.append(line)
                    except json.JSONDecodeError:
                        existing.append(line)

    # Append new snapshot
    existing.append(json.dumps(snapshot, default=str))

    with open(append_path, 'w') as f:
        f.write('\n'.join(existing) + '\n')

    # Write latest snapshot
    with open(latest_path, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Capture daily P&L snapshot")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"],
                        help="Portfolio mode to capture")
    args = parser.parse_args()

    print(f"Daily P&L Capture — {datetime.now().isoformat()}")

    portfolio = load_portfolio(args.mode)
    if not portfolio:
        print(f"  ERROR: No portfolio_{args.mode}.json found")
        sys.exit(1)

    snapshot = compute_pnl_snapshot(portfolio)

    append_path = DATA_DIR / "daily_pnl.jsonl"
    latest_path = DATA_DIR / "daily_pnl_latest.json"

    save_snapshot(snapshot, append_path, latest_path)

    print(f"  Date:          {snapshot['date']}")
    print(f"  Total Value:   ${snapshot['total_value']:,.2f}")
    print(f"  Total P&L:     ${snapshot['total_pnl']:,.2f} ({snapshot['total_pnl_pct']:.2%})")
    print(f"  Daily Return:  {snapshot['daily_return']:.4%}")
    print(f"  Drawdown:      {snapshot['drawdown']:.2%}")
    print(f"  Positions:     {snapshot['positions_count']}")
    print(f"  Saved to:      {append_path}")


if __name__ == "__main__":
    main()
