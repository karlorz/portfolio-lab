#!/usr/bin/env python3
"""
Portfolio-Lab v9.00: Rebalance Health Data Exporter

Generates rebalance_health.json for the RebalanceHealthPanel React component.
Reads order history files and SmartRebalanceData to produce execution timeline.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

DATA_DIR = Path.home() / "projects" / "portfolio-lab" / "data"
ORDERS_DIR = DATA_DIR / "historical_orders"
OUTPUT_PATH = DATA_DIR / "rebalance_health.json"


def _parse_order_file(path: Path) -> dict[str, Any] | None:
    """Parse an order history file (YAML frontmatter + JSON body)."""
    try:
        text = path.read_text()
        # Split frontmatter from body
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                body = parts[2].strip()
            else:
                body = text
        else:
            body = text

        orders = json.loads(body)
        if not isinstance(orders, list) or len(orders) == 0:
            return None

        # Extract timestamp from filename: order_history_YYYYMMDD_HHMMSS_*
        stem = path.stem  # order_history_20260511_143008_bf55ba1899f9a2ef
        ts_parts = stem.split("_")
        if len(ts_parts) >= 3:
            date_str = ts_parts[2]
            time_str = ts_parts[3] if len(ts_parts) > 3 else "000000"
            try:
                ts = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            except ValueError:
                ts = datetime.fromtimestamp(path.stat().st_mtime)
        else:
            ts = datetime.fromtimestamp(path.stat().st_mtime)

        total_value = sum(o.get("estimated_value", 0) for o in orders)
        buy_count = sum(1 for o in orders if o.get("side") == "buy")
        sell_count = sum(1 for o in orders if o.get("side") == "sell")
        symbols = sorted(set(o.get("symbol", "?") for o in orders))

        return {
            "timestamp": ts.isoformat(),
            "date": ts.strftime("%Y-%m-%d"),
            "time": ts.strftime("%H:%M"),
            "orders": len(orders),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_value": round(total_value, 2),
            "symbols": symbols,
            "reasons": sorted(set(o.get("reason", "rebalance") for o in orders)),
        }
    except Exception:
        return None


def generate() -> dict[str, Any]:
    """Generate rebalance health data."""
    # Parse order history
    history = []
    if ORDERS_DIR.exists():
        for f in sorted(ORDERS_DIR.glob("order_history_*.json")):
            entry = _parse_order_file(f)
            if entry:
                history.append(entry)

    # Sort by timestamp
    history.sort(key=lambda e: e["timestamp"])

    # Determine next scheduled rebalance
    # Default: monthly on the 1st, or 30 days after last rebalance
    last_ts = None
    if history:
        last_ts = datetime.fromisoformat(history[-1]["timestamp"])
    else:
        last_ts = datetime.now() - timedelta(days=30)

    next_rebalance = last_ts + timedelta(days=30)

    # Schedule compliance
    on_time = 0
    delayed = 0
    for i, entry in enumerate(history):
        if i == 0:
            continue
        prev_ts = datetime.fromisoformat(history[i - 1]["timestamp"])
        curr_ts = datetime.fromisoformat(entry["timestamp"])
        delta_days = (curr_ts - prev_ts).days
        if 25 <= delta_days <= 35:
            on_time += 1
        else:
            delayed += 1

    # Recent executions (last 10)
    recent = history[-10:] if len(history) > 10 else history
    execution_times = [
        {"date": e["date"], "time": e["time"], "orders": e["orders"],
         "total_value": e["total_value"], "symbols": e["symbols"]}
        for e in reversed(recent)  # Most recent first
    ]

    return {
        "generated": datetime.now().isoformat(),
        "next_rebalance": {
            "date": next_rebalance.strftime("%Y-%m-%d"),
            "days_until": (next_rebalance - datetime.now()).days,
            "frequency": "monthly (~30 days)",
        },
        "schedule_compliance": {
            "on_time": on_time,
            "delayed": delayed,
            "total": on_time + delayed,
            "compliance_pct": round(
                on_time / max(on_time + delayed, 1) * 100, 1
            ),
        },
        "execution_history": execution_times,
        "total_executions": len(history),
    }


def main():
    data = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    # Also copy to public/data/ for dashboard fetch
    public_dir = Path.home() / "projects" / "portfolio-lab" / "public" / "data"
    public_dir.mkdir(parents=True, exist_ok=True)
    with open(public_dir / "rebalance_health.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Rebalance health data exported to {OUTPUT_PATH}")
    print(f"  Executions: {data['total_executions']}")
    print(f"  Next rebalance: {data['next_rebalance']['date']} "
          f"({data['next_rebalance']['days_until']} days)")
    print(f"  Compliance: {data['schedule_compliance']['compliance_pct']}%")


if __name__ == "__main__":
    main()
