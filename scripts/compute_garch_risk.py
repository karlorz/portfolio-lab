#!/usr/bin/env python3
"""Compute GARCH-CVaR risk metrics and write .health_report.json.

Standalone script for cron-based GARCH-CVaR pipeline activation.
Reads portfolio returns from market.db, computes GARCH-filtered CVaR,
and writes results to data/.health_report.json for dashboard consumption.

Usage:
    python scripts/compute_garch_risk.py
    python scripts/compute_garch_risk.py --window 504
"""
import json
import sys
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import MARKET_DB, DATA_DIR, BASE_ALLOCATION
from src.monitor.garch_cvar import calculate_garch_cvar, ARCH_AVAILABLE


def compute_portfolio_returns(db_path: Path, days: int = 504) -> np.ndarray:
    """Load SPY/GLD/TLT daily returns from market.db."""
    if not db_path.exists():
        return np.array([])

    weights = BASE_ALLOCATION
    conn = sqlite3.connect(str(db_path))

    # Get common trading dates
    symbols = list(weights.keys())
    all_dates = set()
    symbol_prices = {}

    for sym in symbols:
        cursor = conn.execute(
            "SELECT date, close FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT ?",
            (sym, days),
        )
        rows = cursor.fetchall()
        if rows:
            symbol_prices[sym] = {r[0]: r[1] for r in rows}
            if not all_dates:
                all_dates = set(symbol_prices[sym].keys())
            else:
                all_dates &= set(symbol_prices[sym].keys())

    conn.close()

    if not all_dates or not symbol_prices:
        return np.array([])

    sorted_dates = sorted(all_dates)

    # Compute weighted portfolio returns
    portfolio_returns = []
    prev_prices = None
    for date in sorted_dates:
        current_prices = {}
        for sym in symbols:
            if sym in symbol_prices and date in symbol_prices[sym]:
                current_prices[sym] = symbol_prices[sym][date]

        if len(current_prices) != len(symbols) or prev_prices is None:
            prev_prices = current_prices
            continue

        daily_return = 0.0
        for sym in symbols:
            if sym in current_prices and sym in prev_prices and prev_prices[sym] > 0:
                daily_return += weights[sym] * (current_prices[sym] / prev_prices[sym] - 1)

        portfolio_returns.append(daily_return)
        prev_prices = current_prices

    return np.array(portfolio_returns)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute GARCH-CVaR risk metrics")
    parser.add_argument("--window", type=int, default=252, help="GARCH lookback window")
    parser.add_argument("--days", type=int, default=504, help="Days of price data to load")
    args = parser.parse_args()

    print(f"GARCH-CVaR Risk Computation — {datetime.now().isoformat()}")
    print(f"  arch library: {'available' if ARCH_AVAILABLE else 'NOT AVAILABLE (will use historical fallback)'}")

    # Load returns
    returns = compute_portfolio_returns(MARKET_DB, days=args.days)
    print(f"  Loaded {len(returns)} portfolio daily returns")

    if len(returns) < 63:
        print("  ERROR: Insufficient data (need at least 63 days)")
        sys.exit(1)

    # Compute GARCH-CVaR
    metrics = calculate_garch_cvar(
        returns=returns,
        current_drawdown=0.0,
        max_drawdown=-0.15,
        window=args.window,
    )

    # Write health report
    report_path = DATA_DIR / ".health_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    from dataclasses import asdict
    import math
    report = asdict(metrics)

    # Add portfolio entropy metrics
    weights = list(BASE_ALLOCATION.values())
    n = len(weights)
    shannon = -sum(w * math.log(w) for w in weights if w > 0)
    effective_n = math.exp(shannon)
    h_max = math.log(n) if n > 1 else 1.0
    normalized_score = (shannon / h_max) * 100.0 if h_max > 0 else 0.0
    hhi = sum(w * w for w in weights)

    report["checks"] = {
        "portfolio_entropy": {
            "name": "portfolio_entropy",
            "status": "good" if normalized_score > 90 else "warning",
            "ok": normalized_score > 70,
            "metrics": {
                "shannon_entropy": round(shannon, 4),
                "effective_n": round(effective_n, 2),
                "normalized_score": round(normalized_score, 1),
                "hhi_index": round(hhi, 4),
            },
        },
    }

    # Derive top-level status for unified dashboard compatibility
    tail = report.get("tail_severity", "normal")
    cvar_ratio = report.get("cvar_ratio", 1.0)
    if tail in ("extreme", "severe") or cvar_ratio > 3.0:
        report["status"] = "unhealthy"
    else:
        report["status"] = "healthy"
    report["summary"] = {"passed": 1, "total_checks": 1}

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Dual-write risk_metrics.json for unified / cvar_metrics consumers
    # (field names match historical risk_metrics schema).
    risk_metrics_path = DATA_DIR / "risk_metrics.json"
    risk_payload = {
        "timestamp": report.get("timestamp") or datetime.now().isoformat(),
        "var_95_daily": report.get("var_95"),
        "cvar_95_daily": report.get("cvar_95"),
        "cvar_ratio": report.get("cvar_ratio"),
        "tail_severity": report.get("tail_severity"),
        "max_drawdown": report.get("max_drawdown"),
        "current_drawdown": report.get("current_drawdown"),
        "volatility_annual": report.get("volatility_annual"),
        "garch_filtered": bool(report.get("garch_filtered", report.get("filter_active", False))),
        "garch_active": bool(report.get("filter_active", False)),
        "source": "compute_garch_risk",
    }
    with open(risk_metrics_path, "w") as f:
        json.dump(risk_payload, f, indent=2, default=str)

    print(f"  VaR 95%:      {metrics.var_95:.2f}%")
    print(f"  CVaR 95%:     {metrics.cvar_95:.2f}%")
    print(f"  CVaR Ratio:   {metrics.cvar_ratio:.2f}")
    print(f"  Tail:         {metrics.tail_severity}")
    print(f"  GARCH Active: {metrics.filter_active}")
    if metrics.filter_active:
        print(f"  Persistence:  {metrics.garch_persistence:.4f}")
        print(f"  Cond Vol:     {metrics.conditional_volatility_current:.2f}%")
    print(f"  Report saved: {report_path}")


if __name__ == "__main__":
    main()
