#!/usr/bin/env python3
"""
Portfolio-Lab v2.92: ETF Premium Dashboard Display

Integrates ETF premium data into the main dashboard display.
Part of v2.92 ETF Premium Monitor feature.

Usage:
    python -m src.monitor.etf_premium_display [--export]
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
ETF_PRICING_PATH = DATA_DIR / "etf_pricing.json"


def load_etf_pricing() -> Optional[Dict]:
    """Load current ETF pricing data."""
    if not ETF_PRICING_PATH.exists():
        return None
    
    try:
        with open(ETF_PRICING_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading ETF pricing: {e}")
        return None


def get_status_color(status: str) -> str:
    """Get ANSI color for status."""
    colors = {
        "normal": "\033[32m",      # Green
        "warning": "\033[33m",      # Yellow
        "critical": "\033[31m",      # Red
    }
    return colors.get(status, "")


def reset_color() -> str:
    return "\033[0m"


def format_premium_display(pricing_data: Dict) -> str:
    """Format ETF pricing for dashboard display."""
    if not pricing_data:
        return "  ETF Premium: Data unavailable\n"
    
    lines = []
    lines.append("╔══════════════════════════════════════════════════════════════════╗")
    lines.append("║  ETF PREMIUM/DISCOUNT MONITOR (v2.92)                             ║")
    lines.append("╠══════════════════════════════════════════════════════════════════╣")
    
    # Portfolio metrics
    portfolio = pricing_data.get("portfolio_metrics", {})
    overall_status = portfolio.get("overall_status", "unknown")
    weighted_premium = portfolio.get("weighted_premium_pct", 0)
    
    color = get_status_color(overall_status)
    reset = reset_color()
    
    lines.append(f"║  Portfolio Status: {color}{overall_status.upper()}{reset}" + " " * (42 - len(overall_status)) + "║")
    lines.append(f"║  Weighted Premium: {weighted_premium:+.3f}%" + " " * 38 + "║")
    
    # Warnings and criticals
    warnings = portfolio.get("warning_symbols", [])
    criticals = portfolio.get("critical_symbols", [])
    
    if warnings:
        warn_str = ", ".join(warnings)
        lines.append(f"║  Warnings: {warn_str}" + " " * (51 - len(warn_str)) + "║")
    if criticals:
        crit_str = ", ".join(criticals)
        lines.append(f"║  Critical: {color}{crit_str}{reset}" + " " * (51 - len(crit_str)) + "║")
    
    lines.append("╠══════════════════════════════════════════════════════════════════╣")
    
    # Individual ETF details
    etfs = pricing_data.get("etfs", {})
    for symbol, data in sorted(etfs.items()):
        status = data.get("alert_status", "normal")
        status_short = status[:4].upper()
        premium = data.get("premium_pct", 0)
        market = data.get("market_price", 0)
        nav = data.get("nav", 0)
        
        color = get_status_color(status)
        lines.append(f"║  {symbol:4} | ${market:>7.2f} | NAV ${nav:>7.2f} | {color}{premium:>+7.3f}%{reset} [{status_short}]        ║")
    
    lines.append("╚══════════════════════════════════════════════════════════════════╝")
    
    return "\n".join(lines)


def get_compact_summary(pricing_data: Dict) -> str:
    """Get compact one-line summary for health checks."""
    if not pricing_data:
        return "ETF Premium: unavailable"
    
    portfolio = pricing_data.get("portfolio_metrics", {})
    overall_status = portfolio.get("overall_status", "unknown")
    weighted_premium = portfolio.get("weighted_premium_pct", 0)
    
    criticals = portfolio.get("critical_symbols", [])
    warnings = portfolio.get("warning_symbols", [])
    
    status_str = f"{overall_status.upper()} ({weighted_premium:+.2f}%)"
    
    if criticals:
        status_str += f" CRITICAL:{','.join(criticals)}"
    elif warnings:
        status_str += f" WARN:{','.join(warnings)}"
    
    return f"ETF Premium: {status_str}"


def export_for_health_check():
    """Export compact summary for health.py integration."""
    pricing_data = load_etf_pricing()
    if pricing_data:
        summary = get_compact_summary(pricing_data)
        print(summary)
        return True
    print("ETF Premium: data unavailable")
    return False


def main():
    parser = argparse.ArgumentParser(description="ETF Premium Dashboard Display v2.92")
    parser.add_argument("--compact", action="store_true", help="Export compact summary for health checks")
    parser.add_argument("--refresh", action="store_true", help="Refresh data from etf_pricing.py")
    args = parser.parse_args()
    
    if args.refresh:
        # Trigger data refresh
        import subprocess
        result = subprocess.run(
            ["python", "-m", "src.data.etf_pricing", "--fetch", "--export"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error refreshing data: {result.stderr}")
    
    if args.compact:
        export_for_health_check()
        return
    
    # Full display
    pricing_data = load_etf_pricing()
    if pricing_data:
        print(format_premium_display(pricing_data))
    else:
        print("No ETF pricing data available.")
        print("Run: python -m src.data.etf_pricing --fetch --export")


if __name__ == "__main__":
    main()
