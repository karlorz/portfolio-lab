#!/usr/bin/env python3
"""
ETF Premium Display for Dashboard (v2.92)

Dashboard integration module for ETF premium monitoring.
Generates formatted output for health.py dashboard display.

Author: Autonomous Agent
Version: v2.92
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETFPremiumDisplay:
    """Generate dashboard-compatible ETF premium display"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_file = self.data_dir / "etf_pricing.json"
    
    def load_pricing_data(self) -> Optional[dict]:
        """Load pricing data from JSON file"""
        if not self.data_file.exists():
            return None
        
        try:
            with open(self.data_file) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load pricing data: {e}")
            return None
    
    def format_premium_badge(self, premium_pct: float, alert_level: str) -> str:
        """Generate ANSI-colored badge for premium status"""
        # ANSI color codes
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        RED = "\033[31m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        
        symbol = "✓" if abs(premium_pct) < 0.0015 else "⚠" if abs(premium_pct) < 0.0025 else "✗"
        
        if alert_level == "critical":
            return f"{RED}{BOLD}{symbol} {premium_pct:+.4%}{RESET}"
        elif alert_level == "warning":
            return f"{YELLOW}{BOLD}{symbol} {premium_pct:+.4%}{RESET}"
        elif alert_level == "elevated":
            return f"{YELLOW}{symbol} {premium_pct:+.4%}{RESET}"
        else:
            return f"{GREEN}{symbol} {premium_pct:+.4%}{RESET}"
    
    def generate_dashboard_section(self) -> str:
        """Generate formatted dashboard section for ETF premiums"""
        data = self.load_pricing_data()
        
        if not data:
            return "\n[ETF Premium Monitor: No data available]\n"
        
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("ETF PREMIUM/DISCOUNT MONITOR v2.92")
        lines.append("=" * 60)
        
        # Parse timestamp
        ts = data.get("timestamp", "Unknown")
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                ts_str = ts
        else:
            ts_str = str(ts)
        
        lines.append(f"Last Update: {ts_str}")
        lines.append("-" * 60)
        
        # Portfolio ETFs (core holdings)
        portfolio_etfs = ["SPY", "GLD", "TLT", "IEF", "EFA", "VXUS", "QQQ", "DBC"]
        
        lines.append("\nCore Portfolio ETFs:")
        lines.append(f"{'Symbol':<8} {'Market':>10} {'Est NAV':>10} {'Premium':>12} {'Alert':>10}")
        lines.append("-" * 60)
        
        pricing = data.get("pricing", {})
        for symbol in portfolio_etfs:
            if symbol in pricing:
                p = pricing[symbol]
                market = p.get("market_price", 0)
                nav = p.get("nav", 0)
                premium = p.get("premium_pct", 0)
                alert = p.get("alert_level", "unknown")
                
                market_str = f"${market:.2f}" if market else "N/A"
                nav_str = f"${nav:.2f}" if nav else "N/A"
                premium_str = f"{premium:+.4%}" if premium else "N/A"
                
                lines.append(f"{symbol:<8} {market_str:>10} {nav_str:>10} {premium_str:>12} {alert:>10}")
        
        # Alerts section
        alerts = data.get("alerts", [])
        if alerts:
            lines.append("\n⚠ Active Alerts:")
            for alert in alerts:
                sym = alert.get("symbol", "?")
                level = alert.get("level", "unknown")
                pct = alert.get("premium_pct", 0)
                lines.append(f"  {sym}: {level.upper()} - Premium {pct:+.4%}")
        
        # Blocking summary
        critical_count = sum(1 for a in alerts if a.get("level") == "critical")
        if critical_count > 0:
            lines.append(f"\n{BOLD}{RED}🔒 TRADING BLOCKED for {critical_count} ETF(s){RESET}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def get_status_summary(self) -> dict:
        """Get machine-readable status summary"""
        data = self.load_pricing_data()
        
        if not data:
            return {
                "available": False,
                "reason": "No pricing data available",
            }
        
        pricing = data.get("pricing", {})
        alerts = data.get("alerts", [])
        
        portfolio_etfs = ["SPY", "GLD", "TLT", "IEF", "EFA", "VXUS", "QQQ", "DBC"]
        
        # Check which portfolio ETFs have critical alerts
        blocked_symbols = []
        warning_symbols = []
        
        for alert in alerts:
            sym = alert.get("symbol")
            level = alert.get("level")
            if sym in portfolio_etfs:
                if level == "critical":
                    blocked_symbols.append(sym)
                elif level in ["warning", "elevated"]:
                    warning_symbols.append(sym)
        
        return {
            "available": True,
            "timestamp": data.get("timestamp"),
            "total_etfs": len(pricing),
            "critical_alerts": sum(1 for a in alerts if a.get("level") == "critical"),
            "warning_alerts": sum(1 for a in alerts if a.get("level") in ["warning", "elevated"]),
            "blocked_symbols": blocked_symbols,
            "warning_symbols": warning_symbols,
            "can_trade": len(blocked_symbols) == 0,
        }


# ANSI codes for use in module
BOLD = "\033[1m"
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RESET = "\033[0m"


def main():
    """CLI entry point"""
    import sys
    
    display = ETFPremiumDisplay()
    
    if len(sys.argv) > 1 and sys.argv[1] == "json":
        # Output JSON for programmatic use
        summary = display.get_status_summary()
        print(json.dumps(summary, indent=2))
    
    elif len(sys.argv) > 1 and sys.argv[1] == "status":
        # Machine-readable status
        summary = display.get_status_summary()
        if summary.get("available"):
            print(f"available|{summary['timestamp']}|{summary['critical_alerts']}|{summary['warning_alerts']}")
        else:
            print("unavailable|no data")
    
    else:
        # Full dashboard display
        print(display.generate_dashboard_section())


if __name__ == "__main__":
    main()
