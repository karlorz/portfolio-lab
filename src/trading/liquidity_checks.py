#!/usr/bin/env python3
"""
Portfolio-Lab v2.92: Liquidity Pre-Trade Checks

Pre-trade liquidity validation including ETF premium/discount checks.
Part of v2.92 ETF Premium Monitor feature.

Usage:
    from src.trading.liquidity_checks import LiquidityChecker
    
    checker = LiquidityChecker()
    eligible, reason = checker.check_trade_eligibility("SPY", "buy", 10000)
"""

import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.etf_pricing import ETFPricingEngine, ETF_PRICING_PATH


@dataclass
class LiquidityCheckResult:
    """Result of pre-trade liquidity check."""
    eligible: bool
    reason: str
    checks_passed: List[str]
    checks_failed: List[str]
    premium_pct: Optional[float] = None
    premium_status: Optional[str] = None
    timestamp: str = ""


class LiquidityChecker:
    """Pre-trade liquidity validation system."""
    
    # Thresholds
    PREMIUM_CRITICAL = 0.30  # 0.30% - block all trades
    PREMIUM_WARNING = 0.15   # 0.15% - caution for large trades
    PREMIUM_LOG = 0.10       # 0.10% - log for analysis
    
    # Size thresholds (% of portfolio)
    LARGE_TRADE_THRESHOLD = 5.0  # 5% of portfolio
    
    def __init__(self):
        self.etf_engine = ETFPricingEngine()
        self.log_path = Path("~/projects/portfolio-lab/data/liquidity_checks.log").expanduser()
    
    def check_trade_eligibility(
        self,
        symbol: str,
        side: str,
        dollar_amount: float,
        portfolio_value: float = 100000.0,
        force: bool = False
    ) -> LiquidityCheckResult:
        """
        Comprehensive pre-trade liquidity check.
        
        Args:
            symbol: ETF symbol
            side: 'buy' or 'sell'
            dollar_amount: Trade dollar amount
            portfolio_value: Total portfolio value for size calc
            force: Bypass warnings (not critical blocks)
            
        Returns:
            LiquidityCheckResult with eligibility and details
        """
        checks_passed = []
        checks_failed = []
        
        # Calculate position size as % of portfolio
        size_pct = (dollar_amount / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        # 1. ETF Premium Check
        premium_check = self._check_premium(symbol, side, size_pct, force)
        premium_pct = premium_check.get("premium_pct")
        premium_status = premium_check.get("status")
        
        if premium_check["passed"]:
            checks_passed.append(f"premium_{premium_status}")
        else:
            checks_failed.append(f"premium_{premium_status}")
        
        # 2. Market Hours Check (placeholder for future)
        # In production, check if market is open
        checks_passed.append("market_hours")
        
        # 3. Position Size Check (placeholder for risk limits)
        if size_pct <= 25:  # Max 25% single trade
            checks_passed.append("position_size")
        else:
            checks_failed.append("position_size_exceeded")
        
        # Determine overall eligibility
        # Critical failures always block
        has_critical = any("critical" in f for f in checks_failed)
        
        # Warnings block unless forced
        has_warning = any("warning" in f for f in checks_failed)
        
        if has_critical:
            eligible = False
            reason = premium_check.get("message", "Critical liquidity check failed")
        elif has_warning and not force:
            eligible = False
            reason = premium_check.get("message", "Warning: use --force to override")
        elif has_warning and force:
            eligible = True
            reason = f"WARNING (forced): {premium_check.get('message', '')}"
        else:
            eligible = True
            reason = "All liquidity checks passed"
        
        # Log the check
        self._log_check(symbol, side, dollar_amount, eligible, reason, premium_pct)
        
        return LiquidityCheckResult(
            eligible=eligible,
            reason=reason,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            premium_pct=premium_pct,
            premium_status=premium_status,
            timestamp=datetime.now().isoformat()
        )
    
    def _check_premium(self, symbol: str, side: str, size_pct: float, force: bool) -> Dict:
        """Check ETF premium/discount."""
        # Fetch latest pricing
        pricing = self.etf_engine.fetch_etf_pricing(symbol)
        
        if not pricing:
            return {
                "passed": True,  # Allow if data unavailable
                "status": "unknown",
                "premium_pct": None,
                "message": "Premium data unavailable - proceeding with caution"
            }
        
        abs_premium = abs(pricing.premium_pct)
        
        # Critical threshold
        if abs_premium > self.PREMIUM_CRITICAL:
            return {
                "passed": False,
                "status": "critical",
                "premium_pct": pricing.premium_pct,
                "message": f"CRITICAL: Premium {pricing.premium_pct:+.2f}% exceeds ±{self.PREMIUM_CRITICAL:.2f}% - trade blocked"
            }
        
        # Warning threshold with size consideration
        if abs_premium > self.PREMIUM_WARNING:
            if size_pct > self.LARGE_TRADE_THRESHOLD:
                return {
                    "passed": force,  # Only pass if forced
                    "status": "warning_large",
                    "premium_pct": pricing.premium_pct,
                    "message": f"WARNING: Premium {pricing.premium_pct:+.2f}% + large size {size_pct:.1f}% - use --force to proceed"
                }
            else:
                return {
                    "passed": True,
                    "status": "warning",
                    "premium_pct": pricing.premium_pct,
                    "message": f"WARNING: Premium {pricing.premium_pct:+.2f}% elevated but within limits"
                }
        
        # Advisory level
        if abs_premium > self.PREMIUM_LOG:
            return {
                "passed": True,
                "status": "advisory",
                "premium_pct": pricing.premium_pct,
                "message": f"ADVISORY: Premium {pricing.premium_pct:+.3f}% slightly elevated"
            }
        
        # Normal
        return {
            "passed": True,
            "status": "normal",
            "premium_pct": pricing.premium_pct,
            "message": f"Premium {pricing.premium_pct:+.3f}% within normal range"
        }
    
    def _log_check(self, symbol: str, side: str, amount: float, 
                   eligible: bool, reason: str, premium_pct: Optional[float]):
        """Log liquidity check for analysis."""
        try:
            timestamp = datetime.now().isoformat()
            status = "PASS" if eligible else "BLOCK"
            premium_str = f"{premium_pct:+.3f}%" if premium_pct else "N/A"
            
            log_entry = f"{timestamp} | {status} | {symbol} | {side} | ${amount:,.0f} | {premium_str} | {reason}\n"
            
            with open(self.log_path, 'a') as f:
                f.write(log_entry)
                
        except Exception as e:
            # Logging failure shouldn't block trades
            pass
    
    def get_recent_blocks(self, hours: int = 24) -> List[Dict]:
        """Get recent blocked trades for analysis."""
        if not self.log_path.exists():
            return []
        
        cutoff = datetime.now() - timedelta(hours=hours)
        blocks = []
        
        try:
            with open(self.log_path, 'r') as f:
                for line in f:
                    if "BLOCK" not in line:
                        continue
                    
                    parts = line.split(" | ")
                    if len(parts) >= 7:
                        ts_str = parts[0]
                        try:
                            ts = datetime.fromisoformat(ts_str)
                            if ts >= cutoff:
                                blocks.append({
                                    "timestamp": ts_str,
                                    "symbol": parts[2],
                                    "side": parts[3],
                                    "amount": parts[4],
                                    "reason": parts[6].strip()
                                })
                        except:
                            continue
        except Exception as e:
            print(f"Error reading log: {e}")
        
        return blocks


def main():
    """CLI for liquidity checks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Liquidity Pre-Trade Checks v2.92")
    parser.add_argument("--symbol", required=True, help="ETF symbol")
    parser.add_argument("--side", required=True, choices=["buy", "sell"], help="Trade side")
    parser.add_argument("--amount", type=float, required=True, help="Dollar amount")
    parser.add_argument("--portfolio", type=float, default=100000, help="Portfolio value")
    parser.add_argument("--force", action="store_true", help="Override warnings")
    parser.add_argument("--recent-blocks", action="store_true", help="Show recent blocked trades")
    
    args = parser.parse_args()
    
    if args.recent_blocks:
        checker = LiquidityChecker()
        blocks = checker.get_recent_blocks(hours=24)
        print(f"\nRecent blocked trades (last 24h): {len(blocks)}")
        for block in blocks[-10:]:  # Show last 10
            print(f"  {block['timestamp'][:19]} | {block['symbol']} {block['side']} | {block['reason'][:50]}")
        return
    
    # Run check
    checker = LiquidityChecker()
    result = checker.check_trade_eligibility(
        args.symbol.upper(),
        args.side,
        args.amount,
        args.portfolio,
        args.force
    )
    
    print(f"\nTrade: {args.side.upper()} ${args.amount:,.0f} {args.symbol.upper()}")
    print(f"Portfolio Value: ${args.portfolio:,.0f}")
    print(f"Position Size: {(args.amount/args.portfolio)*100:.2f}%")
    print()
    print(f"Status: {'✓ ALLOWED' if result.eligible else '✗ BLOCKED'}")
    print(f"Reason: {result.reason}")
    print(f"Premium: {result.premium_pct:+.3f}%" if result.premium_pct else "Premium: N/A")
    print(f"\nChecks Passed: {', '.join(result.checks_passed)}")
    if result.checks_failed:
        print(f"Checks Failed: {', '.join(result.checks_failed)}")


if __name__ == "__main__":
    main()
