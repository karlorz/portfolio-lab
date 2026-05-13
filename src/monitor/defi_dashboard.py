#!/usr/bin/env python3
"""
DeFi Yield Dashboard Monitor
CLI display of DeFi yield spreads and monitoring status.
Part of v2.95 DeFi Yield Monitor infrastructure.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse

class DeFiDashboard:
    """Dashboard display for DeFi yield monitoring"""
    
    def __init__(self, db_path: str = "data/defi_yield_history.db", 
                 json_path: str = "data/defi_monitor.json"):
        self.db_path = Path(db_path)
        self.json_path = Path(json_path)
    
    def display_status(self):
        """Display current DeFi yield monitoring status"""
        print("\n" + "=" * 70)
        print("DeFi Yield Monitor v2.95 - Dashboard".center(70))
        print("=" * 70)
        
        if not self.json_path.exists():
            print("\n⚠️  No status available. Run `python -m src.data.defi_yield_fetcher --update` first")
            return
        
        with open(self.json_path, 'r') as f:
            status = json.load(f)
        
        timestamp = status.get('timestamp', 'N/A')
        treasury_yield = status.get('treasury_yield_3m', 0)
        
        print(f"\n📊 Last Updated: {timestamp}")
        print(f"💵 Risk-Free Rate (3M Treasury): {treasury_yield * 100:.2f}%")
        
        # Display current yields
        yields = status.get('yields', [])
        if yields:
            print("\n" + "-" * 70)
            print("Current DeFi Yields".center(70))
            print("-" * 70)
            print(f"{'Protocol':<15} {'Asset':<12} {'APY':>10} {'TVL ($M)':>15}")
            print("-" * 70)
            
            for y in yields:
                tvl_m = y.get('tvl_usd', 0) / 1_000_000
                print(f"{y.get('protocol', 'N/A'):<15} {y.get('asset', 'N/A'):<12} "
                      f"{y.get('yield_apy', 0) * 100:>9.2f}% {tvl_m:>14.1f}M")
        
        # Display spreads
        spreads = status.get('spreads', [])
        if spreads:
            print("\n" + "-" * 70)
            print("Yield Spreads vs Treasury".center(70))
            print("-" * 70)
            print(f"{'Protocol':<15} {'Spread':>12} {'Signal':>15}")
            print("-" * 70)
            
            for s in spreads:
                signal = s.get('signal', 'monitor')
                signal_icon = {
                    'allocate': '🟢',
                    'consider': '🟡',
                    'monitor': '⚪'
                }.get(signal, '⚪')
                
                spread_pct = s.get('spread', 0) * 100
                print(f"{s.get('protocol', 'N/A'):<15} {spread_pct:>+11.2f}% "
                      f"{signal_icon} {signal:>12}")
        
        # Display alerts
        alerts = status.get('alerts', [])
        if alerts:
            print("\n" + "-" * 70)
            print("⚠️  Active Alerts".center(70))
            print("-" * 70)
            for alert in alerts:
                print(f"  • {alert.get('type', 'Unknown')}: {alert.get('message', '')}")
        else:
            print("\n✅ No active alerts")
        
        # Display recommendation
        print("\n" + "=" * 70)
        print("Recommendation".center(70))
        print("=" * 70)
        
        any_allocate = any(s.get('signal') == 'allocate' for s in spreads)
        any_consider = any(s.get('signal') == 'consider' for s in spreads)
        
        if any_allocate:
            print("🟢 ALLOCATION CANDIDATE: DeFi yields exceed threshold")
            print("   → Review research for potential allocation")
        elif any_consider:
            print("🟡 MONITOR CLOSELY: Approaching allocation threshold")
            print("   → Continue observation, prepare research")
        else:
            print("⚪ MONITORING MODE: DeFi yields below allocation threshold")
            print("   → No action required at current scale ($100K)")
            print("   → Research available at compound/defi-yield-curve-arbitrage-2026")
        
        print("\n" + "=" * 70)
    
    def display_history(self, days: int = 30):
        """Display historical spread data"""
        if not self.db_path.exists():
            print("\n⚠️  No database found. Run update first.")
            return
        
        print(f"\n📈 DeFi Yield History (Last {days} Days)")
        print("=" * 70)
        
        protocols = ["Lido", "Jito", "Aave"]
        
        for protocol in protocols:
            spreads = self._get_spread_history(protocol, days)
            if spreads:
                avg_spread = sum(s['spread'] for s in spreads) / len(spreads)
                max_spread = max(s['spread'] for s in spreads)
                min_spread = min(s['spread'] for s in spreads)
                latest = spreads[0]['spread']
                
                print(f"\n{protocol}:")
                print(f"  Data points: {len(spreads)}")
                print(f"  Latest spread: {latest * 100:+.2f}%")
                print(f"  Avg spread ({days}d): {avg_spread * 100:+.2f}%")
                print(f"  Range: {min_spread * 100:+.2f}% to {max_spread * 100:+.2f}%")
                
                # Signal history
                signal_counts = {}
                for s in spreads:
                    sig = s.get('signal', 'monitor')
                    signal_counts[sig] = signal_counts.get(sig, 0) + 1
                
                print(f"  Signal distribution:")
                for sig, count in sorted(signal_counts.items()):
                    pct = count / len(spreads) * 100
                    print(f"    {sig}: {count} ({pct:.1f}%)")
            else:
                print(f"\n{protocol}: No data available")
        
        print("\n" + "=" * 70)
    
    def _get_spread_history(self, protocol: str, days: int) -> List[Dict]:
        """Get spread history from database"""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM spreads
                WHERE protocol = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (protocol, cutoff))
            return [dict(row) for row in cursor.fetchall()]
    
    def display_thresholds(self):
        """Display monitoring thresholds"""
        print("\n📋 DeFi Yield Monitoring Thresholds")
        print("=" * 70)
        
        print("\nYield Spread Thresholds (vs 3M Treasury):")
        print("  🟢 ALLOCATE:  > +2.00% (sustained for 30+ days)")
        print("  🟡 CONSIDER:  > +1.00%")
        print("  ⚪ MONITOR:   < +1.00% (current state)")
        
        print("\nCorrelation Thresholds (BTC/ETH-SPY):")
        print("  🟢 DIVERSIFY: < 0.25 (allocation candidate)")
        print("  🟡 MARGINAL:  0.25 - 0.50")
        print("  🔴 AVOID:     > 0.75 (no diversification benefit)")
        
        print("\nScale Requirements:")
        print("  Minimum portfolio: $500K for allocation consideration")
        print("  Optimal scale: $1M+ for positive risk-adjusted contribution")
        
        print("\nCurrent Portfolio Status:")
        print("  Scale: $100K - BELOW MINIMUM THRESHOLD")
        print("  Action: Continue monitoring only, no allocation")
        
        print("\n" + "=" * 70)

def main():
    parser = argparse.ArgumentParser(description='DeFi Yield Dashboard Monitor')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--history', type=int, metavar='DAYS', 
                       help='Show history for N days')
    parser.add_argument('--thresholds', action='store_true', 
                       help='Show monitoring thresholds')
    
    args = parser.parse_args()
    
    dashboard = DeFiDashboard()
    
    if args.status:
        dashboard.display_status()
    elif args.history:
        dashboard.display_history(args.history)
    elif args.thresholds:
        dashboard.display_thresholds()
    else:
        # Default: show status
        dashboard.display_status()

if __name__ == '__main__':
    main()
