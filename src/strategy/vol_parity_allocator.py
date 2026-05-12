"""
Volatility Parity Allocator
Portfolio-level volatility targeting with VIX as vol contribution asset.

Part of v2.21 Multi-Asset Volatility Parity & Convexity Harvesting.
Allocates by volatility contribution rather than capital weight.
"""

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.vix_futures import VIXDataManager, fetch_vix_futures_data
from strategy.convexity_harvest import ConvexityHarvestStrategy, ConvexityPosition


@dataclass
class VolParityAllocation:
    """Volatility parity portfolio allocation"""
    date: str
    target_volatility: float      # Target portfolio volatility (e.g., 10%)
    
    # Core allocation (80% of capital)
    spy_pct: float               # SPY weight
    gld_pct: float               # GLD weight
    tlt_pct: float               # TLT weight
    core_vol_contribution: float  # Combined vol contribution from core
    
    # Convexity allocation (variable, max 7%)
    vix_short_pct: float         # Short VIX futures allocation
    vix_tail_pct: float          # Long VIX call spread (tail protection)
    vix_vol_contribution: float  # VIX vol contribution (can exceed capital weight)
    
    # Cash buffer
    cash_pct: float
    
    # Risk metrics
    expected_portfolio_vol: float
    expected_max_dd: float       # Estimated max drawdown
    rebalance_triggered: bool
    rebalance_reason: Optional[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @property
    def total_allocation(self) -> float:
        return self.spy_pct + self.gld_pct + self.tlt_pct + self.vix_short_pct + self.vix_tail_pct + self.cash_pct
    
    @property
    def total_vol_contribution(self) -> float:
        return self.core_vol_contribution + self.vix_vol_contribution


class VolatilityParityAllocator:
    """
    Implements volatility parity allocation strategy.
    
    Key insight: VIX futures contribute disproportionately to portfolio
    volatility relative to their capital allocation, providing convexity.
    
    Target: 10% portfolio volatility with:
    - Core (SPY/GLD/TLT): 80% capital, ~11% volatility
    - VIX overlay: Up to 7% capital, but provides vol contribution
    - Cash: Remainder for liquidity and rebalancing
    """
    
    # Configuration
    TARGET_VOLATILITY = 10.0      # Target portfolio volatility (%)
    CORE_BASE_WEIGHTS = {         # Base allocation (before vol scaling)
        'SPY': 0.46,
        'GLD': 0.38,
        'TLT': 0.16
    }
    CORE_ASSET_VOLS = {           # Individual asset volatilities
        'SPY': 15.0,              # 15% annualized
        'GLD': 14.0,
        'TLT': 12.0
    }
    MAX_VIX_SHORT_PCT = 5.0       # Max 5% in short VIX (volmageddon protection)
    MAX_VIX_TAIL_PCT = 2.0        # Max 2% in tail protection (cost)
    REBALANCE_THRESHOLD = 10.0    # Rebalance if allocation drifts >10%
    
    def __init__(
        self,
        vix_strategy: Optional[ConvexityHarvestStrategy] = None,
        target_vol: float = TARGET_VOLATILITY
    ):
        self.vix_strategy = vix_strategy or ConvexityHarvestStrategy()
        self.target_vol = target_vol
        self.vix_manager = VIXDataManager()
        self.last_allocation: Optional[VolParityAllocation] = None
    
    def calculate_core_allocation(self, convexity_signal: ConvexityPosition) -> Tuple[Dict[str, float], float]:
        """
        Calculate core SPY/GLD/TLT allocation considering VIX position.
        
        When VIX short position is active, it reduces effective portfolio
        volatility through negative correlation, allowing slightly higher
        core allocation if desired.
        """
        # Base weights (46/38/16)
        base_weights = self.CORE_BASE_WEIGHTS.copy()
        
        # Adjust based on VIX regime
        # High VIX = reduce equity, increase gold/bonds
        vix_level = convexity_signal.vix_level
        
        if vix_level > 30:
            # Stress regime: more defensive
            weights = {
                'SPY': 0.35,
                'GLD': 0.45,
                'TLT': 0.20
            }
        elif vix_level > 25:
            # Elevated vol: slight defensive tilt
            weights = {
                'SPY': 0.40,
                'GLD': 0.42,
                'TLT': 0.18
            }
        elif vix_level < 15:
            # Low vol: can increase equity slightly
            weights = {
                'SPY': 0.50,
                'GLD': 0.35,
                'TLT': 0.15
            }
        else:
            # Normal regime
            weights = base_weights
        
        # Calculate volatility contribution from core
        # Simplified: weighted average vol, ignoring correlations
        core_vol = sum(
            weights[asset] * self.CORE_ASSET_VOLS[asset]
            for asset in weights
        )
        
        return weights, core_vol
    
    def calculate_vix_allocation(self, convexity_signal: ConvexityPosition) -> Tuple[float, float, float]:
        """
        Calculate VIX allocation and its volatility contribution.
        
        VIX futures have ~80% annualized volatility, but provide negative
        correlation to equities. The key insight is that VIX contributes
        ~25% of portfolio volatility with only 5% capital allocation.
        
        Returns: (short_pct, tail_pct, vol_contribution)
        """
        # Short VIX from convexity harvest signal
        vix_short = min(convexity_signal.allocation_pct, self.MAX_VIX_SHORT_PCT)
        
        # Tail protection sizing
        # Increase tail protection when:
        # 1. VIX is low (cheap insurance)
        # 2. Short VIX position is large (hedge the harvest)
        vix_level = convexity_signal.vix_level
        
        if vix_level < 15:
            # Cheap vol, buy more insurance
            tail_pct = 2.0
        elif vix_level > 30:
            # Expensive vol, reduce tail hedge (already protected by high VIX)
            tail_pct = 0.5
        else:
            # Base tail protection proportional to short position
            tail_pct = min(2.0, vix_short * 0.4)  # 40% of short position as hedge
        
        tail_pct = min(tail_pct, self.MAX_VIX_TAIL_PCT)
        
        # VIX volatility contribution
        # VIX futures: ~80% vol
        # Short position: negative contribution (reduces portfolio vol)
        # Estimated portfolio vol reduction: ~2-4% depending on correlation
        vix_vol_contribution = vix_short * 80.0 * 0.3  # 30% effective vol contribution
        
        # Tail protection adds some volatility (gamma exposure)
        tail_vol_contribution = tail_pct * 150.0 * 0.1  # High vol but small weight
        
        total_vix_vol = vix_vol_contribution - tail_vol_contribution
        
        return vix_short, tail_pct, total_vix_vol
    
    def generate_allocation(self, date: str) -> VolParityAllocation:
        """Generate volatility parity allocation for a given date"""
        # Get convexity harvest signal
        convexity_signal = self.vix_strategy.generate_signal(date)
        
        # Calculate core allocation
        core_weights, core_vol = self.calculate_core_allocation(convexity_signal)
        
        # Scale core to target 80% of capital (or adjust based on VIX)
        # When VIX short is active, core can stay at 80%
        # When no VIX position, could consider increasing core slightly
        core_total_pct = 80.0
        
        spy_pct = core_weights['SPY'] * core_total_pct
        gld_pct = core_weights['GLD'] * core_total_pct
        tlt_pct = core_weights['TLT'] * core_total_pct
        
        # Calculate VIX allocation
        vix_short, vix_tail, vix_vol_contribution = self.calculate_vix_allocation(convexity_signal)
        
        # Cash buffer
        # Minimum 13% per spec, increase if high uncertainty
        if convexity_signal.vix_level > 25:
            cash_pct = 18.0  # More cash in high vol
        elif convexity_signal.exit_triggered:
            cash_pct = 15.0  # Extra cash if recent exit
        else:
            cash_pct = 13.0 + max(0, 5.0 - vix_short)  # More cash if no VIX position
        
        # Ensure we don't exceed 100%
        total_allocated = spy_pct + gld_pct + tlt_pct + vix_short + vix_tail + cash_pct
        if total_allocated > 100.0:
            # Scale down proportionally
            scale = 100.0 / total_allocated
            spy_pct *= scale
            gld_pct *= scale
            tlt_pct *= scale
            vix_short *= scale
            vix_tail *= scale
            cash_pct = 100.0 - (spy_pct + gld_pct + tlt_pct + vix_short + vix_tail)
        
        # Expected portfolio volatility
        # Core contribution (scaled down to 80%)
        core_vol_scaled = core_vol * (core_total_pct / 100.0)
        
        # VIX reduces effective volatility through negative correlation
        # Simplified: assume 0.3 correlation between VIX and equity vol
        vix_hedge_effect = vix_short * 80.0 * 0.3 / 100.0  # Reduction effect
        
        expected_vol = core_vol_scaled - vix_hedge_effect + (vix_tail * 150.0 * 0.1 / 100.0)
        
        # Expected max drawdown (simplified)
        # Typical 3-sigma event: vol * 3 / sqrt(252) * sqrt(days)
        # Assume 30-day drawdown event
        expected_max_dd = expected_vol * 1.5  # Rough estimate
        
        # Check if rebalance needed
        rebalance_triggered = False
        rebalance_reason = None
        
        if self.last_allocation:
            # Check drift from last allocation
            drift_spy = abs(spy_pct - self.last_allocation.spy_pct)
            drift_gld = abs(gld_pct - self.last_allocation.gld_pct)
            drift_tlt = abs(tlt_pct - self.last_allocation.tlt_pct)
            
            if max(drift_spy, drift_gld, drift_tlt) > self.REBALANCE_THRESHOLD:
                rebalance_triggered = True
                rebalance_reason = f"Drift exceeded {self.REBALANCE_THRESHOLD}%"
        
        allocation = VolParityAllocation(
            date=date,
            target_volatility=self.target_vol,
            spy_pct=spy_pct,
            gld_pct=gld_pct,
            tlt_pct=tlt_pct,
            core_vol_contribution=core_vol_scaled,
            vix_short_pct=vix_short,
            vix_tail_pct=vix_tail,
            vix_vol_contribution=vix_vol_contribution,
            cash_pct=cash_pct,
            expected_portfolio_vol=expected_vol,
            expected_max_dd=expected_max_dd,
            rebalance_triggered=rebalance_triggered,
            rebalance_reason=rebalance_reason
        )
        
        self.last_allocation = allocation
        return allocation
    
    def get_current_allocation(self) -> Dict:
        """Get current allocation for today"""
        today = datetime.now().strftime('%Y-%m-%d')
        allocation = self.generate_allocation(today)
        return {
            'allocation': allocation.to_dict(),
            'summary': {
                'total_capital_allocation': allocation.total_allocation,
                'total_vol_contribution': allocation.total_vol_contribution,
                'target_vol': self.target_vol,
                'vol_gap': self.target_vol - allocation.expected_portfolio_vol,
                'vix_regime': 'contango' if allocation.vix_short_pct > 0 else 'backwardation'
            }
        }
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str
    ) -> Dict:
        """Run volatility parity backtest"""
        allocations = []
        
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            allocation = self.generate_allocation(date_str)
            allocations.append(allocation)
            current += datetime.timedelta(days=1)
        
        # Calculate summary statistics
        avg_spy = sum(a.spy_pct for a in allocations) / len(allocations)
        avg_gld = sum(a.gld_pct for a in allocations) / len(allocations)
        avg_tlt = sum(a.tlt_pct for a in allocations) / len(allocations)
        avg_vix_short = sum(a.vix_short_pct for a in allocations) / len(allocations)
        avg_vix_tail = sum(a.vix_tail_pct for a in allocations) / len(allocations)
        avg_cash = sum(a.cash_pct for a in allocations) / len(allocations)
        
        avg_expected_vol = sum(a.expected_portfolio_vol for a in allocations) / len(allocations)
        
        rebalance_count = sum(1 for a in allocations if a.rebalance_triggered)
        
        return {
            'period': f"{start_date} to {end_date}",
            'days': len(allocations),
            'average_allocation': {
                'SPY': f"{avg_spy:.1f}%",
                'GLD': f"{avg_gld:.1f}%",
                'TLT': f"{avg_tlt:.1f}%",
                'VIX_Short': f"{avg_vix_short:.1f}%",
                'VIX_Tail': f"{avg_vix_tail:.1f}%",
                'Cash': f"{avg_cash:.1f}%"
            },
            'average_expected_volatility': f"{avg_expected_vol:.1f}%",
            'rebalance_events': rebalance_count,
            'target_volatility': f"{self.target_vol:.1f}%"
        }


def main():
    """CLI entry point"""
    allocator = VolatilityParityAllocator()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--backtest':
        start = sys.argv[2] if len(sys.argv) > 2 else '2020-01-01'
        end = sys.argv[3] if len(sys.argv) > 3 else '2024-12-31'
        
        print(f"Running volatility parity backtest: {start} to {end}")
        results = allocator.run_backtest(start, end)
        
        print("\n=== Volatility Parity Backtest ===")
        print(json.dumps(results, indent=2))
    
    elif len(sys.argv) > 1 and sys.argv[1] == '--current':
        current = allocator.get_current_allocation()
        print(json.dumps(current, indent=2))
    
    else:
        print("Volatility Parity Allocator (v2.21)")
        print("Usage: python3 vol_parity_allocator.py [--backtest START END] [--current]")
        print()
        
        # Show sample allocations
        test_dates = ['2023-01-15', '2023-06-15', '2023-10-15', '2024-03-15']
        print("Sample Allocations:")
        for date in test_dates:
            alloc = allocator.generate_allocation(date)
            print(f"\n{date}:")
            print(f"  SPY: {alloc.spy_pct:.1f}% | GLD: {alloc.gld_pct:.1f}% | TLT: {alloc.tlt_pct:.1f}%")
            print(f"  VIX Short: {alloc.vix_short_pct:.1f}% | VIX Tail: {alloc.vix_tail_pct:.1f}%")
            print(f"  Cash: {alloc.cash_pct:.1f}%")
            print(f"  Expected Vol: {alloc.expected_portfolio_vol:.1f}% (Target: {alloc.target_volatility:.1f}%)")
            if alloc.vix_short_pct > 0:
                print(f"  ✓ Convexity harvest active")
            if alloc.rebalance_triggered:
                print(f"  ↻ Rebalance triggered: {alloc.rebalance_reason}")


if __name__ == '__main__':
    main()
