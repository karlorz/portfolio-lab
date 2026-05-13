"""
VIX Insurance Signal Generator - Phase 2 Implementation
Generates tail hedge insurance signals based on VIX options chain data.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsuranceSignal(Enum):
    """Signal states for VIX insurance overlay."""
    NO_POSITION = "no_position"
    ENTER_FULL = "enter_full"      # 1% allocation
    ENTER_HALF = "enter_half"      # 0.5% allocation
    HOLD = "hold"                  # Maintain current position
    ROLL = "roll"                  # Roll to next expiration
    EXIT_PROFIT = "exit_profit"    # Take profit (VIX > 35)
    EXIT_EXPIRE = "exit_expire"  # Near expiration


@dataclass
class VIXInsuranceSignal:
    """Complete insurance signal with position recommendation."""
    timestamp: str
    signal: str
    vix_spot: float
    vix_regime: str  # cheap, fair, expensive
    portfolio_value: float
    
    # Position details
    allocation_percent: float
    allocation_dollars: float
    
    # Selected option
    selected_strike: Optional[float]
    selected_expiration: Optional[str]
    days_to_expiration: Optional[int]
    premium_cost: Optional[float]
    delta: Optional[float]
    breakeven_vix: Optional[float]
    
    # Risk metrics
    max_portfolio_allocation: float  # 1% cap
    annual_budget_used: float
    budget_remaining: float
    
    # Context
    portfolio_near_ath: bool
    existing_position: bool
    days_to_roll: Optional[int]
    
    def to_dict(self) -> dict:
        return asdict(self)


class VIXInsuranceSignalGenerator:
    """
    VIX Call Spread Insurance Signal Generator
    
    Generates entry/exit/roll signals for tail hedge insurance overlay.
    Target: 1% allocation when VIX < 20, portfolio near ATH.
    """
    
    DB_PATH = Path("/root/projects/portfolio-lab/data/vix_options.db")
    OUTPUT_PATH = Path("/root/projects/portfolio-lab/data/signals/vix_insurance_overlay.json")
    
    # Configuration
    MAX_ALLOCATION = 0.01  # 1% of portfolio
    MAX_SINGLE_TRADE = 0.005  # 0.5% max single trade
    
    # VIX regimes
    VIX_CHEAP = 16
    VIX_FAIR = 20
    VIX_EXPENSIVE = 22
    
    # Exit conditions
    VIX_PROFIT_TAKE = 35
    ROLL_DAYS_BEFORE_EXPIRY = 5
    
    def __init__(self, portfolio_value: float = 100000):
        self.portfolio_value = portfolio_value
        self.annual_budget = portfolio_value * 0.01  # 1% annual budget
        self.budget_used_ytd = 0.0
        
    def _load_candidates(self) -> List[Dict]:
        """Load latest insurance candidates from database."""
        conn = sqlite3.connect(self.DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM insurance_candidates 
            WHERE timestamp = (SELECT MAX(timestamp) FROM insurance_candidates)
            ORDER BY delta ASC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def _load_vix_context(self) -> Dict:
        """Load VIX historical context."""
        conn = sqlite3.connect(self.DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Latest VIX
        cursor.execute("""
            SELECT * FROM vix_history 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        latest = cursor.fetchone()
        
        # 30-day history
        cursor.execute("""
            SELECT vix_spot FROM vix_history 
            ORDER BY timestamp DESC 
            LIMIT 30
        """)
        history = [r[0] for r in cursor.fetchall()]
        
        conn.close()
        
        if not latest:
            return {}
        
        return {
            'vix_spot': latest['vix_spot'],
            'vix_9day': latest['vix_9day'],
            'vix_3m': latest['vix_3m'],
            'contango': latest['contango'],
            'history_30d': history,
            'vix_30d_avg': sum(history) / len(history) if history else latest['vix_spot'],
            'timestamp': latest['timestamp']
        }
    
    def _check_portfolio_ath(self) -> bool:
        """
        Check if portfolio is within 10% of all-time high.
        In practice, this would read from portfolio history.
        """
        # Placeholder - would read from portfolio DB
        # For now, assume we're within 10% of ATH
        return True
    
    def _check_existing_position(self) -> Dict:
        """Check for existing insurance positions."""
        # Would query positions database
        # Return empty for now
        return {}
    
    def _calculate_allocation(self, vix_spot: float) -> float:
        """
        Calculate appropriate allocation based on VIX level.
        
        VIX < 16: Full 1% (cheap vol)
        VIX 16-20: 0.5% (fair vol)
        VIX > 20: 0% (expensive vol)
        """
        if vix_spot < self.VIX_CHEAP:
            return self.MAX_ALLOCATION
        elif vix_spot < self.VIX_FAIR:
            return self.MAX_SINGLE_TRADE  # 0.5%
        else:
            return 0.0
    
    def _determine_vix_regime(self, vix_spot: float) -> str:
        """Classify VIX regime."""
        if vix_spot < self.VIX_CHEAP:
            return "cheap"
        elif vix_spot < self.VIX_FAIR:
            return "fair"
        elif vix_spot < self.VIX_EXPENSIVE:
            return "elevated"
        else:
            return "expensive"
    
    def generate_signal(self) -> VIXInsuranceSignal:
        """
        Generate insurance signal based on current market conditions.
        """
        # Load data
        candidates = self._load_candidates()
        context = self._load_vix_context()
        
        if not context:
            return VIXInsuranceSignal(
                timestamp=datetime.now().isoformat(),
                signal=InsuranceSignal.NO_POSITION.value,
                vix_spot=0.0,
                vix_regime="unknown",
                portfolio_value=self.portfolio_value,
                allocation_percent=0.0,
                allocation_dollars=0.0,
                selected_strike=None,
                selected_expiration=None,
                days_to_expiration=None,
                premium_cost=None,
                delta=None,
                breakeven_vix=None,
                max_portfolio_allocation=self.MAX_ALLOCATION,
                annual_budget_used=self.budget_used_ytd,
                budget_remaining=self.annual_budget - self.budget_used_ytd,
                portfolio_near_ath=False,
                existing_position=False,
                days_to_roll=None
            )
        
        vix_spot = context['vix_spot']
        vix_regime = self._determine_vix_regime(vix_spot)
        near_ath = self._check_portfolio_ath()
        existing = self._check_existing_position()
        
        # Default: no position
        signal_type = InsuranceSignal.NO_POSITION
        allocation = 0.0
        selected = None
        
        # Check for existing position exit conditions
        if existing:
            # Would check days to expiration, VIX level for profit take
            pass
        
        # Entry logic
        if not existing and near_ath:
            if vix_spot < self.VIX_FAIR:
                allocation = self._calculate_allocation(vix_spot)
                
                if allocation > 0 and candidates:
                    # Select best candidate (closest to 30-delta)
                    selected = candidates[0]
                    
                    if vix_spot < self.VIX_CHEAP:
                        signal_type = InsuranceSignal.ENTER_FULL
                    else:
                        signal_type = InsuranceSignal.ENTER_HALF
        
        # Build signal
        allocation_dollars = self.portfolio_value * allocation
        
        signal = VIXInsuranceSignal(
            timestamp=context.get('timestamp', datetime.now().isoformat()),
            signal=signal_type.value,
            vix_spot=vix_spot,
            vix_regime=vix_regime,
            portfolio_value=self.portfolio_value,
            allocation_percent=allocation,
            allocation_dollars=allocation_dollars,
            selected_strike=selected.get('strike') if selected else None,
            selected_expiration=selected.get('expiration_date') if selected else None,
            days_to_expiration=selected.get('days_to_expiration') if selected else None,
            premium_cost=selected.get('premium') if selected else None,
            delta=selected.get('delta') if selected else None,
            breakeven_vix=selected.get('breakeven_vix') if selected else None,
            max_portfolio_allocation=self.MAX_ALLOCATION,
            annual_budget_used=self.budget_used_ytd,
            budget_remaining=self.annual_budget - self.budget_used_ytd,
            portfolio_near_ath=near_ath,
            existing_position=bool(existing),
            days_to_roll=None  # Would calculate if existing position
        )
        
        return signal
    
    def export_signal(self, signal: VIXInsuranceSignal):
        """Export signal to JSON for downstream consumers."""
        self.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.OUTPUT_PATH, 'w') as f:
            json.dump(signal.to_dict(), f, indent=2)
        
        logger.info(f"Exported insurance signal to {self.OUTPUT_PATH}")
    
    def run(self) -> VIXInsuranceSignal:
        """Generate and export signal."""
        signal = self.generate_signal()
        self.export_signal(signal)
        return signal


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='VIX Insurance Signal Generator')
    parser.add_argument('--portfolio-value', type=float, default=100000,
                       help='Current portfolio value (default: 100000)')
    parser.add_argument('--run', action='store_true', help='Generate signal')
    
    args = parser.parse_args()
    
    if args.run:
        generator = VIXInsuranceSignalGenerator(portfolio_value=args.portfolio_value)
        signal = generator.run()
        
        print("\n=== VIX Insurance Signal ===\n")
        print(f"Signal: {signal.signal.upper()}")
        print(f"VIX Spot: {signal.vix_spot:.2f} ({signal.vix_regime})")
        print(f"Portfolio ATH: {signal.portfolio_near_ath}")
        print(f"Allocation: {signal.allocation_percent*100:.1f}% (${signal.allocation_dollars:,.0f})")
        
        if signal.selected_strike:
            print(f"\n--- Selected Option ---")
            print(f"Strike: ${signal.selected_strike:.1f}")
            print(f"Expiration: {signal.selected_expiration} ({signal.days_to_expiration} days)")
            print(f"Premium: ${signal.premium_cost:,.0f}")
            print(f"Delta: {signal.delta:.2f}")
            print(f"Breakeven VIX: {signal.breakeven_vix:.1f}")
        
        print(f"\n--- Budget ---")
        print(f"Annual: ${signal.max_portfolio_allocation*signal.portfolio_value:,.0f}")
        print(f"Used YTD: ${signal.annual_budget_used:,.0f}")
        print(f"Remaining: ${signal.budget_remaining:,.0f}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
