"""
VIX Insurance Signal Generator
Generates entry/exit signals for VIX call spread insurance overlay.

Part of v2.44: VIX Call Spread Insurance Overlay
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsuranceSignalType(Enum):
    ENTER = "enter"           # Open new insurance position
    HOLD = "hold"             # Maintain existing position
    ROLL = "roll"             # Roll to next expiration
    EXIT_PROFIT = "exit_profit"    # Take profit on vol spike
    EXIT_EXPIRE = "exit_expire"    # Exit before expiration
    NO_ACTION = "no_action"   # No position, no entry criteria met


@dataclass
class VIXInsuranceSignal:
    """Signal for VIX insurance overlay strategy"""
    timestamp: datetime
    signal_type: InsuranceSignalType
    spot_vix: float
    portfolio_value: float
    
    # Current position (if any)
    position_active: bool
    position_contracts: int
    position_strike: float
    position_expiry: Optional[datetime]
    position_cost_basis: float
    position_current_value: float
    position_pnl: float
    
    # Recommendation
    recommended_action: str
    target_strike: Optional[float]
    target_expiry: Optional[datetime]
    recommended_contracts: int
    estimated_premium: float
    
    # Risk metrics
    days_to_position_expiry: Optional[int]
    vix_percentile_30d: float
    portfolio_distance_from_ath: float
    
    # Metadata
    correlation_vix_spy: float
    insurance_budget_ytd: float
    insurance_payouts_ytd: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['signal_type'] = self.signal_type.value
        data['position_expiry'] = self.position_expiry.isoformat() if self.position_expiry else None
        data['target_expiry'] = self.target_expiry.isoformat() if self.target_expiry else None
        return data


class VIXInsuranceSignalGenerator:
    """
    Generates signals for VIX call spread insurance overlay.
    
    Strategy rules:
    - Enter when VIX < 20 and portfolio within 10% of ATH
    - Size: 1% of portfolio (max)
    - Target: 30-delta calls, 60-day maturity
    - Exit: VIX > 35 (profit taking) or 30 DTE
    """
    
    # Strategy parameters
    VIX_ENTRY_THRESHOLD_HIGH = 22.0   # Don't enter above this
    VIX_ENTRY_THRESHOLD_LOW = 16.0    # Full size below this
    VIX_EXIT_PROFIT = 35.0           # Take profit above this
    VIX_EXIT_STOP = 40.0             # Emergency exit
    
    PORTFOLIO_DISTANCE_THRESHOLD = 0.10  # Within 10% of ATH
    MAX_ALLOCATION_PCT = 0.01           # 1% max allocation
    TARGET_DELTA = 0.30
    MIN_DTE_FOR_ROLL = 30               # Roll when 30 days left
    
    def __init__(self, data_dir: str = "data/signals"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.vix_history_file = self.data_dir / "vix_history.json"
        self.position_file = self.data_dir / "vix_insurance_position.json"
        self.signal_file = self.data_dir / "vix_insurance_signal.json"
        
        # YTD tracking
        self.ytd_file = self.data_dir / "insurance_ytd_stats.json"
    
    def load_position(self) -> Optional[Dict]:
        """Load current insurance position if exists"""
        if not self.position_file.exists():
            return None
        
        with open(self.position_file, 'r') as f:
            return json.load(f)
    
    def save_position(self, position: Optional[Dict]):
        """Save current insurance position"""
        if position is None:
            if self.position_file.exists():
                self.position_file.unlink()
            return
        
        with open(self.position_file, 'w') as f:
            json.dump(position, f, indent=2, default=str)
    
    def load_vix_history(self, days: int = 30) -> List[Dict]:
        """Load VIX history for percentile calculation"""
        if not self.vix_history_file.exists():
            return []
        
        with open(self.vix_history_file, 'r') as f:
            history = json.load(f)
        
        # Filter to last N days
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        return [h for h in history if h.get('timestamp', '') > cutoff]
    
    def save_vix_reading(self, vix: float):
        """Save VIX reading to history"""
        history = []
        if self.vix_history_file.exists():
            with open(self.vix_history_file, 'r') as f:
                history = json.load(f)
        
        history.append({
            'timestamp': datetime.now().isoformat(),
            'vix': vix
        })
        
        # Keep last 90 days
        cutoff = (datetime.now() - timedelta(days=90)).isoformat()
        history = [h for h in history if h.get('timestamp', '') > cutoff]
        
        with open(self.vix_history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def get_ytd_stats(self) -> Dict:
        """Get YTD insurance statistics"""
        if not self.ytd_file.exists():
            return {
                'year': datetime.now().year,
                'total_premiums_paid': 0.0,
                'total_payouts_received': 0.0,
                'num_positions_opened': 0,
                'num_positions_closed': 0,
                'win_rate': 0.0,
                'avg_premium': 0.0
            }
        
        with open(self.ytd_file, 'r') as f:
            stats = json.load(f)
        
        # Reset if new year
        if stats.get('year') != datetime.now().year:
            return {
                'year': datetime.now().year,
                'total_premiums_paid': 0.0,
                'total_payouts_received': 0.0,
                'num_positions_opened': 0,
                'num_positions_closed': 0,
                'win_rate': 0.0,
                'avg_premium': 0.0
            }
        
        return stats
    
    def calculate_position_size(self, portfolio_value: float, vix: float) -> float:
        """
        Calculate position size based on VIX level.
        
        Linear scaling between thresholds:
        - VIX < 16: Full 1%
        - VIX 16-22: Scale linearly from 1% to 0.5%
        - VIX > 22: 0% (don't enter)
        """
        if vix >= self.VIX_ENTRY_THRESHOLD_HIGH:
            return 0.0
        
        if vix <= self.VIX_ENTRY_THRESHOLD_LOW:
            return self.MAX_ALLOCATION_PCT
        
        # Linear interpolation between 1% and 0.5%
        pct = (self.VIX_ENTRY_THRESHOLD_HIGH - vix) / \
              (self.VIX_ENTRY_THRESHOLD_HIGH - self.VIX_ENTRY_THRESHOLD_LOW)
        
        # Scale: at VIX=16 -> 1%, at VIX=22 -> 0.5%
        return 0.005 + (pct * 0.005)
    
    def calculate_vix_percentile(self, vix: float, history: List[Dict]) -> float:
        """Calculate VIX percentile over last 30 days"""
        if not history:
            return 50.0  # Default to median
        
        vix_values = [h['vix'] for h in history]
        vix_values.sort()
        
        # Count values below current
        below = sum(1 for v in vix_values if v < vix)
        return (below / len(vix_values)) * 100
    
    def get_portfolio_distance_from_ath(self) -> float:
        """
        Get portfolio distance from all-time high.
        Returns percentage distance (0.05 = 5% below ATH)
        """
        # Try to load from portfolio data
        portfolio_file = Path("data/portfolio_value.json")
        if portfolio_file.exists():
            with open(portfolio_file, 'r') as f:
                data = json.load(f)
                current = data.get('current_value', 100000)
                ath = data.get('all_time_high', current)
                if ath > 0:
                    return (ath - current) / ath
        
        # Default: assume at ATH (most conservative for entry)
        return 0.0
    
    def get_correlation_vix_spy(self) -> float:
        """Get current VIX-SPY correlation"""
        # Try to load from correlation data
        corr_file = Path("data/correlation_analysis.json")
        if corr_file.exists():
            with open(corr_file, 'r') as f:
                data = json.load(f)
                return data.get('vix_spy_correlation', -0.7)
        
        # Default: historical average
        return -0.7
    
    def generate_signal(self, vix: float, portfolio_value: float = 100000) -> VIXInsuranceSignal:
        """
        Generate insurance signal based on current market conditions.
        
        Args:
            vix: Current VIX spot price
            portfolio_value: Current portfolio value (default 100K)
        
        Returns:
            VIXInsuranceSignal with recommendation
        """
        # Save VIX reading
        self.save_vix_reading(vix)
        
        # Load current position
        position = self.load_position()
        
        # Load history for calculations
        vix_history = self.load_vix_history(days=30)
        vix_percentile = self.calculate_vix_percentile(vix, vix_history)
        
        # Portfolio metrics
        distance_from_ath = self.get_portfolio_distance_from_ath()
        correlation = self.get_correlation_vix_spy()
        
        # YTD stats
        ytd_stats = self.get_ytd_stats()
        
        # Determine signal
        signal_type = InsuranceSignalType.NO_ACTION
        recommended_action = "No action"
        target_strike = None
        target_expiry = None
        recommended_contracts = 0
        estimated_premium = 0.0
        
        # Current position values
        position_active = position is not None
        position_contracts = 0
        position_strike = 0.0
        position_expiry = None
        position_cost_basis = 0.0
        position_current_value = 0.0
        position_pnl = 0.0
        days_to_expiry = None
        
        if position:
            position_contracts = position.get('contracts', 0)
            position_strike = position.get('strike', 0)
            expiry_str = position.get('expiration')
            if expiry_str:
                position_expiry = datetime.fromisoformat(expiry_str)
                days_to_expiry = (position_expiry - datetime.now()).days
            position_cost_basis = position.get('total_premium', 0)
            
            # Estimate current value (simplified)
            # Real value would come from option pricing model or mark-to-market
            intrinsic = max(0, vix - position_strike)
            position_current_value = position_contracts * 100 * intrinsic
            position_pnl = position_current_value - position_cost_basis
        
        # Logic: Check existing position first
        if position_active:
            if vix >= self.VIX_EXIT_PROFIT:
                # Take profit on vol spike
                signal_type = InsuranceSignalType.EXIT_PROFIT
                recommended_action = f"EXIT_PROFIT: VIX {vix:.2f} above profit threshold {self.VIX_EXIT_PROFIT}"
            
            elif days_to_expiry and days_to_expiry <= self.MIN_DTE_FOR_ROLL:
                # Time to roll
                signal_type = InsuranceSignalType.ROLL
                recommended_action = f"ROLL: Position expires in {days_to_expiry} days"
                
                # Calculate new position
                allocation_pct = self.calculate_position_size(portfolio_value, vix)
                allocation_dollars = portfolio_value * allocation_pct
                
                # Estimate: ~$1.00 per contract for 30-delta, 60-dte
                # VIX multiplier is $100
                estimated_contract_cost = 100 * 1.00
                recommended_contracts = int(allocation_dollars / estimated_contract_cost)
                estimated_premium = recommended_contracts * estimated_contract_cost
                
                # Target: 10% OTM roughly = 30-delta
                target_strike = vix * 1.10
                target_expiry = datetime.now() + timedelta(days=60)
            
            else:
                # Hold position
                signal_type = InsuranceSignalType.HOLD
                recommended_action = f"HOLD: Position active, {days_to_expiry} days to expiry, P&L: ${position_pnl:.2f}"
        
        # No position: Check entry criteria
        else:
            if vix >= self.VIX_ENTRY_THRESHOLD_HIGH:
                signal_type = InsuranceSignalType.NO_ACTION
                recommended_action = f"NO_ACTION: VIX {vix:.2f} too high for entry (>={self.VIX_ENTRY_THRESHOLD_HIGH})"
            
            elif distance_from_ath > self.PORTFOLIO_DISTANCE_THRESHOLD:
                signal_type = InsuranceSignalType.NO_ACTION
                recommended_action = f"NO_ACTION: Portfolio {distance_from_ath*100:.1f}% below ATH (threshold: {self.PORTFOLIO_DISTANCE_THRESHOLD*100:.0f}%)"
            
            elif correlation > -0.3:
                signal_type = InsuranceSignalType.NO_ACTION
                recommended_action = f"NO_ACTION: VIX-SPY correlation {correlation:.2f} (hedge not working, need <-0.3)"
            
            else:
                # All criteria met - enter position
                signal_type = InsuranceSignalType.ENTER
                
                allocation_pct = self.calculate_position_size(portfolio_value, vix)
                allocation_dollars = portfolio_value * allocation_pct
                
                # Estimate contract cost
                estimated_contract_cost = 100 * 1.00
                recommended_contracts = max(1, int(allocation_dollars / estimated_contract_cost))
                estimated_premium = recommended_contracts * estimated_contract_cost
                
                # Target parameters
                target_strike = vix * 1.10  # ~10% OTM
                target_expiry = datetime.now() + timedelta(days=60)
                
                recommended_action = f"ENTER: Buy {recommended_contracts} VIX {target_strike:.1f} calls, expiry ~{target_expiry.strftime('%Y-%m-%d')}"
        
        # Create signal
        signal = VIXInsuranceSignal(
            timestamp=datetime.now(),
            signal_type=signal_type,
            spot_vix=vix,
            portfolio_value=portfolio_value,
            
            position_active=position_active,
            position_contracts=position_contracts,
            position_strike=position_strike,
            position_expiry=position_expiry,
            position_cost_basis=position_cost_basis,
            position_current_value=position_current_value,
            position_pnl=position_pnl,
            
            recommended_action=recommended_action,
            target_strike=target_strike,
            target_expiry=target_expiry,
            recommended_contracts=recommended_contracts,
            estimated_premium=estimated_premium,
            
            days_to_position_expiry=days_to_expiry,
            vix_percentile_30d=vix_percentile,
            portfolio_distance_from_ath=distance_from_ath,
            
            correlation_vix_spy=correlation,
            insurance_budget_ytd=ytd_stats['total_premiums_paid'],
            insurance_payouts_ytd=ytd_stats['total_payouts_received']
        )
        
        # Save signal
        with open(self.signal_file, 'w') as f:
            json.dump(signal.to_dict(), f, indent=2)
        
        logger.info(f"VIX Insurance Signal: {signal_type.value} - {recommended_action}")
        
        return signal
    
    def update_position_after_execution(self, signal: VIXInsuranceSignal, 
                                        executed: bool,
                                        actual_contracts: int = 0,
                                        actual_premium: float = 0.0,
                                        actual_strike: float = 0.0,
                                        actual_expiry: Optional[datetime] = None):
        """
        Update position tracking after order execution.
        
        Args:
            signal: The signal that was acted upon
            executed: Whether the order was successfully executed
            actual_contracts: Number of contracts actually traded
            actual_premium: Actual premium paid/received
            actual_strike: Actual strike price
            actual_expiry: Actual expiration date
        """
        ytd_stats = self.get_ytd_stats()
        
        if signal.signal_type == InsuranceSignalType.ENTER and executed:
            # Open new position
            position = {
                'timestamp': datetime.now().isoformat(),
                'contracts': actual_contracts,
                'strike': actual_strike,
                'expiration': actual_expiry.isoformat() if actual_expiry else None,
                'total_premium': actual_premium,
                'entry_vix': signal.spot_vix
            }
            self.save_position(position)
            
            # Update YTD stats
            ytd_stats['total_premiums_paid'] += actual_premium
            ytd_stats['num_positions_opened'] += 1
            ytd_stats['avg_premium'] = ytd_stats['total_premiums_paid'] / ytd_stats['num_positions_opened']
            
        elif signal.signal_type == InsuranceSignalType.EXIT_PROFIT and executed:
            # Close position for profit
            position = self.load_position()
            if position:
                cost_basis = position.get('total_premium', 0)
                # Payout is the sale proceeds (simplified)
                # In reality, this would be mark-to-market or actual sale
                payout = actual_premium  # For calls, this would be sell price
                
                ytd_stats['total_payouts_received'] += payout
                ytd_stats['num_positions_closed'] += 1
                
                # Calculate win rate
                wins = sum(1 for _ in range(ytd_stats['num_positions_closed']) 
                          if ytd_stats['total_payouts_received'] > ytd_stats['total_premiums_paid'])
                ytd_stats['win_rate'] = wins / ytd_stats['num_positions_closed'] if ytd_stats['num_positions_closed'] > 0 else 0
                
                self.save_position(None)  # Clear position
        
        # Save updated YTD stats
        with open(self.ytd_file, 'w') as f:
            json.dump(ytd_stats, f, indent=2)


def main():
    """CLI entry point for testing"""
    import sys
    
    generator = VIXInsuranceSignalGenerator()
    
    # Test with various VIX levels
    test_vix_levels = [15.0, 18.0, 22.0, 35.0, 40.0]
    
    print(f"\n{'='*80}")
    print("VIX INSURANCE SIGNAL GENERATOR TEST")
    print(f"{'='*80}\n")
    
    for vix in test_vix_levels:
        signal = generator.generate_signal(vix=vix, portfolio_value=100000)
        
        print(f"VIX: {vix:.2f}")
        print(f"  Signal: {signal.signal_type.value.upper()}")
        print(f"  Action: {signal.recommended_action}")
        print(f"  Position Active: {signal.position_active}")
        if signal.recommended_contracts > 0:
            print(f"  Recommended: {signal.recommended_contracts} contracts")
            print(f"  Est. Premium: ${signal.estimated_premium:.2f}")
            print(f"  Target Strike: {signal.target_strike:.2f}" if signal.target_strike else "")
        print()
    
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
