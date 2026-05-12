"""
Convexity Harvest Signal Module
Implements VIX term structure carry strategy with risk management.

Part of v2.21 Multi-Asset Volatility Parity & Convexity Harvesting.
Based on CBOE (2024) and AQR (2025) research.
"""

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.vix_futures import VIXDataManager, fetch_vix_futures_data


@dataclass
class ConvexityPosition:
    """Convexity harvest position state"""
    date: str
    allocation_pct: float        # Portfolio allocation (0-5%)
    position_type: str           # 'short_vix', 'long_vix', 'flat'
    vix_level: float
    contango_pct: float
    expected_roll_yield: float
    risk_score: float            # 0-1 risk assessment
    exit_triggered: bool
    exit_reason: Optional[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ConvexityHarvestStrategy:
    """
    VIX term structure convexity harvesting strategy.
    
    Core idea: When VIX futures are in contango (front < back), shorting
    the front month and rolling monthly captures the roll-down yield.
    
    Risk management:
    - Max 5% portfolio allocation (volmageddon protection)
    - Exit on VIX > 35 (stress regime)
    - Exit on 1-day VIX spike > 20%
    - Exit if backwardation persists >3 days
    """
    
    # Configuration constants
    MAX_ALLOCATION_PCT = 5.0      # Never exceed 5% (volmageddon protection)
    VIX_STRESS_THRESHOLD = 35.0   # Exit all shorts
    VIX_SPIKE_THRESHOLD = 20.0    # 1-day VIX move threshold
    CONTANGO_ENTRY_THRESHOLD = 5.0  # Min contango to enter
    STRONG_CONTANGO_THRESHOLD = 10.0  # Max allocation trigger
    BACKWARDATION_EXIT_DAYS = 3   # Days of backwardation before exit
    
    def __init__(self, vix_data_manager: Optional[VIXDataManager] = None):
        self.vix_manager = vix_data_manager or VIXDataManager()
        self.position_history: List[ConvexityPosition] = []
        self.consecutive_backwardation_days = 0
        self.last_vix_level = None
        self.last_allocation = 0.0
    
    def calculate_position_size(
        self,
        contango_pct: float,
        vix_level: float,
        portfolio_value: float = 100000.0
    ) -> Tuple[float, str]:
        """
        Calculate position size based on contango steepness and VIX level.
        
        Returns: (allocation_pct, reasoning)
        """
        # Base case: No position if contango < threshold or VIX in stress zone
        if vix_level > self.VIX_STRESS_THRESHOLD:
            return 0.0, f"VIX stress level ({vix_level:.1f} > {self.VIX_STRESS_THRESHOLD})"
        
        if contango_pct < 0:
            return 0.0, f"Backwardation ({contango_pct:.1f}%) - no short position"
        
        if contango_pct < self.CONTANGO_ENTRY_THRESHOLD:
            return 0.0, f"Contango too flat ({contango_pct:.1f}% < {self.CONTANGO_ENTRY_THRESHOLD}%)"
        
        # Calculate base allocation based on contango steepness
        # 5-10% contango = 2-4% allocation
        # 10%+ contango = 4-5% allocation (max)
        if contango_pct < self.STRONG_CONTANGO_THRESHOLD:
            base_allocation = 2.0 + (contango_pct - self.CONTANGO_ENTRY_THRESHOLD) * 0.4
        else:
            base_allocation = 4.0 + min(1.0, (contango_pct - self.STRONG_CONTANGO_THRESHOLD) * 0.2)
        
        # VIX level adjustment
        # Reduce allocation as VIX approaches stress threshold
        vix_factor = max(0.0, 1.0 - (vix_level / self.VIX_STRESS_THRESHOLD))
        adjusted_allocation = base_allocation * (0.5 + 0.5 * vix_factor)
        
        # Cap at max allocation
        final_allocation = min(adjusted_allocation, self.MAX_ALLOCATION_PCT)
        
        reasoning = f"Contango {contango_pct:.1f}%, VIX {vix_level:.1f}, factor {vix_factor:.2f}"
        
        return final_allocation, reasoning
    
    def check_exit_triggers(
        self,
        vix_level: float,
        contango_pct: float,
        date: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if position should be exited based on risk triggers.
        
        Returns: (should_exit, reason)
        """
        # Trigger 1: VIX stress level
        if vix_level > self.VIX_STRESS_THRESHOLD:
            return True, f"VIX stress ({vix_level:.1f} > {self.VIX_STRESS_THRESHOLD})"
        
        # Trigger 2: VIX spike (if we have previous data)
        if self.last_vix_level is not None:
            vix_change_pct = abs((vix_level - self.last_vix_level) / self.last_vix_level * 100)
            if vix_change_pct > self.VIX_SPIKE_THRESHOLD:
                return True, f"VIX spike ({vix_change_pct:.1f}% in 1 day)"
        
        # Trigger 3: Sustained backwardation
        if contango_pct < 0:
            self.consecutive_backwardation_days += 1
            if self.consecutive_backwardation_days >= self.BACKWARDATION_EXIT_DAYS:
                return True, f"Backwardation persisted ({self.consecutive_backwardation_days} days)"
        else:
            self.consecutive_backwardation_days = 0
        
        # Trigger 4: Emergency circuit breaker (simulated - in production, check actual circuit breaker state)
        # This would integrate with the circuit_breaker.py module
        
        return False, None
    
    def generate_signal(self, date: str) -> ConvexityPosition:
        """
        Generate convexity harvest signal for a given date.
        
        This is the main entry point for daily signal generation.
        """
        # Get VIX term structure for date
        signal_data = self.vix_manager.get_contango_signal(date)
        
        if not signal_data:
            # No data available - flat position
            return ConvexityPosition(
                date=date,
                allocation_pct=0.0,
                position_type='flat',
                vix_level=0.0,
                contango_pct=0.0,
                expected_roll_yield=0.0,
                risk_score=1.0,
                exit_triggered=False,
                exit_reason='No VIX data available'
            )
        
        vix_level = signal_data['vix_level']
        contango_pct = signal_data['contango_spot_1m']
        
        # Check exit triggers first (if we have an existing position)
        should_exit, exit_reason = self.check_exit_triggers(vix_level, contango_pct, date)
        
        if should_exit and self.last_allocation > 0:
            position = ConvexityPosition(
                date=date,
                allocation_pct=0.0,
                position_type='flat',
                vix_level=vix_level,
                contango_pct=contango_pct,
                expected_roll_yield=0.0,
                risk_score=1.0,
                exit_triggered=True,
                exit_reason=exit_reason
            )
        elif contango_pct > 0:
            # Contango regime - potential short VIX position
            allocation, reasoning = self.calculate_position_size(contango_pct, vix_level)
            
            position = ConvexityPosition(
                date=date,
                allocation_pct=allocation,
                position_type='short_vix' if allocation > 0 else 'flat',
                vix_level=vix_level,
                contango_pct=contango_pct,
                expected_roll_yield=signal_data['annualized_roll_yield'] if allocation > 0 else 0.0,
                risk_score=1.0 - (allocation / self.MAX_ALLOCATION_PCT) if allocation > 0 else 1.0,
                exit_triggered=False,
                exit_reason=None
            )
        else:
            # Backwardation - no short position (or flip to long protection)
            position = ConvexityPosition(
                date=date,
                allocation_pct=0.0,
                position_type='flat',  # Could be 'long_vix_protection' with different logic
                vix_level=vix_level,
                contango_pct=contango_pct,
                expected_roll_yield=0.0,
                risk_score=0.5,
                exit_triggered=False,
                exit_reason='Backwardation regime'
            )
        
        # Update state
        self.last_vix_level = vix_level
        self.last_allocation = position.allocation_pct
        self.position_history.append(position)
        
        return position
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0
    ) -> Dict:
        """
        Run historical backtest of convexity harvest strategy.
        
        Returns performance summary.
        """
        # Ensure we have VIX data
        if not self.vix_manager.data:
            fetch_vix_futures_data(start_date, end_date)
        
        positions = []
        capital = initial_capital
        allocated_capital = 0.0
        
        # Generate daily signals
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        daily_returns = []
        
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            
            position = self.generate_signal(date_str)
            positions.append(position)
            
            # Simplified P&L calculation
            # In reality, this would use actual VIX futures prices
            if position.allocation_pct > 0 and position.expected_roll_yield > 0:
                # Assume capturing ~1/12 of annualized roll yield per month
                monthly_yield = position.expected_roll_yield / 12
                daily_return = monthly_yield / 21  # ~21 trading days
                
                # Apply some random noise and occasional VIX spikes
                noise = (hash(date_str) % 100 - 50) / 500  # -0.1% to +0.1%
                
                # Stress events: 1% chance of large loss
                if (hash(date_str + 'stress') % 100) == 0:
                    spike_loss = -5.0  # 5% loss on allocated capital
                else:
                    spike_loss = 0.0
                
                position_return = (daily_return + noise + spike_loss) * (position.allocation_pct / 100)
            else:
                position_return = 0.0
            
            daily_returns.append(position_return)
            capital *= (1 + position_return / 100)
            
            current += timedelta(days=1)
        
        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital * 100
        avg_daily_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0
        
        # Volatility (annualized)
        import math
        variance = sum((r - avg_daily_return) ** 2 for r in daily_returns) / len(daily_returns) if daily_returns else 0
        volatility = math.sqrt(variance) * math.sqrt(252)  # Annualized
        
        # Max drawdown
        peak = initial_capital
        max_dd = 0.0
        running_capital = initial_capital
        for ret in daily_returns:
            running_capital *= (1 + ret / 100)
            if running_capital > peak:
                peak = running_capital
            dd = (peak - running_capital) / peak * 100
            max_dd = max(max_dd, dd)
        
        # Sharpe ratio (simplified, assuming 0% risk-free rate)
        sharpe = (avg_daily_return * 252) / volatility if volatility > 0 else 0
        
        # Count exit triggers
        exits = [p for p in positions if p.exit_triggered]
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return_pct': total_return,
            'annualized_return_pct': total_return / ((datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days / 365),
            'volatility_pct': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd,
            'total_positions': len(positions),
            'days_with_position': len([p for p in positions if p.allocation_pct > 0]),
            'exit_events': len(exits),
            'exit_reasons': list(set(e.exit_reason for e in exits if e.exit_reason))
        }
    
    def get_current_signal(self) -> Dict:
        """Get current convexity harvest signal for today's date"""
        today = datetime.now().strftime('%Y-%m-%d')
        position = self.generate_signal(today)
        return position.to_dict()


def main():
    """CLI entry point for convexity harvest strategy"""
    strategy = ConvexityHarvestStrategy()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--backtest':
        # Run backtest
        start = sys.argv[2] if len(sys.argv) > 2 else '2020-01-01'
        end = sys.argv[3] if len(sys.argv) > 3 else '2024-12-31'
        
        print(f"Running convexity harvest backtest: {start} to {end}")
        results = strategy.run_backtest(start, end)
        
        print("\n=== Backtest Results ===")
        print(f"Period: {results['start_date']} to {results['end_date']}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Annualized Return: {results['annualized_return_pct']:.2f}%")
        print(f"Volatility: {results['volatility_pct']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"Days with Position: {results['days_with_position']}/{results['total_positions']}")
        print(f"Exit Events: {results['exit_events']}")
        if results['exit_reasons']:
            print(f"Exit Reasons: {', '.join(results['exit_reasons'])}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == '--signal':
        # Get current signal
        signal = strategy.get_current_signal()
        print(json.dumps(signal, indent=2))
    
    else:
        # Demo mode - show sample signals
        print("Convexity Harvest Strategy (v2.21)")
        print("Usage: python3 convexity_harvest.py [--backtest START END] [--signal]")
        print()
        
        # Generate sample signals for recent dates
        test_dates = ['2024-01-15', '2024-06-15', '2024-10-15', '2025-01-15']
        print("\nSample Signals:")
        for date in test_dates:
            pos = strategy.generate_signal(date)
            print(f"\n{date}: {pos.position_type.upper()}")
            print(f"  VIX: {pos.vix_level:.2f}")
            print(f"  Contango: {pos.contango_pct:.2f}%")
            print(f"  Allocation: {pos.allocation_pct:.1f}%")
            print(f"  Expected Roll Yield: {pos.expected_roll_yield:.1f}%")
            if pos.exit_triggered:
                print(f"  ⚠️ EXIT TRIGGERED: {pos.exit_reason}")


if __name__ == '__main__':
    main()
