"""
Intraday Execution Cost Model
Implements time-of-day cost estimation for rebalancing optimization
Phase 1 of v2.71 Intraday Seasonality Execution Optimization

References:
- Heston, Korajczyk, Sadka (2010) - Intraday Periodicity
- Bogousslavsky & Muravyev (2023) - Informed Trading Intraday
"""

from dataclasses import dataclass
from datetime import time, datetime
from typing import Dict, Optional, Tuple
import json
import os


@dataclass
class IntradayCostEstimate:
    """Cost estimate for a specific execution scenario"""
    spread_cost_bps: float
    impact_cost_bps: float
    total_cost_bps: float
    confidence: str  # 'high', 'medium', 'low'
    recommended_window: str


class IntradayExecutionCostModel:
    """
    Estimates execution cost by time-of-day using empirical patterns
    
    Based on research synthesis:
    - Midday (11:00-14:00): Tightest spreads, lowest toxicity
    - Opening (9:30-10:00): High volatility, wide spreads - avoid
    - Close (15:30-16:00): Elevated volume but variable spreads - avoid
    """
    
    # Baseline half-spreads in basis points (empirical averages)
    BASELINE_SPREAD_BPS: Dict[str, float] = {
        'SPY': 0.5,
        'QQQ': 0.7,
        'TLT': 1.2,
        'GLD': 1.0,
        'IEF': 0.8,
        'EFA': 2.5,
        'VXUS': 2.0,
        'MTUM': 1.5,
        'VLUE': 1.5,
        'USMV': 1.2,
    }
    
    # Time-of-day multipliers (empirical from market microstructure research)
    # Hour (9-16 ET) -> spread multiplier
    TIME_MULTIPLIERS: Dict[int, float] = {
        9: 3.0,   # 9:30-9:59: Opening volatility, overnight gap digestion
        10: 1.5,  # 10:00-10:59: Normalizing, spreads narrowing
        11: 1.0,  # 11:00-11:59: OPTIMAL - tightest spreads
        12: 1.0,  # 12:00-12:59: OPTIMAL - lunch lull, retail flow
        13: 1.0,  # 13:00-13:59: OPTIMAL - afternoon start, still quiet
        14: 1.2,  # 14:00-14:59: Pre-close positioning begins
        15: 2.0,  # 15:00-15:29: Elevated volume, rebalancing flow
        16: 2.5,  # 15:30-16:00: Close auction, MOC/IOC flow
    }
    
    # Volatility scaling factor for impact model
    VOLATILITY_FACTOR = 20.0  # bps scaling for square root impact model
    
    def __init__(self, custom_profiles_path: Optional[str] = None):
        """
        Initialize cost model with optional custom profiles
        
        Args:
            custom_profiles_path: Path to JSON file with symbol-specific profiles
        """
        self.profiles: Dict[str, dict] = {}
        
        if custom_profiles_path and os.path.exists(custom_profiles_path):
            with open(custom_profiles_path, 'r') as f:
                self.profiles = json.load(f)
    
    def get_baseline_spread(self, symbol: str) -> float:
        """Get baseline spread for symbol, defaulting to 2.0 bps"""
        return self.BASELINE_SPREAD_BPS.get(symbol, 2.0)
    
    def get_time_multiplier(self, hour: int) -> float:
        """Get spread multiplier for hour of day"""
        return self.TIME_MULTIPLIERS.get(hour, 1.5)
    
    def estimate_cost(
        self, 
        symbol: str, 
        hour: int, 
        size_dv_pct: float,
        urgency: str = 'normal'
    ) -> IntradayCostEstimate:
        """
        Estimate total execution cost in basis points
        
        Args:
            symbol: Ticker symbol
            hour: Hour of day (9-16, ET)
            size_dv_pct: Order size as % of daily volume (e.g., 0.01 = 1%)
            urgency: 'low' | 'normal' | 'high' | 'urgent'
        
        Returns:
            IntradayCostEstimate with cost breakdown
        """
        base_spread = self.get_baseline_spread(symbol)
        time_mult = self.get_time_multiplier(hour)
        
        # Half-spread cost (crossing the spread)
        spread_cost = base_spread * time_mult / 2
        
        # Market impact using square root model
        # Cost ~ volatility * sqrt(size/DV)
        # Cap at 100 bps for very large orders
        impact = min(self.VOLATILITY_FACTOR * (size_dv_pct ** 0.5), 100.0)
        
        # Urgency adjustment (urgent orders accept higher costs)
        urgency_mult = {
            'low': 0.9,
            'normal': 1.0,
            'high': 1.3,
            'urgent': 1.6
        }.get(urgency, 1.0)
        
        total_cost = (spread_cost + impact) * urgency_mult
        
        # Determine confidence based on data quality
        if symbol in self.BASELINE_SPREAD_BPS and hour in [11, 12, 13]:
            confidence = 'high'
        elif symbol in self.BASELINE_SPREAD_BPS:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Recommend window
        if hour in [11, 12, 13]:
            window = 'optimal'
        elif hour in [10, 14]:
            window = 'acceptable'
        else:
            window = 'avoid'
        
        return IntradayCostEstimate(
            spread_cost_bps=spread_cost,
            impact_cost_bps=impact,
            total_cost_bps=total_cost,
            confidence=confidence,
            recommended_window=window
        )
    
    def find_optimal_window(
        self, 
        symbol: str, 
        size_dv_pct: float,
        urgency: str = 'normal',
        start_hour: int = 9,
        end_hour: int = 16
    ) -> Tuple[int, IntradayCostEstimate]:
        """
        Find the optimal hour to execute within a time range
        
        Returns:
            Tuple of (optimal_hour, cost_estimate)
        """
        best_hour = start_hour
        best_cost = float('inf')
        best_estimate = None
        
        for hour in range(start_hour, end_hour + 1):
            estimate = self.estimate_cost(symbol, hour, size_dv_pct, urgency)
            if estimate.total_cost_bps < best_cost:
                best_cost = estimate.total_cost_bps
                best_hour = hour
                best_estimate = estimate
        
        return best_hour, best_estimate
    
    def compare_windows(
        self,
        symbol: str,
        size_dv_pct: float,
        urgency: str = 'normal'
    ) -> Dict[int, dict]:
        """
        Compare costs across all trading hours
        
        Returns:
            Dict mapping hour to cost breakdown
        """
        comparison = {}
        for hour in range(9, 17):
            est = self.estimate_cost(symbol, hour, size_dv_pct, urgency)
            comparison[hour] = {
                'spread_cost_bps': round(est.spread_cost_bps, 2),
                'impact_cost_bps': round(est.impact_cost_bps, 2),
                'total_cost_bps': round(est.total_cost_bps, 2),
                'window': est.recommended_window,
                'confidence': est.confidence
            }
        return comparison


class RebalanceScheduler:
    """
    Schedules rebalancing orders during optimal intraday windows
    """
    
    # Priority windows (earliest = preferred)
    OPTIMAL_WINDOWS: list = [
        (time(11, 0), time(13, 0)),   # Primary: tightest spreads, lowest volume
        (time(10, 30), time(14, 30)),  # Secondary: acceptable
        (time(10, 0), time(15, 0)),   # Tertiary: if urgent
    ]
    
    AVOID_WINDOWS: list = [
        (time(9, 30), time(10, 0)),   # Opening volatility
        (time(15, 30), time(16, 0)),   # Close auction
    ]
    
    def __init__(self, urgency: str = 'normal'):
        """
        Args:
            urgency: 'low' | 'normal' | 'high' | 'urgent'
        """
        self.urgency = urgency
        self.max_delay_hours = {
            'low': 8,
            'normal': 4,
            'high': 1,
            'urgent': 0
        }.get(urgency, 4)
        
        self.cost_model = IntradayExecutionCostModel()
    
    def is_optimal_time(self, t: time) -> bool:
        """Check if time falls within optimal windows"""
        for start, end in self.OPTIMAL_WINDOWS:
            if start <= t <= end:
                return True
        return False
    
    def should_avoid(self, t: time) -> bool:
        """Check if time falls within avoid windows"""
        for start, end in self.AVOID_WINDOWS:
            if start <= t <= end:
                return True
        return False
    
    def schedule(
        self, 
        target_time: datetime,
        symbol: str = 'SPY',
        size_dv_pct: float = 0.01
    ) -> datetime:
        """
        Find optimal execution time near target
        
        Args:
            target_time: Desired execution time
            symbol: Ticker to trade
            size_dv_pct: Order size as % of daily volume
        
        Returns:
            Scheduled execution time
        """
        from datetime import timedelta
        
        now = datetime.now()
        deadline = target_time + timedelta(hours=self.max_delay_hours)
        
        # If urgent, execute immediately
        if self.urgency == 'urgent':
            return max(target_time, now)
        
        # Check if target is already optimal
        if self.is_optimal_time(target_time.time()) and target_time >= now:
            return target_time
        
        # Find next optimal window within deadline
        current_date = target_time.date()
        
        for window_start, window_end in self.OPTIMAL_WINDOWS:
            # Try today
            candidate_start = datetime.combine(current_date, window_start)
            candidate_end = datetime.combine(current_date, window_end)
            
            if candidate_start >= now and candidate_end <= deadline:
                return candidate_start
            
            # Try tomorrow if past today's windows
            if candidate_start < now:
                tomorrow = current_date + timedelta(days=1)
                # Skip weekends
                if tomorrow.weekday() < 5:  # Monday=0, Friday=4
                    candidate_start = datetime.combine(tomorrow, window_start)
                    if candidate_start <= deadline + timedelta(days=1):
                        return candidate_start
        
        # Fallback: execute at target if within deadline
        if target_time <= deadline:
            return max(target_time, now)
        
        # Last resort: execute at deadline
        return deadline
    
    def get_schedule_recommendation(
        self,
        symbol: str,
        size_dv_pct: float,
        target_time: datetime
    ) -> dict:
        """
        Get full scheduling recommendation with cost estimates
        """
        scheduled_time = self.schedule(target_time, symbol, size_dv_pct)
        
        # Cost at target time
        target_hour = target_time.hour if target_time.hour >= 9 else 9
        target_cost = self.cost_model.estimate_cost(
            symbol, target_hour, size_dv_pct, self.urgency
        )
        
        # Cost at scheduled time
        sched_hour = scheduled_time.hour if scheduled_time.hour >= 9 else 9
        sched_cost = self.cost_model.estimate_cost(
            symbol, sched_hour, size_dv_pct, self.urgency
        )
        
        savings_bps = target_cost.total_cost_bps - sched_cost.total_cost_bps
        
        return {
            'symbol': symbol,
            'target_time': target_time.isoformat(),
            'scheduled_time': scheduled_time.isoformat(),
            'urgency': self.urgency,
            'target_cost_bps': round(target_cost.total_cost_bps, 2),
            'scheduled_cost_bps': round(sched_cost.total_cost_bps, 2),
            'estimated_savings_bps': round(savings_bps, 2),
            'delay_hours': round((scheduled_time - target_time).total_seconds() / 3600, 1),
            'window_quality': sched_cost.recommended_window,
            'confidence': sched_cost.confidence
        }


def demo():
    """Demonstrate cost model capabilities"""
    print("=" * 60)
    print("Intraday Execution Cost Model - Demo")
    print("=" * 60)
    
    model = IntradayExecutionCostModel()
    scheduler = RebalanceScheduler(urgency='normal')
    
    # Demo 1: Cost comparison for SPY
    print("\n1. SPY Cost Comparison (1% of daily volume):")
    print("-" * 40)
    comparison = model.compare_windows('SPY', size_dv_pct=0.01)
    for hour, data in comparison.items():
        marker = "★" if data['window'] == 'optimal' else " "
        print(f"  {marker} {hour:02d}:00 - {data['total_cost_bps']:.1f} bps "
              f"({data['spread_cost_bps']:.1f} spread + "
              f"{data['impact_cost_bps']:.1f} impact) "
              f"[{data['window']}]")
    
    # Demo 2: Optimal window finder
    print("\n2. Optimal Window Finder:")
    print("-" * 40)
    symbols = ['SPY', 'TLT', 'GLD', 'EFA']
    for sym in symbols:
        hour, estimate = model.find_optimal_window(sym, size_dv_pct=0.01)
        print(f"  {sym}: Hour {hour:02d}:00 @ {estimate.total_cost_bps:.1f} bps")
    
    # Demo 3: Scheduler recommendation
    print("\n3. Scheduler Recommendation:")
    print("-" * 40)
    target = datetime.now().replace(hour=9, minute=30, second=0)
    rec = scheduler.get_schedule_recommendation('SPY', 0.01, target)
    print(f"  Target: {rec['target_time']}")
    print(f"  Scheduled: {rec['scheduled_time']}")
    print(f"  Delay: {rec['delay_hours']} hours")
    print(f"  Target cost: {rec['target_cost_bps']} bps")
    print(f"  Scheduled cost: {rec['scheduled_cost_bps']} bps")
    print(f"  Savings: {rec['estimated_savings_bps']} bps")
    print(f"  Window quality: {rec['window_quality']}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    demo()
