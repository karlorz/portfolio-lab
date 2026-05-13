"""
Rebalance Scheduler - Phase 2 of v2.71 Intraday Seasonality Execution
Implements time window optimization for rebalancing order execution

References:
- Heston, Korajczyk, Sadka (2010) - Intraday Periodicity
- Bogousslavsky & Muravyev (2023) - Informed Trading Intraday
"""

from dataclasses import dataclass, field
from datetime import time, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import yaml
from pathlib import Path

from .intraday_cost_model import IntradayExecutionCostModel, IntradayCostEstimate


class OrderUrgency(Enum):
    """Urgency levels for rebalancing orders"""
    LOW = "low"          # Can wait for optimal window
    NORMAL = "normal"    # Standard priority
    HIGH = "high"        # Execute within hours
    URGENT = "urgent"    # Execute immediately


class ExecutionWindow(Enum):
    """Quality classification for execution time windows"""
    OPTIMAL = "optimal"      # 11:00-14:00 ET - tightest spreads
    ACCEPTABLE = "acceptable"  # 10:00-11:00, 14:00-15:30 ET
    AVOID = "avoid"          # 09:30-10:00, 15:30-16:00 ET - high volatility


@dataclass
class ScheduledOrder:
    """A rebalancing order with scheduling metadata"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    target_shares: float
    target_value: float
    urgency: OrderUrgency
    created_at: datetime
    scheduled_time: Optional[datetime] = None
    execution_window: Optional[ExecutionWindow] = None
    estimated_cost_bps: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'target_shares': self.target_shares,
            'target_value': self.target_value,
            'urgency': self.urgency.value,
            'created_at': self.created_at.isoformat(),
            'scheduled_time': self.scheduled_time.isoformat() if self.scheduled_time else None,
            'execution_window': self.execution_window.value if self.execution_window else None,
            'estimated_cost_bps': self.estimated_cost_bps
        }


@dataclass
class SchedulerConfig:
    """Configuration for the rebalance scheduler"""
    # Time windows (ET)
    optimal_start: time = time(11, 0)
    optimal_end: time = time(14, 0)
    acceptable_windows: List[Tuple[time, time]] = field(default_factory=lambda: [
        (time(10, 0), time(11, 0)),
        (time(14, 0), time(15, 30))
    ])
    avoid_windows: List[Tuple[time, time]] = field(default_factory=lambda: [
        (time(9, 30), time(10, 0)),
        (time(15, 30), time(16, 0))
    ])
    
    # Urgency-based delays
    max_delay_low: timedelta = timedelta(hours=4)      # Wait for optimal
    max_delay_normal: timedelta = timedelta(hours=2)  # Wait within acceptable
    max_delay_high: timedelta = timedelta(minutes=30)  # Minimal delay
    max_delay_urgent: timedelta = timedelta(seconds=0)  # Immediate
    
    # Cost improvement threshold (bps) to justify delay
    cost_threshold_bps: float = 2.0
    
    @classmethod
    def from_yaml(cls, path: str) -> 'SchedulerConfig':
        """Load config from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        if 'optimal_window' in data:
            ow = data['optimal_window']
            config.optimal_start = time(ow['start_hour'], ow.get('start_minute', 0))
            config.optimal_end = time(ow['end_hour'], ow.get('end_minute', 0))
        
        if 'acceptable_windows' in data:
            config.acceptable_windows = [
                (time(w['start_hour'], w.get('start_minute', 0)),
                 time(w['end_hour'], w.get('end_minute', 0)))
                for w in data['acceptable_windows']
            ]
        
        if 'urgency_delays' in data:
            ud = data['urgency_delays']
            config.max_delay_low = timedelta(minutes=ud.get('low_minutes', 240))
            config.max_delay_normal = timedelta(minutes=ud.get('normal_minutes', 120))
            config.max_delay_high = timedelta(minutes=ud.get('high_minutes', 30))
        
        if 'cost_threshold_bps' in data:
            config.cost_threshold_bps = data['cost_threshold_bps']
        
        return config


class RebalanceScheduler:
    """
    Schedules rebalancing orders for optimal execution time windows
    
    Reduces transaction costs by 5-15 bps through intelligent
    time-of-day order placement based on empirical spread patterns.
    """
    
    def __init__(
        self,
        cost_model: Optional[IntradayExecutionCostModel] = None,
        config: Optional[SchedulerConfig] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize scheduler
        
        Args:
            cost_model: Intraday cost estimation model
            config: Scheduler configuration
            config_path: Path to YAML config file (alternative to config)
        """
        self.cost_model = cost_model or IntradayExecutionCostModel()
        
        if config:
            self.config = config
        elif config_path and Path(config_path).exists():
            self.config = SchedulerConfig.from_yaml(config_path)
        else:
            self.config = SchedulerConfig()
        
        self.pending_orders: Dict[str, ScheduledOrder] = {}
        self.executed_orders: List[ScheduledOrder] = []
        self.cancelled_orders: List[ScheduledOrder] = []
    
    def schedule_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        target_shares: float,
        target_value: float,
        urgency: str = 'normal',
        current_time: Optional[datetime] = None,
        avg_daily_volume: Optional[float] = None
    ) -> ScheduledOrder:
        """
        Schedule a rebalancing order for optimal execution
        
        Args:
            order_id: Unique order identifier
            symbol: Ticker symbol
            side: 'buy' or 'sell'
            target_shares: Number of shares to trade
            target_value: Dollar value of trade
            urgency: 'low', 'normal', 'high', 'urgent'
            current_time: Current datetime (defaults to now)
            avg_daily_volume: For cost estimation (optional)
        
        Returns:
            ScheduledOrder with execution timing
        """
        if current_time is None:
            current_time = datetime.now()
        
        urgency_enum = OrderUrgency(urgency)
        
        # Create order
        order = ScheduledOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            target_shares=target_shares,
            target_value=target_value,
            urgency=urgency_enum,
            created_at=current_time
        )
        
        # Calculate size as % of daily volume for cost estimation
        size_dv_pct = 0.01  # Default 1%
        if avg_daily_volume and avg_daily_volume > 0:
            size_dv_pct = (target_shares * target_value / target_shares) / avg_daily_volume / 100
            size_dv_pct = max(0.0001, min(size_dv_pct, 0.1))  # Cap at 10%
        
        # Determine optimal execution time
        scheduled_time, window, cost_estimate = self._calculate_optimal_time(
            symbol, current_time, urgency_enum, size_dv_pct
        )
        
        order.scheduled_time = scheduled_time
        order.execution_window = window
        order.estimated_cost_bps = cost_estimate.total_cost_bps if cost_estimate else None
        
        self.pending_orders[order_id] = order
        
        return order
    
    def _calculate_optimal_time(
        self,
        symbol: str,
        current_time: datetime,
        urgency: OrderUrgency,
        size_dv_pct: float
    ) -> Tuple[datetime, ExecutionWindow, Optional[IntradayCostEstimate]]:
        """
        Calculate optimal execution time based on urgency and cost model
        
        Returns:
            Tuple of (scheduled_time, execution_window, cost_estimate)
        """
        # Urgent orders execute immediately
        if urgency == OrderUrgency.URGENT:
            return current_time, ExecutionWindow.ACCEPTABLE, None
        
        # Get max delay based on urgency
        max_delay = {
            OrderUrgency.LOW: self.config.max_delay_low,
            OrderUrgency.NORMAL: self.config.max_delay_normal,
            OrderUrgency.HIGH: self.config.max_delay_high
        }.get(urgency, timedelta(hours=2))
        
        latest_time = current_time + max_delay
        
        # Find the next optimal window
        current_date = current_time.date()
        current_hour = current_time.hour
        
        # Check if we're already in or can reach optimal window today
        optimal_start_dt = datetime.combine(current_date, self.config.optimal_start)
        optimal_end_dt = datetime.combine(current_date, self.config.optimal_end)
        
        # If optimal window is still available today
        if current_time < optimal_end_dt:
            # If we can make it to optimal window
            if optimal_start_dt >= current_time or current_time >= optimal_start_dt:
                # Calculate cost at 12:00 (middle of optimal)
                cost_12 = self.cost_model.estimate_cost(symbol, 12, size_dv_pct, urgency.value)
                
                # If we're before optimal, schedule for start
                if current_time < optimal_start_dt:
                    wait_time = optimal_start_dt - current_time
                    if wait_time <= max_delay:
                        return optimal_start_dt + timedelta(minutes=5), ExecutionWindow.OPTIMAL, cost_12
                else:
                    # We're in optimal window now
                    return current_time + timedelta(minutes=5), ExecutionWindow.OPTIMAL, cost_12
        
        # Check acceptable windows for today
        for window_start, window_end in self.config.acceptable_windows:
            window_start_dt = datetime.combine(current_date, window_start)
            window_end_dt = datetime.combine(current_date, window_end)
            
            if current_time < window_end_dt:
                if window_start_dt >= current_time:
                    wait_time = window_start_dt - current_time
                    if wait_time <= max_delay:
                        # Estimate cost at middle of window
                        mid_hour = (window_start.hour + window_end.hour) // 2
                        cost = self.cost_model.estimate_cost(symbol, mid_hour, size_dv_pct, urgency.value)
                        return window_start_dt + timedelta(minutes=5), ExecutionWindow.ACCEPTABLE, cost
                else:
                    # We're in this window
                    mid_hour = (current_time.hour + window_end.hour) // 2
                    cost = self.cost_model.estimate_cost(symbol, max(current_time.hour, mid_hour), size_dv_pct, urgency.value)
                    return current_time + timedelta(minutes=5), ExecutionWindow.ACCEPTABLE, cost
        
        # If no good window today, schedule for tomorrow's optimal
        tomorrow = current_date + timedelta(days=1)
        tomorrow_start = datetime.combine(tomorrow, self.config.optimal_start)
        
        # Check if delay is acceptable for low urgency
        if urgency == OrderUrgency.LOW and (tomorrow_start - current_time) <= max_delay:
            cost = self.cost_model.estimate_cost(symbol, 12, size_dv_pct, urgency.value)
            return tomorrow_start + timedelta(minutes=5), ExecutionWindow.OPTIMAL, cost
        
        # Default: execute soon in acceptable window or immediately
        if urgency == OrderUrgency.HIGH:
            return current_time + timedelta(minutes=5), ExecutionWindow.ACCEPTABLE, None
        
        # For normal urgency, try to get to tomorrow
        if urgency == OrderUrgency.NORMAL:
            # If we can wait until tomorrow morning
            tomorrow_am = datetime.combine(tomorrow, time(10, 0))
            if (tomorrow_am - current_time) <= max_delay:
                cost = self.cost_model.estimate_cost(symbol, 10, size_dv_pct, urgency.value)
                return tomorrow_am, ExecutionWindow.ACCEPTABLE, cost
        
        # Execute in next acceptable window or immediately
        return current_time + timedelta(minutes=5), ExecutionWindow.ACCEPTABLE, None
    
    def get_executable_orders(self, current_time: Optional[datetime] = None) -> List[ScheduledOrder]:
        """
        Get orders that should be executed now
        
        Args:
            current_time: Current datetime (defaults to now)
        
        Returns:
            List of orders ready for execution
        """
        if current_time is None:
            current_time = datetime.now()
        
        executable = []
        for order_id, order in list(self.pending_orders.items()):
            if order.scheduled_time and current_time >= order.scheduled_time:
                executable.append(order)
        
        return executable
    
    def mark_executed(self, order_id: str, execution_time: Optional[datetime] = None) -> None:
        """Mark an order as executed"""
        if order_id in self.pending_orders:
            order = self.pending_orders.pop(order_id)
            if execution_time:
                order.scheduled_time = execution_time
            self.executed_orders.append(order)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if order_id in self.pending_orders:
            order = self.pending_orders.pop(order_id)
            self.cancelled_orders.append(order)
            return True
        return False
    
    def get_pending_orders(self) -> List[ScheduledOrder]:
        """Get all pending orders"""
        return list(self.pending_orders.values())
    
    def get_schedule_summary(self) -> Dict[str, Any]:
        """Get summary of scheduler state"""
        return {
            'pending_count': len(self.pending_orders),
            'executed_count': len(self.executed_orders),
            'cancelled_count': len(self.cancelled_orders),
            'pending_by_window': self._count_by_window(self.pending_orders.values()),
            'executed_by_window': self._count_by_window(self.executed_orders),
            'avg_estimated_cost': self._avg_estimated_cost()
        }
    
    def _count_by_window(self, orders) -> Dict[str, int]:
        """Count orders by execution window"""
        counts = {'optimal': 0, 'acceptable': 0, 'avoid': 0, 'unknown': 0}
        for order in orders:
            if order.execution_window:
                counts[order.execution_window.value] += 1
            else:
                counts['unknown'] += 1
        return counts
    
    def _avg_estimated_cost(self) -> Optional[float]:
        """Calculate average estimated cost for pending orders"""
        costs = [o.estimated_cost_bps for o in self.pending_orders.values() if o.estimated_cost_bps]
        if costs:
            return sum(costs) / len(costs)
        return None
    
    def export_schedule(self, path: str) -> None:
        """Export schedule to JSON"""
        data = {
            'pending': [o.to_dict() for o in self.pending_orders.values()],
            'executed': [o.to_dict() for o in self.executed_orders],
            'cancelled': [o.to_dict() for o in self.cancelled_orders],
            'summary': self.get_schedule_summary()
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear_history(self) -> None:
        """Clear executed and cancelled order history"""
        self.executed_orders.clear()
        self.cancelled_orders.clear()


class BatchRebalancer:
    """
    Batch scheduling for portfolio rebalancing events
    
    Handles multiple orders with portfolio-level optimization
    """
    
    def __init__(self, scheduler: Optional[RebalanceScheduler] = None):
        self.scheduler = scheduler or RebalanceScheduler()
    
    def schedule_rebalance(
        self,
        target_allocations: Dict[str, float],
        current_holdings: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        volumes: Dict[str, float],
        urgency: str = 'normal',
        current_time: Optional[datetime] = None
    ) -> List[ScheduledOrder]:
        """
        Schedule a full portfolio rebalancing
        
        Args:
            target_allocations: Target weights by symbol
            current_holdings: Current shares by symbol
            portfolio_value: Total portfolio value
            prices: Current prices by symbol
            volumes: Average daily volumes by symbol
            urgency: Rebalancing urgency
            current_time: Current time
        
        Returns:
            List of scheduled orders
        """
        if current_time is None:
            current_time = datetime.now()
        
        orders = []
        
        for symbol, target_weight in target_allocations.items():
            target_value = portfolio_value * target_weight
            current_shares = current_holdings.get(symbol, 0)
            current_value = current_shares * prices.get(symbol, 0)
            
            diff_value = target_value - current_value
            
            if abs(diff_value) < 100:  # Skip tiny adjustments
                continue
            
            side = 'buy' if diff_value > 0 else 'sell'
            target_shares = abs(diff_value) / prices.get(symbol, 1)
            
            order_id = f"{symbol}_{current_time.strftime('%Y%m%d_%H%M%S')}"
            
            order = self.scheduler.schedule_order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                target_shares=target_shares,
                target_value=abs(diff_value),
                urgency=urgency,
                current_time=current_time,
                avg_daily_volume=volumes.get(symbol)
            )
            
            orders.append(order)
        
        return orders
