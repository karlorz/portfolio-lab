"""
Unit Tests for Rebalance Scheduler
v2.71 Intraday Seasonality Execution - Phase 2
"""

import unittest
from datetime import datetime, time, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution.rebalance_scheduler import (
    RebalanceScheduler,
    ScheduledOrder,
    SchedulerConfig,
    OrderUrgency,
    ExecutionWindow,
    BatchRebalancer
)
from src.execution.intraday_cost_model import IntradayExecutionCostModel


class TestSchedulerConfig(unittest.TestCase):
    """Test configuration loading"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = SchedulerConfig()
        
        self.assertEqual(config.optimal_start, time(11, 0))
        self.assertEqual(config.optimal_end, time(14, 0))
        self.assertEqual(config.cost_threshold_bps, 2.0)
        
        # Check acceptable windows
        self.assertEqual(len(config.acceptable_windows), 2)
        self.assertEqual(config.acceptable_windows[0], (time(10, 0), time(11, 0)))
        
        # Check delays
        self.assertEqual(config.max_delay_low, timedelta(hours=4))
        self.assertEqual(config.max_delay_normal, timedelta(hours=2))
        self.assertEqual(config.max_delay_high, timedelta(minutes=30))
        self.assertEqual(config.max_delay_urgent, timedelta(seconds=0))


class TestRebalanceScheduler(unittest.TestCase):
    """Test core scheduling logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cost_model = IntradayExecutionCostModel()
        self.scheduler = RebalanceScheduler(cost_model=self.cost_model)
    
    def test_schedule_urgent_order(self):
        """Test urgent orders execute immediately"""
        now = datetime(2026, 5, 13, 10, 0, 0)  # 10:00 AM
        
        order = self.scheduler.schedule_order(
            order_id="TEST001",
            symbol="SPY",
            side="buy",
            target_shares=100,
            target_value=50000,
            urgency="urgent",
            current_time=now
        )
        
        # Urgent orders should execute immediately
        self.assertEqual(order.scheduled_time, now)
        self.assertEqual(order.urgency, OrderUrgency.URGENT)
        self.assertIsNone(order.estimated_cost_bps)  # No cost estimate for urgent
    
    def test_schedule_low_urgency_in_optimal_window(self):
        """Test low urgency order scheduled in optimal window"""
        now = datetime(2026, 5, 13, 9, 0, 0)  # 9:00 AM - before optimal
        
        order = self.scheduler.schedule_order(
            order_id="TEST002",
            symbol="SPY",
            side="buy",
            target_shares=100,
            target_value=50000,
            urgency="low",
            current_time=now,
            avg_daily_volume=100000000
        )
        
        # Should wait for optimal window at 11:00
        self.assertIsNotNone(order.scheduled_time)
        self.assertEqual(order.scheduled_time.hour, 11)
        self.assertEqual(order.execution_window, ExecutionWindow.OPTIMAL)
        self.assertIsNotNone(order.estimated_cost_bps)
    
    def test_schedule_normal_urgency_in_optimal_window(self):
        """Test normal urgency order in optimal window"""
        now = datetime(2026, 5, 13, 12, 0, 0)  # 12:00 PM - in optimal window
        
        order = self.scheduler.schedule_order(
            order_id="TEST003",
            symbol="SPY",
            side="sell",
            target_shares=50,
            target_value=25000,
            urgency="normal",
            current_time=now
        )
        
        # Should execute soon (5 min buffer) in optimal window
        self.assertIsNotNone(order.scheduled_time)
        self.assertEqual(order.execution_window, ExecutionWindow.OPTIMAL)
    
    def test_schedule_high_urgency_minimal_delay(self):
        """Test high urgency has minimal delay"""
        now = datetime(2026, 5, 13, 9, 0, 0)  # 9:00 AM
        
        order = self.scheduler.schedule_order(
            order_id="TEST004",
            symbol="SPY",
            side="buy",
            target_shares=100,
            target_value=50000,
            urgency="high",
            current_time=now
        )
        
        # High urgency should not wait 2 hours for optimal
        delay = order.scheduled_time - now
        self.assertLess(delay, timedelta(minutes=10))
    
    def test_get_executable_orders(self):
        """Test retrieving executable orders"""
        now = datetime(2026, 5, 13, 10, 0, 0)
        
        # Schedule orders at different times
        order1 = self.scheduler.schedule_order(
            order_id="EXEC001",
            symbol="SPY",
            side="buy",
            target_shares=100,
            target_value=50000,
            urgency="urgent",
            current_time=now
        )
        
        # Should be executable immediately
        executable = self.scheduler.get_executable_orders(now + timedelta(minutes=1))
        self.assertEqual(len(executable), 1)
        self.assertEqual(executable[0].order_id, "EXEC001")
    
    def test_mark_executed(self):
        """Test marking orders as executed"""
        now = datetime(2026, 5, 13, 10, 0, 0)
        
        self.scheduler.schedule_order(
            order_id="DONE001",
            symbol="SPY",
            side="buy",
            target_shares=100,
            target_value=50000,
            urgency="urgent",
            current_time=now
        )
        
        self.assertEqual(len(self.scheduler.pending_orders), 1)
        
        self.scheduler.mark_executed("DONE001", now)
        
        self.assertEqual(len(self.scheduler.pending_orders), 0)
        self.assertEqual(len(self.scheduler.executed_orders), 1)
    
    def test_cancel_order(self):
        """Test order cancellation"""
        now = datetime(2026, 5, 13, 10, 0, 0)
        
        self.scheduler.schedule_order(
            order_id="CANCEL001",
            symbol="SPY",
            side="buy",
            target_shares=100,
            target_value=50000,
            urgency="normal",
            current_time=now
        )
        
        result = self.scheduler.cancel_order("CANCEL001")
        
        self.assertTrue(result)
        self.assertEqual(len(self.scheduler.pending_orders), 0)
        self.assertEqual(len(self.scheduler.cancelled_orders), 1)
    
    def test_cancel_nonexistent_order(self):
        """Test cancelling non-existent order fails gracefully"""
        result = self.scheduler.cancel_order("NONEXISTENT")
        self.assertFalse(result)
    
    def test_schedule_summary(self):
        """Test schedule summary generation"""
        now = datetime(2026, 5, 13, 10, 0, 0)
        
        # Add some orders
        self.scheduler.schedule_order(
            order_id="SUM001",
            symbol="SPY",
            side="buy",
            target_shares=100,
            target_value=50000,
            urgency="low",
            current_time=now
        )
        
        summary = self.scheduler.get_schedule_summary()
        
        self.assertEqual(summary['pending_count'], 1)
        self.assertEqual(summary['executed_count'], 0)
        self.assertEqual(summary['cancelled_count'], 0)
        self.assertIn('pending_by_window', summary)
    
    def test_cost_estimates_differ_by_time(self):
        """Test that cost estimates vary by time of day"""
        spy = "SPY"
        size_pct = 0.01  # 1% of DV
        
        # Costs should be different at different times
        cost_9 = self.cost_model.estimate_cost(spy, 9, size_pct, "normal")
        cost_12 = self.cost_model.estimate_cost(spy, 12, size_pct, "normal")
        cost_16 = self.cost_model.estimate_cost(spy, 16, size_pct, "normal")
        
        # Midday (optimal) should be cheaper than open/close
        self.assertLess(cost_12.total_cost_bps, cost_9.total_cost_bps)
        self.assertLess(cost_12.total_cost_bps, cost_16.total_cost_bps)
        
        # Check recommendations
        self.assertEqual(cost_12.recommended_window, 'optimal')
        self.assertEqual(cost_9.recommended_window, 'avoid')


class TestBatchRebalancer(unittest.TestCase):
    """Test batch rebalancing functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scheduler = RebalanceScheduler()
        self.batch_rebalancer = BatchRebalancer(self.scheduler)
    
    def test_schedule_rebalance(self):
        """Test scheduling a full portfolio rebalance"""
        now = datetime(2026, 5, 13, 10, 0, 0)
        
        target_allocations = {
            "SPY": 0.46,
            "GLD": 0.38,
            "TLT": 0.16
        }
        
        current_holdings = {
            "SPY": 100,  # Need ~222 shares for 46%
            "GLD": 100,
            "TLT": 100
        }
        
        portfolio_value = 100000
        
        prices = {
            "SPY": 500.0,
            "GLD": 230.0,
            "TLT": 90.0
        }
        
        volumes = {
            "SPY": 100000000,
            "GLD": 10000000,
            "TLT": 20000000
        }
        
        orders = self.batch_rebalancer.schedule_rebalance(
            target_allocations=target_allocations,
            current_holdings=current_holdings,
            portfolio_value=portfolio_value,
            prices=prices,
            volumes=volumes,
            urgency="normal",
            current_time=now
        )
        
        # Should create orders for allocations that need adjustment
        self.assertGreater(len(orders), 0)
        
        # Check that orders are in scheduler
        self.assertEqual(len(self.scheduler.pending_orders), len(orders))
    
    def test_skip_tiny_adjustments(self):
        """Test that tiny adjustments are skipped"""
        now = datetime(2026, 5, 13, 10, 0, 0)
        
        target_allocations = {"SPY": 0.50}
        current_holdings = {"SPY": 100}
        portfolio_value = 100000
        prices = {"SPY": 500.0}  # Current value = $50,000 = 50% - exact
        volumes = {"SPY": 100000000}
        
        orders = self.batch_rebalancer.schedule_rebalance(
            target_allocations=target_allocations,
            current_holdings=current_holdings,
            portfolio_value=portfolio_value,
            prices=prices,
            volumes=volumes,
            urgency="normal",
            current_time=now
        )
        
        # Should skip tiny adjustments
        self.assertEqual(len(orders), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for scheduler"""
    
    def test_end_to_end_scheduling(self):
        """Test full scheduling workflow"""
        scheduler = RebalanceScheduler()
        now = datetime(2026, 5, 13, 9, 0, 0)  # Before market open equivalent
        
        # Schedule low urgency order
        order = scheduler.schedule_order(
            order_id="E2E001",
            symbol="SPY",
            side="buy",
            target_shares=100,
            target_value=50000,
            urgency="low",
            current_time=now,
            avg_daily_volume=100000000
        )
        
        # Should be scheduled for optimal window
        self.assertIsNotNone(order.scheduled_time)
        self.assertEqual(order.execution_window, ExecutionWindow.OPTIMAL)
        
        # At 11:05, should be executable
        executable = scheduler.get_executable_orders(
            datetime(2026, 5, 13, 11, 5, 0)
        )
        self.assertEqual(len(executable), 1)
        
        # Mark as executed
        scheduler.mark_executed("E2E001")
        self.assertEqual(len(scheduler.pending_orders), 0)
        
        # Check summary
        summary = scheduler.get_schedule_summary()
        self.assertEqual(summary['executed_count'], 1)


if __name__ == '__main__':
    unittest.main()
