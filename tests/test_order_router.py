#!/usr/bin/env python3
"""
Tests for order router — signal-to-order conversion, kill switch, retry logic.
"""
import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.broker.order_router import OrderRouter, Signal, OrderPlan


class TestCalculateOrders:
    """Test order calculation from signals + positions."""

    def _make_router(self, tmpdir):
        return OrderRouter(
            signals_file=os.path.join(tmpdir, "signals.json"),
            data_dir=tmpdir,
            paper=True,
            min_order_value=10.0,
        )

    def test_buy_underweight(self):
        """Underweight position → BUY order"""
        with tempfile.TemporaryDirectory() as d:
            router = self._make_router(d)
            signals = [Signal(symbol="SPY", target_allocation=0.50)]
            positions = {"SPY": {"qty": 10, "market_value": 4000}}
            orders = router.calculate_orders(signals, positions, total_value=10000)
            assert len(orders) == 1
            assert orders[0].side == "BUY"
            assert orders[0].symbol == "SPY"
            assert orders[0].estimated_value == pytest.approx(1000, abs=10)

    def test_sell_overweight(self):
        """Overweight position → SELL order"""
        with tempfile.TemporaryDirectory() as d:
            router = self._make_router(d)
            signals = [Signal(symbol="GLD", target_allocation=0.30)]
            positions = {"GLD": {"qty": 50, "market_value": 5000}}
            orders = router.calculate_orders(signals, positions, total_value=10000)
            assert len(orders) == 1
            assert orders[0].side == "SELL"

    def test_skip_small_drift(self):
        """Drift below min_order_value → no order"""
        with tempfile.TemporaryDirectory() as d:
            router = self._make_router(d)
            signals = [Signal(symbol="SPY", target_allocation=0.46)]
            positions = {"SPY": {"qty": 100, "market_value": 4595}}
            orders = router.calculate_orders(signals, positions, total_value=10000)
            assert len(orders) == 0

    def test_liquidate_missing_signal(self):
        """Position not in signals → liquidate order"""
        with tempfile.TemporaryDirectory() as d:
            router = self._make_router(d)
            signals = [Signal(symbol="SPY", target_allocation=0.50)]
            positions = {
                "SPY": {"qty": 50, "market_value": 5000},
                "GLD": {"qty": 20, "market_value": 4000},
            }
            orders = router.calculate_orders(signals, positions, total_value=10000)
            symbols = [o.symbol for o in orders]
            assert "GLD" in symbols
            gld_order = [o for o in orders if o.symbol == "GLD"][0]
            assert gld_order.side == "SELL"

    def test_empty_signals(self):
        """No signals → no orders"""
        with tempfile.TemporaryDirectory() as d:
            router = self._make_router(d)
            orders = router.calculate_orders([], {}, total_value=10000)
            assert len(orders) == 0

    def test_low_total_value(self):
        """Total value < $100 → no orders"""
        with tempfile.TemporaryDirectory() as d:
            router = self._make_router(d)
            signals = [Signal(symbol="SPY", target_allocation=0.50)]
            orders = router.calculate_orders(signals, {}, total_value=50)
            assert len(orders) == 0


class TestKillSwitch:
    """Test kill switch blocks execution."""

    @patch.object(OrderRouter, 'is_ready', return_value=True)
    def test_kill_switch_blocks(self, mock_ready):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            # Create kill switch file
            with open(os.path.join(d, "kill_switch.json"), "w") as f:
                json.dump({"enabled": True, "reason": "test"}, f)

            orders = [OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "test")]
            result = router.execute_orders(orders, dry_run=False)
            assert result["status"] == "blocked"
            assert "Kill switch" in result["message"]

    def test_kill_switch_disabled(self):
        """Disabled kill switch doesn't block"""
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            with open(os.path.join(d, "kill_switch.json"), "w") as f:
                json.dump({"enabled": False}, f)

            # Will fail at Alpaca API (no credentials) but shouldn't be blocked
            orders = [OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "test")]
            result = router.execute_orders(orders, dry_run=False)
            assert result["status"] != "blocked"


class TestDryRun:
    """Test dry-run mode."""

    @patch.object(OrderRouter, 'is_ready', return_value=True)
    def test_dry_run_logs_without_submitting(self, mock_ready):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            orders = [
                OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "test"),
                OrderPlan("GLD", "SELL", 5, "MARKET", 3000, "test"),
            ]
            result = router.execute_orders(orders, dry_run=True)
            assert result["status"] == "dry_run"
            assert result["orders_executed"] == 2
            assert result["orders_failed"] == 0

            # Check log file
            log_path = os.path.join(d, "broker_orders.jsonl")
            assert os.path.exists(log_path)
            with open(log_path) as f:
                lines = [json.loads(l) for l in f.readlines()]
            assert len(lines) == 2
            assert all(l["status"] == "dry_run" for l in lines)


class TestLoadSignals:
    """Test signal loading from JSON."""

    def test_load_valid_signals(self):
        with tempfile.TemporaryDirectory() as d:
            signals_file = os.path.join(d, "signals.json")
            with open(signals_file, "w") as f:
                json.dump({
                    "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
                }, f)

            router = OrderRouter(signals_file=signals_file, data_dir=d)
            signals = router.load_signals()
            assert len(signals) == 3
            assert {s.symbol for s in signals} == {"SPY", "GLD", "TLT"}

    def test_load_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(signals_file="/nonexistent/file.json", data_dir=d)
            signals = router.load_signals()
            assert signals == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
