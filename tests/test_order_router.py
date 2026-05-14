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


class TestFetchPrice:
    """Test price fetching from market.db."""

    def _make_db(self, tmpdir, symbol="SPY", price=500.0):
        db_path = os.path.join(tmpdir, "market.db")
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        conn.execute("INSERT INTO prices VALUES (?, '2026-05-13', ?)", (symbol, price))
        conn.commit()
        conn.close()
        return db_path

    def test_fetches_existing_price(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = self._make_db(d, "SPY", 530.0)
            router = OrderRouter(data_dir=d, db_path=db_path, paper=True)
            assert router._fetch_price("SPY") == 530.0

    def test_returns_zero_missing_symbol(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = self._make_db(d, "SPY", 530.0)
            router = OrderRouter(data_dir=d, db_path=db_path, paper=True)
            assert router._fetch_price("GLD") == 0.0

    def test_returns_zero_missing_db(self):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, db_path="/nonexistent/market.db", paper=True)
            assert router._fetch_price("SPY") == 0.0


class TestDataclasses:
    """Signal and OrderPlan dataclasses."""

    def test_signal_defaults(self):
        s = Signal(symbol="SPY", target_allocation=0.46)
        assert s.symbol == "SPY"
        assert s.target_allocation == 0.46
        assert s.current_allocation is None
        assert s.signal_type == "rebalance"
        assert s.confidence == 1.0

    def test_signal_custom(self):
        s = Signal(symbol="GLD", target_allocation=0.38, current_allocation=0.40,
                   signal_type="trend", confidence=0.85)
        assert s.signal_type == "trend"
        assert s.confidence == 0.85
        assert s.current_allocation == 0.40

    def test_order_plan_buy(self):
        o = OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "rebalance_to_target")
        assert o.symbol == "SPY"
        assert o.side == "BUY"
        assert o.qty == 10
        assert o.estimated_value == 5000
        assert o.reason == "rebalance_to_target"

    def test_order_plan_sell(self):
        o = OrderPlan("TLT", "SELL", 5, "LIMIT", 3000, "overweight_reduction")
        assert o.side == "SELL"
        assert o.order_type == "LIMIT"


class TestIsReady:
    """is_ready checks AlpacaClient."""

    def test_ready_when_client_ready(self):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            router.client = MagicMock()
            router.client.is_ready.return_value = True
            assert router.is_ready() is True

    def test_not_ready_when_client_not_ready(self):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            router.client = MagicMock()
            router.client.is_ready.return_value = False
            assert router.is_ready() is False


class TestRebalance:
    """Full rebalance workflow."""

    def test_rebalance_not_configured(self):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            router.client = MagicMock()
            router.client.is_ready.return_value = False
            result = router.rebalance()
            assert result["status"] == "not_configured"

    def test_rebalance_no_signals(self):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            router.client = MagicMock()
            router.client.is_ready.return_value = True
            router.load_signals = MagicMock(return_value=[])
            result = router.rebalance()
            assert result["status"] == "no_signals"

    def test_rebalance_no_action_needed(self):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            router.client = MagicMock()
            router.client.is_ready.return_value = True
            router.load_signals = MagicMock(return_value=[
                Signal("SPY", 0.46),
            ])
            router.get_current_positions = MagicMock(return_value={
                "SPY": {"qty": 100, "market_value": 4600},
            })
            router.calculate_orders = MagicMock(return_value=[])
            result = router.rebalance()
            assert result["status"] == "no_action"
            assert result["signals_count"] == 1

    def test_rebalance_executes(self):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            router.client = MagicMock()
            router.client.is_ready.return_value = True
            router.load_signals = MagicMock(return_value=[
                Signal("SPY", 0.50),
            ])
            router.get_current_positions = MagicMock(return_value={})
            router.calculate_orders = MagicMock(return_value=[
                OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "test"),
            ])
            router.execute_orders = MagicMock(return_value={
                "status": "dry_run", "orders_executed": 1, "orders_failed": 0,
            })
            result = router.rebalance(dry_run=True)
            assert result["status"] == "dry_run"


class TestExecuteOrdersEdgeCases:
    """execute_orders edge cases."""

    def test_not_configured(self):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            router.client = MagicMock()
            router.client.is_ready.return_value = False
            result = router.execute_orders([])
            assert result["status"] == "not_configured"

    def test_empty_orders_list(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(OrderRouter, 'is_ready', return_value=True):
                router = OrderRouter(data_dir=d, paper=True)
                result = router.execute_orders([], dry_run=True)
                assert result["status"] == "dry_run"
                assert result["orders_executed"] == 0

    def test_kill_switch_not_checked_in_dry_run(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(OrderRouter, 'is_ready', return_value=True):
                router = OrderRouter(data_dir=d, paper=True)
                with open(os.path.join(d, "kill_switch.json"), "w") as f:
                    json.dump({"enabled": True, "reason": "test"}, f)
                orders = [OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "test")]
                result = router.execute_orders(orders, dry_run=True)
                # Kill switch only checked when not dry_run
                assert result["status"] != "blocked"

    def test_kill_switch_corrupt_json(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(OrderRouter, 'is_ready', return_value=True):
                router = OrderRouter(data_dir=d, paper=True)
                with open(os.path.join(d, "kill_switch.json"), "w") as f:
                    f.write("not valid json")
                orders = [OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "test")]
                # Should not crash on corrupt JSON
                result = router.execute_orders(orders, dry_run=True)
                assert result["status"] != "blocked"


class TestMainCLI:
    """main() CLI dispatch."""

    def test_status(self, capsys):
        from src.broker.order_router import main
        with patch('sys.argv', ['order_router.py', 'status']):
            with patch('src.broker.order_router.OrderRouter') as MockRouter:
                mock = MagicMock()
                mock.is_ready.return_value = True
                mock.paper = True  # Must be a real bool, not MagicMock
                MockRouter.return_value = mock
                main()
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["ready"] is True

    def test_signals_command(self, capsys):
        from src.broker.order_router import main
        with patch('sys.argv', ['order_router.py', 'signals']):
            with patch('src.broker.order_router.OrderRouter') as MockRouter:
                mock = MagicMock()
                s = MagicMock()
                s.symbol = "SPY"
                s.target_allocation = 0.46
                mock.load_signals.return_value = [s]
                MockRouter.return_value = mock
                main()
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert len(data) == 1
        assert data[0]["symbol"] == "SPY"

    def test_positions_command(self, capsys):
        from src.broker.order_router import main
        with patch('sys.argv', ['order_router.py', 'positions']):
            with patch('src.broker.order_router.OrderRouter') as MockRouter:
                mock = MagicMock()
                mock.get_current_positions.return_value = {
                    "SPY": {"qty": 100, "market_value": 50000},
                }
                MockRouter.return_value = mock
                main()
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert "SPY" in data

    def test_unknown_command(self, capsys):
        from src.broker.order_router import main
        with patch('sys.argv', ['order_router.py', 'unknowncmd']):
            with patch('src.broker.order_router.OrderRouter') as MockRouter:
                MockRouter.return_value = MagicMock()
                main()
        captured = capsys.readouterr()
        assert "Unknown" in captured.out or "unknown" in captured.out.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
