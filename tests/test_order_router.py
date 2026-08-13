#!/usr/bin/env python3
"""
Tests for order router — signal-to-order conversion, kill switch, retry logic.
"""
import os
import json
import tempfile

import pytest
from unittest.mock import patch, MagicMock

from src.broker.order_router import OrderRouter, Signal, OrderPlan
from src.broker.circuit_breaker import broker_breaker


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset the circuit breaker singleton between tests to prevent state leakage."""
    broker_breaker.reset()
    yield
    broker_breaker.reset()


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
                lines = [json.loads(item) for item in f.readlines()]
            assert len(lines) == 2
            assert all(item["status"] == "dry_run" for item in lines)


class TestMarketSessionGuard:
    """Live order submission should respect market-session policy."""

    @patch.object(OrderRouter, 'is_ready', return_value=True)
    def test_closed_market_blocks_live_orders_before_submit(self, mock_ready):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=False)
            market_session = {
                "session_state": "closed",
                "is_open": False,
                "timestamp": "2026-06-13T16:00:00+00:00",
                "next_open": "2026-06-15T13:30:00+00:00",
                "next_close": "2026-06-12T20:00:00+00:00",
                "extended_hours_allowed": False,
                "allow_live_orders": False,
                "guard_decision": "reject",
                "reason": "market_closed",
            }
            router.client = MagicMock()
            router.client.get_market_session.return_value = market_session

            orders = [OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "test")]
            result = router.execute_orders(orders, dry_run=False, kill_switch_check=False)

        assert result["status"] == "blocked"
        assert "market_closed" in result["message"]
        assert result["market_session"] == market_session
        router.client.submit_order.assert_not_called()

    @patch.object(OrderRouter, 'is_ready', return_value=True)
    def test_extended_hours_allowed_can_submit_live_order(self, mock_ready):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=False)
            market_session = {
                "session_state": "extended_hours",
                "is_open": False,
                "timestamp": "2026-06-12T12:30:00+00:00",
                "next_open": "2026-06-12T13:30:00+00:00",
                "next_close": "2026-06-12T20:00:00+00:00",
                "extended_hours_allowed": True,
                "allow_live_orders": True,
                "guard_decision": "allow",
                "reason": None,
            }
            submitted = MagicMock()
            submitted.id = "order-1"
            submitted.status = "accepted"
            router.client = MagicMock()
            router.client.get_market_session.return_value = market_session
            router.client.submit_order.return_value = submitted

            orders = [OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "test")]
            with patch("src.broker.order_router.time.sleep", return_value=None):
                result = router.execute_orders(orders, dry_run=False, kill_switch_check=False)

        assert result["status"] == "completed"
        assert result["orders_executed"] == 1
        assert result["executed"][0]["market_session"] == market_session
        router.client.submit_order.assert_called_once()


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

    def test_load_signals_consumes_exact_target_allocations(self):
        with tempfile.TemporaryDirectory() as d:
            signals_file = os.path.join(d, "signals.json")
            with open(signals_file, "w") as f:
                json.dump(
                    {"target_allocations": {"SPY": 0.30, "GLD": 0.45, "TLT": 0.25}},
                    f,
                )

            router = OrderRouter(signals_file=signals_file, data_dir=d)
            loaded = {
                signal.symbol: signal.target_allocation
                for signal in router.load_signals()
            }

            assert loaded == {"SPY": 0.30, "GLD": 0.45, "TLT": 0.25}

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
                # Should not crash on corrupt JSON (dry run skips kill switch)
                result = router.execute_orders(orders, dry_run=True)
                assert result["status"] != "blocked"

    def test_kill_switch_corrupt_json_blocks_live(self):
        """Corrupt kill switch JSON must block live orders (fail-closed)."""
        with tempfile.TemporaryDirectory() as d:
            with patch.object(OrderRouter, 'is_ready', return_value=True):
                router = OrderRouter(data_dir=d, paper=True)
                with open(os.path.join(d, "kill_switch.json"), "w") as f:
                    f.write("not valid json")
                orders = [OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "test")]
                result = router.execute_orders(orders, dry_run=False)
                assert result["status"] == "blocked"


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
        err = captured.err.strip()
        data = json.loads(err[err.index("{"):])
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
        err = captured.err.strip()
        data = json.loads(err[err.index("["):])
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
        err = captured.err.strip()
        data = json.loads(err[err.index("{"):])
        assert "SPY" in data

    def test_unknown_command(self, capsys):
        from src.broker.order_router import main
        with patch('sys.argv', ['order_router.py', 'unknowncmd']):
            with patch('src.broker.order_router.OrderRouter') as MockRouter:
                MockRouter.return_value = MagicMock()
                main()
        captured = capsys.readouterr()
        assert "Unknown" in captured.err or "unknown" in captured.err.lower()


class TestSignalExtended:
    """Extended tests for Signal dataclass."""

    def test_signal_all_fields(self):
        s = Signal(symbol="TLT", target_allocation=0.16,
                   current_allocation=0.18, signal_type="vix", confidence=0.7)
        assert s.symbol == "TLT"
        assert s.target_allocation == 0.16
        assert s.current_allocation == 0.18
        assert s.signal_type == "vix"
        assert s.confidence == 0.7

    def test_signal_zero_allocation(self):
        s = Signal(symbol="IEF", target_allocation=0.0)
        assert s.target_allocation == 0.0

    def test_signal_default_confidence(self):
        s = Signal(symbol="SPY", target_allocation=0.5)
        assert s.confidence == 1.0


class TestOrderPlanExtended:
    """Extended tests for OrderPlan dataclass."""

    def test_order_plan_all_fields(self):
        o = OrderPlan("GLD", "BUY", 25.5, "LIMIT", 7500, "trend_follow")
        assert o.symbol == "GLD"
        assert o.side == "BUY"
        assert o.qty == 25.5
        assert o.order_type == "LIMIT"
        assert o.estimated_value == 7500
        assert o.reason == "trend_follow"

    def test_order_plan_market_type(self):
        o = OrderPlan("SPY", "SELL", 100, "MARKET", 50000, "rebalance")
        assert o.order_type == "MARKET"

    def test_order_plan_zero_qty(self):
        o = OrderPlan("SPY", "BUY", 0, "MARKET", 0, "test")
        assert o.qty == 0


class TestCalculateOrdersExtended:
    """Extended tests for calculate_orders."""

    def _make_router(self, tmpdir):
        return OrderRouter(
            signals_file=os.path.join(tmpdir, "signals.json"),
            data_dir=tmpdir,
            paper=True,
            min_order_value=10.0,
        )

    def test_multiple_signals_multiple_positions(self):
        with tempfile.TemporaryDirectory() as d:
            router = self._make_router(d)
            signals = [
                Signal(symbol="SPY", target_allocation=0.46),
                Signal(symbol="GLD", target_allocation=0.38),
                Signal(symbol="TLT", target_allocation=0.16),
            ]
            positions = {
                "SPY": {"qty": 50, "market_value": 25000},
                "GLD": {"qty": 30, "market_value": 6000},
            }
            orders = router.calculate_orders(signals, positions, total_value=100000)
            symbols = {o.symbol for o in orders}
            assert "SPY" in symbols
            assert "GLD" in symbols
            assert "TLT" in symbols

    def test_zero_total_value(self):
        with tempfile.TemporaryDirectory() as d:
            router = self._make_router(d)
            signals = [Signal(symbol="SPY", target_allocation=0.50)]
            orders = router.calculate_orders(signals, {}, total_value=0)
            assert len(orders) == 0

    def test_exact_target_no_order(self):
        """Position at exact target → no order (within min_order_value)."""
        with tempfile.TemporaryDirectory() as d:
            router = self._make_router(d)
            signals = [Signal(symbol="SPY", target_allocation=0.50)]
            positions = {"SPY": {"qty": 100, "market_value": 5000}}
            orders = router.calculate_orders(signals, positions, total_value=10000)
            assert len(orders) == 0

    def test_custom_min_order_value(self):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True, min_order_value=1000.0)
            signals = [Signal(symbol="SPY", target_allocation=0.50)]
            positions = {"SPY": {"qty": 100, "market_value": 4900}}
            orders = router.calculate_orders(signals, positions, total_value=10000)
            # $100 drift < $1000 minimum
            assert len(orders) == 0


class TestLoadSignalsExtended:
    """Extended signal loading tests."""

    def test_load_signals_with_extra_fields(self):
        with tempfile.TemporaryDirectory() as d:
            signals_file = os.path.join(d, "signals.json")
            with open(signals_file, "w") as f:
                json.dump({
                    "target_allocations": {"SPY": 0.46, "GLD": 0.38},
                    "metadata": {"generated_at": "2026-05-14"},
                }, f)
            router = OrderRouter(signals_file=signals_file, data_dir=d)
            signals = router.load_signals()
            assert len(signals) == 2

    def test_load_signals_ignores_advisory_allocation_artifacts_without_target_allocations(self):
        with tempfile.TemporaryDirectory() as d:
            signals_file = os.path.join(d, "signals.json")
            with open(signals_file, "w") as f:
                json.dump({
                    "adaptive_sizing": {
                        "adjusted_allocation": {"SPY": 1.0},
                        "routed": False,
                    },
                    "black_litterman": {
                        "posterior_weights": {"GLD": 1.0},
                        "routed": False,
                    },
                }, f)
            router = OrderRouter(signals_file=signals_file, data_dir=d)
            signals = router.load_signals()
            assert signals == []

    def test_load_signals_empty_allocations(self):
        with tempfile.TemporaryDirectory() as d:
            signals_file = os.path.join(d, "signals.json")
            with open(signals_file, "w") as f:
                json.dump({"target_allocations": {}}, f)
            router = OrderRouter(signals_file=signals_file, data_dir=d)
            signals = router.load_signals()
            assert len(signals) == 0

    def test_load_signals_corrupt_json(self):
        with tempfile.TemporaryDirectory() as d:
            signals_file = os.path.join(d, "signals.json")
            with open(signals_file, "w") as f:
                f.write("not valid json")
            router = OrderRouter(signals_file=signals_file, data_dir=d)
            signals = router.load_signals()
            assert signals == []


class TestRebalanceExtended:
    """Extended rebalance workflow tests."""

    def test_rebalance_dry_run_default(self):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            router.client = MagicMock()
            router.client.is_ready.return_value = True
            router.load_signals = MagicMock(return_value=[Signal("SPY", 0.50)])
            router.get_current_positions = MagicMock(return_value={})
            router.calculate_orders = MagicMock(return_value=[
                OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "test"),
            ])
            router.execute_orders = MagicMock(return_value={
                "status": "dry_run", "orders_executed": 1, "orders_failed": 0,
            })
            # Default dry_run=True
            _ = router.rebalance()
            router.execute_orders.assert_called_once()
            call_kwargs = router.execute_orders.call_args
            assert call_kwargs[1].get("dry_run", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else True) is True


# ---------------------------------------------------------------------------
# __all__ export validation
# ---------------------------------------------------------------------------

class TestExports:
    """Verify module exports."""

    def test_signal_class(self):
        assert Signal is not None

    def test_order_plan_class(self):
        assert OrderPlan is not None

    def test_order_router_class(self):
        assert OrderRouter is not None


# ---------------------------------------------------------------------------
# Signal dataclass tests
# ---------------------------------------------------------------------------

class TestSignalDataclass:
    """Test Signal dataclass."""

    def test_required_fields(self):
        s = Signal("SPY", 0.50)
        assert s.symbol == "SPY"
        assert s.target_allocation == 0.50

    def test_default_values(self):
        s = Signal("GLD", 0.30)
        assert s.current_allocation is None
        assert s.signal_type == "rebalance"
        assert s.confidence == 1.0

    def test_custom_values(self):
        s = Signal("TLT", 0.20, current_allocation=0.15, signal_type="trend", confidence=0.8)
        assert s.current_allocation == 0.15
        assert s.signal_type == "trend"
        assert s.confidence == 0.8


# ---------------------------------------------------------------------------
# OrderPlan dataclass tests
# ---------------------------------------------------------------------------

class TestOrderPlanDataclass:
    """Test OrderPlan dataclass."""

    def test_buy_plan(self):
        plan = OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "underweight")
        assert plan.symbol == "SPY"
        assert plan.side == "BUY"
        assert plan.qty == 10

    def test_sell_plan(self):
        plan = OrderPlan("GLD", "SELL", 5, "MARKET", 1000, "overweight")
        assert plan.side == "SELL"

    def test_limit_order_type(self):
        plan = OrderPlan("TLT", "BUY", 3, "LIMIT", 500, "rebalance")
        assert plan.order_type == "LIMIT"


# ---------------------------------------------------------------------------
# OrderRouter extended tests
# ---------------------------------------------------------------------------

class TestOrderRouterExtended:
    """Extended OrderRouter tests."""

    def test_is_ready_with_mock_client(self):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            router.client = MagicMock()
            router.client.is_ready.return_value = True
            assert router.is_ready() is True

    def test_is_ready_no_client(self):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            router.client = None
            # client=None causes AttributeError, so we test the behavior
            with pytest.raises(AttributeError):
                router.is_ready()

    def test_load_signals_returns_list(self):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            signals = router.load_signals()
            assert isinstance(signals, list)

    def test_get_current_positions_no_client(self):
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            router.client = None
            # client=None causes AttributeError
            with pytest.raises(AttributeError):
                router.get_current_positions()

    def test_calculate_orders_no_signal_change(self):
        """Positions matching target should produce no orders."""
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            signals = [Signal("SPY", 0.50, current_allocation=0.50)]
            positions = {"SPY": {"qty": 10, "market_value": 5000}}
            orders = router.calculate_orders(signals, positions, total_value=10000)
            # No significant drift → no order
            assert all(o.symbol != "SPY" for o in orders) or len(orders) == 0


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

class TestCLI:
    """Test main() callable."""

    def test_main_callable(self):
        from src.broker.order_router import main
        assert callable(main)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ---------------------------------------------------------------------------
# Kill Switch Gate -- comprehensive coverage
# ---------------------------------------------------------------------------

class TestKillSwitchGate:
    """Kill switch gate: every code path through the execute_orders kill switch."""

    # ------------------------------------------------------------------
    # Enabled / disabled
    # ------------------------------------------------------------------

    @patch.object(OrderRouter, 'is_ready', return_value=True)
    def test_kill_switch_enabled_blocks_orders(self, mock_ready):
        """Kill switch enabled -> execute_orders returns status=blocked."""
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            with open(os.path.join(d, "kill_switch.json"), "w") as f:
                json.dump({"enabled": True, "reason": "market_crash"}, f)

            orders = [OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "test")]
            result = router.execute_orders(orders, dry_run=False)
            assert result["status"] == "blocked"
            assert "Kill switch" in result["message"]

    @patch.object(OrderRouter, 'is_ready', return_value=True)
    def test_kill_switch_disabled_allows_orders(self, mock_ready):
        """Kill switch disabled -> orders NOT blocked by kill switch (may fail later)."""
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            with open(os.path.join(d, "kill_switch.json"), "w") as f:
                json.dump({"enabled": False}, f)

            orders = [OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "test")]
            result = router.execute_orders(orders, dry_run=False)
            assert result["status"] != "blocked"

    # ------------------------------------------------------------------
    # Corrupt / missing file
    # ------------------------------------------------------------------

    @patch.object(OrderRouter, 'is_ready', return_value=True)
    def test_corrupt_kill_switch_blocks_for_safety(self, mock_ready):
        """Corrupt kill_switch.json must block (fail-closed)."""
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            with open(os.path.join(d, "kill_switch.json"), "w") as f:
                f.write("not valid json")

            orders = [OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "test")]
            result = router.execute_orders(orders, dry_run=False)
            assert result["status"] == "blocked"
            assert "unreadable" in result["message"].lower()

    @patch.object(OrderRouter, 'is_ready', return_value=True)
    def test_kill_switch_missing_blocks_live_execution(self, mock_ready):
        """No kill_switch.json with kill_switch_check=True must fail-closed (block)."""
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            orders = [OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "test")]
            # No kill_switch.json — production expects the file present (enabled true/false)
            result = router.execute_orders(orders, dry_run=False)
            assert result["status"] == "blocked"
            assert "missing" in result["message"].lower() or "kill switch" in result["message"].lower()

    @patch.object(OrderRouter, 'is_ready', return_value=True)
    def test_kill_switch_not_checked_in_dry_run_even_when_missing(self, mock_ready):
        """dry_run=True skips kill-switch gate entirely (missing file does not block)."""
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            orders = [OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "test")]
            result = router.execute_orders(orders, dry_run=True)
            assert result["status"] != "blocked"

    # ------------------------------------------------------------------
    # Disable flag
    # ------------------------------------------------------------------

    @patch.object(OrderRouter, 'is_ready', return_value=True)
    def test_kill_switch_check_can_be_disabled(self, mock_ready):
        """kill_switch_check=False skips kill switch even when enabled."""
        with tempfile.TemporaryDirectory() as d:
            router = OrderRouter(data_dir=d, paper=True)
            with open(os.path.join(d, "kill_switch.json"), "w") as f:
                json.dump({"enabled": True, "reason": "market_crash"}, f)

            orders = [OrderPlan("SPY", "BUY", 10, "MARKET", 5000, "test")]
            result = router.execute_orders(orders, dry_run=False, kill_switch_check=False)
            # Should NOT be blocked (gate is bypassed)
            assert result["status"] != "blocked"
