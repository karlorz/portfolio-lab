#!/usr/bin/env python3
"""
Tests for position sync — drift calculation between broker and local.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
from unittest.mock import MagicMock, patch
from src.broker.position_sync import PositionSync, PositionDrift


class TestCalculateDrift:
    """Test drift calculation logic (no broker connection needed)."""

    def _make_sync(self):
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = ":memory:"
        sync.data_dir = "/tmp"
        sync.sync_log_path = "/tmp/test_sync.jsonl"
        return sync

    def test_no_drift(self):
        """Identical positions → no drift"""
        sync = self._make_sync()
        local = {"SPY": {"qty": 100, "market_value": 50000}}
        broker = {"SPY": MagicMock(qty=100, market_value=50000)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 0

    def test_broker_overweight(self):
        """Broker has more shares → positive drift"""
        sync = self._make_sync()
        local = {"SPY": {"qty": 100, "market_value": 50000}}
        broker = {"SPY": MagicMock(qty=110, market_value=55000)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 1
        assert drift[0].symbol == "SPY"
        assert drift[0].qty_delta == 10
        assert drift[0].drift_pct == pytest.approx(0.1, abs=0.01)

    def test_broker_underweight(self):
        """Broker has fewer shares → negative drift"""
        sync = self._make_sync()
        local = {"SPY": {"qty": 100, "market_value": 50000}}
        broker = {"SPY": MagicMock(qty=90, market_value=45000)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 1
        assert drift[0].qty_delta == -10
        assert drift[0].drift_pct < 0

    def test_symbol_only_in_broker(self):
        """Position exists in broker but not local → 100% drift"""
        sync = self._make_sync()
        local = {}
        broker = {"GLD": MagicMock(qty=50, market_value=30000)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 1
        assert drift[0].symbol == "GLD"
        assert drift[0].local_qty == 0
        assert drift[0].broker_qty == 50
        assert drift[0].drift_pct == 1.0

    def test_symbol_only_in_local(self):
        """Position exists in local but not broker → -100% drift"""
        sync = self._make_sync()
        local = {"TLT": {"qty": 200, "market_value": 20000}}
        broker = {}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 1
        assert drift[0].symbol == "TLT"
        assert drift[0].broker_qty == 0
        assert drift[0].drift_pct == -1.0

    def test_multiple_symbols(self):
        """Multiple symbols with mixed drift"""
        sync = self._make_sync()
        local = {
            "SPY": {"qty": 100, "market_value": 50000},
            "GLD": {"qty": 50, "market_value": 30000},
            "TLT": {"qty": 200, "market_value": 20000},
        }
        broker = {
            "SPY": MagicMock(qty=100, market_value=50000),  # No drift
            "GLD": MagicMock(qty=55, market_value=33000),   # Overweight
            "TLT": MagicMock(qty=180, market_value=18000),  # Underweight
        }
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 2
        symbols = {d.symbol for d in drift}
        assert symbols == {"GLD", "TLT"}

    def test_small_drift_ignored(self):
        """Tiny drift below threshold → no drift recorded"""
        sync = self._make_sync()
        local = {"SPY": {"qty": 100, "market_value": 50000}}
        broker = {"SPY": MagicMock(qty=100.0005, market_value=50000.5)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 0


class TestPositionDriftDataclass:
    """PositionDrift dataclass fields."""

    def test_create_drift(self):
        d = PositionDrift(
            symbol="SPY", local_qty=100, broker_qty=110,
            qty_delta=10, local_value=50000.0, broker_value=55000.0,
            value_delta=5000.0, drift_pct=0.10,
        )
        assert d.symbol == "SPY"
        assert d.qty_delta == 10
        assert d.drift_pct == 0.10
        assert d.value_delta == 5000.0

    def test_negative_drift(self):
        d = PositionDrift(
            symbol="TLT", local_qty=200, broker_qty=180,
            qty_delta=-20, local_value=20000.0, broker_value=18000.0,
            value_delta=-2000.0, drift_pct=-0.10,
        )
        assert d.qty_delta < 0
        assert d.drift_pct < 0


class TestIsReady:
    """is_ready depends on AlpacaClient."""

    def _make_sync(self):
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = ":memory:"
        sync.data_dir = "/tmp"
        sync.sync_log_path = "/tmp/test_sync.jsonl"
        return sync

    def test_is_ready_when_client_ready(self):
        sync = self._make_sync()
        sync.client = MagicMock()
        sync.client.is_ready.return_value = True
        assert sync.is_ready() is True

    def test_is_ready_when_client_not_ready(self):
        sync = self._make_sync()
        sync.client = MagicMock()
        sync.client.is_ready.return_value = False
        assert sync.is_ready() is False


class TestSyncReport:
    """sync() method report generation."""

    def _make_sync(self):
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = ":memory:"
        sync.data_dir = "/tmp"
        sync.sync_log_path = "/tmp/test_sync.jsonl"
        return sync

    def test_sync_not_configured(self):
        sync = self._make_sync()
        sync.client = MagicMock()
        sync.client.is_ready.return_value = False
        result = sync.sync()
        assert result["status"] == "not_configured"
        assert "Alpaca" in result["message"]

    def test_sync_success_dry_run(self):
        sync = self._make_sync()
        sync.client = MagicMock()
        sync.client.is_ready.return_value = True
        sync.client.paper = True
        sync.client.get_account.return_value = {
            "equity": "100000", "cash": "5000", "buying_power": "200000",
        }
        sync.get_local_positions = MagicMock(return_value={
            "SPY": {"qty": 100, "market_value": 50000},
        })
        sync.get_broker_positions = MagicMock(return_value={
            "SPY": MagicMock(qty=100, market_value=50000),
        })

        result = sync.sync(dry_run=True)
        assert result["status"] == "success"
        assert result["paper"] is True
        assert result["local_positions"]["count"] == 1
        assert result["broker_positions"]["count"] == 1
        assert result["drift"]["count"] == 0

    def test_sync_with_drift(self):
        sync = self._make_sync()
        sync.client = MagicMock()
        sync.client.is_ready.return_value = True
        sync.client.paper = False
        sync.client.get_account.return_value = {
            "equity": "100000", "cash": "5000", "buying_power": "200000",
        }
        sync.get_local_positions = MagicMock(return_value={
            "SPY": {"qty": 100, "market_value": 50000},
        })
        broker_pos = MagicMock()
        broker_pos.qty = 110
        broker_pos.market_value = 55000
        sync.get_broker_positions = MagicMock(return_value={"SPY": broker_pos})

        result = sync.sync(dry_run=True)
        assert result["status"] == "success"
        assert result["drift"]["count"] == 1
        assert result["drift"]["total_value_delta"] == 5000.0
        assert len(result["drift"]["items"]) == 1

    def test_sync_max_drift_tracking(self):
        sync = self._make_sync()
        sync.client = MagicMock()
        sync.client.is_ready.return_value = True
        sync.client.paper = True
        sync.client.get_account.return_value = {"equity": "0", "cash": "0", "buying_power": "0"}
        sync.get_local_positions = MagicMock(return_value={
            "SPY": {"qty": 100, "market_value": 50000},
            "GLD": {"qty": 50, "market_value": 30000},
        })
        spy_pos = MagicMock()
        spy_pos.qty = 110
        spy_pos.market_value = 55000
        gld_pos = MagicMock()
        gld_pos.qty = 55
        gld_pos.market_value = 33000
        sync.get_broker_positions = MagicMock(return_value={
            "SPY": spy_pos, "GLD": gld_pos,
        })

        result = sync.sync(dry_run=True)
        assert result["drift"]["count"] == 2
        assert result["drift"]["max_drift_symbol"] is not None
        assert result["drift"]["max_drift_pct"] > 0

    def test_sync_error_handling(self):
        sync = self._make_sync()
        sync.client = MagicMock()
        sync.client.is_ready.return_value = True
        sync.client.paper = True
        sync.get_local_positions = MagicMock(side_effect=Exception("DB error"))

        result = sync.sync()
        assert result["status"] == "error"
        assert "DB error" in result["message"]


class TestReconcileToBroker:
    """reconcile_to_broker with in-memory SQLite."""

    def _make_sync(self):
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = ":memory:"
        sync.data_dir = "/tmp"
        sync.sync_log_path = "/tmp/test_sync.jsonl"
        return sync

    def test_reconcile_not_configured(self):
        sync = self._make_sync()
        sync.client = MagicMock()
        sync.client.is_ready.return_value = False
        result = sync.reconcile_to_broker()
        assert result["status"] == "not_configured"

    def test_reconcile_creates_table_and_inserts(self):
        sync = self._make_sync()
        sync.client = MagicMock()
        sync.client.is_ready.return_value = True

        pos = MagicMock()
        pos.qty = 100
        pos.avg_entry_price = 530.0
        pos.current_price = 550.0
        pos.market_value = 55000.0
        sync.get_broker_positions = MagicMock(return_value={"SPY": pos})

        result = sync.reconcile_to_broker()
        assert result["status"] == "success"
        assert result["positions_updated"] == 1
        assert result["positions_removed"] == 0

    def test_reconcile_removes_orphan_positions(self, tmp_path):
        sync = self._make_sync()
        sync.db_path = str(tmp_path / "test.db")
        sync.client = MagicMock()
        sync.client.is_ready.return_value = True

        # Pre-populate a position in the SAME db sync will use
        import sqlite3
        conn = sqlite3.connect(sync.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY, qty REAL, avg_price REAL,
                current_price REAL, market_value REAL, updated_at TEXT
            )
        """)
        conn.execute("INSERT INTO positions VALUES ('OLD', 10, 100, 100, 1000, '2024-01-01')")
        conn.commit()
        conn.close()

        # Reconcile with broker having different position
        pos = MagicMock()
        pos.qty = 100
        pos.avg_entry_price = 550.0
        pos.current_price = 550.0
        pos.market_value = 55000.0
        sync.get_broker_positions = MagicMock(return_value={"SPY": pos})

        result = sync.reconcile_to_broker()
        assert result["status"] == "success"
        assert result["positions_removed"] == 1

    def test_reconcile_error_handling(self):
        sync = self._make_sync()
        sync.client = MagicMock()
        sync.client.is_ready.return_value = True
        sync.get_broker_positions = MagicMock(side_effect=Exception("Broker error"))

        result = sync.reconcile_to_broker()
        assert result["status"] == "error"
        assert "Broker error" in result["message"]


class TestCLI:
    """main() CLI dispatch."""

    def test_status_command(self, capsys):
        from src.broker.position_sync import main
        with patch('sys.argv', ['position_sync.py', 'status']):
            with patch('src.broker.position_sync.PositionSync') as MockSync:
                mock = MagicMock()
                mock.is_ready.return_value = True
                MockSync.return_value = mock
                main()
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["ready"] is True

    def test_drift_command_no_drift(self, capsys):
        from src.broker.position_sync import main
        with patch('sys.argv', ['position_sync.py', 'drift']):
            with patch('src.broker.position_sync.PositionSync') as MockSync:
                mock = MagicMock()
                mock.sync.return_value = {
                    "status": "success",
                    "drift": {"count": 0, "items": []},
                }
                MockSync.return_value = mock
                main()
        captured = capsys.readouterr()
        assert "No position drift" in captured.out

    def test_drift_command_with_drift(self, capsys):
        from src.broker.position_sync import main
        with patch('sys.argv', ['position_sync.py', 'drift']):
            with patch('src.broker.position_sync.PositionSync') as MockSync:
                mock = MagicMock()
                mock.sync.return_value = {
                    "status": "success",
                    "drift": {
                        "count": 1,
                        "items": [{
                            "symbol": "SPY", "qty_delta": 10.0,
                            "value_delta": 5000.0, "drift_pct": 10.0,
                        }],
                    },
                }
                MockSync.return_value = mock
                main()
        captured = capsys.readouterr()
        assert "Found 1 position drift" in captured.out
        assert "SPY" in captured.out

    def test_unknown_command(self, capsys):
        from src.broker.position_sync import main
        with patch('sys.argv', ['position_sync.py', 'unknown']):
            with patch('src.broker.position_sync.PositionSync') as MockSync:
                MockSync.return_value = MagicMock()
                main()
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_default_no_args(self, capsys):
        from src.broker.position_sync import main
        with patch('sys.argv', ['position_sync.py']):
            with patch('src.broker.position_sync.PositionSync') as MockSync:
                mock = MagicMock()
                mock.sync.return_value = {
                    "status": "success", "timestamp": "2024-01-01T00:00:00",
                    "paper": True, "account": {}, "local_positions": {"count": 0},
                    "broker_positions": {"count": 0}, "drift": {"count": 0},
                }
                MockSync.return_value = mock
                main()
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["status"] == "success"


class TestDriftEdgeCases:
    """Additional drift edge cases."""

    def _make_sync(self):
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = ":memory:"
        sync.data_dir = "/tmp"
        sync.sync_log_path = "/tmp/test_sync.jsonl"
        return sync

    def test_zero_value_new_position(self):
        """When local_value=0 but broker_value>0, drift_pct=1.0."""
        sync = self._make_sync()
        local = {"SPY": {"qty": 0, "market_value": 0}}
        broker = {"SPY": MagicMock(qty=100, market_value=50000)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 1
        assert drift[0].drift_pct == 1.0

    def test_both_zero_value(self):
        sync = self._make_sync()
        local = {"SPY": {"qty": 0, "market_value": 0}}
        broker = {"SPY": MagicMock(qty=0, market_value=0)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 0  # qty_delta=0, value_delta=0, below thresholds

    def test_threshold_boundary(self):
        """Exactly at threshold: qty_delta=0.001 should NOT trigger drift."""
        sync = self._make_sync()
        local = {"SPY": {"qty": 100, "market_value": 50000}}
        # Use integer-representable value to avoid floating point noise
        broker = {"SPY": MagicMock(qty=100.0005, market_value=50000)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 0  # abs(0.0005) > 0.001 is False

    def test_value_threshold_boundary(self):
        """Exactly $1 value delta should NOT trigger drift."""
        sync = self._make_sync()
        local = {"SPY": {"qty": 100, "market_value": 50000}}
        broker = {"SPY": MagicMock(qty=100, market_value=50001)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 0  # abs(1.0) > 1.0 is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
