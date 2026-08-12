#!/usr/bin/env python3
"""
Tests for position sync — drift calculation between broker and local.
"""

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
        sync.get_local_positions = MagicMock(side_effect=RuntimeError("DB error"))

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
        sync.get_broker_positions = MagicMock(side_effect=RuntimeError("Broker error"))

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
        err = captured.err.strip()
        data = json.loads(err[err.index("{"):])
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
        assert "No position drift" in captured.err

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
        assert "Found 1 position drift" in captured.err
        assert "SPY" in captured.err

    def test_unknown_command(self, capsys):
        from src.broker.position_sync import main
        with patch('sys.argv', ['position_sync.py', 'unknown']):
            with patch('src.broker.position_sync.PositionSync') as MockSync:
                MockSync.return_value = MagicMock()
                main()
        captured = capsys.readouterr()
        assert "Unknown command" in captured.err

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
        err = captured.err.strip()
        data = json.loads(err[err.index("{"):])
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


# ---------------------------------------------------------------------------
# Extended coverage tests
# ---------------------------------------------------------------------------


class TestPositionDriftDataclass:
    """Test PositionDrift dataclass."""

    def test_position_drift_creation(self):
        from src.broker.position_sync import PositionDrift
        drift = PositionDrift(
            symbol="SPY", local_qty=100, broker_qty=105,
            qty_delta=5, local_value=50000, broker_value=52500,
            value_delta=2500, drift_pct=0.05,
        )
        assert drift.symbol == "SPY"
        assert drift.qty_delta == 5
        assert drift.drift_pct == 0.05

    def test_position_drift_negative(self):
        """Negative drift should be representable."""
        from src.broker.position_sync import PositionDrift
        drift = PositionDrift(
            symbol="GLD", local_qty=200, broker_qty=190,
            qty_delta=-10, local_value=38000, broker_value=36100,
            value_delta=-1900, drift_pct=-0.05,
        )
        assert drift.qty_delta == -10
        assert drift.drift_pct == -0.05


class TestCalculateDriftExtended:
    """Extended calculate_drift edge cases."""

    def _make_sync(self):
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = ":memory:"
        sync.data_dir = "/tmp"
        sync.sync_log_path = "/tmp/test_sync.jsonl"
        return sync

    def test_multiple_symbols(self):
        """Should handle drift across multiple symbols."""
        sync = self._make_sync()
        local = {
            "SPY": {"qty": 100, "market_value": 50000},
            "GLD": {"qty": 200, "market_value": 38000},
            "TLT": {"qty": 50, "market_value": 5000},
        }
        broker = {
            "SPY": MagicMock(qty=105, market_value=52500),
            "GLD": MagicMock(qty=200, market_value=38000),
            "TLT": MagicMock(qty=50, market_value=5000),
        }
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 1  # Only SPY has drift
        assert drift[0].symbol == "SPY"

    def test_symbol_in_broker_not_local(self):
        """Symbol only in broker should show as new position."""
        sync = self._make_sync()
        local = {}
        broker = {"SPY": MagicMock(qty=100, market_value=50000)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 1
        assert drift[0].local_qty == 0
        assert drift[0].broker_qty == 100
        assert drift[0].drift_pct == 1.0

    def test_symbol_in_local_not_broker(self):
        """Symbol only in local should show negative drift."""
        sync = self._make_sync()
        local = {"SPY": {"qty": 100, "market_value": 50000}}
        broker = {}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 1
        assert drift[0].broker_qty == 0
        assert drift[0].qty_delta == -100

    def test_no_drift_when_matching(self):
        """Matching positions should produce no drift items."""
        sync = self._make_sync()
        local = {"SPY": {"qty": 100, "market_value": 50000}}
        broker = {"SPY": MagicMock(qty=100, market_value=50000)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 0

    def test_empty_both(self):
        """Empty local and broker should produce no drift."""
        sync = self._make_sync()
        drift = sync.calculate_drift({}, {})
        assert len(drift) == 0


class TestSyncExtended:
    """Extended sync method tests."""

    def _make_sync(self):
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = ":memory:"
        sync.data_dir = "/tmp"
        sync.sync_log_path = "/tmp/test_sync.jsonl"
        sync.client = MagicMock()
        return sync

    def test_sync_not_ready(self):
        """sync() when client not ready should return not_configured."""
        sync = self._make_sync()
        sync.client.is_ready.return_value = False
        result = sync.sync()
        assert result["status"] == "not_configured"

    def test_sync_dry_run_no_log(self, tmp_path):
        """dry_run=True should not write to sync log."""
        sync = self._make_sync()
        sync.client.is_ready.return_value = True
        sync.client.get_positions.return_value = []
        sync.client.get_account.return_value = {"equity": 100000, "cash": 50000, "buying_power": 200000}
        sync.sync_log_path = str(tmp_path / "sync.jsonl")
        sync.get_local_positions = lambda: {}
        sync.get_broker_positions = lambda: {}
        result = sync.sync(dry_run=True)
        assert result["status"] == "success"
        assert not (tmp_path / "sync.jsonl").exists()

    def test_sync_success_report_structure(self):
        """Successful sync should have expected report keys."""
        sync = self._make_sync()
        sync.client.is_ready.return_value = True
        sync.client.get_positions.return_value = []
        sync.client.get_account.return_value = {"equity": 100000, "cash": 50000, "buying_power": 200000}
        sync.get_local_positions = lambda: {"SPY": {"qty": 100, "market_value": 50000}}
        sync.get_broker_positions = lambda: {}
        result = sync.sync(dry_run=True)
        assert "timestamp" in result
        assert "local_positions" in result
        assert "broker_positions" in result
        assert "drift" in result

    def test_sync_error_handling(self):
        """sync() should handle exceptions gracefully."""
        sync = self._make_sync()
        sync.client.is_ready.return_value = True
        sync.client.get_account.side_effect = RuntimeError("API error")
        sync.get_local_positions = lambda: {}
        sync.get_broker_positions = lambda: {}
        result = sync.sync()
        assert result["status"] == "error"
        assert "API error" in result["message"]


class TestReconcileToBrokerExtended:
    """Extended reconcile_to_broker tests."""

    def test_reconcile_not_ready(self):
        """reconcile_to_broker() when not ready should return not_configured."""
        sync = PositionSync.__new__(PositionSync)
        sync.client = MagicMock()
        sync.client.is_ready.return_value = False
        result = sync.reconcile_to_broker()
        assert result["status"] == "not_configured"


# ---------------------------------------------------------------------------
# NEW: Constructor, get_local_positions, and get_broker_positions coverage
# ---------------------------------------------------------------------------


class TestConstructor:
    """PositionSync constructor."""

    def test_default_constructor_paths(self):
        """Default constructor uses MARKET_DB / DATA_DIR."""
        from src.paths import MARKET_DB, DATA_DIR
        sync = PositionSync.__new__(PositionSync)
        PositionSync.__init__(sync)
        assert sync.db_path == str(MARKET_DB)
        assert sync.data_dir == str(DATA_DIR)
        assert sync.sync_log_path.endswith("position_sync.jsonl")

    def test_custom_paths(self):
        """Custom db_path and data_dir are respected."""
        sync = PositionSync.__new__(PositionSync)
        PositionSync.__init__(sync, db_path="/custom/db.sqlite", data_dir="/custom/data")
        assert sync.db_path == "/custom/db.sqlite"
        assert sync.data_dir == "/custom/data"
        assert "/custom/data/position_sync.jsonl" in sync.sync_log_path


class TestGetLocalPositions:
    """get_local_positions with various DB states."""

    def test_missing_db_file(self):
        """Non-existent DB file returns empty dict."""
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = "/nonexistent/path/market.db"
        result = sync.get_local_positions()
        assert result == {}

    def test_empty_positions_table(self, tmp_path):
        """DB with no positions table returns empty dict."""
        import sqlite3
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = str(tmp_path / "empty.db")
        sqlite3.connect(sync.db_path).close()
        result = sync.get_local_positions()
        assert result == {}

    def test_all_zero_qty_filtered(self, tmp_path):
        """Positions with qty=0 should not be returned."""
        import sqlite3
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = str(tmp_path / "zeros.db")
        conn = sqlite3.connect(sync.db_path)
        conn.execute("CREATE TABLE positions (symbol TEXT PRIMARY KEY, qty REAL, avg_price REAL, current_price REAL, market_value REAL, updated_at TEXT)")
        conn.execute("INSERT INTO positions VALUES ('SPY', 0, 500, 510, 0, '2024-01-01')")
        conn.commit()
        conn.close()
        result = sync.get_local_positions()
        assert result == {}

    def test_none_values_handled(self, tmp_path):
        """NULL values in numeric columns should be converted to 0.0."""
        import sqlite3
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = str(tmp_path / "nulls.db")
        conn = sqlite3.connect(sync.db_path)
        conn.execute("CREATE TABLE positions (symbol TEXT PRIMARY KEY, qty REAL, avg_price REAL, current_price REAL, market_value REAL, updated_at TEXT)")
        conn.execute("INSERT INTO positions VALUES ('SPY', 100, NULL, NULL, NULL, NULL)")
        conn.commit()
        conn.close()
        result = sync.get_local_positions()
        assert "SPY" in result
        assert result["SPY"]["qty"] == 100.0
        assert result["SPY"]["avg_price"] == 0.0
        assert result["SPY"]["current_price"] == 0.0
        assert result["SPY"]["market_value"] == 0.0

    def test_valid_position_returned(self, tmp_path):
        """Valid position with non-zero qty is returned correctly."""
        import sqlite3
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = str(tmp_path / "valid.db")
        conn = sqlite3.connect(sync.db_path)
        conn.execute("CREATE TABLE positions (symbol TEXT PRIMARY KEY, qty REAL, avg_price REAL, current_price REAL, market_value REAL, updated_at TEXT)")
        conn.execute("INSERT INTO positions VALUES ('SPY', 100, 500.0, 510.0, 51000.0, '2024-06-01T12:00:00')")
        conn.commit()
        conn.close()
        result = sync.get_local_positions()
        assert result["SPY"]["qty"] == 100.0
        assert result["SPY"]["avg_price"] == 500.0
        assert result["SPY"]["current_price"] == 510.0
        assert result["SPY"]["market_value"] == 51000.0
        assert result["SPY"]["updated_at"] == "2024-06-01T12:00:00"


class TestGetBrokerPositions:
    """get_broker_positions error and edge cases."""

    def _make_sync(self):
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = ":memory:"
        sync.data_dir = "/tmp"
        sync.sync_log_path = "/tmp/test_sync.jsonl"
        return sync

    def test_not_ready_returns_empty(self):
        """When client not ready, returns empty dict."""
        sync = self._make_sync()
        sync.client = MagicMock()
        sync.client.is_ready.return_value = False
        result = sync.get_broker_positions()
        assert result == {}

    def test_exception_returns_empty(self):
        """Exception from get_positions returns empty dict."""
        sync = self._make_sync()
        sync.client = MagicMock()
        sync.client.is_ready.return_value = True
        sync.client.get_positions.side_effect = ConnectionError("Connection failed")
        result = sync.get_broker_positions()
        assert result == {}

    def test_empty_list_returns_empty_dict(self):
        """Empty list from broker returns empty dict."""
        sync = self._make_sync()
        sync.client = MagicMock()
        sync.client.is_ready.return_value = True
        sync.client.get_positions.return_value = []
        result = sync.get_broker_positions()
        assert result == {}


class TestCalculateDriftMoreEdgeCases:
    """Additional calculate_drift edge cases beyond existing coverage."""

    def _make_sync(self):
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = ":memory:"
        sync.data_dir = "/tmp"
        sync.sync_log_path = "/tmp/test_sync.jsonl"
        return sync

    def test_value_threshold_just_above(self):
        """Value_delta just above $1 should trigger drift."""
        sync = self._make_sync()
        local = {"SPY": {"qty": 100, "market_value": 50000}}
        broker = {"SPY": MagicMock(qty=100, market_value=50001.01)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 1
        assert drift[0].value_delta == pytest.approx(1.01, abs=0.01)

    def test_qty_threshold_just_above(self):
        """qty_delta just above 0.001 should trigger drift."""
        sync = self._make_sync()
        local = {"SPY": {"qty": 100, "market_value": 50000}}
        broker = {"SPY": MagicMock(qty=100.002, market_value=50000)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 1
        assert drift[0].qty_delta == pytest.approx(0.002, abs=0.0001)

    def test_missing_market_value_in_local(self):
        """Local position missing market_value field should use 0."""
        sync = self._make_sync()
        local = {"SPY": {"qty": 100}}  # No market_value key
        broker = {"SPY": MagicMock(qty=100, market_value=50000)}
        drift = sync.calculate_drift(local, broker)
        # local value defaults to 0, broker_value=50000, value_delta=50000 > 1.0
        assert len(drift) == 1
        assert drift[0].local_value == 0.0
        assert drift[0].drift_pct == 1.0

    def test_large_position_values(self):
        """Very large position values should not overflow."""
        sync = self._make_sync()
        local = {"SPY": {"qty": 1e6, "market_value": 5e8}}
        broker = {"SPY": MagicMock(qty=1.001e6, market_value=5.005e8)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 1
        assert drift[0].qty_delta == 1000.0
        assert drift[0].value_delta == 500000.0

    def test_both_sides_zero_value(self):
        """Both local and broker have zero value should produce 0% drift."""
        sync = self._make_sync()
        local = {"SPY": {"qty": 0, "market_value": 0}}
        broker = {"SPY": MagicMock(qty=0, market_value=0)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 0


class TestSyncLogging:
    """sync() log file creation and dry-run isolation."""

    def _make_sync(self):
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = ":memory:"
        sync.data_dir = "/tmp"
        sync.sync_log_path = "/tmp/test_sync.jsonl"
        sync.client = MagicMock()
        return sync

    def test_sync_writes_log_when_not_dry_run(self, tmp_path):
        """Non-dry-run sync should write to sync log."""
        sync = self._make_sync()
        sync.client.is_ready.return_value = True
        sync.client.get_account.return_value = {"equity": "100000", "cash": "5000", "buying_power": "200000"}
        sync.client.paper = True
        sync.client.get_positions.return_value = []
        sync.sync_log_path = str(tmp_path / "sync.jsonl")
        sync.data_dir = str(tmp_path)
        sync.get_local_positions = lambda: {}
        sync.get_broker_positions = lambda: {}
        result = sync.sync(dry_run=False)
        assert result["status"] == "success"
        log_path = tmp_path / "sync.jsonl"
        assert log_path.exists()
        contents = log_path.read_text().strip()
        assert len(contents) > 0
        log_entry = json.loads(contents)
        assert log_entry["status"] == "success"

    def test_sync_with_missing_account_fields(self):
        """Account response missing fields should be handled."""
        sync = self._make_sync()
        sync.client.is_ready.return_value = True
        sync.client.paper = True
        sync.client.get_account.return_value = {}  # missing equity/cash/buying_power
        sync.client.get_positions.return_value = []
        sync.get_local_positions = lambda: {}
        sync.get_broker_positions = lambda: {}
        result = sync.sync(dry_run=True)
        assert result["status"] == "success"
        # Missing account fields should be None
        assert result["account"]["equity"] is None

    def test_sync_with_no_drift_max_symbol(self):
        """When drift count is 0, max_drift_symbol should be None."""
        sync = self._make_sync()
        sync.client.is_ready.return_value = True
        sync.client.paper = True
        sync.client.get_account.return_value = {"equity": "100000", "cash": "5000", "buying_power": "200000"}
        sync.client.get_positions.return_value = []
        sync.get_local_positions = lambda: {}
        sync.get_broker_positions = lambda: {}
        result = sync.sync(dry_run=True)
        assert result["drift"]["max_drift_symbol"] is None
        assert result["drift"]["max_drift_pct"] == 0.0

    def test_sync_creates_data_dir(self, tmp_path):
        """sync() should create data_dir if it doesn't exist."""
        sync = self._make_sync()
        nested = tmp_path / "a" / "b" / "c"
        sync.client.is_ready.return_value = True
        sync.client.paper = True
        sync.client.get_account.return_value = {"equity": "0", "cash": "0", "buying_power": "0"}
        sync.client.get_positions.return_value = []
        sync.sync_log_path = str(nested / "sync.jsonl")
        sync.data_dir = str(nested)
        sync.get_local_positions = lambda: {}
        sync.get_broker_positions = lambda: {}
        result = sync.sync(dry_run=False)
        assert result["status"] == "success"
        assert nested.exists()


class TestReconcileExtended:
    """More reconcile_to_broker edge cases."""

    def test_reconcile_empty_broker_positions(self, tmp_path):
        """Empty broker positions should not remove any local (no-op)."""
        import sqlite3
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = str(tmp_path / "empty_broker.db")
        sync.client = MagicMock()
        sync.client.is_ready.return_value = True
        sync.get_broker_positions = MagicMock(return_value={})
        conn = sqlite3.connect(sync.db_path)
        conn.execute("CREATE TABLE positions (symbol TEXT PRIMARY KEY, qty REAL, avg_price REAL, current_price REAL, market_value REAL, updated_at TEXT)")
        conn.execute("INSERT INTO positions VALUES ('SPY', 100, 500, 510, 51000, '2024-01-01')")
        conn.commit()
        conn.close()
        result = sync.reconcile_to_broker()
        assert result["status"] == "success"
        # Position should be removed since broker has no positions
        assert result["positions_removed"] == 1

    def test_reconcile_db_error(self, tmp_path):
        """Error during DB operations should return error status."""
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = str(tmp_path / "nonexistent_sub" / "nope.db")
        sync.client = MagicMock()
        sync.client.is_ready.return_value = True
        pos = MagicMock()
        pos.qty = 100
        pos.avg_entry_price = 500.0
        pos.current_price = 510.0
        pos.market_value = 51000.0
        sync.get_broker_positions = MagicMock(return_value={"SPY": pos})
        result = sync.reconcile_to_broker()
        assert result["status"] == "error"


class TestCLIExtended:
    """Extended CLI command tests."""

    def test_sync_dry_run_flag(self, capsys):
        """--dry-run flag should be passed to sync()."""
        from src.broker.position_sync import main
        with patch('sys.argv', ['position_sync.py', 'sync', '--dry-run']):
            with patch('src.broker.position_sync.PositionSync') as MockSync:
                mock = MagicMock()
                mock.sync.return_value = {"status": "success", "drift": {"count": 0}}
                MockSync.return_value = mock
                main()
                mock.sync.assert_called_once_with(dry_run=True)
        captured = capsys.readouterr()
        err = captured.err.strip()
        data = json.loads(err[err.index("{"):])
        assert data["status"] == "success"

    def test_reconcile_command(self, capsys):
        """Reconcile command calls reconcile_to_broker()."""
        from src.broker.position_sync import main
        with patch('sys.argv', ['position_sync.py', 'reconcile']):
            with patch('src.broker.position_sync.PositionSync') as MockSync:
                mock = MagicMock()
                mock.reconcile_to_broker.return_value = {
                    "status": "success", "timestamp": "2024-01-01",
                    "positions_updated": 2, "positions_removed": 0,
                }
                MockSync.return_value = mock
                main()
        captured = capsys.readouterr()
        err = captured.err.strip()
        data = json.loads(err[err.index("{"):])
        assert data["status"] == "success"
        assert data["positions_updated"] == 2

    def test_drift_with_error_response(self, capsys):
        """Drift command when sync returns error should display message."""
        from src.broker.position_sync import main
        with patch('sys.argv', ['position_sync.py', 'drift']):
            with patch('src.broker.position_sync.PositionSync') as MockSync:
                mock = MagicMock()
                mock.sync.return_value = {"status": "error", "message": "API failure"}
                MockSync.return_value = mock
                main()
        captured = capsys.readouterr()
        assert "Error: API failure" in captured.err

    def test_sync_command_no_args(self, capsys):
        """sync command without args calls sync() with defaults."""
        from src.broker.position_sync import main
        with patch('sys.argv', ['position_sync.py', 'sync']):
            with patch('src.broker.position_sync.PositionSync') as MockSync:
                mock = MagicMock()
                mock.sync.return_value = {"status": "success"}
                MockSync.return_value = mock
                main()
                mock.sync.assert_called_once_with(dry_run=False)


class TestModuleExports:
    """__all__ exports."""

    def test_all_exports(self):
        from src.broker.position_sync import __all__
        assert "PositionDrift" in __all__
        assert "PositionSync" in __all__
        assert len(__all__) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
