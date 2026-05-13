#!/usr/bin/env python3
"""
Tests for Alpaca Broker Client — data classes, order/position construction,
client status checks, price fetching, and paper trading manager.
"""
import sys
import os
import json
import sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.broker.alpaca import (
    OrderSide, OrderType, OrderRequest, Order, Position,
    AlpacaClient, PaperTradingManager, check_alpaca_status,
    ALPACA_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_db(db_path):
    """Create prices table in SQLite."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            date TEXT, symbol TEXT, close REAL, volume INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?)",
        ('2026-01-10', 'SPY', 585.0, 1000000),
    )
    conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?)",
        ('2026-01-10', 'GLD', 200.0, 500000),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_order_side_values(self):
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_type_values(self):
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"


# ---------------------------------------------------------------------------
# OrderRequest tests
# ---------------------------------------------------------------------------

class TestOrderRequest:
    def test_creation(self):
        req = OrderRequest(symbol='SPY', qty=10.0, side=OrderSide.BUY)
        assert req.symbol == 'SPY'
        assert req.qty == 10.0
        assert req.side == OrderSide.BUY

    def test_defaults(self):
        req = OrderRequest(symbol='SPY', qty=1.0, side=OrderSide.BUY)
        assert req.order_type == OrderType.MARKET
        assert req.limit_price is None
        assert req.time_in_force == "day"

    def test_to_dict(self):
        req = OrderRequest(symbol='SPY', qty=5.0, side=OrderSide.SELL)
        d = req.to_dict()
        assert d['symbol'] == 'SPY'
        assert d['qty'] == 5.0
        assert d['side'] == 'sell'
        assert d['type'] == 'market'

    def test_to_dict_limit(self):
        req = OrderRequest(
            symbol='GLD', qty=2.0, side=OrderSide.BUY,
            order_type=OrderType.LIMIT, limit_price=195.0,
        )
        d = req.to_dict()
        assert d['type'] == 'limit'
        assert d['limit_price'] == 195.0


# ---------------------------------------------------------------------------
# Order tests
# ---------------------------------------------------------------------------

class TestOrder:
    def test_creation(self):
        order = Order(
            id='abc123', symbol='SPY', qty=10.0, filled_qty=10.0,
            side='buy', type='market', status='filled',
            created_at='2026-01-10T10:00:00',
        )
        assert order.id == 'abc123'
        assert order.status == 'filled'

    def test_defaults(self):
        order = Order(
            id='x', symbol='SPY', qty=1.0, filled_qty=0.0,
            side='buy', type='market', status='pending',
            created_at='2026-01-10',
        )
        assert order.filled_at is None
        assert order.filled_avg_price is None

    def test_from_alpaca(self):
        mock_order = MagicMock()
        mock_order.id = 'test-id'
        mock_order.symbol = 'SPY'
        mock_order.qty = 10.0
        mock_order.filled_qty = 10.0
        mock_order.side.value = 'buy'
        mock_order.type.value = 'market'
        mock_order.status.value = 'filled'
        mock_order.created_at = datetime(2026, 1, 10, 10, 0, 0)
        mock_order.filled_at = datetime(2026, 1, 10, 10, 0, 1)
        mock_order.filled_avg_price = 585.0

        order = Order.from_alpaca(mock_order)
        assert order.id == 'test-id'
        assert order.symbol == 'SPY'
        assert order.side == 'buy'
        assert order.filled_avg_price == 585.0

    def test_from_alpaca_none_filled_at(self):
        mock_order = MagicMock()
        mock_order.id = 'x'
        mock_order.symbol = 'SPY'
        mock_order.qty = 1.0
        mock_order.filled_qty = 0.0
        mock_order.side.value = 'buy'
        mock_order.type.value = 'market'
        mock_order.status.value = 'pending'
        mock_order.created_at = datetime(2026, 1, 10)
        mock_order.filled_at = None
        mock_order.filled_avg_price = None

        order = Order.from_alpaca(mock_order)
        assert order.filled_at is None
        assert order.filled_avg_price is None


# ---------------------------------------------------------------------------
# Position tests
# ---------------------------------------------------------------------------

class TestPosition:
    def test_creation(self):
        pos = Position(
            symbol='SPY', qty=10.0, avg_entry_price=500.0,
            current_price=585.0, market_value=5850.0,
            unrealized_pl=850.0, unrealized_plpc=0.17,
        )
        assert pos.symbol == 'SPY'
        assert pos.unrealized_plpc == 0.17

    def test_from_alpaca(self):
        mock_pos = MagicMock()
        mock_pos.symbol = 'GLD'
        mock_pos.qty = 5.0
        mock_pos.avg_entry_price = 190.0
        mock_pos.current_price = 200.0
        mock_pos.market_value = 1000.0
        mock_pos.unrealized_pl = 50.0
        mock_pos.unrealized_plpc = 0.05

        pos = Position.from_alpaca(mock_pos)
        assert pos.symbol == 'GLD'
        assert pos.qty == 5.0
        assert pos.unrealized_pl == 50.0


# ---------------------------------------------------------------------------
# AlpacaClient tests
# ---------------------------------------------------------------------------

class TestAlpacaClient:
    def test_init_default(self):
        client = AlpacaClient()
        assert client.paper is True

    def test_init_live(self):
        client = AlpacaClient(paper=False)
        assert client.paper is False

    def test_is_configured_no_env(self):
        client = AlpacaClient()
        with patch.dict(os.environ, {}, clear=True):
            client.api_key = None
            client.api_secret = None
            assert client.is_configured() is False

    def test_is_configured_with_env(self):
        client = AlpacaClient()
        client.api_key = 'test-key'
        client.api_secret = 'test-secret'
        assert client.is_configured() is True

    def test_is_available(self):
        client = AlpacaClient()
        # Just checks ALPACA_AVAILABLE constant
        assert client.is_available() == ALPACA_AVAILABLE

    def test_is_ready_no_sdk(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'
        with patch('src.broker.alpaca.ALPACA_AVAILABLE', False):
            assert client.is_ready() is False

    def test_is_ready_no_creds(self):
        client = AlpacaClient()
        client.api_key = None
        client.api_secret = None
        assert client.is_ready() is False

    def test_fetch_price_with_db(self, tmp_path):
        db_path = tmp_path / "market.db"
        _init_db(db_path)
        client = AlpacaClient()
        price = client._fetch_price('SPY', str(db_path))
        assert price == 585.0

    def test_fetch_price_missing_symbol(self, tmp_path):
        db_path = tmp_path / "market.db"
        _init_db(db_path)
        client = AlpacaClient()
        price = client._fetch_price('AAPL', str(db_path))
        assert price == 0.0

    def test_fetch_price_no_db(self, tmp_path):
        client = AlpacaClient()
        price = client._fetch_price('SPY', str(tmp_path / "nope.db"))
        assert price == 0.0


# ---------------------------------------------------------------------------
# PaperTradingManager tests
# ---------------------------------------------------------------------------

class TestPaperTradingManager:
    def test_init(self, tmp_path):
        manager = PaperTradingManager(data_dir=str(tmp_path))
        assert manager.client.paper is True

    def test_is_ready_no_sdk(self, tmp_path):
        manager = PaperTradingManager(data_dir=str(tmp_path))
        with patch('src.broker.alpaca.ALPACA_AVAILABLE', False):
            assert manager.is_ready() is False

    def test_is_ready_no_creds(self, tmp_path):
        manager = PaperTradingManager(data_dir=str(tmp_path))
        manager.client.api_key = None
        manager.client.api_secret = None
        assert manager.is_ready() is False

    def test_sync_positions_not_configured(self, tmp_path):
        manager = PaperTradingManager(data_dir=str(tmp_path))
        manager.client.api_key = None
        manager.client.api_secret = None
        result = manager.sync_positions()
        assert result['status'] == 'not_configured'

    def test_execute_rebalance_not_configured(self, tmp_path):
        manager = PaperTradingManager(data_dir=str(tmp_path))
        manager.client.api_key = None
        manager.client.api_secret = None
        result = manager.execute_rebalance({'SPY': 0.5})
        assert result['status'] == 'not_configured'

    def test_sync_positions_with_mock(self, tmp_path):
        manager = PaperTradingManager(data_dir=str(tmp_path))

        mock_account = {
            'equity': 100000.0, 'cash': 50000.0, 'status': 'ACTIVE',
        }
        mock_positions = [
            Position('SPY', 10, 500.0, 585.0, 5850.0, 850.0, 0.17),
        ]

        with patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value=mock_account), \
             patch.object(manager.client, 'get_positions', return_value=mock_positions):
            result = manager.sync_positions()

        assert result['position_count'] == 1
        assert result['paper'] is True

    def test_execute_rebalance_dry_run(self, tmp_path):
        manager = PaperTradingManager(data_dir=str(tmp_path))

        mock_account = {'equity': 100000.0}
        mock_positions = [
            Position('SPY', 10, 500.0, 585.0, 5850.0, 850.0, 0.17),
            Position('GLD', 5, 190.0, 200.0, 1000.0, 50.0, 0.05),
        ]

        with patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value=mock_account), \
             patch.object(manager.client, 'get_positions', return_value=mock_positions), \
             patch.object(manager.client, '_fetch_price', return_value=585.0):
            result = manager.execute_rebalance(
                {'SPY': 0.6, 'GLD': 0.4}, total_value=100000, dry_run=True
            )

        assert result['dry_run'] is True
        assert result['order_count'] > 0

    def test_execute_rebalance_skips_small_delta(self, tmp_path):
        manager = PaperTradingManager(data_dir=str(tmp_path))

        mock_account = {'equity': 100000.0}
        # Current SPY value ≈ target value → small delta
        mock_positions = [
            Position('SPY', 10, 500.0, 585.0, 5850.0, 850.0, 0.17),
        ]

        with patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value=mock_account), \
             patch.object(manager.client, 'get_positions', return_value=mock_positions):
            # Target 5.85% = $5850, current = $5850 → delta < $10
            result = manager.execute_rebalance(
                {'SPY': 0.0585}, total_value=100000, dry_run=True
            )

        # Should skip because delta < $10
        assert result['order_count'] == 0


# ---------------------------------------------------------------------------
# check_alpaca_status tests
# ---------------------------------------------------------------------------

class TestCheckAlpacaStatus:
    def test_returns_dict(self):
        with patch.dict(os.environ, {}, clear=True):
            status = check_alpaca_status()
        assert 'sdk_available' in status
        assert 'configured' in status
        assert 'paper' in status

    def test_not_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            status = check_alpaca_status()
        assert status['configured'] is False
        assert status['connected'] is False

    def test_sdk_not_available(self):
        with patch('src.broker.alpaca.ALPACA_AVAILABLE', False), \
             patch.dict(os.environ, {}, clear=True):
            status = check_alpaca_status()
        assert status['sdk_available'] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
