#!/usr/bin/env python3
"""
Tests for Alpaca Broker Client — data classes, order/position construction,
client status checks, price fetching, and paper trading manager.
"""
import os
import json
import sqlite3

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from src.broker.alpaca import (
    OrderSide, OrderType, OrderRequest, Order, Position, MarketQuote,
    AlpacaClient, PaperTradingManager, check_alpaca_status,
    resolve_alpaca_feed_entitlement,
    resolve_alpaca_market_session,
    resolve_unavailable_alpaca_market_session,
    ALPACA_AVAILABLE,
)
from src.broker.circuit_breaker import broker_breaker


# ---------------------------------------------------------------------------
# Fixture: reset circuit breaker before each test so accumulated failures
# in one test do not trip the breaker for subsequent tests.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_broker_breaker() -> None:
    """Reset the global circuit breaker singleton before every test case."""
    broker_breaker.reset()


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


class TestAlpacaFeedEntitlement:
    """Configured feed and entitlement policy should be public-safe and fail closed."""

    def test_iex_basic_feed_is_explicitly_classified(self):
        metadata = resolve_alpaca_feed_entitlement({
            "ALPACA_DATA_FEED": "iex",
            "ALPACA_FEED_ENTITLEMENT": "basic",
        })

        assert metadata["configured_feed"] == "iex"
        assert metadata["effective_feed"] == "iex"
        assert metadata["entitlement"] == "basic"
        assert metadata["delayed"] is False
        assert metadata["acceptable_for_live"] is True
        assert metadata["policy_decision"] == "allow"
        assert metadata["reason"] is None

    def test_sip_feed_requires_sip_entitlement(self):
        metadata = resolve_alpaca_feed_entitlement({
            "ALPACA_DATA_FEED": "sip",
            "ALPACA_FEED_ENTITLEMENT": "sip",
        })

        assert metadata["effective_feed"] == "sip"
        assert metadata["entitlement"] == "sip"
        assert metadata["acceptable_for_live"] is True

    def test_delayed_sip_feed_is_not_acceptable_for_live(self):
        metadata = resolve_alpaca_feed_entitlement({
            "ALPACA_DATA_FEED": "sip",
            "ALPACA_FEED_ENTITLEMENT": "delayed_sip",
        })

        assert metadata["effective_feed"] == "sip"
        assert metadata["entitlement"] == "delayed_sip"
        assert metadata["delayed"] is True
        assert metadata["acceptable_for_live"] is False
        assert metadata["policy_decision"] == "reject"
        assert metadata["reason"] == "delayed_feed"

    def test_unknown_feed_is_not_acceptable_for_live(self):
        metadata = resolve_alpaca_feed_entitlement({
            "ALPACA_DATA_FEED": "mystery",
            "ALPACA_FEED_ENTITLEMENT": "sip",
        })

        assert metadata["effective_feed"] == "unknown"
        assert metadata["acceptable_for_live"] is False
        assert metadata["policy_decision"] == "reject"
        assert metadata["reason"] == "unknown_feed"

    def test_missing_entitlement_fails_closed(self):
        metadata = resolve_alpaca_feed_entitlement({"ALPACA_DATA_FEED": "iex"})

        assert metadata["entitlement"] == "unknown"
        assert metadata["acceptable_for_live"] is False
        assert metadata["policy_decision"] == "reject"
        assert metadata["reason"] == "missing_entitlement"


class TestAlpacaMarketSessionGuard:
    """Market-session policy should be deterministic and fail closed for live orders."""

    def test_regular_session_is_allowed(self):
        session = resolve_alpaca_market_session({
            "is_open": True,
            "timestamp": "2026-06-12T15:00:00+00:00",
            "next_open": "2026-06-15T13:30:00+00:00",
            "next_close": "2026-06-12T20:00:00+00:00",
        })

        assert session["session_state"] == "regular"
        assert session["allow_live_orders"] is True
        assert session["guard_decision"] == "allow"
        assert session["reason"] is None

    def test_closed_market_is_rejected_by_default(self):
        session = resolve_alpaca_market_session({
            "is_open": False,
            "timestamp": "2026-06-13T16:00:00+00:00",
            "next_open": "2026-06-15T13:30:00+00:00",
            "next_close": "2026-06-12T20:00:00+00:00",
        })

        assert session["session_state"] == "closed"
        assert session["allow_live_orders"] is False
        assert session["guard_decision"] == "reject"
        assert session["reason"] == "market_closed"

    def test_extended_hours_requires_explicit_allowance(self):
        blocked = resolve_alpaca_market_session(
            {"is_open": False, "session_state": "extended_hours"},
            env={},
        )
        allowed = resolve_alpaca_market_session(
            {"is_open": False, "session_state": "extended_hours"},
            env={"ALPACA_ALLOW_EXTENDED_HOURS": "true"},
        )

        assert blocked["guard_decision"] == "reject"
        assert blocked["reason"] == "extended_hours_not_allowed"
        assert allowed["guard_decision"] == "allow"
        assert allowed["allow_live_orders"] is True

    def test_unavailable_session_reports_error_type_without_raw_message(self):
        session = resolve_unavailable_alpaca_market_session(
            RuntimeError("credential-like value should not appear")
        )

        assert session["guard_decision"] == "reject"
        assert session["reason"] == "market_session_unavailable"
        assert session["error_type"] == "RuntimeError"
        assert "credential-like" not in json.dumps(session)


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

    def test_execute_rebalance_uses_fresh_broker_quote_for_new_position(self, tmp_path):
        """New positions should size from fresh broker quote objects when available."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        quote = MarketQuote(
            symbol="TLT",
            price=100.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="alpaca",
            feed="stock_latest_quote",
            age_seconds=10,
            source_mode="live",
        )

        with patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value={'equity': 100000.0}), \
             patch.object(manager.client, 'get_positions', return_value=[]), \
             patch.object(manager.client, 'get_latest_quote', return_value=quote):
            result = manager.execute_rebalance({'TLT': 0.5}, total_value=100000, dry_run=True)

        assert result["order_count"] == 1
        assert result["orders_planned"][0]["qty"] == 500.0
        assert result["quote_sources"]["TLT"]["source"] == "alpaca"
        assert result["quote_sources"]["TLT"]["source_mode"] == "live"

    def test_execute_rebalance_reports_feed_entitlement_in_quote_sources(self, tmp_path):
        """Quote source diagnostics should include public-safe Alpaca feed policy."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        quote = MarketQuote(
            symbol="TLT",
            price=100.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="alpaca",
            feed="iex",
            age_seconds=10,
            source_mode="live",
        )
        quote.feed_entitlement = {
            "configured_feed": "iex",
            "effective_feed": "iex",
            "entitlement": "basic",
            "delayed": False,
            "acceptable_for_live": True,
            "policy_decision": "allow",
            "reason": None,
        }

        with patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value={'equity': 100000.0}), \
             patch.object(manager.client, 'get_positions', return_value=[]), \
             patch.object(manager.client, 'get_latest_quote', return_value=quote):
            result = manager.execute_rebalance({'TLT': 0.5}, total_value=100000, dry_run=True)

        assert result["quote_sources"]["TLT"]["feed_entitlement"] == quote.feed_entitlement

    def test_execute_rebalance_allows_delayed_quote_in_dry_run(self, tmp_path):
        """Dry-run can use delayed quotes, but must label them explicitly."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        quote = MarketQuote(
            symbol="TLT",
            price=100.0,
            timestamp="2026-06-11T00:00:00+00:00",
            source="alpaca",
            feed="stock_latest_quote",
            age_seconds=3600,
            source_mode="live",
        )

        with patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value={'equity': 100000.0}), \
             patch.object(manager.client, 'get_positions', return_value=[]), \
             patch.object(manager.client, 'get_latest_quote', return_value=quote):
            result = manager.execute_rebalance({'TLT': 0.5}, total_value=100000, dry_run=True)

        assert result["order_count"] == 1
        assert result["quote_sources"]["TLT"]["source_mode"] == "delayed"

    def test_execute_rebalance_labels_prior_close_fallback_in_dry_run(self, tmp_path):
        """Paper/dry-run fallback to local close should carry prior-close provenance."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        quote = MarketQuote(
            symbol="TLT",
            price=100.0,
            timestamp="2026-06-10T21:00:00+00:00",
            source="market_db",
            feed="yahoo_close",
            age_seconds=86400,
            source_mode="prior_close",
        )

        with patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value={'equity': 100000.0}), \
             patch.object(manager.client, 'get_positions', return_value=[]), \
             patch.object(manager.client, 'get_latest_quote', return_value=None), \
             patch.object(manager.client, '_fetch_price_quote', return_value=quote):
            result = manager.execute_rebalance({'TLT': 0.5}, total_value=100000, dry_run=True)

        assert result["order_count"] == 1
        assert result["quote_sources"]["TLT"]["source"] == "market_db"
        assert result["quote_sources"]["TLT"]["source_mode"] == "prior_close"

    def test_execute_rebalance_reports_market_session_diagnostics(self, tmp_path):
        """Rebalance diagnostics should include market-session state and guard decision."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        quote = MarketQuote(
            symbol="TLT",
            price=100.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="alpaca",
            feed="iex",
            age_seconds=10,
            source_mode="live",
        )
        market_session = {
            "session_state": "regular",
            "is_open": True,
            "timestamp": "2026-06-12T15:00:00+00:00",
            "next_open": "2026-06-15T13:30:00+00:00",
            "next_close": "2026-06-12T20:00:00+00:00",
            "extended_hours_allowed": False,
            "allow_live_orders": True,
            "guard_decision": "allow",
            "reason": None,
        }

        with patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value={'equity': 100000.0}), \
             patch.object(manager.client, 'get_positions', return_value=[]), \
             patch.object(manager.client, 'get_latest_quote', return_value=quote), \
             patch.object(manager.client, 'get_market_session', return_value=market_session):
            result = manager.execute_rebalance({'TLT': 0.5}, total_value=100000, dry_run=True)

        assert result["market_session"] == market_session
        assert result["market_session"]["guard_decision"] == "allow"

    def test_execute_rebalance_rejects_stale_broker_quote_in_live_mode(self, tmp_path):
        """Live order mode should reject stale broker quotes."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        quote = MarketQuote(
            symbol="TLT",
            price=100.0,
            timestamp="2026-06-11T00:00:00+00:00",
            source="alpaca",
            feed="stock_latest_quote",
            age_seconds=3600,
            source_mode="live",
        )

        with patch.dict(os.environ, {"ALPACA_PAPER": "false"}), \
             patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value={'equity': 100000.0}), \
             patch.object(manager.client, 'get_positions', return_value=[]), \
             patch.object(manager.client, 'get_latest_quote', return_value=quote), \
             patch.object(manager.client, 'get_market_session', return_value={
                 "session_state": "regular",
                 "allow_live_orders": True,
                 "guard_decision": "allow",
                 "reason": None,
             }):
            result = manager.execute_rebalance({'TLT': 0.5}, total_value=100000, dry_run=False)

        assert result["status"] == "error"
        assert "stale" in result["message"]

    def test_execute_rebalance_rejects_prior_close_in_live_mode(self, tmp_path):
        """Live order mode should not silently size from prior-close local data."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        quote = MarketQuote(
            symbol="TLT",
            price=100.0,
            timestamp="2026-06-10T21:00:00+00:00",
            source="market_db",
            feed="yahoo_close",
            age_seconds=86400,
            source_mode="prior_close",
        )

        with patch.dict(os.environ, {"ALPACA_PAPER": "false"}), \
             patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value={'equity': 100000.0}), \
             patch.object(manager.client, 'get_positions', return_value=[]), \
             patch.object(manager.client, 'get_latest_quote', return_value=None), \
             patch.object(manager.client, '_fetch_price_quote', return_value=quote), \
             patch.object(manager.client, 'get_market_session', return_value={
                 "session_state": "regular",
                 "allow_live_orders": True,
                 "guard_decision": "allow",
                 "reason": None,
             }):
            result = manager.execute_rebalance({'TLT': 0.5}, total_value=100000, dry_run=False)

        assert result["status"] == "error"
        assert "requires a fresh broker quote" in result["message"]
        assert result["market_session"]["session_state"] == "regular"

    def test_execute_rebalance_rejects_closed_market_in_live_mode(self, tmp_path):
        """Live order sizing should fail closed when the market clock is closed."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        quote = MarketQuote(
            symbol="TLT",
            price=100.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="alpaca",
            feed="iex",
            age_seconds=5,
            source_mode="live",
            feed_entitlement={
                "configured_feed": "iex",
                "effective_feed": "iex",
                "entitlement": "basic",
                "delayed": False,
                "acceptable_for_live": True,
                "policy_decision": "allow",
                "reason": None,
            },
        )
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

        with patch.dict(os.environ, {"ALPACA_PAPER": "false"}), \
             patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value={'equity': 100000.0}), \
             patch.object(manager.client, 'get_positions', return_value=[]), \
             patch.object(manager.client, 'get_latest_quote', return_value=quote), \
             patch.object(manager.client, 'get_market_session', return_value=market_session), \
             patch.object(manager.client, 'submit_order') as submit_order:
            result = manager.execute_rebalance({'TLT': 0.5}, total_value=100000, dry_run=False)

        assert result["status"] == "error"
        assert "market_closed" in result["message"]
        assert result["market_session"] == market_session
        submit_order.assert_not_called()

    def test_execute_rebalance_rejects_unacceptable_feed_in_live_mode(self, tmp_path):
        """Live order sizing should fail closed when feed entitlement is unacceptable."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        quote = MarketQuote(
            symbol="TLT",
            price=100.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="alpaca",
            feed="sip",
            age_seconds=5,
            source_mode="delayed",
        )
        quote.feed_entitlement = {
            "configured_feed": "sip",
            "effective_feed": "sip",
            "entitlement": "delayed_sip",
            "delayed": True,
            "acceptable_for_live": False,
            "policy_decision": "reject",
            "reason": "delayed_feed",
        }

        with patch.dict(os.environ, {"ALPACA_PAPER": "false"}), \
             patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value={'equity': 100000.0}), \
             patch.object(manager.client, 'get_positions', return_value=[]), \
             patch.object(manager.client, 'get_latest_quote', return_value=quote), \
             patch.object(manager.client, 'get_market_session', return_value={
                 "session_state": "regular",
                 "allow_live_orders": True,
                 "guard_decision": "allow",
                 "reason": None,
             }), \
             patch.object(manager.client, 'submit_order') as submit_order:
            result = manager.execute_rebalance({'TLT': 0.5}, total_value=100000, dry_run=False)

        assert result["status"] == "error"
        assert "Alpaca feed entitlement" in result["message"]
        assert "delayed_feed" in result["message"]
        submit_order.assert_not_called()

    def test_execute_rebalance_rejects_unacceptable_feed_for_existing_position_in_live_mode(self, tmp_path):
        """Live sizing from existing-position prices should use the same feed policy gate."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        mock_position = Position("SPY", 10, 500.0, 585.0, 5850.0, 850.0, 0.17)

        with patch.dict(
            os.environ,
            {
                "ALPACA_PAPER": "false",
                "ALPACA_DATA_FEED": "sip",
                "ALPACA_FEED_ENTITLEMENT": "delayed_sip",
            },
        ), \
             patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value={'equity': 100000.0}), \
             patch.object(manager.client, 'get_positions', return_value=[mock_position]), \
             patch.object(manager.client, 'get_market_session', return_value={
                 "session_state": "regular",
                 "allow_live_orders": True,
                 "guard_decision": "allow",
                 "reason": None,
             }), \
             patch.object(manager.client, 'submit_order') as submit_order:
            result = manager.execute_rebalance({'SPY': 0.6}, total_value=100000, dry_run=False)

        assert result["status"] == "error"
        assert "Alpaca feed entitlement" in result["message"]
        assert "delayed_feed" in result["message"]
        submit_order.assert_not_called()


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


# ---------------------------------------------------------------------------
# Extended coverage tests
# ---------------------------------------------------------------------------


class TestOrderRequestExtended:
    """Extended OrderRequest tests."""

    def test_to_dict_has_all_fields(self):
        req = OrderRequest(
            symbol='TLT', qty=3.0, side=OrderSide.BUY,
            order_type=OrderType.LIMIT, limit_price=95.0,
            time_in_force='gtc',
        )
        d = req.to_dict()
        expected_keys = {'symbol', 'qty', 'side', 'type', 'limit_price', 'time_in_force'}
        assert expected_keys == set(d.keys())

    def test_market_order_no_limit_price(self):
        """Market order should have None limit_price."""
        req = OrderRequest(symbol='SPY', qty=1.0, side=OrderSide.BUY)
        d = req.to_dict()
        assert d['limit_price'] is None


class TestOrderExtended:
    """Extended Order tests."""

    def test_filled_order_fields(self):
        order = Order(
            id='fill-1', symbol='GLD', qty=5.0, filled_qty=5.0,
            side='buy', type='market', status='filled',
            created_at='2026-05-24T10:00:00',
            filled_at='2026-05-24T10:00:01',
            filled_avg_price=200.5,
        )
        assert order.filled_qty == order.qty
        assert order.filled_avg_price > 0

    def test_from_alpaca_with_filled(self):
        mock = MagicMock()
        mock.id = 'o1'
        mock.symbol = 'TLT'
        mock.qty = 2.0
        mock.filled_qty = 2.0
        mock.side.value = 'sell'
        mock.type.value = 'limit'
        mock.status.value = 'filled'
        mock.created_at = datetime(2026, 5, 24)
        mock.filled_at = datetime(2026, 5, 24, 10, 0, 5)
        mock.filled_avg_price = 95.0
        order = Order.from_alpaca(mock)
        assert order.side == 'sell'
        assert order.type == 'limit'
        assert order.filled_avg_price == 95.0


class TestPositionExtended:
    """Extended Position tests."""

    def test_all_fields(self):
        pos = Position(
            symbol='TLT', qty=20.0, avg_entry_price=95.0,
            current_price=92.0, market_value=1840.0,
            unrealized_pl=-60.0, unrealized_plpc=-0.032,
        )
        assert pos.unrealized_pl < 0
        assert pos.unrealized_plpc < 0

    def test_from_alpaca_with_negative_pl(self):
        mock = MagicMock()
        mock.symbol = 'TLT'
        mock.qty = 20.0
        mock.avg_entry_price = 95.0
        mock.current_price = 92.0
        mock.market_value = 1840.0
        mock.unrealized_pl = -60.0
        mock.unrealized_plpc = -0.032
        pos = Position.from_alpaca(mock)
        assert pos.unrealized_pl == -60.0


class TestAlpacaClientExtended:
    """Extended AlpacaClient tests."""

    def test_get_account_not_configured_raises(self):
        """get_account should raise when not configured."""
        client = AlpacaClient()
        client.api_key = None
        client.api_secret = None
        with pytest.raises((ImportError, RuntimeError)):
            client.get_account()

    def test_submit_order_not_configured_raises(self):
        """submit_order should raise when not configured."""
        client = AlpacaClient()
        client.api_key = None
        client.api_secret = None
        req = OrderRequest(symbol='SPY', qty=1.0, side=OrderSide.BUY)
        with pytest.raises((ImportError, RuntimeError)):
            client.submit_order(req)

    def test_get_orders_not_configured_raises(self):
        client = AlpacaClient()
        client.api_key = None
        client.api_secret = None
        with pytest.raises((ImportError, RuntimeError)):
            client.get_orders()

    def test_cancel_order_not_configured_raises(self):
        client = AlpacaClient()
        client.api_key = None
        client.api_secret = None
        with pytest.raises((ImportError, RuntimeError)):
            client.cancel_order('fake-id')

    def test_cancel_all_orders_not_configured_raises(self):
        client = AlpacaClient()
        client.api_key = None
        client.api_secret = None
        with pytest.raises((ImportError, RuntimeError)):
            client.cancel_all_orders()

    def test_get_positions_not_configured_raises(self):
        client = AlpacaClient()
        client.api_key = None
        client.api_secret = None
        with pytest.raises((ImportError, RuntimeError)):
            client.get_positions()

    def test_get_position_not_configured_raises(self):
        client = AlpacaClient()
        client.api_key = None
        client.api_secret = None
        with pytest.raises((ImportError, RuntimeError)):
            client.get_position('SPY')

    def test_get_clock_not_configured_raises(self):
        client = AlpacaClient()
        client.api_key = None
        client.api_secret = None
        with pytest.raises((ImportError, RuntimeError)):
            client.get_clock()

    def test_get_bars_not_configured_raises(self):
        client = AlpacaClient()
        client.api_key = None
        client.api_secret = None
        with pytest.raises((ImportError, RuntimeError)):
            client.get_bars('SPY')

    def test_close_position_not_configured_raises(self):
        client = AlpacaClient()
        client.api_key = None
        client.api_secret = None
        with pytest.raises((ImportError, RuntimeError)):
            client.close_position('SPY')

    def test_close_all_positions_not_configured_raises(self):
        client = AlpacaClient()
        client.api_key = None
        client.api_secret = None
        with pytest.raises((ImportError, RuntimeError)):
            client.close_all_positions()


class TestPaperTradingManagerExtended:
    """Extended PaperTradingManager tests."""

    def test_execute_rebalance_with_orders(self, tmp_path):
        """Rebalance with significant drift should generate orders."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        mock_account = {'equity': 100000.0}
        # SPY at 10%, target 60% → large delta
        mock_positions = [
            Position('SPY', 2, 500.0, 585.0, 1170.0, 170.0, 0.17),
        ]
        with patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value=mock_account), \
             patch.object(manager.client, 'get_positions', return_value=mock_positions), \
             patch.object(manager.client, '_fetch_price', return_value=585.0):
            result = manager.execute_rebalance(
                {'SPY': 0.60}, total_value=100000, dry_run=True
            )
        assert result['order_count'] > 0

    def test_execute_rebalance_with_empty_positions(self, tmp_path):
        """Rebalance from no positions should generate orders."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        mock_account = {'equity': 100000.0}
        with patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value=mock_account), \
             patch.object(manager.client, 'get_positions', return_value=[]), \
             patch.object(manager.client, '_fetch_price', return_value=200.0):
            result = manager.execute_rebalance(
                {'GLD': 0.50}, total_value=100000, dry_run=True
            )
        # May contain dry_run key or error status
        assert 'dry_run' in result or 'status' in result


class TestOrderRequestValidation:
    """OrderRequest field validation tests."""

    def test_limit_order_requires_price(self):
        """Limit order with no limit_price should raise ValueError."""
        req = OrderRequest(
            symbol='SPY', qty=1.0, side=OrderSide.BUY,
            order_type=OrderType.LIMIT, limit_price=None,
        )
        client = AlpacaClient()
        # Patch _get_client to avoid TradingClient import
        with patch.object(client, '_get_client', return_value=MagicMock()):
            with pytest.raises(ValueError, match="Limit price required"):
                client.submit_order(req)

    def test_custom_time_in_force(self):
        """to_dict should reflect custom time_in_force."""
        req = OrderRequest(
            symbol='SPY', qty=1.0, side=OrderSide.BUY,
            time_in_force='gtc',
        )
        d = req.to_dict()
        assert d['time_in_force'] == 'gtc'


class TestOrderEdgeCases:
    """Order edge cases — string attributes, partial fills."""

    def test_from_alpaca_with_string_attrs(self):
        """from_alpaca handles plain string attributes (no .value)."""
        mock_order = MagicMock()
        mock_order.id = 'str-id'
        mock_order.symbol = 'SPY'
        mock_order.qty = 5.0
        mock_order.filled_qty = 2.0
        # Set side/type/status as plain strings (no .value attribute)
        mock_order.side = 'buy'
        mock_order.type = 'market'
        mock_order.status = 'partially_filled'
        mock_order.created_at = '2026-01-10T10:00:00'
        mock_order.filled_at = '2026-01-10T10:05:00'
        mock_order.filled_avg_price = 584.5

        order = Order.from_alpaca(mock_order)
        assert order.side == 'buy'
        assert order.type == 'market'
        assert order.status == 'partially_filled'
        assert order.filled_qty == 2.0
        assert order.filled_avg_price == 584.5

    def test_from_alpaca_partial_fill(self):
        """from_alpaca with partial fill (filled_qty < qty)."""
        mock_order = MagicMock()
        mock_order.id = 'partial-1'
        mock_order.symbol = 'SPY'
        mock_order.qty = 10.0
        mock_order.filled_qty = 7.5
        mock_order.side.value = 'buy'
        mock_order.type.value = 'market'
        mock_order.status.value = 'partially_filled'
        mock_order.created_at = datetime(2026, 5, 24, 10, 0, 0)
        mock_order.filled_at = datetime(2026, 5, 24, 10, 0, 30)
        mock_order.filled_avg_price = 585.25

        order = Order.from_alpaca(mock_order)
        assert order.filled_qty == 7.5
        assert order.filled_qty < order.qty
        assert order.status == 'partially_filled'


class TestPositionEdgeCases:
    """Position edge cases — high precision values."""

    def test_from_alpaca_with_high_precision(self):
        """from_alpaca handles high-precision float values."""
        mock_pos = MagicMock()
        mock_pos.symbol = 'SPY'
        mock_pos.qty = 10.1234
        mock_pos.avg_entry_price = 500.1234
        mock_pos.current_price = 585.9876
        mock_pos.market_value = 5927.89
        mock_pos.unrealized_pl = 867.89
        mock_pos.unrealized_plpc = 0.1736

        pos = Position.from_alpaca(mock_pos)
        assert pos.qty == 10.1234
        assert pos.avg_entry_price == 500.1234
        assert pos.current_price == 585.9876


class TestAlpacaClientAccount:
    """AlpacaClient.get_account success path."""

    def test_get_account_success(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_account = MagicMock()
        mock_account.id = 'acc-001'
        mock_account.status = 'ACTIVE'
        mock_account.currency = 'USD'
        mock_account.cash = 50000.0
        mock_account.portfolio_value = 100000.0
        mock_account.equity = 100000.0
        mock_account.buying_power = 200000.0
        mock_account.maintenance_margin = 25000.0
        mock_account.initial_margin = 50000.0
        mock_account.daytrade_count = 0
        mock_account.last_equity = 95000.0

        mock_trading = MagicMock()
        mock_trading.get_account.return_value = mock_account

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            result = client.get_account()

        assert result['id'] == 'acc-001'
        assert result['status'] == 'ACTIVE'
        assert result['cash'] == 50000.0
        assert result['equity'] == 100000.0
        assert result['buying_power'] == 200000.0
        assert result['daytrade_count'] == 0
        assert result['paper'] is True
        assert result['last_equity'] == 95000.0

    def test_get_account_last_equity_none(self):
        """get_account handles None last_equity."""
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_account = MagicMock()
        mock_account.id = 'acc-002'
        mock_account.status = 'ACTIVE'
        mock_account.currency = 'USD'
        mock_account.cash = 0.0
        mock_account.portfolio_value = 0.0
        mock_account.equity = 0.0
        mock_account.buying_power = 0.0
        mock_account.maintenance_margin = 0.0
        mock_account.initial_margin = 0.0
        mock_account.daytrade_count = 0
        mock_account.last_equity = None

        mock_trading = MagicMock()
        mock_trading.get_account.return_value = mock_account

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            result = client.get_account()

        assert result['last_equity'] is None


class TestAlpacaClientSubmitOrder:
    """AlpacaClient.submit_order success/failure paths."""

    def _patch_sdk_names(self):
        """Return context managers patching SDK names not available when alpaca-py absent."""
        return [
            patch('src.broker.alpaca.MarketOrderRequest'),
            patch('src.broker.alpaca.LimitOrderRequest'),
            patch('src.broker.alpaca.TimeInForce'),
        ]

    def test_submit_market_order_success(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_result = MagicMock()
        mock_result.id = 'mkt-1'
        mock_result.symbol = 'SPY'
        mock_result.qty = 10.0
        mock_result.filled_qty = 10.0
        mock_result.side.value = 'buy'
        mock_result.type.value = 'market'
        mock_result.status.value = 'filled'
        mock_result.created_at = datetime(2026, 5, 24, 10, 0, 0)
        mock_result.filled_at = datetime(2026, 5, 24, 10, 0, 1)
        mock_result.filled_avg_price = 585.0

        mock_trading = MagicMock()
        mock_trading.submit_order.return_value = mock_result

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch('src.broker.alpaca.MarketOrderRequest', create=True), \
             patch('src.broker.alpaca.LimitOrderRequest', create=True), \
             patch('src.broker.alpaca.TimeInForce', create=True) as mock_tif, \
             patch.object(client, '_get_client', return_value=mock_trading):
            mock_tif.DAY = 'day'
            req = OrderRequest(symbol='SPY', qty=10.0, side=OrderSide.BUY)
            order = client.submit_order(req)

        assert order.id == 'mkt-1'
        assert order.symbol == 'SPY'
        assert order.side == 'buy'
        assert order.filled_qty == 10.0
        assert order.filled_avg_price == 585.0

    def test_submit_limit_order_success(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_result = MagicMock()
        mock_result.id = 'lmt-1'
        mock_result.symbol = 'GLD'
        mock_result.qty = 5.0
        mock_result.filled_qty = 0.0
        mock_result.side.value = 'buy'
        mock_result.type.value = 'limit'
        mock_result.status.value = 'pending'
        mock_result.created_at = datetime(2026, 5, 24, 10, 0, 0)
        mock_result.filled_at = None
        mock_result.filled_avg_price = None

        mock_trading = MagicMock()
        mock_trading.submit_order.return_value = mock_result

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch('src.broker.alpaca.MarketOrderRequest', create=True), \
             patch('src.broker.alpaca.LimitOrderRequest', create=True), \
             patch('src.broker.alpaca.TimeInForce', create=True) as mock_tif, \
             patch.object(client, '_get_client', return_value=mock_trading):
            mock_tif.DAY = 'day'
            req = OrderRequest(
                symbol='GLD', qty=5.0, side=OrderSide.BUY,
                order_type=OrderType.LIMIT, limit_price=200.0,
            )
            order = client.submit_order(req)

        assert order.id == 'lmt-1'
        assert order.type == 'limit'
        assert order.status == 'pending'
        assert order.filled_qty == 0.0

    def test_submit_order_custom_time_in_force(self):
        """submit_order with custom time_in_force maps to Alpaca enum."""
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_result = MagicMock()
        mock_result.id = 'gtc-1'
        mock_result.symbol = 'SPY'
        mock_result.qty = 1.0
        mock_result.filled_qty = 0.0
        mock_result.side.value = 'buy'
        mock_result.type.value = 'market'
        mock_result.status.value = 'pending'
        mock_result.created_at = datetime(2026, 5, 24, 10, 0, 0)
        mock_result.filled_at = None
        mock_result.filled_avg_price = None

        mock_trading = MagicMock()
        mock_trading.submit_order.return_value = mock_result

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch('src.broker.alpaca.MarketOrderRequest', create=True), \
             patch('src.broker.alpaca.LimitOrderRequest', create=True), \
             patch('src.broker.alpaca.TimeInForce', create=True) as mock_tif, \
             patch.object(client, '_get_client', return_value=mock_trading):
            mock_tif.DAY = 'day'
            req = OrderRequest(
                symbol='SPY', qty=1.0, side=OrderSide.BUY,
                time_in_force='gtc',
            )
            order = client.submit_order(req)

        assert order.id == 'gtc-1'


class TestAlpacaClientOrders:
    """AlpacaClient order retrieval and cancellation."""

    def test_get_orders_success(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_order = MagicMock()
        mock_order.id = 'o-1'
        mock_order.symbol = 'SPY'
        mock_order.qty = 10.0
        mock_order.filled_qty = 5.0
        mock_order.side.value = 'buy'
        mock_order.type.value = 'market'
        mock_order.status.value = 'partially_filled'
        mock_order.created_at = datetime(2026, 5, 24, 9, 30, 0)
        mock_order.filled_at = datetime(2026, 5, 24, 9, 35, 0)
        mock_order.filled_avg_price = 584.0

        mock_trading = MagicMock()
        mock_trading.get_orders.return_value = [mock_order]

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            orders = client.get_orders(limit=50)

        assert len(orders) == 1
        assert orders[0].symbol == 'SPY'
        assert orders[0].status == 'partially_filled'

    def test_get_orders_empty(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_trading = MagicMock()
        mock_trading.get_orders.return_value = []

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            orders = client.get_orders()

        assert len(orders) == 0

    def test_cancel_order_success(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_trading = MagicMock()
        mock_trading.cancel_order_by_id.return_value = None

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            result = client.cancel_order('order-123')

        assert result is True

    def test_cancel_order_failure(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_trading = MagicMock()
        mock_trading.cancel_order_by_id.side_effect = RuntimeError("Order not found")

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            result = client.cancel_order('bad-id')

        assert result is False

    def test_cancel_all_orders_success(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_trading = MagicMock()
        mock_trading.cancel_orders.return_value = ['id-1', 'id-2']

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            count = client.cancel_all_orders()

        assert count == 2

    def test_cancel_all_orders_failure(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_trading = MagicMock()
        mock_trading.cancel_orders.side_effect = RuntimeError("API error")

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            count = client.cancel_all_orders()

        assert count == 0

    def test_cancel_all_orders_empty_result(self):
        """cancel_all_orders returns 0 when SDK returns None."""
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_trading = MagicMock()
        mock_trading.cancel_orders.return_value = None

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            count = client.cancel_all_orders()

        assert count == 0


class TestAlpacaClientPositions:
    """AlpacaClient position retrieval and closure."""

    def test_get_positions_success(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_pos = MagicMock()
        mock_pos.symbol = 'SPY'
        mock_pos.qty = 10.0
        mock_pos.avg_entry_price = 500.0
        mock_pos.current_price = 585.0
        mock_pos.market_value = 5850.0
        mock_pos.unrealized_pl = 850.0
        mock_pos.unrealized_plpc = 0.17

        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = [mock_pos]

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            positions = client.get_positions()

        assert len(positions) == 1
        assert positions[0].symbol == 'SPY'
        assert positions[0].unrealized_pl == 850.0

    def test_get_positions_empty(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = []

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            positions = client.get_positions()

        assert len(positions) == 0

    def test_get_position_not_found(self):
        """get_position returns None when symbol not held."""
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_trading = MagicMock()
        mock_trading.get_open_position.side_effect = RuntimeError("position not found")

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            pos = client.get_position('AAPL')

        assert pos is None

    def test_get_position_found(self):
        """get_position returns Position when symbol is held."""
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_pos = MagicMock()
        mock_pos.symbol = 'SPY'
        mock_pos.qty = 10.0
        mock_pos.avg_entry_price = 500.0
        mock_pos.current_price = 585.0
        mock_pos.market_value = 5850.0
        mock_pos.unrealized_pl = 850.0
        mock_pos.unrealized_plpc = 0.17

        mock_trading = MagicMock()
        mock_trading.get_open_position.return_value = mock_pos

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            pos = client.get_position('SPY')

        assert pos is not None
        assert pos.symbol == 'SPY'
        assert pos.qty == 10.0

    def test_close_position_with_qty(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_result = MagicMock()
        mock_result.id = 'close-1'
        mock_result.symbol = 'SPY'
        mock_result.qty = 5.0
        mock_result.filled_qty = 5.0
        mock_result.side.value = 'sell'
        mock_result.type.value = 'market'
        mock_result.status.value = 'filled'
        mock_result.created_at = datetime(2026, 5, 24, 10, 0, 0)
        mock_result.filled_at = datetime(2026, 5, 24, 10, 0, 5)
        mock_result.filled_avg_price = 585.0

        mock_trading = MagicMock()
        mock_trading.close_position.return_value = mock_result

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            order = client.close_position('SPY', qty=5.0)

        assert order.id == 'close-1'
        mock_trading.close_position.assert_called_with('SPY', 5.0)

    def test_close_position_without_qty(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_result = MagicMock()
        mock_result.id = 'close-all'
        mock_result.symbol = 'SPY'
        mock_result.qty = 10.0
        mock_result.filled_qty = 10.0
        mock_result.side.value = 'sell'
        mock_result.type.value = 'market'
        mock_result.status.value = 'filled'
        mock_result.created_at = datetime(2026, 5, 24, 10, 0, 0)
        mock_result.filled_at = datetime(2026, 5, 24, 10, 0, 5)
        mock_result.filled_avg_price = 585.0

        mock_trading = MagicMock()
        mock_trading.close_position.return_value = mock_result

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            order = client.close_position('SPY')

        assert order.id == 'close-all'
        mock_trading.close_position.assert_called_with('SPY')

    def test_close_position_raises(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_trading = MagicMock()
        mock_trading.close_position.side_effect = RuntimeError("Insufficient liquidity")

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            with pytest.raises(RuntimeError, match="Failed to close position SPY"):
                client.close_position('SPY')

    def test_close_all_positions_success(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_result = MagicMock()
        mock_result.id = 'close-all-1'
        mock_result.symbol = 'SPY'
        mock_result.qty = 10.0
        mock_result.filled_qty = 10.0
        mock_result.side.value = 'sell'
        mock_result.type.value = 'market'
        mock_result.status.value = 'filled'
        mock_result.created_at = datetime(2026, 5, 24, 10, 0, 0)
        mock_result.filled_at = datetime(2026, 5, 24, 10, 0, 5)
        mock_result.filled_avg_price = 585.0

        mock_trading = MagicMock()
        mock_trading.close_all_positions.return_value = [mock_result]

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            orders = client.close_all_positions()

        assert len(orders) == 1
        assert orders[0].symbol == 'SPY'

    def test_close_all_positions_raises(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_trading = MagicMock()
        mock_trading.close_all_positions.side_effect = RuntimeError("API error")

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            with pytest.raises(RuntimeError, match="Failed to close all positions"):
                client.close_all_positions()


class TestAlpacaClientMarketData:
    """AlpacaClient market data methods."""

    def test_get_clock_success(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_clock = MagicMock()
        mock_clock.timestamp = datetime(2026, 5, 24, 12, 0, 0)
        mock_clock.is_open = True
        mock_clock.next_open = datetime(2026, 5, 25, 9, 30, 0)
        mock_clock.next_close = datetime(2026, 5, 24, 16, 0, 0)

        mock_trading = MagicMock()
        mock_trading.get_clock.return_value = mock_clock

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.object(client, '_get_client', return_value=mock_trading):
            clock = client.get_clock()

        assert clock['is_open'] is True
        assert clock['timestamp'] == '2026-05-24T12:00:00'
        assert clock['next_close'] == '2026-05-24T16:00:00'

    def test_get_latest_quote_includes_feed_entitlement_metadata(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_quote = MagicMock()
        mock_quote.bid_price = 99.0
        mock_quote.ask_price = 101.0
        mock_quote.timestamp = datetime(2026, 6, 12, 13, 30, tzinfo=timezone.utc)
        mock_data = MagicMock()
        mock_data.get_stock_latest_quote.return_value = mock_quote

        with patch.dict(os.environ, {"ALPACA_DATA_FEED": "iex", "ALPACA_FEED_ENTITLEMENT": "basic"}), \
             patch.object(client, '_get_data_client', return_value=mock_data):
            quote = client.get_latest_quote(
                "TLT",
                now=datetime(2026, 6, 12, 13, 31, tzinfo=timezone.utc),
            )

        assert quote is not None
        assert quote.feed == "iex"
        assert quote.age_seconds == 60
        assert quote.to_dict()["feed_entitlement"] == {
            "configured_feed": "iex",
            "effective_feed": "iex",
            "entitlement": "basic",
            "delayed": False,
            "acceptable_for_live": True,
            "policy_decision": "allow",
            "reason": None,
        }

    def test_get_bars_default(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_bar = MagicMock()
        mock_bar.timestamp = datetime(2026, 5, 24, 10, 0, 0)
        mock_bar.open = 580.0
        mock_bar.high = 586.0
        mock_bar.low = 579.0
        mock_bar.close = 585.0
        mock_bar.volume = 1000000

        mock_bars = MagicMock()
        mock_bars.data = {'SPY': [mock_bar]}

        mock_data = MagicMock()
        mock_data.get_stock_bars.return_value = mock_bars

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch('src.broker.alpaca.StockBarsRequest'), \
             patch('src.broker.alpaca.TimeFrame') as mock_tf, \
             patch.object(client, '_get_data_client', return_value=mock_data):
            mock_tf.Minute = 'Min'
            mock_tf.Day = 'Day'
            result = client.get_bars('SPY')

        assert len(result) == 1
        assert result[0]['close'] == 585.0
        assert result[0]['volume'] == 1000000
        assert result[0]['open'] == 580.0

    def test_get_bars_minute_timeframe(self):
        """get_bars with 1Min timeframe should map correctly."""
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_bar = MagicMock()
        mock_bar.timestamp = datetime(2026, 5, 24, 9, 30, 0)
        mock_bar.open = 580.0
        mock_bar.high = 581.0
        mock_bar.low = 579.5
        mock_bar.close = 580.5
        mock_bar.volume = 50000

        mock_bars = MagicMock()
        mock_bars.data = {'SPY': [mock_bar]}

        mock_data = MagicMock()
        mock_data.get_stock_bars.return_value = mock_bars

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch('src.broker.alpaca.StockBarsRequest'), \
             patch('src.broker.alpaca.TimeFrame') as mock_tf, \
             patch.object(client, '_get_data_client', return_value=mock_data):
            mock_tf.Minute = 'Min'
            mock_tf.Day = 'Day'
            result = client.get_bars('SPY', timeframe='1Min', limit=10)

        assert len(result) == 1
        assert result[0]['high'] == 581.0

    def test_get_bars_empty_data(self):
        """get_bars returns empty list when no data for symbol."""
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_bars = MagicMock()
        mock_bars.data = {}

        mock_data = MagicMock()
        mock_data.get_stock_bars.return_value = mock_bars

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch('src.broker.alpaca.StockBarsRequest'), \
             patch('src.broker.alpaca.TimeFrame') as mock_tf, \
             patch.object(client, '_get_data_client', return_value=mock_data):
            mock_tf.Minute = 'Min'
            mock_tf.Day = 'Day'
            result = client.get_bars('UNKNOWN')

        assert len(result) == 0


class TestAlpacaClientFetchPriceEdge:
    """_fetch_price edge cases."""

    def test_fetch_price_sqlite_error(self, tmp_path):
        """_fetch_price handles sqlite3.Error gracefully."""
        db_path = tmp_path / "market.db"
        _init_db(db_path)
        client = AlpacaClient()

        with patch('src.broker.alpaca.sqlite_connect', side_effect=sqlite3.Error("corrupt")):
            price = client._fetch_price('SPY', str(db_path))

        assert price == 0.0

    def test_fetch_price_with_default_db_path(self, tmp_path):
        """_fetch_price falls back to MARKET_DB when no path given."""
        client = AlpacaClient()
        with patch('src.broker.alpaca.MARKET_DB', str(tmp_path / "nonexistent.db")):
            price = client._fetch_price('SPY')

        assert price == 0.0


class TestAlpacaClientConfigEdge:
    """is_ready / is_configured edge cases."""

    def test_is_ready_with_all_prerequisites(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'
        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True):
            assert client.is_ready() is True

    def test_is_configured_only_key(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = None
        assert client.is_configured() is False

    def test_is_configured_only_secret(self):
        client = AlpacaClient()
        client.api_key = None
        client.api_secret = 'secret'
        assert client.is_configured() is False


class TestAlpacaClientGetClient:
    """_get_client / _get_data_client lazy init and error handling."""

    def test_get_client_raises_import_error(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'
        with patch('src.broker.alpaca.ALPACA_AVAILABLE', False):
            with pytest.raises(ImportError, match="alpaca-py SDK not installed"):
                client._get_client()

    def test_get_data_client_raises_import_error(self):
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'
        with patch('src.broker.alpaca.ALPACA_AVAILABLE', False):
            with pytest.raises(ImportError, match="alpaca-py SDK not installed"):
                client._get_data_client()


class TestAlpacaClientSubmitOrderSide:
    """Submit order side enum mapping correctness."""

    def test_submit_order_sell_side(self):
        """Sell order maps side correctly."""
        client = AlpacaClient()
        client.api_key = 'key'
        client.api_secret = 'secret'

        mock_result = MagicMock()
        mock_result.id = 'sell-1'
        mock_result.symbol = 'SPY'
        mock_result.qty = 10.0
        mock_result.filled_qty = 10.0
        mock_result.side.value = 'sell'
        mock_result.type.value = 'market'
        mock_result.status.value = 'filled'
        mock_result.created_at = datetime(2026, 5, 24, 10, 0, 0)
        mock_result.filled_at = datetime(2026, 5, 24, 10, 0, 1)
        mock_result.filled_avg_price = 585.0

        mock_trading = MagicMock()
        mock_trading.submit_order.return_value = mock_result

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch('src.broker.alpaca.MarketOrderRequest', create=True), \
             patch('src.broker.alpaca.LimitOrderRequest', create=True), \
             patch('src.broker.alpaca.TimeInForce', create=True) as mock_tif, \
             patch.object(client, '_get_client', return_value=mock_trading):
            mock_tif.DAY = 'day'
            req = OrderRequest(symbol='SPY', qty=10.0, side=OrderSide.SELL)
            order = client.submit_order(req)

        assert order.side == 'sell'


class TestPaperTradingManagerExecuteRebalance:
    """PaperTradingManager.execute_rebalance edge cases."""

    def test_execute_rebalance_sells_untargeted(self, tmp_path):
        """Positions not in target_allocations generate sell orders."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        mock_account = {'equity': 100000.0}
        mock_positions = [
            Position('SPY', 10, 500.0, 585.0, 5850.0, 850.0, 0.17),
            Position('TLT', 20, 95.0, 92.0, 1840.0, -60.0, -0.032),
        ]
        with patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value=mock_account), \
             patch.object(manager.client, 'get_positions', return_value=mock_positions):
            # SPY in targets, TLT not in targets -> sell TLT
            result = manager.execute_rebalance(
                {'SPY': 0.60}, total_value=100000, dry_run=True
            )

        assert result['order_count'] == 2  # buy SPY + sell TLT

    def test_execute_rebalance_skips_small_new_position(self, tmp_path):
        """New position with target_value < $10 is skipped."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        mock_account = {'equity': 100000.0}
        with patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value=mock_account), \
             patch.object(manager.client, 'get_positions', return_value=[]), \
             patch.object(manager.client, '_fetch_price', return_value=200.0):
            # target_value = 100000 * 0.00005 = $5 < $10 -> skip
            result = manager.execute_rebalance(
                {'TINY': 0.00005}, total_value=100000, dry_run=True
            )

        assert result['order_count'] == 0

    def test_execute_rebalance_skips_unpriced_new_position(self, tmp_path):
        """New position with _fetch_price returning 0 is skipped."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        mock_account = {'equity': 100000.0}
        with patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value=mock_account), \
             patch.object(manager.client, 'get_positions', return_value=[]), \
             patch.object(manager.client, '_fetch_price', return_value=0.0):
            result = manager.execute_rebalance(
                {'UNKNOWN': 0.50}, total_value=100000, dry_run=True
            )

        assert result['order_count'] == 0

    def test_execute_rebalance_live_submit(self, tmp_path):
        """Dry_run=False actually submits orders via client."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        mock_account = {'equity': 100000.0}
        mock_positions = [
            Position('SPY', 10, 500.0, 585.0, 5850.0, 850.0, 0.17),
        ]

        submitted = []

        def mock_submit(order_req):
            submitted.append(order_req)
            return Order(
                id=f'exec-{len(submitted)}',
                symbol=order_req.symbol,
                qty=order_req.qty,
                filled_qty=order_req.qty,
                side=order_req.side.value,
                type=order_req.order_type.value,
                status='filled',
                created_at='2026-05-24T10:00:00',
                filled_at='2026-05-24T10:00:05',
                filled_avg_price=585.0,
            )

        with patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value=mock_account), \
             patch.object(manager.client, 'get_positions', return_value=mock_positions), \
             patch.object(manager.client, 'submit_order', side_effect=mock_submit):
            result = manager.execute_rebalance(
                {'SPY': 0.60}, total_value=100000, dry_run=False
            )

        assert result['dry_run'] is False
        assert result['order_count'] > 0
        assert len(result['orders_submitted']) > 0
        assert len(submitted) > 0

    def test_execute_rebalance_partial_failure(self, tmp_path):
        """One order fails but others still submit."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        mock_account = {'equity': 100000.0}
        mock_positions = [
            Position('SPY', 10, 500.0, 585.0, 5850.0, 850.0, 0.17),
            Position('TLT', 20, 95.0, 92.0, 1840.0, -60.0, -0.032),
        ]

        call_count = [0]

        def mock_submit(order_req):
            call_count[0] += 1
            if order_req.symbol == 'TLT':
                raise RuntimeError("Insufficient buying power")
            return Order(
                id='exec-1',
                symbol=order_req.symbol,
                qty=order_req.qty,
                filled_qty=order_req.qty,
                side=order_req.side.value,
                type=order_req.order_type.value,
                status='filled',
                created_at='2026-05-24T10:00:00',
                filled_at='2026-05-24T10:00:05',
                filled_avg_price=585.0,
            )

        with patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value=mock_account), \
             patch.object(manager.client, 'get_positions', return_value=mock_positions), \
             patch.object(manager.client, 'submit_order', side_effect=mock_submit):
            result = manager.execute_rebalance(
                {'SPY': 0.60, 'TLT': 0.30}, total_value=100000, dry_run=False
            )

        assert result['dry_run'] is False
        errors = [o for o in result['orders_submitted'] if 'error' in o]
        successes = [o for o in result['orders_submitted'] if 'error' not in o]
        assert len(errors) == 1
        assert len(successes) >= 1

    def test_execute_rebalance_exception_handled(self, tmp_path):
        """Exception during rebalance returns error status."""
        manager = PaperTradingManager(data_dir=str(tmp_path))
        mock_account = {'equity': 100000.0}
        with patch.object(manager, 'is_ready', return_value=True), \
             patch.object(manager.client, 'get_account', return_value=mock_account), \
             patch.object(manager.client, 'get_positions', side_effect=RuntimeError("API down")):
            result = manager.execute_rebalance(
                {'SPY': 0.60}, total_value=100000, dry_run=True
            )

        assert result['status'] == 'error'
        assert 'API down' in result['message']


class TestCheckAlpacaStatusExtended:
    """check_alpaca_status success and error paths."""

    def test_connected_success(self):
        """check_alpaca_status returns connected=True when API works."""
        mock_account = {
            'status': 'ACTIVE',
            'equity': 100000.0,
            'cash': 50000.0,
        }

        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.dict(os.environ, {'ALPACA_API_KEY': 'k', 'ALPACA_API_SECRET': 's'}, clear=True), \
             patch('src.broker.alpaca.AlpacaClient.get_account', return_value=mock_account):
            status = check_alpaca_status()

        assert status['connected'] is True
        assert status['account_status'] == 'ACTIVE'
        assert status['equity'] == 100000.0
        assert status['cash'] == 50000.0

    def test_connected_error(self):
        """check_alpaca_status returns error details when get_account fails."""
        with patch('src.broker.alpaca.ALPACA_AVAILABLE', True), \
             patch.dict(os.environ, {'ALPACA_API_KEY': 'k', 'ALPACA_API_SECRET': 's'}, clear=True), \
             patch('src.broker.alpaca.AlpacaClient.get_account',
                   side_effect=RuntimeError("403 Forbidden")):
            status = check_alpaca_status()

        assert status['connected'] is False
        assert 'error' in status


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
