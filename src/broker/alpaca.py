"""
Alpaca broker API client for paper and live trading.
Supports fractional shares, paper trading without KYC, and WebSocket streaming.

Circuit breaker integration (``src.broker.circuit_breaker``):
    - ``submit_order`` is decorated with ``@broker_breaker`` -- when the
      circuit is open the call is short-circuited and returns ``None``.
    - ``get_positions`` uses ``broker_breaker.call()`` -- when open it
      returns an empty list.
"""
import logging
import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Mapping
from datetime import datetime, time as dt_time, timezone
from dataclasses import dataclass, asdict
from enum import Enum
from zoneinfo import ZoneInfo

from src.paths import MARKET_DB, DATA_DIR, sqlite_connect
from src.broker.circuit_breaker import (
    BrokerError,
    CircuitBreakerError,
    PYBREAKER_AVAILABLE,
    broker_breaker,
    get_circuit_state,
)


__all__ = ['OrderRequest', 'Order', 'Position', 'MarketQuote', 'QuoteStalenessError',
           'MarketSessionGuardError',
           'resolve_alpaca_feed_entitlement', 'resolve_alpaca_market_session',
           'resolve_unavailable_alpaca_market_session',
           'AlpacaClient', 'PaperTradingManager',
           'check_alpaca_status', 'get_circuit_state', 'RampPhase', 'LiveTransitionManager']

logger = logging.getLogger(__name__)

# Alpaca SDK - optional dependency
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
    # Use SDK's enums
    OrderSide = AlpacaOrderSide
    OrderType = None  # SDK doesn't have this enum, we handle it differently
except ImportError:
    ALPACA_AVAILABLE = False
    # Define our own enums when SDK not available
    class OrderSide(Enum):
        BUY = "buy"
        SELL = "sell"

    class OrderType(Enum):
        MARKET = "market"
        LIMIT = "limit"

    # Stubs so tests can patch these names on the module
    MarketOrderRequest = None
    LimitOrderRequest = None
    TimeInForce = None
    StockBarsRequest = None
    TimeFrame = None


@dataclass
class OrderRequest:
    symbol: str
    qty: float
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    time_in_force: str = "day"  # day, gtc, opg, cls, ioc, fok
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "side": self.side.value,
            "type": self.order_type.value,
            "limit_price": self.limit_price,
            "time_in_force": self.time_in_force,
        }


@dataclass
class Order:
    id: str
    symbol: str
    qty: float
    filled_qty: float
    side: str
    type: str
    status: str  # pending, filled, partial, cancelled
    created_at: str
    filled_at: Optional[str] = None
    filled_avg_price: Optional[float] = None
    
    @classmethod
    def from_alpaca(cls, order) -> "Order":
        return cls(
            id=str(order.id),
            symbol=order.symbol,
            qty=float(order.qty) if order.qty else 0.0,
            filled_qty=float(order.filled_qty) if order.filled_qty else 0.0,
            side=order.side.value if hasattr(order.side, 'value') else str(order.side),
            type=order.type.value if hasattr(order.type, 'value') else str(order.type),
            status=order.status.value if hasattr(order.status, 'value') else str(order.status),
            created_at=order.created_at.isoformat() if hasattr(order.created_at, 'isoformat') else str(order.created_at),
            filled_at=order.filled_at.isoformat() if hasattr(order.filled_at, 'isoformat') else None,
            filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
        )


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float
    
    @classmethod
    def from_alpaca(cls, position) -> "Position":
        return cls(
            symbol=position.symbol,
            qty=float(position.qty),
            avg_entry_price=float(position.avg_entry_price),
            current_price=float(position.current_price),
            market_value=float(position.market_value),
            unrealized_pl=float(position.unrealized_pl),
            unrealized_plpc=float(position.unrealized_plpc),
        )


@dataclass
class MarketQuote:
    """Quote used for order sizing with explicit source and age metadata."""
    symbol: str
    price: float
    timestamp: str
    source: str
    feed: str
    age_seconds: float
    source_mode: str
    feed_entitlement: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QuoteStalenessError(RuntimeError):
    """Raised when live order sizing would use stale or prior-close data."""


class MarketSessionGuardError(RuntimeError):
    """Raised when live order sizing/submission would violate market-session policy."""

    def __init__(self, message: str, market_session: Dict[str, Any]):
        super().__init__(message)
        self.market_session = market_session


def _normalize_feed_value(value: Optional[str]) -> str:
    normalized = (value or "iex").strip().lower().replace("-", "_")
    aliases = {
        "basic": "iex",
        "alpaca_basic": "iex",
        "delayed": "delayed_sip",
        "delayed_sip": "delayed_sip",
        "sip_delayed": "delayed_sip",
    }
    return aliases.get(normalized, normalized)


def _normalize_entitlement_value(value: Optional[str]) -> str:
    normalized = (value or "").strip().lower().replace("-", "_")
    if not normalized:
        return "unknown"
    aliases = {
        "iex": "basic",
        "alpaca_basic": "basic",
        "delayed": "delayed_sip",
        "sip_delayed": "delayed_sip",
    }
    return aliases.get(normalized, normalized)


def resolve_alpaca_feed_entitlement(env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Return public-safe Alpaca market-data feed entitlement metadata.

    The output intentionally records only feed class and entitlement policy,
    never account identifiers, API keys, or credential values.
    """
    values = env if env is not None else os.environ
    configured_feed = _normalize_feed_value(
        values.get("ALPACA_DATA_FEED")
        or values.get("ALPACA_MARKET_DATA_FEED")
        or values.get("ALPACA_FEED")
    )
    entitlement = _normalize_entitlement_value(
        values.get("ALPACA_FEED_ENTITLEMENT")
        or values.get("ALPACA_MARKET_DATA_ENTITLEMENT")
        or values.get("ALPACA_DATA_ENTITLEMENT")
    )

    known_feeds = {"iex", "sip", "delayed_sip", "websocket", "historical_bars"}
    effective_feed = configured_feed if configured_feed in known_feeds else "unknown"
    delayed = effective_feed == "delayed_sip" or entitlement == "delayed_sip"
    acceptable_for_live = False
    reason: Optional[str] = None

    if effective_feed == "unknown":
        reason = "unknown_feed"
    elif entitlement == "unknown":
        reason = "missing_entitlement"
    elif delayed:
        reason = "delayed_feed"
    elif effective_feed == "sip" and entitlement != "sip":
        reason = "insufficient_entitlement"
    elif effective_feed in {"iex", "websocket", "historical_bars"} and entitlement in {"basic", "sip"}:
        acceptable_for_live = True
    elif effective_feed == "sip" and entitlement == "sip":
        acceptable_for_live = True
    else:
        reason = "insufficient_entitlement"

    return {
        "configured_feed": configured_feed,
        "effective_feed": effective_feed,
        "entitlement": entitlement,
        "delayed": delayed,
        "acceptable_for_live": acceptable_for_live,
        "policy_decision": "allow" if acceptable_for_live else "reject",
        "reason": reason,
    }


def _truthy_env(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_session_state(value: Optional[str]) -> str:
    normalized = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "open": "regular",
        "regular_session": "regular",
        "market_open": "regular",
        "extended": "extended_hours",
        "extended_hour": "extended_hours",
        "after_hours": "extended_hours",
        "premarket": "extended_hours",
        "pre_market": "extended_hours",
        "market_closed": "closed",
        "halt": "halted",
        "halted": "halted",
    }
    return aliases.get(normalized, normalized or "unknown")


def _clock_value(clock: Mapping[str, Any], key: str) -> Any:
    if key in clock:
        return clock[key]
    return getattr(clock, key, None)


def _iso_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_extended_hours(timestamp: Any) -> bool:
    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        return False
    eastern = parsed.astimezone(ZoneInfo("America/New_York"))
    if eastern.weekday() >= 5:
        return False
    current = eastern.time()
    premarket = dt_time(4, 0) <= current < dt_time(9, 30)
    after_hours = dt_time(16, 0) < current <= dt_time(20, 0)
    return premarket or after_hours


def resolve_alpaca_market_session(
    clock: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Return public-safe market-session policy metadata for live order guards."""
    values = env if env is not None else os.environ
    extended_hours_allowed = _truthy_env(
        values.get("ALPACA_ALLOW_EXTENDED_HOURS")
        or values.get("BROKER_ALLOW_EXTENDED_HOURS")
    )
    override_state = _normalize_session_state(
        values.get("ALPACA_MARKET_SESSION_STATE")
        or values.get("BROKER_MARKET_SESSION_STATE")
    )

    timestamp = None
    next_open = None
    next_close = None
    is_open = False
    source = "env_override" if override_state != "unknown" else "unavailable"

    if clock is not None:
        timestamp = _clock_value(clock, "timestamp")
        next_open = _clock_value(clock, "next_open")
        next_close = _clock_value(clock, "next_close")
        is_open = bool(_clock_value(clock, "is_open"))
        source = "broker_clock"

    if override_state != "unknown":
        session_state = override_state
    elif clock is None:
        session_state = "unknown"
    elif bool(_clock_value(clock, "halted")):
        session_state = "halted"
    elif _normalize_session_state(_clock_value(clock, "session_state")) != "unknown":
        session_state = _normalize_session_state(_clock_value(clock, "session_state"))
    elif is_open:
        session_state = "regular"
    elif _is_extended_hours(timestamp):
        session_state = "extended_hours"
    else:
        session_state = "closed"

    allow_live_orders = False
    reason: Optional[str] = None

    if session_state == "regular":
        allow_live_orders = True
    elif session_state == "extended_hours":
        allow_live_orders = extended_hours_allowed
        if not allow_live_orders:
            reason = "extended_hours_not_allowed"
    elif session_state == "closed":
        reason = "market_closed"
    elif session_state == "halted":
        reason = "market_halted"
    else:
        reason = "market_session_unknown"

    return {
        "session_state": session_state,
        "is_open": is_open,
        "timestamp": _iso_or_none(timestamp),
        "next_open": _iso_or_none(next_open),
        "next_close": _iso_or_none(next_close),
        "extended_hours_allowed": extended_hours_allowed,
        "allow_live_orders": allow_live_orders,
        "guard_decision": "allow" if allow_live_orders else "reject",
        "reason": reason,
        "source": source,
    }


def resolve_unavailable_alpaca_market_session(error: Optional[BaseException] = None) -> Dict[str, Any]:
    """Return fail-closed session diagnostics when the broker clock is unavailable."""
    session = resolve_alpaca_market_session()
    session["source"] = "unavailable"
    session["reason"] = "market_session_unavailable"
    if error is not None:
        session["error_type"] = type(error).__name__
    return session


class AlpacaClient:
    """
    Alpaca broker client with paper trading support.
    
    Paper trading requires only API key/secret (no KYC).
    Live trading requires funded account.
    """
    
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.api_key = os.environ.get("ALPACA_API_KEY")
        self.api_secret = os.environ.get("ALPACA_API_SECRET")
        self._trading_client: Optional[Any] = None
        self._data_client: Optional[Any] = None
        
    def is_configured(self) -> bool:
        """Check if API credentials are available."""
        return bool(self.api_key and self.api_secret)

    def _fetch_price_quote(
        self,
        symbol: str,
        db_path: str = None,
        *,
        now: Optional[datetime] = None,
    ) -> Optional[MarketQuote]:
        """Fetch latest local close quote from market.db with timestamp metadata."""
        if db_path is None:
            db_path = str(MARKET_DB)
        try:
            if not os.path.exists(db_path):
                return None
            with sqlite_connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT close, date FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT 1",
                    (symbol,)
                )
                row = cursor.fetchone()
            if not row:
                return None
            resolved_now = now or datetime.now(timezone.utc)
            if resolved_now.tzinfo is None:
                resolved_now = resolved_now.replace(tzinfo=timezone.utc)
            close, date_str = row
            quote_ts = datetime.fromisoformat(f"{date_str}T21:00:00+00:00")
            age_seconds = max((resolved_now.astimezone(timezone.utc) - quote_ts).total_seconds(), 0.0)
            return MarketQuote(
                symbol=symbol,
                price=float(close),
                timestamp=quote_ts.isoformat(),
                source="market_db",
                feed="yahoo_close",
                age_seconds=age_seconds,
                source_mode="prior_close",
            )
        except sqlite3.Error:
            logger.warning("Failed to fetch price from market.db for %s", symbol)
            return None

    def _fetch_price(self, symbol: str, db_path: str = None) -> float:
        """Fetch latest price from market.db. Returns 0 if unavailable."""
        quote = self._fetch_price_quote(symbol, db_path)
        return quote.price if quote is not None else 0.0

    def get_latest_quote(self, symbol: str, *, now: Optional[datetime] = None) -> Optional[MarketQuote]:
        """Fetch latest broker quote when Alpaca market data is available.

        The method is best-effort because the Alpaca data SDK is optional in
        this project. Tests patch this method directly; production falls back
        to local prior-close data when broker quotes are unavailable.
        """
        try:
            feed_entitlement = resolve_alpaca_feed_entitlement()
            client = self._get_data_client()
            getter = getattr(client, "get_stock_latest_quote", None)
            if getter is None:
                return None
            quote = getter(symbol)
            bid = float(getattr(quote, "bid_price", 0) or 0)
            ask = float(getattr(quote, "ask_price", 0) or 0)
            price = ask if ask > 0 else bid
            if bid > 0 and ask > 0:
                price = (bid + ask) / 2
            if price <= 0:
                return None
            ts = getattr(quote, "timestamp", None) or datetime.now(timezone.utc)
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if isinstance(ts, str):
                quote_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif isinstance(ts, datetime):
                quote_dt = ts
            else:
                quote_dt = datetime.now(timezone.utc)
            if quote_dt.tzinfo is None:
                quote_dt = quote_dt.replace(tzinfo=timezone.utc)
            resolved_now = now or datetime.now(timezone.utc)
            if resolved_now.tzinfo is None:
                resolved_now = resolved_now.replace(tzinfo=timezone.utc)
            age_seconds = max((resolved_now.astimezone(timezone.utc) - quote_dt.astimezone(timezone.utc)).total_seconds(), 0.0)
            return MarketQuote(
                symbol=symbol,
                price=price,
                timestamp=quote_dt.astimezone(timezone.utc).isoformat(),
                source="alpaca",
                feed=feed_entitlement["effective_feed"],
                age_seconds=age_seconds,
                source_mode="delayed" if feed_entitlement["delayed"] else "live",
                feed_entitlement=feed_entitlement,
            )
        except (ImportError, RuntimeError, OSError, ConnectionError, TimeoutError, ValueError, TypeError) as e:
            logger.debug("Broker quote unavailable for %s: %s", symbol, e)
            return None

    def is_available(self) -> bool:
        """Check if alpaca-py SDK is installed."""
        return ALPACA_AVAILABLE
    
    def is_ready(self) -> bool:
        """Check if client is fully ready (SDK + configured)."""
        return self.is_available() and self.is_configured()
    
    def _get_client(self):
        """Lazy initialization of trading client."""
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py SDK not installed. Run: pip install alpaca-py")
        if not self._trading_client:
            if not self.is_configured():
                raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET not set")
            self._trading_client = TradingClient(
                self.api_key, 
                self.api_secret, 
                paper=self.paper
            )
        return self._trading_client
    
    def _get_data_client(self):
        """Lazy initialization of data client."""
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py SDK not installed")
        if not self._data_client:
            if not self.is_configured():
                raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET not set")
            self._data_client = StockHistoricalDataClient(
                self.api_key, 
                self.api_secret
            )
        return self._data_client
    
    def get_account(self) -> Dict[str, Any]:
        """Get account details."""
        client = self._get_client()
        account = client.get_account()
        return {
            "id": account.id,
            "status": account.status,
            "currency": account.currency,
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "maintenance_margin": float(account.maintenance_margin),
            "initial_margin": float(account.initial_margin),
            "daytrade_count": account.daytrade_count,
            "last_equity": float(account.last_equity) if account.last_equity else None,
            "paper": self.paper,
        }
    
    @broker_breaker
    def submit_order(self, order: OrderRequest) -> Optional[Order]:
        """Submit a new order. Returns None on API/network failure or when
        the circuit breaker is open."""
        client = self._get_client()

        # Convert our OrderRequest to Alpaca format
        side_enum = OrderSide.BUY if order.side == OrderSide.BUY else OrderSide.SELL

        if order.order_type == OrderType.MARKET:
            alpaca_order = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side_enum,
                time_in_force=getattr(TimeInForce, order.time_in_force.upper(), TimeInForce.DAY)
            )
        else:
            if order.limit_price is None:
                raise ValueError("Limit price required for limit orders")
            alpaca_order = LimitOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side_enum,
                time_in_force=getattr(TimeInForce, order.time_in_force.upper(), TimeInForce.DAY),
                limit_price=order.limit_price
            )

        try:
            result = client.submit_order(alpaca_order)
            return Order.from_alpaca(result)
        except (OSError, ConnectionError, TimeoutError, KeyError, ValueError, TypeError, RuntimeError) as e:
            logger.warning("Error submitting order for %s: %s", order.symbol, e)
            raise BrokerError(str(e)) from e

    def get_orders(self, status: Optional[str] = None, limit: int = 100) -> List[Order]:
        """Get list of orders. Returns empty list on API/network failure."""
        client = self._get_client()

        try:
            orders = client.get_orders(limit=limit)
            return [Order.from_alpaca(o) for o in orders]
        except (OSError, ConnectionError, TimeoutError, KeyError, ValueError, TypeError, RuntimeError) as e:
            logger.warning("Error fetching orders: %s", e)
            return []
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        client = self._get_client()
        try:
            client.cancel_order_by_id(order_id)
            return True
        except (OSError, ConnectionError, TimeoutError, KeyError, ValueError, TypeError, RuntimeError) as e:
            logger.warning("Error cancelling order %s: %s", order_id, e)
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        client = self._get_client()
        try:
            result = client.cancel_orders()
            return len(result) if result else 0
        except (OSError, ConnectionError, TimeoutError, KeyError, ValueError, TypeError, RuntimeError) as e:
            logger.warning("Error cancelling all orders: %s", e)
            return 0

    def get_positions(self) -> List[Position]:
        """Get current positions.

        Returns [] on API/network failure or when the circuit breaker is
        open.
        """
        try:
            return broker_breaker.call(self._get_positions_impl)
        except (CircuitBreakerError, BrokerError):
            return []

    def _get_positions_impl(self) -> List[Position]:
        """Inner implementation of get_positions (no circuit breaker)."""
        client = self._get_client()
        try:
            positions = client.get_all_positions()
            return [Position.from_alpaca(p) for p in positions]
        except (OSError, ConnectionError, TimeoutError, KeyError, ValueError, TypeError, RuntimeError) as e:
            logger.warning("Error fetching positions: %s", e)
            raise BrokerError(str(e)) from e
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get specific position."""
        client = self._get_client()
        try:
            position = client.get_open_position(symbol)
            return Position.from_alpaca(position)
        except (OSError, ConnectionError, TimeoutError, KeyError, ValueError, TypeError, RuntimeError) as e:
            logger.warning("Failed to get position for %s: %s", symbol, e)
            return None

    def close_position(self, symbol: str, qty: Optional[float] = None) -> Order:
        """Close a position (fully or partially)."""
        client = self._get_client()
        try:
            if qty:
                result = client.close_position(symbol, qty)
            else:
                result = client.close_position(symbol)
            return Order.from_alpaca(result)
        except (OSError, ConnectionError, TimeoutError, KeyError, ValueError, TypeError, RuntimeError) as e:
            raise RuntimeError(f"Failed to close position {symbol}: {e}")

    def close_all_positions(self) -> List[Order]:
        """Close all positions."""
        client = self._get_client()
        try:
            results = client.close_all_positions(cancel_orders=True)
            return [Order.from_alpaca(o) for o in results]
        except (OSError, ConnectionError, TimeoutError, KeyError, ValueError, TypeError, RuntimeError) as e:
            raise RuntimeError(f"Failed to close all positions: {e}")
    
    def get_clock(self) -> Dict[str, Any]:
        """Get market clock."""
        client = self._get_client()
        clock = client.get_clock()
        return {
            "timestamp": clock.timestamp.isoformat(),
            "is_open": clock.is_open,
            "next_open": clock.next_open.isoformat() if clock.next_open else None,
            "next_close": clock.next_close.isoformat() if clock.next_close else None,
        }

    def get_market_session(self) -> Dict[str, Any]:
        """Get public-safe market-session guard metadata.

        Environment overrides are honored before broker calls so tests and
        emergency operator controls can exercise the policy without touching
        the live Alpaca API.
        """
        has_override = any(
            os.environ.get(key)
            for key in ("ALPACA_MARKET_SESSION_STATE", "BROKER_MARKET_SESSION_STATE")
        )
        if has_override:
            return resolve_alpaca_market_session()
        try:
            return resolve_alpaca_market_session(self.get_clock())
        except (ImportError, RuntimeError, OSError, ConnectionError, TimeoutError, ValueError, TypeError) as exc:
            return resolve_unavailable_alpaca_market_session(exc)
    
    def get_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 100) -> List[Dict]:
        """Get historical bars."""
        client = self._get_data_client()
        
        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, TimeFrame.Minute),
            "15Min": TimeFrame(15, TimeFrame.Minute),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf_map.get(timeframe, TimeFrame.Day),
            limit=limit
        )
        
        bars = client.get_stock_bars(request_params)
        result = []
        for bar in bars.data.get(symbol, []):
            result.append({
                "timestamp": bar.timestamp.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            })
        return result


class PaperTradingManager:
    """
    Manager for paper trading integration with portfolio-lab.
    Handles order routing from local signals to Alpaca paper account.
    """
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = str(DATA_DIR)
        self.client = AlpacaClient(paper=True)
        self.data_dir = data_dir
        self.orders_file = os.path.join(data_dir, "broker_orders.jsonl")
        
    def is_ready(self) -> bool:
        """Check if paper trading can be activated."""
        return self.client.is_available() and self.client.is_configured()

    def _max_quote_age_seconds(self) -> int:
        return int(os.environ.get("BROKER_MAX_QUOTE_AGE_SECONDS", "900"))

    def _is_live_order_mode(self, dry_run: bool) -> bool:
        paper_mode = os.environ.get("ALPACA_PAPER", "true").lower() not in ("false", "0", "no")
        return not dry_run and not paper_mode

    def _position_quote(self, position: Position) -> MarketQuote:
        timestamp = datetime.now(timezone.utc).isoformat()
        return MarketQuote(
            symbol=position.symbol,
            price=position.current_price,
            timestamp=timestamp,
            source="alpaca_position",
            feed="positions",
            age_seconds=0.0,
            source_mode="live",
            feed_entitlement=resolve_alpaca_feed_entitlement(),
        )

    def _raise_if_live_feed_unacceptable(
        self,
        symbol: str,
        dry_run: bool,
        feed_entitlement: Optional[Dict[str, Any]],
    ) -> None:
        if not self._is_live_order_mode(dry_run):
            return
        if feed_entitlement is None or feed_entitlement.get("acceptable_for_live", False):
            return
        reason = feed_entitlement.get("reason") or "unacceptable_feed"
        raise QuoteStalenessError(
            f"Alpaca feed entitlement rejects live order sizing for {symbol}: {reason}"
        )

    def _raise_if_live_market_session_rejected(
        self,
        dry_run: bool,
        market_session: Dict[str, Any],
    ) -> None:
        if not self._is_live_order_mode(dry_run):
            return
        if market_session.get("allow_live_orders", False):
            return
        reason = market_session.get("reason") or "market_session_rejected"
        raise MarketSessionGuardError(
            f"Market session guard rejects live order sizing: {reason}",
            market_session,
        )

    def _resolve_order_quote(self, symbol: str, *, dry_run: bool) -> Optional[MarketQuote]:
        """Resolve quote for order sizing, enforcing freshness for live orders."""
        max_age = self._max_quote_age_seconds()
        broker_quote = self.client.get_latest_quote(symbol)
        if broker_quote is not None:
            feed_entitlement = broker_quote.feed_entitlement or resolve_alpaca_feed_entitlement()
            broker_quote.feed_entitlement = feed_entitlement
            if feed_entitlement.get("delayed"):
                broker_quote.source_mode = "delayed"

            if broker_quote.age_seconds <= max_age:
                self._raise_if_live_feed_unacceptable(symbol, dry_run, feed_entitlement)
                return broker_quote

            broker_quote.source_mode = "delayed"
            if self._is_live_order_mode(dry_run):
                raise QuoteStalenessError(
                    f"Broker quote for {symbol} is stale: {broker_quote.age_seconds:.0f}s > {max_age}s"
                )
            return broker_quote

        local_quote = self.client._fetch_price_quote(symbol)
        if local_quote is None:
            legacy_price = self.client._fetch_price(symbol)
            if legacy_price > 0:
                local_quote = MarketQuote(
                    symbol=symbol,
                    price=legacy_price,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="market_db",
                    feed="legacy_latest_close",
                    age_seconds=0.0,
                    source_mode="prior_close",
                )
        if local_quote is None:
            return None
        if self._is_live_order_mode(dry_run):
            raise QuoteStalenessError(
                f"Live order sizing for {symbol} requires a fresh broker quote; only prior close is available"
            )
        return local_quote
    
    def sync_positions(self) -> Dict[str, Any]:
        """Sync Alpaca positions with local tracking."""
        if not self.is_ready():
            return {"status": "not_configured", "message": "Alpaca API not configured"}
        
        try:
            positions = self.client.get_positions()
            account = self.client.get_account()
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "paper": True,
                "account": account,
                "positions": [asdict(p) for p in positions],
                "position_count": len(positions),
            }
            
            # Append to local tracking file
            os.makedirs(self.data_dir, exist_ok=True)
            with open(self.orders_file, "a") as f:
                f.write(json.dumps({
                    "type": "position_sync",
                    "timestamp": result["timestamp"],
                    "positions": result["positions"],
                    "account_equity": account.get("equity"),
                }) + "\n")
            
            return result
        except (OSError, ConnectionError, TimeoutError, KeyError, ValueError, TypeError, RuntimeError) as e:
            return {"status": "error", "message": str(e)}

    def execute_rebalance(
        self, 
        target_allocations: Dict[str, float], 
        total_value: Optional[float] = None,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Execute portfolio rebalancing based on target allocations.
        
        Args:
            target_allocations: Dict of symbol -> target percentage (0-1)
            total_value: Total portfolio value (uses account equity if None)
            dry_run: If True, only calculate orders without submitting
            
        Returns:
            Dict with planned/executed orders
        """
        if not self.is_ready():
            return {"status": "not_configured", "message": "Alpaca API not configured"}
        market_session: Optional[Dict[str, Any]] = None
        try:
            account = self.client.get_account()
            positions = self.client.get_positions()
            market_session = self.client.get_market_session()
            self._raise_if_live_market_session_rejected(dry_run, market_session)
            
            if total_value is None:
                total_value = account["equity"]
            
            current_positions = {p.symbol: p for p in positions}
            
            orders_to_submit = []
            orders_submitted = []
            quote_sources: Dict[str, Dict[str, Any]] = {}
            
            for symbol, target_pct in target_allocations.items():
                target_value = total_value * target_pct
                
                if symbol in current_positions:
                    pos = current_positions[symbol]
                    current_value = pos.market_value
                    delta = target_value - current_value
                    
                    if abs(delta) < 10:  # Minimum $10 difference to trade
                        continue
                    
                    # Calculate shares to trade (rough estimate using current price)
                    qty = abs(delta) / pos.current_price
                    side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                    position_quote = self._position_quote(pos)
                    self._raise_if_live_feed_unacceptable(symbol, dry_run, position_quote.feed_entitlement)
                    quote_sources[symbol] = position_quote.to_dict()
                else:
                    # New position
                    if target_value < 10:
                        continue
                    quote = self._resolve_order_quote(symbol, dry_run=dry_run)
                    if quote is None or quote.price <= 0:
                        continue  # Skip if no price available
                    quote_sources[symbol] = quote.to_dict()
                    qty = target_value / quote.price
                    side = OrderSide.BUY
                
                order_req = OrderRequest(
                    symbol=symbol,
                    qty=round(qty, 4),
                    side=side,
                    order_type=OrderType.MARKET
                )
                orders_to_submit.append(order_req)
            
            # Handle sells for positions not in target
            for symbol, pos in current_positions.items():
                if symbol not in target_allocations:
                    order_req = OrderRequest(
                        symbol=symbol,
                        qty=pos.qty,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET
                    )
                    orders_to_submit.append(order_req)
            
            if not dry_run:
                for order_req in orders_to_submit:
                    try:
                        order = self.client.submit_order(order_req)
                        orders_submitted.append(asdict(order))
                    except (OSError, ConnectionError, TimeoutError, KeyError, ValueError, TypeError, RuntimeError) as e:
                        orders_submitted.append({
                            "error": str(e),
                            "request": order_req.to_dict()
                        })
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "paper": True,
                "dry_run": dry_run,
                "total_value": total_value,
                "account_equity": account["equity"],
                "target_allocations": target_allocations,
                "orders_planned": [o.to_dict() for o in orders_to_submit],
                "orders_submitted": orders_submitted if not dry_run else [],
                "order_count": len(orders_to_submit),
                "quote_sources": quote_sources,
                "market_session": market_session,
            }
            
            # Log to file
            with open(self.orders_file, "a") as f:
                f.write(json.dumps({
                    "type": "rebalance",
                    "timestamp": result["timestamp"],
                    "dry_run": dry_run,
                    "orders": result["orders_planned"],
                    "quote_sources": quote_sources,
                    "market_session": market_session,
                }) + "\n")
            
            return result

        except MarketSessionGuardError as e:
            return {"status": "error", "message": str(e), "market_session": e.market_session}
        except (OSError, ConnectionError, TimeoutError, KeyError, ValueError, TypeError, RuntimeError) as e:
            error: Dict[str, Any] = {"status": "error", "message": str(e)}
            if market_session is not None:
                error["market_session"] = market_session
            return error


# Convenience functions for CLI usage
def check_alpaca_status() -> Dict[str, Any]:
    """Quick status check for CLI/health monitoring.

    Detects paper vs live mode from ALPACA_PAPER env var (default: True).
    """
    paper_mode = os.environ.get("ALPACA_PAPER", "true").lower() not in ("false", "0", "no")
    client = AlpacaClient(paper=paper_mode)

    status = {
        "sdk_available": ALPACA_AVAILABLE,
        "configured": client.is_configured(),
        "paper": paper_mode,
    }

    if ALPACA_AVAILABLE and client.is_configured():
        try:
            account = client.get_account()
            status["account_status"] = account.get("status")
            status["equity"] = account.get("equity")
            status["cash"] = account.get("cash")
            status["connected"] = True
            # Live account compliance flags
            if not paper_mode:
                status["trading_blocked"] = account.get("trading_blocked", False)
                status["transfers_blocked"] = account.get("transfers_blocked", False)
                status["account_blocked"] = account.get("account_blocked", False)
                status["shorting_enabled"] = account.get("shorting_enabled", False)
        except (OSError, ConnectionError, TimeoutError, KeyError, ValueError, TypeError, RuntimeError) as e:
            status["connected"] = False
            status["error"] = str(e)
    else:
        status["connected"] = False
        if not ALPACA_AVAILABLE:
            status["error"] = "alpaca-py SDK not installed"
        elif not client.is_configured():
            status["error"] = "ALPACA_API_KEY and ALPACA_API_SECRET not set"

    status["feed_entitlement"] = resolve_alpaca_feed_entitlement()

    return status


# ---------------------------------------------------------------------------
# Paper → Live Transition: Ramp Protocol
# ---------------------------------------------------------------------------

class RampPhase(str, Enum):
    """5-phase ramp protocol for transitioning from paper to live trading.

    Each phase increases the allocation percentage. The system must pass
    graduation criteria and remain within risk bounds before advancing.

    Phases:
        PAPER:    Paper trading only (no real capital at risk)
        PHASE_1:  1% of target allocation in live
        PHASE_2:  5% of target allocation in live
        PHASE_3:  25% of target allocation in live
        PHASE_4:  50% of target allocation in live
        LIVE:     100% of target allocation in live
    """
    PAPER = "paper"
    PHASE_1 = "phase_1"      # 1%
    PHASE_2 = "phase_2"      # 5%
    PHASE_3 = "phase_3"      # 25%
    PHASE_4 = "phase_4"      # 50%
    LIVE = "live"            # 100%


# Allocation multiplier per ramp phase
RAMP_ALLOCATION_PCT: Dict[str, float] = {
    RampPhase.PAPER.value: 0.00,
    RampPhase.PHASE_1.value: 0.01,
    RampPhase.PHASE_2.value: 0.05,
    RampPhase.PHASE_3.value: 0.25,
    RampPhase.PHASE_4.value: 0.50,
    RampPhase.LIVE.value: 1.00,
}

# Minimum trading days required at each phase before advancing
RAMP_MIN_DAYS: Dict[str, int] = {
    RampPhase.PAPER.value: 63,     # Full graduation period
    RampPhase.PHASE_1.value: 21,
    RampPhase.PHASE_2.value: 21,
    RampPhase.PHASE_3.value: 42,
    RampPhase.PHASE_4.value: 42,
    RampPhase.LIVE.value: 0,       # Already at full allocation
}

# Maximum drawdown allowed at each phase before automatic rollback
RAMP_MAX_DRAWDOWN: Dict[str, float] = {
    RampPhase.PAPER.value: 1.0,
    RampPhase.PHASE_1.value: 0.10,
    RampPhase.PHASE_2.value: 0.12,
    RampPhase.PHASE_3.value: 0.15,
    RampPhase.PHASE_4.value: 0.20,
    RampPhase.LIVE.value: 0.25,
}

RAMP_STATE_FILE = DATA_DIR / "ramp_state.json"


class LiveTransitionManager:
    """Manages the paper→live transition ramp protocol.

    State is persisted to ``DATA_DIR/ramp_state.json``. The manager tracks:
    - Current ramp phase
    - Days accumulated at current phase
    - Peak equity and max drawdown at current phase
    - Compliance prerequisites before advancing

    Usage::

        mgr = LiveTransitionManager()
        status = mgr.get_status()
        can_advance = mgr.can_advance()
        if can_advance:
            mgr.advance()
    """

    def __init__(self, state_path: Optional[str] = None):
        self.state_path = Path(state_path) if state_path else RAMP_STATE_FILE
        self._state: Dict[str, Any] = {}
        self._load_state()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "phase": RampPhase.PAPER.value,
            "phase_started": None,
            "days_at_phase": 0,
            "peak_equity": None,
            "max_drawdown_pct": 0.0,
            "advancement_log": [],
        }

    def _load_state(self):
        """Load ramp state from disk."""
        if self.state_path.exists():
            try:
                self._state = json.loads(self.state_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load ramp state: %s", e)
                self._state = self._default_state()
        else:
            self._state = self._default_state()

    def _save_state(self):
        """Persist ramp state to disk."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self._state, indent=2))
        except OSError as e:
            logger.error("Failed to save ramp state: %s", e)

    @property
    def phase(self) -> RampPhase:
        return RampPhase(self._state.get("phase", RampPhase.PAPER.value))

    @property
    def allocation_pct(self) -> float:
        """Current allocation percentage for live trading."""
        return RAMP_ALLOCATION_PCT.get(self.phase.value, 0.0)

    @property
    def days_at_phase(self) -> int:
        return self._state.get("days_at_phase", 0)

    @property
    def max_drawdown_pct(self) -> float:
        return self._state.get("max_drawdown_pct", 0.0)

    def get_status(self) -> Dict[str, Any]:
        """Get full ramp status for dashboard integration."""
        return {
            "phase": self.phase.value,
            "allocation_pct": self.allocation_pct,
            "days_at_phase": self.days_at_phase,
            "min_days_required": RAMP_MIN_DAYS.get(self.phase.value, 0),
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_allowed": RAMP_MAX_DRAWDOWN.get(self.phase.value, 1.0),
            "can_advance": self.can_advance(),
            "alpaca_status": check_alpaca_status(),
        }

    def can_advance(self) -> bool:
        """Check if all prerequisites for advancing to next phase are met.

        Prerequisites:
        1. Not already at LIVE phase
        2. Minimum days at current phase accumulated
        3. Max drawdown within bounds
        4. Alpaca account is connected and not blocked
        5. Graduation criteria met (for PAPER → PHASE_1)
        """
        if self.phase == RampPhase.LIVE:
            return False

        # Check days requirement
        min_days = RAMP_MIN_DAYS.get(self.phase.value, 0)
        if self.days_at_phase < min_days:
            return False

        # Check drawdown bound
        max_dd_allowed = RAMP_MAX_DRAWDOWN.get(self.phase.value, 1.0)
        if self.max_drawdown_pct > max_dd_allowed:
            return False

        # Check Alpaca connectivity
        alpaca = check_alpaca_status()
        if not alpaca.get("connected", False):
            return False

        # Check live account is not blocked (for phases beyond PAPER)
        if self.phase != RampPhase.PAPER:
            if alpaca.get("trading_blocked", False):
                return False
            if alpaca.get("account_blocked", False):
                return False

        return True

    def advance(self) -> Dict[str, Any]:
        """Advance to the next ramp phase.

        Returns:
            Dict with success status and new phase info.

        Raises:
            RuntimeError: If prerequisites are not met.
        """
        if not self.can_advance():
            return {
                "success": False,
                "reason": "Prerequisites not met for advancement",
                "status": self.get_status(),
            }

        phase_order = [RampPhase.PAPER, RampPhase.PHASE_1, RampPhase.PHASE_2,
                       RampPhase.PHASE_3, RampPhase.PHASE_4, RampPhase.LIVE]
        current_idx = phase_order.index(self.phase)
        next_phase = phase_order[current_idx + 1]

        log_entry = {
            "from": self.phase.value,
            "to": next_phase.value,
            "timestamp": datetime.now().isoformat(),
            "allocation_pct": RAMP_ALLOCATION_PCT[next_phase.value],
        }

        self._state["phase"] = next_phase.value
        self._state["phase_started"] = datetime.now().isoformat()
        self._state["days_at_phase"] = 0
        self._state["peak_equity"] = None
        self._state["max_drawdown_pct"] = 0.0
        self._state["advancement_log"] = self._state.get("advancement_log", []) + [log_entry]

        self._save_state()
        logger.info("Ramp advanced: %s → %s (allocation: %.0f%%)",
                     log_entry["from"], next_phase.value, log_entry["allocation_pct"] * 100)

        return {"success": True, "new_phase": next_phase.value, **log_entry}

    def rollback(self, reason: str = "") -> Dict[str, Any]:
        """Roll back to the previous ramp phase due to risk breach.

        Returns:
            Dict with rollback status.
        """
        if self.phase == RampPhase.PAPER:
            return {"success": False, "reason": "Already at PAPER phase"}

        phase_order = [RampPhase.PAPER, RampPhase.PHASE_1, RampPhase.PHASE_2,
                       RampPhase.PHASE_3, RampPhase.PHASE_4, RampPhase.LIVE]
        current_idx = phase_order.index(self.phase)
        prev_phase = phase_order[current_idx - 1]

        log_entry = {
            "from": self.phase.value,
            "to": prev_phase.value,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "allocation_pct": RAMP_ALLOCATION_PCT[prev_phase.value],
        }

        self._state["phase"] = prev_phase.value
        self._state["phase_started"] = datetime.now().isoformat()
        self._state["days_at_phase"] = 0
        self._state["peak_equity"] = None
        self._state["max_drawdown_pct"] = 0.0
        self._state["advancement_log"] = self._state.get("advancement_log", []) + [log_entry]

        self._save_state()
        logger.warning("Ramp rolled back: %s → %s (reason: %s)",
                       log_entry["from"], prev_phase.value, reason)

        return {"success": True, "new_phase": prev_phase.value, **log_entry}

    def update_daily(self, equity: float) -> None:
        """Update daily tracking for drawdown monitoring.

        Args:
            equity: Current account equity.
        """
        self._state["days_at_phase"] = self._state.get("days_at_phase", 0) + 1

        if self._state.get("peak_equity") is None or equity > self._state["peak_equity"]:
            self._state["peak_equity"] = equity

        if self._state["peak_equity"] and self._state["peak_equity"] > 0:
            dd_pct = (self._state["peak_equity"] - equity) / self._state["peak_equity"]
            if dd_pct > self._state.get("max_drawdown_pct", 0.0):
                self._state["max_drawdown_pct"] = dd_pct

                # Auto-rollback if drawdown exceeds phase limit
                max_dd_allowed = RAMP_MAX_DRAWDOWN.get(self.phase.value, 1.0)
                if dd_pct > max_dd_allowed:
                    self.rollback(f"Drawdown {dd_pct:.1%} exceeds phase limit {max_dd_allowed:.1%}")
                    return

        self._save_state()


if __name__ == "__main__":
    # CLI interface for testing
    from src.utils.log_config import configure_logging
    configure_logging()
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "status":
            logger.info(json.dumps(check_alpaca_status(), indent=2))
        elif cmd == "account":
            client = AlpacaClient(paper=True)
            if client.is_ready():
                logger.info(json.dumps(client.get_account(), indent=2))
            else:
                logger.info("Alpaca not configured. Set ALPACA_API_KEY and ALPACA_API_SECRET.")
        elif cmd == "positions":
            client = AlpacaClient(paper=True)
            if client.is_ready():
                positions = client.get_positions()
                logger.info(json.dumps([asdict(p) for p in positions], indent=2))
            else:
                logger.info("Alpaca not configured.")
        elif cmd == "sync":
            manager = PaperTradingManager()
            logger.info(json.dumps(manager.sync_positions(), indent=2))
        elif cmd == "ramp":
            ramp_mgr = LiveTransitionManager()
            logger.info(json.dumps(ramp_mgr.get_status(), indent=2))
        elif cmd == "ramp-advance":
            ramp_mgr = LiveTransitionManager()
            result = ramp_mgr.advance()
            logger.info(json.dumps(result, indent=2))
        elif cmd == "ramp-rollback":
            ramp_mgr = LiveTransitionManager()
            reason = sys.argv[2] if len(sys.argv) > 2 else "manual"
            result = ramp_mgr.rollback(reason)
            logger.info(json.dumps(result, indent=2))
        else:
            logger.info("Unknown command: %s", cmd)
            logger.info("Commands: status, account, positions, sync, ramp, ramp-advance, ramp-rollback")
    else:
        # Default: print status
        logger.info(json.dumps(check_alpaca_status(), indent=2))
