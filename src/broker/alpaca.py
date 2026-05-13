"""
Alpaca broker API client for paper and live trading.
Supports fractional shares, paper trading without KYC, and WebSocket streaming.
"""
import os
import json
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Alpaca SDK - optional dependency
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce, OrderClass
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

    def _fetch_price(self, symbol: str, db_path: str = "data/market.db") -> float:
        """Fetch latest price from market.db. Returns 0 if unavailable."""
        try:
            if not os.path.exists(db_path):
                return 0.0
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT close FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT 1",
                (symbol,)
            )
            row = cursor.fetchone()
            conn.close()
            return float(row[0]) if row else 0.0
        except Exception:
            return 0.0
    
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
    
    def submit_order(self, order: OrderRequest) -> Order:
        """Submit a new order."""
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
        
        result = client.submit_order(alpaca_order)
        return Order.from_alpaca(result)
    
    def get_orders(self, status: Optional[str] = None, limit: int = 100) -> List[Order]:
        """Get list of orders."""
        client = self._get_client()
        
        # Get orders from Alpaca
        orders = client.get_orders(limit=limit)
        return [Order.from_alpaca(o) for o in orders]
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        client = self._get_client()
        try:
            client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            print(f"Error cancelling order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        client = self._get_client()
        try:
            result = client.cancel_orders()
            return len(result) if result else 0
        except Exception as e:
            print(f"Error cancelling all orders: {e}")
            return 0
    
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        client = self._get_client()
        positions = client.get_all_positions()
        return [Position.from_alpaca(p) for p in positions]
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get specific position."""
        client = self._get_client()
        try:
            position = client.get_open_position(symbol)
            return Position.from_alpaca(position)
        except Exception:
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
        except Exception as e:
            raise RuntimeError(f"Failed to close position {symbol}: {e}")
    
    def close_all_positions(self) -> List[Order]:
        """Close all positions."""
        client = self._get_client()
        try:
            results = client.close_all_positions(cancel_orders=True)
            return [Order.from_alpaca(o) for o in results]
        except Exception as e:
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
    
    def __init__(self, data_dir: str = "data"):
        self.client = AlpacaClient(paper=True)
        self.data_dir = data_dir
        self.orders_file = os.path.join(data_dir, "broker_orders.jsonl")
        
    def is_ready(self) -> bool:
        """Check if paper trading can be activated."""
        return self.client.is_available() and self.client.is_configured()
    
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
        except Exception as e:
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
        
        try:
            account = self.client.get_account()
            positions = self.client.get_positions()
            
            if total_value is None:
                total_value = account["equity"]
            
            current_positions = {p.symbol: p for p in positions}
            
            orders_to_submit = []
            orders_submitted = []
            
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
                else:
                    # New position
                    if target_value < 10:
                        continue
                    estimated_price = self._fetch_price(symbol)
                    if estimated_price <= 0:
                        continue  # Skip if no price available
                    qty = target_value / estimated_price
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
                    except Exception as e:
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
            }
            
            # Log to file
            with open(self.orders_file, "a") as f:
                f.write(json.dumps({
                    "type": "rebalance",
                    "timestamp": result["timestamp"],
                    "dry_run": dry_run,
                    "orders": result["orders_planned"],
                }) + "\n")
            
            return result
            
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Convenience functions for CLI usage
def check_alpaca_status() -> Dict[str, Any]:
    """Quick status check for CLI/health monitoring."""
    client = AlpacaClient(paper=True)
    
    status = {
        "sdk_available": ALPACA_AVAILABLE,
        "configured": client.is_configured(),
        "paper": True,
    }
    
    if ALPACA_AVAILABLE and client.is_configured():
        try:
            account = client.get_account()
            status["account_status"] = account.get("status")
            status["equity"] = account.get("equity")
            status["cash"] = account.get("cash")
            status["connected"] = True
        except Exception as e:
            status["connected"] = False
            status["error"] = str(e)
    else:
        status["connected"] = False
        if not ALPACA_AVAILABLE:
            status["error"] = "alpaca-py SDK not installed"
        elif not client.is_configured():
            status["error"] = "ALPACA_API_KEY and ALPACA_API_SECRET not set"
    
    return status


if __name__ == "__main__":
    # CLI interface for testing
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "status":
            print(json.dumps(check_alpaca_status(), indent=2))
        elif cmd == "account":
            client = AlpacaClient(paper=True)
            if client.is_ready():
                print(json.dumps(client.get_account(), indent=2))
            else:
                print("Alpaca not configured. Set ALPACA_API_KEY and ALPACA_API_SECRET.")
        elif cmd == "positions":
            client = AlpacaClient(paper=True)
            if client.is_ready():
                positions = client.get_positions()
                print(json.dumps([asdict(p) for p in positions], indent=2))
            else:
                print("Alpaca not configured.")
        elif cmd == "sync":
            manager = PaperTradingManager()
            print(json.dumps(manager.sync_positions(), indent=2))
        else:
            print(f"Unknown command: {cmd}")
            print("Commands: status, account, positions, sync")
    else:
        # Default: print status
        print(json.dumps(check_alpaca_status(), indent=2))
