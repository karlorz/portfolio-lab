#!/usr/bin/env python3
"""
Portfolio-Lab v3.12 Phase 1: 0DTE Yield Enhancement - Position Dataclasses

Data structures for tracking 0DTE option positions and trades.

Usage:
    from src.options.odte_yield_position import (
        ZeroDTEPosition, ZeroDTETrade, ZeroDTETradeType, TradeStatus
    )
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, time
from typing import Optional, Dict, List
from enum import Enum


class ZeroDTETradeType(Enum):
    """Type of 0DTE trade."""
    SHORT_CALL = "short_call"           # Sell naked call (covered by cash)
    SHORT_PUT = "short_put"             # Sell naked put (cash-secured)
    IRON_CONDOR = "iron_condor"         # Sell call and put spreads
    CALL_SPREAD = "call_spread"         # Bear call spread


class TradeStatus(Enum):
    """Position lifecycle status."""
    PENDING = "pending"                 # Order submitted, not filled
    OPEN = "open"                       # Active position
    CLOSED = "closed"                   # Expired or closed normally
    STOPPED = "stopped"                 # Closed by stop/loss
    EXPIRED_ITM = "expired_itm"         # Expired in-the-money (assigned)
    EXPIRED_OTM = "expired_otm"         # Expired out-of-the-money
    ROLLED = "rolled"                   # Closed and reopened


class CloseReason(Enum):
    """Reason for position close."""
    EXPIRATION = "expiration"
    PROFIT_TAKE = "profit_take"
    STOP_LOSS = "stop_loss"
    DELTA_STOP = "delta_stop"
    TIME_EXIT = "time_exit"
    MANUAL = "manual"
    ROLL = "roll"
    EMERGENCY = "emergency"


@dataclass
class Greeks:
    """Option Greeks container."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


@dataclass
class OptionLeg:
    """Single option leg (call or put)."""
    symbol: str                         # Underlying (SPY, SPX, etc.)
    option_symbol: str                  # Full option symbol
    option_type: str                    # 'call' or 'put'
    side: str                           # 'buy' or 'sell'
    quantity: int
    
    # Strike and expiration
    strike: float
    expiration: datetime
    
    # Pricing
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    
    # Greeks
    entry_greeks: Greeks = field(default_factory=Greeks)
    current_greeks: Greeks = field(default_factory=Greeks)
    
    @property
    def is_short(self) -> bool:
        return self.side == 'sell'
    
    @property
    def notional_value(self) -> float:
        """Notional value = strike * quantity * 100"""
        return self.strike * self.quantity * 100
    
    @property
    def premium_received(self) -> float:
        """Premium received (for short) or paid (for long) at entry."""
        mult = 1 if self.is_short else -1
        return self.entry_price * self.quantity * 100 * mult
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L in dollars."""
        if self.is_short:
            # Short: profit when price drops
            return (self.entry_price - self.current_price) * self.quantity * 100
        else:
            # Long: profit when price rises
            return (self.current_price - self.entry_price) * self.quantity * 100
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage of entry premium."""
        if self.entry_price == 0:
            return 0.0
        return self.unrealized_pnl / abs(self.premium_received)


@dataclass
class ZeroDTEPosition:
    """
    Active 0DTE position tracking.
    
    For simple call writing, this is just one short call leg.
    For iron condors, multiple legs are tracked.
    """
    position_id: str
    underlying: str
    trade_type: ZeroDTETradeType
    
    # Entry details
    entry_time: datetime
    entry_spot: float
    entry_vix: float
    
    # Legs
    legs: List[OptionLeg] = field(default_factory=list)
    
    # Status
    status: TradeStatus = TradeStatus.PENDING
    
    # Risk parameters
    stop_loss_price: Optional[float] = None
    profit_take_price: Optional[float] = None
    max_loss: float = 0.0
    
    # Tracking
    total_premium_received: float = 0.0
    commission_paid: float = 0.0
    
    # Close details (populated when closed)
    close_time: Optional[datetime] = None
    close_reason: Optional[CloseReason] = None
    realized_pnl: Optional[float] = None
    
    @property
    def is_active(self) -> bool:
        return self.status in (TradeStatus.OPEN, TradeStatus.PENDING)
    
    @property
    def is_closed(self) -> bool:
        return self.status in (TradeStatus.CLOSED, TradeStatus.STOPPED, 
                              TradeStatus.EXPIRED_ITM, TradeStatus.EXPIRED_OTM, 
                              TradeStatus.ROLLED)
    
    @property
    def net_premium_received(self) -> float:
        """Net premium received across all legs."""
        return sum(leg.premium_received for leg in self.legs)
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L across all legs."""
        return sum(leg.unrealized_pnl for leg in self.legs)
    
    @property
    def portfolio_delta_impact(self) -> float:
        """Delta exposure as fraction of portfolio (simplified)."""
        total_delta = sum(
            leg.current_greeks.delta * leg.quantity * (100 if leg.side == 'sell' else -100)
            for leg in self.legs
        )
        # Assuming portfolio value is tracked separately
        return total_delta  # Caller needs to divide by portfolio value
    
    @property
    def max_profit(self) -> float:
        """Maximum achievable profit (net premium received)."""
        return self.net_premium_received
    
    @property
    def days_to_expiration(self) -> int:
        """Days until expiration."""
        if not self.legs:
            return 0
        now = datetime.now()
        exp = self.legs[0].expiration
        delta = exp - now
        return max(0, delta.days)
    
    @property
    def hours_to_expiration(self) -> float:
        """Hours until expiration."""
        if not self.legs:
            return 0
        now = datetime.now()
        exp = self.legs[0].expiration
        delta = exp - now
        return max(0, delta.total_seconds() / 3600)
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all legs."""
        for leg in self.legs:
            if leg.option_symbol in prices:
                leg.current_price = prices[leg.option_symbol]
    
    def update_greeks(self, greeks: Dict[str, Greeks]):
        """Update Greeks for all legs."""
        for leg in self.legs:
            if leg.option_symbol in greeks:
                leg.current_greeks = greeks[leg.option_symbol]
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "position_id": self.position_id,
            "underlying": self.underlying,
            "trade_type": self.trade_type.value,
            "status": self.status.value,
            "entry_time": self.entry_time.isoformat(),
            "entry_spot": self.entry_spot,
            "entry_vix": self.entry_vix,
            "legs": [
                {
                    "symbol": leg.symbol,
                    "option_symbol": leg.option_symbol,
                    "option_type": leg.option_type,
                    "side": leg.side,
                    "quantity": leg.quantity,
                    "strike": leg.strike,
                    "expiration": leg.expiration.isoformat(),
                    "entry_price": leg.entry_price,
                    "current_price": leg.current_price,
                    "unrealized_pnl": leg.unrealized_pnl,
                    "unrealized_pnl_pct": leg.unrealized_pnl_pct,
                }
                for leg in self.legs
            ],
            "net_premium": self.net_premium_received,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "max_profit": self.max_profit,
            "hours_to_expiration": self.hours_to_expiration,
            "realized_pnl": self.realized_pnl,
            "close_reason": self.close_reason.value if self.close_reason else None,
        }


@dataclass
class ZeroDTETrade:
    """Trade recommendation for 0DTE strategy."""
    trade_id: str
    timestamp: datetime
    underlying: str
    trade_type: ZeroDTETradeType
    
    # Market context
    spot_price: float
    vix: float
    
    # Recommendation details
    legs: List[Dict] = field(default_factory=list)
    total_premium_expected: float = 0.0
    expected_delta: float = 0.0
    
    # Sizing
    recommended_contracts: int = 0
    max_portfolio_allocation_pct: float = 0.0
    
    # Risk metrics
    max_loss_estimate: float = 0.0
    breakeven_price: float = 0.0
    
    # Execution guidance
    urgency: str = "low"              # low (yield enhancement)
    optimal_window_start: time = time(11, 0)
    optimal_window_end: time = time(14, 0)
    
    # Rationale
    rationale: List[str] = field(default_factory=list)
    
    @property
    def is_executable(self) -> bool:
        """Check if trade can be executed now."""
        now = datetime.now().time()
        return self.optimal_window_start <= now <= self.optimal_window_end
    
    def to_order_spec(self) -> Dict:
        """Convert to order specification for broker."""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
            "underlying": self.underlying,
            "trade_type": self.trade_type.value,
            "legs": self.legs,
            "total_premium_expected": self.total_premium_expected,
            "recommended_contracts": self.recommended_contracts,
            "urgency": self.urgency,
            "rationale": self.rationale,
        }


@dataclass
class ZeroDTEPerformance:
    """Performance tracking for 0DTE strategy."""
    # Time period
    start_date: datetime
    end_date: datetime
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # P&L
    total_premium_collected: float = 0.0
    total_losses: float = 0.0
    commissions_paid: float = 0.0
    
    # Assignment tracking
    assignments: int = 0
    assignment_rate: float = 0.0
    
    # Performance metrics
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    win_rate: float = 0.0
    avg_premium_per_trade: float = 0.0
    avg_loss_per_trade: float = 0.0
    profit_factor: float = 0.0
    
    def calculate_metrics(self):
        """Calculate derived metrics."""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
            self.avg_premium_per_trade = self.total_premium_collected / self.total_trades
            self.avg_loss_per_trade = self.total_losses / max(1, self.losing_trades)
            self.gross_pnl = self.total_premium_collected - self.total_losses
            self.net_pnl = self.gross_pnl - self.commissions_paid
            self.profit_factor = self.total_premium_collected / max(0.01, self.total_losses)
            self.assignment_rate = self.assignments / max(1, self.total_trades)


if __name__ == "__main__":
    # Example usage
    from datetime import timedelta
    
    print("=== 0DTE Position Dataclasses ===")
    print()
    
    # Create a sample position
    now = datetime.now()
    exp = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    leg = OptionLeg(
        symbol="SPY",
        option_symbol="SPY251231C00550000",
        option_type="call",
        side="sell",
        quantity=1,
        strike=550.0,
        expiration=exp,
        entry_price=2.50,
        entry_time=now,
        current_price=1.80,
        entry_greeks=Greeks(delta=-0.30, theta=0.15),
        current_greeks=Greeks(delta=-0.25, theta=0.12),
    )
    
    position = ZeroDTEPosition(
        position_id="ODTE_20260101_001",
        underlying="SPY",
        trade_type=ZeroDTETradeType.SHORT_CALL,
        entry_time=now,
        entry_spot=545.0,
        entry_vix=16.5,
        legs=[leg],
        status=TradeStatus.OPEN,
        stop_loss_price=5.00,
        profit_take_price=1.25,
    )
    
    print(f"Position ID: {position.position_id}")
    print(f"Type: {position.trade_type.value}")
    print(f"Status: {position.status.value}")
    print(f"Premium Received: ${position.net_premium_received:.2f}")
    print(f"Unrealized P&L: ${position.total_unrealized_pnl:+.2f}")
    print(f"Hours to Exp: {position.hours_to_expiration:.1f}")
    print()
    
    # Show as dict
    print("JSON export:")
    print(json.dumps(position.to_dict(), indent=2, default=str))
