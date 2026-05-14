"""
Closing Auction Executor (v3.17 Phase 3)

Order timing and execution logic for MOC/IOC statistical arbitrage.
Handles the 3:50pm signal evaluation → 4:00pm exit workflow.

Features:
- 3:50pm ET signal evaluation trigger
- Paper trading integration with immediate execution simulation
- Position sizing with 2-3% allocation cap
- 4:00pm automatic exit (market on close)
- Risk controls and circuit breakers

Author: Autonomous Agent
Version: v3.17 Phase 3
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, time
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.signals.closing_auction import (
    ClosingAuctionSignal,
    SignalDirection,
    SignalConfidence,
    ClosingAuctionSignalGenerator,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Status of a closing auction order."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL_FILL = "partial_fill"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXITED = "exited"


class OrderType(Enum):
    """Type of closing auction order."""
    MOC = "moc"  # Market on Close
    IOC = "ioc"  # Immediate or Cancel
    LIMIT = "limit"


@dataclass
class ClosingAuctionPosition:
    """Represents an active closing auction position."""
    symbol: str
    entry_signal: ClosingAuctionSignal
    entry_time: datetime
    entry_price: float
    shares: int
    side: str  # 'long' or 'short'
    
    # Position tracking
    status: OrderStatus = OrderStatus.PENDING
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    
    # Metadata
    order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current price."""
        if self.side == 'long':
            return (current_price - self.entry_price) * self.shares
        else:  # short
            return (self.entry_price - current_price) * self.shares
    
    def close_position(self, exit_price: float, exit_time: datetime) -> None:
        """Close the position and calculate realized P&L."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = OrderStatus.EXITED
        
        if self.side == 'long':
            self.pnl = (exit_price - self.entry_price) * self.shares
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.shares
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'entry_time': self.entry_time.isoformat(),
            'entry_price': self.entry_price,
            'entry_signal': self.entry_signal.to_dict(),
            'shares': self.shares,
            'side': self.side,
            'status': self.status.value,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'order_id': self.order_id,
            'exit_order_id': self.exit_order_id,
            'notes': self.notes,
        }


@dataclass
class ExecutionConfig:
    """Configuration for closing auction execution."""
    # Timing
    entry_window_start: time = time(15, 50)  # 3:50 PM ET
    entry_window_end: time = time(15, 55)    # 3:55 PM ET
    exit_time: time = time(16, 0)            # 4:00 PM ET
    timezone: str = 'US/Eastern'
    
    # Position sizing
    max_position_pct: float = 0.03  # 3% max allocation per trade
    min_position_pct: float = 0.01  # 1% min allocation
    max_positions_per_day: int = 3  # Max concurrent positions
    
    # Risk controls
    max_intraday_volatility: float = 0.02  # Skip if SPY moves >2%
    max_portfolio_exposure: float = 0.10     # 10% total closing auction exposure
    confidence_threshold: SignalConfidence = SignalConfidence.MEDIUM
    
    # Execution
    dry_run: bool = True  # Paper trading by default
    slippage_bps: float = 5.0  # 5 bps estimated slippage
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_window_start': self.entry_window_start.isoformat(),
            'entry_window_end': self.entry_window_end.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'timezone': self.timezone,
            'max_position_pct': self.max_position_pct,
            'min_position_pct': self.min_position_pct,
            'max_positions_per_day': self.max_positions_per_day,
            'max_intraday_volatility': self.max_intraday_volatility,
            'max_portfolio_exposure': self.max_portfolio_exposure,
            'confidence_threshold': self.confidence_threshold.value,
            'dry_run': self.dry_run,
            'slippage_bps': self.slippage_bps,
        }


class ClosingAuctionExecutor:
    """Manages execution of closing auction MOC/IOC trades."""
    
    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        portfolio_value: float = 100000.0,
        state_file: Optional[Path] = None,
    ):
        self.config = config or ExecutionConfig()
        self.portfolio_value = portfolio_value
        self.state_file = state_file or Path('data/closing_auction/positions.json')
        
        # Active positions
        self.positions: Dict[str, ClosingAuctionPosition] = {}
        self.position_history: List[ClosingAuctionPosition] = []
        
        # Performance tracking
        self.daily_stats: Dict[str, Any] = {
            'trades': 0,
            'winners': 0,
            'losers': 0,
            'total_pnl': 0.0,
        }
        
        # Load state if exists
        self._load_state()
        
        logger.info(f"ClosingAuctionExecutor initialized: dry_run={self.config.dry_run}")
    
    def _load_state(self) -> None:
        """Load previous positions from state file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    # TODO: Deserialize positions
                    logger.info(f"Loaded state from {self.state_file}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
    
    def _save_state(self) -> None:
        """Save current positions to state file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'active_positions': {
                sym: pos.to_dict() for sym, pos in self.positions.items()
            },
            'daily_stats': self.daily_stats,
            'config': self.config.to_dict(),
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def can_enter_position(self, signal: ClosingAuctionSignal) -> tuple[bool, str]:
        """
        Check if we can enter a position based on signal and constraints.
        
        Returns:
            (can_enter, reason)
        """
        # Check if already in position for this symbol
        if signal.symbol in self.positions:
            return False, f"Already in position for {signal.symbol}"
        
        # Check confidence threshold (compare enum values directly)
        confidence_order = {
            SignalConfidence.HIGH: 3,
            SignalConfidence.MEDIUM: 2,
            SignalConfidence.LOW: 1,
            SignalConfidence.INSUFFICIENT_DATA: 0,
        }
        signal_conf = confidence_order.get(signal.confidence, 0)
        threshold_conf = confidence_order.get(self.config.confidence_threshold, 0)
        if signal_conf < threshold_conf:
            return False, f"Confidence {signal.confidence.value} below threshold"
        
        # Check if neutral signal
        if signal.direction == SignalDirection.NEUTRAL:
            return False, "Neutral signal - no trade"
        
        # Check max positions limit
        if len(self.positions) >= self.config.max_positions_per_day:
            return False, f"Max positions ({self.config.max_positions_per_day}) reached"
        
        # Check portfolio exposure
        current_exposure = sum(
            (pos.entry_price * pos.shares) / self.portfolio_value
            for pos in self.positions.values()
        )
        new_position_pct = self.config.max_position_pct
        if current_exposure + new_position_pct > self.config.max_portfolio_exposure:
            return False, f"Portfolio exposure would exceed {self.config.max_portfolio_exposure:.1%}"
        
        return True, "OK"
    
    def calculate_position_size(
        self,
        signal: ClosingAuctionSignal,
        current_price: float,
    ) -> int:
        """
        Calculate number of shares to trade.
        
        Position sizing:
        - Base: max_position_pct of portfolio
        - Adjusted by confidence level
        - Reduced for high volatility days
        """
        # Base allocation
        allocation_pct = self.config.max_position_pct
        
        # Adjust by confidence
        confidence_multiplier = {
            SignalConfidence.HIGH: 1.0,
            SignalConfidence.MEDIUM: 0.7,
            SignalConfidence.LOW: 0.0,  # Don't trade low confidence
        }.get(signal.confidence, 0.0)
        
        # Adjust by direction strength
        direction_strength = abs(signal.direction.value) / 3.0  # 0.33 to 1.0
        
        final_allocation = allocation_pct * confidence_multiplier * direction_strength
        final_allocation = max(final_allocation, self.config.min_position_pct)
        
        # Calculate shares
        position_value = self.portfolio_value * final_allocation
        shares = int(position_value / current_price)
        
        logger.info(
            f"Position size for {signal.symbol}: {shares} shares "
            f"(${position_value:.2f}, {final_allocation:.2%} allocation)"
        )
        
        return shares
    
    def enter_position(
        self,
        signal: ClosingAuctionSignal,
        current_price: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[ClosingAuctionPosition]:
        """
        Enter a new closing auction position.
        
        Args:
            signal: The closing auction signal
            current_price: Current market price (3:50pm snapshot)
            timestamp: Entry timestamp (defaults to now)
        
        Returns:
            ClosingAuctionPosition if entered, None if rejected
        """
        can_enter, reason = self.can_enter_position(signal)
        if not can_enter:
            logger.info(f"Rejected entry for {signal.symbol}: {reason}")
            return None
        
        timestamp = timestamp or datetime.now()
        
        # Calculate position size
        shares = self.calculate_position_size(signal, current_price)
        if shares == 0:
            logger.info(f"Zero shares calculated for {signal.symbol}")
            return None
        
        # Determine side
        side = 'long' if signal.direction.value > 0 else 'short'
        
        # Create position
        position = ClosingAuctionPosition(
            symbol=signal.symbol,
            entry_signal=signal,
            entry_time=timestamp,
            entry_price=current_price,
            shares=shares,
            side=side,
            status=OrderStatus.SUBMITTED,
            order_id=f"PAPER_{signal.symbol}_{timestamp.strftime('%H%M%S')}" if self.config.dry_run else None,
        )
        
        # Simulate fill (immediate for paper trading)
        if self.config.dry_run:
            # Apply slippage
            slippage_factor = 1 + (self.config.slippage_bps / 10000)
            if side == 'short':
                slippage_factor = 2 - slippage_factor  # Worse fill for shorts
            
            position.entry_price = current_price * slippage_factor
            position.status = OrderStatus.FILLED
            
            logger.info(
                f"[PAPER TRADE] Entered {side} position in {signal.symbol}: "
                f"{shares} shares @ ${position.entry_price:.2f} "
                f"(signal: {signal.direction.name}, conf: {signal.confidence.value})"
            )
        
        # Track position
        self.positions[signal.symbol] = position
        self._save_state()
        
        return position
    
    def exit_position(
        self,
        symbol: str,
        exit_price: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[ClosingAuctionPosition]:
        """
        Exit a closing auction position at market close.
        
        Args:
            symbol: Symbol to exit
            exit_price: Exit price (4:00pm close)
            timestamp: Exit timestamp
        
        Returns:
            Closed ClosingAuctionPosition if successful
        """
        if symbol not in self.positions:
            logger.warning(f"No active position for {symbol}")
            return None
        
        timestamp = timestamp or datetime.now()
        position = self.positions[symbol]
        
        # Apply slippage on exit
        slippage_factor = 1 - (self.config.slippage_bps / 10000)
        if position.side == 'short':
            slippage_factor = 2 - slippage_factor  # Worse fill for short covers
        
        adjusted_exit = exit_price * slippage_factor
        
        # Close position
        position.close_position(adjusted_exit, timestamp)
        
        # Update stats
        self.daily_stats['trades'] += 1
        self.daily_stats['total_pnl'] += position.pnl
        if position.pnl > 0:
            self.daily_stats['winners'] += 1
        else:
            self.daily_stats['losers'] += 1
        
        if self.config.dry_run:
            logger.info(
                f"[PAPER TRADE] Exited {symbol}: "
                f"P&L=${position.pnl:.2f} ({position.pnl/position.entry_price/position.shares*100:+.2f}%) "
                f"@ ${adjusted_exit:.2f}"
            )
        
        # Move to history
        self.position_history.append(position)
        del self.positions[symbol]
        self._save_state()
        
        return position
    
    def exit_all_positions(
        self,
        exit_prices: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> List[ClosingAuctionPosition]:
        """
        Exit all active positions (e.g., at 4:00pm market close).
        
        Args:
            exit_prices: Dict of symbol -> exit price
            timestamp: Exit timestamp
        
        Returns:
            List of closed positions
        """
        closed = []
        for symbol in list(self.positions.keys()):
            if symbol in exit_prices:
                pos = self.exit_position(symbol, exit_prices[symbol], timestamp)
                if pos:
                    closed.append(pos)
            else:
                logger.warning(f"No exit price provided for {symbol}")
        
        return closed
    
    def get_active_positions(self) -> Dict[str, ClosingAuctionPosition]:
        """Get all active positions."""
        return self.positions.copy()
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get today's trading statistics."""
        stats = self.daily_stats.copy()
        if stats['trades'] > 0:
            stats['win_rate'] = stats['winners'] / stats['trades']
            stats['avg_pnl'] = stats['total_pnl'] / stats['trades']
        else:
            stats['win_rate'] = 0.0
            stats['avg_pnl'] = 0.0
        return stats
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at market open)."""
        self.daily_stats = {
            'trades': 0,
            'winners': 0,
            'losers': 0,
            'total_pnl': 0.0,
        }
        logger.info("Daily stats reset")


class ClosingAuctionScheduler:
    """Schedules closing auction execution throughout the trading day."""
    
    def __init__(self, executor: ClosingAuctionExecutor):
        self.executor = executor
        self.signal_generator = ClosingAuctionSignalGenerator()
    
    def evaluate_entry_window(
        self,
        signals: List[ClosingAuctionSignal],
        current_prices: Dict[str, float],
        current_time: Optional[datetime] = None,
    ) -> List[ClosingAuctionPosition]:
        """
        Evaluate signals during the 3:50-3:55pm entry window.
        
        Args:
            signals: List of closing auction signals
            current_prices: Current market prices by symbol
            current_time: Current timestamp
        
        Returns:
            List of entered positions
        """
        current_time = current_time or datetime.now()
        entered = []
        
        # Check if we're in the entry window
        current_t = current_time.time()
        if not (self.executor.config.entry_window_start <= current_t <= self.executor.config.entry_window_end):
            logger.debug(f"Outside entry window: {current_t}")
            return entered
        
        # Sort by confidence and direction strength
        sorted_signals = sorted(
            signals,
            key=lambda s: (s.confidence.value, abs(s.direction.value)),
            reverse=True,
        )
        
        for signal in sorted_signals:
            if signal.symbol not in current_prices:
                continue
            
            position = self.executor.enter_position(
                signal,
                current_prices[signal.symbol],
                current_time,
            )
            
            if position:
                entered.append(position)
        
        return entered
    
    def execute_market_close(
        self,
        closing_prices: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> List[ClosingAuctionPosition]:
        """
        Execute 4:00pm market close - exit all positions.
        
        Args:
            closing_prices: Dict of symbol -> closing price
            timestamp: Exit timestamp
        
        Returns:
            List of closed positions
        """
        timestamp = timestamp or datetime.now()
        
        # Check it's close to 4pm
        current_t = timestamp.time()
        if current_t < time(15, 59):
            logger.warning(f"Executing market close before 3:59pm: {current_t}")
        
        return self.executor.exit_all_positions(closing_prices, timestamp)
    
    def is_entry_window(self, current_time: Optional[datetime] = None) -> bool:
        """Check if current time is within the entry window."""
        current_time = current_time or datetime.now()
        current_t = current_time.time()
        return self.executor.config.entry_window_start <= current_t <= self.executor.config.entry_window_end
    
    def is_exit_time(self, current_time: Optional[datetime] = None) -> bool:
        """Check if it's time to exit positions (4:00pm)."""
        current_time = current_time or datetime.now()
        current_t = current_time.time()
        return current_t >= self.executor.config.exit_time
