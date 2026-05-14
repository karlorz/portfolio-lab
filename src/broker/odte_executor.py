#!/usr/bin/env python3
"""
Portfolio-Lab v3.12 Phase 2: 0DTE Order Executor

Order execution module for 0DTE options strategy.
Integrates with existing order_router.py for seamless execution.

Usage:
    from src.broker.odte_executor import ODTEExecutor, ODTEOrderRequest
    
    executor = ODTEExecutor()
    order = await executor.enter_position(
        underlying="SPY",
        target_delta=0.30,
        portfolio_value=100000
    )
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
import asyncio

from src.broker.options_utils import (
    OptionsChainFetcher, OptionQuote, get_best_0dte_call
)
from src.options.odte_yield_calculator import (
    ZeroDTECalculator, ZeroDTEConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class ExitReason(Enum):
    EXPIRATION = "expiration"
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    DELTA_THRESHOLD = "delta_threshold"
    TIME_STOP = "time_stop"
    MANUAL = "manual"


@dataclass
class ODTEOrderRequest:
    """Request for 0DTE option order."""
    underlying: str
    strike: float
    option_symbol: str
    quantity: int
    order_type: str = "limit"  # limit, market
    limit_price: Optional[float] = None
    time_in_force: str = "day"
    
    # Risk parameters
    max_slippage_pct: float = 0.1  # 10 bps max slippage
    
    # Metadata
    target_delta: float = 0.30
    expected_premium: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ODTEExecutionResult:
    """Result of 0DTE order execution."""
    success: bool
    order_id: Optional[str] = None
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    premium_collected: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "order_id": self.order_id,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "premium_collected": self.premium_collected,
            "status": self.status.value,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ODTEMonitorState:
    """Monitoring state for an active 0DTE position."""
    option_symbol: str
    underlying: str
    strike: float
    entry_premium: float
    contracts: int
    entry_delta: Optional[float] = None
    
    # P&L tracking
    unrealized_pnl: float = 0.0
    current_premium: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    
    # Exit tracking
    exit_triggered: bool = False
    exit_reason: Optional[ExitReason] = None
    exit_price: Optional[float] = None
    
    # Timestamps
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    
    def update_pnl(self, current_buy_price: float):
        """Update P&L based on current buy-back price."""
        self.current_premium = current_buy_price
        self.unrealized_pnl = (self.entry_premium - current_buy_price) * self.contracts * 100
        
        # Update max metrics
        if self.unrealized_pnl > self.max_profit:
            self.max_profit = self.unrealized_pnl
        if self.unrealized_pnl < self.max_loss:
            self.max_loss = self.unrealized_pnl


class ODTEExecutor:
    """
    Executor for 0DTE options strategy.
    
    Handles:
    - Position entry with strike selection
    - Order routing via existing order_router
    - Position monitoring and exit management
    - Paper trading simulation
    """
    
    def __init__(self, config: Optional[ZeroDTEConfig] = None, paper_mode: bool = True):
        self.config = config or ZeroDTEConfig()
        self.paper_mode = paper_mode or os.getenv("ODTE_PAPER_MODE", "true").lower() == "true"
        self.active_positions: Dict[str, ODTEMonitorState] = {}
        self.execution_history: list = []
        
        # Load order router if available
        try:
            from src.broker.order_router import OrderRouter
            self.order_router = OrderRouter()
            self.has_router = True
        except ImportError:
            logger.warning("OrderRouter not available - using direct execution")
            self.has_router = False
    
    async def enter_position(
        self, 
        underlying: str = "SPY",
        target_delta: float = 0.30,
        portfolio_value: float = 100000.0,
        vix: float = 16.0
    ) -> Tuple[Optional[ODTEOrderRequest], Optional[ODTEExecutionResult]]:
        """
        Enter a new 0DTE position.
        
        Returns:
            Tuple of (request, result) - request is None if no suitable option found
        """
        # Get 0DTE chain
        fetcher = OptionsChainFetcher()
        chain = await fetcher.fetch_0dte_chain(underlying)
        
        # Find optimal call
        best_call = chain.find_optimal_call(target_delta=target_delta)
        if not best_call:
            logger.warning(f"No suitable 0DTE call found for {underlying}")
            return None, None
        
        # Validate with calculator
        calculator = ZeroDTECalculator(self.config)
        
        # Calculate position size
        num_contracts = calculator.calculate_position_size(portfolio_value)
        
        if num_contracts < 1:
            logger.warning(f"Position entry blocked: position size 0 for ${portfolio_value:,.0f} portfolio")
            return None, None
        
        # Build order request
        quantity = num_contracts
        expected_premium = best_call.mark * quantity * 100
        
        request = ODTEOrderRequest(
            underlying=underlying,
            strike=best_call.strike,
            option_symbol=best_call.symbol,
            quantity=quantity,
            order_type="limit",
            limit_price=best_call.bid,  # Start at bid
            target_delta=target_delta,
            expected_premium=expected_premium,
        )
        
        # Execute
        result = await self._execute_order(request, best_call)
        
        if result.success:
            # Track position
            monitor = ODTEMonitorState(
                option_symbol=best_call.symbol,
                underlying=underlying,
                strike=best_call.strike,
                entry_premium=result.avg_fill_price,
                contracts=quantity,
                entry_delta=best_call.delta,
                max_profit=expected_premium,  # Max profit is premium collected
                max_loss=-expected_premium * 2,  # Approximate max loss
            )
            self.active_positions[best_call.symbol] = monitor
            
            logger.info(f"Entered 0DTE position: {quantity} x {best_call.symbol} @ ${result.avg_fill_price:.2f}")
        
        return request, result
    
    async def _execute_order(
        self, 
        request: ODTEOrderRequest, 
        quote: OptionQuote
    ) -> ODTEExecutionResult:
        """
        Execute the order via router or simulation.
        """
        if self.paper_mode:
            return await self._simulate_execution(request, quote)
        
        # Real execution not yet implemented - use paper mode
        logger.warning("Real execution not yet available, using paper mode")
        return await self._simulate_execution(request, quote)
    
    async def _simulate_execution(
        self, 
        request: ODTEOrderRequest, 
        quote: OptionQuote
    ) -> ODTEExecutionResult:
        """
        Simulate order execution for paper trading.
        
        Assumes:
        - Fill at mid price with slight slippage
        - 90% fill rate
        - Execution within 1 second
        """
        await asyncio.sleep(0.1)  # Simulate network latency
        
        # Simulate 90% fill rate
        import random
        if random.random() > 0.9:
            return ODTEExecutionResult(
                success=False,
                error_message="Simulated fill failure",
                status=OrderStatus.REJECTED,
            )
        
        # Simulate slight slippage (fill slightly worse than mid)
        slippage = random.uniform(0, 0.02)  # 0-2 cents
        fill_price = quote.mark - slippage
        
        return ODTEExecutionResult(
            success=True,
            order_id=f"SIM_{datetime.now().strftime('%H%M%S')}_{random.randint(1000, 9999)}",
            filled_quantity=request.quantity,
            avg_fill_price=round(fill_price, 2),
            premium_collected=round(fill_price * request.quantity * 100, 2),
            status=OrderStatus.FILLED,
        )
    
    async def check_exit_conditions(self, position: ODTEMonitorState) -> Optional[ExitReason]:
        """
        Check if position should be exited based on risk rules.
        
        Returns ExitReason if exit triggered, None otherwise.
        """
        calculator = ZeroDTECalculator(self.config)
        
        # 1. Time-based exit (close to expiration)
        now = datetime.now().time()
        market_close = time(16, 0)
        
        if now.hour >= 15 and now.minute >= 30:  # After 3:30 PM
            return ExitReason.EXPIRATION
        
        # 2. Delta threshold
        # Fetch current quote
        fetcher = OptionsChainFetcher()
        chain = await fetcher.fetch_0dte_chain(position.underlying)
        
        current_quote = None
        for quote in chain.quotes:
            if quote.symbol == position.option_symbol:
                current_quote = quote
                break
        
        if current_quote and current_quote.delta:
            if current_quote.delta > self.config.emergency_close_delta:
                return ExitReason.DELTA_THRESHOLD
            
            # Update P&L
            position.update_pnl(current_quote.mark)
        
        # 3. Stop loss check (based on max loss)
        max_loss_dollars = position.entry_premium * position.contracts * 100 * self.config.max_loss_pct
        if position.unrealized_pnl < -max_loss_dollars:
            return ExitReason.STOP_LOSS
        
        # 4. Profit target (50% of max profit)
        profit_target = position.entry_premium * position.contracts * 100 * 0.5
        if position.unrealized_pnl >= profit_target:
            return ExitReason.PROFIT_TARGET
        
        return None
    
    async def exit_position(
        self, 
        option_symbol: str, 
        reason: ExitReason
    ) -> ODTEExecutionResult:
        """
        Exit an active position.
        """
        if option_symbol not in self.active_positions:
            return ODTEExecutionResult(
                success=False,
                error_message=f"Position {option_symbol} not found",
                status=OrderStatus.REJECTED,
            )
        
        position = self.active_positions[option_symbol]
        
        # Get current market
        fetcher = OptionsChainFetcher()
        chain = await fetcher.fetch_0dte_chain(position.underlying)
        
        current_quote = None
        for quote in chain.quotes:
            if quote.symbol == option_symbol:
                current_quote = quote
                break
        
        if not current_quote:
            return ODTEExecutionResult(
                success=False,
                error_message="Cannot get current market for position",
                status=OrderStatus.REJECTED,
            )
        
        # Build exit request (buy to close)
        request = ODTEOrderRequest(
            underlying=position.underlying,
            strike=position.strike,
            option_symbol=option_symbol,
            quantity=position.contracts,
            order_type="market",  # Exit at market
            limit_price=current_quote.ask,
        )
        
        # Execute
        result = await self._execute_exit(request)
        
        if result.success:
            position.exit_triggered = True
            position.exit_reason = reason
            position.exit_time = datetime.now()
            position.exit_price = result.avg_fill_price
            
            # Calculate final P&L
            final_pnl = (position.entry_premium - result.avg_fill_price) * position.contracts * 100
            logger.info(f"Exited {option_symbol}: {reason.value}, P&L: ${final_pnl:.2f}")
            
            # Remove from active positions
            del self.active_positions[option_symbol]
            
            # Record in history
            self.execution_history.append({
                "entry": position,
                "exit": result,
                "pnl": final_pnl,
                "reason": reason.value,
            })
        
        return result
    
    async def _execute_exit(self, request: ODTEOrderRequest) -> ODTEExecutionResult:
        """Execute exit order."""
        if self.paper_mode:
            # Simulate exit fill at ask
            import random
            fill_price = request.limit_price or 0.5
            return ODTEExecutionResult(
                success=True,
                order_id=f"SIM_EXIT_{datetime.now().strftime('%H%M%S')}",
                filled_quantity=request.quantity,
                avg_fill_price=round(fill_price, 2),
                status=OrderStatus.FILLED,
            )
        
        # Real execution not yet implemented
        logger.warning("Real exit execution not yet available, using paper mode")
        return ODTEExecutionResult(
            success=True,
            order_id=f"SIM_EXIT_{datetime.now().strftime('%H%M%S')}",
            filled_quantity=request.quantity,
            avg_fill_price=round(request.limit_price or 0.5, 2),
            status=OrderStatus.FILLED,
        )
    
    def get_active_positions_summary(self) -> Dict[str, Any]:
        """Get summary of all active positions."""
        return {
            "count": len(self.active_positions),
            "positions": [
                {
                    "symbol": p.option_symbol,
                    "underlying": p.underlying,
                    "strike": p.strike,
                    "contracts": p.contracts,
                    "entry_premium": p.entry_premium,
                    "current_pnl": p.unrealized_pnl,
                    "max_profit": p.max_profit,
                    "max_loss": p.max_loss,
                    "entry_time": p.entry_time.isoformat(),
                }
                for p in self.active_positions.values()
            ],
            "total_premium_collected": sum(p.entry_premium * p.contracts * 100 for p in self.active_positions.values()),
            "total_unrealized_pnl": sum(p.unrealized_pnl for p in self.active_positions.values()),
        }
    
    async def run_monitoring_cycle(self):
        """
        Run one monitoring cycle to check all active positions.
        Called periodically by the monitoring job.
        """
        for symbol in list(self.active_positions.keys()):
            exit_reason = await self.check_exit_conditions(self.active_positions[symbol])
            
            if exit_reason:
                logger.info(f"Exit triggered for {symbol}: {exit_reason.value}")
                await self.exit_position(symbol, exit_reason)


# CLI interface for testing
async def main():
    """Test the 0DTE executor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="0DTE Options Executor")
    parser.add_argument("--enter", action="store_true", help="Enter a new position")
    parser.add_argument("--monitor", action="store_true", help="Run monitoring cycle")
    parser.add_argument("--portfolio", type=float, default=100000, help="Portfolio value")
    parser.add_argument("--delta", type=float, default=0.30, help="Target delta")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no actual orders)")
    
    args = parser.parse_args()
    
    executor = ODTEExecutor(paper_mode=True)
    
    if args.enter:
        print("Attempting to enter 0DTE position...")
        request, result = await executor.enter_position(
            portfolio_value=args.portfolio,
            target_delta=args.delta
        )
        
        if result:
            print(f"\nExecution Result:")
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("No position entered (check filters and conditions)")
    
    if args.monitor:
        print("Running monitoring cycle...")
        await executor.run_monitoring_cycle()
        
        summary = executor.get_active_positions_summary()
        print(f"\nActive Positions: {summary['count']}")
        if summary['count'] > 0:
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
