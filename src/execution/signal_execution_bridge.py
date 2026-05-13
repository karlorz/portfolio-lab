#!/usr/bin/env python3
"""
Portfolio-Lab v2.83: Signal-to-Execution Bridge

Wires SignalIntegrator.get_allocation_deltas() to the execution layer
(RebalanceScheduler). Converts signal-based allocation shifts into scheduled
orders with urgency-based execution windows.

Architecture:
    SignalIntegrator.get_allocation_deltas() → SignalExecutionBridge → 
    RebalanceScheduler.schedule_orders() → IntradayCostModel → Execution

Features:
- Automatic signal-to-order conversion
- Urgency mapping based on signal confidence and regime
- Cost-optimized scheduling (11:00-14:00 ET)
- Dry-run mode for paper trading
- Batch order generation for multi-asset rebalancing

Usage:
    from src.execution.signal_execution_bridge import SignalExecutionBridge
    
    bridge = SignalExecutionBridge()
    
    # Generate and schedule orders from signals
    orders = bridge.generate_orders_from_signals(
        current_portfolio={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        dry_run=True
    )

CLI:
    python -m src.execution.signal_execution_bridge check
    python -m src.execution.signal_execution_bridge rebalance --portfolio 46/38/16 --dry-run
    python -m src.execution.signal_execution_bridge status

References:
    - spec: work/2026-05-13-v283-signal-execution-bridge/
"""

import json
import sqlite3
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_DOWN

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals.integrator import SignalIntegrator, CompositeSignal
from src.execution.rebalance_scheduler import (
    RebalanceScheduler, ScheduledOrder, OrderUrgency, ExecutionWindow
)


@dataclass
class AllocationDelta:
    """Calculated allocation change for an asset"""
    symbol: str
    current_weight: float
    target_weight: float
    delta: float
    confidence: float
    urgency: OrderUrgency
    signal_score: float
    estimated_value: float  # Dollar value of change


@dataclass
class BridgeResult:
    """Result of signal-to-execution bridge operation"""
    timestamp: str
    portfolio_value: float
    regime: str
    deltas: List[AllocationDelta]
    orders: List[ScheduledOrder]
    total_estimated_cost_bps: float
    dry_run: bool
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "portfolio_value": self.portfolio_value,
            "regime": self.regime,
            "num_deltas": len(self.deltas),
            "num_orders": len(self.orders),
            "total_cost_bps": self.total_estimated_cost_bps,
            "dry_run": self.dry_run,
            "deltas": [
                {
                    "symbol": d.symbol,
                    "current": f"{d.current_weight:.1%}",
                    "target": f"{d.target_weight:.1%}",
                    "delta": f"{d.delta:+.1%}",
                    "urgency": d.urgency.value,
                    "confidence": f"{d.confidence:.2f}",
                    "value": f"${d.estimated_value:,.0f}"
                }
                for d in self.deltas
            ],
            "orders": [o.to_dict() for o in self.orders]
        }


class SignalExecutionBridge:
    """
    Bridge between SignalIntegrator and RebalanceScheduler
    
    Converts signal-based allocation recommendations into
    executable orders with cost-optimized scheduling.
    """
    
    # Urgency thresholds based on signal characteristics
    URGENCY_THRESHOLDS = {
        "score": {
            OrderUrgency.URGENT: 0.75,    # |score| > 0.75
            OrderUrgency.HIGH: 0.50,      # |score| > 0.50
            OrderUrgency.NORMAL: 0.25,    # |score| > 0.25
            OrderUrgency.LOW: 0.0,        # Everything else
        },
        "confidence": {
            OrderUrgency.URGENT: 0.80,
            OrderUrgency.HIGH: 0.60,
            OrderUrgency.NORMAL: 0.40,
            OrderUrgency.LOW: 0.0,
        }
    }
    
    # Maximum single-trade deviation from base allocation
    MAX_SINGLE_DELTA = 0.10
    
    # Minimum trade value ($)
    MIN_TRADE_VALUE = 1000.0
    
    def __init__(
        self,
        integrator: Optional[SignalIntegrator] = None,
        scheduler: Optional[RebalanceScheduler] = None,
        portfolio_value: float = 100000.0,
        db_path: Optional[Path] = None
    ):
        self.integrator = integrator or SignalIntegrator()
        self.scheduler = scheduler or RebalanceScheduler()
        self.portfolio_value = portfolio_value
        
        if db_path is None:
            db_path = Path("/root/projects/portfolio-lab/data/market.db")
        self.db_path = db_path
        
        self._price_cache: Dict[str, float] = {}
        
    def _get_latest_price(self, symbol: str) -> Optional[float]:
        """Fetch latest price from database"""
        if symbol in self._price_cache:
            return self._price_cache[symbol]
            
        if not self.db_path.exists():
            return None
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT close FROM prices 
            WHERE symbol = ? 
            ORDER BY date DESC 
            LIMIT 1
        """, (symbol,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            price = row[0]
            self._price_cache[symbol] = price
            return price
        return None
    
    def _calculate_urgency(
        self,
        signal_score: float,
        confidence: float,
        regime: str
    ) -> OrderUrgency:
        """
        Determine order urgency based on signal characteristics
        
        High-confidence, high-magnitude signals get URGENT/HIGH priority.
        Low-confidence or neutral signals get NORMAL/LOW priority.
        """
        abs_score = abs(signal_score)
        
        # Regime-based adjustment
        regime_boost = 0.0
        if regime in ["crisis", "high_vol"]:
            regime_boost = 0.15  # Elevated urgency in crisis
        elif regime == "bull" and signal_score < 0:
            # Bearish signal in bull market - be cautious
            regime_boost = -0.10
            
        adjusted_score = abs_score + regime_boost
        
        # Check combined score + confidence
        combined = (adjusted_score + confidence) / 2
        
        # Apply thresholds
        if combined >= 0.70 or (abs_score >= 0.75 and confidence >= 0.70):
            return OrderUrgency.URGENT
        elif combined >= 0.55 or (abs_score >= 0.50 and confidence >= 0.60):
            return OrderUrgency.HIGH
        elif combined >= 0.35 or (abs_score >= 0.25 and confidence >= 0.40):
            return OrderUrgency.NORMAL
        else:
            return OrderUrgency.LOW
    
    def generate_allocation_deltas(
        self,
        current_portfolio: Dict[str, float],
        max_delta: float = 0.10
    ) -> Tuple[List[AllocationDelta], str]:
        """
        Generate allocation deltas from SignalIntegrator
        
        Returns list of AllocationDelta objects and detected regime.
        """
        deltas = []
        detected_regime = "neutral"
        
        for symbol, current_weight in current_portfolio.items():
            # Get composite signal from integrator
            try:
                signal = self.integrator.get_composite_signal(symbol)
                
                # Track overall regime (use first non-neutral)
                if signal.detected_regime != "neutral" and detected_regime == "neutral":
                    detected_regime = signal.detected_regime
                    
            except Exception as e:
                # Fallback to neutral signal
                signal = CompositeSignal(
                    ticker=symbol,
                    timestamp=datetime.now().isoformat(),
                    component_signals=[],
                    composite_score=0.0,
                    composite_confidence=0.0,
                    primary_drivers=[],
                    signal_agreement="neutral",
                    detected_regime="neutral",
                    weights_used={},
                    expected_accuracy=None
                )
            
            # Calculate target weight adjustment
            direction = 1.0 if signal.composite_score > 0 else -1.0 if signal.composite_score < 0 else 0.0
            strength = min(abs(signal.composite_score) * signal.composite_confidence, 1.0)
            
            # Limit maximum shift
            max_shift = min(max_delta, self.MAX_SINGLE_DELTA)
            weight_adjustment = direction * strength * max_shift
            
            # Calculate target weight
            target_weight = current_weight + weight_adjustment
            
            # Clamp to reasonable bounds
            target_weight = max(0.05, min(0.80, target_weight))
            
            # Calculate delta
            delta = target_weight - current_weight
            
            # Skip if below minimum threshold
            if abs(delta) < 0.01:  # < 1% change
                continue
                
            # Calculate urgency
            urgency = self._calculate_urgency(
                signal.composite_score, signal.composite_confidence, signal.detected_regime
            )
            
            # Calculate dollar value
            estimated_value = abs(delta) * self.portfolio_value
            
            deltas.append(AllocationDelta(
                symbol=symbol,
                current_weight=current_weight,
                target_weight=target_weight,
                delta=delta,
                confidence=signal.composite_confidence,
                urgency=urgency,
                signal_score=signal.composite_score,
                estimated_value=estimated_value
            ))
        
        return deltas, detected_regime
    
    def _deltas_to_orders(
        self,
        deltas: List[AllocationDelta]
    ) -> List[ScheduledOrder]:
        """Convert allocation deltas to scheduled orders"""
        orders = []
        
        for delta in deltas:
            # Skip if below minimum trade value
            if delta.estimated_value < self.MIN_TRADE_VALUE:
                continue
                
            # Determine side
            side = "buy" if delta.delta > 0 else "sell"
            
            # Get current price
            price = self._get_latest_price(delta.symbol)
            if not price or price <= 0:
                continue
                
            # Calculate shares (simplified - no lot sizing)
            target_value = abs(delta.delta) * self.portfolio_value
            target_shares = target_value / price
            
            # Create order ID
            order_id = f"{delta.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(orders):03d}"
            
            # Schedule order using correct method signature
            scheduled_order = self.scheduler.schedule_order(
                order_id=order_id,
                symbol=delta.symbol,
                side=side,
                target_shares=target_shares,
                target_value=target_value,
                urgency=delta.urgency.value
            )
            orders.append(scheduled_order)
        
        return orders
    
    def generate_orders_from_signals(
        self,
        current_portfolio: Dict[str, float],
        dry_run: bool = True,
        max_delta: float = 0.10
    ) -> BridgeResult:
        """
        Main entry: Convert signals to scheduled orders
        
        Args:
            current_portfolio: Current weights e.g. {"SPY": 0.46, ...}
            dry_run: If True, simulate without creating real orders
            max_delta: Maximum single-asset deviation from base
        
        Returns:
            BridgeResult with deltas, orders, and cost estimates
        """
        # Generate allocation deltas from signals
        deltas, regime = self.generate_allocation_deltas(
            current_portfolio, max_delta
        )
        
        # Convert to scheduled orders
        orders = self._deltas_to_orders(deltas)
        
        # Calculate total estimated cost
        total_cost_bps = sum(
            o.estimated_cost_bps or 5.0  # Default 5bps if not calculated
            for o in orders
        )
        
        result = BridgeResult(
            timestamp=datetime.now().isoformat(),
            portfolio_value=self.portfolio_value,
            regime=regime,
            deltas=deltas,
            orders=orders,
            total_estimated_cost_bps=total_cost_bps,
            dry_run=dry_run
        )
        
        if not dry_run:
            # Persist orders to database for execution
            self._persist_orders(orders)
        
        return result
    
    def _persist_orders(self, orders: List[ScheduledOrder]):
        """Persist scheduled orders to execution database"""
        # Placeholder for order persistence
        # Would write to execution/orders table
        pass
    
    def check_signal_health(self) -> Dict[str, Any]:
        """Check health of all signal sources"""
        health = {}
        
        try:
            for source_name, source in self.integrator.sources.items():
                try:
                    # Get sample signal
                    sample = source.generate_signal("SPY")
                    health[source_name] = {
                        "status": "healthy",
                        "last_signal": sample.timestamp if hasattr(sample, 'timestamp') else "unknown",
                        "latency_ms": None  # Would measure actual latency
                    }
                except Exception as e:
                    health[source_name] = {
                        "status": "error",
                        "error": str(e)
                    }
        except Exception as e:
            health["integrator"] = {"status": "error", "error": str(e)}
        
        return health
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get complete bridge status"""
        return {
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": self.portfolio_value,
            "signal_sources": list(self.integrator.sources.keys()),
            "signal_health": self.check_signal_health(),
            "scheduler_config": "optimal_11-14_et",
            "last_rebalance": None  # Would track from DB
        }


def main():
    parser = argparse.ArgumentParser(
        description="Signal-to-Execution Bridge v2.83"
    )
    parser.add_argument(
        "command",
        choices=["check", "rebalance", "status"],
        help="Operation: check signals, generate rebalance, or show status"
    )
    parser.add_argument(
        "--portfolio", "-p",
        default="46/38/16",
        help="Portfolio allocation (e.g., 46/38/16 for SPY/GLD/TLT)"
    )
    parser.add_argument(
        "--value", "-v",
        type=float,
        default=100000.0,
        help="Portfolio value in USD (default: $100k)"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        default=True,
        help="Simulate without creating orders (default: True)"
    )
    parser.add_argument(
        "--live", "-l",
        action="store_true",
        help="Execute live (disables dry-run)"
    )
    parser.add_argument(
        "--max-delta",
        type=float,
        default=0.10,
        help="Maximum single-asset deviation (default: 10%)"
    )
    parser.add_argument(
        "--output", "-o",
        help="JSON output file"
    )
    
    args = parser.parse_args()
    
    # Parse portfolio
    weights = [float(w) for w in args.portfolio.split("/")]
    assets = ["SPY", "GLD", "TLT"][:len(weights)]
    portfolio = {assets[i]: weights[i]/100 if weights[i] > 1 else weights[i] 
                for i in range(len(weights))}
    
    # Normalize
    total = sum(portfolio.values())
    portfolio = {k: v/total for k, v in portfolio.items()}
    
    # Initialize bridge
    bridge = SignalExecutionBridge(portfolio_value=args.value)
    
    if args.command == "check":
        print("="*60)
        print("SIGNAL SOURCE HEALTH CHECK")
        print("="*60)
        
        health = bridge.check_signal_health()
        for source, status in health.items():
            icon = "✓" if status.get("status") == "healthy" else "✗"
            print(f"{icon} {source:20s}: {status['status']}")
            if "error" in status:
                print(f"    Error: {status['error']}")
        
        print("="*60)
        
    elif args.command == "rebalance":
        dry_run = not args.live if args.live else args.dry_run
        mode = "DRY RUN" if dry_run else "LIVE EXECUTION"
        
        print("="*60)
        print(f"SIGNAL-BASED REBALANCE ({mode})")
        print("="*60)
        print(f"Portfolio: {portfolio}")
        print(f"Value: ${args.value:,.0f}")
        print(f"Max Delta: {args.max_delta:.0%}")
        print()
        
        result = bridge.generate_orders_from_signals(
            current_portfolio=portfolio,
            dry_run=dry_run,
            max_delta=args.max_delta
        )
        
        # Display results
        print(f"Detected Regime: {result.regime.upper()}")
        print(f"Allocation Deltas: {len(result.deltas)}")
        print()
        
        print("-"*60)
        print("ALLOCATION CHANGES:")
        print("-"*60)
        for d in result.deltas:
            icon = "↗" if d.delta > 0 else "↘" if d.delta < 0 else "→"
            print(f"{icon} {d.symbol:4s}: {d.current_weight:>6.1%} → {d.target_weight:>6.1%} "
                  f"({d.delta:+.1%}) [{d.urgency.value.upper():6s}] "
                  f"${d.estimated_value:>8,.0f}")
        
        if result.orders:
            print()
            print("-"*60)
            print("SCHEDULED ORDERS:")
            print("-"*60)
            for o in result.orders:
                win = o.execution_window.value if o.execution_window else "unknown"
                cost = f"{o.estimated_cost_bps:.1f}bps" if o.estimated_cost_bps else "~5bps"
                print(f"{o.order_id}: {o.side.upper():4s} {o.symbol:4s} "
                      f"shares={o.target_shares:.2f} "
                      f"value=${o.target_value:,.0f} "
                      f"[{o.urgency.value.upper():6s}] "
                      f"window={win} cost={cost}")
        
        print()
        print("-"*60)
        print(f"Total Estimated Cost: {result.total_estimated_cost_bps:.1f} bps")
        print("="*60)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    elif args.command == "status":
        status = bridge.get_current_status()
        
        print("="*60)
        print("SIGNAL EXECUTION BRIDGE STATUS")
        print("="*60)
        print(f"Timestamp: {status['timestamp']}")
        print(f"Portfolio Value: ${status['portfolio_value']:,.0f}")
        print(f"Signal Sources: {len(status['signal_sources'])}")
        for src in status['signal_sources']:
            print(f"  - {src}")
        print(f"Scheduler: {status['scheduler_config']}")
        print("="*60)


if __name__ == "__main__":
    main()
