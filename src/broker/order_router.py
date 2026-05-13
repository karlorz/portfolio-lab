"""
Order router: Converts portfolio-lab signals to Alpaca orders.
"""
import os
import sys
import json
import sqlite3
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from broker.alpaca import (
    AlpacaClient, OrderRequest, OrderSide, OrderType,
    PaperTradingManager
)


@dataclass
class Signal:
    symbol: str
    target_allocation: float  # 0-1
    current_allocation: Optional[float] = None
    signal_type: str = "rebalance"  # rebalance, trend, vix, ml
    confidence: float = 1.0


@dataclass
class OrderPlan:
    symbol: str
    side: str  # BUY or SELL
    qty: float
    order_type: str
    estimated_value: float
    reason: str


class OrderRouter:
    """
    Routes portfolio-lab signals to Alpaca orders.
    Handles allocation targets, position deltas, and execution.
    """
    
    def __init__(
        self,
        signals_file: str = "public/data/signals.json",
        db_path: str = "data/market.db",
        data_dir: str = "data",
        paper: bool = True,
        min_order_value: float = 10.0,  # Minimum $10 order
    ):
        self.signals_file = signals_file
        self.db_path = db_path
        self.data_dir = data_dir
        self.paper = paper
        self.min_order_value = min_order_value
        self.client = AlpacaClient(paper=paper)
        self.manager = PaperTradingManager(data_dir=data_dir)
        self.orders_log = os.path.join(data_dir, "broker_orders.jsonl")
        
    def is_ready(self) -> bool:
        """Check if router can operate."""
        return self.client.is_ready()
    
    def load_signals(self) -> List[Signal]:
        """Load signals from signals.json."""
        if not os.path.exists(self.signals_file):
            return []
        
        try:
            with open(self.signals_file, "r") as f:
                data = json.load(f)
            
            signals = []
            allocations = data.get("target_allocations", {})
            
            for symbol, allocation in allocations.items():
                signals.append(Signal(
                    symbol=symbol,
                    target_allocation=allocation,
                    signal_type="rebalance"
                ))
            
            return signals
        except Exception as e:
            print(f"Error loading signals: {e}")
            return []
    
    def get_current_positions(self) -> Dict[str, Dict[str, float]]:
        """Get current broker positions."""
        if not self.client.is_ready():
            return {}
        
        try:
            positions = self.client.get_positions()
            return {
                p.symbol: {
                    "qty": p.qty,
                    "market_value": p.market_value,
                }
                for p in positions
            }
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return {}
    
    def calculate_orders(
        self,
        signals: List[Signal],
        positions: Dict[str, Dict[str, float]],
        total_value: Optional[float] = None
    ) -> List[OrderPlan]:
        """
        Calculate orders needed to reach target allocations.
        
        Args:
            signals: Target allocations
            positions: Current broker positions
            total_value: Total portfolio value (uses equity if None)
        
        Returns:
            List of order plans
        """
        if total_value is None and self.client.is_ready():
            try:
                account = self.client.get_account()
                total_value = account.get("equity", 0)
            except:
                total_value = sum(p.get("market_value", 0) for p in positions.values())
        
        if not total_value or total_value < 100:
            return []
        
        orders = []
        signal_symbols = {s.symbol for s in signals}
        
        # Process each signal
        for signal in signals:
            target_value = total_value * signal.target_allocation
            current = positions.get(signal.symbol, {})
            current_value = current.get("market_value", 0)
            current_qty = current.get("qty", 0)
            
            delta = target_value - current_value
            
            # Skip if below minimum order threshold
            if abs(delta) < self.min_order_value:
                continue
            
            # Estimate quantity (will use current price or $100 placeholder)
            if current_value > 0 and current_qty > 0:
                price = current_value / current_qty
            else:
                # Would need to fetch current price
                price = 100.0  # Placeholder
            
            qty = abs(delta) / price
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            
            orders.append(OrderPlan(
                symbol=signal.symbol,
                side=side.value.upper(),
                qty=round(qty, 4),
                order_type="MARKET",
                estimated_value=abs(delta),
                reason=f"Rebalance to {signal.target_allocation*100:.1f}% (current: ${current_value:.2f}, target: ${target_value:.2f})"
            ))
        
        # Handle sells for positions not in signals (full liquidation)
        for symbol, pos in positions.items():
            if symbol not in signal_symbols:
                orders.append(OrderPlan(
                    symbol=symbol,
                    side="SELL",
                    qty=pos.get("qty", 0),
                    order_type="MARKET",
                    estimated_value=pos.get("market_value", 0),
                    reason="Liquidate - not in target allocations"
                ))
        
        return orders
    
    def execute_orders(
        self,
        orders: List[OrderPlan],
        dry_run: bool = True,
        kill_switch_check: bool = True
    ) -> Dict[str, Any]:
        """
        Execute order plans via Alpaca API.
        
        Args:
            orders: List of order plans
            dry_run: If True, only log without submitting
            kill_switch_check: If True, check kill switch before execution
        
        Returns:
            Execution report
        """
        if not self.is_ready():
            return {
                "status": "not_configured",
                "message": "Alpaca API not configured",
                "timestamp": datetime.now().isoformat(),
            }
        
        # Check kill switch if enabled
        if kill_switch_check and not dry_run:
            kill_switch_path = os.path.join(self.data_dir, "kill_switch.json")
            if os.path.exists(kill_switch_path):
                try:
                    with open(kill_switch_path, "r") as f:
                        ks = json.load(f)
                    if ks.get("enabled", False):
                        return {
                            "status": "blocked",
                            "message": "Kill switch is enabled - execution blocked",
                            "timestamp": datetime.now().isoformat(),
                        }
                except:
                    pass
        
        executed = []
        failed = []
        
        for plan in orders:
            order_req = OrderRequest(
                symbol=plan.symbol,
                qty=plan.qty,
                side=OrderSide.BUY if plan.side == "BUY" else OrderSide.SELL,
                order_type=OrderType.MARKET
            )
            
            order_dict = {
                "symbol": plan.symbol,
                "side": plan.side,
                "qty": plan.qty,
                "order_type": plan.order_type,
                "estimated_value": plan.estimated_value,
                "reason": plan.reason,
                "timestamp": datetime.now().isoformat(),
                "paper": self.paper,
                "dry_run": dry_run,
            }
            
            if dry_run:
                order_dict["status"] = "dry_run"
                executed.append(order_dict)
            else:
                # Rate limiting: respect 200 req/min (300ms between orders)
                time.sleep(0.3)

                # Retry with exponential backoff (3 attempts)
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        result = self.client.submit_order(order_req)
                        order_dict["status"] = "submitted"
                        order_dict["order_id"] = result.id
                        order_dict["broker_status"] = result.status
                        order_dict["attempts"] = attempt + 1
                        executed.append(order_dict)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait = 2 ** attempt  # 1s, 2s backoff
                            time.sleep(wait)
                            continue
                        order_dict["status"] = "failed"
                        order_dict["error"] = str(e)
                        order_dict["attempts"] = max_retries
                        failed.append(order_dict)
        
        # Log all orders
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.orders_log, "a") as f:
            for order in executed + failed:
                f.write(json.dumps(order) + "\n")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "dry_run" if dry_run else "completed",
            "paper": self.paper,
            "orders_planned": len(orders),
            "orders_executed": len(executed),
            "orders_failed": len(failed),
            "total_estimated_value": sum(o.estimated_value for o in orders),
            "executed": executed,
            "failed": failed,
        }
    
    def rebalance(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Full rebalance workflow: load signals, calculate orders, execute.
        
        This is the main entry point for automated rebalancing.
        """
        if not self.is_ready():
            return {
                "status": "not_configured",
                "message": "Alpaca API not configured",
            }
        
        # Load signals
        signals = self.load_signals()
        if not signals:
            return {
                "status": "no_signals",
                "message": "No signals found in signals.json",
            }
        
        # Get current positions
        positions = self.get_current_positions()
        
        # Calculate orders
        orders = self.calculate_orders(signals, positions)
        
        if not orders:
            return {
                "status": "no_action",
                "message": "No orders needed - portfolio already aligned",
                "signals_count": len(signals),
                "positions_count": len(positions),
            }
        
        # Execute
        return self.execute_orders(orders, dry_run=dry_run)


def main():
    """CLI interface for order router."""
    import sys
    
    router = OrderRouter()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "status":
            print(json.dumps({
                "ready": router.is_ready(),
                "paper": router.paper,
            }, indent=2))
            
        elif cmd == "signals":
            signals = router.load_signals()
            print(json.dumps([{
                "symbol": s.symbol,
                "target": s.target_allocation,
            } for s in signals], indent=2))
            
        elif cmd == "positions":
            positions = router.get_current_positions()
            print(json.dumps(positions, indent=2))
            
        elif cmd == "plan":
            signals = router.load_signals()
            positions = router.get_current_positions()
            orders = router.calculate_orders(signals, positions)
            print(json.dumps([{
                "symbol": o.symbol,
                "side": o.side,
                "qty": o.qty,
                "value": o.estimated_value,
                "reason": o.reason,
            } for o in orders], indent=2))
            
        elif cmd == "rebalance":
            dry_run = "--live" not in sys.argv
            result = router.rebalance(dry_run=dry_run)
            print(json.dumps(result, indent=2))
            
        else:
            print(f"Unknown command: {cmd}")
            print("Commands: status, signals, positions, plan, rebalance [--live]")
    else:
        # Default: show plan
        signals = router.load_signals()
        positions = router.get_current_positions()
        orders = router.calculate_orders(signals, positions)
        print(f"Signals: {len(signals)}")
        print(f"Current positions: {len(positions)}")
        print(f"Orders needed: {len(orders)}")
        for o in orders:
            print(f"  {o.side} {o.qty} {o.symbol} (~${o.estimated_value:.2f})")


if __name__ == "__main__":
    main()
