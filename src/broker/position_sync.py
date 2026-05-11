"""
Position synchronization between Alpaca broker and local portfolio-lab state.
"""
import os
import sys
import json
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from broker.alpaca import AlpacaClient
from broker.alpaca import Position as AlpacaPosition


@dataclass
class PositionDrift:
    symbol: str
    local_qty: float
    broker_qty: float
    qty_delta: float
    local_value: float
    broker_value: float
    value_delta: float
    drift_pct: float


class PositionSync:
    """
    Synchronizes positions between Alpaca broker and local SQLite database.
    Detects drift and logs reconciliation history.
    """
    
    def __init__(
        self, 
        db_path: str = "data/market.db",
        data_dir: str = "data",
        paper: bool = True
    ):
        self.db_path = db_path
        self.data_dir = data_dir
        self.sync_log_path = os.path.join(data_dir, "position_sync.jsonl")
        self.client = AlpacaClient(paper=paper)
        
    def is_ready(self) -> bool:
        """Check if sync can be performed."""
        return self.client.is_ready()
    
    def get_local_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get positions from local SQLite database."""
        if not os.path.exists(self.db_path):
            return {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if positions table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='positions'
        """)
        if not cursor.fetchone():
            conn.close()
            return {}
        
        cursor.execute("""
            SELECT symbol, qty, avg_price, current_price, market_value, updated_at
            FROM positions
            WHERE qty != 0
        """)
        
        positions = {}
        for row in cursor.fetchall():
            symbol, qty, avg_price, current_price, market_value, updated_at = row
            positions[symbol] = {
                "symbol": symbol,
                "qty": float(qty) if qty else 0.0,
                "avg_price": float(avg_price) if avg_price else 0.0,
                "current_price": float(current_price) if current_price else 0.0,
                "market_value": float(market_value) if market_value else 0.0,
                "updated_at": updated_at,
            }
        
        conn.close()
        return positions
    
    def get_broker_positions(self) -> Dict[str, AlpacaPosition]:
        """Get positions from Alpaca broker."""
        if not self.client.is_ready():
            return {}
        
        try:
            positions = self.client.get_positions()
            return {p.symbol: p for p in positions}
        except Exception as e:
            print(f"Error fetching broker positions: {e}")
            return {}
    
    def calculate_drift(
        self,
        local_positions: Dict[str, Dict[str, Any]],
        broker_positions: Dict[str, AlpacaPosition]
    ) -> List[PositionDrift]:
        """Calculate position drift between local and broker."""
        drift_items = []
        all_symbols = set(local_positions.keys()) | set(broker_positions.keys())
        
        for symbol in all_symbols:
            local = local_positions.get(symbol, {})
            broker = broker_positions.get(symbol)
            
            local_qty = local.get("qty", 0.0)
            broker_qty = broker.qty if broker else 0.0
            
            local_value = local.get("market_value", 0.0)
            broker_value = broker.market_value if broker else 0.0
            
            qty_delta = broker_qty - local_qty
            value_delta = broker_value - local_value
            
            # Calculate drift percentage
            if local_value > 0:
                drift_pct = value_delta / local_value
            elif broker_value > 0:
                drift_pct = 1.0  # New position
            else:
                drift_pct = 0.0
            
            # Only record if there's actual drift
            if abs(qty_delta) > 0.001 or abs(value_delta) > 1.0:
                drift_items.append(PositionDrift(
                    symbol=symbol,
                    local_qty=local_qty,
                    broker_qty=broker_qty,
                    qty_delta=qty_delta,
                    local_value=local_value,
                    broker_value=broker_value,
                    value_delta=value_delta,
                    drift_pct=drift_pct,
                ))
        
        return drift_items
    
    def sync(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Perform full position synchronization.
        
        Returns sync report with drift analysis.
        """
        if not self.is_ready():
            return {
                "status": "not_configured",
                "message": "Alpaca API not configured",
                "timestamp": datetime.now().isoformat(),
            }
        
        try:
            # Get positions from both sources
            local = self.get_local_positions()
            broker = self.get_broker_positions()
            
            # Calculate drift
            drift = self.calculate_drift(local, broker)
            
            # Get account summary
            account = self.client.get_account()
            
            # Build sync report
            report = {
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "paper": self.client.paper,
                "account": {
                    "equity": account.get("equity"),
                    "cash": account.get("cash"),
                    "buying_power": account.get("buying_power"),
                },
                "local_positions": {
                    "count": len(local),
                    "total_value": sum(p.get("market_value", 0) for p in local.values()),
                },
                "broker_positions": {
                    "count": len(broker),
                    "total_value": sum(p.market_value for p in broker.values()),
                },
                "drift": {
                    "count": len(drift),
                    "total_value_delta": sum(d.value_delta for d in drift),
                    "max_drift_symbol": max(drift, key=lambda x: abs(x.drift_pct)).symbol if drift else None,
                    "max_drift_pct": max(abs(d.drift_pct) for d in drift) if drift else 0.0,
                    "items": [
                        {
                            "symbol": d.symbol,
                            "local_qty": d.local_qty,
                            "broker_qty": d.broker_qty,
                            "qty_delta": d.qty_delta,
                            "local_value": round(d.local_value, 2),
                            "broker_value": round(d.broker_value, 2),
                            "value_delta": round(d.value_delta, 2),
                            "drift_pct": round(d.drift_pct * 100, 2),
                        }
                        for d in drift
                    ],
                },
            }
            
            # Log to file
            if not dry_run:
                os.makedirs(self.data_dir, exist_ok=True)
                with open(self.sync_log_path, "a") as f:
                    f.write(json.dumps(report) + "\n")
            
            return report
            
        except Exception as e:
            error_report = {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "message": str(e),
                "paper": self.client.paper,
            }
            return error_report
    
    def reconcile_to_broker(self) -> Dict[str, Any]:
        """
        Update local positions to match broker (broker is source of truth).
        
        This should be called after confirming broker positions are correct.
        """
        if not self.is_ready():
            return {"status": "not_configured"}
        
        try:
            broker_positions = self.get_broker_positions()
            
            # Ensure positions table exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    qty REAL DEFAULT 0,
                    avg_price REAL DEFAULT 0,
                    current_price REAL DEFAULT 0,
                    market_value REAL DEFAULT 0,
                    updated_at TEXT
                )
            """)
            
            # Update local positions
            timestamp = datetime.now().isoformat()
            for symbol, pos in broker_positions.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO positions 
                    (symbol, qty, avg_price, current_price, market_value, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    pos.qty,
                    pos.avg_entry_price,
                    pos.current_price,
                    pos.market_value,
                    timestamp,
                ))
            
            # Remove positions not in broker
            broker_symbols = set(broker_positions.keys())
            cursor.execute("SELECT symbol FROM positions")
            local_symbols = {row[0] for row in cursor.fetchall()}
            
            for symbol in local_symbols - broker_symbols:
                cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            
            conn.commit()
            conn.close()
            
            return {
                "status": "success",
                "timestamp": timestamp,
                "positions_updated": len(broker_positions),
                "positions_removed": len(local_symbols - broker_symbols),
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
            }


def main():
    """CLI interface for position sync."""
    import sys
    
    sync = PositionSync()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "status":
            print(json.dumps({
                "ready": sync.is_ready(),
                "paper": True,
            }, indent=2))
            
        elif cmd == "sync":
            dry_run = "--dry-run" in sys.argv
            result = sync.sync(dry_run=dry_run)
            print(json.dumps(result, indent=2))
            
        elif cmd == "reconcile":
            result = sync.reconcile_to_broker()
            print(json.dumps(result, indent=2))
            
        elif cmd == "drift":
            result = sync.sync(dry_run=True)
            if result.get("status") == "success":
                drift_items = result.get("drift", {}).get("items", [])
                if drift_items:
                    print(f"Found {len(drift_items)} position drifts:")
                    for d in drift_items:
                        print(f"  {d['symbol']}: {d['qty_delta']:+.2f} shares, ${d['value_delta']:+.2f} ({d['drift_pct']:+.2f}%)")
                else:
                    print("No position drift detected")
            else:
                print(f"Error: {result.get('message')}")
        else:
            print(f"Unknown command: {cmd}")
            print("Commands: status, sync [--dry-run], reconcile, drift")
    else:
        # Default: run sync
        result = sync.sync()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
