"""
VIX Insurance Position Manager - Phase 3 Implementation
Tracks open positions, calculates P&L, and manages rolls/exits.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Status of insurance position."""
    OPEN = "open"
    CLOSED_PROFIT = "closed_profit"
    CLOSED_EXPIRE = "closed_expired"
    CLOSED_STOP = "closed_stop"
    ROLL_PENDING = "roll_pending"


@dataclass
class VIXInsurancePosition:
    """Record of a VIX call position."""
    id: Optional[int]
    status: str
    
    # Entry
    entry_date: str
    entry_vix_spot: float
    strike: float
    expiration_date: str
    contracts: int
    premium_paid_per_contract: float
    total_cost: float
    delta_at_entry: float
    days_to_expiration_at_entry: int
    
    # Current / Exit
    current_mark_price: Optional[float]
    current_value: Optional[float]
    unrealized_pnl: Optional[float]
    unrealized_pnl_percent: Optional[float]
    
    exit_date: Optional[str]
    exit_vix_spot: Optional[float]
    exit_price: Optional[float]
    realized_pnl: Optional[float]
    realized_pnl_percent: Optional[float]
    exit_reason: Optional[str]
    
    # Tracking
    days_held: int
    roll_count: int
    budget_impact: float
    
    def to_dict(self) -> dict:
        return asdict(self)


class VIXPositionManager:
    """
    VIX Call Spread Insurance Position Manager
    
    Tracks open positions, calculates P&L, schedules rolls,
    and manages insurance budget across positions.
    """
    
    DB_PATH = Path("/root/projects/portfolio-lab/data/vix_options.db")
    POSITIONS_PATH = Path("/root/projects/portfolio-lab/data/positions/vix_insurance.json")
    
    # Roll parameters
    ROLL_DAYS_BEFORE_EXPIRY = 5
    
    def __init__(self, annual_budget: float = 1000):
        self.annual_budget = annual_budget
        self._init_db()
        self.POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    def _init_db(self):
        """Initialize positions table if not exists."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vix_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL DEFAULT 'open',
                
                entry_date TEXT NOT NULL,
                entry_vix_spot REAL NOT NULL,
                strike REAL NOT NULL,
                expiration_date TEXT NOT NULL,
                contracts INTEGER NOT NULL,
                premium_paid_per_contract REAL NOT NULL,
                total_cost REAL NOT NULL,
                delta_at_entry REAL,
                days_to_expiration_at_entry INTEGER,
                
                current_mark_price REAL,
                current_value REAL,
                unrealized_pnl REAL,
                unrealized_pnl_percent REAL,
                
                exit_date TEXT,
                exit_vix_spot REAL,
                exit_price REAL,
                realized_pnl REAL,
                realized_pnl_percent REAL,
                exit_reason TEXT,
                
                days_held INTEGER DEFAULT 0,
                roll_count INTEGER DEFAULT 0,
                budget_impact REAL NOT NULL,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Position history log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS position_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                event_date TEXT NOT NULL,
                vix_spot REAL,
                mark_price REAL,
                pnl_realized REAL,
                notes TEXT,
                FOREIGN KEY (position_id) REFERENCES vix_positions(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def open_position(self, signal: Dict) -> Optional[int]:
        """
        Record a new position opening.
        
        Args:
            signal: Output from VIXInsuranceSignalGenerator
            
        Returns:
            position_id if successful
        """
        if signal.get('allocation_dollars', 0) <= 0:
            logger.warning("No allocation for new position")
            return None
        
        premium = signal.get('premium_cost', 0)
        if premium <= 0:
            logger.error("Invalid premium cost")
            return None
        
        # Calculate contracts
        allocation = signal['allocation_dollars']
        contracts = int(allocation / premium)
        
        if contracts < 1:
            logger.warning(f"Allocation ${allocation:,.0f} insufficient for premium ${premium:,.0f}")
            return None
        
        total_cost = contracts * premium
        
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO vix_positions 
            (status, entry_date, entry_vix_spot, strike, expiration_date,
             contracts, premium_paid_per_contract, total_cost, delta_at_entry,
             days_to_expiration_at_entry, budget_impact)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            PositionStatus.OPEN.value,
            signal['timestamp'],
            signal['vix_spot'],
            signal['selected_strike'],
            signal['selected_expiration'],
            contracts,
            premium,
            total_cost,
            signal.get('delta'),
            signal.get('days_to_expiration'),
            total_cost
        ))
        
        position_id = cursor.lastrowid
        
        # Log the event
        cursor.execute("""
            INSERT INTO position_history 
            (position_id, event_type, event_date, vix_spot, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (
            position_id,
            'OPEN',
            signal['timestamp'],
            signal['vix_spot'],
            f"Opened {contracts} contracts at strike {signal['selected_strike']}"
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Opened position {position_id}: {contracts} contracts @ ${signal['selected_strike']:.1f}")
        return position_id
    
    def get_open_positions(self) -> List[Dict]:
        """Get all currently open positions."""
        conn = sqlite3.connect(self.DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM vix_positions 
            WHERE status = 'open'
            ORDER BY entry_date DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def mark_to_market(self, position_id: int, current_vix: float, 
                      option_chain: List[Dict]) -> Dict:
        """
        Calculate current position value based on option chain.
        
        Args:
            position_id: Position to mark
            current_vix: Current VIX spot price
            option_chain: Current options chain with pricing
            
        Returns:
            Updated position data
        """
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM vix_positions WHERE id = ?", (position_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return {}
        
        position = {
            'id': row[0],
            'strike': row[4],
            'expiration_date': row[5],
            'contracts': row[6],
            'premium_paid': row[7],
            'total_cost': row[8],
            'days_held': row[21]
        }
        
        # Find matching option in chain
        current_price = None
        for opt in option_chain:
            if (abs(opt['strike'] - position['strike']) < 0.5 and 
                opt['expiration_date'] == position['expiration_date']):
                current_price = opt.get('mid_price') or ((opt.get('bid', 0) + opt.get('ask', 0)) / 2)
                break
        
        if not current_price:
            # Estimate based on intrinsic value + time value
            intrinsic = max(0, current_vix - position['strike'])
            # Simple time decay estimate
            days_to_exp = self._days_to_expiration(position['expiration_date'])
            if days_to_exp > 0:
                time_value = max(0, position['premium_paid'] * (days_to_exp / 60)) * 0.5
            else:
                time_value = 0
            current_price = intrinsic + time_value
        
        current_value = current_price * position['contracts'] * 100
        unrealized_pnl = current_value - position['total_cost']
        unrealized_pnl_pct = (unrealized_pnl / position['total_cost']) * 100 if position['total_cost'] > 0 else 0
        
        # Update position
        cursor.execute("""
            UPDATE vix_positions 
            SET current_mark_price = ?,
                current_value = ?,
                unrealized_pnl = ?,
                unrealized_pnl_percent = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (current_price, current_value, unrealized_pnl, unrealized_pnl_pct, position_id))
        
        conn.commit()
        conn.close()
        
        return {
            'position_id': position_id,
            'mark_price': current_price,
            'current_value': current_value,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_percent': unrealized_pnl_pct,
            'days_held': position['days_held']
        }
    
    def check_roll_needed(self, position_id: int) -> Tuple[bool, str]:
        """
        Check if position needs to be rolled.
        
        Returns:
            (needs_roll, reason)
        """
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT expiration_date, days_to_expiration_at_entry, days_held
            FROM vix_positions 
            WHERE id = ? AND status = 'open'
        """, (position_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return False, "No open position"
        
        exp_date = datetime.strptime(row[0], '%Y-%m-%d').date()
        days_to_exp = (exp_date - datetime.now().date()).days
        
        if days_to_exp <= self.ROLL_DAYS_BEFORE_EXPIRY:
            return True, f"Expiration in {days_to_exp} days (threshold: {self.ROLL_DAYS_BEFORE_EXPIRY})"
        
        return False, f"{days_to_exp} days to expiration"
    
    def close_position(self, position_id: int, exit_price: float, 
                      current_vix: float, reason: str) -> Dict:
        """
        Close a position and record P&L.
        
        Args:
            position_id: Position to close
            exit_price: Price per contract received
            current_vix: Current VIX spot
            reason: Exit reason (profit_take, expire, stop, roll)
            
        Returns:
            Close summary
        """
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT contracts, total_cost, entry_date, entry_vix_spot
            FROM vix_positions 
            WHERE id = ? AND status = 'open'
        """, (position_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return {'error': 'Position not found or already closed'}
        
        contracts, total_cost, entry_date, entry_vix = row
        
        # Calculate P&L
        exit_value = exit_price * contracts * 100
        realized_pnl = exit_value - total_cost
        realized_pnl_pct = (realized_pnl / total_cost) * 100 if total_cost > 0 else 0
        
        days_held = (datetime.now().date() - datetime.strptime(entry_date[:10], '%Y-%m-%d').date()).days
        
        # Determine status
        if realized_pnl > 0:
            status = PositionStatus.CLOSED_PROFIT.value
        elif reason == 'expire':
            status = PositionStatus.CLOSED_EXPIRE.value
        else:
            status = PositionStatus.CLOSED_STOP.value
        
        # Update position
        cursor.execute("""
            UPDATE vix_positions 
            SET status = ?,
                exit_date = ?,
                exit_vix_spot = ?,
                exit_price = ?,
                realized_pnl = ?,
                realized_pnl_percent = ?,
                exit_reason = ?,
                days_held = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (status, datetime.now().isoformat(), current_vix, exit_price,
              realized_pnl, realized_pnl_pct, reason, days_held, position_id))
        
        # Log event
        cursor.execute("""
            INSERT INTO position_history 
            (position_id, event_type, event_date, vix_spot, mark_price, pnl_realized, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (position_id, 'CLOSE', datetime.now().isoformat(), current_vix,
              exit_price, realized_pnl, f"Closed: {reason}"))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Closed position {position_id}: P&L ${realized_pnl:,.0f} ({realized_pnl_pct:+.1f}%) - {reason}")
        
        return {
            'position_id': position_id,
            'exit_price': exit_price,
            'exit_value': exit_value,
            'realized_pnl': realized_pnl,
            'realized_pnl_percent': realized_pnl_pct,
            'days_held': days_held,
            'status': status
        }
    
    def get_budget_status(self) -> Dict:
        """Get current budget status for insurance program."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        # Total spent this year
        current_year = datetime.now().year
        cursor.execute("""
            SELECT SUM(total_cost) FROM vix_positions 
            WHERE strftime('%Y', entry_date) = ?
        """, (str(current_year),))
        
        spent = cursor.fetchone()[0] or 0
        
        # Realized P&L from closed positions
        cursor.execute("""
            SELECT SUM(realized_pnl) FROM vix_positions 
            WHERE strftime('%Y', entry_date) = ? AND status != 'open'
        """, (str(current_year),))
        
        realized = cursor.fetchone()[0] or 0
        
        # Open position value
        cursor.execute("""
            SELECT SUM(current_value) FROM vix_positions 
            WHERE status = 'open'
        """)
        
        open_value = cursor.fetchone()[0] or 0
        
        conn.close()
        
        net_cost = spent + realized  # realized is negative for losses
        remaining = self.annual_budget - spent
        
        return {
            'annual_budget': self.annual_budget,
            'spent_ytd': spent,
            'realized_pnl_ytd': realized,
            'net_insurance_cost': net_cost,
            'remaining_budget': remaining,
            'open_positions_value': open_value,
            'budget_utilization_percent': (spent / self.annual_budget * 100) if self.annual_budget > 0 else 0
        }
    
    def get_performance_stats(self) -> Dict:
        """Get historical performance statistics."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        # All closed positions
        cursor.execute("""
            SELECT * FROM vix_positions 
            WHERE status != 'open'
            ORDER BY entry_date DESC
        """)
        
        closed = cursor.fetchall()
        
        if not closed:
            conn.close()
            return {'message': 'No closed positions yet'}
        
        wins = sum(1 for c in closed if c[19] and c[19] > 0)  # realized_pnl > 0
        losses = len(closed) - wins
        
        total_pnl = sum(c[19] or 0 for c in closed)
        avg_win = sum(c[19] for c in closed if c[19] and c[19] > 0) / wins if wins > 0 else 0
        avg_loss = sum(c[19] for c in closed if c[19] and c[19] < 0) / losses if losses > 0 else 0
        
        conn.close()
        
        return {
            'total_trades': len(closed),
            'winning_trades': wins,
            'losing_trades': losses,
            'win_rate': wins / len(closed) * 100,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else float('inf')
        }
    
    def _days_to_expiration(self, expiration_date: str) -> int:
        """Calculate days to expiration."""
        exp = datetime.strptime(expiration_date, '%Y-%m-%d').date()
        return (exp - datetime.now().date()).days
    
    def export_positions(self):
        """Export all positions to JSON for dashboard."""
        conn = sqlite3.connect(self.DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM vix_positions ORDER BY entry_date DESC")
        positions = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'open_positions': [p for p in positions if p['status'] == 'open'],
            'closed_positions': [p for p in positions if p['status'] != 'open'],
            'budget_status': self.get_budget_status(),
            'performance_stats': self.get_performance_stats()
        }
        
        with open(self.POSITIONS_PATH, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"Exported positions to {self.POSITIONS_PATH}")
        return output


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='VIX Insurance Position Manager')
    parser.add_argument('--status', action='store_true', help='Show position status')
    parser.add_argument('--budget', action='store_true', help='Show budget status')
    parser.add_argument('--performance', action='store_true', help='Show performance stats')
    parser.add_argument('--export', action='store_true', help='Export positions to JSON')
    
    args = parser.parse_args()
    
    manager = VIXPositionManager(annual_budget=1000)
    
    if args.status:
        positions = manager.get_open_positions()
        print("\n=== Open VIX Insurance Positions ===\n")
        if not positions:
            print("No open positions")
        for p in positions:
            pnl_str = ""
            if p.get('unrealized_pnl') is not None:
                pnl_str = f" | Unrealized: ${p['unrealized_pnl']:,.0f} ({p['unrealized_pnl_percent']:+.1f}%)"
            print(f"ID {p['id']}: {p['contracts']} contracts @ ${p['strike']:.1f} strike")
            print(f"  Entry: {p['entry_date'][:10]} at VIX={p['entry_vix_spot']:.1f}")
            print(f"  Expires: {p['expiration_date']} ({p['days_to_expiration_at_entry'] - p['days_held']}d remaining)")
            print(f"  Cost: ${p['total_cost']:,.0f}{pnl_str}")
            print()
    
    elif args.budget:
        budget = manager.get_budget_status()
        print("\n=== VIX Insurance Budget Status ===\n")
        print(f"Annual Budget: ${budget['annual_budget']:,.0f}")
        print(f"Spent YTD: ${budget['spent_ytd']:,.0f} ({budget['budget_utilization_percent']:.1f}%)")
        print(f"Realized P&L: ${budget['realized_pnl_ytd']:,.0f}")
        print(f"Net Insurance Cost: ${budget['net_insurance_cost']:,.0f}")
        print(f"Remaining Budget: ${budget['remaining_budget']:,.0f}")
        print(f"Open Positions Value: ${budget['open_positions_value']:,.0f}")
    
    elif args.performance:
        stats = manager.get_performance_stats()
        print("\n=== VIX Insurance Performance ===\n")
        if 'message' in stats:
            print(stats['message'])
        else:
            print(f"Total Trades: {stats['total_trades']}")
            print(f"Wins: {stats['winning_trades']} | Losses: {stats['losing_trades']}")
            print(f"Win Rate: {stats['win_rate']:.1f}%")
            print(f"Total P&L: ${stats['total_pnl']:,.0f}")
            print(f"Avg Win: ${stats['avg_win']:,.0f} | Avg Loss: ${stats['avg_loss']:,.0f}")
            print(f"Profit Factor: {stats['profit_factor']:.2f}")
    
    elif args.export:
        manager.export_positions()
        print(f"Exported to {manager.POSITIONS_PATH}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
