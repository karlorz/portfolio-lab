"""
VIX Position Manager
Tracks and manages VIX call spread positions for insurance overlay.

Part of v2.44: VIX Call Spread Insurance Overlay - Phase 3
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    ROLL_PENDING = "roll_pending"
    EXPIRED = "expired"


@dataclass
class VIXPosition:
    """Represents a VIX call position"""
    position_id: str
    status: PositionStatus
    
    # Entry details
    entry_date: datetime
    entry_vix: float
    contracts: int
    strike: float
    expiration: datetime
    entry_premium: float
    total_cost: float
    
    # Current state
    current_vix: float
    current_value: float
    unrealized_pnl: float
    
    # Exit details (if closed)
    exit_date: Optional[datetime] = None
    exit_premium: Optional[float] = None
    exit_reason: Optional[str] = None
    realized_pnl: Optional[float] = None
    
    # Metadata
    days_held: int = 0
    days_to_expiry: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        data['entry_date'] = self.entry_date.isoformat()
        data['expiration'] = self.expiration.isoformat()
        data['exit_date'] = self.exit_date.isoformat() if self.exit_date else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VIXPosition':
        """Create from dictionary"""
        return cls(
            position_id=data['position_id'],
            status=PositionStatus(data['status']),
            entry_date=datetime.fromisoformat(data['entry_date']),
            entry_vix=data['entry_vix'],
            contracts=data['contracts'],
            strike=data['strike'],
            expiration=datetime.fromisoformat(data['expiration']),
            entry_premium=data['entry_premium'],
            total_cost=data['total_cost'],
            current_vix=data.get('current_vix', data['entry_vix']),
            current_value=data.get('current_value', 0.0),
            unrealized_pnl=data.get('unrealized_pnl', 0.0),
            exit_date=datetime.fromisoformat(data['exit_date']) if data.get('exit_date') else None,
            exit_premium=data.get('exit_premium'),
            exit_reason=data.get('exit_reason'),
            realized_pnl=data.get('realized_pnl'),
            days_held=data.get('days_held', 0),
            days_to_expiry=data.get('days_to_expiry', 0)
        )


@dataclass
class PositionSummary:
    """Summary of all VIX insurance positions"""
    timestamp: datetime
    
    # Active positions
    active_positions: List[VIXPosition]
    total_active_contracts: int
    total_active_notional: float
    total_active_cost: float
    total_unrealized_pnl: float
    
    # Closed positions (YTD)
    closed_positions_count: int
    total_realized_pnl: float
    win_count: int
    loss_count: int
    win_rate: float
    
    # Cost tracking
    total_premiums_paid: float
    total_payouts_received: float
    net_insurance_cost: float
    
    # Budget
    annual_budget: float
    budget_used: float
    budget_remaining: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'active_positions': [p.to_dict() for p in self.active_positions],
            'total_active_contracts': self.total_active_contracts,
            'total_active_notional': self.total_active_notional,
            'total_active_cost': self.total_active_cost,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'closed_positions_count': self.closed_positions_count,
            'total_realized_pnl': self.total_realized_pnl,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': self.win_rate,
            'total_premiums_paid': self.total_premiums_paid,
            'total_payouts_received': self.total_payouts_received,
            'net_insurance_cost': self.net_insurance_cost,
            'annual_budget': self.annual_budget,
            'budget_used': self.budget_used,
            'budget_remaining': self.budget_remaining
        }


class VIXPositionManager:
    """
    Manages VIX call spread positions for insurance overlay.
    
    Responsibilities:
    - Track open positions
    - Calculate mark-to-market P&L
    - Manage rolls and exits
    - Track budget and costs
    - Generate position summary for dashboard
    """
    
    # Strategy parameters (match signal generator)
    MAX_ALLOCATION_PCT = 0.01  # 1% max allocation
    VIX_MULTIPLIER = 100  # VIX options multiplier
    ANNUAL_BUDGET_PCT = 0.008  # 0.8% annual budget
    
    def __init__(self, portfolio_value: float = 100000, 
                 data_dir: str = "data/options"):
        self.portfolio_value = portfolio_value
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Files
        self.positions_file = self.data_dir / "vix_positions.json"
        self.history_file = self.data_dir / "vix_position_history.json"
        self.summary_file = self.data_dir / "vix_position_summary.json"
        
        # Annual budget
        self.annual_budget = portfolio_value * self.ANNUAL_BUDGET_PCT
        
        logger.info(f"VIX Position Manager initialized: portfolio=${portfolio_value:,.0f}, "
                   f"budget=${self.annual_budget:,.0f}")
    
    def load_positions(self) -> List[VIXPosition]:
        """Load all positions from file"""
        if not self.positions_file.exists():
            return []
        
        with open(self.positions_file, 'r') as f:
            data = json.load(f)
        
        return [VIXPosition.from_dict(p) if isinstance(p, dict) else p for p in data]
    
    def save_positions(self, positions: List[VIXPosition]):
        """Save positions to file"""
        with open(self.positions_file, 'w') as f:
            json.dump([p.to_dict() for p in positions], f, indent=2, default=str)
    
    def load_history(self) -> List[Dict]:
        """Load position history (closed positions)"""
        if not self.history_file.exists():
            return []
        
        with open(self.history_file, 'r') as f:
            return json.load(f)
    
    def save_to_history(self, position: VIXPosition):
        """Add closed position to history"""
        history = self.load_history()
        history.append(position.to_dict())
        
        # Keep last 100 positions
        if len(history) > 100:
            history = history[-100:]
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)
    
    def open_position(self, contracts: int, strike: float, 
                    expiration: datetime, premium: float,
                    current_vix: float) -> VIXPosition:
        """
        Open a new VIX call position.
        
        Args:
            contracts: Number of contracts
            strike: Strike price
            expiration: Expiration date
            premium: Premium per contract
            current_vix: Current VIX spot price
        
        Returns:
            VIXPosition object
        """
        position_id = f"VIX_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        total_cost = contracts * premium * self.VIX_MULTIPLIER
        
        position = VIXPosition(
            position_id=position_id,
            status=PositionStatus.OPEN,
            entry_date=datetime.now(),
            entry_vix=current_vix,
            contracts=contracts,
            strike=strike,
            expiration=expiration,
            entry_premium=premium,
            total_cost=total_cost,
            current_vix=current_vix,
            current_value=total_cost,  # Initially mark at cost
            unrealized_pnl=0.0,
            days_held=0,
            days_to_expiry=(expiration - datetime.now()).days
        )
        
        # Load existing positions
        positions = self.load_positions()
        
        # Check if we already have an active position (shouldn't happen)
        active = [p for p in positions if p.status == PositionStatus.OPEN]
        if active:
            logger.warning(f"Already have {len(active)} active positions. Consider rolling instead.")
        
        positions.append(position)
        self.save_positions(positions)
        
        logger.info(f"Opened VIX position: {position_id}, {contracts} contracts @ {strike}, "
                   f"premium=${premium:.2f}, total_cost=${total_cost:.2f}")
        
        return position
    
    def close_position(self, position_id: str, exit_premium: float,
                      reason: str) -> Optional[VIXPosition]:
        """
        Close a VIX call position.
        
        Args:
            position_id: Position ID to close
            exit_premium: Premium received (sale price)
            reason: Reason for closing
        
        Returns:
            Updated VIXPosition or None if not found
        """
        positions = self.load_positions()
        
        for i, pos in enumerate(positions):
            if pos.position_id == position_id and pos.status == PositionStatus.OPEN:
                # Calculate P&L
                exit_value = pos.contracts * exit_premium * self.VIX_MULTIPLIER
                realized_pnl = exit_value - pos.total_cost
                
                # Update position
                pos.status = PositionStatus.CLOSED
                pos.exit_date = datetime.now()
                pos.exit_premium = exit_premium
                pos.exit_reason = reason
                pos.realized_pnl = realized_pnl
                pos.current_value = exit_value
                pos.days_held = (pos.exit_date - pos.entry_date).days
                
                # Save to history
                self.save_to_history(pos)
                
                # Remove from active positions
                positions.pop(i)
                self.save_positions(positions)
                
                logger.info(f"Closed VIX position: {position_id}, exit_premium=${exit_premium:.2f}, "
                           f"P&L=${realized_pnl:.2f}, reason={reason}")
                
                return pos
        
        logger.warning(f"Position {position_id} not found or already closed")
        return None
    
    def mark_to_market(self, position_id: str, current_vix: float,
                       current_premium: float) -> Optional[VIXPosition]:
        """
        Mark position to market with current option premium.
        
        Args:
            position_id: Position ID
            current_vix: Current VIX spot
            current_premium: Current option premium per contract
        
        Returns:
            Updated position or None
        """
        positions = self.load_positions()
        
        for pos in positions:
            if pos.position_id == position_id and pos.status == PositionStatus.OPEN:
                pos.current_vix = current_vix
                pos.current_value = pos.contracts * current_premium * self.VIX_MULTIPLIER
                pos.unrealized_pnl = pos.current_value - pos.total_cost
                pos.days_held = (datetime.now() - pos.entry_date).days
                pos.days_to_expiry = (pos.expiration - datetime.now()).days
                
                self.save_positions(positions)
                return pos
        
        return None
    
    def mark_all_to_market(self, vix: float, 
                           premium_func=None) -> List[VIXPosition]:
        """
        Mark all active positions to market.
        
        Args:
            vix: Current VIX spot price
            premium_func: Optional function(strike, days_to_expiry) -> premium
        
        Returns:
            List of updated positions
        """
        positions = self.load_positions()
        active = [p for p in positions if p.status == PositionStatus.OPEN]
        
        updated = []
        for pos in active:
            if premium_func:
                days_to_exp = (pos.expiration - datetime.now()).days
                current_premium = premium_func(pos.strike, days_to_exp)
            else:
                # Simple intrinsic value approximation
                # Real implementation would use Black-Scholes or market data
                intrinsic = max(0, vix - pos.strike)
                time_value = max(0, pos.entry_premium * 0.3)  # Simplified
                current_premium = intrinsic + time_value
            
            updated_pos = self.mark_to_market(pos.position_id, vix, current_premium)
            if updated_pos:
                updated.append(updated_pos)
        
        return updated
    
    def check_roll_needed(self) -> List[VIXPosition]:
        """
        Check which positions need to be rolled (30 DTE threshold).
        
        Returns:
            List of positions needing roll
        """
        positions = self.load_positions()
        active = [p for p in positions if p.status == PositionStatus.OPEN]
        
        to_roll = []
        for pos in active:
            days_to_expiry = (pos.expiration - datetime.now()).days
            if days_to_expiry <= 30:
                pos.status = PositionStatus.ROLL_PENDING
                to_roll.append(pos)
        
        if to_roll:
            self.save_positions(positions)
        
        return to_roll
    
    def execute_roll(self, old_position_id: str, new_contracts: int,
                    new_strike: float, new_expiration: datetime,
                    new_premium: float, current_vix: float) -> Tuple[VIXPosition, VIXPosition]:
        """
        Execute a roll: close old position, open new one.
        
        Args:
            old_position_id: Position to roll
            new_contracts: Contracts for new position
            new_strike: New strike price
            new_expiration: New expiration
            new_premium: Premium for new position
            current_vix: Current VIX spot
        
        Returns:
            Tuple of (closed_position, new_position)
        """
        # Mark old position to market first
        positions = self.load_positions()
        old_pos = None
        for p in positions:
            if p.position_id == old_position_id:
                old_pos = p
                break
        
        if not old_pos:
            raise ValueError(f"Position {old_position_id} not found")
        
        # Estimate exit premium (simplified - in reality use market data)
        intrinsic = max(0, current_vix - old_pos.strike)
        time_value = old_pos.entry_premium * 0.2  # Decayed time value
        exit_premium = intrinsic + time_value
        
        # Close old position
        closed = self.close_position(old_position_id, exit_premium, "roll")
        
        # Open new position
        new_pos = self.open_position(new_contracts, new_strike, new_expiration,
                                     new_premium, current_vix)
        
        logger.info(f"Rolled position {old_position_id} -> {new_pos.position_id}")
        
        return closed, new_pos
    
    def get_position_summary(self) -> PositionSummary:
        """
        Generate comprehensive position summary.
        
        Returns:
            PositionSummary with all metrics
        """
        positions = self.load_positions()
        history = self.load_history()
        
        # Active positions
        active = [p for p in positions if p.status == PositionStatus.OPEN]
        
        total_contracts = sum(p.contracts for p in active)
        total_notional = sum(p.contracts * p.strike * self.VIX_MULTIPLIER 
                           for p in active)
        total_cost = sum(p.total_cost for p in active)
        total_unrealized = sum(p.unrealized_pnl for p in active)
        
        # Closed positions
        closed_count = len(history)
        total_realized = sum(h.get('realized_pnl', 0) for h in history)
        wins = sum(1 for h in history if h.get('realized_pnl', 0) > 0)
        losses = closed_count - wins
        win_rate = wins / closed_count if closed_count > 0 else 0.0
        
        # Cost tracking
        total_premiums = sum(h.get('total_cost', 0) for h in history) + total_cost
        total_payouts = sum(h.get('current_value', 0) for h in history 
                          if h.get('status') == PositionStatus.CLOSED.value)
        net_cost = total_premiums - total_payouts
        
        # Budget
        budget_used = net_cost
        budget_remaining = self.annual_budget - budget_used
        
        return PositionSummary(
            timestamp=datetime.now(),
            active_positions=active,
            total_active_contracts=total_contracts,
            total_active_notional=total_notional,
            total_active_cost=total_cost,
            total_unrealized_pnl=total_unrealized,
            closed_positions_count=closed_count,
            total_realized_pnl=total_realized,
            win_count=wins,
            loss_count=losses,
            win_rate=win_rate,
            total_premiums_paid=total_premiums,
            total_payouts_received=total_payouts,
            net_insurance_cost=net_cost,
            annual_budget=self.annual_budget,
            budget_used=budget_used,
            budget_remaining=budget_remaining
        )
    
    def save_summary(self):
        """Save current position summary to file"""
        summary = self.get_position_summary()
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        
        return summary
    
    def get_dashboard_data(self) -> Dict:
        """
        Get simplified data for dashboard display.
        
        Returns:
            Dictionary with key metrics
        """
        summary = self.get_position_summary()
        
        # Simplified view
        active_pos = summary.active_positions[0] if summary.active_positions else None
        
        return {
            'insurance_active': len(summary.active_positions) > 0,
            'position_count': len(summary.active_positions),
            'contracts': summary.total_active_contracts,
            'strike': active_pos.strike if active_pos else None,
            'expiration': active_pos.expiration.isoformat() if active_pos else None,
            'days_to_expiry': active_pos.days_to_expiry if active_pos else None,
            'entry_vix': active_pos.entry_vix if active_pos else None,
            'current_vix': active_pos.current_vix if active_pos else None,
            'unrealized_pnl': summary.total_unrealized_pnl,
            'realized_pnl_ytd': summary.total_realized_pnl,
            'total_cost': summary.total_active_cost,
            'net_cost_ytd': summary.net_insurance_cost,
            'annual_budget': summary.annual_budget,
            'budget_used_pct': (summary.budget_used / summary.annual_budget * 100) 
                            if summary.annual_budget > 0 else 0,
            'win_rate': summary.win_rate,
            'roll_needed': any(p.days_to_expiry <= 30 for p in summary.active_positions)
        }


def main():
    """CLI entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VIX Position Manager')
    parser.add_argument('--portfolio', type=float, default=100000,
                       help='Portfolio value (default: 100000)')
    parser.add_argument('--action', choices=['summary', 'test'],
                       default='summary', help='Action to perform')
    
    args = parser.parse_args()
    
    manager = VIXPositionManager(portfolio_value=args.portfolio)
    
    if args.action == 'summary':
        summary = manager.get_position_summary()
        
        print(f"\n{'='*60}")
        print("VIX INSURANCE POSITION SUMMARY")
        print(f"{'='*60}\n")
        
        print(f"Active Positions: {len(summary.active_positions)}")
        if summary.active_positions:
            for pos in summary.active_positions:
                print(f"  - {pos.position_id}: {pos.contracts} contracts @ {pos.strike}, "
                      f"P&L: ${pos.unrealized_pnl:,.2f}")
        
        print(f"\nTotal Contracts: {summary.total_active_contracts}")
        print(f"Total Notional: ${summary.total_active_notional:,.2f}")
        print(f"Total Cost: ${summary.total_active_cost:,.2f}")
        print(f"Unrealized P&L: ${summary.total_unrealized_pnl:,.2f}")
        
        print(f"\nClosed Positions (YTD): {summary.closed_positions_count}")
        print(f"Realized P&L: ${summary.total_realized_pnl:,.2f}")
        print(f"Win Rate: {summary.win_rate*100:.1f}%")
        
        print(f"\nBudget:")
        print(f"  Annual: ${summary.annual_budget:,.2f}")
        print(f"  Used: ${summary.budget_used:,.2f}")
        print(f"  Remaining: ${summary.budget_remaining:,.2f}")
        
        print(f"\nNet Insurance Cost YTD: ${summary.net_insurance_cost:,.2f}")
        
        # Dashboard data
        print(f"\n{'='*60}")
        print("DASHBOARD DATA")
        print(f"{'='*60}")
        import json
        print(json.dumps(manager.get_dashboard_data(), indent=2))
        
    elif args.action == 'test':
        print("Running test scenario...")
        
        # Simulate opening a position
        from datetime import timedelta
        exp = datetime.now() + timedelta(days=60)
        pos = manager.open_position(
            contracts=10,
            strike=22.0,
            expiration=exp,
            premium=1.0,
            current_vix=18.0
        )
        
        print(f"Opened: {pos.position_id}")
        
        # Mark to market (VIX spike scenario)
        manager.mark_to_market(pos.position_id, 25.0, 3.5)
        
        # Get updated position
        positions = manager.load_positions()
        updated = [p for p in positions if p.position_id == pos.position_id][0]
        
        print(f"After VIX spike to 25:")
        print(f"  Current value: ${updated.current_value:,.2f}")
        print(f"  Unrealized P&L: ${updated.unrealized_pnl:,.2f}")
        
        # Close position
        closed = manager.close_position(pos.position_id, 3.5, "profit_taking")
        print(f"Closed with P&L: ${closed.realized_pnl:,.2f}")
        
        # Final summary
        summary = manager.get_position_summary()
        print(f"\nFinal win rate: {summary.win_rate*100:.0f}%")
        print(f"Net cost: ${summary.net_insurance_cost:,.2f}")


if __name__ == '__main__':
    main()
