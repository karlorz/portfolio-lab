"""
v7.03: Tax Lot Tracker — v7.03
Tracks cost basis per lot per asset, supports FIFO/LIFO/HIFO lot selection,
holding period tracking, and wash sale detection.

Persistence: data/tax_lots_state.json

Usage:
    tracker = TaxLotTracker()
    tracker.add_lot('SPY', 10, 480.0, '2026-05-01')
    tracker.add_lot('SPY', 5, 485.0, '2026-05-10')
    sold = tracker.sell_lots('SPY', 3, method='fifo')
    # sold = [Lot(3, 480.0, ...)]  # oldest lots sold first
    wash = tracker.detect_wash_sales('SPY', '2026-05-01')
    summary = tracker.get_summary()
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum

from src.paths import PROJECT_ROOT


class LotSelectionMethod(Enum):
    """Available lot selection methods for selling."""
    FIFO = "fifo"       # First In, First Out (oldest lots)
    LIFO = "lifo"       # Last In, First Out (newest lots)
    HIFO = "hifo"       # Highest cost basis first (maximize losses)


class HoldingPeriod(Enum):
    SHORT_TERM = "short_term"  # < 1 year
    LONG_TERM = "long_term"    # >= 1 year


@dataclass
class TaxLot:
    """A single tax lot representing a purchase of shares at a specific price/date."""
    symbol: str
    shares: float
    cost_basis_per_share: float
    acquisition_date: str  # ISO format YYYY-MM-DD
    lot_id: str = ""
    
    @property
    def total_cost_basis(self) -> float:
        return round(self.shares * self.cost_basis_per_share, 2)
    
    @property
    def holding_period(self) -> HoldingPeriod:
        acq = datetime.strptime(self.acquisition_date, '%Y-%m-%d').date()
        today = date.today()
        years = (today - acq).days / 365.0
        return HoldingPeriod.LONG_TERM if years >= 1.0 else HoldingPeriod.SHORT_TERM
    
    @property
    def is_long_term(self) -> bool:
        return self.holding_period == HoldingPeriod.LONG_TERM
    
    @property
    def is_short_term(self) -> bool:
        return self.holding_period == HoldingPeriod.SHORT_TERM
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TaxLot':
        return cls(**data)


@dataclass
class SoldLot:
    """Record of a sold lot for tracking realized gains/losses."""
    symbol: str
    shares: float
    cost_basis: float
    sale_price: float
    sale_date: str
    acquisition_date: str
    lot_id: str
    realized_pl: float  # Positive = gain, negative = loss
    
    @property
    def is_loss(self) -> bool:
        return self.realized_pl < 0
    
    @property
    def is_short_term(self) -> bool:
        acq = datetime.strptime(self.acquisition_date, '%Y-%m-%d').date()
        sale = datetime.strptime(self.sale_date, '%Y-%m-%d').date()
        years = (sale - acq).days / 365.0
        return years < 1.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SoldLot':
        return cls(**data)


class TaxLotTracker:
    """
    Tracks cost basis per lot per asset.
    Supports FIFO/LIFO/HIFO lot selection, holding period tracking,
    and wash sale detection.
    """
    
    def __init__(self, state_path: Optional[str] = None):
        self.state_path = Path(state_path) if state_path else PROJECT_ROOT / 'data' / 'tax_lots_state.json'
        self.lots: Dict[str, List[TaxLot]] = {}  # symbol -> list of lots
        self.sold_lots: List[SoldLot] = []
        self._lot_counter = 0
        self._load_state()
    
    def _load_state(self):
        """Load state from JSON file."""
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                self.lots = {}
                for symbol, lot_list in data.get('lots', {}).items():
                    self.lots[symbol] = [TaxLot.from_dict(l) for l in lot_list]
                self.sold_lots = [SoldLot.from_dict(s) for s in data.get('sold_lots', [])]
                self._lot_counter = data.get('lot_counter', 0)
            except (json.JSONDecodeError, KeyError) as e:
                # If file is corrupt, start fresh
                pass
    
    def save_state(self):
        """Persist state to JSON file."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'lots': {
                symbol: [l.to_dict() for l in lot_list]
                for symbol, lot_list in self.lots.items()
            },
            'sold_lots': [s.to_dict() for s in self.sold_lots],
            'lot_counter': self._lot_counter,
        }
        with open(self.state_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _next_lot_id(self) -> str:
        """Generate a unique lot ID."""
        self._lot_counter += 1
        date_str = date.today().strftime('%Y%m%d')
        return f"lot_{date_str}_{self._lot_counter}"
    
    def add_lot(self, symbol: str, shares: float, cost_basis_per_share: float,
                acquisition_date: Optional[str] = None) -> TaxLot:
        """
        Add a new tax lot for a buy transaction.
        
        Args:
            symbol: Ticker symbol (e.g., 'SPY')
            shares: Number of shares purchased
            cost_basis_per_share: Price per share
            acquisition_date: Date of purchase (default: today)
            
        Returns:
            The created TaxLot
        """
        if shares <= 0:
            raise ValueError(f"Shares must be positive, got {shares}")
        if cost_basis_per_share <= 0:
            raise ValueError(f"Cost basis must be positive, got {cost_basis_per_share}")
        
        if acquisition_date is None:
            acquisition_date = date.today().isoformat()
        
        lot_id = self._next_lot_id()
        lot = TaxLot(
            symbol=symbol,
            shares=shares,
            cost_basis_per_share=cost_basis_per_share,
            acquisition_date=acquisition_date,
            lot_id=lot_id,
        )
        
        if symbol not in self.lots:
            self.lots[symbol] = []
        self.lots[symbol].append(lot)
        
        # Sort lots by acquisition date for FIFO consistency
        self.lots[symbol].sort(key=lambda l: l.acquisition_date)
        
        self.save_state()
        return lot
    
    def get_lots(self, symbol: str) -> List[TaxLot]:
        """Get all open lots for a symbol."""
        return self.lots.get(symbol, [])
    
    def get_total_shares(self, symbol: str) -> float:
        """Get total shares held for a symbol."""
        return sum(l.shares for l in self.lots.get(symbol, []))
    
    def get_total_cost_basis(self, symbol: str) -> float:
        """Get total cost basis for a symbol."""
        return sum(l.total_cost_basis for l in self.lots.get(symbol, []))
    
    def get_average_cost_basis(self, symbol: str) -> float:
        """Get average cost basis per share for a symbol."""
        total_shares = self.get_total_shares(symbol)
        if total_shares == 0:
            return 0.0
        return self.get_total_cost_basis(symbol) / total_shares
    
    def _select_lots_fifo(self, symbol: str, shares_to_sell: float) -> List[Tuple[TaxLot, float]]:
        """Select lots using FIFO — oldest lots first."""
        lots = self.get_lots(symbol)
        selected = []
        remaining = shares_to_sell
        
        for lot in lots:
            if remaining <= 0:
                break
            taken = min(lot.shares, remaining)
            selected.append((lot, taken))
            remaining -= taken
        
        if remaining > 0:
            raise ValueError(f"Insufficient shares for {symbol}: need {shares_to_sell}, have {self.get_total_shares(symbol)}")
        
        return selected
    
    def _select_lots_lifo(self, symbol: str, shares_to_sell: float) -> List[Tuple[TaxLot, float]]:
        """Select lots using LIFO — newest lots first."""
        lots = sorted(self.get_lots(symbol), key=lambda l: l.acquisition_date, reverse=True)
        selected = []
        remaining = shares_to_sell
        
        for lot in lots:
            if remaining <= 0:
                break
            taken = min(lot.shares, remaining)
            selected.append((lot, taken))
            remaining -= taken
        
        if remaining > 0:
            raise ValueError(f"Insufficient shares for {symbol}: need {shares_to_sell}, have {self.get_total_shares(symbol)}")
        
        return selected
    
    def _select_lots_hifo(self, symbol: str, shares_to_sell: float) -> List[Tuple[TaxLot, float]]:
        """Select lots using HIFO — highest cost basis first (maximizes losses)."""
        lots = sorted(self.get_lots(symbol), key=lambda l: l.cost_basis_per_share, reverse=True)
        selected = []
        remaining = shares_to_sell
        
        for lot in lots:
            if remaining <= 0:
                break
            taken = min(lot.shares, remaining)
            selected.append((lot, taken))
            remaining -= taken
        
        if remaining > 0:
            raise ValueError(f"Insufficient shares for {symbol}: need {shares_to_sell}, have {self.get_total_shares(symbol)}")
        
        return selected
    
    def sell_lots(self, symbol: str, shares_to_sell: float,
                  sale_price: float, sale_date: Optional[str] = None,
                  method: str = 'fifo') -> List[SoldLot]:
        """
        Sell shares from lots using the specified selection method.
        
        Args:
            symbol: Ticker symbol
            shares_to_sell: Number of shares to sell
            sale_price: Price per share at sale
            sale_date: Date of sale (default: today)
            method: Lot selection method ('fifo', 'lifo', 'hifo')
            
        Returns:
            List of SoldLot records
        """
        if shares_to_sell <= 0:
            raise ValueError(f"Shares to sell must be positive, got {shares_to_sell}")
        if sale_price <= 0:
            raise ValueError(f"Sale price must be positive, got {sale_price}")
        if symbol not in self.lots or not self.lots[symbol]:
            raise ValueError(f"No lots found for {symbol}")
        
        if sale_date is None:
            sale_date = date.today().isoformat()
        
        if method == 'lifo':
            selected = self._select_lots_lifo(symbol, shares_to_sell)
        elif method == 'hifo':
            selected = self._select_lots_hifo(symbol, shares_to_sell)
        else:
            selected = self._select_lots_fifo(symbol, shares_to_sell)
        
        sold_records = []
        for lot, taken in selected:
            realized_pl = round((sale_price - lot.cost_basis_per_share) * taken, 2)
            sold = SoldLot(
                symbol=symbol,
                shares=taken,
                cost_basis=round(lot.cost_basis_per_share * taken, 2),
                sale_price=sale_price,
                sale_date=sale_date,
                acquisition_date=lot.acquisition_date,
                lot_id=lot.lot_id,
                realized_pl=realized_pl,
            )
            sold_records.append(sold)
            self.sold_lots.append(sold)
            
            # Reduce or remove the lot
            if taken >= lot.shares:
                self.lots[symbol].remove(lot)
            else:
                lot.shares -= taken
        
        self.save_state()
        return sold_records
    
    def detect_wash_sales(self, symbol: str, threshold_days: int = 30) -> List[TaxLot]:
        """
        Detect potential wash sales: purchases within threshold_days of a sale at a loss.
        
        Wash sale rule: If you sell a security at a loss and buy the same (or substantially
        identical) security within 30 days before or after the sale, the loss is disallowed.
        
        Returns:
            List of lots acquired within wash sale window of any loss sale
        """
        wash_lots = []
        for sold in self.sold_lots:
            if sold.symbol != symbol or not sold.is_loss:
                continue
            
            sale_dt = datetime.strptime(sold.sale_date, '%Y-%m-%d').date()
            window_start = sale_dt - timedelta(days=threshold_days)
            window_end = sale_dt + timedelta(days=threshold_days)
            
            for lot in self.get_lots(symbol):
                lot_dt = datetime.strptime(lot.acquisition_date, '%Y-%m-%d').date()
                if window_start <= lot_dt <= window_end:
                    if lot not in wash_lots:
                        wash_lots.append(lot)
        
        return wash_lots
    
    def get_unrealized_pl(self, symbol: str, current_price: float) -> float:
        """Get total unrealized P&L for a symbol at the given current price."""
        total_cost = self.get_total_cost_basis(symbol)
        total_shares = self.get_total_shares(symbol)
        current_value = total_shares * current_price
        return round(current_value - total_cost, 2)
    
    def get_realized_pl(self, symbol: Optional[str] = None) -> float:
        """Get total realized P&L across all sold lots."""
        lots = self.sold_lots
        if symbol:
            lots = [l for l in lots if l.symbol == symbol]
        return round(sum(l.realized_pl for l in lots), 2)
    
    def get_summary(self) -> Dict:
        """Get a summary of all tracked lots."""
        summary = {}
        for symbol in sorted(self.lots.keys()):
            lots = self.lots[symbol]
            if not lots:
                continue
            total_shares = sum(l.shares for l in lots)
            total_cost = sum(l.total_cost_basis for l in lots)
            avg_cost = total_cost / total_shares if total_shares else 0
            st_lots = sum(1 for l in lots if l.is_short_term)
            lt_lots = sum(1 for l in lots if l.is_long_term)
            
            summary[symbol] = {
                'total_shares': round(total_shares, 4),
                'total_cost_basis': round(total_cost, 2),
                'average_cost_basis': round(avg_cost, 2),
                'num_lots': len(lots),
                'short_term_lots': st_lots,
                'long_term_lots': lt_lots,
            }
        
        return {
            'holdings': summary,
            'total_symbols': len(summary),
            'total_lots': sum(len(lots) for lots in self.lots.values()),
            'total_realized_pl': self.get_realized_pl(),
            'total_sold_lots': len(self.sold_lots),
        }
    
    def reset(self):
        """Clear all tracked lots for simulation reset."""
        self.lots = {}
        self.sold_lots = []
        self._lot_counter = 0
        self.save_state()
    
    def count_lots_for_symbol(self, symbol: str) -> int:
        """Count number of open lots for a symbol."""
        return len(self.lots.get(symbol, []))
    
    def get_wash_sales_stats(self) -> Dict:
        """Get wash sale statistics across all symbols."""
        stats = {}
        for symbol in self.lots:
            washes = self.detect_wash_sales(symbol)
            stats[symbol] = {
                'wash_lots_count': len(washes),
                'wash_shares': round(sum(l.shares for l in washes), 4),
                'wash_cost_basis': round(sum(l.total_cost_basis for l in washes), 2),
            }
        return stats


# Convenience module-level functions
def create_tracker() -> TaxLotTracker:
    """Create a default TaxLotTracker instance."""
    return TaxLotTracker()


def demo():
    """Demonstrate the tax lot tracker."""
    tracker = TaxLotTracker()
    tracker.reset()
    
    # Simulate purchases
    tracker.add_lot('SPY', 10, 475.0, '2025-01-15')
    tracker.add_lot('SPY', 5, 490.0, '2025-06-01')
    tracker.add_lot('SPY', 8, 482.0, '2026-03-10')
    
    print("=== Tax Lot Tracker Demo ===")
    print(f"SPY shares: {tracker.get_total_shares('SPY')}")
    print(f"SPY avg cost: ${tracker.get_average_cost_basis('SPY'):.2f}")
    print(f"SPY lots: {tracker.count_lots_for_symbol('SPY')}")
    
    # Sell 6 shares FIFO
    sold = tracker.sell_lots('SPY', 6, 500.0, '2026-05-16', method='fifo')
    print(f"\nSold 6 shares FIFO at $500:")
    for s in sold:
        term = "ST" if s.is_short_term else "LT"
        pnl_type = "GAIN" if s.realized_pl > 0 else "LOSS"
        print(f"  {term} Lot {s.lot_id}: {s.shares}sh cost=${s.cost_basis:.2f} → ${s.realized_pl:.2f} {pnl_type}")
    
    # Check wash sales
    washes = tracker.detect_wash_sales('SPY')
    print(f"\nWash lots detected: {len(washes)}")
    
    # Summary
    import json
    print(f"\nSummary:\n{json.dumps(tracker.get_summary(), indent=2)}")
    
    tracker.save_state()


if __name__ == '__main__':
    demo()
