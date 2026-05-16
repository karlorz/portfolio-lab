"""
Tests for TaxLotTracker — v7.03
Tests lot tracking, FIFO/LIFO/HIFO selection, wash sale detection,
and state persistence.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import date, timedelta

from src.rebalancing.tax_lot_tracker import (
    TaxLotTracker,
    TaxLot,
    SoldLot,
    LotSelectionMethod,
    HoldingPeriod,
)


class TestTaxLotCreation:
    """Test creating and tracking tax lots."""

    def test_add_lot_basic(self):
        tracker = TaxLotTracker()
        tracker.reset()
        lot = tracker.add_lot('SPY', 10, 500.0)
        assert lot.symbol == 'SPY'
        assert lot.shares == 10
        assert lot.cost_basis_per_share == 500.0
        assert lot.lot_id.startswith('lot_')
        assert tracker.get_total_shares('SPY') == 10

    def test_add_lot_with_date(self):
        tracker = TaxLotTracker()
        tracker.reset()
        lot = tracker.add_lot('SPY', 10, 500.0, '2025-01-15')
        assert lot.acquisition_date == '2025-01-15'

    def test_add_lot_negative_shares(self):
        tracker = TaxLotTracker()
        tracker.reset()
        with pytest.raises(ValueError, match="Shares must be positive"):
            tracker.add_lot('SPY', -1, 500.0)

    def test_add_lot_zero_cost(self):
        tracker = TaxLotTracker()
        tracker.reset()
        with pytest.raises(ValueError, match="Cost basis must be positive"):
            tracker.add_lot('SPY', 10, 0)

    def test_add_multiple_lots_same_symbol(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 10, 500.0, '2025-01-15')
        tracker.add_lot('SPY', 5, 510.0, '2025-06-01')
        assert tracker.get_total_shares('SPY') == 15
        assert tracker.count_lots_for_symbol('SPY') == 2

    def test_add_multiple_symbols(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 10, 500.0)
        tracker.add_lot('GLD', 20, 190.0)
        assert tracker.get_total_shares('SPY') == 10
        assert tracker.get_total_shares('GLD') == 20


class TestLotProperties:
    """Test TaxLot property accessors."""

    def test_total_cost_basis(self):
        lot = TaxLot('SPY', 10, 500.0, '2025-01-01', 'lot_1')
        assert lot.total_cost_basis == 5000.0

    def test_holding_period_long_term(self):
        past = (date.today() - timedelta(days=400)).isoformat()
        lot = TaxLot('SPY', 10, 500.0, past, 'lot_1')
        assert lot.holding_period == HoldingPeriod.LONG_TERM
        assert lot.is_long_term is True
        assert lot.is_short_term is False

    def test_holding_period_short_term(self):
        recent = (date.today() - timedelta(days=30)).isoformat()
        lot = TaxLot('SPY', 10, 500.0, recent, 'lot_1')
        assert lot.holding_period == HoldingPeriod.SHORT_TERM
        assert lot.is_long_term is False
        assert lot.is_short_term is True

    def test_lot_serialization(self):
        lot = TaxLot('SPY', 10, 500.0, '2025-01-15', 'lot_1')
        d = lot.to_dict()
        assert d['symbol'] == 'SPY'
        assert d['shares'] == 10
        restored = TaxLot.from_dict(d)
        assert restored.symbol == lot.symbol
        assert restored.shares == lot.shares
        assert restored.lot_id == lot.lot_id


class TestLotSelection:
    """Test FIFO/LIFO/HIFO lot selection methods."""

    def test_fifo_sell_all(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 10, 480.0, '2025-01-15')
        tracker.add_lot('SPY', 5, 500.0, '2025-06-01')
        sold = tracker.sell_lots('SPY', 15, 510.0, '2026-05-16', method='fifo')
        assert len(sold) == 2
        assert sold[0].shares == 10  # First lot (older)
        assert sold[1].shares == 5   # Second lot
        assert tracker.get_total_shares('SPY') == 0

    def test_fifo_partial_sell_oldest_first(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 10, 480.0, '2025-01-15')
        tracker.add_lot('SPY', 5, 500.0, '2025-06-01')
        sold = tracker.sell_lots('SPY', 3, 510.0, '2026-05-16', method='fifo')
        assert len(sold) == 1
        assert sold[0].shares == 3
        assert sold[0].acquisition_date == '2025-01-15'
        # Remaining: 7 in first lot, 5 in second
        assert tracker.get_total_shares('SPY') == 12

    def test_lifo_sell_newest_first(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 10, 480.0, '2025-01-15')
        tracker.add_lot('SPY', 5, 500.0, '2025-06-01')
        sold = tracker.sell_lots('SPY', 5, 510.0, '2026-05-16', method='lifo')
        assert len(sold) == 1
        assert sold[0].shares == 5
        assert sold[0].acquisition_date == '2025-06-01'
        assert tracker.get_total_shares('SPY') == 10  # Only first lot remains

    def test_hifo_sell_highest_cost_first(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 10, 480.0, '2025-01-15')
        tracker.add_lot('SPY', 5, 520.0, '2025-06-01')  # Higher cost
        tracker.add_lot('SPY', 8, 500.0, '2025-03-10')
        sold = tracker.sell_lots('SPY', 5, 510.0, '2026-05-16', method='hifo')
        assert len(sold) == 1
        per_share_cost = sold[0].cost_basis / sold[0].shares
        assert abs(per_share_cost - 520.0) < 0.001  # Highest cost first

    def test_insufficient_shares_error(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 10, 500.0, '2025-01-15')
        with pytest.raises(ValueError, match="Insufficient shares"):
            tracker.sell_lots('SPY', 20, 510.0)

    def test_no_lots_error(self):
        tracker = TaxLotTracker()
        tracker.reset()
        with pytest.raises(ValueError, match="No lots found"):
            tracker.sell_lots('SPY', 10, 510.0)

    def test_sold_lot_properties(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 10, 480.0, '2025-01-15')
        sold = tracker.sell_lots('SPY', 5, 500.0, '2026-05-16')
        sl = sold[0]
        assert sl.symbol == 'SPY'
        assert sl.shares == 5
        assert abs(sl.realized_pl - 100.0) < 0.01  # (500-480) * 5
        assert sl.is_loss is False
        assert sl.is_short_term is False  # > 1 year


class TestWashSaleDetection:
    """Test wash sale detection."""

    def test_no_wash_sale(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 10, 500.0, '2025-01-15')
        tracker.sell_lots('SPY', 5, 480.0, '2025-06-01')  # Loss
        # No purchases within 30 days of this sale
        washes = tracker.detect_wash_sales('SPY')
        assert len(washes) == 0

    def test_wash_sale_detected_before_sale(self):
        tracker = TaxLotTracker()
        tracker.reset()
        # Buy within 30 days before a loss sale
        tracker.add_lot('SPY', 10, 500.0, '2025-01-15')  # Old lot
        tracker.add_lot('SPY', 5, 490.0, '2025-05-20')   # Within 30 days before
        tracker.sell_lots('SPY', 8, 480.0, '2025-06-01')  # Loss sale
        washes = tracker.detect_wash_sales('SPY')
        assert len(washes) >= 1

    def test_wash_sale_detected_after_sale(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 5, 500.0, '2025-01-15')
        tracker.sell_lots('SPY', 5, 480.0, '2025-06-01')  # Loss sale
        # Buy within 30 days after the loss sale
        tracker.add_lot('SPY', 5, 485.0, '2025-06-15')
        washes = tracker.detect_wash_sales('SPY')
        assert len(washes) >= 1

    def test_wash_sale_gain_not_counted(self):
        """Wash sales only trigger on loss sales."""
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 10, 500.0, '2025-01-15')
        tracker.sell_lots('SPY', 5, 520.0, '2025-06-01')  # Gain (not loss)
        tracker.add_lot('SPY', 5, 510.0, '2025-06-15')  # Would be wash if it were a loss
        washes = tracker.detect_wash_sales('SPY')
        assert len(washes) == 0  # Sale was a gain, not a loss


class TestPersistence:
    """Test state save/load."""

    def test_save_and_load_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'tax_state.json'
            tracker1 = TaxLotTracker(str(path))
            tracker1.add_lot('SPY', 10, 500.0, '2025-01-15')
            tracker1.add_lot('GLD', 20, 190.0, '2025-04-01')
            tracker1.save_state()
            
            tracker2 = TaxLotTracker(str(path))
            assert tracker2.get_total_shares('SPY') == 10
            assert tracker2.get_total_shares('GLD') == 20
            assert tracker2.count_lots_for_symbol('SPY') == 1
            assert tracker2.count_lots_for_symbol('GLD') == 1

    def test_persistence_after_sell(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'tax_state.json'
            tracker1 = TaxLotTracker(str(path))
            tracker1.add_lot('SPY', 10, 500.0, '2025-01-15')
            tracker1.sell_lots('SPY', 5, 510.0, '2025-06-01')
            tracker1.save_state()
            
            tracker2 = TaxLotTracker(str(path))
            assert tracker2.get_total_shares('SPY') == 5
            assert len(tracker2.sold_lots) == 1

    def test_reset(self):
        tracker = TaxLotTracker()
        tracker.reset()  # Clear any persisted state
        tracker.add_lot('SPY', 10, 500.0, '2025-01-15')
        assert tracker.get_total_shares('SPY') == 10
        tracker.reset()
        assert tracker.get_total_shares('SPY') == 0
        assert len(tracker.sold_lots) == 0


class TestSummaryAndPL:
    """Test portfolio summary and P&L calculations."""

    def test_get_summary_empty(self):
        tracker = TaxLotTracker()
        tracker.reset()
        summary = tracker.get_summary()
        assert summary['total_symbols'] == 0
        assert summary['total_lots'] == 0

    def test_get_summary_with_holdings(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 10, 500.0, '2025-01-15')
        tracker.add_lot('SPY', 5, 520.0, '2025-06-01')
        tracker.add_lot('GLD', 20, 190.0, '2025-04-01')
        summary = tracker.get_summary()
        assert summary['total_symbols'] == 2
        assert summary['total_lots'] == 3
        assert summary['holdings']['SPY']['total_shares'] == 15
        assert summary['holdings']['GLD']['total_shares'] == 20

    def test_average_cost_basis(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 10, 500.0, '2025-01-15')
        tracker.add_lot('SPY', 5, 520.0, '2025-06-01')
        avg = tracker.get_average_cost_basis('SPY')
        expected = (10 * 500 + 5 * 520) / 15
        assert abs(avg - expected) < 0.01

    def test_realized_pl(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 10, 500.0, '2025-01-15')
        tracker.sell_lots('SPY', 5, 520.0)  # Gain of $100
        tracker.sell_lots('SPY', 3, 490.0)  # Loss of $30
        realized = tracker.get_realized_pl('SPY')
        assert abs(realized - 70.0) < 1.0  # $100 - $30 (some rounding)

    def test_unrealized_pl(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 10, 500.0, '2025-01-15')
        pl = tracker.get_unrealized_pl('SPY', 510.0)
        assert abs(pl - 100.0) < 0.01  # (510-500) * 10
