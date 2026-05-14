"""
Tests for Closing Auction Executor (v3.17 Phase 3)

Unit and integration tests for the MOC/IOC execution logic.
"""

import json
import pytest
from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.execution.closing_auction_executor import (
    ClosingAuctionExecutor,
    ClosingAuctionPosition,
    ExecutionConfig,
    OrderStatus,
    ClosingAuctionScheduler,
)
from src.signals.closing_auction import (
    ClosingAuctionSignal,
    SignalDirection,
    SignalConfidence,
    MOCImbalance,
)


class TestClosingAuctionPosition:
    """Test the ClosingAuctionPosition dataclass."""
    
    def test_position_creation(self):
        """Test creating a position."""
        imbalance = MOCImbalance(
            symbol="SPY",
            timestamp=datetime.now(),
            imbalance_shares=1000000,
            paired_shares=5000000,
            reference_price=450.0,
            source="test",
        )
        
        signal = ClosingAuctionSignal(
            symbol="SPY",
            timestamp=datetime.now(),
            direction=SignalDirection.BUY,
            confidence=SignalConfidence.HIGH,
            imbalance=imbalance,
            entry_price=450.0,
            target_exit_price=452.0,
            stop_loss_price=448.0,
            historical_win_rate=0.62,
            historical_count=150,
            max_position_pct=0.03,
            urgency="high",
        )
        
        position = ClosingAuctionPosition(
            symbol="SPY",
            entry_signal=signal,
            entry_time=datetime.now(),
            entry_price=450.0,
            shares=100,
            side='long',
            status=OrderStatus.PENDING,
        )
        
        assert position.symbol == "SPY"
        assert position.entry_price == 450.0
        assert position.shares == 100
        assert position.side == 'long'
        assert position.status == OrderStatus.PENDING
        assert position.pnl == 0.0
    
    def test_position_pnl_calculation_long(self):
        """Test P&L calculation for long position."""
        imbalance = MOCImbalance(
            symbol="SPY",
            timestamp=datetime.now(),
            imbalance_shares=1000000,
            paired_shares=5000000,
            reference_price=450.0,
            source="test",
        )
        
        signal = ClosingAuctionSignal(
            symbol="SPY",
            timestamp=datetime.now(),
            direction=SignalDirection.BUY,
            confidence=SignalConfidence.HIGH,
            imbalance=imbalance,
            entry_price=450.0,
            target_exit_price=452.0,
            stop_loss_price=448.0,
            historical_win_rate=0.62,
            historical_count=150,
            max_position_pct=0.03,
            urgency="high",
        )
        
        position = ClosingAuctionPosition(
            symbol="SPY",
            entry_signal=signal,
            entry_time=datetime.now(),
            entry_price=450.0,
            shares=100,
            side='long',
        )
        
        # Price went up $5
        pnl = position.calculate_pnl(455.0)
        assert pnl == (455.0 - 450.0) * 100  # $500
        
        # Price went down $5
        pnl = position.calculate_pnl(445.0)
        assert pnl == (445.0 - 450.0) * 100  # -$500
    
    def test_position_pnl_calculation_short(self):
        """Test P&L calculation for short position."""
        imbalance = MOCImbalance(
            symbol="SPY",
            timestamp=datetime.now(),
            imbalance_shares=-1000000,
            paired_shares=5000000,
            reference_price=450.0,
            source="test",
        )
        
        signal = ClosingAuctionSignal(
            symbol="SPY",
            timestamp=datetime.now(),
            direction=SignalDirection.SELL,
            confidence=SignalConfidence.HIGH,
            imbalance=imbalance,
            entry_price=450.0,
            target_exit_price=448.0,
            stop_loss_price=452.0,
            historical_win_rate=0.62,
            historical_count=150,
            max_position_pct=0.03,
            urgency="high",
        )
        
        position = ClosingAuctionPosition(
            symbol="SPY",
            entry_signal=signal,
            entry_time=datetime.now(),
            entry_price=450.0,
            shares=100,
            side='short',
        )
        
        # Price went down $5 - profit for short
        pnl = position.calculate_pnl(445.0)
        assert pnl == (450.0 - 445.0) * 100  # $500
        
        # Price went up $5 - loss for short
        pnl = position.calculate_pnl(455.0)
        assert pnl == (450.0 - 455.0) * 100  # -$500
    
    def test_position_close(self):
        """Test closing a position."""
        imbalance = MOCImbalance(
            symbol="SPY",
            timestamp=datetime.now(),
            imbalance_shares=1000000,
            paired_shares=5000000,
            reference_price=450.0,
            source="test",
        )
        
        signal = ClosingAuctionSignal(
            symbol="SPY",
            timestamp=datetime.now(),
            direction=SignalDirection.BUY,
            confidence=SignalConfidence.HIGH,
            imbalance=imbalance,
            entry_price=450.0,
            target_exit_price=452.0,
            stop_loss_price=448.0,
            historical_win_rate=0.62,
            historical_count=150,
            max_position_pct=0.03,
            urgency="high",
        )
        
        position = ClosingAuctionPosition(
            symbol="SPY",
            entry_signal=signal,
            entry_time=datetime.now(),
            entry_price=450.0,
            shares=100,
            side='long',
        )
        
        exit_time = datetime.now()
        position.close_position(455.0, exit_time)
        
        assert position.status == OrderStatus.EXITED
        assert position.exit_price == 455.0
        assert position.exit_time == exit_time
        assert position.pnl == 500.0
    
    def test_position_to_dict(self):
        """Test position serialization."""
        imbalance = MOCImbalance(
            symbol="SPY",
            timestamp=datetime.now(),
            imbalance_shares=1000000,
            paired_shares=5000000,
            reference_price=450.0,
            source="test",
        )
        
        signal = ClosingAuctionSignal(
            symbol="SPY",
            timestamp=datetime.now(),
            direction=SignalDirection.BUY,
            confidence=SignalConfidence.HIGH,
            imbalance=imbalance,
            entry_price=450.0,
            target_exit_price=452.0,
            stop_loss_price=448.0,
            historical_win_rate=0.62,
            historical_count=150,
            max_position_pct=0.03,
            urgency="high",
        )
        
        position = ClosingAuctionPosition(
            symbol="SPY",
            entry_signal=signal,
            entry_time=datetime(2026, 5, 15, 15, 50, 0),
            entry_price=450.0,
            shares=100,
            side='long',
            order_id="TEST123",
        )
        
        d = position.to_dict()
        
        assert d['symbol'] == "SPY"
        assert d['entry_price'] == 450.0
        assert d['shares'] == 100
        assert d['side'] == 'long'
        assert d['order_id'] == "TEST123"
        assert d['status'] == 'pending'


class TestExecutionConfig:
    """Test the ExecutionConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ExecutionConfig()
        
        assert config.entry_window_start == time(15, 50)
        assert config.entry_window_end == time(15, 55)
        assert config.exit_time == time(16, 0)
        assert config.max_position_pct == 0.03
        assert config.min_position_pct == 0.01
        assert config.max_positions_per_day == 3
        assert config.dry_run is True
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = ExecutionConfig()
        d = config.to_dict()
        
        assert d['entry_window_start'] == '15:50:00'
        assert d['entry_window_end'] == '15:55:00'
        assert d['exit_time'] == '16:00:00'
        assert d['max_position_pct'] == 0.03
        assert d['dry_run'] is True


class TestClosingAuctionExecutor:
    """Test the ClosingAuctionExecutor class."""
    
    @pytest.fixture
    def executor(self, tmp_path):
        """Create executor with temporary state file."""
        state_file = tmp_path / "positions.json"
        config = ExecutionConfig(dry_run=True)
        return ClosingAuctionExecutor(
            config=config,
            portfolio_value=100000.0,
            state_file=state_file,
        )
    
    @pytest.fixture
    def buy_signal(self):
        """Create a sample buy signal."""
        imbalance = MOCImbalance(
            symbol="SPY",
            timestamp=datetime.now(),
            imbalance_shares=1000000,
            paired_shares=5000000,
            reference_price=450.0,
            source="test",
        )
        
        return ClosingAuctionSignal(
            symbol="SPY",
            timestamp=datetime.now(),
            direction=SignalDirection.BUY,
            confidence=SignalConfidence.HIGH,
            imbalance=imbalance,
            entry_price=450.0,
            target_exit_price=452.0,
            stop_loss_price=448.0,
            historical_win_rate=0.62,
            historical_count=150,
            max_position_pct=0.03,
            urgency="high",
        )
    
    def test_executor_initialization(self, executor):
        """Test executor initialization."""
        assert executor.config.dry_run is True
        assert executor.portfolio_value == 100000.0
        assert len(executor.positions) == 0
    
    def test_can_enter_position_approved(self, executor, buy_signal):
        """Test that valid signal can enter."""
        can_enter, reason = executor.can_enter_position(buy_signal)
        
        assert can_enter is True
        assert reason == "OK"
    
    def test_can_enter_position_already_in_position(self, executor, buy_signal):
        """Test rejection when already in position."""
        # Enter position first
        position = executor.enter_position(buy_signal, 450.0)
        assert position is not None
        
        # Try to enter again
        can_enter, reason = executor.can_enter_position(buy_signal)
        assert can_enter is False
        assert "Already in position" in reason
    
    def test_can_enter_position_neutral_signal(self, executor):
        """Test rejection for neutral signal."""
        imbalance = MOCImbalance(
            symbol="SPY",
            timestamp=datetime.now(),
            imbalance_shares=100000,
            paired_shares=5000000,
            reference_price=450.0,
            source="test",
        )
        
        neutral_signal = ClosingAuctionSignal(
            symbol="SPY",
            timestamp=datetime.now(),
            direction=SignalDirection.NEUTRAL,
            confidence=SignalConfidence.HIGH,
            imbalance=imbalance,
            entry_price=450.0,
            target_exit_price=450.0,
            stop_loss_price=None,
            historical_win_rate=None,
            historical_count=0,
            max_position_pct=0.0,
            urgency="none",
        )
        
        can_enter, reason = executor.can_enter_position(neutral_signal)
        assert can_enter is False
        assert "Neutral signal" in reason
    
    def test_calculate_position_size(self, executor, buy_signal):
        """Test position sizing calculation."""
        shares = executor.calculate_position_size(buy_signal, 450.0)
        
        # 3% of 100k = $3k, at $450 = ~6.67 shares
        # With HIGH confidence (1.0) and direction 2/3 = 0.67
        # 3% * 1.0 * 0.67 = 2%
        # 2% of 100k = $2k / $450 = ~4.44 shares
        # But minimum is 1%
        expected_min = int(100000 * 0.01 / 450.0)  # min_position_pct
        assert shares >= expected_min
    
    def test_enter_position_paper_trading(self, executor, buy_signal):
        """Test entering position in paper trading mode."""
        position = executor.enter_position(buy_signal, 450.0)
        
        assert position is not None
        assert position.symbol == "SPY"
        assert position.side == 'long'
        assert position.status == OrderStatus.FILLED
        assert position.shares > 0
        assert position.order_id.startswith("PAPER_")
    
    def test_enter_position_short(self, executor):
        """Test entering short position."""
        imbalance = MOCImbalance(
            symbol="SPY",
            timestamp=datetime.now(),
            imbalance_shares=-1000000,
            paired_shares=5000000,
            reference_price=450.0,
            source="test",
        )
        
        sell_signal = ClosingAuctionSignal(
            symbol="SPY",
            timestamp=datetime.now(),
            direction=SignalDirection.SELL,
            confidence=SignalConfidence.HIGH,
            imbalance=imbalance,
            entry_price=450.0,
            target_exit_price=448.0,
            stop_loss_price=452.0,
            historical_win_rate=0.58,
            historical_count=120,
            max_position_pct=0.03,
            urgency="high",
        )
        
        position = executor.enter_position(sell_signal, 450.0)
        
        assert position is not None
        assert position.side == 'short'
    
    def test_exit_position(self, executor, buy_signal):
        """Test exiting a position."""
        # Enter position
        position = executor.enter_position(buy_signal, 450.0)
        assert position is not None
        assert "SPY" in executor.positions
        
        # Exit position
        closed = executor.exit_position("SPY", 455.0)
        
        assert closed is not None
        assert closed.status == OrderStatus.EXITED
        assert closed.pnl > 0  # Profit since price went up
        assert "SPY" not in executor.positions
    
    def test_exit_position_not_found(self, executor):
        """Test exiting non-existent position."""
        closed = executor.exit_position("INVALID", 450.0)
        assert closed is None
    
    def test_exit_all_positions(self, executor, buy_signal):
        """Test exiting all positions at once."""
        # Enter position
        executor.enter_position(buy_signal, 450.0)
        assert len(executor.positions) == 1
        
        # Exit all
        closed = executor.exit_all_positions({"SPY": 455.0})
        
        assert len(closed) == 1
        assert len(executor.positions) == 0
    
    def test_daily_stats_tracking(self, executor, buy_signal):
        """Test daily statistics tracking."""
        # Initially empty
        stats = executor.get_daily_stats()
        assert stats['trades'] == 0
        
        # Make a winning trade
        executor.enter_position(buy_signal, 450.0)
        executor.exit_position("SPY", 455.0)  # $5 gain
        
        stats = executor.get_daily_stats()
        assert stats['trades'] == 1
        assert stats['winners'] == 1
        assert stats['losers'] == 0
        assert stats['win_rate'] == 1.0
        assert stats['total_pnl'] > 0
    
    def test_reset_daily_stats(self, executor, buy_signal):
        """Test resetting daily stats."""
        # Make a trade
        executor.enter_position(buy_signal, 450.0)
        executor.exit_position("SPY", 455.0)
        
        # Reset
        executor.reset_daily_stats()
        
        stats = executor.get_daily_stats()
        assert stats['trades'] == 0
        assert stats['winners'] == 0
        assert stats['total_pnl'] == 0.0


class TestClosingAuctionScheduler:
    """Test the ClosingAuctionScheduler class."""
    
    @pytest.fixture
    def scheduler(self, tmp_path):
        """Create scheduler with temporary state."""
        state_file = tmp_path / "positions.json"
        config = ExecutionConfig(dry_run=True)
        executor = ClosingAuctionExecutor(
            config=config,
            portfolio_value=100000.0,
            state_file=state_file,
        )
        return ClosingAuctionScheduler(executor)
    
    @pytest.fixture
    def buy_signal_spy(self):
        """Create a sample SPY buy signal."""
        imbalance = MOCImbalance(
            symbol="SPY",
            timestamp=datetime.now(),
            imbalance_shares=1000000,
            paired_shares=5000000,
            reference_price=450.0,
            source="test",
        )
        
        return ClosingAuctionSignal(
            symbol="SPY",
            timestamp=datetime.now(),
            direction=SignalDirection.BUY,
            confidence=SignalConfidence.HIGH,
            imbalance=imbalance,
            entry_price=450.0,
            target_exit_price=452.0,
            stop_loss_price=448.0,
            historical_win_rate=0.62,
            historical_count=150,
            max_position_pct=0.03,
            urgency="high",
        )
    
    @pytest.fixture
    def buy_signal_qqq(self):
        """Create a sample QQQ buy signal."""
        imbalance = MOCImbalance(
            symbol="QQQ",
            timestamp=datetime.now(),
            imbalance_shares=500000,
            paired_shares=2500000,
            reference_price=380.0,
            source="test",
        )
        
        return ClosingAuctionSignal(
            symbol="QQQ",
            timestamp=datetime.now(),
            direction=SignalDirection.STRONG_BUY,
            confidence=SignalConfidence.HIGH,
            imbalance=imbalance,
            entry_price=380.0,
            target_exit_price=383.0,
            stop_loss_price=377.0,
            historical_win_rate=0.65,
            historical_count=200,
            max_position_pct=0.03,
            urgency="immediate",
        )
    
    def test_is_entry_window_true(self, scheduler):
        """Test entry window detection within window."""
        # 3:52 PM
        current_time = datetime(2026, 5, 15, 15, 52, 0)
        assert scheduler.is_entry_window(current_time) is True
    
    def test_is_entry_window_false_before(self, scheduler):
        """Test entry window detection before window."""
        # 3:45 PM
        current_time = datetime(2026, 5, 15, 15, 45, 0)
        assert scheduler.is_entry_window(current_time) is False
    
    def test_is_entry_window_false_after(self, scheduler):
        """Test entry window detection after window."""
        # 4:00 PM
        current_time = datetime(2026, 5, 15, 16, 0, 0)
        assert scheduler.is_entry_window(current_time) is False
    
    def test_is_exit_time_true(self, scheduler):
        """Test exit time detection at 4pm."""
        # 4:00 PM
        current_time = datetime(2026, 5, 15, 16, 0, 0)
        assert scheduler.is_exit_time(current_time) is True
    
    def test_is_exit_time_true_after(self, scheduler):
        """Test exit time detection after 4pm."""
        # 4:05 PM
        current_time = datetime(2026, 5, 15, 16, 5, 0)
        assert scheduler.is_exit_time(current_time) is True
    
    def test_is_exit_time_false_before(self, scheduler):
        """Test exit time detection before 4pm."""
        # 3:59 PM
        current_time = datetime(2026, 5, 15, 15, 59, 0)
        assert scheduler.is_exit_time(current_time) is False
    
    def test_evaluate_entry_window_outside_window(self, scheduler, buy_signal_spy):
        """Test evaluation outside entry window returns empty."""
        # 10:00 AM - outside window
        current_time = datetime(2026, 5, 15, 10, 0, 0)
        current_prices = {"SPY": 450.0}
        
        entered = scheduler.evaluate_entry_window(
            [buy_signal_spy],
            current_prices,
            current_time,
        )
        
        assert len(entered) == 0
    
    def test_evaluate_entry_window_in_window(self, scheduler, buy_signal_spy):
        """Test evaluation inside entry window enters position."""
        # 3:52 PM - inside window
        current_time = datetime(2026, 5, 15, 15, 52, 0)
        current_prices = {"SPY": 450.0}
        
        entered = scheduler.evaluate_entry_window(
            [buy_signal_spy],
            current_prices,
            current_time,
        )
        
        assert len(entered) == 1
        assert entered[0].symbol == "SPY"
    
    def test_evaluate_entry_window_multiple_signals(self, scheduler, buy_signal_spy, buy_signal_qqq):
        """Test evaluation with multiple signals."""
        # 3:52 PM
        current_time = datetime(2026, 5, 15, 15, 52, 0)
        current_prices = {"SPY": 450.0, "QQQ": 380.0}
        
        entered = scheduler.evaluate_entry_window(
            [buy_signal_spy, buy_signal_qqq],
            current_prices,
            current_time,
        )
        
        # Should enter both if portfolio limits allow
        assert len(entered) >= 1
    
    def test_execute_market_close(self, scheduler, buy_signal_spy):
        """Test market close execution."""
        # First enter a position
        current_time = datetime(2026, 5, 15, 15, 52, 0)
        scheduler.executor.enter_position(buy_signal_spy, 450.0, current_time)
        
        # Then exit at close
        exit_time = datetime(2026, 5, 15, 16, 0, 0)
        closing_prices = {"SPY": 455.0}
        
        closed = scheduler.execute_market_close(closing_prices, exit_time)
        
        assert len(closed) == 1
        assert closed[0].status == OrderStatus.EXITED
        assert closed[0].pnl > 0


class TestIntegration:
    """Integration tests for the full execution pipeline."""
    
    def test_full_day_simulation(self, tmp_path):
        """Simulate a full trading day."""
        state_file = tmp_path / "positions.json"
        config = ExecutionConfig(
            dry_run=True,
            max_position_pct=0.03,
            max_positions_per_day=3,
        )
        executor = ClosingAuctionExecutor(
            config=config,
            portfolio_value=100000.0,
            state_file=state_file,
        )
        scheduler = ClosingAuctionScheduler(executor)
        
        # Create signals
        signals = []
        for symbol, imbalance_shares in [("SPY", 1000000), ("QQQ", 500000)]:
            imbalance = MOCImbalance(
                symbol=symbol,
                timestamp=datetime.now(),
                imbalance_shares=imbalance_shares,
                paired_shares=5000000,
                reference_price=450.0 if symbol == "SPY" else 380.0,
                source="test",
            )
            
            signal = ClosingAuctionSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                direction=SignalDirection.BUY if imbalance_shares > 0 else SignalDirection.SELL,
                confidence=SignalConfidence.HIGH,
                imbalance=imbalance,
                entry_price=450.0 if symbol == "SPY" else 380.0,
                target_exit_price=452.0 if symbol == "SPY" else 383.0,
                stop_loss_price=448.0 if symbol == "SPY" else 377.0,
                historical_win_rate=0.62,
                historical_count=150,
                max_position_pct=0.03,
                urgency="high",
            )
            signals.append(signal)
        
        # 3:52 PM - Entry window
        entry_time = datetime(2026, 5, 15, 15, 52, 0)
        current_prices = {"SPY": 450.0, "QQQ": 380.0}
        
        entered = scheduler.evaluate_entry_window(signals, current_prices, entry_time)
        assert len(entered) == 2
        
        # Verify active positions
        active = executor.get_active_positions()
        assert len(active) == 2
        
        # 4:00 PM - Market close
        exit_time = datetime(2026, 5, 15, 16, 0, 0)
        closing_prices = {"SPY": 452.0, "QQQ": 382.0}  # Both went up
        
        closed = scheduler.execute_market_close(closing_prices, exit_time)
        assert len(closed) == 2
        
        # Verify stats
        stats = executor.get_daily_stats()
        assert stats['trades'] == 2
        assert stats['winners'] == 2
        assert stats['win_rate'] == 1.0
        assert stats['total_pnl'] > 0
    
    def test_state_persistence(self, tmp_path):
        """Test that state is saved and can be loaded."""
        state_file = tmp_path / "positions.json"
        config = ExecutionConfig(dry_run=True)
        
        # Create executor and enter position
        executor1 = ClosingAuctionExecutor(
            config=config,
            portfolio_value=100000.0,
            state_file=state_file,
        )
        
        imbalance = MOCImbalance(
            symbol="SPY",
            timestamp=datetime.now(),
            imbalance_shares=1000000,
            paired_shares=5000000,
            reference_price=450.0,
            source="test",
        )
        
        signal = ClosingAuctionSignal(
            symbol="SPY",
            timestamp=datetime.now(),
            direction=SignalDirection.BUY,
            confidence=SignalConfidence.HIGH,
            imbalance=imbalance,
            entry_price=450.0,
            target_exit_price=452.0,
            stop_loss_price=448.0,
            historical_win_rate=0.62,
            historical_count=150,
            max_position_pct=0.03,
            urgency="high",
        )
        
        executor1.enter_position(signal, 450.0)
        
        # Verify state file exists
        assert state_file.exists()
        
        # Read and verify state content
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        assert 'active_positions' in state
        assert 'SPY' in state['active_positions']
        assert state['config']['dry_run'] is True
