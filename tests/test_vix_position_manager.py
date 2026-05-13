#!/usr/bin/env python3
"""
Tests for VIX Position Manager — enums, data classes, position lifecycle
(open/mark-to-market/roll-check/close), budget tracking, and performance stats.
"""
import sys
import os
import json
import sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.options.vix_position_manager import (
    PositionStatus, VIXInsurancePosition,
    VIXPositionManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(tmp_path, budget=1000):
    """Create a VIXPositionManager with tmp DB and positions path."""
    manager = VIXPositionManager.__new__(VIXPositionManager)
    manager.annual_budget = budget
    manager.DB_PATH = tmp_path / "vix_options.db"
    manager.POSITIONS_PATH = tmp_path / "positions" / "vix_insurance.json"
    manager.POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    manager._init_db()
    return manager


def _make_signal(**overrides):
    """Create a VIX insurance signal dict."""
    signal = {
        'timestamp': datetime.now().isoformat(),
        'vix_spot': 18.0,
        'selected_strike': 22.0,
        'selected_expiration': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
        'allocation_dollars': 500,
        'premium_cost': 2.50,
        'delta': 0.25,
        'days_to_expiration': 30,
    }
    signal.update(overrides)
    return signal


def _open_position(manager, **signal_overrides):
    """Helper to open a position and return its ID."""
    signal = _make_signal(**signal_overrides)
    return manager.open_position(signal)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestPositionStatus:
    def test_values(self):
        assert PositionStatus.OPEN.value == "open"
        assert PositionStatus.CLOSED_PROFIT.value == "closed_profit"
        assert PositionStatus.CLOSED_EXPIRE.value == "closed_expired"
        assert PositionStatus.CLOSED_STOP.value == "closed_stop"
        assert PositionStatus.ROLL_PENDING.value == "roll_pending"

    def test_members(self):
        assert len(PositionStatus) == 5


# ---------------------------------------------------------------------------
# VIXInsurancePosition dataclass tests
# ---------------------------------------------------------------------------

class TestVIXInsurancePosition:
    def test_creation(self):
        p = VIXInsurancePosition(
            id=1, status='open',
            entry_date='2026-01-01', entry_vix_spot=18.0,
            strike=22.0, expiration_date='2026-02-01',
            contracts=2, premium_paid_per_contract=2.50,
            total_cost=500.0, delta_at_entry=0.25,
            days_to_expiration_at_entry=30,
            current_mark_price=None, current_value=None,
            unrealized_pnl=None, unrealized_pnl_percent=None,
            exit_date=None, exit_vix_spot=None, exit_price=None,
            realized_pnl=None, realized_pnl_percent=None, exit_reason=None,
            days_held=0, roll_count=0, budget_impact=500.0,
        )
        assert p.id == 1
        assert p.strike == 22.0

    def test_to_dict(self):
        p = VIXInsurancePosition(
            id=1, status='open',
            entry_date='2026-01-01', entry_vix_spot=18.0,
            strike=22.0, expiration_date='2026-02-01',
            contracts=2, premium_paid_per_contract=2.50,
            total_cost=500.0, delta_at_entry=0.25,
            days_to_expiration_at_entry=30,
            current_mark_price=None, current_value=None,
            unrealized_pnl=None, unrealized_pnl_percent=None,
            exit_date=None, exit_vix_spot=None, exit_price=None,
            realized_pnl=None, realized_pnl_percent=None, exit_reason=None,
            days_held=0, roll_count=0, budget_impact=500.0,
        )
        d = p.to_dict()
        assert d['id'] == 1
        assert 'strike' in d


# ---------------------------------------------------------------------------
# VIXPositionManager tests
# ---------------------------------------------------------------------------

class TestVIXPositionManager:
    def test_init_creates_db(self, tmp_path):
        manager = _make_manager(tmp_path)
        assert manager.DB_PATH.exists()

    def test_init_creates_tables(self, tmp_path):
        manager = _make_manager(tmp_path)
        conn = sqlite3.connect(str(manager.DB_PATH))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert 'vix_positions' in tables
        assert 'position_history' in tables

    def test_init_creates_positions_dir(self, tmp_path):
        manager = _make_manager(tmp_path)
        assert manager.POSITIONS_PATH.parent.exists()

    # open_position
    def test_open_position_returns_id(self, tmp_path):
        manager = _make_manager(tmp_path)
        pid = _open_position(manager)
        assert pid is not None
        assert pid >= 1

    def test_open_position_stores_in_db(self, tmp_path):
        manager = _make_manager(tmp_path)
        pid = _open_position(manager)
        positions = manager.get_open_positions()
        assert len(positions) == 1
        assert positions[0]['id'] == pid

    def test_open_position_zero_allocation_returns_none(self, tmp_path):
        manager = _make_manager(tmp_path)
        signal = _make_signal(allocation_dollars=0)
        assert manager.open_position(signal) is None

    def test_open_position_zero_premium_returns_none(self, tmp_path):
        manager = _make_manager(tmp_path)
        signal = _make_signal(premium_cost=0)
        assert manager.open_position(signal) is None

    def test_open_position_insufficient_allocation_returns_none(self, tmp_path):
        manager = _make_manager(tmp_path)
        signal = _make_signal(allocation_dollars=10, premium_cost=100)
        assert manager.open_position(signal) is None

    def test_open_position_creates_history_event(self, tmp_path):
        manager = _make_manager(tmp_path)
        _open_position(manager)
        conn = sqlite3.connect(str(manager.DB_PATH))
        count = conn.execute("SELECT COUNT(*) FROM position_history").fetchone()[0]
        conn.close()
        assert count >= 1

    def test_open_multiple_positions(self, tmp_path):
        manager = _make_manager(tmp_path)
        pid1 = _open_position(manager)
        pid2 = _open_position(manager, strike=25.0)
        assert pid1 != pid2
        assert len(manager.get_open_positions()) == 2

    # get_open_positions
    def test_get_open_positions_empty(self, tmp_path):
        manager = _make_manager(tmp_path)
        assert manager.get_open_positions() == []

    def test_get_open_positions_excludes_closed(self, tmp_path):
        manager = _make_manager(tmp_path)
        pid = _open_position(manager)
        manager.close_position(pid, exit_price=3.0, current_vix=20.0, reason='profit_take')
        assert len(manager.get_open_positions()) == 0

    # mark_to_market
    def test_mark_to_market_returns_dict(self, tmp_path):
        manager = _make_manager(tmp_path)
        pid = _open_position(manager)
        result = manager.mark_to_market(pid, current_vix=20.0, option_chain=[])
        assert 'position_id' in result
        assert 'mark_price' in result

    def test_mark_to_market_intrinsic_value(self, tmp_path):
        manager = _make_manager(tmp_path)
        pid = _open_position(manager, selected_strike=20.0, premium_cost=2.0)
        # VIX at 25, strike 20 → intrinsic = 5
        result = manager.mark_to_market(pid, current_vix=25.0, option_chain=[])
        assert result['mark_price'] >= 5.0

    def test_mark_to_market_with_option_chain(self, tmp_path):
        manager = _make_manager(tmp_path)
        signal = _make_signal(strike=22.0, selected_strike=22.0,
                              selected_expiration='2026-06-20')
        pid = manager.open_position(signal)
        chain = [{'strike': 22.0, 'expiration_date': '2026-06-20',
                  'bid': 3.0, 'ask': 4.0}]
        result = manager.mark_to_market(pid, current_vix=20.0, option_chain=chain)
        assert result['mark_price'] == 3.5  # mid of bid/ask

    def test_mark_to_market_nonexistent_position(self, tmp_path):
        manager = _make_manager(tmp_path)
        result = manager.mark_to_market(999, current_vix=20.0, option_chain=[])
        assert result == {}

    # check_roll_needed
    def test_check_roll_needed_no_position(self, tmp_path):
        manager = _make_manager(tmp_path)
        needs_roll, reason = manager.check_roll_needed(999)
        assert needs_roll is False
        assert 'No open position' in reason

    def test_check_roll_needed_far_expiry(self, tmp_path):
        manager = _make_manager(tmp_path)
        pid = _open_position(manager,
                             selected_expiration=(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'))
        needs_roll, reason = manager.check_roll_needed(pid)
        assert needs_roll is False
        assert 'days to expiration' in reason

    def test_check_roll_needed_near_expiry(self, tmp_path):
        manager = _make_manager(tmp_path)
        pid = _open_position(manager,
                             selected_expiration=(datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'))
        needs_roll, reason = manager.check_roll_needed(pid)
        assert needs_roll is True
        assert 'Expiration' in reason

    # close_position
    def test_close_position_returns_dict(self, tmp_path):
        manager = _make_manager(tmp_path)
        pid = _open_position(manager, allocation_dollars=500, premium_cost=2.50)
        result = manager.close_position(pid, exit_price=3.0, current_vix=20.0, reason='profit_take')
        assert 'realized_pnl' in result
        assert result['status'] == PositionStatus.CLOSED_PROFIT.value

    def test_close_position_profit(self, tmp_path):
        manager = _make_manager(tmp_path)
        pid = _open_position(manager, allocation_dollars=500, premium_cost=2.50)
        # Exit at higher price → profit
        result = manager.close_position(pid, exit_price=5.0, current_vix=25.0, reason='profit_take')
        assert result['realized_pnl'] > 0

    def test_close_position_loss(self, tmp_path):
        manager = _make_manager(tmp_path)
        pid = _open_position(manager, allocation_dollars=500, premium_cost=2.50)
        # Exit at near-zero price → loss (exit_value < total_cost)
        result = manager.close_position(pid, exit_price=0.01, current_vix=15.0, reason='stop')
        assert result['realized_pnl'] < 0
        assert result['status'] == PositionStatus.CLOSED_STOP.value

    def test_close_position_expire(self, tmp_path):
        manager = _make_manager(tmp_path)
        pid = _open_position(manager, allocation_dollars=500, premium_cost=2.50)
        result = manager.close_position(pid, exit_price=0.0, current_vix=15.0, reason='expire')
        assert result['status'] == PositionStatus.CLOSED_EXPIRE.value

    def test_close_position_nonexistent(self, tmp_path):
        manager = _make_manager(tmp_path)
        result = manager.close_position(999, exit_price=3.0, current_vix=20.0, reason='test')
        assert 'error' in result

    def test_close_position_already_closed(self, tmp_path):
        manager = _make_manager(tmp_path)
        pid = _open_position(manager)
        manager.close_position(pid, exit_price=3.0, current_vix=20.0, reason='profit_take')
        result = manager.close_position(pid, exit_price=3.0, current_vix=20.0, reason='test')
        assert 'error' in result

    # get_budget_status
    def test_budget_status_no_positions(self, tmp_path):
        manager = _make_manager(tmp_path, budget=1000)
        status = manager.get_budget_status()
        assert status['annual_budget'] == 1000
        assert status['spent_ytd'] == 0

    def test_budget_status_after_open(self, tmp_path):
        manager = _make_manager(tmp_path, budget=1000)
        _open_position(manager, allocation_dollars=500, premium_cost=2.50)
        status = manager.get_budget_status()
        assert status['spent_ytd'] > 0

    def test_budget_status_has_all_fields(self, tmp_path):
        manager = _make_manager(tmp_path)
        status = manager.get_budget_status()
        assert 'annual_budget' in status
        assert 'spent_ytd' in status
        assert 'realized_pnl_ytd' in status
        assert 'remaining_budget' in status
        assert 'budget_utilization_percent' in status

    # get_performance_stats
    def test_performance_stats_no_closed(self, tmp_path):
        manager = _make_manager(tmp_path)
        stats = manager.get_performance_stats()
        assert 'message' in stats

    def test_performance_stats_after_close(self, tmp_path):
        manager = _make_manager(tmp_path)
        pid = _open_position(manager, allocation_dollars=500, premium_cost=2.50)
        manager.close_position(pid, exit_price=3.0, current_vix=20.0, reason='profit_take')
        stats = manager.get_performance_stats()
        assert stats['total_trades'] == 1
        assert stats['winning_trades'] == 1

    def test_performance_stats_win_loss(self, tmp_path):
        manager = _make_manager(tmp_path)
        # Win
        pid1 = _open_position(manager, allocation_dollars=500, premium_cost=2.50)
        manager.close_position(pid1, exit_price=5.0, current_vix=25.0, reason='profit_take')
        # Loss
        pid2 = _open_position(manager, allocation_dollars=500, premium_cost=2.50)
        manager.close_position(pid2, exit_price=0.01, current_vix=15.0, reason='stop')
        stats = manager.get_performance_stats()
        assert stats['total_trades'] == 2
        assert stats['winning_trades'] == 1
        assert stats['losing_trades'] == 1

    def test_performance_stats_has_all_fields(self, tmp_path):
        manager = _make_manager(tmp_path)
        pid = _open_position(manager)
        manager.close_position(pid, exit_price=3.0, current_vix=20.0, reason='profit_take')
        stats = manager.get_performance_stats()
        assert 'win_rate' in stats
        assert 'total_pnl' in stats
        assert 'profit_factor' in stats

    # export_positions
    def test_export_positions_creates_file(self, tmp_path):
        manager = _make_manager(tmp_path)
        _open_position(manager)
        manager.export_positions()
        assert manager.POSITIONS_PATH.exists()

    def test_export_positions_valid_json(self, tmp_path):
        manager = _make_manager(tmp_path)
        _open_position(manager)
        manager.export_positions()
        with open(manager.POSITIONS_PATH) as f:
            data = json.load(f)
        assert 'open_positions' in data
        assert 'closed_positions' in data
        assert 'budget_status' in data

    # Roll parameters
    def test_roll_days_before_expiry(self):
        assert VIXPositionManager.ROLL_DAYS_BEFORE_EXPIRY == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
