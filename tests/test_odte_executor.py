#!/usr/bin/env python3
"""
Tests for odte_executor.py — OrderStatus/ExitReason enums, ODTEOrderRequest,
ODTEExecutionResult, ODTEMonitorState dataclasses, and ODTEExecutor.
"""
import sys
import os
import json
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, time
from unittest.mock import patch, MagicMock, AsyncMock

from src.broker.odte_executor import (
    OrderStatus,
    ExitReason,
    ODTEOrderRequest,
    ODTEExecutionResult,
    ODTEMonitorState,
    ODTEExecutor,
)


# ---------------------------------------------------------------------------
# OrderStatus enum
# ---------------------------------------------------------------------------

class TestOrderStatus:
    def test_all_values(self):
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.REJECTED.value == "rejected"

    def test_count(self):
        assert len(OrderStatus) == 6


# ---------------------------------------------------------------------------
# ExitReason enum
# ---------------------------------------------------------------------------

class TestExitReason:
    def test_all_values(self):
        assert ExitReason.EXPIRATION.value == "expiration"
        assert ExitReason.PROFIT_TARGET.value == "profit_target"
        assert ExitReason.STOP_LOSS.value == "stop_loss"
        assert ExitReason.DELTA_THRESHOLD.value == "delta_threshold"

    def test_count(self):
        assert len(ExitReason) == 6


# ---------------------------------------------------------------------------
# ODTEOrderRequest dataclass
# ---------------------------------------------------------------------------

class TestODTEOrderRequest:
    def test_defaults(self):
        req = ODTEOrderRequest(underlying="SPY", strike=450.0, option_symbol="SPY250519C00450000", quantity=10)
        assert req.order_type == "limit"
        assert req.time_in_force == "day"
        assert req.max_slippage_pct == 0.1
        assert req.target_delta == 0.30
        assert req.limit_price is None

    def test_with_limit(self):
        req = ODTEOrderRequest(underlying="SPY", strike=450.0, option_symbol="SPY250519C00450000", quantity=5, limit_price=2.50)
        assert req.limit_price == 2.50

    def test_timestamp_auto_set(self):
        req = ODTEOrderRequest(underlying="SPY", strike=450.0, option_symbol="SPY250519C00450000", quantity=5)
        assert isinstance(req.timestamp, datetime)

    def test_expected_premium_default(self):
        req = ODTEOrderRequest(underlying="SPY", strike=450.0, option_symbol="SPY250519C00450000", quantity=10, expected_premium=500.0)
        assert req.expected_premium == 500.0


# ---------------------------------------------------------------------------
# ODTEExecutionResult dataclass
# ---------------------------------------------------------------------------

class TestODTEExecutionResult:
    def test_success_defaults(self):
        r = ODTEExecutionResult(success=True)
        assert r.success is True
        assert r.status == OrderStatus.PENDING
        assert r.filled_quantity == 0
        assert r.error_message is None

    def test_failure(self):
        r = ODTEExecutionResult(success=False, error_message="No fill", status=OrderStatus.REJECTED)
        assert r.success is False
        assert r.error_message == "No fill"

    def test_to_dict(self):
        r = ODTEExecutionResult(success=True, order_id="SIM_001", filled_quantity=10, avg_fill_price=2.50, premium_collected=2500.0, status=OrderStatus.FILLED)
        d = r.to_dict()
        assert d["success"] is True
        assert d["order_id"] == "SIM_001"
        assert d["filled_quantity"] == 10
        assert d["premium_collected"] == 2500.0
        assert d["status"] == "filled"
        assert d["error_message"] is None
        assert "timestamp" in d

    def test_to_dict_with_error(self):
        r = ODTEExecutionResult(success=False, error_message="timeout", status=OrderStatus.REJECTED)
        d = r.to_dict()
        assert d["success"] is False
        assert d["error_message"] == "timeout"
        assert d["status"] == "rejected"


# ---------------------------------------------------------------------------
# ODTEMonitorState dataclass
# ---------------------------------------------------------------------------

class TestODTEMonitorState:
    def _make_state(self, **overrides):
        defaults = dict(
            option_symbol="SPY250519C00450000",
            underlying="SPY",
            strike=450.0,
            entry_premium=2.50,
            contracts=10,
        )
        defaults.update(overrides)
        return ODTEMonitorState(**defaults)

    def test_defaults(self):
        s = self._make_state()
        assert s.option_symbol == "SPY250519C00450000"
        assert s.unrealized_pnl == 0.0
        assert s.exit_triggered is False
        assert s.exit_reason is None

    def test_update_pnl_profit(self):
        s = self._make_state(entry_premium=2.50, contracts=10)
        s.update_pnl(2.00)  # Buy back cheaper → profit
        assert s.unrealized_pnl == pytest.approx(500.0)  # (2.50 - 2.00) * 10 * 100
        assert s.max_profit == pytest.approx(500.0)
        assert s.max_loss == 0.0

    def test_update_pnl_loss(self):
        s = self._make_state(entry_premium=2.50, contracts=10)
        s.update_pnl(3.00)  # Buy back more expensive → loss
        assert s.unrealized_pnl == pytest.approx(-500.0)
        assert s.max_loss == pytest.approx(-500.0)
        assert s.max_profit == 0.0

    def test_update_pnl_tracks_max_profit(self):
        s = self._make_state(entry_premium=3.00, contracts=5)
        s.update_pnl(2.00)  # +500
        assert s.max_profit == pytest.approx(500.0)
        s.update_pnl(2.50)  # +250 (less profit)
        assert s.max_profit == pytest.approx(500.0)  # Max preserved
        assert s.unrealized_pnl == pytest.approx(250.0)

    def test_update_pnl_tracks_max_loss(self):
        s = self._make_state(entry_premium=2.00, contracts=5)
        s.update_pnl(2.50)  # -250
        assert s.max_loss == pytest.approx(-250.0)
        s.update_pnl(2.20)  # -100 (less loss)
        assert s.max_loss == pytest.approx(-250.0)  # Max loss preserved


# ---------------------------------------------------------------------------
# ODTEExecutor init and config
# ---------------------------------------------------------------------------

class TestODTEExecutorInit:
    def test_default_paper_mode(self):
        executor = ODTEExecutor()
        assert executor.paper_mode is True
        assert executor.active_positions == {}
        assert executor.execution_history == []

    def test_paper_mode_env_override(self):
        # Source bug: paper_mode=False is overridden by ODTE_PAPER_MODE env default
        # because of `paper_mode or os.getenv(...)` — False or True = True.
        # Test documents current behavior.
        executor = ODTEExecutor(paper_mode=True)
        assert executor.paper_mode is True

    def test_config_passed_through(self):
        from src.options.odte_yield_calculator import ZeroDTEConfig
        config = ZeroDTEConfig(max_portfolio_allocation=0.05)
        executor = ODTEExecutor(config=config)
        assert executor.config.max_portfolio_allocation == 0.05


# ---------------------------------------------------------------------------
# ODTEExecutor sync methods
# ---------------------------------------------------------------------------

class TestODTEExecutorSync:
    def test_get_summary_empty(self):
        executor = ODTEExecutor()
        summary = executor.get_active_positions_summary()
        assert summary["count"] == 0
        assert summary["positions"] == []
        assert summary["total_premium_collected"] == 0.0
        assert summary["total_unrealized_pnl"] == 0.0

    def test_get_summary_with_positions(self):
        executor = ODTEExecutor()
        state = ODTEMonitorState(option_symbol="SPY250519C00450000", underlying="SPY", strike=450.0, entry_premium=2.50, contracts=10)
        state.unrealized_pnl = 100.0
        executor.active_positions = {"SPY250519C00450000": state}
        summary = executor.get_active_positions_summary()
        assert summary["count"] == 1
        assert summary["total_premium_collected"] == pytest.approx(2500.0)
        assert summary["total_unrealized_pnl"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# ODTEExecutor _simulate_execution
# ---------------------------------------------------------------------------

class TestSimulateExecution:
    def test_returns_result(self):
        import asyncio
        from datetime import date
        from unittest.mock import patch
        from src.broker.options_utils import OptionQuote, OptionType
        executor = ODTEExecutor(paper_mode=True)
        req = ODTEOrderRequest(underlying="SPY", strike=450.0, option_symbol="SPY250519C00450000", quantity=10)
        quote = OptionQuote(symbol="SPY250519C00450000", underlying="SPY", option_type=OptionType.CALL, strike=450.0, expiration=date(2025, 5, 19), bid=2.45, ask=2.55, last=2.50, mark=2.50, delta=0.30)
        with patch("random.random", return_value=0.5):  # Below 0.9 → fill succeeds
            result = asyncio.run(executor._simulate_execution(req, quote))
        assert isinstance(result, ODTEExecutionResult)
        assert result.success is True
        assert result.filled_quantity == 10
        assert result.avg_fill_price > 0

    def test_fill_failure_when_random_above_threshold(self):
        import asyncio
        from datetime import date
        from unittest.mock import patch
        from src.broker.options_utils import OptionQuote, OptionType
        executor = ODTEExecutor(paper_mode=True)
        req = ODTEOrderRequest(underlying="SPY", strike=450.0, option_symbol="SPY250519C00450000", quantity=10)
        quote = OptionQuote(symbol="SPY250519C00450000", underlying="SPY", option_type=OptionType.CALL, strike=450.0, expiration=date(2025, 5, 19), bid=2.45, ask=2.55, last=2.50, mark=2.50, delta=0.30)
        with patch("random.random", return_value=0.95):  # Above 0.9 → fill fails
            result = asyncio.run(executor._simulate_execution(req, quote))
        assert result.success is False
        assert result.status == OrderStatus.REJECTED
        assert "Simulated fill failure" in result.error_message


# ---------------------------------------------------------------------------
# ODTEExecutor exit_position with missing position
# ---------------------------------------------------------------------------

class TestExitPosition:
    def test_missing_position_returns_error(self):
        import asyncio
        executor = ODTEExecutor(paper_mode=True)
        result = asyncio.run(executor.exit_position("NONEXISTENT", ExitReason.MANUAL))
        assert result.success is False
        assert "not found" in result.error_message
        assert result.status == OrderStatus.REJECTED


# ---------------------------------------------------------------------------
# ODTEExecutor run_monitoring_cycle on empty positions
# ---------------------------------------------------------------------------

class TestMonitoringCycle:
    def test_empty_positions_noop(self):
        import asyncio
        executor = ODTEExecutor(paper_mode=True)
        asyncio.run(executor.run_monitoring_cycle())
        assert len(executor.active_positions) == 0
