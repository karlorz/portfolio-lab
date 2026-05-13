#!/usr/bin/env python3
"""
Tests for evaluator.py — constants, Position/Portfolio classes, order generation,
order execution, risk limits, performance calculation, and graduation criteria.
"""
import sys
import os
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.strategy.evaluator import (
    PAPER_CONFIG,
    BASE_ALLOCATION,
    REGIME_OVERRIDES,
    Position,
    Portfolio,
    calculate_performance,
    check_graduation_criteria,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_portfolio(tmp_path, cash=100000, positions=None):
    """Create a Portfolio with a temp state file."""
    state_file = tmp_path / "portfolio.json"
    portfolio = Portfolio(state_file, mode="paper")
    portfolio.cash = cash
    if positions:
        portfolio.positions = positions
    return portfolio


def _make_position(**overrides):
    defaults = dict(
        symbol="SPY",
        shares=100,
        avg_price=450.0,
        current_price=460.0,
        value=46000,
        weight=0.46,
        unrealized_pnl=1000,
    )
    defaults.update(overrides)
    return Position(**defaults)


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    def test_paper_config(self):
        assert PAPER_CONFIG["initial_capital"] == 100000
        assert PAPER_CONFIG["max_position_pct"] == 0.4
        assert PAPER_CONFIG["max_drawdown_pct"] == 0.15
        assert PAPER_CONFIG["rebalance_threshold"] == 0.10
        assert PAPER_CONFIG["volatility_target"] == 0.12

    def test_base_allocation(self):
        assert BASE_ALLOCATION["SPY"] == 0.46
        assert BASE_ALLOCATION["GLD"] == 0.38
        assert BASE_ALLOCATION["TLT"] == 0.16
        assert sum(BASE_ALLOCATION.values()) == pytest.approx(1.0)

    def test_regime_overrides(self):
        assert "crisis" in REGIME_OVERRIDES
        assert "vol_spike" in REGIME_OVERRIDES
        assert "low_vol" in REGIME_OVERRIDES
        for regime, alloc in REGIME_OVERRIDES.items():
            assert abs(sum(alloc.values()) - 1.0) < 0.01, f"{regime} doesn't sum to 1"


# ---------------------------------------------------------------------------
# Position Tests
# ---------------------------------------------------------------------------

class TestPosition:

    def test_named_tuple(self):
        p = _make_position(symbol="GLD", shares=200)
        assert p.symbol == "GLD"
        assert p.shares == 200

    def test_fields(self):
        p = _make_position()
        assert p.avg_price == 450.0
        assert p.current_price == 460.0
        assert p.unrealized_pnl == 1000


# ---------------------------------------------------------------------------
# Portfolio — init and state
# ---------------------------------------------------------------------------

class TestPortfolioState:

    def test_new_portfolio(self, tmp_path):
        p = _make_portfolio(tmp_path)
        assert p.cash == 100000
        assert p.positions == {}
        assert p.mode == "paper"

    def test_save_and_load(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=95000)
        p.positions = {"SPY": _make_position()}
        p.save_state()

        p2 = Portfolio(p.state_file, mode="paper")
        assert p2.cash == 95000
        assert "SPY" in p2.positions

    def test_save_preserves_mode(self, tmp_path):
        p = _make_portfolio(tmp_path)
        p.save_state()
        with open(p.state_file) as f:
            state = json.load(f)
        assert state["mode"] == "paper"


# ---------------------------------------------------------------------------
# Portfolio — total_value
# ---------------------------------------------------------------------------

class TestTotalValue:

    def test_cash_only(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        assert p.total_value({}) == 100000

    def test_with_positions(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=50000)
        p.positions = {"SPY": _make_position(shares=100, current_price=460)}
        prices = {"SPY": 470}
        # 50000 cash + 100 * 470 = 97000
        assert p.total_value(prices) == 97000

    def test_missing_price_uses_current(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=50000)
        p.positions = {"SPY": _make_position(shares=100, current_price=460)}
        # No SPY in prices → uses current_price
        assert p.total_value({}) == 96000


# ---------------------------------------------------------------------------
# Portfolio — current_weights
# ---------------------------------------------------------------------------

class TestCurrentWeights:

    def test_empty_positions(self, tmp_path):
        p = _make_portfolio(tmp_path)
        assert p.current_weights({}) == {}

    def test_weights(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {
            "SPY": _make_position(shares=100, current_price=460),
            "GLD": _make_position(symbol="GLD", shares=200, current_price=190),
        }
        prices = {"SPY": 460, "GLD": 190}
        weights = p.current_weights(prices)
        total = 100 * 460 + 200 * 190
        assert weights["SPY"] == pytest.approx(46000 / total)
        assert weights["GLD"] == pytest.approx(38000 / total)


# ---------------------------------------------------------------------------
# Portfolio — calculate_orders
# ---------------------------------------------------------------------------

class TestCalculateOrders:

    def test_no_drift_no_orders(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {
            "SPY": _make_position(shares=100, value=46000),
            "GLD": _make_position(symbol="GLD", shares=200, value=38000),
            "TLT": _make_position(symbol="TLT", shares=100, value=16000),
        }
        prices = {"SPY": 460, "GLD": 190, "TLT": 160}
        # Current weights match base allocation
        orders = p.calculate_orders(BASE_ALLOCATION, prices)
        assert orders == []

    def test_drift_generates_orders(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {
            "SPY": _make_position(shares=150, value=69000),  # Overweight
            "GLD": _make_position(symbol="GLD", shares=100, value=19000),  # Underweight
            "TLT": _make_position(symbol="TLT", shares=100, value=16000),
        }
        prices = {"SPY": 460, "GLD": 190, "TLT": 160}
        orders = p.calculate_orders(BASE_ALLOCATION, prices)
        assert len(orders) > 0

    def test_order_structure(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {
            "SPY": _make_position(shares=150, value=69000),
            "GLD": _make_position(symbol="GLD", shares=100, value=19000),
            "TLT": _make_position(symbol="TLT", shares=100, value=16000),
        }
        prices = {"SPY": 460, "GLD": 190, "TLT": 160}
        orders = p.calculate_orders(BASE_ALLOCATION, prices)
        for o in orders:
            assert "symbol" in o
            assert "side" in o
            assert "shares" in o
            assert o["side"] in ("buy", "sell")

    def test_skips_invalid_prices(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=50000)
        target = {"SPY": 0.50, "GLD": 0.50}
        orders = p.calculate_orders(target, {"SPY": 0, "GLD": -1})
        assert orders == []


# ---------------------------------------------------------------------------
# Portfolio — execute_orders
# ---------------------------------------------------------------------------

class TestExecuteOrders:

    def test_buy_order(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        orders = [{"symbol": "SPY", "side": "buy", "shares": 10, "estimated_price": 460}]
        prices = {"SPY": 460}
        executed = p.execute_orders(orders, prices, slippage=0.0)
        assert len(executed) == 1
        assert "SPY" in p.positions
        assert p.cash < 100000

    def test_sell_order(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=50000)
        p.positions = {"SPY": _make_position(shares=100, avg_price=450)}
        orders = [{"symbol": "SPY", "side": "sell", "shares": 50, "estimated_price": 460}]
        prices = {"SPY": 460}
        executed = p.execute_orders(orders, prices, slippage=0.0)
        assert len(executed) == 1
        assert p.positions["SPY"].shares == 50

    def test_slippage_applied(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        orders = [{"symbol": "SPY", "side": "buy", "shares": 10, "estimated_price": 460}]
        prices = {"SPY": 460}
        executed = p.execute_orders(orders, prices, slippage=0.01)
        # Buy fills at 460 * 1.01 = 464.6
        assert executed[0]["fill_price"] == pytest.approx(464.6)

    def test_partial_fill_on_insufficient_cash(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=1000)
        orders = [{"symbol": "SPY", "side": "buy", "shares": 100, "estimated_price": 460}]
        prices = {"SPY": 460}
        executed = p.execute_orders(orders, prices, slippage=0.0)
        # Can only buy 1000/460 ≈ 2.17 shares
        assert executed[0]["fill_shares"] < 100
        assert p.cash == pytest.approx(0, abs=1)

    def test_sell_full_position_removes(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=50000)
        p.positions = {"SPY": _make_position(shares=100, avg_price=450)}
        orders = [{"symbol": "SPY", "side": "sell", "shares": 100, "estimated_price": 460}]
        prices = {"SPY": 460}
        p.execute_orders(orders, prices, slippage=0.0)
        assert "SPY" not in p.positions


# ---------------------------------------------------------------------------
# Portfolio — check_risk_limits
# ---------------------------------------------------------------------------

class TestRiskLimits:

    def test_no_breach(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        p.positions = {"SPY": _make_position(weight=0.30)}
        assert p.check_risk_limits({"SPY": 460}) is None

    def test_concentration_breach(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {"SPY": _make_position(weight=0.50, value=50000)}
        result = p.check_risk_limits({"SPY": 460})
        assert result is not None
        assert "max_position" in result

    def test_drawdown_breach(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=50000)
        # Create history showing a peak of 100k then drop to 80k
        p.history = [{"total_value": 100000}] * 25
        p.positions = {}
        result = p.check_risk_limits({})
        # Current value = 50000, peak = 100000, DD = 50% > 15%
        assert result is not None
        assert "max_drawdown" in result


# ---------------------------------------------------------------------------
# calculate_performance
# ---------------------------------------------------------------------------

class TestCalculatePerformance:

    def test_structure(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        perf = calculate_performance(p, {})
        assert "timestamp" in perf
        assert "total_value" in perf
        assert "daily_return" in perf

    def test_first_day_zero_return(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        perf = calculate_performance(p, {})
        assert perf["daily_return"] == 0

    def test_positive_return(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        p.history = [{"total_value": 95000}]
        perf = calculate_performance(p, {})
        expected = (100000 - 95000) / 95000
        assert perf["daily_return"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# check_graduation_criteria
# ---------------------------------------------------------------------------

class TestGraduationCriteria:

    def test_too_few_days(self, tmp_path, capsys):
        p = _make_portfolio(tmp_path)
        p.history = [{"total_value": 100000, "daily_return": 0.001}] * 30
        check_graduation_criteria(p)
        captured = capsys.readouterr()
        assert "GRADUATION" not in captured.out

    def test_good_performance(self, tmp_path, capsys):
        p = _make_portfolio(tmp_path)
        # 63 days of positive returns with slight variation
        np.random.seed(42)
        p.history = []
        val = 100000
        for i in range(63):
            ret = 0.002 + np.random.normal(0, 0.0005)  # Positive with noise
            val *= (1 + ret)
            p.history.append({"total_value": val, "daily_return": ret})
        check_graduation_criteria(p)
        captured = capsys.readouterr()
        assert "GRADUATION CANDIDATE" in captured.out

    def test_poor_performance_no_graduation(self, tmp_path, capsys):
        p = _make_portfolio(tmp_path)
        # 63 days of mixed returns with high vol
        np.random.seed(42)
        p.history = []
        val = 100000
        for i in range(63):
            ret = np.random.normal(-0.001, 0.03)
            val *= (1 + ret)
            p.history.append({"total_value": val, "daily_return": ret})
        check_graduation_criteria(p)
        captured = capsys.readouterr()
        assert "GRADUATION CANDIDATE" not in captured.out
