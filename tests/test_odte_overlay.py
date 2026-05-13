#!/usr/bin/env python3
"""
Tests for 0DTE Options Overlay — data classes, GEX calculator, position sizing,
three-stop manager, iron condor construction, and overlay orchestration.
"""
import sys
import os
import json
import sqlite3
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta, time
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.options.odte_overlay import (
    StopType, TradeStatus, Greeks, OptionLeg, IronCondor,
    GEXLevel, GEXProfile, GEXCalculator,
    PositionSizer, ThreeStopManager, ODTEOverlay, ODTEBacktester,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_leg(strike=550.0, side="sell", option_type="call", premium=2.50, qty=1):
    """Create a test OptionLeg."""
    return OptionLeg(
        symbol="SPX",
        option_symbol=f"SPXC{int(strike)}",
        strike=strike,
        expiration=datetime.now().replace(hour=16, minute=0),
        option_type=option_type,
        side=side,
        quantity=qty,
        entry_price=premium,
        current_price=premium,
        greeks=Greeks(delta=-0.16 if side == "sell" else 0.05),
    )


def _make_condor(spot=5500.0, vix=18.0, wing_width=10):
    """Create a test IronCondor."""
    short_call = _make_leg(strike=spot + 25, side="sell", option_type="call", premium=3.0)
    long_call = _make_leg(strike=spot + 25 + wing_width, side="buy", option_type="call", premium=1.0)
    short_put = _make_leg(strike=spot - 25, side="sell", option_type="put", premium=3.0)
    long_put = _make_leg(strike=spot - 25 - wing_width, side="buy", option_type="put", premium=1.0)
    return IronCondor(
        trade_id="TEST_001",
        underlying="SPX",
        entry_time=datetime.now() - timedelta(hours=2),
        short_call=short_call,
        long_call=long_call,
        short_put=short_put,
        long_put=long_put,
        entry_spot=spot,
        entry_vix=vix,
        status=TradeStatus.OPEN,
    )


def _create_market_db(db_path, symbol="SPY", price=550.0, days=30):
    """Create a minimal market.db with price data."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
        PRIMARY KEY (symbol, date))
    """)
    base_date = datetime.now()
    for i in range(days):
        d = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
        noise = np.random.normal(0, 2.0)
        conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                     (symbol, d, round(price + noise, 2)))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    """Test StopType and TradeStatus enums."""

    def test_stop_type_values(self):
        assert StopType.PRICE.value == "price"
        assert StopType.PERCENTAGE.value == "percentage"
        assert StopType.TIME.value == "time"

    def test_trade_status_values(self):
        assert TradeStatus.PENDING.value == "pending"
        assert TradeStatus.OPEN.value == "open"
        assert TradeStatus.CLOSED.value == "closed"
        assert TradeStatus.STOPPED.value == "stopped"


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestGreeks:
    """Test Greeks dataclass."""

    def test_defaults(self):
        g = Greeks()
        assert g.delta == 0.0
        assert g.gamma == 0.0
        assert g.theta == 0.0

    def test_custom_values(self):
        g = Greeks(delta=-0.16, gamma=0.05, theta=0.02, vega=0.01, rho=0.001)
        assert g.delta == -0.16


class TestOptionLeg:
    """Test OptionLeg dataclass."""

    def test_short_leg_properties(self):
        leg = _make_leg(strike=5500.0, side="sell", premium=3.0, qty=2)
        assert leg.is_short is True
        assert leg.premium == -600.0  # 3.0 * 2 * 100 * -1
        assert leg.notional_value == 5500.0 * 2 * 100

    def test_long_leg_properties(self):
        leg = _make_leg(strike=5525.0, side="buy", premium=1.0, qty=2)
        assert leg.is_short is False
        assert leg.premium == 200.0  # 1.0 * 2 * 100 * 1


class TestIronCondor:
    """Test IronCondor dataclass."""

    def test_risk_metrics_auto_calculated(self):
        condor = _make_condor(spot=5500.0)
        assert condor.max_profit > 0
        assert condor.max_loss > 0
        assert condor.wing_width > 0

    def test_breakevens(self):
        condor = _make_condor(spot=5500.0)
        lower, upper = condor.breakevens
        assert lower < condor.short_put.strike
        assert upper > condor.short_call.strike

    def test_to_dict(self):
        condor = _make_condor()
        d = condor.to_dict()
        assert d["trade_id"] == "TEST_001"
        assert d["underlying"] == "SPX"
        assert "entry_spot" in d
        assert "max_profit" in d

    def test_max_profit_is_net_credit(self):
        """Max profit = total net credit received from both sides."""
        condor = _make_condor()
        # short_call premium=3, long_call premium=1 → call credit = -2
        # short_put premium=3, long_put premium=1 → put credit = -2
        # max_profit = abs(-2 + -2) * 100 = 400 (but formula uses abs)
        assert condor.max_profit > 0

    def test_status_default(self):
        """Default status is PENDING."""
        short_call = _make_leg(strike=5525, side="sell", option_type="call", premium=3.0)
        long_call = _make_leg(strike=5535, side="buy", option_type="call", premium=1.0)
        short_put = _make_leg(strike=5475, side="sell", option_type="put", premium=3.0)
        long_put = _make_leg(strike=5465, side="buy", option_type="put", premium=1.0)
        condor = IronCondor(
            trade_id="T2", underlying="SPX", entry_time=datetime.now(),
            short_call=short_call, long_call=long_call,
            short_put=short_put, long_put=long_put,
            entry_spot=5500.0, entry_vix=18.0,
        )
        assert condor.status == TradeStatus.PENDING


# ---------------------------------------------------------------------------
# GEX Calculator tests
# ---------------------------------------------------------------------------

class TestGEXCalculator:
    """Test GEXCalculator with mocked database."""

    def test_get_spot_price_from_db(self, tmp_path):
        """Returns spot price from market.db."""
        db_path = tmp_path / "market.db"
        _create_market_db(db_path, "SPY", price=550.0)
        calc = GEXCalculator(db_path=db_path)
        price = calc._get_spot_price("SPY")
        assert 545.0 < price < 555.0

    def test_get_spot_price_missing_symbol(self, tmp_path):
        """Returns 0.0 for missing symbol."""
        db_path = tmp_path / "market.db"
        _create_market_db(db_path, "SPY", price=550.0)
        calc = GEXCalculator(db_path=db_path)
        assert calc._get_spot_price("NONEXISTENT") == 0.0

    def test_get_spot_price_missing_db(self, tmp_path):
        """Returns 0.0 when DB doesn't exist."""
        calc = GEXCalculator(db_path=tmp_path / "nonexistent.db")
        assert calc._get_spot_price("SPY") == 0.0

    def test_calculate_gex_returns_profile(self, tmp_path):
        """calculate_gex returns a valid GEXProfile."""
        db_path = tmp_path / "market.db"
        _create_market_db(db_path, "SPY", price=550.0)
        calc = GEXCalculator(db_path=db_path)
        profile = calc.calculate_gex("SPY", spot_price=550.0)
        assert isinstance(profile, GEXProfile)
        assert profile.underlying == "SPY"
        assert profile.spot_price == 550.0

    def test_gex_profile_has_levels(self, tmp_path):
        """GEX profile contains strike levels."""
        db_path = tmp_path / "market.db"
        _create_market_db(db_path, "SPY", price=550.0)
        calc = GEXCalculator(db_path=db_path)
        profile = calc.calculate_gex("SPY", spot_price=550.0)
        assert len(profile.levels) > 0

    def test_check_pin_risk_no_risk(self, tmp_path):
        """Spot far from max gamma → no pin risk."""
        profile = GEXProfile(
            underlying="SPY", spot_price=550.0, timestamp=datetime.now(),
            max_gamma_strike=5600.0, max_gamma_abs=300.0,
        )
        db_path = tmp_path / "market.db"
        _create_market_db(db_path, "SPY", price=550.0)
        calc = GEXCalculator(db_path=db_path)
        result = calc.check_pin_risk(profile)
        assert result["pin_risk_detected"] is False
        assert result["recommendation"] == "normal"

    def test_check_pin_risk_detected(self, tmp_path):
        """Spot within 0.5% of max gamma with high gamma → pin risk."""
        profile = GEXProfile(
            underlying="SPY", spot_price=550.0, timestamp=datetime.now(),
            max_gamma_strike=550.5, max_gamma_abs=600.0,
        )
        db_path = tmp_path / "market.db"
        _create_market_db(db_path, "SPY", price=550.0)
        calc = GEXCalculator(db_path=db_path)
        result = calc.check_pin_risk(profile)
        assert result["pin_risk_detected"] is True
        assert result["recommendation"] == "avoid_new_positions"


# ---------------------------------------------------------------------------
# PositionSizer tests
# ---------------------------------------------------------------------------

class TestPositionSizer:
    """Test PositionSizer.calculate_size."""

    def test_basic_sizing(self):
        """Basic position sizing returns valid result."""
        sizer = PositionSizer(portfolio_value=100000.0)
        result = sizer.calculate_size(vix=18.0, wing_width=10, max_loss_per_contract=800)
        assert result["num_contracts"] >= 0
        assert result["risk_pct"] >= 0

    def test_low_vix_larger_size(self):
        """Low VIX allows more contracts than high VIX."""
        sizer_low = PositionSizer(portfolio_value=100000.0)
        result_low = sizer_low.calculate_size(vix=12.0, wing_width=10, max_loss_per_contract=800)

        sizer_high = PositionSizer(portfolio_value=100000.0)
        result_high = sizer_high.calculate_size(vix=30.0, wing_width=10, max_loss_per_contract=800)

        assert result_low["num_contracts"] >= result_high["num_contracts"]

    def test_max_risk_respected(self):
        """Risk never exceeds 2% of portfolio."""
        sizer = PositionSizer(portfolio_value=100000.0)
        result = sizer.calculate_size(vix=18.0, wing_width=10, max_loss_per_contract=800)
        assert result["risk_pct"] <= 0.02 + 0.001  # Small tolerance

    def test_circuit_breaker_reduces_size(self):
        """Circuit breaker scalar reduces position size."""
        sizer_full = PositionSizer(portfolio_value=100000.0, circuit_breaker_scalar=1.0)
        result_full = sizer_full.calculate_size(vix=18.0, wing_width=10, max_loss_per_contract=800)

        sizer_half = PositionSizer(portfolio_value=100000.0, circuit_breaker_scalar=0.5)
        result_half = sizer_half.calculate_size(vix=18.0, wing_width=10, max_loss_per_contract=800)

        assert result_half["num_contracts"] <= result_full["num_contracts"]

    def test_invalid_max_loss_returns_zero(self):
        """Zero or negative max loss → 0 contracts."""
        sizer = PositionSizer(portfolio_value=100000.0)
        result = sizer.calculate_size(vix=18.0, wing_width=10, max_loss_per_contract=0)
        assert result["num_contracts"] == 0

    def test_high_gex_reduces_size(self):
        """High GEX reduces position size by 20%."""
        sizer = PositionSizer(portfolio_value=100000.0)
        gex_low = GEXProfile(underlying="SPY", spot_price=550.0, timestamp=datetime.now(), max_gamma_abs=100)
        result_low = sizer.calculate_size(vix=18.0, wing_width=10, max_loss_per_contract=800, gex_profile=gex_low)

        gex_high = GEXProfile(underlying="SPY", spot_price=550.0, timestamp=datetime.now(), max_gamma_abs=1500)
        result_high = sizer.calculate_size(vix=18.0, wing_width=10, max_loss_per_contract=800, gex_profile=gex_high)

        # High GEX should reduce or equal
        assert result_high["num_contracts"] <= result_low["num_contracts"]

    def test_sizing_notes_present(self):
        """Sizing result includes notes."""
        sizer = PositionSizer(portfolio_value=100000.0)
        result = sizer.calculate_size(vix=18.0, wing_width=10, max_loss_per_contract=800)
        assert len(result["sizing_notes"]) >= 3


# ---------------------------------------------------------------------------
# ThreeStopManager tests
# ---------------------------------------------------------------------------

class TestThreeStopManager:
    """Test ThreeStopManager.check_stops and dynamic adjustments."""

    def test_no_stop_triggered(self):
        """No stop triggered when P&L is within bounds."""
        mgr = ThreeStopManager(profit_target_pct=0.50, stop_loss_pct=2.0)
        condor = _make_condor()
        # P&L = 10% of max profit → no stop
        current_pnl = condor.max_profit * 0.1
        now = datetime.now().replace(hour=14, minute=0)
        should_exit, reason, _ = mgr.check_stops(condor, current_pnl, now)
        assert should_exit is False

    def test_profit_target_hit(self):
        """Profit target triggers exit."""
        mgr = ThreeStopManager(profit_target_pct=0.50)
        condor = _make_condor()
        current_pnl = condor.max_profit * 0.6  # 60% > 50% target
        now = datetime.now().replace(hour=14, minute=0)
        should_exit, reason, _ = mgr.check_stops(condor, current_pnl, now)
        assert should_exit is True
        assert "profit_target" in reason

    def test_stop_loss_hit(self):
        """Stop loss triggers exit."""
        mgr = ThreeStopManager(stop_loss_pct=2.0)
        condor = _make_condor()
        current_pnl = -condor.max_loss * 2.5  # Loss exceeds 2x
        now = datetime.now().replace(hour=14, minute=0)
        should_exit, reason, _ = mgr.check_stops(condor, current_pnl, now)
        assert should_exit is True
        assert "stop_loss" in reason

    def test_time_exit(self):
        """Time-based exit triggers at 3:00 PM."""
        mgr = ThreeStopManager()
        condor = _make_condor()
        now = datetime.now().replace(hour=15, minute=1)
        should_exit, reason, _ = mgr.check_stops(condor, condor.max_profit * 0.1, now)
        assert should_exit is True
        assert "time_exit" in reason

    def test_default_exit_time(self):
        """Default exit time is 3:00 PM."""
        mgr = ThreeStopManager()
        assert mgr.exit_time == time(15, 0)

    def test_dynamic_adjustments_near_gamma(self):
        """Near max gamma → tightened stops."""
        mgr = ThreeStopManager(profit_target_pct=0.50, stop_loss_pct=2.0)
        condor = _make_condor(spot=5500.0)
        gex = GEXProfile(
            underlying="SPX", spot_price=5500.0, timestamp=datetime.now(),
            max_gamma_strike=5502.0,  # Very close to spot
        )
        adj = mgr.calculate_dynamic_adjustments(condor, gex, minutes_to_close=30)
        assert adj["adjusted_profit_target"] < 0.50  # Reduced
        assert adj["adjusted_stop_loss"] > 2.0  # Widened
        assert len(adj["notes"]) > 0

    def test_dynamic_adjustments_safe_zone(self):
        """Far from max gamma → relaxed stops."""
        mgr = ThreeStopManager(profit_target_pct=0.50, stop_loss_pct=2.0)
        condor = _make_condor(spot=5500.0)
        gex = GEXProfile(
            underlying="SPX", spot_price=5500.0, timestamp=datetime.now(),
            max_gamma_strike=5700.0,  # Far from spot
        )
        adj = mgr.calculate_dynamic_adjustments(condor, gex, minutes_to_close=120)
        assert adj["adjusted_profit_target"] > 0.50  # Increased

    def test_dynamic_adjustments_last_hour(self):
        """Last hour → reduced profit target."""
        mgr = ThreeStopManager(profit_target_pct=0.50)
        condor = _make_condor(spot=5500.0)
        gex = GEXProfile(underlying="SPX", spot_price=5500.0, timestamp=datetime.now())
        adj = mgr.calculate_dynamic_adjustments(condor, gex, minutes_to_close=30)
        assert adj["adjusted_profit_target"] < 0.50


# ---------------------------------------------------------------------------
# ODTEOverlay tests
# ---------------------------------------------------------------------------

class TestODTEOverlay:
    """Test ODTEOverlay main class."""

    def _make_overlay(self, tmp_path, portfolio_value=100000.0):
        """Create overlay with mocked paths."""
        overlay = ODTEOverlay.__new__(ODTEOverlay)
        overlay.portfolio_value = portfolio_value
        overlay.mode = "paper"
        overlay.gex_calc = GEXCalculator(db_path=tmp_path / "market.db")
        overlay.stop_mgr = ThreeStopManager()
        overlay.active_trades = []
        overlay.trade_history = []
        return overlay

    def test_wing_width_low_vix(self):
        """VIX < 20 → 10-wide wings."""
        overlay = self._make_overlay(Path("/tmp"))
        assert overlay.get_wing_width(15) == 10
        assert overlay.get_wing_width(19) == 10

    def test_wing_width_mid_vix(self):
        """VIX 20-25 → 15-wide wings."""
        overlay = self._make_overlay(Path("/tmp"))
        assert overlay.get_wing_width(20) == 15
        assert overlay.get_wing_width(25) == 15

    def test_wing_width_high_vix(self):
        """VIX > 25 → 20-wide wings."""
        overlay = self._make_overlay(Path("/tmp"))
        assert overlay.get_wing_width(30) == 20

    def test_circuit_breaker_scalars(self):
        """Circuit breaker status maps to correct scalar."""
        overlay = self._make_overlay(Path("/tmp"))
        assert overlay._get_circuit_breaker_scalar("green") == 1.0
        assert overlay._get_circuit_breaker_scalar("yellow") == 1.0
        assert overlay._get_circuit_breaker_scalar("orange") == 0.75
        assert overlay._get_circuit_breaker_scalar("red") == 0.50
        assert overlay._get_circuit_breaker_scalar("black") == 0.0
        assert overlay._get_circuit_breaker_scalar("unknown") == 0.0

    def test_construct_condor(self, tmp_path):
        """construct_condor returns a valid IronCondor."""
        db_path = tmp_path / "market.db"
        _create_market_db(db_path, "SPY", price=550.0)
        overlay = self._make_overlay(tmp_path)
        overlay.gex_calc = GEXCalculator(db_path=db_path)
        condor = overlay.construct_condor("SPY", 550.0, 18.0, 1.0)
        assert condor is not None
        assert isinstance(condor, IronCondor)
        assert condor.status == TradeStatus.OPEN
        assert condor.max_profit > 0
        assert condor.max_loss > 0

    def test_construct_condor_strikes_ordered(self, tmp_path):
        """Condor strikes are correctly ordered."""
        db_path = tmp_path / "market.db"
        _create_market_db(db_path, "SPY", price=550.0)
        overlay = self._make_overlay(tmp_path)
        overlay.gex_calc = GEXCalculator(db_path=db_path)
        condor = overlay.construct_condor("SPY", 550.0, 18.0, 1.0)
        assert condor.long_put.strike < condor.short_put.strike
        assert condor.short_put.strike < condor.short_call.strike
        assert condor.short_call.strike < condor.long_call.strike

    def test_estimate_premium_positive(self, tmp_path):
        """Premium estimate is always positive."""
        overlay = self._make_overlay(tmp_path)
        p = overlay._estimate_premium(550.0, 555.0, 18.0, 1/252, "call")
        assert p > 0

    def test_estimate_premium_otm_cheaper(self, tmp_path):
        """OTM options cheaper than ATM."""
        overlay = self._make_overlay(tmp_path)
        atm = overlay._estimate_premium(550.0, 550.0, 18.0, 1/252, "call")
        otm = overlay._estimate_premium(550.0, 570.0, 18.0, 1/252, "call")
        assert otm < atm

    def test_get_stats_empty(self, tmp_path):
        """No history → message."""
        overlay = self._make_overlay(tmp_path)
        stats = overlay.get_stats()
        assert "message" in stats

    def test_get_stats_with_history(self, tmp_path):
        """Stats computed from trade history."""
        overlay = self._make_overlay(tmp_path)
        overlay.trade_history = [
            {"realized_pnl": 150.0},
            {"realized_pnl": -80.0},
            {"realized_pnl": 200.0},
        ]
        stats = overlay.get_stats()
        assert stats["total_trades"] == 3
        assert stats["winning_trades"] == 2
        assert stats["losing_trades"] == 1
        assert stats["win_rate"] == pytest.approx(2/3, abs=0.01)
        assert stats["total_pnl"] == 270.0

    def test_manage_positions_empty(self, tmp_path):
        """No active trades → empty actions."""
        overlay = self._make_overlay(tmp_path)
        actions = overlay.manage_positions()
        assert actions == []

    def test_manage_positions_with_trade(self, tmp_path):
        """Active trade gets monitored or closed."""
        overlay = self._make_overlay(tmp_path)
        condor = _make_condor()
        overlay.active_trades = [condor]
        actions = overlay.manage_positions()
        assert len(actions) == 1
        assert actions[0]["trade_id"] == "TEST_001"

    def test_close_position(self, tmp_path):
        """Close position moves to history."""
        overlay = self._make_overlay(tmp_path)
        condor = _make_condor()
        overlay.active_trades = [condor]
        action = overlay._close_position(condor, "test_close", 150.0)
        assert action["action"] == "close"
        assert action["reason"] == "test_close"
        assert len(overlay.active_trades) == 0
        assert len(overlay.trade_history) == 1


# ---------------------------------------------------------------------------
# ODTEBacktester tests
# ---------------------------------------------------------------------------

class TestODTEBacktester:
    """Test ODTEBacktester."""

    def test_run_returns_dict(self):
        """Backtest run returns a valid dict."""
        bt = ODTEBacktester(
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 12, 31),
            portfolio_value=100000.0,
        )
        result = bt.run()
        assert isinstance(result, dict)
        assert "backtest_period" in result

    def test_backtest_period(self):
        """Backtest period matches inputs."""
        start = datetime(2025, 1, 1)
        end = datetime(2025, 12, 31)
        bt = ODTEBacktester(start_date=start, end_date=end)
        result = bt.run()
        assert result["backtest_period"]["start"] == start.isoformat()
        assert result["backtest_period"]["end"] == end.isoformat()


# ---------------------------------------------------------------------------
# GEXLevel / GEXProfile dataclass tests
# ---------------------------------------------------------------------------

class TestGEXDataClasses:
    """Test GEXLevel and GEXProfile."""

    def test_gex_level_creation(self):
        level = GEXLevel(strike=5500.0, gamma_exposure=1000.0, call_gamma=600.0, put_gamma=400.0, net_delta=0.1)
        assert level.strike == 5500.0
        assert level.gamma_exposure == 1000.0

    def test_gex_profile_defaults(self):
        profile = GEXProfile(underlying="SPY", spot_price=550.0, timestamp=datetime.now())
        assert profile.levels == []
        assert profile.max_gamma_strike == 0.0
        assert profile.put_wall == 0.0
        assert profile.total_gamma == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
