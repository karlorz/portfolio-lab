#!/usr/bin/env python3
"""
Tests for 0DTE Yield Enhancement (Phase 1)

Coverage:
- Calculator tests (Greek approximations, premium estimation)
- Selector tests (strike selection logic)
- Position tests (lifecycle, P&L)

Target: 25+ tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from datetime import datetime, time, timedelta
from unittest.mock import MagicMock

from src.options.odte_yield_calculator import (
    ZeroDTECalculator, ZeroDTEConfig, OptionType, MarketCondition
)
from src.options.odte_yield_selector import (
    StrikeSelector, StrikeCandidate, SelectionCriteria, StrikeQuality
)
from src.options.odte_yield_position import (
    ZeroDTEPosition, ZeroDTETrade, ZeroDTETradeType, TradeStatus,
    CloseReason, OptionLeg, Greeks, ZeroDTEPerformance
)


# ============================================================================
# Calculator Tests
# ============================================================================

class TestZeroDTEConfig:
    """Test configuration defaults."""
    
    def test_default_config(self):
        config = ZeroDTEConfig()
        assert config.max_portfolio_allocation == 0.02
        assert config.position_size_pct == 0.005
        assert config.min_vix == 15.0
        assert config.max_vix == 35.0
        assert config.delta_target == 0.30
        assert config.min_premium_pct == 0.004
    
    def test_custom_config(self):
        config = ZeroDTEConfig(
            max_portfolio_allocation=0.03,
            min_vix=12.0,
            max_vix=40.0
        )
        assert config.max_portfolio_allocation == 0.03
        assert config.min_vix == 12.0
        assert config.max_vix == 40.0


class TestMarketCondition:
    """Test market condition classification."""
    
    def test_normal_condition(self):
        calc = ZeroDTECalculator()
        assert calc.classify_market_condition(12.0) == MarketCondition.NORMAL
        assert calc.classify_market_condition(14.9) == MarketCondition.NORMAL
    
    def test_elevated_vol(self):
        calc = ZeroDTECalculator()
        assert calc.classify_market_condition(15.0) == MarketCondition.ELEVATED_VOL
        assert calc.classify_market_condition(21.9) == MarketCondition.ELEVATED_VOL
    
    def test_high_vol(self):
        calc = ZeroDTECalculator()
        assert calc.classify_market_condition(22.0) == MarketCondition.HIGH_VOL
        assert calc.classify_market_condition(29.9) == MarketCondition.HIGH_VOL
    
    def test_extreme_vol(self):
        calc = ZeroDTECalculator()
        assert calc.classify_market_condition(30.0) == MarketCondition.EXTREME
        assert calc.classify_market_condition(50.0) == MarketCondition.EXTREME


class TestEntryAllowance:
    """Test entry permission logic."""
    
    def test_entry_allowed_normal(self):
        calc = ZeroDTECalculator()
        allowed, reason = calc.is_entry_allowed(16.0, time(11, 0), 0.02)
        assert allowed is True
        assert "permitted" in reason.lower()
    
    def test_entry_blocked_low_vix(self):
        calc = ZeroDTECalculator()
        allowed, reason = calc.is_entry_allowed(14.0, time(11, 0), 0.02)
        assert allowed is False
        assert "below minimum" in reason
    
    def test_entry_blocked_high_vix(self):
        calc = ZeroDTECalculator()
        allowed, reason = calc.is_entry_allowed(36.0, time(11, 0), 0.02)
        assert allowed is False
        assert "above maximum" in reason
    
    def test_entry_blocked_early(self):
        calc = ZeroDTECalculator()
        allowed, reason = calc.is_entry_allowed(16.0, time(9, 0), 0.02)
        assert allowed is False
        assert "too early" in reason.lower()
    
    def test_entry_blocked_late(self):
        calc = ZeroDTECalculator()
        allowed, reason = calc.is_entry_allowed(16.0, time(15, 0), 0.02)
        assert allowed is False
        assert "too late" in reason.lower()
    
    def test_entry_blocked_high_delta(self):
        calc = ZeroDTECalculator()
        allowed, reason = calc.is_entry_allowed(16.0, time(11, 0), 0.10)
        assert allowed is False
        assert "delta" in reason.lower()


class TestPremiumEstimation:
    """Test premium estimation calculations."""
    
    def test_atm_premium(self):
        calc = ZeroDTECalculator()
        # ATM call should have mostly time value
        premium = calc.estimate_premium(550.0, 550.0, 16.0, OptionType.CALL)
        assert premium > 0
        # ATM time value approx: S * σ * sqrt(T) * 0.4
        # 550 * 0.16 * sqrt(1/365) * 0.4 ≈ 1.84
        assert 1.0 < premium < 3.0
    
    def test_otm_premium_lower(self):
        calc = ZeroDTECalculator()
        atm = calc.estimate_premium(550.0, 550.0, 16.0, OptionType.CALL)
        otm = calc.estimate_premium(550.0, 560.0, 16.0, OptionType.CALL)
        # OTM should have lower premium than ATM
        assert otm < atm
    
    def test_itm_premium_higher(self):
        calc = ZeroDTECalculator()
        atm = calc.estimate_premium(550.0, 550.0, 16.0, OptionType.CALL)
        itm = calc.estimate_premium(550.0, 540.0, 16.0, OptionType.CALL)
        # ITM should have higher premium (intrinsic + time)
        assert itm > atm
        # ITM should have at least intrinsic
        assert itm >= (550.0 - 540.0)
    
    def test_higher_vol_higher_premium(self):
        calc = ZeroDTECalculator()
        low_vol = calc.estimate_premium(550.0, 560.0, 15.0, OptionType.CALL)
        high_vol = calc.estimate_premium(550.0, 560.0, 25.0, OptionType.CALL)
        assert high_vol > low_vol
    
    def test_put_premium(self):
        calc = ZeroDTECalculator()
        call_premium = calc.estimate_premium(550.0, 540.0, 16.0, OptionType.CALL)
        put_premium = calc.estimate_premium(550.0, 560.0, 16.0, OptionType.PUT)
        # Similar OTM distance should have roughly similar time value
        assert put_premium > 0


class TestDeltaApproximation:
    """Test delta estimation."""
    
    def test_atm_delta_near_50(self):
        calc = ZeroDTECalculator()
        delta = calc.delta_approximation(550.0, 550.0, 16.0)
        # ATM delta should be near 0.5
        assert 0.3 < delta < 0.7
    
    def test_deep_itm_delta_high(self):
        calc = ZeroDTECalculator()
        delta = calc.delta_approximation(550.0, 500.0, 16.0)
        # Deep ITM should have high delta
        assert delta > 0.8
    
    def test_deep_otm_delta_low(self):
        calc = ZeroDTECalculator()
        delta = calc.delta_approximation(550.0, 600.0, 16.0)
        # Deep OTM should have low delta
        assert delta < 0.2


class TestPositionSizing:
    """Test position size calculations."""
    
    def test_size_100k_portfolio(self):
        calc = ZeroDTECalculator()
        contracts = calc.calculate_position_size(100000)
        # 0.5% of 100K = $500, $500 / 100 shares = 5 contracts
        assert contracts == 5
    
    def test_size_50k_portfolio(self):
        calc = ZeroDTECalculator()
        contracts = calc.calculate_position_size(50000)
        # 0.5% of 50K = $250, but minimum effective position
        assert contracts >= 0
    
    def test_custom_max_position(self):
        calc = ZeroDTECalculator()
        contracts = calc.calculate_position_size(100000, max_position_value=1000)
        # $1000 max / 100 = 10 contracts
        assert contracts == 10


class TestExpectedReturn:
    """Test expected return calculations."""
    
    def test_expected_return_basic(self):
        calc = ZeroDTECalculator()
        result = calc.calculate_expected_return(
            premium=2.50, strike=555.0, spot=550.0, vix=16.0, win_rate=0.68
        )
        assert "max_gain" in result
        assert "expected_value" in result
        assert "risk_reward_ratio" in result
        assert result["max_gain"] == 2.50
        assert result["win_rate_assumed"] == 0.68
    
    def test_breakeven_calculation(self):
        calc = ZeroDTECalculator()
        result = calc.calculate_expected_return(
            premium=2.50, strike=555.0, spot=550.0, vix=16.0
        )
        # Short call breakeven = strike + premium
        assert result["breakeven"] == 557.50


class TestEmergencyClose:
    """Test emergency close logic."""
    
    def test_no_stop_normal(self):
        calc = ZeroDTECalculator()
        should_close, reason = calc.check_emergency_close(
            position_delta=-0.25,
            current_premium=1.50,
            entry_premium=2.50,
            current_time=time(13, 0)
        )
        assert should_close is False
        assert "normal" in reason.lower()
    
    def test_stop_high_delta(self):
        calc = ZeroDTECalculator()
        should_close, reason = calc.check_emergency_close(
            position_delta=-0.60,  # Exceeds 0.50 limit
            current_premium=4.00,
            entry_premium=2.50,
            current_time=time(13, 0)
        )
        assert should_close is True
        assert "delta" in reason.lower()
    
    def test_stop_loss_exceeded(self):
        calc = ZeroDTECalculator()
        # Loss = (4.00 - 2.50) / 2.50 = 60%, exceeds 15% limit
        should_close, reason = calc.check_emergency_close(
            position_delta=-0.25,
            current_premium=4.00,
            entry_premium=2.50,
            current_time=time(13, 0)
        )
        assert should_close is True
        assert "loss" in reason.lower()
    
    def test_stop_time_exit(self):
        calc = ZeroDTECalculator()
        should_close, reason = calc.check_emergency_close(
            position_delta=-0.25,
            current_premium=2.00,
            entry_premium=2.50,
            current_time=time(15, 45)  # After 3:30 cutoff
        )
        assert should_close is True
        assert "time" in reason.lower()


class TestNotionalExposure:
    """Test notional calculations."""
    
    def test_notional_single_contract(self):
        calc = ZeroDTECalculator()
        notional = calc.calculate_notional_exposure(550.0, 1)
        assert notional == 550.0 * 1 * 100  # $55,000
    
    def test_notional_multiple_contracts(self):
        calc = ZeroDTECalculator()
        notional = calc.calculate_notional_exposure(550.0, 5)
        assert notional == 550.0 * 5 * 100  # $275,000


class TestPortfolioDeltaImpact:
    """Test portfolio delta impact calculations."""
    
    def test_short_call_negative_delta(self):
        calc = ZeroDTECalculator()
        impact = calc.calculate_portfolio_delta_impact(
            option_delta=0.30,
            num_contracts=1,
            portfolio_value=100000
        )
        # Short call: -delta, -0.30 * 100 / 100000 = -0.0003 or -0.03%
        assert impact < 0
    
    def test_scale_with_contracts(self):
        calc = ZeroDTECalculator()
        impact1 = calc.calculate_portfolio_delta_impact(
            option_delta=0.30, num_contracts=1, portfolio_value=100000
        )
        impact5 = calc.calculate_portfolio_delta_impact(
            option_delta=0.30, num_contracts=5, portfolio_value=100000
        )
        # Should scale linearly
        assert abs(impact5 / impact1 - 5.0) < 0.01


# ============================================================================
# Selector Tests
# ============================================================================

class TestSelectionCriteria:
    """Test selection criteria defaults."""
    
    def test_default_criteria(self):
        criteria = SelectionCriteria()
        assert criteria.target_delta == 0.30
        assert criteria.delta_tolerance == 0.05
        assert criteria.min_premium_pct == 0.004
        assert criteria.max_spread_pct == 0.10
        assert criteria.min_volume == 100
        assert criteria.min_open_interest == 500


class TestStrikeCandidate:
    """Test candidate dataclass."""
    
    def test_candidate_creation(self):
        now = datetime.now()
        candidate = StrikeCandidate(
            underlying="SPY",
            strike=555.0,
            expiration=now,
            bid=2.40,
            ask=2.60,
            mid=2.50,
            delta_estimated=0.28,
            volume=500,
            open_interest=2000,
            quality=StrikeQuality.GOOD
        )
        assert candidate.underlying == "SPY"
        assert candidate.strike == 555.0
        assert candidate.is_valid is True
    
    def test_invalid_candidate(self):
        now = datetime.now()
        candidate = StrikeCandidate(
            underlying="SPY",
            strike=555.0,
            expiration=now,
            bid=2.40,
            ask=2.60,
            mid=2.50,
            quality=StrikeQuality.INVALID
        )
        assert candidate.is_valid is False


class TestStrikeSelection:
    """Test strike selector logic."""
    
    def test_select_strike_returns_candidate(self):
        # Use criteria with lower premium threshold for testing
        criteria = SelectionCriteria(min_premium_pct=0.002)  # 0.2% instead of 0.4%
        selector = StrikeSelector(criteria=criteria)
        candidate = selector.select_strike(550.0, 16.0)
        assert candidate is not None
        assert candidate.strike > 550.0  # Should be OTM for calls
        assert candidate.is_valid
    
    def test_strike_ladder_generation(self):
        criteria = SelectionCriteria(min_premium_pct=0.002)
        selector = StrikeSelector(criteria=criteria)
        ladder = selector.get_strike_ladder(550.0, 16.0)
        assert len(ladder) > 0
        # Check sorted by strike
        for i in range(1, len(ladder)):
            assert ladder[i].strike >= ladder[i-1].strike
    
    def test_strike_has_score(self):
        criteria = SelectionCriteria(min_premium_pct=0.002)
        selector = StrikeSelector(criteria=criteria)
        candidate = selector.select_strike(550.0, 16.0)
        assert candidate is not None
        assert candidate.score > 0
        assert candidate.score <= 100


# ============================================================================
# Position Tests
# ============================================================================

class TestGreeks:
    """Test Greeks dataclass."""
    
    def test_default_greeks(self):
        greeks = Greeks()
        assert greeks.delta == 0.0
        assert greeks.gamma == 0.0
        assert greeks.theta == 0.0
    
    def test_custom_greeks(self):
        greeks = Greeks(delta=-0.30, gamma=0.05, theta=0.15)
        assert greeks.delta == -0.30
        assert greeks.gamma == 0.05
        assert greeks.theta == 0.15


class TestOptionLeg:
    """Test OptionLeg dataclass."""
    
    def test_short_leg_properties(self):
        now = datetime.now()
        exp = now + timedelta(days=1)
        leg = OptionLeg(
            symbol="SPY",
            option_symbol="SPY1231C550",
            option_type="call",
            side="sell",
            quantity=1,
            strike=550.0,
            expiration=exp,
            entry_price=2.50,
            entry_time=now,
            current_price=1.80,
            entry_greeks=Greeks(delta=-0.30)
        )
        assert leg.is_short is True
        assert leg.premium_received == 2.50 * 1 * 100  # $250
        assert leg.notional_value == 550.0 * 1 * 100  # $55,000
        assert leg.unrealized_pnl == (2.50 - 1.80) * 100  # $70 profit
    
    def test_long_leg_properties(self):
        now = datetime.now()
        exp = now + timedelta(days=1)
        leg = OptionLeg(
            symbol="SPY",
            option_symbol="SPY1231C550",
            option_type="call",
            side="buy",
            quantity=1,
            strike=550.0,
            expiration=exp,
            entry_price=2.50,
            entry_time=now,
            current_price=3.00,
        )
        assert leg.is_short is False
        assert leg.premium_received == -2.50 * 100  # -$250 (paid)
        assert leg.unrealized_pnl == (3.00 - 2.50) * 100  # $50 profit
    
    def test_unrealized_pnl_loss(self):
        now = datetime.now()
        exp = now + timedelta(days=1)
        leg = OptionLeg(
            symbol="SPY",
            option_symbol="SPY1231C550",
            option_type="call",
            side="sell",
            quantity=1,
            strike=550.0,
            expiration=exp,
            entry_price=2.50,
            entry_time=now,
            current_price=3.50,  # Price went against us
        )
        assert leg.unrealized_pnl < 0  # Loss


class TestZeroDTEPosition:
    """Test ZeroDTEPosition dataclass."""
    
    def test_position_creation(self):
        now = datetime.now()
        exp = now + timedelta(days=1)
        
        leg = OptionLeg(
            symbol="SPY",
            option_symbol="SPY1231C550",
            option_type="call",
            side="sell",
            quantity=1,
            strike=550.0,
            expiration=exp,
            entry_price=2.50,
            entry_time=now,
            current_price=2.00,
        )
        
        position = ZeroDTEPosition(
            position_id="TEST_001",
            underlying="SPY",
            trade_type=ZeroDTETradeType.SHORT_CALL,
            entry_time=now,
            entry_spot=545.0,
            entry_vix=16.0,
            legs=[leg],
            status=TradeStatus.OPEN
        )
        
        assert position.position_id == "TEST_001"
        assert position.underlying == "SPY"
        assert position.is_active is True
        assert position.is_closed is False
        assert position.net_premium_received == 250.0
        assert position.total_unrealized_pnl == 50.0
    
    def test_max_profit(self):
        now = datetime.now()
        exp = now + timedelta(days=1)
        
        leg = OptionLeg(
            symbol="SPY",
            option_symbol="SPY1231C550",
            option_type="call",
            side="sell",
            quantity=1,
            strike=550.0,
            expiration=exp,
            entry_price=2.50,
            entry_time=now,
            current_price=2.00,
        )
        
        position = ZeroDTEPosition(
            position_id="TEST_001",
            underlying="SPY",
            trade_type=ZeroDTETradeType.SHORT_CALL,
            entry_time=now,
            entry_spot=545.0,
            entry_vix=16.0,
            legs=[leg],
            status=TradeStatus.OPEN
        )
        
        assert position.max_profit == 250.0  # Premium received
    
    def test_to_dict_serialization(self):
        now = datetime.now()
        exp = now + timedelta(days=1)
        
        leg = OptionLeg(
            symbol="SPY",
            option_symbol="SPY1231C550",
            option_type="call",
            side="sell",
            quantity=1,
            strike=550.0,
            expiration=exp,
            entry_price=2.50,
            entry_time=now,
            current_price=2.00,
        )
        
        position = ZeroDTEPosition(
            position_id="TEST_001",
            underlying="SPY",
            trade_type=ZeroDTETradeType.SHORT_CALL,
            entry_time=now,
            entry_spot=545.0,
            entry_vix=16.0,
            legs=[leg],
            status=TradeStatus.OPEN
        )
        
        data = position.to_dict()
        assert data["position_id"] == "TEST_001"
        assert data["underlying"] == "SPY"
        assert data["status"] == "open"
        assert "legs" in data
        assert len(data["legs"]) == 1


class TestZeroDTETrade:
    """Test ZeroDTETrade dataclass."""
    
    def test_trade_creation(self):
        now = datetime.now()
        trade = ZeroDTETrade(
            trade_id="TRADE_001",
            timestamp=now,
            underlying="SPY",
            trade_type=ZeroDTETradeType.SHORT_CALL,
            spot_price=550.0,
            vix=16.0,
            recommended_contracts=5,
            total_premium_expected=12.50
        )
        
        assert trade.trade_id == "TRADE_001"
        assert trade.underlying == "SPY"
        assert trade.urgency == "low"  # Default


class TestZeroDTEPerformance:
    """Test ZeroDTEPerformance tracking."""
    
    def test_calculate_metrics(self):
        perf = ZeroDTEPerformance(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            total_trades=10,
            winning_trades=7,
            losing_trades=3,
            total_premium_collected=2500.0,
            total_losses=800.0,
            commissions_paid=100.0
        )
        
        perf.calculate_metrics()
        
        assert perf.win_rate == 0.7
        assert perf.gross_pnl == 1700.0
        assert perf.net_pnl == 1600.0
        assert perf.profit_factor == 2500.0 / 800.0
    
    def test_zero_trades(self):
        perf = ZeroDTEPerformance(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            total_trades=0
        )
        
        perf.calculate_metrics()
        
        assert perf.win_rate == 0.0
        assert perf.profit_factor == 0.0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests across modules."""
    
    def test_calculator_to_selector_flow(self):
        """Test calculator integrates with selector."""
        calc = ZeroDTECalculator()
        criteria = SelectionCriteria(min_premium_pct=0.002)
        selector = StrikeSelector(calculator=calc, criteria=criteria)
        
        candidate = selector.select_strike(550.0, 16.0)
        assert candidate is not None
        assert candidate.delta_estimated > 0
    
    def test_position_with_leg_consistency(self):
        """Test position P&L calculation with leg."""
        now = datetime.now()
        exp = now + timedelta(days=1)
        
        entry_price = 2.50
        current_price = 1.80
        
        leg = OptionLeg(
            symbol="SPY",
            option_symbol="SPY1231C550",
            option_type="call",
            side="sell",
            quantity=2,
            strike=550.0,
            expiration=exp,
            entry_price=entry_price,
            entry_time=now,
            current_price=current_price,
        )
        
        # P&L should be: (entry - current) * qty * 100
        expected_pnl = (entry_price - current_price) * 2 * 100
        
        position = ZeroDTEPosition(
            position_id="TEST_001",
            underlying="SPY",
            trade_type=ZeroDTETradeType.SHORT_CALL,
            entry_time=now,
            entry_spot=545.0,
            entry_vix=16.0,
            legs=[leg],
            status=TradeStatus.OPEN
        )
        
        assert position.total_unrealized_pnl == expected_pnl


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
