#!/usr/bin/env python3
"""
Tests for Tail Risk Hedge Calculator — enums, configs, VIX percentile,
put/VIX premium estimation, protective put analysis, VIX overlay analysis,
and full recommendation generation.
"""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.risk.tail_hedge_calculator import (
    HedgeStrategy, HedgeAction,
    ProtectivePutConfig, VIXCallConfig, HedgeRecommendation,
    TailRiskHedgeCalculator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_calc(portfolio_value=100000):
    return TailRiskHedgeCalculator(portfolio_value=portfolio_value)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestHedgeStrategy:
    def test_values(self):
        assert HedgeStrategy.PROTECTIVE_PUT.value == "protective_put"
        assert HedgeStrategy.VIX_CALL.value == "vix_call"
        assert HedgeStrategy.VIX_CALL_SPREAD.value == "vix_call_spread"
        assert HedgeStrategy.COLLAR.value == "collar"
        assert HedgeStrategy.HYBRID.value == "hybrid"

    def test_members(self):
        assert len(HedgeStrategy) == 5


class TestHedgeAction:
    def test_values(self):
        assert HedgeAction.ENTER.value == "enter"
        assert HedgeAction.HOLD.value == "hold"
        assert HedgeAction.ROLL.value == "roll"
        assert HedgeAction.TAKE_PROFIT.value == "take_profit"
        assert HedgeAction.NO_ACTION.value == "no_action"

    def test_members(self):
        assert len(HedgeAction) == 5


# ---------------------------------------------------------------------------
# Config dataclass tests
# ---------------------------------------------------------------------------

class TestProtectivePutConfig:
    def test_defaults(self):
        c = ProtectivePutConfig(underlying='SPY')
        assert c.strike_pct == 0.95
        assert c.delta_target == 0.30
        assert c.dte == 60
        assert c.max_hedge_notional == 0.02

    def test_custom(self):
        c = ProtectivePutConfig(underlying='QQQ', strike_pct=0.90, dte=30)
        assert c.underlying == 'QQQ'
        assert c.strike_pct == 0.90
        assert c.dte == 30


class TestVIXCallConfig:
    def test_defaults(self):
        c = VIXCallConfig()
        assert c.strike_vix == 22.0
        assert c.dte == 90
        assert c.max_contracts == 10
        assert c.vix_entry_low == 15.0
        assert c.vix_exit_profit == 35.0


class TestHedgeRecommendation:
    def test_to_dict(self):
        rec = HedgeRecommendation(
            timestamp='2026-01-01', portfolio_value=100000,
            vix_spot=18.0, vix_percentile=45.0,
            underlying_spot={'SPY': 585.0}, portfolio_ath_distance=0.0,
            action=HedgeAction.ENTER, strategy=HedgeStrategy.VIX_CALL,
            contracts=2, strike=22.0, expiry='2026-04-01',
            premium=500, premium_pct=0.5,
            max_loss=500, breakeven=27.0, expected_payout_crisis=3600,
            rationale='test', alternative_actions=['Monitor'],
        )
        d = rec.to_dict()
        assert d['vix_spot'] == 18.0
        assert d['action'] == HedgeAction.ENTER
        assert d['strategy'] == HedgeStrategy.VIX_CALL


# ---------------------------------------------------------------------------
# TailRiskHedgeCalculator tests
# ---------------------------------------------------------------------------

class TestTailRiskHedgeCalculator:
    def test_init(self):
        calc = _make_calc()
        assert calc.portfolio_value == 100000

    def test_init_default_allocation(self):
        calc = _make_calc()
        assert calc.base_allocation == {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}

    def test_init_custom_allocation(self):
        calc = TailRiskHedgeCalculator(
            portfolio_value=50000,
            base_allocation={'SPY': 0.5, 'GLD': 0.5}
        )
        assert calc.portfolio_value == 50000
        assert calc.base_allocation['SPY'] == 0.5

    # VIX percentile
    def test_vix_percentile_low(self):
        calc = _make_calc()
        pct = calc._calculate_vix_percentile(12.0)
        assert pct < 30

    def test_vix_percentile_high(self):
        calc = _make_calc()
        pct = calc._calculate_vix_percentile(30.0)
        assert pct > 60

    def test_vix_percentile_bounded(self):
        calc = _make_calc()
        assert 0 <= calc._calculate_vix_percentile(5.0) <= 100
        assert 0 <= calc._calculate_vix_percentile(80.0) <= 100

    def test_vix_percentile_mean(self):
        calc = _make_calc()
        pct = calc._calculate_vix_percentile(19.5)
        assert 40 <= pct <= 60  # Near median

    # Put premium estimation
    def test_put_premium_positive(self):
        calc = _make_calc()
        premium = calc._estimate_put_premium('SPY', 585.0, 555.0, 60)
        assert premium > 0

    def test_put_premium_itm_more_expensive(self):
        calc = _make_calc()
        otm = calc._estimate_put_premium('SPY', 585.0, 555.0, 60)
        itm = calc._estimate_put_premium('SPY', 585.0, 600.0, 60)
        assert itm > otm

    def test_put_premium_longer_dte_more_expensive(self):
        calc = _make_calc()
        short = calc._estimate_put_premium('SPY', 585.0, 555.0, 30)
        long = calc._estimate_put_premium('SPY', 585.0, 555.0, 90)
        assert long > short

    def test_put_premium_contract_multiplier(self):
        calc = _make_calc()
        premium = calc._estimate_put_premium('SPY', 585.0, 555.0, 60)
        # Should be per-contract (100 shares)
        assert premium > 100  # At least $1 per share * 100

    def test_put_premium_custom_iv(self):
        calc = _make_calc()
        low_iv = calc._estimate_put_premium('SPY', 585.0, 555.0, 60, iv=0.10)
        high_iv = calc._estimate_put_premium('SPY', 585.0, 555.0, 60, iv=0.30)
        assert high_iv > low_iv

    # VIX call premium estimation
    def test_vix_call_premium_positive(self):
        calc = _make_calc()
        premium = calc._estimate_vix_call_premium(18.0, 22.0, 90)
        assert premium > 0

    def test_vix_call_itm_more_expensive(self):
        calc = _make_calc()
        otm = calc._estimate_vix_call_premium(18.0, 22.0, 90)
        itm = calc._estimate_vix_call_premium(25.0, 22.0, 90)
        assert itm > otm

    def test_vix_call_longer_dte_more_expensive(self):
        calc = _make_calc()
        short = calc._estimate_vix_call_premium(18.0, 22.0, 30)
        long = calc._estimate_vix_call_premium(18.0, 22.0, 180)
        assert long > short

    # analyze_protective_put
    def test_analyze_protective_put_returns_dict(self):
        calc = TailRiskHedgeCalculator(portfolio_value=500000)
        result = calc.analyze_protective_put('SPY', 585.0)
        assert 'strike' in result
        assert 'contracts' in result
        assert 'total_premium' in result

    def test_analyze_protective_put_strike(self):
        calc = TailRiskHedgeCalculator(portfolio_value=500000)
        result = calc.analyze_protective_put('SPY', 585.0)
        assert result['strike'] == 585.0 * 0.95

    def test_analyze_protective_put_contracts_positive(self):
        calc = TailRiskHedgeCalculator(portfolio_value=500000)
        result = calc.analyze_protective_put('SPY', 585.0)
        assert result['contracts'] >= 1

    def test_analyze_protective_put_premium_positive(self):
        calc = TailRiskHedgeCalculator(portfolio_value=500000)
        result = calc.analyze_protective_put('SPY', 585.0)
        assert result['total_premium'] > 0

    def test_analyze_protective_put_within_budget(self):
        calc = TailRiskHedgeCalculator(portfolio_value=500000)
        result = calc.analyze_protective_put('SPY', 585.0)
        assert result['cost_ok'] is True

    def test_analyze_protective_put_custom_config(self):
        calc = TailRiskHedgeCalculator(portfolio_value=500000)
        config = ProtectivePutConfig(underlying='SPY', strike_pct=0.90, dte=30)
        result = calc.analyze_protective_put('SPY', 585.0, config=config)
        assert result['strike'] == 585.0 * 0.90
        assert result['dte'] == 30

    def test_analyze_protective_put_default_prices(self):
        calc = TailRiskHedgeCalculator(portfolio_value=500000)
        result = calc.analyze_protective_put('QQQ')
        assert result['current_price'] == 490.0

    # analyze_vix_overlay
    def test_analyze_vix_overlay_returns_dict(self):
        calc = _make_calc()
        result = calc.analyze_vix_overlay(18.0)
        assert 'action' in result
        assert 'vix_percentile' in result

    def test_analyze_vix_overlay_low_vix_enter(self):
        calc = _make_calc()
        result = calc.analyze_vix_overlay(12.0)
        assert result['action'] == 'enter'

    def test_analyze_vix_overlay_mid_vix_enter(self):
        calc = _make_calc()
        result = calc.analyze_vix_overlay(18.0)
        assert result['action'] == 'enter'

    def test_analyze_vix_overlay_high_vix_take_profit(self):
        calc = _make_calc()
        result = calc.analyze_vix_overlay(36.0)
        assert result['action'] == 'take_profit'

    def test_analyze_vix_overlay_extreme_vix_no_action(self):
        calc = _make_calc()
        result = calc.analyze_vix_overlay(45.0)
        assert result['action'] == 'no_action'

    def test_analyze_vix_overlay_contracts_bounded(self):
        calc = _make_calc()
        result = calc.analyze_vix_overlay(12.0)
        assert result['contracts'] <= 10  # max_contracts

    def test_analyze_vix_overlay_has_premium(self):
        calc = _make_calc()
        result = calc.analyze_vix_overlay(18.0)
        assert result['total_premium'] > 0

    def test_analyze_vix_overlay_custom_config(self):
        calc = _make_calc()
        config = VIXCallConfig(strike_vix=25.0, dte=60)
        result = calc.analyze_vix_overlay(18.0, config=config)
        assert result['strike'] == 25.0
        assert result['dte'] == 60

    # get_full_recommendation
    def test_get_full_recommendation_returns_rec(self):
        calc = _make_calc()
        rec = calc.get_full_recommendation(vix_spot=18.0)
        assert isinstance(rec, HedgeRecommendation)

    def test_get_full_recommendation_low_vix(self):
        calc = _make_calc()
        rec = calc.get_full_recommendation(vix_spot=12.0)
        assert rec.action == HedgeAction.ENTER
        assert rec.strategy == HedgeStrategy.VIX_CALL

    def test_get_full_recommendation_drawdown(self):
        calc = TailRiskHedgeCalculator(portfolio_value=500000)
        rec = calc.get_full_recommendation(vix_spot=25.0, portfolio_distance_from_ath=-0.10)
        assert rec.action == HedgeAction.ENTER
        assert rec.strategy == HedgeStrategy.PROTECTIVE_PUT

    def test_get_full_recommendation_neutral(self):
        calc = _make_calc()
        # VIX at 23 = above entry_high (22), below exit_profit (35) → no_action
        rec = calc.get_full_recommendation(vix_spot=23.0, portfolio_distance_from_ath=0.0)
        assert rec.action == HedgeAction.NO_ACTION

    def test_get_full_recommendation_has_rationale(self):
        calc = _make_calc()
        rec = calc.get_full_recommendation(vix_spot=18.0)
        assert len(rec.rationale) > 0

    def test_get_full_recommendation_has_alternatives(self):
        calc = _make_calc()
        rec = calc.get_full_recommendation(vix_spot=18.0)
        assert len(rec.alternative_actions) > 0

    # VIX historical data
    def test_vix_historical_data(self):
        assert TailRiskHedgeCalculator.VIX_HISTORICAL['mean'] == 19.5
        assert TailRiskHedgeCalculator.VIX_HISTORICAL['max'] == 82.7

    def test_vix_term_structure(self):
        assert 30 in TailRiskHedgeCalculator.VIX_TERM_STRUCTURE
        assert 90 in TailRiskHedgeCalculator.VIX_TERM_STRUCTURE
        # Contango: longer DTE → higher multiplier
        assert TailRiskHedgeCalculator.VIX_TERM_STRUCTURE[90] > TailRiskHedgeCalculator.VIX_TERM_STRUCTURE[30]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
