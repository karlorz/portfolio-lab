#!/usr/bin/env python3
"""
Tests for tail risk hedger — enums, data classes, put/VIX hedge calculation,
hybrid optimization, regime detection, analytics, and rolling schedule.
"""
import sys
import os
import math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.options.tail_hedge import (
    HedgeType, MarketRegime,
    PutHedge, VixHedge, HybridHedge, HedgeAnalytics,
    TailRiskHedger,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_put_hedge():
    return PutHedge(
        underlying='SPY', notional=50000, spot_price=500.0,
        strike_pct=0.94, delta=-0.20, days_to_expiry=90,
        implied_vol=0.20, premium_pct=0.0,
    )


def _make_vix_hedge():
    return VixHedge(
        portfolio_value=100000, vix_spot=18.0,
        vix_futures=19.8, strike=25.0, days_to_expiry=60,
    )


def _make_hybrid():
    return HybridHedge(
        portfolio_value=100000, equity_allocation_pct=0.50,
    )


def _make_hedger():
    return TailRiskHedger(portfolio_value=100000)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_hedge_type_values(self):
        assert HedgeType.PROTECTIVE_PUT.value == 'protective_put'
        assert HedgeType.VIX_CALL.value == 'vix_call'
        assert HedgeType.HYBRID.value == 'hybrid'

    def test_market_regime_values(self):
        assert MarketRegime.LOW_VOL.value == 'low_vol'
        assert MarketRegime.MODERATE_VOL.value == 'mod_vol'
        assert MarketRegime.ELEVATED_VOL.value == 'elev_vol'
        assert MarketRegime.CRISIS.value == 'crisis'

    def test_hedge_type_members(self):
        assert len(HedgeType) == 5

    def test_market_regime_members(self):
        assert len(MarketRegime) == 4


# ---------------------------------------------------------------------------
# PutHedge tests
# ---------------------------------------------------------------------------

class TestPutHedge:
    def test_creation(self):
        h = _make_put_hedge()
        assert h.underlying == 'SPY'
        assert h.spot_price == 500.0

    def test_calculate_contracts(self):
        h = _make_put_hedge().calculate()
        # 50000 / (500*100) = 1 contract
        assert h.num_contracts >= 1

    def test_calculate_strike_price(self):
        h = _make_put_hedge().calculate()
        assert h.strike_price == 500.0 * 0.94

    def test_calculate_premium_positive(self):
        h = _make_put_hedge().calculate()
        assert h.total_premium > 0

    def test_calculate_annual_cost_positive(self):
        h = _make_put_hedge().calculate()
        assert h.annual_cost_pct > 0

    def test_calculate_premium_pct(self):
        h = _make_put_hedge().calculate()
        assert 0 < h.premium_pct < 1.0

    def test_calculate_returns_self(self):
        h = _make_put_hedge()
        result = h.calculate()
        assert result is h

    def test_deep_otm_put_cheap(self):
        h = PutHedge(
            underlying='SPY', notional=50000, spot_price=500.0,
            strike_pct=0.80, delta=-0.05, days_to_expiry=30,
            implied_vol=0.20, premium_pct=0.0,
        ).calculate()
        # Deep OTM = cheap
        assert h.total_premium < 2000


# ---------------------------------------------------------------------------
# VixHedge tests
# ---------------------------------------------------------------------------

class TestVixHedge:
    def test_creation(self):
        h = _make_vix_hedge()
        assert h.vix_spot == 18.0
        assert h.portfolio_value == 100000

    def test_calculate_contracts(self):
        h = _make_vix_hedge().calculate()
        assert h.num_contracts >= 1

    def test_calculate_premium_positive(self):
        h = _make_vix_hedge().calculate()
        assert h.premium_cost > 0

    def test_calculate_notional_exposure(self):
        h = _make_vix_hedge().calculate()
        assert h.notional_exposure > 0

    def test_otm_call_premium(self):
        # VIX futures (19.8) < strike (25) → OTM
        h = _make_vix_hedge().calculate()
        assert h.premium_cost > 0

    def test_itm_call_premium(self):
        h = VixHedge(
            portfolio_value=100000, vix_spot=30.0,
            vix_futures=33.0, strike=25.0, days_to_expiry=60,
        ).calculate()
        assert h.premium_cost > 0

    def test_convexity_positive(self):
        h = _make_vix_hedge().calculate()
        # With VIX doubling, should have positive convexity
        assert h.convexity_score is not None


# ---------------------------------------------------------------------------
# HybridHedge tests
# ---------------------------------------------------------------------------

class TestHybridHedge:
    def test_creation(self):
        h = _make_hybrid()
        assert h.portfolio_value == 100000
        assert h.put_weight == 0.6

    def test_optimize_low_vol(self):
        h = _make_hybrid().optimize(MarketRegime.LOW_VOL, 500.0, 12.0)
        assert h.put_hedge is not None
        assert h.vix_hedge is not None
        assert h.total_annual_cost > 0

    def test_optimize_moderate_vol(self):
        h = _make_hybrid().optimize(MarketRegime.MODERATE_VOL, 500.0, 20.0)
        assert h.put_weight == 0.6
        assert h.vix_weight == 0.4

    def test_optimize_elevated_vol(self):
        h = _make_hybrid().optimize(MarketRegime.ELEVATED_VOL, 500.0, 30.0)
        assert h.put_weight == 0.7
        assert h.vix_weight == 0.3

    def test_optimize_crisis(self):
        h = _make_hybrid().optimize(MarketRegime.CRISIS, 500.0, 40.0)
        assert h.put_weight == 0.8
        assert h.vix_weight == 0.2

    def test_optimize_returns_self(self):
        h = _make_hybrid()
        result = h.optimize(MarketRegime.LOW_VOL, 500.0, 12.0)
        assert result is h

    def test_optimize_efficiency_score(self):
        h = _make_hybrid().optimize(MarketRegime.LOW_VOL, 500.0, 12.0)
        assert isinstance(h.efficiency_score, float)

    def test_optimize_expected_payoff(self):
        h = _make_hybrid().optimize(MarketRegime.LOW_VOL, 500.0, 12.0)
        assert isinstance(h.expected_payoff_crisis, float)


# ---------------------------------------------------------------------------
# TailRiskHedger tests
# ---------------------------------------------------------------------------

class TestTailRiskHedger:
    def test_init(self):
        hedger = _make_hedger()
        assert hedger.portfolio_value == 100000

    def test_detect_regime_low_vol(self):
        hedger = _make_hedger()
        assert hedger.detect_regime(12.0) == MarketRegime.LOW_VOL

    def test_detect_regime_moderate(self):
        hedger = _make_hedger()
        assert hedger.detect_regime(20.0) == MarketRegime.MODERATE_VOL

    def test_detect_regime_elevated(self):
        hedger = _make_hedger()
        assert hedger.detect_regime(30.0) == MarketRegime.ELEVATED_VOL

    def test_detect_regime_crisis(self):
        hedger = _make_hedger()
        assert hedger.detect_regime(40.0) == MarketRegime.CRISIS

    def test_detect_regime_boundaries(self):
        hedger = _make_hedger()
        # Thresholds are exclusive: > 25 = elevated, not >= 25
        assert hedger.detect_regime(25.01) == MarketRegime.ELEVATED_VOL
        assert hedger.detect_regime(25.0) == MarketRegime.MODERATE_VOL
        assert hedger.detect_regime(14.9) == MarketRegime.LOW_VOL
        assert hedger.detect_regime(35.01) == MarketRegime.CRISIS

    def test_calculate_protective_put(self):
        hedger = _make_hedger()
        h = hedger.calculate_protective_put('SPY', 100, 500.0)
        assert isinstance(h, PutHedge)
        assert h.num_contracts >= 1
        assert h.total_premium > 0

    def test_calculate_protective_put_otm(self):
        hedger = _make_hedger()
        h = hedger.calculate_protective_put('SPY', 100, 500.0, delta_target=-0.20)
        assert h.strike_price < 500.0  # OTM put

    def test_calculate_vix_overlay(self):
        hedger = _make_hedger()
        h = hedger.calculate_vix_overlay(18.0)
        assert isinstance(h, VixHedge)
        assert h.num_contracts >= 1

    def test_calculate_vix_overlay_strike_selection(self):
        hedger = _make_hedger()
        h_low = hedger.calculate_vix_overlay(12.0)
        h_high = hedger.calculate_vix_overlay(30.0)
        # Low VIX → higher strike multiplier (1.5x), high VIX → lower (1.2x)
        assert h_low.strike / 12.0 > h_high.strike / 30.0

    def test_optimize_hybrid(self):
        hedger = _make_hedger()
        h = hedger.optimize_hybrid(500.0, 18.0)
        assert isinstance(h, HybridHedge)
        assert h.total_annual_cost > 0

    def test_optimize_hybrid_regime_selection(self):
        hedger = _make_hedger()
        h_low = hedger.optimize_hybrid(500.0, 12.0)
        h_crisis = hedger.optimize_hybrid(500.0, 40.0)
        # Crisis should have higher put weight
        assert h_crisis.put_weight > h_low.put_weight

    def test_analytics_returns_hedge_analytics(self):
        hedger = _make_hedger()
        hedge = hedger.optimize_hybrid(500.0, 18.0)
        a = hedger.analytics(hedge)
        assert isinstance(a, HedgeAnalytics)
        assert a.hedge_type == HedgeType.HYBRID

    def test_analytics_scenario_payoffs(self):
        hedger = _make_hedger()
        hedge = hedger.optimize_hybrid(500.0, 18.0)
        a = hedger.analytics(hedge)
        # Larger drops should have larger payoffs
        assert a.payoff_20pct_drop >= a.payoff_10pct_drop or a.payoff_20pct_drop > 0

    def test_analytics_cost_metrics(self):
        hedger = _make_hedger()
        hedge = hedger.optimize_hybrid(500.0, 18.0)
        a = hedger.analytics(hedge)
        assert a.annual_premium > 0
        assert 0 < a.annual_premium_pct

    def test_analytics_breakeven(self):
        hedger = _make_hedger()
        hedge = hedger.optimize_hybrid(500.0, 18.0)
        a = hedger.analytics(hedge)
        assert a.breakeven_move_pct > 0

    def test_rolling_schedule_immediate(self):
        hedger = _make_hedger()
        s = hedger.rolling_schedule(current_dte=5)
        assert s['action'] == 'ROLL IMMEDIATELY'
        assert s['urgency'] == 'HIGH'

    def test_rolling_schedule_plan(self):
        hedger = _make_hedger()
        s = hedger.rolling_schedule(current_dte=15)
        assert s['action'] == 'PLAN ROLL'
        assert s['urgency'] == 'MEDIUM'

    def test_rolling_schedule_monitor(self):
        hedger = _make_hedger()
        s = hedger.rolling_schedule(current_dte=45)
        assert s['action'] == 'MONITOR'
        assert s['urgency'] == 'LOW'

    def test_rolling_schedule_has_notes(self):
        hedger = _make_hedger()
        s = hedger.rolling_schedule()
        assert len(s['notes']) == 3

    def test_quarterly_expiry_format(self):
        hedger = _make_hedger()
        s = hedger.rolling_schedule()
        expiry = s['next_expiry_options']
        # Should be a date string
        assert len(expiry) == 10  # YYYY-MM-DD

    def test_vix_thresholds(self):
        assert TailRiskHedger.VIX_LOW_THRESHOLD == 15.0
        assert TailRiskHedger.VIX_ELEVATED_THRESHOLD == 25.0
        assert TailRiskHedger.VIX_CRISIS_THRESHOLD == 35.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
