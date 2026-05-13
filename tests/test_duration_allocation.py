#!/usr/bin/env python3
"""
Tests for Duration Allocation Engine — leveraged ETF configs, regime classification,
base/leveraged allocation calculation, capital freed, expense drag, duration exposure,
risk scoring, and recommendation generation.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.strategy.duration_allocation import (
    LeveragedETFConfig, LEVERAGED_ETF_REGISTRY,
    DurationAllocationEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(tmp_path, portfolio_value=100000.0):
    """Create a DurationAllocationEngine with tmp state path."""
    with patch('src.strategy.duration_allocation.ALLOCATION_STATE_PATH', tmp_path / "state.json"):
        engine = DurationAllocationEngine(portfolio_value=portfolio_value)
    return engine


# ---------------------------------------------------------------------------
# LeveragedETFConfig tests
# ---------------------------------------------------------------------------

class TestLeveragedETFConfig:
    def test_creation(self):
        cfg = LeveragedETFConfig(
            symbol='TEST', leverage=2.0, expense_ratio=0.01,
            tracking_error=0.002, volatility_decay=0.01, max_portfolio_pct=0.10,
        )
        assert cfg.symbol == 'TEST'
        assert cfg.leverage == 2.0

    def test_fields(self):
        cfg = LeveragedETFConfig(
            symbol='X', leverage=3.0, expense_ratio=0.015,
            tracking_error=0.003, volatility_decay=0.02, max_portfolio_pct=0.05,
        )
        assert cfg.expense_ratio == 0.015
        assert cfg.max_portfolio_pct == 0.05


# ---------------------------------------------------------------------------
# LEVERAGED_ETF_REGISTRY tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_has_tlt(self):
        assert 'TLT' in LEVERAGED_ETF_REGISTRY

    def test_has_ubt(self):
        assert 'UBT' in LEVERAGED_ETF_REGISTRY

    def test_has_tmf(self):
        assert 'TMF' in LEVERAGED_ETF_REGISTRY

    def test_has_ief(self):
        assert 'IEF' in LEVERAGED_ETF_REGISTRY

    def test_tlt_leverage(self):
        assert LEVERAGED_ETF_REGISTRY['TLT'].leverage == 1.0

    def test_ubt_leverage(self):
        assert LEVERAGED_ETF_REGISTRY['UBT'].leverage == 2.0

    def test_tmf_leverage(self):
        assert LEVERAGED_ETF_REGISTRY['TMF'].leverage == 3.0

    def test_ubt_max_pct(self):
        assert LEVERAGED_ETF_REGISTRY['UBT'].max_portfolio_pct == 0.10

    def test_tmf_max_pct(self):
        assert LEVERAGED_ETF_REGISTRY['TMF'].max_portfolio_pct == 0.05


# ---------------------------------------------------------------------------
# DurationAllocationEngine tests
# ---------------------------------------------------------------------------

class TestDurationAllocationEngine:
    def test_init(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine.portfolio_value == 100000.0

    def test_init_default_state(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine.state['current_regime'] == 'unknown'

    # _classify_regime
    def test_classify_regime_steep(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine._classify_regime(150) == 'steep'

    def test_classify_regime_normal(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine._classify_regime(75) == 'normal'

    def test_classify_regime_flat(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine._classify_regime(25) == 'flat'

    def test_classify_regime_inverted(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine._classify_regime(-20) == 'inverted'

    def test_classify_regime_boundary_steep(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine._classify_regime(101) == 'steep'

    def test_classify_regime_boundary_normal(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine._classify_regime(51) == 'normal'

    def test_classify_regime_boundary_flat(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine._classify_regime(1) == 'flat'

    def test_classify_regime_boundary_inverted(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine._classify_regime(0) == 'inverted'

    # calculate_base_allocation
    def test_base_allocation_steep(self, tmp_path):
        engine = _make_engine(tmp_path)
        alloc = engine.calculate_base_allocation('steep')
        assert alloc['TLT'] == 0.70

    def test_base_allocation_normal(self, tmp_path):
        engine = _make_engine(tmp_path)
        alloc = engine.calculate_base_allocation('normal')
        assert alloc['TLT'] == 0.50

    def test_base_allocation_flat(self, tmp_path):
        engine = _make_engine(tmp_path)
        alloc = engine.calculate_base_allocation('flat')
        assert alloc['TLT'] == 0.30

    def test_base_allocation_inverted(self, tmp_path):
        engine = _make_engine(tmp_path)
        alloc = engine.calculate_base_allocation('inverted')
        assert alloc['TLT'] == 0.15

    def test_base_allocation_unknown_defaults_normal(self, tmp_path):
        engine = _make_engine(tmp_path)
        alloc = engine.calculate_base_allocation('nonexistent')
        assert alloc['TLT'] == 0.50

    def test_base_allocation_sums_to_one(self, tmp_path):
        engine = _make_engine(tmp_path)
        for regime in ['steep', 'normal', 'flat', 'inverted']:
            alloc = engine.calculate_base_allocation(regime)
            assert abs(sum(alloc.values()) - 1.0) < 0.001

    # calculate_leveraged_allocation
    def test_leveraged_allocation_none_preference(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine.calculate_leveraged_allocation('normal', 'none')
        assert result['leverage_used'] is False
        assert result['capital_freed'] == 0.0

    def test_leveraged_allocation_ubt_preference(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine.calculate_leveraged_allocation('normal', 'ubt')
        assert result['leverage_used'] is True
        assert 'UBT' in result['allocation']

    def test_leveraged_allocation_tmf_preference(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine.calculate_leveraged_allocation('normal', 'tmf')
        assert result['leverage_used'] is True
        assert 'TMF' in result['allocation']

    def test_leveraged_allocation_has_regime(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine.calculate_leveraged_allocation('steep', 'none')
        assert result['regime'] == 'steep'

    def test_leveraged_allocation_has_expense_drag(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine.calculate_leveraged_allocation('normal', 'none')
        assert 'expense_drag' in result

    def test_leveraged_allocation_has_duration_exposure(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine.calculate_leveraged_allocation('normal', 'none')
        assert 'duration_exposure' in result
        assert result['duration_exposure'] > 0

    def test_leveraged_allocation_risk_score(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine.calculate_leveraged_allocation('normal', 'ubt')
        assert 'risk_score' in result
        assert 0 <= result['risk_score'] <= 3.0

    def test_leveraged_allocation_capital_freed(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine.calculate_leveraged_allocation('normal', 'ubt')
        assert result['capital_freed'] >= 0

    def test_leveraged_allocation_sums_correctly(self, tmp_path):
        engine = _make_engine(tmp_path)
        for pref in ['none', 'ubt', 'tmf']:
            result = engine.calculate_leveraged_allocation('normal', pref)
            total = sum(result['allocation'].values())
            # Should sum to ~16% (the default portfolio_pct=1.0 * base allocation)
            # Actually portfolio_pct=1.0 means 100% of portfolio
            assert abs(total - 1.0) < 0.05

    def test_leveraged_allocation_custom_portfolio_pct(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine.calculate_leveraged_allocation('normal', 'none', portfolio_pct=0.16)
        total = sum(result['allocation'].values())
        assert abs(total - 0.16) < 0.01

    # _normalize_allocation
    def test_normalize_allocation(self, tmp_path):
        engine = _make_engine(tmp_path)
        alloc = {'A': 0.3, 'B': 0.7}
        result = engine._normalize_allocation(alloc, 0.5)
        assert abs(sum(result.values()) - 0.5) < 0.001

    def test_normalize_allocation_empty(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._normalize_allocation({}, 0.5)
        assert result == {}

    # _calculate_capital_freed
    def test_capital_freed_no_leverage(self, tmp_path):
        engine = _make_engine(tmp_path)
        base = {'TLT': 0.50}
        leveraged = {'TLT': 0.50}
        assert engine._calculate_capital_freed(base, leveraged) == 0.0

    def test_capital_freed_with_ubt(self, tmp_path):
        engine = _make_engine(tmp_path)
        base = {'TLT': 0.50}
        leveraged = {'UBT': 0.25, 'TLT': 0.0}
        freed = engine._calculate_capital_freed(base, leveraged)
        assert freed == pytest.approx(0.25, abs=0.01)

    # _calculate_expense_drag
    def test_expense_drag_empty(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine._calculate_expense_drag({}) == 0.0

    def test_expense_drag_tlt_only(self, tmp_path):
        engine = _make_engine(tmp_path)
        drag = engine._calculate_expense_drag({'TLT': 1.0})
        assert drag == pytest.approx(0.0015, abs=0.0001)

    def test_expense_drag_ubt_higher(self, tmp_path):
        engine = _make_engine(tmp_path)
        tlt_drag = engine._calculate_expense_drag({'TLT': 0.16})
        ubt_drag = engine._calculate_expense_drag({'UBT': 0.08})
        assert ubt_drag > tlt_drag

    # _estimate_volatility_decay
    def test_volatility_decay_tlt_zero(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine._estimate_volatility_decay({'TLT': 0.50}) == 0.0

    def test_volatility_decay_ubt_positive(self, tmp_path):
        engine = _make_engine(tmp_path)
        decay = engine._estimate_volatility_decay({'UBT': 0.10})
        assert decay > 0

    def test_volatility_decay_tmf_higher(self, tmp_path):
        engine = _make_engine(tmp_path)
        ubt_decay = engine._estimate_volatility_decay({'UBT': 0.05})
        tmf_decay = engine._estimate_volatility_decay({'TMF': 0.05})
        assert tmf_decay > ubt_decay

    # _calculate_duration_exposure
    def test_duration_exposure_tlt(self, tmp_path):
        engine = _make_engine(tmp_path)
        exposure = engine._calculate_duration_exposure({'TLT': 0.50})
        assert exposure == pytest.approx(0.50 * 18.5, abs=0.1)

    def test_duration_exposure_ief(self, tmp_path):
        engine = _make_engine(tmp_path)
        exposure = engine._calculate_duration_exposure({'IEF': 0.50})
        assert exposure == pytest.approx(0.50 * 7.5, abs=0.1)

    def test_duration_exposure_ubt_multiplier(self, tmp_path):
        engine = _make_engine(tmp_path)
        # UBT at 2x: same capital gives 2x exposure
        tlt_exp = engine._calculate_duration_exposure({'TLT': 0.10})
        ubt_exp = engine._calculate_duration_exposure({'UBT': 0.10})
        assert ubt_exp > tlt_exp

    # _calculate_risk_score
    def test_risk_score_no_leverage(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine._calculate_risk_score({'TLT': 0.50}) == 0.0

    def test_risk_score_ubt(self, tmp_path):
        engine = _make_engine(tmp_path)
        score = engine._calculate_risk_score({'UBT': 0.10})
        assert score == pytest.approx(1.0, abs=0.1)

    def test_risk_score_tmf(self, tmp_path):
        engine = _make_engine(tmp_path)
        score = engine._calculate_risk_score({'TMF': 0.05})
        assert score == pytest.approx(1.5, abs=0.1)

    def test_risk_score_capped_at_3(self, tmp_path):
        engine = _make_engine(tmp_path)
        score = engine._calculate_risk_score({'UBT': 0.50, 'TMF': 0.50})
        assert score == 3.0

    # generate_recommendation
    def test_generate_recommendation_returns_dict(self, tmp_path):
        engine = _make_engine(tmp_path)
        with patch.object(engine, 'get_yield_curve_regime', return_value='normal'):
            rec = engine.generate_recommendation()
        assert 'recommendation' in rec
        assert 'preferred_allocation' in rec

    def test_generate_recommendation_inverted_avoids_leverage(self, tmp_path):
        engine = _make_engine(tmp_path)
        with patch.object(engine, 'get_yield_curve_regime', return_value='inverted'):
            rec = engine.generate_recommendation()
        assert 'inverted' in rec['recommendation'].lower()

    def test_generate_recommendation_has_capital_deployment(self, tmp_path):
        engine = _make_engine(tmp_path)
        with patch.object(engine, 'get_yield_curve_regime', return_value='normal'):
            rec = engine.generate_recommendation()
        assert 'capital_deployment_options' in rec

    # _suggest_capital_deployment
    def test_suggest_capital_deployment_insufficient(self, tmp_path):
        engine = _make_engine(tmp_path)
        suggestions = engine._suggest_capital_deployment(0.005)
        assert suggestions[0]['action'] == 'insufficient_freed_capital'

    def test_suggest_capital_deployment_small(self, tmp_path):
        engine = _make_engine(tmp_path)
        suggestions = engine._suggest_capital_deployment(0.03)
        actions = [s['action'] for s in suggestions]
        assert 'increase_spy' in actions

    def test_suggest_capital_deployment_large(self, tmp_path):
        engine = _make_engine(tmp_path)
        suggestions = engine._suggest_capital_deployment(0.10)
        actions = [s['action'] for s in suggestions]
        assert 'add_ief_barbell' in actions

    # status
    def test_status_returns_dict(self, tmp_path):
        engine = _make_engine(tmp_path)
        with patch.object(engine, 'get_yield_curve_regime', return_value='normal'):
            s = engine.status()
        assert s['engine_version'] == '2.35'
        assert 'leveraged_etf_support' in s

    def test_status_available_etfs(self, tmp_path):
        engine = _make_engine(tmp_path)
        with patch.object(engine, 'get_yield_curve_regime', return_value='normal'):
            s = engine.status()
        assert 'TLT' in s['available_etfs']
        assert 'UBT' in s['available_etfs']

    # get_yield_curve_regime fallback
    def test_get_yield_curve_regime_no_files(self, tmp_path):
        engine = _make_engine(tmp_path)
        with patch('src.strategy.duration_allocation.DATA_DIR', tmp_path):
            with patch('src.strategy.duration_allocation.DB_PATH', tmp_path / "nope.db"):
                regime = engine.get_yield_curve_regime()
        assert regime == 'normal'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
