#!/usr/bin/env python3
"""
Tests for Volatility Targeting Module — enums, configs, volatility estimators
(STD, EWMA, Parkinson, Yang-Zhang), regime classification, position sizing,
risk parity weights, simulation, and portfolio analysis.
"""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.strategy.vol_targeting import (
    VolMethod, TargetStrategy,
    VolMetrics, VolTargetConfig,
    VolatilityEngine, PortfolioVolTarget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(config=None):
    return VolatilityEngine(config)


def _make_returns(n=100, drift=0.0004, vol=0.01, seed=42):
    """Generate deterministic return series."""
    import random
    random.seed(seed)
    return [drift + random.gauss(0, vol) for _ in range(n)]


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestVolMethod:
    def test_values(self):
        assert VolMethod.STD.value == "standard"
        assert VolMethod.EWMA.value == "ewma"
        assert VolMethod.PARKINSON.value == "parkinson"
        assert VolMethod.YANG_ZHANG.value == "yang_zhang"
        assert VolMethod.GARCH.value == "garch"

    def test_members(self):
        assert len(VolMethod) == 5


class TestTargetStrategy:
    def test_values(self):
        assert TargetStrategy.FIXED.value == "fixed"
        assert TargetStrategy.REGIME_ADAPTIVE.value == "regime"
        assert TargetStrategy.RISK_PARITY.value == "risk_parity"
        assert TargetStrategy.DYNAMIC_RISK.value == "dynamic"

    def test_members(self):
        assert len(TargetStrategy) == 4


# ---------------------------------------------------------------------------
# VolTargetConfig tests
# ---------------------------------------------------------------------------

class TestVolTargetConfig:
    def test_defaults(self):
        c = VolTargetConfig()
        assert c.target_vol == 0.10
        assert c.max_leverage == 2.0
        assert c.min_leverage == 0.5
        assert c.lookback_days == 60
        assert c.ewma_lambda == 0.94

    def test_custom(self):
        c = VolTargetConfig(target_vol=0.12, max_leverage=1.5)
        assert c.target_vol == 0.12
        assert c.max_leverage == 1.5


# ---------------------------------------------------------------------------
# VolatilityEngine — std volatility
# ---------------------------------------------------------------------------

class TestStdVolatility:
    def test_empty(self):
        e = _make_engine()
        assert e.calculate_std_volatility([]) == 0.0

    def test_single(self):
        e = _make_engine()
        assert e.calculate_std_volatility([0.01]) == 0.0

    def test_positive(self):
        e = _make_engine()
        rets = _make_returns(100)
        vol = e.calculate_std_volatility(rets)
        assert vol > 0

    def test_annualized(self):
        e = _make_engine()
        rets = _make_returns(100, vol=0.01)
        vol = e.calculate_std_volatility(rets)
        # Annualized should be ~ daily * sqrt(252)
        assert vol > 0.05  # Reasonable range for 1% daily vol

    def test_constant_returns_zero_vol(self):
        e = _make_engine()
        vol = e.calculate_std_volatility([0.001] * 50)
        assert vol == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# VolatilityEngine — EWMA volatility
# ---------------------------------------------------------------------------

class TestEWMATolatility:
    def test_empty(self):
        e = _make_engine()
        assert e.calculate_ewma_volatility([]) == 0.0

    def test_single(self):
        e = _make_engine()
        assert e.calculate_ewma_volatility([0.01]) == 0.0

    def test_positive(self):
        e = _make_engine()
        rets = _make_returns(100)
        vol = e.calculate_ewma_volatility(rets)
        assert vol > 0

    def test_responds_faster_to_spikes(self):
        e = _make_engine()
        # Normal returns then a spike
        rets = [0.001] * 50 + [0.10] + [0.001] * 50
        vol = e.calculate_ewma_volatility(rets)
        assert vol > 0

    def test_custom_lambda(self):
        config = VolTargetConfig(ewma_lambda=0.90)
        e = _make_engine(config)
        rets = _make_returns(100)
        vol = e.calculate_ewma_volatility(rets)
        assert vol > 0


# ---------------------------------------------------------------------------
# VolatilityEngine — Parkinson volatility
# ---------------------------------------------------------------------------

class TestParkinsonVolatility:
    def test_empty(self):
        e = _make_engine()
        assert e.calculate_parkinson_volatility([], []) == 0.0

    def test_mismatched_lengths(self):
        e = _make_engine()
        assert e.calculate_parkinson_volatility([100], []) == 0.0

    def test_single(self):
        e = _make_engine()
        assert e.calculate_parkinson_volatility([100], [98]) == 0.0

    def test_positive(self):
        e = _make_engine()
        highs = [102, 103, 104, 105, 106]
        lows = [98, 99, 100, 101, 102]
        vol = e.calculate_parkinson_volatility(highs, lows)
        assert vol > 0

    def test_tight_range_low_vol(self):
        e = _make_engine()
        highs = [100.1, 100.2, 100.1, 100.2]
        lows = [99.9, 100.0, 99.9, 100.0]
        vol = e.calculate_parkinson_volatility(highs, lows)
        assert vol < 0.20  # Very low vol


# ---------------------------------------------------------------------------
# VolatilityEngine — Yang-Zhang volatility
# ---------------------------------------------------------------------------

class TestYangZhangVolatility:
    def test_empty(self):
        e = _make_engine()
        assert e.calculate_yang_zhang_volatility([], [], [], []) == 0.0

    def test_single(self):
        e = _make_engine()
        assert e.calculate_yang_zhang_volatility([100], [102], [98], [101]) == 0.0

    def test_positive(self):
        e = _make_engine()
        opens = [100, 101, 102, 103, 104]
        highs = [102, 103, 104, 105, 106]
        lows = [98, 99, 100, 101, 102]
        closes = [101, 102, 103, 104, 105]
        vol = e.calculate_yang_zhang_volatility(opens, highs, lows, closes)
        assert vol > 0


# ---------------------------------------------------------------------------
# VolatilityEngine — regime classification
# ---------------------------------------------------------------------------

class TestVolatilityRegime:
    def test_low(self):
        e = _make_engine()
        assert e.get_volatility_regime(0.08) == "low"

    def test_moderate(self):
        e = _make_engine()
        assert e.get_volatility_regime(0.12) == "moderate"

    def test_high(self):
        e = _make_engine()
        assert e.get_volatility_regime(0.20) == "high"

    def test_extreme(self):
        e = _make_engine()
        assert e.get_volatility_regime(0.35) == "extreme"

    def test_boundary_low(self):
        e = _make_engine()
        assert e.get_volatility_regime(0.10) == "moderate"

    def test_boundary_moderate(self):
        e = _make_engine()
        assert e.get_volatility_regime(0.15) == "high"

    def test_boundary_high(self):
        e = _make_engine()
        assert e.get_volatility_regime(0.25) == "extreme"


# ---------------------------------------------------------------------------
# VolatilityEngine — position sizing
# ---------------------------------------------------------------------------

class TestPositionSizing:
    def test_returns_dict(self):
        e = _make_engine()
        result = e.calculate_position_size(0.15)
        assert 'adjusted_leverage' in result
        assert 'target_exposure' in result

    def test_high_vol_deleverages(self):
        e = _make_engine()
        result = e.calculate_position_size(0.30)
        assert result['adjusted_leverage'] < 1.0

    def test_low_vol_leverages(self):
        e = _make_engine()
        result = e.calculate_position_size(0.05)
        assert result['adjusted_leverage'] > 1.0

    def test_max_leverage_capped(self):
        config = VolTargetConfig(max_leverage=1.5)
        e = _make_engine(config)
        result = e.calculate_position_size(0.01)
        assert result['adjusted_leverage'] == 1.5

    def test_min_leverage_floored(self):
        config = VolTargetConfig(min_leverage=0.5)
        e = _make_engine(config)
        result = e.calculate_position_size(1.0)
        assert result['adjusted_leverage'] == 0.5

    def test_zero_vol(self):
        e = _make_engine()
        result = e.calculate_position_size(0.0)
        assert result['adjusted_leverage'] == 1.0

    def test_custom_capital(self):
        e = _make_engine()
        result = e.calculate_position_size(0.10, capital=200000)
        assert result['target_exposure'] == pytest.approx(200000.0)

    def test_custom_target(self):
        e = _make_engine()
        result = e.calculate_position_size(0.20, target_vol=0.15)
        assert result['target_vol'] == 0.15


# ---------------------------------------------------------------------------
# VolatilityEngine — risk parity weights
# ---------------------------------------------------------------------------

class TestRiskParityWeights:
    def test_returns_list(self):
        e = _make_engine()
        weights = e.risk_parity_weights([('A', 0.10), ('B', 0.20)])
        assert isinstance(weights, list)

    def test_sum_to_one(self):
        e = _make_engine()
        weights = e.risk_parity_weights([('A', 0.10), ('B', 0.20), ('C', 0.15)])
        total = sum(w for _, w in weights)
        assert abs(total - 1.0) < 0.001

    def test_low_vol_higher_weight(self):
        e = _make_engine()
        weights = e.risk_parity_weights([('LowVol', 0.10), ('HighVol', 0.30)])
        weight_dict = dict(weights)
        assert weight_dict['LowVol'] > weight_dict['HighVol']

    def test_sorted_descending(self):
        e = _make_engine()
        weights = e.risk_parity_weights([('A', 0.10), ('B', 0.20), ('C', 0.15)])
        for i in range(len(weights) - 1):
            assert weights[i][1] >= weights[i + 1][1]


# ---------------------------------------------------------------------------
# VolatilityEngine — simulation
# ---------------------------------------------------------------------------

class TestSimulateVolTargeting:
    def test_returns_dict(self):
        e = _make_engine()
        vols = [0.12] * 100
        rets = _make_returns(100)
        result = e.simulate_vol_targeting(vols, rets)
        assert 'total_return_pct' in result
        assert 'sharpe_ratio' in result

    def test_final_value_positive(self):
        e = _make_engine()
        vols = [0.12] * 100
        rets = _make_returns(100, drift=0.001)
        result = e.simulate_vol_targeting(vols, rets)
        assert result['final_value'] > 0

    def test_max_drawdown_non_negative(self):
        e = _make_engine()
        vols = [0.12] * 100
        rets = _make_returns(100)
        result = e.simulate_vol_targeting(vols, rets)
        assert result['max_drawdown'] >= 0

    def test_realized_vol_positive(self):
        e = _make_engine()
        vols = [0.12] * 100
        rets = _make_returns(100)
        result = e.simulate_vol_targeting(vols, rets)
        assert result['realized_vol'] > 0


# ---------------------------------------------------------------------------
# PortfolioVolTarget tests
# ---------------------------------------------------------------------------

class TestPortfolioVolTarget:
    def test_analyze_portfolio_returns_dict(self):
        pt = PortfolioVolTarget()
        positions = [
            {'symbol': 'SPY', 'weight': 0.46, 'current_exposure': 46000},
            {'symbol': 'GLD', 'weight': 0.38, 'current_exposure': 38000},
            {'symbol': 'TLT', 'weight': 0.16, 'current_exposure': 16000},
        ]
        vols = [('SPY', 0.16), ('GLD', 0.14), ('TLT', 0.12)]
        result = pt.analyze_portfolio(positions, vols)
        assert 'current_portfolio_vol' in result
        assert 'risk_parity_weights' in result

    def test_analyze_portfolio_has_recommendations(self):
        pt = PortfolioVolTarget()
        positions = [
            {'symbol': 'SPY', 'weight': 0.46, 'current_exposure': 46000},
            {'symbol': 'GLD', 'weight': 0.38, 'current_exposure': 38000},
            {'symbol': 'TLT', 'weight': 0.16, 'current_exposure': 16000},
        ]
        vols = [('SPY', 0.16), ('GLD', 0.14), ('TLT', 0.12)]
        result = pt.analyze_portfolio(positions, vols)
        assert len(result['recommendations']) > 0

    def test_analyze_portfolio_rebalance_flag(self):
        pt = PortfolioVolTarget()
        positions = [
            {'symbol': 'SPY', 'weight': 0.46, 'current_exposure': 46000},
            {'symbol': 'GLD', 'weight': 0.38, 'current_exposure': 38000},
            {'symbol': 'TLT', 'weight': 0.16, 'current_exposure': 16000},
        ]
        vols = [('SPY', 0.16), ('GLD', 0.14), ('TLT', 0.12)]
        result = pt.analyze_portfolio(positions, vols)
        assert isinstance(result['rebalance_needed'], bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
