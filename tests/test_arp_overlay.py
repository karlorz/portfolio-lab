#!/usr/bin/env python3
"""
Tests for Alternative Risk Premia overlay — carry signals, value signals,
allocation adjustments, and signal integrator adapters.
"""
import sys
import os
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.signals.arp_overlay import (
    CarrySignal, ValueSignal,
    CarryCalculator, ValueCalculator,
    ARPOverlay, CarrySignalAdapter, ValueSignalAdapter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_carry_calc(risk_free=0.045):
    return CarryCalculator(risk_free_rate=risk_free)


def _make_value_calc():
    return ValueCalculator()


def _seed_carry_history(calc, symbol, n=100, base=0.02, seed=42):
    """Seed carry history with synthetic data."""
    np.random.seed(seed)
    calc.carry_history[symbol] = []
    for i in range(n):
        dt = datetime.now() - timedelta(days=n - i)
        val = base + np.random.normal(0, 0.005)
        calc.carry_history[symbol].append((dt, val))


def _seed_value_history(calc, symbol, metric='composite', n=100, base=0.05, seed=42):
    """Seed value history with synthetic data."""
    np.random.seed(seed)
    if symbol not in calc.value_history:
        calc.value_history[symbol] = {}
    calc.value_history[symbol][metric] = []
    for i in range(n):
        dt = datetime.now() - timedelta(days=n - i)
        val = base + np.random.normal(0, 0.01)
        calc.value_history[symbol][metric].append((dt, val))


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestCarrySignal:
    """Test CarrySignal dataclass."""

    def test_creation(self):
        sig = CarrySignal(
            symbol='TLT', carry_yield=2.0, percentile_1y=0.6,
            percentile_5y=0.55, signal_score=0.1, confidence=0.8,
            regime='high_carry',
        )
        assert sig.symbol == 'TLT'
        assert sig.regime == 'high_carry'

    def test_fields(self):
        sig = CarrySignal(
            symbol='SPY', carry_yield=1.5, percentile_1y=0.5,
            percentile_5y=0.5, signal_score=0.0, confidence=0.5,
            regime='neutral',
        )
        assert sig.carry_yield == 1.5
        assert sig.signal_score == 0.0


class TestValueSignal:
    """Test ValueSignal dataclass."""

    def test_creation(self):
        sig = ValueSignal(
            symbol='SPY', metric_type='composite', current_value=0.05,
            percentile_1y=0.4, percentile_5y=0.35, percentile_10y=0.30,
            signal_score=0.3, confidence=0.9, regime='cheap',
        )
        assert sig.symbol == 'SPY'
        assert sig.regime == 'cheap'

    def test_fields(self):
        sig = ValueSignal(
            symbol='EFA', metric_type='composite', current_value=0.08,
            percentile_1y=0.7, percentile_5y=0.65, percentile_10y=0.60,
            signal_score=-0.3, confidence=0.7, regime='expensive',
        )
        assert sig.percentile_10y == 0.60
        assert sig.signal_score == -0.3


# ---------------------------------------------------------------------------
# CarryCalculator tests
# ---------------------------------------------------------------------------

class TestCarryCalculator:
    """Test CarryCalculator."""

    def test_init(self):
        calc = _make_carry_calc()
        assert calc.risk_free_rate == 0.045
        assert calc.carry_history == {}

    def test_bond_carry_returns_carry_signal(self):
        calc = _make_carry_calc()
        sig = calc.calculate_bond_carry('TLT', yield_to_maturity=0.045)
        assert isinstance(sig, CarrySignal)
        assert sig.symbol == 'TLT'

    def test_bond_carry_real_yield(self):
        calc = _make_carry_calc()
        sig = calc.calculate_bond_carry('TLT', yield_to_maturity=0.05, inflation_expectation=0.025)
        # Real yield = 0.05 - 0.025 = 0.025, carry_yield = 2.5%
        assert sig.carry_yield == pytest.approx(2.5, abs=0.1)

    def test_bond_carry_regime_high(self):
        calc = _make_carry_calc()
        _seed_carry_history(calc, 'TLT', n=100, base=0.01)
        sig = calc.calculate_bond_carry('TLT', yield_to_maturity=0.06)
        # Very high yield should push to high percentile
        assert sig.regime in ['high_carry', 'neutral']

    def test_bond_carry_stores_history(self):
        calc = _make_carry_calc()
        calc.calculate_bond_carry('TLT', yield_to_maturity=0.045)
        assert 'TLT' in calc.carry_history
        assert len(calc.carry_history['TLT']) == 1

    def test_equity_carry_returns_carry_signal(self):
        calc = _make_carry_calc()
        sig = calc.calculate_equity_carry('SPY', dividend_yield=0.013, buyback_yield=0.018)
        assert isinstance(sig, CarrySignal)
        assert sig.symbol == 'SPY'

    def test_equity_carry_total(self):
        calc = _make_carry_calc(risk_free=0.04)
        sig = calc.calculate_equity_carry('SPY', dividend_yield=0.02, buyback_yield=0.02, earnings_growth=0.05)
        # Total carry = 0.02 + 0.02 + 0.05 - 0.04 = 0.05 → 5%
        assert sig.carry_yield == pytest.approx(5.0, abs=0.5)

    def test_gold_carry_negative(self):
        calc = _make_carry_calc()
        sig = calc.calculate_gold_carry('GLD', real_yield_10y=0.02, storage_cost=0.0025)
        # Gold carry = -0.02 - 0.0025 = -0.0225 → -2.25%
        assert sig.carry_yield < 0

    def test_gold_carry_inverted_signal(self):
        """Gold signal is inverted: low carry = buying opportunity."""
        calc = _make_carry_calc()
        _seed_carry_history(calc, 'GLD', n=100, base=-0.01)
        sig = calc.calculate_gold_carry('GLD', real_yield_10y=0.03)
        # Very negative carry → low percentile → high_carry regime
        assert sig.regime in ['high_carry', 'neutral', 'low_carry']

    def test_confidence_scales_with_history(self):
        calc = _make_carry_calc()
        sig = calc.calculate_bond_carry('TLT', yield_to_maturity=0.045)
        assert sig.confidence < 0.5  # Only 1 data point

        _seed_carry_history(calc, 'TLT', n=300)
        sig2 = calc.calculate_bond_carry('TLT', yield_to_maturity=0.045)
        assert sig2.confidence == 1.0  # 300+ data points

    def test_percentile_no_history(self):
        calc = _make_carry_calc()
        pctl = calc._percentile('NONEXISTENT', 0.05, days=365)
        assert pctl == 0.5

    def test_percentile_insufficient_data(self):
        calc = _make_carry_calc()
        calc.carry_history['TEST'] = [(datetime.now(), 0.01)]  # Only 1 point
        pctl = calc._percentile('TEST', 0.02, days=365)
        assert pctl == 0.5  # Default when < 30 points


# ---------------------------------------------------------------------------
# ValueCalculator tests
# ---------------------------------------------------------------------------

class TestValueCalculator:
    """Test ValueCalculator."""

    def test_init(self):
        calc = _make_value_calc()
        assert calc.value_history == {}

    def test_equity_value_returns_value_signal(self):
        calc = _make_value_calc()
        sig = calc.calculate_equity_value('SPY', pe_ratio=22.5, pb_ratio=4.1, dividend_yield=0.013)
        assert isinstance(sig, ValueSignal)
        assert sig.symbol == 'SPY'
        assert sig.metric_type == 'composite'

    def test_equity_value_composite_score(self):
        calc = _make_value_calc()
        sig = calc.calculate_equity_value('SPY', pe_ratio=20.0, pb_ratio=3.0, dividend_yield=0.02)
        # value_score = (1/20 + 1/3 + 0.02) / 3 ≈ (0.05 + 0.333 + 0.02) / 3 ≈ 0.134
        assert sig.current_value > 0

    def test_equity_value_cheap_regime(self):
        calc = _make_value_calc()
        _seed_value_history(calc, 'SPY', n=100, base=0.10)  # High base → current is cheap
        sig = calc.calculate_equity_value('SPY', pe_ratio=10.0, pb_ratio=1.0, dividend_yield=0.05)
        # Low P/E, low P/B = high value score
        assert sig.current_value > 0

    def test_equity_value_stores_history(self):
        calc = _make_value_calc()
        calc.calculate_equity_value('SPY', pe_ratio=22.5, pb_ratio=4.1, dividend_yield=0.013)
        assert 'SPY' in calc.value_history
        assert 'composite' in calc.value_history['SPY']

    def test_bond_value_returns_value_signal(self):
        calc = _make_value_calc()
        sig = calc.calculate_bond_value('TLT', real_yield=0.02)
        assert isinstance(sig, ValueSignal)
        assert sig.symbol == 'TLT'
        assert sig.metric_type == 'real_yield'

    def test_bond_value_high_yield_cheap(self):
        calc = _make_value_calc()
        _seed_value_history(calc, 'TLT', metric='real_yield', n=100, base=0.01)
        sig = calc.calculate_bond_value('TLT', real_yield=0.04)
        # High real yield = high percentile = cheap
        assert sig.regime in ['cheap', 'fair']

    def test_value_percentile_no_history(self):
        calc = _make_value_calc()
        pctl = calc._percentile('NONEXISTENT', 'composite', 0.05, days=365)
        assert pctl == 0.5

    def test_value_percentile_insufficient_data(self):
        calc = _make_value_calc()
        calc.value_history['TEST'] = {'composite': [(datetime.now(), 0.01)]}
        pctl = calc._percentile('TEST', 'composite', 0.02, days=365)
        assert pctl == 0.5

    def test_negative_pe_returns_zero_earnings_yield(self):
        calc = _make_value_calc()
        sig = calc.calculate_equity_value('SPY', pe_ratio=-10.0, pb_ratio=4.0, dividend_yield=0.01)
        assert sig.current_value >= 0  # earnings_yield clamped to 0


# ---------------------------------------------------------------------------
# ARPOverlay tests
# ---------------------------------------------------------------------------

class TestARPOverlay:
    """Test ARPOverlay."""

    def test_init(self):
        arp = ARPOverlay()
        assert arp.carry_calc.risk_free_rate == 0.045
        assert isinstance(arp.value_calc, ValueCalculator)

    def test_custom_risk_free(self):
        arp = ARPOverlay(risk_free_rate=0.03)
        assert arp.carry_calc.risk_free_rate == 0.03

    def test_generate_signals_returns_dict(self):
        arp = ARPOverlay()
        signals = arp.generate_signals()
        assert isinstance(signals, dict)

    def test_generate_signals_has_key_assets(self):
        arp = ARPOverlay()
        signals = arp.generate_signals()
        assert 'SPY' in signals
        assert 'TLT' in signals
        assert 'GLD' in signals
        assert 'EFA' in signals

    def test_signal_has_carry(self):
        arp = ARPOverlay()
        signals = arp.generate_signals()
        for symbol in ['SPY', 'TLT', 'GLD']:
            assert 'carry' in signals[symbol]
            assert signals[symbol]['carry'] is not None

    def test_gld_no_value_signal(self):
        arp = ARPOverlay()
        signals = arp.generate_signals()
        assert signals['GLD']['value'] is None

    def test_spy_has_value_signal(self):
        arp = ARPOverlay()
        signals = arp.generate_signals()
        assert signals['SPY']['value'] is not None

    def test_composite_score_bounded(self):
        arp = ARPOverlay()
        signals = arp.generate_signals()
        for symbol, sig in signals.items():
            assert -1.0 <= sig['composite_score'] <= 1.0

    def test_composite_confidence_bounded(self):
        arp = ARPOverlay()
        signals = arp.generate_signals()
        for symbol, sig in signals.items():
            assert 0.0 <= sig['composite_confidence'] <= 1.0

    def test_get_allocation_adjustments_returns_dict(self):
        arp = ARPOverlay()
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        adj = arp.get_allocation_adjustments(base)
        assert isinstance(adj, dict)
        assert 'SPY' in adj

    def test_adjustments_bounded(self):
        """Max adjustment is +/- 20% of base weight."""
        arp = ARPOverlay()
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        adj = arp.get_allocation_adjustments(base)
        for symbol, delta in adj.items():
            max_adj = base[symbol] * 0.2
            assert -max_adj <= delta <= max_adj

    def test_unknown_asset_zero_adjustment(self):
        arp = ARPOverlay()
        base = {'SPY': 0.46, 'UNKNOWN': 0.54}
        adj = arp.get_allocation_adjustments(base)
        assert adj['UNKNOWN'] == 0.0

    def test_get_signal_summary_returns_dict(self):
        arp = ARPOverlay()
        summary = arp.get_signal_summary()
        assert 'timestamp' in summary
        assert 'signals' in summary
        assert 'top_carry_trades' in summary
        assert 'top_value_trades' in summary

    def test_top_signals_sorted(self):
        arp = ARPOverlay()
        summary = arp.get_signal_summary()
        carry = summary['top_carry_trades']
        if len(carry) > 1:
            for i in range(len(carry) - 1):
                assert abs(carry[i]['score']) >= abs(carry[i+1]['score'])


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------

class TestCarrySignalAdapter:
    """Test CarrySignalAdapter."""

    def test_init(self):
        adapter = CarrySignalAdapter()
        assert adapter.source_name == 'aqr_carry_premium'

    def test_generate_signal_equity(self):
        adapter = CarrySignalAdapter()
        result = adapter.generate_signal('SPY')
        assert result is not None
        assert result.source_type == 'carry_arp'
        assert -1.0 <= result.signal <= 1.0

    def test_generate_signal_bond(self):
        adapter = CarrySignalAdapter()
        result = adapter.generate_signal('TLT')
        assert result is not None
        assert result.source_type == 'carry_arp'

    def test_generate_signal_gold(self):
        adapter = CarrySignalAdapter()
        result = adapter.generate_signal('GLD')
        assert result is not None

    def test_generate_signal_unknown_returns_none(self):
        adapter = CarrySignalAdapter()
        result = adapter.generate_signal('NONEXISTENT')
        assert result is None

    def test_confidence_bounded(self):
        adapter = CarrySignalAdapter()
        result = adapter.generate_signal('SPY')
        assert 0.0 <= result.confidence <= 1.0


class TestValueSignalAdapter:
    """Test ValueSignalAdapter."""

    def test_init(self):
        adapter = ValueSignalAdapter()
        assert adapter.source_name == 'aqr_value_premium'

    def test_generate_signal_equity(self):
        adapter = ValueSignalAdapter()
        result = adapter.generate_signal('SPY')
        assert result is not None
        assert result.source_type == 'value_arp'
        assert -1.0 <= result.signal <= 1.0

    def test_generate_signal_bond(self):
        adapter = ValueSignalAdapter()
        result = adapter.generate_signal('TLT')
        assert result is not None
        assert result.source_type == 'value_arp'

    def test_generate_signal_unknown_returns_none(self):
        adapter = ValueSignalAdapter()
        result = adapter.generate_signal('NONEXISTENT')
        assert result is None

    def test_confidence_bounded(self):
        adapter = ValueSignalAdapter()
        result = adapter.generate_signal('SPY')
        assert 0.0 <= result.confidence <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
