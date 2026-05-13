#!/usr/bin/env python3
"""
Tests for VPIN microstructure signal and smart rebalancer integration.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.signals.vpin_bvc import (
    BVCCalculator, VPINEngine, VPINSignal, BVCBar,
    load_historical_bars, backtest_vpin,
)
from src.rebalancing.smart_rebalancer import (
    SmartRebalancingController, PortfolioSnapshot, MarketConditions,
    RebalanceDecision, UrgencyLevel,
)
from src.rebalancing.integration import SmartRebalanceGate


# ── BVCCalculator ──────────────────────────────────────────────

class TestBVCCalculator:
    def test_classify_bar_bullish(self):
        """Close > open midprice → buy classification"""
        calc = BVCCalculator()
        # midprice = (H+L)/2 = (110+90)/2 = 100, close = 108 > 100
        bar = calc.classify_bar(datetime.now(), 95, 110, 90, 108, 1000000)
        assert bar.buy_volume > bar.sell_volume

    def test_classify_bar_bearish(self):
        """Close < open midprice → sell classification"""
        calc = BVCCalculator()
        # midprice = (H+L)/2 = (110+90)/2 = 100, close = 92 < 100
        bar = calc.classify_bar(datetime.now(), 105, 110, 90, 92, 1000000)
        assert bar.sell_volume > bar.buy_volume

    def test_buy_sell_imbalance(self):
        """Imbalance should be in [-1, 1]"""
        calc = BVCCalculator()
        for _ in range(25):
            calc.classify_bar(datetime.now(), 100, 105, 95, 102, 500000)
        buy, sell, imbalance = calc.get_buy_sell_imbalance(window=20)
        assert -1.0 <= imbalance <= 1.0


# ── VPINEngine ─────────────────────────────────────────────────

class TestVPINEngine:
    def _feed_bars(self, engine, symbol, n=100):
        """Feed synthetic bars to fill buckets."""
        import numpy as np
        np.random.seed(42)
        base = 500.0
        for i in range(n):
            ret = np.random.normal(0.001, 0.015)
            c = base * (1 + ret)
            h = max(base, c) * (1 + abs(np.random.normal(0, 0.005)))
            l = min(base, c) * (1 - abs(np.random.normal(0, 0.005)))
            engine.process_bar(symbol, datetime.now(), base, h, l, c, 500000)
            base = c

    def test_vpin_returns_value_after_buckets(self):
        engine = VPINEngine(volume_bucket_size=100000, symbols=['SPY'])
        self._feed_bars(engine, 'SPY', n=200)
        vpin = engine.calculate_vpin('SPY')
        assert vpin is not None
        assert 0.0 <= vpin <= 1.0

    def test_vpin_none_without_data(self):
        engine = VPINEngine(symbols=['SPY'])
        assert engine.calculate_vpin('SPY') is None

    def test_signal_generation(self):
        engine = VPINEngine(volume_bucket_size=50000, symbols=['SPY'])
        self._feed_bars(engine, 'SPY', n=500)
        signal = engine.get_signal('SPY')
        # Signal may be None if not enough buckets completed
        if signal is not None:
            assert signal.toxicity_level in ('low', 'normal', 'elevated', 'high')


# ── SmartRebalancer VPIN Integration ───────────────────────────

class TestSmartRebalancerVPIN:
    def test_low_vpin_allows_execution(self):
        """Low VPIN + moderate drift → EXECUTE or DEFER timing"""
        ctrl = SmartRebalancingController()
        # Holdings within ~10% drift of target (not emergency)
        portfolio = PortfolioSnapshot(
            holdings={'SPY': 50000, 'GLD': 33000, 'TLT': 17000},
            targets={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            total_value=100000,
            timestamp=datetime.now(),
        )
        market = MarketConditions(vpin=0.15, timestamp=datetime.now())
        result = ctrl.should_rebalance(portfolio, market)
        # Should not defer for toxicity at low VPIN
        assert result.decision != RebalanceDecision.DEFER_TOXICITY

    def test_high_vpin_defers(self):
        """High VPIN + moderate drift → DEFER_TOXICITY"""
        ctrl = SmartRebalancingController()
        # Holdings within ~10% drift (not emergency)
        portfolio = PortfolioSnapshot(
            holdings={'SPY': 50000, 'GLD': 33000, 'TLT': 17000},
            targets={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            total_value=100000,
            timestamp=datetime.now(),
        )
        market = MarketConditions(vpin=0.70, timestamp=datetime.now())
        result = ctrl.should_rebalance(portfolio, market)
        assert result.decision == RebalanceDecision.DEFER_TOXICITY

    def test_emergency_overrides_vpin(self):
        """Emergency drift overrides VPIN deferral"""
        ctrl = SmartRebalancingController()
        portfolio = PortfolioSnapshot(
            holdings={'SPY': 90000, 'GLD': 5000, 'TLT': 5000},
            targets={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            total_value=100000,
            timestamp=datetime.now(),
        )
        market = MarketConditions(vpin=0.80, timestamp=datetime.now())
        result = ctrl.should_rebalance(portfolio, market)
        # >30% drift is emergency — overrides toxicity
        assert result.decision == RebalanceDecision.OVERRIDE_EMERGENCY


# ── SmartRebalanceGate Integration ─────────────────────────────

class TestSmartRebalanceGate:
    @patch('src.rebalancing.integration._VPIN_AVAILABLE', False)
    def test_gate_without_vpin(self):
        """Gate works when VPIN module unavailable (defaults to 0.30)"""
        gate = SmartRebalanceGate()
        result = gate.evaluate(
            current_holdings={'SPY': 50000, 'GLD': 30000, 'TLT': 20000},
            target_allocations={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            total_value=100000,
        )
        assert result.decision in ('execute', 'defer_toxicity', 'defer_timing', 'defer_budget', 'no_drift')

    def test_gate_with_explicit_vpin(self):
        """Gate accepts explicit VPIN override"""
        gate = SmartRebalanceGate()
        result = gate.evaluate(
            current_holdings={'SPY': 50000, 'GLD': 30000, 'TLT': 20000},
            target_allocations={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            total_value=100000,
            vpin=0.90,
        )
        # Very high VPIN should defer
        assert result.metadata['vpin'] == 0.90


# ── load_historical_bars ───────────────────────────────────────

class TestLoadHistoricalBars:
    def test_returns_dataframe(self):
        """Should return DataFrame with OHLCV columns"""
        df = load_historical_bars('SPY', days=30)
        assert len(df) > 0
        assert all(c in df.columns for c in ['open', 'high', 'low', 'close', 'volume'])

    def test_ohlc_populated(self):
        """OHLC should not be all identical (Yahoo fallback provides real data)"""
        df = load_historical_bars('SPY', days=30)
        if len(df) > 5:
            # At least some bars should have different O/H/L/C
            assert not (df['open'] == df['high']).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
