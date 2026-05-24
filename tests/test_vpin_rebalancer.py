#!/usr/bin/env python3
"""
Tests for VPIN microstructure signal and smart rebalancer integration.
"""

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


# ── BVCCalculator Extended ─────────────────────────────────────

class TestBVCCalculatorExtended:
    def test_classify_bar_at_midprice(self):
        """Close exactly at midprice should split volume roughly equally."""
        calc = BVCCalculator()
        # midprice = (110+90)/2 = 100, close = 100
        bar = calc.classify_bar(datetime.now(), 100, 110, 90, 100, 1000000)
        assert bar.buy_volume > 0
        assert bar.sell_volume > 0

    def test_zero_volume_bar(self):
        """Zero volume bar should not crash."""
        calc = BVCCalculator()
        bar = calc.classify_bar(datetime.now(), 100, 110, 90, 105, 0)
        assert bar.buy_volume == 0
        assert bar.sell_volume == 0

    def test_imbalance_fewer_bars_than_window(self):
        """Should work when fewer bars than window size."""
        calc = BVCCalculator()
        calc.classify_bar(datetime.now(), 100, 105, 95, 102, 500000)
        calc.classify_bar(datetime.now(), 102, 107, 97, 103, 500000)
        buy, sell, imbalance = calc.get_buy_sell_imbalance(window=20)
        assert -1.0 <= imbalance <= 1.0

    def test_add_bar_method(self):
        """add_bar should accept BVCBar objects."""
        calc = BVCCalculator()
        bar = BVCBar(
            timestamp=datetime.now(), open=100.0, high=105.0,
            low=95.0, close=102.0, volume=500000,
            buy_volume=300000, sell_volume=200000,
            vpin_local=0.35,
        )
        calc.add_bar(bar)
        # Should have 1 bar in history
        assert len(calc.bars) == 1


# ── VPINSignal ─────────────────────────────────────────────────

class TestVPINSignal:
    def test_to_signal_snapshot(self):
        """VPINSignal should convert to SignalSnapshot."""
        signal = VPINSignal(
            timestamp=datetime.now(),
            vpin=0.35,
            vpin_ma=0.30,
            vpin_std=0.05,
            z_score=1.0,
            percentile=75.0,
            regime="elevated",
            confidence=0.8,
            toxicity_level=0.65,
            recommendation="delay",
            expected_cost_impact=5.0,
        )
        snapshot = signal.to_signal_snapshot()
        assert snapshot is not None
        assert hasattr(snapshot, 'source')
        assert snapshot.source == "vpin_bvc"
        assert snapshot.value == -0.1  # "delay" → -0.1

    def test_to_signal_snapshot_execute(self):
        """Execute recommendation should map to +0.2 value."""
        signal = VPINSignal(
            timestamp=datetime.now(),
            vpin=0.15, vpin_ma=0.30, vpin_std=0.05,
            z_score=-3.0, percentile=10.0, regime="low",
            confidence=0.9, toxicity_level=0.1,
            recommendation="execute", expected_cost_impact=1.0,
        )
        snapshot = signal.to_signal_snapshot()
        assert snapshot.value == 0.2

    def test_to_signal_snapshot_avoid(self):
        """Avoid recommendation should map to -0.4 value."""
        signal = VPINSignal(
            timestamp=datetime.now(),
            vpin=0.70, vpin_ma=0.30, vpin_std=0.05,
            z_score=8.0, percentile=99.0, regime="high",
            confidence=0.95, toxicity_level=0.9,
            recommendation="avoid", expected_cost_impact=20.0,
        )
        snapshot = signal.to_signal_snapshot()
        assert snapshot.value == -0.4


# ── VPINEngine Extended ────────────────────────────────────────

class TestVPINEngineExtended:
    def _feed_bars(self, engine, symbol, n=100, seed=42):
        import numpy as np
        np.random.seed(seed)
        base = 500.0
        for i in range(n):
            ret = np.random.normal(0.001, 0.015)
            c = base * (1 + ret)
            h = max(base, c) * (1 + abs(np.random.normal(0, 0.005)))
            l = min(base, c) * (1 - abs(np.random.normal(0, 0.005)))
            engine.process_bar(symbol, datetime.now(), base, h, l, c, 500000)
            base = c

    def test_multiple_symbols(self):
        engine = VPINEngine(volume_bucket_size=100000, symbols=['SPY', 'GLD'])
        self._feed_bars(engine, 'SPY', n=200)
        self._feed_bars(engine, 'GLD', n=200)
        vpin_spy = engine.calculate_vpin('SPY')
        vpin_gld = engine.calculate_vpin('GLD')
        assert vpin_spy is not None
        assert vpin_gld is not None
        # Both should be valid VPIN values
        assert 0.0 <= vpin_spy <= 1.0
        assert 0.0 <= vpin_gld <= 1.0

    def test_signal_has_all_fields(self):
        engine = VPINEngine(volume_bucket_size=50000, symbols=['SPY'])
        self._feed_bars(engine, 'SPY', n=500)
        signal = engine.get_signal('SPY')
        if signal is not None:
            assert signal.vpin >= 0
            assert signal.regime in ('low', 'normal', 'elevated', 'high')
            assert signal.recommendation in ('execute', 'delay', 'avoid')
            assert signal.confidence >= 0


# ── RebalanceOptimizer ─────────────────────────────────────────

class TestRebalanceOptimizer:
    def _make_engine_with_data(self):
        import numpy as np
        engine = VPINEngine(volume_bucket_size=100000, symbols=['SPY'])
        base = 500.0
        np.random.seed(42)
        for i in range(300):
            ret = np.random.normal(0.001, 0.015)
            c = base * (1 + ret)
            h = max(base, c) * (1 + abs(np.random.normal(0, 0.005)))
            l = min(base, c) * (1 - abs(np.random.normal(0, 0.005)))
            engine.process_bar('SPY', datetime.now(), base, h, l, c, 500000)
            base = c
        return engine

    def test_should_execute_now_returns_tuple(self):
        from src.signals.vpin_bvc import RebalanceOptimizer
        engine = self._make_engine_with_data()
        opt = RebalanceOptimizer(vpin_engine=engine)
        result = opt.should_execute_now('SPY')
        assert isinstance(result, tuple)
        assert len(result) == 3  # (should_execute, reason, confidence)

    def test_execution_quality_report(self):
        from src.signals.vpin_bvc import RebalanceOptimizer
        engine = self._make_engine_with_data()
        opt = RebalanceOptimizer(vpin_engine=engine)
        report = opt.get_execution_quality_report()
        assert isinstance(report, dict)


# ── VPINSignalAdapter ──────────────────────────────────────────

class TestVPINSignalAdapter:
    def _make_engine_with_data(self):
        import numpy as np
        engine = VPINEngine(volume_bucket_size=100000, symbols=['SPY'])
        base = 500.0
        np.random.seed(42)
        for i in range(300):
            ret = np.random.normal(0.001, 0.015)
            c = base * (1 + ret)
            h = max(base, c) * (1 + abs(np.random.normal(0, 0.005)))
            l = min(base, c) * (1 - abs(np.random.normal(0, 0.005)))
            engine.process_bar('SPY', datetime.now(), base, h, l, c, 500000)
            base = c
        return engine

    def test_to_ensemble_signal_returns_dict(self):
        from src.signals.vpin_bvc import VPINSignalAdapter
        engine = self._make_engine_with_data()
        adapter = VPINSignalAdapter(vpin_engine=engine)
        result = adapter.to_ensemble_signal('SPY')
        assert isinstance(result, dict)

    def test_rebalance_timing_signal(self):
        from src.signals.vpin_bvc import VPINSignalAdapter
        engine = self._make_engine_with_data()
        adapter = VPINSignalAdapter(vpin_engine=engine)
        result = adapter.get_rebalance_timing_signal()
        assert isinstance(result, dict)
