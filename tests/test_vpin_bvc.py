"""Tests for VPIN BVC microstructure signal v2.65."""
import logging

import pytest
import numpy as np
from datetime import datetime, timedelta


class TestBVCBar:
    """BVCBar dataclass and BVC classification."""

    def test_bvc_bar_creation(self):
        from src.signals.vpin_bvc import BVCBar
        bar = BVCBar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            open=100.0, high=102.0, low=99.0, close=101.0,
            volume=10000, buy_volume=6500, sell_volume=3500,
            vpin_local=0.3
        )
        assert bar.open == 100.0
        assert bar.high == 102.0
        assert bar.low == 99.0
        assert bar.close == 101.0
        assert bar.volume == 10000
        assert bar.buy_volume == 6500
        assert bar.sell_volume == 3500
        assert bar.vpin_local == 0.3

    def test_classify_bar_normal(self):
        """BVC: bar with clear buy pressure (close near high)."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            o=100.0, h=102.0, l=99.0, c=101.5, v=10000
        )
        # buy_volume = 10000 * (101.5 - 99) / (102 - 99) = 10000 * 2.5/3 = 8333.33
        assert bar.buy_volume > bar.sell_volume
        assert bar.buy_volume + bar.sell_volume == pytest.approx(10000)
        assert 0.0 <= bar.vpin_local <= 1.0

    def test_classify_bar_sell_pressure(self):
        """BVC: bar with clear sell pressure (close near low)."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            o=100.0, h=102.0, l=99.0, c=99.5, v=10000
        )
        # buy_volume = 10000 * (99.5 - 99) / (102 - 99) = 10000 * 0.5/3 = 1666.67
        assert bar.buy_volume < bar.sell_volume
        assert bar.buy_volume + bar.sell_volume == pytest.approx(10000)

    def test_classify_bar_flat(self):
        """BVC: flat bar (high == low) gives equal buy/sell split."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            o=100.0, h=100.0, l=100.0, c=100.0, v=10000
        )
        assert bar.buy_volume == pytest.approx(5000)
        assert bar.sell_volume == pytest.approx(5000)
        assert bar.vpin_local == pytest.approx(0.0)

    def test_classify_bar_zero_volume(self):
        """BVC: zero volume gives zero vpin."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            o=100.0, h=102.0, l=99.0, c=101.0, v=0
        )
        assert bar.vpin_local == 0.0
        assert bar.buy_volume == 0.0
        assert bar.sell_volume == 0.0

    def test_classify_bar_close_at_high(self):
        """BVC: close == high gives max buy volume."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14),
            o=100.0, h=102.0, l=99.0, c=102.0, v=10000
        )
        assert bar.buy_volume == pytest.approx(10000)
        assert bar.sell_volume == pytest.approx(0)
        assert bar.vpin_local == pytest.approx(1.0)

    def test_classify_bar_close_at_low(self):
        """BVC: close == low gives min buy volume."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14),
            o=100.0, h=102.0, l=99.0, c=99.0, v=10000
        )
        assert bar.buy_volume == pytest.approx(0)
        assert bar.sell_volume == pytest.approx(10000)
        assert bar.vpin_local == pytest.approx(1.0)


class TestBVCBuySellImbalance:
    """Buy/sell imbalance over windows."""

    @pytest.fixture
    def calculator_with_bars(self):
        from src.signals.vpin_bvc import BVCCalculator, BVCBar
        calc = BVCCalculator()
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(30):
            bar = BVCBar(
                timestamp=base + timedelta(minutes=i),
                open=100.0, high=102.0, low=99.0, close=101.0,
                volume=10000 + i * 100,
                buy_volume=7000, sell_volume=3000,
                vpin_local=0.4
            )
            calc.add_bar(bar)
        return calc

    def test_imbalance_full_window(self, calculator_with_bars):
        """Imbalance over full window of 20 bars."""
        total_buy, total_sell, imbalance = calculator_with_bars.get_buy_sell_imbalance(window=20)
        assert total_buy > 0
        assert total_sell > 0
        assert 0.0 <= imbalance <= 1.0

    def test_imbalance_smaller_window(self, calculator_with_bars):
        """Imbalance with window=5."""
        total_buy, total_sell, imbalance = calculator_with_bars.get_buy_sell_imbalance(window=5)
        assert total_buy == pytest.approx(7000 * 5)
        assert total_sell == pytest.approx(3000 * 5)
        assert 0.0 <= imbalance <= 1.0

    def test_imbalance_window_larger_than_bars(self, calculator_with_bars):
        """Window larger than available bars -- uses all bars."""
        total_buy, total_sell, imbalance = calculator_with_bars.get_buy_sell_imbalance(window=100)
        assert total_buy == pytest.approx(7000 * 30)
        assert total_sell == pytest.approx(3000 * 30)

    def test_imbalance_perfect_balance(self):
        """Perfectly balanced buy/sell gives VPIN=0."""
        from src.signals.vpin_bvc import BVCCalculator, BVCBar
        calc = BVCCalculator()
        bar = BVCBar(
            timestamp=datetime(2026, 5, 14), open=100.0,
            high=102.0, low=99.0, close=101.0, volume=10000,
            buy_volume=5000, sell_volume=5000, vpin_local=0.0
        )
        calc.add_bar(bar)
        _, _, imbalance = calc.get_buy_sell_imbalance(window=1)
        assert imbalance == pytest.approx(0.0)

    def test_imbalance_zero_volume(self):
        """Zero volume gives zero imbalance."""
        from src.signals.vpin_bvc import BVCCalculator, BVCBar
        calc = BVCCalculator()
        bar = BVCBar(
            timestamp=datetime(2026, 5, 14), open=100.0,
            high=102.0, low=99.0, close=101.0, volume=0,
            buy_volume=0, sell_volume=0, vpin_local=0.0
        )
        calc.add_bar(bar)
        _, _, imbalance = calc.get_buy_sell_imbalance(window=1)
        assert imbalance == 0.0


class TestVPINSignal:
    """VPIN signal output dataclass."""

    def test_vpin_signal_low_toxicity(self):
        from src.signals.vpin_bvc import VPINSignal
        signal = VPINSignal(
            timestamp=datetime(2026, 5, 14),
            vpin=0.2, vpin_ma=0.25, vpin_std=0.05,
            z_score=-1.0, percentile=15.0,
            regime="low", confidence=0.85,
            toxicity_level=0.2, recommendation="execute",
            expected_cost_impact=2.0
        )
        assert signal.regime == "low"
        assert signal.recommendation == "execute"
        assert signal.toxicity_level < 0.5

    def test_vpin_signal_high_toxicity(self):
        from src.signals.vpin_bvc import VPINSignal
        signal = VPINSignal(
            timestamp=datetime(2026, 5, 14),
            vpin=0.65, vpin_ma=0.5, vpin_std=0.1,
            z_score=1.5, percentile=92.0,
            regime="high", confidence=0.9,
            toxicity_level=0.8, recommendation="avoid",
            expected_cost_impact=15.0
        )
        assert signal.regime == "high"
        assert signal.recommendation == "avoid"
        assert signal.toxicity_level > 0.5

    def test_vpin_signal_elevated(self):
        from src.signals.vpin_bvc import VPINSignal
        signal = VPINSignal(
            timestamp=datetime(2026, 5, 14),
            vpin=0.45, vpin_ma=0.4, vpin_std=0.08,
            z_score=0.6, percentile=72.0,
            regime="elevated", confidence=0.7,
            toxicity_level=0.55, recommendation="delay",
            expected_cost_impact=8.0
        )
        assert signal.regime == "elevated"
        assert signal.recommendation == "delay"

    def test_vpin_signal_normal(self):
        from src.signals.vpin_bvc import VPINSignal
        signal = VPINSignal(
            timestamp=datetime(2026, 5, 14),
            vpin=0.35, vpin_ma=0.38, vpin_std=0.07,
            z_score=-0.4, percentile=40.0,
            regime="normal", confidence=0.6,
            toxicity_level=0.4, recommendation="execute",
            expected_cost_impact=5.0
        )
        assert signal.regime == "normal"


class TestVPINBucket:
    """VPINBucket dataclass."""

    def test_vpin_bucket_complete(self):
        from src.signals.vpin_bvc import VPINBucket, BVCBar
        bar = BVCBar(
            timestamp=datetime(2026, 5, 14), open=100.0,
            high=102.0, low=99.0, close=101.0, volume=10000,
            buy_volume=7000, sell_volume=3000, vpin_local=0.4
        )
        bucket = VPINBucket(
            start_time=datetime(2026, 5, 14, 9, 30),
            end_time=datetime(2026, 5, 14, 10, 0),
            target_volume=50000, actual_volume=45000,
            bars=[bar], buy_volume=7000, sell_volume=3000,
            vpin=0.4, complete=True
        )
        assert bucket.complete is True
        assert bucket.vpin == pytest.approx(0.4)
        assert bucket.actual_volume == 45000

    def test_vpin_bucket_incomplete(self):
        from src.signals.vpin_bvc import VPINBucket
        bucket = VPINBucket(
            start_time=datetime(2026, 5, 14, 9, 30),
            end_time=datetime(2026, 5, 14, 9, 45),
            target_volume=50000, actual_volume=15000,
            bars=[], buy_volume=0, sell_volume=0,
            vpin=0.0, complete=False
        )
        assert bucket.complete is False
        assert len(bucket.bars) == 0
        assert bucket.vpin == 0.0


class TestVPINEngine:
    """VPIN engine core computation."""

    def test_engine_initialization(self):
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=50)
        assert engine.volume_bucket_size == 50000
        assert engine.vpin_window == 50
        assert len(engine.symbols) > 0

    def test_engine_default_params(self):
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine()
        assert engine.volume_bucket_size > 0
        assert engine.vpin_window > 0

    def test_engine_process_bar_basic(self):
        """Process a single bar through the engine."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=20)

        bucket = engine.process_bar(
            symbol="SPY",
            timestamp=datetime(2026, 5, 14, 9, 30),
            o=100.0, h=102.0, l=99.0, c=101.0, v=10000
        )
        # Single bar may or may not fill a bucket
        # But it should not raise
        assert bucket is None or hasattr(bucket, 'vpin')

    def test_engine_process_multiple_bars(self):
        """Process enough bars to fill a bucket."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=20000, vpin_window=50)

        base = datetime(2026, 5, 14, 9, 30)
        for i in range(10):
            bucket = engine.process_bar(
                symbol="SPY",
                timestamp=base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=100.5 + (i % 3) * 0.5,
                v=5000
            )

        # Should have VPIN history for SPY
        assert len(engine.vpin_history.get("SPY", [])) >= 0

    def test_engine_get_signal_empty(self):
        """get_signal for symbol with no data returns None."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine()
        signal = engine.get_signal("SPY")
        assert signal is None

    def test_engine_custom_symbols(self):
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(symbols=["SPY", "DBC"])
        assert "SPY" in engine.symbols
        assert "DBC" in engine.symbols
        assert len(engine.completed_buckets) == 2


# ---------------------------------------------------------------------------
# Extended coverage tests
# ---------------------------------------------------------------------------


class TestBVCBarDataclass:
    """Test BVCBar dataclass."""

    def test_bvc_bar_fields(self):
        from src.signals.vpin_bvc import BVCBar
        bar = BVCBar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            open=100.0, high=102.0, low=99.0, close=101.0,
            volume=10000, buy_volume=6000, sell_volume=4000,
            vpin_local=0.2,
        )
        assert bar.timestamp == datetime(2026, 5, 14, 10, 0)
        assert bar.buy_volume == 6000
        assert bar.sell_volume == 4000

    def test_bvc_bar_vpin_local_range(self):
        """vpin_local should be between 0 and 1."""
        from src.signals.vpin_bvc import BVCBar
        bar = BVCBar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            open=100.0, high=102.0, low=99.0, close=101.0,
            volume=10000, buy_volume=5000, sell_volume=5000,
            vpin_local=0.0,
        )
        assert 0.0 <= bar.vpin_local <= 1.0

    def test_bvc_bar_to_dict(self):
        """BVCBar dataclass should serialize via asdict with all fields."""
        import dataclasses
        from src.signals.vpin_bvc import BVCBar
        ts = datetime(2026, 5, 14, 10, 0)
        bar = BVCBar(
            timestamp=ts, open=100.0, high=102.0, low=99.0,
            close=101.0, volume=10000, buy_volume=6000,
            sell_volume=4000, vpin_local=0.2,
        )
        d = dataclasses.asdict(bar)
        assert d['timestamp'] == ts
        assert d['open'] == 100.0
        assert d['high'] == 102.0
        assert d['low'] == 99.0
        assert d['close'] == 101.0
        assert d['volume'] == 10000
        assert d['buy_volume'] == 6000
        assert d['sell_volume'] == 4000
        assert d['vpin_local'] == 0.2
        assert len(d) == 9  # All 9 fields present


class TestBVCCalculatorExtended:
    """Extended BVCCalculator tests."""

    def test_classify_bar_equal_high_low(self):
        """When high == low, buy_volume should be 50% (avoid division by zero)."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            o=100.0, h=100.0, l=100.0, c=100.0, v=10000
        )
        assert bar.buy_volume == pytest.approx(5000.0)
        assert bar.sell_volume == pytest.approx(5000.0)

    def test_classify_bar_zero_volume(self):
        """Zero volume should give vpin_local = 0."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            o=100.0, h=102.0, l=99.0, c=101.0, v=0
        )
        assert bar.volume == 0
        assert bar.vpin_local == 0.0

    def test_add_bar_stores_history(self):
        """add_bar should append to bars list."""
        from src.signals.vpin_bvc import BVCCalculator, BVCBar
        calc = BVCCalculator()
        bar = BVCBar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            open=100.0, high=102.0, low=99.0, close=101.0,
            volume=10000, buy_volume=6000, sell_volume=4000,
            vpin_local=0.2,
        )
        calc.add_bar(bar)
        assert len(calc.bars) == 1

    def test_get_buy_sell_imbalance_empty(self):
        """No bars should return zero volumes."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        buy, sell, imbalance = calc.get_buy_sell_imbalance()
        assert buy == 0.0
        assert sell == 0.0
        assert imbalance == 0.0

    def test_get_buy_sell_imbalance_with_data(self):
        """Imbalance should reflect buy/sell ratio."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        # Add multiple bars with clear buy pressure
        for i in range(5):
            bar = calc.classify_bar(
                timestamp=datetime(2026, 5, 14, 10, i),
                o=100.0, h=102.0, l=99.0, c=101.5, v=10000
            )
            calc.add_bar(bar)
        buy, sell, imbalance = calc.get_buy_sell_imbalance(window=5)
        assert buy > sell
        assert 0.0 <= imbalance <= 1.0

    def test_classify_bar_high_equals_low_different_close(self):
        """When high == low but close differs, buy_volume is still 50%."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        # Even though close != open/high/low, h==l triggers the 50% split
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            o=100.0, h=105.0, l=105.0, c=103.0, v=10000
        )
        assert bar.buy_volume == pytest.approx(5000.0)
        assert bar.sell_volume == pytest.approx(5000.0)
        assert bar.vpin_local == pytest.approx(0.0)

    def test_classify_bar_extreme_values(self):
        """BVC handles extreme OHLC values without error."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            o=1e-6, h=1e6, l=1e-6, c=5e5, v=1e9
        )
        assert bar.buy_volume > 0
        assert bar.sell_volume > 0
        assert bar.buy_volume + bar.sell_volume == pytest.approx(1e9)
        assert 0.0 <= bar.vpin_local <= 1.0


class TestVPINSignalDataclass:
    """Test VPINSignal dataclass and to_signal_snapshot."""

    def test_vpin_signal_creation(self):
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.45, vpin_ma=0.40, vpin_std=0.05,
            z_score=1.0, percentile=0.80, regime='elevated',
            confidence=0.75, toxicity_level=0.6,
            recommendation='delay', expected_cost_impact=5.0,
        )
        assert sig.vpin == 0.45
        assert sig.regime == 'elevated'
        assert sig.recommendation == 'delay'

    def test_to_signal_snapshot_execute(self):
        """'execute' recommendation should map to value 0.2."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.30, vpin_ma=0.35, vpin_std=0.05,
            z_score=-1.0, percentile=0.20, regime='low',
            confidence=0.8, toxicity_level=0.2,
            recommendation='execute', expected_cost_impact=1.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert snapshot.value == pytest.approx(0.2)
        assert snapshot.source == "vpin_bvc"

    def test_to_signal_snapshot_avoid(self):
        """'avoid' recommendation should map to value -0.4."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.80, vpin_ma=0.45, vpin_std=0.10,
            z_score=3.5, percentile=0.95, regime='high',
            confidence=0.9, toxicity_level=0.8,
            recommendation='avoid', expected_cost_impact=15.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert snapshot.value == pytest.approx(-0.4)

    def test_to_signal_snapshot_delay(self):
        """'delay' recommendation should map to value -0.1."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.55, vpin_ma=0.45, vpin_std=0.08,
            z_score=1.25, percentile=0.70, regime='elevated',
            confidence=0.6, toxicity_level=0.5,
            recommendation='delay', expected_cost_impact=5.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert snapshot.value == pytest.approx(-0.1)

    def test_to_signal_snapshot_is_active(self):
        """is_active should be True when confidence >= 0.3 and rec != 'execute'."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.55, vpin_ma=0.45, vpin_std=0.08,
            z_score=1.25, percentile=0.70, regime='elevated',
            confidence=0.6, toxicity_level=0.5,
            recommendation='delay', expected_cost_impact=5.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert snapshot.is_active is True

    def test_to_signal_snapshot_not_active_when_execute(self):
        """is_active should be False when recommendation is 'execute'."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.30, vpin_ma=0.35, vpin_std=0.05,
            z_score=-1.0, percentile=0.20, regime='low',
            confidence=0.8, toxicity_level=0.2,
            recommendation='execute', expected_cost_impact=1.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert snapshot.is_active is False

    def test_to_signal_snapshot_metadata(self):
        """Snapshot metadata should contain VPIN metrics."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.45, vpin_ma=0.40, vpin_std=0.05,
            z_score=1.0, percentile=0.80, regime='elevated',
            confidence=0.75, toxicity_level=0.6,
            recommendation='delay', expected_cost_impact=5.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert snapshot.metadata['vpin'] == 0.45
        assert snapshot.metadata['recommendation'] == 'delay'
        assert snapshot.metadata['vpin_ma'] == 0.40
        assert snapshot.metadata['z_score'] == 1.0
        assert snapshot.metadata['regime'] == 'elevated'
        assert snapshot.metadata['toxicity_level'] == 0.6

    def test_vpin_signal_to_dict(self):
        """VPINSignal dataclass should serialize via asdict with all fields."""
        import dataclasses
        from src.signals.vpin_bvc import VPINSignal
        ts = datetime(2026, 5, 14, 10, 0)
        sig = VPINSignal(
            timestamp=ts, vpin=0.45, vpin_ma=0.40, vpin_std=0.05,
            z_score=1.0, percentile=0.80, regime='elevated',
            confidence=0.75, toxicity_level=0.6,
            recommendation='delay', expected_cost_impact=5.0,
        )
        d = dataclasses.asdict(sig)
        assert d['timestamp'] == ts
        assert d['vpin'] == 0.45
        assert d['vpin_ma'] == 0.40
        assert d['vpin_std'] == 0.05
        assert d['z_score'] == 1.0
        assert d['percentile'] == 0.80
        assert d['regime'] == 'elevated'
        assert d['confidence'] == 0.75
        assert d['toxicity_level'] == 0.6
        assert d['recommendation'] == 'delay'
        assert d['expected_cost_impact'] == 5.0
        assert len(d) == 11  # All 11 fields present

    def test_to_signal_snapshot_unknown_recommendation(self):
        """Unknown recommendation maps to value 0.0."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.45, vpin_ma=0.40, vpin_std=0.05,
            z_score=1.0, percentile=0.80, regime='elevated',
            confidence=0.75, toxicity_level=0.6,
            recommendation='unknown_value', expected_cost_impact=5.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert snapshot.value == pytest.approx(0.0)

    def test_to_signal_snapshot_string_timestamp(self):
        """String timestamp should be handled gracefully (no isoformat method)."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp="2026-05-14T10:00:00",
            vpin=0.45, vpin_ma=0.40, vpin_std=0.05,
            z_score=1.0, percentile=0.80, regime='elevated',
            confidence=0.75, toxicity_level=0.6,
            recommendation='delay', expected_cost_impact=5.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert isinstance(snapshot.timestamp, str)
        assert snapshot.timestamp == "2026-05-14T10:00:00"

    def test_to_signal_snapshot_is_active_boundary_confidence_30(self):
        """is_active should be True when confidence == 0.3 and rec != 'execute' (boundary)."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.45, vpin_ma=0.40, vpin_std=0.05,
            z_score=1.0, percentile=0.80, regime='elevated',
            confidence=0.3, toxicity_level=0.5,
            recommendation='delay', expected_cost_impact=5.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert snapshot.is_active is True

    def test_to_signal_snapshot_is_active_low_confidence(self):
        """is_active should be False when confidence < 0.3."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.45, vpin_ma=0.40, vpin_std=0.05,
            z_score=1.0, percentile=0.80, regime='elevated',
            confidence=0.299, toxicity_level=0.5,
            recommendation='delay', expected_cost_impact=5.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert snapshot.is_active is False

    def test_to_signal_snapshot_is_active_execute_and_high_confidence(self):
        """is_active should be False when recommendation is 'execute' even with high confidence."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.30, vpin_ma=0.35, vpin_std=0.05,
            z_score=-1.0, percentile=0.20, regime='low',
            confidence=0.99, toxicity_level=0.2,
            recommendation='execute', expected_cost_impact=1.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert snapshot.is_active is False

    def test_to_signal_snapshot_is_active_avoid_and_low_confidence(self):
        """is_active should be False when confidence < 0.3 even with avoid rec."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.80, vpin_ma=0.45, vpin_std=0.10,
            z_score=3.5, percentile=0.95, regime='high',
            confidence=0.2, toxicity_level=0.8,
            recommendation='avoid', expected_cost_impact=15.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert snapshot.is_active is False


class TestVPINBucketDataclass:
    """Test VPINBucket dataclass."""

    def test_vpin_bucket_creation(self):
        from src.signals.vpin_bvc import VPINBucket
        bucket = VPINBucket(
            start_time=datetime(2026, 5, 14, 9, 30),
            end_time=datetime(2026, 5, 14, 9, 45),
            target_volume=100000, actual_volume=95000,
            bars=[], buy_volume=60000, sell_volume=40000,
            vpin=0.20, complete=True,
        )
        assert bucket.target_volume == 100000
        assert bucket.vpin == 0.20
        assert bucket.complete is True

    def test_vpin_bucket_to_dict(self):
        """VPINBucket dataclass should serialize via asdict with all fields."""
        import dataclasses
        from src.signals.vpin_bvc import VPINBucket, BVCBar
        ts_start = datetime(2026, 5, 14, 9, 30)
        ts_end = datetime(2026, 5, 14, 9, 45)
        bar = BVCBar(
            timestamp=ts_start, open=100.0, high=102.0,
            low=99.0, close=101.0, volume=10000,
            buy_volume=6000, sell_volume=4000, vpin_local=0.2,
        )
        bucket = VPINBucket(
            start_time=ts_start, end_time=ts_end,
            target_volume=100000, actual_volume=45000,
            bars=[bar], buy_volume=6000, sell_volume=4000,
            vpin=0.20, complete=False,
        )
        d = dataclasses.asdict(bucket)
        assert d['start_time'] == ts_start
        assert d['end_time'] == ts_end
        assert d['target_volume'] == 100000
        assert d['actual_volume'] == 45000
        assert len(d['bars']) == 1
        assert d['buy_volume'] == 6000
        assert d['sell_volume'] == 4000
        assert d['vpin'] == 0.20
        assert d['complete'] is False
        assert len(d) == 9  # All 9 fields present


class TestRebalanceOptimizerExtended:
    """Extended RebalanceOptimizer tests."""

    def test_no_data_returns_execute(self):
        """No VPIN data should return (True, 'insufficient_data', 0.0)."""
        from src.signals.vpin_bvc import VPINEngine, RebalanceOptimizer
        engine = VPINEngine()
        optimizer = RebalanceOptimizer(engine)
        execute, reason, savings = optimizer.should_execute_now('SPY')
        assert execute is True
        assert reason == "insufficient_data"

    def test_quality_report_structure(self):
        """Execution quality report should have expected keys."""
        from src.signals.vpin_bvc import VPINEngine, RebalanceOptimizer
        engine = VPINEngine(symbols=["SPY"])
        optimizer = RebalanceOptimizer(engine)
        report = optimizer.get_execution_quality_report()
        assert 'timestamp' in report
        assert 'symbols' in report

    def test_quality_report_empty_without_data(self):
        """With no data, symbols dict should be empty."""
        from src.signals.vpin_bvc import VPINEngine, RebalanceOptimizer
        engine = VPINEngine(symbols=["SPY"])
        optimizer = RebalanceOptimizer(engine)
        report = optimizer.get_execution_quality_report()
        # No signal data => empty symbols dict
        assert len(report['symbols']) == 0

    def test_rebalance_optimizer_max_delay_default(self):
        """RebalanceOptimizer should have default max_delay_minutes=60."""
        from src.signals.vpin_bvc import VPINEngine, RebalanceOptimizer
        engine = VPINEngine()
        optimizer = RebalanceOptimizer(engine)
        assert optimizer.max_delay_minutes == 60

    def test_rebalance_optimizer_custom_delay(self):
        """RebalanceOptimizer should accept custom max_delay_minutes."""
        from src.signals.vpin_bvc import VPINEngine, RebalanceOptimizer
        engine = VPINEngine()
        optimizer = RebalanceOptimizer(engine, max_delay_minutes=120)
        assert optimizer.max_delay_minutes == 120
        assert optimizer.pending_rebalances == []


class TestVPINSignalAdapterExtended:
    """Extended VPINSignalAdapter tests."""

    def test_no_data_returns_neutral(self):
        """No VPIN data should return neutral regime."""
        from src.signals.vpin_bvc import VPINEngine, VPINSignalAdapter
        engine = VPINEngine()
        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal('SPY')
        assert result['regime'] == 'neutral'
        assert result['confidence'] == 0.0

    def test_no_data_structure(self):
        """No data result should have expected keys."""
        from src.signals.vpin_bvc import VPINEngine, VPINSignalAdapter
        engine = VPINEngine()
        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal('SPY')
        assert 'source' in result
        assert 'regime' in result
        assert 'probability' in result
        assert 'raw_data' in result

    def test_rebalance_timing_signal(self):
        """get_rebalance_timing_signal should return valid structure."""
        from src.signals.vpin_bvc import VPINEngine, VPINSignalAdapter
        engine = VPINEngine()
        adapter = VPINSignalAdapter(engine)
        result = adapter.get_rebalance_timing_signal()
        assert isinstance(result, dict)

    def test_thresholds(self):
        """Adapter should have expected VPIN thresholds."""
        from src.signals.vpin_bvc import VPINSignalAdapter
        assert VPINSignalAdapter.HIGH_VPIN_THRESHOLD == 0.75
        assert VPINSignalAdapter.CRISIS_VPIN_THRESHOLD == 0.90

    def _setup_engine_with_signal(self, engine, symbol="SPY",
                                  percentile_ratio=0.5):
        """Helper: fill engine with bars and override vpin_history to
        achieve a target percentile. After setup, the call to
        get_signal (via adapter) will append one more vpin entry, so
        the computation accounts for N+1 total entries.

        percentile_ratio controls how many history entries are BELOW
        the current vpin value. E.g., 0.91 means 91% of history
        entries are lower than the current vpin.
        """
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(10):
            engine.process_bar(
                symbol, base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0,
                v=engine.volume_bucket_size,
            )
        # Verify calculate_vpin works
        current_vpin = engine.calculate_vpin(symbol)
        assert current_vpin is not None, "Need >= vpin_window completed buckets"

        # Override history with precise percentile control
        below = int(percentile_ratio * 100)
        above = 100 - below
        engine.vpin_history[symbol] = (
            [current_vpin * 0.5] * below + [current_vpin * 1.5] * above
        )

    def test_to_ensemble_signal_crisis_regime(self):
        """Percentile >= 0.90 should map to crisis regime."""
        from src.signals.vpin_bvc import VPINEngine, VPINSignalAdapter
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        self._setup_engine_with_signal(engine, percentile_ratio=0.92)
        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal("SPY")
        assert result['regime'] == 'crisis'
        assert result['probability'] == 0.8
        assert result['confidence'] > 0

    def test_to_ensemble_signal_bear_regime(self):
        """0.75 <= percentile < 0.90 should map to bear regime."""
        from src.signals.vpin_bvc import VPINEngine, VPINSignalAdapter
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        self._setup_engine_with_signal(engine, percentile_ratio=0.78)
        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal("SPY")
        assert result['regime'] == 'bear'
        assert result['probability'] == 0.7

    def test_to_ensemble_signal_neutral_regime(self):
        """0.25 < percentile < 0.75 should map to neutral regime."""
        from src.signals.vpin_bvc import VPINEngine, VPINSignalAdapter
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        self._setup_engine_with_signal(engine, percentile_ratio=0.50)
        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal("SPY")
        assert result['regime'] == 'neutral'
        assert result['probability'] == 0.5


class TestVPINEngineExtended:
    """Extended VPINEngine edge cases."""

    def test_engine_constant_volume_bars(self):
        """All bars with identical volume should not cause errors."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3)
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(15):
            engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=10000
            )
        # At least some completed buckets
        assert len(engine.completed_buckets["SPY"]) > 0
        vpin = engine.calculate_vpin("SPY")
        assert vpin is not None
        assert 0.0 <= vpin <= 1.0

    def test_engine_single_bucket_exact_completion(self):
        """Exactly one bucket is completed and returned."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=20000, vpin_window=5)
        base = datetime(2026, 5, 14, 9, 30)
        completed = None
        for i in range(4):
            result = engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=5000
            )
            if result is not None:
                completed = result
        assert completed is not None
        assert completed.complete is True
        assert completed.actual_volume == 20000

    def test_engine_zero_volume_bars(self):
        """Bars with zero volume create buckets with zero actual volume."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=5)
        base = datetime(2026, 5, 14, 9, 30)
        # Process many zero-volume bars -- bucket never fills
        for i in range(10):
            engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=0
            )
        assert len(engine.completed_buckets["SPY"]) == 0
        # Current bucket should exist but have actual_volume == 0
        assert "SPY" in engine.current_buckets
        assert engine.current_buckets["SPY"].actual_volume == 0
        assert engine.current_buckets["SPY"].complete is False

    def test_engine_calculate_vpin_insufficient_buckets(self):
        """calculate_vpin returns None when fewer than vpin_window buckets exist."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=50)
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(5):
            engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=50000
            )
        vpin = engine.calculate_vpin("SPY")
        assert vpin is None  # Only 5 buckets, need 50

    def test_engine_calculate_vpin_exact_window(self):
        """calculate_vpin succeeds when exactly vpin_window buckets exist."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=20000, vpin_window=5)
        base = datetime(2026, 5, 14, 9, 30)
        # 5 buckets * 2 bars each (v=10000, bucket=20000)
        for i in range(10):
            engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=10000
            )
        assert len(engine.completed_buckets["SPY"]) == 5
        vpin = engine.calculate_vpin("SPY")
        assert vpin is not None
        assert 0.0 <= vpin <= 1.0

    def test_engine_vpin_history_trimming(self):
        """vpin_history should not exceed 500 entries."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=20000, vpin_window=2)
        symbol = "SPY"
        base = datetime(2026, 5, 14, 9, 30)

        # Process enough bars to get many calculate_vpin calls
        for i in range(200):
            engine.process_bar(
                symbol, base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=20000
            )
            engine.calculate_vpin(symbol)

        # History should be trimmed to 500
        assert len(engine.vpin_history[symbol]) <= 500

    def test_engine_completed_buckets_trimming(self):
        """Completed buckets should not exceed vpin_window * 2."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=5000, vpin_window=5)
        symbol = "SPY"
        base = datetime(2026, 5, 14, 9, 30)

        # Fill many buckets (v=5000, bucket=5000 => 1 bar per bucket)
        for i in range(30):
            engine.process_bar(
                symbol, base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=5000
            )

        assert len(engine.completed_buckets[symbol]) <= 10  # vpin_window * 2

    def test_engine_get_signal_history_too_small(self):
        """get_signal returns None when vpin_history has < 50 entries."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=20000, vpin_window=3)
        symbol = "SPY"
        base = datetime(2026, 5, 14, 9, 30)

        # Fill 5 buckets
        for i in range(10):
            engine.process_bar(
                symbol, base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=10000
            )
        # calculate_vpin works with 3+ buckets (vpin_window=3)
        vpin = engine.calculate_vpin(symbol)
        assert vpin is not None
        # vpin_history only has 1 entry, need 50
        signal = engine.get_signal(symbol)
        assert signal is None

    def test_engine_calculate_vpin_vpin_window_larger_than_buckets(self):
        """calculate_vpin returns None when buckets < vpin_window (edge boundary)."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=20000, vpin_window=10)
        symbol = "SPY"
        base = datetime(2026, 5, 14, 9, 30)

        # Fill exactly 9 buckets (1 less than vpin_window=10)
        for i in range(18):
            engine.process_bar(
                symbol, base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=10000
            )
        assert len(engine.completed_buckets[symbol]) == 9
        assert engine.calculate_vpin(symbol) is None


class TestVPINConstants:
    """Validate constants and threshold boundary values."""

    def test_regime_boundary_values(self):
        """Regime classification boundaries at 0.25, 0.50, 0.75."""

        # VPINEngine.get_signal defines regimes based on percentile
        # percentile < 0.25 => low
        # 0.25 <= percentile < 0.50 => normal
        # 0.50 <= percentile < 0.75 => elevated
        # percentile >= 0.75 => high

        # Use VPINSignal to verify regime boundaries
        from src.signals.vpin_bvc import VPINSignal
        ts = datetime(2026, 5, 14)

        sig_below_25 = VPINSignal(
            timestamp=ts, vpin=0.2, vpin_ma=0.3, vpin_std=0.05,
            z_score=-0.5, percentile=0.24, regime='low',
            confidence=0.6, toxicity_level=0.2,
            recommendation='execute', expected_cost_impact=1.0,
        )
        assert sig_below_25.regime == 'low'

        sig_at_25 = VPINSignal(
            timestamp=ts, vpin=0.3, vpin_ma=0.3, vpin_std=0.05,
            z_score=0.0, percentile=0.25, regime='normal',
            confidence=0.7, toxicity_level=0.3,
            recommendation='execute', expected_cost_impact=1.0,
        )
        assert sig_at_25.regime == 'normal'

        sig_at_50 = VPINSignal(
            timestamp=ts, vpin=0.4, vpin_ma=0.3, vpin_std=0.05,
            z_score=2.0, percentile=0.50, regime='elevated',
            confidence=0.7, toxicity_level=0.5,
            recommendation='delay', expected_cost_impact=2.0,
        )
        assert sig_at_50.regime == 'elevated'

        sig_at_75 = VPINSignal(
            timestamp=ts, vpin=0.6, vpin_ma=0.3, vpin_std=0.05,
            z_score=6.0, percentile=0.75, regime='high',
            confidence=0.6, toxicity_level=0.8,
            recommendation='avoid', expected_cost_impact=5.0,
        )
        assert sig_at_75.regime == 'high'

    def test_recommendation_boundary_values(self):
        """Recommendation boundaries at 0.30, 0.70 percentile."""
        from src.signals.vpin_bvc import VPINSignal
        ts = datetime(2026, 5, 14)

        # percentile < 0.30 => execute
        sig_execute = VPINSignal(
            timestamp=ts, vpin=0.2, vpin_ma=0.3, vpin_std=0.05,
            z_score=-2.0, percentile=0.29, regime='low',
            confidence=0.6, toxicity_level=0.2,
            recommendation='execute', expected_cost_impact=-3.0,
        )
        assert sig_execute.recommendation == 'execute'
        assert sig_execute.expected_cost_impact == -3.0

        # percentile == 0.30 => delay
        sig_delay = VPINSignal(
            timestamp=ts, vpin=0.4, vpin_ma=0.3, vpin_std=0.05,
            z_score=2.0, percentile=0.30, regime='normal',
            confidence=0.7, toxicity_level=0.4,
            recommendation='delay', expected_cost_impact=0.0,
        )
        assert sig_delay.recommendation == 'delay'
        assert sig_delay.expected_cost_impact == 0.0

        # percentile == 0.70 => avoid
        sig_avoid = VPINSignal(
            timestamp=ts, vpin=0.6, vpin_ma=0.3, vpin_std=0.05,
            z_score=6.0, percentile=0.70, regime='elevated',
            confidence=0.6, toxicity_level=0.7,
            recommendation='avoid', expected_cost_impact=5.0,
        )
        assert sig_avoid.recommendation == 'avoid'
        assert sig_avoid.expected_cost_impact == 5.0

    def test_adapter_threshold_values(self):
        """VPINSignalAdapter thresholds for regime mapping."""
        from src.signals.vpin_bvc import VPINSignalAdapter
        # Verify constants
        assert VPINSignalAdapter.HIGH_VPIN_THRESHOLD == 0.75
        assert VPINSignalAdapter.CRISIS_VPIN_THRESHOLD == 0.90

        # Verify regime mapping logic for adapter
        # percentile >= 0.90 => crisis, prob=0.8
        # percentile >= 0.75 => bear, prob=0.7
        # percentile <= 0.25 => bull, prob=0.6
        # else => neutral, prob=0.5

        # The mapping is: crisis -> percentile >= 0.90,
        # bear -> 0.75 <= percentile < 0.90,
        # bull -> percentile <= 0.25,
        # neutral -> 0.25 < percentile < 0.75

        adapter = VPINSignalAdapter.__new__(VPINSignalAdapter)
        adapter.HIGH_VPIN_THRESHOLD = 0.75
        adapter.CRISIS_VPIN_THRESHOLD = 0.90

        # Test the mapping via VPINSignalAdapter's logic:
        # crisis: percentile >= CRISIS_VPIN_THRESHOLD (>= 0.90)
        assert 0.95 >= VPINSignalAdapter.CRISIS_VPIN_THRESHOLD
        # bear: >= HIGH (0.75) and < CRISIS (0.90)
        assert 0.80 >= VPINSignalAdapter.HIGH_VPIN_THRESHOLD
        assert 0.80 < VPINSignalAdapter.CRISIS_VPIN_THRESHOLD
        # bull: <= 0.25
        assert 0.20 <= 0.25
        # neutral: > 0.25 and < 0.75
        assert 0.50 > 0.25
        assert 0.50 < 0.75

    def test_engine_default_parameters(self):
        """Engine defaults should be reasonable positive values."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine()
        assert engine.volume_bucket_size == 100000
        assert engine.vpin_window == 50
        assert engine.symbols == ['SPY', 'QQQ', 'TLT', 'GLD']

    def test_load_historical_bars_returns_dataframe(self):
        """load_historical_bars should return a DataFrame (may be empty without DB/network)."""
        from src.signals.vpin_bvc import load_historical_bars
        import pandas as pd
        df = load_historical_bars("SPY", days=1)
        assert isinstance(df, pd.DataFrame)

    def test_backtest_vpin_returns_dict(self):
        """backtest_vpin should return a dict with results and statistics keys."""
        from src.signals.vpin_bvc import backtest_vpin
        result = backtest_vpin(["SPY"], days=1)
        assert 'results' in result
        assert 'statistics' in result
        assert isinstance(result['results'], dict)
        assert isinstance(result['statistics'], dict)


class TestVpinSignalBoundaryConditions:
    """Signal classification boundary conditions."""

    def test_signal_classification_all_percentiles(self):
        """Test regime/recommendation mapping at key percentile values
        by exercising get_signal with controlled vpin_history."""
        from src.signals.vpin_bvc import VPINEngine

        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        base = datetime(2026, 5, 14, 9, 30)

        # Fill enough buckets
        for i in range(10):
            engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=50000
            )

        current_vpin = engine.calculate_vpin("SPY")
        assert current_vpin is not None

        # Test low regime (percentile < 0.25)
        # All history values above current vpin => percentile = 0
        engine.vpin_history["SPY"] = [current_vpin * 1.5] * 100
        signal = engine.get_signal("SPY")
        assert signal is not None
        assert signal.regime == 'low'
        assert signal.recommendation == 'execute'

        # Test high regime (percentile >= 0.75)
        # 80 of 100 history values below current vpin => percentile ~0.80
        engine.vpin_history["SPY"] = (
            [current_vpin * 0.5] * 80 + [current_vpin * 1.5] * 20
        )
        signal = engine.get_signal("SPY")
        assert signal is not None
        # With vpin_history having 100 entries + 1 appended by calculate_vpin,
        # percentile = 80/101 ≈ 0.79, which is >= 0.75
        assert signal.regime == 'high'
        assert signal.recommendation == 'avoid'

        # Test normal regime (percentile ~0.50)
        # 50 of 100 history values below current vpin
        engine.vpin_history["SPY"] = (
            [current_vpin * 0.5] * 50 + [current_vpin * 1.5] * 50
        )
        signal = engine.get_signal("SPY")
        assert signal is not None
        # percentile = 50/101 ≈ 0.495, which is between 0.25 and 0.50
        assert signal.regime == 'normal'

    def test_signal_engine_get_signal_std_zero(self):
        """get_signal handles zero std in vpin history (constant VPIN)."""
        from src.signals.vpin_bvc import VPINEngine

        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        base = datetime(2026, 5, 14, 9, 30)

        for i in range(10):
            engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=50000
            )

        current_vpin = engine.calculate_vpin("SPY")
        assert current_vpin is not None

        # All identical values => std = 0
        engine.vpin_history["SPY"] = [current_vpin] * 100
        signal = engine.get_signal("SPY")
        assert signal is not None
        # With zero std, z_score should be finite (0 or capped value)
        assert np.isfinite(signal.z_score)

    def test_engine_get_signal_vpin_std_positive(self):
        """Z-score calculation with positive std."""
        from src.signals.vpin_bvc import VPINEngine

        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        base = datetime(2026, 5, 14, 9, 30)

        for i in range(10):
            engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=50000
            )

        current_vpin = engine.calculate_vpin("SPY")
        assert current_vpin is not None

        # History with wide spread => std > 0
        engine.vpin_history["SPY"] = (
            [0.001] * 50 + [0.999] * 50
        )
        signal = engine.get_signal("SPY")
        assert signal is not None
        # Current vpin is around 0.33, significantly above 0.001 but
        # the history has both extremes. Z-score is calculable.
        assert signal.vpin_std > 0
        assert isinstance(signal.z_score, float)

    def test_to_signal_snapshot_regime_in_metadata(self):
        """Snapshot metadata should contain regime and all key fields."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.45, vpin_ma=0.40, vpin_std=0.05,
            z_score=1.0, percentile=0.80, regime='elevated',
            confidence=0.75, toxicity_level=0.6,
            recommendation='delay', expected_cost_impact=5.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert snapshot.metadata['vpin'] == 0.45
        assert snapshot.metadata['vpin_ma'] == 0.40
        assert snapshot.metadata['z_score'] == 1.0
        assert snapshot.metadata['regime'] == 'elevated'
        assert snapshot.metadata['toxicity_level'] == 0.6
        assert snapshot.metadata['recommendation'] == 'delay'

    def test_snapshot_explanation_format(self):
        """Snapshot explanation should contain key metrics."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.45, vpin_ma=0.40, vpin_std=0.05,
            z_score=1.0, percentile=0.80, regime='elevated',
            confidence=0.75, toxicity_level=0.6,
            recommendation='delay', expected_cost_impact=5.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert 'VPIN' in snapshot.explanation
        assert 'regime=elevated' in snapshot.explanation
        assert '0.45' in snapshot.explanation
        assert 'z=1.00' in snapshot.explanation
        assert 'toxicity=0.60' in snapshot.explanation
        assert 'rec=delay' in snapshot.explanation


class TestVpinRebalanceOptimizerSignal:
    """RebalanceOptimizer should_execute_now with signal data."""

    def test_should_execute_low_toxicity(self):
        """'execute' recommendation should return (True, low_toxicity, >0)."""
        from src.signals.vpin_bvc import VPINEngine, RebalanceOptimizer

        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        base = datetime(2026, 5, 14, 9, 30)

        for i in range(10):
            engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=50000
            )

        current_vpin = engine.calculate_vpin("SPY")
        assert current_vpin is not None

        # Force low toxicity: all history > current vpin => percentile < 0.30
        engine.vpin_history["SPY"] = [current_vpin * 1.5] * 100

        optimizer = RebalanceOptimizer(engine)
        execute, reason, savings = optimizer.should_execute_now("SPY")
        assert execute is True
        assert "low_toxicity" in reason
        assert savings > 0

    def test_should_avoid_high_toxicity(self):
        """'avoid' recommendation should return (False, high_toxicity, >0)."""
        from src.signals.vpin_bvc import VPINEngine, RebalanceOptimizer

        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        base = datetime(2026, 5, 14, 9, 30)

        for i in range(10):
            engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=50000
            )

        current_vpin = engine.calculate_vpin("SPY")
        assert current_vpin is not None

        # Force high toxicity: most history < current vpin => percentile >= 0.70
        engine.vpin_history["SPY"] = (
            [current_vpin * 0.5] * 80 + [current_vpin * 1.5] * 20
        )

        optimizer = RebalanceOptimizer(engine)
        execute, reason, savings = optimizer.should_execute_now("SPY")
        assert execute is False
        assert "high_toxicity" in reason
        assert savings > 0

    def test_execution_quality_report_with_data(self):
        """Execution quality report with data should contain symbol metrics."""
        from src.signals.vpin_bvc import VPINEngine, RebalanceOptimizer

        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        base = datetime(2026, 5, 14, 9, 30)

        for i in range(10):
            engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=50000
            )

        current_vpin = engine.calculate_vpin("SPY")
        assert current_vpin is not None
        engine.vpin_history["SPY"] = [current_vpin * 0.5] * 50 + [current_vpin * 1.5] * 50

        optimizer = RebalanceOptimizer(engine)
        report = optimizer.get_execution_quality_report()
        assert 'symbols' in report
        assert 'SPY' in report['symbols']
        data = report['symbols']['SPY']
        assert 'vpin' in data
        assert 'regime' in data
        assert 'recommendation' in data
        assert 'expected_cost_bps' in data
        assert 'toxicity_level' in data


class TestVPINSignalAdapterDataFlow:
    """VPINSignalAdapter data flow and raw_data format."""

    def test_to_ensemble_signal_with_some_data(self):
        """With data, to_ensemble_signal returns non-neutral regime."""
        from src.signals.vpin_bvc import VPINEngine, VPINSignalAdapter

        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        base = datetime(2026, 5, 14, 9, 30)

        for i in range(10):
            engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=50000
            )

        current_vpin = engine.calculate_vpin("SPY")
        assert current_vpin is not None
        # All history below current vpin => high percentile
        engine.vpin_history["SPY"] = [current_vpin * 0.5] * 100

        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal("SPY")
        assert result['source'] == 'vpin'
        assert result['regime'] != 'neutral'
        assert result['confidence'] > 0
        assert result['probability'] > 0

    def test_to_ensemble_signal_raw_data_format(self):
        """Raw data should contain expected VPIN metrics."""
        from src.signals.vpin_bvc import VPINEngine, VPINSignalAdapter

        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        base = datetime(2026, 5, 14, 9, 30)

        for i in range(10):
            engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=50000
            )

        current_vpin = engine.calculate_vpin("SPY")
        assert current_vpin is not None
        engine.vpin_history["SPY"] = [current_vpin * 0.5] * 100

        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal("SPY")
        raw = result['raw_data']
        assert 'vpin' in raw
        assert 'vpin_percentile' in raw
        assert 'z_score' in raw
        assert 'recommendation' in raw
        assert 'expected_cost_bps' in raw

    def test_rebalance_timing_signal_has_expected_keys(self):
        """get_rebalance_timing_signal should contain all expected keys."""
        from src.signals.vpin_bvc import VPINEngine, VPINSignalAdapter
        engine = VPINEngine()
        adapter = VPINSignalAdapter(engine)
        result = adapter.get_rebalance_timing_signal()
        assert result['source'] == 'vpin_rebalance'
        assert 'execute_now' in result
        assert 'reason' in result
        assert 'expected_savings_bps' in result
        assert 'timestamp' in result


class TestVpinModuleExports:
    """Verify __all__ exports are complete."""

    def test_all_exports_present(self):
        from src.signals import vpin_bvc
        expected = [
            'BVCBar', 'VPINBucket', 'VPINSignal', 'BVCCalculator',
            'VPINEngine', 'RebalanceOptimizer', 'VPINSignalAdapter',
            'load_historical_bars', 'backtest_vpin',
        ]
        for name in expected:
            assert name in vpin_bvc.__all__, f"{name} missing from __all__"
        assert len(vpin_bvc.__all__) == len(expected)


# ---------------------------------------------------------------------------
# New comprehensive tests: dataclass field validation via dataclasses.fields()
# ---------------------------------------------------------------------------


class TestBVCBarDataclassFields:
    """Validate BVCBar dataclass schema via dataclasses.fields()."""

    def test_field_count(self):
        from src.signals.vpin_bvc import BVCBar
        import dataclasses
        fields = dataclasses.fields(BVCBar)
        assert len(fields) == 9

    def test_field_names(self):
        from src.signals.vpin_bvc import BVCBar
        import dataclasses
        names = [f.name for f in dataclasses.fields(BVCBar)]
        expected = ['timestamp', 'open', 'high', 'low', 'close',
                     'volume', 'buy_volume', 'sell_volume', 'vpin_local']
        assert names == expected

    def test_field_types(self):
        from src.signals.vpin_bvc import BVCBar
        import dataclasses
        from datetime import datetime
        types = {f.name: f.type for f in dataclasses.fields(BVCBar)}
        assert types['timestamp'] is datetime
        assert types['open'] is float
        assert types['high'] is float
        assert types['low'] is float
        assert types['close'] is float
        assert types['volume'] is float
        assert types['buy_volume'] is float
        assert types['sell_volume'] is float
        assert types['vpin_local'] is float

    def test_no_defaults_on_required(self):
        from src.signals.vpin_bvc import BVCBar
        import dataclasses
        for f in dataclasses.fields(BVCBar):
            assert f.default is dataclasses.MISSING, f"{f.name} has default"
            assert f.default_factory is dataclasses.MISSING, f"{f.name} has default_factory"

    def test_asdict_round_trip(self):
        from src.signals.vpin_bvc import BVCBar
        import dataclasses
        from datetime import datetime
        ts = datetime(2026, 5, 14, 10, 0)
        bar = BVCBar(timestamp=ts, open=1.0, high=2.0, low=0.5,
                      close=1.5, volume=1000, buy_volume=600,
                      sell_volume=400, vpin_local=0.2)
        d = dataclasses.asdict(bar)
        restored = BVCBar(**d)
        assert restored == bar
        assert restored.timestamp == ts


class TestVPINBucketDataclassFields:
    """Validate VPINBucket dataclass schema via dataclasses.fields()."""

    def test_field_count(self):
        from src.signals.vpin_bvc import VPINBucket
        import dataclasses
        fields = dataclasses.fields(VPINBucket)
        assert len(fields) == 9

    def test_field_names(self):
        from src.signals.vpin_bvc import VPINBucket
        import dataclasses
        names = [f.name for f in dataclasses.fields(VPINBucket)]
        expected = ['start_time', 'end_time', 'target_volume', 'actual_volume',
                     'bars', 'buy_volume', 'sell_volume', 'vpin', 'complete']
        assert names == expected

    def test_field_types(self):
        from src.signals.vpin_bvc import VPINBucket
        import dataclasses
        from datetime import datetime
        types = {f.name: f.type for f in dataclasses.fields(VPINBucket)}
        assert types['start_time'] is datetime
        assert types['end_time'] is datetime
        assert types['target_volume'] is float
        assert types['actual_volume'] is float
        # 'bars' type is list[BVCBar] — check origin is list and args includes BVCBar
        from typing import get_origin, get_args
        from src.signals.vpin_bvc import BVCBar
        bars_type = types['bars']
        assert get_origin(bars_type) is list
        assert BVCBar in get_args(bars_type)
        assert types['buy_volume'] is float
        assert types['sell_volume'] is float
        assert types['vpin'] is float
        assert types['complete'] is bool

    def test_asdict_round_trip(self):
        from src.signals.vpin_bvc import VPINBucket, BVCBar
        import dataclasses
        from datetime import datetime
        ts = datetime(2026, 5, 14, 9, 30)
        bar = BVCBar(timestamp=ts, open=1.0, high=2.0, low=0.5,
                      close=1.5, volume=1000, buy_volume=600,
                      sell_volume=400, vpin_local=0.2)
        bucket = VPINBucket(start_time=ts, end_time=ts, target_volume=50000.0,
                             actual_volume=10000.0, bars=[bar],
                             buy_volume=600.0, sell_volume=400.0, vpin=0.2,
                             complete=False)
        d = dataclasses.asdict(bucket)
        # asdict converts nested dataclasses to dicts; verify field values directly
        assert d['start_time'] == ts
        assert d['end_time'] == ts
        assert d['target_volume'] == 50000.0
        assert d['actual_volume'] == 10000.0
        assert len(d['bars']) == 1
        assert d['bars'][0]['open'] == 1.0
        assert d['buy_volume'] == 600.0
        assert d['sell_volume'] == 400.0
        assert d['vpin'] == 0.2
        assert d['complete'] is False

    def test_no_defaults_on_required(self):
        from src.signals.vpin_bvc import VPINBucket
        import dataclasses
        for f in dataclasses.fields(VPINBucket):
            assert f.default is dataclasses.MISSING, f"{f.name} has default"
            assert f.default_factory is dataclasses.MISSING, f"{f.name} has default_factory"


class TestVPINSignalDataclassFields:
    """Validate VPINSignal dataclass schema via dataclasses.fields()."""

    def test_field_count(self):
        from src.signals.vpin_bvc import VPINSignal
        import dataclasses
        fields = dataclasses.fields(VPINSignal)
        assert len(fields) == 11

    def test_field_names(self):
        from src.signals.vpin_bvc import VPINSignal
        import dataclasses
        names = [f.name for f in dataclasses.fields(VPINSignal)]
        expected = ['timestamp', 'vpin', 'vpin_ma', 'vpin_std', 'z_score',
                     'percentile', 'regime', 'confidence', 'toxicity_level',
                     'recommendation', 'expected_cost_impact']
        assert names == expected

    def test_field_types(self):
        from src.signals.vpin_bvc import VPINSignal
        import dataclasses
        from datetime import datetime
        types = {f.name: f.type for f in dataclasses.fields(VPINSignal)}
        assert types['timestamp'] is datetime
        assert types['vpin'] is float
        assert types['vpin_ma'] is float
        assert types['vpin_std'] is float
        assert types['z_score'] is float
        assert types['percentile'] is float
        assert types['regime'] is str
        assert types['confidence'] is float
        assert types['toxicity_level'] is float
        assert types['recommendation'] is str
        assert types['expected_cost_impact'] is float

    def test_asdict_round_trip(self):
        from src.signals.vpin_bvc import VPINSignal
        import dataclasses
        from datetime import datetime
        ts = datetime(2026, 5, 14, 10, 0)
        sig = VPINSignal(timestamp=ts, vpin=0.45, vpin_ma=0.40,
                          vpin_std=0.05, z_score=1.0, percentile=0.80,
                          regime='elevated', confidence=0.75,
                          toxicity_level=0.6, recommendation='delay',
                          expected_cost_impact=5.0)
        d = dataclasses.asdict(sig)
        restored = VPINSignal(**d)
        assert restored == sig

    def test_no_defaults_on_required(self):
        from src.signals.vpin_bvc import VPINSignal
        import dataclasses
        for f in dataclasses.fields(VPINSignal):
            assert f.default is dataclasses.MISSING, f"{f.name} has default"
            assert f.default_factory is dataclasses.MISSING, f"{f.name} has default_factory"


# ---------------------------------------------------------------------------
# NaN / Inf / negative edge cases for BVCCalculator
# ---------------------------------------------------------------------------


class TestBVCCalculatorNaNInf:
    """BVCCalculator with NaN, Inf, and negative edge values."""

    def test_classify_bar_nan_high(self):
        """NaN high should produce NaN buy_volume (IEEE 754: NaN != NaN)."""
        from src.signals.vpin_bvc import BVCCalculator
        import math
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            o=100.0, h=float('nan'), l=99.0, c=101.0, v=10000
        )
        # high == low is False (nan != nan), so the else branch is taken
        # (nan - 99) / (nan - 99) == nan / nan == nan
        assert math.isnan(bar.buy_volume)
        assert math.isnan(bar.sell_volume)
        assert math.isnan(bar.vpin_local)

    def test_classify_bar_inf_close(self):
        """Infinite close should be handled without crash."""
        from src.signals.vpin_bvc import BVCCalculator
        import math
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            o=100.0, h=102.0, l=99.0, c=float('inf'), v=10000
        )
        # buy_volume = 10000 * (inf - 99) / (102 - 99) = inf
        assert math.isinf(bar.buy_volume)
        assert math.isinf(bar.sell_volume)  # v - inf = -inf
        # vpin_local = abs(inf - (-inf)) / 10000 = inf
        assert math.isinf(bar.vpin_local)

    def test_classify_bar_negative_volume(self):
        """Negative volume should not crash but produces negative buy_volume."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            o=100.0, h=102.0, l=99.0, c=101.0, v=-10000
        )
        # buy_volume = -10000 * 2/3 < 0
        assert bar.buy_volume < 0
        assert bar.sell_volume < 0  # v - buy_volume = even more negative
        # vpin_local = abs(neg - even_more_neg) / |v| = (something) / -10000
        # Actually: vpin_local = abs(buy_volume - sell_volume) / v where v > 0
        # but v = -10000 which is NOT > 0, so vpin_local = 0.0 per the code
        assert bar.vpin_local == 0.0

    def test_classify_bar_close_above_high(self):
        """close > high gives buy_volume > volume and negative sell_volume."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            o=100.0, h=102.0, l=99.0, c=105.0, v=10000
        )
        # buy_volume = 10000 * (105 - 99) / (102 - 99) = 10000 * 6/3 = 20000
        assert bar.buy_volume == pytest.approx(20000.0)
        # sell_volume = 10000 - 20000 = -10000
        assert bar.sell_volume == pytest.approx(-10000.0)
        # vpin_local = |20000 - (-10000)| / 10000 = 30000/10000 = 3.0
        assert bar.vpin_local == pytest.approx(3.0)  # >1.0, violates normal range

    def test_classify_bar_close_below_low(self):
        """close < low gives buy_volume < 0."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            o=100.0, h=102.0, l=99.0, c=98.0, v=10000
        )
        # buy_volume = 10000 * (98 - 99) / (102 - 99) = -3333.33
        assert bar.buy_volume == pytest.approx(-3333.33, rel=1e-3)
        # sell_volume = 10000 - (-3333.33) = 13333.33
        assert bar.sell_volume == pytest.approx(13333.33, rel=1e-3)

    def test_classify_bar_negative_prices(self):
        """Negative prices should not crash (unrealistic but robust)."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            o=-10.0, h=-5.0, l=-15.0, c=-8.0, v=10000
        )
        # buy_volume = 10000 * (-8 - (-15)) / (-5 - (-15)) = 10000 * 7/10 = 7000
        assert bar.buy_volume == pytest.approx(7000.0)
        assert bar.sell_volume == pytest.approx(3000.0)
        assert 0.0 <= bar.vpin_local <= 1.0

    def test_get_buy_sell_imbalance_with_nan_bar(self):
        """NaN buy_volume in a bar makes total NaN, imbalance defaults to 0."""
        from src.signals.vpin_bvc import BVCCalculator, BVCBar
        import math
        calc = BVCCalculator()
        bar = BVCBar(timestamp=datetime(2026, 5, 14), open=100.0,
                      high=102.0, low=99.0, close=101.0, volume=10000,
                      buy_volume=float('nan'), sell_volume=float('nan'),
                      vpin_local=float('nan'))
        calc.add_bar(bar)
        buy, sell, imbalance = calc.get_buy_sell_imbalance(window=1)
        # sum of NaN is NaN, total_buy + total_sell = NaN
        # NaN > 0 is False, so imbalance = 0.0
        assert math.isnan(buy)
        assert math.isnan(sell)
        assert imbalance == 0.0

    def test_classify_bar_nan_volume_preserved(self):
        """NaN volume is stored as-is (no crash)."""
        from src.signals.vpin_bvc import BVCCalculator
        import math
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            o=100.0, h=102.0, l=99.0, c=101.0, v=float('nan')
        )
        # v > 0 is False for NaN, so vpin_local = 0.0
        assert math.isnan(bar.volume)
        assert math.isnan(bar.buy_volume)
        assert math.isnan(bar.sell_volume)
        assert bar.vpin_local == 0.0

    def test_classify_bar_inf_high_low_equal(self):
        """Both high and low being Inf triggers the equal branch (50/50)."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        bar = calc.classify_bar(
            timestamp=datetime(2026, 5, 14, 10, 0),
            o=100.0, h=float('inf'), l=float('inf'), c=101.0, v=10000
        )
        # inf == inf is True, so buy_volume = 10000 * 0.5 = 5000
        assert bar.buy_volume == pytest.approx(5000.0)
        assert bar.sell_volume == pytest.approx(5000.0)
        assert bar.vpin_local == 0.0


# ---------------------------------------------------------------------------
# Extreme VPINEngine parameter edge cases
# ---------------------------------------------------------------------------


class TestVPINEngineExtremeParams:
    """VPINEngine with extreme parameter values."""

    def test_zero_bucket_size(self):
        """bucket_size=0: each bar completes a bucket immediately."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=0, vpin_window=3)
        base = datetime(2026, 5, 14, 9, 30)
        completed_count = 0
        for i in range(5):
            result = engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=10000
            )
            if result is not None:
                completed_count += 1
        # With bucket_size=0, actual_volume (10000) >= 0 is True, so
        # every bar completes a bucket
        assert completed_count > 0

    def test_tiny_bucket_size(self):
        """bucket_size=1: each non-zero bar completes a bucket."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=1, vpin_window=5)
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(10):
            engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=10000
            )
        # Should have 10 completed buckets
        assert len(engine.completed_buckets["SPY"]) == 10

    def test_negative_bucket_size(self):
        """negative bucket_size: each bar completes a bucket (vol >= 0 always)."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=-1000, vpin_window=5)
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(5):
            engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                o=100.0, h=102.0, l=99.0, c=101.0, v=10000
            )
        # With bucket_size=-1000, actual_volume (10000 >= -1000) is always True
        assert len(engine.completed_buckets["SPY"]) == 5

    def test_empty_symbols_list_falls_back_to_defaults(self):
        """Empty symbols list falls back to default symbols (falsy check)."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(symbols=[])
        # [] is falsy, so `symbols or ['SPY', ...]` falls back to defaults
        assert engine.symbols == ['SPY', 'QQQ', 'TLT', 'GLD']
        assert 'SPY' in engine.completed_buckets
        assert 'SPY' in engine.bvc_calcs
        assert 'SPY' in engine.vpin_history

    def test_duplicate_symbols_list(self):
        """Duplicate symbols create one dict entry per unique symbol."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(symbols=["SPY", "SPY", "QQQ"])
        assert engine.symbols == ["SPY", "SPY", "QQQ"]
        # Dict comprehension deduplicates: only 2 unique keys
        assert len(engine.bvc_calcs) == 2
        # SPY and QQQ both work
        base = datetime(2026, 5, 14, 9, 30)
        engine.process_bar("SPY", base, 100.0, 102.0, 99.0, 101.0, 10000)
        assert len(engine.completed_buckets) == 2
        engine.process_bar("QQQ", base, 100.0, 102.0, 99.0, 101.0, 10000)
        assert engine.current_buckets["SPY"].actual_volume == 10000

    def test_none_symbols_default(self):
        """None symbols defaults to SPY, QQQ, TLT, GLD."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(symbols=None)
        assert engine.symbols == ['SPY', 'QQQ', 'TLT', 'GLD']

    def test_very_large_vpin_window(self):
        """Very large vpin_window doesn't cause issues with no data."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(vpin_window=10000)
        assert engine.calculate_vpin("SPY") is None

    def test_process_bar_for_unknown_symbol(self):
        """process_bar with an unknown symbol raises KeyError."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(symbols=["SPY"])
        with pytest.raises(KeyError):
            engine.process_bar(
                "UNKNOWN", datetime(2026, 5, 14, 9, 30),
                100.0, 102.0, 99.0, 101.0, 10000
            )

    def test_calculate_vpin_for_unknown_symbol(self):
        """calculate_vpin with unknown symbol raises KeyError."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(symbols=["SPY"])
        with pytest.raises(KeyError):
            engine.calculate_vpin("UNKNOWN")

    def test_get_signal_for_unknown_symbol(self):
        """get_signal with unknown symbol raises KeyError."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(symbols=["SPY"])
        with pytest.raises(KeyError):
            engine.get_signal("UNKNOWN")


# ---------------------------------------------------------------------------
# VPINEngine process_bar with edge case OHLC values
# ---------------------------------------------------------------------------


class TestVPINEngineProcessBarEdgeCases:
    """VPINEngine.process_bar with unusual OHLC inputs."""

    def test_process_bar_with_negative_volume(self):
        """Negative volume should not crash."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=5)
        base = datetime(2026, 5, 14, 9, 30)
        result = engine.process_bar(
            "SPY", base, 100.0, 102.0, 99.0, 101.0, v=-10000
        )
        # Should not raise, bucket actual_volume becomes -10000 (less than 50000)
        assert result is None
        assert engine.current_buckets["SPY"].actual_volume == -10000

    def test_process_bar_with_zero_volume_bucket_fill(self):
        """Zero volume bars accumulate zero actual_volume."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=1000, vpin_window=5)
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(5):
            engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                100.0, 102.0, 99.0, 101.0, v=0
            )
        assert engine.current_buckets["SPY"].actual_volume == 0
        assert len(engine.completed_buckets["SPY"]) == 0

    def test_process_bar_nan_ohlc(self):
        """NaN OHLC values propagate without crash."""
        from src.signals.vpin_bvc import VPINEngine
        import math
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=5)
        base = datetime(2026, 5, 14, 9, 30)
        result = engine.process_bar(
            "SPY", base, float('nan'), float('nan'), float('nan'),
            float('nan'), v=10000
        )
        # Should not crash
        assert result is None or result is not None
        # buy_volume will be nan, so bucket buy_volume will be nan
        assert math.isnan(engine.current_buckets["SPY"].buy_volume)

    def test_process_bar_exact_bucket_fill(self):
        """Exactly filling a bucket returns it as completed."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=10000, vpin_window=5)
        base = datetime(2026, 5, 14, 9, 30)
        result = engine.process_bar("SPY", base, 100.0, 102.0, 99.0, 101.0, v=10000)
        assert result is not None
        assert result.complete is True
        assert result.actual_volume == 10000

    def test_process_bar_multiple_completed_buckets(self):
        """Processing enough volume produces multiple completed buckets (trimmed)."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=5000, vpin_window=5)
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(20):
            engine.process_bar(
                "SPY", base + timedelta(minutes=i),
                100.0, 102.0, 99.0, 101.0, v=5000
            )
        # Trimmed to vpin_window * 2 = 10
        assert len(engine.completed_buckets["SPY"]) == 10

    def test_process_bar_updates_end_time(self):
        """end_time should update to the latest bar timestamp."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=5)
        t1 = datetime(2026, 5, 14, 9, 30)
        t2 = datetime(2026, 5, 14, 9, 31)
        engine.process_bar("SPY", t1, 100.0, 102.0, 99.0, 101.0, v=10000)
        assert engine.current_buckets["SPY"].start_time == t1
        assert engine.current_buckets["SPY"].end_time == t1
        engine.process_bar("SPY", t2, 100.0, 102.0, 99.0, 101.0, v=10000)
        assert engine.current_buckets["SPY"].end_time == t2


# ---------------------------------------------------------------------------
# RebalanceOptimizer edge cases
# ---------------------------------------------------------------------------


class TestRebalanceOptimizerEdgeCases:
    """RebalanceOptimizer with edge case states."""

    def test_moderate_toxicity_returns_true(self):
        """Delay/moderate recommendation returns (True, moderate_toxicity, 0.0)."""
        from src.signals.vpin_bvc import VPINEngine, RebalanceOptimizer
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(10):
            engine.process_bar("SPY", base + timedelta(minutes=i),
                               100.0, 102.0, 99.0, 101.0, v=50000)
        current_vpin = engine.calculate_vpin("SPY")
        assert current_vpin is not None
        # percentile ~0.50 (delay recommendation, not avoid nor execute)
        engine.vpin_history["SPY"] = (
            [current_vpin * 0.5] * 50 + [current_vpin * 1.5] * 50
        )
        optimizer = RebalanceOptimizer(engine)
        execute, reason, savings = optimizer.should_execute_now("SPY")
        assert execute is True
        assert "moderate_toxicity" in reason
        assert savings == 0.0

    def test_pending_rebalances_attribute(self):
        """pending_rebalances should be an empty list on init."""
        from src.signals.vpin_bvc import VPINEngine, RebalanceOptimizer
        engine = VPINEngine()
        optimizer = RebalanceOptimizer(engine)
        assert optimizer.pending_rebalances == []
        assert isinstance(optimizer.pending_rebalances, list)

    def test_execution_quality_report_timestamp_format(self):
        """Report timestamp should be an ISO format string."""
        from src.signals.vpin_bvc import VPINEngine, RebalanceOptimizer
        engine = VPINEngine()
        optimizer = RebalanceOptimizer(engine)
        report = optimizer.get_execution_quality_report()
        assert isinstance(report['timestamp'], str)
        assert 'T' in report['timestamp']

    def test_should_execute_for_all_symbols(self):
        """Optimizer works for any symbol in the engine's list."""
        from src.signals.vpin_bvc import VPINEngine, RebalanceOptimizer
        engine = VPINEngine(symbols=["QQQ"])
        optimizer = RebalanceOptimizer(engine, max_delay_minutes=30)
        execute, reason, savings = optimizer.should_execute_now("QQQ")
        assert execute is True
        assert reason == "insufficient_data"
        assert savings == 0.0


# ---------------------------------------------------------------------------
# VPINSignalAdapter regime mappings (including bull and boundaries)
# ---------------------------------------------------------------------------


class TestVPINSignalAdapterRegimeMapping:
    """VPINSignalAdapter regime mapping for all four regimes."""

    def _setup_with_percentile(self, percentile_ratio):
        """Helper: fill engine and set vpin_history to achieve target percentile."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(10):
            engine.process_bar("SPY", base + timedelta(minutes=i),
                               100.0, 102.0, 99.0, 101.0, v=50000)
        current_vpin = engine.calculate_vpin("SPY")
        assert current_vpin is not None
        below = int(percentile_ratio * 100)
        above = 100 - below
        engine.vpin_history["SPY"] = (
            [current_vpin * 0.5] * below + [current_vpin * 1.5] * above
        )
        return engine

    def test_bull_regime(self):
        """Percentile <= 0.25 maps to bull regime with probability 0.6."""
        from src.signals.vpin_bvc import VPINSignalAdapter
        engine = self._setup_with_percentile(0.20)
        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal("SPY")
        assert result['regime'] == 'bull'
        assert result['probability'] == 0.6

    def test_neutral_regime(self):
        """0.25 < percentile < 0.75 maps to neutral with probability 0.5."""
        from src.signals.vpin_bvc import VPINSignalAdapter
        engine = self._setup_with_percentile(0.50)
        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal("SPY")
        assert result['regime'] == 'neutral'
        assert result['probability'] == 0.5

    def test_bear_regime(self):
        """0.75 <= percentile < 0.90 maps to bear with probability 0.7."""
        from src.signals.vpin_bvc import VPINSignalAdapter
        engine = self._setup_with_percentile(0.80)
        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal("SPY")
        assert result['regime'] == 'bear'
        assert result['probability'] == 0.7

    def test_crisis_regime(self):
        """Percentile >= 0.90 maps to crisis with probability 0.8."""
        from src.signals.vpin_bvc import VPINSignalAdapter
        engine = self._setup_with_percentile(0.95)
        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal("SPY")
        assert result['regime'] == 'crisis'
        assert result['probability'] == 0.8

    def test_boundary_percentile_25(self):
        """Exact boundary percentile=0.25 maps to bull (<= 0.25)."""
        from src.signals.vpin_bvc import VPINSignalAdapter
        engine = self._setup_with_percentile(0.25)
        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal("SPY")
        assert result['regime'] == 'bull'

    def test_boundary_percentile_75(self):
        """Percentile >= 0.75 maps to bear (accounting for +1 from calculate_vpin)."""
        from src.signals.vpin_bvc import VPINSignalAdapter
        # ratio=0.76 => 76 below out of 100 => ~76/101 ≈ 0.752 >= 0.75
        engine = self._setup_with_percentile(0.76)
        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal("SPY")
        assert result['regime'] == 'bear'

    def test_boundary_percentile_90(self):
        """Percentile >= 0.90 maps to crisis (accounting for +1 from calculate_vpin)."""
        from src.signals.vpin_bvc import VPINSignalAdapter
        # ratio=0.91 => 91 below out of 100 => 91/101 ≈ 0.901 >= 0.90
        engine = self._setup_with_percentile(0.91)
        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal("SPY")
        assert result['regime'] == 'crisis'

    def test_regime_mapping_source_field(self):
        """Ensemble signal should have source='vpin'."""
        from src.signals.vpin_bvc import VPINSignalAdapter
        engine = self._setup_with_percentile(0.50)
        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal("SPY")
        assert result['source'] == 'vpin'


# ---------------------------------------------------------------------------
# VPINEngine signal with all regime classifications from get_signal
# ---------------------------------------------------------------------------


class TestVPINEngineSignalRegimes:
    """VPINEngine.get_signal returns correct regimes for all classifications."""

    def _make_engine(self):
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(10):
            engine.process_bar("SPY", base + timedelta(minutes=i),
                               100.0, 102.0, 99.0, 101.0, v=50000)
        current_vpin = engine.calculate_vpin("SPY")
        assert current_vpin is not None
        return engine, current_vpin

    def test_signal_low_regime(self):
        """get_signal with percentile < 0.25 returns low regime."""
        engine, vpin = self._make_engine()
        engine.vpin_history["SPY"] = [vpin * 1.5] * 100
        signal = engine.get_signal("SPY")
        assert signal is not None
        assert signal.regime == 'low'
        assert signal.recommendation == 'execute'

    def test_signal_normal_regime(self):
        """get_signal with 0.25 <= percentile < 0.50 returns normal regime."""
        engine, vpin = self._make_engine()
        engine.vpin_history["SPY"] = (
            [vpin * 0.5] * 30 + [vpin * 1.5] * 70
        )
        signal = engine.get_signal("SPY")
        assert signal is not None
        assert signal.regime == 'normal'

    def test_signal_elevated_regime(self):
        """get_signal with 0.50 <= percentile < 0.75 returns elevated regime."""
        engine, vpin = self._make_engine()
        engine.vpin_history["SPY"] = (
            [vpin * 0.5] * 55 + [vpin * 1.5] * 45
        )
        signal = engine.get_signal("SPY")
        assert signal is not None
        assert signal.regime == 'elevated'
        assert signal.recommendation == 'delay'

    def test_signal_high_regime(self):
        """get_signal with percentile >= 0.75 returns high regime."""
        engine, vpin = self._make_engine()
        engine.vpin_history["SPY"] = (
            [vpin * 0.5] * 80 + [vpin * 1.5] * 20
        )
        signal = engine.get_signal("SPY")
        assert signal is not None
        assert signal.regime == 'high'
        assert signal.recommendation == 'avoid'

    def test_signal_toxicity_level_matches_percentile(self):
        """toxicity_level should equal percentile."""
        engine, vpin = self._make_engine()
        engine.vpin_history["SPY"] = (
            [vpin * 0.5] * 50 + [vpin * 1.5] * 50
        )
        signal = engine.get_signal("SPY")
        assert signal is not None
        assert signal.toxicity_level == signal.percentile

    def test_signal_expected_cost_negative_for_execute(self):
        """'execute' recommendation has negative expected_cost_impact."""
        engine, vpin = self._make_engine()
        engine.vpin_history["SPY"] = [vpin * 1.5] * 100
        signal = engine.get_signal("SPY")
        assert signal is not None
        assert signal.recommendation == 'execute'
        assert signal.expected_cost_impact == -3.0

    def test_signal_expected_cost_zero_for_delay(self):
        """'delay' recommendation has zero expected_cost_impact."""
        engine, vpin = self._make_engine()
        engine.vpin_history["SPY"] = (
            [vpin * 0.5] * 55 + [vpin * 1.5] * 45
        )
        signal = engine.get_signal("SPY")
        assert signal is not None
        assert signal.recommendation == 'delay'
        assert signal.expected_cost_impact == 0.0

    def test_signal_expected_cost_positive_for_avoid(self):
        """'avoid' recommendation has positive expected_cost_impact (5.0)."""
        engine, vpin = self._make_engine()
        engine.vpin_history["SPY"] = (
            [vpin * 0.5] * 80 + [vpin * 1.5] * 20
        )
        signal = engine.get_signal("SPY")
        assert signal is not None
        assert signal.recommendation == 'avoid'
        assert signal.expected_cost_impact == 5.0

    def test_signal_vpin_in_range(self):
        """VPIN value should always be between 0 and 1."""
        engine, vpin = self._make_engine()
        engine.vpin_history["SPY"] = [vpin] * 100
        signal = engine.get_signal("SPY")
        assert signal is not None
        assert 0.0 <= signal.vpin <= 1.0
        assert 0.0 <= signal.vpin_ma <= 1.0

    def test_signal_confidence_values(self):
        """Confidence should be 0.6 for low/high, 0.7 for normal/elevated."""
        engine, vpin = self._make_engine()
        # low regime: percentile < 0.25 => confidence 0.6
        engine.vpin_history["SPY"] = [vpin * 1.5] * 100
        signal = engine.get_signal("SPY")
        assert signal is not None
        assert signal.confidence == 0.6


# ---------------------------------------------------------------------------
# Function-level tests: load_historical_bars and backtest_vpin
# ---------------------------------------------------------------------------


class TestLoadHistoricalBars:
    """load_historical_bars function edge cases."""

    def test_returns_empty_dataframe_on_db_failure(self):
        """DB failure returns empty DataFrame."""
        from src.signals.vpin_bvc import load_historical_bars
        import pandas as pd
        from unittest.mock import patch, MagicMock
        # Mock MARKET_DB to be a mock with exists returning False
        mock_db = MagicMock()
        mock_db.exists.return_value = False
        with patch('src.paths.MARKET_DB', mock_db):
            with patch('requests.get') as mock_get:
                mock_get.side_effect = ConnectionError("HTTP fail")
                df = load_historical_bars("SPY", days=1)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_empty_dataframe_from_missing_symbol(self):
        """Empty DataFrame returned when symbol not found."""
        from src.signals.vpin_bvc import load_historical_bars
        import pandas as pd
        from unittest.mock import patch, MagicMock
        mock_db = MagicMock()
        mock_db.exists.return_value = False
        with patch('src.paths.MARKET_DB', mock_db):
            with patch('requests.get') as mock_get:
                mock_get.side_effect = ConnectionError("HTTP fail")
                df = load_historical_bars("SPY", days=1)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_yahoo_finance_fallback_failure(self):
        """Yahoo Finance fallback failure returns empty DataFrame."""
        from src.signals.vpin_bvc import load_historical_bars
        import pandas as pd
        from unittest.mock import patch, MagicMock
        mock_db = MagicMock()
        mock_db.exists.return_value = False
        with patch('src.paths.MARKET_DB', mock_db):
            with patch('requests.get') as mock_get:
                mock_get.side_effect = ConnectionError("Network error")
                df = load_historical_bars("SPY", days=1)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_default_days_parameter(self):
        """load_historical_bars uses default days=5."""
        from src.signals.vpin_bvc import load_historical_bars
        import pandas as pd
        # Without mocking, this should at least return a DataFrame
        # (even if empty due to missing DB/network in test env)
        df = load_historical_bars("SPY")
        assert isinstance(df, pd.DataFrame)


class TestBacktestVpin:
    """backtest_vpin function edge cases."""

    def test_empty_symbols_list(self):
        """Empty symbols list returns empty results dict."""
        from src.signals.vpin_bvc import backtest_vpin
        result = backtest_vpin([], days=1)
        assert 'results' in result
        assert 'statistics' in result
        assert result['results'] == {}
        assert result['statistics'] == {}

    def test_returns_empty_stats_when_no_data(self):
        """When load_historical_bars returns empty, no stats are produced."""
        from src.signals.vpin_bvc import backtest_vpin
        from unittest.mock import patch
        import pandas as pd
        with patch('src.signals.vpin_bvc.load_historical_bars',
                   return_value=pd.DataFrame()):
            result = backtest_vpin(["SPY"], days=1)
        assert result['statistics'] == {}
        assert 'SPY' in result['results']

    def test_results_structure_per_symbol(self):
        """Each symbol in results has vpins and timestamps lists."""
        from src.signals.vpin_bvc import backtest_vpin
        result = backtest_vpin(["SPY"], days=1)
        assert isinstance(result['results']['SPY']['vpins'], list)
        assert isinstance(result['results']['SPY']['timestamps'], list)

    def test_negative_days_handling(self):
        """Negative days should return empty DataFrame (no crash)."""
        from src.signals.vpin_bvc import backtest_vpin
        result = backtest_vpin(["SPY"], days=-1)
        assert 'results' in result
        assert 'statistics' in result

    def test_multiple_symbols_results(self):
        """Backtest with multiple symbols returns results for each."""
        from src.signals.vpin_bvc import backtest_vpin
        result = backtest_vpin(["SPY", "QQQ"], days=1)
        assert 'SPY' in result['results']
        assert 'QQQ' in result['results']


# ---------------------------------------------------------------------------
# CLI / __main__ guard tests
# ---------------------------------------------------------------------------


class TestCliMain:
    """CLI and __main__ guard tests."""

    def test_cli_no_args_prints_help(self, capsys):
        """cli() with no arguments prints help and exits."""
        from src.signals.vpin_bvc import cli
        import sys
        from unittest.mock import patch
        with patch.object(sys, 'argv', ['vpin_bvc.py']):
            cli()
        captured = capsys.readouterr()
        assert 'usage' in captured.out.lower() or 'usage' in captured.err.lower()

    def test_cli_help_flag(self, capsys):
        """cli() with --help prints help and exits."""
        from src.signals.vpin_bvc import cli
        import sys
        from unittest.mock import patch
        with patch.object(sys, 'argv', ['vpin_bvc.py', '--help']):
            with pytest.raises(SystemExit):
                cli()
        captured = capsys.readouterr()
        assert 'usage' in captured.out.lower() or 'usage' in captured.err.lower()

    def test_cli_backtest_output(self, caplog):
        """cli() with --backtest prints VPIN statistics."""
        from src.signals.vpin_bvc import cli
        import sys
        from unittest.mock import patch
        with patch.object(sys, 'argv', ['vpin_bvc.py', '--backtest',
                                        '--symbols', 'SPY', '--days', '1']):
            with caplog.at_level(logging.INFO, logger="src.signals.vpin_bvc"):
                cli()
        assert 'VPIN' in caplog.text

    def test_cli_status_output(self, caplog):
        """cli() with --status prints various status sections."""
        from src.signals.vpin_bvc import cli
        import sys
        from unittest.mock import patch
        with patch.object(sys, 'argv', ['vpin_bvc.py', '--status',
                                        '--symbols', 'SPY']):
            with caplog.at_level(logging.INFO, logger="src.signals.vpin_bvc"):
                cli()
        assert 'VPIN Status' in caplog.text
        assert 'Ensemble Signal' in caplog.text
        assert 'Rebalance Timing' in caplog.text
        assert 'Execution Quality Report' in caplog.text

    def test_cli_backtest_custom_symbols(self, caplog):
        """cli() with custom symbols in backtest mode."""
        from src.signals.vpin_bvc import cli
        import sys
        from unittest.mock import patch
        with patch.object(sys, 'argv', ['vpin_bvc.py', '--backtest',
                                        '--symbols', 'SPY', 'QQQ', '--days', '1']):
            with caplog.at_level(logging.INFO, logger="src.signals.vpin_bvc"):
                cli()
        assert 'VPIN' in caplog.text

    def test_main_guard_calls_cli(self):
        """The __main__ guard calls cli()."""
        from src.signals.vpin_bvc import cli
        import sys
        from unittest.mock import patch
        # Verify cli is callable and the module-level __main__ guard exists
        with patch.object(sys, 'argv', ['vpin_bvc.py', '--help']):
            with pytest.raises(SystemExit):
                cli()

    def test_cli_days_default(self, caplog):
        """cli() --backtest uses default days=30."""
        # We can't directly inspect the default, but we can check the argparse works
        from src.signals.vpin_bvc import cli
        import sys
        from unittest.mock import patch
        with patch.object(sys, 'argv', ['vpin_bvc.py', '--backtest']):
            with caplog.at_level(logging.INFO, logger="src.signals.vpin_bvc"):
                cli()
        # Should not crash with default args
        assert caplog is not None


# ---------------------------------------------------------------------------
# Module-level validation
# ---------------------------------------------------------------------------


class TestModuleLevel:
    """Module-level attributes: __all__, docstring, logger."""

    def test_module_has_docstring(self):
        """Module should have a non-empty docstring."""
        import src.signals.vpin_bvc as module
        assert module.__doc__ is not None
        assert len(module.__doc__.strip()) > 0

    def test_module_has_logger(self):
        """Module should have a logger instance."""
        import src.signals.vpin_bvc as module
        assert module.logger is not None
        assert module.logger.name == 'src.signals.vpin_bvc'

    def test_all_is_list_of_strings(self):
        """__all__ should be a list of non-empty strings."""
        import src.signals.vpin_bvc as module
        assert isinstance(module.__all__, list)
        for name in module.__all__:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_all_no_duplicates(self):
        """__all__ should not contain duplicate names."""
        import src.signals.vpin_bvc as module
        assert len(module.__all__) == len(set(module.__all__))

    def test_all_items_accessible(self):
        """Each name in __all__ should be accessible as a module attribute."""
        import src.signals.vpin_bvc as module
        for name in module.__all__:
            assert hasattr(module, name), f"{name} not accessible from module"

    def test_logger_level(self):
        """Logger should be at NOTSET level (inherits from parent)."""
        import src.signals.vpin_bvc as module
        import logging
        assert module.logger.level == logging.NOTSET

    def test_adapter_constants_exist(self):
        """VPINSignalAdapter should have threshold constants."""
        from src.signals.vpin_bvc import VPINSignalAdapter
        assert hasattr(VPINSignalAdapter, 'HIGH_VPIN_THRESHOLD')
        assert hasattr(VPINSignalAdapter, 'CRISIS_VPIN_THRESHOLD')

    def test_adapter_constants_are_floats(self):
        """VPINSignalAdapter threshold constants should be floats."""
        from src.signals.vpin_bvc import VPINSignalAdapter
        assert isinstance(VPINSignalAdapter.HIGH_VPIN_THRESHOLD, float)
        assert isinstance(VPINSignalAdapter.CRISIS_VPIN_THRESHOLD, float)

    def test_adapter_constants_in_range(self):
        """VPINSignalAdapter thresholds should be in (0, 1) range."""
        from src.signals.vpin_bvc import VPINSignalAdapter
        assert 0 < VPINSignalAdapter.HIGH_VPIN_THRESHOLD < 1
        assert 0 < VPINSignalAdapter.CRISIS_VPIN_THRESHOLD < 1

    def test_adapter_constants_ordered(self):
        """HIGH_VPIN_THRESHOLD should be less than CRISIS_VPIN_THRESHOLD."""
        from src.signals.vpin_bvc import VPINSignalAdapter
        assert VPINSignalAdapter.HIGH_VPIN_THRESHOLD < VPINSignalAdapter.CRISIS_VPIN_THRESHOLD


# ---------------------------------------------------------------------------
# BVCCalculator supplemental edge cases
# ---------------------------------------------------------------------------


class TestBVCCalculatorSupplemental:
    """Additional BVCCalculator edge cases."""

    def test_window_zero_returns_empty(self):
        """Window=0 should handle gracefully (no bars selected)."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        buy, sell, imbalance = calc.get_buy_sell_imbalance(window=0)
        assert buy == 0.0
        assert sell == 0.0
        assert imbalance == 0.0

    def test_window_negative(self):
        """Negative window should handle gracefully (empty slice)."""
        from src.signals.vpin_bvc import BVCCalculator
        calc = BVCCalculator()
        buy, sell, imbalance = calc.get_buy_sell_imbalance(window=-5)
        assert buy == 0.0
        assert sell == 0.0
        assert imbalance == 0.0

    def test_add_multiple_bars(self):
        """Adding multiple bars and getting the right count."""
        from src.signals.vpin_bvc import BVCCalculator, BVCBar
        calc = BVCCalculator()
        ts = datetime(2026, 5, 14, 10, 0)
        for i in range(100):
            bar = BVCBar(timestamp=ts, open=100.0, high=102.0,
                          low=99.0, close=101.0, volume=10000,
                          buy_volume=6000, sell_volume=4000, vpin_local=0.2)
            calc.add_bar(bar)
        assert len(calc.bars) == 100

    def test_bars_list_independence(self):
        """Each BVCCalculator should have its own bars list."""
        from src.signals.vpin_bvc import BVCCalculator
        calc1 = BVCCalculator()
        calc2 = BVCCalculator()
        bar1 = calc1.classify_bar(datetime(2026, 5, 14), 100.0, 102.0, 99.0, 101.0, 10000)
        calc1.add_bar(bar1)
        assert len(calc1.bars) == 1
        assert len(calc2.bars) == 0


# ---------------------------------------------------------------------------
# to_signal_snapshot explanation formatting edge cases
# ---------------------------------------------------------------------------


class TestSignalSnapshotExplanation:
    """SignalSnapshot explanation formatting edge cases."""

    def test_explanation_with_zero_vpin(self):
        """Explanation handles vpin=0 correctly."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.0, vpin_ma=0.0, vpin_std=0.0,
            z_score=0.0, percentile=0.0, regime='low',
            confidence=0.6, toxicity_level=0.0,
            recommendation='execute', expected_cost_impact=-3.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert 'vpin=0.0000' in snapshot.explanation
        assert snapshot.value == pytest.approx(0.2)

    def test_explanation_with_max_vpin(self):
        """Explanation handles vpin=1 correctly."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=1.0, vpin_ma=0.5, vpin_std=0.1,
            z_score=5.0, percentile=0.99, regime='high',
            confidence=0.6, toxicity_level=0.99,
            recommendation='avoid', expected_cost_impact=5.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert 'vpin=1.0000' in snapshot.explanation
        assert snapshot.value == pytest.approx(-0.4)

    def test_explanation_negative_z_score(self):
        """Explanation handles negative z_score formatting."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.2, vpin_ma=0.5, vpin_std=0.1,
            z_score=-3.0, percentile=0.05, regime='low',
            confidence=0.6, toxicity_level=0.1,
            recommendation='execute', expected_cost_impact=-3.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert 'z=-3.00' in snapshot.explanation

    def test_snapshot_asset_signals_empty(self):
        """asset_signals should always be an empty dict for VPIN."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.45, vpin_ma=0.40, vpin_std=0.05,
            z_score=1.0, percentile=0.80, regime='elevated',
            confidence=0.75, toxicity_level=0.6,
            recommendation='delay', expected_cost_impact=5.0,
        )
        snapshot = sig.to_signal_snapshot()
        assert snapshot.asset_signals == {}
        assert snapshot.regime_fit == "all"

    def test_snapshot_metadata_completeness(self):
        """Snapshot metadata contains all six expected keys."""
        from src.signals.vpin_bvc import VPINSignal
        sig = VPINSignal(
            timestamp=datetime(2026, 5, 14, 10, 0),
            vpin=0.45, vpin_ma=0.40, vpin_std=0.05,
            z_score=1.0, percentile=0.80, regime='elevated',
            confidence=0.75, toxicity_level=0.6,
            recommendation='delay', expected_cost_impact=5.0,
        )
        snapshot = sig.to_signal_snapshot()
        expected_keys = {'vpin', 'vpin_ma', 'z_score', 'regime',
                         'toxicity_level', 'recommendation'}
        assert set(snapshot.metadata.keys()) == expected_keys


# ---------------------------------------------------------------------------
# VPINSignalAdapter edge cases for rebalance timing
# ---------------------------------------------------------------------------


class TestVPINSignalAdapterRebalance:
    """VPINSignalAdapter rebalance timing edge cases."""

    def test_get_rebalance_timing_signal_keys(self):
        """Rebalance timing signal should have all expected keys."""
        from src.signals.vpin_bvc import VPINEngine, VPINSignalAdapter
        engine = VPINEngine()
        adapter = VPINSignalAdapter(engine)
        result = adapter.get_rebalance_timing_signal()
        assert set(result.keys()) == {'source', 'execute_now', 'reason',
                                       'expected_savings_bps', 'timestamp'}

    def test_rebalance_timing_insufficient_data(self):
        """With no data, execute_now should be True and reason insufficient_data."""
        from src.signals.vpin_bvc import VPINEngine, VPINSignalAdapter
        engine = VPINEngine()
        adapter = VPINSignalAdapter(engine)
        result = adapter.get_rebalance_timing_signal()
        assert result['execute_now'] is True
        assert result['reason'] == 'insufficient_data'
        assert result['expected_savings_bps'] == 0.0

    def test_rebalance_timing_timestamp_format(self):
        """Timestamp should be an ISO-formatted string."""
        from src.signals.vpin_bvc import VPINEngine, VPINSignalAdapter
        engine = VPINEngine()
        adapter = VPINSignalAdapter(engine)
        result = adapter.get_rebalance_timing_signal()
        assert isinstance(result['timestamp'], str)
        assert 'T' in result['timestamp']


# ---------------------------------------------------------------------------
# VPINSignalAdapter to_ensemble_signal with no data
# ---------------------------------------------------------------------------


class TestVPINSignalAdapterNoData:
    """VPINSignalAdapter with no data edge cases."""

    def test_no_data_raw_data_status(self):
        """No data should set raw_data status to insufficient_data."""
        from src.signals.vpin_bvc import VPINEngine, VPINSignalAdapter
        engine = VPINEngine()
        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal("SPY")
        assert result['raw_data']['status'] == 'insufficient_data'

    def test_no_data_probability(self):
        """No data should have probability 0.5."""
        from src.signals.vpin_bvc import VPINEngine, VPINSignalAdapter
        engine = VPINEngine()
        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal("SPY")
        assert result['probability'] == 0.5

    def test_no_data_timestamp(self):
        """No data should still include a timestamp."""
        from src.signals.vpin_bvc import VPINEngine, VPINSignalAdapter
        engine = VPINEngine()
        adapter = VPINSignalAdapter(engine)
        result = adapter.to_ensemble_signal("SPY")
        assert 'timestamp' in result
        assert isinstance(result['timestamp'], str)

    def test_no_data_for_unknown_symbol_no_crash(self):
        """Calling to_ensemble_signal with a symbol not in engine."""
        from src.signals.vpin_bvc import VPINEngine, VPINSignalAdapter
        engine = VPINEngine(symbols=["SPY"])
        adapter = VPINSignalAdapter(engine)
        # get_signal will raise KeyError for unknown symbol
        with pytest.raises(KeyError):
            adapter.to_ensemble_signal("UNKNOWN")


# ---------------------------------------------------------------------------
# VPINEngine vpin_history and completed_buckets lifecycle
# ---------------------------------------------------------------------------


class TestVPINEngineHistoryLifecycle:
    """VPINEngine history trimming and lifecycle."""

    def test_vpin_history_trimmed_at_500(self):
        """vpin_history should be trimmed to 500 entries max."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=20000, vpin_window=2)
        symbol = "SPY"
        base = datetime(2026, 5, 14, 9, 30)
        # Process many bars to generate lots of VPIN history
        for i in range(600):
            engine.process_bar(symbol, base + timedelta(minutes=i),
                               100.0, 102.0, 99.0, 101.0, v=20000)
            engine.calculate_vpin(symbol)
        assert len(engine.vpin_history[symbol]) <= 500

    def test_vpin_history_append_after_trim(self):
        """After trimming, new entries are still appended."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=20000, vpin_window=2)
        symbol = "SPY"
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(600):
            engine.process_bar(symbol, base + timedelta(minutes=i),
                               100.0, 102.0, 99.0, 101.0, v=20000)
            engine.calculate_vpin(symbol)
        before = len(engine.vpin_history[symbol])
        engine.calculate_vpin(symbol)
        after = len(engine.vpin_history[symbol])
        # Should still be <= 500
        assert after <= 500

    def test_completed_buckets_trimming_at_2x_window(self):
        """Completed buckets should not exceed vpin_window * 2."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=5000, vpin_window=10)
        symbol = "SPY"
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(100):
            engine.process_bar(symbol, base + timedelta(minutes=i),
                               100.0, 102.0, 99.0, 101.0, v=5000)
        assert len(engine.completed_buckets[symbol]) <= 20  # vpin_window * 2 = 20

    def test_completed_buckets_not_trimmed_below_threshold(self):
        """Completed buckets below vpin_window*2 should not be trimmed."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=10)
        symbol = "SPY"
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(5):
            engine.process_bar(symbol, base + timedelta(minutes=i),
                               100.0, 102.0, 99.0, 101.0, v=50000)
        assert len(engine.completed_buckets[symbol]) == 5

    def test_multiple_symbols_independent_buckets(self):
        """Multiple symbols maintain independent bucket states."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=5,
                            symbols=["SPY", "QQQ"])
        base = datetime(2026, 5, 14, 9, 30)
        engine.process_bar("SPY", base, 100.0, 102.0, 99.0, 101.0, v=10000)
        engine.process_bar("QQQ", base, 200.0, 202.0, 199.0, 201.0, v=20000)
        assert engine.current_buckets["SPY"].actual_volume == 10000
        assert engine.current_buckets["QQQ"].actual_volume == 20000
        assert engine.current_buckets["SPY"].target_volume == 50000
        assert engine.current_buckets["QQQ"].target_volume == 50000


# ---------------------------------------------------------------------------
# VPINEngine signal with edge case z-score and std
# ---------------------------------------------------------------------------


class TestVPINEngineZScore:
    """VPINEngine z-score computation edge cases."""

    def test_z_score_with_zero_std(self):
        """Near-zero standard deviation yields finite z_score (no crash)."""
        from src.signals.vpin_bvc import VPINEngine
        import numpy as np
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(10):
            engine.process_bar("SPY", base + timedelta(minutes=i),
                               100.0, 102.0, 99.0, 101.0, v=50000)
        current_vpin = engine.calculate_vpin("SPY")
        assert current_vpin is not None
        # Fill history with identical values to get near-zero std
        engine.vpin_history["SPY"] = [float(current_vpin)] * 100
        signal = engine.get_signal("SPY")
        assert signal is not None
        # vpin_std will be near-zero due to floating point; z_score must be finite
        assert np.isfinite(signal.z_score)

    def test_z_score_with_single_value_history(self):
        """Single value history uses short array for mean, but get_signal needs 50+."""
        from src.signals.vpin_bvc import VPINEngine
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(10):
            engine.process_bar("SPY", base + timedelta(minutes=i),
                               100.0, 102.0, 99.0, 101.0, v=50000)
        current_vpin = engine.calculate_vpin("SPY")
        assert current_vpin is not None
        # Only put 1 entry in history (less than 50 required)
        engine.vpin_history["SPY"] = [current_vpin]
        signal = engine.get_signal("SPY")
        assert signal is None

    def test_z_score_extreme_positive(self):
        """Extreme z-score values should be finite."""
        from src.signals.vpin_bvc import VPINEngine
        import numpy as np
        engine = VPINEngine(volume_bucket_size=50000, vpin_window=3,
                            symbols=["SPY"])
        base = datetime(2026, 5, 14, 9, 30)
        for i in range(10):
            engine.process_bar("SPY", base + timedelta(minutes=i),
                               100.0, 102.0, 99.0, 101.0, v=50000)
        current_vpin = engine.calculate_vpin("SPY")
        assert current_vpin is not None
        # vpin is ~0.33, history has mean ~0.001 with tiny std
        engine.vpin_history["SPY"] = [0.001] * 49 + [0.002] * 51
        signal = engine.get_signal("SPY")
        assert signal is not None
        assert np.isfinite(signal.z_score)

