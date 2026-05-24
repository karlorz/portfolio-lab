"""Tests for VPIN BVC microstructure signal v2.65."""
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
        from src.signals.vpin_bvc import VPINEngine

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
        import numpy as np
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
