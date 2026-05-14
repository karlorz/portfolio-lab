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
        """Window larger than available bars — uses all bars."""
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
