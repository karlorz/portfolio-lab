"""
Tests for FX Currency Carry Signal Generator (v3.15)
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.signals.fx_carry_signal import FXCarrySignal, FXCarrySignalGenerator


def make_metrics(
    uup_return=3.0, udn_return=-2.0, usd_strength=0.6,
    carry="positive", direction="bullish", vol_regime="low",
    freshness=0.5,
):
    from src.data.fx_fetcher import FXMetrics
    return FXMetrics(
        timestamp="2026-05-16T00:00:00",
        uup_price=28.5, udn_price=18.2,
        uup_return_30d=uup_return, udn_return_30d=udn_return,
        usd_strength_score=usd_strength, carry_regime=carry,
        momentum_direction=direction, volatility_regime=vol_regime,
        data_freshness_hours=freshness,
    )


class TestFXCarrySignal:
    def test_defaults(self):
        sig = FXCarrySignal(
            signal_type="neutral", confidence=0.0, regime="neutral",
            direction="neutral", reason="test",
            spy_shift=0.0, efa_shift=0.0, vxus_shift=0.0,
            is_valid=False,
        )
        assert sig.signal_type == "neutral"
        assert not sig.is_valid

    def test_bullish_signal(self):
        sig = FXCarrySignal(
            signal_type="usd_strength", confidence=0.75, regime="positive",
            direction="bullish", reason="momentum_aligned",
            spy_shift=1.5, efa_shift=-1.5, vxus_shift=-1.5,
            is_valid=True,
        )
        assert sig.is_valid
        assert sig.spy_shift > 0
        assert sig.efa_shift < 0


class TestFXCarrySignalGenerator:
    @pytest.fixture
    def gen(self, tmp_path):
        hist = tmp_path / "fx_history.json"
        with patch('src.signals.fx_carry_signal.FXFetcher', autospec=True):
            g = FXCarrySignalGenerator(signal_history_path=hist)
        return g

    def test_calculate_confidence_bullish(self, gen):
        metrics = make_metrics(uup_return=3.0, direction="bullish")
        conf = gen._calculate_confidence(metrics, "usd_strength")
        assert 0.5 < conf <= 1.0

    def test_calculate_confidence_neutral_returns_zero(self, gen):
        metrics = make_metrics()
        conf = gen._calculate_confidence(metrics, "neutral")
        assert conf == 0.0

    def test_calculate_confidence_capped(self, gen):
        metrics = make_metrics(uup_return=10.0)
        conf = gen._calculate_confidence(metrics, "usd_strength")
        assert conf == 1.0  # Capped

    def test_allocation_shifts_neutral(self, gen):
        s, e, v = gen._calculate_allocation_shifts("neutral", 0.5)
        assert s == 0.0 and e == 0.0 and v == 0.0

    def test_allocation_shifts_usd_strength(self, gen):
        s, e, v = gen._calculate_allocation_shifts("usd_strength", 0.75)
        assert s > 0  # Add SPY
        assert e < 0  # Reduce international
        assert s == -e  # Symmetric

    def test_allocation_shifts_usd_weakness(self, gen):
        s, e, v = gen._calculate_allocation_shifts("usd_weakness", 0.75)
        assert s < 0  # Reduce SPY
        assert e > 0  # Add international

    def test_allocation_shifts_capped_at_max(self, gen):
        s, e, v = gen._calculate_allocation_shifts("usd_strength", 1.0)
        assert s <= gen.MAX_SHIFT

    def test_momentum_conflict_detected(self, gen):
        metrics = make_metrics(uup_return=1.0, udn_return=1.0)
        assert gen._check_momentum_conflict(metrics)

    def test_momentum_no_conflict(self, gen):
        metrics = make_metrics(uup_return=3.0, udn_return=-2.0)
        assert not gen._check_momentum_conflict(metrics)

    def test_generate_signal_bullish(self, gen):
        metrics = make_metrics(uup_return=3.0, udn_return=-2.0,
                               direction="bullish", vol_regime="low")
        with patch.object(gen.fetcher, 'fetch_metrics', return_value=metrics):
            # First call: insufficient persistence (< 5 days)
            s1 = gen.generate_signal()
            assert s1.reason == "insufficient_persistence"
            # Generate 4 more calls to build persistence
            for _ in range(4):
                gen.generate_signal()
            # 6th call should be valid
            signal = gen.generate_signal()
            assert signal.signal_type == "usd_strength"
            assert signal.is_valid
            assert signal.confidence > 0

    def test_generate_signal_high_vol_returns_neutral(self, gen):
        metrics = make_metrics(vol_regime="high")
        with patch.object(gen.fetcher, 'fetch_metrics', return_value=metrics):
            signal = gen.generate_signal()
            assert signal.signal_type == "neutral"
            assert not signal.is_valid
            assert signal.reason == "high_volatility"

    def test_generate_signal_conflict_returns_neutral(self, gen):
        metrics = make_metrics(uup_return=1.0, udn_return=1.0)
        with patch.object(gen.fetcher, 'fetch_metrics', return_value=metrics):
            signal = gen.generate_signal()
            assert signal.signal_type == "neutral"
            assert signal.reason == "momentum_conflict"

    def test_generate_signal_fetch_error(self, gen):
        with patch.object(gen.fetcher, 'fetch_metrics',
                          side_effect=Exception("API error")):
            signal = gen.generate_signal()
            assert signal.signal_type == "neutral"
            assert not signal.is_valid
            assert signal.reason == "data_error"

    def test_get_ensemble_input(self, gen):
        metrics = make_metrics(direction="bullish", vol_regime="low")
        with patch.object(gen.fetcher, 'fetch_metrics', return_value=metrics):
            result = gen.get_ensemble_input()
            assert result["source"] == "fx_carry"
            assert "allocation_shifts" in result
