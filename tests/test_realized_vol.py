"""
Tests for Realized Volatility Pipeline (v5.10)
"""

import pytest
import numpy as np
from src.data.realized_vol import (
    RealizedVolCalculator, RealizedVolPipeline, RealizedVolResult,
    OHLCBar, compute_realized_vol,
)


def make_bars(n=100, trend=0.0, daily_range=0.015):
    """Generate synthetic OHLC bars."""
    rng = np.random.RandomState(42)
    bars = []
    price = 100.0
    for i in range(n):
        o = price
        ret = rng.normal(trend / 252, daily_range / math.sqrt(2))
        c = o * (1 + ret)
        h = max(o, c) * (1 + abs(rng.normal(0, daily_range * 0.3)))
        l = min(o, c) * (1 - abs(rng.normal(0, daily_range * 0.3)))
        price = c
        bars.append(OHLCBar(date=f"2026-{(i//21)+1:02d}-{(i%21)+1:02d}",
                            open=round(o, 2), high=round(h, 2),
                            low=round(l, 2), close=round(c, 2)))
    return bars


import math


class TestRealizedVolCalculator:
    @pytest.fixture
    def calc(self):
        return RealizedVolCalculator()

    @pytest.fixture
    def bars(self):
        return make_bars(100)

    def test_garman_klass_positive(self, calc, bars):
        o = np.array([b.open for b in bars])
        h = np.array([b.high for b in bars])
        l = np.array([b.low for b in bars])
        c = np.array([b.close for b in bars])
        gk = calc.garman_klass(o, h, l, c)
        assert gk > 0

    def test_parkinson_positive(self, calc, bars):
        h = np.array([b.high for b in bars])
        l = np.array([b.low for b in bars])
        pk = calc.parkinson(h, l)
        assert pk > 0

    def test_rogers_satchell_positive(self, calc, bars):
        o = np.array([b.open for b in bars])
        h = np.array([b.high for b in bars])
        l = np.array([b.low for b in bars])
        c = np.array([b.close for b in bars])
        rs = calc.rogers_satchell(o, h, l, c)
        assert rs > 0

    def test_yang_zhang_positive(self, calc, bars):
        o = np.array([b.open for b in bars])
        h = np.array([b.high for b in bars])
        l = np.array([b.low for b in bars])
        c = np.array([b.close for b in bars])
        yz = calc.yang_zhang(o, h, l, c)
        assert yz > 0

    def test_close_to_close_matches_std(self, calc, bars):
        c = np.array([b.close for b in bars])
        cc = calc.close_to_close(c)
        returns = np.diff(np.log(c))
        expected = np.std(returns) * math.sqrt(252)
        assert abs(cc - expected) < 0.001

    def test_composite_is_average(self, calc, bars):
        result = calc.compute(bars, window=20)
        if result.is_valid:
            assert result.composite > 0
            # Should be within range of individual estimators
            ests = [result.garman_klass, result.parkinson,
                    result.rogers_satchell, result.yang_zhang]
            valid = [v for v in ests if v > 0.001]
            if valid:
                assert min(valid) <= result.composite <= max(valid) + 0.01

    def test_insufficient_bars(self, calc):
        result = calc.compute([], window=20)
        assert not result.is_valid

    def test_constant_prices(self, calc):
        bars = [OHLCBar(date=f"2026-01-{(i+1):02d}", open=100, high=100,
                        low=100, close=100) for i in range(50)]
        result = calc.compute(bars, window=20)
        # All estimators should be near zero for constant prices
        assert result.close_to_close < 0.01

    def test_small_window(self, calc):
        bars = make_bars(5)
        result = calc.compute(bars, window=20)
        assert not result.is_valid or result.n_bars < 10


class TestRealizedVolPipeline:
    @pytest.fixture
    def pipeline(self):
        return RealizedVolPipeline()

    def test_compute_current(self, pipeline):
        result = pipeline.compute_current("SPY", window=20)
        assert isinstance(result, RealizedVolResult)
        if result.is_valid:
            assert result.garman_klass >= 0

    def test_convenience_function(self):
        result = compute_realized_vol("SPY", window=20)
        assert isinstance(result, RealizedVolResult)

    def test_result_serializable(self, pipeline):
        result = pipeline.compute_current("SPY", window=10)
        d = result.to_dict()
        assert "garman_klass" in d
        assert "composite" in d
