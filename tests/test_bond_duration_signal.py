"""
Tests for Bond Duration Rotation Signal Generator (v4.80)
"""

import pytest
from datetime import datetime, date

from src.signals.bond_duration_signal import (
    BondDurationCalculator,
    BondDurationSignalGenerator,
    BondDurationSignal,
    YieldCurveRegime,
    RateDirection,
    DurationPosition,
    generate_bond_duration_signal,
)


class TestYieldCurveClassification:
    """Test yield curve regime classification."""

    @pytest.fixture
    def calc(self):
        return BondDurationCalculator()

    def test_steep_curve(self, calc):
        assert calc.classify_curve(1.5) == YieldCurveRegime.STEEP
        assert calc.classify_curve(2.0) == YieldCurveRegime.STEEP

    def test_normal_curve(self, calc):
        assert calc.classify_curve(0.5) == YieldCurveRegime.NORMAL
        assert calc.classify_curve(0.9) == YieldCurveRegime.NORMAL
        assert calc.classify_curve(1.0) == YieldCurveRegime.NORMAL

    def test_flat_curve(self, calc):
        assert calc.classify_curve(0.15) == YieldCurveRegime.FLAT
        assert calc.classify_curve(0.05) == YieldCurveRegime.FLAT
        assert calc.classify_curve(0.3) == YieldCurveRegime.NORMAL  # boundary

    def test_inverted_curve(self, calc):
        assert calc.classify_curve(-0.5) == YieldCurveRegime.INVERTED
        assert calc.classify_curve(-1.0) == YieldCurveRegime.INVERTED
        assert calc.classify_curve(-0.01) == YieldCurveRegime.INVERTED

    def test_boundary_zero_is_inverted(self, calc):
        assert calc.classify_curve(0.0) == YieldCurveRegime.INVERTED


class TestRealRateClassification:
    """Test real rate classification."""

    @pytest.fixture
    def calc(self):
        return BondDurationCalculator()

    def test_attractive(self, calc):
        assert calc.classify_real_rate(2.5) == "attractive"
        assert calc.classify_real_rate(3.0) == "attractive"

    def test_neutral(self, calc):
        assert calc.classify_real_rate(1.0) == "neutral"
        assert calc.classify_real_rate(0.5) == "neutral"
        assert calc.classify_real_rate(2.0) == "neutral"

    def test_unattractive(self, calc):
        assert calc.classify_real_rate(-1.0) == "unattractive"
        assert calc.classify_real_rate(-0.5) == "unattractive"
        assert calc.classify_real_rate(0.0) == "neutral"


class TestRateDirection:
    """Test rate direction classification."""

    @pytest.fixture
    def calc(self):
        return BondDurationCalculator()

    def test_falling(self, calc):
        assert calc.classify_rate_direction(-0.50) == RateDirection.FALLING
        assert calc.classify_rate_direction(-1.00) == RateDirection.FALLING

    def test_stable(self, calc):
        assert calc.classify_rate_direction(0.0) == RateDirection.STABLE
        assert calc.classify_rate_direction(0.20) == RateDirection.STABLE
        assert calc.classify_rate_direction(-0.20) == RateDirection.STABLE

    def test_rising(self, calc):
        assert calc.classify_rate_direction(0.50) == RateDirection.RISING
        assert calc.classify_rate_direction(1.00) == RateDirection.RISING

    def test_boundary_values(self, calc):
        assert calc.classify_rate_direction(-0.30) == RateDirection.STABLE
        assert calc.classify_rate_direction(0.30) == RateDirection.STABLE
        assert calc.classify_rate_direction(-0.31) == RateDirection.FALLING
        assert calc.classify_rate_direction(0.31) == RateDirection.RISING


class TestDurationAllocation:
    """Test duration allocation matrix."""

    @pytest.fixture
    def calc(self):
        return BondDurationCalculator()

    def test_steep_falling_goes_long(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            1.5, 1.0, RateDirection.FALLING, YieldCurveRegime.STEEP
        )
        assert tlt > 0.5
        assert pos == "long"

    def test_inverted_rising_goes_short(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            -0.5, 1.0, RateDirection.RISING, YieldCurveRegime.INVERTED
        )
        assert shy > 0.5
        assert tlt == 0.0
        assert pos == "short"

    def test_normal_stable_intermediate(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            0.5, 1.0, RateDirection.STABLE, YieldCurveRegime.NORMAL
        )
        assert ief > 0.3
        assert pos == "intermediate"

    def test_flat_rising_defensive(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            0.15, 1.0, RateDirection.RISING, YieldCurveRegime.FLAT
        )
        assert shy > ief
        assert tlt < ief

    def test_inverted_falling_intermediate(self, calc):
        """Even when curve is inverted, falling rates favor some duration."""
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            -0.3, 1.0, RateDirection.FALLING, YieldCurveRegime.INVERTED
        )
        assert pos == "intermediate"
        assert tlt >= 0  # Some TLT allowed

    def test_weights_sum_to_one(self, calc):
        """All weight combinations should sum to 1.0."""
        regimes = list(YieldCurveRegime)
        directions = list(RateDirection)
        for regime in regimes:
            for direction in directions:
                tlt, ief, shy, _ = calc.compute_duration_allocation(
                    0.5, 1.0, direction, regime
                )
                total = tlt + ief + shy
                assert abs(total - 1.0) < 0.01, \
                    f"{regime.value}/{direction.value}: {total}"

    def test_real_rate_boost(self, calc):
        """High real rate should shift toward longer duration."""
        # Same regime/direction, different real rates
        tlt1, ief1, shy1, _ = calc.compute_duration_allocation(
            0.5, 1.0, RateDirection.STABLE, YieldCurveRegime.NORMAL
        )
        tlt2, ief2, shy2, _ = calc.compute_duration_allocation(
            0.5, 3.0, RateDirection.STABLE, YieldCurveRegime.NORMAL
        )
        # Higher real rate should increase TLT or keep same
        assert tlt2 >= tlt1

    def test_all_12_cases_valid(self, calc):
        """All 12 regime × direction combos should produce valid weights."""
        regimes = list(YieldCurveRegime)
        directions = list(RateDirection)
        positions_seen = set()
        for regime in regimes:
            for direction in directions:
                tlt, ief, shy, pos = calc.compute_duration_allocation(
                    0.5, 1.0, direction, regime
                )
                assert 0 <= tlt <= 1
                assert 0 <= ief <= 1
                assert 0 <= shy <= 1
                assert pos in ("long", "intermediate", "short")
                positions_seen.add(pos)
        # Should see all 3 positions across the matrix
        assert len(positions_seen) >= 3


class TestEffectiveDuration:
    """Test effective duration computation."""

    @pytest.fixture
    def calc(self):
        return BondDurationCalculator()

    def test_all_tlt_max_duration(self, calc):
        dur = calc.compute_effective_duration(1.0, 0.0, 0.0)
        assert dur == 16.0

    def test_all_shy_min_duration(self, calc):
        dur = calc.compute_effective_duration(0.0, 0.0, 1.0)
        assert dur == 2.0

    def test_blend_intermediate(self, calc):
        dur = calc.compute_effective_duration(0.2, 0.3, 0.5)
        expected = 0.2 * 16 + 0.3 * 7 + 0.5 * 2
        assert abs(dur - expected) < 0.1

    def test_equal_weight(self, calc):
        dur = calc.compute_effective_duration(0.34, 0.33, 0.33)
        # ~8.3 years
        assert 7.0 < dur < 10.0


class TestSignalGeneration:
    """Test complete signal generation."""

    @pytest.fixture
    def generator(self):
        return BondDurationSignalGenerator()

    def test_generate_default_signal(self, generator):
        signal = generator.generate_signal()
        assert isinstance(signal, BondDurationSignal)
        assert signal.is_valid
        assert signal.yield_10y > 0
        assert signal.yield_2y > 0

    def test_generate_with_explicit_params(self, generator):
        signal = generator.generate_signal(
            yield_10y=5.0, yield_2y=4.5, real_rate=2.5, rate_change_6m=-0.5
        )
        assert signal.spread_10y2y == 0.5
        assert signal.curve_regime == "normal"
        assert signal.rate_direction == "falling"
        assert signal.real_rate_regime == "attractive"

    def test_signal_serializable(self, generator):
        signal = generator.generate_signal()
        d = signal.to_dict()
        assert isinstance(d, dict)
        assert "tlt_weight" in d
        assert "effective_duration" in d

    def test_convenience_function(self):
        signal = generate_bond_duration_signal(
            yield_10y=4.5, yield_2y=4.0
        )
        assert isinstance(signal, BondDurationSignal)
        assert signal.spread_10y2y == 0.5

    def test_inverted_curve_signal(self, generator):
        signal = generator.generate_signal(
            yield_10y=4.0, yield_2y=4.5, real_rate=1.5, rate_change_6m=0.5
        )
        assert signal.curve_regime == "inverted"
        assert signal.rate_direction == "rising"
        # Should favor SHY
        assert signal.shy_weight > signal.tlt_weight

    def test_steep_curve_signal(self, generator):
        signal = generator.generate_signal(
            yield_10y=5.0, yield_2y=3.5, real_rate=2.5, rate_change_6m=-0.8
        )
        assert signal.curve_regime == "steep"
        assert signal.rate_direction == "falling"
        # Should favor TLT
        assert signal.tlt_weight > 0.4

    def test_confidence_varied(self, generator):
        """Different regimes should produce different confidence."""
        sig1 = generator.generate_signal(yield_10y=5.0, yield_2y=3.5, rate_change_6m=-0.8)
        sig2 = generator.generate_signal(yield_10y=4.5, yield_2y=4.4, rate_change_6m=0.1)
        # They should differ
        assert sig1.confidence != sig2.confidence

    def test_duration_in_range(self, generator):
        signal = generator.generate_signal()
        assert 2.0 <= signal.effective_duration <= 16.0


class TestEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def calc(self):
        return BondDurationCalculator()

    def test_very_large_spread(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            4.0, 2.0, RateDirection.FALLING, YieldCurveRegime.STEEP
        )
        assert pos == "long"

    def test_very_negative_spread(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            -3.0, 1.0, RateDirection.RISING, YieldCurveRegime.INVERTED
        )
        assert pos == "short"

    def test_zero_weights_valid(self, calc):
        """Some weights can be zero, but never negative."""
        for regime in YieldCurveRegime:
            for direction in RateDirection:
                tlt, ief, shy, _ = calc.compute_duration_allocation(
                    0.5, 1.0, direction, regime
                )
                assert tlt >= 0
                assert ief >= 0
                assert shy >= 0

    def test_extreme_real_rate_still_valid(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            1.0, 10.0, RateDirection.FALLING, YieldCurveRegime.STEEP
        )
        assert abs(tlt + ief + shy - 1.0) < 0.01
