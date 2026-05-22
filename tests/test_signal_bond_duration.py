"""
Tests for src/signals/bond_duration_signal.py — enums, dataclass,
BondDurationCalculator, and BondDurationSignalGenerator.
"""

import pytest
from unittest.mock import patch

from src.signals.bond_duration_signal import (
    YieldCurveRegime,
    RateDirection,
    DurationPosition,
    BondDurationSignal,
    BondDurationCalculator,
    BondDurationSignalGenerator,
    generate_bond_duration_signal,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_signal(**overrides):
    defaults = dict(
        timestamp="2026-01-01T00:00:00",
        yield_10y=4.50,
        yield_2y=4.00,
        spread_10y2y=0.50,
        curve_regime="normal",
        real_rate=2.0,
        real_rate_regime="neutral",
        rate_6m_ago=4.35,
        rate_change_6m=0.15,
        rate_direction="rising",
        tlt_weight=0.10,
        ief_weight=0.40,
        shy_weight=0.50,
        effective_duration=5.1,
        position="intermediate",
        confidence=70.0,
        is_valid=True,
        reason="test",
    )
    defaults.update(overrides)
    return BondDurationSignal(**defaults)


def _patch_ensure_dirs():
    return patch.object(BondDurationSignalGenerator, "_ensure_dirs", return_value=None)


# ── YieldCurveRegime enum ───────────────────────────────────────────


class TestYieldCurveRegime:
    def test_values(self):
        assert YieldCurveRegime.STEEP.value == "steep"
        assert YieldCurveRegime.NORMAL.value == "normal"
        assert YieldCurveRegime.FLAT.value == "flat"
        assert YieldCurveRegime.INVERTED.value == "inverted"

    def test_members(self):
        assert len(YieldCurveRegime) == 4


# ── RateDirection enum ──────────────────────────────────────────────


class TestRateDirection:
    def test_values(self):
        assert RateDirection.FALLING.value == "falling"
        assert RateDirection.STABLE.value == "stable"
        assert RateDirection.RISING.value == "rising"

    def test_members(self):
        assert len(RateDirection) == 3


# ── DurationPosition enum ───────────────────────────────────────────


class TestDurationPosition:
    def test_values(self):
        assert DurationPosition.LONG.value == "long"
        assert DurationPosition.INTERMEDIATE.value == "intermediate"
        assert DurationPosition.SHORT.value == "short"
        assert DurationPosition.BLEND.value == "blend"

    def test_members(self):
        assert len(DurationPosition) == 4


# ── BondDurationSignal dataclass ────────────────────────────────────


class TestBondDurationSignal:
    def test_to_dict(self):
        sig = _make_signal()
        d = sig.to_dict()
        assert isinstance(d, dict)
        assert d["curve_regime"] == "normal"
        assert d["confidence"] == 70.0

    def test_to_dict_has_all_fields(self):
        sig = _make_signal()
        d = sig.to_dict()
        expected_keys = {
            "timestamp", "yield_10y", "yield_2y", "spread_10y2y",
            "curve_regime", "real_rate", "real_rate_regime",
            "rate_6m_ago", "rate_change_6m", "rate_direction",
            "tlt_weight", "ief_weight", "shy_weight",
            "effective_duration", "position", "confidence",
            "is_valid", "reason",
        }
        assert set(d.keys()) == expected_keys


# ── BondDurationCalculator constants ─────────────────────────────────


class TestCalculatorConstants:
    def test_duration_mapping(self):
        assert BondDurationCalculator.DURATION["TLT"] == 16.0
        assert BondDurationCalculator.DURATION["IEF"] == 7.0
        assert BondDurationCalculator.DURATION["SHY"] == 2.0

    def test_spread_thresholds(self):
        assert BondDurationCalculator.SPREAD_STEEP == 1.0
        assert BondDurationCalculator.SPREAD_FLAT == 0.3
        assert BondDurationCalculator.SPREAD_INVERTED == 0.0

    def test_real_rate_thresholds(self):
        assert BondDurationCalculator.REAL_ATTRACTIVE == 2.0
        assert BondDurationCalculator.REAL_UNATTRACTIVE == 0.0

    def test_momentum_lookback(self):
        assert BondDurationCalculator.MOM_LOOKBACK_DAYS == 126


# ── classify_curve() ────────────────────────────────────────────────


class TestClassifyCurve:
    @pytest.fixture
    def calc(self):
        return BondDurationCalculator()

    def test_steep(self, calc):
        assert calc.classify_curve(1.5) == YieldCurveRegime.STEEP

    def test_steep_at_threshold(self, calc):
        """spread > 1.0 required; at 1.0 it's NORMAL."""
        assert calc.classify_curve(1.0) == YieldCurveRegime.NORMAL
        assert calc.classify_curve(1.01) == YieldCurveRegime.STEEP

    def test_normal(self, calc):
        assert calc.classify_curve(0.5) == YieldCurveRegime.NORMAL

    def test_normal_at_flat_boundary(self, calc):
        assert calc.classify_curve(0.3) == YieldCurveRegime.NORMAL

    def test_flat(self, calc):
        assert calc.classify_curve(0.15) == YieldCurveRegime.FLAT

    def test_flat_just_above_zero(self, calc):
        assert calc.classify_curve(0.01) == YieldCurveRegime.FLAT

    def test_inverted_at_zero(self, calc):
        assert calc.classify_curve(0.0) == YieldCurveRegime.INVERTED

    def test_inverted_negative(self, calc):
        assert calc.classify_curve(-0.5) == YieldCurveRegime.INVERTED


# ── classify_real_rate() ────────────────────────────────────────────


class TestClassifyRealRate:
    @pytest.fixture
    def calc(self):
        return BondDurationCalculator()

    def test_attractive(self, calc):
        assert calc.classify_real_rate(2.5) == "attractive"

    def test_attractive_at_threshold(self, calc):
        """real_rate > 2.0 → attractive."""
        assert calc.classify_real_rate(2.01) == "attractive"

    def test_neutral(self, calc):
        assert calc.classify_real_rate(1.0) == "neutral"

    def test_neutral_at_zero(self, calc):
        """real_rate >= 0.0 → neutral."""
        assert calc.classify_real_rate(0.0) == "neutral"

    def test_unattractive(self, calc):
        assert calc.classify_real_rate(-1.0) == "unattractive"

    def test_unattractive_just_below_zero(self, calc):
        assert calc.classify_real_rate(-0.01) == "unattractive"


# ── classify_rate_direction() ───────────────────────────────────────


class TestClassifyRateDirection:
    @pytest.fixture
    def calc(self):
        return BondDurationCalculator()

    def test_falling(self, calc):
        assert calc.classify_rate_direction(-0.50) == RateDirection.FALLING

    def test_falling_at_threshold(self, calc):
        assert calc.classify_rate_direction(-0.31) == RateDirection.FALLING

    def test_stable(self, calc):
        assert calc.classify_rate_direction(0.0) == RateDirection.STABLE

    def test_stable_near_falling(self, calc):
        assert calc.classify_rate_direction(-0.29) == RateDirection.STABLE

    def test_rising(self, calc):
        assert calc.classify_rate_direction(0.50) == RateDirection.RISING

    def test_rising_at_threshold(self, calc):
        assert calc.classify_rate_direction(0.31) == RateDirection.RISING

    def test_stable_near_rising(self, calc):
        assert calc.classify_rate_direction(0.29) == RateDirection.STABLE


# ── compute_duration_allocation() — full strategy matrix ────────────


class TestComputeDurationAllocation:
    @pytest.fixture
    def calc(self):
        return BondDurationCalculator()

    def test_steep_falling_long(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            1.2, 1.5, RateDirection.FALLING, YieldCurveRegime.STEEP
        )
        assert tlt == 0.70
        assert pos == "long"

    def test_steep_stable_long(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            1.2, 1.5, RateDirection.STABLE, YieldCurveRegime.STEEP
        )
        assert tlt == 0.50
        assert pos == "long"

    def test_steep_rising_intermediate(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            1.2, 1.5, RateDirection.RISING, YieldCurveRegime.STEEP
        )
        assert tlt == 0.30
        assert pos == "intermediate"

    def test_normal_falling_long(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            0.5, 1.0, RateDirection.FALLING, YieldCurveRegime.NORMAL
        )
        assert tlt == 0.40
        assert pos == "long"

    def test_normal_stable_intermediate(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            0.5, 1.0, RateDirection.STABLE, YieldCurveRegime.NORMAL
        )
        assert ief == 0.50
        assert pos == "intermediate"

    def test_normal_rising_intermediate(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            0.5, 1.0, RateDirection.RISING, YieldCurveRegime.NORMAL
        )
        assert shy == 0.50
        assert pos == "intermediate"

    def test_flat_falling_intermediate(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            0.15, 0.5, RateDirection.FALLING, YieldCurveRegime.FLAT
        )
        assert ief == 0.50
        assert pos == "intermediate"

    def test_flat_stable_short(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            0.15, 0.5, RateDirection.STABLE, YieldCurveRegime.FLAT
        )
        assert shy == 0.50
        assert pos == "short"

    def test_flat_rising_short(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            0.15, 0.5, RateDirection.RISING, YieldCurveRegime.FLAT
        )
        assert shy == 0.70
        assert pos == "short"

    def test_inverted_falling_intermediate(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            -0.3, -0.5, RateDirection.FALLING, YieldCurveRegime.INVERTED
        )
        assert shy == 0.50
        assert pos == "intermediate"

    def test_inverted_stable_short(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            -0.3, -0.5, RateDirection.STABLE, YieldCurveRegime.INVERTED
        )
        assert shy == 0.70
        assert pos == "short"

    def test_inverted_rising_short(self, calc):
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            -0.3, -0.5, RateDirection.RISING, YieldCurveRegime.INVERTED
        )
        assert shy == 0.80
        assert pos == "short"

    def test_weights_sum_to_one(self, calc):
        """All allocation combos should sum to ~1.0."""
        for regime in YieldCurveRegime:
            for direction in RateDirection:
                tlt, ief, shy, _ = calc.compute_duration_allocation(
                    0.5, 1.0, direction, regime
                )
                assert abs(tlt + ief + shy - 1.0) < 0.01, (
                    f"{regime.value}/{direction.value}: {tlt+ief+shy}"
                )

    def test_real_rate_attractive_tilts_tlt(self, calc):
        """Attractive real rate (>2%) should boost TLT from SHY."""
        tlt_base, _, shy_base, _ = calc.compute_duration_allocation(
            0.5, 0.5, RateDirection.STABLE, YieldCurveRegime.NORMAL
        )
        tlt_boost, _, shy_boost, _ = calc.compute_duration_allocation(
            0.5, 2.5, RateDirection.STABLE, YieldCurveRegime.NORMAL
        )
        assert tlt_boost > tlt_base
        assert shy_boost < shy_base

    def test_real_rate_attractive_no_tilt_when_long(self, calc):
        """Real rate tilt only applies when position != LONG."""
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            1.2, 2.5, RateDirection.FALLING, YieldCurveRegime.STEEP
        )
        # STEEP+FALLING is already LONG, so no tilt applied
        assert tlt == 0.70


# ── compute_effective_duration() ────────────────────────────────────


class TestComputeEffectiveDuration:
    @pytest.fixture
    def calc(self):
        return BondDurationCalculator()

    def test_all_tlt(self, calc):
        dur = calc.compute_effective_duration(1.0, 0.0, 0.0)
        assert dur == 16.0

    def test_all_ief(self, calc):
        dur = calc.compute_effective_duration(0.0, 1.0, 0.0)
        assert dur == 7.0

    def test_all_shy(self, calc):
        dur = calc.compute_effective_duration(0.0, 0.0, 1.0)
        assert dur == 2.0

    def test_blend(self, calc):
        dur = calc.compute_effective_duration(0.5, 0.3, 0.2)
        assert dur == pytest.approx(0.5 * 16 + 0.3 * 7 + 0.2 * 2)


# ── BondDurationSignalGenerator.generate_signal() ───────────────────


class TestGenerateSignal:
    @pytest.fixture
    def gen(self):
        with _patch_ensure_dirs():
            return BondDurationSignalGenerator()

    def test_basic_signal(self, gen):
        sig = gen.generate_signal(yield_10y=4.5, yield_2y=4.0)
        assert sig.is_valid is True
        assert sig.spread_10y2y == 0.50
        assert sig.curve_regime == "normal"

    def test_inverted_curve(self, gen):
        sig = gen.generate_signal(yield_10y=3.5, yield_2y=4.0)
        assert sig.curve_regime == "inverted"
        assert sig.position == "short"

    def test_steep_curve(self, gen):
        sig = gen.generate_signal(yield_10y=5.5, yield_2y=3.5)
        assert sig.curve_regime == "steep"

    def test_real_rate_default(self, gen):
        """Default real_rate = yield_10y - 2.5."""
        sig = gen.generate_signal(yield_10y=4.5, yield_2y=4.0)
        assert sig.real_rate == 2.0

    def test_real_rate_custom(self, gen):
        sig = gen.generate_signal(yield_10y=4.5, yield_2y=4.0, real_rate=3.0)
        assert sig.real_rate == 3.0

    def test_rate_change_default(self, gen):
        """Default rate_change_6m = 0.15."""
        sig = gen.generate_signal(yield_10y=4.5, yield_2y=4.0)
        assert sig.rate_change_6m == 0.15

    def test_rate_change_custom(self, gen):
        sig = gen.generate_signal(yield_10y=4.5, yield_2y=4.0, rate_change_6m=-0.50)
        assert sig.rate_direction == "falling"

    def test_confidence_inverted_rising(self, gen):
        sig = gen.generate_signal(yield_10y=3.5, yield_2y=4.0, rate_change_6m=0.50)
        assert sig.confidence == 90.0

    def test_confidence_steep_falling(self, gen):
        sig = gen.generate_signal(yield_10y=5.5, yield_2y=3.5, rate_change_6m=-0.50)
        assert sig.confidence == 90.0

    def test_confidence_near_flat(self, gen):
        """Spread < 0.15 → low confidence."""
        sig = gen.generate_signal(yield_10y=4.10, yield_2y=4.00)
        assert sig.confidence == 55.0

    def test_confidence_default(self, gen):
        sig = gen.generate_signal(yield_10y=4.5, yield_2y=4.0)
        assert sig.confidence == 70.0

    def test_reason_string(self, gen):
        sig = gen.generate_signal(yield_10y=4.5, yield_2y=4.0)
        assert "Curve=" in sig.reason
        assert "Rate=" in sig.reason
        assert "Real=" in sig.reason

    def test_rate_6m_ago_computed(self, gen):
        sig = gen.generate_signal(yield_10y=4.5, yield_2y=4.0, rate_change_6m=0.30)
        assert sig.rate_6m_ago == pytest.approx(4.2, abs=0.01)

    def test_effective_duration_populated(self, gen):
        sig = gen.generate_signal(yield_10y=4.5, yield_2y=4.0)
        assert sig.effective_duration > 0

    def test_timestamp_present(self, gen):
        sig = gen.generate_signal(yield_10y=4.5, yield_2y=4.0)
        assert sig.timestamp is not None
        assert len(sig.timestamp) > 0

    def test_fetch_yield_data_called_when_none(self, gen):
        with patch.object(gen, "_fetch_yield_data", return_value={"yield_10y": 4.0, "yield_2y": 3.5}):
            sig = gen.generate_signal()
        assert sig.yield_10y == 4.0

    def test_fetch_yield_data_partial(self, gen):
        """When only one yield is None, fetch and fill."""
        with patch.object(gen, "_fetch_yield_data", return_value={"yield_10y": 4.0, "yield_2y": 3.5}):
            sig = gen.generate_signal(yield_2y=3.8)
        assert sig.yield_10y == 4.0
        assert sig.yield_2y == 3.8  # Provided value takes precedence


# ── generate_bond_duration_signal() convenience ─────────────────────


class TestConvenienceFunction:
    def test_returns_signal(self):
        with _patch_ensure_dirs():
            sig = generate_bond_duration_signal(yield_10y=4.5, yield_2y=4.0)
        assert isinstance(sig, BondDurationSignal)
        assert sig.is_valid is True

    def test_passes_params(self):
        with _patch_ensure_dirs():
            sig = generate_bond_duration_signal(
                yield_10y=5.5, yield_2y=3.5, real_rate=3.0, rate_change_6m=-0.5
            )
        assert sig.curve_regime == "steep"
        assert sig.real_rate == 3.0
        assert sig.rate_direction == "falling"
