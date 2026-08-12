"""
Tests for Bond Duration Rotation Signal Generator (v4.80)
"""

import json
import pytest

from src.signals.bond_duration_signal import (
    BondDurationCalculator,
    BondDurationSignalGenerator,
    BondDurationSignal,
    YieldCurveRegime,
    RateDirection,
    DurationPosition,
    generate_bond_duration_signal,
)
from src.signals.signal_snapshot import SignalSnapshot


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
        assert signal.yield_10y > 0
        assert signal.yield_2y > 0
        # Live SSOT / DB → valid; pure textbook defaults → not valid
        if signal.using_defaults:
            assert signal.is_valid is False
            assert signal.source_status == "degraded"
        else:
            assert signal.is_valid is True

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


# ── New test classes: 23 additional tests covering 7 areas ──────────────

class TestBondDurationSignalDataclass:
    """Area 1: to_dict() field completeness and enum values."""

    def test_to_dict_all_fields(self):
        """to_dict() should contain all BondDurationSignal fields incl. provenance."""
        signal = generate_bond_duration_signal(yield_10y=4.5, yield_2y=4.0)
        d = signal.to_dict()
        expected_keys = {
            "timestamp", "yield_10y", "yield_2y", "spread_10y2y",
            "curve_regime", "real_rate", "real_rate_regime",
            "rate_6m_ago", "rate_change_6m", "rate_direction",
            "tlt_weight", "ief_weight", "shy_weight",
            "effective_duration", "position", "confidence", "is_valid",
            "reason",
            "using_defaults", "source_mode", "source_status",
        }
        assert set(d.keys()) == expected_keys, f"Missing keys: {expected_keys - set(d.keys())}"

    def test_to_dict_values_match(self):
        """Field values survive round-trip through to_dict()."""
        signal = generate_bond_duration_signal(
            yield_10y=5.0, yield_2y=4.5, real_rate=2.5, rate_change_6m=-0.5
        )
        d = signal.to_dict()
        assert d["yield_10y"] == signal.yield_10y
        assert d["yield_2y"] == signal.yield_2y
        assert d["spread_10y2y"] == signal.spread_10y2y
        assert d["curve_regime"] == signal.curve_regime
        assert d["real_rate"] == signal.real_rate
        assert d["real_rate_regime"] == signal.real_rate_regime
        assert d["tlt_weight"] == signal.tlt_weight
        assert d["ief_weight"] == signal.ief_weight
        assert d["shy_weight"] == signal.shy_weight
        assert d["effective_duration"] == signal.effective_duration
        assert d["position"] == signal.position
        assert d["confidence"] == signal.confidence
        assert d["is_valid"] == signal.is_valid
        assert d["reason"] == signal.reason

    def test_duration_position_enum_values(self):
        assert DurationPosition.LONG.value == "long"
        assert DurationPosition.INTERMEDIATE.value == "intermediate"
        assert DurationPosition.SHORT.value == "short"
        assert DurationPosition.BLEND.value == "blend"

    def test_yield_curve_regime_enum_values(self):
        assert YieldCurveRegime.STEEP.value == "steep"
        assert YieldCurveRegime.NORMAL.value == "normal"
        assert YieldCurveRegime.FLAT.value == "flat"
        assert YieldCurveRegime.INVERTED.value == "inverted"


class TestConstants:
    """Area 4: Constants validation."""

    def test_calculator_thresholds_consistent(self):
        """Threshold hierarchy: steep > flat > inverted."""
        calc = BondDurationCalculator()
        assert calc.SPREAD_STEEP > calc.SPREAD_FLAT > calc.SPREAD_INVERTED
        assert calc.SPREAD_FLAT > 0
        assert calc.SPREAD_INVERTED == 0.0

    def test_duration_mappings_positive(self):
        """All duration constants should be positive."""
        calc = BondDurationCalculator()
        for etf, dur in calc.DURATION.items():
            assert dur > 0, f"{etf} duration must be positive"
        assert calc.DURATION["TLT"] > calc.DURATION["IEF"] > calc.DURATION["SHY"]

    def test_mom_lookback_positive(self):
        """MOM_LOOKBACK_DAYS should be ~6 months of trading days."""
        calc = BondDurationCalculator()
        assert calc.MOM_LOOKBACK_DAYS == 126
        assert 120 <= calc.MOM_LOOKBACK_DAYS <= 130

    def test_real_rate_thresholds_ordered(self):
        """Attractive threshold should be above unattractive threshold."""
        calc = BondDurationCalculator()
        assert calc.REAL_ATTRACTIVE > calc.REAL_UNATTRACTIVE
        assert calc.REAL_UNATTRACTIVE == 0.0


class TestPreciseBoundaries:
    """Area 3 + 6: Exact boundary values for classifier predicates."""

    @pytest.fixture
    def calc(self):
        return BondDurationCalculator()

    def test_classify_curve_at_exactly_one_point_zero(self, calc):
        """Spread == 1.0% should be NORMAL (>= operator)."""
        assert calc.classify_curve(1.0) == YieldCurveRegime.NORMAL

    def test_classify_curve_at_exactly_zero_point_three(self, calc):
        """Spread == 0.3% should be NORMAL (>= operator)."""
        assert calc.classify_curve(0.3) == YieldCurveRegime.NORMAL

    def test_classify_curve_at_epsilon_above_zero(self, calc):
        """Spread == 0.001% should be FLAT (> 0.0)."""
        assert calc.classify_curve(0.001) == YieldCurveRegime.FLAT

    def test_classify_real_rate_at_boundaries(self, calc):
        """Exactly 0.0% and 2.0% should both be NEUTRAL (>=)."""
        assert calc.classify_real_rate(0.0) == "neutral"
        assert calc.classify_real_rate(2.0) == "neutral"

    def test_classify_real_rate_epsilon_above_attractive(self, calc):
        """2.001% should be attractive (> 2.0)."""
        assert calc.classify_real_rate(2.001) == "attractive"

    def test_classify_real_rate_epsilon_below_unattractive(self, calc):
        """-0.001% should be unattractive (< 0.0)."""
        assert calc.classify_real_rate(-0.001) == "unattractive"


class TestDurationAllocationEdgeCases:
    """Area 2 + 6: Additional allocation boundary combos."""

    @pytest.fixture
    def calc(self):
        return BondDurationCalculator()

    def test_flat_falling_intermediate(self, calc):
        """FLAT + FALLING should produce INTERMEDIATE (IEF-heavy)."""
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            0.15, 1.0, RateDirection.FALLING, YieldCurveRegime.FLAT
        )
        assert pos == "intermediate"
        assert ief > tlt

    def test_normal_rising_intermediate(self, calc):
        """NORMAL + RISING should produce INTERMEDIATE (SHY-heavy)."""
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            0.5, 1.0, RateDirection.RISING, YieldCurveRegime.NORMAL
        )
        assert pos == "intermediate"
        assert shy > tlt

    def test_inverted_stable_max_shy(self, calc):
        """INVERTED + STABLE: 0% TLT, 70% SHY."""
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            -0.3, 1.0, RateDirection.STABLE, YieldCurveRegime.INVERTED
        )
        assert tlt == 0.0
        assert shy == 0.70
        assert pos == "short"

    def test_steep_rising_intermediate(self, calc):
        """STEEP + RISING should produce INTERMEDIATE."""
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            1.5, 1.0, RateDirection.RISING, YieldCurveRegime.STEEP
        )
        assert pos == "intermediate"
        assert 0.25 <= tlt <= 0.35

    def test_real_rate_boost_clamped_by_shy(self, calc):
        """When shy < 0.15, boost = shy (partial shift)."""
        # FLAT + STABLE: shy=0.50 initially
        tlt, ief, shy, pos = calc.compute_duration_allocation(
            0.15, 3.0, RateDirection.STABLE, YieldCurveRegime.FLAT
        )
        # Base: 0.10/0.40/0.50, boost = min(0.15, 0.50) = 0.15
        # Result: TLT=0.25, SHY=0.35
        assert abs(tlt - 0.25) < 0.01
        assert abs(shy - 0.35) < 0.01

    def test_real_rate_boost_skipped_when_already_long(self, calc):
        """When pos is already LONG, real-rate boost does NOT apply."""
        # STEEP + FALLING → LONG regardless of real_rate
        tlt_base, _, shy_base, pos = calc.compute_duration_allocation(
            1.5, 1.0, RateDirection.FALLING, YieldCurveRegime.STEEP
        )
        assert pos == "long"
        tlt_boosted, _, shy_boosted, _ = calc.compute_duration_allocation(
            1.5, 3.0, RateDirection.FALLING, YieldCurveRegime.STEEP
        )
        # Both should be identical since pos is already LONG
        assert tlt_base == tlt_boosted
        assert shy_base == shy_boosted

    def test_real_rate_boost_not_applied_at_boundary(self, calc):
        """real_rate == 2.0 (neutral) should NOT trigger boost (not > 2.0)."""
        tlt_normal, _, shy_normal, _ = calc.compute_duration_allocation(
            0.5, 2.0, RateDirection.STABLE, YieldCurveRegime.NORMAL
        )
        tlt_boosted, _, shy_boosted, _ = calc.compute_duration_allocation(
            0.5, 2.001, RateDirection.STABLE, YieldCurveRegime.NORMAL
        )
        # Exactly at 2.0 → no boost; 2.001 → boost
        assert tlt_boosted > tlt_normal
        assert shy_boosted < shy_normal


class TestSnapshotBridge:
    """Area 5: to_signal_snapshot() bridge method."""

    @pytest.fixture
    def generator(self):
        return BondDurationSignalGenerator()

    def test_to_signal_snapshot_returns_snapshot(self, generator):
        """to_signal_snapshot() returns SignalSnapshot with correct base fields."""
        signal = generator.generate_signal(yield_10y=5.0, yield_2y=4.0)
        snap = signal.to_signal_snapshot()
        assert isinstance(snap, SignalSnapshot)
        assert snap.source == "bond_duration_signal"
        assert snap.timestamp == signal.timestamp
        assert snap.confidence == signal.confidence
        assert snap.is_active == signal.is_valid

    def test_to_signal_snapshot_asset_signals(self, generator):
        """Asset signals should map TLT/IEF/SHY weights."""
        signal = generator.generate_signal(yield_10y=5.0, yield_2y=4.0)
        snap = signal.to_signal_snapshot()
        assert "TLT" in snap.asset_signals
        assert "IEF" in snap.asset_signals
        assert "SHY" in snap.asset_signals
        assert snap.asset_signals["TLT"] == signal.tlt_weight
        assert snap.asset_signals["IEF"] == signal.ief_weight
        assert snap.asset_signals["SHY"] == signal.shy_weight

    def test_to_signal_snapshot_position_maps_value(self, generator):
        """Position should map to directional value: short=-0.5, long=0.5."""
        # Long case
        sig_long = generator.generate_signal(
            yield_10y=5.0, yield_2y=3.5, rate_change_6m=-0.8
        )
        assert sig_long.position == "long"
        assert sig_long.to_signal_snapshot().value == 0.5

        # Short case
        sig_short = generator.generate_signal(
            yield_10y=4.0, yield_2y=4.5, rate_change_6m=0.5
        )
        assert sig_short.position == "short"
        assert sig_short.to_signal_snapshot().value == -0.5

        # Blend case: force by creating signal directly
        import copy
        sig_blend = copy.deepcopy(sig_long)
        sig_blend.position = "blend"
        assert sig_blend.to_signal_snapshot().value == 0.0

    def test_to_signal_snapshot_explanation_format(self, generator):
        """Explanation string should contain key diagnostic fields."""
        signal = generator.generate_signal(yield_10y=5.0, yield_2y=4.0)
        snap = signal.to_signal_snapshot()
        assert signal.curve_regime in snap.explanation
        assert signal.real_rate_regime in snap.explanation
        assert signal.position in snap.explanation
        assert "Bond Duration:" in snap.explanation

    def test_to_signal_snapshot_metadata_keys(self, generator):
        """Metadata should contain all 5 diagnostic keys."""
        signal = generator.generate_signal(yield_10y=5.0, yield_2y=4.0)
        snap = signal.to_signal_snapshot()
        meta_keys = {"curve_regime", "real_rate_regime", "position",
                      "effective_duration", "spread_10y2y"}
        assert set(snap.metadata.keys()) == meta_keys


class TestConfidenceBoundaries:
    """Area 6: Confidence boundary conditions."""

    @pytest.fixture
    def generator(self):
        return BondDurationSignalGenerator()

    def test_confidence_inverted_rising_90(self, generator):
        """INVERTED + RISING → confidence 90.0 (max bearish conviction)."""
        signal = generator.generate_signal(
            yield_10y=3.5, yield_2y=4.0, rate_change_6m=0.5
        )
        assert signal.curve_regime == "inverted"
        assert signal.rate_direction == "rising"
        assert signal.confidence == 90.0

    def test_confidence_steep_falling_90(self, generator):
        """STEEP + FALLING → confidence 90.0 (max bullish conviction)."""
        signal = generator.generate_signal(
            yield_10y=5.0, yield_2y=3.5, rate_change_6m=-0.8
        )
        assert signal.curve_regime == "steep"
        assert signal.rate_direction == "falling"
        assert signal.confidence == 90.0

    def test_confidence_near_flat_spread_55(self, generator):
        """Spread < 15bps → confidence 55.0 (uncertain)."""
        signal = generator.generate_signal(
            yield_10y=4.05, yield_2y=4.00, rate_change_6m=0.1
        )
        # spread = 0.05 = 5bps < 15bps
        assert signal.spread_10y2y == 0.05
        assert signal.confidence == 55.0

    def test_confidence_normal_regime_70(self, generator):
        """Normal conditions → confidence 70.0 (baseline)."""
        # NORMAL + STABLE, spread = 0.50, not near-flat (<0.15)
        signal = generator.generate_signal(
            yield_10y=5.0, yield_2y=4.5, rate_change_6m=0.0
        )
        assert signal.curve_regime == "normal"
        assert signal.spread_10y2y > 0.15
        assert signal.confidence == 70.0

    def test_confidence_above_flat_threshold_not_near_flattening(self, generator):
        """Spread = 50bps (NORMAL, >=0.30), safely >= 15bps → conf 70, not 55."""
        signal = generator.generate_signal(
            yield_10y=5.0, yield_2y=4.5, rate_change_6m=0.0
        )
        assert signal.curve_regime == "normal"
        assert signal.spread_10y2y == 0.50
        assert signal.confidence == 70.0


class TestSaveSignal:
    """Area 7: State persistence edge cases."""

    def test_save_signal_creates_file(self, tmp_path):
        """save_signal() should create a JSON file at OUTPUT_PATH."""
        generator = BondDurationSignalGenerator()
        output = tmp_path / "bond_duration_test.json"
        generator.OUTPUT_PATH = output
        assert not output.exists()
        signal = generator.generate_signal(yield_10y=4.5, yield_2y=4.0)
        generator.save_signal(signal)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_save_signal_content_matches(self, tmp_path):
        """Saved JSON content should match signal.to_dict()."""
        generator = BondDurationSignalGenerator()
        output = tmp_path / "bond_duration_test.json"
        generator.OUTPUT_PATH = output
        signal = generator.generate_signal(yield_10y=5.0, yield_2y=4.5)
        generator.save_signal(signal)
        with open(output) as f:
            saved = json.load(f)
        expected = signal.to_dict()
        # Compare all fields except timestamp (serialized vs in-memory may differ
        # by microseconds)
        for key in expected:
            if key != "timestamp":
                assert saved[key] == expected[key], f"Mismatch for key '{key}'"

    def test_save_signal_timestamp_present(self, tmp_path):
        """Saved JSON should include an ISO-format timestamp."""
        generator = BondDurationSignalGenerator()
        output = tmp_path / "bond_duration_test.json"
        generator.OUTPUT_PATH = output
        signal = generator.generate_signal(yield_10y=4.5, yield_2y=4.0)
        generator.save_signal(signal)
        with open(output) as f:
            saved = json.load(f)
        assert "timestamp" in saved
        assert "T" in saved["timestamp"]  # ISO-8601 contains T separator


class TestGenerateSignalEdgeCases:
    """Area 2 + 6: Edge cases in signal generation."""

    @pytest.fixture
    def generator(self):
        return BondDurationSignalGenerator()

    def test_generate_with_zero_yields(self, generator):
        """Zero yields produce inverted curve (spread=0), confidence=55 (flat)."""
        signal = generator.generate_signal(
            yield_10y=0.0, yield_2y=0.0, real_rate=-2.0, rate_change_6m=0.0
        )
        assert signal.spread_10y2y == 0.0
        assert signal.curve_regime == "inverted"
        # spread = 0.0 < 0.15 → near flat → confidence 55
        assert signal.confidence == 55.0

    def test_generate_with_negative_real_rate(self, generator):
        """Negative real rate → unattractive regime."""
        signal = generator.generate_signal(
            yield_10y=4.5, yield_2y=4.0, real_rate=-1.5, rate_change_6m=0.2
        )
        assert signal.real_rate_regime == "unattractive"
        assert signal.is_valid

    def test_generate_partial_params_none_2y(self, generator):
        """When yield_2y is None, missing leg falls back; defaults demote is_valid."""
        signal = generator.generate_signal(
            yield_10y=5.0, yield_2y=None, real_rate=2.0, rate_change_6m=0.1
        )
        assert isinstance(signal, BondDurationSignal)
        assert signal.yield_10y == 5.0
        # Honesty: any defaulted yield leg marks using_defaults / degraded
        if signal.using_defaults:
            assert signal.is_valid is False
            assert signal.source_status == "degraded"
        else:
            assert signal.is_valid is True

    def test_generate_all_defaults(self, generator):
        """When SSOT and market DB unavailable, textbook defaults are disclosed as degraded."""
        from unittest.mock import patch

        with patch.object(generator, "_fetch_yield_data", return_value={
            "yield_10y": 4.50,
            "yield_2y": 4.00,
            "using_defaults": True,
            "source_mode": "defaults",
            "source_status": "degraded",
        }):
            signal = generator.generate_signal()
        assert isinstance(signal, BondDurationSignal)
        assert signal.yield_10y == 4.50
        assert signal.yield_2y == 4.00
        assert signal.using_defaults is True
        assert signal.source_mode == "defaults"
        assert signal.source_status == "degraded"
        assert signal.is_valid is False
        assert signal.confidence <= 40.0

    def test_yields_ssot_preferred_over_defaults(self, generator):
        """yields.json SSOT levels must drive bond_momentum when available."""
        from unittest.mock import patch

        with patch.object(generator, "_fetch_yield_data", return_value={
            "yield_10y": 4.1839,
            "yield_2y": 3.8049,
            "using_defaults": False,
            "source_mode": "yields_ssot",
            "source_status": "ok",
        }):
            signal = generator.generate_signal()
        assert signal.using_defaults is False
        assert signal.source_mode == "yields_ssot"
        assert signal.is_valid is True
        assert abs(signal.yield_10y - 4.18) < 0.02
        assert abs(signal.yield_2y - 3.80) < 0.02

    def test_generate_with_only_real_rate(self, generator):
        """real_rate overrides auto-estimate when real_rate is 0.0 (falsy!)."""
        # This tests the None check: real_rate=0.0 is falsy but not None
        signal = generator.generate_signal(
            yield_10y=5.0, yield_2y=4.0, real_rate=0.0, rate_change_6m=0.0
        )
        assert signal.real_rate == 0.0
        assert signal.real_rate_regime == "neutral"

    def test_generate_flat_rising_confidence_70_not_55(self, generator):
        """FLAT + RISING, spread >= 0.15 (above near-flat threshold) → conf 70."""
        signal = generator.generate_signal(
            yield_10y=4.2, yield_2y=4.0, real_rate=1.0, rate_change_6m=0.5
        )
        # spread ≈ 0.2, safely >= 0.15 threshold
        assert signal.spread_10y2y >= 0.19
        assert signal.curve_regime == "flat"
        assert signal.rate_direction == "rising"
        assert signal.confidence == 70.0  # NOT 55
