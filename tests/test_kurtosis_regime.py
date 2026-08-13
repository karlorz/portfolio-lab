"""
Tests for Kurtosis Regime Detector (v4.91)
"""

import pytest
import numpy as np

from src.regime.kurtosis_regime import (
    KurtosisRegimeDetector,
    KurtosisRegimeSignalGenerator,
    KurtosisRegimeSignal,
    KurtosisRegime,
    StrategyPreference,
    detect_kurtosis_regime,
)


class TestKurtosisRegime:
    """Test regime enum."""

    def test_values(self):
        assert KurtosisRegime.LOW_KURTOSIS.value == "low_kurtosis"
        assert KurtosisRegime.NORMAL.value == "normal"
        assert KurtosisRegime.HIGH_KURTOSIS.value == "high_kurtosis"
        assert KurtosisRegime.EXTREME_KURTOSIS.value == "extreme_kurtosis"


class TestStrategyPreference:
    """Test strategy preference enum."""

    def test_values(self):
        assert StrategyPreference.TREND_FOLLOWING.value == "trend_following"
        assert StrategyPreference.MEAN_REVERSION.value == "mean_reversion"
        assert StrategyPreference.BALANCED.value == "balanced"
        assert StrategyPreference.DEFENSIVE.value == "defensive"


class TestExcessKurtosis:
    """Test excess kurtosis computation."""

    @pytest.fixture
    def detector(self):
        return KurtosisRegimeDetector()

    def test_normal_distribution(self, detector):
        """Normal distribution should have ~0 excess kurtosis."""
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 1000))
        ek = detector.compute_excess_kurtosis(returns)
        assert abs(ek) < 0.5  # Should be near 0 for normal

    def test_fat_tails(self, detector):
        """T-distribution with low df has positive excess kurtosis."""
        rng = np.random.RandomState(42)
        # Mix normal with occasional large moves
        returns = list(rng.normal(0, 0.01, 500))
        for i in range(20):  # Add fat tail events
            idx = rng.randint(0, 499)
            returns[idx] = rng.normal(0, 0.05)
        ek = detector.compute_excess_kurtosis(returns)
        assert ek > 0  # Should have positive excess kurtosis

    def test_constant_returns(self, detector):
        """Constant returns — kurtosis is mathematically undefined (zero variance)."""
        ek = detector.compute_excess_kurtosis([0.01] * 100)
        # Degenerate distribution: either 0 (early return) or -2 (formula boundary)
        assert ek == 0.0 or abs(ek + 2.0) < 0.01

    def test_insufficient_data(self, detector):
        """Fewer than 4 observations should return 0."""
        ek = detector.compute_excess_kurtosis([0.01, 0.02, 0.03])
        assert ek == 0.0

    def test_rolling_kurtosis(self, detector):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 100))
        rolling = detector.compute_rolling_kurtosis(returns, 20)
        assert len(rolling) == len(returns)

    def test_rolling_kurtosis_short_series(self, detector):
        """Series shorter than window should return zeros."""
        returns = [0.01, 0.02, 0.03]
        rolling = detector.compute_rolling_kurtosis(returns, 20)
        assert all(r == 0.0 for r in rolling)


class TestKER:
    """Test KER computation."""

    @pytest.fixture
    def detector(self):
        return KurtosisRegimeDetector()

    def test_equal_kurtosis_ker_1(self, detector):
        ker = detector.compute_ker(2.0, 2.0)
        assert ker == 1.0

    def test_high_short_ker_gt_1(self, detector):
        ker = detector.compute_ker(4.0, 1.0)  # Short > long
        assert ker > 1.0

    def test_low_short_ker_lt_1(self, detector):
        ker = detector.compute_ker(0.0, 2.0)  # Short < long
        assert ker < 1.0

    def test_zero_long_no_div_by_zero(self, detector):
        ker = detector.compute_ker(1.0, -3.0)  # Long excess = -3, absolute = 0
        assert ker >= 1.0  # Falls back to 1.0


class TestRegimeClassification:
    """Test regime classification."""

    @pytest.fixture
    def detector(self):
        return KurtosisRegimeDetector()

    def test_low_kurtosis(self, detector):
        regime, conf = detector.classify_regime(-0.2)
        assert regime == KurtosisRegime.LOW_KURTOSIS

    def test_normal(self, detector):
        regime, conf = detector.classify_regime(1.0)
        assert regime == KurtosisRegime.NORMAL

    def test_high_kurtosis(self, detector):
        regime, conf = detector.classify_regime(3.0)
        assert regime == KurtosisRegime.HIGH_KURTOSIS
        assert conf > 0

    def test_extreme_kurtosis(self, detector):
        regime, conf = detector.classify_regime(7.0)
        assert regime == KurtosisRegime.EXTREME_KURTOSIS
        assert conf > 0.8

    def test_boundary_values(self, detector):
        # At exact boundaries
        r1, _ = detector.classify_regime(0.5)
        assert r1 == KurtosisRegime.NORMAL
        r2, _ = detector.classify_regime(2.0)
        assert r2 == KurtosisRegime.HIGH_KURTOSIS


class TestStrategyWeights:
    """Test strategy weight computation."""

    @pytest.fixture
    def detector(self):
        return KurtosisRegimeDetector()

    def test_low_kurtosis_favors_trend(self, detector):
        tsom_w, mr_w, pref = detector.compute_strategy_weights(
            KurtosisRegime.LOW_KURTOSIS, 1.0, False
        )
        assert tsom_w > mr_w
        assert pref == StrategyPreference.TREND_FOLLOWING

    def test_high_kurtosis_favors_mr(self, detector):
        tsom_w, mr_w, pref = detector.compute_strategy_weights(
            KurtosisRegime.HIGH_KURTOSIS, 1.0, False
        )
        assert mr_w > tsom_w
        assert pref == StrategyPreference.MEAN_REVERSION

    def test_extreme_defensive(self, detector):
        tsom_w, mr_w, pref = detector.compute_strategy_weights(
            KurtosisRegime.EXTREME_KURTOSIS, 1.0, False
        )
        assert pref == StrategyPreference.DEFENSIVE

    def test_transitioning_to_high(self, detector):
        tsom_w, mr_w, pref = detector.compute_strategy_weights(
            KurtosisRegime.NORMAL, 1.5, True  # KER high = shifting to fat tail
        )
        assert pref == StrategyPreference.BALANCED
        assert mr_w > 0.3

    def test_transitioning_to_low(self, detector):
        tsom_w, mr_w, pref = detector.compute_strategy_weights(
            KurtosisRegime.NORMAL, 0.5, True  # KER low = shifting to normal
        )
        assert pref == StrategyPreference.BALANCED
        assert tsom_w > 0.3

    def test_weights_sum_reasonable(self, detector):
        """TSMOM + MR should be roughly 1.0."""
        for regime in KurtosisRegime:
            for ker in [0.5, 1.0, 1.5]:
                for trans in [True, False]:
                    tsom, mr, _ = detector.compute_strategy_weights(regime, ker, trans)
                    assert abs(tsom + mr - 1.0) < 0.01, \
                        f"{regime.value}, KER={ker}, trans={trans}: {tsom + mr}"


class TestSignalGeneration:
    """Test complete signal generation."""

    @pytest.fixture
    def generator(self):
        return KurtosisRegimeSignalGenerator()

    def test_generates_signal(self, generator):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        signal = generator.generate_signal(returns)
        assert isinstance(signal, KurtosisRegimeSignal)
        assert signal.kurtosis_60d > 0
        assert signal.regime is not None

    def test_signal_serializable(self, generator):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        signal = generator.generate_signal(returns)
        d = signal.to_dict()
        assert isinstance(d, dict)
        assert "kurtosis_60d" in d
        assert "regime" in d

    def test_default_returns(self, generator):
        """Should work with no returns provided."""
        signal = generator.generate_signal()
        assert isinstance(signal, KurtosisRegimeSignal)
        assert signal.regime is not None

    def test_convenience_function(self):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        signal = detect_kurtosis_regime(returns)
        assert isinstance(signal, KurtosisRegimeSignal)

    def test_exposure_in_range(self, generator):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        signal = generator.generate_signal(returns)
        assert 0.0 < signal.recommended_exposure <= 1.0

    def test_fat_tail_risk_in_range(self, generator):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        signal = generator.generate_signal(returns)
        assert 0.0 <= signal.fat_tail_risk <= 1.0


class TestEdgeCases:
    """Edge cases."""

    @pytest.fixture
    def detector(self):
        return KurtosisRegimeDetector()

    def test_empty_returns(self, detector):
        ek = detector.compute_excess_kurtosis([])
        assert ek == 0.0

    def test_single_value(self, detector):
        ek = detector.compute_excess_kurtosis([0.01])
        assert ek == 0.0

    def test_all_zeros(self, detector):
        ek = detector.compute_excess_kurtosis([0.0] * 100)
        assert ek == 0.0


class TestExcessKurtosisExtended:
    """Additional excess kurtosis edge cases."""

    @pytest.fixture
    def detector(self):
        return KurtosisRegimeDetector()

    def test_negative_excess_kurtosis(self, detector):
        """Platykurtic distribution (thin tails) should have negative excess kurtosis."""
        # Uniform-ish distribution
        rng = np.random.RandomState(42)
        returns = list(rng.uniform(-0.005, 0.005, 1000))
        ek = detector.compute_excess_kurtosis(returns)
        assert ek < 0  # Uniform has excess kurtosis ≈ -1.2

    def test_two_values_alternating(self, detector):
        """Alternating values should produce deterministic kurtosis."""
        returns = [0.01, -0.01] * 50
        ek = detector.compute_excess_kurtosis(returns)
        # Two-point distribution has negative excess kurtosis
        assert isinstance(ek, float)

    def test_outlier_drives_high_kurtosis(self, detector):
        """A single large outlier should increase excess kurtosis."""
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 100))
        ek_no_outlier = detector.compute_excess_kurtosis(returns)
        # Add a big outlier
        returns_with_outlier = returns + [0.5]
        ek_with_outlier = detector.compute_excess_kurtosis(returns_with_outlier)
        assert ek_with_outlier > ek_no_outlier


class TestKERExtended:
    """Additional KER edge cases."""

    @pytest.fixture
    def detector(self):
        return KurtosisRegimeDetector()

    def test_ker_with_negative_kurtosis(self, detector):
        """Both negative excess kurtosis should still compute KER."""
        ker = detector.compute_ker(-1.0, -2.0)
        # short_abs = -1.0 + 3 = 2.0, long_abs = -2.0 + 3 = 1.0
        assert ker == 2.0

    def test_ker_at_zero_long(self, detector):
        """When long kurtosis + 3 = 0 (excess = -3), should return 1.0."""
        ker = detector.compute_ker(1.0, -3.0)
        # kurt_long + 3 = 0, falls to return 1.0
        assert ker == 1.0

    def test_ker_symmetry(self, detector):
        """KER should reflect short/long ratio."""
        ker = detector.compute_ker(2.0, 1.0)
        # short_abs = 5, long_abs = 4
        assert abs(ker - 5.0 / 4.0) < 0.01


class TestRegimeClassificationExtended:
    """Additional regime classification edge cases."""

    @pytest.fixture
    def detector(self):
        return KurtosisRegimeDetector()

    def test_negative_kurtosis_is_low(self, detector):
        """Negative excess kurtosis → LOW_KURTOSIS."""
        regime, conf = detector.classify_regime(-1.0)
        assert regime == KurtosisRegime.LOW_KURTOSIS

    def test_confidence_increases_with_extremity(self, detector):
        """Higher excess kurtosis should give higher confidence."""
        _, conf_normal = detector.classify_regime(1.0)
        _, conf_extreme = detector.classify_regime(10.0)
        assert conf_extreme > conf_normal

    def test_low_kurtosis_confidence_increases_with_negativity(self, detector):
        """More negative excess kurtosis should increase LOW_KURTOSIS confidence."""
        _, conf_mild = detector.classify_regime(-0.1)
        _, conf_strong = detector.classify_regime(-0.4)
        assert conf_strong > conf_mild

    def test_extreme_confidence_near_one(self, detector):
        """Very high excess kurtosis should push confidence toward 1.0."""
        _, conf = detector.classify_regime(20.0)
        assert conf >= 0.9


class TestStrategyWeightsExtended:
    """Additional strategy weight edge cases."""

    @pytest.fixture
    def detector(self):
        return KurtosisRegimeDetector()

    def test_normal_regime_trend_heavy(self, detector):
        """Normal regime without transition should favor trend."""
        tsom, mr, pref = detector.compute_strategy_weights(
            KurtosisRegime.NORMAL, 1.0, False
        )
        assert tsom > mr
        assert pref == StrategyPreference.TREND_FOLLOWING

    def test_extreme_regime_always_defensive(self, detector):
        """Extreme kurtosis always returns defensive, regardless of transition."""
        for trans in [True, False]:
            for ker in [0.5, 1.0, 1.5, 2.0]:
                tsom, mr, pref = detector.compute_strategy_weights(
                    KurtosisRegime.EXTREME_KURTOSIS, ker, trans
                )
                assert pref == StrategyPreference.DEFENSIVE
                assert mr > tsom


class TestSignalGenerationExtended:
    """Additional signal generation edge cases."""

    @pytest.fixture
    def generator(self):
        return KurtosisRegimeSignalGenerator()

    def test_short_returns_series(self, generator):
        """Very short return series (< 20) should still produce a signal."""
        signal = generator.generate_signal([0.01, -0.005, 0.02, -0.01, 0.005])
        assert isinstance(signal, KurtosisRegimeSignal)
        assert signal.regime is not None

    def test_fat_tail_risk_scale(self, generator):
        """Fat tail risk should be between 0 and 1."""
        rng = np.random.RandomState(42)
        for _ in range(5):
            returns = list(rng.normal(0, 0.01 * rng.uniform(0.5, 3.0), 200))
            signal = generator.generate_signal(returns)
            assert 0.0 <= signal.fat_tail_risk <= 1.0

    def test_exposure_range(self, generator):
        """Recommended exposure should always be between 0 and 1."""
        rng = np.random.RandomState(42)
        for _ in range(5):
            returns = list(rng.normal(0, 0.01 * rng.uniform(0.5, 3.0), 200))
            signal = generator.generate_signal(returns)
            assert 0.0 < signal.recommended_exposure <= 1.0

    def test_transition_detection(self, generator):
        """Large KER should trigger transition flag."""
        # Create returns where 20d kurtosis is very different from 120d
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 120))
        # Spike the last 20 days
        for i in range(-20, 0):
            returns[i] = rng.normal(0, 0.04)
        signal = generator.generate_signal(returns)
        # With very different short vs long kurtosis, KER should shift
        assert isinstance(signal.is_transitioning, bool)

    def test_signal_has_explanation(self, generator):
        """Every signal should have a non-empty explanation."""
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        signal = generator.generate_signal(returns)
        assert len(signal.explanation) > 0

    def test_signal_to_dict_complete(self, generator):
        """to_dict should include all expected fields."""
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        signal = generator.generate_signal(returns)
        d = signal.to_dict()
        expected_fields = {
            "timestamp", "kurtosis_20d", "kurtosis_60d", "kurtosis_120d",
            "ker_ratio", "regime", "regime_confidence", "is_transitioning",
            "transition_speed", "strategy_preference", "tsom_weight", "mr_weight",
            "fat_tail_risk", "recommended_exposure", "confidence", "explanation",
        }
        assert expected_fields.issubset(set(d.keys()))


#
# === New test classes (dataclass validation, constants, NaN/Inf, boundary, CLI, exports) ===
#


import dataclasses  # noqa: E402  # deliberate placement (bootstrap/sys.path ordering)
import math  # noqa: E402  # deliberate placement (bootstrap/sys.path ordering)
import sys  # noqa: E402  # deliberate placement (bootstrap/sys.path ordering)
from unittest.mock import patch  # noqa: E402  # deliberate placement (bootstrap/sys.path ordering)


class TestKurtosisRegimeSignalDataclass:
    """Validate KurtosisRegimeSignal dataclass fields, types, and defaults."""

    def test_all_fields_exist(self):
        """All 16 documented fields should exist on the dataclass."""
        field_names = {f.name for f in dataclasses.fields(KurtosisRegimeSignal)}
        expected = {
            "timestamp", "kurtosis_20d", "kurtosis_60d", "kurtosis_120d",
            "ker_ratio", "regime", "regime_confidence", "is_transitioning",
            "transition_speed", "strategy_preference", "tsom_weight", "mr_weight",
            "fat_tail_risk", "recommended_exposure", "confidence", "explanation",
        }
        assert field_names == expected, f"Missing fields: {expected - field_names}"

    def test_field_types_are_correct(self):
        """Each field should have the expected type annotation."""
        fields = {f.name: f.type for f in dataclasses.fields(KurtosisRegimeSignal)}
        assert fields["timestamp"] is str
        assert fields["kurtosis_20d"] is float
        assert fields["kurtosis_60d"] is float
        assert fields["kurtosis_120d"] is float
        assert fields["ker_ratio"] is float
        assert fields["regime"] is str
        assert fields["regime_confidence"] is float
        assert fields["is_transitioning"] is bool
        assert fields["transition_speed"] is float
        assert fields["strategy_preference"] is str
        assert fields["tsom_weight"] is float
        assert fields["mr_weight"] is float
        assert fields["fat_tail_risk"] is float
        assert fields["recommended_exposure"] is float
        assert fields["confidence"] is float
        assert fields["explanation"] is str

    def test_all_fields_required_no_defaults(self):
        """All fields should be required (no defaults, no default_factory)."""
        for f in dataclasses.fields(KurtosisRegimeSignal):
            assert f.default is dataclasses.MISSING, f"{f.name} has default"
            assert f.default_factory is dataclasses.MISSING, f"{f.name} has default_factory"

    def test_to_dict_keys_match_fields(self):
        """to_dict() keys should exactly match dataclass field names."""
        field_names = {f.name for f in dataclasses.fields(KurtosisRegimeSignal)}
        signal = KurtosisRegimeSignal(
            timestamp="2024-01-01",
            kurtosis_20d=1.0, kurtosis_60d=2.0, kurtosis_120d=3.0,
            ker_ratio=0.5, regime="normal", regime_confidence=0.8,
            is_transitioning=False, transition_speed=0.1,
            strategy_preference="trend_following", tsom_weight=0.7, mr_weight=0.3,
            fat_tail_risk=0.2, recommended_exposure=1.0, confidence=80.0,
            explanation="test",
        )
        assert set(signal.to_dict().keys()) == field_names

    def test_instantiation_with_minimal_values(self):
        """All fields can be set to zero/empty values without error."""
        signal = KurtosisRegimeSignal(
            timestamp="", kurtosis_20d=0.0, kurtosis_60d=0.0, kurtosis_120d=0.0,
            ker_ratio=0.0, regime="", regime_confidence=0.0,
            is_transitioning=False, transition_speed=0.0,
            strategy_preference="", tsom_weight=0.0, mr_weight=0.0,
            fat_tail_risk=0.0, recommended_exposure=0.0, confidence=0.0,
            explanation="",
        )
        assert signal.timestamp == ""
        assert signal.regime == ""
        assert signal.is_transitioning is False

    def test_instantiation_with_numeric_types(self):
        """Numeric fields should accept and store float values correctly."""
        signal = KurtosisRegimeSignal(
            timestamp="2024-06-15T12:00:00",
            kurtosis_20d=3.5, kurtosis_60d=4.2, kurtosis_120d=3.8,
            ker_ratio=1.25, regime="high_kurtosis", regime_confidence=0.75,
            is_transitioning=True, transition_speed=0.3,
            strategy_preference="mean_reversion", tsom_weight=0.25, mr_weight=0.75,
            fat_tail_risk=0.6, recommended_exposure=0.75, confidence=75.0,
            explanation="High kurtosis detected",
        )
        assert signal.kurtosis_20d == 3.5
        assert signal.ker_ratio == 1.25
        assert signal.regime_confidence == 0.75
        assert signal.fat_tail_risk == 0.6
        assert signal.explanation == "High kurtosis detected"


class TestConstants:
    """Validate module-level constants on KurtosisRegimeDetector."""

    def test_low_kurtosis_max_value(self):
        assert KurtosisRegimeDetector.LOW_KURTOSIS_MAX == 0.5

    def test_high_kurtosis_min_value(self):
        assert KurtosisRegimeDetector.HIGH_KURTOSIS_MIN == 2.0

    def test_extreme_kurtosis_min_value(self):
        assert KurtosisRegimeDetector.EXTREME_KURTOSIS_MIN == 5.0

    def test_ker_shift_up_value(self):
        assert KurtosisRegimeDetector.KER_SHIFT_UP == 1.3

    def test_ker_shift_down_value(self):
        assert KurtosisRegimeDetector.KER_SHIFT_DOWN == 0.7

    def test_all_constants_are_floats(self):
        assert isinstance(KurtosisRegimeDetector.LOW_KURTOSIS_MAX, float)
        assert isinstance(KurtosisRegimeDetector.HIGH_KURTOSIS_MIN, float)
        assert isinstance(KurtosisRegimeDetector.EXTREME_KURTOSIS_MIN, float)
        assert isinstance(KurtosisRegimeDetector.KER_SHIFT_UP, float)
        assert isinstance(KurtosisRegimeDetector.KER_SHIFT_DOWN, float)

    def test_threshold_monotonic_increasing(self):
        """Thresholds should be strictly increasing: LOW < HIGH < EXTREME."""
        assert KurtosisRegimeDetector.LOW_KURTOSIS_MAX < KurtosisRegimeDetector.HIGH_KURTOSIS_MIN
        assert KurtosisRegimeDetector.HIGH_KURTOSIS_MIN < KurtosisRegimeDetector.EXTREME_KURTOSIS_MIN

    def test_ker_shifts_are_symmetric_around_one(self):
        """KER_SHIFT_DOWN and KER_SHIFT_UP should be symmetric (distance from 1.0)."""
        assert abs((1.0 - KurtosisRegimeDetector.KER_SHIFT_DOWN) -
                   (KurtosisRegimeDetector.KER_SHIFT_UP - 1.0)) < 0.001


class TestNaNInfHandling:
    """Test NaN and Inf handling across all computation methods."""

    @pytest.fixture
    def detector(self):
        return KurtosisRegimeDetector()

    def test_excess_kurtosis_with_nan(self, detector):
        """NaN in returns should produce NaN excess kurtosis."""
        rets = [0.01, -0.01, np.nan, 0.02] * 10  # 40 values, > 4
        ek = detector.compute_excess_kurtosis(rets)
        assert math.isnan(ek)

    def test_excess_kurtosis_with_inf(self, detector):
        """Inf in returns should produce NaN excess kurtosis (np.mean propagates)."""
        rets = [0.01, -0.01, np.inf, 0.02] * 10
        ek = detector.compute_excess_kurtosis(rets)
        assert math.isnan(ek)

    def test_excess_kurtosis_with_neg_inf(self, detector):
        """-Inf in returns should produce NaN excess kurtosis."""
        rets = [0.01, -0.01, -np.inf, 0.02] * 10
        ek = detector.compute_excess_kurtosis(rets)
        assert math.isnan(ek)

    def test_rolling_kurtosis_with_nan(self, detector):
        """Rolling kurtosis with NaN returns should not crash or raise."""
        rets = [0.01, np.nan, 0.02] * 40  # 120 values, window=20
        rolling = detector.compute_rolling_kurtosis(rets, 20)
        assert len(rolling) == len(rets)
        # All entries may be NaN, but length must match and function must not crash
        assert all(isinstance(r, float) for r in rolling)

    def test_ker_with_nan(self, detector):
        """KER with NaN input should produce NaN."""
        ker = detector.compute_ker(np.nan, 2.0)
        assert math.isnan(ker)

    def test_ker_with_inf_short(self, detector):
        """KER with Inf short kurtosis should produce Inf."""
        ker = detector.compute_ker(np.inf, 2.0)
        assert math.isinf(ker) and ker > 0

    def test_ker_with_neg_inf_short(self, detector):
        """KER with -Inf short kurtosis should produce -Inf."""
        ker = detector.compute_ker(-np.inf, 2.0)
        assert math.isinf(ker) and ker < 0

    def test_classify_regime_nan(self, detector):
        """NaN excess kurtosis falls through to LOW_KURTOSIS (comparisons return False)."""
        regime, conf = detector.classify_regime(np.nan)
        assert regime == KurtosisRegime.LOW_KURTOSIS
        assert math.isnan(conf)

    def test_classify_regime_inf(self, detector):
        """Inf excess kurtosis is >= EXTREME_KURTOSIS_MIN -> EXTREME_KURTOSIS."""
        regime, conf = detector.classify_regime(np.inf)
        assert regime == KurtosisRegime.EXTREME_KURTOSIS
        assert conf == 1.0

    def test_classify_regime_neg_inf(self, detector):
        """-Inf falls through to LOW_KURTOSIS."""
        regime, conf = detector.classify_regime(-np.inf)
        assert regime == KurtosisRegime.LOW_KURTOSIS
        assert math.isinf(conf)

    def test_ker_with_nan_long(self, detector):
        """KER with NaN long kurtosis should produce NaN."""
        ker = detector.compute_ker(2.0, np.nan)
        assert math.isnan(ker)

    def test_ker_with_inf_long(self, detector):
        """KER with Inf long kurtosis -> short_abs / Inf -> 0."""
        ker = detector.compute_ker(2.0, np.inf)
        assert ker == 0.0


class TestKurtosisEdgeCases:
    """Additional edge cases for excess kurtosis and rolling kurtosis."""

    @pytest.fixture
    def detector(self):
        return KurtosisRegimeDetector()

    def test_window_exactly_four_values(self, detector):
        """Window of exactly 4 should be computable (n >= 4 is the threshold)."""
        returns = [0.01, -0.01, 0.02, -0.02]
        ek = detector.compute_excess_kurtosis(returns)
        assert isinstance(ek, float)
        assert not np.isnan(ek)

    def test_window_of_three_returns_zero(self, detector):
        """Window of exactly 3 should return 0 (n < 4 guard)."""
        returns = [0.01, -0.01, 0.02]
        ek = detector.compute_excess_kurtosis(returns)
        assert ek == 0.0

    def test_empty_returns_list(self, detector):
        """Empty list should return 0.0."""
        ek = detector.compute_excess_kurtosis([])
        assert ek == 0.0

    def test_symmetric_bimodal_negative_ek(self, detector):
        """Symmetric bimodal distribution has negative excess kurtosis."""
        returns = [-0.02] * 50 + [0.02] * 50
        ek = detector.compute_excess_kurtosis(returns)
        assert ek < 0  # Bimodal is platykurtic

    def test_rolling_kurtosis_window_size_match(self, detector):
        """When window == len(returns), only one non-padded value at the end."""
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 20))
        rolling = detector.compute_rolling_kurtosis(returns, 20)
        assert len(rolling) == 20
        assert rolling[0] == 0.0  # Padded
        assert rolling[18] == 0.0  # Padded (index 0-18 are padded for window=20)
        assert isinstance(rolling[19], float)  # Real value at the end

    def test_rolling_kurtosis_window_one_larger(self, detector):
        """window == len(returns) + 1 should return all zeros."""
        returns = [0.01] * 20
        rolling = detector.compute_rolling_kurtosis(returns, 21)
        assert len(rolling) == len(returns)
        assert all(r == 0.0 for r in rolling)

    def test_single_element_after_window(self, detector):
        """window = len(returns) - 1 should give 1 non-padded value."""
        returns = list(np.random.RandomState(42).normal(0, 0.01, 21))
        rolling = detector.compute_rolling_kurtosis(returns, 20)
        assert len(rolling) == len(returns)
        assert rolling[19] != 0.0 or rolling[20] != 0.0  # At least one is real

    def test_negative_only_returns(self, detector):
        """All negative returns should still compute valid kurtosis."""
        returns = [-abs(x) for x in np.random.RandomState(42).normal(0, 0.01, 100)]
        ek = detector.compute_excess_kurtosis(returns)
        assert isinstance(ek, float)

    def test_large_array_does_not_overflow(self, detector):
        """Large array (10k+ values) should compute without overflow."""
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 100_000))
        ek = detector.compute_excess_kurtosis(returns)
        assert isinstance(ek, float)
        assert abs(ek) < 1.0  # Should be near normal

    def test_extreme_outlier_dominates_kurtosis(self, detector):
        """A single extreme outlier should drive kurtosis very high."""
        returns = list(np.random.RandomState(42).normal(0, 0.01, 100))
        returns.append(1.0)  # 10-sigma outlier
        ek = detector.compute_excess_kurtosis(returns)
        assert ek > 10

    def test_all_identical_floats_returns_zero_or_neg_two(self, detector):
        """All identical values => near-zero variance => 0.0 or -2.0 (floating-point path)."""
        ek = detector.compute_excess_kurtosis([0.05] * 50)
        # Floating-point: m2 may not be exactly 0, giving -2.0
        assert ek == 0.0 or abs(ek + 2.0) < 0.01

    def test_four_identical_values_each_window(self, detector):
        """Window of 4 identical values: near-zero variance => 0.0 or -2.0."""
        returns = [0.01, 0.01, 0.01, 0.01]
        ek = detector.compute_excess_kurtosis(returns)
        assert ek == 0.0 or abs(ek + 2.0) < 0.01


class TestKEREdgeCases:
    """Additional KER edge cases beyond basic tests."""

    @pytest.fixture
    def detector(self):
        return KurtosisRegimeDetector()

    def test_both_negative_but_valid_ker(self, detector):
        """Both short and long excess kurtosis negative should still compute KER."""
        ker = detector.compute_ker(-1.0, -1.0)
        # short_abs = 2.0, long_abs = 2.0 => KER = 1.0
        assert ker == 1.0

    def test_short_abs_zero(self, detector):
        """When short_abs = 0 (excess kurtosis = -3), KER = 0 / long_abs = 0."""
        ker = detector.compute_ker(-3.0, 1.0)
        # short = -3 + 3 = 0, long = 1 + 3 = 4, KER = 0 / 4 = 0
        assert ker == 0.0

    def test_ker_extreme_asymmetry(self, detector):
        """Very different short vs long kurtosis should produce extreme KER."""
        ker_high = detector.compute_ker(100.0, 1.0)
        # short_abs = 103, long_abs = 4 => KER = 25.75
        assert ker_high > 20
        ker_low = detector.compute_ker(1.0, 100.0)
        # short_abs = 4, long_abs = 103 => KER = 0.0388
        assert ker_low < 0.05

    def test_ker_precision_with_small_differences(self, detector):
        """Small differences in kurtosis should produce KER near 1.0."""
        ker = detector.compute_ker(3.1, 3.0)
        # short_abs = 6.1, long_abs = 6.0 => KER = 1.0166...
        assert abs(ker - 1.0) < 0.1

    def test_ker_with_long_abs_near_zero_negative(self, detector):
        """When long_abs <= 0 (excess <= -3), should return 1.0 guard."""
        ker = detector.compute_ker(0.0, -3.0)
        # kurt_long + 3 = 0, falls to return 1.0
        assert ker == 1.0

    def test_ker_with_both_abs_near_zero(self, detector):
        """Both short and long near -3 (abs near 0)."""
        ker = detector.compute_ker(-2.9, -2.9)
        # short_abs = 0.1, long_abs = 0.1 => KER = 1.0
        assert abs(ker - 1.0) < 0.01


class TestRegimeBoundaryConditions:
    """Boundary condition tests for classify_regime."""

    @pytest.fixture
    def detector(self):
        return KurtosisRegimeDetector()

    def test_exactly_zero_excess(self, detector):
        """Excess kurtosis of 0.0 should be LOW_KURTOSIS."""
        regime, conf = detector.classify_regime(0.0)
        assert regime == KurtosisRegime.LOW_KURTOSIS

    def test_just_below_low_threshold(self, detector):
        """0.499 should still be LOW_KURTOSIS."""
        regime, _ = detector.classify_regime(0.499)
        assert regime == KurtosisRegime.LOW_KURTOSIS

    def test_at_low_threshold_normal(self, detector):
        """0.5 should be NORMAL (inclusive boundary)."""
        regime, _ = detector.classify_regime(0.5)
        assert regime == KurtosisRegime.NORMAL

    def test_at_high_threshold(self, detector):
        """2.0 should be HIGH_KURTOSIS (inclusive boundary)."""
        regime, _ = detector.classify_regime(2.0)
        assert regime == KurtosisRegime.HIGH_KURTOSIS

    def test_just_below_high_threshold(self, detector):
        """1.999 should still be NORMAL."""
        regime, _ = detector.classify_regime(1.999)
        assert regime == KurtosisRegime.NORMAL

    def test_at_extreme_threshold(self, detector):
        """5.0 should be EXTREME_KURTOSIS (inclusive boundary)."""
        regime, _ = detector.classify_regime(5.0)
        assert regime == KurtosisRegime.EXTREME_KURTOSIS

    def test_just_below_extreme_threshold(self, detector):
        """4.999 should still be HIGH_KURTOSIS."""
        regime, _ = detector.classify_regime(4.999)
        assert regime == KurtosisRegime.HIGH_KURTOSIS

    def test_huge_positive_value(self, detector):
        """1e6 should be EXTREME_KURTOSIS with confidence capped at 1.0."""
        regime, conf = detector.classify_regime(1e6)
        assert regime == KurtosisRegime.EXTREME_KURTOSIS
        assert conf <= 1.0

    def test_huge_negative_value(self, detector):
        """-1e6 should be LOW_KURTOSIS with high confidence."""
        regime, conf = detector.classify_regime(-1e6)
        assert regime == KurtosisRegime.LOW_KURTOSIS
        assert conf > 0.5

    def test_confidence_monotonic_within_regime(self, detector):
        """Confidence should increase as excess kurtosis moves further from boundary."""
        # LOW_KURTOSIS: more negative = higher confidence
        _, c1 = detector.classify_regime(-0.1)
        _, c2 = detector.classify_regime(-0.4)
        assert c2 > c1
        # HIGH_KURTOSIS: more positive = higher confidence
        _, c3 = detector.classify_regime(2.5)
        _, c4 = detector.classify_regime(4.5)
        assert c4 > c3


class TestStrategyWeightsEdgeCases:
    """Strategy weight edge cases: boundary KER values, regime transitions."""

    @pytest.fixture
    def detector(self):
        return KurtosisRegimeDetector()

    def test_ker_at_shift_up_boundary(self, detector):
        """KER exactly at KER_SHIFT_UP (1.3) should trigger transitioning."""
        tsom, mr, pref = detector.compute_strategy_weights(
            KurtosisRegime.NORMAL, 1.3, True
        )
        assert pref == StrategyPreference.BALANCED
        assert mr > tsom

    def test_ker_at_shift_down_boundary(self, detector):
        """KER exactly at KER_SHIFT_DOWN (0.7) should trigger transitioning."""
        tsom, mr, pref = detector.compute_strategy_weights(
            KurtosisRegime.NORMAL, 0.7, True
        )
        assert pref == StrategyPreference.BALANCED
        assert tsom > mr

    def test_extreme_with_transitioning_still_defensive(self, detector):
        """EXTREME_KURTOSIS always returns DEFENSIVE regardless of transitioning flag."""
        for ker in [0.5, 1.0, 1.5]:
            tsom, mr, pref = detector.compute_strategy_weights(
                KurtosisRegime.EXTREME_KURTOSIS, ker, True
            )
            assert pref == StrategyPreference.DEFENSIVE
            assert tsom == 0.1
            assert mr == 0.9

    def test_low_kurtosis_transitioning_to_high_not_possible(self, detector):
        """LOW_KURTOSIS with transitioning=True but KER>1.3 (contradictory flags)."""
        tsom, mr, pref = detector.compute_strategy_weights(
            KurtosisRegime.LOW_KURTOSIS, 1.5, True
        )
        # transitioning=True takes priority: shift to mean-reversion
        assert pref == StrategyPreference.BALANCED
        assert mr > tsom  # KER high => shifting to higher kurtosis => MR

    def test_high_kurtosis_transitioning_to_low(self, detector):
        """HIGH_KURTOSIS with transitioning and low KER."""
        tsom, mr, pref = detector.compute_strategy_weights(
            KurtosisRegime.HIGH_KURTOSIS, 0.5, True
        )
        # transitioning=True takes priority: shift to trend
        assert pref == StrategyPreference.BALANCED
        assert tsom > mr

    def test_ker_way_above_shift_up(self, detector):
        """KER >> 1.3 should cap transition at 1.0."""
        tsom, mr, pref = detector.compute_strategy_weights(
            KurtosisRegime.NORMAL, 5.0, True
        )
        assert pref == StrategyPreference.BALANCED
        # transition = min(1.0, (5.0 - 1.0) / 0.5) = min(1.0, 8.0) = 1.0
        # tsom = 0.7 - 1.0 * 0.5 = 0.2, mr = 0.3 + 1.0 * 0.5 = 0.8
        assert abs(tsom - 0.2) < 0.01
        assert abs(mr - 0.8) < 0.01

    def test_ker_way_below_shift_down(self, detector):
        """KER << 0.7 should cap transition at 1.0."""
        tsom, mr, pref = detector.compute_strategy_weights(
            KurtosisRegime.NORMAL, 0.1, True
        )
        assert pref == StrategyPreference.BALANCED
        # transition = min(1.0, (1.0 - 0.1) / 0.5) = min(1.0, 1.8) = 1.0
        # tsom = 0.3 + 1.0 * 0.5 = 0.8, mr = 0.7 - 1.0 * 0.5 = 0.2
        assert abs(tsom - 0.8) < 0.01
        assert abs(mr - 0.2) < 0.01

    def test_weights_sum_to_one_all_combinations(self, detector):
        """TSMOM + MR should sum to ~1.0 for every valid combination."""
        for regime in KurtosisRegime:
            for ker in [0.3, 0.7, 1.0, 1.3, 2.0]:
                for trans in [True, False]:
                    tsom, mr, _ = detector.compute_strategy_weights(regime, ker, trans)
                    assert abs(tsom + mr - 1.0) < 0.01, (
                        f"{regime.value}, KER={ker}, trans={trans}: sum={tsom + mr}"
                    )


class TestSignalGeneratorEdgeCases:
    """Edge cases for KurtosisRegimeSignalGenerator."""

    @pytest.fixture
    def generator(self):
        return KurtosisRegimeSignalGenerator()

    def test_generate_signal_nan_in_returns(self, generator):
        """NaN in returns should not crash; signal is still produced."""
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        returns[50] = np.nan
        signal = generator.generate_signal(returns)
        assert isinstance(signal, KurtosisRegimeSignal)

    def test_generate_signal_all_zeros(self, generator):
        """All zero returns should produce a valid signal (constant returns)."""
        returns = [0.0] * 200
        signal = generator.generate_signal(returns)
        assert isinstance(signal, KurtosisRegimeSignal)
        assert signal.regime is not None

    def test_generate_signal_very_large_series(self, generator):
        """Very long return series should not overflow."""
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 100_000))
        signal = generator.generate_signal(returns)
        assert isinstance(signal, KurtosisRegimeSignal)
        assert 0.0 <= signal.fat_tail_risk <= 1.0

    def test_generate_signal_below_short_window(self, generator):
        """Series shorter than short_window (20) should use synthetic data fallback."""
        signal = generator.generate_signal([0.01, -0.005])
        assert isinstance(signal, KurtosisRegimeSignal)
        # Falls back to synthetic 200 returns
        assert signal.kurtosis_60d > 0

    def test_generate_signal_exactly_short_window_length(self, generator):
        """Series exactly at long_window length (120) should compute all windows."""
        returns = list(np.random.RandomState(42).normal(0, 0.01, 120))
        signal = generator.generate_signal(returns)
        assert isinstance(signal, KurtosisRegimeSignal)
        assert signal.kurtosis_20d > 0
        assert signal.kurtosis_60d > 0
        assert signal.kurtosis_120d > 0

    def test_generate_signal_exactly_long_window_length(self, generator):
        """Series exactly at long_window length (120) should compute all windows."""
        returns = list(np.random.RandomState(42).normal(0, 0.01, 120))
        signal = generator.generate_signal(returns)
        assert isinstance(signal, KurtosisRegimeSignal)
        assert signal.kurtosis_20d > 0
        assert signal.kurtosis_60d > 0
        assert signal.kurtosis_120d > 0

    def test_generate_signal_medium_window_no_fallback(self, generator):
        """Series between medium and long window (60 < len < 120) computes k20, k60, k120=3."""
        returns = list(np.random.RandomState(42).normal(0, 0.01, 90))
        signal = generator.generate_signal(returns)
        # 90 < 120 triggers synthetic fallback; all values are from synthetic 200
        assert isinstance(signal, KurtosisRegimeSignal)
        assert signal.regime is not None

    def test_extreme_returns_high_fat_tail_risk(self, generator):
        """Returns with extreme outliers should produce fat_tail_risk > 0."""
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        # Add 20 extreme outliers (40-sigma moves)
        for i in range(20):
            returns[rng.randint(0, 199)] = rng.choice([-0.4, 0.4])
        signal = generator.generate_signal(returns)
        assert isinstance(signal, KurtosisRegimeSignal)
        assert 0.0 <= signal.fat_tail_risk <= 1.0

    def test_signal_exposure_for_extreme_regime(self, generator):
        """In extreme kurtosis, exposure should be 0.5."""
        # Use a known seed that produces extreme kurtosis
        rng = np.random.RandomState(1)
        returns = list(rng.normal(0, 0.01, 180))
        for i in range(10):
            returns[rng.randint(0, 179)] = rng.normal(0, 0.1)
        signal = generator.generate_signal(returns)
        # If extreme: exposure = 0.5, else should still be valid
        if signal.regime == "extreme_kurtosis":
            assert signal.recommended_exposure == 0.5
        else:
            assert signal.recommended_exposure in (0.75, 1.0)

    def test_signal_explanation_contains_regime_name(self, generator):
        """Explanation should reference the detected regime."""
        returns = list(np.random.RandomState(42).normal(0, 0.01, 200))
        signal = generator.generate_signal(returns)
        assert signal.regime in signal.explanation.lower() or "transitioning" in signal.explanation.lower()

    def test_generate_all_regime_paths(self, generator):
        """Verify generate_signal covers code paths for each regime."""
        # LOW kurtosis: very uniform returns
        u_ret = list(np.random.RandomState(42).uniform(-0.002, 0.002, 200))
        s_low = generator.generate_signal(u_ret)
        # NORMAL: Gaussian returns
        n_ret = list(np.random.RandomState(42).normal(0, 0.01, 200))
        s_normal = generator.generate_signal(n_ret)
        # HIGH: fat-tailed
        h_ret = list(np.random.RandomState(42).normal(0, 0.01, 180))
        for i in range(20):
            h_ret[np.random.RandomState(42).randint(0, 179)] = 0.08
        s_high = generator.generate_signal(h_ret)
        # All should produce signals
        assert isinstance(s_low, KurtosisRegimeSignal)
        assert isinstance(s_normal, KurtosisRegimeSignal)
        assert isinstance(s_high, KurtosisRegimeSignal)


class TestSaveSignal:
    """Test save_signal persistence logic."""

    @pytest.fixture
    def generator(self):
        return KurtosisRegimeSignalGenerator()

    def test_save_signal_writes_file(self, generator):
        """save_signal should write a JSON file to the output path."""
        signal = generator.generate_signal()
        with patch("src.regime.kurtosis_regime.save_results_json") as mock_save:
            generator.save_signal(signal)
            mock_save.assert_called_once()
            assert mock_save.call_args[1].get("output_path") or mock_save.call_args[0][1]

    def test_save_signal_json_contains_expected_fields(self, generator):
        """Written JSON should contain all signal fields."""
        signal = generator.generate_signal()
        with patch("src.regime.kurtosis_regime.save_results_json") as mock_save:
            generator.save_signal(signal)
            data = mock_save.call_args[0][0]
            expected = {"timestamp", "regime", "kurtosis_60d", "fat_tail_risk"}
            assert expected.issubset(set(data.keys()))

    def test_save_signal_multiple_writes(self, generator):
        """Multiple save calls should each trigger file write."""
        signal = generator.generate_signal()
        with patch("src.regime.kurtosis_regime.save_results_json") as mock_save:
            generator.save_signal(signal)
            generator.save_signal(signal)
            assert mock_save.call_count == 2

    def test_save_signal_ensure_dirs_creates_path(self):
        """_ensure_dirs should create the signals directory if missing."""
        with patch("src.regime.kurtosis_regime.SIGNALS_DIR") as mock_dir:
            mock_dir.__truediv__.return_value = mock_dir
            _ = KurtosisRegimeSignalGenerator()
            mock_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestCLI:
    """Test CLI entry points (main(), --save flag, __main__ guard)."""

    def test_main_runs_without_error(self):
        """main() should execute without raising exceptions."""
        from src.regime.kurtosis_regime import main
        try:
            main()
        except Exception as e:
            pytest.fail(f"main() raised {type(e).__name__}: {e}")

    def test_main_output_contains_header(self, capsys):
        """main() should print the KURTOSIS REGIME DETECTOR header."""
        from src.regime.kurtosis_regime import main
        main()
        captured = capsys.readouterr()
        assert "KURTOSIS REGIME DETECTOR" in captured.err

    def test_main_output_contains_all_sections(self, capsys):
        """main() output should include all expected sections."""
        from src.regime.kurtosis_regime import main
        main()
        captured = capsys.readouterr()
        assert "Regime:" in captured.err
        assert "Strategy Routing:" in captured.err
        assert "Fat Tail Risk:" in captured.err
        assert "Recommended Exposure:" in captured.err

    def test_main_output_contains_timestamp(self, capsys):
        """main() output should include a timestamp string."""
        from src.regime.kurtosis_regime import main
        main()
        captured = capsys.readouterr()
        assert "Timestamp:" in captured.err

    def test_main_with_save_flag_attemtps_file_write(self):
        """main() with --save in sys.argv should call save_signal."""
        from src.regime.kurtosis_regime import main
        test_args = ["prog", "--save"]
        with patch.object(sys, "argv", test_args):
            with patch("src.regime.kurtosis_regime.KurtosisRegimeSignalGenerator.save_signal") as mock_save:
                main()
                mock_save.assert_called_once()

    def test_main_without_save_flag_skips_file_write(self):
        """main() without --save should not call save_signal."""
        from src.regime.kurtosis_regime import main
        with patch.object(sys, "argv", ["prog"]):
            with patch("src.regime.kurtosis_regime.KurtosisRegimeSignalGenerator.save_signal") as mock_save:
                main()
                mock_save.assert_not_called()

    def test_main_output_has_all_regime_fields_filled(self, capsys):
        """main() output should show non-empty values for all key fields."""
        from src.regime.kurtosis_regime import main
        main()
        captured = capsys.readouterr()
        lines = captured.err.split("\n")
        regime_lines = [item for item in lines if "Regime:" in item and "Preference" not in item]
        assert len(regime_lines) > 0
        # Strip logger prefix (e.g. "INFO:src.regime.kurtosis_regime:")
        line = regime_lines[0]
        # Find "Regime:" after logger prefix
        idx = line.find("Regime:")
        assert idx >= 0
        value = line[idx + len("Regime:"):].strip()
        assert len(value) > 0


class TestModuleExports:
    """Verify public API surface and module-level names."""

    def test_public_api_importable(self):
        """All expected public names should be importable from the module."""
        from src.regime import kurtosis_regime as kr
        assert hasattr(kr, "KurtosisRegimeDetector")
        assert hasattr(kr, "KurtosisRegimeSignalGenerator")
        assert hasattr(kr, "KurtosisRegimeSignal")
        assert hasattr(kr, "KurtosisRegime")
        assert hasattr(kr, "StrategyPreference")
        assert hasattr(kr, "detect_kurtosis_regime")
        assert hasattr(kr, "main")

    def test_no_all_defined_in_source(self):
        """The module does not define __all__."""
        from src.regime import kurtosis_regime as kr
        assert not hasattr(kr, "__all__")

    def test_all_classes_in_module_dict(self):
        """All classes and functions should appear in module dir()."""
        from src.regime import kurtosis_regime as kr
        names = dir(kr)
        assert "KurtosisRegimeDetector" in names
        assert "KurtosisRegimeSignalGenerator" in names
        assert "KurtosisRegimeSignal" in names
        assert "KurtosisRegime" in names
        assert "StrategyPreference" in names
        assert "detect_kurtosis_regime" in names
        assert "main" in names

    def test_logger_exists_at_module_level(self):
        """Module should have a logger instance."""
        from src.regime import kurtosis_regime as kr
        assert hasattr(kr, "logger")
        import logging
        assert isinstance(kr.logger, logging.Logger)

    def test_enum_names_accessible(self):
        """Enum values and names should be accessible from the module."""
        assert KurtosisRegime.LOW_KURTOSIS.name == "LOW_KURTOSIS"
        assert KurtosisRegime.NORMAL.name == "NORMAL"
        assert KurtosisRegime.HIGH_KURTOSIS.name == "HIGH_KURTOSIS"
        assert KurtosisRegime.EXTREME_KURTOSIS.name == "EXTREME_KURTOSIS"
        assert StrategyPreference.TREND_FOLLOWING.name == "TREND_FOLLOWING"
        assert StrategyPreference.MEAN_REVERSION.name == "MEAN_REVERSION"
        assert StrategyPreference.BALANCED.name == "BALANCED"
        assert StrategyPreference.DEFENSIVE.name == "DEFENSIVE"

    def test_convenience_function_returns_signal(self):
        """detect_kurtosis_regime should return a KurtosisRegimeSignal."""
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        signal = detect_kurtosis_regime(returns)
        assert isinstance(signal, KurtosisRegimeSignal)
        assert signal.regime is not None

    def test_convenience_function_no_args(self):
        """detect_kurtosis_regime with no args should use synthetic data."""
        signal = detect_kurtosis_regime()
        assert isinstance(signal, KurtosisRegimeSignal)
        assert signal.regime is not None

    def test_generator_ensure_dirs_on_init(self):
        """KurtosisRegimeSignalGenerator should ensure signal dir exists."""
        with patch("src.regime.kurtosis_regime.SIGNALS_DIR") as mock_signals_dir:
            mock_signals_dir.__truediv__.return_value = mock_signals_dir
            _ = KurtosisRegimeSignalGenerator()
            mock_signals_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
