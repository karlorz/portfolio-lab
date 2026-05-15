"""
Tests for Kurtosis Regime Detector (v4.91)
"""

import pytest
import numpy as np
from datetime import datetime

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
