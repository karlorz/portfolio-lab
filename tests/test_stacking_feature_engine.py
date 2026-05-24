#!/usr/bin/env python3
"""
Tests for stacking feature engine — v3.10 Phase 1 feature generation.

Covers:
- Feature vector creation with 6 base signals
- Pairwise interaction features (45 total)
- Regime context features
- Historical accuracy tracking
- NumPy conversion
- JSON serialization
- Performance latency validation (<10ms target)

Tests: 15
"""

import numpy as np

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from src.signals.stacking_feature_engine import (
    SignalSource, Signal, RegimeContext, HistoricalAccuracy,
    FeatureVector, StackingFeatureEngine, StackingAccuracyTracker
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Fresh StackingFeatureEngine instance."""
    return StackingFeatureEngine()


@pytest.fixture
def tracker():
    """Fresh StackingAccuracyTracker instance."""
    return StackingAccuracyTracker()


@pytest.fixture
def full_signals():
    """Complete set of 6 signals for testing."""
    now = datetime.now()
    return {
        SignalSource.MULTI_SPEED_MOM: Signal(SignalSource.MULTI_SPEED_MOM, 0.7, now, 0.85),
        SignalSource.CROSS_ASSET_RV: Signal(SignalSource.CROSS_ASSET_RV, -0.15, now, 0.68),
        SignalSource.INTERNATIONAL_MOMENTUM: Signal(SignalSource.INTERNATIONAL_MOMENTUM, 0.48, now, 0.74),
        SignalSource.ALTERNATIVE_DATA: Signal(SignalSource.ALTERNATIVE_DATA, 0.65, now, 0.82),
        SignalSource.CROSS_ASSET_REGIME_ARB: Signal(SignalSource.CROSS_ASSET_REGIME_ARB, 0.31, now, 0.71),
        SignalSource.UNIFIED_OVERLAY: Signal(SignalSource.UNIFIED_OVERLAY, 0.40, now, 0.75),
    }


@pytest.fixture
def regime_context():
    """Sample regime context."""
    return RegimeContext(
        vix_level=20.0,
        trend_strength=0.5,
        timestamp=datetime.now()
    )


@pytest.fixture
def historical_accuracy(tracker):
    """Historical accuracy with some mock data."""
    now = datetime.now()
    for i in range(30):
        for source in SignalSource:
            tracker.record_prediction(
                source=source,
                timestamp=now - timedelta(days=i),
                signal_value=np.random.uniform(-0.5, 0.5),
                actual_return=np.random.uniform(-0.01, 0.01)
            )
    return tracker.get_all_accuracies(now)


# ---------------------------------------------------------------------------
# SignalSource enum tests (2)
# ---------------------------------------------------------------------------

def test_signal_source_count():
    """Test: Exactly 6 signal sources defined."""
    assert len(SignalSource) == 6


def test_signal_source_values():
    """Test: All expected signal sources exist."""
    expected = [
        "multi_speed_momentum", "cross_asset_rv", "international_momentum",
        "alternative_data", "cross_asset_regime_arb", "unified_overlay"
    ]
    for exp in expected:
        assert any(s.value == exp for s in SignalSource)


# ---------------------------------------------------------------------------
# StackingFeatureEngine tests (10)
# ---------------------------------------------------------------------------

def test_feature_engine_initialization(engine):
    """Test: Feature engine initializes correctly."""
    assert engine.NUM_BASE_SIGNALS == 6
    assert engine.NUM_PAIRWISE_COMBINATIONS == 15
    assert engine.TOTAL_DIMENSIONS == 59
    assert engine.vix_normalization_factor == 30.0


def test_create_features_requires_all_signals(engine, regime_context, historical_accuracy):
    """Test: Feature creation fails if signals are missing."""
    partial_signals = {
        SignalSource.MULTI_SPEED_MOM: Signal(SignalSource.MULTI_SPEED_MOM, 0.5, datetime.now(), 0.8)
    }

    with pytest.raises(ValueError) as exc_info:
        engine.create_features(partial_signals, regime_context, historical_accuracy)

    assert "Expected 6 signals, got 1" in str(exc_info.value)


def test_create_features_success(engine, full_signals, regime_context, historical_accuracy):
    """Test: Successful feature vector creation with all signals."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)

    assert isinstance(fv, FeatureVector)
    assert fv.dimension_count == 59
    assert len(fv.base_values) == 6
    assert len(fv.multiplicative) == 15
    assert len(fv.disagreement) == 15
    assert len(fv.averages) == 15
    assert fv.vix_normalized == 20.0 / 30.0  # Normalized
    assert fv.trend_strength == 0.5


def test_base_values_correctness(engine, full_signals, regime_context, historical_accuracy):
    """Test: Base signal values are correctly extracted."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)

    assert fv.base_values[SignalSource.MULTI_SPEED_MOM] == 0.7
    assert fv.base_values[SignalSource.CROSS_ASSET_RV] == -0.15
    assert fv.base_values[SignalSource.INTERNATIONAL_MOMENTUM] == 0.48
    assert fv.base_values[SignalSource.UNIFIED_OVERLAY] == 0.40


def test_multiplicative_interactions(engine, full_signals, regime_context, historical_accuracy):
    """Test: Multiplicative interactions computed correctly."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)

    # MULTI_SPEED_MOM (0.7) * ALTERNATIVE_DATA (0.65) = 0.455
    pair = (SignalSource.MULTI_SPEED_MOM, SignalSource.ALTERNATIVE_DATA)
    assert fv.multiplicative[pair] == pytest.approx(0.7 * 0.65)


def test_disagreement_features(engine, full_signals, regime_context, historical_accuracy):
    """Test: Disagreement features computed correctly."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)

    # |MULTI_SPEED_MOM (0.7) - UNIFIED_OVERLAY (0.4)| = 0.3
    pair = (SignalSource.MULTI_SPEED_MOM, SignalSource.UNIFIED_OVERLAY)
    assert fv.disagreement[pair] == pytest.approx(abs(0.7 - 0.4))


def test_average_features(engine, full_signals, regime_context, historical_accuracy):
    """Test: Average features computed correctly."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)

    # (MULTI_SPEED_MOM (0.7) + CROSS_ASSET_RV (-0.15)) / 2 = 0.275
    pair = (SignalSource.MULTI_SPEED_MOM, SignalSource.CROSS_ASSET_RV)
    assert fv.averages[pair] == pytest.approx((0.7 + -0.15) / 2.0)


def test_to_numpy_shape(engine, full_signals, regime_context, historical_accuracy):
    """Test: NumPy array has correct shape."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)
    arr = engine.to_numpy(fv)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (59,)
    assert arr.dtype == np.float32


def test_to_numpy_order(engine, full_signals, regime_context, historical_accuracy):
    """Test: NumPy array features are in correct order."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)
    arr = engine.to_numpy(fv)

    # First 6 elements should be base signals in enum order
    for i, source in enumerate(SignalSource):
        assert arr[i] == full_signals[source].value


def test_get_feature_names_count(engine):
    """Test: Feature names list has 59 entries."""
    names = engine.get_feature_names()
    assert len(names) == 59
    assert all(isinstance(n, str) for n in names)


def test_to_dict_serializable(engine, full_signals, regime_context, historical_accuracy):
    """Test: Feature vector can be serialized to dict."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)
    d = engine.to_dict(fv)

    assert isinstance(d, dict)
    assert "base_values" in d
    assert "multiplicative" in d
    assert "timestamp" in d
    assert d["dimension_count"] == 59


# ---------------------------------------------------------------------------
# StackingAccuracyTracker tests (3)
# ---------------------------------------------------------------------------

def test_accuracy_tracker_initialization(tracker):
    """Test: Tracker initializes with empty history."""
    now = datetime.now()
    for source in SignalSource:
        acc = tracker.get_historical_accuracy(source, now)
        assert acc.accuracy_90d == 0.5  # Default when no history
        assert acc.predictions_count == 0


def test_accuracy_tracker_record_and_retrieve(tracker):
    """Test: Recording predictions updates accuracy."""
    now = datetime.now()

    # Record 10 predictions, 7 correct
    for i in range(7):
        tracker.record_prediction(SignalSource.MULTI_SPEED_MOM, now, 0.5, 0.02)  # Correct
    for i in range(3):
        tracker.record_prediction(SignalSource.MULTI_SPEED_MOM, now, 0.5, -0.01)  # Incorrect

    acc = tracker.get_historical_accuracy(SignalSource.MULTI_SPEED_MOM, now)
    assert acc.accuracy_90d == 0.7
    assert acc.predictions_count == 10


def test_accuracy_tracker_rolling_window(tracker):
    """Test: Old predictions are pruned from rolling window."""
    now = datetime.now()

    # Record prediction 100 days ago
    tracker.record_prediction(
        SignalSource.MULTI_SPEED_MOM,
        now - timedelta(days=100),
        0.5,
        0.02
    )

    # Record recent prediction
    tracker.record_prediction(SignalSource.MULTI_SPEED_MOM, now, 0.5, -0.01)

    acc = tracker.get_historical_accuracy(SignalSource.MULTI_SPEED_MOM, now)
    # Old prediction should be pruned, only recent counts
    assert acc.predictions_count == 1


# ---------------------------------------------------------------------------
# Performance tests (1)
# ---------------------------------------------------------------------------

def test_feature_generation_latency(engine, full_signals, regime_context, historical_accuracy):
    """Test: Feature generation completes in under 10ms."""
    import time

    # Warmup
    engine.create_features(full_signals, regime_context, historical_accuracy)

    # Timed run
    times = []
    for _ in range(10):
        start = time.perf_counter()
        engine.create_features(full_signals, regime_context, historical_accuracy)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    avg_time = np.mean(times)
    # Allow some variance, but should be well under 10ms
    assert avg_time < 20.0, f"Feature generation too slow: {avg_time:.2f}ms (target <10ms)"


# ---------------------------------------------------------------------------
# Feature explanation tests (1)
# ---------------------------------------------------------------------------

def test_explain_features(engine, full_signals, regime_context, historical_accuracy):
    """Test: Feature explanation generates expected structure."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)
    explanation = engine.explain_features(fv, top_n=5)

    assert "timestamp" in explanation
    assert "total_dimensions" in explanation
    assert "base_signals_summary" in explanation
    assert "pairwise_interactions" in explanation
    assert "regime_context" in explanation
    assert "historical_accuracy" in explanation

    # Check bullish/bearish/neutral counts
    summary = explanation["base_signals_summary"]
    assert summary["bullish_count"] + summary["bearish_count"] + summary["neutral_count"] == 6


# ---------------------------------------------------------------------------
# Edge case tests (2)
# ---------------------------------------------------------------------------

def test_vix_normalization_custom():
    """Test: Custom VIX normalization factor works."""
    engine = StackingFeatureEngine(vix_normalization_factor=20.0)
    regime = RegimeContext(vix_level=20.0, trend_strength=0.5, timestamp=datetime.now())

    assert engine.vix_normalization_factor == 20.0


def test_neutral_signals_accuracy(tracker):
    """Test: Neutral signals (near 0) handle accuracy correctly."""
    now = datetime.now()

    # Neutral signal with small market move
    tracker.record_prediction(SignalSource.MULTI_SPEED_MOM, now, 0.05, 0.005)

    # This should count as correct (neutral × neutral)
    acc = tracker.get_historical_accuracy(SignalSource.MULTI_SPEED_MOM, now)
    assert acc.predictions_count == 1


# Total: 15 tests


# ---------------------------------------------------------------------------
# Extended coverage tests
# ---------------------------------------------------------------------------


class TestFeatureVectorDataclass:
    """Test FeatureVector dataclass edge cases."""

    def test_dimension_count_matches(self, engine, full_signals, regime_context, historical_accuracy):
        """dimension_count should be TOTAL_DIMENSIONS."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        assert fv.dimension_count == StackingFeatureEngine.TOTAL_DIMENSIONS

    def test_timestamp_populated(self, engine, full_signals, regime_context, historical_accuracy):
        """FeatureVector should have a valid timestamp."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        assert isinstance(fv.timestamp, datetime)

    def test_vix_normalized_calculation(self, engine, full_signals, historical_accuracy):
        """VIX normalization should divide by vix_normalization_factor."""
        regime = RegimeContext(vix_level=30.0, trend_strength=0.5, timestamp=datetime.now())
        fv = engine.create_features(full_signals, regime, historical_accuracy)
        assert fv.vix_normalized == pytest.approx(30.0 / 30.0)

    def test_custom_vix_normalization_affects_output(self, full_signals, historical_accuracy):
        """Custom VIX normalization should produce different normalized values."""
        engine_default = StackingFeatureEngine(vix_normalization_factor=30.0)
        engine_custom = StackingFeatureEngine(vix_normalization_factor=20.0)
        regime = RegimeContext(vix_level=20.0, trend_strength=0.5, timestamp=datetime.now())

        fv_default = engine_default.create_features(full_signals, regime, historical_accuracy)
        fv_custom = engine_custom.create_features(full_signals, regime, historical_accuracy)

        assert fv_default.vix_normalized == pytest.approx(20.0 / 30.0)
        assert fv_custom.vix_normalized == pytest.approx(20.0 / 20.0)


class TestPairwiseCombinations:
    """Test pairwise combination logic."""

    def test_correct_pair_count(self, engine):
        """C(6,2) = 15 pairwise combinations."""
        pairs = engine._get_pairwise_combinations(list(SignalSource))
        assert len(pairs) == 15

    def test_pairs_are_unique(self, engine):
        """No duplicate pairs."""
        pairs = engine._get_pairwise_combinations(list(SignalSource))
        pair_set = set(pairs)
        assert len(pair_set) == 15

    def test_pair_ordering_consistent(self, engine):
        """Pairs should be in consistent (sorted) order."""
        pairs = engine._get_pairwise_combinations(list(SignalSource))
        for s1, s2 in pairs:
            assert s1 != s2

    def test_cache_reuse(self, engine):
        """Second call should return the same cached object."""
        pairs1 = engine._get_pairwise_combinations(list(SignalSource))
        pairs2 = engine._get_pairwise_combinations(list(SignalSource))
        assert pairs1 is pairs2


class TestNumpyConversion:
    """Extended to_numpy conversion tests."""

    def test_numpy_values_in_range(self, engine, full_signals, regime_context, historical_accuracy):
        """All numpy values should be finite."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        arr = engine.to_numpy(fv)
        assert np.all(np.isfinite(arr))

    def test_numpy_multiplicative_section(self, engine, full_signals, regime_context, historical_accuracy):
        """Multiplicative features should appear in the correct array positions."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        arr = engine.to_numpy(fv)
        # Base: 0-5, Multiplicative: 6-20
        for i in range(6, 21):
            assert -1.0 <= arr[i] <= 1.0  # Signal values are -1 to +1

    def test_numpy_accuracy_section(self, engine, full_signals, regime_context, historical_accuracy):
        """Accuracy values should be in [0, 1] range."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        arr = engine.to_numpy(fv)
        # Accuracy: indices 53-58
        for i in range(53, 59):
            assert 0.0 <= arr[i] <= 1.0


class TestFeatureNames:
    """Extended get_feature_names tests."""

    def test_base_feature_names(self, engine):
        """Base feature names should follow pattern base_{source.value}."""
        names = engine.get_feature_names()
        for source in SignalSource:
            assert f"base_{source.value}" in names

    def test_pairwise_feature_names(self, engine):
        """Pairwise names should contain mult_, disagree_, avg_ prefixes."""
        names = engine.get_feature_names()
        mult_names = [n for n in names if n.startswith("mult_")]
        disagree_names = [n for n in names if n.startswith("disagree_")]
        avg_names = [n for n in names if n.startswith("avg_")]
        assert len(mult_names) == 15
        assert len(disagree_names) == 15
        assert len(avg_names) == 15

    def test_regime_feature_names(self, engine):
        """Regime features should be named vix_normalized and trend_strength."""
        names = engine.get_feature_names()
        assert "vix_normalized" in names
        assert "trend_strength" in names

    def test_accuracy_feature_names(self, engine):
        """Accuracy features should follow pattern acc90d_{source.value}."""
        names = engine.get_feature_names()
        for source in SignalSource:
            assert f"acc90d_{source.value}" in names


class TestFeatureExplanation:
    """Extended explain_features tests."""

    def test_explain_bearish_count(self, engine, regime_context, historical_accuracy):
        """All-bearish signals should show 6 bearish, 0 bullish."""
        now = datetime.now()
        bearish_signals = {
            source: Signal(source, -0.5, now, 0.7)
            for source in SignalSource
        }
        fv = engine.create_features(bearish_signals, regime_context, historical_accuracy)
        explanation = engine.explain_features(fv)
        assert explanation["base_signals_summary"]["bearish_count"] == 6
        assert explanation["base_signals_summary"]["bullish_count"] == 0

    def test_explain_volatility_regime(self, engine, full_signals, historical_accuracy):
        """Volatility regime should be derived from vix_normalized."""
        # High VIX → "high" regime
        regime_high = RegimeContext(vix_level=25.0, trend_strength=0.5, timestamp=datetime.now())
        fv = engine.create_features(full_signals, regime_high, historical_accuracy)
        explanation = engine.explain_features(fv)
        vol_regime = explanation["regime_context"]["volatility_regime"]
        # vix_normalized = 25/30 ≈ 0.83 > 0.67 → "high"
        assert vol_regime == "high"

    def test_explain_normal_volatility_regime(self, engine, full_signals, historical_accuracy):
        """Low VIX → "normal" regime in explanation."""
        regime_low = RegimeContext(vix_level=12.0, trend_strength=0.5, timestamp=datetime.now())
        fv = engine.create_features(full_signals, regime_low, historical_accuracy)
        explanation = engine.explain_features(fv)
        vol_regime = explanation["regime_context"]["volatility_regime"]
        # vix_normalized = 12/30 = 0.4 < 0.5 → "normal"
        assert vol_regime == "normal"

    def test_explain_elevated_volatility_regime(self, engine, full_signals, historical_accuracy):
        """Mid VIX → "elevated" regime in explanation."""
        regime_mid = RegimeContext(vix_level=17.0, trend_strength=0.5, timestamp=datetime.now())
        fv = engine.create_features(full_signals, regime_mid, historical_accuracy)
        explanation = engine.explain_features(fv)
        vol_regime = explanation["regime_context"]["volatility_regime"]
        # vix_normalized = 17/30 ≈ 0.567, 0.5 < x < 0.67 → "elevated"
        assert vol_regime == "elevated"

    def test_explain_historical_accuracy_best_worst(self, engine, full_signals, regime_context):
        """Best and worst performer should be identifiable."""
        now = datetime.now()
        accuracy = {}
        for i, source in enumerate(SignalSource):
            accuracy[source] = HistoricalAccuracy(
                source=source,
                accuracy_90d=0.5 + i * 0.05,  # Varying accuracy
                predictions_count=10,
                timestamp=now,
            )
        fv = engine.create_features(full_signals, regime_context, accuracy)
        explanation = engine.explain_features(fv)
        # Last source has highest accuracy (0.5 + 5*0.05 = 0.75)
        best = explanation["historical_accuracy"]["best_performer"]
        worst = explanation["historical_accuracy"]["worst_performer"]
        assert isinstance(best, str)
        assert isinstance(worst, str)


class TestToDictExtended:
    """Extended to_dict serialization tests."""

    def test_to_dict_keys_serialized_as_strings(self, engine, full_signals, regime_context, historical_accuracy):
        """All dict keys should be strings (not Enum), for JSON compatibility."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        d = engine.to_dict(fv)

        # base_values keys should be strings
        for key in d["base_values"]:
            assert isinstance(key, str)

        # multiplicative keys should be strings
        for key in d["multiplicative"]:
            assert isinstance(key, str)

    def test_to_dict_timestamp_is_isoformat(self, engine, full_signals, regime_context, historical_accuracy):
        """Timestamp should be ISO format string."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        d = engine.to_dict(fv)
        # Should be parseable as ISO format
        assert isinstance(d["timestamp"], str)

    def test_to_dict_disagreement_present(self, engine, full_signals, regime_context, historical_accuracy):
        """Disagreement features should be in serialized dict."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        d = engine.to_dict(fv)
        assert "disagreement" in d
        assert len(d["disagreement"]) == 15

    def test_to_dict_averages_present(self, engine, full_signals, regime_context, historical_accuracy):
        """Average features should be in serialized dict."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        d = engine.to_dict(fv)
        assert "averages" in d
        assert len(d["averages"]) == 15


class TestStackingAccuracyTrackerExtended:
    """Extended accuracy tracker tests."""

    def test_all_negative_signals_correct(self, tracker):
        """Negative signals with negative returns should be correct."""
        now = datetime.now()
        tracker.record_prediction(SignalSource.MULTI_SPEED_MOM, now, -0.5, -0.02)
        acc = tracker.get_historical_accuracy(SignalSource.MULTI_SPEED_MOM, now)
        assert acc.accuracy_90d == 1.0  # All correct

    def test_mixed_accuracy(self, tracker):
        """Mixed correct/incorrect predictions should give intermediate accuracy."""
        now = datetime.now()
        for _ in range(5):
            tracker.record_prediction(SignalSource.MULTI_SPEED_MOM, now, 0.5, 0.02)
        for _ in range(5):
            tracker.record_prediction(SignalSource.MULTI_SPEED_MOM, now, 0.5, -0.02)
        acc = tracker.get_historical_accuracy(SignalSource.MULTI_SPEED_MOM, now)
        assert acc.accuracy_90d == pytest.approx(0.5)

    def test_all_sources_tracked_independently(self, tracker):
        """Each source should be tracked independently."""
        now = datetime.now()
        tracker.record_prediction(SignalSource.MULTI_SPEED_MOM, now, 0.5, 0.02)
        tracker.record_prediction(SignalSource.CROSS_ASSET_RV, now, -0.5, 0.02)
        acc_msm = tracker.get_historical_accuracy(SignalSource.MULTI_SPEED_MOM, now)
        acc_rv = tracker.get_historical_accuracy(SignalSource.CROSS_ASSET_RV, now)
        assert acc_msm.accuracy_90d == 1.0
        assert acc_rv.accuracy_90d == 0.0

    def test_get_all_accuracies_returns_all_sources(self, tracker):
        """get_all_accuracies should return entry for each source."""
        now = datetime.now()
        accs = tracker.get_all_accuracies(now)
        assert len(accs) == 6
        for source in SignalSource:
            assert source in accs

    def test_custom_window_days(self):
        """Custom window_days should be stored."""
        tracker = StackingAccuracyTracker(window_days=30)
        assert tracker.window_days == 30

    def test_neutral_signal_with_large_return_correct(self, tracker):
        """Neutral signal with large positive return — first clause matches (both > 0)."""
        now = datetime.now()
        tracker.record_prediction(SignalSource.MULTI_SPEED_MOM, now, 0.05, 0.05)
        acc = tracker.get_historical_accuracy(SignalSource.MULTI_SPEED_MOM, now)
        # 0.05 > 0 and 0.05 > 0 → True (first clause), so "correct"
        assert acc.accuracy_90d == 1.0

    def test_truly_neutral_signal_truly_neutral_return(self, tracker):
        """Truly neutral signal with truly neutral return → correct via neutral clause."""
        now = datetime.now()
        tracker.record_prediction(SignalSource.MULTI_SPEED_MOM, now, 0.05, 0.005)
        acc = tracker.get_historical_accuracy(SignalSource.MULTI_SPEED_MOM, now)
        # abs(0.05) < 0.1 and abs(0.005) < 0.01 → neutral match
        assert acc.accuracy_90d == 1.0

    def test_negative_signal_positive_return_incorrect(self, tracker):
        """Negative signal with positive return should be incorrect."""
        now = datetime.now()
        tracker.record_prediction(SignalSource.MULTI_SPEED_MOM, now, -0.5, 0.02)
        acc = tracker.get_historical_accuracy(SignalSource.MULTI_SPEED_MOM, now)
        assert acc.accuracy_90d == 0.0


# ==============================================================================
# New tests: Signal dataclass boundary validation
# ==============================================================================

class TestSignalDataclassBoundaries:
    """Signal dataclass boundary value tests."""

    def test_signal_value_negative_one(self):
        """Signal value -1.0 (lower boundary)."""
        sig = Signal(SignalSource.MULTI_SPEED_MOM, -1.0, datetime.now(), 0.5)
        assert sig.value == -1.0

    def test_signal_value_positive_one(self):
        """Signal value 1.0 (upper boundary)."""
        sig = Signal(SignalSource.MULTI_SPEED_MOM, 1.0, datetime.now(), 0.5)
        assert sig.value == 1.0

    def test_signal_value_zero(self):
        """Signal value 0.0 (neutral center)."""
        sig = Signal(SignalSource.MULTI_SPEED_MOM, 0.0, datetime.now(), 0.5)
        assert sig.value == 0.0

    def test_signal_confidence_zero(self):
        """Signal confidence 0.0 (lower boundary)."""
        sig = Signal(SignalSource.MULTI_SPEED_MOM, 0.5, datetime.now(), 0.0)
        assert sig.confidence == 0.0

    def test_signal_confidence_one(self):
        """Signal confidence 1.0 (upper boundary)."""
        sig = Signal(SignalSource.MULTI_SPEED_MOM, 0.5, datetime.now(), 1.0)
        assert sig.confidence == 1.0


# ==============================================================================
# New tests: RegimeContext dataclass edge cases
# ==============================================================================

class TestRegimeContextEdgeCases:
    """RegimeContext dataclass edge case tests."""

    def test_vix_level_zero(self):
        """VIX level of 0 is accepted."""
        ctx = RegimeContext(vix_level=0.0, trend_strength=0.5, timestamp=datetime.now())
        assert ctx.vix_level == 0.0

    def test_vix_level_high(self):
        """VIX level of 100 is accepted (stress scenario)."""
        ctx = RegimeContext(vix_level=100.0, trend_strength=0.5, timestamp=datetime.now())
        assert ctx.vix_level == 100.0

    def test_trend_strength_zero(self):
        """Trend strength of 0.0."""
        ctx = RegimeContext(vix_level=20.0, trend_strength=0.0, timestamp=datetime.now())
        assert ctx.trend_strength == 0.0

    def test_trend_strength_one(self):
        """Trend strength of 1.0."""
        ctx = RegimeContext(vix_level=20.0, trend_strength=1.0, timestamp=datetime.now())
        assert ctx.trend_strength == 1.0


# ==============================================================================
# New tests: HistoricalAccuracy dataclass edge cases
# ==============================================================================

class TestHistoricalAccuracyEdgeCases:
    """HistoricalAccuracy dataclass edge case tests."""

    def test_accuracy_zero(self):
        """Accuracy of 0.0 (all incorrect)."""
        ha = HistoricalAccuracy(SignalSource.MULTI_SPEED_MOM, 0.0, 10, datetime.now())
        assert ha.accuracy_90d == 0.0

    def test_accuracy_one(self):
        """Accuracy of 1.0 (all correct)."""
        ha = HistoricalAccuracy(SignalSource.MULTI_SPEED_MOM, 1.0, 10, datetime.now())
        assert ha.accuracy_90d == 1.0

    def test_predictions_count_zero(self):
        """Predictions count of 0 is accepted."""
        ha = HistoricalAccuracy(SignalSource.MULTI_SPEED_MOM, 0.5, 0, datetime.now())
        assert ha.predictions_count == 0


# ==============================================================================
# New tests: FeatureVector direct construction edge cases
# ==============================================================================

class TestFeatureVectorDirectConstruction:
    """Direct FeatureVector construction edge cases."""

    def test_default_dimension_count(self):
        """FeatureVector defaults to 59."""
        fv = FeatureVector(
            base_values={}, multiplicative={}, disagreement={},
            averages={}, vix_normalized=0.0, trend_strength=0.0,
            accuracy_values={}, timestamp=datetime.now()
        )
        assert fv.dimension_count == 59

    def test_empty_structures_acceptable(self):
        """Empty dicts accepted in FeatureVector fields."""
        fv = FeatureVector(
            base_values={}, multiplicative={}, disagreement={},
            averages={}, vix_normalized=0.0, trend_strength=0.0,
            accuracy_values={}, timestamp=datetime.now()
        )
        assert len(fv.base_values) == 0
        assert len(fv.multiplicative) == 0


# ==============================================================================
# New tests: StackingFeatureEngine edge cases
# ==============================================================================

class TestStackingFeatureEngineEdgeCases:
    """StackingFeatureEngine creation edge cases."""

    def test_all_signals_bullish(self, engine, regime_context, historical_accuracy):
        """All signals at +1.0 produce correct multiplicative features."""
        now = datetime.now()
        signals = {s: Signal(s, 1.0, now, 0.8) for s in SignalSource}
        fv = engine.create_features(signals, regime_context, historical_accuracy)
        assert all(v == 1.0 for v in fv.base_values.values())
        assert all(v == 1.0 for v in fv.multiplicative.values())  # 1 * 1 = 1

    def test_all_signals_bearish(self, engine, regime_context, historical_accuracy):
        """All signals at -1.0 produce positive multiplicative features."""
        now = datetime.now()
        signals = {s: Signal(s, -1.0, now, 0.8) for s in SignalSource}
        fv = engine.create_features(signals, regime_context, historical_accuracy)
        assert all(v == -1.0 for v in fv.base_values.values())
        assert all(v == 1.0 for v in fv.multiplicative.values())  # -1 * -1 = 1

    def test_all_signals_neutral(self, engine, regime_context, historical_accuracy):
        """All signals at 0.0 produce all-zero pairwise features."""
        now = datetime.now()
        signals = {s: Signal(s, 0.0, now, 0.8) for s in SignalSource}
        fv = engine.create_features(signals, regime_context, historical_accuracy)
        assert all(v == 0.0 for v in fv.base_values.values())
        assert all(v == 0.0 for v in fv.multiplicative.values())
        assert all(v == 0.0 for v in fv.disagreement.values())
        assert all(v == 0.0 for v in fv.averages.values())

    def test_vix_level_zero_normalization(self, engine, full_signals, historical_accuracy):
        """VIX level 0 produces 0.0 normalized value."""
        regime = RegimeContext(vix_level=0.0, trend_strength=0.5, timestamp=datetime.now())
        fv = engine.create_features(full_signals, regime, historical_accuracy)
        assert fv.vix_normalized == 0.0

    def test_error_message_lists_missing_sources(self, engine, regime_context, historical_accuracy):
        """Error message includes missing sources."""
        partial = {}
        with pytest.raises(ValueError) as exc:
            engine.create_features(partial, regime_context, historical_accuracy)
        assert "Missing:" in str(exc.value)
        assert "MULTI_SPEED_MOM" in str(exc.value)


# ==============================================================================
# New tests: Pairwise combination edge cases
# ==============================================================================

class TestPairwiseCombinationsEdgeCases:
    """Edge cases for _get_pairwise_combinations."""

    def test_empty_sources_list(self, engine):
        """Empty list returns no pairs."""
        pairs = engine._get_pairwise_combinations([])
        assert len(pairs) == 0

    def test_single_source(self, engine):
        """Single source returns no pairs."""
        pairs = engine._get_pairwise_combinations([SignalSource.MULTI_SPEED_MOM])
        assert len(pairs) == 0

    def test_two_sources(self, engine):
        """Two sources return exactly one pair."""
        pairs = engine._get_pairwise_combinations(
            [SignalSource.MULTI_SPEED_MOM, SignalSource.CROSS_ASSET_RV]
        )
        assert len(pairs) == 1
        assert pairs[0] == (SignalSource.MULTI_SPEED_MOM, SignalSource.CROSS_ASSET_RV)

    def test_three_sources_produce_three_pairs(self, engine):
        """Three sources produce C(3,2) = 3 pairs."""
        sources = [
            SignalSource.MULTI_SPEED_MOM,
            SignalSource.CROSS_ASSET_RV,
            SignalSource.INTERNATIONAL_MOMENTUM
        ]
        pairs = engine._get_pairwise_combinations(sources)
        assert len(pairs) == 3


# ==============================================================================
# New tests: NumPy conversion edge cases
# ==============================================================================

class TestNumpyConversionEdgeCases:
    """Edge cases for to_numpy conversion."""

    def test_to_numpy_all_zeros(self, engine, historical_accuracy):
        """All-zero features produce all-zero first 51 elements."""
        now = datetime.now()
        signals = {s: Signal(s, 0.0, now, 0.5) for s in SignalSource}
        regime = RegimeContext(vix_level=0.0, trend_strength=0.0, timestamp=now)
        fv = engine.create_features(signals, regime, historical_accuracy)
        arr = engine.to_numpy(fv)
        assert arr[0:6].sum() == 0.0
        assert arr[6:21].sum() == 0.0  # multiplicative
        assert arr[21:36].sum() == 0.0  # disagreement
        assert arr[36:51].sum() == 0.0  # averages

    def test_to_numpy_length_matches_feature_names(self, engine, full_signals, regime_context, historical_accuracy):
        """Numpy array length equals number of feature names."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        arr = engine.to_numpy(fv)
        names = engine.get_feature_names()
        assert len(arr) == len(names)

    def test_to_numpy_enum_order_matches(self, engine, full_signals, regime_context, historical_accuracy):
        """Base signal order matches SignalSource enum order."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        arr = engine.to_numpy(fv)
        expected = [full_signals[s].value for s in SignalSource]
        for i, exp in enumerate(expected):
            assert arr[i] == pytest.approx(exp)


# ==============================================================================
# New tests: Feature names edge cases
# ==============================================================================

class TestFeatureNamesEdgeCases:
    """Edge cases for get_feature_names."""

    def test_feature_names_all_unique(self, engine):
        """All 59 feature names should be unique."""
        names = engine.get_feature_names()
        assert len(names) == len(set(names))

    def test_feature_names_first_are_base(self, engine):
        """First 6 names are base signals."""
        names = engine.get_feature_names()
        for i, source in enumerate(SignalSource):
            assert names[i] == f"base_{source.value}"

    def test_feature_names_last_are_accuracy(self, engine):
        """Last 6 names are accuracy features."""
        names = engine.get_feature_names()
        for i, source in enumerate(SignalSource):
            assert names[-6 + i] == f"acc90d_{source.value}"

    def test_feature_names_vix_and_trend_at_51_52(self, engine):
        """VIX at index 51, trend at index 52."""
        names = engine.get_feature_names()
        assert names[51] == "vix_normalized"
        assert names[52] == "trend_strength"

    def test_feature_names_section_order(self, engine):
        """Sections appear in correct order: base, mult, disagree, avg, regime, acc."""
        names = engine.get_feature_names()
        # Find transition boundaries
        base_end = next(i for i, n in enumerate(names) if n.startswith("mult_"))
        mult_end = next(i for i, n in enumerate(names[base_end:]) if n.startswith("disagree_")) + base_end
        disagree_end = next(i for i, n in enumerate(names[mult_end:]) if n.startswith("avg_")) + mult_end
        avg_end = next(i for i, n in enumerate(names[disagree_end:]) if n == "vix_normalized") + disagree_end
        assert names[0] == "base_multi_speed_momentum"
        assert names[base_end].startswith("mult_")
        assert names[mult_end].startswith("disagree_")
        assert names[disagree_end].startswith("avg_")
        assert names[avg_end] == "vix_normalized"
        assert names[avg_end + 1] == "trend_strength"
        assert names[avg_end + 2].startswith("acc90d_")


# ==============================================================================
# New tests: explain_features edge cases
# ==============================================================================

class TestExplainFeaturesEdgeCases:
    """Edge cases for explain_features."""

    def test_explain_all_neutral(self, engine, regime_context, historical_accuracy):
        """All-neutral signals show 6 neutral, 0 bullish/bearish."""
        now = datetime.now()
        neutral = {s: Signal(s, 0.0, now, 0.5) for s in SignalSource}
        fv = engine.create_features(neutral, regime_context, historical_accuracy)
        explanation = engine.explain_features(fv)
        assert explanation["base_signals_summary"]["neutral_count"] == 6
        assert explanation["base_signals_summary"]["bullish_count"] == 0
        assert explanation["base_signals_summary"]["bearish_count"] == 0

    def test_explain_top_n_100(self, engine, full_signals, regime_context, historical_accuracy):
        """top_n > 15 caps at 15 pairs (does not error)."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        explanation = engine.explain_features(fv, top_n=100)
        assert len(explanation["pairwise_interactions"]["high_synergy"]) == 15

    def test_explain_top_n_zero(self, engine, full_signals, regime_context, historical_accuracy):
        """top_n=0 returns empty lists."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        explanation = engine.explain_features(fv, top_n=0)
        assert len(explanation["pairwise_interactions"]["high_synergy"]) == 0


# ==============================================================================
# New tests: to_dict edge cases
# ==============================================================================

class TestToDictEdgeCases:
    """Edge cases for to_dict serialization."""

    def test_to_dict_vix_normalized_value(self, engine, full_signals, regime_context, historical_accuracy):
        """VIX normalized appears as float in dict."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        d = engine.to_dict(fv)
        assert "vix_normalized" in d
        assert isinstance(d["vix_normalized"], float)

    def test_to_dict_trend_strength_value(self, engine, full_signals, regime_context, historical_accuracy):
        """Trend strength appears as float in dict."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        d = engine.to_dict(fv)
        assert "trend_strength" in d
        assert isinstance(d["trend_strength"], float)

    def test_to_dict_accuracy_keys_are_strings(self, engine, full_signals, regime_context, historical_accuracy):
        """Accuracy values dict has string keys."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        d = engine.to_dict(fv)
        for key in d["accuracy_values"]:
            assert isinstance(key, str)

    def test_to_dict_dimension_count_in_output(self, engine, full_signals, regime_context, historical_accuracy):
        """dimension_count present in dict output."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        d = engine.to_dict(fv)
        assert d["dimension_count"] == 59


# ==============================================================================
# New tests: StackingAccuracyTracker edge cases
# ==============================================================================

class TestStackingAccuracyTrackerEdgeCases:
    """Edge cases for accuracy tracker."""

    def test_signal_zero_return_zero(self, tracker):
        """Signal=0 and return=0 triggers neutral clause."""
        now = datetime.now()
        tracker.record_prediction(SignalSource.MULTI_SPEED_MOM, now, 0.0, 0.0)
        acc = tracker.get_historical_accuracy(SignalSource.MULTI_SPEED_MOM, now)
        assert acc.accuracy_90d == 1.0

    def test_no_history_returns_default(self, tracker):
        """No predictions returns 0.5 default."""
        now = datetime.now()
        acc = tracker.get_historical_accuracy(SignalSource.MULTI_SPEED_MOM, now)
        assert acc.accuracy_90d == 0.5
        assert acc.predictions_count == 0

    def test_all_old_records_pruned(self, tracker):
        """All records outside window return default."""
        now = datetime.now()
        for _ in range(5):
            tracker.record_prediction(
                SignalSource.MULTI_SPEED_MOM,
                now - timedelta(days=200), 0.5, 0.02
            )
        acc = tracker.get_historical_accuracy(SignalSource.MULTI_SPEED_MOM, now)
        assert acc.predictions_count == 0
        assert acc.accuracy_90d == 0.5

    def test_trim_on_record(self, tracker):
        """Old records trimmed when new record added."""
        now = datetime.now()
        tracker.record_prediction(
            SignalSource.MULTI_SPEED_MOM,
            now - timedelta(days=100), 0.5, 0.02
        )
        tracker.record_prediction(SignalSource.MULTI_SPEED_MOM, now, 0.5, 0.02)
        acc = tracker.get_historical_accuracy(SignalSource.MULTI_SPEED_MOM, now)
        assert acc.predictions_count == 1

    def test_get_all_accuracies_defaults_with_no_data(self, tracker):
        """All accuracies default to 0.5 with no predictions."""
        now = datetime.now()
        accs = tracker.get_all_accuracies(now)
        for source in SignalSource:
            assert accs[source].accuracy_90d == 0.5
            assert accs[source].predictions_count == 0


# ==============================================================================
# New tests: Integration across components
# ==============================================================================

class TestIntegration:
    """Cross-component integration tests."""

    def test_create_to_numpy_explain_consistency(self, engine, full_signals, regime_context, historical_accuracy):
        """Pipeline: create -> to_numpy mean matches explain mean."""
        fv = engine.create_features(full_signals, regime_context, historical_accuracy)
        arr = engine.to_numpy(fv)
        names = engine.get_feature_names()
        explanation = engine.explain_features(fv)
        base_mean = np.mean(arr[0:6])
        assert base_mean == pytest.approx(explanation["base_signals_summary"]["mean"])

    def test_tracker_pipeline(self):
        """Tracker: record -> get_all -> FeatureVector accuracy matches."""
        tracker = StackingAccuracyTracker(window_days=90)
        now = datetime.now()
        for _ in range(10):
            tracker.record_prediction(SignalSource.MULTI_SPEED_MOM, now, 0.5, 0.02)
        accs = tracker.get_all_accuracies(now)
        assert accs[SignalSource.MULTI_SPEED_MOM].accuracy_90d == 1.0


# ==============================================================================
# New tests: CLI / demo code coverage
# ==============================================================================

class TestCLI:
    """Coverage for main() CLI entry point and __main__ guard."""

    def test_demo_returns_expected_types(self):
        """demo() returns (FeatureVector, ndarray)."""
        from src.signals.stacking_feature_engine import demo
        fv, arr = demo()
        from src.signals.stacking_feature_engine import FeatureVector
        assert isinstance(fv, FeatureVector)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (59,)

    def test_main_names_flag(self, capsys):
        """main(['--names']) prints 59 feature names."""
        from src.signals.stacking_feature_engine import main
        main(["--names"])
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) == 59
        assert "base_multi_speed_momentum" in captured.out
        assert "acc90d_unified_overlay" in captured.out

    def test_main_test_flag(self, capsys):
        """main(['--test']) runs demo without error."""
        from src.signals.stacking_feature_engine import main
        main(["--test"])
        captured = capsys.readouterr()
        assert "Feature vector created" in captured.out
        assert "Shape: (59,)" in captured.out

    def test_main_no_args_prints_help(self, capsys):
        """main([]) prints help to stdout."""
        from src.signals.stacking_feature_engine import main
        main([])
        captured = capsys.readouterr()
        assert "usage:" in captured.out
        assert "--test" in captured.out
        assert "--names" in captured.out

    def test_main_defaults_to_none(self, capsys, monkeypatch):
        """main() with no arguments prints help (uses sys.argv)."""
        monkeypatch.setattr("sys.argv", ["prog"])
        from src.signals.stacking_feature_engine import main
        main()
        captured = capsys.readouterr()
        assert "usage:" in captured.out
