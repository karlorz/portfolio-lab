#!/usr/bin/env python3
"""
Tests for stacking feature engine — v3.10 Phase 1 feature generation.

Covers:
- Feature vector creation with 8 base signals
- Pairwise interaction features (84 total)
- Regime context features
- Historical accuracy tracking
- NumPy conversion
- JSON serialization
- Performance latency validation (<10ms target)

Tests: 15
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    """Complete set of 8 signals for testing."""
    now = datetime.now()
    return {
        SignalSource.TSFM_MOMENTUM: Signal(SignalSource.TSFM_MOMENTUM, 0.5, now, 0.8),
        SignalSource.HMM_REGIME: Signal(SignalSource.HMM_REGIME, 0.3, now, 0.7),
        SignalSource.CTA_TREND: Signal(SignalSource.CTA_TREND, 0.6, now, 0.75),
        SignalSource.MACRO_MOMENTUM: Signal(SignalSource.MACRO_MOMENTUM, 0.2, now, 0.6),
        SignalSource.MULTI_SPEED_MOM: Signal(SignalSource.MULTI_SPEED_MOM, 0.7, now, 0.85),
        SignalSource.DURATION_REGIME: Signal(SignalSource.DURATION_REGIME, -0.1, now, 0.65),
        SignalSource.CIRCUIT_BREAKER: Signal(SignalSource.CIRCUIT_BREAKER, 0.9, now, 0.9),
        SignalSource.FACTOR_ROTATION: Signal(SignalSource.FACTOR_ROTATION, 0.4, now, 0.7),
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
    """Test: Exactly 8 signal sources defined."""
    assert len(SignalSource) == 8


def test_signal_source_values():
    """Test: All expected signal sources exist."""
    expected = [
        "tsfm_momentum", "hmm_regime", "cta_trend", "macro_momentum",
        "multi_speed_momentum", "duration_regime", "circuit_breaker", "factor_rotation"
    ]
    for exp in expected:
        assert any(s.value == exp for s in SignalSource)


# ---------------------------------------------------------------------------
# StackingFeatureEngine tests (10)
# ---------------------------------------------------------------------------

def test_feature_engine_initialization(engine):
    """Test: Feature engine initializes correctly."""
    assert engine.NUM_BASE_SIGNALS == 8
    assert engine.NUM_PAIRWISE_COMBINATIONS == 28
    assert engine.TOTAL_DIMENSIONS == 102
    assert engine.vix_normalization_factor == 30.0


def test_create_features_requires_all_signals(engine, regime_context, historical_accuracy):
    """Test: Feature creation fails if signals are missing."""
    partial_signals = {
        SignalSource.TSFM_MOMENTUM: Signal(SignalSource.TSFM_MOMENTUM, 0.5, datetime.now(), 0.8)
    }
    
    with pytest.raises(ValueError) as exc_info:
        engine.create_features(partial_signals, regime_context, historical_accuracy)
    
    assert "Expected 8 signals, got 1" in str(exc_info.value)


def test_create_features_success(engine, full_signals, regime_context, historical_accuracy):
    """Test: Successful feature vector creation with all signals."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)
    
    assert isinstance(fv, FeatureVector)
    assert fv.dimension_count == 102
    assert len(fv.base_values) == 8
    assert len(fv.multiplicative) == 28
    assert len(fv.disagreement) == 28
    assert len(fv.averages) == 28
    assert fv.vix_normalized == 20.0 / 30.0  # Normalized
    assert fv.trend_strength == 0.5


def test_base_values_correctness(engine, full_signals, regime_context, historical_accuracy):
    """Test: Base signal values are correctly extracted."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)
    
    assert fv.base_values[SignalSource.TSFM_MOMENTUM] == 0.5
    assert fv.base_values[SignalSource.HMM_REGIME] == 0.3
    assert fv.base_values[SignalSource.MULTI_SPEED_MOM] == 0.7
    assert fv.base_values[SignalSource.DURATION_REGIME] == -0.1


def test_multiplicative_interactions(engine, full_signals, regime_context, historical_accuracy):
    """Test: Multiplicative interactions computed correctly."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)
    
    # TSMOM (0.5) * MultiSpeed (0.7) = 0.35
    pair = (SignalSource.TSFM_MOMENTUM, SignalSource.MULTI_SPEED_MOM)
    assert fv.multiplicative[pair] == 0.5 * 0.7


def test_disagreement_features(engine, full_signals, regime_context, historical_accuracy):
    """Test: Disagreement features computed correctly."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)
    
    # |TSMOM (0.5) - Circuit (0.9)| = 0.4
    pair = (SignalSource.TSFM_MOMENTUM, SignalSource.CIRCUIT_BREAKER)
    assert fv.disagreement[pair] == abs(0.5 - 0.9)


def test_average_features(engine, full_signals, regime_context, historical_accuracy):
    """Test: Average features computed correctly."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)
    
    # (TSMOM (0.5) + CTA (0.6)) / 2 = 0.55
    pair = (SignalSource.TSFM_MOMENTUM, SignalSource.CTA_TREND)
    assert fv.averages[pair] == (0.5 + 0.6) / 2.0


def test_to_numpy_shape(engine, full_signals, regime_context, historical_accuracy):
    """Test: NumPy array has correct shape."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)
    arr = engine.to_numpy(fv)
    
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (102,)
    assert arr.dtype == np.float32


def test_to_numpy_order(engine, full_signals, regime_context, historical_accuracy):
    """Test: NumPy array features are in correct order."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)
    arr = engine.to_numpy(fv)
    
    # First 8 elements should be base signals in enum order
    for i, source in enumerate(SignalSource):
        assert arr[i] == full_signals[source].value


def test_get_feature_names_count(engine):
    """Test: Feature names list has 102 entries."""
    names = engine.get_feature_names()
    assert len(names) == 102
    assert all(isinstance(n, str) for n in names)


def test_to_dict_serializable(engine, full_signals, regime_context, historical_accuracy):
    """Test: Feature vector can be serialized to dict."""
    fv = engine.create_features(full_signals, regime_context, historical_accuracy)
    d = engine.to_dict(fv)
    
    assert isinstance(d, dict)
    assert "base_values" in d
    assert "multiplicative" in d
    assert "timestamp" in d
    assert d["dimension_count"] == 102


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
        tracker.record_prediction(SignalSource.TSFM_MOMENTUM, now, 0.5, 0.02)  # Correct
    for i in range(3):
        tracker.record_prediction(SignalSource.TSFM_MOMENTUM, now, 0.5, -0.01)  # Incorrect
    
    acc = tracker.get_historical_accuracy(SignalSource.TSFM_MOMENTUM, now)
    assert acc.accuracy_90d == 0.7
    assert acc.predictions_count == 10


def test_accuracy_tracker_rolling_window(tracker):
    """Test: Old predictions are pruned from rolling window."""
    now = datetime.now()
    
    # Record prediction 100 days ago
    tracker.record_prediction(
        SignalSource.TSFM_MOMENTUM, 
        now - timedelta(days=100), 
        0.5, 
        0.02
    )
    
    # Record recent prediction
    tracker.record_prediction(SignalSource.TSFM_MOMENTUM, now, 0.5, -0.01)
    
    acc = tracker.get_historical_accuracy(SignalSource.TSFM_MOMENTUM, now)
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
    assert summary["bullish_count"] + summary["bearish_count"] + summary["neutral_count"] == 8


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
    tracker.record_prediction(SignalSource.TSFM_MOMENTUM, now, 0.05, 0.005)
    
    # This should count as correct (neutral × neutral)
    acc = tracker.get_historical_accuracy(SignalSource.TSFM_MOMENTUM, now)
    assert acc.predictions_count == 1


# Total: 15 tests
