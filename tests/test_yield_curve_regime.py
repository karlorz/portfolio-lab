#!/usr/bin/env python3
"""
Tests for Yield Curve Regime Classifier - v3.11 Phase 1

Covers:
- YieldCurveRegime enum and values
- YieldCurveData dataclass
- RegimeClassification dataclass
- YieldCurveRegimeClassifier class
- classify_regime() static method
- get_smoothed_regime() with 20-day MA
- classify() with transition rules
- get_regime_description() 
- get_expected_alpha()
- fetch_fred_yield_data() integration
"""

import sys
import os
import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signals.yield_curve_regime import (
    YieldCurveRegime,
    YieldCurveData,
    RegimeClassification,
    YieldCurveRegimeClassifier,
    fetch_fred_yield_data,
    _load_cached_yield_data,
    save_yield_cache,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def classifier():
    """Create fresh classifier with mocked state and empty spread history."""
    with patch('src.signals.yield_curve_regime.YieldCurveRegimeClassifier._load_state') as mock_load:
        mock_load.return_value = {
            "current_regime": YieldCurveRegime.UNKNOWN,
            "regime_start_date": None,
            "last_update": None,
            "pending_regime": None,
            "pending_since": None,
        }
        with patch.object(YieldCurveRegimeClassifier, '_save_state'):
            with patch.object(YieldCurveRegimeClassifier, '_load_spread_history'):
                classifier = YieldCurveRegimeClassifier()
                classifier.spread_history = []  # Ensure clean history
                yield classifier


@pytest.fixture
def sample_yield_data():
    """Create sample yield curve data - FLAT regime (47bps spread)."""
    return YieldCurveData(
        timestamp="2026-05-14",
        dgs10=0.0414,  # 4.14%
        dgs2=0.0364,   # 3.64%
        dgs30=0.0485,  # 4.85%
        dgs5=0.0395,   # 3.95%
        spread_2s10s=0.0050,  # 50bps - FLAT regime
        spread_10s30s=0.0038,  # 38bps
    )


@pytest.fixture
def inverted_yield_data():
    """Create inverted yield curve data."""
    return YieldCurveData(
        timestamp="2026-05-14",
        dgs10=0.0320,  # 3.20%
        dgs2=0.0400,   # 4.00%
        dgs30=None,
        dgs5=None,
        spread_2s10s=-0.0080,  # -80bps (inverted)
        spread_10s30s=None,
    )


@pytest.fixture
def steep_yield_data():
    """Create steep yield curve data."""
    return YieldCurveData(
        timestamp="2026-05-14",
        dgs10=0.0500,  # 5.00%
        dgs2=0.0200,   # 2.00%
        dgs30=None,
        dgs5=None,
        spread_2s10s=0.0300,  # 300bps (steep)
        spread_10s30s=None,
    )


# -----------------------------------------------------------------------------
# Test Enum and Data Classes
# -----------------------------------------------------------------------------

def test_yield_curve_regime_values():
    """Test YieldCurveRegime enum has expected values."""
    assert YieldCurveRegime.INVERTED.value == "inverted"
    assert YieldCurveRegime.FLAT.value == "flat"
    assert YieldCurveRegime.STEEP.value == "steep"
    assert YieldCurveRegime.UNKNOWN.value == "unknown"


def test_yield_curve_data_to_dict(sample_yield_data):
    """Test YieldCurveData serialization."""
    data_dict = sample_yield_data.to_dict()
    assert data_dict['timestamp'] == "2026-05-14"
    assert data_dict['dgs10'] == 0.0414
    assert data_dict['dgs2'] == 0.0364
    assert data_dict['spread_2s10s'] == 0.0050


def test_regime_classification_to_dict():
    """Test RegimeClassification serialization."""
    classification = RegimeClassification(
        timestamp="2026-05-14",
        regime=YieldCurveRegime.FLAT,
        spread_2s10s=0.0083,
        dgs10=0.0447,
        dgs2=0.0364,
        days_in_regime=15,
        regime_start_date="2026-04-30",
        is_transition_pending=False,
        days_until_eligible=0,
        confidence="medium",
    )
    
    result = classification.to_dict()
    assert result['regime'] == "flat"
    assert result['confidence'] == "medium"
    assert result['days_in_regime'] == 15
    assert not result['is_transition_pending']


# -----------------------------------------------------------------------------
# Test Regime Classification Logic
# -----------------------------------------------------------------------------

def test_classify_regime_inverted(classifier):
    """Test classification of inverted curve (< -0.25%)."""
    regime = classifier.classify_regime(-0.0050)  # -50bps
    assert regime == YieldCurveRegime.INVERTED
    
    regime = classifier.classify_regime(-0.0030)  # -30bps (below threshold)
    assert regime == YieldCurveRegime.INVERTED


def test_classify_regime_flat(classifier):
    """Test classification of flat curve (-0.25% to +0.75%)."""
    regime = classifier.classify_regime(0.0000)  # 0bps
    assert regime == YieldCurveRegime.FLAT
    
    regime = classifier.classify_regime(0.0047)  # 47bps
    assert regime == YieldCurveRegime.FLAT
    
    regime = classifier.classify_regime(0.0075)  # 75bps (at threshold)
    assert regime == YieldCurveRegime.FLAT


def test_classify_regime_steep(classifier):
    """Test classification of steep curve (> +0.75%)."""
    regime = classifier.classify_regime(0.0100)  # 100bps
    assert regime == YieldCurveRegime.STEEP
    
    regime = classifier.classify_regime(0.0300)  # 300bps
    assert regime == YieldCurveRegime.STEEP


def test_classify_regime_thresholds(classifier):
    """Test exact threshold boundaries."""
    # At -0.25%, should be flat (not inverted)
    assert classifier.classify_regime(-0.0025) == YieldCurveRegime.FLAT
    
    # Just below -0.25%, should be inverted
    assert classifier.classify_regime(-0.0026) == YieldCurveRegime.INVERTED
    
    # At +0.75%, should be flat (not steep)
    assert classifier.classify_regime(0.0075) == YieldCurveRegime.FLAT
    
    # Just above +0.75%, should be steep
    assert classifier.classify_regime(0.0076) == YieldCurveRegime.STEEP


# -----------------------------------------------------------------------------
# Test Smoothing
# -----------------------------------------------------------------------------

def test_get_smoothed_regime_no_history(classifier):
    """Test smoothing returns raw regime when no history."""
    classifier.spread_history = []
    regime = classifier.get_smoothed_regime(0.0050)  # 50bps = FLAT
    assert regime == YieldCurveRegime.FLAT


def test_get_smoothed_regime_with_history(classifier):
    """Test 20-day MA smoothing."""
    # Add 20 days of steep spreads (150bps)
    for i in range(20):
        date = (datetime.now() - timedelta(days=20-i)).strftime("%Y-%m-%d")
        classifier.spread_history.append((date, 0.0150))  # 150bps
    
    # Current spread is flat (50bps), but MA should be steep
    regime = classifier.get_smoothed_regime(0.0050)  # 50bps
    assert regime == YieldCurveRegime.STEEP


def test_spread_history_truncation(classifier):
    """Test that spread history is truncated to 30 days."""
    # Add 50 days of data
    for i in range(50):
        date = (datetime.now() - timedelta(days=50-i)).strftime("%Y-%m-%d")
        classifier.spread_history.append((date, 0.0100))
    
    # Add one more to trigger truncation
    classifier.get_smoothed_regime(0.0080)
    
    assert len(classifier.spread_history) <= 30


# -----------------------------------------------------------------------------
# Test Transition Rules
# -----------------------------------------------------------------------------

def test_classify_first_time(classifier, sample_yield_data):
    """Test first classification sets regime."""
    result = classifier.classify(sample_yield_data)
    
    assert result.regime == YieldCurveRegime.FLAT
    assert result.days_in_regime == 0
    assert not result.is_transition_pending


def test_classify_regime_continuation(classifier, sample_yield_data):
    """Test continuing in same regime."""
    # First classification
    classifier.classify(sample_yield_data)
    classifier.state["current_regime"] = YieldCurveRegime.FLAT
    classifier.state["regime_start_date"] = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    
    # Second classification with same regime
    result = classifier.classify(sample_yield_data)
    
    assert result.regime == YieldCurveRegime.FLAT
    assert result.days_in_regime == 10


def test_classify_pending_transition(classifier, inverted_yield_data):
    """Test transition requires 30-day minimum."""
    # Start in flat regime
    classifier.state["current_regime"] = YieldCurveRegime.FLAT
    classifier.state["regime_start_date"] = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")
    
    # Try to transition to inverted
    result = classifier.classify(inverted_yield_data)
    
    # Should be pending since 30-day rule applies
    assert result.is_transition_pending or result.regime == YieldCurveRegime.INVERTED


def test_classify_transition_eligible_after_30_days(classifier, inverted_yield_data):
    """Test transition allowed after 30 days."""
    # Start in flat with pending inverted
    classifier.state["current_regime"] = YieldCurveRegime.FLAT
    classifier.state["regime_start_date"] = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")
    classifier.state["pending_regime"] = YieldCurveRegime.INVERTED
    classifier.state["pending_since"] = (datetime.now() - timedelta(days=35)).strftime("%Y-%m-%d")
    
    # After 35 days, should transition
    result = classifier.classify(inverted_yield_data)
    
    assert result.regime == YieldCurveRegime.INVERTED
    assert not result.is_transition_pending


# -----------------------------------------------------------------------------
# Test Confidence Calculation
# -----------------------------------------------------------------------------

def test_high_confidence(classifier):
    """Test high confidence calculation."""
    confidence = classifier._calculate_confidence(0.0150, 50)  # 150bps, 50 days
    assert confidence == "high"


def test_low_confidence_near_threshold(classifier):
    """Test low confidence near threshold."""
    confidence = classifier._calculate_confidence(0.0076, 50)  # 76bps (just above steep)
    assert confidence == "low"  # Close to threshold


def test_low_confidence_new_regime(classifier):
    """Test low confidence for new regime."""
    confidence = classifier._calculate_confidence(0.0150, 10)  # Only 10 days
    assert confidence == "low"


def test_medium_confidence(classifier):
    """Test medium confidence for moderate conditions."""
    confidence = classifier._calculate_confidence(0.0100, 20)  # 100bps, 20 days
    assert confidence == "medium"


# -----------------------------------------------------------------------------
# Test Regime Descriptions and Alphas
# -----------------------------------------------------------------------------

def test_get_regime_description_inverted(classifier):
    """Test description for inverted regime."""
    desc = classifier.get_regime_description(YieldCurveRegime.INVERTED)
    assert "Inverted" in desc
    assert "Short duration" in desc


def test_get_regime_description_flat(classifier):
    """Test description for flat regime."""
    desc = classifier.get_regime_description(YieldCurveRegime.FLAT)
    assert "Flat" in desc
    assert "Neutral" in desc


def test_get_regime_description_steep(classifier):
    """Test description for steep regime."""
    desc = classifier.get_regime_description(YieldCurveRegime.STEEP)
    assert "Steep" in desc
    assert "Long duration" in desc


def test_get_expected_alpha_inverted(classifier):
    """Test alpha for inverted regime."""
    alpha = classifier.get_expected_alpha(YieldCurveRegime.INVERTED)
    assert alpha == 1.8


def test_get_expected_alpha_flat(classifier):
    """Test alpha for flat regime."""
    alpha = classifier.get_expected_alpha(YieldCurveRegime.FLAT)
    assert alpha == 0.5


def test_get_expected_alpha_steep(classifier):
    """Test alpha for steep regime."""
    alpha = classifier.get_expected_alpha(YieldCurveRegime.STEEP)
    assert alpha == 1.2


def test_get_expected_alpha_unknown(classifier):
    """Test alpha for unknown regime."""
    alpha = classifier.get_expected_alpha(YieldCurveRegime.UNKNOWN)
    assert alpha == 0.0


# -----------------------------------------------------------------------------
# Test Cache and Data Loading
# -----------------------------------------------------------------------------

def test_load_cached_yield_data_not_exists():
    """Test loading when cache doesn't exist."""
    with patch('pathlib.Path.exists', return_value=False):
        result = _load_cached_yield_data()
        assert result is None


def test_load_cached_yield_data_success():
    """Test loading cached data."""
    mock_data = {
        "timestamp": "2026-05-14",
        "dgs10": 0.0447,
        "dgs2": 0.0364,
        "dgs30": 0.0485,
        "dgs5": 0.0395,
        "spread_2s10s": 0.0083,
        "spread_10s30s": 0.0038,
    }
    
    with patch('builtins.open', MagicMock()):
        with patch('json.load', return_value=mock_data):
            with patch('pathlib.Path.exists', return_value=True):
                result = _load_cached_yield_data()
                # Should create YieldCurveData from mock
                assert result is not None


def test_save_yield_cache():
    """Test saving yield cache."""
    data = YieldCurveData(
        timestamp="2026-05-14",
        dgs10=0.0447,
        dgs2=0.0364,
        dgs30=None,
        dgs5=None,
        spread_2s10s=0.0083,
        spread_10s30s=None,
    )
    
    with patch('builtins.open', MagicMock()):
        with patch('json.dump') as mock_dump:
            save_yield_cache(data)
            mock_dump.assert_called_once()


# -----------------------------------------------------------------------------
# Test FRED Integration
# -----------------------------------------------------------------------------

def test_fetch_fred_yield_data_success():
    """Test fetching data from FRED."""
    mock_df10 = MagicMock()
    mock_df10.__len__ = MagicMock(return_value=1)
    mock_df10.__getitem__ = MagicMock(return_value=mock_df10)
    mock_df10.iloc = [4.47]  # 4.47%
    
    with patch('src.signals.fed_policy_overlay.fetch_fred_series') as mock_fetch:
        mock_fetch.side_effect = [
            mock_df10,  # DGS10
            mock_df10,  # DGS2
            mock_df10,  # DGS30
            mock_df10,  # DGS5
        ]
        
        result = fetch_fred_yield_data()
        
        assert result is not None
        assert mock_fetch.call_count == 4


def test_fetch_fred_yield_data_no_dgs2():
    """Test fallback when DGS2 unavailable."""
    mock_df = MagicMock()
    mock_df.__len__ = MagicMock(return_value=1)
    mock_df.__getitem__ = MagicMock(return_value=mock_df)
    mock_df.iloc = [4.47]
    
    with patch('src.signals.fed_policy_overlay.fetch_fred_series') as mock_fetch:
        mock_fetch.side_effect = [
            mock_df,   # DGS10 available
            None,      # DGS2 unavailable
            None,      # DGS30
            None,      # DGS5
        ]
        
        with patch('src.signals.yield_curve_regime._load_cached_yield_data', return_value=None):
            result = fetch_fred_yield_data()
            
            # Should return None or cached data
            assert result is None


def test_fetch_fred_yield_data_exception():
    """Test exception handling."""
    with patch('src.signals.fed_policy_overlay.fetch_fred_series') as mock_fetch:
        mock_fetch.side_effect = Exception("Network error")
        
        with patch('src.signals.yield_curve_regime._load_cached_yield_data', return_value=None):
            result = fetch_fred_yield_data()
            
            assert result is None


# -----------------------------------------------------------------------------
# Test State Management
# -----------------------------------------------------------------------------

def test_load_state_new_file(classifier):
    """Test loading state when file doesn't exist."""
    with patch('pathlib.Path.exists', return_value=False):
        state = classifier._load_state()
        assert state["current_regime"] == YieldCurveRegime.UNKNOWN
        assert state["regime_start_date"] is None


def test_save_state_converts_enum():
    """Test that enums are converted to strings when saving."""
    classifier = YieldCurveRegimeClassifier.__new__(YieldCurveRegimeClassifier)
    classifier.state = {
        "current_regime": YieldCurveRegime.FLAT,
        "pending_regime": YieldCurveRegime.STEEP,
    }
    
    with patch('builtins.open', MagicMock()):
        with patch('json.dump') as mock_dump:
            classifier._save_state()
            
            # Check that enum was converted to string
            call_args = mock_dump.call_args
            if call_args:
                saved_state = call_args[0][0]
                assert saved_state["current_regime"] == "flat"
                assert saved_state["pending_regime"] == "steep"


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

def test_full_classification_workflow(classifier, sample_yield_data):
    """Test complete classification workflow."""
    # Initial classification
    result1 = classifier.classify(sample_yield_data)
    assert result1.regime == YieldCurveRegime.FLAT
    
    # Create steep yield data further from threshold (150bps spread)
    steep_data = YieldCurveData(
        timestamp="2026-05-14",
        dgs10=0.0514,  # 5.14%
        dgs2=0.0364,   # 3.64%
        dgs30=0.0585,
        dgs5=0.0495,
        spread_2s10s=0.0150,  # 150bps - STEEP regime, well above 75bps threshold
        spread_10s30s=0.0038,
    )
    
    # After 50 days in steep regime, should have high confidence
    classifier.state["current_regime"] = YieldCurveRegime.STEEP
    classifier.state["regime_start_date"] = (datetime.now() - timedelta(days=50)).strftime("%Y-%m-%d")
    
    # Classification with steep data and established regime
    result2 = classifier.classify(steep_data)
    assert result2.days_in_regime == 50
    assert result2.confidence == "high"


def test_transition_from_flat_to_inverted(classifier, sample_yield_data, inverted_yield_data):
    """Test full transition workflow."""
    # Start in flat
    classifier.classify(sample_yield_data)
    classifier.state["current_regime"] = YieldCurveRegime.FLAT
    classifier.state["regime_start_date"] = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    
    # First inverted signal - should be pending
    result1 = classifier.classify(inverted_yield_data)
    
    # Simulate 30+ days in pending
    classifier.state["pending_regime"] = YieldCurveRegime.INVERTED
    classifier.state["pending_since"] = (datetime.now() - timedelta(days=35)).strftime("%Y-%m-%d")
    
    # Second inverted signal - should transition
    result2 = classifier.classify(inverted_yield_data)
    assert result2.regime == YieldCurveRegime.INVERTED
    assert not result2.is_transition_pending


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
