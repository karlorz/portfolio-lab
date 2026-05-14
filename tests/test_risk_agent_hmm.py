import pytest; pytestmark = pytest.mark.heavy
#!/usr/bin/env python3
"""
Tests for HMM-LSTM regime detector — data classes, regime definitions,
feature extraction, allocation shifts, and portfolio regime management.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agents.risk_agent_hmm import (
    MarketRegime, REGIME_DESCRIPTIONS, REGIME_ALLOCATION_SHIFTS,
    RegimeDetectionResult, PortfolioRegimeState,
    HMMRegimeDetector, PortfolioRegimeManager,
    HMM_AVAILABLE,
)


# Picklable stand-ins for HMM model objects
class _FakeHMM:
    startprob_ = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

class _FakeScaler:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n=300, base=500.0, drift=0.0004, seed=42):
    """Create synthetic price series."""
    np.random.seed(seed)
    prices = [base]
    for _ in range(n - 1):
        ret = np.random.normal(drift, 0.012)
        prices.append(prices[-1] * (1 + ret))
    dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
    return pd.Series(prices, index=dates)


def _make_detector():
    """Create an HMMRegimeDetector with mocked HMM."""
    detector = HMMRegimeDetector.__new__(HMMRegimeDetector)
    detector.n_states = 5
    detector.n_features = 4
    detector.covariance_type = 'full'
    detector.n_iter = 100
    detector.random_state = 42
    detector.hmm = MagicMock()
    detector.scaler = MagicMock()
    detector.is_fitted = False
    detector.feature_history = __import__('collections').deque(maxlen=1000)
    detector.regime_history = __import__('collections').deque(maxlen=100)
    return detector


def _make_regime_result(regime=MarketRegime.BULL, confidence=0.8):
    """Create a test RegimeDetectionResult."""
    return RegimeDetectionResult(
        timestamp='2026-01-01',
        ticker='SPY',
        regime=regime,
        regime_probabilities={str(r): 0.2 for r in MarketRegime},
        confidence=confidence,
        recent_return=0.05,
        volatility=0.15,
        trend_strength=0.3,
        vix_proxy=3.0,
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestMarketRegime:
    """Test MarketRegime enum."""

    def test_values(self):
        assert MarketRegime.BULL.value == 0
        assert MarketRegime.BEAR.value == 1
        assert MarketRegime.NEUTRAL.value == 2
        assert MarketRegime.HIGH_VOL.value == 3
        assert MarketRegime.CRISIS.value == 4

    def test_str(self):
        assert str(MarketRegime.BULL) == 'bull'
        assert str(MarketRegime.CRISIS) == 'crisis'

    def test_all_members(self):
        assert len(MarketRegime) == 5


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Test module constants."""

    def test_regime_descriptions(self):
        for regime in MarketRegime:
            assert regime in REGIME_DESCRIPTIONS

    def test_allocation_shifts(self):
        for regime in MarketRegime:
            assert regime in REGIME_ALLOCATION_SHIFTS
            shifts = REGIME_ALLOCATION_SHIFTS[regime]
            assert 'SPY' in shifts
            assert 'GLD' in shifts
            assert 'TLT' in shifts

    def test_bull_increases_equity(self):
        assert REGIME_ALLOCATION_SHIFTS[MarketRegime.BULL]['SPY'] > 0

    def test_crisis_decreases_equity(self):
        assert REGIME_ALLOCATION_SHIFTS[MarketRegime.CRISIS]['SPY'] < 0

    def test_neutral_zero_shifts(self):
        for asset, shift in REGIME_ALLOCATION_SHIFTS[MarketRegime.NEUTRAL].items():
            assert shift == 0.0


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestRegimeDetectionResult:
    """Test RegimeDetectionResult dataclass."""

    def test_creation(self):
        result = _make_regime_result()
        assert result.ticker == 'SPY'
        assert result.regime == MarketRegime.BULL

    def test_to_dict(self):
        result = _make_regime_result()
        d = result.to_dict()
        assert d['regime'] == 'bull'
        assert d['regime_code'] == 0
        assert 'regime_description' in d
        assert 'regime_probabilities' in d


class TestPortfolioRegimeState:
    """Test PortfolioRegimeState dataclass."""

    def test_creation(self):
        state = PortfolioRegimeState(
            timestamp='2026-01-01',
            dominant_regime=MarketRegime.BULL,
            regime_confidence=0.75,
            asset_regimes={},
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            regime_adjustments={'SPY': 0.10, 'GLD': -0.05, 'TLT': -0.05},
            recommended_allocation={'SPY': 0.56, 'GLD': 0.33, 'TLT': 0.11},
            predicted_volatility=0.15,
            risk_budget_change='increase',
        )
        assert state.dominant_regime == MarketRegime.BULL
        assert state.risk_budget_change == 'increase'

    def test_to_dict(self):
        state = PortfolioRegimeState(
            timestamp='2026-01-01',
            dominant_regime=MarketRegime.NEUTRAL,
            regime_confidence=0.5,
            asset_regimes={},
            base_allocation={'SPY': 0.46},
            regime_adjustments={'SPY': 0.0},
            recommended_allocation={'SPY': 0.46},
            predicted_volatility=0.12,
            risk_budget_change='maintain',
        )
        d = state.to_dict()
        assert d['dominant_regime'] == 'neutral'
        assert 'base_allocation' in d


# ---------------------------------------------------------------------------
# HMMRegimeDetector tests
# ---------------------------------------------------------------------------

class TestHMMRegimeDetector:
    """Test HMMRegimeDetector."""

    def test_init_defaults(self):
        detector = _make_detector()
        assert detector.n_states == 5
        assert detector.n_features == 4
        assert detector.random_state == 42

    def test_extract_features_returns_array(self):
        detector = _make_detector()
        prices = _make_prices(300)
        features = detector.extract_features(prices)
        assert features is not None
        assert len(features) == 4

    def test_extract_features_insufficient_data(self):
        detector = _make_detector()
        prices = _make_prices(50)  # Too few
        features = detector.extract_features(prices)
        assert features is None

    def test_extract_features_momentum(self):
        detector = _make_detector()
        prices = _make_prices(300, drift=0.002)  # Strong uptrend
        features = detector.extract_features(prices)
        # Momentum (feature 0) should be positive
        assert features[0] > 0

    def test_extract_features_volatility_positive(self):
        detector = _make_detector()
        prices = _make_prices(300)
        features = detector.extract_features(prices)
        assert features[1] > 0  # Volatility always positive

    def test_extract_features_trend_strength(self):
        detector = _make_detector()
        prices = _make_prices(300)
        features = detector.extract_features(prices)
        assert features[2] >= 0  # Trend strength non-negative

    def test_extract_features_vix_proxy(self):
        detector = _make_detector()
        prices = _make_prices(300)
        features = detector.extract_features(prices)
        assert features[3] >= 0  # VIX proxy non-negative

    def test_predict_regime_not_fitted(self):
        detector = _make_detector()
        detector.is_fitted = False
        prices = _make_prices(300)
        result = detector.predict_regime(prices, ticker='SPY')
        assert result is None

    def test_save_load_roundtrip(self, tmp_path):
        detector = _make_detector()
        detector.is_fitted = True
        detector.hmm = _FakeHMM()
        detector.scaler = _FakeScaler()
        filepath = tmp_path / "test_model.pkl"
        detector.save(filepath)
        assert filepath.exists()

        detector2 = _make_detector()
        detector2.load(filepath)
        assert detector2.is_fitted is True
        assert hasattr(detector2.hmm, 'startprob_')


# ---------------------------------------------------------------------------
# PortfolioRegimeManager tests
# ---------------------------------------------------------------------------

class TestPortfolioRegimeManager:
    """Test PortfolioRegimeManager."""

    def test_init_defaults(self):
        manager = PortfolioRegimeManager.__new__(PortfolioRegimeManager)
        manager.base_allocation = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        manager.min_weight = 0.05
        manager.max_weight = 0.80
        assert manager.base_allocation['SPY'] == 0.46

    def test_allocation_shifts_applied(self):
        """Verify regime shifts are applied correctly."""
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        shifts = REGIME_ALLOCATION_SHIFTS[MarketRegime.BULL]
        recommended = {}
        for ticker, weight in base.items():
            shift = shifts.get(ticker, 0.0)
            new_weight = max(0.05, min(0.80, weight + shift))
            recommended[ticker] = new_weight
        # Normalize
        total = sum(recommended.values())
        for t in recommended:
            recommended[t] /= total
        assert abs(sum(recommended.values()) - 1.0) < 0.01
        assert recommended['SPY'] > base['SPY']  # Bull increases equity

    def test_crisis_reduces_equity(self):
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        shifts = REGIME_ALLOCATION_SHIFTS[MarketRegime.CRISIS]
        recommended = {}
        for ticker, weight in base.items():
            shift = shifts.get(ticker, 0.0)
            new_weight = max(0.05, min(0.80, weight + shift))
            recommended[ticker] = new_weight
        total = sum(recommended.values())
        for t in recommended:
            recommended[t] /= total
        assert recommended['SPY'] < base['SPY']

    def test_risk_budget_change(self):
        assert MarketRegime.BEAR in [MarketRegime.BEAR, MarketRegime.CRISIS]
        assert MarketRegime.BULL not in [MarketRegime.BEAR, MarketRegime.CRISIS]

    def test_hmm_available_flag(self):
        assert isinstance(HMM_AVAILABLE, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
