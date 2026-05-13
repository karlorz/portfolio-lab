#!/usr/bin/env python3
"""
Tests for Wasserstein HMM regime detector — data classes, template matching,
rule-based fallback, feature preparation, state persistence, and regime stats.
"""
import sys
import os
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.strategy.regime_hmm import (
    RegimeState, WassersteinTemplate, WassersteinHMMDetector,
    HMMRegimeCLI, HMM_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detector():
    """Create a WassersteinHMMDetector with mocked DB."""
    detector = WassersteinHMMDetector.__new__(WassersteinHMMDetector)
    detector.n_states = 4
    detector.lookback_days = 252
    detector.template_window = 63
    detector.random_state = 42
    detector.model = None
    detector.templates = {}
    detector.regime_history = __import__('collections').deque(maxlen=252)
    detector.feature_history = __import__('collections').deque(maxlen=252)
    detector.state_to_regime = {}
    detector.regime_to_state = {}
    detector._last_training_date = None
    detector.feature_means = None
    detector.feature_stds = None
    return detector


def _make_feature_data(n=100):
    """Create synthetic feature data."""
    np.random.seed(42)
    data = []
    for i in range(n):
        data.append({
            'date': f'2026-01-{i+1:02d}',
            'vix_level': 15 + np.random.normal(0, 3),
            'vix_change': np.random.normal(0, 1),
            'yield_spread': 1.5 + np.random.normal(0, 0.3),
            'momentum_20d': np.random.normal(0.01, 0.03),
            'momentum_60d': np.random.normal(0.03, 0.05),
            'hyg_spread_proxy': 3.5 + np.random.normal(0, 0.5),
        })
    return data


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestRegimeState:
    """Test RegimeState dataclass."""

    def test_creation(self):
        rs = RegimeState(
            timestamp='2026-01-01T10:00:00',
            regime_label='bull', regime_id=0, probability=0.85,
            vix_level=14.0, vix_change=-1.0, yield_spread=1.5,
            momentum_20d=0.02, momentum_60d=0.08, correlation_proxy=3.5,
            template_distance=0.5, template_confidence=0.7,
        )
        assert rs.regime_label == 'bull'
        assert rs.probability == 0.85

    def test_to_dict(self):
        rs = RegimeState(
            timestamp='2026-01-01', regime_label='bear', regime_id=1,
            probability=0.7, vix_level=25.0, vix_change=2.0,
            yield_spread=-0.5, momentum_20d=-0.03, momentum_60d=-0.05,
            correlation_proxy=5.0, template_distance=1.2, template_confidence=0.6,
        )
        d = rs.to_dict()
        assert d['regime_label'] == 'bear'
        assert 'vix_level' in d


class TestWassersteinTemplate:
    """Test WassersteinTemplate dataclass."""

    def test_creation(self):
        tmpl = WassersteinTemplate(
            regime_label='bull',
            mean_vector=np.array([14.0, 0.0, 1.5, 0.02, 0.08, 3.5]),
            cov_matrix=np.eye(6) * 0.1,
            sample_count=100,
            last_updated='2026-01-01',
        )
        assert tmpl.regime_label == 'bull'
        assert tmpl.sample_count == 100

    def test_wasserstein_distance_1d(self):
        tmpl = WassersteinTemplate(
            regime_label='bull',
            mean_vector=np.array([14.0, 0.0]),
            cov_matrix=np.eye(2) * 0.1,
            sample_count=50,
            last_updated='2026-01-01',
        )
        obs = np.array([14.5, 0.1])
        dist = tmpl.wasserstein_distance(obs)
        assert dist >= 0
        assert np.isfinite(dist)

    def test_wasserstein_distance_close(self):
        """Observations near template mean → small distance."""
        mean = np.array([14.0, 0.0, 1.5])
        tmpl = WassersteinTemplate(
            regime_label='bull', mean_vector=mean,
            cov_matrix=np.eye(3) * 0.1, sample_count=50,
            last_updated='2026-01-01',
        )
        obs = mean + 0.01  # Very close
        dist = tmpl.wasserstein_distance(obs)
        assert dist < 1.0

    def test_wasserstein_distance_far(self):
        """Observations far from template mean → larger distance."""
        mean = np.array([14.0, 0.0, 1.5])
        tmpl = WassersteinTemplate(
            regime_label='bull', mean_vector=mean,
            cov_matrix=np.eye(3) * 0.1, sample_count=50,
            last_updated='2026-01-01',
        )
        obs = mean + 10.0  # Very far
        dist = tmpl.wasserstein_distance(obs)
        assert dist > 5.0

    def test_wasserstein_distance_2d_observations(self):
        """Multi-row observations → uses mean and cov."""
        mean = np.array([14.0, 0.0])
        tmpl = WassersteinTemplate(
            regime_label='bull', mean_vector=mean,
            cov_matrix=np.eye(2) * 0.1, sample_count=50,
            last_updated='2026-01-01',
        )
        obs = np.array([[14.0, 0.0], [14.1, 0.1], [13.9, -0.1]])
        dist = tmpl.wasserstein_distance(obs)
        assert dist >= 0


# ---------------------------------------------------------------------------
# Detector init tests
# ---------------------------------------------------------------------------

class TestDetectorInit:
    """Test WassersteinHMMDetector initialization."""

    def test_default_params(self):
        detector = WassersteinHMMDetector()
        assert detector.n_states == 4
        assert detector.lookback_days == 252
        assert detector.random_state == 42

    def test_custom_params(self):
        detector = WassersteinHMMDetector(n_states=3, lookback_days=126)
        assert detector.n_states == 3
        assert detector.lookback_days == 126

    def test_empty_templates(self):
        detector = WassersteinHMMDetector()
        assert detector.templates == {}

    def test_features_defined(self):
        assert len(WassersteinHMMDetector.FEATURES) == 6

    def test_regime_templates_defined(self):
        assert 'bull' in WassersteinHMMDetector.REGIME_TEMPLATES
        assert 'bear' in WassersteinHMMDetector.REGIME_TEMPLATES
        assert 'crisis' in WassersteinHMMDetector.REGIME_TEMPLATES


# ---------------------------------------------------------------------------
# Feature preparation tests
# ---------------------------------------------------------------------------

class TestPrepareFeatures:
    """Test _prepare_features method."""

    def test_returns_scaled_matrix(self):
        detector = _make_detector()
        data = _make_feature_data(50)
        X = detector._prepare_features(data)
        assert X.shape == (50, 6)

    def test_standardized(self):
        detector = _make_detector()
        data = _make_feature_data(100)
        X = detector._prepare_features(data)
        # Should be roughly zero-mean after standardization
        means = np.mean(X, axis=0)
        np.testing.assert_allclose(means, 0.0, atol=0.1)

    def test_stores_means_and_stds(self):
        detector = _make_detector()
        data = _make_feature_data(50)
        detector._prepare_features(data)
        assert detector.feature_means is not None
        assert detector.feature_stds is not None
        assert len(detector.feature_means) == 6


# ---------------------------------------------------------------------------
# Template matching tests
# ---------------------------------------------------------------------------

class TestMatchToTemplate:
    """Test _match_to_template method."""

    def test_bull_features_match_bull(self):
        detector = _make_detector()
        data = _make_feature_data(50)
        detector._prepare_features(data)
        # Bull template: low VIX, positive momentum
        bull_mean = np.array([14.0, 0.0, 1.5, 0.02, 0.08, 3.5])
        bull_scaled = (bull_mean - detector.feature_means) / detector.feature_stds
        result = detector._match_to_template(bull_scaled)
        assert result == 'bull'

    def test_crisis_features_match_crisis(self):
        detector = _make_detector()
        data = _make_feature_data(50)
        detector._prepare_features(data)
        crisis_mean = np.array([35.0, 2.0, -1.0, -0.05, -0.10, 8.0])
        crisis_scaled = (crisis_mean - detector.feature_means) / detector.feature_stds
        result = detector._match_to_template(crisis_scaled)
        assert result == 'crisis'

    def test_returns_string(self):
        detector = _make_detector()
        data = _make_feature_data(50)
        detector._prepare_features(data)
        result = detector._match_to_template(np.zeros(6))
        assert isinstance(result, str)
        assert result in ['bull', 'bear', 'neutral', 'crisis']


# ---------------------------------------------------------------------------
# Rule-based fallback tests
# ---------------------------------------------------------------------------

class TestRuleBasedRegime:
    """Test _rule_based_regime method."""

    def test_crisis_detection(self):
        detector = _make_detector()
        features = {'vix_level': 35.0, 'momentum_20d': -0.05, 'momentum_60d': -0.10}
        regime, conf = detector._rule_based_regime(features)
        assert regime == 'crisis'
        assert conf == 0.8

    def test_bear_detection(self):
        detector = _make_detector()
        features = {'vix_level': 28.0, 'momentum_20d': -0.01, 'momentum_60d': -0.02}
        regime, conf = detector._rule_based_regime(features)
        assert regime == 'bear'

    def test_bull_detection(self):
        detector = _make_detector()
        features = {'vix_level': 14.0, 'momentum_20d': 0.02, 'momentum_60d': 0.08}
        regime, conf = detector._rule_based_regime(features)
        assert regime == 'bull'

    def test_neutral_detection(self):
        detector = _make_detector()
        features = {'vix_level': 19.0, 'momentum_20d': 0.005, 'momentum_60d': 0.02}
        regime, conf = detector._rule_based_regime(features)
        assert regime == 'neutral'

    def test_confidence_bounded(self):
        detector = _make_detector()
        for features in [
            {'vix_level': 35, 'momentum_20d': -0.05, 'momentum_60d': -0.10},
            {'vix_level': 14, 'momentum_20d': 0.02, 'momentum_60d': 0.08},
            {'vix_level': 19, 'momentum_20d': 0.0, 'momentum_60d': 0.03},
        ]:
            _, conf = detector._rule_based_regime(features)
            assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Fallback regime detection tests
# ---------------------------------------------------------------------------

class TestFallbackRegimeDetection:
    """Test _fallback_regime_detection method."""

    def test_returns_neutral(self):
        detector = _make_detector()
        rs = detector._fallback_regime_detection()
        assert rs.regime_label == 'neutral'
        assert rs.probability == 0.5
        assert rs.regime_id == -1


# ---------------------------------------------------------------------------
# Regime stats tests
# ---------------------------------------------------------------------------

class TestGetRegimeStats:
    """Test get_regime_stats method."""

    def test_empty_history(self):
        detector = _make_detector()
        stats = detector.get_regime_stats()
        assert stats['total_detections'] == 0
        assert stats['regime_distribution'] == {}

    def test_with_history(self):
        detector = _make_detector()
        for label in ['bull', 'bull', 'bear', 'neutral']:
            rs = RegimeState(
                timestamp='2026-01-01', regime_label=label, regime_id=0,
                probability=0.8, vix_level=15.0, vix_change=0.0,
                yield_spread=1.5, momentum_20d=0.01, momentum_60d=0.05,
                correlation_proxy=3.5, template_distance=0.5, template_confidence=0.7,
            )
            detector.regime_history.append(rs)
        stats = detector.get_regime_stats()
        assert stats['total_detections'] == 4
        assert stats['regime_distribution']['bull'] == 2
        assert stats['current_regime'] == 'neutral'


# ---------------------------------------------------------------------------
# State persistence tests
# ---------------------------------------------------------------------------

class TestSaveLoadState:
    """Test save_state and load_state methods."""

    def test_save_creates_file(self, tmp_path):
        detector = _make_detector()
        detector.state_to_regime = {0: 'bull', 1: 'bear'}
        filepath = tmp_path / "hmm_state.json"
        detector.save_state(str(filepath))
        assert filepath.exists()

    def test_save_load_roundtrip(self, tmp_path):
        detector = _make_detector()
        detector.n_states = 4
        detector._last_training_date = '2026-01-01T10:00:00'
        detector.state_to_regime = {0: 'bull', 1: 'bear', 2: 'neutral', 3: 'crisis'}
        filepath = tmp_path / "hmm_state.json"
        detector.save_state(str(filepath))

        detector2 = _make_detector()
        result = detector2.load_state(str(filepath))
        assert result is True
        assert detector2.n_states == 4
        assert detector2.state_to_regime[0] == 'bull'

    def test_load_nonexistent_returns_false(self, tmp_path):
        detector = _make_detector()
        result = detector.load_state(str(tmp_path / "nonexistent.json"))
        assert result is False


# ---------------------------------------------------------------------------
# HMM availability test
# ---------------------------------------------------------------------------

class TestHMMAvailability:
    """Test HMM_AVAILABLE flag."""

    def test_is_boolean(self):
        assert isinstance(HMM_AVAILABLE, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
