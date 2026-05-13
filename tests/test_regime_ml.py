#!/usr/bin/env python3
"""
Tests for regime-conditional ML strategy — data classes, regime detection,
ML scoring, allocation generation, and ensemble smoothing.
"""
import sys
import os
import json
import math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mock heavy dependencies before importing
mock_regime_classifier = MagicMock()
mock_features = MagicMock()
mock_factor_rotation = MagicMock()
mock_ensemble_voter = MagicMock()

# Create mock classes
class MockRegime:
    pass

class MockFeatures:
    def __init__(self):
        self.vix_level = 20.0
        self.spy_correlation_20d = 0.3
        self.return_20d = 0.01

class MockFactorScore:
    def __init__(self, symbol='SPY', factor_name='momentum', momentum_score=0.5,
                 momentum_acceleration=0.1, volatility=0.15, return_12m=0.10,
                 return_6m=0.05, return_3m=0.02, sharpe_12m=0.67, price=500.0, rank=1):
        self.symbol = symbol
        self.factor_name = factor_name
        self.momentum_score = momentum_score
        self.momentum_acceleration = momentum_acceleration
        self.volatility = volatility
        self.return_12m = return_12m
        self.return_6m = return_6m
        self.return_3m = return_3m
        self.sharpe_12m = sharpe_12m
        self.price = price
        self.rank = rank

class MockRegimeClassifier:
    pass

class MockFeaturePipeline:
    def __init__(self, db_path):
        self.db_path = db_path
    def generate_features(self, symbol):
        return MockFeatures()

class MockFactorMomentumEngine:
    def __init__(self, db_path=None, top_n=2):
        self.db_path = db_path
        self.top_n = top_n

class MockEnsembleVoter:
    def evaluate(self):
        return None

mock_regime_classifier.RegimeClassifier = MockRegimeClassifier
mock_regime_classifier.Regime = MockRegime
mock_features.FeaturePipeline = MockFeaturePipeline
mock_features.Features = MockFeatures
mock_factor_rotation.FactorMomentumEngine = MockFactorMomentumEngine
mock_factor_rotation.FactorScore = MockFactorScore
mock_ensemble_voter.EnsembleVoter = MockEnsembleVoter
mock_ensemble_voter.Regime = MockRegime

sys.modules['src.research.regime_classifier'] = mock_regime_classifier
sys.modules['src.research.features'] = mock_features
sys.modules['src.strategy.factor_rotation'] = mock_factor_rotation
sys.modules['src.strategy.ensemble_voter'] = mock_ensemble_voter

from src.strategy.regime_ml import (
    VolatilityRegime, CorrelationRegime,
    RegimeState, RegimeConditionalScore, RegimeTransition,
    RegimeDetector, RegimeMLScorer, EnsembleSmoother,
    RegimeConditionalEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regime(vol='normal', corr='low', risk_score=0.3, label='normal'):
    """Create a test RegimeState."""
    return RegimeState(
        timestamp='2026-01-01',
        vol_regime=VolatilityRegime(vol),
        corr_regime=CorrelationRegime(corr),
        yield_curve_inverted=False,
        liquidity_stress=False,
        momentum_bearish=False,
        risk_score=risk_score,
        regime_label=label,
    )


def _make_factor_score(symbol='SPY', momentum=0.5, vol=0.15, acceleration=0.1):
    """Create a test FactorScore."""
    return MockFactorScore(
        symbol=symbol,
        factor_name='momentum',
        momentum_score=momentum,
        momentum_acceleration=acceleration,
        volatility=vol,
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    """Test VolatilityRegime and CorrelationRegime enums."""

    def test_vol_regime_values(self):
        assert VolatilityRegime.LOW.value == 'low'
        assert VolatilityRegime.NORMAL.value == 'normal'
        assert VolatilityRegime.HIGH.value == 'high'

    def test_corr_regime_values(self):
        assert CorrelationRegime.LOW.value == 'low'
        assert CorrelationRegime.HIGH.value == 'high'


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestRegimeState:
    """Test RegimeState dataclass."""

    def test_creation(self):
        rs = _make_regime()
        assert rs.vol_regime == VolatilityRegime.NORMAL
        assert rs.corr_regime == CorrelationRegime.LOW
        assert rs.risk_score == 0.3

    def test_to_dict(self):
        rs = _make_regime(vol='high', corr='high', risk_score=0.8, label='high_risk')
        d = rs.to_dict()
        assert d['vol_regime'] == 'high'
        assert d['corr_regime'] == 'high'
        assert d['risk_score'] == 0.8
        assert d['regime_label'] == 'high_risk'


class TestRegimeConditionalScore:
    """Test RegimeConditionalScore dataclass."""

    def test_creation(self):
        fs = _make_factor_score()
        regime = _make_regime()
        rcs = RegimeConditionalScore(
            base_score=fs,
            regime=regime,
            vol_adjusted_momentum=0.5,
            corr_adjusted_weight=0.55,
            regime_multiplier=1.0,
            high_vol_confidence=0.3,
            low_vol_confidence=0.8,
            high_corr_confidence=0.3,
            low_corr_confidence=0.9,
            conditional_score=0.55,
            final_allocation_weight=1.2,
        )
        assert rcs.conditional_score == 0.55

    def test_to_dict(self):
        fs = _make_factor_score()
        regime = _make_regime()
        rcs = RegimeConditionalScore(
            base_score=fs, regime=regime,
            vol_adjusted_momentum=0.5, corr_adjusted_weight=0.55,
            regime_multiplier=1.0, high_vol_confidence=0.3,
            low_vol_confidence=0.8, high_corr_confidence=0.3,
            low_corr_confidence=0.9, conditional_score=0.55,
            final_allocation_weight=1.2,
        )
        d = rcs.to_dict()
        assert 'symbol' in d
        assert 'conditional_score' in d
        assert 'confidence' in d


class TestRegimeTransition:
    """Test RegimeTransition dataclass."""

    def test_creation(self):
        from_regime = _make_regime(vol='normal')
        to_regime = _make_regime(vol='high')
        transition = RegimeTransition(
            from_regime=from_regime,
            to_regime=to_regime,
            transition_date='2026-01-15',
            days_in_transition=3,
            transition_confidence=0.7,
        )
        assert transition.days_in_transition == 3
        assert transition.transition_confidence == 0.7


# ---------------------------------------------------------------------------
# RegimeDetector tests
# ---------------------------------------------------------------------------

class TestRegimeDetector:
    """Test RegimeDetector."""

    def test_thresholds_defined(self):
        assert RegimeDetector.VIX_LOW_THRESHOLD == 15.0
        assert RegimeDetector.VIX_HIGH_THRESHOLD == 25.0
        assert RegimeDetector.CORR_HIGH_THRESHOLD == 0.5

    def test_detect_regime_normal(self):
        detector = RegimeDetector.__new__(RegimeDetector)
        detector.feature_pipeline = MockFeaturePipeline('/tmp/test.db')
        regime = detector.detect_regime('SPY')
        assert isinstance(regime, RegimeState)
        assert regime.vol_regime == VolatilityRegime.NORMAL  # VIX=20

    def test_detect_regime_high_vol(self):
        detector = RegimeDetector.__new__(RegimeDetector)
        mock_pipeline = MagicMock()
        mock_features_high = MockFeatures()
        mock_features_high.vix_level = 30.0
        mock_features_high.spy_correlation_20d = 0.3
        mock_features_high.return_20d = 0.01
        mock_pipeline.generate_features.return_value = mock_features_high
        detector.feature_pipeline = mock_pipeline
        regime = detector.detect_regime('SPY')
        assert regime.vol_regime == VolatilityRegime.HIGH

    def test_detect_regime_low_vol(self):
        detector = RegimeDetector.__new__(RegimeDetector)
        mock_pipeline = MagicMock()
        mock_features_low = MockFeatures()
        mock_features_low.vix_level = 12.0
        mock_features_low.spy_correlation_20d = 0.3
        mock_features_low.return_20d = 0.01
        mock_pipeline.generate_features.return_value = mock_features_low
        detector.feature_pipeline = mock_pipeline
        regime = detector.detect_regime('SPY')
        assert regime.vol_regime == VolatilityRegime.LOW

    def test_detect_regime_high_corr(self):
        detector = RegimeDetector.__new__(RegimeDetector)
        mock_pipeline = MagicMock()
        mock_features_corr = MockFeatures()
        mock_features_corr.vix_level = 20.0
        mock_features_corr.spy_correlation_20d = 0.7
        mock_features_corr.return_20d = 0.01
        mock_pipeline.generate_features.return_value = mock_features_corr
        detector.feature_pipeline = mock_pipeline
        regime = detector.detect_regime('SPY')
        assert regime.corr_regime == CorrelationRegime.HIGH

    def test_detect_regime_momentum_bearish(self):
        detector = RegimeDetector.__new__(RegimeDetector)
        mock_pipeline = MagicMock()
        mock_features_bear = MockFeatures()
        mock_features_bear.vix_level = 20.0
        mock_features_bear.spy_correlation_20d = 0.3
        mock_features_bear.return_20d = -0.15
        mock_pipeline.generate_features.return_value = mock_features_bear
        detector.feature_pipeline = mock_pipeline
        regime = detector.detect_regime('SPY')
        assert regime.momentum_bearish is True

    def test_detect_regime_no_features_defaults(self):
        detector = RegimeDetector.__new__(RegimeDetector)
        mock_pipeline = MagicMock()
        mock_pipeline.generate_features.return_value = None
        detector.feature_pipeline = mock_pipeline
        regime = detector.detect_regime('SPY')
        assert regime.vol_regime == VolatilityRegime.NORMAL
        assert regime.risk_score == 0.5
        assert regime.regime_label == 'normal'

    def test_risk_score_bounded(self):
        detector = RegimeDetector.__new__(RegimeDetector)
        detector.feature_pipeline = MockFeaturePipeline('/tmp/test.db')
        regime = detector.detect_regime('SPY')
        assert 0.0 <= regime.risk_score <= 1.0

    def test_regime_label_classification(self):
        detector = RegimeDetector.__new__(RegimeDetector)
        mock_pipeline = MagicMock()
        mock_features_crisis = MockFeatures()
        mock_features_crisis.vix_level = 35.0
        mock_features_crisis.spy_correlation_20d = 0.8
        mock_features_crisis.return_20d = -0.15
        mock_pipeline.generate_features.return_value = mock_features_crisis
        detector.feature_pipeline = mock_pipeline
        regime = detector.detect_regime('SPY')
        # High vol + high corr + bearish → high risk score
        assert regime.risk_score > 0.5


# ---------------------------------------------------------------------------
# RegimeMLScorer tests
# ---------------------------------------------------------------------------

class TestRegimeMLScorer:
    """Test RegimeMLScorer."""

    def test_regime_weights_defined(self):
        assert 'high_vol' in RegimeMLScorer.REGIME_WEIGHTS
        assert 'low_vol' in RegimeMLScorer.REGIME_WEIGHTS
        assert 'high_corr' in RegimeMLScorer.REGIME_WEIGHTS
        assert 'low_corr' in RegimeMLScorer.REGIME_WEIGHTS

    def test_calculate_regime_score_normal(self):
        scorer = RegimeMLScorer.__new__(RegimeMLScorer)
        scorer.current_regime = None
        fs = _make_factor_score(momentum=0.5, vol=0.15)
        regime = _make_regime(vol='normal', corr='low', risk_score=0.3)
        result = scorer.calculate_regime_score(fs, regime)
        assert isinstance(result, RegimeConditionalScore)
        assert result.conditional_score != 0

    def test_calculate_regime_score_high_vol(self):
        scorer = RegimeMLScorer.__new__(RegimeMLScorer)
        scorer.current_regime = None
        fs = _make_factor_score(momentum=0.5, vol=0.15, acceleration=0.2)
        regime = _make_regime(vol='high', corr='low', risk_score=0.5)
        result = scorer.calculate_regime_score(fs, regime)
        # High vol: vol_adjusted = momentum * 0.5 + acceleration * 0.5
        expected = 0.5 * 0.5 + 0.2 * 0.5
        assert abs(result.vol_adjusted_momentum - expected) < 0.01

    def test_calculate_regime_score_low_vol(self):
        scorer = RegimeMLScorer.__new__(RegimeMLScorer)
        scorer.current_regime = None
        fs = _make_factor_score(momentum=0.5)
        regime = _make_regime(vol='low', corr='low', risk_score=0.2)
        result = scorer.calculate_regime_score(fs, regime)
        # Low vol: vol_adjusted = momentum * 1.2
        assert abs(result.vol_adjusted_momentum - 0.6) < 0.01

    def test_calculate_regime_score_high_corr(self):
        scorer = RegimeMLScorer.__new__(RegimeMLScorer)
        scorer.current_regime = None
        fs = _make_factor_score(momentum=0.5)
        regime = _make_regime(vol='normal', corr='high', risk_score=0.5)
        result = scorer.calculate_regime_score(fs, regime)
        # High corr: corr_adjusted = vol_adjusted * 0.8
        assert result.corr_adjusted_weight < result.vol_adjusted_momentum

    def test_regime_multiplier_high_risk(self):
        scorer = RegimeMLScorer.__new__(RegimeMLScorer)
        scorer.current_regime = None
        fs = _make_factor_score()
        regime = _make_regime(risk_score=0.8)
        result = scorer.calculate_regime_score(fs, regime)
        assert result.regime_multiplier == 0.7

    def test_regime_multiplier_low_risk(self):
        scorer = RegimeMLScorer.__new__(RegimeMLScorer)
        scorer.current_regime = None
        fs = _make_factor_score()
        regime = _make_regime(risk_score=0.2)
        result = scorer.calculate_regime_score(fs, regime)
        assert result.regime_multiplier == 1.15

    def test_generate_allocation_empty(self):
        scorer = RegimeMLScorer.__new__(RegimeMLScorer)
        scorer.current_regime = None
        alloc = scorer.generate_allocation({}, top_n=2)
        assert alloc == {'SPY': 1.0}

    def test_generate_allocation_top_n(self):
        scorer = RegimeMLScorer.__new__(RegimeMLScorer)
        scorer.current_regime = None
        fs_a = _make_factor_score(symbol='SPY')
        fs_b = _make_factor_score(symbol='GLD')
        fs_c = _make_factor_score(symbol='TLT')
        regime = _make_regime()

        scores = {
            'SPY': scorer.calculate_regime_score(fs_a, regime),
            'GLD': scorer.calculate_regime_score(fs_b, regime),
            'TLT': scorer.calculate_regime_score(fs_c, regime),
        }
        alloc = scorer.generate_allocation(scores, top_n=2)
        assert len(alloc) == 2

    def test_generate_allocation_sums_to_one(self):
        scorer = RegimeMLScorer.__new__(RegimeMLScorer)
        scorer.current_regime = None
        fs_a = _make_factor_score(symbol='SPY', momentum=0.6)
        fs_b = _make_factor_score(symbol='GLD', momentum=0.4)
        regime = _make_regime()

        scores = {
            'SPY': scorer.calculate_regime_score(fs_a, regime),
            'GLD': scorer.calculate_regime_score(fs_b, regime),
        }
        alloc = scorer.generate_allocation(scores, top_n=2)
        assert abs(sum(alloc.values()) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# EnsembleSmoother tests
# ---------------------------------------------------------------------------

class TestEnsembleSmoother:
    """Test EnsembleSmoother."""

    def test_init(self):
        smoother = EnsembleSmoother()
        assert smoother.transition_days == 5
        assert smoother.transition_history == []

    def test_detect_regime_change_no_previous(self):
        smoother = EnsembleSmoother()
        current = _make_regime(vol='normal')
        assert smoother.detect_regime_change(current, None) is False

    def test_detect_regime_change_vol_change(self):
        smoother = EnsembleSmoother()
        prev = _make_regime(vol='normal')
        curr = _make_regime(vol='high')
        assert smoother.detect_regime_change(curr, prev) is True

    def test_detect_regime_change_corr_change(self):
        smoother = EnsembleSmoother()
        prev = _make_regime(corr='low')
        curr = _make_regime(corr='high')
        assert smoother.detect_regime_change(curr, prev) is True

    def test_detect_regime_change_risk_jump(self):
        smoother = EnsembleSmoother()
        prev = _make_regime(risk_score=0.2)
        curr = _make_regime(risk_score=0.6)
        assert smoother.detect_regime_change(curr, prev) is True

    def test_detect_regime_change_no_change(self):
        smoother = EnsembleSmoother()
        prev = _make_regime(vol='normal', corr='low', risk_score=0.3)
        curr = _make_regime(vol='normal', corr='low', risk_score=0.35)
        assert smoother.detect_regime_change(curr, prev) is False

    def test_calculate_transition_weights_no_previous(self):
        smoother = EnsembleSmoother()
        new = {'SPY': 0.6, 'GLD': 0.4}
        result = smoother.calculate_transition_weights(new, None, _make_regime(), None)
        assert result == new

    def test_calculate_transition_weights_blends(self):
        smoother = EnsembleSmoother()
        new = {'SPY': 0.8, 'GLD': 0.2}
        old = {'SPY': 0.4, 'GLD': 0.6}
        regime = _make_regime()
        transition = RegimeTransition(
            from_regime=regime, to_regime=regime,
            transition_date='2026-01-01', days_in_transition=3,
        )
        result = smoother.calculate_transition_weights(new, old, regime, transition)
        # Should be blended between old and new
        assert 0.4 < result['SPY'] < 0.8
        assert 0.2 < result['GLD'] < 0.6

    def test_update_transition_starts_new(self):
        smoother = EnsembleSmoother()
        prev = _make_regime(vol='normal')
        curr = _make_regime(vol='high')
        transition = smoother.update_transition(curr, prev)
        assert transition is not None
        assert transition.days_in_transition == 0
        assert len(smoother.transition_history) == 1

    def test_update_transition_continues(self):
        smoother = EnsembleSmoother()
        prev = _make_regime(vol='normal')
        curr = _make_regime(vol='high')
        smoother.update_transition(curr, prev)
        # Same regime again → continues transition
        same = _make_regime(vol='high')
        transition = smoother.update_transition(same, curr)
        assert transition is not None
        assert transition.days_in_transition == 1

    def test_smooth_allocation_no_transition(self):
        smoother = EnsembleSmoother()
        raw = {'SPY': 0.6, 'GLD': 0.4}
        regime = _make_regime()
        result, transition = smoother.smooth_allocation(raw, regime, None, None)
        assert result == raw
        assert transition is None

    def test_smooth_allocation_during_transition(self):
        smoother = EnsembleSmoother()
        raw = {'SPY': 0.8, 'GLD': 0.2}
        old = {'SPY': 0.4, 'GLD': 0.6}
        prev = _make_regime(vol='normal')
        curr = _make_regime(vol='high')
        # Start transition
        smoother.update_transition(curr, prev)
        result, transition = smoother.smooth_allocation(raw, curr, prev, old)
        assert transition is not None
        # Should be blended
        assert 0.4 < result['SPY'] < 0.8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
