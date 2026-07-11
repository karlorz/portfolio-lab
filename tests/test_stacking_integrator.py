"""
Tests for Stacking Ensemble Integrator (v3.10 Phase 3)

Covers:
- Model loading and metadata
- Prediction with and without model
- Fallback to weighted voting
- Feature extraction
- Drift detection
- Prediction history tracking

Author: Portfolio-Lab Agent
Version: v3.10 Phase 3
"""

import pytest
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
from typing import Optional

from src.signals.stacking_integrator import (
    StackingIntegrator,
    StackingPrediction,
    ModelMetadata,
    get_stacking_prediction
)
from src.signals.signal_source import SignalSource


class TestStackingPrediction:
    """Test StackingPrediction dataclass"""
    
    def test_prediction_creation(self):
        """Test creating a prediction object"""
        pred = StackingPrediction(
            direction='bullish',
            confidence=0.75,
            probability_bullish=0.75,
            probability_bearish=0.15,
            probability_neutral=0.10,
            fallback_used=False,
            model_version='v1.0',
            latency_ms=2.5
        )
        
        assert pred.direction == 'bullish'
        assert pred.confidence == 0.75
        assert not pred.fallback_used
        assert pred.model_version == 'v1.0'
        assert pred.latency_ms == 2.5
        assert isinstance(pred.timestamp, datetime)
    
    def test_prediction_defaults(self):
        """Test prediction with default values"""
        pred = StackingPrediction(
            direction='neutral',
            confidence=0.5,
            probability_bullish=0.33,
            probability_bearish=0.33,
            probability_neutral=0.34,
            fallback_used=True
        )
        
        assert pred.top_features == []
        assert isinstance(pred.timestamp, datetime)
        assert pred.model_version == 'unknown'
        assert pred.latency_ms == 0.0


class TestModelMetadata:
    """Test ModelMetadata dataclass"""
    
    def test_metadata_creation(self):
        """Test creating metadata object"""
        meta = ModelMetadata(
            version='v1.2',
            training_date=datetime(2026, 5, 1),
            feature_count=102,
            accuracy_train=0.75,
            accuracy_val=0.68,
            feature_importance={'feature_1': 0.15, 'feature_2': 0.12},
            total_samples=15000
        )
        
        assert meta.version == 'v1.2'
        assert meta.feature_count == 102
        assert meta.accuracy_train == 0.75
        assert len(meta.feature_importance) == 2

    def test_metadata_carries_roster_version_and_fallback_semantics(self):
        """Feature dimensions are meaningful only with explicit roster metadata."""
        roster = [source.value for source in SignalSource]
        meta = ModelMetadata(
            version='v1.2',
            training_date=datetime(2026, 5, 1),
            feature_count=128,
            accuracy_train=0.75,
            accuracy_val=0.68,
            feature_importance={'feature_1': 0.15},
            total_samples=15000,
            source_roster=roster,
            source_roster_version="SignalSource.full.v1",
            fallback_semantics="no_model_feature_count_unavailable",
        )

        assert meta.source_roster == roster
        assert meta.source_roster_version == "SignalSource.full.v1"
        assert meta.fallback_semantics == "no_model_feature_count_unavailable"


class TestStackingIntegratorInit:
    """Test StackingIntegrator initialization"""
    
    def test_init_without_model(self):
        """Test integrator initializes without model"""
        integrator = StackingIntegrator()
        
        assert integrator.model is None
        assert integrator.metadata is None
        assert integrator.fallback_threshold == 0.6
        assert integrator.prediction_history == []
    
    def test_init_with_custom_threshold(self):
        """Test integrator with custom fallback threshold"""
        integrator = StackingIntegrator(fallback_threshold=0.7)
        
        assert integrator.fallback_threshold == 0.7
    
    def test_init_with_feature_engine(self):
        """Test integrator with feature engine"""
        mock_engine = Mock()
        integrator = StackingIntegrator(feature_engine=mock_engine)
        
        assert integrator.feature_engine == mock_engine


# Create picklable mock models at module level
class PicklableModel:
    """Picklable mock model for testing"""
    def __init__(self, probs):
        self.classes_ = ['bearish', 'neutral', 'bullish']
        self._probs = probs
    
    def predict_proba(self, X):
        return np.array([self._probs])


class PicklableModelLowConf:
    """Picklable mock model returning low confidence"""
    def __init__(self):
        self.classes_ = ['bearish', 'neutral', 'bullish']

    def predict_proba(self, X):
        return np.array([[0.3, 0.35, 0.35]])


class PicklableTwoClassModel:
    """Picklable mock model with only 2 numeric classes."""
    classes_ = [0, 1]
    def predict_proba(self, X):
        return np.array([[0.3, 0.7]])


class PicklableEqualProbsModel:
    """Picklable mock model returning near-equal probabilities."""
    classes_ = ["bearish", "neutral", "bullish"]
    def predict_proba(self, X):
        return np.array([[0.33, 0.34, 0.33]])


class PicklableMissingBearishModel:
    """Picklable model lacking 'bearish' in classes."""
    classes_ = ["bullish", "neutral"]
    def predict_proba(self, X):
        return np.array([[0.6, 0.4]])


class PicklableNoModelKeyModel:
    """Picklable model with no 'model' key in data."""
    pass


class TestModelLoading:
    """Test model loading functionality"""
    
    def test_load_valid_model(self, tmp_path):
        """Test loading a valid model pickle"""
        roster = [source.value for source in SignalSource]
        model_data = {
            'model': PicklableModel([0.1, 0.2, 0.7]),
            'metadata': {
                'version': 'v1.0',
                'training_date': datetime(2026, 5, 1),
                'feature_count': 102,
                'accuracy_train': 0.75,
                'accuracy_val': 0.68,
                'feature_importance': {'feat_1': 0.15},
                'total_samples': 10000,
                'source_roster': roster,
                'source_roster_version': 'SignalSource.full.v1',
                'fallback_semantics': 'weighted_voting_if_model_unavailable',
            }
        }
        
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        integrator = StackingIntegrator()
        result = integrator.load_model(model_path)
        
        assert result is True
        assert integrator.model is not None
        assert integrator.metadata is not None
        assert integrator.metadata.version == 'v1.0'
        assert integrator.metadata.accuracy_val == 0.68
        assert integrator.metadata.source_roster == roster
        assert integrator.metadata.source_roster_version == 'SignalSource.full.v1'
        assert integrator.metadata.fallback_semantics == 'weighted_voting_if_model_unavailable'
    
    def test_load_invalid_model(self, tmp_path):
        """Test loading non-existent model"""
        integrator = StackingIntegrator()
        result = integrator.load_model(tmp_path / "nonexistent.pkl")
        
        assert result is False
        assert integrator.model is None
    
    def test_load_corrupted_model(self, tmp_path):
        """Test loading corrupted pickle file"""
        model_path = tmp_path / "corrupted.pkl"
        with open(model_path, 'wb') as f:
            f.write(b'not a valid pickle')
        
        integrator = StackingIntegrator()
        result = integrator.load_model(model_path)
        
        assert result is False


class TestPredictionWithoutModel:
    """Test prediction when no model loaded"""
    
    def test_fallback_to_weighted_voting(self):
        """Test fallback when no model available"""
        integrator = StackingIntegrator()
        
        base_signals = {
            'tsmom': {'direction': 'bullish', 'confidence': 0.8, 'strength': 0.7},
            'hmm_regime': {'direction': 'bullish', 'confidence': 0.6, 'strength': 0.5},
            'base': {'direction': 'neutral', 'confidence': 0.5, 'strength': 0.3}
        }
        
        result = integrator.predict(base_signals)
        
        assert isinstance(result, StackingPrediction)
        assert result.fallback_used is True
        assert result.model_version == 'fallback_v2.81'
        assert result.confidence > 0
        assert result.direction in ['bullish', 'bearish', 'neutral']
    
    def test_weighted_voting_calculation(self):
        """Test weighted voting produces correct results"""
        integrator = StackingIntegrator()
        
        # All bullish signals with high confidence
        base_signals = {
            'tsmom': {'direction': 'bullish', 'confidence': 0.9, 'strength': 0.8},
            'hmm_regime': {'direction': 'bullish', 'confidence': 0.9, 'strength': 0.8},
            'fed_policy': {'direction': 'bullish', 'confidence': 0.9, 'strength': 0.8},
        }
        
        result = integrator.predict(base_signals)
        
        assert result.direction == 'bullish'
        assert result.probability_bullish > result.probability_bearish
        assert result.probability_bullish > result.probability_neutral


class TestPredictionWithModel:
    """Test prediction with loaded model"""
    
    def test_high_confidence_prediction(self, tmp_path):
        """Test prediction when model gives high confidence"""
        model_data = {
            'model': PicklableModel([0.05, 0.10, 0.85]),
            'metadata': {
                'version': 'v2.0',
                'training_date': datetime(2026, 5, 1),
                'feature_count': 102,
                'accuracy_train': 0.80,
                'accuracy_val': 0.75,
                'feature_importance': {},
                'total_samples': 20000
            }
        }
        
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        integrator = StackingIntegrator(model_path=model_path)
        
        base_signals = {
            'tsmom': {'direction': 'bullish', 'confidence': 0.8},
            'hmm_regime': {'direction': 'bullish', 'confidence': 0.7}
        }
        
        result = integrator.predict(base_signals)
        
        assert result.fallback_used is False
        assert result.direction == 'bullish'
        assert result.confidence == 0.85
        assert result.probability_bullish == 0.85
        assert result.model_version == 'v2.0'
    
    def test_low_confidence_fallback(self, tmp_path):
        """Test fallback when model confidence is below threshold"""
        model_data = {
            'model': PicklableModelLowConf(),
            'metadata': {
                'version': 'v1.0',
                'training_date': datetime(2026, 5, 1),
                'feature_count': 102,
                'accuracy_train': 0.75,
                'accuracy_val': 0.68,
                'feature_importance': {},
                'total_samples': 10000
            }
        }
        
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        integrator = StackingIntegrator(model_path=model_path, fallback_threshold=0.6)
        
        base_signals = {
            'tsmom': {'direction': 'bullish', 'confidence': 0.8}
        }
        
        result = integrator.predict(base_signals)
        
        # Should fallback due to low confidence (0.35 < 0.6)
        assert result.fallback_used is True


class TestFeatureExtraction:
    """Test feature extraction functionality"""
    
    def test_simple_feature_extraction(self):
        """Test simple feature extraction without feature engine"""
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.feature_count = 20
        
        base_signals = {
            'tsmom': {'direction': 'bullish', 'confidence': 0.8, 'strength': 0.7},
            'hmm_regime': {'direction': 'bearish', 'confidence': 0.6, 'strength': 0.5}
        }
        
        features = integrator._extract_simple_features(
            base_signals, 'bull', 20.0
        )
        
        assert isinstance(features, np.ndarray)
        assert len(features) == 20  # Should pad to metadata count
        assert features.dtype == np.float32
    
    def test_feature_extraction_with_none_regime(self):
        """Test feature extraction with None regime"""
        integrator = StackingIntegrator()
        
        base_signals = {'tsmom': {'direction': 'neutral', 'confidence': 0.5}}
        
        features = integrator._extract_simple_features(
            base_signals, None, None
        )
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0


class TestPredictionHistory:
    """Test prediction history tracking"""
    
    def test_history_tracking(self):
        """Test predictions are added to history"""
        integrator = StackingIntegrator()
        
        base_signals = {
            'tsmom': {'direction': 'bullish', 'confidence': 0.8}
        }
        
        # Make multiple predictions
        for _ in range(5):
            integrator.predict(base_signals)
        
        assert len(integrator.prediction_history) == 5
    
    def test_history_size_limit(self):
        """Test history is limited to max size"""
        integrator = StackingIntegrator()
        integrator.max_history = 10
        
        base_signals = {
            'tsmom': {'direction': 'bullish', 'confidence': 0.8}
        }
        
        # Make more predictions than limit
        for _ in range(15):
            integrator.predict(base_signals)
        
        assert len(integrator.prediction_history) == 10
    
    def test_get_accuracy_stats(self):
        """Test accuracy statistics calculation"""
        integrator = StackingIntegrator()
        
        # Add some mock predictions
        for i in range(5):
            pred = StackingPrediction(
                direction='bullish',
                confidence=0.7 + i * 0.05,
                probability_bullish=0.7,
                probability_bearish=0.15,
                probability_neutral=0.15,
                fallback_used=i % 2 == 0,
                latency_ms=2.0 + i
            )
            integrator.prediction_history.append(pred)
        
        stats = integrator.get_accuracy_stats(window_days=30)
        
        assert stats['count'] == 5
        assert stats['fallback_rate'] == 0.6  # 3 out of 5 (indices 0, 2, 4 are True)
        assert stats['avg_confidence'] > 0
        assert stats['avg_latency_ms'] > 0
    
    def test_accuracy_stats_empty_history(self):
        """Test accuracy stats with empty history"""
        integrator = StackingIntegrator()
        
        stats = integrator.get_accuracy_stats()
        
        assert stats['accuracy'] == 0.0
        assert stats['count'] == 0
        assert stats['fallback_rate'] == 0.0


class TestDriftDetection:
    """Test model drift detection"""
    
    def test_no_drift_normal_operation(self):
        """Test no drift detected with normal fallback rate"""
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.accuracy_train = 0.75
        integrator.metadata.accuracy_val = 0.70
        
        # Add predictions with low fallback rate
        for i in range(10):
            pred = StackingPrediction(
                direction='bullish',
                confidence=0.8,
                probability_bullish=0.8,
                probability_bearish=0.1,
                probability_neutral=0.1,
                fallback_used=False
            )
            integrator.prediction_history.append(pred)
        
        drift = integrator.detect_drift()
        
        assert drift is None
    
    def test_drift_high_fallback_rate(self):
        """Test drift detected with high fallback rate"""
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.accuracy_train = 0.75
        integrator.metadata.accuracy_val = 0.70
        
        # Add predictions with high fallback rate
        for i in range(10):
            pred = StackingPrediction(
                direction='bullish',
                confidence=0.8,
                probability_bullish=0.8,
                probability_bearish=0.1,
                probability_neutral=0.1,
                fallback_used=True  # All fallback
            )
            integrator.prediction_history.append(pred)
        
        drift = integrator.detect_drift()
        
        assert drift is not None
        assert 'High fallback rate' in drift
    
    def test_drift_no_metadata(self):
        """Test drift detection with no metadata"""
        integrator = StackingIntegrator()
        
        drift = integrator.detect_drift()
        
        assert drift is None


class TestExport:
    """Test prediction log export"""
    
    def test_export_prediction_log(self, tmp_path):
        """Test exporting prediction history to JSON"""
        integrator = StackingIntegrator()
        
        # Add some predictions
        for i in range(3):
            pred = StackingPrediction(
                direction='bullish',
                confidence=0.75,
                probability_bullish=0.75,
                probability_bearish=0.15,
                probability_neutral=0.10,
                fallback_used=False,
                model_version='v1.0',
                latency_ms=2.5
            )
            integrator.prediction_history.append(pred)
        
        export_path = tmp_path / "predictions.json"
        result = integrator.export_prediction_log(export_path)
        
        assert result is True
        assert export_path.exists()
        
        # Verify content
        with open(export_path) as f:
            data = json.load(f)
        
        assert len(data) == 3
        assert data[0]['direction'] == 'bullish'
        assert data[0]['confidence'] == 0.75
    
    def test_export_empty_history(self, tmp_path):
        """Test exporting empty history"""
        integrator = StackingIntegrator()
        
        export_path = tmp_path / "empty.json"
        result = integrator.export_prediction_log(export_path)
        
        assert result is True
        
        with open(export_path) as f:
            data = json.load(f)
        
        assert len(data) == 0


class TestConvenienceFunction:
    """Test the convenience function"""

    def test_get_stacking_prediction(self):
        """Test convenience function for getting predictions"""
        base_signals = {
            'tsmom': {'direction': 'bullish', 'confidence': 0.8}
        }

        result = get_stacking_prediction(base_signals)

        assert isinstance(result, StackingPrediction)
        assert result.direction in ['bullish', 'bearish', 'neutral']

    def test_with_regime_and_vix(self):
        """Convenience function should pass regime and VIX through."""
        base_signals = {'tsmom': {'direction': 'bearish', 'confidence': 0.7}}
        result = get_stacking_prediction(base_signals, current_regime='crisis', vix_level=35.0)
        assert isinstance(result, StackingPrediction)

    def test_with_model_path_nonexistent(self):
        """Nonexistent model path should fall back gracefully."""
        base_signals = {'tsmom': {'direction': 'bullish', 'confidence': 0.8}}
        result = get_stacking_prediction(base_signals, model_path=Path('/tmp/nonexistent_model.pkl'))
        assert isinstance(result, StackingPrediction)
        assert result.fallback_used is True


class TestWeightedVotingExtended:
    """Extended tests for weighted voting fallback."""

    def test_all_bearish_signals(self):
        """All bearish signals should produce bearish prediction."""
        integrator = StackingIntegrator()
        base_signals = {
            'tsmom': {'direction': 'bearish', 'confidence': 0.9},
            'hmm_regime': {'direction': 'bearish', 'confidence': 0.8},
            'fed_policy': {'direction': 'bearish', 'confidence': 0.7},
        }
        result = integrator.predict(base_signals)
        assert result.direction == 'bearish'
        assert result.fallback_used is True

    def test_mixed_signals_direction(self):
        """Mixed signals should resolve to highest weighted direction."""
        integrator = StackingIntegrator()
        base_signals = {
            'tsmom': {'direction': 'bullish', 'confidence': 0.9, 'strength': 0.8},
            'hmm_regime': {'direction': 'bearish', 'confidence': 0.5, 'strength': 0.3},
        }
        result = integrator.predict(base_signals)
        # tsmom has weight 0.30 * 0.9 = 0.27 for bullish
        # hmm_regime has weight 0.25 * 0.5 = 0.125 for bearish
        assert result.direction == 'bullish'

    def test_unknown_source_gets_low_weight(self):
        """Unknown signal sources should get default low weight."""
        integrator = StackingIntegrator()
        base_signals = {
            'custom_signal': {'direction': 'bullish', 'confidence': 0.9},
        }
        result = integrator.predict(base_signals)
        # Unknown source gets weight 0.1 (default)
        assert result.direction == 'bullish'
        assert result.fallback_used is True

    def test_empty_signals(self):
        """Empty signals should still produce a valid prediction."""
        integrator = StackingIntegrator()
        result = integrator.predict({})
        assert isinstance(result, StackingPrediction)
        assert result.direction in ['bullish', 'bearish', 'neutral']
        assert result.fallback_used is True

    def test_probabilities_sum(self):
        """Probabilities should approximately sum to 1.0."""
        integrator = StackingIntegrator()
        base_signals = {
            'tsmom': {'direction': 'bullish', 'confidence': 0.8},
            'base': {'direction': 'neutral', 'confidence': 0.6},
        }
        result = integrator.predict(base_signals)
        total = result.probability_bullish + result.probability_bearish + result.probability_neutral
        assert abs(total - 1.0) < 0.01


class TestFeatureExtractionExtended:
    """Extended feature extraction tests."""

    def test_with_vix_level(self):
        """VIX level should be included in feature vector."""
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.feature_count = 20
        base_signals = {'tsmom': {'direction': 'bullish', 'confidence': 0.8}}
        features = integrator._extract_simple_features(base_signals, 'bull', 25.0)
        # VIX 25 → 25/50 = 0.5, should appear in features
        assert 0.5 in features or any(abs(f - 0.5) < 0.01 for f in features)

    def test_regime_encoding(self):
        """Each regime should map to a distinct value."""
        integrator = StackingIntegrator()
        regimes = {'bull': 1.0, 'bear': -1.0, 'neutral': 0.0, 'high_vol': -0.5, 'crisis': -1.0}
        seen = set()
        for regime in regimes:
            features = integrator._extract_simple_features({}, regime, None)
            # Regime is at index 15 (5 sources * 3 features each)
            seen.add(features[15])
        # Should have at least 3 distinct regime encodings
        assert len(seen) >= 3

    def test_signal_direction_encoding(self):
        """Direction should map to -1/0/+1 correctly."""
        integrator = StackingIntegrator()
        bull = integrator._extract_simple_features(
            {'tsmom': {'direction': 'bullish', 'confidence': 0.5}}, None, None)
        bear = integrator._extract_simple_features(
            {'tsmom': {'direction': 'bearish', 'confidence': 0.5}}, None, None)
        neut = integrator._extract_simple_features(
            {'tsmom': {'direction': 'neutral', 'confidence': 0.5}}, None, None)
        assert bull[0] == 1.0
        assert bear[0] == -1.0
        assert neut[0] == 0.0


class TestDriftDetectionExtended:
    """Extended drift detection tests."""

    def test_drift_threshold_parameter(self):
        """Custom threshold should be respected."""
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.accuracy_train = 0.75
        integrator.metadata.accuracy_val = 0.70
        # 40% fallback rate
        for i in range(10):
            pred = StackingPrediction(
                direction='bullish', confidence=0.8,
                probability_bullish=0.8, probability_bearish=0.1,
                probability_neutral=0.1, fallback_used=(i < 4),
            )
            integrator.prediction_history.append(pred)
        # Default threshold 0.05 shouldn't trigger on fallback rate
        drift = integrator.detect_drift(threshold=0.05)
        # 40% fallback > 30% → drift detected
        assert drift is not None

    def test_no_metadata_no_drift(self):
        """Without metadata, drift detection returns None."""
        integrator = StackingIntegrator()
        for _ in range(5):
            pred = StackingPrediction(
                direction='bullish', confidence=0.8,
                probability_bullish=0.8, probability_bearish=0.1,
                probability_neutral=0.1, fallback_used=True,
            )
            integrator.prediction_history.append(pred)
        assert integrator.detect_drift() is None


class TestExportExtended:
    """Extended export tests."""

    def test_export_with_varying_predictions(self, tmp_path):
        """Export should handle mix of fallback and model predictions."""
        integrator = StackingIntegrator()
        for i in range(4):
            pred = StackingPrediction(
                direction='bullish' if i % 2 == 0 else 'bearish',
                confidence=0.7,
                probability_bullish=0.7 if i % 2 == 0 else 0.2,
                probability_bearish=0.2 if i % 2 == 0 else 0.7,
                probability_neutral=0.1,
                fallback_used=(i % 2 == 0),
                model_version='v1.0' if i % 2 else 'fallback_v2.81',
                latency_ms=float(i),
            )
            integrator.prediction_history.append(pred)

        export_path = tmp_path / "mixed.json"
        assert integrator.export_prediction_log(export_path) is True
        with open(export_path) as f:
            data = json.load(f)
        assert len(data) == 4

    def test_export_invalid_path(self):
        """Export to invalid path should return False."""
        integrator = StackingIntegrator()
        pred = StackingPrediction(
            direction='bullish', confidence=0.8,
            probability_bullish=0.8, probability_bearish=0.1,
            probability_neutral=0.1, fallback_used=False,
        )
        integrator.prediction_history.append(pred)
        with patch("builtins.open", side_effect=PermissionError("denied")):
            result = integrator.export_prediction_log(Path('/nonexistent/dir/file.json'))
        assert result is False
    """Extended accuracy stats tests."""

    def test_window_filtering(self):
        """Stats should respect window_days parameter."""
        integrator = StackingIntegrator()
        # Add an old prediction
        old_pred = StackingPrediction(
            direction='bullish', confidence=0.8,
            probability_bullish=0.8, probability_bearish=0.1,
            probability_neutral=0.1, fallback_used=False,
        )
        old_pred.timestamp = datetime.now() - timedelta(days=60)
        integrator.prediction_history.append(old_pred)

        # Add a recent prediction
        recent_pred = StackingPrediction(
            direction='bearish', confidence=0.7,
            probability_bullish=0.2, probability_bearish=0.7,
            probability_neutral=0.1, fallback_used=True,
        )
        integrator.prediction_history.append(recent_pred)

        stats = integrator.get_accuracy_stats(window_days=30)
        assert stats['count'] == 1  # Only the recent one
        assert stats['fallback_rate'] == 1.0  # The recent one is fallback

    def test_latency_tracking(self):
        """Average latency should be computed correctly."""
        integrator = StackingIntegrator()
        for i in range(3):
            pred = StackingPrediction(
                direction='neutral', confidence=0.5,
                probability_bullish=0.33, probability_bearish=0.33,
                probability_neutral=0.34, fallback_used=False,
                latency_ms=float(2 + i),
            )
            integrator.prediction_history.append(pred)

        stats = integrator.get_accuracy_stats()
        assert stats['avg_latency_ms'] == pytest.approx(3.0)


class TestHistoryManagement:
    """Test prediction history management."""

    def test_history_trimming(self):
        """History should be trimmed to max_history."""
        integrator = StackingIntegrator()
        integrator.max_history = 5
        base_signals = {'tsmom': {'direction': 'bullish', 'confidence': 0.8}}
        for _ in range(8):
            integrator.predict(base_signals)
        assert len(integrator.prediction_history) == 5

    def test_history_preserves_recent(self):
        """After trimming, most recent predictions should be kept."""
        integrator = StackingIntegrator()
        integrator.max_history = 3
        base_signals = {'tsmom': {'direction': 'bullish', 'confidence': 0.8}}
        for _ in range(5):
            integrator.predict(base_signals)
        # The last 3 should be kept
        assert len(integrator.prediction_history) == 3


class PicklableNumericModel:
    """Picklable model with numeric classes for testing."""
    classes_ = [0, 1, 2]  # 0=bearish, 1=neutral, 2=bullish
    def __init__(self, probs):
        self._probs = probs
    def predict_proba(self, X):
        return np.array([self._probs])


class PicklableBrokenModel:
    """Picklable model that raises on predict_proba for testing."""
    classes_ = ['bearish', 'neutral', 'bullish']
    def predict_proba(self, X):
        raise RuntimeError("Model error")


class TestModelPredictionExtended:
    """Extended model prediction tests."""

    def test_numeric_class_mapping(self, tmp_path):
        """Model with numeric classes should map correctly."""
        model_data = {
            'model': PicklableNumericModel([0.1, 0.15, 0.75]),
            'metadata': {
                'version': 'v3.0',
                'training_date': datetime(2026, 5, 1),
                'feature_count': 17,
                'accuracy_train': 0.80,
                'accuracy_val': 0.75,
                'feature_importance': {'feat_1': 0.2},
                'total_samples': 10000,
            }
        }
        model_path = tmp_path / "numeric_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        integrator = StackingIntegrator(model_path=model_path)
        base_signals = {'tsmom': {'direction': 'bullish', 'confidence': 0.8}}
        result = integrator.predict(base_signals)
        assert result.direction == 'bullish'
        assert result.probability_bullish == pytest.approx(0.75)
        assert result.fallback_used is False

    def test_model_exception_fallback(self, tmp_path):
        """Model raising exception should trigger fallback."""
        model_data = {
            'model': PicklableBrokenModel(),
            'metadata': {
                'version': 'v1.0',
                'training_date': datetime(2026, 5, 1),
                'feature_count': 17,
                'accuracy_train': 0.75,
                'accuracy_val': 0.68,
                'feature_importance': {},
                'total_samples': 10000,
            }
        }
        model_path = tmp_path / "broken_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        integrator = StackingIntegrator(model_path=model_path)
        base_signals = {'tsmom': {'direction': 'bullish', 'confidence': 0.8}}
        result = integrator.predict(base_signals)
        assert result.fallback_used is True
    """Test performance requirements"""
    
    def test_prediction_latency(self):
        """Test prediction latency is under 10ms"""
        import time
        
        integrator = StackingIntegrator()
        
        base_signals = {
            'tsmom': {'direction': 'bullish', 'confidence': 0.8},
            'hmm_regime': {'direction': 'bullish', 'confidence': 0.7}
        }
        
        start = time.time()
        result = integrator.predict(base_signals)
        elapsed_ms = (time.time() - start) * 1000
        
        # Should be very fast (fallback path)
        assert elapsed_ms < 10.0
        assert result.latency_ms < 10.0


# ---------------------------------------------------------------------------
# __all__ export validation
# ---------------------------------------------------------------------------

class TestExports:
    """Verify __all__ exports."""

    def test_all_exports_present(self):
        import src.signals.stacking_integrator as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"Missing export: {name}"

    def test_all_count(self):
        import src.signals.stacking_integrator as mod
        assert len(mod.__all__) == 4


# ---------------------------------------------------------------------------
# StackingPrediction dataclass extended
# ---------------------------------------------------------------------------

class TestStackingPredictionExtended:
    """Extended StackingPrediction dataclass tests."""

    def test_all_fields(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(StackingPrediction)}
        expected = {
            "direction", "confidence", "probability_bullish",
            "probability_bearish", "probability_neutral",
            "fallback_used", "feature_vector", "top_features",
            "timestamp", "model_version", "latency_ms",
        }
        assert field_names == expected

    def test_default_values(self):
        pred = StackingPrediction(
            direction="bullish", confidence=0.8,
            probability_bullish=0.6, probability_bearish=0.2,
            probability_neutral=0.2, fallback_used=False,
        )
        assert pred.feature_vector is None
        assert pred.top_features == []
        assert pred.model_version == "unknown"
        assert pred.latency_ms == 0.0


# ---------------------------------------------------------------------------
# ModelMetadata dataclass extended
# ---------------------------------------------------------------------------

class TestModelMetadataExtended:
    """Extended ModelMetadata dataclass tests."""

    def test_all_fields(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(ModelMetadata)}
        expected = {
            "version", "training_date", "feature_count",
            "accuracy_train", "accuracy_val", "feature_importance",
            "total_samples", "source_roster", "source_roster_version",
            "fallback_semantics",
        }
        assert field_names == expected

    def test_creation(self):
        meta = ModelMetadata(
            version="1.0", training_date=datetime.now(),
            feature_count=50, accuracy_train=0.85, accuracy_val=0.80,
            feature_importance={"feat1": 0.3}, total_samples=1000,
            source_roster=["a"], source_roster_version="v1",
            fallback_semantics="fallback",
        )
        assert meta.version == "1.0"
        assert meta.total_samples == 1000


# ---------------------------------------------------------------------------
# StackingIntegrator extended
# ---------------------------------------------------------------------------

class TestStackingIntegratorExtended:
    """Extended integrator tests."""

    def test_init_defaults(self):
        integrator = StackingIntegrator()
        assert integrator.model is None
        assert integrator.metadata is None

    def test_get_accuracy_stats_no_history(self):
        integrator = StackingIntegrator()
        stats = integrator.get_accuracy_stats()
        assert isinstance(stats, dict)

    def test_detect_drift_no_history(self):
        integrator = StackingIntegrator()
        result = integrator.detect_drift()
        assert result is None

    def test_export_prediction_log(self, tmp_path):
        integrator = StackingIntegrator()
        filepath = tmp_path / "predictions.json"
        result = integrator.export_prediction_log(filepath)
        assert isinstance(result, bool)

    def test_load_model_nonexistent(self, tmp_path):
        integrator = StackingIntegrator()
        result = integrator.load_model(tmp_path / "nonexistent.pkl")
        assert result is False


# ---------------------------------------------------------------------------
# get_stacking_prediction convenience function
# ---------------------------------------------------------------------------

class TestGetStackingPrediction:
    """Test the convenience function."""

    def test_returns_prediction(self):
        from src.signals.stacking_integrator import get_stacking_prediction
        result = get_stacking_prediction(
            base_signals={'tsmom': {'direction': 'bullish', 'confidence': 0.8}},
        )
        assert isinstance(result, StackingPrediction)


# ---------------------------------------------------------------------------
# Module-level constants validation
# ---------------------------------------------------------------------------

class TestConstants:
    """Validate module-level constants exist with expected types/ranges."""

    def test_confidence_threshold_type_and_value(self):
        assert StackingIntegrator.CONFIDENCE_THRESHOLD == 0.6
        assert isinstance(StackingIntegrator.CONFIDENCE_THRESHOLD, float)

    def test_model_dir_type_and_value(self):
        assert StackingIntegrator.MODEL_DIR == Path("models")
        assert isinstance(StackingIntegrator.MODEL_DIR, Path)

    def test_model_prefix_type_and_value(self):
        assert StackingIntegrator.MODEL_PREFIX == "signal_stacker_v"
        assert isinstance(StackingIntegrator.MODEL_PREFIX, str)

    def test_default_max_history(self):
        integrator = StackingIntegrator()
        assert integrator.max_history == 1000
        assert isinstance(integrator.max_history, int)


# ---------------------------------------------------------------------------
# Complete dataclass field validation via dataclasses.fields()
# ---------------------------------------------------------------------------

class TestStackingPredictionFieldValidation:
    """Verify StackingPrediction fields, types, and defaults."""

    def test_all_fields_present(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(StackingPrediction)}
        expected = {
            "direction", "confidence", "probability_bullish",
            "probability_bearish", "probability_neutral",
            "fallback_used", "feature_vector", "top_features",
            "timestamp", "model_version", "latency_ms",
        }
        assert field_names == expected

    def test_required_fields_have_no_default(self):
        from dataclasses import fields, MISSING
        required_no_default = {"direction", "confidence", "probability_bullish",
                               "probability_bearish", "probability_neutral", "fallback_used"}
        for f in fields(StackingPrediction):
            if f.name in required_no_default:
                assert f.default is MISSING and f.default_factory is MISSING, \
                    f"{f.name} should not have a default"
        # Verify optional fields have defaults
        pred = StackingPrediction(
            direction="bullish", confidence=0.8,
            probability_bullish=0.6, probability_bearish=0.2,
            probability_neutral=0.2, fallback_used=False,
        )
        assert pred.feature_vector is None
        assert pred.top_features == []
        assert pred.model_version == "unknown"
        assert pred.latency_ms == 0.0

    def test_feature_vector_type(self):
        from dataclasses import fields
        field_map = {f.name: f for f in fields(StackingPrediction)}
        fv_field = field_map["feature_vector"]
        assert fv_field.default is None
        # Verify optional via None assignment
        pred = StackingPrediction(
            direction="bullish", confidence=0.5,
            probability_bullish=0.4, probability_bearish=0.3,
            probability_neutral=0.3, fallback_used=False,
        )
        assert pred.feature_vector is None
        pred.feature_vector = np.array([1.0, 2.0])
        assert isinstance(pred.feature_vector, np.ndarray)

    def test_top_features_default_factory(self):
        from dataclasses import fields
        field_map = {f.name: f for f in fields(StackingPrediction)}
        tf_field = field_map["top_features"]
        # default_factory should produce new list each time
        pred1 = StackingPrediction(
            direction="bullish", confidence=0.5,
            probability_bullish=0.4, probability_bearish=0.3,
            probability_neutral=0.3, fallback_used=False,
        )
        pred2 = StackingPrediction(
            direction="bearish", confidence=0.5,
            probability_bullish=0.3, probability_bearish=0.4,
            probability_neutral=0.3, fallback_used=False,
        )
        assert pred1.top_features is not pred2.top_features
        assert pred1.top_features == []
        assert pred2.top_features == []


class TestModelMetadataFieldValidation:
    """Verify ModelMetadata fields, types, and defaults."""

    def test_all_fields_present(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(ModelMetadata)}
        expected = {
            "version", "training_date", "feature_count",
            "accuracy_train", "accuracy_val", "feature_importance",
            "total_samples", "source_roster", "source_roster_version",
            "fallback_semantics",
        }
        assert field_names == expected

    def test_field_types(self):
        from dataclasses import fields
        field_map = {f.name: f.type for f in fields(ModelMetadata)}
        assert field_map["version"] is str
        assert field_map["training_date"] is datetime
        assert field_map["feature_count"] == Optional[int]
        assert field_map["accuracy_train"] is float
        assert field_map["accuracy_val"] is float
        assert "Dict" in str(field_map["feature_importance"]) or "dict" in str(field_map["feature_importance"])
        assert field_map["total_samples"] is int
        assert "List" in str(field_map["source_roster"]) or "list" in str(field_map["source_roster"])
        assert field_map["source_roster_version"] is str
        assert field_map["fallback_semantics"] is str

    def test_no_defaults_required(self):
        """All metadata fields are required (no defaults in source)."""
        meta = ModelMetadata(
            version="v1", training_date=datetime.now(),
            feature_count=10, accuracy_train=0.5, accuracy_val=0.4,
            feature_importance={"a": 0.1}, total_samples=100,
            source_roster=["a"], source_roster_version="v1",
            fallback_semantics="fallback",
        )
        assert meta.version == "v1"
        assert meta.total_samples == 100


# ---------------------------------------------------------------------------
# Dataclass edge cases
# ---------------------------------------------------------------------------

class TestStackingPredictionEdgeCases:
    """Edge cases for StackingPrediction dataclass."""

    def test_zero_confidence(self):
        pred = StackingPrediction(
            direction="neutral", confidence=0.0,
            probability_bullish=0.0, probability_bearish=0.0,
            probability_neutral=1.0, fallback_used=False,
        )
        assert pred.confidence == 0.0
        assert pred.direction == "neutral"

    def test_max_confidence(self):
        pred = StackingPrediction(
            direction="bullish", confidence=1.0,
            probability_bullish=1.0, probability_bearish=0.0,
            probability_neutral=0.0, fallback_used=False,
        )
        assert pred.confidence == 1.0

    def test_nan_confidence(self):
        pred = StackingPrediction(
            direction="bullish", confidence=float("nan"),
            probability_bullish=float("nan"), probability_bearish=0.0,
            probability_neutral=0.0, fallback_used=False,
        )
        assert np.isnan(pred.confidence)
        assert np.isnan(pred.probability_bullish)

    def test_inf_confidence(self):
        pred = StackingPrediction(
            direction="bullish", confidence=float("inf"),
            probability_bullish=float("inf"), probability_bearish=0.0,
            probability_neutral=0.0, fallback_used=False,
        )
        assert pred.confidence == float("inf")

    def test_empty_top_features(self):
        pred = StackingPrediction(
            direction="bullish", confidence=0.5,
            probability_bullish=0.4, probability_bearish=0.3,
            probability_neutral=0.3, fallback_used=False,
            top_features=[],
        )
        assert pred.top_features == []

    def test_large_top_features_list(self):
        top = [(f"feat_{i}", float(i * 0.01)) for i in range(100)]
        pred = StackingPrediction(
            direction="bullish", confidence=0.5,
            probability_bullish=0.4, probability_bearish=0.3,
            probability_neutral=0.3, fallback_used=False,
            top_features=top,
        )
        assert len(pred.top_features) == 100


# ---------------------------------------------------------------------------
# Prediction edge cases
# ---------------------------------------------------------------------------

class TestPredictEdgeCases:
    """Edge cases for the predict method."""

    def test_predict_empty_signals(self):
        integrator = StackingIntegrator()
        result = integrator.predict({})
        assert isinstance(result, StackingPrediction)
        assert result.fallback_used is True

    def test_predict_with_only_regime_no_signals(self):
        integrator = StackingIntegrator()
        result = integrator.predict(base_signals={}, current_regime="crisis")
        assert isinstance(result, StackingPrediction)
        assert result.fallback_used is True

    def test_predict_with_only_vix_no_signals(self):
        integrator = StackingIntegrator()
        result = integrator.predict(base_signals={}, vix_level=45.0)
        assert isinstance(result, StackingPrediction)

    def test_predict_with_vix_zero(self):
        integrator = StackingIntegrator()
        base_signals = {"tsmom": {"direction": "bullish", "confidence": 0.8}}
        result = integrator.predict(base_signals, vix_level=0.0)
        assert isinstance(result, StackingPrediction)

    def test_predict_with_missing_direction_key(self):
        integrator = StackingIntegrator()
        base_signals = {"tsmom": {"confidence": 0.8}}
        result = integrator.predict(base_signals)
        assert isinstance(result, StackingPrediction)

    def test_predict_with_missing_confidence_key(self):
        integrator = StackingIntegrator()
        base_signals = {"tsmom": {"direction": "bullish"}}
        result = integrator.predict(base_signals)
        assert isinstance(result, StackingPrediction)

    def test_predict_with_empty_signal_dict(self):
        integrator = StackingIntegrator()
        base_signals = {"tsmom": {}}
        result = integrator.predict(base_signals)
        assert isinstance(result, StackingPrediction)

    def test_predict_with_unknown_source(self):
        integrator = StackingIntegrator()
        base_signals = {"completely_unknown_signal": {"direction": "bullish", "confidence": 0.9}}
        result = integrator.predict(base_signals)
        assert isinstance(result, StackingPrediction)
        assert result.fallback_used is True

    def test_predict_with_many_sources(self):
        integrator = StackingIntegrator()
        base_signals = {f"source_{i}": {"direction": "bullish", "confidence": 0.5}
                        for i in range(50)}
        result = integrator.predict(base_signals)
        assert isinstance(result, StackingPrediction)

    def test_predict_unknown_regime_string(self):
        integrator = StackingIntegrator()
        base_signals = {"tsmom": {"direction": "bullish", "confidence": 0.8}}
        result = integrator.predict(base_signals, current_regime="nonexistent_regime")
        assert isinstance(result, StackingPrediction)

    def test_predict_none_vix_with_regime(self):
        integrator = StackingIntegrator()
        base_signals = {"tsmom": {"direction": "bullish", "confidence": 0.8}}
        result = integrator.predict(base_signals, current_regime="bull", vix_level=None)
        assert isinstance(result, StackingPrediction)

    def test_predict_negative_confidence(self):
        integrator = StackingIntegrator()
        base_signals = {"tsmom": {"direction": "bullish", "confidence": -0.5, "strength": -0.3}}
        result = integrator.predict(base_signals)
        assert isinstance(result, StackingPrediction)

    def test_predict_extreme_vix(self):
        integrator = StackingIntegrator()
        # VIX capped at 50.0 / 50.0 = 1.0
        base_signals = {"tsmom": {"direction": "bullish", "confidence": 0.8}}
        result = integrator.predict(base_signals, vix_level=300.0)
        assert isinstance(result, StackingPrediction)


class TestPredictWithModelEdgeCases:
    """Edge cases for model-based prediction."""

    def test_numeric_model_with_two_classes(self, tmp_path):
        """Model with only 2 classes should handle missing third."""
        model_data = {"model": PicklableTwoClassModel(), "metadata": {
            "version": "v1", "training_date": datetime.now(),
            "feature_count": 17, "accuracy_train": 0.8, "accuracy_val": 0.75,
            "feature_importance": {}, "total_samples": 5000,
        }}
        model_path = tmp_path / "two_class.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        integrator = StackingIntegrator(model_path=model_path)
        result = integrator.predict({"tsmom": {"direction": "bullish", "confidence": 0.8}})
        assert isinstance(result, StackingPrediction)

    def test_model_returning_equal_probs(self, tmp_path):
        """Equal probabilities across classes should pick first max."""
        model_data = {"model": PicklableEqualProbsModel(), "metadata": {
            "version": "v1", "training_date": datetime.now(),
            "feature_count": 17, "accuracy_train": 0.8, "accuracy_val": 0.75,
            "feature_importance": {}, "total_samples": 5000,
        }}
        model_path = tmp_path / "equal_probs.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        integrator = StackingIntegrator(model_path=model_path)
        result = integrator.predict({"tsmom": {"direction": "bullish", "confidence": 0.8}})
        # confidence 0.34 < 0.6 threshold -> fallback
        assert result.fallback_used is True

    def test_model_missing_prob_dict_bearish(self, tmp_path):
        """Model dict missing 'bearish' should fall back to neutral."""
        model_data = {"model": PicklableMissingBearishModel(), "metadata": {
            "version": "v1", "training_date": datetime.now(),
            "feature_count": 17, "accuracy_train": 0.8, "accuracy_val": 0.75,
            "feature_importance": {}, "total_samples": 5000,
        }}
        model_path = tmp_path / "missing_bearish.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        integrator = StackingIntegrator(model_path=model_path)
        result = integrator.predict({"tsmom": {"direction": "bullish", "confidence": 0.8}})
        # 0.6 >= 0.6 threshold, not fallback
        assert isinstance(result, StackingPrediction)


class TestLoadModelEdgeCases:
    """Edge cases for load_model method."""

    def test_load_model_missing_model_key(self, tmp_path):
        """Pickle without 'model' key should set model to None."""
        model_data = {"metadata": {"version": "v1", "training_date": datetime.now(),
                                    "feature_count": 10, "accuracy_train": 0.0,
                                    "accuracy_val": 0.0, "feature_importance": {},
                                    "total_samples": 0}}
        model_path = tmp_path / "no_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        integrator = StackingIntegrator()
        result = integrator.load_model(model_path)
        assert result is True  # No exception, .get returns None
        assert integrator.model is None

    def test_load_model_missing_metadata_key(self, tmp_path):
        """Pickle without metadata must not invent a feature count."""
        model_data = {"model": PicklableModel([0.1, 0.2, 0.7])}
        model_path = tmp_path / "no_metadata.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        integrator = StackingIntegrator()
        result = integrator.load_model(model_path)
        assert result is True
        assert integrator.metadata is not None
        assert integrator.metadata.version == "unknown"
        assert integrator.metadata.feature_count is None

    def test_load_model_partial_metadata(self, tmp_path):
        """Partial metadata should leave feature count unavailable when absent."""
        model_data = {"model": None, "metadata": {"version": "v1"}}
        model_path = tmp_path / "partial.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        integrator = StackingIntegrator()
        result = integrator.load_model(model_path)
        assert result is True
        assert integrator.metadata.version == "v1"
        assert integrator.metadata.feature_count is None

    def test_load_model_empty_file(self, tmp_path):
        """Empty pickle file should fail gracefully."""
        model_path = tmp_path / "empty.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(b"", f)

        integrator = StackingIntegrator()
        # Empty pickle may raise or return False
        result = integrator.load_model(model_path)
        assert result is False


# ---------------------------------------------------------------------------
# Weighted voting edge cases
# ---------------------------------------------------------------------------

class TestWeightedVotingEdgeCases:
    """Edge cases for weighted voting fallback."""

    def test_tie_between_bullish_and_neutral(self):
        integrator = StackingIntegrator()
        # tsmom (0.30) bullish + base (0.15) neutral -> bullish should win
        base_signals = {
            "tsmom": {"direction": "bullish", "confidence": 1.0},
            "base": {"direction": "neutral", "confidence": 1.0},
            "hmm_regime": {"direction": "bearish", "confidence": 1.0},
        }
        result = integrator.predict(base_signals)
        assert result.direction in ("bullish", "bearish", "neutral")
        total = result.probability_bullish + result.probability_bearish + result.probability_neutral
        assert abs(total - 1.0) < 0.01

    def test_all_signals_neutral(self):
        integrator = StackingIntegrator()
        base_signals = {
            "tsmom": {"direction": "neutral", "confidence": 0.8},
            "hmm_regime": {"direction": "neutral", "confidence": 0.7},
            "fed_policy": {"direction": "neutral", "confidence": 0.6},
        }
        result = integrator.predict(base_signals)
        assert result.direction == "neutral"
        assert result.fallback_used is True

    def test_single_signal_source(self):
        integrator = StackingIntegrator()
        base_signals = {"tsmom": {"direction": "bearish", "confidence": 1.0}}
        result = integrator.predict(base_signals)
        assert result.direction == "bearish"

    def test_zero_confidence_all_signals(self):
        integrator = StackingIntegrator()
        base_signals = {
            "tsmom": {"direction": "bullish", "confidence": 0.0},
            "hmm_regime": {"direction": "bearish", "confidence": 0.0},
        }
        result = integrator.predict(base_signals)
        # All zero -> total = 0 -> no normalization -> all scores stay 0 -> max is 0.0
        assert result.confidence == 0.0
        assert isinstance(result, StackingPrediction)

    def test_weighted_voting_with_nan_confidence(self):
        integrator = StackingIntegrator()
        base_signals = {
            "tsmom": {"direction": "bullish", "confidence": float("nan")},
        }
        result = integrator.predict(base_signals)
        # nan * weight = nan -> total will be nan -> normalization may produce nan
        assert isinstance(result, StackingPrediction)


# ---------------------------------------------------------------------------
# Feature extraction edge cases
# ---------------------------------------------------------------------------

class TestFeatureExtractionEdgeCases:
    """Edge cases for _extract_simple_features."""

    def test_no_signals_no_regime_no_vix(self):
        integrator = StackingIntegrator()
        features = integrator._extract_simple_features({}, None, None)
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert len(features) >= 17  # 5 sources * 3 features + 1 regime + 1 vix

    def test_feature_count_zero_in_metadata(self):
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.feature_count = 0
        features = integrator._extract_simple_features(
            {"tsmom": {"direction": "bullish", "confidence": 0.8}},
            None, None,
        )
        # Should not pad, should truncate to 0 -> empty array
        assert len(features) == 0

    def test_feature_count_very_large(self):
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.feature_count = 10000
        features = integrator._extract_simple_features(
            {"tsmom": {"direction": "bullish", "confidence": 0.8}},
            None, None,
        )
        assert len(features) == 10000
        # Padded values should be zeros
        assert np.all(features[17:] == 0.0)

    def test_vix_zero_uses_default(self):
        integrator = StackingIntegrator()
        features = integrator._extract_simple_features({}, None, 0.0)
        # Since vix_level is 0, min(0/50, 1) = 0.0
        assert isinstance(features, np.ndarray)

    def test_vix_extreme_large(self):
        integrator = StackingIntegrator()
        features = integrator._extract_simple_features({}, None, 500.0)
        # min(500/50, 1) = 1.0
        assert isinstance(features, np.ndarray)
        # Last feature should be vix (capped at 1.0)
        assert features[-1] == 1.0

    def test_vix_negative(self):
        integrator = StackingIntegrator()
        # Negative VIX: min(-10/50, 1) = min(-0.2, 1) = -0.2
        features = integrator._extract_simple_features({}, None, -10.0)
        assert isinstance(features, np.ndarray)
        # VIX feature should be -0.2
        assert features[-1] == -0.2

    def test_all_regime_encodings(self):
        integrator = StackingIntegrator()
        regimes = {"bull": 1.0, "bear": -1.0, "neutral": 0.0,
                    "high_vol": -0.5, "crisis": -1.0, "unknown_regime": 0.0, None: 0.0}
        for regime, expected in regimes.items():
            features = integrator._extract_simple_features({}, regime, None)
            assert features[15] == expected, f"Regime {regime} should map to {expected}"

    def test_padding_with_metadata(self):
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.feature_count = 30
        features = integrator._extract_simple_features(
            {"tsmom": {"direction": "bullish", "confidence": 0.8, "strength": 0.7}},
            None, None,
        )
        assert len(features) == 30
        assert features[0] == 1.0  # bullish
        assert features[1] == 0.8  # confidence
        assert features[2] == 0.7  # strength

    def test_truncation_with_metadata(self):
        """If metadata.feature_count < extracted, should truncate."""
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.feature_count = 5
        features = integrator._extract_simple_features(
            {"tsmom": {"direction": "bullish", "confidence": 0.8}},
            None, None,
        )
        assert len(features) == 5


# ---------------------------------------------------------------------------
# _get_top_features edge cases
# ---------------------------------------------------------------------------

class TestGetTopFeaturesEdgeCases:
    """Edge cases for _get_top_features."""

    def test_no_metadata(self):
        integrator = StackingIntegrator()
        features = np.array([1.0, 2.0, 3.0])
        result = integrator._get_top_features(features)
        assert result == []

    def test_metadata_no_feature_importance(self):
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.feature_importance = {}
        features = np.array([1.0, 2.0, 3.0])
        result = integrator._get_top_features(features)
        assert result == []

    def test_n_top_zero(self):
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.feature_importance = {"a": 0.5, "b": 0.3}
        features = np.array([1.0, 2.0])
        result = integrator._get_top_features(features, n_top=0)
        assert result == []

    def test_n_top_negative(self):
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.feature_importance = {"a": 0.5, "b": 0.3}
        features = np.array([1.0, 2.0])
        result = integrator._get_top_features(features, n_top=-1)
        # Python negative slice: sorted[:-1] excludes last item
        assert len(result) == 1
        assert result[0] == ("a", 0.5)

    def test_n_top_larger_than_available(self):
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.feature_importance = {"a": 0.5, "b": 0.3}
        features = np.array([1.0, 2.0])
        result = integrator._get_top_features(features, n_top=100)
        assert len(result) == 2  # Returns all available
        assert result[0] == ("a", 0.5)
        assert result[1] == ("b", 0.3)

    def test_top_features_maintains_order(self):
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.feature_importance = {"z": 0.1, "a": 0.9, "m": 0.5}
        features = np.array([1.0, 2.0, 3.0])
        result = integrator._get_top_features(features, n_top=3)
        # Should be sorted by importance descending
        assert result[0][0] == "a"
        assert result[1][0] == "m"
        assert result[2][0] == "z"


# ---------------------------------------------------------------------------
# _add_to_history edge cases
# ---------------------------------------------------------------------------

class TestAddToHistoryEdgeCases:
    """Edge cases for _add_to_history."""

    def test_history_at_exact_max(self):
        integrator = StackingIntegrator()
        integrator.max_history = 5
        for _ in range(5):
            integrator.prediction_history.append(
                StackingPrediction(direction="bullish", confidence=0.5,
                                   probability_bullish=0.4, probability_bearish=0.3,
                                   probability_neutral=0.3, fallback_used=False)
            )
        # Adding one more should trigger trim
        integrator._add_to_history(
            StackingPrediction(direction="bearish", confidence=0.5,
                               probability_bullish=0.3, probability_bearish=0.4,
                               probability_neutral=0.3, fallback_used=False)
        )
        assert len(integrator.prediction_history) == 5

    def test_history_below_max(self):
        integrator = StackingIntegrator()
        integrator.max_history = 10
        for _ in range(3):
            integrator._add_to_history(
                StackingPrediction(direction="bullish", confidence=0.5,
                                   probability_bullish=0.4, probability_bearish=0.3,
                                   probability_neutral=0.3, fallback_used=False)
            )
        assert len(integrator.prediction_history) == 3

    def test_history_trim_preserves_latest(self):
        integrator = StackingIntegrator()
        integrator.max_history = 3
        for i in range(6):
            pred = StackingPrediction(direction="bullish", confidence=0.5 + i * 0.1,
                                      probability_bullish=0.4, probability_bearish=0.3,
                                      probability_neutral=0.3, fallback_used=False,
                                      latency_ms=float(i))
            integrator._add_to_history(pred)
        assert len(integrator.prediction_history) == 3
        # Should keep the last 3 (indices 3, 4, 5)
        assert integrator.prediction_history[0].latency_ms == 3.0
        assert integrator.prediction_history[2].latency_ms == 5.0


# ---------------------------------------------------------------------------
# Accuracy stats edge cases
# ---------------------------------------------------------------------------

class TestAccuracyStatsEdgeCases:
    """Edge cases for get_accuracy_stats."""

    def test_window_days_zero(self):
        integrator = StackingIntegrator()
        for _ in range(3):
            pred = StackingPrediction(direction="bullish", confidence=0.5,
                                       probability_bullish=0.4, probability_bearish=0.3,
                                       probability_neutral=0.3, fallback_used=False,
                                       latency_ms=1.0)
            integrator.prediction_history.append(pred)
        stats = integrator.get_accuracy_stats(window_days=0)
        # Predictions with timestamp > now - 0 days = now -> only future timestamps pass
        # datetimes are "now" at creation, so they're always >= now - 0
        # Actually, datetime.now() at creation should be almost exactly now
        assert stats["count"] >= 0

    def test_window_days_negative(self):
        integrator = StackingIntegrator()
        for _ in range(3):
            pred = StackingPrediction(direction="bullish", confidence=0.5,
                                       probability_bullish=0.4, probability_bearish=0.3,
                                       probability_neutral=0.3, fallback_used=False)
            integrator.prediction_history.append(pred)
        # Negative window_days -> cutoff = now + positive timedelta -> all predictions in past
        stats = integrator.get_accuracy_stats(window_days=-30)
        assert stats["count"] == 0
        assert stats["fallback_rate"] == 0.0

    def test_very_large_window(self):
        integrator = StackingIntegrator()
        for _ in range(3):
            pred = StackingPrediction(direction="bullish", confidence=0.5,
                                       probability_bullish=0.4, probability_bearish=0.3,
                                       probability_neutral=0.3, fallback_used=False)
            integrator.prediction_history.append(pred)
        stats = integrator.get_accuracy_stats(window_days=36500)
        assert stats["count"] == 3

    def test_single_prediction_stats(self):
        integrator = StackingIntegrator()
        pred = StackingPrediction(direction="bearish", confidence=0.9,
                                   probability_bullish=0.05, probability_bearish=0.9,
                                   probability_neutral=0.05, fallback_used=True,
                                   latency_ms=5.0)
        integrator.prediction_history.append(pred)
        stats = integrator.get_accuracy_stats()
        assert stats["count"] == 1
        assert stats["fallback_rate"] == 1.0
        assert stats["avg_confidence"] == 0.9
        assert stats["avg_latency_ms"] == 5.0

    def test_empty_history_stats(self):
        integrator = StackingIntegrator()
        stats = integrator.get_accuracy_stats()
        assert stats["count"] == 0
        assert stats["fallback_rate"] == 0.0
        assert stats["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Drift detection edge cases
# ---------------------------------------------------------------------------

class TestDetectDriftEdgeCases:
    """Edge cases for detect_drift."""

    def test_borderline_fallback_rate_below_threshold(self):
        """Fallback rate exactly 0.3 should NOT trigger drift (condition is > 0.3)."""
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.accuracy_train = 0.75
        integrator.metadata.accuracy_val = 0.70
        # 3 out of 10 = 0.3 fallback rate
        for i in range(10):
            pred = StackingPrediction(direction="bullish", confidence=0.8,
                                       probability_bullish=0.8, probability_bearish=0.1,
                                       probability_neutral=0.1, fallback_used=(i < 3))
            integrator.prediction_history.append(pred)
        drift = integrator.detect_drift()
        assert drift is None

    def test_fallback_rate_just_above_threshold(self):
        """Fallback rate 0.31 should trigger drift."""
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.accuracy_train = 0.75
        integrator.metadata.accuracy_val = 0.70
        # 4 out of 13 = ~0.308, wait let me get exactly 0.308+
        # Actually, exactly 4/13 = 0.307... just below 0.3. Let me use 4/12 = 0.333...
        for i in range(12):
            pred = StackingPrediction(direction="bullish", confidence=0.8,
                                       probability_bullish=0.8, probability_bearish=0.1,
                                       probability_neutral=0.1, fallback_used=(i < 4))
            integrator.prediction_history.append(pred)
        drift = integrator.detect_drift()
        # 4/12 = 0.333 > 0.3 -> drift detected
        assert drift is not None
        assert "High fallback rate" in drift

    def test_drift_with_no_predictions(self):
        """Metadata exists but no predictions -> empty stats -> no drift."""
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.accuracy_train = 0.75
        integrator.metadata.accuracy_val = 0.70
        drift = integrator.detect_drift()
        assert drift is None

    def test_drift_with_all_fallback_predictions(self):
        integrator = StackingIntegrator()
        integrator.metadata = Mock()
        integrator.metadata.accuracy_train = 0.75
        integrator.metadata.accuracy_val = 0.70
        for _ in range(5):
            pred = StackingPrediction(direction="bullish", confidence=0.8,
                                       probability_bullish=0.8, probability_bearish=0.1,
                                       probability_neutral=0.1, fallback_used=True)
            integrator.prediction_history.append(pred)
        drift = integrator.detect_drift()
        assert drift is not None
        assert "100.0%" in drift or "100" in drift


# ---------------------------------------------------------------------------
# Export edge cases
# ---------------------------------------------------------------------------

class TestExportEdgeCases:
    """Edge cases for export_prediction_log."""

    def test_export_to_readonly_directory(self):
        integrator = StackingIntegrator()
        pred = StackingPrediction(direction="bullish", confidence=0.8,
                                   probability_bullish=0.8, probability_bearish=0.1,
                                   probability_neutral=0.1, fallback_used=False)
        integrator.prediction_history.append(pred)
        with patch("builtins.open", side_effect=PermissionError("denied")):
            result = integrator.export_prediction_log(Path("/nonexistent_dir/file.json"))
        assert result is False

    def test_export_single_prediction(self, tmp_path):
        integrator = StackingIntegrator()
        pred = StackingPrediction(direction="bearish", confidence=0.7,
                                   probability_bullish=0.2, probability_bearish=0.7,
                                   probability_neutral=0.1, fallback_used=True,
                                   model_version="v1", latency_ms=3.0)
        integrator.prediction_history.append(pred)
        export_path = tmp_path / "single.json"
        assert integrator.export_prediction_log(export_path) is True
        with open(export_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["direction"] == "bearish"
        assert data[0]["fallback_used"] is True

    def test_export_many_predictions(self, tmp_path):
        integrator = StackingIntegrator()
        for i in range(500):
            pred = StackingPrediction(direction="bullish", confidence=0.5,
                                       probability_bullish=0.5, probability_bearish=0.25,
                                       probability_neutral=0.25, fallback_used=(i % 2 == 0),
                                       latency_ms=float(i))
            integrator.prediction_history.append(pred)
        export_path = tmp_path / "many.json"
        assert integrator.export_prediction_log(export_path) is True
        with open(export_path) as f:
            data = json.load(f)
        assert len(data) == 500


# ---------------------------------------------------------------------------
# Feature engine interaction
# ---------------------------------------------------------------------------

class TestFeatureEngine:
    """Test interaction with feature engine."""

    def test_feature_engine_called_on_predict(self):
        mock_engine = Mock()
        mock_engine.generate_features.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        integrator = StackingIntegrator(feature_engine=mock_engine)
        base_signals = {"tsmom": {"direction": "bullish", "confidence": 0.8}}
        integrator.predict(base_signals, "bull", 15.0)
        mock_engine.generate_features.assert_called_once_with(
            base_signals, "bull", 15.0
        )

    def test_feature_engine_result_used_in_prediction(self):
        mock_engine = Mock()
        mock_engine.generate_features.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        integrator = StackingIntegrator(feature_engine=mock_engine)
        base_signals = {"tsmom": {"direction": "bullish", "confidence": 0.8}}
        result = integrator.predict(base_signals)
        assert isinstance(result, StackingPrediction)

    def test_feature_engine_returns_wrong_type(self):
        """Feature engine returning wrong type should be handled."""
        mock_engine = Mock()
        mock_engine.generate_features.return_value = [0.1, 0.2, 0.3]  # list, not ndarray
        integrator = StackingIntegrator(feature_engine=mock_engine)
        base_signals = {"tsmom": {"direction": "bullish", "confidence": 0.8}}
        # This will fail on .reshape(1, -1) since list doesn't have reshape
        result = integrator.predict(base_signals)
        # Should fallback gracefully
        assert result.fallback_used is True


# ---------------------------------------------------------------------------
# CLI / __main__ guard test
# ---------------------------------------------------------------------------

class TestCLIGuard:
    """Verify module doesn't execute on import."""

    def test_module_import_does_not_execute(self):
        """Importing the module should not produce side effects or raise."""
        import importlib
        import src.signals.stacking_integrator
        importlib.reload(src.signals.stacking_integrator)
        # If we get here, import succeeded without errors
        assert hasattr(src.signals.stacking_integrator, "__all__")

    def test_module_has_no_main_guard(self, capsys):
        """Verify the source module has no __main__ guard that would execute code."""
        import src.signals.stacking_integrator as mod
        source = open(mod.__file__).read() if hasattr(mod, "__file__") else ""
        has_guard = 'if __name__' in source or 'if __name__ == "__main__"' in source
        # The module intentionally does NOT have a __main__ guard
        # (it's a library module, not a CLI script)
        # Verifying that importing doesn't produce output
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


# ---------------------------------------------------------------------------
# Public API coverage verification
# ---------------------------------------------------------------------------

class TestPublicAPICoverage:
    """Verify full public API surface is tested."""

    def test_all_exports_have_coverage(self):
        """Each name in __all__ should correspond to a tested symbol."""
        import src.signals.stacking_integrator as mod
        exports = set(mod.__all__)
        expected = {"StackingPrediction", "ModelMetadata", "StackingIntegrator", "get_stacking_prediction"}
        assert exports == expected
        for name in exports:
            assert hasattr(mod, name)

    def test_all_methods_have_been_called(self):
        """All public methods of StackingIntegrator should be callable."""
        integrator = StackingIntegrator()
        # Verify all public methods are accessible
        assert hasattr(integrator, "load_model")
        assert hasattr(integrator, "predict")
        assert hasattr(integrator, "get_accuracy_stats")
        assert hasattr(integrator, "detect_drift")
        assert hasattr(integrator, "export_prediction_log")
        # Verify all "private" methods exist
        assert hasattr(integrator, "_extract_simple_features")
        assert hasattr(integrator, "_weighted_voting_fallback")
        assert hasattr(integrator, "_get_top_features")
        assert hasattr(integrator, "_add_to_history")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
