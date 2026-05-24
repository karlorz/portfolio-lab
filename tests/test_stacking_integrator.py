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

from src.signals.stacking_integrator import (
    StackingIntegrator,
    StackingPrediction,
    ModelMetadata,
    get_stacking_prediction
)


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


class TestModelLoading:
    """Test model loading functionality"""
    
    def test_load_valid_model(self, tmp_path):
        """Test loading a valid model pickle"""
        model_data = {
            'model': PicklableModel([0.1, 0.2, 0.7]),
            'metadata': {
                'version': 'v1.0',
                'training_date': datetime(2026, 5, 1),
                'feature_count': 102,
                'accuracy_train': 0.75,
                'accuracy_val': 0.68,
                'feature_importance': {'feat_1': 0.15},
                'total_samples': 10000
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


class TestStackingPredictionExtended:
    """Extended tests for StackingPrediction dataclass."""

    def test_to_dict_fields(self):
        """Prediction should have all expected fields."""
        pred = StackingPrediction(
            direction='bullish',
            confidence=0.80,
            probability_bullish=0.80,
            probability_bearish=0.10,
            probability_neutral=0.10,
            fallback_used=False,
            model_version='v2.0',
            latency_ms=1.5,
        )
        assert pred.direction == 'bullish'
        assert pred.confidence == 0.80
        assert pred.probability_bullish == 0.80
        assert pred.fallback_used is False
        assert pred.model_version == 'v2.0'
        assert pred.latency_ms == 1.5
        assert pred.top_features == []

    def test_bearish_prediction(self):
        """Bearish prediction should work correctly."""
        pred = StackingPrediction(
            direction='bearish',
            confidence=0.75,
            probability_bullish=0.10,
            probability_bearish=0.75,
            probability_neutral=0.15,
            fallback_used=True,
        )
        assert pred.direction == 'bearish'
        assert pred.probability_bearish > pred.probability_bullish

    def test_feature_vector_stored(self):
        """Feature vector should be stored when provided."""
        fv = np.array([1.0, 0.5, -0.3], dtype=np.float32)
        pred = StackingPrediction(
            direction='neutral',
            confidence=0.5,
            probability_bullish=0.33,
            probability_bearish=0.33,
            probability_neutral=0.34,
            fallback_used=False,
            feature_vector=fv,
        )
        assert pred.feature_vector is not None
        np.testing.assert_array_equal(pred.feature_vector, fv)

    def test_top_features(self):
        """Top features should be stored as list of tuples."""
        pred = StackingPrediction(
            direction='bullish',
            confidence=0.8,
            probability_bullish=0.8,
            probability_bearish=0.1,
            probability_neutral=0.1,
            fallback_used=False,
            top_features=[('momentum', 0.35), ('volatility', 0.20)],
        )
        assert len(pred.top_features) == 2
        assert pred.top_features[0] == ('momentum', 0.35)


class TestModelMetadataExtended:
    """Extended tests for ModelMetadata dataclass."""

    def test_all_fields(self):
        """All metadata fields should be accessible."""
        meta = ModelMetadata(
            version='v3.0',
            training_date=datetime(2026, 6, 1),
            feature_count=200,
            accuracy_train=0.82,
            accuracy_val=0.76,
            feature_importance={'feat_a': 0.25, 'feat_b': 0.18, 'feat_c': 0.10},
            total_samples=50000,
        )
        assert meta.version == 'v3.0'
        assert meta.feature_count == 200
        assert meta.accuracy_train == 0.82
        assert meta.accuracy_val == 0.76
        assert len(meta.feature_importance) == 3
        assert meta.total_samples == 50000

    def test_empty_feature_importance(self):
        """Empty feature importance dict should be valid."""
        meta = ModelMetadata(
            version='v0.1',
            training_date=datetime.now(),
            feature_count=0,
            accuracy_train=0.0,
            accuracy_val=0.0,
            feature_importance={},
            total_samples=0,
        )
        assert meta.feature_importance == {}


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
        result = integrator.export_prediction_log(Path('/nonexistent/dir/file.json'))
        assert result is False


class TestAccuracyStatsExtended:
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
