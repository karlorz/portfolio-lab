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

# Add src to path
import sys
sys.path.insert(0, '/root/projects/portfolio-lab/src')

from signals.stacking_integrator import (
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


class TestPerformance:
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
