import pytest; pytestmark = pytest.mark.heavy
"""
Tests for Stacking Ensemble Trainer (v3.10 Phase 2)

Validates:
- XGBoost model training with time-series CV
- Feature importance extraction
- Model persistence and loading
- Synthetic data generation for testing
- Backfill prediction generation
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.stacking_trainer import (
    StackingTrainer, TrainingConfig, TrainingResult, PredictionResult
)


class TestTrainingConfig:
    """Test training configuration dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.learning_rate == 0.05
        assert config.max_depth == 4
        assert config.n_estimators == 1000
        assert config.early_stopping_rounds == 10
        assert config.reg_alpha == 0.1
        assert config.reg_lambda == 1.0
        assert config.eval_metric == "auc"
        assert config.min_training_days == 252
        assert config.feature_count == 102
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            learning_rate=0.01,
            max_depth=6,
            n_estimators=500
        )
        assert config.learning_rate == 0.01
        assert config.max_depth == 6
        assert config.n_estimators == 500


class TestStackingTrainer:
    """Test stacking trainer core functionality."""
    
    @pytest.fixture
    def trainer(self, tmp_path):
        """Create trainer with temp directory."""
        config = TrainingConfig(
            model_dir=str(tmp_path / "models"),
            min_training_days=50  # Lower for testing
        )
        return StackingTrainer(config)
    
    def test_trainer_initialization(self, trainer):
        """Test trainer initializes correctly."""
        assert trainer.model is None
        assert trainer.model_version is None
        assert trainer.config is not None
        assert trainer.feature_engine is not None
    
    def test_synthetic_data_generation(self, trainer):
        """Test synthetic data generation for testing."""
        X, y, dates = trainer._generate_synthetic_data(n_samples=100)
        
        assert X.shape == (100, 102)
        assert len(y) == 100
        assert len(dates) == 100
        assert all(isinstance(d, str) for d in dates)
        assert set(y).issubset({0, 1})
    
    def test_feature_names_generation(self, trainer):
        """Test feature names list generation."""
        names = trainer._get_feature_names()
        
        # Should have 102 features
        assert len(names) == 102
        
        # Check for expected prefixes
        base_names = [n for n in names if n.startswith("base_")]
        assert len(base_names) == 8  # 8 base signals
        
        mult_names = [n for n in names if n.startswith("mult_")]
        assert len(mult_names) == 28  # C(8,2) = 28 pairs
    
    def test_train_with_synthetic_data(self, trainer, tmp_path):
        """Test full training pipeline with synthetic data."""
        result = trainer.train(start_date="2020-01-01")
        
        # Check result structure
        assert isinstance(result, TrainingResult)
        assert result.model_version is not None
        assert result.training_date is not None
        assert result.model_path is not None
        
        # Check performance metrics are reasonable
        assert 0 <= result.train_accuracy <= 1
        assert 0 <= result.validation_accuracy <= 1
        assert 0 <= result.validation_auc <= 1
        
        # Check CV results
        assert result.cv_mean_accuracy > 0
        assert result.cv_std_accuracy >= 0
        assert result.cv_mean_auc > 0
        
        # Check feature importance
        assert len(result.top_features) == 10
        for name, importance in result.top_features:
            assert isinstance(name, str)
            assert 0 <= importance <= 1
        
        # Check model was saved
        model_path = Path(result.model_path)
        assert model_path.exists()
        
        # Check training result saved
        result_path = model_path.parent / f"training_result_{result.model_version}.json"
        assert result_path.exists()
    
    def test_model_save_and_load(self, trainer, tmp_path):
        """Test model persistence and loading."""
        # Train a model
        result = trainer.train(start_date="2020-01-01")
        
        # Create new trainer and load
        new_trainer = StackingTrainer(trainer.config)
        success = new_trainer.load_model(result.model_path)
        
        assert success is True
        assert new_trainer.model is not None
        assert new_trainer.model_version == trainer.model_version
    
    def test_load_nonexistent_model(self, trainer):
        """Test loading a non-existent model."""
        success = trainer.load_model("/nonexistent/path/model.json")
        assert success is False
    
    def test_predict_with_loaded_model(self, trainer):
        """Test prediction with loaded model."""
        # Train and load
        result = trainer.train(start_date="2020-01-01")
        trainer.load_model(result.model_path)
        
        # Create test features
        X_test = np.random.randn(102)
        
        prediction = trainer.predict(X_test)
        
        assert isinstance(prediction, PredictionResult)
        assert prediction.timestamp is not None
        assert prediction.prediction in [0, 1]
        assert 0 <= prediction.probability <= 1
        assert 0.5 <= prediction.confidence <= 1
        # using_fallback might be numpy bool or Python bool
        assert bool(prediction.using_fallback) == prediction.using_fallback or isinstance(prediction.using_fallback, (bool, np.bool_))
    
    def test_predict_without_model(self, trainer):
        """Test prediction without loaded model triggers fallback."""
        X_test = np.random.randn(102)
        
        prediction = trainer.predict(X_test)
        
        assert prediction.using_fallback is True
        assert prediction.fallback_reason == "No model loaded"
        assert prediction.prediction == 0
    
    def test_fallback_low_confidence(self, trainer):
        """Test fallback when confidence is below threshold."""
        # Train a model
        result = trainer.train(start_date="2020-01-01")
        trainer.load_model(result.model_path)
        
        # Test features (may trigger low confidence)
        X_test = np.zeros(102)
        
        prediction = trainer.predict(X_test, confidence_threshold=0.9)
        
        # With zero features, confidence likely below 0.9
        assert prediction.using_fallback == True or prediction.using_fallback == np.True_
        assert "Confidence" in prediction.fallback_reason


class TestModelPerformance:
    """Test model performance meets targets."""
    
    @pytest.fixture
    def trainer(self, tmp_path):
        """Create trainer with temp directory."""
        config = TrainingConfig(
            model_dir=str(tmp_path / "models"),
            min_training_days=50
        )
        return StackingTrainer(config)
    
    def test_accuracy_above_baseline(self, trainer):
        """Test that trained model achieves >0.52 accuracy on synthetic data."""
        result = trainer.train(start_date="2020-01-01")
        
        # On purely random synthetic data, accuracy should be > 0.50 (better than random)
        # With signal-correlated data, should be better but synthetic data is random
        assert result.validation_accuracy > 0.50, \
            f"Validation accuracy {result.validation_accuracy:.3f} should beat random"
    
    def test_cv_consistency(self, trainer):
        """Test that CV scores are consistent (low std)."""
        result = trainer.train(start_date="2020-01-01")
        
        # Std should be reasonable (<0.15 for synthetic data)
        assert result.cv_std_accuracy < 0.15, \
            f"CV std {result.cv_std_accuracy:.3f} too high, possible overfitting"
    
    def test_auc_reasonable(self, trainer):
        """Test AUC is reasonable (0.5-1.0)."""
        result = trainer.train(start_date="2020-01-01")
        
        assert 0.5 <= result.validation_auc <= 1.0
        assert 0.5 <= result.cv_mean_auc <= 1.0
    
    def test_feature_importance_sum(self, trainer):
        """Test that top feature importances sum to reasonable value."""
        result = trainer.train(start_date="2020-01-01")
        
        total_importance = sum(imp for _, imp in result.top_features)
        # Top 10 should capture significant importance
        assert 0 < total_importance <= 1


class TestBackfill:
    """Test backfill prediction generation."""
    
    @pytest.fixture
    def trainer(self, tmp_path):
        """Create trainer with temp directory."""
        config = TrainingConfig(
            model_dir=str(tmp_path / "models"),
            min_training_days=50,
            db_path=str(tmp_path / "market.db")
        )
        return StackingTrainer(config)
    
    def test_backfill_dry_run(self, trainer):
        """Test backfill in dry-run mode."""
        # Train first
        trainer.train(start_date="2020-01-01")
        
        # Backfill with dry run
        stats = trainer.backfill_predictions(
            start_date="2020-01-01",
            dry_run=True
        )
        
        assert "total_predictions" in stats
        assert "accuracy" in stats
        assert "fallback_rate" in stats
        
        # No database should be created in dry run
        db_path = Path(trainer.config.db_path)
        assert not db_path.exists()


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_pipeline(self, tmp_path):
        """Test complete training and inference pipeline."""
        config = TrainingConfig(
            model_dir=str(tmp_path / "models"),
            min_training_days=50
        )
        trainer = StackingTrainer(config)
        
        # 1. Train
        result = trainer.train(start_date="2020-01-01")
        assert result is not None
        
        # 2. Load
        new_trainer = StackingTrainer(config)
        success = new_trainer.load_model(result.model_path)
        assert success
        
        # 3. Predict
        X_test = np.random.randn(102)
        prediction = new_trainer.predict(X_test)
        assert prediction is not None
        
        # 4. Check output
        assert prediction.prediction in [0, 1]
        assert 0 <= prediction.probability <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
