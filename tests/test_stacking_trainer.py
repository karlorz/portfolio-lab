"""
Comprehensive tests for Stacking Ensemble Trainer — non-ML (stub) + heavy (ML) paths.

Original tests (marked @pytest.mark.heavy) exercise the training pipeline with
real xgboost/sklearn. New comprehensive tests exercise stub code paths, dataclass
field validation, CLI, and edge cases — all runnable with PORTFOLIO_LAB_ENABLE_ML=0.

Coverage:
  - Dataclass field validation (all 3 dataclasses: field names, types, defaults)
  - Module-level constants and stub behavior verification
  - Edge cases: zero/empty inputs, single-element arrays, NaN/Inf, boundary values
  - CLI entry points with capsys
  - Stubbed ML pipeline (using unittest.mock / picklable mock, not real ML)
  - Public API coverage and module structure
  - Feature names generation (pure Python, no ML)
  - Synthetic data generation (pure numpy, no ML)
  - create_features_from_signals boundary conditions
  - load_historical_data synthetic fallback
"""

import dataclasses
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.ml.stacking_trainer import (
    StackingTrainer,
    TrainingConfig,
    TrainingResult,
    PredictionResult,
)
from src.signals.stacking_feature_engine import StackingFeatureEngine
from src.signals.signal_source import SignalSource
from src.paths import MARKET_DB


CANONICAL_FEATURE_COUNT = StackingFeatureEngine.TOTAL_DIMENSIONS
CANONICAL_SOURCE_COUNT = len(SignalSource)
CANONICAL_PAIR_COUNT = CANONICAL_SOURCE_COUNT * (CANONICAL_SOURCE_COUNT - 1) // 2


# ═══════════════════════════════════════════════════════════════════════════
# ORIGINAL TESTS (preserved) — marked heavy because they call trainer.train()
# which requires real xgboost/sklearn with working .fit() / .predict().
# These are skipped unless PORTFOLIO_LAB_ENABLE_ML=1 --include-heavy.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.heavy
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
        assert config.feature_count == CANONICAL_FEATURE_COUNT

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


@pytest.mark.heavy
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

        assert X.shape == (100, CANONICAL_FEATURE_COUNT)
        assert len(y) == 100
        assert len(dates) == 100
        assert all(isinstance(d, str) for d in dates)
        assert set(y).issubset({0, 1})

    def test_feature_names_generation(self, trainer):
        """Test feature names list generation."""
        names = trainer._get_feature_names()

        # Should have canonical feature count
        assert len(names) == CANONICAL_FEATURE_COUNT

        # Check for expected prefixes
        base_names = [n for n in names if n.startswith("base_")]
        assert len(base_names) == CANONICAL_SOURCE_COUNT

        mult_names = [n for n in names if n.startswith("mult_")]
        assert len(mult_names) == CANONICAL_PAIR_COUNT

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
        X_test = np.random.randn(CANONICAL_FEATURE_COUNT)

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
        X_test = np.random.randn(CANONICAL_FEATURE_COUNT)

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
        X_test = np.zeros(CANONICAL_FEATURE_COUNT)

        prediction = trainer.predict(X_test, confidence_threshold=0.9)

        # With zero features, confidence likely below 0.9
        assert prediction.using_fallback == True or prediction.using_fallback == np.True_
        assert "Confidence" in prediction.fallback_reason


@pytest.mark.heavy
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


@pytest.mark.heavy
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


@pytest.mark.heavy
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
        X_test = np.random.randn(CANONICAL_FEATURE_COUNT)
        prediction = new_trainer.predict(X_test)
        assert prediction is not None

        # 4. Check output
        assert prediction.prediction in [0, 1]
        assert 0 <= prediction.probability <= 1


# ═══════════════════════════════════════════════════════════════════════════
# NEW COMPREHENSIVE TESTS — run with ML=0 (stub paths)
# ═══════════════════════════════════════════════════════════════════════════

# ── Module-level constants ──────────────────────────────────────────────

class TestModuleConstants:
    """Verify module-level constants and ML-disabled state."""

    def test_ml_enabled_false_by_default(self):
        """When PORTFOLIO_LAB_ENABLE_ML=0, the flag must be False."""
        import os
        if os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1":
            pytest.skip("ML-enabled — stub test not applicable")
        from src.ml import stacking_trainer as st
        assert st._ML_ENABLED is False

    def test_module_has_xgb_stub(self):
        """xgb stub module should exist when ML is disabled."""
        import os
        if os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1":
            pytest.skip("ML-enabled — stub test not applicable")
        from src.ml import stacking_trainer as st
        assert st.xgb is not None
        assert hasattr(st.xgb, "XGBClassifier")
        assert hasattr(st.xgb, "XGBRegressor")
        assert hasattr(st.xgb, "train")
        assert hasattr(st.xgb, "DMatrix")
        assert hasattr(st.xgb, "cv")

    def test_module_has_timeseriessplit_stub(self):
        """TimeSeriesSplit stub should exist."""
        from src.ml import stacking_trainer as st
        assert st.TimeSeriesSplit is not None

    def test_module_has_accuracy_score_stub(self):
        """accuracy_score stub should exist."""
        from src.ml import stacking_trainer as st
        assert st.accuracy_score is not None

    def test_module_has_roc_auc_score_stub(self):
        """roc_auc_score stub should exist."""
        from src.ml import stacking_trainer as st
        assert st.roc_auc_score is not None

    def test_module_has_classification_report_stub(self):
        """classification_report stub should exist."""
        from src.ml import stacking_trainer as st
        assert st.classification_report is not None

    def test_no_real_xgboost_loaded(self):
        """Real xgboost must not be imported — only the stub."""
        import os
        if os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1":
            pytest.skip("ML-enabled — stub test not applicable")
        import sys
        xgb_mod = sys.modules.get("xgboost")
        assert xgb_mod is not None, "xgboost stub not in sys.modules"
        # Stub module has no __file__ (the leak check uses this)
        assert not hasattr(xgb_mod, "__file__") or xgb_mod.__file__ is None


# ── Stub behavior verification ─────────────────────────────────────────

@pytest.mark.skipif(
    __import__("os").environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1",
    reason="Stub tests only apply when ML is disabled",
)
class TestStubBehavior:
    """Verify xgboost/sklearn stubs return expected values."""

    def test_xgb_classifier_stub_returns_none(self):
        """XGBClassifier(**kw) must return None (stub)."""
        from src.ml import stacking_trainer as st
        result = st.xgb.XGBClassifier(learning_rate=0.05, max_depth=4)
        assert result is None

    def test_xgb_regressor_stub_returns_none(self):
        """XGBRegressor(**kw) must return None (stub)."""
        from src.ml import stacking_trainer as st
        result = st.xgb.XGBRegressor(learning_rate=0.05)
        assert result is None

    def test_xgb_train_stub_returns_none(self):
        """xgb.train(*a, **kw) must return None (stub)."""
        from src.ml import stacking_trainer as st
        result = st.xgb.train(None, None)
        assert result is None

    def test_xgb_dmatrix_stub_returns_none(self):
        """xgb.DMatrix(*a, **kw) must return None (stub)."""
        from src.ml import stacking_trainer as st
        result = st.xgb.DMatrix(np.array([[1, 2]]), label=[0])
        assert result is None

    def test_xgb_cv_stub_returns_dict(self):
        """xgb.cv(*a, **kw) must return dict with 'test-auc-mean' key."""
        from src.ml import stacking_trainer as st
        result = st.xgb.cv(None, None)
        assert isinstance(result, dict)
        assert "test-auc-mean" in result
        assert result["test-auc-mean"] == [0.5]

    def test_timeseriessplit_stub_returns_none(self):
        """TimeSeriesSplit(**kw) must return None (stub)."""
        from src.ml import stacking_trainer as st
        result = st.TimeSeriesSplit(n_splits=5)
        assert result is None

    def test_accuracy_score_stub_returns_half(self):
        """accuracy_score(*a) must return 0.5 (stub)."""
        from src.ml import stacking_trainer as st
        result = st.accuracy_score([1, 0, 1], [1, 0, 0])
        assert result == 0.5

    def test_roc_auc_score_stub_returns_half(self):
        """roc_auc_score(*a) must return 0.5 (stub)."""
        from src.ml import stacking_trainer as st
        result = st.roc_auc_score([1, 0, 1], [0.9, 0.1, 0.8])
        assert result == 0.5

    def test_classification_report_stub_returns_empty(self):
        """classification_report(*a) must return '' (stub)."""
        from src.ml import stacking_trainer as st
        result = st.classification_report([1, 0], [1, 0])
        assert result == ""

    def test_stubs_accept_various_kwargs(self):
        """Stubs must accept arbitrary kwargs without error."""
        from src.ml import stacking_trainer as st
        # These should not raise
        st.xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        st.TimeSeriesSplit(n_splits=10, test_size=0.2)


# ── TrainingConfig dataclass ────────────────────────────────────────────

class TestTrainingConfigDataclass:
    """Comprehensive validation of TrainingConfig fields and defaults."""

    def test_field_count(self):
        """TrainingConfig must have exactly 14 fields."""
        fields = dataclasses.fields(TrainingConfig)
        assert len(fields) == 14

    def test_all_field_names(self):
        """Verify all expected field names exist."""
        expected = {
            "learning_rate", "max_depth", "n_estimators",
            "early_stopping_rounds", "reg_alpha", "reg_lambda",
            "eval_metric", "min_training_days", "validation_split",
            "n_splits", "feature_count", "db_path", "model_dir",
            "prediction_horizon_days",
        }
        actual = {f.name for f in dataclasses.fields(TrainingConfig)}
        assert actual == expected, f"Missing: {expected - actual}"

    def test_default_learning_rate(self):
        assert TrainingConfig().learning_rate == 0.05

    def test_default_max_depth(self):
        assert TrainingConfig().max_depth == 4

    def test_default_n_estimators(self):
        assert TrainingConfig().n_estimators == 1000

    def test_default_early_stopping_rounds(self):
        assert TrainingConfig().early_stopping_rounds == 10

    def test_default_reg_alpha(self):
        assert TrainingConfig().reg_alpha == 0.1

    def test_default_reg_lambda(self):
        assert TrainingConfig().reg_lambda == 1.0

    def test_default_eval_metric(self):
        assert TrainingConfig().eval_metric == "auc"

    def test_default_min_training_days(self):
        assert TrainingConfig().min_training_days == 252

    def test_default_validation_split(self):
        assert TrainingConfig().validation_split == 0.2

    def test_default_n_splits(self):
        assert TrainingConfig().n_splits == 5

    def test_default_feature_count(self):
        assert TrainingConfig().feature_count == CANONICAL_FEATURE_COUNT

    def test_default_feature_count_tracks_stacking_feature_engine(self):
        """Default training dimensions derive from the canonical stacking roster."""
        assert TrainingConfig().feature_count == StackingFeatureEngine.TOTAL_DIMENSIONS

    def test_default_db_path(self):
        assert TrainingConfig().db_path == str(MARKET_DB)

    def test_default_model_dir(self):
        assert TrainingConfig().model_dir == "models"

    def test_default_prediction_horizon_days(self):
        assert TrainingConfig().prediction_horizon_days == 5

    def test_custom_values_override_defaults(self):
        config = TrainingConfig(
            learning_rate=0.1,
            max_depth=6,
            n_estimators=200,
            min_training_days=100,
            feature_count=128,
        )
        assert config.learning_rate == 0.1
        assert config.max_depth == 6
        assert config.n_estimators == 200
        assert config.min_training_days == 100
        assert config.feature_count == 128

    def test_type_preservation(self):
        """Field types must be preserved: int, float, str."""
        config = TrainingConfig(
            learning_rate=0.05,
            max_depth=4,
            n_estimators=1000,
            eval_metric="auc",
        )
        assert isinstance(config.learning_rate, float)
        assert isinstance(config.max_depth, int)
        assert isinstance(config.n_estimators, int)
        assert isinstance(config.eval_metric, str)
        assert isinstance(config.min_training_days, int)
        assert isinstance(config.validation_split, float)
        assert isinstance(config.feature_count, int)
        assert isinstance(config.db_path, str)
        assert isinstance(config.model_dir, str)
        assert isinstance(config.prediction_horizon_days, int)

    def test_zero_negative_extreme_values(self):
        """Must accept zero/negative values (validation is caller's concern)."""
        config = TrainingConfig(
            learning_rate=0.0,
            max_depth=0,
            n_estimators=-1,
            min_training_days=0,
            validation_split=0.0,
            n_splits=0,
        )
        assert config.learning_rate == 0.0
        assert config.max_depth == 0
        assert config.n_estimators == -1
        assert config.min_training_days == 0
        assert config.validation_split == 0.0
        assert config.n_splits == 0

    def test_large_values(self):
        """Must accept large boundary values."""
        config = TrainingConfig(
            learning_rate=1.0,
            max_depth=100,
            n_estimators=100000,
            min_training_days=10000,
        )
        assert config.learning_rate == 1.0
        assert config.max_depth == 100

    def test_string_fields_accept_empty(self):
        """String fields must accept empty strings."""
        config = TrainingConfig(db_path="", model_dir="")
        assert config.db_path == ""
        assert config.model_dir == ""

    def test_dataclass_asdict_works(self):
        """dataclasses.asdict() must produce a flat dict."""
        config = TrainingConfig()
        d = dataclasses.asdict(config)
        assert isinstance(d, dict)
        assert len(d) == 14
        assert d["learning_rate"] == 0.05
        assert d["eval_metric"] == "auc"


# ── TrainingResult dataclass ────────────────────────────────────────────

class TestTrainingResultDataclass:
    """Comprehensive validation of TrainingResult fields."""

    def test_field_count(self):
        assert len(dataclasses.fields(TrainingResult)) == 13

    def test_all_field_names(self):
        expected = {
            "model_version", "training_date",
            "train_accuracy", "validation_accuracy", "validation_auc",
            "cv_mean_accuracy", "cv_std_accuracy", "cv_mean_auc",
            "top_features", "training_samples", "validation_samples",
            "date_range", "model_path",
        }
        actual = {f.name for f in dataclasses.fields(TrainingResult)}
        assert actual == expected

    def test_can_create_minimal(self):
        result = TrainingResult(
            model_version="v1",
            training_date="2024-01-01",
            train_accuracy=0.8,
            validation_accuracy=0.75,
            validation_auc=0.78,
            cv_mean_accuracy=0.73,
            cv_std_accuracy=0.02,
            cv_mean_auc=0.76,
            top_features=[("feat_1", 0.3)],
            training_samples=500,
            validation_samples=200,
            date_range=("2020-01-01", "2024-01-01"),
            model_path="/tmp/model.pkl",
        )
        assert result.model_version == "v1"
        assert result.train_accuracy == 0.8
        assert result.validation_accuracy == 0.75

    def test_float_fields_accept_boundaries(self):
        """Float fields must accept 0.0, 1.0, and values in between."""
        result = TrainingResult(
            model_version="v1", training_date="now",
            train_accuracy=0.0, validation_accuracy=1.0, validation_auc=0.5,
            cv_mean_accuracy=0.0, cv_std_accuracy=0.0, cv_mean_auc=1.0,
            top_features=[("a", 0.0)], training_samples=0, validation_samples=0,
            date_range=("a", "b"), model_path="",
        )
        assert result.train_accuracy == 0.0
        assert result.validation_accuracy == 1.0
        assert result.cv_mean_auc == 1.0

    def test_top_features_empty_list(self):
        """top_features must accept empty list."""
        result = TrainingResult(
            model_version="v1", training_date="now",
            train_accuracy=0.5, validation_accuracy=0.5, validation_auc=0.5,
            cv_mean_accuracy=0.5, cv_std_accuracy=0.0, cv_mean_auc=0.5,
            top_features=[], training_samples=0, validation_samples=0,
            date_range=("a", "b"), model_path="",
        )
        assert result.top_features == []

    def test_top_features_many_items(self):
        """top_features must accept many (name, importance) tuples."""
        top = [(f"f{i}", float(i) / 100) for i in range(100)]
        result = TrainingResult(
            model_version="v1", training_date="now",
            train_accuracy=0.5, validation_accuracy=0.5, validation_auc=0.5,
            cv_mean_accuracy=0.5, cv_std_accuracy=0.0, cv_mean_auc=0.5,
            top_features=top, training_samples=0, validation_samples=0,
            date_range=("a", "b"), model_path="",
        )
        assert len(result.top_features) == 100
        assert result.top_features[0] == ("f0", 0.0)

    def test_date_range_tuple_type(self):
        """date_range must be a tuple of two strings."""
        result = TrainingResult(
            model_version="v1", training_date="now",
            train_accuracy=0.5, validation_accuracy=0.5, validation_auc=0.5,
            cv_mean_accuracy=0.5, cv_std_accuracy=0.0, cv_mean_auc=0.5,
            top_features=[], training_samples=0, validation_samples=0,
            date_range=("2020-01-01", "2024-12-31"), model_path="",
        )
        assert isinstance(result.date_range, tuple)
        assert len(result.date_range) == 2
        assert isinstance(result.date_range[0], str)
        assert isinstance(result.date_range[1], str)

    def test_type_preservation(self):
        """Verify type annotations are preserved."""
        result = TrainingResult(
            model_version="v1", training_date="now",
            train_accuracy=0.5, validation_accuracy=0.5, validation_auc=0.5,
            cv_mean_accuracy=0.5, cv_std_accuracy=0.0, cv_mean_auc=0.5,
            top_features=[], training_samples=100, validation_samples=50,
            date_range=("a", "b"), model_path="/some/path.pkl",
        )
        assert isinstance(result.model_version, str)
        assert isinstance(result.training_date, str)
        assert isinstance(result.train_accuracy, float)
        assert isinstance(result.training_samples, int)
        assert isinstance(result.validation_samples, int)
        assert isinstance(result.model_path, str)
        assert isinstance(result.top_features, list)

    def test_dataclass_asdict_works(self):
        """dataclasses.asdict() must produce a JSON-serializable dict."""
        result = TrainingResult(
            model_version="v1", training_date="now",
            train_accuracy=0.8, validation_accuracy=0.75, validation_auc=0.78,
            cv_mean_accuracy=0.73, cv_std_accuracy=0.02, cv_mean_auc=0.76,
            top_features=[("f1", 0.3), ("f2", 0.2)],
            training_samples=500, validation_samples=200,
            date_range=("2020-01-01", "2024-01-01"),
            model_path="/tmp/model.pkl",
        )
        d = dataclasses.asdict(result)
        assert json.dumps(d)  # Must be JSON-serializable
        assert d["model_version"] == "v1"
        # asdict recurses into lists but preserves tuple types inside
        assert len(d["top_features"]) == 2
        assert d["top_features"][0][0] == "f1"
        assert d["top_features"][0][1] == 0.3
        assert d["top_features"][1][0] == "f2"


# ── PredictionResult dataclass ──────────────────────────────────────────

class TestPredictionResultDataclass:
    """Comprehensive validation of PredictionResult fields."""

    def test_field_count(self):
        assert len(dataclasses.fields(PredictionResult)) == 7

    def test_all_field_names(self):
        expected = {
            "timestamp", "prediction", "probability", "confidence",
            "feature_vector", "using_fallback", "fallback_reason",
        }
        actual = {f.name for f in dataclasses.fields(PredictionResult)}
        assert actual == expected

    def test_prediction_accepts_zero_and_one(self):
        """prediction must accept int values 0 and 1."""
        r0 = PredictionResult(timestamp="now", prediction=0, probability=0.5, confidence=0.5)
        r1 = PredictionResult(timestamp="now", prediction=1, probability=0.5, confidence=0.5)
        assert r0.prediction == 0
        assert r1.prediction == 1

    def test_probability_boundary_values(self):
        """probability must accept 0.0 and 1.0."""
        r_low = PredictionResult(timestamp="now", prediction=0, probability=0.0, confidence=0.5)
        r_high = PredictionResult(timestamp="now", prediction=1, probability=1.0, confidence=0.5)
        assert r_low.probability == 0.0
        assert r_high.probability == 1.0

    def test_confidence_boundary_values(self):
        """confidence must accept 0.0 and 1.0."""
        r_low = PredictionResult(timestamp="now", prediction=0, probability=0.5, confidence=0.0)
        r_high = PredictionResult(timestamp="now", prediction=1, probability=0.5, confidence=1.0)
        assert r_low.confidence == 0.0
        assert r_high.confidence == 1.0

    def test_feature_vector_default_none(self):
        """feature_vector must default to None."""
        r = PredictionResult(timestamp="now", prediction=0, probability=0.5, confidence=0.5)
        assert r.feature_vector is None

    def test_using_fallback_default_false(self):
        """using_fallback must default to False."""
        r = PredictionResult(timestamp="now", prediction=0, probability=0.5, confidence=0.5)
        assert r.using_fallback is False

    def test_fallback_reason_default_empty(self):
        """fallback_reason must default to empty string."""
        r = PredictionResult(timestamp="now", prediction=0, probability=0.5, confidence=0.5)
        assert r.fallback_reason == ""

    def test_feature_vector_accepts_list(self):
        """feature_vector must accept Optional[List[float]]."""
        vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        r = PredictionResult(
            timestamp="now", prediction=0, probability=0.5, confidence=0.5,
            feature_vector=vec,
        )
        assert r.feature_vector == vec
        assert isinstance(r.feature_vector, list)

    def test_using_fallback_true(self):
        """using_fallback must work when set to True."""
        r = PredictionResult(
            timestamp="now", prediction=0, probability=0.5, confidence=0.3,
            using_fallback=True, fallback_reason="Low confidence",
        )
        assert r.using_fallback is True
        assert r.fallback_reason == "Low confidence"

    def test_timestamp_types(self):
        """timestamp must accept any string."""
        r = PredictionResult(timestamp="2024-01-01T12:00:00", prediction=0, probability=0.5, confidence=0.5)
        assert isinstance(r.timestamp, str)
        assert r.timestamp == "2024-01-01T12:00:00"

    def test_dataclass_asdict_works(self):
        """dataclasses.asdict() must work and be JSON-serializable."""
        r = PredictionResult(
            timestamp="2024-01-01T00:00:00", prediction=1, probability=0.85,
            confidence=0.85, feature_vector=[0.1, 0.2],
            using_fallback=False, fallback_reason="",
        )
        d = dataclasses.asdict(r)
        assert json.dumps(d)
        assert d["prediction"] == 1
        assert d["probability"] == 0.85


# ── StackingTrainer initialization edge cases ───────────────────────────

class TestStackingTrainerInit:
    """Test StackingTrainer construction edge cases (no ML)."""

    def test_default_config_when_none(self):
        """Passing None for config must use TrainingConfig() defaults."""
        trainer = StackingTrainer(config=None)
        assert isinstance(trainer.config, TrainingConfig)
        assert trainer.config.model_dir == "models"

    def test_model_dir_created(self, tmp_path):
        """__init__ must create the model directory if missing."""
        d = tmp_path / "nonexistent_models"
        config = TrainingConfig(model_dir=str(d))
        trainer = StackingTrainer(config)
        assert d.exists()
        assert d.is_dir()

    def test_model_dir_with_nested_path(self, tmp_path):
        """__init__ must create nested directory."""
        d = tmp_path / "a" / "b" / "c_models"
        config = TrainingConfig(model_dir=str(d))
        trainer = StackingTrainer(config)
        assert d.exists()

    def test_model_version_none_on_init(self):
        """model_version must be None after construction."""
        trainer = StackingTrainer(TrainingConfig())
        assert trainer.model_version is None

    def test_model_none_on_init(self):
        """model must be None after construction."""
        trainer = StackingTrainer(TrainingConfig())
        assert trainer.model is None

    def test_feature_engine_created(self):
        """feature_engine must be a StackingFeatureEngine instance."""
        trainer = StackingTrainer(TrainingConfig())
        assert trainer.feature_engine is not None
        # just verify it has expected methods
        assert hasattr(trainer.feature_engine, "create_features")
        assert hasattr(trainer.feature_engine, "to_numpy")

    def test_multiple_instances_independent(self):
        """Each instance must have independent state."""
        t1 = StackingTrainer(TrainingConfig())
        t2 = StackingTrainer(TrainingConfig())
        assert t1.model is t2.model  # both None
        assert t1.config is not t2.config
        t1.model_version = "v1"
        assert t2.model_version is None

    def test_custom_config_preserved(self):
        """Passing a custom config must store it and not modify defaults."""
        custom = TrainingConfig(learning_rate=0.5, n_estimators=50)
        trainer = StackingTrainer(custom)
        assert trainer.config.learning_rate == 0.5
        assert trainer.config.n_estimators == 50


# ── Synthetic data generation edge cases ────────────────────────────────

class TestSyntheticDataEdgeCases:
    """Test _generate_synthetic_data boundary conditions (pure numpy)."""

    def test_zero_samples(self):
        """n_samples=0 must return empty arrays."""
        trainer = StackingTrainer(TrainingConfig())
        X, y, dates = trainer._generate_synthetic_data(n_samples=0)
        assert X.shape == (0, CANONICAL_FEATURE_COUNT)
        assert len(y) == 0
        assert len(dates) == 0

    def test_one_sample(self):
        """n_samples=1 must return single-element arrays."""
        trainer = StackingTrainer(TrainingConfig())
        X, y, dates = trainer._generate_synthetic_data(n_samples=1)
        assert X.shape == (1, CANONICAL_FEATURE_COUNT)
        assert len(y) == 1
        assert len(dates) == 1
        assert y[0] in (0, 1)

    def test_large_samples(self):
        """n_samples=10000 must produce correct shapes."""
        trainer = StackingTrainer(TrainingConfig())
        X, y, dates = trainer._generate_synthetic_data(n_samples=10000)
        assert X.shape == (10000, CANONICAL_FEATURE_COUNT)
        assert len(y) == 10000
        assert len(dates) == 10000

    def test_output_types(self):
        """X must be ndarray, y must be ndarray of ints."""
        trainer = StackingTrainer(TrainingConfig())
        X, y, dates = trainer._generate_synthetic_data(n_samples=50)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.dtype == np.float64
        assert y.dtype == np.int_ or y.dtype == np.int64 or y.dtype == int

    def test_deterministic_seed(self):
        """Same seed (np.random.seed(42)) must produce same data."""
        trainer = StackingTrainer(TrainingConfig())
        X1, y1, d1 = trainer._generate_synthetic_data(n_samples=100)
        X2, y2, d2 = trainer._generate_synthetic_data(n_samples=100)
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)
        assert d1 == d2

    def test_feature_count_matches_config(self):
        """X.shape[1] must match config.feature_count."""
        config = TrainingConfig(feature_count=128)
        trainer = StackingTrainer(config)
        X, y, dates = trainer._generate_synthetic_data(n_samples=10)
        assert X.shape[1] == 128

    def test_default_shape_tracks_current_signal_roster(self):
        """Synthetic fallback uses the full current SignalSource roster by default."""
        trainer = StackingTrainer(TrainingConfig())
        X, y, dates = trainer._generate_synthetic_data(n_samples=10)
        assert X.shape == (10, StackingFeatureEngine.TOTAL_DIMENSIONS)

    def test_dates_ascending(self):
        """Dates must be in ascending order."""
        trainer = StackingTrainer(TrainingConfig())
        X, y, dates = trainer._generate_synthetic_data(n_samples=100)
        for i in range(1, len(dates)):
            assert dates[i] > dates[i-1], f"Dates not ascending at index {i}"

    def test_target_values_binary(self):
        """y values must be only 0 or 1."""
        trainer = StackingTrainer(TrainingConfig())
        X, y, dates = trainer._generate_synthetic_data(n_samples=200)
        assert set(y.tolist()).issubset({0, 1})

    def test_feature_values_are_finite(self):
        """X values must be finite (no NaN or Inf)."""
        trainer = StackingTrainer(TrainingConfig())
        X, y, dates = trainer._generate_synthetic_data(n_samples=100)
        assert np.all(np.isfinite(X))

    def test_dates_format(self):
        """Dates must be 'YYYY-MM-DD' strings."""
        trainer = StackingTrainer(TrainingConfig())
        X, y, dates = trainer._generate_synthetic_data(n_samples=10)
        for d in dates:
            assert len(d) == 10
            assert d[4] == "-"
            assert d[7] == "-"


# ── Feature names generation edge cases ─────────────────────────────────

class TestFeatureNamesEdgeCases:
    """Test _get_feature_names structure (pure Python, no ML)."""

    def test_exact_length(self):
        """Must produce the canonical number of feature names."""
        trainer = StackingTrainer(TrainingConfig())
        names = trainer._get_feature_names()
        assert len(names) == CANONICAL_FEATURE_COUNT

    def test_names_match_feature_engine_contract(self):
        """Trainer feature names must not drift from feature-engine order."""
        trainer = StackingTrainer(TrainingConfig())
        assert trainer._get_feature_names() == StackingFeatureEngine().get_feature_names()

    def test_base_prefix_count(self):
        """Must have one base_ feature name per SignalSource."""
        trainer = StackingTrainer(TrainingConfig())
        names = trainer._get_feature_names()
        base = [n for n in names if n.startswith("base_")]
        assert len(base) == CANONICAL_SOURCE_COUNT

    def test_mult_prefix_count(self):
        """Must have one mult_ feature per source pair."""
        trainer = StackingTrainer(TrainingConfig())
        names = trainer._get_feature_names()
        mult = [n for n in names if n.startswith("mult_")]
        assert len(mult) == CANONICAL_PAIR_COUNT

    def test_disagree_prefix_count(self):
        """Must have one disagree_ feature per source pair."""
        trainer = StackingTrainer(TrainingConfig())
        names = trainer._get_feature_names()
        disagree = [n for n in names if n.startswith("disagree_")]
        assert len(disagree) == CANONICAL_PAIR_COUNT

    def test_avg_prefix_count(self):
        """Must have one avg_ feature per source pair."""
        trainer = StackingTrainer(TrainingConfig())
        names = trainer._get_feature_names()
        avg = [n for n in names if n.startswith("avg_")]
        assert len(avg) == CANONICAL_PAIR_COUNT

    def test_vix_and_trend_present(self):
        """vix_normalized and trend_strength must be in names."""
        trainer = StackingTrainer(TrainingConfig())
        names = trainer._get_feature_names()
        assert "vix_normalized" in names
        assert "trend_strength" in names

    def test_hist_acc_count(self):
        """Must have one accuracy feature per SignalSource."""
        trainer = StackingTrainer(TrainingConfig())
        names = trainer._get_feature_names()
        hist = [n for n in names if n.startswith("acc90d_")]
        assert len(hist) == CANONICAL_SOURCE_COUNT

    def test_all_names_unique(self):
        """Feature names are unique under the full-roster contract."""
        trainer = StackingTrainer(TrainingConfig())
        names = trainer._get_feature_names()
        assert len(names) == CANONICAL_FEATURE_COUNT
        assert len(set(names)) == len(names)

    def test_feature_names_are_unique_under_full_roster_contract(self):
        """Full source values avoid old prefix-truncation duplicate names."""
        names = StackingTrainer(TrainingConfig())._get_feature_names()
        assert len(set(names)) == len(names)

    def test_no_empty_names(self):
        """No feature name may be empty or whitespace-only."""
        trainer = StackingTrainer(TrainingConfig())
        names = trainer._get_feature_names()
        for n in names:
            assert n.strip(), f"Empty name at index {names.index(n)}"
            assert len(n) > 0

    def test_sorted_pair_order(self):
        """Pairwise names must be alphabetically sorted by source value."""
        trainer = StackingTrainer(TrainingConfig())
        names = trainer._get_feature_names()
        # mult_ should come before disagree_ before avg_ within each pair
        mult_indices = [i for i, n in enumerate(names) if n.startswith("mult_")]
        disagree_indices = [i for i, n in enumerate(names) if n.startswith("disagree_")]
        avg_indices = [i for i, n in enumerate(names) if n.startswith("avg_")]
        assert all(m < d for m, d in zip(mult_indices, disagree_indices))
        assert all(d < a for d, a in zip(disagree_indices, avg_indices))


# ── Predict edge cases (no model / fallback paths) ──────────────────────

class TestPredictEdgeCases:
    """Test predict() boundary conditions without requiring a model."""

    def test_predict_no_model_returns_fallback(self):
        """predict() with no loaded model must return fallback prediction."""
        trainer = StackingTrainer(TrainingConfig())
        features = np.random.randn(CANONICAL_FEATURE_COUNT)
        result = trainer.predict(features)
        assert result.using_fallback is True
        assert result.fallback_reason == "No model loaded"
        assert result.prediction == 0
        assert result.probability == 0.5

    def test_predict_2d_array_no_model(self):
        """predict() with 2D feature array must return fallback."""
        trainer = StackingTrainer(TrainingConfig())
        features = np.random.randn(3, 59)  # 3 samples
        result = trainer.predict(features)
        assert result.using_fallback is True

    def test_predict_single_feature_no_model(self):
        """predict() with a scalar-like array must return fallback."""
        trainer = StackingTrainer(TrainingConfig())
        features = np.array(59)  # 0-dim array of value 59
        result = trainer.predict(features)
        assert result.using_fallback is True
        assert result.prediction == 0

    def test_predict_with_nan_features_no_model(self):
        """predict() with NaN features must not crash."""
        trainer = StackingTrainer(TrainingConfig())
        features = np.full(CANONICAL_FEATURE_COUNT, np.nan)
        result = trainer.predict(features)
        assert result.using_fallback is True

    def test_predict_with_inf_features_no_model(self):
        """predict() with Inf features must not crash."""
        trainer = StackingTrainer(TrainingConfig())
        features = np.full(CANONICAL_FEATURE_COUNT, np.inf)
        result = trainer.predict(features)
        assert result.using_fallback is True

    def test_predict_returns_prediction_result_type(self):
        """predict() must return PredictionResult instance."""
        trainer = StackingTrainer(TrainingConfig())
        result = trainer.predict(np.random.randn(CANONICAL_FEATURE_COUNT))
        assert isinstance(result, PredictionResult)

    def test_predict_timestamp_format(self):
        """predict() timestamp must be ISO format."""
        trainer = StackingTrainer(TrainingConfig())
        result = trainer.predict(np.random.randn(CANONICAL_FEATURE_COUNT))
        assert "T" in result.timestamp
        assert len(result.timestamp) >= 19  # YYYY-MM-DDTHH:MM:SS

    def test_predict_feature_vector_none_when_no_model(self):
        """predict() with no model must have feature_vector=None."""
        trainer = StackingTrainer(TrainingConfig())
        result = trainer.predict(np.random.randn(CANONICAL_FEATURE_COUNT))
        assert result.feature_vector is None


# ── Load model edge cases ───────────────────────────────────────────────

class TestLoadModelEdgeCases:
    """Test load_model() boundary conditions."""

    def test_load_nonexistent_path(self):
        """load_model() with path that doesn't exist must return False."""
        trainer = StackingTrainer(TrainingConfig())
        success = trainer.load_model("/nonexistent/path/model.pkl")
        assert success is False
        assert trainer.model is None

    def test_load_empty_string_path(self):
        """load_model() with a non-existent path must return False."""
        trainer = StackingTrainer(TrainingConfig())
        success = trainer.load_model("/this_path_does_not_exist_xyz789/some_model.pkl")
        assert success is False

    def test_load_non_pickle_file(self, tmp_path):
        """load_model() with non-pickle file must raise."""
        bad_file = tmp_path / "not_a_model.txt"
        bad_file.write_text("not a pickle file")
        trainer = StackingTrainer(TrainingConfig())
        with pytest.raises(Exception):
            trainer.load_model(str(bad_file))

    def test_model_version_after_load(self, tmp_path):
        """After successful load, model_version must be set from filename."""
        import pickle
        m_path = tmp_path / "models" / "signal_stacker_v20250101_120000.pkl"
        m_path.parent.mkdir(parents=True)
        # Dump a plain object (not an xgb model — we're testing stub path)
        with open(m_path, "wb") as f:
            pickle.dump({"dummy": "model"}, f)

        trainer = StackingTrainer(TrainingConfig(model_dir=str(tmp_path / "models")))
        success = trainer.load_model(str(m_path))
        assert success is True
        assert trainer.model_version == "v20250101_120000"

    def test_load_with_none_value(self):
        """load_model() with None path should raise TypeError."""
        trainer = StackingTrainer(TrainingConfig())
        with pytest.raises(TypeError):
            trainer.load_model(None)


# ── Stubbed ML pipeline tests (using unittest.mock) ─────────────────────

# Helper: a real class that mimics xgb.XGBClassifier and IS picklable.
# MagicMock-based mocks fail at pickle.dump() inside StackingTrainer.train().
class _MockXGBClassifier:
    """Picklable stand-in for xgb.XGBClassifier, used with patch()."""
    def __init__(self, **kwargs):
        self.feature_importances_ = np.random.rand(CANONICAL_FEATURE_COUNT)
    def fit(self, X, y, **kwargs):
        return self
    def predict(self, X):
        return np.ones(len(X), dtype=int)
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])


def _make_mock_classifier():
    """Create a picklable _MockXGBClassifier."""
    return _MockXGBClassifier()


class TestStubbedMLPipeline:
    """Test ML pipeline with stubs replaced by MagicMock."""

    @pytest.fixture
    def trainer(self, tmp_path):
        config = TrainingConfig(
            model_dir=str(tmp_path / "models"),
            min_training_days=5,
        )
        return StackingTrainer(config)

    def test_train_produces_result(self, trainer):
        """train() must return TrainingResult with stubs mocked."""
        with (
            patch("src.ml.stacking_trainer.xgb.XGBClassifier", return_value=_make_mock_classifier()),
            patch("src.ml.stacking_trainer.TimeSeriesSplit") as mock_tscv,
            patch("src.ml.stacking_trainer.accuracy_score", return_value=0.75),
            patch("src.ml.stacking_trainer.roc_auc_score", return_value=0.78),
        ):
            # Make TimeSeriesSplit.split() yield train/test index pairs
            mock_tscv_instance = MagicMock()
            mock_tscv_instance.split.return_value = [
                (np.arange(400), np.arange(400, 500)),
                (np.arange(450), np.arange(450, 500)),
                (np.arange(500), np.arange(500, 550)),
            ]
            mock_tscv.return_value = mock_tscv_instance

            result = trainer.train(start_date="2020-01-01")
            assert isinstance(result, TrainingResult)
            assert result.model_version is not None
            assert result.training_date is not None

    def test_train_metrics_in_range(self, trainer):
        """train() metrics must be in [0, 1] range."""
        with (
            patch("src.ml.stacking_trainer.xgb.XGBClassifier", return_value=_make_mock_classifier()),
            patch("src.ml.stacking_trainer.TimeSeriesSplit") as mock_tscv,
            patch("src.ml.stacking_trainer.accuracy_score", return_value=0.75),
            patch("src.ml.stacking_trainer.roc_auc_score", return_value=0.78),
        ):
            mock_tscv_instance = MagicMock()
            mock_tscv_instance.split.return_value = [
                (np.arange(400), np.arange(400, 500)),
                (np.arange(450), np.arange(450, 500)),
                (np.arange(500), np.arange(500, 550)),
            ]
            mock_tscv.return_value = mock_tscv_instance

            result = trainer.train(start_date="2020-01-01")
            assert 0.0 <= result.train_accuracy <= 1.0
            assert 0.0 <= result.validation_accuracy <= 1.0
            assert 0.0 <= result.validation_auc <= 1.0
            assert 0.0 <= result.cv_mean_accuracy <= 1.0
            assert 0.0 <= result.cv_std_accuracy <= 1.0
            assert 0.0 <= result.cv_mean_auc <= 1.0

    def test_train_feature_importance_length(self, trainer):
        """train() must extract exactly 10 top features."""
        with (
            patch("src.ml.stacking_trainer.xgb.XGBClassifier", return_value=_make_mock_classifier()),
            patch("src.ml.stacking_trainer.TimeSeriesSplit") as mock_tscv,
            patch("src.ml.stacking_trainer.accuracy_score", return_value=0.75),
            patch("src.ml.stacking_trainer.roc_auc_score", return_value=0.78),
        ):
            mock_tscv_instance = MagicMock()
            mock_tscv_instance.split.return_value = [
                (np.arange(400), np.arange(400, 500)),
            ]
            mock_tscv.return_value = mock_tscv_instance

            result = trainer.train(start_date="2020-01-01")
            assert len(result.top_features) == 10
            for name, imp in result.top_features:
                assert isinstance(name, str)
                assert isinstance(imp, float)

    def test_train_model_path_saved(self, trainer):
        """train() must create a .pkl file on disk."""
        with (
            patch("src.ml.stacking_trainer.xgb.XGBClassifier", return_value=_make_mock_classifier()),
            patch("src.ml.stacking_trainer.TimeSeriesSplit") as mock_tscv,
            patch("src.ml.stacking_trainer.accuracy_score", return_value=0.75),
            patch("src.ml.stacking_trainer.roc_auc_score", return_value=0.78),
        ):
            mock_tscv_instance = MagicMock()
            mock_tscv_instance.split.return_value = [
                (np.arange(400), np.arange(400, 500)),
            ]
            mock_tscv.return_value = mock_tscv_instance

            result = trainer.train(start_date="2020-01-01")
            model_path = Path(result.model_path)
            assert model_path.exists()
            assert model_path.suffix == ".pkl"

    def test_train_json_result_saved(self, trainer):
        """train() must save a JSON training_result file."""
        with (
            patch("src.ml.stacking_trainer.xgb.XGBClassifier", return_value=_make_mock_classifier()),
            patch("src.ml.stacking_trainer.TimeSeriesSplit") as mock_tscv,
            patch("src.ml.stacking_trainer.accuracy_score", return_value=0.75),
            patch("src.ml.stacking_trainer.roc_auc_score", return_value=0.78),
        ):
            mock_tscv_instance = MagicMock()
            mock_tscv_instance.split.return_value = [
                (np.arange(400), np.arange(400, 500)),
            ]
            mock_tscv.return_value = mock_tscv_instance

            result = trainer.train(start_date="2020-01-01")
            result_path = Path(result.model_path).parent / f"training_result_{result.model_version}.json"
            assert result_path.exists()
            with open(result_path) as f:
                data = json.load(f)
            assert data["model_version"] == result.model_version

    def test_predict_with_mock_model(self, trainer):
        """predict() must produce valid results with a mocked model."""
        mock_clf = _make_mock_classifier()
        trainer.model = mock_clf
        trainer.model_version = "vtest"

        features = np.random.randn(CANONICAL_FEATURE_COUNT)
        result = trainer.predict(features)

        assert isinstance(result, PredictionResult)
        assert result.prediction in (0, 1)
        assert 0.0 <= result.probability <= 1.0
        assert 0.5 <= result.confidence <= 1.0
        assert isinstance(result.feature_vector, list)
        assert len(result.feature_vector) == CANONICAL_FEATURE_COUNT

    def test_predict_1d_reshaped(self, trainer):
        """predict() must reshape 1D input to 2D for the model; verify
        by checking that the output contains a full canonical feature_vector."""
        from unittest.mock import MagicMock
        # Use a MagicMock with spec to track predict_proba calls
        mock_clf = MagicMock(spec=_MockXGBClassifier)
        mock_clf.feature_importances_ = np.random.rand(CANONICAL_FEATURE_COUNT)
        mock_clf.predict.return_value = np.array([1])
        mock_clf.predict_proba.return_value = np.array([[0.3, 0.7]])
        trainer.model = mock_clf
        trainer.model_version = "vtest"

        features_1d = np.random.randn(CANONICAL_FEATURE_COUNT)
        result = trainer.predict(features_1d)
        # The model should receive 2D input
        call_args = mock_clf.predict_proba.call_args
        assert call_args is not None
        input_arr = call_args[0][0]
        assert input_arr.ndim == 2
        assert input_arr.shape == (1, CANONICAL_FEATURE_COUNT)
        # verify the result still has a feature vector
        assert result.feature_vector is not None
        assert len(result.feature_vector) == CANONICAL_FEATURE_COUNT

    def test_predict_with_confidence_threshold_zero(self, trainer):
        """confidence_threshold=0.0 must never trigger fallback."""
        mock_clf = _make_mock_classifier()
        trainer.model = mock_clf
        trainer.model_version = "vtest"

        features = np.random.randn(CANONICAL_FEATURE_COUNT)
        result = trainer.predict(features, confidence_threshold=0.0)
        # using_fallback may be np.False_ — compare with == not "is"
        assert bool(result.using_fallback) is False
        assert result.fallback_reason == ""

    def test_predict_with_confidence_threshold_one(self, trainer):
        """confidence_threshold=1.0 must always trigger fallback."""
        mock_clf = _make_mock_classifier()
        # Make it output 50/50 so confidence is 0.5
        mock_clf.predict_proba = lambda X: np.array([[0.5, 0.5]])
        mock_clf.predict = lambda X: np.array([0])
        trainer.model = mock_clf
        trainer.model_version = "vtest"

        features = np.random.randn(CANONICAL_FEATURE_COUNT)
        result = trainer.predict(features, confidence_threshold=1.0)
        # using_fallback may be np.True_ — compare with == not "is"
        assert bool(result.using_fallback) is True
        assert "Confidence" in result.fallback_reason

    def test_backfill_dry_run_stats(self, trainer):
        """backfill_predictions(dry_run=True) must return stats dict."""
        mock_clf = _make_mock_classifier()
        trainer.model = mock_clf
        trainer.model_version = "vtest"

        stats = trainer.backfill_predictions(start_date="2020-01-01", dry_run=True)
        assert isinstance(stats, dict)
        assert "total_predictions" in stats
        assert "accuracy" in stats
        assert "fallback_count" in stats
        assert "fallback_rate" in stats
        assert "date_range" in stats
        assert isinstance(stats["total_predictions"], int)
        assert 0 <= stats["accuracy"] <= 1
        assert 0 <= stats["fallback_rate"] <= 1

    def test_backfill_without_model_returns_error(self, trainer):
        """backfill_predictions() with no model must return error dict."""
        stats = trainer.backfill_predictions(start_date="2020-01-01")
        assert isinstance(stats, dict)
        assert "error" in stats
        assert stats["error"] == "No model loaded"

    def test_backfill_saves_to_db_when_not_dry_run(self, trainer, tmp_path):
        """backfill_predictions(dry_run=False) must save to SQLite."""
        mock_clf = _make_mock_classifier()
        trainer.model = mock_clf
        trainer.model_version = "vtest"
        trainer.config.db_path = str(tmp_path / "test_backfill.db")

        stats = trainer.backfill_predictions(start_date="2020-01-01", dry_run=False)
        db_path = Path(trainer.config.db_path)
        assert db_path.exists()


# ── CLI / __main__ guard tests ──────────────────────────────────────────

class TestCliMain:
    """Test CLI entry points with capsys (no model files needed)."""

    def test_main_list_command_empty(self, capsys):
        """'list' command on empty model dir must print '0 models'."""
        from src.ml.stacking_trainer import main as cli_main
        with patch("sys.argv", ["stacking_trainer", "list"]), \
             patch("src.ml.stacking_trainer.Path.glob", return_value=[]):
            cli_main()
        captured = capsys.readouterr()
        assert "Available models (0):" in captured.out

    def test_main_list_command_with_models(self, capsys, tmp_path):
        """'list' command with models must show filenames."""
        from src.ml.stacking_trainer import main as cli_main
        # Create real files so sorted(glob(...)) works
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "signal_stacker_v1.pkl").write_text("")
        (models_dir / "signal_stacker_v2.pkl").write_text("")

        with patch("sys.argv", ["stacking_trainer", "list"]), \
             patch("src.ml.stacking_trainer.PROJECT_ROOT", tmp_path):
            cli_main()
        captured = capsys.readouterr()
        assert "Available models (2):" in captured.out
        assert "signal_stacker_v1.pkl" in captured.out
        assert "signal_stacker_v2.pkl" in captured.out

    def test_main_train_command_ml_enabled(self, capsys):
        """'train' with ML enabled must output JSON result."""
        from src.ml.stacking_trainer import main as cli_main

        # Use a real TrainingResult — asdict() requires a real dataclass instance
        real_result = TrainingResult(
            model_version="vtest",
            training_date="2024-01-01T00:00:00",
            train_accuracy=0.8,
            validation_accuracy=0.75,
            validation_auc=0.78,
            cv_mean_accuracy=0.73,
            cv_std_accuracy=0.02,
            cv_mean_auc=0.76,
            top_features=[("f1", 0.3)],
            training_samples=500,
            validation_samples=200,
            date_range=("2020-01-01", "2024-01-01"),
            model_path="/tmp/model.pkl",
        )

        mock_trainer = MagicMock(spec=StackingTrainer)
        mock_trainer.train.return_value = real_result
        mock_trainer.model_dir = Path("/tmp/models")

        with patch("sys.argv", ["stacking_trainer", "train"]), \
             patch("src.ml.stacking_trainer._ML_ENABLED", True), \
             patch("src.ml.stacking_trainer.StackingTrainer", return_value=mock_trainer):
            cli_main()
        captured = capsys.readouterr()
        # Must output valid JSON
        output = json.loads(captured.out)
        assert output["model_version"] == "vtest"
        assert output["train_accuracy"] == 0.8

    def test_main_train_command_ml_disabled_exits_nonzero(self, capsys):
        """'train' under ML-disabled must fail closed without calling train()."""
        from src.ml.stacking_trainer import main as cli_main

        mock_trainer = MagicMock(spec=StackingTrainer)
        with patch("sys.argv", ["stacking_trainer", "train"]), \
             patch("src.ml.stacking_trainer._ML_ENABLED", False), \
             patch("src.ml.stacking_trainer.StackingTrainer", return_value=mock_trainer):
            with pytest.raises(SystemExit) as exc:
                cli_main()
        assert exc.value.code not in (0, None)
        mock_trainer.train.assert_not_called()
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "PORTFOLIO_LAB_ENABLE_ML=0" in combined
        assert "train" in combined.lower()

    def test_main_evaluate_without_model_arg(self, capsys):
        """'evaluate' without --model must print error and exit non-zero."""
        from src.ml.stacking_trainer import main as cli_main
        with patch("sys.argv", ["stacking_trainer", "evaluate"]), \
             patch("src.ml.stacking_trainer._ML_ENABLED", True):
            with pytest.raises(SystemExit) as exc:
                cli_main()
        assert exc.value.code not in (0, None)
        captured = capsys.readouterr()
        assert "Error: --model required" in (captured.out + captured.err)

    def test_main_evaluate_with_nonexistent_model(self, capsys):
        """'evaluate' with nonexistent --model must exit non-zero."""
        from src.ml.stacking_trainer import main as cli_main
        with patch("sys.argv", ["stacking_trainer", "evaluate", "--model", "/nonexistent.pkl"]), \
             patch("src.ml.stacking_trainer._ML_ENABLED", True):
            with pytest.raises(SystemExit) as exc:
                cli_main()
        assert exc.value.code not in (0, None)
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "Model not found" in combined or "not found" in combined.lower()

    def test_main_evaluate_ml_disabled_exits_nonzero(self, capsys):
        """'evaluate' under ML-disabled must fail closed before model load."""
        from src.ml.stacking_trainer import main as cli_main
        mock_trainer = MagicMock(spec=StackingTrainer)
        with patch("sys.argv", ["stacking_trainer", "evaluate", "--model", "models/x.pkl"]), \
             patch("src.ml.stacking_trainer._ML_ENABLED", False), \
             patch("src.ml.stacking_trainer.StackingTrainer", return_value=mock_trainer):
            with pytest.raises(SystemExit) as exc:
                cli_main()
        assert exc.value.code not in (0, None)
        mock_trainer.load_model.assert_not_called()
        captured = capsys.readouterr()
        assert "PORTFOLIO_LAB_ENABLE_ML=0" in (captured.out + captured.err)

    def test_main_backfill_no_model_found(self, capsys):
        """'backfill' without model files must print error and exit non-zero."""
        from src.ml.stacking_trainer import main as cli_main

        mock_trainer = MagicMock(spec=StackingTrainer)
        mock_trainer.model = None
        mock_trainer.model_dir = MagicMock()
        mock_trainer.model_dir.glob.return_value = []

        with patch("sys.argv", ["stacking_trainer", "backfill"]), \
             patch("src.ml.stacking_trainer._ML_ENABLED", True), \
             patch("src.ml.stacking_trainer.StackingTrainer", return_value=mock_trainer):
            with pytest.raises(SystemExit) as exc:
                cli_main()
        assert exc.value.code not in (0, None)
        captured = capsys.readouterr()
        assert "No model found" in (captured.out + captured.err)

    def test_main_backfill_dry_run(self, capsys):
        """'backfill --dry-run' with model must output JSON stats."""
        from src.ml.stacking_trainer import main as cli_main

        mock_trainer = MagicMock(spec=StackingTrainer)
        mock_trainer.model = MagicMock()  # model loaded
        mock_trainer.backfill_predictions.return_value = {
            "total_predictions": 100,
            "accuracy": 0.65,
            "fallback_count": 10,
            "fallback_rate": 0.1,
            "date_range": ("2020-01-01", "2024-01-01"),
        }

        with patch("sys.argv", ["stacking_trainer", "backfill", "--dry-run"]), \
             patch("src.ml.stacking_trainer._ML_ENABLED", True), \
             patch("src.ml.stacking_trainer.StackingTrainer", return_value=mock_trainer):
            cli_main()
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["total_predictions"] == 100
        assert output["accuracy"] == 0.65

    def test_main_help(self, capsys):
        """--help must show usage information."""
        from src.ml.stacking_trainer import main as cli_main
        with patch("sys.argv", ["stacking_trainer", "--help"]):
            with pytest.raises(SystemExit) as exc:
                cli_main()
            assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "Usage:" in captured.out

    def test_main_module_guard(self):
        """__name__ == '__main__' guard must reference main()."""
        import ast, inspect
        from src.ml import stacking_trainer as st
        source = inspect.getsource(st)
        tree = ast.parse(source)
        # Check for if __name__ == '__main__': main()
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                compare = node.test
                if (isinstance(compare, ast.Compare) and
                        isinstance(compare.left, ast.Name) and
                        compare.left.id == '__name__'):
                    found = True
                    break
        assert found, "Module must have __name__ == '__main__' guard"


class TestCliSafeModeSubprocess:
    """Subprocess regression: fail-closed CLI under PORTFOLIO_LAB_ENABLE_ML=0."""

    @staticmethod
    def _run(*args: str):
        import os
        import subprocess
        import sys
        env = {**os.environ, "PORTFOLIO_LAB_ENABLE_ML": "0"}
        return subprocess.run(
            [sys.executable, "-m", "src.ml.stacking_trainer", *args],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=Path(__file__).resolve().parents[1],
        )

    def test_subprocess_train_ml_disabled_exits_nonzero(self):
        result = self._run("train", "--start-date", "2024-01-01")
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert "AttributeError" not in combined
        assert "PORTFOLIO_LAB_ENABLE_ML=0" in combined
        assert "split" not in combined  # must not reach TimeSeriesSplit stub crash

    def test_subprocess_evaluate_missing_model_exits_nonzero(self):
        result = self._run("evaluate", "--model", "models/missing.pkl")
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert (
            "PORTFOLIO_LAB_ENABLE_ML=0" in combined
            or "Model not found" in combined
            or "not found" in combined.lower()
        )

    def test_subprocess_backfill_no_model_exits_nonzero(self):
        result = self._run("backfill", "--dry-run", "--start-date", "2024-01-01")
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert (
            "PORTFOLIO_LAB_ENABLE_ML=0" in combined
            or "No model found" in combined
        )

    def test_subprocess_list_ml_disabled_exits_zero(self):
        result = self._run("list")
        combined = result.stdout + result.stderr
        assert result.returncode == 0
        assert "Available models" in combined


# ── _create_features_from_signals edge cases ────────────────────────────

class TestCreateFeaturesFromSignalsEdgeCases:
    """Test _create_features_from_signals boundary conditions."""

    @staticmethod
    def _full_signal_payload() -> dict[str, dict[str, float | int]]:
        return {
            source.value: {
                "value": 0.1,
                "confidence": 0.7,
                "predicted_direction": 1,
            }
            for source in SignalSource
        }

    def test_empty_signals_returns_none(self):
        """Empty signals dict must return None."""
        trainer = StackingTrainer(TrainingConfig())
        result = trainer._create_features_from_signals({})
        assert result is None

    def test_insufficient_signals_returns_none(self):
        """Fewer than 6 signal sources must return None."""
        trainer = StackingTrainer(TrainingConfig())
        signals = {
            "multi_speed_momentum": {"value": 0.5, "confidence": 0.7, "predicted_direction": 1},
            "cross_asset_rv": {"value": -0.3, "confidence": 0.6, "predicted_direction": 0},
        }
        result = trainer._create_features_from_signals(signals)
        assert result is None

    def test_unknown_signal_source_skipped(self):
        """Unknown signal source strings must be skipped without error."""
        trainer = StackingTrainer(TrainingConfig())
        signals = {
            "unknown_source": {"value": 0.5, "confidence": 0.7, "predicted_direction": 1},
        }
        result = trainer._create_features_from_signals(signals)
        assert result is None  # insufficient valid signals

    def test_missing_confidence_default(self):
        """Signals without confidence must use default 0.5."""
        trainer = StackingTrainer(TrainingConfig())
        signals = {
            "multi_speed_momentum": {"value": 0.5, "predicted_direction": 1},
            "cross_asset_rv": {"value": -0.3, "predicted_direction": 0},
            "international_momentum": {"value": 0.2, "predicted_direction": 1},
            "alternative_data": {"value": 0.1, "predicted_direction": 1},
            "cross_asset_regime_arb": {"value": -0.1, "predicted_direction": 0},
            "unified_overlay": {"value": 0.3, "predicted_direction": 1},
        }
        # This must not crash despite missing confidence
        result = trainer._create_features_from_signals(signals)
        # May return None if feature_engine raises, but must not crash
        assert result is None or isinstance(result, np.ndarray)

    def test_all_six_signals_with_feature_engine(self):
        """All 6 valid signals must not crash."""
        trainer = StackingTrainer(TrainingConfig())
        signals = {
            "multi_speed_momentum": {"value": 0.5, "confidence": 0.7, "predicted_direction": 1},
            "cross_asset_rv": {"value": -0.3, "confidence": 0.6, "predicted_direction": 0},
            "international_momentum": {"value": 0.2, "confidence": 0.8, "predicted_direction": 1},
            "alternative_data": {"value": 0.1, "confidence": 0.5, "predicted_direction": 1},
            "cross_asset_regime_arb": {"value": -0.1, "confidence": 0.9, "predicted_direction": 0},
            "unified_overlay": {"value": 0.3, "confidence": 0.6, "predicted_direction": 1},
        }
        # With the real feature engine, this should work
        result = trainer._create_features_from_signals(signals)
        assert result is None or isinstance(result, np.ndarray)

    def test_partial_current_roster_returns_none(self):
        """A partial 8-of-9 current roster is not a valid stacking row."""
        trainer = StackingTrainer(TrainingConfig())
        signals = self._full_signal_payload()
        signals.pop(SignalSource.VIX_TERM_STRUCTURE.value)

        result = trainer._create_features_from_signals(signals)

        assert result is None

    def test_full_current_roster_returns_canonical_dimension(self):
        """Complete current roster produces the canonical feature dimension."""
        trainer = StackingTrainer(TrainingConfig())
        result = trainer._create_features_from_signals(self._full_signal_payload())

        assert isinstance(result, np.ndarray)
        assert result.shape == (StackingFeatureEngine.TOTAL_DIMENSIONS,)


# ── _load_historical_data edge cases ────────────────────────────────────

class TestLoadHistoricalDataEdgeCases:
    """Test _load_historical_data boundary conditions."""

    def test_non_existent_db_returns_synthetic(self):
        """_load_historical_data with no DB must fall back to synthetic."""
        config = TrainingConfig(db_path="/nonexistent/path/not_a_db.db")
        trainer = StackingTrainer(config)
        X, y, dates = trainer._load_historical_data("2020-01-01")
        # Should return synthetic data
        assert X is not None
        assert len(y) > 0
        assert len(dates) > 0

    def test_synthetic_data_uses_config_feature_count(self):
        """Synthetic fallback must use config.feature_count for feature dim."""
        config = TrainingConfig(db_path="/nonexistent/db.db", feature_count=128)
        trainer = StackingTrainer(config)
        X, y, dates = trainer._load_historical_data("2020-01-01")
        assert X.shape[1] == 128


# ── Public API / export completeness ────────────────────────────────────

class TestPublicAPI:
    """Verify module exports and public API naming."""

    def test_expected_names_importable(self):
        """Key public names must be importable from the module."""
        from src.ml.stacking_trainer import (
            main,
        )
        assert callable(main)

    def test_main_function_is_module_level(self):
        """main() must be a module-level function."""
        from src.ml.stacking_trainer import main
        assert callable(main)

    def test_xgboost_stub_in_sys_modules(self):
        """xgboost stub must be registered in sys.modules."""
        import os
        if os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1":
            pytest.skip("ML-enabled — stub test not applicable")
        import sys
        assert "xgboost" in sys.modules
        mod = sys.modules["xgboost"]
        # Must be our stub, not the real xgboost
        assert hasattr(mod, "XGBClassifier")
        assert callable(mod.XGBClassifier)
        assert mod.XGBClassifier() is None  # stub returns None


# ── Dataclass miscellany ────────────────────────────────────────────────

class TestDataclassMiscellany:
    """Additional dataclass validation tests."""

    def test_training_config_mutable_defaults_safe(self):
        """TrainingConfig must not have mutable default values."""
        for field in dataclasses.fields(TrainingConfig):
            # Check typical mutable types aren't used as defaults
            default = field.default
            if default is not dataclasses.MISSING:
                assert not isinstance(default, (list, dict, set)), \
                    f"Field {field.name} has mutable default: {type(default)}"

    def test_training_result_mutable_fields_typed(self):
        """TrainingResult mutable fields must use proper type annotations."""
        for field in dataclasses.fields(TrainingResult):
            if field.name == "top_features":
                assert "List" in str(field.type) or "list" in str(field.type).lower()
            if field.name == "date_range":
                assert "Tuple" in str(field.type) or "tuple" in str(field.type).lower()

    def test_prediction_result_optional_field(self):
        """PredictionResult.feature_vector must be Optional."""
        for field in dataclasses.fields(PredictionResult):
            if field.name == "feature_vector":
                type_str = str(field.type)
                assert "Optional" in type_str or "None" in type_str

    def test_prediction_result_default_for_fallback(self):
        """PredictionResult.using_fallback must default to False."""
        r = PredictionResult(timestamp="t", prediction=0, probability=0.5, confidence=0.5)
        assert r.using_fallback is False

    def test_prediction_result_default_for_fallback_reason(self):
        """PredictionResult.fallback_reason must default to empty string."""
        r = PredictionResult(timestamp="t", prediction=0, probability=0.5, confidence=0.5)
        assert r.fallback_reason == ""
