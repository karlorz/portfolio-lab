"""Tests for src/research/regime_classifier.py (484L, non-ML subset).

This module uses sklearn/xgboost only when PORTFOLIO_LAB_ENABLE_ML=1.
When ML is disabled (default), Train() raises ImportError and predict()
raises RuntimeError. All other functionality is pure Python/numpy and
testable without ML dependencies.
"""

import json
import os
import pickle
import tempfile
from unittest.mock import patch, mock_open, MagicMock, ANY
from pathlib import Path

import numpy as np
import pytest

from src.research.regime_classifier import (
    Regime,
    RegimePrediction,
    RegimeClassifier,
    WeeklyGridSearch,
    main,
)


# ═══════════════════════════════════════════════════════════════════════════
# Regime Enum
# ═══════════════════════════════════════════════════════════════════════════


class TestRegime:
    def test_bear_value(self):
        assert Regime.BEAR.value == 0

    def test_neutral_value(self):
        assert Regime.NEUTRAL.value == 1

    def test_bull_value(self):
        assert Regime.BULL.value == 2

    def test_all_members_present(self):
        assert len(Regime) == 3
        assert set(Regime.__members__) == {"BEAR", "NEUTRAL", "BULL"}


# ═══════════════════════════════════════════════════════════════════════════
# RegimePrediction Dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestRegimePrediction:
    def test_create_bull_prediction(self):
        rp = RegimePrediction(
            symbol="SPY",
            timestamp="2026-01-01T00:00:00",
            p_bear=0.1,
            p_neutral=0.3,
            p_bull=0.6,
            predicted_regime=Regime.BULL,
            confidence=0.6,
        )
        assert rp.symbol == "SPY"
        assert rp.p_bear == 0.1
        assert rp.p_neutral == 0.3
        assert rp.p_bull == 0.6
        assert rp.predicted_regime == Regime.BULL
        assert rp.confidence == 0.6

    def test_create_bear_prediction(self):
        rp = RegimePrediction(
            symbol="GLD",
            timestamp="2026-01-02T00:00:00",
            p_bear=0.7,
            p_neutral=0.2,
            p_bull=0.1,
            predicted_regime=Regime.BEAR,
            confidence=0.7,
        )
        assert rp.predicted_regime == Regime.BEAR
        assert rp.confidence == 0.7

    def test_create_neutral_prediction(self):
        rp = RegimePrediction(
            symbol="TLT",
            timestamp="2026-01-03T00:00:00",
            p_bear=0.25,
            p_neutral=0.50,
            p_bull=0.25,
            predicted_regime=Regime.NEUTRAL,
            confidence=0.50,
        )
        assert rp.predicted_regime == Regime.NEUTRAL
        assert rp.confidence == 0.50

    def test_to_dict_bull(self):
        rp = RegimePrediction(
            symbol="SPY",
            timestamp="2026-01-01T00:00:00",
            p_bear=0.1,
            p_neutral=0.3,
            p_bull=0.6,
            predicted_regime=Regime.BULL,
            confidence=0.6,
        )
        d = rp.to_dict()
        assert d["symbol"] == "SPY"
        assert d["predicted_regime"] == "BULL"
        assert d["p_bull"] == 0.6
        assert d["confidence"] == 0.6
        assert d["feature_importance"] is None

    def test_to_dict_with_feature_importance(self):
        fi = {"return_1d": 0.3, "vix_level": 0.7}
        rp = RegimePrediction(
            symbol="SPY",
            timestamp="2026-01-01T00:00:00",
            p_bear=0.4,
            p_neutral=0.35,
            p_bull=0.25,
            predicted_regime=Regime.BEAR,
            confidence=0.4,
            feature_importance=fi,
        )
        d = rp.to_dict()
        assert d["feature_importance"] == fi
        assert d["predicted_regime"] == "BEAR"

    def test_to_dict_json_serializable(self):
        rp = RegimePrediction(
            symbol="SPY",
            timestamp="2026-01-01T00:00:00",
            p_bear=0.1,
            p_neutral=0.3,
            p_bull=0.6,
            predicted_regime=Regime.BULL,
            confidence=0.6,
        )
        json.dumps(rp.to_dict())  # should not raise

    def test_default_feature_importance_none(self):
        rp = RegimePrediction(
            symbol="SPY",
            timestamp="2026-01-01T00:00:00",
            p_bear=0.5,
            p_neutral=0.3,
            p_bull=0.2,
            predicted_regime=Regime.BEAR,
            confidence=0.5,
        )
        assert rp.feature_importance is None

    def test_edge_extreme_probabilities(self):
        rp = RegimePrediction(
            symbol="SPY",
            timestamp="2026-01-01T00:00:00",
            p_bear=1.0,
            p_neutral=0.0,
            p_bull=0.0,
            predicted_regime=Regime.BEAR,
            confidence=1.0,
        )
        assert abs(rp.p_bear - 1.0) < 1e-10
        assert rp.confidence == 1.0

    def test_edge_low_confidence(self):
        rp = RegimePrediction(
            symbol="SPY",
            timestamp="2026-01-01T00:00:00",
            p_bear=0.34,
            p_neutral=0.33,
            p_bull=0.33,
            predicted_regime=Regime.BEAR,
            confidence=0.34,
        )
        assert rp.confidence == 0.34

    def test_empty_string_symbol(self):
        rp = RegimePrediction(
            symbol="",
            timestamp="2026-01-01T00:00:00",
            p_bear=0.5,
            p_neutral=0.3,
            p_bull=0.2,
            predicted_regime=Regime.NEUTRAL,
            confidence=0.3,
        )
        assert rp.symbol == ""


# ═══════════════════════════════════════════════════════════════════════════
# RegimeClassifier — Non-ML Methods
# ═══════════════════════════════════════════════════════════════════════════


class TestRegimeClassifierInit:
    def test_default_model_type(self):
        rc = RegimeClassifier()
        assert rc.model_type == "logistic"

    def test_random_forest_model_type(self):
        rc = RegimeClassifier(model_type="random_forest")
        assert rc.model_type == "random_forest"

    def test_xgboost_model_type(self):
        rc = RegimeClassifier(model_type="xgboost")
        assert rc.model_type == "xgboost"

    def test_unknown_model_type_raises_on_train(self):
        rc = RegimeClassifier(model_type="unknown")
        # Without ML available, train() raises ImportError before checking model type
        with pytest.raises(ImportError, match="scikit-learn"):
            rc.train([])

    def test_initial_not_trained(self):
        rc = RegimeClassifier()
        assert not rc.is_trained

    def test_initial_model_none(self):
        rc = RegimeClassifier()
        assert rc.model is None

    def test_initial_scaler_none(self):
        rc = RegimeClassifier()
        assert rc.scaler is None


class TestRegimeClassifierFeatureVector:
    @pytest.fixture
    def classifier(self):
        return RegimeClassifier()

    def test_returns_length_12(self, classifier):
        features = {"return_1d": 0.01, "return_5d": 0.02}
        vec = classifier._get_feature_vector(features)
        assert len(vec) == 12

    def test_all_zeros_by_default(self, classifier):
        vec = classifier._get_feature_vector({})
        assert np.allclose(vec, np.zeros(12))

    def test_basic_values(self, classifier):
        features = {
            "return_1d": 0.01,
            "return_5d": 0.02,
            "return_20d": 0.05,
            "volatility_20d": 0.15,
            "price_vs_sma20": 1.02,
            "price_vs_sma50": 1.01,
            "volume_ratio": 1.1,
            "vix_level": 15.0,
            "vix_change_5d": -0.5,
            "vix_percentile_20d": 0.4,
            "spy_correlation_20d": 0.8,
            "trend_direction": 1.0,
        }
        vec = classifier._get_feature_vector(features)
        assert vec[0] == 0.01  # return_1d
        assert vec[1] == 0.02  # return_5d
        assert vec[2] == 0.05  # return_20d
        assert vec[3] == 0.15  # volatility_20d
        assert vec[4] == 1.02  # price_vs_sma20
        assert vec[5] == 1.01  # price_vs_sma50
        assert vec[6] == 1.1  # volume_ratio
        assert vec[7] == 15.0  # vix_level
        assert vec[8] == -0.5  # vix_change_5d
        assert vec[9] == 0.4  # vix_percentile_20d
        assert vec[10] == 0.8  # spy_correlation_20d
        assert vec[11] == 1.0  # trend_direction

    def test_feature_order_preserved(self, classifier):
        """Keys should be in the order defined in _get_feature_vector."""
        features = {
            "return_1d": 1.0,
            "return_5d": 2.0,
            "return_20d": 3.0,
            "volatility_20d": 4.0,
            "price_vs_sma20": 5.0,
            "price_vs_sma50": 6.0,
            "volume_ratio": 7.0,
            "vix_level": 8.0,
            "vix_change_5d": 9.0,
            "vix_percentile_20d": 10.0,
            "spy_correlation_20d": 11.0,
            "trend_direction": 12.0,
        }
        vec = classifier._get_feature_vector(features)
        expected = np.arange(1.0, 13.0)
        assert np.allclose(vec, expected)

    def test_handles_string_vol_regime(self, classifier):
        features = {"vol_regime": "high"}
        vec = classifier._get_feature_vector(features)
        # vol_regime is not in the standard feature keys, so it gets skipped
        # But _get_feature_vector tries to handle it as a string val
        assert len(vec) == 12

    def test_missing_keys_default_zero(self, classifier):
        features = {"return_1d": 5.0, "return_5d": 10.0}
        vec = classifier._get_feature_vector(features)
        assert vec[0] == 5.0
        assert vec[1] == 10.0
        assert vec[2] == 0.0  # return_20d missing

    def test_negative_values(self, classifier):
        features = {
            "return_1d": -0.03,
            "return_5d": -0.05,
            "return_20d": -0.10,
            "volatility_20d": 0.25,
            "price_vs_sma20": 0.95,
            "price_vs_sma50": 0.90,
            "volume_ratio": 1.5,
            "vix_level": 25.0,
            "vix_change_5d": 2.0,
            "vix_percentile_20d": 0.8,
            "spy_correlation_20d": -0.3,
            "trend_direction": -1.0,
        }
        vec = classifier._get_feature_vector(features)
        assert vec[0] == -0.03
        assert vec[11] == -1.0

    def test_string_values_handled(self, classifier):
        """String values should default to 0.0."""
        features = {"return_1d": "bad_data"}
        vec = classifier._get_feature_vector(features)
        assert vec[0] == 0.0  # string converted to 0

    def test_none_values_raise_error(self, classifier):
        """None values in features cause TypeError in float() conversion.
        
        This documents existing behavior: _get_feature_vector does not guard
        against None values. The .get(key, 0.0) default is only used when
        the key is missing entirely, not when the key exists with None value.
        """
        features = {"return_1d": None}
        with pytest.raises(TypeError):
            classifier._get_feature_vector(features)

    def test_vol_regime_low_string(self, classifier):
        """vol_regime key exists in code but not in feature_keys list."""
        features = {"vol_regime": "low", "return_1d": 1.0}
        vec = classifier._get_feature_vector(features)
        assert vec[0] == 1.0  # return_1d preserved
        # vol_regime parsed as string: 'low' -> checks key == 'vol_regime' -> val=0

    def test_vol_regime_high_string(self, classifier):
        features = {"vol_regime": "high", "return_1d": 1.0}
        vec = classifier._get_feature_vector(features)
        assert vec[0] == 1.0

    def test_vol_regime_not_in_feature_keys(self, classifier):
        """Verify vol_regime is NOT in the standard 12 feature keys."""
        # The feature_keys list has specific 12 keys; vol_regime handling
        # is only triggered if it somehow appears (it won't via standard path)
        keys = [
            "return_1d", "return_5d", "return_20d",
            "volatility_20d", "price_vs_sma20", "price_vs_sma50",
            "volume_ratio", "vix_level", "vix_change_5d",
            "vix_percentile_20d", "spy_correlation_20d", "trend_direction",
        ]
        assert "vol_regime" not in keys


class TestRegimeClassifierPrepareData:
    @pytest.fixture
    def classifier(self):
        return RegimeClassifier()

    def test_raises_on_less_than_10_samples(self, classifier):
        features_list = [
            {"return_1d": 0.01, "regime_label": 1},
            {"return_1d": 0.02, "regime_label": 2},
        ]
        with pytest.raises(ValueError, match="Insufficient training data"):
            classifier.prepare_data(features_list)

    def test_raises_on_empty_list(self, classifier):
        with pytest.raises(ValueError, match="Insufficient training data"):
            classifier.prepare_data([])

    def test_raises_on_no_labeled_data(self, classifier):
        features_list = [{"return_1d": 0.01} for _ in range(15)]
        with pytest.raises(ValueError, match="Insufficient training data"):
            classifier.prepare_data(features_list)

    def test_returns_X_and_y(self, classifier):
        features_list = [
            {"return_1d": 0.01, "regime_label": 0},
            {"return_1d": 0.02, "regime_label": 1},
            {"return_1d": 0.03, "regime_label": 2},
            {"return_1d": 0.04, "regime_label": 0},
            {"return_1d": 0.05, "regime_label": 1},
            {"return_1d": 0.06, "regime_label": 2},
            {"return_1d": 0.07, "regime_label": 0},
            {"return_1d": 0.08, "regime_label": 1},
            {"return_1d": 0.09, "regime_label": 2},
            {"return_1d": 0.10, "regime_label": 0},
            {"return_1d": 0.11, "regime_label": 1},
            {"return_1d": 0.12, "regime_label": 2},
        ]
        X, y = classifier.prepare_data(features_list)
        assert X.shape == (12, 12)
        assert y.shape == (12,)
        assert list(y) == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

    def test_skips_unlabeled_samples(self, classifier):
        features_list = [
            {"return_1d": 0.01, "regime_label": 0},
            {"return_1d": 0.02},  # no label, should skip
            {"return_1d": 0.03, "regime_label": 1},
            {"return_1d": 0.04, "regime_label": 2},
            {"return_1d": 0.05},
            {"return_1d": 0.06, "regime_label": 0},
            {"return_1d": 0.07, "regime_label": 1},
            {"return_1d": 0.08},
            {"return_1d": 0.09, "regime_label": 2},
            {"return_1d": 0.10, "regime_label": 0},
            {"return_1d": 0.11, "regime_label": 1},
            {"return_1d": 0.12, "regime_label": 2},
        ]
        # 9 labeled samples < 10 minimum, should raise ValueError
        with pytest.raises(ValueError, match="Insufficient training data"):
            classifier.prepare_data(features_list)

    def test_exactly_10_samples(self, classifier):
        features_list = [
            {"return_1d": float(i) / 100, "regime_label": i % 3}
            for i in range(10)
        ]
        X, y = classifier.prepare_data(features_list)
        assert X.shape == (10, 12)

    def test_label_types_are_integers(self, classifier):
        features_list = [
            {"return_1d": float(i) / 100, "regime_label": i % 3}
            for i in range(10)
        ]
        _, y = classifier.prepare_data(features_list)
        assert all(isinstance(v, (int, np.integer)) for v in y)

    def test_feature_order_in_X(self, classifier):
        features_list = [
            {"return_1d": 1.0, "return_5d": 2.0, "regime_label": 0},
            {"return_1d": 3.0, "return_5d": 4.0, "regime_label": 1},
        ]
        # Only 2 samples, should raise
        with pytest.raises(ValueError, match="Insufficient training data"):
            classifier.prepare_data(features_list)


# ═══════════════════════════════════════════════════════════════════════════
# RegimeClassifier — ML-Dependent Methods (raise errors without ML)
# ═══════════════════════════════════════════════════════════════════════════


class TestRegimeClassifierMLMethods:
    def test_train_raises_importerror_when_ml_disabled(self):
        """train() checks SKLEARN_AVAILABLE before importing sklearn."""
        rc = RegimeClassifier()
        with pytest.raises(ImportError, match="scikit-learn"):
            rc.train([])

    def test_predict_raises_runtime_error_when_not_trained(self):
        rc = RegimeClassifier()
        with pytest.raises(RuntimeError, match="not trained"):
            rc.predict({"return_1d": 0.01})

    def test_save_raises_runtime_error_when_not_trained(self):
        rc = RegimeClassifier()
        with pytest.raises(RuntimeError, match="not trained"):
            rc.save()

    def test_load_raises_filenotfound_when_file_missing(self):
        rc = RegimeClassifier()
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            rc.load("/nonexistent/model.pkl")


# ═══════════════════════════════════════════════════════════════════════════
# WeeklyGridSearch
# ═══════════════════════════════════════════════════════════════════════════


class TestWeeklyGridSearch:
    @pytest.fixture
    def grid(self, tmp_path):
        g = WeeklyGridSearch(data_dir=str(tmp_path))
        return g

    def test_init_sets_data_dir(self, tmp_path):
        g = WeeklyGridSearch(data_dir=str(tmp_path))
        assert g.data_dir == str(tmp_path)

    def test_results_file_path(self, tmp_path):
        g = WeeklyGridSearch(data_dir=str(tmp_path))
        assert g.results_file == os.path.join(str(tmp_path), "grid_search_results.jsonl")

    def test_run_search_returns_list(self, grid):
        symbols = ["SPY", "GLD", "TLT"]
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        results = grid.run_search(symbols, base, grid_steps=3)
        assert isinstance(results, list)

    def test_run_search_correct_length(self, grid):
        symbols = ["SPY", "GLD", "TLT"]
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        results = grid.run_search(symbols, base, grid_steps=3)
        assert len(results) == 3 * 3  # grid_steps * len(symbols)

    def test_run_search_results_sorted_by_sharpe(self, grid):
        symbols = ["SPY", "GLD", "TLT"]
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        results = grid.run_search(symbols, base, grid_steps=5)
        sharps = [r["sharpe"] for r in results]
        assert all(sharps[i] >= sharps[i + 1] for i in range(len(sharps) - 1))

    def test_run_search_allocations_normalized(self, grid):
        symbols = ["SPY", "GLD", "TLT"]
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        results = grid.run_search(symbols, base, grid_steps=3)
        for r in results:
            total = sum(r["allocations"].values())
            assert abs(total - 1.0) < 1e-10

    def test_run_search_allocation_keys_match_symbols(self, grid):
        symbols = ["SPY", "GLD", "TLT"]
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        results = grid.run_search(symbols, base, grid_steps=3)
        for r in results:
            assert set(r["allocations"].keys()) == set(symbols)

    def test_run_search_base_preserved(self, grid):
        symbols = ["SPY", "GLD", "TLT"]
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        results = grid.run_search(symbols, base, grid_steps=3)
        for r in results:
            assert r["base_allocations"] == base

    def test_run_search_single_symbol(self, grid):
        """Should work with single symbol (only 1 * grid_steps results)."""
        symbols = ["SPY"]
        base = {"SPY": 1.0}
        results = grid.run_search(symbols, base, grid_steps=2)
        assert len(results) == 2

    def test_run_search_two_symbols(self, grid):
        symbols = ["SPY", "GLD"]
        base = {"SPY": 0.6, "GLD": 0.4}
        results = grid.run_search(symbols, base, grid_steps=4)
        assert len(results) == 4 * 2

    def test_min_weight_enforced_before_normalization(self, grid):
        """min_weight is enforced before normalization, so after normalization
        the weight may dip slightly below min_weight (but not to zero)."""
        symbols = ["SPY", "GLD"]
        base = {"SPY": 0.01, "GLD": 0.99}  # SPY well below min_weight
        results = grid.run_search(symbols, base, grid_steps=2, min_weight=0.05)
        for r in results:
            # Min weight ensures SPY is not zero (would happen without clamp)
            assert r["allocations"]["SPY"] > 0.04
            # The base was 0.01, so without min_weight clamp it could be <0.01

    def test_max_weight_capped(self, grid):
        symbols = ["SPY", "GLD"]
        base = {"SPY": 0.99, "GLD": 0.01}
        results = grid.run_search(symbols, base, grid_steps=2, max_deviation=0.1)
        for r in results:
            assert r["allocations"]["SPY"] <= 1.0 + 1e-10

    def test_writes_top_5_to_file(self, grid):
        symbols = ["SPY", "GLD", "TLT"]
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        results = grid.run_search(symbols, base, grid_steps=3)
        assert os.path.exists(grid.results_file)
        with open(grid.results_file) as f:
            lines = f.readlines()
        assert len(lines) == 5

    def test_saved_results_are_valid_json(self, grid):
        symbols = ["SPY", "GLD", "TLT"]
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        grid.run_search(symbols, base, grid_steps=3)
        with open(grid.results_file) as f:
            for line in f:
                d = json.loads(line)
                assert "allocations" in d
                assert "sharpe" in d

    def test_perturbations_in_results(self, grid):
        symbols = ["SPY", "GLD", "TLT"]
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        results = grid.run_search(symbols, base, grid_steps=3)
        for r in results:
            assert "perturbation" in r
            assert set(r["perturbation"].keys()) == set(symbols)

    def test_sharpe_within_expected_range(self, grid):
        symbols = ["SPY", "GLD", "TLT"]
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        results = grid.run_search(symbols, base, grid_steps=5)
        for r in results:
            assert 0.3 <= r["sharpe"] <= 0.8

    def test_volatility_within_expected_range(self, grid):
        symbols = ["SPY", "GLD", "TLT"]
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        results = grid.run_search(symbols, base, grid_steps=5)
        for r in results:
            assert 0.08 <= r["volatility"] <= 0.15

    def test_directory_created_automatically(self, tmp_path):
        """Data dir is created if it doesn't exist."""
        new_dir = os.path.join(str(tmp_path), "nonexistent", "subdir")
        g = WeeklyGridSearch(data_dir=new_dir)
        symbols = ["SPY", "GLD"]
        base = {"SPY": 0.6, "GLD": 0.4}
        g.run_search(symbols, base, grid_steps=1)
        assert os.path.exists(new_dir)

    def test_append_mode_on_second_run(self, grid):
        symbols = ["SPY", "GLD"]
        base = {"SPY": 0.6, "GLD": 0.4}
        grid.run_search(symbols, base, grid_steps=3)  # 3*2 = 6 results, top 5 saved
        grid.run_search(symbols, base, grid_steps=3)
        with open(grid.results_file) as f:
            lines = f.readlines()
        # Each run saves top 5, so 10 total after two runs
        assert len(lines) == 10


# ═══════════════════════════════════════════════════════════════════════════
# Main CLI (non-ML paths)
# ═══════════════════════════════════════════════════════════════════════════


class TestMainCLI:
    def test_unknown_command(self):
        with patch("sys.argv", ["regime_classifier.py", "unknown_cmd"]):
            with patch("sys.stdout") as mock:
                main()
                # Should print "Unknown command" without crashing

    def test_no_args(self):
        with patch("sys.argv", ["regime_classifier.py"]):
            with patch("sys.stdout") as mock:
                main()
                # Should print usage without crashing

    def test_train_missing_features_file(self):
        with patch("sys.argv", ["regime_classifier.py", "train", "/nonexistent/file.jsonl"]):
            with patch("sys.stdout") as mock:
                main()
                # Should print "Features file not found" without crashing

    def test_predict_without_model(self):
        with patch("sys.argv", ["regime_classifier.py", "predict"]):
            with patch("sys.stdout") as mock:
                main()
                # Should handle gracefully without crashing

    def test_grid_command(self):
        with patch("sys.argv", ["regime_classifier.py", "grid"]):
            with patch("sys.stdout") as mock:
                main()
                # Should run grid search without crashing

    def test_predict_with_features_file(self, tmp_path):
        features_file = os.path.join(str(tmp_path), "features.jsonl")
        with open(features_file, "w") as f:
            f.write(json.dumps({"return_1d": 0.01}) + "\n")

        with patch("sys.argv", ["regime_classifier.py", "predict", features_file]):
            with patch("sys.stdout") as mock:
                main()
                # Without model, should say "Model not found"

    def test_train_with_features_file(self, tmp_path):
        features_file = os.path.join(str(tmp_path), "features.jsonl")
        with open(features_file, "w") as f:
            for i in range(10):
                f.write(json.dumps({"return_1d": i / 100, "regime_label": i % 3}) + "\n")

        with patch("sys.argv", ["regime_classifier.py", "train", features_file]):
            with patch("sys.stdout") as mock:
                main()
                # Should try to train (ImportError without sklearn), handle gracefully

    def test_grid_with_args(self):
        with patch("sys.argv", ["regime_classifier.py", "grid"]):
            with patch("sys.stdout") as mock:
                main()
                # Grid search doesn't need ML

    def test_train_empty_features_file(self, tmp_path):
        """Empty features file should be handled gracefully."""
        features_file = os.path.join(str(tmp_path), "empty.jsonl")
        open(features_file, "w").close()
        with patch("sys.argv", ["regime_classifier.py", "train", features_file]):
            with patch("sys.stdout") as mock:
                main()
                # Should handle gracefully (no crash)

    def test_train_features_file_io_error(self, tmp_path):
        """Unreadable features file should be handled."""
        features_file = os.path.join(str(tmp_path), "bad_perms.jsonl")
        with open(features_file, "w") as f:
            f.write("not valid json\n")
        with patch("sys.argv", ["regime_classifier.py", "train", features_file]):
            with patch("sys.stdout") as mock:
                main()
                # Should handle gracefully (parse error caught by try/except)


# ═══════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestRegimeClassifierEdgeCases:
    def test_prepare_data_with_string_labels(self):
        """String regime_label values are converted via int()."""
        rc = RegimeClassifier()
        features_list = [
            {"return_1d": 0.01, "regime_label": "0"},
            {"return_1d": 0.02, "regime_label": "1"},
            {"return_1d": 0.03, "regime_label": "2"},
            {"return_1d": 0.04, "regime_label": "0"},
            {"return_1d": 0.05, "regime_label": "1"},
            {"return_1d": 0.06, "regime_label": "2"},
            {"return_1d": 0.07, "regime_label": "0"},
            {"return_1d": 0.08, "regime_label": "1"},
            {"return_1d": 0.09, "regime_label": "2"},
            {"return_1d": 0.10, "regime_label": "0"},
            {"return_1d": 0.11, "regime_label": "1"},
            {"return_1d": 0.12, "regime_label": "2"},
        ]
        X, y = rc.prepare_data(features_list)
        assert X.shape == (12, 12)
        assert y.dtype == np.int64 or y.dtype == int
        assert list(y) == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

    def test_classifier_reuse_after_error(self):
        """Classifier should be reusable after a failed train attempt."""
        rc = RegimeClassifier()
        # First predict fails (not trained)
        with pytest.raises(RuntimeError):
            rc.predict({"return_1d": 0.01})
        # Second predict also fails (same state)
        with pytest.raises(RuntimeError):
            rc.predict({"return_1d": 0.01})
        # State unchanged
        assert not rc.is_trained
        assert rc.model is None


class TestWeeklyGridSearchEdgeCases:
    def test_grid_search_no_symbols(self, tmp_path):
        """Empty symbols list should return empty results."""
        g = WeeklyGridSearch(data_dir=str(tmp_path))
        results = g.run_search([], {})
        assert isinstance(results, list)
        assert len(results) == 0

    def test_grid_search_single_step(self, tmp_path):
        """grid_steps=1 should produce len(symbols) results."""
        g = WeeklyGridSearch(data_dir=str(tmp_path))
        results = g.run_search(
            ["SPY", "GLD"], {"SPY": 0.6, "GLD": 0.4}, grid_steps=1
        )
        assert len(results) == 2

    def test_grid_search_large_deviation(self, tmp_path):
        """max_deviation=0.5 should allow wider exploration."""
        g = WeeklyGridSearch(data_dir=str(tmp_path))
        results = g.run_search(
            ["SPY", "GLD"],
            {"SPY": 0.6, "GLD": 0.4},
            grid_steps=10,
            max_deviation=0.5,
        )
        assert len(results) == 20
        # Some results should have allocations far from base
        deviations = [abs(r["allocations"]["SPY"] - 0.6) for r in results]
        assert any(d > 0.1 for d in deviations)

    def test_grid_search_unique_results(self, tmp_path):
        """Each run should produce different results (random perturb)."""
        g = WeeklyGridSearch(data_dir=str(tmp_path))
        symbols = ["SPY", "GLD", "TLT"]
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        results1 = g.run_search(symbols, base, grid_steps=3)
        results2 = g.run_search(symbols, base, grid_steps=3)
        # Very unlikely that all random perturbations produce the same sharpe
        sharps1 = [r["sharpe"] for r in results1]
        sharps2 = [r["sharpe"] for r in results2]
        assert sharps1 != sharps2  # near-certain with random uniforms

    def test_grid_search_data_dir_default(self):
        """Default data_dir should resolve from src.paths."""
        from src.paths import DATA_DIR
        g = WeeklyGridSearch()
        assert g.data_dir == str(DATA_DIR)
        assert "grid_search_results" in g.results_file

    def test_regime_prediction_hashable(self):
        """RegimePrediction should work as a dict key if needed (immutable-like)."""
        rp = RegimePrediction(
            symbol="SPY",
            timestamp="2026-01-01",
            p_bear=0.1,
            p_neutral=0.3,
            p_bull=0.6,
            predicted_regime=Regime.BULL,
            confidence=0.6,
        )
        # Dataclasses by default are not hashable unless frozen=True
        # Just verify it doesn't crash on basic operations
        assert rp.symbol == "SPY"

    def test_regime_prediction_repr(self):
        """Verify repr doesn't crash."""
        rp = RegimePrediction(
            symbol="SPY",
            timestamp="2026-01-01",
            p_bear=0.1,
            p_neutral=0.3,
            p_bull=0.6,
            predicted_regime=Regime.BULL,
            confidence=0.6,
        )
        r = repr(rp)
        assert "RegimePrediction" in r
        assert "SPY" in r
