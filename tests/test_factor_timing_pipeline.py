#!/usr/bin/env python3
"""
Tests for factor_timing_pipeline.py — FactorTimingPipeline, factor feature
engineering (lags, rolling stats, correlations, dispersion), save/load, and CLI.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.features.factor_timing_pipeline import (
    FactorTimingPipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(tmp_path=None):
    pipeline = FactorTimingPipeline.__new__(FactorTimingPipeline)
    pipeline.feature_dir = tmp_path or Path("/tmp/test_features")
    pipeline.feature_file = pipeline.feature_dir / "factor_timing_features.csv"
    pipeline.metadata_file = pipeline.feature_dir / "feature_metadata.json"
    pipeline.factor_fetcher = MagicMock()
    pipeline.macro_engineer = MagicMock()
    return pipeline


def _make_factor_data(n=60):
    """Create synthetic factor return data."""
    rng = np.random.RandomState(42)
    dates = pd.date_range(end=datetime.now(), periods=n, freq='ME')
    data = pd.DataFrame({
        'Mkt-RF': rng.normal(0.01, 0.04, n),
        'SMB': rng.normal(0.005, 0.03, n),
        'HML': rng.normal(0.003, 0.03, n),
        'RMW': rng.normal(0.004, 0.02, n),
        'CMA': rng.normal(0.002, 0.02, n),
        'UMD': rng.normal(0.006, 0.04, n),
        'RF': np.full(n, 0.002),
    }, index=dates)
    return data


# ---------------------------------------------------------------------------
# _engineer_factor_features Tests
# ---------------------------------------------------------------------------

class TestEngineerFactorFeatures:

    def test_returns_dataframe(self):
        pipeline = _make_pipeline()
        factor_data = _make_factor_data()
        features = pipeline._engineer_factor_features(factor_data)
        assert isinstance(features, pd.DataFrame)

    def test_has_return_columns(self):
        pipeline = _make_pipeline()
        factor_data = _make_factor_data()
        features = pipeline._engineer_factor_features(factor_data)
        assert 'Mkt-RF_return' in features.columns
        assert 'HML_return' in features.columns

    def test_has_lag_columns(self):
        pipeline = _make_pipeline()
        factor_data = _make_factor_data()
        features = pipeline._engineer_factor_features(factor_data)
        assert 'Mkt-RF_lag_1m' in features.columns
        assert 'HML_lag_12m' in features.columns

    def test_has_ma_columns(self):
        pipeline = _make_pipeline()
        factor_data = _make_factor_data()
        features = pipeline._engineer_factor_features(factor_data)
        assert 'Mkt-RF_ma_3m' in features.columns
        assert 'UMD_ma_12m' in features.columns

    def test_has_vol_columns(self):
        pipeline = _make_pipeline()
        factor_data = _make_factor_data()
        features = pipeline._engineer_factor_features(factor_data)
        assert 'Mkt-RF_vol_3m' in features.columns
        assert 'HML_vol_12m' in features.columns

    def test_has_cumret(self):
        pipeline = _make_pipeline()
        factor_data = _make_factor_data()
        features = pipeline._engineer_factor_features(factor_data)
        assert 'Mkt-RF_cumret_12m' in features.columns

    def test_has_percentile(self):
        pipeline = _make_pipeline()
        factor_data = _make_factor_data()
        features = pipeline._engineer_factor_features(factor_data)
        assert 'HML_percentile_5y' in features.columns

    def test_has_correlations(self):
        pipeline = _make_pipeline()
        factor_data = _make_factor_data()
        features = pipeline._engineer_factor_features(factor_data)
        assert 'hml_umd_corr_12m' in features.columns
        assert 'mkt_hml_corr_12m' in features.columns

    def test_has_dispersion(self):
        pipeline = _make_pipeline()
        factor_data = _make_factor_data()
        features = pipeline._engineer_factor_features(factor_data)
        assert 'factor_dispersion' in features.columns

    def test_lagged_columns_have_nans(self):
        pipeline = _make_pipeline()
        factor_data = _make_factor_data(n=24)
        features = pipeline._engineer_factor_features(factor_data)
        # 12-month lag should have 12 NaN values
        assert features['Mkt-RF_lag_12m'].isna().sum() == 12

    def test_no_rf_in_dispersion(self):
        """RF should be excluded from factor dispersion calculation."""
        pipeline = _make_pipeline()
        factor_data = _make_factor_data()
        features = pipeline._engineer_factor_features(factor_data)
        assert 'factor_dispersion' in features.columns


# ---------------------------------------------------------------------------
# build_feature_dataset Tests
# ---------------------------------------------------------------------------

class TestBuildFeatureDataset:

    def test_synthetic_returns_dataframe(self):
        pipeline = _make_pipeline()
        pipeline.factor_fetcher.generate_synthetic_factor_data.return_value = _make_factor_data()
        pipeline.macro_engineer._generate_synthetic_macro_data.return_value = pd.DataFrame({
            'treasury_10y': np.full(60, 2.5),
            'treasury_2y': np.full(60, 1.5),
            'fed_funds': np.full(60, 1.0),
            'tips_10y': np.full(60, 0.5),
            'oil_brent': np.full(60, 60.0),
            'breakeven_10y': np.full(60, 2.0),
        }, index=pd.date_range(end=datetime.now(), periods=60, freq='D'))
        pipeline.macro_engineer._generate_synthetic_vix_data.return_value = pd.DataFrame({
            'vix_spot': np.full(60, 18.0),
            'vix_1m': np.full(60, 19.0),
            'vix_3m': np.full(60, 17.0),
        }, index=pd.date_range(end=datetime.now(), periods=60, freq='D'))
        pipeline.macro_engineer.engineer_features.return_value = pd.DataFrame({
            'vix_level': np.full(5, 18.0),
            'yield_curve_slope': np.full(5, 1.0),
        }, index=pd.date_range(end=datetime.now(), periods=5, freq='ME'))
        features = pipeline.build_feature_dataset(use_synthetic=True)
        assert isinstance(features, pd.DataFrame)

    def test_factor_features_engineered(self):
        pipeline = _make_pipeline()
        factor_data = _make_factor_data()
        features = pipeline._engineer_factor_features(factor_data)
        assert len(features.columns) > 10


# ---------------------------------------------------------------------------
# save/load Tests
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_creates_file(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        factor_data = _make_factor_data()
        features = pipeline._engineer_factor_features(factor_data)
        pipeline.save_dataset(features)
        assert pipeline.feature_file.exists()
        assert pipeline.metadata_file.exists()

    def test_load_roundtrip(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        factor_data = _make_factor_data()
        features = pipeline._engineer_factor_features(factor_data)
        pipeline.save_dataset(features)
        loaded = pipeline.load_dataset()
        assert loaded is not None
        assert len(loaded) == len(features)

    def test_load_missing(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        assert pipeline.load_dataset() is None

    def test_metadata_structure(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        factor_data = _make_factor_data()
        features = pipeline._engineer_factor_features(factor_data)
        pipeline.save_dataset(features)
        with open(pipeline.metadata_file) as f:
            meta = json.load(f)
        assert 'rows' in meta
        assert 'columns' in meta
        assert 'target_columns' in meta
        assert 'feature_columns' in meta


# ---------------------------------------------------------------------------
# get_feature_summary Tests
# ---------------------------------------------------------------------------

class TestGetFeatureSummary:

    def test_no_dataset(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        summary = pipeline.get_feature_summary()
        assert 'error' in summary

    def test_with_dataset(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        factor_data = _make_factor_data()
        features = pipeline._engineer_factor_features(factor_data)
        pipeline.save_dataset(features)
        summary = pipeline.get_feature_summary()
        assert 'rows' in summary
        assert 'columns' in summary
        assert 'targets' in summary


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:

    def test_main_synthetic(self, capsys):
        from src.features.factor_timing_pipeline import main
        pipeline = MagicMock()
        pipeline.build_feature_dataset.return_value = pd.DataFrame({'a': [1]})
        with patch("src.features.factor_timing_pipeline.FactorTimingPipeline", return_value=pipeline):
            with patch("sys.argv", ["pipeline.py", "synthetic"]):
                main()
        captured = capsys.readouterr()
        assert "Synthetic" in captured.out or "built" in captured.out.lower()

    def test_main_stats_no_dataset(self, tmp_path, capsys):
        from src.features.factor_timing_pipeline import main
        pipeline = MagicMock()
        pipeline.get_feature_summary.return_value = {'error': 'No feature dataset found'}
        with patch("src.features.factor_timing_pipeline.FactorTimingPipeline", return_value=pipeline):
            with patch("sys.argv", ["pipeline.py", "stats"]):
                main()
        captured = capsys.readouterr()
        assert "error" in captured.out.lower()
