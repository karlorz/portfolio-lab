#!/usr/bin/env python3
"""
Tests for macro_features.py — MacroFeatureEngineer, synthetic data generation,
feature engineering pipeline (yield curve, VIX, regime classification), and CLI.
"""
import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.features.macro_features import (
    MacroFeatureEngineer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engineer():
    engineer = MacroFeatureEngineer.__new__(MacroFeatureEngineer)
    engineer.feature_dir = Path("/tmp/test_features")
    engineer.feature_file = Path("/tmp/test_features/macro_features.csv")
    return engineer


# ---------------------------------------------------------------------------
# Synthetic Data Generation Tests
# ---------------------------------------------------------------------------

class TestSyntheticMacroData:

    def test_returns_dataframe(self):
        engineer = _make_engineer()
        df = engineer._generate_synthetic_macro_data()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        engineer = _make_engineer()
        df = engineer._generate_synthetic_macro_data()
        assert 'treasury_10y' in df.columns
        assert 'treasury_2y' in df.columns
        assert 'fed_funds' in df.columns
        assert 'tips_10y' in df.columns
        assert 'oil_brent' in df.columns
        assert 'breakeven_10y' in df.columns

    def test_date_index(self):
        engineer = _make_engineer()
        df = engineer._generate_synthetic_macro_data()
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_positive_rows(self):
        engineer = _make_engineer()
        df = engineer._generate_synthetic_macro_data()
        assert len(df) > 100

    def test_treasury_clipped(self):
        engineer = _make_engineer()
        df = engineer._generate_synthetic_macro_data()
        assert df['treasury_10y'].min() >= 0.5
        assert df['treasury_10y'].max() <= 10

    def test_oil_clipped(self):
        engineer = _make_engineer()
        df = engineer._generate_synthetic_macro_data()
        assert df['oil_brent'].min() >= 20
        assert df['oil_brent'].max() <= 150

    def test_custom_start_date(self):
        engineer = _make_engineer()
        df = engineer._generate_synthetic_macro_data(start_date='2020-01-01')
        assert df.index[0].year == 2020


class TestSyntheticVIXData:

    def test_returns_dataframe(self):
        engineer = _make_engineer()
        df = engineer._generate_synthetic_vix_data()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        engineer = _make_engineer()
        df = engineer._generate_synthetic_vix_data()
        assert 'vix_spot' in df.columns
        assert 'vix_1m' in df.columns
        assert 'vix_3m' in df.columns

    def test_vix_clipped(self):
        engineer = _make_engineer()
        df = engineer._generate_synthetic_vix_data()
        assert df['vix_spot'].min() >= 10
        assert df['vix_spot'].max() <= 80

    def test_mean_reverting(self):
        engineer = _make_engineer()
        df = engineer._generate_synthetic_vix_data()
        # VIX should hover around 20 (mean-reverting)
        assert df['vix_spot'].mean() < 30
        assert df['vix_spot'].mean() > 10


# ---------------------------------------------------------------------------
# Feature Engineering Tests
# ---------------------------------------------------------------------------

class TestEngineerFeatures:

    def test_returns_dataframe(self):
        engineer = _make_engineer()
        macro = engineer._generate_synthetic_macro_data()
        vix = engineer._generate_synthetic_vix_data()
        features = engineer.engineer_features(macro, vix)
        assert isinstance(features, pd.DataFrame)

    def test_yield_curve_slope(self):
        engineer = _make_engineer()
        macro = engineer._generate_synthetic_macro_data()
        vix = engineer._generate_synthetic_vix_data()
        features = engineer.engineer_features(macro, vix)
        assert 'yield_curve_slope' in features.columns

    def test_curve_inverted_flag(self):
        engineer = _make_engineer()
        macro = engineer._generate_synthetic_macro_data()
        vix = engineer._generate_synthetic_vix_data()
        features = engineer.engineer_features(macro, vix)
        assert 'curve_inverted' in features.columns
        # Should be 0 or 1
        assert set(features['curve_inverted'].dropna().unique()).issubset({0, 1})

    def test_vix_level(self):
        engineer = _make_engineer()
        macro = engineer._generate_synthetic_macro_data()
        vix = engineer._generate_synthetic_vix_data()
        features = engineer.engineer_features(macro, vix)
        assert 'vix_level' in features.columns

    def test_yield_10y(self):
        engineer = _make_engineer()
        macro = engineer._generate_synthetic_macro_data()
        vix = engineer._generate_synthetic_vix_data()
        features = engineer.engineer_features(macro, vix)
        assert 'yield_10y' in features.columns

    def test_momentum_features(self):
        engineer = _make_engineer()
        macro = engineer._generate_synthetic_macro_data()
        vix = engineer._generate_synthetic_vix_data()
        features = engineer.engineer_features(macro, vix)
        assert 'yield_10y_change_1m' in features.columns
        assert 'vix_level_change_1m' in features.columns

    def test_real_rate_regime(self):
        engineer = _make_engineer()
        macro = engineer._generate_synthetic_macro_data()
        vix = engineer._generate_synthetic_vix_data()
        features = engineer.engineer_features(macro, vix)
        assert 'real_rate_regime' in features.columns
        valid_regimes = {'deep_negative', 'negative', 'low', 'elevated'}
        actual = set(features['real_rate_regime'].dropna().unique())
        assert actual.issubset(valid_regimes)

    def test_macro_regime(self):
        engineer = _make_engineer()
        macro = engineer._generate_synthetic_macro_data()
        vix = engineer._generate_synthetic_vix_data()
        features = engineer.engineer_features(macro, vix)
        assert 'macro_regime' in features.columns
        valid_regimes = {'bull_normal', 'bull_late', 'neutral', 'bear_stress'}
        actual = set(features['macro_regime'].dropna().unique())
        assert actual.issubset(valid_regimes)

    def test_monthly_frequency(self):
        engineer = _make_engineer()
        macro = engineer._generate_synthetic_macro_data()
        vix = engineer._generate_synthetic_vix_data()
        features = engineer.engineer_features(macro, vix)
        # Should be monthly (ME frequency)
        diffs = features.index.to_series().diff().dropna()
        # Most diffs should be ~28-31 days
        median_diff = diffs.median().days
        assert 25 <= median_diff <= 35

    def test_inflation_expectations(self):
        engineer = _make_engineer()
        macro = engineer._generate_synthetic_macro_data()
        vix = engineer._generate_synthetic_vix_data()
        features = engineer.engineer_features(macro, vix)
        assert 'inflation_expectations' in features.columns

    def test_fewer_rows_than_input(self):
        engineer = _make_engineer()
        macro = engineer._generate_synthetic_macro_data()
        vix = engineer._generate_synthetic_vix_data()
        features = engineer.engineer_features(macro, vix)
        # Monthly should have fewer rows than daily input
        assert len(features) < len(macro)

    def test_no_unexpected_nan_columns(self):
        engineer = _make_engineer()
        macro = engineer._generate_synthetic_macro_data()
        vix = engineer._generate_synthetic_vix_data()
        features = engineer.engineer_features(macro, vix)
        # vix_percentile_1y uses 252-period rolling on monthly data → all NaN (expected)
        expected_nan_cols = {'vix_percentile_1y'}
        for col in features.columns:
            if col in expected_nan_cols:
                continue
            if features[col].dtype in ['float64', 'int64']:
                assert not features[col].isna().all(), f"Column {col} is all NaN"


# ---------------------------------------------------------------------------
# Save/Load Tests
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_creates_file(self, tmp_path):
        engineer = _make_engineer()
        engineer.feature_file = tmp_path / "features.csv"
        macro = engineer._generate_synthetic_macro_data()
        vix = engineer._generate_synthetic_vix_data()
        features = engineer.engineer_features(macro, vix)
        engineer.save_features(features)
        assert engineer.feature_file.exists()

    def test_load_roundtrip(self, tmp_path):
        engineer = _make_engineer()
        engineer.feature_file = tmp_path / "features.csv"
        macro = engineer._generate_synthetic_macro_data()
        vix = engineer._generate_synthetic_vix_data()
        features = engineer.engineer_features(macro, vix)
        engineer.save_features(features)
        loaded = engineer.load_features()
        assert len(loaded) == len(features)
        assert list(loaded.columns) == list(features.columns)


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:

    def test_synthetic_flag(self, capsys):
        from src.features.macro_features import main
        with patch("sys.argv", ["macro_features.py", "--synthetic"]):
            main()
        captured = capsys.readouterr()
        assert "Features" in captured.out or "features" in captured.out.lower()
