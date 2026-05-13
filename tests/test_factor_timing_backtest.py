#!/usr/bin/env python3
"""
Tests for factor_timing_backtest.py — build_targets, get_feature_columns,
compute_metrics, and walk_forward_backtest with synthetic data.
"""
import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.backtest.factor_timing_backtest import (
    build_targets,
    get_feature_columns,
    compute_metrics,
    walk_forward_backtest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_factor_df(n=60):
    """Create synthetic factor feature DataFrame."""
    rng = np.random.RandomState(42)
    dates = pd.date_range('2020-01-01', periods=n, freq='ME')
    data = {
        'Mkt-RF_return': rng.normal(0.01, 0.04, n),
        'SMB_return': rng.normal(0.005, 0.03, n),
        'HML_return': rng.normal(0.003, 0.03, n),
        'RMW_return': rng.normal(0.004, 0.02, n),
        'CMA_return': rng.normal(0.002, 0.02, n),
        'UMD_return': rng.normal(0.006, 0.04, n),
        'HML_lag_1m': rng.normal(0.003, 0.03, n),
        'UMD_lag_1m': rng.normal(0.006, 0.04, n),
        'HML_ma_3m': rng.normal(0.003, 0.01, n),
        'UMD_ma_3m': rng.normal(0.006, 0.015, n),
        'vix_level': rng.uniform(15, 35, n),
        'yield_curve_slope': rng.uniform(-0.5, 2.0, n),
        'real_yield_10y': rng.uniform(-1, 3, n),
        'factor_dispersion': rng.uniform(0.01, 0.05, n),
        'real_rate_regime': rng.choice(['negative', 'low', 'elevated'], n),
        'macro_regime': rng.choice(['bull_normal', 'neutral', 'bear_stress'], n),
    }
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# build_targets Tests
# ---------------------------------------------------------------------------

class TestBuildTargets:

    def test_returns_dataframe(self):
        df = _make_factor_df()
        targets = build_targets(df)
        assert isinstance(targets, pd.DataFrame)

    def test_has_forward_columns(self):
        df = _make_factor_df()
        targets = build_targets(df)
        assert 'HML_return_fwd1m' in targets.columns
        assert 'UMD_return_fwd1m' in targets.columns
        assert 'RMW_return_fwd1m' in targets.columns

    def test_has_composite(self):
        df = _make_factor_df()
        targets = build_targets(df)
        assert 'composite_fwd1m' in targets.columns

    def test_forward_shift(self):
        df = _make_factor_df()
        targets = build_targets(df)
        # First value of fwd1m should equal second value of original
        assert targets['HML_return_fwd1m'].iloc[0] == pytest.approx(df['HML_return'].iloc[1])

    def test_last_value_nan(self):
        df = _make_factor_df()
        targets = build_targets(df)
        assert pd.isna(targets['HML_return_fwd1m'].iloc[-1])

    def test_composite_is_mean(self):
        df = _make_factor_df()
        targets = build_targets(df)
        fwd_cols = [c for c in targets.columns if c.endswith('_fwd1m') and c != 'composite_fwd1m']
        expected = targets[fwd_cols].iloc[0].mean()
        assert targets['composite_fwd1m'].iloc[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# get_feature_columns Tests
# ---------------------------------------------------------------------------

class TestGetFeatureColumns:

    def test_excludes_raw_returns(self):
        df = _make_factor_df()
        features = get_feature_columns(df)
        for col in features:
            assert not col.endswith('_return'), f"{col} should be excluded"

    def test_excludes_categorical(self):
        df = _make_factor_df()
        features = get_feature_columns(df)
        assert 'real_rate_regime' not in features
        assert 'macro_regime' not in features

    def test_includes_numeric_features(self):
        df = _make_factor_df()
        features = get_feature_columns(df)
        assert 'HML_lag_1m' in features
        assert 'vix_level' in features
        assert 'yield_curve_slope' in features

    def test_returns_list(self):
        df = _make_factor_df()
        features = get_feature_columns(df)
        assert isinstance(features, list)

    def test_no_string_columns(self):
        df = _make_factor_df()
        features = get_feature_columns(df)
        for col in features:
            assert df[col].dtype in ['float64', 'int64', 'float32', 'int32']


# ---------------------------------------------------------------------------
# compute_metrics Tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_returns_dict(self):
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        metrics = compute_metrics(returns, 'test')
        assert isinstance(metrics, dict)

    def test_has_all_keys(self):
        returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 3
        metrics = compute_metrics(returns, 'test')
        assert 'label' in metrics
        assert 'months' in metrics
        assert 'ann_return' in metrics
        assert 'ann_vol' in metrics
        assert 'sharpe' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'total_return' in metrics

    def test_label_preserved(self):
        returns = [0.01, -0.005, 0.02]
        metrics = compute_metrics(returns, 'my_label')
        assert metrics['label'] == 'my_label'

    def test_empty_returns(self):
        metrics = compute_metrics([], 'test')
        assert 'error' in metrics

    def test_single_return(self):
        metrics = compute_metrics([0.01], 'test')
        assert 'error' in metrics

    def test_positive_returns(self):
        returns = [0.01, 0.02, 0.015, 0.01, 0.02] * 3
        metrics = compute_metrics(returns, 'test')
        assert metrics['sharpe'] > 0
        assert metrics['win_rate'] == 100.0

    def test_negative_returns(self):
        returns = [-0.01, -0.02, -0.015, -0.01, -0.02] * 3
        metrics = compute_metrics(returns, 'test')
        assert metrics['sharpe'] < 0
        assert metrics['win_rate'] == 0.0

    def test_max_drawdown_negative(self):
        returns = [0.05, -0.10, 0.03, -0.05, 0.02] * 3
        metrics = compute_metrics(returns, 'test')
        assert metrics['max_drawdown'] <= 0

    def test_annualized(self):
        returns = [0.01] * 12
        metrics = compute_metrics(returns, 'test')
        # ann_return = mean(0.01) * 12 = 0.12
        assert metrics['ann_return'] == pytest.approx(12.0, abs=1.0)

    def test_nan_filtered(self):
        returns = [0.01, float('nan'), 0.02, float('nan'), 0.015] * 3
        metrics = compute_metrics(returns, 'test')
        assert 'error' not in metrics


# ---------------------------------------------------------------------------
# walk_forward_backtest Tests
# ---------------------------------------------------------------------------

class TestWalkForwardBacktest:

    def test_returns_dict(self):
        df = _make_factor_df(n=120)
        targets = build_targets(df)
        feature_cols = get_feature_columns(df)
        results = walk_forward_backtest(df, targets, feature_cols,
                                        train_years=2, start_year=2021, end_year=2022)
        assert isinstance(results, dict)

    def test_has_return_lists(self):
        df = _make_factor_df(n=120)
        targets = build_targets(df)
        feature_cols = get_feature_columns(df)
        results = walk_forward_backtest(df, targets, feature_cols,
                                        train_years=2, start_year=2021, end_year=2022)
        assert 'ml_returns' in results
        assert 'static_returns' in results

    def test_has_predictions(self):
        df = _make_factor_df(n=120)
        targets = build_targets(df)
        feature_cols = get_feature_columns(df)
        results = walk_forward_backtest(df, targets, feature_cols,
                                        train_years=2, start_year=2021, end_year=2022)
        assert 'ml_predictions' in results
        assert 'actual_returns' in results

    def test_has_dates(self):
        df = _make_factor_df(n=120)
        targets = build_targets(df)
        feature_cols = get_feature_columns(df)
        results = walk_forward_backtest(df, targets, feature_cols,
                                        train_years=2, start_year=2021, end_year=2022)
        assert 'dates' in results

    def test_returns_populated(self):
        df = _make_factor_df(n=120)
        targets = build_targets(df)
        feature_cols = get_feature_columns(df)
        results = walk_forward_backtest(df, targets, feature_cols,
                                        train_years=2, start_year=2021, end_year=2022)
        assert len(results['ml_returns']) > 0
        assert len(results['static_returns']) > 0

    def test_ml_and_static_same_length(self):
        df = _make_factor_df(n=120)
        targets = build_targets(df)
        feature_cols = get_feature_columns(df)
        results = walk_forward_backtest(df, targets, feature_cols,
                                        train_years=2, start_year=2021, end_year=2022)
        assert len(results['ml_returns']) == len(results['static_returns'])

    def test_position_scale_clamped(self):
        """ML return should be clipped version of actual * scale."""
        df = _make_factor_df(n=120)
        targets = build_targets(df)
        feature_cols = get_feature_columns(df)
        results = walk_forward_backtest(df, targets, feature_cols,
                                        train_years=2, start_year=2021, end_year=2022)
        for ml_ret, static_ret in zip(results['ml_returns'], results['static_returns']):
            # ML return = actual * clip(1 + pred*10, 0.5, 1.5)
            # So |ml_ret| <= 1.5 * |static_ret| (approximately)
            if not np.isnan(ml_ret) and not np.isnan(static_ret) and static_ret != 0:
                ratio = abs(ml_ret / static_ret)
                assert ratio <= 1.5 + 0.01  # Small tolerance for floating point
