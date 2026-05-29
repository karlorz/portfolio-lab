"""
Tests for compute_metrics_from_returns — lightweight returns→metrics utility.
"""

import pytest
import numpy as np
from src.backtest.metrics import compute_metrics_from_returns


@pytest.fixture
def normal_returns():
    """252 days of ~10% CAGR returns."""
    np.random.seed(42)
    return list(np.random.normal(0.0004, 0.01, 252))


@pytest.fixture
def bear_returns():
    """252 days of negative returns."""
    np.random.seed(42)
    return list(np.random.normal(-0.001, 0.015, 252))


class TestComputeMetricsFromReturns:

    def test_returns_dict_with_expected_keys(self, normal_returns):
        result = compute_metrics_from_returns(normal_returns)
        expected_keys = {'total_return', 'cagr', 'volatility', 'sharpe', 'max_drawdown', 'calmar'}
        assert set(result.keys()) == expected_keys

    def test_positive_cagr_for_positive_returns(self, normal_returns):
        result = compute_metrics_from_returns(normal_returns)
        assert result['cagr'] > 0

    def test_negative_cagr_for_bear(self, bear_returns):
        result = compute_metrics_from_returns(bear_returns)
        assert result['cagr'] < 0

    def test_max_drawdown_is_negative(self, normal_returns):
        result = compute_metrics_from_returns(normal_returns)
        assert result['max_drawdown'] <= 0

    def test_volatility_is_positive(self, normal_returns):
        result = compute_metrics_from_returns(normal_returns)
        assert result['volatility'] > 0

    def test_sharpe_scales_with_risk_free_rate(self, normal_returns):
        result_low_rf = compute_metrics_from_returns(normal_returns, risk_free_rate=0.0)
        result_high_rf = compute_metrics_from_returns(normal_returns, risk_free_rate=0.10)
        assert result_low_rf['sharpe'] > result_high_rf['sharpe']

    def test_empty_returns(self):
        result = compute_metrics_from_returns([])
        assert result['total_return'] == 0.0
        assert result['cagr'] == 0.0
        assert result['sharpe'] == 0.0

    def test_single_return(self):
        result = compute_metrics_from_returns([0.01])
        assert result['total_return'] == pytest.approx(0.01, abs=1e-6)
        assert result['volatility'] == 0.0  # Can't compute std from 1 sample

    def test_constant_returns(self):
        result = compute_metrics_from_returns([0.001] * 252)
        assert result['volatility'] == 0.0  # No variance
        assert result['max_drawdown'] == 0.0  # Monotonically increasing

    def test_match_ubt_validation_computation(self, normal_returns):
        """Results should match the old inline computation."""
        from math import sqrt, pow as mpow
        returns = normal_returns
        total_return = np.prod([1 + r for r in returns]) - 1
        years = len(returns) / 252
        cagr = mpow(1 + total_return, 1 / years) - 1
        daily_vol = np.std(returns, ddof=1)
        ann_vol = daily_vol * sqrt(252)

        result = compute_metrics_from_returns(returns)
        assert result['total_return'] == pytest.approx(total_return, abs=1e-4)
        assert result['cagr'] == pytest.approx(cagr, abs=1e-4)
        assert result['volatility'] == pytest.approx(ann_vol, abs=1e-4)
