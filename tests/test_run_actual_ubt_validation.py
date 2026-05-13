#!/usr/bin/env python3
"""
Tests for run_actual_ubt_validation.py — extract_prices, calculate_returns,
find_overlap, align_series, calculate_metrics, calculate_correlation.
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock

from src.backtest.run_actual_ubt_validation import (
    extract_prices,
    calculate_returns,
    find_overlap,
    align_series,
    calculate_metrics,
    calculate_correlation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_data(symbol="TLT", n=100, start=95.0, drift=0.0003, seed=42):
    rng = np.random.RandomState(seed)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + rng.normal(drift, 0.012)))
    dates = [f"2024-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}" for i in range(n)]
    entries = [{"date": d, "adjClose": p} for d, p in zip(dates, prices)]
    return {symbol: entries}, dates, prices


# ---------------------------------------------------------------------------
# extract_prices Tests
# ---------------------------------------------------------------------------

class TestExtractPrices:

    def test_returns_dates_and_prices(self):
        data, expected_dates, expected_prices = _make_price_data()
        dates, prices = extract_prices(data, "TLT")
        assert len(dates) == len(expected_dates)
        assert len(prices) == len(expected_prices)

    def test_uses_adjclose(self):
        data = {"TLT": [{"date": "2024-01-02", "adjClose": 95.5}]}
        dates, prices = extract_prices(data, "TLT")
        assert prices == [95.5]

    def test_falls_back_to_close(self):
        data = {"TLT": [{"date": "2024-01-02", "close": 96.0}]}
        dates, prices = extract_prices(data, "TLT")
        assert prices == [96.0]

    def test_missing_symbol(self):
        dates, prices = extract_prices({}, "TLT")
        assert dates == []
        assert prices == []

    def test_skips_entries_without_date(self):
        data = {"TLT": [{"adjClose": 95.0}, {"date": "2024-01-02", "adjClose": 96.0}]}
        dates, prices = extract_prices(data, "TLT")
        assert len(dates) == 1

    def test_skips_entries_without_price(self):
        data = {"TLT": [{"date": "2024-01-02"}, {"date": "2024-01-03", "adjClose": 96.0}]}
        dates, prices = extract_prices(data, "TLT")
        assert len(prices) == 1


# ---------------------------------------------------------------------------
# calculate_returns Tests
# ---------------------------------------------------------------------------

class TestCalculateReturns:

    def test_returns_length(self):
        prices = [100, 102, 101, 105]
        returns = calculate_returns(prices)
        assert len(returns) == 3

    def test_first_return(self):
        prices = [100, 102]
        returns = calculate_returns(prices)
        assert returns[0] == pytest.approx(0.02)

    def test_negative_return(self):
        prices = [100, 95]
        returns = calculate_returns(prices)
        assert returns[0] == pytest.approx(-0.05)

    def test_empty_prices(self):
        assert calculate_returns([]) == []

    def test_single_price(self):
        assert calculate_returns([100]) == []


# ---------------------------------------------------------------------------
# find_overlap Tests
# ---------------------------------------------------------------------------

class TestFindOverlap:

    def test_returns_tuple(self):
        dates1 = ["2024-01-01", "2024-01-02", "2024-01-03"]
        dates2 = ["2024-01-02", "2024-01-03", "2024-01-04"]
        result = find_overlap(dates1, dates2)
        assert result is not None
        assert len(result) == 3

    def test_overlap_range(self):
        dates1 = ["2024-01-01", "2024-01-02", "2024-01-03"]
        dates2 = ["2024-01-02", "2024-01-03", "2024-01-04"]
        start, end, count = find_overlap(dates1, dates2)
        assert start == "2024-01-02"
        assert end == "2024-01-03"
        assert count == 2

    def test_no_overlap(self):
        dates1 = ["2024-01-01", "2024-01-02"]
        dates2 = ["2024-02-01", "2024-02-02"]
        assert find_overlap(dates1, dates2) is None

    def test_full_overlap(self):
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        result = find_overlap(dates, dates)
        assert result[2] == 3

    def test_single_overlap(self):
        dates1 = ["2024-01-01", "2024-01-02"]
        dates2 = ["2024-01-02", "2024-01-03"]
        start, end, count = find_overlap(dates1, dates2)
        assert count == 1
        assert start == end


# ---------------------------------------------------------------------------
# align_series Tests
# ---------------------------------------------------------------------------

class TestAlignSeries:

    def test_returns_aligned(self):
        dates1 = ["2024-01-01", "2024-01-02", "2024-01-03"]
        prices1 = [100, 101, 102]
        dates2 = ["2024-01-02", "2024-01-03", "2024-01-04"]
        prices2 = [200, 201, 202]
        dates, p1, p2 = align_series(dates1, prices1, dates2, prices2)
        assert len(dates) == 2
        assert p1 == [101, 102]
        assert p2 == [200, 201]

    def test_no_overlap(self):
        dates1 = ["2024-01-01"]
        prices1 = [100]
        dates2 = ["2024-02-01"]
        prices2 = [200]
        dates, p1, p2 = align_series(dates1, prices1, dates2, prices2)
        assert len(dates) == 0

    def test_full_overlap(self):
        dates = ["2024-01-01", "2024-01-02"]
        p1 = [100, 101]
        p2 = [200, 201]
        d, a1, a2 = align_series(dates, p1, dates, p2)
        assert len(d) == 2


# ---------------------------------------------------------------------------
# calculate_metrics Tests
# ---------------------------------------------------------------------------

class TestCalculateMetrics:

    def test_returns_dict(self):
        returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 50
        metrics = calculate_metrics(returns, ["2024-01-01"] * 250, "Test")
        assert isinstance(metrics, dict)

    def test_has_all_keys(self):
        returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 50
        metrics = calculate_metrics(returns, ["2024-01-01"] * 250, "Test")
        assert 'scenario' in metrics
        assert 'cagr' in metrics
        assert 'volatility' in metrics
        assert 'sharpe' in metrics
        assert 'maxDrawdown' in metrics
        assert 'calmar' in metrics
        assert 'totalReturn' in metrics

    def test_scenario_preserved(self):
        returns = [0.01] * 10
        metrics = calculate_metrics(returns, ["2024-01-01"] * 10, "MyScenario")
        assert metrics['scenario'] == "MyScenario"

    def test_positive_returns(self):
        # Use varied returns so std > 0
        rng = np.random.RandomState(42)
        returns = (rng.normal(0.01, 0.005, 252)).tolist()
        metrics = calculate_metrics(returns, ["2024-01-01"] * 252, "Test")
        assert metrics['cagr'] > 0
        assert metrics['sharpe'] > 0

    def test_max_drawdown_negative(self):
        returns = [0.05, -0.10, 0.03, -0.05, 0.02] * 20
        metrics = calculate_metrics(returns, ["2024-01-01"] * 100, "Test")
        assert metrics['maxDrawdown'] <= 0

    def test_expense_ratio_ubt(self):
        returns = [0.01] * 252
        metrics = calculate_metrics(returns, ["2024-01-01"] * 252, "Actual_UBT")
        assert metrics['annualizedExpenseImpact'] < 0

    def test_expense_ratio_tmf(self):
        returns = [0.01] * 252
        metrics = calculate_metrics(returns, ["2024-01-01"] * 252, "Actual_TMF")
        assert metrics['annualizedExpenseImpact'] < 0

    def test_tracking_error_with_base(self):
        returns = [0.01, -0.005, 0.02] * 100
        base = [0.005, -0.002, 0.01] * 100
        metrics = calculate_metrics(returns, ["2024-01-01"] * 300, "Test",
                                    base_returns=base, expected_multiple=2)
        assert metrics['trackingErrorVsTLT'] > 0

    def test_volatility_decay(self):
        # Use varied returns so volatility decay is non-zero
        rng = np.random.RandomState(42)
        returns = (rng.normal(0.01, 0.015, 252)).tolist()
        metrics = calculate_metrics(returns, ["2024-01-01"] * 252, "Test",
                                    expected_multiple=2)
        assert metrics['volatilityDecayEstimate'] < 0


# ---------------------------------------------------------------------------
# calculate_correlation Tests
# ---------------------------------------------------------------------------

class TestCalculateCorrelation:

    def test_perfect_correlation(self):
        r1 = [0.01, 0.02, -0.01, 0.03]
        corr = calculate_correlation(r1, r1)
        assert corr == pytest.approx(1.0)

    def test_negative_correlation(self):
        r1 = [0.01, 0.02, -0.01, 0.03]
        r2 = [-0.01, -0.02, 0.01, -0.03]
        corr = calculate_correlation(r1, r2)
        assert corr == pytest.approx(-1.0)

    def test_different_lengths(self):
        r1 = [0.01, 0.02, -0.01]
        r2 = [0.01, 0.02, -0.01, 0.03]
        corr = calculate_correlation(r1, r2)
        assert -1 <= corr <= 1

    def test_zero_correlation(self):
        rng = np.random.RandomState(42)
        r1 = rng.normal(0, 0.01, 1000).tolist()
        r2 = rng.normal(0, 0.01, 1000).tolist()
        corr = calculate_correlation(r1, r2)
        assert abs(corr) < 0.2
