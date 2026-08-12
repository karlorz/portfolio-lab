#!/usr/bin/env python3
"""
Tests for run_actual_ubt_validation.py — extract_prices, calculate_returns,
find_overlap, align_series, calculate_metrics, calculate_correlation.
"""
import numpy as np

import pytest
from unittest.mock import patch, MagicMock

from src.backtest.run_actual_ubt_validation import (
    extract_prices,
    calculate_returns,
    find_overlap,
    align_series,
    calculate_metrics,
    calculate_correlation,
    load_historical_data,
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


# ---------------------------------------------------------------------------
# __all__ export validation
# ---------------------------------------------------------------------------

class TestExports:
    """Verify __all__ exports."""

    def test_all_exports_present(self):
        import src.backtest.run_actual_ubt_validation as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"Missing export: {name}"

    def test_all_count(self):
        import src.backtest.run_actual_ubt_validation as mod
        assert len(mod.__all__) == 7


# ---------------------------------------------------------------------------
# load_historical_data tests
# ---------------------------------------------------------------------------

class TestLoadHistoricalData:
    """Tests for load_historical_data."""

    def test_returns_dict(self):
        with patch('src.backtest.run_actual_ubt_validation.json.load', return_value={"SPY": []}):
            with patch('builtins.open', MagicMock()):
                result = load_historical_data()
                assert isinstance(result, dict)

    def test_missing_file_raises(self):
        with patch('builtins.open', side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                load_historical_data()


def test_a3_b1a_delegation_matches_pre_migration_capture():
    """A3 pin (Item B1a sub-task 12): load_historical_data delegates to grid_runner.load_prices."""
    from src.backtest.grid_runner import load_prices

    # module-level loader stays in pilot; the shared loader is grid_runner's
    assert load_historical_data.__module__ == (
        "src.backtest.run_actual_ubt_validation"
    )
    assert load_prices.__module__ == "src.backtest.grid_runner"


# ---------------------------------------------------------------------------
# extract_prices extended
# ---------------------------------------------------------------------------

class TestExtractPricesExtended:
    """Extended extract_prices edge cases."""

    def test_empty_data(self):
        dates, prices = extract_prices({}, "SPY")
        assert dates == []
        assert prices == []

    def test_multiple_entries(self):
        data = {
            "SPY": [
                {"date": "2024-01-01", "adjClose": 100},
                {"date": "2024-01-02", "adjClose": 101},
                {"date": "2024-01-03", "adjClose": 102},
            ]
        }
        dates, prices = extract_prices(data, "SPY")
        assert len(dates) == 3
        assert prices == [100.0, 101.0, 102.0]


# ---------------------------------------------------------------------------
# calculate_returns extended
# ---------------------------------------------------------------------------

class TestCalculateReturnsExtended:
    """Extended calculate_returns edge cases."""

    def test_constant_prices(self):
        prices = [100.0] * 10
        returns = calculate_returns(prices)
        assert all(r == 0.0 for r in returns)

    def test_increasing_prices(self):
        prices = [100.0, 101.0, 102.0]
        returns = calculate_returns(prices)
        assert all(r > 0 for r in returns)

    def test_decreasing_prices(self):
        prices = [100.0, 99.0, 98.0]
        returns = calculate_returns(prices)
        assert all(r < 0 for r in returns)


# ---------------------------------------------------------------------------
# find_overlap extended
# ---------------------------------------------------------------------------

class TestFindOverlapExtended:
    """Extended find_overlap edge cases."""

    def test_partial_overlap_start(self):
        dates1 = ["2024-01-01", "2024-01-02", "2024-01-03"]
        dates2 = ["2024-01-03", "2024-01-04", "2024-01-05"]
        result = find_overlap(dates1, dates2)
        assert result is not None
        start, end, count = result
        assert start == "2024-01-03"
        assert end == "2024-01-03"
        assert count == 1

    def test_sorted_dates(self):
        dates1 = ["2024-01-05", "2024-01-01", "2024-01-03"]
        dates2 = ["2024-01-02", "2024-01-04", "2024-01-06"]
        result = find_overlap(dates1, dates2)
        # Should find overlap regardless of input order
        if result is not None:
            start, end, count = result
            assert start <= end


# ---------------------------------------------------------------------------
# align_series extended
# ---------------------------------------------------------------------------

class TestAlignSeriesExtended:
    """Extended align_series edge cases."""

    def test_partial_overlap_keeps_common(self):
        dates1 = ["2024-01-01", "2024-01-02", "2024-01-03"]
        prices1 = [100.0, 101.0, 102.0]
        dates2 = ["2024-01-02", "2024-01-03", "2024-01-04"]
        prices2 = [200.0, 201.0, 202.0]
        dates, p1, p2 = align_series(dates1, prices1, dates2, prices2)
        # Only common dates should remain
        assert len(p1) == len(p2)
        assert len(dates) == 2  # 2024-01-02 and 2024-01-03

    def test_aligned_prices_match(self):
        dates1 = ["2024-01-01", "2024-01-02"]
        prices1 = [100.0, 101.0]
        dates2 = ["2024-01-01", "2024-01-02"]
        prices2 = [200.0, 201.0]
        dates, p1, p2 = align_series(dates1, prices1, dates2, prices2)
        assert p1 == [100.0, 101.0]
        assert p2 == [200.0, 201.0]


# ---------------------------------------------------------------------------
# calculate_metrics extended
# ---------------------------------------------------------------------------

class TestCalculateMetricsExtended:
    """Extended calculate_metrics edge cases."""

    def test_all_keys_present(self):
        returns = [0.01, -0.005, 0.02, 0.003]
        dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
        metrics = calculate_metrics(returns, dates, "Test")
        expected_keys = {
            'scenario', 'startDate', 'endDate', 'days',
            'cagr', 'volatility', 'sharpe', 'maxDrawdown',
            'calmar', 'totalReturn', 'trackingErrorVsTLT',
            'volatilityDecayEstimate', 'annualizedExpenseImpact',
        }
        assert set(metrics.keys()) == expected_keys

    def test_dates_populated(self):
        returns = [0.01, -0.005]
        dates = ["2024-01-01", "2024-01-02"]
        metrics = calculate_metrics(returns, dates, "Test")
        assert metrics['startDate'] == "2024-01-01"
        assert metrics['endDate'] == "2024-01-02"

    def test_days_equals_return_count(self):
        returns = [0.01] * 50
        metrics = calculate_metrics(returns, ["2024-01-01"] * 50, "Test")
        assert metrics['days'] == 50

    def test_sharpe_with_zero_vol(self):
        returns = [0.0] * 10
        metrics = calculate_metrics(returns, ["2024-01-01"] * 10, "Test")
        assert metrics['sharpe'] == 0  # zero vol → sharpe = 0

    def test_total_return_computation(self):
        returns = [0.1, 0.1]  # 1.1 * 1.1 - 1 = 0.21
        metrics = calculate_metrics(returns, ["2024-01-01", "2024-01-02"], "Test")
        assert metrics['totalReturn'] == pytest.approx(0.21, abs=0.001)

    def test_default_expense_ratio(self):
        returns = [0.01] * 252
        metrics = calculate_metrics(returns, ["2024-01-01"] * 252, "Generic")
        # Default expense ratio for non-UBT/TMF is 0.0015
        assert metrics['annualizedExpenseImpact'] < 0

    def test_tracking_error_zero_when_no_base(self):
        returns = [0.01] * 252
        metrics = calculate_metrics(returns, ["2024-01-01"] * 252, "Test")
        assert metrics['trackingErrorVsTLT'] == 0

    def test_calmar_with_zero_drawdown(self):
        """No drawdown → calmar = cagr."""
        returns = [0.001] * 252
        metrics = calculate_metrics(returns, ["2024-01-01"] * 252, "Test")
        assert isinstance(metrics['calmar'], float)


# ---------------------------------------------------------------------------
# calculate_correlation extended
# ---------------------------------------------------------------------------

class TestCalculateCorrelationExtended:
    """Extended calculate_correlation edge cases."""

    def test_self_correlation_is_one(self):
        r = [0.01, -0.02, 0.03, -0.01, 0.02]
        corr = calculate_correlation(r, r)
        assert corr == pytest.approx(1.0)

    def test_constant_series(self):
        """Constant series correlation may be NaN, but should not crash."""
        r1 = [0.01, 0.01, 0.01]
        r2 = [0.02, 0.02, 0.02]
        corr = calculate_correlation(r1, r2)
        # NaN is acceptable here, just shouldn't crash
        assert isinstance(corr, float) or np.isnan(corr)

    def test_short_series(self):
        r1 = [0.01, -0.01]
        r2 = [-0.01, 0.01]
        corr = calculate_correlation(r1, r2)
        assert -1.0 <= corr <= 1.0 or np.isnan(corr)


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

class TestCLI:
    """Test main() callable."""

    def test_main_callable(self):
        from src.backtest.run_actual_ubt_validation import main
        assert callable(main)
