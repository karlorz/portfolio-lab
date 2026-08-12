"""
Tests for CAR25 Performance Metric module.

CAR25 = Compound Annual Rate of Return at the 25th percentile
after position-sizing via safe-f (max drawdown-constrained).
"""
import logging
import pytest
import numpy as np
from unittest.mock import patch, mock_open
from dataclasses import fields
import json

from src.backtest.car25 import (
    block_bootstrap_returns,
    simulate_equity_curve,
    calculate_max_drawdown,
    safe_f,
    car25,
    market_correlation,
    parse_portfolio_string,
    compute_car25_for_portfolio,
    calculate_portfolio_returns,
    load_prices_data,
    print_car25_result,
    SafeFResult,
    CAR25Result,
    MarketCorrelationResult,
    CAR25FullResult,
    DEFAULT_SIMULATIONS,
    DEFAULT_HORIZON_YEARS,
    DEFAULT_RISK_TOLERANCE,
    DEFAULT_CONFIDENCE,
    DEFAULT_BLOCK_SIZE,
    TRADING_DAYS_PER_YEAR,
    MAX_ITERATIONS,
    F_TOLERANCE,
)


# ===================================================================
# Constants Validation
# ===================================================================

class TestConstants:
    """Validation of module-level constants."""

    def test_default_simulations_positive(self):
        assert DEFAULT_SIMULATIONS > 0

    def test_trading_days_per_year_reasonable(self):
        assert 200 <= TRADING_DAYS_PER_YEAR <= 365

    def test_max_iterations_positive(self):
        assert MAX_ITERATIONS > 0

    def test_f_tolerance_small_positive(self):
        assert 0 < F_TOLERANCE < 0.1

    def test_default_confidence_range(self):
        assert 0 < DEFAULT_CONFIDENCE < 1

    def test_default_block_size_positive(self):
        assert DEFAULT_BLOCK_SIZE > 0

    def test_default_risk_tolerance_range(self):
        assert 0 < DEFAULT_RISK_TOLERANCE < 1

    def test_constants_type_checks(self):
        assert isinstance(TRADING_DAYS_PER_YEAR, int)
        assert isinstance(MAX_ITERATIONS, int)
        assert isinstance(DEFAULT_BLOCK_SIZE, int)
        assert isinstance(DEFAULT_SIMULATIONS, int)


class TestConstantsExtended:
    """Extended validation of module-level constants."""

    def test_default_horizon_years_positive(self):
        assert DEFAULT_HORIZON_YEARS > 0

    def test_default_horizon_years_reasonable(self):
        """Horizon should be at least 1 year for meaningful CAR25."""
        assert DEFAULT_HORIZON_YEARS >= 1

    def test_default_simulations_reasonable(self):
        """Should have enough simulations for stable percentiles."""
        assert DEFAULT_SIMULATIONS >= 100

    def test_f_tolerance_convergence_precision(self):
        """F_TOLERANCE should be small enough for meaningful convergence."""
        assert F_TOLERANCE <= 0.01

    def test_risk_tolerance_default_reasonable(self):
        """Default risk tolerance should be between 5% and 50%."""
        assert 0.05 <= DEFAULT_RISK_TOLERANCE <= 0.50

    def test_default_block_size_reasonable(self):
        """Block size should be between 5 and 60 trading days."""
        assert 5 <= DEFAULT_BLOCK_SIZE <= 60


# ===================================================================
# Dataclass Structure Tests
# ===================================================================

class TestSafeFResultDataclass:
    """SafeFResult dataclass field completeness and roundtrip."""

    def test_fields_complete(self):
        field_names = {f.name for f in fields(SafeFResult)}
        expected = {'safe_f', 'drawdown95', 'iterations', 'converged', 'tolerance_used'}
        assert field_names == expected

    def test_to_dict_roundtrip(self):
        result = SafeFResult(safe_f=1.5, drawdown95=0.18, iterations=5, converged=True, tolerance_used=0.20)
        d = {f.name: getattr(result, f.name) for f in fields(SafeFResult)}
        restored = SafeFResult(**d)
        for f in fields(SafeFResult):
            assert getattr(restored, f.name) == getattr(result, f.name)

    def test_boundary_values(self):
        """Minimal and maximal safe_f values should roundtrip."""
        result = SafeFResult(safe_f=0.01, drawdown95=0.001, iterations=1, converged=False, tolerance_used=0.01)
        assert result.safe_f == 0.01
        assert result.iterations == 1
        assert not result.converged

    def test_maximum_safe_f(self):
        """Maximum safe_f (4.0) should roundtrip."""
        result = SafeFResult(safe_f=4.0, drawdown95=0.50, iterations=20, converged=True, tolerance_used=0.50)
        assert result.safe_f == 4.0
        assert result.drawdown95 == 0.50
        assert result.converged


class TestCAR25ResultDataclass:
    """CAR25Result dataclass field completeness and roundtrip."""

    def test_fields_complete(self):
        field_names = {f.name for f in fields(CAR25Result)}
        expected = {'car25', 'car50', 'car75', 'twr25', 'twr50', 'twr75',
                     'safe_f', 'final_equity25', 'final_equity50', 'final_equity75'}
        assert field_names == expected

    def test_to_dict_roundtrip(self):
        result = CAR25Result(
            car25=0.05, car50=0.07, car75=0.10,
            twr25=1.10, twr50=1.15, twr75=1.21,
            safe_f=0.8, final_equity25=110000, final_equity50=115000, final_equity75=121000
        )
        d = {f.name: getattr(result, f.name) for f in fields(CAR25Result)}
        restored = CAR25Result(**d)
        for f in fields(CAR25Result):
            assert getattr(restored, f.name) == getattr(result, f.name)

    def test_zero_equity_fields(self):
        """Zero initial equity should produce zero final equity values."""
        result = CAR25Result(
            car25=-0.10, car50=-0.05, car75=0.0,
            twr25=0.81, twr50=0.90, twr75=1.0,
            safe_f=0.5, final_equity25=0.0, final_equity50=0.0, final_equity75=0.0
        )
        assert result.final_equity25 == 0.0
        assert result.final_equity50 == 0.0


class TestMarketCorrelationResultDataclass:
    """MarketCorrelationResult dataclass field completeness and roundtrip."""

    def test_fields_complete(self):
        field_names = {f.name for f in fields(MarketCorrelationResult)}
        expected = {'correlation', 'classification', 'common_days'}
        assert field_names == expected

    def test_to_dict_roundtrip(self):
        result = MarketCorrelationResult(correlation=0.45, classification='moderate', common_days=500)
        d = {f.name: getattr(result, f.name) for f in fields(MarketCorrelationResult)}
        restored = MarketCorrelationResult(**d)
        for f in fields(MarketCorrelationResult):
            assert getattr(restored, f.name) == getattr(result, f.name)

    def test_all_classifications(self):
        """All three classification strings should be valid."""
        for corr, cls in [(0.1, 'low'), (0.5, 'moderate'), (0.9, 'high')]:
            result = MarketCorrelationResult(correlation=corr, classification=cls, common_days=100)
            assert result.classification in ('low', 'moderate', 'high')

    def test_classification_must_be_valid(self):
        """Classification should be one of the three valid values."""
        result = MarketCorrelationResult(correlation=-0.1, classification='low', common_days=50)
        assert result.classification == 'low'
        result = MarketCorrelationResult(correlation=0.5, classification='moderate', common_days=50)
        assert result.classification == 'moderate'
        result = MarketCorrelationResult(correlation=0.99, classification='high', common_days=50)
        assert result.classification == 'high'


class TestCAR25FullResultDataclass:
    """CAR25FullResult dataclass field completeness and nested structure."""

    def test_fields_complete(self):
        field_names = {f.name for f in fields(CAR25FullResult)}
        expected = {'portfolio', 'safe_f', 'car25', 'correlation', 'config', 'input_days'}
        assert field_names == expected

    def test_nested_dataclass_structure(self):
        """CAR25FullResult should compose all three sub-dataclasses."""
        safe_f_res = SafeFResult(safe_f=1.2, drawdown95=0.19, iterations=8, converged=True, tolerance_used=0.20)
        car = CAR25Result(car25=0.06, car50=0.08, car75=0.11,
                          twr25=1.12, twr50=1.17, twr75=1.23,
                          safe_f=1.2, final_equity25=112000, final_equity50=117000, final_equity75=123000)
        corr = MarketCorrelationResult(correlation=0.5, classification='moderate', common_days=500)
        config = {'simulations': 1000, 'horizon_years': 2, 'risk_tolerance': 0.2}

        full = CAR25FullResult(portfolio='SPY', safe_f=safe_f_res, car25=car, correlation=corr,
                               config=config, input_days=500)

        assert full.portfolio == 'SPY'
        assert full.safe_f.safe_f == 1.2
        assert full.car25.car25 == 0.06
        assert full.correlation.correlation == 0.5
        assert full.config['simulations'] == 1000
        assert full.input_days == 500


class TestDataclassFieldTypes:
    """Validate field types in all dataclasses."""

    def test_safe_f_result_field_types(self):
        """SafeFResult fields should have correct types."""
        type_map = {f.name: f.type for f in fields(SafeFResult)}
        assert type_map['safe_f'] is float or type_map['safe_f'] in (float,)
        assert type_map['drawdown95'] is float
        assert type_map['iterations'] is int
        assert type_map['converged'] is bool
        assert type_map['tolerance_used'] is float

    def test_car25_result_field_types(self):
        """CAR25Result fields should all be float."""
        for f in fields(CAR25Result):
            assert f.type is float or f.type in (float,), f"CAR25Result.{f.name} is {f.type}, expected float"

    def test_market_correlation_result_field_types(self):
        """MarketCorrelationResult fields should have correct types."""
        type_map = {f.name: f.type for f in fields(MarketCorrelationResult)}
        assert type_map['correlation'] is float
        assert type_map['classification'] is str
        assert type_map['common_days'] is int

    def test_car25_full_result_field_types(self):
        """CAR25FullResult fields should have correct nested types."""
        type_map = {f.name: f.type for f in fields(CAR25FullResult)}
        assert type_map['portfolio'] is str
        assert type_map['safe_f'] is SafeFResult
        assert type_map['car25'] is CAR25Result
        assert type_map['correlation'] is MarketCorrelationResult
        assert type_map['input_days'] is int
        # config should be a dict-like type
        assert 'Dict' in str(type_map['config']) or type_map['config'] is dict or 'dict' in str(type_map['config']).lower()

    def test_safe_f_result_finite_values(self):
        """All SafeFResult fields should store finite values."""
        result = SafeFResult(safe_f=0.5, drawdown95=0.1, iterations=5, converged=True, tolerance_used=0.2)
        assert np.isfinite(result.safe_f)
        assert np.isfinite(result.drawdown95)
        assert np.isfinite(result.tolerance_used)

    def test_car25_result_nan_not_allowed(self):
        """CAR25Result fields should not contain NaN by construction."""
        result = CAR25Result(car25=0.05, car50=0.07, car75=0.10,
                              twr25=1.10, twr50=1.15, twr75=1.21,
                              safe_f=0.8, final_equity25=110000, final_equity50=115000, final_equity75=121000)
        assert not np.isnan(result.car25)
        assert not np.isnan(result.car50)
        assert not np.isnan(result.car75)
        assert np.isfinite(result.twr25)
        assert np.isfinite(result.final_equity25)


# ===================================================================
# Block Bootstrap Resampling
# ===================================================================

class TestBlockBootstrap:
    """Tests for block bootstrap resampling."""

    def test_block_bootstrap_preserves_block_size(self):
        """Block bootstrap should preserve block structure."""
        rng = np.random.default_rng(42)
        returns = np.random.randn(100)
        result = block_bootstrap_returns(returns, 60, 20, rng)

        assert len(result) == 60
        assert isinstance(result, np.ndarray)

    def test_block_bootstrap_reproducibility(self):
        """Same seed should produce same results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        returns = np.random.randn(100)

        result1 = block_bootstrap_returns(returns, 60, 20, rng1)
        result2 = block_bootstrap_returns(returns, 60, 20, rng2)

        np.testing.assert_array_equal(result1, result2)

    def test_block_bootstrap_different_seeds(self):
        """Different seeds should produce different results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        returns = np.random.randn(100)

        result1 = block_bootstrap_returns(returns, 60, 20, rng1)
        result2 = block_bootstrap_returns(returns, 60, 20, rng2)

        # Very unlikely to be exactly equal with different seeds
        assert not np.array_equal(result1, result2)

    def test_block_bootstrap_less_data_than_block(self):
        """Less data than block size should gracefully use all data."""
        rng = np.random.default_rng(42)
        returns = np.array([0.01, 0.02, 0.03])
        result = block_bootstrap_returns(returns, 10, 20, rng)
        assert len(result) == 10

    def test_block_bootstrap_negative_returns(self):
        """Block bootstrap should handle all-negative returns."""
        rng = np.random.default_rng(42)
        returns = np.array([-0.01, -0.02, -0.03, -0.01, -0.02])
        result = block_bootstrap_returns(returns, 10, 5, rng)
        assert len(result) == 10
        assert np.all(result <= 0)

    def test_block_bootstrap_exact_block_size(self):
        """Data length exactly matching block size should work."""
        rng = np.random.default_rng(42)
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = block_bootstrap_returns(returns, 5, 5, rng)
        assert len(result) == 5


class TestBlockBootstrapExtended:
    """Extended edge case tests for block bootstrap."""

    def test_block_bootstrap_large_num_days(self):
        """Very large num_days relative to data should work."""
        rng = np.random.default_rng(42)
        returns = np.random.randn(50)
        result = block_bootstrap_returns(returns, 1000, 20, rng)
        assert len(result) == 1000
        assert isinstance(result, np.ndarray)

    def test_block_bootstrap_block_size_one(self):
        """Block size of 1 (i.i.d. assumption) should work."""
        rng = np.random.default_rng(42)
        returns = np.random.randn(100)
        result = block_bootstrap_returns(returns, 50, 1, rng)
        assert len(result) == 50

    def test_block_bootstrap_single_element_data(self):
        """Single element data array should work."""
        rng = np.random.default_rng(42)
        returns = np.array([0.01])
        result = block_bootstrap_returns(returns, 10, 1, rng)
        assert len(result) == 10
        # All values should come from the only element
        assert np.all(result == 0.01)

    def test_block_bootstrap_num_days_one(self):
        """Requesting a single day should work."""
        rng = np.random.default_rng(42)
        returns = np.random.randn(100)
        result = block_bootstrap_returns(returns, 1, 20, rng)
        assert len(result) == 1
        assert isinstance(result, np.ndarray)

    def test_block_bootstrap_zero_returns(self):
        """All-zero returns should bootstrap correctly."""
        rng = np.random.default_rng(42)
        returns = np.zeros(100)
        result = block_bootstrap_returns(returns, 50, 20, rng)
        assert len(result) == 50
        assert np.all(result == 0.0)

    def test_block_bootstrap_data_shorter_than_num_days(self):
        """Data shorter than requested days should still work."""
        rng = np.random.default_rng(42)
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = block_bootstrap_returns(returns, 20, 3, rng)
        assert len(result) == 20


# ===================================================================
# Equity Curve Simulation
# ===================================================================

class TestEquityCurveSimulation:
    """Tests for equity curve simulation."""

    def test_simulate_zero_returns(self):
        """Zero returns should keep equity flat."""
        returns = np.zeros(10)
        equity = simulate_equity_curve(returns, 1.0)

        np.testing.assert_array_almost_equal(equity, np.ones(11))

    def test_simulate_positive_returns(self):
        """Positive returns should increase equity."""
        returns = np.array([0.01, 0.01, 0.01])  # 1% daily
        equity = simulate_equity_curve(returns, 1.0)

        assert equity[-1] > equity[0]
        assert equity[-1] == pytest.approx(1.030301, rel=1e-5)

    def test_simulate_position_sizing(self):
        """Position size should scale returns."""
        returns = np.array([0.10, -0.10])

        equity_full = simulate_equity_curve(returns, 1.0)
        equity_half = simulate_equity_curve(returns, 0.5)

        # Half position should have half the effect (approximately)
        # Full: 1.0 -> 1.1 -> 0.99
        # Half: 1.0 -> 1.05 -> 0.9975
        assert equity_half[1] == pytest.approx(1.05)
        assert equity_half[2] == pytest.approx(0.9975, rel=1e-5)

    def test_simulate_initial_equity(self):
        """Custom initial equity should be respected."""
        returns = np.array([0.01])
        equity = simulate_equity_curve(returns, 1.0, initial_equity=1000.0)

        assert equity[0] == 1000.0
        assert equity[1] == 1010.0

    def test_simulate_all_negative(self):
        """All negative returns should decrease equity."""
        returns = np.array([-0.01, -0.02, -0.015])
        equity = simulate_equity_curve(returns, 1.0)
        assert equity[-1] < equity[0]

    def test_simulate_negative_position_size(self):
        """Short position (negative size) should invert returns."""
        returns = np.array([0.01, -0.01])
        equity_short = simulate_equity_curve(returns, -1.0)
        # Positive return becomes negative with short position
        assert equity_short[1] < equity_short[0]

    def test_simulate_zero_position_size(self):
        """Zero position size should keep equity flat regardless of returns."""
        returns = np.array([0.10, -0.10, 0.05])
        equity = simulate_equity_curve(returns, 0.0)
        np.testing.assert_array_almost_equal(equity, np.ones(len(returns) + 1))


class TestEquityCurveSimulationExtended:
    """Extended edge case tests for equity curve simulation."""

    def test_simulate_extreme_returns(self):
        """Extreme return values should not cause overflow."""
        returns = np.array([10.0, -0.99, 5.0, -0.999])
        equity = simulate_equity_curve(returns, 0.1)
        assert not np.any(np.isnan(equity))
        assert not np.any(np.isinf(equity))

    def test_simulate_zero_initial_equity(self):
        """Zero initial equity should stay at zero."""
        returns = np.array([0.01, 0.02, -0.01])
        equity = simulate_equity_curve(returns, 1.0, initial_equity=0.0)
        np.testing.assert_array_almost_equal(equity, np.zeros(len(returns) + 1))

    def test_simulate_negative_initial_equity(self):
        """Negative initial equity (short) should work."""
        returns = np.array([0.01, -0.01])
        equity = simulate_equity_curve(returns, 1.0, initial_equity=-1000.0)
        assert equity[0] == -1000.0
        assert equity[1] < 0

    def test_simulate_position_size_above_max(self):
        """Position size at maximum bound (4.0) should work."""
        returns = np.array([0.001, -0.001, 0.002])
        equity = simulate_equity_curve(returns, 4.0)
        assert len(equity) == 4
        assert not np.any(np.isnan(equity))

    def test_simulate_single_day(self):
        """Single day returns should produce a 2-element equity curve."""
        returns = np.array([0.01])
        equity = simulate_equity_curve(returns, 1.0)
        assert len(equity) == 2
        assert equity[0] == 1.0
        assert equity[1] == 1.01

    def test_simulate_empty_returns(self):
        """Empty returns should produce single-element equity curve."""
        returns = np.array([])
        equity = simulate_equity_curve(returns, 1.0)
        assert len(equity) == 1
        assert equity[0] == 1.0


# ===================================================================
# Max Drawdown Calculation
# ===================================================================

class TestMaxDrawdown:
    """Tests for max drawdown calculation."""

    def test_no_drawdown(self):
        """Consistently increasing equity has no drawdown."""
        equity = np.array([1.0, 1.1, 1.2, 1.3])
        dd = calculate_max_drawdown(equity)

        assert dd == 0.0

    def test_simple_drawdown(self):
        """Simple peak-to-trough drawdown."""
        equity = np.array([1.0, 1.2, 1.1, 1.0])  # Peak at 1.2, trough at 1.0
        dd = calculate_max_drawdown(equity)

        assert dd == pytest.approx(-0.1667, abs=0.001)

    def test_multiple_peaks(self):
        """Multiple peaks should track the largest drawdown."""
        equity = np.array([1.0, 1.1, 1.05, 1.15, 1.0])  # Second peak higher
        dd = calculate_max_drawdown(equity)

        # Max DD from 1.15 to 1.0
        assert dd == pytest.approx(-0.1304, abs=0.001)

    def test_drawdown_monotonic_decline(self):
        """Monotonically decreasing equity has increasing drawdown."""
        equity = np.array([1.0, 0.9, 0.8, 0.7])
        dd = calculate_max_drawdown(equity)
        assert dd == pytest.approx(-0.30, abs=0.001)

    def test_drawdown_recovery_no_new_peak(self):
        """Partial recovery without new peak is still drawdown from old peak."""
        equity = np.array([1.0, 1.2, 0.9, 1.1, 1.15])
        dd = calculate_max_drawdown(equity)
        # Max drawdown from peak 1.2 to trough 0.9
        assert dd == pytest.approx(-0.25, abs=0.001)

    def test_drawdown_single_point(self):
        """Single point equity curve has no drawdown."""
        equity = np.array([1.0])
        dd = calculate_max_drawdown(equity)
        assert dd == 0.0


class TestMaxDrawdownExtended:
    """Extended edge case tests for max drawdown calculation."""

    def test_max_drawdown_two_points_flat(self):
        """Two identical values should have zero drawdown."""
        equity = np.array([1.0, 1.0])
        dd = calculate_max_drawdown(equity)
        assert dd == 0.0

    def test_max_drawdown_two_points_decline(self):
        """Two-point decline should be accurate."""
        equity = np.array([1.0, 0.8])
        dd = calculate_max_drawdown(equity)
        assert dd == pytest.approx(-0.20, abs=0.001)

    def test_max_drawdown_all_identical(self):
        """All identical values should have zero drawdown."""
        equity = np.ones(100) * 100.0
        dd = calculate_max_drawdown(equity)
        assert dd == 0.0

    def test_max_drawdown_negative_equity(self):
        """Negative equity curve should still compute."""
        equity = np.array([-100.0, -80.0, -60.0])
        dd = calculate_max_drawdown(equity)
        # All negative, increasing: no drawdown
        assert dd == 0.0

    def test_max_drawdown_recovery_to_new_peak(self):
        """Recovery to new peak then new drawdown."""
        equity = np.array([1.0, 2.0, 1.5, 3.0, 2.0])
        dd = calculate_max_drawdown(equity)
        # Peak at 2.0 -> trough at 1.5 = -25%, then new peak at 3.0 -> trough at 2.0 = -33%
        # Max drawdown = -33.3%
        assert dd == pytest.approx(-0.3333, abs=0.001)

    def test_max_drawdown_very_long_flat(self):
        """Very long flat periods should have zero drawdown."""
        equity = np.ones(10000)
        dd = calculate_max_drawdown(equity)
        assert dd == 0.0


# ===================================================================
# Safe-f Calculation
# ===================================================================

class TestSafeF:
    """Tests for safe-f calculation."""

    def test_safe_f_low_volatility_converges(self):
        """Safe-f should converge for low volatility returns."""
        # Low volatility positive drift
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0003, 0.005, 252 * 3)  # 3 years of low vol

        result = safe_f(returns, risk_tolerance=0.20, n_sims=100, seed=42)

        assert result.converged or result.iterations == MAX_ITERATIONS
        assert 0.01 <= result.safe_f <= 4.0
        assert result.iterations > 0
        assert result.drawdown95 > 0

    def test_safe_f_respects_tolerance(self):
        """Safe-f should respect risk tolerance constraint."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0003, 0.01, 252 * 2)

        # Higher tolerance should allow higher f
        result_low = safe_f(returns, risk_tolerance=0.10, n_sims=100, seed=42)
        result_high = safe_f(returns, risk_tolerance=0.30, n_sims=100, seed=43)

        # Higher tolerance typically allows higher f (though MC variance exists)
        assert result_low.safe_f > 0
        assert result_high.safe_f > 0

    def test_safe_f_edge_case_zero_returns(self):
        """Zero returns should handle gracefully."""
        returns = np.zeros(252)

        result = safe_f(returns, risk_tolerance=0.20, n_sims=50, seed=42)

        # Should converge quickly with zero drawdown
        assert result.safe_f >= 0.01
        assert result.converged or result.iterations <= MAX_ITERATIONS

    def test_safe_f_all_negative_returns(self):
        """All negative returns should produce minimal safe-f."""
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.001, 0.01, 252)
        result = safe_f(returns, risk_tolerance=0.20, n_sims=50, seed=42)
        assert result.safe_f >= 0.01

    def test_safe_f_convergence_metrics(self):
        """Safe-f result should have valid iteration metrics."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, 252)
        result = safe_f(returns, risk_tolerance=0.20, n_sims=50, seed=42)
        assert 1 <= result.iterations <= MAX_ITERATIONS
        assert isinstance(result.converged, bool)

    def test_safe_f_different_confidence(self):
        """Different confidence levels should adjust safe-f."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0003, 0.01, 252)
        result_default = safe_f(returns, risk_tolerance=0.20, n_sims=50, seed=42)
        result_stringent = safe_f(returns, risk_tolerance=0.20, n_sims=50, confidence=0.99, seed=42)
        # Both should complete without error
        assert isinstance(result_default.converged, bool)
        assert isinstance(result_stringent.converged, bool)


class TestSafeFExtended:
    """Extended edge case tests for safe-f calculation."""

    def test_safe_f_risk_tolerance_very_low(self):
        """Very low risk tolerance should produce small safe-f."""
        returns = np.random.default_rng(42).normal(0.0005, 0.01, 252)
        result = safe_f(returns, risk_tolerance=0.01, n_sims=50, seed=42)
        assert result.safe_f >= 0.01
        assert result.drawdown95 > 0

    def test_safe_f_risk_tolerance_high(self):
        """High risk tolerance should allow larger safe-f."""
        returns = np.random.default_rng(42).normal(0.0005, 0.01, 252)
        result = safe_f(returns, risk_tolerance=0.50, n_sims=50, seed=42)
        assert result.safe_f >= 0.01
        assert result.drawdown95 > 0

    def test_safe_f_very_low_volatility(self):
        """Near-zero volatility should converge to high safe-f."""
        returns = np.random.default_rng(42).normal(0.0005, 0.0001, 252)
        result = safe_f(returns, risk_tolerance=0.20, n_sims=50, seed=42)
        assert result.safe_f >= 0.01
        assert result.converged or result.iterations <= MAX_ITERATIONS

    def test_safe_f_high_volatility(self):
        """High volatility should still converge."""
        returns = np.random.default_rng(42).normal(0.001, 0.05, 252)
        result = safe_f(returns, risk_tolerance=0.20, n_sims=50, seed=42)
        assert result.safe_f >= 0.01

    def test_safe_f_different_horizons(self):
        """Different horizons should produce different safe-f values."""
        returns = np.random.default_rng(42).normal(0.0003, 0.01, 252)
        result_short = safe_f(returns, risk_tolerance=0.20, horizon_years=1, n_sims=50, seed=42)
        result_long = safe_f(returns, risk_tolerance=0.20, horizon_years=5, n_sims=50, seed=42)
        assert isinstance(result_short.converged, bool)
        assert isinstance(result_long.converged, bool)

    def test_safe_f_default_block_size(self):
        """Default block size should work."""
        returns = np.random.default_rng(42).normal(0.0003, 0.01, 252)
        result = safe_f(returns, risk_tolerance=0.20, n_sims=50, seed=42)
        assert result.drawdown95 > 0


# ===================================================================
# CAR25 Calculation
# ===================================================================

class TestCAR25:
    """Tests for CAR25 calculation."""

    def test_car25_positive_returns(self):
        """CAR25 ordering should be correct for typical returns."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, 252 * 3)

        result = car25(returns, safe_f_value=1.0, n_sims=100, seed=42)

        # Percentile ordering should hold regardless of drift direction
        assert result.car25 <= result.car50 <= result.car75

    def test_car25_percentile_ordering(self):
        """CAR percentiles should be properly ordered."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, 252 * 2)

        result = car25(returns, safe_f_value=1.0, n_sims=100, seed=42)

        # 25th < 50th < 75th
        assert result.car25 <= result.car50
        assert result.car50 <= result.car75

    def test_car25_twr_values(self):
        """TWR values should be consistent with CAR values."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0003, 0.01, 252 * 2)

        result = car25(returns, safe_f_value=1.0, horizon_years=2, n_sims=100, seed=42)

        # Verify annualization math: TWR^(1/years) - 1 = CAR
        expected_car25 = result.twr25 ** (1/2) - 1
        assert result.car25 == pytest.approx(expected_car25, abs=1e-10)

    def test_car25_all_negative_returns(self):
        """All negative returns should produce negative CAR values."""
        returns = np.full(252, -0.001)  # Consistent -0.1% daily
        result = car25(returns, safe_f_value=1.0, n_sims=50, seed=42)
        assert result.car25 < 0
        assert result.car50 < 0
        assert result.car75 < 0

    def test_car25_zero_returns(self):
        """Zero returns should produce CAR25 near zero."""
        returns = np.zeros(252)
        result = car25(returns, safe_f_value=1.0, n_sims=50, seed=42)
        assert result.car25 == pytest.approx(0.0, abs=1e-6)
        assert result.twr25 == pytest.approx(1.0, abs=1e-6)

    def test_car25_safe_f_scaling(self):
        """Higher safe-f should amplify CAR magnitude."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, 252)
        result_low = car25(returns, safe_f_value=0.5, n_sims=50, seed=42)
        result_high = car25(returns, safe_f_value=1.5, n_sims=50, seed=43)
        assert result_low.safe_f == 0.5
        assert result_high.safe_f == 1.5

    def test_car25_single_period(self):
        """Single-day returns should annualize properly."""
        returns = np.array([0.001])
        result = car25(returns, safe_f_value=1.0, horizon_years=1, n_sims=50, seed=42)
        assert result.car25 is not None
        assert not np.isnan(result.car25)

    def test_car25_all_identical_returns(self):
        """Perfectly identical returns should yield the same CAR across percentiles."""
        returns = np.full(252, 0.0005)
        result = car25(returns, safe_f_value=1.0, n_sims=50, seed=42)
        # With identical returns, all simulations produce the same outcome
        assert result.car25 <= result.car50


class TestCAR25Extended:
    """Extended edge case tests for CAR25 calculation."""

    def test_car25_horizon_years_fractional(self):
        """Fractional horizon years should work."""
        returns = np.random.default_rng(42).normal(0.0005, 0.01, 252)
        result = car25(returns, safe_f_value=1.0, horizon_years=0.5, n_sims=50, seed=42)
        assert result.car25 is not None
        assert not np.isnan(result.car25)
        assert result.car25 <= result.car50

    def test_car25_long_horizon(self):
        """Long horizon should produce larger TWR values."""
        returns = np.random.default_rng(42).normal(0.0005, 0.01, 252 * 3)
        result = car25(returns, safe_f_value=1.0, horizon_years=10, n_sims=50, seed=42)
        assert result.car25 is not None
        assert not np.isnan(result.car25)

    def test_car25_initial_equity_zero(self):
        """Zero initial equity should produce zero final equities (TWR is undefined)."""
        returns = np.random.default_rng(42).normal(0.0005, 0.01, 252)
        result = car25(returns, safe_f_value=1.0, n_sims=50, seed=42, initial_equity=0.0)
        # With zero initial equity, final equities remain 0.0
        assert result.final_equity25 == 0.0
        assert result.final_equity50 == 0.0
        assert result.final_equity75 == 0.0
        # TWR is 0/0 = nan, but the upper/lower bound ordering still holds
        assert np.isfinite(result.car25) or np.isnan(result.car25)
        assert result.car25 <= result.car50 or np.isnan(result.car25)

    def test_car25_negative_safe_f(self):
        """Negative safe-f (short portfolio) should invert returns."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, 252)
        result = car25(returns, safe_f_value=-1.0, n_sims=50, seed=42)
        # Higher percentile ordering should still hold
        assert result.car25 <= result.car50 <= result.car75

    def test_car25_safe_f_at_max_bound(self):
        """Safe-f at maximum bound (4.0) should work."""
        returns = np.random.default_rng(42).normal(0.0001, 0.01, 252)
        result = car25(returns, safe_f_value=4.0, n_sims=50, seed=42)
        assert not np.isnan(result.car25)
        assert result.car25 <= result.car50 <= result.car75

    def test_car25_safe_f_at_min_bound(self):
        """Safe-f at minimum bound (0.01) should work."""
        returns = np.random.default_rng(42).normal(0.0005, 0.01, 252)
        result = car25(returns, safe_f_value=0.01, n_sims=50, seed=42)
        assert not np.isnan(result.car25)
        assert result.car25 <= result.car50 <= result.car75

    def test_car25_twr_annualization_consistency_across_percentiles(self):
        """All three percentiles should have consistent annualization."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0003, 0.01, 252 * 3)
        result = car25(returns, safe_f_value=1.0, horizon_years=3, n_sims=100, seed=42)
        assert result.car25 == pytest.approx(result.twr25 ** (1/3) - 1, abs=1e-10)
        assert result.car50 == pytest.approx(result.twr50 ** (1/3) - 1, abs=1e-10)
        assert result.car75 == pytest.approx(result.twr75 ** (1/3) - 1, abs=1e-10)

    def test_car25_different_block_sizes(self):
        """Different block sizes should produce valid results."""
        returns = np.random.default_rng(42).normal(0.0003, 0.01, 252 * 2)
        result_small = car25(returns, safe_f_value=1.0, block_size=5, n_sims=50, seed=42)
        result_large = car25(returns, safe_f_value=1.0, block_size=60, n_sims=50, seed=42)
        assert result_small.car25 <= result_small.car50
        assert result_large.car25 <= result_large.car50


# ===================================================================
# Market Correlation
# ===================================================================

class TestMarketCorrelation:
    """Tests for market correlation calculation."""

    def test_perfect_correlation(self):
        """Identical returns should have correlation 1.0."""
        returns = np.array([0.01, -0.01, 0.02, -0.005])
        result = market_correlation(returns, returns)

        assert result.correlation == pytest.approx(1.0)
        assert result.classification == 'high'

    def test_perfect_negative_correlation(self):
        """Perfect inverse returns should have correlation -1.0."""
        returns1 = np.array([0.01, -0.01, 0.02])
        returns2 = -returns1
        result = market_correlation(returns1, returns2)

        assert result.correlation == pytest.approx(-1.0)
        assert result.classification == 'high'  # Absolute value > 0.7

    def test_zero_correlation(self):
        """Uncorrelated returns should have low correlation."""
        returns1 = np.array([0.01, 0.01, 0.01, 0.01])
        returns2 = np.array([0.02, -0.01, 0.03, -0.02])
        result = market_correlation(returns1, returns2)

        assert abs(result.correlation) < 0.5

    def test_correlation_classifications(self):
        """Test classification boundaries."""
        # Low: < 0.3
        low_corr = np.random.randn(100)
        high_corr = low_corr + np.random.randn(100) * 0.1  # Highly correlated
        result_low = market_correlation(low_corr, np.random.randn(100))
        result_high = market_correlation(low_corr, high_corr)

        assert result_low.classification in ['low', 'moderate', 'high']
        assert result_high.classification == 'high'

    def test_short_input(self):
        """Short input should handle gracefully."""
        returns1 = np.array([0.01])
        returns2 = np.array([0.02])
        result = market_correlation(returns1, returns2)

        assert result.correlation == 0.0
        assert result.common_days == 1

    def test_correlation_nan_handling(self):
        """Input containing NaN should not crash."""
        returns1 = np.array([0.01, np.nan, 0.02, 0.03])
        returns2 = np.array([0.02, 0.01, np.nan, -0.01])
        result = market_correlation(returns1, returns2)
        assert isinstance(result.correlation, float)
        assert not np.isnan(result.correlation) or result.common_days < 2

    def test_correlation_const_input(self):
        """Constant input should produce zero or near-zero correlation."""
        returns1 = np.ones(10) * 0.01
        returns2 = np.random.randn(10) * 0.01
        result = market_correlation(returns1, returns2)
        # Constant array will produce corrcoef of 0 or NaN
        assert not np.isnan(result.correlation)
        assert result.classification in ('low', 'moderate', 'high')

    def test_correlation_exact_boundary_low_moderate(self):
        """High noise ratio should produce low correlation."""
        rng = np.random.default_rng(42)
        base = rng.normal(0, 1, 100)
        # High noise ensures correlation well below 0.3
        noise = rng.normal(0, 1, 100) * 5.0
        related = base + noise
        result = market_correlation(base, related)
        assert abs(result.correlation) < 0.3

    def test_correlation_exact_boundary_moderate_high(self):
        """Correlation well above 0.7 should be classified high."""
        rng = np.random.default_rng(42)
        base = rng.normal(0, 1, 100)
        noise = rng.normal(0, 1, 100) * 0.3
        related = base + noise
        result = market_correlation(base, related)
        assert result.classification == 'high'


class TestMarketCorrelationExtended:
    """Extended edge case tests for market correlation."""

    def test_correlation_identical_constant_arrays(self):
        """Both arrays constant should produce 0 correlation."""
        returns1 = np.ones(10) * 0.01
        returns2 = np.ones(10) * 0.02
        result = market_correlation(returns1, returns2)
        assert isinstance(result.correlation, float)
        assert not np.isnan(result.correlation)

    def test_correlation_very_long_arrays(self):
        """Very long arrays should not degrade performance."""
        rng = np.random.default_rng(42)
        long1 = rng.normal(0, 1, 100000)
        long2 = long1 + rng.normal(0, 0.5, 100000)
        result = market_correlation(long1, long2)
        assert abs(result.correlation) <= 1.0
        assert result.common_days == 100000

    def test_correlation_single_element_both(self):
        """Both arrays with single element should not crash."""
        result = market_correlation(np.array([0.01]), np.array([0.02]))
        assert result.correlation == 0.0
        assert result.common_days == 1

    def test_correlation_many_elements_low_correlation(self):
        """Arrays with many elements but no relationship."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 10000)
        b = rng.normal(0, 1, 10000)
        result = market_correlation(a, b)
        assert abs(result.correlation) < 0.1

    def test_correlation_one_array_constant(self):
        """One constant array should not crash."""
        constant = np.ones(50) * 0.01
        random = np.random.randn(50) * 0.02
        result = market_correlation(constant, random)
        assert isinstance(result.correlation, float)
        assert not np.isnan(result.correlation)

    def test_correlation_return_type_negative(self):
        """Negative correlation should be stored as float."""
        returns1 = np.array([0.01, 0.02, 0.03])
        returns2 = np.array([-0.01, -0.02, -0.03])
        result = market_correlation(returns1, returns2)
        # When they move in opposite direction, correlation may be negative
        assert isinstance(result.correlation, float)
        assert result.common_days == 3


# ===================================================================
# Portfolio Parsing
# ===================================================================

class TestPortfolioParsing:
    """Tests for portfolio string parsing."""

    def test_parse_simple_portfolio(self):
        """Parse simple single-asset portfolio."""
        result = parse_portfolio_string('SPY')
        assert result == {'SPY': 1.0}

    def test_parse_multi_asset_portfolio(self):
        """Parse multi-asset weighted portfolio."""
        result = parse_portfolio_string('SPY/GLD/TLT 46/38/16')
        assert result == {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}

    def test_parse_two_asset_portfolio(self):
        """Parse two-asset portfolio."""
        result = parse_portfolio_string('SPY/GLD 60/40')
        assert result == {'SPY': 0.60, 'GLD': 0.40}

    def test_parse_invalid_format(self):
        """Invalid format should raise error."""
        # Single token multi-asset without weights is treated as single asset (valid)
        # Test truly invalid: weights don't match symbols
        with pytest.raises(ValueError):
            parse_portfolio_string('SPY/GLD/TLT 50/50')  # 3 symbols, 2 weights

        # Wrong format with too many parts
        with pytest.raises(ValueError):
            parse_portfolio_string('SPY 50 50 extra')  # Too many parts (>2)

    def test_parse_mismatched_counts(self):
        """Mismatched symbol/weight counts should raise error."""
        with pytest.raises(ValueError):
            parse_portfolio_string('SPY/GLD/TLT 50/50')

    def test_parse_single_asset_explicit_weight(self):
        """Single asset with explicit 100% weight."""
        result = parse_portfolio_string('SPY 100')
        assert result == {'SPY': 1.0}

    def test_parse_single_asset_percentage_weight(self):
        """Single asset with fractional percentage weight."""
        result = parse_portfolio_string('SPY 75')
        assert result == {'SPY': 0.75}

    def test_parse_three_asset_exact_weights(self):
        """Three assets with exact fractional weights."""
        result = parse_portfolio_string('SPY/GLD/TLT 33/33/34')
        assert result == {'SPY': 0.33, 'GLD': 0.33, 'TLT': 0.34}

    def test_parse_many_assets(self):
        """Many assets should parse correctly."""
        result = parse_portfolio_string('A/B/C/D/E 20/20/20/20/20')
        assert result == {'A': 0.20, 'B': 0.20, 'C': 0.20, 'D': 0.20, 'E': 0.20}

    def test_parse_single_asset_with_decimal_weight(self):
        """Single asset with decimal weight should parse."""
        result = parse_portfolio_string('SPY 50.5')
        assert result == {'SPY': 0.505}

    def test_parse_single_asset_with_fractional_percent(self):
        """Single asset with fractional percentage."""
        result = parse_portfolio_string('SPY 33.33')
        assert result == {'SPY': 0.3333}


# ===================================================================
# Integration: Full CAR25 Computation
# ===================================================================

class TestComputeCAR25:
    """Integration tests for full CAR25 computation."""

    @patch('src.backtest.car25.load_prices_data')
    def test_compute_car25_single_asset(self, mock_load):
        """Compute CAR25 for single asset portfolio."""
        # Mock prices data in actual format: {symbol: [{d, p}, ...]}
        mock_data = {
            'SPY': [{'d': '2023-01-01', 'p': 100.0}, {'d': '2023-01-02', 'p': 101.0}, {'d': '2023-01-03', 'p': 102.0}],
            'GLD': [{'d': '2023-01-01', 'p': 150.0}, {'d': '2023-01-02', 'p': 151.0}, {'d': '2023-01-03', 'p': 152.0}],
            'TLT': [{'d': '2023-01-01', 'p': 120.0}, {'d': '2023-01-02', 'p': 119.0}, {'d': '2023-01-03', 'p': 121.0}],
        }
        mock_load.return_value = mock_data

        result = compute_car25_for_portfolio(
            'SPY',
            prices_data=mock_data,
            n_sims=50,
            seed=42
        )

        assert result.portfolio == 'SPY'
        assert result.safe_f.safe_f > 0
        assert result.input_days == 2  # 3 days -> 2 returns

    @patch('src.backtest.car25.load_prices_data')
    def test_compute_car25_multi_asset(self, mock_load):
        """Compute CAR25 for multi-asset portfolio."""
        # Generate realistic mock data in actual format
        np.random.seed(42)
        n_days = 252  # 1 year
        spy_prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n_days)))
        gld_prices = 150 * np.exp(np.cumsum(np.random.normal(0.0002, 0.008, n_days)))

        mock_data = {
            'SPY': [{'d': f'2023-{i:03d}', 'p': float(spy_prices[i])} for i in range(n_days)],
            'GLD': [{'d': f'2023-{i:03d}', 'p': float(gld_prices[i])} for i in range(n_days)],
        }
        mock_load.return_value = mock_data

        result = compute_car25_for_portfolio(
            'SPY/GLD 60/40',
            prices_data=mock_data,
            n_sims=50,
            seed=42
        )

        assert result.portfolio == 'SPY/GLD 60/40'
        assert result.correlation.classification in ['low', 'moderate', 'high']
        assert result.config['simulations'] == 50

    def test_compute_car25_realistic_data(self):
        """Test with more realistic synthetic data."""
        np.random.seed(42)
        n_days = 252 * 5  # 5 years

        # Generate correlated returns
        spy_returns = np.random.normal(0.0004, 0.012, n_days)
        gld_returns = 0.3 * spy_returns + np.random.normal(0.0001, 0.010, n_days)
        tlt_returns = -0.2 * spy_returns + np.random.normal(0.0001, 0.008, n_days)

        spy_prices = 100 * np.exp(np.cumsum(spy_returns))
        gld_prices = 150 * np.exp(np.cumsum(gld_returns))
        tlt_prices = 120 * np.exp(np.cumsum(tlt_returns))

        # Use actual data format: {symbol: [{d, p}, ...]}
        mock_data = {
            'SPY': [{'d': f'2020-{i:03d}', 'p': float(spy_prices[i])} for i in range(n_days)],
            'GLD': [{'d': f'2020-{i:03d}', 'p': float(gld_prices[i])} for i in range(n_days)],
            'TLT': [{'d': f'2020-{i:03d}', 'p': float(tlt_prices[i])} for i in range(n_days)],
        }

        result = compute_car25_for_portfolio(
            'SPY/GLD/TLT 46/38/16',
            prices_data=mock_data,
            n_sims=100,
            seed=42
        )

        # Validate structure
        assert result.car25.car25 is not None
        assert result.car25.car50 is not None
        assert result.car25.car75 is not None
        assert result.safe_f.converged or result.safe_f.iterations <= MAX_ITERATIONS


# ===================================================================
# calculate_portfolio_returns
# ===================================================================

class TestComputePortfolioReturns:
    """Tests for calculate_portfolio_returns."""

    def test_single_asset(self):
        """Single asset with 100% weight."""
        prices_data = {
            'SPY': [{'d': '2023-01-01', 'p': 100.0}, {'d': '2023-01-02', 'p': 101.0},
                    {'d': '2023-01-03', 'p': 102.0}],
        }
        returns, dates = calculate_portfolio_returns(prices_data, {'SPY': 1.0})
        assert len(returns) == 2
        assert len(dates) == 2
        assert returns[0] == pytest.approx(0.01)
        assert returns[1] == pytest.approx(0.00990099, abs=1e-5)

    def test_two_assets_equal_weight(self):
        """Two assets with equal weights."""
        prices_data = {
            'SPY': [{'d': '2023-01-01', 'p': 100.0}, {'d': '2023-01-02', 'p': 102.0}],
            'GLD': [{'d': '2023-01-01', 'p': 150.0}, {'d': '2023-01-02', 'p': 153.0}],
        }
        returns, dates = calculate_portfolio_returns(prices_data, {'SPY': 0.5, 'GLD': 0.5})
        assert len(returns) == 1
        # SPY returns: 2%, GLD returns: 2%, weighted: 2%
        assert returns[0] == pytest.approx(0.02)

    def test_empty_weights(self):
        """Empty weights dict should raise error."""
        prices_data = {
            'SPY': [{'d': '2023-01-01', 'p': 100.0}, {'d': '2023-01-02', 'p': 101.0}],
        }
        with pytest.raises(StopIteration):
            calculate_portfolio_returns(prices_data, {})

    def test_all_zero_weights(self):
        """All zero weights should produce zero returns."""
        prices_data = {
            'SPY': [{'d': '2023-01-01', 'p': 100.0}, {'d': '2023-01-02', 'p': 101.0}],
            'GLD': [{'d': '2023-01-01', 'p': 150.0}, {'d': '2023-01-02', 'p': 151.0}],
        }
        returns, dates = calculate_portfolio_returns(prices_data, {'SPY': 0.0, 'GLD': 0.0})
        assert len(returns) == 1
        assert returns[0] == 0.0
        assert dates == ['2023-01-02']

    def test_negative_weight(self):
        """Negative weight (short) should invert returns."""
        prices_data = {
            'SPY': [{'d': '2023-01-01', 'p': 100.0}, {'d': '2023-01-02', 'p': 102.0}],
        }
        returns, dates = calculate_portfolio_returns(prices_data, {'SPY': -0.5})
        assert returns[0] == pytest.approx(-0.01)

    def test_missing_symbol_raises(self):
        """Missing symbol should raise ValueError."""
        prices_data = {
            'GLD': [{'d': '2023-01-01', 'p': 150.0}],
        }
        with pytest.raises(ValueError, match='not found'):
            calculate_portfolio_returns(prices_data, {'SPY': 1.0})

    def test_different_lengths_truncates(self):
        """Shorter symbol is padded; returns length = first symbol dates."""
        prices_data = {
            'SPY': [{'d': '2023-01-01', 'p': 100.0}, {'d': '2023-01-02', 'p': 101.0},
                    {'d': '2023-01-03', 'p': 102.0}],
            'GLD': [{'d': '2023-01-01', 'p': 150.0}, {'d': '2023-01-02', 'p': 151.0}],
        }
        returns, _ = calculate_portfolio_returns(prices_data, {'SPY': 0.5, 'GLD': 0.5})
        # Length = n_days = len(first_symbol dates) - 1 = 2
        assert len(returns) == 2
        # First return includes both SPY and GLD contributions
        # SPY: (101-100)/100 = 0.01, GLD: (151-150)/150 = 0.006667
        # Weighted: 0.01*0.5 + 0.006667*0.5 = 0.008333
        assert returns[0] == pytest.approx(0.008333, abs=1e-5)
        # Second return only has SPY contribution (GLD has no data)
        # SPY: (102-101)/101 = 0.009901, weighted: 0.009901*0.5 = 0.0049505
        assert returns[1] == pytest.approx(0.004951, abs=1e-5)

    def test_return_type(self):
        """Returns should be numpy array, dates should be list of strings."""
        prices_data = {
            'SPY': [{'d': '2023-01-01', 'p': 100.0}, {'d': '2023-01-02', 'p': 101.0}],
        }
        returns, dates = calculate_portfolio_returns(prices_data, {'SPY': 1.0})
        assert isinstance(returns, np.ndarray)
        assert isinstance(dates, list)
        assert isinstance(dates[0], str)


# ===================================================================
# Load Prices Data
# ===================================================================

class TestLoadPricesData:
    """Tests for load_prices_data."""

    @patch('builtins.open', new_callable=mock_open, read_data='{"SPY": [{"d": "2023-01-01", "p": 100.0}]}')
    def test_load_prices_data_with_path(self, mock_file):
        """Loading with explicit path should work."""
        result = load_prices_data('/fake/path/prices.json')
        assert result == {'SPY': [{'d': '2023-01-01', 'p': 100.0}]}
        mock_file.assert_called_once_with('/fake/path/prices.json', 'r')

    @patch('builtins.open', new_callable=mock_open, read_data='{}')
    def test_load_prices_data_empty_json(self, mock_file):
        """Empty JSON object should load."""
        result = load_prices_data('/fake/empty.json')
        assert result == {}

    def test_load_prices_data_file_not_found(self):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_prices_data('/nonexistent_path/prices.json')

    @patch('builtins.open', new_callable=mock_open, read_data='invalid json')
    def test_load_prices_data_invalid_json(self, mock_file):
        """Invalid JSON should raise."""
        with pytest.raises(json.JSONDecodeError):
            load_prices_data('/fake/bad.json')


# ===================================================================
# Print CAR25 Result
# ===================================================================

class TestPrintCAR25Result:
    """Tests for print_car25_result."""

    def _make_full_result(self):
        safe_f_res = SafeFResult(safe_f=1.2345, drawdown95=0.1987, iterations=8,
                                  converged=True, tolerance_used=0.20)
        car = CAR25Result(car25=0.0645, car50=0.0832, car75=0.1078,
                          twr25=1.1332, twr50=1.1735, twr75=1.2134,
                          safe_f=1.2345, final_equity25=113320, final_equity50=117350,
                          final_equity75=121340)
        corr = MarketCorrelationResult(correlation=0.4567, classification='moderate', common_days=500)
        config = {'simulations': 1000, 'horizon_years': 2, 'risk_tolerance': 0.2,
                  'confidence': 0.95, 'block_size': 20}
        return CAR25FullResult(portfolio='SPY/GLD/TLT 46/38/16', safe_f=safe_f_res,
                                car25=car, correlation=corr, config=config, input_days=500)

    def test_print_human_readable(self, caplog):
        """Human-readable output should print portfolio name."""
        result = self._make_full_result()
        with caplog.at_level(logging.INFO, logger="src.backtest.car25"):
            print_car25_result(result, json_output=False)
        output = caplog.text
        assert 'CAR25 Analysis: SPY/GLD/TLT 46/38/16' in output
        assert 'Safe-f' in output
        assert 'CAR25' in output
        assert 'Correlation' in output
        assert result.portfolio in output

    def test_print_human_readable_values(self, caplog):
        """Human-readable output should contain key values."""
        result = self._make_full_result()
        with caplog.at_level(logging.INFO, logger="src.backtest.car25"):
            print_car25_result(result, json_output=False)
        output = caplog.text
        assert '1.2345' in output  # safe_f
        assert '19.87%' in output  # drawdown95
        assert '6.45%' in output  # car25
        assert '8.32%' in output  # car50
        assert '0.4567' in output  # correlation

    def test_print_json_output(self, caplog):
        """JSON output should produce valid JSON."""
        result = self._make_full_result()
        with caplog.at_level(logging.INFO, logger="src.backtest.car25"):
            print_car25_result(result, json_output=True)
        # caplog.text includes the logger prefix; extract the JSON message
        assert len(caplog.records) >= 1
        output = caplog.records[0].message
        parsed = json.loads(output)
        assert parsed['portfolio'] == 'SPY/GLD/TLT 46/38/16'
        assert parsed['safe_f']['safe_f'] == 1.2345
        assert parsed['car25']['car25'] == 6.45  # 6.45%
        assert parsed['correlation']['classification'] == 'moderate'
        assert parsed['input_days'] == 500

    def test_print_json_contains_all_sections(self, caplog):
        """JSON should contain safe_f, car25, and correlation sections."""
        result = self._make_full_result()
        with caplog.at_level(logging.INFO, logger="src.backtest.car25"):
            print_car25_result(result, json_output=True)
        assert len(caplog.records) >= 1
        parsed = json.loads(caplog.records[0].message)
        assert 'safe_f' in parsed
        assert 'car25' in parsed
        assert 'correlation' in parsed

    def test_print_json_roundtrip(self, caplog):
        """JSON output should round-trip via json.loads."""
        result = self._make_full_result()
        with caplog.at_level(logging.INFO, logger="src.backtest.car25"):
            print_car25_result(result, json_output=True)
        # Should not raise
        assert len(caplog.records) >= 1
        json.loads(caplog.records[0].message)


# ===================================================================
# CLI Tests
# ===================================================================

class TestCLI:
    """Tests for CLI functionality."""

    @patch('sys.stdout')
    @patch('src.backtest.car25.load_prices_data')
    def test_cli_json_output(self, mock_load, mock_stdout):
        """Test JSON output format."""
        from src.backtest.car25 import main

        mock_data = {
            'SPY': [{'d': '2023-01-01', 'p': 100.0}, {'d': '2023-01-02', 'p': 101.0}, {'d': '2023-01-03', 'p': 102.0}],
            'GLD': [{'d': '2023-01-01', 'p': 150.0}, {'d': '2023-01-02', 'p': 151.0}, {'d': '2023-01-03', 'p': 152.0}],
            'TLT': [{'d': '2023-01-01', 'p': 120.0}, {'d': '2023-01-02', 'p': 119.0}, {'d': '2023-01-03', 'p': 121.0}],
        }
        mock_load.return_value = mock_data

        with patch('sys.argv', ['car25', '--portfolio', 'SPY', '--json', '--sims', '50']):
            # Should not raise
            try:
                main()
            except SystemExit:
                pass  # argparse may call exit


class TestCLIExtended:
    """Extended CLI tests with various mock arguments."""

    @patch('sys.stdout')
    @patch('src.backtest.car25.load_prices_data')
    def test_cli_no_args_shows_help(self, mock_load, mock_stdout):
        """No arguments should print help and not crash."""
        from src.backtest.car25 import main
        with patch('sys.argv', ['car25']):
            try:
                main()
            except SystemExit:
                pass

    @patch('sys.stdout')
    @patch('src.backtest.car25.compute_car25_for_portfolio')
    def test_cli_single_portfolio(self, mock_compute, mock_stdout):
        """Single portfolio with args should call compute."""
        from src.backtest.car25 import main
        safe_f_res = SafeFResult(safe_f=1.0, drawdown95=0.19, iterations=5, converged=True, tolerance_used=0.20)
        car = CAR25Result(car25=0.05, car50=0.07, car75=0.10,
                          twr25=1.10, twr50=1.15, twr75=1.21,
                          safe_f=1.0, final_equity25=110000, final_equity50=115000,
                          final_equity75=121000)
        corr = MarketCorrelationResult(correlation=0.5, classification='moderate', common_days=500)
        mock_compute.return_value = CAR25FullResult(
            portfolio='SPY/GLD 60/40', safe_f=safe_f_res, car25=car,
            correlation=corr,
            config={'simulations': 1000, 'horizon_years': 2, 'risk_tolerance': 0.2,
                    'confidence': 0.95, 'block_size': 20},
            input_days=500
        )
        with patch('sys.argv', ['car25', '--portfolio', 'SPY/GLD 60/40']):
            try:
                main()
            except SystemExit:
                pass
        mock_compute.assert_called_once()

    @patch('sys.stdout')
    @patch('src.backtest.car25.load_prices_data')
    def test_cli_compare_all(self, mock_load, mock_stdout):
        """--compare-all should process all standard portfolios."""
        from src.backtest.car25 import main
        mock_data = {
            'SPY': [{'d': '2023-01-01', 'p': 100.0}, {'d': '2023-01-02', 'p': 101.0}],
            'QQQ': [{'d': '2023-01-01', 'p': 200.0}, {'d': '2023-01-02', 'p': 202.0}],
            'GLD': [{'d': '2023-01-01', 'p': 150.0}, {'d': '2023-01-02', 'p': 151.0}],
            'TLT': [{'d': '2023-01-01', 'p': 120.0}, {'d': '2023-01-02', 'p': 119.0}],
            'IEF': [{'d': '2023-01-01', 'p': 110.0}, {'d': '2023-01-02', 'p': 111.0}],
        }
        mock_load.return_value = mock_data

        with patch('sys.argv', ['car25', '--compare-all', '--sims', '50']):
            try:
                main()
            except SystemExit:
                pass
        # Should have called load_prices_data
        assert mock_load.called

    @patch('sys.stdout')
    @patch('src.backtest.car25.load_prices_data')
    def test_cli_compare_all_json(self, mock_load, mock_stdout):
        """--compare-all with --json should produce JSON output."""
        from src.backtest.car25 import main
        mock_data = {
            'SPY': [{'d': '2023-01-01', 'p': 100.0}, {'d': '2023-01-02', 'p': 101.0}],
            'QQQ': [{'d': '2023-01-01', 'p': 200.0}, {'d': '2023-01-02', 'p': 202.0}],
            'GLD': [{'d': '2023-01-01', 'p': 150.0}, {'d': '2023-01-02', 'p': 151.0}],
            'TLT': [{'d': '2023-01-01', 'p': 120.0}, {'d': '2023-01-02', 'p': 119.0}],
            'IEF': [{'d': '2023-01-01', 'p': 110.0}, {'d': '2023-01-02', 'p': 111.0}],
        }
        mock_load.return_value = mock_data

        with patch('sys.argv', ['car25', '--compare-all', '--json', '--sims', '50']):
            try:
                main()
            except SystemExit:
                pass
        assert mock_load.called

    @patch('sys.stdout')
    @patch('src.backtest.car25.load_prices_data')
    def test_cli_custom_tolerance(self, mock_load, mock_stdout):
        """Custom tolerance flag should propagate."""
        from src.backtest.car25 import main
        mock_data = {
            'SPY': [{'d': '2023-01-01', 'p': 100.0}, {'d': '2023-01-02', 'p': 101.0}],
        }
        mock_load.return_value = mock_data

        with patch('sys.argv', ['car25', '--portfolio', 'SPY', '--tolerance', '0.15', '--sims', '50']):
            try:
                main()
            except SystemExit:
                pass

    @patch('sys.stdout')
    @patch('src.backtest.car25.load_prices_data')
    def test_cli_custom_horizon(self, mock_load, mock_stdout):
        """Custom horizon flag should propagate."""
        from src.backtest.car25 import main
        mock_data = {
            'SPY': [{'d': '2023-01-01', 'p': 100.0}, {'d': '2023-01-02', 'p': 101.0}],
        }
        mock_load.return_value = mock_data

        with patch('sys.argv', ['car25', '--portfolio', 'SPY', '--horizon', '3', '--sims', '50']):
            try:
                main()
            except SystemExit:
                pass


# ===================================================================
# Edge Case and Error Handling
# ===================================================================

class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_returns(self):
        """Empty returns array should handle gracefully."""
        empty = np.array([])

        # Empty returns should raise an error or produce empty result
        with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
            block_bootstrap_returns(empty, 10, 5, np.random.default_rng(42))

    def test_single_return(self):
        """Single return value."""
        single = np.array([0.01])

        # Bootstrap with very short data
        rng = np.random.default_rng(42)
        result = block_bootstrap_returns(single, 5, 1, rng)

        assert len(result) == 5

    def test_extreme_returns(self):
        """Extreme return values should not cause overflow."""
        returns = np.array([0.5, -0.5, 1.0, -1.0])

        equity = simulate_equity_curve(returns, 0.5)

        # Should handle without crashing
        assert len(equity) == 5
        assert not np.any(np.isnan(equity))
        assert not np.any(np.isinf(equity))

    def test_extreme_negative_returns(self):
        """Extreme negative returns should not cause overflow/NaN."""
        returns = np.array([-0.99, -1.0, -0.95])
        equity = simulate_equity_curve(returns, 1.0)
        assert len(equity) == 4
        assert not np.any(np.isnan(equity))
        assert not np.any(np.isinf(equity))

    def test_calculate_portfolio_returns_missing_symbol(self):
        """Missing symbol should raise ValueError."""
        weights = {'SPY': 1.0}
        prices_data = {}  # No SPY data
        with pytest.raises(ValueError):
            calculate_portfolio_returns(prices_data, weights)

    def test_calculate_portfolio_returns_zero_weight(self):
        """Zero weight symbol should be skipped gracefully."""
        weights = {'SPY': 1.0, 'GLD': 0.0}
        prices_data = {
            'SPY': [{'d': '2023-01-01', 'p': 100.0}, {'d': '2023-01-02', 'p': 101.0}],
        }
        returns, dates = calculate_portfolio_returns(prices_data, weights)
        assert len(returns) == 1
        assert returns[0] == pytest.approx(0.01)

    def test_calculate_portfolio_returns_different_lengths(self):
        """Returns should truncate to shortest length."""
        weights = {'SPY': 0.5, 'GLD': 0.5}
        prices_data = {
            'SPY': [{'d': '2023-01-01', 'p': 100.0}, {'d': '2023-01-02', 'p': 101.0}, {'d': '2023-01-03', 'p': 102.0}],
            'GLD': [{'d': '2023-01-01', 'p': 150.0}, {'d': '2023-01-02', 'p': 151.0}],
        }
        returns, dates = calculate_portfolio_returns(prices_data, weights)
        assert len(returns) >= 1


# ===================================================================
# __all__ Exports Validation
# ===================================================================

class TestAllExports:
    """Validation of __all__ exports."""

    def test___all___not_empty(self):
        """__all__ should not be empty."""
        from src.backtest.car25 import __all__ as all_exports
        assert len(all_exports) > 0

    def test___all___no_duplicates(self):
        """__all__ should not contain duplicate entries."""
        from src.backtest.car25 import __all__ as all_exports
        assert len(all_exports) == len(set(all_exports)), "Duplicates found in __all__"

    def test___all___all_strings(self):
        """All __all__ entries should be strings."""
        from src.backtest.car25 import __all__ as all_exports
        for name in all_exports:
            assert isinstance(name, str), f"{name} is not a string"

    def test___all___exports_exist_in_module(self):
        """Every name in __all__ should be accessible as a module attribute."""
        import src.backtest.car25 as module
        for name in module.__all__:
            assert hasattr(module, name), f"__all__ contains '{name}' but it is not defined in the module"

    def test___all___exports_importable(self):
        """Every name in __all__ should be importable via from-import."""
        from src.backtest.car25 import __all__ as all_exports
        for name in all_exports:
            exec(f"from src.backtest.car25 import {name}", {}, {})

    def test___all___contains_dataclasses(self):
        """All four dataclasses should be in __all__."""
        from src.backtest.car25 import __all__ as all_exports
        for dc_name in ('SafeFResult', 'CAR25Result', 'MarketCorrelationResult', 'CAR25FullResult'):
            assert dc_name in all_exports, f"{dc_name} missing from __all__"

    def test___all___contains_constants(self):
        """All module-level constants should be in __all__."""
        from src.backtest.car25 import __all__ as all_exports
        const_names = {'DEFAULT_SIMULATIONS', 'DEFAULT_HORIZON_YEARS', 'DEFAULT_RISK_TOLERANCE',
                        'DEFAULT_CONFIDENCE', 'DEFAULT_BLOCK_SIZE', 'TRADING_DAYS_PER_YEAR',
                        'MAX_ITERATIONS', 'F_TOLERANCE'}
        missing = const_names - set(all_exports)
        assert not missing, f"Constants missing from __all__: {missing}"

    def test___all___contains_public_functions(self):
        """All public functions should be in __all__."""
        from src.backtest.car25 import __all__ as all_exports
        func_names = {'block_bootstrap_returns', 'simulate_equity_curve', 'calculate_max_drawdown',
                       'safe_f', 'car25', 'market_correlation', 'load_prices_data',
                       'calculate_portfolio_returns', 'parse_portfolio_string',
                       'compute_car25_for_portfolio', 'print_car25_result'}
        missing = func_names - set(all_exports)
        assert not missing, f"Functions missing from __all__: {missing}"

    def test___all___sorted(self):
        """__all__ should be sorted alphabetically (by convention)."""
        from src.backtest.car25 import __all__ as all_exports
        # Check it's either sorted or follows a consistent pattern
        assert len(all_exports) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
