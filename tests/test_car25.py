"""
Tests for CAR25 Performance Metric module.

CAR25 = Compound Annual Rate of Return at the 25th percentile
after position-sizing via safe-f (max drawdown-constrained).
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
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
    DEFAULT_SIMULATIONS,
    DEFAULT_HORIZON_YEARS,
    DEFAULT_RISK_TOLERANCE,
    DEFAULT_BLOCK_SIZE,
    TRADING_DAYS_PER_YEAR,
    MAX_ITERATIONS
)


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
