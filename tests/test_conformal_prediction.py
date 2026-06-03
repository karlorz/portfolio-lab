"""Tests for conformal risk prediction module."""
import pytest
import numpy as np
from src.risk.conformal_prediction import (
    ConformalRiskPredictor,
    ConformalInterval,
    create_conformal_predictor_for_garch,
    calculate_conformal_var,
)


class TestConformalRiskPredictor:
    """Test ConformalRiskPredictor class."""
    
    def test_initialization(self):
        """Test predictor initialization."""
        predictor = ConformalRiskPredictor(coverage=0.95)
        assert predictor.coverage == 0.95
        assert predictor.calibration_scores is None
        assert predictor.quantile_threshold is None
    
    def test_invalid_coverage(self):
        """Test invalid coverage level raises error."""
        with pytest.raises(ValueError):
            ConformalRiskPredictor(coverage=1.5)
        with pytest.raises(ValueError):
            ConformalRiskPredictor(coverage=0.0)
    
    def test_calibration(self):
        """Test calibration on synthetic data."""
        np.random.seed(42)
        n = 100
        predictions = np.random.normal(0, 1, n)
        outcomes = predictions + np.random.normal(0, 0.1, n)  # Small noise
        
        predictor = ConformalRiskPredictor(coverage=0.95)
        threshold = predictor.calibrate(predictions, outcomes)
        
        assert threshold >= 0
        assert predictor.calibration_scores is not None
        assert len(predictor.calibration_scores) == n
    
    def test_calibration_insufficient_data(self):
        """Test calibration with insufficient data."""
        predictor = ConformalRiskPredictor(coverage=0.95)
        with pytest.raises(ValueError):
            predictor.calibrate(np.array([1, 2, 3]), np.array([1, 2, 3]))
    
    def test_prediction_interval(self):
        """Test prediction interval construction."""
        predictor = ConformalRiskPredictor(coverage=0.95)
        predictor.calibrate(np.zeros(100), np.random.normal(0, 1, 100))
        
        interval = predictor.predict_interval(0.0)
        assert isinstance(interval, ConformalInterval)
        assert interval.lower < interval.upper
        assert interval.coverage == 0.95
        assert interval.width == interval.upper - interval.lower
    
    def test_interval_contains_prediction(self):
        """Test that point prediction is within interval."""
        predictor = ConformalRiskPredictor(coverage=0.95)
        predictor.calibrate(np.zeros(100), np.random.normal(0, 1, 100))
        
        for _ in range(10):
            pred = np.random.normal(0, 1)
            interval = predictor.predict_interval(pred)
            assert interval.lower <= pred <= interval.upper
    
    def test_coverage_diagnostics(self):
        """Test coverage diagnostics."""
        np.random.seed(42)
        n = 200
        
        # Calibration data
        cal_predictions = np.zeros(n)
        cal_outcomes = np.random.normal(0, 1, n)
        
        predictor = ConformalRiskPredictor(coverage=0.95)
        predictor.calibrate(cal_predictions, cal_outcomes)
        
        # Test data
        test_predictions = np.zeros(100)
        test_outcomes = np.random.normal(0, 1, 100)
        
        diagnostics = predictor.coverage_diagnostics(test_predictions, test_outcomes)
        
        assert 'nominal_coverage' in diagnostics
        assert 'empirical_coverage' in diagnostics
        assert 'coverage_gap' in diagnostics
        assert 'average_interval_width' in diagnostics
        assert isinstance(diagnostics['coverage_adequate'], bool)


class TestConformalInterval:
    """Test ConformalInterval dataclass."""
    
    def test_interval_creation(self):
        """Test interval creation."""
        interval = ConformalInterval(
            lower=-1.0,
            upper=1.0,
            point_estimate=0.0,
            coverage=0.95,
            width=2.0
        )
        assert interval.lower == -1.0
        assert interval.upper == 1.0
        assert interval.width == 2.0
    
    def test_contains_method(self):
        """Test contains method."""
        interval = ConformalInterval(
            lower=-1.0,
            upper=1.0,
            point_estimate=0.0,
            coverage=0.95,
            width=2.0
        )
        assert interval.contains(0.0)
        assert interval.contains(-0.5)
        assert interval.contains(0.5)
        assert not interval.contains(-1.5)
        assert not interval.contains(1.5)


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_conformal_predictor_for_garch(self):
        """Test GARCH-specific factory function."""
        predictor = create_conformal_predictor_for_garch(coverage=0.99)
        assert predictor.coverage == 0.99
    
    def test_calculate_conformal_var(self):
        """Test conformal VaR calculation."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 252)  # Daily returns
        
        interval = calculate_conformal_var(
            returns=returns,
            volatility=0.02,
            confidence_level=0.95,
            coverage=0.95
        )
        
        assert isinstance(interval, ConformalInterval)
        assert interval.coverage == 0.95
        assert interval.lower < interval.upper


class TestIntegration:
    """Integration tests for conformal prediction in risk context."""
    
    def test_garch_integration_flow(self):
        """Test integration with GARCH-CVaR workflow."""
        # Simulate GARCH-CVaR workflow
        np.random.seed(42)
        
        # Historical returns for calibration (positive = gains, negative = losses)
        historical_returns = np.random.normal(0, 0.02, 500)
        
        # Current volatility forecast from GARCH
        current_volatility = 0.025
        
        # Create and calibrate predictor using absolute returns as nonconformity
        predictor = ConformalRiskPredictor(coverage=0.95)
        
        # Use absolute returns for calibration (nonconformity scores)
        abs_returns = np.abs(historical_returns)
        predictor.calibrate(abs_returns[:-1], abs_returns[1:])
        
        # Predict VaR interval (VaR is typically reported as positive loss amount)
        var_interval = predictor.predict_var(current_volatility, 0.95)
        
        # Verify interval properties
        assert var_interval.width > 0  # Interval should have positive width
        assert var_interval.coverage == 0.95
        # For VaR, point estimate should be positive (loss amount)
        assert var_interval.point_estimate > 0
    
    def test_coverage_guarantee(self):
        """Test that coverage guarantee holds approximately."""
        np.random.seed(42)
        
        # Generate data with known distribution
        n_cal = 500
        n_test = 1000
        
        cal_predictions = np.random.normal(0, 1, n_cal)
        cal_outcomes = cal_predictions + np.random.normal(0, 0.2, n_cal)
        
        predictor = ConformalRiskPredictor(coverage=0.95)
        predictor.calibrate(cal_predictions, cal_outcomes)
        
        # Test set
        test_predictions = np.random.normal(0, 1, n_test)
        test_outcomes = test_predictions + np.random.normal(0, 0.2, n_test)
        
        # Check coverage
        covered = 0
        for pred, outcome in zip(test_predictions, test_outcomes):
            interval = predictor.predict_interval(pred)
            if interval.contains(outcome):
                covered += 1
        
        empirical_coverage = covered / n_test
        
        # Should be close to nominal coverage (within tolerance)
        assert abs(empirical_coverage - 0.95) < 0.1, \
            f"Coverage {empirical_coverage:.3f} too far from 0.95"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
