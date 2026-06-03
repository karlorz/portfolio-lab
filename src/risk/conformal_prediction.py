"""
Conformal Risk Prediction Module

Implements split conformal prediction for uncertainty quantification
in risk estimates. Provides guaranteed coverage (e.g., 95%) for prediction
intervals with theoretical backing.

Split Conformal Prediction:
1. Split data into calibration and test sets
2. Compute nonconformity scores on calibration set
3. Use scores to construct prediction intervals with guaranteed coverage

Usage:
    from src.risk.conformal_prediction import ConformalRiskPredictor
    
    predictor = ConformalRiskPredictor(coverage=0.95)
    predictor.calibrate(calibration_returns)
    intervals = predictor.predict(test_returns)
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConformalInterval:
    """Prediction interval with coverage guarantee."""
    lower: float
    upper: float
    point_estimate: float
    coverage: float  # Nominal coverage (e.g., 0.95)
    width: float     # Interval width
    
    def contains(self, value: float) -> bool:
        """Check if value falls within the interval."""
        return self.lower <= value <= self.upper


class ConformalRiskPredictor:
    """
    Split conformal predictor for risk estimation.
    
    Provides prediction intervals with guaranteed coverage for:
    - VaR (Value at Risk) estimates
    - CVaR (Conditional VaR) estimates
    - Volatility forecasts
    - Return predictions
    
    The predictor is distribution-free and works with any base model.
    """
    
    def __init__(self, coverage: float = 0.95, quantile_method: str = "exact"):
        """
        Initialize conformal predictor.
        
        Args:
            coverage: Nominal coverage level (e.g., 0.95 for 95% coverage)
            quantile_method: Method for quantile estimation
                - "exact": Exact quantile (requires many calibration points)
                - "linear": Linear interpolation
        """
        if not 0 < coverage < 1:
            raise ValueError(f"Coverage must be between 0 and 1, got {coverage}")
        
        self.coverage = coverage
        self.quantile_method = quantile_method
        self.calibration_scores: Optional[np.ndarray] = None
        self.quantile_threshold: Optional[float] = None
        
    def calibrate(self, calibration_predictions: np.ndarray, 
                  calibration_outcomes: np.ndarray) -> float:
        """
        Calibrate the predictor using calibration data.
        
        Args:
            calibration_predictions: Model predictions on calibration set
            calibration_outcomes: Actual outcomes on calibration set
            
        Returns:
            Quantile threshold for prediction intervals
        """
        if len(calibration_predictions) != len(calibration_outcomes):
            raise ValueError("Predictions and outcomes must have same length")
        
        if len(calibration_predictions) < 10:
            raise ValueError("Need at least 10 calibration points")
        
        # Compute nonconformity scores (absolute residuals)
        residuals = np.abs(calibration_predictions - calibration_outcomes)
        
        # Store calibration scores
        self.calibration_scores = residuals
        
        # Compute quantile threshold for desired coverage
        # For finite calibration set, we use ceil((n+1) * coverage) / n
        n = len(residuals)
        quantile_index = int(np.ceil((n + 1) * self.coverage)) - 1
        quantile_index = min(quantile_index, n - 1)
        
        sorted_scores = np.sort(residuals)
        self.quantile_threshold = sorted_scores[quantile_index]
        
        logger.info(
            f"Calibrated conformal predictor: coverage={self.coverage:.2f}, "
            f"threshold={self.quantile_threshold:.4f}, "
            f"calibration_size={n}"
        )
        
        return self.quantile_threshold
    
    def predict_interval(self, point_prediction: float) -> ConformalInterval:
        """
        Construct prediction interval for a new point prediction.
        
        Args:
            point_prediction: Model prediction for new observation
            
        Returns:
            ConformalInterval with guaranteed coverage
        """
        if self.quantile_threshold is None:
            raise RuntimeError("Predictor not calibrated. Call calibrate() first.")
        
        lower = point_prediction - self.quantile_threshold
        upper = point_prediction + self.quantile_threshold
        width = upper - lower
        
        return ConformalInterval(
            lower=lower,
            upper=upper,
            point_estimate=point_prediction,
            coverage=self.coverage,
            width=width
        )
    
    def predict_var(self, volatility_forecast: float, 
                   confidence_level: float = 0.95) -> ConformalInterval:
        """
        Predict VaR interval using conformal calibration.
        
        Args:
            volatility_forecast: Volatility forecast from base model
            confidence_level: VaR confidence level
            
        Returns:
            ConformalInterval for VaR estimate
        """
        # Base VaR estimate (assuming normal distribution for simplicity)
        from scipy.stats import norm
        z_score = norm.ppf(confidence_level)
        base_var = z_score * volatility_forecast
        
        return self.predict_interval(base_var)
    
    def predict_cvar(self, volatility_forecast: float,
                    confidence_level: float = 0.95) -> ConformalInterval:
        """
        Predict CVaR interval using conformal calibration.
        
        Args:
            volatility_forecast: Volatility forecast from base model
            confidence_level: CVaR confidence level
            
        Returns:
            ConformalInterval for CVaR estimate
        """
        # Base CVaR estimate (assuming normal distribution)
        from scipy.stats import norm
        z_score = norm.ppf(confidence_level)
        phi_z = norm.pdf(z_score)
        
        # CVaR = σ * φ(Φ⁻¹(α)) / (1-α)
        base_cvar = volatility_forecast * phi_z / (1 - confidence_level)
        
        return self.predict_interval(base_cvar)
    
    def coverage_diagnostics(self, test_predictions: np.ndarray,
                           test_outcomes: np.ndarray) -> Dict:
        """
        Evaluate coverage on test data.
        
        Args:
            test_predictions: Model predictions on test set
            test_outcomes: Actual outcomes on test set
            
        Returns:
            Dictionary with coverage metrics
        """
        if self.quantile_threshold is None:
            raise RuntimeError("Predictor not calibrated")
        
        # Compute intervals for all test points
        residuals = np.abs(test_predictions - test_outcomes)
        
        # Check empirical coverage
        covered = residuals <= self.quantile_threshold
        empirical_coverage = np.mean(covered)
        
        # Compute average interval width
        intervals = [self.predict_interval(pred) for pred in test_predictions]
        widths = [interval.width for interval in intervals]
        avg_width = np.mean(widths)
        
        # Coverage gap (empirical - nominal)
        coverage_gap = empirical_coverage - self.coverage
        
        # Winkler score (proper scoring rule for intervals)
        # Lower is better
        winkler_scores = []
        for i, interval in enumerate(intervals):
            outcome = test_outcomes[i]
            width = interval.width
            alpha = 1 - self.coverage
            
            if outcome < interval.lower:
                score = width + 2/alpha * (interval.lower - outcome)
            elif outcome > interval.upper:
                score = width + 2/alpha * (outcome - interval.upper)
            else:
                score = width
            winkler_scores.append(score)
        
        avg_winkler = np.mean(winkler_scores)
        
        diagnostics = {
            'nominal_coverage': self.coverage,
            'empirical_coverage': empirical_coverage,
            'coverage_gap': coverage_gap,
            'coverage_adequate': bool(abs(coverage_gap) < 0.05),  # Within 5%
            'average_interval_width': avg_width,
            'average_winkler_score': avg_winkler,
            'quantile_threshold': self.quantile_threshold,
            'n_test_points': len(test_predictions),
        }
        
        logger.info(
            f"Coverage diagnostics: nominal={self.coverage:.2f}, "
            f"empirical={empirical_coverage:.2f}, gap={coverage_gap:.3f}, "
            f"width={avg_width:.4f}"
        )
        
        return diagnostics


def create_conformal_predictor_for_garch(coverage: float = 0.95) -> ConformalRiskPredictor:
    """
    Factory function to create conformal predictor configured for GARCH-CVaR.
    
    Args:
        coverage: Desired coverage level (default: 0.95)
        
    Returns:
        Configured ConformalRiskPredictor
    """
    return ConformalRiskPredictor(coverage=coverage)


def calculate_conformal_var(returns: np.ndarray, 
                          volatility: float,
                          confidence_level: float = 0.95,
                          coverage: float = 0.95) -> ConformalInterval:
    """
    Calculate conformal VaR interval.
    
    Args:
        returns: Historical returns for calibration
        volatility: Current volatility forecast
        confidence_level: VaR confidence level
        coverage: Conformal coverage level
        
    Returns:
        ConformalInterval for VaR
    """
    from scipy.stats import norm
    
    # Calibrate conformal predictor
    predictor = ConformalRiskPredictor(coverage=coverage)
    
    # Use historical returns to estimate nonconformity scores
    # Assume mean zero for simplicity
    historical_var = np.abs(returns) * norm.ppf(confidence_level)
    
    # Calibrate on historical data
    # For simplicity, use historical VaR as "predictions" and actual returns as outcomes
    predictor.calibrate(historical_var[:-1], np.abs(returns[:-1]))
    
    # Predict for current volatility
    return predictor.predict_var(volatility, confidence_level)


# Export for module import
__all__ = [
    'ConformalInterval',
    'ConformalRiskPredictor',
    'create_conformal_predictor_for_garch',
    'calculate_conformal_var',
]
