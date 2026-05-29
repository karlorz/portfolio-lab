"""
Split-Conformal Prediction for Distribution-Free Risk Quantification.

Implements split-conformal prediction intervals with guaranteed coverage
probability (Vovk et al., 2005). Unlike parametric risk measures (GARCH,
normal VaR), conformal prediction makes no distributional assumptions and
provides finite-sample coverage guarantees.

Algorithm:
    1. Split returns into training and calibration sets
    2. Fit a simple location model on training data (mean/median)
    3. Compute nonconformity scores on calibration data: |r_i - prediction|
    4. Take the (1-alpha)(1+1/n) quantile of scores as threshold
    5. Prediction interval: point_estimate ± threshold

Coverage guarantee: P(Y in interval) >= 1-alpha for exchangeable data.

No ML dependencies — pure numpy/stdlib.

References:
    Vovk, Gammerman, Shafer (2005): "Algorithmic Learning in a Random World"
    Shafer, Vovk (2008): "A Tutorial on Conformal Prediction"
    Angelopoulos, Bates (2023): "Conformal Prediction: A Gentle Introduction"
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    'ConformalPrediction',
    'ConformalRiskQuantifier',
    'conformal_var',
    'conformal_cvar',
]


@dataclass
class ConformalPrediction:
    """Result from conformal prediction interval computation."""
    lower: float           # Lower bound of prediction interval
    upper: float           # Upper bound of prediction interval
    point_estimate: float  # Central prediction (median or mean)
    alpha: float           # Significance level (e.g., 0.05 for 95% interval)
    threshold: float       # Nonconformity score threshold

    @property
    def interval_width(self) -> float:
        return self.upper - self.lower


class ConformalRiskQuantifier:
    """Split-conformal prediction for portfolio return risk estimation.

    Provides distribution-free prediction intervals with guaranteed
    coverage probability. Uses median as the point estimator and
    absolute deviation as the nonconformity score.

    Usage:
        crq = ConformalRiskQuantifier(alpha=0.05)
        crq.fit(returns[:500])        # calibration
        pred = crq.predict(returns[500:550])  # prediction
        # pred.lower, pred.upper are 95% prediction bounds
    """

    def __init__(self, alpha: float = 0.05):
        """Initialize with significance level.

        Args:
            alpha: Significance level. 0.05 = 95% prediction interval.
        """
        self.alpha = alpha
        self._threshold: Optional[float] = None
        self._point_estimate: Optional[float] = None
        self._n_cal: int = 0

    def fit(self, returns: np.ndarray) -> 'ConformalRiskQuantifier':
        """Fit on calibration data — compute nonconformity threshold.

        Splits calibration data into train (first half) and calibration
        (second half), fits a median estimator on train, then computes
        the conformal threshold on calibration residuals.

        Args:
            returns: Array of historical daily returns.

        Returns:
            Self for chaining.
        """
        returns = np.asarray(returns, dtype=float)

        if len(returns) == 0:
            self._threshold = 0.0
            self._point_estimate = 0.0
            self._n_cal = 0
            return self

        if len(returns) < 10:
            # Very small dataset — use all data for both training and calibration
            self._point_estimate = float(np.median(returns))
            residuals = np.abs(returns - self._point_estimate)
            self._threshold = float(np.max(residuals))
            self._n_cal = len(returns)
            return self

        # Split: first half for training, second half for calibration
        mid = len(returns) // 2
        train = returns[:mid]
        calib = returns[mid:]

        # Point estimator: median (robust to outliers)
        self._point_estimate = float(np.median(train))

        # Nonconformity scores: absolute deviation from prediction
        residuals = np.abs(calib - self._point_estimate)
        self._n_cal = len(residuals)

        # Conformal threshold: (1-alpha)(1+1/n) quantile
        # This ensures finite-sample coverage guarantee
        quantile_level = (1 - self.alpha) * (1 + 1 / self._n_cal)
        quantile_level = min(quantile_level, 1.0)

        self._threshold = float(np.quantile(residuals, quantile_level))

        logger.info(
            "ConformalRiskQuantifier fitted: n_cal=%d, median=%.6f, threshold=%.6f (alpha=%.2f)",
            self._n_cal, self._point_estimate, self._threshold, self.alpha,
        )
        return self

    def predict(self, new_returns: np.ndarray) -> ConformalPrediction:
        """Compute conformal prediction interval for new observations.

        Args:
            new_returns: Array of new return observations.

        Returns:
            ConformalPrediction with lower/upper bounds.
        """
        if self._threshold is None:
            raise RuntimeError("Must call fit() before predict()")

        new_returns = np.asarray(new_returns, dtype=float)

        if len(new_returns) == 0:
            return ConformalPrediction(
                lower=self._point_estimate,
                upper=self._point_estimate,
                point_estimate=self._point_estimate,
                alpha=self.alpha,
                threshold=0.0,
            )

        # Point prediction: use fitted median
        point = self._point_estimate

        # Prediction interval: point ± threshold
        return ConformalPrediction(
            lower=point - self._threshold,
            upper=point + self._threshold,
            point_estimate=point,
            alpha=self.alpha,
            threshold=self._threshold,
        )


def conformal_var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Compute conformal Value-at-Risk.

    Uses split-conformal prediction to estimate VaR without
    distributional assumptions. The VaR is the lower bound of the
    (1-alpha) prediction interval.

    Args:
        returns: Historical daily returns for calibration.
        alpha: Significance level (0.05 = 95% VaR).

    Returns:
        Conformal VaR as a negative return (loss).
    """
    crq = ConformalRiskQuantifier(alpha=alpha)
    crq.fit(returns)
    pred = crq.predict(returns[-10:])  # Use recent returns for context
    return pred.lower


def conformal_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Compute conformal Conditional VaR (Expected Shortfall).

    Estimates CVaR as the mean of returns below the conformal VaR
    threshold. This combines conformal guarantees for the threshold
    with empirical tail estimation for the conditional expectation.

    Args:
        returns: Historical daily returns for calibration.
        alpha: Significance level (0.05 = 95% CVaR).

    Returns:
        Conformal CVaR as a negative return (loss).
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return 0.0

    # Get conformal VaR as threshold
    var_threshold = conformal_var(returns, alpha=alpha)

    # Mean of returns below the threshold
    tail = returns[returns <= var_threshold]
    if len(tail) == 0:
        # No returns below threshold — use the threshold itself
        return var_threshold

    return float(np.mean(tail))
