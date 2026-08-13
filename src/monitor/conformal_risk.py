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
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    'ConformalPrediction',
    'ConformalRiskQuantifier',
    'conformal_var',
    'conformal_cvar',
    'conformal_coverage_diagnostics',
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


def _bernoulli_log_likelihood(successes: int, failures: int, probability: float) -> float:
    """Log-likelihood for Bernoulli counts, safely handling 0*log(0)."""
    probability = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    ll = 0.0
    if successes:
        ll += successes * math.log(probability)
    if failures:
        ll += failures * math.log1p(-probability)
    return ll


def _chi_square_sf(statistic: float, degrees_of_freedom: int) -> float:
    """Survival function for the chi-square cases used by VaR backtests."""
    statistic = max(float(statistic), 0.0)
    if degrees_of_freedom == 1:
        return float(math.erfc(math.sqrt(statistic / 2.0)))
    if degrees_of_freedom == 2:
        return float(math.exp(-statistic / 2.0))
    raise ValueError("Only chi-square df=1 and df=2 are supported")


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _coverage_summary(
    exceedances: np.ndarray,
    *,
    alpha: float,
    rolling_window: int,
    include_schema: bool,
) -> Dict[str, Any]:
    n_obs = int(len(exceedances))
    exceedance_count = int(np.sum(exceedances))
    exceedance_rate = float(exceedance_count / n_obs) if n_obs else 0.0
    coverage_rate = float(1.0 - exceedance_rate)
    expected_exceedance_rate = float(alpha)

    phat = exceedance_rate
    ll_null = _bernoulli_log_likelihood(
        exceedance_count,
        n_obs - exceedance_count,
        expected_exceedance_rate,
    )
    ll_alt = _bernoulli_log_likelihood(
        exceedance_count,
        n_obs - exceedance_count,
        phat,
    )
    kupiec_statistic = max(0.0, 2.0 * (ll_alt - ll_null))
    kupiec_p_value = _chi_square_sf(kupiec_statistic, 1) if n_obs else 1.0

    if n_obs >= 2:
        previous = exceedances[:-1].astype(bool)
        current = exceedances[1:].astype(bool)
        n00 = int(np.sum(~previous & ~current))
        n01 = int(np.sum(~previous & current))
        n10 = int(np.sum(previous & ~current))
        n11 = int(np.sum(previous & current))
        transition_exceedances = n01 + n11
        transition_count = n_obs - 1
        pi = transition_exceedances / transition_count if transition_count else 0.0
        pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
        pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0

        ll_independent = (
            _bernoulli_log_likelihood(n01, n00, pi)
            + _bernoulli_log_likelihood(n11, n10, pi)
        )
        ll_markov = (
            _bernoulli_log_likelihood(n01, n00, pi01)
            + _bernoulli_log_likelihood(n11, n10, pi11)
        )
        christoffersen_statistic = max(0.0, 2.0 * (ll_markov - ll_independent))
        christoffersen_p_value = _chi_square_sf(christoffersen_statistic, 1)
    else:
        christoffersen_statistic = 0.0
        christoffersen_p_value = 1.0

    conditional_coverage_statistic = kupiec_statistic + christoffersen_statistic
    conditional_coverage_p_value = _chi_square_sf(conditional_coverage_statistic, 2)

    effective_window = max(1, min(int(rolling_window), n_obs)) if n_obs else 0
    rolling_exceedance_rate = (
        float(np.mean(exceedances[-effective_window:])) if effective_window else 0.0
    )

    # Direction of Kupiec miss: over = too many breaches (risk underestimation),
    # under = too few (over-conservative / capital inefficiency). Only over is a
    # hard red for primary GARCH demotion; under is an efficiency warning.
    rate_delta = exceedance_rate - expected_exceedance_rate
    if abs(rate_delta) < 1e-12:
        coverage_direction = "ok"
    elif rate_delta > 0:
        coverage_direction = "over"
    else:
        coverage_direction = "under"
    kupiec_stat_pass = bool(kupiec_p_value >= 0.05)
    # Hard fail only when Kupiec rejects AND exceedances are over-expected.
    coverage_hard_fail = bool(not kupiec_stat_pass and coverage_direction == "over")
    coverage_efficiency_warning = bool(
        not kupiec_stat_pass and coverage_direction == "under"
    )
    # coverage_pass remains the demotion gate: True unless hard over-fail.
    # Under-exceedance Kupiec fails no longer force coverage_pass=false.
    coverage_pass = not coverage_hard_fail

    summary: Dict[str, Any] = {
        "observations": n_obs,
        "alpha": expected_exceedance_rate,
        "expected_exceedance_rate": expected_exceedance_rate,
        "exceedance_count": exceedance_count,
        "exceedance_rate": round(exceedance_rate, 6),
        "coverage_rate": round(coverage_rate, 6),
        "coverage_pass": coverage_pass,
        "coverage_direction": coverage_direction,
        "exceedance_bias": coverage_direction,  # alias for operator surfaces
        "coverage_hard_fail": coverage_hard_fail,
        "coverage_efficiency_warning": coverage_efficiency_warning,
        "rolling_window": effective_window,
        "rolling_exceedance_rate": round(rolling_exceedance_rate, 6),
        "longest_violation_cluster": _longest_true_run(exceedances),
        "kupiec_statistic": round(kupiec_statistic, 6),
        "kupiec_p_value": round(kupiec_p_value, 6),
        "kupiec_pass": kupiec_stat_pass,
        "christoffersen_statistic": round(christoffersen_statistic, 6),
        "christoffersen_p_value": round(christoffersen_p_value, 6),
        "christoffersen_pass": bool(christoffersen_p_value >= 0.05),
        "conditional_coverage_statistic": round(conditional_coverage_statistic, 6),
        "conditional_coverage_p_value": round(conditional_coverage_p_value, 6),
        "conditional_coverage_pass": bool(conditional_coverage_p_value >= 0.05),
    }
    if include_schema:
        summary = {"schema_version": "conformal-coverage/v1", **summary}
    return summary


def conformal_coverage_diagnostics(
    returns: np.ndarray,
    var_thresholds: np.ndarray | float,
    *,
    alpha: float = 0.05,
    rolling_window: int = 252,
    regime_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Backtest conformal VaR coverage with standard exceedance diagnostics.

    An exceedance is a realized return less than or equal to the VaR threshold.
    The function is monitoring-only: it returns machine-readable diagnostics
    and does not prescribe allocation, alerting, or routing decisions.
    """
    returns_arr = np.asarray(returns, dtype=float)
    thresholds_arr = np.asarray(var_thresholds, dtype=float)
    if thresholds_arr.ndim == 0:
        thresholds_arr = np.full_like(returns_arr, float(thresholds_arr), dtype=float)

    if len(returns_arr) != len(thresholds_arr):
        raise ValueError("returns and var_thresholds must have the same length")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    finite_mask = np.isfinite(returns_arr) & np.isfinite(thresholds_arr)
    returns_arr = returns_arr[finite_mask]
    thresholds_arr = thresholds_arr[finite_mask]
    exceedances = returns_arr <= thresholds_arr

    diagnostics = _coverage_summary(
        exceedances,
        alpha=alpha,
        rolling_window=rolling_window,
        include_schema=True,
    )

    if regime_labels is not None and len(regime_labels) == len(finite_mask):
        labels = np.asarray(regime_labels, dtype=object)[finite_mask]
        by_regime: Dict[str, Any] = {}
        for label in dict.fromkeys(str(item) for item in labels):
            regime_mask = labels == label
            by_regime[label] = _coverage_summary(
                exceedances[regime_mask],
                alpha=alpha,
                rolling_window=rolling_window,
                include_schema=False,
            )
        diagnostics["by_regime"] = by_regime

    return diagnostics
