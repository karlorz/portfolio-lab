"""Regime transition forecaster.

Computes empirical transition matrices from historical regime sequences,
models regime persistence via exponential survival, and provides
forward-looking regime probability forecasts for portfolio allocation.

Based on Oliveira et al. (2025) framework step 2: forecast distribution
of future regimes. The project's TwoStageKMeansRegime handles step 1
(classification); this module handles step 2.

Regime persistence data from walk-forward validation:
- NORMAL: 7.6 days, CRISIS: 9.9 days, LOW_VOL: 10.0 days
- HIGH_VOL: 7.1 days, RECOVERY: 1.4 days
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "RegimeTransitionForecaster",
    "RegimeForecast",
    "REGIMES",
    "DEFAULT_PERSISTENCE",
]

# Portfolio-lab regime names
REGIMES: List[str] = ["NORMAL", "CRISIS", "LOW_VOL", "HIGH_VOL", "RECOVERY"]

# Default persistence parameters (mean duration in days)
# From regime walk-forward validation (10 expanding windows)
DEFAULT_PERSISTENCE: Dict[str, float] = {
    "NORMAL": 7.6,
    "CRISIS": 9.9,
    "LOW_VOL": 10.0,
    "HIGH_VOL": 7.1,
    "RECOVERY": 1.4,
}

# Smoothing prior for unobserved transitions (Dirichlet alpha)
_SMOOTHING_ALPHA = 0.01

# Horizon at which persistence blending reaches full weight
_BLEND_HORIZON_DAYS = 30


@dataclass
class RegimeForecast:
    """Result of a regime probability forecast.

    Attributes:
        current_regime: The starting regime.
        horizon_days: Number of days ahead for the forecast.
        probabilities: Dict mapping regime name to probability.
        most_likely: The regime with highest probability.
        transition_matrix: The 5×5 empirical transition matrix used.
        persistence_params: Regime persistence parameters (mean duration).
    """

    current_regime: str
    horizon_days: int
    probabilities: Dict[str, float]
    most_likely: str
    transition_matrix: np.ndarray
    persistence_params: Dict[str, float]


class RegimeTransitionForecaster:
    """Computes regime transition probabilities and forecasts.

    Usage:
        forecaster = RegimeTransitionForecaster()
        forecaster.fit(regime_labels)
        forecast = forecaster.forecast("NORMAL", horizon_days=5)
    """

    def __init__(self, persistence_params: Optional[Dict[str, float]] = None):
        """Initialize forecaster.

        Args:
            persistence_params: Override default regime persistence (mean days).
                If None, uses DEFAULT_PERSISTENCE. Can also be auto-computed
                from data via fit().
        """
        self._persistence = dict(persistence_params or DEFAULT_PERSISTENCE)
        self._transition_matrix: Optional[np.ndarray] = None
        self._regime_to_idx: Dict[str, int] = {r: i for i, r in enumerate(REGIMES)}
        self._is_fitted = False
        self._auto_persistence = persistence_params is None

    @property
    def is_fitted(self) -> bool:
        """Whether fit() has been called."""
        return self._is_fitted

    @property
    def transition_matrix(self) -> np.ndarray:
        """The fitted 5×5 transition matrix."""
        if not self._is_fitted:
            raise ValueError("Not fitted. Call fit() first.")
        return self._transition_matrix

    @property
    def persistence_params(self) -> Dict[str, float]:
        """Regime persistence parameters (mean duration in days)."""
        return dict(self._persistence)

    @property
    def persistence_exit_probs(self) -> Dict[str, float]:
        """Daily exit probability per regime (1/mean_duration)."""
        return {r: 1.0 / d for r, d in self._persistence.items()}

    def fit(
        self,
        labels: List,
        regime_map: Optional[Dict] = None,
    ) -> "RegimeTransitionForecaster":
        """Compute empirical transition matrix from a regime label sequence.

        Args:
            labels: Sequence of regime labels (strings or ints).
                If ints, provide regime_map to convert to regime names.
            regime_map: Optional mapping from int labels to regime names.
                E.g., {0: "NORMAL", 1: "CRISIS", ...}

        Returns:
            self (for chaining)

        Raises:
            ValueError: If labels is empty, has < 2 entries, or contains
                unknown regime names.
        """
        if len(labels) == 0:
            raise ValueError("Label sequence is empty.")
        if len(labels) < 2:
            raise ValueError("Need at least 2 labels to compute transitions.")

        # Convert to string regime names
        if regime_map is not None:
            str_labels = []
            for label in labels:
                name = regime_map.get(int(label) if isinstance(label, (int, np.integer)) else label)
                if name is None:
                    raise ValueError(f"Unknown regime label: {label}")
                str_labels.append(name)
        else:
            str_labels = [str(label) for label in labels]

        # Validate and normalize regime names (handle lowercase from classify_vix_regime)
        normalized = []
        for label in str_labels:
            upper = label.upper()
            if upper not in self._regime_to_idx:
                raise ValueError(
                    f"Unknown regime: '{label}'. Valid: {REGIMES}"
                )
            normalized.append(upper)
        str_labels = normalized

        n = len(REGIMES)
        counts = np.zeros((n, n), dtype=np.float64)

        # Count transitions
        for i in range(len(str_labels) - 1):
            from_idx = self._regime_to_idx[str_labels[i]]
            to_idx = self._regime_to_idx[str_labels[i + 1]]
            counts[from_idx, to_idx] += 1

        # Apply Dirichlet smoothing for unobserved transitions
        counts += _SMOOTHING_ALPHA

        # Normalize rows to get probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        self._transition_matrix = counts / row_sums

        # Auto-compute persistence from data if not explicitly provided
        if self._auto_persistence:
            self._compute_persistence_from_data(str_labels)

        self._is_fitted = True
        logger.info(
            "Fitted regime transition forecaster on %d labels, %d transitions",
            len(labels), len(labels) - 1,
        )
        return self

    def forecast(
        self,
        current_regime: str,
        horizon_days: int = 5,
    ) -> RegimeForecast:
        """Forecast regime probabilities over a horizon.

        Uses matrix power: P^horizon gives the n-step transition
        probabilities. Blends with persistence-adjusted probabilities
        for more realistic forecasts.

        Args:
            current_regime: Starting regime name.
            horizon_days: Number of days to forecast.

        Returns:
            RegimeForecast with probability distribution.

        Raises:
            ValueError: If not fitted or current_regime is unknown.
        """
        if not self._is_fitted:
            raise ValueError("Not fitted. Call fit() first.")
        # Normalize case (handle lowercase from classify_vix_regime)
        current_regime = current_regime.upper()
        if current_regime not in self._regime_to_idx:
            raise ValueError(f"Unknown regime: '{current_regime}'")

        n = len(REGIMES)
        idx = self._regime_to_idx[current_regime]

        # Matrix power for n-step transition
        P_n = np.linalg.matrix_power(self._transition_matrix, horizon_days)

        # Raw forecast from matrix power
        raw_probs = P_n[idx]

        # Blend with persistence-adjusted probabilities
        # Short horizons trust matrix power more; long horizons converge
        # to stationary distribution
        persistence_probs = self._persistence_adjusted_probs(current_regime, horizon_days)
        blend_weight = min(1.0, horizon_days / _BLEND_HORIZON_DAYS)
        blended = (1 - blend_weight) * raw_probs + blend_weight * persistence_probs

        # Re-normalize
        blended = blended / blended.sum()

        probabilities = {REGIMES[i]: float(blended[i]) for i in range(n)}
        most_likely = max(probabilities, key=probabilities.get)

        return RegimeForecast(
            current_regime=current_regime,
            horizon_days=horizon_days,
            probabilities=probabilities,
            most_likely=most_likely,
            transition_matrix=self._transition_matrix.copy(),
            persistence_params=dict(self._persistence),
        )

    def get_signal(
        self,
        current_regime: str,
        horizon_days: int = 5,
    ) -> dict:
        """Return a regime transition forecast signal dict.

        Args:
            current_regime: Current regime name.
            horizon_days: Forecast horizon.

        Returns:
            Dict with regime, confidence, forecast_probs, horizon_days.
        """
        forecast = self.forecast(current_regime, horizon_days)
        confidence = forecast.probabilities[current_regime]
        return {
            "regime": current_regime,
            "confidence": round(confidence, 4),
            "forecast_probs": {k: round(v, 4) for k, v in forecast.probabilities.items()},
            "horizon_days": horizon_days,
            "most_likely": forecast.most_likely,
        }

    def _persistence_adjusted_probs(
        self, current_regime: str, horizon_days: int,
    ) -> np.ndarray:
        """Compute persistence-adjusted regime probabilities.

        Uses exponential survival model: P(still in regime at day t) = exp(-t/mean_duration).
        Remaining probability mass is distributed to other regimes proportionally
        to their transition probabilities.

        Args:
            current_regime: Starting regime.
            horizon_days: Days ahead.

        Returns:
            Array of probabilities for each regime.
        """
        n = len(REGIMES)
        idx = self._regime_to_idx[current_regime]
        mean_duration = self._persistence[current_regime]

        # Survival probability (still in current regime)
        survival = np.exp(-horizon_days / mean_duration)

        # Get transition probabilities to other regimes
        trans_row = self._transition_matrix[idx].copy()
        trans_row[idx] = 0  # Exclude self-transition
        other_sum = trans_row.sum()

        # Distribute exit probability to other regimes
        result = np.zeros(n)
        result[idx] = survival
        if other_sum > 0:
            exit_prob = 1.0 - survival
            for i in range(n):
                if i != idx:
                    result[i] = exit_prob * trans_row[i] / other_sum

        # Normalize
        total = result.sum()
        if total > 0:
            result /= total

        return result

    def _compute_persistence_from_data(self, labels: List[str]) -> None:
        """Estimate regime persistence from run lengths in the data.

        Computes the mean consecutive run length for each regime.

        Args:
            labels: Sequence of string regime labels.
        """
        runs: Dict[str, List[int]] = {r: [] for r in REGIMES}
        current = labels[0]
        count = 1

        for i in range(1, len(labels)):
            if labels[i] == current:
                count += 1
            else:
                runs[current].append(count)
                current = labels[i]
                count = 1
        runs[current].append(count)  # Final run

        for regime in REGIMES:
            if runs[regime]:
                mean_run = sum(runs[regime]) / len(runs[regime])
                self._persistence[regime] = mean_run
