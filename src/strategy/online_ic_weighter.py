"""Online IC-based ensemble weight learning.

Non-ML alternative to the XGBoost stacking integrator. Uses exponential
moving average of per-signal Information Coefficient (IC) values to
dynamically adjust ensemble weights via softmax conversion.

Integrates with ICMonitor for IC tracking. Provides blend_with_static()
for hybrid mode with regime-conditional static weights.

No ML dependencies — pure numpy/stdlib.
"""

import logging
import math
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["OnlineICWeighter"]

# Default parameters
_DEFAULT_HALF_LIFE = 21       # ~1 month of trading days
_DEFAULT_TEMPERATURE = 1.0    # Softmax temperature (lower = more concentrated)
_DEFAULT_MIN_WEIGHT = 0.02    # Exploration floor per signal
_DEFAULT_MAX_WEIGHT = 0.50    # Per-signal cap
_DEFAULT_BLEND_ALPHA = 0.0    # 0=static only, 1=online only
_DEFAULT_TREND_PENALTY = 0.3  # Weight reduction for decaying IC signals


class OnlineICWeighter:
    """Online exponential weighting of signal IC values.

    Tracks per-signal IC via exponential moving average and converts
    to ensemble weights via temperature-scaled softmax.

    Args:
        half_life: EMA half-life in update steps.
        temperature: Softmax temperature (lower = more concentrated).
        min_weight: Minimum weight per signal (exploration floor).
        max_weight: Maximum weight per signal.
        blend_alpha: Blend factor for static weights (0=static, 1=online).
        trend_penalty: Multiplicative penalty for signals with decaying IC trend.
    """

    def __init__(
        self,
        half_life: int = _DEFAULT_HALF_LIFE,
        temperature: float = _DEFAULT_TEMPERATURE,
        min_weight: float = _DEFAULT_MIN_WEIGHT,
        max_weight: float = _DEFAULT_MAX_WEIGHT,
        blend_alpha: float = _DEFAULT_BLEND_ALPHA,
        trend_penalty: float = _DEFAULT_TREND_PENALTY,
    ):
        self.half_life = half_life
        self.temperature = temperature
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.blend_alpha = blend_alpha
        self.trend_penalty = trend_penalty

        # EMA decay factor from half-life: alpha = 1 - exp(-ln2 / half_life)
        self._decay = 1.0 - math.exp(-math.log(2) / half_life)

        # State
        self._ema_values: Dict[str, float] = {}
        self._signal_names: List[str] = []
        self._update_count: int = 0
        self._trends: Dict[str, str] = {}

    def update(self, ic_values: Dict[str, float]) -> None:
        """Update EMA with new IC observations.

        Args:
            ic_values: Dict mapping signal name to IC value.

        Raises:
            ValueError: If ic_values is empty.
        """
        if not ic_values:
            raise ValueError("ic_values is empty.")

        self._update_count += 1

        for name, ic in ic_values.items():
            if name not in self._ema_values:
                self._ema_values[name] = ic
                self._signal_names.append(name)
            else:
                # EMA update: ema = (1-decay)*ema + decay*new_value
                self._ema_values[name] = (1 - self._decay) * self._ema_values[name] + self._decay * ic

    def update_trends(self, trends: Dict[str, str]) -> None:
        """Update IC trend information for penalty application.

        Args:
            trends: Dict mapping signal name to trend string
                ("stable", "improving", "decaying").
        """
        self._trends = dict(trends)

    def get_weights(self) -> Dict[str, float]:
        """Compute normalized ensemble weights from EMA IC values.

        Uses temperature-scaled softmax to convert IC values to weights,
        then applies min/max caps and renormalizes.

        Returns:
            Dict mapping signal name to weight (sums to 1.0).
        """
        if not self._ema_values:
            return {}

        names = list(self._ema_values.keys())
        values = np.array([self._ema_values[n] for n in names])

        # Apply trend penalty for decaying signals
        if self._trends:
            penalties = np.ones(len(names))
            for i, name in enumerate(names):
                if self._trends.get(name) == "decaying":
                    penalties[i] = self.trend_penalty
            values = values * penalties

        # Temperature-scaled softmax
        if self.temperature > 0:
            scaled = values / self.temperature
            # Numerical stability: subtract max
            scaled = scaled - np.max(scaled)
            exp_vals = np.exp(scaled)
            weights = exp_vals / exp_vals.sum()
        else:
            # Zero temperature: winner-take-all
            weights = np.zeros(len(names))
            best = np.argmax(values)
            weights[best] = 1.0

        # Apply min/max caps
        weights = np.clip(weights, self.min_weight, self.max_weight)

        # Renormalize
        total = weights.sum()
        if total > 0:
            weights = weights / total

        return {names[i]: float(weights[i]) for i in range(len(names))}

    def blend_with_static(self, static_weights: Dict[str, float]) -> Dict[str, float]:
        """Blend online weights with static regime-conditional weights.

        Args:
            static_weights: Static weight dict from ensemble config.

        Returns:
            Blended weight dict (sums to 1.0).
        """
        online = self.get_weights()
        all_signals = set(online.keys()) | set(static_weights.keys())

        blended = {}
        for name in all_signals:
            online_w = online.get(name, 0.0)
            static_w = static_weights.get(name, 0.0)
            blended[name] = self.blend_alpha * online_w + (1 - self.blend_alpha) * static_w

        # Renormalize
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        return blended

    def get_weight_vector(self) -> Tuple[np.ndarray, List[str]]:
        """Return weight vector and signal names for array-oriented consumers.

        Returns:
            Tuple of (weight_array, signal_name_list).
        """
        weights = self.get_weights()
        names = list(weights.keys())
        vec = np.array([weights[n] for n in names])
        return vec, names

    def get_state(self) -> Dict:
        """Get serializable state for persistence."""
        return {
            "ema_values": dict(self._ema_values),
            "signal_names": list(self._signal_names),
            "update_count": self._update_count,
            "trends": dict(self._trends),
            "half_life": self.half_life,
            "temperature": self.temperature,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
        }

    def load_state(self, state: Dict) -> None:
        """Restore from serialized state.

        Args:
            state: Dict from get_state().
        """
        self._ema_values = dict(state.get("ema_values", {}))
        self._signal_names = list(state.get("signal_names", []))
        self._update_count = state.get("update_count", 0)
        self._trends = dict(state.get("trends", {}))
