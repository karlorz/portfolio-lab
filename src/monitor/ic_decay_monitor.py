"""
Information Coefficient (IC) decay monitor for signal quality tracking.

Tracks per-signal IC (rank correlation between signal predictions and
actual forward returns) over rolling windows. Detects when a signal's
predictive power is degrading, which is a leading indicator of alpha
decay that SPC monitoring alone cannot catch.

Complements:
- SPC monitor: catches distribution shifts in signal *values*
- Staleness detection: catches timing/data freshness issues
- This module: catches decay in signal *predictive quality*

Usage:
    monitor = ICMonitor()
    monitor.record("alternative_data", prediction=0.3, actual_return=0.005)
    monitor.record("alternative_data", prediction=-0.1, actual_return=-0.002)
    decay_report = monitor.compute_decay()
    # decay_report["alternative_data"]["ic_rolling"] = 0.82
    # decay_report["alternative_data"]["ic_trend"] = "stable"

Environment variables
---------------------
IC_MONITOR_WINDOW : int
    Rolling window size for IC computation (default: 60)
IC_DECAY_THRESHOLD : float
    IC below this triggers "decaying" status (default: 0.05)
IC_STABLE_MIN : float
    IC above this is considered "stable" (default: 0.10)
IC_TREND_WINDOW : int
    Window for IC trend computation (default: 20)
"""

import json
import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import logging

from src.paths import DATA_DIR

logger = logging.getLogger(__name__)

__all__ = ["ICMonitor", "compute_ic_decay_report"]

# Configurable via environment variables
IC_WINDOW_SIZE = int(os.environ.get("IC_MONITOR_WINDOW", "60"))
IC_DECAY_THRESHOLD = float(os.environ.get("IC_DECAY_THRESHOLD", "0.05"))
IC_STABLE_MIN = float(os.environ.get("IC_STABLE_MIN", "0.10"))
IC_TREND_WINDOW = int(os.environ.get("IC_TREND_WINDOW", "20"))
IC_STATE_PATH = DATA_DIR / "ic_monitor_state.json"


def _spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation between two arrays.

    Returns 0.0 if either array has zero variance or insufficient data.
    """
    if len(x) < 5 or len(y) < 5:
        return 0.0

    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    # Remove NaN/inf
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    if len(x_arr) < 5:
        return 0.0

    # Zero-variance check on original values — argsort assigns unique ranks
    # to identical values, masking the lack of real variation
    if np.ptp(x_arr) < 1e-10 or np.ptp(y_arr) < 1e-10:
        return 0.0

    # Rank the values
    x_rank = np.argsort(np.argsort(x_arr)).astype(float)
    y_rank = np.argsort(np.argsort(y_arr)).astype(float)

    # Pearson correlation of ranks
    n = len(x_rank)
    x_mean = x_rank.mean()
    y_mean = y_rank.mean()

    x_dev = x_rank - x_mean
    y_dev = y_rank - y_mean

    num = (x_dev * y_dev).sum()
    den = np.sqrt((x_dev ** 2).sum() * (y_dev ** 2).sum())

    if den < 1e-10:
        return 0.0

    return float(num / den)


class ICMonitor:
    """Track per-signal Information Coefficient over rolling windows.

    For each signal, stores a rolling window of (prediction, actual_return)
    pairs and computes IC as the Spearman rank correlation between them.
    Tracks IC trend to detect decay.

    Attributes:
        window_size: Number of recent observations to include in IC.
        decay_threshold: IC below this triggers "decaying" status.
        stable_min: IC above this is considered "stable".
    """

    def __init__(
        self,
        window_size: int = IC_WINDOW_SIZE,
        decay_threshold: float = IC_DECAY_THRESHOLD,
        stable_min: float = IC_STABLE_MIN,
        trend_window: int = IC_TREND_WINDOW,
    ):
        self.window_size = window_size
        self.decay_threshold = decay_threshold
        self.stable_min = stable_min
        self.trend_window = trend_window

        # Per-signal data: deque of (prediction, actual_return)
        self._data: Dict[str, deque] = {}

    def record(self, signal_name: str, prediction: float, actual_return: float) -> None:
        """Record a signal prediction and the corresponding actual return.

        Args:
            signal_name: Name of the signal (e.g., "alternative_data").
            prediction: The signal's predicted direction/strength.
            actual_return: The actual forward return that materialized.
        """
        if signal_name not in self._data:
            self._data[signal_name] = deque(maxlen=self.window_size)
        self._data[signal_name].append((prediction, actual_return))

    def compute_ic(self, signal_name: str) -> Optional[float]:
        """Compute rolling IC for a specific signal.

        Returns None if insufficient data points.
        """
        if signal_name not in self._data or len(self._data[signal_name]) < 5:
            return None

        data = list(self._data[signal_name])
        predictions = [d[0] for d in data]
        actuals = [d[1] for d in data]

        return _spearman_rank_correlation(predictions, actuals)

    def compute_ic_trend(self, signal_name: str) -> str:
        """Determine IC trend for a signal.

        Returns one of: "stable", "decaying", "improving", "unknown".
        """
        if signal_name not in self._data:
            return "unknown"

        data = list(self._data[signal_name])
        if len(data) < self.trend_window * 2:
            return "unknown"

        # Split into recent and earlier halves
        n = len(data)
        recent = data[n // 2:]
        earlier = data[:n // 2]

        predictions_recent = [d[0] for d in recent]
        actuals_recent = [d[1] for d in recent]
        predictions_earlier = [d[0] for d in earlier]
        actuals_earlier = [d[1] for d in earlier]

        ic_recent = _spearman_rank_correlation(predictions_recent, actuals_recent)
        ic_earlier = _spearman_rank_correlation(predictions_earlier, actuals_earlier)

        diff = ic_recent - ic_earlier

        if ic_recent < self.decay_threshold:
            return "decaying"
        elif diff > 0.05:
            return "improving"
        elif ic_recent > self.stable_min:
            return "stable"
        else:
            return "decaying"

    def compute_decay_report(self) -> Dict[str, Dict]:
        """Generate a decay report for all tracked signals.

        Returns dict mapping signal_name -> {
            "ic_rolling": float,
            "ic_trend": str,
            "observations": int,
            "status": str,  # "healthy", "warning", "critical"
        }
        """
        report = {}
        for signal_name in self._data:
            ic = self.compute_ic(signal_name)
            trend = self.compute_ic_trend(signal_name)
            n_obs = len(self._data[signal_name])

            if ic is None:
                status = "insufficient_data"
            elif ic < self.decay_threshold:
                status = "critical"
            elif ic < self.stable_min or trend == "decaying":
                status = "warning"
            else:
                status = "healthy"

            report[signal_name] = {
                "ic_rolling": round(ic, 4) if ic is not None else None,
                "ic_trend": trend,
                "observations": n_obs,
                "status": status,
            }

        return report

    def get_signals_needing_attention(self) -> List[str]:
        """Return signal names with 'warning' or 'critical' IC status."""
        report = self.compute_decay_report()
        return [
            name for name, data in report.items()
            if data["status"] in ("warning", "critical")
        ]

    def save_state(self, path: Optional[Path] = None) -> Path:
        """Save current monitor state to JSON for persistence across runs."""
        if path is None:
            path = IC_STATE_PATH
        state = {}
        for signal_name, data in self._data.items():
            state[signal_name] = list(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f)
        logger.info("IC monitor state saved: %s (%d signals)", path, len(state))
        return path

    def load_state(self, path: Optional[Path] = None) -> None:
        """Load monitor state from JSON."""
        if path is None:
            path = IC_STATE_PATH
        if not path.exists():
            return
        try:
            with open(path) as f:
                state = json.load(f)
            for signal_name, observations in state.items():
                self._data[signal_name] = deque(maxlen=self.window_size)
                for pred, actual in observations:
                    self._data[signal_name].append((pred, actual))
            logger.info("IC monitor state loaded: %d signals", len(state))
        except (json.JSONDecodeError, OSError, TypeError) as e:
            logger.warning("Failed to load IC monitor state: %s", e)


def compute_ic_decay_report() -> Dict[str, Dict]:
    """Convenience function: compute IC decay report from saved state.

    Creates an ICMonitor, loads any persisted state, and returns
    the decay report. Used by DashboardGenerator.
    """
    monitor = ICMonitor()
    monitor.load_state()
    return monitor.compute_decay_report()
