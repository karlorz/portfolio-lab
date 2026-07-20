"""Statistical Process Control (SPC) for signal quality monitoring.

Applies Shewhart control charts to signal output streams. Maintains a
reference baseline of mean/std computed from a stable period. When a
signal's value breaches 3-sigma limits (computed from the reference)
for 3+ consecutive periods, the signal is flagged as potentially degraded.

The reference baseline is frozen during breach periods and only updated
when the signal returns to normal for a full window, preventing outliers
from inflating the control limits.

SPC catches signal drift before it impacts P&L, complementing the
staleness detection (which catches timing issues) and alerting
(which handles threshold breaches).

Usage:
    monitor = SPCMonitor()
    monitor.record("alternative_data", 0.35)
    monitor.record("alternative_data", 0.38)
    flags = monitor.check_flags()  # Returns list of flagged signals
"""

import json
import math
import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)

# Configurable via environment variables
SPC_WINDOW_SIZE = int(os.environ.get("SPC_WINDOW_SIZE", "60"))
SPC_SIGMA_THRESHOLD = float(os.environ.get("SPC_SIGMA_THRESHOLD", "3.0"))
SPC_CONSECUTIVE_BREACH_LIMIT = int(os.environ.get("SPC_CONSECUTIVE_BREACH_LIMIT", "3"))


class SPCMonitor:
    """Shewhart control chart monitor for signal quality.

    For each tracked signal, maintains a rolling window of recent values
    and a reference baseline (mean, std) for control limit computation.
    The reference baseline is frozen during breach periods and updated
    when the signal returns to normal for a full window, preventing
    outliers from inflating the control limits.

    When a signal's value breaches N-sigma limits for M+ consecutive
    observations, it is flagged as potentially degraded.
    """

    def __init__(
        self,
        window_size: int = SPC_WINDOW_SIZE,
        sigma_threshold: float = SPC_SIGMA_THRESHOLD,
        consecutive_breach_limit: int = SPC_CONSECUTIVE_BREACH_LIMIT,
    ):
        self.window_size = window_size
        self.sigma_threshold = sigma_threshold
        self.consecutive_breach_limit = consecutive_breach_limit

        # Per-signal state: {signal_name: deque of recent values}
        self._windows: Dict[str, deque] = {}
        # Per-signal consecutive breach count
        self._breach_counts: Dict[str, int] = {}
        # Per-signal reference baseline (frozen during breaches)
        self._reference: Dict[str, Optional[Tuple[float, float]]] = {}
        # Per-signal last computed control limits (for reporting)
        self._limits: Dict[str, Dict] = {}

    def record(self, signal_name: str, value: float) -> None:
        """Record a signal value for SPC tracking.

        The reference baseline is used to compute control limits. If no
        reference exists yet, one is computed from the current window
        (once enough samples are available). During breach periods, the
        reference is frozen and not updated.

        Args:
            signal_name: Name of the signal (e.g., "alternative_data")
            value: Current signal value
        """
        if signal_name not in self._windows:
            self._windows[signal_name] = deque(maxlen=self.window_size)
            self._breach_counts[signal_name] = 0
            self._reference[signal_name] = None

        self._windows[signal_name].append(value)

        # Try to establish reference if we don't have one
        ref = self._reference[signal_name]
        if ref is None:
            ref = self._compute_stats(signal_name)
            if ref is not None:
                self._reference[signal_name] = ref

        # Compute control limits from reference
        if ref is not None:
            mean, std = ref
            # Zero-variance honesty: UCL==LCL==mean is not a valid Shewhart chart.
            # Mark limits unavailable and skip breach counting until std > 0.
            if std <= 0:
                self._limits[signal_name] = {
                    "mean": round(mean, 6),
                    "std": 0.0,
                    "ucl": None,
                    "lcl": None,
                    "limits_status": "unavailable_zero_variance",
                    "limits_reason": (
                        "reference std is 0; control limits not defined "
                        "(would collapse to mean and look like calibrated SPC)"
                    ),
                }
                # Do not accumulate breach counts on undefined limits
                self._breach_counts[signal_name] = 0
            else:
                ucl = mean + self.sigma_threshold * std
                lcl = mean - self.sigma_threshold * std
                self._limits[signal_name] = {
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "ucl": round(ucl, 6),
                    "lcl": round(lcl, 6),
                    "limits_status": "ok",
                }
                is_breach = value > ucl or value < lcl
                if is_breach:
                    self._breach_counts[signal_name] += 1
                else:
                    # Reset breach count and refresh reference from current window
                    self._breach_counts[signal_name] = 0
                    self._reference[signal_name] = self._compute_stats(signal_name)

    def check_flags(self) -> List[Dict]:
        """Return list of signals flagged for potential degradation.

        A signal is flagged when its consecutive breach count exceeds
        the configured limit.

        Returns:
            List of dicts with keys: signal, consecutive_breaches,
            mean, std, ucl, lcl, last_value
        """
        flagged = []
        for signal_name, count in self._breach_counts.items():
            if count >= self.consecutive_breach_limit:
                window = self._windows.get(signal_name, deque())
                last_value = window[-1] if window else None
                limits = self._limits.get(signal_name, {})
                flagged.append({
                    "signal": signal_name,
                    "consecutive_breaches": count,
                    "breach_threshold": self.consecutive_breach_limit,
                    "mean": limits.get("mean"),
                    "std": limits.get("std"),
                    "ucl": limits.get("ucl"),
                    "lcl": limits.get("lcl"),
                    "last_value": round(last_value, 6) if last_value is not None else None,
                })
        return flagged

    def get_signal_status(self, signal_name: str) -> Optional[Dict]:
        """Get SPC status for a specific signal.

        Returns:
            Dict with stats and flag status, or None if signal not tracked.
        """
        if signal_name not in self._windows:
            return None

        window = self._windows[signal_name]
        limits = self._limits.get(signal_name, {})
        count = self._breach_counts.get(signal_name, 0)

        status = {
            "signal": signal_name,
            "sample_count": len(window),
            "consecutive_breaches": count,
            "is_flagged": count >= self.consecutive_breach_limit,
            "mean": limits.get("mean"),
            "std": limits.get("std"),
            "ucl": limits.get("ucl"),
            "lcl": limits.get("lcl"),
            "last_value": round(window[-1], 6) if window else None,
        }
        if limits.get("limits_status"):
            status["limits_status"] = limits.get("limits_status")
        if limits.get("limits_reason"):
            status["limits_reason"] = limits.get("limits_reason")
        return status

    def get_all_status(self) -> Dict[str, Dict]:
        """Get SPC status for all tracked signals."""
        return {
            name: self.get_signal_status(name)
            for name in self._windows
            if self.get_signal_status(name) is not None
        }

    def _compute_stats(self, signal_name: str) -> Optional[Tuple[float, float]]:
        """Compute rolling mean and standard deviation for a signal.

        Returns None if fewer than 2 observations (cannot compute std).
        """
        window = self._windows.get(signal_name)
        if window is None or len(window) < 2:
            return None

        values = list(window)
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        return (mean, std)

    def reset(self, signal_name: Optional[str] = None) -> None:
        """Reset SPC state for a specific signal or all signals."""
        if signal_name is None:
            self._windows.clear()
            self._breach_counts.clear()
            self._reference.clear()
            self._limits.clear()
        else:
            self._windows.pop(signal_name, None)
            self._breach_counts.pop(signal_name, None)
            self._reference.pop(signal_name, None)
            self._limits.pop(signal_name, None)

    # ── State persistence ──────────────────────────────────────────────

    def save_state(self, path: Optional[Path] = None) -> None:
        """Persist SPC state to a JSON file for cross-process durability.

        Called after each record() cycle so the next process invocation
        can reload the baseline instead of starting from scratch.

        Args:
            path: File path. Defaults to DATA_DIR / "spc_state.json".
        """
        if path is None:
            from src.paths import DATA_DIR
            path = DATA_DIR / "spc_state.json"

        state = {
            "windows": {name: list(vals) for name, vals in self._windows.items()},
            "breach_counts": dict(self._breach_counts),
            "reference": {
                name: list(ref) if ref is not None else None
                for name, ref in self._reference.items()
            },
            "limits": self._limits,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state))
        except OSError as e:
            logger.warning("SPC state save failed: %s", e)

    def load_state(self, path: Optional[Path] = None) -> bool:
        """Restore SPC state from a JSON file.

        Should be called during __init__ or before first record().

        Args:
            path: File path. Defaults to DATA_DIR / "spc_state.json".

        Returns:
            True if state was loaded, False if no saved state exists.
        """
        if path is None:
            from src.paths import DATA_DIR
            path = DATA_DIR / "spc_state.json"

        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("SPC state load failed: %s", e)
            return False

        self._windows = {
            name: deque(vals, maxlen=self.window_size)
            for name, vals in data.get("windows", {}).items()
        }
        self._breach_counts = data.get("breach_counts", {})
        self._reference = {
            name: tuple(ref) if ref is not None else None
            for name, ref in data.get("reference", {}).items()
        }
        self._limits = data.get("limits", {})
        return True
