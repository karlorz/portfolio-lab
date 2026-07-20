"""
Per-signal walk-forward validation.

Extends portfolio-level walk-forward validation to individual signals.
For each signal, evaluates whether its directional predictions had
actual predictive power (IC) in out-of-sample windows.

This complements:
- Portfolio-level WFE (scripts/walk_forward_validation.py): validates allocation weights
- IC decay monitor (ic_decay_monitor.py): tracks real-time IC degradation
- This module: validates signal quality over historical windows

Architecture:
  - Expanding window: train on [start, T], test on [T+gap, T+gap+test_size]
  - For each window: compute in-sample and out-of-sample IC
  - Signal WFE = mean(OOS IC) / mean(IS IC) if IS IC > 0
  - Signal DSR = Deflated Sharpe Ratio applied to IC distribution
  - Per-signal results stored for dashboard integration

Usage:
    from src.monitor.signal_walk_forward import SignalWalkForwardValidator

    validator = SignalWalkForwardValidator()
    report = validator.validate_signal("alternative_data")
    # report["wfe"] = 0.92
    # report["mean_oos_ic"] = 0.15
    # report["status"] = "validated"
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.paths import DATA_DIR
from src.monitor.ic_decay_monitor import _spearman_rank_correlation

logger = logging.getLogger(__name__)

__all__ = ["SignalWalkForwardValidator", "compute_signal_wfe_report"]

# Configurable via environment variables
WFV_N_SPLITS = int(os.environ.get("WFV_N_SPLITS", "10"))
WFV_TEST_SIZE = int(os.environ.get("WFV_TEST_SIZE", "126"))  # 6 months
WFV_GAP = int(os.environ.get("WFV_GAP", "21"))  # 1 month embargo
WFV_MIN_IC = float(os.environ.get("WFV_MIN_IC", "0.03"))
WFV_STATE_PATH = DATA_DIR / "signal_wfe_state.json"


@dataclass
class SignalWFEWindow:
    """Single walk-forward window result for a signal."""
    window: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_days: int
    test_days: int
    is_ic: float  # In-sample IC
    oos_ic: float  # Out-of-sample IC
    is_n_obs: int
    oos_n_obs: int


@dataclass
class SignalWFEResult:
    """Walk-forward validation result for a single signal."""
    signal_name: str
    wfe: float  # Walk-Forward Efficiency (OOS IC / IS IC)
    mean_is_ic: float
    mean_oos_ic: float
    std_oos_ic: float
    n_windows: int
    positive_oos_ratio: float  # Fraction of windows with positive OOS IC
    status: str  # "validated", "weak", "unvalidated", "insufficient_data"
    windows: List[SignalWFEWindow] = field(default_factory=list)


class SignalWalkForwardValidator:
    """Per-signal walk-forward validation using expanding windows.

    For each signal, requires a history of (prediction, actual_return) pairs.
    Splits the history into expanding windows and computes in-sample and
    out-of-sample IC for each window.

    Attributes:
        n_splits: Number of walk-forward windows.
        test_size: Test period length in observations.
        gap: Embargo period between train and test.
        min_ic: IC below this is considered noise.
    """

    def __init__(
        self,
        n_splits: int = WFV_N_SPLITS,
        test_size: int = WFV_TEST_SIZE,
        gap: int = WFV_GAP,
        min_ic: float = WFV_MIN_IC,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.min_ic = min_ic

    def validate_signal(
        self,
        signal_name: str,
        predictions: List[float],
        actual_returns: List[float],
    ) -> SignalWFEResult:
        """Run walk-forward validation for a single signal.

        Args:
            signal_name: Name of the signal.
            predictions: Historical signal predictions.
            actual_returns: Corresponding actual forward returns.

        Returns:
            SignalWFEResult with WFE, per-window IC, and status.
        """
        n = len(predictions)
        if n != len(actual_returns):
            raise ValueError(f"predictions ({n}) and actual_returns ({len(actual_returns)}) must have same length")

        # Need enough data for at least 2 windows
        min_data = self.test_size + self.gap + 50
        if n < min_data:
            return SignalWFEResult(
                signal_name=signal_name,
                wfe=0.0, mean_is_ic=0.0, mean_oos_ic=0.0, std_oos_ic=0.0,
                n_windows=0, positive_oos_ratio=0.0,
                status="insufficient_data",
            )

        # Generate expanding-window splits
        windows: List[SignalWFEWindow] = []
        preds_arr = np.array(predictions, dtype=float)
        actuals_arr = np.array(actual_returns, dtype=float)

        # Compute split points
        # First test starts at: n - test_size
        # Each subsequent test starts test_size earlier
        total_test_slots = n - (self.test_size + self.gap + 50)
        if total_test_slots < self.n_splits:
            effective_splits = max(1, total_test_slots // self.test_size)
        else:
            effective_splits = self.n_splits

        step = max(1, total_test_slots // effective_splits)

        for i in range(effective_splits):
            # Test period: from test_start to test_start + test_size
            test_end = n - i * step
            test_start = test_end - self.test_size

            if test_start < 50 + self.gap:
                break

            train_end = test_start - self.gap
            if train_end < 50:
                break

            # Expanding window: train = [0, train_end)
            train_preds = preds_arr[:train_end]
            train_actuals = actuals_arr[:train_end]
            test_preds = preds_arr[test_start:test_end]
            test_actuals = actuals_arr[test_start:test_end]

            is_ic = _spearman_rank_correlation(
                train_preds.tolist(), train_actuals.tolist()
            )
            oos_ic = _spearman_rank_correlation(
                test_preds.tolist(), test_actuals.tolist()
            )

            windows.append(SignalWFEWindow(
                window=i + 1,
                train_start=f"obs_0",
                train_end=f"obs_{train_end}",
                test_start=f"obs_{test_start}",
                test_end=f"obs_{test_end}",
                train_days=train_end,
                test_days=len(test_preds),
                is_ic=is_ic,
                oos_ic=oos_ic,
                is_n_obs=train_end,
                oos_n_obs=len(test_preds),
            ))

        if not windows:
            return SignalWFEResult(
                signal_name=signal_name,
                wfe=0.0, mean_is_ic=0.0, mean_oos_ic=0.0, std_oos_ic=0.0,
                n_windows=0, positive_oos_ratio=0.0,
                status="insufficient_data",
            )

        is_ics = [w.is_ic for w in windows]
        oos_ics = [w.oos_ic for w in windows]

        mean_is_ic = float(np.mean(is_ics))
        mean_oos_ic = float(np.mean(oos_ics))
        std_oos_ic = float(np.std(oos_ics)) if len(oos_ics) > 1 else 0.0

        # Walk-Forward Efficiency
        wfe = mean_oos_ic / mean_is_ic if abs(mean_is_ic) > 1e-10 else 0.0

        # Fraction of windows with positive OOS IC
        positive_oos = sum(1 for ic in oos_ics if ic > self.min_ic)
        positive_oos_ratio = positive_oos / len(oos_ics) if oos_ics else 0.0

        # Status classification
        if mean_oos_ic > self.min_ic and wfe > 0.5:
            status = "validated"
        elif mean_oos_ic > 0 and positive_oos_ratio > 0.5:
            status = "weak"
        else:
            status = "unvalidated"

        return SignalWFEResult(
            signal_name=signal_name,
            wfe=round(wfe, 4),
            mean_is_ic=round(mean_is_ic, 4),
            mean_oos_ic=round(mean_oos_ic, 4),
            std_oos_ic=round(std_oos_ic, 4),
            n_windows=len(windows),
            positive_oos_ratio=round(positive_oos_ratio, 4),
            status=status,
            windows=windows,
        )

    def validate_from_ic_monitor(self, ic_data: Dict[str, List[Tuple[float, float]]]) -> Dict[str, Dict]:
        """Validate multiple signals from ICMonitor data.

        Args:
            ic_data: Dict mapping signal_name -> list of (prediction, actual_return) pairs.

        Returns:
            Dict mapping signal_name -> WFE report dict.
        """
        results = {}
        for signal_name, data in ic_data.items():
            if not data:
                results[signal_name] = {
                    "signal_name": signal_name,
                    "wfe": 0.0, "status": "insufficient_data",
                }
                continue

            predictions = [d[0] for d in data]
            actual_returns = [d[1] for d in data]

            result = self.validate_signal(signal_name, predictions, actual_returns)
            results[signal_name] = {
                "signal_name": result.signal_name,
                "wfe": result.wfe,
                "mean_is_ic": result.mean_is_ic,
                "mean_oos_ic": result.mean_oos_ic,
                "std_oos_ic": result.std_oos_ic,
                "n_windows": result.n_windows,
                "positive_oos_ratio": result.positive_oos_ratio,
                "status": result.status,
            }

        return results

    def save_state(self, results: Dict[str, Dict], path: Optional[Path] = None) -> Path:
        """Save WFE validation results to JSON."""
        if path is None:
            path = WFV_STATE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Signal WFE state saved: %s (%d signals)", path, len(results))
        return path

    def load_state(self, path: Optional[Path] = None) -> Dict[str, Dict]:
        """Load WFE validation results from JSON."""
        if path is None:
            path = WFV_STATE_PATH
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load signal WFE state: %s", e)
            return {}


def compute_signal_wfe_report() -> Dict[str, Any]:
    """Convenience function: compute per-signal WFE report from saved state.

    Creates a SignalWalkForwardValidator, loads any persisted state,
    and returns the WFE report. Used by DashboardGenerator.
    """
    validator = SignalWalkForwardValidator()
    signals = validator.load_state()
    if signals:
        statuses = [row.get("status") for row in signals.values() if isinstance(row, dict)]
        if any(status == "validated" for status in statuses):
            status = "validated"
        elif any(status == "weak" for status in statuses):
            status = "weak"
        elif statuses:
            status = "insufficient_resolved_history"
        else:
            status = "no_data"
        return {
            "status": status,
            "signals": signals,
            "resolved_signal_count": len(signals),
        }

    try:
        from src.monitor.ic_decay_monitor import ICMonitor

        monitor = ICMonitor()
        monitor.load_state()
        pending = monitor.get_staged_prediction_count()
        if pending:
            try:
                from src.monitor.ic_decay_monitor import _signal_prediction_backlog
                backlog = _signal_prediction_backlog()
            except Exception:  # noqa: BLE001
                backlog = {}
            return {
                "status": "waiting_for_forward_returns",
                "signals": {},
                "resolved_signal_count": 0,
                "pending_predictions": pending,
                "pending_rows": backlog.get("pending_rows", 0),
                "pending_dates": backlog.get("pending_dates", 0),
                "oldest_unresolved_date": backlog.get("oldest_unresolved_date"),
                "pending_semantics": backlog.get("pending_semantics"),
                "staged_date": monitor.get_staged_date(),
                "label_horizon": "Uses resolved IC prediction/forward-return pairs; pending until forward returns exist.",
            }
    except (ImportError, RuntimeError, OSError, ValueError, TypeError, json.JSONDecodeError) as e:
        logger.warning("Failed to inspect IC state for WFE pending status: %s", e)

    return {
        "status": "no_data",
        "signals": {},
        "resolved_signal_count": 0,
        "pending_predictions": 0,
    }
