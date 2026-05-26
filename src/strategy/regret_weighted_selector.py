"""
v8.03: Regret-Weighted Ensemble Signal Selection

Adjusts ensemble signal weights based on the regret contribution of each signal,
defined as the covariance between signal values and the final ensemble decision.

Based on: Aldridge, I. "Regret Equals Covariance" (arXiv:2605.14019, May 2026).

Key insight: Expected regret in stochastic optimization equals Cov(c, π*(c)) —
the covariance between uncertain parameters and optimal decisions. In portfolio
context, each signal source generates an "uncertain parameter" (its allocation
recommendation) and the ensemble produces a "decision" (final weight). Signals
with high regret (their values strongly covary with decisions) contribute more
uncertainty and should have reduced weight.

This is a more principled alternative to v8.01's heuristic turnover penalty —
instead of penalizing turnover directly, we quantify the regret contribution
of each signal via covariance with the ensemble decision.

Pipeline order (in EnsembleVoter.compute_vote()):
    health_adjustment → basis_pursuit_selector (v8.02)
    → regret_weighted_selector (v8.03) → turnover_validator (v8.01)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.backtest.metrics import save_results_json
from src.paths import DATA_DIR


__all__ = ['DEFAULT_ROLLING_WINDOW', 'DEFAULT_REGRET_LAMBDA', 'NUM_ASSETS', 'MIN_COVARIANCE_PERIODS', 'REGRET_LOW_THRESHOLD', 'REGRET_HIGH_THRESHOLD', 'REGRET_MAX_PENALTY', 'SignalRegretMetrics', 'RegretAdjustmentResult', 'RegretWeightedState', 'RegretWeightedSelector', 'apply_regret_adjustment']

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rolling window for signal-decision covariance computation
DEFAULT_ROLLING_WINDOW = 60

# Default regret penalty scaling factor
DEFAULT_REGRET_LAMBDA = 0.3

# Number of assets to track (SPY, GLD, TLT, IEF, SHY, BTC, ETH)
NUM_ASSETS = 7

# Minimum periods needed for covariance computation
MIN_COVARIANCE_PERIODS = 5

# Regret contribution thresholds (normalized 0-1)
REGRET_LOW_THRESHOLD = 0.2    # Below this: low regret, minimal penalty
REGRET_HIGH_THRESHOLD = 0.6   # Above this: high regret, significant penalty
REGRET_MAX_PENALTY = 0.5      # Maximum weight reduction factor

# State file paths
STATE_FILE = "regret_weighted_state.json"
PERFORMANCE_FILE = DATA_DIR / "regret_weighted_performance.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SignalRegretMetrics:
    """Regret metrics for a single signal source."""
    source: str
    # Per-asset covariance
    asset_covariances: Dict[str, float]  # {asset_name: covariance}
    # Aggregate metrics
    regret_contribution: float     # Sum of absolute covariances
    regret_normalized: float       # Normalized to 0-1
    # Adjusted weight factor
    regret_penalty: float          # 0.0 (none) to REGRET_MAX_PENALTY
    # Regime-conditional
    regime_current: str
    # Metadata
    num_periods: int
    missing_data: bool = False


@dataclass
class RegretAdjustmentResult:
    """Result of regret-based weight adjustment."""
    adjusted_weights: Dict[str, float]     # {signal: weight} after regret adjustment
    regret_metrics: Dict[str, SignalRegretMetrics]  # {signal: metrics}
    lambda_used: float
    num_signals: int
    signals_with_high_regret: List[str]    # Signals with regret > REGRET_HIGH_THRESHOLD
    signals_with_low_regret: List[str]     # Signals with regret < REGRET_LOW_THRESHOLD
    avg_regret: float                      # Average normalized regret across signals


@dataclass
class RegretWeightedState:
    """Persistent state for regret-weighted selection."""
    signal_history: Dict[str, List[float]] = field(default_factory=dict)
    decision_history: Dict[str, List[float]] = field(default_factory=dict)
    rolling_window: int = DEFAULT_ROLLING_WINDOW
    last_regime: str = "normal"
    last_ensemble_decision: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RegretWeightedState":
        return cls(
            signal_history=data.get("signal_history", {}),
            decision_history=data.get("decision_history", {}),
            rolling_window=data.get("rolling_window", DEFAULT_ROLLING_WINDOW),
            last_regime=data.get("last_regime", "normal"),
            last_ensemble_decision=data.get("last_ensemble_decision", 0.0),
        )


# ---------------------------------------------------------------------------
# Core Selector
# ---------------------------------------------------------------------------


class RegretWeightedSelector:
    """
    Adjusts ensemble signal weights based on regret (covariance between
    signal values and ensemble decisions).

    Integration with EnsembleVoter:
        Called after basis_pursuit_selector (v8.02) but before
        turnover_validator (v8.01). Applied multiplicatively with
        the turnover penalty.
    """

    def __init__(
        self,
        state_path: Optional[Path] = None,
        rolling_window: int = DEFAULT_ROLLING_WINDOW,
        regret_lambda: float = DEFAULT_REGRET_LAMBDA,
    ):
        self.state_path = state_path or DATA_DIR / STATE_FILE
        self.rolling_window = rolling_window
        self.regret_lambda = regret_lambda
        self.state = self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adjust_weights(
        self,
        signal_values: Dict[str, float],
        ensemble_decision: float,
        current_weights: Dict[str, float],
        regime: str = "normal",
    ) -> RegretAdjustmentResult:
        """
        Apply regret-based weight adjustment.

        Args:
            signal_values: {signal_name: current_signal_value}
            ensemble_decision: The final ensemble consensus value (aggregate)
            current_weights: {signal_name: current_weight} (from previous steps)
            regime: Current market regime

        Returns:
            RegretAdjustmentResult with adjusted weights
        """
        # Update histories
        self._update_history(signal_values, ensemble_decision)

        # Update regime
        self.state.last_regime = regime

        # Compute regret metrics for each signal
        metrics = {}
        for signal in signal_values:
            metrics[signal] = self._compute_regret(
                signal, regime
            )

        # Apply regret-based adjustment
        adjusted_weights = {}
        high_regret_signals = []
        low_regret_signals = []

        for signal, base_weight in current_weights.items():
            if signal in metrics and not metrics[signal].missing_data:
                m = metrics[signal]
                adjusted_weights[signal] = base_weight * (1.0 - m.regret_penalty)

                if m.regret_normalized > REGRET_HIGH_THRESHOLD:
                    high_regret_signals.append(signal)
                elif m.regret_normalized < REGRET_LOW_THRESHOLD:
                    low_regret_signals.append(signal)
            else:
                # No metrics — use full weight
                adjusted_weights[signal] = base_weight

        # Normalize adjusted weights to sum to 1.0
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}

        # Compute average regret
        normalized_regrets = [
            m.regret_normalized
            for m in metrics.values()
            if not m.missing_data
        ]
        avg_regret = float(np.mean(normalized_regrets)) if normalized_regrets else 0.0

        result = RegretAdjustmentResult(
            adjusted_weights=adjusted_weights,
            regret_metrics=metrics,
            lambda_used=self.regret_lambda,
            num_signals=len(metrics),
            signals_with_high_regret=high_regret_signals,
            signals_with_low_regret=low_regret_signals,
            avg_regret=avg_regret,
        )

        # Log results
        self._log_adjustment(result)

        # Persist ensemble decision for next cycle
        self.state.last_ensemble_decision = ensemble_decision

        # Persist state
        self._save_state()

        # Track performance
        self._track_performance(result)

        return result

    def get_adjusted_weights(
        self,
        current_weights: Dict[str, float],
        signal_values: Dict[str, float],
        ensemble_decision: float,
        regime: str = "normal",
    ) -> Dict[str, float]:
        """
        Convenience method returning only the adjusted weights.

        Args:
            current_weights: {signal_name: current_weight}
            signal_values: {signal_name: signal_value}
            ensemble_decision: The ensemble consensus value
            regime: Current market regime

        Returns:
            {signal_name: adjusted_weight} after regret adjustment
        """
        result = self.adjust_weights(
            signal_values, ensemble_decision, current_weights, regime
        )
        return result.adjusted_weights

    def get_state_diagnostics(self) -> Dict[str, dict]:
        """Return diagnostic info about tracked signals."""
        diag = {}
        for source in self.state.signal_history:
            info = {}
            sh = self.state.signal_history[source]
            info["signal_periods"] = len(sh)
            if sh and len(sh) > 0:
                arr = np.array(sh, dtype=float)
                info["signal_mean"] = float(np.mean(arr))
                info["signal_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            diag[source] = info
        return diag

    # ------------------------------------------------------------------
    # Internal: Regret computation
    # ------------------------------------------------------------------

    def _compute_regret(
        self,
        signal: str,
        regime: str,
    ) -> SignalRegretMetrics:
        """
        Compute regret metrics for a single signal.

        Regret = Cov(signal_value, ensemble_decision) for each asset,
        aggregated as sum of absolute covariances.
        """
        signal_history = self.state.signal_history.get(signal, [])
        decision_history = self.state.decision_history.get("ensemble", [])

        num_periods = min(len(signal_history), len(decision_history))

        if num_periods < MIN_COVARIANCE_PERIODS:
            return SignalRegretMetrics(
                source=signal,
                asset_covariances={},
                regret_contribution=0.0,
                regret_normalized=0.0,
                regret_penalty=0.0,
                regime_current=regime,
                num_periods=num_periods,
                missing_data=True,
            )

        # Align histories to same length
        sig_arr = np.array(signal_history[-num_periods:], dtype=float)
        dec_arr = np.array(decision_history[-num_periods:], dtype=float)

        # Compute covariance
        cov_matrix = np.cov(sig_arr, dec_arr, ddof=1)
        covariance = float(cov_matrix[0, 1])

        # For multi-asset: we use the aggregate decision as a single "asset"
        # In a future enhancement, this could be expanded to per-asset covariances
        asset_covariances = {"ensemble": covariance}

        # Regret contribution: absolute covariance
        regret_contribution = abs(covariance)

        # Normalize: compute expected range adaptively from history variance
        sig_var = float(np.var(sig_arr, ddof=1)) if sig_arr.std() > 1e-10 else 1.0
        dec_var = float(np.var(dec_arr, ddof=1)) if dec_arr.std() > 1e-10 else 1.0
        expected_cov_bound = np.sqrt(sig_var * dec_var)

        if expected_cov_bound > 1e-10:
            regret_normalized = min(regret_contribution / expected_cov_bound, 1.0)
        else:
            regret_normalized = 0.0

        # Compute regret penalty
        # Linear interpolation between LOW and HIGH thresholds
        if regret_normalized <= REGRET_LOW_THRESHOLD:
            regret_penalty = 0.0
        elif regret_normalized >= REGRET_HIGH_THRESHOLD:
            # Scale up to max penalty
            # Additional multiplier for crisis/high_vol regimes
            regime_multiplier = self._get_regime_penalty_multiplier(regime)
            regret_penalty = min(
                REGRET_MAX_PENALTY * regime_multiplier,
                REGRET_MAX_PENALTY,
            )
        else:
            # Linear ramp between thresholds: multiply by regret_lambda
            t = (regret_normalized - REGRET_LOW_THRESHOLD) / (
                REGRET_HIGH_THRESHOLD - REGRET_LOW_THRESHOLD
            )
            regret_penalty = t * self.regret_lambda * REGRET_MAX_PENALTY

        # Cap at max
        regret_penalty = min(regret_penalty, REGRET_MAX_PENALTY)

        return SignalRegretMetrics(
            source=signal,
            asset_covariances=asset_covariances,
            regret_contribution=regret_contribution,
            regret_normalized=regret_normalized,
            regret_penalty=regret_penalty,
            regime_current=regime,
            num_periods=num_periods,
        )

    @staticmethod
    def _get_regime_penalty_multiplier(regime: str) -> float:
        """Get penalty multiplier based on regime."""
        multipliers = {
            "normal": 1.0,
            "high_vol": 1.2,
            "crisis": 1.5,
            "recovery": 0.8,
        }
        return multipliers.get(regime, 1.0)

    # ------------------------------------------------------------------
    # Internal: History management
    # ------------------------------------------------------------------

    def _update_history(
        self,
        signal_values: Dict[str, float],
        ensemble_decision: float,
    ) -> None:
        """Update rolling signal and decision histories."""
        for signal, value in signal_values.items():
            if signal not in self.state.signal_history:
                self.state.signal_history[signal] = []
            history = self.state.signal_history[signal]
            history.append(value)
            if len(history) > self.rolling_window:
                self.state.signal_history[signal] = history[-self.rolling_window:]

        # Track ensemble decision (as aggregate)
        if "ensemble" not in self.state.decision_history:
            self.state.decision_history["ensemble"] = []
        dec_history = self.state.decision_history["ensemble"]
        dec_history.append(ensemble_decision)
        if len(dec_history) > self.rolling_window:
            self.state.decision_history["ensemble"] = dec_history[-self.rolling_window:]

    # ------------------------------------------------------------------
    # Internal: Logging and performance tracking
    # ------------------------------------------------------------------

    def _log_adjustment(self, result: RegretAdjustmentResult) -> None:
        """Log adjustment results."""
        logger.info(
            f"Regret-weighted adjustment: {result.num_signals} signals, "
            f"lambda={result.lambda_used}, avg_regret={result.avg_regret:.3f}"
        )

        if result.signals_with_high_regret:
            logger.info(
                f"High-regret signals penalized: "
                f"{', '.join(result.signals_with_high_regret)}"
                f" (regret > {REGRET_HIGH_THRESHOLD})"
            )

        if result.signals_with_low_regret:
            logger.debug(
                f"Low-regret signals (full weight): "
                f"{', '.join(result.signals_with_low_regret)}"
            )

    def _track_performance(self, result: RegretAdjustmentResult) -> None:
        """Track regret metrics over time for performance analysis."""
        perf_path = self._resolve_perf_path()
        try:
            perf_data = []
            if perf_path.exists():
                try:
                    with open(perf_path) as f:
                        perf_data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Corrupted regret performance file %s, resetting (regret-weighted performance)", perf_path)
                    perf_data = []

            # Keep last 100 entries
            perf_data.append({
                "avg_regret": result.avg_regret,
                "num_signals": result.num_signals,
                "num_high_regret": len(result.signals_with_high_regret),
                "num_low_regret": len(result.signals_with_low_regret),
                "lambda_used": result.lambda_used,
                "regime": self.state.last_regime,
            })
            if len(perf_data) > 100:
                perf_data = perf_data[-100:]

            save_results_json(perf_data, output_path=str(perf_path))
        except (OSError, KeyError, ValueError) as e:
            logger.warning("Failed to track regret-weighted performance: %s", e)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> RegretWeightedState:
        """Load state from JSON file."""
        path = self._resolve_path()
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                return RegretWeightedState.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load regret-weighted state: %s", e)
        return RegretWeightedState(rolling_window=self.rolling_window)

    def _save_state(self) -> None:
        """Save state to JSON file."""
        path = self._resolve_path()
        try:
            save_results_json(self.state.to_dict(), output_path=str(path))
        except OSError as e:
            logger.warning("Failed to save regret-weighted state: %s", e)

    def _resolve_path(self) -> Path:
        """Resolve state path."""
        return Path(str(self.state_path))

    def _resolve_perf_path(self) -> Path:
        """Resolve performance tracking path."""
        return Path(str(PERFORMANCE_FILE))


# ---------------------------------------------------------------------------
# Combined transition function: basis_pursuit → regret_weight → turnover
# ---------------------------------------------------------------------------

def apply_regret_adjustment(
    current_weights: Dict[str, float],
    signal_values: Dict[str, float],
    ensemble_decision: float,
    regime: str = "normal",
) -> Dict[str, float]:
    """
    One-shot regret adjustment convenience function.

    Args:
        current_weights: {signal_name: weight} (from basis_pursuit or base)
        signal_values: {signal_name: signal_value}
        ensemble_decision: Final ensemble consensus value
        regime: Current market regime

    Returns:
        {signal_name: adjusted_weight} after regret adjustment
    """
    selector = RegretWeightedSelector()
    return selector.get_adjusted_weights(
        current_weights, signal_values, ensemble_decision, regime
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="v8.03: Regret-Weighted Ensemble Signal Selection"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show regret diagnostics")
    status_parser.add_argument("--window", type=int, default=DEFAULT_ROLLING_WINDOW)

    # Adjust command
    adjust_parser = subparsers.add_parser("adjust", help="Apply regret-based adjustment")
    adjust_parser.add_argument(
        "--signals",
        nargs="+",
        help="Signal values as name=value pairs (e.g., tsfm_momentum=0.5)",
        required=True,
    )
    adjust_parser.add_argument(
        "--weights",
        nargs="+",
        help="Current weights as name=value pairs (e.g., tsfm_momentum=0.25)",
        required=True,
    )
    adjust_parser.add_argument(
        "--decision",
        type=float,
        default=0.0,
        help="Ensemble consensus decision value",
    )
    adjust_parser.add_argument(
        "--regime",
        default="normal",
        choices=["normal", "high_vol", "crisis", "recovery"],
        help="Market regime for penalty multipliers",
    )

    args = parser.parse_args()

    selector = RegretWeightedSelector()

    if args.command == "status":
        diag = selector.get_state_diagnostics()
        logger.info("Regret-Weighted Selector Status: %d tracked signals (window=%d, lambda=%s)",
                     len(diag), selector.rolling_window, selector.regret_lambda)
        if diag:
            for source, info in sorted(diag.items()):
                logger.info(
                    "  %-30s: periods=%3d, mean=%+.3f, std=%.3f",
                    source,
                    info.get("signal_periods", 0),
                    info.get("signal_mean", 0),
                    info.get("signal_std", 0),
                )
        else:
            logger.info("  No signal history yet. Run 'adjust' to populate.")

    elif args.command == "adjust":
        # Parse signals
        signal_values = {}
        for s in args.signals:
            if "=" not in s:
                logger.warning("Skipping malformed signal: %s", s)
                continue
            parts = s.split("=", 1)
            signal_values[parts[0]] = float(parts[1])

        # Parse weights
        current_weights = {}
        for w in args.weights:
            if "=" not in w:
                logger.warning("Skipping malformed weight: %s", w)
                continue
            parts = w.split("=", 1)
            current_weights[parts[0]] = float(parts[1])

        if not current_weights:
            logger.error("No weights provided")
            return

        result = selector.adjust_weights(
            signal_values, args.decision, current_weights, args.regime
        )

        logger.info("Regret-Weighted Adjustment Result (regime=%s, lambda=%s)", args.regime, result.lambda_used)
        logger.info("Signals: %d | Avg regret: %.3f", result.num_signals, result.avg_regret)
        logger.info("Adjusted weights:")
        for signal, weight in sorted(result.adjusted_weights.items(), key=lambda x: x[1], reverse=True):
            metrics = result.regret_metrics.get(signal)
            penalty = metrics.regret_penalty if metrics else 0.0
            regret = metrics.regret_normalized if metrics else 0.0
            logger.info("  %-30s: %.4f  (regret=%.2f, penalty=%.2f)", signal, weight, regret, penalty)
        if result.signals_with_high_regret:
            logger.info("High-regret signals (> %s): %s",
                         REGRET_HIGH_THRESHOLD, ', '.join(result.signals_with_high_regret))

    else:
        parser.print_help()


if __name__ == "__main__":
    from src.utils.log_config import configure_logging
    configure_logging()
    main()
