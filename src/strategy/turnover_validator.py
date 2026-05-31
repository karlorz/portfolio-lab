"""
v8.01 Turnover-Aware Ensemble Weight Validation

Validates signal stability and penalizes excessive turnover that erodes alpha.
Based on Wang & Hasuike (2026) "Decision-Induced Ranking" (arXiv:2605.01176).

Key insight: Portfolio decisions can be interpreted as ranking over risk- and
transaction-cost-adjusted marginal scores. Prediction inflation from excessive
signal turnover leads to unnecessary rebalancing costs that swamp signal accuracy.

Integration point: Called during EnsembleVoter.compute_vote() after health-based
adjustments but before weight application.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ROLLING_WINDOW = 20          # periods for stability computation
MAX_TURNOVER_PENALTY = 0.5           # max weight reduction (50%)
MIN_SIGNAL_HISTORY = 5               # minimum history needed for computation
DEFAULT_SIGNAL_COST = 0.0005         # assumed transaction cost per signal unit (5bps, median ETF cost)

from src.backtest.metrics import save_results_json
from src.paths import DATA_DIR, RISK_FREE_RATE

DEFAULT_RISK_FREE_RATE = RISK_FREE_RATE / 100   # annual risk-free rate (paths.py stores as percent)


__all__ = ['DEFAULT_ROLLING_WINDOW', 'MAX_TURNOVER_PENALTY', 'MIN_SIGNAL_HISTORY', 'DEFAULT_SIGNAL_COST', 'DEFAULT_RISK_FREE_RATE', 'SignalTurnoverMetrics', 'TurnoverValidatorState', 'TurnoverValidator']

STATE_FILE = "turnover_validator_state.json"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SignalTurnoverMetrics:
    """Turnover metrics for a single signal source."""
    source: str
    # Stability metrics
    signal_std: float                  # rolling std of signal values
    sign_flip_rate: float              # fraction of periods where sign changes
    magnitude_volatility: float        # avg absolute change per period
    # Computed values
    turnover_penalty: float            # 0.0 (none) to MAX_TURNOVER_PENALTY
    stability_score: float             # 0.0 (unstable) to 1.0 (very stable)
    # Marginal score components
    expected_return: float             # expected signal value (mean)
    risk_cost: float                   # cost from signal uncertainty
    turnover_cost: float               # cost from expected turnover
    marginal_score: float              # expected_return - (risk_cost + turnover_cost)
    # History
    num_periods: int                   # periods tracked
    missing_data: bool = False         # true if insufficient history


@dataclass
class TurnoverValidatorState:
    """Persistent state for turnover tracking."""
    signal_history: Dict[str, List[float]] = field(default_factory=dict)
    rolling_window: int = DEFAULT_ROLLING_WINDOW

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TurnoverValidatorState":
        return cls(
            signal_history=data.get("signal_history", {}),
            rolling_window=data.get("rolling_window", DEFAULT_ROLLING_WINDOW),
        )


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------


class TurnoverValidator:
    """
    Computes turnover penalties for signal sources based on stability analysis.

    Tracks rolling signal history for each source and computes:
    - Signal volatility (std of signal values)
    - Sign-flip frequency (how often direction changes)
    - Magnitude volatility (avg absolute change)
    - Marginal score (expected return minus risk and turnover costs)
    """

    def __init__(
        self,
        state_path: Optional[Path] = None,
        rolling_window: int = DEFAULT_ROLLING_WINDOW,
    ):
        self.state_path = state_path or DATA_DIR / STATE_FILE
        self.rolling_window = rolling_window
        self.state = self._load_state()

    # ------------------------------------------------------------------ 
    # Public API
    # ------------------------------------------------------------------

    def update_and_validate(
        self,
        signal_values: Dict[str, float],
    ) -> Dict[str, SignalTurnoverMetrics]:
        """
        Update signal history and compute turnover metrics for all sources.

        Args:
            signal_values: {source_name: signal_value} for current period

        Returns:
            {source_name: SignalTurnoverMetrics} with turnover adjustments
        """
        # Append new values to history
        for source, value in signal_values.items():
            if source not in self.state.signal_history:
                self.state.signal_history[source] = []
            history = self.state.signal_history[source]
            history.append(value)
            # Trim to rolling window
            if len(history) > self.rolling_window:
                self.state.signal_history[source] = history[-self.rolling_window:]

        # Compute metrics for each source
        results = {}
        for source in signal_values:
            history = self.state.signal_history.get(source, [])
            results[source] = self._compute_metrics(source, history)

        # Persist state
        self._save_state()

        return results

    def get_adjusted_weights(
        self,
        base_weights: Dict[str, float],
        signal_values: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply turnover-aware adjustment to base weights.

        Args:
            base_weights: {source_name: base_weight} from regime table
            signal_values: {source_name: current_signal_value}

        Returns:
            {source_name: adjusted_weight} after turnover penalty and
            marginal score ranking
        """
        # Step 1: Compute turnover metrics
        metrics = self.update_and_validate(signal_values)

        # Step 2: Apply turnover penalty to base weights
        adjusted = {}
        for source, base_weight in base_weights.items():
            if source in metrics:
                m = metrics[source]
                if m.missing_data:
                    # Insufficient history — no adjustment
                    adjusted[source] = base_weight
                else:
                    # Apply turnover penalty
                    adj = base_weight * (1.0 - m.turnover_penalty)
                    # Apply marginal score boost (if positive marginal score)
                    if m.marginal_score > 0:
                        boost = 1.0 + min(m.marginal_score * 0.1, 0.2)  # max 20% boost
                        adj *= boost
                    adjusted[source] = max(adj, 0.0)
            else:
                # No metrics available — use full weight
                adjusted[source] = base_weight

        # Step 3: Rank by marginal score for diagnostic output
        ranked = sorted(
            metrics.items(),
            key=lambda x: x[1].marginal_score,
            reverse=True,
        )
        logger.debug(
            "Turnover-adjusted weights (top 3 by marginal score): "
            + ", ".join(
                f"{s}={adjusted.get(s, 0):.4f} (marginal={m.marginal_score:.4f}, "
                f"penalty={m.turnover_penalty:.2f})"
                for s, m in ranked[:3]
            )
        )

        # Log high-turnover signals
        high_turnover = [
            (s, m) for s, m in metrics.items()
            if m.turnover_penalty > 0.3
        ]
        if high_turnover:
            logger.info(
                "High-turnover signals penalized: "
                + ", ".join(f"{s} (penalty={m.turnover_penalty:.2f})" for s, m in high_turnover)
            )

        return adjusted

    def get_state_diagnostics(self) -> Dict[str, dict]:
        """Return diagnostic info about all tracked signals."""
        diag = {}
        for source, history in self.state.signal_history.items():
            metrics = self._compute_metrics(source, history)
            diag[source] = {
                "periods": metrics.num_periods,
                "mean": float(metrics.expected_return),
                "std": float(metrics.signal_std),
                "sign_flip_rate": float(metrics.sign_flip_rate),
                "mag_vol": float(metrics.magnitude_volatility),
                "turnover_penalty": float(metrics.turnover_penalty),
                "stability_score": float(metrics.stability_score),
                "marginal_score": float(metrics.marginal_score),
            }
        return diag

    # ------------------------------------------------------------------ 
    # Internal computation
    # ------------------------------------------------------------------

    def _compute_metrics(self, source: str, history: List[float]) -> SignalTurnoverMetrics:
        """Compute turnover metrics for a single signal history."""
        num_periods = len(history)

        if num_periods < MIN_SIGNAL_HISTORY:
            return SignalTurnoverMetrics(
                source=source,
                signal_std=0.0,
                sign_flip_rate=0.0,
                magnitude_volatility=0.0,
                turnover_penalty=0.0,
                stability_score=0.5,
                expected_return=float(np.mean(history)) if history else 0.0,
                risk_cost=0.0,
                turnover_cost=0.0,
                marginal_score=float(np.mean(history)) if history else 0.0,
                num_periods=num_periods,
                missing_data=True,
            )

        arr = np.array(history, dtype=float)

        # 1. Signal volatility: std of signal values
        signal_std = float(np.std(arr, ddof=1))

        # 2. Sign-flip rate: fraction of transitions where sign changes
        signs = np.sign(arr)
        sign_flips = np.sum(np.abs(np.diff(signs)) > 0) if len(arr) > 1 else 0
        sign_flip_rate = sign_flips / max(len(arr) - 1, 1)

        # 3. Magnitude volatility: avg absolute change per period
        if len(arr) > 1:
            abs_changes = np.abs(np.diff(arr))
            magnitude_volatility = float(np.mean(abs_changes))
        else:
            magnitude_volatility = 0.0

        # 4. Stability score: 0.0 (unstable) to 1.0 (very stable)
        #    Combines inverse of std, sign-flip rate, and magnitude vol
        #    Normalize: lower is more stable
        std_component = 1.0 - min(signal_std * 2.0, 1.0)  # std near 0 = stable
        flip_component = 1.0 - sign_flip_rate  # no flips = stable
        mag_component = 1.0 - min(magnitude_volatility * 3.0, 1.0)  # small changes = stable
        stability_score = float(np.clip(
            (std_component * 0.35 + flip_component * 0.40 + mag_component * 0.25),
            0.0, 1.0
        ))

        # 5. Turnover penalty: 0.0 to MAX_TURNOVER_PENALTY
        #    Penalize based on sign-flip rate and magnitude volatility
        turnover_score = (sign_flip_rate * 0.6 + magnitude_volatility * 0.4)
        turnover_penalty = float(min(turnover_score, MAX_TURNOVER_PENALTY))

        # 6. Marginal score components
        expected_return = float(np.mean(arr))
        # Risk cost: signal uncertainty adjusted for investment horizon
        risk_cost = signal_std * np.sqrt(21 / 252)  # monthly risk scaling
        # Turnover cost: expected transactions per period
        annualized_turnover = sign_flip_rate * 252  # expected flips per year
        turnover_cost = annualized_turnover * DEFAULT_SIGNAL_COST
        marginal_score = expected_return - (risk_cost + turnover_cost)

        return SignalTurnoverMetrics(
            source=source,
            signal_std=signal_std,
            sign_flip_rate=sign_flip_rate,
            magnitude_volatility=magnitude_volatility,
            turnover_penalty=turnover_penalty,
            stability_score=stability_score,
            expected_return=expected_return,
            risk_cost=float(risk_cost),
            turnover_cost=float(turnover_cost),
            marginal_score=float(marginal_score),
            num_periods=num_periods,
        )

    # ------------------------------------------------------------------ 
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> TurnoverValidatorState:
        """Load state from JSON file."""
        path = self._resolve_path()
        try:
            exists = path.exists()
        except OSError as e:
            logger.warning("Failed to check turnover validator state path %s: %s", path, e)
            return TurnoverValidatorState(rolling_window=self.rolling_window)

        if exists:
            try:
                with open(path) as f:
                    data = json.load(f)
                return TurnoverValidatorState.from_dict(data)
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning("Failed to load turnover validator state: %s", e)
        return TurnoverValidatorState(rolling_window=self.rolling_window)

    def _save_state(self) -> None:
        """Save state to JSON file."""
        path = self._resolve_path()
        try:
            save_results_json(self.state.to_dict(), output_path=str(path))
        except OSError as e:
            logger.warning("Failed to save turnover validator state: %s", e)

    def _resolve_path(self) -> Path:
        """Resolve state path."""
        return Path(str(self.state_path))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Turnover-Aware Ensemble Weight Validator"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show validator diagnostics")
    status_parser.add_argument("--window", type=int, default=DEFAULT_ROLLING_WINDOW)

    # Adjust command
    adjust_parser = subparsers.add_parser("adjust", help="Apply turnover adjustment")
    adjust_parser.add_argument(
        "--signals",
        nargs="+",
        help="Signal values as name=value pairs (e.g., tsfm_momentum=0.5)",
    )
    adjust_parser.add_argument(
        "--weights",
        nargs="+",
        help="Base weights as name=value pairs (e.g., tsfm_momentum=0.15)",
    )

    args = parser.parse_args()

    validator = TurnoverValidator()

    if args.command == "status":
        diag = validator.get_state_diagnostics()
        logger.info("Turnover Validator Status: %d tracked signals", len(diag))
        if diag:
            for source, info in sorted(diag.items(), key=lambda x: x[1]["turnover_penalty"], reverse=True):
                logger.info(
                    "  %-25s: periods=%3d, mean=%+.3f, std=%.3f, flip_rate=%.2f, penalty=%.2f, stability=%.2f, marginal=%+.4f",
                    source,
                    info['periods'],
                    info['mean'],
                    info['std'],
                    info['sign_flip_rate'],
                    info['turnover_penalty'],
                    info['stability_score'],
                    info['marginal_score'],
                )
        else:
            logger.info("  No signal history yet. Run with signal values to populate.")

    elif args.command == "adjust":
        if not args.signals:
            logger.error("--signals required")
            return

        # Parse signals
        signal_values = {}
        for s in args.signals:
            if "=" not in s:
                logger.warning("Skipping malformed signal: %s", s)
                continue
            parts = s.split("=", 1)
            signal_values[parts[0]] = float(parts[1])

        # Parse weights (or use defaults)
        base_weights = {}
        if args.weights:
            for w in args.weights:
                if "=" in w:
                    parts = w.split("=", 1)
                    base_weights[parts[0]] = float(parts[1])
        else:
            # Default equal weights
            n = len(signal_values)
            if n > 0:
                for src in signal_values:
                    base_weights[src] = 1.0 / n

        if not base_weights:
            logger.error("No weights to adjust")
            return

        adjusted = validator.get_adjusted_weights(base_weights, signal_values)
        metrics = validator.update_and_validate(signal_values)

        logger.info("Turnover-Adjusted Weights:")
        for src in sorted(adjusted.keys()):
            base = base_weights.get(src, 0)
            adj = adjusted.get(src, 0)
            m = metrics.get(src)
            penalty = m.turnover_penalty if m else 0
            stability = m.stability_score if m else 0
            logger.info("  %-25s: %.4f -> %.4f  (penalty=%.2f, stability=%s)", src, base, adj, penalty, stability)

    else:
        parser.print_help()


if __name__ == "__main__":
    from src.utils.log_config import configure_logging
    configure_logging()
    main()
