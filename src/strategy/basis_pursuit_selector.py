"""
v8.02: Basis-Pursuit Signal Selection

Prunes redundant and near-zero-weight signals from the ensemble voter pipeline,
applying L1-regularized weight selection with regime-adaptive sparsity.

Based on: Afsharhajari & Li, "The Virtue of Sparsity in Complexity"
(arXiv:2604.17166, Apr 2026).

Key insight: Gains from complexity arise from enlarging the space from which
sparse structure can be identified — not from retaining more factors. Removing
redundant and near-zero signals reduces noise and improves Sharpe.

Integration point: Called BEFORE turnover_validator (v8.01) in the
EnsembleVoter pipeline. Pipeline order:
    health_adjustment → basis_pursuit_selector → regret_weighted_selector → turnover_validator
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.backtest.metrics import save_results_json
from src.paths import DATA_DIR


__all__ = ['DEFAULT_ROLLING_WINDOW', 'LAMBDA_BY_REGIME', 'REDUNDANCY_CORRELATION_THRESHOLD', 'MIN_ACTIVE_WEIGHT', 'SPARSITY_ALERT_THRESHOLD', 'DEFAULT_LAMBDA', 'TOP_PRUNED_REPORT_COUNT', 'PrunedSignal', 'BasisPursuitResult', 'BasisPursuitState', 'BasisPursuitSelector']

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rolling window for signal correlation matrix
DEFAULT_ROLLING_WINDOW = 60

# L1 regularization (lambda) by regime
LAMBDA_BY_REGIME = {
    "normal": 0.01,          # Minimal pruning — keep diverse signals
    "high_vol": 0.05,        # Moderate pruning
    "crisis": 0.15,          # Aggressive pruning — only strongest signals
    "recovery": 0.03,        # Light pruning during recovery
    "unknown_regime": 0.01,  # Conservative — treat unknown as normal (same as DEFAULT_LAMBDA)
}

# Correlation threshold for redundant signal pairs
REDUNDANCY_CORRELATION_THRESHOLD = 0.85

# Minimum weight to consider a signal active
MIN_ACTIVE_WEIGHT = 0.01

# Sparsity alert threshold
SPARSITY_ALERT_THRESHOLD = 0.3

# Default lambda for unlisted regimes (conservative — treat as normal/minimal pruning)
DEFAULT_LAMBDA = 0.01

# State file paths
STATE_FILE = "basis_pursuit_state.json"
PERFORMANCE_FILE = DATA_DIR / "basis_pursuit_performance.json"

# Number of top signals to report
TOP_PRUNED_REPORT_COUNT = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PrunedSignal:
    """Information about a pruned signal."""
    signal: str
    weight: float
    reason: str  # "redundant" or "near_zero"
    paired_with: Optional[str] = None  # For redundant pairs
    correlation: Optional[float] = None  # For redundant pairs


@dataclass
class BasisPursuitResult:
    """Result of basis-pursuit selection."""
    active_signals: Dict[str, float]         # {signal_name: adjusted_weight}
    pruned_signals: Dict[str, float]         # {signal_name: base_weight}
    prune_reasons: Dict[str, PrunedSignal]   # {signal_name: PrunedSignal}
    sparsity_ratio: float                     # active / total
    lambda_used: float                        # L1 penalty used
    regime: str                               # Regime used for lambda selection
    correlation_matrix: Optional[float] = None  # Sparsity ratio of correlation matrix
    num_active: int = 0
    num_pruned: int = 0
    total_signals: int = 0

    def is_concentrated(self) -> bool:
        """True if sparsity ratio is below alert threshold."""
        return self.sparsity_ratio < SPARSITY_ALERT_THRESHOLD


@dataclass
class BasisPursuitState:
    """Persistent state for basis pursuit tracking."""
    signal_history: Dict[str, List[float]] = field(default_factory=dict)
    full_weight_history: Dict[str, List[float]] = field(default_factory=dict)
    rolling_window: int = DEFAULT_ROLLING_WINDOW
    last_regime: str = "normal"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BasisPursuitState":
        return cls(
            signal_history=data.get("signal_history", {}),
            full_weight_history=data.get("full_weight_history", {}),
            rolling_window=data.get("rolling_window", DEFAULT_ROLLING_WINDOW),
            last_regime=data.get("last_regime", "normal"),
        )


# ---------------------------------------------------------------------------
# Core Selector
# ---------------------------------------------------------------------------


class BasisPursuitSelector:
    """
    L1-regularized signal selection for the ensemble voter.

    Removes redundant (highly correlated) and near-zero signals to improve
    signal noise and Sharpe ratio.

    Integration with EnsembleVoter:
        Called during compute_vote() after health-based adjustments, but
        before turnover_validator (v8.01) and regret_weighted_selector (v8.03).
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

    def select_signals(
        self,
        signal_values: Dict[str, float],
        base_weights: Dict[str, float],
        regime: str = "normal",
    ) -> BasisPursuitResult:
        """
        Apply basis-pursuit signal selection.

        Args:
            signal_values: {signal_name: current_signal_value}
            base_weights: {signal_name: base_weight} from regime table
            regime: Current market regime

        Returns:
            BasisPursuitResult with active/pruned signals
        """
        # Update history
        self._update_history(signal_values, base_weights)

        # Update regime
        self.state.last_regime = regime

        # Get lambda for this regime
        lambda_l1 = LAMBDA_BY_REGIME.get(regime, DEFAULT_LAMBDA)

        # Step 1: Identify redundant signal pairs via correlation matrix
        redundant_signals = self._find_redundant_signals()

        # Step 2: Apply L1-regularized weight selection
        selector_result = self._apply_l1_selection(
            base_weights, signal_values, lambda_l1, redundant_signals
        )

        # Step 3: Prune signals below minimum weight
        active, pruned, reasons = self._prune_near_zero(
            selector_result, base_weights, redundant_signals
        )

        # Normalize active weights to sum to 1.0
        total_active = sum(active.values())
        if total_active > 0:
            active = {k: v / total_active for k, v in active.items()}
        elif base_weights:
            # All signals pruned — fall back to normalized base weights
            total_base = sum(base_weights.values())
            active = {k: v / total_base for k, v in base_weights.items()} if total_base > 0 else dict(base_weights)
            pruned = {}
            reasons = {}

        # Compute sparsity ratio
        total_signals = len(base_weights)
        num_active = len(active)
        sparsity_ratio = num_active / max(total_signals, 1)

        result = BasisPursuitResult(
            active_signals=active,
            pruned_signals=pruned,
            prune_reasons=reasons,
            sparsity_ratio=sparsity_ratio,
            lambda_used=lambda_l1,
            regime=regime,
            num_active=num_active,
            num_pruned=len(pruned),
            total_signals=total_signals,
        )

        # Log results
        self._log_selection(result)

        # Persist state
        self._save_state()

        # Track performance
        self._track_performance(result)

        return result

    def get_active_weights(
        self,
        base_weights: Dict[str, float],
        signal_values: Dict[str, float],
        regime: str = "normal",
    ) -> Dict[str, float]:
        """
        Convenience method returning only the active (post-pruning) weights.

        Args:
            base_weights: {signal_name: base_weight}
            signal_values: {signal_name: signal_value}
            regime: Current market regime

        Returns:
            {signal_name: adjusted_weight} after pruning
        """
        result = self.select_signals(signal_values, base_weights, regime)
        return result.active_signals

    def get_state_diagnostics(self) -> Dict[str, dict]:
        """Return diagnostic info about tracked signals."""
        diag = {}
        for source in set(self.state.signal_history) | set(self.state.full_weight_history):
            info = {}
            if source in self.state.signal_history:
                sh = self.state.signal_history[source]
                info["signal_periods"] = len(sh)
                if sh:
                    info["signal_mean"] = float(np.mean(sh))
                    info["signal_std"] = float(np.std(sh, ddof=1)) if len(sh) > 1 else 0.0
            if source in self.state.full_weight_history:
                wh = self.state.full_weight_history[source]
                info["weight_periods"] = len(wh)
                if wh:
                    info["weight_mean"] = float(np.mean(wh))
            diag[source] = info
        return diag

    # ------------------------------------------------------------------
    # Internal: Correlation-based redundancy detection
    # ------------------------------------------------------------------

    def _find_redundant_signals(self) -> Dict[str, str]:
        """
        Find redundant signal pairs (|correlation| > threshold).

        Returns:
            Dict mapping redundant signal → retained signal
            (The one with lower marginal contribution is mapped to the one retained)
        """
        signals = list(self.state.signal_history)
        if len(signals) < 2:
            return {}

        # Build correlation matrix from rolling history
        min_periods = min(len(self.state.signal_history[s]) for s in signals)
        if min_periods < 3:
            return {}

        # Align histories to same length
        aligned = {}
        for s in signals:
            hist = self.state.signal_history[s]
            aligned[s] = np.array(hist[-min_periods:], dtype=float)

        # Compute pairwise correlations
        redundant = {}
        for i in range(len(signals)):
            for j in range(i + 1, len(signals)):
                s1, s2 = signals[i], signals[j]
                corr = self._safe_corr(aligned[s1], aligned[s2])
                if corr is not None and abs(corr) > REDUNDANCY_CORRELATION_THRESHOLD:
                    # Keep the signal with higher mean (as proxy for marginal contribution)
                    mean1 = float(np.mean(np.abs(aligned[s1])))
                    mean2 = float(np.mean(np.abs(aligned[s2])))
                    if mean2 >= mean1:
                        redundant[s1] = s2  # s1 is redundant; keep s2
                    else:
                        redundant[s2] = s1  # s2 is redundant; keep s1

        logger.debug(
            f"Basis pursuit: found {len(redundant)} redundant signals "
            f"(threshold={REDUNDANCY_CORRELATION_THRESHOLD})"
        )
        return redundant

    @staticmethod
    def _safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        """Compute Pearson correlation with protection against constant arrays."""
        std_a = np.std(a, ddof=1)
        std_b = np.std(b, ddof=1)
        if std_a < 1e-10 or std_b < 1e-10:
            return None
        corr_matrix = np.corrcoef(a, b)
        return float(corr_matrix[0, 1])

    # ------------------------------------------------------------------
    # Internal: L1-regularized weight selection
    # ------------------------------------------------------------------

    def _apply_l1_selection(
        self,
        base_weights: Dict[str, float],
        signal_values: Dict[str, float],
        lambda_l1: float,
        redundant_signals: Dict[str, str],
    ) -> Dict[str, float]:
        """
        Apply soft L1 regularization to weights via soft-thresholding.

        Solves: min_w ||target - w·signals||² + λ||w||₁

        We approximate this with iterative soft-thresholding:
            w_i_new = soft_threshold(w_i * signal_i_contribution, λ)

        Where soft_threshold(x, t) = sign(x) * max(|x| - t, 0)

        Returns adjusted weights (pre-normalization).
        """
        if not base_weights:
            return {}

        # Compute signal contributions (how much each signal "earns" its weight)
        contributions = {}
        for signal in base_weights:
            if signal in signal_values:
                # Contribution = base_weight * absolute_signal_value
                contributions[signal] = base_weights[signal] * abs(signal_values[signal])
            else:
                contributions[signal] = base_weights[signal]

        # Apply soft-thresholding
        adjusted = {}
        for signal, contribution in contributions.items():
            # Soft-threshold: shrink small contributions toward zero
            if contribution > lambda_l1:
                # Soft-threshold: sign(x) * max(|x| - λ, 0)
                thresholded = np.sign(contribution) * max(abs(contribution) - lambda_l1, 0)
            else:
                thresholded = 0.0

            adjusted[signal] = float(thresholded)

        # Mark redundant signals for pruning (already identified via correlation)
        for redundant_signal in redundant_signals:
            if redundant_signal in adjusted:
                adjusted[redundant_signal] = 0.0

        return adjusted

    @staticmethod
    def _prune_near_zero(
        l1_weights: Dict[str, float],
        original_weights: Dict[str, float],
        redundant_signals: Dict[str, str],
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, PrunedSignal]]:
        """
        Prune signals with near-zero weights after L1 selection.

        Returns:
            (active_signals, pruned_signals, prune_reasons)
        """
        active = {}
        pruned = {}
        reasons = {}

        for signal, weight in l1_weights.items():
            original = original_weights.get(signal, 0.0)

            # Check if pruned as redundant
            if signal in redundant_signals:
                kept_signal = redundant_signals[signal]
                pruned[signal] = original
                # We need the correlation — approximate from existence
                reasons[signal] = PrunedSignal(
                    signal=signal,
                    weight=original,
                    reason="redundant",
                    paired_with=kept_signal,
                    correlation=REDUNDANCY_CORRELATION_THRESHOLD,
                )
                continue

            # Check if near-zero
            if abs(weight) < MIN_ACTIVE_WEIGHT:
                pruned[signal] = original
                reasons[signal] = PrunedSignal(
                    signal=signal,
                    weight=original,
                    reason="near_zero",
                )
            else:
                active[signal] = weight

        return active, pruned, reasons

    # ------------------------------------------------------------------
    # Internal: History management
    # ------------------------------------------------------------------

    def _update_history(
        self,
        signal_values: Dict[str, float],
        base_weights: Dict[str, float],
    ) -> None:
        """Update rolling signal history."""
        for signal, value in signal_values.items():
            if signal not in self.state.signal_history:
                self.state.signal_history[signal] = []
            history = self.state.signal_history[signal]
            history.append(value)
            if len(history) > self.rolling_window:
                self.state.signal_history[signal] = history[-self.rolling_window:]

        for signal, weight in base_weights.items():
            if signal not in self.state.full_weight_history:
                self.state.full_weight_history[signal] = []
            wh = self.state.full_weight_history[signal]
            wh.append(weight)
            if len(wh) > self.rolling_window:
                self.state.full_weight_history[signal] = wh[-self.rolling_window:]

    # ------------------------------------------------------------------
    # Internal: Logging and performance tracking
    # ------------------------------------------------------------------

    def _log_selection(self, result: BasisPursuitResult) -> None:
        """Log selection results."""
        logger.info(
            f"Basis pursuit ({result.regime}, λ={result.lambda_used}): "
            f"{result.num_active}/{result.total_signals} active "
            f"(sparsity={result.sparsity_ratio:.2f})"
        )

        if result.is_concentrated():
            logger.warning(
                f"Signal concentration warning: sparsity={result.sparsity_ratio:.2f}"
            )

        if result.prune_reasons:
            # Report top pruned signals
            sorted_pruned = sorted(
                result.prune_reasons.values(),
                key=lambda p: p.weight,
                reverse=True,
            )
            for p in sorted_pruned[:TOP_PRUNED_REPORT_COUNT]:
                if p.reason == "redundant":
                    logger.info(
                        f"Basis pursuit pruned '{p.signal}' (redundant with "
                        f"'{p.paired_with}', weight={p.weight:.4f})"
                    )
                else:
                    logger.info(
                        f"Basis pursuit pruned '{p.signal}' "
                        f"(near-zero weight={p.weight:.4f})"
                    )

    def _track_performance(self, result: BasisPursuitResult) -> None:
        """Track sparsity ratio over time for performance analysis."""
        perf_path = self._resolve_perf_path()
        try:
            perf_data = []
            if perf_path.exists():
                with open(perf_path) as f:
                    perf_data = json.load(f)

            # Keep last 100 entries
            perf_data.append({
                "sparsity_ratio": result.sparsity_ratio,
                "num_active": result.num_active,
                "num_pruned": result.num_pruned,
                "total_signals": result.total_signals,
                "regime": result.regime,
                "lambda_used": result.lambda_used,
            })
            if len(perf_data) > 100:
                perf_data = perf_data[-100:]

            save_results_json(perf_data, output_path=str(perf_path))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to track basis pursuit performance: %s", e)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> BasisPursuitState:
        """Load state from JSON file."""
        path = self._resolve_path()
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                return BasisPursuitState.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load basis pursuit state: %s", e)
        return BasisPursuitState(rolling_window=self.rolling_window)

    def _save_state(self) -> None:
        """Save state to JSON file."""
        path = self._resolve_path()
        try:
            save_results_json(self.state.to_dict(), output_path=str(path))
        except OSError as e:
            logger.warning("Failed to save basis pursuit state: %s", e)

    def _resolve_path(self) -> Path:
        """Resolve state path."""
        return Path(str(self.state_path))

    def _resolve_perf_path(self) -> Path:
        """Resolve performance tracking path."""
        return Path(str(PERFORMANCE_FILE))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="v8.02: Basis-Pursuit Signal Selection"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show selector diagnostics")
    status_parser.add_argument("--window", type=int, default=DEFAULT_ROLLING_WINDOW)

    # Select command
    select_parser = subparsers.add_parser("select", help="Apply basis-pursuit selection")
    select_parser.add_argument(
        "--signals",
        nargs="+",
        help="Signal values as name=value pairs (e.g., tsfm_momentum=0.5)",
        required=True,
    )
    select_parser.add_argument(
        "--weights",
        nargs="+",
        help="Base weights as name=value pairs (e.g., tsfm_momentum=0.15)",
        required=True,
    )
    select_parser.add_argument(
        "--regime",
        default="normal",
        choices=["normal", "high_vol", "crisis", "recovery"],
        help="Market regime for lambda selection",
    )

    args = parser.parse_args()

    selector = BasisPursuitSelector()

    if args.command == "status":
        diag = selector.get_state_diagnostics()
        print(f"=== Basis Pursuit Selector Status ===")
        print(f"Tracked signals: {len(diag)}")
        print(f"")
        if diag:
            for source, info in sorted(diag.items()):
                print(f"  {source:30s}: signal_periods={info.get('signal_periods', 0):3d}, "
                      f"signal_mean={info.get('signal_mean', 0):+.3f}, "
                      f"signal_std={info.get('signal_std', 0):.3f}")
        else:
            print("  No signal history yet. Run 'select' to populate.")

    elif args.command == "select":
        # Parse signals
        signal_values = {}
        for s in args.signals:
            if "=" not in s:
                print(f"WARN: Skipping malformed signal: {s}")
                continue
            parts = s.split("=", 1)
            signal_values[parts[0]] = float(parts[1])

        # Parse weights
        base_weights = {}
        for w in args.weights:
            if "=" not in w:
                print(f"WARN: Skipping malformed weight: {w}")
                continue
            parts = w.split("=", 1)
            base_weights[parts[0]] = float(parts[1])

        if not base_weights:
            print("ERROR: No weights provided")
            return

        result = selector.select_signals(signal_values, base_weights, args.regime)

        print(f"=== Basis Pursuit Selection Result ===")
        print(f"Regime: {result.regime} | Lambda: {result.lambda_used}")
        print(f"Sparsity: {result.sparsity_ratio:.2f} ({result.num_active}/{result.total_signals})")
        print(f"")
        print("Active signals:")
        for signal, weight in sorted(result.active_signals.items(), key=lambda x: x[1], reverse=True):
            print(f"  + {signal:30s}: {weight:.4f}")
        if result.pruned_signals:
            print(f"")
            print("Pruned signals:")
            for signal, weight in sorted(result.pruned_signals.items(), key=lambda x: x[1], reverse=True):
                reason = result.prune_reasons.get(signal)
                if reason and reason.reason == "redundant":
                    extra = f" (redundant with {reason.paired_with})"
                else:
                    extra = " (near-zero)"
                print(f"  - {signal:30s}: {weight:.4f}{extra}")
        if result.is_concentrated():
            print(f"\n⚠  WARNING: Signal concentration (sparsity < {SPARSITY_ALERT_THRESHOLD})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
