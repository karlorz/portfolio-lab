#!/usr/bin/env python3
"""
v6.09: Adaptive Ensemble Signal Weighting

Dynamically adjusts signal source weights based on rolling performance attribution
data. Replaces hardcoded regime-dependent weights with a feedback loop where
signal sources that perform well get increased weight and poorly performing
sources get reduced weight.

This module reads attribution data produced by PerformanceAttribution and computes
adjusted weight multipliers for the EnsembleVoter.

Usage:
    python -m src.strategy.adaptive_ensemble_weights update   # Compute updated weights
    python -m src.strategy.adaptive_ensemble_weights show     # Display current weights
    python -m src.strategy.adaptive_ensemble_weights reset    # Reset to baseline
"""

import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from src.backtest.metrics import save_results_json
from src.paths import DATA_DIR, ATTRIBUTION_DIR


__all__ = ['DEFAULT_CONFIG', 'WeightAdjustment', 'AdaptiveWeightsState', 'AdaptiveEnsembleWeights']

logger = logging.getLogger(__name__)

STATE_FILE = DATA_DIR / "adaptive_weights_state.json"

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    "window_days": 60,
    "baseline_sharpe": 0.5,
    "min_multiplier": 0.25,
    "max_multiplier": 2.0,
    "neg_sharpe_penalty": 0.25,
    "no_data_multiplier": 1.0,
    "min_weight": 0.01,
    "max_weight": 0.40,
    "min_weight_active_sources": 0.005,
    "decay_half_life": 30,
    "stale_attribution_days": 7,
    "min_readings_per_source": 20,
}


# ─────────────────────────────────────────────
#  Data Classes
# ─────────────────────────────────────────────


@dataclass
class WeightAdjustment:
    """Record of a single weight adjustment for audit/history."""
    timestamp: str
    regime: str
    source: str
    base_weight: float
    multiplier: float
    adjusted_weight: float
    sharpe_contribution: float
    total_readings: int


@dataclass
class AdaptiveWeightsState:
    """Persistent state for adaptive ensemble weights."""
    timestamp: str
    regime: str
    adjusted_weights: Dict[str, float]
    multipliers: Dict[str, float]
    history: List[Dict]  # WeightAdjustment dicts
    baseline_weights: Dict[str, float]
    config: Dict


# ─────────────────────────────────────────────
#  Core Class
# ─────────────────────────────────────────────


class AdaptiveEnsembleWeights:
    """
    Compute and manage adaptive signal source weights based on performance attribution.

    Algorithm:
    - Load baseline regime weights
    - Read latest attribution data
    - For each source with data: multiplier = clamp(sharpe / baseline_sharpe, min, max)
    - For negative sharpe: multiplier = neg_sharpe_penalty
    - For no data: multiplier = no_data_multiplier
    - Apply multiplier to base_weight, normalize, enforce min/max constraints
    - Persistent state saved to disk for EnsembleVoter to read
    """

    def __init__(self, base_weights: Optional[Dict[str, float]] = None, config: Optional[Dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.base_weights = base_weights or {}
        self.state_file = STATE_FILE

        # Current computed state
        self.adjusted_weights: Dict[str, float] = {}
        self.multipliers: Dict[str, float] = {}
        self.history: List[WeightAdjustment] = []
        self.current_regime: str = "normal"

    # ── Public API ──

    def update_weights(self, attribution_data: Dict, regime: str = "normal") -> Dict[str, float]:
        """
        Compute adjusted weights from attribution data.

        Args:
            attribution_data: Dict from attribution JSON (sources dict with per-source metrics)
            regime: Current market regime for baseline weight selection

        Returns:
            Dict[str, float]: Adjusted weights mapping source_name -> weight
        """
        self.current_regime = regime
        sources = attribution_data.get("sources", {})

        if not sources:
            logger.info("No attribution data available, using base weights unchanged")
            self.adjusted_weights = dict(self.base_weights)
            self.multipliers = {k: 1.0 for k in self.base_weights}
            self._save_state()
            return self.adjusted_weights

        # Get baseline weights for this regime
        baseline = self.base_weights

        # Compute multipliers from attribution
        multipliers = {}
        raw_adjusted = {}

        for source_name, base_weight in baseline.items():
            attribution = sources.get(source_name, {})
            multiplier = self._compute_multiplier(source_name, attribution)
            multipliers[source_name] = multiplier
            raw_adjusted[source_name] = base_weight * multiplier

        # Add any attribution-only sources (that might not be in baseline)
        for source_name, attribution in sources.items():
            if source_name not in raw_adjusted:
                multiplier = self._compute_multiplier(source_name, attribution)
                multipliers[source_name] = multiplier
                # For extra sources, use a small default base weight
                base = attribution.get("avg_weight", 0.01)
                if base <= 0:
                    base = 0.01
                raw_adjusted[source_name] = base * multiplier

        # Apply min weight floor
        for source_name in raw_adjusted:
            raw_adjusted[source_name] = max(
                self.config["min_weight"],
                raw_adjusted[source_name]
            )

        # Normalize to sum = 1.0
        total = sum(raw_adjusted.values())
        if total > 0:
            normalized = {k: v / total for k, v in raw_adjusted.items()}
        else:
            normalized = dict(raw_adjusted)

        # Enforce max weight constraint
        for source_name in normalized:
            normalized[source_name] = min(
                self.config["max_weight"],
                normalized[source_name]
            )

        # Re-normalize after max constraint
        total_after_max = sum(normalized.values())
        if total_after_max > 0:
            normalized = {k: v / total_after_max for k, v in normalized.items()}

        # Enforce min weight again after normalization (with floating-point tolerance)
        any_below_min = True
        max_iterations = 10
        iteration = 0
        while any_below_min and iteration < max_iterations:
            any_below_min = False
            for source_name in normalized:
                if normalized[source_name] < self.config["min_weight"] - 1e-8:
                    normalized[source_name] = self.config["min_weight"]
                    any_below_min = True
            if any_below_min:
                total_final = sum(normalized.values())
                if total_final > 0:
                    normalized = {k: v / total_final for k, v in normalized.items()}
            iteration += 1

        # Round weights to avoid floating-point epsilon issues (e.g., 0.0099999994 instead of 0.01)
        normalized = {k: round(v, 6) for k, v in normalized.items()}

        # Final hard clamp for any remaining sub-min weights
        for source_name in normalized:
            if normalized[source_name] < self.config["min_weight"]:
                normalized[source_name] = self.config["min_weight"]

        # One final re-normalization if any clamping occurred
        total_clamped = sum(normalized.values())
        if abs(total_clamped - 1.0) > 1e-6:
            if total_clamped > 0:
                normalized = {k: v / total_clamped for k, v in normalized.items()}
                # Round again
                normalized = {k: round(v, 6) for k, v in normalized.items()}

        self.adjusted_weights = normalized
        self.multipliers = multipliers

        # Log adjustments
        for source_name in normalized:
            base = baseline.get(source_name, 0)
            adj = normalized[source_name]
            mult = multipliers.get(source_name, 1.0)
            if abs(adj - base) > 0.005:
                logger.info(
                    "Adaptive weight: %25s %.4f → %.4f (×%.2f)",
                    source_name, base, adj, mult,
                )

        # Record adjustment in history
        adjustment = WeightAdjustment(
            timestamp=datetime.now().isoformat(),
            regime=regime,
            source="__ensemble__",
            base_weight=0,
            multiplier=0,
            adjusted_weight=0,
            sharpe_contribution=0,
            total_readings=len(sources),
        )
        self.history.append(adjustment)

        # Save state
        self._save_state()
        return self.adjusted_weights

    def get_adjusted_weights(self) -> Dict[str, float]:
        """Get the latest computed adjusted weights."""
        if not self.adjusted_weights:
            self._load_state()
        return self.adjusted_weights

    def get_multipliers(self) -> Dict[str, float]:
        """Get multipliers for each source."""
        if not self.multipliers:
            self._load_state()
        return self.multipliers

    def reset_to_baseline(self) -> Dict[str, float]:
        """Reset weights to baseline values."""
        self.adjusted_weights = dict(self.base_weights)
        self.multipliers = {k: 1.0 for k in self.base_weights}
        self._save_state()
        logger.info("Reset adaptive weights to baseline")
        return self.adjusted_weights

    # ── Internal ──

    def _compute_multiplier(self, source_name: str, attribution: Dict) -> float:
        """
        Compute weight multiplier from attribution data for a single source.

        Algorithm:
        - Get rolling Sharpe contribution from attribution
        - If Sharpe > 0: multiplier = clamp(sharpe / baseline_sharpe, min_mult, max_mult)
        - If Sharpe < 0: multiplier = neg_sharpe_penalty
        - If no data: multiplier = no_data_multiplier
        - If zero readings: multiplier = 1.0 (preserve base weight)
        """
        total_readings = attribution.get("total_readings", 0)
        sharpe = attribution.get("sharpe_contribution", 0)

        if sharpe is None or (isinstance(sharpe, float) and np.isnan(sharpe)):
            return self.config["no_data_multiplier"]

        if total_readings == 0:
            return self.config["no_data_multiplier"]

        # Check if we have enough data
        if total_readings < self.config["min_readings_per_source"]:
            # Scale multiplier toward 1.0 based on data scarcity
            data_ratio = total_readings / self.config["min_readings_per_source"]
            raw_mult = self._raw_multiplier(sharpe)
            return 1.0 + (raw_mult - 1.0) * data_ratio

        return self._raw_multiplier(sharpe)

    def _raw_multiplier(self, sharpe: float) -> float:
        """Compute raw multiplier from Sharpe value without data-scarcity adjustment."""
        if sharpe < 0:
            return self.config["neg_sharpe_penalty"]

        if sharpe == 0:
            return self.config["no_data_multiplier"]

        mult = sharpe / self.config["baseline_sharpe"]
        return float(np.clip(
            mult,
            self.config["min_multiplier"],
            self.config["max_multiplier"],
        ))

    def _save_state(self):
        """Persist current state to JSON."""
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "regime": self.current_regime,
                "adjusted_weights": self.adjusted_weights,
                "multipliers": self.multipliers,
                "history": [asdict(h) for h in self.history[-50:]],  # Keep last 50
                "baseline_weights": self.base_weights,
                "config": self.config,
            }
            save_results_json(state, output_path=str(self.state_file))
            logger.debug("Saved adaptive weights state to %s", self.state_file)
        except (OSError, TypeError, ValueError) as e:
            logger.warning("Failed to save adaptive weights state: %s", e)

    def _load_state(self) -> bool:
        """Load persisted state from disk."""
        try:
            if not self.state_file.exists():
                return False
            with open(self.state_file) as f:
                state = json.load(f)
            self.adjusted_weights = state.get("adjusted_weights", {})
            self.multipliers = state.get("multipliers", {})
            self.current_regime = state.get("regime", "normal")
            self.history = [WeightAdjustment(**h) for h in state.get("history", [])]
            self.base_weights = state.get("baseline_weights", self.base_weights)
            logger.debug("Loaded adaptive weights state from %s", self.state_file)
            return True
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to load adaptive weights state: %s", e)
            return False

    def get_state_dict(self) -> Dict:
        """Return current state as a dict for dashboard integration."""
        top_changes = []
        baseline = self.base_weights

        if self.adjusted_weights and baseline:
            for source, adj_weight in self.adjusted_weights.items():
                base = baseline.get(source, 0)
                diff = adj_weight - base
                top_changes.append({
                    "source": source,
                    "base_weight": round(base, 4),
                    "adjusted_weight": round(adj_weight, 4),
                    "change": round(diff, 4),
                    "multiplier": round(self.multipliers.get(source, 1.0), 2),
                })
            top_changes.sort(key=lambda x: abs(x["change"]), reverse=True)

        return {
            "available": bool(self.adjusted_weights),
            "timestamp": datetime.now().isoformat(),
            "regime": self.current_regime,
            "num_adjusted_sources": len(self.adjusted_weights),
            "adjusted_weights": self.adjusted_weights,
            "multipliers": self.multipliers,
            "top_changes": top_changes[:10],
            "history_count": len(self.history),
        }


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────


def _get_base_weights_from_voter() -> Dict[str, float]:
    """Try to load Regime weights from EnsembleVoter for a given regime."""
    try:
        from src.strategy.ensemble_voter import REGIME_WEIGHTS, Regime
        weights = REGIME_WEIGHTS.get(Regime.NORMAL, {})
        return {k.value if hasattr(k, 'value') else str(k): v for k, v in weights.items()}
    except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
        logger.warning("Could not load base weights from voter: %s", e)
        return {}


def _load_latest_attribution() -> Optional[Dict]:
    """Load the most recent attribution report JSON."""
    if not ATTRIBUTION_DIR.exists():
        logger.warning("Attribution directory not found: %s", ATTRIBUTION_DIR)
        return None

    files = sorted(ATTRIBUTION_DIR.glob("attribution_*.json"), reverse=True)
    if not files:
        logger.warning("No attribution files found")
        return None

    try:
        with open(files[0]) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load attribution file: %s", e)
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive Ensemble Signal Weighting")
    subparsers = parser.add_subparsers(dest="command")

    # Update command
    update_parser = subparsers.add_parser("update", help="Compute updated weights from latest attribution")
    update_parser.add_argument("--regime", default="normal", help="Current market regime")
    update_parser.add_argument("--attribution-file", help="Specific attribution file path")

    # Show command
    show_parser = subparsers.add_parser("show", help="Display current adaptive weights")
    show_parser.add_argument("--top", type=int, default=20, help="Show top N sources")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset weights to baseline")

    args = parser.parse_args()

    # Get base weights
    base_weights = _get_base_weights_from_voter()

    adaptive = AdaptiveEnsembleWeights(base_weights=base_weights)

    if args.command == "update":
        attribution = _load_latest_attribution()
        if not attribution:
            logger.error("No attribution data available. Run performance_attribution report --save first.")
            sys.exit(1)

        adapted = adaptive.update_weights(attribution, regime=args.regime)

        logger.info(
            "Adaptive ensemble weights (regime=%s, attribution=%s, sources=%d)",
            args.regime, attribution.get("timestamp", "?"), len(adapted),
        )

        # Show top sources by weight
        sorted_sources = sorted(adapted.items(), key=lambda x: x[1], reverse=True)
        logger.info(
            "  %-30s %8s %8s %6s",
            "Source", "Base", "Adj", "Mult",
        )
        logger.info("  " + "-" * 54)
        for source_name, adj_weight in sorted_sources[:20]:
            base = base_weights.get(source_name, 0)
            mult = adaptive.multipliers.get(source_name, 1.0)
            logger.info(
                "  %-30s %8.4f %8.4f %5.2fx",
                source_name, base, adj_weight, mult,
            )

        # Biggest changes
        changes = []
        for source_name, adj_weight in adapted.items():
            base = base_weights.get(source_name, 0)
            changes.append((source_name, adj_weight - base, base, adj_weight))
        changes.sort(key=lambda x: abs(x[1]), reverse=True)

        logger.info("  Biggest weight changes:")
        for source_name, diff, base, adj in changes[:5]:
            arrow = "up" if diff > 0 else "down"
            logger.info(
                "    %s %-25s %.4f -> %.4f (%+.4f)",
                arrow, source_name, base, adj, diff,
            )

        logger.info("  State saved to: %s", STATE_FILE)

    elif args.command == "show":
        loaded = adaptive._load_state()
        if not loaded:
            logger.warning("No adaptive weights state found. Run 'update' first.")
            sys.exit(1)

        logger.info("Adaptive ensemble weights (last updated: %s, regime=%s)", adaptive.state_file, adaptive.current_regime)

        sorted_sources = sorted(
            adaptive.adjusted_weights.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        logger.info("  %-30s %10s %6s", "Source", "Adj Weight", "Mult")
        logger.info("  " + "-" * 48)
        for source_name, adj_weight in sorted_sources[:args.top]:
            mult = adaptive.multipliers.get(source_name, 1.0)
            logger.info("  %-30s %10.4f %5.2fx", source_name, adj_weight, mult)

        logger.info("  Baseline vs Adjusted:")
        for source_name, adj_weight in sorted_sources:
            base = adaptive.base_weights.get(source_name, 0)
            diff = adj_weight - base
            if abs(diff) > 0.001:
                arrow = "up" if diff > 0 else "down"
                logger.info("    %s %-25s %.4f -> %.4f", arrow, source_name, base, adj_weight)

    elif args.command == "reset":
        adapted = adaptive.reset_to_baseline()
        logger.info("Reset adaptive weights to baseline (%d sources, state saved to %s)", len(adapted), STATE_FILE)

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    from src.utils.log_config import configure_logging
    configure_logging()
    exit(main())
