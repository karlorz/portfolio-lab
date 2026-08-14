"""EnsembleVoter vote mixin (Item 5 s3 ENSEMBLE-VOTER-MIXINS).
Methods extracted verbatim from src/strategy/ensemble_voter.py.
"""

import logging
from datetime import datetime
from src.paths import ATTRIBUTION_DIR
from src.signals.regime_spec import Regime
from src.signals.regime_spec import SignalReading
from src.signals.signal_source import SignalSource
from src.strategy.ensemble_support import EnsembleVote
from src.strategy.ensemble_support import _get_health_tracker
from src.strategy.ensemble_support import compute_signal_correlation_matrix
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import json
import numpy as np
import os
import random
import sqlite3
logger = logging.getLogger("src.strategy.ensemble_voter")

class VoteMixin:
    def compute_vote(
        self,
        readings: Optional[Dict[SignalSource, SignalReading]] = None,
        regime: Optional[Regime] = None,
        regime_confidence: Optional[float] = None
    ) -> EnsembleVote:
        """Compute ensemble vote with regime-dependent weighting.

        Delegates to sub-methods for each weighting phase:
        1. _resolve_inputs — resolve readings/regime/confidence defaults
        2. _apply_regime_gating — zero out signals net-negative in this regime
        3. _apply_adaptive_weights — attribution-based weight adjustment
        4. _apply_health_weights — reduce weight for poor health scores
        5. _apply_correlation_penalty — reduce weight for redundant signals
        6. _apply_regime_weights — per-regime signal weight multipliers
        7. _apply_utility_reweighting — boost/reduce by profitability (Sharpe contribution + hit rate)
        8. _apply_exploration_noise — epsilon-greedy Dirichlet exploration for weight discovery
        8a. _apply_diversity_floor — minimum weight floor for active signals (N_eff improvement)
        9. _apply_turnover_validation — turnover + basis-pursuit + regret-weighted
       10. _compute_consensus — weighted consensus, agreement, asset biases, action
       11. _persist_vote — save vote and persist regret state
        """
        caller_supplied_readings = readings is not None
        readings, regime, regime_confidence = self._resolve_inputs(
            readings, regime, regime_confidence
        )

        weights = self.get_blended_weights(regime.name)
        weights = self._apply_regime_gating(weights, regime.name, regime_confidence)
        weights = self._apply_adaptive_weights(weights, regime)
        # Batch DK: adaptive may receive non-zero base if blend leaked; re-pin
        weights = self._pin_zero_baseline_weights(weights, regime.name)
        weights = self._apply_ic_weights(weights, regime)
        weights = self._pin_zero_baseline_weights(weights, regime.name)
        weights = self._apply_health_weights(weights)
        # Batch DN: health renorm concentrates mass — enforce documented 50% cap
        weights = self._cap_per_signal_weights(weights, regime.name)
        weights = self._apply_correlation_penalty(weights)
        if os.environ.get("ENSEMBLE_DISABLE_REGIME_WEIGHTS", "").lower() not in ("1", "true"):
            weights = self._apply_regime_weights(weights, regime)
        if os.environ.get("ENSEMBLE_USE_MDP_CONSTRAINT", "").lower() in ("1", "true"):
            weights = self._apply_mdp_constraint(weights)
        weights = self._apply_utility_reweighting(weights, regime)
        weights = self._apply_exploration_noise(weights, regime)
        weights = self._pin_zero_baseline_weights(weights, regime.name)
        weights = self._apply_diversity_floor(weights)
        weights = self._pin_zero_baseline_weights(weights, regime.name)
        weights = self._apply_turnover_validation(weights, readings, regime)
        weights = self._pin_zero_baseline_weights(weights, regime.name)
        # Final cap after any reinflation-capable stages
        weights = self._cap_per_signal_weights(weights, regime.name)

        # Soft-delete static zeros — exclude from analysis equal-weight floors
        soft_delete = self._static_zero_baseline_sources(regime.name)

        # Safety fallback: when explicit readings are provided for analysis/tests,
        # avoid degenerate outcomes where static regime weights entirely mute
        # one or more provided sources. Soft-delete arms stay excluded.
        active_weights = {
            source: max(0.0, weights.get(source, 0.0))
            for source in readings
            if source not in soft_delete
        }
        active_weight_sum = sum(active_weights.values())
        nonzero_sources = [source for source, weight in active_weights.items() if weight > 0]
        floor_eligible = [s for s in readings if s not in soft_delete]

        if readings and active_weight_sum <= 0:
            if caller_supplied_readings and regime == Regime.NORMAL and floor_eligible:
                fallback_weight = 1.0 / len(floor_eligible)
                weights = {
                    source: (fallback_weight if source in floor_eligible else 0.0)
                    for source in readings
                }
                # Preserve any non-reading keys at 0
                for src in soft_delete:
                    weights[src] = 0.0
                logger.info(
                    "All active ensemble weights were zero after adjustments; "
                    "falling back to equal-weight over %d readings (soft-delete pinned)",
                    len(floor_eligible),
                )
            else:
                logger.info(
                    "All active ensemble weights were zero after adjustments; "
                    "preserving zero-weight gating for regime=%s",
                    regime.value if hasattr(regime, "value") else regime,
                )
        elif (
            caller_supplied_readings
            and floor_eligible
            and regime == Regime.NORMAL
            and 0 < len(nonzero_sources) < len(floor_eligible)
        ):
            floor_weight = 0.05 / len(floor_eligible)  # 5% total floor; soft-delete excluded
            blended = {}
            for source in readings:
                if source in soft_delete:
                    blended[source] = 0.0
                else:
                    blended[source] = max(active_weights.get(source, 0.0), floor_weight)
            total = sum(blended.values())
            if total > 0:
                weights = {source: weight / total for source, weight in blended.items()}
                logger.info(
                    "Applied small analysis floor to %d/%d zero-weight provided signals "
                    "(soft-delete arms pinned at 0)",
                    len(floor_eligible) - len(nonzero_sources),
                    len(floor_eligible),
                )
            weights = self._pin_zero_baseline_weights(weights, regime.name)

        # Batch DO: analysis-floor / equal-weight fallback renorm can breach the
        # 50% per-signal cap — re-apply after those paths (and pin soft-delete).
        weights = self._cap_per_signal_weights(weights, regime.name)

        # Batch DP: inactive_signal arms (e.g. intl RS neutral) still held pipeline
        # mass after DN cap; _apply_weights_to_readings zeroed them without renorm,
        # so dashboard active_weights rollup re-concentrated CAR past 50%.
        # Zero inactive mass, renorm, re-cap before assign.
        weights = self._zero_inactive_and_recap(weights, readings, regime.name)

        # Apply weights to readings
        weighted_signals = self._apply_weights_to_readings(readings, weights)

        if not weighted_signals:
            return EnsembleVote(
                timestamp=str(datetime.now()),
                regime=regime,
                regime_confidence=regime_confidence,
                num_sources=0,
                weighted_consensus=0.0,
                agreement_ratio=0.0,
                equity_bias=0.0,
                duration_bias=0.0,
                gold_bias=0.0,
                action="neutral",
                confidence=0.0,
                reasoning="No signals available",
                source_votes=[],
                adaptive_learning=self.get_adaptive_learning_status(regime.name),
            )

        consensus_result = self._compute_consensus(weighted_signals, regime, regime_confidence)
        vote = self._build_vote(weighted_signals, consensus_result, regime, regime_confidence)
        self._persist_vote(vote, consensus_result.weighted_consensus)

        return vote

    def _resolve_inputs(
        self,
        readings: Optional[Dict[SignalSource, SignalReading]],
        regime: Optional[Regime],
        regime_confidence: Optional[float],
    ) -> Tuple[Dict[SignalSource, SignalReading], Regime, float]:
        """Resolve default readings, regime, and confidence."""
        if readings is None:
            if regime is None:
                regime, regime_confidence = self.detect_regime()
            readings = self.current_readings or self.collect_signals(regime=regime)

        if regime is None:
            regime, regime_confidence = self.detect_regime()

        if regime_confidence is None:
            regime_confidence = 0.5

        self.current_regime = regime
        self.current_regime_confidence = regime_confidence
        return readings, regime, regime_confidence

    def _apply_regime_gating(
        self, weights: Dict, regime_name: str, regime_confidence: float = 0.5
    ) -> Dict:
        """Apply regime gating — zero out signals that are net-negative in this regime.
        
        Uses confidence-weighted gating (v3.26) to defer gating when regime confidence is low,
        preventing premature switching on uncertain regime classification.
        """
        # Batch CX: reset disclosure map each vote
        self._regime_gated: Dict[str, str] = {}
        if hasattr(self, 'regime_gate') and self.regime_gate is not None:
            # Get active signals based on confidence and hysteresis
            active_signal_names = self.regime_gate.gate_with_confidence(
                regime_name, 
                regime_confidence
            )
            active_signal_set = set(active_signal_names)
            gate_rules = getattr(self.regime_gate, "gate_rules", None)
            explicit_gate_rules = gate_rules if isinstance(gate_rules, dict) else None
            
            # Zero out signals not in the active list
            gated_weights = {}
            for source, weight in weights.items():
                source_name = source.value if hasattr(source, 'value') else str(source)
                has_explicit_gate = (
                    explicit_gate_rules is None or source_name in explicit_gate_rules
                )
                is_active = source_name in active_signal_set or not has_explicit_gate
                if is_active:
                    gated_weights[source] = weight
                else:
                    gated_weights[source] = 0.0
                    if float(weight or 0.0) > 0.0 or has_explicit_gate:
                        off_regimes = set()
                        if isinstance(explicit_gate_rules, dict):
                            off_regimes = set(explicit_gate_rules.get(source_name) or [])
                        self._regime_gated[source_name] = (
                            f"regime_gate_off({regime_name}"
                            f"{'; off=' + ','.join(sorted(off_regimes)) if off_regimes else ''})"
                        )
            
            total = sum(gated_weights.values())
            if total > 0:
                gated_weights = {k: v / total for k, v in gated_weights.items()}
            
            return gated_weights
        return weights

    def _apply_adaptive_weights(
        self, weights: Dict, regime: Regime
    ) -> Dict:
        """Apply adaptive ensemble weighting (v6.09) if attribution data is fresh enough."""
        try:
            from src.strategy.adaptive_ensemble_weights import AdaptiveEnsembleWeights

            attribution_dir = ATTRIBUTION_DIR
            attribution_files = sorted(attribution_dir.glob("attribution_*.json"), reverse=True)

            if not attribution_files:
                return weights

            with open(attribution_files[0]) as f:
                attribution_data = json.load(f)

            # Check if attribution is stale (>7 days old)
            attr_timestamp = attribution_data.get("timestamp", "")
            if attr_timestamp:
                attr_date = attr_timestamp[:10]
                days_stale = (datetime.now() - datetime.strptime(attr_date, "%Y-%m-%d")).days
            else:
                days_stale = 999

            if days_stale > 7:
                return weights

            # Check if we have enough data points
            sources = attribution_data.get("sources", {})
            total_readings = sum(s.get("total_readings", 0) for s in sources.values())
            num_sources = len(sources)
            avg_readings = total_readings / max(num_sources, 1)

            if avg_readings < 5:
                return weights

            # Build base weights in string-keyed format
            base_str = {k.value: v for k, v in weights.items()}

            adaptive = AdaptiveEnsembleWeights(base_weights=base_str)
            adapted = adaptive.update_weights(attribution_data, regime.value)

            # Convert back to enum-keyed dict
            adaptive_weights_enum = {}
            for source_enum in weights:
                source_str = source_enum.value
                if source_str in adapted:
                    adaptive_weights_enum[source_enum] = adapted[source_str]

            if adaptive_weights_enum:
                logger.info("Using adaptive ensemble weights for regime=%s", regime.value)
                return adaptive_weights_enum
        except (KeyError, ValueError, TypeError, AttributeError, ZeroDivisionError, OSError) as e:
            logger.warning("Could not apply adaptive ensemble weights: %s", e)
        return weights

    def _apply_ic_weights(self, weights: Dict, regime: Regime) -> Dict:
        """Apply IC-based ensemble weight learning (online IC weighter).

        Uses OnlineICWeighter to compute IC-based weights from the ICMonitor
        persisted state, then blends with the current weights. This is gated
        by ENSEMBLE_USE_IC_WEIGHTS env var (default: off).

        The IC weighter:
        1. Loads IC data from ICMonitor persisted state
        2. Computes rolling IC for each signal
        3. Uses EMA with exponential decay to track IC trends
        4. Converts IC values to weights via temperature-scaled softmax
        5. Blends online weights with current static weights

        Expected impact: +0.005-0.01 Sharpe by dynamically reweighting
        signals based on their recent predictive power.
        """
        if not getattr(self, '_use_ic_weights', False):
            self._last_online_ic_learning_status = {
                "status": "disabled",
                "enabled": False,
                "state_available": False,
                "reason": "env_disabled",
            }
            return weights

        if getattr(self, '_ic_weighter', None) is None:
            self._last_online_ic_learning_status = {
                "status": "unavailable",
                "enabled": True,
                "state_available": False,
                "reason": "initialization_failed_or_unavailable",
            }
            return weights

        try:
            from src.monitor.ic_decay_monitor import ICMonitor

            # Load IC monitor state
            monitor = ICMonitor()
            monitor.load_state()

            # Get IC values and trends for each signal
            ic_values: Dict[str, float] = {}
            ic_trends: Dict[str, str] = {}

            for source_enum in weights:
                source_str = source_enum.value
                ic = monitor.compute_ic(source_str)
                if ic is not None and np.isfinite(ic):
                    ic_values[source_str] = ic
                    trend = monitor.compute_ic_trend(source_str)
                    ic_trends[source_str] = trend

            if not ic_values:
                logger.debug("No IC data available for online weight learning")
                self._last_online_ic_learning_status = {
                    "status": "non_effective",
                    "enabled": True,
                    "state_available": True,
                    "reason": "no_ic_data_available",
                }
                return weights

            # Update the OnlineICWeighter with current IC values and trends
            self._ic_weighter.update(ic_values)
            self._ic_weighter.update_trends(ic_trends)

            # Get IC-based weights (raw)
            ic_weights = self._ic_weighter.get_weights()

            if not ic_weights:
                self._last_online_ic_learning_status = {
                    "status": "non_effective",
                    "enabled": True,
                    "state_available": True,
                    "reason": "no_ic_weights_available",
                }
                return weights

            # Convert weights to string format for blending
            current_weights_str = {k.value: v for k, v in weights.items()}

            # Blend IC-based weights with current weights
            # blend_alpha controls how much we trust IC-based weights (0=static, 1=online)
            # Start conservative: 30% IC-based, 70% current
            blend_alpha = float(os.environ.get("ENSEMBLE_IC_WEIGHT_BLEND_ALPHA", "0.3"))
            blended = {}

            for sig_name in current_weights_str:
                ic_w = ic_weights.get(sig_name, 0.0)
                current_w = current_weights_str[sig_name]
                blended[sig_name] = (1.0 - blend_alpha) * current_w + blend_alpha * ic_w

            # Renormalize
            total = sum(blended.values())
            if total > 0:
                blended = {k: v / total for k, v in blended.items()}

            # Convert back to enum-keyed dict
            ic_adjusted = {}
            for source_enum in weights:
                source_str = source_enum.value
                if source_str in blended:
                    ic_adjusted[source_enum] = blended[source_str]

            if ic_adjusted:
                logger.info(
                    "Online IC weights applied (blend_alpha=%.2f): %s",
                    blend_alpha,
                    ', '.join(
                        f'{k.value}={v:.3f}'
                        for k, v in ic_adjusted.items() if v > 0.01
                    )
                )
                self._last_online_ic_learning_status = {
                    "status": "active",
                    "enabled": True,
                    "state_available": True,
                    "reason": "blending_with_static_weights",
                }
                return ic_adjusted

        except (ImportError, KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.warning("Could not apply IC-based weights: %s", e)
            self._last_online_ic_learning_status = {
                "status": "unavailable",
                "enabled": True,
                "state_available": getattr(self, "_ic_weighter", None) is not None,
                "reason": f"ic_weight_application_failed:{type(e).__name__}",
            }

        return weights

    def _apply_health_weights(self, weights: Dict) -> Dict:
        """Apply health-adjusted weighting (v3.12) — soft floor + hard-zero gates.

        ADR-006 residual honesty:
        - require the configured minimum labeled daily cohorts before hard-zero
        - hard-zero non-healthy sleeves with negative IC
        - hard-zero unhealthy sleeves with unknown IC or IC below the approved
          minimum (0.08 by default)
        - otherwise soft floor max(0.2, health_score) for graceful degrade
        - if all arms hard-gated: freeze adaptive blend (all-zero mass, do not
          reinflate toxic arms via renorm)
        """
        self._health_gate_freeze = False
        self._health_gate_slept: list[str] = []
        # Batch CW: source → reason (and optional diagnostics) for disclosure
        self._health_gate_sleep_reasons: Dict[str, str] = {}
        # Batch DU: unhealthy/degraded arms still voting under soft floor
        self._health_gate_soft_floor: Dict[str, str] = {}
        try:
            from src.signals.health_tracker import SignalHealthTracker, SignalHealthStatus
            health_tracker = SignalHealthTracker()
            health_scores = health_tracker.calculate_all_health_scores()

            if not health_scores:
                return weights

            from src.strategy.health_gate_policy import (
                minimum_labeled_daily_cohorts,
                unhealthy_min_ic as resolve_unhealthy_min_ic,
            )

            # ADR-006: health hard-zero is advisory-only and requires a
            # sufficiently labeled daily cohort before excluding vote mass.
            unhealthy_min_ic = resolve_unhealthy_min_ic()
            hard_zero_min_cohorts = minimum_labeled_daily_cohorts()

            adjusted_weights = {}
            slept: list[str] = []
            sleep_reasons: Dict[str, str] = {}
            soft_floor: Dict[str, str] = {}
            for source_enum, base_weight in weights.items():
                source_str = source_enum.value
                if source_str in health_scores:
                    health = health_scores[source_str]
                    status = str(getattr(health, "status", "") or "").lower()
                    hs = float(getattr(health, "health_score", 0.0) or 0.0)
                    ic_raw = getattr(health, "ic", None)
                    try:
                        ic_val = float(ic_raw) if ic_raw is not None else None
                    except (TypeError, ValueError):
                        ic_val = None
                    cohorts_raw = getattr(
                        health,
                        "predictions_count",
                        0,
                    )
                    try:
                        labeled_cohorts = int(cohorts_raw)
                    except (TypeError, ValueError):
                        labeled_cohorts = 0
                    cohort_eligible = labeled_cohorts >= hard_zero_min_cohorts

                    # Batch CY hybrid (evolves BH/CN) + Batch DU min-IC for unhealthy:
                    # - hard sleep toxic arms: IC < 0 (any non-healthy status), or
                    #   unhealthy with unknown IC (fail-closed without IC evidence)
                    # - hard sleep unhealthy with weak IC < UNHEALTHY_MIN_IC
                    # - soft floor max(0.2, hs) when quality is poor but IC ≥ min
                    hard_zero_candidate = False
                    candidate_reason = None
                    if (
                        status != SignalHealthStatus.HEALTHY.value
                        and ic_val is not None
                        and ic_val < 0.0
                    ):
                        hard_zero_candidate = True
                        candidate_reason = (
                            f"negative_ic({ic_val:.3f})"
                            if status != SignalHealthStatus.DEGRADED.value
                            else f"degraded_negative_ic({ic_val:.3f})"
                        )
                    elif status == SignalHealthStatus.UNHEALTHY.value:
                        if ic_val is None:
                            hard_zero_candidate = True
                            candidate_reason = "unhealthy_ic_unknown"
                        elif ic_val < unhealthy_min_ic:
                            hard_zero_candidate = True
                            candidate_reason = (
                                f"unhealthy_weak_ic({ic_val:.3f}<{unhealthy_min_ic:.2f})"
                            )
                        # else: unhealthy + IC≥min → soft floor below
                    hard_zero = hard_zero_candidate and cohort_eligible
                    insufficient_reason = None
                    if hard_zero_candidate and not cohort_eligible:
                        insufficient_reason = (
                            "insufficient_cohorts("
                            f"{labeled_cohorts}<{hard_zero_min_cohorts};"
                            f"{candidate_reason or 'hard_zero'})"
                        )

                    if hard_zero:
                        multiplier = 0.0
                        slept.append(source_str)
                        reason = candidate_reason or "hard_zero"
                        sleep_reasons[source_str] = reason
                        logger.info(
                            "Health-gated %s: weight %.2f%% → 0%% (%s, score=%.2f, ic=%s)",
                            source_str,
                            base_weight * 100,
                            reason,
                            hs,
                            ic_val,
                        )
                    else:
                        multiplier = max(0.2, min(1.0, hs))
                        # Batch DU: only disclose soft-floor when arm still has
                        # vote mass (skip zero-baseline / already-zero weight).
                        still_votes = float(base_weight or 0.0) > 1e-12
                        if still_votes and insufficient_reason:
                            soft_floor[source_str] = insufficient_reason
                        elif still_votes and status == SignalHealthStatus.UNHEALTHY.value:
                            soft_floor[source_str] = (
                                f"unhealthy_soft_floor(score={hs:.2f},ic={ic_val})"
                            )
                        elif (
                            still_votes
                            and status == SignalHealthStatus.DEGRADED.value
                            and hs < 0.55
                        ):
                            soft_floor[source_str] = (
                                f"degraded_soft_floor(score={hs:.2f},ic={ic_val})"
                            )
                        if hs < 0.5 or status == SignalHealthStatus.UNHEALTHY.value:
                            logger.info(
                                "Health-adjusted %s: weight %.2f%% → %.2f%% "
                                "(status=%s, health=%.2f, ic=%s)",
                                source_str,
                                base_weight * 100,
                                base_weight * multiplier * 100,
                                status,
                                hs,
                                ic_val,
                            )
                    adjusted_weights[source_enum] = base_weight * multiplier
                else:
                    adjusted_weights[source_enum] = base_weight

            self._health_gate_slept = slept
            self._health_gate_sleep_reasons = sleep_reasons
            self._health_gate_soft_floor = soft_floor
            total = sum(adjusted_weights.values())
            if total > 0:
                weights = {k: v / total for k, v in adjusted_weights.items()}
            else:
                # All arms unhealthy / zero — freeze; do not reinflate via renorm
                self._health_gate_freeze = True
                weights = {k: 0.0 for k in weights}
                logger.warning(
                    "Health gate freeze: all ensemble arms hard-zeroed (%s); "
                    "adaptive blend contributes zero mass",
                    ", ".join(slept) if slept else "no sources",
                )
        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.warning("Could not apply health-adjusted weights: %s", e)
        return weights

    def _apply_correlation_penalty(self, weights: Dict) -> Dict:
        """Apply correlation penalty to reduce weight of redundant signals.

        Computes pairwise prediction correlations from IC decay data.
        Signals highly correlated with peers have their weights reduced
        to improve ensemble diversification and prevent double-counting.

        The penalty is conservative: max 30% reduction for perfectly
        correlated signals. The penalty factor is 1/(1+mean_abs_corr),
        so a signal correlated at 0.7 with peers gets ~0.59 penalty.
        """
        try:
            corr_data = compute_signal_correlation_matrix()
            penalties = corr_data.get("correlation_penalties", {})
            if not penalties:
                return weights

            redundant = corr_data.get("redundant_pairs", [])
            if redundant:
                logger.info(
                    "Redundant signal pairs detected: %s",
                    ', '.join(f'{s1}/{s2}(r={c:.2f})' for s1, s2, c in redundant)
                )

            adjusted = {}
            for source_enum, base_weight in weights.items():
                source_str = source_enum.value
                penalty = penalties.get(source_str, 1.0)
                # Clip to prevent excessive reduction: min penalty = 0.5
                penalty = max(0.5, penalty)
                adjusted[source_enum] = base_weight * penalty
                if abs(penalty - 1.0) > 0.01:
                    logger.info(
                        "Correlation-penalized %s: %.3f -> %.3f (penalty=%.3f)",
                        source_str, base_weight, adjusted[source_enum], penalty
                    )

            # Re-normalize
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {k: v / total for k, v in adjusted.items()}

            return adjusted
        except (KeyError, ValueError, TypeError, OSError, ImportError) as e:
            logger.warning("Could not apply correlation penalty: %s", e)
        return weights

    def _apply_regime_weights(self, weights: Dict, regime: Regime) -> Dict:
        """Apply per-regime signal weight multipliers.

        Varies ensemble signal weights by macro regime using the
        REGIME_CONDITIONAL_WEIGHTS map. Each regime has multipliers
        reflecting which signals perform well in that environment:

        - CRISIS: boost alternative_data, reduce unified_overlay
        - HIGH_VOL: boost unified_overlay (defensive), reduce intl_momentum
        - NORMAL: baseline (no adjustment)
        - LOW_VOL: boost international_momentum, reduce regime_arb (mean-reversion)
        - RECOVERY: boost international_momentum (post-crisis momentum)

        Multipliers are capped at [0.3, 1.5] per signal and weights are
        renormalized to sum=1.0 after adjustment.
        """
        from src.strategy.ensemble_voter import REGIME_CONDITIONAL_WEIGHTS
        try:
            regime_name = regime.name if hasattr(regime, 'name') else str(regime)
            regime_multipliers = REGIME_CONDITIONAL_WEIGHTS.get(regime_name, {})

            # NORMAL is baseline: all signals at 1.0, no adjustment needed
            if regime_name == "NORMAL" and not any(
                v != 1.0 for v in regime_multipliers.values()
            ):
                return weights

            adjusted = {}
            for source_enum, base_weight in weights.items():
                source_str = source_enum.value
                multiplier = float(regime_multipliers.get(source_str, 1.0))
                # Conservative caps: min 0.3, max 1.5
                multiplier = max(0.3, min(1.5, multiplier))
                adjusted[source_enum] = base_weight * multiplier

            total = sum(adjusted.values())
            if total <= 0:
                return weights

            # Gate signals below 5% of total weight to zero
            min_threshold = 0.05 * total
            gated = {
                k: (v if v >= min_threshold else 0.0)
                for k, v in adjusted.items()
            }
            gated_total = sum(gated.values())
            if gated_total <= 0:
                return weights

            # Renormalize
            result = {k: v / gated_total for k, v in gated.items()}

            if regime_name != "NORMAL":
                logger.info(
                    "Regime-conditional weights (%s): %s",
                    regime_name,
                    ', '.join(
                        f'{k.value}={result[k]:.3f}'
                        for k in result if result[k] > 0
                    )
                )

            return result

        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Could not apply regime-conditional weights: %s", e)
        return weights

    def _apply_utility_reweighting(self, weights: Dict, regime: Regime) -> Dict:
        """Apply utility-based reweighting from signal profitability data.

        Boosts weights for signals with positive Sharpe contribution from
        attribution data, reduces weights for negative contributors.
        This is complementary to health-based adjustment: health measures
        signal reliability (IC, consistency), utility measures profitability.

        The adjustment is conservative: max ±30% weight change, and only
        applied when attribution has enough observations (>=20 readings).
        """
        try:
            attribution_dir = ATTRIBUTION_DIR
            attribution_files = sorted(attribution_dir.glob("attribution_*.json"), reverse=True)

            if not attribution_files:
                return weights

            with open(attribution_files[0]) as f:
                attribution_data = json.load(f)

            # Check freshness
            attr_timestamp = attribution_data.get("timestamp", "")
            if attr_timestamp:
                attr_date = attr_timestamp[:10]
                days_stale = (datetime.now() - datetime.strptime(attr_date, "%Y-%m-%d")).days
            else:
                days_stale = 999

            if days_stale > 7:
                return weights

            sources = attribution_data.get("sources", {})
            if not sources:
                return weights

            adjusted = {}
            for source_enum, base_weight in weights.items():
                source_str = source_enum.value
                source_data = sources.get(source_str, {})

                # Need enough observations for meaningful Sharpe
                total_readings = source_data.get("total_readings", 0)
                if total_readings < 20:
                    adjusted[source_enum] = base_weight
                    continue

                sharpe_contrib = source_data.get("sharpe_contribution", 0.0)
                hit_rate = source_data.get("hit_rate", 0.0)

                # Utility score: blend Sharpe contribution (primary) and hit rate (secondary)
                # Sharpe contribution is in annualized units; normalize to [-1, 1] range
                sharpe_signal = np.clip(sharpe_contrib / 2.0, -1.0, 1.0)  # ±2 Sharpe = full signal
                hit_signal = (hit_rate - 0.5) * 2.0 if hit_rate > 0 else 0.0  # 50% hit rate = neutral

                # Weighted blend: 70% Sharpe, 30% hit rate
                utility_score = 0.7 * sharpe_signal + 0.3 * hit_signal

                # Conservative adjustment: max ±30% weight change
                # Positive utility → boost, negative → reduce
                adjustment = 1.0 + np.clip(utility_score * 0.3, -0.3, 0.3)
                adjusted[source_enum] = base_weight * adjustment

                if abs(utility_score) > 0.1:
                    logger.info("Utility-reweighted %s: %.2f%% → %.2f%% (utility=%.3f, sharpe_contrib=%.3f, hit_rate=%.2f)",
                                source_str, base_weight * 100, adjusted[source_enum] * 100,
                                utility_score, sharpe_contrib, hit_rate)

            # Renormalize
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {k: v / total for k, v in adjusted.items()}

            return adjusted

        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.warning("Could not apply utility-based reweighting: %s", e)
        return weights

    def _apply_exploration_noise(self, weights: Dict, regime: Regime) -> Dict:
        """Apply epsilon-greedy exploration noise to weight allocation.

        With probability exploration_epsilon, draws weights from a Dirichlet
        distribution centered on current weights. This allows the system to
        discover better weight configurations that might otherwise go untested,
        without risking large deviations from the baseline allocation.

        Dirichlet concentration alpha controls how close the samples stay
        to the mean — higher alpha = closer to current weights.
        """
        epsilon = float(os.environ.get("ENSEMBLE_EXPLORATION_EPSILON", "0.05"))
        if random.random() >= epsilon:
            return weights

        # Dirichlet concentration parameter — higher = closer to current weights
        # alpha=10 means samples typically stay within ±10% of current weights
        alpha_base = float(os.environ.get("ENSEMBLE_EXPLORATION_ALPHA", "10.0"))

        weight_values = [weights[k] for k in weights]
        n = len(weight_values)
        if n < 2:
            return weights

        # Dirichlet alpha: concentration * current weight for each component.
        # Soft-delete arms (static zero) get tiny alpha so samples stay ~0, then
        # Batch DK pin zeros them hard after sampling.
        regime_name = regime.name if hasattr(regime, "name") else str(regime)
        soft_delete = self._static_zero_baseline_sources(regime_name)
        alpha = []
        for k, w in zip(weights.keys(), weight_values):
            if k in soft_delete or float(w or 0.0) <= 0.0:
                alpha.append(1e-9)  # near-zero mass; pin finishes the job
            else:
                alpha.append(max(0.1, alpha_base * w))

        # Sample from Dirichlet
        try:
            sampled = np.random.dirichlet(alpha)
            result = {k: float(sampled[i]) for i, k in enumerate(weights)}
            result = self._pin_zero_baseline_weights(result, regime_name)
            logger.info("Exploration noise applied: epsilon=%.2f, regime=%s", epsilon, regime.value)
            return result
        except (ValueError, FloatingPointError) as e:
            logger.warning("Exploration noise failed: %s", e)
            return weights

    def _apply_diversity_floor(
        self,
        weights: Dict,
        floor: Optional[float] = None,
    ) -> Dict:
        """Apply diversity floor — minimum weight for each active signal.

        Prevents weight concentration by ensuring every signal that was
        originally active (weight > 0) retains at least `floor` fraction
        of the total weight. This raises N_eff (effective signal count)
        without overriding the signal quality assessment.

        The floor is applied as a lower bound, not an equalizer: signals
        with higher quality still get proportionally more weight.

        Batch BM: never raise or reinflate arms hard-zeroed by the health
        gate (``_health_gate_slept``). Soft floors must not undo quality sleep.

        Args:
            weights: Current weight dict {SignalSource: weight}.
            floor: Minimum weight fraction per active signal. If None,
                uses DEFAULT_DIVERSITY_FLOOR.

        Returns:
            Adjusted weights dict summing to 1.0 (or all-zero if freeze).
        """
        from src.strategy.ensemble_voter import DEFAULT_DIVERSITY_FLOOR
        if floor is None:
            floor = DEFAULT_DIVERSITY_FLOOR
        if floor <= 0:
            return weights

        slept_names = {
            str(s) for s in (getattr(self, "_health_gate_slept", None) or [])
        }

        def _src_name(source) -> str:
            return source.value if hasattr(source, "value") else str(source)

        # Only apply to signals that were active (weight > 0) and not health-slept
        active = {
            k: v
            for k, v in weights.items()
            if v > 0 and _src_name(k) not in slept_names
        }
        if len(active) <= 1:
            # Still force slept arms to zero if somehow positive
            if slept_names:
                cleaned = dict(weights)
                for k in list(cleaned):
                    if _src_name(k) in slept_names:
                        cleaned[k] = 0.0
                total_c = sum(cleaned.values())
                if total_c > 0:
                    return {k: v / total_c for k, v in cleaned.items()}
                return cleaned
            return weights

        total = sum(weights.values())
        if total <= 0:
            return weights

        # Normalize to get fractional weights; keep slept at 0
        frac = {}
        for k, v in weights.items():
            if _src_name(k) in slept_names:
                frac[k] = 0.0
            else:
                frac[k] = v / total

        # Identify signals below the floor (never raise slept)
        adjusted = dict(frac)
        raised_count = 0
        for source in active:
            if adjusted[source] < floor:
                adjusted[source] = floor
                raised_count += 1

        if raised_count == 0 and not slept_names:
            return weights  # No adjustment needed

        # Re-normalize so weights sum to 1.0 (slept stay 0)
        new_total = sum(adjusted.values())
        if new_total > 0:
            adjusted = {k: v / new_total for k, v in adjusted.items()}
            for k in adjusted:
                if _src_name(k) in slept_names:
                    adjusted[k] = 0.0
            # Renorm once more if slept zeros left a hole (shouldn't)
            nt = sum(adjusted.values())
            if nt > 0 and abs(nt - 1.0) > 1e-9:
                adjusted = {k: v / nt for k, v in adjusted.items()}

        if raised_count:
            logger.info(
                "Diversity floor applied: raised %d/%d active signals (floor=%.1f%%); "
                "health-slept excluded=%d",
                raised_count,
                len(active),
                floor * 100,
                len(slept_names),
            )

        return adjusted

    @staticmethod
    def _extract_signal_values(readings: Dict) -> Dict[str, float]:
        """Build signal_values dict from current readings, skipping NaN."""
        signal_values = {}
        for source_enum in readings:
            source_str = source_enum.value
            reading = readings[source_enum]
            if not np.isnan(reading.value):
                signal_values[source_str] = reading.value
        return signal_values

    def _apply_basis_pursuit(
        self, signal_values: Dict[str, float], base_weights_str: Dict[str, float], regime_value: str
    ) -> Dict[str, float]:
        """Apply basis-pursuit signal selection to prune redundant signals."""
        try:
            from src.strategy.basis_pursuit_selector import BasisPursuitSelector
            bp_selector = BasisPursuitSelector()
            bp_result = bp_selector.select_signals(
                signal_values, base_weights_str, regime=regime_value
            )
            sparsity_msg = (
                f" (sparsity={bp_result.sparsity_ratio:.2f}, "
                f"{bp_result.num_pruned} pruned)"
                if bp_result.num_pruned > 0
                else ""
            )
            logger.debug("Basis-pursuit selection applied%s", sparsity_msg)
            return bp_result.active_signals
        except (ImportError, KeyError, ValueError, TypeError, AttributeError, ZeroDivisionError) as bp_e:
            logger.warning("Could not apply basis-pursuit selection: %s", bp_e)
            return base_weights_str

    def _apply_regret_weighting(
        self, signal_values: Dict[str, float], base_weights_str: Dict[str, float], regime_value: str
    ) -> Dict[str, float]:
        """Apply regret-weighted adjustment to penalize signals with high regret."""
        try:
            from src.strategy.regret_weighted_selector import RegretWeightedSelector
            rw_selector = RegretWeightedSelector()
            prev_decision = getattr(rw_selector.state, 'last_ensemble_decision', 0.0)
            rw_result = rw_selector.adjust_weights(
                signal_values, prev_decision, base_weights_str, regime=regime_value
            )
            if rw_result.signals_with_high_regret:
                logger.info(
                    "Regret-adjusted weights: penalized %s (avg_regret=%.3f)",
                    ', '.join(rw_result.signals_with_high_regret),
                    rw_result.avg_regret
                )
            return rw_result.adjusted_weights
        except (ImportError, KeyError, ValueError, TypeError, AttributeError, OSError) as rw_e:
            logger.warning("Could not apply regret-weighted adjustment: %s", rw_e)
            return base_weights_str

    def _apply_turnover_validation(
        self, weights: Dict, readings: Dict, regime: Regime
    ) -> Dict:
        """Apply turnover-aware weight validation (v8.01) with basis-pursuit and regret-weighted."""
        try:
            from src.strategy.turnover_validator import TurnoverValidator
            turnover_validator = TurnoverValidator()

            signal_values = self._extract_signal_values(readings)
            if not signal_values:
                return weights

            base_weights_str = {source_enum.value: w for source_enum, w in weights.items()}

            base_weights_str = self._apply_basis_pursuit(signal_values, base_weights_str, regime.value)
            base_weights_str = self._apply_regret_weighting(signal_values, base_weights_str, regime.value)

            # Apply turnover adjustment
            adjusted_str = turnover_validator.get_adjusted_weights(
                base_weights_str, signal_values
            )

            # Convert back to enum-keyed dict
            turnover_adjusted = {}
            for source_enum in weights:
                source_str = source_enum.value
                if source_str in adjusted_str:
                    turnover_adjusted[source_enum] = adjusted_str[source_str]
                else:
                    turnover_adjusted[source_enum] = weights[source_enum]

            # Re-normalize to sum to 1.0
            total = sum(turnover_adjusted.values())
            if total > 0:
                weights = {k: v / total for k, v in turnover_adjusted.items()}

            logger.debug(
                "Turnover-adjusted %d signals: %s",
                len(signal_values),
                ', '.join(f'{s}={turnover_adjusted.get(enum, 0):.4f}' for enum, s in [(e, e.value) for e in weights])
            )
        except (KeyError, ValueError, TypeError, AttributeError, ZeroDivisionError, OSError) as e:
            logger.warning("Could not apply turnover-aware weights: %s", e)
        return weights

    def _zero_inactive_and_recap(
        self,
        weights: Dict,
        readings: Dict[SignalSource, SignalReading],
        regime_name: str,
    ) -> Dict:
        """Zero inactive_signal mass, renorm awake arms, re-apply per-signal cap.

        Batch DP: sleeping-expert style — only awake (is_active) arms keep vote
        mass. Soft-delete stays pinned at 0. Re-cap after renorm so no arm
        exceeds DEFAULT_PER_SIGNAL_WEIGHT_CAP once inactive mass is dropped.
        """
        if not weights or not readings:
            return weights
        out = {k: float(v or 0.0) for k, v in weights.items()}
        soft = self._static_zero_baseline_sources(regime_name)
        inactive: list = []
        for source, reading in readings.items():
            if source in soft:
                out[source] = 0.0
                continue
            if not getattr(reading, "is_active", True):
                if float(out.get(source, 0.0) or 0.0) > 1e-12:
                    inactive.append(
                        source.value if hasattr(source, "value") else str(source)
                    )
                out[source] = 0.0
        for src in soft:
            if src in out:
                out[src] = 0.0
        total = sum(max(0.0, v) for v in out.values())
        if total > 0 and abs(total - 1.0) > 1e-9:
            out = {k: max(0.0, float(v or 0.0)) / total for k, v in out.items()}
        out = self._cap_per_signal_weights(out, regime_name)
        if inactive:
            logger.info(
                "Batch DP: redistributed mass from %d inactive_signal arm(s): %s",
                len(inactive),
                ",".join(inactive),
            )
        return out

    def _apply_weights_to_readings(
        self,
        readings: Dict[SignalSource, SignalReading],
        weights: Dict,
    ) -> List[SignalReading]:
        """Assign weights to readings and log predictions for health tracking."""
        weighted_signals = []
        for source, reading in readings.items():
            if source in weights:
                # Batch CV: inactive snapshots stay in the vote trail for
                # disclosure (source_breakdown) but must not move consensus.
                # Batch DP: pipeline already zeroed+renormed inactive mass.
                if getattr(reading, "is_active", True):
                    reading.weight = weights[source]
                else:
                    reading.weight = 0.0
                weighted_signals.append(reading)

        # Log signal predictions for health tracking (v3.12 / Batch DF provenance)
        try:
            tracker = _get_health_tracker()
            if tracker is not None:
                for reading in weighted_signals:
                    meta = getattr(reading, "metadata", None)
                    if not isinstance(meta, dict):
                        meta = {}
                    else:
                        meta = dict(meta)
                    # Compact provenance always stamped for post-fix IC cohorts
                    meta.setdefault("provenance_batch", "df")
                    if getattr(reading, "explanation", None):
                        meta.setdefault("explanation", str(reading.explanation)[:200])
                    meta.setdefault("is_active", bool(getattr(reading, "is_active", True)))
                    tracker.log_prediction_simple(
                        source=reading.source.value,
                        signal_value=reading.value,
                        confidence=reading.confidence,
                        metadata=meta,
                    )
        except (KeyError, ValueError, TypeError, AttributeError, OSError, sqlite3.Error) as e:
            logger.warning("Health tracking log failed: %s", e)

        return weighted_signals
