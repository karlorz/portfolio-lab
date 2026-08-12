"""Offline regime probability fusion prototype.

This module intentionally does not route into live gating or broker-facing
allocation authority. It combines already-published regime probability surfaces
so they can be calibrated and backtested before any promotion decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from src.utils.metrics_io import compute_metrics_from_returns
from src.regime.regime_transition_forecaster import REGIMES

__all__ = [
    "REGIMES",
    "FusedRegimeSignal",
    "RegimeFusionEvaluation",
    "fuse_regime_probabilities",
    "evaluate_next_regime_predictions",
    "evaluate_regime_fusion_backtest",
]


_REGIME_ALIASES = {
    "VOL_SPIKE": "HIGH_VOL",
    "HIGHVOL": "HIGH_VOL",
    "HIGH VOL": "HIGH_VOL",
    "LOWVOL": "LOW_VOL",
    "LOW VOL": "LOW_VOL",
}


@dataclass(frozen=True)
class FusedRegimeSignal:
    """Research-only fused regime distribution."""

    regime: str
    confidence: float
    probabilities: Dict[str, float]
    raw_confidence: float
    bocd_change_prob: float
    bocd_discount: float
    bocd_role: str = "confidence_discount"
    fallback_used: bool = False
    source_weights: Dict[str, float] = field(default_factory=dict)
    scope: str = "research_offline_only"
    live_routed: bool = False

    def to_dict(self) -> Dict[str, object]:
        """Return dashboard/research metadata without implying live authority."""
        return {
            "schema_version": "regime-probability-fusion/v1",
            "regime": self.regime,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
            "raw_confidence": self.raw_confidence,
            "bocd_change_prob": self.bocd_change_prob,
            "bocd_discount": self.bocd_discount,
            "bocd_role": self.bocd_role,
            "fallback_used": self.fallback_used,
            "source_weights": dict(self.source_weights),
            "scope": self.scope,
            "live_routed": self.live_routed,
        }


@dataclass(frozen=True)
class RegimeFusionEvaluation:
    """Offline calibration and portfolio-impact summary."""

    n_observations: int
    persistence_accuracy: float
    mean_brier_score: float
    mean_log_loss: float
    portfolio_impact: Dict[str, Optional[float]] = field(default_factory=dict)
    scope: str = "research_offline_only"
    live_routed: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": "regime-probability-fusion-eval/v1",
            "n_observations": self.n_observations,
            "persistence_accuracy": self.persistence_accuracy,
            "mean_brier_score": self.mean_brier_score,
            "mean_log_loss": self.mean_log_loss,
            "portfolio_impact": dict(self.portfolio_impact),
            "scope": self.scope,
            "live_routed": self.live_routed,
        }


def fuse_regime_probabilities(
    *,
    two_stage_probs: Optional[Mapping[str, float]],
    transition_probs: Optional[Mapping[str, float]],
    bocd_change_prob: float,
    hard_label: Optional[str],
    two_stage_weight: float = 0.6,
    transition_weight: float = 0.4,
    bocd_discount_strength: float = 0.5,
) -> FusedRegimeSignal:
    """Fuse current and forecast regime probabilities for offline evaluation.

    Formula:
    1. Normalize available probability surfaces to the canonical five-regime
       vocabulary.
    2. Blend two-stage and transition probabilities using active-source weights.
    3. If probability surfaces are unavailable, fall back to the current hard
       label.
    4. Treat BOCD as a confidence discount by mixing toward a uniform
       distribution. It never hard-overrides the selected regime in this slice.
    """
    normalized_two_stage = _normalize_probabilities(two_stage_probs)
    normalized_transition = _normalize_probabilities(transition_probs)

    active_sources: list[tuple[str, float, Dict[str, float]]] = []
    if normalized_two_stage is not None and two_stage_weight > 0:
        active_sources.append(("two_stage", float(two_stage_weight), normalized_two_stage))
    if normalized_transition is not None and transition_weight > 0:
        active_sources.append(("transition", float(transition_weight), normalized_transition))

    fallback_used = False
    if active_sources:
        total_weight = sum(weight for _, weight, _ in active_sources)
        source_weights = {
            name: weight / total_weight for name, weight, _ in active_sources
        }
        base_probs = {regime: 0.0 for regime in REGIMES}
        for name, _, probabilities in active_sources:
            source_weight = source_weights[name]
            for regime in REGIMES:
                base_probs[regime] += probabilities[regime] * source_weight
    else:
        fallback_used = True
        source_weights = {"hard_label": 1.0}
        base_probs = _hard_label_distribution(hard_label)

    base_probs = _renormalize(base_probs)
    raw_confidence = max(base_probs.values())

    change_prob = _clamp(float(bocd_change_prob), 0.0, 1.0)
    discount_strength = _clamp(float(bocd_discount_strength), 0.0, 1.0)
    bocd_discount = _clamp(1.0 - change_prob * discount_strength, 0.0, 1.0)

    uniform = _uniform_distribution()
    probabilities = {
        regime: base_probs[regime] * bocd_discount
        + uniform[regime] * (1.0 - bocd_discount)
        for regime in REGIMES
    }
    probabilities = _renormalize(probabilities)

    regime = max(probabilities, key=probabilities.get)
    confidence = probabilities[regime]

    return FusedRegimeSignal(
        regime=regime,
        confidence=confidence,
        probabilities=probabilities,
        raw_confidence=raw_confidence,
        bocd_change_prob=change_prob,
        bocd_discount=bocd_discount,
        fallback_used=fallback_used,
        source_weights=source_weights,
    )


def evaluate_next_regime_predictions(
    *,
    predicted_probabilities: Sequence[Mapping[str, float]],
    actual_next_regimes: Sequence[str],
) -> RegimeFusionEvaluation:
    """Evaluate next-hard-regime calibration for fused distributions."""
    if len(predicted_probabilities) != len(actual_next_regimes):
        raise ValueError("Predictions and actual labels must have the same length.")
    if len(predicted_probabilities) == 0:
        raise ValueError("At least one prediction is required.")

    brier_scores = []
    log_losses = []
    correct = 0

    for raw_prediction, actual in zip(predicted_probabilities, actual_next_regimes):
        probabilities = _normalize_probabilities(raw_prediction)
        if probabilities is None:
            raise ValueError("Predicted probabilities must contain positive mass.")

        actual_regime = _normalize_regime_name(actual)
        predicted_regime = max(probabilities, key=probabilities.get)
        correct += int(predicted_regime == actual_regime)

        brier = 0.0
        for regime in REGIMES:
            target = 1.0 if regime == actual_regime else 0.0
            brier += (probabilities[regime] - target) ** 2
        brier_scores.append(brier)

        actual_probability = max(probabilities[actual_regime], 1e-12)
        log_losses.append(-math.log(actual_probability))

    n = len(predicted_probabilities)
    return RegimeFusionEvaluation(
        n_observations=n,
        persistence_accuracy=correct / n,
        mean_brier_score=float(np.mean(brier_scores)),
        mean_log_loss=float(np.mean(log_losses)),
        portfolio_impact=_empty_portfolio_impact(),
    )


def evaluate_regime_fusion_backtest(
    *,
    predicted_probabilities: Sequence[Mapping[str, float]],
    actual_next_regimes: Sequence[str],
    baseline_returns: Optional[Sequence[float]] = None,
    fused_returns: Optional[Sequence[float]] = None,
    baseline_weights: Optional[Sequence[Mapping[str, float]]] = None,
    fused_weights: Optional[Sequence[Mapping[str, float]]] = None,
) -> RegimeFusionEvaluation:
    """Evaluate calibration plus optional offline portfolio-impact deltas."""
    calibration = evaluate_next_regime_predictions(
        predicted_probabilities=predicted_probabilities,
        actual_next_regimes=actual_next_regimes,
    )

    impact = _empty_portfolio_impact()
    if baseline_returns is not None and fused_returns is not None:
        baseline_metrics = compute_metrics_from_returns(baseline_returns)
        fused_metrics = compute_metrics_from_returns(fused_returns)
        impact["sharpe_delta"] = fused_metrics["sharpe"] - baseline_metrics["sharpe"]
        impact["max_drawdown_delta"] = (
            fused_metrics["max_drawdown"] - baseline_metrics["max_drawdown"]
        )

    if baseline_weights is not None and fused_weights is not None:
        impact["turnover_delta"] = (
            _average_turnover(fused_weights) - _average_turnover(baseline_weights)
        )

    return RegimeFusionEvaluation(
        n_observations=calibration.n_observations,
        persistence_accuracy=calibration.persistence_accuracy,
        mean_brier_score=calibration.mean_brier_score,
        mean_log_loss=calibration.mean_log_loss,
        portfolio_impact=impact,
    )


def _normalize_probabilities(
    probabilities: Optional[Mapping[str, float]],
) -> Optional[Dict[str, float]]:
    if not probabilities:
        return None

    normalized = {regime: 0.0 for regime in REGIMES}
    for raw_regime, raw_probability in probabilities.items():
        try:
            regime = _normalize_regime_name(raw_regime)
        except ValueError:
            continue
        normalized[regime] += max(float(raw_probability), 0.0)

    total = sum(normalized.values())
    if total <= 0:
        return None
    return {regime: value / total for regime, value in normalized.items()}


def _normalize_regime_name(regime: str) -> str:
    normalized = str(regime).strip().upper().replace("-", "_")
    normalized = _REGIME_ALIASES.get(normalized, normalized)
    if normalized not in REGIMES:
        raise ValueError(f"Unknown regime: {regime}")
    return normalized


def _hard_label_distribution(hard_label: Optional[str]) -> Dict[str, float]:
    regime = _normalize_regime_name(hard_label or "NORMAL")
    return {candidate: 1.0 if candidate == regime else 0.0 for candidate in REGIMES}


def _uniform_distribution() -> Dict[str, float]:
    probability = 1.0 / len(REGIMES)
    return {regime: probability for regime in REGIMES}


def _renormalize(probabilities: Mapping[str, float]) -> Dict[str, float]:
    total = sum(max(float(value), 0.0) for value in probabilities.values())
    if total <= 0:
        return _uniform_distribution()
    return {
        regime: max(float(probabilities.get(regime, 0.0)), 0.0) / total
        for regime in REGIMES
    }


def _average_turnover(weights: Sequence[Mapping[str, float]]) -> float:
    if len(weights) < 2:
        return 0.0

    turnovers = []
    for previous, current in zip(weights, weights[1:]):
        assets = set(previous) | set(current)
        one_way_turnover = 0.5 * sum(
            abs(float(current.get(asset, 0.0)) - float(previous.get(asset, 0.0)))
            for asset in assets
        )
        turnovers.append(one_way_turnover)
    return float(np.mean(turnovers)) if turnovers else 0.0


def _empty_portfolio_impact() -> Dict[str, Optional[float]]:
    return {
        "sharpe_delta": None,
        "turnover_delta": None,
        "max_drawdown_delta": None,
    }


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))
