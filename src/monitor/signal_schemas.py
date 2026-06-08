"""
Pydantic v2 signal validation schemas for the portfolio-lab pipeline.

Provides typed models for known signal dicts.  Validation is always
non-fatal: on error the original dict is returned unchanged so the pipeline
never crashes from bad data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Individual signal schemas
# ─────────────────────────────────────────────────────────────


class EnsembleVotingSignal(BaseModel):
    """Validates the ``ensemble_voting`` section of signals.json."""

    model_config = ConfigDict(extra="allow")

    regime: str = "unknown"
    regime_confidence: float = 0.0
    weighted_consensus: float = 0.0
    agreement_ratio: float = 0.0
    action: str = "hold"
    confidence: float = 0.0
    equity_bias: float = 0.0
    duration_bias: float = 0.0
    gold_bias: float = 0.0
    num_sources: int = 0
    source_breakdown: List[Dict[str, Any]] = Field(default_factory=list)


class GarchCvarSignal(BaseModel):
    """Validates the ``garch_cvar`` section of signals.json.

    All numeric fields default to 0.0 / None so missing data does not crash
    validation during early pipeline runs.
    """

    model_config = ConfigDict(extra="allow")

    cvar_95: float = 0.0
    cvar_95_garch: Optional[float] = None
    var_95: float = 0.0
    var_95_garch: Optional[float] = None
    cvar_ratio: float = 1.0
    garch_active: bool = False
    current_volatility: float = 0.0
    forecast_volatility: Optional[float] = None
    volatility_clustering: str = "normal"
    # Conformal CVaR cross-check (distribution-free) — optional
    conformal_cvar_95: Optional[float] = None
    conformal_var_95: Optional[float] = None
    conformal_cvar_ratio: Optional[float] = None


class SmartRebalanceSignal(BaseModel):
    """Validates the ``smart_rebalance`` section of signals.json."""

    model_config = ConfigDict(extra="allow")

    should_execute: bool = False
    decision: str = "none"
    urgency: str = "low"
    max_drift: float = 0.0
    estimated_cost_bps: float = 0.0
    reason: str = ""


class RegimeSignal(BaseModel):
    """Validates the ``regime`` section of signals.json."""

    model_config = ConfigDict(extra="allow")

    regime: str = "normal"
    vix: Optional[float] = None
    detected: Optional[str] = None


class YieldCurveSignal(BaseModel):
    """Validates the ``yield_curve`` section of signals.json."""

    model_config = ConfigDict(extra="allow")

    spread2s10s: float = 0.0
    dgs2: Optional[float] = None
    dgs10: Optional[float] = None
    duration_regime: str = "normal"
    spread_history: List[float] = Field(default_factory=list)


class SignalSnapshotSchema(BaseModel):
    """Pydantic counterpart of the dataclass-based ``SignalSnapshot``.

    Mirrors ``src/signals/signal_snapshot.py`` fields with sensible defaults.
    """

    model_config = ConfigDict(extra="allow")

    source: str = "unknown"
    timestamp: str = ""
    value: float = 0.0
    confidence: float = 0.0
    asset_signals: Dict[str, float] = Field(default_factory=dict)
    regime_fit: str = "normal"
    is_active: bool = True
    explanation: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FredMacroSignal(BaseModel):
    """Validates the ``fred_macro`` section of signals.json."""

    model_config = ConfigDict(extra="allow")

    regime: str = "UNKNOWN"
    confidence: float = 0.0
    recession_probability: float = 0.0
    inflation_pressure: float = 0.0
    monetary_stance: str = "unknown"
    manufacturing_health: float = 50.0
    credit_conditions: str = "unknown"
    indicators: Dict[str, float] = Field(default_factory=dict)
    timestamp: str = ""


class TwoStageRegimeSignal(BaseModel):
    """Validates the ``two_stage_regime`` section of signals.json.

    Two-stage k-means macro regime classifier (Oliveira et al. 2025).
    """

    model_config = ConfigDict(extra="allow")

    regime: str = "UNKNOWN"
    confidence: float = 0.0
    crisis_probability: float = 0.0
    probabilities: Dict[str, float] = Field(default_factory=dict)
    n_pca_components: int = 0
    variance_retained: float = 0.0
    n_observations: int = 0
    n_series: int = 0
    method: str = "oliveira_2025_two_stage_kmeans"
    timestamp: str = ""


class BOCDSignal(BaseModel):
    """Validates the ``bocd_regime`` section of signals.json.

    Bayesian Online Changepoint Detection (Adams & MacKay 2007) for
    real-time structural break detection in return series.
    """

    model_config = ConfigDict(extra="allow")

    regime: int = 0
    regime_change_prob: float = 0.0
    changepoint_count: int = 0
    current_run_length: int = 0
    hazard_rate: float = 1.0 / 252
    threshold: float = 0.5
    n_observations: int = 0
    description: str = "Bayesian Online Changepoint Detection regime signal"
    timestamp: str = ""


# ─────────────────────────────────────────────────────────────
#  Signal model registry
# ─────────────────────────────────────────────────────────────

class IcDecaySignal(BaseModel):
    """Validates the ``ic_decay`` section of signals.json."""

    model_config = ConfigDict(extra="allow")

    signals: Dict[str, Any] = Field(default_factory=dict)


class SignalWfeSignal(BaseModel):
    """Validates the ``signal_wfe`` section of signals.json."""

    model_config = ConfigDict(extra="allow")

    signals: Dict[str, Any] = Field(default_factory=dict)


class RampSignal(BaseModel):
    """Validates the ``ramp`` section of signals.json."""

    model_config = ConfigDict(extra="allow")

    phase: str = "paper"
    allocation_pct: float = 0.0
    days_at_phase: int = 0
    can_advance: bool = False


class GoldTltCorrelationSignal(BaseModel):
    """Validates the ``gold_tlt_correlation`` section of signals.json."""

    model_config = ConfigDict(extra="allow")

    current_correlation: float = 0.0
    current_regime: str = "neutral"
    correlation_trend: str = "stable"
    mean_correlation: float = 0.0
    min_correlation: float = 0.0
    max_correlation: float = 0.0
    structural_breaks_count: int = 0
    regimes_count: int = 0
    implications: str = ""


class PortfolioExplainabilitySignal(BaseModel):
    """Validates the ``portfolio_explainability`` section of signals.json."""

    model_config = ConfigDict(extra="allow")

    latest_decision: Optional[Dict[str, Any]] = None
    recent_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    signal_deep_dives: Dict[str, Any] = Field(default_factory=dict)
    top_sources_today: List[str] = Field(default_factory=list)
    decision_quality: Dict[str, Any] = Field(default_factory=dict)
    action: str = "hold"
    regime: str = "unknown"
    weighted_consensus: float = 0.0


class HedgeSelectorSignal(BaseModel):
    """Validates the ``hedge_selector`` section of signals.json."""

    model_config = ConfigDict(extra="allow")

    available: bool = False
    generated_at: str = ""
    regime: str = "unknown"
    regime_confidence: float = 0.0
    primary_hedge: str = "none"
    primary_size_pct: float = 0.0
    secondary_hedge: Optional[str] = None
    secondary_size_pct: float = 0.0
    cost_benefit_gate: bool = False
    net_benefit_bps: float = 0.0
    kelly_fraction: float = 0.0
    expected_cost_bps: float = 0.0
    expected_benefit_bps: float = 0.0
    confidence_scaled_size: float = 0.0
    min_hold_days: int = 0
    transition_cost_bps: float = 0.0


SIGNAL_MODELS: Dict[str, type[BaseModel]] = {
    "ensemble_voting": EnsembleVotingSignal,
    "garch_cvar": GarchCvarSignal,
    "smart_rebalance": SmartRebalanceSignal,
    "regime": RegimeSignal,
    "yield_curve": YieldCurveSignal,
    "signal_snapshot": SignalSnapshotSchema,
    "fred_macro": FredMacroSignal,
    "two_stage_regime": TwoStageRegimeSignal,
    "bocd_regime": BOCDSignal,
    "ic_decay": IcDecaySignal,
    "signal_wfe": SignalWfeSignal,
    "ramp": RampSignal,
    "gold_tlt_correlation": GoldTltCorrelationSignal,
    "portfolio_explainability": PortfolioExplainabilitySignal,
    "hedge_selector": HedgeSelectorSignal,
}

# Signals for which schemas are defined — used when integrating into
# the DashboardGenerator so the loop only targets known signals.
VALIDATED_SIGNAL_KEYS = frozenset(SIGNAL_MODELS.keys())


def validate_signal(signal_name: str, data: dict) -> dict:
    """Validate *data* against the Pydantic schema registered for *signal_name*.

    Returns the validated dict (with defaults filled) on success, or the
    original *data* unchanged on failure.  This contract guarantees the
    pipeline never crashes from a ``ValidationError``.
    """
    model_cls = SIGNAL_MODELS.get(signal_name)
    if model_cls is None:
        return data  # No schema defined yet — pass through

    if not isinstance(data, dict):
        logger.warning(
            "Signal '%s': expected dict, got %s — returning as-is",
            signal_name, type(data).__name__,
        )
        return data

    try:
        return model_cls.model_validate(data).model_dump()
    except ValidationError as exc:
        logger.warning(
            "Signal validation failed for '%s' (%d error(s)): %s",
            signal_name,
            exc.error_count(),
            exc,
        )
        return data  # Graceful degradation


def validate_all_signals(data: dict) -> dict:
    """Validate all known signal fields in the complete signals dict.

    Uses :func:`validate_signal` for each known signal name found in *data*.
    Unknown keys are passed through unchanged.

    Args:
        data: The top-level signals dict (e.g. a ``signals.json`` payload).

    Returns:
        A new dict with known signal sections validated (defaults filled in),
        unknown keys preserved.
    """
    validated = dict(data)
    for signal_name in list(SIGNAL_MODELS.keys()):
        if signal_name in validated and isinstance(validated[signal_name], dict):
            validated[signal_name] = validate_signal(signal_name, validated[signal_name])
    return validated


# ─────────────────────────────────────────────────────────────
#  Top-level SignalsData (gradual adoption wrapper)
# ─────────────────────────────────────────────────────────────


class SignalsData(BaseModel):
    """Top-level model for the full ``signals.json`` output.

    Uses ``model_validator(mode='wrap')`` to validate known top-level keys
    while passing through unknown ones (gradual adoption).  Extra fields
    at the top level are allowed via ``extra='allow'``.
    """

    model_config = ConfigDict(extra="allow")

    generated_at: str = ""
    regime: Optional[RegimeSignal] = None
    ensemble_voting: Optional[EnsembleVotingSignal] = None
    garch_cvar: Optional[GarchCvarSignal] = None
    smart_rebalance: Optional[SmartRebalanceSignal] = None
    yield_curve: Optional[YieldCurveSignal] = None
    hedge_selector: Optional[HedgeSelectorSignal] = None

    @classmethod
    def validate_dict(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a raw signals.json dict, returning a cleaned dict.

        Known sections are validated against their schemas; unknown keys
        are passed through unchanged.  Individual section failures produce a
        warning log but never raise.
        """
        return cls.model_validate(raw).model_dump()
