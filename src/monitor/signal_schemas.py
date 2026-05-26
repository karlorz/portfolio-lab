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


# ─────────────────────────────────────────────────────────────
#  Signal model registry
# ─────────────────────────────────────────────────────────────

SIGNAL_MODELS: Dict[str, type[BaseModel]] = {
    "ensemble_voting": EnsembleVotingSignal,
    "garch_cvar": GarchCvarSignal,
    "smart_rebalance": SmartRebalanceSignal,
    "regime": RegimeSignal,
    "yield_curve": YieldCurveSignal,
    "signal_snapshot": SignalSnapshotSchema,
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

    @classmethod
    def validate_dict(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a raw signals.json dict, returning a cleaned dict.

        Known sections are validated against their schemas; unknown keys
        are passed through unchanged.  Individual section failures produce a
        warning log but never raise.
        """
        return cls.model_validate(raw).model_dump()
