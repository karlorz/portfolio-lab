#!/usr/bin/env python3
"""
SignalSnapshot — canonical output type for signal modules.

Every signal module should return a SignalSnapshot (or list thereof) from its
main generation function. The ensemble voter's collect_signals() can then
consume them directly without ad-hoc dict unpacking.

SignalSnapshot is a superset of SignalReading (ensemble_voter.py): it adds
metadata for diagnostics and a to_signal_reading() bridge method.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any



__all__ = ['SignalSnapshot']

@dataclass
class SignalSnapshot:
    """Canonical output type for portfolio signal modules.

    Every signal module returns this from its generate() / compute() function.
    The ensemble voter converts it to SignalReading via to_signal_reading().

    Fields align with SignalReading but add metadata for observability.
    """
    source: str                        # Signal module name (e.g. "multi_speed_momentum")
    timestamp: str                     # ISO timestamp of signal generation

    # Core signal: -1 (strong bearish) to +1 (strong bullish)
    value: float

    # Confidence: 0-1 (how reliable this signal is)
    confidence: float

    # Per-asset directional bias: {"SPY": 0.3, "GLD": -0.1, "TLT": 0.0}
    asset_signals: Dict[str, float] = field(default_factory=dict)

    # Which regime this signal works best in ("normal", "crisis", "low_vol", etc.)
    regime_fit: str = "normal"

    # Whether this signal is valid/active
    is_active: bool = True

    # Human-readable explanation
    explanation: str = ""

    # Extra module-specific data (for diagnostics, dashboards, research)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_signal_reading(self):
        """Convert to ensemble_voter.SignalReading.

        Returns a SignalReading that can be fed directly into compute_vote().
        The caller must resolve the SignalSource enum from self.source.
        """
        from src.strategy.ensemble_voter import SignalSource, SignalReading

        # Resolve by value first (source strings use lowercase values like
        # "multi_speed_momentum"), then by enum name as fallback
        source_enum = None
        for member in SignalSource:
            if member.value == self.source:
                source_enum = member
                break
        if source_enum is None:
            # Try enum name match (case-insensitive, dash-to-underscore)
            source_key = self.source.upper().replace("-", "_")
            try:
                source_enum = SignalSource[source_key]
            except KeyError:
                raise ValueError(
                    f"No SignalSource enum for '{self.source}'. "
                    f"Available values: {[m.value for m in SignalSource]}"
                )

        return SignalReading(
            source=source_enum,
            timestamp=self.timestamp,
            value=self.value,
            confidence=self.confidence,
            weight=0.0,  # Set later by regime weighting
            regime_fit=self.regime_fit,
            asset_signals=self.asset_signals if self.asset_signals else None,
            explanation=self.explanation,
            is_active=bool(self.is_active),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalSnapshot":
        """Create from a dict (legacy signal module output)."""
        return cls(
            source=data.get("source", data.get("signal_name", "unknown")),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            value=float(data.get("signal_value", data.get("value", data.get("composite", 0.0)))),
            confidence=float(data.get("confidence", data.get("overall_conviction", 0.5))),
            asset_signals=data.get("asset_signals", {}),
            regime_fit=data.get("regime_fit", "normal"),
            is_active=data.get("is_active", data.get("active", data.get("is_valid", True))),
            explanation=data.get("explanation", ""),
            metadata={k: v for k, v in data.items()
                      if k not in {"source", "signal_name", "timestamp", "signal_value",
                                   "value", "composite", "confidence", "overall_conviction",
                                   "asset_signals", "regime_fit", "is_active", "active",
                                   "is_valid", "explanation"}},
        )
