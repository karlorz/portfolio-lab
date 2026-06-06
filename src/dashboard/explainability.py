"""Portfolio explainability data builder.

Converts the ensemble voting section of ``signals.json`` into a dashboard-ready
explainability payload.  The builder is intentionally pure so cron generation,
tests, and fallback JSON creation all use the same contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_confidence(value: Any) -> float:
    confidence = _safe_float(value, 0.0)
    if confidence > 1.0:
        confidence = confidence / 100.0
    return max(0.0, min(confidence, 1.0))


def _signed_value(signal: Dict[str, Any]) -> float:
    if "value" in signal:
        return _safe_float(signal.get("value"))

    strength = abs(_safe_float(signal.get("strength")))
    direction = str(signal.get("direction", "neutral")).lower()
    if direction == "bearish":
        return -strength
    if direction == "bullish":
        return strength
    return 0.0


def _direction(value: float) -> str:
    if value > 0:
        return "bullish"
    if value < 0:
        return "bearish"
    return "neutral"


def _display_name(source: str) -> str:
    return source.replace("_", " ").title()


def _normalize_sources(source_breakdown: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for raw in source_breakdown:
        if not isinstance(raw, dict):
            continue

        source = str(raw.get("source", "unknown"))
        value = _signed_value(raw)
        weight = max(0.0, _safe_float(raw.get("weight")))
        contribution = round(value * weight, 6)
        normalized.append({
            "source": source,
            "display_name": str(raw.get("display_name") or _display_name(source)),
            "category": str(raw.get("category") or "signal"),
            "value": round(value, 6),
            "direction": _direction(value),
            "strength": round(abs(value), 6),
            "confidence": round(_normalize_confidence(raw.get("confidence")), 6),
            "weight": round(weight, 6),
            "contribution": contribution,
        })

    normalized.sort(key=lambda row: abs(row["contribution"]), reverse=True)
    return normalized


def _top_rows(rows: List[Dict[str, Any]], positive: bool) -> List[Dict[str, Any]]:
    if positive:
        selected = [row for row in rows if row["contribution"] > 0]
        selected.sort(key=lambda row: row["contribution"], reverse=True)
    else:
        selected = [row for row in rows if row["contribution"] < 0]
        selected.sort(key=lambda row: row["contribution"])

    return [
        {
            "source": row["source"],
            "contribution": round(row["contribution"], 6),
            "direction": row["direction"],
        }
        for row in selected[:5]
    ]


def build_portfolio_explainability(
    ensemble_signal: Optional[Dict[str, Any]],
    *,
    analysis_date: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a dashboard explainability payload from an ensemble vote dict."""
    generated_at = timestamp or datetime.now().isoformat()
    analysis_day = analysis_date or generated_at[:10]

    base: Dict[str, Any] = {
        "timestamp": generated_at,
        "analysis_date": analysis_day,
        "latest_decision": None,
        "recent_decisions": [],
        "signal_deep_dives": {},
        "top_sources_today": [],
        "decision_quality": {"status": "no_ensemble_data"},
    }

    if not isinstance(ensemble_signal, dict) or not ensemble_signal:
        return base

    signals = _normalize_sources(ensemble_signal.get("source_breakdown", []))
    top_signals = signals[:5]
    weighted_consensus = _safe_float(ensemble_signal.get("weighted_consensus"))
    consensus_direction = _direction(weighted_consensus)

    signal_deep_dives = {
        row["source"]: {
            "source": row["source"],
            "display_name": row["display_name"],
            "category": row["category"],
            "total_observations": 1,
            "avg_value": row["value"],
            "avg_confidence": row["confidence"],
            "avg_weight": row["weight"],
            "hit_rate": None,
            "sharpe_contribution": None,
        }
        for row in top_signals
    }

    latest_decision = {
        "timestamp": generated_at,
        "period": analysis_day,
        "regime": str(ensemble_signal.get("regime", "unknown")),
        "action": str(ensemble_signal.get("action", "neutral")),
        "confidence": _normalize_confidence(ensemble_signal.get("confidence")),
        "reasoning": str(ensemble_signal.get("reasoning", "")),
        "total_signals": int(_safe_float(ensemble_signal.get("num_sources"), len(signals))),
        "consensus_direction": consensus_direction,
        "agreement_ratio": _safe_float(ensemble_signal.get("agreement_ratio")),
        "weighted_consensus": weighted_consensus,
        "signals": top_signals,
        "top_drivers": _top_rows(signals, positive=True),
        "top_opposers": _top_rows(signals, positive=False),
    }

    base.update({
        "latest_decision": latest_decision,
        "signal_deep_dives": signal_deep_dives,
        "top_sources_today": [row["source"] for row in top_signals],
        "decision_quality": {
            "status": "ok",
            "agreement_ratio": latest_decision["agreement_ratio"],
            "n_eff": _safe_float(ensemble_signal.get("n_eff")),
            "weight_entropy": _safe_float(ensemble_signal.get("weight_entropy")),
        },
    })
    return base


def build_explainability_from_signals_data(
    signals_data: Dict[str, Any],
    *,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build explainability from a complete ``signals.json`` payload."""
    analysis_date = str(
        signals_data.get("generated_at")
        or signals_data.get("timestamp")
        or datetime.now().isoformat()
    )[:10]
    return build_portfolio_explainability(
        signals_data.get("ensemble_voting"),
        analysis_date=analysis_date,
        timestamp=timestamp,
    )
