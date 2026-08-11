"""Stacking / signal-section mixin extracted from ``src.dashboard.generator``.

Class-level cluster C4 (5 methods: stacking feature-count metadata, dormant
and model-backed stacking dashboards, optional signal sections, signal
postprocessors) moved here by Item 24 (2026-08-12). ``DashboardGenerator``
inherits ``_StackingSectionsMixin``. datetime.now/timezone deferred through
the generator module (FakeDateTime patch seam, rule 136e2d9);
``DashboardGenerator._build_stacking_feature_count_metadata`` resolved via
call-time lazy import (class-qualified ref rule, 08-11 18:55Z);
``_get_signal_section_builder`` stays on the owner and resolves via
inheritance.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class _StackingSectionsMixin:
    @staticmethod
    def _build_stacking_feature_count_metadata(integrator: Any) -> Dict[str, Any]:
        """Expose stacking feature count only when loaded model metadata backs it."""
        metadata = getattr(integrator, "metadata", None)
        feature_count = getattr(metadata, "feature_count", None)
        model_loaded = getattr(integrator, "model", None) is not None

        if model_loaded and feature_count is not None:
            return {
                "feature_count": int(feature_count),
                "feature_count_metadata_available": True,
                "feature_count_source": "model_metadata",
                "source_roster": list(getattr(metadata, "source_roster", [])),
                "source_roster_version": getattr(
                    metadata,
                    "source_roster_version",
                    "unavailable_missing_metadata",
                ),
                "fallback_semantics": getattr(
                    metadata,
                    "fallback_semantics",
                    "unavailable_missing_metadata",
                ),
            }

        return {
            "feature_count": None,
            "feature_count_metadata_available": False,
            "feature_count_source": (
                "unavailable_missing_metadata" if model_loaded else "unavailable_no_model"
            ),
            "source_roster": [],
            "source_roster_version": (
                "unavailable_missing_metadata" if model_loaded else "unavailable_no_model"
            ),
            "fallback_semantics": (
                "unavailable_missing_metadata" if model_loaded else "no_model_feature_count_unavailable"
            ),
        }

    @staticmethod
    def _build_stacking_no_model_dashboard(integrator: Any) -> Dict[str, Any]:
        """Build the explicit dormant stacking artifact when no model is loaded."""
        from src.dashboard import generator as _generator  # lazy (patch seams)
        from src.dashboard.generator import DashboardGenerator  # lazy (class-qualified ref rule)
        now_ts = _generator.datetime.now(_generator.timezone.utc).isoformat()
        return {
            "active": False,
            "stacking_available": False,
            "runtime_role": "research_dormant",
            "runtime_status": "unavailable_no_model",
            "live_authoritative": False,
            "routed": False,
            "routed_by": None,
            "prediction_available": False,
            "prediction_direction": "unavailable",
            "confidence": 0.0,
            "probability_bullish": 0.0,
            "probability_bearish": 0.0,
            "probability_neutral": 0.0,
            "fallback_used": False,
            "model_version": "unavailable_no_model",
            "voting_accuracy": None,
            "stacking_accuracy": None,
            "accuracy_metrics_available": False,
            **DashboardGenerator._build_stacking_feature_count_metadata(integrator),
            "latency_ms": 0.0,
            "status_reason": (
                "No stacking model artifact is loaded and no runtime base-signal "
                "input path is available."
            ),
            "operator_message": (
                "Stacking ensemble is research/dormant, not live-authoritative, "
                "and not order-routed."
            ),
            "timestamp": now_ts,
            "generated_at": now_ts,
        }

    @staticmethod
    def _build_stacking_model_dashboard(integrator: Any, prediction: Any) -> Dict[str, Any]:
        """Build the model-backed stacking dashboard artifact."""
        from src.dashboard import generator as _generator  # lazy (patch seams)
        from src.dashboard.generator import DashboardGenerator  # lazy (class-qualified ref rule)
        now_ts = _generator.datetime.now(_generator.timezone.utc).isoformat()
        return {
            "active": True,
            "stacking_available": True,
            "runtime_role": "model_backed_advisory",
            "runtime_status": "model_loaded",
            "live_authoritative": False,
            "routed": False,
            "routed_by": None,
            "prediction_available": prediction is not None,
            "prediction_direction": prediction.direction if prediction else "unavailable",
            "confidence": prediction.confidence if prediction else 0.0,
            "probability_bullish": prediction.probability_bullish if prediction else 0.0,
            "probability_bearish": prediction.probability_bearish if prediction else 0.0,
            "probability_neutral": prediction.probability_neutral if prediction else 0.0,
            "fallback_used": prediction.fallback_used if prediction else False,
            "model_version": prediction.model_version if prediction else "unknown",
            "voting_accuracy": 0.65,
            "stacking_accuracy": 0.76,
            "accuracy_metrics_available": True,
            **DashboardGenerator._build_stacking_feature_count_metadata(integrator),
            "latency_ms": prediction.latency_ms if prediction else 0.0,
            "status_reason": "Stacking model artifact is loaded for advisory inference.",
            "operator_message": (
                "Stacking ensemble is advisory and not order-routed; live routing "
                "still consumes target_allocations."
            ),
            "backtest_finding": (
                "+11% accuracy produces negligible Sharpe gain (2021-2026). "
                "Signal frequency and shift magnitude are binding constraints."
            ),
            "timestamp": now_ts,
            "generated_at": now_ts,
        }

    def _build_optional_signal_sections(
        self,
        output: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Append optional operational sections that precede staleness checks."""
        return self._get_signal_section_builder().build_optional_sections(
            output,
            context,
        )

    def _apply_signal_postprocessors(
        self,
        output: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply staleness, monitoring, alerting, and final signal appenders."""
        return self._get_signal_section_builder().apply_postprocessors(
            output,
            context,
        )
