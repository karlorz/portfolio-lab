"""Signal health and FRED readiness sections for health.json."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "SIGNAL_HEALTH_EXCEPTIONS",
    "build_fred_readiness_section",
    "build_signal_health_section",
    "fred_readiness_unavailable_payload",
    "signal_health_unavailable_payload",
]

SIGNAL_HEALTH_EXCEPTIONS = (
    ImportError,
    AttributeError,
    KeyError,
    ValueError,
    TypeError,
    RuntimeError,
    OSError,
)


def signal_health_unavailable_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "error": f"Failed to get signal health: {exc}",
        "status": "unavailable",
    }


def fred_readiness_unavailable_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "status": "warning",
        "readiness": "unknown",
        "ready": True,
        "blocking": False,
        "reason": "readiness_check_unavailable",
        "message": f"FRED readiness check unavailable: {exc}",
        "remediation": "Verify fredapi availability and FRED readiness dependencies.",
    }


def build_signal_health_section(
    *,
    log_error: Callable[[str, Exception], None] | None = None,
) -> dict[str, Any]:
    """Load SignalHealthTracker report for health.json."""
    try:
        from src.signals.health_tracker import SignalHealthTracker

        report = SignalHealthTracker().get_health_report()
        return {
            "timestamp": report.get("timestamp"),
            "summary": report.get("summary", {}),
            "scores": report.get("scores", {}),
            "alerts": report.get("alerts", []),
            "overall_health": report.get("overall_health", "unknown"),
            "status": report.get("status", report.get("overall_health", "unknown")),
            "label_horizon": report.get("label_horizon"),
        }
    except SIGNAL_HEALTH_EXCEPTIONS as exc:
        if log_error:
            log_error("signal_health", exc)
        else:
            logger.warning("Signal health not available: %s", exc)
        return signal_health_unavailable_payload(exc)


def build_fred_readiness_section(
    *,
    log_error: Callable[[str, Exception], None] | None = None,
) -> dict[str, Any]:
    """Assess FRED credential/cache readiness for health.json."""
    try:
        from src.data.fred_data import get_fred_md_cache_health
        from src.monitor.fred_readiness import assess_fred_readiness

        return assess_fred_readiness(get_fred_md_cache_health())
    except SIGNAL_HEALTH_EXCEPTIONS as exc:
        if log_error:
            log_error("fred_readiness", exc)
        else:
            logger.warning("FRED readiness not available: %s", exc)
        return fred_readiness_unavailable_payload(exc)
