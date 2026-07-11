"""Data pipeline SLO section for health.json."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DATA_PIPELINE_SLO_EXCEPTIONS",
    "build_data_pipeline_slo_section",
    "data_pipeline_slo_unavailable_payload",
]

DATA_PIPELINE_SLO_EXCEPTIONS = (
    ImportError,
    OSError,
    ValueError,
    TypeError,
)

DATA_PIPELINE_SLO_SCHEMA_VERSION = "data-pipeline-slo/v1"


def data_pipeline_slo_unavailable_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": DATA_PIPELINE_SLO_SCHEMA_VERSION,
        "status": "warning",
        "top_dimension": "unknown",
        "error": str(exc),
    }


def build_data_pipeline_slo_section(
    *,
    health_data: dict[str, Any],
    public_dir: Path,
    log_error: Callable[[str, Exception], None] | None = None,
) -> dict[str, Any]:
    """Assemble the data pipeline SLO summary from dashboard artifacts."""
    try:
        from src.monitor.data_pipeline_slo import (
            build_data_pipeline_slo,
            load_data_quality_report,
            load_public_index,
            load_rebalance_health,
            load_signal_staleness,
            load_source_manifest,
        )

        rebalance_health = load_rebalance_health(public_dir)

        return build_data_pipeline_slo(
            health_data=health_data,
            source_manifest=load_source_manifest(public_dir),
            data_quality_report=load_data_quality_report(public_dir),
            public_index=load_public_index(public_dir),
            signal_staleness=load_signal_staleness(public_dir),
            alpaca_feed_entitlement=rebalance_health.get("alpaca_feed_entitlement"),
            market_data_consistency=rebalance_health.get("market_data_consistency"),
        )
    except DATA_PIPELINE_SLO_EXCEPTIONS as exc:
        if log_error:
            log_error("data_pipeline_slo", exc)
        else:
            logger.warning("Data pipeline SLO not available: %s", exc)
        return data_pipeline_slo_unavailable_payload(exc)
