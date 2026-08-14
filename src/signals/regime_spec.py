"""Regime specification: regime enum, signal reading, and regime weights.

Extracted from the ensemble voter (A1-CYCLE-BREAK s2) to break the
signals -> strategy import edge. The voter re-exports these names; the loader
still reads DATA_DIR/ensemble_weights.json (ENSEMBLE_WEIGHTS_FILE override)
exactly as before.
"""

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.paths import DATA_DIR
from src.signals.signal_source import SignalSource


logger = logging.getLogger(__name__)


class Regime(Enum):
    """Market regime classifications."""
    LOW_VOL = "low_vol"      # VIX < 15, calm bull market
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"


@dataclass
class SignalReading:
    """Single signal source reading."""
    source: SignalSource
    timestamp: str
    
    # Signal value: -1 (strong short) to +1 (strong long)
    value: float
    
    # Metadata
    confidence: float  # 0-1
    weight: float    # Dynamic regime weight
    regime_fit: str  # Which regime this signal works best in
    
    # Asset-specific signals (optional)
    asset_signals: Optional[Dict[str, float]] = None
    
    # Reasoning
    explanation: str = ""
    # Batch CV: inactive readings are kept for disclosure but vote weight forced 0
    is_active: bool = True
    # Batch DF: provenance for health tracker (pattern, polarity_policy, composite, …)
    metadata: Optional[Dict[str, Any]] = None



# ``ensemble_weights.json`` is a flat business map for the five Regime values,
# with additive runtime provenance emitted by the shared JSON serializer. Keep
# this allowlist local to the consumer so adding a producer diagnostic does not
# silently promote it into a business regime.
ENSEMBLE_WEIGHT_METADATA_KEYS = frozenset(
    {
        "artifact_id",
        "plane",
        "generated_at",
        "timestamp",
        "updated_at",
        "created_at",
        "checked_at",
        "last_updated",
        "reconciled_at",
        "ssot_reconciled_at",
        "ssot_reconcile_source",
        "generator_git_sha",
        "generator_git_sha_status",
        "last_full_generator_git_sha",
        "generator_git_sha_reason",
        "patch_source",
        "content_patch_source",
        "runtime_provenance",
        "provenance_completeness",
        "schema_version",
        "private_path",
        "public_path",
        "private_mtime",
        "public_mtime",
        "private_content_hash",
        "public_content_hash",
        "dual_write_lag_seconds",
        "dual_write_lag_stale",
        "dual_write_lag_threshold_seconds",
        "dual_write_lag_unit",
        "repo_public_mirror_source",
        "repo_public_mirror_dest",
        "repo_public_mirror_lag",
        "repo_public_mirror_lagging_count",
        "repo_public_mirror_total",
        "mirror_lag_restamped_at",
        "private_health_report",
    }
)


def _normalize_weights(
    weights: Dict[SignalSource, float],
) -> Dict[SignalSource, float]:
    """Scale a regime map to a total of exactly 1.0, preserving pairwise ratios.

    The hardcoded tables carry legacy relative values; appending
    VIX_TERM_STRUCTURE (83a56eb) left each map summing to 1.05. Normalizing
    at build time restores the sum-to-1.0 validity contract of the
    JSON-loaded path without hand-rounding five tables, and stays correct
    if future sources are added.
    """
    total = sum(weights.values())
    if total <= 0.0:
        return weights
    return {source: weight / total for source, weight in weights.items()}


def _build_hardcoded_weights() -> Dict[Regime, Dict[SignalSource, float]]:
    """Return the hardcoded default regime weights (fallback).

    Tables list the legacy relative values (pre-VIX-term maps summed to 1.0;
    VIX_TERM_STRUCTURE 0.05 was appended without scaling down). Each map is
    normalized to total 1.0 so the fallback satisfies the same validity
    contract as the JSON-loaded path.
    """
    return {
        Regime.LOW_VOL: _normalize_weights({
            SignalSource.MULTI_SPEED_MOM: 0.0000,
            SignalSource.CROSS_ASSET_RV: 0.1350,
            SignalSource.ALTERNATIVE_DATA: 0.2650,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.2520,
            SignalSource.CROSS_ASSET_REGIME_ARB: 0.0000,  # marginal in calm markets
            SignalSource.UNIFIED_OVERLAY: 0.1980,
            SignalSource.MULTI_TIMEFRAME_FUSION: 0.1000,
            SignalSource.GOOGLE_TRENDS: 0.0500,
            SignalSource.VIX_TERM_STRUCTURE: 0.0500,  # v3.23: intraday vol timing
        }),
        Regime.NORMAL: _normalize_weights({
            SignalSource.MULTI_SPEED_MOM: 0.0000,
            SignalSource.CROSS_ASSET_RV: 0.1170,
            SignalSource.ALTERNATIVE_DATA: 0.2245,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.2205,
            SignalSource.CROSS_ASSET_REGIME_ARB: 0.1170,
            SignalSource.UNIFIED_OVERLAY: 0.1710,
            SignalSource.MULTI_TIMEFRAME_FUSION: 0.1000,
            SignalSource.GOOGLE_TRENDS: 0.0500,
            SignalSource.VIX_TERM_STRUCTURE: 0.0500,  # v3.23: intraday vol timing
        }),
        Regime.HIGH_VOL: _normalize_weights({
            SignalSource.MULTI_SPEED_MOM: 0.0000,
            SignalSource.CROSS_ASSET_RV: 0.1170,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.1890,
            SignalSource.ALTERNATIVE_DATA: 0.2470,
            SignalSource.CROSS_ASSET_REGIME_ARB: 0.1170,
            SignalSource.UNIFIED_OVERLAY: 0.1800,
            SignalSource.MULTI_TIMEFRAME_FUSION: 0.1000,
            SignalSource.GOOGLE_TRENDS: 0.0500,
            SignalSource.VIX_TERM_STRUCTURE: 0.0500,  # v3.23: intraday vol timing
        }),
        Regime.CRISIS: _normalize_weights({
            SignalSource.MULTI_SPEED_MOM: 0.0000,
            SignalSource.CROSS_ASSET_RV: 0.3285,
            SignalSource.CROSS_ASSET_REGIME_ARB: 0.1530,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.0000,
            SignalSource.ALTERNATIVE_DATA: 0.1300,
            SignalSource.UNIFIED_OVERLAY: 0.2385,
            SignalSource.MULTI_TIMEFRAME_FUSION: 0.1000,
            SignalSource.GOOGLE_TRENDS: 0.0500,
            SignalSource.VIX_TERM_STRUCTURE: 0.0500,  # v3.23: intraday vol timing
        }),
        Regime.RECOVERY: _normalize_weights({
            SignalSource.MULTI_SPEED_MOM: 0.0000,
            SignalSource.ALTERNATIVE_DATA: 0.2245,
            SignalSource.CROSS_ASSET_RV: 0.1170,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.2205,
            SignalSource.CROSS_ASSET_REGIME_ARB: 0.1170,
            SignalSource.UNIFIED_OVERLAY: 0.1710,
            SignalSource.MULTI_TIMEFRAME_FUSION: 0.1000,
            SignalSource.GOOGLE_TRENDS: 0.0500,
            SignalSource.VIX_TERM_STRUCTURE: 0.0500,  # v3.23: intraday vol timing
        }),
    }


def _extract_ensemble_business_payload(
    raw: Any,
    weights_path: Path,
) -> Dict[str, Dict[str, Any]] | None:
    """Separate regime maps from known additive runtime metadata.

    Unknown non-metadata keys are treated as schema drift and fail closed to
    the existing hardcoded weights. This keeps diagnostics visible while
    preventing an arbitrary object from becoming a new business regime.
    """
    if not isinstance(raw, dict):
        logger.warning(
            "Invalid ensemble weights payload in %s: expected an object, using hardcoded defaults",
            weights_path,
        )
        return None

    regime_names = {regime.value for regime in Regime}
    business: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        key_str = str(key)
        if key_str.startswith("_") or key_str in ENSEMBLE_WEIGHT_METADATA_KEYS:
            continue
        if key_str not in regime_names:
            logger.warning(
                "Unknown ensemble weight key '%s' in %s; using hardcoded defaults",
                key_str,
                weights_path,
            )
            return None
        if not isinstance(value, dict):
            logger.warning(
                "Invalid regime map '%s' in %s: expected an object, using hardcoded defaults",
                key_str,
                weights_path,
            )
            return None
        business[key_str] = value
    return business


def _load_regime_weights(
    weights_file: Optional[str] = None,
) -> Dict[Regime, Dict[SignalSource, float]]:
    """Load REGIME_WEIGHTS from JSON config file.

    Supports ENSEMBLE_WEIGHTS_FILE env var override (same pattern as
    PAPER_CONFIG in evaluator.py). Falls back to hardcoded defaults
    if the file doesn't exist or contains invalid data.
    """
    if weights_file is None:
        weights_file = os.environ.get(
            "ENSEMBLE_WEIGHTS_FILE",
            str(DATA_DIR / "ensemble_weights.json")
        )
    weights_path = Path(weights_file)

    if not weights_path.exists():
        logger.info(
            "Ensemble weights file not found at %s, using hardcoded defaults",
            weights_path
        )
        return _build_hardcoded_weights()

    try:
        with open(weights_path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(
            "Failed to load ensemble weights from %s: %s, using hardcoded defaults",
            weights_path, e
        )
        return _build_hardcoded_weights()

    business = _extract_ensemble_business_payload(raw, weights_path)
    if business is None:
        return _build_hardcoded_weights()

    expected_regimes = {regime.value for regime in Regime}
    if set(business) != expected_regimes:
        logger.warning(
            "Invalid ensemble regime set in %s: missing=%s extra=%s; using hardcoded defaults",
            weights_path,
            sorted(expected_regimes - set(business)),
            sorted(set(business) - expected_regimes),
        )
        return _build_hardcoded_weights()

    expected_sources = {source.value for source in SignalSource}
    regime_weights: Dict[Regime, Dict[SignalSource, float]] = {}
    for regime_name, sources in business.items():
        try:
            regime = Regime(regime_name)
        except ValueError:
            # The extraction boundary above owns this diagnostic. Keep this
            # branch defensive if Regime changes between the two operations.
            logger.warning("Unknown ensemble regime '%s' in %s", regime_name, weights_path)
            return _build_hardcoded_weights()

        source_names = {str(source_name) for source_name in sources}
        if source_names != expected_sources:
            logger.warning(
                "Invalid source map for regime '%s' in %s: missing=%s extra=%s; "
                "using hardcoded defaults",
                regime_name,
                weights_path,
                sorted(expected_sources - source_names),
                sorted(source_names - expected_sources),
            )
            return _build_hardcoded_weights()

        regime_dict: Dict[SignalSource, float] = {}
        for source_name, weight in sources.items():
            try:
                source = SignalSource(source_name)
                numeric_weight = float(weight)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid signal source weight '%s': %r in %s; using hardcoded defaults",
                    source_name,
                    weight,
                    weights_path,
                )
                return _build_hardcoded_weights()
            if not np.isfinite(numeric_weight) or numeric_weight < 0:
                logger.warning(
                    "Invalid signal source weight '%s': %r in %s; using hardcoded defaults",
                    source_name,
                    weight,
                    weights_path,
                )
                return _build_hardcoded_weights()
            regime_dict[source] = numeric_weight

        regime_weights[regime] = regime_dict

    # Validate: all regimes should be present
    missing = [r.value for r in Regime if r not in regime_weights]
    if missing:
        logger.warning(
            "Missing regimes in %s: %s, falling back to hardcoded defaults",
            weights_path, missing
        )
        return _build_hardcoded_weights()

    logger.info(
        "Loaded ensemble weights from %s (%d regimes)",
        weights_path, len(regime_weights)
    )
    return regime_weights


REGIME_WEIGHTS = _load_regime_weights()
