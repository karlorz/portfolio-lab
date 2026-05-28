"""Regime-conditional base allocation.

The champion allocation SPY/GLD/TLT 46/38/16 is static across all 5 regimes.
This module varies weights by market regime to improve risk-adjusted returns.

Research-backed defaults:
- NORMAL: 46/38/16 (current champion)
- CRISIS: 40/42/18 (more gold + bonds in crisis)
- HIGH_VOL: 42/40/18 (slight defensive tilt)
- LOW_VOL: 50/34/16 (more equities in calm markets)
- RECOVERY: 52/32/16 (maximize equity upside in recovery)

Override via REGIME_ALLOC_OVERRIDE env var (JSON dict of regime -> weights).
"""

import json
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

__all__ = [
    "REGIME_ALLOCATIONS",
    "DEFAULT_ALLOCATION",
    "get_regime_allocation",
    "get_regime_allocation_with_override",
    "validate_allocations",
]

# Champion baseline
DEFAULT_ALLOCATION: Dict[str, float] = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}

# Regime-conditional allocations (research-backed defaults)
REGIME_ALLOCATIONS: Dict[str, Dict[str, float]] = {
    "normal": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
    "crisis": {"SPY": 0.40, "GLD": 0.42, "TLT": 0.18},
    "high_vol": {"SPY": 0.42, "GLD": 0.40, "TLT": 0.18},
    "low_vol": {"SPY": 0.50, "GLD": 0.34, "TLT": 0.16},
    "recovery": {"SPY": 0.52, "GLD": 0.32, "TLT": 0.16},
}


def get_regime_allocation(regime: str | None) -> Dict[str, float]:
    """Get allocation weights for a regime.

    Args:
        regime: Market regime name (case-insensitive). Falls back to NORMAL
                if unknown or None.

    Returns:
        Dict of {asset: weight} summing to 1.0.
    """
    if not regime:
        return dict(REGIME_ALLOCATIONS["normal"])
    key = regime.lower().strip()
    if key in REGIME_ALLOCATIONS:
        return dict(REGIME_ALLOCATIONS[key])
    logger.warning("Unknown regime '%s', falling back to NORMAL", regime)
    return dict(REGIME_ALLOCATIONS["normal"])


def get_regime_allocation_with_override(regime: str | None) -> Dict[str, float]:
    """Get allocation with env-var override support.

    REGIME_ALLOC_OVERRIDE env var can contain JSON overriding specific regimes.
    Example: {"crisis": {"SPY": 0.35, "GLD": 0.45, "TLT": 0.20}}

    Args:
        regime: Market regime name.

    Returns:
        Dict of {asset: weight} summing to 1.0.
    """
    base = get_regime_allocation(regime)

    override_json = os.environ.get("REGIME_ALLOC_OVERRIDE")
    if not override_json:
        return base

    try:
        overrides = json.loads(override_json)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Malformed REGIME_ALLOC_OVERRIDE, ignoring")
        return base

    if not regime:
        return base

    key = regime.lower().strip()
    if key not in overrides:
        return base

    regime_override = overrides[key]
    if not isinstance(regime_override, dict):
        return base

    # Validate required assets
    required = {"SPY", "GLD", "TLT"}
    if not required.issubset(set(regime_override.keys())):
        logger.warning("Override for '%s' missing assets, ignoring", key)
        return base

    # Normalize if not summing to 1.0
    total = sum(regime_override.values())
    if abs(total - 1.0) > 1e-6 and total > 0:
        regime_override = {k: v / total for k, v in regime_override.items()}
        logger.info("Normalized '%s' override to sum to 1.0", key)

    return dict(regime_override)


def validate_allocations(allocations: Dict[str, Dict[str, float]]) -> List[str]:
    """Validate allocation dict structure and values.

    Args:
        allocations: Dict of {regime: {asset: weight}}.

    Returns:
        List of error strings. Empty list means valid.
    """
    errors = []
    required = {"SPY", "GLD", "TLT"}

    for regime, weights in allocations.items():
        # Check assets
        if not required.issubset(set(weights.keys())):
            missing = required - set(weights.keys())
            errors.append(f"{regime}: missing assets {missing}")

        # Check sum
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            errors.append(f"{regime}: weights sum to {total:.6f}, expected 1.0")

        # Check bounds
        for asset, w in weights.items():
            if w < 0:
                errors.append(f"{regime}/{asset}: negative weight {w}")
            if w > 1.0:
                errors.append(f"{regime}/{asset}: weight {w} > 1.0")

    return errors
