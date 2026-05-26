"""Shared Python utilities for portfolio-lab."""

from __future__ import annotations

from src.paths import VIX_CRISIS_THRESHOLD, VIX_VOL_SPIKE_THRESHOLD, VIX_LOW_VOL_THRESHOLD


def safe_get(data: dict, *keys, default=None):
    """Navigate nested dicts safely.

    Unlike ``d.get('x', {}).get('y')``, this function returns *default*
    when any intermediate value is ``None`` (not just when the key is
    missing).  The chained-.get() pattern raises ``AttributeError`` on
    ``None`` because ``NoneType`` has no ``.get`` attribute.

    Args:
        data: Root dict to traverse.
        *keys: Sequence of keys to follow.
        default: Value returned when any key is missing or maps to ``None``.

    Returns:
        The value at the leaf key, or *default* if traversal fails.

    Examples:
        >>> safe_get({"a": {"b": 1}}, "a", "b")
        1
        >>> safe_get({"a": None}, "a", "b")
        None
        >>> safe_get({"a": None}, "a", "b", default=0)
        0
        >>> safe_get({"a": {"b": {"c": 3}}}, "a", "b", "c")
        3
        >>> safe_get({}, "a", "b", default="missing")
        'missing'
    """
    result = data
    for key in keys:
        if not isinstance(result, dict):
            return default
        result = result.get(key)
        if result is None:
            return default
    return result


def classify_vix_regime(vix_level: float | None, trend_regime: str = "normal") -> str:
    """Classify the current market regime from VIX level and trend signal.

    Uses configurable thresholds from src.paths (VIX_CRISIS_THRESHOLD,
    VIX_VOL_SPIKE_THRESHOLD, VIX_LOW_VOL_THRESHOLD) with composite logic:
    - VIX crisis/vol_spike always overrides trend (market fear is immediate)
    - VIX low_vol requires trend confirmation (avoid false calm)
    - Fallback to trend_regime when VIX unavailable

    Args:
        vix_level: Current VIX close price, or None if unavailable.
        trend_regime: Regime from trend analysis (e.g., from regime_log).

    Returns:
        One of: "crisis", "vol_spike", "low_vol", "normal".
    """
    if vix_level is None:
        return trend_regime

    if vix_level > VIX_CRISIS_THRESHOLD:
        vix_regime = "crisis"
    elif vix_level > VIX_VOL_SPIKE_THRESHOLD:
        vix_regime = "vol_spike"
    elif vix_level < VIX_LOW_VOL_THRESHOLD:
        vix_regime = "low_vol"
    else:
        vix_regime = "normal"

    # Composite: VIX overrides trend in extreme cases
    if vix_regime in ("crisis", "vol_spike"):
        return vix_regime
    elif vix_regime == "low_vol" and trend_regime != "crisis":
        return "low_vol"

    return trend_regime
