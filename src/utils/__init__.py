"""Shared Python utilities for portfolio-lab."""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any, Callable

from src.paths import VIX_CRISIS_THRESHOLD, VIX_VOL_SPIKE_THRESHOLD, VIX_LOW_VOL_THRESHOLD

logger = logging.getLogger(__name__)

__all__ = ["safe_get", "classify_vix_regime", "signal_timeout"]


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


class _TimeoutError(Exception):
    """Internal timeout exception for signal_timeout decorator."""


def signal_timeout(
    default: Any = None,
    seconds: float = 30.0,
    signal_name: str = "",
) -> Callable:
    """Decorator that wraps a function with a timeout and fallback default.

    If the decorated function takes longer than *seconds* to complete, or
    raises an exception, the decorator returns *default* instead. This
    provides graceful degradation for signal generators — a slow or
    crashing signal gets replaced by a neutral value rather than stalling
    or killing the entire pipeline.

    Uses ``threading.Thread`` with ``join(timeout=)`` to implement the
    timeout. This is lighter than subprocess isolation and sufficient
    for CPU-bound signal computations that might hang. It does NOT
    provide OS-level isolation for segfaults — use
    ``multiprocessing.Process`` for that.

    Args:
        default: Value to return on timeout or exception.
        seconds: Maximum allowed execution time in seconds.
        signal_name: Optional name for logging (defaults to function name).

    Returns:
        Decorator that wraps the function with timeout + fallback.

    Examples:
        >>> @signal_timeout(default=None, seconds=10, signal_name="garch_cvar")
        ... def compute_garch():
        ...     # expensive computation
        ...     return {"volatility": 0.15}
        >>> result = compute_garch()  # returns None if >10s or exception
    """
    def decorator(func: Callable) -> Callable:
        name = signal_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            result_container: dict = {"result": default, "done": False, "exc": None}

            def target():
                try:
                    result_container["result"] = func(*args, **kwargs)
                    result_container["done"] = True
                except Exception as e:
                    result_container["exc"] = e

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=seconds)

            if thread.is_alive():
                # Thread is still running — it timed out
                logger.warning("Signal %s timed out after %.1fs", name, seconds)
                return default

            if result_container["exc"] is not None:
                logger.warning("Signal %s failed: %s", name, result_container["exc"])
                return default

            return result_container["result"]

        return wrapper
    return decorator
