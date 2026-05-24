"""Shared Python utilities for portfolio-lab."""

from __future__ import annotations


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
