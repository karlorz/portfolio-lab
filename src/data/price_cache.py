#!/usr/bin/env python3
"""
TTL-cached access to prices.json — single source of truth.

Multiple signal modules read prices.json independently each cron cycle,
causing redundant 2.4MB file reads. This module provides a shared
TTL-cached accessor so only one actual read occurs per TTL window.

Usage:
    from src.data.price_cache import get_prices, invalidate_price_cache

    data = get_prices()               # cached for up to 30 seconds
    invalidate_price_cache()          # force refresh on next call
"""

import json
import os
import threading
from typing import Dict, Any

from cachetools import TTLCache

from src.paths import PRICES_JSON

# TTL configurable via env var; 30s covers a full cron cycle
_PRICE_CACHE_TTL = int(os.environ.get("PRICE_CACHE_TTL_SECONDS", "30"))

# Module-level singleton — initialized once, survives across calls
_PRICE_CACHE: TTLCache = TTLCache(maxsize=1, ttl=_PRICE_CACHE_TTL)
_PRICE_CACHE_LOCK = threading.Lock()


def get_prices() -> Dict[str, Any]:
    """Return prices.json data with up to N-second stale tolerance.

    Thread-safe: the file I/O happens outside the lock; only the
    cache read/write is synchronized.
    """
    with _PRICE_CACHE_LOCK:
        if "prices" in _PRICE_CACHE:
            return _PRICE_CACHE["prices"]

    with open(PRICES_JSON) as f:
        data = json.load(f)

    with _PRICE_CACHE_LOCK:
        _PRICE_CACHE["prices"] = data

    return data


def invalidate_price_cache() -> None:
    """Force cache refresh on next call (for testing or manual trigger)."""
    with _PRICE_CACHE_LOCK:
        _PRICE_CACHE.pop("prices", None)
