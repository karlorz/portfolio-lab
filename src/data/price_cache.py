#!/usr/bin/env python3
"""
TTL-cached access to prices.json — single source of truth.

Multiple signal modules read prices.json independently each cron cycle,
causing redundant 2.4MB file reads. This module provides a shared
TTL-cached accessor so only one actual read occurs per TTL window.

Also provides get_prices_df() — a cached pivoted DataFrame accessor
that eliminates the duplicated pivot/dedup pattern across 8+ modules.

Usage:
    from src.data.price_cache import get_prices, get_prices_df, invalidate_price_cache

    data = get_prices()               # raw dict, cached for up to 30 seconds
    df = get_prices_df()              # pivoted DataFrame, cached separately
    df_spy = get_prices_df(symbols=["SPY", "GLD"])  # subset only
    invalidate_price_cache()          # force refresh on next call
"""

import json
import os
import threading
from typing import Dict, Any, List, Optional

from cachetools import TTLCache

from src.paths import PRICES_JSON

# TTL configurable via env var; 30s covers a full cron cycle
_PRICE_CACHE_TTL = int(os.environ.get("PRICE_CACHE_TTL_SECONDS", "30"))

# Module-level singleton — initialized once, survives across calls
_PRICE_CACHE: TTLCache = TTLCache(maxsize=1, ttl=_PRICE_CACHE_TTL)
_PRICE_CACHE_LOCK = threading.Lock()

# Separate cache for DataFrame views — keyed by frozenset(symbols) or None
_DF_CACHE: TTLCache = TTLCache(maxsize=4, ttl=_PRICE_CACHE_TTL)
_DF_CACHE_LOCK = threading.Lock()


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


def get_prices_df(symbols: Optional[List[str]] = None) -> "pandas.DataFrame":  # noqa: F821  # function-local import pandas below (hot-path cost, intentional)
    """Return a pivoted DataFrame of close prices, cached by symbol subset.

    Converts the raw {symbol: [{d, p}, ...]} dict into a DataFrame with
    dates as index and symbols as columns. Returns a copy to prevent
    mutation of the cached version.

    Args:
        symbols: Optional list of symbols to include. If None, includes all.
                 Useful when only a few symbols are needed (e.g., ["SPY","GLD","TLT"]
                 is ~130KB vs ~1.6MB for the full DataFrame).

    Returns:
        pandas.DataFrame with DatetimeIndex and float columns.

    Raises:
        FileNotFoundError: If prices.json doesn't exist.
    """
    import pandas as pd

    cache_key = frozenset(symbols) if symbols else None

    with _DF_CACHE_LOCK:
        if cache_key in _DF_CACHE:
            return _DF_CACHE[cache_key].copy()

    raw = get_prices()

    records = []
    for sym, entries in raw.items():
        if symbols and sym not in symbols:
            continue
        for entry in entries:
            records.append({"date": entry["d"], "ticker": sym, "price": entry["p"]})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.pivot(index="date", columns="ticker", values="price")
    df = df.sort_index()

    with _DF_CACHE_LOCK:
        _DF_CACHE[cache_key] = df

    return df.copy()


def invalidate_price_cache() -> None:
    """Force cache refresh on next call (for testing or manual trigger)."""
    with _PRICE_CACHE_LOCK:
        _PRICE_CACHE.pop("prices", None)
    with _DF_CACHE_LOCK:
        _DF_CACHE.clear()
