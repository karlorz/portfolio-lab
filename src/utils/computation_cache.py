"""
TTL-cached computations for shared intermediate signal values.

Extends the price_cache.py pattern to expensive computations that are
repeated across signal modules within a single cron cycle (~5 minutes).

Each cache has a TTL matching the cron interval (300s by default,
configurable via COMPUTATION_CACHE_TTL_SECONDS env var).
"""

import os
import threading
from typing import Optional

import cachetools
import numpy as np
import pandas as pd

__all__ = [
    "get_realized_volatility",
    "get_correlation_matrix",
    "get_rolling_returns",
    "invalidate_computation_cache",
]

TTL = int(os.environ.get("COMPUTATION_CACHE_TTL_SECONDS", "300"))

# Cache instances -- one per computation type
_vol_cache: cachetools.TTLCache = cachetools.TTLCache(maxsize=64, ttl=TTL)
_corr_cache: cachetools.TTLCache = cachetools.TTLCache(maxsize=32, ttl=TTL)
_returns_cache: cachetools.TTLCache = cachetools.TTLCache(maxsize=64, ttl=TTL)

# Thread safety
_vol_lock = threading.Lock()
_corr_lock = threading.Lock()
_returns_lock = threading.Lock()


def get_realized_volatility(series: pd.Series, window: int = 21) -> Optional[float]:
    """Get TTL-cached realized volatility for a price series.

    Cache key: (series.name, series.iloc[-1] if len > 0 else None, window, len(series))
    """
    if series.empty:
        return None
    key = (getattr(series, 'name', None), series.iloc[-1], window, len(series))
    with _vol_lock:
        cached = _vol_cache.get(key)
        if cached is not None:
            return cached

    # Compute
    returns = series.pct_change().dropna()
    if len(returns) < window:
        vol = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else None
    else:
        vol = float(returns.rolling(window).std().iloc[-1] * np.sqrt(252))

    with _vol_lock:
        _vol_cache[key] = vol
    return vol


def get_correlation_matrix(df: pd.DataFrame, window: int = 63) -> Optional[pd.DataFrame]:
    """Get TTL-cached rolling correlation matrix.

    Cache key: tuple of sorted column names + window + last date
    """
    if df.empty:
        return None
    key = (tuple(sorted(df.columns)), window, str(df.index[-1]))
    with _corr_lock:
        cached = _corr_cache.get(key)
        if cached is not None:
            return cached

    # Compute
    returns = df.pct_change().dropna()
    if len(returns) < window:
        result = returns.corr()
    else:
        # rolling(w).corr() returns a MultiIndex (date, ticker) DataFrame;
        # extract the clean NxN matrix for the most recent date
        all_corr = returns.rolling(window).corr()
        last_date = all_corr.index.get_level_values(0)[-1]
        result = all_corr.loc[last_date]

    with _corr_lock:
        _corr_cache[key] = result
    return result


def get_rolling_returns(series: pd.Series, window: int = 21) -> Optional[float]:
    """Get TTL-cached rolling return for a price series."""
    if len(series) < 2:
        return None
    key = (getattr(series, 'name', None), series.iloc[-1], window, len(series))
    with _returns_lock:
        cached = _returns_cache.get(key)
        if cached is not None:
            return cached

    # Compute
    if len(series) < window:
        result = float(series.iloc[-1] / series.iloc[0] - 1)
    else:
        result = float(series.iloc[-1] / series.iloc[-window] - 1)

    with _returns_lock:
        _returns_cache[key] = result
    return result


def invalidate_computation_cache() -> None:
    """Clear all computation caches. Called between cron cycles."""
    with _vol_lock, _corr_lock, _returns_lock:
        _vol_cache.clear()
        _corr_cache.clear()
        _returns_cache.clear()
