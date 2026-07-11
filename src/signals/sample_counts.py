"""Shared sample-count provenance helpers for live signal adapters."""

from src.data import price_cache


def get_live_sample_count(ticker: str) -> int:
    """Return the active prices.json history length for one ticker."""
    try:
        entries = price_cache.get_prices().get(ticker, [])
    except (OSError, TypeError, ValueError):
        return 0
    return len(entries) if isinstance(entries, list) else 0
