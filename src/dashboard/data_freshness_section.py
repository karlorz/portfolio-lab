"""Data freshness section for health.json (SQLite-driven assembly)."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.dashboard.health_report import build_symbol_freshness_entry

logger = logging.getLogger(__name__)

__all__ = [
    "DATA_FRESHNESS_EXCEPTIONS",
    "build_data_freshness_section",
]


DATA_FRESHNESS_EXCEPTIONS = (
    sqlite3.Error,
    ValueError,
    TypeError,
    RuntimeError,
    OSError,
)


def build_data_freshness_section(
    *,
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Build per-symbol data_freshness from the prices SQLite table.

    Returns {"data_freshness": {symbol: entry}, "latest_market_date": str|None}.
    The latest_market_date is exposed so callers can cross-reference without
    re-querying. Errors are logged and degrade to an empty freshness map
    rather than failing the whole health.json build.
    """
    freshness: dict[str, Any] = {}
    latest_market_date: str | None = None
    latest_market_dt: datetime | None = None

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM prices")
        row = cursor.fetchone()
        latest_market_date = row[0] if row is not None else None
        if latest_market_date:
            try:
                latest_market_dt = datetime.strptime(latest_market_date, "%Y-%m-%d")
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Failed to parse latest market freshness date '%s': %s",
                    latest_market_date,
                    exc,
                )

        cursor.execute(
            """
            SELECT symbol, MAX(date) as last_date
            FROM prices
            GROUP BY symbol
            """
        )
        for sym, last_date in cursor.fetchall():
            if not last_date:
                continue
            try:
                last_dt = datetime.strptime(last_date, "%Y-%m-%d")
                days_stale = (datetime.now() - last_dt).days
                market_lag_days = (
                    max((latest_market_dt - last_dt).days, 0)
                    if latest_market_dt is not None
                    else days_stale
                )
                freshness[sym] = build_symbol_freshness_entry(
                    last_date=last_date,
                    days_stale=days_stale,
                    market_lag_days=market_lag_days,
                    latest_available_market_date=latest_market_date,
                )
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Failed to parse data freshness date '%s': %s",
                    last_date,
                    exc,
                )
    except DATA_FRESHNESS_EXCEPTIONS as exc:
        logger.warning("Data freshness section not available: %s", exc)

    return {"data_freshness": freshness, "latest_market_date": latest_market_date}
