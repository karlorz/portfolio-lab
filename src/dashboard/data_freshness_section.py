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
    "CRYPTO_SYMBOLS",
    "DATA_FRESHNESS_EXCEPTIONS",
    "build_data_freshness_section",
]

# Crypto trades on a seven-day calendar; traditional assets trade on weekdays.
# Mirror of CRYPTO_SYMBOLS in src/data/symbol_universe.ts (canonical TS source;
# kept as a literal here because the freshness section is Python-only).
CRYPTO_SYMBOLS: frozenset[str] = frozenset({"BTC-USD", "ETH-USD"})

# SQLite strftime('%w', date) is '0' (Sunday) .. '6' (Saturday). Weekend rows
# must never advance the weekday trading reference.
_WEEKDAY_FILTER = "strftime('%w', date) NOT IN ('0', '6')"

DATA_FRESHNESS_EXCEPTIONS = (
    sqlite3.Error,
    ValueError,
    TypeError,
    RuntimeError,
    OSError,
)


def _parse_iso_date(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        logger.warning(
            "Failed to parse data freshness date '%s': %s",
            value,
            exc,
        )
        return None


def _in_clause(symbols: frozenset[str]) -> tuple[str, tuple[str, ...]]:
    """Build a parameterized ``IN (...)/NOT IN (...)`` clause for a symbol set."""
    ordered = tuple(sorted(symbols))
    placeholders = ", ".join("?" for _ in ordered)
    return placeholders, ordered


def build_data_freshness_section(
    *,
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Build per-symbol data_freshness from the prices SQLite table.

    Returns {"data_freshness": {symbol: entry}, "latest_market_date": str|None,
    "latest_crypto_date": str|None}.

    The freshness reference is calendar-aware: traditional market symbols are
    compared against ``latest_market_date`` — the latest weekday bar across
    non-crypto symbols — while crypto symbols use ``latest_crypto_date``, the
    latest bar across crypto symbols on their seven-day calendar. Sunday crypto
    rows therefore never make weekday assets look stale. The reference dates
    are exposed so callers can cross-reference without re-querying.

    Errors are logged and degrade to an empty freshness map rather than
    failing the whole health.json build.
    """
    freshness: dict[str, Any] = {}
    latest_market_date: str | None = None
    latest_crypto_date: str | None = None
    latest_market_dt: datetime | None = None
    latest_crypto_dt: datetime | None = None

    try:
        cursor = conn.cursor()
        # Weekday trading reference: latest non-crypto bar that is not a
        # Saturday/Sunday row. Malformed dates (strftime -> NULL) stay
        # eligible so MAX() string semantics match the previous behavior.
        placeholders, crypto_params = _in_clause(CRYPTO_SYMBOLS)
        if CRYPTO_SYMBOLS:
            cursor.execute(
                f"""
                SELECT MAX(date) FROM prices
                WHERE symbol NOT IN ({placeholders})
                  AND ({_WEEKDAY_FILTER} OR strftime('%w', date) IS NULL)
                """,
                crypto_params,
            )
        else:
            cursor.execute(
                f"SELECT MAX(date) FROM prices WHERE ({_WEEKDAY_FILTER} OR strftime('%w', date) IS NULL)"
            )
        row = cursor.fetchone()
        latest_market_date = row[0] if row is not None else None
        if latest_market_date:
            latest_market_dt = _parse_iso_date(latest_market_date)

        # Crypto reference: seven-day calendar (weekends are trading days).
        if CRYPTO_SYMBOLS:
            cursor.execute(
                f"SELECT MAX(date) FROM prices WHERE symbol IN ({placeholders})",
                crypto_params,
            )
            row = cursor.fetchone()
            latest_crypto_date = row[0] if row is not None else None
            if latest_crypto_date:
                latest_crypto_dt = _parse_iso_date(latest_crypto_date)

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
            last_dt = _parse_iso_date(last_date)
            if last_dt is None:
                continue
            days_stale = (datetime.now() - last_dt).days
            if sym in CRYPTO_SYMBOLS:
                ref_dt = latest_crypto_dt
                ref_date = latest_crypto_date
            else:
                ref_dt = latest_market_dt
                ref_date = latest_market_date
            market_lag_days = (
                max((ref_dt - last_dt).days, 0) if ref_dt is not None else days_stale
            )
            freshness[sym] = build_symbol_freshness_entry(
                last_date=last_date,
                days_stale=days_stale,
                market_lag_days=market_lag_days,
                latest_available_market_date=ref_date,
            )
    except DATA_FRESHNESS_EXCEPTIONS as exc:
        logger.warning("Data freshness section not available: %s", exc)

    return {
        "data_freshness": freshness,
        "latest_market_date": latest_market_date,
        "latest_crypto_date": latest_crypto_date,
    }
