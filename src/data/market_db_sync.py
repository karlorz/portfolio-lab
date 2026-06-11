"""Sync fetched public price JSON into the canonical market SQLite database."""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.paths import MARKET_DB, PRICES_JSON, sqlite_connect

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncSummary:
    """Summary of a price JSON to market.db sync."""

    prices_path: Path
    db_path: Path
    symbols_read: int
    rows_read: int
    rows_upserted: int
    symbols_pruned: int
    pruned_symbols: list[str]
    latest_dates: dict[str, str]


def _ensure_prices_schema(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            updated_at TEXT,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol ON prices(symbol)")


def _load_prices_payload(prices_path: Path) -> dict[str, list[dict[str, Any]]]:
    if not prices_path.exists():
        raise FileNotFoundError(f"prices.json not found: {prices_path}")

    with prices_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"prices.json must contain a symbol-to-records object: {prices_path}")

    prices: dict[str, list[dict[str, Any]]] = {}
    for symbol, records in payload.items():
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("prices.json contains an empty or non-string symbol")
        if not isinstance(records, list):
            raise ValueError(f"{symbol} records must be a list")
        prices[symbol] = records
    if not prices:
        raise ValueError(f"prices.json contains no symbols: {prices_path}")
    return prices


def _validate_price_record(symbol: str, index: int, record: Any) -> tuple[str, float]:
    if not isinstance(record, dict):
        raise ValueError(f"{symbol}[{index}] must be an object with 'd' and 'p'")
    if "d" not in record or "p" not in record:
        raise ValueError(f"{symbol}[{index}] requires 'd' and 'p'")

    date = record["d"]
    price = record["p"]
    if not isinstance(date, str):
        raise ValueError(f"{symbol}[{index}].d must be a YYYY-MM-DD string")
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{symbol}[{index}].d must be a YYYY-MM-DD string") from exc

    if not isinstance(price, int | float) or isinstance(price, bool):
        raise ValueError(f"{symbol}[{index}].p must be numeric")
    price_float = float(price)
    if price_float != price_float or price_float in (float("inf"), float("-inf")):
        raise ValueError(f"{symbol}[{index}].p must be finite")

    return date, price_float


def _build_price_rows(
    prices: dict[str, list[dict[str, Any]]],
    updated_at: str,
) -> tuple[list[tuple[str, str, float, float, float, float, int, str]], dict[str, str]]:
    rows: list[tuple[str, str, float, float, float, float, int, str]] = []
    latest_dates: dict[str, str] = {}

    for symbol, records in prices.items():
        for index, record in enumerate(records):
            date, price = _validate_price_record(symbol, index, record)
            rows.append((symbol, date, price, price, price, price, 0, updated_at))
            if symbol not in latest_dates or date > latest_dates[symbol]:
                latest_dates[symbol] = date

    return rows, latest_dates


def _prune_symbols_absent_from_prices_json(conn, canonical_symbols: set[str]) -> list[str]:
    existing_symbols = {
        row[0] for row in conn.execute("SELECT DISTINCT symbol FROM prices").fetchall()
    }
    pruned_symbols = sorted(existing_symbols - canonical_symbols)
    if not pruned_symbols:
        return []

    placeholders = ",".join("?" for _ in pruned_symbols)
    conn.execute(f"DELETE FROM prices WHERE symbol IN ({placeholders})", pruned_symbols)
    return pruned_symbols


def sync_prices_json_to_market_db(
    prices_path: str | Path = PRICES_JSON,
    db_path: str | Path = MARKET_DB,
) -> SyncSummary:
    """Upsert compact public prices into ``market.db.prices``.

    The public fetcher writes compact adjusted-close records shaped as
    ``{symbol: [{"d": "YYYY-MM-DD", "p": price}]}``. The dashboard health
    generator reads freshness from SQLite, so the data pipeline must keep this
    table in lock-step with the freshly fetched JSON artifact.
    """

    resolved_prices_path = Path(prices_path)
    resolved_db_path = Path(db_path)
    prices = _load_prices_payload(resolved_prices_path)
    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, latest_dates = _build_price_rows(prices, updated_at)

    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite_connect(resolved_db_path)) as conn:
        _ensure_prices_schema(conn)
        pruned_symbols = _prune_symbols_absent_from_prices_json(conn, set(prices))
        conn.executemany(
            """
            INSERT OR REPLACE INTO prices
                (symbol, date, open, high, low, close, volume, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    return SyncSummary(
        prices_path=resolved_prices_path,
        db_path=resolved_db_path,
        symbols_read=len(prices),
        rows_read=len(rows),
        rows_upserted=len(rows),
        symbols_pruned=len(pruned_symbols),
        pruned_symbols=pruned_symbols,
        latest_dates=latest_dates,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", default=str(PRICES_JSON), help="Path to public prices.json")
    parser.add_argument("--db", default=str(MARKET_DB), help="Path to market.db")
    args = parser.parse_args(argv)

    from src.utils.log_config import configure_logging

    configure_logging()
    summary = sync_prices_json_to_market_db(prices_path=args.prices, db_path=args.db)
    latest_date = max(summary.latest_dates.values(), default="none")
    logger.info(
        "Synced %s price rows across %s symbols into %s; pruned=%s; latest date=%s",
        summary.rows_upserted,
        summary.symbols_read,
        summary.db_path,
        summary.symbols_pruned,
        latest_date,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
