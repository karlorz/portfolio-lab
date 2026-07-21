"""Sync fetched public price JSON into the canonical market SQLite database."""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import closing
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.paths import MARKET_DB, PRICES_JSON, sqlite_connect

logger = logging.getLogger(__name__)

PRICE_DATA_SOURCE = "public/data/prices.json"
ADJUSTED_CLOSE_PROXY_SEMANTICS = "adjusted_close_proxy_ohlc"
QUALITY_STATUS_OK = "ok"
QUALITY_STATUS_WARN = "warn"
QUALITY_STATUS_FAIL = "fail"
DEFAULT_SPLIT_LIKE_RETURN_PCT = 40.0
DEFAULT_CRITICAL_RETURN_PCT = 90.0

# Volatility indices (and aliases) do not undergo equity-style splits. Large
# daily % moves (Volmageddon ~+115%, COVID ~+40–45%, Aug-2024 ~+65%) are regime
# jumps, not corporate actions — do not flag split_like / extreme equity heuristics.
VOLATILITY_INDEX_SYMBOLS = frozenset(
    {
        "^VIX",
        "^VIX3M",
        "^VIX6M",
        "^VIX9D",
        "VIX",
        "VIX3M",
        "VIX6M",
        "VIX9D",
    }
)


def is_volatility_index_symbol(symbol: str) -> bool:
    """True for VIX-family indices that skip equity split-like return gates."""
    normalized = (symbol or "").strip().upper()
    if not normalized:
        return False
    if normalized in VOLATILITY_INDEX_SYMBOLS:
        return True
    # Catch ^VIX* / VIX* family without listing every tenor
    bare = normalized[1:] if normalized.startswith("^") else normalized
    return bare.startswith("VIX")


QUALITY_ISSUE_KEYS = (
    "duplicate_dates",
    "empty_symbols",
    "extreme_returns",
    "invalid_dates",
    "invalid_prices",
    "missing_required_keys",
    "non_monotonic_rows",
    "non_object_records",
    "split_like_returns",
    "total",
)

PriceRow = tuple[str, str, float, float, float, float, int, str, str, str, int]


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
    quality_status: str = QUALITY_STATUS_OK
    quality_blocking: bool = False
    quality_issue_counts: dict[str, int] = field(default_factory=dict)
    quality_report: dict[str, Any] = field(default_factory=dict)


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
            data_source TEXT NOT NULL DEFAULT 'unknown',
            price_semantics TEXT NOT NULL DEFAULT 'unknown',
            is_adjusted_close_proxy INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info('prices')")}
    additive_columns = {
        "data_source": "TEXT NOT NULL DEFAULT 'unknown'",
        "price_semantics": "TEXT NOT NULL DEFAULT 'unknown'",
        "is_adjusted_close_proxy": "INTEGER NOT NULL DEFAULT 0",
    }
    for column, definition in additive_columns.items():
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE prices ADD COLUMN {column} {definition}")
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


def _empty_quality_issue_counts() -> dict[str, int]:
    return dict.fromkeys(QUALITY_ISSUE_KEYS, 0)


def _increment_quality_issue(counts: dict[str, int], key: str, amount: int = 1) -> None:
    counts[key] += amount
    counts["total"] += amount


def _compute_return_pct(previous_price: float, current_price: float) -> float:
    return round((current_price - previous_price) / previous_price * 100, 4)


def _audit_prices_payload(
    prices: dict[str, list[dict[str, Any]]],
    *,
    critical_return_pct: float = DEFAULT_CRITICAL_RETURN_PCT,
    split_like_return_pct: float = DEFAULT_SPLIT_LIKE_RETURN_PCT,
) -> dict[str, Any]:
    """Return a compact quality summary for the sync input payload."""
    counts = _empty_quality_issue_counts()
    symbols: list[dict[str, Any]] = []
    first_blocking_error: str | None = None
    has_warning = False

    for symbol, records in prices.items():
        symbol_counts = _empty_quality_issue_counts()
        seen_dates: set[str] = set()
        duplicate_dates: set[str] = set()
        valid_prices: list[tuple[str, float]] = []
        latest_date: str | None = None
        previous_date: str | None = None

        if not records:
            _increment_quality_issue(counts, "empty_symbols")
            _increment_quality_issue(symbol_counts, "empty_symbols")
            first_blocking_error = first_blocking_error or f"{symbol} records must not be empty"

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                _increment_quality_issue(counts, "non_object_records")
                _increment_quality_issue(symbol_counts, "non_object_records")
                first_blocking_error = (
                    first_blocking_error
                    or f"{symbol}[{index}] must be an object with 'd' and 'p'"
                )
                continue
            if "d" not in record or "p" not in record:
                _increment_quality_issue(counts, "missing_required_keys")
                _increment_quality_issue(symbol_counts, "missing_required_keys")
                first_blocking_error = (
                    first_blocking_error or f"{symbol}[{index}] requires 'd' and 'p'"
                )
                continue

            date = record["d"]
            price = record["p"]
            if not isinstance(date, str):
                _increment_quality_issue(counts, "invalid_dates")
                _increment_quality_issue(symbol_counts, "invalid_dates")
                first_blocking_error = (
                    first_blocking_error
                    or f"{symbol}[{index}].d must be a YYYY-MM-DD string"
                )
                continue
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                _increment_quality_issue(counts, "invalid_dates")
                _increment_quality_issue(symbol_counts, "invalid_dates")
                first_blocking_error = (
                    first_blocking_error
                    or f"{symbol}[{index}].d must be a YYYY-MM-DD string"
                )
                continue

            if previous_date is not None and date < previous_date:
                _increment_quality_issue(counts, "non_monotonic_rows")
                _increment_quality_issue(symbol_counts, "non_monotonic_rows")
                first_blocking_error = (
                    first_blocking_error
                    or f"{symbol}[{index}].d must not move backward from {previous_date}"
                )
            previous_date = date

            if date in seen_dates:
                duplicate_dates.add(date)
            seen_dates.add(date)
            latest_date = date if latest_date is None or date > latest_date else latest_date

            if not isinstance(price, int | float) or isinstance(price, bool):
                _increment_quality_issue(counts, "invalid_prices")
                _increment_quality_issue(symbol_counts, "invalid_prices")
                first_blocking_error = (
                    first_blocking_error or f"{symbol}[{index}].p must be numeric"
                )
                continue
            price_float = float(price)
            if (
                price_float != price_float
                or price_float in (float("inf"), float("-inf"))
                or price_float <= 0
            ):
                _increment_quality_issue(counts, "invalid_prices")
                _increment_quality_issue(symbol_counts, "invalid_prices")
                first_blocking_error = first_blocking_error or (
                    "Blocking price quality audit failed: "
                    f"{symbol}[{index}].p must be positive and finite"
                )
                continue

            valid_prices.append((date, price_float))

        if duplicate_dates:
            duplicate_count = len(duplicate_dates)
            _increment_quality_issue(counts, "duplicate_dates", duplicate_count)
            _increment_quality_issue(symbol_counts, "duplicate_dates", duplicate_count)
            first_duplicate = sorted(duplicate_dates)[0]
            first_blocking_error = (
                first_blocking_error or f"Duplicate price date for {symbol}: {first_duplicate}"
            )

        sorted_prices = sorted(valid_prices)
        # VIX-family: skip equity split/extreme return gates (regime jumps, not splits).
        skip_return_anomaly_gates = is_volatility_index_symbol(symbol)
        for (previous_date, previous_price), (date, price) in zip(
            sorted_prices,
            sorted_prices[1:],
            strict=False,
        ):
            if skip_return_anomaly_gates:
                continue
            return_pct = _compute_return_pct(previous_price, price)
            absolute_return_pct = abs(return_pct)
            if absolute_return_pct >= critical_return_pct:
                _increment_quality_issue(counts, "extreme_returns")
                _increment_quality_issue(symbol_counts, "extreme_returns")
                first_blocking_error = first_blocking_error or (
                    "Blocking price quality audit failed: "
                    f"{symbol} return {return_pct:.2f}% from {previous_date} to {date}"
                )
            elif absolute_return_pct >= split_like_return_pct:
                _increment_quality_issue(counts, "split_like_returns")
                _increment_quality_issue(symbol_counts, "split_like_returns")
                has_warning = True

        symbol_warning_count = symbol_counts["split_like_returns"]
        symbol_blocking_count = symbol_counts["total"] - symbol_warning_count
        symbol_status = QUALITY_STATUS_FAIL if symbol_blocking_count else QUALITY_STATUS_OK
        if symbol_status == QUALITY_STATUS_OK and symbol_warning_count:
            symbol_status = QUALITY_STATUS_WARN
        symbol_row: dict[str, Any] = {
            "symbol": symbol,
            "status": symbol_status,
            "row_count": len(records),
            "latest_date": latest_date,
            "issue_counts": symbol_counts,
        }
        if skip_return_anomaly_gates:
            symbol_row["return_anomaly_gates"] = "skipped_volatility_index"
        symbols.append(symbol_row)

    blocking = first_blocking_error is not None
    status = (
        QUALITY_STATUS_FAIL
        if blocking
        else QUALITY_STATUS_WARN
        if has_warning
        else QUALITY_STATUS_OK
    )
    return {
        "status": status,
        "blocking": blocking,
        "first_blocking_error": first_blocking_error,
        "issue_counts": counts,
        "symbols_checked": len(prices),
        "symbols": symbols,
    }


def _build_price_rows(
    prices: dict[str, list[dict[str, Any]]],
    updated_at: str,
) -> tuple[list[PriceRow], dict[str, str]]:
    rows: list[PriceRow] = []
    latest_dates: dict[str, str] = {}

    for symbol, records in prices.items():
        seen_dates: set[str] = set()
        for index, record in enumerate(records):
            date, price = _validate_price_record(symbol, index, record)
            if date in seen_dates:
                raise ValueError(f"Duplicate price date for {symbol}: {date}")
            seen_dates.add(date)
            rows.append(
                (
                    symbol,
                    date,
                    price,
                    price,
                    price,
                    price,
                    0,
                    updated_at,
                    PRICE_DATA_SOURCE,
                    ADJUSTED_CLOSE_PROXY_SEMANTICS,
                    1,
                )
            )
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
    *,
    block_on_quality_warnings: bool = False,
) -> SyncSummary:
    """Upsert compact public prices into ``market.db.prices``.

    The public fetcher writes compact adjusted-close records shaped as
    ``{symbol: [{"d": "YYYY-MM-DD", "p": price}]}``. The dashboard health
    generator reads freshness from SQLite, so the data pipeline must keep this
    table in lock-step with the freshly fetched JSON artifact.

    Compact rows are marked as adjusted-close OHLC proxies. The open, high,
    low, and close fields are equal by construction for close-only compatibility;
    consumers that require true intraday OHLC bars must check the proxy metadata.
    """

    resolved_prices_path = Path(prices_path)
    resolved_db_path = Path(db_path)
    prices = _load_prices_payload(resolved_prices_path)
    quality_report = _audit_prices_payload(prices)
    if quality_report["blocking"] or (
        block_on_quality_warnings and quality_report["status"] == QUALITY_STATUS_WARN
    ):
        raise ValueError(
            quality_report["first_blocking_error"]
            or "Blocking price quality audit failed: warning-level issues are configured as blocking"
        )

    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, latest_dates = _build_price_rows(prices, updated_at)

    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite_connect(resolved_db_path)) as conn:
        _ensure_prices_schema(conn)
        pruned_symbols = _prune_symbols_absent_from_prices_json(conn, set(prices))
        conn.executemany(
            """
            INSERT OR REPLACE INTO prices
                (
                    symbol,
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    updated_at,
                    data_source,
                    price_semantics,
                    is_adjusted_close_proxy
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        quality_status=quality_report["status"],
        quality_blocking=quality_report["blocking"],
        quality_issue_counts=quality_report["issue_counts"],
        quality_report=quality_report,
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
        (
            "Synced %s price rows across %s symbols into %s; pruned=%s; "
            "latest date=%s; quality=%s; quality_issues=%s"
        ),
        summary.rows_upserted,
        summary.symbols_read,
        summary.db_path,
        summary.symbols_pruned,
        latest_date,
        summary.quality_status,
        summary.quality_issue_counts.get("total", 0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
