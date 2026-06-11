import json
import sqlite3

import pytest

from src.data.market_db_sync import sync_prices_json_to_market_db


def write_prices_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def fetch_rows(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            """
            SELECT symbol, date, open, high, low, close, volume
            FROM prices
            ORDER BY symbol, date
            """
        ).fetchall()


def test_sync_updates_stale_market_db_from_prices_json(tmp_path):
    prices_path = tmp_path / "prices.json"
    db_path = tmp_path / "market.db"
    write_prices_json(
        prices_path,
        {
            "SPY": [{"d": "2026-06-10", "p": 612.34}],
            "GLD": [{"d": "2026-06-10", "p": 318.12}],
        },
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE prices (
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
        conn.execute(
            """
            INSERT INTO prices (symbol, date, open, high, low, close, volume, updated_at)
            VALUES ('SPY', '2026-05-21', 590.0, 590.0, 590.0, 590.0, 0, 'old')
            """
        )

    summary = sync_prices_json_to_market_db(prices_path=prices_path, db_path=db_path)

    assert summary.symbols_read == 2
    assert summary.rows_read == 2
    assert summary.rows_upserted == 2
    with sqlite3.connect(db_path) as conn:
        latest = dict(
            conn.execute("SELECT symbol, MAX(date) FROM prices GROUP BY symbol").fetchall()
        )
    assert latest == {"GLD": "2026-06-10", "SPY": "2026-06-10"}


def test_sync_inserts_all_symbols_and_uses_close_as_ohlc_proxy(tmp_path):
    prices_path = tmp_path / "prices.json"
    db_path = tmp_path / "market.db"
    write_prices_json(
        prices_path,
        {
            "SPY": [
                {"d": "2026-06-09", "p": 611.11},
                {"d": "2026-06-10", "p": 612.34},
            ],
            "^VIX": [{"d": "2026-06-10", "p": 14.2}],
        },
    )

    summary = sync_prices_json_to_market_db(prices_path=prices_path, db_path=db_path)

    assert summary.latest_dates == {"SPY": "2026-06-10", "^VIX": "2026-06-10"}
    assert fetch_rows(db_path) == [
        ("SPY", "2026-06-09", 611.11, 611.11, 611.11, 611.11, 0),
        ("SPY", "2026-06-10", 612.34, 612.34, 612.34, 612.34, 0),
        ("^VIX", "2026-06-10", 14.2, 14.2, 14.2, 14.2, 0),
    ]


def test_sync_is_idempotent_for_same_symbol_date_rows(tmp_path):
    prices_path = tmp_path / "prices.json"
    db_path = tmp_path / "market.db"
    write_prices_json(
        prices_path,
        {"SPY": [{"d": "2026-06-10", "p": 612.34}]},
    )

    sync_prices_json_to_market_db(prices_path=prices_path, db_path=db_path)
    sync_prices_json_to_market_db(prices_path=prices_path, db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    assert count == 1


def test_sync_prunes_symbols_absent_from_canonical_prices_json(tmp_path):
    prices_path = tmp_path / "prices.json"
    db_path = tmp_path / "market.db"
    write_prices_json(prices_path, {"SPY": [{"d": "2026-06-10", "p": 612.34}]})
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE prices (
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
        conn.execute(
            """
            INSERT INTO prices (symbol, date, open, high, low, close, volume, updated_at)
            VALUES ('^VIX', '2026-05-22', 20.0, 20.0, 20.0, 20.0, 0, 'old')
            """
        )

    summary = sync_prices_json_to_market_db(prices_path=prices_path, db_path=db_path)

    assert summary.pruned_symbols == ["^VIX"]
    with sqlite3.connect(db_path) as conn:
        symbols = [row[0] for row in conn.execute("SELECT DISTINCT symbol FROM prices")]
    assert symbols == ["SPY"]


def test_sync_creates_prices_table_and_indexes_for_empty_db(tmp_path):
    prices_path = tmp_path / "prices.json"
    db_path = tmp_path / "market.db"
    write_prices_json(prices_path, {"TLT": [{"d": "2026-06-10", "p": 88.75}]})

    sync_prices_json_to_market_db(prices_path=prices_path, db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        table_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'prices'"
        ).fetchone()[0]
        index_names = {
            row[1] for row in conn.execute("PRAGMA index_list('prices')").fetchall()
        }
    assert "PRIMARY KEY (symbol, date)" in table_sql
    assert "idx_prices_date" in index_names
    assert "idx_prices_symbol" in index_names


def test_sync_missing_prices_json_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="prices.json not found"):
        sync_prices_json_to_market_db(
            prices_path=tmp_path / "prices.json",
            db_path=tmp_path / "market.db",
        )


def test_sync_malformed_price_record_fails_the_job(tmp_path):
    prices_path = tmp_path / "prices.json"
    write_prices_json(prices_path, {"SPY": [{"d": "2026-06-10"}]})

    with pytest.raises(ValueError, match="SPY\\[0\\].*requires 'd' and 'p'"):
        sync_prices_json_to_market_db(
            prices_path=prices_path,
            db_path=tmp_path / "market.db",
        )
