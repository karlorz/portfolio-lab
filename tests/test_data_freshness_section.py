"""Tests for data_freshness_section builder."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from src.dashboard.data_freshness_section import build_data_freshness_section


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT)")
    return conn


def _insert(conn: sqlite3.Connection, symbol: str, date: str) -> None:
    conn.execute("INSERT INTO prices (symbol, date) VALUES (?, ?)", (symbol, date))
    conn.commit()


def test_build_data_freshness_section_empty() -> None:
    conn = _make_conn()
    out = build_data_freshness_section(conn=conn)
    assert out["data_freshness"] == {}
    assert out["latest_market_date"] is None
    conn.close()


def test_build_data_freshness_section_single_symbol() -> None:
    conn = _make_conn()
    today = datetime.now().strftime("%Y-%m-%d")
    _insert(conn, "SPY", today)
    out = build_data_freshness_section(conn=conn)
    assert out["latest_market_date"] == today
    entry = out["data_freshness"]["SPY"]
    assert entry["last_update"] == today
    assert entry["days_stale"] == 0
    assert entry["market_lag_days"] == 0
    assert entry["status"] == "fresh"
    assert entry["latest_available_market_date"] == today
    conn.close()


def test_build_data_freshness_section_stale_symbol() -> None:
    conn = _make_conn()
    today = datetime.now().strftime("%Y-%m-%d")
    old = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    _insert(conn, "SPY", today)
    _insert(conn, "GLD", old)
    out = build_data_freshness_section(conn=conn)
    assert out["latest_market_date"] == today
    gld = out["data_freshness"]["GLD"]
    assert gld["days_stale"] == 5
    assert gld["market_lag_days"] == 5
    assert gld["status"] == "critical"
    conn.close()


def test_build_data_freshness_section_multiple_symbols_per_group() -> None:
    """MAX(date) GROUP BY symbol picks the latest date per symbol."""
    conn = _make_conn()
    today = datetime.now().strftime("%Y-%m-%d")
    older = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    _insert(conn, "SPY", older)
    _insert(conn, "SPY", today)
    out = build_data_freshness_section(conn=conn)
    assert out["data_freshness"]["SPY"]["last_update"] == today
    conn.close()


def test_build_data_freshness_section_db_error_degrades_gracefully() -> None:
    """A sqlite error yields empty freshness, not an exception."""
    conn = MagicMock(spec=sqlite3.Connection)
    cursor = MagicMock()
    cursor.execute.side_effect = sqlite3.OperationalError("no such table: prices")
    conn.cursor.return_value = cursor
    out = build_data_freshness_section(conn=conn)
    assert out["data_freshness"] == {}
    assert out["latest_market_date"] is None


def test_build_data_freshness_section_unparseable_date_skipped() -> None:
    """A malformed date row is skipped with a warning, not raised."""
    conn = _make_conn()
    _insert(conn, "SPY", "not-a-date")
    out = build_data_freshness_section(conn=conn)
    # latest_market_date is set (string compare), but the row's date fails to parse
    assert out["latest_market_date"] == "not-a-date"
    assert "SPY" not in out["data_freshness"]
    conn.close()
