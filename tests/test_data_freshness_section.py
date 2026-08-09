"""Tests for data_freshness_section builder."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from src.dashboard.data_freshness_section import build_data_freshness_section

# Fixed reference dates (deterministic on any run day): 2026-08-07 is a
# Friday, 2026-08-08 a Saturday, 2026-08-09 a Sunday, 2026-08-06 a Thursday.
FRIDAY = "2026-08-07"
SATURDAY = "2026-08-08"
SUNDAY = "2026-08-09"
THURSDAY = "2026-08-06"


def _days_since(value: str) -> int:
    return (datetime.now() - datetime.strptime(value, "%Y-%m-%d")).days


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
    assert out["latest_crypto_date"] is None
    conn.close()


def test_build_data_freshness_section_single_symbol() -> None:
    conn = _make_conn()
    _insert(conn, "SPY", FRIDAY)
    out = build_data_freshness_section(conn=conn)
    assert out["latest_market_date"] == FRIDAY
    entry = out["data_freshness"]["SPY"]
    assert entry["last_update"] == FRIDAY
    assert entry["days_stale"] == _days_since(FRIDAY)
    assert entry["market_lag_days"] == 0
    assert entry["status"] == "fresh"
    assert entry["latest_available_market_date"] == FRIDAY
    conn.close()


def test_build_data_freshness_section_stale_symbol() -> None:
    conn = _make_conn()
    _insert(conn, "SPY", FRIDAY)
    _insert(conn, "GLD", "2026-08-02")
    out = build_data_freshness_section(conn=conn)
    assert out["latest_market_date"] == FRIDAY
    gld = out["data_freshness"]["GLD"]
    assert gld["days_stale"] == _days_since("2026-08-02")
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
    assert out["latest_crypto_date"] is None


def test_build_data_freshness_section_unparseable_date_skipped() -> None:
    """A malformed date row is skipped with a warning, not raised."""
    conn = _make_conn()
    _insert(conn, "SPY", "not-a-date")
    out = build_data_freshness_section(conn=conn)
    # latest_market_date is set (string compare), but the row's date fails to parse
    assert out["latest_market_date"] == "not-a-date"
    assert "SPY" not in out["data_freshness"]
    conn.close()


def test_sunday_crypto_rows_keep_weekday_symbols_fresh() -> None:
    """Sunday crypto rows advance only the crypto reference (7-day calendar).

    Regression: the previous single global MAX(date) made 39 weekday assets
    appear two days stale on weekends (artifact SLO warning).
    """
    conn = _make_conn()
    for sym in ("SPY", "GLD"):
        _insert(conn, sym, FRIDAY)
    for sym in ("BTC-USD", "ETH-USD"):
        _insert(conn, sym, SUNDAY)
    out = build_data_freshness_section(conn=conn)
    assert out["latest_market_date"] == FRIDAY
    assert out["latest_crypto_date"] == SUNDAY

    spy = out["data_freshness"]["SPY"]
    assert spy["last_update"] == FRIDAY
    assert spy["market_lag_days"] == 0
    assert spy["status"] == "fresh"
    assert spy["latest_available_market_date"] == FRIDAY

    btc = out["data_freshness"]["BTC-USD"]
    assert btc["last_update"] == SUNDAY
    assert btc["market_lag_days"] == 0
    assert btc["status"] == "fresh"
    assert btc["latest_available_market_date"] == SUNDAY
    conn.close()


def test_no_crypto_rows_crypto_reference_none() -> None:
    """Without crypto rows the crypto reference is None and all symbols use
    the trading reference; a Thursday bar on a Sunday as-of has missed Friday's
    completed session (1 missed → stale under session semantics)."""
    from datetime import datetime, timezone

    conn = _make_conn()
    _insert(conn, "SPY", FRIDAY)
    _insert(conn, "GLD", THURSDAY)
    out = build_data_freshness_section(
        conn=conn,
        as_of=datetime(2026, 8, 9, 20, 0, tzinfo=timezone.utc),  # Sunday 16:00 ET
    )
    assert out["latest_market_date"] == FRIDAY
    assert out["latest_crypto_date"] is None
    assert out["data_freshness"]["GLD"]["market_lag_days"] == 1
    assert out["data_freshness"]["GLD"]["missed_market_sessions"] == 1
    assert out["data_freshness"]["GLD"]["status"] == "stale"
    assert out["data_freshness"]["SPY"]["status"] == "fresh"
    conn.close()


def test_weekend_row_for_traditional_symbol_does_not_advance_reference() -> None:
    """A stray weekend row for a traditional symbol must not advance the
    weekday trading reference; status still reflects missed sessions."""
    from datetime import datetime, timezone

    conn = _make_conn()
    _insert(conn, "SPY", FRIDAY)
    _insert(conn, "SPY", SATURDAY)  # stray weekend row
    _insert(conn, "GLD", THURSDAY)
    out = build_data_freshness_section(
        conn=conn,
        as_of=datetime(2026, 8, 9, 20, 0, tzinfo=timezone.utc),  # Sunday 16:00 ET
    )
    assert out["latest_market_date"] == FRIDAY
    gld = out["data_freshness"]["GLD"]
    assert gld["market_lag_days"] == 1
    assert gld["missed_market_sessions"] == 1
    assert gld["status"] == "stale"
    assert out["data_freshness"]["SPY"]["status"] == "fresh"
    conn.close()


def test_sparse_vix3m_reports_honest_trading_lag() -> None:
    """Sparse VIX-family rows keep honest lag in this section (advisory
    semantics live in price_quality.ts, untouched here)."""
    conn = _make_conn()
    _insert(conn, "SPY", FRIDAY)
    _insert(conn, "^VIX3M", "2026-07-17")
    out = build_data_freshness_section(conn=conn)
    vix3m = out["data_freshness"]["^VIX3M"]
    assert vix3m["market_lag_days"] == 21
    assert vix3m["status"] == "critical"
    assert vix3m["latest_available_market_date"] == FRIDAY
    assert out["data_freshness"]["SPY"]["status"] == "fresh"
    conn.close()


# ── Task 4: session-aware fields with injected as-of ───────────────────

def test_session_fields_present_with_injected_as_of() -> None:
    """Entries expose session fields; Sunday as-of keeps Friday bars fresh."""
    from datetime import datetime, timezone

    conn = _make_conn()
    for sym in ("SPY", "GLD", "TLT"):
        _insert(conn, sym, FRIDAY)
    _insert(conn, "BTC-USD", SUNDAY)
    out = build_data_freshness_section(
        conn=conn,
        as_of=datetime(2026, 8, 9, 20, 0, tzinfo=timezone.utc),  # Sunday 16:00 ET
    )
    spy = out["data_freshness"]["SPY"]
    assert spy["missed_market_sessions"] == 0
    assert spy["last_expected_completed_session"] == FRIDAY
    assert spy["status"] == "fresh"
    assert spy["status_basis"] == "missed_sessions"
    assert spy["calendar_age_days"] == spy["days_stale"]
    conn.close()


def test_genuine_missed_sessions_escalate_with_as_of() -> None:
    from datetime import datetime, timezone

    conn = _make_conn()
    _insert(conn, "SPY", "2026-08-03")  # Monday; as-of Friday 17:00 ET
    _insert(conn, "GLD", FRIDAY)
    out = build_data_freshness_section(
        conn=conn,
        as_of=datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc),  # Fri 17:00 ET
    )
    assert out["data_freshness"]["SPY"]["missed_market_sessions"] == 4
    assert out["data_freshness"]["SPY"]["status"] == "critical"
    assert out["data_freshness"]["GLD"]["missed_market_sessions"] == 0
    assert out["data_freshness"]["GLD"]["status"] == "fresh"
    conn.close()
