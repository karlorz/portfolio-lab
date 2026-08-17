"""Tests for src.monitor.market_data_consistency."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import pytest

from src.monitor.market_data_consistency import (
    reconcile_compact_prices_with_market_db,
    reconcile_price_providers,
    broker_market_data_consistency_report,
    require_true_ohlc_price_rows,
    is_adjusted_close_proxy_price_row,
    MarketDataSemanticsError,
)


def _init_market_db(db_path: Path, rows: list[tuple[str, str, float]]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("""
            CREATE TABLE prices (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                close REAL NOT NULL,
                PRIMARY KEY (symbol, date)
            )
        """)
        conn.executemany("INSERT INTO prices (symbol, date, close) VALUES (?, ?, ?)", rows)
        conn.commit()


def test_reconcile_compact_prices_missing_prices_unavailable(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    _init_market_db(db_path, [("SPY", "2026-08-15", 500.0)])
    prices_path = tmp_path / "nonexistent_prices.json"

    res = reconcile_compact_prices_with_market_db(prices_path=prices_path, db_path=db_path)
    assert res["status"] == "unavailable"
    assert res["failure_type"] == "compact_prices_unavailable"


def test_reconcile_compact_prices_missing_db_fails(tmp_path: Path) -> None:
    prices_path = tmp_path / "prices.json"
    prices_path.write_text(json.dumps({"SPY": [{"d": "2026-08-15", "p": 500.0}]}), encoding="utf-8")
    db_path = tmp_path / "nonexistent_market.db"

    res = reconcile_compact_prices_with_market_db(prices_path=prices_path, db_path=db_path)
    assert res["status"] == "fail"
    assert res["failure_type"] == "market_db_unavailable"


def test_reconcile_compact_prices_clean_match(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    _init_market_db(db_path, [
        ("SPY", "2026-08-15", 500.0),
        ("GLD", "2026-08-15", 200.0),
    ])
    prices_path = tmp_path / "prices.json"
    prices_path.write_text(
        json.dumps({
            "SPY": [{"d": "2026-08-15", "p": 500.0}],
            "GLD": [{"d": "2026-08-15", "p": 200.0}],
        }),
        encoding="utf-8",
    )

    res = reconcile_compact_prices_with_market_db(prices_path=prices_path, db_path=db_path)
    assert res["status"] == "ok"
    assert res["symbols_checked"] == 2
    assert len(res["top_offenders"]) == 0


def test_reconcile_compact_prices_missing_symbol_and_stale_lag_and_divergence(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    _init_market_db(db_path, [
        ("SPY", "2026-08-14", 500.0),  # lags prices.json (2026-08-15) by 1d
        ("GLD", "2026-08-15", 205.0),  # diverges from 200.0
        # TLT missing from SQLite
    ])
    prices_path = tmp_path / "prices.json"
    prices_path.write_text(
        json.dumps({
            "SPY": [{"d": "2026-08-15", "p": 500.0}],
            "GLD": [{"d": "2026-08-15", "p": 200.0}],
            "TLT": [{"d": "2026-08-15", "p": 95.0}],
        }),
        encoding="utf-8",
    )

    res = reconcile_compact_prices_with_market_db(prices_path=prices_path, db_path=db_path)
    assert res["status"] == "fail"
    assert res["symbols_checked"] == 3
    issues = {off["issue"] for off in res["top_offenders"]}
    assert "missing_sqlite_symbol" in issues
    assert "stale_latest_date" in issues
    assert "latest_price_divergence" in issues


def test_require_true_ohlc_price_rows() -> None:
    clean_rows = [{"symbol": "SPY", "date": "2026-08-15", "open": 500.0, "high": 502.0, "low": 498.0, "close": 501.0}]
    assert require_true_ohlc_price_rows(clean_rows) == clean_rows

    proxy_rows = [{"symbol": "SPY", "date": "2026-08-15", "is_adjusted_close_proxy": True}]
    assert is_adjusted_close_proxy_price_row(proxy_rows[0]) is True
    with pytest.raises(MarketDataSemanticsError):
        require_true_ohlc_price_rows(proxy_rows)


def test_reconcile_price_providers() -> None:
    primary = [
        {"symbol": "SPY", "date": "2026-08-15", "adj_close": 500.0},
        {"symbol": "GLD", "date": "2026-08-15", "adj_close": 200.0},
    ]
    secondary = [
        {"symbol": "SPY", "date": "2026-08-15", "adj_close": 500.1},
        {"symbol": "GLD", "date": "2026-08-14", "adj_close": 200.0},  # lags by 1d
    ]
    res = reconcile_price_providers(primary, secondary, max_latest_lag_days=0)
    assert res["status"] == "warning"
    assert "stale_latest_date" in res["issue_counts"]


def test_broker_market_data_consistency_report(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    _init_market_db(db_path, [
        ("SPY", "2026-08-15", 500.0),
        ("GLD", "2026-08-01", 200.0),  # old date -> stale
        ("TLT", "2026-08-15", 100.0),  # broker will have 110.0 -> diverged
    ])

    positions = [
        {"symbol": "SPY", "current_price": 500.5},   # <1% difference -> ok
        {"symbol": "GLD", "current_price": 200.0},   # stale local
        {"symbol": "TLT", "current_price": 110.0},   # 10% diff -> diverged
        {"symbol": "IEF", "current_price": 95.0},    # missing local
    ]

    fixed_now = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
    res = broker_market_data_consistency_report(
        positions=positions,
        db_path=db_path,
        now=fixed_now,
        max_local_age_days=3,
        warn_threshold_pct=2.0,
    )

    assert res["status"] == "warning"
    assert len(res["rows"]) == 4
    statuses = {r["symbol"]: r["status"] for r in res["rows"]}
    assert statuses["SPY"] == "ok"
    assert statuses["GLD"] == "stale_local"
    assert statuses["TLT"] == "diverged"
    assert statuses["IEF"] == "missing_local"
