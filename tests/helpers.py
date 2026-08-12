#!/usr/bin/env python3
"""
Shared plain-function helpers for the test_generator split family
(TEST-GENERATOR-SPLIT, 2026-08-12).

Moved VERBATIM from tests/test_generator.py (no cleanup/refactor while
moving — fixture/helper relocation must not change behavior). This is a
plain module (NOT conftest.py) so importing it has no pytest-wide side
effects; the autouse ``_isolate_live_ensemble_and_ic_health`` fixture does
NOT live here — it is defined verbatim in each split test file (moving it
to conftest.py would pollute the full ~15k-test suite).
"""
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.dashboard.generator import DashboardGenerator


def _create_market_db(db_path, symbols=None, days=30, base_price=500.0):
    """Create a market.db with price data for testing."""
    if symbols is None:
        symbols = ['SPY', 'GLD', 'TLT', 'QQQ']
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
        PRIMARY KEY (symbol, date))
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS regime_log (
            date TEXT, regime TEXT, vix_level REAL, detected_at TEXT
        )
    """)
    base_date = datetime.now()
    for sym in symbols:
        for i in range(days):
            d = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            noise = np.random.normal(0, 2.0)
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                         (sym, d, round(base_price + noise, 2)))
    conn.commit()
    conn.close()


def _make_generator(tmp_path):
    """Create a DashboardGenerator with a test database."""
    db_path = tmp_path / "market.db"
    _create_market_db(db_path)
    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen.conn = sqlite3.connect(str(db_path))
    gen.conn.row_factory = sqlite3.Row
    return gen, db_path


def _write_ok_source_manifest(public_dir: Path) -> None:
    """Write a compact live source manifest for healthy health-json fixtures."""
    (public_dir / "source_manifest.json").write_text(json.dumps({
        "artifacts": [
            {
                "artifact": "prices.json",
                "provider": "Yahoo Finance",
                "feed": "chart/v8",
                "source_mode": "live",
                "status": "success",
                "data_quality": {
                    "artifact": "data_quality.json",
                    "schema_version": "price-data-quality/v1",
                    "status": "ok",
                    "issue_counts": {"total": 0},
                },
            },
        ],
    }))


def _write_data_quality_report(public_dir: Path, *, status: str = "ok", stale_latest_dates: int = 0) -> None:
    """Write a compact current data_quality.json report for alert/SLO fixtures."""
    issue_counts = {
        "duplicate_dates": 0,
        "empty_symbols": 0,
        "extreme_returns": 0,
        "internal_gaps": 0,
        "invalid_dates": 0,
        "invalid_prices": 0,
        "missing_required_keys": 0,
        "non_monotonic_rows": 0,
        "non_object_records": 0,
        "split_like_returns": 0,
        "stale_latest_dates": stale_latest_dates,
        "total": stale_latest_dates,
    }
    (public_dir / "data_quality.json").write_text(json.dumps({
        "artifact": "data_quality.json",
        "schema_version": "price-data-quality/v1",
        "generated_at": "2026-06-16T12:00:00Z",
        "status": status,
        "issue_counts": issue_counts,
        "symbols": [
            {"symbol": "SPY", "status": "ok", "latest_date": "2026-06-15"},
            {
                "symbol": "GLD",
                "status": "fail" if stale_latest_dates else "ok",
                "latest_date": "2026-06-11" if stale_latest_dates else "2026-06-15",
                "stale_latest_date": {
                    "reference_date": "2026-06-15",
                    "latest_date": "2026-06-11",
                    "latest_lag_days": 2,
                } if stale_latest_dates else None,
            },
        ],
    }))
