"""Tests for broker/local market data consistency reporting."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.broker.alpaca import Position
from src.monitor.market_data_consistency import (
    broker_market_data_consistency_report,
    reconcile_price_providers,
)


def _price_db(path: Path, rows: list[tuple[str, str, float]]) -> Path:
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE prices (
                symbol TEXT,
                date TEXT,
                close REAL,
                PRIMARY KEY (symbol, date)
            )
            """
        )
        conn.executemany("INSERT INTO prices (symbol, date, close) VALUES (?, ?, ?)", rows)
    return path


def test_consistency_report_ok_for_matching_prices(tmp_path: Path) -> None:
    db = _price_db(tmp_path / "market.db", [("SPY", "2026-06-11", 600.0)])
    positions = [Position("SPY", 10, 500.0, 600.5, 6005.0, 1005.0, 0.2)]

    report = broker_market_data_consistency_report(
        positions,
        db_path=db,
        warn_threshold_pct=1.0,
        now=datetime(2026, 6, 11, 22, tzinfo=timezone.utc),
    )

    assert report["status"] == "ok"
    assert report["rows"][0]["difference_pct"] == 0.0833
    assert report["warnings"] == []


def test_consistency_report_warns_on_material_divergence(tmp_path: Path) -> None:
    db = _price_db(tmp_path / "market.db", [("SPY", "2026-06-11", 600.0)])
    positions = [Position("SPY", 10, 500.0, 630.0, 6300.0, 1300.0, 0.26)]

    report = broker_market_data_consistency_report(
        positions,
        db_path=db,
        warn_threshold_pct=2.0,
        now=datetime(2026, 6, 11, 22, tzinfo=timezone.utc),
    )

    assert report["status"] == "warning"
    assert report["rows"][0]["status"] == "diverged"
    assert "broker/local price differs" in report["warnings"][0]


def test_consistency_report_warns_on_stale_local_price(tmp_path: Path) -> None:
    db = _price_db(tmp_path / "market.db", [("SPY", "2026-06-01", 600.0)])
    positions = [Position("SPY", 10, 500.0, 600.0, 6000.0, 1000.0, 0.2)]

    report = broker_market_data_consistency_report(
        positions,
        db_path=db,
        max_local_age_days=3,
        now=datetime(2026, 6, 11, 22, tzinfo=timezone.utc),
    )

    assert report["status"] == "warning"
    assert report["rows"][0]["status"] == "stale_local"
    assert "local market data stale" in report["warnings"][0]


def test_consistency_report_degrades_without_broker_credentials() -> None:
    with patch.dict("os.environ", {}, clear=True):
        report = broker_market_data_consistency_report(positions=None)

    assert report["status"] == "unavailable"
    assert report["reason"] == "alpaca_not_configured"
    assert report["rows"] == []


def test_provider_reconciliation_passes_for_matching_adjusted_close_rows() -> None:
    primary_rows = [
        {"symbol": "SPY", "date": "2026-06-11", "adj_close": 600.0},
        {"symbol": "GLD", "date": "2026-06-11", "adj_close": 310.0},
    ]
    secondary_rows = [
        {"symbol": "SPY", "date": "2026-06-11", "adjusted_close": 600.05},
        {"symbol": "GLD", "date": "2026-06-11", "adjusted_close": 310.02},
    ]

    report = reconcile_price_providers(
        primary_rows,
        secondary_rows,
        primary_provider="Licensed Fixture",
        secondary_provider="Yahoo Fixture",
        required_symbols=["SPY", "GLD"],
        adjusted_close_tolerance_pct=0.05,
        max_latest_lag_days=0,
    )

    assert report["status"] == "ok"
    assert report["failure_type"] is None
    assert report["symbols_checked"] == 2
    assert report["issue_counts"] == {}
    assert report["top_offenders"] == []


def test_provider_reconciliation_flags_adjusted_close_divergence_above_tolerance() -> None:
    primary_rows = [{"symbol": "SPY", "date": "2026-06-11", "adj_close": 612.0}]
    secondary_rows = [{"symbol": "SPY", "date": "2026-06-11", "adj_close": 600.0}]

    report = reconcile_price_providers(
        primary_rows,
        secondary_rows,
        primary_provider="Licensed Fixture",
        secondary_provider="Yahoo Fixture",
        required_symbols=["SPY"],
        adjusted_close_tolerance_pct=1.0,
    )

    assert report["status"] == "warning"
    assert report["failure_type"] == "provider_divergence"
    assert report["issue_counts"] == {"adjusted_close_divergence": 1}
    assert report["top_offenders"][0]["symbol"] == "SPY"
    assert report["top_offenders"][0]["issue"] == "adjusted_close_divergence"
    assert report["top_offenders"][0]["difference_pct"] == 2.0


def test_provider_reconciliation_flags_missing_symbols_and_stale_latest_dates() -> None:
    primary_rows = [
        {"symbol": "SPY", "date": "2026-06-11", "adj_close": 600.0},
        {"symbol": "GLD", "date": "2026-06-10", "adj_close": 310.0},
    ]
    secondary_rows = [
        {"symbol": "SPY", "date": "2026-06-10", "adj_close": 600.0},
    ]

    report = reconcile_price_providers(
        primary_rows,
        secondary_rows,
        primary_provider="Licensed Fixture",
        secondary_provider="Yahoo Fixture",
        required_symbols=["SPY", "GLD"],
        max_latest_lag_days=0,
    )

    assert report["status"] == "warning"
    assert report["failure_type"] == "provider_divergence"
    assert report["issue_counts"] == {
        "missing_symbol": 1,
        "stale_latest_date": 2,
    }
    assert {offender["issue"] for offender in report["top_offenders"]} == {
        "missing_symbol",
        "stale_latest_date",
    }
    assert "Yahoo Fixture missing GLD" in report["message"]


def test_provider_reconciliation_classifies_empty_provider_as_outage() -> None:
    report = reconcile_price_providers(
        [{"symbol": "SPY", "date": "2026-06-11", "adj_close": 600.0}],
        [],
        primary_provider="Licensed Fixture",
        secondary_provider="Yahoo Fixture",
        required_symbols=["SPY"],
    )

    assert report["status"] == "unavailable"
    assert report["failure_type"] == "provider_outage"
    assert report["outage_provider"] == "Yahoo Fixture"
    assert report["issue_counts"] == {"provider_outage": 1}
