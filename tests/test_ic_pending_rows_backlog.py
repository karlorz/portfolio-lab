"""IC decay report discloses row-level prediction backlog vs staged pending."""
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest


def test_compute_ic_decay_report_includes_pending_rows(tmp_path, monkeypatch):
    from src.monitor import ic_decay_monitor as icm

    db = tmp_path / "market.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE signal_predictions (
            prediction_date TEXT,
            actual_direction REAL
        )
        """
    )
    # 3 pending, 1 resolved
    conn.executemany(
        "INSERT INTO signal_predictions VALUES (?, ?)",
        [
            ("2026-05-01", None),
            ("2026-06-01", None),
            ("2026-07-01", None),
            ("2026-07-10", 1.0),
        ],
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(icm, "MARKET_DB", db, raising=False)
    # force helper to use our db via MARKET_DB path patch on paths module
    with patch("src.paths.MARKET_DB", db):
        # empty staged state
        monkeypatch.setattr(icm, "IC_STATE_PATH", tmp_path / "missing_ic_state.json")
        report = icm.compute_ic_decay_report()
    assert report["pending_rows"] == 3
    assert report["pending_dates"] == 3
    assert report["oldest_unresolved_date"] == "2026-05-01"
    assert report["total_predictions"] == 4
    assert report["resolved_predictions"] == 1
    assert "pending_predictions" in report
    assert "staged" in (report.get("pending_semantics") or "").lower() or "pending_rows" in (
        report.get("pending_semantics") or ""
    )


def test_backlog_uses_timestamp_column_like_production(tmp_path):
    """Production signal_predictions has timestamp, not prediction_date."""
    from src.monitor.ic_decay_monitor import _signal_prediction_backlog

    db = tmp_path / "market.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE signal_predictions (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            source TEXT,
            actual_direction INTEGER
        )
        """
    )
    conn.executemany(
        "INSERT INTO signal_predictions (timestamp, source, actual_direction) VALUES (?,?,?)",
        [
            ("2026-05-24T10:00:00", "a", None),
            ("2026-06-01T10:00:00", "a", None),
            ("2026-07-01T10:00:00", "a", 1),
        ],
    )
    conn.commit()
    conn.close()

    backlog = _signal_prediction_backlog(db)
    assert backlog["pending_rows"] == 2
    assert backlog["pending_dates"] == 2
    assert backlog["oldest_unresolved_date"] == "2026-05-24"
    assert backlog["total_predictions"] == 3
    assert backlog["resolved_predictions"] == 1
