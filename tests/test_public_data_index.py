"""Tests for the public data index manifest contract."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.dashboard.generator import DashboardGenerator


def _create_market_db(db_path: Path, days: int = 30) -> None:
    """Create a minimal market database for dashboard generation."""
    conn = sqlite3.connect(str(db_path))
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS regime_log (
            date TEXT,
            regime TEXT,
            vix_level REAL,
            detected_at TEXT
        )
        """
    )
    base_date = datetime.now()
    for symbol in ("SPY", "GLD", "TLT", "QQQ"):
        for offset in range(days):
            date = (base_date - timedelta(days=offset)).strftime("%Y-%m-%d")
            close = round(500 + np.random.normal(0, 2.0), 2)
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)", (symbol, date, close))
    conn.commit()
    conn.close()


def _make_generator(tmp_path: Path) -> DashboardGenerator:
    db_path = tmp_path / "market.db"
    _create_market_db(db_path)
    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen.conn = sqlite3.connect(str(db_path))
    gen.conn.row_factory = sqlite3.Row
    return gen


def _run_generator(tmp_path: Path) -> dict:
    gen = _make_generator(tmp_path)
    with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            gen.run()
    with open(tmp_path / "index.json") as f:
        return json.load(f)


def _entries_by_filename(index: dict) -> dict[str, dict]:
    return {entry["filename"]: entry for entry in index["entries"]}


def test_public_index_keeps_files_list_and_adds_typed_entries(tmp_path: Path) -> None:
    index = _run_generator(tmp_path)

    assert index["schema_version"] == "public-data-index/v1"
    assert "dashboard.json" in index["files"]
    assert "generated_at" in index

    entries = _entries_by_filename(index)
    dashboard_entry = entries["dashboard.json"]
    assert dashboard_entry["category"] == "dashboard"
    assert dashboard_entry["schema_version"] == "dashboard/v1"
    assert dashboard_entry["status"] == "present"
    assert dashboard_entry["validation_status"] == "not_applicable"
    assert dashboard_entry["size_bytes"] > 0
    assert len(dashboard_entry["sha256"]) == 64
    assert dashboard_entry["generated_at"]


def test_public_index_represents_missing_optional_labs_files(tmp_path: Path) -> None:
    index = _run_generator(tmp_path)

    entries = _entries_by_filename(index)
    labs_registry = entries["labs_registry.json"]
    assert labs_registry["category"] == "labs"
    assert labs_registry["schema_version"] == "labs-registry/v1"
    assert labs_registry["status"] == "missing"
    assert labs_registry["validation_status"] == "missing"
    assert labs_registry["size_bytes"] is None
    assert labs_registry["sha256"] is None
    assert "labs_registry.json" not in index["files"]


def test_public_index_marks_invalid_labs_artifact_without_breaking_run(tmp_path: Path) -> None:
    invalid_registry = {
        "schema_version": "labs-registry/v1",
        "generated_at": "2026-06-08T00:00:00",
        "experiments": [
            {
                "experiment_id": "bad-registry-row",
                "artifact_path": "public/data/bad.json",
                "status": "validated",
                "provenance_status": "present",
                "baseline_deltas": {},
            }
        ],
    }
    (tmp_path / "labs_registry.json").write_text(json.dumps(invalid_registry))

    index = _run_generator(tmp_path)

    labs_registry = _entries_by_filename(index)["labs_registry.json"]
    assert labs_registry["status"] == "present"
    assert labs_registry["validation_status"] == "invalid"
    assert labs_registry["size_bytes"] > 0
    assert len(labs_registry["sha256"]) == 64
    assert any("$.experiments[0].metrics" in error for error in labs_registry["validation_errors"])
