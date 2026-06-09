"""Integration smoke tests for offline Labs dashboard artifact publication."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.dashboard.generator import DashboardGenerator


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _create_market_db(db_path: Path, days: int = 30) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
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
        CREATE TABLE regime_log (
            date TEXT,
            regime TEXT,
            vix_level REAL,
            detected_at TEXT
        )
        """
    )
    base_date = datetime(2026, 6, 9)
    for symbol_index, symbol in enumerate(("SPY", "GLD", "TLT", "QQQ")):
        for offset in range(days):
            date = (base_date - timedelta(days=offset)).strftime("%Y-%m-%d")
            close = round(100 + symbol_index * 25 + offset * 0.1, 2)
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)", (symbol, date, close))
    conn.commit()
    conn.close()


def _run_dashboard_generation(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public"
    db_path = tmp_path / "market.db"
    _create_market_db(db_path)
    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen.conn = sqlite3.connect(str(db_path))
    gen.conn.row_factory = sqlite3.Row
    with patch("src.dashboard.generator.DATA_DIR", data_dir):
        with patch("src.dashboard.generator.PUBLIC_DIR", public_dir):
            gen.run()
    return json.loads((public_dir / "index.json").read_text()), data_dir, public_dir


def _entries_by_filename(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["filename"]: entry for entry in index["entries"]}


def _artifact_generated_at(path: Path) -> str:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return str(payload["generated_at"])
    if isinstance(payload, list) and payload:
        return str(payload[0]["generated_at"])
    raise AssertionError(f"{path.name} does not expose generated_at metadata")


def _assert_indexed_labs_artifact(
    *,
    index: dict[str, Any],
    filename: str,
    expected_schema_version: str,
    public_dir: Path,
) -> None:
    entries = _entries_by_filename(index)
    entry = entries[filename]
    artifact_path = public_dir / filename

    assert artifact_path.exists()
    assert filename in index["files"]
    assert entry["category"] == "labs"
    assert entry["schema_version"] == expected_schema_version
    assert entry["status"] == "present"
    assert entry["validation_status"] == "valid"
    assert entry["validation_errors"] == []
    assert entry["size_bytes"] == artifact_path.stat().st_size
    assert entry["size_budget"]["status"] == "within_budget"
    assert entry["size_budget"]["render_strategy"] == "direct"
    assert len(entry["sha256"]) == 64
    assert entry["generated_at"] == _artifact_generated_at(artifact_path)


def test_dashboard_generation_publishes_and_indexes_available_labs_artifacts(tmp_path: Path) -> None:
    marker = tmp_path / "unsafe-replay-ran.txt"
    data_dir = tmp_path / "data"
    _write_json(
        data_dir / "backtest_results" / "labs_smoke_results.json",
        {
            "experiment_id": "labs-smoke-candidate",
            "sharpe_ratio": 1.05,
            "cagr": 10.2,
            "volatility": 11.4,
            "max_drawdown": -18.0,
            "baseline_sharpe": 0.97,
            "baseline_cagr": 9.8,
            "baseline_max_dd": -20.0,
            "dsr": 0.98,
            "wfe": 1.1,
        },
    )
    _write_json(
        data_dir / "labs_replay_targets.json",
        [
            {
                "experiment_id": "labs-smoke-unsafe-replay",
                "artifact_path": "data/labs-smoke-unsafe-replay.json",
                "status": "candidate",
                "provenance_status": "sidecar",
                "metrics": {"sharpe": 0.95},
                "baseline_deltas": {},
                "command": f"python -c \"from pathlib import Path; Path({str(marker)!r}).write_text('ran')\"",
                "replay_safe": False,
                "fetches_market_data": False,
            }
        ],
    )

    index, _data_dir, public_dir = _run_dashboard_generation(tmp_path)

    expected_labs_artifacts = {
        "labs_registry.json": "labs-registry/v1",
        "labs_scorecards.json": "labs-scorecard/v1",
        "labs_replays.json": "labs-replay/v1",
        "labs_validation.json": "labs-validation/v1",
    }
    generated_labs_files = {path.name for path in public_dir.glob("*labs*.json")}
    assert marker.exists() is False
    assert expected_labs_artifacts.keys() <= generated_labs_files
    assert generated_labs_files <= set(index["files"])

    for filename, schema_version in expected_labs_artifacts.items():
        _assert_indexed_labs_artifact(
            index=index,
            filename=filename,
            expected_schema_version=schema_version,
            public_dir=public_dir,
        )

    registry = json.loads((public_dir / "labs_registry.json").read_text())
    assert registry["experiments"][0]["experiment_id"] == "labs-smoke-candidate"

    replay_rows = json.loads((public_dir / "labs_replays.json").read_text())
    assert replay_rows[0]["experiment_id"] == "labs-smoke-unsafe-replay"
    assert replay_rows[0]["status"] == "warning"
    assert replay_rows[0]["failure_reason"] == "safety_skip"


def test_dashboard_generation_indexes_missing_optional_labs_endpoints(tmp_path: Path) -> None:
    index, _data_dir, public_dir = _run_dashboard_generation(tmp_path)
    entries = _entries_by_filename(index)

    for filename in ("labs_registry.json", "labs_scorecards.json", "labs_replays.json"):
        entry = entries[filename]
        assert filename not in index["files"]
        assert entry["category"] == "labs"
        assert entry["status"] == "missing"
        assert entry["validation_status"] == "missing"
        assert entry["size_bytes"] is None
        assert entry["size_budget"]["status"] == "missing"
        assert entry["size_budget"]["render_strategy"] == "missing"
        assert entry["sha256"] is None
        assert entry["generated_at"] == index["generated_at"]

    _assert_indexed_labs_artifact(
        index=index,
        filename="labs_validation.json",
        expected_schema_version="labs-validation/v1",
        public_dir=public_dir,
    )
    validation_report = json.loads((public_dir / "labs_validation.json").read_text())
    assert validation_report["results"] == []
