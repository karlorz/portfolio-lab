"""Batch EG: canonical event-time execution timeline (not snapshot rewrites).

Live friction: rebalance_health.execution_history listed the same 2026-07-11
fill from many order-history-YYYY-MM-DD.json rewrites (total_executions=96,
UI five rows all July 11). Schedule already used event-date dedupe (DS);
timeline + total count still used raw parse list.

Research (event time vs processing time): unique execution SLI counts by
event day; keep raw/snapshot rewrite counts as forensic metrics only.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.dashboard.generator import project_execution_timeline_onto_health
from src.monitor.rebalance_health import generate


def _daily_summary(
    dir_path: Path,
    file_date: str,
    event_ts: str,
    symbols: list[str] | None = None,
) -> Path:
    symbols = symbols or ["SPY", "GLD"]
    path = dir_path / f"order-history-{file_date}.json"
    path.write_text(
        json.dumps(
            {
                "date": file_date,
                "write_day": file_date,
                "total_orders": len(symbols),
                "recent_orders": [
                    {
                        "symbol": sym,
                        "side": "buy",
                        "estimated_value": 1000.0,
                        "reason": "rebalance_up",
                        "timestamp": event_ts,
                    }
                    for sym in symbols
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_execution_history_dedupes_daily_snapshot_rewrites() -> None:
    """Many write-day files with same fill event → one timeline row."""
    import src.monitor.rebalance_health as rh

    original_orders = rh.ORDERS_DIR
    try:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # historical_orders empty; daily summaries live in parent of ORDERS_DIR
            hist = root / "historical_orders"
            hist.mkdir()
            rh.ORDERS_DIR = hist
            # Same July-11 fills rewritten on 12 consecutive write days
            for day in range(11, 23):
                file_date = f"2026-07-{day:02d}"
                _daily_summary(
                    root,
                    file_date,
                    event_ts="2026-07-11T00:20:02.531326",
                )
            # Distinct older event
            _daily_summary(
                root,
                "2026-06-15",
                event_ts="2026-05-11T03:20:31.447694",
                symbols=["SPY", "GLD", "TLT"],
            )
            result = generate()
    finally:
        rh.ORDERS_DIR = original_orders

    history = result["execution_history"]
    dates = [row["date"] for row in history]
    # Unique event days only (most recent first)
    assert dates == ["2026-07-11", "2026-05-11"]
    assert result["canonical_execution_days"] == 2
    # Operator-facing total = unique event days, not raw rewrite count
    assert result["total_executions"] == 2
    # Forensics preserved
    assert result["raw_history_entries"] >= 13
    assert result["snapshot_rewrite_files"] >= 10
    assert result["execution_timeline_policy"] == (
        "canonical_event_day; raw rewrites forensic only"
    )


def test_project_execution_timeline_onto_health_flags_rewrite_inflation() -> None:
    health = project_execution_timeline_onto_health(
        {"status": "ok"},
        {
            "canonical_execution_days": 4,
            "total_executions": 4,
            "raw_history_entries": 96,
            "snapshot_rewrite_files": 55,
            "execution_timeline_policy": "canonical_event_day; raw rewrites forensic only",
            "next_rebalance": {
                "last_execution_at": "2026-07-11T00:20:02+00:00",
                "overdue": False,
            },
        },
    )
    assert health["rebalance_unique_execution_days"] == 4
    assert health["rebalance_raw_history_entries"] == 96
    assert health["rebalance_snapshot_rewrite_files"] == 55
    assert health["rebalance_execution_timeline_status"] == "rewrite_inflated"
    assert "rewrite" in (health.get("rebalance_execution_timeline_badge") or "")


def test_project_execution_timeline_ok_when_no_inflation() -> None:
    health = project_execution_timeline_onto_health(
        {"status": "ok"},
        {
            "canonical_execution_days": 3,
            "total_executions": 3,
            "raw_history_entries": 3,
            "snapshot_rewrite_files": 0,
        },
    )
    assert health["rebalance_execution_timeline_status"] == "ok"
    assert health["rebalance_unique_execution_days"] == 3
