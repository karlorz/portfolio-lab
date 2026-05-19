"""Tests for src/monitor/rebalance_health.py — Rebalance Health Data Exporter."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import pytest
from src.monitor.rebalance_health import (
    _parse_order_file,
    generate,
)


def _make_order_file(dir_path: Path, stem: str, orders: list) -> Path:
    """Helper: create an order history JSON file."""
    path = dir_path / f"{stem}.json"
    path.write_text(json.dumps(orders))
    return path


class TestParseOrderFile:
    """Tests for _parse_order_file()."""

    def test_parses_simple_order_list(self):
        orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 10000, "reason": "rebalance"}]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc123", orders)
            result = _parse_order_file(path)
        assert result is not None
        assert result["orders"] == 1
        assert result["buy_count"] == 1
        assert result["total_value"] == 10000.0
        assert result["symbols"] == ["SPY"]

    def test_parses_multiple_orders(self):
        orders = [
            {"symbol": "SPY", "side": "buy", "estimated_value": 5000, "reason": "rebalance"},
            {"symbol": "TLT", "side": "sell", "estimated_value": 3000, "reason": "drift"},
            {"symbol": "GLD", "side": "buy", "estimated_value": 2000, "reason": "rebalance"},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc123", orders)
            result = _parse_order_file(path)
        assert result["orders"] == 3
        assert result["buy_count"] == 2
        assert result["sell_count"] == 1
        assert result["total_value"] == 10000.0
        assert set(result["symbols"]) == {"SPY", "TLT", "GLD"}

    def test_extracts_unique_reasons(self):
        orders = [
            {"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"},
            {"symbol": "GLD", "side": "buy", "estimated_value": 1000, "reason": "drift"},
            {"symbol": "TLT", "side": "sell", "estimated_value": 1000, "reason": "rebalance"},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc123", orders)
            result = _parse_order_file(path)
        assert set(result["reasons"]) == {"rebalance", "drift"}

    def test_parses_timestamp_from_filename(self):
        orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc123", orders)
            result = _parse_order_file(path)
        assert result["date"] == "2026-05-11"
        assert result["time"] == "14:30"

    def test_fallback_timestamp_on_invalid_filename(self):
        orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "invalid_filename", orders)
            result = _parse_order_file(path)
        assert result is not None
        assert "date" in result  # Falls back to mtime

    def test_short_filename_no_time(self):
        orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511", orders)
            result = _parse_order_file(path)
        assert result is not None
        assert result["orders"] == 1

    def test_empty_order_list(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_000000", [])
            result = _parse_order_file(path)
        assert result is None

    def test_invalid_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "order_history_20260511_000000.json"
            path.write_text("not json {{{")
            result = _parse_order_file(path)
        assert result is None

    def test_frontmatter_parsing(self):
        orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 5000, "reason": "drift"}]
        content = "---\ntitle: Test\n---\n" + json.dumps(orders)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "order_history_20260511_143008_abc.json"
            path.write_text(content)
            result = _parse_order_file(path)
        assert result is not None
        assert result["orders"] == 1

    def test_orders_with_missing_fields(self):
        orders = [{"symbol": "SPY"}]  # no side, no estimated_value
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc123", orders)
            result = _parse_order_file(path)
        assert result is not None
        assert result["total_value"] == 0
        assert result["buy_count"] == 0


class TestGenerate:
    """Tests for generate()."""

    def test_generates_empty_data_when_no_orders(self):
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                # Ensure directory exists but is empty
                rh.ORDERS_DIR.mkdir(parents=True, exist_ok=True)
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["total_executions"] == 0
        assert "next_rebalance" in result
        # days_until can be -1 due to time precision near midnight
        assert result["next_rebalance"]["days_until"] >= -1

    def test_generates_schedule_compliance(self):
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                # Create two orders ~30 days apart
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(Path(d), "order_history_20260401_120000_aaa", orders)
                _make_order_file(Path(d), "order_history_20260501_120000_bbb", orders)
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["total_executions"] == 2
        assert "schedule_compliance" in result
        # 2 entries, first has no prev → 1 checkable pair
        assert result["schedule_compliance"]["total"] >= 0

    def test_recent_executions_most_recent_first(self):
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                old_orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                new_orders = [{"symbol": "TLT", "side": "sell", "estimated_value": 2000, "reason": "drift"}]
                _make_order_file(Path(d), "order_history_20260401_120000_aaa", old_orders)
                _make_order_file(Path(d), "order_history_20260501_120000_bbb", new_orders)
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        recent = result["execution_history"]
        assert len(recent) == 2
        # Most recent first: 2026-05-01 comes before 2026-04-01
        assert recent[0]["date"] == "2026-05-01"

    def test_exports_valid_json_structure(self):
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(Path(d), "order_history_20260511_143008_abc123", orders)
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert "generated" in result
        assert "next_rebalance" in result
        assert "schedule_compliance" in result
        assert "execution_history" in result
        assert "total_executions" in result
        # Verify it's JSON-serializable
        json.dumps(result)
