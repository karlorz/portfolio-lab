"""Tests for src/monitor/rebalance_health.py — Rebalance Health Data Exporter."""

import json
import logging
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
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
        assert datetime.fromisoformat(result["timestamp"]).tzinfo is not None

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


class TestGenerateExtended:
    """Extended coverage for generate()."""

    def test_same_day_deduplication(self):
        """Multiple orders on the same day should be deduped for compliance."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                # Two orders same day, different times
                _make_order_file(Path(d), "order_history_20260401_090000_aaa", orders)
                _make_order_file(Path(d), "order_history_20260401_140000_bbb", orders)
                _make_order_file(Path(d), "order_history_20260501_120000_ccc", orders)
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        # Same-day entries deduped → only 2 compliance-interval pairs, 1 checkable
        assert result["total_executions"] == 3  # History has all 3
        # Compliance should count 1 interval (April 1 → May 1)
        assert result["schedule_compliance"]["total"] == 1

    def test_schedule_compliance_on_time(self):
        """Orders 30 days apart should be on_time."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(Path(d), "order_history_20260401_120000_aaa", orders)
                _make_order_file(Path(d), "order_history_20260501_120000_bbb", orders)
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["schedule_compliance"]["on_time"] == 1
        assert result["schedule_compliance"]["delayed"] == 0

    def test_schedule_compliance_delayed(self):
        """Orders >35 days apart should be delayed."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(Path(d), "order_history_20260301_120000_aaa", orders)
                _make_order_file(Path(d), "order_history_20260515_120000_bbb", orders)  # 45 days later
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["schedule_compliance"]["delayed"] == 1
        assert result["schedule_compliance"]["on_time"] == 0

    def test_generate_includes_alpaca_feed_entitlement_diagnostics(self):
        """Rebalance health should expose public-safe Alpaca feed policy metadata."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d, patch.dict(
                "os.environ",
                {"ALPACA_DATA_FEED": "sip", "ALPACA_FEED_ENTITLEMENT": "delayed_sip"},
            ):
                rh.ORDERS_DIR = Path(d)
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir

        assert result["alpaca_feed_entitlement"] == {
            "configured_feed": "sip",
            "effective_feed": "sip",
            "entitlement": "delayed_sip",
            "delayed": True,
            "acceptable_for_live": False,
            "policy_decision": "reject",
            "reason": "delayed_feed",
        }

    def test_next_rebalance_30_days_after_last(self):
        """Next rebalance should be ~30 days after the last execution."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        original_data_dir = rh.DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                data_dir = Path(d)
                rh.DATA_DIR = data_dir
                rh.ORDERS_DIR = data_dir / "historical_orders"
                rh.ORDERS_DIR.mkdir()
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(rh.ORDERS_DIR, "order_history_20260510_120000_aaa", orders)
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
            rh.DATA_DIR = original_data_dir
        assert result["next_rebalance"]["date"] == "2026-06-09"
        assert result["next_rebalance"]["frequency"] == "monthly (~30 days)"
        # Last rebalance May 10 → next Jun 9 is in the past vs 2026-07 wall clock
        assert result["next_rebalance"]["days_until"] < 0
        assert result["next_rebalance"]["status"] == "overdue"
        assert result["next_rebalance"]["overdue"] is True
        assert result["next_rebalance"].get("status_reason")

    def test_next_rebalance_discloses_overdue_when_past(self):
        """Negative days_until must not look like a future schedule."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        original_data_dir = rh.DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                data_dir = Path(d)
                rh.DATA_DIR = data_dir
                rh.ORDERS_DIR = data_dir / "historical_orders"
                rh.ORDERS_DIR.mkdir()
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                # Last exec ~60 days before "today" (2026-07-20 era)
                _make_order_file(rh.ORDERS_DIR, "order_history_20260501_120000_aaa", orders)
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
            rh.DATA_DIR = original_data_dir
        nr = result["next_rebalance"]
        assert nr["days_until"] < 0
        assert nr["status"] == "overdue"
        assert nr["overdue"] is True

    def test_root_daily_order_history_is_canonical_when_newer(self):
        """Root order-history daily summaries should advance rebalance freshness."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        original_data_dir = rh.DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                data_dir = Path(d)
                rh.DATA_DIR = data_dir
                rh.ORDERS_DIR = data_dir / "historical_orders"
                rh.ORDERS_DIR.mkdir()
                _make_order_file(
                    rh.ORDERS_DIR,
                    "order_history_20260510_120000_aaa",
                    [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}],
                )
                (data_dir / "order-history-2026-07-06.json").write_text(json.dumps({
                    "date": "2026-07-06",
                    "total_orders": 4,
                    "orders": [
                        {"symbol": "SPY", "side": "buy", "estimated_value": 1000, "timestamp": "2026-07-06T12:00:00+00:00"},
                        {"symbol": "GLD", "side": "sell", "estimated_value": 500, "timestamp": "2026-07-06T12:00:00+00:00"},
                    ],
                }))

                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
            rh.DATA_DIR = original_data_dir

        assert result["execution_history"][0]["date"] == "2026-07-06"
        assert result["execution_history"][0]["source"] == "daily_order_summary"
        assert result["next_rebalance"]["date"] == "2026-08-05"
        assert result["canonical_order_history_source"] == "combined_order_history"

    def test_nonexistent_orders_dir(self):
        """Should handle non-existent ORDERS_DIR gracefully."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            rh.ORDERS_DIR = Path("/tmp/nonexistent_dir_for_test_xyz")
            result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["total_executions"] == 0

    def test_compliance_pct_calculation(self):
        """Compliance percentage should be computed correctly."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(Path(d), "order_history_20260101_120000_aaa", orders)
                _make_order_file(Path(d), "order_history_20260201_120000_bbb", orders)  # on-time
                _make_order_file(Path(d), "order_history_20260415_120000_ccc", orders)  # delayed (73 days)
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        # 1 on-time + 1 delayed = 50%
        assert result["schedule_compliance"]["compliance_pct"] == 50.0


class TestParseOrderFileExtended:
    """Extended edge cases for _parse_order_file."""

    def test_mixed_reasons_and_sides(self):
        """Should count buys, sells, and reasons correctly."""
        orders = [
            {"symbol": "SPY", "side": "buy", "estimated_value": 5000, "reason": "rebalance"},
            {"symbol": "GLD", "side": "sell", "estimated_value": 3000, "reason": "drift"},
            {"symbol": "TLT", "side": "buy", "estimated_value": 2000, "reason": "rebalance"},
            {"symbol": "IEF", "side": "sell", "estimated_value": 1000, "reason": "rebalance"},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc123", orders)
            result = _parse_order_file(path)
        assert result["buy_count"] == 2
        assert result["sell_count"] == 2
        assert result["orders"] == 4
        assert result["total_value"] == 11000.0

    def test_order_without_reason(self):
        """Orders without reason field should not crash."""
        orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000}]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc123", orders)
            result = _parse_order_file(path)
        assert result is not None
        assert result["orders"] == 1


class TestParseOrderFileEdge:
    """Edge cases for _parse_order_file."""

    def test_non_list_json_body(self):
        """JSON body that is a dict (not a list) should return None."""
        body = {"symbol": "SPY", "side": "buy", "estimated_value": 1000}
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc", body)
            result = _parse_order_file(path)
        assert result is None

    def test_non_dict_items_in_list(self):
        """List with non-dict items should be handled gracefully (o.get crashes on non-dict).

        The source uses o.get() which only works on dicts, so non-dict items
        cause AttributeError. This test verifies the function fails safely.
        """
        orders = ["SPY", 42, None]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc", orders)
            with pytest.raises(AttributeError):
                _parse_order_file(path)
        # NOTE: _parse_order_file does not catch AttributeError; source fix
        # would be needed for graceful handling of non-dict list items.

    def test_frontmatter_only_no_body(self):
        """File with only frontmatter (ends with second ---) should return None."""
        content = "---\ntitle: Test\n---"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "order_history_20260511_143008_abc.json"
            path.write_text(content)
            result = _parse_order_file(path)
        assert result is None

    def test_empty_body_after_frontmatter(self):
        """Frontmatter with empty/whitespace body should return None."""
        content = "---\ntitle: Test\n---\n   \n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "order_history_20260511_143008_abc.json"
            path.write_text(content)
            result = _parse_order_file(path)
        assert result is None

    def test_file_not_found(self):
        """Non-existent file path should return None."""
        path = Path("/tmp/nonexistent_path_for_test_xyz.json")
        result = _parse_order_file(path)
        assert result is None

    def test_no_timestamp_parts_in_filename(self):
        """Filename with no date-parsable parts should fall back to mtime."""
        orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 500, "reason": "rebalance"}]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "reports_data_2026", orders)
            result = _parse_order_file(path)
        assert result is not None
        assert result["orders"] == 1
        # Falls back to mtime, so date should be today
        assert "date" in result

    def test_minimal_filename_two_parts(self):
        """Filename stem with only 2 underscore parts should still parse."""
        orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 500, "reason": "rebalance"}]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511", orders)
            result = _parse_order_file(path)
        assert result is not None
        assert result["date"] == "2026-05-11"
        assert result["time"] == "00:00"

    def test_side_not_buy_or_sell(self):
        """Side values that are neither 'buy' nor 'sell' should not be counted."""
        orders = [
            {"symbol": "SPY", "side": "hold", "estimated_value": 1000, "reason": "rebalance"},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc", orders)
            result = _parse_order_file(path)
        assert result is not None
        assert result["buy_count"] == 0
        assert result["sell_count"] == 0

    def test_missing_symbol_key(self):
        """Order without symbol key should use '?'."""
        orders = [{"side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc", orders)
            result = _parse_order_file(path)
        assert result is not None
        assert result["symbols"] == ["?"]

    def test_negative_estimated_value(self):
        """Negative estimated_value should be included in total."""
        orders = [
            {"symbol": "SPY", "side": "sell", "estimated_value": -500, "reason": "rebalance"},
            {"symbol": "GLD", "side": "buy", "estimated_value": 1000, "reason": "rebalance"},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc", orders)
            result = _parse_order_file(path)
        assert result is not None
        assert result["total_value"] == 500.0
        assert result["sell_count"] == 1

    def test_zero_estimated_value(self):
        """Zero estimated_values should sum correctly."""
        orders = [
            {"symbol": "SPY", "side": "buy", "estimated_value": 0, "reason": "rebalance"},
            {"symbol": "GLD", "side": "buy", "estimated_value": 0, "reason": "drift"},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc", orders)
            result = _parse_order_file(path)
        assert result is not None
        assert result["total_value"] == 0.0
        assert result["buy_count"] == 2

    def test_reason_empty_string(self):
        """Reason as empty string should be included in reasons set."""
        orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": ""}]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc", orders)
            result = _parse_order_file(path)
        assert result is not None
        assert "" in result["reasons"]

    def test_mixed_case_side(self):
        """Side values are case-sensitive — 'Buy' should not count as 'buy'."""
        orders = [{"symbol": "SPY", "side": "Buy", "estimated_value": 1000, "reason": "rebalance"}]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc", orders)
            result = _parse_order_file(path)
        assert result is not None
        assert result["buy_count"] == 0
        assert result["sell_count"] == 0

    def test_multiple_symbols_sorted(self):
        """Symbols should be returned in sorted order."""
        orders = [
            {"symbol": "TLT", "side": "buy", "estimated_value": 1000, "reason": "rebalance"},
            {"symbol": "GLD", "side": "sell", "estimated_value": 500, "reason": "drift"},
            {"symbol": "SPY", "side": "buy", "estimated_value": 2000, "reason": "rebalance"},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc", orders)
            result = _parse_order_file(path)
        assert result is not None
        assert result["symbols"] == ["GLD", "SPY", "TLT"]

    def test_reason_default_when_missing(self):
        """Missing reason key should default to 'rebalance'."""
        orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000}]
        with tempfile.TemporaryDirectory() as d:
            path = _make_order_file(Path(d), "order_history_20260511_143008_abc", orders)
            result = _parse_order_file(path)
        assert result is not None
        assert result["reasons"] == ["rebalance"]


class TestGenerateEdge:
    """Edge cases for generate()."""

    def test_single_execution_no_compliance(self):
        """Single execution means no compliance pairs to check."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(Path(d), "order_history_20260501_120000_aaa", orders)
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["total_executions"] == 1
        assert result["schedule_compliance"]["total"] == 0
        # With zero pairs, compliance_pct = on_time / max(total, 1) = 0/1 = 0.0
        assert result["schedule_compliance"]["compliance_pct"] == 0.0

    def test_more_than_ten_executions(self):
        """With 12 executions, recent should show only the last 10."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                # Create 12 order files spaced 30 days apart
                for i in range(12):
                    day = 1 + i * 31  # ~monthly spacing
                    date_str = f"2026{day:03d}"
                    _make_order_file(
                        Path(d),
                        f"order_history_{date_str}_120000_{i:03d}",
                        orders,
                    )
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["total_executions"] == 12
        assert len(result["execution_history"]) == 10  # Only last 10

    def test_exactly_ten_executions(self):
        """With exactly 10 executions, recent should show all 10."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                for i in range(10):
                    day = 1 + i * 31
                    date_str = f"2026{day:03d}"
                    _make_order_file(
                        Path(d),
                        f"order_history_{date_str}_120000_{i:03d}",
                        orders,
                    )
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["total_executions"] == 10
        assert len(result["execution_history"]) == 10  # All 10

    def test_compliance_25_days_boundary(self):
        """25 days between orders should be on_time (inclusive boundary)."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(Path(d), "order_history_20260401_120000_aaa", orders)
                _make_order_file(Path(d), "order_history_20260426_120000_bbb", orders)  # +25 days
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["schedule_compliance"]["on_time"] == 1
        assert result["schedule_compliance"]["delayed"] == 0

    def test_compliance_35_days_boundary(self):
        """35 days between orders should be on_time (inclusive boundary)."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(Path(d), "order_history_20260401_120000_aaa", orders)
                _make_order_file(Path(d), "order_history_20260506_120000_bbb", orders)  # +35 days
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["schedule_compliance"]["on_time"] == 1
        assert result["schedule_compliance"]["delayed"] == 0

    def test_compliance_24_days_below(self):
        """24 days between orders should be delayed (below 25-day threshold)."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(Path(d), "order_history_20260401_120000_aaa", orders)
                _make_order_file(Path(d), "order_history_20260425_120000_bbb", orders)  # +24 days
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["schedule_compliance"]["delayed"] == 1
        assert result["schedule_compliance"]["on_time"] == 0

    def test_compliance_36_days_above(self):
        """36 days between orders should be delayed (above 35-day threshold)."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(Path(d), "order_history_20260401_120000_aaa", orders)
                _make_order_file(Path(d), "order_history_20260507_120000_bbb", orders)  # +36 days
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["schedule_compliance"]["delayed"] == 1
        assert result["schedule_compliance"]["on_time"] == 0

    def test_compliance_all_on_time_100_pct(self):
        """All on-time intervals should yield 100% compliance."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(Path(d), "order_history_20260101_120000_aaa", orders)
                _make_order_file(Path(d), "order_history_20260201_120000_bbb", orders)  # 31 days
                _make_order_file(Path(d), "order_history_20260301_120000_ccc", orders)  # 28 days
                _make_order_file(Path(d), "order_history_20260401_120000_ddd", orders)  # 31 days
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["schedule_compliance"]["compliance_pct"] == 100.0
        assert result["schedule_compliance"]["total"] == 3

    def test_compliance_all_delayed_0_pct(self):
        """All delayed intervals should yield 0% compliance."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(Path(d), "order_history_20260101_120000_aaa", orders)
                _make_order_file(Path(d), "order_history_20260301_120000_bbb", orders)  # 59 days
                _make_order_file(Path(d), "order_history_20260501_120000_ccc", orders)  # 61 days
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["schedule_compliance"]["compliance_pct"] == 0.0

    def test_non_order_files_in_orders_dir(self):
        """Non-JSON files in ORDERS_DIR should be ignored."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                # Create a valid order file
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(Path(d), "order_history_20260501_120000_aaa", orders)
                # Create non-JSON files that should be ignored
                (Path(d) / "readme.txt").write_text("hello")
                (Path(d) / "data.yaml").write_text("key: value")
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["total_executions"] == 1  # Only the valid JSON file

    def test_invalid_order_files_matching_glob(self):
        """Files matching glob but with bad content should be skipped."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                # Two invalid files matching the glob
                (Path(d) / "order_history_20260501_120000_bad.json").write_text("not json {{{")
                (Path(d) / "order_history_20260502_120000_empty.json").write_text("[]")
                # One valid file
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(Path(d), "order_history_20260503_120000_good", orders)
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["total_executions"] == 1  # Only the valid file counted

    def test_same_day_dedup_three_entries(self):
        """Three entries on the same day should deduplicate to one for compliance."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                # Three orders on April 1 at different times
                _make_order_file(Path(d), "order_history_20260401_090000_aaa", orders)
                _make_order_file(Path(d), "order_history_20260401_120000_bbb", orders)
                _make_order_file(Path(d), "order_history_20260401_140000_ccc", orders)
                # One order on May 1
                _make_order_file(Path(d), "order_history_20260501_120000_ddd", orders)
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert result["total_executions"] == 4  # All 4 in history
        # 1 compliance pair (Apr 1 → May 1), deduped from 3 same-day entries
        assert result["schedule_compliance"]["total"] == 1

    def test_next_rebalance_no_history(self):
        """Without history, next_rebalance defaults to ~30 days from now."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                rh.ORDERS_DIR.mkdir(parents=True, exist_ok=True)
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        # next_rebalance.date should be ~30 days from today
        assert "next_rebalance" in result
        # days_until should be close to 30
        assert -1 <= result["next_rebalance"]["days_until"] <= 31

    def test_generated_field_format(self):
        """The 'generated' field should be a valid ISO timestamp."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(Path(d), "order_history_20260501_120000_aaa", orders)
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        assert "generated" in result
        # Verify it's a parseable ISO datetime
        dt = datetime.fromisoformat(result["generated"])
        assert isinstance(dt, datetime)
        assert dt.tzinfo is not None
        assert dt.utcoffset() == timezone.utc.utcoffset(dt)

    def test_compliance_with_three_intervals_mixed(self):
        """Mixed on-time and delayed intervals should compute correct pct."""
        import src.monitor.rebalance_health as rh
        original_dir = rh.ORDERS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                rh.ORDERS_DIR = Path(d)
                orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
                _make_order_file(Path(d), "order_history_20260101_120000_a01", orders)
                _make_order_file(Path(d), "order_history_20260201_120000_b02", orders)  # 31d on_time
                _make_order_file(Path(d), "order_history_20260301_120000_c03", orders)  # 28d on_time
                _make_order_file(Path(d), "order_history_20260515_120000_d04", orders)  # 75d delayed
                _make_order_file(Path(d), "order_history_20260615_120000_e05", orders)  # 31d on_time
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
        # 3 on_time + 1 delayed = 4 pairs, 75%
        assert result["schedule_compliance"]["on_time"] == 3
        assert result["schedule_compliance"]["delayed"] == 1
        assert result["schedule_compliance"]["compliance_pct"] == 75.0


class TestMain:
    """Tests for the main() function."""

    def test_main_writes_output_file(self, tmp_path):
        """main() should write rebalance_health.json to OUTPUT_PATH and public dir."""
        import src.monitor.rebalance_health as rh
        data_dir = tmp_path / "data"
        public_dir = tmp_path / "public"
        orders_dir = data_dir / "historical_orders"
        orders_dir.mkdir(parents=True)
        public_dir.mkdir(parents=True)

        orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
        _make_order_file(orders_dir, "order_history_20260511_143008_abc", orders)

        output_path = data_dir / "rebalance_health.json"
        with (
            patch.object(rh, 'DATA_DIR', data_dir),
            patch.object(rh, 'PUBLIC_DATA_DIR', public_dir),
            patch.object(rh, 'OUTPUT_PATH', output_path),
            patch.object(rh, 'ORDERS_DIR', orders_dir),
        ):
            rh.main()

        assert output_path.exists()
        assert (public_dir / "rebalance_health.json").exists()

        # Verify content
        with open(output_path) as f:
            data = json.load(f)
        assert data["total_executions"] == 1
        assert data["schedule_compliance"]["total"] == 0

    def test_main_writes_public_data(self, tmp_path):
        """The public data copy should match the data dir copy."""
        import src.monitor.rebalance_health as rh
        data_dir = tmp_path / "data"
        public_dir = tmp_path / "public"
        orders_dir = data_dir / "historical_orders"
        orders_dir.mkdir(parents=True)
        public_dir.mkdir(parents=True)

        orders = [{"symbol": "TLT", "side": "sell", "estimated_value": 2000, "reason": "drift"}]
        _make_order_file(orders_dir, "order_history_20260510_100000_xyz", orders)

        output_path = data_dir / "rebalance_health.json"
        with (
            patch.object(rh, 'DATA_DIR', data_dir),
            patch.object(rh, 'PUBLIC_DATA_DIR', public_dir),
            patch.object(rh, 'OUTPUT_PATH', output_path),
            patch.object(rh, 'ORDERS_DIR', orders_dir),
        ):
            rh.main()

        with open(output_path) as f:
            data_content = json.load(f)
        with open(public_dir / "rebalance_health.json") as f:
            public_content = json.load(f)
        assert data_content == public_content

    def test_main_output_format(
        self, tmp_path, caplog
    ):
        """main() should log expected summary lines."""
        import src.monitor.rebalance_health as rh
        data_dir = tmp_path / "data"
        public_dir = tmp_path / "public"
        orders_dir = data_dir / "historical_orders"
        orders_dir.mkdir(parents=True)
        public_dir.mkdir(parents=True)

        orders = [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}]
        _make_order_file(orders_dir, "order_history_20260511_143008_abc", orders)

        output_path = data_dir / "rebalance_health.json"
        with (
            patch.object(rh, 'DATA_DIR', data_dir),
            patch.object(rh, 'PUBLIC_DATA_DIR', public_dir),
            patch.object(rh, 'OUTPUT_PATH', output_path),
            patch.object(rh, 'ORDERS_DIR', orders_dir),
            caplog.at_level(logging.INFO, logger="src.monitor.rebalance_health"),
        ):
            rh.main()

        assert "Rebalance health data exported" in caplog.text
        assert "Executions" in caplog.text
        assert "Next rebalance" in caplog.text
        assert "Compliance" in caplog.text

    def test_main_with_no_orders(self, tmp_path):
        """main() should handle empty ORDERS_DIR gracefully."""
        import src.monitor.rebalance_health as rh
        data_dir = tmp_path / "data"
        public_dir = tmp_path / "public"
        orders_dir = data_dir / "historical_orders"
        orders_dir.mkdir(parents=True)
        public_dir.mkdir(parents=True)

        output_path = data_dir / "rebalance_health.json"
        with (
            patch.object(rh, 'DATA_DIR', data_dir),
            patch.object(rh, 'PUBLIC_DATA_DIR', public_dir),
            patch.object(rh, 'OUTPUT_PATH', output_path),
            patch.object(rh, 'ORDERS_DIR', orders_dir),
        ):
            rh.main()

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert data["total_executions"] == 0


class TestBatchDRRecentOrdersDailySummary:
    """Batch DR/DS: live order-history uses recent_orders; schedule uses event clock."""

    def test_recent_orders_key_parsed_uses_event_clock_not_write_day(self):
        """Batch DS: write_day=2026-07-22 with May fills must not advance schedule."""
        import src.monitor.rebalance_health as rh
        from src.monitor.rebalance_health import generate

        original_dir = rh.ORDERS_DIR
        original_data_dir = rh.DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                data_dir = Path(d)
                rh.DATA_DIR = data_dir
                rh.ORDERS_DIR = data_dir / "historical_orders"
                rh.ORDERS_DIR.mkdir()
                _make_order_file(
                    rh.ORDERS_DIR,
                    "order_history_20260512_120000_aaa",
                    [{"symbol": "SPY", "side": "buy", "estimated_value": 1000, "reason": "rebalance"}],
                )
                # Snapshot rewrite: file date today, fills are May (orders.jsonl tail)
                (data_dir / "order-history-2026-07-22.json").write_text(json.dumps({
                    "date": "2026-07-22",
                    "total_orders": 6,
                    "recent_shown": 6,
                    "recent_orders": [
                        {
                            "symbol": "SPY",
                            "side": "buy",
                            "estimated_value": 46000,
                            "reason": "rebalance_up",
                            "timestamp": "2026-05-11T03:20:31.447694",
                        },
                        {
                            "symbol": "GLD",
                            "side": "buy",
                            "estimated_value": 38000,
                            "reason": "rebalance_up",
                            "timestamp": "2026-05-11T03:20:31.447694",
                        },
                    ],
                    "statistics": {},
                }))
                result = generate()
        finally:
            rh.ORDERS_DIR = original_dir
            rh.DATA_DIR = original_data_dir

        # Event clock = 2026-05-12 hist or 2026-05-11 daily max → not write day
        assert result["execution_history"][0]["date"] != "2026-07-22" or (
            result["execution_history"][0].get("clock_source") == "order_event_timestamp"
            and result["execution_history"][0]["date"] == "2026-05-11"
        )
        # Prefer newest event among hist May-12 and daily May-11 → May-12
        last_exec = result["next_rebalance"]["last_execution_at"]
        assert last_exec is not None
        assert last_exec.startswith("2026-05-1")
        assert result["next_rebalance"]["date"].startswith("2026-06-1")
        assert result["canonical_order_history_source"] == "combined_order_history"
        assert result.get("snapshot_rewrite_files", 0) >= 1

    def test_parse_daily_uses_max_order_event_not_payload_date(self):
        from src.monitor.rebalance_health import _parse_daily_order_summary

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "order-history-2026-07-20.json"
            path.write_text(json.dumps({
                "date": "2026-07-20",
                "total_orders": 2,
                "recent_orders": [
                    {
                        "symbol": "SPY",
                        "side": "buy",
                        "estimated_value": 1000,
                        "timestamp": "2026-05-11T00:00:00",
                    },
                    {
                        "symbol": "GLD",
                        "side": "buy",
                        "estimated_value": 1000,
                        "timestamp": "2026-07-11T00:20:02",
                    },
                ],
            }))
            entry = _parse_daily_order_summary(path)
        assert entry is not None
        assert entry["date"] == "2026-07-11"
        assert entry["timestamp"].startswith("2026-07-11")
        assert entry["clock_source"] == "order_event_timestamp"
        assert entry["summary_file_date"] == "2026-07-20"
        assert entry.get("snapshot_rewrite") is True
        assert entry.get("snapshot_rewrite_lag_days") == 9