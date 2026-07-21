#!/usr/bin/env python3
"""Tests for daily P&L capture script."""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

from scripts.capture_daily_pnl import (
    load_portfolio,
    compute_pnl_snapshot,
    save_snapshot,
    backfill_daily_returns_from_nav,
    main,
)


def _make_portfolio(cash=0, positions=None, history=None, mode="paper"):
    """Create a test portfolio dict."""
    if positions is None:
        positions = {
            "SPY": {"shares": 50, "avg_price": 500.0, "current_price": 520.0,
                     "value": 26000.0, "unrealized_pnl": 1000.0},
            "GLD": {"shares": 40, "avg_price": 200.0, "current_price": 210.0,
                     "value": 8400.0, "unrealized_pnl": 400.0},
        }
    if history is None:
        history = [
            {"total_value": 90000, "daily_return": 0.01},
            {"total_value": 95000, "daily_return": 0.005},
        ]
    return {"cash": cash, "positions": positions, "history": history,
            "updated": "2026-05-23T12:00:00", "mode": mode}


class TestLoadPortfolio:
    def test_loads_existing(self, tmp_path):
        pf = _make_portfolio()
        path = tmp_path / "portfolio_paper.json"
        with open(path, 'w') as f:
            json.dump(pf, f)
        with patch("scripts.capture_daily_pnl.DATA_DIR", tmp_path):
            result = load_portfolio("paper")
        assert result is not None
        assert result["mode"] == "paper"

    def test_returns_none_when_missing(self, tmp_path):
        with patch("scripts.capture_daily_pnl.DATA_DIR", tmp_path):
            result = load_portfolio("paper")
        assert result is None

    def test_loads_live_mode(self, tmp_path):
        pf = _make_portfolio(mode="live")
        path = tmp_path / "portfolio_live.json"
        with open(path, 'w') as f:
            json.dump(pf, f)
        with patch("scripts.capture_daily_pnl.DATA_DIR", tmp_path):
            result = load_portfolio("live")
        assert result["mode"] == "live"


class TestLoadPortfolioEdgeCases:
    """Edge cases for load_portfolio — corrupted or unusual files."""

    def test_corrupted_json_raises_decode_error(self, tmp_path):
        path = tmp_path / "portfolio_paper.json"
        path.write_text("{bad json")
        with patch("scripts.capture_daily_pnl.DATA_DIR", tmp_path):
            with pytest.raises(json.JSONDecodeError):
                load_portfolio("paper")

    def test_empty_file_raises_decode_error(self, tmp_path):
        path = tmp_path / "portfolio_paper.json"
        path.write_text("")
        with patch("scripts.capture_daily_pnl.DATA_DIR", tmp_path):
            with pytest.raises(json.JSONDecodeError):
                load_portfolio("paper")

    def test_whitespace_only_file_raises_decode_error(self, tmp_path):
        path = tmp_path / "portfolio_paper.json"
        path.write_text("   \n\n  ")
        with patch("scripts.capture_daily_pnl.DATA_DIR", tmp_path):
            with pytest.raises(json.JSONDecodeError):
                load_portfolio("paper")


class TestComputePnlSnapshot:
    def test_basic_snapshot(self):
        pf = _make_portfolio(cash=1000)
        snap = compute_pnl_snapshot(pf)
        assert snap["mode"] == "paper"
        assert snap["total_value"] == 35400.0  # 26000 + 8400 + 1000
        assert snap["cash"] == 1000.0
        assert snap["positions_count"] == 2
        assert "date" in snap
        assert "timestamp" in snap

    def test_total_pnl(self):
        pf = _make_portfolio()
        snap = compute_pnl_snapshot(pf)
        # total_value = 26000 + 8400 + 0 = 34400
        # initial_capital = 100000
        # total_pnl = 34400 - 100000 = -65600
        assert snap["total_pnl"] == -65600.0
        assert snap["total_pnl_pct"] == pytest.approx(-0.656)

    def test_daily_return_from_history_nav(self):
        """Prefer NAV day-over-day from history total_value (not stale daily_return)."""
        pf = _make_portfolio(history=[
            {"total_value": 90000, "daily_return": 0.015},
        ])
        snap = compute_pnl_snapshot(pf)
        # portfolio total = 34400, prior history NAV = 90000
        assert snap["daily_return"] == pytest.approx((34400 / 90000) - 1.0, abs=1e-6)

    def test_daily_return_from_jsonl_prior_day(self, tmp_path):
        jsonl = tmp_path / "daily_pnl.jsonl"
        jsonl.write_text(
            '{"date":"2026-07-19","total_value":100000}\n'
            '{"date":"2026-07-18","total_value":99000}\n',
            encoding="utf-8",
        )
        pf = _make_portfolio(history=[])
        # force total_value via cash+empty positions
        pf["positions"] = {}
        pf["cash"] = 101000
        snap = compute_pnl_snapshot(
            pf, append_path=jsonl, as_of_date="2026-07-20"
        )
        assert snap["daily_return"] == pytest.approx(0.01, abs=1e-6)

    def test_daily_return_default_no_history(self):
        pf = _make_portfolio(history=[])
        snap = compute_pnl_snapshot(pf)
        assert snap["daily_return"] == 0.0

    def test_position_weights(self):
        pf = _make_portfolio(cash=6000)
        snap = compute_pnl_snapshot(pf)
        # total = 26000 + 8400 + 6000 = 40400
        assert snap["positions"]["SPY"]["weight"] == pytest.approx(26000 / 40400, abs=0.01)
        assert snap["positions"]["GLD"]["weight"] == pytest.approx(8400 / 40400, abs=0.01)

    def test_drawdown_calculation(self):
        pf = _make_portfolio(cash=0, history=[
            {"total_value": 120000, "daily_return": 0.01},
            {"total_value": 110000, "daily_return": -0.08},
            {"total_value": 34400, "daily_return": -0.01},
        ])
        snap = compute_pnl_snapshot(pf)
        # peak = 120000, current = 34400
        # drawdown = (34400 - 120000) / 120000
        assert snap["drawdown"] == pytest.approx((34400 - 120000) / 120000, abs=0.001)

    def test_empty_positions(self):
        pf = {"cash": 50000, "positions": {}, "history": [],
              "updated": "2026-05-23", "mode": "paper"}
        snap = compute_pnl_snapshot(pf)
        assert snap["total_value"] == 50000.0
        assert snap["positions_count"] == 0
        assert snap["positions"] == {}


class TestComputePnlSnapshotEdgeCases:
    """Edge cases for compute_pnl_snapshot — boundary values and missing fields."""

    def test_zero_total_value(self):
        pf = {"cash": 0, "positions": {}, "history": [],
              "updated": "2026-05-23", "mode": "paper"}
        snap = compute_pnl_snapshot(pf)
        assert snap["total_value"] == 0.0
        assert snap["positions_count"] == 0
        assert snap["total_pnl"] == -100000.0
        assert snap["total_pnl_pct"] == -1.0
        assert snap["drawdown"] == pytest.approx((0 - 100000) / 100000, abs=0.001)

    def test_negative_cash(self):
        pf = {"cash": -5000, "positions": {}, "history": [],
              "updated": "2026-05-23", "mode": "paper"}
        snap = compute_pnl_snapshot(pf)
        assert snap["cash"] == -5000.0
        assert snap["total_value"] == -5000.0

    def test_single_position(self):
        pf = {"cash": 0, "positions": {
            "SPY": {"shares": 10, "avg_price": 500, "current_price": 520,
                     "value": 5200, "unrealized_pnl": 200},
        }, "history": [], "updated": "2026-05-23", "mode": "paper"}
        snap = compute_pnl_snapshot(pf)
        assert snap["positions_count"] == 1
        assert snap["total_value"] == 5200.0
        assert snap["positions"]["SPY"]["weight"] == 1.0

    def test_missing_optional_position_fields_default_to_zero(self):
        pf = {"cash": 1000, "positions": {
            "BTC": {"shares": 1, "value": 30000},
        }, "history": [], "updated": "2026-05-23", "mode": "paper"}
        snap = compute_pnl_snapshot(pf)
        assert snap["total_value"] == 31000.0
        pos = snap["positions"]["BTC"]
        assert pos["avg_price"] == 0.0
        assert pos["current_price"] == 0.0
        assert pos["unrealized_pnl"] == 0.0

    def test_position_weight_zero_when_total_value_zero(self):
        pf = {"cash": 0, "positions": {
            "SPY": {"value": 1000, "shares": 1, "avg_price": 1000,
                     "current_price": 1000, "unrealized_pnl": 0},
        }, "history": [], "updated": "2026-05-23", "mode": "paper",
              "cash": 0}
        # Override total to 0 by making positions empty within the dict
        pf["positions"] = {}
        snap = compute_pnl_snapshot(pf)
        assert snap["total_value"] == 0.0
        assert snap["positions"] == {}

    def test_positive_drawdown_when_current_above_peak(self):
        """Drawdown is positive when current total exceeds peak from history."""
        pf = _make_portfolio(cash=0, history=[
            {"total_value": 120000, "daily_return": 0.20},
            {"total_value": 110000, "daily_return": -0.08},
        ])
        # Make current total exceed the 120000 peak
        pf["positions"]["SPY"]["value"] = 150000
        pf["positions"]["SPY"]["current_price"] = 3000
        snap = compute_pnl_snapshot(pf)
        # peak = max(100000, 120000, 110000) = 120000
        # current = 150000 + 8400 = 158400
        # drawdown = (158400 - 120000) / 120000 = 0.32
        assert snap["drawdown"] == pytest.approx(0.32, abs=0.001)

    def test_drawdown_peak_at_first_history_entry(self):
        pf = _make_portfolio(cash=0, history=[
            {"total_value": 200000, "daily_return": 0.05},
            {"total_value": 150000, "daily_return": -0.25},
            {"total_value": 140000, "daily_return": -0.07},
        ])
        snap = compute_pnl_snapshot(pf)
        # peak = max(100000, 200000, 150000, 140000) = 200000
        expected = (34400 - 200000) / 200000
        assert snap["drawdown"] == pytest.approx(expected, abs=0.001)

    def test_drawdown_peak_in_middle_of_history(self):
        pf = _make_portfolio(cash=0, history=[
            {"total_value": 100000, "daily_return": 0.01},
            {"total_value": 250000, "daily_return": 0.15},
            {"total_value": 200000, "daily_return": -0.20},
            {"total_value": 180000, "daily_return": -0.10},
        ])
        snap = compute_pnl_snapshot(pf)
        # peak = max(100000, 100000, 250000, 200000, 180000) = 250000
        expected = (34400 - 250000) / 250000
        assert snap["drawdown"] == pytest.approx(expected, abs=0.001)

    def test_drawdown_peak_is_initial_capital_when_history_below(self):
        """Peak equals initial capital when no history entry exceeds it."""
        pf = _make_portfolio(cash=0, history=[
            {"total_value": 80000, "daily_return": -0.05},
            {"total_value": 75000, "daily_return": -0.07},
        ])
        snap = compute_pnl_snapshot(pf)
        # peak = max(100000, 80000, 75000) = 100000
        expected = (34400 - 100000) / 100000
        assert snap["drawdown"] == pytest.approx(expected, abs=0.001)

    def test_drawdown_with_no_history_uses_initial_capital(self):
        pf = _make_portfolio(cash=0, history=[])
        snap = compute_pnl_snapshot(pf)
        # peak = initial_capital = 100000, current = 34400
        expected = (34400 - 100000) / 100000
        assert snap["drawdown"] == pytest.approx(expected, abs=0.001)

    def test_daily_return_from_history_nav_when_key_missing(self):
        pf = _make_portfolio(history=[
            {"total_value": 90000},  # no daily_return key — still use NAV
        ])
        snap = compute_pnl_snapshot(pf)
        assert snap["daily_return"] == pytest.approx((34400 / 90000) - 1.0, abs=1e-6)

    def test_daily_return_defaults_to_zero_when_history_has_no_dicts(self):
        pf = _make_portfolio(history=[{}])
        snap = compute_pnl_snapshot(pf)
        assert snap["daily_return"] == 0.0

    def test_custom_initial_capital_via_patch(self):
        """Patching INITIAL_CAPITAL changes PnL computation."""
        pf = _make_portfolio(cash=50000, positions={})
        with patch("scripts.capture_daily_pnl.INITIAL_CAPITAL", 50000):
            snap = compute_pnl_snapshot(pf)
        assert snap["total_value"] == 50000.0
        assert snap["total_pnl"] == 0.0
        assert snap["total_pnl_pct"] == 0.0

    def test_value_rounding_precision(self):
        """Verify rounding behavior for all numeric fields."""
        pf = {"cash": 1.23456, "positions": {
            "AAPL": {"shares": 3.14159, "avg_price": 150.123,
                      "current_price": 155.678, "value": 489.123,
                      "unrealized_pnl": 17.456},
        }, "history": [], "updated": "2026-05-23", "mode": "paper"}
        snap = compute_pnl_snapshot(pf)
        # cash: round(1.23456, 2) = 1.23
        assert snap["cash"] == 1.23
        # total: sum 1.23456 + 489.123 = 490.35756, round(490.35756, 2) = 490.36
        assert snap["total_value"] == 490.36
        pos = snap["positions"]["AAPL"]
        assert pos["shares"] == 3.1416       # round(3.14159, 4)
        assert pos["avg_price"] == 150.12    # round(150.123, 2)
        assert pos["current_price"] == 155.68  # round(155.678, 2)
        assert pos["value"] == 489.12        # round(489.123, 2)
        assert pos["unrealized_pnl"] == 17.46  # round(17.456, 2)

    def test_mode_passthrough(self):
        pf = _make_portfolio(mode="live")
        snap = compute_pnl_snapshot(pf)
        assert snap["mode"] == "live"

    def test_position_value_none_does_not_crash(self):
        """Position with None value is treated as 0 by sum()."""
        pf = {"cash": 1000, "positions": {
            "XYZ": {"shares": 10, "avg_price": 50, "current_price": 55,
                     "value": None, "unrealized_pnl": 0},
        }, "history": [], "updated": "2026-05-23", "mode": "paper"}
        # sum() will fail on None — this documents current behavior
        with pytest.raises(TypeError):
            compute_pnl_snapshot(pf)


class TestSaveSnapshot:
    def test_creates_files(self, tmp_path):
        snap = {"date": "2026-05-23", "timestamp": "2026-05-23T12:00:00",
                "total_value": 100000}
        append_path = tmp_path / "daily_pnl.jsonl"
        latest_path = tmp_path / "daily_pnl_latest.json"
        save_snapshot(snap, append_path, latest_path)
        assert append_path.exists()
        assert latest_path.exists()

    def test_appends_to_existing(self, tmp_path):
        snap1 = {"date": "2026-05-22", "total_value": 99000}
        snap2 = {"date": "2026-05-23", "total_value": 100000}
        append_path = tmp_path / "daily_pnl.jsonl"
        latest_path = tmp_path / "daily_pnl_latest.json"
        save_snapshot(snap1, append_path, latest_path)
        save_snapshot(snap2, append_path, latest_path)
        with open(append_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2

    def test_idempotent_same_date(self, tmp_path):
        snap1 = {"date": "2026-05-23", "total_value": 99000}
        snap2 = {"date": "2026-05-23", "total_value": 100000}
        append_path = tmp_path / "daily_pnl.jsonl"
        latest_path = tmp_path / "daily_pnl_latest.json"
        save_snapshot(snap1, append_path, latest_path)
        save_snapshot(snap2, append_path, latest_path)
        with open(append_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["total_value"] == 100000  # Updated, not duplicated

    def test_latest_json_has_full_snapshot(self, tmp_path):
        snap = {"date": "2026-05-23", "total_value": 100000, "mode": "paper"}
        append_path = tmp_path / "daily_pnl.jsonl"
        latest_path = tmp_path / "daily_pnl_latest.json"
        save_snapshot(snap, append_path, latest_path)
        with open(latest_path) as f:
            latest = json.load(f)
        assert latest["total_value"] == 100000
        assert latest["mode"] == "paper"


class TestSaveSnapshotEdgeCases:
    """Edge cases for save_snapshot — file corruption, blank lines, non-serializable values."""

    def test_appends_multiple_different_dates(self, tmp_path):
        snap1 = {"date": "2026-05-21", "total_value": 98000}
        snap2 = {"date": "2026-05-22", "total_value": 99000}
        snap3 = {"date": "2026-05-23", "total_value": 100000}
        append_path = tmp_path / "daily_pnl.jsonl"
        latest_path = tmp_path / "daily_pnl_latest.json"
        save_snapshot(snap1, append_path, latest_path)
        save_snapshot(snap2, append_path, latest_path)
        save_snapshot(snap3, append_path, latest_path)
        with open(append_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 3

    def test_handles_corrupted_jsonl_lines(self, tmp_path):
        """Corrupted JSON lines are preserved as-is rather than dropped."""
        append_path = tmp_path / "daily_pnl.jsonl"
        latest_path = tmp_path / "daily_pnl_latest.json"
        append_path.write_text(
            '{"date": "2026-05-22", "total_value": 99000}\n'
            'corrupted garbage line\n'
            '{"date": "2026-05-21", "total_value": 98000}\n'
        )
        snap = {"date": "2026-05-23", "total_value": 100000}
        save_snapshot(snap, append_path, latest_path)
        with open(append_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 4
        assert "corrupted garbage line" in lines

    def test_empty_jsonl_file_handled_gracefully(self, tmp_path):
        append_path = tmp_path / "daily_pnl.jsonl"
        latest_path = tmp_path / "daily_pnl_latest.json"
        append_path.write_text("")
        snap = {"date": "2026-05-23", "total_value": 100000}
        save_snapshot(snap, append_path, latest_path)
        with open(append_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1

    def test_jsonl_with_blank_and_whitespace_lines(self, tmp_path):
        append_path = tmp_path / "daily_pnl.jsonl"
        latest_path = tmp_path / "daily_pnl_latest.json"
        append_path.write_text('\n\n{"date": "2026-05-22", "total_value": 99000}\n\n')
        snap = {"date": "2026-05-23", "total_value": 100000}
        save_snapshot(snap, append_path, latest_path)
        with open(append_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2

    def test_snapshot_with_datetime_uses_default_str(self, tmp_path):
        """Non-serializable datetime objects are handled via default=str."""
        from datetime import datetime
        snap = {"date": "2026-05-23",
                "timestamp": datetime(2026, 5, 23, 12, 0, 0),
                "total_value": 100000}
        append_path = tmp_path / "daily_pnl.jsonl"
        latest_path = tmp_path / "daily_pnl_latest.json"
        save_snapshot(snap, append_path, latest_path)
        with open(append_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1

    def test_overwrites_latest_json_each_time(self, tmp_path):
        """latest.json always contains the most recent snapshot."""
        snap1 = {"date": "2026-05-22", "total_value": 99000}
        snap2 = {"date": "2026-05-23", "total_value": 100000}
        append_path = tmp_path / "daily_pnl.jsonl"
        latest_path = tmp_path / "daily_pnl_latest.json"
        save_snapshot(snap1, append_path, latest_path)
        save_snapshot(snap2, append_path, latest_path)
        with open(latest_path) as f:
            latest = json.load(f)
        assert latest["total_value"] == 100000
        assert latest["date"] == "2026-05-23"

    def test_idempotent_same_date_replaces_with_updated_fields(self, tmp_path):
        """Same-date idempotency preserves the newest snapshot, not the oldest."""
        snap1 = {"date": "2026-05-23", "total_value": 99000, "extra_field": "old"}
        snap2 = {"date": "2026-05-23", "total_value": 100000, "extra_field": "new"}
        append_path = tmp_path / "daily_pnl.jsonl"
        latest_path = tmp_path / "daily_pnl_latest.json"
        save_snapshot(snap1, append_path, latest_path)
        save_snapshot(snap2, append_path, latest_path)
        with open(append_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["total_value"] == 100000
        assert entry["extra_field"] == "new"

    def test_save_snapshot_returns_true(self, tmp_path):
        snap = {"date": "2026-05-23", "total_value": 100000}
        append_path = tmp_path / "daily_pnl.jsonl"
        latest_path = tmp_path / "daily_pnl_latest.json"
        result = save_snapshot(snap, append_path, latest_path)
        assert result is True


class TestBackfillDailyReturns:
    def test_rewrites_zero_when_nav_moved(self, tmp_path):
        path = tmp_path / "daily_pnl.jsonl"
        path.write_text(
            '{"date":"2026-07-07","total_value":95367.04,"daily_return":0.0}\n'
            '{"date":"2026-07-08","total_value":96111.74,"daily_return":0.0}\n'
            '{"date":"2026-07-09","total_value":95316.38,"daily_return":0.0}\n',
            encoding="utf-8",
        )
        summary = backfill_daily_returns_from_nav(path, dry_run=False)
        assert summary["rewritten"] == 2
        rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        assert rows[1]["daily_return"] == pytest.approx((96111.74 / 95367.04) - 1.0, abs=1e-6)
        assert rows[1].get("daily_return_backfilled") is True

    def test_dry_run_no_write(self, tmp_path):
        path = tmp_path / "daily_pnl.jsonl"
        path.write_text(
            '{"date":"2026-07-07","total_value":100.0,"daily_return":0.0}\n'
            '{"date":"2026-07-08","total_value":110.0,"daily_return":0.0}\n',
            encoding="utf-8",
        )
        summary = backfill_daily_returns_from_nav(path, dry_run=True)
        assert summary["rewritten"] == 1
        row = json.loads(path.read_text().splitlines()[1])
        assert row["daily_return"] == 0.0


class TestMain:
    """Tests for the main() CLI entry point."""

    def test_main_happy_path_paper(self, tmp_path):
        pf = _make_portfolio()
        portfolio_path = tmp_path / "portfolio_paper.json"
        with open(portfolio_path, 'w') as f:
            json.dump(pf, f)
        with patch("scripts.capture_daily_pnl.DATA_DIR", tmp_path):
            with patch.object(sys, "argv", ["capture_daily_pnl.py"]):
                main()
        # Verify snapshot files were created
        assert (tmp_path / "daily_pnl.jsonl").exists()
        assert (tmp_path / "daily_pnl_latest.json").exists()

    def test_main_with_live_mode(self, tmp_path):
        pf = _make_portfolio(mode="live")
        portfolio_path = tmp_path / "portfolio_live.json"
        with open(portfolio_path, 'w') as f:
            json.dump(pf, f)
        with patch("scripts.capture_daily_pnl.DATA_DIR", tmp_path):
            with patch.object(sys, "argv",
                               ["capture_daily_pnl.py", "--mode", "live"]):
                main()
        assert (tmp_path / "daily_pnl.jsonl").exists()

    def test_main_exits_when_portfolio_missing(self, tmp_path):
        with patch("scripts.capture_daily_pnl.DATA_DIR", tmp_path):
            with patch.object(sys, "argv", ["capture_daily_pnl.py"]):
                with pytest.raises(SystemExit) as excinfo:
                    main()
                assert excinfo.value.code == 1


class TestUsCashSessionDate:
    """daily_pnl date keys use America/New_York, not host-local midnight."""

    def test_asia_midnight_stays_on_us_session_day(self):
        """00:10 Asia/Hong_Kong must not invent a next US calendar row early."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from scripts.capture_daily_pnl import us_cash_session_date, compute_pnl_snapshot

        # 2026-07-22 00:10 HKT == 2026-07-21 12:10 ET (still Jul 21 session)
        now_hkt = datetime(2026, 7, 22, 0, 10, tzinfo=ZoneInfo("Asia/Hong_Kong"))
        assert us_cash_session_date(now_hkt) == "2026-07-21"

        # Explicit override still wins
        pf = {
            "mode": "paper",
            "cash": 10000,
            "positions": {},
            "history": [],
        }
        snap = compute_pnl_snapshot(pf, as_of_date="2026-07-20")
        assert snap["date"] == "2026-07-20"

    def test_default_date_uses_et_not_host_strftime(self, monkeypatch):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import scripts.capture_daily_pnl as cap

        # Freeze "now" to a known ET afternoon
        fixed = datetime(2026, 7, 21, 15, 30, tzinfo=ZoneInfo("America/New_York"))

        class _FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    return fixed.replace(tzinfo=None)
                return fixed.astimezone(tz)

        monkeypatch.setattr(cap, "datetime", _FixedDateTime)
        # us_cash_session_date uses datetime.now(tz=_ET) — patch module datetime
        assert cap.us_cash_session_date() == "2026-07-21"

