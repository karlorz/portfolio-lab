#!/usr/bin/env python3
"""Tests for daily P&L capture script."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from scripts.capture_daily_pnl import (
    load_portfolio,
    compute_pnl_snapshot,
    save_snapshot,
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

    def test_daily_return_from_history(self):
        pf = _make_portfolio(history=[
            {"total_value": 90000, "daily_return": 0.015},
        ])
        snap = compute_pnl_snapshot(pf)
        assert snap["daily_return"] == 0.015

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
