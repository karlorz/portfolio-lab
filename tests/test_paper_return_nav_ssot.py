"""Tests for five-surface paper return / NAV SSOT (c358).

Drives real helpers in ``src.monitor.paper_return_ssot`` and the real
``scripts.capture_daily_pnl`` entry path under a temp DATA_DIR.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.monitor.paper_return_ssot import (
    apply_capture_ssot_side_effects,
    compare_five_surfaces,
    load_session_ssot,
    material_return,
    write_paper_trading_performance_from_ssot,
)
from scripts.capture_daily_pnl import (
    main,
    save_snapshot,
)


def _write_portfolio(
    data_dir: Path,
    *,
    total_value: float = 100000.0,
    history=None,
    wrong_history_return: float | None = None,
) -> Path:
    cash = 10000.0
    equity = total_value - cash
    portfolio = {
        "mode": "paper",
        "cash": cash,
        "positions": {
            "SPY": {
                "shares": 100,
                "avg_price": 400.0,
                "current_price": equity / 100.0,
                "value": equity,
                "unrealized_pnl": 0.0,
            }
        },
        "history": history
        if history is not None
        else [
            {
                "session_date": "2026-07-20",
                "date": "2026-07-20",
                "total_value": 99000.0,
                "daily_return": 0.001,
            }
        ],
        "updated": "2026-07-21T20:00:00",
    }
    if wrong_history_return is not None:
        portfolio["history"].append(
            {
                "session_date": "2026-07-21",
                "date": "2026-07-21",
                "total_value": total_value,
                "daily_return": wrong_history_return,
            }
        )
    path = data_dir / "portfolio_paper.json"
    path.write_text(json.dumps(portfolio, indent=2) + "\n", encoding="utf-8")
    return path


def _seed_prior_session(data_dir: Path, date: str = "2026-07-20", value: float = 99000.0) -> None:
    row = {
        "date": date,
        "total_value": value,
        "daily_return": 0.001,
        "cash": 10000.0,
        "mode": "paper",
        "return_source": "capture_daily_pnl",
    }
    (data_dir / "daily_pnl.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )


class TestMaterialReturn:
    def test_zero_is_material(self):
        assert material_return(0.0) is True

    def test_micro_noise_dropped(self):
        assert material_return(1e-8) is False

    def test_real_session_kept(self):
        assert material_return(0.001749) is True


class TestWriteSsotAndSnapshot:
    def test_snapshot_current_value_matches_ssot_nav(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _seed_prior_session(data_dir)
        # Write today's SSOT row
        today = {
            "date": "2026-07-21",
            "total_value": 100100.0,
            "daily_return": 0.011111,
            "mode": "paper",
        }
        with open(data_dir / "daily_pnl.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(today) + "\n")
        (data_dir / "daily_pnl_latest.json").write_text(
            json.dumps(today, indent=2), encoding="utf-8"
        )
        # Stale snapshot claiming wrong NAV
        stale = {
            "date": "2026-07-21",
            "performance": {"current_value": 94834.87, "days_tracked": 71},
        }
        (data_dir / "paper-trading-performance-2026-07-21.json").write_text(
            json.dumps(stale), encoding="utf-8"
        )

        path = write_paper_trading_performance_from_ssot(
            data_dir, session_date="2026-07-21", current_value=100100.0
        )
        assert path is not None
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["performance"]["current_value"] == pytest.approx(100100.0)
        assert payload["return_source"] == "daily_pnl.jsonl_session"
        assert payload["performance"]["current_value_source"] == "daily_pnl_ssot"

    def test_side_effects_align_history_and_five_surfaces(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _seed_prior_session(data_dir, value=99000.0)
        # Wrong-sign history for same session (c358 dual SSOT)
        _write_portfolio(
            data_dir,
            total_value=100100.0,
            wrong_history_return=-0.000487,
        )
        snapshot = {
            "date": "2026-07-21",
            "total_value": 100100.0,
            "daily_return": 0.011111,
            "cash": 10000.0,
            "mode": "paper",
        }
        save_snapshot(
            snapshot,
            data_dir / "daily_pnl.jsonl",
            data_dir / "daily_pnl_latest.json",
        )
        result = apply_capture_ssot_side_effects(data_dir, snapshot, mode="paper")
        assert result["history"]["updated"] is True
        assert result["paper_trading_performance"] is not None

        paper = json.loads((data_dir / "portfolio_paper.json").read_text(encoding="utf-8"))
        last = paper["history"][-1]
        assert last["daily_return"] == pytest.approx(0.011111, abs=1e-6)
        assert last["return_source"] == "daily_pnl_ssot"

        comparison = compare_five_surfaces(data_dir, session_date="2026-07-21")
        assert comparison["agree"] is True, comparison["disagreements"]
        assert comparison["ssot"]["total_value"] == pytest.approx(100100.0)


class TestCaptureEntryPath:
    def test_main_writes_ssot_and_fresh_snapshot(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _seed_prior_session(data_dir, value=99000.0)
        _write_portfolio(
            data_dir,
            total_value=100100.0,
            wrong_history_return=-0.0005,
        )
        # Stale snapshot
        (data_dir / "paper-trading-performance-2026-07-21.json").write_text(
            json.dumps(
                {
                    "date": "2026-07-21",
                    "performance": {"current_value": 94834.87},
                }
            ),
            encoding="utf-8",
        )

        with patch("scripts.capture_daily_pnl.DATA_DIR", data_dir), patch(
            "src.monitor.paper_return_ssot.DATA_DIR", data_dir, create=True
        ), patch.object(sys, "argv", ["capture_daily_pnl.py", "--mode", "paper"]):
            # Also patch us_cash_session_date for stable session key
            with patch(
                "scripts.capture_daily_pnl.us_cash_session_date",
                return_value="2026-07-21",
            ):
                main()

        latest = json.loads((data_dir / "daily_pnl_latest.json").read_text(encoding="utf-8"))
        assert latest["date"] == "2026-07-21"
        assert latest["total_value"] == pytest.approx(100100.0)
        assert latest.get("write_ssot") == "daily_pnl.jsonl"
        assert "daily_return" in latest

        snap_path = data_dir / "paper-trading-performance-2026-07-21.json"
        assert snap_path.exists()
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        assert snap["performance"]["current_value"] == pytest.approx(100100.0)

        # Idempotent second capture: same session key, same sign
        first_return = latest["daily_return"]
        with patch("scripts.capture_daily_pnl.DATA_DIR", data_dir), patch.object(
            sys, "argv", ["capture_daily_pnl.py", "--mode", "paper"]
        ), patch(
            "scripts.capture_daily_pnl.us_cash_session_date",
            return_value="2026-07-21",
        ):
            main()
        latest2 = json.loads((data_dir / "daily_pnl_latest.json").read_text(encoding="utf-8"))
        assert latest2["date"] == "2026-07-21"
        assert latest2["daily_return"] == pytest.approx(first_return, abs=1e-6)
        # No opposite-sign dual SSOT in history
        paper = json.loads((data_dir / "portfolio_paper.json").read_text(encoding="utf-8"))
        last = paper["history"][-1]
        assert last["daily_return"] == pytest.approx(first_return, abs=1e-6)
        assert (last["daily_return"] >= 0) == (first_return >= 0)

        comparison = compare_five_surfaces(data_dir, session_date="2026-07-21")
        assert comparison["agree"] is True, comparison["disagreements"]

    def test_load_session_ssot_prefers_latest(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _seed_prior_session(data_dir)
        latest = {
            "date": "2026-07-21",
            "total_value": 100050.0,
            "daily_return": 0.010606,
        }
        (data_dir / "daily_pnl_latest.json").write_text(
            json.dumps(latest), encoding="utf-8"
        )
        with open(data_dir / "daily_pnl.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(latest) + "\n")
        ssot = load_session_ssot(data_dir, session_date="2026-07-21")
        assert ssot is not None
        assert ssot["return_source"] == "daily_pnl_latest"
        assert ssot["total_value"] == pytest.approx(100050.0)
