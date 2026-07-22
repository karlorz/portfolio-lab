"""Batch EB: paper return five-surface SSOT agreement on compact health.

Live friction: portfolio_paper.history daily_return (0.008) disagreed with
daily_pnl SSOT (0.0) while NAV matched — multi-surface drift without compact
health SLI. Research: single SSOT for NAV/session return; previous-day base.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.dashboard.generator import project_paper_return_ssot_onto_health
from src.monitor.health_check import refresh_signals_health_kill_fields
from src.monitor.paper_return_ssot import apply_capture_ssot_side_effects


def test_project_disagreement_soft_warns() -> None:
    health: dict = {"status": "ok"}
    cmp = {
        "agree": False,
        "ssot": {
            "date": "2026-07-22",
            "daily_return": 0.0,
            "total_value": 94906.22,
            "return_source": "daily_pnl_latest",
        },
        "disagreements": [
            {
                "surface": "portfolio_paper_history",
                "why_not": "return_mismatch",
                "ssot_return": 0.0,
                "observed_return": 0.008347,
            }
        ],
    }
    out = project_paper_return_ssot_onto_health(health, cmp)
    assert out["paper_return_ssot_agree"] is False
    assert out["paper_return_ssot_status"] == "disagree"
    assert out["paper_return_ssot_date"] == "2026-07-22"
    assert out["paper_return_ssot_disagreement_count"] == 1
    assert "portfolio_paper_history" in (out.get("paper_return_ssot_surfaces") or "")
    assert out["status"] == "warning"


def test_project_agree_ok() -> None:
    health: dict = {"status": "ok"}
    cmp = {
        "agree": True,
        "ssot": {
            "date": "2026-07-22",
            "daily_return": 0.0,
            "total_value": 94906.22,
            "return_source": "daily_pnl_latest",
        },
        "disagreements": [],
    }
    out = project_paper_return_ssot_onto_health(health, cmp)
    assert out["paper_return_ssot_agree"] is True
    assert out["paper_return_ssot_status"] == "ok"
    assert out["status"] == "ok"


def test_partial_refresh_reprojects_ssot(
    tmp_path: Path, monkeypatch
) -> None:
    public = tmp_path / "public"
    private = tmp_path / "data"
    public.mkdir()
    private.mkdir()
    # Seed SSOT + mismatched history
    (private / "daily_pnl_latest.json").write_text(
        json.dumps(
            {
                "date": "2026-07-22",
                "daily_return": 0.0,
                "total_value": 100000.0,
                "return_source": "capture_daily_pnl",
            }
        ),
        encoding="utf-8",
    )
    (private / "daily_pnl.jsonl").write_text(
        json.dumps(
            {
                "date": "2026-07-22",
                "daily_return": 0.0,
                "total_value": 100000.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (private / "portfolio_paper.json").write_text(
        json.dumps(
            {
                "mode": "paper",
                "cash": 10000.0,
                "positions": {
                    "SPY": {
                        "shares": 100,
                        "avg_price": 400.0,
                        "current_price": 900.0,
                        "value": 90000.0,
                    }
                },
                "history": [
                    {
                        "session_date": "2026-07-22",
                        "date": "2026-07-22",
                        "total_value": 100000.0,
                        "daily_return": 0.008,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    signals = {
        "generated_at": "2026-07-22T05:00:00+00:00",
        "generator_git_sha": "deadbeef",
        "generator_git_sha_status": "full_generate",
        "health": {"status": "ok"},
    }
    (public / "signals.json").write_text(json.dumps(signals), encoding="utf-8")
    (private / "signals.json").write_text(json.dumps(signals), encoding="utf-8")
    (private / "kill_switch.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8"
    )
    (private / "incidents.json").write_text(
        json.dumps({"open_count": 0, "status": "ok"}), encoding="utf-8"
    )

    monkeypatch.setattr(
        "src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False
    )
    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", private, raising=False)

    report = {
        "status": "ok",
        "system_status": "ok",
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0, "status": "ok"},
    }
    refresh_signals_health_kill_fields(
        report, public_dir=public, data_dir=private
    )
    out = json.loads((public / "signals.json").read_text(encoding="utf-8"))
    h = out.get("health") or {}
    assert h.get("paper_return_ssot_agree") is False
    assert h.get("paper_return_ssot_status") == "disagree"
    assert h.get("status") == "warning"

    # Align then re-project should clear
    apply_capture_ssot_side_effects(
        private,
        {
            "date": "2026-07-22",
            "daily_return": 0.0,
            "total_value": 100000.0,
        },
        mode="paper",
    )
    refresh_signals_health_kill_fields(
        report, public_dir=public, data_dir=private
    )
    out2 = json.loads((public / "signals.json").read_text(encoding="utf-8"))
    h2 = out2.get("health") or {}
    assert h2.get("paper_return_ssot_agree") is True
    assert h2.get("paper_return_ssot_status") == "ok"
