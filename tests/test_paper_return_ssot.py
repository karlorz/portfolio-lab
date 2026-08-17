"""Unit tests for src.monitor.paper_return_ssot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.monitor.paper_return_ssot import (
    DEFAULT_NAV_EPS,
    DEFAULT_RETURN_EPS,
    FIVE_SURFACES,
    _as_float,
    _read_json,
    align_portfolio_history_to_ssot,
    apply_capture_ssot_side_effects,
    compare_five_surfaces,
    load_daily_pnl_sessions,
    load_session_ssot,
    material_return,
    read_surface_session,
    values_agree,
    write_paper_trading_performance_from_ssot,
)


class TestMaterialReturn:
    """Test material_return predicate."""

    def test_exact_zero_is_material(self):
        assert material_return(0.0) is True
        assert material_return(0) is True
        assert material_return("0.0") is True

    def test_above_floor_is_material(self):
        assert material_return(0.01) is True
        assert material_return(-0.01) is True
        assert material_return(1e-5) is True

    def test_micro_noise_below_floor_is_dropped(self):
        assert material_return(1e-7) is False
        assert material_return(-1e-8) is False

    def test_invalid_types_return_false(self):
        assert material_return(None) is False
        assert material_return("not_a_number") is False
        assert material_return({}) is False


class TestLoadDailyPnlSessions:
    """Test load_daily_pnl_sessions JSONL loader."""

    def test_missing_file_returns_empty(self, tmp_path: Path):
        assert load_daily_pnl_sessions(tmp_path) == []

    def test_corrupt_and_blank_lines_skipped(self, tmp_path: Path):
        pnl = tmp_path / "daily_pnl.jsonl"
        pnl.write_text(
            "\n"
            "not json\n"
            "42\n"
            '{"date": "2026-08-10", "daily_return": 0.01, "total_value": 10000.0}\n'
            '{"invalid": "no date"}\n'
            "\n",
            encoding="utf-8",
        )
        sessions = load_daily_pnl_sessions(tmp_path)
        assert len(sessions) == 1
        assert sessions[0]["date"] == "2026-08-10"

    def test_sessions_deduped_and_ordered_by_date(self, tmp_path: Path):
        pnl = tmp_path / "daily_pnl.jsonl"
        lines = [
            json.dumps({"date": "2026-08-12", "daily_return": 0.02, "total_value": 10200.0}),
            json.dumps({"date": "2026-08-10", "daily_return": 0.01, "total_value": 10000.0}),
            json.dumps({"date": "2026-08-10", "daily_return": 0.015, "total_value": 10050.0}),  # overwrite
            json.dumps({"date": "2026-08-11", "daily_return": -0.005, "total_value": 10000.0}),
        ]
        pnl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        sessions = load_daily_pnl_sessions(tmp_path)
        assert len(sessions) == 3
        assert [s["date"] for s in sessions] == ["2026-08-10", "2026-08-11", "2026-08-12"]
        assert sessions[0]["total_value"] == 10050.0


class TestLoadSessionSSOT:
    """Test load_session_ssot helper."""

    def test_missing_all_returns_none(self, tmp_path: Path):
        assert load_session_ssot(tmp_path) is None

    def test_latest_file_preferred_when_matching(self, tmp_path: Path):
        latest = tmp_path / "daily_pnl_latest.json"
        latest.write_text(
            json.dumps({"date": "2026-08-15", "daily_return": 0.01, "total_value": 10500.0}),
            encoding="utf-8",
        )
        ssot = load_session_ssot(tmp_path)
        assert ssot is not None
        assert ssot["date"] == "2026-08-15"
        assert ssot["return_source"] == "daily_pnl_latest"
        assert ssot["total_value"] == 10500.0

    def test_named_date_fallback_to_jsonl_when_latest_differs(self, tmp_path: Path):
        latest = tmp_path / "daily_pnl_latest.json"
        latest.write_text(
            json.dumps({"date": "2026-08-15", "daily_return": 0.01, "total_value": 10500.0}),
            encoding="utf-8",
        )
        pnl = tmp_path / "daily_pnl.jsonl"
        pnl.write_text(
            json.dumps({"date": "2026-08-10", "daily_return": 0.005, "total_value": 10000.0}) + "\n",
            encoding="utf-8",
        )
        ssot = load_session_ssot(tmp_path, session_date="2026-08-10")
        assert ssot is not None
        assert ssot["date"] == "2026-08-10"
        assert ssot["return_source"] == "daily_pnl.jsonl"
        assert ssot["daily_return"] == 0.005

    def test_named_date_missing_in_jsonl_returns_none(self, tmp_path: Path):
        pnl = tmp_path / "daily_pnl.jsonl"
        pnl.write_text(
            json.dumps({"date": "2026-08-10", "daily_return": 0.005, "total_value": 10000.0}) + "\n",
            encoding="utf-8",
        )
        assert load_session_ssot(tmp_path, session_date="2026-08-99") is None


class TestValuesAgree:
    """Test values_agree floating point tolerance helper."""

    def test_none_guards(self):
        assert values_agree(None, 1.0, eps=0.01) is False
        assert values_agree(1.0, None, eps=0.01) is False
        assert values_agree(None, None, eps=0.01) is False

    def test_within_and_exceeding_eps(self):
        assert values_agree(1.0, 1.000001, eps=DEFAULT_RETURN_EPS) is True
        assert values_agree(1.0, 1.001, eps=DEFAULT_RETURN_EPS) is False
        assert values_agree(100.0, 100.01, eps=DEFAULT_NAV_EPS) is True
        assert values_agree(100.0, 100.05, eps=DEFAULT_NAV_EPS) is False


class TestAlignPortfolioHistoryToSSOT:
    """Test align_portfolio_history_to_ssot function."""

    def test_missing_portfolio_returns_reason(self, tmp_path: Path):
        res = align_portfolio_history_to_ssot(tmp_path / "nonexistent.json", {"date": "2026-08-10"})
        assert res["updated"] is False
        assert res["reason"] == "missing_portfolio"

    def test_corrupt_portfolio_returns_reason(self, tmp_path: Path):
        f = tmp_path / "portfolio.json"
        f.write_text("corrupted", encoding="utf-8")
        res = align_portfolio_history_to_ssot(f, {"date": "2026-08-10"})
        assert res["updated"] is False
        assert "read_error" in res["reason"]

    def test_incomplete_ssot_returns_reason(self, tmp_path: Path):
        f = tmp_path / "portfolio.json"
        f.write_text(json.dumps({"history": []}), encoding="utf-8")
        res = align_portfolio_history_to_ssot(f, {"date": "2026-08-10"})
        assert res["updated"] is False
        assert res["reason"] == "incomplete_ssot"

    def test_append_when_empty_history(self, tmp_path: Path):
        f = tmp_path / "portfolio.json"
        f.write_text(json.dumps({"cash": 1000.0, "history": []}), encoding="utf-8")
        ssot = {"date": "2026-08-10", "daily_return": 0.01, "total_value": 10100.0}
        res = align_portfolio_history_to_ssot(f, ssot)
        assert res["updated"] is True
        data = json.loads(f.read_text(encoding="utf-8"))
        assert len(data["history"]) == 1
        assert data["history"][0]["session_date"] == "2026-08-10"
        assert data["history"][0]["total_value"] == 10100.0

    def test_update_existing_matching_row_when_disagreeing(self, tmp_path: Path):
        f = tmp_path / "portfolio.json"
        f.write_text(
            json.dumps({
                "history": [
                    {"date": "2026-08-10", "daily_return": 0.0, "total_value": 10000.0}
                ]
            }),
            encoding="utf-8",
        )
        ssot = {"date": "2026-08-10", "daily_return": 0.02, "total_value": 10200.0}
        res = align_portfolio_history_to_ssot(f, ssot)
        assert res["updated"] is True
        data = json.loads(f.read_text(encoding="utf-8"))
        assert data["history"][0]["daily_return"] == 0.02
        assert data["history"][0]["total_value"] == 10200.0

    def test_dry_run_does_not_modify_disk(self, tmp_path: Path):
        f = tmp_path / "portfolio.json"
        initial = json.dumps({"history": []})
        f.write_text(initial, encoding="utf-8")
        ssot = {"date": "2026-08-10", "daily_return": 0.01, "total_value": 10100.0}
        res = align_portfolio_history_to_ssot(f, ssot, dry_run=True)
        assert res["updated"] is True
        assert res["dry_run"] is True
        assert f.read_text(encoding="utf-8") == initial


class TestWritePaperTradingPerformance:
    """Test write_paper_trading_performance_from_ssot snapshot generation."""

    def test_empty_sessions_returns_none(self, tmp_path: Path):
        assert write_paper_trading_performance_from_ssot(tmp_path) is None

    def test_single_and_multi_session_calculation(self, tmp_path: Path):
        pnl = tmp_path / "daily_pnl.jsonl"
        lines = [
            json.dumps({"date": "2026-08-10", "daily_return": 0.01, "total_value": 10000.0}),
            json.dumps({"date": "2026-08-11", "daily_return": 0.02, "total_value": 10200.0}),
            json.dumps({"date": "2026-08-12", "daily_return": -0.01, "total_value": 10098.0}),
        ]
        pnl.write_text("\n".join(lines) + "\n", encoding="utf-8")

        out = write_paper_trading_performance_from_ssot(tmp_path, session_date="2026-08-12")
        assert out is not None
        assert out.is_file()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["date"] == "2026-08-12"
        assert payload["performance"]["days_tracked"] == 3
        assert payload["performance"]["current_value"] == 10098.0
        assert payload["daily_returns_distribution"]["positive_days"] == 2
        assert payload["daily_returns_distribution"]["negative_days"] == 1
        assert payload["schema_version"] == "paper-trading-performance/v3-ssot"


class TestReadSurfaceSessionAndCompare:
    """Test read_surface_session and compare_five_surfaces."""

    def test_unknown_surface_returns_why_not(self, tmp_path: Path):
        res = read_surface_session(tmp_path, "unknown_surface")
        assert res["available"] is False
        assert "unknown_surface" in res["why_not"]

    def test_missing_write_ssot_reports_disagreement(self, tmp_path: Path):
        cmp = compare_five_surfaces(tmp_path)
        assert cmp["agree"] is False
        assert cmp["ssot"] is None
        assert len(cmp["disagreements"]) == 1

    def test_coherent_surfaces_agree(self, tmp_path: Path):
        # 1. daily_pnl
        pnl = tmp_path / "daily_pnl.jsonl"
        pnl.write_text(
            json.dumps({"date": "2026-08-12", "daily_return": 0.01, "total_value": 10100.0}) + "\n",
            encoding="utf-8",
        )
        latest = tmp_path / "daily_pnl_latest.json"
        latest.write_text(
            json.dumps({"date": "2026-08-12", "daily_return": 0.01, "total_value": 10100.0}),
            encoding="utf-8",
        )
        # 2. portfolio_paper
        paper = tmp_path / "portfolio_paper.json"
        paper.write_text(
            json.dumps({
                "cash": 10100.0,
                "history": [
                    {"session_date": "2026-08-12", "daily_return": 0.01, "total_value": 10100.0}
                ]
            }),
            encoding="utf-8",
        )
        # 3. paper-trading-performance
        write_paper_trading_performance_from_ssot(tmp_path, session_date="2026-08-12")

        cmp = compare_five_surfaces(tmp_path, session_date="2026-08-12")
        assert cmp["agree"] is True
        assert len(cmp["disagreements"]) == 0


class TestApplyCaptureSSOTSideEffects:
    """Test apply_capture_ssot_side_effects orchestrator."""

    def test_side_effects_align_portfolio_and_write_performance(self, tmp_path: Path):
        paper = tmp_path / "portfolio_paper.json"
        paper.write_text(json.dumps({"cash": 10000.0, "history": []}), encoding="utf-8")
        pnl = tmp_path / "daily_pnl.jsonl"
        pnl.write_text(
            json.dumps({"date": "2026-08-12", "daily_return": 0.01, "total_value": 10100.0}) + "\n",
            encoding="utf-8",
        )
        snapshot = {"date": "2026-08-12", "daily_return": 0.01, "total_value": 10100.0}

        effects = apply_capture_ssot_side_effects(tmp_path, snapshot, mode="paper")
        assert effects["history"]["updated"] is True
        assert effects["paper_trading_performance"] is not None
        assert Path(effects["paper_trading_performance"]).is_file()
