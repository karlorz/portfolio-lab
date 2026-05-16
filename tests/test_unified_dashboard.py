"""
Tests for v6.08 Unified System Dashboard.

Covers:
- Section reader functions for all state files
- Main dashboard generation
- CLI flags (--save, --json, --status-text, --check)
- Edge cases (missing files, corrupt JSON)
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.monitor.unified_dashboard import (
    _read_json,
    _get_health_section,
    _get_portfolio_section,
    _get_risk_section,
    _get_tca_section,
    _get_overlays_section,
    _get_regime_section,
    _get_attribution_section,
    _get_cron_section,
    _get_risk_history_section,
    generate_unified_dashboard,
    generate_status_text,
    print_summary,
    main,
)

# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def sample_health_report():
    return {
        "timestamp": "2026-05-16T18:00:00",
        "status": "healthy",
        "checks": {
            "data_freshness": {"name": "data_freshness", "status": "ok", "ok": True},
            "cron_execution": {"name": "cron_execution", "status": "ok", "ok": True},
            "portfolio": {"name": "portfolio", "status": "ok", "ok": True},
            "graduation": {"name": "graduation", "status": "candidate_ready", "ok": True},
            "kill_switches": {"name": "kill_switches", "status": "ok", "ok": True},
            "circuit_breaker": {"name": "circuit_breaker", "status": "green", "ok": True},
            "cvar_metrics": {"name": "cvar_metrics", "status": "moderate", "ok": True},
            "portfolio_entropy": {"name": "portfolio_entropy", "status": "good", "ok": True},
            "wiki_sync": {"name": "wiki_sync", "status": "ok", "ok": True},
        },
        "alerts": ["PROMOTION CANDIDATE: Ready for live trading approval"],
        "summary": {"total_checks": 9, "passed": 9, "failed": 0},
    }


@pytest.fixture
def sample_portfolio():
    return {
        "cash": 5000.0,
        "positions": {
            "SPY": {"symbol": "SPY", "shares": 60, "value": 45000, "unrealized_pnl": 200, "avg_price": 740.0, "current_price": 750.0},
            "GLD": {"symbol": "GLD", "shares": 80, "value": 35000, "unrealized_pnl": -100, "avg_price": 430.0, "current_price": 437.5},
            "TLT": {"symbol": "TLT", "shares": 170, "value": 15000, "unrealized_pnl": 50, "avg_price": 85.0, "current_price": 88.2},
        },
        "history": [{"timestamp": "2026-05-15", "total_value": 100000}],
        "updated": "2026-05-16T18:00:00",
        "mode": "paper",
    }


@pytest.fixture
def sample_risk_metrics():
    return {
        "timestamp": "2026-05-16T18:00:00",
        "var_95_daily": -1.41,
        "cvar_95_daily": -2.02,
        "cvar_ratio": 1.43,
        "tail_severity": "moderate",
        "max_drawdown": -11.28,
        "current_drawdown": -1.69,
        "volatility_annual": 12.89,
        "garch_active": False,
        "garch_filtered": False,
    }


@pytest.fixture
def sample_tca_scorecard():
    return {
        "generated": "2026-05-16T08:00:00",
        "total_orders": 3,
        "total_notional": 100000.0,
        "avg_slippage_bps": -10.0,
        "avg_quality_score": 43.3,
        "weighted_slippage_bps": -10.0,
        "by_symbol": {
            "SPY": {"count": 1, "notional": 46046.0, "slippage_bps": -10.0, "quality": 45.0},
            "GLD": {"count": 1, "notional": 38038.0, "slippage_bps": -10.0, "quality": 45.0},
            "TLT": {"count": 1, "notional": 15916.0, "slippage_bps": -10.0, "quality": 40.0},
        },
    }


@pytest.fixture
def sample_tca_feedback():
    return {
        "version": "6.05",
        "overall_quality": 43.3,
        "urgency_global_offset": -0.1,
        "min_trade_global_multiplier": 2.0,
        "cost_calibration_global": 1.6,
        "symbols": {
            "SPY": {"avg_quality": 45.0, "quality_bucket": "poor", "urgency_offset": -0.2, "cost_calibration": 1.6},
            "GLD": {"avg_quality": 45.0, "quality_bucket": "poor", "urgency_offset": -0.2, "cost_calibration": 1.6},
            "TLT": {"avg_quality": 40.0, "quality_bucket": "poor", "urgency_offset": -0.2, "cost_calibration": 1.6},
        },
        "quality_timeline": [],
    }


@pytest.fixture
def sample_regime_classifier():
    return {
        "current_regime": "normal",
        "previous_regime": "normal",
        "regime_start_date": None,
        "last_updated": "2026-05-16T14:00:00",
        "history": [{"timestamp": "2026-05-14", "regime": "normal", "confidence": 0.7}],
        "last_reading": {"confidence": 0.3},
    }


@pytest.fixture
def sample_regime_optimizer():
    return {
        "current_regime": "unknown",
        "regime_confidence": 0.3,
        "last_updated": "2026-05-16T17:00:00",
        "method": "cost_aware",
        "weights": {"SPY": 0.36, "GLD": 0.38, "TLT": 0.16},
        "expected_return": 0.0612,
        "expected_volatility": 0.098,
        "expected_sharpe": 0.216,
        "solver_status": "optimal",
        "constraints_satisfied": True,
    }


@pytest.fixture
def sample_risk_budget():
    return {
        "timestamp": "2026-05-16T17:00:00",
        "regime": "normal",
        "weights": {"SPY": 0.49, "GLD": 0.38, "TLT": 0.13},
        "all_budgets_met": False,
        "portfolio_vol_before": 0.111,
        "portfolio_vol_after": 0.0996,
    }


@pytest.fixture
def sample_cron_status():
    return {
        "jobs": [
            {"name": "portfolio-lab-data", "status": "ok", "last_run": "2026-05-16T18:00:00", "duration_seconds": 12.5, "backend": "hermes"},
            {"name": "portfolio-lab-health", "status": "ok", "last_run": "2026-05-16T18:00:00", "duration_seconds": 3.2, "backend": "hermes"},
            {"name": "portfolio-lab-eval", "status": "error", "last_run": "2026-05-16T17:30:00", "duration_seconds": 45.0, "backend": "hermes"},
        ]
    }


@pytest.fixture
def sample_risk_history():
    return [
        {"timestamp": "2026-05-13T14:00:00", "var_95": -1.37, "cvar_95": -2.0, "cvar_ratio": 1.46, "current_drawdown": -0.59, "volatility_annual": 12.83},
        {"timestamp": "2026-05-16T18:00:00", "var_95": -1.41, "cvar_95": -2.02, "cvar_ratio": 1.43, "current_drawdown": -1.69, "volatility_annual": 12.89},
    ]


# ─────────────────────────────────────────────
#  Tests: _read_json
# ─────────────────────────────────────────────


class TestReadJson:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('{"key": "value"}')
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            result = _read_json("test.json")
            assert result == {"key": "value"}

    def test_returns_none_for_missing_file(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            result = _read_json("nonexistent.json")
            assert result is None

    def test_returns_none_for_corrupt_json(self, tmp_path):
        f = tmp_path / "corrupt.json"
        f.write_text("{invalid json}")
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            result = _read_json("corrupt.json")
            assert result is None


# ─────────────────────────────────────────────
#  Tests: Section Readers
# ─────────────────────────────────────────────


class TestHealthSection:
    def test_returns_healthy(self, tmp_path, sample_health_report):
        f = tmp_path / ".health_report.json"
        f.write_text(json.dumps(sample_health_report))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_health_section()
            assert section["available"] is True
            assert section["status"] == "healthy"
            assert section["checks_passed"] == 9
            assert section["checks_total"] == 9
            assert len(section["alerts"]) == 1

    def test_returns_not_available_when_missing(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_health_section()
            assert section["available"] is False

    def test_components_listed(self, tmp_path, sample_health_report):
        f = tmp_path / ".health_report.json"
        f.write_text(json.dumps(sample_health_report))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_health_section()
            assert "data_freshness" in section["components"]
            assert section["components"]["data_freshness"]["ok"] is True


class TestPortfolioSection:
    def test_returns_positions(self, tmp_path, sample_portfolio):
        f = tmp_path / "portfolio_paper.json"
        f.write_text(json.dumps(sample_portfolio))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_portfolio_section()
            assert section["available"] is True
            assert section["total_value"] == 100000.0
            assert section["cash"] == 5000.0
            assert section["cash_pct"] == 5.0
            assert len(section["positions"]) == 3
            # Positions sorted by value descending
            assert section["positions"][0]["symbol"] == "SPY"

    def test_returns_not_available_when_missing(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_portfolio_section()
            assert section["available"] is False

    def test_handles_empty_positions(self, tmp_path):
        f = tmp_path / "portfolio_paper.json"
        f.write_text(json.dumps({"cash": 100000, "positions": {}, "mode": "paper"}))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_portfolio_section()
            assert section["available"] is True
            assert section["total_value"] == 100000.0
            assert len(section["positions"]) == 0


class TestRiskSection:
    def test_returns_metrics(self, tmp_path, sample_risk_metrics):
        f = tmp_path / "risk_metrics.json"
        f.write_text(json.dumps(sample_risk_metrics))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_risk_section()
            assert section["available"] is True
            assert section["var_95_daily"] == -1.41
            assert section["cvar_ratio"] == 1.43
            assert section["tail_severity"] == "moderate"

    def test_returns_not_available_when_missing(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_risk_section()
            assert section["available"] is False


class TestTCASection:
    def test_returns_scorecard(self, tmp_path, sample_tca_scorecard):
        f = tmp_path / "tca_scorecard.json"
        f.write_text(json.dumps(sample_tca_scorecard))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_tca_section()
            assert section["available"] is True
            assert section["scorecard"]["total_orders"] == 3
            assert section["scorecard"]["avg_slippage_bps"] == -10.0

    def test_returns_feedback(self, tmp_path, sample_tca_feedback):
        f = tmp_path / "tca_feedback_state.json"
        f.write_text(json.dumps(sample_tca_feedback))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_tca_section()
            assert section["available"] is True
            assert section["feedback"]["overall_quality"] == 43.3
            assert section["feedback"]["quality_label"] == "poor"

    def test_returns_both(self, tmp_path, sample_tca_scorecard, sample_tca_feedback):
        (tmp_path / "tca_scorecard.json").write_text(json.dumps(sample_tca_scorecard))
        (tmp_path / "tca_feedback_state.json").write_text(json.dumps(sample_tca_feedback))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_tca_section()
            assert section["scorecard"]["total_orders"] == 3
            assert section["feedback"]["overall_quality"] == 43.3

    def test_returns_not_available(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_tca_section()
            assert section["available"] is False


class TestOverlaysSection:
    def test_all_overlays_present(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_overlays_section()
            # Even without state files, all overlay keys should exist
            for key in ["collar", "crypto", "bond_duration", "vix_term_structure", "mean_reversion"]:
                assert key in section
            assert "_meta" in section

    def test_counts_active_overlays(self, tmp_path):
        # Create bond_duration state as active
        bond = tmp_path / "bond_duration_state.json"
        bond.write_text(json.dumps({"status": "active", "current_position": "long", "tlt_weight": 1.0}))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_overlays_section()
            assert section["_meta"]["active_count"] == 1
            assert section["_meta"]["total_count"] == 5
            assert section["bond_duration"]["active"] is True
            assert section["collar"]["active"] is False


class TestRegimeSection:
    def test_returns_all_components(self, tmp_path, sample_regime_classifier, sample_regime_optimizer, sample_risk_budget):
        (tmp_path / "regime_classifier_state.json").write_text(json.dumps(sample_regime_classifier))
        (tmp_path / "regime_optimizer_state.json").write_text(json.dumps(sample_regime_optimizer))
        (tmp_path / "risk_budget_state.json").write_text(json.dumps(sample_risk_budget))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_regime_section()
            assert section["available"] is True
            assert section["classifier"]["current_regime"] == "normal"
            assert section["optimizer"]["method"] == "cost_aware"
            assert section["risk_budget"]["regime"] == "normal"

    def test_not_available(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_regime_section()
            assert section["available"] is False


class TestCronSection:
    def test_counts_statuses(self, tmp_path, sample_cron_status):
        f = tmp_path / "cron_status.json"
        f.write_text(json.dumps(sample_cron_status))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_cron_section()
            assert section["available"] is True
            assert section["total"] == 3
            assert section["ok"] == 2
            assert section["errors"] == 1

    def test_not_available(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_cron_section()
            assert section["available"] is False


class TestRiskHistorySection:
    def test_returns_trend(self, tmp_path, sample_risk_history):
        f = tmp_path / "risk_metrics_history.json"
        f.write_text(json.dumps(sample_risk_history))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_risk_history_section()
            assert section["available"] is True
            assert section["data_points"] == 2
            assert section["trend"]["var_95"]["first"] == -1.37
            assert section["trend"]["var_95"]["last"] == -1.41

    def test_not_available_with_single_point(self, tmp_path):
        f = tmp_path / "risk_metrics_history.json"
        f.write_text(json.dumps([{"timestamp": "2026-05-16", "var_95": -1.41}]))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_risk_history_section()
            assert section["available"] is False

    def test_not_available_when_missing(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_risk_history_section()
            assert section["available"] is False


# ─────────────────────────────────────────────
#  Tests: Attribution Section
# ─────────────────────────────────────────────


class TestAttributionSection:
    def test_returns_sources_when_dir_exists(self, tmp_path):
        attr_dir = tmp_path / "attribution"
        attr_dir.mkdir()
        attr_file = attr_dir / "attribution_2026-05-16.json"
        attr_file.write_text(json.dumps({
            "timestamp": "2026-05-16T12:00:00",
            "analysis_days": 90,
            "sources": {
                "tsfm_momentum": {
                    "display_name": "TSFM Momentum",
                    "category": "trend",
                    "hit_rate": 0.55,
                    "win_rate": 0.52,
                    "total_return_bps": 120.5,
                    "sharpe_contribution": 0.15,
                    "avg_weight": 0.20,
                    "active_days": 45,
                }
            },
        }))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_attribution_section()
            assert section["available"] is True
            assert section["analysis_days"] == 90
            assert len(section["sources"]) == 1
            assert section["sources"][0]["name"] == "TSFM Momentum"

    def test_not_available_when_dir_missing(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_attribution_section()
            assert section["available"] is False


# ─────────────────────────────────────────────
#  Tests: Main Generator
# ─────────────────────────────────────────────


class TestGenerateUnifiedDashboard:
    def test_returns_all_sections(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            dashboard = generate_unified_dashboard()
            assert "dashboard_version" in dashboard
            assert dashboard["dashboard_version"] == "v6.08"
            assert "generated_at" in dashboard
            for key in ["health", "portfolio", "risk", "risk_history", "tca", "overlays", "regime", "attribution", "cron"]:
                assert key in dashboard, f"Missing section: {key}"

    def test_health_available_with_real_data(self):
        """Integration test: read the actual health report."""
        dashboard = generate_unified_dashboard()
        health = dashboard["health"]
        assert health["available"] is True
        assert health["status"] in ("healthy", "unhealthy")


class TestGenerateStatusText:
    def test_returns_string(self):
        text = generate_status_text()
        assert isinstance(text, str)
        assert len(text) > 10
        assert "Unified:" in text or "✅" in text or "⚠️" in text


class TestPrintSummary:
    def test_does_not_raise(self, capsys):
        dashboard = generate_unified_dashboard()
        print_summary(dashboard)  # Should not raise
        captured = capsys.readouterr()
        assert "PORTFOLIO-LAB UNIFIED DASHBOARD" in captured.out


class TestCLI:
    def test_save_flag(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            test_args = ["unified_dashboard.py", "--save"]
            with patch.object(sys, "argv", test_args):
                main()
            saved = tmp_path / "unified_dashboard.json"
            assert saved.exists()
            data = json.loads(saved.read_text())
            assert data["dashboard_version"] == "v6.08"

    def test_status_text_flag(self):
        test_args = ["unified_dashboard.py", "--status-text"]
        with patch.object(sys, "argv", test_args):
            main()  # Should print status text

    def test_check_flag_healthy(self, tmp_path):
        # Create a healthy health report
        hr = tmp_path / ".health_report.json"
        hr.write_text(json.dumps({
            "status": "healthy",
            "checks": {},
            "alerts": [],
            "summary": {"total_checks": 1, "passed": 1, "failed": 0},
        }))
        # Create cron status with no errors
        cs = tmp_path / "cron_status.json"
        cs.write_text(json.dumps({"jobs": [{"name": "test", "status": "ok"}]}))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            test_args = ["unified_dashboard.py", "--check"]
            with patch.object(sys, "argv", test_args):
                try:
                    main()
                    assert False, "Should have sys.exit(0)"
                except SystemExit as e:
                    assert e.code == 0

    def test_check_flag_unhealthy(self, tmp_path):
        # Create an unhealthy health report
        hr = tmp_path / ".health_report.json"
        hr.write_text(json.dumps({
            "status": "unhealthy",
            "checks": {},
            "alerts": ["CRITICAL: Data stale"],
            "summary": {"total_checks": 1, "passed": 0, "failed": 1},
        }))
        # Create cron status with errors
        cs = tmp_path / "cron_status.json"
        cs.write_text(json.dumps({"jobs": [{"name": "test", "status": "error"}]}))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            test_args = ["unified_dashboard.py", "--check"]
            with patch.object(sys, "argv", test_args):
                try:
                    main()
                    assert False, "Should have sys.exit(1)"
                except SystemExit as e:
                    assert e.code == 1
