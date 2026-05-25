"""
Tests for v6.08 Unified System Dashboard.

Covers:
- Section reader functions for all state files
- Main dashboard generation
- CLI flags (--save, --json, --status-text, --check)
- Edge cases (missing files, corrupt JSON)
"""

import json
import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest

# Ensure src is on path

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
    _get_adaptive_weights_section,
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
    """TCA section always unavailable — producer removed v977."""

    def test_returns_not_available(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_tca_section()
            assert section["available"] is False


class TestOverlaysSection:
    def test_vix_term_structure_present(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_overlays_section()
            assert "vix_term_structure" in section
            assert "_meta" in section

    def test_counts_active_overlays(self, tmp_path):
        # Create vixy hedge state as active
        vixy = tmp_path / "vixy_hedge_state.json"
        vixy.write_text(json.dumps({"current_allocation": 0.05, "last_signal_date": "2026-01-01", "regime": "contango"}))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_overlays_section()
            assert section["_meta"]["active_count"] == 1
            assert section["_meta"]["total_count"] == 1
            assert section["vix_term_structure"]["active"] is True


class TestRegimeSection:
    """Regime section always unavailable — producers removed v974-v977."""

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


class TestAdaptiveWeightsSection:
    """Tests for _get_adaptive_weights_section."""

    def test_not_available_when_no_file(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_adaptive_weights_section()
            assert section["available"] is False

    def test_not_available_when_no_adjusted_weights(self, tmp_path):
        f = tmp_path / "adaptive_weights_state.json"
        f.write_text(json.dumps({"baseline_weights": {"A": 0.5}}))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_adaptive_weights_section()
            assert section["available"] is False

    def test_available_with_valid_data(self, tmp_path):
        f = tmp_path / "adaptive_weights_state.json"
        f.write_text(json.dumps({
            "timestamp": "2026-05-24T10:00:00",
            "regime": "normal",
            "adjusted_weights": {"ALT_DATA": 0.35, "CROSS_RV": 0.10},
            "baseline_weights": {"ALT_DATA": 0.30, "CROSS_RV": 0.15},
            "multipliers": {"ALT_DATA": 1.17, "CROSS_RV": 0.67},
            "history": [{"ts": "t1"}, {"ts": "t2"}],
        }))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_adaptive_weights_section()
            assert section["available"] is True
            assert section["num_sources"] == 2
            assert section["regime"] == "normal"
            assert section["history_count"] == 2

    def test_top_boosted_and_reduced(self, tmp_path):
        f = tmp_path / "adaptive_weights_state.json"
        f.write_text(json.dumps({
            "adjusted_weights": {"A": 0.50, "B": 0.05, "C": 0.30},
            "baseline_weights": {"A": 0.30, "B": 0.25, "C": 0.30},
            "multipliers": {"A": 1.67, "B": 0.20, "C": 1.0},
        }))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_adaptive_weights_section()
            boosted = section["top_boosted"]
            reduced = section["top_reduced"]
            assert any(c["source"] == "A" for c in boosted)
            assert any(c["source"] == "B" for c in reduced)

    def test_changes_sorted_by_abs_change(self, tmp_path):
        f = tmp_path / "adaptive_weights_state.json"
        f.write_text(json.dumps({
            "adjusted_weights": {"A": 0.50, "B": 0.10, "C": 0.40},
            "baseline_weights": {"A": 0.30, "B": 0.20, "C": 0.50},
            "multipliers": {},
        }))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_adaptive_weights_section()
            changes = section["top_changes"]
            # A: +0.20, B: -0.10, C: -0.10 → A first
            assert changes[0]["source"] == "A"


class TestFormatHelpers:
    """Tests for _fmt, _fmt_pct, _status_badge."""

    def test_fmt_none(self):
        from src.monitor.unified_dashboard import _fmt
        assert _fmt(None) == "N/A"

    def test_fmt_float(self):
        from src.monitor.unified_dashboard import _fmt
        assert _fmt(3.14159) == "3.14"

    def test_fmt_float_with_suffix(self):
        from src.monitor.unified_dashboard import _fmt
        assert _fmt(3.14159, "%") == "3.14%"

    def test_fmt_int(self):
        from src.monitor.unified_dashboard import _fmt
        assert _fmt(42) == "42"

    def test_fmt_pct_none(self):
        from src.monitor.unified_dashboard import _fmt_pct
        assert _fmt_pct(None) == "N/A"

    def test_fmt_pct_float(self):
        from src.monitor.unified_dashboard import _fmt_pct
        assert _fmt_pct(12.345) == "12.35%"

    def test_status_badge_ok(self):
        from src.monitor.unified_dashboard import _status_badge
        assert _status_badge(True) == "✅"

    def test_status_badge_fail(self):
        from src.monitor.unified_dashboard import _status_badge
        assert _status_badge(False) == "❌"


class TestRiskHistoryEdgeCases:
    """Additional edge cases for risk history section."""

    def test_single_data_point_not_available(self, tmp_path):
        f = tmp_path / "risk_metrics_history.json"
        f.write_text(json.dumps([{"timestamp": "2026-01-01"}]))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_risk_history_section()
            assert section["available"] is False
            assert section["data_points"] == 1

    def test_empty_list_not_available(self, tmp_path):
        f = tmp_path / "risk_metrics_history.json"
        f.write_text(json.dumps([]))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_risk_history_section()
            assert section["available"] is False

    def test_corrupt_json_not_available(self, tmp_path):
        f = tmp_path / "risk_metrics_history.json"
        f.write_text("not valid json{{{")
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_risk_history_section()
            assert section["available"] is False
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
    def test_does_not_raise(self, caplog):
        dashboard = generate_unified_dashboard()
        with caplog.at_level(logging.INFO, logger="src.monitor.unified_dashboard"):
            print_summary(dashboard)  # Should not raise
        assert "PORTFOLIO-LAB UNIFIED DASHBOARD" in caplog.text


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


# ─────────────────────────────────────────────
#  Expanded Coverage: Health Section Edge Cases
# ─────────────────────────────────────────────


class TestHealthSectionEdgeCases:
    """Additional edge cases for _get_health_section: GARCH flat format, missing fields."""

    def test_garch_flat_healthy(self, tmp_path):
        report = {
            "timestamp": "2026-05-16T18:00:00",
            "tail_severity": "normal",
            "cvar_ratio": 1.5,
            "var_95": -1.41,
            "cvar_95": -2.02,
        }
        f = tmp_path / ".health_report.json"
        f.write_text(json.dumps(report))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_health_section()
            assert section["available"] is True
            assert section["status"] == "healthy"
            assert section["checks_passed"] == 1
            assert section["checks_total"] == 1
            assert len(section["alerts"]) == 0
            assert section["components"]["garch_cvar"]["ok"] is True

    def test_garch_flat_extreme_tail(self, tmp_path):
        report = {
            "tail_severity": "extreme",
            "cvar_ratio": 2.0,
        }
        f = tmp_path / ".health_report.json"
        f.write_text(json.dumps(report))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_health_section()
            assert section["available"] is True
            assert section["status"] == "unhealthy"
            assert len(section["alerts"]) == 1
            assert "extreme" in section["alerts"][0]

    def test_garch_flat_severe_tail(self, tmp_path):
        report = {
            "tail_severity": "severe",
            "cvar_ratio": 1.0,
        }
        f = tmp_path / ".health_report.json"
        f.write_text(json.dumps(report))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_health_section()
            assert section["available"] is True
            assert section["status"] == "unhealthy"

    def test_garch_flat_high_cvar_ratio(self, tmp_path):
        report = {
            "tail_severity": "moderate",
            "cvar_ratio": 4.0,
        }
        f = tmp_path / ".health_report.json"
        f.write_text(json.dumps(report))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_health_section()
            assert section["available"] is True
            assert section["status"] == "unhealthy"
            assert "CVaR ratio: 4.00" in section["alerts"][0]

    def test_garch_flat_missing_tail_defaults_normal(self, tmp_path):
        report = {"cvar_ratio": 0.5}
        f = tmp_path / ".health_report.json"
        f.write_text(json.dumps(report))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_health_section()
            assert section["available"] is True
            assert section["status"] == "healthy"

    def test_legacy_empty_checks(self, tmp_path):
        report = {"status": "healthy", "checks": {}, "alerts": []}
        f = tmp_path / ".health_report.json"
        f.write_text(json.dumps(report))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_health_section()
            assert section["available"] is True
            assert section["status"] == "healthy"
            assert section["checks_passed"] == 0
            assert section["components"] == {}

    def test_legacy_missing_summary(self, tmp_path):
        report = {"status": "healthy", "checks": {"a": {"status": "ok", "ok": True}}}
        f = tmp_path / ".health_report.json"
        f.write_text(json.dumps(report))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_health_section()
            assert section["available"] is True
            # Missing summary → defaults to 0
            assert section["checks_passed"] == 0

    def test_garch_var_cvar_passed_through(self, tmp_path):
        report = {"tail_severity": "normal", "cvar_ratio": 1.0, "var_95": -2.5, "cvar_95": -3.8}
        f = tmp_path / ".health_report.json"
        f.write_text(json.dumps(report))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_health_section()
            comp = section["components"]["garch_cvar"]
            assert comp["var_95"] == -2.5
            assert comp["cvar_95"] == -3.8


# ─────────────────────────────────────────────
#  Expanded Coverage: Portfolio Section Edge Cases
# ─────────────────────────────────────────────


class TestPortfolioSectionEdgeCases:
    """Edge cases for _get_portfolio_section: missing cash, zero total, partial fields."""

    def test_missing_cash_key(self, tmp_path):
        data = {"positions": {"SPY": {"symbol": "SPY", "shares": 10, "value": 10000}}}
        f = tmp_path / "portfolio_paper.json"
        f.write_text(json.dumps(data))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_portfolio_section()
            assert section["available"] is True
            assert section["cash"] == 0.0
            assert section["total_value"] == 10000.0

    def test_zero_total_value(self, tmp_path):
        data = {"cash": 0, "positions": {"SPY": {"symbol": "SPY", "shares": 0, "value": 0}}}
        f = tmp_path / "portfolio_paper.json"
        f.write_text(json.dumps(data))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_portfolio_section()
            assert section["available"] is True
            assert section["total_value"] == 0.0
            assert section["cash_pct"] == 0.0
            assert section["positions"][0]["weight"] == 0

    def test_partial_position_fields(self, tmp_path):
        """Position missing avg_price and current_price should not raise."""
        data = {
            "cash": 5000,
            "positions": {
                "SPY": {"symbol": "SPY", "shares": 10, "value": 10000, "unrealized_pnl": 0},
            },
        }
        f = tmp_path / "portfolio_paper.json"
        f.write_text(json.dumps(data))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_portfolio_section()
            pos = section["positions"][0]
            assert pos["avg_price"] is None
            assert pos["current_price"] is None

    def test_missing_history(self, tmp_path):
        data = {"cash": 5000, "positions": {}}
        f = tmp_path / "portfolio_paper.json"
        f.write_text(json.dumps(data))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_portfolio_section()
            assert section["history_count"] == 0

    def test_negative_cash(self, tmp_path):
        data = {"cash": -500, "positions": {"SPY": {"symbol": "SPY", "shares": 10, "value": 10000, "unrealized_pnl": 0}}}
        f = tmp_path / "portfolio_paper.json"
        f.write_text(json.dumps(data))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_portfolio_section()
            assert section["cash"] == -500.0
            assert section["total_value"] == 9500.0


# ─────────────────────────────────────────────
#  Expanded Coverage: Overlays Edge Cases
# ─────────────────────────────────────────────


class TestOverlaysSectionEdgeCases:
    """Edge cases for _get_overlays_section: inactive, dict allocation, missing VIXY fields."""

    def test_zero_allocation_is_inactive(self, tmp_path):
        vixy = tmp_path / "vixy_hedge_state.json"
        vixy.write_text(json.dumps({"current_allocation": 0, "last_signal_date": "2026-01-01", "regime": "contango"}))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_overlays_section()
            assert section["vix_term_structure"]["active"] is False
            assert section["_meta"]["active_count"] == 0

    def test_negative_allocation_is_inactive(self, tmp_path):
        vixy = tmp_path / "vixy_hedge_state.json"
        vixy.write_text(json.dumps({"current_allocation": -0.05, "regime": "backwardation"}))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_overlays_section()
            assert section["vix_term_structure"]["active"] is False

    def test_missing_regime_field(self, tmp_path):
        vixy = tmp_path / "vixy_hedge_state.json"
        vixy.write_text(json.dumps({"current_allocation": 0.1, "last_signal_date": "2026-01-01"}))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_overlays_section()
            assert section["vix_term_structure"]["regime"] is None

    def test_dict_allocation_in_overlay(self, tmp_path):
        """VIXY allocation can be a dict. Verify it passes through as-is."""
        alloc_dict = {"SPY": 0.4, "GLD": 0.6}
        vixy = tmp_path / "vixy_hedge_state.json"
        vixy.write_text(json.dumps({"current_allocation": alloc_dict, "last_signal_date": "2026-01-01", "regime": "contango"}))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_overlays_section()
            assert section["vix_term_structure"]["active"] is True
            assert section["vix_term_structure"]["allocation"] == alloc_dict


# ─────────────────────────────────────────────
#  Expanded Coverage: Risk Section Edge Cases
# ─────────────────────────────────────────────


class TestRiskSectionEdgeCases:
    """Edge cases for _get_risk_section: missing optional fields."""

    def test_missing_optional_fields(self, tmp_path):
        data = {"timestamp": "2026-05-16T18:00:00", "var_95_daily": -1.5}
        f = tmp_path / "risk_metrics.json"
        f.write_text(json.dumps(data))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_risk_section()
            assert section["available"] is True
            assert section["cvar_95_daily"] is None
            assert section["garch_active"] is False
            assert section["garch_filtered"] is False

    def test_garch_active_flag(self, tmp_path):
        data = {"timestamp": "2026-05-16T18:00:00", "garch_active": True, "garch_filtered": True}
        f = tmp_path / "risk_metrics.json"
        f.write_text(json.dumps(data))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_risk_section()
            assert section["garch_active"] is True
            assert section["garch_filtered"] is True

    def test_all_none_values(self, tmp_path):
        data = {
            "timestamp": "2026-05-16T18:00:00",
            "var_95_daily": None,
            "cvar_95_daily": None,
            "cvar_ratio": None,
        }
        f = tmp_path / "risk_metrics.json"
        f.write_text(json.dumps(data))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_risk_section()
            assert section["available"] is True
            assert section["var_95_daily"] is None


# ─────────────────────────────────────────────
#  Expanded Coverage: Cron Section Edge Cases
# ─────────────────────────────────────────────


class TestCronSectionEdgeCases:
    """Edge cases for _get_cron_section: status counting, missing fields."""

    def test_all_ok(self, tmp_path):
        jobs = {"jobs": [{"name": "a", "status": "ok"}, {"name": "b", "status": "ok"}]}
        f = tmp_path / "cron_status.json"
        f.write_text(json.dumps(jobs))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_cron_section()
            assert section["ok"] == 2
            assert section["errors"] == 0
            assert section["pending"] == 0

    def test_all_errors(self, tmp_path):
        jobs = {"jobs": [{"name": "a", "status": "error"}, {"name": "b", "status": "failed"}]}
        f = tmp_path / "cron_status.json"
        f.write_text(json.dumps(jobs))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_cron_section()
            assert section["ok"] == 0
            assert section["errors"] == 2
            assert section["pending"] == 0

    def test_all_pending(self, tmp_path):
        jobs = {"jobs": [{"name": "a", "status": "pending"}, {"name": "b", "status": "pending"}]}
        f = tmp_path / "cron_status.json"
        f.write_text(json.dumps(jobs))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_cron_section()
            assert section["ok"] == 0
            assert section["pending"] == 2
            assert section["errors"] == 0

    def test_mixed_statuses(self, tmp_path):
        jobs = {
            "jobs": [
                {"name": "a", "status": "ok"},
                {"name": "b", "status": "pending"},
                {"name": "c", "status": "error"},
                {"name": "d", "status": "unknown"},
            ]
        }
        f = tmp_path / "cron_status.json"
        f.write_text(json.dumps(jobs))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_cron_section()
            assert section["ok"] == 1
            assert section["pending"] == 1
            assert section["errors"] == 2
            assert section["total"] == 4

    def test_jobs_missing_optional_fields(self, tmp_path):
        jobs = {"jobs": [{"name": "a"}, {"name": "b", "status": "ok"}]}
        f = tmp_path / "cron_status.json"
        f.write_text(json.dumps(jobs))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_cron_section()
            assert section["ok"] == 1
            assert section["errors"] == 1  # "a" has no status → not ok/pending
            assert section["jobs"][0]["duration_seconds"] is None


# ─────────────────────────────────────────────
#  Expanded Coverage: Adaptive Weights Edge Cases
# ─────────────────────────────────────────────


class TestAdaptiveWeightsEdgeCases:
    """Edge cases for _get_adaptive_weights_section."""

    def test_empty_adjusted_weights(self, tmp_path):
        data = {
            "adjusted_weights": {},
            "baseline_weights": {},
            "multipliers": {},
        }
        f = tmp_path / "adaptive_weights_state.json"
        f.write_text(json.dumps(data))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_adaptive_weights_section()
            assert section["available"] is False

    def test_all_zero_changes(self, tmp_path):
        data = {
            "adjusted_weights": {"A": 0.30, "B": 0.20},
            "baseline_weights": {"A": 0.30, "B": 0.20},
            "multipliers": {"A": 1.0, "B": 1.0},
        }
        f = tmp_path / "adaptive_weights_state.json"
        f.write_text(json.dumps(data))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_adaptive_weights_section()
            assert section["available"] is True
            assert len(section["top_boosted"]) == 0
            assert len(section["top_reduced"]) == 0
            assert all(c["change"] == 0 for c in section["top_changes"])

    def test_missing_baseline_weights(self, tmp_path):
        data = {
            "adjusted_weights": {"A": 0.35},
            "multipliers": {"A": 1.17},
        }
        f = tmp_path / "adaptive_weights_state.json"
        f.write_text(json.dumps(data))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_adaptive_weights_section()
            assert section["available"] is True
            change = section["top_changes"][0]
            assert change["base_weight"] == 0  # default from baseline.get(k, 0)

    def test_single_source(self, tmp_path):
        data = {
            "adjusted_weights": {"ONLY": 0.50},
            "baseline_weights": {"ONLY": 0.30},
            "multipliers": {"ONLY": 1.67},
            "history": [{"ts": "t1"}],
        }
        f = tmp_path / "adaptive_weights_state.json"
        f.write_text(json.dumps(data))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_adaptive_weights_section()
            assert section["num_sources"] == 1
            assert len(section["top_boosted"]) == 1
            assert len(section["top_reduced"]) == 0
            assert section["history_count"] == 1

    def test_boosted_and_reduced_by_sign(self, tmp_path):
        data = {
            "adjusted_weights": {"X": 0.50, "Y": 0.10, "Z": 0.40},
            "baseline_weights": {"X": 0.30, "Y": 0.30, "Z": 0.40},
            "multipliers": {"X": 1.67, "Y": 0.33, "Z": 1.0},
        }
        f = tmp_path / "adaptive_weights_state.json"
        f.write_text(json.dumps(data))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_adaptive_weights_section()
            assert [c["source"] for c in section["top_boosted"]] == ["X"]
            assert [c["source"] for c in section["top_reduced"]] == ["Y"]


# ─────────────────────────────────────────────
#  Expanded Coverage: Attribution Edge Cases
# ─────────────────────────────────────────────


class TestAttributionSectionEdgeCases:
    """Edge cases for _get_attribution_section: empty dir, corrupt files, None values."""

    def test_empty_attribution_dir(self, tmp_path):
        attr_dir = tmp_path / "attribution"
        attr_dir.mkdir()
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_attribution_section()
            assert section["available"] is False

    def test_corrupt_attribution_file(self, tmp_path):
        attr_dir = tmp_path / "attribution"
        attr_dir.mkdir()
        f = attr_dir / "attribution_2026-05-16.json"
        f.write_text("{corrupt json!!!")
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_attribution_section()
            assert section["available"] is False

    def test_source_with_none_return(self, tmp_path):
        attr_dir = tmp_path / "attribution"
        attr_dir.mkdir()
        data = {
            "analysis_days": 30,
            "sources": {
                "a": {"display_name": "Alpha", "total_return_bps": None},
                "b": {"display_name": "Beta", "total_return_bps": 50},
            },
        }
        f = attr_dir / "attribution_2026-05-16.json"
        f.write_text(json.dumps(data))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_attribution_section()
            # Sorted by abs(total_return_bps) descending — None → 0 in abs()
            assert section["sources"][0]["source"] == "b"  # 50 > 0
            assert section["sources"][1]["source"] == "a"  # None → 0

    def test_source_missing_display_name(self, tmp_path):
        attr_dir = tmp_path / "attribution"
        attr_dir.mkdir()
        data = {
            "analysis_days": 30,
            "sources": {"a": {"total_return_bps": 100}},
        }
        f = attr_dir / "attribution_2026-05-16.json"
        f.write_text(json.dumps(data))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_attribution_section()
            assert section["sources"][0]["name"] == "a"  # fallback to key

    def test_multiple_attribution_files_picks_latest(self, tmp_path):
        attr_dir = tmp_path / "attribution"
        attr_dir.mkdir()
        older = attr_dir / "attribution_2026-05-15.json"
        older.write_text(json.dumps({"analysis_days": 60, "sources": {"a": {"display_name": "Old", "total_return_bps": 10}}}))
        newer = attr_dir / "attribution_2026-05-16.json"
        newer.write_text(json.dumps({"analysis_days": 90, "sources": {"b": {"display_name": "New", "total_return_bps": 20}}}))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            section = _get_attribution_section()
            assert section["analysis_days"] == 90
            assert section["sources"][0]["name"] == "New"


# ─────────────────────────────────────────────
#  Expanded Coverage: Formatting Edge Cases
# ─────────────────────────────────────────────


class TestFormatHelpersEdgeCases:
    """Edge cases for _fmt, _fmt_pct, _status_badge."""

    def test_fmt_zero(self):
        from src.monitor.unified_dashboard import _fmt
        assert _fmt(0) == "0"
        assert _fmt(0.0) == "0.00"

    def test_fmt_negative(self):
        from src.monitor.unified_dashboard import _fmt
        assert _fmt(-3.5) == "-3.50"

    def test_fmt_very_large(self):
        from src.monitor.unified_dashboard import _fmt
        assert _fmt(1_234_567.89) == "1234567.89"

    def test_fmt_string_value(self):
        from src.monitor.unified_dashboard import _fmt
        assert _fmt("hello") == "hello"

    def test_fmt_pct_zero(self):
        from src.monitor.unified_dashboard import _fmt_pct
        assert _fmt_pct(0) == "0.00%"

    def test_fmt_pct_negative(self):
        from src.monitor.unified_dashboard import _fmt_pct
        assert _fmt_pct(-5.5) == "-5.50%"

    def test_fmt_with_suffix_none(self):
        from src.monitor.unified_dashboard import _fmt
        assert _fmt(None, "%") == "N/A"

    def test_fmt_bool_values(self):
        from src.monitor.unified_dashboard import _fmt
        assert _fmt(True) == "True"

    def test_status_badge_truthy(self):
        from src.monitor.unified_dashboard import _status_badge
        assert _status_badge(1) == "✅"
        assert _status_badge("yes") == "✅"
        assert _status_badge("") == "❌"


# ─────────────────────────────────────────────
#  Expanded Coverage: Dashboard Generation Edge Cases
# ─────────────────────────────────────────────


class TestDashboardGenerationEdgeCases:
    """Edge cases for generate_unified_dashboard: empty data dir, field completeness."""

    def test_all_sections_not_available_in_empty_dir(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            dashboard = generate_unified_dashboard()
            for key in ["health", "portfolio", "risk", "risk_history", "tca", "regime", "attribution", "adaptive_weights", "cron"]:
                section = dashboard[key]
                assert "available" in section, f"{key} missing 'available' key"
                assert section["available"] is False, f"{key} should be unavailable"
            # Overlays section uses _meta instead of available
            assert "_meta" in dashboard["overlays"]

    def test_dashboard_version_constant(self):
        dashboard = generate_unified_dashboard()
        assert dashboard["dashboard_version"] == "v6.08"

    def test_generated_at_is_populated(self):
        dashboard = generate_unified_dashboard()
        assert "generated_at" in dashboard
        assert dashboard["generated_at"] != ""

    def test_overlays_has_meta_in_output(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            dashboard = generate_unified_dashboard()
            assert "_meta" in dashboard["overlays"]

    def test_adaptive_weights_not_in_output_when_no_file(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            dashboard = generate_unified_dashboard()
            assert dashboard["adaptive_weights"]["available"] is False

    def test_all_top_level_keys_present(self):
        dashboard = generate_unified_dashboard()
        expected = {
            "dashboard_version", "generated_at", "generated_at_local",
            "health", "portfolio", "risk", "risk_history", "tca",
            "overlays", "regime", "attribution", "adaptive_weights", "cron",
        }
        assert set(dashboard.keys()) == expected


# ─────────────────────────────────────────────
#  Expanded Coverage: generate_status_text Edge Cases
# ─────────────────────────────────────────────


class TestGenerateStatusTextEdgeCases:
    """Edge cases for generate_status_text: unavailable sections."""

    def test_all_unavailable(self, tmp_path):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            text = generate_status_text()
            assert isinstance(text, str)
            # Health unavailable → health_ok = False, cron unavailable → cron_ok = False
            assert "⚠️" in text or "✅" in text
            assert "val=$0" in text
            assert "dd=0.0%" in text

    def test_cron_has_errors(self, tmp_path):
        # Create cron with errors; no health file
        cs = tmp_path / "cron_status.json"
        cs.write_text(json.dumps({"jobs": [{"name": "test", "status": "error"}]}))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            text = generate_status_text()
            assert "cron=err" in text

    def test_cron_no_errors(self, tmp_path):
        cs = tmp_path / "cron_status.json"
        cs.write_text(json.dumps({"jobs": [{"name": "test", "status": "ok"}]}))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            text = generate_status_text()
            assert "cron=ok" in text

    def test_portfolio_with_value(self, tmp_path):
        pp = tmp_path / "portfolio_paper.json"
        pp.write_text(json.dumps({"cash": 25000, "positions": {"SPY": {"symbol": "SPY", "shares": 100, "value": 75000, "unrealized_pnl": 500}}}))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            text = generate_status_text()
            assert "val=$100,000" in text or "val=$100000" in text or "val=100000" in text


# ─────────────────────────────────────────────
#  Expanded Coverage: print_summary Edge Cases
# ─────────────────────────────────────────────


class TestPrintSummaryEdgeCases:
    """Edge cases for print_summary: all unavailable sections, edge display values."""

    def test_all_sections_unavailable(self, tmp_path, caplog):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            dashboard = generate_unified_dashboard()
            with caplog.at_level(logging.INFO, logger="src.monitor.unified_dashboard"):
                print_summary(dashboard)
            assert "HEALTH: not available" in caplog.text
            assert "PORTFOLIO: not available" in caplog.text
            assert "RISK: not available" in caplog.text
            assert "REGIME: not available" in caplog.text
            assert "TCA: not available" in caplog.text
            assert "ATTRIBUTION: not available" in caplog.text
            assert "CRON: not available" in caplog.text

    def test_dashboard_version_displayed(self, caplog):
        dashboard = generate_unified_dashboard()
        with caplog.at_level(logging.INFO, logger="src.monitor.unified_dashboard"):
            print_summary(dashboard)
        assert "v6.08" in caplog.text

    def test_risk_with_deep_drawdown(self, tmp_path, caplog):
        risk = tmp_path / "risk_metrics.json"
        risk.write_text(json.dumps({
            "var_95_daily": -3.0,
            "cvar_95_daily": -5.0,
            "cvar_ratio": 2.5,
            "tail_severity": "severe",
            "max_drawdown": -35.0,
            "current_drawdown": -25.0,
            "volatility_annual": 25.0,
        }))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            dashboard = generate_unified_dashboard()
            with caplog.at_level(logging.INFO, logger="src.monitor.unified_dashboard"):
                print_summary(dashboard)
            assert "🚨" in caplog.text  # drawdown badge for < -20
            assert "SEVERE" in caplog.text
            assert "-25.00%" in caplog.text

    def test_overlays_vixy_dict_in_print(self, tmp_path, caplog):
        vixy = tmp_path / "vixy_hedge_state.json"
        vixy.write_text(json.dumps({"current_allocation": {"SPY": 0.5, "GLD": 0.5}, "last_signal_date": "2026-01-01", "regime": "contango"}))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            dashboard = generate_unified_dashboard()
            with caplog.at_level(logging.INFO, logger="src.monitor.unified_dashboard"):
                print_summary(dashboard)
            # Dict allocation branch: prints SPY=50.00% GLD=50.00%
            assert "SPY=50.00%" in caplog.text
            assert "GLD=50.00%" in caplog.text

    def test_overlays_vixy_scalar_in_print(self, tmp_path, caplog):
        vixy = tmp_path / "vixy_hedge_state.json"
        vixy.write_text(json.dumps({"current_allocation": 0.15, "last_signal_date": "2026-01-01", "regime": "contango"}))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            dashboard = generate_unified_dashboard()
            with caplog.at_level(logging.INFO, logger="src.monitor.unified_dashboard"):
                print_summary(dashboard)
            # Scalar allocation branch: prints alloc=15.00%
            assert "alloc=15.00%" in caplog.text

    def test_cron_section_with_mixed_jobs(self, tmp_path, caplog):
        cs = tmp_path / "cron_status.json"
        cs.write_text(json.dumps({
            "jobs": [
                {"name": "ok-job", "status": "ok", "duration_seconds": 5.0},
                {"name": "err-job", "status": "error", "duration_seconds": None},
                {"name": "pending-job", "status": "pending", "duration_seconds": 0},
            ]
        }))
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            dashboard = generate_unified_dashboard()
            with caplog.at_level(logging.INFO, logger="src.monitor.unified_dashboard"):
                print_summary(dashboard)
            assert "1/3 ok" in caplog.text
            assert "1 errors" in caplog.text


# ─────────────────────────────────────────────
#  Expanded Coverage: CLI Main Edge Cases
# ─────────────────────────────────────────────


class TestCLIEdgeCases:
    """Edge cases for the main() CLI entry point."""

    def test_no_flag_calls_print_summary(self, tmp_path, caplog):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            test_args = ["unified_dashboard.py"]
            with patch.object(sys, "argv", test_args):
                with caplog.at_level(logging.INFO, logger="src.monitor.unified_dashboard"):
                    main()
            assert "UNIFIED DASHBOARD" in caplog.text

    def test_json_flag_output(self, tmp_path, capsys):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            test_args = ["unified_dashboard.py", "--json"]
            with patch.object(sys, "argv", test_args):
                main()
            saved = tmp_path / "unified_dashboard.json"
            assert saved.exists()
            data = json.loads(saved.read_text())
            assert "dashboard_version" in data

    def test_save_and_json_both_set(self, tmp_path, capsys):
        """--save and --json both trigger save; only one save needed."""
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            test_args = ["unified_dashboard.py", "--save", "--json"]
            with patch.object(sys, "argv", test_args):
                main()
            saved = tmp_path / "unified_dashboard.json"
            assert saved.exists()

    def test_check_flag_saves_and_exits_zero(self, tmp_path):
        """--check also saves because --save is not passed, but should exit."""
        hr = tmp_path / ".health_report.json"
        hr.write_text(json.dumps({"status": "healthy", "checks": {}, "alerts": [], "summary": {"total_checks": 1, "passed": 1, "failed": 0}}))
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

    def test_status_text_flag_returns_string(self, tmp_path, capsys):
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            test_args = ["unified_dashboard.py", "--status-text"]
            with patch.object(sys, "argv", test_args):
                main()
            captured = capsys.readouterr()
            assert len(captured.out) > 0

    def test_check_flag_with_errors_only(self, tmp_path):
        """Cron has errors; health is fine → should exit 1."""
        hr = tmp_path / ".health_report.json"
        hr.write_text(json.dumps({"status": "healthy", "checks": {}, "alerts": [], "summary": {"total_checks": 1, "passed": 1, "failed": 0}}))
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


# ─────────────────────────────────────────────
#  Expanded Coverage: _read_json Edge Cases
# ─────────────────────────────────────────────


class TestReadJsonEdgeCases:
    """Edge cases for _read_json: empty file, non-dict JSON, binary content."""

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text("")
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            result = _read_json("empty.json")
            assert result is None

    def test_json_array(self, tmp_path):
        f = tmp_path / "array.json"
        f.write_text("[1, 2, 3]")
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            result = _read_json("array.json")
            assert result == [1, 2, 3]

    def test_json_null(self, tmp_path):
        f = tmp_path / "null.json"
        f.write_text("null")
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            result = _read_json("null.json")
            assert result is None

    def test_json_number(self, tmp_path):
        f = tmp_path / "number.json"
        f.write_text("42")
        with patch("src.monitor.unified_dashboard.DATA_DIR", tmp_path):
            result = _read_json("number.json")
            assert result == 42
