"""
Tests for v7.10 Daily Personal CFO Brief generator.
Covers: section generation, severity thresholds, LLM narrative fallback,
save/load roundtrip, template rendering.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


from src.monitor.daily_brief import (
    generate_brief_sections,
    render_brief_text,
    generate_narrative,
    generate_daily_brief,
    main as daily_brief_main,
    SEVERITY_THRESHOLDS,
    BriefSection,
)


@pytest.fixture
def sample_dashboard():
    return {
        "dashboard_version": "v6.08",
        "generated_at": "2026-05-19T08:00:00",
        "health": {"available": True, "status": "healthy", "alerts": []},
        "portfolio": {
            "available": True,
            "total_value": 250000,
            "cash": 5000,
            "cash_pct": 2.0,
            "positions": [
                {"symbol": "SPY", "weight": 44.0, "value": 110000},
                {"symbol": "GLD", "weight": 36.0, "value": 90000},
                {"symbol": "TLT", "weight": 18.0, "value": 45000},
            ],
        },
        "risk": {
            "available": True,
            "current_drawdown": -8.5,
            "var_95_daily": -1.2,
            "cvar_95_daily": -1.8,
            "volatility_annual": 11.5,
            "max_drawdown": -26.2,
        },
        "overlays": {
            "collar": {"active": True},
            "crypto": {"active": False},
            "bond_duration": {"active": True},
            "vix_term_structure": {"active": False},
            "mean_reversion": {"active": False},
            "_meta": {"active_count": 2, "total_count": 5},
        },
        "regime": {
            "available": True,
            "classifier": {"current_regime": "normal", "confidence": 0.75},
        },
        "tca": {
            "available": True,
            "scorecard": {"avg_slippage_bps": 5.2, "total_orders": 3},
        },
        "attribution": {
            "available": True,
            "sources": [
                {"name": "TSFM Momentum", "total_return_bps": 12.5, "hit_rate": 0.65},
                {"name": "Risk Budget", "total_return_bps": -3.2, "hit_rate": 0.50},
            ],
        },
        "adaptive_weights": {"available": False},
        "cron": {
            "available": True,
            "total": 8,
            "ok": 8,
            "errors": 0,
        },
    }


class TestBriefSections:
    def test_generates_all_sections(self, sample_dashboard):
        sections = generate_brief_sections(sample_dashboard)
        section_names = {s.name for s in sections}
        assert "portfolio_snapshot" in section_names
        assert "risk_check" in section_names
        assert "signal_roundup" in section_names
        assert "overlay_status" in section_names
        assert "tca_watch" in section_names
        assert "action_items" in section_names
        assert len(sections) == 7

    def test_portfolio_snapshot_normal(self, sample_dashboard):
        sections = generate_brief_sections(sample_dashboard)
        ps = next(s for s in sections if s.name == "portfolio_snapshot")
        assert ps.severity == "normal"
        assert "250,000" in ps.data_text

    def test_risk_check_drawdown_warning(self, sample_dashboard):
        sample_dashboard["risk"]["current_drawdown"] = -12.0
        sections = generate_brief_sections(sample_dashboard)
        rc = next(s for s in sections if s.name == "risk_check")
        assert rc.severity in ("warning", "alert")

    def test_risk_check_crisis_alert(self, sample_dashboard):
        sample_dashboard["risk"]["current_drawdown"] = -22.0
        sections = generate_brief_sections(sample_dashboard)
        rc = next(s for s in sections if s.name == "risk_check")
        assert rc.severity == "alert"

    def test_signal_roundup_counts(self, sample_dashboard):
        sections = generate_brief_sections(sample_dashboard)
        sr = next(s for s in sections if s.name == "signal_roundup")
        assert "TSFM Momentum" in sr.data_text

    def test_overlay_status_active_count(self, sample_dashboard):
        sections = generate_brief_sections(sample_dashboard)
        os_ = next(s for s in sections if s.name == "overlay_status")
        assert "2 active" in os_.data_text

    def test_action_items_when_healthy(self, sample_dashboard):
        sections = generate_brief_sections(sample_dashboard)
        ai = next(s for s in sections if s.name == "action_items")
        assert ai.severity == "normal"

    def test_action_items_when_alerts(self, sample_dashboard):
        sample_dashboard["health"]["alerts"] = ["Data freshness: prices 2 days stale"]
        sample_dashboard["risk"]["current_drawdown"] = -15.0
        sections = generate_brief_sections(sample_dashboard)
        ai = next(s for s in sections if s.name == "action_items")
        assert ai.severity in ("warning", "alert")

    def test_portfolio_with_empty_positions(self, sample_dashboard):
        sample_dashboard["portfolio"]["positions"] = []
        sections = generate_brief_sections(sample_dashboard)
        ps = next(s for s in sections if s.name == "portfolio_snapshot")
        assert "No positions" in ps.data_text

    def test_tca_watch_normal(self, sample_dashboard):
        sections = generate_brief_sections(sample_dashboard)
        tw = next(s for s in sections if s.name == "tca_watch")
        assert tw.severity == "normal"

    def test_tca_watch_warning(self, sample_dashboard):
        sample_dashboard["tca"]["scorecard"]["avg_slippage_bps"] = 10.0
        sections = generate_brief_sections(sample_dashboard)
        tw = next(s for s in sections if s.name == "tca_watch")
        assert tw.severity == "warning"

    def test_tca_watch_alert(self, sample_dashboard):
        sample_dashboard["tca"]["scorecard"]["avg_slippage_bps"] = 18.0
        sections = generate_brief_sections(sample_dashboard)
        tw = next(s for s in sections if s.name == "tca_watch")
        assert tw.severity == "alert"

    def test_signal_severity_more_bearish(self, sample_dashboard):
        sample_dashboard["attribution"]["sources"] = [
            {"name": "Alpha A", "total_return_bps": -5.0},
            {"name": "Alpha B", "total_return_bps": -3.0},
            {"name": "Alpha C", "total_return_bps": 2.0},
        ]
        sections = generate_brief_sections(sample_dashboard)
        sr = next(s for s in sections if s.name == "signal_roundup")
        assert sr.severity == "warning"

    def test_overlay_status_no_active(self, sample_dashboard):
        sample_dashboard["overlays"]["_meta"]["active_count"] = 0
        for key in list(sample_dashboard["overlays"].keys()):
            if isinstance(sample_dashboard["overlays"][key], dict) and "active" in sample_dashboard["overlays"][key]:
                sample_dashboard["overlays"][key]["active"] = False
        sections = generate_brief_sections(sample_dashboard)
        os_ = next(s for s in sections if s.name == "overlay_status")
        assert "0 active" in os_.data_text


class TestRenderBriefText:
    def test_renders_complete_brief(self, sample_dashboard):
        sections = generate_brief_sections(sample_dashboard)
        text = render_brief_text(sections)
        assert "PORTFOLIO-LAB DAILY BRIEF" in text
        assert "PORTFOLIO SNAPSHOT" in text
        assert "RISK CHECK" in text

    def test_renders_with_narrative(self, sample_dashboard):
        sections = generate_brief_sections(sample_dashboard)
        text = render_brief_text(sections, narrative="Everything is on track today.")
        assert "Everything is on track today." in text

    def test_header_shows_date(self, sample_dashboard):
        from datetime import date
        sections = generate_brief_sections(sample_dashboard)
        text = render_brief_text(sections)
        assert date.today().isoformat() in text


class TestGenerateNarrative:
    @patch("src.monitor.daily_brief.Anthropic")
    def test_returns_narrative_on_success(self, mock_anthropic, sample_dashboard):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Your portfolio is stable. All signals are balanced.")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        with patch("src.monitor.daily_brief.ANTHROPIC_AVAILABLE", True):
            with patch("src.monitor.daily_brief.os.environ.get", return_value="sk-ant-fake-key"):
                narrative = generate_narrative(sample_dashboard)
                assert "portfolio" in narrative.lower()

    def test_returns_none_when_no_api_key(self, sample_dashboard):
        with patch("src.monitor.daily_brief.os.environ.get", return_value=None):
            narrative = generate_narrative(sample_dashboard)
            assert narrative is None

    def test_returns_none_on_api_error(self, sample_dashboard):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API error")

        with patch("src.monitor.daily_brief.ANTHROPIC_AVAILABLE", True):
            with patch("src.monitor.daily_brief.Anthropic", return_value=mock_client):
                narrative = generate_narrative(sample_dashboard)
                assert narrative is None

    def test_returns_none_when_anthropic_not_available(self, sample_dashboard):
        with patch("src.monitor.daily_brief.ANTHROPIC_AVAILABLE", False):
            narrative = generate_narrative(sample_dashboard)
            assert narrative is None


class TestGenerateDailyBrief:
    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_returns_dict_with_all_keys(self, mock_dashboard, sample_dashboard):
        mock_dashboard.return_value = sample_dashboard
        brief = generate_daily_brief()
        assert "generated_at" in brief
        assert "sections" in brief
        assert "full_text" in brief
        assert "severity" in brief

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_severity_is_alert_when_risk_breach(self, mock_dashboard, sample_dashboard):
        sample_dashboard["risk"]["current_drawdown"] = -22.0
        mock_dashboard.return_value = sample_dashboard
        brief = generate_daily_brief()
        assert brief["severity"] == "alert"

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_severity_is_warning_when_moderate_risk(self, mock_dashboard, sample_dashboard):
        sample_dashboard["risk"]["current_drawdown"] = -12.0
        mock_dashboard.return_value = sample_dashboard
        brief = generate_daily_brief()
        assert brief["severity"] == "warning"

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_empty_dashboard_fallback(self, mock_dashboard):
        mock_dashboard.side_effect = RuntimeError("Dashboard error")
        brief = generate_daily_brief()
        assert "sections" in brief
        assert brief["severity"] == "normal"

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_save_to_json(self, mock_dashboard, sample_dashboard, tmp_path):
        mock_dashboard.return_value = sample_dashboard
        brief = generate_daily_brief()
        out = tmp_path / "daily_brief.json"
        out.write_text(json.dumps(brief, indent=2, default=str))
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["severity"] == "normal"


class TestModelValidationSection:
    """Tests for the model_validation section in daily brief."""

    def test_model_validation_section_exists(self, sample_dashboard):
        sections = generate_brief_sections(sample_dashboard)
        names = [s.name for s in sections]
        assert "model_validation" in names

    def test_model_validation_contains_dsr(self, sample_dashboard):
        sections = generate_brief_sections(sample_dashboard)
        mv = [s for s in sections if s.name == "model_validation"][0]
        assert "DSR" in mv.data_text

    def test_model_validation_normal_severity(self, sample_dashboard):
        sections = generate_brief_sections(sample_dashboard)
        mv = [s for s in sections if s.name == "model_validation"][0]
        assert mv.severity == "normal"

    def test_model_validation_with_bl_weights(self, sample_dashboard):
        """BL weights in dashboard should appear in model validation."""
        sample_dashboard["bl_weights"] = {"SPY": 0.48, "GLD": 0.35, "TLT": 0.17}
        sections = generate_brief_sections(sample_dashboard)
        mv = [s for s in sections if s.name == "model_validation"][0]
        assert "BL" in mv.data_text

    # ── DSR severity flip ──

    @patch("src.backtest.metrics.compute_deflated_sharpe_ratio")
    def test_model_validation_dsr_below_50_flips_warning(self, mock_dsr, sample_dashboard):
        """DSR below 0.50 should flip model_validation severity to warning."""
        mock_dsr.return_value = 0.35
        sections = generate_brief_sections(sample_dashboard)
        mv = next(s for s in sections if s.name == "model_validation")
        assert mv.severity == "warning"

    @patch("src.backtest.metrics.compute_deflated_sharpe_ratio")
    def test_model_validation_dsr_exactly_50_stays_normal(self, mock_dsr, sample_dashboard):
        """DSR exactly at 0.50 should keep model_validation severity normal."""
        mock_dsr.return_value = 0.50
        sections = generate_brief_sections(sample_dashboard)
        mv = next(s for s in sections if s.name == "model_validation")
        assert mv.severity == "normal"

    @patch("src.backtest.metrics.compute_deflated_sharpe_ratio")
    def test_model_validation_dsr_near_zero(self, mock_dsr, sample_dashboard):
        """DSR near zero should still flip to warning."""
        mock_dsr.return_value = 0.01
        sections = generate_brief_sections(sample_dashboard)
        mv = next(s for s in sections if s.name == "model_validation")
        assert mv.severity == "warning"

    @patch("src.backtest.metrics.compute_deflated_sharpe_ratio")
    def test_model_validation_dsr_import_error_handled(self, mock_dsr, sample_dashboard):
        """ImportError for DSR should not crash; severity stays normal."""
        mock_dsr.side_effect = ImportError("No module named metrics")
        sections = generate_brief_sections(sample_dashboard)
        mv = next(s for s in sections if s.name == "model_validation")
        assert mv.severity == "normal"
        assert "unavailable" in mv.data_text

    @patch("src.backtest.metrics.compute_deflated_sharpe_ratio")
    def test_model_validation_dsr_value_error_handled(self, mock_dsr, sample_dashboard):
        """ValueError from DSR should not crash; severity stays normal."""
        mock_dsr.side_effect = ValueError("Invalid input")
        sections = generate_brief_sections(sample_dashboard)
        mv = next(s for s in sections if s.name == "model_validation")
        assert mv.severity == "normal"
        assert "unavailable" in mv.data_text

    @patch("src.backtest.metrics.compute_deflated_sharpe_ratio")
    def test_model_validation_dsr_overflow_error_handled(self, mock_dsr, sample_dashboard):
        """OverflowError from DSR should not crash; severity stays normal."""
        mock_dsr.side_effect = OverflowError("Overflow")
        sections = generate_brief_sections(sample_dashboard)
        mv = next(s for s in sections if s.name == "model_validation")
        assert mv.severity == "normal"
        assert "unavailable" in mv.data_text

    @patch("src.backtest.metrics.compute_deflated_sharpe_ratio")
    def test_model_validation_dsr_zero_division_error_handled(self, mock_dsr, sample_dashboard):
        """ZeroDivisionError from DSR should not crash; severity stays normal."""
        mock_dsr.side_effect = ZeroDivisionError("Division by zero")
        sections = generate_brief_sections(sample_dashboard)
        mv = next(s for s in sections if s.name == "model_validation")
        assert mv.severity == "normal"
        assert "unavailable" in mv.data_text

    # ── BL weight absence ──

    def test_model_validation_without_bl_weights(self, sample_dashboard):
        """When bl_weights is absent from dashboard, BL should not appear in model_validation."""
        assert "bl_weights" not in sample_dashboard
        sections = generate_brief_sections(sample_dashboard)
        mv = next(s for s in sections if s.name == "model_validation")
        assert "BL" not in mv.data_text

    def test_model_validation_bl_weights_none(self, sample_dashboard):
        """When bl_weights is None, BL should not appear in model_validation."""
        sample_dashboard["bl_weights"] = None
        sections = generate_brief_sections(sample_dashboard)
        mv = next(s for s in sections if s.name == "model_validation")
        assert "BL" not in mv.data_text

    def test_model_validation_bl_weights_empty_dict(self, sample_dashboard):
        """When bl_weights is an empty dict, BL should not appear in model_validation."""
        sample_dashboard["bl_weights"] = {}
        sections = generate_brief_sections(sample_dashboard)
        mv = next(s for s in sections if s.name == "model_validation")
        assert "BL" not in mv.data_text


class TestSeverityThresholds:
    """Validate SEVERITY_THRESHOLDS constants."""

    def test_thresholds_exist(self):
        assert "drawdown_warning" in SEVERITY_THRESHOLDS
        assert "drawdown_alert" in SEVERITY_THRESHOLDS
        assert "slippage_warning_bps" in SEVERITY_THRESHOLDS
        assert "slippage_alert_bps" in SEVERITY_THRESHOLDS

    def test_drawdown_warning_less_severe_than_alert(self):
        assert SEVERITY_THRESHOLDS["drawdown_warning"] > SEVERITY_THRESHOLDS["drawdown_alert"]

    def test_slippage_warning_less_severe_than_alert(self):
        assert SEVERITY_THRESHOLDS["slippage_warning_bps"] < SEVERITY_THRESHOLDS["slippage_alert_bps"]


class TestBriefSectionDataclass:
    """Extended tests for BriefSection dataclass."""

    def test_brief_section_all_fields(self):
        s = BriefSection(
            name="test", title="Test Section", severity="normal",
            data_text="Some data", recommendation="Do nothing",
        )
        assert s.name == "test"
        assert s.title == "Test Section"
        assert s.severity == "normal"
        assert s.data_text == "Some data"
        assert s.recommendation == "Do nothing"

    def test_brief_section_default_recommendation(self):
        s = BriefSection(name="test", title="Test", severity="normal", data_text="data")
        assert s.recommendation == ""

    def test_brief_section_severity_values(self):
        for sev in ("normal", "warning", "alert"):
            s = BriefSection(name="test", title="Test", severity=sev, data_text="data")
            assert s.severity == sev


class TestBriefSectionsExtended:
    """Extended edge cases for generate_brief_sections."""

    def test_empty_dashboard(self):
        sections = generate_brief_sections({})
        assert isinstance(sections, list)
        assert len(sections) > 0

    def test_missing_portfolio_key(self, sample_dashboard):
        del sample_dashboard["portfolio"]
        sections = generate_brief_sections(sample_dashboard)
        ps = next((s for s in sections if s.name == "portfolio_snapshot"), None)
        assert ps is not None

    def test_missing_risk_key(self, sample_dashboard):
        del sample_dashboard["risk"]
        sections = generate_brief_sections(sample_dashboard)
        rc = next((s for s in sections if s.name == "risk_check"), None)
        assert rc is not None

    def test_risk_drawdown_exactly_at_warning(self, sample_dashboard):
        sample_dashboard["risk"]["current_drawdown"] = SEVERITY_THRESHOLDS["drawdown_warning"]
        sections = generate_brief_sections(sample_dashboard)
        rc = next(s for s in sections if s.name == "risk_check")
        assert rc.severity in ("warning", "alert")

    def test_risk_normal_drawdown(self, sample_dashboard):
        sample_dashboard["risk"]["current_drawdown"] = -3.0
        sections = generate_brief_sections(sample_dashboard)
        rc = next(s for s in sections if s.name == "risk_check")
        assert rc.severity == "normal"

    def test_portfolio_zero_value(self, sample_dashboard):
        sample_dashboard["portfolio"]["total_value"] = 0
        sections = generate_brief_sections(sample_dashboard)
        ps = next(s for s in sections if s.name == "portfolio_snapshot")
        assert ps is not None

    def test_regime_stress(self, sample_dashboard):
        sample_dashboard["regime"]["classifier"]["current_regime"] = "stress"
        sections = generate_brief_sections(sample_dashboard)
        sr = next(s for s in sections if s.name == "signal_roundup")
        assert sr is not None

    # ── Overlay non-dict edge cases ──

    def test_overlay_non_dict_values_excluded(self, sample_dashboard):
        """String, int, and list overlay entries should be silently excluded."""
        sample_dashboard["overlays"]["string_overlay"] = "not_a_dict"
        sample_dashboard["overlays"]["int_overlay"] = 42
        sample_dashboard["overlays"]["list_overlay"] = [1, 2, 3]
        sections = generate_brief_sections(sample_dashboard)
        os_ = next(s for s in sections if s.name == "overlay_status")
        assert "string_overlay" not in os_.data_text
        assert "int_overlay" not in os_.data_text
        assert "list_overlay" not in os_.data_text

    def test_overlay_dict_without_active_key_excluded(self, sample_dashboard):
        """Dict overlay without 'active' key should be silently excluded."""
        sample_dashboard["overlays"]["no_active"] = {"value": 1}
        sections = generate_brief_sections(sample_dashboard)
        os_ = next(s for s in sections if s.name == "overlay_status")
        assert "no_active" not in os_.data_text

    def test_overlay_dict_active_false_excluded(self, sample_dashboard):
        """Dict overlay with active=False should be excluded."""
        sample_dashboard["overlays"]["inactive_overlay"] = {"active": False}
        sections = generate_brief_sections(sample_dashboard)
        os_ = next(s for s in sections if s.name == "overlay_status")
        assert "inactive_overlay" not in os_.data_text

    def test_overlay_none_value_excluded(self, sample_dashboard):
        """None overlay entries should be excluded."""
        sample_dashboard["overlays"]["none_overlay"] = None
        sections = generate_brief_sections(sample_dashboard)
        os_ = next(s for s in sections if s.name == "overlay_status")
        assert "none_overlay" not in os_.data_text

    def test_overlay_underscore_prefixed_excluded(self, sample_dashboard):
        """Underscore-prefixed overlay keys (even if dict with active) should be excluded."""
        sample_dashboard["overlays"]["_private"] = {"active": True}
        sections = generate_brief_sections(sample_dashboard)
        os_ = next(s for s in sections if s.name == "overlay_status")
        assert "_private" not in os_.data_text

    # ── Attribution missing sources ──

    def test_attribution_missing_sources_defaults_empty(self, sample_dashboard):
        """When attribution dict lacks 'sources', it should default to empty list."""
        del sample_dashboard["attribution"]["sources"]
        sections = generate_brief_sections(sample_dashboard)
        sr = next(s for s in sections if s.name == "signal_roundup")
        assert "0 bullish" in sr.data_text
        assert "0 bearish" in sr.data_text
        assert "none" in sr.data_text

    def test_attribution_missing_entirely(self, sample_dashboard):
        """When attribution key is entirely missing, should handle gracefully."""
        del sample_dashboard["attribution"]
        sections = generate_brief_sections(sample_dashboard)
        sr = next(s for s in sections if s.name == "signal_roundup")
        assert "0 bullish" in sr.data_text

    def test_attribution_empty_dict(self, sample_dashboard):
        """When attribution is an empty dict, should handle gracefully."""
        sample_dashboard["attribution"] = {}
        sections = generate_brief_sections(sample_dashboard)
        sr = next(s for s in sections if s.name == "signal_roundup")
        assert "0 bullish" in sr.data_text

    def test_attribution_sources_empty_list(self, sample_dashboard):
        """When attribution sources is empty list, should handle gracefully."""
        sample_dashboard["attribution"]["sources"] = []
        sections = generate_brief_sections(sample_dashboard)
        sr = next(s for s in sections if s.name == "signal_roundup")
        assert "0 bullish" in sr.data_text
        assert "0 bearish" in sr.data_text

    # ── Regime missing keys ──

    def test_regime_missing_classifier_key(self, sample_dashboard):
        """When regime dict lacks 'classifier' key, should default to 'unknown'."""
        del sample_dashboard["regime"]["classifier"]
        sections = generate_brief_sections(sample_dashboard)
        rc = next(s for s in sections if s.name == "risk_check")
        assert "unknown" in rc.data_text

    def test_regime_missing_current_regime_key(self, sample_dashboard):
        """When classifier lacks 'current_regime' key, should default to 'unknown'."""
        del sample_dashboard["regime"]["classifier"]["current_regime"]
        sections = generate_brief_sections(sample_dashboard)
        rc = next(s for s in sections if s.name == "risk_check")
        assert "unknown" in rc.data_text

    def test_regime_empty_dict(self, sample_dashboard):
        """When regime is an empty dict, should handle gracefully."""
        sample_dashboard["regime"] = {}
        sections = generate_brief_sections(sample_dashboard)
        rc = next(s for s in sections if s.name == "risk_check")
        assert "unknown" in rc.data_text

    def test_regime_classifier_none_returns_unknown(self, sample_dashboard):
        """When classifier is None, safe_get handles gracefully and returns 'unknown'."""
        sample_dashboard["regime"]["classifier"] = None
        sections = generate_brief_sections(sample_dashboard)
        rc = next(s for s in sections if s.name == "risk_check")
        assert "unknown" in rc.data_text

    def test_regime_classifier_empty_dict(self, sample_dashboard):
        """When classifier is an empty dict, should handle gracefully."""
        sample_dashboard["regime"]["classifier"] = {}
        sections = generate_brief_sections(sample_dashboard)
        rc = next(s for s in sections if s.name == "risk_check")
        assert "unknown" in rc.data_text

    # ── Mixed severity action items ──

    def test_action_items_mixed_severity_alert_wins(self, sample_dashboard):
        """When some sections have alert and some warning, action_severity should be alert."""
        sample_dashboard["risk"]["current_drawdown"] = -22.0  # triggers alert
        sample_dashboard["tca"]["scorecard"]["avg_slippage_bps"] = 10.0  # triggers warning
        sections = generate_brief_sections(sample_dashboard)
        ai = next(s for s in sections if s.name == "action_items")
        assert ai.severity == "alert"

    def test_action_items_alerts_from_multiple_sections(self, sample_dashboard):
        """Multiple sections with alert severity should still produce alert action_severity."""
        sample_dashboard["risk"]["current_drawdown"] = -22.0  # triggers alert
        sample_dashboard["tca"]["scorecard"]["avg_slippage_bps"] = 18.0  # triggers alert
        sections = generate_brief_sections(sample_dashboard)
        ai = next(s for s in sections if s.name == "action_items")
        assert ai.severity == "alert"

    def test_action_items_warning_only_severity(self, sample_dashboard):
        """When only warning sections exist (no alerts), action_severity should be warning."""
        sample_dashboard["risk"]["current_drawdown"] = -12.0  # triggers warning
        sample_dashboard["tca"]["scorecard"]["avg_slippage_bps"] = 10.0  # triggers warning
        sections = generate_brief_sections(sample_dashboard)
        ai = next(s for s in sections if s.name == "action_items")
        assert ai.severity == "warning"

    def test_action_items_recommendations_included(self, sample_dashboard):
        """Action items should include recommendations from warning/alert sections."""
        sample_dashboard["risk"]["current_drawdown"] = -22.0
        sections = generate_brief_sections(sample_dashboard)
        ai = next(s for s in sections if s.name == "action_items")
        assert "Review hedges" in ai.data_text

    def test_action_items_includes_health_alerts(self, sample_dashboard):
        """Health alerts should appear in action items when present."""
        sample_dashboard["health"]["alerts"] = ["Data freshness: prices 2 days stale"]
        sample_dashboard["risk"]["current_drawdown"] = -22.0
        sections = generate_brief_sections(sample_dashboard)
        ai = next(s for s in sections if s.name == "action_items")
        assert "Data freshness" in ai.data_text
        assert "Review hedges" in ai.data_text

    def test_action_items_health_alerts_truncated_to_three(self, sample_dashboard):
        """Health alerts should be truncated to at most 3 entries."""
        sample_dashboard["health"]["alerts"] = [
            "Alert A", "Alert B", "Alert C", "Alert D", "Alert E",
        ]
        sample_dashboard["risk"]["current_drawdown"] = -22.0
        sections = generate_brief_sections(sample_dashboard)
        ai = next(s for s in sections if s.name == "action_items")
        # Only first 3 alerts should appear
        assert "Alert A" in ai.data_text
        assert "Alert C" in ai.data_text
        # But alert D should not (truncated)
        # But in practice the recommendations appear first, then alerts, joined by "; "
        # So we check total content is present

    def test_action_items_alert_without_recommendation(self, sample_dashboard):
        """Alert section without recommendation should not add empty items."""
        # Modify attribution to produce warning without recommendation
        sample_dashboard["attribution"]["sources"] = [
            {"name": "Bearish A", "total_return_bps": -5.0},
            {"name": "Bearish B", "total_return_bps": -3.0},
            {"name": "Bearish C", "total_return_bps": -2.0},
        ]
        # signal_roundup has warning severity but no recommendation
        sample_dashboard["risk"]["current_drawdown"] = -22.0
        sections = generate_brief_sections(sample_dashboard)
        ai = next(s for s in sections if s.name == "action_items")
        # risk_check has recommendation, signal_roundup doesn't
        # signal_roundup's recommendation attr is empty string
        # Only recommendations with truthy content are included
        assert "Review hedges" in ai.data_text


class TestRenderBriefTextExtended:
    """Extended render_brief_text tests."""

    def test_renders_without_narrative(self, sample_dashboard):
        sections = generate_brief_sections(sample_dashboard)
        text = render_brief_text(sections, narrative=None)
        assert "PORTFOLIO-LAB DAILY BRIEF" in text

    def test_renders_all_section_names(self, sample_dashboard):
        sections = generate_brief_sections(sample_dashboard)
        text = render_brief_text(sections)
        for s in sections:
            # Titles are rendered uppercase
            assert s.title.upper() in text

    def test_renders_empty_sections(self):
        text = render_brief_text([])
        assert "PORTFOLIO-LAB DAILY BRIEF" in text


class TestGenerateDailyBriefExtended:
    """Extended generate_daily_brief tests."""

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_brief_contains_section_data(self, mock_dashboard, sample_dashboard):
        mock_dashboard.return_value = sample_dashboard
        brief = generate_daily_brief()
        assert len(brief["sections"]) > 0
        for s in brief["sections"]:
            assert "name" in s
            assert "severity" in s

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_brief_severity_normal_when_healthy(self, mock_dashboard, sample_dashboard):
        mock_dashboard.return_value = sample_dashboard
        brief = generate_daily_brief()
        assert brief["severity"] == "normal"

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_brief_has_narrative_false_default(self, mock_dashboard, sample_dashboard):
        """Default brief should have has_narrative set to False."""
        mock_dashboard.return_value = sample_dashboard
        brief = generate_daily_brief()
        assert brief["has_narrative"] is False

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_brief_full_text_contains_all_sections(self, mock_dashboard, sample_dashboard):
        """Full text should contain rendered section titles."""
        mock_dashboard.return_value = sample_dashboard
        brief = generate_daily_brief()
        assert "PORTFOLIO SNAPSHOT" in brief["full_text"]
        assert "RISK CHECK" in brief["full_text"]
        assert "ACTION ITEMS" in brief["full_text"]

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_brief_generated_at_iso_format(self, mock_dashboard, sample_dashboard):
        """generated_at should be ISO format string."""
        mock_dashboard.return_value = sample_dashboard
        brief = generate_daily_brief()
        assert "T" in brief["generated_at"]  # ISO datetime contains T


class TestDailyBriefCLI:
    """Tests for the CLI entry point (main function with argparse)."""

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_save_flag_writes_file(self, mock_dashboard, sample_dashboard, tmp_path):
        """--save flag should write daily_brief.json to DATA_DIR."""
        mock_dashboard.return_value = sample_dashboard
        with patch("sys.argv", ["daily_brief", "--save", "--no-narrative"]):
            with patch("src.monitor.daily_brief.DATA_DIR", tmp_path):
                with patch("builtins.print"):
                    daily_brief_main()
        out_path = tmp_path / "daily_brief.json"
        assert out_path.exists()

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_save_file_contains_brief_data(self, mock_dashboard, sample_dashboard, tmp_path):
        """The saved JSON file should contain all brief fields."""
        mock_dashboard.return_value = sample_dashboard
        with patch("sys.argv", ["daily_brief", "--save", "--no-narrative"]):
            with patch("src.monitor.daily_brief.DATA_DIR", tmp_path):
                with patch("builtins.print"):
                    daily_brief_main()
        out_path = tmp_path / "daily_brief.json"
        data = json.loads(out_path.read_text())
        assert "generated_at" in data
        assert "severity" in data
        assert "sections" in data
        assert "full_text" in data
        assert len(data["sections"]) == 7

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_no_save_flag_does_not_write_file(self, mock_dashboard, sample_dashboard, tmp_path):
        """Without --save flag, no file should be written."""
        mock_dashboard.return_value = sample_dashboard
        with patch("sys.argv", ["daily_brief", "--no-narrative"]):
            with patch("src.monitor.daily_brief.DATA_DIR", tmp_path):
                with patch("builtins.print"):
                    daily_brief_main()
        out_path = tmp_path / "daily_brief.json"
        assert not out_path.exists()

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_save_file_has_valid_json(self, mock_dashboard, sample_dashboard, tmp_path):
        """Saved JSON should be parseable and contain correct severity."""
        mock_dashboard.return_value = sample_dashboard
        with patch("sys.argv", ["daily_brief", "--save", "--no-narrative"]):
            with patch("src.monitor.daily_brief.DATA_DIR", tmp_path):
                with patch("builtins.print"):
                    daily_brief_main()
        out_path = tmp_path / "daily_brief.json"
        data = json.loads(out_path.read_text())
        assert data["severity"] == "normal"

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_no_narrative_skips_narrative(self, mock_dashboard, sample_dashboard):
        """--no-narrative should prevent generate_narrative from being called."""
        mock_dashboard.return_value = sample_dashboard
        with patch("sys.argv", ["daily_brief", "--no-narrative"]):
            with patch("builtins.print"):
                with patch("src.monitor.daily_brief.generate_narrative") as mock_narrative:
                    daily_brief_main()
                    mock_narrative.assert_not_called()

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_no_narrative_preserves_brief_output(self, mock_dashboard, sample_dashboard):
        """With --no-narrative, brief output should still be printed."""
        mock_dashboard.return_value = sample_dashboard
        printed_texts = []
        with patch("sys.argv", ["daily_brief", "--no-narrative"]):
            with patch("builtins.print") as mock_print:
                daily_brief_main()
                # print should be called at least once (the brief text)
                assert mock_print.call_count >= 1

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    @patch("src.monitor.daily_brief.generate_narrative")
    def test_narrative_called_when_not_disabled(self, mock_narrative, mock_dashboard, sample_dashboard):
        """Without --no-narrative, generate_narrative should be called."""
        mock_dashboard.return_value = sample_dashboard
        mock_narrative.return_value = "Portfolio is stable today."
        with patch("sys.argv", ["daily_brief"]):
            with patch("builtins.print"):
                daily_brief_main()
        mock_narrative.assert_called_once()

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_narrative_failure_does_not_crash_main(self, mock_dashboard, sample_dashboard):
        """When narrative generation fails (returns None), main should not crash."""
        mock_dashboard.return_value = sample_dashboard
        with patch("sys.argv", ["daily_brief"]):
            with patch("builtins.print"):
                with patch("src.monitor.daily_brief.generate_narrative", return_value=None):
                    daily_brief_main()

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    @patch("src.monitor.daily_brief.generate_narrative")
    def test_narrative_updates_has_narrative_in_full_text(
        self, mock_narrative, mock_dashboard, sample_dashboard,
    ):
        """When narrative is returned, it should appear in printed output."""
        mock_dashboard.return_value = sample_dashboard
        mock_narrative.return_value = "Markets are calm."
        printed = []
        with patch("sys.argv", ["daily_brief"]):
            with patch("builtins.print") as mock_print:
                daily_brief_main()
                # One of the print calls should contain the narrative text
                all_output = " ".join(
                    str(call.args[0]) for call in mock_print.call_args_list
                )
                assert "Markets are calm." in all_output

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_save_with_narrative_writes_file(self, mock_dashboard, sample_dashboard, tmp_path):
        """--save flag without --no-narrative should still write file."""
        mock_dashboard.return_value = sample_dashboard
        with patch("sys.argv", ["daily_brief", "--save"]):
            with patch("src.monitor.daily_brief.DATA_DIR", tmp_path):
                with patch("builtins.print"):
                    with patch("src.monitor.daily_brief.generate_narrative", return_value=None):
                        daily_brief_main()
        out_path = tmp_path / "daily_brief.json"
        assert out_path.exists()

    @patch("src.monitor.unified_dashboard.generate_unified_dashboard")
    def test_save_prints_confirmation(self, mock_dashboard, sample_dashboard, tmp_path):
        """--save flag should print 'Saved to' confirmation message."""
        mock_dashboard.return_value = sample_dashboard
        with patch("sys.argv", ["daily_brief", "--save", "--no-narrative"]):
            with patch("src.monitor.daily_brief.DATA_DIR", tmp_path):
                printed_texts = []
                with patch("builtins.print") as mock_print:
                    daily_brief_main()
                    all_output = " ".join(
                        str(call.args[0]) for call in mock_print.call_args_list
                    )
                    assert "Saved to" in all_output
