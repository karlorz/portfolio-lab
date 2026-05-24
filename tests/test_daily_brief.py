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
        mock_client.messages.create.side_effect = Exception("API error")

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
        mock_dashboard.side_effect = Exception("Dashboard error")
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
