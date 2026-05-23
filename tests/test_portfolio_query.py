"""
Tests for v7.10 Natural Language Portfolio Query.
Covers: context building, prompt formatting, fallback behavior, response parsing.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


from src.chat.portfolio_query import (
    build_query_context,
    format_system_prompt,
    answer_query,
    fallback_answer,
    SUPPORTED_QUERY_TYPES,
)


@pytest.fixture
def sample_dashboard():
    return {
        "portfolio": {
            "available": True,
            "total_value": 250000,
            "positions": [
                {"symbol": "SPY", "weight": 46.0, "value": 115000},
                {"symbol": "GLD", "weight": 38.0, "value": 95000},
                {"symbol": "TLT", "weight": 16.0, "value": 40000},
            ],
        },
        "risk": {
            "available": True,
            "current_drawdown": -8.5,
            "var_95_daily": -1.2,
            "cvar_95_daily": -1.8,
            "volatility_annual": 11.5,
        },
        "overlays": {
            "collar": {"active": True},
            "crypto": {"active": False},
            "bond_duration": {"active": True},
            "_meta": {"active_count": 2, "total_count": 5},
        },
        "regime": {
            "available": True,
            "classifier": {"current_regime": "normal"},
        },
        "tca": {
            "available": True,
            "scorecard": {"avg_slippage_bps": 5.2, "avg_quality_score": 72, "total_orders": 3},
        },
        "attribution": {
            "available": True,
            "sources": [
                {"name": "TSFM Momentum", "total_return_bps": 12.5, "hit_rate": 0.65},
                {"name": "Risk Budget", "total_return_bps": -3.2, "hit_rate": 0.50},
            ],
        },
        "health": {"available": True, "status": "healthy", "alerts": []},
        "cron": {"available": True, "total": 8, "ok": 8, "errors": 0},
    }


class TestBuildQueryContext:
    def test_builds_context_with_all_sections(self, sample_dashboard):
        context = build_query_context(sample_dashboard)
        assert "Portfolio" in context
        assert "250,000" in context
        assert "SPY" in context

    def test_context_includes_risk_data(self, sample_dashboard):
        context = build_query_context(sample_dashboard)
        assert "DD" in context
        assert "8.5" in context

    def test_context_includes_overlay_status(self, sample_dashboard):
        context = build_query_context(sample_dashboard)
        assert "collar" in context.lower()
        assert "bond_duration" in context.lower()

    def test_context_handles_missing_sections(self):
        dashboard = {"portfolio": {"available": False}}
        context = build_query_context(dashboard)
        assert "not available" in context.lower()


class TestFormatSystemPrompt:
    def test_formats_prompt_with_context(self, sample_dashboard):
        context = build_query_context(sample_dashboard)
        prompt = format_system_prompt(context)
        assert "portfolio management assistant" in prompt.lower()
        assert "Portfolio" in prompt

    def test_prompt_includes_instruction(self):
        prompt = format_system_prompt("TEST CONTEXT")
        assert "concise" in prompt.lower()
        assert "TEST CONTEXT" in prompt


class TestAnswerQuery:
    @patch("src.chat.portfolio_query.ANTHROPIC_AVAILABLE", True)
    def test_returns_llm_answer_when_available(self, sample_dashboard):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Your equity exposure is 46% in SPY.")]
        mock_client.messages.create.return_value = mock_response

        with patch("src.chat.portfolio_query.Anthropic", return_value=mock_client):
            with patch("src.chat.portfolio_query.os.environ.get", return_value="fake-key"):
                result = answer_query("What is my equity exposure?", sample_dashboard)
                assert "equity" in result.lower()

    def test_returns_fallback_when_no_api_key(self, sample_dashboard):
        with patch("src.chat.portfolio_query.os.environ.get", return_value=None):
            result = answer_query("What is my equity exposure?", sample_dashboard)
            assert "SPY" in result

    def test_returns_fallback_on_error(self, sample_dashboard):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API down")

        with patch("src.chat.portfolio_query.ANTHROPIC_AVAILABLE", True):
            with patch("src.chat.portfolio_query.Anthropic", return_value=mock_client):
                with patch("src.chat.portfolio_query.os.environ.get", return_value="fake-key"):
                    result = answer_query("What is my equity exposure?", sample_dashboard)
                    assert len(result) > 0


class TestFallbackAnswer:
    def test_equity_exposure_query(self, sample_dashboard):
        answer = fallback_answer("what is my equity exposure", sample_dashboard)
        assert "SPY" in answer

    def test_drawdown_query(self, sample_dashboard):
        answer = fallback_answer("what is my current drawdown", sample_dashboard)
        assert "8.5" in answer

    def test_overlay_query(self, sample_dashboard):
        answer = fallback_answer("which overlays are active", sample_dashboard)
        assert "collar" in answer.lower()

    def test_slippage_query(self, sample_dashboard):
        answer = fallback_answer("what is my average slippage", sample_dashboard)
        assert "5.2" in answer

    def test_unknown_query(self, sample_dashboard):
        answer = fallback_answer("tell me a joke", sample_dashboard)
        assert "not sure" in answer.lower() or "try asking" in answer.lower()


class TestSupportedQueryTypes:
    def test_has_portfolio_queries(self):
        assert "equity exposure" in SUPPORTED_QUERY_TYPES["portfolio"]

    def test_has_risk_queries(self):
        assert any("drawdown" in q for q in SUPPORTED_QUERY_TYPES["risk"])

    def test_has_signal_queries(self):
        assert any("bearish" in q for q in SUPPORTED_QUERY_TYPES["signals"])

    def test_has_overlay_queries(self):
        assert any("overlay" in q for q in SUPPORTED_QUERY_TYPES["overlays"])

    def test_has_tca_queries(self):
        assert any("slippage" in q for q in SUPPORTED_QUERY_TYPES["costs"])
