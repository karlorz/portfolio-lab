"""Tests for sentiment_client.py — all API calls are mocked."""

import json
import os
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-openai")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-anthropic")

from src.llm.sentiment_client import (
    CostTracker,
    SentimentResult,
    SentimentAnalyzer,
    OpenAIGPT4oMiniClient,
    ClaudeSonnetClient,
    BudgetExceededError,
    LLMClient,
    _estimate_tokens,
    PRICING,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SENTIMENT = {
    "sentiment": "bullish",
    "confidence": 0.85,
    "key_factors": ["beat earnings", "strong guidance", "margin expansion"],
    "price_impact": "strong_positive",
    "time_horizon": "short_term",
    "summary": "AAPL reported strong Q3 earnings beating estimates by 15%.",
}


@pytest.fixture
def cost_tracker():
    return CostTracker(daily_budget_usd=5.0)


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class TestCostTracker:
    def test_record_accumulates(self, cost_tracker):
        cost_tracker.record("gpt-4o-mini", 1000, 200, 0.001, cached_tokens=500)
        assert cost_tracker.call_count == 1
        assert cost_tracker.total_cost_usd == pytest.approx(0.001)

    def test_within_budget(self, cost_tracker):
        assert cost_tracker.within_budget()
        cost_tracker.record("gpt-4o-mini", 100, 50, 4.90)
        assert cost_tracker.within_budget()
        cost_tracker.record("gpt-4o-mini", 100, 50, 0.20)
        assert not cost_tracker.within_budget()

    def test_check_budget_raises(self, cost_tracker):
        cost_tracker.record("gpt-4o-mini", 100, 50, 4.80)
        with pytest.raises(BudgetExceededError, match="Daily budget"):
            cost_tracker.check_budget(estimated_cost=0.50)

    def test_check_budget_passes(self, cost_tracker):
        cost_tracker.check_budget(estimated_cost=1.0)

    def test_budget_remaining_pct(self, cost_tracker):
        cost_tracker.record("gpt-4o-mini", 100, 50, 2.50)
        assert cost_tracker.budget_remaining_pct() == pytest.approx(0.5)

    def test_to_dict(self, cost_tracker):
        cost_tracker.record("gpt-4o-mini", 100, 50, 0.01, cached_tokens=30)
        d = cost_tracker.to_dict()
        assert d["call_count"] == 1
        assert "token_counts" in d

    def test_save_daily_report(self, cost_tracker, tmp_path):
        report_path = cost_tracker.save_daily_report()
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert "date" in report


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------

class TestCostComputation:
    def _client(self, model):
        c = OpenAIGPT4oMiniClient.__new__(OpenAIGPT4oMiniClient)
        c.model = model
        return c

    def test_gpt4o_mini_no_cache(self):
        c = self._client("gpt-4o-mini")
        cost = c._compute_cost(1000, 0, 500)
        expected = (1000 * 0.15 + 500 * 0.60) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_gpt4o_mini_with_cache(self):
        c = self._client("gpt-4o-mini")
        cost = c._compute_cost(1000, 800, 500)
        expected = (200 * 0.15 + 800 * 0.075 + 500 * 0.60) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_claude_with_cache(self):
        c = ClaudeSonnetClient.__new__(ClaudeSonnetClient)
        c.model = "claude-sonnet-4-5-20250929"
        cost = c._compute_cost(10000, 8000, 1000)
        expected = (2000 * 3.00 + 8000 * 0.30 + 1000 * 15.00) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_zero_tokens(self):
        c = self._client("gpt-4o-mini")
        assert c._compute_cost(0, 0, 0) == 0.0


# ---------------------------------------------------------------------------
# Document routing
# ---------------------------------------------------------------------------

class TestDocumentRouting:
    def _analyzer(self):
        a = SentimentAnalyzer.__new__(SentimentAnalyzer)
        a.gpt4o_mini = MagicMock()
        a.claude_sonnet = MagicMock()
        return a

    def test_short_text_routes_to_gpt(self):
        a = self._analyzer()
        assert a._select_client("Short headline", "headline") == a.gpt4o_mini

    def test_earnings_call_routes_to_claude(self):
        a = self._analyzer()
        assert a._select_client("text", "earnings_call") == a.claude_sonnet

    def test_10k_routes_to_claude(self):
        a = self._analyzer()
        assert a._select_client("text", "filing_10k") == a.claude_sonnet

    def test_long_text_routes_to_claude(self):
        a = self._analyzer()
        assert a._select_client("word " * 20000, "general") == a.claude_sonnet

    def test_force_model_overrides(self):
        a = SentimentAnalyzer()
        a.gpt4o_mini.analyze = MagicMock(return_value=MagicMock())
        a.analyze("text", force_model="gpt4o_mini")
        a.gpt4o_mini.analyze.assert_called_once()


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

class TestTokenEstimation:
    def test_short(self):
        assert _estimate_tokens("hello") == 1

    def test_long(self):
        assert _estimate_tokens("a" * 4000) == 1000

    def test_empty(self):
        assert _estimate_tokens("") == 0


# ---------------------------------------------------------------------------
# SentimentResult
# ---------------------------------------------------------------------------

class TestSentimentResult:
    def test_to_dict(self):
        r = SentimentResult(
            sentiment="bullish", confidence=0.9, key_factors=["earnings"],
            price_impact="positive", time_horizon="short_term", summary="Good.",
            model="gpt-4o-mini", cost_usd=0.001, prompt_tokens=100,
            cached_tokens=0, completion_tokens=50,
        )
        d = r.to_dict()
        assert d["sentiment"] == "bullish"
        assert d["confidence"] == 0.9
        assert d["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# OpenAI client (mocked)
# ---------------------------------------------------------------------------

class TestOpenAIClient:
    @patch("src.llm.sentiment_client.openai.OpenAI")
    def test_call_api_parses_json(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        usage = MagicMock()
        usage.prompt_tokens = 150
        usage.prompt_tokens_details.cached_tokens = 50
        usage.completion_tokens = 80

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps(SAMPLE_SENTIMENT)
        resp.usage = usage
        mock_client.chat.completions.create.return_value = resp

        client = OpenAIGPT4oMiniClient(api_key="test")
        parsed, pt, ct, cpt = client._call_api("AAPL beat earnings", "sys", 1024, 0.1)

        assert parsed["sentiment"] == "bullish"
        assert pt == 150
        assert ct == 50
        assert cpt == 80

    @patch("src.llm.sentiment_client.openai.OpenAI")
    def test_uses_json_mode(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.prompt_tokens_details.cached_tokens = 0
        usage.completion_tokens = 50

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps(SAMPLE_SENTIMENT)
        resp.usage = usage
        mock_client.chat.completions.create.return_value = resp

        client = OpenAIGPT4oMiniClient(api_key="test")
        client._call_api("test", "sys", 1024, 0.1)

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# Claude client (mocked)
# ---------------------------------------------------------------------------

class TestClaudeClient:
    @patch("src.llm.sentiment_client.anthropic.Anthropic")
    def test_call_api_parses_json(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        usage = MagicMock()
        usage.input_tokens = 5000
        usage.cache_read_input_tokens = 4000
        usage.output_tokens = 300

        content = MagicMock()
        content.text = json.dumps(SAMPLE_SENTIMENT)

        resp = MagicMock()
        resp.content = [content]
        resp.usage = usage
        mock_client.messages.create.return_value = resp

        client = ClaudeSonnetClient(api_key="test")
        parsed, pt, ct, cpt = client._call_api("10-K content", "sys", 4096, 0.1)

        assert parsed["sentiment"] == "bullish"
        assert pt == 5000
        assert ct == 4000
        assert cpt == 300

    @patch("src.llm.sentiment_client.anthropic.Anthropic")
    def test_handles_markdown_wrapped_json(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        usage = MagicMock()
        usage.input_tokens = 100
        usage.cache_read_input_tokens = 0
        usage.output_tokens = 50

        content = MagicMock()
        content.text = f"```json\n{json.dumps(SAMPLE_SENTIMENT)}\n```"

        resp = MagicMock()
        resp.content = [content]
        resp.usage = usage
        mock_client.messages.create.return_value = resp

        client = ClaudeSonnetClient(api_key="test")
        parsed, _, _, _ = client._call_api("test", "sys", 1024, 0.1)
        assert parsed["sentiment"] == "bullish"

    @patch("src.llm.sentiment_client.anthropic.Anthropic")
    def test_uses_cache_control(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        usage = MagicMock()
        usage.input_tokens = 100
        usage.cache_read_input_tokens = 0
        usage.output_tokens = 50

        content = MagicMock()
        content.text = json.dumps(SAMPLE_SENTIMENT)

        resp = MagicMock()
        resp.content = [content]
        resp.usage = usage
        mock_client.messages.create.return_value = resp

        client = ClaudeSonnetClient(api_key="test")
        client._call_api("test", "sys", 1024, 0.1)

        kwargs = mock_client.messages.create.call_args.kwargs
        system_blocks = kwargs["system"]
        assert isinstance(system_blocks, list)
        assert system_blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @patch("src.llm.sentiment_client.openai.OpenAI")
    @patch("src.llm.sentiment_client.time.sleep")
    def test_retries_on_rate_limit(self, mock_sleep, mock_cls):
        import openai

        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.prompt_tokens_details.cached_tokens = 0
        usage.completion_tokens = 50

        success = MagicMock()
        success.choices = [MagicMock()]
        success.choices[0].message.content = json.dumps(SAMPLE_SENTIMENT)
        success.usage = usage

        mock_client.chat.completions.create.side_effect = [
            openai.RateLimitError(message="rate limited", response=MagicMock(status_code=429, headers={}), body=None),
            openai.RateLimitError(message="rate limited", response=MagicMock(status_code=429, headers={}), body=None),
            success,
        ]

        client = OpenAIGPT4oMiniClient(api_key="test")
        result = client.analyze("AAPL earnings", cost_tracker=None)
        assert result.sentiment == "bullish"
        assert mock_sleep.call_count == 2

    @patch("src.llm.sentiment_client.openai.OpenAI")
    def test_raises_auth_error_immediately(self, mock_cls):
        import openai

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            message="bad key", response=MagicMock(status_code=401, headers={}), body=None,
        )

        client = OpenAIGPT4oMiniClient(api_key="bad")
        with pytest.raises(openai.AuthenticationError):
            client.analyze("test", cost_tracker=None)


# ---------------------------------------------------------------------------
# End-to-end integration (mocked)
# ---------------------------------------------------------------------------

class TestIntegration:
    @patch("src.llm.sentiment_client.openai.OpenAI")
    def test_analyze_end_to_end(self, mock_cls, tmp_path):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        usage = MagicMock()
        usage.prompt_tokens = 200
        usage.prompt_tokens_details.cached_tokens = 100
        usage.completion_tokens = 80

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps(SAMPLE_SENTIMENT)
        resp.usage = usage
        mock_client.chat.completions.create.return_value = resp

        analyzer = SentimentAnalyzer(daily_budget_usd=5.0)
        result = analyzer.analyze("AAPL beat earnings by 15%", document_type="headline")

        assert result.sentiment == "bullish"
        assert result.confidence == 0.85
        assert len(result.key_factors) == 3
        assert result.cost_usd > 0

        summary = analyzer.cost_summary()
        assert summary["call_count"] == 1


# ---------------------------------------------------------------------------
# Pricing sanity
# ---------------------------------------------------------------------------

class TestPricing:
    def test_all_models_have_pricing(self):
        for model in ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-5-20250929"]:
            assert model in PRICING

    def test_pricing_structure(self):
        for model, prices in PRICING.items():
            assert "input" in prices
            assert "cached_input" in prices
            assert "output" in prices
            assert prices["cached_input"] <= prices["input"]
