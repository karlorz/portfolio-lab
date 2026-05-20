"""Tests for sentiment_client.py — all API calls are mocked."""

import importlib
import json
import os
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-openai")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-anthropic")

# Other test files (e.g. test_sentiment_analyzer.py) may replace
# sys.modules["src.llm.sentiment_client"] with a MagicMock during
# collection.  If that happened, restore the real module so our
# tests can import and patch it properly.
_SC_KEY = "src.llm.sentiment_client"
if _SC_KEY in sys.modules:
    _existing = sys.modules[_SC_KEY]
    # A real module has __spec__; a MagicMock does not.
    if not hasattr(_existing, "__spec__") or _existing.__spec__ is None:
        del sys.modules[_SC_KEY]
        # Clear the parent package so it re-discovers the real module
        sys.modules.pop("src.llm", None)
        importlib.invalidate_caches()

import src.llm.sentiment_client as _sc_mod

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
# Exception stubs for environments without openai/anthropic SDKs
# ---------------------------------------------------------------------------

class _StubOpenAIError(Exception):
    def __init__(self, message="", response=None, body=None):
        self.message = message
        self.response = response
        self.body = body
        super().__init__(message)

class _StubRateLimitError(_StubOpenAIError): pass
class _StubAuthenticationError(_StubOpenAIError): pass
class _StubAPIConnectionError(_StubOpenAIError): pass
class _StubAPIStatusError(_StubOpenAIError):
    def __init__(self, message="", response=None, body=None):
        self.status_code = getattr(response, "status_code", 0) if response else 0
        super().__init__(message, response, body)

class _StubAnthError(Exception):
    def __init__(self, message="", response=None, body=None):
        self.message = message
        self.response = response
        self.body = body
        super().__init__(message)

class _StubAnthRateLimitError(_StubAnthError): pass
class _StubAnthAuthenticationError(_StubAnthError): pass
class _StubAnthAPIConnectionError(_StubAnthError): pass
class _StubAnthAPIStatusError(_StubAnthError):
    def __init__(self, message="", response=None, body=None):
        self.status_code = getattr(response, "status_code", 0) if response else 0
        super().__init__(message)


def _make_openai_stub():
    """Create a stub openai module with exception classes for patching."""
    return MagicMock(
        OpenAI=MagicMock,
        RateLimitError=_StubRateLimitError,
        AuthenticationError=_StubAuthenticationError,
        APIConnectionError=_StubAPIConnectionError,
        APIStatusError=_StubAPIStatusError,
    )


def _make_anthropic_stub():
    """Create a stub anthropic module with exception classes for patching."""
    return MagicMock(
        Anthropic=MagicMock,
        RateLimitError=_StubAnthRateLimitError,
        AuthenticationError=_StubAnthAuthenticationError,
        APIConnectionError=_StubAnthAPIConnectionError,
        APIStatusError=_StubAnthAPIStatusError,
    )


@pytest.fixture(autouse=True)
def _inject_sdk_stubs():
    """Inject openai/anthropic stubs for the duration of each test.

    When the SDKs are not installed (safe mode), _sc_mod.openai and
    _sc_mod.anthropic are None, which breaks @patch() because None
    has no attributes.  We temporarily swap in stub modules that carry
    the exception classes the source code uses in isinstance() checks.
    """
    saved_openai = _sc_mod.openai
    saved_anthropic = _sc_mod.anthropic
    injected = False

    if _sc_mod.openai is None:
        _sc_mod.openai = _make_openai_stub()
        injected = True

    if _sc_mod.anthropic is None:
        _sc_mod.anthropic = _make_anthropic_stub()
        injected = True

    yield

    if injected:
        _sc_mod.openai = saved_openai
        _sc_mod.anthropic = saved_anthropic


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
        # Build a SentimentAnalyzer with mocked client constructors
        with patch.object(_sc_mod.openai, "OpenAI") as mock_oai, \
             patch.object(_sc_mod.anthropic, "Anthropic") as mock_ant:
            mock_oai.return_value = MagicMock()
            mock_ant.return_value = MagicMock()
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
    def test_call_api_parses_json(self):
        mock_client = MagicMock()
        usage = MagicMock()
        usage.prompt_tokens = 150
        usage.prompt_tokens_details.cached_tokens = 50
        usage.completion_tokens = 80

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps(SAMPLE_SENTIMENT)
        resp.usage = usage
        mock_client.chat.completions.create.return_value = resp

        with patch.object(_sc_mod.openai, "OpenAI", return_value=mock_client):
            client = OpenAIGPT4oMiniClient(api_key="test")
            parsed, pt, ct, cpt = client._call_api("AAPL beat earnings", "sys", 1024, 0.1)

        assert parsed["sentiment"] == "bullish"
        assert pt == 150
        assert ct == 50
        assert cpt == 80

    def test_uses_json_mode(self):
        mock_client = MagicMock()
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.prompt_tokens_details.cached_tokens = 0
        usage.completion_tokens = 50

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps(SAMPLE_SENTIMENT)
        resp.usage = usage
        mock_client.chat.completions.create.return_value = resp

        with patch.object(_sc_mod.openai, "OpenAI", return_value=mock_client):
            client = OpenAIGPT4oMiniClient(api_key="test")
            client._call_api("test", "sys", 1024, 0.1)

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# Claude client (mocked)
# ---------------------------------------------------------------------------

class TestClaudeClient:
    def test_call_api_parses_json(self):
        mock_client = MagicMock()
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

        with patch.object(_sc_mod.anthropic, "Anthropic", return_value=mock_client):
            client = ClaudeSonnetClient(api_key="test")
            parsed, pt, ct, cpt = client._call_api("10-K content", "sys", 4096, 0.1)

        assert parsed["sentiment"] == "bullish"
        assert pt == 5000
        assert ct == 4000
        assert cpt == 300

    def test_handles_markdown_wrapped_json(self):
        mock_client = MagicMock()
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

        with patch.object(_sc_mod.anthropic, "Anthropic", return_value=mock_client):
            client = ClaudeSonnetClient(api_key="test")
            parsed, _, _, _ = client._call_api("test", "sys", 1024, 0.1)

        assert parsed["sentiment"] == "bullish"

    def test_uses_cache_control(self):
        mock_client = MagicMock()
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

        with patch.object(_sc_mod.anthropic, "Anthropic", return_value=mock_client):
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
    def test_retries_on_rate_limit(self):
        _openai = _sc_mod.openai

        mock_client = MagicMock()
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.prompt_tokens_details.cached_tokens = 0
        usage.completion_tokens = 50

        success = MagicMock()
        success.choices = [MagicMock()]
        success.choices[0].message.content = json.dumps(SAMPLE_SENTIMENT)
        success.usage = usage

        mock_client.chat.completions.create.side_effect = [
            _openai.RateLimitError(message="rate limited", response=MagicMock(status_code=429, headers={}), body=None),
            _openai.RateLimitError(message="rate limited", response=MagicMock(status_code=429, headers={}), body=None),
            success,
        ]

        with patch.object(_sc_mod.openai, "OpenAI", return_value=mock_client), \
             patch("src.llm.sentiment_client.time.sleep"):
            client = OpenAIGPT4oMiniClient(api_key="test")
            result = client.analyze("AAPL earnings", cost_tracker=None)

        assert result.sentiment == "bullish"

    def test_raises_auth_error_immediately(self):
        _openai = _sc_mod.openai

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _openai.AuthenticationError(
            message="bad key", response=MagicMock(status_code=401, headers={}), body=None,
        )

        with patch.object(_sc_mod.openai, "OpenAI", return_value=mock_client):
            client = OpenAIGPT4oMiniClient(api_key="bad")
            with pytest.raises(_openai.AuthenticationError):
                client.analyze("test", cost_tracker=None)


# ---------------------------------------------------------------------------
# End-to-end integration (mocked)
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_analyze_end_to_end(self, tmp_path):
        mock_client = MagicMock()
        usage = MagicMock()
        usage.prompt_tokens = 200
        usage.prompt_tokens_details.cached_tokens = 100
        usage.completion_tokens = 80

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps(SAMPLE_SENTIMENT)
        resp.usage = usage
        mock_client.chat.completions.create.return_value = resp

        with patch.object(_sc_mod.openai, "OpenAI", return_value=mock_client), \
             patch.object(_sc_mod.anthropic, "Anthropic", return_value=MagicMock()):
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
