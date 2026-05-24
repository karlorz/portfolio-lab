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

    def test_record_new_model_creates_entry(self):
        ct = CostTracker(daily_budget_usd=5.0)
        ct.record("gpt-4o", 2000, 500, 0.01)
        assert ct.token_counts["gpt-4o"]["input"] == 2000
        assert ct.token_counts["gpt-4o"]["output"] == 500
        assert ct.token_counts["gpt-4o"]["cached"] == 0

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

    def test_to_dict_no_calls(self):
        ct = CostTracker(daily_budget_usd=10.0)
        d = ct.to_dict()
        assert d["call_count"] == 0
        assert d["total_cost_usd"] == 0.0
        assert d["budget_remaining_pct"] == 100.0
        assert d["token_counts"] == {}

    def test_budget_remaining_pct_exhausted(self):
        ct = CostTracker(daily_budget_usd=1.0)
        ct.record("gpt-4o-mini", 1000, 500, 1.0)
        assert ct.budget_remaining_pct() == pytest.approx(0.0)

    def test_budget_remaining_pct_full(self):
        ct = CostTracker(daily_budget_usd=10.0)
        assert ct.budget_remaining_pct() == pytest.approx(1.0)

    def test_save_daily_report(self, cost_tracker, tmp_path):
        report_path = cost_tracker.save_daily_report()
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert "date" in report
        assert "total_cost_usd" in report


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

    def test_unicode_characters(self):
        assert _estimate_tokens("hello world") == 2

    def test_single_char(self):
        assert _estimate_tokens("x") == 0

    def test_three_chars(self):
        assert _estimate_tokens("abc") == 0

    def test_four_chars(self):
        assert _estimate_tokens("abcd") == 1


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

    def test_to_dict_all_fields_complete(self):
        r = SentimentResult(
            sentiment="bearish", confidence=0.75, key_factors=["weak sales", "low margins"],
            price_impact="negative", time_horizon="medium_term", summary="Disappointing quarter.",
            model="claude-sonnet-4-5-20250929", cost_usd=0.015, prompt_tokens=5000,
            cached_tokens=4000, completion_tokens=300,
        )
        d = r.to_dict()
        assert d["sentiment"] == "bearish"
        assert d["confidence"] == 0.75
        assert d["key_factors"] == ["weak sales", "low margins"]
        assert d["price_impact"] == "negative"
        assert d["time_horizon"] == "medium_term"
        assert d["summary"] == "Disappointing quarter."
        assert d["model"] == "claude-sonnet-4-5-20250929"
        assert d["cost_usd"] == 0.015
        assert d["prompt_tokens"] == 5000
        assert d["cached_tokens"] == 4000
        assert d["completion_tokens"] == 300
        assert len(d) == 11

    def test_to_dict_empty_factors(self):
        r = SentimentResult(
            sentiment="neutral", confidence=0.5, key_factors=[],
            price_impact="neutral", time_horizon="short_term", summary="No clear signal.",
            model="gpt-4o-mini", cost_usd=0.0, prompt_tokens=0,
            cached_tokens=0, completion_tokens=0,
        )
        d = r.to_dict()
        assert d["key_factors"] == []
        assert d["confidence"] == 0.5

    def test_to_dict_zero_values(self):
        r = SentimentResult(
            sentiment="neutral", confidence=0.0, key_factors=[],
            price_impact="neutral", time_horizon="intraday", summary="",
            model="none", cost_usd=0.0, prompt_tokens=0,
            cached_tokens=0, completion_tokens=0,
        )
        d = r.to_dict()
        assert d["confidence"] == 0.0
        assert d["cost_usd"] == 0.0
        assert d["summary"] == ""


# ---------------------------------------------------------------------------
# LLMResponse dataclass
# ---------------------------------------------------------------------------

class TestLLMResponse:
    def test_all_fields(self):
        resp = _sc_mod.LLMResponse(
            content='{"sentiment": "bullish"}',
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            latency_ms=450,
        )
        assert resp.content == '{"sentiment": "bullish"}'
        assert resp.model == "gpt-4o-mini"
        assert resp.input_tokens == 100
        assert resp.output_tokens == 50
        assert resp.cost_usd == 0.001
        assert resp.latency_ms == 450

    def test_default_cached_tokens_zero(self):
        resp = _sc_mod.LLMResponse(
            content="test", model="gpt-4o-mini",
            input_tokens=10, output_tokens=5, cost_usd=0.0, latency_ms=100,
        )
        assert resp.cached_tokens == 0

    def test_default_parsed_json_none(self):
        resp = _sc_mod.LLMResponse(
            content="test", model="gpt-4o-mini",
            input_tokens=10, output_tokens=5, cost_usd=0.0, latency_ms=100,
        )
        assert resp.parsed_json is None

    def test_explicit_cached_tokens(self):
        resp = _sc_mod.LLMResponse(
            content="test", model="claude-sonnet-4-5-20250929",
            input_tokens=5000, output_tokens=300, cost_usd=0.015, latency_ms=1200,
            cached_tokens=4000,
        )
        assert resp.cached_tokens == 4000

    def test_explicit_parsed_json(self):
        parsed = {"sentiment": "bullish", "confidence": 0.9}
        resp = _sc_mod.LLMResponse(
            content=json.dumps(parsed), model="gpt-4o-mini",
            input_tokens=100, output_tokens=50, cost_usd=0.001, latency_ms=400,
            parsed_json=parsed,
        )
        assert resp.parsed_json == parsed


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

    def test_prompt_tokens_details_is_none(self):
        """prompt_tokens_details may be None (not just missing cached_tokens)."""
        mock_client = MagicMock()
        usage = MagicMock()
        usage.prompt_tokens = 200
        usage.prompt_tokens_details = None
        usage.completion_tokens = 80

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps(SAMPLE_SENTIMENT)
        resp.usage = usage
        mock_client.chat.completions.create.return_value = resp

        with patch.object(_sc_mod.openai, "OpenAI", return_value=mock_client):
            client = OpenAIGPT4oMiniClient(api_key="test")
            parsed, pt, ct, cpt = client._call_api("text", "sys", 1024, 0.1)

        assert parsed["sentiment"] == "bullish"
        assert pt == 200
        assert ct == 0
        assert cpt == 80

    def test_empty_text_returns_valid_json(self):
        mock_client = MagicMock()
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.prompt_tokens_details.cached_tokens = 0
        usage.completion_tokens = 5
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps({
            "sentiment": "neutral", "confidence": 0.5, "key_factors": [],
            "price_impact": "neutral", "time_horizon": "short_term", "summary": "",
        })
        resp.usage = usage
        mock_client.chat.completions.create.return_value = resp

        with patch.object(_sc_mod.openai, "OpenAI", return_value=mock_client):
            client = OpenAIGPT4oMiniClient(api_key="test")
            parsed, _, _, _ = client._call_api("", "sys", 1024, 0.1)

        assert parsed["sentiment"] == "neutral"


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

    def test_backtick_fence_without_json(self):
        """Claude may use plain ``` fences without the json tag."""
        mock_client = MagicMock()
        usage = MagicMock()
        usage.input_tokens = 100
        usage.cache_read_input_tokens = 0
        usage.output_tokens = 50

        content = MagicMock()
        content.text = f"```\n{json.dumps(SAMPLE_SENTIMENT)}\n```"

        resp = MagicMock()
        resp.content = [content]
        resp.usage = usage
        mock_client.messages.create.return_value = resp

        with patch.object(_sc_mod.anthropic, "Anthropic", return_value=mock_client):
            client = ClaudeSonnetClient(api_key="test")
            parsed, _, _, _ = client._call_api("test", "sys", 1024, 0.1)

        assert parsed["sentiment"] == "bullish"

    def test_cache_read_input_tokens_zero(self):
        """When cache_read_input_tokens attribute is missing entirely."""
        mock_client = MagicMock()
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 50
        del usage.cache_read_input_tokens

        content = MagicMock()
        content.text = json.dumps(SAMPLE_SENTIMENT)

        resp = MagicMock()
        resp.content = [content]
        resp.usage = usage
        mock_client.messages.create.return_value = resp

        with patch.object(_sc_mod.anthropic, "Anthropic", return_value=mock_client):
            client = ClaudeSonnetClient(api_key="test")
            _, _, ct, _ = client._call_api("test", "sys", 1024, 0.1)

        assert ct == 0


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

    def test_connection_error_retries(self):
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
            _openai.APIConnectionError(message="connection reset"),
            success,
        ]

        with patch.object(_sc_mod.openai, "OpenAI", return_value=mock_client), \
             patch("src.llm.sentiment_client.time.sleep"):
            client = OpenAIGPT4oMiniClient(api_key="test")
            result = client.analyze("text", cost_tracker=None)

        assert result.sentiment == "bullish"

    def test_server_error_5xx_retries(self):
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
            _openai.APIStatusError(
                message="server error", response=MagicMock(status_code=503, headers={}), body=None,
            ),
            success,
        ]

        with patch.object(_sc_mod.openai, "OpenAI", return_value=mock_client), \
             patch("src.llm.sentiment_client.time.sleep"):
            client = OpenAIGPT4oMiniClient(api_key="test")
            result = client.analyze("text", cost_tracker=None)

        assert result.sentiment == "bullish"

    def test_server_error_below_500_does_not_retry(self):
        _openai = _sc_mod.openai

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _openai.APIStatusError(
            message="bad request", response=MagicMock(status_code=400, headers={}), body=None,
        )

        with patch.object(_sc_mod.openai, "OpenAI", return_value=mock_client), \
             patch("src.llm.sentiment_client.time.sleep"):
            client = OpenAIGPT4oMiniClient(api_key="test")
            with pytest.raises(_openai.APIStatusError):
                client.analyze("text", cost_tracker=None)

    def test_retry_exhaustion_runtime_error(self):
        _openai = _sc_mod.openai

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _openai.RateLimitError(
            message="always rate limited",
            response=MagicMock(status_code=429, headers={}), body=None,
        )

        with patch.object(_sc_mod.openai, "OpenAI", return_value=mock_client), \
             patch("src.llm.sentiment_client.time.sleep"):
            client = OpenAIGPT4oMiniClient(api_key="test", max_retries=1)
            with pytest.raises(_openai.RateLimitError, match="always rate limited"):
                client.analyze("text", cost_tracker=None)

    def test_non_retryable_exception_propagates(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ValueError("unexpected error")

        with patch.object(_sc_mod.openai, "OpenAI", return_value=mock_client), \
             patch("src.llm.sentiment_client.time.sleep"):
            client = OpenAIGPT4oMiniClient(api_key="test")
            with pytest.raises(ValueError, match="unexpected error"):
                client.analyze("text", cost_tracker=None)

    def test_max_retries_zero_immediate_failure(self):
        _openai = _sc_mod.openai

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _openai.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}), body=None,
        )

        with patch.object(_sc_mod.openai, "OpenAI", return_value=mock_client), \
             patch("src.llm.sentiment_client.time.sleep"):
            client = OpenAIGPT4oMiniClient(api_key="test", max_retries=0)
            with pytest.raises(_openai.RateLimitError, match="rate limited"):
                client.analyze("text", cost_tracker=None)


# ---------------------------------------------------------------------------
# SentimentAnalyzer (mocked)
# ---------------------------------------------------------------------------

class TestSentimentAnalyzer:
    def _make_disabled(self):
        """Override env vars so SentimentAnalyzer sees no keys."""
        import os as _os
        with patch.dict(_os.environ, {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}):
            return SentimentAnalyzer()

    def test_disabled_returns_neutral(self):
        analyzer = self._make_disabled()
        result = analyzer.analyze("Some text")
        assert result.sentiment == "neutral"
        assert result.confidence == 0.0
        assert result.model == "none"
        assert result.cost_usd == 0.0
        assert result.key_factors == ["llm_disabled"]

    def test_disabled_cost_summary_raises(self):
        analyzer = self._make_disabled()
        with pytest.raises(AttributeError):
            analyzer.cost_summary()

    def test_disabled_analyze_batch_returns_neutral(self):
        analyzer = self._make_disabled()
        results = analyzer.analyze_batch(["text1", "text2"])
        assert len(results) == 2
        for r in results:
            assert r.sentiment == "neutral"

    def test_analyze_batch_empty(self):
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

        with patch.object(_sc_mod.openai, "OpenAI", return_value=mock_client), \
             patch.object(_sc_mod.anthropic, "Anthropic", return_value=MagicMock()):
            analyzer = SentimentAnalyzer()
            results = analyzer.analyze_batch([])

        assert results == []

    def test_analyze_batch_single(self):
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

        with patch.object(_sc_mod.openai, "OpenAI", return_value=mock_client), \
             patch.object(_sc_mod.anthropic, "Anthropic", return_value=MagicMock()):
            analyzer = SentimentAnalyzer()
            results = analyzer.analyze_batch(["AAPL earnings"])

        assert len(results) == 1
        assert results[0].sentiment == "bullish"
        assert analyzer.cost_summary()["call_count"] == 1

    def test_select_client_long_doc_type(self):
        a = SentimentAnalyzer.__new__(SentimentAnalyzer)
        a.gpt4o_mini = MagicMock()
        a.claude_sonnet = MagicMock()
        for doc_type in ["earnings_call", "filing_10k", "filing_10q"]:
            assert a._select_client("short text", doc_type) == a.claude_sonnet

    def test_select_client_long_text_boundary(self):
        a = SentimentAnalyzer.__new__(SentimentAnalyzer)
        a.gpt4o_mini = MagicMock()
        a.claude_sonnet = MagicMock()
        # _LONG_DOC_TOKENS = 4000, _estimate_tokens = len // 4
        # Exactly 4000 tokens needs 16000 chars (4000 * 4)
        # At <= 4000 tokens, text routes to gpt4o_mini
        # At > 4000 tokens, text routes to claude_sonnet
        long_text = "a" * 16004  # 4001 tokens -> > 4000 -> claude
        short_text = "a" * 16000  # 4000 tokens -> not > 4000 -> gpt
        assert a._select_client(long_text, "general") == a.claude_sonnet
        assert a._select_client(short_text, "general") == a.gpt4o_mini

    def test_force_model_claude_override(self):
        with patch.object(_sc_mod.openai, "OpenAI") as mock_oai, \
             patch.object(_sc_mod.anthropic, "Anthropic") as mock_ant:
            mock_oai.return_value = MagicMock()
            mock_ant.return_value = MagicMock()
            a = SentimentAnalyzer()
            a.claude_sonnet.analyze = MagicMock(return_value=MagicMock())
            a.analyze("short text", force_model="claude")
            a.claude_sonnet.analyze.assert_called_once()

    def test_analyze_with_cost_tracking(self, tmp_path):
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
        assert result.prompt_tokens == 200
        assert result.cached_tokens == 100
        summary = analyzer.cost_summary()
        assert summary["call_count"] == 1
        assert summary["total_cost_usd"] > 0


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

    def test_all_prices_positive(self):
        for model, prices in PRICING.items():
            assert prices["input"] > 0
            assert prices["cached_input"] > 0
            assert prices["output"] > 0


# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------

class TestConstants:
    def test_sentiment_schema_required_fields(self):
        assert "required" in _sc_mod.SENTIMENT_SCHEMA
        required = _sc_mod.SENTIMENT_SCHEMA["required"]
        assert "sentiment" in required
        assert "confidence" in required
        assert "key_factors" in required
        assert "price_impact" in required
        assert "time_horizon" in required
        assert "summary" in required

    def test_sentiment_schema_sentiment_enum(self):
        sentiment_prop = _sc_mod.SENTIMENT_SCHEMA["properties"]["sentiment"]
        assert sentiment_prop["enum"] == ["bullish", "bearish", "neutral"]

    def test_sentiment_schema_price_impact_enum(self):
        price_prop = _sc_mod.SENTIMENT_SCHEMA["properties"]["price_impact"]
        assert price_prop["enum"] == [
            "strong_positive", "positive", "neutral", "negative", "strong_negative",
        ]

    def test_sentiment_schema_confidence_bounds(self):
        conf_prop = _sc_mod.SENTIMENT_SCHEMA["properties"]["confidence"]
        assert conf_prop["minimum"] == 0.0
        assert conf_prop["maximum"] == 1.0

    def test_system_prompt_non_empty(self):
        assert len(_sc_mod.SYSTEM_PROMPT) > 50
        assert "financial analyst" in _sc_mod.SYSTEM_PROMPT.lower()

    def test_long_doc_tokens_positive(self):
        assert isinstance(_sc_mod._LONG_DOC_TOKENS, int)
        assert _sc_mod._LONG_DOC_TOKENS > 0

    def test_default_daily_budget_positive(self):
        assert _sc_mod.DEFAULT_DAILY_BUDGET > 0
