#!/usr/bin/env python3
"""
Portfolio-Lab v2.22: LLM Sentiment Client

Unified client for GPT-4o-mini (primary) and Claude Sonnet (complex docs).
Handles retries, cost tracking, prompt caching, and JSON-mode structured output.

Research basis: GPT-4o-mini achieves 76% accuracy at $0.15/$0.60 per 1M tokens
with prompt caching providing ~90% savings on repeated prefixes.

Usage:
    from src.llm.sentiment_client import SentimentAnalyzer

    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("AAPL reported record earnings...")

CLI:
    python -m src.llm.sentiment_client analyze "AAPL reported record Q3 earnings"
    python -m src.llm.sentiment_client analyze --model claude --file transcript.txt
    python -m src.llm.sentiment_client costs
"""

import json
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

COST_DIR = Path("~/projects/portfolio-lab/data/llm_costs").expanduser()

# ---------------------------------------------------------------------------
# Pricing per 1M tokens (input / cached_input / output)
# ---------------------------------------------------------------------------

PRICING = {
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "cached_input": 0.30, "output": 15.00},
}

DEFAULT_DAILY_BUDGET = 50.0  # USD

# Long document threshold — switch to Claude above this
_LONG_DOC_TOKENS = 4000  # ~16K chars


# ---------------------------------------------------------------------------
# Sentiment schema for JSON structured output
# ---------------------------------------------------------------------------

SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "key_factors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Top 3-5 factors driving the sentiment",
        },
        "price_impact": {
            "type": "string",
            "enum": ["strong_positive", "positive", "neutral", "negative", "strong_negative"],
        },
        "time_horizon": {"type": "string", "enum": ["intraday", "short_term", "medium_term", "long_term"]},
        "summary": {"type": "string", "description": "One-sentence summary"},
    },
    "required": ["sentiment", "confidence", "key_factors", "price_impact", "time_horizon", "summary"],
}

SYSTEM_PROMPT = """You are a senior financial analyst specializing in sentiment analysis.
Analyze the provided financial text and return a structured JSON sentiment assessment.

Guidelines:
- Focus on actionable market signals
- Consider both explicit statements and implicit tone
- Weight recent developments more heavily than historical context
- Account for sector-specific language and norms
- Confidence should reflect clarity of the signal, not magnitude of expected move"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Response from an LLM call with metadata."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    cached_tokens: int = 0
    parsed_json: Optional[dict[str, Any]] = None


@dataclass
class SentimentResult:
    """Parsed sentiment analysis result."""
    sentiment: str       # bullish | bearish | neutral
    confidence: float    # 0.0 - 1.0
    key_factors: list[str]
    price_impact: str    # strong_positive | positive | neutral | negative | strong_negative
    time_horizon: str    # intraday | short_term | medium_term | long_term
    summary: str
    model: str
    cost_usd: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "key_factors": self.key_factors,
            "price_impact": self.price_impact,
            "time_horizon": self.time_horizon,
            "summary": self.summary,
            "model": self.model,
            "cost_usd": self.cost_usd,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
        }


# ---------------------------------------------------------------------------
# Cost tracking with daily budget enforcement
# ---------------------------------------------------------------------------

class BudgetExceededError(Exception):
    pass


@dataclass
class CostTracker:
    """Tracks per-request costs with daily budget enforcement and file persistence."""
    daily_budget_usd: float = DEFAULT_DAILY_BUDGET
    total_cost_usd: float = 0.0
    call_count: int = 0
    token_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, model: str, input_tokens: int, output_tokens: int, cost: float, cached_tokens: int = 0):
        self.total_cost_usd += cost
        self.call_count += 1
        if model not in self.token_counts:
            self.token_counts[model] = {"input": 0, "output": 0, "cached": 0}
        self.token_counts[model]["input"] += input_tokens
        self.token_counts[model]["output"] += output_tokens
        self.token_counts[model]["cached"] += cached_tokens

    def check_budget(self, estimated_cost: float = 0.0) -> None:
        if self.total_cost_usd + estimated_cost > self.daily_budget_usd:
            raise BudgetExceededError(
                f"Daily budget ${self.daily_budget_usd:.2f} exceeded. "
                f"Current: ${self.total_cost_usd:.4f}, estimated: ${estimated_cost:.4f}"
            )

    def within_budget(self) -> bool:
        return self.total_cost_usd < self.daily_budget_usd

    def budget_remaining_pct(self) -> float:
        return max(0.0, 1.0 - self.total_cost_usd / self.daily_budget_usd)

    def to_dict(self) -> dict[str, Any]:
        return {
            "daily_budget_usd": self.daily_budget_usd,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "call_count": self.call_count,
            "budget_remaining_pct": round(self.budget_remaining_pct() * 100, 1),
            "token_counts": self.token_counts,
        }

    def save_daily_report(self) -> Path:
        """Persist daily cost report to disk."""
        COST_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        report_path = COST_DIR / f"costs_{date_str}.json"
        report = {"date": date_str, "updated_at": datetime.now().isoformat(), **self.to_dict()}
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        return report_path


# ---------------------------------------------------------------------------
# Abstract base client
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract base with retry logic and exponential backoff."""

    def __init__(self, model: str, max_retries: int = 3, initial_backoff: float = 1.0, max_backoff: float = 60.0):
        self.model = model
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff

    def _backoff(self, attempt: int) -> float:
        import random
        delay = min(self.initial_backoff * (2 ** attempt), self.max_backoff)
        return delay * (0.5 + random.random())

    @abstractmethod
    def _call_api(self, text: str, system_prompt: str, max_tokens: int, temperature: float) -> tuple[dict, int, int, int]:
        """Returns (parsed_json, prompt_tokens, cached_tokens, completion_tokens)."""
        ...

    def analyze(
        self,
        text: str,
        document_type: str = "general",
        system_prompt: str = SYSTEM_PROMPT,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        cost_tracker: Optional[CostTracker] = None,
    ) -> SentimentResult:
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                parsed, prompt_tok, cached_tok, comp_tok = self._call_api(text, system_prompt, max_tokens, temperature)
                cost = self._compute_cost(prompt_tok, cached_tok, comp_tok)

                if cost_tracker:
                    cost_tracker.check_budget(cost)
                    cost_tracker.record(self.model, prompt_tok, comp_tok, cost, cached_tok)

                return SentimentResult(
                    sentiment=parsed.get("sentiment", "neutral"),
                    confidence=float(parsed.get("confidence", 0.5)),
                    key_factors=parsed.get("key_factors", []),
                    price_impact=parsed.get("price_impact", "neutral"),
                    time_horizon=parsed.get("time_horizon", "short_term"),
                    summary=parsed.get("summary", ""),
                    model=self.model,
                    cost_usd=cost,
                    prompt_tokens=prompt_tok,
                    cached_tokens=cached_tok,
                    completion_tokens=comp_tok,
                )

            except (Exception,) as e:
                import openai, anthropic
                if isinstance(e, (openai.AuthenticationError, anthropic.AuthenticationError)):
                    raise
                if isinstance(e, (openai.RateLimitError, anthropic.RateLimitError,
                                  openai.APIConnectionError, anthropic.APIConnectionError)):
                    last_error = e
                    if attempt < self.max_retries:
                        wait = self._backoff(attempt)
                        print(f"[retry {attempt+1}/{self.max_retries}] {type(e).__name__} — waiting {wait:.1f}s", file=sys.stderr)
                        time.sleep(wait)
                        continue
                # Server errors: retry
                status = getattr(e, "status_code", 0)
                if isinstance(e, (openai.APIStatusError, anthropic.APIStatusError)) and status >= 500 and attempt < self.max_retries:
                    last_error = e
                    wait = self._backoff(attempt)
                    print(f"[retry {attempt+1}/{self.max_retries}] {status} — waiting {wait:.1f}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                raise

        raise RuntimeError(f"All {self.max_retries} retries exhausted") from last_error

    def _compute_cost(self, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
        pricing = PRICING.get(self.model, PRICING["gpt-4o-mini"])
        regular = max(0, prompt_tokens - cached_tokens)
        input_cost = (regular * pricing["input"] + cached_tokens * pricing["cached_input"]) / 1_000_000
        output_cost = completion_tokens * pricing["output"] / 1_000_000
        return input_cost + output_cost


# ---------------------------------------------------------------------------
# OpenAI GPT-4o-mini client (prompt caching ~90% savings)
# ---------------------------------------------------------------------------

class OpenAIGPT4oMiniClient(LLMClient):
    """GPT-4o-mini for fast, cheap sentiment analysis.

    Prompt caching: OpenAI automatically caches repeated system prompts and
    document prefixes. Cached tokens cost ~50% less. To maximize hits:
    - Keep system prompt identical across requests
    - Batch similar document types together
    - Put stable context at the start of messages
    """

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        import openai as _openai
        super().__init__(model="gpt-4o-mini", max_retries=max_retries)
        self._openai = _openai
        self.client = _openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def _call_api(self, text: str, system_prompt: str, max_tokens: int, temperature: float):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze the sentiment of this financial text:\n\n{text}"},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)
        usage = response.usage
        cached = getattr(getattr(usage, "prompt_tokens_details", None) or {}, "cached_tokens", 0) or 0
        return parsed, usage.prompt_tokens, cached, usage.completion_tokens


# ---------------------------------------------------------------------------
# Claude Sonnet client (complex documents)
# ---------------------------------------------------------------------------

class ClaudeSonnetClient(LLMClient):
    """Claude Sonnet for complex financial document analysis.

    Best for: earnings call transcripts, 10-K/10-Q filings, multi-document
    comparative analysis, nuanced regulatory language.

    Supports prompt caching via cache_control on system prompt blocks.
    """

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        import anthropic as _anthropic
        super().__init__(model="claude-sonnet-4-5-20250929", max_retries=max_retries)
        self._anthropic = _anthropic
        self.client = _anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def _call_api(self, text: str, system_prompt: str, max_tokens: int, temperature: float):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral", "ttl": "5m"},
            }],
            messages=[{
                "role": "user",
                "content": (
                    "Analyze the sentiment of this financial text. "
                    "Return a JSON object with: sentiment (bullish/bearish/neutral), "
                    "confidence (0-1), key_factors (array), price_impact, time_horizon, summary.\n\n"
                    f"Text:\n\n{text}"
                ),
            }],
        )

        content = response.content[0].text
        # Claude sometimes wraps JSON in markdown fences
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)
        cached = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        return parsed, response.usage.input_tokens, cached, response.usage.output_tokens


# ---------------------------------------------------------------------------
# Unified SentimentAnalyzer (auto-routes by document type)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    return len(text) // 4


class SentimentAnalyzer:
    """Unified interface — routes to GPT-4o-mini for short docs, Claude for long/complex.

    Usage:
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("AAPL beat earnings by 15%")
        result = analyzer.analyze(long_transcript, document_type="earnings_call")
        result = analyzer.analyze(text, force_model="claude")
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        daily_budget_usd: float = DEFAULT_DAILY_BUDGET,
    ):
        self.gpt4o_mini = OpenAIGPT4oMiniClient(api_key=openai_api_key)
        self.claude_sonnet = ClaudeSonnetClient(api_key=anthropic_api_key)
        self.cost_tracker = CostTracker(daily_budget_usd=daily_budget_usd)

    _LONG_DOC_TYPES = {"earnings_call", "filing_10k", "filing_10q"}

    def _select_client(self, text: str, document_type: str) -> LLMClient:
        if document_type in self._LONG_DOC_TYPES:
            return self.claude_sonnet
        if _estimate_tokens(text) > _LONG_DOC_TOKENS:
            return self.claude_sonnet
        return self.gpt4o_mini

    def analyze(
        self,
        text: str,
        document_type: str = "general",
        force_model: Optional[str] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> SentimentResult:
        if force_model == "gpt4o_mini":
            client = self.gpt4o_mini
        elif force_model == "claude":
            client = self.claude_sonnet
        else:
            client = self._select_client(text, document_type)

        return client.analyze(text, document_type=document_type, system_prompt=system_prompt, cost_tracker=self.cost_tracker)

    def analyze_batch(self, texts: list[str], document_type: str = "general") -> list[SentimentResult]:
        return [self.analyze(t, document_type) for t in texts]

    def cost_summary(self) -> dict[str, Any]:
        return self.cost_tracker.to_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python -m src.llm.sentiment_client analyze "AAPL reported record Q3 earnings"')
        print("  python -m src.llm.sentiment_client analyze --file transcript.txt")
        print("  python -m src.llm.sentiment_client analyze --model claude --type earnings_call --file transcript.txt")
        print("  python -m src.llm.sentiment_client costs")
        sys.exit(1)

    command = sys.argv[1]

    if command == "costs":
        tracker = CostTracker()
        print(json.dumps(tracker.to_dict(), indent=2))
        return

    if command != "analyze":
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)

    args = sys.argv[2:]
    force_model = None
    document_type = "general"
    text = None
    filepath = None

    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            force_model = args[i + 1]; i += 2
        elif args[i] == "--type" and i + 1 < len(args):
            document_type = args[i + 1]; i += 2
        elif args[i] == "--file" and i + 1 < len(args):
            filepath = args[i + 1]; i += 2
        else:
            text = args[i]; i += 1

    if filepath:
        text = Path(filepath).read_text()
    elif text is None:
        text = sys.stdin.read()

    if not text:
        print("Error: no text provided", file=sys.stderr)
        sys.exit(1)

    analyzer = SentimentAnalyzer()
    result = analyzer.analyze(text, document_type=document_type, force_model=force_model)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
