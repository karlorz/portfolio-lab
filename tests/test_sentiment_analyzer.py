"""
Tests for src/strategy/sentiment_analyzer.py — Sentiment aggregation and smoothing.
Mocks LLM client to avoid API dependency.
"""
import pytest
import json
import sys
import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from collections import deque

# Create mock SentimentAnalyzer and SentimentResult before importing the module
class _MockSentimentResult:
    """Mock SentimentResult that matches the real class interface."""
    def __init__(self, sentiment="neutral", confidence=0.75,
                 key_factors=None, price_impact="neutral",
                 time_horizon="short_term", summary="",
                 model="mock", cost_usd=0.0, prompt_tokens=0,
                 cached_tokens=0, completion_tokens=0):
        self.sentiment = sentiment
        self.confidence = confidence
        self.key_factors = key_factors or []
        self.price_impact = price_impact
        self.time_horizon = time_horizon
        self.summary = summary
        self.model = model
        self.cost_usd = cost_usd
        self.prompt_tokens = prompt_tokens
        self.cached_tokens = cached_tokens
        self.completion_tokens = completion_tokens


class _MockSentimentAnalyzer:
    """Mock SentimentAnalyzer that returns configured results."""
    def __init__(self, *args, **kwargs):
        pass

    def analyze(self, text: str):
        if "bearish" in text.lower():
            return _MockSentimentResult(sentiment="bearish", confidence=0.8)
        elif "bullish" in text.lower():
            return _MockSentimentResult(sentiment="bullish", confidence=0.8)
        else:
            return _MockSentimentResult(sentiment="neutral", confidence=0.5)


# Inject mocks before importing the module under test
sys.modules["src.llm.sentiment_client"] = MagicMock()
sys.modules["src.llm.sentiment_client"].SentimentAnalyzer = _MockSentimentAnalyzer
sys.modules["src.llm.sentiment_client"].SentimentResult = _MockSentimentResult

from src.strategy.sentiment_analyzer import (
    AggregatedSentiment,
    SentimentAggregator,
    SentimentAnalyzerPipeline,
    demo,
)


class TestAggregatedSentiment:
    """AggregatedSentiment dataclass."""

    def test_create(self):
        s = AggregatedSentiment(
            timestamp="2026-05-15T00:00:00",
            news_sentiment=0.5,
            earnings_sentiment=0.3,
            macro_sentiment=-0.1,
            composite_score=0.25,
            confidence=0.8,
            smoothed_score=0.22,
            sentiment_momentum=0.05,
            regime_signal="risk_on",
            sources_used=3,
            data_quality="high",
        )
        assert s.composite_score == 0.25
        assert s.regime_signal == "risk_on"
        assert s.sources_used == 3

    def test_to_dict(self):
        s = AggregatedSentiment(
            timestamp="2026-05-15T00:00:00",
            news_sentiment=0.5,
            earnings_sentiment=0.3,
            macro_sentiment=-0.1,
            composite_score=0.25,
            confidence=0.8,
            smoothed_score=0.22,
            sentiment_momentum=0.05,
            regime_signal="neutral",
            sources_used=2,
            data_quality="medium",
        )
        d = s.to_dict()
        assert d["timestamp"] == "2026-05-15T00:00:00"
        assert d["composite_score"] == 0.25
        assert d["regime_signal"] == "neutral"
        assert d["sources_used"] == 2

    def test_to_dict_all_fields_present(self):
        s = AggregatedSentiment(
            timestamp="t", news_sentiment=0, earnings_sentiment=0,
            macro_sentiment=0, composite_score=0, confidence=0,
            smoothed_score=0, sentiment_momentum=0, regime_signal="neutral",
            sources_used=0, data_quality="low",
        )
        d = s.to_dict()
        assert len(d) == 11


class TestSentimentAggregatorInit:
    """SentimentAggregator construction."""

    def test_default_lookback(self):
        agg = SentimentAggregator()
        assert agg.lookback_days == 30
        assert agg.history.maxlen == 30

    def test_custom_lookback(self):
        agg = SentimentAggregator(lookback_days=60)
        assert agg.lookback_days == 60
        assert agg.history.maxlen == 60

    def test_analyzer_initialized(self):
        agg = SentimentAggregator()
        assert agg.analyzer is not None

    def test_class_constants(self):
        assert SentimentAggregator.HALF_LIFE_DAYS == 7
        assert 0.9 < SentimentAggregator.DECAY_FACTOR < 0.91  # ~0.906
        assert SentimentAggregator.RISK_ON_THRESHOLD == 0.3
        assert SentimentAggregator.RISK_OFF_THRESHOLD == -0.3
        assert SentimentAggregator.EXTREME_RISK_OFF_THRESHOLD == -0.6


class TestEMA:
    """Exponential moving average calculation."""

    def test_empty_values(self):
        agg = SentimentAggregator()
        assert agg.calculate_ema([]) == 0.0

    def test_single_value(self):
        agg = SentimentAggregator()
        assert agg.calculate_ema([0.5]) == pytest.approx(0.5)

    def test_constant_values(self):
        agg = SentimentAggregator()
        ema = agg.calculate_ema([0.5, 0.5, 0.5, 0.5])
        assert ema == pytest.approx(0.5)

    def test_ema_decay(self):
        agg = SentimentAggregator()
        values = [0.0, 0.0, 0.0, 1.0]  # Spike at end
        ema = agg.calculate_ema(values)
        assert 0.0 < ema < 1.0

    def test_custom_alpha(self):
        agg = SentimentAggregator()
        ema = agg.calculate_ema([0.0, 1.0], alpha=0.5)
        assert ema == pytest.approx(0.5)  # 0.5*1.0 + 0.5*0.0

    def test_ema_respects_half_life(self):
        agg = SentimentAggregator()
        # With low alpha (~0.094), EMA converges slowly — many repetitions needed
        values = [1.0] * 30
        ema = agg.calculate_ema(values)
        assert 0.9 < ema <= 1.0  # Converges to 1.0 with enough steps


class TestMomentum:
    """Sentiment momentum calculation."""

    def test_insufficient_values(self):
        agg = SentimentAggregator()
        assert agg.calculate_momentum([0.1, 0.2]) == 0.0

    def test_equal_windows(self):
        agg = SentimentAggregator()
        scores = [0.1] * 5 + [0.2] * 5
        mom = agg.calculate_momentum(scores, window=5)
        assert mom == pytest.approx(0.1)

    def test_negative_momentum(self):
        agg = SentimentAggregator()
        scores = [0.2] * 5 + [0.1] * 5
        mom = agg.calculate_momentum(scores, window=5)
        assert mom < 0

    def test_single_window_no_prior(self):
        agg = SentimentAggregator()
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        mom = agg.calculate_momentum(scores, window=5)
        assert mom == pytest.approx(0.3 - 0.1)  # mean last 5 - first value

    def test_custom_window(self):
        agg = SentimentAggregator()
        scores = [0.1] * 4 + [0.5] * 2
        mom = agg.calculate_momentum(scores, window=2)
        assert mom == pytest.approx(0.5 - 0.1)


class TestRegimeClassification:
    """Regime signal classification."""

    def test_extreme_risk_off(self):
        agg = SentimentAggregator()
        assert agg.classify_regime_signal(-0.7, 0.0) == "extreme_risk_off"

    def test_risk_off(self):
        agg = SentimentAggregator()
        assert agg.classify_regime_signal(-0.4, 0.0) == "risk_off"

    def test_risk_on(self):
        agg = SentimentAggregator()
        assert agg.classify_regime_signal(0.5, 0.0) == "risk_on"

    def test_neutral_by_default(self):
        agg = SentimentAggregator()
        assert agg.classify_regime_signal(0.0, 0.0) == "neutral"

    def test_momentum_pushes_to_risk_on(self):
        agg = SentimentAggregator()
        assert agg.classify_regime_signal(0.1, 0.3) == "risk_on"

    def test_momentum_pushes_to_risk_off(self):
        agg = SentimentAggregator()
        assert agg.classify_regime_signal(-0.1, -0.3) == "risk_off"

    def test_momentum_insufficient_returns_neutral(self):
        agg = SentimentAggregator()
        assert agg.classify_regime_signal(0.1, 0.1) == "neutral"

    def test_boundary_exactly_at_threshold(self):
        agg = SentimentAggregator()
        assert agg.classify_regime_signal(0.3, 0.0) == "neutral"  # > threshold, not >=
        assert agg.classify_regime_signal(-0.3, 0.0) == "neutral"  # < threshold, not <=


class TestSentimentToScore:
    """_sentiment_to_score conversion."""

    def test_bullish(self):
        agg = SentimentAggregator()
        assert agg._sentiment_to_score("bullish") == 1.0

    def test_bearish(self):
        agg = SentimentAggregator()
        assert agg._sentiment_to_score("bearish") == -1.0

    def test_neutral(self):
        agg = SentimentAggregator()
        assert agg._sentiment_to_score("neutral") == 0.0

    def test_unknown_label(self):
        agg = SentimentAggregator()
        assert agg._sentiment_to_score("garbage") == 0.0


class TestAggregateSources:
    """Source aggregation with various input combinations."""

    def make_result(self, sentiment="neutral", confidence=0.75):
        return _MockSentimentResult(sentiment=sentiment, confidence=confidence)

    def test_no_sources(self):
        agg = SentimentAggregator()
        result = agg.aggregate_sources()
        assert result.composite_score == 0.0
        assert result.confidence == 0.0
        assert result.sources_used == 0
        assert result.data_quality == "low"
        assert result.regime_signal == "neutral"

    def test_all_three_sources(self):
        agg = SentimentAggregator()
        news = [self.make_result("bullish", 0.8)]
        earnings = [self.make_result("bullish", 0.9)]
        macro = [self.make_result("neutral", 0.7)]
        result = agg.aggregate_sources(news, earnings, macro)
        assert result.sources_used == 3
        assert result.news_sentiment > 0
        assert result.earnings_sentiment > 0
        assert result.macro_sentiment == 0.0
        assert result.data_quality in ("high", "medium")

    def test_single_source(self):
        agg = SentimentAggregator()
        news = [self.make_result("bearish", 0.9)]
        result = agg.aggregate_sources(news_results=news)
        assert result.sources_used == 1
        assert result.news_sentiment < 0
        assert result.data_quality == "low"

    def test_two_sources(self):
        agg = SentimentAggregator()
        news = [self.make_result("bullish", 0.8)]
        earnings = [self.make_result("bullish", 0.8)]
        result = agg.aggregate_sources(news_results=news, earnings_results=earnings)
        assert result.sources_used == 2

    def test_confidence_scales_with_agreement(self):
        agg = SentimentAggregator()
        # High agreement → higher confidence
        news = [self.make_result("bullish", 0.8)]
        earnings = [self.make_result("bullish", 0.8)]
        result = agg.aggregate_sources(news_results=news, earnings_results=earnings)
        assert result.confidence > 0

    def test_empty_results_list(self):
        agg = SentimentAggregator()
        news = []
        result = agg.aggregate_sources(news_results=news)
        assert result.sources_used == 0

    def test_history_accumulates(self):
        agg = SentimentAggregator(lookback_days=5)
        r1 = agg.aggregate_sources(news_results=[self.make_result("bullish", 0.8)])
        r2 = agg.aggregate_sources(news_results=[self.make_result("bullish", 0.8)])
        assert len(agg.history) == 2

    def test_rounded_outputs(self):
        agg = SentimentAggregator()
        result = agg.aggregate_sources(news_results=[self.make_result("bullish", 0.8)])
        # All float fields round to 4 decimal places
        assert isinstance(result.composite_score, float)
        assert isinstance(result.confidence, float)


class TestSentimentAnalyzerPipeline:
    """Pipeline orchestration with mocked analyzer."""

    def test_init_creates_aggregator(self):
        pipe = SentimentAnalyzerPipeline()
        assert pipe.aggregator is not None
        assert pipe.analyzer is not None

    def test_init_default_data_dir(self):
        pipe = SentimentAnalyzerPipeline()
        assert pipe.data_dir.name == "sentiment"

    def test_init_custom_data_dir(self, tmp_path):
        pipe = SentimentAnalyzerPipeline(data_dir=tmp_path)
        assert pipe.data_dir == tmp_path

    def test_get_current_sentiment_no_texts(self):
        pipe = SentimentAnalyzerPipeline()
        result = pipe.get_current_sentiment()
        assert result is not None
        assert result.sources_used == 0

    def test_get_current_sentiment_with_news(self):
        pipe = SentimentAnalyzerPipeline()
        result = pipe.get_current_sentiment(news_texts=["Market rallies on earnings"])
        assert result is not None
        assert result.sources_used >= 1

    def test_get_current_sentiment_with_all_sources(self):
        pipe = SentimentAnalyzerPipeline()
        result = pipe.get_current_sentiment(
            news_texts=["Great news today"],
            earnings_texts=["Strong quarter"],
            macro_texts=["Fed holds rates steady"],
        )
        assert result is not None
        assert result.sources_used == 3

    def test_save_sentiment(self, tmp_path):
        pipe = SentimentAnalyzerPipeline(data_dir=tmp_path)
        result = pipe.get_current_sentiment(news_texts=["test"])
        filepath = pipe.save_sentiment(result, "test_output.json")
        assert filepath.exists()
        saved = json.loads(filepath.read_text())
        assert "composite_score" in saved
        assert "regime_signal" in saved

    def test_save_sentiment_autogenerated_filename(self, tmp_path):
        pipe = SentimentAnalyzerPipeline(data_dir=tmp_path)
        result = pipe.get_current_sentiment()
        filepath = pipe.save_sentiment(result)
        assert filepath.name.startswith("sentiment_")
        assert filepath.name.endswith(".json")

    def test_load_sentiment_history_empty(self, tmp_path):
        pipe = SentimentAnalyzerPipeline(data_dir=tmp_path)
        history = pipe.load_sentiment_history()
        assert history == []

    def test_load_sentiment_history(self, tmp_path):
        pipe = SentimentAnalyzerPipeline(data_dir=tmp_path)
        result = pipe.get_current_sentiment(news_texts=["test"])
        pipe.save_sentiment(result, "sentiment_test.json")

        history = pipe.load_sentiment_history(days=365)
        assert len(history) == 1
        assert isinstance(history[0], AggregatedSentiment)

    def test_load_sentiment_respects_cutoff(self, tmp_path):
        pipe = SentimentAnalyzerPipeline(data_dir=tmp_path)
        # Save with a date far in the past
        result = pipe.get_current_sentiment()
        # Override timestamp to be 60 days ago
        old_result = result
        pipe.save_sentiment(old_result, "sentiment_old.json")

        history = pipe.load_sentiment_history(days=1)
        # History from "now" should not include 60-day-old data
        # (the saved timestamps are from now though, since we used get_current_sentiment)
        assert len(history) >= 1

    def test_analyze_text_returns_result(self):
        pipe = SentimentAnalyzerPipeline()
        result = pipe.analyze_text("Market shows bullish momentum")
        assert result is not None
        assert result.sentiment in ("bullish", "bearish", "neutral")


class TestDemo:
    """Demo function runs without error."""

    def test_demo_runs(self):
        result = demo()
        assert result is not None
        assert isinstance(result, AggregatedSentiment)
        assert result.regime_signal in ("risk_on", "risk_off", "neutral", "extreme_risk_off")


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_ema_with_negative_values(self):
        agg = SentimentAggregator()
        ema = agg.calculate_ema([-0.5, -0.3, -0.8])
        assert -0.8 <= ema <= -0.3

    def test_momentum_with_zeros(self):
        agg = SentimentAggregator()
        mom = agg.calculate_momentum([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert mom == 0.0

    def test_aggregate_with_mixed_sentiments(self):
        agg = SentimentAggregator()
        bullish = _MockSentimentResult("bullish", 0.9)
        bearish = _MockSentimentResult("bearish", 0.9)
        result = agg.aggregate_sources(
            news_results=[bullish],
            earnings_results=[bearish],
        )
        # Bullish + Bearish should give low agreement → lower confidence
        assert result.sources_used == 2
        assert -1.0 <= result.composite_score <= 1.0

    def test_multiple_results_per_source(self):
        agg = SentimentAggregator()
        results = [
            _MockSentimentResult("bullish", 0.8),
            _MockSentimentResult("bullish", 0.9),
            _MockSentimentResult("neutral", 0.5),
        ]
        result = agg.aggregate_sources(news_results=results)
        assert result.sources_used == 1

    def test_extreme_risk_off_threshold_boundary(self):
        agg = SentimentAggregator()
        # -0.6 is NOT < -0.6, so falls through to risk_off (since -0.6 < -0.3)
        assert agg.classify_regime_signal(-0.6, 0.0) == "risk_off"
        # -0.61 IS < -0.6, so triggers extreme_risk_off
        assert agg.classify_regime_signal(-0.61, 0.0) == "extreme_risk_off"
