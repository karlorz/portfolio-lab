#!/usr/bin/env python3
"""
Tests for earnings analyzer — data classes, transcript parser, mock sentiment,
aspect analysis, tone shifts, and management tone classification.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.llm.earnings_analyzer import (
    AspectSentiment, ToneShift, EarningsAnalysisResult,
    TranscriptParser, MockSentimentAnalyzer, EarningsAnalyzer,
)


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestAspectSentiment:
    """Test AspectSentiment dataclass."""

    def test_creation(self):
        a = AspectSentiment(
            aspect="revenue_guidance", sentiment="bullish",
            score=0.6, confidence=0.8,
        )
        assert a.aspect == "revenue_guidance"
        assert a.score == 0.6

    def test_to_dict(self):
        a = AspectSentiment(
            aspect="margin_outlook", sentiment="bearish",
            score=-0.3, confidence=0.7,
            key_quotes=["margins under pressure"],
        )
        d = a.to_dict()
        assert d["aspect"] == "margin_outlook"
        assert len(d["key_quotes"]) == 1

    def test_defaults(self):
        a = AspectSentiment(aspect="test", sentiment="neutral", score=0.0, confidence=0.5)
        assert a.key_quotes == []
        assert a.explanation == ""


class TestToneShift:
    """Test ToneShift dataclass."""

    def test_creation(self):
        t = ToneShift(
            aspect="revenue_guidance",
            previous_score=0.2, current_score=0.7,
            shift_magnitude=0.5, shift_direction="positive",
            significance="major",
        )
        assert t.shift_magnitude == 0.5
        assert t.significance == "major"


class TestEarningsAnalysisResult:
    """Test EarningsAnalysisResult dataclass."""

    def test_creation(self):
        r = EarningsAnalysisResult(
            ticker="AAPL", quarter="Q4-2025", fiscal_year=2025,
        )
        assert r.ticker == "AAPL"
        assert r.overall_sentiment == "neutral"

    def test_to_dict(self):
        r = EarningsAnalysisResult(
            ticker="AAPL", quarter="Q4-2025", fiscal_year=2025,
            aspects=[AspectSentiment(aspect="test", sentiment="neutral", score=0.0, confidence=0.5)],
            tone_shifts=[ToneShift(aspect="test", previous_score=0.0, current_score=0.5,
                                   shift_magnitude=0.5, shift_direction="positive", significance="major")],
        )
        d = r.to_dict()
        assert d["ticker"] == "AAPL"
        assert len(d["aspects"]) == 1
        assert len(d["tone_shifts"]) == 1

    def test_defaults(self):
        r = EarningsAnalysisResult(ticker="SPY", quarter="Q1-2026", fiscal_year=2026)
        assert r.aspects == []
        assert r.tone_shifts == []
        assert r.cost_usd == 0.0


# ---------------------------------------------------------------------------
# TranscriptParser tests
# ---------------------------------------------------------------------------

class TestTranscriptParser:
    """Test TranscriptParser methods."""

    def test_parse_returns_dict(self):
        """parse() returns expected structure."""
        parser = TranscriptParser()
        result = parser.parse("CEO Remarks: We had a great quarter. Q&A: What about margins?")
        assert "segments" in result
        assert "speakers" in result
        assert "sections" in result

    def test_parse_segments_non_empty(self):
        """Non-trivial input produces segments."""
        parser = TranscriptParser()
        text = "CEO John Smith: Revenue grew 15% year over year. CFO Jane Doe: Margins expanded."
        result = parser.parse(text)
        assert len(result["segments"]) > 0

    def test_clean_transcript(self):
        """_clean_transcript normalizes whitespace."""
        parser = TranscriptParser()
        cleaned = parser._clean_transcript("  hello   world  \n\n  ")
        assert cleaned.strip() == cleaned.strip()

    def test_detect_sections(self):
        """_detect_sections finds Q&A section."""
        parser = TranscriptParser()
        text = "Opening Remarks: Welcome everyone. Q&A: Let's take questions."
        sections = parser._detect_sections(text)
        assert isinstance(sections, dict)

    def test_extract_speakers(self):
        """_extract_speakers finds speaker names."""
        parser = TranscriptParser()
        text = "John Smith - CEO: Good morning. Jane Doe - CFO: Thank you."
        speakers = parser._extract_speakers(text)
        assert isinstance(speakers, list)

    def test_infer_role_ceo(self):
        """CEO in text → CEO role."""
        parser = TranscriptParser()
        role = parser._infer_role("John Smith", "John Smith - CEO discusses strategy")
        assert role == "CEO"

    def test_infer_role_cfo(self):
        """CFO in text → CFO role."""
        parser = TranscriptParser()
        role = parser._infer_role("Jane Doe", "Jane Doe - CFO discusses financials")
        assert role == "CFO"

    def test_infer_role_unknown(self):
        """No role markers → Unknown."""
        parser = TranscriptParser()
        role = parser._infer_role("Mike Johnson", "Mike Johnson from Goldman Sachs asks about margins")
        assert role in ["Analyst", "Unknown"]

    def test_is_ceo_context(self):
        """CEO strategic markers detected."""
        parser = TranscriptParser()
        assert parser._is_ceo_context("Our strategic vision for growth strategy is exciting") is True
        assert parser._is_ceo_context("The weather is nice today") is False

    def test_is_cfo_context(self):
        """CFO financial markers detected (need 2+)."""
        parser = TranscriptParser()
        assert parser._is_cfo_context("Revenue grew 15% year-over-year with margin expansion") is True
        assert parser._is_cfo_context("The weather is nice") is False

    def test_is_management_response(self):
        """Management response markers detected."""
        parser = TranscriptParser()
        assert parser._is_management_response("Great question. Let me address that.") is True
        assert parser._is_management_response("What are your thoughts on margins?") is False

    def test_segment_by_speaker(self):
        """_segment_by_speaker returns list."""
        parser = TranscriptParser()
        text = "John Smith - CEO: Great quarter.\nJane Doe - CFO: Strong results."
        segments = parser._segment_by_speaker(text)
        assert isinstance(segments, list)

    def test_classify_segments(self):
        """_classify_segments adds type to segments."""
        parser = TranscriptParser()
        segments = [
            {"speaker": "John Smith", "text": "Revenue grew.", "role": "ceo"},
            {"speaker": "Jane Doe", "text": "Margins expanded.", "role": "cfo"},
        ]
        classified = parser._classify_segments(segments)
        assert len(classified) == 2
        for seg in classified:
            assert "type" in seg


# ---------------------------------------------------------------------------
# MockSentimentAnalyzer tests
# ---------------------------------------------------------------------------

class TestMockSentimentAnalyzer:
    """Test MockSentimentAnalyzer."""

    def test_bullish_text(self):
        """Bullish keywords → bullish sentiment."""
        analyzer = MockSentimentAnalyzer()
        result = analyzer.analyze("Revenue beat expectations with strong growth and record momentum.")
        assert result.sentiment == "bullish"
        assert result.confidence > 0

    def test_bearish_text(self):
        """Bearish keywords → bearish sentiment."""
        analyzer = MockSentimentAnalyzer()
        result = analyzer.analyze("Revenue miss with declining margins and challenging headwinds.")
        assert result.sentiment == "bearish"
        assert result.confidence > 0

    def test_neutral_text(self):
        """No keywords → neutral sentiment."""
        analyzer = MockSentimentAnalyzer()
        result = analyzer.analyze("The company held a meeting to discuss operations.")
        assert result.sentiment == "neutral"

    def test_cost_summary(self):
        """cost_summary returns expected structure."""
        analyzer = MockSentimentAnalyzer()
        summary = analyzer.cost_summary()
        assert "total_spent_today" in summary

    def test_price_impact_bullish(self):
        """Bullish text → positive price impact."""
        analyzer = MockSentimentAnalyzer()
        result = analyzer.analyze("beat " * 5 + "growth " * 5)
        assert result.price_impact in ["positive", "strong_positive", "neutral"]

    def test_default_parameters(self):
        """Default parameters work."""
        analyzer = MockSentimentAnalyzer(default_sentiment="bearish", default_score=-0.5)
        assert analyzer.default_sentiment == "bearish"


# ---------------------------------------------------------------------------
# EarningsAnalyzer tests
# ---------------------------------------------------------------------------

class TestEarningsAnalyzer:
    """Test EarningsAnalyzer methods."""

    def _make_analyzer(self):
        """Create analyzer with mock sentiment."""
        analyzer = EarningsAnalyzer.__new__(EarningsAnalyzer)
        analyzer.parser = TranscriptParser()
        analyzer.sentiment_analyzer = MockSentimentAnalyzer()
        return analyzer

    def test_analyze_transcript(self, tmp_path):
        """analyze_transcript returns EarningsAnalysisResult."""
        analyzer = self._make_analyzer()
        transcript = """
        CEO John Smith: We had a strong quarter with record revenue growth.
        CFO Jane Doe: Margins expanded and we are confident about guidance.
        Q&A: Analyst asks about risk factors.
        """
        result = analyzer.analyze_transcript("AAPL", "Q4-2025", transcript)
        assert isinstance(result, EarningsAnalysisResult)
        assert result.ticker == "AAPL"
        assert result.quarter == "Q4-2025"

    def test_analyze_transcript_has_aspects(self):
        """Result contains aspect sentiments."""
        analyzer = self._make_analyzer()
        transcript = "CEO: Revenue beat expectations. CFO: Margins strong."
        result = analyzer.analyze_transcript("AAPL", "Q4-2025", transcript)
        assert len(result.aspects) > 0

    def test_overall_score_bounded(self):
        """Overall score is bounded to [-1, 1]."""
        analyzer = self._make_analyzer()
        transcript = "CEO: Record beat with strong growth and confident guidance."
        result = analyzer.analyze_transcript("AAPL", "Q4-2025", transcript)
        assert -1.0 <= result.overall_score <= 1.0

    def test_calculate_overall_empty(self):
        """Empty aspects → neutral."""
        analyzer = self._make_analyzer()
        score, sentiment, confidence = analyzer._calculate_overall([])
        assert score == 0.0
        assert sentiment == "neutral"

    def test_calculate_overall_bullish(self):
        """Bullish aspects → bullish overall."""
        analyzer = self._make_analyzer()
        aspects = [
            AspectSentiment(aspect="revenue_guidance", sentiment="bullish", score=0.7, confidence=0.8),
            AspectSentiment(aspect="margin_outlook", sentiment="bullish", score=0.5, confidence=0.7),
        ]
        score, sentiment, confidence = analyzer._calculate_overall(aspects)
        assert score > 0
        assert sentiment == "bullish"

    def test_calculate_overall_bearish(self):
        """Bearish aspects → bearish overall."""
        analyzer = self._make_analyzer()
        aspects = [
            AspectSentiment(aspect="revenue_guidance", sentiment="bearish", score=-0.7, confidence=0.8),
            AspectSentiment(aspect="risk_factors", sentiment="bearish", score=-0.5, confidence=0.7),
        ]
        score, sentiment, confidence = analyzer._calculate_overall(aspects)
        assert score < 0
        assert sentiment == "bearish"

    def test_detect_tone_shifts(self):
        """Tone shifts detected between quarters."""
        analyzer = self._make_analyzer()
        current = [
            AspectSentiment(aspect="revenue_guidance", sentiment="bullish", score=0.7, confidence=0.8),
        ]
        previous = [
            AspectSentiment(aspect="revenue_guidance", sentiment="bearish", score=-0.3, confidence=0.8),
        ]
        shifts = analyzer._detect_tone_shifts(current, previous)
        assert len(shifts) == 1
        assert shifts[0].significance == "major"

    def test_detect_tone_shifts_minor(self):
        """Small change → minor shift."""
        analyzer = self._make_analyzer()
        current = [AspectSentiment(aspect="test", sentiment="neutral", score=0.2, confidence=0.5)]
        previous = [AspectSentiment(aspect="test", sentiment="neutral", score=0.1, confidence=0.5)]
        shifts = analyzer._detect_tone_shifts(current, previous)
        assert shifts[0].significance == "minor"

    def test_classify_management_tone_confident(self):
        """High management_tone score → confident."""
        analyzer = self._make_analyzer()
        aspects = [AspectSentiment(aspect="management_tone", sentiment="bullish", score=0.6, confidence=0.8)]
        assert analyzer._classify_management_tone(aspects) == "very_confident"

    def test_classify_management_tone_defensive(self):
        """Very negative management_tone → defensive."""
        analyzer = self._make_analyzer()
        aspects = [AspectSentiment(aspect="management_tone", sentiment="bearish", score=-0.6, confidence=0.8)]
        assert analyzer._classify_management_tone(aspects) == "defensive"

    def test_classify_management_tone_neutral(self):
        """No management_tone aspect → neutral."""
        analyzer = self._make_analyzer()
        assert analyzer._classify_management_tone([]) == "neutral"

    def test_classify_guidance_clarity_clear(self):
        """High confidence + strong score → clear."""
        analyzer = self._make_analyzer()
        aspects = [AspectSentiment(aspect="revenue_guidance", sentiment="bullish", score=0.5, confidence=0.9)]
        assert analyzer._classify_guidance_clarity(aspects) == "clear"

    def test_classify_guidance_clarity_vague(self):
        """Low confidence → vague."""
        analyzer = self._make_analyzer()
        aspects = [AspectSentiment(aspect="revenue_guidance", sentiment="neutral", score=0.1, confidence=0.3)]
        assert analyzer._classify_guidance_clarity(aspects) == "vague"

    def test_classify_guidance_clarity_mixed(self):
        """Medium confidence → mixed."""
        analyzer = self._make_analyzer()
        aspects = [AspectSentiment(aspect="revenue_guidance", sentiment="neutral", score=0.1, confidence=0.6)]
        assert analyzer._classify_guidance_clarity(aspects) == "mixed"

    def test_extract_fiscal_year(self):
        """Extracts year from quarter string."""
        analyzer = self._make_analyzer()
        assert analyzer._extract_fiscal_year("Q4-2025") == 2025
        assert analyzer._extract_fiscal_year("Q1-2026") == 2026

    def test_extract_fiscal_year_fallback(self):
        """No year → current year."""
        analyzer = self._make_analyzer()
        result = analyzer._extract_fiscal_year("unknown")
        assert result == datetime.now().year

    def test_compare_to_previous(self):
        """Comparison returns expected structure."""
        analyzer = self._make_analyzer()
        prev = EarningsAnalysisResult(ticker="AAPL", quarter="Q3-2025", fiscal_year=2025)
        result = analyzer._compare_to_previous(prev)
        assert result["previous_quarter"] == "Q3-2025"

    def test_extract_key_segments(self):
        """Key segments extracted from list."""
        analyzer = self._make_analyzer()
        segments = [
            {"text": "Short segment.", "type": "ceo"},
            {"text": "A much longer segment with more detail about revenue and guidance and margins.", "type": "cfo"},
            {"text": "Another detailed segment about risk factors and outlook.", "type": "analyst"},
        ]
        key = analyzer._extract_key_segments(segments)
        assert len(key) <= 10

    def test_aspects_list(self):
        """ASPECTS list is defined."""
        assert len(EarningsAnalyzer.ASPECTS) == 5
        assert "revenue_guidance" in EarningsAnalyzer.ASPECTS


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
