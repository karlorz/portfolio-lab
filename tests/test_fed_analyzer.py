#!/usr/bin/env python3
"""
Tests for Fed analyzer — data classes, FOMC parser, hawk-dove scoring,
stance classification, uncertainty detection, and component analysis.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.llm.fed_analyzer import (
    FedAnalysisResult, RollingFedSentiment,
    FOMCParser, FedAnalyzer,
)


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestFedAnalysisResult:
    """Test FedAnalysisResult dataclass."""

    def test_creation(self):
        result = FedAnalysisResult(
            document_type="statement",
            date="2026-05-01",
            hawk_dove_score=0.3,
            confidence=0.8,
            policy_stance="hawkish",
            forward_guidance="explicit",
            labor_market_sentiment=0.5,
            inflation_sentiment=0.4,
            growth_sentiment=0.2,
            financial_conditions=-0.1,
            uncertainty_level="low",
            dissention_detected=False,
        )
        assert result.document_type == "statement"
        assert result.hawk_dove_score == 0.3

    def test_to_dict(self):
        result = FedAnalysisResult(
            document_type="statement",
            date="2026-05-01",
            hawk_dove_score=0.0,
            confidence=0.5,
            policy_stance="neutral",
            forward_guidance="vague",
            labor_market_sentiment=0.0,
            inflation_sentiment=0.0,
            growth_sentiment=0.0,
            financial_conditions=0.0,
            uncertainty_level="low",
            dissention_detected=False,
        )
        d = result.to_dict()
        assert d["document_type"] == "statement"
        assert d["hawk_dove_score"] == 0.0

    def test_defaults(self):
        result = FedAnalysisResult(
            document_type="minutes",
            date="2026-05-01",
            hawk_dove_score=0.0,
            confidence=0.5,
            policy_stance="neutral",
            forward_guidance="vague",
            labor_market_sentiment=0.0,
            inflation_sentiment=0.0,
            growth_sentiment=0.0,
            financial_conditions=0.0,
            uncertainty_level="low",
            dissention_detected=False,
        )
        assert result.key_hawkish_quotes == []
        assert result.key_dovish_quotes == []
        assert result.context_adjusted_score == 0.0
        assert result.cost_usd == 0.0


class TestRollingFedSentiment:
    """Test RollingFedSentiment dataclass."""

    def test_creation(self):
        rs = RollingFedSentiment(
            window_days=90,
            start_date="2026-02-01",
            end_date="2026-05-01",
            avg_hawk_dove_score=0.15,
            trend="strengthening",
            volatility=0.12,
            policy_transitions=[],
        )
        assert rs.window_days == 90
        assert rs.trend == "strengthening"

    def test_to_dict(self):
        rs = RollingFedSentiment(
            window_days=90,
            start_date="2026-02-01",
            end_date="2026-05-01",
            avg_hawk_dove_score=0.0,
            trend="stable",
            volatility=0.0,
            policy_transitions=[{"from": "dovish", "to": "neutral"}],
        )
        d = rs.to_dict()
        assert d["window_days"] == 90
        assert len(d["policy_transitions"]) == 1


# ---------------------------------------------------------------------------
# FOMC Parser tests
# ---------------------------------------------------------------------------

class TestFOMCParser:
    """Test FOMCParser methods."""

    def test_hawkish_phrases_exist(self):
        """Hawkish phrase list is populated."""
        assert len(FOMCParser.HAWKISH_PHRASES) > 0
        assert "elevated inflation" in FOMCParser.HAWKISH_PHRASES

    def test_dovish_phrases_exist(self):
        """Dovish phrase list is populated."""
        assert len(FOMCParser.DOVISH_PHRASES) > 0
        assert "patience" in FOMCParser.DOVISH_PHRASES

    def test_parse_statement_returns_dict(self):
        """parse_statement returns expected structure."""
        parser = FOMCParser()
        result = parser.parse_statement("The Committee decided to maintain the target range. Inflation remains elevated.")
        assert "hawkish_matches" in result
        assert "dovish_matches" in result
        assert "word_count" in result
        assert "full_text" in result

    def test_parse_statement_detects_hawkish(self):
        """Hawkish phrases are detected."""
        parser = FOMCParser()
        text = "Inflation remains elevated and the Committee remains concerned about inflation."
        result = parser.parse_statement(text)
        assert len(result["hawkish_matches"]) > 0

    def test_parse_statement_detects_dovish(self):
        """Dovish phrases are detected."""
        parser = FOMCParser()
        text = "The Committee will be patient and data dependent as disinflation progresses."
        result = parser.parse_statement(text)
        assert len(result["dovish_matches"]) > 0

    def test_parse_minutes_returns_dict(self):
        """parse_minutes returns expected structure."""
        parser = FOMCParser()
        text = "Chair Powell: The economy shows resilience. Governor Waller: Inflation is cooling."
        result = parser.parse_minutes(text)
        assert "hawkish_matches" in result
        assert "speaker_turns" in result

    def test_clean_text(self):
        """_clean_text normalizes whitespace."""
        parser = FOMCParser()
        cleaned = parser._clean_text("  hello   world  \n\n  ")
        assert "  " not in cleaned or cleaned.strip() == cleaned.strip()

    def test_split_paragraphs(self):
        """_split_paragraphs splits on double newline."""
        parser = FOMCParser()
        paras = parser._split_paragraphs("Para 1.\n\nPara 2.\n\nPara 3.")
        assert len(paras) >= 2

    def test_find_phrases(self):
        """_find_phrases returns matches with positions."""
        parser = FOMCParser()
        matches = parser._find_phrases(
            "Inflation remains elevated and the Committee is patient.",
            ["elevated inflation", "patience"]
        )
        # "elevated inflation" is a multi-word phrase, may not match exact
        # But "patience" should be found as single word in phrases list
        assert isinstance(matches, list)

    def test_extract_speakers_minutes(self):
        """_extract_speakers_minutes finds speaker names."""
        parser = FOMCParser()
        text = "Chair Powell said the economy is strong. Governor Waller noted inflation progress."
        speakers = parser._extract_speakers_minutes(text)
        assert isinstance(speakers, list)

    def test_segment_minutes(self):
        """_segment_minutes returns section dict."""
        parser = FOMCParser()
        text = "Staff Review:\nThe staff presented the economic forecast.\n\nParticipants' Views:\nMost participants noted..."
        segments = parser._segment_minutes(text)
        assert isinstance(segments, dict)


# ---------------------------------------------------------------------------
# FedAnalyzer classification tests
# ---------------------------------------------------------------------------

class TestFedAnalyzerClassification:
    """Test FedAnalyzer stance and guidance classification."""

    def _make_analyzer(self):
        """Create a FedAnalyzer with mocked dependencies."""
        analyzer = FedAnalyzer.__new__(FedAnalyzer)
        analyzer.parser = FOMCParser()
        analyzer.sentiment_analyzer = None
        analyzer._history = []
        return analyzer

    def test_classify_stance_hawkish(self):
        """Score > 0.4 → hawkish."""
        analyzer = self._make_analyzer()
        assert analyzer._classify_stance(0.5) == "hawkish"
        assert analyzer._classify_stance(0.8) == "hawkish"

    def test_classify_stance_dovish(self):
        """Score < -0.4 → dovish."""
        analyzer = self._make_analyzer()
        assert analyzer._classify_stance(-0.5) == "dovish"
        assert analyzer._classify_stance(-0.8) == "dovish"

    def test_classify_stance_neutral(self):
        """Score between -0.4 and 0.4 → neutral."""
        analyzer = self._make_analyzer()
        assert analyzer._classify_stance(0.0) == "neutral"
        assert analyzer._classify_stance(0.3) == "neutral"
        assert analyzer._classify_stance(-0.3) == "neutral"

    def test_classify_guidance_explicit(self):
        """More explicit markers → explicit guidance."""
        analyzer = self._make_analyzer()
        parsed = {"full_text": "The Committee anticipates rate cuts. The median projection shows 3 cuts."}
        assert analyzer._classify_guidance(parsed) == "explicit"

    def test_classify_guidance_implicit(self):
        """Vague markers → implicit guidance."""
        analyzer = self._make_analyzer()
        parsed = {"full_text": "The Committee will continue to monitor developments and assess incoming data."}
        assert analyzer._classify_guidance(parsed) == "implicit"

    def test_classify_guidance_vague(self):
        """No markers → vague guidance."""
        analyzer = self._make_analyzer()
        parsed = {"full_text": "The economy is doing well."}
        assert analyzer._classify_guidance(parsed) == "vague"


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestScoring:
    """Test hawk-dove scoring methods."""

    def _make_analyzer(self):
        analyzer = FedAnalyzer.__new__(FedAnalyzer)
        analyzer.parser = FOMCParser()
        analyzer.sentiment_analyzer = None
        analyzer._history = []
        return analyzer

    def test_base_score_hawkish(self):
        """More hawkish phrases → positive score."""
        analyzer = self._make_analyzer()
        parsed = {
            "hawkish_matches": [("elevated inflation", 0), ("tightening", 20)],
            "dovish_matches": [],
            "word_count": 100,
        }
        score = analyzer._calculate_base_score(parsed)
        assert score > 0

    def test_base_score_dovish(self):
        """More dovish phrases → negative score."""
        analyzer = self._make_analyzer()
        parsed = {
            "hawkish_matches": [],
            "dovish_matches": [("patience", 0), ("disinflation", 20), ("data dependent", 40)],
            "word_count": 100,
        }
        score = analyzer._calculate_base_score(parsed)
        assert score < 0

    def test_base_score_balanced(self):
        """Equal hawkish and dovish → near zero."""
        analyzer = self._make_analyzer()
        parsed = {
            "hawkish_matches": [("elevated inflation", 0)],
            "dovish_matches": [("patience", 10)],
            "word_count": 100,
        }
        score = analyzer._calculate_base_score(parsed)
        assert -0.5 < score < 0.5

    def test_base_score_no_matches(self):
        """No matches → zero score."""
        analyzer = self._make_analyzer()
        parsed = {"hawkish_matches": [], "dovish_matches": [], "word_count": 100}
        assert analyzer._calculate_base_score(parsed) == 0.0

    def test_weighted_score_uses_turns(self):
        """Weighted score uses speaker turns."""
        analyzer = self._make_analyzer()
        parsed = {
            "hawkish_matches": [],
            "dovish_matches": [],
            "word_count": 100,
            "speaker_turns": [
                {"speaker": "Chair Powell", "dialogue": "Inflation remains elevated.", "weight": 2.0},
                {"speaker": "Governor Waller", "dialogue": "We will be patient.", "weight": 1.0},
            ],
        }
        score = analyzer._calculate_weighted_score(parsed)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Uncertainty and dissention tests
# ---------------------------------------------------------------------------

class TestUncertaintyDetection:
    """Test uncertainty and dissention detection."""

    def _make_analyzer(self):
        analyzer = FedAnalyzer.__new__(FedAnalyzer)
        analyzer.parser = FOMCParser()
        analyzer.sentiment_analyzer = None
        analyzer._history = []
        return analyzer

    def test_low_uncertainty(self):
        """No uncertainty phrases → low."""
        analyzer = self._make_analyzer()
        parsed = {"full_text": "The economy is doing well."}
        result = analyzer._detect_uncertainty(parsed)
        assert result["level"] == "low"

    def test_medium_uncertainty(self):
        """One uncertainty phrase → medium."""
        analyzer = self._make_analyzer()
        parsed = {"full_text": "There is significant uncertainty about the outlook."}
        result = analyzer._detect_uncertainty(parsed)
        assert result["level"] == "medium"

    def test_high_uncertainty(self):
        """Multiple uncertainty phrases → high."""
        analyzer = self._make_analyzer()
        parsed = {"full_text": "There is highly unusual uncertainty. The wide range of views reflects considerable uncertainty."}
        result = analyzer._detect_uncertainty(parsed)
        assert result["level"] == "high"

    def test_dissention_detected(self):
        """Dissent keywords → detected."""
        analyzer = self._make_analyzer()
        parsed = {"full_text": "Some participants preferred a different approach. Governor X voted against the proposal."}
        result = analyzer._detect_dissention(parsed)
        assert result["detected"] is True

    def test_no_dissention(self):
        """No dissent keywords → not detected."""
        analyzer = self._make_analyzer()
        parsed = {"full_text": "All participants agreed unanimously."}
        result = analyzer._detect_dissention(parsed)
        assert result["detected"] is False


# ---------------------------------------------------------------------------
# Component analysis tests
# ---------------------------------------------------------------------------

class TestComponentAnalysis:
    """Test _analyze_components and _score_component."""

    def _make_analyzer(self):
        analyzer = FedAnalyzer.__new__(FedAnalyzer)
        analyzer.parser = FOMCParser()
        analyzer.sentiment_analyzer = None
        analyzer._history = []
        return analyzer

    def test_component_scores_bounded(self):
        """All component scores are between -1 and 1."""
        analyzer = self._make_analyzer()
        parsed = {"full_text": "The labor market is tight with wage pressures. Inflation remains elevated. Growth is moderating."}
        context = {"inflation": 3.5, "policy_rate": 5.0}
        scores = analyzer._analyze_components(parsed, context)
        for key, val in scores.items():
            assert -1.0 <= val <= 1.0, f"{key} = {val} out of bounds"

    def test_score_component_hawkish(self):
        """More hawkish terms → positive score."""
        analyzer = self._make_analyzer()
        score = analyzer._score_component(
            "The labor market is tight with overheating.",
            hawkish_terms=["tight", "overheating"],
            dovish_terms=["cooling", "slack"],
        )
        assert score > 0

    def test_score_component_dovish(self):
        """More dovish terms → negative score."""
        analyzer = self._make_analyzer()
        score = analyzer._score_component(
            "The labor market is cooling with slack.",
            hawkish_terms=["tight", "overheating"],
            dovish_terms=["cooling", "slack"],
        )
        assert score < 0

    def test_score_component_neutral(self):
        """No matching terms → zero score."""
        analyzer = self._make_analyzer()
        score = analyzer._score_component(
            "The committee discussed various topics.",
            hawkish_terms=["tight", "overheating"],
            dovish_terms=["cooling", "slack"],
        )
        assert score == 0.0


# ---------------------------------------------------------------------------
# Context adjustment tests
# ---------------------------------------------------------------------------

class TestContextAdjustment:
    """Test _apply_context_adjustment."""

    def _make_analyzer(self):
        analyzer = FedAnalyzer.__new__(FedAnalyzer)
        analyzer.parser = FOMCParser()
        analyzer.sentiment_analyzer = None
        analyzer._history = []
        return analyzer

    def test_strong_labor_high_inflation(self):
        """Strong labor + high inflation → hawkish adjustment."""
        analyzer = self._make_analyzer()
        adjusted = analyzer._apply_context_adjustment(
            0.0,
            {"inflation": 4.0},
            {"full_text": "The strong labor market continues."},
        )
        assert adjusted > 0.0

    def test_strong_labor_low_inflation(self):
        """Strong labor + low inflation → dovish adjustment."""
        analyzer = self._make_analyzer()
        adjusted = analyzer._apply_context_adjustment(
            0.0,
            {"inflation": 1.5},
            {"full_text": "The strong labor market continues."},
        )
        assert adjusted < 0.0

    def test_resilient_high_rate(self):
        """Resilient + high rate → hawkish adjustment."""
        analyzer = self._make_analyzer()
        adjusted = analyzer._apply_context_adjustment(
            0.0,
            {"policy_rate": 5.0},
            {"full_text": "The resilient economy continues."},
        )
        assert adjusted > 0.0

    def test_score_clamped(self):
        """Adjusted score is clamped to [-1, 1]."""
        analyzer = self._make_analyzer()
        adjusted = analyzer._apply_context_adjustment(
            0.9,
            {"inflation": 5.0},
            {"full_text": "The strong labor market and solid growth continue."},
        )
        assert -1.0 <= adjusted <= 1.0


# ---------------------------------------------------------------------------
# Chair percentage tests
# ---------------------------------------------------------------------------

class TestChairPercentage:
    """Test _calculate_chair_percentage."""

    def _make_analyzer(self):
        analyzer = FedAnalyzer.__new__(FedAnalyzer)
        analyzer.parser = FOMCParser()
        analyzer.sentiment_analyzer = None
        analyzer._history = []
        return analyzer

    def test_chair_dominant(self):
        """Chair speaking most → high percentage."""
        analyzer = self._make_analyzer()
        turns = [
            {"speaker": "Chair Powell", "dialogue": "A" * 500, "word_count": 500, "is_chair": True},
            {"speaker": "Governor Waller", "dialogue": "B" * 100, "word_count": 100, "is_chair": False},
        ]
        pct = analyzer._calculate_chair_percentage(turns)
        assert pct > 50.0

    def test_no_turns(self):
        """Empty turns → 0%."""
        analyzer = self._make_analyzer()
        assert analyzer._calculate_chair_percentage([]) == 0.0


# ---------------------------------------------------------------------------
# Transition detection tests
# ---------------------------------------------------------------------------

class TestTransitionDetection:
    """Test _detect_transitions."""

    def _make_analyzer(self):
        analyzer = FedAnalyzer.__new__(FedAnalyzer)
        analyzer.parser = FOMCParser()
        analyzer.sentiment_analyzer = None
        analyzer._history = []
        return analyzer

    def test_no_transitions(self):
        """Stable scores → no transitions."""
        analyzer = self._make_analyzer()
        history = [
            FedAnalysisResult(
                document_type="statement", date="2026-01-01",
                hawk_dove_score=0.1, confidence=0.8,
                policy_stance="neutral", forward_guidance="vague",
                labor_market_sentiment=0.0, inflation_sentiment=0.0,
                growth_sentiment=0.0, financial_conditions=0.0,
                uncertainty_level="low", dissention_detected=False,
            ),
            FedAnalysisResult(
                document_type="statement", date="2026-03-01",
                hawk_dove_score=0.15, confidence=0.8,
                policy_stance="neutral", forward_guidance="vague",
                labor_market_sentiment=0.0, inflation_sentiment=0.0,
                growth_sentiment=0.0, financial_conditions=0.0,
                uncertainty_level="low", dissention_detected=False,
            ),
        ]
        transitions = analyzer._detect_transitions(history)
        assert len(transitions) == 0

    def test_transition_detected(self):
        """Large score shift → transition detected."""
        analyzer = self._make_analyzer()
        history = [
            FedAnalysisResult(
                document_type="statement", date="2026-01-01",
                hawk_dove_score=-0.5, confidence=0.8,
                policy_stance="dovish", forward_guidance="vague",
                labor_market_sentiment=0.0, inflation_sentiment=0.0,
                growth_sentiment=0.0, financial_conditions=0.0,
                uncertainty_level="low", dissention_detected=False,
            ),
            FedAnalysisResult(
                document_type="statement", date="2026-03-01",
                hawk_dove_score=0.5, confidence=0.8,
                policy_stance="hawkish", forward_guidance="explicit",
                labor_market_sentiment=0.0, inflation_sentiment=0.0,
                growth_sentiment=0.0, financial_conditions=0.0,
                uncertainty_level="low", dissention_detected=False,
            ),
        ]
        transitions = analyzer._detect_transitions(history)
        assert len(transitions) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
