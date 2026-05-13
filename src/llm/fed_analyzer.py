#!/usr/bin/env python3
"""
Portfolio-Lab v2.22: Fed Analyzer

FOMC statement and meeting minutes sentiment analysis with hawk-dove scoring.
Implements context-aware classification and Chair speech weighting.

Usage:
    from src.llm.fed_analyzer import FedAnalyzer

    analyzer = FedAnalyzer()
    result = analyzer.analyze_statement(statement_text, date="2026-05-07")
    result = analyzer.analyze_minutes(minutes_text, date="2026-04-30")

CLI:
    python -m src.llm.fed_analyzer statement --date 2026-05-07 --file statement.txt
    python -m src.llm.fed_analyzer minutes --date 2026-04-30 --file minutes.txt
    python -m src.llm.fed_analyzer mock-test
"""

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from src.llm.earnings_analyzer import MockSentimentAnalyzer

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class FedAnalysisResult:
    """Complete FOMC communication analysis."""
    document_type: str  # "statement" or "minutes"
    date: str  # ISO format date
    
    # Hawk-dove spectrum: -1.0 (dovish) to +1.0 (hawkish)
    hawk_dove_score: float
    confidence: float  # 0.0 to 1.0
    
    # Key indicators
    policy_stance: str  # dovish, neutral, hawkish
    forward_guidance: str  # explicit, implicit, vague
    
    # Component scores
    labor_market_sentiment: float  # -1 to 1
    inflation_sentiment: float  # -1 to 1
    growth_sentiment: float  # -1 to 1
    financial_conditions: float  # -1 to 1 (tightening/loosening)
    
    # Uncertainty measures
    uncertainty_level: str  # low, medium, high
    dissention_detected: bool
    
    # Context-aware interpretation (required but with default_factory for mutables)
    economic_context: dict = field(default_factory=dict)
    context_adjusted_score: float = 0.0  # Score adjusted for economic backdrop
    
    # Key quotes (mutable defaults)
    key_hawkish_quotes: list[str] = field(default_factory=list)
    key_dovish_quotes: list[str] = field(default_factory=list)
    dissention_details: list[str] = field(default_factory=list)
    
    # Metadata
    word_count: int = 0
    speaker_count: int = 0  # For minutes
    management_tone: str = "neutral"  # Legacy field for compatibility
    guidance_clarity: str = "clear"  # Legacy field
    chair_speaking_time_pct: float = 0.0  # For minutes
    meeting_date: Optional[str] = None
    
    # LLM usage
    model_used: str = ""
    cost_usd: float = 0.0
    latency_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RollingFedSentiment:
    """Rolling window aggregation of Fed sentiment."""
    window_days: int
    start_date: str
    end_date: str
    
    avg_hawk_dove_score: float
    trend: str  # strengthening, weakening, stable
    volatility: float  # standard deviation of scores
    
    policy_transitions: list[dict]  # detected regime changes
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Fed Statement Parser
# ---------------------------------------------------------------------------

class FOMCParser:
    """Parse FOMC statements and meeting minutes."""
    
    # Common FOMC phrases and their hawkish/dovish indicators
    HAWKISH_PHRASES = [
        "elevated inflation", "inflation remains elevated", "concern about inflation",
        "tightening", "restrictive", "higher for longer", "additional rate increases",
        "remain restrictive", "unacceptable inflation", "price stability",
        "cool the economy", "cooling demand", "reduce accommodation",
        "wage pressures", "labor market tightness", "too tight",
    ]
    
    DOVISH_PHRASES = [
        "patience", "data dependent", "cumulative tightening", "lags",
        "disinflation", "inflation moving down", "progress on inflation",
        "balanced risks", "symmetric", "maximum employment", "soft landing",
        "careful", "cautious", "gradual", "measured", "time to pause",
        "hold steady", "assess incoming data", "monitor developments",
        "downside risks", "support the economy", "below target",
    ]
    
    NEUTRAL_PHRASES = [
        "will continue to monitor", "assess appropriate policy",
        "determine the extent", "maintain the target range",
        "ongoing increases", "future adjustments", "depending on data",
    ]
    
    # Context-dependent phrases that need economic state interpretation
    CONTEXT_DEPENDENT = {
        "strong labor market": {
            "hot_economy": "hawkish",  # Fed worried about overheating
            "weak_economy": "dovish",  # Fed comfortable with strength
        },
        "solid growth": {
            "high_inflation": "hawkish",  # More fuel on fire
            "low_inflation": "dovish",    # Healthy expansion okay
        },
        "resilient economy": {
            "tight_policy": "hawkish",    # Can handle more tightening
            "loose_policy": "dovish",     # Good news, no change needed
        },
    }
    
    SPEAKER_PATTERNS = [
        r"Chair(?:man|woman)?\s+([A-Z][a-z]+)",
        r"Vice Chair(?:man|woman)?\s+([A-Z][a-z]+)",
        r"Governor\s+([A-Z][a-z]+)",
        r"President\s+([A-Z][a-z]+)",
    ]
    
    def __init__(self):
        self.hawkish_pattern = re.compile(
            r'\b(?:' + '|'.join(re.escape(p) for p in self.HAWKISH_PHRASES) + r')\b',
            re.IGNORECASE
        )
        self.dovish_pattern = re.compile(
            r'\b(?:' + '|'.join(re.escape(p) for p in self.DOVISH_PHRASES) + r')\b',
            re.IGNORECASE
        )
    
    def parse_statement(self, text: str) -> dict[str, Any]:
        """Parse FOMC statement."""
        cleaned = self._clean_text(text)
        
        return {
            "full_text": cleaned,
            "paragraphs": self._split_paragraphs(cleaned),
            "hawkish_matches": self._find_phrases(cleaned, self.HAWKISH_PHRASES),
            "dovish_matches": self._find_phrases(cleaned, self.DOVISH_PHRASES),
            "word_count": len(cleaned.split()),
            "is_statement": True,
        }
    
    def parse_minutes(self, text: str) -> dict[str, Any]:
        """Parse FOMC meeting minutes."""
        cleaned = self._clean_text(text)
        
        # Extract speaker turns
        speakers = self._extract_speakers_minutes(cleaned)
        
        # Segment by topic sections
        sections = self._segment_minutes(cleaned)
        
        return {
            "full_text": cleaned,
            "speakers": speakers,
            "sections": sections,
            "speaker_turns": self._extract_turns(cleaned),
            "hawkish_matches": self._find_phrases(cleaned, self.HAWKISH_PHRASES),
            "dovish_matches": self._find_phrases(cleaned, self.DOVISH_PHRASES),
            "word_count": len(cleaned.split()),
            "is_minutes": True,
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean FOMC text formatting."""
        # Remove page numbers, headers, footers
        text = re.sub(r'\[?\d{1,2}\]?\s*of\s*\d{1,2}', '', text)
        text = re.sub(r'FEDERAL RESERVE', '', text, flags=re.IGNORECASE)
        text = re.sub(r'FOMC', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        return [p.strip() for p in text.split('\n\n') if p.strip()]
    
    def _find_phrases(self, text: str, phrases: list[str]) -> list[tuple[str, int]]:
        """Find phrase matches with positions."""
        matches = []
        text_lower = text.lower()
        for phrase in phrases:
            for match in re.finditer(r'\b' + re.escape(phrase.lower()) + r'\b', text_lower):
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                matches.append((phrase, match.start(), context))
        return matches
    
    def _extract_speakers_minutes(self, text: str) -> list[dict]:
        """Extract speakers from minutes."""
        speakers = []
        seen = set()
        
        for pattern in self.SPEAKER_PATTERNS:
            matches = re.findall(pattern, text)
            for name in matches:
                if name not in seen:
                    seen.add(name)
                    # Determine role
                    role = "Participant"
                    if "Chair" in pattern:
                        role = "Chair"
                    elif "Vice" in pattern:
                        role = "Vice Chair"
                    elif "Governor" in pattern:
                        role = "Governor"
                    elif "President" in pattern:
                        role = "Reserve Bank President"
                    
                    speakers.append({
                        "name": name,
                        "role": role,
                        "is_chair": "Chair" in pattern and "Vice" not in pattern,
                    })
        
        return speakers
    
    def _segment_minutes(self, text: str) -> dict[str, str]:
        """Segment minutes by standard sections."""
        # Standard FOMC minutes sections
        section_patterns = {
            "staff_review": r"(?:Staff Review|Review of Economic Conditions)",
            "policy_discussion": r"(?:Participants' Views|Monetary Policy Discussion)",
            "inflation_discussion": r"(?:Inflation|Price Stability)",
            "labor_discussion": r"(?:Employment|Labor Market)",
            "committee_decision": r"(?:Committee Policy Decision|Policy Decision)",
            "forward_guidance": r"(?:Forward Guidance|Policy Normalization)",
        }
        
        sections = {}
        lines = text.split('\n')
        
        for section_name, pattern in section_patterns.items():
            for i, line in enumerate(lines):
                if re.search(pattern, line, re.IGNORECASE):
                    # Find section content (until next section or end)
                    start = i
                    end = len(lines)
                    for other_pattern in section_patterns.values():
                        if other_pattern != pattern:
                            for j in range(i + 1, len(lines)):
                                if re.search(other_pattern, lines[j], re.IGNORECASE):
                                    end = min(end, j)
                                    break
                    
                    sections[section_name] = '\n'.join(lines[start:end])
                    break
        
        return sections
    
    def _extract_turns(self, text: str) -> list[dict]:
        """Extract speaker turns from minutes."""
        turns = []
        
        # Simple pattern: Name followed by dialogue
        turn_pattern = r'(?:^|\n)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)[\s]*[:\-\u2013\u2014][\s]*([^\n]+(?:\n(?![A-Z][a-z]+[:\-\u2013\u2014])[^\n]+)*)'
        
        for match in re.finditer(turn_pattern, text):
            speaker = match.group(1)
            dialogue = match.group(2).strip()
            
            # Weight chair's comments 2x
            is_chair = any(title in speaker for title in ["Powell", "Bernanke", "Yellen", "Greenspan"])
            
            turns.append({
                "speaker": speaker,
                "dialogue": dialogue,
                "word_count": len(dialogue.split()),
                "is_chair": is_chair,
                "weight": 2.0 if is_chair else 1.0,
            })
        
        return turns


# ---------------------------------------------------------------------------
# Fed Analyzer
# ---------------------------------------------------------------------------

class FedAnalyzer:
    """Analyze FOMC communications for monetary policy sentiment."""
    
    def __init__(self, sentiment_analyzer: Optional[Any] = None):
        self.parser = FOMCParser()
        self.history: list[FedAnalysisResult] = []
        
        # Use provided analyzer or mock
        if sentiment_analyzer:
            self.sentiment_analyzer = sentiment_analyzer
        else:
            self.sentiment_analyzer = MockSentimentAnalyzer()
    
    def analyze_statement(
        self,
        text: str,
        date: str,
        economic_context: Optional[dict] = None,
    ) -> FedAnalysisResult:
        """
        Analyze FOMC statement.
        
        Args:
            text: Statement text
            date: ISO format date (YYYY-MM-DD)
            economic_context: Optional economic data (GDP growth, unemployment, inflation)
        
        Returns:
            FedAnalysisResult with hawk-dove score and components
        """
        start_time = datetime.now()
        
        # Parse statement
        parsed = self.parser.parse_statement(text)
        
        # Calculate base hawk-dove score from phrase counts
        base_score = self._calculate_base_score(parsed)
        
        # Apply context adjustment
        context = economic_context or self._infer_economic_context(date)
        adjusted_score = self._apply_context_adjustment(base_score, context, parsed)
        
        # Detect uncertainty and dissention (more relevant in minutes, but check statement too)
        uncertainty = self._detect_uncertainty(parsed)
        
        # Classify policy stance
        stance = self._classify_stance(adjusted_score)
        guidance = self._classify_guidance(parsed)
        
        # Extract component sentiments
        components = self._analyze_components(parsed, context)
        
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        result = FedAnalysisResult(
            document_type="statement",
            date=date,
            hawk_dove_score=adjusted_score,
            confidence=self._calculate_confidence(parsed, uncertainty),
            policy_stance=stance,
            forward_guidance=guidance,
            labor_market_sentiment=components["labor"],
            inflation_sentiment=components["inflation"],
            growth_sentiment=components["growth"],
            financial_conditions=components["financial"],
            uncertainty_level=uncertainty["level"],
            dissention_detected=False,  # Rare in statements
            dissention_details=[],
            economic_context=context,
            context_adjusted_score=adjusted_score,
            key_hawkish_quotes=self._extract_key_quotes(parsed["hawkish_matches"]),
            key_dovish_quotes=self._extract_key_quotes(parsed["dovish_matches"]),
            word_count=parsed["word_count"],
            speaker_count=1,  # Committee as entity
            meeting_date=date,
            chair_speaking_time_pct=100.0,  # Unified statement
            cost_usd=0.0,  # Mock analyzer
            latency_ms=elapsed_ms,
        )
        
        # Store in history
        self.history.append(result)
        
        return result
    
    def analyze_minutes(
        self,
        text: str,
        date: str,
        economic_context: Optional[dict] = None,
    ) -> FedAnalysisResult:
        """Analyze FOMC meeting minutes."""
        start_time = datetime.now()
        
        # Parse minutes
        parsed = self.parser.parse_minutes(text)
        
        # Calculate weighted score (Chair 2x weight)
        base_score = self._calculate_weighted_score(parsed)
        
        # Apply context
        context = economic_context or self._infer_economic_context(date)
        adjusted_score = self._apply_context_adjustment(base_score, context, parsed)
        
        # Detect dissention (key difference from statements)
        dissention = self._detect_dissention(parsed)
        uncertainty = self._detect_uncertainty(parsed)
        
        # Calculate chair speaking percentage
        chair_pct = self._calculate_chair_percentage(parsed.get("speaker_turns", []))
        
        stance = self._classify_stance(adjusted_score)
        guidance = self._classify_guidance(parsed)
        components = self._analyze_components(parsed, context)
        
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        result = FedAnalysisResult(
            document_type="minutes",
            date=date,
            hawk_dove_score=adjusted_score,
            confidence=self._calculate_confidence(parsed, uncertainty, dissention),
            policy_stance=stance,
            forward_guidance=guidance,
            labor_market_sentiment=components["labor"],
            inflation_sentiment=components["inflation"],
            growth_sentiment=components["growth"],
            financial_conditions=components["financial"],
            uncertainty_level=uncertainty["level"],
            dissention_detected=dissention["detected"],
            dissention_details=dissention["details"],
            economic_context=context,
            context_adjusted_score=adjusted_score,
            key_hawkish_quotes=self._extract_key_quotes(parsed["hawkish_matches"]),
            key_dovish_quotes=self._extract_key_quotes(parsed["dovish_matches"]),
            word_count=parsed["word_count"],
            speaker_count=len(parsed.get("speakers", [])),
            meeting_date=date,
            chair_speaking_time_pct=chair_pct,
            cost_usd=0.0,
            latency_ms=elapsed_ms,
        )
        
        self.history.append(result)
        
        return result
    
    def get_rolling_sentiment(self, days: int = 90) -> RollingFedSentiment:
        """Calculate rolling sentiment over a window."""
        cutoff = datetime.now() - timedelta(days=days)
        recent = [
            r for r in self.history
            if datetime.fromisoformat(r.date) > cutoff
        ]
        
        if not recent:
            return RollingFedSentiment(
                window_days=days,
                start_date="",
                end_date="",
                avg_hawk_dove_score=0.0,
                trend="insufficient_data",
                volatility=0.0,
                policy_transitions=[],
            )
        
        scores = [r.hawk_dove_score for r in recent]
        avg_score = sum(scores) / len(scores)
        
        # Calculate volatility
        if len(scores) > 1:
            mean_sq = sum(s**2 for s in scores) / len(scores)
            volatility = (mean_sq - avg_score**2) ** 0.5
        else:
            volatility = 0.0
        
        # Detect trend
        if len(scores) >= 3:
            recent_avg = sum(scores[-3:]) / 3
            older_avg = sum(scores[:3]) / 3 if len(scores) >= 6 else scores[0]
            
            diff = recent_avg - older_avg
            if abs(diff) > 0.3:
                trend = "strengthening" if diff > 0 else "weakening"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        # Detect transitions
        transitions = self._detect_transitions(recent)
        
        return RollingFedSentiment(
            window_days=days,
            start_date=min(r.date for r in recent),
            end_date=max(r.date for r in recent),
            avg_hawk_dove_score=avg_score,
            trend=trend,
            volatility=volatility,
            policy_transitions=transitions,
        )
    
    def _calculate_base_score(self, parsed: dict) -> float:
        """Calculate base hawk-dove score from phrase matches."""
        hawkish_count = len(parsed.get("hawkish_matches", []))
        dovish_count = len(parsed.get("dovish_matches", []))
        
        # Normalize by word count
        words = parsed.get("word_count", 1)
        hawkish_density = hawkish_count / (words / 100)  # per 100 words
        dovish_density = dovish_count / (words / 100)
        
        # Score from -1 (dovish) to +1 (hawkish)
        if hawkish_density + dovish_density > 0:
            score = (hawkish_density - dovish_density) / (hawkish_density + dovish_density)
        else:
            score = 0.0
        
        # Scale to -1 to 1
        return max(-1.0, min(1.0, score))
    
    def _calculate_weighted_score(self, parsed: dict) -> float:
        """Calculate weighted score for minutes (Chair 2x)."""
        turns = parsed.get("speaker_turns", [])
        
        if not turns:
            return self._calculate_base_score(parsed)
        
        total_score = 0.0
        total_weight = 0.0
        
        for turn in turns:
            # Simple sentiment from dialogue
            dialogue = turn["dialogue"].lower()
            hawkish = sum(1 for p in self.parser.HAWKISH_PHRASES if p in dialogue)
            dovish = sum(1 for p in self.parser.DOVISH_PHRASES if p in dialogue)
            
            if hawkish + dovish > 0:
                turn_score = (hawkish - dovish) / (hawkish + dovish)
            else:
                turn_score = 0.0
            
            weight = turn.get("weight", 1.0)
            total_score += turn_score * weight
            total_weight += weight
        
        if total_weight > 0:
            return max(-1.0, min(1.0, total_score / total_weight))
        return 0.0
    
    def _apply_context_adjustment(
        self,
        base_score: float,
        context: dict,
        parsed: dict,
    ) -> float:
        """Apply context-aware adjustment to score."""
        adjusted = base_score
        
        # "Strong labor market" interpretation
        text_lower = parsed.get("full_text", "").lower()
        
        if "strong labor market" in text_lower or "tight labor market" in text_lower:
            inflation = context.get("inflation", 2.5)
            if inflation > 3.0:
                # Hot economy + high inflation = hawkish interpretation
                adjusted += 0.15
            elif inflation < 2.0:
                # Weak inflation context = dovish interpretation
                adjusted -= 0.10
        
        if "solid growth" in text_lower or "resilient" in text_lower:
            policy_rate = context.get("policy_rate", 5.0)
            if policy_rate > 4.5:
                # Already tight policy = hawkish (can handle it)
                adjusted += 0.10
            elif policy_rate < 2.0:
                # Loose policy + growth = dovish (no urgency to tighten)
                adjusted -= 0.05
        
        return max(-1.0, min(1.0, adjusted))
    
    def _infer_economic_context(self, date: str) -> dict:
        """Infer economic context from date (simplified - would use real data)."""
        # Simplified: return typical values
        # In production, would query economic data API
        return {
            "gdp_growth": 2.5,
            "unemployment": 3.8,
            "inflation": 2.5,  # Core PCE
            "policy_rate": 5.0,  # Fed funds
            "inferred": True,
        }
    
    def _detect_uncertainty(self, parsed: dict) -> dict:
        """Detect uncertainty level in communication."""
        text = parsed.get("full_text", "").lower()
        
        uncertainty_phrases = [
            "highly uncertain", "unusual uncertainty", "difficult to predict",
            "wide range of views", "considerable uncertainty", "significant uncertainty",
        ]
        
        count = sum(1 for p in uncertainty_phrases if p in text)
        
        if count >= 3:
            return {"level": "high", "count": count}
        elif count >= 1:
            return {"level": "medium", "count": count}
        else:
            return {"level": "low", "count": 0}
    
    def _detect_dissention(self, parsed: dict) -> dict:
        """Detect dissention/voting differences in minutes."""
        text = parsed.get("full_text", "").lower()
        
        dissention_indicators = [
            "dissent", "preferred", "would have", "voted against",
            "favored a different", "alternative proposal", "some participants",
            "a few participants", "several participants",
        ]
        
        detected = []
        for indicator in dissention_indicators:
            if indicator in text:
                # Extract surrounding context
                idx = text.find(indicator)
                start = max(0, idx - 100)
                end = min(len(text), idx + 200)
                context = text[start:end].strip()
                detected.append(context)
        
        return {
            "detected": len(detected) > 0,
            "details": detected[:3],  # Top 3 instances
        }
    
    def _classify_stance(self, score: float) -> str:
        """Classify policy stance from score."""
        if score > 0.4:
            return "hawkish"
        elif score < -0.4:
            return "dovish"
        else:
            return "neutral"
    
    def _classify_guidance(self, parsed: dict) -> str:
        """Classify forward guidance type."""
        text = parsed.get("full_text", "").lower()
        
        # Check for explicit guidance
        explicit_markers = [
            "anticipates", "expects", "projected", "dot plot",
            "median projection", "target range",
        ]
        vague_markers = [
            "will continue to monitor", "data dependent", "assess",
            "depending on", "evaluate", "review",
        ]
        
        explicit_count = sum(1 for m in explicit_markers if m in text)
        vague_count = sum(1 for m in vague_markers if m in text)
        
        if explicit_count > vague_count:
            return "explicit"
        elif vague_count > 0:
            return "implicit"
        else:
            return "vague"
    
    def _analyze_components(self, parsed: dict, context: dict) -> dict[str, float]:
        """Analyze sentiment by component area."""
        text = parsed.get("full_text", "").lower()
        
        # Labor market sentiment
        labor_hawkish = ["tight", "shortage", "wage pressure", "overheating"]
        labor_dovish = ["cooling", "softening", "balance", "slack"]
        labor_score = self._score_component(text, labor_hawkish, labor_dovish)
        
        # Inflation sentiment
        inflation_hawkish = ["elevated", "persistent", "sticky", "concern"]
        inflation_dovish = ["disinflation", "moving down", "transitory", "expect to decline"]
        inflation_score = self._score_component(text, inflation_hawkish, inflation_dovish)
        
        # Growth sentiment
        growth_hawkish = ["overheating", "unsustainable", "excess demand"]
        growth_dovish = ["below potential", "soft landing", "moderating"]
        growth_score = self._score_component(text, growth_hawkish, growth_dovish)
        
        # Financial conditions
        financial_hawkish = ["easing", "loosening", "accommodative"]
        financial_dovish = ["tightening", "restrictive", "credit contraction"]
        financial_score = self._score_component(text, financial_hawkish, financial_dovish)
        
        return {
            "labor": labor_score,
            "inflation": inflation_score,
            "growth": growth_score,
            "financial": financial_score,
        }
    
    def _score_component(self, text: str, hawkish_terms: list, dovish_terms: list) -> float:
        """Score a specific component."""
        hawk_count = sum(1 for t in hawkish_terms if t in text)
        dov_count = sum(1 for t in dovish_terms if t in text)
        
        if hawk_count + dov_count == 0:
            return 0.0
        
        score = (hawk_count - dov_count) / (hawk_count + dov_count)
        return max(-1.0, min(1.0, score))
    
    def _calculate_confidence(
        self,
        parsed: dict,
        uncertainty: dict,
        dissention: Optional[dict] = None,
    ) -> float:
        """Calculate confidence in analysis."""
        base_confidence = 0.7
        
        # Reduce confidence for high uncertainty
        if uncertainty.get("level") == "high":
            base_confidence -= 0.15
        elif uncertainty.get("level") == "medium":
            base_confidence -= 0.05
        
        # Reduce for dissention
        if dissention and dissention.get("detected"):
            base_confidence -= 0.10
        
        # Reduce for short documents
        if parsed.get("word_count", 0) < 200:
            base_confidence -= 0.10
        
        return max(0.3, min(0.95, base_confidence))
    
    def _calculate_chair_percentage(self, turns: list[dict]) -> float:
        """Calculate Chair's speaking time percentage."""
        if not turns:
            return 0.0
        
        chair_words = sum(t["word_count"] for t in turns if t.get("is_chair", False))
        total_words = sum(t["word_count"] for t in turns)
        
        if total_words > 0:
            return (chair_words / total_words) * 100
        return 0.0
    
    def _extract_key_quotes(self, matches: list[tuple]) -> list[str]:
        """Extract key quotes from phrase matches."""
        return [match[2] for match in matches[:5]]  # Top 5 with context
    
    def _detect_transitions(self, history: list[FedAnalysisResult]) -> list[dict]:
        """Detect policy regime transitions."""
        transitions = []
        
        if len(history) < 2:
            return transitions
        
        # Sort by date
        sorted_history = sorted(history, key=lambda r: r.date)
        
        for i in range(1, len(sorted_history)):
            prev = sorted_history[i - 1]
            curr = sorted_history[i]
            
            # Detect significant change
            score_change = abs(curr.hawk_dove_score - prev.hawk_dove_score)
            stance_change = curr.policy_stance != prev.policy_stance
            
            if score_change > 0.4 or stance_change:
                transitions.append({
                    "date": curr.date,
                    "previous_stance": prev.policy_stance,
                    "new_stance": curr.policy_stance,
                    "score_change": curr.hawk_dove_score - prev.hawk_dove_score,
                    "significance": "major" if score_change > 0.6 else "moderate",
                })
        
        return transitions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("""Usage:
  python -m src.llm.fed_analyzer statement --date 2026-05-07 --file statement.txt
  python -m src.llm.fed_analyzer minutes --date 2026-04-30 --file minutes.txt
  python -m src.llm.fed_analyzer rolling --days 90
  python -m src.llm.fed_analyzer mock-test
        """)
        sys.exit(1)
    
    command = sys.argv[1]
    analyzer = FedAnalyzer()
    
    if command == "mock-test":
        # Mock FOMC statement
        mock_statement = """
        Recent indicators suggest that economic activity has continued to expand at a 
        solid pace. Job gains have remained robust, and the unemployment rate has 
        remained low. Inflation has eased over the past year but remains elevated.
        
        The Committee seeks to achieve maximum employment and inflation at the rate 
        of 2 percent over the longer run. The Committee judges that the risks to 
        achieving its employment and inflation goals have moved toward better balance. 
        In support of these goals, the Committee decided to maintain the target range 
        for the federal funds rate at 5-1/4 to 5-1/2 percent.
        
        In considering any adjustments to the target range for the federal funds rate, 
        the Committee will carefully assess incoming data, the evolving outlook, and 
        the balance of risks. The Committee does not expect it will be appropriate 
        to reduce the target range until it has gained greater confidence that inflation 
        is moving sustainably toward 2 percent.
        
        In addition, the Committee will continue reducing its holdings of Treasury 
        securities and agency debt and agency mortgage-backed securities. The Committee 
        is strongly committed to returning inflation to its 2 percent objective.
        """
        
        result = analyzer.analyze_statement(mock_statement, "2026-05-07")
        print(json.dumps(result.to_dict(), indent=2))
        return
    
    if command == "statement":
        date = None
        filepath = None
        
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--date" and i + 1 < len(sys.argv):
                date = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--file" and i + 1 < len(sys.argv):
                filepath = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        
        if not date or not filepath:
            print("Error: --date and --file required")
            sys.exit(1)
        
        text = Path(filepath).read_text()
        result = analyzer.analyze_statement(text, date)
        print(json.dumps(result.to_dict(), indent=2))
        return
    
    if command == "minutes":
        date = None
        filepath = None
        
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--date" and i + 1 < len(sys.argv):
                date = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--file" and i + 1 < len(sys.argv):
                filepath = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        
        if not date or not filepath:
            print("Error: --date and --file required")
            sys.exit(1)
        
        text = Path(filepath).read_text()
        result = analyzer.analyze_minutes(text, date)
        print(json.dumps(result.to_dict(), indent=2))
        return
    
    if command == "rolling":
        days = 90
        
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--days" and i + 1 < len(sys.argv):
                days = int(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        
        result = analyzer.get_rolling_sentiment(days)
        print(json.dumps(result.to_dict(), indent=2))
        return
    
    print(f"Unknown command: {command}")
    sys.exit(1)


if __name__ == "__main__":
    main()
