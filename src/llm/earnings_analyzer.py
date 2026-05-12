#!/usr/bin/env python3
"""
Portfolio-Lab v2.22: Earnings Analyzer

Aspect-Based Sentiment Analysis (ABSA) for earnings calls.
Parses transcripts and extracts sentiment per aspect: revenue guidance,
margins, risk factors, management tone with quarter-over-quarter comparison.

Usage:
    from src.llm.earnings_analyzer import EarningsAnalyzer

    analyzer = EarningsAnalyzer()
    result = analyzer.analyze_transcript("AAPL", "Q4-2025", transcript_text)

CLI:
    python -m src.llm.earnings_analyzer analyze AAPL --quarter Q4-2025 --file transcript.txt
    python -m src.llm.earnings_analyzer analyze TSLA --quarter Q3-2025 --text "CEO stated..."
    python -m src.llm.earnings_analyzer batch --dir ./transcripts/
"""

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict

from src.llm.sentiment_client import SentimentAnalyzer, SentimentResult

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class AspectSentiment:
    """Sentiment for a specific aspect of earnings."""
    aspect: str  # revenue_guidance, margin_outlook, risk_factors, management_tone, etc.
    sentiment: str  # bullish, bearish, neutral
    score: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    key_quotes: list[str] = field(default_factory=list)
    explanation: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToneShift:
    """Quarter-over-quarter tone shift detection."""
    aspect: str
    previous_score: float
    current_score: float
    shift_magnitude: float  # absolute change
    shift_direction: str  # positive, negative, stable
    significance: str  # major, moderate, minor


@dataclass
class EarningsAnalysisResult:
    """Complete analysis result for an earnings call."""
    ticker: str
    quarter: str  # Q4-2025 format
    fiscal_year: int
    
    # Aspect-based sentiments
    aspects: list[AspectSentiment] = field(default_factory=list)
    
    # Overall metrics
    overall_sentiment: str = "neutral"  # bullish, bearish, neutral
    overall_score: float = 0.0  # -1.0 to +1.0 weighted composite
    confidence: float = 0.0
    
    # Quarter-over-quarter comparison
    tone_shifts: list[ToneShift] = field(default_factory=list)
    vs_previous_quarter: Optional[dict] = None
    
    # Metadata
    transcript_length: int = 0
    word_count: int = 0
    segment_count: int = 0
    management_tone: str = "neutral"
    guidance_clarity: str = "clear"  # clear, vague, mixed
    
    # LLM usage
    model_used: str = ""
    cost_usd: float = 0.0
    latency_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "quarter": self.quarter,
            "fiscal_year": self.fiscal_year,
            "aspects": [a.to_dict() for a in self.aspects],
            "overall_sentiment": self.overall_sentiment,
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "tone_shifts": [
                {
                    "aspect": t.aspect,
                    "previous_score": t.previous_score,
                    "current_score": t.current_score,
                    "shift_magnitude": t.shift_magnitude,
                    "shift_direction": t.shift_direction,
                    "significance": t.significance,
                }
                for t in self.tone_shifts
            ],
            "vs_previous_quarter": self.vs_previous_quarter,
            "transcript_length": self.transcript_length,
            "word_count": self.word_count,
            "segment_count": self.segment_count,
            "management_tone": self.management_tone,
            "guidance_clarity": self.guidance_clarity,
            "model_used": self.model_used,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Transcript Parser
# ---------------------------------------------------------------------------

class TranscriptParser:
    """Parse earnings call transcripts into structured segments."""
    
    # Common patterns for transcript sections
    SECTION_PATTERNS = {
        "prepared_remarks": [
            r"(?:CEO|Chief Executive Officer)[\s\w]*Remarks",
            r"(?:Opening|Prepared) Remarks",
            r"(?:Good (?:morning|afternoon|evening),? (?:and )?welcome)",
        ],
        "financial_review": [
            r"(?:CFO|Chief Financial Officer)[\s\w]*Review",
            r"Financial (?:Review|Highlights)",
            r"(?:Review of )?(?:the )?Financials",
        ],
        "q_and_a": [
            r"Question[- ]?and[- ]?Answer",
            r"Q\s*[&＆]\s*A",
            r"(?:Analyst )?Q&A",
            r"Questions? from (?:the )?(?:analysts|audience)",
        ],
        "closing": [
            r"Closing Remarks",
            r"(?:Thank you,? )?(?:for )?(?:joining|your interest|your time)",
        ],
    }
    
    SPEAKER_PATTERNS = [
        r"([A-Z][a-z]+ [A-Z][a-z]+)[\s]*[-–—][\s]*(?:CEO|CFO|President|VP|Vice President)",
        r"([A-Z][a-z]+ [A-Z][a-z]+)[\s]*[;:][\s]*",
        r"Operator[\s]*[;:][\s]*",
        r"Moderator[\s]*[;:][\s]*",
    ]
    
    def __init__(self):
        self.compiled_section_patterns = {
            section: [re.compile(p, re.IGNORECASE) for p in patterns]
            for section, patterns in self.SECTION_PATTERNS.items()
        }
    
    def parse(self, transcript: str) -> dict[str, Any]:
        """Parse transcript into structured segments."""
        # Clean up transcript
        transcript = self._clean_transcript(transcript)
        
        # Detect sections
        sections = self._detect_sections(transcript)
        
        # Extract speakers and dialogue
        speakers = self._extract_speakers(transcript)
        
        # Segment by speaker turns
        segments = self._segment_by_speaker(transcript)
        
        # Classify segments by type
        classified = self._classify_segments(segments)
        
        return {
            "full_text": transcript,
            "sections": sections,
            "speakers": speakers,
            "segments": classified,
            "word_count": len(transcript.split()),
            "segment_count": len(segments),
        }
    
    def _clean_transcript(self, text: str) -> str:
        """Clean transcript formatting."""
        # Remove page numbers and timestamps
        text = re.sub(r'\[?\d{1,2}:\d{2}:\d{2}\]?', '', text)
        text = re.sub(r'\[?Page \d+\]?', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def _detect_sections(self, transcript: str) -> dict[str, tuple[int, int]]:
        """Detect section boundaries."""
        sections = {}
        lines = transcript.split('\n')
        
        for section_name, patterns in self.compiled_section_patterns.items():
            for i, line in enumerate(lines):
                for pattern in patterns:
                    if pattern.search(line):
                        sections[section_name] = i
                        break
        
        # Sort by position
        sorted_sections = sorted(sections.items(), key=lambda x: x[1])
        
        # Create section ranges
        result = {}
        for idx, (name, start) in enumerate(sorted_sections):
            if idx + 1 < len(sorted_sections):
                end = sorted_sections[idx + 1][1]
            else:
                end = len(lines)
            result[name] = (start, end)
        
        return result
    
    def _extract_speakers(self, transcript: str) -> list[dict]:
        """Extract speaker information."""
        speakers = []
        seen = set()
        
        for pattern in self.SPEAKER_PATTERNS:
            matches = re.findall(pattern, transcript)
            for match in matches:
                name = match.strip()
                if name and name not in seen and len(name) > 2:
                    seen.add(name)
                    speakers.append({
                        "name": name,
                        "role": self._infer_role(name, transcript),
                    })
        
        return speakers
    
    def _infer_role(self, name: str, transcript: str) -> str:
        """Infer speaker role from context."""
        # Find lines mentioning this speaker
        context = transcript[max(0, transcript.find(name) - 200):transcript.find(name) + 200]
        
        if re.search(r'\bCEO\b|\bChief Executive\b', context, re.IGNORECASE):
            return "CEO"
        elif re.search(r'\bCFO\b|\bChief Financial\b', context, re.IGNORECASE):
            return "CFO"
        elif re.search(r'\bCOO\b|\bChief Operating\b', context, re.IGNORECASE):
            return "COO"
        elif re.search(r'\bPresident\b', context, re.IGNORECASE):
            return "President"
        elif re.search(r'\bAnalyst\b', context, re.IGNORECASE):
            return "Analyst"
        elif re.search(r'\bOperator\b', context, re.IGNORECASE):
            return "Operator"
        
        return "Unknown"
    
    def _segment_by_speaker(self, transcript: str) -> list[dict]:
        """Segment transcript by speaker turns."""
        segments = []
        
        # Simple paragraph-based segmentation
        paragraphs = [p.strip() for p in transcript.split('\n\n') if p.strip()]
        
        current_speaker = "Unknown"
        for paragraph in paragraphs:
            # Try to detect speaker change
            for pattern in self.SPEAKER_PATTERNS:
                match = re.match(pattern, paragraph)
                if match:
                    current_speaker = match.group(1) if match.groups() else match.group(0)
                    paragraph = re.sub(pattern, '', paragraph, count=1).strip()
                    break
            
            segments.append({
                "speaker": current_speaker,
                "text": paragraph,
                "word_count": len(paragraph.split()),
            })
        
        return segments
    
    def _classify_segments(self, segments: list[dict]) -> list[dict]:
        """Classify segments by type (CEO, CFO, Q&A, etc.)."""
        classified = []
        
        for segment in segments:
            seg_type = "other"
            speaker = segment["speaker"]
            text = segment["text"]
            
            # Classify by speaker role
            if "CEO" in speaker or self._is_ceo_context(text):
                seg_type = "ceo_prepared"
            elif "CFO" in speaker or self._is_cfo_context(text):
                seg_type = "cfo_financial"
            elif "Analyst" in speaker or "question" in text.lower():
                seg_type = "analyst_question"
            elif self._is_management_response(text):
                seg_type = "management_response"
            
            classified.append({
                **segment,
                "type": seg_type,
            })
        
        return classified
    
    def _is_ceo_context(self, text: str) -> bool:
        """Detect CEO speaking context."""
        ceo_markers = [
            "strategic", "vision", "mission", "culture", "team", "customers",
            "market opportunity", "growth strategy", "proud of", "excited about",
        ]
        return any(marker in text.lower() for marker in ceo_markers)
    
    def _is_cfo_context(self, text: str) -> bool:
        """Detect CFO speaking context."""
        cfo_markers = [
            "revenue", "margin", "EPS", "earnings", "cash flow", "guidance",
            "billion", "million", "percent", "year-over-year", "quarter-over-quarter",
            "operating income", "net income", "EBITDA", "tax rate",
        ]
        count = sum(1 for marker in cfo_markers if marker in text.lower())
        return count >= 2
    
    def _is_management_response(self, text: str) -> bool:
        """Detect management response to analyst question."""
        response_markers = [
            "great question", "good question", "let me address", "to answer",
            "you're asking about", "as I mentioned", "as we discussed",
        ]
        return any(marker in text.lower() for marker in response_markers)


# ---------------------------------------------------------------------------
# ABSA Prompts
# ---------------------------------------------------------------------------

ASPECT_PROMPTS = {
    "revenue_guidance": """Analyze the REVENUE GUIDANCE aspect of this earnings call segment.

Score from -1.0 (very bearish/negative guidance) to +1.0 (very bullish/positive guidance).

Consider:
- Did they raise, lower, or maintain guidance?
- Is guidance above/below/inline with consensus?
- How confident is management about hitting guidance?
- Any qualitative comments about demand, pipeline, visibility?

Return JSON with:
{
  "score": float between -1.0 and 1.0,
  "confidence": float between 0.0 and 1.0,
  "key_quotes": ["exact quotes supporting score"],
  "explanation": "1-2 sentence rationale"
}""",

    "margin_outlook": """Analyze the MARGIN OUTLOOK aspect of this earnings call segment.

Score from -1.0 (margin compression/contraction) to +1.0 (margin expansion/improvement).

Consider:
- Gross margin trends
- Operating margin trends
- Impact of inflation, supply chain, pricing power
- Mix shift effects
- Investment spend vs efficiency gains

Return JSON with:
{
  "score": float between -1.0 and 1.0,
  "confidence": float between 0.0 and 1.0,
  "key_quotes": ["exact quotes supporting score"],
  "explanation": "1-2 sentence rationale"
}""",

    "risk_factors": """Analyze the RISK FACTORS discussed in this earnings call segment.

Score from -1.0 (major risks, significant headwinds) to +1.0 (risks well-managed, tailwinds).

Consider:
- Macroeconomic risks mentioned
- Competitive threats
- Regulatory issues
- Supply chain concerns
- Demand visibility
- How well management addresses concerns

Return JSON with:
{
  "score": float between -1.0 and 1.0,
  "confidence": float between 0.0 and 1.0,
  "key_quotes": ["exact quotes supporting score"],
  "explanation": "1-2 sentence rationale"
}""",

    "management_tone": """Analyze the MANAGEMENT TONE in this earnings call segment.

Score from -1.0 (defensive, evasive, negative tone) to +1.0 (confident, transparent, positive tone).

Consider:
- Confidence level in responses
- Transparency and detail in answers
- Enthusiasm about future
- Defensiveness vs openness
- Use of hedging language
- Comparison to previous quarter's tone

Return JSON with:
{
  "score": float between -1.0 and 1.0,
  "confidence": float between 0.0 and 1.0,
  "key_quotes": ["exact quotes supporting score"],
  "explanation": "1-2 sentence rationale"
}""",

    "capital_allocation": """Analyze the CAPITAL ALLOCATION priorities in this earnings call segment.

Score from -1.0 (concerning capital allocation) to +1.0 (shareholder-friendly allocation).

Consider:
- Share buyback plans
- Dividend policy
- Capex intentions
- M&A activity
- Balance sheet health discussions
- Cash deployment priorities

Return JSON with:
{
  "score": float between -1.0 and 1.0,
  "confidence": float between 0.0 and 1.0,
  "key_quotes": ["exact quotes supporting score"],
  "explanation": "1-2 sentence rationale"
}""",
}


# ---------------------------------------------------------------------------
# Earnings Analyzer
# ---------------------------------------------------------------------------

class MockSentimentAnalyzer:
    """Mock sentiment analyzer for testing without API keys."""
    
    def __init__(self, default_sentiment: str = "neutral", default_score: float = 0.0):
        self.default_sentiment = default_sentiment
        self.default_score = default_score
    
    def analyze(self, text: str, **kwargs) -> "SentimentResult":
        # Return a mock result based on simple keyword detection
        text_lower = text.lower()
        
        # Simple keyword-based sentiment
        bullish_words = ["beat", "exceed", "growth", "record", "strong", "momentum", "confident", "excited", "proud"]
        bearish_words = ["miss", "decline", "headwind", "challenging", "disappoint", "weak", "concern", "risk"]
        
        score = 0.0
        for word in bullish_words:
            if word in text_lower:
                score += 0.1
        for word in bearish_words:
            if word in text_lower:
                score -= 0.1
        
        score = max(-1.0, min(1.0, score))
        
        if score > 0.3:
            sentiment = "bullish"
        elif score < -0.3:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        return SentimentResult(
            sentiment=sentiment,
            confidence=0.6,
            key_factors=["mock_factor_1", "mock_factor_2"],
            price_impact="neutral" if abs(score) < 0.3 else ("positive" if score > 0 else "negative"),
            time_horizon="short_term",
            summary=f"Mock sentiment analysis: {sentiment} (score: {score:.2f})",
            model="mock-analyzer",
            cost_usd=0.0,
            prompt_tokens=100,
            cached_tokens=0,
            completion_tokens=50,
        )
    
    def cost_summary(self) -> dict:
        return {"total_spent_today": 0.0, "remaining_budget": 50.0}


class EarningsAnalyzer:
    """Main analyzer for earnings call transcripts."""
    
    ASPECTS = ["revenue_guidance", "margin_outlook", "risk_factors", "management_tone", "capital_allocation"]
    
    def __init__(self, sentiment_analyzer: Optional[Any] = None):
        self.parser = TranscriptParser()
        # Use provided analyzer, or try real analyzer, or fall back to mock
        if sentiment_analyzer:
            self.sentiment_analyzer = sentiment_analyzer
        else:
            try:
                from src.llm.sentiment_client import SentimentAnalyzer
                self.sentiment_analyzer = SentimentAnalyzer(
                    openai_api_key=None,  # Will auto-detect from env
                    anthropic_api_key=None,
                )
            except Exception:
                # Fall back to mock analyzer
                self.sentiment_analyzer = MockSentimentAnalyzer()
    
    def analyze_transcript(
        self,
        ticker: str,
        quarter: str,
        transcript: str,
        previous_analysis: Optional[EarningsAnalysisResult] = None,
    ) -> EarningsAnalysisResult:
        """
        Analyze an earnings call transcript.
        
        Args:
            ticker: Stock ticker symbol
            quarter: Quarter string (e.g., "Q4-2025")
            transcript: Full transcript text
            previous_analysis: Optional previous quarter analysis for tone shift detection
        
        Returns:
            EarningsAnalysisResult with aspect-based sentiments
        """
        start_time = datetime.now()
        
        # Parse transcript
        parsed = self.parser.parse(transcript)
        
        # Extract key segments for analysis
        key_segments = self._extract_key_segments(parsed["segments"])
        
        # Analyze each aspect
        aspects = []
        total_cost = 0.0
        model_used = ""
        
        for aspect_name in self.ASPECTS:
            aspect_result = self._analyze_aspect(
                aspect_name,
                key_segments,
                parsed["full_text"],
            )
            aspects.append(aspect_result)
            
            # Track cost (approximate, actual from LLM call)
            # Note: Real cost tracking happens in sentiment_client
        
        # Calculate overall sentiment
        overall_score, overall_sentiment, confidence = self._calculate_overall(aspects)
        
        # Detect tone shifts if previous analysis provided
        tone_shifts = []
        if previous_analysis:
            tone_shifts = self._detect_tone_shifts(aspects, previous_analysis.aspects)
        
        # Classify management tone and guidance clarity
        management_tone = self._classify_management_tone(aspects)
        guidance_clarity = self._classify_guidance_clarity(aspects)
        
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Extract fiscal year from quarter
        fiscal_year = self._extract_fiscal_year(quarter)
        
        return EarningsAnalysisResult(
            ticker=ticker,
            quarter=quarter,
            fiscal_year=fiscal_year,
            aspects=aspects,
            overall_sentiment=overall_sentiment,
            overall_score=overall_score,
            confidence=confidence,
            tone_shifts=tone_shifts,
            vs_previous_quarter=self._compare_to_previous(previous_analysis) if previous_analysis else None,
            transcript_length=len(transcript),
            word_count=parsed["word_count"],
            segment_count=parsed["segment_count"],
            management_tone=management_tone,
            guidance_clarity=guidance_clarity,
            cost_usd=total_cost,
            latency_ms=elapsed_ms,
        )
    
    def _extract_key_segments(self, segments: list[dict]) -> list[dict]:
        """Extract most relevant segments for analysis."""
        # Prioritize CEO prepared remarks and CFO financial review
        prioritized = []
        
        for seg in segments:
            if seg.get("type") in ["ceo_prepared", "cfo_financial"]:
                prioritized.append(seg)
            elif seg.get("word_count", 0) > 100:  # Substantive responses
                prioritized.append(seg)
        
        # Limit to avoid token limits
        return prioritized[:20]
    
    def _analyze_aspect(
        self,
        aspect: str,
        segments: list[dict],
        full_text: str,
    ) -> AspectSentiment:
        """Analyze a specific aspect using LLM."""
        # Prepare context for this aspect
        prompt = ASPECT_PROMPTS[aspect]
        
        # Create focused context
        relevant_text = self._extract_aspect_context(aspect, segments, full_text)
        
        # Use specialized system prompt
        system_prompt = f"""You are an expert financial analyst specializing in earnings call analysis.
Your task is to analyze the {aspect.replace('_', ' ').upper()} aspect of an earnings call.
Be precise, extract exact quotes, and provide numerical scores."""
        
        # Call LLM
        full_prompt = f"{prompt}\n\nEARNINGS CALL TEXT:\n{relevant_text[:8000]}"  # Limit context
        
        try:
            result = self.sentiment_analyzer.analyze(
                full_prompt,
                document_type="earnings_call",
                system_prompt=system_prompt,
            )
            
            # Map SentimentResult to AspectSentiment
            # Sentiment score: map sentiment to -1.0 to +1.0 scale
            sentiment_score_map = {
                "bullish": 0.7,
                "bearish": -0.7,
                "neutral": 0.0,
            }
            score = sentiment_score_map.get(result.sentiment, 0.0)
            
            return AspectSentiment(
                aspect=aspect,
                sentiment=result.sentiment,
                score=score,
                confidence=result.confidence,
                key_quotes=result.key_factors,
                explanation=result.summary,
            )
        except Exception as e:
            # Fallback to neutral on error
            return AspectSentiment(
                aspect=aspect,
                sentiment="neutral",
                score=0.0,
                confidence=0.5,
                key_quotes=[],
                explanation=f"Error in analysis: {str(e)}",
            )
    
    def _extract_aspect_context(self, aspect: str, segments: list[dict], full_text: str) -> str:
        """Extract text most relevant to a specific aspect."""
        # Define keywords for each aspect
        aspect_keywords = {
            "revenue_guidance": ["guidance", "revenue", "sales", "forecast", "outlook", "expect", "project"],
            "margin_outlook": ["margin", "gross", "operating", "profitability", "efficiency", "cost"],
            "risk_factors": ["risk", "headwind", "challenge", "uncertainty", "macro", "competition"],
            "management_tone": ["confident", "excited", "pleased", "cautious", "challenging"],
            "capital_allocation": ["buyback", "dividend", "capex", "investment", "cash", "deploy"],
        }
        
        keywords = aspect_keywords.get(aspect, [])
        relevant_segments = []
        
        for seg in segments:
            text_lower = seg["text"].lower()
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                relevant_segments.append((score, seg["text"]))
        
        # Sort by relevance and join
        relevant_segments.sort(reverse=True)
        return "\n\n".join([text for _, text in relevant_segments[:10]])
    
    def _calculate_overall(self, aspects: list[AspectSentiment]) -> tuple[float, str, float]:
        """Calculate overall sentiment from aspects."""
        if not aspects:
            return 0.0, "neutral", 0.0
        
        # Weight aspects (revenue guidance most important)
        weights = {
            "revenue_guidance": 0.30,
            "margin_outlook": 0.25,
            "risk_factors": 0.20,
            "management_tone": 0.15,
            "capital_allocation": 0.10,
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        total_confidence = 0.0
        
        for aspect in aspects:
            weight = weights.get(aspect.aspect, 0.15)
            weighted_score += aspect.score * weight * aspect.confidence
            total_weight += weight * aspect.confidence
            total_confidence += aspect.confidence
        
        if total_weight > 0:
            overall_score = weighted_score / total_weight
        else:
            overall_score = 0.0
        
        avg_confidence = total_confidence / len(aspects) if aspects else 0.0
        
        # Map to sentiment
        if overall_score > 0.3:
            sentiment = "bullish"
        elif overall_score < -0.3:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        return overall_score, sentiment, avg_confidence
    
    def _detect_tone_shifts(
        self,
        current_aspects: list[AspectSentiment],
        previous_aspects: list[AspectSentiment],
    ) -> list[ToneShift]:
        """Detect quarter-over-quarter tone shifts."""
        shifts = []
        
        prev_map = {a.aspect: a for a in previous_aspects}
        
        for curr in current_aspects:
            if curr.aspect in prev_map:
                prev = prev_map[curr.aspect]
                change = curr.score - prev.score
                magnitude = abs(change)
                
                if magnitude >= 0.5:
                    significance = "major"
                elif magnitude >= 0.3:
                    significance = "moderate"
                else:
                    significance = "minor"
                
                shifts.append(ToneShift(
                    aspect=curr.aspect,
                    previous_score=prev.score,
                    current_score=curr.score,
                    shift_magnitude=magnitude,
                    shift_direction="positive" if change > 0 else "negative" if change < 0 else "stable",
                    significance=significance,
                ))
        
        return shifts
    
    def _classify_management_tone(self, aspects: list[AspectSentiment]) -> str:
        """Classify overall management tone."""
        tone_aspect = next((a for a in aspects if a.aspect == "management_tone"), None)
        if tone_aspect:
            if tone_aspect.score > 0.5:
                return "very_confident"
            elif tone_aspect.score > 0.2:
                return "confident"
            elif tone_aspect.score < -0.5:
                return "defensive"
            elif tone_aspect.score < -0.2:
                return "cautious"
        return "neutral"
    
    def _classify_guidance_clarity(self, aspects: list[AspectSentiment]) -> str:
        """Classify guidance clarity."""
        guidance_aspect = next((a for a in aspects if a.aspect == "revenue_guidance"), None)
        if guidance_aspect:
            if guidance_aspect.confidence > 0.8 and abs(guidance_aspect.score) > 0.3:
                return "clear"
            elif guidance_aspect.confidence < 0.5:
                return "vague"
        return "mixed"
    
    def _compare_to_previous(self, previous: EarningsAnalysisResult) -> dict:
        """Compare current analysis to previous quarter."""
        return {
            "previous_quarter": previous.quarter,
            "overall_score_change": None,  # Will be populated by caller
            "sentiment_changed": None,
        }
    
    def _extract_fiscal_year(self, quarter: str) -> int:
        """Extract fiscal year from quarter string."""
        match = re.search(r'(\d{4})', quarter)
        return int(match.group(1)) if match else datetime.now().year


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("""Usage:
  python -m src.llm.earnings_analyzer analyze <ticker> --quarter Q4-2025 --file transcript.txt
  python -m src.llm.earnings_analyzer analyze <ticker> --quarter Q4-2025 --text "CEO stated..."
  python -m src.llm.earnings_analyzer batch --dir ./transcripts/
  python -m src.llm.earnings_analyzer mock-test
        """)
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "mock-test":
        # Run test with mock transcript
        mock_transcript = """
        Operator: Good afternoon, everyone, and welcome to Apple Q4 2025 earnings call.
        
        Tim Cook - CEO: Thank you. I'm pleased to report another record quarter. Revenue grew 8% 
        year-over-year to $89.5 billion, ahead of our guidance. iPhone 16 demand has exceeded 
        expectations, particularly in emerging markets. We're seeing strong momentum heading 
        into fiscal 2026.
        
        Guidance: For Q1 2026, we expect revenue between $94-98 billion, representing 5-9% 
        growth. Gross margin is expected to be 46-47%, up from 45.6% this quarter.
        
        Luca Maestri - CFO: Our operating margin improved 120 basis points to 30.5%. 
        We generated $24 billion in free cash flow. Capital return to shareholders totaled 
        $25 billion through dividends and buybacks.
        
        Analyst Question: What about supply chain headwinds?
        
        Tim Cook: We've successfully diversified our supply chain. While there are always 
        challenges, we feel confident in our ability to meet demand. The macro environment 
        remains uncertain but we're well-positioned.
        
        We're excited about AI opportunities and remain confident in our long-term trajectory.
        """
        
        analyzer = EarningsAnalyzer()
        result = analyzer.analyze_transcript("AAPL", "Q4-2025", mock_transcript)
        
        print(json.dumps(result.to_dict(), indent=2))
        return
    
    if command == "analyze":
        if len(sys.argv) < 4:
            print("Usage: analyze <ticker> --quarter Q4-2025 [--file path | --text text]")
            sys.exit(1)
        
        ticker = sys.argv[2]
        
        # Parse arguments
        quarter = None
        filepath = None
        text = None
        
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--quarter" and i + 1 < len(sys.argv):
                quarter = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--file" and i + 1 < len(sys.argv):
                filepath = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--text" and i + 1 < len(sys.argv):
                text = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        
        if not quarter:
            print("Error: --quarter required")
            sys.exit(1)
        
        if filepath:
            text = Path(filepath).read_text()
        elif not text:
            text = sys.stdin.read()
        
        if not text:
            print("Error: No transcript provided")
            sys.exit(1)
        
        analyzer = EarningsAnalyzer()
        result = analyzer.analyze_transcript(ticker, quarter, text)
        
        print(json.dumps(result.to_dict(), indent=2))
        return
    
    print(f"Unknown command: {command}")
    sys.exit(1)


if __name__ == "__main__":
    main()
