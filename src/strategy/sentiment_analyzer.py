#!/usr/bin/env python3
"""
Portfolio-Lab v2.30 Phase 2: Sentiment Analyzer
Aggregates and smooths LLM sentiment signals for regime detection.

Implements:
- Multi-source sentiment aggregation (news, earnings, macro)
- Exponential smoothing with 7-day half-life
- Sentiment momentum calculation
- Regime-ready signal output
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from collections import deque
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm.sentiment_client import SentimentAnalyzer, SentimentResult


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment across multiple sources with smoothing."""
    timestamp: str
    
    # Raw scores by source (-1 to 1)
    news_sentiment: float
    earnings_sentiment: float
    macro_sentiment: float
    
    # Aggregated score
    composite_score: float  # -1 (bearish) to 1 (bullish)
    confidence: float  # 0 to 1
    
    # Smoothed signals
    smoothed_score: float  # EMA smoothed
    sentiment_momentum: float  # Rate of change
    
    # Regime classification
    regime_signal: str  # risk_on, neutral, risk_off, extreme_risk_off
    
    # Metadata
    sources_used: int
    data_quality: str  # high, medium, low
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "news_sentiment": self.news_sentiment,
            "earnings_sentiment": self.earnings_sentiment,
            "macro_sentiment": self.macro_sentiment,
            "composite_score": self.composite_score,
            "confidence": self.confidence,
            "smoothed_score": self.smoothed_score,
            "sentiment_momentum": self.sentiment_momentum,
            "regime_signal": self.regime_signal,
            "sources_used": self.sources_used,
            "data_quality": self.data_quality,
        }


class SentimentAggregator:
    """
    Aggregates sentiment from multiple sources with exponential smoothing.
    7-day half-life decay for sentiment scores as per research.
    """
    
    # Half-life of 7 days → decay factor for daily data
    HALF_LIFE_DAYS = 7
    DECAY_FACTOR = 0.5 ** (1 / HALF_LIFE_DAYS)  # ~0.906 per day
    
    # Regime thresholds
    RISK_ON_THRESHOLD = 0.3
    RISK_OFF_THRESHOLD = -0.3
    EXTREME_RISK_OFF_THRESHOLD = -0.6
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self.history: deque = deque(maxlen=lookback_days)
        try:
            self.analyzer = SentimentAnalyzer()
        except Exception as e:
            print(f"Warning: Could not initialize LLM client in aggregator ({e})")
            self.analyzer = None
        
    def calculate_ema(self, values: List[float], alpha: Optional[float] = None) -> float:
        """Calculate exponential moving average with 7-day half-life."""
        if not values:
            return 0.0
        if alpha is None:
            alpha = 1 - self.DECAY_FACTOR
        
        ema = values[0]
        for value in values[1:]:
            ema = alpha * value + (1 - alpha) * ema
        return ema
    
    def calculate_momentum(self, scores: List[float], window: int = 5) -> float:
        """Calculate sentiment momentum (rate of change)."""
        if len(scores) < window:
            return 0.0
        recent = np.mean(scores[-window:])
        previous = np.mean(scores[-window*2:-window]) if len(scores) >= window*2 else scores[0]
        return recent - previous
    
    def classify_regime_signal(self, smoothed_score: float, momentum: float) -> str:
        """Classify sentiment into regime signal."""
        if smoothed_score < self.EXTREME_RISK_OFF_THRESHOLD:
            return "extreme_risk_off"
        elif smoothed_score < self.RISK_OFF_THRESHOLD:
            return "risk_off"
        elif smoothed_score > self.RISK_ON_THRESHOLD:
            return "risk_on"
        else:
            # Check momentum for early signals
            if momentum > 0.2 and smoothed_score > 0:
                return "risk_on"
            elif momentum < -0.2 and smoothed_score < 0:
                return "risk_off"
            return "neutral"
    
    def aggregate_sources(
        self,
        news_results: Optional[List[SentimentResult]] = None,
        earnings_results: Optional[List[SentimentResult]] = None,
        macro_results: Optional[List[SentimentResult]] = None,
    ) -> AggregatedSentiment:
        """
        Aggregate sentiment from multiple sources.
        
        Weights: News 30%, Earnings 40%, Macro 30%
        """
        timestamp = datetime.now().isoformat()
        
        # Process news sentiment
        news_score = 0.0
        if news_results:
            news_scores = [
                self._sentiment_to_score(r.sentiment) * r.confidence
                for r in news_results
            ]
            news_score = np.mean(news_scores) if news_scores else 0.0
        
        # Process earnings sentiment
        earnings_score = 0.0
        if earnings_results:
            earnings_scores = [
                self._sentiment_to_score(r.sentiment) * r.confidence
                for r in earnings_results
            ]
            earnings_score = np.mean(earnings_scores) if earnings_scores else 0.0
        
        # Process macro sentiment
        macro_score = 0.0
        if macro_results:
            macro_scores = [
                self._sentiment_to_score(r.sentiment) * r.confidence
                for r in macro_results
            ]
            macro_score = np.mean(macro_scores) if macro_scores else 0.0
        
        # Weighted composite
        weights = {"news": 0.30, "earnings": 0.40, "macro": 0.30}
        sources_used = sum([
            1 if news_results else 0,
            1 if earnings_results else 0,
            1 if macro_results else 0
        ])
        
        if sources_used == 0:
            composite = 0.0
            confidence = 0.0
            data_quality = "low"
        else:
            # Normalize weights based on available sources
            total_weight = (
                weights["news"] * (1 if news_results else 0) +
                weights["earnings"] * (1 if earnings_results else 0) +
                weights["macro"] * (1 if macro_results else 0)
            )
            normalized_weights = {
                k: v / total_weight for k, v in weights.items()
            }
            
            composite = (
                news_score * normalized_weights.get("news", 0) +
                earnings_score * normalized_weights.get("earnings", 0) +
                macro_score * normalized_weights.get("macro", 0)
            )
            
            # Confidence based on number of sources and agreement
            confidences = []
            if news_results:
                confidences.append(np.mean([r.confidence for r in news_results]))
            if earnings_results:
                confidences.append(np.mean([r.confidence for r in earnings_results]))
            if macro_results:
                confidences.append(np.mean([r.confidence for r in macro_results]))
            
            confidence = np.mean(confidences) if confidences else 0.5
            
            # Check agreement
            scores = [s for s in [news_score, earnings_score, macro_score] if s != 0]
            if len(scores) >= 2:
                agreement = 1 - (np.std(scores) / 2)  # Normalize std to 0-1
                confidence *= (0.5 + 0.5 * agreement)
            
            data_quality = "high" if sources_used >= 2 and confidence > 0.7 else \
                          "medium" if sources_used >= 2 else "low"
        
        # Add to history for smoothing
        self.history.append(composite)
        
        # Calculate smoothed score
        smoothed = self.calculate_ema(list(self.history))
        
        # Calculate momentum
        momentum = self.calculate_momentum(list(self.history))
        
        # Classify regime
        regime = self.classify_regime_signal(smoothed, momentum)
        
        return AggregatedSentiment(
            timestamp=timestamp,
            news_sentiment=round(news_score, 4),
            earnings_sentiment=round(earnings_score, 4),
            macro_sentiment=round(macro_score, 4),
            composite_score=round(composite, 4),
            confidence=round(confidence, 4),
            smoothed_score=round(smoothed, 4),
            sentiment_momentum=round(momentum, 4),
            regime_signal=regime,
            sources_used=sources_used,
            data_quality=data_quality,
        )
    
    def _sentiment_to_score(self, sentiment: str) -> float:
        """Convert sentiment label to numeric score."""
        mapping = {
            "bullish": 1.0,
            "bearish": -1.0,
            "neutral": 0.0,
        }
        return mapping.get(sentiment, 0.0)


class SentimentAnalyzerPipeline:
    """
    High-level pipeline for sentiment analysis.
    Orchestrates data fetching, LLM analysis, and aggregation.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("~/projects/portfolio-lab/data/sentiment").expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.aggregator = SentimentAggregator()
        try:
            self.analyzer = SentimentAnalyzer()
        except Exception as e:
            print(f"Warning: Could not initialize LLM client ({e})")
            self.analyzer = None
        
    def analyze_text(self, text: str, source_type: str = "news") -> Optional[SentimentResult]:
        """Analyze a single text snippet."""
        if self.analyzer is None:
            return None
        return self.analyzer.analyze(text)
    
    def get_current_sentiment(
        self,
        news_texts: Optional[List[str]] = None,
        earnings_texts: Optional[List[str]] = None,
        macro_texts: Optional[List[str]] = None,
    ) -> AggregatedSentiment:
        """
        Get current aggregated sentiment from provided texts.
        
        Args:
            news_texts: List of news headlines/articles
            earnings_texts: List of earnings call transcripts
            macro_texts: List of macro commentary (Fed speeches, etc.)
        """
        news_results = None
        earnings_results = None
        macro_results = None
        
        if self.analyzer is not None:
            if news_texts:
                news_results = [self.analyzer.analyze(text) for text in news_texts[:5]]
            
            if earnings_texts:
                earnings_results = [self.analyzer.analyze(text) for text in earnings_texts[:3]]
            
            if macro_texts:
                macro_results = [self.analyzer.analyze(text) for text in macro_texts[:3]]
        else:
            # Mock results for demo without API keys
            from src.llm.sentiment_client import SentimentResult
            from datetime import datetime
            
            def mock_result(text: str, sentiment: str) -> SentimentResult:
                return SentimentResult(
                    sentiment=sentiment,
                    confidence=0.75,
                    key_factors=["mock_factor"],
                    price_impact="positive" if sentiment == "bullish" else "negative" if sentiment == "bearish" else "neutral",
                    time_horizon="short_term",
                    summary=f"Mock analysis of: {text[:30]}...",
                    model="mock",
                    cost_usd=0.0,
                    prompt_tokens=0,
                    cached_tokens=0,
                    completion_tokens=0,
                )
            
            if news_texts:
                news_results = [mock_result(t, "bullish") for t in news_texts]
            if earnings_texts:
                earnings_results = [mock_result(t, "bullish") for t in earnings_texts]
            if macro_texts:
                macro_results = [mock_result(t, "neutral") for t in macro_texts]
        
        return self.aggregator.aggregate_sources(
            news_results=news_results,
            earnings_results=earnings_results,
            macro_results=macro_results,
        )
    
    def save_sentiment(self, sentiment: AggregatedSentiment, filename: Optional[str] = None):
        """Save sentiment to JSON file."""
        if filename is None:
            filename = f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.data_dir / filename
        with open(filepath, 'w') as f:
            json.dump(sentiment.to_dict(), f, indent=2)
        
        return filepath
    
    def load_sentiment_history(self, days: int = 30) -> List[AggregatedSentiment]:
        """Load sentiment history from saved files."""
        history = []
        cutoff = datetime.now() - timedelta(days=days)
        
        for filepath in self.data_dir.glob("sentiment_*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    ts = datetime.fromisoformat(data["timestamp"])
                    if ts > cutoff:
                        history.append(AggregatedSentiment(**data))
            except Exception:
                continue
        
        return sorted(history, key=lambda x: x.timestamp)


def demo():
    """Demo the sentiment analyzer with sample data."""
    print("=" * 60)
    print("Portfolio-Lab v2.30: Sentiment Analyzer Demo")
    print("=" * 60)
    
    pipeline = SentimentAnalyzerPipeline()
    
    # Sample texts for demonstration
    sample_news = [
        "Federal Reserve signals potential rate cuts as inflation cools",
        "Tech stocks rally on strong AI earnings reports",
        "Market volatility remains elevated ahead of jobs report",
    ]
    
    sample_earnings = [
        "Management expressed confidence in Q4 guidance, citing strong demand",
    ]
    
    sample_macro = [
        "Fed Chair Powell's speech emphasized data-dependent approach to policy",
    ]
    
    print("\nAnalyzing sample texts...")
    sentiment = pipeline.get_current_sentiment(
        news_texts=sample_news,
        earnings_texts=sample_earnings,
        macro_texts=sample_macro,
    )
    
    print(f"\n{'─' * 60}")
    print("Aggregated Sentiment Results:")
    print(f"{'─' * 60}")
    print(f"Timestamp: {sentiment.timestamp}")
    print(f"News Sentiment: {sentiment.news_sentiment:+.4f}")
    print(f"Earnings Sentiment: {sentiment.earnings_sentiment:+.4f}")
    print(f"Macro Sentiment: {sentiment.macro_sentiment:+.4f}")
    print(f"{'─' * 60}")
    print(f"Composite Score: {sentiment.composite_score:+.4f}")
    print(f"Confidence: {sentiment.confidence:.2%}")
    print(f"Smoothed Score: {sentiment.smoothed_score:+.4f}")
    print(f"Momentum: {sentiment.sentiment_momentum:+.4f}")
    print(f"{'─' * 60}")
    print(f"Regime Signal: {sentiment.regime_signal.upper()}")
    print(f"Data Quality: {sentiment.data_quality}")
    print(f"Sources Used: {sentiment.sources_used}")
    
    # Save results
    filepath = pipeline.save_sentiment(sentiment, "demo_sentiment.json")
    print(f"\n✓ Saved to: {filepath}")
    
    return sentiment


if __name__ == "__main__":
    demo()
