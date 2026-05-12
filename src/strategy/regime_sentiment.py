#!/usr/bin/env python3
"""
Portfolio-Lab v2.30 Phase 3: Regime Sentiment Integration
Combines technical regime detection with LLM sentiment signals.

Implements weighted regime score:
    regime_score = 0.7 * technical_regime + 0.3 * sentiment_regime

Integrates with:
- regime_ml.py (v2.20) for technical regime detection
- circuit_breaker.py (v2.14) for risk management
- sentiment_analyzer.py (v2.30 Phase 2) for sentiment signals
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from enum import Enum
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategy.sentiment_analyzer import (
    SentimentAggregator, 
    AggregatedSentiment,
    SentimentAnalyzerPipeline
)


class RegimeSentiment(Enum):
    """Sentiment-based regime classification"""
    EXTREME_BULLISH = "extreme_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    EXTREME_BEARISH = "extreme_bearish"


@dataclass
class CombinedRegimeSignal:
    """
    Combined regime signal from technical and sentiment sources.
    Used for allocation decisions and risk management.
    """
    timestamp: str
    
    # Component signals
    technical_regime: str  # From regime_ml.py
    technical_confidence: float
    sentiment_regime: str  # From sentiment_analyzer.py
    sentiment_confidence: float
    
    # Combined score (-1 to 1, -1 = risk-off, 1 = risk-on)
    combined_score: float
    combined_regime: str  # extreme_risk_off, risk_off, neutral, risk_on, extreme_risk_on
    
    # Weights used
    technical_weight: float
    sentiment_weight: float
    
    # Risk management signals
    circuit_breaker_level: str  # green, yellow, orange, red
    position_scaling_factor: float  # 0.0 to 1.0
    
    # Allocation recommendations
    equity_tilt: float  # -1 (min) to 1 (max)
    bond_duration_tilt: float  # -1 (short) to 1 (long)
    gold_tilt: float  # -1 (min) to 1 (max)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "technical_regime": self.technical_regime,
            "technical_confidence": self.technical_confidence,
            "sentiment_regime": self.sentiment_regime,
            "sentiment_confidence": self.sentiment_confidence,
            "combined_score": self.combined_score,
            "combined_regime": self.combined_regime,
            "technical_weight": self.technical_weight,
            "sentiment_weight": self.sentiment_weight,
            "circuit_breaker_level": self.circuit_breaker_level,
            "position_scaling_factor": self.position_scaling_factor,
            "equity_tilt": self.equity_tilt,
            "bond_duration_tilt": self.bond_duration_tilt,
            "gold_tilt": self.gold_tilt,
        }


class RegimeSentimentIntegrator:
    """
    Integrates technical regime detection with LLM sentiment analysis.
    
    Default weights: 70% technical, 30% sentiment (configurable)
    Sentiment weight increases when technical signals are ambiguous.
    """
    
    # Default weights
    DEFAULT_TECHNICAL_WEIGHT = 0.70
    DEFAULT_SENTIMENT_WEIGHT = 0.30
    
    # Regime classification thresholds
    EXTREME_RISK_ON_THRESHOLD = 0.6
    RISK_ON_THRESHOLD = 0.3
    RISK_OFF_THRESHOLD = -0.3
    EXTREME_RISK_OFF_THRESHOLD = -0.6
    
    # Circuit breaker mapping
    CB_GREEN_THRESHOLD = 0.2
    CB_YELLOW_THRESHOLD = -0.2
    CB_ORANGE_THRESHOLD = -0.5
    # Below CB_ORANGE = red
    
    def __init__(
        self,
        technical_weight: Optional[float] = None,
        sentiment_weight: Optional[float] = None,
    ):
        self.technical_weight = technical_weight or self.DEFAULT_TECHNICAL_WEIGHT
        self.sentiment_weight = sentiment_weight or self.DEFAULT_SENTIMENT_WEIGHT
        
        # Validate weights sum to 1.0
        total = self.technical_weight + self.sentiment_weight
        if abs(total - 1.0) > 0.001:
            # Normalize
            self.technical_weight /= total
            self.sentiment_weight /= total
    
    def map_sentiment_to_score(self, sentiment_regime: str) -> float:
        """Map sentiment regime to numeric score (-1 to 1)."""
        mapping = {
            "risk_on": 0.5,
            "neutral": 0.0,
            "risk_off": -0.5,
            "extreme_risk_off": -0.8,
        }
        return mapping.get(sentiment_regime, 0.0)
    
    def map_technical_to_score(self, technical_regime: str) -> float:
        """Map technical regime to numeric score (-1 to 1)."""
        # Based on v2.20 regime_ml.py regime labels
        mapping = {
            "bullish_momentum": 0.7,
            "neutral_trending": 0.2,
            "volatile_chop": 0.0,
            "bearish_momentum": -0.5,
            "crisis": -0.8,
            "recovery": 0.4,
            "expansion": 0.6,
            "contraction": -0.4,
        }
        return mapping.get(technical_regime, 0.0)
    
    def adjust_weights(
        self,
        technical_confidence: float,
        sentiment_confidence: float,
    ) -> Tuple[float, float]:
        """
        Dynamically adjust weights based on confidence levels.
        Increase sentiment weight when technical confidence is low.
        """
        if technical_confidence < 0.5 and sentiment_confidence > 0.7:
            # Boost sentiment when technical is uncertain but sentiment is strong
            tech_w = 0.5
            sent_w = 0.5
        elif technical_confidence < 0.3:
            # Heavy sentiment weighting when technical is very uncertain
            tech_w = 0.4
            sent_w = 0.6
        else:
            tech_w = self.technical_weight
            sent_w = self.sentiment_weight
        
        return tech_w, sent_w
    
    def classify_combined_regime(self, score: float) -> str:
        """Classify combined score into regime."""
        if score >= self.EXTREME_RISK_ON_THRESHOLD:
            return "extreme_risk_on"
        elif score >= self.RISK_ON_THRESHOLD:
            return "risk_on"
        elif score <= self.EXTREME_RISK_OFF_THRESHOLD:
            return "extreme_risk_off"
        elif score <= self.RISK_OFF_THRESHOLD:
            return "risk_off"
        else:
            return "neutral"
    
    def determine_circuit_breaker(self, score: float) -> str:
        """Map combined score to circuit breaker level."""
        if score >= self.CB_GREEN_THRESHOLD:
            return "green"
        elif score >= self.CB_YELLOW_THRESHOLD:
            return "yellow"
        elif score >= self.CB_ORANGE_THRESHOLD:
            return "orange"
        else:
            return "red"
    
    def calculate_position_scaling(self, regime: str) -> float:
        """Calculate position scaling factor based on regime."""
        scaling = {
            "extreme_risk_on": 1.0,
            "risk_on": 0.95,
            "neutral": 0.85,
            "risk_off": 0.70,
            "extreme_risk_off": 0.50,
        }
        return scaling.get(regime, 0.85)
    
    def calculate_allocation_tilts(
        self,
        score: float,
        regime: str,
    ) -> Tuple[float, float, float]:
        """
        Calculate allocation tilts for equity, bond duration, and gold.
        
        Returns:
            (equity_tilt, bond_duration_tilt, gold_tilt)
            Each in range [-1, 1]
        """
        # Equity tilt: increases with positive score
        equity_tilt = np.clip(score * 1.5, -1.0, 1.0)
        
        # Bond duration: decreases (shorten) in risk-off, increases in risk-on
        if regime in ["risk_off", "extreme_risk_off"]:
            bond_duration_tilt = -0.5  # Shorten duration
        elif regime == "extreme_risk_on":
            bond_duration_tilt = 0.3  # Slight extension
        else:
            bond_duration_tilt = 0.0
        
        # Gold tilt: increases in risk-off regimes as hedge
        if regime in ["risk_off", "extreme_risk_off"]:
            gold_tilt = 0.7
        elif score < -0.2:
            gold_tilt = 0.4
        else:
            gold_tilt = 0.0
        
        return equity_tilt, bond_duration_tilt, gold_tilt
    
    def combine_signals(
        self,
        technical_regime: str,
        technical_confidence: float,
        sentiment: AggregatedSentiment,
    ) -> CombinedRegimeSignal:
        """
        Combine technical and sentiment signals into unified regime signal.
        
        Args:
            technical_regime: Regime label from regime_ml.py
            technical_confidence: Confidence score 0-1 from technical model
            sentiment: Aggregated sentiment from sentiment_analyzer.py
        """
        timestamp = datetime.now().isoformat()
        
        # Get scores
        tech_score = self.map_technical_to_score(technical_regime)
        sent_score = self.map_sentiment_to_score(sentiment.regime_signal)
        
        # Adjust weights based on confidence
        tech_w, sent_w = self.adjust_weights(
            technical_confidence,
            sentiment.confidence,
        )
        
        # Calculate combined score
        combined_score = tech_w * tech_score + sent_w * sent_score
        
        # Classify regime
        combined_regime = self.classify_combined_regime(combined_score)
        
        # Risk management
        cb_level = self.determine_circuit_breaker(combined_score)
        position_scale = self.calculate_position_scaling(combined_regime)
        
        # Allocation tilts
        eq_tilt, bond_tilt, gold_tilt = self.calculate_allocation_tilts(
            combined_score,
            combined_regime,
        )
        
        return CombinedRegimeSignal(
            timestamp=timestamp,
            technical_regime=technical_regime,
            technical_confidence=technical_confidence,
            sentiment_regime=sentiment.regime_signal,
            sentiment_confidence=sentiment.confidence,
            combined_score=round(combined_score, 4),
            combined_regime=combined_regime,
            technical_weight=round(tech_w, 4),
            sentiment_weight=round(sent_w, 4),
            circuit_breaker_level=cb_level,
            position_scaling_factor=round(position_scale, 4),
            equity_tilt=round(eq_tilt, 4),
            bond_duration_tilt=round(bond_tilt, 4),
            gold_tilt=round(gold_tilt, 4),
        )


class RegimeSentimentPipeline:
    """
    High-level pipeline for regime-sentiment integration.
    Wraps sentiment analysis and regime combination.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("~/projects/portfolio-lab/data/regime_sentiment").expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sentiment_pipeline = SentimentAnalyzerPipeline()
        self.integrator = RegimeSentimentIntegrator()
    
    def get_combined_signal(
        self,
        technical_regime: str,
        technical_confidence: float,
        news_texts: Optional[list] = None,
        earnings_texts: Optional[list] = None,
        macro_texts: Optional[list] = None,
    ) -> CombinedRegimeSignal:
        """
        Get combined regime signal from technical and sentiment inputs.
        
        Args:
            technical_regime: Current regime from regime_ml.py
            technical_confidence: Model confidence 0-1
            news_texts: Optional news headlines/articles
            earnings_texts: Optional earnings transcripts
            macro_texts: Optional macro commentary
        """
        # Get sentiment
        sentiment = self.sentiment_pipeline.get_current_sentiment(
            news_texts=news_texts,
            earnings_texts=earnings_texts,
            macro_texts=macro_texts,
        )
        
        # Combine
        return self.integrator.combine_signals(
            technical_regime=technical_regime,
            technical_confidence=technical_confidence,
            sentiment=sentiment,
        )
    
    def save_signal(self, signal: CombinedRegimeSignal, filename: Optional[str] = None):
        """Save combined signal to JSON."""
        if filename is None:
            filename = f"regime_signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.data_dir / filename
        with open(filepath, 'w') as f:
            json.dump(signal.to_dict(), f, indent=2)
        
        return filepath
    
    def get_current_allocation_weights(
        self,
        signal: CombinedRegimeSignal,
        base_allocation: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Calculate allocation weights based on combined signal.
        
        Args:
            signal: Combined regime signal
            base_allocation: Base allocation (default: SPY/GLD/TLT 46/38/16)
        
        Returns:
            Adjusted allocation dict
        """
        if base_allocation is None:
            base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        else:
            base = base_allocation.copy()
        
        # Apply tilts
        spy_adj = signal.equity_tilt * 0.15  # Max ±15% adjustment
        gld_adj = signal.gold_tilt * 0.10  # Max ±10% adjustment
        tlt_adj = signal.bond_duration_tilt * 0.08  # Max ±8% adjustment
        
        adjusted = {
            "SPY": np.clip(base.get("SPY", 0) + spy_adj, 0.20, 0.70),
            "GLD": np.clip(base.get("GLD", 0) + gld_adj, 0.20, 0.50),
            "TLT": np.clip(base.get("TLT", 0) + tlt_adj, 0.05, 0.25),
        }
        
        # Normalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: round(v / total, 4) for k, v in adjusted.items()}
        
        return adjusted


def demo():
    """Demo the regime-sentiment integration."""
    print("=" * 70)
    print("Portfolio-Lab v2.30 Phase 3: Regime-Sentiment Integration Demo")
    print("=" * 70)
    
    pipeline = RegimeSentimentPipeline()
    
    # Sample technical regime input
    technical_regime = "bullish_momentum"
    technical_confidence = 0.75
    
    # Sample sentiment inputs
    news_texts = [
        "Tech stocks surge on AI breakthrough announcements",
        "Fed signals dovish stance with cooling inflation data",
    ]
    
    macro_texts = [
        "Economic indicators suggest soft landing scenario",
    ]
    
    print(f"\nTechnical Regime: {technical_regime} (confidence: {technical_confidence})")
    print(f"Sentiment Sources: {len(news_texts)} news, 0 earnings, {len(macro_texts)} macro")
    
    # Get combined signal
    signal = pipeline.get_combined_signal(
        technical_regime=technical_regime,
        technical_confidence=technical_confidence,
        news_texts=news_texts,
        macro_texts=macro_texts,
    )
    
    print(f"\n{'─' * 70}")
    print("COMBINED REGIME SIGNAL")
    print(f"{'─' * 70}")
    print(f"Timestamp: {signal.timestamp}")
    print(f"\nComponent Scores:")
    print(f"  Technical Regime: {signal.technical_regime} (conf: {signal.technical_confidence:.2%})")
    print(f"  Sentiment Regime: {signal.sentiment_regime} (conf: {signal.sentiment_confidence:.2%})")
    print(f"\nWeighting Applied:")
    print(f"  Technical Weight: {signal.technical_weight:.1%}")
    print(f"  Sentiment Weight: {signal.sentiment_weight:.1%}")
    print(f"\nCombined Result:")
    print(f"  Score: {signal.combined_score:+.4f}")
    print(f"  Regime: {signal.combined_regime.upper()}")
    print(f"{'─' * 70}")
    print(f"Risk Management:")
    print(f"  Circuit Breaker: {signal.circuit_breaker_level.upper()}")
    print(f"  Position Scale: {signal.position_scaling_factor:.1%}")
    print(f"{'─' * 70}")
    print(f"Allocation Tilts:")
    print(f"  Equity (SPY): {signal.equity_tilt:+.4f}")
    print(f"  Bond Duration (TLT): {signal.bond_duration_tilt:+.4f}")
    print(f"  Gold (GLD): {signal.gold_tilt:+.4f}")
    
    # Calculate allocation
    allocation = pipeline.get_current_allocation_weights(signal)
    print(f"\n{'─' * 70}")
    print(f"Adjusted Allocation (from base 46/38/16):")
    for symbol, weight in allocation.items():
        print(f"  {symbol}: {weight:.2%}")
    
    # Save
    filepath = pipeline.save_signal(signal, "demo_regime_signal.json")
    print(f"\n{'─' * 70}")
    print(f"✓ Saved to: {filepath}")
    
    return signal


if __name__ == "__main__":
    demo()
