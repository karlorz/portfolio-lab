#!/usr/bin/env python3
"""
Portfolio-Lab v2.24: Signal Integrator Module

Master signal aggregation engine that combines multiple signal sources into 
unified portfolio decisions with weighted composite scoring, regime-based
dynamic weight adjustment, and conflict resolution.

Architecture:
- SignalIntegrator: Master orchestrator for signal aggregation
- SignalSource: Abstract base for all signal types
- Implementations: Technical, Macro, AlternativeData, LLMSentiment, Options

Signal Weights (Base):
- Momentum: 0.30 (price-based trend signals)
- Value: 0.20 (fundamental/valuation signals)  
- Macro: 0.20 (Fed policy, economic indicators)
- Quality: 0.15 (earnings quality, balance sheet)
- Sentiment: 0.15 (alternative data + LLM sentiment)

Usage:
    from src.signals.integrator import SignalIntegrator
    
    integrator = SignalIntegrator()
    
    # Get composite signal for single asset
    result = integrator.get_composite_signal("SPY")
    
    # Get portfolio allocation deltas
    deltas = integrator.get_allocation_deltas(
        current_alloc={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    )

CLI:
    python -m src.signals.integrator composite --asset SPY
    python -m src.signals.integrator portfolio --current 46/38/16
    python -m src.signals.integrator history --days 30
"""

import json
import sqlite3
import argparse
import sys
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Existing module imports
from src.data.alternative_data import AlternativeDataClient, AlternativeDataSignal
from src.llm.sentiment_client import SentimentAnalyzer, SentimentResult
from src.options.odte_overlay import GEXCalculator, GEXProfile

# ---------------------------------------------------------------------------
# Constants and Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
DB_PATH = DATA_DIR / "signals.db"

# Base signal weights (adjusted dynamically by regime)
BASE_WEIGHTS = {
    "momentum": 0.30,
    "value": 0.20,
    "macro": 0.20,
    "quality": 0.15,
    "sentiment": 0.15,
}

# Regime-specific weight adjustments
REGIME_WEIGHTS = {
    "bull": {"momentum": 0.40, "value": 0.15, "macro": 0.15, "quality": 0.15, "sentiment": 0.15},
    "bear": {"momentum": 0.20, "value": 0.25, "macro": 0.30, "quality": 0.15, "sentiment": 0.10},
    "neutral": {"momentum": 0.25, "value": 0.25, "macro": 0.20, "quality": 0.15, "sentiment": 0.15},
    "crisis": {"momentum": 0.10, "value": 0.20, "macro": 0.35, "quality": 0.20, "sentiment": 0.15},
    "high_vol": {"momentum": 0.15, "value": 0.20, "macro": 0.25, "quality": 0.25, "sentiment": 0.15},
}

# Minimum signal sources required for valid composite
MIN_SIGNAL_SOURCES = 3

# Signal normalization bounds
SIGNAL_MIN = -1.0
SIGNAL_MAX = 1.0

# Allocation delta bounds
MAX_DELTA_PCT = 0.05  # +/- 5%


class SignalType(Enum):
    """Types of signals supported by the integrator."""
    MOMENTUM = "momentum"
    VALUE = "value"
    MACRO = "macro"
    QUALITY = "quality"
    SENTIMENT = "sentiment"


class RegimeType(Enum):
    """Market regime classifications."""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    CRISIS = "crisis"
    HIGH_VOL = "high_vol"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SignalSourceResult:
    """Result from a single signal source."""
    source_type: str  # e.g., 'momentum', 'macro', 'sentiment'
    source_name: str  # e.g., 'dual_momentum', 'fed_analyzer', 'llm_sentiment'
    
    # Signal value (-1.0 to +1.0)
    signal: float
    confidence: float  # 0.0 to 1.0
    
    # Metadata
    raw_score: float  # Original unnormalized score
    raw_unit: str  # Unit of raw score (e.g., 'z_score', 'pct_change')
    
    # Historical accuracy
    historical_accuracy: Optional[float] = None  # 0.0 to 1.0
    sample_count: int = 0
    
    # Context
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompositeSignal:
    """Aggregated composite signal for an asset."""
    ticker: str
    timestamp: str
    
    # Component signals
    component_signals: List[SignalSourceResult] = field(default_factory=list)
    
    # Composite score (-1.0 to +1.0)
    composite_score: float = 0.0
    composite_confidence: float = 0.0
    
    # Attribution
    primary_drivers: List[str] = field(default_factory=list)  # Top 3 sources
    signal_agreement: str = "neutral"  # aligned, mixed, conflicting
    
    # Regime context
    detected_regime: str = "neutral"
    weights_used: Dict[str, float] = field(default_factory=dict)
    
    # Historical performance
    expected_accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "composite_score": round(self.composite_score, 4),
            "composite_confidence": round(self.composite_confidence, 4),
            "primary_drivers": self.primary_drivers,
            "signal_agreement": self.signal_agreement,
            "detected_regime": self.detected_regime,
            "weights_used": self.weights_used,
            "expected_accuracy": self.expected_accuracy,
            "component_count": len(self.component_signals),
            "components": [c.to_dict() for c in self.component_signals],
        }


@dataclass
class AllocationDelta:
    """Recommended allocation change for an asset."""
    ticker: str
    current_weight: float
    recommended_weight: float
    delta: float  # Change amount (-0.05 to +0.05)
    
    # Rationale
    composite_score: float
    confidence: float
    primary_reason: str
    
    # Constraints
    max_position: float = 0.60
    min_position: float = 0.05
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioRecommendation:
    """Full portfolio recommendation from signal integration."""
    timestamp: str
    
    # Current vs recommended
    current_allocation: Dict[str, float]
    recommended_allocation: Dict[str, float]
    deltas: List[AllocationDelta]
    
    # Overall metrics
    composite_sentiment: str  # bullish, bearish, neutral
    confidence: float
    regime: str
    
    # Risk metrics
    expected_volatility: Optional[float] = None
    max_drawdown_estimate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "composite_sentiment": self.composite_sentiment,
            "confidence": round(self.confidence, 4),
            "regime": self.regime,
            "current_allocation": self.current_allocation,
            "recommended_allocation": self.recommended_allocation,
            "deltas": [d.to_dict() for d in self.deltas],
            "expected_volatility": self.expected_volatility,
            "max_drawdown_estimate": self.max_drawdown_estimate,
        }


# ---------------------------------------------------------------------------
# Database Setup
# ---------------------------------------------------------------------------

def init_database():
    """Initialize SQLite database for signal history and accuracy tracking."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Signal history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signal_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_name TEXT NOT NULL,
            signal REAL,
            confidence REAL,
            raw_score REAL,
            raw_unit TEXT,
            historical_accuracy REAL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Composite signals table
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS composite_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            composite_score REAL,
            composite_confidence REAL,
            detected_regime TEXT,
            weights_used TEXT,
            primary_drivers TEXT,
            signal_agreement TEXT,
            expected_accuracy REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, timestamp)
        )
    """)
    
    # Accuracy tracking table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signal_accuracy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_name TEXT NOT NULL,
            prediction_timestamp TEXT,
            predicted_signal REAL,
            actual_return REAL,
            horizon_days INTEGER,
            accuracy_score REAL,
            error REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Portfolio recommendations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            current_allocation TEXT,
            recommended_allocation TEXT,
            composite_sentiment TEXT,
            confidence REAL,
            regime TEXT,
            deltas TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_signal_history_ticker_ts 
        ON signal_history(ticker, timestamp)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_accuracy_source 
        ON signal_accuracy(ticker, source_type, source_name)
    """)
    
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# SignalSource Base Class
# ---------------------------------------------------------------------------

class SignalSource(ABC):
    """Abstract base class for all signal sources."""
    
    def __init__(self, source_type: str, source_name: str):
        self.source_type = source_type
        self.source_name = source_name
        self.db_path = DB_PATH
        init_database()
    
    @abstractmethod
    def generate_signal(self, ticker: str) -> Optional[SignalSourceResult]:
        """Generate a signal for the given ticker."""
        pass
    
    @abstractmethod
    def get_historical_accuracy(self, ticker: str, horizon_days: int = 21) -> Optional[float]:
        """Get historical prediction accuracy for this source."""
        pass
    
    def _normalize_signal(self, raw_score: float, 
                          min_expected: float = -1.0, 
                          max_expected: float = 1.0) -> float:
        """Normalize raw score to [-1, 1] range."""
        if max_expected == min_expected:
            return 0.0
        
        normalized = 2 * (raw_score - min_expected) / (max_expected - min_expected) - 1
        return max(SIGNAL_MIN, min(SIGNAL_MAX, normalized))
    
    def _store_signal(self, ticker: str, result: SignalSourceResult):
        """Store signal in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO signal_history 
            (ticker, timestamp, source_type, source_name, signal, confidence,
             raw_score, raw_unit, historical_accuracy, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, result.timestamp, self.source_type, self.source_name,
            result.signal, result.confidence, result.raw_score, result.raw_unit,
            result.historical_accuracy, json.dumps(result.metadata)
        ))
        
        conn.commit()
        conn.close()


# ---------------------------------------------------------------------------
# Technical Signal Implementation
# ---------------------------------------------------------------------------

class TechnicalSignal(SignalSource):
    """
    Technical analysis signal source.
    
    Generates signals from:
    - Dual momentum (price vs 200-day SMA, 12-month returns)
    - Mean reversion (RSI, Bollinger Bands)
    - Trend strength (ADX)
    """
    
    def __init__(self):
        super().__init__("momentum", "technical")
        self.market_db = DATA_DIR / "market.db"
    
    def generate_signal(self, ticker: str) -> Optional[SignalSourceResult]:
        """Generate technical signal for ticker."""
        momentum_score = self._calculate_momentum(ticker)
        mean_reversion_score = self._calculate_mean_reversion(ticker)
        
        if momentum_score is None:
            return None
        
        # Combine signals (70% momentum, 30% mean reversion)
        combined = momentum_score["score"] * 0.7 + mean_reversion_score * 0.3
        
        # Confidence based on trend strength and data quality
        confidence = min(0.9, 0.5 + momentum_score["trend_strength"] * 0.4)
        
        result = SignalSourceResult(
            source_type=self.source_type,
            source_name=self.source_name,
            signal=round(max(SIGNAL_MIN, min(SIGNAL_MAX, combined)), 4),
            confidence=round(confidence, 4),
            raw_score=combined,
            raw_unit="composite_zscore",
            historical_accuracy=self.get_historical_accuracy(ticker),
            sample_count=momentum_score.get("sample_count", 0),
            metadata={
                "momentum_12m": momentum_score.get("return_12m", 0),
                "above_sma_200": momentum_score.get("above_sma", False),
                "mean_reversion": mean_reversion_score,
                "trend_strength": momentum_score.get("trend_strength", 0),
            }
        )
        
        self._store_signal(ticker, result)
        return result
    
    def _calculate_momentum(self, ticker: str) -> Optional[Dict]:
        """Calculate dual momentum metrics."""
        if not self.market_db.exists():
            return None
        
        conn = sqlite3.connect(self.market_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT date, close FROM prices 
            WHERE symbol = ? 
            AND date >= date('now', '-400 days')
            ORDER BY date DESC
        """, (ticker,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 200:
            return None
        
        closes = [r[1] for r in reversed(rows)]
        
        current = closes[-1]
        sma_200 = statistics.mean(closes[-200:])
        
        # 12-month return (approx 252 trading days)
        return_12m = (current / closes[-min(252, len(closes)-1)]) - 1 if len(closes) > 252 else 0
        
        # 6-month return
        return_6m = (current / closes[-min(126, len(closes)-1)]) - 1 if len(closes) > 126 else 0
        
        # Above SMA check
        above_sma = current > sma_200
        
        # Trend strength (simple ADX proxy)
        daily_changes = [abs(closes[i] - closes[i-1]) for i in range(-20, 0)]
        trend_strength = min(1.0, statistics.mean(daily_changes) / current * 100) if daily_changes else 0
        
        # Composite score
        if above_sma and return_12m > 0:
            score = min(1.0, return_12m * 2 + return_6m)
        elif above_sma:
            score = return_6m  # Slightly bullish
        elif return_12m < -0.15:
            score = max(-1.0, -0.5 + return_12m * 2)
        else:
            score = return_12m * 1.5  # Mild bearish
        
        return {
            "score": score,
            "return_12m": return_12m,
            "return_6m": return_6m,
            "above_sma": above_sma,
            "trend_strength": trend_strength,
            "sample_count": len(closes),
        }
    
    def _calculate_mean_reversion(self, ticker: str) -> float:
        """Calculate mean reversion signal (RSI proxy)."""
        if not self.market_db.exists():
            return 0.0
        
        conn = sqlite3.connect(self.market_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT close FROM prices 
            WHERE symbol = ? 
            AND date >= date('now', '-30 days')
            ORDER BY date DESC
        """, (ticker,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 14:
            return 0.0
        
        closes = [r[0] for r in reversed(rows)]
        
        # Simple RSI-like calculation
        gains = []
        losses = []
        for i in range(1, min(15, len(closes))):
            change = (closes[i] - closes[i-1]) / closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = statistics.mean(gains) if gains else 0
        avg_loss = statistics.mean(losses) if losses else 0.0001
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Convert RSI to signal (oversold = bullish, overbought = bearish)
        if rsi < 30:
            return min(1.0, (30 - rsi) / 30)  # Bullish mean reversion
        elif rsi > 70:
            return max(-1.0, -(rsi - 70) / 30)  # Bearish mean reversion
        else:
            return 0.0  # Neutral
    
    def get_historical_accuracy(self, ticker: str, horizon_days: int = 21) -> Optional[float]:
        """Get historical accuracy from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT accuracy_score FROM signal_accuracy
            WHERE ticker = ? AND source_type = ? AND source_name = ?
            AND horizon_days = ?
            ORDER BY prediction_timestamp DESC
            LIMIT 50
        """, (ticker, self.source_type, self.source_name, horizon_days))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return None
        
        # Return average accuracy
        accuracies = [r[0] for r in rows if r[0] is not None]
        return statistics.mean(accuracies) if accuracies else None


# ---------------------------------------------------------------------------
