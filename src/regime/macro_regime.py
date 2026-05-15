"""
Macro Regime Synthesis Engine (v4.30 Phase 1-2)

Integrates 9 macro signals into unified regime classifier:
- Fed policy (v2.54)
- Yield curve (v2.17)
- Credit spreads (v3.14)
- FX carry (v3.19)
- Commodity curve (v3.20)
- Bond momentum (v3.30)
- International equity (v3.13)
- VPIN microstructure (v2.65)
- Equity TSMOM (v2.52)

Target: +0.03 Sharpe improvement through improved regime detection (78% → 85%+).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
from pathlib import Path


class MacroRegime(Enum):
    """Six-state macro regime classification."""
    RISK_ON_GROWTH = "risk_on_growth"      # Fed easing, steep curve, low credit
    RISK_ON_LATE = "risk_on_late"          # Fed tightening, flat curve, low credit
    NEUTRAL = "neutral"                    # Mixed signals, no strong regime
    RISK_OFF_ROTATION = "risk_off_rotation" # Credit widening, FX unwind
    DEFENSIVE = "defensive"                # Inverted curve, flight to quality
    CRISIS = "crisis"                      # Multiple risk-off signals


class SignalState(Enum):
    """Standardized signal states mapped to [-1, 0, +1]."""
    NEGATIVE = -1   # Risk-off, tightening, distress
    NEUTRAL = 0     # Neutral, normal, mixed
    POSITIVE = 1    # Risk-on, easing, expansion


@dataclass
class SignalInput:
    """Individual signal input with metadata."""
    name: str
    state: SignalState
    raw_value: float
    confidence: float  # 0-100 based on data quality
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass
class RegimeClassification:
    """Complete regime classification result."""
    timestamp: datetime
    regime: MacroRegime
    confidence: float  # 0-100%
    signal_agreement: float  # How aligned are signals
    signal_strength: float  # Magnitude of weighted sum
    weighted_sum: float  # [-1, +1] scale
    signal_breakdown: Dict[str, float]  # Individual signal contributions
    regime_duration_days: int  # Days in current regime
    recommended_action: str  # Portfolio adjustment recommendation
    

class MacroRegimeSynthesizer:
    """
    Cross-asset macro regime synthesizer integrating 9 signals.
    
    Architecture:
    1. Signal harmonization: Map all to [-1, 0, +1]
    2. Weighted voting: Historical accuracy-based weights
    3. Regime classification: 6-state model
    4. Confidence scoring: Based on signal agreement
    5. Tactical shifts: Dynamic allocation recommendations
    """
    
    # Signal registry with metadata
    SIGNAL_REGISTRY = {
        "fed_policy": {
            "display_name": "Fed Policy",
            "mapping": {"easing": 1, "neutral": 0, "tightening": -1},
            "max_weight": 0.25,
            "min_weight": 0.05,
        },
        "yield_curve": {
            "display_name": "Yield Curve",
            "mapping": {"steep": 1, "flat": 0, "inverted": -1},
            "max_weight": 0.20,
            "min_weight": 0.05,
        },
        "credit_spread": {
            "display_name": "Credit Spreads",
            "mapping": {"normal": 1, "elevated": 0, "distressed": -1},
            "max_weight": 0.20,
            "min_weight": 0.05,
        },
        "fx_carry": {
            "display_name": "FX Carry",
            "mapping": {"safe": 1, "unwind_risk": -1},
            "max_weight": 0.15,
            "min_weight": 0.05,
        },
        "commodity_curve": {
            "display_name": "Commodity Curve",
            "mapping": {"backwardation": 1, "contango": -1},
            "max_weight": 0.10,
            "min_weight": 0.05,
        },
        "bond_momentum": {
            "display_name": "Bond Momentum",
            "mapping": {"shy": -1, "ief": 0, "tlt": 1},
            "max_weight": 0.15,
            "min_weight": 0.05,
        },
        "intl_equity": {
            "display_name": "International Equity",
            "mapping": {"downtrend": -1, "mixed": 0, "uptrend": 1},
            "max_weight": 0.10,
            "min_weight": 0.05,
        },
        "vpin": {
            "display_name": "VPIN Microstructure",
            "mapping": {"normal": 0, "toxic": -1},
            "max_weight": 0.10,
            "min_weight": 0.05,
        },
        "equity_tsmom": {
            "display_name": "Equity TSMOM",
            "mapping": {"risk_off": -1, "risk_on": 1},
            "max_weight": 0.20,
            "min_weight": 0.05,
        },
    }
    
    # Regime state definitions (signal combinations)
    REGIME_DEFINITIONS = {
        MacroRegime.RISK_ON_GROWTH: {
            "description": "Fed easing, steep curve, low credit spreads",
            "primary_signals": ["fed_policy", "yield_curve", "credit_spread"],
            "conditions": {
                "fed_policy": 1,      # Easing
                "yield_curve": 1,     # Steep
                "credit_spread": 1,   # Normal/Tight
            },
            "min_match": 2,  # Need at least 2 of 3 primary signals
        },
        MacroRegime.RISK_ON_LATE: {
            "description": "Fed tightening, flat curve, low credit spreads",
            "primary_signals": ["fed_policy", "yield_curve", "credit_spread"],
            "conditions": {
                "fed_policy": -1,     # Tightening
                "yield_curve": 0,     # Flat
                "credit_spread": 1,   # Normal
            },
            "min_match": 2,
        },
        MacroRegime.NEUTRAL: {
            "description": "Mixed signals, no strong macro bias",
            "primary_signals": [],
            "conditions": {},
            "min_match": 0,  # Default when no strong regime
        },
        MacroRegime.RISK_OFF_ROTATION: {
            "description": "Credit widening, FX unwind risk",
            "primary_signals": ["credit_spread", "fx_carry"],
            "conditions": {
                "credit_spread": 0,   # Elevated
                "fx_carry": -1,       # Unwind risk
            },
            "min_match": 1,
        },
        MacroRegime.DEFENSIVE: {
            "description": "Inverted curve, flight to quality",
            "primary_signals": ["yield_curve", "bond_momentum"],
            "conditions": {
                "yield_curve": -1,    # Inverted
                "bond_momentum": 1,   # TLT momentum (flight to quality)
            },
            "min_match": 1,
        },
        MacroRegime.CRISIS: {
            "description": "Multiple risk-off signals active",
            "primary_signals": ["fed_policy", "yield_curve", "credit_spread", "fx_carry", "vpin"],
            "conditions": {},
            "min_match": 4,  # 4+ negative signals
        },
    }
    
    # Portfolio allocation shifts by regime
    REGIME_SHIFTS = {
        MacroRegime.RISK_ON_GROWTH: {"spy": +0.10, "gld": -0.05, "tlt": -0.05},
        MacroRegime.RISK_ON_LATE: {"spy": +0.05, "gld": -0.02, "tlt": -0.03},
        MacroRegime.NEUTRAL: {"spy": 0.00, "gld": 0.00, "tlt": 0.00},
        MacroRegime.RISK_OFF_ROTATION: {"spy": -0.05, "gld": +0.03, "tlt": +0.02},
        MacroRegime.DEFENSIVE: {"spy": -0.10, "gld": +0.05, "tlt": +0.05},
        MacroRegime.CRISIS: {"spy": -0.15, "gld": +0.08, "tlt": +0.07},
    }
    
    def __init__(self, db_path: Optional[str] = None, weights: Optional[Dict[str, float]] = None):
        """
        Initialize synthesizer with optional database and weights.
        
        Args:
            db_path: Path to SQLite DB for storing history
            weights: Optional custom signal weights (recalculated if None)
        """
        self.db_path = db_path or "data/macro_regime_history.db"
        self.weights = weights or self._calculate_default_weights()
        self.current_regime: Optional[MacroRegime] = None
        self.regime_start_date: Optional[datetime] = None
        self.history: List[RegimeClassification] = []
        
        self._init_database()
        self._load_last_regime()
    
    def _init_database(self):
        """Initialize SQLite database for regime history."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS regime_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                regime TEXT NOT NULL,
                confidence REAL NOT NULL,
                signal_agreement REAL NOT NULL,
                signal_strength REAL NOT NULL,
                weighted_sum REAL NOT NULL,
                signal_breakdown TEXT NOT NULL,
                duration_days INTEGER NOT NULL,
                recommended_action TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON regime_history(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_last_regime(self):
        """Load the most recent regime classification."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, regime FROM regime_history
            ORDER BY timestamp DESC LIMIT 1
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            self.current_regime = MacroRegime(result[1])
            self.regime_start_date = datetime.fromisoformat(result[0])
    
    def _calculate_default_weights(self) -> Dict[str, float]:
        """Calculate default signal weights based on historical importance."""
        # Base weights (sum to 1.0)
        base_weights = {
            "fed_policy": 0.15,
            "yield_curve": 0.15,
            "credit_spread": 0.15,
            "equity_tsmom": 0.15,
            "bond_momentum": 0.10,
            "fx_carry": 0.10,
            "intl_equity": 0.08,
            "commodity_curve": 0.07,
            "vpin": 0.05,
        }
        return base_weights
    
    def update_weights_from_accuracy(self, 
                                     accuracy_history: Dict[str, float],
                                     temperature: float = 0.5):
        """
        Update signal weights based on historical accuracy.
        
        Args:
            accuracy_history: Dict mapping signal name to accuracy (0-1)
            temperature: Softmax temperature (lower = more aggressive weighting)
        """
        if not accuracy_history:
            return
        
        # Filter to known signals
        accuracies = {k: v for k, v in accuracy_history.items() 
                     if k in self.SIGNAL_REGISTRY}
        
        if len(accuracies) < 3:
            # Not enough data to recalibrate
            return
        
        # Softmax weighting
        exp_accuracies = {k: np.exp(v / temperature) for k, v in accuracies.items()}
        total = sum(exp_accuracies.values())
        
        new_weights = {k: v / total for k, v in exp_accuracies.items()}
        
        # Apply min/max constraints
        for signal, config in self.SIGNAL_REGISTRY.items():
            if signal in new_weights:
                new_weights[signal] = max(
                    config["min_weight"],
                    min(config["max_weight"], new_weights[signal])
                )
        
        # Renormalize
        total = sum(new_weights.values())
        self.weights = {k: v / total for k, v in new_weights.items()}
    
    def harmonize_signal(self, signal_name: str, raw_state: str) -> SignalState:
        """
        Map raw signal state to standardized [-1, 0, +1].
        
        Args:
            signal_name: Signal identifier
            raw_state: Raw state string from signal source
            
        Returns:
            Standardized SignalState
        """
        if signal_name not in self.SIGNAL_REGISTRY:
            return SignalState.NEUTRAL
        
        mapping = self.SIGNAL_REGISTRY[signal_name]["mapping"]
        
        # Normalize raw state
        raw_lower = raw_state.lower().replace(" ", "_")
        
        # Direct mapping
        if raw_lower in mapping:
            value = mapping[raw_lower]
            return SignalState(value)
        
        # Alternative mappings
        if raw_lower in ["positive", "bullish", "expansion", "up"]:
            return SignalState.POSITIVE
        elif raw_lower in ["negative", "bearish", "contraction", "down"]:
            return SignalState.NEGATIVE
        elif raw_lower in ["mixed", "uncertain", "transition"]:
            return SignalState.NEUTRAL
        
        return SignalState.NEUTRAL
    
    def calculate_confidence(self, 
                           signals: Dict[str, SignalInput],
                           weighted_sum: float) -> Tuple[float, float]:
        """
        Calculate confidence score based on signal agreement and strength.
        
        Args:
            signals: Dict of signal inputs
            weighted_sum: Weighted sum of signals [-1, +1]
            
        Returns:
            (confidence_pct, agreement_score)
        """
        if not signals:
            return 0.0, 0.0
        
        # Signal agreement: how aligned are signals with weighted sum direction
        weighted_abs = sum(
            abs(s.state.value) * self.weights.get(name, 0)
            for name, s in signals.items()
        )
        
        total_weight = sum(
            self.weights.get(name, 0) 
            for name in signals.keys()
        )
        
        if total_weight == 0 or weighted_abs == 0:
            return 0.0, 0.0
        
        # Agreement = weighted absolute sum / total weight
        agreement = weighted_abs / total_weight
        
        # Strength = absolute weighted sum / weighted absolute sum
        strength = abs(weighted_sum) / weighted_abs
        
        # Confidence = agreement * strength, scaled to 0-100
        confidence = agreement * strength * 100
        
        return min(confidence, 100), agreement
    
    def classify_regime(self, 
                       signals: Dict[str, SignalInput],
                       min_confidence: float = 60.0) -> RegimeClassification:
        """
        Classify macro regime from signal inputs.
        
        Args:
            signals: Dict mapping signal name to SignalInput
            min_confidence: Minimum confidence to override neutral
            
        Returns:
            RegimeClassification with regime, confidence, and recommendations
        """
        # Calculate weighted sum
        weighted_sum = 0.0
        signal_breakdown = {}
        
        for signal_name, signal_input in signals.items():
            weight = self.weights.get(signal_name, 0)
            contribution = signal_input.state.value * weight
            weighted_sum += contribution
            signal_breakdown[signal_name] = {
                "state": signal_input.state.value,
                "weight": weight,
                "contribution": contribution,
                "confidence": signal_input.confidence,
            }
        
        # Calculate confidence and agreement
        confidence, agreement = self.calculate_confidence(signals, weighted_sum)
        
        # Determine regime
        regime = self._match_regime(signals, weighted_sum, confidence, min_confidence)
        
        # Calculate regime duration
        if regime == self.current_regime:
            duration = (datetime.now() - (self.regime_start_date or datetime.now())).days
        else:
            duration = 0
            self.current_regime = regime
            self.regime_start_date = datetime.now()
        
        # Generate recommendation
        recommendation = self._generate_recommendation(regime, confidence)
        
        classification = RegimeClassification(
            timestamp=datetime.now(),
            regime=regime,
            confidence=confidence,
            signal_agreement=agreement,
            signal_strength=abs(weighted_sum),
            weighted_sum=weighted_sum,
            signal_breakdown=signal_breakdown,
            regime_duration_days=duration,
            recommended_action=recommendation,
        )
        
        return classification
    
    def _match_regime(self,
                     signals: Dict[str, SignalInput],
                     weighted_sum: float,
                     confidence: float,
                     min_confidence: float) -> MacroRegime:
        """
        Match signal pattern to regime definition.
        
        Priority:
        1. Crisis: 4+ negative signals
        2. Check specific regime patterns
        3. Weighted sum direction for neutral cases
        """
        # Check crisis condition first
        negative_count = sum(
            1 for s in signals.values() 
            if s.state == SignalState.NEGATIVE
        )
        if negative_count >= 4:
            return MacroRegime.CRISIS
        
        # Check specific regime patterns
        for regime, definition in self.REGIME_DEFINITIONS.items():
            if regime == MacroRegime.NEUTRAL:
                continue
            
            matches = 0
            for signal_name, expected_state in definition["conditions"].items():
                if signal_name in signals:
                    if signals[signal_name].state.value == expected_state:
                        matches += 1
            
            if matches >= definition["min_match"]:
                # Found a match - check confidence
                if confidence >= min_confidence:
                    return regime
                else:
                    # Low confidence - stay neutral
                    return MacroRegime.NEUTRAL
        
        # Default: use weighted sum direction
        if confidence < min_confidence:
            return MacroRegime.NEUTRAL
        elif weighted_sum > 0.3:
            return MacroRegime.RISK_ON_GROWTH
        elif weighted_sum < -0.3:
            return MacroRegime.DEFENSIVE
        else:
            return MacroRegime.NEUTRAL
    
    def _generate_recommendation(self, 
                                regime: MacroRegime, 
                                confidence: float) -> str:
        """Generate portfolio action recommendation."""
        shifts = self.REGIME_SHIFTS.get(regime, {})
        
        if confidence < 60:
            return "HOLD: Insufficient confidence for regime shift"
        
        action_parts = []
        for asset, shift in shifts.items():
            if abs(shift) >= 0.05:
                direction = "+" if shift > 0 else ""
                action_parts.append(f"{asset.upper()}{direction}{shift*100:.0f}%")
        
        if not action_parts:
            return "HOLD: Maintain base allocation"
        
        regime_name = regime.value.replace("_", " ").upper()
        return f"{regime_name}: {', '.join(action_parts)}"
    
    def get_allocation_overlay(self, 
                              regime: MacroRegime,
                              confidence: float,
                              base_allocation: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate allocation overlay for a given regime.
        
        Args:
            regime: Current macro regime
            confidence: Classification confidence (0-100)
            base_allocation: Base portfolio allocation (e.g., 46/38/16)
            
        Returns:
            Adjusted allocation dict
        """
        if confidence < 60:
            # Low confidence - no changes
            return base_allocation.copy()
        
        # Scale shifts by confidence (max at 90%+ confidence)
        scale = min(1.0, confidence / 90.0)
        
        shifts = self.REGIME_SHIFTS.get(regime, {})
        
        adjusted = base_allocation.copy()
        for asset, shift in shifts.items():
            if asset in adjusted:
                adjusted[asset] += shift * scale
        
        # Ensure no negative allocations
        adjusted = {k: max(0.0, v) for k, v in adjusted.items()}
        
        # Renormalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    def persist_classification(self, classification: RegimeClassification):
        """Save classification to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO regime_history 
            (timestamp, regime, confidence, signal_agreement, signal_strength,
             weighted_sum, signal_breakdown, duration_days, recommended_action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            classification.timestamp.isoformat(),
            classification.regime.value,
            classification.confidence,
            classification.signal_agreement,
            classification.signal_strength,
            classification.weighted_sum,
            json.dumps(classification.signal_breakdown),
            classification.regime_duration_days,
            classification.recommended_action,
        ))
        
        conn.commit()
        conn.close()
        
        self.history.append(classification)
    
    def get_regime_history(self, days: int = 90) -> List[Dict]:
        """Get recent regime history from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT timestamp, regime, confidence, weighted_sum, recommended_action
            FROM regime_history
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        ''', (cutoff,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "timestamp": r[0],
                "regime": r[1],
                "confidence": r[2],
                "weighted_sum": r[3],
                "recommendation": r[4],
            }
            for r in results
        ]
    
    def to_dict(self, classification: RegimeClassification) -> Dict:
        """Convert classification to dictionary for serialization."""
        return {
            "timestamp": classification.timestamp.isoformat(),
            "regime": classification.regime.value,
            "regime_display": classification.regime.value.replace("_", " ").title(),
            "confidence": round(classification.confidence, 2),
            "signal_agreement": round(classification.signal_agreement, 3),
            "signal_strength": round(classification.signal_strength, 3),
            "weighted_sum": round(classification.weighted_sum, 3),
            "regime_duration_days": classification.regime_duration_days,
            "recommended_action": classification.recommended_action,
            "allocation_shifts": self.REGIME_SHIFTS.get(classification.regime, {}),
            "signal_breakdown": classification.signal_breakdown,
        }


class SignalRegistry:
    """
    Registry for signal metadata and harmonization rules.
    
    Provides centralized signal definitions for consistency
    across the macro regime synthesis system.
    """
    
    @classmethod
    def get_signal_metadata(cls, signal_name: str) -> Optional[Dict]:
        """Get metadata for a signal."""
        return MacroRegimeSynthesizer.SIGNAL_REGISTRY.get(signal_name)
    
    @classmethod
    def list_signals(cls) -> List[str]:
        """List all registered signal names."""
        return list(MacroRegimeSynthesizer.SIGNAL_REGISTRY.keys())
    
    @classmethod
    def validate_signal_input(cls, signal_name: str, raw_state: str) -> bool:
        """Validate if a signal input is recognized."""
        if signal_name not in MacroRegimeSynthesizer.SIGNAL_REGISTRY:
            return False
        
        registry = MacroRegimeSynthesizer.SIGNAL_REGISTRY[signal_name]
        mapping = registry["mapping"]
        
        return raw_state.lower() in mapping or raw_state.lower() in [
            "positive", "negative", "neutral", "bullish", "bearish", "mixed"
        ]


# Convenience function for standalone usage
def create_default_synthesizer(db_path: Optional[str] = None) -> MacroRegimeSynthesizer:
    """Create synthesizer with default configuration."""
    return MacroRegimeSynthesizer(db_path=db_path)


def classify_current_regime(signals: Dict[str, str], 
                           db_path: Optional[str] = None) -> Dict:
    """
    One-shot function to classify regime from raw signal states.
    
    Args:
        signals: Dict mapping signal name to raw state string
        db_path: Optional database path for persistence
        
    Returns:
        Classification result as dictionary
    """
    synthesizer = create_default_synthesizer(db_path)
    
    # Harmonize inputs
    harmonized = {}
    for signal_name, raw_state in signals.items():
        state = synthesizer.harmonize_signal(signal_name, raw_state)
        harmonized[signal_name] = SignalInput(
            name=signal_name,
            state=state,
            raw_value=0.0,  # Not used in this simplified interface
            confidence=80.0,  # Default confidence
            timestamp=datetime.now(),
        )
    
    # Classify
    classification = synthesizer.classify_regime(harmonized)
    
    # Persist if database configured
    if db_path:
        synthesizer.persist_classification(classification)
    
    return synthesizer.to_dict(classification)
