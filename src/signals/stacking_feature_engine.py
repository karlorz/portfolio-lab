#!/usr/bin/env python3
"""
Portfolio-Lab v3.10 Phase 1: Stacking Feature Engineering

Generates 102-dimensional feature vectors for XGBoost meta-learner from base signals:
- 8 base signal values
- 84 pairwise interaction features (28 pairs × 3 types: multiplicative, disagreement, average)
- 2 regime context features (VIX normalized, trend strength)
- 8 historical accuracy features (90-day rolling)

Usage:
    from src.signals.stacking_feature_engine import StackingFeatureEngine
    
    engine = StackingFeatureEngine()
    features = engine.create_features(signals, regime_context, historical_accuracy)
    
    # Or standalone
    python -m src.signals.stacking_feature_engine --test

Performance:
- Feature generation latency: <10ms for 8 signals
- Memory footprint: ~50KB per feature vector
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
from enum import Enum
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class SignalSource(Enum):
    """Signal source identifiers for stacking features."""
    TSFM_MOMENTUM = "tsfm_momentum"
    HMM_REGIME = "hmm_regime"
    CTA_TREND = "cta_trend"
    MACRO_MOMENTUM = "macro_momentum"
    MULTI_SPEED_MOM = "multi_speed_momentum"
    DURATION_REGIME = "duration_regime"
    CIRCUIT_BREAKER = "circuit_breaker"
    FACTOR_ROTATION = "factor_rotation"


@dataclass
class Signal:
    """Base signal reading."""
    source: SignalSource
    value: float  # -1 to +1
    timestamp: datetime
    confidence: float  # 0 to 1


@dataclass
class RegimeContext:
    """Market regime context features."""
    vix_level: float
    trend_strength: float
    timestamp: datetime


@dataclass
class HistoricalAccuracy:
    """Rolling historical accuracy for each signal source."""
    source: SignalSource
    accuracy_90d: float  # 0 to 1
    predictions_count: int
    timestamp: datetime


@dataclass  
class FeatureVector:
    """Complete 102-dimensional feature vector."""
    # Base signals (8)
    base_values: Dict[SignalSource, float]
    
    # Pairwise interactions (84)
    multiplicative: Dict[Tuple[SignalSource, SignalSource], float]  # 28
    disagreement: Dict[Tuple[SignalSource, SignalSource], float]     # 28
    averages: Dict[Tuple[SignalSource, SignalSource], float]        # 28
    
    # Regime context (2)
    vix_normalized: float
    trend_strength: float
    
    # Historical accuracy (8)
    accuracy_values: Dict[SignalSource, float]
    
    # Metadata
    timestamp: datetime
    dimension_count: int = 102


class StackingFeatureEngine:
    """
    Generate 102-dimensional feature vectors for XGBoost meta-learner.
    
    Features:
    - Base signal values (8)
    - Pairwise multiplicative interactions (28)
    - Pairwise disagreement features (28)
    - Pairwise average features (28)
    - Regime context (2)
    - Historical accuracy (8)
    
    Total: 102 features
    """
    
    NUM_BASE_SIGNALS = 8
    NUM_PAIRWISE_COMBINATIONS = 28  # C(8,2) = 28
    NUM_REGIME_FEATURES = 2
    NUM_ACCURACY_FEATURES = 8
    TOTAL_DIMENSIONS = 102
    
    def __init__(self, vix_normalization_factor: float = 30.0):
        """
        Initialize feature engine.
        
        Args:
            vix_normalization_factor: Divisor for VIX normalization (default 30.0)
        """
        self.vix_normalization_factor = vix_normalization_factor
        self._pairwise_cache: Optional[List[Tuple[SignalSource, SignalSource]]] = None
    
    def _get_pairwise_combinations(self, sources: List[SignalSource]) -> List[Tuple[SignalSource, SignalSource]]:
        """Get all pairwise combinations of signal sources."""
        if self._pairwise_cache is None:
            self._pairwise_cache = list(combinations(sources, 2))
        return self._pairwise_cache
    
    def create_features(
        self,
        signals: Dict[SignalSource, Signal],
        regime_context: RegimeContext,
        historical_accuracy: Dict[SignalSource, HistoricalAccuracy]
    ) -> FeatureVector:
        """
        Generate complete 102-dimensional feature vector.
        
        Args:
            signals: Dictionary of Signal objects keyed by SignalSource
            regime_context: Current market regime context
            historical_accuracy: Dictionary of HistoricalAccuracy by SignalSource
            
        Returns:
            FeatureVector with all 102 features computed
            
        Raises:
            ValueError: If not all 8 signal sources are provided
        """
        # Validate input
        if len(signals) != self.NUM_BASE_SIGNALS:
            missing = set(SignalSource) - set(signals.keys())
            raise ValueError(f"Expected {self.NUM_BASE_SIGNALS} signals, got {len(signals)}. Missing: {missing}")
        
        # Base signal values (8 features)
        base_values = {source: signal.value for source, signal in signals.items()}
        
        # Pairwise combinations
        sources = list(signals.keys())
        pairs = self._get_pairwise_combinations(sources)
        
        # Pairwise interaction features (84 features = 28 pairs × 3 types)
        multiplicative = {}
        disagreement = {}
        averages = {}
        
        for s1, s2 in pairs:
            v1 = signals[s1].value
            v2 = signals[s2].value
            
            # Multiplicative interaction (captures synergy/antagonism)
            multiplicative[(s1, s2)] = v1 * v2
            
            # Disagreement (absolute difference, captures conflict)
            disagreement[(s1, s2)] = abs(v1 - v2)
            
            # Average (mean prediction, simple consensus)
            averages[(s1, s2)] = (v1 + v2) / 2.0
        
        # Regime context features (2 features)
        vix_normalized = regime_context.vix_level / self.vix_normalization_factor
        trend_strength = regime_context.trend_strength
        
        # Historical accuracy features (8 features)
        accuracy_values = {
            source: hist.accuracy_90d 
            for source, hist in historical_accuracy.items()
        }
        
        return FeatureVector(
            base_values=base_values,
            multiplicative=multiplicative,
            disagreement=disagreement,
            averages=averages,
            vix_normalized=vix_normalized,
            trend_strength=trend_strength,
            accuracy_values=accuracy_values,
            timestamp=datetime.now()
        )
    
    def to_numpy(self, feature_vector: FeatureVector) -> np.ndarray:
        """
        Convert FeatureVector to numpy array for XGBoost inference.
        
        Returns:
            numpy array of shape (102,) with all features concatenated
            Order: base (8) + multiplicative (28) + disagreement (28) + 
                   averages (28) + regime (2) + accuracy (8)
        """
        features = []
        
        # Base signals in fixed order
        for source in SignalSource:
            features.append(feature_vector.base_values[source])
        
        # Pairwise features in fixed order (sorted by source enum value)
        pairs = sorted(feature_vector.multiplicative.keys(), 
                      key=lambda x: (x[0].value, x[1].value))
        
        for pair in pairs:
            features.append(feature_vector.multiplicative[pair])
        
        for pair in pairs:
            features.append(feature_vector.disagreement[pair])
        
        for pair in pairs:
            features.append(feature_vector.averages[pair])
        
        # Regime context
        features.append(feature_vector.vix_normalized)
        features.append(feature_vector.trend_strength)
        
        # Historical accuracy in fixed order
        for source in SignalSource:
            features.append(feature_vector.accuracy_values[source])
        
        return np.array(features, dtype=np.float32)
    
    def to_dict(self, feature_vector: FeatureVector) -> Dict:
        """Convert FeatureVector to dictionary for JSON serialization."""
        return {
            "base_values": {k.value: v for k, v in feature_vector.base_values.items()},
            "multiplicative": {f"{k[0].value}_{k[1].value}": v 
                              for k, v in feature_vector.multiplicative.items()},
            "disagreement": {f"{k[0].value}_{k[1].value}": v 
                            for k, v in feature_vector.disagreement.items()},
            "averages": {f"{k[0].value}_{k[1].value}": v 
                        for k, v in feature_vector.averages.items()},
            "vix_normalized": feature_vector.vix_normalized,
            "trend_strength": feature_vector.trend_strength,
            "accuracy_values": {k.value: v for k, v in feature_vector.accuracy_values.items()},
            "timestamp": feature_vector.timestamp.isoformat(),
            "dimension_count": feature_vector.dimension_count
        }
    
    def get_feature_names(self) -> List[str]:
        """
        Get ordered list of feature names matching numpy array order.
        
        Returns:
            List of 102 feature name strings
        """
        names = []
        
        # Base signals
        for source in SignalSource:
            names.append(f"base_{source.value}")
        
        # Pairwise combinations for interaction features
        sources = list(SignalSource)
        pairs = self._get_pairwise_combinations(sources)
        pairs_sorted = sorted(pairs, key=lambda x: (x[0].value, x[1].value))
        
        # Multiplicative interactions
        for s1, s2 in pairs_sorted:
            names.append(f"mult_{s1.value}_{s2.value}")
        
        # Disagreement features
        for s1, s2 in pairs_sorted:
            names.append(f"disagree_{s1.value}_{s2.value}")
        
        # Average features
        for s1, s2 in pairs_sorted:
            names.append(f"avg_{s1.value}_{s2.value}")
        
        # Regime context
        names.append("vix_normalized")
        names.append("trend_strength")
        
        # Historical accuracy
        for source in SignalSource:
            names.append(f"acc90d_{source.value}")
        
        return names
    
    def explain_features(self, feature_vector: FeatureVector, top_n: int = 10) -> Dict:
        """
        Generate human-readable explanation of feature vector.
        
        Args:
            feature_vector: The feature vector to explain
            top_n: Number of top features to highlight
            
        Returns:
            Dictionary with explanation summary
        """
        explanations = {
            "timestamp": feature_vector.timestamp.isoformat(),
            "total_dimensions": self.TOTAL_DIMENSIONS,
            "base_signals_summary": {
                "mean": np.mean(list(feature_vector.base_values.values())),
                "std": np.std(list(feature_vector.base_values.values())),
                "bullish_count": sum(1 for v in feature_vector.base_values.values() if v > 0.1),
                "bearish_count": sum(1 for v in feature_vector.base_values.values() if v < -0.1),
                "neutral_count": sum(1 for v in feature_vector.base_values.values() if abs(v) <= 0.1)
            },
            "pairwise_interactions": {
                "high_synergy": sorted(
                    [(f"{k[0].value}-{k[1].value}", v) 
                     for k, v in feature_vector.multiplicative.items()],
                    key=lambda x: x[1], reverse=True
                )[:top_n],
                "high_disagreement": sorted(
                    [(f"{k[0].value}-{k[1].value}", v) 
                     for k, v in feature_vector.disagreement.items()],
                    key=lambda x: x[1], reverse=True
                )[:top_n]
            },
            "regime_context": {
                "vix_normalized": round(feature_vector.vix_normalized, 3),
                "vix_level": round(feature_vector.vix_normalized * self.vix_normalization_factor, 2),
                "trend_strength": round(feature_vector.trend_strength, 3),
                "volatility_regime": "high" if feature_vector.vix_normalized > 0.67 else 
                                   "elevated" if feature_vector.vix_normalized > 0.5 else "normal"
            },
            "historical_accuracy": {
                "mean_accuracy": round(np.mean(list(feature_vector.accuracy_values.values())), 3),
                "best_performer": max(feature_vector.accuracy_values.items(), key=lambda x: x[1])[0].value,
                "worst_performer": min(feature_vector.accuracy_values.items(), key=lambda x: x[1])[0].value
            }
        }
        
        return explanations


class StackingAccuracyTracker:
    """
    Track historical accuracy of base signals for feature engineering.
    
    Maintains rolling 90-day accuracy per signal source, updating
    based on actual forward returns.
    """
    
    def __init__(self, window_days: int = 90, db_path: Optional[Path] = None):
        """
        Initialize accuracy tracker.
        
        Args:
            window_days: Rolling window for accuracy calculation (default 90)
            db_path: Path to SQLite database for persistence (optional)
        """
        self.window_days = window_days
        self.db_path = db_path
        self._history: Dict[SignalSource, List[Tuple[datetime, float, bool]]] = {
            source: [] for source in SignalSource
        }
    
    def record_prediction(
        self,
        source: SignalSource,
        timestamp: datetime,
        signal_value: float,
        actual_return: float
    ) -> None:
        """
        Record a prediction outcome for accuracy tracking.
        
        Args:
            source: Signal source that made prediction
            timestamp: When prediction was made
            signal_value: Signal value (-1 to +1, positive = bullish)
            actual_return: Actual forward return (positive = up)
        """
        # Correct if signal direction matches return direction
        correct = (signal_value > 0 and actual_return > 0) or \
                 (signal_value < 0 and actual_return < 0) or \
                 (abs(signal_value) < 0.1 and abs(actual_return) < 0.01)  # neutral
        
        self._history[source].append((timestamp, signal_value, correct))
        
        # Trim old history
        cutoff = timestamp - timedelta(days=self.window_days)
        self._history[source] = [
            (t, v, c) for t, v, c in self._history[source] if t >= cutoff
        ]
    
    def get_historical_accuracy(
        self,
        source: SignalSource,
        as_of: datetime
    ) -> HistoricalAccuracy:
        """
        Get rolling historical accuracy for a signal source.
        
        Args:
            source: Signal source to query
            as_of: Timestamp for calculation
            
        Returns:
            HistoricalAccuracy with accuracy and prediction count
        """
        cutoff = as_of - timedelta(days=self.window_days)
        recent = [(t, v, c) for t, v, c in self._history[source] if t >= cutoff]
        
        if not recent:
            # Default to 0.5 (no information) if no history
            return HistoricalAccuracy(
                source=source,
                accuracy_90d=0.5,
                predictions_count=0,
                timestamp=as_of
            )
        
        accuracy = sum(1 for _, _, c in recent if c) / len(recent)
        
        return HistoricalAccuracy(
            source=source,
            accuracy_90d=accuracy,
            predictions_count=len(recent),
            timestamp=as_of
        )
    
    def get_all_accuracies(
        self,
        as_of: datetime
    ) -> Dict[SignalSource, HistoricalAccuracy]:
        """Get historical accuracy for all signal sources."""
        return {
            source: self.get_historical_accuracy(source, as_of)
            for source in SignalSource
        }


# ---------------------------------------------------------------------------
# CLI and Testing
# ---------------------------------------------------------------------------

def demo():
    """Demonstrate feature engineering with synthetic signals."""
    import time
    
    print("=" * 70)
    print("Portfolio-Lab v3.10: Stacking Feature Engine Demo")
    print("=" * 70)
    
    # Initialize engine
    engine = StackingFeatureEngine()
    tracker = StackingAccuracyTracker()
    
    # Create synthetic signals
    signals = {
        SignalSource.TSFM_MOMENTUM: Signal(
            source=SignalSource.TSFM_MOMENTUM,
            value=0.65,
            timestamp=datetime.now(),
            confidence=0.82
        ),
        SignalSource.HMM_REGIME: Signal(
            source=SignalSource.HMM_REGIME,
            value=0.42,
            timestamp=datetime.now(),
            confidence=0.71
        ),
        SignalSource.CTA_TREND: Signal(
            source=SignalSource.CTA_TREND,
            value=0.58,
            timestamp=datetime.now(),
            confidence=0.79
        ),
        SignalSource.MACRO_MOMENTUM: Signal(
            source=SignalSource.MACRO_MOMENTUM,
            value=0.31,
            timestamp=datetime.now(),
            confidence=0.65
        ),
        SignalSource.MULTI_SPEED_MOM: Signal(
            source=SignalSource.MULTI_SPEED_MOM,
            value=0.72,
            timestamp=datetime.now(),
            confidence=0.85
        ),
        SignalSource.DURATION_REGIME: Signal(
            source=SignalSource.DURATION_REGIME,
            value=-0.15,
            timestamp=datetime.now(),
            confidence=0.68
        ),
        SignalSource.CIRCUIT_BREAKER: Signal(
            source=SignalSource.CIRCUIT_BREAKER,
            value=0.95,
            timestamp=datetime.now(),
            confidence=0.91
        ),
        SignalSource.FACTOR_ROTATION: Signal(
            source=SignalSource.FACTOR_ROTATION,
            value=0.48,
            timestamp=datetime.now(),
            confidence=0.74
        )
    }
    
    # Add some mock historical accuracy data
    for source in SignalSource:
        for i in range(45):
            tracker.record_prediction(
                source=source,
                timestamp=datetime.now() - timedelta(days=i*2),
                signal_value=np.random.uniform(-0.8, 0.8),
                actual_return=np.random.uniform(-0.02, 0.02)
            )
    
    # Create regime context
    regime_context = RegimeContext(
        vix_level=18.5,
        trend_strength=0.67,
        timestamp=datetime.now()
    )
    
    # Get historical accuracies
    historical_accuracy = tracker.get_all_accuracies(datetime.now())
    
    print("\n1. Creating Feature Vector...")
    start = time.time()
    feature_vector = engine.create_features(signals, regime_context, historical_accuracy)
    elapsed_ms = (time.time() - start) * 1000
    
    print(f"   ✓ Feature vector created in {elapsed_ms:.2f}ms")
    print(f"   ✓ Dimensions: {feature_vector.dimension_count}")
    
    print("\n2. Converting to NumPy...")
    features_np = engine.to_numpy(feature_vector)
    print(f"   ✓ Shape: {features_np.shape}")
    print(f"   ✓ Dtype: {features_np.dtype}")
    print(f"   ✓ Memory: {features_np.nbytes} bytes")
    
    print("\n3. Feature Breakdown:")
    print(f"   - Base signals (8): mean={np.mean(list(feature_vector.base_values.values())):.3f}")
    print(f"   - Multiplicative interactions (28): range [{min(feature_vector.multiplicative.values()):.3f}, {max(feature_vector.multiplicative.values()):.3f}]")
    print(f"   - Disagreement features (28): range [{min(feature_vector.disagreement.values()):.3f}, {max(feature_vector.disagreement.values()):.3f}]")
    print(f"   - Average features (28): range [{min(feature_vector.averages.values()):.3f}, {max(feature_vector.averages.values()):.3f}]")
    print(f"   - Regime context (2): VIX={feature_vector.vix_normalized:.3f}, trend={feature_vector.trend_strength:.3f}")
    print(f"   - Historical accuracy (8): mean={np.mean(list(feature_vector.accuracy_values.values())):.3f}")
    
    print("\n4. Feature Names Sample (first 20):")
    names = engine.get_feature_names()
    for name in names[:20]:
        print(f"   - {name}")
    print(f"   ... and {len(names)-20} more")
    
    print("\n5. Explanation Summary:")
    explanation = engine.explain_features(feature_vector, top_n=5)
    print(f"   Timestamp: {explanation['timestamp']}")
    print(f"   Base signals: {explanation['base_signals_summary']['bullish_count']} bullish, "
          f"{explanation['base_signals_summary']['bearish_count']} bearish, "
          f"{explanation['base_signals_summary']['neutral_count']} neutral")
    print(f"   Volatility regime: {explanation['regime_context']['volatility_regime']}")
    print(f"   Historical accuracy (mean): {explanation['historical_accuracy']['mean_accuracy']:.3f}")
    print(f"   Best performer: {explanation['historical_accuracy']['best_performer']}")
    
    print("\n6. Top 5 Synergistic Pairs (multiplicative):")
    for pair, value in explanation['pairwise_interactions']['high_synergy']:
        print(f"   - {pair}: {value:.3f}")
    
    print("\n7. Top 5 Disagreeing Pairs:")
    for pair, value in explanation['pairwise_interactions']['high_disagreement']:
        print(f"   - {pair}: {value:.3f}")
    
    print("\n" + "=" * 70)
    print("Demo complete. Performance target: <10ms ✓" if elapsed_ms < 10 else "Demo complete. Performance: >10ms")
    print("=" * 70)
    
    return feature_vector, features_np


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stacking Feature Engine")
    parser.add_argument("--test", action="store_true", help="Run demo/test")
    parser.add_argument("--names", action="store_true", help="Print all feature names")
    
    args = parser.parse_args()
    
    if args.test:
        demo()
    elif args.names:
        engine = StackingFeatureEngine()
        for name in engine.get_feature_names():
            print(name)
    else:
        parser.print_help()
