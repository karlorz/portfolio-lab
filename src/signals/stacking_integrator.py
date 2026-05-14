"""
Stacking Ensemble Integrator - Production Inference
Phase 3 of v3.10 Signal Stacking Ensemble

Provides real-time inference using trained XGBoost meta-learner
with fallback to weighted voting when confidence is low.

Author: Portfolio-Lab Agent
Version: v3.10 Phase 3
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import time

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class StackingPrediction:
    """Prediction output from stacking ensemble"""
    direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0.0 to 1.0
    probability_bullish: float
    probability_bearish: float
    probability_neutral: float
    fallback_used: bool  # True if weighted voting used instead
    feature_vector: Optional[np.ndarray] = None
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    model_version: str = "unknown"
    latency_ms: float = 0.0


@dataclass
class ModelMetadata:
    """Metadata for loaded stacking model"""
    version: str
    training_date: datetime
    feature_count: int
    accuracy_train: float
    accuracy_val: float
    feature_importance: Dict[str, float]
    total_samples: int


class StackingIntegrator:
    """
    Production stacking ensemble integrator.
    
    Loads trained XGBoost meta-learner and provides real-time
    inference with automatic fallback to weighted voting.
    """
    
    # Confidence threshold for fallback to weighted voting
    CONFIDENCE_THRESHOLD = 0.6
    
    # Model file paths
    MODEL_DIR = Path("models")
    MODEL_PREFIX = "signal_stacker_v"
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        fallback_threshold: float = 0.6,
        feature_engine=None
    ):
        """
        Initialize stacking integrator.
        
        Args:
            model_path: Path to pickled XGBoost model
            fallback_threshold: Confidence below which to use weighted voting
            feature_engine: FeatureEngine instance for feature generation
        """
        self.model_path = model_path
        self.fallback_threshold = fallback_threshold
        self.feature_engine = feature_engine
        self.model = None
        self.metadata: Optional[ModelMetadata] = None
        self.prediction_history: List[StackingPrediction] = []
        self.max_history = 1000
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: Path) -> bool:
        """
        Load trained XGBoost model from pickle file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get('model')
            metadata_dict = model_data.get('metadata', {})
            
            self.metadata = ModelMetadata(
                version=metadata_dict.get('version', 'unknown'),
                training_date=metadata_dict.get('training_date', datetime.now()),
                feature_count=metadata_dict.get('feature_count', 102),
                accuracy_train=metadata_dict.get('accuracy_train', 0.0),
                accuracy_val=metadata_dict.get('accuracy_val', 0.0),
                feature_importance=metadata_dict.get('feature_importance', {}),
                total_samples=metadata_dict.get('total_samples', 0)
            )
            
            logger.info(
                f"Loaded stacking model v{self.metadata.version} "
                f"({self.metadata.feature_count} features, "
                f"val_acc={self.metadata.accuracy_val:.3f})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            self.model = None
            self.metadata = None
            return False
    
    def predict(
        self,
        base_signals: Dict[str, Dict[str, Any]],
        current_regime: Optional[str] = None,
        vix_level: Optional[float] = None
    ) -> StackingPrediction:
        """
        Generate prediction using stacking ensemble.
        
        Args:
            base_signals: Dictionary of signal sources and their outputs
                         Format: {'tsmom': {'direction': 'bullish', ...}, ...}
            current_regime: Current market regime from HMM detector
            vix_level: Current VIX level for context
            
        Returns:
            StackingPrediction with direction, confidence, and metadata
        """
        import time
        start_time = time.time()
        
        # Generate feature vector
        if self.feature_engine:
            features = self.feature_engine.generate_features(
                base_signals, current_regime, vix_level
            )
        else:
            # Fallback: simple feature extraction
            features = self._extract_simple_features(
                base_signals, current_regime, vix_level
            )
        
        # Check if model is available
        if self.model is None:
            logger.warning("No model loaded, using weighted voting fallback")
            return self._weighted_voting_fallback(
                base_signals, features, start_time
            )
        
        try:
            # Reshape for single prediction
            X = features.reshape(1, -1)
            
            # Get probabilities from model
            probabilities = self.model.predict_proba(X)[0]
            
            # Map to directions (assuming model outputs in order)
            classes = self.model.classes_
            prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
            
            # Handle different class labels
            if 'bullish' in prob_dict:
                p_bull = prob_dict['bullish']
                p_bear = prob_dict.get('bearish', prob_dict.get('neutral', 0.0))
                p_neut = prob_dict.get('neutral', 0.0)
            else:
                # Numeric classes: assume 0=bearish, 1=neutral, 2=bullish
                p_bull = probabilities[2] if len(probabilities) > 2 else 0.0
                p_bear = probabilities[0]
                p_neut = probabilities[1] if len(probabilities) > 1 else 0.0
            
            # Determine direction and confidence
            max_prob = max(p_bull, p_bear, p_neut)
            
            if p_bull == max_prob:
                direction = 'bullish'
            elif p_bear == max_prob:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            confidence = max_prob
            
            # Check fallback threshold
            if confidence < self.fallback_threshold:
                logger.info(
                    f"Confidence {confidence:.3f} below threshold {self.fallback_threshold}, "
                    "using weighted voting"
                )
                return self._weighted_voting_fallback(
                    base_signals, features, start_time
                )
            
            # Get top features
            top_features = self._get_top_features(features)
            
            latency_ms = (time.time() - start_time) * 1000
            
            prediction = StackingPrediction(
                direction=direction,
                confidence=confidence,
                probability_bullish=p_bull,
                probability_bearish=p_bear,
                probability_neutral=p_neut,
                fallback_used=False,
                feature_vector=features,
                top_features=top_features,
                model_version=self.metadata.version if self.metadata else "unknown",
                latency_ms=latency_ms
            )
            
            self._add_to_history(prediction)
            
            logger.debug(
                f"Stacking prediction: {direction} ({confidence:.3f}) "
                f"in {latency_ms:.2f}ms"
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Model prediction failed: {e}, using fallback")
            return self._weighted_voting_fallback(
                base_signals, features, start_time
            )
    
    def _extract_simple_features(
        self,
        base_signals: Dict[str, Dict[str, Any]],
        current_regime: Optional[str],
        vix_level: Optional[float]
    ) -> np.ndarray:
        """
        Simple feature extraction when feature engine unavailable.
        Creates a basic feature vector from signal directions and strengths.
        """
        features = []
        
        # Signal directions mapped to numeric
        direction_map = {'bearish': -1, 'neutral': 0, 'bullish': 1}
        
        # Extract from common signal sources
        for source in ['tsmom', 'hmm_regime', 'fed_policy', 'ai_agent', 
                       'duration_overlay', 'base']:
            signal = base_signals.get(source, {})
            
            # Direction as numeric
            direction = direction_map.get(signal.get('direction', 'neutral'), 0)
            features.append(float(direction))
            
            # Confidence/strength
            features.append(float(signal.get('confidence', 0.5)))
            features.append(float(signal.get('strength', 0.5)))
        
        # Regime encoding
        regime_map = {
            'bull': 1.0, 'bear': -1.0, 'neutral': 0.0,
            'high_vol': -0.5, 'crisis': -1.0
        }
        regime_value = regime_map.get(current_regime, 0.0) if current_regime else 0.0
        features.append(float(regime_value))
        
        # VIX level (normalized)
        if vix_level:
            features.append(float(min(vix_level / 50.0, 1.0)))
        else:
            features.append(0.2)  # Default neutral
        
        # Pad to expected feature count if metadata available
        if self.metadata:
            while len(features) < self.metadata.feature_count:
                features.append(0.0)
            features = features[:self.metadata.feature_count]
        
        return np.array(features, dtype=np.float32)
    
    def _weighted_voting_fallback(
        self,
        base_signals: Dict[str, Dict[str, Any]],
        features: np.ndarray,
        start_time: float
    ) -> StackingPrediction:
        """
        Fallback to weighted voting when stacking confidence is low.
        Uses signal weights from v2.81 ensemble voter.
        """
        # Default weights (v2.81 ensemble)
        weights = {
            'tsmom': 0.30,
            'hmm_regime': 0.25,
            'fed_policy': 0.20,
            'ai_agent': 0.10,
            'duration_overlay': 0.05,
            'base': 0.10
        }
        
        # Calculate weighted vote
        direction_scores = {'bullish': 0.0, 'bearish': 0.0, 'neutral': 0.0}
        
        for source, signal in base_signals.items():
            direction = signal.get('direction', 'neutral')
            confidence = signal.get('confidence', 0.5)
            weight = weights.get(source, 0.1)
            
            direction_scores[direction] += weight * confidence
        
        # Normalize
        total = sum(direction_scores.values())
        if total > 0:
            for d in direction_scores:
                direction_scores[d] /= total
        
        # Select winner
        direction = max(direction_scores.keys(), key=lambda k: direction_scores.get(k, 0.0))
        confidence = direction_scores[direction]
        
        latency_ms = (time.time() - start_time) * 1000
        
        prediction = StackingPrediction(
            direction=direction,
            confidence=confidence,
            probability_bullish=direction_scores['bullish'],
            probability_bearish=direction_scores['bearish'],
            probability_neutral=direction_scores['neutral'],
            fallback_used=True,
            feature_vector=features,
            top_features=[("weighted_voting", 1.0)],
            model_version="fallback_v2.81",
            latency_ms=latency_ms
        )
        
        self._add_to_history(prediction)
        
        logger.debug(
            f"Fallback prediction: {direction} ({confidence:.3f}) "
            f"in {latency_ms:.2f}ms"
        )
        
        return prediction
    
    def _get_top_features(
        self,
        features: np.ndarray,
        n_top: int = 10
    ) -> List[Tuple[str, float]]:
        """Get top N most important features for this prediction"""
        if not self.metadata or not self.metadata.feature_importance:
            return []
        
        # Get feature names and importances
        feat_importance = self.metadata.feature_importance
        
        # Sort by importance
        sorted_features = sorted(
            feat_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_features[:n_top]
    
    def _add_to_history(self, prediction: StackingPrediction) -> None:
        """Add prediction to history with size limit"""
        self.prediction_history.append(prediction)
        
        # Trim history if needed
        if len(self.prediction_history) > self.max_history:
            self.prediction_history = self.prediction_history[-self.max_history:]
    
    def get_accuracy_stats(self, window_days: int = 30) -> Dict[str, float]:
        """
        Calculate accuracy statistics over recent history.
        
        Args:
            window_days: Number of days to include in calculation
            
        Returns:
            Dictionary with accuracy metrics
        """
        cutoff = datetime.now() - timedelta(days=window_days)
        recent = [p for p in self.prediction_history if p.timestamp > cutoff]
        
        if not recent:
            return {'accuracy': 0.0, 'count': 0, 'fallback_rate': 0.0}
        
        # Note: Actual accuracy requires comparing to realized returns
        # This is a placeholder for the tracking structure
        fallback_count = sum(1 for p in recent if p.fallback_used)
        
        return {
            'accuracy': 0.0,
            'count': int(len(recent)),
            'fallback_rate': float(fallback_count / len(recent)) if recent else 0.0,
            'avg_confidence': float(np.mean([p.confidence for p in recent])),
            'avg_latency_ms': float(np.mean([p.latency_ms for p in recent]))
        }
    
    def detect_drift(self, threshold: float = 0.05) -> Optional[str]:
        """
        Detect model drift by comparing recent vs training performance.
        
        Args:
            threshold: Accuracy drop threshold for drift alert
            
        Returns:
            Alert message if drift detected, None otherwise
        """
        if not self.metadata:
            return None
        
        stats = self.get_accuracy_stats(window_days=30)
        
        # Compare recent accuracy to training
        # Note: This requires realized returns tracking
        train_acc = self.metadata.accuracy_train
        val_acc = self.metadata.accuracy_val
        
        # If we had actual accuracy tracking:
        # recent_acc = stats['accuracy']
        # if recent_acc < val_acc - threshold:
        #     return f"Model drift detected: accuracy {recent_acc:.3f} vs validation {val_acc:.3f}"
        
        # For now, monitor fallback rate as proxy
        if stats['fallback_rate'] > 0.3:
            return f"High fallback rate: {stats['fallback_rate']:.1%} predictions using weighted voting"
        
        return None
    
    def export_prediction_log(self, filepath: Path) -> bool:
        """Export prediction history to JSON for analysis"""
        try:
            records = []
            for p in self.prediction_history:
                records.append({
                    'timestamp': p.timestamp.isoformat(),
                    'direction': p.direction,
                    'confidence': p.confidence,
                    'fallback_used': p.fallback_used,
                    'model_version': p.model_version,
                    'latency_ms': p.latency_ms
                })
            
            with open(filepath, 'w') as f:
                json.dump(records, f, indent=2)
            
            logger.info(f"Exported {len(records)} predictions to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export predictions: {e}")
            return False


# Convenience function for direct use
def get_stacking_prediction(
    base_signals: Dict[str, Dict[str, Any]],
    model_path: Optional[Path] = None,
    current_regime: Optional[str] = None,
    vix_level: Optional[float] = None
) -> StackingPrediction:
    """
    Convenience function to get stacking prediction without managing integrator state.
    
    Args:
        base_signals: Signal outputs from all sources
        model_path: Optional path to model file
        current_regime: Market regime
        vix_level: VIX level
        
    Returns:
        StackingPrediction
    """
    integrator = StackingIntegrator(model_path=model_path)
    return integrator.predict(base_signals, current_regime, vix_level)
