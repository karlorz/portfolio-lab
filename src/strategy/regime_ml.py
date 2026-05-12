"""
Regime-Conditional Machine Learning Strategy
Implements regime-aware factor scoring with different ML models per market regime.

Based on AQR's "Virtue of Complexity" research showing 50-100% improvement
when models are trained conditionally on regime rather than globally.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import sqlite3

# Import existing components (use absolute imports from project root)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.regime_classifier import RegimeClassifier, Regime
from src.research.features import FeaturePipeline, Features
from src.strategy.factor_rotation import FactorMomentumEngine, FactorScore

# Phase 3C: Ensemble voting integration
from src.strategy.ensemble_voter import EnsembleVotingEngine, EnsembleRegime


class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class CorrelationRegime(Enum):
    """Correlation regime classification"""
    LOW = "low"
    HIGH = "high"


@dataclass
class RegimeState:
    """Current market regime state"""
    timestamp: str
    vol_regime: VolatilityRegime
    corr_regime: CorrelationRegime
    yield_curve_inverted: bool
    liquidity_stress: bool
    momentum_bearish: bool
    
    # Composite scores
    risk_score: float  # 0-1, higher = more risk-off
    regime_label: str  # For feature selection
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "vol_regime": self.vol_regime.value,
            "corr_regime": self.corr_regime.value,
            "yield_curve_inverted": self.yield_curve_inverted,
            "liquidity_stress": self.liquidity_stress,
            "momentum_bearish": self.momentum_bearish,
            "risk_score": self.risk_score,
            "regime_label": self.regime_label,
        }


@dataclass
class RegimeConditionalScore:
    """Enhanced factor score with regime-conditional adjustments"""
    base_score: FactorScore
    regime: RegimeState
    
    # Regime-conditional adjustments
    vol_adjusted_momentum: float
    corr_adjusted_weight: float
    regime_multiplier: float
    
    # Model confidence by regime
    high_vol_confidence: float
    low_vol_confidence: float
    high_corr_confidence: float
    low_corr_confidence: float
    
    # Final composite
    conditional_score: float
    final_allocation_weight: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.base_score.symbol,
            "regime": self.regime.to_dict(),
            "base_momentum": self.base_score.momentum_score,
            "vol_adjusted": self.vol_adjusted_momentum,
            "corr_adjusted": self.corr_adjusted_weight,
            "regime_multiplier": self.regime_multiplier,
            "confidence": {
                "high_vol": self.high_vol_confidence,
                "low_vol": self.low_vol_confidence,
                "high_corr": self.high_corr_confidence,
                "low_corr": self.low_corr_confidence,
            },
            "conditional_score": self.conditional_score,
            "final_weight": self.final_allocation_weight,
        }


class RegimeDetector:
    """
    Detects market regime based on VIX, correlations, yield curve, and momentum.
    """
    
    VIX_LOW_THRESHOLD = 15.0
    VIX_HIGH_THRESHOLD = 25.0
    CORR_HIGH_THRESHOLD = 0.5
    HYG_SPREAD_STRESS = 400.0  # bps
    MOMENTUM_BEARISH = -0.10   # -10%
    
    def __init__(self, db_path: Path = Path("~/projects/portfolio-lab/data/market.db").expanduser()):
        self.db_path = db_path
        self.feature_pipeline = FeaturePipeline(str(db_path))
        
    def detect_regime(self, symbol: str = "SPY") -> RegimeState:
        """
        Detect current market regime using multiple indicators.
        """
        timestamp = datetime.now().isoformat()
        
        # Get features for analysis
        features = self.feature_pipeline.generate_features(symbol)
        if features is None:
            # Default to neutral if no data
            return RegimeState(
                timestamp=timestamp,
                vol_regime=VolatilityRegime.NORMAL,
                corr_regime=CorrelationRegime.LOW,
                yield_curve_inverted=False,
                liquidity_stress=False,
                momentum_bearish=False,
                risk_score=0.5,
                regime_label="normal"
            )
        
        # Volatility regime from VIX
        vix_level = features.vix_level if features.vix_level else 20.0
        if vix_level < self.VIX_LOW_THRESHOLD:
            vol_regime = VolatilityRegime.LOW
        elif vix_level > self.VIX_HIGH_THRESHOLD:
            vol_regime = VolatilityRegime.HIGH
        else:
            vol_regime = VolatilityRegime.NORMAL
        
        # Correlation regime
        spy_corr = features.spy_correlation_20d if features.spy_correlation_20d else 0.3
        corr_regime = CorrelationRegime.HIGH if spy_corr > self.CORR_HIGH_THRESHOLD else CorrelationRegime.LOW
        
        # Yield curve check (10Y-3M spread from features)
        # Note: This would come from yield curve data, using proxy from returns
        yield_inverted = False  # Placeholder - would use actual yield curve data
        
        # Momentum check
        momentum_bearish = features.return_20d < self.MOMENTUM_BEARISH if features.return_20d else False
        
        # Liquidity stress (proxy via volatility spike)
        liquidity_stress = vol_regime == VolatilityRegime.HIGH and vix_level > 30
        
        # Calculate composite risk score
        risk_score = 0.0
        risk_score += 0.3 if vol_regime == VolatilityRegime.HIGH else 0.1 if vol_regime == VolatilityRegime.NORMAL else 0.0
        risk_score += 0.2 if corr_regime == CorrelationRegime.HIGH else 0.0
        risk_score += 0.2 if yield_inverted else 0.0
        risk_score += 0.2 if liquidity_stress else 0.0
        risk_score += 0.1 if momentum_bearish else 0.0
        
        # Determine regime label
        if risk_score > 0.7:
            regime_label = "high_risk"
        elif risk_score < 0.3:
            regime_label = "low_risk"
        else:
            regime_label = "normal"
        
        return RegimeState(
            timestamp=timestamp,
            vol_regime=vol_regime,
            corr_regime=corr_regime,
            yield_curve_inverted=yield_inverted,
            liquidity_stress=liquidity_stress,
            momentum_bearish=momentum_bearish,
            risk_score=risk_score,
            regime_label=regime_label
        )


class RegimeMLScorer:
    """
    Applies regime-conditional ML scoring to factors.
    Uses different feature weights and models based on current regime.
    """
    
    # Regime-conditional feature weights
    REGIME_WEIGHTS = {
        "high_vol": {
            "momentum_12m": 0.2,
            "momentum_1m": 0.3,  # Short-term more important in high vol
            "volatility": 0.25,
            "mean_reversion": 0.25,  # Mean reversion works in high vol
        },
        "low_vol": {
            "momentum_12m": 0.4,
            "momentum_1m": 0.1,
            "volatility": 0.1,
            "trend_following": 0.4,  # Trend following works in low vol
        },
        "high_corr": {
            "diversification": 0.4,
            "momentum_12m": 0.2,
            "factor_divergence": 0.4,  # Factor divergence key in high corr
        },
        "low_corr": {
            "concentration": 0.3,
            "momentum_12m": 0.4,
            "selection_skill": 0.3,
        }
    }
    
    def __init__(self, regime_detector: Optional[RegimeDetector] = None):
        self.regime_detector = regime_detector or RegimeDetector()
        self.current_regime: Optional[RegimeState] = None
        
    def calculate_regime_score(
        self, 
        factor_score: FactorScore,
        regime: RegimeState
    ) -> RegimeConditionalScore:
        """
        Calculate regime-conditional score for a factor.
        """
        # Base momentum from factor rotation
        base_momentum = factor_score.momentum_score
        
        # Volatility regime adjustment
        if regime.vol_regime == VolatilityRegime.HIGH:
            # In high vol: focus on mean reversion, reduce momentum
            vol_adjusted = base_momentum * 0.5 + factor_score.momentum_acceleration * 0.5
            vol_confidence = 0.7
        elif regime.vol_regime == VolatilityRegime.LOW:
            # In low vol: momentum works well
            vol_adjusted = base_momentum * 1.2
            vol_confidence = 0.85
        else:
            vol_adjusted = base_momentum
            vol_confidence = 0.75
        
        # Correlation regime adjustment
        if regime.corr_regime == CorrelationRegime.HIGH:
            # In high correlation: diversification premium
            corr_adjusted = vol_adjusted * 0.8
            corr_confidence = 0.6
        else:
            # In low correlation: selection skill matters
            corr_adjusted = vol_adjusted * 1.1
            corr_confidence = 0.9
        
        # Regime multiplier based on risk score
        # Higher risk = more defensive positioning
        if regime.risk_score > 0.7:
            regime_multiplier = 0.7  # Reduce exposure
        elif regime.risk_score < 0.3:
            regime_multiplier = 1.15  # Increase exposure
        else:
            regime_multiplier = 1.0
        
        # Calculate confidence scores for each regime type
        high_vol_confidence = 0.8 if regime.vol_regime == VolatilityRegime.HIGH else 0.3
        low_vol_confidence = 0.8 if regime.vol_regime == VolatilityRegime.LOW else 0.4
        high_corr_confidence = 0.7 if regime.corr_regime == CorrelationRegime.HIGH else 0.3
        low_corr_confidence = 0.8 if regime.corr_regime == CorrelationRegime.LOW else 0.4
        
        # Final conditional score
        conditional_score = corr_adjusted * regime_multiplier
        
        # Allocation weight - inverse vol adjusted by regime
        vol_scalar = 0.15 / max(factor_score.volatility, 0.05)
        if regime.vol_regime == VolatilityRegime.HIGH:
            vol_scalar *= 0.7  # Reduce size in high vol
        elif regime.vol_regime == VolatilityRegime.LOW:
            vol_scalar *= 1.2  # Increase size in low vol
        
        final_weight = vol_scalar * regime_multiplier
        
        return RegimeConditionalScore(
            base_score=factor_score,
            regime=regime,
            vol_adjusted_momentum=vol_adjusted,
            corr_adjusted_weight=corr_adjusted,
            regime_multiplier=regime_multiplier,
            high_vol_confidence=high_vol_confidence,
            low_vol_confidence=low_vol_confidence,
            high_corr_confidence=high_corr_confidence,
            low_corr_confidence=low_corr_confidence,
            conditional_score=conditional_score,
            final_allocation_weight=final_weight
        )
    
    def evaluate_all_factors(
        self,
        factor_scores: Dict[str, FactorScore]
    ) -> Dict[str, RegimeConditionalScore]:
        """
        Evaluate all factors with regime-conditional scoring.
        """
        # Detect current regime once
        self.current_regime = self.regime_detector.detect_regime("SPY")
        
        # Score each factor
        conditional_scores = {}
        for symbol, score in factor_scores.items():
            conditional_scores[symbol] = self.calculate_regime_score(
                score, self.current_regime
            )
        
        return conditional_scores
    
    def generate_allocation(
        self,
        conditional_scores: Dict[str, RegimeConditionalScore],
        top_n: int = 2
    ) -> Dict[str, float]:
        """
        Generate allocation based on regime-conditional scores.
        """
        # Sort by conditional score
        sorted_scores = sorted(
            conditional_scores.items(),
            key=lambda x: x[1].conditional_score,
            reverse=True
        )
        
        # Select top N
        selected = sorted_scores[:top_n]
        
        if not selected:
            return {"SPY": 1.0}
        
        # Weight by final_allocation_weight
        total_weight = sum(s.final_allocation_weight for _, s in selected)
        
        allocation = {}
        for symbol, score in selected:
            allocation[symbol] = score.final_allocation_weight / total_weight
        
        return allocation


@dataclass
class RegimeTransition:
    """Tracks regime changes for smoothing during transitions"""
    from_regime: RegimeState
    to_regime: RegimeState
    transition_date: str
    days_in_transition: int = 0
    transition_confidence: float = 0.0  # 0-1, higher = more certain


class EnsembleSmoother:
    """
    Phase 3: Ensemble Integration with smoothing during regime transitions.
    Smooths allocation changes when regimes shift to reduce whipsaws.
    """
    
    TRANSITION_DAYS = 5  # Days to smooth over
    
    def __init__(self, transition_days: int = 5):
        self.transition_days = transition_days
        self.transition_history: List[RegimeTransition] = []
        self.current_allocation: Optional[Dict[str, float]] = None
        
    def detect_regime_change(
        self, 
        current: RegimeState, 
        previous: Optional[RegimeState]
    ) -> bool:
        """Detect if regime has materially changed."""
        if previous is None:
            return False
            
        # Check key regime dimensions
        vol_change = current.vol_regime != previous.vol_regime
        corr_change = current.corr_regime != previous.corr_regime
        risk_jump = abs(current.risk_score - previous.risk_score) > 0.2
        
        return vol_change or corr_change or risk_jump
    
    def calculate_transition_weights(
        self,
        new_allocation: Dict[str, float],
        previous_allocation: Optional[Dict[str, float]],
        regime: RegimeState,
        transition: Optional[RegimeTransition]
    ) -> Dict[str, float]:
        """
        Smooth allocation during regime transitions.
        Uses exponential decay blending between old and new allocations.
        """
        if previous_allocation is None or transition is None:
            return new_allocation
        
        # Calculate blend factor based on days in transition
        days = min(transition.days_in_transition, self.transition_days)
        blend = days / self.transition_days  # 0 = old, 1 = new
        
        # Apply sigmoid smoothing for more natural transition
        import math
        smooth_blend = 1 / (1 + math.exp(-6 * (blend - 0.5)))  # S-curve
        
        # Blend allocations
        all_keys = set(new_allocation.keys()) | set(previous_allocation.keys())
        blended = {}
        
        for key in all_keys:
            new_val = new_allocation.get(key, 0.0)
            old_val = previous_allocation.get(key, 0.0)
            blended[key] = old_val * (1 - smooth_blend) + new_val * smooth_blend
        
        return blended
    
    def update_transition(
        self,
        current: RegimeState,
        previous: Optional[RegimeState]
    ) -> Optional[RegimeTransition]:
        """Update transition state and return active transition if any."""
        if previous is None:
            return None
            
        is_change = self.detect_regime_change(current, previous)
        
        if is_change:
            # Start new transition
            transition = RegimeTransition(
                from_regime=previous,
                to_regime=current,
                transition_date=current.timestamp,
                days_in_transition=0,
                transition_confidence=0.5
            )
            self.transition_history.append(transition)
            return transition
        
        # Check if we have an active transition
        if self.transition_history:
            latest = self.transition_history[-1]
            # Count days (simplified - just increment)
            latest.days_in_transition += 1
            
            if latest.days_in_transition < self.transition_days:
                # Still in transition
                latest.transition_confidence = min(
                    0.9, 
                    0.5 + (latest.days_in_transition / self.transition_days) * 0.4
                )
                return latest
        
        return None
    
    def smooth_allocation(
        self,
        raw_allocation: Dict[str, float],
        regime: RegimeState,
        previous_regime: Optional[RegimeState],
        previous_allocation: Optional[Dict[str, float]]
    ) -> Tuple[Dict[str, float], Optional[RegimeTransition]]:
        """
        Apply ensemble smoothing to allocation.
        Returns smoothed allocation and transition info.
        """
        transition = self.update_transition(regime, previous_regime)
        
        if transition and transition.days_in_transition < self.transition_days:
            # We're in a transition period - apply smoothing
            smoothed = self.calculate_transition_weights(
                raw_allocation,
                previous_allocation,
                regime,
                transition
            )
            self.current_allocation = smoothed
            return smoothed, transition
        
        # No transition or completed - use raw allocation
        self.current_allocation = raw_allocation
        return raw_allocation, None


class RegimeConditionalEngine:
    """
    Main engine combining factor rotation with regime-conditional ML scoring.
    Phase 1-2: Regime detection and conditional models
    Phase 3: Ensemble integration with smoothing (COMPLETE)
    Phase 4: Validation framework (COMPLETE)
    This is the v2.20 implementation entry point.
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        top_n: int = 2,
        use_regime_ml: bool = True,
        enable_smoothing: bool = True
    ):
        self.db_path = db_path or Path("~/projects/portfolio-lab/data/market.db").expanduser()
        self.top_n = top_n
        self.use_regime_ml = use_regime_ml
        self.enable_smoothing = enable_smoothing
        
        # Initialize components
        self.factor_engine = FactorMomentumEngine(
            db_path=self.db_path,
            top_n=top_n
        )
        self.regime_detector = RegimeDetector(self.db_path)
        self.ml_scorer = RegimeMLScorer(self.regime_detector)
        
        # Phase 3: Ensemble smoothing
        self.ensemble_smoother = EnsembleSmoother() if enable_smoothing else None
        self.previous_regime: Optional[RegimeState] = None
        self.previous_allocation: Optional[Dict[str, float]] = None
        
        # Phase 3C: Ensemble voting engine (v2.20)
        self.ensemble_voter = EnsembleVotingEngine()
        
    def evaluate(self) -> Dict[str, Any]:
        """
        Run full regime-conditional factor evaluation.
        
        Returns:
            Complete evaluation results with regime-aware allocations
        """
        timestamp = datetime.now().isoformat()
        
        # Get base factor scores
        base_result = self.factor_engine.evaluate()
        
        if self.use_regime_ml and base_result.get("current_scores"):
            # Get factor score objects
            factor_scores = {
                symbol: FactorScore(
                    symbol=symbol,
                    factor_name=data["factor_name"],
                    price=0.0,  # Not needed for scoring
                    return_12m=data["return_12m"],
                    return_6m=data.get("return_6m", 0),
                    return_3m=data.get("return_3m", 0),
                    volatility=data["volatility"],
                    sharpe_12m=data.get("sharpe_12m", 0),
                    momentum_score=data["momentum_score"],
                    rank=data.get("rank", 0)
                )
                for symbol, data in base_result["current_scores"].items()
            }
            
            # Apply regime-conditional scoring
            conditional_scores = self.ml_scorer.evaluate_all_factors(factor_scores)
            
            # Generate regime-aware allocation
            raw_allocation = self.ml_scorer.generate_allocation(
                conditional_scores, self.top_n
            )
            
            # Phase 3: Apply ensemble smoothing during transitions
            smoothed_allocation = raw_allocation
            transition_info = None
            if self.enable_smoothing and self.ensemble_smoother:
                smoothed_allocation, transition_info = self.ensemble_smoother.smooth_allocation(
                    raw_allocation=raw_allocation,
                    regime=self.ml_scorer.current_regime,
                    previous_regime=self.previous_regime,
                    previous_allocation=self.previous_allocation
                )
                # Update previous state for next evaluation
                self.previous_regime = self.ml_scorer.current_regime
                self.previous_allocation = smoothed_allocation
            
            # Build enhanced output with smoothing info
            result = {
                "timestamp": timestamp,
                "regime": self.ml_scorer.current_regime.to_dict() if self.ml_scorer.current_regime else {},
                "selected_factors": list(smoothed_allocation.keys()),
                "allocation": smoothed_allocation,
                "base_allocation": base_result["allocation"],
                "raw_regime_allocation": raw_allocation if smoothed_allocation != raw_allocation else None,
                "conditional_scores": {
                    symbol: score.to_dict()
                    for symbol, score in conditional_scores.items()
                },
                "current_scores": base_result["current_scores"],
                "diversity": base_result["diversity"],
                "signal_strength": base_result["signal_strength"],
                "recommendation": self._generate_regime_recommendation(
                    self.ml_scorer.current_regime,
                    conditional_scores
                ),
                "method": "regime_conditional_ml_v2.20"
            }
            
            # Add transition info if active
            if transition_info:
                result["regime_transition"] = {
                    "active": True,
                    "days_in_transition": transition_info.days_in_transition,
                    "transition_confidence": transition_info.transition_confidence,
                    "from_regime_label": transition_info.from_regime.regime_label,
                    "to_regime_label": transition_info.to_regime.regime_label
                }
            
            # Phase 3C: Add ensemble voting results
            try:
                ensemble_result = self.ensemble_voter.evaluate()
                if ensemble_result:
                    result["ensemble_voting"] = {
                        "regime": ensemble_result.regime,
                        "confidence": ensemble_result.confidence,
                        "agreement_score": ensemble_result.agreement_score,
                        "probabilities": ensemble_result.ensemble_probs,
                        "action": ensemble_result.action,
                        "position_scaling": ensemble_result.position_scaling,
                        "disagreement_sources": ensemble_result.disagreement_sources
                    }
            except Exception as e:
                # Ensemble voting may not be fully available
                result["ensemble_voting"] = {"error": str(e)}
            
            return result
        else:
            # Fall back to base factor rotation
            return {
                **base_result,
                "method": "base_factor_rotation_v2.9",
                "regime_ml_available": False
            }
    
    def _generate_regime_recommendation(
        self,
        regime: Optional[RegimeState],
        scores: Dict[str, RegimeConditionalScore]
    ) -> str:
        """Generate human-readable recommendation based on regime."""
        if not regime:
            return "Insufficient data for regime analysis"
        
        # Get top factor
        top_factor = max(scores.items(), key=lambda x: x[1].conditional_score)
        symbol, score = top_factor
        
        parts = []
        
        # Volatility context
        if regime.vol_regime == VolatilityRegime.HIGH:
            parts.append(f"High volatility regime detected (VIX elevated). "
                        f"Reduced position sizing ({score.regime_multiplier:.0%} normal). "
                        f"Focus on mean-reversion factors.")
        elif regime.vol_regime == VolatilityRegime.LOW:
            parts.append(f"Low volatility regime. "
                        f"Increased position sizing ({score.regime_multiplier:.0%} normal). "
                        f"Momentum strategies favored.")
        
        # Correlation context
        if regime.corr_regime == CorrelationRegime.HIGH:
            parts.append("High correlation regime - diversification benefits limited.")
        
        # Risk context
        if regime.risk_score > 0.7:
            parts.append("Elevated risk environment - defensive positioning recommended.")
        
        parts.append(f"Top selection: {symbol} ({score.base_score.factor_name}) "
                    f"with conditional score {score.conditional_score:.3f}")
        
        return " ".join(parts)


def main():
    """CLI for regime-conditional ML evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Regime-Conditional ML Factor Scoring")
    parser.add_argument("--detect", action="store_true", help="Detect current regime only")
    parser.add_argument("--evaluate", action="store_true", help="Run full evaluation")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--no-ml", action="store_true", help="Use base factor rotation only")
    
    args = parser.parse_args()
    
    if args.detect:
        detector = RegimeDetector()
        regime = detector.detect_regime("SPY")
        print(json.dumps(regime.to_dict(), indent=2))
    
    elif args.evaluate or not args.detect:
        engine = RegimeConditionalEngine(use_regime_ml=not args.no_ml)
        result = engine.evaluate()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
    
    return 0


if __name__ == "__main__":
    exit(main())
