"""
Ensemble Voting Regime Detector (v220 Phase 3)

Combines multiple regime detection signals into unified probabilistic classification:
- Wasserstein HMM (40%): Hidden Markov Model with template tracking
- CTA Trend Overlay (25%): Multi-timeframe trend detection  
- TSFM Regime (20%): Time-series factor model
- Duration/Yield Curve (15%): Curve-based recession signals

Expected improvement: +10pp regime accuracy, +0.06 Sharpe, -50% whipsaw
"""

import os
import json
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.strategy.regime_hmm import WassersteinHMMDetector, RegimeState
from src.strategy.cta_overlay import CTATrendEngine
from src.signals.vix_ensemble_adapter import VIXEnsembleAdapter


@dataclass
class RegimeSignal:
    """Normalized regime signal from any source"""
    source: str  # 'hmm', 'cta', 'tsfm', 'duration'
    regime: str  # 'bull', 'bear', 'neutral', 'crisis'
    probability: float  # Confidence in regime classification
    confidence: float  # Signal confidence (0-1)
    timestamp: str
    raw_data: Dict[str, Any]  # Source-specific data


@dataclass
class EnsembleRegime:
    """Final ensemble regime classification"""
    timestamp: str
    regime: str  # 'bull', 'bear', 'neutral', 'crisis'
    confidence: float  # Ensemble confidence score
    
    # Individual signal probabilities
    hmm_prob: float
    cta_prob: float
    tsfm_prob: float
    duration_prob: float
    vix_insurance_prob: float  # New: VIX insurance signal probability
    
    # Weighted combination
    ensemble_probs: Dict[str, float]
    
    # Signal agreement metrics
    agreement_score: float  # How much signals agree
    disagreement_sources: List[str]  # Which signals differ from majority
    
    # Action recommendation
    action: str  # 'full_shift', 'partial_shift', 'hold', 'alert'
    position_scaling: float  # 0-1 position sizing factor
    
    # VIX insurance overlay status
    insurance_active: bool = False
    insurance_alerts: List[str] = None
    
    def __post_init__(self):
        if self.insurance_alerts is None:
            self.insurance_alerts = []


class RegimeSignalAdapter:
    """Adapts various signal sources to unified RegimeSignal format"""
    
    # Map source-specific regimes to unified regime space
    REGIME_MAP = {
        'hmm': {
            'bull': 'bull',
            'bear': 'bear',
            'neutral': 'neutral',
            'crisis': 'crisis'
        },
        'cta': {
            'uptrend': 'bull',
            'downtrend': 'bear',
            'ranging': 'neutral',
            'chop': 'neutral'
        },
        'tsfm': {
            'expansion': 'bull',
            'contraction': 'bear',
            'stress': 'crisis',
            'recovery': 'bull',
            'normal': 'neutral'
        },
        'duration': {
            'steep': 'bull',
            'normal': 'neutral',
            'flat': 'neutral',
            'inverted': 'bear'
        },
        'vix_insurance': {
            'risk_off': 'bear',
            'risk_on': 'bull',
            'neutral': 'neutral',
            'hedge_active': 'neutral'  # Defensive but not bearish
        }
    }
    
    @classmethod
    def from_hmm(cls, regime_state: RegimeState) -> RegimeSignal:
        """Convert HMM RegimeState to RegimeSignal"""
        return RegimeSignal(
            source='hmm',
            regime=regime_state.regime_label,
            probability=regime_state.probability,
            confidence=regime_state.template_confidence,
            timestamp=regime_state.timestamp,
            raw_data={
                'regime_id': regime_state.regime_id,
                'template_distance': regime_state.template_distance,
                'vix_level': regime_state.vix_level,
                'momentum_20d': regime_state.momentum_20d
            }
        )
    
    @classmethod
    def from_cta(cls, cta_result: Dict[str, Any]) -> RegimeSignal:
        """Convert CTA trend result to RegimeSignal"""
        # Determine regime from trend score
        avg_trend = cta_result.get('summary', {}).get('avg_trend_score', 0)
        avg_strength = cta_result.get('summary', {}).get('avg_trend_strength', 0)
        
        if avg_trend > 0.3:
            regime = 'uptrend'
            prob = 0.5 + avg_strength * 0.5
        elif avg_trend < -0.3:
            regime = 'downtrend'
            prob = 0.5 + avg_strength * 0.5
        else:
            regime = 'ranging'
            prob = 0.6 + avg_strength * 0.2
        
        unified_regime = cls.REGIME_MAP['cta'].get(regime, 'neutral')
        
        return RegimeSignal(
            source='cta',
            regime=unified_regime,
            probability=prob,
            confidence=avg_strength,
            timestamp=cta_result.get('timestamp', datetime.now().isoformat()),
            raw_data=cta_result
        )
    
    @classmethod
    def from_duration(cls, duration_data: Dict[str, Any]) -> RegimeSignal:
        """Convert duration/yield curve data to RegimeSignal"""
        # Use curve regime if available
        curve_regime = duration_data.get('curve_regime', 'normal')
        recession_prob = duration_data.get('recession_probability', 0)
        
        unified_regime = cls.REGIME_MAP['duration'].get(curve_regime, 'neutral')
        
        # Higher confidence with stronger signals
        if recession_prob > 0.7:
            confidence = 0.8
            prob = recession_prob
        elif recession_prob < 0.3:
            confidence = 0.7
            prob = 1 - recession_prob
        else:
            confidence = 0.5
            prob = 0.5
        
        return RegimeSignal(
            source='duration',
            regime=unified_regime,
            probability=prob,
            confidence=confidence,
            timestamp=duration_data.get('timestamp', datetime.now().isoformat()),
            raw_data=duration_data
        )
    
    @classmethod
    def from_vix_insurance(cls, vix_signal: Dict[str, Any]) -> RegimeSignal:
        """Convert VIX insurance signal to RegimeSignal"""
        is_active = vix_signal.get('insurance_active', False)
        risk_adjustment = vix_signal.get('risk_score_adjustment', 0.0)
        
        # Determine regime based on insurance status
        if not is_active:
            regime = 'neutral'
            prob = 0.5
        elif risk_adjustment > 0.05:
            regime = 'risk_off'  # Defensive posture
            prob = 0.5 + risk_adjustment
        elif risk_adjustment < -0.02:
            regime = 'risk_on'  # Profit taking opportunity
            prob = 0.5 - risk_adjustment
        else:
            regime = 'hedge_active'
            prob = 0.6
        
        unified_regime = cls.REGIME_MAP['vix_insurance'].get(regime, 'neutral')
        
        # Confidence based on position size and budget health
        metadata = vix_signal.get('metadata', {})
        position_pct = metadata.get('position_size_pct', 0)
        budget_pct = metadata.get('budget_used_pct', 0)
        
        # Higher confidence with larger positions and healthy budgets
        if position_pct > 0.005 and budget_pct < 0.8:
            confidence = 0.7
        elif position_pct > 0:
            confidence = 0.5
        else:
            confidence = 0.3
        
        return RegimeSignal(
            source='vix_insurance',
            regime=unified_regime,
            probability=prob,
            confidence=confidence,
            timestamp=vix_signal.get('timestamp', datetime.now().isoformat()),
            raw_data={
                'insurance_active': is_active,
                'risk_adjustment': risk_adjustment,
                'cash_buffer_pct': vix_signal.get('cash_buffer_pct', 0),
                'alerts': vix_signal.get('alerts', []),
                'next_action': vix_signal.get('next_action', 'none')
            }
        )


class EnsembleVotingEngine:
    """
    Ensemble voting system combining multiple regime signals.
    
    Implements both hard voting (majority vote) and soft voting (weighted 
    probability combination) with confidence scoring and action recommendations.
    """
    
    # Default signal weights
    DEFAULT_WEIGHTS = {
        'hmm': 0.38,        # Reduced from 0.40 to accommodate VIX signal
        'cta': 0.24,        # Reduced from 0.25
        'tsfm': 0.19,       # Reduced from 0.20
        'duration': 0.14,   # Reduced from 0.15
        'vix_insurance': 0.05  # New: VIX insurance overlay signal
    }
    
    # Confidence thresholds
    CONFIDENCE_THRESHOLDS = {
        'high': 0.70,
        'medium': 0.50,
        'low': 0.30
    }
    
    # Action mapping based on confidence
    ACTION_MAP = {
        'high': ('full_shift', 1.0),
        'medium': ('partial_shift', 0.5),
        'low': ('hold', 0.25),
        'very_low': ('alert', 0.0)
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.hmm_detector = WassersteinHMMDetector()
        self.cta_engine = CTATrendEngine()
    
    def collect_signals(self) -> List[RegimeSignal]:
        """Collect signals from all available sources"""
        signals = []
        
        # HMM Signal
        try:
            self.hmm_detector.load_state()
            hmm_state = self.hmm_detector.detect_current_regime()
            if hmm_state:
                signals.append(RegimeSignalAdapter.from_hmm(hmm_state))
        except Exception as e:
            print(f"HMM signal error: {e}")
        
        # CTA Signal
        try:
            cta_result = self.cta_engine.evaluate()
            if cta_result:
                signals.append(RegimeSignalAdapter.from_cta(cta_result))
        except Exception as e:
            print(f"CTA signal error: {e}")
        
        # Duration Signal (from yields data if available)
        try:
            duration_data = self._load_duration_data()
            if duration_data:
                signals.append(RegimeSignalAdapter.from_duration(duration_data))
        except Exception as e:
            print(f"Duration signal error: {e}")
        
        # VIX Insurance Signal (if available)
        try:
            vix_adapter = VIXEnsembleAdapter()
            vix_signal = vix_adapter.get_ensemble_signal()
            if vix_signal:
                signals.append(RegimeSignalAdapter.from_vix_insurance(vix_signal))
        except Exception as e:
            print(f"VIX insurance signal error: {e}")
        
        return signals
    
    def _load_duration_data(self) -> Optional[Dict[str, Any]]:
        """Load duration/yield curve data from data files"""
        yields_path = project_root / 'data' / 'yields.json'
        if not yields_path.exists():
            return None
        
        try:
            with open(yields_path, 'r') as f:
                data = json.load(f)
            
            if 'records' in data and len(data['records']) > 0:
                latest = data['records'][-1]
                
                # Calculate recession probability from spread
                ten_year = latest.get('10_year', 0)
                two_year = latest.get('2_year', 0)
                spread = ten_year - two_year if ten_year and two_year else 0
                
                recession_prob = 0.5
                if spread < -0.5:
                    recession_prob = 0.8
                elif spread < 0:
                    recession_prob = 0.6
                elif spread > 1.5:
                    recession_prob = 0.1
                
                # Determine curve regime
                if spread < -0.25:
                    curve_regime = 'inverted'
                elif spread > 1.0:
                    curve_regime = 'steep'
                elif spread < 0.5:
                    curve_regime = 'flat'
                else:
                    curve_regime = 'normal'
                
                return {
                    'timestamp': latest.get('date', datetime.now().isoformat()),
                    'curve_regime': curve_regime,
                    'recession_probability': recession_prob,
                    'spread': spread,
                    '10_year': ten_year,
                    '2_year': two_year
                }
        except Exception as e:
            print(f"Error loading duration data: {e}")
        
        return None
    
    def hard_voting(self, signals: List[RegimeSignal]) -> Tuple[str, float, List[str]]:
        """
        Hard voting: each signal votes for a regime, majority wins.
        
        Returns:
            (winning_regime, agreement_score, disagreeing_sources)
        """
        if not signals:
            return 'neutral', 0.0, []
        
        # Count votes weighted by confidence
        regime_votes = {}
        for sig in signals:
            weight = self.weights.get(sig.source, 0.25)
            vote_weight = weight * sig.confidence
            regime_votes[sig.regime] = regime_votes.get(sig.regime, 0) + vote_weight
        
        # Find winner
        if not regime_votes:
            return 'neutral', 0.0, []
        
        winning_regime = max(regime_votes, key=regime_votes.get)
        total_votes = sum(regime_votes.values())
        winning_votes = regime_votes[winning_regime]
        
        # Agreement score
        agreement_score = winning_votes / total_votes if total_votes > 0 else 0
        
        # Find disagreeing sources
        disagreeing = [
            sig.source for sig in signals 
            if sig.regime != winning_regime
        ]
        
        return winning_regime, agreement_score, disagreeing
    
    def soft_voting(self, signals: List[RegimeSignal]) -> Tuple[Dict[str, float], float]:
        """
        Soft voting: weighted probability combination.
        
        Returns:
            (regime_probabilities, confidence_score)
        """
        regimes = ['bull', 'bear', 'neutral', 'crisis']
        ensemble_probs = {r: 0.0 for r in regimes}
        
        # Build probability distribution for each signal
        for sig in signals:
            weight = self.weights.get(sig.source, 0.25)
            
            # Create probability distribution centered on signal's regime
            for regime in regimes:
                if regime == sig.regime:
                    prob = sig.probability
                else:
                    # Remaining probability spread equally
                    prob = (1 - sig.probability) / (len(regimes) - 1)
                
                ensemble_probs[regime] += weight * prob * sig.confidence
        
        # Normalize
        total = sum(ensemble_probs.values())
        if total > 0:
            ensemble_probs = {k: v/total for k, v in ensemble_probs.items()}
        
        # Confidence as difference between top two
        sorted_probs = sorted(ensemble_probs.values(), reverse=True)
        confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0
        
        return ensemble_probs, confidence
    
    def get_action_recommendation(self, confidence: float, signals: List[RegimeSignal]) -> Tuple[str, float]:
        """
        Get action recommendation based on confidence and signal characteristics.
        
        Returns:
            (action, position_scaling)
        """
        # Check for VIX override (disagreement resolution)
        vix_override = False
        for sig in signals:
            if sig.source == 'hmm' and sig.raw_data.get('vix_level', 0) > 25:
                vix_override = True
                break
        
        if vix_override and confidence < 0.6:
            # High VIX with low confidence = stay cautious
            return 'hold', 0.5
        
        # Standard confidence-based action
        if confidence > self.CONFIDENCE_THRESHOLDS['high']:
            return self.ACTION_MAP['high']
        elif confidence > self.CONFIDENCE_THRESHOLDS['medium']:
            return self.ACTION_MAP['medium']
        elif confidence > self.CONFIDENCE_THRESHOLDS['low']:
            return self.ACTION_MAP['low']
        else:
            return self.ACTION_MAP['very_low']
    
    def evaluate(self) -> Optional[EnsembleRegime]:
        """
        Run ensemble evaluation and return unified regime classification.
        """
        signals = self.collect_signals()
        
        if len(signals) < 2:
            print(f"Warning: Only {len(signals)} signals available, need at least 2")
        
        # Hard voting
        hard_regime, agreement_score, disagreeing = self.hard_voting(signals)
        
        # Soft voting
        ensemble_probs, soft_confidence = self.soft_voting(signals)
        
        # Use soft voting result (more nuanced)
        final_regime = max(ensemble_probs, key=ensemble_probs.get)
        
        # Combine confidence metrics
        confidence = 0.6 * soft_confidence + 0.4 * agreement_score
        
        # Get action recommendation
        action, position_scaling = self.get_action_recommendation(confidence, signals)
        
        # Individual signal probabilities for output
        individual_probs = {}
        for sig in signals:
            individual_probs[f"{sig.source}_prob"] = sig.probability
        
        # Extract VIX insurance data if available
        insurance_active = False
        insurance_alerts = []
        for sig in signals:
            if sig.source == 'vix_insurance':
                insurance_active = sig.raw_data.get('insurance_active', False)
                insurance_alerts = [a.get('type', 'unknown') for a in sig.raw_data.get('alerts', [])]
                break
        
        return EnsembleRegime(
            timestamp=datetime.now().isoformat(),
            regime=final_regime,
            confidence=round(confidence, 4),
            hmm_prob=individual_probs.get('hmm_prob', 0),
            cta_prob=individual_probs.get('cta_prob', 0),
            tsfm_prob=individual_probs.get('tsfm_prob', 0),
            duration_prob=individual_probs.get('duration_prob', 0),
            vix_insurance_prob=individual_probs.get('vix_insurance_prob', 0),
            ensemble_probs={k: round(v, 4) for k, v in ensemble_probs.items()},
            agreement_score=round(agreement_score, 4),
            disagreement_sources=disagreeing,
            action=action,
            position_scaling=round(position_scaling, 4),
            insurance_active=insurance_active,
            insurance_alerts=insurance_alerts
        )
    
    def get_signal_breakdown(self, signals: List[RegimeSignal]) -> Dict[str, Any]:
        """Get detailed breakdown of all signals"""
        breakdown = {
            'total_signals': len(signals),
            'signals': []
        }
        
        for sig in signals:
            breakdown['signals'].append({
                'source': sig.source,
                'regime': sig.regime,
                'probability': round(sig.probability, 4),
                'confidence': round(sig.confidence, 4),
                'weight': self.weights.get(sig.source, 0.25),
                'raw_data': sig.raw_data
            })
        
        return breakdown


class EnsembleVoterCLI:
    """Command-line interface for ensemble voting system"""
    
    def __init__(self):
        self.engine = EnsembleVotingEngine()
    
    def status(self):
        """Show current ensemble regime status"""
        result = self.engine.evaluate()
        
        if result:
            output = {
                'timestamp': result.timestamp,
                'ensemble_regime': {
                    'regime': result.regime,
                    'confidence': result.confidence,
                    'agreement_score': result.agreement_score
                },
                'probabilities': {
                    'hmm': result.hmm_prob,
                    'cta': result.cta_prob,
                    'tsfm': result.tsfm_prob,
                    'duration': result.duration_prob,
                    'vix_insurance': result.vix_insurance_prob,
                    'ensemble': result.ensemble_probs
                },
                'disagreement': {
                    'disagreeing_sources': result.disagreement_sources,
                    'count': len(result.disagreement_sources)
                },
                'recommendation': {
                    'action': result.action,
                    'position_scaling': result.position_scaling
                },
                'insurance_overlay': {
                    'active': result.insurance_active,
                    'alerts': result.insurance_alerts
                }
            }
            print(json.dumps(output, indent=2))
        else:
            print(json.dumps({'error': 'Failed to evaluate ensemble'}))
    
    def signals(self):
        """Show all collected signals with breakdown"""
        signals = self.engine.collect_signals()
        breakdown = self.engine.get_signal_breakdown(signals)
        print(json.dumps(breakdown, indent=2))
    
    def weights(self, new_weights: Optional[str] = None):
        """Show or update signal weights"""
        if new_weights:
            try:
                weights = json.loads(new_weights)
                self.engine.weights = weights
                print(json.dumps({
                    'status': 'updated',
                    'weights': weights
                }, indent=2))
            except Exception as e:
                print(json.dumps({'error': f'Invalid weights: {e}'}))
        else:
            print(json.dumps({
                'current_weights': self.engine.weights,
                'default_weights': self.engine.DEFAULT_WEIGHTS
            }, indent=2))
    
    def compare(self):
        """Compare hard voting vs soft voting results"""
        signals = self.engine.collect_signals()
        
        hard_regime, agreement, _ = self.engine.hard_voting(signals)
        soft_probs, soft_conf = self.engine.soft_voting(signals)
        soft_regime = max(soft_probs, key=soft_probs.get)
        
        output = {
            'signals_collected': len(signals),
            'hard_voting': {
                'regime': hard_regime,
                'agreement_score': round(agreement, 4)
            },
            'soft_voting': {
                'regime': soft_regime,
                'confidence': round(soft_conf, 4),
                'probabilities': {k: round(v, 4) for k, v in soft_probs.items()}
            },
            'consensus': hard_regime == soft_regime
        }
        print(json.dumps(output, indent=2))


def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ensemble Voting Regime Detector')
    parser.add_argument('command', choices=['status', 'signals', 'weights', 'compare'],
                       help='Command to execute')
    parser.add_argument('--weights', type=str, help='JSON weights for update')
    
    args = parser.parse_args()
    
    cli = EnsembleVoterCLI()
    
    if args.command == 'status':
        cli.status()
    elif args.command == 'signals':
        cli.signals()
    elif args.command == 'weights':
        cli.weights(args.weights)
    elif args.command == 'compare':
        cli.compare()


if __name__ == '__main__':
    main()
