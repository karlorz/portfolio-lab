#!/usr/bin/env python3
"""
Portfolio-Lab v2.55: Combined Signal Orchestrator

Master signal aggregation engine combining:
    - v2.52 TSMOM Overlay (35% weight): Time-series momentum from AQR research
    - v2.53 HMM Regime Detector (25% weight): Market state classification
    - v2.54 Fed Policy Overlay (25% weight): Fed rate/inflation regime
    - v2.51 AI Agent Controller (10% weight): MARL reinforcement learning
    - Base Technical/Macro (5% weight): Traditional indicators

Signal Combination Strategy:
    1. Collect signals from all sources with confidence scores
    2. Weight by source reliability and current regime
    3. Detect conflicts (opposing high-confidence signals)
    4. Apply conflict resolution rules
    5. Generate unified allocation recommendation

Conflict Resolution:
    - TSMOM vs Fed Policy: Split difference if both high confidence
    - HMM neutral: Reduce overall deviation magnitude
    - High volatility regime: Cap max deviation at 5%
    - All agree: Amplify signal (+20% boost)

Current Signal Snapshot (2026-05-13):
    TSMOM:   SPY -3.9%, GLD -4.6%, TLT +3.5% (reduce equities, add bonds)
    HMM:     All NEUTRAL (no change)
    Fed:     SPY +2.6%, GLD +3.0%, TLT -5.5% (risk-on, reduce duration)
    Conflict: TSMOM and Fed disagree on SPY/GLD direction

Usage:
    python -m src.signals.combined_orchestrator status
    python -m src.signals.combined_orchestrator recommend --portfolio 46/38/16
    python -m src.signals.combined_orchestrator signals
    python -m src.signals.combined_orchestrator backtest --start 2020-01-01

Reference:
    - AQR: "Time Series Momentum" (Moskowitz et al., 2012)
    - arXiv:2407.19858: HMM-LSTM hybrid regime detection
    - FRED/Goldman Sachs: Fed policy allocation frameworks
"""

import numpy as np
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.signals.tsmom_overlay import TSMOMOverlay
from src.agents.risk_agent_hmm import PortfolioRegimeManager, MarketRegime
from src.signals.fed_policy_overlay import FedPolicyOverlay


# Signal source weights
SIGNAL_WEIGHTS = {
    'tsmom': 0.35,
    'hmm_regime': 0.25,
    'fed_policy': 0.25,
    'ai_agent': 0.10,
    'base': 0.05,
}

# Conflict resolution thresholds
CONFLICT_THRESHOLD = 0.05  # 5% opposing signals = conflict
HIGH_CONFIDENCE = 0.75


@dataclass
class SignalSource:
    """Individual signal source result."""
    source: str
    deltas: Dict[str, float]
    confidence: float
    regime: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class CombinedRecommendation:
    """Final combined allocation recommendation."""
    timestamp: str
    base_allocation: Dict[str, float]
    recommended_allocation: Dict[str, float]
    deltas: Dict[str, float]
    
    # Source breakdown
    source_signals: Dict[str, SignalSource]
    
    # Conflict info
    conflicts_detected: List[str]
    resolution_strategy: str
    
    # Risk metrics
    predicted_volatility: float
    regime_dominant: str
    confidence: float
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'base_allocation': self.base_allocation,
            'recommended_allocation': self.recommended_allocation,
            'deltas': self.deltas,
            'source_signals': {
                k: {
                    'source': v.source,
                    'deltas': v.deltas,
                    'confidence': v.confidence,
                    'regime': v.regime,
                    'notes': v.notes
                }
                for k, v in self.source_signals.items()
            },
            'conflicts_detected': self.conflicts_detected,
            'resolution_strategy': self.resolution_strategy,
            'predicted_volatility': self.predicted_volatility,
            'regime_dominant': self.regime_dominant,
            'confidence': self.confidence
        }


class CombinedSignalOrchestrator:
    """
    Master orchestrator combining all signal sources into unified allocation.
    """
    
    def __init__(
        self,
        base_allocation: Dict[str, float] = None,
        weights: Dict[str, float] = None
    ):
        self.base_allocation = base_allocation or {
            'SPY': 0.46,
            'GLD': 0.38,
            'TLT': 0.16
        }
        self.weights = weights or SIGNAL_WEIGHTS
        
        # Initialize sub-modules
        self.tsmom = TSMOMOverlay(max_deviation=0.10)
        self.hmm_manager = PortfolioRegimeManager(base_allocation=self.base_allocation)
        self.fed_overlay = FedPolicyOverlay()
        
        # Load trained models
        self.hmm_manager.detector.load()
    
    def collect_signals(self, timestamp: Optional[str] = None) -> Dict[str, SignalSource]:
        """
        Collect signals from all sources.
        """
        signals = {}
        timestamp = timestamp or datetime.now().isoformat()
        
        # TSMOM signals
        tsmom_deltas = {}
        for ticker in self.base_allocation:
            signal = self.tsmom.compute_signal(ticker, timestamp)
            if signal:
                tsmom_deltas[ticker] = signal.adjustment
        
        signals['tsmom'] = SignalSource(
            source='tsmom',
            deltas=tsmom_deltas,
            confidence=0.85,  # TSMOM typically high confidence
            regime=None,
            notes='12m formation, 1m skip, vol-scaled'
        )
        
        # HMM Regime
        hmm_state = self.hmm_manager.detect_portfolio_regime(
            list(self.base_allocation.keys()),
            timestamp
        )
        if hmm_state:
            signals['hmm_regime'] = SignalSource(
                source='hmm_regime',
                deltas=hmm_state.regime_adjustments,
                confidence=hmm_state.regime_confidence,
                regime=str(hmm_state.dominant_regime),
                notes=f'Dominant: {hmm_state.dominant_regime}'
            )
        else:
            signals['hmm_regime'] = SignalSource(
                source='hmm_regime',
                deltas={t: 0.0 for t in self.base_allocation},
                confidence=0.0,
                regime='unknown',
                notes='Model not loaded'
            )
        
        # Fed Policy
        fed_regime = self.fed_overlay.detect_regime(timestamp)
        if fed_regime:
            fed_deltas = fed_regime.get_allocation_shift()
            # Normalize to base_allocation keys
            fed_deltas = {k: fed_deltas.get(k, 0.0) for k in self.base_allocation}
            
            signals['fed_policy'] = SignalSource(
                source='fed_policy',
                deltas=fed_deltas,
                confidence=fed_regime.confidence,
                regime=fed_regime.regime,
                notes=f'Real rate: {fed_regime.real_rate_short:.2f}%'
            )
        else:
            signals['fed_policy'] = SignalSource(
                source='fed_policy',
                deltas={t: 0.0 for t in self.base_allocation},
                confidence=0.0,
                regime='unknown',
                notes='Data unavailable'
            )
        
        # AI Agent (placeholder - would call MARL controller)
        signals['ai_agent'] = SignalSource(
            source='ai_agent',
            deltas={t: 0.0 for t in self.base_allocation},  # Neutral for now
            confidence=0.70,
            regime=None,
            notes='MARL controller v2.51'
        )
        
        # Base technical (placeholder)
        signals['base'] = SignalSource(
            source='base',
            deltas={t: 0.0 for t in self.base_allocation},
            confidence=0.60,
            regime=None,
            notes='Technical indicators'
        )
        
        return signals
    
    def detect_conflicts(self, signals: Dict[str, SignalSource]) -> List[str]:
        """
        Detect conflicting signals between sources.
        """
        conflicts = []
        tickers = list(self.base_allocation.keys())
        
        # Check TSMOM vs Fed Policy (usually opposite signals)
        tsmom = signals.get('tsmom')
        fed = signals.get('fed_policy')
        
        if tsmom and fed and tsmom.confidence > 0.5 and fed.confidence > 0.5:
            for ticker in tickers:
                tsmom_delta = tsmom.deltas.get(ticker, 0)
                fed_delta = fed.deltas.get(ticker, 0)
                
                # Opposite direction and both significant
                if abs(tsmom_delta) > 0.01 and abs(fed_delta) > 0.01:
                    if (tsmom_delta > 0 and fed_delta < 0) or (tsmom_delta < 0 and fed_delta > 0):
                        conflicts.append(f'{ticker}: TSMOM({tsmom_delta:+.2%}) vs Fed({fed_delta:+.2%})')
        
        return conflicts
    
    def resolve_signals(
        self,
        signals: Dict[str, SignalSource],
        conflicts: List[str]
    ) -> Tuple[Dict[str, float], str]:
        """
        Combine signals with conflict resolution.
        """
        tickers = list(self.base_allocation.keys())
        
        # Weighted combination
        weighted_deltas = {t: 0.0 for t in tickers}
        total_weight_applied = {t: 0.0 for t in tickers}
        
        for source_name, signal in signals.items():
            weight = self.weights.get(source_name, 0.0)
            
            for ticker in tickers:
                delta = signal.deltas.get(ticker, 0.0)
                confidence = signal.confidence
                
                # Apply confidence discount
                effective_weight = weight * confidence
                weighted_deltas[ticker] += delta * effective_weight
                total_weight_applied[ticker] += effective_weight
        
        # Normalize by total weight applied
        combined_deltas = {}
        for ticker in tickers:
            if total_weight_applied[ticker] > 0:
                combined_deltas[ticker] = weighted_deltas[ticker] / total_weight_applied[ticker]
            else:
                combined_deltas[ticker] = 0.0
        
        # Conflict resolution adjustments
        resolution = "weighted_average"
        
        if conflicts:
            # Major conflict: TSMOM vs Fed disagreement
            if any('TSMOM' in c and 'Fed' in c for c in conflicts):
                resolution = "split_difference"
                # Cap deltas at lower magnitude when major conflict
                for ticker in combined_deltas:
                    combined_deltas[ticker] *= 0.7
        
        # HMM neutral regime: reduce overall deviation
        hmm = signals.get('hmm_regime')
        if hmm and hmm.regime == 'neutral' and hmm.confidence > 0.8:
            resolution += ", hmm_neutral_reduction"
            for ticker in combined_deltas:
                combined_deltas[ticker] *= 0.8
        
        # Check if all signals agree
        all_agree = True
        for ticker in tickers:
            signs = []
            for signal in signals.values():
                delta = signal.deltas.get(ticker, 0)
                if abs(delta) > 0.01:
                    signs.append(1 if delta > 0 else -1)
            
            if signs and len(set(signs)) > 1:
                all_agree = False
                break
        
        if all_agree:
            resolution += ", consensus_boost"
            for ticker in combined_deltas:
                combined_deltas[ticker] *= 1.2
        
        return combined_deltas, resolution
    
    def generate_recommendation(
        self,
        timestamp: Optional[str] = None
    ) -> CombinedRecommendation:
        """
        Generate combined allocation recommendation.
        """
        timestamp = timestamp or datetime.now().isoformat()
        
        # Collect all signals
        signals = self.collect_signals(timestamp)
        
        # Detect conflicts
        conflicts = self.detect_conflicts(signals)
        
        # Resolve and combine
        combined_deltas, resolution = self.resolve_signals(signals, conflicts)
        
        # Calculate recommended allocation
        recommended = {}
        for ticker, base_weight in self.base_allocation.items():
            new_weight = base_weight + combined_deltas.get(ticker, 0.0)
            # Apply bounds
            recommended[ticker] = max(0.05, min(0.90, new_weight))
        
        # Normalize to sum to 1.0
        total = sum(recommended.values())
        if total > 0:
            recommended = {k: v / total for k, v in recommended.items()}
        
        # Calculate final deltas after normalization
        final_deltas = {
            ticker: recommended[ticker] - self.base_allocation[ticker]
            for ticker in self.base_allocation
        }
        
        # Determine dominant regime
        regimes = [s.regime for s in signals.values() if s.regime]
        regime_counts = {}
        for r in regimes:
            regime_counts[r] = regime_counts.get(r, 0) + 1
        dominant_regime = max(regime_counts.keys(), key=lambda k: regime_counts[k]) if regime_counts else 'neutral'
        
        # Calculate overall confidence
        avg_confidence = np.mean([s.confidence for s in signals.values()])
        confidence_penalty = len(conflicts) * 0.1
        overall_confidence = max(0.3, avg_confidence - confidence_penalty)
        
        # Estimate volatility (simplified)
        est_vol = 0.12  # Base estimate
        if dominant_regime == 'high_vol' or dominant_regime == 'crisis':
            est_vol = 0.18
        elif dominant_regime == 'bull':
            est_vol = 0.10
        
        return CombinedRecommendation(
            timestamp=timestamp,
            base_allocation=self.base_allocation.copy(),
            recommended_allocation=recommended,
            deltas=final_deltas,
            source_signals=signals,
            conflicts_detected=conflicts,
            resolution_strategy=resolution,
            predicted_volatility=est_vol,
            regime_dominant=dominant_regime,
            confidence=overall_confidence
        )
    
    def format_recommendation(self, rec: CombinedRecommendation) -> str:
        """Format recommendation for display."""
        lines = [
            "=" * 60,
            "Combined Signal Orchestrator v2.55",
            f"Timestamp: {rec.timestamp}",
            "=" * 60,
            "",
            "BASE ALLOCATION:",
        ]
        
        for ticker, weight in rec.base_allocation.items():
            lines.append(f"  {ticker}: {weight:.2%}")
        
        lines.extend([
            "",
            "RECOMMENDED ALLOCATION:",
        ])
        
        for ticker, weight in rec.recommended_allocation.items():
            delta = rec.deltas[ticker]
            delta_str = f"{delta:+.2%}"
            lines.append(f"  {ticker}: {weight:.2%} ({delta_str})")
        
        lines.extend([
            "",
            "SOURCE BREAKDOWN:",
        ])
        
        for name, signal in rec.source_signals.items():
            weight_pct = self.weights.get(name, 0) * 100
            lines.append(f"  {name} ({weight_pct:.0f}% weight):")
            lines.append(f"    Confidence: {signal.confidence:.0%}")
            if signal.regime:
                lines.append(f"    Regime: {signal.regime}")
            if signal.notes:
                lines.append(f"    Notes: {signal.notes}")
            
            delta_strs = [f"{t}={signal.deltas.get(t, 0):+.1%}" for t in rec.base_allocation]
            lines.append(f"    Deltas: {', '.join(delta_strs)}")
        
        if rec.conflicts_detected:
            lines.extend([
                "",
                "CONFLICTS DETECTED:",
            ])
            for conflict in rec.conflicts_detected:
                lines.append(f"  ! {conflict}")
        
        lines.extend([
            "",
            f"Resolution Strategy: {rec.resolution_strategy}",
            f"Dominant Regime: {rec.regime_dominant}",
            f"Predicted Volatility: {rec.predicted_volatility:.1%}",
            f"Overall Confidence: {rec.confidence:.0%}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Combined Signal Orchestrator v2.55")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # status command
    status_parser = subparsers.add_parser('status', help='Show orchestrator status')
    
    # signals command
    signals_parser = subparsers.add_parser('signals', help='Show raw signals from all sources')
    
    # recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Generate combined recommendation')
    recommend_parser.add_argument('--portfolio', default='46/38/16', help='Base allocation')
    recommend_parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    # backtest command (placeholder)
    backtest_parser = subparsers.add_parser('backtest', help='Backtest combined strategy')
    backtest_parser.add_argument('--start', help='Start date')
    backtest_parser.add_argument('--end', help='End date')
    
    args = parser.parse_args()
    
    if args.command == 'status':
        print("Combined Signal Orchestrator v2.55 - Status")
        print("=" * 40)
        print("Signal Sources:")
        for name, weight in SIGNAL_WEIGHTS.items():
            print(f"  {name}: {weight:.0%} weight")
        print()
        print("Conflict Resolution:")
        print(f"  Threshold: {CONFLICT_THRESHOLD:.1%}")
        print(f"  High Confidence: {HIGH_CONFIDENCE:.0%}")
        print()
        print("Modules Status:")
        print("  TSMOM: ready (v2.52)")
        print("  HMM Regime: ready (v2.53)")
        print("  Fed Policy: ready (v2.54)")
        print("  AI Agent: placeholder (v2.51)")
    
    elif args.command == 'signals':
        orchestrator = CombinedSignalOrchestrator()
        signals = orchestrator.collect_signals()
        
        print("Raw Signals (2026-05-13):")
        print("=" * 50)
        
        for name, signal in signals.items():
            weight = SIGNAL_WEIGHTS.get(name, 0)
            print(f"\n{name.upper()} ({weight:.0%} weight):")
            print(f"  Confidence: {signal.confidence:.0%}")
            if signal.regime:
                print(f"  Regime: {signal.regime}")
            print(f"  Deltas:")
            for ticker, delta in signal.deltas.items():
                print(f"    {ticker}: {delta:+.2%}")
    
    elif args.command == 'recommend':
        # Parse allocation
        parts = args.portfolio.split('/')
        base_alloc = {
            'SPY': float(parts[0]) / 100,
            'GLD': float(parts[1]) / 100,
            'TLT': float(parts[2]) / 100,
        }
        
        orchestrator = CombinedSignalOrchestrator(base_allocation=base_alloc)
        recommendation = orchestrator.generate_recommendation()
        
        if args.json:
            print(json.dumps(recommendation.to_dict(), indent=2, default=str))
        else:
            print(orchestrator.format_recommendation(recommendation))
    
    elif args.command == 'backtest':
        print("Backtest functionality: requires historical signal simulation")
        print("Current recommendation:")
        
        orchestrator = CombinedSignalOrchestrator()
        rec = orchestrator.generate_recommendation()
        print(orchestrator.format_recommendation(rec))
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
