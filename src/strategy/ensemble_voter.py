"""
Portfolio-Lab v2.58: Ensemble Signal Voter

Multi-source signal aggregation with regime-dependent weighting and health-adjusted weighting.
Implements soft voting with confidence-based consensus for portfolio decisions.

Sources:
- TSFM Factor Momentum (v2.15) - Factor-based momentum signals
- HMM Regime Detector (v2.20.1) - Latent state classification
- CTA Trend Overlay (v2.10+) - Multi-timeframe trend following
- Macro Momentum (v2.57) - Business cycle / monetary policy
- Multi-Speed Momentum (v2.56) - Speed-diversified trends
- Duration/Yield Curve (v2.17-2.18) - Rate regime detection
- Circuit Breaker (v2.14) - Risk limits and controls

Voting Strategy:
- Normal regime: TSFM 40%, MultiSpeed 25%, CTA 20%, Macro 10%, Duration 5%
- High vol regime: HMM 35%, CTA 30%, MultiSpeed 20%, Macro 10%, Circuit 5%
- Crisis regime: Circuit 35%, CTA 35%, HMM 20%, Macro 10%

Health-Adjusted Weighting (v3.12):
- Signals with health < 0.5 get weight reduced by 50%
- Signals with health >= 0.7 get full weight
- Health scores calculated from 90-day rolling accuracy

Consensus threshold: 2/3 weighted signals agree for action

Usage:
    python -m src.strategy.ensemble_voter vote
    python -m src.strategy.ensemble_voter recommend --portfolio 46/38/16
    python -m src.strategy.ensemble_voter explain
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import sys
import logging

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class Regime(Enum):
    """Market regime classifications."""
    NORMAL = "normal"
    HIGH_VOL = "high_vol"  
    CRISIS = "crisis"
    RECOVERY = "recovery"


class SignalSource(Enum):
    """Available signal sources."""
    TSFM_MOMENTUM = "tsfm_momentum"           # v2.15 Factor momentum
    HMM_REGIME = "hmm_regime"                 # v2.20.1 Wasserstein HMM
    CTA_TREND = "cta_trend"                   # v2.10+ CTA overlay
    MACRO_MOMENTUM = "macro_momentum"         # v2.57 Macro signals
    MULTI_SPEED_MOM = "multi_speed_momentum"  # v2.56 Multi-speed
    DURATION_REGIME = "duration_regime"       # v2.17-2.18 Yield curve
    CIRCUIT_BREAKER = "circuit_breaker"     # v2.14 Risk controls
    FACTOR_ROTATION = "factor_rotation"       # v3.00 Quality+Momentum overlay
    CLOSING_AUCTION = "closing_auction"       # v3.17 MOC/IOC imbalance signals
    UNIFIED_OVERLAY = "unified_overlay"       # v4.90 Multi-overlay orchestration
    MEAN_REVERSION = "mean_reversion"         # v4.81 VIX-gated mean-reversion
    TRANSFORMER_REGIME = "transformer_regime"  # v3.18 Transformer regime detector


@dataclass
class SignalReading:
    """Single signal source reading."""
    source: SignalSource
    timestamp: str
    
    # Signal value: -1 (strong short) to +1 (strong long)
    value: float
    
    # Metadata
    confidence: float  # 0-1
    weight: float    # Dynamic regime weight
    regime_fit: str  # Which regime this signal works best in
    
    # Asset-specific signals (optional)
    asset_signals: Optional[Dict[str, float]] = None
    
    # Reasoning
    explanation: str = ""


@dataclass
class EnsembleVote:
    """Aggregated ensemble decision."""
    timestamp: str
    regime: Regime
    regime_confidence: float
    
    # Consensus metrics
    num_sources: int
    weighted_consensus: float  # -1 to +1
    agreement_ratio: float     # % of signals agreeing with consensus
    
    # Per-asset recommendations
    equity_bias: float      # SPY direction
    duration_bias: float    # TLT direction
    gold_bias: float        # GLD direction
    
    # Final recommendation
    action: str            # "increase_equity", "decrease_equity", "neutral", "risk_off"
    confidence: float      # 0-1
    reasoning: str
    
    # Source breakdown
    source_votes: List[SignalReading]


# Regime-dependent weights
REGIME_WEIGHTS = {
    Regime.NORMAL: {
        SignalSource.TSFM_MOMENTUM: 0.35,
        SignalSource.MULTI_SPEED_MOM: 0.21,
        SignalSource.CTA_TREND: 0.15,
        SignalSource.MACRO_MOMENTUM: 0.09,
        SignalSource.FACTOR_ROTATION: 0.05,   # v3.00 Quality+Momentum overlay
        SignalSource.DURATION_REGIME: 0.05,
        SignalSource.MEAN_REVERSION: 0.03,    # v4.81 VIX-gated (mostly idle in normal)
        SignalSource.HMM_REGIME: 0.02,       # Minimal in normal
        SignalSource.CIRCUIT_BREAKER: 0.0,  # Off in normal
        SignalSource.CLOSING_AUCTION: 0.03,  # v3.17 MOC signals
        SignalSource.UNIFIED_OVERLAY: 0.02,  # v4.90 Multi-overlay orchestration
        SignalSource.TRANSFORMER_REGIME: 0.05,  # v3.18 Transformer regime detection
    },
    Regime.HIGH_VOL: {
        SignalSource.HMM_REGIME: 0.27,
        SignalSource.CTA_TREND: 0.24,
        SignalSource.MEAN_REVERSION: 0.08,   # v4.81 VIX-gated (active in high vol)
        SignalSource.MULTI_SPEED_MOM: 0.17,
        SignalSource.MACRO_MOMENTUM: 0.09,
        SignalSource.FACTOR_ROTATION: 0.05,   # v3.00 Quality+Momentum overlay
        SignalSource.CIRCUIT_BREAKER: 0.05,
        SignalSource.TSFM_MOMENTUM: 0.02,
        SignalSource.DURATION_REGIME: 0.0,
        SignalSource.CLOSING_AUCTION: 0.03,  # v3.17 MOC signals
        SignalSource.TRANSFORMER_REGIME: 0.08,  # v3.18 Most useful in volatile transitions
    },
    Regime.CRISIS: {
        SignalSource.CIRCUIT_BREAKER: 0.30,
        SignalSource.CTA_TREND: 0.30,
        SignalSource.HMM_REGIME: 0.18,
        SignalSource.MACRO_MOMENTUM: 0.09,
        SignalSource.FACTOR_ROTATION: 0.03,   # Reduced in crisis (defensive factor focus)
        SignalSource.MEAN_REVERSION: 0.03,    # v4.81 VIX-gated (mostly frozen in crisis)
        SignalSource.MULTI_SPEED_MOM: 0.03,
        SignalSource.TSFM_MOMENTUM: 0.0,
        SignalSource.DURATION_REGIME: 0.0,
        SignalSource.CLOSING_AUCTION: 0.03,  # v3.17 MOC signals
        SignalSource.UNIFIED_OVERLAY: 0.01,  # v4.90 Multi-overlay orchestration
        SignalSource.TRANSFORMER_REGIME: 0.03,  # v3.18 Low weight in crisis (regime obvious)
    },
    Regime.RECOVERY: {
        SignalSource.MULTI_SPEED_MOM: 0.24,
        SignalSource.HMM_REGIME: 0.20,
        SignalSource.CTA_TREND: 0.17,
        SignalSource.TSFM_MOMENTUM: 0.13,
        SignalSource.MACRO_MOMENTUM: 0.09,
        SignalSource.FACTOR_ROTATION: 0.06,   # Higher in recovery (momentum captures)
        SignalSource.MEAN_REVERSION: 0.05,    # v4.81 VIX-gated (declining VIX, moderate)
        SignalSource.DURATION_REGIME: 0.03,
        SignalSource.CIRCUIT_BREAKER: 0.0,
        SignalSource.CLOSING_AUCTION: 0.03,  # v3.17 MOC signals
        SignalSource.TRANSFORMER_REGIME: 0.06,  # v3.18 Detect recovery transitions
    }
}


class EnsembleVoter:
    """
    Multi-source signal ensemble with regime-adaptive weighting.
    
    Collects signals from all strategy modules, applies regime-dependent
    weighting, and produces consensus recommendations.
    """
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        regime_detector: Optional[str] = None
    ):
        self.data_path = data_path or Path("~/projects/portfolio-lab/data").expanduser()
        self.db_path = self.data_path / "ensemble_signals.db"
        self._init_db()
        
        # Current readings cache
        self.current_readings: Dict[SignalSource, SignalReading] = {}
        self.current_regime: Regime = Regime.NORMAL
        self.current_regime_confidence: float = 0.5
    
    def _init_db(self):
        """Initialize signal history database."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ensemble_votes (
                    timestamp TEXT PRIMARY KEY,
                    regime TEXT,
                    regime_confidence REAL,
                    num_sources INTEGER,
                    consensus REAL,
                    agreement_ratio REAL,
                    equity_bias REAL,
                    duration_bias REAL,
                    gold_bias REAL,
                    action TEXT,
                    confidence REAL,
                    reasoning TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS source_readings (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    source TEXT,
                    value REAL,
                    confidence REAL,
                    weight REAL,
                    regime_fit TEXT,
                    explanation TEXT
                )
            """)
    
    def detect_regime(self, price_data: Optional[pd.DataFrame] = None) -> Tuple[Regime, float]:
        """
        Detect current market regime from available data.
        
        Uses simple heuristics (can be enhanced with HMM later):
        - Crisis: VIX > 30 or max drawdown > 10% over 20 days
        - High vol: VIX > 20 or vol of vol elevated
        - Recovery: Recent drawdown followed by positive momentum
        - Normal: Otherwise
        """
        if price_data is None:
            price_data = self._load_price_data()
        
        if price_data is None or price_data.empty:
            return Regime.NORMAL, 0.5
        
        # Compute key indicators
        spy = price_data.get('SPY', price_data.iloc[:, 0])
        returns = spy.pct_change().dropna()
        
        if len(returns) < 20:
            return Regime.NORMAL, 0.5
        
        # 20-day realized vol (annualized)
        vol_20d = returns.tail(20).std() * np.sqrt(252)
        
        # Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns / running_max - 1).iloc[-1]
        
        # 20-day momentum
        mom_20d = returns.tail(20).sum()
        
        # Regime detection
        if vol_20d > 0.30 or drawdown < -0.10:
            regime = Regime.CRISIS
            confidence = min(abs(drawdown) * 5, 0.9) if drawdown < -0.05 else 0.5
        elif vol_20d > 0.20 or (drawdown < -0.05 and mom_20d < 0):
            regime = Regime.HIGH_VOL
            confidence = min(vol_20d * 3, 0.8)
        elif drawdown < -0.03 and mom_20d > 0.02:
            regime = Regime.RECOVERY
            confidence = min(mom_20d * 20, 0.7)
        else:
            regime = Regime.NORMAL
            confidence = max(0.5, 1.0 - vol_20d * 2)
        
        return regime, confidence
    
    def _load_price_data(self) -> Optional[pd.DataFrame]:
        """Load price data from JSON."""
        prices_path = Path("~/projects/portfolio-lab/public/data/prices.json").expanduser()
        
        if not prices_path.exists():
            return None
        
        with open(prices_path) as f:
            data = json.load(f)
        
        frames = []
        for symbol, pdata in data.items():
            if isinstance(pdata, list) and len(pdata) > 0 and 'd' in pdata[0]:
                df = pd.DataFrame(pdata)
                df['date'] = pd.to_datetime(df['d'])
                df.set_index('date', inplace=True)
                df.rename(columns={'p': symbol}, inplace=True)
                frames.append(df[[symbol]])
        
        if frames:
            df = pd.concat(frames, axis=1)
            df.sort_index(inplace=True)
            return df
        
        return None
    
    def collect_signals(self, date: Optional[str] = None) -> Dict[SignalSource, SignalReading]:
        """
        Collect signals from all available sources.
        
        This aggregates:
        - Multi-speed momentum (primary trend signal)
        - Macro momentum (regime context)
        - CTA trend overlay (crisis alpha)
        """
        readings = {}
        
        # 1. Multi-Speed Momentum (v2.56)
        try:
            from src.signals.multi_speed_momentum import MultiSpeedMomentum
            msm = MultiSpeedMomentum()
            
            # Get ensemble signals for each asset
            msm_signals = {}
            for ticker in ['SPY', 'TLT', 'GLD']:
                try:
                    sig = msm.get_signal_for_ticker(ticker, date)
                    if sig is not None:
                        msm_signals[ticker] = sig
                except Exception as e:
                    pass
            
            if msm_signals:
                avg_signal = sum(msm_signals.values()) / len(msm_signals)
                readings[SignalSource.MULTI_SPEED_MOM] = SignalReading(
                    source=SignalSource.MULTI_SPEED_MOM,
                    timestamp=str(datetime.now()),
                    value=avg_signal,
                    confidence=0.7,
                    weight=0.0,
                    regime_fit="all",
                    asset_signals=msm_signals,
                    explanation=f"Multi-speed momentum: avg_signal={avg_signal:.3f}, assets={list(msm_signals.keys())}"
                )
        except ImportError:
            pass
        
        # 2. Macro Momentum (v2.57)
        try:
            from src.signals.macro_momentum import MacroMomentumEngine
            engine = MacroMomentumEngine()
            reading = engine.compute_reading(date)
            
            # Aggregate macro signal from biases
            macro_value = (reading.equity_bias + reading.duration_bias + reading.gold_bias) / 3
            
            readings[SignalSource.MACRO_MOMENTUM] = SignalReading(
                source=SignalSource.MACRO_MOMENTUM,
                timestamp=reading.timestamp,
                value=macro_value,
                confidence=0.6,
                weight=0.0,
                regime_fit=reading.regime_classification,
                asset_signals={
                    'SPY': reading.equity_bias,
                    'TLT': reading.duration_bias,
                    'GLD': reading.gold_bias
                },
                explanation=f"Regime: {reading.regime_classification}, Aggregate: {reading.aggregate_score:+.3f}"
            )
        except ImportError as e:
            pass
        
        # 3. CTA Trend (if available)
        # Placeholder - would load from existing CTA module
        
        # 4. Closing Auction Signal (v3.17)
        try:
            from src.signals.closing_auction import ClosingAuctionSignalGenerator, SignalConfidence
            
            # Load latest MOC signals from JSON if available
            signal_path = Path("data/signals/closing_auction.json")
            if signal_path.exists():
                with open(signal_path) as f:
                    signal_data = json.load(f)
                
                # Filter to tradeable signals with medium+ confidence
                tradeable = [
                    s for s in signal_data.get('tradeable_signals', [])
                    if s.get('confidence') in ['high', 'medium']
                ]
                
                if tradeable:
                    # Aggregate signal: average direction score
                    avg_direction = sum(s.get('direction_score', 0) for s in tradeable) / len(tradeable)
                    # Normalize to -1..1 range
                    signal_value = max(-1, min(1, avg_direction / 3))
                    
                    readings[SignalSource.CLOSING_AUCTION] = SignalReading(
                        source=SignalSource.CLOSING_AUCTION,
                        timestamp=signal_data.get('timestamp', str(datetime.now())),
                        value=signal_value,
                        confidence=0.6 if any(s.get('confidence') == 'high' for s in tradeable) else 0.5,
                        weight=0.0,
                        regime_fit="all",
                        asset_signals={s['symbol']: s.get('direction_score', 0) / 3 for s in tradeable},
                        explanation=f"MOC imbalance: {len(tradeable)} tradeable signals, avg_direction={avg_direction:+.2f}"
                    )
        except Exception as e:
            pass
        
        # 5. Factor Rotation Signal (v3.00)
        try:
            from src.signals.factor_rotation import FactorRotationIntegrator
            integrator = FactorRotationIntegrator()
            signal = integrator.get_signal_for_ensemble(date)
            
            readings[SignalSource.FACTOR_ROTATION] = SignalReading(
                source=SignalSource.FACTOR_ROTATION,
                timestamp=signal["date"],
                value=signal["signal_value"],
                confidence=signal["confidence"],
                weight=0.0,
                regime_fit=signal["direction"],
                asset_signals={
                    'MTUM': signal["factor_allocations"].get('MTUM', 0),
                    'QUAL': signal["factor_allocations"].get('QUAL', 0),
                    'USMV': signal["factor_allocations"].get('USMV', 0),
                    'VLUE': signal["factor_allocations"].get('VLUE', 0),
                },
                explanation=f"Factor rotation: {signal['rationale'][0] if signal['rationale'] else 'No additional info'}"
            )
        except ImportError:
            pass
        
        # 6. VIX-Gated Mean-Reversion Signal (v4.81)
        try:
            from src.strategy.mean_reversion_overlay import get_mean_reversion_ensemble_signals
            mr_signals = get_mean_reversion_ensemble_signals()
            mr = mr_signals.get("mean_reversion", {})
            
            if mr:
                readings[SignalSource.MEAN_REVERSION] = SignalReading(
                    source=SignalSource.MEAN_REVERSION,
                    timestamp=str(datetime.now()),
                    value=mr.get("signal_value", 0.0),
                    confidence=0.7 if mr.get("active") else 0.3,
                    weight=0.0,
                    regime_fit="high_vol",
                    asset_signals={
                        'SPY': mr.get("signal_value", 0.0),
                    },
                    explanation=f"Mean-reversion: {mr.get('rationale', 'idle')}, alloc={mr.get('allocation_pct', 0):.1f}%, VIX={mr.get('vix_level', 0):.1f}, regime={mr.get('vix_regime', 'N/A')}"
                )
        except ImportError:
            pass
        
        self.current_readings = readings
        return readings
    
    def compute_vote(
        self,
        readings: Optional[Dict[SignalSource, SignalReading]] = None,
        regime: Optional[Regime] = None,
        regime_confidence: Optional[float] = None
    ) -> EnsembleVote:
        """
        Compute ensemble vote with regime-dependent weighting.
        """
        if readings is None:
            readings = self.current_readings or self.collect_signals()
        
        if regime is None:
            regime, regime_confidence = self.detect_regime()
        
        if regime_confidence is None:
            regime_confidence = 0.5
        
        self.current_regime = regime
        self.current_regime_confidence = regime_confidence
        
        # Get weights for regime
        weights = REGIME_WEIGHTS[regime]
        
        # Apply health-adjusted weighting (v3.12)
        # Reduce weight for signals with poor health scores
        try:
            from src.signals.health_tracker import SignalHealthTracker
            health_tracker = SignalHealthTracker()
            health_scores = health_tracker.calculate_all_health_scores()
            
            if health_scores:
                adjusted_weights = {}
                for source_enum, base_weight in weights.items():
                    source_str = source_enum.value
                    if source_str in health_scores:
                        health = health_scores[source_str]
                        # Health multiplier: min 0.2, full weight at health >= 0.7
                        multiplier = max(0.2, min(1.0, health.health_score))
                        adjusted_weights[source_enum] = base_weight * multiplier
                        if health.health_score < 0.5:
                            logger.info(f"Health-adjusted {source_str}: weight {base_weight:.2%} → {adjusted_weights[source_enum]:.2%} (health={health.health_score:.2f})")
                    else:
                        adjusted_weights[source_enum] = base_weight  # No health data, use full weight
                
                # Normalize to sum to 1.0
                total = sum(adjusted_weights.values())
                if total > 0:
                    weights = {k: v / total for k, v in adjusted_weights.items()}
        except Exception as e:
            logger.warning(f"Could not apply health-adjusted weights: {e}")
        
        # Apply weights to readings
        weighted_signals = []
        for source, reading in readings.items():
            if source in weights:
                reading.weight = weights[source]
                weighted_signals.append(reading)
        
        if not weighted_signals:
            return EnsembleVote(
                timestamp=str(datetime.now()),
                regime=regime,
                regime_confidence=regime_confidence,
                num_sources=0,
                weighted_consensus=0.0,
                agreement_ratio=0.0,
                equity_bias=0.0,
                duration_bias=0.0,
                gold_bias=0.0,
                action="neutral",
                confidence=0.0,
                reasoning="No signals available",
                source_votes=[]
            )
        
        # Compute consensus - handle NaN values
        valid_signals = [
            (r.value, r.weight) 
            for r in weighted_signals 
            if not np.isnan(r.value)
        ]
        
        if valid_signals:
            total_weight = sum(w for _, w in valid_signals)
            if total_weight == 0:
                total_weight = 1.0
            weighted_consensus = sum(v * w for v, w in valid_signals) / total_weight
        else:
            weighted_consensus = 0.0
            total_weight = 1.0
        
        # Agreement ratio: % of weighted signals agreeing with consensus
        agreement = sum(
            r.weight for r in weighted_signals
            if np.sign(r.value) == np.sign(weighted_consensus) or abs(r.value) < 0.1
        ) / total_weight
        
        # Asset-specific consensus
        assets = ['SPY', 'TLT', 'GLD']
        asset_biases = {}
        
        for asset in assets:
            asset_signals = [
                (r.asset_signals.get(asset, 0), r.weight)
                for r in weighted_signals
                if r.asset_signals and asset in r.asset_signals and not np.isnan(r.asset_signals.get(asset, np.nan))
            ]
            
            if asset_signals:
                total_w = sum(w for _, w in asset_signals) or 1.0
                asset_biases[asset] = sum(v * w for v, w in asset_signals) / total_w
            else:
                asset_biases[asset] = weighted_consensus  # Fallback
        
        # Determine action
        equity_bias = asset_biases.get('SPY', weighted_consensus)
        duration_bias = asset_biases.get('TLT', 0)
        gold_bias = asset_biases.get('GLD', 0)
        
        if regime == Regime.CRISIS:
            action = "risk_off"
            action_confidence = regime_confidence
        elif equity_bias > 0.3 and agreement > 0.6:
            action = "increase_equity"
            action_confidence = agreement * abs(equity_bias)
        elif equity_bias < -0.3 and agreement > 0.6:
            action = "decrease_equity"
            action_confidence = agreement * abs(equity_bias)
        else:
            action = "neutral"
            action_confidence = 0.5
        
        # Build reasoning
        reasons = [
            f"Regime: {regime.value} (confidence: {regime_confidence:.1%})",
            f"Sources: {len(weighted_signals)}, Consensus: {weighted_consensus:+.3f}",
            f"Agreement: {agreement:.1%}",
            f"Equity bias: {equity_bias:+.3f}, Duration: {duration_bias:+.3f}, Gold: {gold_bias:+.3f}"
        ]
        
        for r in weighted_signals[:3]:
            reasons.append(f"  {r.source.value}: {r.value:+.3f} (w={r.weight:.2f}, conf={r.confidence:.1%})")
        
        vote = EnsembleVote(
            timestamp=str(datetime.now()),
            regime=regime,
            regime_confidence=regime_confidence,
            num_sources=len(weighted_signals),
            weighted_consensus=weighted_consensus,
            agreement_ratio=agreement,
            equity_bias=equity_bias,
            duration_bias=duration_bias,
            gold_bias=gold_bias,
            action=action,
            confidence=action_confidence,
            reasoning="\n".join(reasons),
            source_votes=weighted_signals
        )
        
        # Save to DB
        self._save_vote(vote)
        
        return vote
    
    def _save_vote(self, vote: EnsembleVote):
        """Save vote to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ensemble_votes
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                vote.timestamp,
                vote.regime.value,
                vote.regime_confidence,
                vote.num_sources,
                vote.weighted_consensus,
                vote.agreement_ratio,
                vote.equity_bias,
                vote.duration_bias,
                vote.gold_bias,
                vote.action,
                vote.confidence,
                vote.reasoning
            ))
    
    def recommend_allocation(
        self,
        base_allocation: Dict[str, float] = None,
        vote: Optional[EnsembleVote] = None,
        max_shift: float = 0.10
    ) -> Dict[str, Dict]:
        """
        Generate allocation recommendation based on ensemble vote.
        
        Returns shifts from base allocation for each asset.
        """
        if base_allocation is None:
            base_allocation = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        
        if vote is None:
            vote = self.compute_vote()
        
        # Apply shifts based on biases
        shifts = {
            'SPY': np.clip(vote.equity_bias * max_shift, -max_shift, max_shift),
            'TLT': np.clip(vote.duration_bias * max_shift, -max_shift, max_shift),
            'GLD': np.clip(vote.gold_bias * max_shift, -max_shift, max_shift),
        }
        
        # Risk-off override
        if vote.regime == Regime.CRISIS:
            shifts['SPY'] = -max_shift * 0.5  # Reduce equity
            shifts['GLD'] = max_shift * 0.3   # Increase gold
            shifts['TLT'] = max_shift * 0.2   # Increase bonds
        
        result = {}
        total_shift = 0
        
        for asset, base in base_allocation.items():
            shift = shifts.get(asset, 0)
            new_alloc = base + shift
            
            result[asset] = {
                'base': base,
                'shift': shift,
                'new': np.clip(new_alloc, 0.05, 0.95),  # Bounds
                'bias': shifts.get(asset, 0),
            }
            total_shift += shift
        
        # Normalize to sum to 1
        total = sum(r['new'] for r in result.values())
        for asset in result:
            result[asset]['new'] /= total
            result[asset]['normalized_shift'] = result[asset]['new'] - result[asset]['base']
        
        return {
            'assets': result,
            'regime': vote.regime.value,
            'confidence': vote.confidence,
            'action': vote.action,
            'consensus': vote.weighted_consensus,
            'timestamp': vote.timestamp
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Ensemble Signal Voter')
    subparsers = parser.add_subparsers(dest='command')
    
    # Vote command
    vote_parser = subparsers.add_parser('vote', help='Compute ensemble vote')
    vote_parser.add_argument('--date', help='Date for signal (default: latest)')
    
    # Recommend command
    rec_parser = subparsers.add_parser('recommend', help='Generate allocation recommendation')
    rec_parser.add_argument('--portfolio', default='46/38/16', help='Base allocation SPY/GLD/TLT')
    rec_parser.add_argument('--max-shift', type=float, default=0.10, help='Max allocation shift')
    
    # Explain command
    exp_parser = subparsers.add_parser('explain', help='Explain current vote reasoning')
    
    args = parser.parse_args()
    
    voter = EnsembleVoter()
    
    if args.command == 'vote':
        readings = voter.collect_signals(args.date)
        vote = voter.compute_vote(readings)
        
        print("\n=== Ensemble Vote ===")
        print(f"Timestamp: {vote.timestamp}")
        print(f"Regime: {vote.regime.value.upper()} (confidence: {vote.regime_confidence:.1%})")
        print(f"\nSources: {vote.num_sources}")
        print(f"Consensus: {vote.weighted_consensus:+.3f}")
        print(f"Agreement: {vote.agreement_ratio:.1%}")
        print(f"\nAsset Biases:")
        print(f"  Equity (SPY):   {vote.equity_bias:+.3f}")
        print(f"  Duration (TLT): {vote.duration_bias:+.3f}")
        print(f"  Gold (GLD):     {vote.gold_bias:+.3f}")
        print(f"\nRecommended Action: {vote.action.upper()}")
        print(f"Confidence: {vote.confidence:.1%}")
    
    elif args.command == 'recommend':
        weights = [float(w) / 100 for w in args.portfolio.split('/')]
        base = {'SPY': weights[0], 'GLD': weights[1], 'TLT': weights[2]}
        
        vote = voter.compute_vote()
        rec = voter.recommend_allocation(base, vote, args.max_shift)
        
        print(f"\n=== Allocation Recommendation ===")
        print(f"Base: {args.portfolio}")
        print(f"Regime: {rec['regime'].upper()} (confidence: {rec['confidence']:.1%})")
        print(f"Consensus: {rec['consensus']:+.3f}")
        print(f"\nRecommended Allocation:")
        
        for asset, data in rec['assets'].items():
            print(f"  {asset}: {data['base']:.1%} → {data['new']:.1%} (shift: {data['normalized_shift']:+.1%})")
    
    elif args.command == 'explain':
        vote = voter.compute_vote()
        
        print("\n=== Ensemble Vote Explanation ===")
        print(vote.reasoning)
        print(f"\nActive Sources ({len(vote.source_votes)}):")
        for src in vote.source_votes:
            print(f"  {src.source.value:25} | value: {src.value:+.3f} | weight: {src.weight:.2f} | conf: {src.confidence:.1%}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
