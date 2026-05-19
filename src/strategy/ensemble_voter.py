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
    HMM_REGIME = "hmm_regime"                 # v2.20.1 Wasserstein HMM  # DEPRECATED
    CTA_TREND = "cta_trend"                   # v2.10+ CTA overlay  # DEPRECATED
    MACRO_MOMENTUM = "macro_momentum"         # v2.57 Macro signals  # DEPRECATED
    MULTI_SPEED_MOM = "multi_speed_momentum"  # v2.56 Multi-speed
    DURATION_REGIME = "duration_regime"       # v2.17-2.18 Yield curve
    CIRCUIT_BREAKER = "circuit_breaker"     # v2.14 Risk controls  # DEPRECATED
    FACTOR_ROTATION = "factor_rotation"       # v3.00 Quality+Momentum overlay  # DEPRECATED
    CLOSING_AUCTION = "closing_auction"       # v3.17 MOC/IOC imbalance signals  # DEPRECATED
    UNIFIED_OVERLAY = "unified_overlay"       # v4.90 Multi-overlay orchestration  # DEPRECATED
    MEAN_REVERSION = "mean_reversion"         # v4.81 VIX-gated mean-reversion  # DEPRECATED
    TRANSFORMER_REGIME = "transformer_regime"  # v3.18 Transformer regime detector  # DEPRECATED
    TRANSIENT_FACTORS = "transient_factors"   # v5.01 Transient statistical factors  # DEPRECATED
    VISIBILITY_GRAPH = "visibility_graph"     # v5.41 VGRSI network-science indicator  # DEPRECATED
    VP_MACD = "vp_macd"                       # v5.55 Volume-Price Adjusted MACD  # DEPRECATED
    CROSS_ASSET_RV = "cross_asset_rv"         # v5.71 Cross-asset relative value
    REGIME_CLASSIFIER = "regime_classifier"   # v5.73 ML-Light Regime Predictor  # DEPRECATED
    FACTOR_TIMING = "factor_timing"          # v6.02 Factor timing (cross-sectional Z-scores)  # DEPRECATED
    RISK_BUDGET = "risk_budget"              # v6.04 Factor risk budgeting & scenario analysis  # DEPRECATED
    LLM_NARRATIVE = "llm_narrative"          # v7.01 LLM macro/narrative signal  # DEPRECATED
    TAX_AWARE = "tax_aware"                  # v7.03 Tax-aware rebalancing alpha  # DEPRECATED
    VIXY_HEDGE = "vixy_hedge"              # v7.04 Dynamic VIXY hedge sizing  # DEPRECATED
    MULTI_TIMEFRAME_FUSION = "multi_timeframe_fusion"  # v8.06 Multi-timeframe fusion  # DEPRECATED
    MACRO_REGIME_SYNTHESIS = "macro_regime_synthesis"  # v8.07 Meta-regime consensus  # DEPRECATED
    FX_CARRY = "fx_carry"                  # v3.15 FX Currency Carry  # DEPRECATED
    INTERNATIONAL_MOMENTUM = "international_momentum"  # v3.13 International equity momentum
    COMMODITY_CURVE = "commodity_curve"    # v3.20 Commodity curve overlay  # DEPRECATED
    ALTERNATIVE_DATA = "alternative_data"  # v9.00 Alternative data signal (SEC EDGAR, NewsAPI, jobs)
    CROSS_ASSET_REGIME_ARB = "cross_asset_regime_arb"  # v8.09 Cross-asset regime arbitrage  # DEPRECATED
    ZERO_DTE = "zero_dte"  # v3.12 0DTE options yield enhancement  # DEPRECATED


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


# Regime-dependent weights (6 survivor signals, renormalized per regime)
REGIME_WEIGHTS = {
    Regime.NORMAL: {
        SignalSource.TSFM_MOMENTUM: 0.5000,
        SignalSource.MULTI_SPEED_MOM: 0.3400,
        SignalSource.DURATION_REGIME: 0.1000,
        SignalSource.ALTERNATIVE_DATA: 0.0400,
        SignalSource.CROSS_ASSET_RV: 0.0100,
        SignalSource.INTERNATIONAL_MOMENTUM: 0.0100,
    },
    Regime.HIGH_VOL: {
        SignalSource.MULTI_SPEED_MOM: 0.7143,
        SignalSource.TSFM_MOMENTUM: 0.0714,
        SignalSource.CROSS_ASSET_RV: 0.0714,
        SignalSource.INTERNATIONAL_MOMENTUM: 0.0714,
        SignalSource.ALTERNATIVE_DATA: 0.0714,
        SignalSource.DURATION_REGIME: 0.0000,
    },
    Regime.CRISIS: {
        SignalSource.MULTI_SPEED_MOM: 0.7500,
        SignalSource.CROSS_ASSET_RV: 0.2500,
        SignalSource.TSFM_MOMENTUM: 0.0000,
        SignalSource.DURATION_REGIME: 0.0000,
        SignalSource.INTERNATIONAL_MOMENTUM: 0.0000,
        SignalSource.ALTERNATIVE_DATA: 0.0000,
    },
    Regime.RECOVERY: {
        SignalSource.MULTI_SPEED_MOM: 0.5000,
        SignalSource.TSFM_MOMENTUM: 0.3125,
        SignalSource.DURATION_REGIME: 0.0625,
        SignalSource.ALTERNATIVE_DATA: 0.0625,
        SignalSource.CROSS_ASSET_RV: 0.03125,
        SignalSource.INTERNATIONAL_MOMENTUM: 0.03125,
    }
}


# ── Epsilon-Greedy Contextual Bandit for Dynamic Signal Weighting ──

class BanditWeighter:
    """Epsilon-greedy contextual bandit for dynamic signal weight adaptation.

    Tracks rolling Sharpe per (signal, regime_bin). With epsilon probability
    explores a random signal; otherwise exploits the best-performing signal
    for the current regime. Softmax converts Sharpe estimates to weights.

    No external dependencies -- pure numpy.
    """
    def __init__(
        self,
        signals: list,
        epsilon: float = 0.1,
        window: int = 252,
        temperature: float = 1.0,
    ):
        self.signals = list(signals)
        self.epsilon = epsilon
        self.window = window
        self.temperature = temperature
        self._history: dict = {}

    def select(self, regime: str) -> str:
        """Select a signal using epsilon-greedy strategy."""
        import random
        if random.random() < self.epsilon:
            return random.choice(self.signals)
        best_signal = self.signals[0]
        best_sharpe = -float("inf")
        for sig in self.signals:
            sh = self._rolling_sharpe(sig, regime)
            if sh > best_sharpe:
                best_sharpe = sh
                best_signal = sig
        return best_signal

    def update(self, signal: str, regime: str, daily_return: float):
        """Record a daily return observation for a signal in a regime."""
        if regime not in self._history:
            self._history[regime] = {}
        if signal not in self._history[regime]:
            self._history[regime][signal] = []
        self._history[regime][signal].append(daily_return)
        if len(self._history[regime][signal]) > self.window:
            self._history[regime][signal] = \
                self._history[regime][signal][-self.window:]

    def get_weights(self, regime: str):
        """Get softmax-normalized weights for all signals in a regime.

        Returns None if insufficient data for this regime (cold start).
        Returns dict mapping signal_name -> weight (sums to 1.0).
        """
        if regime not in self._history:
            return None
        sharpes = {}
        for sig in self.signals:
            sharpes[sig] = self._rolling_sharpe(sig, regime)
        # Check minimum history requirement
        total_obs = sum(
            len(self._history.get(regime, {}).get(s, []))
            for s in self.signals
        )
        if total_obs < len(self.signals) * 21:
            return None
        return self._softmax(sharpes)

    def _rolling_sharpe(self, signal: str, regime: str) -> float:
        """Compute rolling Sharpe ratio for a signal in a regime."""
        hist = self._history.get(regime, {}).get(signal, [])
        if len(hist) < 21:
            return 0.0
        arr = np.array(hist[-self.window:])
        mu = np.mean(arr)
        sigma = np.std(arr)
        if sigma < 1e-12:
            return 0.0
        return float(mu / sigma * np.sqrt(252))

    def _softmax(self, sharpes: dict) -> dict:
        """Convert Sharpe estimates to weights via softmax."""
        values = np.array([sharpes[s] for s in self.signals])
        values = values - np.max(values)  # numerical stability
        if self.temperature > 0:
            values = values / self.temperature
        exp_values = np.exp(values)
        total = np.sum(exp_values)
        if total < 1e-12:
            w = 1.0 / len(self.signals)
            return {s: w for s in self.signals}
        return {sig: float(exp_values[i] / total)
                for i, sig in enumerate(self.signals)}


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

        # Initialize bandit weighter for dynamic weight adaptation (vSpring Cleaning)
        survivor_values = [
            s.value for s in [
                SignalSource.TSFM_MOMENTUM,
                SignalSource.CROSS_ASSET_RV,
                SignalSource.INTERNATIONAL_MOMENTUM,
                SignalSource.ALTERNATIVE_DATA,
                SignalSource.MULTI_SPEED_MOM,
                SignalSource.DURATION_REGIME,
            ]
        ]
        self.bandit = BanditWeighter(
            signals=survivor_values,
            epsilon=0.1,
            window=252,
        )
        self._bandit_blend = 0.0
        self._bandit_observations = 0
    
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
        
        # 7. Transient Statistical Factors Signal (v5.01)
        try:
            from src.monitor.transient_factors import generate_ensemble_signal
            tf_signal = generate_ensemble_signal()
            
            if tf_signal and tf_signal.get("signal_value") is not None:
                sig_val = tf_signal["signal_value"]
                conf = tf_signal.get("confidence", 0.5)
                readings[SignalSource.TRANSIENT_FACTORS] = SignalReading(
                    source=SignalSource.TRANSIENT_FACTORS,
                    timestamp=str(datetime.now()),
                    value=sig_val,
                    confidence=conf,
                    weight=0.0,
                    regime_fit="high_vol",
                    asset_signals={
                        'SPY': sig_val,
                    },
                    explanation=f"Transient factors: stability={tf_signal.get('stability', 0):.2f}, trend={tf_signal.get('trend', 'N/A')}, n_factors={tf_signal.get('n_factors', 0)}, transition={tf_signal.get('transition_score', 0):.2f}"
                )
        except ImportError:
            pass

        # 8. Visibility Graph Signal (v5.41)
        try:
            from src.signals.visibility_graph import get_ensemble_signal
            vg_signal = get_ensemble_signal()

            if vg_signal and vg_signal.get("signal_value") is not None:
                sig_val = vg_signal["signal_value"]
                conf = vg_signal.get("confidence", 0.5)
                readings[SignalSource.VISIBILITY_GRAPH] = SignalReading(
                    source=SignalSource.VISIBILITY_GRAPH,
                    timestamp=str(datetime.now()),
                    value=sig_val,
                    confidence=conf,
                    weight=0.0,
                    regime_fit="normal",
                    asset_signals={
                        'SPY': sig_val,
                    },
                    explanation=f"VGRSI: {vg_signal.get('rationale', 'N/A')}"
                )
        except ImportError:
            pass

        # 9. VP-MACD Signal (v5.55)
        try:
            from src.signals.vp_macd import generate_signal
            vp_signal = generate_signal(ticker="SPY")

            if vp_signal is not None and vp_signal.vp_macd_value is not None:
                readings[SignalSource.VP_MACD] = SignalReading(
                    source=SignalSource.VP_MACD,
                    timestamp=vp_signal.timestamp,
                    value=vp_signal.vp_macd_value,
                    confidence=vp_signal.confidence,
                    weight=0.0,
                    regime_fit="all",
                    asset_signals={
                        'SPY': vp_signal.vp_macd_value,
                    },
                    explanation=f"VP-MACD: {vp_signal.vp_macd_signal}, hist={vp_signal.histogram:.4f}, thresh={vp_signal.volatility_adjusted_threshold:.4f}, vol={vp_signal.regime}"
                )
        except ImportError:
            pass

        # 10. Cross-Asset Relative Value (v5.71)
        try:
            from src.signals.cross_asset_relative_value import CrossAssetRVScanner
            rv_scanner = CrossAssetRVScanner()
            rv_signal = rv_scanner.get_ensemble_signal()

            if rv_signal.get("signal_value") is not None:
                readings[SignalSource.CROSS_ASSET_RV] = SignalReading(
                    source=SignalSource.CROSS_ASSET_RV,
                    timestamp=rv_signal.get("timestamp", str(datetime.now())),
                    value=rv_signal["signal_value"],
                    confidence=rv_signal.get("confidence", 0.5),
                    weight=0.0,
                    regime_fit="all",
                    asset_signals=rv_signal.get("asset_signals", {}),
                    explanation=f"Cross-asset RV: z={rv_signal.get('avg_z_score', 0):+.2f}, diverged={rv_signal.get('num_diverged', 0)}/{rv_signal.get('total_pairs', 0)} pairs"
                )
        except ImportError:
            pass

        # 11. Factor Timing Signal (v6.02)
        try:
            # Detect current regime or default to normal
            current_regime = self.detect_regime() if hasattr(self, 'detect_regime') else (None, 0.5)
            regime_name = current_regime[0].value if current_regime[0] else "normal"

            from src.signals.factor_timing_signal import get_ensemble_signal
            ft_signal = get_ensemble_signal(regime=regime_name)

            if ft_signal.get("active") and ft_signal.get("signal_value") is not None:
                readings[SignalSource.FACTOR_TIMING] = SignalReading(
                    source=SignalSource.FACTOR_TIMING,
                    timestamp=str(datetime.now()),
                    value=ft_signal["signal_value"],
                    confidence=ft_signal.get("confidence", 0.5),
                    weight=0.0,
                    regime_fit=regime_name,
                    asset_signals=ft_signal.get("factor_scores", {}),
                    explanation=f"Factor timing: top={ft_signal.get('top_factor', '?')} ({ft_signal.get('factor_scores', {}).get(ft_signal.get('top_factor', ''), 0):+.2f}σ), "
                                f"bottom={ft_signal.get('bottom_factor', '?')}, "
                                f"urgency={ft_signal.get('urgency', 0):.2f}, "
                                f"divergence={ft_signal.get('factor_divergence', 0):.2f}σ"
                )
        except Exception:
            pass

        # 12. LLM Narrative Signal (v7.01)
        try:
            from src.signals.llm_narrative_signal import get_narrative_signal
            narrative = get_narrative_signal()

            if narrative.get("value") is not None:
                asset_signals = narrative.get("asset_signals", {})
                readings[SignalSource.LLM_NARRATIVE] = SignalReading(
                    source=SignalSource.LLM_NARRATIVE,
                    timestamp=str(datetime.now()),
                    value=narrative["value"],
                    confidence=narrative.get("confidence", 0.3),
                    weight=0.0,
                    regime_fit="all",
                    asset_signals=asset_signals,
                    explanation=f"Macro narrative: {narrative.get('macro_health', '?')} "
                                f"(score={narrative['value']:+.2f}). "
                                f"FOMC: {narrative.get('fomc_tone', 'neutral')}. "
                                f"Signal: SPY={asset_signals.get('SPY', 0):+.2f}, "
                                f"TLT={asset_signals.get('TLT', 0):+.2f}, "
                                f"GLD={asset_signals.get('GLD', 0):+.2f}. "
                                f"{narrative.get('num_releases', 0)} releases analyzed."
                )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"LLM narrative signal unavailable: {e}")
            pass

        # 13. Macro Regime Meta-Synthesis (v8.07)
        try:
            from src.signals.macro_regime_synthesis import MetaRegimeSynthesizer
            mrs = MetaRegimeSynthesizer()
            ensemble_signal = mrs.get_ensemble_signal()
            regime_name, regime_conf = mrs.get_regime_for_ensemble_voter()

            readings[SignalSource.MACRO_REGIME_SYNTHESIS] = SignalReading(
                source=SignalSource.MACRO_REGIME_SYNTHESIS,
                timestamp=str(datetime.now()),
                value=ensemble_signal,
                confidence=regime_conf,
                weight=0.0,
                regime_fit=regime_name,
                asset_signals={
                    'SPY': ensemble_signal,
                },
                explanation=f"Macro meta-regime: consensus={regime_name}, signal={ensemble_signal:+.4f}, confidence={regime_conf:.1%}"
            )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Macro regime synthesis unavailable: {e}")
            pass

        # 14. FX Currency Carry (v3.15)
        try:
            from src.signals.fx_carry_signal import FXCarrySignalGenerator
            fx_gen = FXCarrySignalGenerator()
            fx_signal = fx_gen.generate_signal()

            if fx_signal.is_valid:
                # Map signal_type to numeric value: usd_strength=-1, usd_weakness=+1, neutral=0
                signal_map = {"usd_strength": -0.5, "usd_weakness": 0.5, "neutral": 0.0}
                signal_value = signal_map.get(fx_signal.signal_type, 0.0)

                readings[SignalSource.FX_CARRY] = SignalReading(
                    source=SignalSource.FX_CARRY,
                    timestamp=fx_signal.timestamp,
                    value=signal_value,
                    confidence=fx_signal.confidence,
                    weight=0.0,
                    regime_fit="all",
                    asset_signals={
                        'SPY': fx_signal.spy_shift,
                        'EFA': fx_signal.efa_shift,
                        'VXUS': fx_signal.vxus_shift,
                    },
                    explanation=f"FX Carry: {fx_signal.signal_type}, "
                                f"regime={fx_signal.regime}, dir={fx_signal.direction}, "
                                f"reason={fx_signal.reason}, spy_shift={fx_signal.spy_shift:+.1f}%"
                )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"FX Carry signal unavailable: {e}")
            pass

        # 15. International Equity Momentum (v3.13)
        try:
            from src.signals.international_momentum import InternationalMomentumGenerator

            # Load price data for SPY, EFA, EEM
            price_data = self._load_price_data()
            if price_data is not None and not price_data.empty:
                # Compute basic 6-month momentum for international comparison
                window = 126  # ~6 months of trading days
                required_cols = [c for c in ['SPY', 'EFA', 'EEM'] if c in price_data.columns]
                if len(required_cols) >= 2:
                    # Build data dict in format expected by InternationalMomentumGenerator
                    recent = price_data[required_cols].iloc[-window:] if len(price_data) >= window else price_data[required_cols]
                    if len(recent) >= 20:
                        efa_mom = (recent['EFA'].iloc[-1] / recent['EFA'].iloc[0] - 1) * 100 if 'EFA' in recent else 0.0
                        eem_mom = (recent['EEM'].iloc[-1] / recent['EEM'].iloc[0] - 1) * 100 if 'EEM' in recent else 0.0
                        spy_mom = (recent['SPY'].iloc[-1] / recent['SPY'].iloc[0] - 1) * 100

                        data = {
                            'timestamp': str(datetime.now()),
                            'relative': {
                                'efa_momentum_6m': efa_mom,
                                'eem_momentum_6m': eem_mom,
                                'spy_momentum_6m': spy_mom,
                                'efa_vs_spy': efa_mom - spy_mom,
                                'eem_vs_spy': eem_mom - spy_mom,
                            },
                            'data_fresh': True,
                        }

                        intl_gen = InternationalMomentumGenerator()
                        intl_signal = intl_gen.generate_signal(data)
                        signal_value = 0.0
                        if intl_signal.signal_type == "efa_lead":
                            signal_value = 0.3
                        elif intl_signal.signal_type == "eem_lead":
                            signal_value = 0.4
                        elif intl_signal.signal_type == "neutral":
                            signal_value = 0.0

                        readings[SignalSource.INTERNATIONAL_MOMENTUM] = SignalReading(
                            source=SignalSource.INTERNATIONAL_MOMENTUM,
                            timestamp=intl_signal.timestamp,
                            value=signal_value,
                            confidence=intl_signal.confidence,
                            weight=0.0,
                            regime_fit="all",
                            asset_signals={
                                'SPY': intl_signal.spy_shift,
                                'EFA': intl_signal.efa_shift,
                                'EEM': intl_signal.eem_shift,
                            },
                            explanation=f"Intl Momentum: {intl_signal.signal_type}, "
                                        f"conf={intl_signal.confidence_level}, "
                                        f"EFA/SPY={efa_mom - spy_mom:+.2%}, "
                                        f"EEM/SPY={eem_mom - spy_mom:+.2%}, "
                                        f"VIX_filter={intl_signal.vix_filter_active}"
                        )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"International momentum unavailable: {e}")
            pass

        # 16. Commodity Curve Overlay (v3.20)
        try:
            from src.signals.commodity_curve import fetch_curve_signal, CurveRegime

            # Check DBC (broad commodity) curve regime
            dbc_signal = fetch_curve_signal("DBC")
            gsg_signal = fetch_curve_signal("GSG")

            # Compute consensus signal: backwardation=+1, flat=0, contango=-1
            def regime_value(r: CurveRegime) -> float:
                if r == CurveRegime.BACKWARDATION:
                    return 0.5
                elif r == CurveRegime.CONTANGO:
                    return -0.5
                return 0.0

            dbc_val = regime_value(dbc_signal.regime)
            gsg_val = regime_value(gsg_signal.regime)
            avg_signal = (dbc_val + gsg_val) / 2.0

            readings[SignalSource.COMMODITY_CURVE] = SignalReading(
                source=SignalSource.COMMODITY_CURVE,
                timestamp=str(datetime.now()),
                value=avg_signal,
                confidence=0.5,
                weight=0.0,
                regime_fit="all",
                asset_signals={
                    'DBC': dbc_val,
                    'GSG': gsg_val,
                },
                explanation=f"Commodity Curve: DBC={dbc_signal.regime.name}({dbc_signal.spread_pct:+.2f}%), "
                            f"GSG={gsg_signal.regime.name}({gsg_signal.spread_pct:+.2f}%), "
                            f"consensus={avg_signal:+.2f}"
            )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Commodity curve unavailable: {e}")
            pass

        # 17. Alternative Data (v9.00) — SEC EDGAR, NewsAPI, Jobs data
        try:
            alt_data_file = Path("~/projects/portfolio-lab/data/signals").expanduser() / "alternative_data_latest.json"
            if alt_data_file.exists():
                import json as json_mod
                with open(alt_data_file) as f:
                    alt_data = json_mod.load(f)

                regime_map = {"bull": 0.4, "bear": -0.4, "neutral": 0.0, "crisis": -0.7}
                signal_value = regime_map.get(alt_data.get("regime", "neutral"), 0.0)

                readings[SignalSource.ALTERNATIVE_DATA] = SignalReading(
                    source=SignalSource.ALTERNATIVE_DATA,
                    timestamp=alt_data.get("timestamp", str(datetime.now())),
                    value=signal_value,
                    confidence=alt_data.get("confidence", 0.5),
                    weight=0.0,
                    regime_fit="all",
                    asset_signals={"SPY": signal_value},
                    explanation=f"Alt Data: regime={alt_data.get('regime')}, "
                                f"prob={alt_data.get('probability', 0):.2f}, "
                                f"conf={alt_data.get('confidence', 0):.2f}"
                )
        except Exception as e:
            logger.debug(f"Alternative data unavailable: {e}")
            pass

        # 18. Cross-Asset Regime Arbitrage (v8.09)
        try:
            from src.signals.cross_asset_regime_arb import CrossAssetRegimeArbDetector
            arb_detector = CrossAssetRegimeArbDetector()
            arb_signal = arb_detector.get_ensemble_signal()

            if arb_signal.get("active") and arb_signal.get("signal_value") is not None:
                readings[SignalSource.CROSS_ASSET_REGIME_ARB] = SignalReading(
                    source=SignalSource.CROSS_ASSET_REGIME_ARB,
                    timestamp=arb_signal.get("timestamp", str(datetime.now())),
                    value=arb_signal["signal_value"],
                    confidence=arb_signal.get("confidence", 0.5),
                    weight=0.0,
                    regime_fit="all",
                    asset_signals=arb_signal.get("asset_signals", {}),
                    explanation=f"Cross-asset regime arb: pattern={arb_signal.get('pattern', '?')}, "
                                f"eq={arb_signal.get('equity_regime', '?')}, "
                                f"bd={arb_signal.get('bond_regime', '?')}, "
                                f"gd={arb_signal.get('gold_regime', '?')}, "
                                f"persist={arb_signal.get('persistence_days', 0)}d, "
                                f"{arb_signal.get('explanation', '')}"
                )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Cross-asset regime arb unavailable: {e}")
            pass

        # 19. 0DTE Options Yield Enhancement (v3.12)
        try:
            odte_state_file = Path("~/projects/portfolio-lab/data/odte_state.json").expanduser()
            if odte_state_file.exists():
                import json as json_mod
                with open(odte_state_file) as f:
                    odte_state = json_mod.load(f)

                # 0DTE signal: slightly positive when selling calls for yield
                # (implies modest bullish view with volatility premium capture)
                # Signal value based on active positions and recent performance
                active_positions = odte_state.get("active_positions", 0)
                recent_profit = odte_state.get("recent_profit", 0.0)
                cumulative_pnl = odte_state.get("cumulative_pnl", 0.0)

                # Base signal: small positive yield enhancement bias
                base_signal = 0.05

                # Adjust for active positions (more positions = more yield capture)
                position_bonus = min(0.10, active_positions * 0.03)

                # Adjust for recent performance
                perf_bonus = min(0.05, max(-0.05, recent_profit * 0.01))

                signal_value = min(0.20, max(-0.10, base_signal + position_bonus + perf_bonus))

                readings[SignalSource.ZERO_DTE] = SignalReading(
                    source=SignalSource.ZERO_DTE,
                    timestamp=str(datetime.now()),
                    value=signal_value,
                    confidence=0.4,  # Moderate confidence (yield enhancement, not directional)
                    weight=0.0,
                    regime_fit="normal",
                    asset_signals={"SPY": signal_value},
                    explanation=f"0DTE Yield Enhancement: {active_positions} active positions, "
                                f"recent P&L=${recent_profit:.2f}, "
                                f"cumulative=${cumulative_pnl:.2f}"
                )
            else:
                # No state file yet — signal is neutral but present
                readings[SignalSource.ZERO_DTE] = SignalReading(
                    source=SignalSource.ZERO_DTE,
                    timestamp=str(datetime.now()),
                    value=0.0,
                    confidence=0.3,
                    weight=0.0,
                    regime_fit="normal",
                    asset_signals={"SPY": 0.0},
                    explanation="0DTE Yield Enhancement: No active positions (state file not found)"
                )
        except Exception as e:
            logger.debug(f"0DTE signal unavailable: {e}")
            pass

        self.current_readings = readings
        return readings

    def get_blended_weights(self, regime_name: str) -> dict:
        """Get regime weights blended between static REGIME_WEIGHTS and bandit.

        Cold start: 100% static. After 252 observations: 30% static, 70% bandit.
        """
        # Get static weights for this regime
        regime_enum = getattr(Regime, regime_name, Regime.NORMAL)
        static = dict(REGIME_WEIGHTS.get(regime_enum, {}))

        # Get bandit weights
        bandit = self.bandit.get_weights(regime_name)

        if bandit is None:
            return static  # Cold start: 100% static

        # Blend: shifts from 100/0 to 30/70 static/bandit over 252 observations
        blend = min(0.7, self._bandit_observations / 252 * 0.7)

        blended = {}
        for sig_enum, static_w in static.items():
            sig_value = sig_enum.value
            bandit_w = bandit.get(sig_value, static_w)
            blended[sig_enum] = static_w * (1 - blend) + bandit_w * blend

        return blended

    def apply_goal_risk_budget(self, base_allocation: dict) -> dict:
        """Scale allocation weights based on investment goals from goals.json.

        Reads goals.json via src.config.goals, computes risk budget multiplier,
        and shifts allocation toward safer assets proportionally.
        """
        try:
            from src.config.goals import load_goals, get_risk_budget_multiplier
            goals = load_goals()
            risk_mult = get_risk_budget_multiplier(goals)
        except Exception:
            risk_mult = 1.0

        if risk_mult >= 1.0:
            return base_allocation

        safe_assets = {"SHY", "IEF", "BIL", "TLT"}
        total = sum(base_allocation.values()) if base_allocation else 1.0
        if total == 0:
            return base_allocation

        shifted = {}
        risky_reduction = 0.0
        for asset, weight in base_allocation.items():
            if asset in safe_assets:
                shifted[asset] = weight
            else:
                reduced = weight * risk_mult
                shifted[asset] = reduced
                risky_reduction += weight - reduced

        # Redistribute reduced risk to safe assets proportionally
        safe_total = sum(shifted.get(a, 0) for a in safe_assets if a in shifted)
        if safe_total > 0 and risky_reduction > 0:
            for asset in safe_assets:
                if asset in shifted:
                    shifted[asset] += risky_reduction * (shifted[asset] / safe_total)

        # Renormalize
        new_total = sum(shifted.values())
        if new_total == 0:
            return base_allocation
        return {k: v / new_total * total for k, v in shifted.items()}

    def update_bandit(self, signal_returns: dict, regime_name: str):
        """Update bandit with observed returns for each signal."""
        for sig_value, daily_return in signal_returns.items():
            self.bandit.update(
                signal=sig_value,
                regime=regime_name,
                daily_return=daily_return,
            )
        self._bandit_observations += 1

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
        
        # Get weights for regime (blended with bandit if available)
        weights = self.get_blended_weights(regime.name)
        
        # Apply adaptive ensemble weighting (v6.09) if available
        try:
            from src.strategy.adaptive_ensemble_weights import AdaptiveEnsembleWeights
            
            # Try to load latest attribution data
            attribution_dir = Path("~/projects/portfolio-lab/data/attribution").expanduser()
            attribution_files = sorted(attribution_dir.glob("attribution_*.json"), reverse=True)
            
            if attribution_files:
                with open(attribution_files[0]) as f:
                    attribution_data = json.load(f)
                
                # Check if attribution is stale (>7 days old)
                attr_timestamp = attribution_data.get("timestamp", "")
                if attr_timestamp:
                    attr_date = attr_timestamp[:10]
                    days_stale = (datetime.now() - datetime.strptime(attr_date, "%Y-%m-%d")).days
                else:
                    days_stale = 999
                
                if days_stale <= 7:
                    # Check if we have enough data points
                    sources = attribution_data.get("sources", {})
                    total_readings = sum(s.get("total_readings", 0) for s in sources.values())
                    num_sources = len(sources)
                    avg_readings = total_readings / max(num_sources, 1)
                    
                    if avg_readings >= 5:  # Minimum average readings to enable adaptive
                        # Build base weights in string-keyed format
                        base_str = {k.value: v for k, v in weights.items()}
                        
                        adaptive = AdaptiveEnsembleWeights(base_weights=base_str)
                        adapted = adaptive.update_weights(attribution_data, regime.value)
                        
                        # Convert back to enum-keyed dict for EnsembleVoter
                        adaptive_weights_enum = {}
                        for source_enum in weights:
                            source_str = source_enum.value
                            if source_str in adapted:
                                adaptive_weights_enum[source_enum] = adapted[source_str]
                        
                        if adaptive_weights_enum:
                            logger.info(f"Using adaptive ensemble weights for regime={regime.value}")
                            weights = adaptive_weights_enum
        except Exception as e:
            logger.warning(f"Could not apply adaptive ensemble weights: {e}")
        
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
        
        # Apply turnover-aware weight validation (v8.01)
        # Penalizes signals that cause excessive rebalancing
        try:
            from src.strategy.turnover_validator import TurnoverValidator
            turnover_validator = TurnoverValidator()
            
            # Build signal_values dict from current readings
            signal_values = {}
            for source_enum in readings:
                source_str = source_enum.value
                reading = readings[source_enum]
                if not np.isnan(reading.value):
                    signal_values[source_str] = reading.value
            
            if signal_values:
                # Build base weights dict from regime weights (string-keyed)
                base_weights_str = {}
                for source_enum, w in weights.items():
                    base_weights_str[source_enum.value] = w
                
                # --- v8.02: Basis-Pursuit Signal Selection ---
                # Prune redundant and near-zero signals via L1 regularization
                try:
                    from src.strategy.basis_pursuit_selector import BasisPursuitSelector
                    bp_selector = BasisPursuitSelector()
                    bp_result = bp_selector.select_signals(
                        signal_values, base_weights_str, regime=regime.value
                    )
                    base_weights_str = bp_result.active_signals
                    sparsity_msg = (
                        f" (sparsity={bp_result.sparsity_ratio:.2f}, "
                        f"{bp_result.num_pruned} pruned)"
                        if bp_result.num_pruned > 0
                        else ""
                    )
                    logger.debug(f"Basis-pursuit selection applied{sparsity_msg}")
                except Exception as bp_e:
                    logger.warning(f"Could not apply basis-pursuit selection: {bp_e}")
                
                # --- v8.03: Regret-Weighted Adjustment ---
                # Penalize signals with high regret (covariance with ensemble decision)
                try:
                    from src.strategy.regret_weighted_selector import RegretWeightedSelector
                    rw_selector = RegretWeightedSelector()
                    # Use persisted previous ensemble decision, defaulting to 0.0
                    prev_decision = getattr(rw_selector.state, 'last_ensemble_decision', 0.0)
                    rw_result = rw_selector.adjust_weights(
                        signal_values, prev_decision, base_weights_str, regime=regime.value
                    )
                    base_weights_str = rw_result.adjusted_weights
                    if rw_result.signals_with_high_regret:
                        logger.info(
                            f"Regret-adjusted weights: penalized "
                            f"{', '.join(rw_result.signals_with_high_regret)}"
                            f" (avg_regret={rw_result.avg_regret:.3f})"
                        )
                except Exception as rw_e:
                    logger.warning(f"Could not apply regret-weighted adjustment: {rw_e}")
                
                # Apply turnover adjustment
                adjusted_str = turnover_validator.get_adjusted_weights(
                    base_weights_str, signal_values
                )
                
                # Convert back to enum-keyed dict
                turnover_adjusted = {}
                for source_enum in weights:
                    source_str = source_enum.value
                    if source_str in adjusted_str:
                        turnover_adjusted[source_enum] = adjusted_str[source_str]
                    else:
                        turnover_adjusted[source_enum] = weights[source_enum]
                
                # Re-normalize to sum to 1.0
                total = sum(turnover_adjusted.values())
                if total > 0:
                    weights = {k: v / total for k, v in turnover_adjusted.items()}
                    
                logger.debug(
                    f"Turnover-adjusted {len(signal_values)} signals: "
                    f"{', '.join(f'{s}={turnover_adjusted.get(enum, 0):.4f}' for enum, s in [(e, e.value) for e in weights])}"
                )
        except Exception as e:
            logger.warning(f"Could not apply turnover-aware weights: {e}")
        
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
        
        # Persist ensemble decision for next regret-weighted cycle (v8.03)
        try:
            from src.strategy.regret_weighted_selector import RegretWeightedSelector
            rw_selector = RegretWeightedSelector()
            rw_selector.state.last_ensemble_decision = weighted_consensus
            rw_selector._save_state()
        except Exception as rw_e:
            logger.debug(f"Could not persist ensemble decision to regret-weighted state: {rw_e}")
        
        # Save to DB
        self._save_vote(vote)
        
        return vote
    
    def _save_vote(self, vote: EnsembleVote):
        """Save vote to database, including per-source readings (v5.70)."""
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

            # v5.70: Save individual source readings for attribution
            for reading in vote.source_votes:
                try:
                    conn.execute("""
                        INSERT INTO source_readings
                        (timestamp, source, value, confidence, weight, regime_fit, explanation)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        vote.timestamp,
                        reading.source.value if hasattr(reading.source, 'value') else str(reading.source),
                        float(reading.value),
                        float(reading.confidence),
                        float(reading.weight),
                        reading.regime_fit or "",
                        (reading.explanation or "")[:500],
                    ))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to save source reading {reading.source}: {e}")
    
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
