"""
Portfolio-Lab v2.58: Ensemble Signal Voter

Multi-source signal aggregation with regime-dependent weighting and health-adjusted weighting.
Implements soft voting with confidence-based consensus for portfolio decisions.

Active Sources (6):
- Multi-Speed Momentum (v2.56) - Speed-diversified trends
- Cross-Asset Relative Value (v5.71) - Mean-reversion triggers
- International Equity Momentum (v3.13) - EFA/EEM vs SPY
- Alternative Data (v9.00) - SEC EDGAR, NewsAPI, jobs
- Cross-Asset Regime Arbitrage (v8.09) - Divergence detection
- Unified Overlay (v4.90) - Collar + bond + crypto + calendar

Weight Adjustments (applied in order):
1. Static REGIME_WEIGHTS (per-regime allocation)
2. Adaptive ensemble weighting (v6.09, from attribution data)
3. Health-adjusted weighting (v3.12, from signal health scores)
4. Turnover-aware validation (v8.01, with basis-pursuit + regret-weighted)

Consensus threshold: 2/3 weighted signals agree for action

Usage:
    python -m src.strategy.ensemble_voter vote
    python -m src.strategy.ensemble_voter recommend --portfolio 46/38/16
    python -m src.strategy.ensemble_voter explain
"""

import json
import random
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import sys
import logging

from src.paths import DATA_DIR, PRICES_JSON, ATTRIBUTION_DIR, BASE_ALLOCATION, PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class Regime(Enum):
    """Market regime classifications."""
    NORMAL = "normal"
    HIGH_VOL = "high_vol"  
    CRISIS = "crisis"
    RECOVERY = "recovery"


class SignalSource(Enum):
    """Available signal sources."""
    MULTI_SPEED_MOM = "multi_speed_momentum"  # v2.56 Multi-speed
    CROSS_ASSET_RV = "cross_asset_rv"         # v5.71 Cross-asset relative value
    INTERNATIONAL_MOMENTUM = "international_momentum"  # v3.13 International equity momentum
    ALTERNATIVE_DATA = "alternative_data"  # v9.00 Alternative data signal (SEC EDGAR, NewsAPI, jobs)
    CROSS_ASSET_REGIME_ARB = "cross_asset_regime_arb"  # v8.09 Cross-asset regime arbitrage
    UNIFIED_OVERLAY = "unified_overlay"       # v4.90 (ref'd by orchestrator_ensemble_bridge)


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


# Regime-dependent weights (6 active signals, renormalized per regime)
# v9.19: Removed TSFM_MOMENTUM and DURATION_REGIME (no data feeds in collect_signals).
# v9.23: Capped MULTI_SPEED_MOM at 50% to reduce single-point-of-failure.
# v9.26: Reduced MSM from 50% to 25%, redistributed to ALT_DATA and INTL_MOM.
# v9.35: Reduced MSM from 21% to 10% (net-negative -0.012 Sharpe per v9.24).
#        Redistributed excess to ALT_DATA (+0.015 Sharpe) and INTL_MOM (+0.02 Sharpe).
# Weights sum=1.0 per regime.
REGIME_WEIGHTS = {
    Regime.NORMAL: {
        SignalSource.MULTI_SPEED_MOM: 0.0000,
        SignalSource.CROSS_ASSET_RV: 0.1300,
        SignalSource.ALTERNATIVE_DATA: 0.3050,
        SignalSource.INTERNATIONAL_MOMENTUM: 0.2450,
        SignalSource.CROSS_ASSET_REGIME_ARB: 0.1300,
        SignalSource.UNIFIED_OVERLAY: 0.1900,
    },
    Regime.HIGH_VOL: {
        SignalSource.MULTI_SPEED_MOM: 0.0000,
        SignalSource.CROSS_ASSET_RV: 0.1300,
        SignalSource.INTERNATIONAL_MOMENTUM: 0.2100,
        SignalSource.ALTERNATIVE_DATA: 0.3300,
        SignalSource.CROSS_ASSET_REGIME_ARB: 0.1300,
        SignalSource.UNIFIED_OVERLAY: 0.2000,
    },
    Regime.CRISIS: {
        SignalSource.MULTI_SPEED_MOM: 0.0000,
        SignalSource.CROSS_ASSET_RV: 0.3650,
        SignalSource.CROSS_ASSET_REGIME_ARB: 0.1700,
        SignalSource.INTERNATIONAL_MOMENTUM: 0.0000,
        SignalSource.ALTERNATIVE_DATA: 0.2000,
        SignalSource.UNIFIED_OVERLAY: 0.2650,
    },
    Regime.RECOVERY: {
        SignalSource.MULTI_SPEED_MOM: 0.0000,
        SignalSource.ALTERNATIVE_DATA: 0.3050,
        SignalSource.CROSS_ASSET_RV: 0.1300,
        SignalSource.INTERNATIONAL_MOMENTUM: 0.2450,
        SignalSource.CROSS_ASSET_REGIME_ARB: 0.1300,
        SignalSource.UNIFIED_OVERLAY: 0.1900,
    }
}


# ── Epsilon-Greedy Contextual Bandit for Dynamic Signal Weighting ──

class BanditWeighter:
    """Epsilon-greedy contextual bandit for dynamic signal weight adaptation.

    Tracks rolling Sharpe per (signal, regime_bin). With epsilon probability
    explores a random signal; otherwise exploits the best-performing signal
    for the current regime. Softmax converts Sharpe estimates to weights.

    No external dependencies beyond numpy (already imported).
    """
    def __init__(
        self,
        signals: List[str],
        epsilon: float = 0.1,
        window: int = 252,
        temperature: float = 1.0,
    ):
        self.signals = list(signals)
        self.epsilon = epsilon
        self.window = window
        self.temperature = temperature
        # _history[regime][signal] = list of daily returns (rolling window)
        self._history: dict = {}

    def select(self, regime: str) -> str:
        """Select a signal using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.choice(self.signals)
        # Exploit: pick signal with best rolling Sharpe in this regime
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
        # Trim to window
        if len(self._history[regime][signal]) > self.window:
            self._history[regime][signal] = \
                self._history[regime][signal][-self.window:]

    def get_weights(self, regime: str) -> dict | None:
        """Get softmax-normalized weights for all signals in a regime.

        Returns None if no data exists for this regime (cold start).
        Returns dict mapping signal_name -> weight (sums to 1.0).
        """
        if regime not in self._history:
            return None
        sharpes = {}
        for sig in self.signals:
            sharpes[sig] = self._rolling_sharpe(sig, regime)
        return self._softmax(sharpes)

    def _rolling_sharpe(self, signal: str, regime: str) -> float:
        """Compute rolling Sharpe ratio for a signal in a regime."""
        hist = self._history.get(regime, {}).get(signal, [])
        if len(hist) < 21:  # Need at least 1 month
            return 0.0
        arr = np.array(hist[-self.window:])
        mu = np.mean(arr)
        sigma = np.std(arr)
        if sigma < 1e-10:
            return 0.0
        return float(mu / sigma * np.sqrt(252))

    def _softmax(self, sharpes: dict) -> dict:
        """Convert Sharpe estimates to weights via softmax."""
        values = np.array([sharpes[s] for s in self.signals])
        # Subtract max for numerical stability
        values = values - np.max(values)
        if self.temperature > 0:
            values = values / self.temperature
        exp_values = np.exp(values)
        total = np.sum(exp_values)
        if total < 1e-10:
            # All equal if everything is zero
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
        self.data_path = data_path or DATA_DIR
        self.db_path = self.data_path / "ensemble_signals.db"
        self._init_db()

        # Current readings cache
        self.current_readings: Dict[SignalSource, SignalReading] = {}
        self.current_regime: Regime = Regime.NORMAL
        self.current_regime_confidence: float = 0.5

        # Bandit weighter for dynamic signal weight adaptation
        self.bandit = BanditWeighter(
            signals=[s.value for s in SignalSource],
            epsilon=0.1,
            window=252,
        )
        self.bandit_observations: int = 0

        # Regime gate — disables signals in regimes where they are net-negative
        from src.signals.regime_gate import RegimeGate
        self.regime_gate = RegimeGate()
        self._prev_regime: Optional[str] = None
        self._days_in_regime: int = 999  # Start assuming stable regime

    
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
        prices_path = PRICES_JSON
        
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
    
    def collect_signals(self, date: Optional[str] = None, regime: Optional[Regime] = None) -> Dict[SignalSource, SignalReading]:
        """
        Collect signals from active (non-deprecated) sources.

        If regime is provided, skip signal sources with zero weight for that
        regime — avoids wasted computation on signals that won't affect the vote.

        Active sources (5 survivor signals per v9.19 pruning):
        - Multi-speed momentum (primary trend signal)
        - Cross-asset relative value (mean-reversion triggers)
        - International equity momentum (EFA/VXUS trend)
        - Alternative data (SEC EDGAR, NewsAPI, jobs)
        - Cross-asset regime arbitrage (divergence detection)

        Removed in v9.19 (deprecated, zero weight, no data feeds):
        MACRO_MOMENTUM, CLOSING_AUCTION, FACTOR_ROTATION, MEAN_REVERSION,
        TRANSIENT_FACTORS, VISIBILITY_GRAPH, VP_MACD, FACTOR_TIMING,
        LLM_NARRATIVE, MACRO_REGIME_SYNTHESIS, FX_CARRY, COMMODITY_CURVE,
        ZERO_DTE, TSFM_MOMENTUM (no data feed), DURATION_REGIME (no data feed).
        """
        # Determine which signals have non-zero weight for this regime
        active_sources = None
        if regime is not None:
            regime_weights = REGIME_WEIGHTS.get(regime, {})
            active_sources = {src for src, w in regime_weights.items() if w > 0}

        readings = {}

        # 1. Multi-Speed Momentum (v2.56) — typed SignalSnapshot bridge
        try:
            from src.signals.multi_speed_momentum import MultiSpeedMomentum
            msm = MultiSpeedMomentum()
            snapshot = msm.get_signal_snapshot(tickers=['SPY', 'TLT', 'GLD'], date=date)
            if snapshot.is_active:
                readings[SignalSource.MULTI_SPEED_MOM] = snapshot.to_signal_reading()
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Multi-speed momentum unavailable: {e}")

        # 2. Cross-Asset Relative Value (v5.71) — typed SignalSnapshot bridge
        try:
            from src.signals.cross_asset_relative_value import CrossAssetRVScanner
            rv_scanner = CrossAssetRVScanner()
            snapshot = rv_scanner.get_signal_snapshot()
            if snapshot.is_active:
                readings[SignalSource.CROSS_ASSET_RV] = snapshot.to_signal_reading()
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Cross-asset RV unavailable: {e}")

        # 3. International Equity Momentum (v3.13) — typed SignalSnapshot bridge
        # Skip if zero weight for current regime
        if active_sources is not None and SignalSource.INTERNATIONAL_MOMENTUM not in active_sources:
            logger.debug(f"Skipping INTERNATIONAL_MOMENTUM: zero weight for regime={regime.value if regime else '?' }")
        else:
            try:
                from src.signals.international_momentum import InternationalMomentumGenerator

                # Load price data for SPY, EFA, EEM
                price_data = self._load_price_data()
                if price_data is not None and not price_data.empty:
                    window = 126  # ~6 months of trading days
                    required_cols = [c for c in ['SPY', 'EFA', 'EEM'] if c in price_data.columns]
                    if len(required_cols) >= 2:
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
                            snapshot = intl_signal.to_signal_snapshot()
                            if snapshot.is_active:
                                readings[SignalSource.INTERNATIONAL_MOMENTUM] = snapshot.to_signal_reading()
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"International momentum unavailable: {e}")


        # 4. Alternative Data (v9.00) — typed SignalSnapshot bridge
        # Skip if zero weight for current regime
        if active_sources is not None and SignalSource.ALTERNATIVE_DATA not in active_sources:
            logger.debug(f"Skipping ALTERNATIVE_DATA: zero weight for regime={regime.value if regime else '?' }")
        else:
            try:
                from src.signals.alternative_data_signal import AlternativeDataSignalGenerator
                alt_gen = AlternativeDataSignalGenerator()
                snapshot = alt_gen.get_signal_snapshot()
                if snapshot.is_active:
                    readings[SignalSource.ALTERNATIVE_DATA] = snapshot.to_signal_reading()
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Alternative data unavailable: {e}")


        # 5. Cross-Asset Regime Arbitrage (v8.09) — typed SignalSnapshot bridge
        try:
            from src.signals.cross_asset_regime_arb import CrossAssetRegimeArbDetector
            arb_detector = CrossAssetRegimeArbDetector()
            snapshot = arb_detector.get_signal_snapshot()
            if snapshot.is_active:
                readings[SignalSource.CROSS_ASSET_REGIME_ARB] = snapshot.to_signal_reading()
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Cross-asset regime arb unavailable: {e}")


        # 6. Unified Overlay (v4.90) — collar + bond_duration + crypto + calendar
        # Skip if zero weight for current regime
        if active_sources is not None and SignalSource.UNIFIED_OVERLAY not in active_sources:
            logger.debug(f"Skipping UNIFIED_OVERLAY: zero weight for regime={regime.value if regime else '?' }")
        else:
            try:
                from .orchestrator_ensemble_bridge import OrchestratorEnsembleBridge
                bridge = OrchestratorEnsembleBridge()
                unified_reading = bridge.get_ensemble_reading()
                readings[SignalSource.UNIFIED_OVERLAY] = unified_reading
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Unified overlay unavailable: {e}")

        self.current_readings = readings
        return readings

    def get_blended_weights(self, regime_name: str) -> dict:
        """Get regime weights blended between static REGIME_WEIGHTS and bandit.

        Starts 100% static (bandit_blend=0.0), gradually shifts toward
        up to 70% bandit after 252 days of observations.
        """
        regime_enum = getattr(Regime, regime_name, Regime.NORMAL)
        static = dict(REGIME_WEIGHTS.get(regime_enum, {}))

        # If bandit not initialized (e.g. test fixtures bypassing __init__), fall back
        if not hasattr(self, 'bandit') or self.bandit is None:
            return static

        bandit = self.bandit.get_weights(regime_name)

        if bandit is None:
            return static  # Cold start: 100% static

        # Blend: starts 100% static, shifts to 30/70 static/bandit after 252 days
        blend = min(0.7, self.bandit_observations / 252 * 0.7)

        # Convert static keys from SignalSource enum to string values for matching
        static_by_value = {k.value: v for k, v in static.items()}

        blended = {}
        for sig_value in static_by_value:
            bandit_w = bandit.get(sig_value, 0.0)
            static_w = static_by_value[sig_value]
            blended[sig_value] = static_w * (1 - blend) + bandit_w * blend

        # Normalize to sum=1.0
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        # Convert back to SignalSource keys
        value_to_source = {s.value: s for s in SignalSource}
        return {value_to_source[k]: v for k, v in blended.items() if k in value_to_source}

    def update_bandit(self, signal_value: str, regime_name: str, daily_return: float):
        """Update bandit with observed return for a signal in a regime."""
        self.bandit.update(signal_value, regime_name, daily_return)
        self.bandit_observations += 1

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
            logger.warning("Failed to load goals for risk budget, using risk_mult=1.0")
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
            # Detect regime first so collect_signals can skip zero-weight sources
            if regime is None:
                regime, regime_confidence = self.detect_regime()
            readings = self.current_readings or self.collect_signals(regime=regime)

        if regime is None:
            regime, regime_confidence = self.detect_regime()
        
        if regime_confidence is None:
            regime_confidence = 0.5
        
        self.current_regime = regime
        self.current_regime_confidence = regime_confidence

        # Get weights for regime (blended with bandit if available)
        weights = self.get_blended_weights(regime.name)

        # Apply regime gating — zero out signals that are net-negative in this regime
        if hasattr(self, 'regime_gate') and self.regime_gate is not None:
            weights = self.regime_gate.filter_weights(weights, regime.name)
            # Renormalize so weights sum to 1.0
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
        
        # Apply adaptive ensemble weighting (v6.09) if available
        try:
            from src.strategy.adaptive_ensemble_weights import AdaptiveEnsembleWeights
            
            # Try to load latest attribution data
            attribution_dir = ATTRIBUTION_DIR
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
            base_allocation = BASE_ALLOCATION
        
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
