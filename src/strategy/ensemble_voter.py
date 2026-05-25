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
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import logging

from src.paths import DATA_DIR, PRICES_JSON, ATTRIBUTION_DIR, BASE_ALLOCATION, sqlite_connect
from src.utils import safe_get


__all__ = ['Regime', 'SignalSource', 'SignalReading', 'EnsembleVote', 'REGIME_WEIGHTS', 'BanditWeighter', 'EnsembleVoter']

logger = logging.getLogger(__name__)

# Module-level health tracker singleton (lazy initialized)
_health_tracker = None

def _get_health_tracker():
    global _health_tracker
    if _health_tracker is None:
        try:
            from src.signals.health_tracker import SignalHealthTracker
            _health_tracker = SignalHealthTracker()
        except (ImportError, OSError, KeyError, ValueError, TypeError) as e:
            logger.warning("SignalHealthTracker unavailable: %s", e)
    return _health_tracker


class Regime(Enum):
    """Market regime classifications."""
    LOW_VOL = "low_vol"      # VIX < 15, calm bull market
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class SignalSource(Enum):
    """Available signal sources."""
    MULTI_SPEED_MOM = "multi_speed_momentum"  # Multi-speed momentum
    CROSS_ASSET_RV = "cross_asset_rv"         # Cross-asset relative value
    INTERNATIONAL_MOMENTUM = "international_momentum"  # International equity momentum
    ALTERNATIVE_DATA = "alternative_data"  # Alternative data signal (SEC EDGAR, NewsAPI, jobs)
    CROSS_ASSET_REGIME_ARB = "cross_asset_regime_arb"  # Cross-asset regime arbitrage
    UNIFIED_OVERLAY = "unified_overlay"       # Unified overlay (ref'd by orchestrator_ensemble_bridge)


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
# MSM disabled (net-negative -0.012 Sharpe), weight redistributed to ALT_DATA and INTL_MOM.
# Weights sum=1.0 per regime.
REGIME_WEIGHTS = {
    Regime.LOW_VOL: {
        SignalSource.MULTI_SPEED_MOM: 0.0000,
        SignalSource.CROSS_ASSET_RV: 0.1500,
        SignalSource.ALTERNATIVE_DATA: 0.3500,
        SignalSource.INTERNATIONAL_MOMENTUM: 0.2800,
        SignalSource.CROSS_ASSET_REGIME_ARB: 0.0000,  # marginal in calm markets
        SignalSource.UNIFIED_OVERLAY: 0.2200,
    },
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
    """Thompson Sampling contextual bandit for dynamic signal weight adaptation.

    Tracks per-signal reward distribution using Gaussian-Gamma conjugate priors.
    Thompson Sampling samples from posterior to balance exploration/exploitation,
    converging 2-3x faster than epsilon-greedy in cold-start (<21 observations).

    Falls back to epsilon-greedy when posterior is uninformative (<2 observations).
    Softmax converts sampled Sharpe estimates to weights.

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
        # Thompson Sampling priors: Gaussian-Gamma conjugate
        # mu_0, lambda_0 (prior precision scaling), alpha_0, beta_0
        self._mu_0 = 0.0
        self._lambda_0 = 1.0
        self._alpha_0 = 2.0  # shape — weak prior
        self._beta_0 = 1.0   # rate — weak prior

    def select(self, regime: str) -> str:
        """Select a signal using Thompson Sampling with epsilon-greedy fallback."""
        # Epsilon-greedy exploration (small probability of random)
        if random.random() < self.epsilon:
            return random.choice(self.signals)

        # Thompson Sampling: sample Sharpe from posterior for each signal
        sampled_sharpes = {}
        has_sufficient_data = False
        for sig in self.signals:
            n = len(safe_get(self._history, regime, sig, default=[]))
            if n >= 2:
                has_sufficient_data = True
                sampled_sharpes[sig] = self._sample_sharpe(sig, regime)
            else:
                sampled_sharpes[sig] = random.gauss(0.0, 1.0)  # uninformative prior

        # If no signal has sufficient data, fall back to rolling Sharpe
        if not has_sufficient_data:
            best_signal = self.signals[0]
            best_sharpe = -float("inf")
            for sig in self.signals:
                sh = self._rolling_sharpe(sig, regime)
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_signal = sig
            return best_signal

        return max(sampled_sharpes, key=sampled_sharpes.get)

    def _sample_sharpe(self, signal: str, regime: str) -> float:
        """Sample a Sharpe ratio from the Gaussian-Gamma posterior."""
        hist = safe_get(self._history, regime, signal, default=[])
        n = len(hist)
        if n < 2:
            return 0.0

        arr = np.array(hist[-self.window:])
        x_bar = np.mean(arr)

        # Posterior parameters (Gaussian-Gamma conjugate update)
        lambda_n = self._lambda_0 + n
        mu_n = (self._lambda_0 * self._mu_0 + n * x_bar) / lambda_n
        alpha_n = self._alpha_0 + n / 2.0
        beta_n = (self._beta_0
                  + 0.5 * np.sum((arr - x_bar) ** 2)
                  + (self._lambda_0 * n * (x_bar - self._mu_0) ** 2)
                  / (2.0 * lambda_n))

        # Sample precision tau ~ Gamma(alpha_n, beta_n)
        # numpy gamma uses shape/scale, so scale = 1/beta_n
        if beta_n > 1e-10 and alpha_n > 0:
            tau = np.random.gamma(alpha_n, 1.0 / beta_n)
        else:
            tau = 1.0  # fallback

        # Sample mean mu ~ Normal(mu_n, 1/(lambda_n * tau))
        if tau > 1e-10:
            sigma_mu = 1.0 / np.sqrt(lambda_n * tau)
            mu_sample = np.random.normal(mu_n, sigma_mu)
        else:
            mu_sample = mu_n

        # Convert sampled mean to annualized Sharpe
        # Sharpe = mu / sigma * sqrt(252), and sigma = 1/sqrt(tau)
        if tau > 1e-10:
            sigma = 1.0 / np.sqrt(tau)
            return float(mu_sample / sigma * np.sqrt(252))
        return 0.0

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
        hist = safe_get(self._history, regime, signal, default=[])
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
            n = len(self.signals)
            w = 1.0 / n if n > 0 else 0.0
            return {s: w for s in self.signals}
        return {sig: float(exp_values[i] / total)
                for i, sig in enumerate(self.signals)}


class EnsembleVoter:
    """
    Multi-source signal ensemble with regime-adaptive weighting.

    Collects signals from all strategy modules, applies regime-dependent
    weighting, and produces consensus recommendations.
    """

    # Regime detection thresholds
    CRISIS_VOL_THRESHOLD = 0.30        # 20d annualized vol above this → CRISIS
    CRISIS_DRAWDOWN_THRESHOLD = -0.10  # Drawdown below this → CRISIS
    HIGH_VOL_VOL_THRESHOLD = 0.20      # 20d annualized vol above this → HIGH_VOL
    HIGH_VOL_DRAWDOWN_THRESHOLD = -0.05
    HIGH_VOL_MOM_THRESHOLD = 0.0       # Negative momentum with drawdown → HIGH_VOL
    LOW_VOL_VOL_THRESHOLD = 0.12       # 20d annualized vol below this → LOW_VOL
    LOW_VOL_MOM_THRESHOLD = 0.01       # Positive momentum required for LOW_VOL
    RECOVERY_DRAWDOWN_THRESHOLD = -0.03
    RECOVERY_MOM_THRESHOLD = 0.02
    
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
        with sqlite_connect(self.db_path) as conn:
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
        if vol_20d > self.CRISIS_VOL_THRESHOLD or drawdown < self.CRISIS_DRAWDOWN_THRESHOLD:
            regime = Regime.CRISIS
            confidence = min(abs(drawdown) * 5, 0.9) if drawdown < self.HIGH_VOL_DRAWDOWN_THRESHOLD else 0.5
        elif vol_20d > self.HIGH_VOL_VOL_THRESHOLD or (drawdown < self.HIGH_VOL_DRAWDOWN_THRESHOLD and mom_20d < self.HIGH_VOL_MOM_THRESHOLD):
            regime = Regime.HIGH_VOL
            confidence = min(vol_20d * 3, 0.8)
        elif drawdown < self.RECOVERY_DRAWDOWN_THRESHOLD and mom_20d > self.RECOVERY_MOM_THRESHOLD:
            regime = Regime.RECOVERY
            confidence = min(mom_20d * 20, 0.7)
        elif vol_20d < self.LOW_VOL_VOL_THRESHOLD and mom_20d > self.LOW_VOL_MOM_THRESHOLD:
            regime = Regime.LOW_VOL
            confidence = max(0.5, 1.0 - vol_20d * 4)
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

        # Collect from each signal source
        self._collect_msm_signal(readings, active_sources, regime, date)
        self._collect_cross_asset_rv_signal(readings)
        self._collect_intl_momentum_signal(readings, active_sources, regime)
        self._collect_alt_data_signal(readings, active_sources, regime)
        self._collect_regime_arb_signal(readings, active_sources, regime)
        self._collect_unified_overlay_signal(readings, active_sources, regime)

        self.current_readings = readings
        return readings

    def _should_skip(self, source: SignalSource, active_sources, regime: Optional[Regime]) -> bool:
        """Check if a signal source should be skipped for the current regime."""
        if active_sources is not None and source not in active_sources:
            logger.debug("Skipping %s: zero weight for regime=%s", source.value, regime.value if regime else '?')
            return True
        return False

    def _collect_msm_signal(self, readings: Dict, active_sources, regime: Optional[Regime], date: Optional[str]) -> None:
        """Collect multi-speed momentum signal."""
        if self._should_skip(SignalSource.MULTI_SPEED_MOM, active_sources, regime):
            return
        try:
            from src.signals.multi_speed_momentum import MultiSpeedMomentum
            msm = MultiSpeedMomentum()
            snapshot = msm.get_signal_snapshot(tickers=['SPY', 'TLT', 'GLD'], date=date)
            if snapshot.is_active:
                readings[SignalSource.MULTI_SPEED_MOM] = snapshot.to_signal_reading()
        except ImportError:
            pass
        except (AttributeError, KeyError, ValueError, TypeError, OSError, RuntimeError) as e:
            logger.warning("Multi-speed momentum unavailable: %s", e)

    def _collect_cross_asset_rv_signal(self, readings: Dict) -> None:
        """Collect cross-asset relative value signal."""
        try:
            from src.signals.cross_asset_relative_value import CrossAssetRVScanner
            rv_scanner = CrossAssetRVScanner()
            snapshot = rv_scanner.get_signal_snapshot()
            if snapshot.is_active:
                readings[SignalSource.CROSS_ASSET_RV] = snapshot.to_signal_reading()
        except ImportError:
            pass
        except (AttributeError, KeyError, ValueError, TypeError, OSError, RuntimeError) as e:
            logger.warning("Cross-asset RV unavailable: %s", e)

    def _collect_intl_momentum_signal(
        self, readings: Dict, active_sources, regime: Optional[Regime],
    ) -> None:
        """Collect international equity momentum signal."""
        if self._should_skip(SignalSource.INTERNATIONAL_MOMENTUM, active_sources, regime):
            return
        try:
            from src.signals.international_momentum import InternationalMomentumGenerator

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
        except (AttributeError, KeyError, ValueError, TypeError, OSError, RuntimeError) as e:
            logger.warning("International momentum unavailable: %s", e)

    def _collect_alt_data_signal(
        self, readings: Dict, active_sources, regime: Optional[Regime],
    ) -> None:
        """Collect alternative data signal."""
        if self._should_skip(SignalSource.ALTERNATIVE_DATA, active_sources, regime):
            return
        try:
            from src.signals.alternative_data_signal import AlternativeDataSignalGenerator
            alt_gen = AlternativeDataSignalGenerator()
            snapshot = alt_gen.get_signal_snapshot()
            if snapshot.is_active:
                readings[SignalSource.ALTERNATIVE_DATA] = snapshot.to_signal_reading()
        except ImportError:
            pass
        except (AttributeError, KeyError, ValueError, TypeError, OSError, RuntimeError) as e:
            logger.warning("Alternative data unavailable: %s", e)

    def _collect_regime_arb_signal(self, readings: Dict, active_sources, regime: Optional[Regime]) -> None:
        """Collect cross-asset regime arbitrage signal."""
        if self._should_skip(SignalSource.CROSS_ASSET_REGIME_ARB, active_sources, regime):
            return
        try:
            from src.signals.cross_asset_regime_arb import CrossAssetRegimeArbDetector
            arb_detector = CrossAssetRegimeArbDetector()
            snapshot = arb_detector.get_signal_snapshot()
            if snapshot.is_active:
                readings[SignalSource.CROSS_ASSET_REGIME_ARB] = snapshot.to_signal_reading()
        except ImportError:
            logger.warning("Cross-asset regime arb module not available")
        except (AttributeError, KeyError, ValueError, TypeError, OSError, RuntimeError) as e:
            logger.warning("Cross-asset regime arb unavailable: %s", e)

    def _collect_unified_overlay_signal(
        self, readings: Dict, active_sources, regime: Optional[Regime],
    ) -> None:
        """Collect unified overlay signal (collar + bond_duration + crypto + calendar)."""
        if self._should_skip(SignalSource.UNIFIED_OVERLAY, active_sources, regime):
            return
        try:
            from .orchestrator_ensemble_bridge import OrchestratorEnsembleBridge
            bridge = OrchestratorEnsembleBridge()
            unified_reading = bridge.get_ensemble_reading()
            readings[SignalSource.UNIFIED_OVERLAY] = unified_reading
        except ImportError:
            pass
        except (AttributeError, KeyError, ValueError, TypeError, OSError, RuntimeError) as e:
            logger.warning("Unified overlay unavailable: %s", e)

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

    def get_rebalance_config(self) -> Dict[str, Any]:
        """
        Return current regime and rebalancing parameters for the
        SmartRebalanceGate/Controller to use regime-adaptive thresholds.

        Returns:
            Dict with 'regime' key (e.g. 'normal', 'crisis', 'high_vol',
            'low_vol', 'recovery') for the rebalancing controller.
        """
        regime_map = {
            Regime.LOW_VOL: 'low_vol',
            Regime.NORMAL: 'normal',
            Regime.HIGH_VOL: 'high_vol',
            Regime.CRISIS: 'crisis',
            Regime.RECOVERY: 'recovery',
        }
        return {
            'regime': regime_map.get(self.current_regime, 'normal'),
            'regime_confidence': self.current_regime_confidence,
        }

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
        except (ImportError, OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            logger.warning("Failed to load goals for risk budget, using risk_mult=1.0: %s", e)
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
        """Compute ensemble vote with regime-dependent weighting.

        Delegates to sub-methods for each weighting phase:
        1. _resolve_inputs — resolve readings/regime/confidence defaults
        2. _apply_regime_gating — zero out signals net-negative in this regime
        3. _apply_adaptive_weights — attribution-based weight adjustment
        4. _apply_health_weights — reduce weight for poor health scores
        5. _apply_turnover_validation — turnover + basis-pursuit + regret-weighted
        6. _compute_consensus — weighted consensus, agreement, asset biases, action
        7. _persist_vote — save vote and persist regret state
        """
        readings, regime, regime_confidence = self._resolve_inputs(
            readings, regime, regime_confidence
        )

        weights = self.get_blended_weights(regime.name)
        weights = self._apply_regime_gating(weights, regime.name)
        weights = self._apply_adaptive_weights(weights, regime)
        weights = self._apply_health_weights(weights)
        weights = self._apply_turnover_validation(weights, readings, regime)

        # Apply weights to readings
        weighted_signals = self._apply_weights_to_readings(readings, weights)

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

        consensus_result = self._compute_consensus(weighted_signals, regime, regime_confidence)
        vote = self._build_vote(weighted_signals, consensus_result, regime, regime_confidence)
        self._persist_vote(vote, consensus_result.weighted_consensus)

        return vote

    def _resolve_inputs(
        self,
        readings: Optional[Dict[SignalSource, SignalReading]],
        regime: Optional[Regime],
        regime_confidence: Optional[float],
    ) -> Tuple[Dict[SignalSource, SignalReading], Regime, float]:
        """Resolve default readings, regime, and confidence."""
        if readings is None:
            if regime is None:
                regime, regime_confidence = self.detect_regime()
            readings = self.current_readings or self.collect_signals(regime=regime)

        if regime is None:
            regime, regime_confidence = self.detect_regime()

        if regime_confidence is None:
            regime_confidence = 0.5

        self.current_regime = regime
        self.current_regime_confidence = regime_confidence
        return readings, regime, regime_confidence

    def _apply_regime_gating(
        self, weights: Dict, regime_name: str
    ) -> Dict:
        """Apply regime gating — zero out signals that are net-negative in this regime."""
        if hasattr(self, 'regime_gate') and self.regime_gate is not None:
            weights = self.regime_gate.filter_weights(weights, regime_name)
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
        return weights

    def _apply_adaptive_weights(
        self, weights: Dict, regime: Regime
    ) -> Dict:
        """Apply adaptive ensemble weighting (v6.09) if attribution data is fresh enough."""
        try:
            from src.strategy.adaptive_ensemble_weights import AdaptiveEnsembleWeights

            attribution_dir = ATTRIBUTION_DIR
            attribution_files = sorted(attribution_dir.glob("attribution_*.json"), reverse=True)

            if not attribution_files:
                return weights

            with open(attribution_files[0]) as f:
                attribution_data = json.load(f)

            # Check if attribution is stale (>7 days old)
            attr_timestamp = attribution_data.get("timestamp", "")
            if attr_timestamp:
                attr_date = attr_timestamp[:10]
                days_stale = (datetime.now() - datetime.strptime(attr_date, "%Y-%m-%d")).days
            else:
                days_stale = 999

            if days_stale > 7:
                return weights

            # Check if we have enough data points
            sources = attribution_data.get("sources", {})
            total_readings = sum(s.get("total_readings", 0) for s in sources.values())
            num_sources = len(sources)
            avg_readings = total_readings / max(num_sources, 1)

            if avg_readings < 5:
                return weights

            # Build base weights in string-keyed format
            base_str = {k.value: v for k, v in weights.items()}

            adaptive = AdaptiveEnsembleWeights(base_weights=base_str)
            adapted = adaptive.update_weights(attribution_data, regime.value)

            # Convert back to enum-keyed dict
            adaptive_weights_enum = {}
            for source_enum in weights:
                source_str = source_enum.value
                if source_str in adapted:
                    adaptive_weights_enum[source_enum] = adapted[source_str]

            if adaptive_weights_enum:
                logger.info("Using adaptive ensemble weights for regime=%s", regime.value)
                return adaptive_weights_enum
        except (KeyError, ValueError, TypeError, AttributeError, ZeroDivisionError, OSError) as e:
            logger.warning("Could not apply adaptive ensemble weights: %s", e)
        return weights

    def _apply_health_weights(self, weights: Dict) -> Dict:
        """Apply health-adjusted weighting (v3.12) — reduce weight for poor health scores."""
        try:
            from src.signals.health_tracker import SignalHealthTracker
            health_tracker = SignalHealthTracker()
            health_scores = health_tracker.calculate_all_health_scores()

            if not health_scores:
                return weights

            adjusted_weights = {}
            for source_enum, base_weight in weights.items():
                source_str = source_enum.value
                if source_str in health_scores:
                    health = health_scores[source_str]
                    multiplier = max(0.2, min(1.0, health.health_score))
                    adjusted_weights[source_enum] = base_weight * multiplier
                    if health.health_score < 0.5:
                        logger.info("Health-adjusted %s: weight %.2f%% → %.2f%% (health=%.2f)", source_str, base_weight * 100, adjusted_weights[source_enum] * 100, health.health_score)
                else:
                    adjusted_weights[source_enum] = base_weight

            total = sum(adjusted_weights.values())
            if total > 0:
                weights = {k: v / total for k, v in adjusted_weights.items()}
        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.warning("Could not apply health-adjusted weights: %s", e)
        return weights

    def _apply_turnover_validation(
        self, weights: Dict, readings: Dict, regime: Regime
    ) -> Dict:
        """Apply turnover-aware weight validation (v8.01) with basis-pursuit and regret-weighted."""
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

            if not signal_values:
                return weights

            # Build base weights dict from regime weights (string-keyed)
            base_weights_str = {source_enum.value: w for source_enum, w in weights.items()}

            # --- v8.02: Basis-Pursuit Signal Selection ---
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
                logger.debug("Basis-pursuit selection applied%s", sparsity_msg)
            except (ImportError, KeyError, ValueError, TypeError, AttributeError, ZeroDivisionError) as bp_e:
                logger.warning("Could not apply basis-pursuit selection: %s", bp_e)

            # --- v8.03: Regret-Weighted Adjustment ---
            try:
                from src.strategy.regret_weighted_selector import RegretWeightedSelector
                rw_selector = RegretWeightedSelector()
                prev_decision = getattr(rw_selector.state, 'last_ensemble_decision', 0.0)
                rw_result = rw_selector.adjust_weights(
                    signal_values, prev_decision, base_weights_str, regime=regime.value
                )
                base_weights_str = rw_result.adjusted_weights
                if rw_result.signals_with_high_regret:
                    logger.info(
                        "Regret-adjusted weights: penalized %s (avg_regret=%.3f)",
                        ', '.join(rw_result.signals_with_high_regret),
                        rw_result.avg_regret
                    )
            except (ImportError, KeyError, ValueError, TypeError, AttributeError, OSError) as rw_e:
                logger.warning("Could not apply regret-weighted adjustment: %s", rw_e)

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
                "Turnover-adjusted %d signals: %s",
                len(signal_values),
                ', '.join(f'{s}={turnover_adjusted.get(enum, 0):.4f}' for enum, s in [(e, e.value) for e in weights])
            )
        except (KeyError, ValueError, TypeError, AttributeError, ZeroDivisionError, OSError) as e:
            logger.warning("Could not apply turnover-aware weights: %s", e)
        return weights

    def _apply_weights_to_readings(
        self,
        readings: Dict[SignalSource, SignalReading],
        weights: Dict,
    ) -> List[SignalReading]:
        """Assign weights to readings and log predictions for health tracking."""
        weighted_signals = []
        for source, reading in readings.items():
            if source in weights:
                reading.weight = weights[source]
                weighted_signals.append(reading)

        # Log signal predictions for health tracking (v3.12)
        try:
            tracker = _get_health_tracker()
            if tracker is not None:
                for reading in weighted_signals:
                    tracker.log_prediction_simple(
                        source=reading.source.value,
                        signal_value=reading.value,
                        confidence=reading.confidence,
                    )
        except (KeyError, ValueError, TypeError, AttributeError, OSError, sqlite3.Error) as e:
            logger.warning("Health tracking log failed: %s", e)

        return weighted_signals

    @dataclass
    class _ConsensusResult:
        """Internal intermediate result from consensus computation."""
        weighted_consensus: float
        agreement: float
        equity_bias: float
        duration_bias: float
        gold_bias: float
        action: str
        action_confidence: float

    def _compute_consensus(
        self,
        weighted_signals: List[SignalReading],
        regime: Regime,
        regime_confidence: float,
    ) -> '_ConsensusResult':
        """Compute weighted consensus, agreement ratio, and asset biases."""
        # Weighted consensus — handle NaN values
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

        return self._ConsensusResult(
            weighted_consensus=weighted_consensus,
            agreement=agreement,
            equity_bias=equity_bias,
            duration_bias=duration_bias,
            gold_bias=gold_bias,
            action=action,
            action_confidence=action_confidence,
        )

    def _build_vote(
        self,
        weighted_signals: List[SignalReading],
        consensus: '_ConsensusResult',
        regime: Regime,
        regime_confidence: float,
    ) -> EnsembleVote:
        """Build EnsembleVote from weighted signals and consensus result."""
        reasons = [
            f"Regime: {regime.value} (confidence: {regime_confidence:.1%})",
            f"Sources: {len(weighted_signals)}, Consensus: {consensus.weighted_consensus:+.3f}",
            f"Agreement: {consensus.agreement:.1%}",
            f"Equity bias: {consensus.equity_bias:+.3f}, Duration: {consensus.duration_bias:+.3f}, Gold: {consensus.gold_bias:+.3f}"
        ]

        for r in weighted_signals[:3]:
            reasons.append(f"  {r.source.value}: {r.value:+.3f} (w={r.weight:.2f}, conf={r.confidence:.1%})")

        return EnsembleVote(
            timestamp=str(datetime.now()),
            regime=regime,
            regime_confidence=regime_confidence,
            num_sources=len(weighted_signals),
            weighted_consensus=consensus.weighted_consensus,
            agreement_ratio=consensus.agreement,
            equity_bias=consensus.equity_bias,
            duration_bias=consensus.duration_bias,
            gold_bias=consensus.gold_bias,
            action=consensus.action,
            confidence=consensus.action_confidence,
            reasoning="\n".join(reasons),
            source_votes=weighted_signals
        )

    def _persist_vote(self, vote: EnsembleVote, weighted_consensus: float) -> None:
        """Persist ensemble decision for regret-weighted cycle and save vote to DB."""
        # Persist ensemble decision for next regret-weighted cycle (v8.03)
        try:
            from src.strategy.regret_weighted_selector import RegretWeightedSelector
            rw_selector = RegretWeightedSelector()
            rw_selector.state.last_ensemble_decision = weighted_consensus
            rw_selector._save_state()
        except (ImportError, OSError, KeyError, ValueError, TypeError, AttributeError) as rw_e:
            logger.warning("Could not persist ensemble decision to regret-weighted state: %s", rw_e)

        # Check for IC-based signal decay alerts
        try:
            _tracker = _get_health_tracker()
            if _tracker is not None:
                alerts = _tracker.detect_ic_alerts()
                if alerts:
                    alert_names = [a.source for a in alerts]
                    logger.warning("IC decay alerts detected: %s", alert_names)
        except (KeyError, ValueError, TypeError, AttributeError, OSError, sqlite3.Error) as ic_e:
            logger.warning("IC alert check failed: %s", ic_e)

        # Save to DB
        self._save_vote(vote)
    
    def _save_vote(self, vote: EnsembleVote):
        """Save vote to database, including per-source readings (v5.70)."""
        with sqlite_connect(self.db_path) as conn:
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
                    logger.warning("Failed to save source reading %s: %s", reading.source, e)
    
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

    def get_bl_views(
        self,
        vote: Optional[EnsembleVote] = None,
        tau: float = 0.15,
        prior: str = "equal",
    ) -> Dict[str, Any]:
        """Generate Black-Litterman views from ensemble vote.

        Maps equity_bias, duration_bias, and gold_bias from the
        current ensemble consensus to BL absolute views, with view
        confidence derived from signal health scores.

        Args:
            vote: Pre-computed vote (default: compute fresh).
            tau: BL tau parameter (view weight). Default 0.15.
            prior: Prior type — "equal" or "market".

        Returns:
            Dict with 'views' (BLViews), 'tau', 'prior', and
            'health_scores_used' keys.
        """
        from src.strategy.black_litterman_mapper import map_biases_to_views, BLViews

        if vote is None:
            vote = self.compute_vote()

        # Collect health scores from tracker
        health_scores = {}
        tracker = _get_health_tracker()
        if tracker is not None:
            try:
                report = tracker.get_health_report()
                for source_name, data in report.get('sources', {}).items():
                    score = data.get('health_score', 0.5)
                    health_scores[source_name] = score
            except (KeyError, ValueError, TypeError, AttributeError, OSError, sqlite3.Error) as e:
                logger.warning("Could not get health scores for BL views: %s", e)

        views = map_biases_to_views(
            equity_bias=vote.equity_bias,
            duration_bias=vote.duration_bias,
            gold_bias=vote.gold_bias,
            health_scores=health_scores if health_scores else None,
            tau=tau,
            prior=prior,
        )

        return {
            'views': views,
            'tau': tau,
            'prior': prior,
            'health_scores_used': health_scores,
            'equity_bias': vote.equity_bias,
            'duration_bias': vote.duration_bias,
            'gold_bias': vote.gold_bias,
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
