"""EnsembleVoter collect mixin (Item 5 s3 ENSEMBLE-VOTER-MIXINS).
Methods extracted verbatim from src/strategy/ensemble_voter.py.
"""

import logging
from src.data.price_cache import get_prices_df
from src.paths import sqlite_connect
from src.signals.regime_spec import REGIME_WEIGHTS
from src.signals.regime_spec import Regime
from src.signals.regime_spec import SignalReading
from src.signals.signal_source import SignalSource
from src.strategy.signal_aggregator import SignalAggregator
from src.utils.computation_cache import get_realized_volatility
from typing import Dict
from typing import Optional
from typing import Tuple
import pandas as pd
logger = logging.getLogger("src.strategy.ensemble_voter")

class CollectMixin:
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

        if len(spy) < 21:
            return Regime.NORMAL, 0.5

        # 20-day realized vol (annualized, TTL-cached)
        vol_20d = get_realized_volatility(spy, window=20)
        if vol_20d is None:
            return Regime.NORMAL, 0.5

        # Returns for momentum and drawdown calculations
        returns = spy.pct_change().dropna()
        
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
        """Load price data from JSON (TTL-cached)."""
        try:
            df = get_prices_df()
            if df.empty:
                return None
            return df
        except (FileNotFoundError, ValueError):
            return None

    def collect_signals(self, date: Optional[str] = None, regime: Optional[Regime] = None) -> Dict[SignalSource, SignalReading]:
        """
        Collect signals from active sources.

        If regime is provided, skip signal sources with zero weight for that
        regime — avoids wasted computation on signals that won't affect the vote.

        Active sources (7 signals):
        - Multi-speed momentum (primary trend signal)
        - Cross-asset relative value (mean-reversion triggers)
        - International equity momentum (EFA/VXUS trend)
        - Alternative data (SEC EDGAR, NewsAPI, jobs)
        - Cross-asset regime arbitrage (divergence detection)
        - Unified overlay (collar + bond + crypto + calendar)
        - Multi-timeframe fusion (v806 redo — timeframe decomposition)

        Collection is delegated to ``self.signal_aggregator`` so the collaborator
        can be injected or stubbed without rewriting vote logic.
        """
        aggregator = self._ensure_signal_aggregator()
        readings = aggregator.collect(date=date, regime=regime)
        self.current_readings = readings
        return readings

    def _ensure_signal_aggregator(self):
        """Return the signal aggregator, creating a default if missing.

        Fixtures that construct via ``EnsembleVoter.__new__`` never run
        ``__init__``; keep collection working for those paths and for
        intentional late injection.
        """
        aggregator = getattr(self, "signal_aggregator", None)
        if aggregator is None:
            aggregator = SignalAggregator(
                load_price_data=lambda: self._load_price_data(),
                regime_weights=REGIME_WEIGHTS,
            )
            self.signal_aggregator = aggregator
        return aggregator

    def _should_skip(self, source: SignalSource, active_sources, regime: Optional[Regime]) -> bool:
        """Check if a signal source should be skipped for the current regime."""
        return self._ensure_signal_aggregator().should_skip(source, active_sources, regime)

    def _collect_msm_signal(self, readings: Dict, active_sources, regime: Optional[Regime], date: Optional[str]) -> None:
        """Collect multi-speed momentum signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_msm_signal(readings, active_sources, regime, date)

    def _collect_cross_asset_rv_signal(
        self, readings: Dict, active_sources, regime: Optional[Regime],
    ) -> None:
        """Collect cross-asset relative value signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_cross_asset_rv_signal(readings, active_sources, regime)

    def _collect_intl_momentum_signal(
        self, readings: Dict, active_sources, regime: Optional[Regime],
    ) -> None:
        """Collect international equity momentum signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_intl_momentum_signal(readings, active_sources, regime)

    def _collect_alt_data_signal(
        self, readings: Dict, active_sources, regime: Optional[Regime],
    ) -> None:
        """Collect alternative data signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_alt_data_signal(readings, active_sources, regime)

    def _collect_regime_arb_signal(self, readings: Dict, active_sources, regime: Optional[Regime]) -> None:
        """Collect cross-asset regime arbitrage signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_regime_arb_signal(readings, active_sources, regime)

    def _collect_unified_overlay_signal(
        self, readings: Dict, active_sources, regime: Optional[Regime],
    ) -> None:
        """Collect unified overlay signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_unified_overlay_signal(readings, active_sources, regime)

    def _collect_mtf_signal(
        self, readings: Dict, active_sources, regime: Optional[Regime],
        date: Optional[str] = None,
    ) -> None:
        """Collect multi-timeframe fusion signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_mtf_signal(readings, active_sources, regime, date)

    def _collect_google_trends(
        self, readings: dict, active_sources: set, regime, date: str
    ) -> None:
        """Collect Google Trends sentiment signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_google_trends(readings, active_sources, regime)

    def _collect_vix_term_structure_signal(self, readings: dict, active_sources: set, regime) -> None:
        """Collect VIX term structure signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_vix_term_structure_signal(readings, active_sources, regime)

    @staticmethod
    def _static_zero_baseline_sources(regime_name: str) -> set:
        """SignalSource keys with intentional REGIME_WEIGHTS soft-delete (weight 0).

        Soft-delete is economic/ADR policy — bandit blend, adaptive floors,
        exploration noise, and diversity floors must not reinflate these arms
        until a human promotes non-zero static weights.
        """
        regime_enum = getattr(Regime, str(regime_name).upper(), Regime.NORMAL)
        static = REGIME_WEIGHTS.get(regime_enum, {}) or {}
        zeros: set = set()
        for src, w in static.items():
            try:
                if float(w or 0.0) <= 0.0:
                    zeros.add(src)
            except (TypeError, ValueError):
                zeros.add(src)
        return zeros

    @staticmethod
    def _pin_zero_baseline_weights(weights: Dict, regime_name: str) -> Dict:
        """Force soft-delete arms to 0 and renormalize remaining mass.

        Batch DK: bandit ε-mass + adaptive min_weight + Dirichlet exploration
        previously reinflated multi_speed_momentum (~5–13% vote mass) despite
        REGIME_WEIGHTS soft-delete. Pin after each reinflation-capable stage.
        """
        from src.strategy.ensemble_voter import EnsembleVoter
        if not weights:
            return weights
        zeros = EnsembleVoter._static_zero_baseline_sources(regime_name)
        if not zeros:
            return weights
        pinned = dict(weights)
        changed = False
        for src in zeros:
            if src in pinned and float(pinned.get(src) or 0.0) != 0.0:
                pinned[src] = 0.0
                changed = True
            elif src in pinned:
                pinned[src] = 0.0
        if not changed and all(
            float(pinned.get(s) or 0.0) == 0.0 for s in zeros if s in pinned
        ):
            # Still renorm if zeros already 0 but total drifted
            total = sum(float(v or 0.0) for v in pinned.values())
            if total > 0 and abs(total - 1.0) > 1e-9:
                return {k: float(v or 0.0) / total for k, v in pinned.items()}
            return pinned
        total = sum(float(v or 0.0) for v in pinned.values())
        if total > 0:
            pinned = {k: float(v or 0.0) / total for k, v in pinned.items()}
        return pinned
