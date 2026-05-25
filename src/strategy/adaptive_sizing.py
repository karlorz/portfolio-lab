#!/usr/bin/env python3
"""
v5.74: Adaptive Position Sizing Overlay

Adjusts base 46/38/16 (SPY/GLD/TLT) allocation based on:
- Regime (from ML-Light Regime Classifier)
- Volatility regime (internal volatility classification)
- Signal conviction (from ensemble voter)
- Drawdown state (from circuit breaker)

Pure numpy/stdlib — no ML dependencies.

Design:
1. Read current regime, vol, ensemble state, drawdown
2. Compute adjustment factors for each asset (-0.10 to +0.10 per factor)
3. Aggregate with configurable weights
4. Apply hard bounds (SPY 36-56%, GLD 28-48%, bonds 6-26%)
5. Output scaled allocation

Usage:
    python -m src.strategy.adaptive_sizing adjust    # Get adjusted allocation
    python -m src.strategy.adaptive_sizing status    # Current sizing state
    python -m src.strategy.adaptive_sizing simulate  # Backtest vs static
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.paths import BASE_ALLOCATION, DATA_DIR, PRICES_JSON, sqlite_connect
from src.data.price_cache import get_prices
from src.backtest.metrics import save_results_json

import numpy as np


__all__ = ['HARD_BOUNDS', 'MAX_FACTOR_ADJUSTMENT', 'REGIME_ADJUSTMENTS', 'CONFIDENCE_SCALING', 'SizingFactors', 'SizingDecision', 'AdaptiveSizer']

logger = logging.getLogger(__name__)

STATE_PATH = DATA_DIR / "adaptive_sizing_state.json"

# Hard bounds
HARD_BOUNDS = {
    "SPY": (0.36, 0.56),
    "GLD": (0.28, 0.48),
    "TLT": (0.06, 0.26),
    "IEF": (0.00, 0.10),
    "SHY": (0.00, 0.10),
}

# Maximum adjustment from a single factor (absolute)
MAX_FACTOR_ADJUSTMENT = 0.08

# ── Regime-Based Adjustments ────────────────────────────────────────────────

# How each regime shifts allocation (SPY, GLD, TLT adjustments)
REGIME_ADJUSTMENTS = {
    "low_vol":    {"SPY": +0.04, "GLD": -0.02, "TLT": -0.02},
    "normal":     {"SPY": +0.00, "GLD": +0.00, "TLT": +0.00},
    "high_vol":   {"SPY": -0.03, "GLD": +0.02, "TLT": +0.01},
    "crisis":     {"SPY": -0.08, "GLD": +0.05, "TLT": +0.03},
    "recovery":   {"SPY": +0.02, "GLD": -0.01, "TLT": -0.01},
    "unknown":    {"SPY": +0.00, "GLD": +0.00, "TLT": +0.00},
}

# Confidence scaling: low confidence -> reduce adjustment
CONFIDENCE_SCALING = {0.5: 0.5, 0.7: 0.8, 0.9: 1.0}


@dataclass
class SizingFactors:
    """Factor readings that influence position sizing."""
    timestamp: str
    regime: str
    regime_confidence: float
    spy_vol_20d: float
    spy_mom_20d: float
    spy_drawdown_60d: float
    ensemble_signal: float
    ensemble_agreement: float
    circuit_breaker_severity: str
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SizingDecision:
    """Complete sizing decision output."""
    timestamp: str
    base_allocation: Dict[str, float]
    adjusted_allocation: Dict[str, float]
    adjustments: Dict[str, float]  # Per-asset net adjustment
    regime_adjustment: Dict[str, float]
    volatility_adjustment: Dict[str, float]
    signal_adjustment: Dict[str, float]
    drawdown_adjustment: Dict[str, float]
    factors: SizingFactors

    def to_dict(self) -> dict:
        d = asdict(self)
        d["base_allocation"] = self.base_allocation
        d["adjusted_allocation"] = self.adjusted_allocation
        return d


class AdaptiveSizer:
    """
    Adjusts base allocation weights using regime, volatility, signal, and drawdown factors.
    No ML dependencies — pure numpy/stdlib.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.state_path = self.data_dir / "adaptive_sizing_state.json"
        self.prices: Optional[Dict] = None
        
        # State
        self.last_allocation: Dict[str, float] = dict(BASE_ALLOCATION)
        self.last_decision: Optional[SizingDecision] = None
        self._load_state()

    # ── Factor Loading ───────────────────────────────────────────────────────

    def _load_prices(self) -> Optional[Dict]:
        """Load price data from JSON (TTL-cached)."""
        try:
            self.prices = get_prices()
            return self.prices
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to load prices: %s", e)
            return None

    def _get_series(self, symbol: str) -> Optional[np.ndarray]:
        """Get price series as numpy array."""
        if self.prices is None:
            self._load_prices()
        if self.prices is None or symbol not in self.prices:
            return None
        return np.array([p["p"] for p in self.prices[symbol]])

    def _load_regime_state(self) -> Tuple[str, float]:
        """Load current regime — uses state file when available, VIX-based detection as fallback.

        If a regime_classifier_state.json exists (from test fixtures or legacy writes),
        use it directly. Otherwise, fall back to VIX-based live detection from
        the market database. Returns unknown with low confidence if neither works.
        """
        # Primary: state file (supports test fixtures and legacy writes)
        regime_state_path = self.data_dir / "regime_classifier_state.json"
        try:
            if regime_state_path.exists():
                state = json.loads(regime_state_path.read_text())
                regime = state.get("current_regime", "unknown")
                last = state.get("last_reading", {})
                conf = last.get("confidence", 0.5)

                # Validate that the regime is valid
                if regime not in REGIME_ADJUSTMENTS:
                    return "unknown", 0.3
                return regime, float(conf)
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to load regime state: %s", e)

        # Fallback: VIX-based live detection (no stale state file available)
        try:
            from src.paths import MARKET_DB
            import sqlite3
            if MARKET_DB.exists():
                from src.strategy.evaluator import get_current_regime
                with sqlite_connect(str(MARKET_DB)) as conn:
                    regime = get_current_regime(conn)
                if regime in REGIME_ADJUSTMENTS:
                    return regime, 0.8
        except (OSError, sqlite3.Error, KeyError, ValueError, TypeError, ImportError) as e:
            logger.warning("VIX-based regime detection unavailable: %s", e)

        return "unknown", 0.3

    def _load_circuit_breaker(self) -> str:
        """Load circuit breaker severity."""
        cb_path = self.data_dir / ".circuit_breaker_state.json"
        try:
            if cb_path.exists():
                state = json.loads(cb_path.read_text())
                return state.get("severity", "ok")
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to load circuit breaker state: %s", e)
        return "ok"

    def _load_ensemble_signal(self) -> Tuple[float, float]:
        """Load latest ensemble signal value and agreement."""
        ev_path = self.data_dir / "ensemble_weights.json"
        try:
            if ev_path.exists():
                state = json.loads(ev_path.read_text())
                signal = float(state.get("composite_signal", state.get("weighted_consensus", 0.0)))
                agreement = float(state.get("agreement_ratio", 0.5))
                return signal, agreement
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.error("Failed to load ensemble signal from %s: %s", ev_path, e)
        return 0.0, 0.5

    def _compute_vol_adjustment(self, vol_20d: float) -> Dict[str, float]:
        """
        Volatility-based adjustment.
        High vol: reduce SPY, increase GLD/TLT
        Low vol: increase SPY, reduce GLD/TLT
        """
        # Baseline vol threshold ~14% annualized
        target_vol = 0.14
        if vol_20d <= 0.001:
            return {"SPY": 0.0, "GLD": 0.0, "TLT": 0.0}
        
        vol_ratio = target_vol / vol_20d
        # Clamp to [0.5, 1.5] range
        vol_ratio = max(0.5, min(1.5, vol_ratio))
        
        # SPY adjustment: inverse to vol
        spy_adj = 0.0
        gld_adj = 0.0
        tlt_adj = 0.0
        
        if vol_ratio < 0.8:
            # Vol is HIGH — reduce SPY, add GLD/TLT
            factor = (0.8 - vol_ratio) / 0.3  # 0 to 1 scaling
            spy_adj = -MAX_FACTOR_ADJUSTMENT * factor * 0.7
            gld_adj = +MAX_FACTOR_ADJUSTMENT * factor * 0.5
            tlt_adj = +MAX_FACTOR_ADJUSTMENT * factor * 0.3
        elif vol_ratio > 1.2:
            # Vol is LOW — increase SPY
            factor = (vol_ratio - 1.2) / 0.3
            spy_adj = +MAX_FACTOR_ADJUSTMENT * factor * 0.7
            gld_adj = -MAX_FACTOR_ADJUSTMENT * factor * 0.3
            tlt_adj = -MAX_FACTOR_ADJUSTMENT * factor * 0.3
        
        return {"SPY": spy_adj, "GLD": gld_adj, "TLT": tlt_adj}

    def _compute_signal_adjustment(self, ensemble_signal: float, agreement: float) -> Dict[str, float]:
        """
        Ensemble signal-based adjustment.
        Strong consensus + high agreement: follow signal direction.
        Weak agreement: minimal adjustment.
        """
        # Only adjust when agreement > 0.5 (majority)
        if agreement <= 0.5 or abs(ensemble_signal) < 0.1:
            return {"SPY": 0.0, "GLD": 0.0, "TLT": 0.0}
        
        # Scale: -1 to +1 -> 0 to 1
        factor = min(1.0, abs(ensemble_signal)) * (agreement - 0.5) * 2
        
        if ensemble_signal > 0:
            # Bullish — increase SPY
            return {"SPY": +MAX_FACTOR_ADJUSTMENT * factor * 0.5,
                    "GLD": -MAX_FACTOR_ADJUSTMENT * factor * 0.3,
                    "TLT": -MAX_FACTOR_ADJUSTMENT * factor * 0.3}
        else:
            # Bearish — decrease SPY, increase GLD/TLT
            return {"SPY": -MAX_FACTOR_ADJUSTMENT * factor * 0.5,
                    "GLD": +MAX_FACTOR_ADJUSTMENT * factor * 0.3,
                    "TLT": +MAX_FACTOR_ADJUSTMENT * factor * 0.3}

    def _compute_drawdown_adjustment(self, drawdown_60d: float, severity: str) -> Dict[str, float]:
        """
        Drawdown-based adjustment.
        Increasing drawdown: reduce risk.
        """
        adj = {"SPY": 0.0, "GLD": 0.0, "TLT": 0.0}
        
        # Circuit breaker override
        if severity in ("critical", "severe"):
            adj["SPY"] = -MAX_FACTOR_ADJUSTMENT * 0.8
            adj["GLD"] = +MAX_FACTOR_ADJUSTMENT * 0.5
            adj["TLT"] = +MAX_FACTOR_ADJUSTMENT * 0.4
            return adj
        
        if severity == "elevated":
            adj["SPY"] = -MAX_FACTOR_ADJUSTMENT * 0.4
            adj["GLD"] = +MAX_FACTOR_ADJUSTMENT * 0.3
            adj["TLT"] = +MAX_FACTOR_ADJUSTMENT * 0.2
            return adj
        
        # Gradual drawdown adjustment
        if drawdown_60d < -0.10:
            # >10% drawdown
            factor = min(1.0, (abs(drawdown_60d) - 0.10) / 0.10)
            adj["SPY"] = -MAX_FACTOR_ADJUSTMENT * factor * 0.5
            adj["GLD"] = +MAX_FACTOR_ADJUSTMENT * factor * 0.3
            adj["TLT"] = +MAX_FACTOR_ADJUSTMENT * factor * 0.3
        
        return adj

    def _apply_bounds(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Apply hard bounds to allocation."""
        result = {}
        for asset, weight in allocation.items():
            lo, hi = HARD_BOUNDS.get(asset, (0.0, 1.0))
            result[asset] = max(lo, min(hi, weight))
        
        # Normalize to ensure weights sum to 1.0
        total = sum(result.values())
        if total > 0:
            for asset in result:
                result[asset] /= total
        
        return result

    # ── Main Sizing Logic ────────────────────────────────────────────────────

    def compute_allocation(self) -> SizingDecision:
        """
        Compute adjusted allocation from current factors.
        """
        # Load factors
        regime, regime_conf = self._load_regime_state()
        cb_severity = self._load_circuit_breaker()
        ensemble_signal, ensemble_agreement = self._load_ensemble_signal()
        
        # Compute market factors from price data
        self._load_prices()
        spy = self._get_series("SPY")
        
        spy_vol_20d = 0.0
        spy_mom_20d = 0.0
        spy_drawdown_60d = 0.0
        
        if spy is not None and len(spy) > 60:
            returns = np.diff(spy) / spy[:-1]
            spy_vol_20d = float(np.std(returns[-20:]) * np.sqrt(252))
            spy_mom_20d = float(np.sum(returns[-20:]))
            running_max = np.maximum.accumulate(spy[-60:])
            drawdowns = spy[-60:] / running_max - 1
            spy_drawdown_60d = float(np.min(drawdowns))
        
        # Compute adjustments
        regime_adj = REGIME_ADJUSTMENTS.get(regime, REGIME_ADJUSTMENTS["unknown"])
        # Scale by confidence
        regime_adj = {k: v * CONFIDENCE_SCALING.get(round(regime_conf * 2) / 2, 0.8)
                      for k, v in regime_adj.items()}
        
        vol_adj = self._compute_vol_adjustment(spy_vol_20d)
        signal_adj = self._compute_signal_adjustment(ensemble_signal, ensemble_agreement)
        dd_adj = self._compute_drawdown_adjustment(spy_drawdown_60d, cb_severity)
        
        # Aggregate adjustments (equal weight for each factor)
        net_adj = {}
        for asset in BASE_ALLOCATION:
            net_adj[asset] = (regime_adj.get(asset, 0.0) * 0.35 +
                              vol_adj.get(asset, 0.0) * 0.30 +
                              signal_adj.get(asset, 0.0) * 0.20 +
                              dd_adj.get(asset, 0.0) * 0.15)
        
        # Apply adjustments to base
        adjusted = {}
        for asset, base_weight in BASE_ALLOCATION.items():
            adjusted[asset] = base_weight + net_adj.get(asset, 0.0)
        
        # Apply bounds
        adjusted = self._apply_bounds(adjusted)
        
        # Build factors
        factors = SizingFactors(
            timestamp=datetime.now().isoformat(),
            regime=regime,
            regime_confidence=regime_conf,
            spy_vol_20d=spy_vol_20d,
            spy_mom_20d=spy_mom_20d,
            spy_drawdown_60d=spy_drawdown_60d,
            ensemble_signal=ensemble_signal,
            ensemble_agreement=ensemble_agreement,
            circuit_breaker_severity=cb_severity,
        )
        
        # Build decision
        decision = SizingDecision(
            timestamp=factors.timestamp,
            base_allocation=dict(BASE_ALLOCATION),
            adjusted_allocation=adjusted,
            adjustments=net_adj,
            regime_adjustment=regime_adj,
            volatility_adjustment=vol_adj,
            signal_adjustment=signal_adj,
            drawdown_adjustment=dd_adj,
            factors=factors,
        )
        
        # Save state
        self.last_allocation = adjusted
        self.last_decision = decision
        self._save_state(decision)
        
        return decision

    # ── State Persistence ─────────────────────────────────────────────────────

    def _load_state(self):
        """Load persisted sizing state."""
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text())
            self.last_allocation = state.get("last_allocation", dict(BASE_ALLOCATION))
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to load sizing state: %s", e)

    def _save_state(self, decision: SizingDecision):
        """Save sizing state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "last_allocation": decision.adjusted_allocation,
            "base_allocation": decision.base_allocation,
            "last_updated": decision.timestamp,
            "regime": decision.factors.regime,
            "regime_confidence": decision.factors.regime_confidence,
            "net_adjustments": decision.adjustments,
            "spy_vol_20d": decision.factors.spy_vol_20d,
            "spy_mom_20d": decision.factors.spy_mom_20d,
            "ensemble_signal": decision.factors.ensemble_signal,
            "ensemble_agreement": decision.factors.ensemble_agreement,
        }
        try:
            save_results_json(state, output_path=str(self.state_path))
        except (OSError, TypeError) as e:
            logger.warning("Failed to save sizing state: %s", e)

    # ── Status Report ────────────────────────────────────────────────────────

    def print_status(self, decision: Optional[SizingDecision] = None):
        """Print formatted sizing status."""
        if decision is None:
            decision = self.compute_allocation()
        
        logger.info("=" * 70)
        logger.info("  ADAPTIVE POSITION SIZING v5.74")
        logger.info("=" * 70)
        logger.info(f"  Timestamp:   {decision.timestamp[:19]}")
        logger.info("")
        logger.info("  Factors:")
        logger.info(f"    Regime:          {decision.factors.regime.upper()} (conf={decision.factors.regime_confidence:.0%})")
        logger.info(f"    SPY 20d Vol:     {decision.factors.spy_vol_20d:.1%}")
        logger.info(f"    SPY 20d Mom:     {decision.factors.spy_mom_20d:.2%}")
        logger.info(f"    SPY 60d DD:      {decision.factors.spy_drawdown_60d:.2%}")
        logger.info(f"    Ensemble Signal: {decision.factors.ensemble_signal:+.3f} (agreement={decision.factors.ensemble_agreement:.0%})")
        logger.info(f"    Circuit Breaker: {decision.factors.circuit_breaker_severity}")
        logger.info("")
        logger.info("  Allocation:")
        logger.info(f"    {'Asset':6s} {'Base':>8s} {'Adj':>8s} {'Regime':>8s} {'Vol':>8s} {'Signal':>8s} {'DD':>8s}")
        logger.info(f"    {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for asset in ["SPY", "GLD", "TLT"]:
            base = decision.base_allocation.get(asset, 0)
            adj = decision.adjusted_allocation.get(asset, 0)
            ra = decision.regime_adjustment.get(asset, 0)
            va = decision.volatility_adjustment.get(asset, 0)
            sa = decision.signal_adjustment.get(asset, 0)
            da = decision.drawdown_adjustment.get(asset, 0)
            logger.info(f"    {asset:6s} {base:>7.1%} {adj:>7.1%} {ra:>+7.2%} {va:>+7.2%} {sa:>+7.2%} {da:>+7.2%}")
        logger.info("")
        logger.info("  Net Adjustments:")
        for asset, adj in decision.adjustments.items():
            logger.info(f"    {asset:6s}: {adj:+.2%}")
        logger.info("")
        logger.info("  Signal Interpretation:")
        signal = decision.factors.ensemble_signal
        if signal > 0.3:
            logger.info("    Bullish — tilt toward equities")
        elif signal < -0.3:
            logger.info("    Bearish — tilt toward safe havens")
        else:
            logger.info("    Neutral — near base allocation")


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    """CLI entry point for adaptive sizing."""
    import sys
    
    sizer = AdaptiveSizer()
    
    if len(sys.argv) < 2 or sys.argv[1] == "adjust":
        decision = sizer.compute_allocation()
        sizer.print_status(decision)
    
    elif sys.argv[1] == "status":
        # Show current state without re-computing
        state_path = STATE_PATH
        if state_path.exists():
            print(json.dumps(json.loads(state_path.read_text()), indent=2))
        else:
            print("No state file found. Run 'adjust' first.")
    
    elif sys.argv[1] == "simulate":
        # Simple backtest: compare static vs dynamic over last 252 days
        print("Simulation: sampling from historical data every 21 days...")
        sizer._load_prices()
        spy = sizer._get_series("SPY")
        if spy is None or len(spy) < 252:
            print("Insufficient data for simulation")
            return
        
        # Sample every 21 trading days (~monthly)
        n = len(spy)
        sample_indices = list(range(n - 60, n, 21))
        
        print(f"  Evaluating {len(sample_indices)} rebalance points over {len(sample_indices) * 21} trading days")
        print()
        
        
        for i, idx in enumerate(sample_indices):
            # We'd need a proper backtest engine for rigorous comparison
            # This is a simplified simulation
            pass
        
        print("  Full simulation requires backtest engine integration.")
        print("  Use: python -m src.backtest.backtest_engine for rigorous comparison.")
    
    else:
        print("Usage: python -m src.strategy.adaptive_sizing [adjust|status|simulate]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
