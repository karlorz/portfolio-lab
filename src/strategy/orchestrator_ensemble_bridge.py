"""
Orchestrator-EnsembleVoter Bridge - v4.90 Integration
Connects the unified overlay orchestrator to the EnsembleVoter (v2.58).

Converts the unified orchestrator's multi-asset recommendation into
ensemble voter signal readings. The unified overlay becomes a top-level
signal source alongside TSFM, HMM, CTA, etc.

Usage:
    python -m src.strategy.orchestrator_ensemble_bridge signal
    python -m src.strategy.orchestrator_ensemble_bridge integrate
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

from .unified_orchestrator import UnifiedOrchestrator, UnifiedRecommendation
from .ensemble_voter import SignalSource, SignalReading, Regime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BridgeSignalType(Enum):
    ALLOCATION = "allocation"      # Portfolio weight recommendation
    DIRECTION = "direction"        # Directional bias per asset
    RISK = "risk"                  # Risk overlay signal


@dataclass
class UnifiedSignalReading:
    """
    Ensemble-compatible signal reading from the unified orchestrator.

    Maps the orchestrator's 7-asset allocation to the ensemble voter's
    signal format (-1 to +1 directional bias per asset class).
    """
    timestamp: str
    source: str  # "unified_overlay"

    # Overall signal (-1 to +1): composite directional bias
    value: float
    confidence: float
    weight: float  # Weight in ensemble (recommended: 0.20)

    # Per-asset directional signals
    spy_signal: float    # -1 (bearish) to +1 (bullish)
    gld_signal: float
    tlt_signal: float
    ief_signal: float
    shy_signal: float
    btc_signal: float
    eth_signal: float

    # Risk signal: -1 (de-risk) to +1 (risk-on)
    risk_signal: float

    # Execution timing signal: 0 (defer) to 1 (execute now)
    execution_signal: float

    # Explanation
    explanation: str
    num_overlays_active: int
    conflict_count: int

    def to_dict(self) -> dict:
        return asdict(self)

    def to_signal_reading(self) -> SignalReading:
        """Convert to standard EnsembleVoter SignalReading."""
        return SignalReading(
            source=SignalSource.UNIFIED_OVERLAY,
            timestamp=self.timestamp,
            value=self.value,
            confidence=self.confidence,
            weight=self.weight,
            regime_fit="all",  # Unified overlay works in all regimes
            asset_signals={
                "SPY": self.spy_signal,
                "GLD": self.gld_signal,
                "TLT": self.tlt_signal,
                "IEF": self.ief_signal,
                "SHY": self.shy_signal,
                "BTC": self.btc_signal,
                "ETH": self.eth_signal,
            },
            explanation=self.explanation,
        )


class OrchestratorEnsembleBridge:
    """
    Bridge between unified orchestrator and ensemble voter.

    Converts multi-asset allocation recommendations into
    directional signals the ensemble voter can consume.

    Weight recommendation: 20% in ensemble (highest single-source weight).
    The unified overlay aggregates 4 sub-overlays, so it deserves
    significant weight in the final vote.
    """

    RECOMMENDED_ENSEMBLE_WEIGHT = 0.20

    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    OUTPUT_PATH = DATA_DIR / "signals" / "unified_ensemble_signal.json"

    def __init__(self):
        self._orch = UnifiedOrchestrator()

    def _ensure_dirs(self):
        sig_dir = self.DATA_DIR / "signals"
        sig_dir.mkdir(parents=True, exist_ok=True)

    def generate_signal(self) -> UnifiedSignalReading:
        """Generate ensemble-compatible signal from unified orchestrator."""
        rec = self._orch.recommend()

        # Convert allocation weights to directional signals (-1 to +1)
        spy_sig = self._weight_to_signal(rec.spy, rec.baseline_spy)
        gld_sig = self._weight_to_signal(rec.gld, rec.baseline_gld)
        tlt_sig = self._weight_to_signal(rec.tlt, rec.baseline_tlt)

        # Baseline for IEF/SHY/BTC/ETH is 0
        ief_sig = self._weight_to_signal(rec.ief, 0.0)
        shy_sig = self._weight_to_signal(rec.shy, 0.0)
        btc_sig = self._weight_to_signal(rec.btc, 0.0)
        eth_sig = self._weight_to_signal(rec.eth, 0.0)

        # Composite value: average of active asset signals
        asset_signals = [spy_sig, gld_sig, tlt_sig, ief_sig, shy_sig, btc_sig, eth_sig]
        non_zero = [s for s in asset_signals if abs(s) > 0.05]
        composite = np.mean(non_zero) if non_zero else 0.0

        # Risk signal: inverted by conflict count and SPY reduction
        if rec.conflict_count > 2:
            risk_signal = -0.5  # Significant conflicts → de-risk
        elif rec.total_spy_delta < -0.03:
            risk_signal = -0.3  # Reducing equity → cautious
        elif rec.total_spy_delta > 0.02:
            risk_signal = 0.3   # Adding equity → risk-on
        else:
            risk_signal = 0.0

        # Execution signal from calendar modifier
        exec_signal = rec.calendar_modifier

        # Active overlay count
        active_count = sum(
            1 for c in rec.contributions
            if c.status == "active"
        )

        # Explanation
        explanation = (
            f"Unified({active_count}/4 active): "
            f"SPY {rec.spy:.1%} (base {rec.baseline_spy:.0%}), "
            f"GLD {rec.gld:.1%}, TLT {rec.tlt:.1%}, "
            f"IEF {rec.ief:.1%}, SHY {rec.shy:.1%}, "
            f"BTC {rec.btc:.1%}, ETH {rec.eth:.1%}. "
            f"Risk: {risk_signal:+.1f}, Sharpe est: {rec.estimated_sharpe:.3f}"
        )

        return UnifiedSignalReading(
            timestamp=datetime.now().isoformat(),
            source="unified_overlay",
            value=round(float(composite), 4),
            confidence=round(rec.confidence / 100, 3),
            weight=self.RECOMMENDED_ENSEMBLE_WEIGHT,
            spy_signal=round(spy_sig, 3),
            gld_signal=round(gld_sig, 3),
            tlt_signal=round(tlt_sig, 3),
            ief_signal=round(ief_sig, 3),
            shy_signal=round(shy_sig, 3),
            btc_signal=round(btc_sig, 3),
            eth_signal=round(eth_sig, 3),
            risk_signal=round(risk_signal, 3),
            execution_signal=round(exec_signal, 3),
            explanation=explanation,
            num_overlays_active=active_count,
            conflict_count=rec.conflict_count,
        )

    @staticmethod
    def _weight_to_signal(current: float, baseline: float) -> float:
        """Convert a weight allocation to a -1/+1 directional signal.

        delta > +5pp → +1 (strong overweight)
        delta +2pp to +5pp → +0.5 (moderate overweight)
        delta -2pp to +2pp → 0 (neutral)
        delta -5pp to -2pp → -0.5 (moderate underweight)
        delta < -5pp → -1 (strong underweight)
        """
        delta = current - baseline
        if delta > 0.05:
            return 1.0
        elif delta > 0.02:
            return 0.5
        elif delta > -0.02:
            return 0.0
        elif delta > -0.05:
            return -0.5
        return -1.0

    def save_signal(self, signal: UnifiedSignalReading):
        self._ensure_dirs()
        with open(self.OUTPUT_PATH, "w") as f:
            json.dump(signal.to_dict(), f, indent=2)

    def get_ensemble_reading(self) -> SignalReading:
        """Get a SignalReading ready for ensemble voter consumption."""
        unified = self.generate_signal()
        return unified.to_signal_reading()

    def compare_with_ensemble_source(self, other_source: str) -> Dict:
        """Compare unified overlay signal with another ensemble source."""
        unified = self.generate_signal()
        return {
            "unified_value": unified.value,
            "unified_confidence": unified.confidence,
            "unified_weight": unified.weight,
            "compared_source": other_source,
            "active_overlays": unified.num_overlays_active,
            "conflicts": unified.conflict_count,
            "recommendation": (
                "integrate" if unified.num_overlays_active >= 2
                else "standalone"
            ),
        }


def get_unified_ensemble_signal() -> UnifiedSignalReading:
    """Convenience function."""
    bridge = OrchestratorEnsembleBridge()
    return bridge.generate_signal()


def get_unified_ensemble_reading() -> SignalReading:
    """Get SignalReading for direct ensemble voter consumption."""
    bridge = OrchestratorEnsembleBridge()
    return bridge.get_ensemble_reading()


def main():
    import sys
    bridge = OrchestratorEnsembleBridge()
    signal = bridge.generate_signal()

    print("=" * 60)
    print("ORCHESTRATOR-ENSEMBLE VOTER BRIDGE v4.90")
    print("=" * 60)
    print(f"Timestamp: {signal.timestamp}")
    print(f"Source: {signal.source}")
    print()
    print("Composite Signal:")
    print(f"  Value: {signal.value:+.3f}")
    print(f"  Confidence: {signal.confidence:.1%}")
    print(f"  Weight: {signal.weight:.0%} (recommended in ensemble)")
    print()
    print("Asset Directional Signals (-1 bearish to +1 bullish):")
    print(f"  SPY: {signal.spy_signal:+.1f}")
    print(f"  GLD: {signal.gld_signal:+.1f}")
    print(f"  TLT: {signal.tlt_signal:+.1f}")
    print(f"  IEF: {signal.ief_signal:+.1f}")
    print(f"  SHY: {signal.shy_signal:+.1f}")
    print(f"  BTC: {signal.btc_signal:+.1f}")
    print(f"  ETH: {signal.eth_signal:+.1f}")
    print()
    print(f"Risk Signal: {signal.risk_signal:+.1f}")
    print(f"Execution Signal: {signal.execution_signal:.2f}")
    print()
    print(f"Active Overlays: {signal.num_overlays_active}/4")
    print(f"Conflicts: {signal.conflict_count}")
    print()
    print(f"Explanation: {signal.explanation}")
    print("=" * 60)

    if "--save" in sys.argv:
        bridge.save_signal(signal)

    if "--reading" in sys.argv:
        reading = bridge.get_ensemble_reading()
        print(f"\nEnsembleVoter SignalReading ready: "
              f"source={reading.source.value}, value={reading.value:.3f}")


if __name__ == "__main__":
    main()
