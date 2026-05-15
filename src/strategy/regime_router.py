"""
Strategy Regime Router - v4.91 Implementation
Routes between trend-following (TSMOM) and mean-reversion based on kurtosis regime.

Integrates with:
- TSMOM Overlay (v2.52): trend-following during low-kurtosis regimes
- VIX Mean-Reversion (v4.81): mean-reversion during high-kurtosis regimes
- Kurtosis Regime Detector (v4.91): regime classification engine

Usage:
    python -m src.strategy.regime_router route
    python -m src.strategy.regime_router status
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

from ..regime.kurtosis_regime import (
    KurtosisRegimeSignalGenerator,
    KurtosisRegime,
    StrategyPreference,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RouterDecision:
    """Strategy routing decision."""
    timestamp: str

    # Kurtosis state
    kurtosis_regime: str
    ker_ratio: float
    is_transitioning: bool

    # Strategy weights
    tsom_weight: float         # Trend-following allocation
    mr_weight: float           # Mean-reversion allocation
    cash_weight: float         # Unallocated / defensive
    strategy_preference: str

    # Exposure
    recommended_exposure: float
    fat_tail_risk: float

    # Meta
    confidence: float
    explanation: str
    is_actionable: bool

    def to_dict(self) -> dict:
        return asdict(self)


class RegimeRouter:
    """
    Routes capital between trend-following and mean-reversion strategies
    based on kurtosis regime.

    Portfolio-level integration: the router's TSMOM/MR split feeds into
    the overall portfolio allocation alongside the unified orchestrator.
    """

    ENSEMBLE_WEIGHT = 0.05  # 5% weight in ensemble voter

    def __init__(self):
        self._signal_gen = KurtosisRegimeSignalGenerator()

    def route(self, returns: Optional[List[float]] = None) -> RouterDecision:
        """Generate strategy routing decision."""
        signal = self._signal_gen.generate_signal(returns)

        # Cash weight: uninvested during extreme regimes or transitions
        if signal.strategy_preference == "defensive":
            cash = 0.3
        elif signal.is_transitioning:
            cash = 0.1
        else:
            cash = 0.0

        # Adjust TSMOM/MR weights to account for cash
        if cash > 0:
            scale = 1.0 - cash
            tsom = signal.tsom_weight * scale
            mr = signal.mr_weight * scale
        else:
            tsom = signal.tsom_weight
            mr = signal.mr_weight

        actionable = (
            signal.is_transitioning or
            signal.strategy_preference != "trend_following"
        )

        return RouterDecision(
            timestamp=datetime.now().isoformat(),
            kurtosis_regime=signal.regime,
            ker_ratio=signal.ker_ratio,
            is_transitioning=signal.is_transitioning,
            tsom_weight=round(tsom, 3),
            mr_weight=round(mr, 3),
            cash_weight=round(cash, 2),
            strategy_preference=signal.strategy_preference,
            recommended_exposure=signal.recommended_exposure,
            fat_tail_risk=signal.fat_tail_risk,
            confidence=signal.confidence,
            explanation=signal.explanation,
            is_actionable=actionable,
        )

    def get_ensemble_signal(self, returns: Optional[List[float]] = None) -> Dict:
        """Get ensemble voter signal for integration."""
        decision = self.route(returns)

        # Convert to ensemble-compatible signal
        if decision.strategy_preference == "trend_following":
            value = 0.3  # Positive = favor trend/momentum
        elif decision.strategy_preference == "mean_reversion":
            value = -0.2  # Negative = favor mean-reversion
        elif decision.strategy_preference == "defensive":
            value = -0.5  # Strong defensive signal
        else:
            value = 0.0

        return {
            "source": "regime_router",
            "signal": value,
            "weight": self.ENSEMBLE_WEIGHT,
            "confidence": decision.confidence / 100,
            "recommendation": decision.explanation,
            "tsom_weight": decision.tsom_weight,
            "mr_weight": decision.mr_weight,
        }


def route_regime(returns: Optional[List[float]] = None) -> RouterDecision:
    """Convenience function."""
    router = RegimeRouter()
    return router.route(returns)


def main():
    import sys
    router = RegimeRouter()

    # Generate test returns with regime shifts
    rng = np.random.RandomState(42)
    n = 200
    returns = []
    for i in range(n):
        if 80 <= i < 120:  # High-kurtosis period
            returns.append(rng.normal(0, 0.025) if rng.random() < 0.1 else rng.normal(0, 0.01))
        else:
            returns.append(rng.normal(0, 0.01))

    decision = router.route(returns)

    print("=" * 60)
    print("STRATEGY REGIME ROUTER v4.91")
    print("=" * 60)
    print(f"Timestamp: {decision.timestamp}")
    print()
    print("Kurtosis State:")
    print(f"  Regime: {decision.kurtosis_regime}")
    print(f"  KER: {decision.ker_ratio:.3f}")
    print(f"  Transitioning: {decision.is_transitioning}")
    print()
    print("Strategy Allocation:")
    print(f"  TSMOM (Trend): {decision.tsom_weight:.0%}")
    print(f"  Mean-Reversion: {decision.mr_weight:.0%}")
    print(f"  Cash/Defensive: {decision.cash_weight:.0%}")
    print(f"  Preference: {decision.strategy_preference}")
    print()
    print(f"Exposure: {decision.recommended_exposure:.0%}")
    print(f"Fat Tail Risk: {decision.fat_tail_risk:.1%}")
    print(f"Confidence: {decision.confidence:.0f}%")
    print()
    print(f"Actionable: {decision.is_actionable}")
    print(f"Explanation: {decision.explanation}")
    print("=" * 60)


if __name__ == "__main__":
    main()
