"""
Kurtosis Regime Detection - v4.91 Implementation
Detects market regime changes using rolling excess kurtosis of return distributions.

Based on IIT KGP Quant Games 2026 winning strategy:
- KER (Kurtosis-based Entropy Ratio) for regime detection
- LOW_KURTOSIS: normal markets, trend-following works
- HIGH_KURTOSIS: fat tails, mean-reversion preferred
- TRANSITIONING: between regimes, reduce allocation

Complements HMM regime detector (v2.53) with distributional approach.
No ML deps — pure numpy.

Usage:
    python -m src.regime.kurtosis_regime detect
    python -m src.regime.kurtosis_regime status
"""

import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KurtosisRegime(Enum):
    LOW_KURTOSIS = "low_kurtosis"        # Normal distribution (k < 3.5)
    NORMAL = "normal"                     # Slightly elevated (3.5 < k < 5)
    HIGH_KURTOSIS = "high_kurtosis"      # Fat tails (5 < k < 8)
    EXTREME_KURTOSIS = "extreme_kurtosis"  # Crisis tails (k > 8)


class StrategyPreference(Enum):
    TREND_FOLLOWING = "trend_following"    # TSMOM preferred
    MEAN_REVERSION = "mean_reversion"      # VIX mean-reversion preferred
    BALANCED = "balanced"                   # Equal weight
    DEFENSIVE = "defensive"                 # Reduce all exposure


@dataclass
class KurtosisRegimeSignal:
    """Complete kurtosis regime assessment."""
    timestamp: str

    # Raw metrics
    kurtosis_20d: float      # Short-term kurtosis
    kurtosis_60d: float      # Medium-term
    kurtosis_120d: float     # Long-term
    ker_ratio: float         # Kurtosis Entropy Ratio (20d/120d)

    # Regime
    regime: str
    regime_confidence: float
    is_transitioning: bool
    transition_speed: float  # How fast is kurtosis changing

    # Strategy preference
    strategy_preference: str
    tsom_weight: float       # Trend-following allocation (0-1)
    mr_weight: float         # Mean-reversion allocation (0-1)

    # Risk
    fat_tail_risk: float     # 0-1, higher = more tail risk
    recommended_exposure: float  # 0-1, recommended portfolio exposure

    confidence: float
    explanation: str

    def to_dict(self) -> dict:
        return asdict(self)


class KurtosisRegimeDetector:
    """
    Rolling kurtosis regime detector.

    Computes excess kurtosis across multiple time windows and the
    KER (Kurtosis Entropy Ratio) for regime classification.

    Excess kurtosis = sample kurtosis - 3 (normal distribution = 0 excess)
    KER = kurtosis_short / kurtosis_long (regime change indicator)
    """

    # Regime thresholds (excess kurtosis)
    LOW_KURTOSIS_MAX = 0.5     # k < 3.5 (excess < 0.5)
    HIGH_KURTOSIS_MIN = 2.0    # k > 5 (excess > 2)
    EXTREME_KURTOSIS_MIN = 5.0 # k > 8 (excess > 5)

    # KER thresholds
    KER_SHIFT_UP = 1.3    # KER > 1.3 = regime shifting to high kurtosis
    KER_SHIFT_DOWN = 0.7  # KER < 0.7 = regime shifting to low kurtosis

    def __init__(self, short_window: int = 20, medium_window: int = 60,
                 long_window: int = 120):
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window

    def compute_excess_kurtosis(self, returns: List[float]) -> float:
        """Compute excess kurtosis (sample kurtosis - 3)."""
        n = len(returns)
        if n < 4:
            return 0.0

        mean = np.mean(returns)
        # Use n (not n-1) for sample kurtosis formula
        m2 = np.sum((returns - mean) ** 2) / n
        m4 = np.sum((returns - mean) ** 4) / n

        if m2 == 0:
            return 0.0

        # Excess kurtosis: (m4 / m2^2) - 3
        kurt = (m4 / (m2 ** 2)) - 3
        return float(kurt)

    def compute_rolling_kurtosis(self, returns: List[float],
                                  window: int) -> List[float]:
        """Compute rolling excess kurtosis series."""
        if len(returns) < window:
            return [0.0] * len(returns)

        result = [0.0] * (window - 1)
        for i in range(window - 1, len(returns)):
            window_rets = returns[i - window + 1:i + 1]
            result.append(self.compute_excess_kurtosis(window_rets))

        return result

    def compute_ker(self, kurt_short: float, kurt_long: float) -> float:
        """Kurtosis Entropy Ratio: short-term / long-term kurtosis."""
        if kurt_long + 3 <= 0:
            return 1.0
        short_abs = kurt_short + 3  # Absolute kurtosis
        long_abs = kurt_long + 3
        if long_abs <= 0:
            return 1.0
        return short_abs / long_abs

    def classify_regime(self, excess_kurtosis: float) -> Tuple[KurtosisRegime, float]:
        """Classify regime from excess kurtosis value."""
        if excess_kurtosis >= self.EXTREME_KURTOSIS_MIN:
            strength = min(1.0, (excess_kurtosis - 5) / 5)
            return KurtosisRegime.EXTREME_KURTOSIS, 0.8 + strength * 0.2
        elif excess_kurtosis >= self.HIGH_KURTOSIS_MIN:
            strength = (excess_kurtosis - 2) / 3
            return KurtosisRegime.HIGH_KURTOSIS, 0.6 + strength * 0.2
        elif excess_kurtosis >= self.LOW_KURTOSIS_MAX:
            strength = (excess_kurtosis - 0.5) / 1.5
            return KurtosisRegime.NORMAL, 0.5 + strength * 0.2
        else:
            strength = 1.0 - excess_kurtosis / 0.5
            return KurtosisRegime.LOW_KURTOSIS, 0.5 + strength * 0.3

    def compute_strategy_weights(
        self, regime: KurtosisRegime, ker: float, is_transitioning: bool
    ) -> Tuple[float, float, StrategyPreference]:
        """
        Compute TSMOM (trend) vs Mean-Reversion weights.

        LOW_KURTOSIS → TSMOM heavy (trend works in normal markets)
        HIGH_KURTOSIS → Mean-reversion heavy (dips are mean-reverting)
        TRANSITIONING → balanced + reduce exposure
        """
        if regime == KurtosisRegime.EXTREME_KURTOSIS:
            return 0.1, 0.9, StrategyPreference.DEFENSIVE

        if is_transitioning:
            if ker > self.KER_SHIFT_UP:
                # Moving toward higher kurtosis → shift to mean-reversion
                transition = min(1.0, (ker - 1.0) / 0.5)
                tsom = 0.7 - transition * 0.5
                mr = 0.3 + transition * 0.5
                return max(0.1, tsom), min(0.9, mr), StrategyPreference.BALANCED
            else:
                # Moving toward lower kurtosis → shift to trend
                transition = min(1.0, (1.0 - ker) / 0.5)
                tsom = 0.3 + transition * 0.5
                mr = 0.7 - transition * 0.5
                return min(0.9, tsom), max(0.1, mr), StrategyPreference.BALANCED

        if regime == KurtosisRegime.LOW_KURTOSIS:
            return 0.85, 0.15, StrategyPreference.TREND_FOLLOWING
        elif regime == KurtosisRegime.NORMAL:
            return 0.70, 0.30, StrategyPreference.TREND_FOLLOWING
        elif regime == KurtosisRegime.HIGH_KURTOSIS:
            return 0.25, 0.75, StrategyPreference.MEAN_REVERSION

        return 0.50, 0.50, StrategyPreference.BALANCED


class KurtosisRegimeSignalGenerator:
    """
    Main signal generator for kurtosis regime detection.
    """

    OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "signals" / "kurtosis_regime.json"

    def __init__(self):
        self.detector = KurtosisRegimeDetector()
        self._ensure_dirs()

    def _ensure_dirs(self):
        sig_dir = Path(__file__).parent.parent.parent / "data" / "signals"
        sig_dir.mkdir(parents=True, exist_ok=True)

    def generate_signal(self, returns: Optional[List[float]] = None) -> KurtosisRegimeSignal:
        """Generate kurtosis regime signal from return series."""
        if returns is None or len(returns) < self.detector.long_window:
            # Generate synthetic returns for demo/testing
            rng = np.random.RandomState(42)
            returns = list(rng.normal(0, 0.01, 200))

        # Compute kurtosis at different windows
        if len(returns) >= self.detector.short_window:
            k20 = self.detector.compute_excess_kurtosis(
                returns[-self.detector.short_window:]
            )
        else:
            k20 = 0.0

        if len(returns) >= self.detector.medium_window:
            k60 = self.detector.compute_excess_kurtosis(
                returns[-self.detector.medium_window:]
            )
        else:
            k60 = 0.0

        if len(returns) >= self.detector.long_window:
            k120 = self.detector.compute_excess_kurtosis(
                returns[-self.detector.long_window:]
            )
        else:
            k120 = 0.0

        ker = self.detector.compute_ker(k20, k120)

        # Use 60-day kurtosis for regime classification (medium-term = most reliable)
        regime, confidence = self.detector.classify_regime(k60)

        # Detect transition
        is_transitioning = ker > self.detector.KER_SHIFT_UP or ker < self.detector.KER_SHIFT_DOWN
        transition_speed = abs(ker - 1.0)

        tsom_w, mr_w, preference = self.detector.compute_strategy_weights(
            regime, ker, is_transitioning
        )

        # Fat tail risk: 0-1 scale
        if k60 > 5:
            fat_tail_risk = 1.0
        elif k60 > 2:
            fat_tail_risk = (k60 - 2) / 3
        else:
            fat_tail_risk = max(0, k60 / 2)

        # Recommended portfolio exposure
        if regime == KurtosisRegime.EXTREME_KURTOSIS:
            exposure = 0.5
        elif is_transitioning:
            exposure = 0.75
        else:
            exposure = 1.0

        # Explanation
        if regime == KurtosisRegime.EXTREME_KURTOSIS:
            explanation = f"Extreme kurtosis ({k60 + 3:.1f}): fat tails, prefer defense"
        elif regime == KurtosisRegime.HIGH_KURTOSIS:
            explanation = f"High kurtosis ({k60 + 3:.1f}): route to mean-reversion ({mr_w:.0%})"
        elif is_transitioning:
            direction = "↑" if ker > 1 else "↓"
            explanation = f"Transitioning {direction} (KER={ker:.2f}): balanced routing"
        else:
            explanation = f"Low/normal kurtosis ({k60 + 3:.1f}): route to trend ({tsom_w:.0%})"

        return KurtosisRegimeSignal(
            timestamp=datetime.now().isoformat(),
            kurtosis_20d=round(k20 + 3, 2),
            kurtosis_60d=round(k60 + 3, 2),
            kurtosis_120d=round(k120 + 3, 2),
            ker_ratio=round(ker, 3),
            regime=regime.value,
            regime_confidence=round(confidence, 3),
            is_transitioning=is_transitioning,
            transition_speed=round(transition_speed, 3),
            strategy_preference=preference.value,
            tsom_weight=round(tsom_w, 3),
            mr_weight=round(mr_w, 3),
            fat_tail_risk=round(fat_tail_risk, 3),
            recommended_exposure=round(exposure, 2),
            confidence=round(confidence * 100, 1),
            explanation=explanation,
        )

    def save_signal(self, signal: KurtosisRegimeSignal):
        with open(self.OUTPUT_PATH, "w") as f:
            json.dump(signal.to_dict(), f, indent=2)


def detect_kurtosis_regime(returns: Optional[List[float]] = None) -> KurtosisRegimeSignal:
    """Convenience function."""
    gen = KurtosisRegimeSignalGenerator()
    return gen.generate_signal(returns)


def main():
    import sys
    gen = KurtosisRegimeSignalGenerator()

    # Generate test returns: normal + some fat-tail periods
    rng = np.random.RandomState(42)
    returns = list(rng.normal(0, 0.01, 180))
    # Add some fat-tail days
    for i in [50, 55, 100, 105, 150, 155]:
        if i < len(returns):
            returns[i] = rng.normal(0, 0.04)

    signal = gen.generate_signal(returns)

    print("=" * 60)
    print("KURTOSIS REGIME DETECTOR v4.91")
    print("=" * 60)
    print(f"Timestamp: {signal.timestamp}")
    print()
    print("Kurtosis (absolute):")
    print(f"  20-day: {signal.kurtosis_20d:.2f}")
    print(f"  60-day: {signal.kurtosis_60d:.2f}")
    print(f"  120-day: {signal.kurtosis_120d:.2f}")
    print(f"  KER: {signal.ker_ratio:.3f}")
    print()
    print(f"Regime: {signal.regime}")
    print(f"Confidence: {signal.regime_confidence:.1%}")
    print(f"Transitioning: {signal.is_transitioning}")
    print()
    print("Strategy Routing:")
    print(f"  Preference: {signal.strategy_preference}")
    print(f"  TSMOM (trend): {signal.tsom_weight:.0%}")
    print(f"  Mean-Reversion: {signal.mr_weight:.0%}")
    print()
    print(f"Fat Tail Risk: {signal.fat_tail_risk:.1%}")
    print(f"Recommended Exposure: {signal.recommended_exposure:.0%}")
    print(f"Explanation: {signal.explanation}")
    print("=" * 60)

    if "--save" in sys.argv:
        gen.save_signal(signal)


if __name__ == "__main__":
    main()
