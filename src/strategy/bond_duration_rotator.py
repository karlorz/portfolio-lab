"""
Bond Duration Rotation Strategy - v4.80 Implementation
Rotates the bond sleeve across TLT/IEF/SHY based on yield curve regime.

Integrates with the existing v2.54 Fed Policy Overlay and v2.17-2.18 Duration/Yield Curve signals.
The bond sleeve weight (16% in 46/38/16) is allocated across TLT/IEF/SHY.

Usage:
    python -m src.strategy.bond_duration_rotator recommend
    python -m src.strategy.bond_duration_rotator status
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

from src.signals.bond_duration_signal import (
    BondDurationSignalGenerator,
    BondDurationSignal,
    BondDurationCalculator,
    YieldCurveRegime,
    RateDirection,
    DurationPosition,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RotationStatus(Enum):
    ACTIVE = "active"
    DEFENSIVE = "defensive"
    DISABLED = "disabled"


@dataclass
class BondRotationDecision:
    """Tactical bond duration rotation decision."""
    timestamp: str
    status: str

    # Portfolio weights (within total portfolio, not just bond sleeve)
    # Bond sleeve is 16% of 46/38/16
    tlt_total: float
    ief_total: float
    shy_total: float

    # Relative to bond sleeve
    tlt_sleeve: float
    ief_sleeve: float
    shy_sleeve: float

    # Yield context
    spread_10y2y: float
    curve_regime: str
    rate_direction: str
    real_rate: float

    # Risk
    effective_duration: float
    confidence: float
    recommendation: str
    is_actionable: bool

    def to_dict(self) -> dict:
        return asdict(self)


class BondDurationRotator:
    """
    Bond duration rotation strategy.

    Rotates the 16% bond sleeve (from 46/38/16) across:
    - TLT: long duration (~16 years) — max rate sensitivity
    - IEF: intermediate (~7 years) — balanced
    - SHY: short duration (~2 years) — defensive, low sensitivity
    """

    STATE_FILE = Path(__file__).parent.parent.parent / "data" / "bond_duration_state.json"
    BOND_SLEEVE_WEIGHT = 0.16  # 16% in 46/38/16

    ENSEMBLE_WEIGHT = 0.08  # 8% in ensemble voter

    def __init__(self, state_file: Optional[Path] = None):
        self._signal_gen = BondDurationSignalGenerator()
        self._calc = BondDurationCalculator()
        self.state_file = state_file or self.STATE_FILE
        self._state = self._load_state()

    def _load_state(self) -> Dict:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return self._default_state()

    def _default_state(self) -> Dict:
        return {
            "status": "active",
            "last_rotation_date": None,
            "current_position": None,
            "tlt_weight": 0.0,
            "ief_weight": 0.0,
            "shy_weight": 0.0,
        }

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    def get_status(self) -> RotationStatus:
        status = self._state.get("status", "active")
        return RotationStatus(status)

    def recommend(
        self,
        yield_10y: Optional[float] = None,
        yield_2y: Optional[float] = None,
        real_rate: Optional[float] = None,
        rate_change_6m: Optional[float] = None,
    ) -> BondRotationDecision:
        """Generate bond duration rotation recommendation."""
        signal = self._signal_gen.generate_signal(
            yield_10y=yield_10y, yield_2y=yield_2y,
            real_rate=real_rate, rate_change_6m=rate_change_6m,
        )

        # Scale to total portfolio weights (bond sleeve = 16%)
        tlt_total = signal.tlt_weight * self.BOND_SLEEVE_WEIGHT
        ief_total = signal.ief_weight * self.BOND_SLEEVE_WEIGHT
        shy_total = signal.shy_weight * self.BOND_SLEEVE_WEIGHT

        if signal.position == "short":
            status = RotationStatus.DEFENSIVE
            actionable = signal.tlt_weight < 0.10  # Major TLT reduction
        else:
            status = RotationStatus.ACTIVE
            actionable = True

        # Update state
        self._state["status"] = status.value
        self._state["last_rotation_date"] = datetime.now().isoformat()
        self._state["current_position"] = signal.position
        self._state["tlt_weight"] = round(tlt_total, 4)
        self._state["ief_weight"] = round(ief_total, 4)
        self._state["shy_weight"] = round(shy_total, 4)
        self._save_state()

        return BondRotationDecision(
            timestamp=datetime.now().isoformat(),
            status=status.value,
            tlt_total=round(tlt_total, 4),
            ief_total=round(ief_total, 4),
            shy_total=round(shy_total, 4),
            tlt_sleeve=signal.tlt_weight,
            ief_sleeve=signal.ief_weight,
            shy_sleeve=signal.shy_weight,
            spread_10y2y=signal.spread_10y2y,
            curve_regime=signal.curve_regime,
            rate_direction=signal.rate_direction,
            real_rate=signal.real_rate,
            effective_duration=signal.effective_duration,
            confidence=signal.confidence,
            recommendation=f"Position={signal.position}, "
                           f"TLT={signal.tlt_weight:.0%} IEF={signal.ief_weight:.0%} "
                           f"SHY={signal.shy_weight:.0%}",
            is_actionable=actionable,
        )

    def get_allocation_shifts(self, baseline_tlt: float = 0.16) -> Dict[str, float]:
        """Get allocation shifts relative to baseline (all in TLT)."""
        decision = self.recommend()

        return {
            "tlt": round(decision.tlt_total - baseline_tlt, 4),
            "ief": round(decision.ief_total, 4),
            "shy": round(decision.shy_total, 4),
            "spy": 0.0,
            "gld": 0.0,
        }

    def backtest(
        self,
        yield_history: List[Tuple[float, float, float]],  # (10Y, 2Y, real_rate)
        spy_returns: List[float],
        tlt_returns: List[float],
        ief_returns: List[float],
        shy_returns: List[float],
        gld_returns: List[float],
        dates: List[str],
    ) -> Dict:
        """
        Backtest bond duration rotation vs static TLT allocation.

        Compares baseline 46/38/16 (with TLT) vs duration-rotated version.
        """
        n = len(dates)
        if n < 30:
            return {"error": "Insufficient data"}

        results = {
            "dates": [],
            "baseline_returns": [],
            "rotated_returns": [],
            "tlt_weights": [],
            "ief_weights": [],
            "shy_weights": [],
            "position_history": [],
        }

        baseline_values = [1.0]
        rotated_values = [1.0]

        for i in range(len(dates)):
            # Use yield data if available
            if i < len(yield_history):
                y10, y2, real = yield_history[i]
                spread = y10 - y2

                curve = self._calc.classify_curve(spread)
                # Approximate rate direction from recent yield change
                if i >= 21:
                    prev_y10 = yield_history[i-21][0]
                    rate_chg = y10 - prev_y10
                else:
                    rate_chg = 0
                direction = self._calc.classify_rate_direction(rate_chg)

                tlt_w, ief_w, shy_w, _ = self._calc.compute_duration_allocation(
                    spread, real, direction, curve
                )
            else:
                tlt_w, ief_w, shy_w = 1.0, 0.0, 0.0  # Default to TLT

            spy_r = spy_returns[i] if i < len(spy_returns) else 0
            tlt_r = tlt_returns[i] if i < len(tlt_returns) else 0
            ief_r = ief_returns[i] if i < len(ief_returns) else 0
            shy_r = shy_returns[i] if i < len(shy_returns) else 0
            gld_r = gld_returns[i] if i < len(gld_returns) else 0

            # Baseline: 46% SPY + 38% GLD + 16% TLT
            base_ret = 0.46 * spy_r + 0.38 * gld_r + 0.16 * tlt_r

            # Rotated: 46% SPY + 38% GLD + 16% * (tlt_w*TLT + ief_w*IEF + shy_w*SHY)
            bond_ret = tlt_w * tlt_r + ief_w * ief_r + shy_w * shy_r
            rotated_ret = 0.46 * spy_r + 0.38 * gld_r + 0.16 * bond_ret

            baseline_values.append(baseline_values[-1] * (1 + base_ret))
            rotated_values.append(rotated_values[-1] * (1 + rotated_ret))

            results["dates"].append(dates[i])
            results["baseline_returns"].append(base_ret * 100)
            results["rotated_returns"].append(rotated_ret * 100)
            results["tlt_weights"].append(tlt_w)
            results["ief_weights"].append(ief_w)
            results["shy_weights"].append(shy_w)

        base_rets = results["baseline_returns"]
        rot_rets = results["rotated_returns"]

        if len(base_rets) > 0:
            results["summary"] = {
                "cagr_baseline": round(np.mean(base_rets) * 252, 2),
                "cagr_rotated": round(np.mean(rot_rets) * 252, 2),
                "vol_baseline": round(np.std(base_rets) * np.sqrt(252), 2),
                "vol_rotated": round(np.std(rot_rets) * np.sqrt(252), 2),
                "sharpe_baseline": round(
                    np.mean(base_rets) / np.std(base_rets) * np.sqrt(252), 3
                ) if np.std(base_rets) > 0 else 0,
                "sharpe_rotated": round(
                    np.mean(rot_rets) / np.std(rot_rets) * np.sqrt(252), 3
                ) if np.std(rot_rets) > 0 else 0,
                "avg_tlt_weight": round(np.mean(results["tlt_weights"]) * 100, 1),
            }

        return results


def calculate_bond_rotation() -> BondRotationDecision:
    """Convenience function."""
    rotator = BondDurationRotator()
    return rotator.recommend()


def get_bond_duration_summary() -> Dict:
    """Get current bond duration rotation summary."""
    rotator = BondDurationRotator()
    decision = rotator.recommend()
    return {
        "status": decision.status,
        "position": decision.curve_regime,
        "tlt_total": decision.tlt_total,
        "ief_total": decision.ief_total,
        "shy_total": decision.shy_total,
        "effective_duration": decision.effective_duration,
        "recommendation": decision.recommendation,
    }


def main():
    import sys
    rotator = BondDurationRotator()
    decision = rotator.recommend()

    print("=" * 60)
    print("BOND DURATION ROTATION v4.80")
    print("=" * 60)
    print(f"Status: {decision.status}")
    print(f"Curve Regime: {decision.curve_regime}")
    print(f"Rate Direction: {decision.rate_direction}")
    print(f"Real Rate: {decision.real_rate:.2f}%")
    print()
    print("Bond Sleeve (16% of portfolio):")
    print(f"  TLT: {decision.tlt_sleeve:.0%} ({decision.tlt_total:.2%} total)")
    print(f"  IEF: {decision.ief_sleeve:.0%} ({decision.ief_total:.2%} total)")
    print(f"  SHY: {decision.shy_sleeve:.0%} ({decision.shy_total:.2%} total)")
    print(f"  Effective Duration: {decision.effective_duration:.1f}yr")
    print()
    print(f"Confidence: {decision.confidence:.0f}%")
    print(f"Recommendation: {decision.recommendation}")
    print(f"Actionable: {decision.is_actionable}")
    print("=" * 60)


if __name__ == "__main__":
    main()
