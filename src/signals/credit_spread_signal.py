"""
Credit spread signal generator for ensemble voter integration.

Generates allocation recommendations based on LQD/HYG spread dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from src.data.credit_fetcher import CreditFetcher, CreditData, RISK_OFF_THRESHOLD, RISK_ON_THRESHOLD


class CreditSignalType(Enum):
    """Credit regime signal types."""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"


class AllocationShift(Enum):
    """Allocation adjustment directions."""
    INCREASE = "increase"
    DECREASE = "decrease"
    HOLD = "hold"


@dataclass(frozen=True)
class AllocationRecommendation:
    """Single asset allocation recommendation."""
    symbol: str
    current_weight: float
    recommended_weight: float
    shift: AllocationShift
    shift_percent: float  # Percentage point change
    rationale: str


@dataclass(frozen=True)
class CreditSpreadSignal:
    """Complete credit spread signal with allocation recommendations."""
    timestamp: str
    signal_type: CreditSignalType
    confidence: float
    spread_absolute: float
    spread_zscore: float
    trend_direction: str
    persistence_days: int
    volatility_regime: str
    is_active: bool
    recommendations: list[AllocationRecommendation]
    summary: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "spread_absolute": self.spread_absolute,
            "spread_zscore": self.spread_zscore,
            "trend_direction": self.trend_direction,
            "persistence_days": self.persistence_days,
            "volatility_regime": self.volatility_regime,
            "is_active": self.is_active,
            "recommendations": [
                {
                    "symbol": r.symbol,
                    "current_weight": r.current_weight,
                    "recommended_weight": r.recommended_weight,
                    "shift": r.shift.value,
                    "shift_percent": r.shift_percent,
                    "rationale": r.rationale
                }
                for r in self.recommendations
            ],
            "summary": self.summary
        }


class CreditSpreadSignalGenerator:
    """Generates credit spread signals and allocation recommendations."""

    # Risk control parameters (from spec)
    VIX_CUTOFF = 35  # Disable signal during extreme volatility
    MIN_PERSISTENCE_DAYS = 5  # Minimum days for signal activation
    MAX_HOLDING_DAYS = 15  # Maximum holding period
    MAX_ALLOCATION_SHIFT = 0.03  # Max 3% shift per signal

    # Base portfolio weights (46/38/16 from champion portfolio)
    BASE_WEIGHTS = {
        "SPY": 0.46,
        "GLD": 0.38,
        "TLT": 0.16
    }

    # Allocation shifts by signal type
    RISK_OFF_SHIFTS = {
        "SPY": -0.03,   # Reduce equity
        "GLD": 0.01,    # Add gold
        "TLT": 0.02     # Add duration
    }

    RISK_ON_SHIFTS = {
        "SPY": 0.02,    # Add equity
        "GLD": 0.00,
        "TLT": -0.02    # Reduce duration
    }

    def __init__(self, vix_level: float = 0.0):
        """
        Initialize signal generator.

        Args:
            vix_level: Current VIX level for risk control (default 0 = unknown)
        """
        self.vix_level = vix_level
        self.fetcher = CreditFetcher()

    def _is_signal_disabled(self) -> tuple[bool, str]:
        """Check if signal should be disabled due to risk controls."""
        if self.vix_level > self.VIX_CUTOFF:
            return True, f"VIX {self.vix_level:.1f} > cutoff {self.VIX_CUTOFF}"
        return False, ""

    def _determine_signal_type(
        self,
        spread: float,
        persistence: int,
        confidence: float
    ) -> tuple[CreditSignalType, bool, str]:
        """Determine signal type and activation status."""
        disabled, reason = self._is_signal_disabled()
        if disabled:
            return CreditSignalType.NEUTRAL, False, reason

        # Check persistence requirement
        if persistence < self.MIN_PERSISTENCE_DAYS:
            return CreditSignalType.NEUTRAL, False, f"Persistence {persistence} < {self.MIN_PERSISTENCE_DAYS}"

        # Determine signal based on spread
        if spread < RISK_OFF_THRESHOLD and confidence > 0.3:
            return CreditSignalType.RISK_OFF, True, f"Spread {spread:+.2%} below threshold {RISK_OFF_THRESHOLD:+.2%}"
        elif spread > RISK_ON_THRESHOLD and confidence > 0.3:
            return CreditSignalType.RISK_ON, True, f"Spread {spread:+.2%} above threshold {RISK_ON_THRESHOLD:+.2%}"
        else:
            return CreditSignalType.NEUTRAL, False, "Spread within neutral band"

    def _generate_recommendations(
        self,
        signal_type: CreditSignalType
    ) -> list[AllocationRecommendation]:
        """Generate allocation recommendations for each asset."""
        recommendations = []

        if signal_type == CreditSignalType.NEUTRAL:
            # No changes for neutral
            for symbol, weight in self.BASE_WEIGHTS.items():
                recommendations.append(AllocationRecommendation(
                    symbol=symbol,
                    current_weight=weight,
                    recommended_weight=weight,
                    shift=AllocationShift.HOLD,
                    shift_percent=0.0,
                    rationale="Neutral regime — maintain base allocation"
                ))
            return recommendations

        # Get shifts based on signal type
        shifts = self.RISK_OFF_SHIFTS if signal_type == CreditSignalType.RISK_OFF else self.RISK_ON_SHIFTS

        for symbol, base_weight in self.BASE_WEIGHTS.items():
            shift = shifts.get(symbol, 0.0)
            new_weight = base_weight + shift

            # Clamp to valid range
            new_weight = max(0.05, min(0.80, new_weight))

            # Determine shift direction
            if shift > 0.001:
                shift_type = AllocationShift.INCREASE
            elif shift < -0.001:
                shift_type = AllocationShift.DECREASE
            else:
                shift_type = AllocationShift.HOLD

            # Generate rationale
            if signal_type == CreditSignalType.RISK_OFF:
                if symbol == "SPY":
                    rationale = "Risk-off regime — reduce equity exposure"
                elif symbol == "TLT":
                    rationale = "Risk-off regime — add duration for flight-to-quality"
                elif symbol == "GLD":
                    rationale = "Risk-off regime — add gold as defensive hedge"
                else:
                    rationale = "Risk-off regime adjustment"
            else:  # RISK_ON
                if symbol == "SPY":
                    rationale = "Risk-on regime — increase equity exposure"
                elif symbol == "TLT":
                    rationale = "Risk-on regime — reduce duration, favor growth"
                elif symbol == "GLD":
                    rationale = "Risk-on regime — gold neutral"
                else:
                    rationale = "Risk-on regime adjustment"

            recommendations.append(AllocationRecommendation(
                symbol=symbol,
                current_weight=base_weight,
                recommended_weight=new_weight,
                shift=shift_type,
                shift_percent=round(shift * 100, 2),
                rationale=rationale
            ))

        return recommendations

    def generate_signal(self, force_refresh: bool = False) -> CreditSpreadSignal:
        """Generate complete credit spread signal."""
        # Fetch credit data
        try:
            data = self.fetcher.fetch_all(force_refresh=force_refresh)
        except Exception as e:
            # Return neutral signal on error
            return CreditSpreadSignal(
                timestamp=datetime.now().isoformat(),
                signal_type=CreditSignalType.NEUTRAL,
                confidence=0.0,
                spread_absolute=0.0,
                spread_zscore=0.0,
                trend_direction="unknown",
                persistence_days=0,
                volatility_regime="unknown",
                is_active=False,
                recommendations=self._generate_recommendations(CreditSignalType.NEUTRAL),
                summary=f"Error fetching data: {e}"
            )

        s = data.spread

        # Determine signal type and activation
        signal_type, is_active, reason = self._determine_signal_type(
            s.spread_absolute,
            s.persistence_days,
            s.confidence
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(signal_type)

        # Build summary
        if signal_type == CreditSignalType.NEUTRAL:
            summary = f"Neutral credit regime. {reason}"
        elif signal_type == CreditSignalType.RISK_OFF:
            summary = (
                f"Risk-off signal: High yield underperforming investment grade "
                f"by {abs(s.spread_absolute):.2%}. Confidence: {s.confidence:.1%}. "
                f"Reduce equity, add duration and gold."
            )
        else:  # RISK_ON
            summary = (
                f"Risk-on signal: High yield outperforming investment grade "
                f"by {s.spread_absolute:.2%}. Confidence: {s.confidence:.1%}. "
                f"Increase equity, reduce duration."
            )

        return CreditSpreadSignal(
            timestamp=data.timestamp,
            signal_type=signal_type,
            confidence=s.confidence,
            spread_absolute=s.spread_absolute,
            spread_zscore=s.spread_zscore,
            trend_direction=s.trend_direction,
            persistence_days=s.persistence_days,
            volatility_regime=s.volatility_regime,
            is_active=is_active,
            recommendations=recommendations,
            summary=summary
        )


def get_credit_signal(vix_level: float = 0.0, force_refresh: bool = False) -> dict:
    """
    Convenience function for external integration.

    Args:
        vix_level: Current VIX for risk control
        force_refresh: Force data refresh from Yahoo Finance

    Returns:
        Signal dictionary for ensemble voter integration
    """
    generator = CreditSpreadSignalGenerator(vix_level=vix_level)
    signal = generator.generate_signal(force_refresh=force_refresh)
    return signal.to_dict()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Credit spread signal generator"
    )
    parser.add_argument(
        "--vix", type=float, default=0.0,
        help="Current VIX level for risk control"
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Force refresh from Yahoo Finance"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--integrator", action="store_true",
        help="Output in signal integrator format"
    )

    args = parser.parse_args()

    generator = CreditSpreadSignalGenerator(vix_level=args.vix)
    signal = generator.generate_signal(force_refresh=args.fetch)

    if args.json:
        import json
        print(json.dumps(signal.to_dict(), indent=2))
    elif args.integrator:
        # Format for signal integrator consumption
        print(f"signal_type={signal.signal_type.value}")
        print(f"confidence={signal.confidence:.4f}")
        print(f"is_active={signal.is_active}")
        print(f"spread={signal.spread_absolute:.4f}")
        print(f"zscore={signal.spread_zscore:.4f}")
        for rec in signal.recommendations:
            if rec.shift_percent != 0:
                print(f"shift:{rec.symbol}={rec.shift_percent:+.2f}%")
    else:
        print(f"\n{'='*70}")
        print(f"Credit Spread Signal — {signal.timestamp[:19]}")
        print(f"{'='*70}")
        print(f"Signal Type:    {signal.signal_type.value.upper()}")
        print(f"Active:         {'YES' if signal.is_active else 'NO'}")
        print(f"Confidence:     {signal.confidence:.1%}")
        print(f"Spread:         {signal.spread_absolute:+.2%} (HYG - LQD)")
        print(f"Z-Score:        {signal.spread_zscore:+.2f}")
        print(f"Trend:          {signal.trend_direction}")
        print(f"Persistence:    {signal.persistence_days} days")
        print(f"Vol Regime:     {signal.volatility_regime}")
        print(f"\nRecommendations:")
        print(f"{'Asset':<8} {'Current':>10} {'Target':>10} {'Change':>10}")
        print("-" * 45)
        for rec in signal.recommendations:
            change_str = f"{rec.shift_percent:+.2f}%" if rec.shift_percent != 0 else "—"
            print(f"{rec.symbol:<8} {rec.current_weight:>9.1%} {rec.recommended_weight:>9.1%} {change_str:>10}")
        print(f"\nSummary: {signal.summary}")
        print(f"{'='*70}")

    return 0 if signal.signal_type != CreditSignalType.NEUTRAL or not signal.is_active else 0


if __name__ == "__main__":
    exit(main())
