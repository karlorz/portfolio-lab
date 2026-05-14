"""FX Currency Carry signal generator for portfolio allocation.

Generates tactical allocation signals based on USD strength momentum
from UUP/UDN ETFs. Integrates with ensemble voter for macro regime detection.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
from pathlib import Path

from src.data.fx_fetcher import fetch_fx_metrics, load_latest_metrics, FXMetrics

# Data directory (consistent with other modules)
DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()


class FXSignalType(Enum):
    """Types of FX carry signals."""
    USD_STRENGTH = "usd_strength"
    USD_WEAKNESS = "usd_weakness"
    NEUTRAL = "neutral"


# Signal thresholds
USD_BULL_THRESHOLD = 2.0  # UUP up >2%
USD_BEAR_THRESHOLD = 2.0  # UDN up >2%
CONFLICT_THRESHOLD = -1.0  # Both moving opposite directions

# Risk limits
MAX_DXY_VOLATILITY = 15.0  # Disable signal above this
MIN_PERSISTENCE_DAYS = 5
MAX_HOLDING_DAYS = 15
MAX_ALLOCATION_SHIFT = 2.0  # Percentage points


@dataclass
class FXCarrySignal:
    """FX carry signal with allocation recommendations."""
    timestamp: str
    signal_type: str  # usd_strength, usd_weakness, neutral
    confidence: float  # 0.0 to 1.0
    
    # Allocation shifts (percentage points, can be negative)
    spy_shift: float
    efa_shift: float  # International equity
    vxus_shift: float  # Total international
    
    # Signal metadata
    uup_return_30d: float
    udn_return_30d: float
    usd_strength_score: float
    
    # Risk controls
    is_active: bool
    reason_inactive: Optional[str] = None
    
    # Signal history
    days_active: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "signal_type": self.signal_type,
            "confidence": self.confidence,
            "allocation_shifts": {
                "spy": self.spy_shift,
                "efa": self.efa_shift,
                "vxus": self.vxus_shift,
            },
            "uup_return_30d": self.uup_return_30d,
            "udn_return_30d": self.udn_return_30d,
            "usd_strength_score": self.usd_strength_score,
            "is_active": self.is_active,
            "reason_inactive": self.reason_inactive,
            "days_active": self.days_active,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _get_signal_persistence() -> int:
    """Get number of days current signal type has persisted."""
    # This would ideally read from history; simplified here
    # In production, this would check signal history file
    return 0  # Placeholder - would track actual persistence


def generate_signal(metrics: Optional[FXMetrics] = None) -> FXCarrySignal:
    """Generate FX carry signal from metrics.
    
    Returns:
        FXCarrySignal with allocation recommendations
    """
    if metrics is None:
        metrics = fetch_fx_metrics()
    
    timestamp = datetime.now().isoformat()
    
    # Default neutral signal
    signal_type = FXSignalType.NEUTRAL.value
    confidence = 0.0
    spy_shift = 0.0
    efa_shift = 0.0
    vxus_shift = 0.0
    is_active = True
    reason_inactive = None
    
    # Check for conflicting signals (both UUP/UDN positive)
    if metrics.uup_return_30d > 0 and metrics.udn_return_30d > 0:
        is_active = False
        reason_inactive = "momentum_conflict: both UUP/UDN positive"
    
    # Check volatility cutoff
    elif metrics.volatility_regime == "high":
        is_active = False
        reason_inactive = f"high_volatility: {metrics.volatility_regime}"
    
    else:
        # Determine signal type
        if metrics.uup_return_30d > USD_BULL_THRESHOLD and metrics.udn_return_30d < CONFLICT_THRESHOLD:
            signal_type = FXSignalType.USD_STRENGTH.value
            confidence = min(abs(metrics.uup_return_30d) / 4.0, 1.0)
            
            # USD strength: Reduce international, add to SPY
            shift = min(MAX_ALLOCATION_SHIFT * confidence, MAX_ALLOCATION_SHIFT)
            spy_shift = shift
            efa_shift = -shift / 2
            vxus_shift = -shift / 2
            
        elif metrics.udn_return_30d > USD_BEAR_THRESHOLD and metrics.uup_return_30d < CONFLICT_THRESHOLD:
            signal_type = FXSignalType.USD_WEAKNESS.value
            confidence = min(abs(metrics.udn_return_30d) / 4.0, 1.0)
            
            # USD weakness: Add international, reduce SPY
            shift = min(MAX_ALLOCATION_SHIFT * confidence, MAX_ALLOCATION_SHIFT)
            spy_shift = -shift
            efa_shift = shift / 2
            vxus_shift = shift / 2
    
    return FXCarrySignal(
        timestamp=timestamp,
        signal_type=signal_type,
        confidence=confidence,
        spy_shift=spy_shift,
        efa_shift=efa_shift,
        vxus_shift=vxus_shift,
        uup_return_30d=metrics.uup_return_30d,
        udn_return_30d=metrics.udn_return_30d,
        usd_strength_score=metrics.usd_strength_score,
        is_active=is_active,
        reason_inactive=reason_inactive,
        days_active=_get_signal_persistence()
    )


def get_current_signal() -> Dict[str, Any]:
    """Get current signal as dictionary for API/ensemble integration."""
    signal = generate_signal()
    return signal.to_dict()


def get_allocation_impact() -> Dict[str, Any]:
    """Get allocation impact summary for dashboard."""
    signal = generate_signal()
    
    impact = {
        "timestamp": signal.timestamp,
        "fx_signal_active": signal.is_active,
        "signal_type": signal.signal_type,
        "confidence": f"{signal.confidence:.1%}",
        "allocations": {
            "spy": f"{signal.spy_shift:+.1f}%" if signal.spy_shift != 0 else "unchanged",
            "efa": f"{signal.efa_shift:+.1f}%" if signal.efa_shift != 0 else "unchanged",
            "vxus": f"{signal.vxus_shift:+.1f}%" if signal.vxus_shift != 0 else "unchanged",
        },
        "total_shift": abs(signal.spy_shift) + abs(signal.efa_shift) + abs(signal.vxus_shift),
    }
    
    if not signal.is_active:
        impact["status"] = "INACTIVE"
        impact["reason"] = signal.reason_inactive
    else:
        impact["status"] = "ACTIVE"
    
    return impact


def save_signal(signal: FXCarrySignal, filepath: Optional[Path] = None) -> Path:
    """Save signal to JSON file."""
    if filepath is None:
        filepath = DATA_DIR / "fx_signal.json"
    
    if filepath is not None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(signal.to_json())
    
    return filepath


def load_latest_signal(filepath: Optional[Path] = None) -> Optional[FXCarrySignal]:
    """Load latest signal from file."""
    actual_filepath = filepath if filepath is not None else DATA_DIR / "fx_signal.json"
    
    if not actual_filepath.exists():
        return None
    
    with open(actual_filepath) as f:
        data = json.load(f)
    
    return FXCarrySignal(
        timestamp=data["timestamp"],
        signal_type=data["signal_type"],
        confidence=data["confidence"],
        spy_shift=data["allocation_shifts"]["spy"],
        efa_shift=data["allocation_shifts"]["efa"],
        vxus_shift=data["allocation_shifts"]["vxus"],
        uup_return_30d=data["uup_return_30d"],
        udn_return_30d=data["udn_return_30d"],
        usd_strength_score=data["usd_strength_score"],
        is_active=data["is_active"],
        reason_inactive=data.get("reason_inactive"),
        days_active=data.get("days_active", 0),
    )


def get_ensemble_input() -> Dict[str, Any]:
    """Get signal formatted for ensemble voter integration.
    
    Returns dict with:
        - signal: bullish/bearish/neutral
        - confidence: 0.0-1.0
        - weight: recommended ensemble weight (2%)
        - allocation_delta: dict of symbol->shift
    """
    signal = generate_signal()
    
    signal_map = {
        "usd_strength": "bullish",
        "usd_weakness": "bearish",
        "neutral": "neutral"
    }
    
    return {
        "source": "fx_carry",
        "signal": signal_map.get(signal.signal_type, "neutral"),
        "confidence": signal.confidence,
        "weight": 0.02,  # 2% ensemble weight
        "allocation_delta": {
            "SPY": signal.spy_shift,
            "EFA": signal.efa_shift,
            "VXUS": signal.vxus_shift,
        },
        "is_active": signal.is_active,
        "timestamp": signal.timestamp,
    }


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FX Currency Carry signal generator")
    parser.add_argument("--signal", action="store_true", help="Generate and print signal")
    parser.add_argument("--save", action="store_true", help="Save signal to file")
    parser.add_argument("--impact", action="store_true", help="Print allocation impact")
    parser.add_argument("--ensemble", action="store_true", help="Print ensemble input format")
    
    args = parser.parse_args()
    
    if args.save:
        signal = generate_signal()
        path = save_signal(signal)
        print(f"Signal saved to {path}")
    
    if args.impact:
        impact = get_allocation_impact()
        print(json.dumps(impact, indent=2))
    
    if args.ensemble:
        ensemble_input = get_ensemble_input()
        print(json.dumps(ensemble_input, indent=2))
    
    if args.signal or not any([args.save, args.impact, args.ensemble]):
        signal = generate_signal()
        print(signal.to_json())


if __name__ == "__main__":
    main()
