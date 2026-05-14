"""
FX Currency Carry Signal Generator
Generates allocation signals based on USD strength momentum.

Part of v3.15: FX Currency Carry Overlay
"""

import os
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from src.data.fx_fetcher import FXFetcher, FXMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FXCarrySignal:
    """Currency carry signal with allocation recommendations."""
    signal_type: str  # usd_strength, usd_weakness, neutral
    confidence: float  # 0.0 to 1.0
    regime: str  # positive/negative/neutral
    direction: str  # bullish/bearish/neutral
    reason: str
    
    # Allocation shifts (percentage points)
    spy_shift: float
    efa_shift: float  # Developed markets
    vxus_shift: float  # Total international
    
    # Risk control flags
    is_valid: bool
    persistence_days: int = 0
    max_hold_days: int = 15
    
    timestamp: str = ""


class FXCarrySignalGenerator:
    """Generates FX carry signals with risk controls."""
    
    # Thresholds (based on 30-day momentum %)
    USD_BULL_THRESHOLD = 2.0
    USD_BEAR_THRESHOLD = -2.0
    DXY_VOL_CUTOFF = 0.15  # 15% annualized
    MAX_SHIFT = 2.0  # Maximum 2% allocation shift
    MIN_PERSISTENCE_DAYS = 5
    MAX_HOLD_DAYS = 15
    
    def __init__(self, signal_history_path: Optional[Path] = None):
        self.fetcher = FXFetcher()
        self.signal_history_path = signal_history_path or Path("data/fx_signal_history.json")
    
    def _load_signal_history(self) -> Dict[str, Any]:
        """Load signal persistence history."""
        if self.signal_history_path.exists():
            with open(self.signal_history_path, 'r') as f:
                return json.load(f)
        return {"signals": [], "last_signal_type": "neutral", "days_in_regime": 0}
    
    def _save_signal_history(self, history: Dict[str, Any]):
        """Save signal persistence history."""
        self.signal_history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.signal_history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _update_persistence(self, current_signal: str) -> int:
        """Update and return days in current signal regime."""
        history = self._load_signal_history()
        
        if current_signal == history["last_signal_type"]:
            history["days_in_regime"] += 1
        else:
            history["days_in_regime"] = 1
            history["last_signal_type"] = current_signal
        
        history["signals"].append({
            "signal": current_signal,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 30 days
        cutoff = datetime.now().timestamp() - (30 * 24 * 3600)
        history["signals"] = [
            s for s in history["signals"]
            if datetime.fromisoformat(s["timestamp"]).timestamp() > cutoff
        ]
        
        self._save_signal_history(history)
        return history["days_in_regime"]
    
    def _check_momentum_conflict(self, metrics: FXMetrics) -> bool:
        """Check if both UUP and UDN are positive (confusing signal)."""
        return metrics.uup_return_30d > 0 and metrics.udn_return_30d > 0
    
    def _calculate_confidence(self, metrics: FXMetrics, signal_type: str) -> float:
        """Calculate signal confidence based on momentum strength."""
        if signal_type == "neutral":
            return 0.0
        
        # Use stronger of the two returns normalized to 4%
        if signal_type == "usd_strength":
            raw_confidence = abs(metrics.uup_return_30d) / 4.0
        else:  # usd_weakness
            raw_confidence = abs(metrics.udn_return_30d) / 4.0
        
        return min(raw_confidence, 1.0)
    
    def _calculate_allocation_shifts(self, signal_type: str, confidence: float) -> Tuple[float, float, float]:
        """Calculate allocation shifts for each asset class."""
        if signal_type == "neutral":
            return 0.0, 0.0, 0.0
        
        # Scale shift by confidence, max 2%
        base_shift = self.MAX_SHIFT * confidence
        
        if signal_type == "usd_strength":
            # Reduce international, add to SPY
            return base_shift, -base_shift, -base_shift
        else:  # usd_weakness
            # Add international, reduce SPY
            return -base_shift, base_shift, base_shift
    
    def generate_signal(self) -> FXCarrySignal:
        """Generate FX carry signal with full risk controls."""
        try:
            metrics = self.fetcher.fetch_metrics()
        except Exception as e:
            logger.error(f"Failed to fetch FX metrics: {e}")
            return FXCarrySignal(
                signal_type="neutral",
                confidence=0.0,
                regime="neutral",
                direction="neutral",
                reason="data_error",
                spy_shift=0.0,
                efa_shift=0.0,
                vxus_shift=0.0,
                is_valid=False,
                timestamp=datetime.now().isoformat()
            )
        
        # Risk controls check - don't update persistence until we know the actual signal
        vol_check = metrics.volatility_regime != "high"
        
        if not vol_check:
            persistence = self._update_persistence("neutral")
            logger.warning(f"High volatility detected ({metrics.volatility_regime}), returning neutral")
            return FXCarrySignal(
                signal_type="neutral",
                confidence=0.0,
                regime=metrics.carry_regime,
                direction=metrics.momentum_direction,
                reason="high_volatility",
                spy_shift=0.0,
                efa_shift=0.0,
                vxus_shift=0.0,
                is_valid=False,
                persistence_days=persistence,
                timestamp=metrics.timestamp
            )
        
        if self._check_momentum_conflict(metrics):
            persistence = self._update_persistence("neutral")
            logger.warning("Momentum conflict detected (both UUP/UDN positive), returning neutral")
            return FXCarrySignal(
                signal_type="neutral",
                confidence=0.0,
                regime=metrics.carry_regime,
                direction="neutral",
                reason="momentum_conflict",
                spy_shift=0.0,
                efa_shift=0.0,
                vxus_shift=0.0,
                is_valid=False,
                persistence_days=persistence,
                timestamp=metrics.timestamp
            )
        
        # Determine signal type
        if metrics.momentum_direction == "bullish":
            signal_type = "usd_strength"
        elif metrics.momentum_direction == "bearish":
            signal_type = "usd_weakness"
        else:
            signal_type = "neutral"
        
        # Update persistence for actual signal
        persistence = self._update_persistence(signal_type)
        
        # Check minimum persistence
        if signal_type != "neutral" and persistence < self.MIN_PERSISTENCE_DAYS:
            logger.info(f"Signal {signal_type} only {persistence} days, waiting for {self.MIN_PERSISTENCE_DAYS}")
            return FXCarrySignal(
                signal_type="neutral",
                confidence=0.0,
                regime=metrics.carry_regime,
                direction=metrics.momentum_direction,
                reason="insufficient_persistence",
                spy_shift=0.0,
                efa_shift=0.0,
                vxus_shift=0.0,
                is_valid=False,
                persistence_days=persistence,
                timestamp=metrics.timestamp
            )
        
        # Calculate confidence and shifts
        confidence = self._calculate_confidence(metrics, signal_type)
        spy_shift, efa_shift, vxus_shift = self._calculate_allocation_shifts(signal_type, confidence)
        
        return FXCarrySignal(
            signal_type=signal_type,
            confidence=round(confidence, 2),
            regime=metrics.carry_regime,
            direction=metrics.momentum_direction,
            reason="momentum_aligned",
            spy_shift=round(spy_shift, 1),
            efa_shift=round(efa_shift, 1),
            vxus_shift=round(vxus_shift, 1),
            is_valid=True,
            persistence_days=persistence,
            max_hold_days=self.MAX_HOLD_DAYS,
            timestamp=metrics.timestamp
        )
    
    def get_ensemble_input(self) -> Dict[str, Any]:
        """Get signal formatted for ensemble voter integration."""
        signal = self.generate_signal()
        
        return {
            "source": "fx_carry",
            "signal": signal.signal_type,
            "confidence": signal.confidence,
            "is_valid": signal.is_valid,
            "allocation_shifts": {
                "SPY": signal.spy_shift,
                "EFA": signal.efa_shift,
                "VXUS": signal.vxus_shift
            },
            "metadata": {
                "regime": signal.regime,
                "direction": signal.direction,
                "reason": signal.reason,
                "persistence_days": signal.persistence_days
            },
            "timestamp": signal.timestamp
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="FX Currency Carry Signal Generator")
    parser.add_argument("--signal", action="store_true", help="Generate current signal")
    parser.add_argument("--ensemble", action="store_true", help="Output ensemble format")
    parser.add_argument("--metrics", action="store_true", help="Show underlying metrics")
    
    args = parser.parse_args()
    
    generator = FXCarrySignalGenerator()
    
    if args.ensemble:
        result = generator.get_ensemble_input()
        print(json.dumps(result, indent=2))
    elif args.metrics:
        metrics = generator.fetcher.fetch_metrics()
        print(json.dumps({
            "uup_return_30d": metrics.uup_return_30d,
            "udn_return_30d": metrics.udn_return_30d,
            "usd_strength_score": metrics.usd_strength_score,
            "carry_regime": metrics.carry_regime,
            "momentum_direction": metrics.momentum_direction,
            "volatility_regime": metrics.volatility_regime,
            "data_freshness_hours": metrics.data_freshness_hours
        }, indent=2))
    else:
        signal = generator.generate_signal()
        print(json.dumps({
            "signal_type": signal.signal_type,
            "confidence": signal.confidence,
            "regime": signal.regime,
            "direction": signal.direction,
            "reason": signal.reason,
            "spy_shift": signal.spy_shift,
            "efa_shift": signal.efa_shift,
            "vxus_shift": signal.vxus_shift,
            "is_valid": signal.is_valid,
            "persistence_days": signal.persistence_days,
            "timestamp": signal.timestamp
        }, indent=2))


if __name__ == "__main__":
    main()
