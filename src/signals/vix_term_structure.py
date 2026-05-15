"""
VIX Term Structure Signal Generator - v4.50 Implementation
Generates tactical overlay signals based on VIX/VIX3M/VIX6M term structure slope.

Target: +0.03 to +0.04 Sharpe improvement through drawdown avoidance.
Based on research: VIX term structure slope predicts equity returns better than absolute VIX level.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VIXRegime(Enum):
    """VIX term structure regime classification."""
    EXTREME_CONTANGO = "extreme_contango"      # VIX3M/VIX > 1.15 (complacency)
    CONTANGO = "contango"                       # VIX3M/VIX 1.0-1.15 (normal)
    FLAT = "flat"                               # VIX3M/VIX 0.95-1.0 (neutral)
    BACKWARDATION = "backwardation"             # VIX3M/VIX 0.8-0.95 (caution)
    EXTREME_BACKWARDATION = "extreme_backwardation"  # VIX3M/VIX < 0.8 (crisis)


class VIXSignalState(Enum):
    """Signal states for portfolio overlay."""
    RISK_ON = 1         # Increase equity exposure
    NEUTRAL = 0         # Maintain baseline
    RISK_OFF = -1       # Reduce equity, add defensive


@dataclass
class VIXTermStructureSignal:
    """Complete VIX term structure signal with tactical recommendation."""
    timestamp: str
    signal_state: str  # risk_on, neutral, risk_off
    signal_value: float  # -1.0 to +1.0
    
    # Raw inputs
    vix_spot: float
    vix3m: Optional[float]
    vix6m: Optional[float]
    slope_vix3m_vix: float  # VIX3M / VIX ratio
    
    # Regime classification
    regime: str
    regime_strength: float  # 0-1
    
    # Composite components
    slope_signal: float  # -1 to +1
    roll_yield_signal: float
    vix_zscore_signal: float
    curve_shape_signal: float
    
    # Portfolio overlay recommendation
    spy_shift: float  # Percentage point shift (-0.10 to +0.05)
    gld_shift: float
    tlt_shift: float
    
    # Confidence and constraints
    confidence: float  # 0-100%
    is_valid: bool
    reason: str
    
    def to_dict(self) -> dict:
        return asdict(self)


class VIXTermStructureCalculator:
    """
    Calculates VIX term structure slope and generates tactical signals.
    
    Key insight: VIX3M/VIX ratio predicts equity returns better than spot VIX.
    - Backwardation (VIX > VIX3M): Risk-off, reduce equity
    - Contango (VIX < VIX3M): Risk-on or neutral depending on steepness
    """
    
    # Signal thresholds based on research
    EXTREME_CONTANGO_THRESHOLD = 1.15   # Complacency warning
    CONTANGO_THRESHOLD = 1.00           # Normal market
    FLAT_UPPER = 1.00
    FLAT_LOWER = 0.95
    BACKWARDATION_THRESHOLD = 0.80      # Risk-off warning
    
    # VIX level context
    VIX_CHEAP = 16.0
    VIX_FAIR = 20.0
    VIX_EXPENSIVE = 25.0
    
    def __init__(self, history_days: int = 252):
        self.history_days = history_days
        self.vix_history: List[Tuple[str, float]] = []
    
    def add_vix_reading(self, date: str, vix: float):
        """Add VIX reading to history for Z-score calculation."""
        self.vix_history.append((date, vix))
        if len(self.vix_history) > self.history_days:
            self.vix_history.pop(0)
    
    def calculate_slope_signal(self, vix: float, vix3m: float) -> float:
        """
        Map VIX3M/VIX ratio to [-1, +1] signal.
        
        < 0.85: Extreme backwardation (risk-off) -> -1
        0.85-1.0: Backwardation (caution) -> -0.5 to 0
        1.0-1.15: Normal contango -> 0 to +0.5
        > 1.15: Steep contango (complacency) -> +0.5 to +1
        """
        if vix <= 0 or vix3m <= 0:
            return 0.0
        
        slope = vix3m / vix
        
        if slope < 0.85:
            return -1.0
        elif slope < 1.0:
            # Linear interpolation from -1.0 to -0.5
            return -1.0 + (slope - 0.85) / 0.15 * 0.5
        elif slope < 1.15:
            # Linear interpolation from 0 to +0.5
            return (slope - 1.0) / 0.15 * 0.5
        else:
            # Cap at +1.0 for extreme contango
            return min(0.5 + (slope - 1.15) / 0.15 * 0.5, 1.0)
    
    def calculate_roll_yield_signal(self, vix: float, vix3m: float) -> float:
        """
        Roll yield signal: (VIX3M - VIX) / VIX3M normalized to [-1, 1].
        Positive = contango (futures > spot), negative = backwardation.
        """
        if vix3m <= 0:
            return 0.0
        
        roll_yield = (vix3m - vix) / vix3m
        # Normalize: typical range -0.2 to +0.2
        return max(-1.0, min(1.0, roll_yield * 5))
    
    def calculate_vix_zscore_signal(self, vix: float) -> float:
        """
        VIX Z-score relative to 1-year history, mapped to [-1, 1].
        High VIX = negative signal (risk-off), low VIX = positive (risk-on).
        """
        if len(self.vix_history) < 60:  # Need at least 60 days
            return 0.0
        
        vix_values = [v for _, v in self.vix_history]
        mean_vix = np.mean(vix_values)
        std_vix = np.std(vix_values)
        
        if std_vix == 0:
            return 0.0
        
        zscore = (vix - mean_vix) / std_vix
        # Invert: high VIX = risk-off (-1), low VIX = risk-on (+1)
        # Typical Z-score range: -2 to +2
        signal = -max(-1.0, min(1.0, float(zscore) / 2))
        return signal
    
    def calculate_curve_shape_signal(self, vix3m: float, vix6m: Optional[float]) -> float:
        """
        Curve shape using VIX6M/VIX3M if available.
        Steepening = risk building, flattening = normalization.
        """
        if vix6m is None or vix3m <= 0:
            return 0.0
        
        curve_shape = vix6m / vix3m
        # Normalize around 1.0, typical range 0.9 to 1.1
        return max(-1.0, min(1.0, (curve_shape - 1.0) * 10))
    
    def classify_regime(self, slope: float) -> Tuple[VIXRegime, float]:
        """Classify VIX term structure regime and return strength."""
        if slope >= self.EXTREME_CONTANGO_THRESHOLD:
            strength = min(1.0, (slope - 1.15) / 0.15 + 0.5)
            return VIXRegime.EXTREME_CONTANGO, strength
        elif slope >= self.CONTANGO_THRESHOLD:
            strength = (slope - 1.0) / 0.15
            return VIXRegime.CONTANGO, strength
        elif slope >= self.FLAT_LOWER:
            strength = (1.0 - slope) / 0.05
            return VIXRegime.FLAT, strength
        elif slope >= self.BACKWARDATION_THRESHOLD:
            strength = (0.95 - slope) / 0.15
            return VIXRegime.BACKWARDATION, strength
        else:
            strength = min(1.0, (0.80 - slope) / 0.10 + 0.5)
            return VIXRegime.EXTREME_BACKWARDATION, strength
    
    def calculate_composite_signal(
        self,
        vix: float,
        vix3m: Optional[float],
        vix6m: Optional[float],
        date: str
    ) -> Dict:
        """
        Calculate composite signal using weighted components.
        
        Weights based on research:
        - Slope: 40% (primary predictor)
        - Roll yield: 25% (carry signal)
        - VIX Z-score: 20% (absolute vol context)
        - Curve shape: 15% (confirmation)
        """
        if vix3m is None or vix3m <= 0:
            logger.warning(f"[{date}] VIX3M unavailable, using VIX spot proxy")
            # Use VIX level as fallback
            if vix < self.VIX_CHEAP:
                slope_signal = 0.5  # Complacent
            elif vix < self.VIX_FAIR:
                slope_signal = 0.0  # Normal
            elif vix < self.VIX_EXPENSIVE:
                slope_signal = -0.3  # Elevated
            else:
                slope_signal = -0.8  # Stress
            vix3m = vix * (1.1 if slope_signal > 0 else 0.9)
        
        # Calculate individual signals
        slope_signal = self.calculate_slope_signal(vix, vix3m)
        roll_signal = self.calculate_roll_yield_signal(vix, vix3m)
        zscore_signal = self.calculate_vix_zscore_signal(vix)
        curve_signal = self.calculate_curve_shape_signal(vix3m, vix6m)
        
        # Weighted composite
        composite = (
            0.40 * slope_signal +
            0.25 * roll_signal +
            0.20 * zscore_signal +
            0.15 * curve_signal
        )
        
        # Bound to [-1, 1]
        composite = max(-1.0, min(1.0, composite))
        
        return {
            "composite": composite,
            "slope_signal": slope_signal,
            "roll_yield_signal": roll_signal,
            "vix_zscore_signal": zscore_signal,
            "curve_shape_signal": curve_signal,
            "slope": vix3m / vix if vix > 0 else 1.0
        }
    
    def get_allocation_shifts(self, signal: float) -> Dict[str, float]:
        """
        Map composite signal to allocation shifts.
        
        Signal ranges and shifts:
        +0.7 to +1.0 (Complacent): SPY +5%, GLD -3%, TLT -2%
        +0.3 to +0.7 (Normal): No change
        -0.3 to +0.3 (Neutral): No change
        -0.7 to -0.3 (Caution): SPY -5%, GLD +3%, TLT +2%
        -1.0 to -0.7 (Risk-Off): SPY -10%, GLD +5%, TLT +5%
        """
        if signal >= 0.7:
            return {"spy": 0.05, "gld": -0.03, "tlt": -0.02}
        elif signal >= 0.3:
            return {"spy": 0.02, "gld": -0.01, "tlt": -0.01}
        elif signal >= -0.3:
            return {"spy": 0.0, "gld": 0.0, "tlt": 0.0}
        elif signal >= -0.7:
            return {"spy": -0.05, "gld": 0.03, "tlt": 0.02}
        else:
            return {"spy": -0.10, "gld": 0.05, "tlt": 0.05}


class VIXTermStructureSignalGenerator:
    """
    Main signal generator for VIX term structure tactical overlay.
    
    Fetches data, calculates signals, and generates portfolio recommendations.
    """
    
    DATA_DIR = Path(__file__).parent.parent.parent / 'data'
    VIX_DATA_PATH = DATA_DIR / 'vix_term_structure.json'
    OUTPUT_PATH = DATA_DIR / 'signals' / 'vix_term_structure_signal.json'
    
    def __init__(self):
        self.calculator = VIXTermStructureCalculator()
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """Ensure output directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / 'signals').mkdir(parents=True, exist_ok=True)
    
    def load_vix_data(self) -> Dict:
        """Load VIX term structure data from storage."""
        if not self.VIX_DATA_PATH.exists():
            logger.warning(f"VIX data file not found: {self.VIX_DATA_PATH}")
            return {}
        
        try:
            with open(self.VIX_DATA_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading VIX data: {e}")
            return {}
    
    def fetch_current_vix(self) -> Optional[Dict]:
        """
        Fetch current VIX levels from data sources.
        
        For production: Implement real-time API fetch from CBOE
        For now: Use latest from stored data
        """
        data = self.load_vix_data()
        if not data:
            return None
        
        # Get latest date
        latest_date = max(data.keys())
        return data[latest_date]
    
    def generate_signal(self, date: Optional[str] = None) -> VIXTermStructureSignal:
        """Generate complete VIX term structure signal."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Load historical data for context
        historical_data = self.load_vix_data()
        
        # Build VIX history for Z-score
        for d in sorted(historical_data.keys())[-252:]:
            vix = historical_data[d].get('vix_spot', 0)
            if vix > 0:
                self.calculator.add_vix_reading(d, vix)
        
        # Get current readings
        current = historical_data.get(date)
        
        if current is None:
            # Try to fetch current
            current = self.fetch_current_vix()
            if current is None:
                return VIXTermStructureSignal(
                    timestamp=datetime.now().isoformat(),
                    signal_state="neutral",
                    signal_value=0.0,
                    vix_spot=0.0,
                    vix3m=None,
                    vix6m=None,
                    slope_vix3m_vix=1.0,
                    regime="unknown",
                    regime_strength=0.0,
                    slope_signal=0.0,
                    roll_yield_signal=0.0,
                    vix_zscore_signal=0.0,
                    curve_shape_signal=0.0,
                    spy_shift=0.0,
                    gld_shift=0.0,
                    tlt_shift=0.0,
                    confidence=0.0,
                    is_valid=False,
                    reason="No VIX data available"
                )
        
        vix = current.get('vix_spot', 0)
        vix3m = current.get('front_month')  # Using front month as proxy for VIX3M
        vix6m = current.get('third_month')  # Third month as VIX6M proxy
        
        # Calculate composite signal
        components = self.calculator.calculate_composite_signal(
            vix=vix,
            vix3m=vix3m,
            vix6m=vix6m,
            date=date
        )
        
        # Classify regime
        regime, strength = self.calculator.classify_regime(components['slope'])
        
        # Map to signal state
        composite = components['composite']
        if composite > 0.5:
            signal_state = VIXSignalState.RISK_ON
        elif composite < -0.5:
            signal_state = VIXSignalState.RISK_OFF
        else:
            signal_state = VIXSignalState.NEUTRAL
        
        # Get allocation shifts
        shifts = self.calculator.get_allocation_shifts(composite)
        
        # Calculate confidence based on data quality
        confidence = 50.0  # Base confidence
        if vix3m is not None:
            confidence += 30.0
        if vix6m is not None:
            confidence += 10.0
        if len(self.calculator.vix_history) >= 60:
            confidence += 10.0
        
        return VIXTermStructureSignal(
            timestamp=datetime.now().isoformat(),
            signal_state=signal_state.name,
            signal_value=composite,
            vix_spot=vix,
            vix3m=vix3m,
            vix6m=vix6m,
            slope_vix3m_vix=components['slope'],
            regime=regime.value,
            regime_strength=strength,
            slope_signal=components['slope_signal'],
            roll_yield_signal=components['roll_yield_signal'],
            vix_zscore_signal=components['vix_zscore_signal'],
            curve_shape_signal=components['curve_shape_signal'],
            spy_shift=shifts['spy'],
            gld_shift=shifts['gld'],
            tlt_shift=shifts['tlt'],
            confidence=confidence,
            is_valid=True,
            reason=f"VIX={vix:.2f}, Slope={components['slope']:.3f}, Regime={regime.value}"
        )
    
    def save_signal(self, signal: VIXTermStructureSignal):
        """Save signal to disk."""
        self.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.OUTPUT_PATH, 'w') as f:
                json.dump(signal.to_dict(), f, indent=2)
            logger.info(f"Saved VIX signal to {self.OUTPUT_PATH}")
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
    
    def get_signal_history(self, days: int = 30) -> List[VIXTermStructureSignal]:
        """Generate signals for historical dates."""
        historical_data = self.load_vix_data()
        signals = []
        
        dates = sorted(historical_data.keys())[-days:]
        
        for date in dates:
            signal = self.generate_signal(date)
            if signal.is_valid:
                signals.append(signal)
        
        return signals


def main():
    """CLI entry point for signal generation."""
    generator = VIXTermStructureSignalGenerator()
    signal = generator.generate_signal()
    
    print("=" * 60)
    print("VIX TERM STRUCTURE SIGNAL GENERATOR v4.50")
    print("=" * 60)
    print(f"Timestamp: {signal.timestamp}")
    print(f"Signal State: {signal.signal_state}")
    print(f"Signal Value: {signal.signal_value:.3f}")
    print()
    print(f"VIX Spot: {signal.vix_spot:.2f}")
    print(f"VIX3M: {signal.vix3m}")
    print(f"VIX6M: {signal.vix6m}")
    print(f"Slope (VIX3M/VIX): {signal.slope_vix3m_vix:.3f}")
    print()
    print(f"Regime: {signal.regime}")
    print(f"Regime Strength: {signal.regime_strength:.2f}")
    print()
    print("Component Signals:")
    print(f"  Slope Signal: {signal.slope_signal:.3f}")
    print(f"  Roll Yield: {signal.roll_yield_signal:.3f}")
    print(f"  VIX Z-Score: {signal.vix_zscore_signal:.3f}")
    print(f"  Curve Shape: {signal.curve_shape_signal:.3f}")
    print()
    print("Portfolio Shifts:")
    print(f"  SPY: {signal.spy_shift:+.1%}")
    print(f"  GLD: {signal.gld_shift:+.1%}")
    print(f"  TLT: {signal.tlt_shift:+.1%}")
    print()
    print(f"Confidence: {signal.confidence:.0f}%")
    print(f"Valid: {signal.is_valid}")
    print(f"Reason: {signal.reason}")
    print("=" * 60)
    
    # Save signal
    generator.save_signal(signal)
    
    return signal


if __name__ == '__main__':
    main()
