"""
VIX Term Structure Tactical Overlay - v4.50 Phase 3 Implementation
Regime-based equity exposure timing through VIX term structure slope.

Target: +0.03 to +0.04 Sharpe improvement through drawdown avoidance.
Integrates with SmartRebalanceGate for VPIN-aware execution.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

from src.signals.vix_term_structure import (
    VIXTermStructureCalculator,
    VIXTermStructureSignal,
    VIXRegime,
    VIXSignalState
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VIXOverlayStatus(Enum):
    """Current overlay execution status."""
    ACTIVE = "active"           # Overlay actively adjusting
    HOLDING = "holding"         # Min holding period enforced
    FROZEN = "frozen"           # VPIN toxicity override
    DISABLED = "disabled"       # Overlay disabled (e.g., extreme VIX spike)


@dataclass
class AllocationShift:
    """Recommended allocation shift for a single asset."""
    symbol: str
    shift_pct: float  # Percentage point shift (e.g., +5.0 = +5%)
    max_daily_shift: float  # Maximum single-day shift
    rationale: str


@dataclass
class VIXOverlayDecision:
    """Complete overlay decision with all constraints applied."""
    timestamp: str
    status: str  # active, holding, frozen, disabled
    
    # Raw signal
    signal_value: float
    regime: str
    
    # Target shifts (before constraints)
    target_spy_shift: float
    target_gld_shift: float
    target_tlt_shift: float
    
    # Allowed shifts (after constraints)
    allowed_spy_shift: float
    allowed_gld_shift: float
    allowed_tlt_shift: float
    
    # Constraint reasons
    constraints_applied: List[str]
    
    # Execution guidance
    urgency: str  # immediate, gradual, deferred
    vpin_override: bool
    vpin_threshold: float
    
    def to_dict(self) -> dict:
        return asdict(self)


class VIXTermStructureOverlay:
    """
    Tactical overlay using VIX term structure for equity exposure timing.
    
    Allocation shifts based on VIX3M/VIX slope:
    - Extreme contango (complacency): Reduce equity, add defensive
    - Backwardation (fear): Reduce equity, add defensive
    - Normal contango: Maintain baseline
    
    Constraints:
    - Max 5% daily shift
    - Min 5-day holding period
    - VPIN override freezes execution
    - Extreme VIX spike (+50% day) disables overlay
    """
    
    # Allocation shift table by signal value
    ALLOCATION_SHIFTS = {
        # Signal range: (SPY shift, GLD shift, TLT shift)
        (0.7, 1.0): (0.05, -0.03, -0.02),   # Complacent - risk building
        (0.3, 0.7): (0.0, 0.0, 0.0),         # Normal - baseline
        (-0.3, 0.3): (0.0, 0.0, 0.0),        # Neutral - no signal
        (-0.7, -0.3): (-0.05, 0.03, 0.02),   # Caution - defensive
        (-1.0, -0.7): (-0.10, 0.05, 0.05),   # Risk-off - crisis mode
    }
    
    # Constraints
    MAX_DAILY_SHIFT = 0.05  # 5% max single-day shift
    MIN_HOLDING_DAYS = 5
    VPIN_FREEZE_THRESHOLD = 0.70
    VIX_SPIKE_DISABLE = 0.50  # +50% single-day VIX spike
    
    # Baseline allocation (SPY/GLD/TLT 46/38/16)
    BASELINE = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    
    def __init__(
        self,
        max_daily_shift: float = 0.05,
        min_holding_days: int = 5,
        vpin_threshold: float = 0.70,
        state_file: Optional[Path] = None
    ):
        self.max_daily_shift = max_daily_shift
        self.min_holding_days = min_holding_days
        self.vpin_threshold = vpin_threshold
        self.state_file = state_file or Path("data/vix_overlay_state.json")
        
        self.calculator = VIXTermStructureCalculator()
        
        # State tracking
        self.current_allocation = dict(self.BASELINE)
        self.last_shift_date: Optional[datetime] = None
        self.shift_history: List[Dict] = []
        self.consecutive_holding_days = 0
        self.disabled_until: Optional[datetime] = None
        
        self._load_state()
    
    def _load_state(self):
        """Load overlay state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    self.current_allocation = state.get("allocation", dict(self.BASELINE))
                    last_shift = state.get("last_shift_date")
                    if last_shift:
                        self.last_shift_date = datetime.fromisoformat(last_shift)
                    self.shift_history = state.get("shift_history", [])
                    disabled = state.get("disabled_until")
                    if disabled:
                        self.disabled_until = datetime.fromisoformat(disabled)
                logger.info(f"Loaded VIX overlay state: {self.current_allocation}")
            except Exception as e:
                logger.warning(f"Could not load overlay state: {e}")
    
    def _save_state(self):
        """Save overlay state to file."""
        state = {
            "allocation": self.current_allocation,
            "last_shift_date": self.last_shift_date.isoformat() if self.last_shift_date else None,
            "shift_history": self.shift_history[-100:],  # Keep last 100
            "disabled_until": self.disabled_until.isoformat() if self.disabled_until else None,
            "updated_at": datetime.now().isoformat()
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _check_vix_spike(self, current_vix: float, vix_history: List[float]) -> bool:
        """Check if VIX spiked >50% in one day."""
        if not vix_history or len(vix_history) < 2:
            return False
        
        prev_vix = vix_history[-2] if len(vix_history) >= 2 else vix_history[-1]
        if prev_vix <= 0:
            return False
        
        spike_pct = (current_vix - prev_vix) / prev_vix
        return spike_pct >= self.VIX_SPIKE_DISABLE
    
    def _get_allocation_shifts(self, signal_value: float) -> Tuple[float, float, float]:
        """Get allocation shifts based on signal value."""
        for (low, high), (spy, gld, tlt) in self.ALLOCATION_SHIFTS.items():
            if low <= signal_value <= high:
                return spy, gld, tlt
        
        # Default to neutral if outside ranges
        if signal_value > 1.0:
            return self.ALLOCATION_SHIFTS[(0.7, 1.0)]
        else:
            return self.ALLOCATION_SHIFTS[(-1.0, -0.7)]
    
    def _apply_constraints(
        self,
        target_shifts: Dict[str, float],
        days_since_last_shift: int,
        vpin_toxicity: float,
        vix_spike_detected: bool
    ) -> Tuple[Dict[str, float], List[str], VIXOverlayStatus]:
        """
        Apply all constraints to target shifts.
        
        Returns:
            (allowed_shifts, constraint_reasons, status)
        """
        constraints = []
        
        # Check disabled status
        if self.disabled_until and datetime.now() < self.disabled_until:
            constraints.append(f"Overlay disabled until {self.disabled_until}")
            return {k: 0.0 for k in target_shifts}, constraints, VIXOverlayStatus.DISABLED
        
        # Check VIX spike
        if vix_spike_detected:
            self.disabled_until = datetime.now() + timedelta(days=1)
            constraints.append(f"VIX spike detected: overlay disabled 24h")
            self._save_state()
            return {k: 0.0 for k in target_shifts}, constraints, VIXOverlayStatus.DISABLED
        
        # Check VPIN
        if vpin_toxicity > self.vpin_threshold:
            constraints.append(f"VPIN {vpin_toxicity:.2f} > threshold {self.vpin_threshold}: execution frozen")
            return {k: 0.0 for k in target_shifts}, constraints, VIXOverlayStatus.FROZEN
        
        # Check holding period
        if days_since_last_shift < self.min_holding_days and self.last_shift_date:
            constraints.append(f"Min holding period: {days_since_last_shift}/{self.min_holding_days} days")
            return {k: 0.0 for k in target_shifts}, constraints, VIXOverlayStatus.HOLDING
        
        # Apply max daily shift constraint
        allowed_shifts = {}
        for symbol, target in target_shifts.items():
            # Limit to max daily shift
            if abs(target) > self.max_daily_shift:
                sign = 1 if target > 0 else -1
                allowed = sign * self.max_daily_shift
                constraints.append(f"{symbol}: capped from {target:+.1%} to {allowed:+.1%}")
            else:
                allowed = target
            allowed_shifts[symbol] = allowed
        
        return allowed_shifts, constraints, VIXOverlayStatus.ACTIVE
    
    def calculate_overlay(
        self,
        signal: VIXTermStructureSignal,
        vix_history: Optional[List[float]] = None,
        vpin_toxicity: float = 0.0,
        current_date: Optional[datetime] = None
    ) -> VIXOverlayDecision:
        """
        Calculate overlay decision with all constraints.
        
        Args:
            signal: VIX term structure signal
            vix_history: Recent VIX readings for spike detection
            vpin_toxicity: Current VPIN reading (0-1)
            current_date: Current date for holding period calc
        
        Returns:
            VIXOverlayDecision with allowed shifts
        """
        current_date = current_date or datetime.now()
        
        # Calculate days since last shift
        days_since_last = 999  # Default high if no previous shift
        if self.last_shift_date:
            days_since_last = (current_date - self.last_shift_date).days
        
        # Check VIX spike
        vix_spike = False
        if vix_history and signal.vix_spot > 0:
            vix_spike = self._check_vix_spike(signal.vix_spot, vix_history)
        
        # Get target shifts from signal
        spy_shift, gld_shift, tlt_shift = self._get_allocation_shifts(signal.signal_value)
        target_shifts = {
            "SPY": spy_shift,
            "GLD": gld_shift,
            "TLT": tlt_shift
        }
        
        # Apply constraints
        allowed_shifts, constraints, status = self._apply_constraints(
            target_shifts,
            days_since_last,
            vpin_toxicity,
            vix_spike
        )
        
        # Determine urgency
        if status == VIXOverlayStatus.FROZEN:
            urgency = "deferred"
        elif status == VIXOverlayStatus.DISABLED:
            urgency = "deferred"
        elif status == VIXOverlayStatus.HOLDING:
            urgency = "deferred"
        elif abs(signal.signal_value) > 0.7:
            urgency = "immediate"
        elif abs(signal.signal_value) > 0.4:
            urgency = "gradual"
        else:
            urgency = "gradual"
        
        decision = VIXOverlayDecision(
            timestamp=current_date.isoformat(),
            status=status.value,
            signal_value=signal.signal_value,
            regime=signal.regime,
            target_spy_shift=target_shifts["SPY"],
            target_gld_shift=target_shifts["GLD"],
            target_tlt_shift=target_shifts["TLT"],
            allowed_spy_shift=allowed_shifts["SPY"],
            allowed_gld_shift=allowed_shifts["GLD"],
            allowed_tlt_shift=allowed_shifts["TLT"],
            constraints_applied=constraints,
            urgency=urgency,
            vpin_override=vpin_toxicity > self.vpin_threshold,
            vpin_threshold=self.vpin_threshold
        )
        
        # Update state if shift executed
        if status == VIXOverlayStatus.ACTIVE and any(abs(v) > 0.001 for v in allowed_shifts.values()):
            self._execute_shift(allowed_shifts, current_date, decision)
        
        return decision
    
    def _execute_shift(
        self,
        shifts: Dict[str, float],
        current_date: datetime,
        decision: VIXOverlayDecision
    ):
        """Execute allocation shift and update state."""
        # Update current allocation
        for symbol, shift in shifts.items():
            if symbol in self.current_allocation:
                self.current_allocation[symbol] += shift
        
        # Normalize to ensure sum = 1.0
        total = sum(self.current_allocation.values())
        if abs(total - 1.0) > 0.001:
            self.current_allocation = {
                k: v / total for k, v in self.current_allocation.items()
            }
        
        # Update tracking
        self.last_shift_date = current_date
        self.shift_history.append({
            "date": current_date.isoformat(),
            "shifts": shifts,
            "signal_value": decision.signal_value,
            "regime": decision.regime,
            "new_allocation": dict(self.current_allocation)
        })
        
        self._save_state()
        
        logger.info(f"VIX overlay shift executed: {shifts}")
        logger.info(f"New allocation: {self.current_allocation}")
    
    def get_current_allocation(self) -> Dict[str, float]:
        """Get current tactical allocation."""
        return dict(self.current_allocation)
    
    def reset_to_baseline(self):
        """Reset allocation to baseline."""
        self.current_allocation = dict(self.BASELINE)
        self.last_shift_date = None
        self._save_state()
        logger.info("VIX overlay reset to baseline")
    
    def get_shift_history(self, days: int = 90) -> List[Dict]:
        """Get recent shift history."""
        cutoff = datetime.now() - timedelta(days=days)
        return [
            h for h in self.shift_history
            if datetime.fromisoformat(h["date"]) > cutoff
        ]
    
    def get_summary(self) -> Dict:
        """Get overlay summary for dashboard."""
        recent_shifts = self.get_shift_history(30)
        
        return {
            "status": "active" if not self.disabled_until else "disabled",
            "current_allocation": self.current_allocation,
            "baseline_allocation": self.BASELINE,
            "active_drifts": {
                k: self.current_allocation[k] - self.BASELINE[k]
                for k in self.BASELINE.keys()
            },
            "shifts_30d": len(recent_shifts),
            "last_shift": self.last_shift_date.isoformat() if self.last_shift_date else None,
            "holding_days_remaining": max(0, self.min_holding_days - 
                (datetime.now() - self.last_shift_date).days if self.last_shift_date else 0),
            "disabled_until": self.disabled_until.isoformat() if self.disabled_until else None
        }


class VIXOverlayIntegrator:
    """
    Integrates VIX term structure overlay with ensemble voter and SmartRebalanceGate.
    
    Provides weighted signal contribution to portfolio decisions.
    """
    
    # Ensemble weight (15% as per spec)
    ENSEMBLE_WEIGHT = 0.15
    
    def __init__(self, overlay: Optional[VIXTermStructureOverlay] = None):
        self.overlay = overlay or VIXTermStructureOverlay()
    
    def get_ensemble_contribution(
        self,
        signal: VIXTermStructureSignal,
        vpin_toxicity: float = 0.0
    ) -> Dict[str, Any]:
        """
        Get ensemble voter contribution.
        
        Returns signal in standard format for ensemble aggregation.
        """
        decision = self.overlay.calculate_overlay(signal, vpin_toxicity=vpin_toxicity)
        
        # Calculate net directional signal
        net_shift = decision.allowed_spy_shift  # Primary signal from equity shift
        
        # Normalize to -1 to +1 for ensemble
        ensemble_signal = np.clip(net_shift * 10, -1, 1)  # Scale: 10% shift = 1.0 signal
        
        return {
            "source": "vix_term_structure",
            "weight": self.ENSEMBLE_WEIGHT,
            "signal": ensemble_signal,
            "confidence": signal.confidence / 100.0,
            "regime": signal.regime,
            "status": decision.status,
            "shift_recommendation": {
                "SPY": decision.allowed_spy_shift,
                "GLD": decision.allowed_gld_shift,
                "TLT": decision.allowed_tlt_shift
            },
            "urgency": decision.urgency
        }
    
    def integrate_with_rebalance_gate(
        self,
        gate_status: Dict[str, Any],
        decision: VIXOverlayDecision
    ) -> Dict[str, Any]:
        """
        Integrate with SmartRebalanceGate status.
        
        Coordinates VIX overlay execution with VPIN and cost constraints.
        """
        # Check if rebalance gate allows execution
        gate_allows = gate_status.get("can_execute", True)
        vpin_status = gate_status.get("vpin_status", "normal")
        
        # If VIX says immediate but gate says defer, gate wins
        if decision.urgency == "immediate" and not gate_allows:
            return {
                "execute": False,
                "reason": f"VIX overlay deferred: {gate_status.get('reason', 'gate blocked')}",
                "vix_urgency": decision.urgency,
                "gate_status": vpin_status,
                "deferred_shift": {
                    "SPY": decision.allowed_spy_shift,
                    "GLD": decision.allowed_gld_shift,
                    "TLT": decision.allowed_tlt_shift
                }
            }
        
        return {
            "execute": decision.status == "active",
            "reason": f"VIX overlay: {decision.status}",
            "vix_urgency": decision.urgency,
            "gate_status": vpin_status,
            "shift": {
                "SPY": decision.allowed_spy_shift,
                "GLD": decision.allowed_gld_shift,
                "TLT": decision.allowed_tlt_shift
            }
        }


def calculate_vix_overlay(
    vix_spot: float,
    vix3m: float,
    vix6m: Optional[float] = None,
    vix_history: Optional[List[float]] = None,
    vpin_toxicity: float = 0.0
) -> VIXOverlayDecision:
    """
    Convenience function for calculating VIX overlay decision.
    
    Args:
        vix_spot: Current VIX spot
        vix3m: Current VIX3M
        vix6m: Optional VIX6M
        vix_history: Recent VIX for spike detection
        vpin_toxicity: Current VPIN (0-1)
    
    Returns:
        VIXOverlayDecision with allowed shifts
    """
    calculator = VIXTermStructureCalculator()
    
    # Build signal
    signal_data = calculator.calculate_composite_signal(vix_spot, vix3m, vix6m, 
                                                         datetime.now().strftime("%Y-%m-%d"))
    
    # Create a minimal signal object for overlay
    from src.signals.vix_term_structure import VIXTermStructureSignal
    signal = VIXTermStructureSignal(
        timestamp=datetime.now().isoformat(),
        signal_state="risk_off" if signal_data["composite"] < -0.3 else 
                    "risk_on" if signal_data["composite"] > 0.3 else "neutral",
        signal_value=signal_data["composite"],
        vix_spot=vix_spot,
        vix3m=vix3m,
        vix6m=vix6m,
        slope_vix3m_vix=signal_data.get("slope", vix3m/vix_spot if vix_spot > 0 else 1.0),
        regime="backwardation" if signal_data["composite"] < -0.3 else
               "contango" if signal_data["composite"] > 0.3 else "flat",
        regime_strength=abs(signal_data["composite"]),
        slope_signal=signal_data["slope_signal"],
        roll_yield_signal=signal_data["roll_yield_signal"],
        vix_zscore_signal=signal_data["vix_zscore_signal"],
        curve_shape_signal=signal_data["curve_shape_signal"],
        spy_shift=0.0,
        gld_shift=0.0,
        tlt_shift=0.0,
        confidence=70.0,
        is_valid=True,
        reason="Overlay convenience function"
    )
    
    # Calculate overlay
    overlay = VIXTermStructureOverlay()
    return overlay.calculate_overlay(signal, vix_history, vpin_toxicity)


def get_vix_overlay_summary() -> Dict:
    """Get current overlay summary for dashboard/API."""
    overlay = VIXTermStructureOverlay()
    return overlay.get_summary()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with current market conditions
    decision = calculate_vix_overlay(
        vix_spot=22.5,
        vix3m=20.2,
        vix6m=21.0,
        vpin_toxicity=0.55
    )
    
    print(f"VIX Overlay Decision:")
    print(f"  Status: {decision.status}")
    print(f"  Signal: {decision.signal_value:.3f}")
    print(f"  Regime: {decision.regime}")
    print(f"  Allowed Shifts:")
    print(f"    SPY: {decision.allowed_spy_shift:+.1%}")
    print(f"    GLD: {decision.allowed_gld_shift:+.1%}")
    print(f"    TLT: {decision.allowed_tlt_shift:+.1%}")
    print(f"  Urgency: {decision.urgency}")
    print(f"  Constraints: {decision.constraints_applied}")
