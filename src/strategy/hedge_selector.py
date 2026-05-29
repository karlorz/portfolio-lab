"""
Dynamic Hedge Selector - Regime-Adaptive Tail Risk Management

Selects optimal hedge instrument based on market regime, confidence, and cost-benefit analysis.
Unifies VIXY sizing, VIX calls, collar, and put spread under a single framework.

Usage:
    python -m src.strategy.hedge_selector select
    python -m src.strategy.hedge_selector status
"""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Tuple

import numpy as np

from src.paths import DATA_DIR
from src.backtest.metrics import save_results_json

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Regime thresholds (VIX-based)
    "normal_threshold": 20.0,
    "elevated_threshold": 30.0,
    "stress_threshold": 40.0,
    
    # Hedge instrument costs (annual bps)
    "vixy_cost_bps": 130,      # 0.85% expense + roll yield + decay
    "vix_calls_cost_bps": 250,  # premium decay
    "collar_cost_bps": 180,     # opportunity cost
    "put_spread_cost_bps": 350,  # premium
    
    # Minimum hold period (trading days)
    "min_hold_days": 5,
    
    # Confidence thresholds
    "high_confidence": 0.80,
    "medium_confidence": 0.60,
    "low_confidence": 0.40,
    
    # State file
    "state_file": str(DATA_DIR / "hedge_selector_state.json"),
}


# ── Enums ──────────────────────────────────────────────────────────────────

class HedgeInstrument(Enum):
    """Available hedge instruments."""
    VIXY = "vixy"
    VIX_CALLS = "vix_calls"
    COLLAR = "collar"
    PUT_SPREAD = "put_spread"
    NONE = "none"


class RegimeLabel(Enum):
    """Market regime labels."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    STRESS = "stress"
    CRISIS = "crisis"
    RECOVERY = "recovery"


# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class HedgeRecommendation:
    """Recommendation from the hedge selector."""
    timestamp: str
    regime: str
    regime_confidence: float
    
    # Primary hedge
    primary_hedge: str
    primary_size_pct: float
    
    # Secondary hedge (optional)
    secondary_hedge: Optional[str]
    secondary_size_pct: float
    
    # Cost-benefit
    expected_benefit_bps: float
    expected_cost_bps: float
    net_benefit_bps: float
    cost_benefit_gate: bool
    
    # Sizing
    kelly_fraction: float
    confidence_scaled_size: float
    
    # Metadata
    min_hold_days: int
    transition_cost_bps: float
    
    source: str = "hedge_selector"


@dataclass
class HedgeSelectorState:
    """Persistent state for hedge selector."""
    timestamp: str
    current_hedge: str
    current_size_pct: float
    days_in_position: int
    last_switch_date: Optional[str]
    ytd_switches: int
    ytd_cost_bps: float


# ── Hedge Selector ─────────────────────────────────────────────────────────

class HedgeSelector:
    """
    Dynamic hedge selector that chooses optimal instrument based on regime.
    
    Implements regime-based selection with cost-benefit gating and confidence scaling.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._state: Optional[HedgeSelectorState] = None
    
    def select(
        self,
        vix_level: float,
        regime_confidence: float,
        regime_label: Optional[str] = None,
        vix_history: Optional[List[float]] = None,
        portfolio_value: float = 1_000_000,
    ) -> HedgeRecommendation:
        """
        Select optimal hedge instrument based on current regime.
        
        Args:
            vix_level: Current VIX level
            regime_confidence: Confidence in regime classification (0-1)
            regime_label: Explicit regime label (overrides VIX-based classification)
            vix_history: Historical VIX levels for regime detection
            portfolio_value: Current portfolio value for sizing
            
        Returns:
            HedgeRecommendation with selected instrument and sizing
        """
        now = datetime.now().isoformat()
        
        # 1. Classify regime
        if regime_label:
            regime = RegimeLabel(regime_label)
        else:
            regime = self._classify_regime(vix_level)
        
        # 2. Check if we need to switch (min hold period)
        state = self._load_state()
        if state.days_in_position < self.config["min_hold_days"]:
            logger.info("Min hold period not met (%d/%d days), maintaining current hedge",
                       state.days_in_position, self.config["min_hold_days"])
            return self._maintain_current(regime, regime_confidence, now)
        
        # 3. Select hedge based on regime
        primary, secondary = self._select_instruments(regime, vix_level)
        
        # 4. Compute sizing
        primary_size = self._compute_size(primary, regime, regime_confidence, vix_level)
        secondary_size = self._compute_size(secondary, regime, regime_confidence, vix_level) * 0.5  # Secondary gets 50%
        
        # 5. Cost-benefit gate
        benefit = self._estimate_benefit(primary, primary_size, vix_level)
        cost = self._estimate_cost(primary, primary_size)
        net_benefit = benefit - cost
        gate = net_benefit > 0 or regime == RegimeLabel.CRISIS
        
        # 6. Kelly fraction
        kelly = self._kelly_fraction(regime, regime_confidence)
        
        # 7. Confidence scaling
        conf_scale = self._confidence_scale(regime_confidence)
        primary_size_scaled = primary_size * conf_scale
        secondary_size_scaled = secondary_size * conf_scale
        
        return HedgeRecommendation(
            timestamp=now,
            regime=regime.value,
            regime_confidence=regime_confidence,
            primary_hedge=primary.value,
            primary_size_pct=round(primary_size_scaled, 2),
            secondary_hedge=secondary.value if secondary != HedgeInstrument.NONE else None,
            secondary_size_pct=round(secondary_size_scaled, 2),
            expected_benefit_bps=round(benefit, 1),
            expected_cost_bps=round(cost, 1),
            net_benefit_bps=round(net_benefit, 1),
            cost_benefit_gate=gate,
            kelly_fraction=round(kelly, 3),
            confidence_scaled_size=round(primary_size_scaled, 2),
            min_hold_days=self.config["min_hold_days"],
            transition_cost_bps=25.0,  # estimated transition cost
        )
    
    def _classify_regime(self, vix_level: float) -> RegimeLabel:
        """Classify regime based on VIX level."""
        if vix_level < self.config["normal_threshold"]:
            return RegimeLabel.NORMAL
        elif vix_level < self.config["elevated_threshold"]:
            return RegimeLabel.ELEVATED
        elif vix_level < self.config["stress_threshold"]:
            return RegimeLabel.STRESS
        else:
            return RegimeLabel.CRISIS
    
    def _select_instruments(
        self, regime: RegimeLabel, vix_level: float
    ) -> Tuple[HedgeInstrument, HedgeInstrument]:
        """Select primary and secondary hedge instruments based on regime."""
        
        # Regime → hedge mapping (from compound page)
        mapping = {
            RegimeLabel.NORMAL: (HedgeInstrument.VIXY, HedgeInstrument.NONE),
            RegimeLabel.ELEVATED: (HedgeInstrument.VIXY, HedgeInstrument.VIX_CALLS),
            RegimeLabel.STRESS: (HedgeInstrument.PUT_SPREAD, HedgeInstrument.VIXY),
            RegimeLabel.CRISIS: (HedgeInstrument.COLLAR, HedgeInstrument.NONE),
            RegimeLabel.RECOVERY: (HedgeInstrument.VIXY, HedgeInstrument.NONE),
        }
        
        return mapping.get(regime, (HedgeInstrument.NONE, HedgeInstrument.NONE))
    
    def _compute_size(
        self,
        instrument: HedgeInstrument,
        regime: RegimeLabel,
        confidence: float,
        vix_level: float,
    ) -> float:
        """Compute hedge size as percentage of portfolio."""
        
        if instrument == HedgeInstrument.NONE:
            return 0.0
        
        # Base sizing by regime
        base_sizes = {
            HedgeInstrument.VIXY: {
                RegimeLabel.NORMAL: 1.0,
                RegimeLabel.ELEVATED: 4.0,
                RegimeLabel.STRESS: 8.0,
                RegimeLabel.CRISIS: 10.0,
                RegimeLabel.RECOVERY: 2.0,
            },
            HedgeInstrument.VIX_CALLS: {
                RegimeLabel.NORMAL: 0.0,
                RegimeLabel.ELEVATED: 2.0,
                RegimeLabel.STRESS: 3.0,
                RegimeLabel.CRISIS: 0.0,
                RegimeLabel.RECOVERY: 0.0,
            },
            HedgeInstrument.COLLAR: {
                RegimeLabel.NORMAL: 0.0,
                RegimeLabel.ELEVATED: 0.0,
                RegimeLabel.STRESS: 0.0,
                RegimeLabel.CRISIS: 5.0,
                RegimeLabel.RECOVERY: 0.0,
            },
            HedgeInstrument.PUT_SPREAD: {
                RegimeLabel.NORMAL: 0.0,
                RegimeLabel.ELEVATED: 0.0,
                RegimeLabel.STRESS: 6.0,
                RegimeLabel.CRISIS: 8.0,
                RegimeLabel.RECOVERY: 0.0,
            },
        }
        
        base = base_sizes.get(instrument, {}).get(regime, 0.0)
        
        # VIX scaling (higher VIX → slightly larger hedge)
        vix_scale = 1.0 + (vix_level - 20) / 200  # ±10% at VIX 0/40
        
        return base * vix_scale
    
    def _estimate_benefit(
        self, instrument: HedgeInstrument, size_pct: float, vix_level: float
    ) -> float:
        """Estimate expected benefit in basis points during a -15% SPY shock."""
        
        if instrument == HedgeInstrument.NONE or size_pct == 0:
            return 0.0
        
        # Benefit multipliers (approximate hedge P&L during -15% SPY shock)
        multipliers = {
            HedgeInstrument.VIXY: 350,      # VIXY gains ~3.5x SPY loss
            HedgeInstrument.VIX_CALLS: 500,      # VIX calls gain more
            HedgeInstrument.COLLAR: 150,     # Collar caps downside
            HedgeInstrument.PUT_SPREAD: 400, # Put spread gains
        }
        
        mult = multipliers.get(instrument, 0)
        shock_magnitude = 15.0  # -15% SPY shock
        
        # Benefit = size * multiplier * shock_magnitude / 100
        benefit = size_pct * mult * shock_magnitude / 100
        
        # Adjust by VIX (higher VIX → more protection needed → higher benefit)
        vix_adjustment = 1.0 + (vix_level - 20) / 100
        
        return benefit * vix_adjustment
    
    def _estimate_cost(self, instrument: HedgeInstrument, size_pct: float) -> float:
        """Estimate annual cost in basis points."""
        
        if instrument == HedgeInstrument.NONE or size_pct == 0:
            return 0.0
        
        cost_bps = self.config.get(f"{instrument.value}_cost_bps", 200)
        
        # Annualize: cost_bps is annual rate, applied to size
        return size_pct * cost_bps / 100
    
    def _kelly_fraction(self, regime: RegimeLabel, confidence: float) -> float:
        """Compute Kelly fraction for hedge sizing."""
        
        # Base Kelly by regime (more conservative in crisis)
        base_kelly = {
            RegimeLabel.NORMAL: 0.1,
            RegimeLabel.ELEVATED: 0.2,
            RegimeLabel.STRESS: 0.3,
            RegimeLabel.CRISIS: 0.4,
            RegimeLabel.RECOVERY: 0.15,
        }
        
        kelly = base_kelly.get(regime, 0.1)
        
        # Scale by confidence
        return kelly * confidence
    
    def _confidence_scale(self, confidence: float) -> float:
        """Scale hedge size by regime confidence."""
        
        if confidence >= self.config["high_confidence"]:
            return 1.0
        elif confidence >= self.config["medium_confidence"]:
            return 0.5
        elif confidence >= self.config["low_confidence"]:
            return 0.25
        else:
            return 0.0
    
    def _maintain_current(
        self, regime: RegimeLabel, confidence: float, timestamp: str
    ) -> HedgeRecommendation:
        """Maintain current hedge position."""
        
        state = self._load_state()
        
        return HedgeRecommendation(
            timestamp=timestamp,
            regime=regime.value,
            regime_confidence=confidence,
            primary_hedge=state.current_hedge,
            primary_size_pct=state.current_size_pct,
            secondary_hedge=None,
            secondary_size_pct=0.0,
            expected_benefit_bps=0.0,
            expected_cost_bps=0.0,
            net_benefit_bps=0.0,
            cost_benefit_gate=True,
            kelly_fraction=0.0,
            confidence_scaled_size=state.current_size_pct,
            min_hold_days=self.config["min_hold_days"],
            transition_cost_bps=0.0,
        )
    
    def _load_state(self) -> HedgeSelectorState:
        """Load persistent state."""
        state_file = self.config["state_file"]
        
        try:
            with open(state_file) as f:
                import json
                data = json.load(f)
            return HedgeSelectorState(**data)
        except (FileNotFoundError, KeyError, TypeError):
            return HedgeSelectorState(
                timestamp=datetime.now().isoformat(),
                current_hedge="none",
                current_size_pct=0.0,
                days_in_position=999,  # Allow immediate switch
                last_switch_date=None,
                ytd_switches=0,
                ytd_cost_bps=0.0,
            )
    
    def save_state(self, rec: HedgeRecommendation):
        """Save state after recommendation."""
        state = self._load_state()
        
        # Check if hedge changed
        if rec.primary_hedge != state.current_hedge:
            days_in_position = 0
            ytd_switches = state.ytd_switches + 1
        else:
            days_in_position = state.days_in_position + 1
            ytd_switches = state.ytd_switches
        
        new_state = HedgeSelectorState(
            timestamp=datetime.now().isoformat(),
            current_hedge=rec.primary_hedge,
            current_size_pct=rec.primary_size_pct,
            days_in_position=days_in_position,
            last_switch_date=datetime.now().isoformat() if days_in_position == 0 else state.last_switch_date,
            ytd_switches=ytd_switches,
            ytd_cost_bps=state.ytd_cost_bps + rec.expected_cost_bps,
        )
        
        save_results_json(asdict(new_state), output_path=self.config["state_file"])
        logger.info("Hedge selector state saved: %s (%.1f%%)", rec.primary_hedge, rec.primary_size_pct)
    
    def get_signal_snapshot(self) -> Dict:
        """Get signal snapshot for dashboard integration."""
        state = self._load_state()
        return {
            "current_hedge": state.current_hedge,
            "current_size_pct": state.current_size_pct,
            "days_in_position": state.days_in_position,
            "last_switch_date": state.last_switch_date,
            "ytd_switches": state.ytd_switches,
            "ytd_cost_bps": state.ytd_cost_bps,
        }


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamic Hedge Selector")
    parser.add_argument("mode", nargs="?", default="status",
                       choices=["select", "status"])
    parser.add_argument("--vix", type=float, default=18.0,
                       help="VIX level")
    parser.add_argument("--confidence", type=float, default=0.8,
                       help="Regime confidence (0-1)")
    parser.add_argument("--regime", type=str, default=None,
                       help="Explicit regime label")
    
    args = parser.parse_args()
    
    selector = HedgeSelector()
    
    if args.mode == "status":
        state = selector._load_state()
        print(f"Current hedge: {state.current_hedge} ({state.current_size_pct:.1f}%)")
        print(f"Days in position: {state.days_in_position}")
        print(f"YTD switches: {state.ytd_switches}")
        print(f"YTD cost: {state.ytd_cost_bps:.1f} bps")
    
    elif args.mode == "select":
        rec = selector.select(
            vix_level=args.vix,
            regime_confidence=args.confidence,
            regime_label=args.regime,
        )
        
        print(f"\n=== Hedge Selection ===")
        print(f"  Regime:            {rec.regime} (conf={rec.regime_confidence:.2f})")
        print(f"  Primary hedge:     {rec.primary_hedge} ({rec.primary_size_pct:.1f}%)")
        if rec.secondary_hedge:
            print(f"  Secondary hedge:   {rec.secondary_hedge} ({rec.secondary_size_pct:.1f}%)")
        print(f"  Cost-benefit gate: {'PASS' if rec.cost_benefit_gate else 'FAIL'}")
        print(f"  Net benefit:       {rec.net_benefit_bps:.1f} bps")
        print(f"  Kelly fraction:    {rec.kelly_fraction:.3f}")


if __name__ == "__main__":
    main()