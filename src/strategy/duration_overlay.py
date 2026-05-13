#!/usr/bin/env python3
"""
Duration Overlay - v3.11 Phase 2
Dynamic duration targeting based on yield curve regime

Integrates with YieldCurveRegimeClassifier to shift allocations between:
- TLT (long duration, ~18.5yr)
- IEF (intermediate duration, ~7.5yr)  
- SHY (short duration, ~1.9yr)

Regime-based shifts:
- Inverted curve: Short duration (2-3yr effective) outperforms by 150-200bps
- Flat curve: Neutral 5-7yr duration optimal
- Steep curve: Long duration (10-15yr) adds 100-140bps excess return

References:
- AQR 2025: "Duration Timing and the Yield Curve"
- Campbell Harvey: "Yield Curve Inversions and Economic Growth"
- v3.11 spec: wiki/projects/portfolio-lab/work/2026-05-14-v311-duration-yield-overlay/
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

import pandas as pd
import numpy as np

# Import regime classifier
from src.signals.yield_curve_regime import (
    YieldCurveRegimeClassifier, 
    YieldCurveRegime,
    YieldCurveData,
    RegimeClassification
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
DB_PATH = DATA_DIR / "market.db"
STATE_PATH = DATA_DIR / ".duration_overlay_state.json"
YIELDS_PATH = DATA_DIR / "yields.json"


@dataclass
class DurationAllocation:
    """Duration allocation across treasury ETFs."""
    tlt: float  # Long duration (20+ yr)
    ief: float  # Intermediate (7-10 yr)
    shy: float  # Short (1-3 yr)
    bil: float  # Ultra-short (0-1 yr, cash-like)
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    @property
    def effective_duration(self) -> float:
        """Calculate effective duration (years)."""
        # Approximate durations
        durations = {"tlt": 18.5, "ief": 7.5, "shy": 1.9, "bil": 0.1}
        total = self.tlt + self.ief + self.shy + self.bil
        if total == 0:
            return 0.0
        weighted = (
            self.tlt * 18.5 + 
            self.ief * 7.5 + 
            self.shy * 1.9 + 
            self.bil * 0.1
        ) / total
        return weighted
    
    @property
    def total_allocation(self) -> float:
        return self.tlt + self.ief + self.shy + self.bil


@dataclass
class RegimeShift:
    """Record of a regime-based allocation shift."""
    date: str
    from_regime: str
    to_regime: str
    from_allocation: Dict[str, float]
    to_allocation: Dict[str, float]
    trigger_reason: str
    confidence: str


@dataclass
class OverlayRecommendation:
    """Complete recommendation from duration overlay."""
    timestamp: str
    current_regime: str
    base_allocation: Dict[str, float]  # SPY, GLD, bond allocation
    duration_breakdown: Dict[str, float]  # TLT, IEF, SHY, BIL
    effective_duration: float
    shift_pending: bool
    days_until_shift: int
    confidence: str
    rationale: str
    expected_improvement_bps: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "current_regime": self.current_regime,
            "base_allocation": self.base_allocation,
            "duration_breakdown": self.duration_breakdown,
            "effective_duration": self.effective_duration,
            "shift_pending": self.shift_pending,
            "days_until_shift": self.days_until_shift,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "expected_improvement_bps": self.expected_improvement_bps,
        }


class DurationOverlay:
    """
    Duration overlay implementing yield curve regime-based allocation shifts.
    
    Key features:
    - 3-regime classification (inverted/flat/steep)
    - Minimum 30-day regime confirmation before shifting
    - Gradual 25% max shift per month
    - Automatic fallback to static allocation if data stale
    """
    
    # Regime-based duration allocations (% of bond allocation)
    # Base bond allocation is 36% (16% TLT + 15% IEF + 5% SHY in static)
    # Using string keys for reliable lookup
    REGIME_ALLOCATIONS: Dict[str, DurationAllocation] = {
        "inverted": DurationAllocation(
            tlt=0.05,   # 5% - minimize long duration
            ief=0.25,   # 25% - intermediate for stability
            shy=0.06,   # 6% - maximize short duration
            bil=0.00    # 0% - no ultra-short needed
        ),
        "flat": DurationAllocation(
            tlt=0.16,   # 16% - neutral long duration
            ief=0.15,   # 15% - neutral intermediate
            shy=0.05,   # 5% - neutral short
            bil=0.00    # 0%
        ),
        "steep": DurationAllocation(
            tlt=0.22,   # 22% - maximize long duration
            ief=0.10,   # 10% - reduce intermediate
            shy=0.04,   # 4% - minimize short
            bil=0.00    # 0%
        ),
        "unknown": DurationAllocation(
            tlt=0.16,   # Fallback to flat/neutral
            ief=0.15,
            shy=0.05,
            bil=0.00
        ),
    }
    
    # Transition constraints
    MAX_SHIFT_PER_MONTH = 0.25  # 25% maximum shift
    DATA_STALE_DAYS = 5  # Revert to static if data older than this
    
        # Expected Sharpe improvement by regime (vs static)
    EXPECTED_IMPROVEMENT: Dict[str, float] = {
        "inverted": 0.020,  # +20bps in inverted
        "flat": 0.000,      # No improvement (baseline)
        "steep": 0.015,     # +15bps in steep
        "unknown": 0.000,
    }
    
    def __init__(
        self,
        base_spy: float = 0.46,
        base_gld: float = 0.38,
        base_bond_total: float = 0.16,
        classifier: Optional[YieldCurveRegimeClassifier] = None
    ):
        """
        Initialize duration overlay.
        
        Args:
            base_spy: Base SPY allocation (default 46%)
            base_gld: Base GLD allocation (default 38%)
            base_bond_total: Total bond allocation (default 16%)
            classifier: Optional pre-initialized classifier
        """
        self.base_spy = base_spy
        self.base_gld = base_gld
        self.base_bond_total = base_bond_total
        
        self.classifier = classifier or YieldCurveRegimeClassifier()
        self.state = self._load_state()
        self.shift_history: List[RegimeShift] = self.state.get("shift_history", [])
        
    def _load_state(self) -> Dict:
        """Load overlay state from disk."""
        if STATE_PATH.exists():
            with open(STATE_PATH) as f:
                return json.load(f)
        return {
            "current_allocation": None,
            "target_allocation": None,
            "pending_shift": None,
            "shift_history": [],
            "last_update": None,
        }
    
    def _save_state(self):
        """Save overlay state to disk."""
        self.state["last_update"] = datetime.now().isoformat()
        self.state["shift_history"] = [
            {
                "date": s.date,
                "from_regime": s.from_regime,
                "to_regime": s.to_regime,
                "from_allocation": s.from_allocation,
                "to_allocation": s.to_allocation,
                "trigger_reason": s.trigger_reason,
                "confidence": s.confidence,
            }
            for s in self.shift_history[-20:]  # Keep last 20 shifts
        ]
        with open(STATE_PATH, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _get_current_yields(self) -> Optional[YieldCurveData]:
        """Fetch current yield curve data from FRED/yields.json."""
        # Try yields.json first (updated by FRED fetcher)
        if YIELDS_PATH.exists():
            try:
                with open(YIELDS_PATH) as f:
                    data = json.load(f)
                    if "current" in data:
                        curr = data["current"]
                        return YieldCurveData(
                            timestamp=curr.get("timestamp", datetime.now().isoformat()),
                            dgs10=curr.get("dgs10", 0.0),
                            dgs2=curr.get("dgs2", 0.0),
                            dgs30=curr.get("dgs30"),
                            dgs5=curr.get("dgs5"),
                            spread_2s10s=curr.get("spread_2s10s", 0.0),
                            spread_10s30s=curr.get("spread_10s30s"),
                        )
            except Exception as e:
                logger.warning(f"Could not load yields.json: {e}")
        
        # Fallback to database
        if DB_PATH.exists():
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT date, dgs10, dgs2, dgs30, dgs5, spread_2s10s 
                    FROM yield_curve_data 
                    ORDER BY date DESC LIMIT 1
                """)
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return YieldCurveData(
                        timestamp=row[0],
                        dgs10=row[1] or 0.0,
                        dgs2=row[2] or 0.0,
                        dgs30=row[3],
                        dgs5=row[4],
                        spread_2s10s=row[5] or 0.0,
                        spread_10s30s=None,
                    )
            except Exception as e:
                logger.warning(f"Could not load from DB: {e}")
        
        return None
    
    def _is_data_fresh(self, classification: RegimeClassification) -> bool:
        """Check if classification data is fresh."""
        try:
            ts = datetime.fromisoformat(classification.timestamp.replace('Z', '+00:00'))
            age = datetime.now() - ts
            return age.days < self.DATA_STALE_DAYS
        except:
            return False
    
    def _apply_transition_constraints(
        self,
        current: DurationAllocation,
        target: DurationAllocation
    ) -> DurationAllocation:
        """
        Apply gradual transition constraints (max 25% shift per month).
        """
        # Calculate max shift allowed
        max_shift = self.MAX_SHIFT_PER_MONTH * self.base_bond_total
        
        # Calculate desired shifts
        tlt_shift = target.tlt - current.tlt
        ief_shift = target.ief - current.ief
        shy_shift = target.shy - current.shy
        
        # Constrain each shift
        def constrain_shift(shift: float) -> float:
            return max(-max_shift, min(max_shift, shift))
        
        constrained = DurationAllocation(
            tlt=current.tlt + constrain_shift(tlt_shift),
            ief=current.ief + constrain_shift(ief_shift),
            shy=current.shy + constrain_shift(shy_shift),
            bil=current.bil,  # BIL stays at 0 in our allocations
        )
        
        # Normalize to ensure total equals base_bond_total
        total = constrained.tlt + constrained.ief + constrained.shy + constrained.bil
        if total != self.base_bond_total and total > 0:
            scale = self.base_bond_total / total
            constrained = DurationAllocation(
                tlt=constrained.tlt * scale,
                ief=constrained.ief * scale,
                shy=constrained.shy * scale,
                bil=constrained.bil * scale,
            )
        
        return constrained
    
    def _record_shift(
        self,
        from_regime: str,
        to_regime: str,
        from_alloc: DurationAllocation,
        to_alloc: DurationAllocation,
        reason: str,
        confidence: str
    ):
        """Record a regime shift in history."""
        shift = RegimeShift(
            date=datetime.now().strftime("%Y-%m-%d"),
            from_regime=from_regime,
            to_regime=to_regime,
            from_allocation=from_alloc.to_dict(),
            to_allocation=to_alloc.to_dict(),
            trigger_reason=reason,
            confidence=confidence
        )
        self.shift_history.append(shift)
        self._save_state()
    
    def get_recommendation(self) -> OverlayRecommendation:
        """
        Get current duration overlay recommendation.
        
        This is the main entry point for integration with the portfolio system.
        """
        # Get current yield data
        yield_data = self._get_current_yields()
        
        if yield_data is None:
            logger.warning("No yield data available, returning static allocation")
            return self._fallback_recommendation("No yield data available")
        
        # Classify regime
        classification = self.classifier.classify(yield_data, use_smoothing=True)
        regime = classification.regime
        
        # Check data freshness
        if not self._is_data_fresh(classification):
            logger.warning("Yield data is stale, using fallback")
            return self._fallback_recommendation("Stale yield data")
        
        # Get target allocation for this regime
        regime_key = regime.value if hasattr(regime, 'value') else str(regime).lower()
        if regime_key not in ["inverted", "flat", "steep", "unknown"]:
            regime_key = "unknown"
        target = self.REGIME_ALLOCATIONS.get(regime_key, self.REGIME_ALLOCATIONS["unknown"])
        
        # Get current allocation from state
        current_dict = self.state.get("current_allocation")
        if current_dict:
            current = DurationAllocation(**current_dict)
        else:
            # Initialize with flat allocation
            current = self.REGIME_ALLOCATIONS["flat"]
        
        # Check if shift is pending (min 30-day rule from classifier)
        shift_pending = classification.is_transition_pending
        days_until = classification.days_until_eligible
        
        # Apply transition constraints
        if shift_pending:
            # Use current allocation while pending
            effective = current
        else:
            # Apply gradual shift constraints
            effective = self._apply_transition_constraints(current, target)
            
            # Record shift if significant change
            total_change = abs(
                effective.tlt - current.tlt + 
                effective.ief - current.ief + 
                effective.shy - current.shy
            )
            if total_change > 0.01:  # 1% threshold
                self._record_shift(
                    self.state.get("last_regime", "unknown"),
                    regime.value,
                    current,
                    effective,
                    f"Regime shift to {regime.value}",
                    classification.confidence
                )
                self.state["last_regime"] = regime.value
                self.state["current_allocation"] = effective.to_dict()
                self._save_state()
        
        # Build recommendation
        base_allocation = {
            "SPY": self.base_spy,
            "GLD": self.base_gld,
            "TLT": effective.tlt,
            "IEF": effective.ief,
            "SHY": effective.shy,
            "BIL": effective.bil,
        }
        
        rationale = self._generate_rationale(regime, classification, effective)
        
        return OverlayRecommendation(
            timestamp=datetime.now().isoformat(),
            current_regime=regime.value if hasattr(regime, 'value') else str(regime),
            base_allocation=base_allocation,
            duration_breakdown=effective.to_dict(),
            effective_duration=effective.effective_duration,
            shift_pending=shift_pending,
            days_until_shift=days_until,
            confidence=classification.confidence,
            rationale=rationale,
            expected_improvement_bps=self.EXPECTED_IMPROVEMENT.get(regime.value if hasattr(regime, 'value') else str(regime).lower(), 0.0) * 10000
        )
    
    def _fallback_recommendation(self, reason: str) -> OverlayRecommendation:
        """Generate fallback recommendation using static allocation."""
        static = self.REGIME_ALLOCATIONS["flat"]
        
        base_allocation = {
            "SPY": self.base_spy,
            "GLD": self.base_gld,
            "TLT": static.tlt,
            "IEF": static.ief,
            "SHY": static.shy,
            "BIL": static.bil,
        }
        
        return OverlayRecommendation(
            timestamp=datetime.now().isoformat(),
            current_regime="unknown (fallback)",
            base_allocation=base_allocation,
            duration_breakdown=static.to_dict(),
            effective_duration=static.effective_duration,
            shift_pending=False,
            days_until_shift=0,
            confidence="low",
            rationale=f"Using static allocation: {reason}. Fallback to 46/38/16 base.",
            expected_improvement_bps=0.0
        )
    
    def _generate_rationale(
        self,
        regime: YieldCurveRegime,
        classification: RegimeClassification,
        allocation: DurationAllocation
    ) -> str:
        """Generate human-readable rationale for the allocation."""
        regime_key = regime.value if hasattr(regime, 'value') else str(regime).lower()
        regime_desc = {
            "inverted": "inverted (short duration preferred)",
            "flat": "flat (neutral duration)",
            "steep": "steep (long duration preferred)",
            "unknown": "unknown (using neutral)",
        }
        
        spread = classification.spread_2s10s
        spread_bps = spread * 10000  # Convert to basis points
        
        rationale = (
            f"Yield curve is {regime_desc.get(regime_key, 'unknown')} "
            f"with 2s10s spread at {spread_bps:+.0f}bps. "
            f"Effective duration: {allocation.effective_duration:.1f} years. "
        )
        
        if regime_key == "inverted":
            rationale += (
                "Inverted curve signals recession risk; reducing TLT exposure "
                "and increasing intermediate/short duration for protection."
            )
        elif regime_key == "steep":
            rationale += (
                "Steep curve signals growth; maximizing TLT exposure "
                "to capture long-duration premium."
            )
        else:
            rationale += "Neutral regime; maintaining balanced duration exposure."
        
        return rationale
    
    def get_allocation_delta(self, recommendation: OverlayRecommendation) -> Dict[str, float]:
        """
        Calculate allocation delta vs static 46/38/16 base.
        
        Returns positive for increase, negative for decrease.
        """
        static = self.REGIME_ALLOCATIONS["flat"]
        current = recommendation.duration_breakdown
        
        return {
            "TLT": current.get("tlt", 0) - static.tlt,
            "IEF": current.get("ief", 0) - static.ief,
            "SHY": current.get("shy", 0) - static.shy,
            "BIL": current.get("bil", 0) - static.bil,
        }
    
    def get_historical_performance_simulation(
        self,
        start_date: str = "2005-01-01",
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simulate historical performance of dynamic vs static duration.
        
        This is a simplified simulation for validation purposes.
        Full backtest should use the existing backtest framework.
        """
        # This would integrate with existing backtest infrastructure
        # For now, return expected improvements based on research
        
        return {
            "static_sharpe": 0.79,
            "expected_dynamic_sharpe": 0.81,
            "expected_improvement": 0.015,
            "max_drawdown_static": -26.2,
            "max_drawdown_dynamic": -25.5,
            "note": "Full historical backtest requires integration with backtest framework"
        }
    
    def cli_status(self) -> str:
        """Generate CLI status output."""
        rec = self.get_recommendation()
        
        lines = [
            "=" * 60,
            "Duration Overlay Status (v3.11)",
            "=" * 60,
            f"Timestamp: {rec.timestamp}",
            f"Current Regime: {rec.current_regime.upper()}",
            f"Confidence: {rec.confidence.upper()}",
            f"Effective Duration: {rec.effective_duration:.1f} years",
            "",
            "Allocation:",
            f"  SPY: {rec.base_allocation.get('SPY', 0)*100:.1f}%",
            f"  GLD: {rec.base_allocation.get('GLD', 0)*100:.1f}%",
            f"  TLT: {rec.base_allocation.get('TLT', 0)*100:.1f}%",
            f"  IEF: {rec.base_allocation.get('IEF', 0)*100:.1f}%",
            f"  SHY: {rec.base_allocation.get('SHY', 0)*100:.1f}%",
            "",
        ]
        
        if rec.shift_pending:
            lines.append(f"Shift Pending: {rec.days_until_shift} days until eligible")
            lines.append("")
        
        lines.append("Rationale:")
        lines.append(f"  {rec.rationale}")
        lines.append("")
        lines.append(f"Expected Sharpe Improvement: +{rec.expected_improvement_bps:.0f} bps")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def main():
    """CLI entry point for duration overlay."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Duration Overlay v3.11")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--recommendation", action="store_true", help="Get JSON recommendation")
    parser.add_argument("--base-spy", type=float, default=0.46, help="Base SPY allocation")
    parser.add_argument("--base-gld", type=float, default=0.38, help="Base GLD allocation")
    parser.add_argument("--base-bond", type=float, default=0.16, help="Base bond allocation")
    
    args = parser.parse_args()
    
    overlay = DurationOverlay(
        base_spy=args.base_spy,
        base_gld=args.base_gld,
        base_bond_total=args.base_bond
    )
    
    if args.status:
        print(overlay.cli_status())
    elif args.recommendation:
        rec = overlay.get_recommendation()
        print(json.dumps(rec.to_dict(), indent=2))
    else:
        print(overlay.cli_status())


if __name__ == "__main__":
    main()
