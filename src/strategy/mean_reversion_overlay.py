#!/usr/bin/env python3
"""
Portfolio-Lab v4.81: VIX-Gated Mean-Reversion Overlay Strategy

Entry/exit rule engine with position sizing and state persistence.
Integrates with EnsembleVoter for portfolio-level decisions.

Usage:
    python -m src.strategy.mean_reversion_overlay status   # Current status
    python -m src.strategy.mean_reversion_overlay allocate  # Get recommended allocation
    python -m src.strategy.mean_reversion_overlay history   # Trade history
"""

import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STATE_PATH = DATA_DIR / "mean_reversion_state.json"
TRADES_PATH = DATA_DIR / "mean_reversion_trades.json"

# Import from signal module
sys.path.insert(0, str(PROJECT_ROOT))
from src.signals.mean_reversion_signal import (
    VIXMeanReversionCalculator,
    MeanReversionSignal,
    MeanReversionTrade,
    BASE_ALLOC_PCT,
    MAX_ALLOC_PCT,
    MAX_HOLD_DAYS,
    DATA_DIR,
    STATE_PATH,
)

# Ensemble voter integration weight
ENSEMBLE_WEIGHT = 0.05  # 5% weight in EnsembleVoter
FUNDING_SOURCE = "GLD"  # Funded from GLD allocation


@dataclass
class MeanReversionAllocation:
    """Allocation recommendation from the mean-reversion overlay."""
    timestamp: str
    active: bool
    allocation_pct: float
    entry_price: Optional[float]
    hold_days: int
    trade_return_pct: float
    vix_level: float
    vix_regime: str
    rationale: str
    fund_from: str  # Source asset to fund allocation
    
    # Ensemble integration
    ensemble_signal_value: float  # -1 to +1
    ensemble_weight: float
    
    # Entry conditions summary
    spy_3d_return: float
    spy_above_200ma: bool
    vpin_ok: bool
    entry_conditions_met: bool


class MeanReversionOverlay:
    """Strategy overlay for VIX-gated mean-reversion dip-buying."""

    def __init__(self, state_path: Path = STATE_PATH):
        self.state_path = state_path
        self.calculator = VIXMeanReversionCalculator()

    def get_status(self) -> Dict:
        """Get current overlay status."""
        signal = self.calculator.generate_signal()
        state = self.calculator.compute_trade_state()
        
        return {
            "timestamp": signal.timestamp,
            "active": state.get("active", False),
            "vix_regime": signal.vix_regime,
            "vix_level": signal.vix_level,
            "spy_price": signal.spy_price,
            "entry_triggered": signal.entry_triggered,
            "entry_conditions_met": signal.entry_triggered,
            "entry_reason": signal.entry_reason,
            "trade_state": signal.trade_state,
            "trade_entry_price": signal.trade_entry_price,
            "trade_hold_days": signal.trade_hold_days,
            "trade_return_pct": signal.trade_return_pct,
            "recommended_allocation_pct": signal.recommended_allocation_pct,
            "allocation_rationale": signal.allocation_rationale,
            "signal_value": signal.signal_value,
            "signal_strength": signal.signal_strength,
            "fund_from": FUNDING_SOURCE,
            "ensemble_weight": ENSEMBLE_WEIGHT,
        }

    def get_allocation(self) -> MeanReversionAllocation:
        """Get recommended allocation from the overlay."""
        signal = self.calculator.generate_signal()
        
        return MeanReversionAllocation(
            timestamp=signal.timestamp,
            active=signal.trade_state in ("entering", "active"),
            allocation_pct=signal.recommended_allocation_pct,
            entry_price=signal.trade_entry_price,
            hold_days=signal.trade_hold_days,
            trade_return_pct=signal.trade_return_pct,
            vix_level=signal.vix_level,
            vix_regime=signal.vix_regime,
            rationale=signal.allocation_rationale,
            fund_from=FUNDING_SOURCE,
            ensemble_signal_value=signal.signal_value,
            ensemble_weight=ENSEMBLE_WEIGHT,
            spy_3d_return=signal.spy_3d_return,
            spy_above_200ma=signal.spy_above_200ma,
            vpin_ok=signal.vpin_ok,
            entry_conditions_met=signal.entry_triggered,
        )

    def get_trade_history(self) -> List[Dict]:
        """Get completed trade history."""
        if not TRADES_PATH.exists():
            return []
        try:
            with open(TRADES_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, TypeError):
            return []

    def get_trade_summary(self) -> Dict:
        """Get summary of trade performance."""
        trades = self.get_trade_history()
        if not trades:
            return {"total_trades": 0, "message": "No trades recorded"}
        
        win = [t for t in trades if t.get("return_pct", 0) > 0]
        loss = [t for t in trades if t.get("return_pct", 0) <= 0]
        
        returns = [t.get("return_pct", 0) for t in trades]
        exits = {}
        for t in trades:
            r = t.get("exit_reason", "unknown")
            exits[r] = exits.get(r, 0) + 1
        
        return {
            "total_trades": len(trades),
            "win_rate_pct": round(len(win) / len(trades) * 100, 1) if trades else 0,
            "avg_win_pct": round(np.mean([t["return_pct"] for t in win]), 2) if win else 0,
            "avg_loss_pct": round(np.mean([t["return_pct"] for t in loss]), 2) if loss else 0,
            "total_return_pct": round(sum(returns), 2),
            "avg_return_pct": round(np.mean(returns), 2) if returns else 0,
            "avg_hold_days": round(np.mean([t["hold_days"] for t in trades]), 1) if trades else 0,
            "best_trade_pct": round(max(returns), 2) if returns else 0,
            "worst_trade_pct": round(min(returns), 2) if returns else 0,
            "exit_reasons": exits,
        }

    def reset_state(self) -> bool:
        """Reset trade state (for manual override)."""
        try:
            self.calculator.save_trade_state({
                "active": False,
                "entry_date": None,
                "entry_price": None,
                "entry_vix": None,
                "hold_days": 0,
                "allocation_pct": 0.0,
            })
            logger.info("Mean-reversion overlay state reset")
            return True
        except Exception as e:
            logger.error(f"Failed to reset state: {e}")
            return False


# Convenience function for EnsembleVoter integration
def get_mean_reversion_ensemble_signals() -> Dict:
    """
    Get ensemble-ready signals for the EnsembleVoter.
    Returns dict with signal_value and weight.
    """
    overlay = MeanReversionOverlay()
    allocation = overlay.get_allocation()
    
    return {
        "mean_reversion": {
            "signal_value": allocation.ensemble_signal_value,
            "weight": allocation.ensemble_weight,
            "active": allocation.active,
            "allocation_pct": allocation.allocation_pct,
            "rationale": allocation.rationale,
            "vix_regime": allocation.vix_regime,
            "vix_level": allocation.vix_level,
        }
    }


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VIX-Gated Mean-Reversion Overlay")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    subparsers.add_parser("status", help="Show current overlay status")
    subparsers.add_parser("allocate", help="Get allocation recommendation")
    subparsers.add_parser("history", help="Show trade history")
    subparsers.add_parser("summary", help="Show trade summary")
    subparsers.add_parser("reset", help="Reset overlay state")
    
    args = parser.parse_args()
    overlay = MeanReversionOverlay()
    
    if args.command == "status":
        print(json.dumps(overlay.get_status(), indent=2, default=str))
    
    elif args.command == "allocate":
        alloc = overlay.get_allocation()
        print(json.dumps(asdict(alloc), indent=2, default=str))
    
    elif args.command == "history":
        history = overlay.get_trade_history()
        print(json.dumps(history, indent=2, default=str))
    
    elif args.command == "summary":
        summary = overlay.get_trade_summary()
        print(json.dumps(summary, indent=2, default=str))
    
    elif args.command == "reset":
        if overlay.reset_state():
            print(json.dumps({"status": "ok", "message": "State reset"}))
        else:
            print(json.dumps({"status": "error", "message": "Reset failed"}))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
