#!/usr/bin/env python3
"""
Portfolio-Lab Alpha: Drawdown Circuit Breaker
Graduated drawdown protection with automatic position reduction.

Implements:
- 10% drawdown: Yellow alert (warning only)
- 15% drawdown: Orange alert (reduce position sizes by 25%)
- 20% drawdown: Red alert (reduce position sizes by 50%)
- 25% drawdown: Full kill switch (close all positions)

Reference: CME Group (2024). "Quantifying CTA Risk Management."
"""

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
DB_PATH = DATA_DIR / "market.db"
CIRCUIT_STATE_PATH = DATA_DIR / ".circuit_breaker_state.json"


class DrawdownCircuitBreaker:
    """
    Graduated drawdown protection system.
    
    Monitors portfolio value and triggers graduated responses:
    - 10%: Warning (alert only)
    - 15%: Caution (25% position reduction)
    - 20%: Critical (50% position reduction)  
    - 25%: Emergency (close all positions)
    
    Also supports leveraged ETF (UBT/TMF) specific thresholds as part of
    v2.35 Capital Efficiency implementation.
    """
    
    THRESHOLDS = {
        "green": 0.0,    # Normal operation
        "yellow": 0.10,  # 10% drawdown - warning
        "orange": 0.15,  # 15% drawdown - reduce 25%
        "red": 0.20,     # 20% drawdown - reduce 50%
        "black": 0.25,   # 25% drawdown - full stop
    }
    
    POSITION_SCALARS = {
        "green": 1.0,
        "yellow": 1.0,   # No reduction, just alert
        "orange": 0.75,  # Reduce to 75% of normal size
        "red": 0.50,     # Reduce to 50% of normal size
        "black": 0.0,    # Close all positions
    }
    
    # v2.35: Leveraged ETF specific thresholds
    # Based on 2x/3x leverage of underlying TLT (which has ~14% volatility)
    LEVERED_ETF_THRESHOLDS = {
        "UBT": {  # 2x TLT
            "daily_loss": 0.05,      # -5% daily (2x normal TLT move)
            "weekly_loss": 0.12,     # -12% weekly
            "monthly_loss": 0.15,    # -15% monthly = ~100bps rate rise
            "volatility_spike": 0.35,  # 30-day vol exceeds 35%
            "max_position_pct": 0.10,  # Max 10% portfolio (vs 16% TLT)
        },
        "TMF": {  # 3x TLT
            "daily_loss": 0.075,     # -7.5% daily (3x normal TLT move)
            "weekly_loss": 0.18,     # -18% weekly
            "monthly_loss": 0.225,   # -22.5% monthly
            "volatility_spike": 0.50,  # 30-day vol exceeds 50%
            "max_position_pct": 0.05,  # Max 5% portfolio (high risk)
        }
    }
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.state = self._load_state()
        
    def _load_state(self) -> Dict:
        """Load circuit breaker state from disk."""
        if CIRCUIT_STATE_PATH.exists():
            with open(CIRCUIT_STATE_PATH) as f:
                return json.load(f)
        return {
            "status": "green",
            "max_drawdown": 0.0,
            "peak_value": None,
            "triggered_at": None,
            "last_check": None,
            "reduction_count": 0
        }
    
    def _save_state(self):
        """Save circuit breaker state to disk."""
        self.state["last_check"] = datetime.now().isoformat()
        with open(CIRCUIT_STATE_PATH, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_portfolio_value_history(self) -> list:
        """Get portfolio value history from database."""
        if not DB_PATH.exists():
            return []
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Try to get from portfolio_value table first
        try:
            cursor.execute("""
                SELECT date, value FROM portfolio_value 
                WHERE date >= date('now', '-{} days')
                ORDER BY date ASC
            """.format(self.lookback_days))
            
            history = [{"date": row[0], "value": row[1]} for row in cursor.fetchall()]
            
            if history:
                conn.close()
                return history
                
        except sqlite3.OperationalError:
            pass
        
        # Fallback: try portfolio_paper.json
        conn.close()
        
        portfolio_file = DATA_DIR / "portfolio_paper.json"
        if portfolio_file.exists():
            with open(portfolio_file) as f:
                portfolio = json.load(f)
            return portfolio.get("history", [])
        
        return []
    
    def calculate_drawdown(self, history: list) -> Tuple[float, float, Optional[str]]:
        """
        Calculate current drawdown from peak.
        
        Returns: (current_drawdown, peak_value, peak_date)
        """
        if not history or len(history) < 2:
            return 0.0, 0.0, None
        
        # Get values
        values = [h.get("value", h.get("total_value", 0)) for h in history]
        dates = [h.get("date", "") for h in history]
        
        # Find peak in lookback period
        peak_idx = 0
        peak_value = values[0]
        
        for i, v in enumerate(values):
            if v > peak_value:
                peak_value = v
                peak_idx = i
        
        # Current value is last in history
        current_value = values[-1]
        
        # Calculate drawdown
        if peak_value > 0:
            drawdown = (peak_value - current_value) / peak_value
        else:
            drawdown = 0.0
        
        peak_date = dates[peak_idx] if peak_idx < len(dates) else None
        
        return drawdown, peak_value, peak_date
    
    def determine_status(self, drawdown: float) -> str:
        """Determine circuit breaker status from drawdown."""
        if drawdown >= self.THRESHOLDS["black"]:
            return "black"
        elif drawdown >= self.THRESHOLDS["red"]:
            return "red"
        elif drawdown >= self.THRESHOLDS["orange"]:
            return "orange"
        elif drawdown >= self.THRESHOLDS["yellow"]:
            return "yellow"
        else:
            return "green"
    
    def check_and_update(self) -> Dict:
        """
        Check drawdown and update circuit breaker state.
        
        Returns status dict with recommendations.
        """
        # Get portfolio history
        history = self.get_portfolio_value_history()
        
        if not history:
            return {
                "status": "unknown",
                "drawdown": None,
                "message": "No portfolio history available",
                "action": "none",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate drawdown
        drawdown, peak_value, peak_date = self.calculate_drawdown(history)
        
        # Determine status
        new_status = self.determine_status(drawdown)
        old_status = self.state.get("status", "green")
        
        # Update state
        self.state["max_drawdown"] = max(drawdown, self.state.get("max_drawdown", 0))
        self.state["peak_value"] = peak_value
        
        # Check for escalation
        status_changed = new_status != old_status
        escalated = self._is_escalation(old_status, new_status)
        
        if escalated:
            self.state["triggered_at"] = datetime.now().isoformat()
            self.state["reduction_count"] = self.state.get("reduction_count", 0) + 1
        
        self.state["status"] = new_status
        self._save_state()
        
        # Build response
        position_scalar = self.POSITION_SCALARS[new_status]
        
        result = {
            "status": new_status,
            "previous_status": old_status,
            "status_changed": status_changed,
            "escalated": escalated,
            "drawdown_pct": round(drawdown * 100, 2),
            "peak_value": round(peak_value, 2),
            "peak_date": peak_date,
            "current_value": round(history[-1].get("value", history[-1].get("total_value", 0)), 2),
            "position_scalar": position_scalar,
            "action": self._get_action(new_status),
            "message": self._get_message(new_status, drawdown),
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _is_escalation(self, old: str, new: str) -> bool:
        """Check if this is an escalation (moving to more severe status)."""
        severity = {"green": 0, "yellow": 1, "orange": 2, "red": 3, "black": 4}
        return severity.get(new, 0) > severity.get(old, 0)
    
    def _get_action(self, status: str) -> str:
        """Get recommended action for status."""
        actions = {
            "green": "normal_operation",
            "yellow": "monitor_closely",
            "orange": "reduce_positions_25pct",
            "red": "reduce_positions_50pct",
            "black": "close_all_positions"
        }
        return actions.get(status, "unknown")
    
    def _get_message(self, status: str, drawdown: float) -> str:
        """Get human-readable message for status."""
        messages = {
            "green": "Portfolio operating normally",
            "yellow": f"Warning: Portfolio in {drawdown*100:.1f}% drawdown. Monitor closely.",
            "orange": f"CAUTION: Portfolio in {drawdown*100:.1f}% drawdown. Reduce positions by 25%.",
            "red": f"CRITICAL: Portfolio in {drawdown*100:.1f}% drawdown. Reduce positions by 50%.",
            "black": f"EMERGENCY: Portfolio in {drawdown*100:.1f}% drawdown. Close all positions immediately."
        }
        return messages.get(status, "Unknown status")
    
    def reset(self, reason: str = "manual_reset"):
        """Reset circuit breaker state."""
        self.state = {
            "status": "green",
            "max_drawdown": 0.0,
            "peak_value": None,
            "triggered_at": None,
            "last_check": datetime.now().isoformat(),
            "reduction_count": 0,
            "reset_reason": reason,
            "reset_at": datetime.now().isoformat()
        }
        self._save_state()
        print(f"✓ Circuit breaker reset ({reason})")
    
    def get_status(self) -> Dict:
        """Get current circuit breaker status without checking."""
        return {
            "status": self.state.get("status", "unknown"),
            "max_drawdown_ever": round(self.state.get("max_drawdown", 0) * 100, 2),
            "peak_value": self.state.get("peak_value"),
            "triggered_at": self.state.get("triggered_at"),
            "reduction_count": self.state.get("reduction_count", 0),
            "last_check": self.state.get("last_check")
        }


def main():
    """CLI interface for circuit breaker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Drawdown Circuit Breaker")
    parser.add_argument("--check", action="store_true", help="Check drawdown and update status")
    parser.add_argument("--status", action="store_true", help="Get current status")
    parser.add_argument("--reset", type=str, metavar="REASON", help="Reset circuit breaker")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    cb = DrawdownCircuitBreaker()
    
    if args.check:
        result = cb.check_and_update()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n📊 Circuit Breaker Check: {result['timestamp']}")
            print(f"   Status: {result['status'].upper()}")
            print(f"   Drawdown: {result['drawdown_pct']}%")
            print(f"   Peak: ${result['peak_value']:,.2f} on {result['peak_date']}")
            print(f"   Current: ${result['current_value']:,.2f}")
            print(f"   Position Scalar: {result['position_scalar']}")
            print(f"   Action: {result['action']}")
            if result['status_changed']:
                print(f"   ⚠️  STATUS CHANGED from {result['previous_status']}!")
            print(f"\n   Message: {result['message']}")
    
    elif args.reset:
        cb.reset(args.reset)
        print(f"✓ Circuit breaker reset: {args.reset}")
    
    else:  # default to status
        status = cb.get_status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"\n📊 Circuit Breaker Status")
            print(f"   Current Status: {status['status'].upper()}")
            print(f"   Max Drawdown (ever): {status['max_drawdown_ever']}%")
            print(f"   Peak Value: ${status['peak_value']:,.2f}" if status['peak_value'] else "   Peak Value: N/A")
            print(f"   Last Check: {status['last_check']}")
            print(f"   Times Triggered: {status['reduction_count']}")
            if status['triggered_at']:
                print(f"   Last Triggered: {status['triggered_at']}")


if __name__ == "__main__":
    main()
