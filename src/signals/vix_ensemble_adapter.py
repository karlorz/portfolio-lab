"""
VIX Ensemble Adapter (v2.44 Phase 4)

Maps VIX insurance position status to RegimeSignal format for ensemble voting.
Translates VIX insurance overlay into defensive posture signals.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

from src.signals.vix_insurance_signal import VIXInsuranceSignalGenerator, InsuranceSignalType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VIXEnsembleStatus:
    """VIX insurance status formatted for ensemble integration"""
    timestamp: str
    insurance_active: bool
    position_size_pct: float  # As % of portfolio (0.01 = 1%)
    days_to_expiry: Optional[int]
    unrealized_pnl_pct: float  # P&L as % of cost basis
    cost_basis: float
    current_value: float
    budget_used_pct: float  # YTD budget consumed
    
    # Derived signals
    defensive_bias: float  # 0.0 to 0.15 added to risk score
    cash_buffer_increase: float  # Additional cash buffer %
    
    # Status flags
    roll_pending: bool  # Roll needed within 7 days
    profit_opportunity: bool  # VIX > 35, potential take-profit
    budget_exhausted: bool  # >80% YTD budget used
    correlation_healthy: bool  # VIX-SPY correlation < -0.3
    
    # Next action
    next_action: str  # 'roll', 'hold', 'exit', 'enter', 'none'
    action_urgency: str  # 'immediate', 'soon', 'routine', 'none'


class VIXEnsembleAdapter:
    """
    Adapts VIX insurance position status to ensemble signal format.
    
    Signal translation rules:
    - Insurance active → defensive posture (+0.05 to +0.10 risk score)
    - Budget depleted → no action possible (signal only)
    - Roll pending → maintenance required flag
    - Profit opportunity (VIX >35) → risk_on signal (profit taking)
    - Correlation breakdown → warning flag
    """
    
    # Risk score adjustments
    DEFENSIVE_BIAS_ACTIVE = 0.05
    DEFENSIVE_BIAS_NEAR_EXPIRY = 0.10
    DEFENSIVE_BIAS_PROFIT = -0.05  # Can take more risk after profit
    
    # Cash buffer adjustments
    CASH_BUFFER_INCREASE = 0.005  # +0.5% when insurance active
    
    # Thresholds
    BUDGET_WARNING_PCT = 0.80
    ROLL_WARNING_DTE = 7
    PROFIT_VIX_THRESHOLD = 35.0
    CORRELATION_BREAKDOWN_THRESHOLD = -0.3
    
    def __init__(self, data_dir: str = "data/signals"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.status_file = self.data_dir / "vix_ensemble_status.json"
        self.signal_generator = VIXInsuranceSignalGenerator(data_dir)
        
    def generate_status(self) -> VIXEnsembleStatus:
        """Generate current VIX ensemble status from insurance signal"""
        # Get latest VIX insurance signal
        signal = self.signal_generator.generate_signal()
        
        # Calculate derived metrics
        insurance_active = signal.position_active
        position_size_pct = signal.position_cost_basis / signal.portfolio_value if signal.portfolio_value > 0 else 0
        
        days_to_expiry = signal.days_to_position_expiry if signal.days_to_position_expiry else None
        
        unrealized_pnl_pct = (
            (signal.position_current_value - signal.position_cost_basis) / signal.position_cost_basis * 100
            if signal.position_cost_basis > 0 else 0
        )
        
        budget_used_pct = signal.insurance_budget_ytd / (signal.portfolio_value * 0.01) if signal.portfolio_value > 0 else 0
        
        # Determine defensive bias
        defensive_bias = 0.0
        cash_buffer_increase = 0.0
        
        if insurance_active:
            base_bias = self.DEFENSIVE_BIAS_ACTIVE
            
            # Increase bias if near expiry
            if days_to_expiry and days_to_expiry < 30:
                base_bias = self.DEFENSIVE_BIAS_NEAR_EXPIRY
            
            # Reduce bias if large profit (can take more risk)
            if unrealized_pnl_pct > 100:
                base_bias += self.DEFENSIVE_BIAS_PROFIT
            
            defensive_bias = max(0.0, min(0.15, base_bias))
            cash_buffer_increase = self.CASH_BUFFER_INCREASE
        
        # Status flags
        roll_pending = days_to_expiry is not None and days_to_expiry <= self.ROLL_WARNING_DTE if days_to_expiry else False
        profit_opportunity = signal.spot_vix > self.PROFIT_VIX_THRESHOLD
        budget_exhausted = budget_used_pct > self.BUDGET_WARNING_PCT
        correlation_healthy = signal.correlation_vix_spy < self.CORRELATION_BREAKDOWN_THRESHOLD
        
        # Determine next action
        next_action = "none"
        action_urgency = "none"
        
        if profit_opportunity and insurance_active:
            next_action = "exit"
            action_urgency = "immediate"
        elif roll_pending and insurance_active:
            next_action = "roll"
            action_urgency = "soon" if days_to_expiry and days_to_expiry <= 3 else "routine"
        elif insurance_active:
            next_action = "hold"
            action_urgency = "routine"
        elif signal.signal_type == InsuranceSignalType.ENTER:
            next_action = "enter"
            action_urgency = "routine"
        elif budget_exhausted:
            next_action = "none"
            action_urgency = "none"
        
        status = VIXEnsembleStatus(
            timestamp=datetime.now().isoformat(),
            insurance_active=insurance_active,
            position_size_pct=position_size_pct,
            days_to_expiry=days_to_expiry,
            unrealized_pnl_pct=unrealized_pnl_pct,
            cost_basis=signal.position_cost_basis,
            current_value=signal.position_current_value,
            budget_used_pct=budget_used_pct,
            defensive_bias=defensive_bias,
            cash_buffer_increase=cash_buffer_increase,
            roll_pending=roll_pending,
            profit_opportunity=profit_opportunity,
            budget_exhausted=budget_exhausted,
            correlation_healthy=correlation_healthy,
            next_action=next_action,
            action_urgency=action_urgency
        )
        
        return status
    
    def get_ensemble_signal(self) -> Dict[str, Any]:
        """
        Generate ensemble-compatible signal dictionary.
        
        Returns dict with:
        - insurance_active: bool
        - risk_score_adjustment: float (added to ensemble risk score)
        - cash_buffer_pct: float (additional cash buffer)
        - weight_adjustments: dict of component weight changes
        - alerts: list of active alerts
        """
        status = self.generate_status()
        
        # Calculate weight adjustments
        weight_adjustments = {}
        if status.insurance_active:
            # Slightly reduce HMM weight (less need for regime detection when hedged)
            weight_adjustments['hmm'] = -0.02
            # Slightly increase cash allocation
            weight_adjustments['cash'] = status.cash_buffer_increase
        
        # Build alerts
        alerts = []
        if status.roll_pending:
            alerts.append({
                "type": "roll_needed",
                "message": f"VIX insurance roll needed in {status.days_to_expiry} days",
                "urgency": status.action_urgency
            })
        
        if status.profit_opportunity:
            alerts.append({
                "type": "profit_opportunity",
                "message": f"VIX at elevated level ({status.unrealized_pnl_pct:.1f}% unrealized gain)",
                "urgency": "immediate" if status.unrealized_pnl_pct > 50 else "routine"
            })
        
        if status.budget_exhausted:
            alerts.append({
                "type": "budget_exhausted",
                "message": f"Insurance budget {status.budget_used_pct:.0%} consumed",
                "urgency": "routine"
            })
        
        if not status.correlation_healthy:
            alerts.append({
                "type": "correlation_breakdown",
                "message": "VIX-SPY correlation breakdown - hedge may not work",
                "urgency": "immediate"
            })
        
        return {
            "timestamp": status.timestamp,
            "insurance_active": status.insurance_active,
            "risk_score_adjustment": status.defensive_bias,
            "cash_buffer_pct": status.cash_buffer_increase,
            "weight_adjustments": weight_adjustments,
            "alerts": alerts,
            "next_action": status.next_action,
            "action_urgency": status.action_urgency,
            "metadata": {
                "position_size_pct": status.position_size_pct,
                "days_to_expiry": status.days_to_expiry,
                "unrealized_pnl_pct": status.unrealized_pnl_pct,
                "budget_used_pct": status.budget_used_pct
            }
        }
    
    def save_status(self, status: Optional[VIXEnsembleStatus] = None) -> Path:
        """Save ensemble status to JSON file"""
        if status is None:
            status = self.generate_status()
        
        data = asdict(status)
        data['timestamp'] = status.timestamp
        
        with open(self.status_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"VIX ensemble status saved to {self.status_file}")
        return self.status_file
    
    def load_status(self) -> Optional[VIXEnsembleStatus]:
        """Load ensemble status from JSON file"""
        if not self.status_file.exists():
            return None
        
        with open(self.status_file, 'r') as f:
            data = json.load(f)
        
        return VIXEnsembleStatus(**data)


def main():
    """CLI entry point for testing"""
    adapter = VIXEnsembleAdapter()
    
    # Generate and display status
    status = adapter.generate_status()
    
    print("\n=== VIX Ensemble Status ===")
    print(f"Timestamp: {status.timestamp}")
    print(f"Insurance Active: {status.insurance_active}")
    print(f"Position Size: {status.position_size_pct:.2%}")
    print(f"Days to Expiry: {status.days_to_expiry}")
    print(f"Unrealized P&L: {status.unrealized_pnl_pct:.1f}%")
    print(f"Budget Used: {status.budget_used_pct:.1%}")
    print(f"Defensive Bias: +{status.defensive_bias:.2f}")
    print(f"Cash Buffer: +{status.cash_buffer_increase:.2%}")
    print(f"Next Action: {status.next_action} ({status.action_urgency})")
    
    # Generate ensemble signal
    signal = adapter.get_ensemble_signal()
    print("\n=== Ensemble Signal ===")
    print(json.dumps(signal, indent=2, default=str))
    
    # Save status
    adapter.save_status(status)
    print(f"\nStatus saved to: {adapter.status_file}")


if __name__ == "__main__":
    main()
