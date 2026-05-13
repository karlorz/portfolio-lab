"""
Smart Rebalancing Integration — v2.90 Phase 2
Bridges SmartRebalancingController with the signal execution pipeline.

Adds rebalance gating: signals propose allocation shifts, smart controller
decides WHEN to execute based on drift, VPIN, timing, and budget.

Usage:
    from src.rebalancing.integration import SmartRebalanceGate

    gate = SmartRebalanceGate()
    decision = gate.evaluate(
        current_holdings={'SPY': 46000, 'GLD': 38000, 'TLT': 16000},
        target_allocations={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
        total_value=100000,
    )
    if decision.should_execute:
        # Proceed with rebalance
        pass
"""

import json
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass

from .smart_rebalancer import (
    SmartRebalancingController,
    PortfolioSnapshot,
    MarketConditions,
    RebalanceDecision,
    RebalanceDecisionResult,
    UrgencyLevel,
)


@dataclass
class RebalanceGateResult:
    """Result from the smart rebalance gate."""
    should_execute: bool
    decision: str
    urgency: str
    max_drift: float
    estimated_cost_bps: float
    reason: str
    metadata: Dict[str, Any]


class SmartRebalanceGate:
    """
    Gate that wraps SmartRebalancingController for use in the
    signal execution pipeline. Evaluates whether a proposed rebalance
    should execute now or be deferred.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.controller = SmartRebalancingController(config_path)
        self._vpin_cache: Dict[str, float] = {}

    def update_vpin(self, vpin: float):
        """Update current VPIN reading (from v2.65 or synthetic)."""
        self._vpin_cache['current'] = vpin

    def evaluate(
        self,
        current_holdings: Dict[str, float],
        target_allocations: Dict[str, float],
        total_value: float,
        vpin: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> RebalanceGateResult:
        """
        Evaluate whether a rebalance should execute.

        Args:
            current_holdings: symbol -> current market value
            target_allocations: symbol -> target allocation (0-1)
            total_value: total portfolio value
            vpin: current VPIN reading (uses cached if None)
            now: current time (uses datetime.now() if None)

        Returns:
            RebalanceGateResult with execute/defer decision
        """
        if now is None:
            now = datetime.now()

        # Create portfolio snapshot
        portfolio = PortfolioSnapshot(
            holdings=current_holdings,
            targets=target_allocations,
            total_value=total_value,
            timestamp=now,
        )

        # Get VPIN
        if vpin is None:
            vpin = self._vpin_cache.get('current', 0.30)

        # Create market conditions
        market = MarketConditions(vpin=vpin, timestamp=now)

        # Get decision from controller
        result = self.controller.should_rebalance(portfolio, market, now=now)

        should_execute = result.decision in (
            RebalanceDecision.EXECUTE,
            RebalanceDecision.OVERRIDE_EMERGENCY,
        )

        return RebalanceGateResult(
            should_execute=should_execute,
            decision=result.decision.value,
            urgency=result.urgency.value,
            max_drift=result.max_drift,
            estimated_cost_bps=result.estimated_cost_bps,
            reason=result.reason,
            metadata={
                'drift_details': result.drift_details,
                'vpin': vpin,
                'in_optimal_window': self.controller._in_optimal_window(now),
                'ytd_cost_bps': self.controller.cost_tracker.ytd_total_bps,
                'remaining_budget_pct': self.controller.cost_tracker.remaining_budget_pct,
            },
        )

    def record_execution(self, cost_bps: float, date: str, symbols: list):
        """Record a completed rebalance for budget tracking."""
        self.controller.record_rebalance(cost_bps, date, symbols)

    def get_status(self) -> Dict[str, Any]:
        """Get current gate status for dashboard."""
        return self.controller.get_status()

    def to_json(self) -> str:
        """Serialize current status to JSON for dashboard/API."""
        return json.dumps(self.get_status(), indent=2)


def create_gate_from_config(config_path: str = 'config/smart_rebalance.yaml') -> SmartRebalanceGate:
    """Factory function to create a gate from config file."""
    return SmartRebalanceGate(config_path)


# CLI interface
if __name__ == '__main__':
    import sys

    gate = SmartRebalanceGate()

    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        print(gate.to_json())
    elif len(sys.argv) > 1 and sys.argv[1] == 'check':
        # Example check
        result = gate.evaluate(
            current_holdings={'SPY': 52000, 'GLD': 33000, 'TLT': 15000},
            target_allocations={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            total_value=100000,
        )
        print(f"Decision: {result.decision}")
        print(f"Should execute: {result.should_execute}")
        print(f"Urgency: {result.urgency}")
        print(f"Max drift: {result.max_drift:.1%}")
        print(f"Estimated cost: {result.estimated_cost_bps:.1f} bps")
        print(f"Reason: {result.reason}")
    else:
        print("Usage: python -m src.rebalancing.integration [status|check]")
