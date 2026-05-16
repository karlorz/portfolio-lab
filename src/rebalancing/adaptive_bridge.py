"""
v5.75: Adaptive Sizing → Rebalance Scheduler Bridge
Integrates AdaptivePositionSizer with SmartRebalanceGate so that target
allocations adapt dynamically to market conditions (regime, volatility,
signal conviction, drawdown) before drift is evaluated.

Flow:
  1. Query AdaptiveSizer for current optimal allocation (SPY/GLD/TLT)
  2. Pass dynamic targets to SmartRebalanceGate for drift calculation
  3. Gate evaluates timing, VPIN, and cost as usual

Usage:
    from src.rebalancing.adaptive_bridge import AdaptiveRebalanceBridge

    bridge = AdaptiveRebalanceBridge()
    result = bridge.evaluate(
        current_holdings={'SPY': 47000, 'GLD': 37000, 'TLT': 16000},
        total_value=100000,
    )
    if result.should_execute:
        # Rebalance to bridge.target_allocation (dynamic)
        print(f"Rebalance to: {bridge.target_allocation}")
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from src.rebalancing.integration import SmartRebalanceGate, RebalanceGateResult
from src.strategy.adaptive_sizing import AdaptiveSizer, BASE_ALLOCATION

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class AdaptiveRebalanceResult:
    """Extended rebalance result with adaptive sizing information."""
    should_execute: bool
    decision: str
    urgency: str
    max_drift: float
    estimated_cost_bps: float
    reason: str
    static_target: Dict[str, float]
    dynamic_target: Dict[str, float]
    target_diff: Dict[str, float]  # dynamic - static per asset
    sizing_adjustments: Dict[str, float]  # net per-asset adjustments from sizer
    sizing_regime: str
    sizing_vol: float
    gate_result: RebalanceGateResult


class AdaptiveRebalanceBridge:
    """
    Combines AdaptiveSizer with SmartRebalanceGate.

    The bridge:
    1. Runs the AdaptiveSizer to compute the current optimal allocation
    2. Uses that as the dynamic target for the rebalance gate
    3. Also checks drift against static targets for comparison reporting
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        data_dir: Optional[Path] = None,
    ):
        self.sizer = AdaptiveSizer(data_dir=data_dir or (PROJECT_ROOT / "data"))
        self.gate = SmartRebalanceGate(config_path=config_path)
        self.last_sizing_decision = None
        self._target_allocation: Optional[Dict[str, float]] = None

    @property
    def target_allocation(self) -> Dict[str, float]:
        """Current dynamic target allocation from the sizer."""
        if self._target_allocation is None:
            self._target_allocation = dict(BASE_ALLOCATION)
        return self._target_allocation

    def refresh_targets(self) -> Dict[str, float]:
        """Compute fresh targets from the AdaptiveSizer."""
        try:
            decision = self.sizer.compute_allocation()
            self.last_sizing_decision = decision
            self._target_allocation = dict(decision.adjusted_allocation)
            logger.info(
                "Adaptive targets: SPY=%.1f%% GLD=%.1f%% TLT=%.1f%% (regime=%s, vol=%.1f%%)",
                decision.adjusted_allocation.get("SPY", 0) * 100,
                decision.adjusted_allocation.get("GLD", 0) * 100,
                decision.adjusted_allocation.get("TLT", 0) * 100,
                decision.factors.regime,
                decision.factors.spy_vol_20d * 100,
            )
        except Exception as e:
            logger.warning("Adaptive sizing failed, using static targets: %s", e)
            self._target_allocation = dict(BASE_ALLOCATION)
        return self._target_allocation

    def evaluate(
        self,
        current_holdings: Dict[str, float],
        total_value: float,
        vpin: Optional[float] = None,
        now: Optional[datetime] = None,
        use_dynamic_targets: bool = True,
    ) -> AdaptiveRebalanceResult:
        """
        Evaluate whether to rebalance, with optional adaptive targets.

        Args:
            current_holdings: symbol -> current market value
            total_value: total portfolio value
            vpin: optional VPIN override
            now: optional datetime override
            use_dynamic_targets: if True, use AdaptiveSizer targets;
                                 if False, use static BASE_ALLOCATION

        Returns:
            AdaptiveRebalanceResult with both static and dynamic analysis
        """
        if now is None:
            now = datetime.now()

        # Get static targets (base 46/38/16)
        static_targets = dict(BASE_ALLOCATION)

        # Get dynamic targets from sizer
        if use_dynamic_targets:
            dynamic_targets = self.refresh_targets()
        else:
            dynamic_targets = dict(static_targets)

        # Run gate with dynamic targets
        gate_result = self.gate.evaluate(
            current_holdings=current_holdings,
            target_allocations=dynamic_targets,
            total_value=total_value,
            vpin=vpin,
            now=now,
        )

        # Compute target difference for reporting
        target_diff = {}
        sizing_adjustments = {}
        for asset in ["SPY", "GLD", "TLT"]:
            dyn = dynamic_targets.get(asset, 0)
            sta = static_targets.get(asset, 0)
            target_diff[asset] = round(dyn - sta, 4)

        # Get sizing adjustments from the last decision
        if self.last_sizing_decision:
            for asset in ["SPY", "GLD", "TLT"]:
                sizing_adjustments[asset] = round(
                    self.last_sizing_decision.adjustments.get(asset, 0), 4
                )

        regime = (self.last_sizing_decision.factors.regime
                  if self.last_sizing_decision else "unknown")
        vol = (self.last_sizing_decision.factors.spy_vol_20d
               if self.last_sizing_decision else 0.0)

        return AdaptiveRebalanceResult(
            should_execute=gate_result.should_execute,
            decision=gate_result.decision,
            urgency=gate_result.urgency,
            max_drift=gate_result.max_drift,
            estimated_cost_bps=gate_result.estimated_cost_bps,
            reason=gate_result.reason,
            static_target=static_targets,
            dynamic_target=dynamic_targets,
            target_diff=target_diff,
            sizing_adjustments=sizing_adjustments,
            sizing_regime=regime,
            sizing_vol=vol,
            gate_result=gate_result,
        )

    def record_execution(self, cost_bps: float, date: str, symbols: list):
        """Record a completed rebalance."""
        self.gate.record_execution(cost_bps, date, symbols)

    def get_status(self) -> Dict[str, Any]:
        """Get combined status of bridge, gate, and sizer."""
        gate_status = self.gate.get_status()
        sizing_state = {}
        if self.last_sizing_decision:
            sizing_state = {
                "regime": self.last_sizing_decision.factors.regime,
                "regime_confidence": self.last_sizing_decision.factors.regime_confidence,
                "spy_vol_20d": self.last_sizing_decision.factors.spy_vol_20d,
                "spy_drawdown_60d": self.last_sizing_decision.factors.spy_drawdown_60d,
                "ensemble_signal": self.last_sizing_decision.factors.ensemble_signal,
                "circuit_breaker": self.last_sizing_decision.factors.circuit_breaker_severity,
                "dynamic_allocation": self.target_allocation,
            }

        return {
            "bridge": {
                "active": True,
                "use_dynamic_targets": True,
                "current_target": self.target_allocation,
                "static_baseline": dict(BASE_ALLOCATION),
            },
            "gate": gate_status,
            "sizer": sizing_state,
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.get_status(), indent=2, default=str)


def demo():
    """Demonstrate the adaptive rebalance bridge."""
    bridge = AdaptiveRebalanceBridge()

    print("=" * 70)
    print("  ADAPTIVE REBALANCE BRIDGE — v5.75")
    print("=" * 70)

    # Sample portfolio near base allocation
    holdings = {
        "SPY": 47000,
        "GLD": 37000,
        "TLT": 16000,
    }
    total = sum(holdings.values())

    result = bridge.evaluate(
        current_holdings=holdings,
        total_value=total,
    )

    print(f"\n  Current Holdings: {holdings}")
    print(f"  Total Value:     ${total:,.0f}")
    print()
    print(f"  Static Target:   {result.static_target}")
    print(f"  Dynamic Target:  {result.dynamic_target}")
    print(f"  Target Diff:     {result.target_diff}")
    print()
    print(f"  Sizing Regime:   {result.sizing_regime}")
    print(f"  Sizing Vol:      {result.sizing_vol:.1%}")
    print(f"  Adjustments:     {result.sizing_adjustments}")
    print()
    print(f"  Decision:        {result.decision}")
    print(f"  Should Execute:  {result.should_execute}")
    print(f"  Urgency:         {result.urgency}")
    print(f"  Max Drift:       {result.max_drift:.2%}")
    print(f"  Est Cost:        {result.estimated_cost_bps:.1f} bps")
    print(f"  Reason:          {result.reason}")

    # Compare: what would happen with static targets?
    static_result = bridge.evaluate(
        current_holdings=holdings,
        total_value=total,
        use_dynamic_targets=False,
    )
    print(f"\n  (Static-only drift: {static_result.max_drift:.2%})")
    print()

    status = bridge.get_status()
    print(f"  Bridge Status:   active={status['bridge']['active']}")
    print(f"  YTD Cost:        {status['gate']['ytd_cost_pct']:.3f}%")
    print(f"  Remaining Budget:{status['gate']['remaining_budget_pct']:.3f}%")
    print()


if __name__ == "__main__":
    demo()
