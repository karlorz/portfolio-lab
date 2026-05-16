"""
Smart Rebalancing Controller — v2.90
Unified rebalancing with drift triggers, VPIN timing, and cost optimization.

v7.03 added: Tax lot tracking, tax-aware rebalancing, and simulation.
"""

from .smart_rebalancer import (
    SmartRebalancingController,
    PortfolioSnapshot,
    MarketConditions,
    RebalanceDecision,
    RebalanceDecisionResult,
    UrgencyLevel,
    CostBudgetTracker,
)
from .integration import SmartRebalanceGate
from .adaptive_bridge import AdaptiveRebalanceBridge, AdaptiveRebalanceResult
from .tax_lot_tracker import TaxLotTracker, TaxLot, SoldLot, LotSelectionMethod, HoldingPeriod
from .tax_aware_rebalancer import (
    TaxAwareRebalancer,
    TaxCostEstimate,
    RebalanceAction,
    RebalancePlan,
    TaxAwareMode,
)
from .tax_simulator import TaxSimulator, SimulationResult, SimulationIteration

__all__ = [
    'SmartRebalancingController',
    'SmartRebalanceGate',
    'AdaptiveRebalanceBridge',
    'AdaptiveRebalanceResult',
    'PortfolioSnapshot',
    'MarketConditions',
    'RebalanceDecision',
    'RebalanceDecisionResult',
    'UrgencyLevel',
    'CostBudgetTracker',
    'TaxLotTracker',
    'TaxLot',
    'SoldLot',
    'LotSelectionMethod',
    'HoldingPeriod',
    'TaxAwareRebalancer',
    'TaxCostEstimate',
    'RebalanceAction',
    'RebalancePlan',
    'TaxAwareMode',
    'TaxSimulator',
    'SimulationResult',
    'SimulationIteration',
]
