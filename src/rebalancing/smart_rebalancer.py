"""
Smart Rebalancing Controller — v2.90
Combines drift-based triggers, VPIN microstructure timing, and intraday
seasonality optimization to minimize transaction costs while maintaining
portfolio tracking accuracy.

Integrates:
- v2.24 Drift-based rebalancing (±10% threshold)
- v2.65 VPIN microstructure timing
- v2.71 Intraday seasonality execution

Target: 40%+ cost reduction vs calendar rebalancing.
"""

import json
import yaml
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum


class RebalanceDecision(Enum):
    EXECUTE = "execute"
    DEFER_TOXICITY = "defer_toxicity"
    DEFER_TIMING = "defer_timing"
    DEFER_BUDGET = "defer_budget"
    SKIP_LOW_DRIFT = "skip_low_drift"
    OVERRIDE_EMERGENCY = "override_emergency"


class UrgencyLevel(Enum):
    LOW = "low"           # Drift 10-12%, can wait
    MODERATE = "moderate" # Drift 12-15%
    HIGH = "high"         # Drift 15-20%
    EMERGENCY = "emergency"  # Drift > 20%, override all


@dataclass
class PortfolioSnapshot:
    """Current portfolio state for drift calculation."""
    holdings: Dict[str, float]      # symbol -> current market value
    targets: Dict[str, float]       # symbol -> target allocation (0-1)
    total_value: float
    timestamp: datetime


@dataclass
class MarketConditions:
    """Current market microstructure conditions."""
    vpin: float                     # Volume-Synchronized Probability of Informed Trading (0-1)
    vix: Optional[float] = None
    spread_bps: Optional[Dict[str, float]] = None  # symbol -> spread in bps
    timestamp: Optional[datetime] = None


@dataclass
class RebalanceDecisionResult:
    """Output of the should_rebalance decision."""
    decision: RebalanceDecision
    urgency: UrgencyLevel
    max_drift: float
    drift_details: Dict[str, float]  # symbol -> drift %
    vpin: float
    estimated_cost_bps: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostBudgetTracker:
    """Tracks YTD rebalancing costs against annual budget."""
    annual_limit_pct: float = 0.005     # 0.5% default
    warning_threshold_pct: float = 0.004  # Alert at 80%
    ytd_costs: List[Dict] = field(default_factory=list)

    @property
    def ytd_total_bps(self) -> float:
        return sum(c.get('cost_bps', 0) for c in self.ytd_costs)

    @property
    def ytd_total_pct(self) -> float:
        return self.ytd_total_bps / 10000

    @property
    def remaining_budget_pct(self) -> float:
        return max(0, self.annual_limit_pct - self.ytd_total_pct)

    def add_cost(self, cost_bps: float, date: str, symbols: List[str]):
        self.ytd_costs.append({
            'cost_bps': cost_bps,
            'date': date,
            'symbols': symbols,
        })

    def is_over_budget(self) -> bool:
        return self.ytd_total_pct >= self.annual_limit_pct

    def is_warning(self) -> bool:
        return self.ytd_total_pct >= self.warning_threshold_pct


class SmartRebalancingController:
    """
    Unified rebalancing controller combining drift triggers, VPIN timing,
    and intraday seasonality optimization.
    """

    DEFAULT_CONFIG = {
        'drift_threshold': 0.10,
        'urgency_levels': {
            'emergency': 0.95,   # Drift > 20%
            'high': 0.70,        # Drift 15-20%
            'moderate': 0.50,    # Drift 12-15%
            'low': 0.30,         # Drift 10-12%
        },
        'vpin': {
            'threshold': 0.50,
            'default': 0.30,     # Default when VPIN not available
        },
        'timing': {
            'optimal_start': 11,  # 11:00 ET
            'optimal_end': 14,    # 14:00 ET
            'low_urgency_can_wait': True,
        },
        'cost_budget': {
            'annual_limit': 0.005,
            'warning_threshold': 0.004,
        },
        'fallback': {
            'deferral_max_hours': 4,
            'force_if_drift_exceeds': 0.25,
        },
        'safety': {
            'max_deferral_hours': 4,
            'max_single_trade_cost_bps': 15,
            'max_annual_cost_pct': 0.006,
            'min_drift_override': 0.08,
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.cost_tracker = CostBudgetTracker(
            annual_limit_pct=self.config['cost_budget']['annual_limit'],
            warning_threshold_pct=self.config['cost_budget']['warning_threshold'],
        )
        self.deferred_until: Optional[datetime] = None
        self.last_rebalance: Optional[datetime] = None

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load config from YAML file or use defaults."""
        config = self.DEFAULT_CONFIG.copy()
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
                if user_config and 'smart_rebalancing' in user_config:
                    self._deep_merge(config, user_config['smart_rebalancing'])
        return config

    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge override into base dict."""
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

    def calculate_drift(self, portfolio: PortfolioSnapshot) -> Tuple[float, Dict[str, float]]:
        """
        Calculate maximum drift and per-asset drift details.
        Drift = |current_allocation - target_allocation| / target_allocation
        """
        drift_details = {}
        max_drift = 0.0

        for symbol, target_alloc in portfolio.targets.items():
            current_value = portfolio.holdings.get(symbol, 0)
            current_alloc = current_value / portfolio.total_value if portfolio.total_value > 0 else 0
            drift = abs(current_alloc - target_alloc) / target_alloc if target_alloc > 0 else 0
            drift_details[symbol] = round(drift, 4)
            max_drift = max(max_drift, drift)

        return round(max_drift, 4), drift_details

    def calculate_urgency(self, max_drift: float) -> UrgencyLevel:
        """Map drift level to urgency classification."""
        if max_drift > 0.20:
            return UrgencyLevel.EMERGENCY
        elif max_drift > 0.15:
            return UrgencyLevel.HIGH
        elif max_drift > 0.12:
            return UrgencyLevel.MODERATE
        else:
            return UrgencyLevel.LOW

    def estimate_cost_bps(self, vpin: float, in_optimal_window: bool) -> float:
        """
        Estimate execution cost in basis points.
        Cost = Base_Spread × VPIN_Multiplier × Time_Multiplier + Fixed
        """
        base_spread = 0.0003   # 3 bps
        fixed = 0.0002         # 2 bps

        # VPIN multiplier: higher VPIN = higher cost
        vpin_mult = max(1.0, 1.0 + (vpin - 0.30) * 2.0)
        vpin_mult = min(vpin_mult, 2.0)

        # Time multiplier: outside optimal window = higher cost
        if in_optimal_window:
            time_mult = 1.0
        else:
            now = datetime.now()
            hour = now.hour
            if hour < 10:
                time_mult = 1.25   # Opening volatility
            elif hour >= 15.5:
                time_mult = 1.15   # Closing auction
            else:
                time_mult = 1.05   # Mid-morning / afternoon

        cost = base_spread * vpin_mult * time_mult + fixed
        return round(cost * 10000, 2)  # Convert to bps

    def _in_optimal_window(self, now: Optional[datetime] = None) -> bool:
        """Check if current time is in optimal execution window (11am-2pm ET)."""
        if now is None:
            now = datetime.now()
        hour = now.hour
        start = self.config['timing']['optimal_start']
        end = self.config['timing']['optimal_end']
        return start <= hour < end

    def should_rebalance(
        self,
        portfolio: PortfolioSnapshot,
        market: MarketConditions,
        now: Optional[datetime] = None,
    ) -> RebalanceDecisionResult:
        """
        Core decision engine: should we rebalance now, defer, or skip?

        Decision flow:
        1. Check drift threshold (skip if below)
        2. Calculate urgency from drift
        3. Check VPIN toxicity (defer if high and not urgent)
        4. Check timing window (defer if low urgency and outside window)
        5. Check cost budget (defer if over budget)
        6. Safety overrides (force if drift > 25%)
        """
        if now is None:
            now = datetime.now()

        # Step 1: Drift check
        max_drift, drift_details = self.calculate_drift(portfolio)
        min_drift = self.config['safety']['min_drift_override']

        if max_drift < self.config['drift_threshold']:
            return RebalanceDecisionResult(
                decision=RebalanceDecision.SKIP_LOW_DRIFT,
                urgency=UrgencyLevel.LOW,
                max_drift=max_drift,
                drift_details=drift_details,
                vpin=market.vpin,
                estimated_cost_bps=0,
                reason=f"drift_below_threshold ({max_drift:.1%} < {self.config['drift_threshold']:.1%})",
            )

        # Step 2: Urgency
        urgency = self.calculate_urgency(max_drift)
        vpin = market.vpin if market.vpin is not None else self.config['vpin']['default']

        # Step 3: Safety override — force if drift > 25%
        force_threshold = self.config['fallback']['force_if_drift_exceeds']
        if max_drift > force_threshold:
            cost = self.estimate_cost_bps(vpin, self._in_optimal_window(now))
            return RebalanceDecisionResult(
                decision=RebalanceDecision.OVERRIDE_EMERGENCY,
                urgency=UrgencyLevel.EMERGENCY,
                max_drift=max_drift,
                drift_details=drift_details,
                vpin=vpin,
                estimated_cost_bps=cost,
                reason=f"emergency_override (drift {max_drift:.1%} > {force_threshold:.1%})",
            )

        # Step 4: VPIN toxicity check
        vpin_threshold = self.config['vpin']['threshold']
        if vpin > vpin_threshold and urgency != UrgencyLevel.EMERGENCY:
            self.consecutive_deferrals = getattr(self, 'consecutive_deferrals', 0) + 1
            max_deferrals = self.config['fallback']['deferral_max_hours']  # Reused as max deferral count
            if self.consecutive_deferrals > max_deferrals:
                # Max deferral count exceeded — force execution
                self.consecutive_deferrals = 0
                cost = self.estimate_cost_bps(vpin, self._in_optimal_window(now))
                return RebalanceDecisionResult(
                    decision=RebalanceDecision.EXECUTE,
                    urgency=urgency,
                    max_drift=max_drift,
                    drift_details=drift_details,
                    vpin=vpin,
                    estimated_cost_bps=cost,
                    reason=f"max_deferral_exceeded (VPIN={vpin:.2f}, deferred {self.consecutive_deferrals}x)",
                )
            return RebalanceDecisionResult(
                decision=RebalanceDecision.DEFER_TOXICITY,
                urgency=urgency,
                max_drift=max_drift,
                drift_details=drift_details,
                vpin=vpin,
                estimated_cost_bps=0,
                reason=f"high_toxicity_defer (VPIN={vpin:.2f} > {vpin_threshold}, defer #{self.consecutive_deferrals})",
            )
        else:
            self.consecutive_deferrals = 0

        # Step 5: Timing window check
        in_window = self._in_optimal_window(now)
        if (urgency == UrgencyLevel.LOW
                and not in_window
                and self.config['timing']['low_urgency_can_wait']):
            return RebalanceDecisionResult(
                decision=RebalanceDecision.DEFER_TIMING,
                urgency=urgency,
                max_drift=max_drift,
                drift_details=drift_details,
                vpin=vpin,
                estimated_cost_bps=0,
                reason=f"wait_for_optimal_window (next: {self.config['timing']['optimal_start']}:00 ET)",
            )

        # Step 6: Cost budget check
        if self.cost_tracker.is_over_budget():
            if urgency != UrgencyLevel.EMERGENCY:
                return RebalanceDecisionResult(
                    decision=RebalanceDecision.DEFER_BUDGET,
                    urgency=urgency,
                    max_drift=max_drift,
                    drift_details=drift_details,
                    vpin=vpin,
                    estimated_cost_bps=0,
                    reason=f"cost_budget_exceeded (YTD: {self.cost_tracker.ytd_total_bps:.1f} bps)",
                )

        # All checks passed — execute
        cost = self.estimate_cost_bps(vpin, in_window)
        return RebalanceDecisionResult(
            decision=RebalanceDecision.EXECUTE,
            urgency=urgency,
            max_drift=max_drift,
            drift_details=drift_details,
            vpin=vpin,
            estimated_cost_bps=cost,
            reason="execute",
            metadata={
                'in_optimal_window': in_window,
                'ytd_cost_bps': self.cost_tracker.ytd_total_bps,
                'remaining_budget_pct': self.cost_tracker.remaining_budget_pct,
            },
        )

    def record_rebalance(self, cost_bps: float, date: str, symbols: List[str]):
        """Record a completed rebalance for budget tracking."""
        self.cost_tracker.add_cost(cost_bps, date, symbols)
        self.last_rebalance = datetime.fromisoformat(date) if 'T' in date else datetime.strptime(date, '%Y-%m-%d')
        self.deferred_until = None

    def get_status(self) -> Dict[str, Any]:
        """Get current controller status for dashboard/monitoring."""
        return {
            'ytd_cost_bps': self.cost_tracker.ytd_total_bps,
            'ytd_cost_pct': round(self.cost_tracker.ytd_total_pct * 100, 3),
            'remaining_budget_pct': round(self.cost_tracker.remaining_budget_pct * 100, 3),
            'is_over_budget': self.cost_tracker.is_over_budget(),
            'is_warning': self.cost_tracker.is_warning(),
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'deferred_until': self.deferred_until.isoformat() if self.deferred_until else None,
            'config': {
                'drift_threshold': self.config['drift_threshold'],
                'vpin_threshold': self.config['vpin']['threshold'],
                'optimal_window': f"{self.config['timing']['optimal_start']}:00-{self.config['timing']['optimal_end']}:00 ET",
                'annual_cost_limit': f"{self.config['cost_budget']['annual_limit'] * 100:.1f}%",
            },
        }


def create_sample_portfolio() -> PortfolioSnapshot:
    """Create a sample portfolio for testing."""
    return PortfolioSnapshot(
        holdings={
            'SPY': 46000,
            'GLD': 38000,
            'TLT': 16000,
        },
        targets={
            'SPY': 0.46,
            'GLD': 0.38,
            'TLT': 0.16,
        },
        total_value=100000,
        timestamp=datetime.now(),
    )


def demo():
    """Demonstrate the smart rebalancing controller."""
    controller = SmartRebalancingController()

    # Scenario 1: No drift — skip
    portfolio = create_sample_portfolio()
    market = MarketConditions(vpin=0.30)
    result = controller.should_rebalance(portfolio, market)
    print(f"Scenario 1 (no drift): {result.decision.value} — {result.reason}")

    # Scenario 2: 12% drift, low VPIN, in window — execute
    portfolio.holdings['SPY'] = 52000
    portfolio.holdings['GLD'] = 33000
    portfolio.holdings['TLT'] = 15000
    now = datetime(2026, 5, 13, 12, 0)  # Noon ET
    result = controller.should_rebalance(portfolio, market, now=now)
    print(f"Scenario 2 (12% drift, noon): {result.decision.value} — {result.reason}")
    print(f"  Urgency: {result.urgency.value}, Cost: {result.estimated_cost_bps:.1f} bps")

    # Scenario 3: 12% drift, high VPIN — defer
    market_high_vpin = MarketConditions(vpin=0.60)
    result = controller.should_rebalance(portfolio, market_high_vpin, now=now)
    print(f"Scenario 3 (12% drift, VPIN=0.60): {result.decision.value} — {result.reason}")

    # Scenario 4: 12% drift, outside optimal window — defer
    morning = datetime(2026, 5, 13, 9, 30)  # Market open
    result = controller.should_rebalance(portfolio, market, now=morning)
    print(f"Scenario 4 (12% drift, 9:30am): {result.decision.value} — {result.reason}")

    # Scenario 5: 26% drift — emergency override
    portfolio.holdings['SPY'] = 60000
    portfolio.holdings['GLD'] = 28000
    portfolio.holdings['TLT'] = 12000
    result = controller.should_rebalance(portfolio, market_high_vpin, now=morning)
    print(f"Scenario 5 (26% drift, high VPIN, morning): {result.decision.value} — {result.reason}")

    # Print status
    print(f"\nController Status: {json.dumps(controller.get_status(), indent=2)}")


if __name__ == '__main__':
    demo()
