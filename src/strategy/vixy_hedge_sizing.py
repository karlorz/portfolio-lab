"""
Dynamic VIXY Hedge Sizing - v7.04 Implementation
QuantPedia-style robust VIXY model: VIXY allocation = VIX level / 1000.

Builds on v4.50 VIX term structure overlay and v4.60 cashless collar.
Provides explicit VIXY/hedge ETF allocation sizing with cost-benefit analysis
and EnsembleVoter integration.

Usage:
    python -m src.strategy.vixy_hedge_sizing status
    python -m src.strategy.vixy_hedge_sizing recommend
    python -m src.strategy.vixy_hedge_sizing backtest --start 2006-01-01
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd

from src.backtest.metrics import save_results_json
from src.paths import DATA_DIR



__all__ = ['DEFAULT_CONFIG', 'HedgeRegime', 'HedgeAction', 'VIXYHedgeConfig', 'VIXYHedgeSignal', 'VIXYHedgeState', 'VIXYHedgeSizer']

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "min_hedge_pct": 0.0,        # Minimum VIXY allocation %
    "max_hedge_pct": 10.0,       # Maximum VIXY allocation %
    "cost_threshold": 2.0,       # Max acceptable cost/benefit ratio
    "vixy_expense_ratio": 0.0085,  # VIXY ETF annual expense ratio
    "monthly_decay_pct": 0.05,    # Monthly premium decay estimate
    "spy_shock_pct": -15.0,       # Standard stress scenario
    "state_file": str(DATA_DIR / "vixy_hedge_state.json"),
    "ensemble_weight_normal": 0.05,   # 5% in normal regime
    "ensemble_weight_stress": 0.10,   # 10% in stress/crisis
}

# ── Enums ──────────────────────────────────────────────────────────────────

class HedgeRegime(Enum):
    """VIX-based hedge sizing regime."""
    NORMAL = "normal"       # VIX < 20
    ELEVATED = "elevated"   # VIX 20-30
    STRESS = "stress"       # VIX 30-40
    CRISIS = "crisis"       # VIX > 40


class HedgeAction(Enum):
    """Recommended hedge action."""
    INCREASE = "increase"
    MAINTAIN = "maintain"
    DECREASE = "decrease"
    FREEZE = "freeze"


# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class VIXYHedgeConfig:
    """Configuration for VIXY hedge sizing."""
    min_hedge_pct: float = 0.0
    max_hedge_pct: float = 10.0
    cost_threshold: float = 2.0
    vixy_expense_ratio: float = 0.0085
    monthly_decay_pct: float = 0.05
    spy_shock_pct: float = -15.0
    ensemble_weight_normal: float = 0.05
    ensemble_weight_stress: float = 0.10


@dataclass
class VIXYHedgeSignal:
    """Hedge sizing signal for EnsembleVoter and rebalancing."""
    timestamp: str
    vix_level: float
    regime: str
    allocation_pct: float          # Recommended VIXY allocation (0-10%)
    action: str                    # increase / maintain / decrease / freeze
    signal_value: float            # -1 (reduce hedge) to +1 (increase hedge)
    confidence: float              # 0-1 based on VIX data freshness

    # Cost analysis
    annual_cost_bps: float         # Estimated annual cost in basis points
    monthly_decay_bps: float       # Estimated monthly premium decay

    # Benefit estimation
    expected_gain_shock: float     # Expected % gain during -15% SPY shock
    hedge_efficiency: float        # Benefit / cost ratio

    # Integration
    ensemble_weight: float         # Recommended weight in EnsembleVoter
    collar_complement: float       # Remaining collar hedge needed (0-5%)

    # Source
    source: str = "vixy_hedge"


@dataclass
class VIXYHedgeState:
    """Persistent state for VIXY hedge tracker."""
    timestamp: str
    current_allocation: float       # Current VIXY allocation %
    target_allocation: float        # Target VIXY allocation %
    vix_level: float
    regime: str
    ytd_cost_bps: float             # Year-to-date cost in bps
    ytd_benefit_bps: float          # Year-to-date benefit in bps
    hedge_efficiency: float          # Cumulative efficiency ratio
    total_signals: int = 0
    last_rebalance: Optional[str] = None


# ── VIXY Hedge Sizer ──────────────────────────────────────────────────────

class VIXYHedgeSizer:
    """
    Dynamic VIXY hedge sizing engine.

    Computes VIXY allocation proportional to VIX level, with regime-aware
    adjustments, cost tracking, and EnsembleVoter signal generation.
    """

    def __init__(self, config: Optional[Dict] = None, project_root: Optional[Path] = None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.config = VIXYHedgeConfig(
            min_hedge_pct=cfg["min_hedge_pct"],
            max_hedge_pct=cfg["max_hedge_pct"],
            cost_threshold=cfg["cost_threshold"],
            vixy_expense_ratio=cfg["vixy_expense_ratio"],
            monthly_decay_pct=cfg["monthly_decay_pct"],
            spy_shock_pct=cfg["spy_shock_pct"],
            ensemble_weight_normal=cfg["ensemble_weight_normal"],
            ensemble_weight_stress=cfg["ensemble_weight_stress"],
        )
        self._project_root = project_root or DATA_DIR.parent
        if project_root:
            self._state_file = self._project_root / cfg["state_file"]
        else:
            self._state_file = DATA_DIR / "vixy_hedge_state.json"
        self._state: Optional[VIXYHedgeState] = None

    # ── Regime Classification ──────────────────────────────────────────

    @staticmethod
    def classify_regime(vix_level: float) -> HedgeRegime:
        """Classify VIX level into hedge regime."""
        if vix_level < 20:
            return HedgeRegime.NORMAL
        elif vix_level < 30:
            return HedgeRegime.ELEVATED
        elif vix_level < 40:
            return HedgeRegime.STRESS
        else:
            return HedgeRegime.CRISIS

    # ── Core Allocation Logic ──────────────────────────────────────────

    def compute_allocation(self, vix_level: float) -> float:
        """
        Compute VIXY allocation using QuantPedia-style VIX/1000 model
        with regime-aware floor/ceiling.

        VIX=14 → 1.4%, VIX=28 → 2.8%, VIX=45 → 4.5% (capped at max)
        """
        # Base: VIX/1000 = allocation fraction → *100 for percentage
        # VIX=28 → 28/1000 = 2.8% allocation
        raw_allocation = vix_level / 10.0

        regime = self.classify_regime(vix_level)

        # Regime-aware floors and ceilings
        regime_bounds = {
            HedgeRegime.NORMAL:   (0.0, 2.0),    # VIX<20: 0-2%
            HedgeRegime.ELEVATED: (1.0, 3.5),    # VIX 20-30: 1-3.5%
            HedgeRegime.STRESS:   (2.0, 6.0),    # VIX 30-40: 2-6%
            HedgeRegime.CRISIS:   (3.0, self.config.max_hedge_pct),  # >40: 3-10%
        }

        floor, ceiling = regime_bounds[regime]

        # Apply regime bounds
        allocation = max(floor, min(ceiling, raw_allocation))

        # Never exceed global max
        allocation = min(allocation, self.config.max_hedge_pct)
        allocation = max(allocation, self.config.min_hedge_pct)

        return round(allocation, 2)

    def compute_allocation_with_vol_scale(self, vix_level: float, vol_ratio: float = 1.0) -> float:
        """
        Compute VIXY allocation with volatility scaling.
        When realized vol > VIX, scale down; when realized vol < VIX, scale up.
        """
        base = self.compute_allocation(vix_level)
        # Clip vol scaling to [0.5, 1.5] range
        scale = max(0.5, min(1.5, 1.0 / max(vol_ratio, 0.1)))
        return round(base * scale, 2)

    # ── Cost Analysis ──────────────────────────────────────────────────

    def estimate_annual_cost(self, allocation_pct: float) -> float:
        """Estimate annual cost in basis points for a given VIXY allocation."""
        # VIXY expense ratio + estimated roll yield drag + premium decay
        expense_cost = allocation_pct * self.config.vixy_expense_ratio * 100 * 100  # to bps
        decay_cost = allocation_pct * self.config.monthly_decay_pct * 12 * 100     # to bps
        return round(expense_cost + decay_cost, 1)

    def estimate_monthly_cost(self, allocation_pct: float) -> float:
        """Estimate monthly cost in basis points."""
        expense_monthly = allocation_pct * self.config.vixy_expense_ratio * 100 * 100 / 12
        decay_monthly = allocation_pct * self.config.monthly_decay_pct * 100
        return round(expense_monthly + decay_monthly, 1)

    # ── Benefit Estimation ─────────────────────────────────────────────

    def estimate_gain_during_shock(self, allocation_pct: float,
                                    spy_shock_pct: Optional[float] = None) -> float:
        """
        Estimate VIXY % gain during a SPY drawdown.
        VIXY historically gains ~40-60% during -15% SPY shocks.
        Uses conservative 40% beta estimate.
        """
        shock = spy_shock_pct or self.config.spy_shock_pct
        # VIXY beta to SPY during crises: roughly -3x to -5x (inverse)
        # Conservative estimate: |beta|=3.5, VIXY gains when SPY falls
        vixy_beta_magnitude = 3.5
        vixy_move = abs(shock) * vixy_beta_magnitude / 100.0  # fractional gain
        portfolio_protection = allocation_pct * vixy_move  # % of portfolio protected
        return round(portfolio_protection * 100, 1)  # in bps

    def compute_hedge_efficiency(self, allocation_pct: float,
                                  vix_level: float) -> float:
        """
        Compute hedge efficiency ratio: expected benefit / cost.
        Ratio > 1.0 means the hedge is cost-effective.
        """
        annual_cost = self.estimate_annual_cost(allocation_pct)
        expected_benefit = self.estimate_gain_during_shock(allocation_pct)

        if annual_cost < 0.01:
            return 0.0

        # Adjust benefit by probability of shock (VIX-implied)
        # Higher VIX = higher probability of large move
        shock_probability = min(0.5, max(0.02, vix_level / 1000.0))

        adjusted_benefit = expected_benefit * shock_probability * 12  # annualize
        efficiency = adjusted_benefit / annual_cost
        return round(efficiency, 2)

    # ── Signal Generation ──────────────────────────────────────────────

    def get_signal(self, vix_level: float,
                   vol_ratio: float = 1.0,
                   data_freshness: float = 1.0) -> VIXYHedgeSignal:
        """Generate full hedge sizing signal."""
        now = datetime.now().isoformat()
        regime = self.classify_regime(vix_level)
        allocation = self.compute_allocation_with_vol_scale(vix_level, vol_ratio)

        # Determine action
        prev_alloc = self._state.current_allocation if self._state else 0.0
        action = self._determine_action(allocation, prev_alloc, regime)

        # Signal value: -1 to +1
        if regime == HedgeRegime.CRISIS:
            signal_value = 1.0
        elif regime == HedgeRegime.STRESS:
            signal_value = 0.7
        elif regime == HedgeRegime.ELEVATED:
            signal_value = 0.3
        else:
            signal_value = 0.0

        # Costs
        annual_cost = self.estimate_annual_cost(allocation)
        monthly_cost = self.estimate_monthly_cost(allocation)
        expected_gain = self.estimate_gain_during_shock(allocation)
        efficiency = self.compute_hedge_efficiency(allocation, vix_level)

        # Ensemble weight
        if regime in (HedgeRegime.CRISIS, HedgeRegime.STRESS):
            ensemble_weight = self.config.ensemble_weight_stress
        else:
            ensemble_weight = self.config.ensemble_weight_normal

        # Collar complement: when VIXY provides heavy hedge, reduce collar
        if allocation >= 8.0:
            collar_complement = 0.0
        elif allocation >= 5.0:
            collar_complement = 1.0
        elif allocation >= 2.0:
            collar_complement = 2.0
        else:
            collar_complement = 3.0  # Full collar needed when VIXY minimal

        return VIXYHedgeSignal(
            timestamp=now,
            vix_level=vix_level,
            regime=regime.value,
            allocation_pct=allocation,
            action=action.value,
            signal_value=signal_value,
            confidence=data_freshness,
            annual_cost_bps=annual_cost,
            monthly_decay_bps=monthly_cost,
            expected_gain_shock=expected_gain,
            hedge_efficiency=efficiency,
            ensemble_weight=ensemble_weight,
            collar_complement=collar_complement,
        )

    def _determine_action(self, target: float, current: float,
                          regime: HedgeRegime) -> HedgeAction:
        """Determine whether to increase, maintain, or decrease hedge."""
        if regime == HedgeRegime.CRISIS:
            return HedgeAction.FREEZE  # Don't change during crisis

        delta = target - current
        if delta > 1.0:
            return HedgeAction.INCREASE
        elif delta < -0.5:
            return HedgeAction.DECREASE
        else:
            return HedgeAction.MAINTAIN

    # ── State Persistence ──────────────────────────────────────────────

    def load_state(self) -> VIXYHedgeState:
        """Load persistent hedge state from disk."""
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    data = json.load(f)
                self._state = VIXYHedgeState(**data)
                return self._state
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Corrupt state file: %s, using defaults", e)
        self._state = VIXYHedgeState(
            timestamp=datetime.now().isoformat(),
            current_allocation=0.0,
            target_allocation=0.0,
            vix_level=0.0,
            regime=HedgeRegime.NORMAL.value,
            ytd_cost_bps=0.0,
            ytd_benefit_bps=0.0,
            hedge_efficiency=0.0,
        )
        return self._state

    def save_state(self, signal: VIXYHedgeSignal):
        """Save hedge state after a signal update."""
        now = datetime.now().isoformat()
        prev = self._state or self.load_state()

        # YTD tracking with simple accumulation
        ytd_cost = prev.ytd_cost_bps + signal.monthly_decay_bps
        ytd_benefit = prev.ytd_benefit_bps  # Updated when shock actually occurs

        # Reset YTD in January
        if datetime.now().month == 1 and prev.last_rebalance:
            last_dt = datetime.fromisoformat(prev.last_rebalance)
            if last_dt.year < datetime.now().year:
                ytd_cost = signal.monthly_decay_bps
                ytd_benefit = 0.0

        self._state = VIXYHedgeState(
            timestamp=now,
            current_allocation=prev.current_allocation,
            target_allocation=signal.allocation_pct,
            vix_level=signal.vix_level,
            regime=signal.regime,
            ytd_cost_bps=round(ytd_cost, 1),
            ytd_benefit_bps=round(ytd_benefit, 1),
            hedge_efficiency=signal.hedge_efficiency,
            total_signals=prev.total_signals + 1,
            last_rebalance=prev.last_rebalance,
        )

        save_results_json(asdict(self._state), output_path=str(self._state_file))
        logger.info("VIXY hedge state saved: %s%% allocation", signal.allocation_pct)

    def update_after_rebalance(self, actual_allocation: float):
        """Update state after a rebalance executes."""
        self.load_state()
        self._state.current_allocation = actual_allocation
        self._state.last_rebalance = datetime.now().isoformat()
        save_results_json(asdict(self._state), output_path=str(self._state_file))

    # ── Status Report ─────────────────────────────────────────────────

    def status(self) -> Dict:
        """Return current hedge status as a dictionary."""
        state = self.load_state()
        return {
            "current_allocation_pct": state.current_allocation,
            "target_allocation_pct": state.target_allocation,
            "vix_level": state.vix_level,
            "regime": state.regime,
            "ytd_cost_bps": state.ytd_cost_bps,
            "ytd_benefit_bps": state.ytd_benefit_bps,
            "hedge_efficiency": state.hedge_efficiency,
            "total_signals": state.total_signals,
            "last_rebalance": state.last_rebalance,
        }

    # ── Collar Coordination ───────────────────────────────────────────

    def get_combined_hedge_coverage(self, vixy_allocation: float,
                                     collar_coverage: float = 0.0) -> str:
        """Report total hedge coverage combining VIXY + collar."""
        total = vixy_allocation + collar_coverage
        return (
            f"Total hedge: VIXY {vixy_allocation:.1f}% + "
            f"collar {collar_coverage:.1f}% = {total:.1f}% portfolio protection"
        )

    def should_disable_collar(self, vixy_allocation: float) -> bool:
        """When VIXY allocation is high enough, collar is unnecessary."""
        return vixy_allocation >= self.config.max_hedge_pct * 0.8


# ── CLI ────────────────────────────────────────────────────────────────────

def _format_status(sizer: VIXYHedgeSizer) -> str:
    """Format status output."""
    status = sizer.status()
    lines = [
        "=== VIXY Hedge Sizing Status ===",
        f"  Current allocation: {status['current_allocation_pct']:.1f}%",
        f"  Target allocation:  {status['target_allocation_pct']:.1f}%",
        f"  VIX level:          {status['vix_level']:.1f}",
        f"  Regime:             {status['regime']}",
        f"  YTD cost:           {status['ytd_cost_bps']:.1f} bps",
        f"  YTD benefit:        {status['ytd_benefit_bps']:.1f} bps",
        f"  Efficiency:         {status['hedge_efficiency']:.2f}x",
        f"  Total signals:      {status['total_signals']}",
        f"  Last rebalance:     {status['last_rebalance'] or 'never'}",
    ]
    return "\n".join(lines)


def _format_recommend(signal: VIXYHedgeSignal) -> str:
    """Format recommendation output."""
    lines = [
        "=== VIXY Hedge Recommendation ===",
        f"  VIX:               {signal.vix_level:.1f}",
        f"  Regime:            {signal.regime}",
        f"  Allocation:        {signal.allocation_pct:.1f}%",
        f"  Action:            {signal.action}",
        f"  Signal:            {signal.signal_value:+.2f}",
        f"  Annual cost:       {signal.annual_cost_bps:.1f} bps",
        f"  Monthly decay:     {signal.monthly_decay_bps:.1f} bps",
        f"  Expected gain:     {signal.expected_gain_shock:.1f} bps (in -15% shock)",
        f"  Efficiency:        {signal.hedge_efficiency:.2f}x",
        f"  Ensemble weight:   {signal.ensemble_weight:.0%}",
        f"  Collar complement: {signal.collar_complement:.1f}%",
    ]
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="VIXY Hedge Sizing v7.04")
    parser.add_argument("mode", nargs="?", default="status",
                       choices=["status", "recommend", "backtest", "update"])
    parser.add_argument("--vix", type=float, default=None,
                       help="VIX level (default: try to fetch from live data)")
    parser.add_argument("--save", action="store_true",
                       help="Save recommendation to state file")
    parser.add_argument("--start", type=str, default="2006-01-01",
                       help="Backtest start date")
    args = parser.parse_args()

    sizer = VIXYHedgeSizer()

    if args.mode == "status":
        logger.info(_format_status(sizer))

    elif args.mode == "recommend":
        # Try to get VIX from existing data or use provided value
        vix = args.vix
        if vix is None:
            try:
                from src.signals.vix_term_structure import VIXTermStructureGenerator
                gen = VIXTermStructureGenerator()
                spot = gen.get_vix_spot()
                vix = spot if spot else 18.0
            except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
                logger.warning("Failed to get VIX spot from term structure, defaulting to 18.0: %s", e)
                vix = 18.0  # Default for development

        signal = sizer.get_signal(vix)
        logger.info(_format_recommend(signal))

        if args.save:
            sizer.save_state(signal)
            logger.info("State saved to %s", sizer._state_file)

    elif args.mode == "update":
        vix = args.vix or 18.0
        signal = sizer.get_signal(vix)
        sizer.save_state(signal)
        logger.info("Updated: allocation=%d%% (VIX=%s)", signal.allocation_pct, vix)

    elif args.mode == "backtest":
        _run_backtest(sizer, args.start)


def _run_backtest(sizer: VIXYHedgeSizer, start_date: str):
    """Simple historical backtest of VIXY hedge strategy."""
    logger.info("=== VIXY Hedge Backtest (%s to present) ===", start_date)
    try:
        from src.signals.vix_term_structure import VIXTermStructureGenerator
        gen = VIXTermStructureGenerator()
        history = gen.get_vix_history()

        if not history:
            logger.warning("No VIX history available. Using simulated data.")
            # Simulate with realistic VIX distribution
            np.random.seed(42)
            dates = pd.date_range(start_date, datetime.now(), freq='D')
            vix_levels = np.random.lognormal(mean=2.8, sigma=0.4, size=len(dates))
            history = list(zip(dates, vix_levels))
        else:
            history = [(pd.Timestamp(d), v) for d, v in history if d >= start_date]

        allocations = []
        costs = []
        for date, vix in history[-2520:]:  # Last ~10 years
            alloc = sizer.compute_allocation(float(vix))
            allocations.append(alloc)
            costs.append(sizer.estimate_annual_cost(alloc))

        avg_alloc = np.mean(allocations)
        avg_cost = np.mean(costs)
        max_alloc = np.max(allocations)
        days_hedged = sum(1 for a in allocations if a > 0)

        logger.info("  Period:            %s to %s", history[0][0].date(), history[-1][0].date())
        logger.info("  Trading days:      %d", len(allocations))
        logger.info("  Avg allocation:    %.1f%%", avg_alloc)
        logger.info("  Max allocation:    %.1f%%", max_alloc)
        logger.info("  Days hedged:       %d (%d%%)", days_hedged, 100 * days_hedged // len(allocations))
        logger.info("  Avg annual cost:   %.1f bps", avg_cost)
        logger.info("  Hedge efficiency:  %.2fx", sizer.compute_hedge_efficiency(avg_alloc, 20))
    except ImportError:
        logger.warning("Backtest requires pandas and VIX history. Run 'recommend' for single-point analysis.")
    except (KeyError, ValueError, TypeError, ZeroDivisionError, AttributeError, RuntimeError) as e:
        logger.error("Backtest error: %s", e)


if __name__ == "__main__":
    from src.utils.log_config import configure_logging
    configure_logging()
    main()
