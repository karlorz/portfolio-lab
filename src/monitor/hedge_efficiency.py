"""
Hedge Efficiency Monitor - v7.04 Cost-Benefit Analysis
Tracks VIXY hedge performance during drawdowns, computes running efficiency,
and provides historical comparison with other hedging strategies.

Usage:
    python -m src.monitor.hedge_efficiency status
    python -m src.monitor.hedge_efficiency report
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HedgeComparison(Enum):
    """Hedge strategy comparison labels."""
    VIXY = "vixy"
    COLLAR = "collar"
    TREND_FOLLOWING = "trend_following"
    CASH = "cash"


@dataclass
class DrawdownEvent:
    """A market drawdown event with hedge performance."""
    start_date: str
    end_date: str
    spy_drawdown_pct: float          # SPY decline %
    vixy_gain_pct: float              # VIXY gain during this period
    portfolio_protected_pct: float    # % of portfolio protected
    hedge_efficiency: float           # Benefit / cost for this event


@dataclass
class HedgeEfficiencyReport:
    """Full efficiency report for a hedging strategy."""
    timestamp: str
    strategy: str
    current_allocation: float

    # Running metrics
    ytd_cost_bps: float
    ytd_benefit_bps: float
    running_efficiency: float          # Cumulative benefit / cost

    # Historical drawdown analysis
    recent_drawdowns: List[Dict]
    avg_protection_pct: float
    max_protection_pct: float

    # Comparison
    vs_collar_efficiency: float
    vs_trend_efficiency: float
    vs_cash_efficiency: float

    # Recommendation
    recommendation: str
    efficiency_grade: str              # A-F


class HedgeEfficiencyMonitor:
    """
    Monitors and reports hedge cost-benefit efficiency.

    Tracks VIXY performance during market drawdowns, computes running
    hedge efficiency scores, and compares against alternative strategies.
    """

    def __init__(self, project_root: Optional[Path] = None):
        self._project_root = project_root or Path(__file__).resolve().parent.parent.parent
        self._state_file = self._project_root / "data" / "hedge_efficiency_state.json"

    # ── Drawdown Detection ────────────────────────────────────────────

    def detect_drawdowns(self, spy_returns: List[float],
                         threshold_pct: float = -5.0) -> List[Tuple[int, int, float]]:
        """
        Detect drawdown events from daily SPY returns.
        Returns list of (start_idx, end_idx, peak_to_trough_pct).
        """
        if not spy_returns:
            return []

        cumulative = np.cumprod(1 + np.array(spy_returns) / 100.0)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak * 100

        events = []
        in_drawdown = False
        start_idx = 0
        max_dd = 0.0

        for i, dd in enumerate(drawdown):
            if dd <= threshold_pct and not in_drawdown:
                in_drawdown = True
                start_idx = i
                max_dd = dd
            elif in_drawdown:
                if dd < max_dd:
                    max_dd = dd
                if dd > -2.0:  # Recovery threshold
                    events.append((start_idx, i, max_dd))
                    in_drawdown = False
                    max_dd = 0.0

        # Capture ongoing drawdown
        if in_drawdown:
            events.append((start_idx, len(drawdown) - 1, max_dd))

        return events

    # ── Hedge Efficiency Computation ──────────────────────────────────

    def compute_event_efficiency(self, spy_drawdown_pct: float,
                                  vixy_gain_pct: float,
                                  allocation_pct: float,
                                  annual_cost_bps: float,
                                  event_days: int) -> float:
        """
        Compute hedge efficiency for a single drawdown event.
        Returns benefit/cost ratio.
        """
        # Benefit: portfolio % protected during this event
        portfolio_protection = allocation_pct * vixy_gain_pct / 100.0 * 100  # bps

        # Cost: prorated annual cost for event duration
        event_cost = annual_cost_bps * event_days / 365.0

        if event_cost < 0.01:
            return 0.0

        return round(portfolio_protection / event_cost, 2)

    def compute_running_efficiency(self, allocation_pct: float,
                                    vix_level: float,
                                    ytd_cost_bps: float,
                                    ytd_benefit_bps: float) -> float:
        """Compute running efficiency from YTD metrics."""

        if ytd_cost_bps < 0.01:
            return 0.0

        return round(ytd_benefit_bps / ytd_cost_bps, 2)

    # ── Strategy Comparison ───────────────────────────────────────────

    def compare_strategies(self, vixy_efficiency: float,
                           allocation_pct: float) -> Dict[str, float]:
        """
        Compare VIXY hedge against collar, trend-following, and cash.
        Uses historical benchmarks for comparison.
        """
        # Historical average efficiencies from backtests
        benchmarks = {
            "collar": 1.8,           # Collar typically 1.5-2.0x
            "trend_following": 2.2,  # Trend following 1.8-2.5x
            "cash": 0.0,             # No hedge = no cost, no protection
        }

        return {
            "vixy": round(vixy_efficiency, 2),
            "collar": benchmarks["collar"],
            "trend_following": benchmarks["trend_following"],
            "cash": benchmarks["cash"],
            "vixy_rank": self._rank_efficiency(vixy_efficiency, benchmarks),
        }

    @staticmethod
    def _rank_efficiency(vixy_eff: float,
                          benchmarks: Dict[str, float]) -> int:
        """Rank VIXY efficiency among strategies (1=best)."""
        all_effs = [vixy_eff] + list(benchmarks.values())
        sorted_effs = sorted(all_effs, reverse=True)
        return sorted_effs.index(vixy_eff) + 1

    # ── Efficiency Grade ──────────────────────────────────────────────

    @staticmethod
    def grade_efficiency(efficiency: float) -> str:
        """Assign letter grade to hedge efficiency."""
        if efficiency >= 2.0:
            return "A"    # Excellent — hedge pays for itself 2x+
        elif efficiency >= 1.5:
            return "B"    # Good — hedge provides meaningful net benefit
        elif efficiency >= 1.0:
            return "C"    # Marginal — hedge roughly breaks even
        elif efficiency >= 0.5:
            return "D"    # Poor — hedge costs more than it protects
        else:
            return "F"    # Failing — hedge is destroying value

    # ── Full Report Generation ────────────────────────────────────────

    def generate_report(self, allocation_pct: float,
                        vix_level: float,
                        ytd_cost_bps: float = 0.0,
                        ytd_benefit_bps: float = 0.0,
                        spy_returns: Optional[List[float]] = None,
                        vixy_returns: Optional[List[float]] = None,
                        event_dates: Optional[List[str]] = None,
                        ) -> HedgeEfficiencyReport:
        """Generate a full hedge efficiency report."""
        now = datetime.now().isoformat()
        running_eff = self.compute_running_efficiency(
            allocation_pct, vix_level, ytd_cost_bps, ytd_benefit_bps
        )

        # Detect and analyze drawdown events
        recent_drawdowns = []
        if spy_returns and vixy_returns:
            events = self.detect_drawdowns(spy_returns)
            for start, end, dd_pct in events[-5:]:  # Last 5 drawdowns
                event_days = end - start + 1
                # Cumulative VIXY return during event
                vixy_gain = float(np.prod(
                    1 + np.array(vixy_returns[start:end+1]) / 100.0
                ) - 1) * 100
                protection = allocation_pct * vixy_gain / 100.0 * 100

                event_eff = self.compute_event_efficiency(
                    abs(dd_pct), vixy_gain, allocation_pct,
                    ytd_cost_bps, event_days
                )

                start_date = event_dates[start] if event_dates else f"t+{start}"
                end_date = event_dates[end] if event_dates else f"t+{end}"

                recent_drawdowns.append({
                    "start": start_date,
                    "end": end_date,
                    "spy_drawdown_pct": round(dd_pct, 1),
                    "vixy_gain_pct": round(vixy_gain, 1),
                    "portfolio_protected_bps": round(protection, 1),
                    "efficiency": round(event_eff, 2),
                })

        # Protection stats
        protections = [d["portfolio_protected_bps"] for d in recent_drawdowns]
        avg_protection = float(np.mean(protections)) if protections else 0.0
        max_protection = float(np.max(protections)) if protections else 0.0

        # Comparison
        comparison = self.compare_strategies(running_eff, allocation_pct)

        # Recommendation
        grade = self.grade_efficiency(running_eff)
        if grade in ("A", "B"):
            recommendation = "Hedge is cost-effective. Maintain or increase allocation."
        elif grade == "C":
            recommendation = "Hedge is marginally effective. Monitor costs closely."
        else:
            recommendation = "Hedge is not cost-effective. Consider reducing allocation or switching strategy."

        return HedgeEfficiencyReport(
            timestamp=now,
            strategy="VIXY Dynamic Hedge",
            current_allocation=allocation_pct,
            ytd_cost_bps=round(ytd_cost_bps, 1),
            ytd_benefit_bps=round(ytd_benefit_bps, 1),
            running_efficiency=round(running_eff, 2),
            recent_drawdowns=recent_drawdowns,
            avg_protection_pct=round(avg_protection, 1),
            max_protection_pct=round(max_protection, 1),
            vs_collar_efficiency=round(comparison["collar"], 2),
            vs_trend_efficiency=round(comparison["trend_following"], 2),
            vs_cash_efficiency=round(comparison["cash"], 2),
            recommendation=recommendation,
            efficiency_grade=grade,
        )

    # ── Dashboard Stats ───────────────────────────────────────────────

    def get_dashboard_stats(self) -> Dict:
        """Return dashboard-friendly stats."""
        state = self._load_state()
        return {
            "current_allocation_pct": state.get("current_allocation", 0.0),
            "ytd_cost_bps": state.get("ytd_cost_bps", 0.0),
            "max_benefit_bps": state.get("max_benefit_bps", 0.0),
            "efficiency_score": state.get("efficiency_score", 0.0),
            "efficiency_grade": self.grade_efficiency(
                state.get("efficiency_score", 0.0)
            ),
            "last_updated": state.get("timestamp", ""),
        }

    def _load_state(self) -> Dict:
        """Load persisted efficiency state."""
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    def save_state(self, report: HedgeEfficiencyReport):
        """Persist efficiency report."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hedge Efficiency Monitor v7.04")
    parser.add_argument("mode", nargs="?", default="status",
                       choices=["status", "report"])
    parser.add_argument("--allocation", type=float, default=None,
                       help="Current VIXY allocation %")
    parser.add_argument("--vix", type=float, default=None,
                       help="Current VIX level")
    args = parser.parse_args()

    monitor = HedgeEfficiencyMonitor()

    if args.mode == "status":
        stats = monitor.get_dashboard_stats()
        print("=== Hedge Efficiency Status ===")
        print(f"  Allocation:  {stats['current_allocation_pct']:.1f}%")
        print(f"  YTD cost:    {stats['ytd_cost_bps']:.1f} bps")
        print(f"  Max benefit: {stats['max_benefit_bps']:.1f} bps")
        print(f"  Efficiency:  {stats['efficiency_score']:.2f}x")
        print(f"  Grade:       {stats['efficiency_grade']}")
        print(f"  Updated:     {stats['last_updated']}")

    elif args.mode == "report":
        alloc = args.allocation or 0.0
        vix = args.vix or 18.0
        report = monitor.generate_report(alloc, vix)
        print("=== Hedge Efficiency Report ===")
        print(f"  Strategy:      {report.strategy}")
        print(f"  Allocation:    {report.current_allocation:.1f}%")
        print(f"  YTD cost:      {report.ytd_cost_bps:.1f} bps")
        print(f"  YTD benefit:   {report.ytd_benefit_bps:.1f} bps")
        print(f"  Efficiency:    {report.running_efficiency:.2f}x")
        print(f"  Grade:         {report.efficiency_grade}")
        print(f"  Avg protection:{report.avg_protection_pct:.1f} bps")
        print(f"  vs Collar:     {report.vs_collar_efficiency:.2f}x")
        print(f"  vs Trend:      {report.vs_trend_efficiency:.2f}x")
        print(f"  Recommend:     {report.recommendation}")
        if report.recent_drawdowns:
            print(f"  Recent events: {len(report.recent_drawdowns)}")
            for dd in report.recent_drawdowns:
                print(f"    {dd['start']} → {dd['end']}: SPY {dd['spy_drawdown_pct']:.1f}%, "
                      f"VIXY {dd['vixy_gain_pct']:+.1f}%, "
                      f"eff {dd['efficiency']:.2f}x")


if __name__ == "__main__":
    main()
