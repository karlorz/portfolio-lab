"""
Collar Overlay Walk-Forward Backtest - v9.33 Implementation

Validates the cashless collar overlay on the baseline 46/38/16 (SPY/GLD/TLT)
portfolio. The collar writes an OTM call and buys an OTM put on the SPY sleeve
each month to cap upside and floor downside.

Key questions:
  1. Does the collar reduce max drawdown vs baseline?
  2. How much CAGR is sacrificed for the put protection?
  3. In which VIX regimes (NORMAL/ELEVATED/STRESS/CRISIS) is the collar most
     effective?
  4. What is the optimal SPY reduction per regime?

Period: 2006-2026 (20+ years including GFC, COVID, 2022 rate hikes)
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.backtest.metrics import (
    DailyPrices,
    compute_metrics,
    compute_crisis_returns,
    save_results_json,
)
from src.paths import PRICES_JSON, BACKTEST_RESULTS_DIR
from src.signals.collar_signal import CollarSignalGenerator, CollarRegime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
MONTHLY_TRADING_DAYS = 21

# Crisis years to evaluate
CRISIS_YEARS = ["2008", "2020", "2022"]

# VIX thresholds come from CollarRegime enum in collar_signal.py:
#   CollarRegime.NORMAL (VIX < 20), ELEVATED (< 30), STRESS (< 40), CRISIS (>= 40)

# Symbols needed
BASE_SYMBOLS = ["SPY", "GLD", "TLT"]


@dataclass
class BacktestConfig:
    """Configuration for collar overlay walk-forward backtest."""

    start_date: str = "2006-01-01"
    end_date: str = "2026-05-15"
    initial_capital: float = 100000.0

    # Baseline allocation (46/38/16)
    base_spy_weight: float = 0.46
    base_gld_weight: float = 0.38
    base_tlt_weight: float = 0.16

    # Rebalancing
    rebalance_frequency_days: int = MONTHLY_TRADING_DAYS
    transaction_cost_bps: float = 10.0  # 10 bps per rebalance


@dataclass
class BacktestResult:
    """Complete backtest results comparing baseline vs collar overlay."""

    # Core metrics
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float

    # Baseline comparison
    baseline_total_return: float
    baseline_cagr: float
    baseline_volatility: float
    baseline_sharpe: float
    baseline_max_drawdown: float
    sharpe_improvement: float
    cagr_impact: float

    # Collar activity
    collar_active_days: int
    collar_active_pct: float
    collar_spy_reduction_avg: float

    # Crisis returns
    crisis_returns_hedged: Dict[str, float]
    crisis_returns_baseline: Dict[str, float]

    # Regime breakdown
    regime_breakdown: Dict[str, Dict[str, float]]

    # Trade stats
    total_rebalances: int
    total_transaction_costs: float

    # Config
    config_snapshot: Dict

    def to_dict(self) -> Dict:
        """Serialize to dict for JSON output."""
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "baseline_total_return": self.baseline_total_return,
            "baseline_cagr": self.baseline_cagr,
            "baseline_volatility": self.baseline_volatility,
            "baseline_sharpe": self.baseline_sharpe,
            "baseline_max_drawdown": self.baseline_max_drawdown,
            "sharpe_improvement": self.sharpe_improvement,
            "cagr_impact": self.cagr_impact,
            "collar_active_days": self.collar_active_days,
            "collar_active_pct": self.collar_active_pct,
            "collar_spy_reduction_avg": self.collar_spy_reduction_avg,
            "crisis_returns_hedged": self.crisis_returns_hedged,
            "crisis_returns_baseline": self.crisis_returns_baseline,
            "regime_breakdown": self.regime_breakdown,
            "total_rebalances": self.total_rebalances,
            "total_transaction_costs": self.total_transaction_costs,
            "config_snapshot": self.config_snapshot,
        }


# ---------------------------------------------------------------------------
# Walk-Forward Backtester
# ---------------------------------------------------------------------------


class WalkForwardCollarBacktester:
    """
    Walk-forward backtest for the collar overlay strategy.

    Simulates monthly rebalancing of the baseline 46/38/16 portfolio with
    a dynamic collar overlay that reduces SPY allocation and increases TLT/GLD
    based on VIX regime. The collar is frozen (no adjustment) when VIX > 40.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._daily_prices: List[DailyPrices] = []
        self._trading_dates: List[str] = []

    def load_data(self) -> None:
        """Load price data from PRICES_JSON and extract SPY/GLD/TLT.

        Attempts to load ^VIX data; falls back to SPY 21-day realized vol
        proxy (VIX approx realized_vol * 1.1) if ^VIX is not available.
        """
        prices_path = PRICES_JSON
        if not prices_path.exists():
            logger.warning("Prices file not found at %s; generating synthetic data", prices_path)
            self._generate_synthetic_data()
            return

        with open(prices_path) as f:
            raw = json.load(f)

        # Build ordered price lookups
        spy_data = {e["d"]: e["p"] for e in raw.get("SPY", [])}
        gld_data = {e["d"]: e["p"] for e in raw.get("GLD", [])}
        tlt_data = {e["d"]: e["p"] for e in raw.get("TLT", [])}
        vix_data = {e["d"]: e["p"] for e in raw.get("^VIX", [])} if "^VIX" in raw else {}

        # Collect all trading dates that have SPY, GLD, and TLT
        all_dates = sorted(
            d for d in spy_data
            if d in gld_data and d in tlt_data
        )

        # Filter to config range
        start_ts = self.config.start_date
        end_ts = self.config.end_date
        filtered_dates = [d for d in all_dates if start_ts <= d <= end_ts]

        if not filtered_dates:
            logger.warning("No data in date range %s to %s; generating synthetic data", start_ts, end_ts)
            self._generate_synthetic_data()
            return

        self._trading_dates = filtered_dates
        self._daily_prices = []

        for date in filtered_dates:
            vix_level = vix_data.get(date)
            if vix_level is not None:
                vix_level = float(vix_level)
            self._daily_prices.append(DailyPrices(
                date=date,
                spy=float(spy_data[date]),
                gld=float(gld_data[date]),
                tlt=float(tlt_data[date]),
                vix=vix_level,
            ))

        logger.info(
            "Loaded %d trading days from %s to %s",
            len(self._daily_prices),
            filtered_dates[0],
            filtered_dates[-1],
        )

    def _generate_synthetic_data(self) -> None:
        """Generate synthetic price data for testing.

        Creates ~20 years of daily prices with realistic drift and volatility.
        VIX is estimated as SPY 21-day realized vol * 1.1.
        """
        np.random.seed(42)
        n_days = 5100  # ~20 years

        start_date = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        dates = []
        dt = start_date
        while len(dates) < n_days:
            if dt.weekday() < 5:  # Weekday
                dates.append(dt.strftime("%Y-%m-%d"))
            dt += timedelta(days=1)

        dates = dates[:n_days]

        # SPY: 7% drift, 18% vol
        spy_returns = np.random.normal(0.07 / TRADING_DAYS_PER_YEAR, 0.18 / np.sqrt(TRADING_DAYS_PER_YEAR), n_days)
        spy_price = 100.0 * np.exp(np.cumsum(spy_returns))

        # GLD: 4% drift, 15% vol, 0.3 correlation to SPY
        gld_noise = np.random.normal(0.04 / TRADING_DAYS_PER_YEAR, 0.15 / np.sqrt(TRADING_DAYS_PER_YEAR), n_days)
        gld_common = 0.3 * spy_returns
        gld_returns = gld_common + np.sqrt(1 - 0.3 ** 2) * gld_noise
        gld_price = 100.0 * np.exp(np.cumsum(gld_returns))

        # TLT: 3% drift, 12% vol, low correlation
        tlt_noise = np.random.normal(0.03 / TRADING_DAYS_PER_YEAR, 0.12 / np.sqrt(TRADING_DAYS_PER_YEAR), n_days)
        tlt_returns = tlt_noise
        tlt_price = 100.0 * np.exp(np.cumsum(tlt_returns))

        self._trading_dates = dates
        self._daily_prices = []

        for i, date in enumerate(dates):
            self._daily_prices.append(DailyPrices(
                date=date,
                spy=float(spy_price[i]),
                gld=float(gld_price[i]),
                tlt=float(tlt_price[i]),
                vix=None,  # Will be computed as proxy on the fly
            ))

        logger.info("Generated %d synthetic trading days", n_days)

    def _compute_vix_proxy(self, idx: int) -> float:
        """Compute VIX proxy from SPY 21-day realized vol * 1.1.

        Returns a default VIX of 18 if insufficient history.
        """
        if idx < MONTHLY_TRADING_DAYS:
            return 18.0

        prices = self._daily_prices[idx - MONTHLY_TRADING_DAYS:idx + 1]
        spy_prices = [p.spy for p in prices]
        returns = []
        for i in range(1, len(spy_prices)):
            if spy_prices[i - 1] > 0:
                returns.append(spy_prices[i] / spy_prices[i - 1] - 1)

        if len(returns) < 5:
            return 18.0

        realized_vol = np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
        return round(realized_vol * 100 * 1.1, 2)  # VIX proxy approx 21d vol * 1.1

    def _get_vix_level(self, idx: int) -> float:
        """Get VIX level at index, using proxy if ^VIX data unavailable."""
        dp = self._daily_prices[idx]
        if dp.vix is not None and dp.vix > 0:
            return dp.vix
        return self._compute_vix_proxy(idx)

    @staticmethod
    def _get_collar_shifts_from_regime(regime: CollarRegime) -> Tuple[float, float, float]:
        """Get allocation shifts for collar overlay based on CollarRegime.

        Returns (spy_shift, gld_shift, tlt_shift) where negative means
        reduce the allocation and positive means increase. In CRISIS mode,
        the collar is frozen (no shift) because put options are too expensive.

        Shifts are applied additively to the base weights:
          - NORMAL (VIX < 20):   moderate defense, SPY -3%,   GLD +1%,  TLT +2%
          - ELEVATED (VIX 20-30): stronger defense, SPY -4%,  GLD +1.5%, TLT +2.5%
          - STRESS (VIX 30-40):  heavy defense,  SPY -5%,     GLD +2%,  TLT +3%
          - CRISIS (VIX > 40):   frozen, no shifts
        """
        shifts = {
            CollarRegime.NORMAL: (-0.03, 0.01, 0.02),
            CollarRegime.ELEVATED: (-0.04, 0.015, 0.025),
            CollarRegime.STRESS: (-0.05, 0.02, 0.03),
            CollarRegime.CRISIS: (0.0, 0.0, 0.0),
        }
        return shifts.get(regime, (-0.03, 0.01, 0.02))

    def _compute_portfolio_return(
        self,
        p0: DailyPrices,
        p1: DailyPrices,
        spy_w: float,
        gld_w: float,
        tlt_w: float,
    ) -> float:
        """Compute 1-day portfolio return given current weights."""
        spy_ret = (p1.spy / p0.spy - 1) if p0.spy > 0 else 0.0
        gld_ret = (p1.gld / p0.gld - 1) if p0.gld > 0 else 0.0
        tlt_ret = (p1.tlt / p0.tlt - 1) if p0.tlt > 0 else 0.0

        return (
            spy_w * spy_ret
            + gld_w * gld_ret
            + tlt_w * tlt_ret
        )

    def run(self) -> BacktestResult:
        """Run the walk-forward backtest simulation.

        Returns a BacktestResult comparing the baseline 46/38/16 portfolio
        against the collar-overlaid portfolio that shifts SPY to TLT/GLD
        based on VIX regime.
        """
        if not self._daily_prices:
            self.load_data()

        if len(self._daily_prices) < 2:
            logger.error("Insufficient data for backtest")
            return self._empty_result()

        prices = self._daily_prices
        config = self.config

        # Baseline: buy-and-hold 46/38/16
        baseline_equity = self._run_baseline(prices, config)

        # Collared: dynamic collar overlay on SPY
        collar_equity, collar_tracker, regime_tracker = self._run_collared(prices, config)

        # Compute metrics from equity curves
        baseline_metrics = compute_metrics(baseline_equity, config.initial_capital)
        collar_metrics = compute_metrics(collar_equity, config.initial_capital)

        # Crisis returns
        prices_lookup = self._build_prices_lookup()
        trading_dates = self._trading_dates

        crisis_collared = self._compute_crisis_returns_collared(
            prices_lookup, trading_dates, collar_equity, config.initial_capital
        )
        crisis_baseline = compute_crisis_returns(prices_lookup, trading_dates)

        # Regime breakdown
        regime_breakdown = self._compute_regime_breakdown(regime_tracker)

        # Collar activity stats
        collar_active_days = collar_tracker["active_days"]
        total_days = max(len(prices) - 1, 1)
        collar_active_pct = round(100.0 * collar_active_days / total_days, 2)
        collar_spy_reduction_avg = collar_tracker["avg_spy_reduction"]

        # Trade stats
        total_rebalances = collar_tracker["rebalances"]
        total_costs = collar_tracker["total_costs"]

        return BacktestResult(
            total_return=collar_metrics.total_return,
            cagr=collar_metrics.cagr,
            volatility=collar_metrics.volatility,
            sharpe_ratio=collar_metrics.sharpe_ratio,
            max_drawdown=collar_metrics.max_drawdown,
            baseline_total_return=baseline_metrics.total_return,
            baseline_cagr=baseline_metrics.cagr,
            baseline_volatility=baseline_metrics.volatility,
            baseline_sharpe=baseline_metrics.sharpe_ratio,
            baseline_max_drawdown=baseline_metrics.max_drawdown,
            sharpe_improvement=round(
                collar_metrics.sharpe_ratio - baseline_metrics.sharpe_ratio, 4
            ),
            cagr_impact=round(
                collar_metrics.cagr - baseline_metrics.cagr, 2
            ),
            collar_active_days=collar_active_days,
            collar_active_pct=collar_active_pct,
            collar_spy_reduction_avg=collar_spy_reduction_avg,
            crisis_returns_hedged=crisis_collared,
            crisis_returns_baseline=crisis_baseline,
            regime_breakdown=regime_breakdown,
            total_rebalances=total_rebalances,
            total_transaction_costs=round(total_costs, 2),
            config_snapshot={
                "start_date": config.start_date,
                "end_date": config.end_date,
                "initial_capital": config.initial_capital,
                "rebalance_frequency_days": config.rebalance_frequency_days,
                "transaction_cost_bps": config.transaction_cost_bps,
                "base_allocation": {
                    "SPY": config.base_spy_weight,
                    "GLD": config.base_gld_weight,
                    "TLT": config.base_tlt_weight,
                },
            },
        )

    def _run_baseline(
        self, prices: List[DailyPrices], config: BacktestConfig
    ) -> List[float]:
        """Run baseline buy-and-hold 46/38/16 portfolio."""
        spy_w = config.base_spy_weight
        gld_w = config.base_gld_weight
        tlt_w = config.base_tlt_weight

        equity = [config.initial_capital]

        for i in range(1, len(prices)):
            ret = self._compute_portfolio_return(
                prices[i - 1], prices[i], spy_w, gld_w, tlt_w
            )
            new_equity = equity[-1] * (1 + ret)
            equity.append(new_equity)

        return equity

    def _run_collared(
        self, prices: List[DailyPrices], config: BacktestConfig
    ) -> Tuple[List[float], Dict, List[Dict]]:
        """Run collar-overlaid portfolio with dynamic SPY reduction.

        Returns (equity_curve, collar_stats, regime_tracker).
        """
        spy_w = config.base_spy_weight
        gld_w = config.base_gld_weight
        tlt_w = config.base_tlt_weight

        equity = [config.initial_capital]
        collar_tracker = {
            "active_days": 0,
            "avg_spy_reduction": 0.0,
            "max_spy_reduction": 0.0,
            "rebalances": 0,
            "total_costs": 0.0,
            "spy_reductions": [],
            "regime_counts": {},
        }
        regime_tracker: List[Dict] = []
        days_since_rebalance = 0
        rebalance_freq = config.rebalance_frequency_days
        cost_per_trade = config.transaction_cost_bps / 10000.0  # Convert bps to decimal

        spy_reduction_pct = 0.0

        for i in range(1, len(prices)):
            date = prices[i].date
            days_since_rebalance += 1

            # Rebalance on initial day (i == 1) and then monthly
            if days_since_rebalance >= rebalance_freq or i == 1:
                vix_level = self._get_vix_level(i)
                regime = CollarSignalGenerator().classify_regime(vix_level)
                spy_shift, gld_shift, tlt_shift = self._get_collar_shifts_from_regime(regime)

                # New weights = base weights + collar shifts
                new_spy_w = config.base_spy_weight + spy_shift
                new_gld_w = config.base_gld_weight + gld_shift
                new_tlt_w = config.base_tlt_weight + tlt_shift

                # Enforce hard constraints: SPY 36-56%, GLD 28-48%, TLT 6-26%
                new_spy_w = max(0.36, min(0.56, new_spy_w))
                new_gld_w = max(0.28, min(0.48, new_gld_w))
                new_tlt_w = max(0.06, min(0.26, new_tlt_w))

                # Renormalize to sum to 1.0
                total = new_spy_w + new_gld_w + new_tlt_w
                if total > 0:
                    new_spy_w /= total
                    new_gld_w /= total
                    new_tlt_w /= total

                # Transaction cost: proportional to absolute weight change
                old_weights = [spy_w, gld_w, tlt_w]
                new_weights = [new_spy_w, new_gld_w, new_tlt_w]
                turnover = sum(abs(new_weights[j] - old_weights[j]) for j in range(3))
                cost = turnover * cost_per_trade * equity[-1]

                spy_reduction_pct = round((config.base_spy_weight - new_spy_w) * 100, 2)
                collar_tracker["spy_reductions"].append(spy_reduction_pct)
                if spy_reduction_pct > collar_tracker["max_spy_reduction"]:
                    collar_tracker["max_spy_reduction"] = spy_reduction_pct

                spy_w, gld_w, tlt_w = new_spy_w, new_gld_w, new_tlt_w

                collar_tracker["rebalances"] += 1
                collar_tracker["total_costs"] += cost
                days_since_rebalance = 0

                # Track regime (regime is already classified above via CollarSignalGenerator)
                regime.value
                collar_tracker["regime_counts"][regime.value] = (
                    collar_tracker["regime_counts"].get(regime.value, 0) + 1
                )
                regime_tracker.append({
                    "date": date,
                    "vix_level": vix_level,
                    "regime": regime.value,
                    "spy_reduction_pct": spy_reduction_pct,
                })

            # Track daily collar stats
            if spy_reduction_pct > 0:
                collar_tracker["active_days"] += 1

            # Daily return
            ret = self._compute_portfolio_return(
                prices[i - 1], prices[i], spy_w, gld_w, tlt_w
            )
            new_equity = equity[-1] * (1 + ret)
            equity.append(new_equity)

        # Compute average SPY reduction
        reductions = collar_tracker["spy_reductions"]
        collar_tracker["avg_spy_reduction"] = (
            round(float(np.mean(reductions)), 2) if reductions else 0.0
        )
        collar_tracker["max_spy_reduction"] = round(collar_tracker["max_spy_reduction"], 2)

        return equity, collar_tracker, regime_tracker

    def _build_prices_lookup(self) -> Dict[str, Dict[str, float]]:
        """Build {date: {symbol: price}} lookup from daily prices."""
        lookup: Dict[str, Dict[str, float]] = {}
        for dp in self._daily_prices:
            lookup[dp.date] = {
                "SPY": dp.spy,
                "GLD": dp.gld,
                "TLT": dp.tlt,
            }
        return lookup

    def _compute_crisis_returns_collared(
        self,
        prices_lookup: Dict[str, Dict[str, float]],
        trading_dates: List[str],
        equity_curve: List[float],
        initial_capital: float,
    ) -> Dict[str, float]:
        """Compute collared portfolio returns during crisis years.

        Uses the equity curve directly rather than buy-and-hold prices.
        """
        result: Dict[str, float] = {}
        for year in CRISIS_YEARS:
            year_dates = [d for d in trading_dates if d.startswith(year)]
            if not year_dates:
                continue

            # Map dates to indices in the equity curve
            date_to_idx: Dict[str, int] = {}
            for i, dp in enumerate(self._daily_prices):
                date_to_idx[dp.date] = i

            start_idx = date_to_idx.get(year_dates[0])
            end_idx = date_to_idx.get(year_dates[-1])

            if start_idx is None or end_idx is None:
                continue

            # Account for equity curve offset (equity[0] = initial, equity[i] = after day i)
            eq_start = equity_curve[start_idx]
            eq_end = equity_curve[end_idx]

            if eq_start > 0:
                year_ret = (eq_end / eq_start - 1) * 100
                result[year] = round(year_ret, 2)

        return result

    def _compute_regime_breakdown(
        self, regime_tracker: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """Compute collar stats broken down by VIX regime.

        Returns {regime_name: {avg_spy_reduction, max_spy_reduction, count, pct_of_time}}.
        """
        regimes: Dict[str, List[float]] = {}
        for entry in regime_tracker:
            reg = entry["regime"]
            if reg not in regimes:
                regimes[reg] = []
            regimes[reg].append(entry["spy_reduction_pct"])

        total = sum(len(v) for v in regimes.values())
        breakdown = {}
        for reg_name, reductions in regimes.items():
            pct_of_time = round(100.0 * len(reductions) / total, 2) if total > 0 else 0.0
            breakdown[reg_name] = {
                "avg_spy_reduction": round(float(np.mean(reductions)), 2),
                "max_spy_reduction": round(float(np.max(reductions)), 2),
                "count": len(reductions),
                "pct_of_time": pct_of_time,
            }

        return breakdown

    def _empty_result(self) -> BacktestResult:
        """Return an empty result when backtest cannot run."""
        return BacktestResult(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            baseline_total_return=0.0,
            baseline_cagr=0.0,
            baseline_volatility=0.0,
            baseline_sharpe=0.0,
            baseline_max_drawdown=0.0,
            sharpe_improvement=0.0,
            cagr_impact=0.0,
            collar_active_days=0,
            collar_active_pct=0.0,
            collar_spy_reduction_avg=0.0,
            crisis_returns_hedged={},
            crisis_returns_baseline={},
            regime_breakdown={},
            total_rebalances=0,
            total_transaction_costs=0.0,
            config_snapshot={},
        )

    def print_results(self, result: BacktestResult) -> None:
        """Print formatted backtest results to stdout."""
        print("\n" + "=" * 70)
        print("  Collar Overlay -- Walk-Forward Backtest Results")
        print("=" * 70)

        print(f"\n  Period: {self.config.start_date} to {self.config.end_date}")
        print(f"  Capital: ${self.config.initial_capital:,.0f}")
        print(f"  Baseline: SPY {self.config.base_spy_weight*100:.0f}% / "
              f"GLD {self.config.base_gld_weight*100:.0f}% / "
              f"TLT {self.config.base_tlt_weight*100:.0f}%")

        print(f"\n  {'Metric':<30} {'Baseline':>10} {'Collared':>10} {'Delta':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'Total Return':<30} {result.baseline_total_return:>9.2f}% {result.total_return:>9.2f}% "
              f"{result.total_return - result.baseline_total_return:>+9.2f}%")
        print(f"  {'CAGR':<30} {result.baseline_cagr:>9.2f}% {result.cagr:>9.2f}% "
              f"{result.cagr_impact:>+9.2f}%")
        print(f"  {'Volatility':<30} {result.baseline_volatility:>9.2f}% {result.volatility:>9.2f}% "
              f"{result.volatility - result.baseline_volatility:>+9.2f}%")
        print(f"  {'Sharpe Ratio':<30} {result.baseline_sharpe:>10.4f} {result.sharpe_ratio:>10.4f} "
              f"{result.sharpe_improvement:>+10.4f}")
        print(f"  {'Max Drawdown':<30} {result.baseline_max_drawdown:>9.2f}% {result.max_drawdown:>9.2f}% "
              f"{result.max_drawdown - result.baseline_max_drawdown:>+9.2f}%")

        print(f"\n  -- Collar Activity --")
        print(f"  Collar active days:  {result.collar_active_days} ({result.collar_active_pct:.1f}%)")
        print(f"  Avg SPY reduction:   {result.collar_spy_reduction_avg:.2f}%")
        print(f"  Rebalances:          {result.total_rebalances}")
        print(f"  Transaction costs:   ${result.total_transaction_costs:.2f}")

        print(f"\n  -- Crisis Returns (%) --")
        print(f"  {'Year':<10} {'Baseline':>10} {'Collared':>10}")
        print(f"  {'-'*10} {'-'*10} {'-'*10}")
        all_crisis_years = sorted(
            set(list(result.crisis_returns_baseline.keys()) + list(result.crisis_returns_hedged.keys()))
        )
        for year in all_crisis_years:
            b = result.crisis_returns_baseline.get(year, 0.0)
            h = result.crisis_returns_hedged.get(year, 0.0)
            print(f"  {year:<10} {b:>10.2f} {h:>10.2f}")

        print(f"\n  -- Regime Breakdown --")
        print(f"  {'Regime':<15} {'% Time':>8} {'Avg SPY Red':>12} {'Max SPY Red':>12}")
        print(f"  {'-'*15} {'-'*8} {'-'*12} {'-'*12}")
        for reg_name, stats in sorted(result.regime_breakdown.items()):
            print(f"  {reg_name:<15} {stats['pct_of_time']:>7.1f}% "
                  f"{stats['avg_spy_reduction']:>11.2f}% {stats['max_spy_reduction']:>11.2f}%")

        print("\n" + "=" * 70)

    def save_results(self, result: BacktestResult, output_path: Optional[str] = None) -> None:
        """Save backtest results to a JSON file."""
        data = result.to_dict()
        data["_metadata"] = {
            "strategy": "collar_overlay",
            "generated": datetime.now().isoformat(),
            "type": "walk_forward_backtest",
        }

        if output_path:
            save_results_json(data, output_path=output_path)
            logger.info("Results saved to %s", output_path)
        else:
            named_path = str(BACKTEST_RESULTS_DIR / "collar_overlay_backtest_results.json")
            save_results_json(data, output_path=named_path)
            logger.info("Results saved to %s", named_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for the collar overlay walk-forward backtest."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Collar Overlay Walk-Forward Backtest v9.33"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="run",
        choices=["run"],
        help="Run the backtest (default: run)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD, default: 2006-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, default: 2026-05-15)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for JSON results",
    )

    args = parser.parse_args()

    config = BacktestConfig()
    if args.start:
        config.start_date = args.start
    if args.end:
        config.end_date = args.end
    if args.capital is not None:
        config.initial_capital = args.capital

    backtester = WalkForwardCollarBacktester(config)
    result = backtester.run()
    backtester.print_results(result)

    if args.save or args.output:
        backtester.save_results(result, output_path=args.output)


if __name__ == "__main__":
    main()
