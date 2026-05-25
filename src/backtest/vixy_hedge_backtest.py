"""
VIXY Hedge Sizing Walk-Forward Backtest - v9.32 Implementation

Validates the VIXY hedge overlay on the baseline 46/38/16 (SPY/GLD/TLT)
portfolio. The hedge dynamically allocates to VIXY based on VIX level using
the QuantPedia-style VIX/10 model, funded from the SPY sleeve.

Key questions:
  1. Does dynamic VIXY hedge reduce max drawdown vs baseline?
  2. How much does the hedge cost in CAGR?
  3. Is there a regime (NORMAL/ELEVATED/STRESS/CRISIS) where the hedge is
     most effective?
  4. What is the optimal hedge cap?

Period: 2006-2026 (20+ years including GFC, COVID, 2022 rate hikes)
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.backtest.metrics import (
    BacktestConfig as _BaseConfig,
    BacktestResult,
    DailyPrices,
    compute_metrics,
    compute_crisis_returns,
    save_results_json,
)
from src.paths import PRICES_JSON, BACKTEST_RESULTS_DIR
from src.strategy.vixy_hedge_sizing import VIXYHedgeSizer


__all__ = ['TRADING_DAYS_PER_YEAR', 'MONTHLY_TRADING_DAYS', 'CRISIS_YEARS', 'BASE_SYMBOLS', 'VIX_SYMBOL', 'BacktestConfig', 'WalkForwardVIXYBacktester']

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
MONTHLY_TRADING_DAYS = 21

# Crisis years to evaluate
CRISIS_YEARS = ["2008", "2020", "2022"]

# Symbols needed
BASE_SYMBOLS = ["SPY", "GLD", "TLT"]
VIX_SYMBOL = "^VIX"


@dataclass
class BacktestConfig(_BaseConfig):
    """VIXY hedge backtest config — inherits core fields from metrics.BacktestConfig."""

    # VIXY hedge constraints
    max_hedge_pct: float = 6.0  # Hard cap on VIXY allocation



def _result_to_dict(result: BacktestResult) -> Dict:
    """Serialize BacktestResult (canonical + extras) to dict for JSON output."""
    d = {
        "total_return": result.total_return,
        "cagr": result.cagr,
        "volatility": result.volatility,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "total_rebalances": result.total_rebalances,
        "total_transaction_costs": result.total_transaction_costs,
    }
    if result.baseline_sharpe is not None:
        d["baseline_sharpe"] = result.baseline_sharpe
    if result.sharpe_improvement is not None:
        d["sharpe_improvement"] = result.sharpe_improvement
    if result.crisis_returns is not None:
        d["crisis_returns"] = result.crisis_returns
    d.update(result.extras)
    return d


# ---------------------------------------------------------------------------
# Walk-Forward Backtester
# ---------------------------------------------------------------------------


class WalkForwardVIXYBacktester:
    """
    Walk-forward backtest for the VIXY hedge sizing strategy.

    Simulates monthly rebalancing of the baseline 46/38/16 portfolio with
    a dynamic VIXY hedge funded from the SPY sleeve. The VIXY allocation
    is determined by VixyHedgeSizer.compute_allocation() using VIX level.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.sizer = VIXYHedgeSizer()
        self._daily_prices: List[DailyPrices] = []
        self._trading_dates: List[str] = []

    def load_data(self) -> None:
        """Load price data from PRICES_JSON and extract SPY/GLD/TLT.

        Attempts to load ^VIX data; falls back to SPY 21-day realized vol
        proxy (VIX ≈ realized_vol * 1.1) if ^VIX is not available.
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
        return round(realized_vol * 100 * 1.1, 2)  # VIX proxy ≈ 21d vol * 1.1

    def _get_vix_level(self, idx: int) -> float:
        """Get VIX level at index, using proxy if ^VIX data unavailable."""
        dp = self._daily_prices[idx]
        if dp.vix is not None and dp.vix > 0:
            return dp.vix
        return self._compute_vix_proxy(idx)

    def _compute_portfolio_return(
        self,
        p0: DailyPrices,
        p1: DailyPrices,
        spy_w: float,
        gld_w: float,
        tlt_w: float,
        vixy_w: float,
    ) -> float:
        """Compute 1-day portfolio return given current weights."""
        spy_ret = (p1.spy / p0.spy - 1) if p0.spy > 0 else 0.0
        gld_ret = (p1.gld / p0.gld - 1) if p0.gld > 0 else 0.0
        tlt_ret = (p1.tlt / p0.tlt - 1) if p0.tlt > 0 else 0.0

        # VIXY return: estimate from SPY inverse relationship
        # VIXY typically moves inversely to SPY at ~3.5x leverage
        vixy_ret = -spy_ret * 3.5 if spy_ret < 0 else -spy_ret * 2.0

        return (
            spy_w * spy_ret
            + gld_w * gld_ret
            + tlt_w * tlt_ret
            + vixy_w * vixy_ret
        )

    def run(self) -> BacktestResult:
        """Run the walk-forward backtest simulation.

        Returns a BacktestResult with hedged portfolio metrics, baseline
        comparison, hedge activity stats, crisis returns, and regime breakdown.
        """
        if not self._daily_prices:
            self.load_data()

        if len(self._daily_prices) < 2:
            logger.error("Insufficient data for backtest")
            return self._empty_result()

        prices = self._daily_prices
        config = self.config

        # ── Baseline: buy-and-hold 46/38/16 ──────────────────────────────
        baseline_equity = self._run_baseline(prices, config)

        # ── Hedged: dynamic VIXY allocation ──────────────────────────────
        hedge_equity, hedge_tracker, regime_tracker = self._run_hedged(prices, config)

        # Compute metrics from equity curves
        baseline_metrics = compute_metrics(baseline_equity, config.initial_capital)
        hedge_metrics = compute_metrics(hedge_equity, config.initial_capital)

        # Crisis returns
        prices_lookup = self._build_prices_lookup()
        trading_dates = self._trading_dates

        crisis_hedged = self._compute_crisis_returns_hedged(
            prices_lookup, trading_dates, hedge_equity, config.initial_capital
        )
        crisis_baseline = compute_crisis_returns(prices_lookup, trading_dates)

        # Regime breakdown
        regime_breakdown = self._compute_regime_breakdown(regime_tracker)

        # Hedge activity stats
        hedge_active_days = hedge_tracker["active_days"]
        total_days = max(len(prices) - 1, 1)
        hedge_active_pct = round(100.0 * hedge_active_days / total_days, 2)
        avg_hedge_pct = hedge_tracker["avg_pct"]
        max_hedge_pct = hedge_tracker["max_pct"]

        # Trade stats
        total_rebalances = hedge_tracker["rebalances"]
        total_costs = hedge_tracker["total_costs"]

        return BacktestResult(
            total_return=hedge_metrics.total_return,
            cagr=hedge_metrics.cagr,
            volatility=hedge_metrics.volatility,
            sharpe_ratio=hedge_metrics.sharpe_ratio,
            max_drawdown=hedge_metrics.max_drawdown,
            baseline_sharpe=baseline_metrics.sharpe_ratio,
            sharpe_improvement=round(
                hedge_metrics.sharpe_ratio - baseline_metrics.sharpe_ratio, 4
            ),
            total_rebalances=total_rebalances,
            total_transaction_costs=round(total_costs, 2),
            crisis_returns=crisis_hedged,
            extras={
                "baseline_total_return": baseline_metrics.total_return,
                "baseline_cagr": baseline_metrics.cagr,
                "baseline_volatility": baseline_metrics.volatility,
                "baseline_sharpe": baseline_metrics.sharpe_ratio,
                "baseline_max_drawdown": baseline_metrics.max_drawdown,
                "cagr_impact": round(
                    hedge_metrics.cagr - baseline_metrics.cagr, 2
                ),
                "hedge_active_days": hedge_active_days,
                "hedge_active_pct": hedge_active_pct,
                "avg_hedge_pct": avg_hedge_pct,
                "max_hedge_pct": max_hedge_pct,
                "crisis_returns_hedged": crisis_hedged,
                "crisis_returns_baseline": crisis_baseline,
                "regime_breakdown": regime_breakdown,
                "config_snapshot": {
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                    "initial_capital": config.initial_capital,
                    "max_hedge_pct": config.max_hedge_pct,
                    "rebalance_frequency_days": config.rebalance_frequency_days,
                    "transaction_cost_bps": config.transaction_cost_bps,
                    "base_allocation": {
                        "SPY": config.base_weights['SPY'],
                        "GLD": config.base_weights['GLD'],
                        "TLT": config.base_weights['TLT'],
                    },
                },
            },
        )

    def _run_baseline(
        self, prices: List[DailyPrices], config: BacktestConfig
    ) -> List[float]:
        """Run baseline buy-and-hold 46/38/16 portfolio."""
        spy_w = config.base_weights['SPY']
        gld_w = config.base_weights['GLD']
        tlt_w = config.base_weights['TLT']

        equity = [config.initial_capital]

        for i in range(1, len(prices)):
            ret = self._compute_portfolio_return(
                prices[i - 1], prices[i], spy_w, gld_w, tlt_w, 0.0
            )
            new_equity = equity[-1] * (1 + ret)
            equity.append(new_equity)

        return equity

    def _run_hedged(
        self, prices: List[DailyPrices], config: BacktestConfig
    ) -> Tuple[List[float], Dict, List[Dict]]:
        """Run hedged portfolio with dynamic VIXY allocation.

        Returns (equity_curve, hedge_stats, regime_tracker).
        """
        spy_w = config.base_weights['SPY']
        gld_w = config.base_weights['GLD']
        tlt_w = config.base_weights['TLT']
        vixy_w = 0.0

        equity = [config.initial_capital]
        hedge_tracker = {
            "active_days": 0,
            "avg_pct": 0.0,
            "max_pct": 0.0,
            "rebalances": 0,
            "total_costs": 0.0,
            "allocations": [],
        }
        regime_tracker: List[Dict] = []
        days_since_rebalance = 0
        rebalance_freq = config.rebalance_frequency_days
        cost_per_trade = config.transaction_cost_bps / 10000.0  # Convert bps to decimal

        for i in range(1, len(prices)):
            date = prices[i].date
            days_since_rebalance += 1

            # Rebalance on initial day (i == 1) and then monthly
            if days_since_rebalance >= rebalance_freq or i == 1:
                vix_level = self._get_vix_level(i)

                # Get hedge allocation from VixyHedgeSizer
                raw_allocation = self.sizer.compute_allocation(vix_level)

                # Apply hard cap from config
                vixy_target = min(raw_allocation, config.max_hedge_pct)

                # Fund VIXY from SPY sleeve
                new_spy_w = spy_w - vixy_target / 100.0
                new_spy_w = max(0.0, new_spy_w)  # No shorting

                # Rebalance: set new weights, pay transaction cost
                old_weights = [spy_w, gld_w, tlt_w, vixy_w]
                new_weights = [new_spy_w, gld_w, tlt_w, vixy_target / 100.0]

                # Transaction cost: proportional to absolute weight change
                turnover = sum(abs(new_weights[j] - old_weights[j]) for j in range(4))
                cost = turnover * cost_per_trade * equity[-1]

                spy_w, vixy_w = new_spy_w, new_weights[3]

                hedge_tracker["rebalances"] += 1
                hedge_tracker["total_costs"] += cost
                days_since_rebalance = 0

                # Track regime
                regime = self.sizer.classify_regime(vix_level).value
                regime_tracker.append({
                    "date": date,
                    "vix_level": vix_level,
                    "regime": regime,
                    "hedge_pct": vixy_target,
                })

            # Track daily hedge stats
            vixy_pct = vixy_w * 100
            hedge_tracker["allocations"].append(vixy_pct)
            if vixy_pct > 0:
                hedge_tracker["active_days"] += 1
            if vixy_pct > hedge_tracker["max_pct"]:
                hedge_tracker["max_pct"] = vixy_pct

            # Daily return
            ret = self._compute_portfolio_return(
                prices[i - 1], prices[i], spy_w, gld_w, tlt_w, vixy_w
            )
            new_equity = equity[-1] * (1 + ret)
            equity.append(new_equity)

        # Compute average hedge percentage
        allocs = hedge_tracker["allocations"]
        hedge_tracker["avg_pct"] = round(float(np.mean(allocs)), 2) if allocs else 0.0
        hedge_tracker["max_pct"] = round(hedge_tracker["max_pct"], 2)

        return equity, hedge_tracker, regime_tracker

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

    def _compute_crisis_returns_hedged(
        self,
        prices_lookup: Dict[str, Dict[str, float]],
        trading_dates: List[str],
        equity_curve: List[float],
        initial_capital: float,
    ) -> Dict[str, float]:
        """Compute hedged portfolio returns during crisis years.

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
        """Compute hedge stats broken down by VIX regime.

        Returns {regime_name: {avg_hedge_pct, max_hedge_pct, count, pct_of_time}}.
        """
        regimes = {}
        for entry in regime_tracker:
            reg = entry["regime"]
            if reg not in regimes:
                regimes[reg] = []
            regimes[reg].append(entry["hedge_pct"])

        total = sum(len(v) for v in regimes.values())
        breakdown = {}
        for reg_name, hedges in regimes.items():
            pct_of_time = round(100.0 * len(hedges) / total, 2) if total > 0 else 0.0
            breakdown[reg_name] = {
                "avg_hedge_pct": round(float(np.mean(hedges)), 2),
                "max_hedge_pct": round(float(np.max(hedges)), 2),
                "count": len(hedges),
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
            total_rebalances=0,
            total_transaction_costs=0.0,
            extras={
                "baseline_total_return": 0.0,
                "baseline_cagr": 0.0,
                "baseline_volatility": 0.0,
                "baseline_sharpe": 0.0,
                "baseline_max_drawdown": 0.0,
                "cagr_impact": 0.0,
                "hedge_active_days": 0,
                "hedge_active_pct": 0.0,
                "avg_hedge_pct": 0.0,
                "max_hedge_pct": 0.0,
                "crisis_returns_hedged": {},
                "crisis_returns_baseline": {},
                "regime_breakdown": {},
                "config_snapshot": {},
            },
        )

    def print_results(self, result: BacktestResult) -> None:
        """Print formatted backtest results to stdout."""
        logger.info("\n" + "=" * 70)
        logger.info("  VIXY Hedge Sizing — Walk-Forward Backtest Results")
        logger.info("=" * 70)

        logger.info(f"\n  Period: {self.config.start_date} to {self.config.end_date}")
        logger.info(f"  Capital: ${self.config.initial_capital:,.0f}")
        logger.info(f"  Baseline: SPY {self.config.base_weights['SPY']*100:.0f}% / "
              f"GLD {self.config.base_weights['GLD']*100:.0f}% / "
              f"TLT {self.config.base_weights['TLT']*100:.0f}%")

        logger.info(f"\n  {'Metric':<30} {'Baseline':>10} {'Hedged':>10} {'Delta':>10}")
        logger.info(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
        logger.info(f"  {'Total Return':<30} {result.extras['baseline_total_return']:>9.2f}% {result.total_return:>9.2f}% "
              f"{result.total_return - result.extras['baseline_total_return']:>+9.2f}%")
        logger.info(f"  {'CAGR':<30} {result.extras['baseline_cagr']:>9.2f}% {result.cagr:>9.2f}% "
              f"{result.extras['cagr_impact']:>+9.2f}%")
        logger.info(f"  {'Volatility':<30} {result.extras['baseline_volatility']:>9.2f}% {result.volatility:>9.2f}% "
              f"{result.volatility - result.extras['baseline_volatility']:>+9.2f}%")
        logger.info(f"  {'Sharpe Ratio':<30} {result.extras['baseline_sharpe']:>10.4f} {result.sharpe_ratio:>10.4f} "
              f"{result.sharpe_improvement:>+10.4f}")
        logger.info(f"  {'Max Drawdown':<30} {result.extras['baseline_max_drawdown']:>9.2f}% {result.max_drawdown:>9.2f}% "
              f"{result.max_drawdown - result.extras['baseline_max_drawdown']:>+9.2f}%")

        logger.info(f"\n  ── Hedge Activity ──")
        logger.info(f"  Hedge active days:  {result.extras['hedge_active_days']} ({result.extras['hedge_active_pct']:.1f}%)")
        logger.info(f"  Avg hedge:          {result.extras['avg_hedge_pct']:.2f}%")
        logger.info(f"  Max hedge:          {result.extras['max_hedge_pct']:.2f}%")
        logger.info(f"  Rebalances:         {result.total_rebalances}")
        logger.info(f"  Transaction costs:  ${result.total_transaction_costs:.2f}")

        logger.info(f"\n  ── Crisis Returns (%) ──")
        logger.info(f"  {'Year':<10} {'Baseline':>10} {'Hedged':>10}")
        logger.info(f"  {'-'*10} {'-'*10} {'-'*10}")
        crisis_baseline = result.extras.get('crisis_returns_baseline', {})
        crisis_hedged = result.extras.get('crisis_returns_hedged', {})
        all_crisis_years = sorted(
            set(crisis_baseline) | set(crisis_hedged)
        )
        for year in all_crisis_years:
            b = crisis_baseline.get(year, 0.0)
            h = crisis_hedged.get(year, 0.0)
            logger.info(f"  {year:<10} {b:>10.2f} {h:>10.2f}")

        logger.info(f"\n  ── Regime Breakdown ──")
        logger.info(f"  {'Regime':<15} {'% Time':>8} {'Avg Hedge':>10} {'Max Hedge':>10}")
        logger.info(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*10}")
        for reg_name, stats in sorted(result.extras.get('regime_breakdown', {}).items()):
            logger.info(f"  {reg_name:<15} {stats['pct_of_time']:>7.1f}% "
                  f"{stats['avg_hedge_pct']:>9.2f}% {stats['max_hedge_pct']:>9.2f}%")

        logger.info("\n" + "=" * 70)

    def save_results(self, result: BacktestResult, output_path: Optional[str] = None) -> None:
        """Save backtest results to a JSON file."""
        data = _result_to_dict(result)
        data["_metadata"] = {
            "strategy": "vixy_hedge",
            "generated": datetime.now().isoformat(),
            "type": "walk_forward_backtest",
        }

        if output_path:
            save_results_json(data, output_path=output_path)
            logger.info("Results saved to %s", output_path)
        else:
            named_path = str(BACKTEST_RESULTS_DIR / "vixy_hedge_backtest_results.json")
            save_results_json(data, output_path=named_path)
            logger.info("Results saved to %s", named_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for the VIXY hedge walk-forward backtest."""
    import argparse

    parser = argparse.ArgumentParser(
        description="VIXY Hedge Walk-Forward Backtest v9.32"
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
        "--max-hedge",
        type=float,
        default=None,
        help="Max VIXY hedge %% (default: 6.0)",
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
    if args.max_hedge is not None:
        config.max_hedge_pct = args.max_hedge

    backtester = WalkForwardVIXYBacktester(config)
    result = backtester.run()
    backtester.print_results(result)

    if args.save or args.output:
        backtester.save_results(result, output_path=args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
