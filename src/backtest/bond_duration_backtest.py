"""
Bond Duration Rotation Walk-Forward Backtest - v9.33 Implementation

Validates the bond duration rotation overlay on the baseline 46/38/16
(SPY/GLD/TLT) portfolio. The rotation dynamically allocates the 16% bond
sleeve across TLT (16yr), IEF (7yr), and SHY (2yr) based on TLT 60-day
momentum mapped to yield curve context for the production
BondDurationCalculator.

Strategy (via src/signals/bond_duration_signal.BondDurationCalculator):
- TLT 60-day momentum -> approximate yield curve context (spread, rate_chg)
- BondDurationCalculator.classify_curve()/classify_rate_direction()
- BondDurationCalculator.compute_duration_allocation() -> 4x3 regime matrix

Key questions:
  1. Does dynamic duration rotation improve Sharpe vs static TLT allocation?
  2. How does rotation perform in rising vs falling rate environments?
  3. Is there evidence that shortening duration during rate hikes protects capital?

Period: 2006-2026 (20+ years including GFC, 2013 taper tantrum, 2022 rate hikes)
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
from src.signals.bond_duration_signal import BondDurationCalculator

logging.basicConfig(level=logging.INFO)
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
BOND_SYMBOLS = ["TLT", "IEF", "SHY"]

# Duration estimates (years)
DURATION = {"TLT": 16.0, "IEF": 7.0, "SHY": 2.0}

# Bond sleeve allocation by TLT momentum regime
# (tlt_sleeve_pct, ief_sleeve_pct, shy_sleeve_pct, label)
RISING_ALLOCATION = (0.70, 0.20, 0.10, "rising")
FALLING_ALLOCATION = (0.10, 0.30, 0.60, "falling")
NEUTRAL_ALLOCATION = (0.40, 0.40, 0.20, "neutral")

# TLT 60-day momentum thresholds
MOMENTUM_LOOKBACK = 60
RISING_THRESHOLD = 0.01   # 1% gain = rising
FALLING_THRESHOLD = -0.01  # -1% loss = falling

# Bond sleeve fraction of total portfolio
BOND_SLEEVE = 0.16


@dataclass
class BacktestConfig(_BaseConfig):
    """Configuration for bond duration walk-forward backtest.

    Inherits canonical fields (start_date, end_date, initial_capital,
    base_weights, rebalance_frequency_days, transaction_cost_bps) from
    BacktestConfig in metrics.py.
    """

    # Backtest-specific: momentum lookback
    momentum_lookback_days: int = MOMENTUM_LOOKBACK


class WalkForwardBondDurationBacktester:
    """
    Walk-forward backtest for bond duration rotation strategy.

    Simulates monthly rebalancing of the baseline 46/38/16 portfolio with
    a dynamic bond sleeve allocation across TLT/IEF/SHY based on TLT 60-day
    momentum.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._daily_prices: List[DailyPrices] = []
        self._trading_dates: List[str] = []

    def load_data(self) -> None:
        """Load price data from PRICES_JSON and extract SPY/GLD/TLT/IEF/SHY.

        Falls back to synthetic data generation if the file is not found.
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
        ief_data = {e["d"]: e["p"] for e in raw.get("IEF", [])} if "IEF" in raw else {}
        shy_data = {e["d"]: e["p"] for e in raw.get("SHY", [])} if "SHY" in raw else {}

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
            self._daily_prices.append(DailyPrices(
                date=date,
                spy=float(spy_data[date]),
                gld=float(gld_data[date]),
                tlt=float(tlt_data[date]),
                ief=float(ief_data.get(date, tlt_data[date])),  # Fallback to TLT if missing
                shy=float(shy_data.get(date, tlt_data[date])),  # Fallback to TLT if missing
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
        IEF and SHY are modelled with lower duration sensitivity than TLT.
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

        # IEF: 2.5% drift, 8% vol, 0.8 correlation to TLT (lower duration = lower vol)
        ief_noise = np.random.normal(0.025 / TRADING_DAYS_PER_YEAR, 0.08 / np.sqrt(TRADING_DAYS_PER_YEAR), n_days)
        ief_common = 0.8 * tlt_returns
        ief_returns = ief_common + np.sqrt(1 - 0.8 ** 2) * ief_noise
        ief_price = 100.0 * np.exp(np.cumsum(ief_returns))

        # SHY: 2% drift, 3% vol, 0.5 correlation to TLT (very short duration)
        shy_noise = np.random.normal(0.02 / TRADING_DAYS_PER_YEAR, 0.03 / np.sqrt(TRADING_DAYS_PER_YEAR), n_days)
        shy_common = 0.5 * tlt_returns
        shy_returns = shy_common + np.sqrt(1 - 0.5 ** 2) * shy_noise
        shy_price = 100.0 * np.exp(np.cumsum(shy_returns))

        self._trading_dates = dates
        self._daily_prices = []

        for i, date in enumerate(dates):
            self._daily_prices.append(DailyPrices(
                date=date,
                spy=float(spy_price[i]),
                gld=float(gld_price[i]),
                tlt=float(tlt_price[i]),
                ief=float(ief_price[i]),
                shy=float(shy_price[i]),
            ))

        logger.info("Generated %d synthetic trading days", n_days)

    def _compute_tlt_60d_momentum(self, idx: int) -> float:
        """Compute TLT 60-day return ending at index idx.

        Returns 0.0 if insufficient history.
        """
        lookback = self.config.momentum_lookback_days
        if idx < lookback:
            return 0.0

        p0 = self._daily_prices[idx - lookback].tlt
        p1 = self._daily_prices[idx].tlt

        if p0 <= 0:
            return 0.0

        return p1 / p0 - 1

    def _classify_momentum(self, momentum: float) -> str:
        """Classify TLT momentum as rising, falling, or neutral."""
        if momentum > RISING_THRESHOLD:
            return "rising"
        elif momentum < FALLING_THRESHOLD:
            return "falling"
        return "neutral"

    def _get_bond_sleeve_allocation(self, momentum_regime: str) -> Tuple[float, float, float, str]:
        """Get TLT/IEF/SHY sleeve weights for the given momentum regime."""
        if momentum_regime == "rising":
            return RISING_ALLOCATION
        elif momentum_regime == "falling":
            return FALLING_ALLOCATION
        return NEUTRAL_ALLOCATION

    def _momentum_to_yield_context(self, tlt_momentum: float) -> Tuple[float, float, float]:
        """Map TLT 60-day momentum to approximate yield curve context.

        TLT price up -> yields falling -> curve steep/normal, rates falling
        TLT price down -> yields rising -> curve flat/inverted, rates rising

        Returns (spread, real_rate, rate_chg) for BondDurationCalculator.
        real_rate is set to 1.5 (neutral) since we lack CPI data in the backtest.
        """
        if tlt_momentum > 0.05:
            # Strong positive: yields falling sharply, curve likely steep
            spread = 0.8
            rate_chg = -0.5
        elif tlt_momentum > 0.01:
            # Moderate positive: yields falling, curve normal
            spread = 0.5
            rate_chg = -0.2
        elif tlt_momentum >= -0.01:
            # Neutral: yields stable, curve normal/flat border
            spread = 0.3
            rate_chg = 0.0
        elif tlt_momentum >= -0.05:
            # Moderate negative: yields rising, curve flat
            spread = 0.1
            rate_chg = 0.3
        else:
            # Strong negative: yields rising sharply, curve inverted
            spread = -0.2
            rate_chg = 0.6

        return spread, 1.5, rate_chg

    def _compute_effective_duration(self, tlt_w: float, ief_w: float, shy_w: float) -> float:
        """Compute weighted average effective duration in years."""
        return (
            tlt_w * DURATION["TLT"]
            + ief_w * DURATION["IEF"]
            + shy_w * DURATION["SHY"]
        )

    def _compute_portfolio_return(
        self,
        p0: DailyPrices,
        p1: DailyPrices,
        spy_w: float,
        gld_w: float,
        bond_w: float,
        tlt_sleeve: float,
        ief_sleeve: float,
        shy_sleeve: float,
    ) -> float:
        """Compute 1-day portfolio return given current weights."""
        spy_ret = (p1.spy / p0.spy - 1) if p0.spy > 0 else 0.0
        gld_ret = (p1.gld / p0.gld - 1) if p0.gld > 0 else 0.0
        tlt_ret = (p1.tlt / p0.tlt - 1) if p0.tlt > 0 else 0.0
        ief_ret = (p1.ief / p0.ief - 1) if p0.ief > 0 else 0.0
        shy_ret = (p1.shy / p0.shy - 1) if p0.shy > 0 else 0.0

        # Bond sleeve composite return: weighted average of TLT, IEF, SHY
        bond_composite_ret = (
            tlt_sleeve * tlt_ret
            + ief_sleeve * ief_ret
            + shy_sleeve * shy_ret
        )

        return (
            spy_w * spy_ret
            + gld_w * gld_ret
            + bond_w * bond_composite_ret
        )

    def run(self) -> BacktestResult:
        """Run the walk-forward backtest simulation.

        Returns a BacktestResult with rotated portfolio metrics, baseline
        comparison, rotation activity stats, crisis returns, and regime breakdown.
        """
        if not self._daily_prices:
            self.load_data()

        if len(self._daily_prices) < 2:
            logger.error("Insufficient data for backtest")
            return self._empty_result()

        prices = self._daily_prices
        config = self.config

        # ── Baseline: buy-and-hold 46/38/16 (all bonds in TLT) ─────────────
        baseline_equity = self._run_baseline(prices, config)

        # ── Rotated: dynamic bond duration allocation ──────────────────────
        rotated_equity, tracker, regime_tracker = self._run_rotated(prices, config)

        # Compute metrics from equity curves
        baseline_metrics = compute_metrics(baseline_equity, config.initial_capital)
        rotated_metrics = compute_metrics(rotated_equity, config.initial_capital)

        # Crisis returns
        prices_lookup = self._build_prices_lookup()
        trading_dates = self._trading_dates

        crisis_rotated = self._compute_crisis_returns_rotated(
            prices_lookup, trading_dates, rotated_equity, config.initial_capital
        )
        crisis_baseline = compute_crisis_returns(prices_lookup, trading_dates)

        # Regime breakdown
        regime_breakdown = self._compute_regime_breakdown(regime_tracker)

        # Rotation activity stats
        rotation_active_days = tracker["active_days"]
        total_days = max(len(prices) - 1, 1)
        rotation_active_pct = round(100.0 * rotation_active_days / total_days, 2)
        avg_effective_duration = tracker["avg_effective_duration"]
        avg_tlt_weight = tracker["avg_tlt_weight"]
        avg_ief_weight = tracker["avg_ief_weight"]
        avg_shy_weight = tracker["avg_shy_weight"]

        # Trade stats
        total_rebalances = tracker["rebalances"]
        total_costs = tracker["total_costs"]

        return BacktestResult(
            total_return=rotated_metrics.total_return,
            cagr=rotated_metrics.cagr,
            volatility=rotated_metrics.volatility,
            sharpe_ratio=rotated_metrics.sharpe_ratio,
            max_drawdown=rotated_metrics.max_drawdown,
            baseline_sharpe=baseline_metrics.sharpe_ratio,
            sharpe_improvement=round(
                rotated_metrics.sharpe_ratio - baseline_metrics.sharpe_ratio, 4
            ),
            total_rebalances=total_rebalances,
            total_transaction_costs=round(total_costs, 2),
            extras={
                "baseline_total_return": baseline_metrics.total_return,
                "baseline_cagr": baseline_metrics.cagr,
                "baseline_volatility": baseline_metrics.volatility,
                "baseline_max_drawdown": baseline_metrics.max_drawdown,
                "cagr_impact": round(
                    rotated_metrics.cagr - baseline_metrics.cagr, 2
                ),
                "rotation_active_days": rotation_active_days,
                "rotation_active_pct": rotation_active_pct,
                "avg_effective_duration": avg_effective_duration,
                "avg_tlt_weight": avg_tlt_weight,
                "avg_ief_weight": avg_ief_weight,
                "avg_shy_weight": avg_shy_weight,
                "crisis_returns_rotated": crisis_rotated,
                "crisis_returns_baseline": crisis_baseline,
                "regime_breakdown": regime_breakdown,
                "config_snapshot": {
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                    "initial_capital": config.initial_capital,
                    "rebalance_frequency_days": config.rebalance_frequency_days,
                    "transaction_cost_bps": config.transaction_cost_bps,
                    "momentum_lookback_days": config.momentum_lookback_days,
                    "base_allocation": {
                        "SPY": config.base_weights['SPY'],
                        "GLD": config.base_weights['GLD'],
                        "TLT": config.base_weights['TLT'],
                    },
                    "bond_sleeve_allocation": "BondDurationCalculator.compute_duration_allocation() via _momentum_to_yield_context()",
                },
            },
        )

    def _run_baseline(
        self, prices: List[DailyPrices], config: BacktestConfig
    ) -> List[float]:
        """Run baseline buy-and-hold 46/38/16 portfolio (all bonds in TLT)."""
        spy_w = config.base_weights['SPY']
        gld_w = config.base_weights['GLD']
        tlt_w = config.base_weights['TLT']

        equity = [config.initial_capital]

        for i in range(1, len(prices)):
            ret = self._compute_portfolio_return(
                prices[i - 1], prices[i], spy_w, gld_w, tlt_w,
                1.0, 0.0, 0.0,  # All bonds in TLT
            )
            new_equity = equity[-1] * (1 + ret)
            equity.append(new_equity)

        return equity

    def _run_rotated(
        self, prices: List[DailyPrices], config: BacktestConfig
    ) -> Tuple[List[float], Dict, List[Dict]]:
        """Run rotated portfolio with dynamic bond duration allocation.

        Returns (equity_curve, rotation_stats, regime_tracker).
        """
        spy_w = config.base_weights['SPY']
        gld_w = config.base_weights['GLD']
        bond_w = config.base_weights['TLT']

        # Sleeve weights within the bond sleeve
        tlt_sleeve = 1.0
        ief_sleeve = 0.0
        shy_sleeve = 0.0

        equity = [config.initial_capital]
        tracker = {
            "active_days": 0,
            "rebalances": 0,
            "total_costs": 0.0,
            "sleeve_weights": [],
            "effective_durations": [],
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
                tlt_momentum = self._compute_tlt_60d_momentum(i)

                # Use production BondDurationCalculator via momentum-to-yield-context mapping
                spread, real_rate, rate_chg = self._momentum_to_yield_context(tlt_momentum)
                calc = BondDurationCalculator()
                curve_regime = calc.classify_curve(spread)
                rate_direction = calc.classify_rate_direction(rate_chg)
                new_tlt_s, new_ief_s, new_shy_s, label = calc.compute_duration_allocation(
                    spread, real_rate, rate_direction, curve_regime
                )

                # Transaction cost: proportional to absolute weight change in the bond sleeve
                # The bond sleeve is bond_w of the total portfolio, so turnover is scaled
                sleeve_turnover = (
                    abs(new_tlt_s - tlt_sleeve)
                    + abs(new_ief_s - ief_sleeve)
                    + abs(new_shy_s - shy_sleeve)
                )
                # Scale turnover by bond sleeve weight (only the bond portion changes)
                portfolio_turnover = sleeve_turnover * bond_w
                cost = portfolio_turnover * cost_per_trade * equity[-1]

                old_duration = self._compute_effective_duration(tlt_sleeve, ief_sleeve, shy_sleeve)
                new_duration = self._compute_effective_duration(new_tlt_s, new_ief_s, new_shy_s)

                tlt_sleeve, ief_sleeve, shy_sleeve = new_tlt_s, new_ief_s, new_shy_s

                tracker["rebalances"] += 1
                tracker["total_costs"] += cost
                days_since_rebalance = 0

                regime_tracker.append({
                    "date": date,
                    "tlt_momentum": round(tlt_momentum, 4),
                    "momentum_regime": label,
                    "tlt_sleeve": tlt_sleeve,
                    "ief_sleeve": ief_sleeve,
                    "shy_sleeve": shy_sleeve,
                    "effective_duration": round(new_duration, 1),
                    "duration_change": round(new_duration - old_duration, 1),
                })

            # Track daily stats
            eff_dur = self._compute_effective_duration(tlt_sleeve, ief_sleeve, shy_sleeve)
            tracker["sleeve_weights"].append({
                "tlt": tlt_sleeve,
                "ief": ief_sleeve,
                "shy": shy_sleeve,
            })
            tracker["effective_durations"].append(eff_dur)

            # Count as "active" when TLT sleeve differs from baseline (1.0)
            if abs(tlt_sleeve - 1.0) > 0.01:
                tracker["active_days"] += 1

            # Daily return
            ret = self._compute_portfolio_return(
                prices[i - 1], prices[i],
                spy_w, gld_w, bond_w,
                tlt_sleeve, ief_sleeve, shy_sleeve,
            )
            new_equity = equity[-1] * (1 + ret)
            equity.append(new_equity)

        # Compute averages
        durations = tracker["effective_durations"]
        tracker["avg_effective_duration"] = round(float(np.mean(durations)), 1) if durations else 0.0

        tlt_vals = [s["tlt"] for s in tracker["sleeve_weights"]]
        ief_vals = [s["ief"] for s in tracker["sleeve_weights"]]
        shy_vals = [s["shy"] for s in tracker["sleeve_weights"]]
        tracker["avg_tlt_weight"] = round(float(np.mean(tlt_vals)), 4) if tlt_vals else 0.0
        tracker["avg_ief_weight"] = round(float(np.mean(ief_vals)), 4) if ief_vals else 0.0
        tracker["avg_shy_weight"] = round(float(np.mean(shy_vals)), 4) if shy_vals else 0.0

        return equity, tracker, regime_tracker

    def _build_prices_lookup(self) -> Dict[str, Dict[str, float]]:
        """Build {date: {symbol: price}} lookup from daily prices."""
        lookup: Dict[str, Dict[str, float]] = {}
        for dp in self._daily_prices:
            lookup[dp.date] = {
                "SPY": dp.spy,
                "GLD": dp.gld,
                "TLT": dp.tlt,
                "IEF": dp.ief,
                "SHY": dp.shy,
            }
        return lookup

    def _compute_crisis_returns_rotated(
        self,
        prices_lookup: Dict[str, Dict[str, float]],
        trading_dates: List[str],
        equity_curve: List[float],
        initial_capital: float,
    ) -> Dict[str, float]:
        """Compute rotated portfolio returns during crisis years.

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
        """Compute rotation stats broken down by TLT momentum regime.

        Returns {regime_name: {pct_of_time, avg_duration, count, ...}}.
        """
        regimes = {}
        for entry in regime_tracker:
            reg = entry["momentum_regime"]
            if reg not in regimes:
                regimes[reg] = []
            regimes[reg].append(entry)

        total = sum(len(v) for v in regimes.values())
        breakdown = {}
        for reg_name, entries in regimes.items():
            pct_of_time = round(100.0 * len(entries) / total, 2) if total > 0 else 0.0
            avg_dur = float(np.mean([e["effective_duration"] for e in entries]))
            breakdown[reg_name] = {
                "pct_of_time": pct_of_time,
                "count": len(entries),
                "avg_effective_duration": round(avg_dur, 1),
                "avg_tlt_sleeve": round(float(np.mean([e["tlt_sleeve"] for e in entries])), 4),
                "avg_ief_sleeve": round(float(np.mean([e["ief_sleeve"] for e in entries])), 4),
                "avg_shy_sleeve": round(float(np.mean([e["shy_sleeve"] for e in entries])), 4),
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
            sharpe_improvement=0.0,
            total_rebalances=0,
            total_transaction_costs=0.0,
            extras={
                "baseline_total_return": 0.0,
                "baseline_cagr": 0.0,
                "baseline_volatility": 0.0,
                "baseline_max_drawdown": 0.0,
                "cagr_impact": 0.0,
                "rotation_active_days": 0,
                "rotation_active_pct": 0.0,
                "avg_effective_duration": 0.0,
                "avg_tlt_weight": 0.0,
                "avg_ief_weight": 0.0,
                "avg_shy_weight": 0.0,
                "crisis_returns_rotated": {},
                "crisis_returns_baseline": {},
                "regime_breakdown": {},
                "config_snapshot": {},
            },
        )

    def print_results(self, result: BacktestResult) -> None:
        """Print formatted backtest results to stdout."""
        print("\n" + "=" * 70)
        print("  Bond Duration Rotation — Walk-Forward Backtest Results")
        print("=" * 70)

        print(f"\n  Period: {self.config.start_date} to {self.config.end_date}")
        print(f"  Capital: ${self.config.initial_capital:,.0f}")
        print(f"  Baseline: SPY {self.config.base_weights['SPY']*100:.0f}% / "
              f"GLD {self.config.base_weights['GLD']*100:.0f}% / "
              f"Bonds {self.config.base_weights['TLT']*100:.0f}%")

        e = result.extras

        print(f"\n  {'Metric':<30} {'Baseline':>10} {'Rotated':>10} {'Delta':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'Total Return':<30} {e['baseline_total_return']:>9.2f}% {result.total_return:>9.2f}% "
              f"{result.total_return - e['baseline_total_return']:>+9.2f}%")
        print(f"  {'CAGR':<30} {e['baseline_cagr']:>9.2f}% {result.cagr:>9.2f}% "
              f"{e['cagr_impact']:>+9.2f}%")
        print(f"  {'Volatility':<30} {e['baseline_volatility']:>9.2f}% {result.volatility:>9.2f}% "
              f"{result.volatility - e['baseline_volatility']:>+9.2f}%")
        print(f"  {'Sharpe Ratio':<30} {result.baseline_sharpe:>10.4f} {result.sharpe_ratio:>10.4f} "
              f"{result.sharpe_improvement:>+10.4f}")
        print(f"  {'Max Drawdown':<30} {e['baseline_max_drawdown']:>9.2f}% {result.max_drawdown:>9.2f}% "
              f"{result.max_drawdown - e['baseline_max_drawdown']:>+9.2f}%")

        print(f"\n  -- Rotation Activity --")
        print(f"  Rotation active days:  {e['rotation_active_days']} ({e['rotation_active_pct']:.1f}%)")
        print(f"  Avg effective duration: {e['avg_effective_duration']:.1f} yr")
        print(f"  Avg sleeve weights:    TLT {e['avg_tlt_weight']:.1%} / "
              f"IEF {e['avg_ief_weight']:.1%} / SHY {e['avg_shy_weight']:.1%}")
        print(f"  Rebalances:            {result.total_rebalances}")
        print(f"  Transaction costs:     ${result.total_transaction_costs:.2f}")

        print(f"\n  -- Crisis Returns (%) --")
        print(f"  {'Year':<10} {'Baseline':>10} {'Rotated':>10}")
        print(f"  {'-'*10} {'-'*10} {'-'*10}")
        crisis_baseline = e.get("crisis_returns_baseline", {})
        crisis_rotated = e.get("crisis_returns_rotated", {})
        all_crisis_years = sorted(
            set(list(crisis_baseline.keys()) + list(crisis_rotated.keys()))
        )
        for year in all_crisis_years:
            b = crisis_baseline.get(year, 0.0)
            r = crisis_rotated.get(year, 0.0)
            print(f"  {year:<10} {b:>10.2f} {r:>10.2f}")

        print(f"\n  -- Regime Breakdown --")
        print(f"  {'Regime':<12} {'% Time':>8} {'Avg Dur':>8} {'TLT':>8} {'IEF':>8} {'SHY':>8}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        regime_breakdown = e.get("regime_breakdown", {})
        for reg_name in ["rising", "neutral", "falling"]:
            stats = regime_breakdown.get(reg_name)
            if stats:
                print(f"  {reg_name:<12} {stats['pct_of_time']:>7.1f}% "
                      f"{stats['avg_effective_duration']:>7.1f} "
                      f"{stats['avg_tlt_sleeve']:>7.1%} "
                      f"{stats['avg_ief_sleeve']:>7.1%} "
                      f"{stats['avg_shy_sleeve']:>7.1%}")

        print("\n" + "=" * 70)

    def save_results(self, result: BacktestResult, output_path: Optional[str] = None) -> None:
        """Save backtest results to a JSON file."""
        from dataclasses import asdict

        data = asdict(result)
        data["_metadata"] = {
            "strategy": "bond_duration",
            "generated": datetime.now().isoformat(),
            "type": "walk_forward_backtest",
        }

        if output_path:
            save_results_json(data, output_path=output_path)
            logger.info("Results saved to %s", output_path)
        else:
            named_path = str(BACKTEST_RESULTS_DIR / "bond_duration_backtest_results.json")
            save_results_json(data, output_path=named_path)
            logger.info("Results saved to %s", named_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for the bond duration walk-forward backtest."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Bond Duration Rotation Walk-Forward Backtest v9.33"
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

    backtester = WalkForwardBondDurationBacktester(config)
    result = backtester.run()
    backtester.print_results(result)

    if args.save or args.output:
        backtester.save_results(result, output_path=args.output)


if __name__ == "__main__":
    main()
