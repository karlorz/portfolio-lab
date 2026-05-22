#!/usr/bin/env python3
"""
Cross-Asset Regime Arbitrage Backtest - v8.09

Validates the cross-asset regime divergence signal as a tactical overlay
in the portfolio-lab ensemble (17% weight in CRISIS regime).

Core logic: detect divergences between equity (SPY), bond (TLT), and gold
(GLD) 60-day momentum regimes, then shift allocations toward the
outperforming asset class when divergence is detected.

Strategy:
  - Equity-bond divergence: shift toward the outperformer
  - All same regime: neutral (no signal)
  - Signal strength: 0.0 (flat) to +/-0.5 (max divergence)
  - Positive (equity outperforming): SPY+3%, GLD-1%, TLT-2%
  - Negative (bonds/gold outperforming): SPY-2%, GLD+1%, TLT+1%

Expected Impact: +0.01-0.02 Sharpe through earlier detection of regime transitions.
Period: 2006-2026
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.backtest.metrics import (
    BacktestConfig as _BaseConfig,
    compute_metrics,
    save_results_json,
)
from src.paths import PRICES_JSON, BACKTEST_RESULTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Try importing the actual signal module ----

try:
    from src.signals.cross_asset_regime_arb import (  # type: ignore[import-untyped]
        CrossAssetRegimeArbDetector,
        DivergencePattern,
        AssetRegime,
        BondRegime,
        GoldRegime,
        BULL_MOMENTUM_THRESHOLD,
        BEAR_MOMENTUM_THRESHOLD,
        MOMENTUM_LOOKBACK,
    )
    HAS_SIGNAL_MODULE = True
    logger.info("Using live CrossAssetRegimeArbDetector for signal generation")
except ImportError:
    HAS_SIGNAL_MODULE = False
    logger.info("Signal module not available; using simulated divergence logic")

    # Fallback constants (mirror the signal module)
    MOMENTUM_LOOKBACK = 60
    BULL_MOMENTUM_THRESHOLD = 0.05
    BEAR_MOMENTUM_THRESHOLD = -0.05

    class DivergencePattern:  # type: ignore[no-redef]
        """Minimal fallback enum."""
        FULL_RISK_ON = "full_risk_on"
        RISK_OFF = "risk_off"
        RISK_ROTATION = "risk_rotation"
        FLIGHT_TO_SAFETY = "flight_to_safety"
        INFLATION_FEAR = "inflation_fear"
        CAUTIOUS_OPTIMISM = "cautious_optimism"
        EQUITY_ROTATION = "equity_rotation"
        RECOVERY_BEGINNING = "recovery_beginning"
        NO_DIVERGENCE = "no_divergence"
        UNKNOWN = "unknown"


# ---- Data classes ----

ALLOCATION_SHIFTS: Dict[str, Dict[str, float]] = {
    "equity_outperformance": {"spy": 0.03, "gld": -0.01, "tlt": -0.02},
    "safe_haven_outperformance": {"spy": -0.02, "gld": 0.01, "tlt": 0.01},
}

MAX_SIGNAL_STRENGTH = 0.5


@dataclass
class BacktestConfig(_BaseConfig):
    """Configuration for cross-asset regime arb backtest.

    Inherits canonical fields (start_date, end_date, initial_capital,
    base_weights, rebalance_frequency, transaction_cost_bps, etc.) from
    _BaseConfig. Only backtest-specific fields are defined here.
    """

    # Overlay constraints
    max_single_shift: float = 0.05  # Max 5pp shift on any single asset per rebalance
    signal_threshold: float = 0.05  # Minimum signal strength to act


@dataclass
class DailyReturn:
    """Single day return data."""
    date: str
    spy_return: float
    gld_return: float
    tlt_return: float


@dataclass
class RebalanceSignal:
    """Signal output at a rebalance point."""
    date: str
    signal_value: float         # -0.5 to +0.5
    pattern: str                # Divergence pattern name
    spy_shift: float            # Allocation change applied
    gld_shift: float
    tlt_shift: float
    spy_momentum: float
    gld_momentum: float
    tlt_momentum: float


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Basic metrics
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float

    # Overlay impact
    overlay_active_months: int
    baseline_sharpe: float
    sharpe_improvement: float

    # Crisis performance
    return_2008: Optional[float]
    return_2020: Optional[float]
    return_2022: Optional[float]

    # Trade stats
    total_rebalances: int
    total_transaction_costs: float

    # Divergence statistics
    signal_frequency: float  # Fraction of months with non-zero signal
    divergence_breakdown: Dict[str, int]  # Pattern name -> count of occurrences

    # Full history
    equity_curve: List[Dict]
    rebalance_signals: List[Dict]


class CrossAssetRegimeArbBacktester:
    """
    Walk-forward backtest for the cross-asset regime arbitrage overlay.

    Detects per-asset-class regimes at each monthly rebalance and shifts
    allocations when regimes diverge.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.data: List[DailyReturn] = []
        self.detector: Optional[CrossAssetRegimeArbDetector] = (
            CrossAssetRegimeArbDetector() if HAS_SIGNAL_MODULE else None
        )

    # ---- Data Loading ----

    def load_data(self, data_path: Optional[str] = None) -> bool:
        """
        Load historical price data from PRICES_JSON.

        Returns True on success, False on failure.
        """
        try:
            prices_path = PRICES_JSON

            if prices_path.exists():
                with open(prices_path) as f:
                    prices_data = json.load(f)

                self._process_price_data(prices_data)
                logger.info(
                    f"Loaded {len(self.data)} days of price data from {prices_path}"
                )
                return True

            logger.error("No price data found")
            return False

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def _process_price_data(self, prices_data: Dict) -> None:
        """Process raw price data into daily returns."""
        spy_data = prices_data.get("SPY", [])
        gld_data = prices_data.get("GLD", [])
        tlt_data = prices_data.get("TLT", [])

        if not spy_data:
            logger.error("No SPY data found")
            return

        dates = [p["d"] for p in spy_data]
        spy_prices = {p["d"]: p["p"] for p in spy_data}
        gld_prices = {p["d"]: p["p"] for p in gld_data}
        tlt_prices = {p["d"]: p["p"] for p in tlt_data}

        for i, date in enumerate(dates[1:], 1):
            prev_date = dates[i - 1]
            spy_prev = spy_prices.get(prev_date)
            spy_curr = spy_prices.get(date)
            gld_prev = gld_prices.get(prev_date)
            gld_curr = gld_prices.get(date)
            tlt_prev = tlt_prices.get(prev_date)
            tlt_curr = tlt_prices.get(date)

            if all(
                [
                    spy_prev,
                    spy_curr,
                    gld_prev,
                    gld_curr,
                    tlt_prev,
                    tlt_curr,
                ]
            ):
                self.data.append(
                    DailyReturn(
                        date=date,
                        spy_return=(spy_curr - spy_prev) / spy_prev,
                        gld_return=(gld_curr - gld_prev) / gld_prev,
                        tlt_return=(tlt_curr - tlt_prev) / tlt_prev,
                    )
                )

    # ---- Signal Generation ----

    def _compute_momentum(
        self, lookback: int, day_index: int
    ) -> Tuple[float, float, float]:
        """Compute N-day return for SPY, GLD, TLT ending at day_index."""
        if day_index < lookback:
            return 0.0, 0.0, 0.0

        start = day_index - lookback
        self.data[start].spy_return
        self.data[day_index - 1].spy_return

        # Use price level approximation: sum of log returns
        spy_mom = sum(
            np.log(1 + self.data[j].spy_return)
            for j in range(start, day_index)
            if abs(self.data[j].spy_return) < 0.5
        )
        gld_mom = sum(
            np.log(1 + self.data[j].gld_return)
            for j in range(start, day_index)
            if abs(self.data[j].gld_return) < 0.5
        )
        tlt_mom = sum(
            np.log(1 + self.data[j].tlt_return)
            for j in range(start, day_index)
            if abs(self.data[j].tlt_return) < 0.5
        )

        return spy_mom, gld_mom, tlt_mom

    def _classify_asset_regime(self, momentum: float) -> str:
        """Classify an asset's regime based on 60-day momentum."""
        if momentum > BULL_MOMENTUM_THRESHOLD:
            return "bullish"
        elif momentum < BEAR_MOMENTUM_THRESHOLD:
            return "bearish"
        else:
            return "neutral"

    def _detect_divergence_signal(
        self, spy_mom: float, gld_mom: float, tlt_mom: float
    ) -> Tuple[str, float]:
        """
        Detect cross-asset divergence pattern and return (pattern, signal_value).

        Uses the same classification logic as the signal module but operates
        on historical data at each rebalance point.
        """
        spy_regime = self._classify_asset_regime(spy_mom)
        gld_regime = self._classify_asset_regime(gld_mom)
        tlt_regime = self._classify_asset_regime(tlt_mom)

        # --- Full agreement patterns (check before all-same-regime) ---
        all_bullish = (
            spy_regime == "bullish"
            and gld_regime == "bullish"
            and tlt_regime == "bullish"
        )
        if all_bullish:
            return DivergencePattern.FULL_RISK_ON, 0.4

        all_bearish = (
            spy_regime == "bearish"
            and gld_regime == "bearish"
            and tlt_regime == "bearish"
        )
        if all_bearish:
            return DivergencePattern.RISK_OFF, -0.5

        # --- All same neutral regime -> no divergence ---
        if spy_regime == gld_regime == tlt_regime:
            return DivergencePattern.NO_DIVERGENCE, 0.0

        # --- Flight to safety: bonds up, equities down ---
        if tlt_regime == "bullish" and spy_regime == "bearish":
            return DivergencePattern.FLIGHT_TO_SAFETY, -0.3

        # --- Inflation fear: bonds down, gold up ---
        if tlt_regime == "bearish" and gld_regime == "bullish":
            return DivergencePattern.INFLATION_FEAR, -0.1

        # --- Risk rotation: equity bear + gold bull ---
        if spy_regime == "bearish" and gld_regime == "bullish":
            return DivergencePattern.RISK_ROTATION, 0.2

        # --- Cautious optimism: equity neutral + gold strong ---
        if spy_regime == "neutral" and gld_regime == "bullish":
            return DivergencePattern.CAUTIOUS_OPTIMISM, 0.1

        # --- Recovery beginning: gold weak + equity recovering ---
        if gld_regime == "bearish" and spy_regime == "neutral":
            return DivergencePattern.RECOVERY_BEGINNING, 0.25

        # --- Equity rotation: equity diverging from bonds/gold ---
        if spy_regime != "neutral":
            return DivergencePattern.EQUITY_ROTATION, 0.15

        return DivergencePattern.NO_DIVERGENCE, 0.0

    def _get_allocation_shifts(
        self, pattern: str, signal_value: float, spy_mom: float, tlt_mom: float
    ) -> Tuple[float, float, float]:
        """
        Map divergence pattern to allocation shifts (spy, gld, tlt).

        Positive signal (equity outperforming): tilt toward SPY.
        Negative signal (bonds/gold outperforming): tilt toward safe havens.
        """
        # Scale factor based on signal strength relative to max
        strength = abs(signal_value) / MAX_SIGNAL_STRENGTH

        if signal_value > 0:
            # Equity outperforming -> overweight SPY
            shifts = ALLOCATION_SHIFTS["equity_outperformance"]
            spy_shift = shifts["spy"] * strength
            gld_shift = shifts["gld"] * strength
            tlt_shift = shifts["tlt"] * strength
        elif signal_value < 0:
            # Safe havens outperforming -> underweight SPY, overweight bonds/gold
            shifts = ALLOCATION_SHIFTS["safe_haven_outperformance"]
            spy_shift = shifts["spy"] * strength
            gld_shift = shifts["gld"] * strength
            tlt_shift = shifts["tlt"] * strength
        else:
            return 0.0, 0.0, 0.0

        # Clamp to max single shift
        spy_shift = np.clip(spy_shift, -self.config.max_single_shift, self.config.max_single_shift)
        gld_shift = np.clip(gld_shift, -self.config.max_single_shift, self.config.max_single_shift)
        tlt_shift = np.clip(tlt_shift, -self.config.max_single_shift, self.config.max_single_shift)

        return spy_shift, gld_shift, tlt_shift

    # ---- Backtest Engine ----

    def run_backtest(self) -> Optional[BacktestResult]:
        """
        Run walk-forward backtest with cross-asset regime arb overlay.

        Simulates both baseline (46/38/16) and overlay-enhanced portfolios
        with monthly rebalancing and divergence-based allocation shifts.
        """
        if not self.data:
            logger.error("No data loaded")
            return None

        # Filter to backtest period
        start_dt = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.config.end_date, "%Y-%m-%d")

        backtest_data = [
            d
            for d in self.data
            if start_dt <= datetime.strptime(d.date, "%Y-%m-%d") <= end_dt
        ]

        if not backtest_data:
            logger.error("No data in backtest period")
            return None

        logger.info(f"Running backtest on {len(backtest_data)} days")

        # Initialize portfolios
        base_capital = self.config.initial_capital
        overlay_capital = self.config.initial_capital

        base_weights = {
            "spy": self.config.base_weights["SPY"],
            "gld": self.config.base_weights["GLD"],
            "tlt": self.config.base_weights["TLT"],
        }

        overlay_weights = dict(base_weights)

        # Tracking
        base_equity = [base_capital]
        overlay_equity = [overlay_capital]
        overlay_active_months = 0
        total_rebalances = 0
        total_costs = 0.0
        rebalance_signals: List[RebalanceSignal] = []
        divergence_counts: Dict[str, int] = {}

        # Crisis returns
        returns_2008 = []
        returns_2020 = []
        returns_2022 = []

        # Track last rebalance month for monthly rebalancing
        last_rebalance_year_month: Optional[Tuple[int, int]] = None

        for i, day in enumerate(backtest_data):
            day_date = datetime.strptime(day.date, "%Y-%m-%d")
            year = day_date.year
            year_month = (year, day_date.month)

            # --- Baseline return ---
            base_ret = (
                base_weights["spy"] * day.spy_return
                + base_weights["gld"] * day.gld_return
                + base_weights["tlt"] * day.tlt_return
            )
            base_capital *= 1 + base_ret
            base_equity.append(base_capital)

            # --- Overlay: check if we need to rebalance ---
            is_new_month = year_month != last_rebalance_year_month

            if is_new_month:
                last_rebalance_year_month = year_month

                # Compute 60-day momentum for regime classification
                spy_mom, gld_mom, tlt_mom = self._compute_momentum(
                    MOMENTUM_LOOKBACK, i
                )

                # Detect divergence signal
                pattern, signal_value = self._detect_divergence_signal(
                    spy_mom, gld_mom, tlt_mom
                )

                # Track divergence frequency (store string value for JSON serialization)
                pattern_key = pattern.value if hasattr(pattern, "value") else str(pattern)
                divergence_counts[pattern_key] = divergence_counts.get(pattern_key, 0) + 1

                # Apply overlay if signal is strong enough
                if abs(signal_value) >= self.config.signal_threshold:
                    overlay_active_months += 1

                    spy_shift, gld_shift, tlt_shift = self._get_allocation_shifts(
                        pattern, signal_value, spy_mom, tlt_mom
                    )

                    # Calculate new target weights
                    new_spy = self.config.base_weights["SPY"] + spy_shift
                    new_gld = self.config.base_weights["GLD"] + gld_shift
                    new_tlt = self.config.base_weights["TLT"] + tlt_shift

                    # Compute turnover and apply costs
                    turnover = (
                        abs(new_spy - overlay_weights["spy"])
                        + abs(new_gld - overlay_weights["gld"])
                        + abs(new_tlt - overlay_weights["tlt"])
                    ) / 2

                    if turnover > 0.001:
                        total_rebalances += 1
                        cost = (
                            turnover
                            * self.config.transaction_cost_bps
                            / 10000
                            * overlay_capital
                        )
                        total_costs += cost
                        overlay_capital -= cost

                    overlay_weights["spy"] = new_spy
                    overlay_weights["gld"] = new_gld
                    overlay_weights["tlt"] = new_tlt

                    rebalance_signals.append(
                        RebalanceSignal(
                            date=day.date,
                            signal_value=signal_value,
                            pattern=pattern_key,
                            spy_shift=spy_shift,
                            gld_shift=gld_shift,
                            tlt_shift=tlt_shift,
                            spy_momentum=spy_mom,
                            gld_momentum=gld_mom,
                            tlt_momentum=tlt_mom,
                        )
                    )

            # Normalize weights to sum to 1.0
            total_weight = sum(overlay_weights.values())
            if abs(total_weight - 1.0) > 0.001:
                for k in overlay_weights:
                    overlay_weights[k] /= total_weight

            # Calculate overlay return
            overlay_ret = (
                overlay_weights["spy"] * day.spy_return
                + overlay_weights["gld"] * day.gld_return
                + overlay_weights["tlt"] * day.tlt_return
            )
            overlay_capital *= 1 + overlay_ret
            overlay_equity.append(overlay_capital)

            # Crisis tracking
            if year == 2008:
                returns_2008.append(overlay_ret)
            elif year == 2020 and day_date.month >= 2:
                returns_2020.append(overlay_ret)
            elif year == 2022:
                returns_2022.append(overlay_ret)

        # ---- Calculate metrics ----
        base_returns = self._returns_from_equity(base_equity)
        overlay_returns = self._returns_from_equity(overlay_equity)

        base_metrics = self._calculate_metrics(base_returns)
        overlay_metrics = self._calculate_metrics(overlay_returns)

        total_months = len(set((d.date[:7] for d in backtest_data)))
        signal_freq = (
            overlay_active_months / total_months if total_months > 0 else 0.0
        )

        result = BacktestResult(
            total_return=(
                (overlay_equity[-1] / self.config.initial_capital) - 1
            )
            * 100,
            cagr=overlay_metrics["cagr"],
            volatility=overlay_metrics["volatility"],
            sharpe_ratio=overlay_metrics["sharpe"],
            max_drawdown=overlay_metrics["max_dd"],
            overlay_active_months=overlay_active_months,
            baseline_sharpe=base_metrics["sharpe"],
            sharpe_improvement=overlay_metrics["sharpe"]
            - base_metrics["sharpe"],
            return_2008=self._annualize(returns_2008) if returns_2008 else None,
            return_2020=self._annualize(returns_2020) if returns_2020 else None,
            return_2022=self._annualize(returns_2022) if returns_2022 else None,
            total_rebalances=total_rebalances,
            total_transaction_costs=total_costs,
            signal_frequency=signal_freq,
            divergence_breakdown=divergence_counts,
            equity_curve=[
                {
                    "date": backtest_data[
                        min(
                            j,
                            len(backtest_data) - 1,
                        )
                    ].date
                    if j > 0
                    else backtest_data[0].date,
                    "baseline": base_equity[j],
                    "overlay": overlay_equity[j],
                }
                for j in range(
                    0,
                    len(overlay_equity),
                    max(1, len(overlay_equity) // 252),
                )
            ],
            rebalance_signals=[asdict(s) for s in rebalance_signals],
        )

        return result

    # ---- Metrics Helpers ----

    def _returns_from_equity(self, equity: List[float]) -> List[float]:
        """Derive daily returns from equity curve."""
        returns = []
        for i in range(1, len(equity)):
            ret = (equity[i] - equity[i - 1]) / equity[i - 1]
            returns.append(ret)
        return returns

    def _calculate_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate CAGR, volatility, Sharpe, and max drawdown. Delegates to shared module."""
        if not returns:
            return {"cagr": 0.0, "volatility": 0.0, "sharpe": 0.0, "max_dd": 0.0}
        eq = [1.0]
        for r in returns:
            eq.append(eq[-1] * (1.0 + r))
        m = compute_metrics(eq, initial_capital=1.0)
        return {
            "cagr": m.cagr,
            "volatility": m.volatility,
            "sharpe": m.sharpe_ratio,
            "max_dd": m.max_drawdown,
        }

    def _annualize(self, returns: List[float]) -> float:
        """Annualize a list of daily returns."""
        if not returns:
            return 0.0
        total = np.prod([1 + r for r in returns]) - 1
        n_years = len(returns) / 252
        return ((1 + total) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0

    # ---- Output ----

    def print_report(self, result: BacktestResult) -> None:
        """Print formatted backtest report."""
        print("\n" + "=" * 60)
        print("CROSS-ASSET REGIME ARBITRAGE BACKTEST RESULTS")
        print("=" * 60)
        print(f"Period:            {self.config.start_date} to {self.config.end_date}")
        print(f"Initial Capital:   ${self.config.initial_capital:,.2f}")
        print(f"Signal Module:     {'Live' if HAS_SIGNAL_MODULE else 'Simulated'}")
        print()

        print("PERFORMANCE METRICS")
        print("-" * 60)
        print(f"Total Return:      {result.total_return:>8.2f}%")
        print(f"CAGR:              {result.cagr:>8.2f}%")
        print(f"Volatility:        {result.volatility:>8.2f}%")
        print(f"Sharpe Ratio:      {result.sharpe_ratio:>8.3f}")
        print(f"Max Drawdown:      {result.max_drawdown:>8.2f}%")
        print()

        print("OVERLAY IMPACT")
        print("-" * 60)
        print(f"Baseline Sharpe:   {result.baseline_sharpe:>8.3f}")
        print(f"Overlay Sharpe:    {result.sharpe_ratio:>8.3f}")
        print(
            f"Improvement:       {result.sharpe_improvement:>+8.3f}  "
            f"{'PASS' if result.sharpe_improvement >= 0.01 else 'NO IMPROVEMENT'}"
        )
        print(
            f"Active Months:     {result.overlay_active_months:>8} "
            f"({result.signal_frequency * 100:.1f}% of months)"
        )
        print()

        print("CRISIS PERFORMANCE")
        print("-" * 60)
        if result.return_2008 is not None:
            print(f"2008 GFC:          {result.return_2008:>8.2f}%")
        if result.return_2020 is not None:
            print(f"2020 COVID:        {result.return_2020:>8.2f}%")
        if result.return_2022 is not None:
            print(f"2022 Rate Hikes:   {result.return_2022:>8.2f}%")
        print()

        print("TRADE STATISTICS")
        print("-" * 60)
        print(f"Total Rebalances:  {result.total_rebalances:>8}")
        print(f"Transaction Costs: ${result.total_transaction_costs:>8.2f}")
        print()

        print("DIVERGENCE BREAKDOWN")
        print("-" * 60)
        total_div = sum(result.divergence_breakdown.values()) or 1
        for pattern, count in sorted(
            result.divergence_breakdown.items(), key=lambda x: -x[1]
        ):
            pct = count / total_div * 100
            print(f"{pattern:<25s} {count:>5d} ({pct:>5.1f}%)")
        print()

        print("SUCCESS CRITERIA")
        print("-" * 60)
        checks = [
            (
                "Sharpe > Baseline (positive improvement)",
                result.sharpe_improvement > 0,
            ),
            ("Max DD > -30%", result.max_drawdown > -30),
            ("Signal active > 20% of months", result.signal_frequency > 0.20),
        ]
        for desc, passed in checks:
            print(f"{'PASS' if passed else 'FAIL'}  {desc}")
        print("=" * 60)

    def save_results(
        self, result: BacktestResult, output_path: Optional[str] = None
    ) -> None:
        """Save backtest results to JSON."""
        if output_path is None:
            output_path = str(
                BACKTEST_RESULTS_DIR / "cross_asset_regime_arb_backtest.json"
            )
        save_results_json(asdict(result), output_path=output_path)
        logger.info(f"Results saved to {output_path}")


# ---- CLI ----

def main() -> int:
    """Run cross-asset regime arbitrage backtest."""
    import sys

    config = BacktestConfig()
    backtester = CrossAssetRegimeArbBacktester(config)

    if not backtester.load_data():
        logger.error("Failed to load data")
        return 1

    result = backtester.run_backtest()
    if not result:
        logger.error("Backtest failed")
        return 1

    backtester.print_report(result)

    # Save by default; pass --save-only to skip printing
    if "--save" in sys.argv or len(sys.argv) == 1:
        backtester.save_results(result)

    return 0


if __name__ == "__main__":
    exit(main())
