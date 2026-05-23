"""
Alternative Data Signal Backtest — v9.00 Signal Validation

Validates whether the hardcoded regime->signal mappings used by the
ALTERNATIVE_DATA ensemble signal (bull=0.4, bear=-0.4, neutral=0.0,
crisis=-0.7) add or subtract alpha relative to the 46/38/16 baseline.

Since live alternative data (SEC EDGAR, NewsAPI, jobs data) is not
available historically, this backtest infers the regime from the VIX
level, falling back to SPY 60-day momentum when VIX data is absent.

Period: 2006-2026 (21 years, including GFC, COVID, 2022 rate hikes)
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.backtest.metrics import (
    BacktestConfig as _BaseConfig,
    BacktestResult,
    compute_metrics,
    save_results_json,
)
from src.paths import PRICES_JSON, BACKTEST_RESULTS_DIR
from src.signals.alternative_data_signal import AlternativeDataSignalGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig(_BaseConfig):
    """Configuration for alternative data signal backtest.

    Inherits from the canonical BacktestConfig (start_date, end_date,
    initial_capital, base_weights, rebalance_frequency, transaction_cost_bps).
    Backtest-specific fields are defined below.
    """

    # Overlay constraints
    max_signal_shift: float = 0.05  # 5% max shift per rebalance beyond target
    min_holding_period: int = 20  # ~1 month minimum hold

    # Regime thresholds (VIX-based)
    vix_bull_threshold: float = 15.0
    vix_bear_threshold: float = 20.0
    vix_crisis_threshold: float = 30.0


@dataclass
class DailyReturn:
    """Single day return data for backtest."""

    date: str
    spy_return: float
    gld_return: float
    tlt_return: float
    vix_spot: Optional[float] = None


class AlternativeDataBacktester:
    """
    Walk-forward backtest for the ALTERNATIVE_DATA ensemble signal.

    Simulates the continuous composite_score signal used in the ensemble
    voter to determine whether it adds or subtracts alpha over the
    46/38/16 baseline across 2006-2026.

    Signal computation: np.clip(spy_60d_return * 2.0, -1, 1) — a
    continuous passthrough that mirrors the production code path
    (ensemble_voter.py reads composite_score and clips to [-1, 1]).
    The old discrete REGIME_SIGNAL_MAP is kept only as a reference.
    """

    # Legacy regime->signal mapping (kept for reference; NOT used for signal computation)
    REGIME_SIGNAL_MAP = {
        "bull": 0.4,
        "bear": -0.4,
        "neutral": 0.0,
        "crisis": -0.7,
    }

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.data: List[DailyReturn] = []
        self._signal_generator = AlternativeDataSignalGenerator()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, data_path: Optional[str] = None) -> bool:
        """Load historical price data from PRICES_JSON."""
        try:
            prices_path = PRICES_JSON

            if prices_path.exists():
                with open(prices_path) as f:
                    prices_data = json.load(f)

                self._process_price_data(prices_data)
                logger.info(
                    "Loaded %d days of price data (VIX unavailable, "
                    "using SPY return proxy)",
                    len(self.data),
                )
                return True

            logger.error("No price data found at %s", prices_path)
            return False

        except Exception as e:
            logger.error("Failed to load data: %s", e)
            return False

    def _process_price_data(self, prices_data: Dict) -> None:
        """Process raw price data into daily returns.

        Expects the standard prices.json format:
            {symbol: [{"d": "YYYY-MM-DD", "p": price}, ...]}
        """
        spy_data = prices_data.get("SPY", [])
        gld_data = prices_data.get("GLD", [])
        tlt_data = prices_data.get("TLT", [])

        if not spy_data:
            logger.error("No SPY data found in prices.json")
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
                [spy_prev, spy_curr, gld_prev, gld_curr, tlt_prev, tlt_curr]
            ):
                self.data.append(
                    DailyReturn(
                        date=date,
                        spy_return=(spy_curr - spy_prev) / spy_prev,
                        gld_return=(gld_curr - gld_prev) / gld_prev,
                        tlt_return=(tlt_curr - tlt_prev) / tlt_prev,
                    )
                )

    # ------------------------------------------------------------------
    # Regime inference
    # ------------------------------------------------------------------

    def infer_regime_from_spy_return(
        self, spy_60d_return: float
    ) -> str:
        """Infer market regime from SPY 60-day return using production code.

        Uses AlternativeDataSignalGenerator._determine_regime() to classify
        the composite_score (scaled SPY return) into regime buckets.

        Production regime labels are mapped to backtest regime labels:
            risk_on  -> bull
            neutral  -> neutral
            risk_off -> bear

        Note: the production classifier does not produce a "crisis" label;
        strongly negative returns map to "bear".
        """
        composite_score = float(np.clip(spy_60d_return * 2.0, -1.0, 1.0))
        production_regime = self._signal_generator._determine_regime(composite_score)
        regime_map = {"risk_on": "bull", "risk_off": "bear", "neutral": "neutral"}
        return regime_map.get(production_regime, "neutral")

    def get_signal_and_regime(
        self, day: DailyReturn, past_60d_returns: List[float]
    ) -> Tuple[str, float]:
        """Compute regime and signal value for a given day.

        Infers the regime using SPY 60-day return as proxy (VIX data
        is not available in the price dataset), then computes a continuous
        signal value that mirrors the production code path:
        np.clip(spy_60d_return * 2.0, -1, 1).

        This replaces the old discrete REGIME_SIGNAL_MAP which quantized
        the signal into 4 fixed values, losing information about the
        strength of the underlying regime.

        Returns:
            (regime_name, signal_value)
        """
        spy_60d_return = (
            np.prod(1 + np.array(past_60d_returns[-60:])) - 1
            if len(past_60d_returns) >= 60
            else np.prod(1 + np.array(past_60d_returns)) - 1
        )

        regime = self.infer_regime_from_spy_return(spy_60d_return)
        # Continuous signal: scale 60d return to [-1, 1] range
        signal = float(np.clip(spy_60d_return * 2.0, -1.0, 1.0))
        return regime, signal

    # ------------------------------------------------------------------
    # Allocation shifts
    # ------------------------------------------------------------------

    def get_allocation_shifts(
        self, signal_value: float
    ) -> Tuple[float, float, float]:
        """Convert signal value to asset allocation shifts.

        Positive signal (bull): tilt toward equities
        Negative signal (bear/crisis): tilt toward safe havens
        Neutral signal: no shift

        Returns:
            (spy_shift, gld_shift, tlt_shift) as decimal fractions
        """
        if signal_value > 0:
            # Bull: SPY+3%, GLD-2%, TLT-1%
            return 0.03, -0.02, -0.01
        elif signal_value < 0 and signal_value > -0.5:
            # Bear: SPY-3%, GLD+2%, TLT+1%
            return -0.03, 0.02, 0.01
        elif signal_value <= -0.5:
            # Crisis: SPY-5%, GLD+3%, TLT+2%
            return -0.05, 0.03, 0.02
        else:
            return 0.0, 0.0, 0.0

    # ------------------------------------------------------------------
    # Core backtest
    # ------------------------------------------------------------------

    def run_backtest(self) -> Optional[BacktestResult]:
        """Run backtest comparing baseline vs alternative-data overlay.

        Simulates monthly rebalancing with the ALTERNATIVE_DATA signal
        overlay applied on rebalance days. Baseline is the 46/38/16
        static portfolio.
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

        logger.info(
            "Running alternative data backtest on %d days (%s to %s)",
            len(backtest_data),
            self.config.start_date,
            self.config.end_date,
        )

        # -- Initialise portfolios -----------------------------------------
        base_capital = self.config.initial_capital
        overlay_capital = self.config.initial_capital

        base_weights = {
            "spy": self.config.base_weights["SPY"],
            "gld": self.config.base_weights["GLD"],
            "tlt": self.config.base_weights["TLT"],
        }

        overlay_weights = dict(base_weights)

        # -- Tracking variables --------------------------------------------
        base_equity = [base_capital]
        overlay_equity = [overlay_capital]

        last_rebalance_date = None
        total_rebalances = 0
        rebalance_sizes = []
        total_costs = 0.0
        overlay_active_months = 0
        monthly_signals_applied = 0

        regime_distribution: Dict[str, int] = {
            "bull": 0,
            "bear": 0,
            "neutral": 0,
            "crisis": 0,
        }
        regime_returns: Dict[str, List[float]] = {
            "bull": [],
            "bear": [],
            "neutral": [],
            "crisis": [],
        }

        returns_2008 = []
        returns_2020 = []
        returns_2022 = []

        # -- Rolling buffer for SPY 60-day return -------------------------
        spy_rolling_returns: List[float] = []

        for i, day in enumerate(backtest_data):
            day_date = datetime.strptime(day.date, "%Y-%m-%d")
            year = day_date.year

            # Accumulate rolling returns
            spy_rolling_returns.append(day.spy_return)

            # -- Baseline return -------------------------------------------
            base_ret = (
                base_weights["spy"] * day.spy_return
                + base_weights["gld"] * day.gld_return
                + base_weights["tlt"] * day.tlt_return
            )
            base_capital *= 1 + base_ret
            base_equity.append(base_capital)

            # -- Determine if this is a rebalance month --------------------
            is_rebalance_day = self._is_rebalance_day(
                day.date, last_rebalance_date
            )

            # Infer regime and signal
            regime, signal = self.get_signal_and_regime(
                day, spy_rolling_returns
            )

            # Track regime distribution
            regime_distribution[regime] = (
                regime_distribution.get(regime, 0) + 1
            )

            if is_rebalance_day:
                last_rebalance_date = day.date

                # Apply overlay: shift weights based on signal sign
                spy_shift, gld_shift, tlt_shift = (
                    self.get_allocation_shifts(signal)
                )

                if abs(signal) > 0.001:
                    monthly_signals_applied += 1

                target_spy = (
                    self.config.base_weights["SPY"] + spy_shift
                )
                target_gld = (
                    self.config.base_weights["GLD"] + gld_shift
                )
                target_tlt = (
                    self.config.base_weights["TLT"] + tlt_shift
                )

                # Calculate turnover and costs
                turnover = (
                    abs(target_spy - overlay_weights["spy"])
                    + abs(target_gld - overlay_weights["gld"])
                    + abs(target_tlt - overlay_weights["tlt"])
                ) / 2

                if turnover > 0.001:
                    total_rebalances += 1
                    rebalance_sizes.append(turnover)
                    cost = (
                        turnover
                        * self.config.transaction_cost_bps
                        / 10000
                        * overlay_capital
                    )
                    total_costs += cost
                    overlay_capital -= cost

                overlay_weights["spy"] = target_spy
                overlay_weights["gld"] = target_gld
                overlay_weights["tlt"] = target_tlt

                # Normalise weights to sum to 1.0
                total_w = sum(overlay_weights.values())
                if abs(total_w - 1.0) > 0.001:
                    for k in overlay_weights:
                        overlay_weights[k] /= total_w

                # Track active months (signal applied means overlay active)
                if abs(signal) > 0.001:
                    overlay_active_months += 1

            # -- Overlay return --------------------------------------------
            overlay_ret = (
                overlay_weights["spy"] * day.spy_return
                + overlay_weights["gld"] * day.gld_return
                + overlay_weights["tlt"] * day.tlt_return
            )
            overlay_capital *= 1 + overlay_ret
            overlay_equity.append(overlay_capital)

            # -- Track regime-based returns --------------------------------
            regime_returns.setdefault(regime, []).append(overlay_ret)

            # -- Crisis period tracking ------------------------------------
            if year == 2008:
                returns_2008.append(overlay_ret)
            elif year == 2020 and day_date.month >= 2:
                returns_2020.append(overlay_ret)
            elif year == 2022:
                returns_2022.append(overlay_ret)

        # -- Compute metrics -----------------------------------------------
        base_ret_list = self._returns_from_equity(base_equity)
        overlay_ret_list = self._returns_from_equity(overlay_equity)

        base_metrics = self._calculate_metrics(base_ret_list)
        overlay_metrics = self._calculate_metrics(overlay_ret_list)

        avg_regime_returns = self._annualize_regime_returns(
            regime_returns
        )

        total_months = max(
            len([d for d in backtest_data if d.date.endswith("-01")]),
            1,
        )
        overlay_active_pct = (
            overlay_active_months / total_months * 100
        )

        result = BacktestResult(
            total_return=(
                overlay_equity[-1] / self.config.initial_capital - 1
            )
            * 100,
            cagr=overlay_metrics["cagr"],
            volatility=overlay_metrics["volatility"],
            sharpe_ratio=overlay_metrics["sharpe"],
            max_drawdown=overlay_metrics["max_dd"],
            baseline_sharpe=base_metrics["sharpe"],
            sharpe_improvement=(
                overlay_metrics["sharpe"] - base_metrics["sharpe"]
            ),
            total_rebalances=total_rebalances,
            total_transaction_costs=total_costs,
            crisis_returns={
                "2008": (
                    self._annualize_returns(returns_2008)
                    if returns_2008
                    else None
                ),
                "2020": (
                    self._annualize_returns(returns_2020)
                    if returns_2020
                    else None
                ),
                "2022": (
                    self._annualize_returns(returns_2022)
                    if returns_2022
                    else None
                ),
            },
            extras={
                "overlay_active_months": overlay_active_months,
                "overlay_active_pct": overlay_active_pct,
                "avg_rebalance_size": (
                    np.mean(rebalance_sizes) if rebalance_sizes else 0
                ),
                "regime_distribution": regime_distribution,
                "regime_returns": avg_regime_returns,
                "equity_curve": [
                    {
                        "date": backtest_data[
                            min(i, len(backtest_data) - 1)
                        ].date
                        if i > 0
                        else backtest_data[0].date,
                        "baseline": base_equity[i],
                        "overlay": overlay_equity[i],
                    }
                    for i in range(
                        0,
                        len(overlay_equity),
                        max(1, len(overlay_equity) // 252),
                    )
                ],
            },
        )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rebalance_day(
        date_str: str, last_rebalance: Optional[str]
    ) -> bool:
        """Check if this is a monthly rebalance day (first trading day
        of month or when month changes from last rebalance)."""
        dt = datetime.strptime(date_str, "%Y-%m-%d")

        # First trading day of any month
        if dt.day <= 3:
            return True

        # Month boundary crossed since last rebalance
        if last_rebalance:
            last_dt = datetime.strptime(last_rebalance, "%Y-%m-%d")
            if dt.month != last_dt.month or dt.year != last_dt.year:
                return True

        return False

    @staticmethod
    def _returns_from_equity(equity: List[float]) -> List[float]:
        """Calculate daily returns from an equity curve."""
        returns = []
        for i in range(1, len(equity)):
            returns.append(
                (equity[i] - equity[i - 1]) / equity[i - 1]
            )
        return returns

    @staticmethod
    def _calculate_metrics(returns: List[float]) -> Dict:
        """Calculate CAGR, volatility, Sharpe, max drawdown. Delegates to shared module."""
        if not returns:
            return {"cagr": 0, "volatility": 0, "sharpe": 0, "max_dd": 0}
        # Build equity curve from returns
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

    @staticmethod
    def _annualize_returns(returns: List[float]) -> float:
        """Annualise a list of daily returns."""
        if not returns:
            return 0.0
        total = np.prod([1 + r for r in returns]) - 1
        n_years = len(returns) / 252
        return (
            ((1 + total) ** (1 / n_years) - 1) * 100
            if n_years > 0
            else 0.0
        )

    @staticmethod
    def _annualize_regime_returns(
        regime_returns: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Compute annualised mean return per regime."""
        result = {}
        for regime, rets in regime_returns.items():
            if rets:
                result[regime] = np.mean(rets) * 252 * 100
            else:
                result[regime] = 0.0
        return result

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_results(
        self,
        result: BacktestResult,
        output_path: Optional[str] = None,
    ) -> None:
        """Save backtest results to JSON."""
        from dataclasses import asdict

        if output_path is None:
            output_path = str(
                BACKTEST_RESULTS_DIR
                / "alternative_data_backtest.json"
            )

        save_results_json(asdict(result), output_path=output_path)
        logger.info("Results saved to %s", output_path)

    def print_report(self, result: BacktestResult) -> None:
        """Print a formatted backtest report."""
        total_days = len(self.data)
        total_months = max(
            len(
                [
                    d
                    for d in self.data
                    if d.date.endswith("-01")
                ]
            ),
            1,
        )

        print("\n" + "=" * 64)
        print(
            "  ALTERNATIVE DATA SIGNAL BACKTEST  "
        )
        print("  Validating hardcoded regime->signal mappings")
        print("=" * 64)
        print(
            f"  Period:         {self.config.start_date} to "
            f"{self.config.end_date}"
        )
        print(
            f"  Data days:      {total_days} "
            f"({total_months} months)"
        )
        print(
            f"  Initial cap:    "
            f"${self.config.initial_capital:,.0f}"
        )
        print(f"  Baseline:       46/38/16 (SPY/GLD/TLT)")
        print()

        # -- Performance metrics --
        print("  PERFORMANCE METRICS")
        print("  " + "-" * 60)
        print(
            f"  Total Return:       "
            f"{result.total_return:>9.2f}%"
        )
        print(f"  CAGR:               {result.cagr:>9.2f}%")
        print(
            f"  Volatility:         "
            f"{result.volatility:>9.2f}%"
        )
        print(
            f"  Sharpe Ratio:       "
            f"{result.sharpe_ratio:>9.3f}"
        )
        print(
            f"  Max Drawdown:       "
            f"{result.max_drawdown:>9.2f}%"
        )
        print()

        # -- Overlay impact --
        print("  OVERLAY IMPACT")
        print("  " + "-" * 60)
        print(
            f"  Baseline Sharpe:    "
            f"{result.baseline_sharpe:>9.3f}"
        )
        print(
            f"  Overlay Sharpe:     "
            f"{result.sharpe_ratio:>9.3f}"
        )
        improvement = result.sharpe_improvement
        imp_label = (
            "POSITIVE" if improvement > 0 else "NEGATIVE"
        )
        print(
            f"  Improvement:        "
            f"{improvement:>+9.3f}  ({imp_label})"
        )
        print(
            f"  Active months:      "
            f"{result.extras['overlay_active_months']:>9} "
            f"({result.extras['overlay_active_pct']:.1f}%)"
        )
        print()

        # -- Crisis performance --
        print("  CRISIS PERFORMANCE (annualised)")
        print("  " + "-" * 60)
        crisis = result.crisis_returns or {}
        for label, key in [
            ("2008 GFC", "2008"),
            ("2020 COVID", "2020"),
            ("2022 Rate Hikes", "2022"),
        ]:
            val = crisis.get(key)
            if val is not None:
                print(f"  {label:20s}  {val:>+9.2f}%")
        print()

        # -- Trade stats --
        print("  TRADE STATISTICS")
        print("  " + "-" * 60)
        print(
            f"  Total rebalances:   "
            f"{result.total_rebalances:>9}"
        )
        print(
            f"  Avg rebalance size: "
            f"{result.extras['avg_rebalance_size'] * 100:>9.2f}%"
        )
        print(
            f"  Transaction costs:  "
            f"${result.total_transaction_costs:>9.2f}"
        )
        print()

        # -- Regime distribution --
        total_regime = sum(result.extras["regime_distribution"].values())
        print("  REGIME DISTRIBUTION")
        print("  " + "-" * 60)
        for regime in ["bull", "neutral", "bear", "crisis"]:
            count = result.extras["regime_distribution"].get(regime, 0)
            pct = count / total_regime * 100 if total_regime else 0
            ann_ret = result.extras["regime_returns"].get(regime, 0.0)
            signal = self.REGIME_SIGNAL_MAP.get(regime, 0.0)
            print(
                f"  {regime:10s}  count={count:>5} "
                f"({pct:>5.1f}%)  "
                f"return={ann_ret:>+7.2f}%  "
                f"signal={signal:>+.1f}"
            )
        print()

        # -- Regime->signal mapping table --
        print("  REGIME -> SIGNAL MAPPING (ensemble_voter.py L581)")
        print("  " + "-" * 60)
        for regime, signal in self.REGIME_SIGNAL_MAP.items():
            ann_ret = result.extras["regime_returns"].get(regime, 0.0)
            verdict = "OK" if (
                (signal > 0 and ann_ret > 0)
                or (signal < 0 and ann_ret < 0)
                or (signal == 0 and abs(ann_ret) < 1.0)
            ) else "MISMATCH"
            print(
                f"  {regime:10s} -> signal {signal:>+4.1f}  "
                f"  realised {ann_ret:>+7.2f}%  "
                f"[{verdict}]"
            )
        print()

        # -- Success criteria --
        print("  SUCCESS CRITERIA")
        print("  " + "-" * 60)
        checks = [
            (
                "Sharpe >= baseline (signal adds alpha)",
                improvement >= 0,
            ),
            (
                "Max DD <= baseline max DD (-26.2%)",
                result.max_drawdown >= -26.2,
            ),
            (
                "Reasonable rebalance count (< 400)",
                result.total_rebalances < 400,
            ),
        ]
        for desc, passed in checks:
            mark = "PASS" if passed else "FAIL"
            print(f"  [{mark}] {desc}")
        print("=" * 64)


def main() -> int:
    """Run alternative data signal backtest from CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Alternative Data Signal Backtest"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run"],
        help="Command to execute (default: run)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON",
    )
    args = parser.parse_args()

    backtester = AlternativeDataBacktester()

    if not backtester.load_data():
        logger.error("Failed to load price data")
        return 1

    result = backtester.run_backtest()
    if not result:
        logger.error("Backtest failed")
        return 1

    backtester.print_report(result)

    if args.save:
        backtester.save_results(result)

    return 0


if __name__ == "__main__":
    exit(main())
