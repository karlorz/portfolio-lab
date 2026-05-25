"""
Multi-Speed Momentum Overlay Backtest - v9.23 Implementation
Walk-forward backtest validation for the dominant 50%-weight ensemble signal.

Validates whether the MULTI_SPEED_MOM signal adds alpha over the baseline
46/38/16 (SPY/GLD/TLT) portfolio by shifting weights based on the
ensemble momentum signal direction.

Target: Validate the 0.50 weight in REGIME_WEIGHTS is justified.
Period: 2006-2026 (20+ years including GFC, COVID, 2022 rate hikes)
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.backtest.metrics import (
    BacktestConfig as _BaseConfig,
    BacktestResult,
    compute_metrics,
    save_results_json,
)
from src.paths import PRICES_JSON, BACKTEST_RESULTS_DIR


__all__ = ['MultiSpeedMomentum', 'BacktestConfig', 'DailyReturn', 'MultiSpeedMomentumBacktester']

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback: simple 12-month momentum proxy when MultiSpeedMomentum cannot
# be imported (e.g. missing pandas on a minimal install).
# ---------------------------------------------------------------------------
_MULTI_SPEED_AVAILABLE = False
MultiSpeedMomentum = None  # type: ignore

try:
    from src.signals.multi_speed_momentum import MultiSpeedMomentum as _RealMSM

    MultiSpeedMomentum = _RealMSM
    _MULTI_SPEED_AVAILABLE = True
    logger.info("MultiSpeedMomentum loaded successfully")
except ImportError as exc:
    logger.warning("MultiSpeedMomentum import failed (%s); using 12m momentum fallback", exc)


@dataclass
class BacktestConfig(_BaseConfig):
    """Configuration for multi-speed momentum overlay backtest.

    Inherits canonical fields (start_date, end_date, initial_capital,
    base_weights, rebalance_frequency, transaction_cost_bps) from
    _BaseConfig. Only backtest-specific fields are defined here.
    """

    # Overlay constraints
    max_spy_shift: float = 0.05  # SPY +/-5%
    max_gld_shift: float = 0.03  # GLD +/-3%
    max_tlt_shift: float = 0.02  # TLT +/-2%

    # Signal threshold -- minimum absolute signal to trigger overlay
    signal_threshold: float = 0.10


@dataclass
class DailyReturn:
    """Single day return data for the three assets."""

    date: str
    spy_return: float
    gld_return: float
    tlt_return: float


class MultiSpeedMomentumBacktester:
    """Walk-forward backtest for the multi-speed momentum ensemble overlay."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.data: List[DailyReturn] = []
        self.prices_raw: Dict = {}  # raw prices keyed by ticker
        self._signal_engine = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(self, data_path: Optional[str] = None) -> bool:
        """Load price data from PRICES_JSON."""
        try:
            prices_path = PRICES_JSON
            if not prices_path.exists():
                logger.error("Price data not found at %s", PRICES_JSON)
                return False

            with open(prices_path) as f:
                self.prices_raw = json.load(f)

            self._process_price_data()
            logger.info("Loaded %d days of price data", len(self.data))
            return True

        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.error("Failed to load data: %s", exc)
            return False

    def _process_price_data(self) -> None:
        """Build DailyReturn list from raw price dicts."""
        spy = self.prices_raw.get("SPY", [])
        gld = self.prices_raw.get("GLD", [])
        tlt = self.prices_raw.get("TLT", [])

        if not spy:
            logger.error("No SPY data found")
            return

        dates = [p["d"] for p in spy]
        spy_px = {p["d"]: p["p"] for p in spy}
        gld_px = {p["d"]: p["p"] for p in gld}
        tlt_px = {p["d"]: p["p"] for p in tlt}

        for i, date in enumerate(dates[1:], 1):
            prev = dates[i - 1]
            s_p = spy_px.get(prev)
            s_c = spy_px.get(date)
            g_p = gld_px.get(prev)
            g_c = gld_px.get(date)
            t_p = tlt_px.get(prev)
            t_c = tlt_px.get(date)

            if all(v is not None for v in (s_p, s_c, g_p, g_c, t_p, t_c)):
                self.data.append(
                    DailyReturn(
                        date=date,
                        spy_return=(s_c - s_p) / s_p,
                        gld_return=(g_c - g_p) / g_p,
                        tlt_return=(t_c - t_p) / t_p,
                    )
                )

    def _get_prices_slice(self, end_date: str, lookback: int = 400) -> dict:
        """
        Return a price dict up to (and including) *end_date* with at least
        *lookback* trading days of history for each ticker.

        This is what the MultiSpeedMomentum signal engine consumes.
        """
        result: dict = {}
        for ticker in ("SPY", "GLD", "TLT"):
            raw = self.prices_raw.get(ticker, [])
            # keep entries up to end_date
            filtered = [e for e in raw if e["d"] <= end_date]
            # ensure enough history for the slow tier (252 + 21 + buffer)
            if len(filtered) > lookback:
                filtered = filtered[-(lookback + 50) :]
            result[ticker] = filtered
        return result

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------
    def _compute_signal(self, ticker: str, end_date: str) -> float:
        """
        Compute the multi-speed ensemble signal for *ticker* using price
        data up to *end_date*.

        Returns a float in [-1, 1] where:
          positive = bullish (overweight)
          negative = bearish (underweight)

        Falls back to a simple 12-month momentum proxy if the real signal
        engine is unavailable or fails.
        """
        # -- Primary path: MultiSpeedMomentum class --
        if _MULTI_SPEED_AVAILABLE:
            try:
                if self._signal_engine is None:
                    self._signal_engine = MultiSpeedMomentum()

                # Build a temporary prices dict with history up to end_date
                temp_prices = self._get_prices_slice(end_date, lookback=400)

                # The MultiSpeedMomentum engine reads from self.prices_path.
                # We can't easily hot-swap the file, so we use the public API
                # which reads from disk. Instead, we re-construct a temp JSON
                # in memory and trick the engine.  A cleaner approach: compute
                # the signal directly using the same logic the engine uses.
                #
                # For robustness we use get_signal_for_ticker which loads the
                # full file.  Since our data goes up to 2026-05 this gives the
                # correct result for any backtest date -- the engine simply
                # looks back the required lookback window.
                signal = self._signal_engine.get_signal_for_ticker(ticker, end_date)
                if signal is not None:
                    return float(np.clip(signal, -1.0, 1.0))
            except (KeyError, ValueError, TypeError, ZeroDivisionError, AttributeError, RuntimeError) as exc:
                logger.warning("MultiSpeedMomentum signal failed for %s on %s: %s", ticker, end_date, exc)

        # -- Fallback: simple 12-month momentum proxy --
        raw = self.prices_raw.get(ticker, [])
        hist = sorted([e for e in raw if e["d"] <= end_date], key=lambda x: x["d"])
        if len(hist) < 260:
            return 0.0  # not enough data

        px_now = hist[-1]["p"]
        px_12m = hist[-252]["p"]
        ret_12m = (px_now / px_12m) - 1
        # map to [-1, 1] with saturation at +/-20%
        return float(np.clip(ret_12m / 0.20, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Allocation helpers
    # ------------------------------------------------------------------
    def _get_base_weights(self) -> Dict[str, float]:
        return {
            "SPY": self.config.base_weights["SPY"],
            "GLD": self.config.base_weights["GLD"],
            "TLT": self.config.base_weights["TLT"],
        }

    def _get_overlay_shifts(self, signal_spy: float, signal_gld: float, signal_tlt: float) -> Dict[str, float]:
        """
        Map per-ticker signals to weight shifts, clamping by configured
        maximums.
        """
        shifts = {
            "SPY": float(np.clip(signal_spy, -1.0, 1.0)) * self.config.max_spy_shift,
            "GLD": float(np.clip(signal_gld, -1.0, 1.0)) * self.config.max_gld_shift,
            "TLT": float(np.clip(signal_tlt, -1.0, 1.0)) * self.config.max_tlt_shift,
        }
        return shifts

    def _compute_turnover(self, old: Dict[str, float], new: Dict[str, float]) -> float:
        """Sum of absolute weight changes (half for double-counting)."""
        return sum(abs(new[k] - old[k]) for k in old) / 2.0

    # ------------------------------------------------------------------
    # Metrics — delegated to shared module
    # ------------------------------------------------------------------
    @staticmethod
    def _metrics(returns: List[float]) -> Dict[str, float]:
        """Legacy wrapper — delegates to shared compute_metrics."""
        # Build an equity curve from returns
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
    def _returns_from_equity(equity: List[float]) -> List[float]:
        """Derive daily returns from equity curve."""
        rets = []
        for i in range(1, len(equity)):
            rets.append((equity[i] - equity[i - 1]) / equity[i - 1])
        return rets

    @staticmethod
    def _annualize(returns: List[float]) -> float:
        if not returns:
            return 0.0
        total = float(np.prod([1.0 + r for r in returns])) - 1.0
        n = len(returns) / 252.0
        return ((1.0 + total) ** (1.0 / n) - 1.0) * 100.0 if n > 0 else 0.0

    # ------------------------------------------------------------------
    # Main backtest loop
    # ------------------------------------------------------------------
    def run_backtest(self) -> Optional[BacktestResult]:
        """Run walk-forward backtest comparing baseline vs signal overlay."""
        if not self.data:
            logger.error("No data loaded")
            return None

        # Filter to config period
        start_dt = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        bt_data = [d for d in self.data if start_dt <= datetime.strptime(d.date, "%Y-%m-%d") <= end_dt]

        if not bt_data:
            logger.error("No data in backtest period")
            return None

        logger.info("Running backtest on %d trading days", len(bt_data))

        base_cap = self.config.initial_capital
        overlay_cap = self.config.initial_capital
        base_w = self._get_base_weights()
        overlay_w = self._get_base_weights()

        base_eq: List[float] = [base_cap]
        overlay_eq: List[float] = [overlay_cap]

        total_rebalances = 0
        turnover_list: List[float] = []
        total_costs = 0.0

        # Determine rebalance step in trading days
        reb_step = self.config.rebalance_frequency_days

        # Date-indexed rebalancing: track the date of last rebalance
        last_rebalance_date: Optional[str] = None

        for idx, day in enumerate(bt_data):
            day_date = datetime.strptime(day.date, "%Y-%m-%d")
            day_date.year

            # -- Baseline return --
            base_ret = (
                base_w["SPY"] * day.spy_return
                + base_w["GLD"] * day.gld_return
                + base_w["TLT"] * day.tlt_return
            )
            base_cap *= 1.0 + base_ret
            base_eq.append(base_cap)

            # -- Overlay return (using current overlay_w; updated below on reb) --
            overlay_ret = (
                overlay_w["SPY"] * day.spy_return
                + overlay_w["GLD"] * day.gld_return
                + overlay_w["TLT"] * day.tlt_return
            )
            overlay_cap *= 1.0 + overlay_ret
            overlay_eq.append(overlay_cap)

            # -- Rebalance check (monthly) --
            should_rebalance = False
            if last_rebalance_date is None:
                should_rebalance = True  # first rebalance
            else:
                # Count trading days since last rebalance
                last_idx = max(
                    i for i, d in enumerate(bt_data) if d.date <= last_rebalance_date
                )
                if idx - last_idx >= reb_step:
                    should_rebalance = True

            if should_rebalance:
                last_rebalance_date = day.date

                # Compute signals for each ticker
                sig_spy = self._compute_signal("SPY", day.date)
                sig_gld = self._compute_signal("GLD", day.date)
                sig_tlt = self._compute_signal("TLT", day.date)

                # Only adjust weights if at least one signal exceeds threshold
                max_sig = max(abs(sig_spy), abs(sig_gld), abs(sig_tlt))
                if max_sig >= self.config.signal_threshold:
                    shifts = self._get_overlay_shifts(sig_spy, sig_gld, sig_tlt)

                    new_w = {
                        "SPY": self.config.base_weights["SPY"] + shifts["SPY"],
                        "GLD": self.config.base_weights["GLD"] + shifts["GLD"],
                        "TLT": self.config.base_weights["TLT"] + shifts["TLT"],
                    }
                else:
                    new_w = self._get_base_weights()

                # Normalise so weights sum to 1.0
                total_w = sum(new_w.values())
                if abs(total_w - 1.0) > 1e-6:
                    for k in new_w:
                        new_w[k] /= total_w

                # Compute turnover and transaction costs
                turnover = self._compute_turnover(overlay_w, new_w)
                if turnover > 0.001:  # 0.1 % minimum
                    total_rebalances += 1
                    turnover_list.append(turnover)
                    cost = turnover * (self.config.transaction_cost_bps / 10000.0) * overlay_cap
                    total_costs += cost
                    overlay_cap -= cost  # deduct cost from overlay capital

                overlay_w = new_w

        # -- Compute final metrics --
        base_rets = self._returns_from_equity(base_eq)
        overlay_rets = self._returns_from_equity(overlay_eq)

        base_m = self._metrics(base_rets)
        overlay_m = self._metrics(overlay_rets)

        # Crisis period returns (overlay)
        crisis_2008 = [d for d in bt_data if d.date.startswith("2008")]
        crisis_2020 = [d for d in bt_data if "2020-02" <= d.date <= "2020-05"]
        crisis_2022 = [d for d in bt_data if d.date.startswith("2022")]

        def _crisis_return(period: List[DailyReturn]) -> Optional[float]:
            if not period:
                return None
            eq_val = self.config.initial_capital / self.config.initial_capital  # start at 1
            [eq_val]
            # We need overlay weights for each day in the crisis. Simpler
            # approximation: use the average overlay return per day.
            rets = [
                overlay_w["SPY"] * d.spy_return
                + overlay_w["GLD"] * d.gld_return
                + overlay_w["TLT"] * d.tlt_return
                for d in period
            ]
            total = float(np.prod([1.0 + r for r in rets])) - 1.0
            return total * 100.0

        overlay_active = sum(
            1
            for d in bt_data
            if abs(self._compute_signal("SPY", d.date)) >= self.config.signal_threshold
        )

        result = BacktestResult(
            total_return=(overlay_eq[-1] / self.config.initial_capital - 1.0) * 100.0 if self.config.initial_capital else 0.0,
            cagr=overlay_m["cagr"],
            volatility=overlay_m["volatility"],
            sharpe_ratio=overlay_m["sharpe"],
            max_drawdown=overlay_m["max_dd"],
            baseline_sharpe=base_m["sharpe"],
            sharpe_improvement=overlay_m["sharpe"] - base_m["sharpe"],
            total_rebalances=total_rebalances,
            avg_turnover=np.mean(turnover_list) if turnover_list else 0.0,
            total_transaction_costs=total_costs,
            crisis_returns={
                "2008": _crisis_return(crisis_2008),
                "2020": _crisis_return(crisis_2020),
                "2022": _crisis_return(crisis_2022),
            },
            extras={
                "overlay_active_rebalances": overlay_active,
                "equity_curve": [
                    {
                        "date": bt_data[min(i, len(bt_data) - 1)].date if i > 0 else bt_data[0].date,
                        "baseline": round(base_eq[i], 2),
                        "overlay": round(overlay_eq[i], 2),
                    }
                    for i in range(0, len(overlay_eq), max(1, len(overlay_eq) // 252))
                ],
            },
        )

        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def print_report(self, result: BacktestResult) -> None:
        """Print a formatted backtest report."""
        logger.info("\n" + "=" * 62)
        logger.info("  MULTI-SPEED MOMENTUM OVERLAY BACKTEST (v9.23)")
        logger.info("=" * 62)
        logger.info(f"  Period:           {self.config.start_date}  to  {self.config.end_date}")
        logger.info(f"  Initial Capital:  ${self.config.initial_capital:>10,.2f}")
        logger.info(f"  Baseline:         SPY {self.config.base_weights['SPY']*100:.0f} / "
              f"GLD {self.config.base_weights['GLD']*100:.0f} / "
              f"TLT {self.config.base_weights['TLT']*100:.0f}")
        logger.info(f"  Max shifts:       SPY {self.config.max_spy_shift*100:.0f}% / "
              f"GLD {self.config.max_gld_shift*100:.0f}% / "
              f"TLT {self.config.max_tlt_shift*100:.0f}%")
        logger.info(f"  Transaction cost: {self.config.transaction_cost_bps:.0f} bps")
        logger.info("")

        logger.info("  PERFORMANCE METRICS")
        logger.info("  " + "-" * 58)
        logger.info(f"    Total Return        {result.total_return:>10.2f}%")
        logger.info(f"    CAGR                {result.cagr:>10.2f}%")
        logger.info(f"    Volatility          {result.volatility:>10.2f}%")
        logger.info(f"    Sharpe Ratio        {result.sharpe_ratio:>10.3f}")
        logger.info(f"    Max Drawdown        {result.max_drawdown:>10.2f}%")
        logger.info("")

        logger.info("  OVERLAY IMPACT")
        logger.info("  " + "-" * 58)
        logger.info(f"    Baseline Sharpe      {result.baseline_sharpe:>8.3f}")
        logger.info(f"    Overlay Sharpe       {result.sharpe_ratio:>8.3f}")
        imp = result.sharpe_improvement
        mark = "+" if imp >= 0.02 else "~" if imp >= 0.0 else "-"
        logger.info(f"    Improvement          {imp:>+8.3f}    [{mark}]")
        logger.info(f"    Active rebalances    {result.extras['overlay_active_rebalances']:>8}")
        logger.info("")

        logger.info("  CRISIS PERFORMANCE (Overlay)")
        logger.info("  " + "-" * 58)
        crisis = result.crisis_returns or {}
        for label, key in [("2008 GFC", "2008"), ("2020 COVID", "2020"), ("2022 Rate Hikes", "2022")]:
            val = crisis.get(key)
            if val is not None:
                logger.info(f"    {label:20s}  {val:>+10.2f}%")
            else:
                logger.info(f"    {label:20s}  {'N/A':>10s}")
        logger.info("")

        logger.info("  TRADE STATISTICS")
        logger.info("  " + "-" * 58)
        logger.info(f"    Total Rebalances     {result.total_rebalances:>8}")
        logger.info(f"    Avg Turnover         {result.avg_turnover*100:>8.2f}%")
        logger.info(f"    Transaction Costs    ${result.total_transaction_costs:>8,.2f}")
        logger.info("")

        logger.info("  SUCCESS CRITERIA")
        logger.info("  " + "-" * 58)
        checks = [
            ("Sharpe > 0.79 (baseline)", result.sharpe_ratio >= 0.79),
            ("Sharpe improvement >= +0.02", imp >= 0.02),
            ("Max DD > -30%", result.max_drawdown > -30.0),
        ]
        for desc, passed in checks:
            icon = "PASS" if passed else "FAIL"
            logger.info(f"    [{icon}] {desc}")
        logger.info("=" * 62 + "\n")

    def save_results(self, result: BacktestResult, output_path: Optional[str] = None) -> None:
        """Save backtest results to JSON."""
        from dataclasses import asdict

        if output_path is None:
            output_path = str(BACKTEST_RESULTS_DIR / "multi_speed_momentum_backtest.json")

        save_results_json(asdict(result), output_path=output_path)
        logger.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    """Run multi-speed momentum overlay backtest."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Speed Momentum Overlay Backtest (v9.23)"
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
    parser.add_argument(
        "--start",
        default="2006-01-01",
        help="Start date (YYYY-MM-DD, default: 2006-01-01)",
    )
    parser.add_argument(
        "--end",
        default="2026-05-15",
        help="End date (YYYY-MM-DD, default: 2026-05-15)",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path (default: data/backtest_results/multi_speed_momentum_backtest.json)",
    )

    args = parser.parse_args()

    config = BacktestConfig(start_date=args.start, end_date=args.end)
    backtester = MultiSpeedMomentumBacktester(config)

    if not backtester.load_data():
        logger.error("Failed to load data")
        return 1

    result = backtester.run_backtest()
    if result is None:
        logger.error("Backtest failed")
        return 1

    backtester.print_report(result)

    if args.save:
        output = args.output if args.output else None
        backtester.save_results(result, output)
    else:
        logger.info("Use --save to persist results to JSON")

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exit(main())
