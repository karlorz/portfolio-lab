"""
Behavioral Sentiment Walk-Forward Backtest — v2.70 Phase 4
Validates the behavioral sentiment overlay against historical data.

Compares baseline 46/38/16 (SPY/GLD/TLT) vs behavioral sentiment overlay
across the full available history. Uses VIX-proxy signals from the
historical_backfill method for pre-2024 periods where no real SKEW/PCR
data exists.

Metrics:
- CAGR, Vol, Sharpe, Max DD for both baseline and overlay
- Crisis year decomposition (2022)
- Regime-specific performance (VIX <15, 15-20, 20-25, 25-30, >30)
- False positive rate: fraction of signals that moved against the
  subsequent 20-day return
- Signal activity: % days active, avg equity shift

Usage:
    python -m src.backtest.behavioral_sentiment_backtest run
    python -m src.backtest.behavioral_sentiment_backtest run --output results.json
    python -m src.backtest.behavioral_sentiment_backtest run --start 2021-05-10
"""

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default baseline allocation
BASELINE_SPY = 0.46
BASELINE_GLD = 0.38
BASELINE_TLT = 0.16

# Max equity shift from behavioral signal
MAX_SHIFT = 0.05  # ±5%

# Benchmark comparison: TSMOM expected Sharpe
TSMOM_EXPECTED_SHARPE = 0.96

from src.paths import DATA_DIR

# Paths
DEFAULT_CACHE_DB = DATA_DIR / "market.db"


@dataclass
class BehavioralBacktestResult:
    """Complete backtest result for behavioral sentiment overlay."""

    timestamp: str

    # Parameters
    start_date: str
    end_date: str
    trading_days: int

    # Baseline 46/38/16
    baseline_cagr: float
    baseline_vol: float
    baseline_sharpe: float
    baseline_max_dd: float
    baseline_crisis_2022: float

    # Behavioral overlay
    overlay_cagr: float
    overlay_vol: float
    overlay_sharpe: float
    overlay_max_dd: float
    overlay_crisis_2022: float

    # Improvements
    sharpe_delta: float
    dd_improvement: float
    cagr_delta: float

    # Signal quality
    signal_days_pct: float
    buy_signal_days: int
    sell_signal_days: int
    neutral_days: int
    avg_equity_shift: float
    false_positive_rate: float
    mean_signal_return_20d: float

    # Regime performance (Sharpe by VIX bucket)
    regime_vix_low_sharpe: float     # VIX < 15
    regime_vix_normal_sharpe: float  # VIX 15-20
    regime_vix_elevated_sharpe: float  # VIX 20-25
    regime_vix_high_sharpe: float    # VIX 25-30
    regime_vix_crisis_sharpe: float  # VIX > 30

    # Target validation
    meets_sharpe_target: bool  # delta >= +0.03

    def to_dict(self) -> dict:
        return asdict(self)


class BehavioralSentimentBacktest:
    """Walk-forward backtest for behavioral sentiment overlay."""

    def __init__(self, cache_db: Path = None):
        if cache_db is None:
            cache_db = DEFAULT_CACHE_DB
        self.cache_db = cache_db

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_prices(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> Dict[str, Dict[str, float]]:
        """Load daily close prices for given symbols, indexed by date."""
        prices: Dict[str, Dict[str, float]] = {s: {} for s in symbols}
        try:
            with sqlite3.connect(self.cache_db) as conn:
                placeholders = ",".join("?" for _ in symbols)
                cursor = conn.execute(
                    f"""SELECT symbol, date, close FROM prices
                        WHERE symbol IN ({placeholders})
                        AND date >= ? AND date <= ?
                        ORDER BY date""",
                    (*symbols, start_date, end_date),
                )
                for symbol, date_str, close in cursor.fetchall():
                    if close is not None and close > 0:
                        prices[symbol][date_str] = float(close)
        except Exception as e:
            logger.error(f"Failed to load prices: {e}")
        return prices

    def _load_signals(
        self, start_date: str, end_date: str
    ) -> List[Dict]:
        """Load historical behavioral sentiment signals via backfill."""
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        signal_gen = BehavioralSentimentSignal(cache_db=self.cache_db)
        return signal_gen.historical_backfill(start_date, end_date)

    # ------------------------------------------------------------------
    # Core backtest
    # ------------------------------------------------------------------

    def run(
        self,
        start_date: str = "2021-05-10",
        end_date: Optional[str] = None,
    ) -> BehavioralBacktestResult:
        """Run the full walk-forward backtest."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Behavioral sentiment backtest: {start_date} → {end_date}")

        # Load prices
        prices = self._load_prices(["SPY", "GLD", "TLT"], start_date, end_date)

        # Build aligned date list
        spy_dates = set(prices["SPY"].keys())
        gld_dates = set(prices["GLD"].keys())
        tlt_dates = set(prices["TLT"].keys())
        common_dates = sorted(spy_dates & gld_dates & tlt_dates)

        if len(common_dates) < 60:
            logger.error(f"Insufficient data: {len(common_dates)} common trading days")
            return self._empty_result(start_date, end_date)

        # Load signals (signals use their own VIX data, slightly different dates)
        signals = self._load_signals(start_date, end_date)
        signal_map: Dict[str, Dict] = {s["date"]: s for s in signals}

        # Run backtest
        (
            baseline_returns,
            overlay_returns,
            signal_stats,
        ) = self._simulate(common_dates, prices, signal_map)

        # Compute metrics
        metrics = self._compute_metrics(
            common_dates, baseline_returns, overlay_returns, signal_stats
        )
        metrics.start_date = common_dates[0]
        metrics.end_date = common_dates[-1]
        metrics.trading_days = len(common_dates)
        metrics.timestamp = datetime.now().isoformat()

        return metrics

    def _simulate(
        self,
        dates: List[str],
        prices: Dict[str, Dict[str, float]],
        signal_map: Dict[str, Dict],
    ) -> Tuple[List[float], List[float], Dict]:
        """Simulate baseline and overlay portfolios day by day."""
        baseline_returns = []
        overlay_returns = []

        # Signal tracking
        buy_days = 0
        sell_days = 0
        neutral_days = 0
        total_shift = 0.0
        shift_count = 0
        signal_returns_20d: List[float] = []
        false_positives = 0
        total_non_neutral = 0

        # State
        prev_baseline_val = None
        prev_overlay_val = None

        for i, date_str in enumerate(dates):
            spy = prices["SPY"][date_str]
            gld = prices["GLD"][date_str]
            tlt = prices["TLT"][date_str]

            # Baseline portfolio value (normalized)
            baseline_val = (
                BASELINE_SPY * spy
                + BASELINE_GLD * gld
                + BASELINE_TLT * tlt
            )

            # Get signal for this date (nearest available)
            sig = signal_map.get(date_str)
            equity_shift = 0.0
            signal_type = "neutral"
            if sig and not sig.get("regime_suppressed", False):
                equity_shift = sig.get("equity_shift_pct", 0.0) / 100.0
                signal_type = sig.get("signal_type", "neutral")

            # Track signal stats
            if "buy" in signal_type:
                buy_days += 1
                total_non_neutral += 1
                total_shift += abs(equity_shift)
                shift_count += 1
            elif "sell" in signal_type:
                sell_days += 1
                total_non_neutral += 1
                total_shift += abs(equity_shift)
                shift_count += 1
            else:
                neutral_days += 1

            # Apply equity shift: move from GLD to SPY (buy) or SPY to GLD (sell)
            # TLT is held constant to avoid conflating bond duration effects
            adj_spy = BASELINE_SPY + equity_shift
            adj_gld = BASELINE_GLD - equity_shift
            adj_tlt = BASELINE_TLT

            # Clamp to [0, 1]
            adj_spy = max(0.0, min(1.0, adj_spy))
            adj_gld = max(0.0, min(1.0, adj_gld))

            overlay_val = adj_spy * spy + adj_gld * gld + adj_tlt * tlt

            # Daily returns
            if prev_baseline_val is not None and prev_baseline_val > 0:
                baseline_ret = (baseline_val / prev_baseline_val) - 1.0
                overlay_ret = (overlay_val / prev_overlay_val) - 1.0
                baseline_returns.append(baseline_ret)
                overlay_returns.append(overlay_ret)

            prev_baseline_val = baseline_val
            prev_overlay_val = overlay_val

            # False positive check: look ahead 20 days
            if total_non_neutral > 0 and signal_type != "neutral":
                lookahead = min(i + 20, len(dates) - 1)
                future_baseline_val = (
                    BASELINE_SPY * prices["SPY"][dates[lookahead]]
                    + BASELINE_GLD * prices["GLD"][dates[lookahead]]
                    + BASELINE_TLT * prices["TLT"][dates[lookahead]]
                )
                future_ret = (future_baseline_val / baseline_val) - 1.0 if baseline_val > 0 else 0.0
                signal_returns_20d.append(future_ret)

                # A buy signal is "correct" if the future return is positive
                # A sell signal is "correct" if the future return is negative
                is_correct = (
                    ("buy" in signal_type and future_ret > 0)
                    or ("sell" in signal_type and future_ret < 0)
                    or (signal_type == "neutral")
                )
                if not is_correct:
                    false_positives += 1

        stats = {
            "buy_days": buy_days,
            "sell_days": sell_days,
            "neutral_days": neutral_days,
            "total_days": len(dates),
            "avg_shift": (total_shift / shift_count) if shift_count > 0 else 0.0,
            "false_positives": false_positives,
            "total_non_neutral": total_non_neutral,
            "signal_returns_20d": signal_returns_20d,
        }

        return baseline_returns, overlay_returns, stats

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        dates: List[str],
        baseline_rets: List[float],
        overlay_rets: List[float],
        stats: Dict,
    ) -> BehavioralBacktestResult:
        """Compute all performance metrics from return series."""
        result = BehavioralBacktestResult(
            timestamp="",
            start_date="",
            end_date="",
            trading_days=0,
            baseline_cagr=0.0,
            baseline_vol=0.0,
            baseline_sharpe=0.0,
            baseline_max_dd=0.0,
            baseline_crisis_2022=0.0,
            overlay_cagr=0.0,
            overlay_vol=0.0,
            overlay_sharpe=0.0,
            overlay_max_dd=0.0,
            overlay_crisis_2022=0.0,
            sharpe_delta=0.0,
            dd_improvement=0.0,
            cagr_delta=0.0,
            signal_days_pct=0.0,
            buy_signal_days=stats["buy_days"],
            sell_signal_days=stats["sell_days"],
            neutral_days=stats["neutral_days"],
            avg_equity_shift=0.0,
            false_positive_rate=0.0,
            mean_signal_return_20d=0.0,
            regime_vix_low_sharpe=0.0,
            regime_vix_normal_sharpe=0.0,
            regime_vix_elevated_sharpe=0.0,
            regime_vix_high_sharpe=0.0,
            regime_vix_crisis_sharpe=0.0,
            meets_sharpe_target=False,
        )

        if len(baseline_rets) < 20:
            return result

        arr_bl = np.array(baseline_rets, dtype=np.float64)
        arr_ol = np.array(overlay_rets, dtype=np.float64)

        # CAGR
        bl_cum = np.prod(1.0 + arr_bl)
        ol_cum = np.prod(1.0 + arr_ol)
        years = len(arr_bl) / 252.0
        result.baseline_cagr = round((bl_cum ** (1.0 / years) - 1.0) * 100, 2) if years > 0 else 0.0
        result.overlay_cagr = round((ol_cum ** (1.0 / years) - 1.0) * 100, 2) if years > 0 else 0.0
        result.cagr_delta = round(result.overlay_cagr - result.baseline_cagr, 2)

        # Volatility (annualized)
        result.baseline_vol = round(float(np.std(arr_bl, ddof=1)) * math.sqrt(252) * 100, 2)
        result.overlay_vol = round(float(np.std(arr_ol, ddof=1)) * math.sqrt(252) * 100, 2)

        # Sharpe (assuming 0% risk-free for simplicity)
        bl_mean_daily = float(np.mean(arr_bl))
        ol_mean_daily = float(np.mean(arr_ol))
        bl_std_daily = max(float(np.std(arr_bl, ddof=1)), 1e-8)
        ol_std_daily = max(float(np.std(arr_ol, ddof=1)), 1e-8)
        result.baseline_sharpe = round((bl_mean_daily / bl_std_daily) * math.sqrt(252), 3)
        result.overlay_sharpe = round((ol_mean_daily / ol_std_daily) * math.sqrt(252), 3)
        result.sharpe_delta = round(result.overlay_sharpe - result.baseline_sharpe, 3)

        # Max drawdown
        result.baseline_max_dd = round(self._max_drawdown(arr_bl) * 100, 2)
        result.overlay_max_dd = round(self._max_drawdown(arr_ol) * 100, 2)
        result.dd_improvement = round(
            abs(result.baseline_max_dd) - abs(result.overlay_max_dd), 2
        )

        # Crisis year 2022
        result.baseline_crisis_2022 = round(
            self._year_return(dates, arr_bl, "2022") * 100, 2
        )
        result.overlay_crisis_2022 = round(
            self._year_return(dates, arr_ol, "2022") * 100, 2
        )

        # Signal stats
        td = stats["total_days"]
        non_neutral = stats["buy_days"] + stats["sell_days"]
        result.signal_days_pct = round(non_neutral / td * 100, 1) if td > 0 else 0.0
        result.avg_equity_shift = round(stats["avg_shift"] * 100, 2)
        total_nn = int(stats["total_non_neutral"])
        false_pos = int(stats["false_positives"])
        result.false_positive_rate = (
            round(false_pos / total_nn * 100, 1) if total_nn > 0 else 0.0
        )
        sig_rets = stats["signal_returns_20d"]
        result.mean_signal_return_20d = (
            round(float(np.mean(sig_rets)) * 100, 2) if sig_rets else 0.0
        )

        # Regime-specific Sharpe (by VIX level) — single VIX load, one pass
        regime_sharpes = self._all_regime_sharpes(dates, arr_ol)
        result.regime_vix_low_sharpe = round(regime_sharpes[0], 3)
        result.regime_vix_normal_sharpe = round(regime_sharpes[1], 3)
        result.regime_vix_elevated_sharpe = round(regime_sharpes[2], 3)
        result.regime_vix_high_sharpe = round(regime_sharpes[3], 3)
        result.regime_vix_crisis_sharpe = round(regime_sharpes[4], 3)

        # Target validation
        result.meets_sharpe_target = result.sharpe_delta >= 0.03

        return result

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        """Compute maximum drawdown from daily returns."""
        cumulative = np.cumprod(1.0 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return float(np.min(drawdown))

    @staticmethod
    def _year_return(
        dates: List[str], returns: np.ndarray, year: str
    ) -> float:
        """Compute total return for a specific calendar year."""
        indices = [i for i, d in enumerate(dates) if d.startswith(year)]
        if not indices or indices[0] >= len(returns):
            return 0.0
        start_idx = max(0, indices[0] - 1)  # include Dec 31 carry
        end_idx = indices[-1]
        if end_idx < start_idx:
            return 0.0
        segment = returns[start_idx:end_idx]
        if len(segment) == 0:
            return 0.0
        return float(np.prod(1.0 + segment) - 1.0)

    def _all_regime_sharpes(
        self, dates: List[str], returns: np.ndarray
    ) -> tuple:
        """Compute Sharpe for all 5 VIX regimes in a single pass.

        Loads VIX prices once, buckets returns into 5 lists, returns
        (low, normal, elevated, high, crisis) Sharpe tuple.
        """
        buckets = {k: [] for k in ["low", "normal", "elevated", "high", "crisis"]}

        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    """SELECT date, close FROM prices
                       WHERE symbol = '^VIX'
                       AND date >= ? AND date <= ?
                       ORDER BY date""",
                    (dates[0], dates[-1]),
                )
                vix_prices = {row[0]: float(row[1]) for row in cursor.fetchall()}
        except sqlite3.Error:
            logger.exception("Failed to fetch VIX prices from cache_db")
            return (0.0, 0.0, 0.0, 0.0, 0.0)

        for i, d in enumerate(dates):
            if i >= len(returns):
                break
            vix = vix_prices.get(d)
            if vix is None:
                continue
            if vix < 15:
                buckets["low"].append(returns[i])
            elif vix < 20:
                buckets["normal"].append(returns[i])
            elif vix < 25:
                buckets["elevated"].append(returns[i])
            elif vix < 30:
                buckets["high"].append(returns[i])
            else:
                buckets["crisis"].append(returns[i])

        result = []
        for key in ["low", "normal", "elevated", "high", "crisis"]:
            arr = buckets[key]
            if len(arr) < 5:
                result.append(0.0)
            else:
                a = np.array(arr, dtype=np.float64)
                mean_d = float(np.mean(a))
                std_d = max(float(np.std(a, ddof=1)), 1e-8)
                result.append((mean_d / std_d) * math.sqrt(252))

        return tuple(result)

    def _empty_result(
        self, start_date: str, end_date: str
    ) -> BehavioralBacktestResult:
        """Return an empty result when insufficient data."""
        return BehavioralBacktestResult(
            timestamp=datetime.now().isoformat(),
            start_date=start_date,
            end_date=end_date,
            trading_days=0,
            baseline_cagr=0.0,
            baseline_vol=0.0,
            baseline_sharpe=0.0,
            baseline_max_dd=0.0,
            baseline_crisis_2022=0.0,
            overlay_cagr=0.0,
            overlay_vol=0.0,
            overlay_sharpe=0.0,
            overlay_max_dd=0.0,
            overlay_crisis_2022=0.0,
            sharpe_delta=0.0,
            dd_improvement=0.0,
            cagr_delta=0.0,
            signal_days_pct=0.0,
            buy_signal_days=0,
            sell_signal_days=0,
            neutral_days=0,
            avg_equity_shift=0.0,
            false_positive_rate=0.0,
            mean_signal_return_20d=0.0,
            regime_vix_low_sharpe=0.0,
            regime_vix_normal_sharpe=0.0,
            regime_vix_elevated_sharpe=0.0,
            regime_vix_high_sharpe=0.0,
            regime_vix_crisis_sharpe=0.0,
            meets_sharpe_target=False,
        )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Behavioral Sentiment Walk-Forward Backtest"
    )
    parser.add_argument("mode", nargs="?", default="run", help="run | summary")
    parser.add_argument("--start", type=str, default="2021-05-10", help="Start date")
    parser.add_argument("--end", type=str, default=None, help="End date")
    parser.add_argument("--output", type=str, default=None, help="Save JSON output")
    parser.add_argument("--summary", action="store_true", help="Print summary only")

    args = parser.parse_args()

    backtest = BehavioralSentimentBacktest()
    result = backtest.run(start_date=args.start, end_date=args.end)

    if args.mode == "summary" or args.summary:
        print(f"\n=== Behavioral Sentiment Backtest Summary ===")
        print(f"Period: {result.start_date} → {result.end_date} ({result.trading_days} days)")
        print(f"\nBaseline 46/38/16:")
        print(f"  CAGR: {result.baseline_cagr}%  Vol: {result.baseline_vol}%  "
              f"Sharpe: {result.baseline_sharpe}  MaxDD: {result.baseline_max_dd}%")
        print(f"  2022: {result.baseline_crisis_2022}%")
        print(f"\nBehavioral Overlay:")
        print(f"  CAGR: {result.overlay_cagr}%  Vol: {result.overlay_vol}%  "
              f"Sharpe: {result.overlay_sharpe}  MaxDD: {result.overlay_max_dd}%")
        print(f"  2022: {result.overlay_crisis_2022}%")
        print(f"\nDelta:")
        print(f"  Sharpe: {result.sharpe_delta:+.3f}  "
              f"MaxDD: {result.dd_improvement:+.1f}pp  "
              f"CAGR: {result.cagr_delta:+.1f}pp")
        print(f"\nSignal Quality:")
        print(f"  Active: {result.signal_days_pct}% of days  "
              f"Buy: {result.buy_signal_days}  Sell: {result.sell_signal_days}  "
              f"Neutral: {result.neutral_days}")
        print(f"  Avg equity shift: {result.avg_equity_shift}%  "
              f"False positive rate: {result.false_positive_rate}%")
        print(f"  Mean 20d signal return: {result.mean_signal_return_20d}%")
        print(f"\nRegime Sharpe (overlay):")
        print(f"  VIX<15: {result.regime_vix_low_sharpe}  "
              f"VIX 15-20: {result.regime_vix_normal_sharpe}  "
              f"VIX 20-25: {result.regime_vix_elevated_sharpe}")
        print(f"  VIX 25-30: {result.regime_vix_high_sharpe}  "
              f"VIX>30: {result.regime_vix_crisis_sharpe}")
        print(f"\nTarget: Sharpe delta >= +0.03 → "
              f"{'MET' if result.meets_sharpe_target else 'NOT MET'}")

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nResults saved to {args.output}")
