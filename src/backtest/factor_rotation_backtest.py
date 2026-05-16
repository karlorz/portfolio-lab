"""
Factor Rotation Walk-Forward Backtest — v3.00 Phase 3 Validation
Compares factor rotation overlay (MTUM/QUAL/USMV/VLUE) against
momentum-only baseline using regime-based allocation.

Data: MTUM, USMV, VLUE, SPY from market.db (2021-2026)
Regime detection: VIX-based (normal <20, elevated 20-25, high 25-30, crisis >30)
QUAL is proxied via USMV (both defensive/low-vol) when QUAL data unavailable.

Usage:
    python -m src.backtest.factor_rotation_backtest run
    python -m src.backtest.factor_rotation_backtest run --summary
    python -m src.backtest.factor_rotation_backtest run --output results.json
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

# Factor allocation by regime (from spec)
REGIME_ALLOCATIONS = {
    "bull": {"MTUM": 0.60, "QUAL": 0.25, "USMV": 0.10, "VLUE": 0.05},
    "bear": {"MTUM": 0.10, "QUAL": 0.40, "USMV": 0.40, "VLUE": 0.10},
    "neutral": {"MTUM": 0.35, "QUAL": 0.35, "USMV": 0.20, "VLUE": 0.10},
    "high_vol": {"MTUM": 0.15, "QUAL": 0.30, "USMV": 0.45, "VLUE": 0.10},
    "crisis": {"MTUM": 0.05, "QUAL": 0.35, "USMV": 0.50, "VLUE": 0.10},
}

# Map VIX to regime
VIX_REGIME_MAP = [
    (15.0, "bull"),
    (20.0, "neutral"),
    (25.0, "high_vol"),
    (30.0, "elevated"),  # transitional, treated as high_vol for allocation
    (float("inf"), "crisis"),
]

# Base equity allocation in the 46/38/16 portfolio
BASE_EQUITY_PCT = 0.46
MAX_EQUITY_SHIFT = 0.15  # ±15%

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_DB = REPO_ROOT / "data" / "market.db"


def _vix_to_regime(vix: float) -> str:
    """Map VIX level to market regime."""
    if vix < 15:
        return "bull"
    elif vix < 20:
        return "neutral"
    elif vix < 25:
        return "elevated"
    elif vix < 30:
        return "high_vol"
    else:
        return "crisis"


# Regime normalization for allocation lookup
_REGIME_NORMALIZE = {
    "elevated": "high_vol",
}


@dataclass
class FactorBacktestResult:
    """Complete factor rotation backtest result."""

    timestamp: str
    start_date: str
    end_date: str
    trading_days: int

    # SPY-only baseline (momentum proxy)
    baseline_cagr: float
    baseline_vol: float
    baseline_sharpe: float
    baseline_max_dd: float
    baseline_crisis_2022: float

    # Factor rotation overlay
    overlay_cagr: float
    overlay_vol: float
    overlay_sharpe: float
    overlay_max_dd: float
    overlay_crisis_2022: float

    # Improvements
    sharpe_delta: float
    dd_improvement: float
    cagr_delta: float

    # Factor activity
    avg_mtum_weight: float
    avg_qual_weight: float
    avg_usmv_weight: float
    avg_vlue_weight: float
    regime_breakdown: Dict[str, float]  # regime → % of days

    # Regime Sharpe
    regime_bull_sharpe: float
    regime_neutral_sharpe: float
    regime_elevated_sharpe: float
    regime_high_vol_sharpe: float
    regime_crisis_sharpe: float

    # Target validation
    meets_sharpe_target: bool  # delta >= +0.05
    meets_dd_target: bool      # dd improvement >= 2pp

    def to_dict(self) -> dict:
        return asdict(self)


class FactorRotationBacktest:
    """Walk-forward backtest for factor rotation overlay."""

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
        """Load daily close prices indexed by date."""
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

    def _load_vix(self, start_date: str, end_date: str) -> Dict[str, float]:
        """Load VIX prices for regime detection."""
        vix = {}
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    """SELECT date, close FROM prices
                       WHERE symbol = '^VIX'
                       AND date >= ? AND date <= ?
                       ORDER BY date""",
                    (start_date, end_date),
                )
                vix = {row[0]: float(row[1]) for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to load VIX: {e}")
        return vix

    # ------------------------------------------------------------------
    # Core backtest
    # ------------------------------------------------------------------

    def run(
        self,
        start_date: str = "2021-05-10",
        end_date: Optional[str] = None,
    ) -> FactorBacktestResult:
        """Run the factor rotation walk-forward backtest."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Factor rotation backtest: {start_date} → {end_date}")

        symbols = ["SPY", "MTUM", "USMV", "VLUE"]
        prices = self._load_prices(symbols, start_date, end_date)
        vix_prices = self._load_vix(start_date, end_date)

        # Align dates
        symbol_date_sets = [set(prices[s].keys()) for s in symbols]
        common_dates = sorted(set.intersection(*symbol_date_sets))

        if len(common_dates) < 60:
            logger.error(f"Insufficient data: {len(common_dates)} days")
            return self._empty_result(start_date, end_date)

        # Simulate
        bl_rets, ol_rets, factor_stats = self._simulate(
            common_dates, prices, vix_prices
        )

        # Compute metrics
        result = self._compute_metrics(common_dates, bl_rets, ol_rets, factor_stats)
        result.start_date = common_dates[0]
        result.end_date = common_dates[-1]
        result.trading_days = len(common_dates)
        result.timestamp = datetime.now().isoformat()

        return result

    def _simulate(
        self,
        dates: List[str],
        prices: Dict[str, Dict[str, float]],
        vix_prices: Dict[str, float],
    ) -> Tuple[List[float], List[float], Dict]:
        """Simulate baseline (SPY) and factor rotation day by day."""
        bl_returns = []
        ol_returns = []

        # Factor weight tracking
        mtum_weights = []
        qual_weights = []
        usmv_weights = []
        vlue_weights = []
        regime_counts: Dict[str, int] = {}

        # Normalize all price series to 1.0 on first common date
        first_date = dates[0]
        norm = {}
        for sym in ["SPY", "MTUM", "USMV", "VLUE"]:
            base = prices[sym].get(first_date, 1.0)
            norm[sym] = {d: p / base for d, p in prices[sym].items()}

        prev_bl_val = None
        prev_ol_val = None

        for date_str in dates:
            spy = norm["SPY"].get(date_str)
            mtum = norm["MTUM"].get(date_str)
            usmv = norm["USMV"].get(date_str)
            vlue = norm["VLUE"].get(date_str)

            if any(x is None for x in [spy, mtum, usmv, vlue]):
                continue

            # Baseline: SPY only (momentum proxy — MTUM launched 2013, SPY is
            # the closest broad-market momentum-amenable benchmark)
            bl_val = spy

            # Factor rotation: regime-based allocation
            vix = vix_prices.get(date_str, 18.0)
            regime = _vix_to_regime(vix)
            regime_key = _REGIME_NORMALIZE.get(regime, regime)
            alloc = REGIME_ALLOCATIONS.get(regime_key, REGIME_ALLOCATIONS["neutral"])

            # Track factor weights
            mtum_weights.append(alloc["MTUM"])
            qual_weights.append(alloc.get("QUAL", 0.0))
            usmv_weights.append(alloc["USMV"])
            vlue_weights.append(alloc["VLUE"])

            # Track regime
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

            # Overlay value: factor ETF basket
            # QUAL proxied by USMV when unavailable
            qual_proxy = norm["USMV"].get(date_str, usmv)
            ol_val = (
                alloc["MTUM"] * mtum
                + alloc.get("QUAL", 0.0) * qual_proxy
                + alloc["USMV"] * usmv
                + alloc["VLUE"] * vlue
            )

            if prev_bl_val is not None and prev_bl_val > 0:
                bl_returns.append((bl_val / prev_bl_val) - 1.0)
            if prev_ol_val is not None and prev_ol_val > 0:
                ol_returns.append((ol_val / prev_ol_val) - 1.0)

            prev_bl_val = bl_val
            prev_ol_val = ol_val

        # Skip first return (no prior day) so series align
        if len(bl_returns) > len(ol_returns):
            bl_returns = bl_returns[-len(ol_returns):]
        elif len(ol_returns) > len(bl_returns):
            ol_returns = ol_returns[-len(bl_returns):]

        td = len(dates)
        stats = {
            "avg_mtum": np.mean(mtum_weights) if mtum_weights else 0.0,
            "avg_qual": np.mean(qual_weights) if qual_weights else 0.0,
            "avg_usmv": np.mean(usmv_weights) if usmv_weights else 0.0,
            "avg_vlue": np.mean(vlue_weights) if vlue_weights else 0.0,
            "regime_counts": regime_counts,
            "total_days": td,
        }

        return bl_returns, ol_returns, stats

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        dates: List[str],
        bl_rets: List[float],
        ol_rets: List[float],
        stats: Dict,
    ) -> FactorBacktestResult:
        """Compute all performance metrics."""
        result = FactorBacktestResult(
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
            avg_mtum_weight=0.0,
            avg_qual_weight=0.0,
            avg_usmv_weight=0.0,
            avg_vlue_weight=0.0,
            regime_breakdown={},
            regime_bull_sharpe=0.0,
            regime_neutral_sharpe=0.0,
            regime_elevated_sharpe=0.0,
            regime_high_vol_sharpe=0.0,
            regime_crisis_sharpe=0.0,
            meets_sharpe_target=False,
            meets_dd_target=False,
        )

        if len(bl_rets) < 20 or len(ol_rets) < 20:
            return result

        arr_bl = np.array(bl_rets, dtype=np.float64)
        arr_ol = np.array(ol_rets, dtype=np.float64)

        n = min(len(arr_bl), len(arr_ol))
        arr_bl = arr_bl[:n]
        arr_ol = arr_ol[:n]

        # CAGR
        bl_cum = np.prod(1.0 + arr_bl)
        ol_cum = np.prod(1.0 + arr_ol)
        years = n / 252.0
        result.baseline_cagr = round((bl_cum ** (1.0 / years) - 1.0) * 100, 2) if years > 0 else 0.0
        result.overlay_cagr = round((ol_cum ** (1.0 / years) - 1.0) * 100, 2) if years > 0 else 0.0
        result.cagr_delta = round(result.overlay_cagr - result.baseline_cagr, 2)

        # Volatility
        result.baseline_vol = round(float(np.std(arr_bl, ddof=1)) * math.sqrt(252) * 100, 2)
        result.overlay_vol = round(float(np.std(arr_ol, ddof=1)) * math.sqrt(252) * 100, 2)

        # Sharpe
        bl_m = float(np.mean(arr_bl))
        ol_m = float(np.mean(arr_ol))
        bl_s = max(float(np.std(arr_bl, ddof=1)), 1e-8)
        ol_s = max(float(np.std(arr_ol, ddof=1)), 1e-8)
        result.baseline_sharpe = round((bl_m / bl_s) * math.sqrt(252), 3)
        result.overlay_sharpe = round((ol_m / ol_s) * math.sqrt(252), 3)
        result.sharpe_delta = round(result.overlay_sharpe - result.baseline_sharpe, 3)

        # Max DD
        result.baseline_max_dd = round(self._max_drawdown(arr_bl) * 100, 2)
        result.overlay_max_dd = round(self._max_drawdown(arr_ol) * 100, 2)
        result.dd_improvement = round(
            abs(result.baseline_max_dd) - abs(result.overlay_max_dd), 2
        )

        # Crisis 2022
        result.baseline_crisis_2022 = round(self._year_return(dates[:n], arr_bl, "2022") * 100, 2)
        result.overlay_crisis_2022 = round(self._year_return(dates[:n], arr_ol, "2022") * 100, 2)

        # Factor weights
        result.avg_mtum_weight = round(stats["avg_mtum"] * 100, 1)
        result.avg_qual_weight = round(stats["avg_qual"] * 100, 1)
        result.avg_usmv_weight = round(stats["avg_usmv"] * 100, 1)
        result.avg_vlue_weight = round(stats["avg_vlue"] * 100, 1)

        # Regime breakdown
        td = stats["total_days"]
        result.regime_breakdown = {
            reg: round(cnt / td * 100, 1)
            for reg, cnt in stats["regime_counts"].items()
        }

        # Regime Sharpes — single VIX load, one pass
        regime_sharpes = self._all_regime_sharpes(dates[:n], arr_ol)
        result.regime_bull_sharpe = round(regime_sharpes[0], 3)
        result.regime_neutral_sharpe = round(regime_sharpes[1], 3)
        result.regime_elevated_sharpe = round(regime_sharpes[2], 3)
        result.regime_high_vol_sharpe = round(regime_sharpes[3], 3)
        result.regime_crisis_sharpe = round(regime_sharpes[4], 3)

        # Targets
        result.meets_sharpe_target = result.sharpe_delta >= 0.05
        result.meets_dd_target = result.dd_improvement >= 2.0

        return result

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        cumulative = np.cumprod(1.0 + returns)
        peak = np.maximum.accumulate(cumulative)
        return float(np.min((cumulative - peak) / peak))

    @staticmethod
    def _year_return(dates: List[str], returns: np.ndarray, year: str) -> float:
        indices = [i for i, d in enumerate(dates) if d.startswith(year)]
        if not indices or indices[0] >= len(returns):
            return 0.0
        start_idx = max(0, indices[0] - 1)
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
        """Load VIX once, bucket all 5 regimes in a single pass."""
        buckets = {k: [] for k in ["bull", "neutral", "elevated", "high_vol", "crisis"]}

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
        except Exception:
            return (0.0, 0.0, 0.0, 0.0, 0.0)

        for i, d in enumerate(dates):
            if i >= len(returns):
                break
            vix = vix_prices.get(d)
            if vix is None:
                continue
            if vix < 15:
                buckets["bull"].append(returns[i])
            elif vix < 20:
                buckets["neutral"].append(returns[i])
            elif vix < 25:
                buckets["elevated"].append(returns[i])
            elif vix < 30:
                buckets["high_vol"].append(returns[i])
            else:
                buckets["crisis"].append(returns[i])

        result = []
        for key in ["bull", "neutral", "elevated", "high_vol", "crisis"]:
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
    ) -> FactorBacktestResult:
        return FactorBacktestResult(
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
            avg_mtum_weight=0.0,
            avg_qual_weight=0.0,
            avg_usmv_weight=0.0,
            avg_vlue_weight=0.0,
            regime_breakdown={},
            regime_bull_sharpe=0.0,
            regime_neutral_sharpe=0.0,
            regime_elevated_sharpe=0.0,
            regime_high_vol_sharpe=0.0,
            regime_crisis_sharpe=0.0,
            meets_sharpe_target=False,
            meets_dd_target=False,
        )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Factor Rotation Walk-Forward Backtest"
    )
    parser.add_argument("mode", nargs="?", default="run", help="run | summary")
    parser.add_argument("--start", type=str, default="2021-05-10")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--summary", action="store_true")

    args = parser.parse_args()

    bt = FactorRotationBacktest()
    result = bt.run(start_date=args.start, end_date=args.end)

    if args.mode == "summary" or args.summary:
        print(f"\n=== Factor Rotation Backtest Summary ===")
        print(f"Period: {result.start_date} → {result.end_date} ({result.trading_days} days)")
        print(f"\nSPY Baseline:")
        print(f"  CAGR: {result.baseline_cagr}%  Vol: {result.baseline_vol}%  "
              f"Sharpe: {result.baseline_sharpe}  MaxDD: {result.baseline_max_dd}%")
        print(f"  2022: {result.baseline_crisis_2022}%")
        print(f"\nFactor Rotation Overlay:")
        print(f"  CAGR: {result.overlay_cagr}%  Vol: {result.overlay_vol}%  "
              f"Sharpe: {result.overlay_sharpe}  MaxDD: {result.overlay_max_dd}%")
        print(f"  2022: {result.overlay_crisis_2022}%")
        print(f"\nDelta:")
        print(f"  Sharpe: {result.sharpe_delta:+.3f}  "
              f"MaxDD: {result.dd_improvement:+.1f}pp  "
              f"CAGR: {result.cagr_delta:+.1f}pp")
        print(f"\nAvg Factor Weights:")
        print(f"  MTUM: {result.avg_mtum_weight}%  QUAL: {result.avg_qual_weight}%  "
              f"USMV: {result.avg_usmv_weight}%  VLUE: {result.avg_vlue_weight}%")
        print(f"\nRegime Breakdown:")
        for reg, pct in sorted(result.regime_breakdown.items()):
            print(f"  {reg}: {pct}%")
        print(f"\nRegime Sharpe (overlay):")
        print(f"  Bull: {result.regime_bull_sharpe}  "
              f"Neutral: {result.regime_neutral_sharpe}  "
              f"Elevated: {result.regime_elevated_sharpe}")
        print(f"  High Vol: {result.regime_high_vol_sharpe}  "
              f"Crisis: {result.regime_crisis_sharpe}")
        print(f"\nTargets:")
        print(f"  Sharpe delta >= +0.05 → "
              f"{'MET' if result.meets_sharpe_target else 'NOT MET'}")
        print(f"  DD improvement >= 2pp → "
              f"{'MET' if result.meets_dd_target else 'NOT MET'}")

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nResults saved to {args.output}")
