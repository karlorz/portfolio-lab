"""
Combined Overlay Backtest - v4.90 Validation Engine
Runs all 4 tactical overlays on historical data to validate theoretical Sharpe projections.

Overlays tested:
- v4.60 Cashless Collar (drawdown protection)
- v3.50 Calendar Seasonality (execution timing)
- v4.70 Crypto Tactical (uncorrelated alpha)
- v4.80 Bond Duration Rotation (fixed-income)

Compares baseline 46/38/16 vs combined overlay portfolio across full history.

Usage:
    python -m src.backtest.combined_overlay_backtest run
    python -m src.backtest.combined_overlay_backtest run --output results.json
"""

import logging
import math
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from src.paths import BASE_ALLOCATION, DATA_DIR, sqlite_connect
from typing import Optional, Dict, List, Tuple

import numpy as np

from src.backtest.rolling_vol import precomputed_rolling_volatility
from src.backtest.metrics import (
    BacktestResult,
    DEFAULT_TRANSACTION_COST_BPS,
    build_profitability_evidence,
    compute_one_way_turnover,
    save_results_json,
)

__all__ = [
    'CombinedOverlayBacktest', 'run_combined_backtest',
]

logger = logging.getLogger(__name__)


class CombinedOverlayBacktest:
    """
    Historical backtest combining all tactical overlays.

    Uses simplified signal models that don't require ML deps.
    Each overlay adds its contribution to the baseline allocation.
    """

    BASELINE = {k.lower(): v for k, v in BASE_ALLOCATION.items()}

    # Approximate asset proxies
    # SHY ~ short-term treasuries (low vol, low return)
    # IEF ~ intermediate treasuries
    # BTC/ETH ~ synthetic high-vol assets

    REQUIRED_ASSETS = ("SPY", "GLD", "TLT", "IEF", "SHY", "BTC", "ETH", "VIX")

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        *,
        allow_synthetic: bool = False,
        allow_proxy: bool = False,
        transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    ):
        self.data_dir = data_dir or DATA_DIR
        self.allow_synthetic = allow_synthetic
        self.allow_proxy = allow_proxy
        self.transaction_cost_bps = transaction_cost_bps
        self._data_mode = "real"

    def _load_historical_data(self) -> Dict:
        """Load complete historical data or an explicitly allowed diagnostic.

        B1b keep+document (partial-delegation escape, Item B1b sub-task 1):
        keeps its own market.db query — ``pd.read_sql_query`` full-table read
        + per-symbol filter producing the ``{symbol: {"dates": [...],
        "prices": [...]}}`` list shape, which differs from
        ``grid_runner.load_prices_market_db``'s date-indexed
        ``{symbol: {date: close}}`` dict shape; also maps VIX→^VIX and aligns
        symbols to common dates. Not delegated.
        """
        db_path = self.data_dir / "market.db"
        data = {}

        # Try loading from market database
        if db_path.exists():
            try:
                import pandas as pd
                with sqlite_connect(str(db_path)) as conn:
                    df = pd.read_sql_query(
                        "SELECT date, symbol, close FROM prices ORDER BY date",
                        conn
                    )

                if not df.empty:
                    for sym in self.REQUIRED_ASSETS:
                        db_symbol = "^VIX" if sym == "VIX" else sym
                        sym_df = df[df["symbol"] == db_symbol]
                        if not sym_df.empty:
                            data[sym] = {
                                "dates": sym_df["date"].tolist(),
                                "prices": sym_df["close"].tolist(),
                            }
                    if set(data) == set(self.REQUIRED_ASSETS):
                        common_dates = set(data[self.REQUIRED_ASSETS[0]]["dates"])
                        for symbol in self.REQUIRED_ASSETS[1:]:
                            common_dates &= set(data[symbol]["dates"])
                        aligned_dates = sorted(common_dates)
                        if aligned_dates:
                            for symbol in self.REQUIRED_ASSETS:
                                by_date = dict(zip(
                                    data[symbol]["dates"],
                                    data[symbol]["prices"],
                                ))
                                data[symbol] = {
                                    "dates": aligned_dates,
                                    "prices": [by_date[value] for value in aligned_dates],
                                }
                            self._data_mode = "proxy"
                            return data
            except (ImportError, OSError, KeyError, ValueError, TypeError, RuntimeError) as e:
                logger.warning("Database load failed: %s", e)

        if not self.allow_synthetic:
            missing = sorted(set(self.REQUIRED_ASSETS) - set(data))
            raise ValueError(
                "decision-grade run requires complete real market data; "
                f"missing assets: {', '.join(missing) or 'aligned observations'}"
            )

        self._data_mode = "synthetic"
        return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> Dict:
        """Generate realistic synthetic price data for backtest."""
        rng = np.random.RandomState(42)
        n_days = 5040  # ~20 years of trading days

        # Realistic annual parameters
        symbols = {
            "SPY": (0.10, 0.17),   # 10% return, 17% vol
            "GLD": (0.08, 0.18),   # 8% return, 18% vol
            "TLT": (0.04, 0.14),   # 4% return, 14% vol
            "IEF": (0.03, 0.07),   # 3% return, 7% vol
            "SHY": (0.02, 0.02),   # 2% return, 2% vol
            "BTC": (0.50, 0.75),   # 50% return, 75% vol
            "ETH": (0.40, 0.85),   # 40% return, 85% vol
            "VIX": (0.0, 1.0),     # Mean-reverting
        }

        data = {}
        dates = []
        start = date(2006, 1, 3)

        for i in range(n_days):
            d = start + timedelta(days=i)
            # Skip weekends
            if d.weekday() >= 5:
                continue
            dates.append(d.isoformat())

        n = len(dates)

        for sym, (mu, vol) in symbols.items():
            daily_mu = mu / 252
            daily_vol = vol / math.sqrt(252)

            if sym == "VIX":
                # Mean-reverting VIX
                vix = np.ones(n) * 18.0
                for i in range(1, n):
                    vix[i] = vix[i-1] + rng.normal(0, 1.5)
                    vix[i] = max(8, min(80, vix[i] * 0.9 + 18 * 0.1))
                data[sym] = {
                    "dates": dates,
                    "prices": vix.tolist(),
                }
            else:
                price = 100.0 if sym not in ("BTC", "ETH") else (1000.0 if sym == "BTC" else 200.0)
                returns = rng.normal(daily_mu, daily_vol, n)
                # Add crisis periods
                crisis_mask_2008 = np.zeros(n, dtype=bool)
                crisis_mask_2008[500:560] = True  # ~2008
                crisis_mask_2020 = np.zeros(n, dtype=bool)
                crisis_mask_2020[3500:3540] = True  # ~2020
                crisis_mask_2022 = np.zeros(n, dtype=bool)
                crisis_mask_2022[4000:4060] = True  # ~2022

                if sym == "SPY":
                    returns[crisis_mask_2008] = rng.normal(-0.003, 0.03, crisis_mask_2008.sum())
                    returns[crisis_mask_2020] = rng.normal(-0.002, 0.04, crisis_mask_2020.sum())
                    returns[crisis_mask_2022] = rng.normal(-0.001, 0.02, crisis_mask_2022.sum())
                if sym == "TLT":
                    returns[crisis_mask_2008] = rng.normal(0.001, 0.015, crisis_mask_2008.sum())
                    returns[crisis_mask_2020] = rng.normal(0.002, 0.02, crisis_mask_2020.sum())

                prices = [price]
                for r in returns:
                    prices.append(prices[-1] * (1 + r))

                data[sym] = {
                    "dates": dates,
                    "prices": prices[1:],  # align length
                }

        return data

    def _compute_returns(self, prices: List[float]) -> List[float]:
        return [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]

    def _compute_rolling_vol(self, returns: List[float], window: int = 30) -> List[float]:
        return precomputed_rolling_volatility(
            returns,
            window=window,
            fallback_vol=0.16,
            warmup_std_min_index=1,
        )

    def _collar_signal(self, vix: float, spy_ret: float) -> float:
        """Simplified collar: reduce SPY when VIX high."""
        if vix > 40:
            return -0.05  # Crisis — strong reduction
        elif vix > 30:
            return -0.03  # Stress
        elif vix > 20:
            return -0.01  # Elevated
        return 0.0

    def _bond_duration_signal(self, yield_spread: float, rate_change: float) -> Tuple[float, float, float]:
        """Simplified bond duration rotation."""
        if yield_spread > 1.0 and rate_change < -0.3:
            return 0.70, 0.20, 0.10  # TLT heavy
        elif yield_spread < 0 and rate_change > 0.3:
            return 0.05, 0.25, 0.70  # SHY heavy
        else:
            return 0.20, 0.50, 0.30  # Balanced

    def _crypto_signal(self, btc_mom_6m: float, btc_vol: float,
                       eth_mom_6m: float, eth_vol: float) -> float:
        """Simplified crypto: enter on positive momentum, exit on extreme vol."""
        if btc_vol > 1.0 or eth_vol > 1.0:
            return 0.0
        if btc_mom_6m <= 0 and eth_mom_6m <= 0:
            return 0.0
        # Scale by momentum strength
        avg_mom = (max(0, btc_mom_6m) + max(0, eth_mom_6m)) / 2
        weight = min(0.05, 0.03 * (1 + avg_mom))
        return weight

    def run_backtest(self) -> BacktestResult:
        """Run combined overlay backtest."""
        data = self._load_historical_data()
        if self._data_mode == "proxy" and not self.allow_proxy:
            raise ValueError(
                "combined overlay proxy data requires explicit diagnostic opt-in"
            )

        spy_prices = data["SPY"]["prices"]
        gld_prices = data["GLD"]["prices"]
        tlt_prices = data["TLT"]["prices"]
        ief_prices = data["IEF"]["prices"]
        shy_prices = data["SHY"]["prices"]
        btc_prices = data["BTC"]["prices"]
        eth_prices = data["ETH"]["prices"]
        vix_data = data["VIX"]["prices"]
        dates = data["SPY"]["dates"]

        n = len(spy_prices)
        len(spy_prices)

        # Returns
        spy_rets = self._compute_returns(spy_prices)
        gld_rets = self._compute_returns(gld_prices)
        tlt_rets = self._compute_returns(tlt_prices)
        ief_rets = self._compute_returns(ief_prices)
        shy_rets = self._compute_returns(shy_prices)
        btc_rets = self._compute_returns(btc_prices)
        eth_rets = self._compute_returns(eth_prices)

        # Rolling vols
        self._compute_rolling_vol(spy_rets, 60)
        btc_vol = self._compute_rolling_vol(btc_rets, 30)
        eth_vol = self._compute_rolling_vol(eth_rets, 30)

        min_len = min(len(spy_rets), n - 1)

        baseline_value = 1.0
        peak_baseline = 1.0

        daily_baseline = []
        dd_baseline = []

        # Tracking
        collar_weights = []
        crypto_weights = []
        tlt_weights = []
        active_counts = []
        evidence_dates = []
        evidence_returns = []
        evidence_turnovers = []
        previous_weights = None

        # Crisis tracking
        crisis_2008_base = []
        crisis_2008_comb = []
        crisis_2020_base = []
        crisis_2020_comb = []
        crisis_2022_base = []
        crisis_2022_comb = []

        for i in range(180, min_len):  # Skip first 180 days for momentum
            # Current VIX
            vix = vix_data[min(i, len(vix_data)-1)]

            # 6-month momentum
            btc_mom = btc_prices[i] / btc_prices[max(0, i-126)] - 1 if i >= 126 else 0
            eth_mom = eth_prices[i] / eth_prices[max(0, i-126)] - 1 if i >= 126 else 0

            # Yield proxy: use TLT returns as inverse yield proxy
            # Approximate yield spread from TLT/SPY ratio
            rate_change = (tlt_prices[i] / tlt_prices[max(0, i-126)] - 1) * 0.5

            # Simulated yield spread (from TLT price action)
            tlt_mom = tlt_prices[i] / tlt_prices[max(0, i-126)] - 1 if i >= 126 else 0
            yield_spread = tlt_mom * 2  # Proxy: TLT momentum ≈ yield direction

            # Collar overlay
            collar_delta = self._collar_signal(vix, spy_rets[min(i, len(spy_rets)-1)])
            collar_weights.append(abs(collar_delta) > 0)

            # Crypto overlay
            crypto_w = self._crypto_signal(
                btc_mom, btc_vol[min(i, len(btc_vol)-1)],
                eth_mom, eth_vol[min(i, len(eth_vol)-1)],
            )
            crypto_weights.append(crypto_w > 0)

            # Bond duration rotation
            tlt_w, ief_w, shy_w = self._bond_duration_signal(yield_spread, rate_change)
            tlt_weights.append(tlt_w)

            # Active overlay count
            active = (1 if abs(collar_delta) > 0 else 0) + \
                     (1 if crypto_w > 0 else 0) + \
                     (1 if tlt_w != 1.0 else 0)
            active_counts.append(active)

            # Combined allocation
            spy_w = max(0.36, min(0.56, self.BASELINE["spy"] + collar_delta))
            gld_w = max(0.28, min(0.48, self.BASELINE["gld"] - crypto_w))
            tlt_alloc = self.BASELINE["tlt"] * tlt_w
            ief_alloc = self.BASELINE["tlt"] * ief_w
            shy_alloc = self.BASELINE["tlt"] * shy_w
            btc_w = crypto_w * 0.6
            eth_w = crypto_w * 0.4

            # Normalize
            total = spy_w + gld_w + tlt_alloc + ief_alloc + shy_alloc + btc_w + eth_w
            spy_w /= total
            gld_w /= total
            tlt_alloc /= total
            ief_alloc /= total
            shy_alloc /= total
            btc_w /= total
            eth_w /= total
            current_weights = {
                "SPY": spy_w,
                "GLD": gld_w,
                "TLT": tlt_alloc,
                "IEF": ief_alloc,
                "SHY": shy_alloc,
                "BTC": btc_w,
                "ETH": eth_w,
            }
            turnover = compute_one_way_turnover(
                previous_weights or {},
                current_weights,
            )
            previous_weights = current_weights

            # Daily returns
            s_r = spy_rets[min(i, len(spy_rets)-1)]
            g_r = gld_rets[min(i, len(gld_rets)-1)]
            t_r = tlt_rets[min(i, len(tlt_rets)-1)]
            i_r = ief_rets[min(i, len(ief_rets)-1)]
            sh_r = shy_rets[min(i, len(shy_rets)-1)]
            b_r = btc_rets[min(i, len(btc_rets)-1)]
            e_r = eth_rets[min(i, len(eth_rets)-1)]

            base_ret = (
                self.BASELINE["spy"] * s_r +
                self.BASELINE["gld"] * g_r +
                self.BASELINE["tlt"] * t_r
            )
            comb_ret = (
                spy_w * s_r + gld_w * g_r +
                tlt_alloc * t_r + ief_alloc * i_r + shy_alloc * sh_r +
                btc_w * b_r + eth_w * e_r
            )
            evidence_dates.append(dates[min(i + 1, len(dates) - 1)])
            evidence_returns.append(float(comb_ret))
            evidence_turnovers.append(float(turnover))

            baseline_value *= (1 + base_ret)

            peak_baseline = max(peak_baseline, baseline_value)

            daily_baseline.append(base_ret * 100)
            dd_baseline.append((baseline_value / peak_baseline - 1) * 100)

            # Crisis tracking
            date_str = evidence_dates[-1]
            if "2008" in date_str or "2009" in date_str:
                crisis_2008_base.append(base_ret * 100)
                crisis_2008_comb.append(comb_ret * 100)
            if "2020" in date_str:
                crisis_2020_base.append(base_ret * 100)
                crisis_2020_comb.append(comb_ret * 100)
            if "2022" in date_str:
                crisis_2022_base.append(base_ret * 100)
                crisis_2022_comb.append(comb_ret * 100)

        # Compute summary statistics
        if len(daily_baseline) > 0:
            b_cagr = np.mean(daily_baseline) * 252
            b_vol = np.std(daily_baseline) * math.sqrt(252)
            b_sharpe = b_cagr / b_vol if b_vol > 0 else 0
            b_max_dd = min(dd_baseline) if dd_baseline else 0

            c08b = np.sum(crisis_2008_base) if crisis_2008_base else 0
            c08c = np.sum(crisis_2008_comb) if crisis_2008_comb else 0
            c20b = np.sum(crisis_2020_base) if crisis_2020_base else 0
            c20c = np.sum(crisis_2020_comb) if crisis_2020_comb else 0
            c22b = np.sum(crisis_2022_base) if crisis_2022_base else 0
            c22c = np.sum(crisis_2022_comb) if crisis_2022_comb else 0
        else:
            b_cagr = b_vol = b_sharpe = 0
            b_max_dd = 0
            c08b = c08c = c20b = c20c = c22b = c22c = 0

        evidence = build_profitability_evidence(
            dates=evidence_dates,
            gross_returns=evidence_returns,
            turnovers=evidence_turnovers,
            assets=self.REQUIRED_ASSETS,
            data_mode=self._data_mode,
            provenance={
                "source": (
                    "deterministic synthetic generator"
                    if self._data_mode == "synthetic"
                    else "prices table with yield-spread proxy signals"
                ),
                "path": str(self.data_dir / "market.db"),
            },
            transaction_cost_bps=self.transaction_cost_bps,
            initial_capital=100000.0,
            point_in_time=True,
            diagnostic_opt_in=(
                self.allow_synthetic
                if self._data_mode == "synthetic"
                else self.allow_proxy
            ),
        )
        canonical = evidence["metrics"]["net"]
        total_return = canonical["total_return"]

        return BacktestResult(
            total_return=total_return,
            cagr=canonical["cagr"],
            volatility=canonical["volatility"],
            sharpe_ratio=canonical["sharpe_ratio"],
            max_drawdown=canonical["max_drawdown"],
            total_rebalances=evidence["turnover"]["rebalance_count"],
            total_transaction_costs=evidence["costs"]["total_dollars"],
            avg_turnover=evidence["turnover"]["average_per_rebalance"],
            extras={
                "timestamp": datetime.now().isoformat(),
                "start_date": dates[180] if len(dates) > 180 else dates[0],
                "end_date": dates[-1] if dates else "N/A",
                "trading_days": len(daily_baseline),
                "baseline_cagr": round(b_cagr, 2),
                "baseline_vol": round(b_vol, 2),
                "baseline_sharpe": round(b_sharpe, 3),
                "baseline_max_dd": round(b_max_dd, 2),
                "combined_cagr": canonical["cagr"],
                "combined_vol": canonical["volatility"],
                "combined_max_dd": canonical["max_drawdown"],
                "sharpe_delta": round(canonical["sharpe_ratio"] - b_sharpe, 3),
                "dd_improvement": round(b_max_dd - canonical["max_drawdown"], 2),
                "cagr_delta": round(canonical["cagr"] - b_cagr, 2),
                "collar_active_pct": round(
                    sum(collar_weights) / len(collar_weights) * 100, 1
                ) if collar_weights else 0,
                "crypto_active_pct": round(
                    sum(crypto_weights) / len(crypto_weights) * 100, 1
                ) if crypto_weights else 0,
                "bond_rotation_avg_tlt": round(
                    np.mean(tlt_weights) * 100, 1
                ) if tlt_weights else 0,
                "avg_overlays_active": round(
                    np.mean(active_counts), 1
                ) if active_counts else 0,
                "meets_sharpe_target": canonical["sharpe_ratio"] >= 0.90,
                "meets_dd_target": canonical["max_drawdown"] >= -22.0,
                "profitability_evidence": evidence,
            },
            crisis_returns={
                "2008_baseline": round(c08b, 2),
                "2008_combined": round(c08c, 2),
                "2020_baseline": round(c20b, 2),
                "2020_combined": round(c20c, 2),
                "2022_baseline": round(c22b, 2),
                "2022_combined": round(c22c, 2),
            },
        )


def run_combined_backtest(
    *,
    allow_synthetic: bool = False,
    allow_proxy: bool = False,
) -> BacktestResult:
    """Convenience function."""
    bt = CombinedOverlayBacktest(
        allow_synthetic=allow_synthetic,
        allow_proxy=allow_proxy,
    )
    return bt.run_backtest()


def main():
    bt = CombinedOverlayBacktest(
        allow_synthetic="--allow-synthetic" in sys.argv,
        allow_proxy="--allow-proxy" in sys.argv,
    )
    result = bt.run_backtest()
    e = result.extras

    logger.info("=" * 60)
    logger.info("COMBINED OVERLAY BACKTEST v4.90")
    logger.info("=" * 60)
    logger.info(f"Period: {e['start_date']} \u2192 {e['end_date']}")
    logger.info(f"Trading Days: {e['trading_days']}")
    logger.info("")
    logger.info(f"{'Metric':<20} {'Baseline':>10} {'Combined':>10} {'Delta':>10}")
    logger.info("-" * 50)
    logger.info(f"{'CAGR':<20} {e['baseline_cagr']:>9.2f}% {e['combined_cagr']:>9.2f}% {e['cagr_delta']:>+9.2f}%")
    logger.info(f"{'Volatility':<20} {e['baseline_vol']:>9.2f}% {e['combined_vol']:>9.2f}%")
    logger.info(f"{'Sharpe':<20} {e['baseline_sharpe']:>10.3f} {result.sharpe_ratio:>10.3f} {e['sharpe_delta']:>+10.3f}")
    logger.info(f"{'Max Drawdown':<20} {e['baseline_max_dd']:>9.2f}% {e['combined_max_dd']:>9.2f}% {e['dd_improvement']:>+9.2f}pp")
    logger.info("")
    logger.info("Crisis Returns:")
    crisis = result.crisis_returns or {}
    logger.info(f"  {'2008':<8} {crisis.get('2008_baseline', 0):>9.2f}% {crisis.get('2008_combined', 0):>9.2f}%")
    logger.info(f"  {'2020':<8} {crisis.get('2020_baseline', 0):>9.2f}% {crisis.get('2020_combined', 0):>9.2f}%")
    logger.info(f"  {'2022':<8} {crisis.get('2022_baseline', 0):>9.2f}% {crisis.get('2022_combined', 0):>9.2f}%")
    logger.info("")
    logger.info("Overlay Activity:")
    logger.info(f"  Collar active: {e['collar_active_pct']:.0f}% of days")
    logger.info(f"  Crypto active: {e['crypto_active_pct']:.0f}% of days")
    logger.info(f"  Avg TLT weight: {e['bond_rotation_avg_tlt']:.0f}%")
    logger.info(f"  Avg overlays active: {e['avg_overlays_active']:.1f}/4")
    logger.info("")
    logger.info("Targets:")
    logger.info(f"  Sharpe >= 0.90: {'YES' if e['meets_sharpe_target'] else 'NO'} "
          f"({result.sharpe_ratio:.3f})")
    logger.info(f"  Max DD >= -22%: {'YES' if e['meets_dd_target'] else 'NO'} "
          f"({e['combined_max_dd']:.1f}%)")
    logger.info("=" * 60)

    # Save if requested
    if "--save" in sys.argv:
        from dataclasses import asdict
        out_path = bt.data_dir / "backtest_results" / "combined_overlay.json"
        save_results_json(asdict(result), output_path=str(out_path))
        logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    from src.utils.log_config import configure_logging
    configure_logging()
    main()
