#!/usr/bin/env python3
"""
Cross-Asset Relative Value Signal Backtest — v5.71 Validation
Walk-forward backtest for the cross-asset RV z-score mean-reversion signal.

Validates whether z-score-based mean-reversion triggers between SPY/GLD/TLT
add alpha over the baseline 46/38/16 portfolio. Signal weight: 12-33% in ensemble.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.backtest.metrics import (
    BacktestConfig as _BaseConfig,
    BacktestResult,
    compute_metrics,
    compute_crisis_returns,
    save_results_json,
)
from src.paths import PRICES_JSON, BACKTEST_RESULTS_DIR
from src.utils import safe_get
from src.signals.cross_asset_relative_value import CrossAssetRVScanner, ZSCORE_ENTRY


__all__ = ['BacktestConfig', 'CrossAssetRVBacktester']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig(_BaseConfig):
    """Cross-asset RV backtest config — inherits core fields from metrics.BacktestConfig."""
    z_score_window: int = 60  # 60-day rolling window for z-score
    max_shift: float = 0.04


class CrossAssetRVBacktester:
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.price_data: Dict[str, List[Dict]] = {}
        self.dates: List[str] = []
        self.prices: Dict[str, Dict[str, float]] = {}

    def load_data(self, data_path: Optional[str] = None) -> bool:
        path = Path(data_path) if data_path else PRICES_JSON
        if not path.exists():
            logger.error("Price data not found: %s", path)
            return False

        with open(path) as f:
            self.price_data = json.load(f)

        all_dates = set()
        for symbol, bars in self.price_data.items():
            for bar in bars:
                date = bar.get('d', '')
                if date:
                    all_dates.add(date)

        self.dates = sorted(all_dates)
        for date in self.dates:
            self.prices[date] = {}
        for symbol, bars in self.price_data.items():
            for bar in bars:
                date = bar.get('d', '')
                price = bar.get('p', bar.get('close'))
                if date and price and date in self.prices:
                    self.prices[date][symbol] = float(price)

        logger.info("Loaded %d trading days, %d symbols", len(self.dates), len(self.price_data))
        return True

    def _get_signal(self, date: str) -> Tuple[str, float]:
        """Compute cross-asset RV signal via z-score of rolling returns."""
        idx = self.dates.index(date) if date in self.dates else -1
        if idx < self.config.z_score_window + 1:
            return "neutral", 0.0

        # Compute rolling returns for SPY, GLD, TLT
        symbols = ['SPY', 'GLD', 'TLT']
        returns_window = {sym: [] for sym in symbols}

        for j in range(idx - self.config.z_score_window, idx):
            d_now = self.dates[j]
            d_prev = self.dates[j - 1]
            for sym in symbols:
                p_now = safe_get(self.prices, d_now, sym)
                p_prev = safe_get(self.prices, d_prev, sym)
                if p_now and p_prev and p_prev > 0:
                    returns_window[sym].append(p_now / p_prev - 1)

        # Check data sufficiency
        if any(len(v) < 20 for v in returns_window.values()):
            return "neutral", 0.0

        # Compute z-scores of recent returns vs window using production scanner
        scanner = CrossAssetRVScanner()
        z_scores = {}
        for sym in symbols:
            r = np.array(returns_window[sym])
            z_arr, _, _ = scanner._compute_z_score(r, window=len(r) - 1)
            z_scores[sym] = z_arr[-1] if not np.isnan(z_arr[-1]) else 0.0

        # Signal: mean-reversion — if SPY z > threshold, expect reversion down
        # Find the most extreme z-score
        spy_z = z_scores.get('SPY', 0.0)
        gld_z = z_scores.get('GLD', 0.0)
        tlt_z = z_scores.get('TLT', 0.0)

        (abs(spy_z) + abs(gld_z) + abs(tlt_z)) / 3

        # SPY mean-reversion signal (negative z = oversold = buy)
        if abs(spy_z) > ZSCORE_ENTRY:
            # Mean-reversion: bet against the extreme
            signal_value = -np.sign(spy_z) * min(abs(spy_z) / 5.0, 0.5)
            direction = "spy_reversion"
        elif abs(gld_z) > ZSCORE_ENTRY:
            signal_value = np.sign(gld_z) * min(abs(gld_z) / 5.0, 0.3)
            direction = "gld_reversion"
        else:
            signal_value = 0.0
            direction = "neutral"

        return direction, signal_value

    def run_backtest(self) -> Optional[BacktestResult]:
        if not self.dates:
            return None

        start = self.config.start_date
        end = self.config.end_date
        trading_days = [d for d in self.dates if start <= d <= end]
        if len(trading_days) < 60:
            return None

        capital = self.config.initial_capital
        weights = {
            'SPY': self.config.base_weights['SPY'],
            'GLD': self.config.base_weights['GLD'],
            'TLT': self.config.base_weights['TLT'],
        }

        rebalance_count = 0
        total_cost = 0.0
        equity_curve = [capital]
        signal_counts = {"spy_reversion": 0, "gld_reversion": 0, "neutral": 0}
        crisis_returns = {}
        diverged_count = 0

        prev_month = trading_days[0][:7]

        for i, date in enumerate(trading_days):
            day_prices = self.prices.get(date, {})
            if i > 0:
                prev_date = trading_days[i - 1]
                prev_prices = self.prices.get(prev_date, {})
                daily_ret = 0.0
                for sym, w in weights.items():
                    p_now = day_prices.get(sym)
                    p_prev = prev_prices.get(sym)
                    if p_now and p_prev and p_prev > 0:
                        daily_ret += w * (p_now / p_prev - 1)
                capital *= (1 + daily_ret)
                equity_curve.append(capital)

            month = date[:7]
            if month != prev_month and i > 0:
                prev_month = month
                signal_type, signal_value = self._get_signal(date)
                signal_counts[signal_type] += 1

                if abs(signal_value) > 0.01:
                    diverged_count += 1

                # Apply signal as allocation shift
                shift = signal_value * self.config.max_shift
                new_weights = {
                    'SPY': self.config.base_weights['SPY'] + shift,
                    'GLD': self.config.base_weights['GLD'] - shift * 0.3,
                    'TLT': self.config.base_weights['TLT'] - shift * 0.7,
                }
                for sym in new_weights:
                    new_weights[sym] = max(0.05, min(0.60, new_weights[sym]))

                for sym in new_weights:
                    total_cost += abs(new_weights[sym] - weights[sym]) * self.config.transaction_cost_bps / 10000
                weights = new_weights
                rebalance_count += 1

        # Compute metrics using shared module
        metrics = compute_metrics(equity_curve, self.config.initial_capital)

        # Crisis year returns using the overlay equity curve
        # (NOT base_weights — overlay has time-varying allocations)
        crisis_returns = compute_crisis_returns(
            self.prices, trading_days,
            crisis_years=['2008', '2020', '2022'],
            equity_curve=equity_curve,
        )

        total_rebalances = max(rebalance_count, 1)
        diverged_pct = diverged_count / total_rebalances * 100

        return BacktestResult(
            total_return=round((capital / self.config.initial_capital - 1) * 100, 2) if self.config.initial_capital else 0.0,
            cagr=round(metrics.cagr, 2),
            volatility=round(metrics.volatility, 2),
            sharpe_ratio=round(metrics.sharpe_ratio, 2),
            max_drawdown=round(metrics.max_drawdown, 2),
            total_rebalances=rebalance_count,
            total_transaction_costs=round(total_cost * 10000, 1),
            crisis_returns=crisis_returns,
            extras={
                "strategy_name": "Cross-Asset Relative Value Overlay",
                "start_date": trading_days[0],
                "end_date": trading_days[-1],
                "initial_capital": self.config.initial_capital,
                "final_value": round(capital, 2),
                "signal_distribution": signal_counts,
                "avg_z_score": 0.0,
                "diverged_pct": round(diverged_pct, 1),
            },
        )

    def print_report(self, result: BacktestResult):
        ex = result.extras
        print(f"\n{'='*60}")
        print(f"Cross-Asset Relative Value Signal Backtest")
        print(f"{'='*60}")
        print(f"Period: {ex.get('start_date', '?')} to {ex.get('end_date', '?')}")
        print(f"CAGR: {result.cagr:.2f}%")
        print(f"Sharpe: {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"Rebalances: {result.total_rebalances}")
        print(f"Total Cost: {result.total_transaction_costs:.1f} bps")
        print(f"Diverged Months: {ex.get('diverged_pct', 0.0):.1f}%")
        print(f"\nCrisis Years:")
        for year, ret in sorted((result.crisis_returns or {}).items()):
            print(f"  {year}: {ret:+.2f}%")
        print(f"\nSignal Distribution:")
        for sig, count in ex.get('signal_distribution', {}).items():
            print(f"  {sig}: {count}")

    def save_results(self, result: BacktestResult, output_path: str = None):
        from dataclasses import asdict
        data = asdict(result)
        save_results_json(
            data,
            output_path=output_path or str(BACKTEST_RESULTS_DIR / "cross_asset_rv_backtest.json"),
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cross-Asset RV Backtest")
    parser.add_argument("command", choices=["run"], help="Command")
    parser.add_argument("--save", action="store_true", help="Save results")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2026-05-15")
    args = parser.parse_args()

    config = BacktestConfig(start_date=args.start, end_date=args.end)
    bt = CrossAssetRVBacktester(config)
    if not bt.load_data():
        return
    result = bt.run_backtest()
    if result:
        bt.print_report(result)
        if args.save:
            bt.save_results(result)


if __name__ == "__main__":
    main()
