#!/usr/bin/env python3
"""
International Momentum Signal Backtest — v3.13 Validation
Walk-forward backtest for the international equity momentum signal (EFA/EEM vs SPY).

Validates whether EFA/EEM relative momentum adds alpha over the baseline
46/38/16 (SPY/GLD/TLT) portfolio. Signal weight: 10-15% in ensemble.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.backtest.metrics import (
    BacktestConfig as _BaseConfig,
    BacktestResult,
    compute_metrics,
    compute_crisis_returns,
    save_results_json,
)
from src.paths import PRICES_JSON, BACKTEST_RESULTS_DIR
from src.signals.international_momentum import InternationalMomentumGenerator, SignalType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig(_BaseConfig):
    """International momentum backtest config — inherits core fields from metrics.BacktestConfig."""
    lookback_days: int = 126  # 6-month momentum
    max_shift: float = 0.05


class InternationalMomentumBacktester:
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.price_data: Dict[str, List[Dict]] = {}
        self.dates: List[str] = []
        self.prices: Dict[str, Dict[str, float]] = {}  # date -> {symbol: price}

    def load_data(self, data_path: Optional[str] = None) -> bool:
        path = Path(data_path) if data_path else PRICES_JSON
        if not path.exists():
            logger.error(f"Price data not found: {path}")
            return False

        with open(path) as f:
            self.price_data = json.load(f)

        # Build date-indexed price lookup
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

        logger.info(f"Loaded {len(self.dates)} trading days, {len(self.price_data)} symbols")
        return True

    def _get_signal(self, date: str) -> Tuple[str, float]:
        """Compute international momentum signal using production signal logic."""
        idx = self.dates.index(date) if date in self.dates else -1
        if idx < self.config.lookback_days:
            return "neutral", 0.0

        lookback_date = self.dates[idx - self.config.lookback_days]

        spy_now = self.prices.get(date, {}).get('SPY')
        spy_then = self.prices.get(lookback_date, {}).get('SPY')
        efa_now = self.prices.get(date, {}).get('EFA')
        efa_then = self.prices.get(lookback_date, {}).get('EFA')

        if not all([spy_now, spy_then, efa_now, efa_then]):
            return "neutral", 0.0

        # Compute relative momentum in decimal format for production signal logic
        spy_momentum_6m = spy_now / spy_then - 1
        efa_momentum_6m = efa_now / efa_then - 1
        efa_vs_spy = efa_momentum_6m - spy_momentum_6m

        # Use production InternationalMomentumGenerator._determine_signal_type
        # for the core signal logic (thresholds, confidence calculation).
        # We skip __init__ because generate_signal() requires database access
        # (VIX, correlation, save) that isn't available in the backtest context.
        generator = object.__new__(InternationalMomentumGenerator)
        signal_type, confidence = generator._determine_signal_type(efa_vs_spy, 0.0)

        # Map SignalType to backtest return format
        if signal_type in (SignalType.EFA_LEAD, SignalType.EEM_LEAD):
            return "efa_lead", min(confidence, 0.5)
        return "neutral", 0.0

    def run_backtest(self) -> Optional[BacktestResult]:
        if not self.dates:
            return None

        # Filter to date range
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
        signal_counts = {"efa_lead": 0, "spy_lead": 0, "neutral": 0}
        crisis_returns = {}

        prev_month = trading_days[0][:7]

        for i, date in enumerate(trading_days):
            # Daily returns
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

            # Monthly rebalance
            month = date[:7]
            if month != prev_month and i > 0:
                prev_month = month
                signal_type, signal_value = self._get_signal(date)
                signal_counts[signal_type] += 1

                # Shift allocation based on signal
                shift = signal_value * self.config.max_shift
                new_weights = {
                    'SPY': self.config.base_weights['SPY'] + shift,
                    'GLD': self.config.base_weights['GLD'] - shift * 0.5,
                    'TLT': self.config.base_weights['TLT'] - shift * 0.5,
                }
                # Clamp
                for sym in new_weights:
                    new_weights[sym] = max(0.05, min(0.60, new_weights[sym]))

                # Transaction cost
                for sym in new_weights:
                    total_cost += abs(new_weights[sym] - weights[sym]) * self.config.transaction_cost_bps / 10000
                weights = new_weights
                rebalance_count += 1

            # Crisis tracking
            year = date[:4]
            if year in ('2008', '2020', '2022') and year not in crisis_returns:
                # Full-year return computed at end
                pass

        # Compute metrics using shared module
        metrics = compute_metrics(equity_curve, self.config.initial_capital)

        # Crisis year returns using the overlay equity curve
        # (NOT base_weights — overlay has time-varying allocations)
        crisis_returns = compute_crisis_returns(
            self.prices, trading_days,
            crisis_years=['2008', '2020', '2022'],
            equity_curve=equity_curve,
        )

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
                "strategy_name": "International Momentum Overlay",
                "start_date": trading_days[0],
                "end_date": trading_days[-1],
                "initial_capital": self.config.initial_capital,
                "final_value": round(capital, 2),
                "signal_distribution": signal_counts,
            },
        )

    def print_report(self, result: BacktestResult):
        ex = result.extras
        print(f"\n{'='*60}")
        print(f"International Momentum Signal Backtest")
        print(f"{'='*60}")
        print(f"Period: {ex.get('start_date', '?')} to {ex.get('end_date', '?')}")
        print(f"CAGR: {result.cagr:.2f}%")
        print(f"Sharpe: {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"Rebalances: {result.total_rebalances}")
        print(f"Total Cost: {result.total_transaction_costs:.1f} bps")
        print(f"\nCrisis Years:")
        for year, ret in sorted((result.crisis_returns or {}).items()):
            print(f"  {year}: {ret:+.2f}%")
        print(f"\nSignal Distribution:")
        for sig, count in ex.get('signal_distribution', {}).items():
            print(f"  {sig}: {count}")

    def save_results(self, result: BacktestResult, output_path: str = None):
        from dataclasses import asdict
        save_results_json(
            asdict(result),
            output_path=output_path or str(BACKTEST_RESULTS_DIR / "intl_momentum_backtest.json"),
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="International Momentum Backtest")
    parser.add_argument("command", choices=["run"], help="Command")
    parser.add_argument("--save", action="store_true", help="Save results")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2026-05-15")
    args = parser.parse_args()

    config = BacktestConfig(start_date=args.start, end_date=args.end)
    bt = InternationalMomentumBacktester(config)
    if not bt.load_data():
        return
    result = bt.run_backtest()
    if result:
        bt.print_report(result)
        if args.save:
            bt.save_results(result)


if __name__ == "__main__":
    main()
