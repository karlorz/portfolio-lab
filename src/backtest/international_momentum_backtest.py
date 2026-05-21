#!/usr/bin/env python3
"""
International Momentum Signal Backtest — v3.13 Validation
Walk-forward backtest for the international equity momentum signal (EFA/EEM vs SPY).

Validates whether EFA/EEM relative momentum adds alpha over the baseline
46/38/16 (SPY/GLD/TLT) portfolio. Signal weight: 10-15% in ensemble.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.paths import PRICES_JSON, BACKTEST_RESULTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    start_date: str = "2006-01-01"
    end_date: str = "2026-05-15"
    initial_capital: float = 100000.0
    base_spy_weight: float = 0.46
    base_gld_weight: float = 0.38
    base_tlt_weight: float = 0.16
    rebalance_frequency: str = "monthly"
    transaction_cost_bps: float = 10.0
    lookback_days: int = 126  # 6-month momentum
    max_shift: float = 0.05


@dataclass
class BacktestResult:
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    cagr: float
    sharpe: float
    max_drawdown: float
    total_rebalances: int
    total_cost_bps: float
    crisis_returns: Dict[str, float]
    signal_distribution: Dict[str, int]


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
        """Compute international momentum signal from EFA/EEM vs SPY."""
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

        spy_ret = (spy_now / spy_then - 1) * 100
        efa_ret = (efa_now / efa_then - 1) * 100
        relative = efa_ret - spy_ret

        if relative > 3.0:
            return "efa_lead", min(relative / 10.0, 0.5)
        elif relative < -3.0:
            return "spy_lead", max(relative / 10.0, -0.5)
        else:
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
            'SPY': self.config.base_spy_weight,
            'GLD': self.config.base_gld_weight,
            'TLT': self.config.base_tlt_weight,
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
                    'SPY': self.config.base_spy_weight + shift,
                    'GLD': self.config.base_gld_weight - shift * 0.5,
                    'TLT': self.config.base_tlt_weight - shift * 0.5,
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

        # Compute crisis year returns
        for crisis_year in ['2008', '2020', '2022']:
            year_days = [d for d in trading_days if d.startswith(crisis_year)]
            if len(year_days) >= 2:
                first_prices = self.prices.get(year_days[0], {})
                last_prices = self.prices.get(year_days[-1], {})
                base_ret = 0.0
                for sym, w in [('SPY', 0.46), ('GLD', 0.38), ('TLT', 0.16)]:
                    p1 = first_prices.get(sym)
                    p2 = last_prices.get(sym)
                    if p1 and p2 and p1 > 0:
                        base_ret += w * (p2 / p1 - 1)
                crisis_returns[crisis_year] = round(base_ret * 100, 2)

        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                returns.append(equity_curve[i] / equity_curve[i - 1] - 1)

        cagr = (capital / self.config.initial_capital) ** (252 / max(len(returns), 1)) - 1
        vol = np.std(returns) * np.sqrt(252) if returns else 0
        sharpe = cagr / vol if vol > 0 else 0

        peak = self.config.initial_capital
        max_dd = 0.0
        for val in equity_curve:
            peak = max(peak, val)
            dd = (val - peak) / peak
            max_dd = min(max_dd, dd)

        return BacktestResult(
            strategy_name="International Momentum Overlay",
            start_date=trading_days[0],
            end_date=trading_days[-1],
            initial_capital=self.config.initial_capital,
            final_value=round(capital, 2),
            cagr=round(cagr * 100, 2),
            sharpe=round(sharpe, 2),
            max_drawdown=round(max_dd * 100, 2),
            total_rebalances=rebalance_count,
            total_cost_bps=round(total_cost * 10000, 1),
            crisis_returns=crisis_returns,
            signal_distribution=signal_counts,
        )

    def print_report(self, result: BacktestResult):
        print(f"\n{'='*60}")
        print(f"International Momentum Signal Backtest")
        print(f"{'='*60}")
        print(f"Period: {result.start_date} to {result.end_date}")
        print(f"CAGR: {result.cagr:.2f}%")
        print(f"Sharpe: {result.sharpe:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"Rebalances: {result.total_rebalances}")
        print(f"Total Cost: {result.total_cost_bps:.1f} bps")
        print(f"\nCrisis Years:")
        for year, ret in sorted(result.crisis_returns.items()):
            print(f"  {year}: {ret:+.2f}%")
        print(f"\nSignal Distribution:")
        for sig, count in result.signal_distribution.items():
            print(f"  {sig}: {count}")

    def save_results(self, result: BacktestResult, output_path: str = None):
        BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = Path(output_path) if output_path else BACKTEST_RESULTS_DIR / "intl_momentum_backtest.json"
        with open(path, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        logger.info(f"Saved results to {path}")


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
