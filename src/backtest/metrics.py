"""
Shared backtest metrics, dataclasses, and utilities.

Consolidates duplicated BacktestResult definitions and metric computation
functions that were previously copy-pasted across 11+ backtest files.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

from src.paths import BASE_ALLOCATION


@dataclass
class BacktestMetrics:
    """Core portfolio metrics computed from an equity curve."""
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float

    # Trade stats
    total_rebalances: int = 0
    total_transaction_costs: float = 0.0
    avg_turnover: float = 0.0


@dataclass
class OverlayMetrics:
    """Overlay-specific metrics comparing signal vs baseline."""
    baseline_sharpe: float
    sharpe_improvement: float
    overlay_active_count: int = 0
    overlay_active_pct: float = 0.0


@dataclass
class CrisisReturns:
    """Returns during known crisis years."""
    returns: Dict[str, float] = field(default_factory=dict)

    def get(self, year: str) -> Optional[float]:
        return self.returns.get(year)


def compute_metrics(
    equity_curve: List[float],
    initial_capital: float,
    trading_days_per_year: int = 252,
) -> BacktestMetrics:
    """Compute core portfolio metrics from an equity curve.

    Args:
        equity_curve: List of portfolio values over time.
        initial_capital: Starting capital.
        trading_days_per_year: Annualization factor (default 252).

    Returns:
        BacktestMetrics with CAGR, Sharpe, max drawdown, etc.
    """
    if len(equity_curve) < 2:
        return BacktestMetrics(
            total_return=0.0, cagr=0.0, volatility=0.0,
            sharpe_ratio=0.0, max_drawdown=0.0,
        )

    returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i - 1] > 0:
            returns.append(equity_curve[i] / equity_curve[i - 1] - 1)
        else:
            returns.append(0.0)

    total_return = (equity_curve[-1] / initial_capital) - 1 if initial_capital > 0 else 0.0

    n_days = max(len(returns), 1)
    cagr = (equity_curve[-1] / initial_capital) ** (trading_days_per_year / n_days) - 1 \
        if initial_capital > 0 and equity_curve[-1] > 0 else 0.0

    vol = np.std(returns) * np.sqrt(trading_days_per_year) if returns else 0.0
    sharpe = cagr / vol if vol > 0 else 0.0

    # Max drawdown
    peak = initial_capital if initial_capital > 0 else max(equity_curve) if equity_curve else 1.0
    max_dd = 0.0
    for val in equity_curve:
        peak = max(peak, val)
        dd = (val - peak) / peak if peak > 0 else 0.0
        max_dd = min(max_dd, dd)

    return BacktestMetrics(
        total_return=round(total_return * 100, 2),
        cagr=round(cagr * 100, 2),
        volatility=round(vol * 100, 2),
        sharpe_ratio=round(sharpe, 4),
        max_drawdown=round(max_dd * 100, 2),
    )


def compute_crisis_returns(
    prices: Dict[str, Dict[str, float]],
    trading_days: List[str],
    crisis_years: List[str] = None,
    base_weights: Dict[str, float] = None,
) -> Dict[str, float]:
    """Compute portfolio returns during crisis years.

    Args:
        prices: {date: {symbol: price}} lookup.
        trading_days: Sorted list of trading dates.
        crisis_years: Years to compute (default: ['2008', '2020', '2022']).
        base_weights: Asset weights (default: SPY 0.46, GLD 0.38, TLT 0.16).

    Returns:
        {year: return_pct} dict.
    """
    if crisis_years is None:
        crisis_years = ['2008', '2020', '2022']
    if base_weights is None:
        base_weights = BASE_ALLOCATION

    result = {}
    for year in crisis_years:
        year_days = [d for d in trading_days if d.startswith(year)]
        if len(year_days) < 2:
            continue
        first_prices = prices.get(year_days[0], {})
        last_prices = prices.get(year_days[-1], {})
        year_ret = 0.0
        for sym, w in base_weights.items():
            p1 = first_prices.get(sym)
            p2 = last_prices.get(sym)
            if p1 and p2 and p1 > 0:
                year_ret += w * (p2 / p1 - 1)
        result[year] = round(year_ret * 100, 2)

    return result


def print_metrics_report(metrics: BacktestMetrics, title: str = "Backtest Results"):
    """Print a formatted metrics report to stdout."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Total Return: {metrics.total_return:.2f}%")
    print(f"CAGR: {metrics.cagr:.2f}%")
    print(f"Volatility: {metrics.volatility:.2f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2f}%")
    if metrics.total_rebalances > 0:
        print(f"Rebalances: {metrics.total_rebalances}")
        print(f"Transaction Costs: {metrics.total_transaction_costs:.2f}")


def save_results_json(data: dict, output_path: str = None, default_dir: Path = None):
    """Save results dict to JSON file.

    Args:
        data: Dict to serialize.
        output_path: Explicit output path (overrides default_dir).
        default_dir: Directory for auto-named output.
    """
    if output_path:
        path = Path(output_path)
    elif default_dir:
        default_dir.mkdir(parents=True, exist_ok=True)
        path = default_dir / "backtest_results.json"
    else:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=_json_serializer)


def _json_serializer(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
