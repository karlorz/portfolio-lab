"""
Shared backtest metrics, dataclasses, and utilities.

Consolidates duplicated BacktestResult definitions and metric computation
functions that were previously copy-pasted across 11+ backtest files.
"""

import json
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.paths import BASE_ALLOCATION

logger = logging.getLogger(__name__)
from src.costs.etf_cost_table import ETF_COST_BPS as _ETF_COST_BPS


__all__ = ['BacktestConfig', 'DailyPrices', 'BacktestResult', 'BacktestMetrics', 'OverlayMetrics', 'CrisisReturns', 'compute_metrics', 'compute_crisis_returns', 'print_metrics_report', 'compute_deflated_sharpe_ratio', 'save_results_json']

# ── Module-level constants ──────────────────────────────────────────
TRADING_DAYS_PER_YEAR: int = 252
DEFAULT_CRISIS_YEARS: List[str] = ['2008', '2020', '2022']
REBALANCE_FREQUENCY_DAYS: int = 21   # Monthly (~21 trading days)
DEFAULT_TRANSACTION_COST_BPS: float = 10.0

# ── Shared Dataclass Consolidation (v953) ───────────────────────────
# These replace duplicated definitions across 14+ backtest files.
# Backtest-specific extras use the 'extras' dict to avoid field clashes.


@dataclass
class BacktestConfig:
    """Canonical backtest configuration shared across all backtest files.

    Backtest-specific parameters go in ``extras`` (e.g. ``extras=dict(max_shift=0.05)``).
    """

    start_date: str = "2006-01-01"
    end_date: str = "2026-05-15"
    initial_capital: float = 100000.0

    # Base allocation (default: 46/38/16 SPY/GLD/TLT)
    base_weights: Dict[str, float] = field(default_factory=lambda: dict(BASE_ALLOCATION))

    # Rebalancing
    rebalance_frequency_days: int = REBALANCE_FREQUENCY_DAYS  # monthly (~21 trading days)
    rebalance_frequency: str = "monthly"  # string alias for compatibility
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS

    # Per-ETF transaction costs (bps). Falls back to transaction_cost_bps for unknown symbols.
    transaction_costs_by_symbol: Dict[str, float] = field(default_factory=lambda: dict(_ETF_COST_BPS))

    # Backtest-specific extras (shift limits, thresholds, lookbacks, etc.)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DailyPrices:
    """Canonical daily price snapshot shared across backtest files.

    Core fields (date / spy / gld / tlt) are required; additional symbols
    use Optional fields or go in ``extras``.
    """

    date: str
    spy: float
    gld: float
    tlt: float
    vix: Optional[float] = None
    ief: Optional[float] = None
    shy: Optional[float] = None
    btc: Optional[float] = None
    eth: Optional[float] = None
    extras: Dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Canonical backtest result shared across all backtest files.

    Core metrics mirror ``BacktestMetrics``; overlay-specific and
    file-specific fields go in ``extras``.
    """

    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float

    # Trade / cost stats
    total_rebalances: int = 0
    total_transaction_costs: float = 0.0
    avg_turnover: float = 0.0

    # Overlay comparison (optional)
    baseline_sharpe: Optional[float] = None
    sharpe_improvement: Optional[float] = None

    # Extras for backtest-specific fields
    extras: Dict[str, Any] = field(default_factory=dict)

    # Crisis year returns (optional — populated by compute_crisis_returns)
    crisis_returns: Optional[Dict[str, float]] = None


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
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
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
    equity_curve: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute maximum drawdown during crisis years.

    Uses worst peak-to-trough return within each year instead of
    simple first-day/last-day price change, which misses intra-year
    drawdowns like the 2020 crash-then-recovery pattern.

    When ``equity_curve`` is provided, portfolio values are read from
    it directly (one value per ``trading_days`` entry), which correctly
    reflects dynamic overlay weights.  When omitted, the function falls
    back to computing a static-weight buy-and-hold portfolio from
    ``prices`` and ``base_weights``.

    Args:
        prices: {date: {symbol: price}} lookup.
        trading_days: Sorted list of trading dates.
        crisis_years: Years to compute (default: ['2008', '2020', '2022']).
        base_weights: Static asset weights for fallback (default: 46/38/16).
        equity_curve: Portfolio values aligned 1:1 with trading_days.
            When set, ``prices`` and ``base_weights`` are ignored.

    Returns:
        {year: max_drawdown_pct} dict (negative values = losses).
    """
    if crisis_years is None:
        crisis_years = DEFAULT_CRISIS_YEARS
    if base_weights is None:
        base_weights = BASE_ALLOCATION

    # Pre-map trading day to equity curve index for O(1) lookups
    if equity_curve is not None:
        day_to_idx = {d: i for i, d in enumerate(trading_days)}

    result = {}
    for year in crisis_years:
        year_days = [d for d in trading_days if d.startswith(year)]
        if len(year_days) < 2:
            continue

        # Build daily portfolio values for the year
        portfolio_values: List[float] = []
        if equity_curve is not None:
            for day in year_days:
                idx = day_to_idx.get(day)
                if idx is not None and idx < len(equity_curve):
                    portfolio_values.append(equity_curve[idx])
        else:
            for day in year_days:
                day_prices = prices.get(day, {})
                pv = sum(w * day_prices.get(sym, 0) for sym, w in base_weights.items())
                portfolio_values.append(pv)

        if not portfolio_values or portfolio_values[0] == 0:
            continue

        # Find maximum drawdown (worst peak-to-trough)
        peak = portfolio_values[0]
        max_dd = 0.0
        for pv in portfolio_values:
            if pv > peak:
                peak = pv
            dd = (peak - pv) / peak
            if dd > max_dd:
                max_dd = dd

        result[year] = round(float(-max_dd * 100), 2)  # Negative = loss

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


def compute_deflated_sharpe_ratio(
    sharpe_ratio: float,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Compute the Deflated Sharpe Ratio (DSR).

    DSR corrects for multiple testing when a Sharpe ratio is selected
    from N independent trials. It estimates the probability that the
    best Sharpe among N strategies is statistically significant.

    Reference: Bailey & Lopez de Prado, "The Deflated Sharpe Ratio",
    Journal of Portfolio Management, 2014.

    Args:
        sharpe_ratio: The observed Sharpe ratio (annualized).
        n_trials: Number of independent strategy trials.
        n_observations: Number of return observations (trading days).
        skew: Skewness of returns (default 0.0 for symmetric).
        kurtosis: Excess kurtosis of returns (default 3.0 for normal).
        trading_days_per_year: Annualization factor (default 252).

    Returns:
        DSR value between 0 and 1. Values > 0.95 indicate statistical
        significance at the 5% level. Values > 0.50 suggest the Sharpe
        is likely positive after multiple-testing correction.
    """
    import math

    if n_trials < 1 or n_observations < 1 or sharpe_ratio == 0:
        return 0.0

    # Annualized Sharpe variance (under null hypothesis SR=0)
    # V(SR) ≈ (1 - skew * SR + (kurtosis - 1) / 4 * SR^2) / (T - 1)
    # Simplified for SR≈0 under null: V(SR) ≈ 1 / (T - 1) * annualization
    # But using the full formula for accuracy
    t_years = n_observations / trading_days_per_year
    var_sr = (1 - skew * sharpe_ratio + (kurtosis - 1) / 4 * sharpe_ratio ** 2) / max(n_observations - 1, 1)

    # Expected maximum Sharpe under null (from N independent trials)
    # E[max(SR)] ≈ (1 - gamma) * Z^{-1}(1 - 1/N) + gamma * Z^{-1}(1 - 1/(N*e))
    # where gamma ≈ 0.5772 (Euler-Mascheroni constant)
    # Simplified: E[max(SR)] ≈ sqrt(V(SR)) * (1 - gamma) * Z^{-1}(1 - 1/N)
    # For N > 5, use the approximation: E[max] ≈ sqrt(V(SR)) * sqrt(2 * ln(N))

    if n_trials == 1:
        expected_max_sr = 0.0
    else:
        expected_max_sr = math.sqrt(var_sr) * math.sqrt(2 * math.log(n_trials))

    # Standard deviation of the maximum Sharpe under null
    # sigma_max ≈ sqrt(V(SR)) * sqrt(pi/6 / ln(N)) for large N
    if n_trials > 2:
        sigma_max = math.sqrt(var_sr) * math.sqrt(math.pi / (6 * math.log(n_trials)))
    else:
        sigma_max = math.sqrt(var_sr)

    # DSR = Phi((SR - E[max]) / sigma_max)
    # where Phi is the standard normal CDF
    if sigma_max <= 0:
        return 1.0 if sharpe_ratio > expected_max_sr else 0.0

    z = (sharpe_ratio - expected_max_sr) / sigma_max

    # Approximate standard normal CDF using the logistic function
    # Phi(z) ≈ 1 / (1 + exp(-1.7 * z)) — close approximation
    dsr = 1.0 / (1.0 + math.exp(-1.7 * z))

    return round(dsr, 4)


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
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=_json_serializer)
    except (OSError, TypeError) as e:
        logger.error("Failed to save results to %s: %s", path, e)
        raise


def _json_serializer(obj):
    """Handle numpy and pandas types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    try:
        import pandas as pd
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Timedelta):
            return obj.total_seconds()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
