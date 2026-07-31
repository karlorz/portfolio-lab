"""
Shared backtest metrics, dataclasses, and utilities.

Consolidates duplicated BacktestResult definitions and metric computation
functions that were previously copy-pasted across 11+ backtest files.
"""

import json
import hashlib
import logging
import os
import numpy as np
from dataclasses import asdict, dataclass, field
from pathlib import Path
from scipy import stats as sp_stats
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from src.paths import BASE_ALLOCATION

logger = logging.getLogger(__name__)
from src.costs.etf_cost_table import ETF_COST_BPS as _ETF_COST_BPS

# Dashboard / Caddy-served JSON must be world-readable (Batch HZ residual after
# multi-dest fchmod on signals). save_results_json still uses open()+write.
_PUBLIC_JSON_MODE = 0o644


__all__ = ['BacktestConfig', 'DailyPrices', 'BacktestResult', 'BacktestMetrics', 'OverlayMetrics', 'CrisisReturns', 'compute_metrics', 'compute_one_way_turnover', 'build_profitability_evidence', 'compute_crisis_returns', 'print_metrics_report', 'compute_deflated_sharpe_ratio', 'build_data_snapshot_provenance', 'require_data_snapshot_provenance', 'save_results_json']

# ── Module-level constants ──────────────────────────────────────────
TRADING_DAYS_PER_YEAR: int = 252
DEFAULT_CRISIS_YEARS: List[str] = ['2008', '2020', '2022']
REBALANCE_FREQUENCY_DAYS: int = 21   # Monthly (~21 trading days)
DEFAULT_TRANSACTION_COST_BPS: float = 10.0
DATA_SNAPSHOT_SCHEMA_VERSION: str = "data-snapshot/v1"
PROFITABILITY_EVIDENCE_SCHEMA_VERSION: str = "profitability-evidence/v1"

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

    # Risk-adjusted ratios
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    tail_ratio: float = 0.0
    omega_ratio: float = 0.0

    # Distribution shape
    skewness: float = 0.0
    kurtosis: float = 0.0

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

    returns_arr = np.array(returns)

    # Sortino ratio (downside deviation)
    downside_returns = returns_arr[returns_arr < 0]
    downside_dev = np.std(downside_returns) * np.sqrt(trading_days_per_year) if len(downside_returns) > 0 else 0.0
    sortino = cagr / downside_dev if downside_dev > 0 else 0.0

    # Calmar ratio (CAGR / abs max drawdown)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Tail ratio (95th percentile gain / abs 5th percentile loss)
    if len(returns_arr) >= 20:
        p95 = np.percentile(returns_arr, 95)
        p5 = np.percentile(returns_arr, 5)
        tail_ratio = p95 / abs(p5) if p5 != 0 else 0.0
    else:
        tail_ratio = 0.0

    # Omega ratio (probability-weighted gain / loss)
    threshold = 0.0
    gains = returns_arr[returns_arr > threshold] - threshold
    losses = threshold - returns_arr[returns_arr <= threshold]
    omega = float(np.sum(gains) / np.sum(losses)) if np.sum(losses) > 0 else 0.0

    # Skewness and kurtosis
    skewness = float(sp_stats.skew(returns_arr)) if len(returns_arr) >= 3 else 0.0
    kurtosis = float(sp_stats.kurtosis(returns_arr)) if len(returns_arr) >= 4 else 0.0

    return BacktestMetrics(
        total_return=round(total_return * 100, 2),
        cagr=round(cagr * 100, 2),
        volatility=round(vol * 100, 2),
        sharpe_ratio=round(sharpe, 4),
        max_drawdown=round(max_dd * 100, 2),
        sortino_ratio=round(sortino, 4),
        calmar_ratio=round(calmar, 4),
        tail_ratio=round(tail_ratio, 4),
        omega_ratio=round(omega, 4),
        skewness=round(skewness, 4),
        kurtosis=round(kurtosis, 4),
    )


def compute_one_way_turnover(
    previous_weights: Mapping[str, float],
    new_weights: Mapping[str, float],
) -> float:
    """Return one-way turnover, including an implicit residual cash weight."""
    supplied_assets = set(previous_weights) | set(new_weights)
    ordered_assets = sorted(supplied_assets | {"CASH"})
    previous = {
        asset: float(previous_weights.get(asset, 0.0))
        for asset in ordered_assets
    }
    current = {
        asset: float(new_weights.get(asset, 0.0))
        for asset in ordered_assets
    }

    if "CASH" not in supplied_assets:
        previous["CASH"] = 1.0 - sum(previous.values())
        current["CASH"] = 1.0 - sum(current.values())

    return sum(
        abs(current.get(asset, 0.0) - previous.get(asset, 0.0))
        for asset in ordered_assets
    ) / 2.0


def build_profitability_evidence(
    *,
    dates: Sequence[str],
    gross_returns: Sequence[float],
    turnovers: Optional[Sequence[float]],
    assets: Sequence[str],
    data_mode: str,
    provenance: Mapping[str, Any],
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    initial_capital: float = 100000.0,
    point_in_time: bool = True,
    diagnostic_opt_in: bool = False,
    require_real_data: bool = False,
    missing_data_policy: str = "fail_closed",
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, Any]:
    """Build the canonical, reproducible profitability evidence artifact.

    The input returns and turnover must already share one strictly increasing
    daily index. Transaction costs are charged on the same date as turnover:
    ``net_return = gross_return - turnover * bps / 10_000``. All summary
    metrics are delegated to :func:`compute_metrics`.

    Non-real data is diagnostic-only and requires an explicit opt-in. Such
    evidence is never eligible for promotion, even when the calculation is
    otherwise point-in-time safe.
    """
    mode = str(data_mode).lower()
    if mode not in {"real", "proxy", "synthetic"}:
        raise ValueError("data_mode must be one of: real, proxy, synthetic")
    if mode != "real" and not diagnostic_opt_in:
        raise ValueError(
            f"{mode} profitability evidence requires explicit diagnostic opt-in"
        )
    if require_real_data and mode != "real":
        raise ValueError(
            f"decision-grade profitability evidence requires real data, got {mode}"
        )
    if not provenance:
        raise ValueError("profitability evidence requires data provenance")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be positive")

    normalized_dates = [
        value.isoformat() if hasattr(value, "isoformat") else str(value)
        for value in dates
    ]
    gross = [float(value) for value in gross_returns]
    turnover_values = (
        [0.0] * len(gross)
        if turnovers is None
        else [float(value) for value in turnovers]
    )
    if not normalized_dates:
        raise ValueError("profitability evidence requires at least one observation")
    if len(normalized_dates) != len(gross) or len(gross) != len(turnover_values):
        raise ValueError(
            "dates, gross_returns, and turnovers must have aligned lengths"
        )
    if any(
        normalized_dates[index] >= normalized_dates[index + 1]
        for index in range(len(normalized_dates) - 1)
    ):
        raise ValueError("profitability evidence dates must be strictly increasing")
    if not assets or any(not str(asset).strip() for asset in assets):
        raise ValueError("profitability evidence requires named assets")
    if len(set(map(str, assets))) != len(assets):
        raise ValueError("profitability evidence assets must be unique")
    if not all(np.isfinite(value) and value > -1.0 for value in gross):
        raise ValueError("gross_returns must be finite and greater than -1")
    if not all(np.isfinite(value) and value >= 0.0 for value in turnover_values):
        raise ValueError("turnovers must be finite and non-negative")

    gross_equity = float(initial_capital)
    net_equity = float(initial_capital)
    gross_curve = [gross_equity]
    net_curve = [net_equity]
    trace: List[Dict[str, Any]] = []
    total_cost_dollars = 0.0
    total_cost_return = 0.0
    max_reconciliation_error = 0.0

    for observation_date, gross_return, turnover in zip(
        normalized_dates, gross, turnover_values
    ):
        cost_return = turnover * transaction_cost_bps / 10000.0
        net_return = gross_return - cost_return
        if net_return <= -1.0:
            raise ValueError(
                f"transaction costs make net return invalid on {observation_date}"
            )

        cost_dollars = net_equity * cost_return
        gross_equity *= 1.0 + gross_return
        net_equity *= 1.0 + net_return
        gross_curve.append(gross_equity)
        net_curve.append(net_equity)
        reconciliation_error = abs(gross_return - cost_return - net_return)
        max_reconciliation_error = max(
            max_reconciliation_error,
            reconciliation_error,
        )
        total_cost_return += cost_return
        total_cost_dollars += cost_dollars
        trace.append({
            "date": observation_date,
            "gross_return": gross_return,
            "turnover": turnover,
            "cost_return": cost_return,
            "cost_dollars": cost_dollars,
            "net_return": net_return,
            "gross_equity": gross_equity,
            "net_equity": net_equity,
        })

    gross_metrics = compute_metrics(
        gross_curve,
        initial_capital,
        trading_days_per_year=trading_days_per_year,
    )
    net_metrics = compute_metrics(
        net_curve,
        initial_capital,
        trading_days_per_year=trading_days_per_year,
    )
    rebalance_count = sum(value > 0 for value in turnover_values)
    total_turnover = float(sum(turnover_values))

    return {
        "schema_version": PROFITABILITY_EVIDENCE_SCHEMA_VERSION,
        "promotion_eligible": bool(mode == "real" and point_in_time),
        "point_in_time": bool(point_in_time),
        "coverage": {
            "start_date": normalized_dates[0],
            "end_date": normalized_dates[-1],
            "observations": len(normalized_dates),
            "aligned_daily_dates": True,
        },
        "assets": [str(asset) for asset in assets],
        "data": {
            "mode": mode,
            "diagnostic_opt_in": bool(diagnostic_opt_in),
            "missing_data_policy": missing_data_policy,
            "provenance": dict(provenance),
        },
        "costs": {
            "transaction_cost_bps": float(transaction_cost_bps),
            "application": "incurred_date",
            "formula": "net_return = gross_return - turnover * bps / 10000",
            "total_return_deduction": float(total_cost_return),
            "total_dollars": float(total_cost_dollars),
            "max_reconciliation_error": float(max_reconciliation_error),
        },
        "turnover": {
            "definition": "one_way_half_sum_absolute_weight_change",
            "rebalance_count": int(rebalance_count),
            "total": total_turnover,
            "average_per_rebalance": (
                total_turnover / rebalance_count if rebalance_count else 0.0
            ),
        },
        "metrics": {
            "source": "src.backtest.metrics.compute_metrics",
            "parameters": {
                "initial_capital": float(initial_capital),
                "trading_days_per_year": int(trading_days_per_year),
            },
            "gross": asdict(gross_metrics),
            "net": asdict(net_metrics),
        },
        "trace": trace,
    }


def compute_metrics_from_returns(
    returns: List[float],
    risk_free_rate: Optional[float] = None,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, float]:
    """Compute core metrics directly from daily returns.

    Lightweight alternative to compute_metrics() when you have raw
    returns instead of an equity curve. Returns a flat dict for easy
    integration into backtest scripts.

    Args:
        returns: List or array of daily returns (e.g., [0.01, -0.005, ...]).
        risk_free_rate: Annual risk-free rate (default from RISK_FREE_RATE).
        trading_days_per_year: Annualization factor (default 252).

    Returns:
        Dict with keys: total_return, cagr, volatility, sharpe, max_drawdown, calmar.
        Values are decimals (not percentages) for direct use in calculations.
    """
    if risk_free_rate is None:
        from src.paths import RISK_FREE_RATE
        risk_free_rate = RISK_FREE_RATE / 100

    returns_arr = np.array(returns, dtype=float)
    n = len(returns_arr)

    if n == 0:
        return {
            'total_return': 0.0, 'cagr': 0.0, 'volatility': 0.0,
            'sharpe': 0.0, 'max_drawdown': 0.0, 'calmar': 0.0,
        }

    total_return = float(np.prod(1 + returns_arr) - 1)
    years = n / trading_days_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    daily_vol = float(np.std(returns_arr, ddof=1)) if n > 1 else 0.0
    annualized_vol = daily_vol * np.sqrt(trading_days_per_year)

    # Sharpe: (CAGR - Rf) / vol
    sharpe = (cagr - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0.0

    # Max drawdown from cumulative returns
    cumulative = np.cumprod(1 + returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdown))

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        'total_return': round(total_return, 6),
        'cagr': round(cagr, 6),
        'volatility': round(annualized_vol, 6),
        'sharpe': round(sharpe, 4),
        'max_drawdown': round(max_dd, 6),
        'calmar': round(calmar, 4),
    }


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
    logger.info(f"\n{'='*60}")
    logger.info(f"{title}")
    logger.info(f"{'='*60}")
    logger.info(f"Total Return: {metrics.total_return:.2f}%")
    logger.info(f"CAGR: {metrics.cagr:.2f}%")
    logger.info(f"Volatility: {metrics.volatility:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    logger.info(f"Sortino Ratio: {metrics.sortino_ratio:.4f}")
    logger.info(f"Calmar Ratio: {metrics.calmar_ratio:.4f}")
    logger.info(f"Tail Ratio: {metrics.tail_ratio:.4f}")
    logger.info(f"Omega Ratio: {metrics.omega_ratio:.4f}")
    logger.info(f"Skewness: {metrics.skewness:.4f}")
    logger.info(f"Kurtosis: {metrics.kurtosis:.4f}")
    logger.info(f"Max Drawdown: {metrics.max_drawdown:.2f}%")
    if metrics.total_rebalances > 0:
        logger.info(f"Rebalances: {metrics.total_rebalances}")
        logger.info(f"Transaction Costs: {metrics.total_transaction_costs:.2f}")


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_value(row: Any) -> Optional[str]:
    if isinstance(row, Mapping):
        value = row.get("d") or row.get("date") or row.get("timestamp")
    elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and row:
        value = row[0]
    else:
        value = None
    if not isinstance(value, str) or not value:
        return None
    return value[:10]


def _price_payload_stats(payload: Any) -> tuple[int, list[str], dict[str, Optional[str]]]:
    symbols: set[str] = set()
    dates: list[str] = []
    row_count = 0

    if isinstance(payload, Mapping):
        for symbol, series in payload.items():
            if not isinstance(symbol, str) or not isinstance(series, list):
                continue
            symbols.add(symbol)
            for row in series:
                row_count += 1
                date = _date_value(row)
                if date is not None:
                    dates.append(date)
    elif isinstance(payload, list):
        for row in payload:
            if not isinstance(row, Mapping):
                continue
            row_count += 1
            symbol = row.get("symbol")
            if isinstance(symbol, str):
                symbols.add(symbol)
            date = _date_value(row)
            if date is not None:
                dates.append(date)

    date_range: dict[str, Optional[str]] = {
        "start": min(dates) if dates else None,
        "end": max(dates) if dates else None,
    }
    return row_count, sorted(symbols), date_range


def _source_manifest_artifact(path: Path, artifact_name: str) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    artifacts = payload.get("artifacts") if isinstance(payload, Mapping) else None
    if not isinstance(artifacts, list):
        return None
    for row in artifacts:
        if isinstance(row, Mapping) and row.get("artifact") == artifact_name:
            return row
    return None


def _manifest_date_range(row: Mapping[str, Any] | None) -> dict[str, Optional[str]]:
    if row is None:
        return {"start": None, "end": None}

    range_value = row.get("date_range")
    if isinstance(range_value, Mapping):
        start = range_value.get("start")
        end = range_value.get("end")
    elif isinstance(range_value, Sequence) and not isinstance(range_value, (str, bytes)) and len(range_value) >= 2:
        start, end = range_value[0], range_value[1]
    else:
        start = row.get("first_observation") or row.get("start_date") or row.get("start")
        end = row.get("latest_observation") or row.get("end_date") or row.get("end")

    return {
        "start": str(start)[:10] if start else None,
        "end": str(end)[:10] if end else None,
    }


def build_data_snapshot_provenance(
    price_artifact_path: str | Path,
    *,
    source_manifest_path: str | Path | None = None,
    symbol_universe: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build deterministic, public-safe provenance for historical data inputs."""
    price_path = Path(price_artifact_path)
    with open(price_path) as f:
        price_payload = json.load(f)

    row_count, inferred_symbols, date_range = _price_payload_stats(price_payload)
    source_path = Path(source_manifest_path) if source_manifest_path is not None else None
    source_row = _source_manifest_artifact(source_path, price_path.name) if source_path is not None else None

    manifest_symbols = source_row.get("symbols") if source_row is not None else None
    if symbol_universe is not None:
        resolved_symbols = sorted({str(symbol) for symbol in symbol_universe})
    elif inferred_symbols:
        resolved_symbols = inferred_symbols
    elif isinstance(manifest_symbols, list):
        resolved_symbols = sorted(str(symbol) for symbol in manifest_symbols)
    else:
        resolved_symbols = []

    if row_count == 0 and source_row is not None and isinstance(source_row.get("row_count"), int):
        row_count = int(source_row["row_count"])
    if date_range == {"start": None, "end": None}:
        date_range = _manifest_date_range(source_row)

    price_hash = _sha256_file(price_path)
    source_hash = _sha256_file(source_path) if source_path is not None and source_path.exists() else None
    fingerprint = {
        "schema_version": DATA_SNAPSHOT_SCHEMA_VERSION,
        "price_snapshot_hash": price_hash,
        "source_manifest_hash": source_hash,
        "row_count": row_count,
        "date_range": date_range,
        "symbol_universe": resolved_symbols,
    }
    snapshot_hash = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": DATA_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": f"data-snapshot:{snapshot_hash[:16]}",
        "snapshot_hash": snapshot_hash,
        "price_artifact_path": price_path.name,
        "source_manifest_path": source_path.name if source_path is not None else None,
        **fingerprint,
    }


def require_data_snapshot_provenance(data: Mapping[str, Any], *, strict: bool = False) -> list[str]:
    """Warn or fail when a result artifact lacks required data snapshot provenance."""
    warnings: list[str] = []
    snapshot = data.get("_data_snapshot")
    if not isinstance(snapshot, Mapping):
        warnings.append("missing _data_snapshot provenance")

    for warning in warnings:
        logger.warning(warning)
    if strict and warnings:
        raise ValueError("; ".join(warnings))
    return warnings


def save_results_json(
    data: dict,
    output_path: str = None,
    default_dir: Path = None,
    validator: Callable[[dict], dict] = None,
    experiment_manifest: Optional[Dict[str, Any]] = None,
    data_snapshot: Optional[Mapping[str, Any]] = None,
):
    """Save results dict to JSON file.

    Args:
        data: Dict to serialize.
        output_path: Explicit output path (overrides default_dir).
        default_dir: Directory for auto-named output.
        validator: Optional validation function. If provided, data is passed
            through this function before serialization. Should return validated
            data or the original data on validation failure.
        experiment_manifest: Optional provenance config for experiment result
            artifacts. When provided, must include ``experiment_id`` and may
            include ``manifest_mode`` (embedded or sidecar), command, module,
            config_snapshot, env_keys, and input_paths. Normal JSON writes are
            unchanged when omitted.
        data_snapshot: Optional historical data snapshot provenance to embed
            under ``_data_snapshot``. Normal JSON writes are unchanged when
            omitted.
    """
    if data_snapshot is not None:
        data = dict(data)
        data["_data_snapshot"] = dict(data_snapshot)

    if validator is not None:
        try:
            data = validator(data)
        except Exception as e:
            logger.warning("Validation callback failed: %s", e)

    if output_path:
        path = Path(output_path)
    elif default_dir:
        default_dir.mkdir(parents=True, exist_ok=True)
        path = default_dir / "backtest_results.json"
    else:
        return

    # Public artifacts have a smaller disclosure surface than private monitor
    # files.  Apply the shared projection here as a last-mile guard for legacy
    # producers that still call save_results_json directly.
    public_output = False
    try:
        from src.dashboard.public_projection import (
            is_public_output_path,
            prepare_payload_for_write,
        )

        public_output = is_public_output_path(path)
        if public_output:
            data = prepare_payload_for_write(data, path, public=True)
    except Exception as projection_exc:  # noqa: BLE001 - preserve legacy saves
        logger.warning("Public payload projection failed for %s: %s", path, projection_exc)

    if experiment_manifest is not None:
        from src.research.experiment_manifest import save_experiment_result_json

        manifest_config = dict(experiment_manifest)
        experiment_id = manifest_config.pop("experiment_id")
        save_experiment_result_json(data, path, experiment_id=experiment_id, **manifest_config)
        _maybe_record_backtest_experiment(data, path, experiment_manifest)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from src.monitor.signal_authority import serialize_json_payload

        path.write_text(
            serialize_json_payload(
                data,
                output_path=path,
                public=public_output,
            ),
            encoding="utf-8",
        )
        # Batch HZ: normalize mode so public dashboard dual-writes via
        # save_results_json never leave sticky 0600 (Caddy 403). Safe for
        # private backtest artifacts on lab hosts (not secrets).
        try:
            os.chmod(path, _PUBLIC_JSON_MODE)
        except OSError as chmod_exc:
            logger.warning("chmod %s after save_results_json failed: %s", path, chmod_exc)
    except (OSError, TypeError, ValueError) as e:
        logger.error("Failed to save results to %s: %s", path, e)
        raise

    _maybe_record_backtest_experiment(data, path, experiment_manifest)


def _maybe_record_backtest_experiment(
    data: dict,
    path: Path,
    experiment_manifest: Optional[Dict[str, Any]],
) -> None:
    """Append experiment row to decision registry when saving result JSON."""
    if experiment_manifest is None:
        return
    experiment_id = experiment_manifest.get("experiment_id")
    if not experiment_id:
        return
    try:
        from src.monitor.decision_registry import record_backtest_experiment

        record_backtest_experiment(
            data,
            experiment_id=str(experiment_id),
            output_path=path,
            name=str(experiment_manifest.get("name") or experiment_id),
            hypothesis=str(experiment_manifest.get("hypothesis") or ""),
            tags=["experiment_manifest"],
        )
    except (ImportError, ValueError, OSError, TypeError) as e:
        logger.warning("Decision registry backtest record skipped: %s", e)


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
