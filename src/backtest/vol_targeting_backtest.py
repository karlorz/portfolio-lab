#!/usr/bin/env python3
"""
Volatility Targeting Overlay Backtest.

Tests whether scaling portfolio positions to maintain a constant volatility
target improves risk-adjusted returns for the champion allocation.

Academic basis: Moreira & Muir (2017) show +0.30-0.50 Sharpe for equities;
Lohre et al. (2020) show +0.02-0.05 for multi-asset portfolios.

Methodology:
- Compute portfolio realized volatility over a rolling window
- When vol < target: scale up positions (up to max_leverage)
- When vol > target: scale down positions (deleverage)
- Rebalance monthly

Usage:
    python -m src.backtest.vol_targeting_backtest
    python -m src.backtest.vol_targeting_backtest --target-vol 0.11 --save
"""

import json
import logging
from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.paths import (
    BASE_ALLOCATION,
    DATA_DIR,
    PRICES_JSON,
    REGIME_VOL_LOOKBACKS,
    REGIME_VOL_SCALING_EXPONENTS,
    RISK_FREE_RATE,
)
from src.backtest.metrics import (
    BacktestMetrics,
    compute_metrics,
    save_results_json,
)
from src.data.price_cache import get_prices_df
from src.signals.vix_term_structure import VIXTermStructureSignalGenerator, VIXRegime

logger = logging.getLogger(__name__)

__all__ = [
    "VolTargetResult",
    "RegimeVolTargetResult",
    "REGIME_VOL_TARGETS",
    "compute_vol_target_backtest",
    "compute_regime_conditional_vol_target_backtest",
]


@dataclass
class VolTargetResult:
    """Result from volatility targeting backtest."""
    analysis_date: str
    base_allocation: Dict[str, float]
    target_vol: float
    vol_lookback: int
    max_leverage: float
    static_sharpe: float
    vol_target_sharpe: float
    sharpe_delta: float
    static_cagr: float
    vol_target_cagr: float
    static_max_dd: float
    vol_target_max_dd: float
    mean_leverage: float
    max_leverage_reached: float
    leverage_above_1_pct: float  # % of days with leverage > 1
    summary: str


def _load_prices() -> pd.DataFrame:
    """Load price data through the shared TTL-cached price DataFrame accessor."""
    symbols = ["SPY", "GLD", "TLT", "IEF"]
    df = get_prices_df(symbols=symbols).dropna()
    if len(df) <= 1:
        try:
            raw = json.loads(PRICES_JSON.read_text())
            records = []
            for sym in symbols:
                for entry in raw.get(sym, []):
                    records.append({"date": entry["d"], "ticker": sym, "price": entry["p"]})
            if records:
                fallback_df = pd.DataFrame(records)
                fallback_df["date"] = pd.to_datetime(fallback_df["date"])
                fallback_df = fallback_df.pivot(index="date", columns="ticker", values="price")
                fallback_df = fallback_df.sort_index().dropna()
                if len(fallback_df) > len(df):
                    df = fallback_df
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.warning("Fallback price load from %s failed: %s", PRICES_JSON, exc)
    df.index.name = "date"
    return df


def _compute_vol_target_leverage(
    realized_vol: float,
    target_vol: float,
    max_leverage: float = 2.0,
    smoothing: float = 0.67,
    prev_leverage: float = 1.0,
    scaling_exponent: float = 1.0,
) -> float:
    """Compute leverage factor for volatility targeting.

    Args:
        realized_vol: Current realized volatility (annualized)
        target_vol: Target volatility (annualized)
        max_leverage: Maximum allowed leverage
        smoothing: Smoothing coefficient (0-1). Lower = more responsive.
        prev_leverage: Previous period's leverage factor
        scaling_exponent: Exponent applied to target/realized vol ratio.
            1.0 is legacy linear scaling; lower values dampen changes.

    Returns:
        Leverage factor (1.0 = no scaling)
    """
    if realized_vol <= 0 or target_vol <= 0:
        return 1.0

    safe_exponent = max(0.0, float(scaling_exponent))
    raw_leverage = (target_vol / realized_vol) ** safe_exponent
    # Smooth to prevent excessive turnover
    smoothed = smoothing * raw_leverage + (1 - smoothing) * prev_leverage
    return round(max(1.0 / max_leverage, min(max_leverage, smoothed)), 4)


def _get_regime_scaling_exponent(
    regime: str,
    regime_scaling_exponents: Optional[Dict[str, float]] = None,
) -> float:
    """Return leverage scaling exponent for a regime with safe fallback."""
    config = regime_scaling_exponents or REGIME_VOL_SCALING_EXPONENTS
    try:
        return max(0.0, float(config.get(regime, 1.0)))
    except (TypeError, ValueError):
        return 1.0


def _get_regime_vol_lookback(
    regime: str,
    regime_lookbacks: Optional[Dict[str, int]] = None,
    default_lookback: int = 63,
) -> int:
    """Return realized-volatility lookback for a regime with safe fallback."""
    config = regime_lookbacks or REGIME_VOL_LOOKBACKS
    try:
        return max(2, int(config.get(regime, default_lookback)))
    except (TypeError, ValueError):
        return max(2, int(default_lookback))


def _compute_realized_vol(
    portfolio_returns: np.ndarray,
    end_idx: int,
    lookback: int,
    fallback_vol: float = 0.15,
) -> float:
    """Compute annualized realized vol over a bounded historical window."""
    window = max(2, int(lookback))
    if end_idx >= window:
        hist_returns = portfolio_returns[end_idx - window : end_idx]
    else:
        hist_returns = portfolio_returns[1:end_idx] if end_idx > 1 else np.array([0.0])
    if len(hist_returns) <= 2:
        return fallback_vol
    return float(np.std(hist_returns) * np.sqrt(252))


def _precompute_realized_vols(
    portfolio_returns: np.ndarray,
    lookbacks: List[int],
    fallback_vol: float = 0.15,
) -> Dict[int, np.ndarray]:
    """Precompute realized-vol arrays for each lookback using prefix sums."""
    returns = np.asarray(portfolio_returns, dtype=float)
    n_days = len(returns)
    prefix_sum = np.concatenate(([0.0], np.cumsum(returns)))
    prefix_sq_sum = np.concatenate(([0.0], np.cumsum(returns * returns)))
    realized_vols: Dict[int, np.ndarray] = {}

    for raw_lookback in sorted(set(lookbacks)):
        window = max(2, int(raw_lookback))
        vols = np.full(n_days, fallback_vol, dtype=float)
        for end_idx in range(n_days):
            if end_idx >= window:
                start_idx = end_idx - window
            elif end_idx > 1:
                start_idx = 1
            else:
                continue

            count = end_idx - start_idx
            if count <= 2:
                continue

            total = prefix_sum[end_idx] - prefix_sum[start_idx]
            total_sq = prefix_sq_sum[end_idx] - prefix_sq_sum[start_idx]
            mean = total / count
            variance = max(0.0, (total_sq / count) - (mean * mean))
            vols[end_idx] = float(np.sqrt(variance) * np.sqrt(252))

        realized_vols[window] = vols

    return realized_vols


def compute_vol_target_backtest(
    base_allocation: Optional[Dict[str, float]] = None,
    target_vol: float = 0.11,
    vol_lookback: int = 63,
    max_leverage: float = 2.0,
    smoothing: float = 0.67,
    rebalance_days: int = 21,
    transaction_cost_bps: float = 10.0,
    save: bool = False,
) -> VolTargetResult:
    """Run volatility targeting overlay backtest.

    Args:
        base_allocation: Base allocation (default: 46/38/16)
        target_vol: Target annualized volatility (default: 0.11)
        vol_lookback: Volatility estimation window in days (default: 63)
        max_leverage: Maximum leverage multiple (default: 2.0)
        smoothing: Leverage smoothing coefficient (default: 0.67)
        rebalance_days: Rebalance frequency (default: 21 = monthly)
        transaction_cost_bps: Transaction cost in basis points
        save: Whether to save results

    Returns:
        VolTargetResult
    """
    if base_allocation is None:
        base_allocation = dict(BASE_ALLOCATION, IEF=0.00)

    logger.info("Loading prices for volatility targeting backtest")
    prices = _load_prices()
    logger.info("Loaded %d days of price data", len(prices))

    symbols = ["SPY", "GLD", "TLT", "IEF"]
    n_days = len(prices)

    # Buy-and-hold returns for the portfolio at each step
    portfolio_returns = np.zeros(n_days)
    portfolio_returns[0] = 0.0

    for i in range(1, n_days):
        ret = 0.0
        for sym in symbols:
            w = base_allocation.get(sym, 0)
            if w > 0:
                sym_ret = prices[sym].iloc[i] / prices[sym].iloc[i - 1] - 1
                ret += w * sym_ret
        portfolio_returns[i] = ret

    # Static backtest
    static_equity = [100000.0]
    for i in range(1, n_days):
        static_equity.append(static_equity[-1] * (1 + portfolio_returns[i]))

    # Vol-targeted backtest
    capital = 100000.0
    vol_target_equity = [capital]
    leverage_history = [1.0]
    prev_leverage = 1.0
    cash = 0.0
    sanitized_vol_lookback = max(2, int(vol_lookback))
    realized_vols = _precompute_realized_vols(
        portfolio_returns,
        [sanitized_vol_lookback],
    )[sanitized_vol_lookback]

    for i in range(1, n_days):
        realized_vol = float(realized_vols[i])

        # Compute target leverage
        leverage = _compute_vol_target_leverage(
            realized_vol, target_vol, max_leverage, smoothing, prev_leverage,
        )

        # Apply leverage to daily return
        # Leverage > 1 means borrow at risk-free rate to scale up
        # Leverage < 1 means hold cash for the remainder
        excess_leverage = leverage - 1.0
        if excess_leverage > 0:
            # Borrow cash to invest more
            borrowed = capital * excess_leverage
            invested_return = (capital + borrowed) * portfolio_returns[i]
            # Borrowing cost: risk-free rate (from paths.py RISK_FREE_RATE env var)
            daily_rf = RISK_FREE_RATE / 100 / 252
            borrowing_cost = borrowed * daily_rf
            daily_pnl = invested_return - borrowing_cost
        else:
            # Hold cash: (1-leverage) fraction in cash earning risk-free
            invested_return = capital * leverage * portfolio_returns[i]
            cash_return = capital * (1 - leverage) * (RISK_FREE_RATE / 100 / 252)
            daily_pnl = invested_return + cash_return

        capital += daily_pnl

        # Rebalance cost: transaction cost on leverage change
        if abs(leverage - prev_leverage) > 0.01:
            trade_value = abs(leverage - prev_leverage) * capital
            capital -= trade_value * transaction_cost_bps / 10000

        vol_target_equity.append(capital)
        leverage_history.append(leverage)
        prev_leverage = leverage

    # Compute metrics
    static_metrics = compute_metrics(
        equity_curve=static_equity,
        initial_capital=100000.0,
    )
    vol_metrics = compute_metrics(
        equity_curve=vol_target_equity,
        initial_capital=100000.0,
    )

    static_sharpe = static_metrics.sharpe_ratio
    vol_target_sharpe = vol_metrics.sharpe_ratio
    sharpe_delta = round(vol_target_sharpe - static_sharpe, 4)

    mean_leverage = round(float(np.mean(leverage_history)), 4)
    max_lev = round(float(np.max(leverage_history)), 4)
    above_1_pct = round(
        float(sum(1 for l in leverage_history if l > 1.01) / len(leverage_history)), 4
    )

    # Summary
    if sharpe_delta > 0.03:
        verdict = "VOL TARGET STRONG — significant Sharpe improvement"
    elif sharpe_delta > 0.01:
        verdict = "VOL TARGET MODEST — mild improvement, reduced tail risk"
    elif sharpe_delta > -0.01:
        verdict = "VOL TARGET NEUTRAL — similar risk-adjusted returns"
    else:
        verdict = "VOL TARGET NEGATIVE — reduces returns without offsetting benefit"

    summary = (
        f"Vol targeting ({target_vol:.0%} target, {vol_lookback}d lookback): "
        f"Sharpe {vol_target_sharpe:.4f} vs static {static_sharpe:.4f} "
        f"(delta {sharpe_delta:+.4f}). "
        f"Mean leverage: {mean_leverage:.2f}x, max: {max_lev:.2f}x, "
        f"above 1x: {above_1_pct:.1%}. {verdict}"
    )

    result = VolTargetResult(
        analysis_date=datetime.now().isoformat(),
        base_allocation=base_allocation,
        target_vol=target_vol,
        vol_lookback=vol_lookback,
        max_leverage=max_leverage,
        static_sharpe=round(static_sharpe, 4),
        vol_target_sharpe=round(vol_target_sharpe, 4),
        sharpe_delta=sharpe_delta,
        static_cagr=round(static_metrics.cagr, 4),
        vol_target_cagr=round(vol_metrics.cagr, 4),
        static_max_dd=round(static_metrics.max_drawdown, 4),
        vol_target_max_dd=round(vol_metrics.max_drawdown, 4),
        mean_leverage=mean_leverage,
        max_leverage_reached=max_lev,
        leverage_above_1_pct=above_1_pct,
        summary=summary,
    )

    if save:
        output_path = DATA_DIR / "vol_targeting_backtest.json"
        save_results_json(asdict(result), output_path=str(output_path))
        logger.info("Saved results to %s", output_path)

    logger.info("Vol Targeting Backtest Results:")
    logger.info("  Static Sharpe:     %.4f (CAGR: %.4f, DD: %.4f)",
                static_sharpe, static_metrics.cagr, static_metrics.max_drawdown)
    logger.info("  Vol Target Sharpe: %.4f (CAGR: %.4f, DD: %.4f)",
                vol_target_sharpe, vol_metrics.cagr, vol_metrics.max_drawdown)
    logger.info("  Delta:             %+.4f", sharpe_delta)
    logger.info("  Leverage:          mean=%.2fx, max=%.2fx, >1x=%.1f%%",
                mean_leverage, max_lev, above_1_pct * 100)

    return result


# ── Regime-Conditional Volatility Targeting ──────────────────────────────

# Target volatilities by regime (annualized).
# Defensive defaults validated with regime-specific scaling/lookbacks.
# Crisis/HighVol targets are low enough to reduce drawdown below 18%;
# LowVol retains a moderate risk budget without letting leverage dominate.
REGIME_VOL_TARGETS: Dict[str, float] = {
    "CRISIS": 0.03,
    "HIGH_VOL": 0.05,
    "NORMAL": 0.08,
    "LOW_VOL": 0.10,
    "RECOVERY": 0.09,
}

# VIX term structure regime to vol target bias mapping
# Based on research: VIX3M/VIX ratio predicts equity returns better than absolute VIX level
VIX_REGIME_VOL_BIAS: Dict[str, float] = {
    "extreme_contango": 0.02,    # Complacency: slightly higher risk budget
    "contango": 0.01,            # Normal: minor boost
    "flat": 0.0,                 # Neutral: no adjustment
    "backwardation": -0.01,      # Caution: reduce risk slightly
    "extreme_backwardation": -0.03,  # Crisis: significant risk reduction
}

def _load_vix_term_structure_data() -> Dict:
    """Load VIX term structure data for backtest integration."""
    vix_data_path = DATA_DIR / 'vix_term_structure.json'
    if not vix_data_path.exists():
        return {}
    try:
        with open(vix_data_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return {}


def _get_vix_regime_for_date(
    date_str: str,
    vix_data: Dict,
    sorted_dates: Optional[List[str]] = None,
) -> Optional[str]:
    """Get VIX term structure regime for a given date."""
    if date_str in vix_data:
        return vix_data[date_str].get('regime')
    dates = sorted_dates if sorted_dates is not None else sorted(vix_data)
    if not dates:
        return None
    idx = bisect_right(dates, date_str) - 1
    if idx < 0:
        return None
    return vix_data[dates[idx]].get('regime')


def _classify_regime_from_vol(
    realized_vol: float,
    median_vol: float,
    prev_regime: str = "NORMAL",
    vol_declining: bool = False,
) -> str:
    """Classify regime from realized portfolio volatility relative to median.

    Uses realized vol relative to the portfolio's long-term median:
    - >1.7x median → CRISIS
    - 1.25-1.7x median → HIGH_VOL
    - 0.75-1.25x median → NORMAL
    - <0.75x median → LOW_VOL
    - Declining from CRISIS/HIGH_VOL → RECOVERY

    This self-calibrates for any portfolio's typical volatility level.
    """
    if median_vol <= 0:
        return "NORMAL"

    ratio = realized_vol / median_vol

    if ratio >= 1.7:
        return "CRISIS"
    elif ratio >= 1.25:
        return "HIGH_VOL"
    elif ratio < 0.75:
        return "LOW_VOL"
    elif prev_regime in ("CRISIS", "HIGH_VOL") and vol_declining:
        return "RECOVERY"
    else:
        return "NORMAL"


@dataclass
class RegimeVolTargetResult:
    """Result from regime-conditional volatility targeting backtest."""
    analysis_date: str
    base_allocation: Dict[str, float]
    regime_targets: Dict[str, float]
    regime_scaling_exponents: Dict[str, float]
    regime_lookbacks: Dict[str, int]
    vol_lookback: int
    max_leverage: float
    static_sharpe: float
    vol_target_sharpe: float
    sharpe_delta: float
    static_cagr: float
    vol_target_cagr: float
    static_max_dd: float
    vol_target_max_dd: float
    mean_leverage: float
    max_leverage_reached: float
    leverage_above_1_pct: float
    # Per-regime breakdown
    regime_breakdown: Dict[str, Dict[str, float]]
    summary: str


def compute_regime_conditional_vol_target_backtest(
    base_allocation: Optional[Dict[str, float]] = None,
    regime_targets: Optional[Dict[str, float]] = None,
    regime_scaling_exponents: Optional[Dict[str, float]] = None,
    regime_lookbacks: Optional[Dict[str, int]] = None,
    vol_lookback: int = 63,
    max_leverage: float = 2.0,
    smoothing: float = 0.67,
    rebalance_days: int = 21,
    transaction_cost_bps: float = 10.0,
    save: bool = False,
) -> RegimeVolTargetResult:
    """Run regime-conditional volatility targeting overlay backtest.

    Varies the volatility target based on the current VIX regime:
    - CRISIS: low target (deleverage aggressively)
    - HIGH_VOL: moderately low target
    - NORMAL: baseline target
    - LOW_VOL: higher target (take more risk)
    - RECOVERY: slightly elevated target

    Args:
        base_allocation: Base allocation (default: 46/38/16)
        regime_targets: Dict mapping regime → target vol (default: REGIME_VOL_TARGETS)
        regime_scaling_exponents: Dict mapping regime → leverage scaling exponent
        regime_lookbacks: Dict mapping regime → realized-vol lookback in days
        vol_lookback: Volatility estimation window in days
        max_leverage: Maximum leverage multiple
        smoothing: Leverage smoothing coefficient
        rebalance_days: Rebalance frequency
        transaction_cost_bps: Transaction cost in basis points
        save: Whether to save results

    Returns:
        RegimeVolTargetResult with per-regime breakdown
    """
    if base_allocation is None:
        base_allocation = dict(BASE_ALLOCATION, IEF=0.00)
    if regime_targets is None:
        regime_targets = dict(REGIME_VOL_TARGETS)
    if regime_scaling_exponents is None:
        regime_scaling_exponents = dict(REGIME_VOL_SCALING_EXPONENTS)
    if regime_lookbacks is None:
        regime_lookbacks = dict(REGIME_VOL_LOOKBACKS)

    logger.info("Loading prices for regime-conditional vol targeting backtest")
    prices = _load_prices()
    logger.info("Loaded %d days of price data", len(prices))

    # Load VIX term structure data for enhanced regime classification
    vix_data = _load_vix_term_structure_data()
    vix_dates = sorted(vix_data) if vix_data else []
    logger.info("Loaded VIX term structure data: %d records", len(vix_data))

    symbols = ["SPY", "GLD", "TLT", "IEF"]
    n_days = len(prices)

    # Portfolio daily returns
    portfolio_returns = np.zeros(n_days)
    for i in range(1, n_days):
        ret = 0.0
        for sym in symbols:
            w = base_allocation.get(sym, 0)
            if w > 0:
                sym_ret = prices[sym].iloc[i] / prices[sym].iloc[i - 1] - 1
                ret += w * sym_ret
        portfolio_returns[i] = ret

    # Static backtest
    static_equity = [100000.0]
    for i in range(1, n_days):
        static_equity.append(static_equity[-1] * (1 + portfolio_returns[i]))

    # Regime-conditional vol-targeted backtest
    capital = 100000.0
    vol_target_equity = [capital]
    leverage_history = [1.0]
    target_vol_history: List[float] = []
    regime_history: List[str] = []
    prev_leverage = 1.0
    prev_regime = "NORMAL"

    lookback_candidates = {vol_lookback}
    for regime in set(regime_targets) | set(regime_lookbacks):
        lookback_candidates.add(
            _get_regime_vol_lookback(
                regime, regime_lookbacks, default_lookback=vol_lookback,
            )
        )
    realized_vols_by_lookback = _precompute_realized_vols(
        portfolio_returns,
        list(lookback_candidates),
    )
    classification_lookback = max(2, int(vol_lookback))
    classification_vols = realized_vols_by_lookback[classification_lookback]

    # Compute long-term median vol for relative regime classification.
    rolling_vols = classification_vols[classification_lookback:]
    median_vol = float(np.median(rolling_vols)) if len(rolling_vols) > 0 else 0.11
    logger.info("Portfolio median %dd realized vol: %.4f", classification_lookback, median_vol)

    # Per-regime stats
    regime_days: Dict[str, int] = {r: 0 for r in regime_targets}
    regime_total_leverage: Dict[str, float] = {r: 0.0 for r in regime_targets}

    for i in range(1, n_days):
        # Classify regime from the stable baseline window, then estimate
        # realized vol with the regime-specific adaptive lookback.
        classification_vol = float(classification_vols[i])

        # Determine current regime from realized vol trend
        if i >= classification_lookback * 2:
            earlier_vol = float(classification_vols[i - classification_lookback])
            vol_declining = classification_vol < earlier_vol * 0.85
        else:
            vol_declining = False
        regime = _classify_regime_from_vol(classification_vol, median_vol, prev_regime, vol_declining)
        adaptive_lookback = _get_regime_vol_lookback(
            regime, regime_lookbacks, default_lookback=vol_lookback,
        )
        if adaptive_lookback not in realized_vols_by_lookback:
            realized_vols_by_lookback.update(
                _precompute_realized_vols(portfolio_returns, [adaptive_lookback])
            )
        realized_vol = float(realized_vols_by_lookback[adaptive_lookback][i])
        scaling_exponent = _get_regime_scaling_exponent(
            regime, regime_scaling_exponents,
        )
        target_vol = regime_targets.get(regime, 0.09)

        # Enhance regime classification with VIX term structure signal
        # VIX slope is a proven predictor of equity returns better than absolute VIX level
        if vix_data:
            # Convert index to string date (works with DatetimeIndex)
            date_str = str(prices.index[i])[:10]  # "2024-01-05"
            vix_regime = _get_vix_regime_for_date(date_str, vix_data, vix_dates)
            if vix_regime:
                vix_bias = VIX_REGIME_VOL_BIAS.get(vix_regime, 0.0)
                target_vol = max(0.03, min(0.15, target_vol + vix_bias))
                logger.debug("VIX regime %s: target vol adjusted by %+.3f to %.3f",
                           vix_regime, vix_bias, target_vol)

        # Compute target leverage
        leverage = _compute_vol_target_leverage(
            realized_vol,
            target_vol,
            max_leverage,
            smoothing,
            prev_leverage,
            scaling_exponent=scaling_exponent,
        )

        # Apply leverage
        excess_leverage = leverage - 1.0
        if excess_leverage > 0:
            borrowed = capital * excess_leverage
            invested_return = (capital + borrowed) * portfolio_returns[i]
            daily_rf = RISK_FREE_RATE / 100 / 252
            borrowing_cost = borrowed * daily_rf
            daily_pnl = invested_return - borrowing_cost
        else:
            invested_return = capital * leverage * portfolio_returns[i]
            cash_return = capital * (1 - leverage) * (RISK_FREE_RATE / 100 / 252)
            daily_pnl = invested_return + cash_return

        capital += daily_pnl

        # Rebalance cost
        if abs(leverage - prev_leverage) > 0.01:
            trade_value = abs(leverage - prev_leverage) * capital
            capital -= trade_value * transaction_cost_bps / 10000

        vol_target_equity.append(capital)
        leverage_history.append(leverage)
        target_vol_history.append(target_vol)
        regime_history.append(regime)

        # Track per-regime stats
        regime_days[regime] = regime_days.get(regime, 0) + 1
        regime_total_leverage[regime] = regime_total_leverage.get(regime, 0.0) + leverage

        prev_leverage = leverage
        prev_regime = regime

    # Compute metrics
    static_metrics = compute_metrics(
        equity_curve=static_equity,
        initial_capital=100000.0,
    )
    vol_metrics = compute_metrics(
        equity_curve=vol_target_equity,
        initial_capital=100000.0,
    )

    static_sharpe = static_metrics.sharpe_ratio
    vol_target_sharpe = vol_metrics.sharpe_ratio
    sharpe_delta = round(vol_target_sharpe - static_sharpe, 4)

    mean_leverage = round(float(np.mean(leverage_history)), 4)
    max_lev = round(float(np.max(leverage_history)), 4)
    above_1_pct = round(
        float(sum(1 for l in leverage_history if l > 1.01) / len(leverage_history)), 4
    )

    # Per-regime breakdown
    total_days = sum(regime_days.values())
    regime_breakdown: Dict[str, Dict[str, float]] = {}
    for reg in sorted(regime_days.keys()):
        days = regime_days.get(reg, 0)
        regime_breakdown[reg] = {
            "days": days,
            "pct_of_time": round(days / max(total_days, 1), 4),
            "mean_leverage": round(
                regime_total_leverage.get(reg, 0.0) / max(days, 1), 4
            ),
            "target_vol": round(regime_targets.get(reg, 0.09), 4),
            "scaling_exponent": round(
                _get_regime_scaling_exponent(reg, regime_scaling_exponents), 4
            ),
            "vol_lookback": _get_regime_vol_lookback(
                reg, regime_lookbacks, default_lookback=vol_lookback,
            ),
        }

    # Summary
    if sharpe_delta > 0.03:
        verdict = "REGIME-CONDITIONAL VOL TARGET STRONG"
    elif sharpe_delta > 0.01:
        verdict = "REGIME-CONDITIONAL VOL TARGET MODEST"
    elif sharpe_delta > -0.01:
        verdict = "REGIME-CONDITIONAL VOL TARGET NEUTRAL"
    else:
        verdict = "REGIME-CONDITIONAL VOL TARGET NEGATIVE"

    summary = (
        f"Regime-conditional ({vol_lookback}d lookback): "
        f"Sharpe {vol_target_sharpe:.4f} vs static {static_sharpe:.4f} "
        f"(delta {sharpe_delta:+.4f}). "
        f"Mean leverage: {mean_leverage:.2f}x, max: {max_lev:.2f}x. "
        f"Targets: {regime_targets}. "
        f"Scaling exponents: {regime_scaling_exponents}. "
        f"Lookbacks: {regime_lookbacks}. {verdict}"
    )

    result = RegimeVolTargetResult(
        analysis_date=datetime.now().isoformat(),
        base_allocation=base_allocation,
        regime_targets=regime_targets,
        regime_scaling_exponents=regime_scaling_exponents,
        regime_lookbacks=regime_lookbacks,
        vol_lookback=vol_lookback,
        max_leverage=max_leverage,
        static_sharpe=round(static_sharpe, 4),
        vol_target_sharpe=round(vol_target_sharpe, 4),
        sharpe_delta=sharpe_delta,
        static_cagr=round(static_metrics.cagr, 4),
        vol_target_cagr=round(vol_metrics.cagr, 4),
        static_max_dd=round(static_metrics.max_drawdown, 4),
        vol_target_max_dd=round(vol_metrics.max_drawdown, 4),
        mean_leverage=mean_leverage,
        max_leverage_reached=max_lev,
        leverage_above_1_pct=above_1_pct,
        regime_breakdown=regime_breakdown,
        summary=summary,
    )

    if save:
        output_path = DATA_DIR / "regime_vol_targeting_backtest.json"
        save_results_json(asdict(result), output_path=str(output_path))
        logger.info("Saved regime-conditional results to %s", output_path)

    logger.info("Regime-Conditional Vol Targeting Backtest Results:")
    logger.info("  Static Sharpe:     %.4f", static_sharpe)
    logger.info("  Vol Target Sharpe: %.4f (delta %+.4f)", vol_target_sharpe, sharpe_delta)
    logger.info("  Mean Leverage:     %.2fx", mean_leverage)
    for reg, info in regime_breakdown.items():
        logger.info("  %s: %d days (%.1f%%), mean lev %.2fx",
                    reg, int(info["days"]), info["pct_of_time"] * 100, info["mean_leverage"])

    return result


def main():
    """CLI entry point."""
    import argparse
    from src.utils.log_config import configure_logging
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Volatility Targeting Overlay Backtest"
    )
    parser.add_argument("--target-vol", type=float, default=0.11,
                        help="Target annualized volatility (default: 0.11)")
    parser.add_argument("--vol-lookback", type=int, default=63,
                        help="Volatility estimation window in days (default: 63)")
    parser.add_argument("--max-leverage", type=float, default=2.0,
                        help="Maximum leverage multiple (default: 2.0)")
    parser.add_argument("--smoothing", type=float, default=0.67,
                        help="Leverage smoothing coefficient (default: 0.67)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to JSON")
    parser.add_argument("--regime-conditional", action="store_true",
                        help="Run regime-conditional volatility targeting")
    args = parser.parse_args()

    if args.regime_conditional:
        result = compute_regime_conditional_vol_target_backtest(
            vol_lookback=args.vol_lookback,
            max_leverage=args.max_leverage,
            smoothing=args.smoothing,
            save=args.save,
        )

        logger.info("\n%s", "=" * 60)
        logger.info("REGIME-CONDITIONAL VOLATILITY TARGETING BACKTEST")
        logger.info("%s", "=" * 60)
        logger.info("  Static Sharpe:      %.4f", result.static_sharpe)
        logger.info("  Vol Target Sharpe:  %.4f", result.vol_target_sharpe)
        logger.info("  Delta:              %+.4f", result.sharpe_delta)
        logger.info("  Static CAGR:        %.4f", result.static_cagr)
        logger.info("  Vol Target CAGR:    %.4f", result.vol_target_cagr)
        logger.info("  Mean Leverage:      %.2fx", result.mean_leverage)
        logger.info("  Max Leverage:       %.2fx", result.max_leverage_reached)
        logger.info("  Leverage > 1x:      %.1f%%", result.leverage_above_1_pct * 100)
        logger.info("\n  Regime breakdown:")
        for reg, info in result.regime_breakdown.items():
            logger.info("    %-12s: %4dd (%6.1f%%) target=%d%% mean_lev=%.2fx",
                        reg, int(info['days']), info['pct_of_time'] * 100,
                        int(info['target_vol'] * 100), info['mean_leverage'])
        logger.info("\n  %s", result.summary)
        logger.info("%s", "=" * 60)
    else:
        result = compute_vol_target_backtest(
            target_vol=args.target_vol,
            vol_lookback=args.vol_lookback,
            max_leverage=args.max_leverage,
            smoothing=args.smoothing,
            save=args.save,
        )

    logger.info("\n%s", "=" * 60)
    logger.info("VOLATILITY TARGETING OVERLAY BACKTEST")
    logger.info("%s", "=" * 60)
    logger.info("  Static Sharpe:      %.4f", result.static_sharpe)
    logger.info("  Vol Target Sharpe:  %.4f", result.vol_target_sharpe)
    logger.info("  Delta:              %+.4f", result.sharpe_delta)
    logger.info("  Static CAGR:        %.4f", result.static_cagr)
    logger.info("  Vol Target CAGR:    %.4f", result.vol_target_cagr)
    logger.info("  Static Max DD:      %.4f", result.static_max_dd)
    logger.info("  Vol Target Max DD:  %.4f", result.vol_target_max_dd)
    logger.info("  Mean Leverage:      %.2fx", result.mean_leverage)
    logger.info("  Max Leverage:       %.2fx", result.max_leverage_reached)
    logger.info("  Leverage > 1x:      %.1f%%", result.leverage_above_1_pct * 100)
    logger.info("\n  %s", result.summary)
    logger.info("%s", "=" * 60)


if __name__ == "__main__":
    main()
