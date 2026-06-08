#!/usr/bin/env python3
"""
Combined Regime-Conditional Allocation × Vol Targeting Overlay Backtest.

Tests whether applying regime-conditional VOLATILITY TARGETING on top of
regime-conditional ALLOCATION yields additive Sharpe improvement.

Layer 1: Regime-conditional base allocation (aggressive tilt)
  - NORMAL:    46/38/16
  - CRISIS:    35/45/20
  - HIGH_VOL:  38/42/20
  - LOW_VOL:   55/30/15
  - RECOVERY:  58/27/15

Layer 2: Regime-conditional vol targeting overlay
  - CRISIS:    target 5%   (deleverage aggressively)
  - HIGH_VOL:  target 7%   (moderately deleverage)
  - NORMAL:    target 9%   (baseline)
  - LOW_VOL:   target 11%  (take more risk)
  - RECOVERY:  target 10%  (slightly elevated)

Comparison against 4 baselines:
  1. Static champion (46/38/16)
  2. Regime-conditional allocation only
  3. Regime-conditional vol targeting only (on static allocation)
  4. Combined system

Usage:
    python -m src.backtest.combined_regime_alloc_vol_target run
    python -m src.backtest.combined_regime_alloc_vol_target run --save
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from src.paths import DATA_DIR, PRICES_JSON, RISK_FREE_RATE
from src.backtest.metrics import compute_metrics, save_results_json
from src.backtest.rolling_vol import precomputed_rolling_volatility
from src.strategy.regime_allocation import REGIME_ALLOCATIONS, DEFAULT_ALLOCATION

logger = logging.getLogger(__name__)

__all__ = [
    "CombinedRegimeResult",
    "run_combined_backtest",
]

TRADING_DAYS = 252
INITIAL_CAPITAL = 100000.0
MAX_LEVERAGE = 1.5

# Regime-conditional vol targets (same as REGIME_VOL_TARGETS in vol_targeting_backtest.py)
REGIME_VOL_TARGETS: Dict[str, float] = {
    "crisis": 0.05,
    "high_vol": 0.07,
    "normal": 0.09,
    "low_vol": 0.11,
    "recovery": 0.10,
}


def classify_regime(
    returns: np.ndarray,
    lookback: int = 252,
) -> str:
    """Classify regime using volatility and recent return percentile.

    Mirrors the classify_regime_simple() from regime_allocation_backtest.py.
    """
    if len(returns) < lookback:
        return "normal"

    recent = returns[-lookback:]

    vols = []
    for i in range(63, len(recent)):
        v = float(np.std(recent[i - 63 : i]) * np.sqrt(TRADING_DAYS))
        vols.append(v)
    if not vols:
        return "normal"

    recent_vol = float(np.std(recent[-63:]) * np.sqrt(TRADING_DAYS))
    median_vol = float(np.median(vols))

    if recent_vol > 1.7 * median_vol and float(np.mean(recent[-21:])) < -0.01:
        return "crisis"
    elif recent_vol > 1.25 * median_vol:
        return "high_vol"
    elif recent_vol < 0.75 * median_vol:
        return "low_vol"
    elif recent_vol < 1.0 * median_vol and float(np.mean(recent[-21:])) > 0.005:
        return "recovery"
    else:
        return "normal"


def load_prices() -> Dict[str, np.ndarray]:
    """Load price data from prices.json.

    Format: { "SPY": [{"d": "2005-01-03", "p": 81.38}, ...], ... }
    """
    with open(PRICES_JSON) as f:
        raw = json.load(f)
    prices = {}
    for symbol, records in raw.items():
        if isinstance(records, list) and len(records) > 0 and isinstance(records[0], dict):
            prices[symbol] = np.array([r["p"] for r in records], dtype=float)
    return prices


def _compute_vol_target_leverage(
    realized_vol: float,
    target_vol: float,
    max_leverage: float = MAX_LEVERAGE,
    smoothing: float = 0.67,
    prev_leverage: float = 1.0,
) -> float:
    """Compute leverage factor for volatility targeting."""
    if realized_vol <= 0:
        return 1.0
    raw_leverage = target_vol / realized_vol
    smoothed = smoothing * raw_leverage + (1 - smoothing) * prev_leverage
    return round(max(1.0 / max_leverage, min(max_leverage, smoothed)), 4)


def _normalized_weights(allocation: Dict[str, float]) -> tuple[float, float, float]:
    """Return normalized SPY/GLD/TLT weights preserving legacy defaults."""
    w_spy = allocation.get("SPY", 0.46)
    w_gld = allocation.get("GLD", 0.38)
    w_tlt = allocation.get("TLT", 0.16)
    total = w_spy + w_gld + w_tlt
    if abs(total - 1.0) > 1e-6 and total > 0:
        w_spy /= total
        w_gld /= total
        w_tlt /= total
    return w_spy, w_gld, w_tlt


def _precompute_allocation_realized_vols(
    spy_ret: np.ndarray,
    gld_ret: np.ndarray,
    tlt_ret: np.ndarray,
    weights: tuple[float, float, float],
    lookback: int = 63,
) -> list[float]:
    """Precompute legacy realized-vol values for one allocation."""
    w_spy, w_gld, w_tlt = weights
    port_returns = w_spy * spy_ret + w_gld * gld_ret + w_tlt * tlt_ret

    # The legacy implementation rebuilt a zero-initialized history array and
    # only populated entries from day 63 onward. Preserve that warmup contract.
    vol_returns = np.array(port_returns, dtype=float, copy=True)
    vol_returns[:lookback] = 0.0

    return precomputed_rolling_volatility(
        vol_returns,
        window=lookback,
        fallback_vol=0.15,
        warmup_std_min_index=lookback,
        annualization_factor=TRADING_DAYS,
    )


@dataclass
class CombinedRegimeRow:
    """Single backtest result row."""
    label: str
    cagr: float
    vol: float
    sharpe: float
    max_dd: float
    sortino: float
    calmar: float
    mean_leverage: float
    max_leverage_reached: float
    sharpe_vs_static: float
    regime_counts: Dict[str, int]


@dataclass
class CombinedRegimeResult:
    """Complete combined regime allocation × vol targeting results."""
    timestamp: str
    data_range: str
    n_days: int
    static_cagr: float
    static_vol: float
    static_sharpe: float
    static_max_dd: float
    champion_alloc: Dict[str, float]
    rows: List[dict]
    best_sharpe_row: Optional[dict]
    combined_sharpe_delta: float
    recommendation: str


def backtest_strategy(
    prices: Dict[str, np.ndarray],
    label: str,
    allocation_map: Dict[str, Dict[str, float]],
    default_alloc: Dict[str, float],
    vol_target_map: Optional[Dict[str, float]] = None,
    apply_vol_target: bool = False,
) -> CombinedRegimeRow:
    """Run a single backtest strategy.

    Args:
        prices: Dict of {symbol: price_array}
        label: Strategy label
        allocation_map: Dict of {regime: {symbol: weight}} for regime-conditional alloc
        default_alloc: Default allocation when regime unknown
        vol_target_map: Dict of {regime: target_vol} for vol targeting
        apply_vol_target: Whether to apply vol targeting overlay

    Returns:
        CombinedRegimeRow with metrics
    """
    spy = prices.get("SPY", np.array([]))
    gld = prices.get("GLD", np.array([]))
    tlt = prices.get("TLT", np.array([]))

    if len(spy) < 300:
        logger.error("Not enough price data: %d days", len(spy))
        return CombinedRegimeRow(
            label=label, cagr=0, vol=0, sharpe=0, max_dd=0,
            sortino=0, calmar=0, mean_leverage=1.0, max_leverage_reached=1.0,
            sharpe_vs_static=0, regime_counts={},
        )

    min_len = min(len(spy), len(gld), len(tlt))
    spy, gld, tlt = spy[:min_len], gld[:min_len], tlt[:min_len]

    # Daily returns
    spy_ret = spy[1:] / spy[:-1] - 1
    gld_ret = gld[1:] / gld[:-1] - 1
    tlt_ret = tlt[1:] / tlt[:-1] - 1

    n = len(spy_ret)
    capital = float(INITIAL_CAPITAL)
    equity_curve = [capital]
    regime_sequence = []
    leverage_history = [1.0]
    prev_leverage = 1.0
    realized_vol_cache: Dict[tuple[float, float, float], list[float]] = {}

    for i in range(n):
        # Build return history up to day i for regime classification
        hist_returns = spy_ret[:i]
        regime = classify_regime(hist_returns)
        regime_sequence.append(regime)

        # Layer 1: Get allocation for this regime
        if regime in allocation_map:
            alloc = allocation_map[regime]
        else:
            alloc = default_alloc

        weights = _normalized_weights(alloc)
        w_spy, w_gld, w_tlt = weights

        # Portfolio return (before leverage)
        port_ret = w_spy * spy_ret[i] + w_gld * gld_ret[i] + w_tlt * tlt_ret[i]

        # Layer 2: Vol targeting overlay
        if apply_vol_target and vol_target_map is not None and i >= 63:
            # Compute realized vol over 63-day window
            realized_vols = realized_vol_cache.get(weights)
            if realized_vols is None:
                realized_vols = _precompute_allocation_realized_vols(
                    spy_ret, gld_ret, tlt_ret, weights,
                )
                realized_vol_cache[weights] = realized_vols
            realized_vol = realized_vols[i] if i < len(realized_vols) else 0.15

            # Determine regime-conditional target vol
            target_vol = vol_target_map.get(regime, 0.09)

            leverage = _compute_vol_target_leverage(
                realized_vol, target_vol, MAX_LEVERAGE, 0.67, prev_leverage,
            )

            # Apply leverage
            excess_leverage = leverage - 1.0
            if excess_leverage > 0:
                borrowed = capital * excess_leverage
                invested_return = (capital + borrowed) * port_ret
                daily_rf = RISK_FREE_RATE / 100 / TRADING_DAYS
                borrowing_cost = borrowed * daily_rf
                daily_pnl = invested_return - borrowing_cost
            else:
                invested_return = capital * leverage * port_ret
                cash_return = capital * (1 - leverage) * (RISK_FREE_RATE / 100 / TRADING_DAYS)
                daily_pnl = invested_return + cash_return

            # Transaction cost on leverage change
            if abs(leverage - prev_leverage) > 0.01:
                trade_value = abs(leverage - prev_leverage) * capital
                daily_pnl -= trade_value * 10 / 10000  # 10 bps

            capital += daily_pnl
            prev_leverage = leverage
            leverage_history.append(leverage)
        else:
            capital *= (1.0 + port_ret)
            leverage_history.append(1.0)

        equity_curve.append(capital)

    # Compute metrics
    metrics = compute_metrics(equity_curve, INITIAL_CAPITAL)

    # Regime counts
    unique, counts = np.unique(regime_sequence, return_counts=True)
    regime_counts = dict(zip(unique, counts))

    # Leverage stats
    mean_lev = float(np.mean(leverage_history))
    max_lev = float(np.max(leverage_history))

    return CombinedRegimeRow(
        label=label,
        cagr=metrics.cagr,
        vol=metrics.volatility,
        sharpe=metrics.sharpe_ratio,
        max_dd=metrics.max_drawdown,
        sortino=metrics.sortino_ratio,
        calmar=metrics.calmar_ratio,
        mean_leverage=mean_lev,
        max_leverage_reached=max_lev,
        sharpe_vs_static=0.0,  # Fill after comparison
        regime_counts=regime_counts,
    )


def run_combined_backtest(save: bool = False) -> CombinedRegimeResult:
    """Run combined regime allocation × vol targeting backtest."""
    logger.info("Loading price data...")
    prices = load_prices()
    spy = prices.get("SPY", np.array([]))
    n_days = len(spy)
    logger.info("Loaded %d days of price data", n_days)

    static_alloc = DEFAULT_ALLOCATION  # 46/38/16
    regime_alloc_map = REGIME_ALLOCATIONS  # Aggressive tilt defaults

    # === 1. Static champion (no regime, no vol) ===
    logger.info("[1/4] Running static champion (46/38/16)...")
    static = backtest_strategy(
        prices, "Static Champion (46/38/16)",
        allocation_map={}, default_alloc=static_alloc,
        apply_vol_target=False,
    )

    # === 2. Regime-conditional allocation only (no vol target) ===
    logger.info("[2/4] Running regime-conditional allocation...")
    regime_only = backtest_strategy(
        prices, "Regime Alloc Only",
        allocation_map=regime_alloc_map, default_alloc=static_alloc,
        apply_vol_target=False,
    )

    # === 3. Regime-conditional vol targeting only (static allocation) ===
    logger.info("[3/4] Running regime vol targeting only (static alloc)...")
    vol_only = backtest_strategy(
        prices, "Vol Target Only",
        allocation_map={}, default_alloc=static_alloc,
        vol_target_map=REGIME_VOL_TARGETS, apply_vol_target=True,
    )

    # === 4. Combined: regime-conditional allocation + vol targeting ===
    logger.info("[4/4] Running combined system...")
    combined = backtest_strategy(
        prices, "Combined System",
        allocation_map=regime_alloc_map, default_alloc=static_alloc,
        vol_target_map=REGIME_VOL_TARGETS, apply_vol_target=True,
    )

    # Fill sharpe_vs_static
    baseline_sharpe = static.sharpe
    for row in [static, regime_only, vol_only, combined]:
        row.sharpe_vs_static = round(row.sharpe - baseline_sharpe, 4)

    rows = [static, regime_only, vol_only, combined]
    rows_sorted = sorted(rows, key=lambda r: r.sharpe, reverse=True)
    best_row = rows_sorted[0]

    # Print results
    logger.info("")
    logger.info("=" * 78)
    logger.info("COMBINED REGIME-CONDITIONAL ALLOCATION × VOL TARGETING BACKTEST")
    logger.info("=" * 78)
    logger.info("Data range: %d trading days", n_days)
    logger.info("")
    header = f"{'Strategy':<42} {'Sharpe':>8} {'Delta':>8} {'CAGR':>8} {'Vol':>8} {'MaxDD':>8} {'Lev':>6}"
    logger.info(header)
    logger.info("-" * 78)
    for r in rows:
        sign = "+" if r.sharpe_vs_static >= 0 else ""
        logger.info(
            "  %-40s %8.4f %s%7.4f %7.2f%% %7.2f%% %7.2f%% %6.2fx",
            r.label, r.sharpe, sign, r.sharpe_vs_static,
            r.cagr * 100, r.vol * 100, r.max_dd * 100, r.mean_leverage,
        )
    logger.info("")
    logger.info("Best by Sharpe: %s (Sharpe %.4f)", best_row.label, best_row.sharpe)
    logger.info("=" * 78)

    # Combined delta vs static
    combined_delta = round(combined.sharpe - static.sharpe, 4)

    if combined_delta > 0.07:
        verdict = "STRONG COMBINED — allocation × vol targeting are complementary"
    elif combined_delta > 0.04:
        verdict = "MODERATE COMBINED — meaningful additive improvement"
    elif combined_delta > 0.02:
        verdict = "MILD COMBINED — small additive benefit"
    else:
        verdict = "NEUTRAL/NEGATIVE — limited or negative combined effect"

    # Check additivity: combined delta vs (alloc_delta + vol_delta)
    alloc_delta = regime_only.sharpe_vs_static
    vol_delta = vol_only.sharpe_vs_static
    expected_additive = round(alloc_delta + vol_delta, 4)
    # Interaction: combined - (alloc + vol) — negative means overlap/cannibalization
    interaction = round(combined_delta - expected_additive, 4)

    logger.info("Additivity analysis:")
    logger.info("  Allocation delta:     %+.4f", alloc_delta)
    logger.info("  Vol targeting delta:  %+.4f", vol_delta)
    logger.info("  Expected additive:    %+.4f", expected_additive)
    logger.info("  Combined actual:      %+.4f", combined_delta)
    logger.info("  Interaction term:     %+.4f (negative = overlap)", interaction)

    recommendation = (
        f"Best strategy: {best_row.label} (Sharpe {best_row.sharpe:.4f}). "
        f"Combined improvement: {combined_delta:+.4f} vs static. "
        f"Interaction term: {interaction:+.4f}. "
        f"Verdict: {verdict}. "
    )

    result = CombinedRegimeResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        data_range=f"2005-01-03 to 2026-05-08 ({n_days} days)",
        n_days=n_days,
        static_cagr=static.cagr,
        static_vol=static.vol,
        static_sharpe=static.sharpe,
        static_max_dd=static.max_dd,
        champion_alloc=static_alloc,
        rows=[asdict(r) for r in rows],
        best_sharpe_row=asdict(best_row),
        combined_sharpe_delta=combined_delta,
        recommendation=recommendation,
    )

    if save:
        output_path = str(DATA_DIR / "combined_regime_alloc_vol_target_results.json")
        save_results_json(
            asdict(result),
            output_path,
            experiment_manifest={
                "experiment_id": "combined-regime-alloc-vol-target",
                "manifest_mode": "sidecar",
                "module": __name__,
                "command": "python -m src.backtest.combined_regime_alloc_vol_target run --save",
                "config_snapshot": {"regime_vol_targets": REGIME_VOL_TARGETS},
                "input_paths": [PRICES_JSON],
            },
        )
        logger.info("Saved results to %s", output_path)

    return result


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Combined Regime-Conditional Allocation × Vol Targeting Backtest",
    )
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run"], help="Run backtest")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()
    run_combined_backtest(save=args.save)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    )
    main()
