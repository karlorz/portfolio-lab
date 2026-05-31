"""
Regime-Conditional Allocation Backtest.

Compares static champion allocation (SPY/GLD/TLT 46/38/16) against
regime-conditional allocation that varies weights by macro regime.

Tests 4 strategies:
1. Static champion: 46/38/16 across all regimes
2. Regime-conditional: research-backed defaults (regime_allocation.py)
3. Aggressive tilt: stronger regime differentiation
4. Mild tilt: moderate regime differentiation

Usage:
    python -m src.backtest.regime_allocation_backtest
    python -m src.backtest.regime_allocation_backtest --save
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.paths import DATA_DIR, PRICES_JSON
from src.backtest.metrics import compute_metrics, save_results_json
from src.strategy.regime_allocation import REGIME_ALLOCATIONS, DEFAULT_ALLOCATION

logger = logging.getLogger(__name__)

__all__ = [
    "RegimeAllocBacktestRow",
    "RegimeAllocBacktestResult",
    "run_regime_alloc_backtest",
    "run_regime_sweep",
]

TRADING_DAYS = 252
INITIAL_CAPITAL = 100000.0


def classify_regime_simple(
    returns: np.ndarray,
    lookback: int = 252,
) -> str:
    """Classify regime using volatility and recent return percentile.

    Simple heuristic classifier for backtesting when FRED-MD data is unavailable.
    Mirrors the portfolio-lab regime definitions used in the live system.

    Labels:
    - crisis:   vol > 1.7x median vol AND recent 21d return < -1%
    - high_vol: vol > 1.25x median vol
    - low_vol:  vol < 0.75x median vol
    - recovery: vol between 0.75x and 1.0x median vol AND positive 21d momentum
    - normal:   everything else
    """
    if len(returns) < lookback:
        return "normal"

    recent = returns[-lookback:]

    # Rolling vol history for percentile comparison
    vols = []
    for i in range(63, len(recent)):  # need at least 63 days for vol estimate
        v = np.std(recent[i - 63 : i]) * np.sqrt(TRADING_DAYS)
        vols.append(v)
    if not vols:
        return "normal"

    recent_vol = np.std(recent[-63:]) * np.sqrt(TRADING_DAYS)
    median_vol = np.median(vols)

    # Label based on relative volatility and recent return
    if recent_vol > 1.7 * median_vol and np.mean(recent[-21:]) < -0.01:
        return "crisis"
    elif recent_vol > 1.25 * median_vol:
        return "high_vol"
    elif recent_vol < 0.75 * median_vol:
        return "low_vol"
    elif recent_vol < 1.0 * median_vol and np.mean(recent[-21:]) > 0.005:
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


def backtest_allocation(
    prices: Dict[str, np.ndarray],
    allocation_map: Dict[str, Dict[str, float]],
    default_alloc: Dict[str, float],
    initial_capital: float = INITIAL_CAPITAL,
) -> Dict:
    """Run backtest for a regime-conditional allocation strategy.

    Args:
        prices: Dict of {symbol: price_array}
        allocation_map: Dict of {regime: {symbol: weight}}
        default_alloc: Default allocation when regime not in map
        initial_capital: Starting capital

    Returns:
        Dict of backtest metrics and metadata
    """
    spy = prices.get("SPY", np.array([]))
    gld = prices.get("GLD", np.array([]))
    tlt = prices.get("TLT", np.array([]))

    if len(spy) < 300:
        logger.error("Not enough price data: %d days", len(spy))
        return {}

    min_len = min(len(spy), len(gld), len(tlt))
    spy, gld, tlt = spy[:min_len], gld[:min_len], tlt[:min_len]

    # Daily returns
    spy_ret = spy[1:] / spy[:-1] - 1
    gld_ret = gld[1:] / gld[:-1] - 1
    tlt_ret = tlt[1:] / tlt[:-1] - 1

    n = len(spy_ret)
    equity = float(initial_capital)
    equity_curve = [equity]
    regime_sequence = []

    for i in range(n):
        # Build return history up to day i for regime classification
        hist_returns = spy_ret[:i]
        regime = classify_regime_simple(hist_returns)
        regime_sequence.append(regime)

        # Get allocation for this regime
        if regime in allocation_map:
            alloc = allocation_map[regime]
        else:
            alloc = default_alloc

        w_spy = alloc.get("SPY", 0.46)
        w_gld = alloc.get("GLD", 0.38)
        w_tlt = alloc.get("TLT", 0.16)

        # Normalize to 1.0
        total = w_spy + w_gld + w_tlt
        if abs(total - 1.0) > 1e-6 and total > 0:
            w_spy /= total
            w_gld /= total
            w_tlt /= total

        port_ret = w_spy * spy_ret[i] + w_gld * gld_ret[i] + w_tlt * tlt_ret[i]
        equity *= (1.0 + port_ret)
        equity_curve.append(equity)

    # Compute metrics
    metrics = compute_metrics(equity_curve, initial_capital)

    # Count regime occurrences
    unique, counts = np.unique(regime_sequence, return_counts=True)
    regime_counts = dict(zip(unique, counts))

    return {
        "cagr": metrics.cagr,
        "vol": metrics.volatility,
        "sharpe": metrics.sharpe_ratio,
        "max_dd": metrics.max_drawdown,
        "total_return": metrics.total_return,
        "sortino": metrics.sortino_ratio,
        "calmar": metrics.calmar_ratio,
        "regime_counts": regime_counts,
    }


@dataclass
class RegimeAllocBacktestRow:
    """Single result row."""
    label: str
    cagr: float
    vol: float
    sharpe: float
    max_dd: float
    sortino: float
    calmar: float
    regime_counts: Dict[str, int]
    sharpe_delta: float


@dataclass
class RegimeAllocBacktestResult:
    """Complete regime allocation backtest results."""
    timestamp: str
    data_range: str
    n_days: int
    baseline_cagr: float
    baseline_vol: float
    baseline_sharpe: float
    baseline_max_dd: float
    champion_alloc: Dict[str, float]
    rows: List[dict]
    best_sharpe_row: Optional[dict]
    recommendation: str


def run_regime_alloc_backtest(save: bool = False) -> RegimeAllocBacktestResult:
    """Main backtest comparing static vs regime-conditional allocation."""
    logger.info("Loading price data...")
    prices = load_prices()
    spy = prices.get("SPY", np.array([]))
    n_days = len(spy)
    logger.info("Loaded %d days of price data", n_days)

    static_alloc = DEFAULT_ALLOCATION

    # === Strategy 1: Static champion ===
    logger.info("Running static champion backtest...")
    static_result = backtest_allocation(
        prices, allocation_map={}, default_alloc=static_alloc,
    )

    # === Strategy 2: Research-backed regime-conditional ===
    logger.info("Running regime-conditional (research-backed defaults)...")
    regime_result = backtest_allocation(
        prices, allocation_map=REGIME_ALLOCATIONS, default_alloc=static_alloc,
    )

    # === Strategy 3: Aggressive tilt ===
    aggressive_alloc = {
        "normal": static_alloc,
        "crisis": {"SPY": 0.35, "GLD": 0.45, "TLT": 0.20},
        "high_vol": {"SPY": 0.38, "GLD": 0.42, "TLT": 0.20},
        "low_vol": {"SPY": 0.55, "GLD": 0.30, "TLT": 0.15},
        "recovery": {"SPY": 0.58, "GLD": 0.27, "TLT": 0.15},
    }
    logger.info("Running aggressive tilt backtest...")
    aggressive_result = backtest_allocation(
        prices, allocation_map=aggressive_alloc, default_alloc=static_alloc,
    )

    # === Strategy 4: Mild tilt ===
    mild_alloc = {
        "normal": static_alloc,
        "crisis": {"SPY": 0.43, "GLD": 0.40, "TLT": 0.17},
        "high_vol": {"SPY": 0.44, "GLD": 0.39, "TLT": 0.17},
        "low_vol": {"SPY": 0.48, "GLD": 0.36, "TLT": 0.16},
        "recovery": {"SPY": 0.50, "GLD": 0.34, "TLT": 0.16},
    }
    logger.info("Running mild tilt backtest...")
    mild_result = backtest_allocation(
        prices, allocation_map=mild_alloc, default_alloc=static_alloc,
    )

    baseline_sharpe = static_result.get("sharpe", 0.0)
    baseline_cagr = static_result.get("cagr", 0.0)
    baseline_vol = static_result.get("vol", 0.0)
    baseline_max_dd = static_result.get("max_dd", 0.0)

    rows = [
        RegimeAllocBacktestRow(
            label="Static Champion (46/38/16)",
            cagr=static_result.get("cagr", 0.0),
            vol=static_result.get("vol", 0.0),
            sharpe=static_result.get("sharpe", 0.0),
            max_dd=static_result.get("max_dd", 0.0),
            sortino=static_result.get("sortino", 0.0),
            calmar=static_result.get("calmar", 0.0),
            regime_counts=static_result.get("regime_counts", {}),
            sharpe_delta=0.0,
        ),
        RegimeAllocBacktestRow(
            label="Regime-Conditional (research-backed)",
            cagr=regime_result.get("cagr", 0.0),
            vol=regime_result.get("vol", 0.0),
            sharpe=regime_result.get("sharpe", 0.0),
            max_dd=regime_result.get("max_dd", 0.0),
            sortino=regime_result.get("sortino", 0.0),
            calmar=regime_result.get("calmar", 0.0),
            regime_counts=regime_result.get("regime_counts", {}),
            sharpe_delta=regime_result.get("sharpe", 0.0) - baseline_sharpe,
        ),
        RegimeAllocBacktestRow(
            label="Regime-Conditional (aggressive tilt)",
            cagr=aggressive_result.get("cagr", 0.0),
            vol=aggressive_result.get("vol", 0.0),
            sharpe=aggressive_result.get("sharpe", 0.0),
            max_dd=aggressive_result.get("max_dd", 0.0),
            sortino=aggressive_result.get("sortino", 0.0),
            calmar=aggressive_result.get("calmar", 0.0),
            regime_counts=aggressive_result.get("regime_counts", {}),
            sharpe_delta=aggressive_result.get("sharpe", 0.0) - baseline_sharpe,
        ),
        RegimeAllocBacktestRow(
            label="Regime-Conditional (mild tilt)",
            cagr=mild_result.get("cagr", 0.0),
            vol=mild_result.get("vol", 0.0),
            sharpe=mild_result.get("sharpe", 0.0),
            max_dd=mild_result.get("max_dd", 0.0),
            sortino=mild_result.get("sortino", 0.0),
            calmar=mild_result.get("calmar", 0.0),
            regime_counts=mild_result.get("regime_counts", {}),
            sharpe_delta=mild_result.get("sharpe", 0.0) - baseline_sharpe,
        ),
    ]

    # Find best by Sharpe
    rows_sorted = sorted(rows, key=lambda r: r.sharpe, reverse=True)
    best_row = rows_sorted[0]

    logger.info("")
    logger.info("=" * 72)
    logger.info("REGIME-CONDITIONAL ALLOCATION BACKTEST")
    logger.info("=" * 72)
    logger.info("Data range: %d trading days", n_days)
    logger.info("Static champion (46/38/16):  Sharpe=%.4f  CAGR=%.2f%%  Vol=%.2f%%  MaxDD=%.2f%%",
                baseline_sharpe, baseline_cagr, baseline_vol, baseline_max_dd)
    logger.info("")
    for row in rows:
        sign = "+" if row.sharpe_delta >= 0 else ""
        logger.info("  %-50s Sharpe=%.4f (%s%.4f)  CAGR=%.2f%%  Vol=%.2f%%  MaxDD=%.2f%%",
                    row.label, row.sharpe, sign, row.sharpe_delta,
                    row.cagr, row.vol, row.max_dd)
    logger.info("")
    logger.info("Best by Sharpe: %s", best_row.label)
    logger.info("=" * 72)

    recommendation = (
        f"Best strategy: {best_row.label}. "
        "If regime-conditional beats static, activate via REGIME_ALLOC_ENABLED=1 in cron env."
    )

    result = RegimeAllocBacktestResult(
        timestamp=datetime.utcnow().isoformat(),
        data_range=f"2005-01-03 to 2026-05-08 ({n_days} days)",
        n_days=n_days,
        baseline_cagr=baseline_cagr,
        baseline_vol=baseline_vol,
        baseline_sharpe=baseline_sharpe,
        baseline_max_dd=baseline_max_dd,
        champion_alloc=static_alloc,
        rows=[asdict(r) for r in rows],
        best_sharpe_row=asdict(best_row),
        recommendation=recommendation,
    )

    if save:
        output_path = str(DATA_DIR / "regime_alloc_backtest_results.json")
        save_results_json(asdict(result), output_path)
        logger.info("Saved results to %s", output_path)

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Regime-Conditional Allocation Backtest")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run"], help="Run backtest")
    args = parser.parse_args()
    run_regime_alloc_backtest(save=args.save)
