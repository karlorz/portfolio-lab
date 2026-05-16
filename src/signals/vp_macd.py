#!/usr/bin/env python3
"""
Portfolio-Lab v5.55: Volume-Price Adjusted MACD (VP-MACD) Signal

Implements a volume-price-adjusted MACD trading strategy based on:
arXiv 2604.26063 — "A Volume-Price-Adjusted MACD Trading Strategy with
Sensitivity Calibration for U.S. Equity Indices"

VP-MACD improves on traditional MACD by incorporating:
1. Volume weighting: MACD EMAs are volume-weighted when volume data available
2. Volatility-adjusted signal line: threshold widens during high vol to reduce whipsaws
3. Return-magnitude filter: uses daily return vs volatility to filter breakouts

Data Sources:
- prices.json (close prices, full history 2005-2026)
- market.db (volume data 2021+, graceful fallback)

Usage:
    python -m src.signals.vp_macd signal --ticker SPY   # Generate current signal
    python -m src.signals.vp_macd backtest --ticker SPY  # Historical backtest
    python -m src.signals.vp_macd status                 # Show current state
"""

import json
import logging
import sqlite3
import argparse
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PRICES_PATH = PROJECT_ROOT / "public/data/prices.json"
DB_PATH = DATA_DIR / "market.db"
STATE_PATH = DATA_DIR / "signals/vp_macd_state.json"
SIGNALS_DIR = DATA_DIR / "signals"

# Default parameters (from arXiv 2604.26063)
DEFAULT_FAST = 12           # Fast EMA period
DEFAULT_SLOW = 26           # Slow EMA period
DEFAULT_SIGNAL = 9          # Signal line EMA period
DEFAULT_VOL_MULT = 1.5      # Volatility multiplier for signal threshold
DEFAULT_VOL_WINDOW = 20     # Volatility estimation window
DEFAULT_FILTER_THRESHOLD = 0.5  # Min return/vol ratio for breakout filter

# VPIN gating
VPIN_MAX = 0.6              # Max VPIN to enable signal

# Signal values
SIGNAL_STRONG_SHORT = -1.0
SIGNAL_SHORT = -0.5
SIGNAL_NEUTRAL = 0.0
SIGNAL_LONG = 0.5
SIGNAL_STRONG_LONG = 1.0


@dataclass
class VPMACDSignal:
    """VP-MACD signal output."""
    timestamp: str
    ticker: str
    macd_line: float
    signal_line: float
    histogram: float
    volatility_adjusted_threshold: float
    vp_macd_value: float          # -1 to +1 direction
    vp_macd_signal: str           # strong_short/short/neutral/long/strong_long
    confidence: float             # 0-1
    regime: str                   # vol regime classification
    volume_available: bool
    fast_period: int
    slow_period: int
    signal_period: int
    vol_multiplier: float
    details: Dict = field(default_factory=dict)


def _load_prices(ticker: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load close prices and dates from prices.json.

    Returns (dates, prices) numpy arrays sorted chronologically.
    """
    if not PRICES_PATH.exists():
        logger.error(f"Prices file not found: {PRICES_PATH}")
        return np.array([]), np.array([])

    with open(PRICES_PATH) as f:
        data = json.load(f)

    symbol_data = data.get(ticker, data.get(ticker.upper(), []))
    if not symbol_data:
        logger.error(f"Ticker {ticker} not found in prices.json")
        return np.array([]), np.array([])

    # Sort by date and extract
    sorted_data = sorted(symbol_data, key=lambda x: x['d'])
    dates = np.array([item['d'] for item in sorted_data])
    prices = np.array([item['p'] for item in sorted_data], dtype=float)

    return dates, prices


def _load_volume(ticker: str) -> Tuple[List[str], np.ndarray]:
    """Load volume data from market.db when available.

    Returns (dates_list, volume_array) or empty arrays if unavailable.
    """
    if not DB_PATH.exists():
        return [], np.array([])

    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        rows = cursor.execute(
            "SELECT date, volume FROM prices WHERE symbol=? AND volume > 0 ORDER BY date",
            (ticker.upper(),)
        ).fetchall()
        conn.close()

        if rows:
            dates = [r[0] for r in rows]
            volumes = np.array([r[1] for r in rows], dtype=float)
            return dates, volumes
    except Exception as e:
        logger.warning(f"Failed to load volume data: {e}")

    return [], np.array([])


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Compute Exponential Moving Average, handling NaN inputs."""
    if len(values) < period:
        return np.full_like(values, np.nan)

    result = np.full_like(values, np.nan)
    multiplier = 2.0 / (period + 1.0)

    # Find first valid index (where we have `period` non-NaN values)
    valid_indices = np.where(~np.isnan(values))[0]
    if len(valid_indices) < period:
        return result

    # Compute first EMA value as SMA of first `period` valid values
    first_valid_idx = valid_indices[period - 1]
    result[first_valid_idx] = np.mean(values[valid_indices[:period]])

    # Propagate EMA forward
    for i in range(first_valid_idx + 1, len(values)):
        if np.isnan(values[i]):
            result[i] = result[i - 1]  # Hold previous value
        else:
            result[i] = (values[i] - result[i - 1]) * multiplier + result[i - 1]

    return result


def _volume_weighted_ema(
    prices: np.ndarray,
    volumes: Optional[np.ndarray],
    period: int
) -> np.ndarray:
    """Compute Volume-Weighted EMA.

    Uses volume as weight if available, otherwise falls back to standard EMA.
    """
    if volumes is not None and len(volumes) == len(prices) and np.sum(volumes) > 0:
        # Volume-weighted: each price is weighted by its volume
        weighted_prices = prices * volumes / np.mean(volumes)
        return _ema(weighted_prices, period)
    else:
        return _ema(prices, period)


def _compute_vp_macd(
    prices: np.ndarray,
    volumes: Optional[np.ndarray],
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
    signal_period: int = DEFAULT_SIGNAL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute VP-MACD line, signal line, and histogram.

    Returns:
        (macd_line, signal_line, histogram) arrays
    """
    # Volume-weighted fast and slow EMAs
    fast_ema = _volume_weighted_ema(prices, volumes, fast)
    slow_ema = _volume_weighted_ema(prices, volumes, slow)

    # MACD line
    macd_line = fast_ema - slow_ema

    # Signal line (EMA of MACD line)
    signal_line = _ema(macd_line, signal_period)

    # Histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def _compute_volatility(
    prices: np.ndarray,
    window: int = DEFAULT_VOL_WINDOW,
) -> np.ndarray:
    """Compute rolling annualized volatility from daily returns."""
    if len(prices) < window + 1:
        return np.full(len(prices), np.nan)

    returns = np.diff(prices) / prices[:-1]
    vol = np.full(len(prices), np.nan)

    for i in range(window, len(returns) + 1):
        vol[i] = np.std(returns[i - window:i]) * np.sqrt(252)

    return vol


def _classify_vol_regime(volatility: float) -> str:
    """Classify volatility regime."""
    if volatility < 0.12:
        return "low"
    elif volatility < 0.20:
        return "normal"
    elif volatility < 0.30:
        return "elevated"
    else:
        return "high"


def generate_signal(
    ticker: str = "SPY",
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
    signal_period: int = DEFAULT_SIGNAL,
    vol_mult: float = DEFAULT_VOL_MULT,
    filter_threshold: float = DEFAULT_FILTER_THRESHOLD,
    vol_window: int = DEFAULT_VOL_WINDOW,
) -> Optional[VPMACDSignal]:
    """Generate VP-MACD signal for the given ticker.

    Returns None if data is insufficient.
    """
    # Load prices
    dates, prices = _load_prices(ticker)
    if len(prices) < slow + signal_period + 10:
        logger.error(f"Insufficient price data for {ticker}: {len(prices)} points")
        return None

    # Load volume (optional, graceful degradation)
    vol_dates, volumes = _load_volume(ticker)
    volume_available = len(volumes) > 0

    # Sync volume to price dates if available
    synced_volumes = None
    if volume_available:
        # Build date → volume map
        vol_map = dict(zip(vol_dates, volumes))
        synced_volumes = np.array([vol_map.get(d, 0.0) for d in dates])
        # Check if any volume was found
        if np.sum(synced_volumes) == 0:
            synced_volumes = None
            volume_available = False

    # Compute VP-MACD
    macd_line, signal_line, histogram = _compute_vp_macd(
        prices, synced_volumes, fast, slow, signal_period
    )

    # Compute volatility
    vol = _compute_volatility(prices, vol_window)

    # Compute volatility-adjusted threshold
    # Base threshold from MACD std, scaled by vol_mult
    valid_macd = macd_line[~np.isnan(macd_line)]
    if len(valid_macd) < 50:
        return None

    base_threshold = np.std(valid_macd[-252:])  # 1 year std
    vol_adjusted_threshold = base_threshold * vol_mult

    # Current values
    current_macd = macd_line[-1]
    current_signal = signal_line[-1]
    current_hist = histogram[-1]
    current_vol = vol[-1] if not np.isnan(vol[-1]) else 0.15

    # Vol regime
    vol_regime = _classify_vol_regime(current_vol)

    # Return-magnitude filter: recent daily return vs volatility
    if len(prices) >= 3:
        daily_return = (prices[-1] - prices[-3]) / prices[-3]
        return_vol_ratio = abs(daily_return) / (current_vol / np.sqrt(252))
    else:
        daily_return = 0.0
        return_vol_ratio = 0.0

    # Compute VP-MACD signal direction
    # Signal is based on histogram cross of vol-adjusted threshold
    if current_hist > vol_adjusted_threshold and return_vol_ratio > filter_threshold:
        # Strong bullish: histogram above threshold + meaningful move
        direction = SIGNAL_STRONG_LONG
        signal_name = "strong_long"
        confidence = min(0.9, 0.5 + 0.3 * min(1.0, current_hist / (vol_adjusted_threshold * 2)))
    elif current_hist > 0 and current_hist > base_threshold:
        # Moderate bullish
        direction = SIGNAL_LONG
        signal_name = "long"
        confidence = min(0.7, 0.4 + 0.2 * min(1.0, current_hist / vol_adjusted_threshold))
    elif current_hist < -vol_adjusted_threshold and return_vol_ratio > filter_threshold:
        # Strong bearish
        direction = SIGNAL_STRONG_SHORT
        signal_name = "strong_short"
        confidence = min(0.9, 0.5 + 0.3 * min(1.0, abs(current_hist) / (vol_adjusted_threshold * 2)))
    elif current_hist < 0 and abs(current_hist) > base_threshold:
        # Moderate bearish
        direction = SIGNAL_SHORT
        signal_name = "short"
        confidence = min(0.7, 0.4 + 0.2 * min(1.0, abs(current_hist) / vol_adjusted_threshold))
    else:
        # Neutral
        direction = SIGNAL_NEUTRAL
        signal_name = "neutral"
        confidence = 0.3

    # Reduce confidence in high vol (signal less reliable)
    if vol_regime == "high":
        confidence *= 0.7
    elif vol_regime == "elevated":
        confidence *= 0.85

    timestamp = datetime.now(timezone.utc).isoformat()

    signal = VPMACDSignal(
        timestamp=timestamp,
        ticker=ticker,
        macd_line=float(current_macd),
        signal_line=float(current_signal),
        histogram=float(current_hist),
        volatility_adjusted_threshold=float(vol_adjusted_threshold),
        vp_macd_value=direction,
        vp_macd_signal=signal_name,
        confidence=round(confidence, 4),
        regime=vol_regime,
        volume_available=volume_available,
        fast_period=fast,
        slow_period=slow,
        signal_period=signal_period,
        vol_multiplier=vol_mult,
        details={
            "daily_return_pct": round(float(daily_return * 100), 4),
            "return_vol_ratio": round(float(return_vol_ratio), 4),
            "annualized_vol": round(float(current_vol * 100), 2),
            "base_threshold": round(float(base_threshold), 6),
            "prices_available": len(prices),
            "volume_available_count": len(volumes) if volume_available else 0,
        }
    )

    return signal


def backtest(
    ticker: str = "SPY",
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
    signal_period: int = DEFAULT_SIGNAL,
    vol_mult: float = DEFAULT_VOL_MULT,
) -> Dict:
    """Run historical backtest of VP-MACD vs baseline MACD."""
    # Load prices
    dates, prices = _load_prices(ticker)
    if len(prices) < slow + signal_period + 252:
        return {"error": "Insufficient data"}

    # Load volume
    vol_dates, volumes = _load_volume(ticker)
    volume_available = len(volumes) > 0
    synced_volumes = None
    if volume_available:
        vol_map = dict(zip(vol_dates, volumes))
        synced_volumes = np.array([vol_map.get(d, 0.0) for d in dates])
        if np.sum(synced_volumes) == 0:
            synced_volumes = None

    # Compute VP-MACD
    vp_macd_line, vp_signal_line, vp_hist = _compute_vp_macd(
        prices, synced_volumes, fast, slow, signal_period
    )

    # Compute baseline MACD (no volume weighting)
    baseline_macd, baseline_signal, baseline_hist = _compute_vp_macd(
        prices, None, fast, slow, signal_period
    )

    # Compute volatility
    vol = _compute_volatility(prices)
    vol_regimes = [_classify_vol_regime(v) if not np.isnan(v) else "unknown" for v in vol]

    # Compute returns
    returns = np.diff(prices) / prices[:-1]

    # Generate signals and track performance
    vp_positions = np.zeros(len(prices))
    baseline_positions = np.zeros(len(prices))
    vp_trades = 0
    baseline_trades = 0

    for i in range(slow + signal_period, len(prices)):
        # VP-MACD signal
        base_threshold = np.std(vp_macd_line[max(0, i-252):i]) if i >= 252 else np.std(vp_macd_line[:i])
        adj_threshold = base_threshold * vol_mult

        if vp_hist[i] > adj_threshold:
            vp_positions[i] = 1.0
        elif vp_hist[i] < -adj_threshold:
            vp_positions[i] = -1.0
        else:
            vp_positions[i] = 0.0

        if i > 0 and vp_positions[i] != vp_positions[i-1]:
            vp_trades += 1

        # Baseline MACD (standard)
        if baseline_hist[i] > 0:
            baseline_positions[i] = 1.0
        elif baseline_hist[i] < 0:
            baseline_positions[i] = -1.0
        else:
            baseline_positions[i] = 0.0

        if i > 0 and baseline_positions[i] != baseline_positions[i-1]:
            baseline_trades += 1

    # Compute strategy returns (for periods where position is available)
    start_idx = slow + signal_period
    vp_strat_returns = returns[start_idx:] * vp_positions[start_idx:-1]
    base_strat_returns = returns[start_idx:] * baseline_positions[start_idx:-1]

    # Compute metrics
    def _compute_metrics(strat_ret):
        if len(strat_ret) < 10:
            return {"sharpe": 0, "total_return": 0, "max_dd": 0}
        total_ret = float(np.prod(1 + strat_ret) - 1)
        ann_ret = float(np.mean(strat_ret) * 252)
        ann_vol = float(np.std(strat_ret) * np.sqrt(252))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = np.cumprod(1 + strat_ret)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max
        max_dd = float(np.min(dd))
        win_rate = float(np.mean(strat_ret > 0))
        return {
            "total_return_pct": round(total_ret * 100, 2),
            "annualized_return_pct": round(ann_ret * 100, 2),
            "annualized_vol_pct": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "win_rate": round(win_rate, 4),
            "num_trades": 0,  # filled below
        }

    vp_metrics = _compute_metrics(vp_strat_returns)
    base_metrics = _compute_metrics(base_strat_returns)
    vp_metrics["num_trades"] = vp_trades
    base_metrics["num_trades"] = baseline_trades

    # Regime-specific performance
    regime_perf = {}
    for regime in ["low", "normal", "elevated", "high"]:
        regime_indices = [
            i for i in range(start_idx, len(prices) - 1)
            if vol_regimes[i] == regime
        ]
        if regime_indices:
            regime_ret = returns[regime_indices]
            vp_reg_ret = returns[regime_indices] * vp_positions[regime_indices]
            base_reg_ret = returns[regime_indices] * baseline_positions[regime_indices]

            regime_perf[regime] = {
                "periods": len(regime_indices),
                "vp_sharpe": round(
                    (np.mean(vp_reg_ret) * 252) / (np.std(vp_reg_ret) * np.sqrt(252))
                    if np.std(vp_reg_ret) > 0 else 0, 4
                ),
                "baseline_sharpe": round(
                    (np.mean(base_reg_ret) * 252) / (np.std(base_reg_ret) * np.sqrt(252))
                    if np.std(base_reg_ret) > 0 else 0, 4
                ),
            }

    # Correlation between strategies
    common = min(len(vp_strat_returns), len(base_strat_returns))
    if common > 10 and np.std(vp_strat_returns[:common]) > 0 and np.std(base_strat_returns[:common]) > 0:
        correlation = float(np.corrcoef(vp_strat_returns[:common], base_strat_returns[:common])[0, 1])
    else:
        correlation = 0.0

    return {
        "ticker": ticker,
        "period": f"{dates[start_idx]} to {dates[-1]}",
        "vp_macd": vp_metrics,
        "baseline_macd": base_metrics,
        "improvement": {
            "sharpe_delta": round(vp_metrics["sharpe"] - base_metrics["sharpe"], 4),
            "return_delta_pct": round(vp_metrics["total_return_pct"] - base_metrics["total_return_pct"], 2),
            "max_dd_delta_pct": round(vp_metrics["max_drawdown_pct"] - base_metrics["max_drawdown_pct"], 2),
        },
        "correlation": round(correlation, 4),
        "volume_available": volume_available,
        "parameters": {
            "fast": fast,
            "slow": slow,
            "signal": signal_period,
            "vol_mult": vol_mult,
        },
        "regime_performance": regime_perf,
    }


def _save_state(signal: VPMACDSignal) -> None:
    """Save signal state to JSON."""
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(asdict(signal), f, indent=2, default=str)


def _load_state() -> Optional[Dict]:
    """Load saved signal state."""
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return None


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="VP-MACD Signal Generator")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Signal command
    signal_parser = subparsers.add_parser("signal", help="Generate current signal")
    signal_parser.add_argument("--ticker", default="SPY", help="Ticker symbol")
    signal_parser.add_argument("--fast", type=int, default=DEFAULT_FAST)
    signal_parser.add_argument("--slow", type=int, default=DEFAULT_SLOW)
    signal_parser.add_argument("--signal", type=int, default=DEFAULT_SIGNAL)
    signal_parser.add_argument("--vol-mult", type=float, default=DEFAULT_VOL_MULT)

    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Run historical backtest")
    bt_parser.add_argument("--ticker", default="SPY", help="Ticker symbol")
    bt_parser.add_argument("--fast", type=int, default=DEFAULT_FAST)
    bt_parser.add_argument("--slow", type=int, default=DEFAULT_SLOW)
    bt_parser.add_argument("--signal", type=int, default=DEFAULT_SIGNAL)
    bt_parser.add_argument("--vol-mult", type=float, default=DEFAULT_VOL_MULT)

    # Status command
    subparsers.add_parser("status", help="Show saved signal state")

    args = parser.parse_args()

    if args.command == "signal":
        signal = generate_signal(
            ticker=args.ticker,
            fast=args.fast,
            slow=args.slow,
            signal_period=args.signal,
            vol_mult=args.vol_mult,
        )
        if signal:
            _save_state(signal)
            print(json.dumps(asdict(signal), indent=2, default=str))
            logger.info(
                f"VP-MACD({args.ticker}): {signal.vp_macd_signal} "
                f"(value={signal.vp_macd_value}, confidence={signal.confidence}, "
                f"vol_regime={signal.regime})"
            )
        else:
            logger.error("Failed to generate signal")
            sys.exit(1)

    elif args.command == "backtest":
        result = backtest(
            ticker=args.ticker,
            fast=args.fast,
            slow=args.slow,
            signal_period=args.signal,
            vol_mult=args.vol_mult,
        )
        print(json.dumps(result, indent=2, default=str))

        if "error" not in result:
            vp = result["vp_macd"]
            base = result["baseline_macd"]
            impr = result["improvement"]
            logger.info(
                f"VP-MACD Backtest ({args.ticker}): "
                f"Sharpe {vp['sharpe']} vs Baseline {base['sharpe']} "
                f"(Δ={impr['sharpe_delta']:+}), "
                f"Return {vp['total_return_pct']}% vs {base['total_return_pct']}%"
            )

    elif args.command == "status":
        state = _load_state()
        if state:
            print(json.dumps(state, indent=2, default=str))
        else:
            print("No saved state found. Run `vp_macd signal` first.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
