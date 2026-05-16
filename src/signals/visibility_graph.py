#!/usr/bin/env python3
"""
v5.41 — Visibility Graph Signal (VGRSI)

Network-science-based technical indicator from arXiv 2605.01300.

Converts price time series into a natural visibility graph and computes
backward visibility count to generate relative-strength signals.

Features:
- Natural visibility graph construction (monotonic stack, O(n))
- VGRSI: normalized backward visibility count (0-100)
- Overbought/oversold thresholds with trend confirmation
- Integration with ensemble voter via SignalSource.VISIBILITY_GRAPH

Usage:
    python -m src.signals.visibility_graph signal        # Generate current signal
    python -m src.signals.visibility_graph backtest      # Historical backtest
    python -m src.signals.visibility_graph explain       # Explain current signal

Reference:
    Longo, L. et al. (2026). "Visibility graphs can make money in financial markets."
    arXiv:2605.01300 [q-fin.ST]
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PRICES_PATH = PROJECT_ROOT / "public/data/prices.json"
STATE_PATH = DATA_DIR / "signals/visibility_graph_signal.json"

# Constants
LOOKBACK_DAYS = 90               # Rolling computation window
VGRSI_OVERSOLD = 30              # Buy signal threshold
VGRSI_OVERBOUGHT = 70            # Sell signal threshold
VGRSI_NEUTRAL = 50               # Neutral center

# Signal strength mapping
SIGNAL_STRENGTH_MAP = {
    "strong_buy": 1.0,
    "moderate_buy": 0.5,
    "neutral": 0.0,
    "moderate_sell": -0.5,
    "strong_sell": -1.0,
}

# Trend confirmation: require price above/below MA
TREND_CONFIRMATION_MA = 50       # Days for moving average confirmation

# Ensemble integration weight
ENSEMBLE_WEIGHT = 0.03           # 3% weight in ensemble voter


@dataclass
class VisibilityGraphSignal:
    """VGRSI signal output."""
    symbol: str
    timestamp: str

    # VGRSI values
    vgrsi: float = 50.0          # 0-100 scale
    backward_visibility: int = 0  # Raw backward visibility count
    max_possible_vis: int = 0    # Maximum possible visibility in window

    # Signal
    signal_strength: float = 0.0  # -1 to +1
    signal_label: str = "neutral"

    # Trend confirmation
    price_vs_ma: float = 0.0     # % deviation from moving average
    trend_confirmed: bool = False

    # Metadata
    n_visible_peaks: int = 0
    n_visible_troughs: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


def load_price_data(symbol: str = "SPY") -> Optional[np.ndarray]:
    """Load price data from prices.json."""
    try:
        if not PRICES_PATH.exists():
            logger.warning(f"Prices file not found: {PRICES_PATH}")
            return None

        with open(PRICES_PATH) as f:
            data = json.load(f)

        if symbol not in data:
            logger.warning(f"Symbol {symbol} not found in price data")
            return None

        raw = data[symbol]
        # Extract prices (handle both {d,p} and simple list formats)
        if isinstance(raw, dict) and "p" in raw:
            prices = raw["p"]
        elif isinstance(raw, list):
            prices = raw
        elif isinstance(raw, dict):
            # Try to find price-like keys
            for key in ("close", "price", "adjclose", "c"):
                if key in raw:
                    prices = raw[key]
                    break
            else:
                # Assume dict values are prices
                prices = list(raw.values())
        else:
            logger.warning(f"Unknown price format for {symbol}")
            return None

        arr = np.array(prices, dtype=np.float64)
        if len(arr) == 0:
            logger.warning(f"Empty price series for {symbol}")
            return None

        logger.info(f"Loaded {len(arr)} price points for {symbol}")
        return arr

    except Exception as e:
        logger.error(f"Error loading price data: {e}")
        return None


def compute_visibility_graph(prices: np.ndarray) -> np.ndarray:
    """
    Compute natural visibility graph using monotonic stack (O(n)).

    The natural visibility graph connects two points (ta, ya) and (tb, yb)
    if every intermediate point (tc, yc) satisfies:
        yc < ya + (yb - ya) * (tc - ta) / (tb - ta)

    Uses a monotonic stack to compute backward visibility counts efficiently.

    Returns:
        Array of backward visibility counts for each point.
    """
    n = len(prices)
    if n < 3:
        return np.ones(n, dtype=np.int32)

    vis_counts = np.ones(n, dtype=np.int32)  # At least visible to self

    # We use a monotonic stack that maintains decreasing slopes
    # For each point i, we check visibility with previous points
    # A point j is visible from i if the slope from j to i is greater
    # than all slopes between j and any k where j < k < i

    # This is O(n²) worst-case but O(n) amortized for typical price series
    # For the natural visibility graph, we check each pair (j, i) for j < i
    # using the angle condition

    # Optimized approach: maintain stack of "candidate" visible points
    # A point j is a candidate for visibility from i if it's a local maximum
    # relative to intermediate points

    # For each i, compute visibility to all j < i
    for i in range(2, n):
        # Check visibility with all previous points
        # Optimization: track the maximum slope seen so far
        max_slope = -np.inf
        xi, yi = float(i), float(prices[i])

        for j in range(i - 1, -1, -1):
            xj, yj = float(j), float(prices[j])
            slope = (yi - yj) / (xi - xj)

            if slope > max_slope:
                # Point j is visible from i
                max_slope = slope
                vis_counts[i] += 1
            # If slope <= max_slope, point j is not visible from i
            # (hidden behind a previous point with higher slope)

    return vis_counts


def compute_visibility_graph_optimized(prices: np.ndarray) -> np.ndarray:
    """
    Optimized visibility graph using monotonic angular stack.

    O(n) algorithm: maintains a stack of points sorted by slope angle.
    When processing a new point, pops points from stack that are no longer
    visible (their angle is less than the new point's angle).
    """
    n = len(prices)
    if n < 3:
        return np.ones(n, dtype=np.int32)

    vis_counts = np.ones(n, dtype=np.int32)

    # Stack stores indices of points that are candidates for visibility
    # Key insight: for the natural visibility graph, we only need to track
    # points where the cumulative maximum slope changes
    stack = [0]  # First point is always visible

    for i in range(1, n):
        xi, yi = float(i), float(prices[i])

        # Count visible points: the stack size + 1 (current point)
        # The stack contains all points visible from previous points
        # The current point is visible to itself
        vis_counts[i] = len(stack) + 1

        # Update stack: remove points that will be hidden by this point
        # A point is hidden if the slope from it to the new point is
        # greater than the slope from the previous candidate
        while len(stack) >= 2:
            j = stack[-1]       # Last visible point
            k = stack[-2]       # Before that

            xj, yj = float(j), float(prices[j])
            xk, yk = float(k), float(prices[k])

            # Check if point j is hidden by the line from k to i
            # This means: yj lies below the line connecting (xk, yk) and (xi, yi)
            # yj < yk + (yi - yk) * (xj - xk) / (xi - xk)
            line_val = yk + (yi - yk) * (xj - xk) / (xi - xk)

            if yj <= line_val:
                # j is hidden (or collinear), remove it
                stack.pop()
            else:
                break

        stack.append(i)

    return vis_counts


def compute_vgrsi(
    prices: np.ndarray,
    lookback: int = LOOKBACK_DAYS
) -> Tuple[float, int, int, int, int]:
    """
    Compute VGRSI (Visibility Graph RSI) from price series.

    Returns:
        Tuple of (vgrsi, backward_visibility, max_possible_vis,
                  n_visible_peaks, n_visible_troughs)
    """
    if len(prices) < 3:
        return (50.0, 0, 0, 0, 0)

    # Use only the lookback window
    window = prices[-min(lookback, len(prices)):]

    # Compute visibility graph
    try:
        # Try optimized algorithm first
        vis_counts = compute_visibility_graph_optimized(window)
    except Exception:
        # Fallback to standard algorithm
        vis_counts = compute_visibility_graph(window)

    # Backward visibility of the most recent point
    backward_visibility = int(vis_counts[-1])

    # Maximum possible visibility: number of points in window minus 1
    max_possible_vis = len(window) - 1

    # Normalize to 0-100 scale (VGRSI)
    if max_possible_vis > 0:
        vgrsi = min(100.0, max(0.0, 100.0 * backward_visibility / max_possible_vis))
    else:
        vgrsi = 50.0

    # Count visible peaks and troughs for the latest point
    n_visible_peaks = 0
    n_visible_troughs = 0
    latest_price = window[-1]

    # Re-analyze: which visible points are peaks vs troughs
    for idx in range(len(window) - 1):
        # Check if this point is visible from the latest point
        # (we can use the same visibility test)
        j = idx
        i = len(window) - 1
        xj, yj = float(j), float(window[j])
        xi, yi = float(i), float(window[i])

        # Check all points between j and i
        visible = True
        max_slope_encountered = -np.inf
        for k in range(j + 1, i):
            xk, yk = float(k), float(window[k])
            slope = (yi - yk) / (xi - xk) if abs(xi - xk) > 1e-10 else 0
            if slope <= max_slope_encountered:
                visible = False
                break
            max_slope_encountered = slope

        if visible:
            local_max = True
            local_min = True
            # Check if it's a peak or trough (neighbors comparison)
            if idx > 0:
                if window[idx] <= window[idx - 1]:
                    local_max = False
            if idx < len(window) - 2:
                if window[idx] <= window[idx + 1]:
                    local_max = False
            if idx > 0:
                if window[idx] >= window[idx - 1]:
                    local_min = False
            if idx < len(window) - 2:
                if window[idx] >= window[idx + 1]:
                    local_min = False

            if local_max:
                n_visible_peaks += 1
            if local_min:
                n_visible_troughs += 1

    return (vgrsi, backward_visibility, max_possible_vis,
            n_visible_peaks, n_visible_troughs)


def classify_signal(
    vgrsi: float,
    price: float,
    prices: np.ndarray,
    ma_period: int = TREND_CONFIRMATION_MA
) -> Tuple[float, str, float, bool]:
    """
    Classify VGRSI into trading signal with trend confirmation.

    Returns:
        Tuple of (signal_strength, signal_label, price_vs_ma, trend_confirmed)
    """
    # Compute moving average
    if len(prices) >= ma_period:
        ma = float(np.mean(prices[-ma_period:]))
        price_vs_ma = (price - ma) / ma * 100  # % deviation
        trend_confirmed = price > ma  # Uptrend confirmation
    else:
        ma = float(np.mean(prices))
        price_vs_ma = 0.0
        trend_confirmed = True  # No rejection without data

    # Determine signal
    if vgrsi <= VGRSI_OVERSOLD:
        # Oversold: buy signal (but require uptrend confirmation for safety)
        if trend_confirmed:
            # Stronger buy when oversold but in uptrend
            depth = (VGRSI_OVERSOLD - vgrsi) / VGRSI_OVERSOLD  # 0 to 1
            signal_strength = 0.5 + 0.5 * min(depth, 1.0)
            signal_label = "strong_buy" if signal_strength > 0.75 else "moderate_buy"
        else:
            # In downtrend: weaker buy (cautious dip-buy)
            signal_strength = 0.3
            signal_label = "moderate_buy"

    elif vgrsi >= VGRSI_OVERBOUGHT:
        # Overbought: sell signal
        if not trend_confirmed:
            # Stronger sell when overbought and below MA (exhaustion)
            excess = (vgrsi - VGRSI_OVERBOUGHT) / (100 - VGRSI_OVERBOUGHT)
            signal_strength = -0.5 - 0.5 * min(excess, 1.0)
            signal_label = "strong_sell" if signal_strength < -0.75 else "moderate_sell"
        else:
            # In uptrend: weaker sell (pullback risk but trend intact)
            signal_strength = -0.3
            signal_label = "moderate_sell"

    else:
        # Neutral zone
        signal_strength = 0.0
        signal_label = "neutral"

    return (signal_strength, signal_label, price_vs_ma, trend_confirmed)


def generate_signal(symbol: str = "SPY") -> Optional[VisibilityGraphSignal]:
    """Generate complete VGRSI signal for a symbol."""
    prices = load_price_data(symbol)
    if prices is None or len(prices) < 10:
        logger.warning(f"Insufficient price data for {symbol}")
        return None

    # Compute VGRSI
    vgrsi, bv, maxv, peaks, troughs = compute_vgrsi(prices)

    # Classify signal
    current_price = float(prices[-1])
    signal_strength, signal_label, price_vs_ma, trend_confirmed = classify_signal(
        vgrsi, current_price, prices
    )

    result = VisibilityGraphSignal(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc).isoformat(),
        vgrsi=round(vgrsi, 2),
        backward_visibility=bv,
        max_possible_vis=maxv,
        signal_strength=round(signal_strength, 4),
        signal_label=signal_label,
        price_vs_ma=round(price_vs_ma, 2),
        trend_confirmed=trend_confirmed,
        n_visible_peaks=peaks,
        n_visible_troughs=troughs,
    )

    return result


def get_ensemble_signal(symbol: str = "SPY") -> Optional[dict]:
    """Generate signal in ensemble voter format."""
    signal = generate_signal(symbol)
    if signal is None:
        return None

    return {
        "signal_value": signal.signal_strength,
        "confidence": min(1.0, abs(signal.vgrsi - 50) / 50),
        "vgrsi": signal.vgrsi,
        "signal_label": signal.signal_label,
        "trend_confirmed": signal.trend_confirmed,
        "price_vs_ma": signal.price_vs_ma,
        "backward_visibility": signal.backward_visibility,
        "n_visible_peaks": signal.n_visible_peaks,
        "n_visible_troughs": signal.n_visible_troughs,
        "weight": ENSEMBLE_WEIGHT,
        "rationale": f"VGRSI={signal.vgrsi:.1f} ({signal.signal_label}), "
                     f"vis={signal.backward_visibility}/{signal.max_possible_vis}, "
                     f"peaks={signal.n_visible_peaks}, "
                     f"troughs={signal.n_visible_troughs}, "
                     f"MA_dev={signal.price_vs_ma:+.1f}%"
    }


def run_backtest(symbol: str = "SPY", save: bool = False) -> dict:
    """
    Run historical backtest of VGRSI signal vs buy-and-hold.

    Generates signals for each available day and computes performance.
    """
    prices = load_price_data(symbol)
    if prices is None or len(prices) < LOOKBACK_DAYS + 10:
        return {"error": "Insufficient data for backtest"}

    n = len(prices)
    signals = []
    daily_returns = []
    vg_returns = []

    for i in range(LOOKBACK_DAYS, n):
        window = prices[:i + 1]
        vgrsi, bv, maxv, peaks, troughs = compute_vgrsi(window)

        current_price = float(prices[i])
        prev_price = float(prices[i - 1]) if i > 0 else current_price

        signal_strength, signal_label, _, trend_confirmed = classify_signal(
            vgrsi, current_price, window
        )

        daily_return = (current_price / prev_price - 1) * 100

        if signal_label in ("strong_buy", "moderate_buy"):
            position = 1.0
        elif signal_label in ("strong_sell", "moderate_sell"):
            position = 0.0  # Flat (can't short in basic version)
        else:
            position = 0.5  # Neutral = 50% exposure

        vg_return = daily_return * position
        signals.append(position)
        daily_returns.append(daily_return)
        vg_returns.append(vg_return)

    if len(daily_returns) < 10:
        return {"error": "Backtest produced insufficient data"}

    # Compute metrics
    bh_returns = np.array(daily_returns)
    vg_returns = np.array(vg_returns)

    bh_cumulative = float(np.cumprod(1 + bh_returns / 100)[-1] - 1) * 100
    vg_cumulative = float(np.cumprod(1 + vg_returns / 100)[-1] - 1) * 100

    bh_sharpe = float(np.mean(bh_returns) / np.std(bh_returns) * np.sqrt(252)) if np.std(bh_returns) > 0 else 0
    vg_sharpe = float(np.mean(vg_returns) / np.std(vg_returns) * np.sqrt(252)) if np.std(vg_returns) > 0 else 0

    bh_max_dd = compute_max_drawdown(bh_returns)
    vg_max_dd = compute_max_drawdown(vg_returns)

    # Signal statistics
    n_buy = sum(1 for s in signals if s > 0.5)
    n_sell = sum(1 for s in signals if s < 0.5)
    n_neutral = sum(1 for s in signals if s == 0.5)

    result = {
        "symbol": symbol,
        "period": f"{LOOKBACK_DAYS} days to {len(daily_returns)} trading days",
        "n_signals": len(signals),
        "buy_and_hold": {
            "cumulative_return_pct": round(bh_cumulative, 2),
            "annualized_return_pct": round(bh_cumulative / len(daily_returns) * 252, 2),
            "sharpe": round(bh_sharpe, 4),
            "max_drawdown_pct": round(bh_max_dd, 2),
        },
        "vgrsi_strategy": {
            "cumulative_return_pct": round(vg_cumulative, 2),
            "annualized_return_pct": round(vg_cumulative / len(daily_returns) * 252, 2),
            "sharpe": round(vg_sharpe, 4),
            "max_drawdown_pct": round(vg_max_dd, 2),
        },
        "signal_breakdown": {
            "buy_signals": n_buy,
            "sell_signals": n_sell,
            "neutral_signals": n_neutral,
        },
    }

    if save:
        backtest_path = DATA_DIR / "backtests/visibility_graph_backtest.json"
        backtest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(backtest_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Backtest saved to {backtest_path}")

    return result


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown from percentage return series."""
    cumulative = np.cumprod(1 + returns / 100)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max * 100
    return float(np.min(drawdowns))


def format_signal_table(signals: List[dict]) -> str:
    """Format multiple signal results as a table."""
    if not signals:
        return "No signals to display."

    header = f"{'Symbol':<8} {'VGRSI':<8} {'Signal':<14} {'Strength':<10} {'Vis':<6} {'Peaks':<6} {'Troughs':<6} {'MA Dev':<8} {'Trend':<7}"
    sep = "-" * len(header)
    rows = [header, sep]

    for s in signals:
        rows.append(
            f"{s.get('symbol', 'N/A'):<8} "
            f"{s.get('vgrsi', 0):<8.1f} "
            f"{s.get('signal_label', 'N/A'):<14} "
            f"{s.get('signal_strength', 0):<+10.3f} "
            f"{s.get('backward_visibility', 0):<6} "
            f"{s.get('n_visible_peaks', 0):<6} "
            f"{s.get('n_visible_troughs', 0):<6} "
            f"{s.get('price_vs_ma', 0):<+8.1f} "
            f"{'YES' if s.get('trend_confirmed') else 'NO':<7}"
        )

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio-Lab v5.41: Visibility Graph Signal (VGRSI)"
    )
    parser.add_argument(
        "command",
        choices=["signal", "backtest", "explain", "ensemble"],
        help="Command to execute"
    )
    parser.add_argument(
        "--symbol", "-s",
        default="SPY",
        help="Symbol to analyze (default: SPY)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to file"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "GLD", "TLT", "QQQ", "IEF"],
        help="Symbols for multi-symbol commands"
    )

    args = parser.parse_args()

    if args.command == "signal":
        result = generate_signal(args.symbol)
        if result:
            print(f"\n=== VGRSI Signal: {args.symbol} ===")
            print(f"VGRSI: {result.vgrsi:.1f}/100")
            print(f"Signal: {result.signal_label} (strength: {result.signal_strength:+.3f})")
            print(f"Backward Visibility: {result.backward_visibility}/{result.max_possible_vis}")
            print(f"Visible Peaks/Troughs: {result.n_visible_peaks}/{result.n_visible_troughs}")
            print(f"Price vs MA({TREND_CONFIRMATION_MA}d): {result.price_vs_ma:+.1f}%")
            print(f"Trend Confirmed: {'YES' if result.trend_confirmed else 'NO'}")

            if args.save:
                STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(STATE_PATH, "w") as f:
                    f.write(result.to_json())
                logger.info(f"Signal saved to {STATE_PATH}")
        else:
            print(f"Failed to generate signal for {args.symbol}")

    elif args.command == "backtest":
        result = run_backtest(args.symbol, save=args.save)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\n=== VGRSI Backtest: {result['symbol']} ===")
            print(f"Period: {result['period']}")
            print(f"Total signals: {result['n_signals']}")
            print(f"\n  {'Metric':<30} {'Buy & Hold':<16} {'VGRSI':<16}")
            print(f"  {'-'*30} {'-'*16} {'-'*16}")
            bh = result['buy_and_hold']
            vg = result['vgrsi_strategy']
            print(f"  {'Cumulative Return':<30} {bh['cumulative_return_pct']:<+15.2f}% {vg['cumulative_return_pct']:<+15.2f}%")
            print(f"  {'Ann. Return':<30} {bh['annualized_return_pct']:<+15.2f}% {vg['annualized_return_pct']:<+15.2f}%")
            print(f"  {'Sharpe Ratio':<30} {bh['sharpe']:<16.4f} {vg['sharpe']:<16.4f}")
            print(f"  {'Max Drawdown':<30} {bh['max_drawdown_pct']:<15.2f}% {vg['max_drawdown_pct']:<15.2f}%")
            print(f"\nSignal breakdown: {result['signal_breakdown']}")

    elif args.command == "explain":
        result = generate_signal(args.symbol)
        if result:
            print(f"\n=== VGRSI Explanation: {args.symbol} ===")
            print(f"\nWhat is VGRSI?")
            print(f"  VGRSI (Visibility Graph Relative Strength Index) uses network")
            print(f"  science to analyze price structure. Instead of comparing price")
            print(f"  changes like RSI, it measures how many historical price points")
            print(f"  are 'visible' from the current position.")
            print(f"\nCurrent State:")
            print(f"  VGRSI = {result.vgrsi:.1f} (range: 0-100)")
            print(f"  Oversold threshold: {VGRSI_OVERSOLD} (buy zone)")
            print(f"  Overbought threshold: {VGRSI_OVERBOUGHT} (sell zone)")
            print(f"  Interpretation: {'OVERSOLD - Buy signal' if result.vgrsi <= VGRSI_OVERSOLD else 'OVERBOUGHT - Sell signal' if result.vgrsi >= VGRSI_OVERBOUGHT else 'NEUTRAL - No clear signal'}")
            print(f"\nVisibility Analysis:")
            print(f"  Out of {result.max_possible_vis} historical points,")
            print(f"  {result.backward_visibility} are visible from current price.")
            print(f"  Visible peaks: {result.n_visible_peaks}")
            print(f"  Visible troughs: {result.n_visible_troughs}")
            print(f"  High visibility = more structure visible = mature trend")
            print(f"  Low visibility = less structure visible = trend transition")
            print(f"\nTrend Confirmation:")
            print(f"  Price vs {TREND_CONFIRMATION_MA}-day MA: {result.price_vs_ma:+.1f}%")
            print(f"  Confirmed: {'YES' if result.trend_confirmed else 'NO'}")
        else:
            print(f"Failed to generate signal for {args.symbol}")

    elif args.command == "ensemble":
        signals = []
        for sym in args.symbols:
            sig = get_ensemble_signal(sym)
            if sig:
                signals.append({
                    "symbol": sym,
                    "vgrsi": sig["vgrsi"],
                    "signal_strength": sig["signal_value"],
                    "signal_label": sig["signal_label"],
                    "backward_visibility": sig["backward_visibility"],
                    "n_visible_peaks": sig["n_visible_peaks"],
                    "n_visible_troughs": sig["n_visible_troughs"],
                    "price_vs_ma": sig["price_vs_ma"],
                    "trend_confirmed": sig["trend_confirmed"],
                })

        print(f"\n=== VGRSI Multi-Symbol Signals ===\n")
        print(format_signal_table(signals))

        if args.save:
            ensemble_path = DATA_DIR / "signals/visibility_graph_ensemble.json"
            ensemble_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ensemble_path, "w") as f:
                json.dump(signals, f, indent=2, default=str)
            logger.info(f"Ensemble signals saved to {ensemble_path}")


if __name__ == "__main__":
    main()
