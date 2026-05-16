"""
Volatility-Volume-Gap Day Classifier - v5.30 Implementation

Lightweight day classifier using observable price features to predict the
trading day regime. Inspired by arXiv 2605.11423 ("Validated Volatility-Volume-Gap
Classifier for Regime Identification in MNQ").

Adapted for close-only daily price data (the project's primary data format).
Features:
1. Daily Return   : (close[t] - close[t-1]) / close[t-1]  (gap proxy)
2. Volume Anomaly : always 1.0 (volume data not available in close-only feeds)
3. Return Vol     : abs(daily return) / rolling 20-day mean abs(return) (vol proxy)

Dual-mode: If OHLCV data is available (5+ columns), uses full features.
           If close-only data (1 column), uses adapted daily-return features.

Classification yields one of five regimes:
- TREND_UP     : positive daily return, moderate vol
- TREND_DOWN   : negative daily return, moderate vol
- MEAN_REVERT  : small return (near zero), low vol
- HIGH_VOL     : large abs(return) + elevated relative vol
- CRISIS       : extreme return + extreme vol

No ML dependencies. Pure numpy implementation.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
STATE_FILE = DATA_DIR / "vol_volume_gap_state.json"


class DayRegime(str, Enum):
    """Classification of a single trading day."""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    MEAN_REVERT = "mean_revert"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


@dataclass
class DayFeatures:
    """Three-feature representation for a single day."""
    daily_return: float         # (C_t - C_{t-1}) / C_{t-1}
    volume_anomaly: float       # V / V_avg (always 1.0 for close-only)
    return_vol_ratio: float     # |return| / rolling_avg(|return|)
    regime: DayRegime = DayRegime.UNKNOWN
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "daily_return": round(self.daily_return, 6),
            "volume_anomaly": round(self.volume_anomaly, 4),
            "return_vol_ratio": round(self.return_vol_ratio, 4),
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class ClassifierConfig:
    """Configurable thresholds for regime classification.

    Thresholds calibrated for SPY daily returns.
    """
    # Daily return thresholds (fractional)
    ret_extreme: float = 0.04       # |return| > 4% = extreme (SPY, ~2 sigma)
    ret_large: float = 0.015        # |return| > 1.5% = large
    ret_small: float = 0.004        # |return| < 0.4% = small

    # Relative volatility thresholds
    rel_vol_extreme: float = 3.0    # > 3x avg abs return = extreme
    rel_vol_elevated: float = 2.0   # > 2x avg abs return = elevated

    # Fixed vol thresholds
    vol_lookback: int = 20


def compute_features(
    prices: np.ndarray,
    config: Optional[ClassifierConfig] = None,
) -> Optional[DayFeatures]:
    """Compute features from price history.

    Args:
        prices: nxK array where K >= 1 (close prices or OHLCV).
                Last row is today's data.
                - If K >= 4: uses [open, high, low, close]
                - If K == 1: uses close prices

    Returns:
        DayFeatures or None if insufficient data.
    """
    if config is None:
        config = ClassifierConfig()

    n = len(prices)
    if n < config.vol_lookback + 2:
        logger.warning(
            f"Insufficient data: need {config.vol_lookback + 2} days, got {n}"
        )
        return None

    # Extract close prices (last column if OHLCV, first/last column if close-only)
    if prices.shape[1] >= 4:
        closes = prices[:, 3]
        today_close = closes[-1]
        prev_close = closes[-2]

        # OHLCV mode: use overnight gap and opening range
        today_open = prices[-1, 0]
        today_high = prices[-1, 1]
        overnight_gap = (today_open - prev_close) / prev_close if prev_close != 0 else 0.0
        opening_half_range = (today_high - today_open) / today_open if today_open != 0 else 0.0

        # For close-only compat, use daily return as primary feature
        daily_return = (today_close - prev_close) / prev_close if prev_close != 0 else 0.0

        # Use overnight_gap as primary (more informative) but keep daily return
        primary_return = overnight_gap
        # Return vol ratio computed from daily returns
        all_returns = np.diff(closes) / closes[:-1]
        all_abs_returns = np.abs(all_returns)
    else:
        # Close-only mode: prices[:, 0] is the close
        closes = prices[:, 0]
        today_close = closes[-1]
        prev_close = closes[-2]

        daily_return = (today_close - prev_close) / prev_close if prev_close != 0 else 0.0
        primary_return = daily_return

        all_returns = np.diff(closes) / closes[:-1]
        all_abs_returns = np.abs(all_returns)

    # 2. Volume Anomaly (always 1.0 for close-only data)
    volume_anomaly = 1.0

    # 3. Return Volatility Ratio: |today_return| / avg(|return|)
    lookback_returns = np.abs(all_returns[-(config.vol_lookback + 1):-1])  # excl today
    avg_abs_return = np.mean(lookback_returns) if len(lookback_returns) > 0 else 0.001
    if avg_abs_return < 0.0001:
        avg_abs_return = 0.0001
    return_vol_ratio = abs(primary_return) / avg_abs_return

    return DayFeatures(
        daily_return=float(daily_return),
        volume_anomaly=float(volume_anomaly),
        return_vol_ratio=float(return_vol_ratio),
    )


def classify_day(
    features: DayFeatures,
    config: Optional[ClassifierConfig] = None,
) -> DayFeatures:
    """Classify a trading day regime from its three features.

    Rule-based classifier using configurable thresholds.
    """
    if config is None:
        config = ClassifierConfig()

    gap = abs(features.daily_return)
    gap_sign = np.sign(features.daily_return)
    return_vol = features.return_vol_ratio

    # Crisis detection: extreme return + extreme rel vol
    if gap >= config.ret_extreme and return_vol >= config.rel_vol_extreme:
        features.regime = DayRegime.CRISIS
        features.confidence = 0.90
        return features

    # High volatility: large return + elevated vol, or very elevated vol
    if (gap >= config.ret_large and return_vol >= config.rel_vol_elevated) or return_vol >= config.rel_vol_extreme:
        features.regime = DayRegime.HIGH_VOL
        features.confidence = 0.80
        return features

    # Trend up: positive return, at least small, moderate vol
    if gap_sign > 0 and gap >= config.ret_small and return_vol < config.rel_vol_elevated:
        features.regime = DayRegime.TREND_UP
        features.confidence = 0.70
        return features

    # Trend down: negative return, at least small, moderate vol
    if gap_sign < 0 and gap >= config.ret_small and return_vol < config.rel_vol_elevated:
        features.regime = DayRegime.TREND_DOWN
        features.confidence = 0.70
        return features

    # Mean revert: very small return
    if gap < config.ret_small:
        features.regime = DayRegime.MEAN_REVERT
        features.confidence = 0.60
        return features

    # Fallback: use return sign
    if gap_sign > 0:
        features.regime = DayRegime.TREND_UP
        features.confidence = 0.50
    else:
        features.regime = DayRegime.TREND_DOWN
        features.confidence = 0.50

    return features


def load_prices(symbol: str = "SPY") -> Optional[np.ndarray]:
    """Load close-only price data from the project's price JSON.

    Format: {"SPY": [{"d": "2021-05-10", "p": 390.34}, ...], ...}
    Returns nx1 numpy array of close prices, earliest to latest.
    """
    # Try multiple possible locations
    candidates = [
        DATA_DIR / "prices.json",
        DATA_DIR / ".." / "public" / "data" / "prices.json",
        Path("~/projects/portfolio-lab/data/prices.json").expanduser(),
        Path("~/projects/portfolio-lab/public/data/prices.json").expanduser(),
        Path("data/prices.json"),
        Path("public/data/prices.json"),
    ]
    price_file = None
    for p in candidates:
        if p.exists():
            price_file = p
            break

    if price_file is None:
        logger.error("Price data not found")
        return None

    try:
        with open(price_file) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load price data: {e}")
        return None

    # Find symbol data — format: {symbol: [{"d": ..., "p": ...}, ...]}
    if isinstance(raw, dict):
        symbol_data = raw.get(symbol)
    elif isinstance(raw, list):
        symbol_data = None
        for entry in raw:
            if isinstance(entry, dict) and entry.get("s") == symbol:
                symbol_data = entry
                break
    else:
        symbol_data = None

    if symbol_data is None:
        logger.error(f"Symbol {symbol} not found in price data")
        return None

    # Extract price points
    if isinstance(symbol_data, list):
        # Format: [{"d": "...", "p": 123.45}, ...]
        try:
            closes = np.array([item.get("p", item.get("close", 0)) for item in symbol_data], dtype=np.float64)
            if len(closes) == 0:
                logger.error(f"No price data for {symbol}")
                return None
            # Ensure chronological order (earliest first)
            return closes.reshape(-1, 1)
        except (IndexError, TypeError, ValueError) as e:
            logger.error(f"Error extracting {symbol} prices: {e}")
            return None
    elif isinstance(symbol_data, dict):
        # Alternative format: {"t": [...], "c": [...]} or {"o": [...], "h": [...], ...}
        closes_list = symbol_data.get("c") or symbol_data.get("closes") or symbol_data.get("p", [])
        if closes_list:
            arr = np.array(closes_list, dtype=np.float64)
            return arr.reshape(-1, 1)
        logger.error(f"Could not parse {symbol} data format")
        return None
    else:
        logger.error(f"Unexpected data type for {symbol}: {type(symbol_data).__name__}")
        return None


def detect_regime(
    symbol: str = "SPY",
    config: Optional[ClassifierConfig] = None,
) -> Dict:
    """Main entry point: load data, compute features, classify day.

    Returns dict with regime, features, confidence, and timestamp.
    """
    prices = load_prices(symbol)
    if prices is None:
        return {"status": "error", "message": f"Could not load {symbol} data"}

    features = compute_features(prices, config)
    if features is None:
        return {"status": "error", "message": "Insufficient data for feature computation"}

    classified = classify_day(features, config)

    result = {
        "status": "ok",
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "features": classified.to_dict(),
        "state": _build_state(classified, prices),
    }
    return result


def _build_state(
    features: DayFeatures,
    prices: np.ndarray,
) -> dict:
    """Build state dict with recent price context."""
    closes = prices[:, 0] if prices.shape[1] == 1 else prices[:, 3]
    last_5 = closes[-5:]
    ret_5d = (last_5[-1] - last_5[0]) / last_5[0] if last_5[0] > 0 else 0.0
    last_20 = closes[-20:] if len(closes) >= 20 else closes
    ret_20d = (last_20[-1] - last_20[0]) / last_20[0] if last_20[0] > 0 else 0.0

    return {
        "price_last": float(closes[-1]),
        "return_5d": round(float(ret_5d), 6),
        "return_20d": round(float(ret_20d), 6),
        "n_days": len(prices),
    }


def save_state(result: Dict, state_file: Optional[Path] = None) -> None:
    """Save detection result to state file."""
    if state_file is None:
        state_file = STATE_FILE
    state_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(state_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"State saved to {state_file}")
    except OSError as e:
        logger.error(f"Failed to save state: {e}")


def load_state(state_file: Optional[Path] = None) -> Optional[Dict]:
    """Load the last saved detection result."""
    if state_file is None:
        state_file = STATE_FILE
    if not state_file.exists():
        return None
    try:
        with open(state_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load state: {e}")
        return None


def get_same_day_signal(
    symbol: str = "SPY",
    config: Optional[ClassifierConfig] = None,
) -> Dict:
    """Get same-day execution signal based on regime classification.

    Returns a dict with:
    - regime: classification of current day
    - confidence: how confident the model is
    - execution_adjustment: multiplier for rebalance execution (0.0-1.0)
      - CRISIS: 0.0 (freeze execution)
      - HIGH_VOL: 0.5 (partial, be careful)
      - TREND_UP: 1.0 (normal execution)
      - TREND_DOWN: 1.0 (normal execution)
      - MEAN_REVERT: 0.8 (slight caution)
    """
    result = detect_regime(symbol, config)
    if result.get("status") != "ok":
        return {"status": "error", "message": result.get("message")}

    regime = result["features"]["regime"]

    adjustment_map = {
        DayRegime.CRISIS.value: 0.0,
        DayRegime.HIGH_VOL.value: 0.5,
        DayRegime.TREND_UP.value: 1.0,
        DayRegime.TREND_DOWN.value: 1.0,
        DayRegime.MEAN_REVERT.value: 0.8,
        DayRegime.UNKNOWN.value: 0.8,
    }

    signal = {
        "status": "ok",
        "symbol": symbol,
        "regime": regime,
        "confidence": result["features"]["confidence"],
        "execution_adjustment": adjustment_map.get(regime, 0.8),
        "features": result["features"],
        "state": result["state"],
        "timestamp": result["timestamp"],
    }
    return signal


def main_cli() -> None:
    """CLI entry point for the vol-volume-gap day classifier."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Volatility-Volume-Gap Day Regime Classifier"
    )
    parser.add_argument(
        "action",
        choices=["detect", "signal"],
        default="detect",
        nargs="?",
        help="Action to perform (default: detect)",
    )
    parser.add_argument(
        "--symbol",
        default="SPY",
        help="Symbol to analyze (default: SPY)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save result to state file",
    )

    args = parser.parse_args()

    if args.action == "signal":
        result = get_same_day_signal(args.symbol)
    else:
        result = detect_regime(args.symbol)

    if args.save and result.get("status") == "ok":
        save_state(result)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main_cli()
