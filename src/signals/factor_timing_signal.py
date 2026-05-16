"""
Factor Timing Signal Generator - v6.02 Implementation

Computes cross-sectional factor Z-scores from available factor ETFs (MTUM, USMV, QUAL, VLUE)
and generates regime-based tilt signals for the EnsembleVoter.

Key Components:
- FactorTimingCalculator: Loads prices, computes multi-horizon momentum, cross-sectional ranks
- Regime-based tilt mapping (bull→momentum, bear→low vol, high vol→quality+low vol)
- Output: FactorTimingResult dataclass with Z-scores, composite urgency, top/bottom factors

Expected impact: +0.02-0.05 Sharpe through systematic factor rotation.
Complements existing TSMOM overlay (different signal source — cross-sectional vs time-series).

Usage:
    python -m src.signals.factor_timing_signal compute
    python -m src.signals.factor_timing_signal explain
"""

import json
import logging
import numpy as np
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PRICES_PATH = PROJECT_ROOT / "public" / "data" / "prices.json"

# Factor ETF universe (subset available in price data)
FACTOR_ETFS = {
    "MTUM": {"factor": "momentum", "name": "iShares MSCI USA Momentum Factor", "rank": 0},
    "USMV": {"factor": "low_vol", "name": "iShares MSCI USA Min Vol Factor", "rank": 1},
    "QUAL": {"factor": "quality", "name": "iShares MSCI USA Quality Factor", "rank": 2},
    "VLUE": {"factor": "value", "name": "iShares MSCI USA Value Factor", "rank": 3},
}

# Factor categories for regime tilting
FACTOR_CATEGORIES = {
    "aggressive": ["MTUM"],       # Momentum — perform in bull
    "defensive": ["USMV", "QUAL"],   # Low vol + quality — perform in bear/high vol
    "neutral": ["VLUE"],          # Value — mixed regime dependence
}

# Momentum horizons (trading days)
MOMENTUM_HORIZONS = {
    "short": 63,    # ~3 months
    "medium": 126,  # ~6 months
    "long": 252,    # ~12 months
}

# Regime-based factor tilt weights
# Higher weight = more preferred in that regime
REGIME_FACTOR_TILTS = {
    "normal": {"MTUM": 0.35, "USMV": 0.20, "QUAL": 0.30, "VLUE": 0.15},
    "bull": {"MTUM": 0.50, "USMV": 0.10, "QUAL": 0.30, "VLUE": 0.10},
    "bear": {"MTUM": 0.10, "USMV": 0.40, "QUAL": 0.35, "VLUE": 0.15},
    "high_vol": {"MTUM": 0.10, "USMV": 0.45, "QUAL": 0.35, "VLUE": 0.10},
    "crisis": {"MTUM": 0.05, "USMV": 0.50, "QUAL": 0.40, "VLUE": 0.05},
}


@dataclass
class FactorMomentum:
    """Multi-horizon momentum scores for a single factor ETF."""
    symbol: str
    factor_name: str
    short_momentum: float      # ~3m return
    medium_momentum: float     # ~6m return
    long_momentum: float       # ~12m return
    composite_z: float         # Cross-sectional Z-score (blended horizons)
    rank: int                  # Cross-sectional rank (1 = best)
    data_points: int           # Number of valid price points


@dataclass
class FactorTimingResult:
    """Complete factor timing signal output."""
    timestamp: str
    factor_scores: Dict[str, float]           # Symbol -> composite Z-score
    factor_rankings: List[Tuple[str, float]]  # Sorted (symbol, score) desc
    composite_urgency: float                  # 0.0-1.0 urgency modifier
    top_factor: str                           # Best performing factor symbol
    bottom_factor: str                        # Worst performing factor symbol
    regime_tilt: Dict[str, float]             # Regime-based preference weights
    signal_value: float                       # -1 to +1 aggregate signal for EnsembleVoter
    factor_divergence: float                  # Spread between top and bottom (0-2)
    explanation: str = ""                     # Human-readable summary

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def load_factor_prices(
    symbols: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Load price arrays for factor ETFs from prices.json.

    Args:
        symbols: List of tickers to load (default: all FACTOR_ETFS keys)

    Returns:
        Dict mapping symbol -> numpy array of close prices (chronological)
    """
    if symbols is None:
        symbols = list(FACTOR_ETFS.keys())

    if not PRICES_PATH.exists():
        logger.error(f"Prices file not found: {PRICES_PATH}")
        return {}

    try:
        with open(PRICES_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading prices: {e}")
        return {}

    result = {}
    for symbol in symbols:
        symbol_data = data.get(symbol, data.get(symbol.upper()))
        if not symbol_data or not isinstance(symbol_data, list):
            logger.warning(f"No price data for {symbol}")
            continue

        # Sort chronologically and extract prices
        sorted_data = sorted(symbol_data, key=lambda x: x.get("d", ""))
        prices = np.array([item["p"] for item in sorted_data], dtype=float)

        if len(prices) < 260:  # Need at least 1 year of data
            logger.warning(f"Insufficient data for {symbol}: {len(prices)} points (< 260)")
            continue

        result[symbol] = prices

    return result


def compute_returns(prices: np.ndarray, lookback: int) -> Optional[float]:
    """Compute simple return over lookback period.

    Args:
        prices: Chronological price array (most recent last)
        lookback: Number of trading days to look back

    Returns:
        Return as decimal (e.g., 0.05 for 5%), or None if insufficient data
    """
    if len(prices) < lookback + 1:
        return None
    if prices[-lookback - 1] == 0:
        return None

    return (prices[-1] - prices[-lookback - 1]) / prices[-lookback - 1]


def compute_factor_scores(
    prices_dict: Dict[str, np.ndarray],
) -> Dict[str, FactorMomentum]:
    """Compute multi-horizon momentum scores for each factor.

    For each factor ETF, computes short (3m), medium (6m), and long (12m)
    momentum returns, then cross-sectionally normalizes to Z-scores.

    Args:
        prices_dict: Dict mapping symbol -> price array

    Returns:
        Dict mapping symbol -> FactorMomentum namedtuple
    """
    # Step 1: Compute raw returns for each horizon
    raw_returns: Dict[str, Dict[str, Optional[float]]] = {}
    for symbol, prices in prices_dict.items():
        raw_returns[symbol] = {
            horizon: compute_returns(prices, days)
            for horizon, days in MOMENTUM_HORIZONS.items()
        }

    # Step 2: Cross-sectional normalization for each horizon
    z_scores: Dict[str, Dict[str, float]] = {}
    for symbol in prices_dict:
        z_scores[symbol] = {}

    for horizon in MOMENTUM_HORIZONS:
        valid_returns = {
            s: r[horizon]
            for s, r in raw_returns.items()
            if r[horizon] is not None
        }

        if len(valid_returns) < 2:
            # Not enough data for cross-sectional Z-score
            for symbol in prices_dict:
                z_scores[symbol][horizon] = 0.0
            continue

        values = np.array(list(valid_returns.values()))
        mean = np.mean(values)
        std = np.std(values, ddof=1)

        if std < 1e-10:  # Degenerate case — all equal
            for symbol in prices_dict:
                z_scores[symbol][horizon] = 0.0
        else:
            for symbol in prices_dict:
                if symbol in valid_returns:
                    z_scores[symbol][horizon] = (valid_returns[symbol] - mean) / std
                else:
                    z_scores[symbol][horizon] = 0.0

    # Step 3: Compute composite Z-score (weighted average of horizons)
    # short: 20%, medium: 40%, long: 40% — longer horizons get more weight
    horizon_weights = {"short": 0.20, "medium": 0.40, "long": 0.40}

    factor_scores = {}
    for symbol in prices_dict:
        composite = sum(
            horizon_weights[h] * z_scores[symbol][h]
            for h in MOMENTUM_HORIZONS
        )

        # Get data point count
        data_points = int(len(prices_dict[symbol]))
        meta = FACTOR_ETFS.get(symbol, {"factor": "unknown"})

        sr = raw_returns[symbol]
        short_mom = float(sr["short"]) if sr["short"] is not None else 0.0
        medium_mom = float(sr["medium"]) if sr["medium"] is not None else 0.0
        long_mom = float(sr["long"]) if sr["long"] is not None else 0.0

        factor_scores[symbol] = FactorMomentum(
            symbol=symbol,
            factor_name=meta["factor"],
            short_momentum=short_mom,
            medium_momentum=medium_mom,
            long_momentum=long_mom,
            composite_z=round(float(composite), 4),
            rank=0,  # Set below
            data_points=data_points,
        )

    # Step 4: Rank by composite Z-score
    sorted_symbols = sorted(
        factor_scores.keys(),
        key=lambda s: factor_scores[s].composite_z,
        reverse=True,
    )
    for rank, symbol in enumerate(sorted_symbols):
        factor_scores[symbol].rank = rank + 1

    return factor_scores


def compute_regime_tilt(
    factor_scores: Dict[str, FactorMomentum],
    regime: str = "normal",
) -> Dict[str, float]:
    """Compute regime-based factor tilt weights.

    Blends cross-sectional momentum scores with regime preferences.

    Args:
        factor_scores: Computed factor momentum scores
        regime: Market regime name (normal, bull, bear, high_vol, crisis)

    Returns:
        Dict mapping symbol -> tilt weight (sums to ~1.0)
    """
    regime = regime.lower() if regime in REGIME_FACTOR_TILTS else "normal"
    base_weights = REGIME_FACTOR_TILTS[regime]

    # Adjust weights based on actual momentum Z-scores
    # If a factor has very negative Z-score in a regime where it's preferred,
    # reduce its weight (momentum confirms regime preference)
    adjusted = {}
    for symbol in factor_scores:
        z = factor_scores[symbol].composite_z
        base_w = base_weights.get(symbol, 0.25)

        # Modulation: if Z-score > +1 (strong momentum), amplify preference
        # If Z-score < -1 (strongly negative), penalize
        if z > 1.0:
            modulation = 1.0 + 0.2 * min(z, 2.0)
        elif z < -1.0:
            modulation = 1.0 - 0.2 * min(abs(z), 2.0)
        else:
            modulation = 1.0

        adjusted[symbol] = base_w * max(0.5, modulation)

    # Normalize to sum to 1.0
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {s: w / total for s, w in adjusted.items()}

    return adjusted


def compute_signal_value(
    factor_scores: Dict[str, FactorMomentum],
    regime_tilt: Dict[str, float],
) -> float:
    """Compute aggregate signal value (-1 to +1) for EnsembleVoter.

    Positive = preference for aggressive factors (bullish tilt)
    Negative = preference for defensive factors (bearish tilt)

    Weighted sum of Z-scores, weighted by regime tilt.
    """
    signal = 0.0
    for symbol, tilt_w in regime_tilt.items():
        if symbol in factor_scores:
            z = factor_scores[symbol].composite_z
            signal += tilt_w * z

    # Clamp to [-1, +1]
    return float(np.clip(signal / 2.0, -1.0, 1.0))


def compute_composite_urgency(
    factor_scores: Dict[str, FactorMomentum],
) -> float:
    """Compute composite urgency modifier (0.0-1.0).

    High urgency when:
    - Factor divergence is large (winners and losers far apart)
    - Top factor has strong positive momentum

    Low urgency when:
    - All factors clustered near zero
    - No clear leader
    """
    z_values = np.array([fs.composite_z for fs in factor_scores.values()])
    if len(z_values) < 2:
        return 0.5

    divergence = float(np.max(z_values) - np.min(z_values))
    max_z = float(np.max(z_values))

    # Divergence contributes up to 0.7, top Z contributes up to 0.3
    urgency = min(divergence / 4.0, 0.7) + min(max(0, max_z) / 3.0, 0.3)

    return float(np.clip(urgency, 0.0, 1.0))


def compute_factor_divergence(
    factor_scores: Dict[str, FactorMomentum],
) -> float:
    """Compute spread between best and worst factor Z-score (0-2 scale)."""
    z_values = [fs.composite_z for fs in factor_scores.values()]
    if len(z_values) < 2:
        return 0.0
    return float(np.clip(max(z_values) - min(z_values), 0.0, 2.0))


def generate_timing_signal(
    regime: str = "normal",
) -> Optional[FactorTimingResult]:
    """Generate complete factor timing signal.

    Args:
        regime: Market regime name (passed from EnsembleVoter)

    Returns:
        FactorTimingResult or None if insufficient data
    """
    prices_dict = load_factor_prices()
    if len(prices_dict) < 2:
        logger.error("Insufficient factor price data to generate signal")
        return None

    factor_scores = compute_factor_scores(prices_dict)
    if len(factor_scores) < 2:
        logger.error("Insufficient factor scores computed")
        return None

    regime_tilt = compute_regime_tilt(factor_scores, regime)
    signal_value = compute_signal_value(factor_scores, regime_tilt)
    composite_urgency = compute_composite_urgency(factor_scores)
    divergence = compute_factor_divergence(factor_scores)

    # Rankings
    sorted_factors = sorted(
        factor_scores.values(),
        key=lambda fs: fs.composite_z,
        reverse=True,
    )

    top_factor = sorted_factors[0].symbol if sorted_factors else "UNKNOWN"
    bottom_factor = sorted_factors[-1].symbol if len(sorted_factors) > 1 else "UNKNOWN"

    # Factor scores dict
    factor_scores_dict = {
        s: fs.composite_z for s, fs in factor_scores.items()
    }

    # Rankings list
    rankings = [(fs.symbol, fs.composite_z) for fs in sorted_factors]

    # Build explanation
    explanations = [
        f"Regime: {regime}",
        f"Top factor: {top_factor} ({factor_scores[top_factor].composite_z:+.2f}σ)",
        f"Bottom factor: {bottom_factor} ({factor_scores[bottom_factor].composite_z:+.2f}σ)",
        f"Divergence: {divergence:.2f}σ",
    ]
    for symbol, z in rankings:
        fs = factor_scores[symbol]
        explanations.append(
            f"  {symbol} ({fs.factor_name}): short={fs.short_momentum:+.2%}, "
            f"medium={fs.medium_momentum:+.2%}, long={fs.long_momentum:+.2%}, z={z:+.2f}"
        )

    return FactorTimingResult(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        factor_scores=factor_scores_dict,
        factor_rankings=rankings,
        composite_urgency=round(composite_urgency, 4),
        top_factor=top_factor,
        bottom_factor=bottom_factor,
        regime_tilt=regime_tilt,
        signal_value=round(signal_value, 4),
        factor_divergence=round(divergence, 4),
        explanation="\n".join(explanations),
    )


def get_ensemble_signal(
    regime: str = "normal",
) -> Dict:
    """Get factor timing signal formatted for ensemble voter integration.

    This is the primary entry point called by ensemble_voter.collect_signals().

    Args:
        regime: Market regime name

    Returns:
        Dict with signal_value, confidence, urgency, factor_details
    """
    result = generate_timing_signal(regime)
    if result is None:
        return {
            "signal_value": 0.0,
            "confidence": 0.0,
            "urgency": 0.5,
            "active": False,
            "explanation": "Insufficient factor data available",
        }

    return {
        "signal_value": result.signal_value,
        "confidence": min(0.7, 0.3 + 0.4 * result.composite_urgency),
        "urgency": result.composite_urgency,
        "top_factor": result.top_factor,
        "bottom_factor": result.bottom_factor,
        "factor_scores": result.factor_scores,
        "factor_rankings": result.factor_rankings,
        "factor_divergence": result.factor_divergence,
        "regime_tilt": result.regime_tilt,
        "active": True,
        "explanation": result.explanation,
    }


def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.signals.factor_timing_signal [compute|explain] [--regime <regime>]")
        return

    command = sys.argv[1]
    regime = "normal"
    if "--regime" in sys.argv:
        idx = sys.argv.index("--regime")
        if idx + 1 < len(sys.argv):
            regime = sys.argv[idx + 1]

    if command == "compute":
        result = generate_timing_signal(regime)
        if result:
            print(result.to_json())
        else:
            print('{"error": "Failed to generate signal"}')

    elif command == "explain":
        result = generate_timing_signal(regime)
        if result:
            print(result.explanation)
        else:
            print("Failed to generate signal — insufficient data")

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
