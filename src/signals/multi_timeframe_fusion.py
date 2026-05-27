"""Multi-Timeframe Signal Fusion (v806 redo).

Decomposes ensemble signals into 3 timeframe buckets (short/medium/long)
and fuses them with regime-dependent weights. Produces a single
MULTI_TIMEFRAME_FUSION signal for the ensemble voter.

Design:
- Each signal's underlying price series is analyzed at 3 lookback windows
- Short (5d): mean-reversion, news-driven, tactical
- Medium (21d): monthly momentum, earnings cycles
- Long (63d): quarterly trends, regime transitions
- Fusion weights vary by market regime (CRISIS emphasizes short-term,
  LOW_VOL emphasizes long-term)
- ALTERNATIVE_DATA bypasses decomposition (news is instantaneous)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.signals.signal_snapshot import SignalSnapshot
from src.utils.computation_cache import get_realized_volatility

logger = logging.getLogger(__name__)

__all__ = [
    "MultiTimeframeFusion",
    "TimeframeComponent",
    "FUSION_WEIGHTS",
    "TIMEFRAMES",
]

# ── Timeframe Definitions ──────────────────────────────────────────────────

TIMEFRAMES = {
    "short": {"lookback_days": 5, "description": "Mean-reversion, news-driven"},
    "medium": {"lookback_days": 21, "description": "Monthly momentum, earnings"},
    "long": {"lookback_days": 63, "description": "Quarterly trends, regime shifts"},
}

# ── Regime-Dependent Fusion Weights ────────────────────────────────────────

FUSION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "crisis": {"short": 0.50, "medium": 0.30, "long": 0.20},
    "high_vol": {"short": 0.40, "medium": 0.35, "long": 0.25},
    "normal": {"short": 0.25, "medium": 0.40, "long": 0.35},
    "low_vol": {"short": 0.20, "medium": 0.35, "long": 0.45},
    "recovery": {"short": 0.30, "medium": 0.40, "long": 0.30},
}

# Default fusion weights when regime is unknown
_DEFAULT_FUSION_WEIGHTS = FUSION_WEIGHTS["normal"]

@dataclass
class TimeframeComponent:
    """Signal value at a specific timeframe."""

    timeframe: str
    lookback_days: int
    value: float
    confidence: float

    def __post_init__(self):
        self.value = float(np.clip(self.value, -1.0, 1.0))
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))


class MultiTimeframeFusion:
    """Decompose signals into timeframe buckets and fuse with regime weights.

    Args:
        prices_df: DataFrame with DatetimeIndex and ticker columns (SPY, GLD, TLT).
                   If None/empty, get_signal_snapshot() returns inactive.
    """

    def __init__(self, prices_df: Optional[pd.DataFrame] = None):
        self.prices_df = prices_df

    def get_signal_snapshot(
        self,
        tickers: Optional[List[str]] = None,
        date: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> SignalSnapshot:
        """Generate a SignalSnapshot for ensemble voter consumption.

        Args:
            tickers: Tickers to analyze (default: SPY, GLD, TLT).
            date: Unused (for interface compatibility).
            regime: Market regime for fusion weights (default: "normal").

        Returns:
            SignalSnapshot with multi-timeframe fusion value.
        """
        if tickers is None:
            tickers = ["SPY", "GLD", "TLT"]

        if self.prices_df is None or self.prices_df.empty:
            return SignalSnapshot(
                source="multi_timeframe_fusion",
                timestamp=str(datetime.now()),
                value=0.0,
                confidence=0.0,
                regime_fit="all",
                is_active=False,
                explanation="Multi-timeframe fusion: no price data available",
            )

        if regime is None:
            regime = "normal"

        # Get per-asset timeframe decomposition
        per_asset = self._get_per_asset_signals(self.prices_df, tickers)

        if not per_asset:
            return SignalSnapshot(
                source="multi_timeframe_fusion",
                timestamp=str(datetime.now()),
                value=0.0,
                confidence=0.0,
                regime_fit="all",
                is_active=False,
                explanation="Multi-timeframe fusion: no valid tickers",
            )

        # Fuse each asset independently, then aggregate
        asset_signals = {}
        for asset, components in per_asset.items():
            fused = self._fuse_components(components, regime)
            asset_signals[asset] = fused

        # Overall value = mean of per-asset fused signals
        values = list(asset_signals.values())
        overall_value = float(np.mean(values)) if values else 0.0

        # Overall confidence = mean of per-asset confidences
        confidences = []
        for components in per_asset.values():
            conf = self._compute_fusion_confidence(components, regime)
            confidences.append(conf)
        overall_confidence = float(np.mean(confidences)) if confidences else 0.0

        # Build timeframe breakdown for metadata
        timeframe_breakdown = {}
        for asset, components in per_asset.items():
            timeframe_breakdown[asset] = {
                tf: {"value": comp.value, "confidence": comp.confidence}
                for tf, comp in components.items()
            }

        explanation_parts = [f"{k}={v:.3f}" for k, v in asset_signals.items()]
        explanation = (
            f"Multi-timeframe fusion ({regime}): "
            + ", ".join(explanation_parts)
            + f" | overall={overall_value:.3f}, conf={overall_confidence:.3f}"
        )

        return SignalSnapshot(
            source="multi_timeframe_fusion",
            timestamp=str(datetime.now()),
            value=overall_value,
            confidence=overall_confidence,
            asset_signals=asset_signals,
            regime_fit="all",
            is_active=True,
            explanation=explanation,
            metadata={"timeframe_breakdown": timeframe_breakdown, "regime": regime},
        )

    def _get_per_asset_signals(
        self,
        prices_df: pd.DataFrame,
        tickers: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, TimeframeComponent]]:
        """Compute per-asset, per-timeframe signal components.

        Returns:
            Dict of {ticker: {timeframe: TimeframeComponent}}
        """
        if tickers is None:
            tickers = ["SPY", "GLD", "TLT"]

        result = {}
        for ticker in tickers:
            if ticker not in prices_df.columns:
                continue
            prices = prices_df[ticker].dropna()
            if prices.empty:
                continue
            components = self._decompose_momentum(ticker, prices)
            if components:
                result[ticker] = components
        return result

    def _decompose_momentum(
        self, ticker: str, prices: pd.Series
    ) -> Dict[str, TimeframeComponent]:
        """Decompose a price series into timeframe momentum components.

        Each timeframe computes a simple return over its lookback period,
        normalized to [-1, 1] via tanh(vol-scaled return).

        Args:
            ticker: Ticker name (for logging).
            prices: Price series with DatetimeIndex.

        Returns:
            Dict of {timeframe: TimeframeComponent}.
        """
        components = {}
        # Precompute returns once per ticker, reuse across timeframes
        all_returns = prices.pct_change().dropna()

        for tf_name, tf_config in TIMEFRAMES.items():
            lookback = tf_config["lookback_days"]

            if len(prices) < lookback:
                # Insufficient data — zero confidence
                components[tf_name] = TimeframeComponent(
                    timeframe=tf_name,
                    lookback_days=lookback,
                    value=0.0,
                    confidence=0.0,
                )
                continue

            # Compute lookback return
            start_price = prices.iloc[-lookback]
            end_price = prices.iloc[-1]

            if start_price <= 0 or np.isnan(start_price) or np.isnan(end_price):
                components[tf_name] = TimeframeComponent(
                    timeframe=tf_name,
                    lookback_days=lookback,
                    value=0.0,
                    confidence=0.0,
                )
                continue

            raw_return = (end_price / start_price) - 1.0

            # Volatility normalization: use shared TTL-cached computation cache
            returns = all_returns.iloc[-lookback:]
            cached_vol = get_realized_volatility(returns, window=lookback)
            realized_vol = cached_vol if cached_vol is not None else 0.15
            if realized_vol < 0.01:
                realized_vol = 0.01  # Floor to avoid division explosion

            # Vol-scaled signal via tanh (bounded [-1, 1])
            vol_scaled = raw_return / realized_vol
            signal_value = float(np.tanh(vol_scaled))

            # Confidence: based on data coverage and vol regime stability
            data_coverage = min(len(prices) / lookback, 1.0)
            # Higher confidence when vol is moderate (not too high, not too low)
            vol_confidence = 1.0 - abs(realized_vol - 0.15) / 0.30
            vol_confidence = float(np.clip(vol_confidence, 0.3, 1.0))
            confidence = data_coverage * vol_confidence

            components[tf_name] = TimeframeComponent(
                timeframe=tf_name,
                lookback_days=lookback,
                value=signal_value,
                confidence=confidence,
            )

        return components

    def _fuse_components(
        self,
        components: Dict[str, TimeframeComponent],
        regime: str,
    ) -> float:
        """Fuse timeframe components into a single signal value.

        Uses confidence-weighted average with regime-dependent fusion weights.
        Components with zero confidence are excluded.

        Args:
            components: Dict of {timeframe: TimeframeComponent}.
            regime: Market regime key for FUSION_WEIGHTS.

        Returns:
            Fused signal value in [-1, 1].
        """
        weights = FUSION_WEIGHTS.get(regime, _DEFAULT_FUSION_WEIGHTS)

        weighted_sum = 0.0
        weight_sum = 0.0

        for tf_name, comp in components.items():
            if comp.confidence <= 0:
                continue
            w = weights.get(tf_name, 0.0) * comp.confidence
            weighted_sum += w * comp.value
            weight_sum += w

        if weight_sum <= 0:
            return 0.0

        return float(np.clip(weighted_sum / weight_sum, -1.0, 1.0))

    def _compute_fusion_confidence(
        self,
        components: Dict[str, TimeframeComponent],
        regime: str,
    ) -> float:
        """Compute overall confidence reflecting timeframe agreement.

        Higher when timeframes agree (same sign and similar magnitude),
        lower when they diverge.

        Args:
            components: Dict of {timeframe: TimeframeComponent}.
            regime: Market regime (unused currently, reserved for future).

        Returns:
            Confidence in [0, 1].
        """
        weights = FUSION_WEIGHTS.get(regime, _DEFAULT_FUSION_WEIGHTS)
        values = []
        confs = []
        for tf_name, comp in components.items():
            if comp.confidence > 0:
                values.append(comp.value)
                confs.append(comp.confidence * weights.get(tf_name, 0.33))

        if not values:
            return 0.0

        # Mean confidence (weighted by fusion weights)
        mean_conf = float(np.average(confs)) if confs else 0.0

        # Agreement: 1 - normalized std of values (high agreement = low std)
        if len(values) > 1:
            value_std = float(np.std(values))
            # Normalize by max possible std for values in [-1, 1]
            agreement = 1.0 - min(value_std / 2.0, 1.0)
        else:
            agreement = 0.5  # Single timeframe: moderate confidence

        # Blend: 60% mean confidence, 40% agreement
        return float(np.clip(0.6 * mean_conf + 0.4 * agreement, 0.0, 1.0))
