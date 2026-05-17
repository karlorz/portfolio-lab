"""
Portfolio-Lab v8.06: Multi-Timeframe Signal Fusion

Decomposes all 19+ signal sources by investment horizon (short/medium/long)
and fuses them with timeframe-appropriate weighting. Prevents short-term
noise from corrupting long-term allocation decisions while ensuring
short-term tactical signals aren't diluted by slow-moving trend signals.

Key Concepts:
  - SHORT (<1 month): VP-MACD, Mean Reversion, Closing Auction, VIXY Hedge
  - MEDIUM (1-6 months): CTA Trend, Multi-Speed Mom, Factor Rotation, etc.
  - LONG (6-12 months): TSMOM, Duration Regime, Fed Policy, Tax-Aware

Design:
  - Timeframe decomposition: each signal assigned to one timeframe bucket
  - Within-bucket consensus: weighted average with confidence
  - Cross-timeframe fusion: long-term determines core bias, medium for tilt, short for tactical
  - Regime-aware adjustment: in high vol, medium-term gets higher weight
  - Output: single fused signal (-1 to +1) per asset class

Usage:
    python -m src.signals.multi_timeframe_fusion fuse
    python -m src.signals.multi_timeframe_fusion explain
"""

import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Investment timeframe horizons."""
    SHORT = "short"       # < 1 month
    MEDIUM = "medium"     # 1-6 months
    LONG = "long"         # 6-12 months


@dataclass
class TimeframeSignal:
    """A signal with its timeframe classification."""
    source: str
    timeframe: Timeframe
    value: float       # -1 to +1
    confidence: float   # 0 to 1
    weight: float       # within-bucket weight
    explanation: str = ""


@dataclass
class TimeframeBucket:
    """Aggregated signals within a timeframe bucket."""
    timeframe: Timeframe
    signals: List[TimeframeSignal]
    consensus: float     # -1 to +1
    agreement: float     # 0 to 1 (how aligned are signals within bucket)
    active_count: int
    total_weight: float  # weight of this bucket in final fusion
    explanation: str = ""


@dataclass
class FusedResult:
    """Final fused signal output."""
    timestamp: str
    overall_signal: float        # -1 to +1
    confidence: float            # 0 to 1
    short_term_signal: float
    medium_term_signal: float
    long_term_signal: float
    equity_bias: float           # SPY recommendation
    duration_bias: float         # TLT recommendation
    gold_bias: float             # GLD recommendation
    regime: str
    buckets: Dict[str, TimeframeBucket]
    explanation: str


# Signal-to-timeframe mapping
# Based on empirical observation of signal construction and historical lookback
SIGNAL_TIMEFRAMES = {
    # Short-term (<1 month)
    "tsfm_momentum": Timeframe.LONG,
    "hmm_regime": Timeframe.MEDIUM,
    "cta_trend": Timeframe.MEDIUM,
    "macro_momentum": Timeframe.LONG,
    "multi_speed_momentum": Timeframe.MEDIUM,
    "duration_regime": Timeframe.LONG,
    "circuit_breaker": Timeframe.SHORT,
    "factor_rotation": Timeframe.MEDIUM,
    "closing_auction": Timeframe.SHORT,
    "unified_overlay": Timeframe.LONG,
    "mean_reversion": Timeframe.SHORT,
    "transformer_regime": Timeframe.MEDIUM,
    "transient_factors": Timeframe.SHORT,
    "visibility_graph": Timeframe.MEDIUM,
    "vp_macd": Timeframe.SHORT,
    "cross_asset_rv": Timeframe.MEDIUM,
    "regime_classifier": Timeframe.MEDIUM,
    "factor_timing": Timeframe.MEDIUM,
    "risk_budget": Timeframe.LONG,
    "llm_narrative": Timeframe.MEDIUM,
    "tax_aware": Timeframe.LONG,
    "vixy_hedge": Timeframe.SHORT,
}


# Default timeframe bucket weights (adjusted based on regime)
TIMEFRAME_BUCKET_WEIGHTS = {
    "normal": {
        Timeframe.SHORT: 0.15,
        Timeframe.MEDIUM: 0.35,
        Timeframe.LONG: 0.50,
    },
    "high_vol": {
        Timeframe.SHORT: 0.25,
        Timeframe.MEDIUM: 0.45,
        Timeframe.LONG: 0.30,
    },
    "crisis": {
        Timeframe.SHORT: 0.40,
        Timeframe.MEDIUM: 0.40,
        Timeframe.LONG: 0.20,
    },
    "recovery": {
        Timeframe.SHORT: 0.10,
        Timeframe.MEDIUM: 0.30,
        Timeframe.LONG: 0.60,
    },
}


class MultiTimeframeFusion:
    """
    Multi-timeframe signal fusion engine.
    
    Takes raw signal readings from all sources, classifies by timeframe,
    computes within-bucket consensus, then fuses across timeframes.
    """

    def __init__(self, state_path: Optional[str] = None):
        self.state_path = state_path or str(
            project_root / "data" / "multi_timeframe_state.json"
        )
        self._load_state()

    def _load_state(self):
        """Load persisted state."""
        path = Path(self.state_path)
        if path.exists():
            try:
                with open(path) as f:
                    self.state = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.state = self._default_state()
        else:
            self.state = self._default_state()

    def _default_state(self) -> dict:
        return {
            "initialized": datetime.now().isoformat(),
            "fusion_count": 0,
            "last_fusion": None,
            "previous_signals": {},
            "history": [],
        }

    def _save_state(self):
        """Persist state."""
        path = Path(self.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def classify_signals(
        self, raw_signals: Dict[str, float],
        signal_confidences: Optional[Dict[str, float]] = None,
        signal_explanations: Optional[Dict[str, str]] = None,
        regime: str = "normal"
    ) -> Tuple[Dict[str, TimeframeBucket], str]:
        """
        Classify raw signals into timeframe buckets.
        
        Args:
            raw_signals: {source_name: value} mapping
            signal_confidences: {source_name: confidence}
            signal_explanations: {source_name: explanation}
            regime: Current market regime
            
        Returns:
            (buckets_dict, explanation_string)
        """
        confidences = signal_confidences or {}
        explanations = signal_explanations or {}
        
        # Always initialize buckets (even for empty signals)
        buckets = {
            tf.value: TimeframeBucket(
                timeframe=tf,
                signals=[],
                consensus=0.0,
                agreement=0.0,
                active_count=0,
                total_weight=TIMEFRAME_BUCKET_WEIGHTS.get(
                    regime, TIMEFRAME_BUCKET_WEIGHTS["normal"]
                ).get(tf, 0.33),
                explanation=f"No {tf.value}-term signals active"
            )
            for tf in Timeframe
        }
        
        if not raw_signals:
            return buckets, "No signals provided for classification"
        
        # Distribute signals into buckets
        for source_name, value in raw_signals.items():
            tf = SIGNAL_TIMEFRAMES.get(source_name)
            if tf is None:
                continue
            
            confidence = confidences.get(source_name, 0.5)
            # Within-bucket weight: confidence * 1 (equal weight base)
            within_weight = confidence
            
            signal = TimeframeSignal(
                source=source_name,
                timeframe=tf,
                value=max(-1.0, min(1.0, float(value))),
                confidence=float(confidence),
                weight=float(within_weight),
                explanation=explanations.get(source_name, "")
            )
            buckets[tf.value].signals.append(signal)
            buckets[tf.value].active_count += 1
        
        # Compute within-bucket consensus
        explanations_parts = []
        for tf in Timeframe:
            bucket = buckets[tf.value]
            if bucket.active_count == 0:
                bucket.consensus = 0.0
                bucket.agreement = 0.0
                bucket.explanation = f"No {tf.value}-term signals active"
                explanations_parts.append(bucket.explanation)
                continue
            
            signals = bucket.signals
            total_weight = sum(s.weight for s in signals if s.weight > 0)
            if total_weight == 0:
                bucket.consensus = 0.0
                bucket.agreement = 0.0
            else:
                weighted_sum = sum(s.value * s.weight for s in signals)
                bucket.consensus = weighted_sum / total_weight
            
            # Agreement: how aligned are signals (1 = all same direction)
            directions = [1 if s.value > 0.1 else (-1 if s.value < -0.1 else 0) for s in signals]
            if len(directions) > 0:
                dominant = max(set(directions), key=directions.count)
                agreement_count = directions.count(dominant)
                bucket.agreement = agreement_count / len(directions)
            else:
                bucket.agreement = 0.0
            
            signal_summary = ", ".join(
                f"{s.source}={s.value:.2f}(c={s.confidence:.2f})"
                for s in signals[:5]
            )
            if len(signals) > 5:
                signal_summary += f" ... +{len(signals) - 5} more"
            
            bucket.explanation = (
                f"{tf.value}-term: {bucket.active_count} signals, "
                f"consensus={bucket.consensus:.3f}, "
                f"agreement={bucket.agreement:.1%}, "
                f"signals=[{signal_summary}]"
            )
            explanations_parts.append(bucket.explanation)
        
        return buckets, "\n".join(explanations_parts)

    def fuse(
        self,
        raw_signals: Dict[str, float],
        signal_confidences: Optional[Dict[str, float]] = None,
        signal_explanations: Optional[Dict[str, str]] = None,
        regime: str = "normal"
    ) -> FusedResult:
        """
        Full fusion pipeline: classify -> bucket -> cross-timeframe fuse.
        
        Args:
            raw_signals: {source_name: value} mapping
            signal_confidences: {source_name: confidence}
            signal_explanations: {source_name: explanation}
            regime: Current market regime
            
        Returns:
            FusedResult with per-asset bias and overall signal
        """
        if regime not in TIMEFRAME_BUCKET_WEIGHTS:
            regime = "normal"
        
        buckets, classification_exp = self.classify_signals(
            raw_signals, signal_confidences, signal_explanations, regime
        )
        
        bucket_weights = TIMEFRAME_BUCKET_WEIGHTS[regime]
        
        # Cross-timeframe fusion
        total_weighted_signal = 0.0
        total_weight = 0.0
        total_confidence = 0.0
        bucket_fused = {}
        
        for tf in Timeframe:
            bucket = buckets[tf.value]
            bw = bucket_weights.get(tf, 0.33)
            bucket.total_weight = bw
            bucket_fused[tf.value] = bucket
            
            total_weighted_signal += bucket.consensus * bw
            total_weight += bw
            total_confidence += (abs(bucket.consensus) * 0.5 + 0.5) * bw
        
        overall_signal = total_weighted_signal / total_weight if total_weight > 0 else 0.0
        overall_confidence = total_confidence / total_weight if total_weight > 0 else 0.5
        
        # Compute per-asset bias from bucket signals
        # Equity: weighted average of all signals
        equity_bias = self._compute_asset_bias(
            buckets, bucket_weights, "equity"
        )
        duration_bias = self._compute_asset_bias(
            buckets, bucket_weights, "duration"
        )
        gold_bias = self._compute_asset_bias(
            buckets, bucket_weights, "gold"
        )
        
        explanation_parts = [
            f"Multi-Timeframe Fusion ({regime} regime)",
            f"Overall signal: {overall_signal:.3f} (confidence: {overall_confidence:.1%})",
            f"Short-term: {buckets[Timeframe.SHORT.value].consensus:.3f} @ {bucket_weights[Timeframe.SHORT]:.0%}",
            f"Medium-term: {buckets[Timeframe.MEDIUM.value].consensus:.3f} @ {bucket_weights[Timeframe.MEDIUM]:.0%}",
            f"Long-term: {buckets[Timeframe.LONG.value].consensus:.3f} @ {bucket_weights[Timeframe.LONG]:.0%}",
            f"Equity bias: {equity_bias:.3f}",
            f"Duration bias: {duration_bias:.3f}",
            f"Gold bias: {gold_bias:.3f}",
            "",
            classification_exp,
        ]
        
        result = FusedResult(
            timestamp=datetime.now().isoformat(),
            overall_signal=round(overall_signal, 4),
            confidence=round(overall_confidence, 4),
            short_term_signal=round(buckets[Timeframe.SHORT.value].consensus, 4),
            medium_term_signal=round(buckets[Timeframe.MEDIUM.value].consensus, 4),
            long_term_signal=round(buckets[Timeframe.LONG.value].consensus, 4),
            equity_bias=round(equity_bias, 4),
            duration_bias=round(duration_bias, 4),
            gold_bias=round(gold_bias, 4),
            regime=regime,
            buckets=bucket_fused,
            explanation="\n".join(explanation_parts),
        )
        
        # Update state
        self.state["fusion_count"] += 1
        self.state["last_fusion"] = result.timestamp
        self.state["previous_signals"] = {
            k: v for k, v in raw_signals.items()
        }
        self.state["history"].append({
            "timestamp": result.timestamp,
            "overall_signal": result.overall_signal,
            "short": result.short_term_signal,
            "medium": result.medium_term_signal,
            "long": result.long_term_signal,
            "regime": regime,
        })
        # Keep only last 30 entries
        if len(self.state["history"]) > 30:
            self.state["history"] = self.state["history"][-30:]
        
        self._save_state()
        
        return result

    def _compute_asset_bias(
        self,
        buckets: Dict[str, TimeframeBucket],
        bucket_weights: Dict[Timeframe, float],
        asset_type: str
    ) -> float:
        """
        Compute per-asset bias from signal buckets.
        
        Uses signal assignment heuristics:
        - Equity signals: tsfm_momentum, multi_speed_momentum, cta_trend,
          mean_reversion, factor_rotation, cross_asset_rv, factor_timing
        - Duration signals: duration_regime, macro_momentum, risk_budget, tax_aware
        - Gold signals: unified_overlay, llm_narrative, vixy_hedge
        """
        equity_sources = {
            "tsfm_momentum", "multi_speed_momentum", "cta_trend",
            "mean_reversion", "factor_rotation", "cross_asset_rv",
            "factor_timing", "transformer_regime", "transient_factors",
            "vp_macd", "closing_auction", "visibility_graph",
        }
        duration_sources = {
            "duration_regime", "macro_momentum", "risk_budget",
            "tax_aware", "unified_overlay",
        }
        gold_sources = {
            "unified_overlay", "llm_narrative", "vixy_hedge",
        }
        
        if asset_type == "equity":
            relevant = equity_sources
        elif asset_type == "duration":
            relevant = duration_sources
        elif asset_type == "gold":
            relevant = gold_sources
        else:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for tf_str, bucket in buckets.items():
            bw = bucket_weights.get(Timeframe(tf_str), 0.33)
            for signal in bucket.signals:
                if signal.source in relevant:
                    weighted_sum += signal.value * signal.confidence * bw
                    total_weight += signal.confidence * bw
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_signal_timeframe(self, source_name: str) -> Optional[str]:
        """Get the timeframe classification for a signal source."""
        tf = SIGNAL_TIMEFRAMES.get(source_name)
        return tf.value if tf else None

    def get_timeframe_breakdown(self) -> Dict[str, List[str]]:
        """Get a human-readable breakdown of all signals by timeframe."""
        breakdown = {tf.value: [] for tf in Timeframe}
        for source, tf in SIGNAL_TIMEFRAMES.items():
            breakdown[tf.value].append(source)
        return breakdown


def load_signal_data() -> Tuple[Dict[str, float], Dict[str, float], str]:
    """
    Load signal data from the ensemble voter's state file.
    
    Returns:
        (raw_signals, confidences, regime)
    """
    state_path = project_root / "data" / "ensemble_state.json"
    if not state_path.exists():
        # Try alternative paths
        alt_paths = [
            project_root / "data" / "ensemble_voter_state.json",
            project_root / "data" / "signal_state.json",
            Path("/tmp/ensemble_state.json"),
        ]
        for p in alt_paths:
            if p.exists():
                state_path = p
                break
        else:
            return {}, {}, "normal"
    
    try:
        with open(state_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}, {}, "normal"
    
    signals = {}
    confidences = {}
    regime = data.get("regime", "normal")
    
    for vote in data.get("source_votes", []):
        source = vote.get("source", "")
        if isinstance(source, dict):
            source = source.get("value", "")
        if not source:
            continue
        signals[source] = vote.get("value", 0.0)
        confidences[source] = vote.get("confidence", 0.5)
    
    return signals, confidences, regime


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Timeframe Signal Fusion Engine"
    )
    parser.add_argument(
        "action",
        choices=["fuse", "explain", "breakdown", "status"],
        help="Action to perform"
    )
    parser.add_argument(
        "--regime", default=None,
        help="Force a specific regime (normal/high_vol/crisis/recovery)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save results to state file"
    )
    
    args = parser.parse_args()
    
    fusion = MultiTimeframeFusion()
    
    if args.action == "breakdown":
        breakdown = fusion.get_timeframe_breakdown()
        print("\n=== Signal Timeframe Breakdown ===\n")
        for tf, sources in breakdown.items():
            print(f"{tf.upper():>8} ({len(sources)} signals):")
            for s in sources:
                print(f"  - {s}")
            print()
        return
    
    if args.action == "status":
        print("\n=== Multi-Timeframe Fusion Status ===\n")
        print(f"Fusion count: {fusion.state['fusion_count']}")
        print(f"Last fusion: {fusion.state['last_fusion']}")
        print(f"Total signal sources: {len(SIGNAL_TIMEFRAMES)}")
        print(f"\nSignal classification:")
        for tf in Timeframe:
            count = sum(1 for s in SIGNAL_TIMEFRAMES if SIGNAL_TIMEFRAMES[s] == tf)
            print(f"  {tf.value}: {count} sources")
        return
    
    if args.action == "explain":
        signals, confidences, live_regime = load_signal_data()
        result = fusion.fuse(
            signals, confidences, regime=args.regime or live_regime
        )
        print(f"\n=== Multi-Timeframe Fusion Explanation ===\n")
        print(result.explanation)
        return
    
    # Default: fuse
    signals, confidences, live_regime = load_signal_data()
    regime = args.regime or live_regime
    
    if not signals:
        print("No signal data available. Using synthetic demo data...")
        # Generate synthetic demo for CLI demo
        signals = {
            "tsfm_momentum": 0.35,
            "hmm_regime": 0.10,
            "cta_trend": 0.25,
            "macro_momentum": 0.20,
            "multi_speed_momentum": 0.15,
            "duration_regime": -0.10,
            "circuit_breaker": 0.0,
            "factor_rotation": 0.05,
            "closing_auction": 0.02,
            "unified_overlay": 0.08,
            "mean_reversion": -0.03,
            "transformer_regime": 0.12,
            "transient_factors": 0.05,
            "visibility_graph": 0.08,
            "vp_macd": 0.10,
            "cross_asset_rv": 0.15,
            "regime_classifier": 0.05,
            "factor_timing": 0.10,
            "risk_budget": 0.05,
            "llm_narrative": 0.12,
            "tax_aware": 0.03,
            "vixy_hedge": 0.05,
        }
        confidences = {k: 0.5 + abs(v) * 0.3 for k, v in signals.items()}
    
    result = fusion.fuse(signals, confidences, regime=regime)
    
    print(f"\n=== Multi-Timeframe Fusion Result ===\n")
    print(f"Timestamp:  {result.timestamp}")
    print(f"Regime:     {result.regime}")
    print(f"Overall:    {result.overall_signal:+.4f}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"\nTimeframe Breakdown:")
    print(f"  Short-term:  {result.short_term_signal:+.4f}")
    print(f"  Medium-term: {result.medium_term_signal:+.4f}")
    print(f"  Long-term:   {result.long_term_signal:+.4f}")
    print(f"\nAsset Bias:")
    print(f"  Equity:   {result.equity_bias:+.4f}")
    print(f"  Duration: {result.duration_bias:+.4f}")
    print(f"  Gold:     {result.gold_bias:+.4f}")
    print(f"\nExplanation:")
    print(result.explanation)


if __name__ == "__main__":
    main()
