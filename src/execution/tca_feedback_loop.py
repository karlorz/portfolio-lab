#!/usr/bin/env python3
"""
Portfolio-Lab v6.05: TCA-to-Execution Feedback Loop

Closes the loop between post-trade TCA (v6.00) and execution scheduling (v2.83).
Uses historical execution quality scores to adaptively adjust:

1. Urgency thresholds per symbol (poor execution → lower urgency → wait for better windows)
2. Minimum trade values per symbol (high slippage → require larger trades)
3. Cost model calibration (actual vs estimated slippage → adjust estimates)

Architecture:
    TCAScorecard → TCAFeedbackLoop → SignalExecutionBridge / RebalanceScheduler

    TCAFeedbackLoop reads historical quality scores and produces:
    - symbol_urgency_offsets: Dict[str, float] — adjustments to urgency thresholds
    - symbol_min_trade_multipliers: Dict[str, float] — adjustments to min trade values
    - cost_calibration_factors: Dict[str, float] — multiplier on estimated costs
    - aggregate_feedback_quality: float — overall execution health (0-100)

Integration points:
    - SignalExecutionBridge._calculate_urgency() can apply urgency_offsets
    - SignalExecutionBridge._deltas_to_orders() can apply min_trade_multipliers
    - RebalanceScheduler._calculate_optimal_time() can apply cost calibration

Usage:
    from src.execution.tca_feedback_loop import TCAFeedbackLoop
    
    loop = TCAFeedbackLoop()
    feedback = loop.generate_feedback()  # Read TCA data, compute adjustments
    loop.apply_feedback()                # Persist state & return adjustments

CLI:
    python -m src.execution.tca_feedback_loop check
    python -m src.execution.tca_feedback_loop adjust
    python -m src.execution.tca_feedback_loop status
    python -m src.execution.tca_feedback_loop reset
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SymbolExecutionProfile:
    """Execution quality profile for a single symbol."""
    symbol: str
    total_orders: int
    avg_slippage_bps: float
    avg_quality: float
    trend_slope: float  # Positive = improving
    recent_avg_quality: float
    slippage_volatility: float  # Std dev of slippage
    quality_bucket: str  # 'excellent', 'good', 'fair', 'poor', 'bad'

    # Computed adjustments
    urgency_offset: float = 0.0  # -0.3 to +0.1 (negative = less urgent)
    min_trade_multiplier: float = 1.0  # 0.9 to 3.0
    cost_calibration_factor: float = 1.0  # 0.8 to 2.0

    @property
    def feedback_quality(self) -> float:
        """Overall feedback quality score 0-100."""
        base = self.avg_quality
        # Penalize if trend is deteriorating
        trend_penalty = max(0.0, -self.trend_slope * 10.0)
        # Penalize high volatility
        vol_penalty = min(15.0, self.slippage_volatility * 3.0)
        return float(max(0.0, min(100.0, base - trend_penalty - vol_penalty)))


@dataclass
class FeedbackState:
    """Persistent state for the TCA feedback loop."""
    version: str = "6.05"
    generated: str = ""
    overall_quality: float = 75.0
    urgency_global_offset: float = 0.0  # Global urgency adjustment
    min_trade_global_multiplier: float = 1.0
    cost_calibration_global: float = 1.0
    symbols: Dict[str, SymbolExecutionProfile] = field(default_factory=dict)
    previous_quality: List[float] = field(default_factory=list)
    previous_adjustments: int = 0
    status: str = "ok"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "generated": self.generated,
            "overall_quality": round(self.overall_quality, 1),
            "urgency_global_offset": round(self.urgency_global_offset, 3),
            "min_trade_global_multiplier": round(self.min_trade_global_multiplier, 2),
            "cost_calibration_global": round(self.cost_calibration_global, 2),
            "symbols": {
                sym: {
                    "avg_quality": round(p.avg_quality, 1),
                    "trend_slope": round(p.trend_slope, 4),
                    "quality_bucket": p.quality_bucket,
                    "urgency_offset": round(p.urgency_offset, 3),
                    "min_trade_multiplier": round(p.min_trade_multiplier, 2),
                    "cost_calibration": round(p.cost_calibration_factor, 2),
                    "feedback_quality": p.feedback_quality,
                }
                for sym, p in self.symbols.items()
            },
            "quality_timeline": self.previous_quality[-30:],
            "total_adjustments": self.previous_adjustments,
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# Feedback Loop Engine
# ---------------------------------------------------------------------------


class TCAFeedbackLoop:
    """
    Reads TCA scorecard data and computes execution feedback adjustments.

    The feedback loop runs on a schedule (every rebalance cycle) and:
    1. Loads TCA scorecard data from tca_scorecard.json or runs on-demand
    2. Computes per-symbol execution profiles from historical quality data
    3. Maps quality to urgency offsets, min trade multipliers, cost calibration
    4. Persists feedback state for consumption by the execution layer
    5. Tracks quality trends over time to detect improvement/degradation
    """

    # Quality bucket thresholds
    EXCELLENT_MIN = 90
    GOOD_MIN = 70
    FAIR_MIN = 50
    POOR_MIN = 20
    BAD_MIN = 0

    # Default data directories
    DATA_DIR = Path(__file__).parent.parent.parent / "data"

    # Feedback state path
    STATE_PATH = DATA_DIR / "tca_feedback_state.json"

    # TCA scorecard path
    SCORECARD_PATH = DATA_DIR / "tca_scorecard.json"

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or self.DATA_DIR
        self.state_path = self.data_dir / "tca_feedback_state.json"
        self.scorecard_path = self.data_dir / "tca_scorecard.json"

        # Load existing state
        self.state = self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> FeedbackState:
        """Load existing feedback state from JSON."""
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                # Rehydrate symbols
                symbols = {}
                for sym, sd in data.get("symbols", {}).items():
                    # Access latest scorecard data for full profile
                    symbols[sym] = SymbolExecutionProfile(
                        symbol=sym,
                        total_orders=sd.get("total_orders", 0),
                        avg_slippage_bps=sd.get("avg_slippage_bps", 0.0),
                        avg_quality=sd.get("avg_quality", sd.get("quality_bucket_avg", 75.0)),
                        trend_slope=sd.get("trend_slope", 0.0),
                        recent_avg_quality=sd.get("recent_quality", sd.get("avg_quality", 75.0)),
                        slippage_volatility=sd.get("slippage_volatility", 1.0),
                        quality_bucket=sd.get("quality_bucket", "fair"),
                        urgency_offset=sd.get("urgency_offset", 0.0),
                        min_trade_multiplier=sd.get("min_trade_multiplier", 1.0),
                        cost_calibration_factor=sd.get("cost_calibration", 1.0),
                    )
                return FeedbackState(
                    version=data.get("version", "6.05"),
                    generated=data.get("generated", ""),
                    overall_quality=data.get("overall_quality", 75.0),
                    urgency_global_offset=data.get("urgency_global_offset", 0.0),
                    min_trade_global_multiplier=data.get("min_trade_global_multiplier", 1.0),
                    cost_calibration_global=data.get("cost_calibration_global", 1.0),
                    symbols=symbols,
                    previous_quality=data.get("quality_timeline", []),
                    previous_adjustments=data.get("total_adjustments", 0),
                    status=data.get("status", "ok"),
                )
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Could not load feedback state: {e}", file=sys.stderr)
        return FeedbackState(
            generated=datetime.now(timezone.utc).isoformat()
        )

    def _save_state(self) -> None:
        """Save current feedback state to JSON."""
        self.state.generated = datetime.now(timezone.utc).isoformat()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    # ------------------------------------------------------------------
    # TCA data loading
    # ------------------------------------------------------------------

    def _load_scorecard(self) -> Optional[dict]:
        """Load TCA scorecard data from cached JSON."""
        if self.scorecard_path.exists():
            try:
                with open(self.scorecard_path) as f:
                    data = json.load(f)
                if data.get("status") != "no_data" and data.get("total_orders", 0) > 0:
                    return data
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load scorecard: {e}", file=sys.stderr)

        # Fallback: try running scorecard directly
        try:
            from src.execution.tca_scorecard import TCAScorecard
            scorecard = TCAScorecard(data_dir=str(self.data_dir))
            report = scorecard.generate_daily_report(days=60)
            if report.get("status") != "no_data" and report.get("total_orders", 0) > 0:
                # Cache it
                with open(self.scorecard_path, "w") as f:
                    json.dump(report, f, indent=2)
                return report
        except Exception as e:
            print(f"Warning: Could not run scorecard: {e}", file=sys.stderr)

        return None

    def _fetch_symbol_from_scorecard(self, scorecard: dict,
                                      sym: str) -> Optional[dict]:
        """Get symbol-specific data from scorecard."""
        by_symbol = scorecard.get("by_symbol", {})
        peer_groups = scorecard.get("peer_groups", {})

        # Direct symbol data
        sym_data = by_symbol.get(sym)

        # Aggregate peer groups for this symbol
        sym_peers = [
            pg for key, pg in peer_groups.items()
            if pg.get("symbol") == sym
        ]

        return {
            "direct": sym_data,
            "peer_groups": sym_peers,
            "trend": scorecard.get("trend", {}),
        }

    # ------------------------------------------------------------------
    # Profile computation
    # ------------------------------------------------------------------

    def _compute_quality_bucket(self, quality: float) -> str:
        """Classify quality score into a bucket."""
        if quality >= self.EXCELLENT_MIN:
            return "excellent"
        elif quality >= self.GOOD_MIN:
            return "good"
        elif quality >= self.FAIR_MIN:
            return "fair"
        elif quality >= self.POOR_MIN:
            return "poor"
        else:
            return "bad"

    def _compute_slippage_volatility(
        self, scorecard: dict, symbol: str
    ) -> float:
        """Estimate slippage volatility from peer group data."""
        peer_groups = scorecard.get("peer_groups", {})
        volatilities = []
        for key, pg in peer_groups.items():
            if pg.get("symbol") == symbol:
                v = pg.get("std_slippage_bps", 0)
                if v > 0:
                    volatilities.append(v)
        if volatilities:
            return sum(volatilities) / len(volatilities)
        return 1.0

    def _urgency_from_quality(self, quality: float, trend_slope: float) -> float:
        """
        Map execution quality to urgency offset.

        Poor execution → negative offset (less urgent → wait for better windows)
        Good execution → slightly positive offset (maintain normal urgency)

        Range: -0.30 (max penalty) to +0.10 (small bonus)
        """
        if quality >= 90:
            return 0.05  # Slight bonus
        elif quality >= 70:
            return 0.0   # Neutral
        elif quality >= 50:
            return -0.10  # Small penalty
        elif quality >= 20:
            return -0.20  # Moderate penalty
        else:
            return -0.30  # Severe penalty

    def _min_trade_from_quality(self, quality: float, trend_slope: float) -> float:
        """
        Map execution quality to min trade multiplier.

        Poor execution → higher multiplier (only trade larger lots)
        Good execution → lower multiplier (small trades are fine)

        Range: 0.9 (best) to 3.0 (worst)
        """
        if quality >= 90:
            return 0.9
        elif quality >= 70:
            return 1.0
        elif quality >= 50:
            return 1.5
        elif quality >= 20:
            return 2.0
        else:
            return 3.0

    def _cost_calibration_from_quality(
        self, quality: float, trend_slope: float
    ) -> float:
        """
        Map execution quality to cost calibration factor.

        Poor execution → higher factor (actual costs exceed estimates)
        Good execution → factor near 1.0 (estimates are accurate)

        Range: 0.8 to 2.0
        """
        if quality >= 90:
            return 0.8  # Costs overestimated
        elif quality >= 70:
            return 1.0
        elif quality >= 50:
            return 1.3  # Costs slightly underestimated
        elif quality >= 20:
            return 1.6  # Costs significantly underestimated
        else:
            return 2.0  # Costs badly underestimated

    def _compute_symbol_profile(
        self, symbol: str, scorecard: dict
    ) -> Optional[SymbolExecutionProfile]:
        """Compute execution profile for a single symbol from scorecard data."""
        sym_data = self._fetch_symbol_from_scorecard(scorecard, symbol)
        direct = sym_data.get("direct")
        trend = sym_data.get("trend", {})
        peer_groups = sym_data.get("peer_groups", [])

        if not direct and not peer_groups:
            # No data for this symbol — use existing state if available
            existing = self.state.symbols.get(symbol)
            if existing:
                return existing
            return None

        # Gather quality metrics
        if direct:
            avg_quality = direct.get("quality", 75.0)
            avg_slippage = direct.get("slippage_bps", 0.0)
            total_orders = direct.get("count", 0)
        else:
            # Aggregate from peer groups
            qualities = [pg.get("mean_quality", 75.0) for pg in peer_groups]
            slippages = [pg.get("mean_slippage_bps", 0.0) for pg in peer_groups]
            counts = [pg.get("count", 0) for pg in peer_groups]
            avg_quality = sum(qualities) / len(qualities) if qualities else 75.0
            avg_slippage = sum(slippages) / len(slippages) if slippages else 0.0
            total_orders = sum(counts) if counts else 0

        # Trend data (from scorecard overall, or per-symbol if available)
        trend_slope = trend.get("slope", 0.0)
        recent_avg = trend.get("recent_avg", avg_quality)

        # Slippage volatility
        slippage_vol = self._compute_slippage_volatility(scorecard, symbol)

        # Quality bucket
        bucket = self._compute_quality_bucket(avg_quality)

        # Compute adjustments
        urgency_off = self._urgency_from_quality(avg_quality, trend_slope)
        min_trade_mult = self._min_trade_from_quality(avg_quality, trend_slope)
        cost_cal = self._cost_calibration_from_quality(avg_quality, trend_slope)

        return SymbolExecutionProfile(
            symbol=symbol,
            total_orders=total_orders,
            avg_slippage_bps=avg_slippage,
            avg_quality=avg_quality,
            trend_slope=trend_slope,
            recent_avg_quality=recent_avg,
            slippage_volatility=slippage_vol,
            quality_bucket=bucket,
            urgency_offset=urgency_off,
            min_trade_multiplier=min_trade_mult,
            cost_calibration_factor=cost_cal,
        )

    # ------------------------------------------------------------------
    # Global adjustments
    # ------------------------------------------------------------------

    def _compute_global_quality(self) -> float:
        """Compute overall execution quality across all symbols."""
        if not self.state.symbols:
            return 75.0
        # Weighted by number of orders per symbol
        total_orders = sum(
            p.total_orders for p in self.state.symbols.values()
        )
        if total_orders == 0:
            return 75.0
        weighted = sum(
            p.avg_quality * p.total_orders
            for p in self.state.symbols.values()
        ) / total_orders
        return weighted

    def _compute_global_adjustments(self) -> Tuple[float, float, float]:
        """
        Compute global (aggregate) adjustment factors.

        Returns:
            (urgency_global_offset, min_trade_global_mult, cost_calibration_global)
        """
        if not self.state.symbols:
            return (0.0, 1.0, 1.0)

        # Average of per-symbol adjustments, weighted by order count
        total_orders = sum(p.total_orders for p in self.state.symbols.values())
        if total_orders == 0:
            return (0.0, 1.0, 1.0)

        urgency_sum = sum(
            p.urgency_offset * p.total_orders
            for p in self.state.symbols.values()
        )
        trade_sum = sum(
            p.min_trade_multiplier * p.total_orders
            for p in self.state.symbols.values()
        )
        cost_sum = sum(
            p.cost_calibration_factor * p.total_orders
            for p in self.state.symbols.values()
        )

        return (
            urgency_sum / total_orders / 2,  # Scale down global urgency offset
            trade_sum / total_orders,
            cost_sum / total_orders,
        )

    # ------------------------------------------------------------------
    # Main feedback generation
    # ------------------------------------------------------------------

    def generate_feedback(self, scorecard: Optional[dict] = None) -> FeedbackState:
        """
        Generate feedback adjustments from TCA data.

        Args:
            scorecard: Pre-loaded scorecard data. If None, tries to load.

        Returns:
            Updated FeedbackState with computed adjustments.
        """
        if scorecard is None:
            scorecard = self._load_scorecard()

        if scorecard is None:
            # No data available — return current state as-is
            self.state.status = "no_tca_data"
            self._save_state()
            return self.state

        # Get known symbols from scorecard
        known_symbols = list((scorecard or {}).get("by_symbol", {}).keys())

        # If no symbols in scorecard, try peer groups
        if not known_symbols:
            peer_groups = scorecard.get("peer_groups", {})
            known_symbols = list(set(
                pg.get("symbol") for pg in peer_groups.values()
            ))

        # Default symbols if nothing found
        if not known_symbols:
            known_symbols = list(self.state.symbols.keys())

        # Also include our known symbols for continuity
        all_symbols = list(set(
            known_symbols + list(self.state.symbols.keys())
        ))

        # Compute profiles for all symbols
        new_profiles = {}
        for sym in all_symbols:
            profile = self._compute_symbol_profile(sym, scorecard)
            if profile:
                new_profiles[sym] = profile

        # Update state
        self.state.symbols = new_profiles

        # Compute global metrics
        self.state.overall_quality = self._compute_global_quality()
        urg_global, trade_global, cost_global = self._compute_global_adjustments()
        self.state.urgency_global_offset = urg_global
        self.state.min_trade_global_multiplier = trade_global
        self.state.cost_calibration_global = cost_global

        # Track quality timeline
        self.state.previous_quality.append(self.state.overall_quality)
        if len(self.state.previous_quality) > 60:
            self.state.previous_quality = self.state.previous_quality[-60:]

        self.state.previous_adjustments += 1
        self.state.status = "active"

        # Persist
        self._save_state()

        return self.state

    def get_adjustments(self) -> Dict[str, Any]:
        """
        Get current feedback adjustments for consumption by execution layer.

        Returns a dict with:
        - urgency_offsets: Dict[symbol, offset_value]
        - min_trade_multipliers: Dict[symbol, multiplier]
        - cost_calibration_factors: Dict[symbol, factor]
        - global_urgency_offset: float
        - global_min_trade_multiplier: float
        - global_cost_calibration: float
        - overall_quality: float
        """
        if not self.state.symbols:
            return {
                "urgency_offsets": {},
                "min_trade_multipliers": {},
                "cost_calibration_factors": {},
                "global_urgency_offset": 0.0,
                "global_min_trade_multiplier": 1.0,
                "global_cost_calibration": 1.0,
                "overall_quality": 75.0,
                "status": "no_data",
            }

        return {
            "urgency_offsets": {
                sym: p.urgency_offset for sym, p in self.state.symbols.items()
            },
            "min_trade_multipliers": {
                sym: p.min_trade_multiplier for sym, p in self.state.symbols.items()
            },
            "cost_calibration_factors": {
                sym: p.cost_calibration_factor
                for sym, p in self.state.symbols.items()
            },
            "global_urgency_offset": self.state.urgency_global_offset,
            "global_min_trade_multiplier": self.state.min_trade_global_multiplier,
            "global_cost_calibration": self.state.cost_calibration_global,
            "overall_quality": self.state.overall_quality,
            "total_adjustments": self.state.previous_adjustments,
            "status": self.state.status,
        }

    def reset(self) -> None:
        """Reset feedback state to defaults."""
        self.state = FeedbackState(
            generated=datetime.now(timezone.utc).isoformat()
        )
        self._save_state()

    def print_summary(self) -> str:
        """Generate a human-readable summary of feedback state."""
        lines = []
        lines.append("=" * 60)
        lines.append("  TCA-TO-EXECUTION FEEDBACK LOOP (v6.05)")
        lines.append("=" * 60)

        lines.append(f"  Status:            {self.state.status}")
        lines.append(f"  Overall Quality:   {self.state.overall_quality:.1f}/100")
        lines.append(f"  Total Adjustments: {self.state.previous_adjustments}")
        lines.append(f"  Quality Timeline:  {len(self.state.previous_quality)} points")

        # Global adjustments
        lines.append("")
        lines.append("  ── Global Adjustments ──")
        lines.append(f"    Urgency Offset:      {self.state.urgency_global_offset:+.3f}")
        lines.append(f"    Min Trade Multiplier: {self.state.min_trade_global_multiplier:.2f}x")
        lines.append(f"    Cost Calibration:     {self.state.cost_calibration_global:.2f}x")

        # Per-symbol details
        if self.state.symbols:
            lines.append("")
            lines.append("  ── Per-Symbol Adjustments ──")
            lines.append(f"  {'Symbol':6s} {'Bucket':12s} {'Quality':>8s} {'Trend':>7s} "
                         f"{'U Off':>6s} {'M Trade':>8s} {'Cost Cal':>8s}")
            lines.append(f"  {'------':6s} {'------------':12s} {'-------':>8s} "
                         f"{'-----':>7s} {'-----':>6s} {'-------':>8s} {'-------':>8s}")
            for sym in sorted(self.state.symbols.keys()):
                p = self.state.symbols[sym]
                trend_dir = "↑" if p.trend_slope > 0.1 else \
                           "↓" if p.trend_slope < -0.1 else "→"
                lines.append(
                    f"  {sym:6s} {p.quality_bucket:12s} "
                    f"{p.avg_quality:>6.1f}/100 "
                    f" {trend_dir}{p.trend_slope:>+5.2f} "
                    f"{p.urgency_offset:>+5.2f} "
                    f"{p.min_trade_multiplier:>6.2f}x "
                    f"{p.cost_calibration_factor:>6.2f}x"
                )

        # Meaning
        lines.append("")
        lines.append("  ── Interpretation ──")
        lines.append("  Urgency Offset:    Negative = less urgent (wait for better windows)")
        lines.append("  Min Trade Mult:    >1.0 = require larger trades")
        lines.append("  Cost Calibration:  >1.0 = actual costs exceed estimates")
        lines.append("")

        # Recommendation
        if self.state.overall_quality >= 85:
            lines.append("  > EXCELLENT: Execution quality is strong. Maintain current parameters.")
        elif self.state.overall_quality >= 65:
            lines.append("  > GOOD: Execution quality is acceptable. Minor adjustments applied.")
        elif self.state.overall_quality >= 40:
            lines.append("  > FAIR: Some execution concerns. Adjustments applied to tighten.")
        elif self.state.overall_quality >= 15:
            lines.append("  > POOR: Significant execution issues. Consider reviewing broker/execution.")
        else:
            lines.append("  > CRITICAL: Execution quality is severely degraded. Halt trading.")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Integration: Apply feedback to SignalExecutionBridge
# ---------------------------------------------------------------------------


def apply_urgency_adjustment(
    base_score: float,
    base_confidence: float,
    symbol: str,
    feedback: Dict[str, Any],
    regime: str = "neutral",
) -> float:
    """
    Apply TCA feedback urgency adjustment to a signal score.

    Call this from SignalExecutionBridge._calculate_urgency() to get
    a TCA-adjusted combined score before threshold comparison.

    Args:
        base_score: Original composite score from signal.
        base_confidence: Original confidence from signal.
        symbol: Asset symbol for per-symbol adjustment.
        feedback: Output of TCAFeedbackLoop.get_adjustments().
        regime: Current market regime.

    Returns:
        Adjusted combined score (urgency threshold input).
    """
    combined = (abs(base_score) + base_confidence) / 2

    # Apply symbol-level urgency offset
    urgency_offset = feedback.get("urgency_offsets", {}).get(symbol, 0.0)

    # Apply global urgency offset
    global_offset = feedback.get("global_urgency_offset", 0.0)

    # Total offset (symbol + global)
    total_offset = urgency_offset + global_offset

    return combined + total_offset


def apply_min_trade_adjustment(
    min_trade_value: float,
    symbol: str,
    feedback: Dict[str, Any],
) -> float:
    """
    Apply TCA feedback min trade value adjustment.

    Call this from SignalExecutionBridge._deltas_to_orders() to get
    an adjusted minimum trade value.

    Args:
        min_trade_value: Base minimum trade value (e.g. $1000).
        symbol: Asset symbol.
        feedback: Output of TCAFeedbackLoop.get_adjustments().

    Returns:
        Adjusted minimum trade value.
    """
    symbol_mult = feedback.get("min_trade_multipliers", {}).get(symbol, 1.0)
    global_mult = feedback.get("global_min_trade_multiplier", 1.0)
    return min_trade_value * symbol_mult * global_mult


def apply_cost_calibration(
    estimated_cost_bps: float,
    symbol: str,
    feedback: Dict[str, Any],
) -> float:
    """
    Apply TCA feedback cost calibration to estimated execution costs.

    Call this to adjust cost estimates before reporting.

    Args:
        estimated_cost_bps: Base estimated cost in bps.
        symbol: Asset symbol.
        feedback: Output of TCAFeedbackLoop.get_adjustments().

    Returns:
        Calibrated cost estimate in bps.
    """
    symbol_cal = feedback.get("cost_calibration_factors", {}).get(symbol, 1.0)
    global_cal = feedback.get("global_cost_calibration", 1.0)
    return estimated_cost_bps * symbol_cal * global_cal


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="TCA-to-Execution Feedback Loop v6.05"
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # Check
    sub.add_parser("check", help="Check current feedback state")

    # Adjust (generate new feedback)
    adjust_parser = sub.add_parser("adjust", help="Generate new feedback adjustments")
    adjust_parser.add_argument("--save", action="store_true",
                                help="Save adjustments to state file")

    # Status
    sub.add_parser("status", help="Print detailed status")

    # Reset
    sub.add_parser("reset", help="Reset feedback state to defaults")

    args = parser.parse_args()

    loop = TCAFeedbackLoop()

    if args.command == "check":
        feedback = loop.get_adjustments()
        print(f"TCA Feedback Loop Status: {feedback['status']}")
        print(f"Overall Quality: {feedback['overall_quality']:.1f}/100")
        print(f"Total Adjustments: {feedback.get('total_adjustments', 0)}")
        print(f"Urgency Offsets: {len(feedback['urgency_offsets'])} symbols")
        print(f"Min Trade Mult: {len(feedback['min_trade_multipliers'])} symbols")
        print(f"Cost Calibration: {len(feedback['cost_calibration_factors'])} symbols")

    elif args.command == "adjust":
        state = loop.generate_feedback()
        print(loop.print_summary())

    elif args.command == "status":
        print(loop.print_summary())

    elif args.command == "reset":
        loop.reset()
        print("Feedback state reset to defaults.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
