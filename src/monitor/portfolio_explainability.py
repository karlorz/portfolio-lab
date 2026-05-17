#!/usr/bin/env python3
"""
v8.07: Portfolio Explainability Dashboard

Answers "Why did the portfolio do what it did?" by generating decision
provenance reports: per-signal-source contribution breakdowns for every
allocation decision, plus a today-mode that explains current positioning.

Builds on top of:
- PerformanceAttribution (v5.70) — per-source hit rates & contributions
- UnifiedDashboard (v6.08) — state file aggregation
- EnsembleVoter decision logs — consensus & reasoning

Usage:
    python -m src.monitor.portfolio_explainability explain     # Latest decision
    python -m src.monitor.portfolio_explainability today        # Current positioning
    python -m src.monitor.portfolio_explainability signal <src> # Signal deep-dive
    python -m src.monitor.portfolio_explainability history      # Last 5 decisions
    python -m src.monitor.portfolio_explainability all          # Full report
"""

import json
import logging
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()

# ─────────────────────────────────────────────
#  Data Models
# ─────────────────────────────────────────────


@dataclass
class SignalContribution:
    """A single signal's contribution to a decision moment."""

    source: str
    display_name: str
    category: str
    value: float  # Signal value (-1 to +1)
    confidence: float  # 0 to 1
    weight: float  # Ensemble weight
    regime_fit: str  # How well signal fits current regime
    explanation: str  # Human-readable rationale
    contribution_pct: float  # % of total directional push
    is_tiebreaking: bool  # Was this signal the tiebreaker?


@dataclass
class DecisionExplanation:
    """Complete explanation for a single portfolio decision."""

    timestamp: str
    period: str  # e.g. "2026-05-17"
    regime: str  # Market regime at decision time
    action: str  # e.g. "rebalance", "hold", "hedge"
    confidence: float
    reasoning: str  # Aggregate reasoning text

    # Asset allocation changes
    asset_changes: Dict[str, float]  # {asset: delta_pct}
    current_allocation: Dict[str, float]  # {asset: target_pct}

    # Signal breakdown
    total_signals: int
    consensus_direction: str  # "bullish", "bearish", "neutral"
    agreement_ratio: float
    signals: List[SignalContribution]

    # Top drivers
    top_drivers: List[str]  # Top 3 signal sources driving this decision
    top_opposers: List[str]  # Top 3 signals opposing

    # Attribution context
    attribution_summary: Optional[Dict[str, Any]] = None


@dataclass
class SignalDeepDive:
    """Deep-dive analysis of a single signal source over time."""

    source: str
    display_name: str
    category: str
    total_observations: int
    avg_value: float
    avg_confidence: float
    avg_weight: float
    hit_rate: float
    sharpe_contribution: float
    avg_return_bps: float
    recent_trend: str  # "improving", "degrading", "stable"
    regime_fit_distribution: Dict[str, int]
    correlation_with_peers: float


@dataclass
class ExplainabilityReport:
    """Complete explainability report."""

    timestamp: str
    analysis_date: str
    latest_decision: Optional[DecisionExplanation] = None
    recent_decisions: List[DecisionExplanation] = field(default_factory=list)
    signal_deep_dives: Dict[str, SignalDeepDive] = field(default_factory=dict)
    top_sources_today: List[str] = field(default_factory=list)
    decision_quality: str = "unknown"


# ─────────────────────────────────────────────
#  Data Loaders
# ─────────────────────────────────────────────


def _read_json(path: str) -> Optional[Any]:
    """Safely read a JSON file."""
    full_path = DATA_DIR / path if not path.startswith("/") else Path(path)
    if not full_path.exists():
        return None
    try:
        with open(full_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def _get_latest_ensemble_votes(n: int = 5) -> List[Dict[str, Any]]:
    """Get latest ensemble votes from SQLite DB."""
    db_path = DATA_DIR / "ensemble_signals.db"
    if not db_path.exists():
        logger.warning("Ensemble DB not found: %s", db_path)
        return []

    votes = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT timestamp, regime, consensus, agreement_ratio,
                       equity_bias, duration_bias, gold_bias,
                       action, confidence, reasoning
                FROM ensemble_votes
                ORDER BY timestamp DESC
                LIMIT ?
            """, (n,))

            for row in cursor.fetchall():
                votes.append({
                    "timestamp": row["timestamp"],
                    "regime": row["regime"],
                    "consensus": row["consensus"],
                    "agreement_ratio": row["agreement_ratio"],
                    "equity_bias": row["equity_bias"],
                    "duration_bias": row["duration_bias"],
                    "gold_bias": row["gold_bias"],
                    "action": row["action"],
                    "confidence": row["confidence"],
                    "reasoning": row["reasoning"],
                })
    except Exception as e:
        logger.error("Error reading ensemble votes: %s", e)

    return votes


def _get_source_readings_for_vote(vote_ts: str) -> List[Dict[str, Any]]:
    """Get source readings matching a vote timestamp."""
    db_path = DATA_DIR / "ensemble_signals.db"
    if not db_path.exists():
        return []

    readings = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Match within 5-minute window of vote timestamp
            cursor = conn.cursor()
            cursor.execute("""
                SELECT source, value, confidence, weight, regime_fit, explanation
                FROM source_readings
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY ABS(weight) DESC
            """, (vote_ts, vote_ts[:19] + "Z"))

            for row in cursor.fetchall():
                readings.append({
                    "source": row["source"],
                    "value": row["value"],
                    "confidence": row["confidence"],
                    "weight": row["weight"],
                    "regime_fit": row["regime_fit"],
                    "explanation": row["explanation"],
                })
    except Exception as e:
        logger.debug("No source readings for %s: %s", vote_ts, e)

    return readings


def _get_all_source_readings(limit: int = 5000) -> Dict[str, List[Dict]]:
    """Get all source readings grouped by source."""
    db_path = DATA_DIR / "ensemble_signals.db"
    if not db_path.exists():
        return {}

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT source, value, confidence, weight, regime_fit, explanation, timestamp
                FROM source_readings
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            for row in cursor.fetchall():
                grouped[row["source"]].append({
                    "source": row["source"],
                    "value": row["value"],
                    "confidence": row["confidence"],
                    "weight": row["weight"],
                    "regime_fit": row["regime_fit"],
                    "explanation": row["explanation"],
                    "timestamp": row["timestamp"],
                })
    except Exception as e:
        logger.warning("Could not read source readings: %s", e)

    return dict(grouped)


def _get_current_allocation() -> Dict[str, float]:
    """Get current portfolio target allocation from position state."""
    position_state = _read_json("position_state.json")
    if position_state and "target_allocation" in position_state:
        return position_state["target_allocation"]

    # Fallback: check paper trading state
    paper_state = _read_json("paper_trading_state.json")
    if paper_state and "positions" in paper_state:
        alloc = {}
        for sym, pos in paper_state["positions"].items():
            alloc[sym] = pos.get("target_pct", 0)
        return alloc

    return {}


def _get_latest_attribution() -> Optional[Dict[str, Any]]:
    """Get latest performance attribution report."""
    attr_dir = DATA_DIR / "attribution"
    if not attr_dir.exists():
        return None
    files = sorted(attr_dir.glob("attribution_*.json"), reverse=True)
    if not files:
        return None
    return _read_json(str(files[0]))


SIGNAL_SOURCE_META = {
    "tsfm_momentum": {"name": "TSFM Factor Momentum", "category": "trend"},
    "hmm_regime": {"name": "HMM Regime Detector", "category": "regime"},
    "cta_trend": {"name": "CTA Trend Overlay", "category": "trend"},
    "macro_momentum": {"name": "Macro Momentum", "category": "macro"},
    "multi_speed_momentum": {"name": "Multi-Speed Momentum", "category": "trend"},
    "duration_regime": {"name": "Duration/Yield Curve", "category": "rates"},
    "circuit_breaker": {"name": "Circuit Breaker", "category": "risk"},
    "factor_rotation": {"name": "Factor Rotation", "category": "factor"},
    "closing_auction": {"name": "Closing Auction MOC", "category": "execution"},
    "unified_overlay": {"name": "Unified Overlay", "category": "orchestration"},
    "mean_reversion": {"name": "Mean Reversion", "category": "meanrev"},
    "transformer_regime": {"name": "Transformer Regime", "category": "regime"},
    "transient_factors": {"name": "Transient Factors", "category": "factor"},
    "visibility_graph": {"name": "Visibility Graph (VGRSI)", "category": "network"},
    "vp_macd": {"name": "VP-MACD", "category": "momentum"},
    "cross_asset_rv": {"name": "Cross-Asset RV", "category": "meanrev"},
    "multi_timeframe_fusion": {"name": "Multi-Timeframe Fusion", "category": "fusion"},
    "behavioral_sentiment": {"name": "Behavioral Sentiment", "category": "sentiment"},
    "llm_narrative": {"name": "LLM Narrative Signal", "category": "sentiment"},
    "vix_hedge": {"name": "VIXY Hedge", "category": "hedging"},
    "bond_duration_rotation": {"name": "Bond Duration Rotation", "category": "rates"},
    "crypto_tactical": {"name": "Crypto Tactical", "category": "crypto"},
    "collar_overlay": {"name": "Cashless Collar", "category": "options"},
    "calendar_seasonality": {"name": "Calendar Seasonality", "category": "calendar"},
    "fed_policy": {"name": "Fed Policy Overlay", "category": "macro"},
}

# Category grouping for display
CATEGORY_EMOJI = {
    "trend": "📈",
    "regime": "🌦",
    "macro": "🌍",
    "rates": "💰",
    "risk": "🛡",
    "factor": "📊",
    "execution": "⚡",
    "orchestration": "🎯",
    "meanrev": "🔄",
    "network": "🕸",
    "momentum": "🚀",
    "fusion": "🔬",
    "sentiment": "🧠",
    "hedging": "🛡",
    "crypto": "₿",
    "options": "📐",
    "calendar": "📅",
    "other": "❓",
}


# ─────────────────────────────────────────────
#  Explainability Engine
# ─────────────────────────────────────────────


class PortfolioExplainability:
    """Generate decision provenance and explainability reports."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

    def explain_latest_decision(self) -> Optional[DecisionExplanation]:
        """Generate explanation for the most recent portfolio decision."""
        votes = _get_latest_ensemble_votes(1)
        if not votes:
            logger.warning("No ensemble votes found to explain.")
            return None
        return self._explain_vote(votes[0])

    def explain_recent_decisions(self, n: int = 5) -> List[DecisionExplanation]:
        """Generate explanations for the last N decisions."""
        votes = _get_latest_ensemble_votes(n)
        explanations = [self._explain_vote(v) for v in votes if v]
        return [e for e in explanations if e is not None]

    def _explain_vote(self, vote: Dict[str, Any]) -> Optional[DecisionExplanation]:
        """Build a DecisionExplanation for a single ensemble vote."""
        ts = vote.get("timestamp", datetime.now().isoformat())
        period = ts[:10]

        # Get source readings matching this vote
        readings = _get_source_readings_for_vote(ts)
        if not readings:
            readings = self._get_fallback_readings(period)

        # Get allocation context
        current_allocation = _get_current_allocation()
        asset_changes = self._estimate_asset_changes(vote, current_allocation)

        # Build signal contributions
        signals = self._build_signal_contributions(readings)

        # Calculate consensus direction and agreement
        direction_vals = [s.value for s in signals if abs(s.value) > 0.05]
        consensus_dir = "neutral"
        if direction_vals:
            mean_dir = np.mean(direction_vals)
            if mean_dir > 0.15:
                consensus_dir = "bullish"
            elif mean_dir < -0.15:
                consensus_dir = "bearish"

        # Find top drivers and opposers
        sorted_by_contribution = sorted(
            signals, key=lambda s: abs(s.contribution_pct), reverse=True
        )
        top_drivers = [
            s.display_name for s in sorted_by_contribution[:3]
            if s.contribution_pct > 0
        ]
        top_opposers = [
            s.display_name for s in sorted_by_contribution[:3]
            if s.contribution_pct < 0
        ]

        # Attribution context
        latest_attr = _get_latest_attribution()

        return DecisionExplanation(
            timestamp=ts,
            period=period,
            regime=vote.get("regime", "unknown"),
            action=vote.get("action", "hold"),
            confidence=vote.get("confidence", 0.5),
            reasoning=vote.get("reasoning", ""),
            asset_changes=asset_changes,
            current_allocation=current_allocation,
            total_signals=len(signals),
            consensus_direction=consensus_dir,
            agreement_ratio=vote.get("agreement_ratio", 0.0),
            signals=signals,
            top_drivers=top_drivers,
            top_opposers=top_opposers,
            attribution_summary=latest_attr,
        )

    def _get_fallback_readings(self, period: str) -> List[Dict[str, Any]]:
        """Fallback: get any readings from the same day."""
        all_readings = _get_all_source_readings(100)
        fallback = []
        for src, readings in all_readings.items():
            day_readings = [r for r in readings if r.get("timestamp", "").startswith(period)]
            if day_readings:
                fallback.append(day_readings[0])
        return fallback

    def _build_signal_contributions(
        self, readings: List[Dict[str, Any]]
    ) -> List[SignalContribution]:
        """Build SignalContribution list from raw readings."""
        if not readings:
            return []

        # Normalize values
        abs_values = [abs(float(r.get("value", 0))) for r in readings if r.get("value")]
        total_push = sum(abs_values) if abs_values else 1.0

        signals = []
        for r in readings:
            src = r.get("source", "unknown")
            meta = SIGNAL_SOURCE_META.get(src, {"name": src, "category": "other"})
            value = float(r.get("value", 0))
            confidence = float(r.get("confidence", 0.5))
            weight = float(r.get("weight", 0))

            cont_pct = (abs(value) / total_push) * 100 if total_push > 0 else 0
            is_tiebreaking = cont_pct > 30 and len(readings) > 1

            signals.append(SignalContribution(
                source=src,
                display_name=meta["name"],
                category=meta.get("category", "other"),
                value=value,
                confidence=confidence,
                weight=weight,
                regime_fit=r.get("regime_fit", "neutral"),
                explanation=r.get("explanation", ""),
                contribution_pct=round(cont_pct, 1),
                is_tiebreaking=is_tiebreaking,
            ))

        return sorted(signals, key=lambda s: abs(s.contribution_pct), reverse=True)

    def _estimate_asset_changes(
        self, vote: Dict[str, Any], allocation: Dict[str, float]
    ) -> Dict[str, float]:
        """Estimate asset allocation changes implied by the vote."""
        changes = {}
        for key, label in [("equity_bias", "SPY"), ("gold_bias", "GLD"),
                           ("duration_bias", "TLT")]:
            bias = vote.get(key, 0)
            if isinstance(bias, (int, float)) and bias != 0:
                changes[label] = round(float(bias), 2)
        return changes

    def signal_deep_dive(self, source_name: str) -> Optional[SignalDeepDive]:
        """Generate deep-dive analysis for a single signal source."""
        # Search by source key or display name
        source_key = None
        for key, meta in SIGNAL_SOURCE_META.items():
            if source_name.lower() in key.lower() or source_name.lower() in meta["name"].lower():
                source_key = key
                break

        if not source_key:
            logger.warning("Unknown signal source: %s", source_name)
            return None

        all_readings = _get_all_source_readings(5000)
        readings = all_readings.get(source_key, [])
        if not readings:
            logger.warning("No readings found for signal: %s (%s)", source_key, source_name)
            return None

        meta = SIGNAL_SOURCE_META.get(source_key, {"name": source_key, "category": "other"})
        values = []
        confidences = []
        weights = []
        regime_fits: Dict[str, int] = defaultdict(int)

        # Sort by timestamp descending for trend analysis (most recent first)
        readings_sorted = sorted(readings, key=lambda r: r.get("timestamp", ""), reverse=True)

        for r in readings_sorted:
            v = float(r.get("value", 0))
            values.append(v)
            confidences.append(float(r.get("confidence", 0.5)))
            weights.append(float(r.get("weight", 0)))
            regime_fits[r.get("regime_fit", "neutral")] += 1

        avg_val = float(np.mean(values)) if values else 0.0
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        avg_w = float(np.mean(weights)) if weights else 0.0

        # Trend detection
        # Split into recent (last 20%) vs older (first 80%)
        n = len(values)
        if n >= 20:
            recent_vals = values[:n // 5]
            older_vals = values[n // 5:]
            recent_avg = float(np.mean(recent_vals))
            older_avg = float(np.mean(older_vals))
            if recent_avg > older_avg * 1.1:
                trend = "improving"
            elif recent_avg < older_avg * 0.9:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        # Get attribution data for hit rate and sharpe
        attr = _get_latest_attribution()
        hit_rate = 0.0
        sharpe_contrib = 0.0
        avg_return = 0.0
        if attr and "sources" in attr:
            src_data = attr["sources"].get(source_key, {})
            hit_rate = src_data.get("hit_rate", 0.0)
            sharpe_contrib = src_data.get("sharpe_contribution", 0.0)
            avg_return = src_data.get("avg_return_bps", 0.0)

        return SignalDeepDive(
            source=source_key,
            display_name=meta["name"],
            category=meta.get("category", "other"),
            total_observations=len(readings),
            avg_value=round(avg_val, 4),
            avg_confidence=round(avg_conf, 4),
            avg_weight=round(avg_w, 4),
            hit_rate=round(hit_rate, 4),
            sharpe_contribution=round(sharpe_contrib, 4),
            avg_return_bps=round(avg_return, 2),
            recent_trend=trend,
            regime_fit_distribution=dict(regime_fits),
            correlation_with_peers=0.0,
        )

    def generate_report(self) -> ExplainabilityReport:
        """Generate complete explainability report."""
        now = datetime.now()

        latest = self.explain_latest_decision()
        recent = self.explain_recent_decisions(5)

        # Deep-dive top sources
        top_sources = []
        signal_dives = {}
        if latest:
            for sig in latest.signals[:5]:
                dive = self.signal_deep_dive(sig.source)
                if dive:
                    signal_dives[sig.source] = dive
                    top_sources.append(sig.display_name)

        return ExplainabilityReport(
            timestamp=now.isoformat(),
            analysis_date=now.strftime("%Y-%m-%d"),
            latest_decision=latest,
            recent_decisions=recent,
            signal_deep_dives=signal_dives,
            top_sources_today=top_sources,
        )

    def to_json(self, report: ExplainabilityReport) -> str:
        """Serialize report to JSON."""
        return json.dumps(asdict(report), indent=2, default=str)

    def save_report(self, report: ExplainabilityReport) -> Path:
        """Save report to disk."""
        output_dir = self.data_dir / "explainability"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"explainability_{report.analysis_date}.json"
        with open(path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        logger.info("Saved explainability report: %s", path)
        return path


# ─────────────────────────────────────────────
#  CLI Output
# ─────────────────────────────────────────────


def print_explain(decision: DecisionExplanation):
    """Pretty-print a single decision explanation."""
    print()
    print("=" * 74)
    print(f"  📋 DECISION EXPLANATION — {decision.period}")
    print("=" * 74)
    print(f"  Time:      {decision.timestamp}")
    print(f"  Regime:    {decision.regime}")
    print(f"  Action:    {decision.action.upper()}")
    print(f"  Confidence: {decision.confidence:.0%}")
    print(f"  Consensus: {decision.consensus_direction} (agreement {decision.agreement_ratio:.0%})")
    print()

    # Reasoning
    if decision.reasoning:
        print(f"  📝 Reasoning:")
        print(f"     {decision.reasoning[:300]}")
        print()

    # Asset changes
    if decision.asset_changes:
        print(f"  📊 Allocation Shifts:")
        for asset, delta in decision.asset_changes.items():
            arrow = "↑" if delta > 0 else "↓"
            print(f"     {asset}: {arrow} {abs(delta):+.1f}%")
        print()

    # Current allocation
    if decision.current_allocation:
        print(f"  🎯 Current Allocation:")
        for asset, pct in sorted(decision.current_allocation.items(),
                                  key=lambda x: -x[1]):
            print(f"     {asset}: {pct:.1f}%")
        print()

    # Signal breakdown
    print(f"  🔍 Signal Breakdown ({decision.total_signals} sources):")
    print(f"  {'Source':32} {'Value':>7} {'Conf':>6} {'Weight':>7} {'Push':>6} {'Regime Fit':>12}")
    print(f"  {'-'*68}")
    for sig in decision.signals:
        emoji = CATEGORY_EMOJI.get(sig.category, "❓")
        tie = " ⚡" if sig.is_tiebreaking else ""
        extra = f" ★" if sig.display_name in decision.top_drivers else ""
        val_str = f"{sig.value:+.2f}"
        print(f"  {emoji} {sig.display_name:30}{val_str:>7} {sig.confidence:.0%}  "
              f"{sig.weight:.2f}  {sig.contribution_pct:5.1f}% {sig.regime_fit:>12}{tie}{extra}")
    print()

    # Top drivers and opposers
    if decision.top_drivers:
        print(f"  🚀 Top Drivers: {', '.join(decision.top_drivers)}")
    if decision.top_opposers:
        print(f"  🛑 Top Opposers: {', '.join(decision.top_opposers)}")
    print()

    # Attribution context
    if decision.attribution_summary:
        srcs = decision.attribution_summary.get("sources", {})
        top_srcs = sorted(srcs.items(),
                          key=lambda x: x[1].get("sharpe_contribution", 0) if isinstance(x[1], dict) else 0,
                          reverse=True)[:3]
        if top_srcs:
            print(f"  📈 Top Attribution (Sharpe Contribution):")
            for src_key, src_data in top_srcs:
                if isinstance(src_data, dict):
                    print(f"     {src_key:25} Sharpe={src_data.get('sharpe_contribution',0):+.2f} "
                          f"Hit={src_data.get('hit_rate',0):.1%}")
    print("=" * 74)
    print()


def print_today(explainer: PortfolioExplainability):
    """Print current portfolio positioning explanation."""
    report = explainer.generate_report()
    decision = report.latest_decision

    print()
    print("=" * 74)
    print(f"  🗓️  TODAY'S PORTFOLIO EXPLAINER — {report.analysis_date}")
    print("=" * 74)

    # Current allocation
    alloc = _get_current_allocation()
    if alloc:
        print(f"\n  🎯 Current Target Allocation:")
        for asset, pct in sorted(alloc.items(), key=lambda x: -x[1]):
            bar = "█" * int(pct / 2) + "░" * max(0, 25 - int(pct / 2))
            print(f"     {asset:6} {pct:5.1f}%  {bar}")
        print()

    if decision:
        print(f"  📋 Latest Decision:")
        print(f"     Action:      {decision.action.upper()}")
        print(f"     Regime:      {decision.regime}")
        print(f"     Confidence:  {decision.confidence:.0%}")
        print(f"     Consensus:   {decision.consensus_direction} "
              f"(agreement {decision.agreement_ratio:.0%})")
        print(f"     Top Drivers: {', '.join(decision.top_drivers[:3])}")

        # Signal activity heat
        print(f"\n  🔥 Signal Activity Heatmap:")
        categories_seen = set()
        for sig in decision.signals:
            if sig.category not in categories_seen:
                emoji = CATEGORY_EMOJI.get(sig.category, "❓")
                cat_name = sig.category.replace("_", " ").title()
                group = [s for s in decision.signals if s.category == sig.category]
                avg_val = np.mean([s.value for s in group])
                direction = "🟢" if avg_val > 0.05 else ("🔴" if avg_val < -0.05 else "⚪")
                signals_str = ", ".join(f"{s.display_name}({s.value:+.2f})" for s in group[:3])
                print(f"     {emoji} {cat_name:15} {direction} {signals_str}")
                categories_seen.add(sig.category)
    else:
        print("\n  ⚠️  No recent decision data available.")

    # Signal health
    if report.signal_deep_dives:
        print(f"\n  📊 Signal Health:")
        print(f"  {'Source':28} {'Obs':>4} {'AvgVal':>8} {'Conf':>6} {'HitRate':>9} {'Trend':>12}")
        print(f"  {'-'*67}")
        for src_key, dive in report.signal_deep_dives.items():
            trend_emoji = {"improving": "📈", "degrading": "📉", "stable": "➡️",
                           "insufficient_data": "❓"}.get(dive.recent_trend, "❓")
            print(f"  {dive.display_name:28}{dive.total_observations:>4d} "
                  f"{dive.avg_value:>8.2f} {dive.avg_confidence:.0%}  "
                  f"{dive.hit_rate:.1%}   {trend_emoji} {dive.recent_trend:>10}")
        print()

    print("=" * 74)
    print()


def print_signal_deep_dive(dive: SignalDeepDive):
    """Pretty-print a signal deep-dive."""
    emoji = CATEGORY_EMOJI.get(dive.category, "❓")
    trend_emoji = {"improving": "📈", "degrading": "📉", "stable": "➡️",
                   "insufficient_data": "❓"}.get(dive.recent_trend, "❓")

    print()
    print("=" * 74)
    print(f"  {emoji} SIGNAL DEEP DIVE — {dive.display_name}")
    print("=" * 74)
    print(f"  Source Key:  {dive.source}")
    print(f"  Category:    {dive.category}")
    print(f"  Observations: {dive.total_observations}")
    print(f"  Avg Value:   {dive.avg_value:+.4f}")
    print(f"  Avg Confidence: {dive.avg_confidence:.1%}")
    print(f"  Avg Weight:  {dive.avg_weight:.4f}")
    print()
    print(f"  📊 Performance:")
    print(f"     Hit Rate:        {dive.hit_rate:.1%}")
    print(f"     Sharpe Contrib:  {dive.sharpe_contribution:+.2f}")
    print(f"     Avg Return:      {dive.avg_return_bps:+.2f} bps")
    print()
    print(f"  {trend_emoji} Trend: {dive.recent_trend}")
    print()
    print(f"  📋 Regime Fit Distribution:")
    for regime, count in sorted(dive.regime_fit_distribution.items(),
                                 key=lambda x: -x[1]):
        pct = count / max(dive.total_observations, 1) * 100
        bar = "█" * int(pct / 5)
        print(f"     {regime:15} {count:4d} ({pct:4.1f}%) {bar}")
    print("=" * 74)
    print()


def print_history(decisions: List[DecisionExplanation]):
    """Print a timeline of recent decisions."""
    if not decisions:
        print("\nNo decision history available.\n")
        return

    print()
    print("=" * 74)
    print(f"  📅 RECENT DECISION HISTORY ({len(decisions)} decisions)")
    print("=" * 74)
    print()

    for i, d in enumerate(decisions):
        print(f"  ── Decision #{i + 1} ──")
        print(f"     {d.timestamp}")
        print(f"     Action: {d.action.upper()} | Regime: {d.regime} | "
              f"Confidence: {d.confidence:.0%}")
        print(f"     Consensus: {d.consensus_direction} "
              f"(agreement {d.agreement_ratio:.0%})")
        top = d.top_drivers[:3]
        if top:
            print(f"     Drivers: {', '.join(top)}")
        if d.asset_changes:
            changes = ", ".join(f"{a}{d:+.1f}%" for a, d in d.asset_changes.items())
            print(f"     Shifts:  {changes}")
        print()

    print("=" * 74)
    print()


# ─────────────────────────────────────────────
#  Main CLI
# ─────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Portfolio Explainability Dashboard (v8.07)"
    )
    subparsers = parser.add_subparsers(dest="command")

    explain_parser = subparsers.add_parser("explain", help="Explain latest portfolio decision")
    explain_parser.add_argument("--save", action="store_true", help="Save report")

    today_parser = subparsers.add_parser("today", help="Today's portfolio positioning explainer")
    today_parser.add_argument("--save", action="store_true", help="Save report")

    signal_parser = subparsers.add_parser("signal", help="Deep-dive into a specific signal source")
    signal_parser.add_argument("source", help="Signal source name or partial match")

    history_parser = subparsers.add_parser("history", help="Recent decision history")
    history_parser.add_argument("--n", type=int, default=5, help="Number of decisions")

    all_parser = subparsers.add_parser("all", help="Full explainability report")
    all_parser.add_argument("--save", action="store_true", help="Save report")

    args = parser.parse_args()
    explainer = PortfolioExplainability()

    if args.command == "explain":
        decision = explainer.explain_latest_decision()
        if decision:
            print_explain(decision)
            if args.save:
                report = explainer.generate_report()
                explainer.save_report(report)
        else:
            print("\n⚠️  No decision data available. Has the ensemble voter been running?\n")

    elif args.command == "today":
        print_today(explainer)
        if args.save:
            report = explainer.generate_report()
            explainer.save_report(report)

    elif args.command == "signal":
        dive = explainer.signal_deep_dive(args.source)
        if dive:
            print_signal_deep_dive(dive)
        else:
            print(f"\n⚠️  Signal source '{args.source}' not found.\n")
            print("Available sources:")
            for key, meta in sorted(SIGNAL_SOURCE_META.items()):
                print(f"  - {key:30} {meta['name']}")

    elif args.command == "history":
        decisions = explainer.explain_recent_decisions(getattr(args, "n", 5))
        print_history(decisions)

    elif args.command == "all":
        report = explainer.generate_report()
        if report.latest_decision:
            print_explain(report.latest_decision)
        if report.recent_decisions:
            print_history(report.recent_decisions)
        if report.signal_deep_dives:
            for dive in report.signal_deep_dives.values():
                print_signal_deep_dive(dive)
        if args.save:
            explainer.save_report(report)
        print(f"\n📁 Report saved to: data/explainability/\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
