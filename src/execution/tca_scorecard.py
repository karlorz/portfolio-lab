#!/usr/bin/env python3
"""
Portfolio-Lab v6.00: TCA Scorecard — Aggregation and dashboard companion.

Provides higher-level aggregation of TCA results with:
- Peer-group normalization (Z-scores within symbol/size buckets)
- Trend analysis (improving or deteriorating execution)
- Broker/venue comparison framework
- CLI dashboard output

Usage:
    from src.execution.tca_scorecard import TCAScorecard
    scorecard = TCAScorecard()
    report = scorecard.generate_daily_report()
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .tca_engine import TCAEngine, TCAOrderResult, TCAAggregate


@dataclass
class TCAPeerGroup:
    """Peer group statistics for normalization."""
    symbol: str
    count: int
    mean_slippage_bps: float
    std_slippage_bps: float
    mean_quality: float
    size_bucket: str  # 'micro', 'small', 'medium', 'large'

    @property
    def z_score(self) -> float:
        """Z-score of this group's mean slippage (0 if no variance)."""
        if self.std_slippage_bps < 0.001:
            return 0.0
        return self.mean_slippage_bps / self.std_slippage_bps


@dataclass
class TCATrend:
    """Execution quality trend over time."""
    period_days: int
    scores: List[float]
    slope: float  # Positive = improving
    recent_avg: float  # Last 5 orders
    overall_avg: float


class TCAScorecard:
    """
    TCA Scorecard — aggregates, normalizes, and trends execution quality.

    Features:
    - Peer-group normalization within symbol/size buckets
    - Execution quality trending (improving or deteriorating)
    - Daily/weekly report generation
    - Dashboard data export
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.engine = TCAEngine(data_dir=data_dir)
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent.parent / "data")
        self.data_dir = Path(data_dir)

    def generate_daily_report(self, days: int = 30) -> dict:
        """
        Generate a complete daily TCA report as a dict.

        Structure:
        - summary: high-level metrics
        - by_symbol: per-symbol breakdown
        - peer_groups: size-bucket normalized stats
        - trend: execution quality over time
        """
        results = self.engine.analyze_recent_orders(days=days)
        if not results:
            return {
                "generated": datetime.now(timezone.utc).isoformat(),
                "period_days": days,
                "total_orders": 0,
                "status": "no_data",
            }

        agg = self.engine.aggregate(results)
        peer_groups = self._compute_peer_groups(results)
        trend = self._compute_trend(results)

        return {
            "generated": datetime.now(timezone.utc).isoformat(),
            "period_days": days,
            "total_orders": agg.total_orders,
            "total_notional": round(agg.total_notional, 2),
            "avg_slippage_bps": round(agg.avg_slippage_bps, 2),
            "avg_quality_score": round(agg.avg_quality_score, 1),
            "weighted_slippage_bps": round(agg.weighted_slippage_bps, 2),
            "by_symbol": {
                sym: {
                    "count": v["count"],
                    "notional": round(v["notional"], 2),
                    "slippage_bps": round(v["slippage_bps"], 2),
                    "quality": round(v["quality"], 1),
                }
                for sym, v in agg.by_symbol.items()
            },
            "peer_groups": peer_groups,
            "trend": {
                "scores": trend.scores[-20:] if trend.scores else [],
                "slope": round(trend.slope, 4),
                "recent_avg": round(trend.recent_avg, 1),
                "overall_avg": round(trend.overall_avg, 1),
            },
            "quality_distribution": self._compute_quality_distribution(results),
            "status": "ok",
        }

    def _compute_peer_groups(
        self, results: List[TCAOrderResult]
    ) -> Dict[str, dict]:
        """Compute peer-group statistics for normalization."""
        groups = defaultdict(list)
        for r in results:
            sym = r.order.symbol
            size = r.order.fill_value
            if size < 10000:
                bucket = "micro"
            elif size < 50000:
                bucket = "small"
            elif size < 100000:
                bucket = "medium"
            else:
                bucket = "large"
            groups[(sym, bucket)].append(r)

        out = {}
        for (sym, bucket), orders in groups.items():
            slippages = [o.order.slippage_bps for o in orders]
            qualities = [o.impact.quality_score for o in orders]
            n = len(slippages)
            mean_slip = sum(slippages) / n
            std_slip = (sum((s - mean_slip) ** 2 for s in slippages) / n) ** 0.5
            mean_q = sum(qualities) / n

            key = f"{sym}_{bucket}"
            out[key] = {
                "symbol": sym,
                "size_bucket": bucket,
                "count": n,
                "mean_slippage_bps": round(mean_slip, 2),
                "std_slippage_bps": round(std_slip, 2),
                "mean_quality": round(mean_q, 1),
                "z_score": round(mean_slip / max(std_slip, 0.001), 2),
            }
        return out

    def _compute_trend(self, results: List[TCAOrderResult]) -> TCATrend:
        """Compute execution quality trend."""
        # Order chronologically (oldest first for regression)
        ordered = list(results)
        ordered.reverse()
        scores = [r.impact.quality_score for r in ordered]

        if len(scores) < 2:
            return TCATrend(
                period_days=30,
                scores=scores,
                slope=0.0,
                recent_avg=float(scores[0]) if scores else 0.0,
                overall_avg=float(scores[0]) if scores else 0.0,
            )

        # Simple linear regression slope
        n = len(scores)
        x_vals = list(range(n))
        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, scores))
        den = sum((x - x_mean) ** 2 for x in x_vals)
        slope = num / den if den != 0 else 0.0

        recent = scores[-5:] if len(scores) >= 5 else scores

        return TCATrend(
            period_days=30,
            scores=scores,
            slope=slope,
            recent_avg=sum(recent) / len(recent),
            overall_avg=y_mean,
        )

    def _compute_quality_distribution(
        self, results: List[TCAOrderResult]
    ) -> Dict[str, int]:
        """Bucket quality scores into ranges."""
        buckets = {
            "excellent_90_100": 0,
            "good_70_89": 0,
            "fair_50_69": 0,
            "poor_20_49": 0,
            "bad_0_19": 0,
        }
        for r in results:
            q = r.impact.quality_score
            if q >= 90:
                buckets["excellent_90_100"] += 1
            elif q >= 70:
                buckets["good_70_89"] += 1
            elif q >= 50:
                buckets["fair_50_69"] += 1
            elif q >= 20:
                buckets["poor_20_49"] += 1
            else:
                buckets["bad_0_19"] += 1
        return buckets

    def print_summary(self, days: int = 30) -> str:
        """Print a human-readable summary of TCA metrics."""
        report = self.generate_daily_report(days=days)

        if report.get("status") == "no_data":
            return f"No TCA data for the last {days} days."

        lines = []
        lines.append("=" * 56)
        lines.append("  TCA SCORECARD SUMMARY")
        lines.append("=" * 56)
        lines.append(f"  Period:       {report['period_days']} days")
        lines.append(f"  Orders:       {report['total_orders']}")
        lines.append(f"  Notional:     ${report['total_notional']:>10,.2f}")
        lines.append(f"  Avg Slippage: {report['avg_slippage_bps']:>+8.2f} bps")
        lines.append(f"  VWAP Slippage:{report['weighted_slippage_bps']:>+8.2f} bps")
        lines.append(f"  Avg Quality:  {report['avg_quality_score']:>7.1f}/100")
        lines.append("")

        # Trend
        trend = report.get("trend", {})
        if trend and trend.get("scores"):
            direction = "↑ improving" if trend.get("slope", 0) > 0.1 else \
                        "↓ deteriorating" if trend.get("slope", 0) < -0.1 else \
                        "→ stable"
            lines.append(f"  Trend:     {direction} (slope={trend.get('slope', 0):+.4f})")
            lines.append(f"  Last 5:    {trend.get('recent_avg', 0):.1f}")
            lines.append(f"  Overall:   {trend.get('overall_avg', 0):.1f}")
            lines.append("")

        # Quality distribution
        dist = report.get("quality_distribution", {})
        if dist:
            lines.append("  Quality Distribution:")
            for bucket, count in sorted(dist.items()):
                if count > 0:
                    lines.append(f"    {bucket:20s}: {count}")
            lines.append("")

        # By symbol
        by_sym = report.get("by_symbol", {})
        if by_sym:
            lines.append(f"  {'Symbol':6s} {'Orders':6s} {'Notional':>10s} "
                         f"{'Slippage':>10s} {'Quality':>8s}")
            lines.append(f"  {'------':6s} {'------':6s} {'----------':>10s} "
                         f"{'----------':>10s} {'-------':>8s}")
            for sym in sorted(by_sym.keys()):
                v = by_sym[sym]
                lines.append(
                    f"  {sym:6s} {v['count']:6d} ${v['notional']:>8,.0f} "
                    f"{v['slippage_bps']:>+9.2f}bps {v['quality']:>6.1f}"
                )

        lines.append("=" * 56)
        return "\n".join(lines)

    def export_dashboard(self, days: int = 30) -> dict:
        """Export full dashboard data."""
        report = self.generate_daily_report(days=days)
        return report

    def save_dashboard(self, days: int = 30) -> None:
        """Save dashboard data to JSON file."""
        report = self.generate_daily_report(days=days)
        out_path = self.data_dir / "tca_scorecard.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved TCA scorecard to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="TCA Scorecard")
    sub = parser.add_subparsers(dest="command")

    summary_parser = sub.add_parser("summary", help="Print TCA summary")
    summary_parser.add_argument("--days", type=int, default=30)

    sub.add_parser("export", help="Export dashboard JSON")

    args = parser.parse_args()
    scorecard = TCAScorecard()

    if args.command == "summary":
        print(scorecard.print_summary(days=args.days))
    elif args.command == "export":
        scorecard.save_dashboard(days=30)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
