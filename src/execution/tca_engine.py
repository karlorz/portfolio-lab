#!/usr/bin/env python3
"""
Portfolio-Lab v6.00: Post-Trade Transaction Cost Analysis Engine

Tracks slippage vs arrival price, partitions impact into permanent vs temporary
using Almgren-Chriss decomposition, and generates execution quality scores (0-100).

References:
    - Almgren & Chriss (2000) "Optimal Execution of Portfolio Transactions"
    - Almgren et al. (2005) "Direct Estimation of Equity Market Impact"
    - Kissell & Glantz (2003) "Optimal Trading Strategies"
    - tcapy (cuemacro/tcapy) architecture patterns

Architecture:
    Order Log → Arrival Price Lookup → Impact Decomposition → Quality Scorecard

Integration:
    - EnsembleVoter: SignalSource.EXECUTION_TCA (5% weight, negative)
    - SmartRebalanceGate: min_order_size threshold from TCA
    - Dashboard: Execution quality panel

Usage:
    from src.execution.tca_engine import TCAEngine

    engine = TCAEngine()
    results = engine.analyze_recent_orders(days=30)
    engine.print_report(results)

CLI:
    python -m src.execution.tca_engine report --days 30
    python -m src.execution.tca_engine score --order-id <id>
    python -m src.execution.tca_engine status
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

from src.paths import DATA_DIR

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OrderRecord:
    """Single executed order from the order log."""
    symbol: str
    side: str  # 'buy' or 'sell'
    shares: float
    estimated_price: float
    estimated_value: float
    fill_price: float
    fill_shares: float
    fill_value: float
    timestamp: str
    reason: str = ""
    drift_before: float = 0.0

    @property
    def slippage_bps(self) -> float:
        """Signed slippage in basis points. Negative = worse for us."""
        if self.estimated_price <= 0:
            return 0.0
        raw = (self.fill_price - self.estimated_price) / self.estimated_price
        # Buy: positive slippage (paid more) is bad → negative
        # Sell: negative slippage (received less) is bad → negative
        if self.side == 'buy':
            return -raw * 10000
        else:
            return raw * 10000

    @property
    def fill_rate(self) -> float:
        """Fraction of shares filled (0-1)."""
        if self.shares <= 0:
            return 1.0
        return min(self.fill_shares / self.shares, 1.0)

    @property
    def notional(self) -> float:
        """Fill notional in dollars."""
        return self.fill_value


@dataclass
class ImpactDecomposition:
    """Almgren-Chriss style impact breakdown for a single order."""
    total_slippage_bps: float
    permanent_impact_bps: float
    temporary_impact_bps: float
    timing_luck_bps: float
    spread_cost_bps: float

    # Quality score 0-100
    quality_score: float

    def to_dict(self) -> dict:
        return {
            "total_slippage_bps": round(self.total_slippage_bps, 2),
            "permanent_impact_bps": round(self.permanent_impact_bps, 2),
            "temporary_impact_bps": round(self.temporary_impact_bps, 2),
            "timing_luck_bps": round(self.timing_luck_bps, 2),
            "spread_cost_bps": round(self.spread_cost_bps, 2),
            "quality_score": round(self.quality_score, 1),
        }


@dataclass
class TCAOrderResult:
    """Full TCA analysis for a single order."""
    order: OrderRecord
    impact: ImpactDecomposition
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.order.symbol,
            "side": self.order.side,
            "fill_value": round(self.order.fill_value, 2),
            "fill_rate": round(self.order.fill_rate, 4),
            "slippage_bps": round(self.order.slippage_bps, 2),
            "impact": self.impact.to_dict(),
            "timestamp": self.timestamp,
        }


@dataclass
class TCAAggregate:
    """Aggregated TCA metrics across a set of orders."""
    total_orders: int
    total_notional: float
    avg_slippage_bps: float
    avg_quality_score: float
    weighted_slippage_bps: float  # Volume-weighted
    by_symbol: Dict[str, Dict[str, Any]]
    by_side: Dict[str, Dict[str, Any]]
    period_days: int

    def to_dict(self) -> dict:
        return {
            "total_orders": self.total_orders,
            "total_notional": round(self.total_notional, 2),
            "avg_slippage_bps": round(self.avg_slippage_bps, 2),
            "avg_quality_score": round(self.avg_quality_score, 1),
            "weighted_slippage_bps": round(self.weighted_slippage_bps, 2),
            "by_symbol": {k: {sk: round(sv, 2) if isinstance(sv, float) else sv
                              for sk, sv in v.items()}
                          for k, v in self.by_symbol.items()},
            "by_side": {k: {sk: round(sv, 2) if isinstance(sv, float) else sv
                            for sk, sv in v.items()}
                        for k, v in self.by_side.items()},
            "period_days": self.period_days,
        }


# ---------------------------------------------------------------------------
# Core TCA Engine
# ---------------------------------------------------------------------------

class TCAEngine:
    """
    Post-trade Transaction Cost Analysis engine.

    Features:
    - Parses order logs from data/orders.jsonl
    - Calculates implementation shortfall (arrival price vs fill price)
    - Almgren-Chriss impact decomposition
    - Execution quality scoring (0-100)
    - Aggregation by symbol, side, and time period
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = str(DATA_DIR)
        self.data_dir = Path(data_dir)
        self.order_log_path = self.data_dir / "orders.jsonl"

        # Typical spread assumptions by asset class (in bps)
        self._default_spreads = {
            "SPY": 1.0,
            "QQQ": 1.2,
            "GLD": 1.5,
            "TLT": 2.0,
            "IEF": 1.5,
            "SHY": 1.0,
            "BTC": 5.0,
            "ETH": 6.0,
        }

    # ------------------------------------------------------------------
    # Order loading
    # ------------------------------------------------------------------

    def load_orders(self, days: Optional[int] = None) -> List[OrderRecord]:
        """
        Load order records from the JSONL order log.

        Args:
            days: If set, only orders from the last N days.

        Returns:
            List of OrderRecord instances, newest first.
        """
        if not self.order_log_path.exists():
            return []

        cutoff = None
        if days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        records = []
        with open(self.order_log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                record = OrderRecord(
                    symbol=data.get("symbol", ""),
                    side=data.get("side", "buy"),
                    shares=float(data.get("shares", 0)),
                    estimated_price=float(data.get("estimated_price", 0)),
                    estimated_value=float(data.get("estimated_value", 0)),
                    fill_price=float(data.get("fill_price", 0)),
                    fill_shares=float(data.get("fill_shares", 0)),
                    fill_value=float(data.get("fill_value", 0)),
                    timestamp=data.get("timestamp", ""),
                    reason=data.get("reason", ""),
                    drift_before=float(data.get("drift_before", 0)),
                )

                # Skip invalid records
                if record.fill_value <= 0 or record.fill_price <= 0:
                    continue

                # Apply time filter
                if cutoff and record.timestamp:
                    try:
                        ts = datetime.fromisoformat(record.timestamp)
                        if ts < cutoff:
                            continue
                    except (ValueError, TypeError):
                        pass

                records.append(record)

        # Sort newest first
        records.sort(key=lambda r: r.timestamp or "", reverse=True)
        return records

    # ------------------------------------------------------------------
    # Impact decomposition
    # ------------------------------------------------------------------

    def decompose_impact(self, order: OrderRecord) -> ImpactDecomposition:
        """
        Decompose total implementation shortfall into components.

        Uses a simplified Almgren-Chriss model:
        - Total slippage = fill_price - arrival_price (in bps)
        - Spread cost = estimated bid-ask spread (in bps)
        - Permanent impact = estimated information leakage
        - Temporary impact = liquidity demand premium
        - Timing luck = residual (market move during execution)

        For paper trading, fills are at market so spread cost is the
        dominant component. The decomposition is calibrated for small
        institutional-sized orders ($500-$50,000 notional).

        Args:
            order: The executed order record.

        Returns:
            ImpactDecomposition with partitioned costs.
        """
        total_bps = order.slippage_bps

        # 1. Spread cost: estimated from asset class defaults
        spread_bps = self._default_spreads.get(order.symbol, 2.0)
        # Half-spread for the direction we trade
        spread_cost = spread_bps * 0.5

        # 2. Permanent impact: Almgren-Chriss γ * σ * (Q/V)^0.5
        # For small orders (0.01-0.1% of ADV), permanent impact is small
        participation_pct = self._estimate_participation(order)
        vol_factor = 0.15  # ~15% annual vol scaled to daily
        permanent_impact = vol_factor * math.sqrt(participation_pct * 100) * 0.5

        # 3. Temporary impact: η * σ * (Q/(V*τ))^0.6
        temp_pct = participation_pct / max(order.fill_rate, 0.01)
        temp_factor = 0.3  # Temporary impact coefficient
        temporary_impact = temp_factor * math.sqrt(temp_pct * 100) * 0.5

        # 4. Timing luck: residual after removing known components
        timing_luck = total_bps - spread_cost - permanent_impact - temporary_impact

        # 5. Quality score: 0-100 based on how close fill is to arrival
        score = self._compute_quality_score(
            total_bps=total_bps,
            spread_cost=spread_cost,
            fill_rate=order.fill_rate,
            side=order.side,
            notional=order.fill_value,
        )

        return ImpactDecomposition(
            total_slippage_bps=total_bps,
            permanent_impact_bps=permanent_impact,
            temporary_impact_bps=temporary_impact,
            timing_luck_bps=timing_luck,
            spread_cost_bps=spread_cost,
            quality_score=score,
        )

    def _estimate_participation(self, order: OrderRecord) -> float:
        """
        Estimate participation rate (order size / estimated daily volume).

        Uses a rule of thumb: for small notional orders, participation
        is typically 0.1-2% of daily volume. Returns as decimal (0-1).
        """
        # Rough daily volume estimates by symbol (shares)
        adv_estimates = {
            "SPY": 70_000_000,
            "QQQ": 40_000_000,
            "GLD": 8_000_000,
            "TLT": 10_000_000,
            "IEF": 5_000_000,
            "SHY": 3_000_000,
            "BTC": 500_000,
            "ETH": 300_000,
        }

        adv = adv_estimates.get(order.symbol, 5_000_000)
        participation = order.fill_shares / max(adv, 1)
        return max(min(participation, 0.05), 0.0001)  # Clamp 0.01% - 5%

    def _compute_quality_score(
        self,
        total_bps: float,
        spread_cost: float,
        fill_rate: float,
        side: str,
        notional: float,
    ) -> float:
        """
        Compute execution quality score 0-100.

        Scoring logic:
        - Base: 85 points
        - Slippage penalty: -10 per bps of adverse slippage (capped at -50)
        - Fill rate bonus: +10 for 100%, +5 for >95%, 0 otherwise
        - Large order penalty: -5 if notional > $50,000 (harder to fill)
        """
        score = 85.0

        # Slippage penalty: more slippage = worse score
        # For buys: negative slippage (paid more) is bad
        # For sells: positive slippage (got less) is bad
        adverse_slippage = abs(total_bps)
        if adverse_slippage > spread_cost:
            excess = adverse_slippage - spread_cost
            penalty = min(excess * 10, 50.0)
            score -= penalty

        # Fill rate bonus
        if fill_rate >= 0.999:
            score += 10
        elif fill_rate >= 0.95:
            score += 5
        elif fill_rate < 0.5:
            score -= 10

        # Large order penalty (only for substantial notional)
        if notional > 50_000:
            score -= 5

        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_orders(self, orders: List[OrderRecord]) -> List[TCAOrderResult]:
        """
        Run full TCA analysis on a list of orders.

        Args:
            orders: List of OrderRecord instances.

        Returns:
            List of TCAOrderResult with impact decomposition.
        """
        results = []
        now = datetime.now(timezone.utc).isoformat()
        for order in orders:
            impact = self.decompose_impact(order)
            results.append(TCAOrderResult(
                order=order,
                impact=impact,
                timestamp=now,
            ))
        return results

    def analyze_recent_orders(self, days: int = 30) -> List[TCAOrderResult]:
        """
        Convenience: load and analyze recent orders.

        Args:
            days: Lookback period.

        Returns:
            List of TCAOrderResult.
        """
        orders = self.load_orders(days=days)
        return self.analyze_orders(orders)

    def aggregate(self, results: List[TCAOrderResult]) -> TCAAggregate:
        """
        Aggregate TCA results across multiple orders.

        Computes volume-weighted averages and breakdowns by symbol and side.

        Args:
            results: TCA analysis results.

        Returns:
            Aggregated metrics.
        """
        if not results:
            return TCAAggregate(
                total_orders=0,
                total_notional=0.0,
                avg_slippage_bps=0.0,
                avg_quality_score=0.0,
                weighted_slippage_bps=0.0,
                by_symbol={},
                by_side={},
                period_days=0,
            )

        total_notional = sum(r.order.fill_value for r in results)
        total_orders = len(results)

        # Weighted average slippage
        if total_notional > 0:
            weighted_slippage = sum(
                r.order.slippage_bps * r.order.fill_value for r in results
            ) / total_notional
        else:
            weighted_slippage = 0.0

        avg_slippage = sum(r.order.slippage_bps for r in results) / total_orders
        avg_quality = sum(r.impact.quality_score for r in results) / total_orders

        # By symbol
        by_symbol = defaultdict(lambda: {"count": 0, "notional": 0.0,
                                          "slippage_bps": 0.0, "quality": 0.0})
        for r in results:
            sym = r.order.symbol
            by_symbol[sym]["count"] += 1
            by_symbol[sym]["notional"] += r.order.fill_value
            by_symbol[sym]["slippage_bps"] += r.order.slippage_bps
            by_symbol[sym]["quality"] += r.impact.quality_score

        for sym in by_symbol:
            c = by_symbol[sym]["count"]
            by_symbol[sym]["slippage_bps"] /= max(c, 1)
            by_symbol[sym]["quality"] /= max(c, 1)

        # By side
        by_side = defaultdict(lambda: {"count": 0, "notional": 0.0,
                                        "slippage_bps": 0.0, "quality": 0.0})
        for r in results:
            s = r.order.side
            by_side[s]["count"] += 1
            by_side[s]["notional"] += r.order.fill_value
            by_side[s]["slippage_bps"] += r.order.slippage_bps
            by_side[s]["quality"] += r.impact.quality_score

        for s in by_side:
            c = by_side[s]["count"]
            by_side[s]["slippage_bps"] /= max(c, 1)
            by_side[s]["quality"] /= max(c, 1)

        return TCAAggregate(
            total_orders=total_orders,
            total_notional=total_notional,
            avg_slippage_bps=avg_slippage,
            avg_quality_score=avg_quality,
            weighted_slippage_bps=weighted_slippage,
            by_symbol=dict(by_symbol),
            by_side=dict(by_side),
            period_days=0,  # Will be computed by caller if needed
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, results: List[TCAOrderResult],
                     aggregate: Optional[TCAAggregate] = None) -> str:
        """
        Generate a formatted TCA report string.

        Args:
            results: Individual order analyses.
            aggregate: Pre-computed aggregate (or None to auto-compute).

        Returns:
            Formatted report text.
        """
        if aggregate is None:
            aggregate = self.aggregate(results)

        lines = []
        lines.append("=" * 60)
        lines.append("  POST-TRADE TRANSACTION COST ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append(f"  Period: {aggregate.period_days} days")
        lines.append(f"  Orders: {aggregate.total_orders}")
        lines.append(f"  Total Notional: ${aggregate.total_notional:,.2f}")
        lines.append(f"  Avg Slippage: {aggregate.avg_slippage_bps:+.2f} bps")
        lines.append(f"  VWAP Slippage: {aggregate.weighted_slippage_bps:+.2f} bps")
        lines.append(f"  Avg Quality Score: {aggregate.avg_quality_score:.1f}/100")
        lines.append("")

        if aggregate.by_symbol:
            lines.append("  ── By Symbol ──")
            for sym in sorted(aggregate.by_symbol.keys()):
                s = aggregate.by_symbol[sym]
                lines.append(f"    {sym:6s}: {s['count']:2d} orders, "
                             f"${s['notional']:>8,.0f}, "
                             f"slippage {s['slippage_bps']:+.2f} bps, "
                             f"quality {s['quality']:.1f}")
            lines.append("")

        if aggregate.by_side:
            lines.append("  ── By Side ──")
            for side in sorted(aggregate.by_side.keys()):
                s = aggregate.by_side[side]
                lines.append(f"    {side:6s}: {s['count']:2d} orders, "
                             f"${s['notional']:>8,.0f}, "
                             f"slippage {s['slippage_bps']:+.2f} bps, "
                             f"quality {s['quality']:.1f}")
            lines.append("")

        # Recent individual orders (top 10)
        if results:
            lines.append("  ── Recent Order Details ──")
            for r in results[:10]:
                i = r.impact
                lines.append(
                    f"    {r.order.symbol:6s} {r.order.side:4s} "
                    f"${r.order.fill_value:>8,.0f} "
                    f"slip={r.order.slippage_bps:+.2f}bps "
                    f"perm={i.permanent_impact_bps:+.2f} "
                    f"temp={i.temporary_impact_bps:+.2f} "
                    f"score={i.quality_score:.0f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Data export for dashboard
    # ------------------------------------------------------------------

    def export_dashboard_data(self, results: List[TCAOrderResult]) -> dict:
        """
        Export TCA data in dashboard-friendly JSON format.

        Args:
            results: TCA analysis results.

        Returns:
            Dict with summary + per-order data.
        """
        agg = self.aggregate(results)

        # Recent quality trend (last 20 orders)
        recent_scores = [r.impact.quality_score for r in results[:20]]
        recent_scores.reverse()  # Chronological

        return {
            "summary": agg.to_dict(),
            "quality_trend": recent_scores,
            "recent_orders": [r.to_dict() for r in results[:20]],
            "generated": datetime.now(timezone.utc).isoformat(),
        }

    def save_dashboard_data(self, results: List[TCAOrderResult]) -> None:
        """Save dashboard JSON to data/tca_dashboard.json."""
        data = self.export_dashboard_data(results)
        out_path = self.data_dir / "tca_dashboard.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved TCA dashboard data to {out_path}")


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-Trade Transaction Cost Analysis"
    )
    sub = parser.add_subparsers(dest="command")

    # report
    report_parser = sub.add_parser("report",
                                    help="Generate TCA report for recent orders")
    report_parser.add_argument("--days", type=int, default=30,
                               help="Lookback period in days")
    report_parser.add_argument("--save", action="store_true",
                               help="Save dashboard data to JSON")

    # score
    score_parser = sub.add_parser("score", help="Score a specific order")
    score_parser.add_argument("order_id", nargs="?", help="Order ID or index (1-based)")

    # status
    sub.add_parser("status", help="Quick TCA status check")

    args = parser.parse_args()

    engine = TCAEngine()

    if args.command == "report":
        results = engine.analyze_recent_orders(days=args.days)
        if not results:
            print(f"No orders found in the last {args.days} days.")
            sys.exit(0)

        agg = engine.aggregate(results)
        agg.period_days = args.days
        report = engine.print_report(results, agg)
        print(report)

        if args.save:
            engine.save_dashboard_data(results)

    elif args.command == "score":
        orders = engine.load_orders(days=90)
        if not orders:
            print("No orders found.")
            sys.exit(0)

        if args.order_id:
            try:
                idx = int(args.order_id) - 1
                if idx < 0 or idx >= len(orders):
                    print(f"Order index {args.order_id} out of range "
                          f"(1-{len(orders)})")
                    sys.exit(1)
                target = [orders[idx]]
            except ValueError:
                # Try matching by symbol
                target = [o for o in orders if o.symbol == args.order_id.upper()]
                if not target:
                    print(f"No orders found for symbol {args.order_id}")
                    sys.exit(1)
        else:
            target = orders[:1]

        results = engine.analyze_orders(target)
        for r in results:
            print(json.dumps(r.to_dict(), indent=2))

    elif args.command == "status":
        orders = engine.load_orders(days=7)
        if not orders:
            print("No orders in the last 7 days.")
            return

        results = engine.analyze_orders(orders)
        agg = engine.aggregate(results)
        print(f"Orders (7d): {agg.total_orders}")
        print(f"Notional:    ${agg.total_notional:,.2f}")
        print(f"Avg slip:    {agg.avg_slippage_bps:+.2f} bps")
        print(f"Avg quality: {agg.avg_quality_score:.1f}/100")
        print(f"VWAP slip:   {agg.weighted_slippage_bps:+.2f} bps")


if __name__ == "__main__":
    main()
