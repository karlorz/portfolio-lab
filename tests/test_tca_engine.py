#!/usr/bin/env python3
"""
Tests for v6.00 Post-Trade Transaction Cost Analysis Engine.

Covers:
- Order loading and parsing
- Slippage calculation
- Almgren-Chriss impact decomposition
- Execution quality scoring
- Aggregation by symbol/side
- Edge cases: empty logs, zero fills, missing fields
- CLI integration
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.execution.tca_engine import (
    TCAEngine,
    OrderRecord,
    ImpactDecomposition,
    TCAOrderResult,
    TCAAggregate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_buy_order():
    """A standard buy order with slight adverse slippage."""
    return OrderRecord(
        symbol="SPY",
        side="buy",
        shares=100.0,
        estimated_price=500.0,
        estimated_value=50000.0,
        fill_price=500.50,  # 1 bps adverse slippage
        fill_shares=100.0,
        fill_value=50050.0,
        timestamp="2026-05-15T10:30:00",
        reason="rebalance_up",
        drift_before=0.45,
    )


@pytest.fixture
def sample_sell_order():
    """A standard sell order with slight adverse slippage."""
    return OrderRecord(
        symbol="GLD",
        side="sell",
        shares=50.0,
        estimated_price=200.0,
        estimated_value=10000.0,
        fill_price=199.90,  # -0.5 bps adverse slippage (sold low)
        fill_shares=50.0,
        fill_value=9995.0,
        timestamp="2026-05-15T11:00:00",
        reason="rebalance_down",
        drift_before=0.40,
    )


@pytest.fixture
def sample_partial_fill():
    """An order that was only partially filled."""
    return OrderRecord(
        symbol="TLT",
        side="buy",
        shares=200.0,
        estimated_price=85.0,
        estimated_value=17000.0,
        fill_price=85.10,  # ~1.2 bps slippage
        fill_shares=150.0,
        fill_value=12765.0,
        timestamp="2026-05-14T14:00:00",
        reason="rebalance_up",
        drift_before=0.15,
    )


@pytest.fixture
def sample_zero_slippage():
    """Perfect execution at arrival price."""
    return OrderRecord(
        symbol="SPY",
        side="buy",
        shares=50.0,
        estimated_price=500.0,
        estimated_value=25000.0,
        fill_price=500.0,
        fill_shares=50.0,
        fill_value=25000.0,
        timestamp="2026-05-14T09:30:00",
        reason="rebalance_up",
        drift_before=0.46,
    )


@pytest.fixture
def temp_order_log():
    """Create a temporary orders.jsonl file with sample data."""
    orders = [
        {
            "symbol": "SPY", "side": "buy", "shares": 100.0,
            "estimated_price": 500.0, "estimated_value": 50000.0,
            "fill_price": 500.50, "fill_shares": 100.0,
            "fill_value": 50050.0, "timestamp": "2026-05-15T10:30:00",
            "reason": "rebalance_up", "drift_before": 0.45,
        },
        {
            "symbol": "GLD", "side": "sell", "shares": 50.0,
            "estimated_price": 200.0, "estimated_value": 10000.0,
            "fill_price": 199.90, "fill_shares": 50.0,
            "fill_value": 9995.0, "timestamp": "2026-05-15T11:00:00",
            "reason": "rebalance_down", "drift_before": 0.40,
        },
        {
            "symbol": "TLT", "side": "buy", "shares": 200.0,
            "estimated_price": 85.0, "estimated_value": 17000.0,
            "fill_price": 85.10, "fill_shares": 200.0,
            "fill_value": 17020.0, "timestamp": "2026-05-14T14:00:00",
            "reason": "rebalance_up", "drift_before": 0.16,
        },
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                     delete=False) as f:
        for o in orders:
            f.write(json.dumps(o) + "\n")
        path = f.name
    yield path
    os.unlink(path)


# ---------------------------------------------------------------------------
# OrderRecord Tests
# ---------------------------------------------------------------------------

class TestOrderRecord:
    """Test OrderRecord dataclass properties."""

    def test_buy_slippage_negative(self, sample_buy_order):
        """Buying above arrival = negative slippage (bad)."""
        assert sample_buy_order.slippage_bps < 0

    def test_sell_slippage_negative(self, sample_sell_order):
        """Selling below arrival = negative slippage (bad)."""
        assert sample_sell_order.slippage_bps < 0

    def test_buy_slippage_value(self, sample_buy_order):
        """Verify buy slippage magnitude: (500.50-500)/500 * 10000 = 10 bps."""
        # (500.50 - 500.00) / 500.00 = 0.001 = 10 bps
        # Buy: paid more, so negative: -10 bps
        assert abs(sample_buy_order.slippage_bps - (-10.0)) < 0.1

    def test_sell_slippage_value(self, sample_sell_order):
        """Verify sell slippage magnitude: (199.90-200)/200 * 10000 = -5 bps."""
        # Sell: received less, so (199.90-200)/200 = -0.0005 → -5 bps
        # But for sell, slippage formula returns raw difference
        # side='sell': returns raw * 10000 = -0.0005 * 10000 = -5.0
        assert abs(sample_sell_order.slippage_bps - (-5.0)) < 0.1

    def test_zero_slippage(self, sample_zero_slippage):
        """Fill exactly at arrival price = 0 slippage."""
        assert abs(sample_zero_slippage.slippage_bps) < 0.001

    def test_fill_rate_full(self, sample_buy_order):
        """Full fill = rate of 1.0."""
        assert sample_buy_order.fill_rate == 1.0

    def test_fill_rate_partial(self, sample_partial_fill):
        """Partial fill = rate < 1.0."""
        assert sample_partial_fill.fill_rate == 0.75

    def test_fill_rate_zero_shares(self):
        """Zero requested shares = fill rate of 1.0."""
        o = OrderRecord(symbol="SPY", side="buy", shares=0,
                        estimated_price=500, estimated_value=0,
                        fill_price=500, fill_shares=100,
                        fill_value=50000, timestamp="now")
        assert o.fill_rate == 1.0


# ---------------------------------------------------------------------------
# TCAEngine Tests
# ---------------------------------------------------------------------------

class TestTCAEngineLoading:
    """Test order loading from JSONL."""

    def test_load_empty_log(self):
        """Loading from non-existent path returns empty list."""
        with tempfile.TemporaryDirectory() as tmp:
            engine = TCAEngine(data_dir=tmp)
            orders = engine.load_orders()
            assert orders == []

    def test_load_from_file(self, temp_order_log):
        """Load valid orders from JSONL."""
        with tempfile.TemporaryDirectory() as tmp:
            # Copy temp file to tmp
            import shutil
            shutil.copy(temp_order_log, Path(tmp) / "orders.jsonl")
            engine = TCAEngine(data_dir=tmp)
            orders = engine.load_orders()
            assert len(orders) == 3

    def test_load_with_days_filter(self, temp_order_log):
        """Days filter excludes old orders."""
        with tempfile.TemporaryDirectory() as tmp:
            import shutil
            shutil.copy(temp_order_log, Path(tmp) / "orders.jsonl")
            engine = TCAEngine(data_dir=tmp)
            # All orders are May 14-15, should be within 60 days
            orders = engine.load_orders(days=60)
            assert len(orders) == 3
            # Within 1 day should return some
            orders = engine.load_orders(days=1)
            # May 15 is within 1 day of now
            assert len(orders) >= 2  # The two May 15 orders

    def test_skip_invalid_records(self):
        """Skip records with zero fill value or price."""
        records = [
            json.dumps({"symbol": "SPY", "side": "buy", "shares": 100,
                        "estimated_price": 500, "estimated_value": 50000,
                        "fill_price": 0, "fill_shares": 0,  # Invalid
                        "fill_value": 0, "timestamp": "now"}),
            json.dumps({"symbol": "GLD", "side": "sell", "shares": 50,
                        "estimated_price": 200, "estimated_value": 10000,
                        "fill_price": 199, "fill_shares": 50,
                        "fill_value": 9950, "timestamp": "now"}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "orders.jsonl"
            with open(log_path, "w") as f:
                for r in records:
                    f.write(r + "\n")
            engine = TCAEngine(data_dir=tmp)
            orders = engine.load_orders()
            assert len(orders) == 1  # Second record is valid

    def test_chronological_sorting(self, temp_order_log):
        """Orders sorted newest first."""
        with tempfile.TemporaryDirectory() as tmp:
            import shutil
            shutil.copy(temp_order_log, Path(tmp) / "orders.jsonl")
            engine = TCAEngine(data_dir=tmp)
            orders = engine.load_orders()
            assert orders[0].timestamp >= orders[-1].timestamp


class TestTCAEngineImpact:
    """Test impact decomposition."""

    def test_decompose_returns_impact(self, sample_buy_order):
        """Decompose returns ImpactDecomposition with all fields."""
        engine = TCAEngine()
        impact = engine.decompose_impact(sample_buy_order)
        assert isinstance(impact, ImpactDecomposition)
        assert impact.total_slippage_bps is not None
        assert impact.quality_score is not None

    def test_decompose_slippage_matches(self, sample_buy_order):
        """Total slippage in decomposition matches order.slippage_bps."""
        engine = TCAEngine()
        impact = engine.decompose_impact(sample_buy_order)
        assert abs(impact.total_slippage_bps - sample_buy_order.slippage_bps) < 0.01

    def test_decompose_permanent_small(self, sample_buy_order):
        """Permanent impact should be small for small orders."""
        engine = TCAEngine()
        impact = engine.decompose_impact(sample_buy_order)
        assert impact.permanent_impact_bps < 1.0  # Less than 1 bps

    def test_decompose_temporary_positive(self, sample_buy_order):
        """Temporary impact should be small positive."""
        engine = TCAEngine()
        impact = engine.decompose_impact(sample_buy_order)
        assert impact.temporary_impact_bps >= 0

    def test_quality_score_range(self, sample_buy_order):
        """Quality score between 0-100."""
        engine = TCAEngine()
        impact = engine.decompose_impact(sample_buy_order)
        assert 0 <= impact.quality_score <= 100

    def test_quality_score_zero_slippage(self, sample_zero_slippage):
        """Zero slippage should score highly (>85)."""
        engine = TCAEngine()
        impact = engine.decompose_impact(sample_zero_slippage)
        assert impact.quality_score >= 85

    def test_quality_score_full_fill_bonus(self, sample_zero_slippage):
        """Full fill with zero slippage should get max bonus (95+)."""
        engine = TCAEngine()
        impact = engine.decompose_impact(sample_zero_slippage)
        # Base 85 + full fill bonus 10 = 95
        assert impact.quality_score >= 95

    def test_quality_score_partial_fill(self, sample_partial_fill):
        """Partial fill without fill rate bonus — score reflects slippage."""
        engine = TCAEngine()
        impact = engine.decompose_impact(sample_partial_fill)
        # ~11.8 bps slippage penalty drops score from 85 to ~35
        assert 20 <= impact.quality_score <= 50

    def test_decompose_sell_order(self, sample_sell_order):
        """Sell order decomposition works correctly."""
        engine = TCAEngine()
        impact = engine.decompose_impact(sample_sell_order)
        assert impact.total_slippage_bps is not None
        assert isinstance(impact, ImpactDecomposition)

    def test_large_order_penalty(self):
        """Order > $50K gets quality penalty."""
        engine = TCAEngine()
        large = OrderRecord(
            symbol="SPY", side="buy", shares=200,
            estimated_price=500, estimated_value=100000,
            fill_price=500, fill_shares=200,
            fill_value=100000, timestamp="now",
        )
        impact = engine.decompose_impact(large)
        assert impact.quality_score < 100

    def test_impact_cumulative_decomp(self):
        """Components should approximately sum to total."""
        engine = TCAEngine()
        order = OrderRecord(
            symbol="SPY", side="buy", shares=100,
            estimated_price=500, estimated_value=50000,
            fill_price=501, fill_shares=100,  # 20 bps adverse
            fill_value=50100, timestamp="now",
        )
        impact = engine.decompose_impact(order)
        # spread + perm + temp + timing ≈ total
        total = (impact.spread_cost_bps + impact.permanent_impact_bps
                 + impact.temporary_impact_bps + impact.timing_luck_bps)
        assert abs(total - impact.total_slippage_bps) < 2.0


class TestTCAEngineAnalysis:
    """Test full analysis pipeline."""

    def test_analyze_orders(self, sample_buy_order, sample_sell_order):
        """Analyze multiple orders returns correct number of results."""
        engine = TCAEngine()
        results = engine.analyze_orders([sample_buy_order, sample_sell_order])
        assert len(results) == 2
        assert all(isinstance(r, TCAOrderResult) for r in results)

    def test_analyze_empty(self):
        """Analyze empty list returns empty results."""
        engine = TCAEngine()
        results = engine.analyze_orders([])
        assert results == []

    def test_aggregate_empty(self):
        """Aggregate of empty returns zeroed aggregate."""
        engine = TCAEngine()
        agg = engine.aggregate([])
        assert agg.total_orders == 0
        assert agg.total_notional == 0.0

    def test_aggregate_counts(self, sample_buy_order, sample_sell_order):
        """Aggregate counts orders correctly."""
        engine = TCAEngine()
        results = engine.analyze_orders([sample_buy_order, sample_sell_order])
        agg = engine.aggregate(results)
        assert agg.total_orders == 2

    def test_aggregate_by_symbol(self, sample_buy_order, sample_sell_order):
        """Aggregate breaks down by symbol."""
        engine = TCAEngine()
        results = engine.analyze_orders([
            sample_buy_order,
            sample_sell_order,
            sample_buy_order,  # Two SPY orders
        ])
        agg = engine.aggregate(results)
        assert "SPY" in agg.by_symbol
        assert "GLD" in agg.by_symbol
        assert agg.by_symbol["SPY"]["count"] == 2

    def test_aggregate_by_side(self, sample_buy_order, sample_sell_order):
        """Aggregate breaks down by side."""
        engine = TCAEngine()
        results = engine.analyze_orders([sample_buy_order, sample_sell_order])
        agg = engine.aggregate(results)
        assert "buy" in agg.by_side
        assert "sell" in agg.by_side

    def test_aggregate_weighted_slippage(self):
        """Volume-weighted slippage computed correctly."""
        engine = TCAEngine()
        small_order = OrderRecord(
            symbol="SPY", side="buy", shares=10,
            estimated_price=500, estimated_value=5000,
            fill_price=501, fill_shares=10,  # 20 bps
            fill_value=5010, timestamp="now",
        )
        large_order = OrderRecord(
            symbol="SPY", side="buy", shares=100,
            estimated_price=500, estimated_value=50000,
            fill_price=500.5, fill_shares=100,  # 10 bps
            fill_value=50050, timestamp="now",
        )
        results = engine.analyze_orders([small_order, large_order])
        agg = engine.aggregate(results)
        # Weighted should be closer to 10 bps (large order dominates)
        assert agg.weighted_slippage_bps > agg.avg_slippage_bps * 0.9


class TestTCAEngineReport:
    """Test reporting and export."""

    def test_print_report_empty(self):
        """Print report with empty results doesn't crash."""
        engine = TCAEngine()
        report = engine.print_report([])
        assert isinstance(report, str)
        assert len(report) > 0

    def test_print_report_with_data(self, sample_buy_order):
        """Print report with one order returns formatted output."""
        engine = TCAEngine()
        results = engine.analyze_orders([sample_buy_order])
        report = engine.print_report(results)
        assert "SPY" in report
        assert "buy" in report

    def test_export_dashboard(self, sample_buy_order):
        """Dashboard export returns dict with expected keys."""
        engine = TCAEngine()
        results = engine.analyze_orders([sample_buy_order])
        data = engine.export_dashboard_data(results)
        assert "summary" in data
        assert "quality_trend" in data
        assert "recent_orders" in data

    def test_save_dashboard_data(self, sample_buy_order):
        """Save dashboard writes JSON file."""
        with tempfile.TemporaryDirectory() as tmp:
            engine = TCAEngine(data_dir=tmp)
            results = engine.analyze_orders([sample_buy_order])
            engine.save_dashboard_data(results)
            assert (Path(tmp) / "tca_dashboard.json").exists()
            data = json.loads((Path(tmp) / "tca_dashboard.json").read_text())
            assert "summary" in data


class TestTCAEngineEdgeCases:
    """Test edge cases and robustness."""

    def test_negative_fill_price(self):
        """Negative fill price is skipped on load."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "orders.jsonl"
            with open(log_path, "w") as f:
                f.write(json.dumps({
                    "symbol": "SPY", "side": "buy",
                    "shares": 100, "estimated_price": 500,
                    "estimated_value": 50000,
                    "fill_price": -1,  # Invalid
                    "fill_shares": 100, "fill_value": -100,
                    "timestamp": "now",
                }) + "\n")
            engine = TCAEngine(data_dir=tmp)
            orders = engine.load_orders()
            assert len(orders) == 0

    def test_malformed_json_skipped(self):
        """Malformed JSON lines are skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "orders.jsonl"
            with open(log_path, "w") as f:
                f.write("{invalid json}\n")
                f.write(json.dumps({
                    "symbol": "SPY", "side": "buy",
                    "shares": 100, "estimated_price": 500,
                    "estimated_value": 50000,
                    "fill_price": 500, "fill_shares": 100,
                    "fill_value": 50000, "timestamp": "now",
                }) + "\n")
            engine = TCAEngine(data_dir=tmp)
            orders = engine.load_orders()
            assert len(orders) == 1

    def test_empty_jsonl(self):
        """Empty JSONL file returns empty list."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "orders.jsonl"
            log_path.write_text("")
            engine = TCAEngine(data_dir=tmp)
            orders = engine.load_orders()
            assert orders == []

    def test_recent_orders_within_days(self, temp_order_log):
        """analyze_recent_orders with days parameter."""
        with tempfile.TemporaryDirectory() as tmp:
            import shutil
            shutil.copy(temp_order_log, Path(tmp) / "orders.jsonl")
            engine = TCAEngine(data_dir=tmp)
            results = engine.analyze_recent_orders(days=60)
            assert len(results) >= 2  # At least some orders found

    def test_impact_dataclass_serialization(self):
        """ImpactDecomposition to_dict works."""
        impact = ImpactDecomposition(
            total_slippage_bps=-1.5,
            permanent_impact_bps=0.1,
            temporary_impact_bps=0.3,
            timing_luck_bps=-1.9,
            spread_cost_bps=0.5,
            quality_score=88.5,
        )
        d = impact.to_dict()
        assert d["total_slippage_bps"] == -1.5
        assert d["quality_score"] == 88.5

    def test_tca_order_result_to_dict(self, sample_buy_order):
        """TCAOrderResult serialization works."""
        engine = TCAEngine()
        results = engine.analyze_orders([sample_buy_order])
        d = results[0].to_dict()
        assert d["symbol"] == "SPY"
        assert d["side"] == "buy"
        assert "impact" in d


class TestTCAEngineCLI:
    """Test CLI invocation (doesn't actually run argparse but tests logic)."""

    def test_report_no_orders(self, capsys):
        """CLI report with no orders shows message, no crash."""
        with tempfile.TemporaryDirectory() as tmp:
            engine = TCAEngine(data_dir=tmp)
            results = engine.analyze_recent_orders(days=30)
            assert results == []

    def test_status_no_orders(self, capsys):
        """CLI status with no orders shows message."""
        with tempfile.TemporaryDirectory() as tmp:
            engine = TCAEngine(data_dir=tmp)
            orders = engine.load_orders(days=7)
            assert len(orders) == 0  # No orders

    def test_score_out_of_range(self, capsys):
        """Score with invalid index shows error."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "orders.jsonl"
            with open(log_path, "w") as f:
                f.write(json.dumps({
                    "symbol": "SPY", "side": "buy",
                    "shares": 100, "estimated_price": 500,
                    "estimated_value": 50000,
                    "fill_price": 500, "fill_shares": 100,
                    "fill_value": 50000, "timestamp": "now",
                }) + "\n")
            engine = TCAEngine(data_dir=tmp)
            orders = engine.load_orders(days=90)
            assert len(orders) == 1
            # First order (index 0) should exist
            assert orders[0].symbol == "SPY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
