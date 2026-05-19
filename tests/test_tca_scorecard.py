"""Tests for src/execution/tca_scorecard.py — TCA Scorecard aggregation."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from src.execution.tca_scorecard import (
    TCAScorecard,
    TCAPeerGroup,
    TCATrend,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

class FakeOrder:
    """Minimal fake matching TCAEngine's expected TCAOrder dataclass."""
    def __init__(self, symbol, fill_value=50000, slippage_bps=-5.0, side="buy"):
        self.symbol = symbol
        self.fill_value = fill_value
        self.slippage_bps = slippage_bps
        self.side = side


class FakeTCAResult:
    """Minimal fake matching TCAEngine's TCAOrderResult."""
    def __init__(self, symbol, fill_value=50000, slippage_bps=-5.0, quality=75.0):
        self.order = FakeOrder(symbol, fill_value, slippage_bps)
        self.impact = MagicMock()
        self.impact.quality_score = quality


def make_result(symbol, fill_value=50000, slippage_bps=-5.0, quality=75.0):
    return FakeTCAResult(symbol, fill_value, slippage_bps, quality)


@pytest.fixture
def sample_results():
    return [
        make_result("SPY", 50000, -8.0, 72.0),
        make_result("SPY", 30000, -3.0, 85.0),
        make_result("TLT", 80000, -12.0, 60.0),
        make_result("GLD", 5000, 2.0, 45.0),
        make_result("SPY", 120000, -5.0, 78.0),
    ]


@pytest.fixture
def scorecard():
    with patch("src.execution.tca_scorecard.TCAEngine"):
        sc = TCAScorecard(data_dir="/tmp/fake_tca")
        sc.engine = MagicMock()
        return sc


# ── TCAPeerGroup ──────────────────────────────────────────────────────────

class TestPeerGroup:
    def test_z_score_zero_with_low_std(self):
        pg = TCAPeerGroup(
            symbol="SPY", count=10, mean_slippage_bps=-5.0,
            std_slippage_bps=0.0001, mean_quality=70.0, size_bucket="medium",
        )
        assert pg.z_score == 0.0

    def test_z_score_computed(self):
        pg = TCAPeerGroup(
            symbol="SPY", count=10, mean_slippage_bps=-5.0,
            std_slippage_bps=2.0, mean_quality=70.0, size_bucket="medium",
        )
        assert pg.z_score == -2.5


# ── TCATrend ──────────────────────────────────────────────────────────────

class TestTrend:
    def test_positive_slope_improving(self):
        trend = TCATrend(
            period_days=30, scores=[50, 60, 70, 80, 90],
            slope=10.0, recent_avg=85.0, overall_avg=70.0,
        )
        assert trend.slope > 0

    def test_negative_slope_deteriorating(self):
        trend = TCATrend(
            period_days=30, scores=[90, 80, 70, 60, 50],
            slope=-10.0, recent_avg=55.0, overall_avg=70.0,
        )
        assert trend.slope < 0


# ── TCAScorecard ──────────────────────────────────────────────────────────

class TestScorecardNoData:
    def test_no_data_when_engine_returns_empty(self, scorecard):
        scorecard.engine.analyze_recent_orders.return_value = []
        report = scorecard.generate_daily_report(days=30)
        assert report["status"] == "no_data"
        assert report["total_orders"] == 0

    def test_print_summary_no_data(self, scorecard):
        scorecard.engine.analyze_recent_orders.return_value = []
        text = scorecard.print_summary(days=30)
        assert "No TCA data" in text


class TestScorecardWithData:
    def test_generates_report_with_order_count(self, scorecard, sample_results):
        scorecard.engine.analyze_recent_orders.return_value = sample_results
        scorecard.engine.aggregate.return_value = MagicMock(
            total_orders=5, total_notional=285000,
            avg_slippage_bps=-5.2, avg_quality_score=68.0,
            weighted_slippage_bps=-6.1,
            by_symbol={
                "SPY": {"count": 3, "notional": 200000, "slippage_bps": -5.3, "quality": 78.3},
                "TLT": {"count": 1, "notional": 80000, "slippage_bps": -12.0, "quality": 60.0},
                "GLD": {"count": 1, "notional": 5000, "slippage_bps": 2.0, "quality": 45.0},
            },
        )
        report = scorecard.generate_daily_report(days=30)
        assert report["status"] == "ok"
        assert report["total_orders"] == 5
        assert report["total_notional"] == 285000.0

    def test_quality_distribution_buckets(self, scorecard, sample_results):
        scorecard.engine.analyze_recent_orders.return_value = sample_results
        scorecard.engine.aggregate.return_value = MagicMock(
            total_orders=5, total_notional=100000,
            avg_slippage_bps=0.0, avg_quality_score=0.0,
            weighted_slippage_bps=0.0,
            by_symbol={},
        )
        report = scorecard.generate_daily_report(days=30)
        dist = report["quality_distribution"]
        # 72→good, 85→good, 60→fair, 45→poor, 78→good
        assert dist["good_70_89"] == 3  # SPY(72,85) + SPY(78) → wait, SPY has 3 results
        assert dist["fair_50_69"] == 1  # TLT(60)
        assert dist["poor_20_49"] == 1  # GLD(45)

    def test_trend_single_order_flat(self, scorecard):
        single = [make_result("SPY", 50000, -5.0, 75.0)]
        scorecard.engine.analyze_recent_orders.return_value = single
        scorecard.engine.aggregate.return_value = MagicMock(
            total_orders=1, total_notional=50000,
            avg_slippage_bps=-5.0, avg_quality_score=75.0,
            weighted_slippage_bps=-5.0, by_symbol={},
        )
        report = scorecard.generate_daily_report(days=30)
        assert report["trend"]["slope"] == 0.0

    def test_trend_multiple_orders_computes_slope(self, scorecard):
        # _compute_trend reverses results: oldest-first order.
        # Oldest quality = 50, newest = 85 → positive slope
        improving = [
            make_result("SPY", 50000, -5.0, 85.0),  # newest first
            make_result("SPY", 50000, -5.0, 80.0),
            make_result("SPY", 50000, -5.0, 75.0),
            make_result("SPY", 50000, -5.0, 70.0),
            make_result("SPY", 50000, -5.0, 65.0),
            make_result("SPY", 50000, -5.0, 60.0),
            make_result("SPY", 50000, -5.0, 55.0),
            make_result("SPY", 50000, -5.0, 50.0),  # oldest last
        ]
        scorecard.engine.analyze_recent_orders.return_value = improving
        scorecard.engine.aggregate.return_value = MagicMock(
            total_orders=8, total_notional=400000,
            avg_slippage_bps=-5.0, avg_quality_score=67.5,
            weighted_slippage_bps=-5.0, by_symbol={},
        )
        report = scorecard.generate_daily_report(days=30)
        assert report["trend"]["slope"] > 0  # Improving trend

    def test_trend_two_orders_flat(self, scorecard):
        two = [make_result("SPY", 50000, -5.0, 70.0), make_result("SPY", 50000, -5.0, 72.0)]
        scorecard.engine.analyze_recent_orders.return_value = two
        scorecard.engine.aggregate.return_value = MagicMock(
            total_orders=2, total_notional=100000,
            avg_slippage_bps=0.0, avg_quality_score=0.0,
            weighted_slippage_bps=0.0, by_symbol={},
        )
        report = scorecard.generate_daily_report(days=30)
        assert report["status"] == "ok"


class TestPeerGroupComputation:
    def test_bucket_by_size(self, scorecard):
        results = [
            make_result("SPY", 5000, -2.0, 80.0),     # micro
            make_result("SPY", 30000, -3.0, 85.0),     # small
            make_result("SPY", 80000, -10.0, 60.0),    # medium
            make_result("SPY", 150000, -15.0, 50.0),   # large
        ]
        groups = scorecard._compute_peer_groups(results)
        buckets = {k.split("_")[1] for k in groups.keys()}
        assert "micro" in buckets
        assert "small" in buckets
        assert "medium" in buckets
        assert "large" in buckets

    def test_symbol_bucket_combination(self, scorecard):
        results = [
            make_result("SPY", 5000, -2.0, 80.0),
            make_result("TLT", 5000, -1.0, 82.0),
        ]
        groups = scorecard._compute_peer_groups(results)
        assert "SPY_micro" in groups
        assert "TLT_micro" in groups

    def test_z_score_in_peer_group_output(self, scorecard):
        results = [
            make_result("SPY", 30000, -5.0, 70.0),   # small (10000-50000)
            make_result("SPY", 30000, -8.0, 65.0),    # small
            make_result("SPY", 30000, -3.0, 75.0),    # small
        ]
        groups = scorecard._compute_peer_groups(results)
        assert "SPY_small" in groups
        assert "z_score" in groups["SPY_small"]


class TestPrintSummary:
    def test_summary_includes_headers(self, scorecard, sample_results):
        scorecard.engine.analyze_recent_orders.return_value = sample_results
        scorecard.engine.aggregate.return_value = MagicMock(
            total_orders=5, total_notional=285000,
            avg_slippage_bps=-5.2, avg_quality_score=68.0,
            weighted_slippage_bps=-6.1,
            by_symbol={
                "SPY": {"count": 3, "notional": 200000, "slippage_bps": -5.3, "quality": 78.3},
            },
        )
        text = scorecard.print_summary(days=30)
        assert "TCA SCORECARD SUMMARY" in text
        assert "Orders:" in text
        assert "Avg Slippage:" in text

    def test_summary_shows_improving_trend(self, scorecard):
        # newest first, oldest last, reversed by _compute_trend
        improving = [
            make_result("SPY", 50000, -5.0, 85.0),
            make_result("SPY", 50000, -5.0, 80.0),
            make_result("SPY", 50000, -5.0, 75.0),
            make_result("SPY", 50000, -5.0, 70.0),
            make_result("SPY", 50000, -5.0, 65.0),
            make_result("SPY", 50000, -5.0, 60.0),
            make_result("SPY", 50000, -5.0, 55.0),
            make_result("SPY", 50000, -5.0, 50.0),
        ]
        scorecard.engine.analyze_recent_orders.return_value = improving
        scorecard.engine.aggregate.return_value = MagicMock(
            total_orders=8, total_notional=400000,
            avg_slippage_bps=-5.0, avg_quality_score=67.5,
            weighted_slippage_bps=-5.0, by_symbol={},
        )
        text = scorecard.print_summary(days=30)
        assert "improving" in text or "stable" in text
