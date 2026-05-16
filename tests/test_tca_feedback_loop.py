"""
Tests for TCA-to-Execution Feedback Loop (v6.05)

Tests the feedback loop module that bridges TCA quality scores
back into the execution scheduling layer.
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.execution.tca_feedback_loop import (
    TCAFeedbackLoop,
    FeedbackState,
    SymbolExecutionProfile,
    apply_urgency_adjustment,
    apply_min_trade_adjustment,
    apply_cost_calibration,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_scorecard():
    """Sample TCA scorecard data for testing."""
    return {
        "total_orders": 25,
        "total_notional": 125000.0,
        "avg_slippage_bps": -3.5,
        "avg_quality_score": 72.3,
        "weighted_slippage_bps": -2.8,
        "by_symbol": {
            "SPY": {"count": 10, "notional": 50000.0, "slippage_bps": -2.1, "quality": 78.5},
            "GLD": {"count": 8, "notional": 40000.0, "slippage_bps": -4.2, "quality": 65.0},
            "TLT": {"count": 7, "notional": 35000.0, "slippage_bps": -3.8, "quality": 71.0},
        },
        "peer_groups": {
            "SPY_small": {
                "symbol": "SPY", "size_bucket": "small", "count": 5,
                "mean_slippage_bps": -2.0, "std_slippage_bps": 1.5, "mean_quality": 80.0, "z_score": -1.33,
            },
            "SPY_medium": {
                "symbol": "SPY", "size_bucket": "medium", "count": 5,
                "mean_slippage_bps": -2.2, "std_slippage_bps": 1.8, "mean_quality": 77.0, "z_score": -1.22,
            },
            "GLD_small": {
                "symbol": "GLD", "size_bucket": "small", "count": 8,
                "mean_slippage_bps": -4.2, "std_slippage_bps": 2.5, "mean_quality": 65.0, "z_score": -1.68,
            },
            "TLT_small": {
                "symbol": "TLT", "size_bucket": "small", "count": 7,
                "mean_slippage_bps": -3.8, "std_slippage_bps": 2.0, "mean_quality": 71.0, "z_score": -1.90,
            },
        },
        "trend": {
            "scores": [70, 72, 68, 74, 75, 73, 76, 72, 71, 74],
            "slope": 0.15,
            "recent_avg": 72.4,
            "overall_avg": 72.5,
        },
        "quality_distribution": {"excellent_90_100": 2, "good_70_89": 15, "fair_50_69": 6, "poor_20_49": 2, "bad_0_19": 0},
        "status": "ok",
    }


@pytest.fixture
def feedback_state_json():
    """Sample persisted feedback state."""
    return {
        "version": "6.05",
        "generated": "2026-05-16T12:00:00",
        "overall_quality": 72.5,
        "urgency_global_offset": -0.05,
        "min_trade_global_multiplier": 1.35,
        "cost_calibration_global": 1.15,
        "symbols": {
            "SPY": {
                "total_orders": 10,
                "avg_slippage_bps": -2.1,
                "avg_quality": 78.5,
                "trend_slope": 0.15,
                "recent_quality": 77.2,
                "slippage_volatility": 1.65,
                "quality_bucket": "good",
                "urgency_offset": 0.0,
                "min_trade_multiplier": 1.0,
                "cost_calibration": 1.0,
            },
            "GLD": {
                "total_orders": 8,
                "avg_slippage_bps": -4.2,
                "avg_quality": 65.0,
                "trend_slope": -0.08,
                "recent_quality": 63.5,
                "slippage_volatility": 2.5,
                "quality_bucket": "fair",
                "urgency_offset": -0.10,
                "min_trade_multiplier": 1.5,
                "cost_calibration": 1.3,
            },
            "TLT": {
                "total_orders": 7,
                "avg_slippage_bps": -3.8,
                "avg_quality": 71.0,
                "trend_slope": 0.05,
                "recent_quality": 72.0,
                "slippage_volatility": 2.0,
                "quality_bucket": "good",
                "urgency_offset": 0.0,
                "min_trade_multiplier": 1.0,
                "cost_calibration": 1.0,
            },
        },
        "quality_timeline": [70.0, 72.5, 71.0, 73.5, 72.5],
        "total_adjustments": 5,
        "status": "active",
    }


@pytest.fixture
def feedback_loop(tmp_path):
    """Create a TCAFeedbackLoop with a temp data directory."""
    return TCAFeedbackLoop(data_dir=tmp_path)


# ---------------------------------------------------------------------------
# Test: SymbolExecutionProfile
# ---------------------------------------------------------------------------


class TestSymbolExecutionProfile:
    """Test the SymbolExecutionProfile data class."""

    def test_feedback_quality_excellent(self):
        """Feedback quality should be high for excellent execution."""
        profile = SymbolExecutionProfile(
            symbol="SPY", total_orders=10, avg_slippage_bps=-1.0,
            avg_quality=95.0, trend_slope=0.2, recent_avg_quality=94.0,
            slippage_volatility=0.5, quality_bucket="excellent",
        )
        assert profile.feedback_quality > 90

    def test_feedback_quality_poor_with_trend_penalty(self):
        """Poor quality with deteriorating trend should score very low."""
        profile = SymbolExecutionProfile(
            symbol="GLD", total_orders=5, avg_slippage_bps=-10.0,
            avg_quality=25.0, trend_slope=-1.5, recent_avg_quality=20.0,
            slippage_volatility=5.0, quality_bucket="bad",
        )
        assert profile.feedback_quality < 30

    def test_feedback_quality_vol_penalty(self):
        """High slippage volatility should reduce feedback quality."""
        profile_low_vol = SymbolExecutionProfile(
            symbol="SPY", total_orders=10, avg_slippage_bps=-2.0,
            avg_quality=75.0, trend_slope=0.0, recent_avg_quality=75.0,
            slippage_volatility=0.5, quality_bucket="good",
        )
        profile_high_vol = SymbolExecutionProfile(
            symbol="GLD", total_orders=10, avg_slippage_bps=-2.0,
            avg_quality=75.0, trend_slope=0.0, recent_avg_quality=75.0,
            slippage_volatility=8.0, quality_bucket="good",
        )
        assert profile_low_vol.feedback_quality > profile_high_vol.feedback_quality

    def test_feedback_quality_clamped(self):
        """Feedback quality should be clamped 0-100."""
        profile = SymbolExecutionProfile(
            symbol="SPY", total_orders=1, avg_slippage_bps=0.0,
            avg_quality=5.0, trend_slope=-5.0, recent_avg_quality=3.0,
            slippage_volatility=10.0, quality_bucket="bad",
        )
        assert profile.feedback_quality >= 0.0

    def test_feedback_quality_improving(self):
        """Improving trend should maintain better score."""
        profile_improving = SymbolExecutionProfile(
            symbol="SPY", total_orders=10, avg_slippage_bps=-2.0,
            avg_quality=70.0, trend_slope=2.0, recent_avg_quality=72.0,
            slippage_volatility=1.0, quality_bucket="good",
        )
        profile_declining = SymbolExecutionProfile(
            symbol="GLD", total_orders=10, avg_slippage_bps=-2.0,
            avg_quality=70.0, trend_slope=-2.0, recent_avg_quality=68.0,
            slippage_volatility=1.0, quality_bucket="good",
        )
        assert profile_improving.feedback_quality > profile_declining.feedback_quality


# ---------------------------------------------------------------------------
# Test: FeedbackState
# ---------------------------------------------------------------------------


class TestFeedbackState:
    """Test the FeedbackState data class."""

    def test_to_dict_defaults(self):
        """Default state should serialize without errors."""
        state = FeedbackState()
        d = state.to_dict()
        assert d["version"] == "6.05"
        assert d["overall_quality"] == 75.0
        assert d["status"] == "ok"
        assert d["symbols"] == {}

    def test_to_dict_with_symbols(self):
        """State with symbols should include adjustment data."""
        state = FeedbackState(
            overall_quality=65.0,
            symbols={
                "GLD": SymbolExecutionProfile(
                    symbol="GLD", total_orders=5, avg_slippage_bps=-5.0,
                    avg_quality=55.0, trend_slope=-0.3, recent_avg_quality=52.0,
                    slippage_volatility=3.0, quality_bucket="fair",
                    urgency_offset=-0.10, min_trade_multiplier=1.5,
                    cost_calibration_factor=1.3,
                ),
            },
        )
        d = state.to_dict()
        assert "GLD" in d["symbols"]
        gld = d["symbols"]["GLD"]
        assert gld["urgency_offset"] == -0.10
        assert gld["min_trade_multiplier"] == 1.5
        assert gld["cost_calibration"] == 1.3


# ---------------------------------------------------------------------------
# Test: TCAFeedbackLoop initialization
# ---------------------------------------------------------------------------


class TestTCAFeedbackLoopInit:
    """Test initialization and state loading."""

    def test_init_default_dir(self):
        """Should use default data dir."""
        loop = TCAFeedbackLoop()
        assert loop.data_dir == TCAFeedbackLoop.DATA_DIR

    def test_init_custom_dir(self, tmp_path):
        """Should accept custom data dir."""
        loop = TCAFeedbackLoop(data_dir=tmp_path)
        assert loop.data_dir == tmp_path

    def test_init_no_state_file(self, tmp_path):
        """Should create default state when no file exists."""
        loop = TCAFeedbackLoop(data_dir=tmp_path)
        assert loop.state.status == "ok"
        assert loop.state.overall_quality == 75.0
        assert loop.state.symbols == {}

    def test_init_loads_state(self, tmp_path, feedback_state_json):
        """Should load existing state from file."""
        state_path = tmp_path / "tca_feedback_state.json"
        with open(state_path, "w") as f:
            json.dump(feedback_state_json, f)

        loop = TCAFeedbackLoop(data_dir=tmp_path)
        assert loop.state.overall_quality == 72.5
        assert "SPY" in loop.state.symbols
        assert "GLD" in loop.state.symbols
        assert loop.state.symbols["SPY"].avg_quality == 78.5

    def test_init_bad_state_file(self, tmp_path):
        """Should handle corrupted state file gracefully."""
        state_path = tmp_path / "tca_feedback_state.json"
        with open(state_path, "w") as f:
            f.write("{invalid json")

        loop = TCAFeedbackLoop(data_dir=tmp_path)
        # Should create a default state instead of crashing
        assert loop.state.status == "ok"


# ---------------------------------------------------------------------------
# Test: Scorecard loading
# ---------------------------------------------------------------------------


class TestScorecardLoading:
    """Test TCA scorecard data loading."""

    def test_load_scorecard_from_cache(self, feedback_loop, sample_scorecard):
        """Should load scorecard from cached JSON."""
        scorecard_path = feedback_loop.data_dir / "tca_scorecard.json"
        with open(scorecard_path, "w") as f:
            json.dump(sample_scorecard, f)

        result = feedback_loop._load_scorecard()
        assert result is not None
        assert result["total_orders"] == 25
        assert result["status"] == "ok"

    def test_load_scorecard_no_data_status(self, feedback_loop):
        """Should skip scorecard with no data status."""
        scorecard_path = feedback_loop.data_dir / "tca_scorecard.json"
        with open(scorecard_path, "w") as f:
            json.dump({"status": "no_data", "total_orders": 0}, f)

        result = feedback_loop._load_scorecard()
        assert result is None  # Falls through to try running scorecard

    def test_load_scorecard_no_cache(self, feedback_loop):
        """Should run scorecard when no cache exists."""
        # No scorecard file — will try to import TCAScorecard
        result = feedback_loop._load_scorecard()
        # May be None if TCAScorecard has deps issues, but shouldn't crash
        assert result is None or isinstance(result, dict)

    def test_fetch_symbol_from_scorecard(self, feedback_loop, sample_scorecard):
        """Should extract symbol-specific data from scorecard."""
        result = feedback_loop._fetch_symbol_from_scorecard(sample_scorecard, "SPY")
        assert result is not None
        assert result["direct"]["quality"] == 78.5
        assert result["direct"]["count"] == 10
        assert len(result["peer_groups"]) == 2  # SPY_small + SPY_medium

    def test_fetch_symbol_from_scorecard_missing(self, feedback_loop, sample_scorecard):
        """Should handle missing symbol gracefully."""
        result = feedback_loop._fetch_symbol_from_scorecard(sample_scorecard, "BTC")
        assert result is not None
        assert result["direct"] is None
        assert result["peer_groups"] == []


# ---------------------------------------------------------------------------
# Test: Quality bucket computation
# ---------------------------------------------------------------------------


class TestQualityBucket:
    """Test quality bucket classification."""

    def test_excellent(self, feedback_loop):
        assert feedback_loop._compute_quality_bucket(95) == "excellent"
        assert feedback_loop._compute_quality_bucket(90) == "excellent"

    def test_good(self, feedback_loop):
        assert feedback_loop._compute_quality_bucket(80) == "good"
        assert feedback_loop._compute_quality_bucket(70) == "good"

    def test_fair(self, feedback_loop):
        assert feedback_loop._compute_quality_bucket(60) == "fair"
        assert feedback_loop._compute_quality_bucket(50) == "fair"

    def test_poor(self, feedback_loop):
        assert feedback_loop._compute_quality_bucket(35) == "poor"
        assert feedback_loop._compute_quality_bucket(20) == "poor"

    def test_bad(self, feedback_loop):
        assert feedback_loop._compute_quality_bucket(10) == "bad"
        assert feedback_loop._compute_quality_bucket(0) == "bad"

    def test_boundaries(self, feedback_loop):
        """Test all quality bucket boundaries."""
        assert feedback_loop._compute_quality_bucket(89) == "good"  # Below excellent
        assert feedback_loop._compute_quality_bucket(69) == "fair"  # Below good
        assert feedback_loop._compute_quality_bucket(49) == "poor"  # Below fair
        assert feedback_loop._compute_quality_bucket(19) == "bad"   # Below poor


# ---------------------------------------------------------------------------
# Test: Adjustment computation
# ---------------------------------------------------------------------------


class TestAdjustmentComputation:
    """Test the mapping from quality to numeric adjustments."""

    def test_urgency_excellent(self, feedback_loop):
        """Excellent quality should get a slight urgency bonus."""
        offset = feedback_loop._urgency_from_quality(95, 0.2)
        assert offset == 0.05

    def test_urgency_good(self, feedback_loop):
        """Good quality should get neutral urgency."""
        offset = feedback_loop._urgency_from_quality(80, 0.1)
        assert offset == 0.0

    def test_urgency_fair(self, feedback_loop):
        """Fair quality should get small urgency penalty."""
        offset = feedback_loop._urgency_from_quality(60, 0.0)
        assert offset == -0.10

    def test_urgency_poor(self, feedback_loop):
        """Poor quality should get moderate urgency penalty."""
        offset = feedback_loop._urgency_from_quality(35, 0.0)
        assert offset == -0.20

    def test_urgency_bad(self, feedback_loop):
        """Bad quality should get severe urgency penalty."""
        offset = feedback_loop._urgency_from_quality(10, 0.0)
        assert offset == -0.30

    def test_min_trade_excellent(self, feedback_loop):
        """Excellent quality should reduce min trade threshold."""
        mult = feedback_loop._min_trade_from_quality(95, 0.2)
        assert mult == 0.9

    def test_min_trade_good(self, feedback_loop):
        """Good quality should keep base threshold."""
        mult = feedback_loop._min_trade_from_quality(80, 0.1)
        assert mult == 1.0

    def test_min_trade_fair(self, feedback_loop):
        """Fair quality should increase min trade threshold."""
        mult = feedback_loop._min_trade_from_quality(60, 0.0)
        assert mult == 1.5

    def test_min_trade_poor(self, feedback_loop):
        mult = feedback_loop._min_trade_from_quality(35, 0.0)
        assert mult == 2.0

    def test_min_trade_bad(self, feedback_loop):
        mult = feedback_loop._min_trade_from_quality(10, 0.0)
        assert mult == 3.0

    def test_cost_calibration_excellent(self, feedback_loop):
        """Excellent quality should reduce cost estimate factor."""
        cal = feedback_loop._cost_calibration_from_quality(95, 0.2)
        assert cal == 0.8

    def test_cost_calibration_good(self, feedback_loop):
        cal = feedback_loop._cost_calibration_from_quality(80, 0.1)
        assert cal == 1.0

    def test_cost_calibration_fair(self, feedback_loop):
        cal = feedback_loop._cost_calibration_from_quality(60, 0.0)
        assert cal == 1.3

    def test_cost_calibration_poor(self, feedback_loop):
        cal = feedback_loop._cost_calibration_from_quality(35, 0.0)
        assert cal == 1.6

    def test_cost_calibration_bad(self, feedback_loop):
        cal = feedback_loop._cost_calibration_from_quality(10, 0.0)
        assert cal == 2.0


# ---------------------------------------------------------------------------
# Test: Symbol profile computation
# ---------------------------------------------------------------------------


class TestSymbolProfileComputation:
    """Test computing per-symbol execution profiles from scorecard data."""

    def test_compute_symbol_profile_direct(self, feedback_loop, sample_scorecard):
        """Should compute profile from direct scorecard data."""
        profile = feedback_loop._compute_symbol_profile("SPY", sample_scorecard)
        assert profile is not None
        assert profile.symbol == "SPY"
        assert profile.avg_quality == 78.5
        assert profile.total_orders == 10

    def test_compute_symbol_profile_quality_bucket(self, feedback_loop, sample_scorecard):
        """Should assign correct quality bucket."""
        profile_spy = feedback_loop._compute_symbol_profile("SPY", sample_scorecard)
        profile_gld = feedback_loop._compute_symbol_profile("GLD", sample_scorecard)
        assert profile_spy.quality_bucket == "good"
        assert profile_gld.quality_bucket == "fair"

    def test_compute_symbol_profile_adjustments(self, feedback_loop, sample_scorecard):
        """Should compute urgency/min_trade/cost adjustments."""
        profile_gld = feedback_loop._compute_symbol_profile("GLD", sample_scorecard)
        assert profile_gld.urgency_offset == -0.10  # Fair quality
        assert profile_gld.min_trade_multiplier == 1.5
        assert profile_gld.cost_calibration_factor == 1.3

    def test_compute_symbol_profile_with_trend(self, feedback_loop, sample_scorecard):
        """Should use trend data from scorecard."""
        profile = feedback_loop._compute_symbol_profile("SPY", sample_scorecard)
        assert profile.trend_slope == 0.15  # From scorecard overall trend

    def test_compute_symbol_profile_missing(self, feedback_loop, sample_scorecard):
        """Should return None for unknown symbol with no existing state."""
        profile = feedback_loop._compute_symbol_profile("BTC", sample_scorecard)
        assert profile is None

    def test_compute_symbol_profile_returns_existing(self, feedback_loop, sample_scorecard):
        """Should return existing state for symbols with no scorecard data."""
        # Set up existing state
        existing = SymbolExecutionProfile(
            symbol="BTC", total_orders=3, avg_slippage_bps=-5.0,
            avg_quality=60.0, trend_slope=0.1, recent_avg_quality=62.0,
            slippage_volatility=2.0, quality_bucket="fair",
        )
        feedback_loop.state.symbols["BTC"] = existing

        profile = feedback_loop._compute_symbol_profile("BTC", sample_scorecard)
        assert profile is not None
        assert profile.symbol == "BTC"
        assert profile.avg_quality == 60.0  # From existing state


# ---------------------------------------------------------------------------
# Test: Global adjustments
# ---------------------------------------------------------------------------


class TestGlobalAdjustments:
    """Test global (aggregate) adjustment computation."""

    def test_global_quality_no_symbols(self, feedback_loop):
        """Should return default for empty state."""
        quality = feedback_loop._compute_global_quality()
        assert quality == 75.0

    def test_global_quality_weighted(self, feedback_loop):
        """Should compute weighted average of per-symbol quality."""
        feedback_loop.state.symbols = {
            "SPY": SymbolExecutionProfile(
                symbol="SPY", total_orders=10, avg_slippage_bps=-2.0,
                avg_quality=80.0, trend_slope=0.0, recent_avg_quality=80.0,
                slippage_volatility=1.0, quality_bucket="good",
            ),
            "GLD": SymbolExecutionProfile(
                symbol="GLD", total_orders=5, avg_slippage_bps=-4.0,
                avg_quality=60.0, trend_slope=0.0, recent_avg_quality=60.0,
                slippage_volatility=2.0, quality_bucket="fair",
            ),
        }
        # Weighted: (80*10 + 60*5) / 15 = 73.33
        quality = feedback_loop._compute_global_quality()
        assert quality == pytest.approx(73.33, abs=0.1)

    def test_global_adjustments_weighted(self, feedback_loop):
        """Should compute weighted global adjustment factors."""
        feedback_loop.state.symbols = {
            "SPY": SymbolExecutionProfile(
                symbol="SPY", total_orders=10, avg_slippage_bps=-2.0,
                avg_quality=80.0, trend_slope=0.0, recent_avg_quality=80.0,
                slippage_volatility=1.0, quality_bucket="good",
                urgency_offset=0.0, min_trade_multiplier=1.0, cost_calibration_factor=1.0,
            ),
            "GLD": SymbolExecutionProfile(
                symbol="GLD", total_orders=5, avg_slippage_bps=-4.0,
                avg_quality=60.0, trend_slope=0.0, recent_avg_quality=60.0,
                slippage_volatility=2.0, quality_bucket="fair",
                urgency_offset=-0.10, min_trade_multiplier=1.5, cost_calibration_factor=1.3,
            ),
        }
        urg, trade, cost = feedback_loop._compute_global_adjustments()
        # urgency global: offset = ((0*10 + -0.10*5) / 15) / 2 = -0.0167
        # min trade: (1*10 + 1.5*5) / 15 = 1.167
        # cost cal: (1*10 + 1.3*5) / 15 = 1.10
        assert urg == pytest.approx(-0.0167, abs=0.01)
        assert trade == pytest.approx(1.167, abs=0.01)
        assert cost == pytest.approx(1.10, abs=0.01)


# ---------------------------------------------------------------------------
# Test: Full feedback generation
# ---------------------------------------------------------------------------


class TestFeedbackGeneration:
    """Test the full feedback generation pipeline."""

    def test_generate_feedback_with_scorecard(self, feedback_loop, sample_scorecard):
        """Should generate feedback from scorecard data."""
        state = feedback_loop.generate_feedback(scorecard=sample_scorecard)
        assert state.status == "active"
        assert "SPY" in state.symbols
        assert "GLD" in state.symbols
        assert state.previous_adjustments == 1

    def test_generate_feedback_no_data(self, feedback_loop):
        """Should handle missing data gracefully."""
        state = feedback_loop.generate_feedback(scorecard=None)
        assert state.status == "no_tca_data"

    def test_generate_feedback_updates_quality_timeline(self, feedback_loop, sample_scorecard):
        """Should append overall quality to timeline."""
        state = feedback_loop.generate_feedback(scorecard=sample_scorecard)
        assert len(state.previous_quality) == 1

        # Second call should append
        state2 = feedback_loop.generate_feedback(scorecard=sample_scorecard)
        assert len(state2.previous_quality) == 2

    def test_generate_feedback_persists(self, feedback_loop, sample_scorecard):
        """Should save state to disk after generation."""
        state_path = feedback_loop.data_dir / "tca_feedback_state.json"
        assert not state_path.exists()

        feedback_loop.generate_feedback(scorecard=sample_scorecard)
        assert state_path.exists()

        # Verify content
        with open(state_path) as f:
            data = json.load(f)
        assert data["overall_quality"] > 0
        assert "SPY" in data["symbols"]

    def test_generate_feedback_updates_existing_state(self, feedback_loop, sample_scorecard):
        """Should update existing state rather than replacing."""
        # First run
        feedback_loop.generate_feedback(scorecard=sample_scorecard)
        first_adjustments = feedback_loop.state.previous_adjustments

        # Second run
        feedback_loop.generate_feedback(scorecard=sample_scorecard)
        assert feedback_loop.state.previous_adjustments == first_adjustments + 1


# ---------------------------------------------------------------------------
# Test: Getting adjustments
# ---------------------------------------------------------------------------


class TestGetAdjustments:
    """Test retrieving adjustment data."""

    def test_get_adjustments_no_data(self, feedback_loop):
        """Should return default adjustments when no data."""
        adj = feedback_loop.get_adjustments()
        assert adj["status"] == "no_data"
        assert adj["urgency_offsets"] == {}
        assert adj["overall_quality"] == 75.0

    def test_get_adjustments_after_feedback(self, feedback_loop, sample_scorecard):
        """Should return computed adjustments after feedback generation."""
        feedback_loop.generate_feedback(scorecard=sample_scorecard)
        adj = feedback_loop.get_adjustments()
        assert adj["status"] == "active"
        assert "SPY" in adj["urgency_offsets"]
        assert "GLD" in adj["urgency_offsets"]
        assert "TLT" in adj["urgency_offsets"]
        assert adj["overall_quality"] > 0
        assert adj["total_adjustments"] >= 1

    def test_get_adjustments_urgency_offsets(self, feedback_loop, sample_scorecard):
        """GLD (fair quality) should have negative urgency offset."""
        feedback_loop.generate_feedback(scorecard=sample_scorecard)
        adj = feedback_loop.get_adjustments()
        # GLD quality = 65 (fair) -> urgency_offset = -0.10
        assert adj["urgency_offsets"]["GLD"] == -0.10
        # SPY quality = 78.5 (good) -> urgency_offset = 0.0
        assert adj["urgency_offsets"]["SPY"] == 0.0

    def test_get_adjustments_min_trade_multipliers(self, feedback_loop, sample_scorecard):
        """GLD should have higher min trade multiplier."""
        feedback_loop.generate_feedback(scorecard=sample_scorecard)
        adj = feedback_loop.get_adjustments()
        assert adj["min_trade_multipliers"]["GLD"] == 1.5
        assert adj["min_trade_multipliers"]["SPY"] == 1.0


# ---------------------------------------------------------------------------
# Test: Integration helper functions
# ---------------------------------------------------------------------------


class TestIntegrationHelpers:
    """Test the apply_* helper functions."""

    def test_apply_urgency_adjustment_basic(self):
        """Should apply per-symbol urgency offset."""
        feedback = {
            "urgency_offsets": {"SPY": 0.0, "GLD": -0.10},
            "global_urgency_offset": 0.0,
        }
        result = apply_urgency_adjustment(0.5, 0.5, "GLD", feedback)
        # base combined = (0.5 + 0.5)/2 = 0.5, with -0.10 offset = 0.4
        assert result == pytest.approx(0.40)

    def test_apply_urgency_adjustment_global(self):
        """Should apply global urgency offset."""
        feedback = {
            "urgency_offsets": {"SPY": 0.0, "GLD": -0.10},
            "global_urgency_offset": -0.05,
        }
        result = apply_urgency_adjustment(0.5, 0.5, "SPY", feedback)
        # combined = 0.5, symbol offset = 0.0, global offset = -0.05 => 0.45
        assert result == pytest.approx(0.45)

    def test_apply_urgency_adjustment_missing_symbol(self):
        """Should handle missing symbol gracefully."""
        feedback = {
            "urgency_offsets": {"SPY": 0.0},
            "global_urgency_offset": 0.0,
        }
        result = apply_urgency_adjustment(0.5, 0.5, "UNKNOWN", feedback)
        assert result == 0.5  # No adjustment

    def test_apply_min_trade_adjustment(self):
        """Should apply min trade multiplier."""
        feedback = {
            "min_trade_multipliers": {"SPY": 1.0, "GLD": 2.0},
            "global_min_trade_multiplier": 1.0,
        }
        base = 1000.0
        adjusted = apply_min_trade_adjustment(base, "GLD", feedback)
        assert adjusted == 2000.0  # 1000 * 2.0

    def test_apply_min_trade_adjustment_global(self):
        """Should apply global min trade multiplier."""
        feedback = {
            "min_trade_multipliers": {"SPY": 1.0},
            "global_min_trade_multiplier": 1.35,
        }
        adjusted = apply_min_trade_adjustment(1000.0, "SPY", feedback)
        assert adjusted == 1350.0

    def test_apply_cost_calibration(self):
        """Should apply cost calibration factor."""
        feedback = {
            "cost_calibration_factors": {"TLT": 1.5},
            "global_cost_calibration": 1.0,
        }
        calibrated = apply_cost_calibration(5.0, "TLT", feedback)
        assert calibrated == 7.5  # 5.0 * 1.5

    def test_apply_cost_calibration_global_and_symbol(self):
        """Should apply both symbol and global factors."""
        feedback = {
            "cost_calibration_factors": {"TLT": 1.3},
            "global_cost_calibration": 1.15,
        }
        calibrated = apply_cost_calibration(5.0, "TLT", feedback)
        assert calibrated == pytest.approx(7.475)  # 5.0 * 1.3 * 1.15

    def test_apply_all_missing_symbol(self):
        """All helpers should handle missing symbols."""
        feedback = {
            "urgency_offsets": {},
            "min_trade_multipliers": {},
            "cost_calibration_factors": {},
            "global_urgency_offset": 0.0,
            "global_min_trade_multiplier": 1.0,
            "global_cost_calibration": 1.0,
        }
        assert apply_urgency_adjustment(0.5, 0.5, "SPY", feedback) == 0.5
        assert apply_min_trade_adjustment(1000.0, "SPY", feedback) == 1000.0
        assert apply_cost_calibration(5.0, "SPY", feedback) == 5.0


# ---------------------------------------------------------------------------
# Test: Reset
# ---------------------------------------------------------------------------


class TestReset:
    """Test resetting feedback state."""

    def test_reset_clears_state(self, feedback_loop, sample_scorecard):
        """Should clear all state and symbols."""
        # Populate state
        feedback_loop.generate_feedback(scorecard=sample_scorecard)
        assert len(feedback_loop.state.symbols) > 0

        # Reset
        feedback_loop.reset()
        assert feedback_loop.state.symbols == {}
        assert feedback_loop.state.overall_quality == 75.0
        assert feedback_loop.state.previous_adjustments == 0

    def test_reset_persists(self, feedback_loop, sample_scorecard):
        """Should persist cleared state to disk."""
        feedback_loop.generate_feedback(scorecard=sample_scorecard)
        feedback_loop.reset()

        with open(feedback_loop.data_dir / "tca_feedback_state.json") as f:
            data = json.load(f)
        assert data["symbols"] == {}
        assert data["overall_quality"] == 75.0


# ---------------------------------------------------------------------------
# Test: Print summary
# ---------------------------------------------------------------------------


class TestPrintSummary:
    """Test the human-readable summary output."""

    def test_summary_empty(self, feedback_loop):
        """Should produce output when no data."""
        summary = feedback_loop.print_summary()
        assert "TCA-TO-EXECUTION FEEDBACK LOOP" in summary
        assert "75.0" in summary  # Default quality

    def test_summary_with_data(self, feedback_loop, sample_scorecard):
        """Should include symbol-level data in summary."""
        feedback_loop.generate_feedback(scorecard=sample_scorecard)
        summary = feedback_loop.print_summary()
        assert "SPY" in summary
        assert "GLD" in summary
        assert "TLT" in summary
        assert "active" in summary

    def test_summary_includes_adjustments(self, feedback_loop, sample_scorecard):
        """Should display adjustment values."""
        feedback_loop.generate_feedback(scorecard=sample_scorecard)
        summary = feedback_loop.print_summary()
        # Should show per-symbol detail header
        assert "Per-Symbol Adjustments" in summary

    def test_summary_interpretation_section(self, feedback_loop):
        """Should include interpretation guidance."""
        summary = feedback_loop.print_summary()
        assert "Interpretation" in summary
        assert "Urgency Offset" in summary
        assert "Min Trade" in summary
        assert "Cost Calibration" in summary


# ---------------------------------------------------------------------------
# Test: Integration scenario
# ---------------------------------------------------------------------------


class TestFullIntegration:
    """End-to-end test of the feedback loop."""

    def test_full_cycle(self, tmp_path, sample_scorecard):
        """Test a complete feedback cycle."""
        loop = TCAFeedbackLoop(data_dir=tmp_path)

        # Step 1: No data — should return defaults
        assert loop.get_adjustments()["overall_quality"] == 75.0

        # Step 2: Generate feedback from scorecard
        state = loop.generate_feedback(scorecard=sample_scorecard)
        assert state.status == "active"
        assert len(state.symbols) == 3  # SPY, GLD, TLT

        # Step 3: Get adjustments for execution layer
        adj = loop.get_adjustments()
        assert len(adj["urgency_offsets"]) == 3
        assert len(adj["min_trade_multipliers"]) == 3
        assert len(adj["cost_calibration_factors"]) == 3

        # Step 4: Apply adjustments
        spy_urgency = apply_urgency_adjustment(0.6, 0.7, "SPY", adj)
        gld_urgency = apply_urgency_adjustment(0.6, 0.7, "GLD", adj)
        spy_min_trade = apply_min_trade_adjustment(1000.0, "SPY", adj)
        gld_min_trade = apply_min_trade_adjustment(1000.0, "GLD", adj)

        # SPY: good quality, no urgency penalty → higher urgency combination
        # GLD: fair quality, -0.10 urgency penalty → lower urgency
        assert spy_urgency > gld_urgency
        assert gld_min_trade > spy_min_trade  # GLD needs larger trades

        # Step 5: Verify persistence
        assert (tmp_path / "tca_feedback_state.json").exists()

        # Step 6: Reload and verify
        loop2 = TCAFeedbackLoop(data_dir=tmp_path)
        adj2 = loop2.get_adjustments()
        assert adj2["overall_quality"] == pytest.approx(adj["overall_quality"], abs=0.1)
        assert adj2["urgency_offsets"] == adj["urgency_offsets"]

    def test_quality_improves_adjustments(self, tmp_path):
        """Test that improving quality relaxes adjustments."""
        loop = TCAFeedbackLoop(data_dir=tmp_path)

        # Start with poor execution
        poor_scorecard = {
            "total_orders": 10,
            "total_notional": 50000.0,
            "avg_slippage_bps": -12.0,
            "avg_quality_score": 25.0,
            "weighted_slippage_bps": -10.0,
            "by_symbol": {
                "SPY": {"count": 10, "notional": 50000.0, "slippage_bps": -12.0, "quality": 25.0},
            },
            "peer_groups": {
                "SPY_small": {
                    "symbol": "SPY", "size_bucket": "small", "count": 10,
                    "mean_slippage_bps": -12.0, "std_slippage_bps": 3.0, "mean_quality": 25.0, "z_score": -4.0,
                },
            },
            "trend": {"scores": [25, 25, 25], "slope": 0.0, "recent_avg": 25.0, "overall_avg": 25.0},
            "quality_distribution": {},
            "status": "ok",
        }
        loop.generate_feedback(scorecard=poor_scorecard)
        poor_adj = loop.get_adjustments()

        # Now improve execution
        good_scorecard = {**poor_scorecard}
        good_scorecard["by_symbol"]["SPY"]["quality"] = 85.0
        good_scorecard["avg_quality_score"] = 85.0
        good_scorecard["peer_groups"]["SPY_small"]["mean_quality"] = 85.0
        good_scorecard["trend"]["scores"] = [80, 82, 85]
        good_scorecard["trend"]["slope"] = 2.5
        good_scorecard["trend"]["recent_avg"] = 85.0

        loop.generate_feedback(scorecard=good_scorecard)
        good_adj = loop.get_adjustments()

        # Better quality → less restrictive adjustments
        assert poor_adj["urgency_offsets"]["SPY"] < good_adj["urgency_offsets"]["SPY"]
        assert poor_adj["min_trade_multipliers"]["SPY"] > good_adj["min_trade_multipliers"]["SPY"]
        assert poor_adj["overall_quality"] < good_adj["overall_quality"]
