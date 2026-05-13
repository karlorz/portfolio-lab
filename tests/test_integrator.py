#!/usr/bin/env python3
"""
Tests for signal integrator — data structures, normalization, allocation deltas.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from src.signals.integrator import (
    SignalSourceResult, CompositeSignal, AllocationDelta,
    PortfolioRecommendation, SignalSource,
)


class TestDataStructures:
    """Test dataclass serialization."""

    def test_signal_source_result_to_dict(self):
        r = SignalSourceResult(
            timestamp=datetime.now().isoformat(),
            source_type="technical",
            source_name="momentum",
            signal=0.5,
            confidence=0.8,
            raw_score=1.2,
            raw_unit="return_12m",
            historical_accuracy=0.65,
            metadata={"lookback": 252},
        )
        d = r.to_dict()
        assert d["signal"] == 0.5
        assert d["confidence"] == 0.8
        assert d["source_type"] == "technical"

    def test_composite_signal_to_dict(self):
        c = CompositeSignal(
            timestamp=datetime.now().isoformat(),
            ticker="SPY",
            composite_score=0.3,
            composite_confidence=0.7,
            detected_regime="normal",
            primary_drivers=["momentum"],
        )
        d = c.to_dict()
        assert d["ticker"] == "SPY"
        assert d["composite_score"] == 0.3
        assert d["detected_regime"] == "normal"

    def test_allocation_delta_to_dict(self):
        a = AllocationDelta(
            ticker="SPY",
            current_weight=0.46,
            recommended_weight=0.50,
            delta=0.04,
            composite_score=0.5,
            confidence=0.8,
            primary_reason="momentum",
        )
        d = a.to_dict()
        assert d["delta"] == 0.04

    def test_portfolio_recommendation_to_dict(self):
        p = PortfolioRecommendation(
            timestamp=datetime.now().isoformat(),
            current_allocation={"SPY": 0.46, "GLD": 0.38},
            recommended_allocation={"SPY": 0.50, "GLD": 0.34},
            deltas=[],
            composite_sentiment="bullish",
            confidence=0.7,
            regime="normal",
        )
        d = p.to_dict()
        assert d["composite_sentiment"] == "bullish"
        assert d["regime"] == "normal"


class TestNormalizeSignal:
    """Test signal normalization to [-1, 1]."""

    def _make_source(self):
        """Create a minimal SignalSource subclass for testing."""
        class TestSource(SignalSource):
            def generate_signal(self, ticker):
                return None
            def get_historical_accuracy(self, ticker, horizon_days=21):
                return None
        return TestSource("test", "test")

    def test_midpoint(self):
        s = self._make_source()
        assert s._normalize_signal(0.0, -1.0, 1.0) == 0.0

    def test_max_maps_to_one(self):
        s = self._make_source()
        assert s._normalize_signal(1.0, -1.0, 1.0) == 1.0

    def test_min_maps_to_neg_one(self):
        s = self._make_source()
        assert s._normalize_signal(-1.0, -1.0, 1.0) == -1.0

    def test_clipping(self):
        s = self._make_source()
        assert s._normalize_signal(5.0, -1.0, 1.0) == 1.0
        assert s._normalize_signal(-5.0, -1.0, 1.0) == -1.0

    def test_equal_range_returns_zero(self):
        s = self._make_source()
        assert s._normalize_signal(0.5, 0.5, 0.5) == 0.0

    def test_asymmetric_range(self):
        s = self._make_source()
        # Range [0, 10], value 5 → midpoint → 0.0
        assert s._normalize_signal(5.0, 0.0, 10.0) == 0.0
        # Range [0, 10], value 10 → 1.0
        assert s._normalize_signal(10.0, 0.0, 10.0) == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
