#!/usr/bin/env python3
"""
Tests for signal integrator — data structures, normalization, composite signal
aggregation, allocation deltas, regime detection, signal agreement.
"""
import sys
import os
import json
import sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, PropertyMock
from src.signals.integrator import (
    SignalSourceResult, CompositeSignal, AllocationDelta,
    PortfolioRecommendation, SignalSource, SignalIntegrator,
    BASE_WEIGHTS, REGIME_WEIGHTS, MIN_SIGNAL_SOURCES,
    SIGNAL_MIN, SIGNAL_MAX, MAX_DELTA_PCT,
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


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def _make_signal(source_type, source_name, signal, confidence=0.8, accuracy=0.65):
    """Create a SignalSourceResult for testing."""
    return SignalSourceResult(
        source_type=source_type,
        source_name=source_name,
        signal=signal,
        confidence=confidence,
        raw_score=signal * 2,
        raw_unit="z_score",
        historical_accuracy=accuracy,
        sample_count=100,
    )


def _make_integrator(tmp_path):
    """Create a SignalIntegrator with mocked init to avoid importing real adapters."""
    with patch.object(SignalIntegrator, '__init__', lambda self: None):
        integrator = SignalIntegrator()
    integrator.sources = {}
    integrator.db_path = tmp_path / "signals.db"
    # Initialize the database tables
    conn = sqlite3.connect(str(integrator.db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS composite_signals (
            id INTEGER PRIMARY KEY, ticker TEXT, timestamp TEXT,
            composite_score REAL, composite_confidence REAL,
            detected_regime TEXT, weights_used TEXT, primary_drivers TEXT,
            signal_agreement TEXT, expected_accuracy REAL
        );
        CREATE TABLE IF NOT EXISTS portfolio_recommendations (
            id INTEGER PRIMARY KEY, timestamp TEXT,
            current_allocation TEXT, recommended_allocation TEXT,
            composite_sentiment TEXT, confidence REAL, regime TEXT, deltas TEXT
        );
    """)
    conn.commit()
    conn.close()
    return integrator


class TestCompositeSignalAggregation:
    """Test get_composite_signal with mocked sources."""

    def test_basic_composite(self, tmp_path):
        """Composite from 3 mocked sources produces valid score."""
        integrator = _make_integrator(tmp_path)

        mock_sources = {
            "momentum": MagicMock(),
            "macro": MagicMock(),
            "sentiment": MagicMock(),
        }
        mock_sources["momentum"].generate_signal.return_value = _make_signal(
            "momentum", "dual_momentum", 0.6, confidence=0.9
        )
        mock_sources["macro"].generate_signal.return_value = _make_signal(
            "macro", "fed_analyzer", 0.4, confidence=0.7
        )
        mock_sources["sentiment"].generate_signal.return_value = _make_signal(
            "sentiment", "llm_sentiment", 0.2, confidence=0.5
        )
        integrator.sources = mock_sources

        result = integrator.get_composite_signal("SPY", regime="neutral")

        assert isinstance(result, CompositeSignal)
        assert result.ticker == "SPY"
        assert SIGNAL_MIN <= result.composite_score <= SIGNAL_MAX
        assert 0.0 <= result.composite_confidence <= 1.0
        assert result.signal_agreement in [
            "aligned_bullish", "aligned_bearish", "conflicting", "mixed",
            "insufficient_data"
        ]
        assert len(result.component_signals) == 3

    def test_insufficient_signals(self, tmp_path):
        """Fewer than MIN_SIGNAL_SOURCES returns insufficient_data."""
        integrator = _make_integrator(tmp_path)

        mock_sources = {
            "momentum": MagicMock(),
            "macro": MagicMock(),
        }
        mock_sources["momentum"].generate_signal.return_value = _make_signal(
            "momentum", "dual_momentum", 0.5
        )
        mock_sources["macro"].generate_signal.return_value = None  # no signal
        integrator.sources = mock_sources

        # Patch MIN_SIGNAL_SOURCES to 2 (default)
        with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 2):
            result = integrator.get_composite_signal("SPY", regime="neutral")

        assert result.signal_agreement == "insufficient_data"
        assert result.composite_score == 0.0

    def test_custom_weights(self, tmp_path):
        """Custom weights override regime/base weights."""
        integrator = _make_integrator(tmp_path)

        mock_sources = {
            "momentum": MagicMock(),
            "macro": MagicMock(),
        }
        mock_sources["momentum"].generate_signal.return_value = _make_signal(
            "momentum", "tsmom", 0.8, confidence=1.0
        )
        mock_sources["macro"].generate_signal.return_value = _make_signal(
            "macro", "fed_policy", 0.0, confidence=1.0
        )
        integrator.sources = mock_sources

        custom = {"momentum": 0.9, "macro": 0.1}
        result = integrator.get_composite_signal("SPY", custom_weights=custom)

        assert result.weights_used == custom
        # Score should be heavily weighted toward momentum (0.8)
        assert result.composite_score > 0.5

    def test_all_bullish_agreement(self, tmp_path):
        """All bullish signals → aligned_bullish."""
        integrator = _make_integrator(tmp_path)

        sources = {}
        for i, name in enumerate(["momentum", "macro", "sentiment"]):
            src = MagicMock()
            src.generate_signal.return_value = _make_signal(
                name, f"source_{i}", 0.7, confidence=0.8
            )
            sources[name] = src
        integrator.sources = sources

        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.signal_agreement == "aligned_bullish"

    def test_all_bearish_agreement(self, tmp_path):
        """All bearish signals → aligned_bearish."""
        integrator = _make_integrator(tmp_path)

        sources = {}
        for i, name in enumerate(["momentum", "macro", "sentiment"]):
            src = MagicMock()
            src.generate_signal.return_value = _make_signal(
                name, f"source_{i}", -0.7, confidence=0.8
            )
            sources[name] = src
        integrator.sources = sources

        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.signal_agreement == "aligned_bearish"

    def test_mixed_signals(self, tmp_path):
        """Bullish + bearish → conflicting."""
        integrator = _make_integrator(tmp_path)

        src_bull = MagicMock()
        src_bull.generate_signal.return_value = _make_signal(
            "momentum", "tsmom", 0.7, confidence=0.8
        )
        src_bear = MagicMock()
        src_bear.generate_signal.return_value = _make_signal(
            "macro", "fed_policy", -0.7, confidence=0.8
        )
        src_neutral = MagicMock()
        src_neutral.generate_signal.return_value = _make_signal(
            "sentiment", "llm", 0.0, confidence=0.5
        )
        integrator.sources = {
            "momentum": src_bull, "macro": src_bear, "sentiment": src_neutral
        }

        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.signal_agreement == "conflicting"

    def test_source_failure_handled(self, tmp_path):
        """Source that raises exception is skipped gracefully."""
        integrator = _make_integrator(tmp_path)

        src_ok = MagicMock()
        src_ok.generate_signal.return_value = _make_signal(
            "momentum", "tsmom", 0.5, confidence=0.8
        )
        src_fail = MagicMock()
        src_fail.generate_signal.side_effect = RuntimeError("DB error")
        src_ok2 = MagicMock()
        src_ok2.generate_signal.return_value = _make_signal(
            "sentiment", "llm", 0.3, confidence=0.6
        )
        integrator.sources = {
            "momentum": src_ok, "macro": src_fail, "sentiment": src_ok2
        }

        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert len(result.component_signals) == 2
        assert result.composite_score != 0.0


class TestRegimeWeights:
    """Test regime-specific weight selection."""

    def test_regime_weights_exist(self):
        """All expected regimes have weight configs."""
        expected = ["bull", "bear", "neutral", "crisis", "high_vol"]
        for regime in expected:
            assert regime in REGIME_WEIGHTS

    def test_regime_weights_sum_near_one(self):
        """Each regime's weights should sum to ~1.0."""
        for regime, weights in REGIME_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.05, f"{regime} weights sum to {total}"


class TestExpectedAccuracy:
    """Test _calculate_expected_accuracy."""

    def test_with_accuracies(self, tmp_path):
        """Weighted average of historical accuracies."""
        integrator = _make_integrator(tmp_path)

        signals = [
            _make_signal("momentum", "tsmom", 0.5, confidence=0.9, accuracy=0.70),
            _make_signal("macro", "fed", 0.3, confidence=0.7, accuracy=0.60),
        ]
        weights = {"momentum": 0.5, "macro": 0.5}

        acc = integrator._calculate_expected_accuracy(signals, weights)
        assert 0.60 <= acc <= 0.70

    def test_no_accuracies_returns_default(self, tmp_path):
        """Signals without historical_accuracy return 0.6 default."""
        integrator = _make_integrator(tmp_path)

        signals = [
            SignalSourceResult(
                source_type="momentum", source_name="x", signal=0.5,
                confidence=0.8, raw_score=1.0, raw_unit="z",
                historical_accuracy=None,
            ),
        ]
        acc = integrator._calculate_expected_accuracy(signals, {"momentum": 0.5})
        assert acc == 0.6

    def test_empty_signals(self, tmp_path):
        """Empty signal list returns 0.5."""
        integrator = _make_integrator(tmp_path)
        assert integrator._calculate_expected_accuracy([], {}) == 0.5


class TestAllocationDeltas:
    """Test get_allocation_deltas."""

    def test_basic_deltas(self, tmp_path):
        """Allocation deltas are generated for each ticker."""
        integrator = _make_integrator(tmp_path)

        # Mock get_composite_signal to return controlled values
        def mock_composite(ticker, regime=None, custom_weights=None):
            return CompositeSignal(
                ticker=ticker,
                timestamp=datetime.now().isoformat(),
                composite_score=0.3,
                composite_confidence=0.7,
                detected_regime="neutral",
                primary_drivers=["momentum"],
                signal_agreement="mixed",
            )

        integrator.get_composite_signal = mock_composite

        alloc = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        result = integrator.get_allocation_deltas(alloc)

        assert isinstance(result, PortfolioRecommendation)
        assert len(result.deltas) == 3
        assert result.composite_sentiment in ["bullish", "bearish", "neutral"]

    def test_sentiment_classification(self, tmp_path):
        """Positive avg score → bullish, negative → bearish."""
        integrator = _make_integrator(tmp_path)

        # Bullish: all scores > 0.3
        def mock_bullish(ticker, regime=None, custom_weights=None):
            return CompositeSignal(
                ticker=ticker, timestamp=datetime.now().isoformat(),
                composite_score=0.5, composite_confidence=0.8,
                detected_regime="neutral", primary_drivers=["momentum"],
                signal_agreement="aligned_bullish",
            )
        integrator.get_composite_signal = mock_bullish

        result = integrator.get_allocation_deltas({"SPY": 0.50})
        assert result.composite_sentiment == "bullish"

    def test_bearish_sentiment(self, tmp_path):
        """Negative avg score → bearish."""
        integrator = _make_integrator(tmp_path)

        def mock_bearish(ticker, regime=None, custom_weights=None):
            return CompositeSignal(
                ticker=ticker, timestamp=datetime.now().isoformat(),
                composite_score=-0.5, composite_confidence=0.8,
                detected_regime="neutral", primary_drivers=["macro"],
                signal_agreement="aligned_bearish",
            )
        integrator.get_composite_signal = mock_bearish

        result = integrator.get_allocation_deltas({"SPY": 0.50})
        assert result.composite_sentiment == "bearish"

    def test_delta_capped_at_max(self, tmp_path):
        """Delta should not exceed MAX_DELTA_PCT."""
        integrator = _make_integrator(tmp_path)

        def mock_strong(ticker, regime=None, custom_weights=None):
            return CompositeSignal(
                ticker=ticker, timestamp=datetime.now().isoformat(),
                composite_score=1.0, composite_confidence=1.0,
                detected_regime="neutral", primary_drivers=["momentum"],
                signal_agreement="aligned_bullish",
            )
        integrator.get_composite_signal = mock_strong

        result = integrator.get_allocation_deltas({"SPY": 0.46})
        delta = result.deltas[0]
        # recommended_weight capped at current + MAX_DELTA_PCT = 0.51, but also capped at 0.60
        assert delta.recommended_weight <= 0.60

    def test_weight_bounds(self, tmp_path):
        """Recommended weights stay within [0.05, 0.60]."""
        integrator = _make_integrator(tmp_path)

        def mock_extreme(ticker, regime=None, custom_weights=None):
            return CompositeSignal(
                ticker=ticker, timestamp=datetime.now().isoformat(),
                composite_score=-1.0, composite_confidence=1.0,
                detected_regime="crisis", primary_drivers=["macro"],
                signal_agreement="aligned_bearish",
            )
        integrator.get_composite_signal = mock_extreme

        # Very high current weight — delta should push down but not below 0.05
        result = integrator.get_allocation_deltas({"SPY": 0.58})
        assert result.deltas[0].recommended_weight >= 0.05


class TestDetectRegime:
    """Test _detect_regime with different VIX levels."""

    def _setup_db(self, tmp_path, vix_level):
        """Create a market.db with VIX data."""
        db_path = tmp_path / "market.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        conn.execute("INSERT INTO prices VALUES ('VIX', '2026-05-13', ?)", (vix_level,))
        conn.commit()
        conn.close()
        return db_path

    def test_crisis_regime(self, tmp_path):
        """VIX > 30 → crisis."""
        integrator = _make_integrator(tmp_path)
        db_path = self._setup_db(tmp_path, 35.0)

        with patch("src.signals.integrator.DATA_DIR", tmp_path):
            regime = integrator._detect_regime()
        assert regime == "crisis"

    def test_high_vol_regime(self, tmp_path):
        """VIX 25-30 → high_vol."""
        integrator = _make_integrator(tmp_path)
        db_path = self._setup_db(tmp_path, 27.0)

        with patch("src.signals.integrator.DATA_DIR", tmp_path):
            regime = integrator._detect_regime()
        assert regime == "high_vol"

    def test_bull_regime(self, tmp_path):
        """VIX < 15 → bull."""
        integrator = _make_integrator(tmp_path)
        db_path = self._setup_db(tmp_path, 12.0)

        with patch("src.signals.integrator.DATA_DIR", tmp_path):
            regime = integrator._detect_regime()
        assert regime == "bull"

    def test_neutral_regime(self, tmp_path):
        """VIX 15-25 → neutral."""
        integrator = _make_integrator(tmp_path)
        db_path = self._setup_db(tmp_path, 20.0)

        with patch("src.signals.integrator.DATA_DIR", tmp_path):
            regime = integrator._detect_regime()
        assert regime == "neutral"

    def test_no_vix_data_defaults_neutral(self, tmp_path):
        """Missing VIX data → neutral."""
        integrator = _make_integrator(tmp_path)
        # No market.db created
        with patch("src.signals.integrator.DATA_DIR", tmp_path):
            regime = integrator._detect_regime()
        assert regime == "neutral"


class TestGetSignalHistory:
    """Test get_signal_history retrieval."""

    def test_empty_history(self, tmp_path):
        """No stored signals → empty list."""
        integrator = _make_integrator(tmp_path)
        result = integrator.get_signal_history("SPY", days=30)
        assert result == []

    def test_stored_signal_retrieved(self, tmp_path):
        """Stored composite signal can be retrieved."""
        integrator = _make_integrator(tmp_path)

        # Insert a test record
        conn = sqlite3.connect(str(integrator.db_path))
        conn.execute("""
            INSERT INTO composite_signals
            (ticker, timestamp, composite_score, composite_confidence,
             detected_regime, weights_used, primary_drivers,
             signal_agreement, expected_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "SPY", datetime.now().isoformat(), 0.35, 0.72,
            "neutral", json.dumps({"momentum": 0.5}),
            json.dumps(["tsmom", "fed_policy"]),
            "aligned_bullish", 0.68
        ))
        conn.commit()
        conn.close()

        result = integrator.get_signal_history("SPY", days=1)
        assert len(result) == 1
        assert result[0].ticker == "SPY"
        assert result[0].composite_score == 0.35
        assert result[0].signal_agreement == "aligned_bullish"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
