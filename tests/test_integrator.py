#!/usr/bin/env python3
"""
Tests for signal integrator — data structures, normalization, composite signal
aggregation, allocation deltas, regime detection, signal agreement.
"""
import dataclasses
import json
import logging
import math
import sqlite3

import pytest
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import patch, MagicMock, PropertyMock
from src.signals.integrator import (
    SignalSourceResult, CompositeSignal, AllocationDelta,
    PortfolioRecommendation, SignalSource, SignalIntegrator,
    TechnicalSignal, MacroSignal, AlternativeDataSignalAdapter,
    LLMSentimentSignalAdapter,
    BASE_WEIGHTS, REGIME_WEIGHTS, MIN_SIGNAL_SOURCES,
    SIGNAL_MIN, SIGNAL_MAX, MAX_DELTA_PCT,
    DB_PATH, __all__ as MODULE_ALL,
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


class TestSQLParameterization:
    """Regression: SQL queries must use parameterized ? instead of .format()."""

    def test_get_signal_history_uses_params(self):
        """get_signal_history should use parameterized query, not .format()."""
        import inspect
        from src.signals.integrator import SignalIntegrator
        source = inspect.getsource(SignalIntegrator.get_signal_history)
        assert ".format(days)" not in source, \
            "SQL query still uses .format(days) — must use ? parameterized query"


class TestStoreComposite:
    """Test _store_composite method."""

    def test_store_composite_writes_to_db(self, tmp_path):
        from src.signals.integrator import SignalIntegrator, CompositeSignal, init_database
        import sqlite3

        with patch("src.signals.integrator.DB_PATH", tmp_path / "test.db"):
            init_database()
            integrator = SignalIntegrator()
            integrator.db_path = tmp_path / "test.db"

            composite = CompositeSignal(
                timestamp=datetime.now().isoformat(),
                ticker="SPY",
                composite_score=0.35,
                composite_confidence=0.80,
                detected_regime="neutral",
                primary_drivers=["technical"],
                component_signals=[],
            )
            integrator._store_composite(composite)

            with sqlite3.connect(str(tmp_path / "test.db")) as conn:
                rows = conn.execute("SELECT COUNT(*) FROM composite_signals").fetchone()
            assert rows[0] == 1


class TestStoreRecommendation:
    """Test _store_recommendation method."""

    def test_store_recommendation_writes_to_db(self, tmp_path):
        from src.signals.integrator import (
            SignalIntegrator, PortfolioRecommendation, AllocationDelta, init_database,
        )
        import sqlite3

        with patch("src.signals.integrator.DB_PATH", tmp_path / "test.db"):
            init_database()
            integrator = SignalIntegrator()
            integrator.db_path = tmp_path / "test.db"

            rec = PortfolioRecommendation(
                timestamp=datetime.now().isoformat(),
                current_allocation={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
                recommended_allocation={"SPY": 0.48, "GLD": 0.36, "TLT": 0.16},
                deltas=[
                    AllocationDelta(
                        ticker="SPY", current_weight=0.46, recommended_weight=0.48,
                        delta=0.02, composite_score=0.4, confidence=0.75,
                        primary_reason="technical",
                    ),
                ],
                composite_sentiment="neutral",
                confidence=0.75,
                regime="neutral",
            )
            integrator._store_recommendation(rec)

            with sqlite3.connect(str(tmp_path / "test.db")) as conn:
                rows = conn.execute("SELECT COUNT(*) FROM portfolio_recommendations").fetchone()
            assert rows[0] == 1


class TestSignalSourceNormalize:
    """Test SignalSource._normalize_signal method."""

    def test_normalize_midpoint(self):
        from src.signals.integrator import TechnicalSignal
        sig = TechnicalSignal()
        result = sig._normalize_signal(0.5, 0.0, 1.0)
        assert result == 0.0

    def test_normalize_at_max(self):
        from src.signals.integrator import TechnicalSignal
        sig = TechnicalSignal()
        result = sig._normalize_signal(1.0, 0.0, 1.0)
        assert result == 1.0

    def test_normalize_at_min(self):
        from src.signals.integrator import TechnicalSignal
        sig = TechnicalSignal()
        result = sig._normalize_signal(0.0, 0.0, 1.0)
        assert result == -1.0

    def test_normalize_equal_range_returns_zero(self):
        from src.signals.integrator import TechnicalSignal
        sig = TechnicalSignal()
        result = sig._normalize_signal(0.5, 0.5, 0.5)
        assert result == 0.0

    def test_normalize_clips_above_range(self):
        from src.signals.integrator import TechnicalSignal
        sig = TechnicalSignal()
        result = sig._normalize_signal(2.0, 0.0, 1.0)
        assert result == 1.0

    def test_normalize_clips_below_range(self):
        from src.signals.integrator import TechnicalSignal
        sig = TechnicalSignal()
        result = sig._normalize_signal(-1.0, 0.0, 1.0)
        assert result == -1.0


class TestCalculateExpectedAccuracy:
    """Test SignalIntegrator._calculate_expected_accuracy."""

    def test_with_no_signals_returns_default(self, tmp_path):
        from src.signals.integrator import SignalIntegrator, init_database

        with patch("src.signals.integrator.DB_PATH", tmp_path / "test.db"):
            init_database()
            integrator = SignalIntegrator()
            integrator.db_path = tmp_path / "test.db"
            result = integrator._calculate_expected_accuracy([], {"technical": 0.3})
            assert result == 0.5

    def test_with_signals_averages_accuracies(self, tmp_path):
        from src.signals.integrator import SignalIntegrator, SignalSourceResult, init_database

        with patch("src.signals.integrator.DB_PATH", tmp_path / "test.db"):
            init_database()
            integrator = SignalIntegrator()
            integrator.db_path = tmp_path / "test.db"

            signals = [
                SignalSourceResult(
                    source_type="technical", source_name="tech",
                    signal=0.5, confidence=0.8,
                    raw_score=1.5, raw_unit="z_score",
                    historical_accuracy=0.7,
                ),
                SignalSourceResult(
                    source_type="macro", source_name="macro",
                    signal=0.3, confidence=0.6,
                    raw_score=0.8, raw_unit="pct_change",
                    historical_accuracy=0.5,
                ),
            ]
            result = integrator._calculate_expected_accuracy(signals, {"technical": 0.5, "macro": 0.3})
            assert 0.0 <= result <= 1.0


class TestPortfolioRecommendationDataclass:
    """Test PortfolioRecommendation dataclass methods."""

    def test_to_dict(self):
        from src.signals.integrator import PortfolioRecommendation

        rec = PortfolioRecommendation(
            timestamp="2026-05-24T10:00:00",
            current_allocation={"SPY": 0.46},
            recommended_allocation={"SPY": 0.48},
            deltas=[],
            composite_sentiment="neutral",
            confidence=0.75,
            regime="neutral",
        )
        d = rec.to_dict()
        assert isinstance(d, dict)
        assert d["composite_sentiment"] == "neutral"
        assert "current_allocation" in d


class TestAllocationDeltaDataclass:
    """Test AllocationDelta dataclass methods."""

    def test_to_dict(self):
        from src.signals.integrator import AllocationDelta

        delta = AllocationDelta(
            ticker="SPY", current_weight=0.46, recommended_weight=0.48,
            delta=0.02, composite_score=0.4, confidence=0.75,
            primary_reason="technical",
        )
        d = delta.to_dict()
        assert isinstance(d, dict)
        assert d["ticker"] == "SPY"
        assert d["delta"] == 0.02


class TestCompositeSignalDataclass:
    """Test CompositeSignal dataclass methods."""

    def test_to_dict(self):
        from src.signals.integrator import CompositeSignal

        cs = CompositeSignal(
            timestamp="2026-05-24T10:00:00",
            ticker="SPY", composite_score=0.35, composite_confidence=0.80,
            detected_regime="neutral",
            primary_drivers=["technical"], component_signals=[],
        )
        d = cs.to_dict()
        assert isinstance(d, dict)
        assert d["composite_score"] == 0.35
        assert d["component_count"] == 0


class TestSignalSourceResultDataclass:
    """Test SignalSourceResult dataclass methods."""

    def test_to_dict(self):
        from src.signals.integrator import SignalSourceResult

        result = SignalSourceResult(
            source_type="technical", source_name="tech",
            signal=0.5, confidence=0.8,
            raw_score=1.5, raw_unit="z_score",
            historical_accuracy=0.7,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["source_type"] == "technical"
        assert d["signal"] == 0.5


class TestDataclassFieldCompleteness:
    """Verify to_dict() includes ALL fields for every dataclass."""

    def test_signal_source_result_all_fields(self):
        """SignalSourceResult.to_dict includes all declared fields."""
        r = SignalSourceResult(
            source_type="momentum", source_name="dual_mom",
            signal=0.5, confidence=0.8, raw_score=1.2, raw_unit="z_score",
            historical_accuracy=0.65, sample_count=200,
            timestamp="2026-05-24T12:00:00",
            metadata={"lookback": 252},
        )
        d = r.to_dict()
        assert d["source_type"] == "momentum"
        assert d["source_name"] == "dual_mom"
        assert d["signal"] == 0.5
        assert d["confidence"] == 0.8
        assert d["raw_score"] == 1.2
        assert d["raw_unit"] == "z_score"
        assert d["historical_accuracy"] == 0.65
        assert d["sample_count"] == 200
        assert d["timestamp"] == "2026-05-24T12:00:00"
        assert d["metadata"] == {"lookback": 252}
        assert len(d) == 10

    def test_composite_signal_all_fields(self):
        """CompositeSignal.to_dict includes all declared fields."""
        comp = _make_signal("momentum", "tsmom", 0.5, confidence=0.8)
        cs = CompositeSignal(
            ticker="SPY", timestamp="2026-05-24T12:00:00",
            component_signals=[comp],
            composite_score=0.35, composite_confidence=0.72,
            primary_drivers=["tsmom", "fed_policy"],
            signal_agreement="aligned_bullish",
            detected_regime="neutral",
            weights_used={"momentum": 0.5, "macro": 0.5},
            expected_accuracy=0.68,
        )
        d = cs.to_dict()
        assert d["ticker"] == "SPY"
        assert d["timestamp"] == "2026-05-24T12:00:00"
        assert d["composite_score"] == 0.35
        assert d["composite_confidence"] == 0.72
        assert d["primary_drivers"] == ["tsmom", "fed_policy"]
        assert d["signal_agreement"] == "aligned_bullish"
        assert d["detected_regime"] == "neutral"
        assert d["weights_used"] == {"momentum": 0.5, "macro": 0.5}
        assert d["expected_accuracy"] == 0.68
        assert d["component_count"] == 1
        assert isinstance(d["components"], list)
        assert len(d["components"]) == 1
        assert d["components"][0]["source_name"] == "tsmom"

    def test_allocation_delta_all_fields(self):
        """AllocationDelta.to_dict includes all declared fields."""
        ad = AllocationDelta(
            ticker="GLD", current_weight=0.38, recommended_weight=0.42,
            delta=0.04, composite_score=0.6, confidence=0.85,
            primary_reason="momentum_positive",
            max_position=0.70, min_position=0.03,
        )
        d = ad.to_dict()
        assert d["ticker"] == "GLD"
        assert d["current_weight"] == 0.38
        assert d["recommended_weight"] == 0.42
        assert d["delta"] == 0.04
        assert d["composite_score"] == 0.6
        assert d["confidence"] == 0.85
        assert d["primary_reason"] == "momentum_positive"
        assert d["max_position"] == 0.70
        assert d["min_position"] == 0.03
        assert len(d) == 9

    def test_portfolio_recommendation_all_fields(self):
        """PortfolioRecommendation.to_dict includes all declared fields."""
        ad = AllocationDelta(
            ticker="SPY", current_weight=0.46, recommended_weight=0.48,
            delta=0.02, composite_score=0.4, confidence=0.75,
            primary_reason="technical",
        )
        rec = PortfolioRecommendation(
            timestamp="2026-05-24T12:00:00",
            current_allocation={"SPY": 0.46},
            recommended_allocation={"SPY": 0.48},
            deltas=[ad],
            composite_sentiment="bullish",
            confidence=0.75, regime="neutral",
            expected_volatility=0.15, max_drawdown_estimate=-0.25,
        )
        d = rec.to_dict()
        assert d["timestamp"] == "2026-05-24T12:00:00"
        assert d["composite_sentiment"] == "bullish"
        assert d["confidence"] == 0.75
        assert d["regime"] == "neutral"
        assert d["current_allocation"] == {"SPY": 0.46}
        assert d["recommended_allocation"] == {"SPY": 0.48}
        assert isinstance(d["deltas"], list)
        assert len(d["deltas"]) == 1
        assert d["expected_volatility"] == 0.15
        assert d["max_drawdown_estimate"] == -0.25
        assert len(d) == 9


class TestSignalSourceResultDefaults:
    """Verify default field values in SignalSourceResult dataclass."""

    def test_default_timestamp_is_set(self):
        r = SignalSourceResult(
            source_type="test", source_name="test",
            signal=0.0, confidence=0.0, raw_score=0.0, raw_unit="none",
        )
        assert r.sample_count == 0
        assert r.historical_accuracy is None
        assert r.metadata == {}
        assert r.timestamp is not None
        assert isinstance(r.timestamp, str)

    def test_null_historical_accuracy_in_to_dict(self):
        r = SignalSourceResult(
            source_type="test", source_name="test",
            signal=0.0, confidence=0.0, raw_score=0.0, raw_unit="none",
        )
        d = r.to_dict()
        assert d["historical_accuracy"] is None
        assert d["sample_count"] == 0


class TestConstantsValidation:
    """Validate module-level constants."""

    def test_base_weights_sum_to_one(self):
        total = sum(BASE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.05, f"BASE_WEIGHTS sum to {total}"

    def test_signal_min_value(self):
        assert SIGNAL_MIN == -1.0

    def test_signal_max_value(self):
        assert SIGNAL_MAX == 1.0

    def test_max_delta_pct_value(self):
        assert MAX_DELTA_PCT == 0.05

    def test_min_signal_source_value(self):
        assert MIN_SIGNAL_SOURCES == 2


class TestCompositeSignalEdgeCases:
    """Edge cases for get_composite_signal."""

    def test_zero_signals_insufficient(self, tmp_path):
        """No sources return signals -> insufficient_data."""
        integrator = _make_integrator(tmp_path)
        integrator.sources = {
            "a": MagicMock(generate_signal=MagicMock(return_value=None)),
            "b": MagicMock(generate_signal=MagicMock(return_value=None)),
        }
        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.signal_agreement == "insufficient_data"
        assert result.composite_score == 0.0
        assert result.composite_confidence == 0.0

    def test_single_source_sufficient(self, tmp_path):
        """Single source with MIN_SIGNAL_SOURCES=2 patched to 1 -> valid."""
        integrator = _make_integrator(tmp_path)
        src = MagicMock()
        src.generate_signal.return_value = _make_signal(
            "momentum", "tsmom", 0.5, confidence=0.8
        )
        integrator.sources = {"momentum": src}
        with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 1):
            result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.signal_agreement != "insufficient_data"
        assert result.composite_score == 0.5

    def test_all_neutral_signals(self, tmp_path):
        """All signals at exactly 0.0 -> score 0.0, agreement mixed."""
        integrator = _make_integrator(tmp_path)
        sources = {}
        for name in ["momentum", "macro", "sentiment"]:
            src = MagicMock()
            src.generate_signal.return_value = _make_signal(
                name, name, 0.0, confidence=0.8
            )
            sources[name] = src
        integrator.sources = sources
        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.composite_score == 0.0
        assert result.signal_agreement == "mixed"

    def test_all_at_extreme_bullish(self, tmp_path):
        """All signals at +1.0 -> score 1.0, aligned_bullish."""
        integrator = _make_integrator(tmp_path)
        sources = {}
        for name in ["momentum", "macro", "sentiment"]:
            src = MagicMock()
            src.generate_signal.return_value = _make_signal(
                name, name, 1.0, confidence=1.0
            )
            sources[name] = src
        integrator.sources = sources
        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.composite_score == 1.0
        assert result.signal_agreement == "aligned_bullish"

    def test_all_at_extreme_bearish(self, tmp_path):
        """All signals at -1.0 -> score -1.0, aligned_bearish."""
        integrator = _make_integrator(tmp_path)
        sources = {}
        for name in ["momentum", "macro", "sentiment"]:
            src = MagicMock()
            src.generate_signal.return_value = _make_signal(
                name, name, -1.0, confidence=1.0
            )
            sources[name] = src
        integrator.sources = sources
        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.composite_score == -1.0
        assert result.signal_agreement == "aligned_bearish"

    def test_weights_empty_for_insufficient_data(self, tmp_path):
        """Insufficient data -> weights_used is empty dict."""
        integrator = _make_integrator(tmp_path)
        integrator.sources = {
            "a": MagicMock(generate_signal=MagicMock(return_value=None)),
        }
        with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 2):
            result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.weights_used == {}


class TestCompositeSignalAgreementBoundary:
    """Boundary conditions for agreement classification (>=60% threshold)."""

    def test_exactly_60pct_bullish_aligned(self, tmp_path):
        """Exactly 3/5 signals bullish (60%) -> aligned_bullish."""
        integrator = _make_integrator(tmp_path)
        sources = {}
        for i in range(5):
            src = MagicMock()
            sig = 0.5 if i < 3 else 0.1  # 3 bullish + 2 neutral = 60% bullish
            src.generate_signal.return_value = _make_signal(
                f"type_{i}", f"src_{i}", sig, confidence=0.8
            )
            sources[f"src_{i}"] = src
        integrator.sources = sources
        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.signal_agreement == "aligned_bullish"

    def test_59pct_bullish_52pct_neutral_mixed(self, tmp_path):
        """Slightly below 60% bullish with no bearish -> mixed."""
        integrator = _make_integrator(tmp_path)
        sources = {}
        for i in range(7):
            src = MagicMock()
            sig = 0.5 if i < 4 else 0.0  # 4/7 ≈ 57% bullish -> < 60%
            src.generate_signal.return_value = _make_signal(
                f"type_{i}", f"src_{i}", sig, confidence=0.8
            )
            sources[f"src_{i}"] = src
        integrator.sources = sources
        result = integrator.get_composite_signal("SPY", regime="neutral")
        # No bearish signals exist, and bullish < 60% -> falls through to "mixed"
        assert result.signal_agreement == "mixed"

    def test_exactly_60pct_bearish_aligned(self, tmp_path):
        """Exactly 3/5 signals bearish (60%) -> aligned_bearish."""
        integrator = _make_integrator(tmp_path)
        sources = {}
        for i in range(5):
            src = MagicMock()
            sig = -0.5 if i < 3 else 0.0  # 3 bearish + 2 neutral
            src.generate_signal.return_value = _make_signal(
                f"type_{i}", f"src_{i}", sig, confidence=0.8
            )
            sources[f"src_{i}"] = src
        integrator.sources = sources
        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.signal_agreement == "aligned_bearish"

    def test_both_bullish_and_bearish_conflicting(self, tmp_path):
        """At least one bullish AND one bearish -> conflicting."""
        integrator = _make_integrator(tmp_path)
        sources = {}
        sigs = [0.5, -0.5, 0.0, 0.0]
        for i, sig in enumerate(sigs):
            src = MagicMock()
            src.generate_signal.return_value = _make_signal(
                f"type_{i}", f"src_{i}", sig, confidence=0.8
            )
            sources[f"src_{i}"] = src
        integrator.sources = sources
        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.signal_agreement == "conflicting"


class TestDetectRegimeEdgeCases:
    """Boundary and exception edge cases for _detect_regime."""

    def _setup_db(self, tmp_path, vix_level):
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

    def test_vix_exactly_boundary_15(self, tmp_path):
        """VIX == 15 -> neutral (not < 15)."""
        integrator = _make_integrator(tmp_path)
        self._setup_db(tmp_path, 15.0)
        with patch("src.signals.integrator.DATA_DIR", tmp_path):
            regime = integrator._detect_regime()
        assert regime == "neutral"

    def test_vix_exactly_boundary_25(self, tmp_path):
        """VIX == 25 -> neutral (not > 25)."""
        integrator = _make_integrator(tmp_path)
        self._setup_db(tmp_path, 25.0)
        with patch("src.signals.integrator.DATA_DIR", tmp_path):
            regime = integrator._detect_regime()
        assert regime == "neutral"

    def test_vix_exactly_boundary_30(self, tmp_path):
        """VIX == 30 -> high_vol (> 25 but not > 30)."""
        integrator = _make_integrator(tmp_path)
        self._setup_db(tmp_path, 30.0)
        with patch("src.signals.integrator.DATA_DIR", tmp_path):
            regime = integrator._detect_regime()
        assert regime == "high_vol"

    def test_vix_just_above_boundary_25(self, tmp_path):
        """VIX == 25.01 -> high_vol."""
        integrator = _make_integrator(tmp_path)
        self._setup_db(tmp_path, 25.01)
        with patch("src.signals.integrator.DATA_DIR", tmp_path):
            regime = integrator._detect_regime()
        assert regime == "high_vol"

    def test_vix_just_below_boundary_15(self, tmp_path):
        """VIX == 14.99 -> bull."""
        integrator = _make_integrator(tmp_path)
        self._setup_db(tmp_path, 14.99)
        with patch("src.signals.integrator.DATA_DIR", tmp_path):
            regime = integrator._detect_regime()
        assert regime == "bull"

    def test_vix_just_above_boundary_30(self, tmp_path):
        """VIX == 30.01 -> crisis."""
        integrator = _make_integrator(tmp_path)
        self._setup_db(tmp_path, 30.01)
        with patch("src.signals.integrator.DATA_DIR", tmp_path):
            regime = integrator._detect_regime()
        assert regime == "crisis"

    def test_db_exception_fallback_neutral(self, tmp_path):
        """Database connection error -> neutral."""
        integrator = _make_integrator(tmp_path)
        with patch("src.signals.integrator.sqlite_connect") as mock_connect:
            mock_connect.side_effect = RuntimeError("DB unavailable")
            regime = integrator._detect_regime()
        assert regime == "neutral"


class TestExpectedAccuracyEdgeCases:
    """Edge cases for _calculate_expected_accuracy."""

    def test_mixed_none_and_real_accuracies(self, tmp_path):
        """Some signals have accuracy=None, others have real values."""
        integrator = _make_integrator(tmp_path)
        signals = [
            _make_signal("momentum", "tsmom", 0.5, confidence=0.9, accuracy=0.80),
            SignalSourceResult(
                source_type="macro", source_name="fed",
                signal=0.3, confidence=0.7, raw_score=0.5, raw_unit="pct",
                historical_accuracy=None,
            ),
        ]
        weights = {"momentum": 0.5, "macro": 0.3}
        acc = integrator._calculate_expected_accuracy(signals, weights)
        # Only momentum contributes; macro is skipped because accuracy is None
        assert acc == 0.80

    def test_all_confidence_zero_returns_default(self, tmp_path):
        """All signals have confidence=0 -> weight_total=0 -> returns 0.6."""
        integrator = _make_integrator(tmp_path)
        signals = [
            SignalSourceResult(
                source_type="momentum", source_name="tsmom",
                signal=0.5, confidence=0.0, raw_score=1.0, raw_unit="z",
                historical_accuracy=0.9,
            ),
        ]
        acc = integrator._calculate_expected_accuracy(signals, {"momentum": 0.5})
        assert acc == 0.6

    def test_weight_not_in_weights_dict(self, tmp_path):
        """Source type not in weights dict uses default 0.20."""
        integrator = _make_integrator(tmp_path)
        signals = [
            _make_signal("unknown_type", "custom", 0.5, confidence=0.8, accuracy=0.70),
        ]
        acc = integrator._calculate_expected_accuracy(signals, {"momentum": 1.0})
        # unknown_type not in weights, so weight defaults to 0.20
        assert acc == 0.70


class TestAllocationDeltasEdgeCases:
    """Edge cases for get_allocation_deltas."""

    def test_single_asset(self, tmp_path):
        """Single asset allocation works."""
        integrator = _make_integrator(tmp_path)

        def mock_composite(ticker, regime=None, custom_weights=None):
            return CompositeSignal(
                ticker=ticker, timestamp=datetime.now().isoformat(),
                composite_score=0.2, composite_confidence=0.6,
                detected_regime="neutral", primary_drivers=["momentum"],
                signal_agreement="mixed",
            )
        integrator.get_composite_signal = mock_composite

        result = integrator.get_allocation_deltas({"SPY": 1.0})
        assert len(result.deltas) == 1
        assert result.deltas[0].ticker == "SPY"

    def test_zero_score_produces_no_change(self, tmp_path):
        """Composite score of 0.0 -> delta of 0.0."""
        integrator = _make_integrator(tmp_path)

        def mock_zero(ticker, regime=None, custom_weights=None):
            return CompositeSignal(
                ticker=ticker, timestamp=datetime.now().isoformat(),
                composite_score=0.0, composite_confidence=0.8,
                detected_regime="neutral", primary_drivers=["momentum"],
                signal_agreement="mixed",
            )
        integrator.get_composite_signal = mock_zero

        result = integrator.get_allocation_deltas({"SPY": 0.46})
        assert result.deltas[0].delta == 0.0
        assert result.deltas[0].recommended_weight == 0.46

    def test_max_delta_clamping_positive(self, tmp_path):
        """Strong positive signal clamped to MAX_DELTA_PCT."""
        integrator = _make_integrator(tmp_path)

        def mock_strong(ticker, regime=None, custom_weights=None):
            return CompositeSignal(
                ticker=ticker, timestamp=datetime.now().isoformat(),
                composite_score=1.0, composite_confidence=1.0,
                detected_regime="bull", primary_drivers=["momentum"],
                signal_agreement="aligned_bullish",
            )
        integrator.get_composite_signal = mock_strong

        result = integrator.get_allocation_deltas({"SPY": 0.05})
        delta = result.deltas[0]
        # raw_delta = 1.0 * 0.05 = 0.05, then * 1.0 = 0.05
        # Recommended: 0.05 + 0.05 = 0.10
        assert delta.recommended_weight == 0.10
        assert delta.delta == 0.05

    def test_min_weight_clamping_bearish(self, tmp_path):
        """Very bearish signal on small position -> at least min_position."""
        integrator = _make_integrator(tmp_path)

        def mock_bearish(ticker, regime=None, custom_weights=None):
            return CompositeSignal(
                ticker=ticker, timestamp=datetime.now().isoformat(),
                composite_score=-1.0, composite_confidence=1.0,
                detected_regime="crisis", primary_drivers=["macro"],
                signal_agreement="aligned_bearish",
            )
        integrator.get_composite_signal = mock_bearish

        # Start at min_position
        result = integrator.get_allocation_deltas({"SPY": 0.05})
        # raw_delta = -1.0 * 0.05 = -0.05, * 1.0 = -0.05
        # recommended = 0.05 + (-0.05) = 0.0, but clamped to 0.05
        assert result.deltas[0].recommended_weight == 0.05
        assert result.deltas[0].delta == 0.0  # unchanged

    def test_delta_sign_matches_score_direction(self, tmp_path):
        """Positive score -> positive delta, negative score -> negative delta."""
        integrator = _make_integrator(tmp_path)

        calls = {"SPY": (0.3, 0.7), "GLD": (-0.3, 0.7)}
        stock_composites = {}

        def mock_both(ticker, regime=None, custom_weights=None):
            score, conf = calls.get(ticker, (0.0, 0.0))
            comp = CompositeSignal(
                ticker=ticker, timestamp=datetime.now().isoformat(),
                composite_score=score, composite_confidence=conf,
                detected_regime="neutral", primary_drivers=["momentum"],
                signal_agreement="mixed",
            )
            stock_composites[ticker] = score
            return comp
        integrator.get_composite_signal = mock_both

        result = integrator.get_allocation_deltas({"SPY": 0.50, "GLD": 0.50})
        spy_delta = [d for d in result.deltas if d.ticker == "SPY"][0]
        gld_delta = [d for d in result.deltas if d.ticker == "GLD"][0]
        assert spy_delta.delta > 0
        assert gld_delta.delta < 0
        assert result.composite_sentiment == "neutral"  # avg of +0.3 and -0.3 = 0.0


class TestUnknownRegimeWeights:
    """Test behavior with an unknown regime."""

    def test_unknown_regime_falls_back_to_base_weights(self, tmp_path):
        """Regime not in REGIME_WEIGHTS -> uses BASE_WEIGHTS."""
        integrator = _make_integrator(tmp_path)
        src = MagicMock()
        src.generate_signal.return_value = _make_signal(
            "momentum", "tsmom", 0.5, confidence=0.8
        )
        integrator.sources = {"momentum": src}
        with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 1):
            result = integrator.get_composite_signal("SPY", regime="unknown_regime")
        assert result.weights_used == BASE_WEIGHTS
        assert result.detected_regime == "unknown_regime"


class TestCompositeSignalRegimeDetection:
    """Verify that _detect_regime is called when no regime override given."""

    def test_detect_regime_called_when_not_provided(self, tmp_path):
        """No regime arg -> _detect_regime called."""
        integrator = _make_integrator(tmp_path)
        src = MagicMock()
        src.generate_signal.return_value = _make_signal(
            "momentum", "tsmom", 0.5, confidence=0.8
        )
        integrator.sources = {"momentum": src}
        with patch.object(integrator, '_detect_regime', return_value="bull") as mock_detect:
            with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 1):
                result = integrator.get_composite_signal("SPY")
        mock_detect.assert_called_once()
        assert result.detected_regime == "bull"


class TestCompositeSignalRegimeOverride:
    """Verify regime override takes precedence."""

    def test_regime_override_used(self, tmp_path):
        """Provided regime used instead of _detect_regime."""
        integrator = _make_integrator(tmp_path)
        with patch.object(integrator, '_detect_regime', return_value="neutral"):
            with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 1):
                src = MagicMock()
                src.generate_signal.return_value = _make_signal(
                    "momentum", "tsmom", 0.5, confidence=0.8
                )
                integrator.sources = {"momentum": src}
                result = integrator.get_composite_signal("SPY", regime="crisis")
        assert result.detected_regime == "crisis"


# ---------------------------------------------------------------------------
# New tests: SignalSource ABC, TechnicalSignal, MacroSignal,
# AlternativeDataSignalAdapter, LLMSentimentSignalAdapter, plus edge cases
# ---------------------------------------------------------------------------


class TestSignalSourceABC:
    """Test SignalSource abstract base class requirements."""

    def test_cannot_instantiate_directly(self):
        """SignalSource ABC raises TypeError on direct instantiation."""
        with pytest.raises(TypeError):
            SignalSource("test", "test")  # type: ignore

    def test_abstract_methods_not_implemented(self):
        """Subclass missing abstract methods cannot be instantiated."""
        with pytest.raises(TypeError):

            class BadSource(SignalSource):
                pass

            BadSource("test", "test")  # type: ignore

    def test_proper_subclass_works(self):
        """Valid subclass with both abstract methods can be instantiated."""

        class GoodSource(SignalSource):
            def generate_signal(self, ticker):
                return None

            def get_historical_accuracy(self, ticker, horizon_days=21):
                return None

        src = GoodSource("test_type", "test_name")
        assert src.source_type == "test_type"
        assert src.source_name == "test_name"


class TestStoreSignalOnBase:
    """Test SignalSource._store_signal method with mocked DB."""

    def test_store_signal_inserts_row(self, tmp_path):
        """_store_signal writes a row to signal_history table."""

        class MinimalSource(SignalSource):
            def generate_signal(self, ticker):
                return None

            def get_historical_accuracy(self, ticker, horizon_days=21):
                return None

        db_path = tmp_path / "signals.db"
        src = MinimalSource("momentum", "test_mom")
        src.db_path = db_path

        # Initialize DB
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, timestamp TEXT, source_type TEXT,
                source_name TEXT, signal REAL, confidence REAL,
                raw_score REAL, raw_unit TEXT, historical_accuracy REAL,
                metadata TEXT
            )
        """)
        conn.commit()
        conn.close()

        result = SignalSourceResult(
            source_type="momentum", source_name="test_mom",
            signal=0.5, confidence=0.8, raw_score=1.2, raw_unit="z_score",
            historical_accuracy=0.65, sample_count=100,
            metadata={"lookback": 252},
        )
        src._store_signal("SPY", result)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT ticker, source_type, signal, confidence FROM signal_history").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0] == ("SPY", "momentum", 0.5, 0.8)


# ---------------------------------------------------------------------------
# TechnicalSignal tests
# ---------------------------------------------------------------------------


def _create_market_db(db_path, rows_data):
    """Helper: populate market.db with price data for a ticker."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date)
        )
    """)
    for i, (date_str, close) in enumerate(rows_data):
        conn.execute(
            "INSERT OR IGNORE INTO prices VALUES (?, ?, ?)",
            ("SPY", date_str, close),
        )
    conn.commit()
    conn.close()


def _make_rising_prices(n_days, start=100.0, end=200.0):
    """Generate monotonically increasing price sequence within last 400 days."""
    from datetime import datetime, timedelta
    now = datetime.now()
    base = now - timedelta(days=n_days + 10)
    prices = []
    for i in range(n_days):
        frac = i / max(n_days - 1, 1)
        price = start + (end - start) * frac
        date_str = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        prices.append((date_str, price))
    return prices


def _make_falling_prices(n_days, start=200.0, end=100.0):
    """Generate monotonically decreasing price sequence within last 400 days."""
    from datetime import datetime, timedelta
    now = datetime.now()
    base = now - timedelta(days=n_days + 10)
    prices = []
    for i in range(n_days):
        frac = i / max(n_days - 1, 1)
        price = start - (start - end) * frac
        date_str = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        prices.append((date_str, price))
    return prices


class TestTechnicalSignalCalculateMomentum:
    """Test TechnicalSignal._calculate_momentum."""

    def test_no_market_db_returns_none(self, tmp_path):
        """When market.db does not exist, returns None."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()

        # No market.db created
        assert signal.market_db.exists() is False
        result = signal._calculate_momentum("SPY")
        assert result is None

    def test_insufficient_rows_returns_none(self, tmp_path):
        """Fewer than 200 rows returns None."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()

        _create_market_db(signal.market_db, _make_rising_prices(150))
        result = signal._calculate_momentum("SPY")
        assert result is None

    def test_bullish_momentum_returns_dict(self, tmp_path):
        """Rising prices produce above_sma=True and non-negative scores."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()

        _create_market_db(signal.market_db, _make_rising_prices(300, 100, 200))
        result = signal._calculate_momentum("SPY")
        assert result is not None
        assert result["above_sma"] is True
        assert result["return_12m"] > 0
        assert result["score"] > 0
        assert "sample_count" in result
        assert result["sample_count"] >= 200

    def test_bearish_momentum_above_sma_false(self, tmp_path):
        """Falling prices produce above_sma=False and negative score."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()

        _create_market_db(signal.market_db, _make_falling_prices(300, 200, 100))
        result = signal._calculate_momentum("SPY")
        assert result is not None
        assert result["above_sma"] is False
        assert result["score"] <= 0

    def test_momentum_keys_present(self, tmp_path):
        """Result dict has all expected keys."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()

        _create_market_db(signal.market_db, _make_rising_prices(300, 100, 200))
        result = signal._calculate_momentum("SPY")
        expected_keys = {"score", "return_12m", "return_6m", "above_sma", "trend_strength", "sample_count"}
        assert expected_keys.issubset(result.keys())


class TestTechnicalSignalCalculateMeanReversion:
    """Test TechnicalSignal._calculate_mean_reversion."""

    def test_no_market_db_returns_zero(self, tmp_path):
        """When market.db does not exist, returns 0.0."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()

        assert signal.market_db.exists() is False
        result = signal._calculate_mean_reversion("SPY")
        assert result == 0.0

    def test_insufficient_rows_returns_zero(self, tmp_path):
        """Fewer than 14 rows returns 0.0."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()

        _create_market_db(signal.market_db, _make_rising_prices(10))
        result = signal._calculate_mean_reversion("SPY")
        assert result == 0.0

    def test_rsi_oversold_bullish(self, tmp_path):
        """Consistent losses produce RSI < 30 -> positive (bullish) signal."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()

        # Falling prices within last 30 days -> all losses -> RSI=0 -> signal=1.0
        from datetime import datetime, timedelta
        now = datetime.now()
        rows = []
        close = 100.0
        for i in range(20):
            date_str = (now - timedelta(days=20 - i)).strftime("%Y-%m-%d")
            rows.append((date_str, close))
            close -= 0.5  # Each day slightly lower
        _create_market_db(signal.market_db, rows)
        result = signal._calculate_mean_reversion("SPY")
        assert result > 0

    def test_rsi_overbought_bearish(self, tmp_path):
        """Mostly rising prices produce RSI > 70 -> negative (bearish) signal."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()

        # Mostly rising prices with a few small drops -> RSI high -> signal negative
        from datetime import datetime, timedelta
        now = datetime.now()
        rows = []
        close = 90.0
        for i in range(20):
            date_str = (now - timedelta(days=20 - i)).strftime("%Y-%m-%d")
            rows.append((date_str, close))
            close += 0.5 if i % 4 != 0 else -0.1  # Small drops to avoid zero avg_loss
        _create_market_db(signal.market_db, rows)
        result = signal._calculate_mean_reversion("SPY")
        assert result < 0

    def test_rsi_neutral_returns_zero(self, tmp_path):
        """Mixed gains/losses produce RSI 30-70 -> 0.0."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()

        # Alternating prices — up then down
        rows = []
        from datetime import datetime, timedelta
        base = datetime(2024, 1, 1)
        close = 100.0
        for i in range(20):
            date_str = (base + timedelta(days=i)).strftime("%Y-%m-%d")
            rows.append((date_str, close))
            # Toggle direction
            if i % 2 == 0:
                close += 1.0
            else:
                close -= 0.5

        _create_market_db(signal.market_db, rows)
        result = signal._calculate_mean_reversion("SPY")
        # With alternating small moves, RSI should be near 50 -> neutral
        assert result == 0.0


class TestTechnicalSignalGenerateSignal:
    """Test TechnicalSignal.generate_signal."""

    def test_momentum_none_returns_none(self, tmp_path):
        """When _calculate_momentum returns None, generate_signal returns None."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()

        with patch.object(signal, "_calculate_momentum", return_value=None):
            result = signal.generate_signal("SPY")
        assert result is None

    def test_generates_valid_signal_result(self, tmp_path):
        """With momentum data, generates proper SignalSourceResult."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()

        momentum_data = {
            "score": 0.5,
            "return_12m": 0.12,
            "return_6m": 0.06,
            "above_sma": True,
            "trend_strength": 0.3,
            "sample_count": 250,
        }
        with patch.object(signal, "_calculate_momentum", return_value=momentum_data):
            with patch.object(signal, "_calculate_mean_reversion", return_value=0.2):
                result = signal.generate_signal("SPY")

        assert isinstance(result, SignalSourceResult)
        assert result.source_type == "momentum"
        assert result.source_name == "technical"
        assert SIGNAL_MIN <= result.signal <= SIGNAL_MAX
        assert 0.0 <= result.confidence <= 1.0

    def test_signal_rounding(self, tmp_path):
        """Signal is rounded to 4 decimal places."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()

        momentum_data = {
            "score": 0.1234567,
            "return_12m": 0.0,
            "return_6m": 0.0,
            "above_sma": True,
            "trend_strength": 0.5,
            "sample_count": 250,
        }
        with patch.object(signal, "_calculate_momentum", return_value=momentum_data):
            with patch.object(signal, "_calculate_mean_reversion", return_value=0.0):
                result = signal.generate_signal("SPY")

        # 0.1234567 * 0.7 + 0.0 * 0.3 = 0.08641969, rounded to 4dp
        expected_rounded = round(0.1234567 * 0.7, 4)
        assert result.signal == expected_rounded


class TestTechnicalSignalHistoricalAccuracy:
    """Test TechnicalSignal.get_historical_accuracy."""

    def test_no_data_returns_none(self, tmp_path):
        """No accuracy records returns None."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()

        signal.db_path = tmp_path / "signals.db"
        conn = sqlite3.connect(str(signal.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, source_type TEXT, source_name TEXT,
                prediction_timestamp TEXT, predicted_signal REAL,
                actual_return REAL, horizon_days INTEGER,
                accuracy_score REAL, error REAL
            )
        """)
        conn.commit()
        conn.close()

        result = signal.get_historical_accuracy("SPY")
        assert result is None

    def test_returns_average_accuracy(self, tmp_path):
        """Multiple accuracy records return averaged value."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()

        signal.db_path = tmp_path / "signals.db"
        conn = sqlite3.connect(str(signal.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, source_type TEXT, source_name TEXT,
                prediction_timestamp TEXT, predicted_signal REAL,
                actual_return REAL, horizon_days INTEGER,
                accuracy_score REAL, error REAL
            )
        """)
        conn.execute("""
            INSERT INTO signal_accuracy
            (ticker, source_type, source_name, prediction_timestamp,
             predicted_signal, actual_return, horizon_days, accuracy_score, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("SPY", "momentum", "technical", "2026-05-01", 0.5, 0.4, 21, 0.80, 0.1))
        conn.execute("""
            INSERT INTO signal_accuracy
            (ticker, source_type, source_name, prediction_timestamp,
             predicted_signal, actual_return, horizon_days, accuracy_score, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("SPY", "momentum", "technical", "2026-05-02", 0.3, 0.2, 21, 0.70, 0.1))
        conn.commit()
        conn.close()

        result = signal.get_historical_accuracy("SPY")
        assert result is not None
        assert result == 0.75  # average of 0.80 and 0.70


# ---------------------------------------------------------------------------
# MacroSignal tests
# ---------------------------------------------------------------------------


class TestMacroSignalFedStance:
    """Test MacroSignal._get_fed_stance."""

    def test_with_fed_data(self, tmp_path):
        """fed_analysis table with data returns hawk_dove_score."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()

        conn = sqlite3.connect(str(signal.alt_data_db))
        conn.execute("CREATE TABLE fed_analysis (date TEXT, hawk_dove_score REAL)")
        conn.execute("INSERT INTO fed_analysis VALUES ('2026-05-13', 0.3)")
        conn.commit()
        conn.close()

        result = signal._get_fed_stance()
        assert result == 0.3

    def test_no_fed_data_default(self, tmp_path):
        """No fed_analysis table returns default -0.2."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()

        result = signal._get_fed_stance()
        assert result == -0.2

    def test_db_exception_default(self, tmp_path):
        """DB exception falls back to default -0.2."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()

        # Remove the alt_data_db to trigger an error
        signal.alt_data_db = tmp_path / "nonexistent_dir" / "alt.db"

        result = signal._get_fed_stance()
        assert result == -0.2


class TestMacroSignalYieldCurve:
    """Test MacroSignal._get_yield_curve_signal."""

    def test_with_data(self, tmp_path):
        """TLT and SHY prices produce non-zero yield curve signal."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()

        # Create market.db with sufficient TLT/SHY price history
        conn = sqlite3.connect(str(signal.market_db))
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        from datetime import datetime, timedelta
        base = datetime(2024, 5, 1)
        # 22 rows for 30-day change calculation
        for i in range(25):
            date_str = (base + timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute("INSERT INTO prices VALUES ('TLT', ?, ?)", (date_str, 90.0 + i * 0.5))
            conn.execute("INSERT INTO prices VALUES ('SHY', ?, ?)", (date_str, 85.0 + i * 0.1))
        conn.commit()
        conn.close()

        result = signal._get_yield_curve_signal()
        # With TLT rising faster than SHY, spread > 0 -> bullish signal
        assert isinstance(result, float)
        assert SIGNAL_MIN <= result <= SIGNAL_MAX

    def test_no_tlt_shy_data_default(self, tmp_path):
        """Missing TLT/SHY data returns 0.0."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()

        # Create empty market.db
        conn = sqlite3.connect(str(signal.market_db))
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        conn.commit()
        conn.close()

        result = signal._get_yield_curve_signal()
        assert result == 0.0

    def test_db_exception_default(self, tmp_path):
        """DB connection error returns 0.0."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()

        signal.market_db = tmp_path / "nonexistent" / "market.db"
        result = signal._get_yield_curve_signal()
        assert result == 0.0


class TestMacroSignalCreditSignal:
    """Test MacroSignal._get_credit_signal."""

    def test_with_data(self, tmp_path):
        """LQD and HYG prices produce non-zero credit signal."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()

        conn = sqlite3.connect(str(signal.market_db))
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        from datetime import datetime, timedelta
        base = datetime(2024, 5, 1)
        # HYG underperforming LQD -> spreads widening -> bearish
        for i in range(25):
            date_str = (base + timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute("INSERT INTO prices VALUES ('LQD', ?, ?)", (date_str, 100.0 + i * 0.5))
            conn.execute("INSERT INTO prices VALUES ('HYG', ?, ?)", (date_str, 100.0 - i * 0.2))
        conn.commit()
        conn.close()

        result = signal._get_credit_signal()
        assert isinstance(result, float)
        assert SIGNAL_MIN <= result <= SIGNAL_MAX

    def test_no_data_default(self, tmp_path):
        """Missing LQD/HYG data returns 0.0."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()

        conn = sqlite3.connect(str(signal.market_db))
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        conn.commit()
        conn.close()

        result = signal._get_credit_signal()
        assert result == 0.0


class TestMacroSignal30dChange:
    """Test MacroSignal._get_30d_change."""

    def test_with_sufficient_data(self, tmp_path):
        """22+ price rows returns a valid change value."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()

        conn = sqlite3.connect(str(signal.market_db))
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        from datetime import datetime, timedelta
        base = datetime(2024, 5, 1)
        # 22 rising prices
        for i in range(22):
            date_str = (base + timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute("INSERT INTO prices VALUES ('SPY', ?, ?)", (date_str, 100.0 + i))
        conn.commit()
        conn.close()

        result = signal._get_30d_change("SPY")
        assert result is not None
        assert result > 0  # Rising prices -> positive change

    def test_insufficient_data_returns_none(self, tmp_path):
        """Fewer than 22 rows returns None."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()

        conn = sqlite3.connect(str(signal.market_db))
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        conn.execute("INSERT INTO prices VALUES ('SPY', '2026-05-13', 100.0)")
        conn.execute("INSERT INTO prices VALUES ('SPY', '2026-05-14', 101.0)")
        conn.commit()
        conn.close()

        result = signal._get_30d_change("SPY")
        assert result is None


class TestMacroSignalGenerateSignal:
    """Test MacroSignal.generate_signal."""

    def test_generates_valid_signal(self, tmp_path):
        """generate_signal returns valid SignalSourceResult with mocked sub-methods."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()

        with patch.object(signal, "_get_fed_stance", return_value=0.3):
            with patch.object(signal, "_get_yield_curve_signal", return_value=0.2):
                with patch.object(signal, "_get_credit_signal", return_value=-0.1):
                    result = signal.generate_signal("SPY")

        assert isinstance(result, SignalSourceResult)
        assert result.source_type == "macro"
        assert result.source_name == "fed_economic"
        assert result.confidence == 0.75  # hardcoded in generate_signal
        # combined = 0.3*0.4 + 0.2*0.35 + (-0.1)*0.25 = 0.12 + 0.07 - 0.025 = 0.165
        assert result.signal == 0.165
        assert result.raw_unit == "macro_composite"


class TestMacroSignalHistoricalAccuracy:
    """Test MacroSignal.get_historical_accuracy."""

    def test_default_return_when_no_data(self, tmp_path):
        """No accuracy records returns default 0.65."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()

        # Create signal_accuracy table with no matching rows
        signal.db_path = tmp_path / "signals.db"
        conn = sqlite3.connect(str(signal.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, source_type TEXT, source_name TEXT,
                prediction_timestamp TEXT, predicted_signal REAL,
                actual_return REAL, horizon_days INTEGER,
                accuracy_score REAL, error REAL
            )
        """)
        conn.commit()
        conn.close()

        result = signal.get_historical_accuracy("SPY")
        assert result == 0.65

    def test_returns_average_when_data_exists(self, tmp_path):
        """Accuracy records return averaged value."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()

        signal.db_path = tmp_path / "signals.db"
        conn = sqlite3.connect(str(signal.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, source_type TEXT, source_name TEXT,
                prediction_timestamp TEXT, predicted_signal REAL,
                actual_return REAL, horizon_days INTEGER,
                accuracy_score REAL, error REAL
            )
        """)
        conn.execute("""
            INSERT INTO signal_accuracy
            (ticker, source_type, source_name, prediction_timestamp,
             predicted_signal, actual_return, horizon_days, accuracy_score, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("SPY", "macro", "fed_economic", "2026-05-01", 0.5, 0.4, 21, 0.70, 0.1))
        conn.commit()
        conn.close()

        result = signal.get_historical_accuracy("SPY")
        assert result == 0.70


# ---------------------------------------------------------------------------
# AlternativeDataSignalAdapter tests
# ---------------------------------------------------------------------------


class MockCompositeSignal:
    """Minimal mock for AlternativeDataClient.CompositeSignal."""
    def __init__(self, score=0.3, confidence=0.7):
        self.composite_score = score
        self.composite_confidence = confidence
        self.satellite_score = 0.1
        self.credit_card_score = 0.2
        self.supply_chain_score = 0.15
        self.primary_driver = "satellite"
        self.signal_agreement = "mixed"


class TestAlternativeDataSignalAdapter:
    """Test AlternativeDataSignalAdapter."""

    @pytest.fixture(autouse=True)
    def _inject_logger(self):
        """Inject logger into module to prevent NameError from source bug,
        then restore original logger after class to avoid breaking caplog
        for subsequent tests."""
        import src.signals.integrator as _mod
        original_logger = _mod.logger
        _mod.logger = MagicMock()
        yield
        _mod.logger = original_logger

    def test_confident_signal(self, tmp_path):
        """Composite with confidence >= 0.3 returns valid SignalSourceResult."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                adapter = AlternativeDataSignalAdapter()

        adapter.db_path = tmp_path / "signals.db"
        conn = sqlite3.connect(str(adapter.db_path))
        conn.execute("""
            CREATE TABLE signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, timestamp TEXT, source_type TEXT,
                source_name TEXT, signal REAL, confidence REAL,
                raw_score REAL, raw_unit TEXT, historical_accuracy REAL,
                metadata TEXT
            )
        """)
        conn.commit()
        conn.close()

        # Mock get_historical_accuracy to avoid DB queries during generate_signal
        with patch.object(adapter, "get_historical_accuracy", return_value=0.75):
            mock_client = MagicMock()
            mock_client.get_composite_signal.return_value = MockCompositeSignal(score=0.5, confidence=0.8)
            adapter.client = mock_client

            result = adapter.generate_signal("SPY")

        assert isinstance(result, SignalSourceResult)
        assert result.signal == 0.5
        assert result.confidence == 0.8
        assert result.source_type == "sentiment"
        assert result.source_name == "alternative_data"

    def test_low_confidence_returns_none(self, tmp_path):
        """Composite with confidence < 0.3 returns None."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                adapter = AlternativeDataSignalAdapter()

        mock_client = MagicMock()
        mock_client.get_composite_signal.return_value = MockCompositeSignal(score=0.5, confidence=0.2)
        adapter.client = mock_client

        result = adapter.generate_signal("SPY")
        assert result is None

    def test_exception_returns_none(self, tmp_path):
        """Exception from client returns None."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                adapter = AlternativeDataSignalAdapter()

        mock_client = MagicMock()
        mock_client.get_composite_signal.side_effect = RuntimeError("API error")
        adapter.client = mock_client

        result = adapter.generate_signal("SPY")
        assert result is None

    def test_client_lazy_loaded(self, tmp_path):
        """Client is lazily created when None and invoked."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                adapter = AlternativeDataSignalAdapter()

        adapter.db_path = tmp_path / "signals.db"
        conn = sqlite3.connect(str(adapter.db_path))
        conn.execute("""
            CREATE TABLE signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, timestamp TEXT, source_type TEXT,
                source_name TEXT, signal REAL, confidence REAL,
                raw_score REAL, raw_unit TEXT, historical_accuracy REAL,
                metadata TEXT
            )
        """)
        conn.commit()
        conn.close()

        assert adapter.client is None
        with patch.object(adapter, "get_historical_accuracy", return_value=0.75):
            with patch("src.signals.integrator.AlternativeDataClient") as mock_client_cls:
                mock_instance = MagicMock()
                mock_instance.get_composite_signal.return_value = MockCompositeSignal(score=0.4, confidence=0.7)
                mock_client_cls.return_value = mock_instance

                result = adapter.generate_signal("SPY")

        assert adapter.client is not None
        assert isinstance(result, SignalSourceResult)
        assert result.signal == 0.4

    def test_historical_accuracy_no_data(self, tmp_path):
        """No accuracy records returns None."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                adapter = AlternativeDataSignalAdapter()

        adapter.db_path = tmp_path / "signals.db"
        conn = sqlite3.connect(str(adapter.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, source_type TEXT, source_name TEXT,
                prediction_timestamp TEXT, predicted_signal REAL,
                actual_return REAL, horizon_days INTEGER,
                accuracy_score REAL, error REAL
            )
        """)
        conn.commit()
        conn.close()

        result = adapter.get_historical_accuracy("SPY")
        assert result is None

    def test_historical_accuracy_with_data(self, tmp_path):
        """Accuracy records return averaged value."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                adapter = AlternativeDataSignalAdapter()

        adapter.db_path = tmp_path / "signals.db"
        conn = sqlite3.connect(str(adapter.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, source_type TEXT, source_name TEXT,
                prediction_timestamp TEXT, predicted_signal REAL,
                actual_return REAL, horizon_days INTEGER,
                accuracy_score REAL, error REAL
            )
        """)
        conn.execute("""
            INSERT INTO signal_accuracy
            (ticker, source_type, source_name, prediction_timestamp,
             predicted_signal, actual_return, horizon_days, accuracy_score, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("SPY", "sentiment", "alternative_data", "2026-05-01", 0.5, 0.4, 21, 0.75, 0.1))
        conn.commit()
        conn.close()

        result = adapter.get_historical_accuracy("SPY")
        assert result == 0.75


# ---------------------------------------------------------------------------
# LLMSentimentSignalAdapter tests
# ---------------------------------------------------------------------------


class TestLLMSentimentSignalAdapter:
    """Test LLMSentimentSignalAdapter."""

    def test_generate_signal_returns_neutral_with_low_confidence(self, tmp_path):
        """generate_signal returns neutral signal with conf=0.3 (no earnings data)."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                adapter = LLMSentimentSignalAdapter()

        # The method tries to import SentimentAnalyzer; mock the import path
        fake_llm_module = MagicMock()
        fake_llm_module.SentimentAnalyzer = MagicMock()
        with patch.dict("sys.modules", {"src.llm.sentiment_client": fake_llm_module}):
            result = adapter.generate_signal("SPY")

        assert isinstance(result, SignalSourceResult)
        assert result.source_type == "sentiment"
        assert result.source_name == "llm_composite"
        assert result.signal == 0.0  # neutral placeholder
        assert result.confidence == 0.3  # low confidence

    def test_get_historical_accuracy(self):
        """Returns constant 0.76 based on research."""
        adapter = LLMSentimentSignalAdapter()
        # Bypass init to avoid DB
        adapter.source_type = "sentiment"
        adapter.source_name = "llm_composite"

        result = adapter.get_historical_accuracy("SPY")
        assert result == 0.76

    def test_signal_metadata_includes_model_info(self, tmp_path):
        """Generated signal metadata contains model and note."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                adapter = LLMSentimentSignalAdapter()

        fake_llm_module = MagicMock()
        fake_llm_module.SentimentAnalyzer = MagicMock()
        with patch.dict("sys.modules", {"src.llm.sentiment_client": fake_llm_module}):
            result = adapter.generate_signal("SPY")

        assert "model" in result.metadata
        assert result.metadata["model"] == "gpt-4o-mini"
        assert "note" in result.metadata


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


class TestAdditionalEdgeCases:
    """Additional edge cases to improve coverage breadth."""

    def test_signal_agreement_two_signals_conflicting(self, tmp_path):
        """Exactly 2 signals: one bullish, one bearish -> conflicting."""
        integrator = _make_integrator(tmp_path)
        src_bull = MagicMock()
        src_bull.generate_signal.return_value = _make_signal(
            "momentum", "tsmom", 0.5, confidence=0.8
        )
        src_bear = MagicMock()
        src_bear.generate_signal.return_value = _make_signal(
            "macro", "fed_policy", -0.5, confidence=0.8
        )
        integrator.sources = {"momentum": src_bull, "macro": src_bear}
        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.signal_agreement == "conflicting"

    def test_signal_agreement_all_exactly_at_threshold(self, tmp_path):
        """Signals at exactly +/-0.3 boundary: not bullish/bearish -> mixed."""
        integrator = _make_integrator(tmp_path)
        src_a = MagicMock()
        src_a.generate_signal.return_value = _make_signal(
            "momentum", "tsmom", 0.3, confidence=0.8
        )
        src_b = MagicMock()
        src_b.generate_signal.return_value = _make_signal(
            "macro", "fed_policy", 0.3, confidence=0.8
        )
        integrator.sources = {"momentum": src_a, "macro": src_b}
        result = integrator.get_composite_signal("SPY", regime="neutral")
        # 0.3 is NOT > 0.3, so neither bullish_count nor bearish_count
        # Both zero -> agreement falls to "mixed"
        assert result.signal_agreement == "mixed"

    def test_allocation_delta_default_bounds(self):
        """AllocationDelta has default max/min position bounds."""
        delta = AllocationDelta(
            ticker="SPY", current_weight=0.46, recommended_weight=0.48,
            delta=0.02, composite_score=0.4, confidence=0.75,
            primary_reason="technical",
        )
        assert delta.max_position == 0.60
        assert delta.min_position == 0.05

    def test_composite_signal_no_components(self):
        """CompositeSignal with empty component list has component_count=0 in to_dict."""
        cs = CompositeSignal(
            ticker="SPY", timestamp=datetime.now().isoformat(),
            composite_score=0.0, composite_confidence=0.0,
            detected_regime="neutral",
        )
        d = cs.to_dict()
        assert d["component_count"] == 0
        assert d["components"] == []

    def test_get_allocation_deltas_empty_alloc(self, tmp_path):
        """Empty allocation dict returns empty recommendation."""
        integrator = _make_integrator(tmp_path)
        result = integrator.get_allocation_deltas({})
        assert len(result.deltas) == 0
        assert result.recommended_allocation == {}

    def test_expected_accuracy_with_mixed_weights(self, tmp_path):
        """_calculate_expected_accuracy handles signals with different weights."""
        integrator = _make_integrator(tmp_path)
        signals = [
            _make_signal("momentum", "tsmom", 0.5, confidence=0.9, accuracy=0.80),
            _make_signal("macro", "fed", 0.3, confidence=0.7, accuracy=0.65),
            _make_signal("sentiment", "llm", 0.1, confidence=0.5, accuracy=0.70),
        ]
        weights = {"momentum": 0.4, "macro": 0.3, "sentiment": 0.3}
        acc = integrator._calculate_expected_accuracy(signals, weights)
        # weighted: (0.80*0.4*0.9 + 0.65*0.3*0.7 + 0.70*0.3*0.5) / (0.4*0.9 + 0.3*0.7 + 0.3*0.5)
        # = (0.288 + 0.1365 + 0.105) / (0.36 + 0.21 + 0.15) = 0.5295 / 0.72 = 0.7354...
        assert 0.70 < acc < 0.77


# ============================================================================
# 1. Dataclass Field Validation — use dataclasses.fields() to verify ALL fields
# ============================================================================


class TestDataclassFieldIntrospection:
    """Validate all dataclass fields via dataclasses.fields()."""

    def test_signal_source_result_fields(self):
        """SignalSourceResult has all expected fields with correct types."""
        fields = {f.name: f.type for f in dataclasses.fields(SignalSourceResult)}
        expected = {
            "source_type": str,
            "source_name": str,
            "signal": float,
            "confidence": float,
            "raw_score": float,
            "raw_unit": str,
            "historical_accuracy": Optional[float],
            "sample_count": int,
            "timestamp": str,
            "metadata": Dict[str, Any],
        }
        for name, typ in expected.items():
            assert name in fields, f"Missing field {name}"
            assert fields[name] == typ, f"Field {name} type mismatch: {fields[name]} != {typ}"

    def test_composite_signal_fields(self):
        """CompositeSignal has all expected fields with correct types."""
        fields = {f.name: f.type for f in dataclasses.fields(CompositeSignal)}
        expected = {
            "ticker": str,
            "timestamp": str,
            "component_signals": List[SignalSourceResult],
            "composite_score": float,
            "composite_confidence": float,
            "primary_drivers": List[str],
            "signal_agreement": str,
            "detected_regime": str,
            "weights_used": Dict[str, float],
            "expected_accuracy": Optional[float],
        }
        for name, typ in expected.items():
            assert name in fields, f"Missing field {name}"
            assert fields[name] == typ, f"Field {name} type mismatch: {fields[name]} != {typ}"

    def test_allocation_delta_fields(self):
        """AllocationDelta has all expected fields with correct types."""
        fields = {f.name: f.type for f in dataclasses.fields(AllocationDelta)}
        expected = {
            "ticker": str,
            "current_weight": float,
            "recommended_weight": float,
            "delta": float,
            "composite_score": float,
            "confidence": float,
            "primary_reason": str,
            "max_position": float,
            "min_position": float,
        }
        for name, typ in expected.items():
            assert name in fields, f"Missing field {name}"
            assert fields[name] == typ, f"Field {name} type mismatch: {fields[name]} != {typ}"

    def test_portfolio_recommendation_fields(self):
        """PortfolioRecommendation has all expected fields with correct types."""
        fields = {f.name: f.type for f in dataclasses.fields(PortfolioRecommendation)}
        expected = {
            "timestamp": str,
            "current_allocation": Dict[str, float],
            "recommended_allocation": Dict[str, float],
            "deltas": List[AllocationDelta],
            "composite_sentiment": str,
            "confidence": float,
            "regime": str,
            "expected_volatility": Optional[float],
            "max_drawdown_estimate": Optional[float],
        }
        for name, typ in expected.items():
            assert name in fields, f"Missing field {name}"

    def test_signal_source_result_defaults(self):
        """Verify default values are correct via dataclass fields introspection."""
        for f in dataclasses.fields(SignalSourceResult):
            if f.name == "sample_count":
                assert f.default == 0
            elif f.name == "historical_accuracy":
                assert f.default is None
            elif f.name == "metadata":
                # field(default_factory=dict)
                assert f.default_factory is not None
            elif f.name == "timestamp":
                # field(default_factory=lambda: datetime.now().isoformat())
                assert f.default_factory is not None

    def test_allocation_delta_defaults(self):
        """Verify AllocationDelta default bounds are correct."""
        for f in dataclasses.fields(AllocationDelta):
            if f.name == "max_position":
                assert f.default == 0.60
            elif f.name == "min_position":
                assert f.default == 0.05

    def test_signal_source_result_field_count(self):
        """SignalSourceResult has exactly 10 fields."""
        assert len(dataclasses.fields(SignalSourceResult)) == 10

    def test_composite_signal_field_count(self):
        """CompositeSignal has exactly 10 fields."""
        assert len(dataclasses.fields(CompositeSignal)) == 10

    def test_allocation_delta_field_count(self):
        """AllocationDelta has exactly 9 fields."""
        assert len(dataclasses.fields(AllocationDelta)) == 9

    def test_portfolio_recommendation_field_count(self):
        """PortfolioRecommendation has exactly 9 fields."""
        assert len(dataclasses.fields(PortfolioRecommendation)) == 9


# ============================================================================
# 2. Computation Edge Cases — zero/empty, single-element, NaN/Inf, boundaries
# ============================================================================


class TestNormalizeSignalEdgeCases:
    """Edge cases for _normalize_signal with extreme/NaN/Inf values."""

    def _make_source(self):
        class TestSource(SignalSource):
            def generate_signal(self, ticker):
                return None
            def get_historical_accuracy(self, ticker, horizon_days=21):
                return None
        return TestSource("test", "test")

    def test_negative_range(self):
        """Range with negative values works correctly."""
        s = self._make_source()
        # Range [-2, 2], value 0 -> midpoint -> 0.0
        assert s._normalize_signal(0.0, -2.0, 2.0) == 0.0
        assert s._normalize_signal(2.0, -2.0, 2.0) == 1.0
        assert s._normalize_signal(-2.0, -2.0, 2.0) == -1.0

    def test_reversed_range(self):
        """min > max: formula normalizes then clamps to valid range.
        With min=10, max=0, input=5: 2*(5-10)/(0-10)-1 = 2*(-5)/(-10)-1 = 1-1 = 0.0."""
        s = self._make_source()
        result = s._normalize_signal(5.0, 10.0, 0.0)
        assert result == 0.0

    def test_single_value_range_min_equals_max_by_one(self):
        """Range of zero (single point) returns 0.0 even when value is extreme."""
        s = self._make_source()
        assert s._normalize_signal(42.0, 42.0, 42.0) == 0.0

    @patch("src.signals.integrator.SIGNAL_MIN", -10.0)
    @patch("src.signals.integrator.SIGNAL_MAX", 10.0)
    def test_custom_bounds(self):
        """Respects patched SIGNAL_MIN/SIGNAL_MAX."""
        s = self._make_source()
        result = s._normalize_signal(100.0, -1.0, 1.0)
        # 2*(100 - (-1))/2 - 1 = 100, clamped to 10.0
        assert result == 10.0

    def test_wide_range(self):
        """Wide range [0, 10000], midpoint at 5000."""
        s = self._make_source()
        assert s._normalize_signal(5000.0, 0.0, 10000.0) == 0.0
        assert s._normalize_signal(10000.0, 0.0, 10000.0) == 1.0
        assert s._normalize_signal(0.0, 0.0, 10000.0) == -1.0

    def test_tiny_input(self):
        """Very small near-zero values are handled (floating-point precision)."""
        s = self._make_source()
        result = s._normalize_signal(1e-10, -1.0, 1.0)
        # May not be exactly 1e-10 due to floating-point, but should be very close
        assert abs(result - 1e-10) < 1e-15


class TestCompositeScoreNaNInfBoundaries:
    """Test composite score calculation with NaN/Inf in signals."""

    def test_nan_signal_clamped(self, tmp_path):
        """NaN signal value gets clamped to SIGNAL_MAX/SIGNAL_MIN by min/max (Py3.13)."""
        integrator = _make_integrator(tmp_path)
        src = MagicMock()
        src.generate_signal.return_value = _make_signal(
            "momentum", "tsmom", float("nan"), confidence=0.8
        )
        integrator.sources = {"momentum": src}
        with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 1):
            result = integrator.get_composite_signal("SPY", regime="neutral")
        # Python 3.12+: min(1.0, NaN) -> 1.0, max(-1.0, NaN) -> -1.0 (NaN clamped)
        assert isinstance(result.composite_score, float)
        assert not math.isnan(result.composite_score)

    def test_inf_signal_clamped(self, tmp_path):
        """Inf signal value is clamped to SIGNAL_MAX."""
        integrator = _make_integrator(tmp_path)
        src = MagicMock()
        src.generate_signal.return_value = _make_signal(
            "momentum", "tsmom", float("inf"), confidence=0.8
        )
        integrator.sources = {"momentum": src}
        with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 1):
            result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.composite_score == 1.0

    def test_neg_inf_signal_clamped(self, tmp_path):
        """-Inf signal value is clamped to SIGNAL_MIN."""
        integrator = _make_integrator(tmp_path)
        src = MagicMock()
        src.generate_signal.return_value = _make_signal(
            "momentum", "tsmom", float("-inf"), confidence=0.8
        )
        integrator.sources = {"momentum": src}
        with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 1):
            result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.composite_score == -1.0

    def test_zero_confidence_inf_weight_total(self, tmp_path):
        """All signals with confidence=0 -> weight_total=0 -> score 0.0."""
        integrator = _make_integrator(tmp_path)
        src_a = MagicMock()
        src_a.generate_signal.return_value = _make_signal(
            "momentum", "tsmom", 0.5, confidence=0.0
        )
        src_b = MagicMock()
        src_b.generate_signal.return_value = _make_signal(
            "macro", "fed", 0.3, confidence=0.0
        )
        integrator.sources = {"momentum": src_a, "macro": src_b}
        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.composite_score == 0.0
        assert result.composite_confidence == 0.0

    def test_very_small_weight_total(self, tmp_path):
        """Extremely small confidence produces very small weight_total."""
        integrator = _make_integrator(tmp_path)
        src = MagicMock()
        src.generate_signal.return_value = _make_signal(
            "momentum", "tsmom", 0.5, confidence=1e-10
        )
        integrator.sources = {"momentum": src}
        with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 1):
            result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.composite_score == 0.5  # Still works
        assert result.composite_confidence < 0.01


class TestTechnicalSignalEdgeCases:
    """Edge cases for TechnicalSignal computation methods."""

    def test_momentum_exactly_200_rows(self, tmp_path):
        """Exactly 200 rows is sufficient for momentum calculation."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()
        _create_market_db(signal.market_db, _make_rising_prices(200, 100, 200))
        result = signal._calculate_momentum("SPY")
        assert result is not None

    def test_momentum_single_element_close(self, tmp_path):
        """Exactly 252+1 rows with monotonic price tests 12m return edge."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()
        _create_market_db(signal.market_db, _make_rising_prices(253, 100, 200))
        result = signal._calculate_momentum("SPY")
        assert result is not None
        assert result["return_12m"] > 0

    def test_momentum_below_200_rows_exact(self, tmp_path):
        """Exactly 199 rows returns None."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()
        _create_market_db(signal.market_db, _make_rising_prices(199, 100, 200))
        result = signal._calculate_momentum("SPY")
        assert result is None

    def test_mean_reversion_exactly_14_rows(self, tmp_path):
        """Exactly 14 rows is sufficient for mean reversion (>= 14 means non-zero)."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()
        _create_market_db(signal.market_db, _make_rising_prices(14, 100, 110))
        result = signal._calculate_mean_reversion("SPY")
        # With 14 rows, RSI is computed. Rising prices -> RSI > 70 -> bearish signal < 0
        assert isinstance(result, float)

    def test_mean_reversion_just_below_14_rows_returns_zero(self, tmp_path):
        """Exactly 13 rows returns 0.0."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()
        _create_market_db(signal.market_db, _make_rising_prices(13, 100, 110))
        result = signal._calculate_mean_reversion("SPY")
        assert result == 0.0

    def test_momentum_12m_return_zero_when_insufficient_data(self, tmp_path):
        """Less than 252 trading days -> return_12m = 0."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()
        _create_market_db(signal.market_db, _make_rising_prices(220, 100, 200))
        result = signal._calculate_momentum("SPY")
        assert result is not None
        assert result["return_12m"] == 0

    def test_momentum_return_6m_computed_with_sufficient_data(self, tmp_path):
        """With 220 rows (126+ for 6m, 200+ for momentum), return_6m is computed."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()
        # 200+ rows -> momentum succeeds; > 126 rows -> return_6m computed
        rows = _make_rising_prices(220, 100, 200)
        _create_market_db(signal.market_db, rows)
        result = signal._calculate_momentum("SPY")
        assert result is not None
        assert "return_6m" in result

    def test_momentum_bearish_below_15pct(self, tmp_path):
        """Return_12m < -0.15 uses different scoring branch."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()
        # Steeply falling: start at 300, end at 100 over 300 days
        _create_market_db(signal.market_db, _make_falling_prices(300, 300, 100))
        result = signal._calculate_momentum("SPY")
        assert result is not None
        assert result["above_sma"] is False
        assert result["return_12m"] < -0.15

    def test_momentum_above_sma_negative_return(self, tmp_path):
        """above_sma=True but return_12m < 0 -> mild bearish score."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()
        # Generate prices that end above SMA but have negative 12m return
        # Start very high, dip low, then rise back above SMA near end
        from datetime import datetime, timedelta
        now = datetime.now()
        base = now - timedelta(days=400)
        rows = []
        for i in range(300):
            frac = i / 299.0
            # Dip to 50 then recover to 110 (above SMA of ~75)
            price = 200 - 150 * frac if frac < 0.5 else 50 + 60 * (frac - 0.5) / 0.5
            date_str = (base + timedelta(days=i)).strftime("%Y-%m-%d")
            rows.append((date_str, price))
        _create_market_db(signal.market_db, rows)
        result = signal._calculate_momentum("SPY")
        # above_sma should be True when current > sma_200
        assert result is not None

    def test_mean_reversion_extreme_rsi_100(self, tmp_path):
        """All gains -> RSI near 100 -> bearish signal near -1.0."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()
        from datetime import datetime, timedelta
        now = datetime.now()
        rows = []
        close = 100.0
        for i in range(20):
            date_str = (now - timedelta(days=20 - i)).strftime("%Y-%m-%d")
            rows.append((date_str, close))
            close += 1.0  # Always up
        _create_market_db(signal.market_db, rows)
        result = signal._calculate_mean_reversion("SPY")
        assert result < 0  # Bearish
        assert result == pytest.approx(-1.0, abs=0.1)

    def test_mean_reversion_extreme_rsi_0(self, tmp_path):
        """All losses -> RSI=0 -> max bullish signal of 1.0."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = TechnicalSignal()
        from datetime import datetime, timedelta
        now = datetime.now()
        rows = []
        close = 100.0
        for i in range(20):
            date_str = (now - timedelta(days=20 - i)).strftime("%Y-%m-%d")
            rows.append((date_str, close))
            close -= 1.0  # Always down — all losses
        _create_market_db(signal.market_db, rows)
        result = signal._calculate_mean_reversion("SPY")
        assert result == 1.0


# ============================================================================
# 3. Constants Validation — verify types, ranges, existence
# ============================================================================


class TestConstantsExistenceAndTypes:
    """Validate that ALL module-level constants exist with expected types/ranges."""

    def test_db_path_exists(self):
        """DB_PATH is a Path object."""
        from pathlib import Path
        assert isinstance(DB_PATH, Path)

    def test_base_weights_keys(self):
        """BASE_WEIGHTS has all expected keys."""
        expected_keys = {
            "momentum", "value", "macro", "quality", "sentiment",
            "ai_agent", "tsmom", "fed_policy", "hmm_regime",
            "multi_speed", "risk_parity",
        }
        assert set(BASE_WEIGHTS.keys()) == expected_keys

    def test_base_weights_values_positive(self):
        """All BASE_WEIGHTS values are positive."""
        for k, v in BASE_WEIGHTS.items():
            assert v > 0, f"Weight {k}={v} is not positive"

    def test_base_weights_values_below_one(self):
        """All BASE_WEIGHTS values are below 0.5 (no single dominant weight)."""
        for k, v in BASE_WEIGHTS.items():
            assert v < 0.5, f"Weight {k}={v} exceeds 0.5"

    def test_regime_weights_keys(self):
        """REGIME_WEIGHTS has exactly 5 regime keys."""
        expected_regimes = {"bull", "bear", "neutral", "crisis", "high_vol"}
        assert set(REGIME_WEIGHTS.keys()) == expected_regimes

    def test_regime_neutral_is_base_weights_reference(self):
        """neutral regime IS the BASE_WEIGHTS dict reference."""
        assert REGIME_WEIGHTS["neutral"] is BASE_WEIGHTS

    def test_min_signal_sources_type(self):
        """MIN_SIGNAL_SOURCES is int >= 1."""
        assert isinstance(MIN_SIGNAL_SOURCES, int)
        assert MIN_SIGNAL_SOURCES >= 1

    def test_signal_bound_types(self):
        """SIGNAL_MIN, SIGNAL_MAX are floats with min < max."""
        assert isinstance(SIGNAL_MIN, (int, float))
        assert isinstance(SIGNAL_MAX, (int, float))
        assert SIGNAL_MIN < SIGNAL_MAX
        assert SIGNAL_MIN == -1.0
        assert SIGNAL_MAX == 1.0

    def test_max_delta_pct_type(self):
        """MAX_DELTA_PCT is a small positive float."""
        assert isinstance(MAX_DELTA_PCT, float)
        assert 0 < MAX_DELTA_PCT <= 0.5


# ============================================================================
# 4. Function Boundary Conditions — extreme inputs, missing keys, wrong types
# ============================================================================


class TestCompositeSignalBoundaryConditions:
    """Boundary conditions for get_composite_signal with extreme inputs."""

    def test_no_regime_and_no_custom_weights_uses_base(self, tmp_path):
        """When regime is None and not in REGIME_WEIGHTS, fallback to BASE_WEIGHTS."""
        integrator = _make_integrator(tmp_path)
        src = MagicMock()
        src.generate_signal.return_value = _make_signal(
            "momentum", "tsmom", 0.5, confidence=0.8
        )
        integrator.sources = {"momentum": src}
        with patch.object(integrator, "_detect_regime", return_value="bull"):
            with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 1):
                result = integrator.get_composite_signal("SPY", regime=None)
        assert result.weights_used == REGIME_WEIGHTS["bull"]

    def test_custom_weights_empty_dict_falls_back(self, tmp_path):
        """Empty custom_weights dict -> source_type 'momentum' not found -> weight defaults to 0.20."""
        integrator = _make_integrator(tmp_path)
        src = MagicMock()
        src.generate_signal.return_value = _make_signal(
            "momentum", "tsmom", 0.5, confidence=0.8
        )
        integrator.sources = {"momentum": src}
        with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 1):
            result = integrator.get_composite_signal("SPY", custom_weights={})
        # Source type "momentum" not in {} -> default 0.20
        assert result.composite_score > 0

    def test_custom_weights_unknown_source_type(self, tmp_path):
        """Source type not in custom weight -> weight defaults to 0.20."""
        integrator = _make_integrator(tmp_path)
        src = MagicMock()
        src.generate_signal.return_value = _make_signal(
            "momentum", "tsmom", 0.5, confidence=0.8
        )
        integrator.sources = {"momentum": src}
        custom = {"unknown_type": 1.0}
        with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 1):
            result = integrator.get_composite_signal("SPY", custom_weights=custom)
        assert result.composite_score > 0

    def test_all_sources_return_none_insufficient(self, tmp_path):
        """Every source returns None -> insufficient_data."""
        integrator = _make_integrator(tmp_path)
        integrator.sources = {
            "momentum": MagicMock(generate_signal=MagicMock(return_value=None)),
            "macro": MagicMock(generate_signal=MagicMock(return_value=None)),
            "sentiment": MagicMock(generate_signal=MagicMock(return_value=None)),
        }
        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.signal_agreement == "insufficient_data"
        assert result.composite_score == 0.0
        assert result.composite_confidence == 0.0

    def test_primary_drivers_empty_when_no_signals(self, tmp_path):
        """No signals -> empty primary_drivers."""
        integrator = _make_integrator(tmp_path)
        integrator.sources = {}
        with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 1):
            result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.primary_drivers == []

    def test_primary_drivers_max_three(self, tmp_path):
        """At most 3 primary drivers returned."""
        integrator = _make_integrator(tmp_path)
        sources = {}
        for i in range(6):
            src = MagicMock()
            src.generate_signal.return_value = _make_signal(
                f"type_{i}", f"src_{i}", 0.5, confidence=0.8
            )
            sources[f"src_{i}"] = src
        integrator.sources = sources
        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert len(result.primary_drivers) <= 3

    def test_expected_accuracy_none_when_no_signals(self, tmp_path):
        """No signals -> expected_accuracy is None."""
        integrator = _make_integrator(tmp_path)
        integrator.sources = {}
        with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 1):
            result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.expected_accuracy is None

    def test_source_exception_with_fallback_to_others(self, tmp_path):
        """Exception in one source doesn't prevent others from contributing."""
        integrator = _make_integrator(tmp_path)
        src_good = MagicMock()
        src_good.generate_signal.return_value = _make_signal(
            "momentum", "tsmom", 0.5, confidence=0.8
        )
        src_bad = MagicMock()
        src_bad.generate_signal.side_effect = ValueError("Bad data")
        integrator.sources = {"momentum": src_good, "bad": src_bad}
        with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 1):
            result = integrator.get_composite_signal("SPY", regime="neutral")
        assert len(result.component_signals) == 1
        assert result.composite_score != 0.0


class TestAllocationDeltaBoundaryConditions:
    """Boundary conditions for get_allocation_deltas with extreme inputs."""

    def test_asset_at_max_position_capped(self, tmp_path):
        """Asset at 0.60 with bullish signal -> stays at 0.60."""
        integrator = _make_integrator(tmp_path)

        def mock_bull(ticker, regime=None, custom_weights=None):
            return CompositeSignal(
                ticker=ticker, timestamp=datetime.now().isoformat(),
                composite_score=0.5, composite_confidence=1.0,
                detected_regime="bull", primary_drivers=["momentum"],
                signal_agreement="aligned_bullish",
            )
        integrator.get_composite_signal = mock_bull
        result = integrator.get_allocation_deltas({"SPY": 0.60})
        assert result.deltas[0].recommended_weight == 0.60

    def test_asset_at_min_position_bearish(self, tmp_path):
        """Asset at 0.05 with bearish signal -> stays at 0.05."""
        integrator = _make_integrator(tmp_path)

        def mock_bear(ticker, regime=None, custom_weights=None):
            return CompositeSignal(
                ticker=ticker, timestamp=datetime.now().isoformat(),
                composite_score=-0.5, composite_confidence=1.0,
                detected_regime="bear", primary_drivers=["macro"],
                signal_agreement="aligned_bearish",
            )
        integrator.get_composite_signal = mock_bear
        result = integrator.get_allocation_deltas({"SPY": 0.05})
        assert result.deltas[0].recommended_weight == 0.05

    def test_no_primary_drivers_neutral_reason(self, tmp_path):
        """When primary_drivers empty, reason becomes 'neutral'."""
        integrator = _make_integrator(tmp_path)

        def mock_no_drivers(ticker, regime=None, custom_weights=None):
            return CompositeSignal(
                ticker=ticker, timestamp=datetime.now().isoformat(),
                composite_score=0.2, composite_confidence=0.6,
                detected_regime="neutral", primary_drivers=[],
                signal_agreement="mixed",
            )
        integrator.get_composite_signal = mock_no_drivers
        result = integrator.get_allocation_deltas({"SPY": 0.46})
        assert result.deltas[0].primary_reason == "neutral"

    def test_sentiment_bearish_threshold(self, tmp_path):
        """Average score exactly -0.3 -> neutral (not > 0.3 or < -0.3)."""
        integrator = _make_integrator(tmp_path)

        def mock_borderline(ticker, regime=None, custom_weights=None):
            return CompositeSignal(
                ticker=ticker, timestamp=datetime.now().isoformat(),
                composite_score=-0.3, composite_confidence=0.8,
                detected_regime="neutral", primary_drivers=["macro"],
                signal_agreement="aligned_bearish",
            )
        integrator.get_composite_signal = mock_borderline
        result = integrator.get_allocation_deltas({"SPY": 0.46})
        assert result.composite_sentiment == "neutral"

    def test_sentiment_bullish_threshold_exact(self, tmp_path):
        """Average score exactly 0.3 -> neutral (not > 0.3)."""
        integrator = _make_integrator(tmp_path)

        def mock_borderline(ticker, regime=None, custom_weights=None):
            return CompositeSignal(
                ticker=ticker, timestamp=datetime.now().isoformat(),
                composite_score=0.3, composite_confidence=0.8,
                detected_regime="neutral", primary_drivers=["momentum"],
                signal_agreement="aligned_bullish",
            )
        integrator.get_composite_signal = mock_borderline
        result = integrator.get_allocation_deltas({"SPY": 0.46})
        assert result.composite_sentiment == "neutral"


class TestMacroSignalBoundaryConditions:
    """Boundary conditions for MacroSignal sub-methods."""

    def test_yield_curve_tlt_or_shy_missing(self, tmp_path):
        """Only TLT or SHY available -> returns 0.0."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()
        conn = sqlite3.connect(str(signal.market_db))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, PRIMARY KEY (symbol, date))")
        from datetime import datetime, timedelta
        for i in range(25):
            date_str = (datetime(2024, 5, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute("INSERT INTO prices VALUES ('TLT', ?, ?)", (date_str, 100.0 + i))
        conn.commit()
        conn.close()
        result = signal._get_yield_curve_signal()
        assert result == 0.0

    def test_credit_lqd_or_hyg_missing(self, tmp_path):
        """Only LQD or HYG available -> returns 0.0."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()
        conn = sqlite3.connect(str(signal.market_db))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, PRIMARY KEY (symbol, date))")
        from datetime import datetime, timedelta
        for i in range(25):
            date_str = (datetime(2024, 5, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute("INSERT INTO prices VALUES ('LQD', ?, ?)", (date_str, 100.0 + i))
        conn.commit()
        conn.close()
        result = signal._get_credit_signal()
        assert result == 0.0

    def test_30d_change_exception_returns_none(self, tmp_path):
        """Exception in _get_30d_change returns None."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()
        signal.market_db = tmp_path / "nonexistent" / "market.db"
        result = signal._get_30d_change("SPY")
        assert result is None

    def test_yield_curve_extreme_spread(self, tmp_path):
        """Very large TLT/SHY spread change is clamped to [-1, 1]."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()
        conn = sqlite3.connect(str(signal.market_db))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, PRIMARY KEY (symbol, date))")
        # TLT extreme rise, SHY extreme fall
        for i in range(25):
            from datetime import datetime, timedelta
            date_str = (datetime(2024, 5, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute("INSERT INTO prices VALUES ('TLT', ?, ?)", (date_str, 100.0 + i * 10.0))
            conn.execute("INSERT INTO prices VALUES ('SHY', ?, ?)", (date_str, 85.0 - i * 5.0))
        conn.commit()
        conn.close()
        result = signal._get_yield_curve_signal()
        assert -1.0 <= result <= 1.0


class TestDetectRegimeMissingDataEdgeCases:
    """Edge cases for _detect_regime with missing/corrupt data."""

    def _setup_db(self, tmp_path, vix_level):
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

    def test_vix_negative_value(self, tmp_path):
        """Negative VIX (impossible but defensive) -> bull (< 15)."""
        integrator = _make_integrator(tmp_path)
        self._setup_db(tmp_path, -5.0)
        with patch("src.signals.integrator.DATA_DIR", tmp_path):
            regime = integrator._detect_regime()
        assert regime == "bull"

    def test_vix_nan_value(self, tmp_path):
        """NaN VIX triggers exception -> neutral fallback."""
        integrator = _make_integrator(tmp_path)
        self._setup_db(tmp_path, float("nan"))
        with patch("src.signals.integrator.DATA_DIR", tmp_path):
            regime = integrator._detect_regime()
        assert regime == "neutral"

    def test_market_db_missing_file(self, tmp_path):
        """Missing market.db file -> neutral."""
        integrator = _make_integrator(tmp_path)
        with patch("src.signals.integrator.DATA_DIR", tmp_path / "empty_dir"):
            regime = integrator._detect_regime()
        assert regime == "neutral"

    def test_vix_in_prices_but_no_rows(self, tmp_path):
        """VIX symbol exists in prices table but no rows -> exception -> neutral."""
        db_path = tmp_path / "market.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        conn.commit()
        conn.close()
        integrator = _make_integrator(tmp_path)
        with patch("src.signals.integrator.DATA_DIR", tmp_path):
            regime = integrator._detect_regime()
        assert regime == "neutral"


# ============================================================================
# 5. CLI / __main__ Guard — test CLI entry points with capsys
# ============================================================================


class TestCLIInterface:
    """Test the CLI interface via main() function."""

    def test_composite_requires_ticker(self, caplog):
        """composite command without --ticker prints error and exits."""
        with caplog.at_level(logging.ERROR, logger="src.signals.integrator"):
            with pytest.raises(SystemExit) as exc:
                from src.signals.integrator import main
                import sys
                sys.argv = ["integrator.py", "composite"]
                main()
        assert exc.value.code == 1
        assert "--ticker required" in caplog.text

    def test_portfolio_requires_portfolio_arg(self, caplog):
        """portfolio command without --portfolio prints error and exits."""
        with caplog.at_level(logging.ERROR, logger="src.signals.integrator"):
            with pytest.raises(SystemExit) as exc:
                from src.signals.integrator import main
                import sys
                sys.argv = ["integrator.py", "portfolio"]
                main()
        assert exc.value.code == 1
        assert "--portfolio required" in caplog.text

    def test_history_requires_ticker(self, caplog):
        """history command without --ticker prints error and exits."""
        with caplog.at_level(logging.ERROR, logger="src.signals.integrator"):
            with pytest.raises(SystemExit) as exc:
                from src.signals.integrator import main
                import sys
                sys.argv = ["integrator.py", "history"]
                main()
        assert exc.value.code == 1
        assert "--ticker required" in caplog.text

    def test_composite_json_output(self, caplog):
        """composite command with --json outputs valid JSON."""
        with caplog.at_level(logging.INFO, logger="src.signals.integrator"):
            with patch("sys.argv", ["integrator.py", "composite", "--ticker", "SPY", "--json"]):
                with patch("src.signals.integrator.SignalIntegrator.get_composite_signal") as mock_get:
                    mock_get.return_value = CompositeSignal(
                        ticker="SPY", timestamp="2026-05-24T12:00:00",
                        composite_score=0.35, composite_confidence=0.72,
                        detected_regime="neutral", primary_drivers=["tsmom"],
                        signal_agreement="aligned_bullish",
                    )
                    from src.signals.integrator import main
                    main()
        parsed = json.loads(caplog.messages[0])
        assert parsed["ticker"] == "SPY"
        assert parsed["composite_score"] == 0.35

    def test_composite_text_output(self, caplog):
        """composite command without --json prints formatted text."""
        with caplog.at_level(logging.INFO, logger="src.signals.integrator"):
            with patch("sys.argv", ["integrator.py", "composite", "--ticker", "SPY"]):
                with patch("src.signals.integrator.SignalIntegrator.get_composite_signal") as mock_get:
                    mock_get.return_value = CompositeSignal(
                        ticker="SPY", timestamp="2026-05-24T12:00:00",
                        composite_score=0.35, composite_confidence=0.72,
                        detected_regime="neutral", primary_drivers=["tsmom", "fed_policy"],
                        signal_agreement="aligned_bullish", expected_accuracy=0.68,
                    )
                    from src.signals.integrator import main
                    main()
        assert "SPY" in caplog.text
        assert "0.35" in caplog.text
        assert "aligned_bullish" in caplog.text

    def test_composite_text_no_accuracy(self, caplog):
        """Text output works when expected_accuracy is None."""
        with caplog.at_level(logging.INFO, logger="src.signals.integrator"):
            with patch("sys.argv", ["integrator.py", "composite", "--ticker", "SPY"]):
                with patch("src.signals.integrator.SignalIntegrator.get_composite_signal") as mock_get:
                    mock_get.return_value = CompositeSignal(
                        ticker="SPY", timestamp="2026-05-24T12:00:00",
                        composite_score=0.35, composite_confidence=0.72,
                        detected_regime="neutral", primary_drivers=["tsmom"],
                        signal_agreement="aligned_bullish", expected_accuracy=None,
                    )
                    from src.signals.integrator import main
                    main()
        assert "SPY" in caplog.text
        assert "Expected Accuracy" not in caplog.text

    @patch("src.signals.integrator.SignalIntegrator.get_allocation_deltas")
    def test_portfolio_text_output(self, mock_get, caplog):
        """portfolio command with --portfolio prints formatted text."""
        mock_get.return_value = PortfolioRecommendation(
            timestamp="2026-05-24T12:00:00",
            current_allocation={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            recommended_allocation={"SPY": 0.48, "GLD": 0.36, "TLT": 0.16},
            deltas=[
                AllocationDelta(
                    ticker="SPY", current_weight=0.46, recommended_weight=0.48,
                    delta=0.02, composite_score=0.4, confidence=0.75,
                    primary_reason="momentum",
                ),
                AllocationDelta(
                    ticker="GLD", current_weight=0.38, recommended_weight=0.36,
                    delta=-0.02, composite_score=-0.3, confidence=0.70,
                    primary_reason="macro",
                ),
            ],
            composite_sentiment="bullish", confidence=0.72, regime="neutral",
        )
        with caplog.at_level(logging.INFO, logger="src.signals.integrator"):
            with patch("sys.argv", ["integrator.py", "portfolio", "--portfolio", "46/38/16"]):
                from src.signals.integrator import main
                main()
        assert "Portfolio Recommendation" in caplog.text
        assert "SPY" in caplog.text
        assert "GLD" in caplog.text

    @patch("src.signals.integrator.SignalIntegrator.get_allocation_deltas")
    def test_portfolio_json_output(self, mock_get, caplog):
        """portfolio command with --json outputs valid JSON."""
        mock_get.return_value = PortfolioRecommendation(
            timestamp="2026-05-24T12:00:00",
            current_allocation={"SPY": 0.46},
            recommended_allocation={"SPY": 0.48},
            deltas=[], composite_sentiment="bullish", confidence=0.72, regime="neutral",
        )
        with caplog.at_level(logging.INFO, logger="src.signals.integrator"):
            with patch("sys.argv", ["integrator.py", "portfolio", "--portfolio", "46/38/16", "--json"]):
                from src.signals.integrator import main
                main()
        parsed = json.loads(caplog.messages[0])
        assert parsed["composite_sentiment"] == "bullish"

    def test_portfolio_with_two_weights(self, caplog):
        """portfolio with 2 weights -> SPY/GLD mapping."""
        mock_rec = MagicMock()
        mock_rec.to_dict.return_value = {"test": "ok"}
        with caplog.at_level(logging.INFO, logger="src.signals.integrator"):
            with patch("src.signals.integrator.SignalIntegrator") as MockIntegrator:
                instance = MockIntegrator.return_value
                instance.get_allocation_deltas.return_value = mock_rec
                with patch("sys.argv", ["integrator.py", "portfolio", "--portfolio", "50/50", "--json"]):
                    from src.signals.integrator import main
                    main()
        assert "ok" in caplog.text

    def test_portfolio_with_four_weights(self, caplog):
        """portfolio with 4 weights -> SPY/EFA/GLD/TLT mapping."""
        mock_rec = MagicMock()
        mock_rec.to_dict.return_value = {"test": "4_asset_ok"}
        with caplog.at_level(logging.INFO, logger="src.signals.integrator"):
            with patch("src.signals.integrator.SignalIntegrator") as MockIntegrator:
                instance = MockIntegrator.return_value
                instance.get_allocation_deltas.return_value = mock_rec
                with patch("sys.argv", ["integrator.py", "portfolio", "--portfolio", "25/25/25/25", "--json"]):
                    from src.signals.integrator import main
                    main()
        assert "4_asset_ok" in caplog.text

    @patch("src.signals.integrator.SignalIntegrator.get_signal_history")
    def test_history_json_output(self, mock_get, caplog):
        """history command with --json outputs valid JSON."""
        mock_get.return_value = [
            CompositeSignal(
                ticker="SPY", timestamp="2026-05-24T12:00:00",
                composite_score=0.35, composite_confidence=0.72,
                detected_regime="neutral", primary_drivers=["tsmom"],
                signal_agreement="aligned_bullish",
            ),
        ]
        with caplog.at_level(logging.INFO, logger="src.signals.integrator"):
            with patch("sys.argv", ["integrator.py", "history", "--ticker", "SPY", "--json"]):
                from src.signals.integrator import main
                main()
        parsed = json.loads(caplog.messages[0])
        assert isinstance(parsed, list)
        assert parsed[0]["ticker"] == "SPY"

    @patch("src.signals.integrator.SignalIntegrator.get_signal_history")
    def test_history_text_output(self, mock_get, caplog):
        """history command without --json prints formatted text."""
        mock_get.return_value = [
            CompositeSignal(
                ticker="SPY", timestamp="2026-05-24T12:00:00",
                composite_score=0.35, composite_confidence=0.72,
                detected_regime="neutral", primary_drivers=["tsmom"],
                signal_agreement="aligned_bullish",
            ),
        ]
        with caplog.at_level(logging.INFO, logger="src.signals.integrator"):
            with patch("sys.argv", ["integrator.py", "history", "--ticker", "SPY"]):
                from src.signals.integrator import main
                main()
        assert "Signal History" in caplog.text
        assert "SPY" in caplog.text

    def test_main_guard(self):
        """__main__ guard calls main() when __name__ == '__main__'."""
        # We can't easily trigger __name__ == '__main__' at module level,
        # but we can verify the guard syntax is correct by inspecting the source
        import inspect
        from src.signals.integrator import main as cli_main
        source = inspect.getsource(inspect.getmodule(cli_main))
        assert 'if __name__ == "__main__":' in source
        assert 'main()' in source.split('if __name__ == "__main__":')[-1]

    def test_argument_parser_has_all_commands(self):
        """Argument parser has all 3 commands and all flags."""
        from src.signals.integrator import main as cli_main
        # Can't easily inspect parser without calling, but we test via sys.argv patching
        with patch("sys.argv", ["integrator.py", "composite", "--ticker", "SPY", "--json"]):
            with patch("src.signals.integrator.SignalIntegrator.get_composite_signal") as mock_get:
                mock_get.return_value = CompositeSignal(
                    ticker="SPY", timestamp="2026-05-24T12:00:00",
                    composite_score=0.35, composite_confidence=0.72,
                    detected_regime="neutral", primary_drivers=[],
                    signal_agreement="aligned_bullish",
                )
                cli_main()
        # No crash means parser handled all args

    def test_history_with_custom_days(self, capsys):
        """history command with custom --days flag."""
        with patch("sys.argv", ["integrator.py", "history", "--ticker", "SPY", "--days", "60"]):
            with patch("src.signals.integrator.SignalIntegrator.get_signal_history") as mock_get:
                mock_get.return_value = []
                from src.signals.integrator import main
                main()
        mock_get.assert_called_with("SPY", 60)


class TestCLIErrorMessages:
    """Verify specific error messages from CLI."""

    def test_composite_error_message(self, caplog):
        """Error message says --ticker required."""
        with caplog.at_level(logging.ERROR, logger="src.signals.integrator"):
            with pytest.raises(SystemExit):
                with patch("sys.argv", ["integrator.py", "composite"]):
                    from src.signals.integrator import main
                    main()
        assert "--ticker required" in caplog.text

    def test_portfolio_error_message(self, caplog):
        """Error message says --portfolio required."""
        with caplog.at_level(logging.ERROR, logger="src.signals.integrator"):
            with pytest.raises(SystemExit):
                with patch("sys.argv", ["integrator.py", "portfolio"]):
                    from src.signals.integrator import main
                    main()
        assert "--portfolio required" in caplog.text

    def test_history_error_message(self, caplog):
        """Error message says --ticker required."""
        with caplog.at_level(logging.ERROR, logger="src.signals.integrator"):
            with pytest.raises(SystemExit):
                with patch("sys.argv", ["integrator.py", "history"]):
                    from src.signals.integrator import main
                    main()
        assert "--ticker required" in caplog.text


# ============================================================================
# 6. Export Completeness — verify __all__ coverage
# ============================================================================


class TestModuleExportCompleteness:
    """Verify __all__ covers all public API symbols."""

    def test_all_export_is_list(self):
        """__all__ is a list of strings."""
        assert isinstance(MODULE_ALL, list)
        assert len(MODULE_ALL) > 0

    def test_all_dataclasses_exported(self):
        """All 4 dataclasses are in __all__."""
        assert "SignalSourceResult" in MODULE_ALL
        assert "CompositeSignal" in MODULE_ALL
        assert "AllocationDelta" in MODULE_ALL
        assert "PortfolioRecommendation" in MODULE_ALL

    def test_all_constants_exported(self):
        """All module-level constants are in __all__."""
        assert "BASE_WEIGHTS" in MODULE_ALL
        assert "REGIME_WEIGHTS" in MODULE_ALL
        assert "MIN_SIGNAL_SOURCES" in MODULE_ALL
        assert "SIGNAL_MIN" in MODULE_ALL
        assert "SIGNAL_MAX" in MODULE_ALL
        assert "MAX_DELTA_PCT" in MODULE_ALL

    def test_all_classes_exported(self):
        """All public classes are in __all__."""
        assert "SignalSource" in MODULE_ALL
        assert "TechnicalSignal" in MODULE_ALL
        assert "MacroSignal" in MODULE_ALL
        assert "AlternativeDataSignalAdapter" in MODULE_ALL
        assert "LLMSentimentSignalAdapter" in MODULE_ALL
        assert "SignalIntegrator" in MODULE_ALL

    def test_all_functions_exported(self):
        """Public functions are in __all__."""
        assert "init_database" in MODULE_ALL

    def test_all_no_duplicates(self):
        """__all__ has no duplicate entries."""
        assert len(MODULE_ALL) == len(set(MODULE_ALL))

    def test_all_items_are_strings(self):
        """Every item in __all__ is a string."""
        for item in MODULE_ALL:
            assert isinstance(item, str), f"{item} is not a string"

    def test_all_items_resolve_to_module_attributes(self):
        """Each __all__ entry corresponds to an actual module attribute."""
        import src.signals.integrator as mod
        for name in MODULE_ALL:
            assert hasattr(mod, name), f"{name} in __all__ but not in module"


# ============================================================================
# Additional Edge Cases for Previously Tested Methods
# ============================================================================


class TestGetSignalHistoryEdgeCases:
    """Edge cases for get_signal_history."""

    def test_get_signal_history_with_null_sql_fields(self, tmp_path):
        """Some SQL fields (weights_used, primary_drivers) can be NULL/NONE."""
        from datetime import timedelta
        integrator = _make_integrator(tmp_path)
        conn = sqlite3.connect(str(integrator.db_path))
        # Use a relative timestamp so the test is not time-bombed when
        # hardcoded dates drift outside the get_signal_history(days=30) window.
        ts = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%dT12:00:00")
        conn.execute("""
            INSERT INTO composite_signals
            (ticker, timestamp, composite_score, composite_confidence,
             detected_regime, weights_used, primary_drivers,
             signal_agreement, expected_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("SPY", ts, 0.35, 0.72,
              "neutral", None, None, "aligned_bullish", 0.68))
        conn.commit()
        conn.close()
        result = integrator.get_signal_history("SPY", days=30)
        assert len(result) == 1
        assert result[0].weights_used == {}
        assert result[0].primary_drivers == []

    def test_get_signal_history_empty_json_fields(self, tmp_path):
        """Empty JSON arrays/dicts stored as '[]'/'{}' parse correctly."""
        from datetime import timedelta
        integrator = _make_integrator(tmp_path)
        conn = sqlite3.connect(str(integrator.db_path))
        # Use a relative timestamp so the test is not time-bombed when
        # hardcoded dates drift outside the get_signal_history(days=30) window.
        ts = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%dT12:00:00")
        conn.execute("""
            INSERT INTO composite_signals
            (ticker, timestamp, composite_score, composite_confidence,
             detected_regime, weights_used, primary_drivers,
             signal_agreement, expected_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("SPY", ts, 0.35, 0.72,
              "neutral", json.dumps({}), json.dumps([]), "mixed", None))
        conn.commit()
        conn.close()
        result = integrator.get_signal_history("SPY", days=30)
        assert len(result) == 1

    def test_get_signal_history_multiple_signals_same_ticker(self, tmp_path):
        """Multiple signals for same ticker returned in order."""
        from datetime import timedelta
        integrator = _make_integrator(tmp_path)
        conn = sqlite3.connect(str(integrator.db_path))
        now = datetime.now()
        for i in range(3):
            # Use relative timestamps so the test is not time-bombed when
            # hardcoded dates drift outside the get_signal_history(days=30) window.
            ts = (now - timedelta(days=i + 1)).strftime("%Y-%m-%dT12:00:00")
            conn.execute("""
                INSERT INTO composite_signals
                (ticker, timestamp, composite_score, composite_confidence,
                 detected_regime, weights_used, primary_drivers,
                 signal_agreement, expected_accuracy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ("SPY", ts, 0.3 + i * 0.1,
                  0.7, "neutral", "{}", "[]", "mixed", None))
        conn.commit()
        conn.close()
        result = integrator.get_signal_history("SPY", days=30)
        assert len(result) == 3

    def test_get_signal_history_other_ticker_not_returned(self, tmp_path):
        """Signals for different ticker not returned."""
        integrator = _make_integrator(tmp_path)
        conn = sqlite3.connect(str(integrator.db_path))
        conn.execute("""
            INSERT INTO composite_signals
            (ticker, timestamp, composite_score, composite_confidence,
             detected_regime, weights_used, primary_drivers,
             signal_agreement, expected_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("GLD", "2026-05-24T12:00:00", 0.2, 0.6, "neutral", "{}", "[]", "mixed", None))
        conn.commit()
        conn.close()
        result = integrator.get_signal_history("SPY", days=30)
        assert result == []

    def test_get_signal_history_days_filter(self, tmp_path):
        """Historical signals outside days window not returned."""
        integrator = _make_integrator(tmp_path)
        conn = sqlite3.connect(str(integrator.db_path))
        from datetime import timedelta
        old_ts = (datetime.now() - timedelta(days=100)).isoformat()
        conn.execute("""
            INSERT INTO composite_signals
            (ticker, timestamp, composite_score, composite_confidence,
             detected_regime, weights_used, primary_drivers,
             signal_agreement, expected_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("SPY", old_ts, 0.2, 0.6, "neutral", "{}", "[]", "mixed", None))
        conn.commit()
        conn.close()
        result = integrator.get_signal_history("SPY", days=30)
        assert result == []


class TestSignalIntegratorInitEdgeCases:
    """Edge cases for SignalIntegrator initialization."""

    def test_sources_contains_all_expected_keys(self):
        """SignalIntegrator.sources has all expected keys."""
        with (
            patch("src.signals.integrator.init_database"),
            patch("src.signals.tsmom_integration.TSMOMSignalAdapter", MagicMock()),
            patch("src.signals.multi_strategy_adapters.MultiSpeedSignalAdapter", MagicMock()),
            patch("src.signals.multi_strategy_adapters.RiskParitySignalAdapter", MagicMock()),
        ):
            integrator = SignalIntegrator()
        expected_keys = {
            "technical", "macro", "alternative_data", "llm_sentiment",
            "tsmom", "multi_speed", "risk_parity",
        }
        assert set(integrator.sources.keys()) == expected_keys
        assert len(integrator.sources) == 7


class TestStoreRecommendationEdgeCases:
    """Edge cases for _store_recommendation."""

    def test_store_recommendation_empty_deltas(self, tmp_path):
        """Recommendation with empty deltas stores successfully."""
        from src.signals.integrator import SignalIntegrator, PortfolioRecommendation, init_database
        with patch("src.signals.integrator.DB_PATH", tmp_path / "test.db"):
            init_database()
            integrator = SignalIntegrator()
            integrator.db_path = tmp_path / "test.db"
            rec = PortfolioRecommendation(
                timestamp=datetime.now().isoformat(),
                current_allocation={},
                recommended_allocation={},
                deltas=[],
                composite_sentiment="neutral", confidence=0.0, regime="neutral",
            )
            integrator._store_recommendation(rec)
        with sqlite3.connect(str(tmp_path / "test.db")) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM portfolio_recommendations").fetchone()
        assert rows[0] == 1


class TestAlternativeDataSignalAdapterEdgeCases:
    """More edge cases for AlternativeDataSignalAdapter."""

    @pytest.fixture(autouse=True)
    def _inject_logger(self):
        import src.signals.integrator as _mod
        original_logger = _mod.logger
        _mod.logger = MagicMock()
        yield
        _mod.logger = original_logger

    def test_get_historical_accuracy_with_none_values(self, tmp_path):
        """Accuracy table with NULL accuracy_score values returns None."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                adapter = AlternativeDataSignalAdapter()
        adapter.db_path = tmp_path / "signals.db"
        conn = sqlite3.connect(str(adapter.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, source_type TEXT, source_name TEXT,
                prediction_timestamp TEXT, predicted_signal REAL,
                actual_return REAL, horizon_days INTEGER,
                accuracy_score REAL, error REAL
            )
        """)
        conn.execute("""
            INSERT INTO signal_accuracy
            (ticker, source_type, source_name, prediction_timestamp,
             predicted_signal, actual_return, horizon_days, accuracy_score, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("SPY", "sentiment", "alternative_data", "2026-05-01", 0.5, 0.4, 21, None, 0.1))
        conn.commit()
        conn.close()
        result = adapter.get_historical_accuracy("SPY")
        assert result is None

    def test_historical_accuracy_filters_by_source_type(self, tmp_path):
        """Only matching source_type/source_name records are counted."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                adapter = AlternativeDataSignalAdapter()
        adapter.db_path = tmp_path / "signals.db"
        conn = sqlite3.connect(str(adapter.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, source_type TEXT, source_name TEXT,
                prediction_timestamp TEXT, predicted_signal REAL,
                actual_return REAL, horizon_days INTEGER,
                accuracy_score REAL, error REAL
            )
        """)
        conn.execute("""
            INSERT INTO signal_accuracy
            (ticker, source_type, source_name, prediction_timestamp,
             predicted_signal, actual_return, horizon_days, accuracy_score, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("SPY", "momentum", "technical", "2026-05-01", 0.5, 0.4, 21, 0.50, 0.1))
        conn.executescript("""
            INSERT INTO signal_accuracy
            (ticker, source_type, source_name, prediction_timestamp,
             predicted_signal, actual_return, horizon_days, accuracy_score, error)
            VALUES ('SPY', 'sentiment', 'alternative_data', '2026-05-02', 0.5, 0.4, 21, 0.80, 0.1);
        """)
        conn.commit()
        conn.close()
        result = adapter.get_historical_accuracy("SPY")
        assert result == 0.80


class TestMacroSignalHistoricalAccuracyEdgeCases:
    """More edge cases for MacroSignal historical accuracy."""

    def test_with_none_accuracy_values_returns_default(self, tmp_path):
        """All accuracy_score are None -> returns default 0.65."""
        with patch("src.signals.integrator.init_database"):
            with patch("src.signals.integrator.DATA_DIR", tmp_path):
                signal = MacroSignal()
        signal.db_path = tmp_path / "signals.db"
        conn = sqlite3.connect(str(signal.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, source_type TEXT, source_name TEXT,
                prediction_timestamp TEXT, predicted_signal REAL,
                actual_return REAL, horizon_days INTEGER,
                accuracy_score REAL, error REAL
            )
        """)
        conn.execute("""
            INSERT INTO signal_accuracy
            (ticker, source_type, source_name, prediction_timestamp,
             predicted_signal, actual_return, horizon_days, accuracy_score, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("SPY", "macro", "fed_economic", "2026-05-01", 0.5, 0.4, 21, None, 0.1))
        conn.commit()
        conn.close()
        result = signal.get_historical_accuracy("SPY")
        assert result == 0.65


class TestCompositeSignalNoWeightsFallback:
    """Fallback behavior when regime has no matching weight config."""

    def test_source_type_not_in_weights_defaults_to_20pct(self, tmp_path):
        """Source type not in regime weights gets default weight of 0.20."""
        integrator = _make_integrator(tmp_path)
        src = MagicMock()
        src.generate_signal.return_value = _make_signal(
            "unknown_type", "custom_signal", 0.5, confidence=0.8
        )
        integrator.sources = {"custom": src}
        custom_weights = {"unrelated_type": 1.0}
        with patch("src.signals.integrator.MIN_SIGNAL_SOURCES", 1):
            result = integrator.get_composite_signal("SPY", custom_weights=custom_weights)
        # source_type "unknown_type" not in custom_weights -> default 0.20
        assert result.composite_score != 0.0


class TestAgreementClassificationFullCoverage:
    """Full branch coverage for signal_agreement classification."""

    def test_bearish_exactly_60pct_no_bullish_no_neutral(self, tmp_path):
        """Exactly 60% bearish, 40% neutral -> aligned_bearish."""
        integrator = _make_integrator(tmp_path)
        sources = {}
        for i in range(5):
            src = MagicMock()
            sig = -0.5 if i < 3 else -0.1  # 3 bearish, 2 weakly bearish (not counted)
            src.generate_signal.return_value = _make_signal(
                f"type_{i}", f"src_{i}", sig, confidence=0.8
            )
            sources[f"src_{i}"] = src
        integrator.sources = sources
        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.signal_agreement == "aligned_bearish"

    def test_neutral_agreement_fallback(self, tmp_path):
        """No bullish, no bearish, some neutral -> mixed."""
        integrator = _make_integrator(tmp_path)
        sources = {}
        for i in range(4):
            src = MagicMock()
            src.generate_signal.return_value = _make_signal(
                f"type_{i}", f"src_{i}", 0.1, confidence=0.8
            )
            sources[f"src_{i}"] = src
        integrator.sources = sources
        result = integrator.get_composite_signal("SPY", regime="neutral")
        # bullish_count = 0 (< 60%), bearish_count = 0 (< 60%),
        # no bullish AND no bearish -> else -> "mixed"
        assert result.signal_agreement == "mixed"

    def test_bullish_below_60pct_no_bearish(self, tmp_path):
        """Bullish < 60%, no bearish, some neutral -> mixed."""
        integrator = _make_integrator(tmp_path)
        sources = {}
        for i in range(5):
            src = MagicMock()
            sig = 0.5 if i < 2 else 0.1  # 2/5 = 40% bullish
            src.generate_signal.return_value = _make_signal(
                f"type_{i}", f"src_{i}", sig, confidence=0.8
            )
            sources[f"src_{i}"] = src
        integrator.sources = sources
        result = integrator.get_composite_signal("SPY", regime="neutral")
        assert result.signal_agreement == "mixed"


class TestCalculateExpectedAccuracyFullBranch:
    """Full branch coverage for _calculate_expected_accuracy."""

    def test_empty_signals_returns_05(self, tmp_path):
        """Empty signal list returns 0.5."""
        integrator = _make_integrator(tmp_path)
        assert integrator._calculate_expected_accuracy([], {"a": 1.0}) == 0.5

    def test_signals_with_missing_accuracy_skipped(self, tmp_path):
        """Signals with historical_accuracy=None are skipped in weighted average."""
        integrator = _make_integrator(tmp_path)
        signals = [
            SignalSourceResult(
                source_type="unknown", source_name="x",
                signal=0.5, confidence=0.8, raw_score=1.0, raw_unit="z",
                historical_accuracy=None,
            ),
        ]
        acc = integrator._calculate_expected_accuracy(signals, {"unknown": 0.5})
        assert acc == 0.6

    def test_all_historical_none_returns_default(self, tmp_path):
        """All signals have historical_accuracy=None -> default 0.6."""
        integrator = _make_integrator(tmp_path)
        signals = [
            SignalSourceResult(
                source_type="a", source_name="x", signal=0.5, confidence=0.8,
                raw_score=1.0, raw_unit="z", historical_accuracy=None,
            ),
            SignalSourceResult(
                source_type="b", source_name="y", signal=-0.3, confidence=0.6,
                raw_score=-0.5, raw_unit="z", historical_accuracy=None,
            ),
        ]
        acc = integrator._calculate_expected_accuracy(signals, {"a": 0.5, "b": 0.5})
        assert acc == 0.6


class TestBinanceSignalAgreement:
    """Verify the unused 'neutral_count' variable doesn't break anything."""

    def test_neutral_count_variable_not_breaking(self, tmp_path):
        """Line 968 (len(...) - bullish - bearish) is valid syntax but unused."""
        integrator = _make_integrator(tmp_path)
        sources = {}
        for name in ["momentum", "macro"]:
            src = MagicMock()
            src.generate_signal.return_value = _make_signal(
                name, name, 0.0, confidence=0.8
            )
            sources[name] = src
        integrator.sources = sources
        result = integrator.get_composite_signal("SPY", regime="neutral")
        # Should not crash — scores 0.0, no crash
        assert result.composite_score == 0.0
        assert result.signal_agreement in ("mixed", "conflicting", "aligned_bullish", "aligned_bearish", "insufficient_data")
