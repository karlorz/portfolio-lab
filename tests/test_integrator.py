#!/usr/bin/env python3
"""
Tests for signal integrator — data structures, normalization, composite signal
aggregation, allocation deltas, regime detection, signal agreement.
"""
import json
import sqlite3

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

