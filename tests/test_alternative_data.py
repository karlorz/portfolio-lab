#!/usr/bin/env python3
"""
Tests for alternative data module — data classes, adapters, composite signals,
earnings predictions.
"""
import sys
import os
import json
import sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.alternative_data import (
    AlternativeDataSignal, CompositeSignal, EarningsPrediction,
    AlternativeDataClient, SatelliteDataAdapter, CreditCardAdapter,
    SupplyChainAdapter, init_database, ALT_DATA_DB,
)


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestDataClasses:
    """Test dataclass serialization."""

    def test_alternative_data_signal_to_dict(self):
        sig = AlternativeDataSignal(
            ticker="SPY", source="satellite", signal_type="momentum",
            score=0.5, confidence=0.8, raw_value=12.5, raw_unit="pct_change",
            period_days=30, z_score=1.2, percentile=85.0,
            trend_direction="improving", data_timestamp=datetime.now().isoformat(),
        )
        d = sig.to_dict()
        assert d["ticker"] == "SPY"
        assert d["score"] == 0.5
        assert d["source"] == "satellite"
        assert d["trend_direction"] == "improving"

    def test_composite_signal_to_dict(self):
        cs = CompositeSignal(
            ticker="SPY", satellite_score=0.4, credit_card_score=0.6,
            supply_chain_score=0.2, composite_score=0.45,
            composite_confidence=0.7, primary_driver="credit_card",
            signal_agreement="aligned",
        )
        d = cs.to_dict()
        assert d["ticker"] == "SPY"
        assert d["primary_driver"] == "credit_card"
        assert d["composite_score"] == 0.45

    def test_composite_signal_defaults(self):
        cs = CompositeSignal(ticker="GLD")
        assert cs.composite_score == 0.0
        assert cs.composite_confidence == 0.0
        assert cs.satellite_score is None
        assert cs.signal_agreement == "neutral"

    def test_earnings_prediction_to_dict(self):
        ep = EarningsPrediction(
            ticker="AAPL", quarter="Q4-2025",
            predicted_revenue_growth=8.5, revenue_surprise_probability=0.72,
            revenue_direction="beat", confidence=0.65,
            primary_signals=["satellite", "credit_card"],
        )
        d = ep.to_dict()
        assert d["ticker"] == "AAPL"
        assert d["revenue_direction"] == "beat"
        assert len(d["primary_signals"]) == 2


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------

class TestDatabase:
    """Test database initialization."""

    def test_init_database_creates_tables(self, tmp_path):
        """init_database creates all expected tables."""
        with patch("src.data.alternative_data.DATA_DIR", tmp_path):
            with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
                init_database()

        conn = sqlite3.connect(str(tmp_path / "alt.db"))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "satellite_data" in tables
        assert "credit_card_data" in tables
        assert "supply_chain_data" in tables
        assert "alt_data_signals" in tables


# ---------------------------------------------------------------------------
# Adapter tests (synthetic data path)
# ---------------------------------------------------------------------------

class TestSatelliteAdapter:
    """Test SatelliteDataAdapter with synthetic data."""

    def test_calculate_signal_returns_signal(self, tmp_path):
        """Adapter returns valid AlternativeDataSignal."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("WMT", days=30)

        assert isinstance(signal, AlternativeDataSignal)
        assert signal.source == "satellite"
        assert signal.ticker == "WMT"
        assert -1.0 <= signal.score <= 1.0
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.trend_direction in ["improving", "deteriorating", "stable", "insufficient_data"]

    def test_signal_stored_in_db(self, tmp_path):
        """Signal is persisted to database."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("WMT", days=30)

            conn = sqlite3.connect(str(tmp_path / "alt.db"))
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM alt_data_signals WHERE ticker = ?", ("WMT",))
            rows = cursor.fetchall()
            conn.close()

        assert len(rows) >= 1

    def test_synthetic_data_generation(self, tmp_path):
        """Synthetic data produces valid records."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
            data = adapter.fetch_data("WMT", days=30)

        assert len(data) > 0
        assert "date" in data[0]
        assert "parking_occupancy_pct" in data[0]


class TestCreditCardAdapter:
    """Test CreditCardAdapter with synthetic data."""

    def test_calculate_signal_returns_signal(self, tmp_path):
        """Adapter returns valid signal."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("AMZN", days=30)

        assert isinstance(signal, AlternativeDataSignal)
        assert signal.source == "credit_card"
        assert -1.0 <= signal.score <= 1.0

    def test_signal_type_is_spending(self, tmp_path):
        """Credit card signal type is spending_momentum."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("AMZN", days=30)

        assert signal.signal_type == "spending_momentum"


class TestSupplyChainAdapter:
    """Test SupplyChainAdapter with synthetic data."""

    def test_calculate_signal_returns_signal(self, tmp_path):
        """Adapter returns valid signal."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("SPY", days=30)

        assert isinstance(signal, AlternativeDataSignal)
        assert signal.source == "supply_chain"
        assert -1.0 <= signal.score <= 1.0

    def test_signal_type_is_efficiency(self, tmp_path):
        """Supply chain signal type is operational_efficiency."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("AAPL", days=30)

        assert signal.signal_type == "operational_efficiency"


# ---------------------------------------------------------------------------
# AlternativeDataClient tests
# ---------------------------------------------------------------------------

class TestAlternativeDataClient:
    """Test the unified client."""

    def test_composite_signal(self, tmp_path):
        """Composite signal aggregates all sources."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            composite = client.get_composite_signal("SPY", days=30)

        assert isinstance(composite, CompositeSignal)
        assert composite.ticker == "SPY"
        assert -1.0 <= composite.composite_score <= 1.0
        assert 0.0 <= composite.composite_confidence <= 1.0
        assert composite.primary_driver in ["satellite", "credit_card", "supply_chain", "none"]

    def test_composite_weights(self, tmp_path):
        """Source weights sum to 1.0."""
        assert abs(sum(AlternativeDataClient.SOURCE_WEIGHTS.values()) - 1.0) < 0.01

    def test_agreement_detection_aligned(self, tmp_path):
        """All sources bullish → aligned."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()

            # Mock all adapters to return bullish signals
            bull = AlternativeDataSignal(
                ticker="SPY", source="satellite", signal_type="momentum",
                score=0.6, confidence=0.8, raw_value=10.0, raw_unit="pct",
                period_days=30, z_score=1.5, percentile=90.0,
                trend_direction="improving", data_timestamp=datetime.now().isoformat(),
            )
            client.get_satellite_signal = lambda t, d=30: bull
            client.get_credit_card_signal = lambda t, d=30: AlternativeDataSignal(
                ticker="SPY", source="credit_card", signal_type="spending",
                score=0.5, confidence=0.7, raw_value=8.0, raw_unit="pct",
                period_days=30, z_score=1.2, percentile=85.0,
                trend_direction="improving", data_timestamp=datetime.now().isoformat(),
            )
            client.get_supply_chain_signal = lambda t, d=30: AlternativeDataSignal(
                ticker="SPY", source="supply_chain", signal_type="efficiency",
                score=0.4, confidence=0.6, raw_value=5.0, raw_unit="pct",
                period_days=30, z_score=0.8, percentile=75.0,
                trend_direction="improving", data_timestamp=datetime.now().isoformat(),
            )

            composite = client.get_composite_signal("SPY")
            assert composite.signal_agreement == "aligned"
            assert composite.composite_score > 0

    def test_agreement_detection_conflicting(self, tmp_path):
        """Bullish + bearish → conflicting."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()

            client.get_satellite_signal = lambda t, d=30: AlternativeDataSignal(
                ticker="SPY", source="satellite", signal_type="momentum",
                score=0.6, confidence=0.8, raw_value=10.0, raw_unit="pct",
                period_days=30, z_score=1.5, percentile=90.0,
                trend_direction="improving", data_timestamp=datetime.now().isoformat(),
            )
            client.get_credit_card_signal = lambda t, d=30: AlternativeDataSignal(
                ticker="SPY", source="credit_card", signal_type="spending",
                score=-0.5, confidence=0.7, raw_value=-8.0, raw_unit="pct",
                period_days=30, z_score=-1.2, percentile=15.0,
                trend_direction="deteriorating", data_timestamp=datetime.now().isoformat(),
            )
            client.get_supply_chain_signal = lambda t, d=30: None

            composite = client.get_composite_signal("SPY")
            assert composite.signal_agreement == "conflicting"

    def test_insufficient_data_agreement(self, tmp_path):
        """Only one source → insufficient_data."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()

            client.get_satellite_signal = lambda t, d=30: AlternativeDataSignal(
                ticker="SPY", source="satellite", signal_type="momentum",
                score=0.3, confidence=0.5, raw_value=5.0, raw_unit="pct",
                period_days=30, z_score=0.5, percentile=65.0,
                trend_direction="stable", data_timestamp=datetime.now().isoformat(),
            )
            client.get_credit_card_signal = lambda t, d=30: None
            client.get_supply_chain_signal = lambda t, d=30: None

            composite = client.get_composite_signal("SPY")
            assert composite.signal_agreement == "insufficient_data"


class TestEarningsPrediction:
    """Test earnings prediction logic."""

    def test_strong_bullish_predicts_beat(self, tmp_path):
        """High composite score → beat prediction."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()

            # Mock composite with high score
            mock_composite = CompositeSignal(
                ticker="SPY", satellite_score=0.7, credit_card_score=0.8,
                supply_chain_score=0.5, composite_score=0.6,
                composite_confidence=0.7, primary_driver="credit_card",
                signal_agreement="aligned",
            )
            client.get_composite_signal = lambda t, d=30: mock_composite

            pred = client.get_earnings_prediction("SPY", "Q4-2025")
            assert pred is not None
            assert pred.revenue_direction == "beat"
            assert pred.predicted_revenue_growth > 5.0

    def test_bearish_predicts_miss(self, tmp_path):
        """Negative composite score → miss prediction."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()

            mock_composite = CompositeSignal(
                ticker="SPY", satellite_score=-0.5, credit_card_score=-0.6,
                supply_chain_score=-0.3, composite_score=-0.5,
                composite_confidence=0.7, primary_driver="credit_card",
                signal_agreement="aligned",
            )
            client.get_composite_signal = lambda t, d=30: mock_composite

            pred = client.get_earnings_prediction("SPY", "Q4-2025")
            assert pred is not None
            assert pred.revenue_direction == "miss"
            assert pred.predicted_revenue_growth < 0

    def test_low_confidence_returns_none(self, tmp_path):
        """Low confidence → None prediction."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()

            mock_composite = CompositeSignal(
                ticker="SPY", composite_score=0.1, composite_confidence=0.2,
            )
            client.get_composite_signal = lambda t, d=30: mock_composite

            pred = client.get_earnings_prediction("SPY", "Q4-2025")
            assert pred is None

    def test_inline_direction(self, tmp_path):
        """Neutral score → inline prediction."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()

            mock_composite = CompositeSignal(
                ticker="SPY", satellite_score=0.05, credit_card_score=0.1,
                composite_score=0.05, composite_confidence=0.6,
                primary_driver="credit_card", signal_agreement="mixed",
            )
            client.get_composite_signal = lambda t, d=30: mock_composite

            pred = client.get_earnings_prediction("SPY", "Q1-2026")
            assert pred is not None
            assert pred.revenue_direction == "inline"


class TestBatchSignals:
    """Test batch signal retrieval."""

    def test_batch_returns_multiple(self, tmp_path):
        """Batch signals returns results for each ticker."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            results = client.get_batch_signals(["SPY", "GLD"], days=30)

        assert "SPY" in results
        assert "GLD" in results
        assert isinstance(results["SPY"], CompositeSignal)
        assert isinstance(results["GLD"], CompositeSignal)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
