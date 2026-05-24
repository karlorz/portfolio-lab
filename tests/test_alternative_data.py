#!/usr/bin/env python3
"""
Tests for alternative data module — data classes, adapters, composite signals,
earnings predictions.
"""
import io
import json
import logging
import sqlite3
import sys

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
            signal = adapter.calculate_signal("AAPL", days=30)

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


class TestAlternativeDataSignalDataclass:
    """Extended tests for AlternativeDataSignal dataclass."""

    def test_all_fields_present(self):
        sig = AlternativeDataSignal(
            ticker="SPY", source="satellite", signal_type="momentum",
            score=0.5, confidence=0.8, raw_value=12.5, raw_unit="pct_change",
            period_days=30, z_score=1.2, percentile=85.0,
            trend_direction="improving", data_timestamp="2026-01-01T00:00:00",
        )
        assert sig.ticker == "SPY"
        assert sig.source == "satellite"
        assert sig.signal_type == "momentum"
        assert sig.score == 0.5
        assert sig.confidence == 0.8
        assert sig.raw_value == 12.5
        assert sig.raw_unit == "pct_change"
        assert sig.period_days == 30
        assert sig.z_score == 1.2
        assert sig.percentile == 85.0
        assert sig.trend_direction == "improving"
        assert sig.data_timestamp == "2026-01-01T00:00:00"
        assert sig.model_version == "v2.23.0"

    def test_to_dict_completeness(self):
        sig = AlternativeDataSignal(
            ticker="GLD", source="credit_card", signal_type="spending",
            score=-0.3, confidence=0.6, raw_value=-5.0, raw_unit="index",
            period_days=60, z_score=-0.8, percentile=25.0,
            trend_direction="deteriorating", data_timestamp="2026-02-01T00:00:00",
        )
        d = sig.to_dict()
        expected_keys = {
            "ticker", "source", "signal_type", "score", "confidence",
            "raw_value", "raw_unit", "period_days", "z_score", "percentile",
            "trend_direction", "data_timestamp", "signal_generated", "model_version",
        }
        assert set(d.keys()) == expected_keys

    def test_default_fields(self):
        sig = AlternativeDataSignal(
            ticker="TLT", source="supply_chain", signal_type="efficiency",
            score=0.0, confidence=0.0, raw_value=0.0, raw_unit="count",
            period_days=1, z_score=0.0, percentile=50.0,
            trend_direction="stable", data_timestamp="2026-01-01",
        )
        assert sig.signal_generated is not None
        assert sig.model_version == "v2.23.0"

    def test_extreme_score_values(self):
        sig = AlternativeDataSignal(
            ticker="SPY", source="satellite", signal_type="momentum",
            score=1.0, confidence=1.0, raw_value=999.0, raw_unit="pct",
            period_days=90, z_score=3.0, percentile=99.9,
            trend_direction="improving", data_timestamp="2026-01-01",
        )
        assert sig.score == 1.0
        d = sig.to_dict()
        assert d["score"] == 1.0

    def test_negative_score(self):
        sig = AlternativeDataSignal(
            ticker="SPY", source="credit_card", signal_type="surprise",
            score=-1.0, confidence=0.5, raw_value=-50.0, raw_unit="pct",
            period_days=30, z_score=-2.5, percentile=1.0,
            trend_direction="deteriorating", data_timestamp="2026-01-01",
        )
        assert sig.score == -1.0
        assert sig.z_score == -2.5


class TestCompositeSignalDataclass:
    """Extended tests for CompositeSignal dataclass."""

    def test_all_fields_present(self):
        cs = CompositeSignal(
            ticker="SPY", satellite_score=0.4, credit_card_score=0.6,
            supply_chain_score=0.2, composite_score=0.45,
            composite_confidence=0.7, primary_driver="credit_card",
            signal_agreement="aligned", historical_accuracy=0.72,
        )
        assert cs.ticker == "SPY"
        assert cs.satellite_score == 0.4
        assert cs.credit_card_score == 0.6
        assert cs.supply_chain_score == 0.2
        assert cs.composite_score == 0.45
        assert cs.composite_confidence == 0.7
        assert cs.primary_driver == "credit_card"
        assert cs.signal_agreement == "aligned"
        assert cs.historical_accuracy == 0.72
        assert cs.timestamp is not None

    def test_to_dict_completeness(self):
        cs = CompositeSignal(ticker="GLD")
        d = cs.to_dict()
        expected_keys = {
            "ticker", "satellite_score", "credit_card_score", "supply_chain_score",
            "composite_score", "composite_confidence", "primary_driver",
            "signal_agreement", "historical_accuracy", "timestamp",
        }
        assert set(d.keys()) == expected_keys

    def test_minimal_initialization(self):
        cs = CompositeSignal(ticker="TLT")
        assert cs.satellite_score is None
        assert cs.credit_card_score is None
        assert cs.supply_chain_score is None
        assert cs.composite_score == 0.0
        assert cs.composite_confidence == 0.0
        assert cs.primary_driver == ""
        assert cs.signal_agreement == "neutral"
        assert cs.historical_accuracy is None


class TestEarningsPredictionDataclass:
    """Extended tests for EarningsPrediction dataclass."""

    def test_all_fields_present(self):
        ep = EarningsPrediction(
            ticker="AAPL", quarter="Q4-2025",
            predicted_revenue_growth=8.5, revenue_surprise_probability=0.72,
            revenue_direction="beat", confidence=0.65,
            primary_signals=["satellite", "credit_card"],
            historical_accuracy=0.68, earnings_date="2026-01-28",
        )
        assert ep.ticker == "AAPL"
        assert ep.quarter == "Q4-2025"
        assert ep.predicted_revenue_growth == 8.5
        assert ep.revenue_surprise_probability == 0.72
        assert ep.revenue_direction == "beat"
        assert ep.confidence == 0.65
        assert len(ep.primary_signals) == 2
        assert ep.historical_accuracy == 0.68
        assert ep.earnings_date == "2026-01-28"

    def test_to_dict_completeness(self):
        ep = EarningsPrediction(
            ticker="MSFT", quarter="Q1-2026",
            predicted_revenue_growth=-3.0, revenue_surprise_probability=0.55,
            revenue_direction="miss", confidence=0.5,
            primary_signals=["supply_chain"],
        )
        d = ep.to_dict()
        expected_keys = {
            "ticker", "quarter", "predicted_revenue_growth",
            "revenue_surprise_probability", "revenue_direction", "confidence",
            "primary_signals", "historical_accuracy", "prediction_date", "earnings_date",
        }
        assert set(d.keys()) == expected_keys

    def test_defaults(self):
        ep = EarningsPrediction(
            ticker="SPY", quarter="Q2-2026",
            predicted_revenue_growth=0.0, revenue_surprise_probability=0.3,
            revenue_direction="inline", confidence=0.4,
            primary_signals=[],
        )
        assert ep.historical_accuracy is None
        assert ep.prediction_date is not None
        assert ep.earnings_date is None


class TestAlternativeDataAdapter:
    """Extended tests for AlternativeDataAdapter ABC."""

    def test_adapter_init(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            from src.data.alternative_data import AlternativeDataAdapter
            adapter = SatelliteDataAdapter()
            assert adapter.source_name == "satellite"

    def test_satellite_adapter_source_name(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            adapter = SatelliteDataAdapter()
            assert adapter.source_name == "satellite"

    def test_credit_card_adapter_source_name(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            adapter = CreditCardAdapter()
            assert adapter.source_name == "credit_card"

    def test_supply_chain_adapter_source_name(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            adapter = SupplyChainAdapter()
            assert adapter.source_name == "supply_chain"


class TestDatabaseExtended:
    """Extended database tests."""

    def test_init_database_idempotent(self, tmp_path):
        """Calling init_database twice should not error."""
        with patch("src.data.alternative_data.DATA_DIR", tmp_path):
            with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
                init_database()
                init_database()  # Should not raise
        assert (tmp_path / "alt.db").exists()

    def test_alt_data_db_path(self):
        """ALT_DATA_DB should be under DATA_DIR."""
        assert "alternative_data" in str(ALT_DATA_DB)
        assert str(ALT_DATA_DB).endswith(".db")


class TestSatelliteAdapterExtended:
    """Extended satellite adapter tests."""

    def test_satellite_signal_trend_directions(self, tmp_path):
        """Trend direction should be one of the valid values."""
        valid_directions = {"improving", "deteriorating", "stable", "insufficient_data"}
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("TGT", days=30)
        assert signal.trend_direction in valid_directions

    def test_satellite_signal_score_range(self, tmp_path):
        """Score should always be in [-1, 1]."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
            for ticker in ["WMT", "TGT", "COST"]:
                signal = adapter.calculate_signal(ticker, days=30)
                assert -1.0 <= signal.score <= 1.0, f"{ticker} score {signal.score} out of range"

    def test_satellite_signal_confidence_range(self, tmp_path):
        """Confidence should always be in [0, 1]."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("WMT", days=30)
        assert 0.0 <= signal.confidence <= 1.0

    def test_satellite_signal_period_days(self, tmp_path):
        """Signal should reflect the requested period."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("WMT", days=60)
        assert signal.period_days == 60

    def test_satellite_fetch_data_keys(self, tmp_path):
        """Fetched data should have expected keys."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
            data = adapter.fetch_data("WMT", days=30)
        for record in data:
            assert "date" in record
            assert "parking_occupancy_pct" in record


class TestCreditCardAdapterExtended:
    """Extended credit card adapter tests."""

    def test_credit_card_signal_score_range(self, tmp_path):
        """Score should always be in [-1, 1]."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("AMZN", days=30)
        assert -1.0 <= signal.score <= 1.0

    def test_credit_card_signal_confidence_range(self, tmp_path):
        """Confidence should always be in [0, 1]."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("AMZN", days=30)
        assert 0.0 <= signal.confidence <= 1.0

    def test_credit_card_fetch_data_keys(self, tmp_path):
        """Fetched data should have expected keys."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
            data = adapter.fetch_data("AMZN", days=30)
        for record in data:
            assert "date" in record

    def test_credit_card_different_tickers(self, tmp_path):
        """Different tickers should produce valid signals."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
            for ticker in ["AAPL", "MSFT", "GOOG"]:
                signal = adapter.calculate_signal(ticker, days=30)
                assert isinstance(signal, AlternativeDataSignal)
                assert signal.source == "credit_card"


class TestSupplyChainAdapterExtended:
    """Extended supply chain adapter tests."""

    def test_supply_chain_signal_score_range(self, tmp_path):
        """Score should always be in [-1, 1]."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("AAPL", days=30)
        assert -1.0 <= signal.score <= 1.0

    def test_supply_chain_signal_confidence_range(self, tmp_path):
        """Confidence should always be in [0, 1]."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("AAPL", days=30)
        assert 0.0 <= signal.confidence <= 1.0

    def test_supply_chain_fetch_data_keys(self, tmp_path):
        """Fetched data should have expected keys."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"
            data = adapter.fetch_data("AAPL", days=30)
        for record in data:
            assert "date" in record


class TestAlternativeDataClientExtended:
    """Extended tests for AlternativeDataClient."""

    def test_source_weights_sum_to_one(self):
        """Source weights should sum to approximately 1.0."""
        assert abs(sum(AlternativeDataClient.SOURCE_WEIGHTS.values()) - 1.0) < 0.01

    def test_source_weight_keys(self):
        """Source weight keys should match the three adapters."""
        assert set(AlternativeDataClient.SOURCE_WEIGHTS.keys()) == {
            "satellite", "credit_card", "supply_chain"
        }

    def test_credit_card_highest_weight(self):
        """Credit card should have the highest weight."""
        weights = AlternativeDataClient.SOURCE_WEIGHTS
        assert weights["credit_card"] > weights["satellite"]
        assert weights["credit_card"] > weights["supply_chain"]

    def test_composite_signal_none_sources(self, tmp_path):
        """All sources returning None should give zero composite."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            client.get_satellite_signal = lambda t, d=30: None
            client.get_credit_card_signal = lambda t, d=30: None
            client.get_supply_chain_signal = lambda t, d=30: None

            composite = client.get_composite_signal("SPY")
        assert composite.composite_score == 0.0
        assert composite.composite_confidence == 0.0
        assert composite.signal_agreement == "insufficient_data"

    def test_composite_mixed_agreement(self, tmp_path):
        """Weak conflicting signals should give mixed agreement."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            client.get_satellite_signal = lambda t, d=30: AlternativeDataSignal(
                ticker="SPY", source="satellite", signal_type="momentum",
                score=0.15, confidence=0.6, raw_value=5.0, raw_unit="pct",
                period_days=30, z_score=0.3, percentile=60.0,
                trend_direction="stable", data_timestamp="2026-01-01",
            )
            client.get_credit_card_signal = lambda t, d=30: AlternativeDataSignal(
                ticker="SPY", source="credit_card", signal_type="spending",
                score=0.1, confidence=0.5, raw_value=3.0, raw_unit="pct",
                period_days=30, z_score=0.2, percentile=55.0,
                trend_direction="stable", data_timestamp="2026-01-01",
            )
            client.get_supply_chain_signal = lambda t, d=30: AlternativeDataSignal(
                ticker="SPY", source="supply_chain", signal_type="efficiency",
                score=-0.1, confidence=0.5, raw_value=-2.0, raw_unit="pct",
                period_days=30, z_score=-0.1, percentile=45.0,
                trend_direction="stable", data_timestamp="2026-01-01",
            )
            composite = client.get_composite_signal("SPY")
        assert composite.signal_agreement == "mixed"

    def test_earnings_prediction_directions(self, tmp_path):
        """All three earnings directions should be achievable."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()

            # Beat
            beat_composite = CompositeSignal(
                ticker="SPY", composite_score=0.6, composite_confidence=0.7,
                satellite_score=0.6, credit_card_score=0.7, supply_chain_score=0.5,
                primary_driver="credit_card", signal_agreement="aligned",
            )
            client.get_composite_signal = lambda t, d=30: beat_composite
            pred = client.get_earnings_prediction("SPY", "Q1-2026")
            assert pred is not None
            assert pred.revenue_direction == "beat"

            # Inline
            inline_composite = CompositeSignal(
                ticker="SPY", composite_score=0.0, composite_confidence=0.6,
                satellite_score=0.05, credit_card_score=0.1,
                primary_driver="credit_card", signal_agreement="neutral",
            )
            client.get_composite_signal = lambda t, d=30: inline_composite
            pred = client.get_earnings_prediction("SPY", "Q2-2026")
            assert pred is not None
            assert pred.revenue_direction == "inline"

            # Miss
            miss_composite = CompositeSignal(
                ticker="SPY", composite_score=-0.5, composite_confidence=0.7,
                satellite_score=-0.5, credit_card_score=-0.6, supply_chain_score=-0.3,
                primary_driver="credit_card", signal_agreement="aligned",
            )
            client.get_composite_signal = lambda t, d=30: miss_composite
            pred = client.get_earnings_prediction("SPY", "Q3-2026")
            assert pred is not None
            assert pred.revenue_direction == "miss"

    def test_batch_signals_empty_list(self, tmp_path):
        """Batch with empty list should return empty dict."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            results = client.get_batch_signals([], days=30)
        assert results == {}

    def test_earnings_prediction_primary_signals(self, tmp_path):
        """Primary signals should list available data sources."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            composite = CompositeSignal(
                ticker="SPY", composite_score=0.4, composite_confidence=0.6,
                satellite_score=0.4, credit_card_score=0.5, supply_chain_score=0.3,
                primary_driver="credit_card", signal_agreement="aligned",
            )
            client.get_composite_signal = lambda t, d=30: composite
            pred = client.get_earnings_prediction("SPY", "Q1-2026")
        assert "satellite" in pred.primary_signals
        assert "credit_card" in pred.primary_signals
        assert "supply_chain" in pred.primary_signals

    def test_earnings_confidence_below_threshold(self, tmp_path):
        """Confidence below 0.4 should return None."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            composite = CompositeSignal(
                ticker="SPY", composite_score=0.5, composite_confidence=0.3,
                primary_driver="satellite",
            )
            client.get_composite_signal = lambda t, d=30: composite
            pred = client.get_earnings_prediction("SPY", "Q1-2026")
        assert pred is None

    def test_composite_score_rounding(self, tmp_path):
        """Composite score should be rounded to 3 decimal places."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            client.get_satellite_signal = lambda t, d=30: AlternativeDataSignal(
                ticker="SPY", source="satellite", signal_type="momentum",
                score=0.333, confidence=0.7, raw_value=5.0, raw_unit="pct",
                period_days=30, z_score=0.5, percentile=65.0,
                trend_direction="improving", data_timestamp="2026-01-01",
            )
            client.get_credit_card_signal = lambda t, d=30: AlternativeDataSignal(
                ticker="SPY", source="credit_card", signal_type="spending",
                score=0.667, confidence=0.8, raw_value=8.0, raw_unit="pct",
                period_days=30, z_score=1.2, percentile=85.0,
                trend_direction="improving", data_timestamp="2026-01-01",
            )
            client.get_supply_chain_signal = lambda t, d=30: None
            composite = client.get_composite_signal("SPY")
        # Score should be rounded to 3 places
        assert len(str(composite.composite_score).split(".")[-1]) <= 3 or composite.composite_score == 0.0


class TestConstantsExtended:
    """Validate module constants."""

    def test_alt_data_db_is_path(self):
        from src.data.alternative_data import ALT_DATA_DB
        assert isinstance(ALT_DATA_DB, Path)

    def test_satellite_adapter_class(self):
        from src.data.alternative_data import SatelliteDataAdapter
        assert issubclass(SatelliteDataAdapter, object)

    def test_credit_card_adapter_class(self):
        from src.data.alternative_data import CreditCardAdapter
        assert issubclass(CreditCardAdapter, object)

    def test_supply_chain_adapter_class(self):
        from src.data.alternative_data import SupplyChainAdapter
        assert issubclass(SupplyChainAdapter, object)

    def test_client_class(self):
        from src.data.alternative_data import AlternativeDataClient
        assert hasattr(AlternativeDataClient, "SOURCE_WEIGHTS")


# ---------------------------------------------------------------------------
# Adapter init with custom db_path
# ---------------------------------------------------------------------------

class TestAdapterInitCustom:
    """Test adapters can be initialized with custom db paths."""

    def test_satellite_custom_db_path(self, tmp_path):
        db = tmp_path / "custom.db"
        with patch("src.data.alternative_data.ALT_DATA_DB", db):
            adapter = SatelliteDataAdapter()
            adapter.db_path = db
        assert adapter.source_name == "satellite"
        assert adapter.db_path == db

    def test_credit_card_custom_db_path(self, tmp_path):
        db = tmp_path / "custom.db"
        with patch("src.data.alternative_data.ALT_DATA_DB", db):
            adapter = CreditCardAdapter()
            adapter.db_path = db
        assert adapter.source_name == "credit_card"
        assert adapter.db_path == db

    def test_supply_chain_custom_db_path(self, tmp_path):
        db = tmp_path / "custom.db"
        with patch("src.data.alternative_data.ALT_DATA_DB", db):
            adapter = SupplyChainAdapter()
            adapter.db_path = db
        assert adapter.source_name == "supply_chain"
        assert adapter.db_path == db


# ---------------------------------------------------------------------------
# Non-member ticker handling (tickers not in the adapter's known set)
# ---------------------------------------------------------------------------

class TestNonMemberTicker:
    """Test behavior when ticker is not in the adapter's known set."""

    def test_satellite_non_retail_returns_empty(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
            data = adapter.fetch_data("MSFT", days=30)
        assert data == []

    def test_credit_card_non_consumer_returns_empty(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
            data = adapter.fetch_data("JPM", days=30)
        assert data == []

    def test_supply_chain_non_supply_returns_empty(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"
            data = adapter.fetch_data("JPM", days=30)
        assert data == []


class TestNonMemberSignal:
    """Signal for a non-member ticker should return zero-confidence signal."""

    def test_satellite_non_retail_signal_zero_confidence(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("MSFT", days=30)
        assert signal.score == 0.0
        assert signal.confidence == 0.0
        assert signal.trend_direction == "insufficient_data"

    def test_credit_card_non_consumer_signal_zero_confidence(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("JPM", days=30)
        assert signal.score == 0.0
        assert signal.confidence == 0.0
        assert signal.trend_direction == "insufficient_data"

    def test_supply_chain_non_supply_signal_zero_confidence(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("JPM", days=30)
        assert signal.score == 0.0
        assert signal.confidence == 0.0
        assert signal.trend_direction == "insufficient_data"


# ---------------------------------------------------------------------------
# Database round-trip tests
# ---------------------------------------------------------------------------

class TestDatabaseRoundTrip:
    """Store data rows and read them back for each table."""

    def test_satellite_store_and_read(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"

            rows = [("AAPL", "2026-01-15", 80.0, 5.0, 2500, 0.85, "synthetic")]
            adapter._store_data(rows)

            with sqlite3.connect(str(tmp_path / "alt.db")) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ticker, date, parking_occupancy_pct FROM satellite_data WHERE ticker = ?", ("AAPL",))
                result = cursor.fetchone()

        assert result is not None
        assert result[0] == "AAPL"
        assert result[1] == "2026-01-15"
        assert result[2] == 80.0

    def test_credit_card_store_and_read(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"

            rows = [("AMZN", "2026-01-15", 12.0, 1.0, 105.0, 52.0, 75.0, 0.82, "synthetic")]
            adapter._store_data(rows)

            with sqlite3.connect(str(tmp_path / "alt.db")) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ticker, spending_growth_yoy FROM credit_card_data WHERE ticker = ?", ("AMZN",))
                result = cursor.fetchone()

        assert result is not None
        assert result[0] == "AMZN"
        assert result[1] == 12.0

    def test_supply_chain_store_and_read(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"

            rows = [("NKE", "2026-01-15", 105.0, 42.0, 22.0, 145.0, 0.75, "synthetic")]
            adapter._store_data(rows)

            with sqlite3.connect(str(tmp_path / "alt.db")) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ticker, container_throughput_index FROM supply_chain_data WHERE ticker = ?", ("NKE",))
                result = cursor.fetchone()

        assert result is not None
        assert result[0] == "NKE"
        assert result[1] == 105.0


# ---------------------------------------------------------------------------
# Signal calculation edge cases
# ---------------------------------------------------------------------------

class TestSignalCalcEdgeCases:
    """Edge cases in calculate_signal for each adapter."""

    def test_satellite_calculate_signal_minimal_days(self, tmp_path):
        """Very short days parameter should still produce a signal."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("WMT", days=1)
        assert isinstance(signal, AlternativeDataSignal)
        assert -1.0 <= signal.score <= 1.0
        assert signal.period_days == 1

    def test_credit_card_signal_large_days(self, tmp_path):
        """Large days parameter should still produce valid signal."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("AMZN", days=200)
        assert isinstance(signal, AlternativeDataSignal)
        assert signal.period_days == 200

    def test_satellite_signal_default_days(self, tmp_path):
        """Default days parameter (30) should work."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("COST")
        assert signal.period_days == 30

    def test_supply_chain_raw_unit_is_composite_index(self, tmp_path):
        """Supply chain signal raw_unit should be composite_index."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = adapter.calculate_signal("AAPL", days=30)
        assert signal.raw_unit == "composite_index"


# ---------------------------------------------------------------------------
# CLI main function tests
# ---------------------------------------------------------------------------

class TestCLIMain:
    """Test the CLI main() argument parsing and dispatching."""

    def test_no_args_prints_usage(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        from src.data.alternative_data import main as cli_main
        with patch.object(sys, "argv", ["alternative_data.py"]):
            with pytest.raises(SystemExit):
                cli_main()
        assert "Usage:" in caplog.text

    def test_unknown_command_exits(self):
        from src.data.alternative_data import main as cli_main
        with patch("sys.argv", ["alternative_data.py", "unknown_cmd"]):
            with patch("sys.stdout", new_callable=io.StringIO):
                with pytest.raises(SystemExit):
                    cli_main()

    def test_fetch_missing_ticker_exits(self):
        from src.data.alternative_data import main as cli_main
        with patch("sys.argv", ["alternative_data.py", "fetch", "--source", "satellite"]):
            with patch("sys.stdout", new_callable=io.StringIO):
                with pytest.raises(SystemExit):
                    cli_main()

    def test_earnings_missing_quarter_exits(self):
        from src.data.alternative_data import main as cli_main
        with patch("sys.argv", ["alternative_data.py", "earnings", "--ticker", "AAPL"]):
            with patch("sys.stdout", new_callable=io.StringIO):
                with pytest.raises(SystemExit):
                    cli_main()

    def test_earnings_missing_ticker_exits(self):
        from src.data.alternative_data import main as cli_main
        with patch("sys.argv", ["alternative_data.py", "earnings", "--quarter", "Q4-2025"]):
            with patch("sys.stdout", new_callable=io.StringIO):
                with pytest.raises(SystemExit):
                    cli_main()

    def test_batch_missing_tickers_exits(self):
        from src.data.alternative_data import main as cli_main
        with patch("sys.argv", ["alternative_data.py", "batch"]):
            with patch("sys.stdout", new_callable=io.StringIO):
                with pytest.raises(SystemExit):
                    cli_main()

    def test_fetch_satellite_calls_adapter(self, tmp_path):
        from src.data.alternative_data import main as cli_main
        db = tmp_path / "alt.db"
        with patch("src.data.alternative_data.ALT_DATA_DB", db):
            with patch("sys.argv", ["alternative_data.py", "fetch", "--ticker", "AAPL", "--source", "satellite"]):
                with patch("sys.stdout", new_callable=io.StringIO):
                    cli_main()

    def test_composite_command_runs(self, tmp_path):
        from src.data.alternative_data import main as cli_main
        db = tmp_path / "alt.db"
        with patch("src.data.alternative_data.ALT_DATA_DB", db):
            with patch("sys.argv", ["alternative_data.py", "composite", "--ticker", "AAPL"]):
                with patch("sys.stdout", new_callable=io.StringIO):
                    cli_main()


# ---------------------------------------------------------------------------
# Client delegation tests
# ---------------------------------------------------------------------------

class TestClientDelegation:
    """Test that AlternativeDataClient helper methods delegate to adapters."""

    def test_get_satellite_signal_delegates(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            with patch.object(client.satellite, "calculate_signal", return_value=None) as mock_method:
                result = client.get_satellite_signal("AAPL", days=45)
        mock_method.assert_called_once_with("AAPL", 45)
        assert result is None

    def test_get_credit_card_signal_delegates(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            with patch.object(client.credit_card, "calculate_signal", return_value=None) as mock_method:
                result = client.get_credit_card_signal("AMZN", days=60)
        mock_method.assert_called_once_with("AMZN", 60)
        assert result is None

    def test_get_supply_chain_signal_delegates(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            with patch.object(client.supply_chain, "calculate_signal", return_value=None) as mock_method:
                result = client.get_supply_chain_signal("NKE", days=90)
        mock_method.assert_called_once_with("NKE", 90)
        assert result is None


# ---------------------------------------------------------------------------
# Earnings prediction boundary conditions
# ---------------------------------------------------------------------------

class TestEarningsPredictionEdgeCases:
    """Test boundary conditions in get_earnings_prediction mapping."""

    def test_score_just_above_05_predicts_beat(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            composite = CompositeSignal(
                ticker="SPY", composite_score=0.51, composite_confidence=0.7,
                satellite_score=0.5, credit_card_score=0.6, supply_chain_score=0.4,
                primary_driver="credit_card", signal_agreement="aligned",
            )
            client.get_composite_signal = lambda t, d=30: composite
            pred = client.get_earnings_prediction("SPY", "Q1-2026")
        assert pred is not None
        assert pred.revenue_direction == "beat"
        assert pred.predicted_revenue_growth > 10.0

    def test_score_just_below_neg_02_predicts_miss(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            composite = CompositeSignal(
                ticker="SPY", composite_score=-0.21, composite_confidence=0.7,
                satellite_score=-0.3, credit_card_score=-0.4,
                primary_driver="credit_card", signal_agreement="aligned",
            )
            client.get_composite_signal = lambda t, d=30: composite
            pred = client.get_earnings_prediction("SPY", "Q1-2026")
        assert pred is not None
        assert pred.revenue_direction == "miss"
        assert pred.predicted_revenue_growth < 0

    def test_exact_zero_score_inline(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            composite = CompositeSignal(
                ticker="SPY", composite_score=0.0, composite_confidence=0.6,
                satellite_score=0.0, credit_card_score=0.0,
                primary_driver="credit_card", signal_agreement="neutral",
            )
            client.get_composite_signal = lambda t, d=30: composite
            pred = client.get_earnings_prediction("SPY", "Q1-2026")
        assert pred is not None
        assert pred.revenue_direction == "inline"
        assert pred.predicted_revenue_growth == 0.0

    def test_score_at_02_boundary_beat(self, tmp_path):
        """Score exactly 0.2 falls into the > -0.2 && <= 0.2 bucket -> inline."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            composite = CompositeSignal(
                ticker="SPY", composite_score=0.2, composite_confidence=0.6,
                satellite_score=0.2, credit_card_score=0.3,
                primary_driver="credit_card", signal_agreement="aligned",
            )
            client.get_composite_signal = lambda t, d=30: composite
            pred = client.get_earnings_prediction("SPY", "Q1-2026")
        assert pred is not None
        # score == 0.2 fails the > 0.2 check, falls to elif > -0.2 -> inline
        assert pred.revenue_direction == "inline"
        assert pred.predicted_revenue_growth == 3.0

    def test_all_signals_none_in_prediction(self, tmp_path):
        """When composite has no per-source scores, primary_signals should be empty."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            composite = CompositeSignal(
                ticker="SPY", composite_score=0.3, composite_confidence=0.6,
                primary_driver="satellite", signal_agreement="insufficient_data",
            )
            client.get_composite_signal = lambda t, d=30: composite
            pred = client.get_earnings_prediction("SPY", "Q1-2026")
        assert pred is not None
        assert pred.primary_signals == []


# ---------------------------------------------------------------------------
# Database resilience tests
# ---------------------------------------------------------------------------

class TestDatabaseResilience:
    """Test database creation and resilience under various conditions."""

    def test_init_database_creates_data_dir(self, tmp_path):
        """DATA_DIR should be created if it doesn't exist."""
        new_dir = tmp_path / "nonexistent" / "deep" / "dir"
        with patch("src.data.alternative_data.DATA_DIR", new_dir):
            with patch("src.data.alternative_data.ALT_DATA_DB", new_dir / "alt.db"):
                init_database()
        assert new_dir.exists()
        assert (new_dir / "alt.db").exists()

    def test_double_init_preserves_inserted_data(self, tmp_path):
        """Calling init_database twice should not wipe existing data."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()

            # Insert data directly
            with sqlite3.connect(str(tmp_path / "alt.db")) as conn:
                conn.execute("""
                    INSERT INTO alt_data_signals
                    (ticker, source, signal_type, score, confidence, raw_value,
                     period_days, z_score, percentile, trend_direction, data_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, ("TEST", "satellite", "test", 0.5, 0.8, 10.0, 30, 1.0, 75.0, "improving", "2026-01-01"))

            # Re-init
            init_database()

            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM alt_data_signals WHERE ticker = ?", ("TEST",))
            count = cursor.fetchone()[0]

        assert count == 1

    def test_unique_constraint_on_ticker_date(self, tmp_path):
        """satellite_data UNIQUE(ticker, date) should prevent duplicate rows."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"

            rows = [
                ("AAPL", "2026-01-15", 80.0, 5.0, 2500, 0.85, "test"),
                ("AAPL", "2026-01-15", 85.0, 6.0, 2600, 0.90, "test"),  # same ticker+date
            ]
            adapter._store_data(rows)

            with sqlite3.connect(str(tmp_path / "alt.db")) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM satellite_data WHERE ticker = ? AND date = ?", ("AAPL", "2026-01-15"))
                count = cursor.fetchone()[0]

        # INSERT OR REPLACE -> second row replaces first, so count should be 1
        assert count == 1


# ---------------------------------------------------------------------------
# Signal store round-trip tests
# ---------------------------------------------------------------------------

class TestSignalStoreRoundTrip:
    """Test that _store_signal persists signals correctly for each adapter."""

    def _make_signal(self, ticker: str, source: str, score: float = 0.5) -> AlternativeDataSignal:
        return AlternativeDataSignal(
            ticker=ticker, source=source, signal_type="test",
            score=score, confidence=0.7, raw_value=10.0, raw_unit="pct",
            period_days=30, z_score=1.0, percentile=75.0,
            trend_direction="stable", data_timestamp="2026-01-01",
        )

    def test_satellite_store_signal(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = self._make_signal("AAPL", "satellite")
            adapter._store_signal(signal)

            with sqlite3.connect(str(tmp_path / "alt.db")) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ticker, score, confidence FROM alt_data_signals WHERE ticker = ?", ("AAPL",))
                row = cursor.fetchone()
        assert row is not None
        assert row[0] == "AAPL"
        assert row[1] == 0.5
        assert row[2] == 0.7

    def test_credit_card_store_signal(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = self._make_signal("AMZN", "credit_card", score=-0.3)
            adapter._store_signal(signal)

            with sqlite3.connect(str(tmp_path / "alt.db")) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ticker, score FROM alt_data_signals WHERE ticker = ?", ("AMZN",))
                row = cursor.fetchone()
        assert row is not None
        assert row[0] == "AMZN"
        assert row[1] == -0.3

    def test_supply_chain_store_signal(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"
            signal = self._make_signal("NKE", "supply_chain", score=0.8)
            adapter._store_signal(signal)

            with sqlite3.connect(str(tmp_path / "alt.db")) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ticker, score FROM alt_data_signals WHERE ticker = ?", ("NKE",))
                row = cursor.fetchone()
        assert row is not None
        assert row[0] == "NKE"
        assert row[1] == 0.8


# ---------------------------------------------------------------------------
# Batch signals edge cases
# ---------------------------------------------------------------------------

class TestBatchSignalsExtended:
    """Additional batch signal tests."""

    def test_batch_includes_non_member_tickers(self, tmp_path):
        """Batch with mixed member and non-member tickers should return all."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            results = client.get_batch_signals(["AAPL", "JPM"], days=30)

        assert "AAPL" in results
        assert "JPM" in results
        # JPM is not in any ticker set, so composite confidence should be 0.0
        assert results["JPM"].composite_confidence == 0.0

    def test_batch_single_ticker(self, tmp_path):
        """Batch with a single ticker should return one result."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            results = client.get_batch_signals(["WMT"], days=30)
        assert len(results) == 1
        assert "WMT" in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ---------------------------------------------------------------------------
# Database table structure and column verification
# ---------------------------------------------------------------------------

class TestDatabaseTableStructure:
    """Verify all five tables have the expected column schemas."""

    def test_satellite_data_columns(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
        conn = sqlite3.connect(str(tmp_path / "alt.db"))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(satellite_data)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()
        expected = {"id", "ticker", "date", "parking_occupancy_pct",
                     "occupancy_vs_last_year_pct", "store_count",
                     "data_quality_score", "source", "created_at"}
        assert expected.issubset(cols)

    def test_credit_card_data_columns(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
        conn = sqlite3.connect(str(tmp_path / "alt.db"))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(credit_card_data)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()
        expected = {"id", "ticker", "date", "spending_growth_yoy",
                     "spending_growth_mom", "transaction_volume_index",
                     "avg_ticket_size", "category_rank_pct",
                     "data_quality_score", "source", "created_at"}
        assert expected.issubset(cols)

    def test_supply_chain_data_columns(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
        conn = sqlite3.connect(str(tmp_path / "alt.db"))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(supply_chain_data)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()
        expected = {"id", "ticker", "date", "container_throughput_index",
                     "inventory_days_coverage", "supplier_lead_time_days",
                     "shipping_cost_index", "data_quality_score",
                     "source", "created_at"}
        assert expected.issubset(cols)

    def test_alt_data_signals_columns(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
        conn = sqlite3.connect(str(tmp_path / "alt.db"))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(alt_data_signals)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()
        expected = {"id", "ticker", "source", "signal_type", "score",
                     "confidence", "raw_value", "period_days", "z_score",
                     "percentile", "trend_direction", "data_timestamp",
                     "signal_generated"}
        assert expected.issubset(cols)

    def test_prediction_accuracy_columns(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
        conn = sqlite3.connect(str(tmp_path / "alt.db"))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(prediction_accuracy)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()
        expected = {"id", "ticker", "quarter", "prediction_date",
                     "predicted_revenue_growth", "actual_revenue_growth",
                     "prediction_error", "primary_signals",
                     "accuracy_score", "recorded_at"}
        assert expected.issubset(cols)


# ---------------------------------------------------------------------------
# AlternativeDataSignal boundary and validation tests
# ---------------------------------------------------------------------------

class TestAlternativeDataSignalBoundaries:
    """Boundary-condition tests on AlternativeDataSignal fields."""

    def test_model_version_custom(self):
        sig = AlternativeDataSignal(
            ticker="SPY", source="satellite", signal_type="momentum",
            score=0.0, confidence=0.0, raw_value=0.0, raw_unit="pct",
            period_days=30, z_score=0.0, percentile=50.0,
            trend_direction="stable", data_timestamp="2026-01-01",
            model_version="v3.0.0",
        )
        assert sig.model_version == "v3.0.0"

    def test_signal_generated_dynamic(self):
        sig = AlternativeDataSignal(
            ticker="SPY", source="satellite", signal_type="momentum",
            score=0.0, confidence=0.0, raw_value=0.0, raw_unit="pct",
            period_days=30, z_score=0.0, percentile=50.0,
            trend_direction="stable", data_timestamp="2026-01-01",
        )
        assert sig.signal_generated is not None
        assert "T" in sig.signal_generated  # isoformat includes T

    def test_extreme_negative_z_score(self):
        sig = AlternativeDataSignal(
            ticker="SPY", source="satellite", signal_type="momentum",
            score=-0.8, confidence=0.6, raw_value=-30.0, raw_unit="pct",
            period_days=30, z_score=-3.0, percentile=0.1,
            trend_direction="deteriorating", data_timestamp="2026-01-01",
        )
        assert sig.z_score == -3.0

    def test_percentile_extremes(self):
        sig_0 = AlternativeDataSignal(
            ticker="A", source="satellite", signal_type="m",
            score=0.0, confidence=0.0, raw_value=0.0, raw_unit="pct",
            period_days=30, z_score=0.0, percentile=0.0,
            trend_direction="stable", data_timestamp="2026-01-01",
        )
        sig_100 = AlternativeDataSignal(
            ticker="B", source="credit_card", signal_type="m",
            score=0.0, confidence=0.0, raw_value=0.0, raw_unit="pct",
            period_days=30, z_score=0.0, percentile=100.0,
            trend_direction="stable", data_timestamp="2026-01-01",
        )
        assert sig_0.percentile == 0.0
        assert sig_100.percentile == 100.0

    def test_zero_confidence_and_score(self):
        sig = AlternativeDataSignal(
            ticker="SPY", source="satellite", signal_type="momentum",
            score=0.0, confidence=0.0, raw_value=0.0, raw_unit="pct",
            period_days=30, z_score=0.0, percentile=50.0,
            trend_direction="stable", data_timestamp="2026-01-01",
        )
        assert sig.score == 0.0
        assert sig.confidence == 0.0


# ---------------------------------------------------------------------------
# CompositeSignal edge-case tests
# ---------------------------------------------------------------------------

class TestCompositeSignalEdgeCases:
    """Edge-case and boundary tests for CompositeSignal."""

    def test_historical_accuracy_present(self):
        cs = CompositeSignal(
            ticker="SPY", satellite_score=0.5, credit_card_score=0.6,
            supply_chain_score=0.4, composite_score=0.52,
            composite_confidence=0.7, primary_driver="credit_card",
            signal_agreement="aligned", historical_accuracy=0.85,
        )
        assert cs.historical_accuracy == 0.85
        d = cs.to_dict()
        assert d["historical_accuracy"] == 0.85

    def test_negative_composite_score(self):
        cs = CompositeSignal(
            ticker="SPY", satellite_score=-0.5, credit_card_score=-0.6,
            supply_chain_score=-0.3, composite_score=-0.47,
            composite_confidence=0.6, primary_driver="satellite",
            signal_agreement="conflicting",
        )
        assert cs.composite_score == -0.47
        assert cs.primary_driver == "satellite"

    def test_some_scores_none(self):
        cs = CompositeSignal(
            ticker="SPY", satellite_score=0.4, credit_card_score=None,
            supply_chain_score=None, composite_score=0.14,
            composite_confidence=0.35, primary_driver="satellite",
            signal_agreement="insufficient_data",
        )
        assert cs.satellite_score == 0.4
        assert cs.credit_card_score is None
        assert cs.supply_chain_score is None
        d = cs.to_dict()
        assert d["satellite_score"] == 0.4
        assert d["credit_card_score"] is None

    def test_timestamp_dynamic_default(self):
        cs = CompositeSignal(ticker="TLT")
        assert cs.timestamp is not None
        assert "T" in cs.timestamp


# ---------------------------------------------------------------------------
# EarningsPrediction boundary tests
# ---------------------------------------------------------------------------

class TestEarningsPredictionBoundaries:
    """Boundary-condition tests for earnings prediction mapping."""

    def test_score_just_above_02_beat(self, tmp_path):
        """Score just above 0.2 should map to beat (mild) branch."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            composite = CompositeSignal(
                ticker="SPY", composite_score=0.21, composite_confidence=0.6,
                satellite_score=0.3, credit_card_score=0.4,
                primary_driver="credit_card", signal_agreement="aligned",
            )
            client.get_composite_signal = lambda t, d=30: composite
            pred = client.get_earnings_prediction("SPY", "Q1-2026")
        assert pred is not None
        assert pred.revenue_direction == "beat"
        assert 5.0 < pred.predicted_revenue_growth < 10.0

    def test_score_just_below_neg_02_inline(self, tmp_path):
        """Score very close to 0 from below should map to inline."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            composite = CompositeSignal(
                ticker="SPY", composite_score=-0.19, composite_confidence=0.6,
                credit_card_score=-0.2,
                primary_driver="credit_card", signal_agreement="insufficient_data",
            )
            client.get_composite_signal = lambda t, d=30: composite
            pred = client.get_earnings_prediction("SPY", "Q1-2026")
        assert pred is not None
        assert pred.revenue_direction == "inline"

    def test_score_strong_bearish_large_negative(self, tmp_path):
        """Very large negative score should clamp miss direction."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            composite = CompositeSignal(
                ticker="SPY", composite_score=-0.9, composite_confidence=0.7,
                satellite_score=-0.8, credit_card_score=-0.9,
                primary_driver="credit_card", signal_agreement="aligned",
            )
            client.get_composite_signal = lambda t, d=30: composite
            pred = client.get_earnings_prediction("SPY", "Q1-2026")
        assert pred is not None
        assert pred.revenue_direction == "miss"
        assert pred.confidence > 0

    def test_only_satellite_signal_in_prediction(self, tmp_path):
        """Only satellite available among primary_signals."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            composite = CompositeSignal(
                ticker="SPY", composite_score=0.4, composite_confidence=0.6,
                satellite_score=0.5,
                primary_driver="satellite", signal_agreement="insufficient_data",
            )
            client.get_composite_signal = lambda t, d=30: composite
            pred = client.get_earnings_prediction("SPY", "Q1-2026")
        assert pred is not None
        assert pred.primary_signals == ["satellite"]

    def test_only_supply_chain_signal_in_prediction(self, tmp_path):
        """Only supply_chain available among primary_signals."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            composite = CompositeSignal(
                ticker="SPY", composite_score=0.3, composite_confidence=0.6,
                supply_chain_score=0.4,
                primary_driver="supply_chain", signal_agreement="insufficient_data",
            )
            client.get_composite_signal = lambda t, d=30: composite
            pred = client.get_earnings_prediction("SPY", "Q1-2026")
        assert pred is not None
        assert pred.primary_signals == ["supply_chain"]


# ---------------------------------------------------------------------------
# Satellite score mapping branch tests (controlled data via fetch_data mock)
# ---------------------------------------------------------------------------

class TestSatelliteScoreMappingBranches:
    """Exercise each branch of satellite score mapping in calculate_signal."""

    def _make_adapter(self, tmp_path) -> SatelliteDataAdapter:
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
        return adapter

    def _mock_fetch_data(self, adapter, occupancy_yoy_values: list[float]):
        """Replace fetch_data with controlled data."""
        data = []
        for i, yoy in enumerate(occupancy_yoy_values):
            data.append({
                "ticker": "WMT",
                "date": f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                "parking_occupancy_pct": 75.0,
                "occupancy_vs_last_year_pct": yoy,
                "store_count": 2500,
                "data_quality_score": 0.85,
                "source": "test",
            })
        adapter.fetch_data = lambda ticker, days=90: data

    def test_current_avg_gt_10_bullish(self, tmp_path):
        """current_avg > 10 triggers strong bullish branch."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
        yoy_values = [15.0] * 60  # 60 values, all > 10
        self._mock_fetch_data(adapter, yoy_values)
        signal = adapter.calculate_signal("WMT", days=30)
        assert signal.score > 0.5, f"Expected bullish score, got {signal.score}"
        assert signal.trend_direction in ("improving", "stable")

    def test_current_avg_gt_5_mildly_bullish(self, tmp_path):
        """current_avg between 5 and 10 triggers moderately bullish branch."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
        yoy_values = [7.0] * 60
        self._mock_fetch_data(adapter, yoy_values)
        signal = adapter.calculate_signal("WMT", days=30)
        assert 0.3 <= signal.score <= 0.6, f"Expected moderate score, got {signal.score}"

    def test_current_avg_gt_0_neutral(self, tmp_path):
        """current_avg between 0 and 5 triggers neutral branch."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
        yoy_values = [2.5] * 60
        self._mock_fetch_data(adapter, yoy_values)
        signal = adapter.calculate_signal("WMT", days=30)
        assert 0 < signal.score < 0.3, f"Expected low positive score, got {signal.score}"

    def test_current_avg_gt_neg_5_mildly_bearish(self, tmp_path):
        """current_avg between -5 and 0 triggers slightly bearish branch."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
        yoy_values = [-2.0] * 60
        self._mock_fetch_data(adapter, yoy_values)
        signal = adapter.calculate_signal("WMT", days=30)
        assert -0.5 < signal.score < 0, f"Expected slightly negative score, got {signal.score}"

    def test_current_avg_le_neg_5_bearish(self, tmp_path):
        """current_avg <= -5 triggers bearish branch."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
        yoy_values = [-8.0] * 60
        self._mock_fetch_data(adapter, yoy_values)
        signal = adapter.calculate_signal("WMT", days=30)
        assert signal.score < -0.3, f"Expected bearish score, got {signal.score}"


# ---------------------------------------------------------------------------
# Credit card score mapping branch tests
# ---------------------------------------------------------------------------

class TestCreditCardScoreMappingBranches:
    """Exercise each branch of credit card score mapping."""

    def _mock_fetch_data(self, adapter, yoy_values: list[float]):
        data = []
        for i, yoy in enumerate(yoy_values):
            data.append({
                "ticker": "AMZN",
                "date": f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                "spending_growth_yoy": yoy,
                "spending_growth_mom": yoy / 12,
                "transaction_volume_index": 105.0,
                "avg_ticket_size": 52.0,
                "category_rank_pct": 75.0,
                "data_quality_score": 0.82,
                "source": "test",
            })
        adapter.fetch_data = lambda ticker, days=90: data

    def test_yoy_gt_15_very_bullish(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
        self._mock_fetch_data(adapter, [18.0] * 60)
        signal = adapter.calculate_signal("AMZN", days=30)
        assert signal.score > 0.5, f"Expected very bullish, got {signal.score}"

    def test_yoy_gt_10_bullish(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
        self._mock_fetch_data(adapter, [12.0] * 60)
        signal = adapter.calculate_signal("AMZN", days=30)
        assert 0.3 <= signal.score <= 0.7, f"Expected bullish, got {signal.score}"

    def test_yoy_gt_5_mildly_bullish(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
        self._mock_fetch_data(adapter, [7.0] * 60)
        signal = adapter.calculate_signal("AMZN", days=30)
        assert 0.1 < signal.score < 0.5, f"Expected mildly bullish, got {signal.score}"

    def test_yoy_gt_0_neutral(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
        self._mock_fetch_data(adapter, [2.0] * 60)
        signal = adapter.calculate_signal("AMZN", days=30)
        assert 0 < signal.score < 0.3, f"Expected neutral, got {signal.score}"

    def test_yoy_le_0_bearish(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
        self._mock_fetch_data(adapter, [-3.0] * 60)
        signal = adapter.calculate_signal("AMZN", days=30)
        assert signal.score < 0, f"Expected bearish, got {signal.score}"


# ---------------------------------------------------------------------------
# Supply chain score mapping branch tests
# ---------------------------------------------------------------------------

class TestSupplyChainScoreMappingBranches:
    """Exercise the supply chain composite score paths."""

    def _mock_fetch_data(self, adapter, throughput_recent: float, inventory_recent: float,
                         lead_time_recent: float, throughput_hist: float = 100.0,
                         inventory_hist: float = 45.0, lead_time_hist: float = 21.0,
                         recent_count: int = 30, hist_count: int = 30):
        """Supply different values for recent vs historical periods to get non-zero z-score.
        Recent entries come FIRST (indices 0..recent_count-1) because calculate_signal
        uses data[:days] for recent and data[days:90] for historical.
        Values are slightly jittered so within-group stddev > 0."""
        import random
        data = []
        # Recent entries first (used as data[:days] in calculate_signal)
        for i in range(recent_count):
            jitter = random.uniform(-0.5, 0.5)
            data.append({
                "ticker": "AAPL",
                "date": f"2026-02-{(i % 28) + 1:02d}",
                "container_throughput_index": throughput_recent + jitter,
                "inventory_days_coverage": inventory_recent + jitter * 0.2,
                "supplier_lead_time_days": lead_time_recent + jitter * 0.1,
                "shipping_cost_index": 150.0,
                "data_quality_score": 0.75,
                "source": "test",
            })
        # Historical entries second (used as data[days:90])
        for i in range(hist_count):
            jitter = random.uniform(-0.5, 0.5)
            data.append({
                "ticker": "AAPL",
                "date": f"2026-01-{(i % 28) + 1:02d}",
                "container_throughput_index": throughput_hist + jitter,
                "inventory_days_coverage": inventory_hist + jitter * 0.2,
                "supplier_lead_time_days": lead_time_hist + jitter * 0.1,
                "shipping_cost_index": 150.0,
                "data_quality_score": 0.75,
                "source": "test",
            })
        adapter.fetch_data = lambda ticker, days=90: data

    def test_efficient_supply_chain_positive_score(self, tmp_path):
        """Recent efficient metrics vs historical moderate -> positive score."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"
        self._mock_fetch_data(adapter,
                              throughput_recent=120.0, inventory_recent=35.0, lead_time_recent=15.0,
                              throughput_hist=100.0, inventory_hist=45.0, lead_time_hist=21.0)
        signal = adapter.calculate_signal("AAPL", days=30)
        assert signal.score > 0, f"Expected positive score, got {signal.score}"

    def test_inefficient_supply_chain_negative_score(self, tmp_path):
        """Recent inefficient metrics vs historical moderate -> negative score."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"
        self._mock_fetch_data(adapter,
                              throughput_recent=80.0, inventory_recent=55.0, lead_time_recent=28.0,
                              throughput_hist=100.0, inventory_hist=45.0, lead_time_hist=21.0)
        signal = adapter.calculate_signal("AAPL", days=30)
        assert signal.score < 0, f"Expected negative score, got {signal.score}"

    def test_neutral_supply_chain_score(self, tmp_path):
        """Near-baseline metrics -> near-zero score."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"
        self._mock_fetch_data(adapter,
                              throughput_recent=100.0, inventory_recent=45.0, lead_time_recent=21.0,
                              throughput_hist=100.0, inventory_hist=45.0, lead_time_hist=21.0)
        signal = adapter.calculate_signal("AAPL", days=30)
        assert abs(signal.score) < 0.3, f"Expected near-zero score, got {signal.score}"

    def test_single_historical_record_no_hist_scores(self, tmp_path):
        """When historical has only 1 record, hist_scores is not defined."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"
        self._mock_fetch_data(adapter,
                              throughput_recent=100.0, inventory_recent=45.0, lead_time_recent=21.0,
                              throughput_hist=100.0, inventory_hist=45.0, lead_time_hist=21.0,
                              recent_count=30, hist_count=1)
        signal = adapter.calculate_signal("AAPL", days=30)
        assert isinstance(signal, AlternativeDataSignal)
        assert -1.0 <= signal.score <= 1.0


# ---------------------------------------------------------------------------
# Database UNIQUE constraint extended tests
# ---------------------------------------------------------------------------

class TestDatabaseUniqueConstraintsExtended:
    """Verify UNIQUE(ticker, date) on credit_card and supply_chain tables."""

    def test_credit_card_unique_ticker_date(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
            rows = [
                ("AMZN", "2026-01-15", 12.0, 1.0, 105.0, 52.0, 75.0, 0.82, "test"),
                ("AMZN", "2026-01-15", 15.0, 2.0, 110.0, 55.0, 80.0, 0.90, "test"),
            ]
            adapter._store_data(rows)
            with sqlite3.connect(str(tmp_path / "alt.db")) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM credit_card_data WHERE ticker=? AND date=?",
                    ("AMZN", "2026-01-15"),
                )
                count = cursor.fetchone()[0]
        assert count == 1  # INSERT OR REPLACE -> second replaces first

    def test_supply_chain_unique_ticker_date(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"
            rows = [
                ("NKE", "2026-02-10", 105.0, 42.0, 22.0, 145.0, 0.75, "test"),
                ("NKE", "2026-02-10", 110.0, 40.0, 20.0, 140.0, 0.80, "test"),
            ]
            adapter._store_data(rows)
            with sqlite3.connect(str(tmp_path / "alt.db")) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM supply_chain_data WHERE ticker=? AND date=?",
                    ("NKE", "2026-02-10"),
                )
                count = cursor.fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# AlternativeDataClient adapter type verification
# ---------------------------------------------------------------------------

class TestClientAdapters:
    """Verify AlternativeDataClient creates correct adapter types."""

    def test_adapter_attributes_are_correct_types(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
        assert isinstance(client.satellite, SatelliteDataAdapter)
        assert isinstance(client.credit_card, CreditCardAdapter)
        assert isinstance(client.supply_chain, SupplyChainAdapter)

    def test_client_init_calls_init_database(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            with patch("src.data.alternative_data.init_database") as mock_init:
                client = AlternativeDataClient()
        assert mock_init.call_count >= 1


# ---------------------------------------------------------------------------
# Adapter _store_data None-value edge cases
# ---------------------------------------------------------------------------

class TestAdapterDataNullFields:
    """Verify adapters handle None/NULL fields in stored data."""

    def test_satellite_null_occupancy(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
            rows = [("WMT", "2026-03-01", None, None, 2500, None, "test")]
            adapter._store_data(rows)
            data = adapter.fetch_data("WMT", days=90)
        assert len(data) >= 1
        assert data[0]["parking_occupancy_pct"] is None
        assert data[0]["occupancy_vs_last_year_pct"] is None
        assert data[0]["store_count"] == 2500

    def test_credit_card_null_growth(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
            rows = [("AMZN", "2026-03-01", None, None, None, None, None, None, "test")]
            adapter._store_data(rows)
            data = adapter.fetch_data("AMZN", days=90)
        assert len(data) >= 1
        assert data[0]["spending_growth_yoy"] is None
        assert data[0]["data_quality_score"] == 0.8

    def test_supply_chain_null_metrics(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"
            rows = [("AAPL", "2026-03-01", None, None, None, None, None, "test")]
            adapter._store_data(rows)
            data = adapter.fetch_data("AAPL", days=90)
        assert len(data) >= 1
        assert data[0]["container_throughput_index"] is None
        assert data[0]["data_quality_score"] == 0.75


# ---------------------------------------------------------------------------
# Adapter fetch_data with multiple tickers and stale data edge case
# ---------------------------------------------------------------------------

class TestAdapterFetchDataEdgeCases:
    """Edge cases for fetch_data across adapters."""

    def test_satellite_store_then_fetch_different_ticker(self, tmp_path):
        """Data stored for one ticker should not leak to another."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SatelliteDataAdapter()
            adapter.db_path = tmp_path / "alt.db"
            rows = [("WMT", "2026-04-01", 80.0, 5.0, 2500, 0.85, "test")]
            adapter._store_data(rows)
            data_other = adapter.fetch_data("TGT", days=90)
        assert len(data_other) > 0  # TGT is a retail ticker, gets synthetic data
        assert data_other[0]["ticker"] == "TGT"

    def test_credit_card_store_multiple_dates(self, tmp_path):
        """Multiple date entries for same ticker should all be retrievable."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = CreditCardAdapter()
            adapter.db_path = tmp_path / "alt.db"
            rows = [
                ("AMZN", "2026-01-01", 8.0, 0.7, 100.0, 50.0, 75.0, 0.82, "test"),
                ("AMZN", "2026-01-02", 8.5, 0.8, 101.0, 51.0, 76.0, 0.82, "test"),
                ("AMZN", "2026-01-03", 9.0, 0.9, 102.0, 52.0, 77.0, 0.82, "test"),
            ]
            adapter._store_data(rows)
            data = adapter.fetch_data("AMZN", days=90)
        assert len(data) >= 3

    def test_supply_chain_fetch_after_store_uses_cached(self, tmp_path):
        """After storing data, fetch_data should return stored (not synthetic)."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            adapter = SupplyChainAdapter()
            adapter.db_path = tmp_path / "alt.db"
            rows = [("CAT", "2026-04-15", 110.0, 40.0, 20.0, 140.0, 0.80, "manual")]
            adapter._store_data(rows)
            data = adapter.fetch_data("CAT", days=90)
        assert len(data) >= 1
        assert data[0]["source"] == "manual"
        assert data[0]["container_throughput_index"] == 110.0


# ---------------------------------------------------------------------------
# CLI additional command tests
# ---------------------------------------------------------------------------

class TestCLIAdditionalCommands:
    """Additional CLI command path coverage."""

    def test_fetch_with_source_all(self, tmp_path, caplog):
        import logging
        caplog.set_level(logging.INFO)
        from src.data.alternative_data import main as cli_main
        db = tmp_path / "alt.db"
        with patch("src.data.alternative_data.ALT_DATA_DB", db):
            with patch.object(sys, "argv", [
                "alternative_data.py", "fetch", "--ticker", "AAPL",
                "--source", "all", "--days", "10",
            ]):
                cli_main()
        assert "Satellite Signal" in caplog.text
        assert "Credit Card Signal" in caplog.text
        assert "Supply Chain Signal" in caplog.text

    def test_earnings_insufficient_data(self, tmp_path, caplog):
        import logging
        caplog.set_level(logging.INFO)
        from src.data.alternative_data import main as cli_main
        db = tmp_path / "alt.db"
        with patch("src.data.alternative_data.ALT_DATA_DB", db):
            with patch.object(sys, "argv", [
                "alternative_data.py", "earnings", "--ticker", "JPM",
                "--quarter", "Q1-2026",
            ]):
                cli_main()
        assert "Insufficient data" in caplog.text

    def test_fetch_satellite_source_specific(self, tmp_path, caplog):
        import logging
        caplog.set_level(logging.INFO)
        from src.data.alternative_data import main as cli_main
        db = tmp_path / "alt.db"
        with patch("src.data.alternative_data.ALT_DATA_DB", db):
            with patch.object(sys, "argv", [
                "alternative_data.py", "fetch", "--ticker", "AAPL",
                "--source", "satellite", "--days", "5",
            ]):
                cli_main()
        assert "Satellite Signal" in caplog.text
        assert "Credit Card Signal" not in caplog.text
        assert "Supply Chain Signal" not in caplog.text


# ---------------------------------------------------------------------------
# Composite signal with confidence filtering edge cases
# ---------------------------------------------------------------------------

class TestCompositeSignalConfidenceFiltering:
    """Test confidence filtering in composite signal calculation."""

    def test_low_confidence_signal_excluded(self, tmp_path):
        """Signals with confidence <= 0.3 should be excluded from composite."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            client.get_satellite_signal = lambda t, d=30: AlternativeDataSignal(
                ticker="SPY", source="satellite", signal_type="momentum",
                score=0.8, confidence=0.2, raw_value=10.0, raw_unit="pct",
                period_days=30, z_score=1.5, percentile=90.0,
                trend_direction="improving", data_timestamp="2026-01-01",
            )
            client.get_credit_card_signal = lambda t, d=30: None
            client.get_supply_chain_signal = lambda t, d=30: None
            composite = client.get_composite_signal("SPY")
        assert composite.composite_score == 0.0
        assert composite.composite_confidence == 0.0

    def test_single_source_with_high_confidence(self, tmp_path):
        """One high-confidence signal still produces composite."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            client.get_satellite_signal = lambda t, d=30: AlternativeDataSignal(
                ticker="SPY", source="satellite", signal_type="momentum",
                score=0.5, confidence=0.8, raw_value=8.0, raw_unit="pct",
                period_days=30, z_score=1.0, percentile=80.0,
                trend_direction="improving", data_timestamp="2026-01-01",
            )
            client.get_credit_card_signal = lambda t, d=30: None
            client.get_supply_chain_signal = lambda t, d=30: None
            composite = client.get_composite_signal("SPY")
        assert composite.composite_score > 0
        assert composite.composite_confidence > 0
        assert composite.primary_driver == "satellite"


# ---------------------------------------------------------------------------
# Storage of prediction_accuracy table (init_database creates it)
# ---------------------------------------------------------------------------

class TestPredictionAccuracyTable:
    """Verify prediction_accuracy table is created and usable."""

    def test_prediction_accuracy_insert_and_read(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            with sqlite3.connect(str(tmp_path / "alt.db")) as conn:
                conn.execute("""
                    INSERT INTO prediction_accuracy
                    (ticker, quarter, prediction_date, predicted_revenue_growth,
                     actual_revenue_growth, prediction_error, primary_signals, accuracy_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, ("AAPL", "Q4-2025", "2025-10-01", 8.5, 9.2, -0.7,
                      '["satellite","credit_card"]', 0.92))
                cursor = conn.cursor()
                cursor.execute("SELECT ticker, accuracy_score FROM prediction_accuracy WHERE ticker=?", ("AAPL",))
                row = cursor.fetchone()
        assert row is not None
        assert row[0] == "AAPL"
        assert row[1] == 0.92

    def test_prediction_accuracy_unique_constraint(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            with sqlite3.connect(str(tmp_path / "alt.db")) as conn:
                conn.execute("""
                    INSERT INTO prediction_accuracy
                    (ticker, quarter, prediction_date, predicted_revenue_growth,
                     actual_revenue_growth, prediction_error, primary_signals, accuracy_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, ("AAPL", "Q4-2025", "2025-10-01", 8.5, 9.2, -0.7, "[]", 0.92))
                conn.execute("""
                    INSERT OR REPLACE INTO prediction_accuracy
                    (ticker, quarter, prediction_date, predicted_revenue_growth,
                     actual_revenue_growth, prediction_error, primary_signals, accuracy_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, ("AAPL", "Q4-2025", "2025-10-01", 7.0, 7.5, -0.5, "[]", 0.85))
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM prediction_accuracy
                    WHERE ticker=? AND quarter=?
                """, ("AAPL", "Q4-2025"))
                count = cursor.fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# AlternativeDataClient batch signals with varied confidence levels
# ---------------------------------------------------------------------------

class TestBatchSignalsVariedConfidence:
    """Varied confidence level handling in batch signals."""

    def test_batch_mixed_confidence_all_returned(self, tmp_path):
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            results = client.get_batch_signals(["WMT", "COST", "HD"], days=30)
        assert len(results) == 3
        for ticker in ("WMT", "COST", "HD"):
            assert isinstance(results[ticker], CompositeSignal)
            assert results[ticker].ticker == ticker

    def test_batch_twenty_tickers(self, tmp_path):
        """Batch with a larger ticker list should handle all correctly."""
        with patch("src.data.alternative_data.ALT_DATA_DB", tmp_path / "alt.db"):
            init_database()
            client = AlternativeDataClient()
            tickers = ["WMT", "TGT", "COST", "HD", "LOW", "NKE", "MCD", "SBUX",
                        "TJX", "ROST", "DG", "DLTR", "BBY", "KSS", "JWN",
                        "M", "GPS", "URBN"]
            results = client.get_batch_signals(tickers, days=15)
        assert len(results) == len(tickers)
        for ticker in tickers:
            assert ticker in results
