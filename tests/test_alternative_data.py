#!/usr/bin/env python3
"""
Tests for alternative data module — data classes, adapters, composite signals,
earnings predictions.
"""
import json
import sqlite3

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
