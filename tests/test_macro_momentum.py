#!/usr/bin/env python3
"""
Tests for macro momentum signals — data classes, theme computation,
regime classification, allocation shifts, and FRED data loading.
"""
import sys
import os
import json
import sqlite3
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.signals.macro_momentum import (
    MacroTheme, MacroSignal, MacroMomentumReading,
    MacroMomentumEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_data(n_days=400, seed=42):
    """Create synthetic price DataFrame with SPY, GLD, TLT, SHY columns."""
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    data = {}
    for ticker, drift in [('SPY', 0.0004), ('GLD', 0.0002), ('TLT', 0.0001), ('SHY', 0.00005)]:
        prices = [500.0 if ticker != 'SHY' else 80.0]
        for _ in range(n_days - 1):
            ret = np.random.normal(drift, 0.012 if ticker != 'SHY' else 0.002)
            prices.append(prices[-1] * (1 + ret))
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


def _make_engine(tmp_path=None):
    """Create a MacroMomentumEngine with test database."""
    db_path = tmp_path / "macro_test.db" if tmp_path else Path("/tmp/macro_test.db")
    engine = MacroMomentumEngine.__new__(MacroMomentumEngine)
    engine.db_path = db_path
    engine.fred_api_key = None
    engine.price_data = None
    engine.macro_data = {}
    engine._init_db()
    return engine


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestMacroTheme:
    """Test MacroTheme enum."""

    def test_values(self):
        assert MacroTheme.BUSINESS_CYCLE.value == "business_cycle"
        assert MacroTheme.INTERNATIONAL_TRADE.value == "international_trade"
        assert MacroTheme.MONETARY_POLICY.value == "monetary_policy"
        assert MacroTheme.RISK_SENTIMENT.value == "risk_sentiment"

    def test_all_members(self):
        assert len(MacroTheme) == 4


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestMacroSignal:
    """Test MacroSignal dataclass."""

    def test_creation(self):
        sig = MacroSignal(
            theme=MacroTheme.BUSINESS_CYCLE,
            timestamp='2026-01-01',
            primary_score=0.5,
        )
        assert sig.theme == MacroTheme.BUSINESS_CYCLE
        assert sig.primary_score == 0.5
        assert sig.composite_score == 0.0
        assert sig.confidence == 0.0

    def test_optional_fields(self):
        sig = MacroSignal(
            theme=MacroTheme.RISK_SENTIMENT,
            timestamp='2026-01-01',
            primary_score=0.3,
            secondary_score=-0.1,
            tertiary_score=0.2,
            composite_score=0.15,
            confidence=0.8,
        )
        assert sig.secondary_score == -0.1
        assert sig.tertiary_score == 0.2


class TestMacroMomentumReading:
    """Test MacroMomentumReading dataclass."""

    def test_creation(self):
        bc = MacroSignal(theme=MacroTheme.BUSINESS_CYCLE, timestamp='2026-01-01', primary_score=0.5)
        it = MacroSignal(theme=MacroTheme.INTERNATIONAL_TRADE, timestamp='2026-01-01', primary_score=0.1)
        mp = MacroSignal(theme=MacroTheme.MONETARY_POLICY, timestamp='2026-01-01', primary_score=-0.2)
        rs = MacroSignal(theme=MacroTheme.RISK_SENTIMENT, timestamp='2026-01-01', primary_score=0.3)
        reading = MacroMomentumReading(
            timestamp='2026-01-01',
            business_cycle=bc, international_trade=it,
            monetary_policy=mp, risk_sentiment=rs,
            aggregate_score=0.2, regime_classification='expansion',
            equity_bias=0.5, duration_bias=-0.3, gold_bias=0.1, risk_off_score=0.1,
        )
        assert reading.regime_classification == 'expansion'
        assert reading.equity_bias == 0.5


# ---------------------------------------------------------------------------
# Engine init tests
# ---------------------------------------------------------------------------

class TestEngineInit:
    """Test MacroMomentumEngine initialization."""

    def test_db_created(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine.db_path.exists()

    def test_fred_series_defined(self):
        engine = _make_engine()
        assert 'GDPC1' in engine.FRED_SERIES
        assert 'UNRATE' in engine.FRED_SERIES

    def test_price_proxies_defined(self):
        engine = _make_engine()
        assert 'SPY' in engine.PRICE_PROXIES
        assert 'GLD' in engine.PRICE_PROXIES


# ---------------------------------------------------------------------------
# Business cycle signal tests
# ---------------------------------------------------------------------------

class TestBusinessCycleSignal:
    """Test compute_business_cycle_signal."""

    def test_returns_macro_signal(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        sig = engine.compute_business_cycle_signal()
        assert isinstance(sig, MacroSignal)
        assert sig.theme == MacroTheme.BUSINESS_CYCLE

    def test_score_bounded(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        sig = engine.compute_business_cycle_signal()
        assert -1.0 <= sig.primary_score <= 1.0
        assert -1.0 <= sig.composite_score <= 1.0

    def test_no_spy_returns_neutral(self):
        engine = _make_engine()
        engine.price_data = pd.DataFrame({'GLD': [100, 101, 102]})
        sig = engine.compute_business_cycle_signal()
        assert sig.primary_score == 0.0
        assert sig.confidence == 0.0

    def test_confidence_with_data(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        sig = engine.compute_business_cycle_signal()
        assert sig.confidence > 0

    def test_secondary_score_populated(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        sig = engine.compute_business_cycle_signal()
        assert sig.secondary_score is not None


# ---------------------------------------------------------------------------
# Monetary policy signal tests
# ---------------------------------------------------------------------------

class TestMonetaryPolicySignal:
    """Test compute_monetary_policy_signal."""

    def test_returns_macro_signal(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        sig = engine.compute_monetary_policy_signal()
        assert isinstance(sig, MacroSignal)
        assert sig.theme == MacroTheme.MONETARY_POLICY

    def test_score_bounded(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        sig = engine.compute_monetary_policy_signal()
        assert -1.0 <= sig.primary_score <= 1.0

    def test_no_tlt_returns_neutral(self):
        engine = _make_engine()
        engine.price_data = pd.DataFrame({'SPY': [100, 101, 102]})
        sig = engine.compute_monetary_policy_signal()
        assert sig.primary_score == 0.0


# ---------------------------------------------------------------------------
# Risk sentiment signal tests
# ---------------------------------------------------------------------------

class TestRiskSentimentSignal:
    """Test compute_risk_sentiment_signal."""

    def test_returns_macro_signal(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        sig = engine.compute_risk_sentiment_signal()
        assert isinstance(sig, MacroSignal)
        assert sig.theme == MacroTheme.RISK_SENTIMENT

    def test_score_bounded(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        sig = engine.compute_risk_sentiment_signal()
        assert -1.0 <= sig.primary_score <= 1.0

    def test_no_spy_returns_neutral(self):
        engine = _make_engine()
        engine.price_data = pd.DataFrame({'GLD': [100, 101, 102]})
        sig = engine.compute_risk_sentiment_signal()
        assert sig.primary_score == 0.0

    def test_tertiary_score_with_gold(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        sig = engine.compute_risk_sentiment_signal()
        assert sig.tertiary_score is not None


# ---------------------------------------------------------------------------
# International trade signal tests
# ---------------------------------------------------------------------------

class TestInternationalTradeSignal:
    """Test compute_international_trade_signal."""

    def test_returns_macro_signal(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        sig = engine.compute_international_trade_signal()
        assert isinstance(sig, MacroSignal)
        assert sig.theme == MacroTheme.INTERNATIONAL_TRADE

    def test_score_bounded(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        sig = engine.compute_international_trade_signal()
        assert -1.0 <= sig.primary_score <= 1.0

    def test_no_spy_returns_neutral(self):
        engine = _make_engine()
        engine.price_data = pd.DataFrame({'TLT': [100, 101, 102]})
        sig = engine.compute_international_trade_signal()
        assert sig.primary_score == 0.0


# ---------------------------------------------------------------------------
# Compute reading tests
# ---------------------------------------------------------------------------

class TestComputeReading:
    """Test compute_reading method."""

    def test_returns_reading(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        reading = engine.compute_reading()
        assert isinstance(reading, MacroMomentumReading)

    def test_has_all_themes(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        reading = engine.compute_reading()
        assert reading.business_cycle.theme == MacroTheme.BUSINESS_CYCLE
        assert reading.international_trade.theme == MacroTheme.INTERNATIONAL_TRADE
        assert reading.monetary_policy.theme == MacroTheme.MONETARY_POLICY
        assert reading.risk_sentiment.theme == MacroTheme.RISK_SENTIMENT

    def test_aggregate_score_bounded(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        reading = engine.compute_reading()
        assert -1.0 <= reading.aggregate_score <= 1.0

    def test_regime_classification_valid(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        reading = engine.compute_reading()
        assert reading.regime_classification in [
            'expansion', 'slowdown', 'recovery', 'risk_off', 'neutral'
        ]

    def test_biases_bounded(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        reading = engine.compute_reading()
        assert -1.0 <= reading.equity_bias <= 1.0
        assert -1.0 <= reading.duration_bias <= 1.0
        assert -1.0 <= reading.gold_bias <= 1.0

    def test_risk_off_non_negative(self):
        engine = _make_engine()
        engine.price_data = _make_price_data()
        reading = engine.compute_reading()
        assert reading.risk_off_score >= 0.0


# ---------------------------------------------------------------------------
# Allocation shift tests
# ---------------------------------------------------------------------------

class TestGetAllocationShift:
    """Test get_allocation_shift method."""

    def test_returns_dict(self):
        engine = _make_engine()
        bc = MacroSignal(theme=MacroTheme.BUSINESS_CYCLE, timestamp='2026-01-01', primary_score=0.5, composite_score=0.5)
        it = MacroSignal(theme=MacroTheme.INTERNATIONAL_TRADE, timestamp='2026-01-01', primary_score=0.1, composite_score=0.1)
        mp = MacroSignal(theme=MacroTheme.MONETARY_POLICY, timestamp='2026-01-01', primary_score=-0.2, composite_score=-0.2)
        rs = MacroSignal(theme=MacroTheme.RISK_SENTIMENT, timestamp='2026-01-01', primary_score=0.3, composite_score=0.3)
        reading = MacroMomentumReading(
            timestamp='2026-01-01',
            business_cycle=bc, international_trade=it,
            monetary_policy=mp, risk_sentiment=rs,
            aggregate_score=0.2, regime_classification='expansion',
            equity_bias=0.5, duration_bias=-0.3, gold_bias=0.1, risk_off_score=0.1,
        )
        shifts = engine.get_allocation_shift(reading)
        assert 'SPY' in shifts
        assert 'TLT' in shifts
        assert 'GLD' in shifts

    def test_shifts_proportional_to_biases(self):
        engine = _make_engine()
        bc = MacroSignal(theme=MacroTheme.BUSINESS_CYCLE, timestamp='2026-01-01', primary_score=0.5, composite_score=0.5)
        it = MacroSignal(theme=MacroTheme.INTERNATIONAL_TRADE, timestamp='2026-01-01', primary_score=0.1, composite_score=0.1)
        mp = MacroSignal(theme=MacroTheme.MONETARY_POLICY, timestamp='2026-01-01', primary_score=-0.2, composite_score=-0.2)
        rs = MacroSignal(theme=MacroTheme.RISK_SENTIMENT, timestamp='2026-01-01', primary_score=0.3, composite_score=0.3)
        reading = MacroMomentumReading(
            timestamp='2026-01-01',
            business_cycle=bc, international_trade=it,
            monetary_policy=mp, risk_sentiment=rs,
            aggregate_score=0.2, regime_classification='expansion',
            equity_bias=0.5, duration_bias=-0.3, gold_bias=0.1, risk_off_score=0.1,
        )
        shifts = engine.get_allocation_shift(reading)
        assert shifts['SPY'] == pytest.approx(0.05, abs=0.01)
        assert shifts['TLT'] == pytest.approx(-0.03, abs=0.01)

    def test_risk_off_boosts_gold(self):
        """High risk_off_score boosts GLD and reduces SPY."""
        engine = _make_engine()
        bc = MacroSignal(theme=MacroTheme.BUSINESS_CYCLE, timestamp='2026-01-01', primary_score=-0.5, composite_score=-0.5)
        it = MacroSignal(theme=MacroTheme.INTERNATIONAL_TRADE, timestamp='2026-01-01', primary_score=0.0, composite_score=0.0)
        mp = MacroSignal(theme=MacroTheme.MONETARY_POLICY, timestamp='2026-01-01', primary_score=0.0, composite_score=0.0)
        rs = MacroSignal(theme=MacroTheme.RISK_SENTIMENT, timestamp='2026-01-01', primary_score=-0.8, composite_score=-0.8)
        reading = MacroMomentumReading(
            timestamp='2026-01-01',
            business_cycle=bc, international_trade=it,
            monetary_policy=mp, risk_sentiment=rs,
            aggregate_score=-0.4, regime_classification='risk_off',
            equity_bias=-0.5, duration_bias=0.3, gold_bias=0.4, risk_off_score=0.8,
        )
        shifts = engine.get_allocation_shift(reading)
        # Risk-off override: GLD gets extra, SPY gets reduced
        base_gold_shift = reading.gold_bias * 0.10
        assert shifts['GLD'] > base_gold_shift


# ---------------------------------------------------------------------------
# FRED data tests
# ---------------------------------------------------------------------------

class TestFetchFredData:
    """Test fetch_fred_data method."""

    def test_no_api_key_returns_none(self):
        engine = _make_engine()
        engine.fred_api_key = None
        result = engine.fetch_fred_data('GDPC1')
        assert result is None

    def test_caches_to_db(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine.fred_api_key = 'test_key'
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            'observations': [
                {'date': '2026-01-01', 'value': '100.0'},
                {'date': '2026-02-01', 'value': '101.0'},
            ]
        }
        with patch('src.signals.macro_momentum.requests.get', return_value=mock_resp):
            result = engine.fetch_fred_data('GDPC1')
        assert result is not None
        assert len(result) == 2

        # Verify cached in DB
        conn = sqlite3.connect(str(engine.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM macro_series WHERE series_id = 'GDPC1'")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 2


class TestLoadCachedFred:
    """Test load_cached_fred method."""

    def test_empty_returns_none(self):
        engine = _make_engine()
        result = engine.load_cached_fred('NONEXISTENT')
        assert result is None

    def test_returns_cached_series(self, tmp_path):
        engine = _make_engine(tmp_path)
        # Insert test data
        conn = sqlite3.connect(str(engine.db_path))
        conn.execute("INSERT INTO macro_series VALUES ('TEST', '2026-01-01', 100.0)")
        conn.execute("INSERT INTO macro_series VALUES ('TEST', '2026-02-01', 101.0)")
        conn.commit()
        conn.close()

        result = engine.load_cached_fred('TEST')
        assert result is not None
        assert len(result) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
