#!/usr/bin/env python3
"""
Tests for FX Currency Carry Data Fetcher (src/data/fx_fetcher.py).
Tests FXMetrics dataclass, FXFetcher caching/logic, and signal generation.
No network calls — Yahoo API is mocked.
"""
import sys
import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.data.fx_fetcher import FXMetrics, FXFetcher, CACHE_TTL_HOURS, DB_PATH


# ---------------------------------------------------------------------------
# FXMetrics dataclass tests
# ---------------------------------------------------------------------------

class TestFXMetrics:
    def test_creation(self):
        m = FXMetrics(
            timestamp="2026-05-21T00:00:00",
            uup_price=28.5,
            udn_price=22.1,
            uup_return_30d=1.5,
            udn_return_30d=-1.2,
            usd_strength_score=0.34,
            carry_regime="positive",
            momentum_direction="bullish",
            volatility_regime="medium",
            data_freshness_hours=0.5,
        )
        assert m.uup_price == 28.5
        assert m.carry_regime == "positive"
        assert m.momentum_direction == "bullish"

    def test_usd_strength_bounded(self):
        m = FXMetrics(
            timestamp="2026-05-21T00:00:00", uup_price=28.5, udn_price=22.1,
            uup_return_30d=1.5, udn_return_30d=-1.2,
            usd_strength_score=-1.0, carry_regime="negative",
            momentum_direction="bearish", volatility_regime="high",
            data_freshness_hours=0.5,
        )
        assert -1.0 <= m.usd_strength_score <= 1.0

    def test_carry_regime_values(self):
        for regime in ("positive", "negative", "neutral"):
            m = FXMetrics(
                timestamp="2026-05-21T00:00:00", uup_price=28.5, udn_price=22.1,
                uup_return_30d=1.5, udn_return_30d=-1.2,
                usd_strength_score=0.0, carry_regime=regime,
                momentum_direction="neutral", volatility_regime="low",
                data_freshness_hours=0.5,
            )
            assert m.carry_regime == regime

    def test_volatility_regime_values(self):
        for vol in ("low", "medium", "high"):
            m = FXMetrics(
                timestamp="2026-05-21T00:00:00", uup_price=28.5, udn_price=22.1,
                uup_return_30d=1.5, udn_return_30d=-1.2,
                usd_strength_score=0.0, carry_regime="neutral",
                momentum_direction="neutral", volatility_regime=vol,
                data_freshness_hours=0.5,
            )
            assert m.volatility_regime == vol

    def test_asdict_roundtrip(self):
        m = FXMetrics(
            timestamp="2026-05-21T00:00:00", uup_price=28.5, udn_price=22.1,
            uup_return_30d=1.5, udn_return_30d=-1.2,
            usd_strength_score=0.34, carry_regime="positive",
            momentum_direction="bullish", volatility_regime="medium",
            data_freshness_hours=0.5,
        )
        d = m.__dict__
        assert d["uup_price"] == 28.5
        assert d["carry_regime"] == "positive"


# ---------------------------------------------------------------------------
# FXFetcher database tests
# ---------------------------------------------------------------------------

class TestFXFetcherDB:
    def test_init_db_creates_table(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = FXFetcher(db_path=db)
        with sqlite3.connect(db) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='fx_cache'"
            )
            assert cursor.fetchone() is not None

    def test_cache_fresh_when_recent(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = FXFetcher(db_path=db)
        now = datetime.now().isoformat()
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO fx_cache (symbol, price, price_30d_ago, volatility_30d, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("UUP", 28.5, 27.8, 0.1, now),
            )
            conn.commit()
        assert fetcher._is_cache_fresh("UUP") is True

    def test_cache_stale_when_old(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = FXFetcher(db_path=db)
        old = (datetime.now() - timedelta(hours=CACHE_TTL_HOURS + 1)).isoformat()
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO fx_cache (symbol, price, price_30d_ago, volatility_30d, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("UUP", 28.5, 27.8, 0.1, old),
            )
            conn.commit()
        assert fetcher._is_cache_fresh("UUP") is False

    def test_cache_missing_returns_false(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = FXFetcher(db_path=db)
        assert fetcher._is_cache_fresh("NONEXISTENT") is False


# ---------------------------------------------------------------------------
# FXFetcher Yahoo mock tests
# ---------------------------------------------------------------------------

def _mock_yahoo_response(price=28.5, price_30d_ago=27.8, volatility=0.1):
    """Build a mock Yahoo Finance API response."""
    n = 60
    timestamps = [1700000000 + i * 86400 for i in range(n)]
    prices = [price_30d_ago + (price - price_30d_ago) * i / n for i in range(n)]
    return {
        "chart": {
            "result": [{
                "timestamp": timestamps,
                "indicators": {
                    "adjclose": [{"adjclose": prices}]
                }
            }]
        }
    }


class TestFXFetcherFetch:
    def test_fetch_yahoo_parses_response(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = FXFetcher(db_path=db)
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_yahoo_response()
        mock_resp.raise_for_status = MagicMock()

        with patch("src.data.fx_fetcher.requests.get", return_value=mock_resp):
            data = fetcher._fetch_yahoo("UUP")
        assert "price" in data
        assert "price_30d_ago" in data
        assert "volatility_30d" in data
        assert data["price"] > 0

    def test_fetch_yahoo_insufficient_data(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = FXFetcher(db_path=db)
        # Only 10 data points
        timestamps = [1700000000 + i * 86400 for i in range(10)]
        prices = [28.0 + i * 0.1 for i in range(10)]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "chart": {
                "result": [{
                    "timestamp": timestamps,
                    "indicators": {"adjclose": [{"adjclose": prices}]}
                }]
            }
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("src.data.fx_fetcher.requests.get", return_value=mock_resp):
            with pytest.raises(ValueError, match="Insufficient data"):
                fetcher._fetch_yahoo("UUP")

    def test_get_cached_or_fetch_uses_cache(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = FXFetcher(db_path=db)
        now = datetime.now().isoformat()
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO fx_cache (symbol, price, price_30d_ago, volatility_30d, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("UUP", 28.5, 27.8, 0.1, now),
            )
            conn.commit()

        # Should NOT call Yahoo — cache is fresh
        with patch("src.data.fx_fetcher.requests.get") as mock_get:
            data = fetcher._get_cached_or_fetch("UUP")
            mock_get.assert_not_called()
        assert data["price"] == 28.5

    def test_get_cached_or_fetch_fetches_when_stale(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = FXFetcher(db_path=db)
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_yahoo_response()
        mock_resp.raise_for_status = MagicMock()

        with patch("src.data.fx_fetcher.requests.get", return_value=mock_resp):
            data = fetcher._get_cached_or_fetch("UUP")
        assert data["price"] > 0


# ---------------------------------------------------------------------------
# FXFetcher metrics computation tests
# ---------------------------------------------------------------------------

class TestFXFetcherMetrics:
    def _make_fetcher_with_mock(self, tmp_path, uup_price=28.5, udn_price=22.1,
                                 uup_30d=27.8, udn_30d=22.5, uup_vol=0.1, udn_vol=0.1):
        """Create fetcher with cached data (no network calls)."""
        db = tmp_path / "test.db"
        fetcher = FXFetcher(db_path=db)
        now = datetime.now().isoformat()
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO fx_cache (symbol, price, price_30d_ago, volatility_30d, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("UUP", uup_price, uup_30d, uup_vol, now),
            )
            conn.execute(
                "INSERT INTO fx_cache (symbol, price, price_30d_ago, volatility_30d, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("UDN", udn_price, udn_30d, udn_vol, now),
            )
            conn.commit()
        return fetcher

    def test_positive_carry_regime(self, tmp_path):
        # UUP strong, UDN weak → positive carry, bullish
        fetcher = self._make_fetcher_with_mock(
            tmp_path, uup_price=30.0, udn_price=20.0,
            uup_30d=27.0, udn_30d=22.0
        )
        metrics = fetcher.fetch_metrics()
        assert metrics.carry_regime == "positive"
        assert metrics.momentum_direction == "bullish"

    def test_negative_carry_regime(self, tmp_path):
        # UDN strong, UUP weak → negative carry, bearish
        fetcher = self._make_fetcher_with_mock(
            tmp_path, uup_price=25.0, udn_price=25.0,
            uup_30d=27.0, udn_30d=22.0
        )
        metrics = fetcher.fetch_metrics()
        assert metrics.carry_regime == "negative"
        assert metrics.momentum_direction == "bearish"

    def test_neutral_carry_regime(self, tmp_path):
        # Both similar → neutral
        fetcher = self._make_fetcher_with_mock(
            tmp_path, uup_price=28.5, udn_price=22.1,
            uup_30d=28.0, udn_30d=22.0
        )
        metrics = fetcher.fetch_metrics()
        assert metrics.carry_regime == "neutral"
        assert metrics.momentum_direction == "neutral"

    def test_usd_strength_score_range(self, tmp_path):
        fetcher = self._make_fetcher_with_mock(
            tmp_path, uup_price=30.0, udn_price=20.0,
            uup_30d=27.0, udn_30d=22.0
        )
        metrics = fetcher.fetch_metrics()
        assert -1.0 <= metrics.usd_strength_score <= 1.0

    def test_low_volatility_regime(self, tmp_path):
        fetcher = self._make_fetcher_with_mock(
            tmp_path, uup_vol=0.05, udn_vol=0.06
        )
        metrics = fetcher.fetch_metrics()
        assert metrics.volatility_regime == "low"

    def test_medium_volatility_regime(self, tmp_path):
        fetcher = self._make_fetcher_with_mock(
            tmp_path, uup_vol=0.10, udn_vol=0.12
        )
        metrics = fetcher.fetch_metrics()
        assert metrics.volatility_regime == "medium"

    def test_high_volatility_regime(self, tmp_path):
        fetcher = self._make_fetcher_with_mock(
            tmp_path, uup_vol=0.20, udn_vol=0.18
        )
        metrics = fetcher.fetch_metrics()
        assert metrics.volatility_regime == "high"

    def test_returns_30d_calculation(self, tmp_path):
        fetcher = self._make_fetcher_with_mock(
            tmp_path, uup_price=30.0, udn_price=22.0,
            uup_30d=28.0, udn_30d=22.5
        )
        metrics = fetcher.fetch_metrics()
        # UUP return: (30 - 28) / 28 * 100 ≈ 7.14%
        assert metrics.uup_return_30d > 0
        # UDN return: (22 - 22.5) / 22.5 * 100 ≈ -2.22%
        assert metrics.udn_return_30d < 0


# ---------------------------------------------------------------------------
# FXFetcher signal generation tests
# ---------------------------------------------------------------------------

class TestFXFetcherSignal:
    def _make_fetcher_with_regime(self, tmp_path, carry_regime="positive",
                                   momentum_direction="bullish", vol_regime="medium"):
        db = tmp_path / "test.db"
        fetcher = FXFetcher(db_path=db)
        # Mock fetch_metrics to return controlled FXMetrics
        mock_metrics = FXMetrics(
            timestamp=datetime.now().isoformat(),
            uup_price=28.5, udn_price=22.1,
            uup_return_30d=3.0, udn_return_30d=-2.0,
            usd_strength_score=0.5, carry_regime=carry_regime,
            momentum_direction=momentum_direction,
            volatility_regime=vol_regime,
            data_freshness_hours=0.5,
        )
        fetcher.fetch_metrics = MagicMock(return_value=mock_metrics)
        return fetcher, mock_metrics

    def test_signal_positive_carry(self, tmp_path):
        fetcher, _ = self._make_fetcher_with_regime(
            tmp_path, carry_regime="positive", momentum_direction="bullish"
        )
        signal = fetcher.get_signal()
        assert signal["signal_type"] == "usd_strength"
        assert signal["confidence"] > 0
        assert signal["reason"] == "momentum_aligned"

    def test_signal_negative_carry(self, tmp_path):
        fetcher, _ = self._make_fetcher_with_regime(
            tmp_path, carry_regime="negative", momentum_direction="bearish"
        )
        signal = fetcher.get_signal()
        assert signal["signal_type"] == "usd_weakness"
        assert signal["confidence"] > 0

    def test_signal_neutral_regime(self, tmp_path):
        fetcher, _ = self._make_fetcher_with_regime(
            tmp_path, carry_regime="neutral", momentum_direction="neutral"
        )
        signal = fetcher.get_signal()
        assert signal["signal_type"] == "neutral"
        assert signal["confidence"] == 0.0
        assert signal["reason"] == "no_clear_regime"

    def test_signal_high_volatility_neutral(self, tmp_path):
        fetcher, _ = self._make_fetcher_with_regime(
            tmp_path, carry_regime="positive", momentum_direction="bullish",
            vol_regime="high"
        )
        signal = fetcher.get_signal()
        # High vol overrides everything → neutral
        assert signal["signal_type"] == "neutral"
        assert signal["confidence"] == 0.0
        assert signal["reason"] == "high_volatility"

    def test_signal_has_required_fields(self, tmp_path):
        fetcher, _ = self._make_fetcher_with_regime(tmp_path)
        signal = fetcher.get_signal()
        for field in ("signal_type", "confidence", "regime", "direction", "reason", "timestamp"):
            assert field in signal


# ---------------------------------------------------------------------------
# FXFetcher save_metrics tests
# ---------------------------------------------------------------------------

class TestFXFetcherSave:
    def test_save_metrics_creates_json(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = FXFetcher(db_path=db)
        metrics = FXMetrics(
            timestamp=datetime.now().isoformat(),
            uup_price=28.5, udn_price=22.1,
            uup_return_30d=1.5, udn_return_30d=-1.2,
            usd_strength_score=0.34, carry_regime="positive",
            momentum_direction="bullish", volatility_regime="medium",
            data_freshness_hours=0.5,
        )
        output = tmp_path / "fx_metrics.json"
        fetcher.save_metrics(metrics, output)
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["uup_price"] == 28.5
        assert data["carry_regime"] == "positive"

    def test_save_metrics_default_path(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = FXFetcher(db_path=db)
        metrics = FXMetrics(
            timestamp=datetime.now().isoformat(),
            uup_price=28.5, udn_price=22.1,
            uup_return_30d=1.5, udn_return_30d=-1.2,
            usd_strength_score=0.34, carry_regime="neutral",
            momentum_direction="neutral", volatility_regime="low",
            data_freshness_hours=0.5,
        )
        # Use tmp_path as default output dir
        output = tmp_path / "fx_metrics.json"
        fetcher.save_metrics(metrics, output)
        assert output.exists()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestFXFetcherEdgeCases:
    def test_zero_price_30d_ago_no_division_error(self, tmp_path):
        """If price_30d_ago is 0, fetch_metrics should not crash on division."""
        db = tmp_path / "test.db"
        fetcher = FXFetcher(db_path=db)
        now = datetime.now().isoformat()
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO fx_cache (symbol, price, price_30d_ago, volatility_30d, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("UUP", 28.5, 0.0, 0.1, now),
            )
            conn.execute(
                "INSERT INTO fx_cache (symbol, price, price_30d_ago, volatility_30d, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("UDN", 22.1, 22.0, 0.1, now),
            )
            conn.commit()
        # This will raise ZeroDivisionError — that's expected behavior
        with pytest.raises(ZeroDivisionError):
            fetcher.fetch_metrics()

    def test_none_prices_in_yahoo_response(self, tmp_path):
        """Yahoo API sometimes returns None prices — should be filtered."""
        db = tmp_path / "test.db"
        fetcher = FXFetcher(db_path=db)
        timestamps = [1700000000 + i * 86400 for i in range(60)]
        prices = [28.0 + i * 0.05 if i % 3 != 0 else None for i in range(60)]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "chart": {
                "result": [{
                    "timestamp": timestamps,
                    "indicators": {"adjclose": [{"adjclose": prices}]}
                }]
            }
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("src.data.fx_fetcher.requests.get", return_value=mock_resp):
            data = fetcher._fetch_yahoo("UUP")
        assert data["price"] > 0

    def test_custom_db_path(self, tmp_path):
        custom_db = tmp_path / "custom" / "fx.db"
        fetcher = FXFetcher(db_path=custom_db)
        assert fetcher.db_path == custom_db
        assert custom_db.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
