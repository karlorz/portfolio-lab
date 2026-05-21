#!/usr/bin/env python3
"""
Tests for src/data/international_fetcher.py — InternationalDataFetcher.
Covers: MomentumMetrics, RelativeMomentum, InternationalData dataclasses,
cache operations, momentum calculation, relative momentum, signal summary.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.international_fetcher import (
    InternationalDataFetcher,
    MomentumMetrics,
    RelativeMomentum,
    InternationalData,
    CACHE_TTL_HOURS,
    MOMENTUM_WINDOWS,
    SYMBOLS,
    HAS_DEPS,
)


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestMomentumMetrics:
    def test_creation(self):
        m = MomentumMetrics(
            symbol="EFA", price=75.0, momentum_1m=0.02, momentum_3m=-0.01,
            momentum_6m=0.05, volatility_20d=0.15, sharpe_6m=0.33,
            timestamp="2026-01-01",
        )
        assert m.symbol == "EFA"
        assert m.price == 75.0
        assert m.momentum_6m == 0.05

    def test_to_dict(self):
        m = MomentumMetrics(
            symbol="EEM", price=40.0, momentum_1m=0.03, momentum_3m=0.01,
            momentum_6m=-0.02, volatility_20d=0.20, sharpe_6m=-0.10,
            timestamp="2026-01-01",
        )
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["symbol"] == "EEM"
        assert d["momentum_6m"] == -0.02


class TestRelativeMomentum:
    def test_creation(self):
        r = RelativeMomentum(
            symbol="relative_momentum", efa_momentum_6m=0.05, eem_momentum_6m=-0.02,
            spy_momentum_6m=0.03, efa_vs_spy=0.02, eem_vs_spy=-0.05,
            signal="efa_lead", confidence=0.5, timestamp="2026-01-01",
        )
        assert r.signal == "efa_lead"
        assert r.efa_vs_spy == 0.02

    def test_to_dict(self):
        r = RelativeMomentum(
            symbol="relative_momentum", efa_momentum_6m=0.05, eem_momentum_6m=-0.02,
            spy_momentum_6m=0.03, efa_vs_spy=0.02, eem_vs_spy=-0.05,
            signal="neutral", confidence=0.0, timestamp="2026-01-01",
        )
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["signal"] == "neutral"


class TestInternationalData:
    def test_creation(self):
        efa = MomentumMetrics(
            symbol="EFA", price=75.0, momentum_1m=0.02, momentum_3m=-0.01,
            momentum_6m=0.05, volatility_20d=0.15, sharpe_6m=0.33,
            timestamp="2026-01-01",
        )
        rel = RelativeMomentum(
            symbol="relative_momentum", efa_momentum_6m=0.05, eem_momentum_6m=-0.02,
            spy_momentum_6m=0.03, efa_vs_spy=0.02, eem_vs_spy=-0.05,
            signal="efa_lead", confidence=0.5, timestamp="2026-01-01",
        )
        data = InternationalData(
            timestamp="2026-01-01", metrics={"EFA": efa}, relative=rel,
            data_fresh=True,
        )
        assert data.data_fresh is True
        assert "EFA" in data.metrics

    def test_to_dict(self):
        efa = MomentumMetrics(
            symbol="EFA", price=75.0, momentum_1m=0.02, momentum_3m=-0.01,
            momentum_6m=0.05, volatility_20d=0.15, sharpe_6m=0.33,
            timestamp="2026-01-01",
        )
        rel = RelativeMomentum(
            symbol="relative_momentum", efa_momentum_6m=0.05, eem_momentum_6m=-0.02,
            spy_momentum_6m=0.03, efa_vs_spy=0.02, eem_vs_spy=-0.05,
            signal="neutral", confidence=0.0, timestamp="2026-01-01",
        )
        data = InternationalData(
            timestamp="2026-01-01", metrics={"EFA": efa}, relative=rel,
            data_fresh=False,
        )
        d = data.to_dict()
        assert isinstance(d, dict)
        assert d["data_fresh"] is False
        assert "metrics" in d
        assert "relative" in d


# ---------------------------------------------------------------------------
# Fetcher init + cache
# ---------------------------------------------------------------------------

class TestFetcherInit:
    def test_init_creates_cache_table(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = InternationalDataFetcher(cache_db=db)
        conn = sqlite3.connect(db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='international_cache'"
        )
        assert cursor.fetchone() is not None
        conn.close()


class TestCacheOperations:
    def test_is_cache_fresh_no_entry(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = InternationalDataFetcher(cache_db=db)
        assert fetcher._is_cache_fresh("EFA") is False

    def test_is_cache_fresh_recent(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = InternationalDataFetcher(cache_db=db)
        # Insert recent entry
        with sqlite3.connect(db) as conn:
            conn.execute("""
                INSERT INTO international_cache (symbol, data, timestamp, price, momentum_6m)
                VALUES (?, ?, ?, ?, ?)
            """, ("EFA", "{}", datetime.now().isoformat(), 75.0, 0.05))
            conn.commit()
        assert fetcher._is_cache_fresh("EFA") is True

    def test_is_cache_fresh_stale(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = InternationalDataFetcher(cache_db=db)
        stale_time = (datetime.now() - timedelta(hours=CACHE_TTL_HOURS + 1)).isoformat()
        with sqlite3.connect(db) as conn:
            conn.execute("""
                INSERT INTO international_cache (symbol, data, timestamp, price, momentum_6m)
                VALUES (?, ?, ?, ?, ?)
            """, ("EFA", "{}", stale_time, 75.0, 0.05))
            conn.commit()
        assert fetcher._is_cache_fresh("EFA") is False

    def test_get_cached_empty_db(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = InternationalDataFetcher(cache_db=db)
        # prices_cache table may not exist
        result = fetcher._get_cached("EFA")
        assert result is None


# ---------------------------------------------------------------------------
# Momentum calculation (requires pandas/numpy)
# ---------------------------------------------------------------------------

class TestCalculateMomentum:
    @pytest.mark.skipif(not HAS_DEPS, reason="pandas/numpy not available")
    def test_valid_data(self, tmp_path):
        import pandas as pd
        import numpy as np
        db = tmp_path / "test.db"
        fetcher = InternationalDataFetcher(cache_db=db)
        # Create 200 days of price data
        dates = pd.date_range("2025-01-01", periods=200, freq="B")
        prices = 100 * (1 + np.random.normal(0.0003, 0.01, 200)).cumprod()
        df = pd.DataFrame({
            "date": dates,
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": 1000000,
        })
        metrics = fetcher.calculate_momentum(df, "EFA")
        assert metrics.symbol == "EFA"
        assert metrics.price > 0
        assert -1 <= metrics.momentum_6m <= 1  # Reasonable range

    @pytest.mark.skipif(not HAS_DEPS, reason="pandas/numpy not available")
    def test_insufficient_data_raises(self, tmp_path):
        import pandas as pd
        db = tmp_path / "test.db"
        fetcher = InternationalDataFetcher(cache_db=db)
        df = pd.DataFrame({"close": [100, 101, 102]}, index=[0, 1, 2])
        with pytest.raises(ValueError, match="Insufficient data"):
            fetcher.calculate_momentum(df, "EFA")


# ---------------------------------------------------------------------------
# Relative momentum calculation
# ---------------------------------------------------------------------------

class TestCalculateRelativeMomentum:
    def test_efa_lead(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = InternationalDataFetcher(cache_db=db)
        efa = MomentumMetrics(
            symbol="EFA", price=75.0, momentum_1m=0.02, momentum_3m=-0.01,
            momentum_6m=0.10, volatility_20d=0.15, sharpe_6m=0.67,
            timestamp="2026-01-01",
        )
        eem = MomentumMetrics(
            symbol="EEM", price=40.0, momentum_1m=0.03, momentum_3m=0.01,
            momentum_6m=-0.02, volatility_20d=0.20, sharpe_6m=-0.10,
            timestamp="2026-01-01",
        )
        spy = MomentumMetrics(
            symbol="SPY", price=500.0, momentum_1m=0.01, momentum_3m=0.02,
            momentum_6m=0.03, volatility_20d=0.12, sharpe_6m=0.25,
            timestamp="2026-01-01",
        )
        result = fetcher.calculate_relative_momentum(efa, eem, spy)
        # EFA outperforms SPY by 7% (0.10 - 0.03) > 0.05 threshold
        assert result.signal == "efa_lead"
        assert result.efa_vs_spy == pytest.approx(0.07, abs=0.01)
        assert result.confidence > 0

    def test_eem_lead(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = InternationalDataFetcher(cache_db=db)
        efa = MomentumMetrics(
            symbol="EFA", price=75.0, momentum_1m=0.02, momentum_3m=-0.01,
            momentum_6m=0.03, volatility_20d=0.15, sharpe_6m=0.20,
            timestamp="2026-01-01",
        )
        eem = MomentumMetrics(
            symbol="EEM", price=40.0, momentum_1m=0.03, momentum_3m=0.01,
            momentum_6m=0.15, volatility_20d=0.20, sharpe_6m=0.75,
            timestamp="2026-01-01",
        )
        spy = MomentumMetrics(
            symbol="SPY", price=500.0, momentum_1m=0.01, momentum_3m=0.02,
            momentum_6m=0.03, volatility_20d=0.12, sharpe_6m=0.25,
            timestamp="2026-01-01",
        )
        result = fetcher.calculate_relative_momentum(efa, eem, spy)
        # EEM outperforms SPY by 12% (0.15 - 0.03) > 0.08 threshold
        assert result.signal == "eem_lead"
        assert result.eem_vs_spy == pytest.approx(0.12, abs=0.01)

    def test_neutral_signal(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = InternationalDataFetcher(cache_db=db)
        # All same momentum
        efa = MomentumMetrics(
            symbol="EFA", price=75.0, momentum_1m=0.02, momentum_3m=0.01,
            momentum_6m=0.03, volatility_20d=0.15, sharpe_6m=0.20,
            timestamp="2026-01-01",
        )
        eem = MomentumMetrics(
            symbol="EEM", price=40.0, momentum_1m=0.02, momentum_3m=0.01,
            momentum_6m=0.03, volatility_20d=0.20, sharpe_6m=0.15,
            timestamp="2026-01-01",
        )
        spy = MomentumMetrics(
            symbol="SPY", price=500.0, momentum_1m=0.02, momentum_3m=0.01,
            momentum_6m=0.03, volatility_20d=0.12, sharpe_6m=0.25,
            timestamp="2026-01-01",
        )
        result = fetcher.calculate_relative_momentum(efa, eem, spy)
        assert result.signal == "neutral"
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# fetch_all (mocked)
# ---------------------------------------------------------------------------

class TestFetchAll:
    @pytest.mark.skipif(not HAS_DEPS, reason="pandas/numpy not available")
    def test_fetch_all_with_mock(self, tmp_path):
        import pandas as pd
        import numpy as np
        db = tmp_path / "test.db"
        fetcher = InternationalDataFetcher(cache_db=db)

        # Create a mock DataFrame with enough data for momentum calc
        dates = pd.date_range("2025-01-01", periods=200, freq="B")
        def make_df(symbol, trend=0.0003):
            prices = 100 * (1 + np.random.normal(trend, 0.01, 200)).cumprod()
            return pd.DataFrame({
                "date": dates,
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": 1000000,
            })

        with patch.object(fetcher, "fetch_symbol", side_effect=lambda s: make_df(s)):
            data = fetcher.fetch_all()
            assert isinstance(data, InternationalData)
            assert len(data.metrics) == 3
            assert "EFA" in data.metrics
            assert "EEM" in data.metrics
            assert "SPY" in data.metrics
            assert data.relative is not None

    def test_fetch_all_with_failure(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = InternationalDataFetcher(cache_db=db)

        def failing_fetch(symbol):
            raise Exception(f"Failed to fetch {symbol}")

        with patch.object(fetcher, "fetch_symbol", side_effect=failing_fetch):
            data = fetcher.fetch_all()
            assert isinstance(data, InternationalData)
            assert data.data_fresh is False
            # Should have placeholder metrics
            assert len(data.metrics) == 3
            for m in data.metrics.values():
                assert m.price == 0.0


# ---------------------------------------------------------------------------
# Save snapshot
# ---------------------------------------------------------------------------

class TestSaveSnapshot:
    def test_saves_json(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = InternationalDataFetcher(cache_db=db)
        efa = MomentumMetrics(
            symbol="EFA", price=75.0, momentum_1m=0.02, momentum_3m=-0.01,
            momentum_6m=0.05, volatility_20d=0.15, sharpe_6m=0.33,
            timestamp="2026-01-01",
        )
        rel = RelativeMomentum(
            symbol="relative_momentum", efa_momentum_6m=0.05, eem_momentum_6m=-0.02,
            spy_momentum_6m=0.03, efa_vs_spy=0.02, eem_vs_spy=-0.05,
            signal="neutral", confidence=0.0, timestamp="2026-01-01",
        )
        data = InternationalData(
            timestamp="2026-01-01", metrics={"EFA": efa}, relative=rel,
            data_fresh=True,
        )
        output = tmp_path / "test_output.json"
        fetcher.save_snapshot(data, output_path=output)
        assert output.exists()
        with open(output) as f:
            loaded = json.load(f)
        assert loaded["timestamp"] == "2026-01-01"
        assert loaded["data_fresh"] is True


# ---------------------------------------------------------------------------
# Signal summary (mocked)
# ---------------------------------------------------------------------------

class TestSignalSummary:
    def test_get_signal_summary(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = InternationalDataFetcher(cache_db=db)
        mock_data = InternationalData(
            timestamp="2026-01-01",
            metrics={
                "EFA": MomentumMetrics(
                    symbol="EFA", price=75.0, momentum_1m=0.02, momentum_3m=-0.01,
                    momentum_6m=0.05, volatility_20d=0.15, sharpe_6m=0.33,
                    timestamp="2026-01-01",
                ),
                "EEM": MomentumMetrics(
                    symbol="EEM", price=40.0, momentum_1m=0.03, momentum_3m=0.01,
                    momentum_6m=-0.02, volatility_20d=0.20, sharpe_6m=-0.10,
                    timestamp="2026-01-01",
                ),
                "SPY": MomentumMetrics(
                    symbol="SPY", price=500.0, momentum_1m=0.01, momentum_3m=0.02,
                    momentum_6m=0.03, volatility_20d=0.12, sharpe_6m=0.25,
                    timestamp="2026-01-01",
                ),
            },
            relative=RelativeMomentum(
                symbol="relative_momentum", efa_momentum_6m=0.05, eem_momentum_6m=-0.02,
                spy_momentum_6m=0.03, efa_vs_spy=0.02, eem_vs_spy=-0.05,
                signal="efa_lead", confidence=0.2, timestamp="2026-01-01",
            ),
            data_fresh=True,
        )
        with patch.object(fetcher, "fetch_all", return_value=mock_data):
            summary = fetcher.get_signal_summary()
            assert summary["signal"] == "efa_lead"
            assert summary["data_fresh"] is True
            assert "prices" in summary
            assert summary["prices"]["EFA"] == 75.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_momentum_windows(self):
        assert MOMENTUM_WINDOWS == [21, 63, 126]

    def test_symbols(self):
        assert "EFA" in SYMBOLS
        assert "EEM" in SYMBOLS
        assert "SPY" in SYMBOLS

    def test_cache_ttl(self):
        assert CACHE_TTL_HOURS > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
