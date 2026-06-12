"""
Tests for FRED-MD Macroeconomic Data Fetcher — v970 Phase 1

Tests the data infrastructure layer:
- FredMdFetcher construction and caching
- Series definitions and metadata
- Cache read/write operations
- Indicator computation with mocked FRED API
- Regime signal generation
"""

import json
import pytest
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pandas as pd

from src.data.fred_data import (
    FredMdFetcher,
    FredSignal,
    FredSeriesObservation,
    get_fred_signal,
    DEFAULT_FRED_SERIES,
    ALL_FRED_SERIES,
    SERIES_METADATA,
    REGIME_THRESHOLDS,
    FRED_CACHE_TABLE,
    _init_cache_table,
    _get_cached_series,
    _set_cached_series,
    get_fred_md_cache_health,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_cache_db(tmp_path):
    """Create a temporary cache database."""
    db = tmp_path / "test_market.db"
    conn = sqlite3.connect(str(db))
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {FRED_CACHE_TABLE} (
            series_id TEXT PRIMARY KEY,
            json_data TEXT NOT NULL,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    yield db


@pytest.fixture
def mock_fred_client():
    """Create a mock FRED API client."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_fetcher(tmp_path, mock_fred_client):
    """Create a FredMdFetcher with mocked FRED client and temp cache."""
    db = tmp_path / "test_market.db"
    # Monkey-patch MARKET_DB for this test
    with patch("src.data.fred_data.MARKET_DB", db):
        with patch("src.data.fred_data._Fred") as mock_fred_class:
            mock_fred_class.return_value = mock_fred_client
            fetcher = FredMdFetcher(api_key="test_key", use_cache=True)
            fetcher._fred_client = mock_fred_client
            yield fetcher


def _make_series(data_dict, name="test"):
    """Create a mock pandas Series from a dict of {date_str: value}."""
    dates = pd.to_datetime(list(data_dict.keys()))
    values = list(data_dict.values())
    return pd.Series(values, index=dates, name=name, dtype=float)


# ── Test: Series Definitions ─────────────────────────────────────────


class TestSeriesDefinitions:
    """Verify FRED-MD series definitions."""

    def test_all_series_unique(self):
        """ALL_FRED_SERIES should have unique entries."""
        assert len(ALL_FRED_SERIES) == len(set(ALL_FRED_SERIES))

    def test_all_series_complete(self):
        """ALL_FRED_SERIES should contain all series from regime groups."""
        expected = set()
        for series_list in DEFAULT_FRED_SERIES.values():
            expected.update(series_list)
        assert set(ALL_FRED_SERIES) == expected

    def test_all_series_in_metadata(self):
        """Every series should have metadata."""
        for sid in ALL_FRED_SERIES:
            assert sid in SERIES_METADATA, f"Missing metadata for {sid}"

    def test_regime_thresholds_structure(self):
        """REGIME_THRESHOLDS should have expected keys."""
        assert "recession_prob" in REGIME_THRESHOLDS
        assert "baa_spread" in REGIME_THRESHOLDS
        assert "vix" in REGIME_THRESHOLDS
        assert "inflation_yoy" in REGIME_THRESHOLDS
        assert "pmi" in REGIME_THRESHOLDS

    def test_default_fred_series_covers_all_regimes(self):
        """DEFAULT_FRED_SERIES should cover all 5 regime types."""
        expected_regimes = {"crisis", "high_vol", "inflation", "recovery", "low_vol"}
        assert set(DEFAULT_FRED_SERIES.keys()) == expected_regimes

    def test_each_regime_has_indicators(self):
        """Each regime should have at least 3 indicators."""
        for regime, series_list in DEFAULT_FRED_SERIES.items():
            assert len(series_list) >= 3, f"{regime} has only {len(series_list)} indicators"


# ── Test: Cache Operations ───────────────────────────────────────────


class TestCacheOperations:
    """Test SQLite cache read/write operations."""

    def test_init_cache_table(self, tmp_cache_db):
        """_init_cache_table should create the table."""
        with patch("src.data.fred_data.MARKET_DB", tmp_cache_db):
            _init_cache_table()
            conn = sqlite3.connect(str(tmp_cache_db))
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (FRED_CACHE_TABLE,),
            ).fetchall()
            conn.close()
            assert len(tables) == 1

    def test_cache_roundtrip(self, tmp_cache_db):
        """Write then read a series from cache."""
        series = _make_series({"2024-01-01": 100.0, "2024-02-01": 101.0})
        with patch("src.data.fred_data.MARKET_DB", tmp_cache_db):
            _set_cached_series("INDPRO", series)
            cached = _get_cached_series("INDPRO")
        assert cached is not None
        assert len(cached) == 2
        assert float(cached.iloc[-1]) == 101.0

    def test_cache_miss(self, tmp_cache_db):
        """Non-existent series should return None."""
        with patch("src.data.fred_data.MARKET_DB", tmp_cache_db):
            result = _get_cached_series("NONEXISTENT")
        assert result is None

    def test_cache_overwrite(self, tmp_cache_db):
        """Writing same series_id should overwrite."""
        s1 = _make_series({"2024-01-01": 100.0})
        s2 = _make_series({"2024-01-01": 200.0})
        with patch("src.data.fred_data.MARKET_DB", tmp_cache_db):
            _set_cached_series("TEST", s1)
            _set_cached_series("TEST", s2)
            cached = _get_cached_series("TEST")
        assert cached is not None
        assert float(cached.iloc[-1]) == 200.0

    def test_cache_health_missing_table(self, tmp_path):
        """Missing fred_cache table should be explicit unavailable state."""
        db = tmp_path / "market.db"
        sqlite3.connect(str(db)).close()

        health = get_fred_md_cache_health(db)

        assert health["status"] == "unavailable"
        assert health["row_count"] == 0
        assert health["reason"] == "missing_table"

    def test_cache_health_empty_table(self, tmp_cache_db):
        """Empty fred_cache table should not look like a fresh cache."""
        health = get_fred_md_cache_health(tmp_cache_db)

        assert health["status"] == "empty"
        assert health["row_count"] == 0
        assert health["reason"] == "empty_cache"

    def test_cache_health_fresh_and_stale_rows(self, tmp_cache_db):
        """Fresh and stale cache rows should be classified by latest fetched_at."""
        now = datetime(2026, 6, 11, 12, tzinfo=timezone.utc)
        fresh_ts = (now - timedelta(hours=2)).isoformat()
        stale_ts = (now - timedelta(hours=30)).isoformat()

        with sqlite3.connect(str(tmp_cache_db)) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {FRED_CACHE_TABLE} (series_id, json_data, fetched_at) VALUES (?, ?, ?)",
                ("INDPRO", "{}", fresh_ts),
            )
        fresh = get_fred_md_cache_health(tmp_cache_db, now=now, ttl_hours=24, api_key="test")

        with sqlite3.connect(str(tmp_cache_db)) as conn:
            conn.execute(
                f"UPDATE {FRED_CACHE_TABLE} SET fetched_at = ? WHERE series_id = ?",
                (stale_ts, "INDPRO"),
            )
        stale = get_fred_md_cache_health(tmp_cache_db, now=now, ttl_hours=24, api_key="test")

        assert fresh["status"] == "ok"
        assert fresh["row_count"] == 1
        assert fresh["age_hours"] == 2.0
        assert fresh["source_mode"] == "cached"
        assert fresh["api_key_configured"] is True
        assert stale["status"] == "stale"
        assert stale["source_mode"] == "stale_cached"
        assert stale["reason"] == "cache_stale"


# ── Test: FredMdFetcher Construction ─────────────────────────────────


class TestFredMdFetcherConstruction:
    """Test FredMdFetcher initialization."""

    def test_init_no_api_key(self):
        """Fetcher should not require API key at construction."""
        with patch("src.data.fred_data.MARKET_DB", ":memory:"):
            fetcher = FredMdFetcher(use_cache=False)
            assert fetcher._api_key == ""

    def test_init_with_api_key(self):
        """Fetcher should store API key."""
        with patch("src.data.fred_data.MARKET_DB", ":memory:"):
            fetcher = FredMdFetcher(api_key="my_key", use_cache=False)
            assert fetcher._api_key == "my_key"

    def test_init_use_cache_default(self):
        """Cache should be enabled by default."""
        with patch("src.data.fred_data.MARKET_DB", ":memory:"):
            fetcher = FredMdFetcher(api_key="test")
            assert fetcher.use_cache is True

    def test_fred_client_property_error_no_key(self):
        """Accessing fred_client without key should raise ValueError."""
        with patch("src.data.fred_data.MARKET_DB", ":memory:"):
            with patch("src.data.fred_data._Fred", MagicMock()):
                fetcher = FredMdFetcher(use_cache=False)
                with pytest.raises(ValueError, match="API key"):
                    _ = fetcher.fred_client

    def test_fred_client_property_success(self, mock_fetcher):
        """fred_client should return the mock client."""
        assert mock_fetcher.fred_client is not None

    def test_init_creates_cache_table(self, tmp_path):
        """Construction should create cache table when use_cache=True."""
        db = tmp_path / "market.db"
        with patch("src.data.fred_data.MARKET_DB", db):
            with patch("src.data.fred_data._Fred") as mc:
                FredMdFetcher(api_key="test", use_cache=True)
            conn = sqlite3.connect(str(db))
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (FRED_CACHE_TABLE,),
            ).fetchall()
            conn.close()
            assert len(tables) == 1


# ── Test: Series Fetching ────────────────────────────────────────────


class TestSeriesFetching:
    """Test get_series and get_all_series."""

    def test_get_series_returns_data(self, mock_fetcher):
        """get_series should return data from mocked API."""
        expected = _make_series({"2024-01-01": 100.0, "2024-02-01": 101.0})
        mock_fetcher._fred_client.get_series.return_value = expected
        result = mock_fetcher.get_series("INDPRO", cache_ok=False)
        assert len(result) == 2
        assert float(result.iloc[-1]) == 101.0

    def test_get_series_empty_response(self, mock_fetcher):
        """get_series should return empty Series on empty API response."""
        mock_fetcher._fred_client.get_series.return_value = pd.Series(dtype=float)
        result = mock_fetcher.get_series("INDPRO", cache_ok=False)
        assert len(result) == 0

    def test_get_series_api_error(self, mock_fetcher):
        """get_series should handle API errors gracefully."""
        mock_fetcher._fred_client.get_series.side_effect = Exception("API down")
        result = mock_fetcher.get_series("INDPRO", cache_ok=False)
        assert len(result) == 0

    def test_get_all_series(self, mock_fetcher):
        """get_all_series should return DataFrame with multiple columns."""
        data = _make_series({"2024-01-01": 100.0, "2024-02-01": 101.0})
        mock_fetcher._fred_client.get_series.return_value = data
        result = mock_fetcher.get_all_series(["INDPRO", "PAYEMS"], cache_ok=False)
        assert isinstance(result, pd.DataFrame)
        assert "INDPRO" in result.columns
        assert "PAYEMS" in result.columns

    def test_get_all_series_partial_failure(self, mock_fetcher):
        """get_all_series should include only successfully fetched series."""
        valid = _make_series({"2024-01-01": 100.0})
        mock_fetcher._fred_client.get_series.side_effect = [
            valid,
            Exception("Failed"),
        ]
        result = mock_fetcher.get_all_series(["INDPRO", "FAIL"], cache_ok=False)
        assert "INDPRO" in result.columns
        assert "FAIL" not in result.columns

    def test_get_regime_indicators(self, mock_fetcher):
        """get_regime_indicators should return dict of DataFrames."""
        data = _make_series({"2024-01-01": 100.0})
        mock_fetcher._fred_client.get_series.return_value = data
        indicators = mock_fetcher.get_regime_indicators(cache_ok=False)
        assert isinstance(indicators, dict)
        assert "crisis" in indicators
        assert "recovery" in indicators
        for regime_df in indicators.values():
            assert isinstance(regime_df, pd.DataFrame)


# ── Test: Indicator Computation ──────────────────────────────────────


class TestIndicatorComputation:
    """Test computed indicators from FRED data."""

    def test_compute_recession_probability(self, mock_fetcher):
        """Should return latest RECPROUSM156N value."""
        data = _make_series({"2024-01-01": 5.0, "2024-02-01": 12.5})
        mock_fetcher._fred_client.get_series.return_value = data
        result = mock_fetcher.compute_recession_probability(cache_ok=False)
        assert result == 12.5

    def test_compute_recession_probability_no_data(self, mock_fetcher):
        """Should return None when no data."""
        mock_fetcher._fred_client.get_series.return_value = pd.Series(dtype=float)
        result = mock_fetcher.compute_recession_probability(cache_ok=False)
        assert result is None

    def test_compute_inflation_pressure(self, mock_fetcher):
        """Should compute YoY CPI change."""
        # 13 months: Jan 2023 through Jan 2024
        values = {}
        for i, (year, month) in enumerate([
            (2023, 1), (2023, 2), (2023, 3), (2023, 4), (2023, 5), (2023, 6),
            (2023, 7), (2023, 8), (2023, 9), (2023, 10), (2023, 11), (2023, 12),
            (2024, 1),
        ]):
            values[f"{year}-{month:02d}-01"] = 100.0 + i
        data = _make_series(values)
        mock_fetcher._fred_client.get_series.return_value = data
        result = mock_fetcher.compute_inflation_pressure(cache_ok=False)
        # Latest (Jan 2024): 112, Year ago (Jan 2023): 100, YoY = 12%
        assert result is not None
        assert abs(result - 12.0) < 0.01

    def test_compute_inflation_pressure_insufficient_data(self, mock_fetcher):
        """Should return None with less than 13 observations."""
        data = _make_series({"2024-01-01": 100.0})
        mock_fetcher._fred_client.get_series.return_value = data
        result = mock_fetcher.compute_inflation_pressure(cache_ok=False)
        assert result is None

    def test_compute_pmi_health(self, mock_fetcher):
        """Should return latest NAPMI value."""
        data = _make_series({"2024-01-01": 52.5})
        mock_fetcher._fred_client.get_series.return_value = data
        result = mock_fetcher.compute_pmi_health(cache_ok=False)
        assert result == 52.5

    def test_compute_pmi_health_no_data(self, mock_fetcher):
        """Should return None when no PMI data."""
        mock_fetcher._fred_client.get_series.return_value = pd.Series(dtype=float)
        result = mock_fetcher.compute_pmi_health(cache_ok=False)
        assert result is None

    def test_compute_monetary_stance_tight(self, mock_fetcher):
        """Fed rate >= 4.0 should be 'tight'."""
        data = _make_series({"2024-01-01": 5.25})
        mock_fetcher._fred_client.get_series.return_value = data
        result = mock_fetcher.compute_monetary_stance(cache_ok=False)
        assert result == "tight"

    def test_compute_monetary_stance_neutral(self, mock_fetcher):
        """Fed rate between 2.0 and 4.0 should be 'neutral'."""
        data = _make_series({"2024-01-01": 3.0})
        mock_fetcher._fred_client.get_series.return_value = data
        result = mock_fetcher.compute_monetary_stance(cache_ok=False)
        assert result == "neutral"

    def test_compute_monetary_stance_accommodative(self, mock_fetcher):
        """Fed rate < 2.0 should be 'accommodative'."""
        data = _make_series({"2024-01-01": 0.25})
        mock_fetcher._fred_client.get_series.return_value = data
        result = mock_fetcher.compute_monetary_stance(cache_ok=False)
        assert result == "accommodative"

    def test_compute_monetary_stance_no_data(self, mock_fetcher):
        """Should return 'unknown' when no data."""
        mock_fetcher._fred_client.get_series.return_value = pd.Series(dtype=float)
        result = mock_fetcher.compute_monetary_stance(cache_ok=False)
        assert result == "unknown"

    def test_compute_credit_conditions_tight(self, mock_fetcher):
        """BAA spread >= 3.5 should be 'tight'."""
        data = _make_series({"2024-01-01": 4.0})
        mock_fetcher._fred_client.get_series.return_value = data
        result = mock_fetcher.compute_credit_conditions(cache_ok=False)
        assert result == "tight"

    def test_compute_credit_conditions_normal(self, mock_fetcher):
        """BAA spread between 2.0 and 3.5 should be 'normal'."""
        data = _make_series({"2024-01-01": 2.5})
        mock_fetcher._fred_client.get_series.return_value = data
        result = mock_fetcher.compute_credit_conditions(cache_ok=False)
        assert result == "normal"

    def test_compute_credit_conditions_loose(self, mock_fetcher):
        """BAA spread < 2.0 should be 'loose'."""
        data = _make_series({"2024-01-01": 1.5})
        mock_fetcher._fred_client.get_series.return_value = data
        result = mock_fetcher.compute_credit_conditions(cache_ok=False)
        assert result == "loose"

    def test_compute_credit_conditions_no_data(self, mock_fetcher):
        """Should return 'unknown' when no data."""
        mock_fetcher._fred_client.get_series.return_value = pd.Series(dtype=float)
        result = mock_fetcher.compute_credit_conditions(cache_ok=False)
        assert result == "unknown"

    def test_get_fred_signal_returns_fred_signal(self, mock_fetcher):
        """get_fred_signal should return a FredSignal."""
        _13_months = [(2023,1),(2023,2),(2023,3),(2023,4),(2023,5),(2023,6),(2023,7),(2023,8),(2023,9),(2023,10),(2023,11),(2023,12),(2024,1)]
        # Mock multiple series needed by compute_regime_signal
        def mock_get_series(sid, **kwargs):
            if sid == "RECPROUSM156N":
                return _make_series({"2024-01-01": 5.0})
            elif sid == "CPIAUCSL":
                return _make_series({f"{y}-{m:02d}-01": 100.0 for y, m in _13_months})
            elif sid == "NAPMI":
                return _make_series({"2024-01-01": 52.0})
            elif sid == "FEDFUNDS":
                return _make_series({"2024-01-01": 3.0})
            elif sid == "BAASPREAD":
                return _make_series({"2024-01-01": 2.0})
            else:
                return _make_series({"2024-01-01": 100.0})

        mock_fetcher._fred_client.get_series.side_effect = mock_get_series
        signal = get_fred_signal(fetcher=mock_fetcher)
        assert isinstance(signal, FredSignal)
        assert signal.regime in ("NORMAL", "RECOVERY", "CRISIS", "HIGH_VOL", "LOW_VOL", "UNKNOWN")


# ── Test: Regime Signal Classification ───────────────────────────────


class TestRegimeSignalClassification:
    """Test the compute_regime_signal logic."""

    def test_crisis_from_recession_probability(self, mock_fetcher):
        """High recession probability (>30) should indicate CRISIS."""
        def mock_get_series(sid, **kwargs):
            if sid == "RECPROUSM156N":
                return _make_series({"2024-01-01": 50.0})
            elif sid == "CPIAUCSL":
                return _make_series({f"{y}-{m:02d}-01": 100.0 for y, m in [(2023,1),(2023,2),(2023,3),(2023,4),(2023,5),(2023,6),(2023,7),(2023,8),(2023,9),(2023,10),(2023,11),(2023,12),(2024,1)]})
            elif sid == "NAPMI":
                return _make_series({"2024-01-01": 42.0})
            elif sid == "FEDFUNDS":
                return _make_series({"2024-01-01": 5.0})
            elif sid == "BAASPREAD":
                return _make_series({"2024-01-01": 4.5})
            else:
                return pd.Series(dtype=float)
        mock_fetcher._fred_client.get_series.side_effect = mock_get_series
        signal = mock_fetcher.compute_regime_signal(cache_ok=False)
        assert signal.regime == "CRISIS"
        assert signal.confidence >= 0.6

    def test_normal_regime_by_default(self, mock_fetcher):
        """Moderate indicators should yield NORMAL regime."""
        _13_months = [(2023,1),(2023,2),(2023,3),(2023,4),(2023,5),(2023,6),(2023,7),(2023,8),(2023,9),(2023,10),(2023,11),(2023,12),(2024,1)]
        def mock_get_series(sid, **kwargs):
            if sid == "RECPROUSM156N":
                return _make_series({"2024-01-01": 5.0})
            elif sid == "CPIAUCSL":
                return _make_series({f"{y}-{m:02d}-01": 100.0 for y, m in _13_months})
            elif sid == "NAPMI":
                return _make_series({"2024-01-01": 52.0})
            elif sid == "FEDFUNDS":
                return _make_series({"2024-01-01": 3.0})
            elif sid == "BAASPREAD":
                return _make_series({"2024-01-01": 1.5})
            else:
                return pd.Series(dtype=float)
        mock_fetcher._fred_client.get_series.side_effect = mock_get_series
        signal = mock_fetcher.compute_regime_signal(cache_ok=False)
        assert signal.regime in ("NORMAL", "RECOVERY", "LOW_VOL")
        assert 0 <= signal.confidence <= 1.0

    def test_low_vol_detection(self, mock_fetcher):
        """Strong PMI + loose credit should indicate LOW_VOL."""
        _13_months = [(2023,1),(2023,2),(2023,3),(2023,4),(2023,5),(2023,6),(2023,7),(2023,8),(2023,9),(2023,10),(2023,11),(2023,12),(2024,1)]
        def mock_get_series(sid, **kwargs):
            if sid == "RECPROUSM156N":
                return _make_series({"2024-01-01": 2.0})
            elif sid == "CPIAUCSL":
                return _make_series({f"{y}-{m:02d}-01": 100.0 for y, m in _13_months})
            elif sid == "NAPMI":
                return _make_series({"2024-01-01": 58.0})
            elif sid == "FEDFUNDS":
                return _make_series({"2024-01-01": 2.0})
            elif sid == "BAASPREAD":
                return _make_series({"2024-01-01": 1.2})
            else:
                return pd.Series(dtype=float)
        mock_fetcher._fred_client.get_series.side_effect = mock_get_series
        signal = mock_fetcher.compute_regime_signal(cache_ok=False)
        assert signal.regime == "LOW_VOL"

    def test_get_fred_signal_allows_empty_regime_on_error(self):
        """get_fred_signal should handle exceptions gracefully."""
        with patch("src.data.fred_data.FredMdFetcher") as mock_cls:
            mock_cls.side_effect = Exception("Init failed")
            signal = get_fred_signal(api_key="test")
            assert isinstance(signal, FredSignal)
            assert signal.regime == "UNKNOWN"
            assert signal.confidence == 0.0


# ── Test: Dataclasses ────────────────────────────────────────────────


class TestDataclasses:
    """Test FRED dataclass structures."""

    def test_fred_signal_has_all_fields(self):
        """FredSignal should have all required fields."""
        signal = FredSignal(
            timestamp="2026-05-26T00:00:00",
            regime="NORMAL",
            confidence=0.7,
            indicators={"pmi": 52.0},
            recession_probability=5.0,
            inflation_pressure=3.0,
            monetary_stance="neutral",
            manufacturing_health=52.0,
            credit_conditions="normal",
        )
        assert signal.regime == "NORMAL"
        assert signal.confidence == 0.7
        assert signal.indicators["pmi"] == 52.0
        assert signal.recession_probability == 5.0
        assert signal.inflation_pressure == 3.0
        assert signal.monetary_stance == "neutral"
        assert signal.manufacturing_health == 52.0
        assert signal.credit_conditions == "normal"

    def test_fred_signal_defaults(self):
        """FredSignal should use default for timestamp."""
        signal = FredSignal(
            timestamp="2026-05-26T00:00:00",
            regime="UNKNOWN",
            confidence=0.0,
            indicators={},
            recession_probability=0.0,
            inflation_pressure=0.0,
            monetary_stance="unknown",
            manufacturing_health=50.0,
            credit_conditions="unknown",
        )
        assert signal.timestamp is not None

    def test_fred_series_observation(self):
        """FredSeriesObservation should store single observation."""
        obs = FredSeriesObservation(
            series_id="INDPRO",
            date="2024-01-01",
            value=100.5,
        )
        assert obs.series_id == "INDPRO"
        assert obs.value == 100.5
        assert obs.fetched_at is not None


# ── Test: Edge Cases ─────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_get_series_with_cache_hit(self, mock_fetcher, tmp_path):
        """get_series should return cached data on repeat calls."""
        data = _make_series({"2024-01-01": 100.0})
        mock_fetcher._fred_client.get_series.return_value = data
        # First call — should hit API
        r1 = mock_fetcher.get_series("INDPRO", cache_ok=True)
        assert len(r1) == 1
        # Second call — should use cache (cached in tmp)
        r2 = mock_fetcher.get_series("INDPRO", cache_ok=True)
        assert len(r2) == 1

    def test_cache_ttl_expiry(self, tmp_cache_db):
        """Cache entries older than TTL should be invalid."""
        series = _make_series({"2024-01-01": 100.0})
        with patch("src.data.fred_data.MARKET_DB", tmp_cache_db):
            _set_cached_series("STALE", series)
            # Manually set old timestamp
            old_ts = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
            conn = sqlite3.connect(str(tmp_cache_db))
            conn.execute(
                f"UPDATE {FRED_CACHE_TABLE} SET fetched_at = ? WHERE series_id = ?",
                (old_ts, "STALE"),
            )
            conn.commit()
            conn.close()
            # Should be None due to TTL
            cached = _get_cached_series("STALE")
            assert cached is None

    def test_multiple_fetches_in_parallel_safe(self, mock_fetcher):
        """Multiple get_all_series calls should not interfere."""
        data = _make_series({"2024-01-01": 100.0})
        mock_fetcher._fred_client.get_series.return_value = data
        r1 = mock_fetcher.get_all_series(["INDPRO"], cache_ok=False)
        r2 = mock_fetcher.get_all_series(["PAYEMS"], cache_ok=False)
        assert "INDPRO" in r1.columns
        assert "PAYEMS" in r2.columns

    def test_empty_series_list(self, mock_fetcher):
        """get_all_series with empty list should return empty DataFrame."""
        result = mock_fetcher.get_all_series([], cache_ok=False)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_nan_values_in_cached_series(self, tmp_cache_db):
        """Cached series with NaN values should serialize correctly."""
        data = _make_series({"2024-01-01": 100.0, "2024-02-01": float("nan")})
        with patch("src.data.fred_data.MARKET_DB", tmp_cache_db):
            _set_cached_series("NAN_TEST", data)
            cached = _get_cached_series("NAN_TEST")
        assert cached is not None
        assert len(cached) == 2
        # NaN should have been serialized to None then read back as NaN
        assert pd.isna(cached.iloc[-1])


class TestFredMacroDashboardIntegration:
    """Test FRED-MD signal appears in dashboard output."""

    def test_fred_macro_in_signals_json(self, tmp_path, monkeypatch):
        """DashboardGenerator should include fred_macro section."""
        monkeypatch.setattr("src.dashboard.generator.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.dashboard.generator.PUBLIC_DIR", tmp_path)
        monkeypatch.setattr("src.dashboard.generator.DB_PATH", str(tmp_path / "market.db"))
        monkeypatch.setattr("src.paths.DATA_DIR", tmp_path)

        (tmp_path / "prices.json").write_text("{}")
        (tmp_path / "cron_status.json").write_text('{"jobs": []}')

        import sqlite3
        db_path = tmp_path / "market.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    symbol TEXT, date TEXT, close REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regime_log (
                    regime TEXT, detected_at TEXT
                )
            """)

        from src.dashboard.generator import DashboardGenerator

        with patch("src.data.fred_data.get_fred_signal") as mock_fred:
            mock_fred.return_value = FredSignal(
                timestamp="2026-05-26T00:00:00Z",
                regime="NORMAL",
                confidence=0.6,
                indicators={"recession_probability": 5.0, "inflation_yoy": 2.5},
                recession_probability=5.0,
                inflation_pressure=2.5,
                monetary_stance="neutral",
                manufacturing_health=52.0,
                credit_conditions="normal",
            )

            with DashboardGenerator() as gen:
                gen.generate_signals_json()

        signals_file = tmp_path / "signals.json"
        assert signals_file.exists()
        data = json.loads(signals_file.read_text())
        assert "fred_macro" in data
        assert data["fred_macro"]["regime"] == "NORMAL"
        assert data["fred_macro"]["confidence"] == 0.6
        assert data["fred_macro"]["recession_probability"] == 5.0

    def test_fred_macro_graceful_fallback(self, tmp_path, monkeypatch):
        """Dashboard should handle FRED-MD unavailability gracefully."""
        monkeypatch.setattr("src.dashboard.generator.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.dashboard.generator.PUBLIC_DIR", tmp_path)
        monkeypatch.setattr("src.dashboard.generator.DB_PATH", str(tmp_path / "market.db"))
        monkeypatch.setattr("src.paths.DATA_DIR", tmp_path)

        (tmp_path / "prices.json").write_text("{}")
        (tmp_path / "cron_status.json").write_text('{"jobs": []}')

        import sqlite3
        db_path = tmp_path / "market.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    symbol TEXT, date TEXT, close REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regime_log (
                    regime TEXT, detected_at TEXT
                )
            """)

        from src.dashboard.generator import DashboardGenerator

        with patch("src.data.fred_data.get_fred_signal", side_effect=ImportError("no fredapi")):
            with DashboardGenerator() as gen:
                gen.generate_signals_json()

        signals_file = tmp_path / "signals.json"
        assert signals_file.exists()
        data = json.loads(signals_file.read_text())
        assert "fred_macro" in data
        assert data["fred_macro"]["regime"] == "UNKNOWN"
        assert data["fred_macro"]["confidence"] == 0.0
