"""
Tests for src/data/pipeline.py — Data Pipeline.

Covers: init_db, fetch_yahoo, detect_regime, check_data_quality,
and the symbolic constants.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from src.data import pipeline as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_symbols_core(self):
        assert "SPY" in pl.SYMBOLS["core"]
        assert "GLD" in pl.SYMBOLS["core"]
        assert "TLT" in pl.SYMBOLS["core"]

    def test_symbols_risk_indicators(self):
        assert "^VIX" in pl.SYMBOLS["risk_indicators"]

    def test_symbols_alternatives(self):
        assert "BTC-USD" in pl.SYMBOLS["alternatives"]

    def test_symbols_factors(self):
        assert "MTUM" in pl.SYMBOLS["factors"]

    def test_all_symbols_flat(self):
        assert "SPY" in pl.ALL_SYMBOLS
        assert "BTC-USD" in pl.ALL_SYMBOLS
        assert "^VIX" in pl.ALL_SYMBOLS
        assert len(pl.ALL_SYMBOLS) == sum(len(v) for v in pl.SYMBOLS.values())


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

class TestInitDB:
    def test_creates_database(self, tmp_path):
        """Creates a valid SQLite database file."""
        db_path = tmp_path / "market.db"
        with patch.object(pl, "DB_PATH", db_path), \
             patch.object(pl, "DATA_DIR", tmp_path):
            conn = pl.init_db()
            assert db_path.exists()
            conn.close()

    def test_creates_prices_table(self, tmp_path):
        """Creates the prices table with correct schema."""
        db_path = tmp_path / "market.db"
        with patch.object(pl, "DB_PATH", db_path), \
             patch.object(pl, "DATA_DIR", tmp_path):
            conn = pl.init_db()
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prices'")
            assert cursor.fetchone() is not None
            conn.close()

    def test_creates_regime_log_table(self, tmp_path):
        """Creates the regime_log table."""
        db_path = tmp_path / "market.db"
        with patch.object(pl, "DB_PATH", db_path), \
             patch.object(pl, "DATA_DIR", tmp_path):
            conn = pl.init_db()
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='regime_log'")
            assert cursor.fetchone() is not None
            conn.close()

    def test_creates_data_quality_table(self, tmp_path):
        """Creates the data_quality table."""
        db_path = tmp_path / "market.db"
        with patch.object(pl, "DB_PATH", db_path), \
             patch.object(pl, "DATA_DIR", tmp_path):
            conn = pl.init_db()
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data_quality'")
            assert cursor.fetchone() is not None
            conn.close()

    def test_idempotent(self, tmp_path):
        """Calling init_db multiple times does not raise."""
        db_path = tmp_path / "market.db"
        with patch.object(pl, "DB_PATH", db_path), \
             patch.object(pl, "DATA_DIR", tmp_path):
            c1 = pl.init_db()
            c2 = pl.init_db()  # should not raise
            c1.close()
            c2.close()

    def test_data_dir_created(self, tmp_path):
        """Creates the data directory if missing."""
        db_path = tmp_path / "sub" / "market.db"
        with patch.object(pl, "DB_PATH", db_path), \
             patch.object(pl, "DATA_DIR", tmp_path / "sub"):
            conn = pl.init_db()
            assert (tmp_path / "sub").exists()
            conn.close()


# ---------------------------------------------------------------------------
# fetch_yahoo
# ---------------------------------------------------------------------------

class _AsyncCM:
    """Async context manager for mocking aiohttp responses."""
    def __init__(self, response):
        self._response = response
    async def __aenter__(self):
        return self._response
    async def __aexit__(self, *args):
        pass

class TestFetchYahoo:
    @pytest.fixture
    def mock_response_ok(self):
        """Build a mock aiohttp response returning valid chart data."""
        ts = [1700000000 + 86400 * i for i in range(5)]

        class MockResp:
            status = 200
            async def json(self):
                return {
                    "chart": {
                        "result": [{
                            "timestamp": ts,
                            "indicators": {
                                "adjclose": [{"adjclose": [150.0, 151.0, 152.0, 153.0, 154.0]}],
                                "quote": [{
                                    "open": [149.5, 150.5, 151.5, 152.5, 153.5],
                                    "high": [150.5, 151.5, 152.5, 153.5, 154.5],
                                    "low": [149.0, 150.0, 151.0, 152.0, 153.0],
                                    "close": [150.0, 151.0, 152.0, 153.0, 154.0],
                                    "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
                                }]
                            }
                        }]
                    }
                }

        return MockResp()

    @pytest.fixture
    def mock_response_no_adjclose(self):
        """Response without adjclose — falls back to close."""
        ts = [1700000000 + 86400 * i for i in range(3)]

        class MockResp:
            status = 200
            async def json(self):
                return {
                    "chart": {
                        "result": [{
                            "timestamp": ts,
                            "indicators": {
                                "adjclose": [{}],
                                "quote": [{
                                    "open": [100.0, 101.0, 102.0],
                                    "high": [102.0, 103.0, 104.0],
                                    "low": [99.0, 100.0, 101.0],
                                    "close": [100.0, 101.0, 102.0],
                                    "volume": [500000, 550000, 600000],
                                }]
                            }
                        }]
                    }
                }

        return MockResp()

    @pytest.fixture
    def mock_response_empty(self):
        """Response with no result."""
        class MockResp:
            status = 200
            async def json(self):
                return {"chart": {"result": []}}
        return MockResp()

    @pytest.fixture
    def mock_response_non200(self):
        """Non-200 response returns empty list."""
        class MockResp:
            status = 404
            async def json(self):
                return {}
        return MockResp()

    def _make_async_cm(self, resp):
        """Build an async context manager that returns resp."""
        return _AsyncCM(resp)

    def _make_session(self, resp=None, raise_on_get=False):
        """Build a mock session whose .get() returns an async CM."""
        class MockSession:
            def __init__(self, response, raise_err):
                self._response = response
                self._raise_err = raise_err
            def get(self, *args, **kwargs):
                if self._raise_err:
                    raise Exception("Connection error")
                return _AsyncCM(self._response) if self._response else _AsyncCM(None)
        return MockSession(resp, raise_on_get)

    @pytest.mark.anyio
    async def test_returns_parsed_records(self, mock_response_ok):
        """Returns a list of parsed price records."""
        session = self._make_session(mock_response_ok)

        records = await pl.fetch_yahoo("SPY", session)

        assert len(records) == 5
        assert records[0]["close"] == 150.0

    @pytest.mark.anyio
    async def test_each_record_has_required_fields(self, mock_response_ok):
        """Each record has date, open, high, low, close, volume."""
        session = self._make_session(mock_response_ok)

        records = await pl.fetch_yahoo("SPY", session)

        for r in records:
            assert "date" in r
            assert "open" in r
            assert "high" in r
            assert "low" in r
            assert "close" in r
            assert "volume" in r

    @pytest.mark.anyio
    async def test_date_format(self, mock_response_ok):
        """Date field is YYYY-MM-DD."""
        session = self._make_session(mock_response_ok)

        records = await pl.fetch_yahoo("SPY", session)
        for r in records:
            assert len(r["date"]) == 10
            assert r["date"][4] == "-"
            assert r["date"][7] == "-"

    @pytest.mark.anyio
    async def test_falls_back_to_close_when_no_adjclose(self, mock_response_no_adjclose):
        """Uses close price when adjclose is unavailable."""
        session = self._make_session(mock_response_no_adjclose)

        records = await pl.fetch_yahoo("SPY", session)
        assert len(records) == 3
        assert records[0]["close"] == 100.0

    @pytest.mark.anyio
    async def test_returns_empty_for_empty_result(self, mock_response_empty):
        """Returns empty list when API returns no result."""
        session = self._make_session(mock_response_empty)

        records = await pl.fetch_yahoo("SPY", session)
        assert records == []

    @pytest.mark.anyio
    async def test_returns_empty_for_non200(self, mock_response_non200):
        """Returns empty list when API returns non-200."""
        session = self._make_session(mock_response_non200)

        records = await pl.fetch_yahoo("SPY", session)
        assert records == []

    @pytest.mark.anyio
    async def test_handles_network_error(self):
        """Returns empty list on network error."""
        session = self._make_session(raise_on_get=True)

        records = await pl.fetch_yahoo("SPY", session)
        assert records == []

    @pytest.mark.anyio
    async def test_handles_null_prices_in_response(self):
        """Skips records with null close prices."""
        ts = [1700000000]

        class MockResp:
            status = 200
            async def json(self):
                return {
                    "chart": {
                        "result": [{
                            "timestamp": ts,
                            "indicators": {
                                "adjclose": [{"adjclose": [None]}],
                                "quote": [{
                                    "open": [None], "high": [None], "low": [None],
                                    "close": [None], "volume": [None],
                                }]
                            }
                        }]
                    }
                }

        session = self._make_session(MockResp())

        records = await pl.fetch_yahoo("SPY", session)
        assert records == []


# ---------------------------------------------------------------------------
# detect_regime
# ---------------------------------------------------------------------------

class TestDetectRegime:
    def _make_conn(self, tmp_path, prices_rows):
        """Helper: create in-memory DB with prices and regime_log tables."""
        conn = sqlite3.connect(str(tmp_path / "m.db"))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT, date TEXT, close REAL,
                PRIMARY KEY (symbol, date)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_log (
                id INTEGER PRIMARY KEY,
                date TEXT, regime TEXT, vix_level REAL,
                detected_at TEXT
            )
        """)
        for row in prices_rows:
            conn.execute("INSERT OR IGNORE INTO prices (symbol, date, close) VALUES (?, ?, ?)", row)
        conn.commit()
        return conn

    def test_crisis_when_vix_above_30(self, tmp_path):
        """Returns 'crisis' when VIX > 30."""
        today = datetime.now().strftime("%Y-%m-%d")
        rows = []
        # 63 days of VIX > 30 and SPY data
        for i in range(63):
            d = (datetime.now() - timedelta(days=62 - i)).strftime("%Y-%m-%d")
            rows.append(("VIX", d, 35.0 + (i % 5)))
            rows.append(("SPY", d, 400.0 - i * 0.5))
        conn = self._make_conn(tmp_path, rows)

        with patch.object(pl, "DB_PATH", tmp_path / "m.db"):
            regime = pl.detect_regime(conn)
            assert regime == "crisis"

    def test_vol_spike_when_vix_above_ma20_times_1_5(self, tmp_path):
        """Returns 'vol_spike' when VIX spikes above 1.5x its 20-day MA."""
        rows = []
        for i in range(63):
            d = (datetime.now() - timedelta(days=62 - i)).strftime("%Y-%m-%d")
            # First 61 days VIX=12, last 2 days VIX=26 (sudden spike)
            vix = 26.0 if i >= 61 else 12.0
            rows.append(("VIX", d, vix))
            rows.append(("SPY", d, 400.0 - i * 0.3))
        conn = self._make_conn(tmp_path, rows)

        with patch.object(pl, "DB_PATH", tmp_path / "m.db"):
            regime = pl.detect_regime(conn)
            assert regime == "vol_spike"

    def test_low_vol_when_vix_below_15(self, tmp_path):
        """Returns 'low_vol' when VIX < 15."""
        rows = []
        for i in range(63):
            d = (datetime.now() - timedelta(days=62 - i)).strftime("%Y-%m-%d")
            rows.append(("VIX", d, 12.0 + (i % 3)))
            rows.append(("SPY", d, 410.0 + i * 0.2))
        conn = self._make_conn(tmp_path, rows)

        with patch.object(pl, "DB_PATH", tmp_path / "m.db"):
            regime = pl.detect_regime(conn)
            assert regime == "low_vol"

    def test_normal_otherwise(self, tmp_path):
        """Returns 'normal' when VIX is between 15 and 30 without spike."""
        rows = []
        for i in range(63):
            d = (datetime.now() - timedelta(days=62 - i)).strftime("%Y-%m-%d")
            rows.append(("VIX", d, 18.0 + (i % 5)))
            rows.append(("SPY", d, 400.0 + i * 0.1))
        conn = self._make_conn(tmp_path, rows)

        with patch.object(pl, "DB_PATH", tmp_path / "m.db"):
            regime = pl.detect_regime(conn)
            assert regime == "normal"

    def test_returns_none_when_insufficient_data(self, tmp_path):
        """Returns None when fewer than 20 rows."""
        conn = self._make_conn(tmp_path, [("SPY", "2026-05-20", 400.0)])
        with patch.object(pl, "DB_PATH", tmp_path / "m.db"):
            result = pl.detect_regime(conn)
            assert result is None

    def test_returns_none_when_missing_vix(self, tmp_path):
        """Returns None when no VIX data."""
        rows = []
        for i in range(63):
            d = (datetime.now() - timedelta(days=62 - i)).strftime("%Y-%m-%d")
            rows.append(("SPY", d, 400.0))
        conn = self._make_conn(tmp_path, rows)
        with patch.object(pl, "DB_PATH", tmp_path / "m.db"):
            result = pl.detect_regime(conn)
            assert result is None

    def test_returns_none_when_missing_spy(self, tmp_path):
        """Returns None when no SPY data."""
        rows = []
        for i in range(63):
            d = (datetime.now() - timedelta(days=62 - i)).strftime("%Y-%m-%d")
            rows.append(("VIX", d, 18.0))
        conn = self._make_conn(tmp_path, rows)
        with patch.object(pl, "DB_PATH", tmp_path / "m.db"):
            result = pl.detect_regime(conn)
            assert result is None

    def test_logs_to_regime_log_table(self, tmp_path):
        """Detection is logged to regime_log table."""
        rows = []
        for i in range(63):
            d = (datetime.now() - timedelta(days=62 - i)).strftime("%Y-%m-%d")
            rows.append(("VIX", d, 12.0))
            rows.append(("SPY", d, 410.0))
        conn = self._make_conn(tmp_path, rows)
        with patch.object(pl, "DB_PATH", tmp_path / "m.db"):
            pl.detect_regime(conn)
        cursor = conn.execute("SELECT regime, vix_level FROM regime_log ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "low_vol"
        assert row[1] == 12.0


# ---------------------------------------------------------------------------
# check_data_quality
# ---------------------------------------------------------------------------

class TestCheckDataQuality:
    def test_returns_dict_with_all_symbols(self, tmp_path):
        """Returns quality dict for all symbols."""
        conn = sqlite3.connect(str(tmp_path / "m.db"))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT, date TEXT, close REAL, updated_at TEXT,
                PRIMARY KEY (symbol, date)
            )
        """)
        conn.execute(
            "INSERT INTO prices (symbol, date, close, updated_at) VALUES (?, ?, ?, ?)",
            ("SPY", datetime.now().strftime("%Y-%m-%d"), 400.0, datetime.now().isoformat()),
        )
        conn.commit()

        quality = pl.check_data_quality(conn)
        assert isinstance(quality, dict)
        for sym in pl.ALL_SYMBOLS:
            assert sym in quality

    def test_staleness_detected(self, tmp_path):
        """Symbols with old data are marked needs_refresh=True."""
        conn = sqlite3.connect(str(tmp_path / "m.db"))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT, date TEXT, close REAL, updated_at TEXT,
                PRIMARY KEY (symbol, date)
            )
        """)
        old_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        conn.execute(
            "INSERT INTO prices (symbol, date, close, updated_at) VALUES (?, ?, ?, ?)",
            ("SPY", old_date, 400.0, (datetime.now() - timedelta(days=30)).isoformat()),
        )
        conn.commit()

        quality = pl.check_data_quality(conn)
        assert quality["SPY"]["needs_refresh"] is True
        assert quality["SPY"]["staleness_hours"] > 24

    def test_fresh_data_no_refresh_needed(self, tmp_path):
        """Recently updated symbols are marked needs_refresh=False."""
        conn = sqlite3.connect(str(tmp_path / "m.db"))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT, date TEXT, close REAL, updated_at TEXT,
                PRIMARY KEY (symbol, date)
            )
        """)
        today = datetime.now().strftime("%Y-%m-%d")
        conn.execute(
            "INSERT INTO prices (symbol, date, close, updated_at) VALUES (?, ?, ?, ?)",
            ("SPY", today, 400.0, datetime.now().isoformat()),
        )
        conn.commit()

        quality = pl.check_data_quality(conn)
        assert quality["SPY"]["needs_refresh"] is False

    def test_none_last_date_becomes_stale(self, tmp_path):
        """Symbols with no data have high staleness."""
        conn = sqlite3.connect(str(tmp_path / "m.db"))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT, date TEXT, close REAL, updated_at TEXT,
                PRIMARY KEY (symbol, date)
            )
        """)
        conn.commit()

        quality = pl.check_data_quality(conn)
        assert quality["SPY"]["last_date"] is None
        assert quality["SPY"]["staleness_hours"] == 9999
        assert quality["SPY"]["needs_refresh"] is True

    def test_record_counts(self, tmp_path):
        """Returns correct record count per symbol."""
        conn = sqlite3.connect(str(tmp_path / "m.db"))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT, date TEXT, close REAL, updated_at TEXT,
                PRIMARY KEY (symbol, date)
            )
        """)
        for i in range(5):
            d = (datetime.now() - timedelta(days=4 - i)).strftime("%Y-%m-%d")
            conn.execute(
                "INSERT OR IGNORE INTO prices (symbol, date, close, updated_at) VALUES (?, ?, ?, ?)",
                ("SPY", d, 400.0 + i, datetime.now().isoformat()),
            )
        conn.commit()

        quality = pl.check_data_quality(conn)
        assert quality["SPY"]["records"] == 5


# ---------------------------------------------------------------------------
# main entry point (smoke test, no network calls)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_main_handles_no_fetch_needed(monkeypatch, tmp_path, capsys):
    """When all data is fresh, main skips fetching."""
    db_path = tmp_path / "market.db"
    monkeypatch.setattr(pl, "DB_PATH", db_path)
    monkeypatch.setattr(pl, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pl, "CACHE_DIR", tmp_path / "cache")

    conn = pl.init_db()
    # Mark ALL symbols as fresh so no fetch happens
    today = datetime.now().strftime("%Y-%m-%d")
    for sym in pl.ALL_SYMBOLS:
        conn.execute(
            "INSERT INTO prices (symbol, date, close, updated_at) VALUES (?, ?, ?, ?)",
            (sym, today, 100.0, datetime.now().isoformat()),
        )
    conn.commit()
    conn.close()

    await pl.main()
    captured = capsys.readouterr()
    assert "All data fresh" in captured.out
