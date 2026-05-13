#!/usr/bin/env python3
"""
Tests for defi_yield_fetcher.py — YieldData/YieldSpread dataclasses,
DeFiYieldDatabase storage, DeFiYieldMonitor spread calculation and alerts,
and CLI commands.
"""
import sys
import os
import json
import sqlite3
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from src.data.defi_yield_fetcher import (
    YieldData,
    YieldSpread,
    DeFiYieldFetcher,
    TreasuryYieldFetcher,
    DeFiYieldDatabase,
    DeFiYieldMonitor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yield_data(**overrides):
    defaults = dict(
        protocol="Lido",
        asset="stETH",
        yield_apy=0.035,
        tvl_usd=1e9,
        timestamp=datetime.utcnow().isoformat(),
        source="lido",
    )
    defaults.update(overrides)
    return YieldData(**defaults)


def _make_yield_spread(**overrides):
    defaults = dict(
        protocol="Lido",
        asset="stETH",
        defi_yield=0.035,
        treasury_yield=0.05,
        spread=-0.015,
        correlation_30d=None,
        signal="monitor",
        timestamp=datetime.utcnow().isoformat(),
    )
    defaults.update(overrides)
    return YieldSpread(**defaults)


# ---------------------------------------------------------------------------
# YieldData Tests
# ---------------------------------------------------------------------------

class TestYieldData:

    def test_fields(self):
        yd = _make_yield_data(protocol="Aave", asset="USDC", yield_apy=0.042)
        assert yd.protocol == "Aave"
        assert yd.asset == "USDC"
        assert yd.yield_apy == 0.042

    def test_to_dict(self):
        yd = _make_yield_data()
        # YieldData is a dataclass, so asdict works
        from dataclasses import asdict
        d = asdict(yd)
        assert "protocol" in d
        assert "yield_apy" in d


# ---------------------------------------------------------------------------
# YieldSpread Tests
# ---------------------------------------------------------------------------

class TestYieldSpread:

    def test_fields(self):
        ys = _make_yield_spread(signal="allocate", spread=0.025)
        assert ys.signal == "allocate"
        assert ys.spread == 0.025

    def test_signal_types(self):
        for sig in ["monitor", "consider", "allocate"]:
            ys = _make_yield_spread(signal=sig)
            assert ys.signal == sig


# ---------------------------------------------------------------------------
# DeFiYieldDatabase Tests
# ---------------------------------------------------------------------------

class TestDeFiYieldDatabase:

    def test_init_creates_tables(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = DeFiYieldDatabase(str(db_path))
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "yields" in tables
        assert "spreads" in tables

    def test_store_yield(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = DeFiYieldDatabase(str(db_path))
        yd = _make_yield_data()
        db.store_yield(yd)
        results = db.get_latest_yields(hours=1)
        assert len(results) == 1
        assert results[0]["protocol"] == "Lido"

    def test_store_spread(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = DeFiYieldDatabase(str(db_path))
        ys = _make_yield_spread()
        db.store_spread(ys)
        results = db.get_spread_history("Lido", days=1)
        assert len(results) == 1
        assert results[0]["signal"] == "monitor"

    def test_get_latest_yields_empty(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = DeFiYieldDatabase(str(db_path))
        results = db.get_latest_yields(hours=24)
        assert results == []

    def test_get_spread_history_empty(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = DeFiYieldDatabase(str(db_path))
        results = db.get_spread_history("Lido", days=30)
        assert results == []

    def test_store_multiple_yields(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = DeFiYieldDatabase(str(db_path))
        db.store_yield(_make_yield_data(protocol="Lido", yield_apy=0.035))
        db.store_yield(_make_yield_data(protocol="Aave", yield_apy=0.042))
        db.store_yield(_make_yield_data(protocol="Jito", yield_apy=0.065))
        results = db.get_latest_yields(hours=1)
        assert len(results) == 3

    def test_get_latest_yields_respects_hours(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = DeFiYieldDatabase(str(db_path))
        # Store an old yield (30 hours ago)
        old_ts = (datetime.utcnow() - timedelta(hours=30)).isoformat()
        db.store_yield(_make_yield_data(timestamp=old_ts))
        # Store a recent yield
        db.store_yield(_make_yield_data(protocol="Aave", timestamp=datetime.utcnow().isoformat()))
        results = db.get_latest_yields(hours=24)
        assert len(results) == 1
        assert results[0]["protocol"] == "Aave"

    def test_spread_history_protocol_filter(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = DeFiYieldDatabase(str(db_path))
        db.store_spread(_make_yield_spread(protocol="Lido"))
        db.store_spread(_make_yield_spread(protocol="Aave", defi_yield=0.042))
        results = db.get_spread_history("Lido", days=1)
        assert len(results) == 1
        assert results[0]["protocol"] == "Lido"


# ---------------------------------------------------------------------------
# DeFiYieldMonitor — _calculate_spread
# ---------------------------------------------------------------------------

class TestCalculateSpread:

    def test_spread_calculation(self, tmp_path):
        db_path = tmp_path / "test.db"
        monitor = DeFiYieldMonitor(db_path=str(db_path))
        yd = _make_yield_data(yield_apy=0.07)
        spread = monitor._calculate_spread(yd, 0.05)
        assert spread.spread == pytest.approx(0.02)
        assert spread.signal == "allocate"

    def test_signal_monitor(self, tmp_path):
        db_path = tmp_path / "test.db"
        monitor = DeFiYieldMonitor(db_path=str(db_path))
        yd = _make_yield_data(yield_apy=0.04)
        spread = monitor._calculate_spread(yd, 0.05)
        assert spread.signal == "monitor"

    def test_signal_consider(self, tmp_path):
        db_path = tmp_path / "test.db"
        monitor = DeFiYieldMonitor(db_path=str(db_path))
        yd = _make_yield_data(yield_apy=0.062)
        spread = monitor._calculate_spread(yd, 0.05)
        assert spread.signal == "consider"

    def test_signal_allocate(self, tmp_path):
        db_path = tmp_path / "test.db"
        monitor = DeFiYieldMonitor(db_path=str(db_path))
        yd = _make_yield_data(yield_apy=0.075)
        spread = monitor._calculate_spread(yd, 0.05)
        assert spread.signal == "allocate"

    def test_negative_spread(self, tmp_path):
        db_path = tmp_path / "test.db"
        monitor = DeFiYieldMonitor(db_path=str(db_path))
        yd = _make_yield_data(yield_apy=0.02)
        spread = monitor._calculate_spread(yd, 0.05)
        assert spread.spread == pytest.approx(-0.03)
        assert spread.signal == "monitor"

    def test_threshold_boundary(self, tmp_path):
        db_path = tmp_path / "test.db"
        monitor = DeFiYieldMonitor(db_path=str(db_path))
        # Use values that avoid floating point precision issues
        yd = _make_yield_data(yield_apy=0.0615)
        spread = monitor._calculate_spread(yd, 0.05)
        assert spread.signal == "consider"  # 1.15% spread > 1% threshold


# ---------------------------------------------------------------------------
# DeFiYieldMonitor — _check_alerts
# ---------------------------------------------------------------------------

class TestCheckAlerts:

    def test_high_spread_alert(self, tmp_path):
        db_path = tmp_path / "test.db"
        monitor = DeFiYieldMonitor(db_path=str(db_path))
        spreads = [{"protocol": "Lido", "spread": 0.025, "signal": "allocate"}]
        alerts = monitor._check_alerts([], spreads)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "high_spread"

    def test_no_alert_for_monitor(self, tmp_path):
        db_path = tmp_path / "test.db"
        monitor = DeFiYieldMonitor(db_path=str(db_path))
        spreads = [{"protocol": "Lido", "spread": 0.005, "signal": "monitor"}]
        alerts = monitor._check_alerts([], spreads)
        assert len(alerts) == 0

    def test_multiple_protocols(self, tmp_path):
        db_path = tmp_path / "test.db"
        monitor = DeFiYieldMonitor(db_path=str(db_path))
        spreads = [
            {"protocol": "Lido", "spread": 0.025, "signal": "allocate"},
            {"protocol": "Aave", "spread": 0.005, "signal": "monitor"},
            {"protocol": "Jito", "spread": 0.030, "signal": "allocate"},
        ]
        alerts = monitor._check_alerts([], spreads)
        assert len(alerts) == 2


# ---------------------------------------------------------------------------
# DeFiYieldMonitor — get_status / get_history
# ---------------------------------------------------------------------------

class TestMonitorStatus:

    def test_get_status_no_file(self, tmp_path):
        db_path = tmp_path / "test.db"
        monitor = DeFiYieldMonitor(db_path=str(db_path))
        monitor.output_path = tmp_path / "nonexistent.json"
        status = monitor.get_status()
        assert "error" in status

    def test_get_status_with_file(self, tmp_path):
        db_path = tmp_path / "test.db"
        monitor = DeFiYieldMonitor(db_path=str(db_path))
        monitor.output_path = tmp_path / "status.json"
        expected = {"timestamp": "2026-05-14", "yields": []}
        with open(monitor.output_path, "w") as f:
            json.dump(expected, f)
        status = monitor.get_status()
        assert status["timestamp"] == "2026-05-14"

    def test_get_history_empty(self, tmp_path):
        db_path = tmp_path / "test.db"
        monitor = DeFiYieldMonitor(db_path=str(db_path))
        history = monitor.get_history(days=30)
        assert history["period_days"] == 30
        assert history["protocols"] == {}

    def test_get_history_with_data(self, tmp_path):
        db_path = tmp_path / "test.db"
        monitor = DeFiYieldMonitor(db_path=str(db_path))
        # Store spreads
        monitor.db.store_spread(_make_yield_spread(protocol="Lido", spread=0.02))
        monitor.db.store_spread(_make_yield_spread(protocol="Lido", spread=0.03))
        history = monitor.get_history(days=1)
        assert "Lido" in history["protocols"]
        assert history["protocols"]["Lido"]["data_points"] == 2
        assert history["protocols"]["Lido"]["avg_spread_30d"] == pytest.approx(0.025)

    def test_get_history_signal_from_latest(self, tmp_path):
        db_path = tmp_path / "test.db"
        monitor = DeFiYieldMonitor(db_path=str(db_path))
        monitor.db.store_spread(_make_yield_spread(protocol="Aave", signal="monitor"))
        monitor.db.store_spread(_make_yield_spread(protocol="Aave", signal="consider"))
        history = monitor.get_history(days=1)
        # Latest should be "consider" (inserted last)
        assert history["protocols"]["Aave"]["signal"] == "consider"


# ---------------------------------------------------------------------------
# DeFiYieldMonitor — update (async, mocked)
# ---------------------------------------------------------------------------

class TestMonitorUpdate:

    def test_update_stores_yields(self, tmp_path):
        db_path = tmp_path / "test.db"
        monitor = DeFiYieldMonitor(db_path=str(db_path))
        monitor.output_path = tmp_path / "status.json"

        mock_yields = [
            _make_yield_data(protocol="Lido", yield_apy=0.035),
            _make_yield_data(protocol="Aave", yield_apy=0.042),
        ]

        async def run():
            with patch.object(DeFiYieldFetcher, 'fetch_all_yields', new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = mock_yields
                with patch.object(TreasuryYieldFetcher, 'fetch_3m_treasury', new_callable=AsyncMock) as mock_treasury:
                    mock_treasury.return_value = 0.05
                    return await monitor.update()

        result = asyncio.run(run())
        assert "yields" in result
        assert "spreads" in result
        assert len(result["yields"]) == 2
        assert monitor.output_path.exists()

    def test_update_generates_alerts(self, tmp_path):
        db_path = tmp_path / "test.db"
        monitor = DeFiYieldMonitor(db_path=str(db_path))
        monitor.output_path = tmp_path / "status.json"

        mock_yields = [_make_yield_data(protocol="Lido", yield_apy=0.08)]

        async def run():
            with patch.object(DeFiYieldFetcher, 'fetch_all_yields', new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = mock_yields
                with patch.object(TreasuryYieldFetcher, 'fetch_3m_treasury', new_callable=AsyncMock) as mock_treasury:
                    mock_treasury.return_value = 0.05
                    return await monitor.update()

        result = asyncio.run(run())
        assert len(result["alerts"]) == 1
        assert result["alerts"][0]["type"] == "high_spread"


# ---------------------------------------------------------------------------
# Monitor Thresholds
# ---------------------------------------------------------------------------

class TestMonitorThresholds:

    def test_spread_thresholds(self):
        assert DeFiYieldMonitor.SPREAD_THRESHOLD_ALLOCATE == 0.02
        assert DeFiYieldMonitor.SPREAD_THRESHOLD_CONSIDER == 0.01

    def test_correlation_threshold(self):
        assert DeFiYieldMonitor.CORRELATION_THRESHOLD == 0.60

    def test_tvl_decline_threshold(self):
        assert DeFiYieldMonitor.TVL_DECLINE_THRESHOLD == -0.20


# ---------------------------------------------------------------------------
# DeFiYieldFetcher — context manager
# ---------------------------------------------------------------------------

class TestDeFiYieldFetcherCM:

    def test_context_manager(self):
        async def run():
            fetcher = DeFiYieldFetcher()
            async with fetcher as f:
                assert f.session is not None

        asyncio.run(run())


# ---------------------------------------------------------------------------
# TreasuryYieldFetcher Tests
# ---------------------------------------------------------------------------

class TestTreasuryYieldFetcher:

    def test_get_api_key_fallback(self):
        fetcher = TreasuryYieldFetcher()
        key = fetcher._get_api_key()
        # Should return the default or env var
        assert isinstance(key, str)

    def test_fetch_3m_treasury_fallback(self):
        """When API fails, should return fallback rate."""
        async def run():
            fetcher = TreasuryYieldFetcher()
            with patch("aiohttp.ClientSession") as mock_session_cls:
                mock_session = AsyncMock()
                mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                mock_session.get.side_effect = Exception("network error")
                return await fetcher.fetch_3m_treasury()

        result = asyncio.run(run())
        assert result == 0.0525  # Fallback


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:

    def test_main_no_args(self, capsys):
        """No args should print help."""
        from src.data.defi_yield_fetcher import main
        with patch("sys.argv", ["defi_yield_fetcher.py"]):
            main()
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "DeFi" in captured.out

    def test_status_command(self, tmp_path, capsys):
        from src.data.defi_yield_fetcher import main
        with patch("sys.argv", ["defi_yield_fetcher.py", "--status"]):
            with patch("src.data.defi_yield_fetcher.DeFiYieldMonitor") as mock_cls:
                mock_monitor = MagicMock()
                mock_monitor.get_status.return_value = {"error": "No status"}
                mock_cls.return_value = mock_monitor
                main()
        captured = capsys.readouterr()
        assert "No status" in captured.out

    def test_history_command(self, tmp_path, capsys):
        from src.data.defi_yield_fetcher import main
        with patch("sys.argv", ["defi_yield_fetcher.py", "--history", "30"]):
            with patch("src.data.defi_yield_fetcher.DeFiYieldMonitor") as mock_cls:
                mock_monitor = MagicMock()
                mock_monitor.get_history.return_value = {"period_days": 30, "protocols": {}}
                mock_cls.return_value = mock_monitor
                main()
        captured = capsys.readouterr()
        assert "30" in captured.out
