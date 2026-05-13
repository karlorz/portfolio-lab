#!/usr/bin/env python3
"""
Tests for position sync — drift calculation between broker and local.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock
from src.broker.position_sync import PositionSync, PositionDrift


class TestCalculateDrift:
    """Test drift calculation logic (no broker connection needed)."""

    def _make_sync(self):
        sync = PositionSync.__new__(PositionSync)
        sync.db_path = ":memory:"
        sync.data_dir = "/tmp"
        sync.sync_log_path = "/tmp/test_sync.jsonl"
        return sync

    def test_no_drift(self):
        """Identical positions → no drift"""
        sync = self._make_sync()
        local = {"SPY": {"qty": 100, "market_value": 50000}}
        broker = {"SPY": MagicMock(qty=100, market_value=50000)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 0

    def test_broker_overweight(self):
        """Broker has more shares → positive drift"""
        sync = self._make_sync()
        local = {"SPY": {"qty": 100, "market_value": 50000}}
        broker = {"SPY": MagicMock(qty=110, market_value=55000)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 1
        assert drift[0].symbol == "SPY"
        assert drift[0].qty_delta == 10
        assert drift[0].drift_pct == pytest.approx(0.1, abs=0.01)

    def test_broker_underweight(self):
        """Broker has fewer shares → negative drift"""
        sync = self._make_sync()
        local = {"SPY": {"qty": 100, "market_value": 50000}}
        broker = {"SPY": MagicMock(qty=90, market_value=45000)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 1
        assert drift[0].qty_delta == -10
        assert drift[0].drift_pct < 0

    def test_symbol_only_in_broker(self):
        """Position exists in broker but not local → 100% drift"""
        sync = self._make_sync()
        local = {}
        broker = {"GLD": MagicMock(qty=50, market_value=30000)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 1
        assert drift[0].symbol == "GLD"
        assert drift[0].local_qty == 0
        assert drift[0].broker_qty == 50
        assert drift[0].drift_pct == 1.0

    def test_symbol_only_in_local(self):
        """Position exists in local but not broker → -100% drift"""
        sync = self._make_sync()
        local = {"TLT": {"qty": 200, "market_value": 20000}}
        broker = {}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 1
        assert drift[0].symbol == "TLT"
        assert drift[0].broker_qty == 0
        assert drift[0].drift_pct == -1.0

    def test_multiple_symbols(self):
        """Multiple symbols with mixed drift"""
        sync = self._make_sync()
        local = {
            "SPY": {"qty": 100, "market_value": 50000},
            "GLD": {"qty": 50, "market_value": 30000},
            "TLT": {"qty": 200, "market_value": 20000},
        }
        broker = {
            "SPY": MagicMock(qty=100, market_value=50000),  # No drift
            "GLD": MagicMock(qty=55, market_value=33000),   # Overweight
            "TLT": MagicMock(qty=180, market_value=18000),  # Underweight
        }
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 2
        symbols = {d.symbol for d in drift}
        assert symbols == {"GLD", "TLT"}

    def test_small_drift_ignored(self):
        """Tiny drift below threshold → no drift recorded"""
        sync = self._make_sync()
        local = {"SPY": {"qty": 100, "market_value": 50000}}
        broker = {"SPY": MagicMock(qty=100.0005, market_value=50000.5)}
        drift = sync.calculate_drift(local, broker)
        assert len(drift) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
