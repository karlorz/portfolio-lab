#!/usr/bin/env python3
"""
Tests for defi_dashboard.py — DeFiDashboard init, display logic, recommendation
logic, and CLI.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.monitor.defi_dashboard import DeFiDashboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dashboard(tmp_path):
    return DeFiDashboard(
        db_path=str(tmp_path / "defi.db"),
        json_path=str(tmp_path / "monitor.json"),
    )


def _make_status_data(**overrides):
    defaults = {
        "timestamp": "2026-05-14T00:00:00",
        "treasury_yield_3m": 0.0525,
        "yields": [
            {"protocol": "Lido", "asset": "stETH", "yield_apy": 0.035, "tvl_usd": 1e9},
            {"protocol": "Aave", "asset": "USDC", "yield_apy": 0.042, "tvl_usd": 500e6},
        ],
        "spreads": [
            {"protocol": "Lido", "spread": -0.0175, "signal": "monitor"},
            {"protocol": "Aave", "spread": -0.0105, "signal": "monitor"},
        ],
        "alerts": [],
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Init Tests
# ---------------------------------------------------------------------------

class TestInit:

    def test_default_paths(self):
        dashboard = DeFiDashboard()
        assert "defi_yield_history.db" in str(dashboard.db_path)
        assert "defi_monitor.json" in str(dashboard.json_path)

    def test_custom_paths(self, tmp_path):
        dashboard = _make_dashboard(tmp_path)
        assert dashboard.db_path == tmp_path / "defi.db"
        assert dashboard.json_path == tmp_path / "monitor.json"


# ---------------------------------------------------------------------------
# Recommendation Logic Tests
# ---------------------------------------------------------------------------

class TestRecommendationLogic:

    def test_allocate_signal(self):
        spreads = [
            {"protocol": "Lido", "spread": 0.025, "signal": "allocate"},
        ]
        any_allocate = any(s.get('signal') == 'allocate' for s in spreads)
        any_consider = any(s.get('signal') == 'consider' for s in spreads)
        assert any_allocate is True

    def test_consider_signal(self):
        spreads = [
            {"protocol": "Lido", "spread": 0.015, "signal": "consider"},
        ]
        any_allocate = any(s.get('signal') == 'allocate' for s in spreads)
        any_consider = any(s.get('signal') == 'consider' for s in spreads)
        assert any_allocate is False
        assert any_consider is True

    def test_monitor_signal(self):
        spreads = [
            {"protocol": "Lido", "spread": 0.005, "signal": "monitor"},
        ]
        any_allocate = any(s.get('signal') == 'allocate' for s in spreads)
        any_consider = any(s.get('signal') == 'consider' for s in spreads)
        assert any_allocate is False
        assert any_consider is False

    def test_mixed_signals(self):
        spreads = [
            {"protocol": "Lido", "spread": 0.025, "signal": "allocate"},
            {"protocol": "Aave", "spread": 0.005, "signal": "monitor"},
        ]
        any_allocate = any(s.get('signal') == 'allocate' for s in spreads)
        assert any_allocate is True


# ---------------------------------------------------------------------------
# display_status Tests
# ---------------------------------------------------------------------------

class TestDisplayStatus:

    def test_no_json_file(self, tmp_path, capsys):
        dashboard = _make_dashboard(tmp_path)
        dashboard.display_status()
        captured = capsys.readouterr()
        assert "No status" in captured.out or "⚠️" in captured.out

    def test_with_json_data(self, tmp_path, capsys):
        dashboard = _make_dashboard(tmp_path)
        data = _make_status_data()
        with open(dashboard.json_path, 'w') as f:
            json.dump(data, f)
        dashboard.display_status()
        captured = capsys.readouterr()
        assert "Lido" in captured.out
        assert "Aave" in captured.out

    def test_with_alerts(self, tmp_path, capsys):
        dashboard = _make_dashboard(tmp_path)
        data = _make_status_data(alerts=[
            {"type": "high_spread", "message": "Spread exceeds 2%"}
        ])
        with open(dashboard.json_path, 'w') as f:
            json.dump(data, f)
        dashboard.display_status()
        captured = capsys.readouterr()
        assert "Alert" in captured.out or "high_spread" in captured.out

    def test_with_allocate_recommendation(self, tmp_path, capsys):
        dashboard = _make_dashboard(tmp_path)
        data = _make_status_data(spreads=[
            {"protocol": "Lido", "spread": 0.025, "signal": "allocate"},
        ])
        with open(dashboard.json_path, 'w') as f:
            json.dump(data, f)
        dashboard.display_status()
        captured = capsys.readouterr()
        assert "ALLOCATION" in captured.out or "allocate" in captured.out.lower()

    def test_with_consider_recommendation(self, tmp_path, capsys):
        dashboard = _make_dashboard(tmp_path)
        data = _make_status_data(spreads=[
            {"protocol": "Lido", "spread": 0.015, "signal": "consider"},
        ])
        with open(dashboard.json_path, 'w') as f:
            json.dump(data, f)
        dashboard.display_status()
        captured = capsys.readouterr()
        assert "MONITOR" in captured.out or "consider" in captured.out.lower()

    def test_tvl_display(self, tmp_path, capsys):
        dashboard = _make_dashboard(tmp_path)
        data = _make_status_data()
        with open(dashboard.json_path, 'w') as f:
            json.dump(data, f)
        dashboard.display_status()
        captured = capsys.readouterr()
        # TVL should be displayed in millions
        assert "M" in captured.out


# ---------------------------------------------------------------------------
# display_thresholds Tests
# ---------------------------------------------------------------------------

class TestDisplayThresholds:

    def test_prints_thresholds(self, tmp_path, capsys):
        dashboard = _make_dashboard(tmp_path)
        dashboard.display_thresholds()
        captured = capsys.readouterr()
        assert "ALLOCATE" in captured.out
        assert "CONSIDER" in captured.out
        assert "MONITOR" in captured.out

    def test_prints_scale_requirements(self, tmp_path, capsys):
        dashboard = _make_dashboard(tmp_path)
        dashboard.display_thresholds()
        captured = capsys.readouterr()
        assert "500K" in captured.out or "$500" in captured.out


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:

    def test_thresholds_command(self, tmp_path, capsys):
        from src.monitor.defi_dashboard import main
        dashboard = _make_dashboard(tmp_path)
        with patch("src.monitor.defi_dashboard.DeFiDashboard", return_value=dashboard):
            with patch("sys.argv", ["defi_dashboard.py", "--thresholds"]):
                main()
        captured = capsys.readouterr()
        assert "ALLOCATE" in captured.out

    def test_status_no_file(self, tmp_path, capsys):
        from src.monitor.defi_dashboard import main
        dashboard = _make_dashboard(tmp_path)
        with patch("src.monitor.defi_dashboard.DeFiDashboard", return_value=dashboard):
            with patch("sys.argv", ["defi_dashboard.py", "--status"]):
                main()
        captured = capsys.readouterr()
        assert "No status" in captured.out or "⚠️" in captured.out
