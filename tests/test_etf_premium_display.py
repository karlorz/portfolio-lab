#!/usr/bin/env python3
"""
Tests for etf_premium_display.py — load_etf_pricing, get_status_color,
format_premium_display, get_compact_summary, and CLI.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.monitor.etf_premium_display import (
    load_etf_pricing,
    get_status_color,
    reset_color,
    format_premium_display,
    get_compact_summary,
    export_for_health_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pricing_data(**overrides):
    defaults = {
        "portfolio_metrics": {
            "overall_status": "normal",
            "weighted_premium_pct": 0.05,
            "warning_symbols": [],
            "critical_symbols": [],
        },
        "etfs": {
            "SPY": {"market_price": 450.0, "nav": 449.95, "premium_pct": 0.011, "alert_status": "normal"},
            "GLD": {"market_price": 190.0, "nav": 190.05, "premium_pct": -0.026, "alert_status": "normal"},
            "TLT": {"market_price": 95.0, "nav": 95.10, "premium_pct": -0.105, "alert_status": "warning"},
        },
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# load_etf_pricing Tests
# ---------------------------------------------------------------------------

class TestLoadETFPricing:

    def test_missing_file(self, tmp_path):
        import src.monitor.etf_premium_display as mod
        old = mod.ETF_PRICING_PATH
        mod.ETF_PRICING_PATH = tmp_path / "nonexistent.json"
        try:
            assert load_etf_pricing() is None
        finally:
            mod.ETF_PRICING_PATH = old

    def test_valid_json(self, tmp_path):
        import src.monitor.etf_premium_display as mod
        old = mod.ETF_PRICING_PATH
        path = tmp_path / "pricing.json"
        mod.ETF_PRICING_PATH = path
        try:
            data = _make_pricing_data()
            with open(path, 'w') as f:
                json.dump(data, f)
            loaded = load_etf_pricing()
            assert loaded is not None
            assert "portfolio_metrics" in loaded
        finally:
            mod.ETF_PRICING_PATH = old

    def test_invalid_json(self, tmp_path):
        import src.monitor.etf_premium_display as mod
        old = mod.ETF_PRICING_PATH
        path = tmp_path / "bad.json"
        mod.ETF_PRICING_PATH = path
        try:
            with open(path, 'w') as f:
                f.write("not json{{{")
            assert load_etf_pricing() is None
        finally:
            mod.ETF_PRICING_PATH = old


# ---------------------------------------------------------------------------
# get_status_color Tests
# ---------------------------------------------------------------------------

class TestGetStatusColor:

    def test_normal_green(self):
        color = get_status_color("normal")
        assert color == "\033[32m"

    def test_warning_yellow(self):
        color = get_status_color("warning")
        assert color == "\033[33m"

    def test_critical_red(self):
        color = get_status_color("critical")
        assert color == "\033[31m"

    def test_unknown_empty(self):
        color = get_status_color("unknown")
        assert color == ""


# ---------------------------------------------------------------------------
# reset_color Tests
# ---------------------------------------------------------------------------

class TestResetColor:

    def test_returns_escape(self):
        assert reset_color() == "\033[0m"


# ---------------------------------------------------------------------------
# format_premium_display Tests
# ---------------------------------------------------------------------------

class TestFormatPremiumDisplay:

    def test_empty_data(self):
        result = format_premium_display(None)
        assert "unavailable" in result.lower()

    def test_empty_dict(self):
        result = format_premium_display({})
        assert "unavailable" in result.lower()

    def test_has_header(self):
        data = _make_pricing_data()
        result = format_premium_display(data)
        assert "ETF PREMIUM" in result

    def test_has_portfolio_status(self):
        data = _make_pricing_data()
        result = format_premium_display(data)
        assert "NORMAL" in result

    def test_has_weighted_premium(self):
        data = _make_pricing_data()
        result = format_premium_display(data)
        assert "+0.050%" in result

    def test_has_etf_details(self):
        data = _make_pricing_data()
        result = format_premium_display(data)
        assert "SPY" in result
        assert "GLD" in result
        assert "TLT" in result

    def test_has_warnings(self):
        data = _make_pricing_data(
            portfolio_metrics={
                "overall_status": "warning",
                "weighted_premium_pct": 0.10,
                "warning_symbols": ["TLT"],
                "critical_symbols": [],
            }
        )
        result = format_premium_display(data)
        assert "TLT" in result

    def test_has_criticals(self):
        data = _make_pricing_data(
            portfolio_metrics={
                "overall_status": "critical",
                "weighted_premium_pct": 0.50,
                "warning_symbols": [],
                "critical_symbols": ["SPY"],
            }
        )
        result = format_premium_display(data)
        assert "CRITICAL" in result or "SPY" in result

    def test_status_colors(self):
        for status in ["normal", "warning", "critical"]:
            data = _make_pricing_data(
                portfolio_metrics={
                    "overall_status": status,
                    "weighted_premium_pct": 0.05,
                    "warning_symbols": [],
                    "critical_symbols": [],
                }
            )
            result = format_premium_display(data)
            assert status.upper() in result


# ---------------------------------------------------------------------------
# get_compact_summary Tests
# ---------------------------------------------------------------------------

class TestCompactSummary:

    def test_empty_data(self):
        assert get_compact_summary(None) == "ETF Premium: unavailable"

    def test_normal_status(self):
        data = _make_pricing_data()
        summary = get_compact_summary(data)
        assert "NORMAL" in summary
        assert "+0.05%" in summary

    def test_with_warnings(self):
        data = _make_pricing_data(
            portfolio_metrics={
                "overall_status": "warning",
                "weighted_premium_pct": 0.10,
                "warning_symbols": ["TLT"],
                "critical_symbols": [],
            }
        )
        summary = get_compact_summary(data)
        assert "WARN" in summary
        assert "TLT" in summary

    def test_with_criticals(self):
        data = _make_pricing_data(
            portfolio_metrics={
                "overall_status": "critical",
                "weighted_premium_pct": 0.50,
                "warning_symbols": [],
                "critical_symbols": ["SPY"],
            }
        )
        summary = get_compact_summary(data)
        assert "CRITICAL" in summary
        assert "SPY" in summary

    def test_no_warnings_or_criticals(self):
        data = _make_pricing_data()
        summary = get_compact_summary(data)
        assert "WARN" not in summary
        assert "CRITICAL" not in summary


# ---------------------------------------------------------------------------
# export_for_health_check Tests
# ---------------------------------------------------------------------------

class TestExportHealthCheck:

    def test_with_data(self, tmp_path, capsys):
        import src.monitor.etf_premium_display as mod
        old = mod.ETF_PRICING_PATH
        path = tmp_path / "pricing.json"
        mod.ETF_PRICING_PATH = path
        try:
            with open(path, 'w') as f:
                json.dump(_make_pricing_data(), f)
            result = export_for_health_check()
            assert result is True
            captured = capsys.readouterr()
            assert "ETF Premium" in captured.out
        finally:
            mod.ETF_PRICING_PATH = old

    def test_without_data(self, tmp_path, capsys):
        import src.monitor.etf_premium_display as mod
        old = mod.ETF_PRICING_PATH
        mod.ETF_PRICING_PATH = tmp_path / "nonexistent.json"
        try:
            result = export_for_health_check()
            assert result is False
            captured = capsys.readouterr()
            assert "unavailable" in captured.out.lower()
        finally:
            mod.ETF_PRICING_PATH = old


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:

    def test_compact_flag(self, tmp_path, capsys):
        import src.monitor.etf_premium_display as mod
        old = mod.ETF_PRICING_PATH
        path = tmp_path / "pricing.json"
        mod.ETF_PRICING_PATH = path
        try:
            with open(path, 'w') as f:
                json.dump(_make_pricing_data(), f)
            from src.monitor.etf_premium_display import main
            with patch("sys.argv", ["etf_premium_display.py", "--compact"]):
                main()
            captured = capsys.readouterr()
            assert "ETF Premium" in captured.out
        finally:
            mod.ETF_PRICING_PATH = old

    def test_no_data(self, tmp_path, capsys):
        import src.monitor.etf_premium_display as mod
        old = mod.ETF_PRICING_PATH
        mod.ETF_PRICING_PATH = tmp_path / "nonexistent.json"
        try:
            from src.monitor.etf_premium_display import main
            with patch("sys.argv", ["etf_premium_display.py"]):
                main()
            captured = capsys.readouterr()
            assert "No ETF" in captured.out or "unavailable" in captured.out.lower()
        finally:
            mod.ETF_PRICING_PATH = old
