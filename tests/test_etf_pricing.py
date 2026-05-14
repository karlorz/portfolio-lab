"""
Tests for src/data/etf_pricing.py — ETF premium/discount monitor.
No ML deps. Mocks network calls and file I/O.
"""
import pytest
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.etf_pricing import (
    ETFPremium,
    ETFPricingEngine,
    PORTFOLIO_ALLOCATION,
    ETF_PRICING_PATH,
    ETF_HISTORY_PATH,
)


class TestETFPremium:
    """ETFPremium dataclass."""

    def test_create_normal(self):
        p = ETFPremium(
            symbol="SPY", timestamp="2024-06-15T12:00:00",
            market_price=550.0, nav=549.5, premium_pct=0.091,
            alert_status="normal", bid_ask_spread=0.012,
        )
        assert p.symbol == "SPY"
        assert p.market_price == 550.0
        assert p.nav == 549.5
        assert p.premium_pct == 0.091
        assert p.alert_status == "normal"

    def test_create_warning(self):
        p = ETFPremium(
            symbol="GLD", timestamp="2024-06-15T12:00:00",
            market_price=200.0, nav=199.6, premium_pct=0.20,
            alert_status="warning", bid_ask_spread=0.05,
        )
        assert p.alert_status == "warning"

    def test_create_critical(self):
        p = ETFPremium(
            symbol="TLT", timestamp="2024-06-15T12:00:00",
            market_price=95.0, nav=94.5, premium_pct=0.53,
            alert_status="critical", bid_ask_spread=0.08,
        )
        assert p.alert_status == "critical"
        assert abs(p.premium_pct) > 0.30

    def test_negative_premium(self):
        p = ETFPremium(
            symbol="SPY", timestamp="2024-06-15T12:00:00",
            market_price=548.0, nav=550.0, premium_pct=-0.36,
            alert_status="critical", bid_ask_spread=0.03,
        )
        assert p.premium_pct < 0

    def test_to_dict(self):
        p = ETFPremium(
            symbol="SPY", timestamp="t", market_price=550.0, nav=549.5,
            premium_pct=0.091, alert_status="normal", bid_ask_spread=0.012,
        )
        d = p.to_dict()
        assert d["symbol"] == "SPY"
        assert d["market_price"] == 550.0
        assert d["nav"] == 549.5
        assert d["premium_pct"] == 0.091
        assert d["alert_status"] == "normal"

    def test_default_fields(self):
        p = ETFPremium(
            symbol="SPY", timestamp="t", market_price=550.0, nav=550.0,
            premium_pct=0.0, alert_status="normal", bid_ask_spread=0.01,
        )
        assert p.volume_24h is None
        assert p.data_source == "calculated"


class TestPortfolioMetrics:
    """calculate_portfolio_metrics — pure computation, no I/O."""

    def make_pricing(self, symbol, premium_pct, alert_status="normal"):
        return ETFPremium(
            symbol=symbol, timestamp="t", market_price=100.0, nav=100.0,
            premium_pct=premium_pct, alert_status=alert_status, bid_ask_spread=0.01,
        )

    def test_empty_input(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        result = engine.calculate_portfolio_metrics({})
        assert result == {}

    def test_single_etf(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        data = {"SPY": self.make_pricing("SPY", 0.10)}
        result = engine.calculate_portfolio_metrics(data)
        assert result["weighted_premium_pct"] == pytest.approx(0.10)
        assert result["overall_status"] == "normal"
        assert result["etfs_tracked"] == 1

    def test_weighted_average(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        data = {
            "SPY": self.make_pricing("SPY", 0.20),
            "GLD": self.make_pricing("GLD", 0.10),
            "TLT": self.make_pricing("TLT", 0.05),
        }
        result = engine.calculate_portfolio_metrics(data)
        expected = (0.20 * 0.46 + 0.10 * 0.38 + 0.05 * 0.16) / (0.46 + 0.38 + 0.16)
        assert result["weighted_premium_pct"] == pytest.approx(expected)

    def test_overall_status_warning(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        data = {
            "SPY": self.make_pricing("SPY", 0.10),
            "GLD": self.make_pricing("GLD", 0.05, "warning"),
        }
        result = engine.calculate_portfolio_metrics(data)
        assert result["overall_status"] == "warning"
        assert "GLD" in result["warning_symbols"]

    def test_overall_status_critical(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        data = {
            "SPY": self.make_pricing("SPY", 0.10),
            "GLD": self.make_pricing("GLD", 0.05, "critical"),
        }
        result = engine.calculate_portfolio_metrics(data)
        assert result["overall_status"] == "critical"
        assert "GLD" in result["critical_symbols"]

    def test_critical_overrides_warning(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        data = {
            "SPY": self.make_pricing("SPY", 0.10, "warning"),
            "GLD": self.make_pricing("GLD", 0.05, "critical"),
        }
        result = engine.calculate_portfolio_metrics(data)
        assert result["overall_status"] == "critical"

    def test_unknown_symbol_skipped(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        data = {"UNKNOWN": self.make_pricing("UNKNOWN", 0.50)}
        result = engine.calculate_portfolio_metrics(data)
        assert result["weighted_premium_pct"] == 0.0
        assert result["etfs_tracked"] == 1


class TestProxyNAV:
    """calculate_proxy_nav logic."""

    def test_no_history_uses_default(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        with patch.object(engine, 'load_pricing_history', return_value=[]):
            nav, source = engine.calculate_proxy_nav("SPY", 550.0)
            assert nav == pytest.approx(550.0 * 0.9998)
            assert source == "estimated"

    def test_with_history_uses_median(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        history = [
            {"premium_pct": 0.10, "timestamp": "2024-06-15T10:00:00"},
            {"premium_pct": 0.05, "timestamp": "2024-06-15T11:00:00"},
            {"premium_pct": 0.20, "timestamp": "2024-06-15T12:00:00"},
        ]
        with patch.object(engine, 'load_pricing_history', return_value=history):
            nav, source = engine.calculate_proxy_nav("SPY", 550.0)
            # median premium = 0.10, nav = 550 / (1 + 0.10/100) = 550 / 1.001 ≈ 549.45
            expected = 550.0 / (1 + 0.10 / 100)
            assert nav == pytest.approx(expected)
            assert source == "calculated"

    def test_history_without_premium_falls_back(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        history = [{"timestamp": "2024-06-15T10:00:00"}]  # No premium_pct
        with patch.object(engine, 'load_pricing_history', return_value=history):
            nav, source = engine.calculate_proxy_nav("SPY", 550.0)
            assert nav == pytest.approx(550.0 * 0.9998)
            assert source == "estimated"


class TestTradeEligibility:
    """check_trade_eligibility logic."""

    def make_pricing(self, symbol, premium_pct, alert_status="normal"):
        return ETFPremium(
            symbol=symbol, timestamp="t", market_price=100.0, nav=100.0,
            premium_pct=premium_pct, alert_status=alert_status, bid_ask_spread=0.01,
        )

    def test_normal_returns_ok(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        pricing = self.make_pricing("SPY", 0.05)
        with patch.object(engine, 'fetch_etf_pricing', return_value=pricing):
            eligible, reason = engine.check_trade_eligibility("SPY", "buy")
            assert eligible is True
            assert "OK" in reason

    def test_pricing_unavailable(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        with patch.object(engine, 'fetch_etf_pricing', return_value=None):
            eligible, reason = engine.check_trade_eligibility("SPY", "buy")
            assert eligible is True
            assert "pricing unavailable" in reason

    def test_critical_blocks_trade(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        pricing = self.make_pricing("SPY", 0.35, "critical")
        with patch.object(engine, 'fetch_etf_pricing', return_value=pricing):
            eligible, reason = engine.check_trade_eligibility("SPY", "buy")
            assert eligible is False
            assert "CRITICAL" in reason

    def test_warning_with_small_size_allows(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        pricing = self.make_pricing("SPY", 0.20, "warning")
        with patch.object(engine, 'fetch_etf_pricing', return_value=pricing):
            eligible, reason = engine.check_trade_eligibility("SPY", "buy", size_pct=3.0)
            assert eligible is True
            assert "WARNING" in reason

    def test_warning_with_large_size_blocks(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        pricing = self.make_pricing("SPY", 0.20, "warning")
        with patch.object(engine, 'fetch_etf_pricing', return_value=pricing):
            eligible, reason = engine.check_trade_eligibility("SPY", "buy", size_pct=6.0)
            assert eligible is False
            assert "WARNING" in reason

    def test_boundary_size_exactly_5(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        pricing = self.make_pricing("SPY", 0.20, "warning")
        with patch.object(engine, 'fetch_etf_pricing', return_value=pricing):
            eligible, reason = engine.check_trade_eligibility("SPY", "buy", size_pct=5.0)
            # size_pct > 5.0 → False; size_pct=5.0 NOT > 5.0 → True
            assert eligible is True

    def test_slightly_elevated_advisory(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        pricing = self.make_pricing("SPY", 0.12)
        with patch.object(engine, 'fetch_etf_pricing', return_value=pricing):
            eligible, reason = engine.check_trade_eligibility("SPY", "sell")
            assert eligible is True
            assert "ADVISORY" in reason


class TestSaveLoadHistory:
    """File I/O for pricing history."""

    def test_load_history_no_file(self):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        with patch('src.data.etf_pricing.ETF_HISTORY_PATH', Path('/nonexistent/path.json')):
            history = engine.load_pricing_history("SPY")
            assert history == []

    def test_save_and_load(self, tmp_path):
        engine = ETFPricingEngine.__new__(ETFPricingEngine)
        pricing = {
            "SPY": ETFPremium(
                symbol="SPY", timestamp="2024-06-15T12:00:00",
                market_price=550.0, nav=549.5, premium_pct=0.091,
                alert_status="normal", bid_ask_spread=0.012,
            ),
            "GLD": ETFPremium(
                symbol="GLD", timestamp="2024-06-15T12:00:01",
                market_price=200.0, nav=199.6, premium_pct=0.20,
                alert_status="warning", bid_ask_spread=0.05,
            ),
        }
        metrics = {
            "timestamp": "2024-06-15T12:00:00",
            "weighted_premium_pct": 0.12,
            "overall_status": "warning",
            "warning_symbols": ["GLD"],
            "critical_symbols": [],
            "etfs_tracked": 2,
        }

        with patch('src.data.etf_pricing.ETF_PRICING_PATH', tmp_path / 'etf_pricing.json'):
            with patch('src.data.etf_pricing.ETF_HISTORY_PATH', tmp_path / 'etf_history.json'):
                engine.save_pricing_data(pricing, metrics)
                assert (tmp_path / 'etf_pricing.json').exists()
                assert (tmp_path / 'etf_history.json').exists()

    def test_portfolio_allocation_constants(self):
        assert PORTFOLIO_ALLOCATION["SPY"] == 0.46
        assert PORTFOLIO_ALLOCATION["GLD"] == 0.38
        assert PORTFOLIO_ALLOCATION["TLT"] == 0.16
        assert abs(sum(PORTFOLIO_ALLOCATION.values()) - 1.0) < 0.01


class TestDisplayPricing:
    """display_pricing output format."""

    def test_display_normal_portfolio(self, capsys):
        from src.data.etf_pricing import display_pricing
        pricing = {
            "SPY": ETFPremium(
                symbol="SPY", timestamp="t", market_price=550.0, nav=549.5,
                premium_pct=0.091, alert_status="normal", bid_ask_spread=0.012,
            ),
        }
        metrics = {
            "overall_status": "normal",
            "weighted_premium_pct": 0.091,
            "warning_symbols": [],
            "critical_symbols": [],
        }
        display_pricing(pricing, metrics)
        captured = capsys.readouterr()
        assert "NORMAL" in captured.out
        assert "SPY" in captured.out

    def test_display_critical_portfolio(self, capsys):
        from src.data.etf_pricing import display_pricing
        pricing = {
            "TLT": ETFPremium(
                symbol="TLT", timestamp="t", market_price=95.0, nav=94.5,
                premium_pct=0.53, alert_status="critical", bid_ask_spread=0.08,
            ),
        }
        metrics = {
            "overall_status": "critical",
            "weighted_premium_pct": 0.53,
            "warning_symbols": [],
            "critical_symbols": ["TLT"],
        }
        display_pricing(pricing, metrics)
        captured = capsys.readouterr()
        assert "CRITICAL" in captured.out
        assert "TLT" in captured.out

    def test_display_with_warnings(self, capsys):
        from src.data.etf_pricing import display_pricing
        pricing = {
            "GLD": ETFPremium(
                symbol="GLD", timestamp="t", market_price=200.0, nav=199.6,
                premium_pct=0.20, alert_status="warning", bid_ask_spread=0.05,
            ),
        }
        metrics = {
            "overall_status": "warning",
            "weighted_premium_pct": 0.20,
            "warning_symbols": ["GLD"],
            "critical_symbols": [],
        }
        display_pricing(pricing, metrics)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
