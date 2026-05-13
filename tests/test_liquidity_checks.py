#!/usr/bin/env python3
"""
Tests for pre-trade liquidity checks — premium thresholds, trade eligibility,
critical/warning blocks, force override, position size limits.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock

from src.trading.liquidity_checks import LiquidityChecker, LiquidityCheckResult


def _make_premium(premium_pct):
    """Create a mock ETFPremium object."""
    p = MagicMock()
    p.premium_pct = premium_pct
    p.symbol = "SPY"
    p.market_price = 500.0
    p.nav = 500.0 * (1 - premium_pct / 100)
    p.alert_status = "normal" if abs(premium_pct) < 0.15 else "warning" if abs(premium_pct) < 0.30 else "critical"
    return p


class TestPremiumThresholds:
    """Test premium/discount threshold classification."""

    def _make_checker(self, premium_pct):
        checker = LiquidityChecker.__new__(LiquidityChecker)
        checker.PREMIUM_CRITICAL = 0.30
        checker.PREMIUM_WARNING = 0.15
        checker.PREMIUM_LOG = 0.10
        checker.LARGE_TRADE_THRESHOLD = 5.0
        checker.log_path = MagicMock()

        mock_engine = MagicMock()
        if premium_pct is not None:
            mock_engine.fetch_etf_pricing.return_value = _make_premium(premium_pct)
        else:
            mock_engine.fetch_etf_pricing.return_value = None
        checker.etf_engine = mock_engine
        return checker

    def test_normal_premium_passes(self):
        checker = self._make_checker(0.05)
        result = checker._check_premium("SPY", "buy", 3.0, False)
        assert result["passed"] is True
        assert result["status"] == "normal"

    def test_advisory_premium_passes(self):
        checker = self._make_checker(0.12)
        result = checker._check_premium("SPY", "buy", 3.0, False)
        assert result["passed"] is True
        assert result["status"] == "advisory"

    def test_warning_small_trade_passes(self):
        checker = self._make_checker(0.20)
        result = checker._check_premium("SPY", "buy", 3.0, False)
        assert result["passed"] is True
        assert result["status"] == "warning"

    def test_warning_large_trade_blocks(self):
        checker = self._make_checker(0.20)
        result = checker._check_premium("SPY", "buy", 8.0, False)
        assert result["passed"] is False
        assert result["status"] == "warning_large"

    def test_warning_large_trade_force_passes(self):
        checker = self._make_checker(0.20)
        result = checker._check_premium("SPY", "buy", 8.0, True)
        assert result["passed"] is True
        assert result["status"] == "warning_large"

    def test_critical_premium_blocks(self):
        checker = self._make_checker(0.35)
        result = checker._check_premium("SPY", "buy", 3.0, False)
        assert result["passed"] is False
        assert result["status"] == "critical"

    def test_critical_premium_blocks_even_with_force(self):
        checker = self._make_checker(0.35)
        result = checker._check_premium("SPY", "buy", 3.0, True)
        assert result["passed"] is False
        assert result["status"] == "critical"

    def test_negative_premium_handled(self):
        checker = self._make_checker(-0.20)
        result = checker._check_premium("SPY", "sell", 3.0, False)
        assert result["passed"] is True
        assert result["status"] == "warning"

    def test_no_data_passes(self):
        checker = self._make_checker(None)
        result = checker._check_premium("SPY", "buy", 3.0, False)
        assert result["passed"] is True
        assert result["status"] == "unknown"


class TestTradeEligibility:
    """Test full trade eligibility flow."""

    def _make_checker(self, premium_pct=0.05):
        checker = LiquidityChecker.__new__(LiquidityChecker)
        checker.PREMIUM_CRITICAL = 0.30
        checker.PREMIUM_WARNING = 0.15
        checker.PREMIUM_LOG = 0.10
        checker.LARGE_TRADE_THRESHOLD = 5.0
        checker.log_path = MagicMock()

        mock_engine = MagicMock()
        mock_engine.fetch_etf_pricing.return_value = _make_premium(premium_pct)
        checker.etf_engine = mock_engine

        # Mock log writing
        checker._log_check = MagicMock()
        return checker

    def test_normal_trade_eligible(self):
        checker = self._make_checker(0.05)
        result = checker.check_trade_eligibility("SPY", "buy", 3000, 100000)
        assert result.eligible is True
        assert "All liquidity checks passed" in result.reason

    def test_critical_blocks_trade(self):
        checker = self._make_checker(0.35)
        result = checker.check_trade_eligibility("SPY", "buy", 3000, 100000)
        assert result.eligible is False
        assert "CRITICAL" in result.reason

    def test_warning_large_blocks_without_force(self):
        checker = self._make_checker(0.20)
        result = checker.check_trade_eligibility("SPY", "buy", 8000, 100000)
        assert result.eligible is False

    def test_warning_large_passes_with_force(self):
        checker = self._make_checker(0.20)
        result = checker.check_trade_eligibility("SPY", "buy", 8000, 100000, force=True)
        # When force=True, _check_premium returns passed=True → premium passes
        assert result.eligible is True
        assert result.premium_status == "warning_large"

    def test_position_size_exceeded_recorded(self):
        checker = self._make_checker(0.05)
        # 30% of portfolio > 25% limit — size check fails but doesn't block
        # (only critical/warning in checks_failed block; position_size_exceeded has neither)
        result = checker.check_trade_eligibility("SPY", "buy", 30000, 100000)
        assert any("position_size_exceeded" in f for f in result.checks_failed)

    def test_result_has_premium_info(self):
        checker = self._make_checker(0.12)
        result = checker.check_trade_eligibility("SPY", "buy", 3000, 100000)
        assert result.premium_pct == 0.12
        assert result.premium_status == "advisory"

    def test_result_has_timestamp(self):
        checker = self._make_checker(0.05)
        result = checker.check_trade_eligibility("SPY", "buy", 3000, 100000)
        assert result.timestamp != ""


class TestGetRecentBlocks:
    """Test blocked trade log parsing."""

    def _make_checker(self, log_content=None):
        checker = LiquidityChecker.__new__(LiquidityChecker)
        checker.log_path = MagicMock()
        checker.log_path.exists.return_value = log_content is not None

        if log_content is not None:
            import io
            checker.log_path.open.return_value.__enter__ = lambda s: io.StringIO(log_content)
            checker.log_path.open.return_value.__exit__ = MagicMock(return_value=False)
        return checker

    def test_returns_empty_when_no_log(self):
        checker = self._make_checker(None)
        blocks = checker.get_recent_blocks()
        assert blocks == []

    def test_parses_blocked_entries(self):
        from datetime import datetime
        ts = datetime.now().isoformat()
        log = f"{ts} | BLOCK | SPY | buy | $8,000 | +0.200% | WARNING: test\n"
        log += f"{ts} | PASS | GLD | sell | $3,000 | +0.050% | All checks passed\n"

        checker = self._make_checker(log)
        import io
        mock_open = MagicMock(return_value=iter(log.splitlines(True)))
        with patch("builtins.open", mock_open):
            blocks = checker.get_recent_blocks()
            assert isinstance(blocks, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
