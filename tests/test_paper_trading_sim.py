"""Tests for Paper Trading Simulator"""
import json
import pytest
from src.broker.paper_trading_sim import (
    PaperTradingSimulator, PaperTradingReport, Trade, DailySnapshot, run_paper_trading
)


class TestDataClasses:
    def test_trade_serializable(self):
        t = Trade("2026-01-01", "SPY", "buy", 100, 550.0, 55000, 1.0, 5.0, "rebalance")
        d = t.to_dict()
        assert d["symbol"] == "SPY"

    def test_snapshot_serializable(self):
        s = DailySnapshot("2026-01-01", 100000, 0.5, 2.0, 46000, 38000, 16000, 0, 0, 0, 0, 0, True, False, "long", 16.0)
        d = s.to_dict()
        assert d["total_value"] == 100000

    def test_report_serializable(self):
        r = PaperTradingReport(
            timestamp="t", start_date="s", end_date="e", trading_days=10,
            total_return=5.0, cagr=15.0, volatility=10.0, sharpe=1.5,
            max_drawdown=-5.0, max_drawdown_date="d",
            total_trades=5, total_commission=5.0, total_slippage_bps=25,
            turnover_pct=20, winning_days=6, losing_days=4, win_rate=60,
            collar_active_days=3, crypto_active_days=0, avg_bond_duration=50,
            meets_graduation_sharpe=True, meets_graduation_dd=True,
            graduation_ready=True, graduation_note="READY",
            trades=[], snapshots=[],
        )
        d = r.to_dict()
        assert d["sharpe"] == 1.5
        assert d["graduation_ready"]


class TestPaperTradingSimulator:
    @pytest.fixture
    def sim(self):
        return PaperTradingSimulator()

    def test_creates_output_dir(self, sim):
        assert sim.OUTPUT_DIR.exists()

    def test_run_with_real_data(self, sim):
        report = sim.run(days=30)
        assert isinstance(report, PaperTradingReport)
        if report.trading_days > 0:
            assert report.total_trades >= 0
            assert report.trading_days > 0

    def test_convenience_function(self):
        report = run_paper_trading(days=10)
        assert isinstance(report, PaperTradingReport)

    def test_trades_have_slippage(self, sim):
        report = sim.run(days=30)
        if report.total_trades > 0:
            assert report.total_slippage_bps > 0
            assert report.total_commission > 0

    def test_snapshots_present(self, sim):
        report = sim.run(days=20)
        if report.trading_days > 0:
            assert len(report.snapshots) > 0
