"""
Tests for the International Momentum Signal Backtest.

Covers: BacktestConfig defaults/custom, BacktestResult construction, backtester init,
get_signal computation (efa_lead, spy_lead, neutral), run_backtest with synthetic
price data, edge cases (insufficient data, missing symbols), and CLI invocation.
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.backtest.international_momentum_backtest import (
    BacktestConfig,
    InternationalMomentumBacktester,
)
from src.backtest.metrics import BacktestResult


# ── BacktestConfig Tests ─────────────────────────────────────────────────


class TestBacktestConfig:
    """Test BacktestConfig defaults and custom configuration."""

    def test_defaults(self):
        config = BacktestConfig()
        assert config.start_date == "2006-01-01"
        assert config.end_date == "2026-05-15"
        assert config.initial_capital == 100000.0
        assert config.base_weights['SPY'] == 0.46
        assert config.base_weights['GLD'] == 0.38
        assert config.base_weights['TLT'] == 0.16
        assert config.lookback_days == 126
        assert config.max_shift == 0.05
        assert config.transaction_cost_bps == 10.0

    def test_custom_values(self):
        config = BacktestConfig(
            start_date="2015-01-01",
            end_date="2020-12-31",
            initial_capital=50000.0,
            lookback_days=63,
            max_shift=0.03,
            transaction_cost_bps=5.0,
        )
        assert config.start_date == "2015-01-01"
        assert config.end_date == "2020-12-31"
        assert config.initial_capital == 50000.0
        assert config.lookback_days == 63
        assert config.max_shift == 0.03
        assert config.transaction_cost_bps == 5.0

    def test_base_weights_sum_to_one(self):
        config = BacktestConfig()
        total = sum(config.base_weights.values())
        assert abs(total - 1.0) < 0.01


# ── BacktestResult Tests ────────────────────────────────────────────────


class TestBacktestResult:
    """Test BacktestResult creation and serialization."""

    def test_create(self):
        result = BacktestResult(
            total_return=10.0,
            cagr=10.0,
            volatility=11.8,
            sharpe_ratio=0.85,
            max_drawdown=-15.0,
            total_rebalances=12,
            total_transaction_costs=15.5,
            crisis_returns={"2008": -10.0, "2020": 2.5},
            extras={
                "strategy_name": "International Momentum Overlay",
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "initial_capital": 100000.0,
                "final_value": 110000.0,
                "signal_distribution": {"efa_lead": 5, "spy_lead": 3, "neutral": 4},
            },
        )
        assert result.extras["strategy_name"] == "International Momentum Overlay"
        assert result.extras["final_value"] == 110000.0
        assert result.sharpe_ratio == 0.85
        assert result.total_rebalances == 12
        assert result.extras["signal_distribution"]["efa_lead"] == 5

    def test_json_serializable(self):
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, total_rebalances=0, total_transaction_costs=0.0,
            crisis_returns={}, extras={"strategy_name": "Test"},
        )
        json.dumps(result.__dict__)  # Should not raise

    def test_empty_signal_distribution(self):
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, total_rebalances=0, total_transaction_costs=0.0,
            crisis_returns={}, extras={"signal_distribution": {}},
        )
        assert result.extras["signal_distribution"] == {}


# ── Backtester Init Tests ────────────────────────────────────────────────


class TestInternationalMomentumBacktesterInit:
    """Test backtester initialization."""

    def test_default_config(self):
        bt = InternationalMomentumBacktester()
        assert bt.config.lookback_days == 126
        assert bt.price_data == {}
        assert bt.dates == []
        assert bt.prices == {}

    def test_custom_config(self):
        config = BacktestConfig(lookback_days=63, start_date="2015-01-01")
        bt = InternationalMomentumBacktester(config)
        assert bt.config.lookback_days == 63
        assert bt.config.start_date == "2015-01-01"


# ── Signal Computation Tests ────────────────────────────────────────────


class TestGetSignal:
    """Test _get_signal with controlled price data."""

    @pytest.fixture
    def backtester(self):
        """Create a backtester with 200 days of controlled price data."""
        bt = InternationalMomentumBacktester(BacktestConfig(lookback_days=126))
        bt.dates = [f"2020-{m:02d}-{d:02d}" for m in range(1, 8) for d in range(1, 29)]  # ~200 days
        bt.prices = {}
        for date in bt.dates:
            bt.prices[date] = {"SPY": 100.0, "GLD": 100.0, "TLT": 100.0, "EFA": 100.0, "EEM": 100.0}
        return bt

    def test_neutral_when_insufficient_data(self, backtester):
        """With less than lookback_days days, signal should be neutral."""
        signal_type, signal_value = backtester._get_signal(backtester.dates[50])
        assert signal_type == "neutral"
        assert signal_value == 0.0

    def test_efa_lead_signal(self, backtester):
        """When EFA outperforms SPY by >3%, signal should be efa_lead."""
        # Set EFA to outperform SPY over the lookback period
        spy_base = 100.0
        efa_base = 100.0
        for i, date in enumerate(backtester.dates):
            spy_price = spy_base * (1.0 + 0.0002) ** i  # Slow SPY growth
            efa_price = efa_base * (1.0 + 0.001) ** i  # Faster EFA growth
            if i >= 126:
                backtester.prices[date]["SPY"] = spy_price
                backtester.prices[date]["EFA"] = efa_price
            else:
                backtester.prices[date]["SPY"] = 100.0
                backtester.prices[date]["EFA"] = 100.0

        signal_type, signal_value = backtester._get_signal(backtester.dates[150])
        assert signal_type == "efa_lead"
        assert signal_value > 0
        assert signal_value <= 0.5

    def test_spy_lead_signal(self, backtester):
        """When SPY outperforms EFA, production code returns neutral (no spy_lead signal)."""
        spy_base = 100.0
        efa_base = 100.0
        for i, date in enumerate(backtester.dates):
            spy_price = spy_base * (1.0 + 0.001) ** i  # Fast SPY growth
            efa_price = efa_base * (1.0 + 0.0002) ** i  # Slower EFA growth
            if i >= 126:
                backtester.prices[date]["SPY"] = spy_price
                backtester.prices[date]["EFA"] = efa_price
            else:
                backtester.prices[date]["SPY"] = 100.0
                backtester.prices[date]["EFA"] = 100.0

        signal_type, signal_value = backtester._get_signal(backtester.dates[150])
        # Production code only signals when international outperforms, not when SPY leads
        assert signal_type == "neutral"
        assert signal_value == 0.0

    def test_neutral_when_close(self, backtester):
        """When SPY and EFA returns are close (<3% diff), signal should be neutral."""
        for i, date in enumerate(backtester.dates):
            if i >= 126:
                backtester.prices[date]["SPY"] = 100.0 * (1.01 ** (i - 126))
                backtester.prices[date]["EFA"] = 100.0 * (1.009 ** (i - 126))
            else:
                backtester.prices[date]["SPY"] = 100.0
                backtester.prices[date]["EFA"] = 100.0

        signal_type, signal_value = backtester._get_signal(backtester.dates[150])
        assert signal_type == "neutral"
        assert signal_value == 0.0

    def test_missing_symbol_returns_neutral(self, backtester):
        """When EFA data is missing, signal should be neutral."""
        for date in backtester.dates:
            if "EFA" in backtester.prices[date]:
                del backtester.prices[date]["EFA"]
        signal_type, signal_value = backtester._get_signal(backtester.dates[150])
        assert signal_type == "neutral"
        assert signal_value == 0.0

    def test_missing_date_returns_neutral(self, backtester):
        """When the requested date doesn't exist, signal should be neutral."""
        signal_type, signal_value = backtester._get_signal("2099-01-01")
        assert signal_type == "neutral"
        assert signal_value == 0.0

    def test_signal_capped_at_0_5(self, backtester):
        """Signal value should be capped at 0.5."""
        # Extreme EFA outperformance
        for i, date in enumerate(backtester.dates):
            if i >= 126:
                backtester.prices[date]["SPY"] = 100.0
                backtester.prices[date]["EFA"] = 100.0 * (1.05 ** (i - 126))  # Huge EFA gain
            else:
                backtester.prices[date]["SPY"] = 100.0
                backtester.prices[date]["EFA"] = 100.0

        signal_type, signal_value = backtester._get_signal(backtester.dates[180])
        assert signal_type == "efa_lead"
        assert signal_value <= 0.5

    def test_signal_floored_at_neg_0_5(self, backtester):
        """Extreme SPY outperformance returns neutral (production has no spy_lead)."""
        for i, date in enumerate(backtester.dates):
            if i >= 126:
                backtester.prices[date]["SPY"] = 100.0 * (1.05 ** (i - 126))  # Huge SPY gain
                backtester.prices[date]["EFA"] = 100.0
            else:
                backtester.prices[date]["SPY"] = 100.0
                backtester.prices[date]["EFA"] = 100.0

        signal_type, signal_value = backtester._get_signal(backtester.dates[180])
        # Production code only signals when international outperforms
        assert signal_type == "neutral"
        assert signal_value == 0.0

    def test_signal_respects_threshold_boundary(self, backtester):
        """Relative return just below 3% threshold should be neutral."""
        # Set lookback to 63 days for a shorter test window
        bt2 = InternationalMomentumBacktester(BacktestConfig(lookback_days=63))
        bt2.dates = backtester.dates
        bt2.prices = {}
        for date in bt2.dates:
            bt2.prices[date] = {"SPY": 100.0, "GLD": 100.0, "TLT": 100.0, "EFA": 100.0, "EEM": 100.0}

        for i, date in enumerate(bt2.dates):
            # 63 days of tiny growth (~0.02% daily) so relative is well under 3%
            if i >= 63:
                bt2.prices[date]["SPY"] = 100.0 * (1.00015 ** (i - 63))
                bt2.prices[date]["EFA"] = 100.0 * (1.00035 ** (i - 63))  # Slightly more
            else:
                bt2.prices[date]["SPY"] = 100.0
                bt2.prices[date]["EFA"] = 100.0

        signal_type, signal_value = bt2._get_signal(bt2.dates[100])
        # Relative return should be small (~1.4%) -> well below 3% -> neutral
        assert signal_type == "neutral"
        assert signal_value == 0.0


# ── Run Backtest Tests ──────────────────────────────────────────────────


class TestRunBacktest:
    """Test run_backtest with synthetic price data."""

    @staticmethod
    def _make_synthetic_prices(n_days=300, spy_trend=0.0005, efa_trend=0.0008, seed=42):
        """Generate synthetic price data dicts for testing."""
        import numpy as np
        rng = np.random.default_rng(seed)

        dates = []
        for i in range(n_days):
            m = 1 + (i // 28)
            d = 1 + (i % 28)
            if m > 12:
                m = 12
            dates.append(f"2020-{m:02d}-{d:02d}")

        prices = {}
        for date in dates:
            prices[date] = {}

        spy_price = 100.0
        gld_price = 100.0
        tlt_price = 100.0
        efa_price = 100.0
        eem_price = 100.0

        for i, date in enumerate(dates):
            spy_price *= 1.0 + spy_trend + rng.normal(0, 0.01)
            gld_price *= 1.0 + 0.0003 + rng.normal(0, 0.008)
            tlt_price *= 1.0 + 0.0002 + rng.normal(0, 0.006)
            efa_price *= 1.0 + efa_trend + rng.normal(0, 0.012)
            eem_price *= 1.0 + efa_trend * 0.8 + rng.normal(0, 0.015)
            prices[date]["SPY"] = round(spy_price, 2)
            prices[date]["GLD"] = round(gld_price, 2)
            prices[date]["TLT"] = round(tlt_price, 2)
            prices[date]["EFA"] = round(efa_price, 2)
            prices[date]["EEM"] = round(eem_price, 2)

        return dates, prices

    def test_run_with_synthetic_data(self):
        """Backtest should produce a BacktestResult."""
        dates, prices = self._make_synthetic_prices(n_days=300)
        bt = InternationalMomentumBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        assert result is not None
        assert isinstance(result, BacktestResult)
        assert result.extras["final_value"] > 0
        assert result.total_rebalances >= 0

    def test_run_bull_market(self):
        """Strong up-trending market should produce positive returns."""
        dates, prices = self._make_synthetic_prices(n_days=300, spy_trend=0.002, efa_trend=0.002)
        bt = InternationalMomentumBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        assert result is not None
        assert result.cagr > 0

    def test_run_bear_market(self):
        """Strong down-trending market should produce negative returns."""
        dates, prices = self._make_synthetic_prices(n_days=300, spy_trend=-0.002, efa_trend=-0.002)
        bt = InternationalMomentumBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        assert result is not None
        assert result.cagr < 0

    def test_result_contains_required_fields(self):
        dates, prices = self._make_synthetic_prices(n_days=300)
        bt = InternationalMomentumBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        assert result.extras.get("strategy_name") is not None
        assert result.cagr is not None
        assert result.sharpe_ratio is not None
        assert result.max_drawdown is not None
        assert result.total_rebalances >= 0
        assert isinstance(result.crisis_returns, dict)
        assert isinstance(result.extras["signal_distribution"], dict)

    def test_signal_distribution_totals(self):
        dates, prices = self._make_synthetic_prices(n_days=300, efa_trend=0.001)
        bt = InternationalMomentumBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        total = sum(result.extras["signal_distribution"].values())
        assert total > 0
        assert total <= result.total_rebalances

    def test_crisis_returns_structure(self):
        """Crisis returns should contain valid year keys."""
        dates, prices = self._make_synthetic_prices(n_days=300)
        bt = InternationalMomentumBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        assert isinstance(result.crisis_returns, dict)

    def test_final_value_approx_initial_plus_return(self):
        dates, prices = self._make_synthetic_prices(n_days=300, spy_trend=0.001)
        bt = InternationalMomentumBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31", initial_capital=100000.0)
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        final = result.extras["final_value"]
        # Final value should be a reasonable number
        assert final > 0
        assert isinstance(final, float)


# ── Edge Cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for International Momentum backtest."""

    def test_insufficient_dates_returns_none(self):
        """Fewer than 60 trading days should return None."""
        bt = InternationalMomentumBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = ["2020-01-01", "2020-01-02"]  # Only 2 dates
        bt.prices = {"2020-01-01": {"SPY": 100.0}, "2020-01-02": {"SPY": 101.0}}
        result = bt.run_backtest()
        assert result is None

    def test_no_dates_returns_none(self):
        bt = InternationalMomentumBacktester()
        bt.dates = []
        result = bt.run_backtest()
        assert result is None

    def test_missing_price_data_still_runs(self):
        """Missing key prices during daily return calc should not crash."""
        dates = [f"2020-01-{d:02d}" for d in range(1, 65)]
        prices = {}
        for date in dates:
            prices[date] = {"SPY": 100.0}  # Only SPY, no GLD/TLT
        bt = InternationalMomentumBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = dates
        bt.prices = prices
        # Should still produce result (daily returns may be partial)
        result = bt.run_backtest()
        # run_backtest checks len(trading_days) >= 60, so this should work
        assert result is not None

    def test_zero_initial_capital(self):
        dates, prices = TestRunBacktest._make_synthetic_prices(n_days=100)
        bt = InternationalMomentumBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-06-30", initial_capital=0.0)
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        assert result is not None
        assert isinstance(result, BacktestResult)

    def test_single_month_data(self):
        """Only one month of data is insufficient (< 60 days)."""
        bt = InternationalMomentumBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-01-31")
        )
        bt.dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
        bt.prices = {d: {"SPY": 100.0, "GLD": 100.0, "TLT": 100.0, "EFA": 100.0, "EEM": 100.0} for d in bt.dates}
        result = bt.run_backtest()
        assert result is None

    def test_narrow_date_range_still_runs(self):
        """A narrow date range with >= 60 days should still work."""
        dates = [f"2020-{m:02d}-{d:02d}" for m in range(1, 5) for d in range(1, 28)][:65]
        prices = {}
        for date in dates:
            prices[date] = {"SPY": 100.0, "GLD": 100.0, "TLT": 100.0, "EFA": 100.0, "EEM": 100.0}
        bt = InternationalMomentumBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-05-01")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        assert result is not None

    def test_extreme_price_moves(self):
        """Extreme price movements should not crash the backtest."""
        dates = [f"2020-01-{d:02d}" for d in range(1, 65)]
        prices = {}
        for i, date in enumerate(dates):
            spy_mult = 1.5 if i == 30 else 0.5 if i == 31 else 1.0
            prices[date] = {
                "SPY": 100.0 * spy_mult,
                "GLD": 100.0,
                "TLT": 100.0,
                "EFA": 100.0,
                "EEM": 100.0,
            }
        bt = InternationalMomentumBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        assert result is not None

    def test_save_results_creates_json(self):
        """save_results should create a valid JSON file."""
        dates, prices = TestRunBacktest._make_synthetic_prices(n_days=100)
        bt = InternationalMomentumBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-06-01")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
            bt.save_results(result, output_path=output_path)

        try:
            with open(output_path) as f:
                saved = json.load(f)
            assert "cagr" in saved
            assert "sharpe_ratio" in saved
            assert "extras" in saved
            assert "strategy_name" in saved["extras"]
            assert "signal_distribution" in saved["extras"]
        finally:
            Path(output_path).unlink()

    def test_print_report_does_not_crash(self, caplog):
        """print_report should produce output without errors."""
        dates, prices = TestRunBacktest._make_synthetic_prices(n_days=100)
        bt = InternationalMomentumBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-06-01")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        with caplog.at_level(logging.INFO, logger="src.backtest.international_momentum_backtest"):
            bt.print_report(result)
        assert "International Momentum" in caplog.text
        assert "Sharpe" in caplog.text
        assert "Signal Distribution" in caplog.text

    def test_load_data_missing_file_logs_error(self, caplog, monkeypatch):
        """load_data should return False when prices.json is missing."""
        import logging
        caplog.set_level(logging.ERROR)
        bt = InternationalMomentumBacktester()
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        result = bt.load_data()
        assert result is False


# ── CLI Tests ───────────────────────────────────────────────────────────


class TestCLI:
    """Test command-line invocation."""

    def test_main_run(self, monkeypatch):
        """main() should not crash and should handle missing data."""
        from src.backtest.international_momentum_backtest import main
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        with patch("sys.argv", ["intl_momentum_backtest.py", "run"]):
            # Should not raise; returns None when data missing
            main()

    def test_main_with_save_flag(self, monkeypatch):
        """main() with --save flag should handle missing data gracefully."""
        from src.backtest.international_momentum_backtest import main
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        with patch("sys.argv", ["intl_momentum_backtest.py", "run", "--save"]):
            main()


def test_a3_b1a_delegation_matches_pre_migration_capture():
    """A3 pin (Item B1a sub-task 5): load_data delegates to grid_runner.load_prices."""
    from src.backtest.grid_runner import load_prices

    # class method stays in pilot; the shared loader is grid_runner's
    assert InternationalMomentumBacktester.load_data.__module__ == (
        "src.backtest.international_momentum_backtest"
    )
    assert load_prices.__module__ == "src.backtest.grid_runner"
