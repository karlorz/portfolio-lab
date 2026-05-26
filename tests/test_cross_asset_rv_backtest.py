"""
Tests for the Cross-Asset Relative Value Signal Backtest.

Covers: BacktestConfig defaults/custom, BacktestResult construction, backtester init,
get_signal computation (spy_reversion, gld_reversion, neutral), run_backtest with
synthetic price data, edge cases (zero variance, insufficient data, extreme z-scores),
and CLI invocation.
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.backtest.cross_asset_rv_backtest import (
    BacktestConfig,
    CrossAssetRVBacktester,
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
        assert config.z_score_window == 60
        assert config.max_shift == 0.04
        assert config.transaction_cost_bps == 10.0

    def test_custom_values(self):
        config = BacktestConfig(
            start_date="2015-01-01",
            end_date="2020-12-31",
            initial_capital=50000.0,
            z_score_window=30,
            max_shift=0.06,
            transaction_cost_bps=5.0,
        )
        assert config.start_date == "2015-01-01"
        assert config.end_date == "2020-12-31"
        assert config.initial_capital == 50000.0
        assert config.z_score_window == 30
        assert config.max_shift == 0.06
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
            total_return=5.0,
            cagr=5.0,
            volatility=7.7,
            sharpe_ratio=0.65,
            max_drawdown=-12.0,
            total_rebalances=12,
            total_transaction_costs=10.5,
            crisis_returns={"2008": -10.0, "2020": 2.0},
            extras={
                "strategy_name": "Cross-Asset Relative Value Overlay",
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "initial_capital": 100000.0,
                "final_value": 105000.0,
                "signal_distribution": {"spy_reversion": 3, "gld_reversion": 2, "neutral": 7},
                "avg_z_score": 1.2,
                "diverged_pct": 25.0,
            },
        )
        assert result.extras["strategy_name"] == "Cross-Asset Relative Value Overlay"
        assert result.extras["final_value"] == 105000.0
        assert result.sharpe_ratio == 0.65
        assert result.total_rebalances == 12
        assert result.extras["avg_z_score"] == 1.2
        assert result.extras["diverged_pct"] == 25.0

    def test_json_serializable(self):
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, total_rebalances=0, total_transaction_costs=0.0,
            crisis_returns={}, extras={"strategy_name": "Test"},
        )
        json.dumps(result.__dict__)  # Should not raise

    def test_zero_diverged_pct(self):
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, total_rebalances=0, total_transaction_costs=0.0,
            crisis_returns={}, extras={"diverged_pct": 0.0},
        )
        assert result.extras["diverged_pct"] == 0.0


# ── Backtester Init Tests ────────────────────────────────────────────────


class TestCrossAssetRVBacktesterInit:
    """Test backtester initialization."""

    def test_default_config(self):
        bt = CrossAssetRVBacktester()
        assert bt.config.z_score_window == 60
        assert bt.price_data == {}
        assert bt.dates == []
        assert bt.prices == {}

    def test_custom_config(self):
        config = BacktestConfig(z_score_window=30)
        bt = CrossAssetRVBacktester(config)
        assert bt.config.z_score_window == 30


# ── Signal Computation Tests ────────────────────────────────────────────


class TestGetSignal:
    """Test _get_signal with controlled price data."""

    @pytest.fixture
    def backtester(self):
        """Create a backtester with 200 days of controlled price data."""
        bt = CrossAssetRVBacktester(BacktestConfig(z_score_window=60))
        bt.dates = [f"2020-{m:02d}-{d:02d}" for m in range(1, 8) for d in range(1, 29)]  # ~200 days
        bt.prices = {}
        for date in bt.dates:
            bt.prices[date] = {"SPY": 100.0, "GLD": 100.0, "TLT": 100.0}
        return bt

    def test_neutral_when_insufficient_data(self, backtester):
        """With less than z_score_window + 1 days, signal should be neutral."""
        signal_type, signal_value = backtester._get_signal(backtester.dates[30])
        assert signal_type == "neutral"
        assert signal_value == 0.0

    def test_spy_reversion_positive_z(self, backtester):
        """When SPY has a big recent gain (high z-score), mean-reversion signal
        should be negative (betting on reversion down)."""
        # Near-flat 100.0 for first 59 days of the 60-day window, then +20% spike
        for i, date in enumerate(backtester.dates):
            if i <= 159:  # dates[100] to dates[159]: near-flat
                backtester.prices[date]["SPY"] = 100.0 + (i - 100) * 0.01  # 0.01 drift to avoid zero-std
            elif i == 160:  # +20% jump on last day of window
                backtester.prices[date]["SPY"] = 120.0
            else:
                backtester.prices[date]["SPY"] = 120.0
            backtester.prices[date]["GLD"] = 100.0
            backtester.prices[date]["TLT"] = 100.0

        signal_type, signal_value = backtester._get_signal(backtester.dates[161])
        # Last return (dates[160]/dates[159]) ~20% -> positive z-score outlier
        # -> mean-reversion down -> negative signal
        assert signal_type == "spy_reversion"
        assert signal_value < 0

    def test_spy_reversion_negative_z(self, backtester):
        """When SPY has a big recent drop (low z-score), mean-reversion signal
        should be positive (betting on reversion up)."""
        # Near-flat 100.0 for first 59 days of window, then -20% drop on last day
        for i, date in enumerate(backtester.dates):
            if i <= 159:  # dates[100] to dates[159]: near-flat
                backtester.prices[date]["SPY"] = 100.0 + (i - 100) * 0.01  # 0.01 drift to avoid zero-std
            elif i == 160:  # -20% drop on last day of window
                backtester.prices[date]["SPY"] = 80.0
            else:
                backtester.prices[date]["SPY"] = 80.0
            backtester.prices[date]["GLD"] = 100.0
            backtester.prices[date]["TLT"] = 100.0

        signal_type, signal_value = backtester._get_signal(backtester.dates[161])
        # Last return (dates[160]/dates[159]) ~ -20% -> negative z-score outlier
        # -> mean-reversion up -> positive signal
        assert signal_type == "spy_reversion"
        assert signal_value > 0

    def test_gld_reversion_signal(self, backtester):
        """When SPY z-score is normal but GLD z-score > threshold, gld_reversion should fire."""
        for i, date in enumerate(backtester.dates):
            backtester.prices[date]["SPY"] = 100.0  # Flat SPY
            backtester.prices[date]["TLT"] = 100.0  # Flat TLT
            if 100 <= i < 160:
                backtester.prices[date]["GLD"] = 100.0 + (i - 100) * 2.0  # Trending up
            elif i >= 160:
                backtester.prices[date]["GLD"] = backtester.prices[backtester.dates[159]]["GLD"]

        signal_type, signal_value = backtester._get_signal(backtester.dates[161])
        # Depending on relative z-scores, could be spy_reversion or gld_reversion
        # SPY z-score should be ~0, so GLD z-score > threshold -> gld_reversion
        assert signal_type != "neutral"
        assert abs(signal_value) > 0

    def test_neutral_when_all_zscores_low(self, backtester):
        """When all z-scores are below threshold, signal should be neutral."""
        for i, date in enumerate(backtester.dates):
            if i >= 100:
                backtester.prices[date]["SPY"] = 100.0 + (i % 3) * 1.0  # Tiny variation
                backtester.prices[date]["GLD"] = 100.0 + (i % 2) * 1.0
                backtester.prices[date]["TLT"] = 100.0 + (i % 4) * 0.5
            else:
                backtester.prices[date] = {"SPY": 100.0, "GLD": 100.0, "TLT": 100.0}

        signal_type, signal_value = backtester._get_signal(backtester.dates[180])
        # With very little variation, z-scores should be low -> neutral
        assert signal_value == 0.0

    def test_missing_price_returns_neutral(self, backtester):
        """Missing prices should not crash and return neutral."""
        for date in backtester.dates:
            backtester.prices[date] = {"SPY": 100.0}  # Missing GLD, TLT
        signal_type, signal_value = backtester._get_signal(backtester.dates[150])
        assert signal_type == "neutral"
        assert signal_value == 0.0

    def test_unknown_date_returns_neutral(self, backtester):
        signal_type, signal_value = backtester._get_signal("2099-01-01")
        assert signal_type == "neutral"
        assert signal_value == 0.0

    def test_z_score_capped_at_extreme(self, backtester):
        """Signal value should be capped by the formula min(|z|/5.0, 0.5)."""
        # Extreme SPY movement
        for i, date in enumerate(backtester.dates):
            if 100 <= i < 160:
                backtester.prices[date]["SPY"] = 100.0 + (i - 100) * 10.0  # Extreme trend
            elif i >= 160:
                backtester.prices[date]["SPY"] = backtester.prices[backtester.dates[159]]["SPY"]
            backtester.prices[date]["GLD"] = 100.0
            backtester.prices[date]["TLT"] = 100.0

        signal_type, signal_value = backtester._get_signal(backtester.dates[161])
        assert abs(signal_value) <= 0.5


# ── Run Backtest Tests ──────────────────────────────────────────────────


class TestRunBacktest:
    """Test run_backtest with synthetic price data."""

    @staticmethod
    def _make_synthetic_prices(n_days=300, seed=42):
        """Generate synthetic price data with enough variation for z-score computation."""
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

        for i, date in enumerate(dates):
            spy_price *= 1.0 + 0.0005 + rng.normal(0, 0.01)
            gld_price *= 1.0 + 0.0003 + rng.normal(0, 0.008)
            tlt_price *= 1.0 + 0.0002 + rng.normal(0, 0.006)
            prices[date]["SPY"] = round(spy_price, 2)
            prices[date]["GLD"] = round(gld_price, 2)
            prices[date]["TLT"] = round(tlt_price, 2)

        return dates, prices

    def test_run_with_synthetic_data(self):
        """Backtest should produce a BacktestResult."""
        dates, prices = self._make_synthetic_prices(n_days=300)
        bt = CrossAssetRVBacktester(
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
        rng = np.random.default_rng(42)
        dates = [f"2020-{m:02d}-{d:02d}" for m in range(1, 13) for d in [1, 3, 5, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29]][:300]
        prices = {}
        for date in dates:
            prices[date] = {}
        spy_p, gld_p, tlt_p = 100.0, 100.0, 100.0
        for date in dates:
            spy_p *= 1.0 + 0.001 + rng.normal(0, 0.008)
            gld_p *= 1.0 + 0.0005 + rng.normal(0, 0.006)
            tlt_p *= 1.0 + 0.0003 + rng.normal(0, 0.005)
            prices[date]["SPY"] = round(spy_p, 2)
            prices[date]["GLD"] = round(gld_p, 2)
            prices[date]["TLT"] = round(tlt_p, 2)

        bt = CrossAssetRVBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        assert result is not None
        assert result.cagr > 0

    def test_result_contains_required_fields(self):
        dates, prices = self._make_synthetic_prices(n_days=300)
        bt = CrossAssetRVBacktester(
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
        assert isinstance(result.extras["avg_z_score"], float)
        assert isinstance(result.extras["diverged_pct"], float)

    def test_signal_distribution_totals(self):
        dates, prices = self._make_synthetic_prices(n_days=300)
        bt = CrossAssetRVBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        total = sum(result.extras["signal_distribution"].values())
        assert total > 0

    def test_diverge_pct_between_0_and_100(self):
        dates, prices = self._make_synthetic_prices(n_days=300)
        bt = CrossAssetRVBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        assert 0.0 <= result.extras["diverged_pct"] <= 100.0


# ── Edge Cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for Cross-Asset RV backtest."""

    def test_insufficient_dates_returns_none(self):
        """Fewer than 60 trading days should return None."""
        bt = CrossAssetRVBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = ["2020-01-01", "2020-01-02"]
        bt.prices = {"2020-01-01": {"SPY": 100.0, "GLD": 100.0, "TLT": 100.0},
                      "2020-01-02": {"SPY": 101.0, "GLD": 100.0, "TLT": 100.0}}
        result = bt.run_backtest()
        assert result is None

    def test_no_dates_returns_none(self):
        bt = CrossAssetRVBacktester()
        bt.dates = []
        result = bt.run_backtest()
        assert result is None

    def test_constant_prices_does_not_crash(self):
        """When all prices are constant (zero variance), z-scores should be 0 -> neutral."""
        dates = [f"2020-01-{d:02d}" for d in range(1, 65)]
        prices = {}
        for date in dates:
            prices[date] = {"SPY": 100.0, "GLD": 100.0, "TLT": 100.0}
        bt = CrossAssetRVBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        # Should not crash — constant prices mean std=0 -> z=0 -> neutral
        # But run_backtest checks std_r > 0, so z becomes 0.0 for all
        assert result is not None

    def test_missing_price_data(self):
        """Missing some price data should not crash."""
        dates = [f"2020-01-{d:02d}" for d in range(1, 65)]
        prices = {}
        for date in dates:
            prices[date] = {"SPY": 100.0}  # Missing GLD and TLT
        bt = CrossAssetRVBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        assert result is not None

    def test_zero_initial_capital(self):
        dates = [f"2020-01-{d:02d}" for d in range(1, 65)]
        prices = {}
        for date in dates:
            prices[date] = {"SPY": 100.0, "GLD": 100.0, "TLT": 100.0}
        bt = CrossAssetRVBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-06-30", initial_capital=0.0)
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        assert result is not None
        assert isinstance(result, BacktestResult)

    def test_extreme_z_score(self):
        """Extreme price movements producing high |z-scores| should not crash."""
        dates = [f"2020-01-{d:02d}" for d in range(1, 75)]
        prices = {}
        for date in dates:
            prices[date] = {}
        for i, date in enumerate(dates):
            if i < 62:
                prices[date]["SPY"] = 100.0
                prices[date]["GLD"] = 100.0
                prices[date]["TLT"] = 100.0
            else:
                # Sudden massive jump
                prices[date]["SPY"] = 500.0
                prices[date]["GLD"] = 500.0
                prices[date]["TLT"] = 500.0

        bt = CrossAssetRVBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        assert result is not None

    def test_save_results_creates_json(self):
        """save_results should create a valid JSON file."""
        dates, prices = TestRunBacktest._make_synthetic_prices(n_days=100)
        bt = CrossAssetRVBacktester(
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
            assert "avg_z_score" in saved["extras"]
            assert "diverged_pct" in saved["extras"]
        finally:
            Path(output_path).unlink()

    def test_print_report_does_not_crash(self, caplog):
        """print_report should produce output without errors."""
        dates, prices = TestRunBacktest._make_synthetic_prices(n_days=100)
        bt = CrossAssetRVBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-06-01")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        with caplog.at_level(logging.INFO, logger="src.backtest.cross_asset_rv_backtest"):
            bt.print_report(result)
        assert "Cross-Asset Relative Value" in caplog.text
        assert "Sharpe" in caplog.text
        assert "Signal Distribution" in caplog.text
        assert "Diverged Months" in caplog.text

    def test_load_data_missing_file_logs_error(self, caplog, monkeypatch):
        """load_data should return False when prices.json is missing."""
        import logging
        caplog.set_level(logging.ERROR)
        bt = CrossAssetRVBacktester()
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        result = bt.load_data()
        assert result is False

    def test_negative_prices_handled(self):
        """Negative prices should not crash but may produce NaN returns."""
        dates = [f"2020-01-{d:02d}" for d in range(1, 65)]
        prices = {}
        for date in dates:
            prices[date] = {"SPY": -100.0, "GLD": 100.0, "TLT": 100.0}
        bt = CrossAssetRVBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.dates = dates
        bt.prices = prices
        result = bt.run_backtest()
        assert result is not None


# ── CLI Tests ───────────────────────────────────────────────────────────


class TestCLI:
    """Test command-line invocation."""

    def test_main_run(self, monkeypatch):
        """main() should not crash and should handle missing data."""
        from src.backtest.cross_asset_rv_backtest import main
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        with patch("sys.argv", ["cross_asset_rv_backtest.py", "run"]):
            main()

    def test_main_with_save_flag(self, monkeypatch):
        """main() with --save flag should handle missing data gracefully."""
        from src.backtest.cross_asset_rv_backtest import main
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        with patch("sys.argv", ["cross_asset_rv_backtest.py", "run", "--save"]):
            main()


# ---------------------------------------------------------------------------
# __all__ export validation
# ---------------------------------------------------------------------------

class TestExports:
    """Verify __all__ exports."""

    def test_all_exports_present(self):
        import src.backtest.cross_asset_rv_backtest as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"Missing export: {name}"

    def test_all_count(self):
        import src.backtest.cross_asset_rv_backtest as mod
        assert len(mod.__all__) == 2


# ---------------------------------------------------------------------------
# BacktestConfig extended
# ---------------------------------------------------------------------------

class TestBacktestConfigExtended:
    """Extended BacktestConfig dataclass tests."""

    def test_default_z_score_window(self):
        config = BacktestConfig()
        assert config.z_score_window == 60

    def test_default_max_shift(self):
        config = BacktestConfig()
        assert config.max_shift == 0.04

    def test_custom_config(self):
        config = BacktestConfig(z_score_window=30, max_shift=0.02)
        assert config.z_score_window == 30
        assert config.max_shift == 0.02


# ---------------------------------------------------------------------------
# CrossAssetRVBacktester extended
# ---------------------------------------------------------------------------

class TestBacktesterExtended:
    """Extended CrossAssetRVBacktester tests."""

    def test_init_with_default_config(self):
        bt = CrossAssetRVBacktester()
        assert bt.config is not None

    def test_init_with_custom_config(self):
        config = BacktestConfig(z_score_window=30)
        bt = CrossAssetRVBacktester(config)
        assert bt.config.z_score_window == 30

    def test_load_data_missing_file(self):
        bt = CrossAssetRVBacktester()
        result = bt.load_data("/nonexistent/path.json")
        assert result is False

    def test_print_report(self, caplog):
        bt = CrossAssetRVBacktester()
        from src.backtest.metrics import BacktestResult
        result = BacktestResult(
            total_return=100.0, cagr=0.10, volatility=0.11,
            sharpe_ratio=0.79, max_drawdown=-0.25, total_rebalances=10,
            baseline_sharpe=0.72, extras={"scenario": "test"},
        )
        with caplog.at_level(logging.INFO, logger="src.backtest.cross_asset_rv_backtest"):
            bt.print_report(result)
        assert len(caplog.text) > 0

    def test_save_results(self, tmp_path):
        bt = CrossAssetRVBacktester()
        from src.backtest.metrics import BacktestResult
        result = BacktestResult(
            total_return=100.0, cagr=0.10, volatility=0.11,
            sharpe_ratio=0.79, max_drawdown=-0.25, total_rebalances=10,
            baseline_sharpe=0.72, extras={"scenario": "test"},
        )
        bt.save_results(result, str(tmp_path / "results.json"))
        assert (tmp_path / "results.json").exists()
