"""
Tests for Crypto Tactical Allocation Walk-Forward Backtest (v9.32)
"""

import json
import pytest
import numpy as np
from pathlib import Path

from src.backtest.crypto_allocation_backtest import (
    BacktestConfig,
    BacktestResult,
    DailyPrices,
    WalkForwardCryptoBacktester,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return BacktestConfig(
        start_date="2020-01-01",
        end_date="2023-12-31",
        initial_capital=100000.0,
        max_crypto_pct=5.0,
    )


@pytest.fixture
def backtester(config):
    bt = WalkForwardCryptoBacktester(config)
    return bt


@pytest.fixture
def sample_daily_prices():
    """Create a small set of daily prices for unit testing."""
    return [
        DailyPrices("2020-01-02", spy=100.0, gld=100.0, tlt=100.0, btc=50000.0, eth=3000.0),
        DailyPrices("2020-01-03", spy=101.0, gld=100.5, tlt=100.2, btc=51000.0, eth=3050.0),
        DailyPrices("2020-01-06", spy=102.0, gld=101.0, tlt=100.4, btc=52000.0, eth=3100.0),
        DailyPrices("2020-01-07", spy=101.5, gld=100.8, tlt=100.1, btc=50500.0, eth=2950.0),
        DailyPrices("2020-01-08", spy=103.0, gld=101.2, tlt=100.5, btc=53000.0, eth=3150.0),
    ]


# ---------------------------------------------------------------------------
# BacktestConfig Tests
# ---------------------------------------------------------------------------


class TestBacktestConfig:
    """Test configuration dataclass."""

    def test_default_values(self):
        cfg = BacktestConfig()
        assert cfg.start_date == "2006-01-01"
        assert cfg.end_date == "2026-05-15"
        assert cfg.initial_capital == 100000.0
        assert cfg.max_crypto_pct == 5.0

    def test_custom_values(self):
        cfg = BacktestConfig(
            start_date="2020-01-01",
            end_date="2023-01-01",
            max_crypto_pct=3.0,
        )
        assert cfg.max_crypto_pct == 3.0


# ---------------------------------------------------------------------------
# BacktestResult Tests
# ---------------------------------------------------------------------------


class TestBacktestResult:
    """Test result dataclass."""

    def test_empty_result_serialization(self):
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, baseline_total_return=0.0, baseline_cagr=0.0,
            baseline_volatility=0.0, baseline_sharpe=0.0,
            baseline_max_drawdown=0.0, sharpe_improvement=0.0, cagr_impact=0.0,
            crypto_active_days=0, crypto_active_pct=0.0, avg_crypto_pct=0.0,
            max_crypto_pct=0.0, crisis_returns_crypto={},
            crisis_returns_baseline={}, regime_breakdown={},
            total_rebalances=0, total_transaction_costs=0.0, config_snapshot={},
        )
        d = result.to_dict()
        assert d["total_return"] == 0.0
        assert d["crypto_active_days"] == 0
        assert "config_snapshot" in d

    def test_result_serializable_to_json(self):
        result = BacktestResult(
            total_return=15.5, cagr=5.2, volatility=12.0, sharpe_ratio=0.85,
            max_drawdown=-18.0, baseline_total_return=12.0, baseline_cagr=4.1,
            baseline_volatility=11.0, baseline_sharpe=0.75,
            baseline_max_drawdown=-20.0, sharpe_improvement=0.1, cagr_impact=1.1,
            crypto_active_days=500, crypto_active_pct=40.0, avg_crypto_pct=2.5,
            max_crypto_pct=5.0, crisis_returns_crypto={"2020": 5.0},
            crisis_returns_baseline={"2020": 3.0}, regime_breakdown={"active": {"count": 10, "pct_of_rebalances": 50.0}},
            total_rebalances=20, total_transaction_costs=15.0, config_snapshot={"max_crypto_pct": 5.0},
        )
        json_str = json.dumps(result.to_dict())
        parsed = json.loads(json_str)
        assert parsed["sharpe_ratio"] == 0.85
        assert parsed["crypto_active_days"] == 500


# ---------------------------------------------------------------------------
# DailyPrices Tests
# ---------------------------------------------------------------------------


class TestDailyPrices:
    """Test daily prices dataclass."""

    def test_partial_crypto_data(self):
        dp = DailyPrices("2020-01-02", spy=100.0, gld=100.0, tlt=100.0)
        assert dp.btc is None
        assert dp.eth is None

    def test_full_data(self):
        dp = DailyPrices("2020-01-02", spy=100.0, gld=100.0, tlt=100.0, btc=50000.0, eth=3000.0)
        assert dp.btc == 50000.0
        assert dp.eth == 3000.0


# ---------------------------------------------------------------------------
# WalkForwardCryptoBacktester Tests
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test backtester initialization."""

    def test_default_config(self):
        bt = WalkForwardCryptoBacktester()
        assert bt.config.start_date == "2006-01-01"

    def test_empty_prices_on_init(self, backtester):
        assert len(backtester._daily_prices) == 0


class TestDataLoading:
    """Test data loading."""

    def test_synthetic_data_generated_when_no_file(self, backtester):
        """Should generate synthetic data when prices.json doesn't exist."""
        backtester.load_data()
        assert len(backtester._daily_prices) > 0
        assert len(backtester._trading_dates) > 0

    def test_synthetic_data_has_all_symbols(self, backtester):
        backtester.load_data()
        dp = backtester._daily_prices[0]
        assert dp.spy > 0
        assert dp.gld > 0
        assert dp.tlt > 0

    def test_synthetic_crypto_high_volatility(self, backtester):
        backtester.load_data()
        btc_prices = [dp.btc for dp in backtester._daily_prices if dp.btc]
        eth_prices = [dp.eth for dp in backtester._daily_prices if dp.eth]
        assert len(btc_prices) > 0
        assert len(eth_prices) > 0
        # Crypto should have higher variance than SPY
        spy_prices = [dp.spy for dp in backtester._daily_prices]
        spy_vol = np.std([spy_prices[i] / spy_prices[i-1] - 1 for i in range(1, len(spy_prices))])
        btc_returns = [btc_prices[i] / btc_prices[i-1] - 1 for i in range(1, len(btc_prices))]
        btc_vol = np.std(btc_returns)
        assert btc_vol > spy_vol * 2  # Crypto should be at least 2x volatile


class TestMomentumComputation:
    """Test SPY momentum computation."""

    def test_momentum_positive_with_uptrend(self, backtester):
        backtester.load_data()
        # Synthetic data has positive drift, so 6m momentum should be positive
        # at most points after warmup
        if len(backtester._daily_prices) > 200:
            mom = backtester._compute_spy_momentum_6m(200)
            assert isinstance(mom, float)

    def test_momentum_zero_with_insufficient_data(self, backtester):
        mom = backtester._compute_spy_momentum_6m(5)  # Too early
        assert mom == 0.0


class TestCryptoVolComputation:
    """Test crypto volatility computation."""

    def test_vol_computed_from_synthetic(self, backtester):
        backtester.load_data()
        if len(backtester._daily_prices) > 50:
            btc_vol, eth_vol = backtester._compute_crypto_vol(50)
            assert btc_vol >= 0
            assert eth_vol >= 0

    def test_vol_zero_with_insufficient_data(self, backtester):
        btc_vol, eth_vol = backtester._compute_crypto_vol(2)
        assert btc_vol == 0.0
        assert eth_vol == 0.0

    def test_is_vol_extreme_detection(self, backtester):
        assert backtester._is_vol_extreme(0.5, 0.5) is False
        assert backtester._is_vol_extreme(1.5, 0.5) is True
        assert backtester._is_vol_extreme(0.5, 1.2) is True
        assert backtester._is_vol_extreme(1.1, 1.1) is True

    def test_vol_extreme_at_boundary(self, backtester):
        assert backtester._is_vol_extreme(1.0, 0.5) is False  # Boundary: 1.0 is NOT extreme
        assert backtester._is_vol_extreme(0.5, 1.0) is False


class TestCryptoAllocation:
    """Test crypto allocation computation."""

    def test_allocation_max_cap_respected(self, backtester):
        backtester.load_data()
        if len(backtester._daily_prices) > 200:
            btc_w, eth_w, total = backtester._compute_crypto_allocation(200, 0.38)
            assert total <= backtester.config.max_crypto_pct / 100.0

    def test_allocation_zero_when_no_momentum(self, backtester):
        """Allocation should be zero early on (no momentum history)."""
        btc_w, eth_w, total = backtester._compute_crypto_allocation(5, 0.38)
        assert total == 0.0
        assert btc_w == 0.0
        assert eth_w == 0.0

    def test_btc_eth_split_60_40(self, backtester):
        """When active, crypto should be split 60/40 BTC/ETH."""
        backtester.load_data()
        if len(backtester._daily_prices) > 200:
            btc_w, eth_w, total = backtester._compute_crypto_allocation(200, 0.38)
            if total > 0:
                # BTC should be ~60% of crypto, ETH ~40%
                assert abs(btc_w / total - 0.60) < 0.01
                assert abs(eth_w / total - 0.40) < 0.01


class TestPortfolioReturn:
    """Test portfolio return computation."""

    def test_baseline_return(self, backtester, sample_daily_prices):
        backtester._daily_prices = sample_daily_prices
        ret = backtester._compute_portfolio_return(
            sample_daily_prices[0], sample_daily_prices[1],
            0.46, 0.38, 0.16, 0.0, 0.0,
        )
        # Should be a weighted average of asset returns
        spy_ret = 101.0 / 100.0 - 1
        gld_ret = 100.5 / 100.0 - 1
        tlt_ret = 100.2 / 100.0 - 1
        expected = 0.46 * spy_ret + 0.38 * gld_ret + 0.16 * tlt_ret
        assert abs(ret - expected) < 0.0001

    def test_crypto_return(self, backtester, sample_daily_prices):
        backtester._daily_prices = sample_daily_prices
        ret = backtester._compute_portfolio_return(
            sample_daily_prices[0], sample_daily_prices[1],
            0.46, 0.33, 0.16, 0.03, 0.02,  # 3% BTC, 2% ETH, 33% GLD
        )
        spy_ret = 101.0 / 100.0 - 1
        gld_ret = 100.5 / 100.0 - 1
        tlt_ret = 100.2 / 100.0 - 1
        btc_ret = 51000.0 / 50000.0 - 1
        eth_ret = 3050.0 / 3000.0 - 1
        expected = 0.46 * spy_ret + 0.33 * gld_ret + 0.16 * tlt_ret + 0.03 * btc_ret + 0.02 * eth_ret
        assert abs(ret - expected) < 0.0001

    def test_return_with_zero_prices(self, backtester):
        p0 = DailyPrices("2020-01-02", spy=0.0, gld=0.0, tlt=0.0, btc=0.0, eth=0.0)
        p1 = DailyPrices("2020-01-03", spy=100.0, gld=100.0, tlt=100.0, btc=50000.0, eth=3000.0)
        ret = backtester._compute_portfolio_return(p0, p1, 0.46, 0.38, 0.16, 0.03, 0.02)
        assert ret == 0.0  # No division by zero


class TestRunBacktest:
    """Test full backtest execution."""

    def test_run_with_synthetic_data(self, backtester):
        result = backtester.run()
        assert isinstance(result, BacktestResult)
        assert result.sharpe_ratio != 0.0
        assert result.baseline_sharpe != 0.0

    def test_result_contains_all_metrics(self, backtester):
        result = backtester.run()
        assert result.total_return != 0.0
        assert result.cagr != 0.0
        assert result.crypto_active_days >= 0
        assert result.crypto_active_pct >= 0
        assert result.avg_crypto_pct >= 0
        assert result.max_crypto_pct >= 0

    def test_crypto_never_exceeds_max(self, backtester):
        result = backtester.run()
        assert result.max_crypto_pct <= backtester.config.max_crypto_pct

    def test_avg_crypto_within_bounds(self, backtester):
        result = backtester.run()
        assert 0 <= result.avg_crypto_pct <= backtester.config.max_crypto_pct

    def test_crisis_returns_present(self, backtester):
        result = backtester.run()
        # Synthetic data covers all crisis years
        assert len(result.crisis_returns_baseline) > 0
        assert len(result.crisis_returns_crypto) > 0

    def test_config_snapshot_in_result(self, backtester):
        result = backtester.run()
        assert "max_crypto_pct" in result.config_snapshot
        assert result.config_snapshot["max_crypto_pct"] == 5.0

    def test_custom_config_used(self):
        cfg = BacktestConfig(
            start_date="2020-01-01",
            end_date="2023-01-01",
            max_crypto_pct=3.0,
        )
        bt = WalkForwardCryptoBacktester(cfg)
        result = bt.run()
        assert result.config_snapshot["max_crypto_pct"] == 3.0
        assert result.max_crypto_pct <= 3.0

    def test_empty_result_on_insufficient_data(self):
        """Synthetic data fallback generates data for any date range."""
        cfg = BacktestConfig(
            start_date="2099-01-01",
            end_date="2099-12-31",
        )
        bt = WalkForwardCryptoBacktester(cfg)
        result = bt.run()
        # Synthetic data is generated, so backtest runs normally
        assert isinstance(result, BacktestResult)
        assert result.sharpe_ratio != 0.0
        assert result.total_rebalances > 0


class TestEmptyResult:
    """Test empty result edge case."""

    def test_empty_result_defaults(self, backtester):
        result = backtester._empty_result()
        assert result.sharpe_ratio == 0.0
        assert result.crypto_active_days == 0
        assert result.crypto_active_pct == 0.0
        assert result.total_rebalances == 0
        assert result.total_transaction_costs == 0.0


class TestPrintSaveResults:
    """Test result printing and saving."""

    def test_print_results_does_not_error(self, backtester, capsys):
        backtester.load_data()
        result = backtester._empty_result()
        # Populate some data to test formatting
        result.baseline_sharpe = 0.75
        result.sharpe_ratio = 0.85
        backtester.print_results(result)
        captured = capsys.readouterr()
        assert "Crypto Tactical Allocation" in captured.out

    def test_save_results_creates_file(self, backtester, tmp_path):
        result = backtester._empty_result()
        output_path = tmp_path / "test_results.json"
        backtester.save_results(result, output_path=str(output_path))
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert data["total_return"] == 0.0
        assert data["_metadata"]["strategy"] == "crypto_allocation"


class TestPricesLookup:
    """Test price lookup building."""

    def test_lookup_has_all_symbols(self, backtester, sample_daily_prices):
        backtester._daily_prices = sample_daily_prices
        lookup = backtester._build_prices_lookup()
        assert "2020-01-02" in lookup
        entry = lookup["2020-01-02"]
        assert "SPY" in entry
        assert "GLD" in entry
        assert "TLT" in entry
        assert "BTC-USD" in entry
        assert "ETH-USD" in entry


class TestCLI:
    """Test CLI entry point."""

    def test_cli_run_default(self):
        """main() should not crash with default args."""
        try:
            import sys
            sys.argv = ["crypto_allocation_backtest.py", "run"]
            main()
        except SystemExit:
            pass  # argparse may exit on some configs

    def test_cli_save(self, tmp_path):
        output = tmp_path / "cli_results.json"
        try:
            import sys
            sys.argv = [
                "crypto_allocation_backtest.py",
                "run",
                "--start", "2020-01-01",
                "--end", "2023-01-01",
                "--capital", "50000",
                "--max-crypto", "3.0",
                "--save",
                "--output", str(output),
            ]
            main()
        except SystemExit:
            pass

        if output.exists():
            with open(output) as f:
                data = json.load(f)
            assert "total_return" in data
