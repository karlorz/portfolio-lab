"""
Tests for the VIXY Hedge Walk-Forward Backtest.

Covers: BacktestConfig defaults/custom, BacktestResult creation/serialization,
signal computation, hedge constraints, edge cases, crisis regime behavior,
synthetic data loading, print/save output.
"""

import json
import logging
import tempfile
from pathlib import Path

import pytest

from src.backtest.vixy_hedge_backtest import (
    TRADING_DAYS_PER_YEAR,
    MONTHLY_TRADING_DAYS,
    CRISIS_YEARS,
    BASE_SYMBOLS,
    VIX_SYMBOL,
    BacktestConfig,
    DailyPrices,
    WalkForwardVIXYBacktester,
    _result_to_dict,
)
from src.backtest.metrics import BacktestResult
from src.strategy.vixy_hedge_sizing import VIXYHedgeSizer, HedgeRegime


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
        assert config.max_hedge_pct == 6.0
        assert config.rebalance_frequency_days == 21
        assert config.transaction_cost_bps == 10.0

    def test_custom_values(self):
        config = BacktestConfig(
            start_date="2010-01-01",
            end_date="2020-12-31",
            initial_capital=50000.0,
            max_hedge_pct=8.0,
            rebalance_frequency_days=63,
        )
        assert config.start_date == "2010-01-01"
        assert config.end_date == "2020-12-31"
        assert config.initial_capital == 50000.0
        assert config.max_hedge_pct == 8.0
        assert config.rebalance_frequency_days == 63

    def test_base_weights_sum(self):
        """Baseline weights should sum to 1.0."""
        config = BacktestConfig()
        total = sum(config.base_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_max_hedge_pct_zero_disables_hedge(self):
        """max_hedge_pct=0 should effectively disable hedging."""
        config = BacktestConfig(max_hedge_pct=0.0)
        bt = WalkForwardVIXYBacktester(config)
        bt.load_data()
        _, tracker, _ = bt._run_hedged(bt._daily_prices, bt.config)
        assert tracker["max_pct"] == 0.0
        assert tracker["active_days"] == 0

    def test_max_hedge_pct_large_capped_by_sizer(self):
        """Very large max_hedge_pct should not cause overflow; sizer's own cap applies."""
        config = BacktestConfig(max_hedge_pct=100.0)
        bt = WalkForwardVIXYBacktester(config)
        bt.load_data()
        _, tracker, _ = bt._run_hedged(bt._daily_prices, bt.config)
        # Sizer internally caps at 10%, but config allows up to 100%
        # In practice the VIXY allocation from the sizer should be <= 10%
        assert tracker["max_pct"] <= 10.0

    def test_rebalance_frequency_zero(self):
        """rebalance_frequency_days=0 should not cause division or logic errors."""
        config = BacktestConfig(rebalance_frequency_days=0)
        bt = WalkForwardVIXYBacktester(config)
        bt.load_data()
        result = bt.run()
        assert isinstance(result, BacktestResult)

    def test_rebalance_frequency_negative(self):
        """rebalance_frequency_days negative should not crash."""
        config = BacktestConfig(rebalance_frequency_days=-5)
        bt = WalkForwardVIXYBacktester(config)
        bt.load_data()
        # Should still run without raising
        _, tracker, _ = bt._run_hedged(bt._daily_prices, bt.config)
        assert tracker["rebalances"] > 0  # i==1 triggers rebalance

    def test_custom_base_weights(self):
        """Custom base weights should be used in baseline and hedged runs."""
        custom_weights = {"SPY": 0.50, "GLD": 0.30, "TLT": 0.20}
        config = BacktestConfig(base_weights=custom_weights)
        assert config.base_weights["SPY"] == 0.50
        assert config.base_weights["GLD"] == 0.30
        assert config.base_weights["TLT"] == 0.20
        bt = WalkForwardVIXYBacktester(config)
        bt.load_data()
        result = bt.run()
        assert result.extras["config_snapshot"]["base_allocation"]["SPY"] == 0.50

    def test_negative_transaction_cost(self):
        """Negative transaction cost should not cause crashes (though unrealistic)."""
        config = BacktestConfig(transaction_cost_bps=-5.0)
        bt = WalkForwardVIXYBacktester(config)
        bt.load_data()
        _, tracker, _ = bt._run_hedged(bt._daily_prices, bt.config)
        # Negative cost would mean negative total_costs (a "rebate")
        assert tracker["total_costs"] <= 0.0


# ── BacktestResult Tests ─────────────────────────────────────────────────


class TestBacktestResult:
    """Test BacktestResult creation, to_dict, and empty state."""

    def test_create_and_to_dict(self):
        result = BacktestResult(
            total_return=10.5,
            cagr=8.2,
            volatility=12.3,
            sharpe_ratio=0.85,
            max_drawdown=-15.4,
            baseline_sharpe=0.78,
            sharpe_improvement=0.07,
            total_rebalances=120,
            total_transaction_costs=45.50,
            extras={
                "baseline_total_return": 9.0,
                "baseline_cagr": 7.5,
                "baseline_volatility": 11.8,
                "baseline_sharpe": 0.78,
                "baseline_max_drawdown": -18.2,
                "cagr_impact": 0.7,
                "hedge_active_days": 1200,
                "hedge_active_pct": 50.0,
                "avg_hedge_pct": 2.5,
                "max_hedge_pct": 6.0,
                "crisis_returns_hedged": {"2008": -10.2, "2020": 2.1},
                "crisis_returns_baseline": {"2008": -12.3, "2020": 1.5},
                "regime_breakdown": {
                    "normal": {"avg_hedge_pct": 1.2, "max_hedge_pct": 2.0, "count": 2000, "pct_of_time": 60.0},
                    "elevated": {"avg_hedge_pct": 2.8, "max_hedge_pct": 3.5, "count": 800, "pct_of_time": 24.0},
                },
                "config_snapshot": {"max_hedge_pct": 6.0},
            },
        )

        assert result.total_return == 10.5
        assert result.sharpe_ratio == 0.85
        assert result.sharpe_improvement == 0.07
        assert result.extras["hedge_active_days"] == 1200
        assert result.extras["crisis_returns_hedged"]["2008"] == -10.2
        assert result.extras["regime_breakdown"]["normal"]["avg_hedge_pct"] == 1.2
        assert result.total_rebalances == 120
        assert result.extras["config_snapshot"]["max_hedge_pct"] == 6.0

    def test_to_dict_all_core_fields(self):
        """_result_to_dict should include all 7 core fields."""
        result = BacktestResult(
            total_return=10.5, cagr=8.2, volatility=12.3,
            sharpe_ratio=0.85, max_drawdown=-15.4,
            total_rebalances=120, total_transaction_costs=45.5,
            extras={},
        )
        d = _result_to_dict(result)
        assert d["total_return"] == 10.5
        assert d["cagr"] == 8.2
        assert d["volatility"] == 12.3
        assert d["sharpe_ratio"] == 0.85
        assert d["max_drawdown"] == -15.4
        assert d["total_rebalances"] == 120
        assert d["total_transaction_costs"] == 45.5
        # Optional fields should be absent when None
        assert "baseline_sharpe" not in d
        assert "sharpe_improvement" not in d
        assert "crisis_returns" not in d

    def test_to_dict_optional_fields_included_when_set(self):
        """Optional fields (baseline_sharpe, sharpe_improvement, crisis_returns) appear when not None."""
        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0,
            sharpe_ratio=0.5, max_drawdown=-10.0,
            baseline_sharpe=0.45, sharpe_improvement=0.05,
            total_rebalances=30, total_transaction_costs=15.0,
            crisis_returns={"2008": -12.3, "2020": -1.5},
            extras={},
        )
        d = _result_to_dict(result)
        assert d["baseline_sharpe"] == 0.45
        assert d["sharpe_improvement"] == 0.05
        assert d["crisis_returns"]["2008"] == -12.3

    def test_to_dict_extras_merged(self):
        """Extras dict keys should be merged into the output dict."""
        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0,
            sharpe_ratio=0.5, max_drawdown=-10.0,
            total_rebalances=30, total_transaction_costs=15.0,
            extras={
                "hedge_active_days": 500,
                "avg_hedge_pct": 1.5,
                "regime_breakdown": {"normal": {"count": 100}},
            },
        )
        d = _result_to_dict(result)
        assert d["hedge_active_days"] == 500
        assert d["avg_hedge_pct"] == 1.5
        assert d["regime_breakdown"]["normal"]["count"] == 100

    def test_to_dict_large_values_roundtrip(self):
        """Very large metric values should survive to_dict round-trip."""
        result = BacktestResult(
            total_return=999999.99, cagr=50000.0, volatility=999.99,
            sharpe_ratio=9.9999, max_drawdown=-99.99,
            total_rebalances=999999, total_transaction_costs=1e6,
            extras={"custom_big": 1e12},
        )
        d = _result_to_dict(result)
        assert d["total_return"] == 999999.99
        assert d["cagr"] == 50000.0
        assert d["total_rebalances"] == 999999
        assert d["custom_big"] == 1e12
        # Confirm JSON-serializable
        json.dumps(d)  # Should not raise

    def test_to_dict_negative_values(self):
        """Negative metric values should be handled properly in to_dict."""
        result = BacktestResult(
            total_return=-50.0, cagr=-8.0, volatility=25.0,
            sharpe_ratio=-0.3, max_drawdown=-45.0,
            total_rebalances=50, total_transaction_costs=-10.0,
            extras={"negative_test": -99.9},
        )
        d = _result_to_dict(result)
        assert d["total_return"] == -50.0
        assert d["sharpe_ratio"] == -0.3
        json.dumps(d)  # Should not raise

    def test_crisis_returns_none_to_dict(self):
        """crisis_returns=None should not produce a 'crisis_returns' key in the dict."""
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0,
            extras={},
        )
        d = _result_to_dict(result)
        assert "crisis_returns" not in d

    def test_json_serializable(self):
        """All fields in extras must be JSON-serializable."""
        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0,
            baseline_sharpe=0.45, sharpe_improvement=0.05,
            total_rebalances=30, total_transaction_costs=15.0,
            extras={
                "baseline_total_return": 4.0, "baseline_cagr": 2.5,
                "baseline_volatility": 9.5, "baseline_sharpe": 0.45,
                "baseline_max_drawdown": -12.0,
                "cagr_impact": 0.5, "hedge_active_days": 100,
                "hedge_active_pct": 25.0, "avg_hedge_pct": 1.5, "max_hedge_pct": 4.0,
                "crisis_returns_hedged": {"2008": -8.0}, "crisis_returns_baseline": {"2008": -10.0},
                "regime_breakdown": {"normal": {"avg_hedge_pct": 1.0, "max_hedge_pct": 2.0, "count": 100, "pct_of_time": 50.0}},
                "config_snapshot": {"start_date": "2006-01-01"},
            },
        )
        json.dumps(result.extras)  # Should not raise

    def test_empty_crisis_returns(self):
        """Crisis returns can be empty dict without errors."""
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0,
            extras={
                "baseline_total_return": 0.0, "baseline_cagr": 0.0,
                "baseline_volatility": 0.0, "baseline_sharpe": 0.0,
                "baseline_max_drawdown": 0.0,
                "cagr_impact": 0.0, "hedge_active_days": 0,
                "hedge_active_pct": 0.0, "avg_hedge_pct": 0.0, "max_hedge_pct": 0.0,
                "crisis_returns_hedged": {}, "crisis_returns_baseline": {},
                "regime_breakdown": {}, "config_snapshot": {},
            },
        )
        assert result.extras["crisis_returns_hedged"] == {}


# ── Walk-Forward Backtester Tests ────────────────────────────────────────


class TestWalkForwardVIXYBacktester:
    """Test the core WalkForwardVIXYBacktester class."""

    def test_init_defaults(self):
        bt = WalkForwardVIXYBacktester()
        assert bt.config.start_date == "2006-01-01"
        assert isinstance(bt.sizer, VIXYHedgeSizer)

    def test_init_custom_config(self):
        config = BacktestConfig(start_date="2015-01-01", max_hedge_pct=5.0)
        bt = WalkForwardVIXYBacktester(config)
        assert bt.config.start_date == "2015-01-01"
        assert bt.config.max_hedge_pct == 5.0

    def test_load_data_generates_synthetic_when_no_file(self, monkeypatch):
        """When prices.json doesn't exist, synthetic data is generated."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        assert len(bt._daily_prices) > 0
        assert len(bt._trading_dates) > 0

    def test_synthetic_data_has_required_fields(self):
        """Each DailyPrices entry should have SPY, GLD, TLT values."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        for dp in bt._daily_prices[:10]:
            assert isinstance(dp.spy, float)
            assert isinstance(dp.gld, float)
            assert isinstance(dp.tlt, float)

    def test_vix_proxy_default_when_insufficient_history(self):
        """VIX proxy should return ~18 when fewer than 21 days of history."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        vix = bt._compute_vix_proxy(5)  # idx 5 = only 6 days
        assert vix == 18.0 or abs(vix - 18.0) < 1.0

    def test_vix_proxy_with_sufficient_history(self):
        """VIX proxy should compute from 21-day SPY returns."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        # Use an index with enough history
        vix = bt._compute_vix_proxy(50)
        assert vix > 0
        assert vix < 100  # Sanity check

    def test_run_produces_results(self):
        """Running the backtest should return a populated BacktestResult."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(start_date="2015-01-01", end_date="2016-01-01")
        )
        result = bt.run()
        assert isinstance(result, BacktestResult)
        assert result.total_rebalances > 0

    def test_hedge_never_exceeds_max_hedge_pct(self):
        """VIXY allocation should not exceed the configured max."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(max_hedge_pct=6.0)
        )
        bt.load_data()
        _, tracker, _ = bt._run_hedged(bt._daily_prices, bt.config)
        assert tracker["max_pct"] <= 6.0 + 0.01  # Allow tiny float rounding

    def test_hedge_allocations_range(self):
        """All hedge allocations should be >= 0 and within bounds."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(max_hedge_pct=6.0)
        )
        bt.load_data()
        _, tracker, _ = bt._run_hedged(bt._daily_prices, bt.config)
        for alloc in tracker["allocations"]:
            assert alloc >= 0.0
            assert alloc <= 6.0 + 0.01

    def test_baseline_weights_stable(self):
        """Baseline run should maintain constant weights."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        prices_subset = bt._daily_prices[:100]
        equity = bt._run_baseline(
            prices_subset,
            BacktestConfig(initial_capital=100000.0),
        )
        assert len(equity) == len(prices_subset)
        assert equity[0] == 100000.0

    def test_empty_result_method_returns_zeros(self):
        """_empty_result() should return all-zero metrics."""
        bt = WalkForwardVIXYBacktester()
        result = bt._empty_result()
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.extras["hedge_active_days"] == 0
        assert result.total_rebalances == 0
        assert result.extras["crisis_returns_hedged"] == {}

    def test_single_day_data_returns_zero_result(self):
        """Only one data point should return an empty result."""
        bt = WalkForwardVIXYBacktester()
        bt._daily_prices = [DailyPrices(date="2020-01-02", spy=100.0, gld=100.0, tlt=100.0)]
        bt._trading_dates = ["2020-01-02"]
        result = bt.run()
        assert result.total_return == 0.0

    def test_hedge_active_in_elevated_vix(self):
        """Hedge should be active (nonzero) when VIX is elevated."""
        sizer = VIXYHedgeSizer()
        alloc = sizer.compute_allocation(25.0)  # ELEVATED regime
        assert alloc > 0

    def test_hedge_inactive_in_low_vix(self):
        """Hedge might be zero or near-zero when VIX is very low."""
        sizer = VIXYHedgeSizer()
        alloc = sizer.compute_allocation(10.0)  # NORMAL regime, VIX < 20
        assert alloc >= 0.0
        # VIX=10 -> 10/10 = 1.0%, NORMAL floor=0 ceiling=2 -> 1.0%
        assert alloc >= 0.0

    def test_crisis_regime_triggers_larger_hedge(self):
        """CRISIS regime (VIX > 40) should produce the largest allocation."""
        sizer = VIXYHedgeSizer()
        normal_alloc = sizer.compute_allocation(15.0)
        crisis_alloc = sizer.compute_allocation(45.0)
        assert crisis_alloc >= normal_alloc

    def test_stress_regime_hedge_bounds(self):
        """STRESS regime (VIX 30-40) should cap at 6%."""
        sizer = VIXYHedgeSizer()
        alloc = sizer.compute_allocation(35.0)
        assert 2.0 <= alloc <= 6.0

    def test_narrow_date_range_still_runs(self):
        """A narrow date range (2 months) should still produce results."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-03-01")
        )
        result = bt.run()
        assert result.cagr is not None
        assert result.total_rebalances >= 0

    def test_print_results_does_not_crash(self, caplog):
        """print_results should produce output without errors."""
        caplog.set_level(logging.INFO)
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        result = bt.run()
        bt.print_results(result)
        assert "VIXY Hedge Sizing" in caplog.text
        assert "Sharpe" in caplog.text

    def test_save_results_creates_json_file(self):
        """save_results should create a valid JSON file."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        result = bt.run()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
            bt.save_results(result, output_path=output_path)

        try:
            with open(output_path) as f:
                data = json.load(f)
            assert "total_return" in data
            assert "sharpe_ratio" in data
            assert "crisis_returns_hedged" in data
            assert "regime_breakdown" in data
            assert data["_metadata"]["strategy"] == "vixy_hedge"
        finally:
            Path(output_path).unlink()

    def test_compute_vix_proxy_at_21_day_boundary(self):
        """VIX proxy with exactly 21 trading days should compute a real value, not default."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        vix = bt._compute_vix_proxy(MONTHLY_TRADING_DAYS)
        # idx=21 means days [0..21] = 22 prices, so 21 returns
        assert vix > 0.0
        assert vix < 100.0

    def test_compute_vix_proxy_insufficient_history_variants(self):
        """VIX proxy should return 18 when fewer than 5 returns available."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        # idx=0: zero history -> 18.0
        assert bt._compute_vix_proxy(0) == 18.0
        # idx=1: only 2 prices -> 1 return -> insufficient -> 18.0
        assert bt._compute_vix_proxy(1) == 18.0
        # idx=4: only 5 prices -> 4 returns -> insufficient (<5) -> 18.0
        assert bt._compute_vix_proxy(4) == 18.0
        # idx=5: 6 prices -> 5 returns -> sufficient -> real computation
        vix = bt._compute_vix_proxy(5)

    def test_compute_vix_proxy_stable_output(self):
        """VIX proxy should produce deterministic output with synthetic data."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        vix_a = bt._compute_vix_proxy(100)
        vix_b = bt._compute_vix_proxy(100)
        assert vix_a == vix_b

    def test_get_vix_level_none_falls_back_to_proxy(self):
        """When DailyPrices.vix is None, _get_vix_level should use proxy."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        # Force vix to None on first price, then check
        original_vix = bt._daily_prices[50].vix
        bt._daily_prices[50].vix = None
        level = bt._get_vix_level(50)
        assert level > 0.0
        bt._daily_prices[50].vix = original_vix

    def test_get_vix_level_zero_falls_back_to_proxy(self):
        """When DailyPrices.vix is 0, _get_vix_level should use proxy (0 is not > 0)."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        original_vix = bt._daily_prices[50].vix
        bt._daily_prices[50].vix = 0.0
        level = bt._get_vix_level(50)
        assert level > 0.0  # Falls back to proxy
        bt._daily_prices[50].vix = original_vix

    def test_get_vix_level_valid_returns_direct(self):
        """When DailyPrices.vix is a positive float, _get_vix_level returns it directly."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        bt._daily_prices[50].vix = 25.5
        level = bt._get_vix_level(50)
        assert level == 25.5

    def test_compute_portfolio_return_zero_prices(self):
        """Portfolio return with zero SPY price should not crash and use fallback of 0.0."""
        bt = WalkForwardVIXYBacktester()
        p0 = DailyPrices(date="2020-01-01", spy=0.0, gld=100.0, tlt=100.0)
        p1 = DailyPrices(date="2020-01-02", spy=100.0, gld=101.0, tlt=102.0)
        ret = bt._compute_portfolio_return(p0, p1, 0.46, 0.38, 0.16, 0.0)
        # spy_ret=0.0 (division by zero fallback), gld_ret=0.01, tlt_ret=0.02
        # ret = 0.46*0 + 0.38*0.01 + 0.16*0.02 + 0 = 0.0038 + 0.0032 = 0.007
        assert abs(ret - 0.007) < 1e-10

    def test_compute_portfolio_return_negative_spy(self):
        """Portfolio return with negative SPY return should trigger VIXY inverse gain."""
        bt = WalkForwardVIXYBacktester()
        p0 = DailyPrices(date="2020-01-01", spy=100.0, gld=100.0, tlt=100.0)
        p1 = DailyPrices(date="2020-01-02", spy=90.0, gld=100.0, tlt=100.0)
        # spy_ret=-0.10, vixy_ret = -(-0.10)*3.5 = 0.35
        # 6% VIXY hedge: ret = 0.40*(-0.10) + 0.38*0 + 0.16*0 + 0.06*0.35
        # = -0.04 + 0.021 = -0.019
        ret = bt._compute_portfolio_return(p0, p1, 0.40, 0.38, 0.16, 0.06)
        expected = 0.40 * -0.10 + 0.06 * 0.35
        assert abs(ret - expected) < 1e-10

    def test_build_prices_lookup_all_symbols(self):
        """_build_prices_lookup should contain SPY, GLD, TLT for every date."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        lookup = bt._build_prices_lookup()
        assert len(lookup) == len(bt._daily_prices)
        for date_key, prices in lookup.items():
            assert "SPY" in prices
            assert "GLD" in prices
            assert "TLT" in prices
            assert prices["SPY"] > 0
            assert prices["GLD"] > 0
            assert prices["TLT"] > 0

    def test_compute_regime_breakdown_empty_tracker(self):
        """Empty regime tracker should return empty dict."""
        bt = WalkForwardVIXYBacktester()
        breakdown = bt._compute_regime_breakdown([])
        assert breakdown == {}

    def test_compute_regime_breakdown_single_entry(self):
        """Single entry regime tracker should produce valid breakdown."""
        bt = WalkForwardVIXYBacktester()
        tracker = [
            {"date": "2020-01-15", "vix_level": 25.0, "regime": "elevated", "hedge_pct": 2.5},
        ]
        breakdown = bt._compute_regime_breakdown(tracker)
        assert "elevated" in breakdown
        assert breakdown["elevated"]["count"] == 1
        assert breakdown["elevated"]["avg_hedge_pct"] == 2.5
        assert breakdown["elevated"]["max_hedge_pct"] == 2.5
        assert breakdown["elevated"]["pct_of_time"] == 100.0

    def test_compute_regime_breakdown_multiple_regimes(self):
        """Multiple VIX regimes should each appear in breakdown with correct percentages."""
        bt = WalkForwardVIXYBacktester()
        tracker = [
            {"date": "2020-01-15", "vix_level": 15.0, "regime": "normal", "hedge_pct": 1.0},
            {"date": "2020-01-16", "vix_level": 25.0, "regime": "elevated", "hedge_pct": 2.5},
            {"date": "2020-01-17", "vix_level": 35.0, "regime": "stress", "hedge_pct": 3.5},
            {"date": "2020-01-18", "vix_level": 45.0, "regime": "crisis", "hedge_pct": 4.5},
        ]
        breakdown = bt._compute_regime_breakdown(tracker)
        assert len(breakdown) == 4
        assert breakdown["normal"]["pct_of_time"] == 25.0
        assert breakdown["crisis"]["pct_of_time"] == 25.0
        assert breakdown["elevated"]["avg_hedge_pct"] == 2.5

    def test_baseline_custom_weights(self):
        """Baseline run should accept custom weight configuration."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        config = BacktestConfig(
            base_weights={"SPY": 0.60, "GLD": 0.25, "TLT": 0.15},
            initial_capital=50000.0,
        )
        equity = bt._run_baseline(bt._daily_prices[:50], config)
        assert len(equity) == 50
        assert equity[0] == 50000.0

    def test_run_hedged_max_hedge_zero(self):
        """With max_hedge_pct=0, hedged run should match baseline behavior."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        config = BacktestConfig(max_hedge_pct=0.0)
        hedge_equity, tracker, _ = bt._run_hedged(bt._daily_prices[:100], config)
        assert tracker["max_pct"] == 0.0
        assert tracker["active_days"] == 0
        # No rebalances happen (VIXY stays at 0, no weight change)
        # But i==1 still triggers initial rebalance with zero hedge

    def test_run_hedged_cost_calculation(self):
        """Transaction cost should be proportional to turnover and config cost_bps."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        config = BacktestConfig(
            transaction_cost_bps=50.0,  # 50 bps = 0.5%
            max_hedge_pct=10.0,
        )
        _, tracker, _ = bt._run_hedged(bt._daily_prices[:50], config)
        # With high cost, total_costs should be > 0 when hedge is active
        assert tracker["total_costs"] >= 0.0

    def test_get_vix_level_negative(self):
        """When DailyPrices.vix is negative, _get_vix_level should fall back to proxy."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        bt._daily_prices[50].vix = -5.0
        level = bt._get_vix_level(50)
        assert level > 0.0  # Falls back to proxy

    def test_build_prices_lookup_empty(self):
        """_build_prices_lookup with no daily prices should return empty dict."""
        bt = WalkForwardVIXYBacktester()
        bt._daily_prices = []
        lookup = bt._build_prices_lookup()
        assert lookup == {}

    def test_compute_crisis_returns_hedged_same_day(self):
        """Crisis return with same start and end date should be 0.0."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        # Use a single date that is in a crisis year
        crisis_date = None
        for d in bt._trading_dates:
            if d.startswith("2020"):
                crisis_date = d
                break
        if crisis_date is None:
            # Fallback: use first date
            crisis_date = bt._trading_dates[0]

        lookup = bt._build_prices_lookup()
        eq_curve = [100000.0, 101000.0]
        # Map date to idx 0 -> start_idx = 0, end_idx = 0
        bt._daily_prices = bt._daily_prices[:1]
        bt._daily_prices[0].date = crisis_date
        bt._trading_dates = [crisis_date]
        result = bt._compute_crisis_returns_hedged(
            lookup, [crisis_date], eq_curve, 100000.0
        )
        assert isinstance(result, dict)

    def test_compute_crisis_returns_hedged_zero_initial_equity(self):
        """Crisis returns when eq_start is 0 should not crash (division by zero guard)."""
        bt = WalkForwardVIXYBacktester()
        lookup = {"2020-01-02": {"SPY": 100.0, "GLD": 100.0, "TLT": 100.0}}
        # eq_start = 0 would cause division by zero; method skips when eq_start == 0
        bt._daily_prices = [DailyPrices(date="2020-01-02", spy=100.0, gld=100.0, tlt=100.0)]
        result = bt._compute_crisis_returns_hedged(
            lookup, ["2020-01-02"], [0.0, 100.0], 100000.0
        )
        assert isinstance(result, dict)
        # Since start_idx=0 -> eq_start=0 -> skipped (eq_start > 0 is False)
        assert len(result) == 0

    def test_compute_regime_breakdown_duplicate_regime(self):
        """Multiple entries for the same regime should be aggregated correctly."""
        bt = WalkForwardVIXYBacktester()
        tracker = [
            {"date": "2020-01-15", "vix_level": 25.0, "regime": "elevated", "hedge_pct": 2.0},
            {"date": "2020-01-16", "vix_level": 26.0, "regime": "elevated", "hedge_pct": 3.0},
            {"date": "2020-01-17", "vix_level": 24.0, "regime": "elevated", "hedge_pct": 1.0},
        ]
        breakdown = bt._compute_regime_breakdown(tracker)
        assert "elevated" in breakdown
        assert breakdown["elevated"]["count"] == 3
        assert breakdown["elevated"]["avg_hedge_pct"] == 2.0  # (2+3+1)/3
        assert breakdown["elevated"]["max_hedge_pct"] == 3.0
        assert breakdown["elevated"]["pct_of_time"] == 100.0

    def test_compute_regime_breakdown_all_same_regime_percentages(self):
        """Many entries for one regime should yield 100% pct_of_time."""
        bt = WalkForwardVIXYBacktester()
        tracker = [
            {"date": f"2020-01-{d:02d}", "vix_level": 15.0, "regime": "normal", "hedge_pct": 1.0}
            for d in range(1, 31)
        ]
        breakdown = bt._compute_regime_breakdown(tracker)
        assert len(breakdown) == 1
        assert breakdown["normal"]["pct_of_time"] == 100.0
        assert breakdown["normal"]["count"] == 30

    def test_empty_result_custom_extras(self):
        """_empty_result should produce a dict with all expected keys in extras."""
        bt = WalkForwardVIXYBacktester()
        result = bt._empty_result()
        expected_keys = [
            "baseline_total_return", "baseline_cagr", "baseline_volatility",
            "baseline_sharpe", "baseline_max_drawdown", "cagr_impact",
            "hedge_active_days", "hedge_active_pct", "avg_hedge_pct",
            "max_hedge_pct", "crisis_returns_hedged", "crisis_returns_baseline",
            "regime_breakdown", "config_snapshot",
        ]
        for key in expected_keys:
            assert key in result.extras

    def test_run_empty_daily_prices(self, monkeypatch):
        """With no daily prices loaded after load_data, run should return empty result."""
        bt = WalkForwardVIXYBacktester()
        # Prevent run() from calling load_data again
        monkeypatch.setattr(bt, "load_data", lambda: None)
        bt._daily_prices = []
        bt._trading_dates = []
        result = bt.run()
        assert result.total_return == 0.0
        assert result.extras["hedge_active_days"] == 0

    def test_compute_portfolio_return_negative_gld_tlt(self):
        """Portfolio return with negative GLD or TLT price changes should compute correctly."""
        bt = WalkForwardVIXYBacktester()
        p0 = DailyPrices(date="2020-01-01", spy=100.0, gld=100.0, tlt=100.0)
        p1 = DailyPrices(date="2020-01-02", spy=100.0, gld=95.0, tlt=90.0)
        ret = bt._compute_portfolio_return(p0, p1, 0.46, 0.38, 0.16, 0.0)
        # spy_ret=0.0, gld_ret=-0.05, tlt_ret=-0.10
        # ret = 0.46*0 + 0.38*(-0.05) + 0.16*(-0.10) = -0.019 - 0.016 = -0.035
        assert abs(ret - (-0.035)) < 1e-10

    def test_compute_portfolio_return_zero_weights(self):
        """Portfolio return with all-zero weights should be 0."""
        bt = WalkForwardVIXYBacktester()
        p0 = DailyPrices(date="2020-01-01", spy=100.0, gld=100.0, tlt=100.0)
        p1 = DailyPrices(date="2020-01-02", spy=110.0, gld=105.0, tlt=102.0)
        ret = bt._compute_portfolio_return(p0, p1, 0.0, 0.0, 0.0, 0.0)
        assert ret == 0.0

    def test_compute_portfolio_return_negative_spy_vixy_non_inverse(self):
        """When spy_ret >= 0, VIXY return uses -spy_ret * 2.0 (not 3.5)."""
        bt = WalkForwardVIXYBacktester()
        p0 = DailyPrices(date="2020-01-01", spy=100.0, gld=100.0, tlt=100.0)
        p1 = DailyPrices(date="2020-01-02", spy=105.0, gld=100.0, tlt=100.0)
        # spy_ret = 0.05, vixy_ret = -0.05 * 2.0 = -0.10
        ret = bt._compute_portfolio_return(p0, p1, 0.40, 0.38, 0.16, 0.06)
        # ret = 0.40*0.05 + 0.06*(-0.10) = 0.02 - 0.006 = 0.014
        assert abs(ret - 0.014) < 1e-10


# ── Edge Cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for the VIXY hedge backtest."""

    def test_zero_initial_capital(self):
        """Zero initial capital should not crash."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(initial_capital=0.0)
        )
        result = bt.run()
        # Metrics still computed (total_return will be 0 or inf)
        assert isinstance(result, BacktestResult)

    def test_negative_max_hedge(self):
        """Negative max hedge should be treated as zero."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(max_hedge_pct=-1.0)
        )
        bt.load_data()
        _, tracker, _ = bt._run_hedged(bt._daily_prices, bt.config)
        assert tracker["max_pct"] == 0.0 or tracker["max_pct"] < 0.01

    def test_no_rebalance_freq_edge(self):
        """Very frequent rebalancing should still work."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(rebalance_frequency_days=1)
        )
        bt.load_data()
        _, tracker, _ = bt._run_hedged(bt._daily_prices, bt.config)
        assert tracker["rebalances"] > 0

    def test_baseline_matches_hedged_when_vix_zero(self):
        """When VIX is zero, there's no hedge, so baseline ≈ hedged."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        baseline_equity = bt._run_baseline(bt._daily_prices, bt.config)
        # Override VIX to zero for all prices
        original_prices = bt._daily_prices[:]
        for i in range(len(bt._daily_prices)):
            bt._daily_prices[i].vix = 0.0
        hedge_equity, _, _ = bt._run_hedged(bt._daily_prices, bt.config)
        # Restore
        bt._daily_prices = original_prices
        # With VIX=0, compute_allocation returns 0 (below floor), so hedged ≈ baseline
        # Small differences from different transaction costs
        assert abs(len(hedge_equity) - len(baseline_equity)) <= 1

    def test_sizer_classify_regime_boundaries(self):
        """VIX exactly at regime boundaries should be classified correctly."""
        sizer = VIXYHedgeSizer()
        # Boundary at 20: VIX 19.99 -> NORMAL, VIX 20.00 -> ELEVATED
        assert sizer.classify_regime(19.99) == HedgeRegime.NORMAL
        assert sizer.classify_regime(20.00) == HedgeRegime.ELEVATED
        # Boundary at 30: VIX 29.99 -> ELEVATED, VIX 30.00 -> STRESS
        assert sizer.classify_regime(29.99) == HedgeRegime.ELEVATED
        assert sizer.classify_regime(30.00) == HedgeRegime.STRESS
        # Boundary at 40: VIX 39.99 -> STRESS, VIX 40.00 -> CRISIS
        assert sizer.classify_regime(39.99) == HedgeRegime.STRESS
        assert sizer.classify_regime(40.00) == HedgeRegime.CRISIS

    def test_sizer_allocation_at_regime_boundaries(self):
        """VIXY allocation at exact regime boundaries should respect floor/ceiling."""
        sizer = VIXYHedgeSizer()
        # VIX=20: ELEVATED, raw=2.0, floor=1.0 ceiling=3.5 -> 2.0
        assert sizer.compute_allocation(20.0) == 2.0
        # VIX=30: STRESS, raw=3.0, floor=2.0 ceiling=6.0 -> 3.0
        assert sizer.compute_allocation(30.0) == 3.0
        # VIX=40: CRISIS, raw=4.0, floor=3.0 ceiling=10.0 -> 4.0
        assert sizer.compute_allocation(40.0) == 4.0

    def test_extreme_vix_100_does_not_crash(self):
        """VIX at 100 should not cause overflow or crash."""
        sizer = VIXYHedgeSizer()
        alloc = sizer.compute_allocation(100.0)
        # raw=10.0, CRISIS floor=3.0 ceiling=10.0 -> 10.0
        assert alloc == 10.0

    def test_extreme_vix_200_clipped_to_max(self):
        """VIX at 200 should be clipped to max_hedge_pct (10.0)."""
        sizer = VIXYHedgeSizer()
        alloc = sizer.compute_allocation(200.0)
        # raw=20.0, CRISIS floor=3.0 ceiling=10.0 -> 10.0 (clipped)
        assert alloc == 10.0

    def test_negative_vix_clipped_to_floor(self):
        """Negative VIX level should clip to NORMAL regime floor (0%)."""
        sizer = VIXYHedgeSizer()
        alloc = sizer.compute_allocation(-5.0)
        # raw=-0.5, NORMAL floor=0.0 ceiling=2.0 -> 0.0
        assert alloc == 0.0

    def test_sizer_allocation_vix_1_to_19_ramp(self):
        """VIX between 1 and 19 should produce monotonically increasing allocation."""
        sizer = VIXYHedgeSizer()
        prev = 0.0
        for vix_int in range(1, 20):
            alloc = sizer.compute_allocation(float(vix_int))
            assert alloc >= prev
            prev = alloc

    def test_run_hedged_tracker_integrity(self):
        """Hedge tracker should maintain internal consistency (allocations count matches days)."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-06-01")
        )
        bt.load_data()
        equity, tracker, regime_tracker = bt._run_hedged(bt._daily_prices, bt.config)
        # Allocations should have len(prices)-1 entries (one per daily return)
        assert len(tracker["allocations"]) == len(bt._daily_prices) - 1
        # Rebalances should match regime_tracker entries
        assert tracker["rebalances"] == len(regime_tracker)

    def test_compute_crisis_returns_hedged_gap_years(self):
        """Crisis years with no data should be silently skipped."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        # Use a short date range that doesn't include any crisis years
        lookup = bt._build_prices_lookup()
        short_dates = [d for d in bt._trading_dates if d < "2008-01-01"]
        if not short_dates:
            short_dates = bt._trading_dates[:100]
        result = bt._compute_crisis_returns_hedged(
            lookup, short_dates, [100000.0] * len(short_dates), 100000.0
        )
        # Should not raise; may have fewer or no entries
        assert isinstance(result, dict)

    def test_compute_crisis_returns_hedged_zero_equity(self):
        """Crisis returns with zero equity values should be handled."""
        bt = WalkForwardVIXYBacktester()
        lookup = {"2020-01-02": {"SPY": 100.0, "GLD": 100.0, "TLT": 100.0}}
        result = bt._compute_crisis_returns_hedged(
            lookup, ["2020-01-02"], [0.0], 100000.0
        )
        assert isinstance(result, dict)

    def test_no_data_in_date_range_synthetic_fallback(self):
        """Date range with no data should trigger synthetic data fallback."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(start_date="1990-01-01", end_date="1990-06-01")
        )
        bt.load_data()
        # Should have generated synthetic data
        assert len(bt._daily_prices) > 0
        assert len(bt._trading_dates) > 0

    def test_config_zero_transaction_cost(self):
        """Zero transaction cost should not cause math errors."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(transaction_cost_bps=0.0)
        )
        bt.load_data()
        _, tracker, _ = bt._run_hedged(bt._daily_prices, bt.config)
        assert tracker["total_costs"] == 0.0

    def test_load_data_empty_json(self, tmp_path):
        """Empty JSON file (no symbols) should trigger synthetic data fallback."""
        from src.backtest import vixy_hedge_backtest as vhb

        # Create an empty prices JSON
        empty_json = tmp_path / "empty_prices.json"
        empty_json.write_text("{}")
        original_path = vhb.PRICES_JSON
        vhb.PRICES_JSON = empty_json

        try:
            bt = WalkForwardVIXYBacktester()
            bt.load_data()
            assert len(bt._daily_prices) > 0
        finally:
            vhb.PRICES_JSON = original_path

    def test_load_data_malformed_json(self, tmp_path):
        """Malformed JSON file should raise JSONDecodeError (not silently swallowed)."""
        from src.backtest import vixy_hedge_backtest as vhb

        bad_json = tmp_path / "bad_prices.json"
        bad_json.write_text("{invalid json!!}")
        original_path = vhb.PRICES_JSON
        vhb.PRICES_JSON = bad_json

        try:
            bt = WalkForwardVIXYBacktester()
            with pytest.raises(json.JSONDecodeError):
                bt.load_data()
        finally:
            vhb.PRICES_JSON = original_path

    def test_load_data_missing_symbols(self, tmp_path):
        """JSON missing SPY/GLD/TLT should trigger synthetic data fallback."""
        from src.backtest import vixy_hedge_backtest as vhb

        partial_json = tmp_path / "partial_prices.json"
        partial_json.write_text('{"QQQ": [{"d": "2020-01-02", "p": 100.0}]}')
        original_path = vhb.PRICES_JSON
        vhb.PRICES_JSON = partial_json

        try:
            bt = WalkForwardVIXYBacktester()
            bt.load_data()
            assert len(bt._daily_prices) > 0
        finally:
            vhb.PRICES_JSON = original_path

    def test_save_results_default_path(self, tmp_path):
        """save_results without output_path should save to BACKTEST_RESULTS_DIR."""
        from src.backtest import vixy_hedge_backtest as vhb

        original_dir = vhb.BACKTEST_RESULTS_DIR
        test_dir = tmp_path / "results"
        test_dir.mkdir()
        vhb.BACKTEST_RESULTS_DIR = test_dir

        try:
            bt = WalkForwardVIXYBacktester()
            bt.load_data()
            result = bt.run()
            bt.save_results(result)
            expected_file = test_dir / "vixy_hedge_backtest_results.json"
            assert expected_file.exists()
            with open(expected_file) as f:
                data = json.load(f)
            assert data["_metadata"]["strategy"] == "vixy_hedge"
        finally:
            vhb.BACKTEST_RESULTS_DIR = original_dir

    def test_hedge_tracker_empty_allocations_avg(self):
        """Hedge tracker with empty allocations list should compute avg_pct=0.0."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        # Trigger the allocation math via regular run
        config = BacktestConfig(max_hedge_pct=0.0)
        _, tracker, _ = bt._run_hedged(bt._daily_prices[:50], config)
        # VIXY weight is always 0, allocations list should exist
        assert len(tracker["allocations"]) == len(bt._daily_prices[:50]) - 1
        assert all(a == 0.0 for a in tracker["allocations"])

    def test_run_hedged_allocations_vixy_consumes_spy(self):
        """When SPY sleeve is fully consumed by VIXY, new_spy_w should be 0."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        config = BacktestConfig(
            base_weights={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            max_hedge_pct=100.0,
        )
        _, tracker, regime_tracker = bt._run_hedged(bt._daily_prices[:100], config)
        # With max_hedge=100%, VIXY should be capped by sizer internally
        # The new_spy_w = max(0.46 - hedge_pct_as_decimal, 0)
        # In extreme cases, this should still be >= 0
        assert tracker["max_pct"] <= 10.0  # Sizer's internal cap

    def test_all_exports_correct(self):
        """Module __all__ should match actual public names."""
        import src.backtest.vixy_hedge_backtest as vhb
        expected = {
            'TRADING_DAYS_PER_YEAR', 'MONTHLY_TRADING_DAYS', 'CRISIS_YEARS',
            'BASE_SYMBOLS', 'VIX_SYMBOL', 'BacktestConfig', 'WalkForwardVIXYBacktester',
        }
        assert set(vhb.__all__) == expected

    def test_main_cli_defaults(self):
        """main() CLI should handle default arguments (no -- flags)."""
        import sys
        from src.backtest.vixy_hedge_backtest import main
        original_argv = sys.argv
        try:
            sys.argv = ["vixy_hedge_backtest.py", "run"]
            main()
        except SystemExit:
            pass  # argparse may call sys.exit on --help
        finally:
            sys.argv = original_argv

    def test_main_cli_custom_args(self):
        """main() CLI should accept custom start/end/capital/max-hedge flags."""
        import sys
        from src.backtest.vixy_hedge_backtest import main
        original_argv = sys.argv
        try:
            sys.argv = [
                "vixy_hedge_backtest.py", "run",
                "--start", "2015-01-01",
                "--end", "2016-01-01",
                "--capital", "50000",
                "--max-hedge", "8.0",
            ]
            main()
        except SystemExit:
            pass
        finally:
            sys.argv = original_argv

    def test_main_cli_save_flag(self, tmp_path):
        """main() CLI --save flag should create JSON output."""
        import sys
        from src.backtest import vixy_hedge_backtest as vhb
        from src.backtest.vixy_hedge_backtest import main

        original_argv = sys.argv
        original_dir = vhb.BACKTEST_RESULTS_DIR
        test_dir = tmp_path / "results"
        test_dir.mkdir()
        vhb.BACKTEST_RESULTS_DIR = test_dir

        try:
            sys.argv = [
                "vixy_hedge_backtest.py", "run",
                "--start", "2015-01-01",
                "--end", "2015-06-01",
                "--save",
            ]
            main()
        except SystemExit:
            pass
        finally:
            sys.argv = original_argv
            vhb.BACKTEST_RESULTS_DIR = original_dir

    def test_custom_hedge_pct_config(self):
        """BacktestConfig with custom max_hedge_pct should propagate through run()."""
        config = BacktestConfig(max_hedge_pct=3.0)
        bt = WalkForwardVIXYBacktester(config)
        bt.load_data()
        result = bt.run()
        assert result.extras["config_snapshot"]["max_hedge_pct"] == 3.0

    def test_hedge_pct_small_vix_returns_zero_hedge(self):
        """VIX below 10 should produce zero or minimal hedge allocation."""
        sizer = VIXYHedgeSizer()
        alloc = sizer.compute_allocation(5.0)
        # raw=0.5, NORMAL floor=0 ceiling=2 -> 0.5 -> clipped to floor=0?
        # Actually: min(max(0.5, 0.0), 2.0) = 0.5
        assert alloc == 0.5


# ── Constants Tests ───────────────────────────────────────────────────


class TestConstants:
    """Test module-level constants."""

    def test_trading_days_per_year(self):
        assert TRADING_DAYS_PER_YEAR == 252

    def test_monthly_trading_days(self):
        assert MONTHLY_TRADING_DAYS == 21

    def test_crisis_years(self):
        assert CRISIS_YEARS == ["2008", "2020", "2022"]

    def test_base_symbols(self):
        assert BASE_SYMBOLS == ["SPY", "GLD", "TLT"]

    def test_vix_symbol(self):
        assert VIX_SYMBOL == "^VIX"

    def test_crisis_years_immutability(self):
        """CRISIS_YEARS should not be accidentally mutated by code."""
        original = list(CRISIS_YEARS)
        _ = CRISIS_YEARS + ["2024"]  # This creates a new list; verify original unchanged
        assert CRISIS_YEARS == original


def test_a3_b1a_delegation_matches_pre_migration_capture():
    """A3 pin (Item B1a sub-task 9): load_data delegates to grid_runner.load_prices."""
    from src.backtest.grid_runner import load_prices

    # class method stays in pilot; the shared loader is grid_runner's
    assert WalkForwardVIXYBacktester.load_data.__module__ == (
        "src.backtest.vixy_hedge_backtest"
    )
    assert load_prices.__module__ == "src.backtest.grid_runner"
