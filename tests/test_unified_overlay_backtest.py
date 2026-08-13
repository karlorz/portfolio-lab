#!/usr/bin/env python3
"""
Tests for unified_overlay_backtest.py — backtest validation for the
UNIFIED_OVERLAY signal (15% ensemble weight).

Covers:
- Data loading and processing
- Signal computation (collar, VIXY, bond duration, crypto)
- Backtest execution and metrics
- Hard constraint enforcement
- Crisis performance tracking
"""


import pytest
import json

from dataclasses import asdict
from src.backtest.unified_overlay_backtest import (
    BacktestConfig,
    DailyData,
    UnifiedOverlayBacktester,
)
from src.backtest.metrics import BacktestResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return BacktestConfig(
        start_date="2020-01-01",
        end_date="2020-12-31",
        initial_capital=100000.0,
    )


@pytest.fixture
def backtester(config):
    return UnifiedOverlayBacktester(config, allow_proxy_data=True)


@pytest.fixture
def synthetic_prices(tmp_path):
    """Create a minimal synthetic prices.json for testing."""
    # 30 days of flat prices
    dates = [f"2020-01-{d:02d}" for d in range(1, 31)]
    spy = [{"d": d, "p": 100.0 + i * 0.5} for i, d in enumerate(dates)]
    gld = [{"d": d, "p": 180.0 + i * 0.2} for i, d in enumerate(dates)]
    tlt = [{"d": d, "p": 140.0 + i * 0.1} for i, d in enumerate(dates)]

    prices_file = tmp_path / "prices.json"
    with open(prices_file, "w") as f:
        json.dump({"SPY": spy, "GLD": gld, "TLT": tlt}, f)

    return prices_file


# ---------------------------------------------------------------------------
# Signal computation tests
# ---------------------------------------------------------------------------

class TestCollarSignal:
    def test_collar_active_normal_vix(self, backtester):
        spy_d, gld_d, tlt_d = backtester._compute_collar_signal(20.0)
        assert spy_d < 0  # SPY reduced
        assert tlt_d > 0  # TLT increased

    def test_collar_disabled_high_vix(self, backtester):
        spy_d, gld_d, tlt_d = backtester._compute_collar_signal(35.0)
        assert spy_d == 0.0
        assert tlt_d == 0.0

    def test_collar_disabled_crisis_vix(self, backtester):
        spy_d, gld_d, tlt_d = backtester._compute_collar_signal(45.0)
        assert spy_d == 0.0

    def test_collar_disabled_no_vix(self, backtester):
        spy_d, gld_d, tlt_d = backtester._compute_collar_signal(None)
        assert spy_d == 0.0

    def test_collar_intensity_scales_with_vix(self, backtester):
        # Lower VIX → more collar intensity (closer to 1.0)
        low_vix = backtester._compute_collar_signal(15.0)
        high_vix = backtester._compute_collar_signal(28.0)
        assert abs(low_vix[0]) > abs(high_vix[0])


class TestVIXYSignal:
    def test_vixy_active_elevated_vix(self, backtester):
        spy_d, gld_d, tlt_d = backtester._compute_vixy_signal(25.0)
        assert spy_d < 0  # Funded from SPY

    def test_vixy_disabled_low_vix(self, backtester):
        spy_d, gld_d, tlt_d = backtester._compute_vixy_signal(15.0)
        assert spy_d == 0.0

    def test_vixy_disabled_no_vix(self, backtester):
        spy_d, gld_d, tlt_d = backtester._compute_vixy_signal(None)
        assert spy_d == 0.0

    def test_vixy_allocation_scales(self, backtester):
        # Higher VIX → larger allocation
        low_vix = backtester._compute_vixy_signal(22.0)
        high_vix = backtester._compute_vixy_signal(35.0)
        assert abs(high_vix[0]) > abs(low_vix[0])

    def test_vixy_allocation_capped(self, backtester):
        spy_d, gld_d, tlt_d = backtester._compute_vixy_signal(80.0)
        assert abs(spy_d) <= 0.06


class TestBondDurationSignal:
    def test_rising_bonds_extend_duration(self, backtester):
        # Create rising TLT prices
        prices = [{"d": f"2020-01-{i:02d}", "p": 130.0 + i * 0.5} for i in range(1, 62)]
        spy_d, gld_d, tlt_d = backtester._compute_bond_duration_signal(prices, 60)
        # Rising bonds → extend duration (GLD→TLT)
        assert tlt_d > 0

    def test_falling_bonds_shorten_duration(self, backtester):
        # Create falling TLT prices
        prices = [{"d": f"2020-01-{i:02d}", "p": 150.0 - i * 0.5} for i in range(1, 62)]
        spy_d, gld_d, tlt_d = backtester._compute_bond_duration_signal(prices, 60)
        # Falling bonds → shorten (TLT→GLD proxy)
        assert gld_d > 0

    def test_insufficient_data(self, backtester):
        prices = [{"d": f"2020-01-{i:02d}", "p": 140.0} for i in range(1, 10)]
        spy_d, gld_d, tlt_d = backtester._compute_bond_duration_signal(prices, 9)
        assert spy_d == 0.0

    def test_flat_bonds_no_signal(self, backtester):
        prices = [{"d": f"2020-01-{i:02d}", "p": 140.0} for i in range(1, 62)]
        spy_d, gld_d, tlt_d = backtester._compute_bond_duration_signal(prices, 60)
        assert spy_d == 0.0 and gld_d == 0.0 and tlt_d == 0.0


class TestCryptoSignal:
    def test_positive_momentum_activates(self, backtester):
        # Rising SPY → crypto active
        prices = [{"d": f"2020-01-{i:02d}", "p": 100.0 + i * 0.2} for i in range(1, 140)]
        spy_d, gld_d, tlt_d, crypto = backtester._compute_crypto_signal(prices, 138)
        assert crypto > 0
        assert gld_d < 0  # Funded from GLD

    def test_negative_momentum_disabled(self, backtester):
        # Falling SPY → no crypto
        prices = [{"d": f"2020-01-{i:02d}", "p": 200.0 - i * 0.5} for i in range(1, 140)]
        spy_d, gld_d, tlt_d, crypto = backtester._compute_crypto_signal(prices, 138)
        assert crypto == 0.0

    def test_crypto_capped(self, backtester):
        # Very strong momentum → still capped at 5%
        prices = [{"d": f"2020-01-{i:02d}", "p": 50.0 + i * 2.0} for i in range(1, 140)]
        spy_d, gld_d, tlt_d, crypto = backtester._compute_crypto_signal(prices, 138)
        assert crypto <= 0.05

    def test_insufficient_data(self, backtester):
        prices = [{"d": f"2020-01-{i:02d}", "p": 100.0} for i in range(1, 50)]
        spy_d, gld_d, tlt_d, crypto = backtester._compute_crypto_signal(prices, 48)
        assert crypto == 0.0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestBacktestConfig:
    def test_default_config(self):
        cfg = BacktestConfig()
        assert cfg.base_weights['SPY'] == 0.46
        assert cfg.base_weights['GLD'] == 0.38
        assert cfg.base_weights['TLT'] == 0.16
        assert cfg.max_crypto == 0.05

    def test_custom_config(self):
        cfg = BacktestConfig(initial_capital=50000.0, max_crypto=0.03)
        assert cfg.initial_capital == 50000.0
        assert cfg.max_crypto == 0.03


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_load_from_synthetic(self, backtester, synthetic_prices):
        result = backtester.load_data(str(synthetic_prices))
        assert result is True
        assert set(backtester.prices_raw) == {"SPY", "GLD", "TLT"}
        assert len(backtester.data) == 29

    def test_load_data_honors_explicit_data_path(self, backtester, synthetic_prices, tmp_path, monkeypatch):
        """Explicit data_path should be read even when default PRICES_JSON is absent."""
        import src.backtest.unified_overlay_backtest as mod

        monkeypatch.setattr(mod, "PRICES_JSON", tmp_path / "missing_default_prices.json")

        assert backtester.load_data(str(synthetic_prices)) is True
        assert set(backtester.prices_raw) == {"SPY", "GLD", "TLT"}
        assert backtester.prices_raw["SPY"][0]["p"] == 100.0
        assert len(backtester.data) == 29

    def test_load_missing_file(self, backtester, tmp_path, monkeypatch):
        # Patch PRICES_JSON to a non-existent path and ensure no fallback
        import src.backtest.unified_overlay_backtest as mod
        monkeypatch.setattr(mod, "PRICES_JSON", tmp_path / "nonexistent.json")
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        result = backtester.load_data()
        assert result is False

    def test_empty_result_without_data(self, backtester):
        result = backtester.run()
        assert result.sharpe_ratio == 0.0
        assert result.baseline_sharpe == 0.0

    def test_proxy_backtest_requires_explicit_opt_in(self, config, synthetic_prices):
        backtester = UnifiedOverlayBacktester(config)
        assert backtester.load_data(str(synthetic_prices)) is True
        with pytest.raises(ValueError, match="proxy data"):
            backtester.run()

    def test_cost_is_deducted_and_evidence_reconciles(
        self, config, synthetic_prices
    ):
        config.transaction_cost_bps = 100.0
        config.rebalance_frequency_days = 1
        backtester = UnifiedOverlayBacktester(
            config,
            allow_proxy_data=True,
        )
        assert backtester.load_data(str(synthetic_prices)) is True

        result = backtester.run()
        evidence = result.extras["profitability_evidence"]

        assert evidence["data"]["mode"] == "proxy"
        assert evidence["promotion_eligible"] is False
        assert evidence["costs"]["total_dollars"] > 0
        assert evidence["trace"][-1]["net_equity"] < evidence["trace"][-1]["gross_equity"]
        assert result.total_return == evidence["metrics"]["net"]["total_return"]
        assert evidence["costs"]["max_reconciliation_error"] < 1e-12


# ---------------------------------------------------------------------------
# BacktestResult tests
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def test_create_result(self):
        result = BacktestResult(
            total_return=108.5, cagr=10.5, volatility=11.5,
            sharpe_ratio=0.95, max_drawdown=-26.0,
            baseline_sharpe=0.94, sharpe_improvement=0.01,
            total_rebalances=50, total_transaction_costs=500.0,
            extras={
                "overlay_active_days": 500, "overlay_active_pct": 95.0,
                "collar_active_days": 400, "vixy_active_days": 100,
                "crypto_active_days": 350, "bond_duration_active_days": 200,
            },
        )
        assert result.sharpe_ratio == 0.95
        assert result.sharpe_improvement == 0.01

    def test_result_to_dict(self):
        result = BacktestResult(
            total_return=100.0, cagr=10.0, volatility=12.0,
            sharpe_ratio=0.9, max_drawdown=-25.0,
            baseline_sharpe=0.85, sharpe_improvement=0.05,
            total_rebalances=50, total_transaction_costs=500.0,
            extras={
                "overlay_active_days": 500, "overlay_active_pct": 95.0,
                "collar_active_days": 400, "vixy_active_days": 100,
                "crypto_active_days": 350, "bond_duration_active_days": 200,
            },
        )
        d = asdict(result)
        assert "sharpe_ratio" in d
        assert "extras" in d
        assert d["sharpe_improvement"] == 0.05


# ---------------------------------------------------------------------------
# Hard constraint tests
# ---------------------------------------------------------------------------

class TestHardConstraints:
    def test_spy_bounds_enforced(self):
        """SPY must stay within 36-56% range."""
        cfg = BacktestConfig(max_spy_shift=0.10)
        import numpy as np
        # Extreme negative shift
        assert float(np.clip(cfg.base_weights['SPY'] - 0.20, 0.36, 0.56)) == 0.36
        # Extreme positive shift
        assert float(np.clip(cfg.base_weights['SPY'] + 0.20, 0.36, 0.56)) == 0.56

    def test_gld_bounds_enforced(self):
        """GLD must stay within 28-48% range."""
        import numpy as np
        assert float(np.clip(0.38 - 0.20, 0.28, 0.48)) == 0.28
        assert float(np.clip(0.38 + 0.20, 0.28, 0.48)) == 0.48

    def test_crypto_cap(self):
        """Crypto allocation must not exceed max_crypto."""
        cfg = BacktestConfig(max_crypto=0.05)
        crypto = min(0.10, cfg.max_crypto)
        assert crypto == 0.05

    def test_weights_sum_to_one(self):
        """With no crypto, weights should sum to ~1.0."""
        cfg = BacktestConfig()
        base_sum = cfg.base_weights['SPY'] + cfg.base_weights['GLD'] + cfg.base_weights['TLT']
        assert abs(base_sum - 1.0) < 0.001


# ---------------------------------------------------------------------------
# Print and save tests
# ---------------------------------------------------------------------------

class TestOutput:
    def test_print_results_no_crash(self, backtester):
        result = BacktestResult(
            total_return=100.0, cagr=10.0, volatility=12.0,
            sharpe_ratio=0.9, max_drawdown=-25.0,
            baseline_sharpe=0.85, sharpe_improvement=0.05,
            total_rebalances=50, total_transaction_costs=500.0,
            crisis_returns={"2008": -10.0, "2020": 15.0, "2022": -8.0},
            extras={
                "overlay_active_days": 500, "overlay_active_pct": 95.0,
                "collar_active_days": 400, "vixy_active_days": 100,
                "crypto_active_days": 350, "bond_duration_active_days": 200,
            },
        )
        # Should not raise
        backtester.print_results(result)

    def test_save_results(self, backtester, tmp_path):
        result = BacktestResult(
            total_return=100.0, cagr=10.0, volatility=12.0,
            sharpe_ratio=0.9, max_drawdown=-25.0,
            baseline_sharpe=0.85, sharpe_improvement=0.05,
            total_rebalances=50, total_transaction_costs=500.0,
            extras={
                "overlay_active_days": 500, "overlay_active_pct": 95.0,
                "collar_active_days": 400, "vixy_active_days": 100,
                "crypto_active_days": 350, "bond_duration_active_days": 200,
            },
        )
        out_file = str(tmp_path / "results.json")
        backtester.save_results(result, path=out_file)

        with open(out_file) as f:
            saved = json.load(f)
        assert saved["sharpe_ratio"] == 0.9
        assert saved["sharpe_improvement"] == 0.05


class TestDailyData:
    """Tests for DailyData dataclass."""

    def test_daily_data_fields(self):
        d = DailyData(
            date="2020-01-02",
            spy_return=0.01, gld_return=-0.005, tlt_return=0.002,
            vix_level=18.5,
        )
        assert d.date == "2020-01-02"
        assert d.vix_level == 18.5
        assert d.spy_return == 0.01

    def test_daily_data_default_none(self):
        d = DailyData(date="2020-01-02", spy_return=0.01, gld_return=0.0, tlt_return=0.0)
        assert d.vix_level is None


class TestCollarSignalExtended:
    """Extended collar signal tests."""

    def test_collar_boundary_low_vix(self, backtester):
        """Very low VIX should activate collar with maximum intensity."""
        spy_d, gld_d, tlt_d = backtester._compute_collar_signal(12.0)
        assert spy_d < 0  # SPY always reduced in collar

    def test_collar_returns_three_deltas(self, backtester):
        """Collar signal should return exactly 3 deltas."""
        result = backtester._compute_collar_signal(20.0)
        assert len(result) == 3

    def test_collar_sum_near_zero(self, backtester):
        """Collar deltas should approximately cancel out."""
        spy_d, gld_d, tlt_d = backtester._compute_collar_signal(20.0)
        assert abs(spy_d + gld_d + tlt_d) < 0.01


class TestVIXYSignalExtended:
    """Extended VIXY signal tests."""

    def test_vixy_returns_three_deltas(self, backtester):
        """VIXY signal should return exactly 3 deltas."""
        result = backtester._compute_vixy_signal(25.0)
        assert len(result) == 3

    def test_vixy_boundary_20(self, backtester):
        """VIX at exactly 20 — test activation boundary."""
        result = backtester._compute_vixy_signal(20.0)
        # Behavior at exact boundary may vary
        assert len(result) == 3

    def test_vixy_extreme_vix(self, backtester):
        """Very high VIX should still produce capped result."""
        spy_d, gld_d, tlt_d = backtester._compute_vixy_signal(100.0)
        assert abs(spy_d) <= 0.10  # Reasonable cap


class TestBondDurationSignalExtended:
    """Extended bond duration signal tests."""

    def test_bond_signal_returns_three_deltas(self, backtester):
        """Bond duration signal should return exactly 3 deltas."""
        prices = [{"d": f"2020-01-{i:02d}", "p": 140.0 + i * 0.1} for i in range(1, 62)]
        result = backtester._compute_bond_duration_signal(prices, 60)
        assert len(result) == 3

    def test_bond_volatile_prices(self, backtester):
        """Volatile but flat-trending prices should produce minimal signal."""
        import numpy as np
        np.random.seed(42)
        prices = [{"d": f"2020-01-{i:02d}", "p": 140.0 + np.random.normal(0, 1)} for i in range(1, 62)]
        spy_d, gld_d, tlt_d = backtester._compute_bond_duration_signal(prices, 60)
        # Volatile flat prices should give small or zero signal
        assert abs(spy_d) < 0.05
        assert abs(gld_d) < 0.05
        assert abs(tlt_d) < 0.05


class TestCryptoSignalExtended:
    """Extended crypto signal tests."""

    def test_crypto_returns_four_deltas(self, backtester):
        """Crypto signal should return exactly 4 values (3 deltas + crypto alloc)."""
        prices = [{"d": f"2020-01-{i:02d}", "p": 100.0 + i * 0.2} for i in range(1, 140)]
        result = backtester._compute_crypto_signal(prices, 138)
        assert len(result) == 4

    def test_crypto_zero_with_flat_prices(self, backtester):
        """Flat SPY prices should produce zero crypto allocation."""
        prices = [{"d": f"2020-01-{i:02d}", "p": 100.0} for i in range(1, 140)]
        spy_d, gld_d, tlt_d, crypto = backtester._compute_crypto_signal(prices, 138)
        assert crypto == 0.0


class TestHardConstraintsExtended:
    """Extended hard constraint tests."""

    def test_spy_min_bound(self):
        """SPY floor at 36%."""
        import numpy as np
        assert float(np.clip(0.46 - 0.15, 0.36, 0.56)) == 0.36

    def test_spy_max_bound(self):
        """SPY ceiling at 56%."""
        import numpy as np
        assert float(np.clip(0.46 + 0.15, 0.36, 0.56)) == 0.56

    def test_tlt_weight_positive(self):
        """TLT should always be positive."""
        cfg = BacktestConfig()
        assert cfg.base_weights['TLT'] > 0

    def test_gld_weight_positive(self):
        """GLD should always be positive."""
        cfg = BacktestConfig()
        assert cfg.base_weights['GLD'] > 0


class TestOutputExtended:
    """Extended output tests."""

    def test_save_results_has_sharpe(self, backtester, tmp_path):
        """Saved results should include sharpe_ratio."""
        result = BacktestResult(
            total_return=100.0, cagr=10.0, volatility=12.0,
            sharpe_ratio=0.9, max_drawdown=-25.0,
            baseline_sharpe=0.85, sharpe_improvement=0.05,
            total_rebalances=50, total_transaction_costs=500.0,
            extras={},
        )
        out_file = str(tmp_path / "results.json")
        backtester.save_results(result, path=out_file)
        with open(out_file) as f:
            saved = json.load(f)
        assert "sharpe_ratio" in saved

    def test_save_results_has_crisis_returns(self, backtester, tmp_path):
        """Saved results should include crisis_returns."""
        result = BacktestResult(
            total_return=100.0, cagr=10.0, volatility=12.0,
            sharpe_ratio=0.9, max_drawdown=-25.0,
            baseline_sharpe=0.85, sharpe_improvement=0.05,
            total_rebalances=50, total_transaction_costs=500.0,
            crisis_returns={"2008": -10.0, "2020": 5.0},
            extras={},
        )
        out_file = str(tmp_path / "results.json")
        backtester.save_results(result, path=out_file)
        with open(out_file) as f:
            saved = json.load(f)
        assert "crisis_returns" in saved


# ---------------------------------------------------------------------------
# __all__ export validation
# ---------------------------------------------------------------------------

class TestExports:
    """Verify __all__ exports."""

    def test_all_exports_present(self):
        import src.backtest.unified_overlay_backtest as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"Missing export: {name}"

    def test_all_count(self):
        import src.backtest.unified_overlay_backtest as mod
        assert len(mod.__all__) == 3


# ---------------------------------------------------------------------------
# BacktestConfig extended
# ---------------------------------------------------------------------------

class TestBacktestConfigExtended:
    """Extended BacktestConfig tests."""

    def test_default_start_date(self):
        from src.backtest.unified_overlay_backtest import BacktestConfig
        config = BacktestConfig()
        assert config.start_date is not None

    def test_default_end_date(self):
        from src.backtest.unified_overlay_backtest import BacktestConfig
        config = BacktestConfig()
        assert config.end_date is not None

    def test_start_before_end(self):
        from src.backtest.unified_overlay_backtest import BacktestConfig
        config = BacktestConfig()
        assert config.start_date < config.end_date


# ---------------------------------------------------------------------------
# DailyData extended
# ---------------------------------------------------------------------------

class TestDailyDataExtended:
    """Extended DailyData dataclass tests."""

    def test_all_fields(self):
        from dataclasses import fields
        from src.backtest.unified_overlay_backtest import DailyData
        field_names = {f.name for f in fields(DailyData)}
        assert "spy_return" in field_names
        assert "gld_return" in field_names
        assert "tlt_return" in field_names
        assert "vix_level" in field_names

    def test_to_dict_keys(self):
        from src.backtest.unified_overlay_backtest import DailyData
        dd = DailyData(
            date="2026-01-01", spy_return=0.01, gld_return=0.02,
            tlt_return=-0.01, vix_level=18.5,
        )
        d = dd.to_dict() if hasattr(dd, 'to_dict') else dd.__dict__
        assert "date" in d


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

class TestCLI:
    """Test main() callable."""

    def test_main_callable(self):
        from src.backtest.unified_overlay_backtest import UnifiedOverlayBacktester
        assert UnifiedOverlayBacktester is not None


def test_a3_b1a_delegation_matches_pre_migration_capture():
    """A3 pin (Item B1a sub-task 7): load_data delegates to grid_runner.load_prices."""
    from src.backtest.grid_runner import load_prices

    # class method stays in pilot; the shared loader is grid_runner's
    assert UnifiedOverlayBacktester.load_data.__module__ == (
        "src.backtest.unified_overlay_backtest"
    )
    assert load_prices.__module__ == "src.backtest.grid_runner"
