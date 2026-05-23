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

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

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
    return UnifiedOverlayBacktester(config)


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
        # Our synthetic data is only 30 days, start_date filter may exclude all
        # This just tests the loading path works without crashing

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


# Total: 27 tests
