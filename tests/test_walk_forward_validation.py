#!/usr/bin/env python3
"""
Tests for scripts/walk_forward_validation.py — walk-forward validation.
"""
import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import under scripts namespace
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from walk_forward_validation import (
    generate_grid_configs, run_single_window, run_walk_forward,
)


class TestGenerateGridConfigs:
    """Tests for grid config generation."""

    def test_produces_configs(self):
        configs = generate_grid_configs()
        assert len(configs) > 30

    def test_champion_present(self):
        configs = generate_grid_configs()
        champion = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        assert champion in configs

    def test_all_weights_sum_to_one(self):
        configs = generate_grid_configs()
        for config in configs:
            total = sum(config.values())
            assert abs(total - 1.0) < 0.01, f"Weights don't sum to 1: {config}"

    def test_all_weights_non_negative(self):
        configs = generate_grid_configs()
        for config in configs:
            for sym, w in config.items():
                assert w >= 0, f"{sym} has negative weight: {w}"

    def test_fine_sweep_range(self):
        """Fine sweep configs should include the champion region."""
        configs = generate_grid_configs()
        spy_vals = [c["SPY"] for c in configs if c.get("TLT", 0) > 0]
        assert min(spy_vals) <= 0.46
        assert max(spy_vals) >= 0.54


class TestRunSingleWindow:
    """Tests for single walk-forward window execution."""

    @pytest.fixture
    def synthetic_prices(self):
        """Synthetic price data with known properties."""
        np.random.seed(42)
        dates = pd.date_range("2010-01-01", periods=2000, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.012, 2000)),
            "GLD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.011, 2000)),
            "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.009, 2000)),
        }, index=dates)
        return prices

    def test_basic_window(self, synthetic_prices):
        configs = generate_grid_configs()
        train_idx = np.arange(0, 1500)
        test_idx = np.arange(1520, 1750)  # 20-day gap

        result = run_single_window(synthetic_prices, train_idx, test_idx, configs)
        assert result is not None
        assert "is_sharpe" in result
        assert "oos_sharpe" in result
        assert "champion_weights" in result

    def test_is_sharpe_positive_for_good_data(self, synthetic_prices):
        configs = generate_grid_configs()
        train_idx = np.arange(0, 1500)
        test_idx = np.arange(1520, 1750)

        result = run_single_window(synthetic_prices, train_idx, test_idx, configs)
        # With positive drift, IS Sharpe should generally be positive
        assert result["is_sharpe"] > 0

    def test_champion_weights_valid(self, synthetic_prices):
        configs = generate_grid_configs()
        train_idx = np.arange(0, 1500)
        test_idx = np.arange(1520, 1750)

        result = run_single_window(synthetic_prices, train_idx, test_idx, configs)
        weights = result["champion_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_too_few_train_days(self, synthetic_prices):
        configs = generate_grid_configs()
        train_idx = np.arange(0, 30)  # Too few
        test_idx = np.arange(50, 100)

        result = run_single_window(synthetic_prices, train_idx, test_idx, configs)
        assert result is None

    def test_date_ranges_present(self, synthetic_prices):
        configs = generate_grid_configs()
        train_idx = np.arange(0, 1500)
        test_idx = np.arange(1520, 1750)

        result = run_single_window(synthetic_prices, train_idx, test_idx, configs)
        assert "train_start" in result
        assert "test_end" in result
        assert result["train_days"] > 0
        assert result["test_days"] > 0


class TestRunWalkForward:
    """Tests for full walk-forward validation pipeline."""

    @pytest.fixture
    def long_synthetic_prices(self):
        """Longer synthetic price data for walk-forward testing."""
        np.random.seed(123)
        dates = pd.date_range("2005-01-03", periods=3000, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.012, 3000)),
            "GLD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.011, 3000)),
            "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.009, 3000)),
        }, index=dates)
        return prices

    def test_basic_walk_forward(self, long_synthetic_prices):
        result = run_walk_forward(
            long_synthetic_prices,
            n_splits=3,
            test_size=126,
            gap=10,
        )
        assert "n_windows" in result
        assert result["n_windows"] > 0
        assert "walk_forward_efficiency" in result

    def test_oos_sharpe_distribution(self, long_synthetic_prices):
        result = run_walk_forward(
            long_synthetic_prices,
            n_splits=3,
            test_size=126,
            gap=10,
        )
        oos = result["oos_sharpe"]
        assert "mean" in oos
        assert "std" in oos
        assert "min" in oos
        assert "max" in oos

    def test_weight_consistency(self, long_synthetic_prices):
        result = run_walk_forward(
            long_synthetic_prices,
            n_splits=3,
            test_size=126,
            gap=10,
        )
        weights = result["champion_weight_consistency"]
        for sym in ["SPY", "GLD", "TLT"]:
            assert sym in weights
            assert "mean" in weights[sym]
            assert "std" in weights[sym]

    def test_dsr_computed(self, long_synthetic_prices):
        result = run_walk_forward(
            long_synthetic_prices,
            n_splits=3,
            test_size=126,
            gap=10,
        )
        assert "dsr_champion_oos" in result
        assert "dsr_average_oos" in result
        assert 0.0 <= result["dsr_champion_oos"] <= 1.0
        assert 0.0 <= result["dsr_average_oos"] <= 1.0

    def test_windows_list_populated(self, long_synthetic_prices):
        result = run_walk_forward(
            long_synthetic_prices,
            n_splits=3,
            test_size=126,
            gap=10,
        )
        assert len(result["windows"]) == result["n_windows"]
        for w in result["windows"]:
            assert "is_sharpe" in w
            assert "oos_sharpe" in w
            assert "champion_weights" in w


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
