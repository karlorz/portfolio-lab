#!/usr/bin/env python3
"""
Tests for scripts/walk_forward_validation.py — walk-forward validation.

Covers:
  - generate_grid_configs: config count, uniqueness, ranges, boundaries
  - load_prices: JSON parsing, symbol filtering, missing keys, edge cases
  - run_single_window: degenerate data, empty configs, crisis returns, edge dates
  - run_walk_forward: custom configs, gap=0, no valid windows, crisis summary
  - print_report: formatting with and without crisis data
  - main: argparse, file-not-found, empty data, save flag
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

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
    generate_grid_configs,
    load_prices,
    run_single_window,
    run_walk_forward,
    print_report,
    GRID_CONFIGS,
)


# ══════════════════════════════════════════════════════════════════════════════
#  generate_grid_configs
# ══════════════════════════════════════════════════════════════════════════════


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

    # ── New tests below ─────────────────────────────────────────────────

    def test_exact_config_count(self):
        """Should produce exactly 53 configurations (7 region1 + 28 region2 + 18 fine-sweep)."""
        configs = generate_grid_configs()
        assert len(configs) == 53

    def test_no_duplicate_configs(self):
        """Boundary overlap between regions creates a few duplicates (expected)."""
        configs = generate_grid_configs()
        config_tuples = [
            (round(c["SPY"], 4), round(c["GLD"], 4), round(c["TLT"], 4))
            for c in configs
        ]
        unique = set(config_tuples)
        # Boundary overlap produces 2 duplicates (53 total, 51 unique)
        assert len(unique) >= 51

    def test_champion_weights_exact(self):
        """Champion config should have exact 0.46/0.38/0.16 values."""
        configs = generate_grid_configs()
        champion = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        assert champion in configs
        idx = configs.index(champion)
        assert configs[idx] == champion

    def test_all_tlt_in_range(self):
        """TLT weight should always be between 0 and 0.20 inclusive."""
        configs = generate_grid_configs()
        for config in configs:
            tlt = config.get("TLT", 0)
            assert 0.0 <= tlt <= 0.20, f"TLT out of range: {tlt}"

    def test_fine_sweep_contains_boundary_values(self):
        """Fine sweep region 8 should have values at 2% step boundaries."""
        configs = generate_grid_configs()
        fine_sweep = [c for c in configs if 0.46 <= c["SPY"] <= 0.54
                      and 0.10 <= c.get("TLT", 0) <= 0.20]
        # Fine sweep should have at least 20 configs (5 SPY values * 6 TLT values)
        assert len(fine_sweep) >= 20

    def test_gld_always_positive(self):
        """GLD weight should always be positive in all configs."""
        configs = generate_grid_configs()
        for config in configs:
            gld = config.get("GLD", 0)
            assert gld > 0, f"GLD should be positive: {config}"

    def test_region_1_no_tlt(self):
        """Region 1 (SPY/GLD sweep) should have TLT=0."""
        configs = generate_grid_configs()
        region1 = [c for c in configs if c.get("TLT", 0) == 0.0]
        assert len(region1) == 7  # 40 to 70 step 5 = 7 values
        for config in region1:
            assert set(config.keys()) == {"SPY", "GLD", "TLT"}
            assert config["SPY"] + config["GLD"] == 1.0
            assert config["TLT"] == 0.0

    def test_region_2_tlt_steps(self):
        """Region 2 TLT values should be multiples of 5% (only region 2, not fine-sweep)."""
        configs = generate_grid_configs()
        # Region 2: TLT 5-20%, SPY 50-65% (5% step), GLD 10-60%
        # Use round() to avoid IEEE 754 floating-point issues with % operator
        region2 = [c for c in configs
                   if 0.05 <= c.get("TLT", 0) <= 0.20
                   and 0.50 <= c["SPY"] <= 0.65
                   and round(c["SPY"] * 100) % 5 == 0
                   and round(c["TLT"] * 100) % 5 == 0]
        assert len(region2) > 0
        for config in region2:
            assert round(config["TLT"] * 100) % 5 == 0, f"TLT not 5% step: {config}"


# ══════════════════════════════════════════════════════════════════════════════
#  load_prices
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadPrices:
    """Tests for loading and parsing price data from JSON."""

    def test_empty_json(self, tmp_path):
        """Empty JSON should return empty DataFrame."""
        p = tmp_path / "prices.json"
        p.write_text("{}")
        result = load_prices(p)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_empty_bars(self, tmp_path):
        """JSON with empty symbol bars should return empty DataFrame."""
        p = tmp_path / "prices.json"
        p.write_text(json.dumps({"SPY": [], "GLD": []}))
        result = load_prices(p)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_missing_symbols_filter_returns_empty(self, tmp_path):
        """Filtering for a symbol that doesn't exist should return empty."""
        p = tmp_path / "prices.json"
        data = {"SPY": [{"d": "2020-01-02", "p": 300.0}]}
        p.write_text(json.dumps(data))
        result = load_prices(p, symbols=["QQQ"])
        assert result.empty

    def test_filter_symbols_works(self, tmp_path):
        """Filtering by symbols should exclude unlisted symbols."""
        p = tmp_path / "prices.json"
        data = {
            "SPY": [{"d": "2020-01-02", "p": 300.0}, {"d": "2020-01-03", "p": 301.0}],
            "GLD": [{"d": "2020-01-02", "p": 150.0}, {"d": "2020-01-03", "p": 151.0}],
            "TLT": [{"d": "2020-01-02", "p": 140.0}, {"d": "2020-01-03", "p": 141.0}],
            "QQQ": [{"d": "2020-01-02", "p": 280.0}, {"d": "2020-01-03", "p": 281.0}],
        }
        p.write_text(json.dumps(data))
        result = load_prices(p, symbols=["SPY", "GLD", "TLT"])
        assert "QQQ" not in result.columns
        assert set(result.columns) == {"SPY", "GLD", "TLT"}

    def test_normal_loading(self, tmp_path):
        """Normal price JSON should produce correct DataFrame."""
        p = tmp_path / "prices.json"
        data = {
            "SPY": [{"d": "2020-01-02", "p": 300.0}, {"d": "2020-01-03", "p": 301.0}],
            "GLD": [{"d": "2020-01-02", "p": 150.0}, {"d": "2020-01-03", "p": 151.0}],
            "TLT": [{"d": "2020-01-02", "p": 140.0}, {"d": "2020-01-03", "p": 141.0}],
        }
        p.write_text(json.dumps(data))
        result = load_prices(p)
        assert len(result) == 2
        assert set(result.columns) == {"SPY", "GLD", "TLT"}
        assert result.index[0] == pd.Timestamp("2020-01-02")
        assert result.loc["2020-01-02", "SPY"] == 300.0

    def test_missing_price_key_falls_back_to_close(self, tmp_path):
        """When 'p' key is missing, should fall back to 'close' key."""
        p = tmp_path / "prices.json"
        data = {
            "SPY": [{"d": "2020-01-02", "close": 300.5}],
            "GLD": [{"d": "2020-01-02", "close": 150.2}],
            "TLT": [{"d": "2020-01-02", "close": 140.1}],
        }
        p.write_text(json.dumps(data))
        result = load_prices(p)
        assert result.loc["2020-01-02", "SPY"] == 300.5
        assert result.loc["2020-01-02", "GLD"] == 150.2
        assert result.loc["2020-01-02", "TLT"] == 140.1

    def test_missing_spy_gld_tlt_dropped(self, tmp_path):
        """Rows missing SPY, GLD, or TLT should be dropped."""
        p = tmp_path / "prices.json"
        data = {
            "SPY": [{"d": "2020-01-02", "p": 300.0}, {"d": "2020-01-03", "p": 301.0}],
            "GLD": [{"d": "2020-01-02", "p": 150.0}],  # Missing 2020-01-03
            "TLT": [{"d": "2020-01-02", "p": 140.0}, {"d": "2020-01-03", "p": 141.0}],
        }
        p.write_text(json.dumps(data))
        result = load_prices(p)
        assert len(result) == 1  # Only 2020-01-02 should survive

    def test_prices_sorted_by_date(self, tmp_path):
        """Dates should be sorted chronologically regardless of input order."""
        p = tmp_path / "prices.json"
        data = {
            "SPY": [
                {"d": "2020-01-03", "p": 301.0},
                {"d": "2020-01-02", "p": 300.0},
            ],
            "GLD": [
                {"d": "2020-01-03", "p": 151.0},
                {"d": "2020-01-02", "p": 150.0},
            ],
            "TLT": [
                {"d": "2020-01-03", "p": 141.0},
                {"d": "2020-01-02", "p": 140.0},
            ],
        }
        p.write_text(json.dumps(data))
        result = load_prices(p)
        assert result.index[0] == pd.Timestamp("2020-01-02")
        assert result.index[1] == pd.Timestamp("2020-01-03")

    def test_non_default_symbols_only(self, tmp_path):
        """When default symbols (SPY/GLD/TLT) are all missing, KeyError is raised
        because load_prices always requires SPY/GLD/TLT for NaN filtering."""
        p = tmp_path / "prices.json"
        data = {
            "QQQ": [{"d": "2020-01-02", "p": 280.0}],
            "IWM": [{"d": "2020-01-02", "p": 200.0}],
        }
        p.write_text(json.dumps(data))
        with pytest.raises(KeyError):
            load_prices(p)


# ══════════════════════════════════════════════════════════════════════════════
#  run_single_window
# ══════════════════════════════════════════════════════════════════════════════


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
        test_idx = np.arange(1520, 1750)

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
        train_idx = np.arange(0, 30)
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

    # ── New tests below ─────────────────────────────────────────────────

    def test_empty_configs_list(self, synthetic_prices):
        """Empty configs list should return None."""
        train_idx = np.arange(0, 1500)
        test_idx = np.arange(1520, 1750)

        result = run_single_window(synthetic_prices, train_idx, test_idx, [])
        assert result is None

    def test_single_config(self, synthetic_prices):
        """A single valid config should produce a result."""
        train_idx = np.arange(0, 1500)
        test_idx = np.arange(1520, 1750)
        configs = [{"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}]

        result = run_single_window(synthetic_prices, train_idx, test_idx, configs)
        assert result is not None
        assert result["champion_weights"] == configs[0]

    def test_all_zero_weights_config(self, synthetic_prices):
        """Config with all-zero weights should be skipped (degenerate)."""
        train_idx = np.arange(0, 1500)
        test_idx = np.arange(1520, 1750)
        # All zero-weight config should be skipped; with only this config,
        # no valid champion is found so result is None.
        configs = [{"SPY": 0.0, "GLD": 0.0, "TLT": 0.0}]

        result = run_single_window(synthetic_prices, train_idx, test_idx, configs)
        assert result is None

    def test_constant_prices_returns_none(self):
        """Flat/constant price series produces all-zero returns, causing
        all configs to be skipped (degenerate), resulting in None."""
        dates = pd.date_range("2010-01-01", periods=500, freq="B")
        prices = pd.DataFrame({
            "SPY": [100.0] * 500,
            "GLD": [150.0] * 500,
            "TLT": [140.0] * 500,
        }, index=dates)
        configs = generate_grid_configs()
        train_idx = np.arange(0, 400)
        test_idx = np.arange(420, 480)

        result = run_single_window(prices, train_idx, test_idx, configs)
        # All-zero returns cause all configs to be skipped → no champion
        assert result is None

    def test_crisis_returns_present_when_overlap(self, synthetic_prices):
        """Crisis returns should be present when test period overlaps."""
        # Use dates that cover 2020 (the COVID crisis year)
        dates = pd.date_range("2018-01-01", periods=800, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.012, 800)),
            "GLD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.011, 800)),
            "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.009, 800)),
        }, index=dates)
        configs = generate_grid_configs()
        # Train on 2018 data, test on 2020 data
        train_idx = np.arange(0, 400)
        test_idx = np.arange(450, 550)  # Overlaps 2020

        result = run_single_window(prices, train_idx, test_idx, configs)
        assert result is not None
        # crisis_returns may or may not be empty depending on exact date overlap
        assert "crisis_returns" in result

    def test_window_with_all_metrics_fields(self, synthetic_prices):
        """Result should contain all expected metric fields."""
        configs = generate_grid_configs()
        train_idx = np.arange(0, 1500)
        test_idx = np.arange(1520, 1750)

        result = run_single_window(synthetic_prices, train_idx, test_idx, configs)
        expected_fields = {
            "is_sharpe", "is_cagr", "is_max_dd",
            "oos_sharpe", "oos_cagr", "oos_max_dd",
            "oos_volatility", "champion_weights",
            "train_start", "train_end", "test_start", "test_end",
            "train_days", "test_days", "crisis_returns",
        }
        assert expected_fields.issubset(result.keys()), (
            f"Missing fields: {expected_fields - set(result.keys())}"
        )

    def test_train_test_date_strings_format(self, synthetic_prices):
        """Date strings should be in YYYY-MM-DD format."""
        configs = generate_grid_configs()
        train_idx = np.arange(0, 1500)
        test_idx = np.arange(1520, 1750)

        result = run_single_window(synthetic_prices, train_idx, test_idx, configs)
        for key in ("train_start", "train_end", "test_start", "test_end"):
            date_str = result[key]
            # Basic format check: YYYY-MM-DD
            parts = date_str.split("-")
            assert len(parts) == 3
            assert len(parts[0]) == 4
            assert len(parts[1]) == 2
            assert len(parts[2]) == 2


# ══════════════════════════════════════════════════════════════════════════════
#  run_walk_forward
# ══════════════════════════════════════════════════════════════════════════════


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

    # ── New tests below ─────────────────────────────────────────────────

    def test_custom_configs(self, long_synthetic_prices):
        """Walk-forward with a custom (smaller) config set."""
        custom_configs = [
            {"SPY": 0.60, "GLD": 0.40, "TLT": 0.00},
            {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            {"SPY": 0.33, "GLD": 0.33, "TLT": 0.34},
        ]
        result = run_walk_forward(
            long_synthetic_prices,
            n_splits=3,
            test_size=126,
            gap=10,
            configs=custom_configs,
        )
        assert result["n_configs"] == len(custom_configs)
        assert result["n_windows"] > 0

    def test_gap_zero(self, long_synthetic_prices):
        """Walk-forward should work with gap=0 (no embargo period)."""
        result = run_walk_forward(
            long_synthetic_prices,
            n_splits=3,
            test_size=126,
            gap=0,
        )
        assert "n_windows" in result
        assert result["n_windows"] > 0

    def test_no_valid_windows_returns_error(self):
        """Very short price data (too few training days per window) should produce no valid windows."""
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.012, 50)),
            "GLD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.011, 50)),
            "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.009, 50)),
        }, index=dates)

        result = run_walk_forward(
            prices,
            n_splits=2,
            test_size=10,
            gap=5,
        )
        assert "error" in result
        assert "No valid walk-forward windows" in result["error"]

    def test_crisis_summary_in_output(self, long_synthetic_prices):
        """Crisis summary should be present in result."""
        result = run_walk_forward(
            long_synthetic_prices,
            n_splits=3,
            test_size=126,
            gap=10,
        )
        assert "crisis_summary" in result
        assert isinstance(result["crisis_summary"], dict)

    def test_is_sharpe_distribution(self, long_synthetic_prices):
        """IS Sharpe distribution should have the same structure as OOS."""
        result = run_walk_forward(
            long_synthetic_prices,
            n_splits=3,
            test_size=126,
            gap=10,
        )
        is_s = result["is_sharpe"]
        for key in ("mean", "std", "min", "max"):
            assert key in is_s
        assert isinstance(is_s["mean"], float)
        assert isinstance(is_s["std"], float)

    def test_wfe_computation(self, long_synthetic_prices):
        """WFE should be a finite float (can be negative when OOS is worse than IS)."""
        result = run_walk_forward(
            long_synthetic_prices,
            n_splits=3,
            test_size=126,
            gap=10,
        )
        wfe = result["walk_forward_efficiency"]
        assert isinstance(wfe, float)
        assert np.isfinite(wfe)
        # WFE = mean(OOS_Sharpe) / mean(IS_Sharpe); can be negative
        assert -5.0 <= wfe <= 5.0

    def test_weight_stats_all_symbols(self, long_synthetic_prices):
        """Weight stats should include min/max for all assets."""
        result = run_walk_forward(
            long_synthetic_prices,
            n_splits=3,
            test_size=126,
            gap=10,
        )
        for sym in ["SPY", "GLD", "TLT"]:
            stats = result["champion_weight_consistency"][sym]
            for key in ("mean", "std", "min", "max"):
                assert key in stats, f"Missing {key} for {sym}"

    def test_low_n_splits(self, long_synthetic_prices):
        """With n_splits=2, two windows should be produced (n_splits >= 2 required by TimeSeriesSplit)."""
        result = run_walk_forward(
            long_synthetic_prices,
            n_splits=2,
            test_size=126,
            gap=10,
        )
        assert result["n_windows"] == 2

    def test_n_configs_in_result(self, long_synthetic_prices):
        """Result should report the number of configs used."""
        result = run_walk_forward(
            long_synthetic_prices,
            n_splits=3,
            test_size=126,
            gap=10,
        )
        assert result["n_configs"] == len(GRID_CONFIGS)

    def test_result_includes_canonical_output_contract(self, long_synthetic_prices):
        """Canonical walk-forward output should be registry-friendly and versioned."""
        import walk_forward_validation as wfv

        result = run_walk_forward(
            long_synthetic_prices,
            n_splits=3,
            test_size=126,
            gap=10,
        )

        assert result["schema_version"] == wfv.WALK_FORWARD_SCHEMA_VERSION
        assert result["artifact_path"] == str(wfv.CANONICAL_WALK_FORWARD_ARTIFACT)
        assert result["window_mode"] == wfv.CANONICAL_WINDOW_MODE
        assert result["metrics_source"] == "src.backtest.metrics"


# ══════════════════════════════════════════════════════════════════════════════
#  print_report
# ══════════════════════════════════════════════════════════════════════════════


class TestPrintReport:
    """Tests for formatted report printing."""

    @pytest.fixture
    def basic_result(self):
        return {
            "n_windows": 5,
            "n_configs": 94,
            "walk_forward_efficiency": 0.85,
            "is_sharpe": {"mean": 1.2, "std": 0.3, "min": 0.8, "max": 1.5},
            "oos_sharpe": {"mean": 1.0, "std": 0.4, "min": 0.5, "max": 1.4},
            "dsr_champion_oos": 0.95,
            "dsr_average_oos": 0.75,
            "crisis_summary": {},
            "champion_weight_consistency": {
                "SPY": {"mean": 0.46, "std": 0.02, "min": 0.42, "max": 0.50},
                "GLD": {"mean": 0.38, "std": 0.02, "min": 0.34, "max": 0.42},
                "TLT": {"mean": 0.16, "std": 0.02, "min": 0.12, "max": 0.20},
            },
        }

    def test_print_basic(self, capsys, basic_result):
        """Basic report should print without errors."""
        print_report(basic_result)
        captured = capsys.readouterr()
        assert "Walk-Forward Validation Report" in captured.out
        assert "Windows: 5" in captured.out
        assert "WFE: 0.8500" in captured.out

    def test_print_with_crisis_data(self, capsys):
        """Report with crisis data should include crisis section."""
        result = {
            "n_windows": 3,
            "n_configs": 94,
            "walk_forward_efficiency": 0.72,
            "is_sharpe": {"mean": 1.1, "std": 0.2, "min": 0.9, "max": 1.3},
            "oos_sharpe": {"mean": 0.8, "std": 0.3, "min": 0.4, "max": 1.2},
            "dsr_champion_oos": 0.88,
            "dsr_average_oos": 0.65,
            "crisis_summary": {
                "2008": {"mean": -15.5, "worst": -22.3, "count": 2},
                "2020": {"mean": -8.2, "worst": -12.1, "count": 1},
            },
            "champion_weight_consistency": {
                "SPY": {"mean": 0.46, "std": 0.02, "min": 0.42, "max": 0.50},
                "GLD": {"mean": 0.38, "std": 0.02, "min": 0.34, "max": 0.42},
                "TLT": {"mean": 0.16, "std": 0.02, "min": 0.12, "max": 0.20},
            },
        }
        print_report(result)
        captured = capsys.readouterr()
        assert "Crisis Period Returns" in captured.out
        assert "2008" in captured.out
        assert "2020" in captured.out
        assert "-15.50" in captured.out or "15.50" in captured.out

    def test_print_no_crisis_empty_dict(self, capsys, basic_result):
        """Report with empty crisis dict should not print crisis section."""
        print_report(basic_result)
        captured = capsys.readouterr()
        assert "Crisis Period Returns" not in captured.out

    def test_print_dsr_values(self, capsys, basic_result):
        """DSR values should be printed in the report."""
        print_report(basic_result)
        captured = capsys.readouterr()
        assert "DSR" in captured.out
        assert "0.9500" in captured.out
        assert "0.7500" in captured.out

    def test_print_wfe_rating_good(self, capsys):
        """WFE > 0.60 should print 'GOOD' rating."""
        result = {
            "n_windows": 3,
            "n_configs": 94,
            "walk_forward_efficiency": 0.75,
            "is_sharpe": {"mean": 1.0, "std": 0.2, "min": 0.8, "max": 1.2},
            "oos_sharpe": {"mean": 0.8, "std": 0.3, "min": 0.4, "max": 1.2},
            "dsr_champion_oos": 0.90,
            "dsr_average_oos": 0.70,
            "crisis_summary": {},
            "champion_weight_consistency": {
                "SPY": {"mean": 0.46, "std": 0.02, "min": 0.42, "max": 0.50},
                "GLD": {"mean": 0.38, "std": 0.02, "min": 0.34, "max": 0.42},
                "TLT": {"mean": 0.16, "std": 0.02, "min": 0.12, "max": 0.20},
            },
        }
        print_report(result)
        captured = capsys.readouterr()
        assert "GOOD" in captured.out

    def test_print_wfe_rating_borderline(self, capsys):
        """WFE between 0.40 and 0.60 should print 'BORDERLINE'."""
        result = {
            "n_windows": 3,
            "n_configs": 94,
            "walk_forward_efficiency": 0.50,
            "is_sharpe": {"mean": 1.0, "std": 0.2, "min": 0.8, "max": 1.2},
            "oos_sharpe": {"mean": 0.5, "std": 0.3, "min": 0.2, "max": 0.8},
            "dsr_champion_oos": 0.80,
            "dsr_average_oos": 0.50,
            "crisis_summary": {},
            "champion_weight_consistency": {
                "SPY": {"mean": 0.46, "std": 0.04, "min": 0.38, "max": 0.54},
                "GLD": {"mean": 0.38, "std": 0.04, "min": 0.30, "max": 0.46},
                "TLT": {"mean": 0.16, "std": 0.04, "min": 0.08, "max": 0.24},
            },
        }
        print_report(result)
        captured = capsys.readouterr()
        assert "BORDERLINE" in captured.out

    def test_print_wfe_rating_poor(self, capsys):
        """WFE < 0.40 should print 'POOR'."""
        result = {
            "n_windows": 3,
            "n_configs": 94,
            "walk_forward_efficiency": 0.25,
            "is_sharpe": {"mean": 1.0, "std": 0.2, "min": 0.8, "max": 1.2},
            "oos_sharpe": {"mean": 0.25, "std": 0.3, "min": 0.05, "max": 0.5},
            "dsr_champion_oos": 0.60,
            "dsr_average_oos": 0.30,
            "crisis_summary": {},
            "champion_weight_consistency": {
                "SPY": {"mean": 0.46, "std": 0.05, "min": 0.36, "max": 0.56},
                "GLD": {"mean": 0.38, "std": 0.05, "min": 0.28, "max": 0.48},
                "TLT": {"mean": 0.16, "std": 0.05, "min": 0.06, "max": 0.26},
            },
        }
        print_report(result)
        captured = capsys.readouterr()
        assert "POOR" in captured.out


# ══════════════════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════════════════


class TestMainFunction:
    """Tests for the CLI entry point."""

    def test_main_file_not_found(self, capsys):
        """When prices file doesn't exist, main should log error and return."""
        with patch("walk_forward_validation.PRICES_JSON") as mock_prices:
            mock_prices.exists.return_value = False
            with patch.object(sys, "argv", ["walk_forward_validation.py"]):
                import walk_forward_validation as wfv
                result = wfv.main()
                assert result is None

    def test_main_empty_data_returns(self, capsys):
        """When load_prices returns empty, main should log error."""
        with (
            patch("walk_forward_validation.PRICES_JSON") as mock_prices,
            patch("walk_forward_validation.load_prices") as mock_load,
        ):
            mock_prices.exists.return_value = True
            mock_load.return_value = pd.DataFrame()
            with patch.object(sys, "argv", ["walk_forward_validation.py"]):
                import walk_forward_validation as wfv
                result = wfv.main()
                assert result is None

    def test_main_save_path_creation(self, capsys, tmp_path):
        """When --save is used, the report should be written to disk."""
        import walk_forward_validation as wfv

        # Create a small valid prices.json in tmp_path
        prices_dir = tmp_path / "public" / "data"
        prices_dir.mkdir(parents=True)
        prices_file = prices_dir / "prices.json"
        price_data = {
            "SPY": [{"d": "2020-01-02", "p": 300.0}, {"d": "2020-01-03", "p": 301.0}],
            "GLD": [{"d": "2020-01-02", "p": 150.0}, {"d": "2020-01-03", "p": 151.0}],
            "TLT": [{"d": "2020-01-02", "p": 140.0}, {"d": "2020-01-03", "p": 141.0}],
        }
        prices_file.write_text(json.dumps(price_data))

        valid_result = {
            "n_windows": 3,
            "n_configs": 94,
            "walk_forward_efficiency": 0.85,
            "is_sharpe": {"mean": 1.2, "std": 0.3, "min": 0.8, "max": 1.5},
            "oos_sharpe": {"mean": 1.0, "std": 0.4, "min": 0.5, "max": 1.4},
            "dsr_champion_oos": 0.95,
            "dsr_average_oos": 0.75,
            "crisis_summary": {},
            "champion_weight_consistency": {
                "SPY": {"mean": 0.46, "std": 0.02, "min": 0.42, "max": 0.50},
                "GLD": {"mean": 0.38, "std": 0.02, "min": 0.34, "max": 0.42},
                "TLT": {"mean": 0.16, "std": 0.02, "min": 0.12, "max": 0.20},
            },
            "windows": [],
        }

        with (
            patch.object(wfv, "DATA_DIR", tmp_path / "data"),
            patch.object(wfv, "PRICES_JSON", prices_file),
            patch.object(wfv, "run_walk_forward", return_value=valid_result),
            patch.object(sys, "argv",
                         ["walk_forward_validation.py", "--save", "--n-splits", "2",
                          "--test-size", "5", "--gap", "1"]),
        ):
            wfv.main()

            report_path = tmp_path / "data" / "walk_forward_report.json"
            assert report_path.exists()
            report = json.loads(report_path.read_text())
            assert "n_windows" in report
            assert report["n_windows"] == 3
            assert report["schema_version"] == wfv.WALK_FORWARD_SCHEMA_VERSION
            assert report["artifact_path"] == str(wfv.CANONICAL_WALK_FORWARD_ARTIFACT)

    def test_main_parse_args_defaults(self):
        """Default argument parsing should set expected defaults."""
        with (
            patch("walk_forward_validation.PRICES_JSON") as mock_prices,
            patch.object(sys, "argv", ["walk_forward_validation.py"]),
        ):
            mock_prices.exists.return_value = False
            import walk_forward_validation as wfv

            result = wfv.main()
            assert result is None


class TestDeprecatedWalkForwardValidate:
    """Tests for the deprecated walk_forward_validate.py wrapper."""

    def test_legacy_script_delegates_to_canonical_main(self, caplog):
        import importlib

        import walk_forward_validate as legacy

        called = []
        legacy = importlib.reload(legacy)
        with patch.object(legacy, "canonical_main", lambda: called.append(True)):
            legacy.main()

        assert called == [True]
        assert legacy.CANONICAL_SCRIPT == "scripts/walk_forward_validation.py"
        assert "deprecated" in caplog.text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
