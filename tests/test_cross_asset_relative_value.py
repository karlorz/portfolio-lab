#!/usr/bin/env python3
"""
Tests for Cross-Asset Relative Value Scanner (v5.71).
"""

import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock


import pytest

from src.signals.cross_asset_relative_value import (
    CrossAssetRVScanner,
    CrossAssetRVSignal,
    PairReading,
    CROSS_ASSET_PAIRS,
    ZSCORE_ENTRY,
    ZSCORE_EXIT,
    LOOKBACK,
    print_scan,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices(tmp_path):
    """Create a realistic price JSON file for testing."""
    import string as s
    dates = []
    base = datetime(2026, 1, 1)
    for i in range(200):
        d = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        # Only include weekdays
        dt = base + timedelta(days=i)
        if dt.weekday() < 5:
            dates.append(d)

    prices = {
        "SPY": [{"d": d, "p": 500.0 * (1 + np.random.normal(0.0005, 0.01))}
                for d in dates],
        "QQQ": [{"d": d, "p": 450.0 * (1 + np.random.normal(0.0006, 0.012))}
                for d in dates],
        "EFA": [{"d": d, "p": 80.0 * (1 + np.random.normal(0.0003, 0.008))}
                for d in dates],
        "GLD": [{"d": d, "p": 180.0 * (1 + np.random.normal(0.0002, 0.007))}
                for d in dates],
        "BTC": [{"d": d, "p": 40000.0 * (1 + np.random.normal(0.001, 0.03))}
                for d in dates],
        "TLT": [{"d": d, "p": 95.0 * (1 + np.random.normal(0.0001, 0.006))}
                for d in dates],
        "IEF": [{"d": d, "p": 105.0 * (1 + np.random.normal(0.0001, 0.004))}
                for d in dates],
    }

    # Create diverging SPY trend for higher z-score
    for i, d in enumerate(dates):
        if i >= 150:  # Last 50 days, make SPY diverge from QQQ
            prices["SPY"][i]["p"] *= 1.005  # SPY rallying
            prices["QQQ"][i]["p"] *= 0.998  # QQQ lagging

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    prices_path = data_dir / "prices.json"
    with open(prices_path, "w") as f:
        json.dump(prices, f)
    return data_dir


@pytest.fixture
def scanner(sample_prices):
    return CrossAssetRVScanner(data_dir=sample_prices)


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_load_price_data(self, scanner):
        assert scanner._load_price_data() is True
        assert len(scanner.prices) >= 5
        assert "SPY" in scanner.prices
        assert len(scanner.dates) > 0

    def test_load_price_data_missing(self, tmp_path):
        empty_dir = tmp_path / "nope"
        empty_dir.mkdir()
        scanner = CrossAssetRVScanner(data_dir=empty_dir)
        # Patch to prevent fallback to project prices.json
        with patch.object(
            scanner, '_load_price_data',
            return_value=False
        ):
            assert scanner._load_price_data() is False
        assert len(scanner.prices) == 0


# ---------------------------------------------------------------------------
# Computation tests
# ---------------------------------------------------------------------------

class TestComputations:
    def test_compute_returns(self, scanner):
        prices = np.array([100.0, 101.0, 102.0, 99.0, 98.0])
        rets = scanner._compute_returns(prices, period=2)
        assert len(rets) == 5
        assert np.isnan(rets[0])
        assert np.isnan(rets[1])
        assert not np.isnan(rets[2])
        # rets[2] = 102/100 - 1 = 0.02
        assert rets[2] == pytest.approx(0.02, abs=1e-5)
        # rets[4] = 98/102 - 1 = -0.0392157...
        assert rets[4] == pytest.approx(98.0 / 102.0 - 1, abs=1e-5)

    def test_compute_returns_insufficient(self, scanner):
        prices = np.array([100.0, 101.0])  # Only 2 data points
        rets = scanner._compute_returns(prices, period=5)
        assert np.all(np.isnan(rets))

    def test_compute_z_score(self, scanner):
        # Create a series with enough points (>= window + MIN_HISTORY)
        np.random.seed(42)
        n = 60
        values = np.random.normal(0, 1, n).cumsum()
        values[-1] = 15.0  # Extreme final value
        z_scores, means, stds = scanner._compute_z_score(values, window=30)
        assert len(z_scores) == n
        # First 30 should be nan
        assert np.all(np.isnan(z_scores[:30]))
        # Last value should be extreme
        assert z_scores[-1] > 2.0, f"Expected extreme z-score, got {z_scores[-1]}"

    def test_compute_z_score_constant(self, scanner):
        values = np.ones(50)
        z_scores, means, stds = scanner._compute_z_score(values, window=10)
        assert np.all(np.isnan(z_scores))  # std is 0

    def test_compute_z_score_nan_handling(self, scanner):
        values = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        z_scores, means, stds = scanner._compute_z_score(values, window=5)
        assert len(z_scores) == 10


# ---------------------------------------------------------------------------
# Pair scanning tests
# ---------------------------------------------------------------------------

class TestPairScanning:
    def test_scan_all_pairs(self, scanner):
        scanner._load_price_data()
        signal = scanner.scan_all()
        assert isinstance(signal, CrossAssetRVSignal)
        assert len(signal.pairs) == len(CROSS_ASSET_PAIRS)

    def test_scan_specific_pair(self, scanner):
        scanner._load_price_data()
        reading = scanner.scan_pair("spy_qqq")
        assert reading is not None
        assert reading.pair_name == "spy_qqq"
        assert reading.symbol_a == "SPY"
        assert reading.symbol_b == "QQQ"
        assert isinstance(reading.z_score, float)
        assert isinstance(reading.signal_value, float)

    def test_scan_unknown_pair(self, scanner):
        reading = scanner.scan_pair("unknown")
        assert reading is None

    def test_pair_has_returns(self, scanner):
        scanner._load_price_data()
        reading = scanner.scan_pair("spy_gld")
        assert reading is not None
        assert isinstance(reading.return_a_60d, float)
        assert isinstance(reading.return_b_60d, float)

    def test_pair_regime_detection(self, scanner):
        scanner._load_price_data()
        reading = scanner.scan_pair("tlt_ief")
        assert reading is not None
        assert reading.regime in ("diverged_bull", "diverged_bear", "converged", "neutral")
        assert 0.0 <= reading.conviction <= 1.0

    def test_scan_all_no_data(self, tmp_path):
        scanner = CrossAssetRVScanner(data_dir=tmp_path / "empty2")
        with patch.object(scanner, '_load_price_data', return_value=False):
            signal = scanner.scan_all()
            assert signal.total_pairs == len(CROSS_ASSET_PAIRS)
            assert len(signal.pairs) == 0


# ---------------------------------------------------------------------------
# Signal generation tests
# ---------------------------------------------------------------------------

class TestSignalGeneration:
    def test_get_ensemble_signal(self, scanner):
        scanner._load_price_data()
        es = scanner.get_ensemble_signal()
        assert isinstance(es, dict)
        assert "signal_value" in es
        assert "confidence" in es
        assert "asset_signals" in es
        assert "SPY" in es["asset_signals"]
        assert "GLD" in es["asset_signals"]
        assert "TLT" in es["asset_signals"]
        assert -1.0 <= es["signal_value"] <= 1.0

    def test_get_ensemble_signal_no_data(self, tmp_path):
        scanner = CrossAssetRVScanner(data_dir=tmp_path / "nodata2")
        with patch.object(scanner, '_load_price_data', return_value=False):
            es = scanner.get_ensemble_signal()
            assert es["signal_value"] == 0.0
            assert es["confidence"] == 0.0

    def test_ensemble_signal_bounds(self, scanner):
        scanner._load_price_data()
        es = scanner.get_ensemble_signal()
        for asset, bias in es["asset_signals"].items():
            assert -1.0 <= bias <= 1.0, f"{asset} bias {bias} out of bounds"
        assert 0.0 <= es["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# State persistence tests
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_load_save_state(self, scanner):
        state = {"spy_qqq": {"active": True, "days_active": 5, "entry_zscore": 2.5}}
        scanner._save_state(state)
        loaded = scanner._load_state()
        assert loaded["spy_qqq"]["active"] is True
        assert loaded["spy_qqq"]["days_active"] == 5

    def test_load_no_state(self, tmp_path):
        scanner = CrossAssetRVScanner(data_dir=tmp_path / "f")
        state = scanner._load_state()
        assert state == {}

    def test_state_updates_on_scan(self, scanner):
        scanner._load_price_data()
        reading = scanner.scan_pair("spy_qqq")
        state = scanner._load_state()
        assert "spy_qqq" in state
        assert "last_zscore" in state["spy_qqq"]
        assert "last_scan" in state["spy_qqq"]


# ---------------------------------------------------------------------------
# CROSS_ASSET_PAIRS config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_all_pairs_defined(self):
        assert len(CROSS_ASSET_PAIRS) == 5
        expected = {"spy_qqq", "spy_efa", "gld_btc", "tlt_ief", "spy_gld"}
        assert set(CROSS_ASSET_PAIRS.keys()) == expected

    def test_pair_structure(self):
        for name, (a, b, desc) in CROSS_ASSET_PAIRS.items():
            assert isinstance(a, str) and len(a) > 0
            assert isinstance(b, str) and len(b) > 0
            assert isinstance(desc, str) and len(desc) > 0

    def test_thresholds(self):
        assert ZSCORE_ENTRY == 2.0
        assert ZSCORE_EXIT == 0.5


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestDataClasses:
    def test_pair_reading_to_dict(self):
        reading = PairReading(
            pair_name="spy_qqq", symbol_a="SPY", symbol_b="QQQ",
            return_a_60d=5.0, return_b_60d=2.0, return_differential=3.0,
            z_score=2.5, z_score_mean=0.0, z_score_std=1.0,
            signal_value=-0.75, regime="diverged_bull", conviction=0.8,
            active=True, days_active=3, entry_zscore=2.5,
        )
        d = reading.to_dict()
        assert d["pair_name"] == "spy_qqq"
        assert d["z_score"] == 2.5
        assert d["conviction"] == 0.8

    def test_signal_to_dict(self):
        reading = PairReading(
            pair_name="spy_qqq", symbol_a="SPY", symbol_b="QQQ",
            return_a_60d=5.0, return_b_60d=2.0, return_differential=3.0,
            z_score=2.5, z_score_mean=0.0, z_score_std=1.0,
            signal_value=-0.75, regime="diverged_bull", conviction=0.8,
            active=True, days_active=3, entry_zscore=2.5,
        )
        signal = CrossAssetRVSignal(
            timestamp="2026-01-01",
            pairs={"spy_qqq": reading},
            avg_z_score=2.5,
            max_divergence=2.5,
            num_diverged=1,
            total_pairs=5,
            risk_on_score=-0.5,
            duration_score=0.0,
            overall_conviction=0.8,
        )
        d = signal.to_dict()
        assert d["num_diverged"] == 1
        assert "pairs" in d
        assert d["pairs"]["spy_qqq"]["z_score"] == 2.5


# ---------------------------------------------------------------------------
# Print tests (smoke tests)
# ---------------------------------------------------------------------------

class TestPrint:
    def test_print_scan_empty(self, capsys):
        signal = CrossAssetRVSignal(
            timestamp="2026-01-01",
            pairs={},
            avg_z_score=0.0,
            max_divergence=0.0,
            num_diverged=0,
            total_pairs=5,
            risk_on_score=0.0,
            duration_score=0.0,
            overall_conviction=0.0,
        )
        print_scan(signal)
        captured = capsys.readouterr()
        assert "CROSS-ASSET RELATIVE VALUE SCAN" in captured.out

    def test_print_scan_with_data(self, capsys):
        reading = PairReading(
            pair_name="spy_qqq", symbol_a="SPY", symbol_b="QQQ",
            return_a_60d=5.0, return_b_60d=2.0, return_differential=3.0,
            z_score=2.5, z_score_mean=0.0, z_score_std=1.0,
            signal_value=-0.75, regime="diverged_bull", conviction=0.8,
            active=True, days_active=3, entry_zscore=2.5,
        )
        signal = CrossAssetRVSignal(
            timestamp="2026-01-01",
            pairs={"spy_qqq": reading},
            avg_z_score=2.5,
            max_divergence=2.5,
            num_diverged=1,
            total_pairs=5,
            risk_on_score=-0.5,
            duration_score=0.0,
            overall_conviction=0.8,
        )
        print_scan(signal)
        captured = capsys.readouterr()
        assert "spy_qqq" in captured.out
        assert "diverged_bull" in captured.out
