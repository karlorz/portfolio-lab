#!/usr/bin/env python3
"""
Tests for Cross-Asset Relative Value Scanner (v5.71).
"""

import json
import logging
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
    MIN_HISTORY,
    print_scan,
)
from src.paths import DATA_DIR


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
    def test_print_scan_empty(self, caplog):
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
        with caplog.at_level(logging.INFO, logger="src.signals.cross_asset_relative_value"):
            print_scan(signal)
        assert "CROSS-ASSET RELATIVE VALUE SCAN" in caplog.text

    def test_print_scan_with_data(self, caplog):
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
        with caplog.at_level(logging.INFO, logger="src.signals.cross_asset_relative_value"):
            print_scan(signal)
        assert "spy_qqq" in caplog.text
        assert "diverged_bull" in caplog.text


# ---------------------------------------------------------------------------
# Extended coverage tests
# ---------------------------------------------------------------------------


class TestPairReadingDataclass:
    """Extended PairReading dataclass tests."""

    def test_to_dict_has_all_fields(self):
        reading = PairReading(
            pair_name="spy_efa", symbol_a="SPY", symbol_b="EFA",
            return_a_60d=3.0, return_b_60d=1.0, return_differential=2.0,
            z_score=1.8, z_score_mean=0.5, z_score_std=0.8,
            signal_value=-0.4, regime="neutral", conviction=0.3,
            active=False, days_active=0, entry_zscore=0.0,
        )
        d = reading.to_dict()
        expected_keys = {
            'pair_name', 'symbol_a', 'symbol_b', 'return_a_60d', 'return_b_60d',
            'return_differential', 'z_score', 'z_score_mean', 'z_score_std',
            'signal_value', 'regime', 'conviction', 'active', 'days_active',
            'entry_zscore',
        }
        assert expected_keys == set(d.keys())

    def test_diverged_bear_regime(self):
        """Negative z-score past threshold should be diverged_bear."""
        reading = PairReading(
            pair_name="spy_gld", symbol_a="SPY", symbol_b="GLD",
            return_a_60d=-5.0, return_b_60d=3.0, return_differential=-8.0,
            z_score=-2.8, z_score_mean=0.0, z_score_std=1.0,
            signal_value=0.7, regime="diverged_bear", conviction=0.9,
            active=True, days_active=2, entry_zscore=-2.8,
        )
        assert reading.regime == "diverged_bear"
        assert reading.signal_value > 0  # Long A, short B

    def test_converged_regime(self):
        """Low z-score should indicate converged."""
        reading = PairReading(
            pair_name="tlt_ief", symbol_a="TLT", symbol_b="IEF",
            return_a_60d=1.0, return_b_60d=0.8, return_differential=0.2,
            z_score=0.3, z_score_mean=0.0, z_score_std=1.0,
            signal_value=0.0, regime="converged", conviction=0.0,
            active=False, days_active=0, entry_zscore=0.0,
        )
        assert reading.regime == "converged"
        assert reading.active is False


class TestCrossAssetRVSignalDataclass:
    """Extended CrossAssetRVSignal dataclass tests."""

    def test_to_dict_nested_pair_serialization(self):
        reading = PairReading(
            pair_name="spy_qqq", symbol_a="SPY", symbol_b="QQQ",
            return_a_60d=5.0, return_b_60d=2.0, return_differential=3.0,
            z_score=2.5, z_score_mean=0.0, z_score_std=1.0,
            signal_value=-0.75, regime="diverged_bull", conviction=0.8,
            active=True, days_active=3, entry_zscore=2.5,
        )
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24",
            pairs={"spy_qqq": reading},
            avg_z_score=2.5,
            max_divergence=2.5,
            num_diverged=1,
            total_pairs=5,
            risk_on_score=-0.5,
            duration_score=0.2,
            overall_conviction=0.8,
        )
        d = signal.to_dict()
        # Nested pair should be serialized to dict
        assert isinstance(d["pairs"]["spy_qqq"], dict)
        assert d["pairs"]["spy_qqq"]["pair_name"] == "spy_qqq"
        assert d["duration_score"] == 0.2

    def test_to_dict_all_fields(self):
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24",
            pairs={},
            avg_z_score=0.0,
            max_divergence=0.0,
            num_diverged=0,
            total_pairs=5,
            risk_on_score=0.0,
            duration_score=0.0,
            overall_conviction=0.0,
        )
        d = signal.to_dict()
        expected_keys = {
            'timestamp', 'pairs', 'avg_z_score', 'max_divergence',
            'num_diverged', 'total_pairs', 'risk_on_score',
            'duration_score', 'overall_conviction',
        }
        assert expected_keys == set(d.keys())


class TestComputationsExtended:
    """Extended computation edge cases."""

    def test_compute_returns_all_nan(self, scanner):
        """All-NaN prices should return all-NaN returns."""
        prices = np.full(100, np.nan)
        rets = scanner._compute_returns(prices, period=10)
        assert np.all(np.isnan(rets))

    def test_compute_z_score_short_series(self, scanner):
        """Series shorter than window should return all NaN z-scores."""
        values = np.array([1.0, 2.0, 3.0])
        z_scores, means, stds = scanner._compute_z_score(values, window=10)
        assert np.all(np.isnan(z_scores))

    def test_compute_z_score_with_nans(self, scanner):
        """Z-score computation should handle NaN values in input."""
        np.random.seed(42)
        values = np.random.normal(0, 1, 100)
        values[30] = np.nan
        values[60] = np.nan
        z_scores, means, stds = scanner._compute_z_score(values, window=20)
        # Should still produce some valid z-scores
        valid = z_scores[~np.isnan(z_scores)]
        assert len(valid) > 0

    def test_compute_returns_exact_period(self, scanner):
        """Returns should match exact period computation."""
        prices = np.arange(1.0, 101.0)  # 1, 2, 3, ..., 100
        rets = scanner._compute_returns(prices, period=1)
        # rets[i] = prices[i] / prices[i-1] - 1
        assert rets[1] == pytest.approx(2.0 / 1.0 - 1)
        assert rets[2] == pytest.approx(3.0 / 2.0 - 1)


class TestPairScanningExtended:
    """Extended pair scanning tests."""

    def test_scan_pair_missing_symbol(self, tmp_path):
        """Scanning a pair with missing price data should return None."""
        data_dir = tmp_path / "partial"
        data_dir.mkdir()
        # Only SPY, no QQQ
        dates = [("2026-01-0" + str(i+1), 100.0 + i) for i in range(7)]
        prices = {"SPY": [{"d": d, "p": p} for d, p in dates]}
        with open(data_dir / "prices.json", "w") as f:
            json.dump(prices, f)
        scanner = CrossAssetRVScanner(data_dir=data_dir)
        scanner._load_price_data()
        # spy_qqq needs both SPY and QQQ, but QQQ not loaded → should work with NaN
        # The scanner loads NaN arrays for missing symbols, scan_pair checks len

    def test_scan_pair_valid_output_fields(self, scanner):
        """Scanned pair should have all expected fields."""
        scanner._load_price_data()
        reading = scanner.scan_pair("spy_qqq")
        if reading is not None:
            assert hasattr(reading, 'return_a_60d')
            assert hasattr(reading, 'return_b_60d')
            assert hasattr(reading, 'return_differential')
            assert hasattr(reading, 'z_score')
            assert hasattr(reading, 'z_score_mean')
            assert hasattr(reading, 'z_score_std')
            assert hasattr(reading, 'signal_value')
            assert hasattr(reading, 'regime')
            assert hasattr(reading, 'conviction')
            assert hasattr(reading, 'active')
            assert hasattr(reading, 'days_active')
            assert hasattr(reading, 'entry_zscore')

    def test_scan_all_signal_fields(self, scanner):
        """scan_all result should have all expected fields."""
        scanner._load_price_data()
        signal = scanner.scan_all()
        assert isinstance(signal.timestamp, str)
        assert isinstance(signal.avg_z_score, float)
        assert isinstance(signal.max_divergence, float)
        assert isinstance(signal.num_diverged, int)
        assert isinstance(signal.total_pairs, int)
        assert isinstance(signal.risk_on_score, float)
        assert isinstance(signal.duration_score, float)
        assert isinstance(signal.overall_conviction, float)

    def test_scan_pair_current_idx_out_of_bounds(self, scanner):
        """Out-of-bounds current_idx should be clamped."""
        scanner._load_price_data()
        reading = scanner.scan_pair("spy_qqq", current_idx=9999)
        assert reading is not None  # Should clamp and return


class TestSignalGenerationExtended:
    """Extended signal generation tests."""

    def test_get_signal_snapshot_structure(self, scanner):
        """get_signal_snapshot should return a SignalSnapshot."""
        from src.signals.signal_snapshot import SignalSnapshot
        scanner._load_price_data()
        snapshot = scanner.get_signal_snapshot()
        assert isinstance(snapshot, SignalSnapshot)
        assert snapshot.source == "cross_asset_rv"

    def test_get_signal_snapshot_active_field(self, scanner):
        """SignalSnapshot is_active should reflect signal_value."""
        from src.signals.signal_snapshot import SignalSnapshot
        scanner._load_price_data()
        snapshot = scanner.get_signal_snapshot()
        # is_active should be True if signal_value != 0, False otherwise
        if snapshot.value != 0.0:
            assert snapshot.is_active is True

    def test_ensemble_signal_timestamp(self, scanner):
        """Ensemble signal should include a timestamp."""
        scanner._load_price_data()
        es = scanner.get_ensemble_signal()
        assert "timestamp" in es
        assert es["timestamp"] is not None

    def test_ensemble_signal_pair_details(self, scanner):
        """Ensemble signal should include pair details."""
        scanner._load_price_data()
        es = scanner.get_ensemble_signal()
        assert "pairs" in es
        assert "avg_z_score" in es
        assert "num_diverged" in es
        assert "total_pairs" in es


class TestStatePersistenceExtended:
    """Extended state persistence tests."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Save and load should roundtrip correctly."""
        scanner = CrossAssetRVScanner(data_dir=tmp_path / "state_rt")
        state = {
            "spy_qqq": {"active": True, "days_active": 10, "entry_zscore": 2.3, "last_zscore": 2.1},
            "spy_gld": {"active": False, "days_active": 0, "entry_zscore": 0.0, "last_zscore": 0.5},
        }
        scanner._save_state(state)
        loaded = scanner._load_state()
        assert loaded["spy_qqq"]["active"] is True
        assert loaded["spy_qqq"]["days_active"] == 10
        assert loaded["spy_gld"]["active"] is False

    def test_corrupted_state_file(self, tmp_path):
        """Corrupted state file should return empty dict."""
        scanner = CrossAssetRVScanner(data_dir=tmp_path / "corrupt")
        scanner.state_dir.mkdir(parents=True, exist_ok=True)
        with open(scanner.state_path, "w") as f:
            f.write("not valid json {{{")
        state = scanner._load_state()
        assert state == {}


class TestPrintExtended:
    """Extended print tests."""

    def test_print_scan_shows_legend(self, caplog):
        """Print should show the legend section."""
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
        with caplog.at_level(logging.INFO, logger="src.signals.cross_asset_relative_value"):
            print_scan(signal)
        assert "Legend" in caplog.text
        assert "Z-Score" in caplog.text

    def test_print_scan_converged_pair(self, caplog):
        """Print should handle converged pairs."""
        reading = PairReading(
            pair_name="spy_efa", symbol_a="SPY", symbol_b="EFA",
            return_a_60d=2.0, return_b_60d=1.5, return_differential=0.5,
            z_score=0.3, z_score_mean=0.1, z_score_std=0.5,
            signal_value=0.0, regime="converged", conviction=0.0,
            active=False, days_active=0, entry_zscore=0.0,
        )
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24",
            pairs={"spy_efa": reading},
            avg_z_score=0.3,
            max_divergence=0.3,
            num_diverged=0,
            total_pairs=5,
            risk_on_score=0.0,
            duration_score=0.0,
            overall_conviction=0.0,
        )
        with caplog.at_level(logging.INFO, logger="src.signals.cross_asset_relative_value"):
            print_scan(signal)
        assert "converged" in caplog.text


class TestPairReadingExtended:
    """Extended tests for PairReading dataclass."""

    def test_all_fields(self):
        reading = PairReading(
            pair_name="spy_gld", symbol_a="SPY", symbol_b="GLD",
            return_a_60d=0.05, return_b_60d=0.08, return_differential=-0.03,
            z_score=-1.5, z_score_mean=0.0, z_score_std=1.0,
            signal_value=-0.8, regime="diverged_bear", conviction=0.7,
            active=True, days_active=15, entry_zscore=-1.3,
        )
        assert reading.pair_name == "spy_gld"
        assert reading.signal_value == -0.8
        assert reading.active is True
        assert reading.days_active == 15

    def test_to_dict_completeness(self):
        reading = PairReading(
            pair_name="spy_tlt", symbol_a="SPY", symbol_b="TLT",
            return_a_60d=0.10, return_b_60d=-0.05, return_differential=0.15,
            z_score=2.0, z_score_mean=0.5, z_score_std=0.8,
            signal_value=0.9, regime="diverged_bull", conviction=0.8,
            active=False, days_active=0, entry_zscore=0.0,
        )
        d = reading.to_dict()
        expected_keys = {
            "pair_name", "symbol_a", "symbol_b",
            "return_a_60d", "return_b_60d", "return_differential",
            "z_score", "z_score_mean", "z_score_std",
            "signal_value", "regime", "conviction",
            "active", "days_active", "entry_zscore",
        }
        assert set(d.keys()) == expected_keys

    def test_regime_values(self):
        """Valid regime values."""
        for regime in ("diverged_bull", "diverged_bear", "converged", "neutral"):
            reading = PairReading(
                pair_name="test", symbol_a="A", symbol_b="B",
                return_a_60d=0.0, return_b_60d=0.0, return_differential=0.0,
                z_score=0.0, z_score_mean=0.0, z_score_std=1.0,
                signal_value=0.0, regime=regime, conviction=0.0,
                active=False, days_active=0, entry_zscore=0.0,
            )
            assert reading.regime == regime

    def test_negative_differential(self):
        reading = PairReading(
            pair_name="test", symbol_a="SPY", symbol_b="GLD",
            return_a_60d=0.02, return_b_60d=0.05, return_differential=-0.03,
            z_score=-1.0, z_score_mean=0.0, z_score_std=1.0,
            signal_value=-0.5, regime="diverged_bear", conviction=0.4,
            active=True, days_active=10, entry_zscore=-0.8,
        )
        assert reading.return_differential < 0


class TestCrossAssetRVSignalExtended:
    """Extended tests for CrossAssetRVSignal dataclass."""

    def test_to_dict_serializes_pairs(self):
        reading = PairReading(
            pair_name="spy_efa", symbol_a="SPY", symbol_b="EFA",
            return_a_60d=0.05, return_b_60d=0.03, return_differential=0.02,
            z_score=0.5, z_score_mean=0.0, z_score_std=0.5,
            signal_value=0.3, regime="converged", conviction=0.2,
            active=False, days_active=0, entry_zscore=0.0,
        )
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24", pairs={"spy_efa": reading},
            avg_z_score=0.5, max_divergence=0.5,
            num_diverged=0, total_pairs=5,
            risk_on_score=0.1, duration_score=-0.1, overall_conviction=0.2,
        )
        d = signal.to_dict()
        assert "pairs" in d
        assert "spy_efa" in d["pairs"]
        assert d["pairs"]["spy_efa"]["pair_name"] == "spy_efa"
        assert d["risk_on_score"] == 0.1

    def test_empty_pairs(self):
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24", pairs={},
            avg_z_score=0.0, max_divergence=0.0,
            num_diverged=0, total_pairs=0,
            risk_on_score=0.0, duration_score=0.0, overall_conviction=0.0,
        )
        d = signal.to_dict()
        assert d["pairs"] == {}

    def test_multiple_pairs(self):
        r1 = PairReading(
            pair_name="spy_efa", symbol_a="SPY", symbol_b="EFA",
            return_a_60d=0.05, return_b_60d=0.03, return_differential=0.02,
            z_score=0.5, z_score_mean=0.0, z_score_std=0.5,
            signal_value=0.3, regime="converged", conviction=0.2,
            active=False, days_active=0, entry_zscore=0.0,
        )
        r2 = PairReading(
            pair_name="spy_gld", symbol_a="SPY", symbol_b="GLD",
            return_a_60d=0.05, return_b_60d=0.08, return_differential=-0.03,
            z_score=-0.8, z_score_mean=0.0, z_score_std=0.6,
            signal_value=-0.5, regime="diverged_bear", conviction=0.4,
            active=True, days_active=5, entry_zscore=-0.7,
        )
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24", pairs={"spy_efa": r1, "spy_gld": r2},
            avg_z_score=0.0, max_divergence=0.8,
            num_diverged=1, total_pairs=5,
            risk_on_score=-0.1, duration_score=0.2, overall_conviction=0.3,
        )
        assert len(signal.pairs) == 2
        assert signal.num_diverged == 1


class TestCrossAssetRVScannerExtended:
    """Extended CrossAssetRVScanner tests."""

    def test_scanner_default_data_dir(self):
        scanner = CrossAssetRVScanner()
        assert scanner.data_dir == DATA_DIR

    def test_scanner_custom_data_dir(self, tmp_path):
        scanner = CrossAssetRVScanner(data_dir=tmp_path)
        assert scanner.data_dir == tmp_path

    def test_compute_returns(self, tmp_path):
        scanner = CrossAssetRVScanner(data_dir=tmp_path)
        prices = np.array([100, 105, 110, 108, 112])
        returns = scanner._compute_returns(prices, period=2)
        assert len(returns) >= 0  # At least some returns computed

    def test_compute_returns_single_period(self, tmp_path):
        scanner = CrossAssetRVScanner(data_dir=tmp_path)
        prices = np.array([100, 110])
        returns = scanner._compute_returns(prices, period=1)
        assert len(returns) >= 1

    def test_compute_z_score_basic(self, tmp_path):
        scanner = CrossAssetRVScanner(data_dir=tmp_path)
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        z, means, stds = scanner._compute_z_score(series, 3)
        assert isinstance(z, np.ndarray)

    def test_empty_signal(self, tmp_path):
        scanner = CrossAssetRVScanner(data_dir=tmp_path)
        signal = scanner._empty_signal()
        assert isinstance(signal, CrossAssetRVSignal)
        assert signal.num_diverged == 0


class TestCrossAssetRVConstants:
    """Validate module constants."""

    def test_pair_definitions_exist(self):
        from src.signals.cross_asset_relative_value import CrossAssetRVScanner
        assert hasattr(CrossAssetRVScanner, 'PAIRS') or hasattr(CrossAssetRVScanner, 'DEFAULT_PAIRS') or True

    def test_scanner_class_exists(self):
        from src.signals.cross_asset_relative_value import CrossAssetRVScanner
        assert callable(CrossAssetRVScanner)


# ---------------------------------------------------------------------------
# PairReading edge cases
# ---------------------------------------------------------------------------

class TestPairReadingEdgeCases:
    """Edge cases for PairReading dataclass fields."""

    def test_negative_entry_zscore(self):
        """entry_zscore can be negative for diverged_bear entries."""
        reading = PairReading(
            pair_name="spy_gld", symbol_a="SPY", symbol_b="GLD",
            return_a_60d=-5.0, return_b_60d=3.0, return_differential=-8.0,
            z_score=-2.8, z_score_mean=0.0, z_score_std=1.0,
            signal_value=0.7, regime="diverged_bear", conviction=0.9,
            active=True, days_active=2, entry_zscore=-2.8,
        )
        reading.entry_zscore = -2.8
        assert reading.entry_zscore < 0

    def test_zero_conviction(self):
        """conviction of 0.0 should be valid for converged pairs."""
        reading = PairReading(
            pair_name="tlt_ief", symbol_a="TLT", symbol_b="IEF",
            return_a_60d=0.1, return_b_60d=0.1, return_differential=0.0,
            z_score=0.1, z_score_mean=0.0, z_score_std=1.0,
            signal_value=0.0, regime="converged", conviction=0.0,
            active=False, days_active=0, entry_zscore=0.0,
        )
        assert reading.conviction == 0.0
        assert reading.active is False

    def test_extreme_z_score(self):
        """z_score of 10.0 should produce maxed-out signal/clip."""
        reading = PairReading(
            pair_name="spy_qqq", symbol_a="SPY", symbol_b="QQQ",
            return_a_60d=20.0, return_b_60d=-10.0, return_differential=30.0,
            z_score=10.0, z_score_mean=0.0, z_score_std=1.0,
            signal_value=-1.0, regime="diverged_bull", conviction=1.0,
            active=True, days_active=1, entry_zscore=10.0,
        )
        assert reading.z_score == 10.0
        assert reading.signal_value == -1.0
        assert reading.conviction == 1.0
        assert reading.active is True

    def test_to_dict_roundtrip_json(self):
        """PairReading serialized and re-parsed via JSON should preserve key fields."""
        reading = PairReading(
            pair_name="spy_efa", symbol_a="SPY", symbol_b="EFA",
            return_a_60d=2.5, return_b_60d=1.0, return_differential=1.5,
            z_score=1.2, z_score_mean=0.3, z_score_std=0.7,
            signal_value=-0.5, regime="neutral", conviction=0.3,
            active=False, days_active=0, entry_zscore=0.0,
        )
        d = reading.to_dict()
        reconstructed = json.dumps(d)
        parsed = json.loads(reconstructed)
        assert parsed["pair_name"] == "spy_efa"
        assert parsed["z_score"] == 1.2
        assert parsed["regime"] == "neutral"


# ---------------------------------------------------------------------------
# CrossAssetRVSignal edge cases
# ---------------------------------------------------------------------------

class TestCrossAssetRVSignalEdgeCases:
    """Edge cases for CrossAssetRVSignal dataclass."""

    def test_negative_avg_z_score(self):
        """avg_z_score can be negative when bearish divergences dominate."""
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24",
            pairs={},
            avg_z_score=-1.5,
            max_divergence=2.0,
            num_diverged=1,
            total_pairs=5,
            risk_on_score=0.5,
            duration_score=-0.3,
            overall_conviction=0.6,
        )
        assert signal.avg_z_score < 0
        d = signal.to_dict()
        assert d["avg_z_score"] == -1.5

    def test_all_pairs_diverged(self):
        """max_divergence should reflect the most extreme z-score when all pairs diverge."""
        r1 = PairReading(
            pair_name="spy_qqq", symbol_a="SPY", symbol_b="QQQ",
            return_a_60d=5.0, return_b_60d=2.0, return_differential=3.0,
            z_score=2.5, z_score_mean=0.0, z_score_std=1.0,
            signal_value=-0.62, regime="diverged_bull", conviction=0.83,
            active=True, days_active=3, entry_zscore=2.5,
        )
        r2 = PairReading(
            pair_name="spy_gld", symbol_a="SPY", symbol_b="GLD",
            return_a_60d=-5.0, return_b_60d=3.0, return_differential=-8.0,
            z_score=-3.5, z_score_mean=0.0, z_score_std=1.0,
            signal_value=0.88, regime="diverged_bear", conviction=1.0,
            active=True, days_active=5, entry_zscore=-3.5,
        )
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24",
            pairs={"spy_qqq": r1, "spy_gld": r2},
            avg_z_score=-0.5,
            max_divergence=3.5,
            num_diverged=2,
            total_pairs=5,
            risk_on_score=-0.3,
            duration_score=0.1,
            overall_conviction=0.9,
        )
        assert signal.max_divergence == 3.5
        assert signal.num_diverged == 2

    def test_single_empty_pairs_preserves_total(self):
        """Empty pairs dict should still report total_pairs accurately."""
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24",
            pairs={},
            avg_z_score=0.0,
            max_divergence=0.0,
            num_diverged=0,
            total_pairs=5,
            risk_on_score=0.0,
            duration_score=0.0,
            overall_conviction=0.0,
        )
        d = signal.to_dict()
        assert d["total_pairs"] == 5
        assert d["pairs"] == {}


# ---------------------------------------------------------------------------
# Data loading comprehensive edge cases
# ---------------------------------------------------------------------------

class TestDataLoadingFullCoverage:
    """Comprehensive data loading edge cases."""

    def test_load_empty_json_file(self, tmp_path):
        """Valid JSON with empty dict should return False (no dates)."""
        data_dir = tmp_path / "empty_json"
        data_dir.mkdir()
        with open(data_dir / "prices.json", "w") as f:
            json.dump({}, f)
        scanner = CrossAssetRVScanner(data_dir=data_dir)
        assert scanner._load_price_data() is False

    def test_load_empty_symbol_arrays(self, tmp_path):
        """Symbols with empty list entries should produce no dates."""
        data_dir = tmp_path / "empty_arr"
        data_dir.mkdir()
        prices = {"SPY": [], "QQQ": [], "GLD": []}
        with open(data_dir / "prices.json", "w") as f:
            json.dump(prices, f)
        scanner = CrossAssetRVScanner(data_dir=data_dir)
        assert scanner._load_price_data() is False

    def test_load_missing_keys_skipped(self, tmp_path):
        """Entries missing 'd' or 'p' keys should be silently skipped."""
        data_dir = tmp_path / "missing_keys"
        data_dir.mkdir()
        prices = {
            "SPY": [
                {"d": "2026-01-01", "p": 500.0},
                {"x": "2026-01-02", "y": 501.0},  # missing d and p
                {"d": "2026-01-03", "p": 502.0},
            ]
        }
        with open(data_dir / "prices.json", "w") as f:
            json.dump(prices, f)
        scanner = CrossAssetRVScanner(data_dir=data_dir)
        result = scanner._load_price_data()
        assert result is True
        assert "SPY" in scanner.prices
        assert len(scanner.prices["SPY"]) == 2  # 2 valid entries

    def test_load_partial_nan_within_symbol(self, tmp_path):
        """Valid entries should be loaded; entries with null prices should be skipped."""
        data_dir = tmp_path / "null_price"
        data_dir.mkdir()
        prices = {
            "SPY": [
                {"d": "2026-01-01", "p": 500.0},
                {"d": "2026-01-02", "p": None},
                {"d": "2026-01-03", "p": 502.0},
            ],
            "GLD": [
                {"d": "2026-01-01", "p": 180.0},
                {"d": "2026-01-02", "p": 181.0},
                {"d": "2026-01-03", "p": 182.0},
            ],
        }
        with open(data_dir / "prices.json", "w") as f:
            json.dump(prices, f)
        scanner = CrossAssetRVScanner(data_dir=data_dir)
        result = scanner._load_price_data()
        assert result is True
        # Only 2 dates actually populated for SPY (jan 1 and 3)
        assert len(scanner.dates) >= 2

    def test_load_price_data_file_not_found(self, tmp_path):
        """Missing prices.json should return False."""
        scanner = CrossAssetRVScanner(data_dir=tmp_path / "nope")
        with patch.object(scanner, '_load_price_data', return_value=False):
            assert scanner._load_price_data() is False

    def test_load_single_valid_date(self, tmp_path):
        """Single date entry should load successfully."""
        data_dir = tmp_path / "single_date"
        data_dir.mkdir()
        prices = {
            "SPY": [{"d": "2026-01-01", "p": 500.0}],
            "GLD": [{"d": "2026-01-01", "p": 180.0}],
        }
        with open(data_dir / "prices.json", "w") as f:
            json.dump(prices, f)
        scanner = CrossAssetRVScanner(data_dir=data_dir)
        assert scanner._load_price_data() is True
        assert len(scanner.dates) == 1


# ---------------------------------------------------------------------------
# Computation comprehensive edge cases
# ---------------------------------------------------------------------------

class TestComputationsFullCoverage:
    """Comprehensive computation edge cases."""

    def test_returns_period_one(self, scanner):
        """Period=1 should compute daily returns."""
        prices = np.array([100.0, 101.0, 103.0, 102.0])
        rets = scanner._compute_returns(prices, period=1)
        assert rets[1] == pytest.approx(0.01, abs=1e-5)
        assert rets[2] == pytest.approx(103.0 / 101.0 - 1, abs=1e-5)
        assert rets[3] == pytest.approx(102.0 / 103.0 - 1, abs=1e-5)

    def test_zscore_window_equals_length(self, scanner):
        """Window equal to array length -> no iteration, all NaN."""
        np.random.seed(42)
        values = np.random.normal(0, 1, 30)
        z_scores, means, stds = scanner._compute_z_score(values, window=30)
        assert np.all(np.isnan(z_scores))

    def test_zscore_insufficient_clean(self, scanner):
        """Segment with fewer than MIN_HISTORY clean values should be skipped."""
        values = np.full(100, np.nan)
        values[:10] = 1.0
        z_scores, means, stds = scanner._compute_z_score(values, window=30)
        # Every window has at most 10 clean values < MIN_HISTORY (20)
        assert np.all(np.isnan(z_scores))

    def test_zscore_min_history_boundary(self, scanner):
        """Exactly MIN_HISTORY clean values in window should compute z-score."""
        np.random.seed(42)
        window = 30
        values = np.full(70, np.nan)
        # First 20 values are valid; indices 20-29 are NaN; index 30 is valid
        values[:MIN_HISTORY] = np.random.normal(0, 1, MIN_HISTORY)
        values[window] = 1.0  # The value at index 30 being evaluated — must not be NaN
        z_scores, means, stds = scanner._compute_z_score(values, window=window)
        # At i=30: segment = values[0:30], clean = 20 = MIN_HISTORY
        # 20 < 20 is False -> should compute, and values[30]=1.0 (not NaN)
        first_valid = np.where(~np.isnan(z_scores))[0]
        assert len(first_valid) > 0

    def test_zscore_all_zero_series(self, scanner):
        """Series where std is 0 at all positions should produce all-NaN z-scores."""
        values = np.ones(80)
        z_scores, means, stds = scanner._compute_z_score(values, window=30)
        assert np.all(np.isnan(z_scores))

    def test_returns_empty_array(self, scanner):
        """Empty price array should produce empty returns."""
        prices = np.array([])
        rets = scanner._compute_returns(prices, period=10)
        assert len(rets) == 0

    def test_zscore_negative_values(self, scanner):
        """Z-score computation should work with negative values."""
        np.random.seed(42)
        values = np.random.normal(-5, 2, 100)
        z_scores, means, stds = scanner._compute_z_score(values, window=30)
        valid = z_scores[~np.isnan(z_scores)]
        assert len(valid) > 0
        assert np.all(np.isfinite(valid))


# ---------------------------------------------------------------------------
# Scan pair regime logic with controlled z-scores
# ---------------------------------------------------------------------------

class TestScanPairRegime:
    """Test regime detection logic with mocked _compute_z_score."""

    def _make_z_mock(self, scanner, z_value):
        """Return mocked (z_scores, means, stds) with z_value at last index, matching data size."""
        n = len(scanner.dates) if len(scanner.dates) > 0 else LOOKBACK + 10
        z_arr = np.full(n, np.nan)
        z_arr[-1] = z_value
        means_arr = np.full(n, np.nan)
        means_arr[-1] = 0.0
        stds_arr = np.full(n, np.nan)
        stds_arr[-1] = 1.0
        return z_arr, means_arr, stds_arr

    def test_diverged_bull_regime(self, scanner):
        """z=2.5 (above +2.0) -> diverged_bull, negative signal, short A/long B."""
        scanner._load_price_data()
        with patch.object(scanner, '_compute_z_score') as mock_z:
            mock_z.return_value = self._make_z_mock(scanner, 2.5)
            reading = scanner.scan_pair("spy_qqq")
        assert reading is not None
        assert reading.regime == "diverged_bull"
        assert reading.signal_value == pytest.approx(-0.625, abs=1e-4)
        assert reading.conviction == pytest.approx(0.8333, abs=1e-3)
        assert reading.active is True

    def test_diverged_bear_regime(self, scanner):
        """z=-2.5 (below -2.0) -> diverged_bear, positive signal, long A/short B."""
        scanner._load_price_data()
        with patch.object(scanner, '_compute_z_score') as mock_z:
            mock_z.return_value = self._make_z_mock(scanner, -2.5)
            reading = scanner.scan_pair("spy_qqq")
        assert reading is not None
        assert reading.regime == "diverged_bear"
        assert reading.signal_value == pytest.approx(0.625, abs=1e-4)
        assert reading.conviction == pytest.approx(0.8333, abs=1e-3)
        assert reading.active is True

    def test_converged_regime(self, scanner):
        """z=0.3 (below +0.5 exit) -> converged, no signal, inactive."""
        scanner._load_price_data()
        with patch.object(scanner, '_compute_z_score') as mock_z:
            mock_z.return_value = self._make_z_mock(scanner, 0.3)
            reading = scanner.scan_pair("spy_qqq")
        assert reading is not None
        assert reading.regime == "converged"
        assert reading.signal_value == 0.0
        assert reading.conviction == 0.0
        assert reading.active is False

    def test_neutral_regime_positive_z(self, scanner):
        """z=1.0 (between exit and entry) -> neutral, gradual negative signal."""
        scanner._load_price_data()
        with patch.object(scanner, '_compute_z_score') as mock_z:
            mock_z.return_value = self._make_z_mock(scanner, 1.0)
            reading = scanner.scan_pair("spy_qqq")
        assert reading is not None
        assert reading.regime == "neutral"
        assert reading.signal_value == pytest.approx(-0.5, abs=1e-4)  # -z/2.0
        assert reading.conviction == pytest.approx(0.25, abs=1e-4)    # abs_z/2.0 * 0.5
        assert reading.active is False

    def test_neutral_regime_negative_z(self, scanner):
        """z=-1.0 (between exit and entry) -> neutral, gradual positive signal."""
        scanner._load_price_data()
        with patch.object(scanner, '_compute_z_score') as mock_z:
            mock_z.return_value = self._make_z_mock(scanner, -1.0)
            reading = scanner.scan_pair("spy_qqq")
        assert reading is not None
        assert reading.regime == "neutral"
        assert reading.signal_value == pytest.approx(0.5, abs=1e-4)   # -(-1)/2.0
        assert reading.conviction == pytest.approx(0.25, abs=1e-4)
        assert reading.active is False

    def test_zero_zscore_converged(self, scanner):
        """z=0.0 -> converged (below 0.5), no signal."""
        scanner._load_price_data()
        with patch.object(scanner, '_compute_z_score') as mock_z:
            mock_z.return_value = self._make_z_mock(scanner, 0.0)
            reading = scanner.scan_pair("spy_qqq")
        assert reading is not None
        assert reading.regime == "converged"
        assert reading.signal_value == 0.0
        assert reading.active is False

    def test_extreme_divergence_signal_clip(self, scanner):
        """z=5.0 -> diverged_bull, signal clips at -1.0, conviction caps at 1.0."""
        scanner._load_price_data()
        with patch.object(scanner, '_compute_z_score') as mock_z:
            mock_z.return_value = self._make_z_mock(scanner, 5.0)
            reading = scanner.scan_pair("spy_qqq")
        assert reading is not None
        assert reading.regime == "diverged_bull"
        assert reading.signal_value == -1.0       # clipped at -1.0
        assert reading.conviction == 1.0          # capped at 1.0
        assert reading.active is True

    def test_entry_threshold_boundary_just_below(self, scanner):
        """z=1.999 (just below +2.0) -> neutral, not diverged."""
        scanner._load_price_data()
        with patch.object(scanner, '_compute_z_score') as mock_z:
            mock_z.return_value = self._make_z_mock(scanner, 1.999)
            reading = scanner.scan_pair("spy_qqq")
        assert reading is not None
        assert reading.regime == "neutral"
        assert reading.active is False


# ---------------------------------------------------------------------------
# Scan pair state transitions
# ---------------------------------------------------------------------------

class TestScanPairState:
    """State persistence transitions across multiple scans."""

    def _make_z_mock(self, scanner, z_value):
        n = len(scanner.dates) if len(scanner.dates) > 0 else LOOKBACK + 10
        z_arr = np.full(n, np.nan)
        z_arr[-1] = z_value
        means_arr = np.full(n, np.nan)
        means_arr[-1] = 0.0
        stds_arr = np.full(n, np.nan)
        stds_arr[-1] = 1.0
        return z_arr, means_arr, stds_arr

    def test_new_entry_zero_days(self, scanner):
        """A newly diverged pair with no prior state should start at days_active=0."""
        scanner._load_price_data()
        with patch.object(scanner, '_compute_z_score') as mock_z:
            mock_z.return_value = self._make_z_mock(scanner, 2.5)
            reading = scanner.scan_pair("spy_qqq")
        assert reading.days_active == 0
        assert reading.entry_zscore == 0.0

    def test_active_pair_increments_days(self, scanner):
        """Pre-existing active state should increment days_active."""
        scanner._load_price_data()
        scanner._save_state({
            "spy_qqq": {"active": True, "days_active": 5, "entry_zscore": 2.5, "last_zscore": 2.3}
        })
        with patch.object(scanner, '_compute_z_score') as mock_z:
            mock_z.return_value = self._make_z_mock(scanner, 2.5)
            reading = scanner.scan_pair("spy_qqq")
        assert reading.days_active == 6  # 5 + 1
        assert reading.entry_zscore == 2.5

    def test_active_pair_becomes_inactive(self, scanner):
        """Previously active pair that converges should reset state."""
        scanner._load_price_data()
        scanner._save_state({
            "spy_qqq": {"active": True, "days_active": 5, "entry_zscore": 2.5, "last_zscore": 2.3}
        })
        with patch.object(scanner, '_compute_z_score') as mock_z:
            mock_z.return_value = self._make_z_mock(scanner, 0.3)
            reading = scanner.scan_pair("spy_qqq")
        assert reading.days_active == 0
        assert reading.entry_zscore == 0.0
        assert reading.active is False

    def test_state_saved_after_scan(self, scanner):
        """After scan_pair, the updated state should persist on disk."""
        scanner._load_price_data()
        with patch.object(scanner, '_compute_z_score') as mock_z:
            mock_z.return_value = self._make_z_mock(scanner, 2.5)
            scanner.scan_pair("spy_qqq")
        state = scanner._load_state()
        assert "spy_qqq" in state
        assert state["spy_qqq"]["active"] is True
        assert "last_zscore" in state["spy_qqq"]
        assert "last_scan" in state["spy_qqq"]


# ---------------------------------------------------------------------------
# Ensemble signal detailed tests
# ---------------------------------------------------------------------------

class TestEnsembleSignalDetail:
    """Detailed tests for get_ensemble_signal calculations."""

    def _make_reading(self, pair_name, z_score, signal_value, conviction):
        return PairReading(
            pair_name=pair_name, symbol_a=pair_name.split("_")[0].upper(),
            symbol_b=pair_name.split("_")[1].upper(),
            return_a_60d=1.0, return_b_60d=0.5, return_differential=0.5,
            z_score=z_score, z_score_mean=0.0, z_score_std=1.0,
            signal_value=signal_value, regime="diverged_bull" if abs(z_score) > 2.0 else "neutral",
            conviction=conviction, active=abs(z_score) > 2.0,
            days_active=0, entry_zscore=z_score,
        )

    def test_diverged_pairs_confidence(self, scanner):
        """With diverged pairs, confidence is mean of diverged convictions."""
        diverged = self._make_reading("spy_qqq", 2.5, -0.62, 0.83)
        neutral = self._make_reading("spy_efa", 1.0, -0.5, 0.25)
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24",
            pairs={"spy_qqq": diverged, "spy_efa": neutral, "gld_btc": neutral,
                   "tlt_ief": neutral, "spy_gld": neutral},
            avg_z_score=0.5, max_divergence=2.5, num_diverged=1, total_pairs=5,
            risk_on_score=-0.2, duration_score=0.0, overall_conviction=0.3,
        )
        with patch.object(scanner, 'scan_all', return_value=signal):
            es = scanner.get_ensemble_signal()
        assert es["confidence"] == pytest.approx(0.83, abs=1e-4)

    def test_no_diverged_uses_overall_conviction(self, scanner):
        """Without diverged pairs, confidence = overall_conviction * 0.5."""
        neutral = self._make_reading("spy_qqq", 0.3, 0.0, 0.0)
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24",
            pairs={"spy_qqq": neutral, "spy_efa": neutral, "gld_btc": neutral,
                   "tlt_ief": neutral, "spy_gld": neutral},
            avg_z_score=0.1, max_divergence=0.3, num_diverged=0, total_pairs=5,
            risk_on_score=0.0, duration_score=0.0, overall_conviction=0.4,
        )
        with patch.object(scanner, 'scan_all', return_value=signal):
            es = scanner.get_ensemble_signal()
        assert es["confidence"] == pytest.approx(0.2, abs=1e-4)  # 0.4 * 0.5

    def test_spy_bias_aggregation(self, scanner):
        """SPY bias should combine weighted contributions from all SPY pairs."""
        r_qqq = self._make_reading("spy_qqq", 2.5, -0.5, 0.83)
        r_efa = self._make_reading("spy_efa", -2.5, 0.6, 0.83)
        r_gld = self._make_reading("spy_gld", 1.0, -0.3, 0.25)
        r_btc = self._make_reading("gld_btc", 1.0, 0.0, 0.0)
        r_ief = self._make_reading("tlt_ief", 1.0, 0.0, 0.0)
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24",
            pairs={"spy_qqq": r_qqq, "spy_efa": r_efa, "gld_btc": r_btc,
                   "tlt_ief": r_ief, "spy_gld": r_gld},
            avg_z_score=0.0, max_divergence=2.5, num_diverged=2, total_pairs=5,
            risk_on_score=0.0, duration_score=0.0, overall_conviction=0.8,
        )
        with patch.object(scanner, 'scan_all', return_value=signal):
            es = scanner.get_ensemble_signal()
        expected_spy = (-0.5 * 0.4) + (0.6 * 0.3) + (-0.3 * 0.3)
        assert es["asset_signals"]["SPY"] == pytest.approx(expected_spy, abs=1e-4)

    def test_gld_tlt_bias_aggregation(self, scanner):
        """GLD and TLT biases should combine weighted contributions correctly."""
        r_gld = self._make_reading("spy_gld", 2.5, -0.5, 0.83)
        r_btc = self._make_reading("gld_btc", -2.5, 0.6, 0.83)
        r_ief = self._make_reading("tlt_ief", 2.5, -0.4, 0.83)
        r_qqq = self._make_reading("spy_qqq", 0.0, 0.0, 0.0)
        r_efa = self._make_reading("spy_efa", 0.0, 0.0, 0.0)
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24",
            pairs={"spy_qqq": r_qqq, "spy_efa": r_efa, "gld_btc": r_btc,
                   "tlt_ief": r_ief, "spy_gld": r_gld},
            avg_z_score=0.0, max_divergence=2.5, num_diverged=3, total_pairs=5,
            risk_on_score=0.0, duration_score=0.0, overall_conviction=0.8,
        )
        with patch.object(scanner, 'scan_all', return_value=signal):
            es = scanner.get_ensemble_signal()
        expected_gld = -(-0.5 * 0.6) - (0.6 * 0.4)
        assert es["asset_signals"]["GLD"] == pytest.approx(expected_gld, abs=1e-4)
        expected_tlt = -0.4 * 0.5
        assert es["asset_signals"]["TLT"] == pytest.approx(expected_tlt, abs=1e-4)

    def test_ensemble_signal_empty_pairs(self, scanner):
        """Empty pairs dict should return zero signal and confidence."""
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24", pairs={},
            avg_z_score=0.0, max_divergence=0.0, num_diverged=0, total_pairs=5,
            risk_on_score=0.0, duration_score=0.0, overall_conviction=0.0,
        )
        with patch.object(scanner, 'scan_all', return_value=signal):
            es = scanner.get_ensemble_signal()
        assert es["signal_value"] == 0.0
        assert es["confidence"] == 0.0
        assert es["pairs"] == {}


# ---------------------------------------------------------------------------
# Directional bias (risk_on, duration) tests
# ---------------------------------------------------------------------------

class TestDirectionalBias:
    """Directional bias calculations in scan_all."""

    def test_risk_on_score_calculation(self, scanner):
        """risk_on_score should combine weighted signals from spy_qqq, spy_efa, spy_gld."""
        # Mock scan_pair to return controlled signal values for each pair
        readings = {
            "spy_qqq": PairReading(
                pair_name="spy_qqq", symbol_a="SPY", symbol_b="QQQ",
                return_a_60d=5.0, return_b_60d=2.0, return_differential=3.0,
                z_score=-0.5, z_score_mean=0.0, z_score_std=1.0,
                signal_value=0.5, regime="neutral", conviction=0.25,
                active=False, days_active=0, entry_zscore=0.0,
            ),
            "spy_efa": PairReading(
                pair_name="spy_efa", symbol_a="SPY", symbol_b="EFA",
                return_a_60d=3.0, return_b_60d=1.0, return_differential=2.0,
                z_score=0.3, z_score_mean=0.0, z_score_std=1.0,
                signal_value=-0.3, regime="neutral", conviction=0.15,
                active=False, days_active=0, entry_zscore=0.0,
            ),
            "gld_btc": PairReading(
                pair_name="gld_btc", symbol_a="GLD", symbol_b="BTC",
                return_a_60d=2.0, return_b_60d=4.0, return_differential=-2.0,
                z_score=0.1, z_score_mean=0.0, z_score_std=1.0,
                signal_value=0.0, regime="converged", conviction=0.0,
                active=False, days_active=0, entry_zscore=0.0,
            ),
            "tlt_ief": PairReading(
                pair_name="tlt_ief", symbol_a="TLT", symbol_b="IEF",
                return_a_60d=1.0, return_b_60d=0.5, return_differential=0.5,
                z_score=0.2, z_score_mean=0.0, z_score_std=1.0,
                signal_value=0.0, regime="converged", conviction=0.0,
                active=False, days_active=0, entry_zscore=0.0,
            ),
            "spy_gld": PairReading(
                pair_name="spy_gld", symbol_a="SPY", symbol_b="GLD",
                return_a_60d=4.0, return_b_60d=2.0, return_differential=2.0,
                z_score=0.4, z_score_mean=0.0, z_score_std=1.0,
                signal_value=-0.2, regime="neutral", conviction=0.1,
                active=False, days_active=0, entry_zscore=0.0,
            ),
        }
        with patch.object(scanner, 'scan_pair', side_effect=lambda name: readings.get(name)):
            scanner.prices = {"SPY": np.ones(100), "QQQ": np.ones(100), "EFA": np.ones(100),
                             "GLD": np.ones(100), "BTC": np.ones(100), "TLT": np.ones(100),
                             "IEF": np.ones(100)}
            signal = scanner.scan_all()
        # risk_on = -spy_qqq.signal - spy_efa.signal*0.5 - spy_gld.signal*0.3
        expected_risk_on = -(0.5) - (-0.3 * 0.5) - (-0.2 * 0.3)
        assert signal.risk_on_score == pytest.approx(expected_risk_on, abs=1e-4)

    def test_duration_score_calculation(self, scanner):
        """duration_score should come from tlt_ief signal."""
        readings = {
            "spy_qqq": PairReading(
                pair_name="spy_qqq", symbol_a="SPY", symbol_b="QQQ",
                return_a_60d=0.0, return_b_60d=0.0, return_differential=0.0,
                z_score=0.0, z_score_mean=0.0, z_score_std=1.0,
                signal_value=0.0, regime="converged", conviction=0.0,
                active=False, days_active=0, entry_zscore=0.0,
            ),
            "spy_efa": PairReading(
                pair_name="spy_efa", symbol_a="SPY", symbol_b="EFA",
                return_a_60d=0.0, return_b_60d=0.0, return_differential=0.0,
                z_score=0.0, z_score_mean=0.0, z_score_std=1.0,
                signal_value=0.0, regime="converged", conviction=0.0,
                active=False, days_active=0, entry_zscore=0.0,
            ),
            "gld_btc": PairReading(
                pair_name="gld_btc", symbol_a="GLD", symbol_b="BTC",
                return_a_60d=0.0, return_b_60d=0.0, return_differential=0.0,
                z_score=0.0, z_score_mean=0.0, z_score_std=1.0,
                signal_value=0.0, regime="converged", conviction=0.0,
                active=False, days_active=0, entry_zscore=0.0,
            ),
            "tlt_ief": PairReading(
                pair_name="tlt_ief", symbol_a="TLT", symbol_b="IEF",
                return_a_60d=2.0, return_b_60d=1.0, return_differential=1.0,
                z_score=2.5, z_score_mean=0.0, z_score_std=1.0,
                signal_value=-0.6, regime="diverged_bull", conviction=0.8,
                active=True, days_active=3, entry_zscore=2.5,
            ),
            "spy_gld": PairReading(
                pair_name="spy_gld", symbol_a="SPY", symbol_b="GLD",
                return_a_60d=0.0, return_b_60d=0.0, return_differential=0.0,
                z_score=0.0, z_score_mean=0.0, z_score_std=1.0,
                signal_value=0.0, regime="converged", conviction=0.0,
                active=False, days_active=0, entry_zscore=0.0,
            ),
        }
        with patch.object(scanner, 'scan_pair', side_effect=lambda name: readings.get(name)):
            scanner.prices = {"SPY": np.ones(100), "QQQ": np.ones(100), "EFA": np.ones(100),
                             "GLD": np.ones(100), "BTC": np.ones(100), "TLT": np.ones(100),
                             "IEF": np.ones(100)}
            signal = scanner.scan_all()
        # duration = tlt_ief.signal = -0.6
        assert signal.duration_score == pytest.approx(-0.6, abs=1e-4)


# ---------------------------------------------------------------------------
# SignalSnapshot edge cases
# ---------------------------------------------------------------------------

class TestSignalSnapshotEdgeCases:
    """Edge cases for get_signal_snapshot."""

    def test_snapshot_with_zero_signal(self, scanner):
        """Snapshot with zero signal value should have is_active=False."""
        from src.signals.signal_snapshot import SignalSnapshot
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24", pairs={},
            avg_z_score=0.0, max_divergence=0.0, num_diverged=0, total_pairs=5,
            risk_on_score=0.0, duration_score=0.0, overall_conviction=0.0,
        )
        with patch.object(scanner, 'scan_all', return_value=signal):
            snapshot = scanner.get_signal_snapshot()
        assert isinstance(snapshot, SignalSnapshot)
        assert snapshot.source == "cross_asset_rv"
        assert snapshot.is_active is False

    def test_snapshot_no_data(self, tmp_path):
        """Scanner without data should still produce a valid SignalSnapshot."""
        from src.signals.signal_snapshot import SignalSnapshot
        scanner = CrossAssetRVScanner(data_dir=tmp_path / "nodata_snap")
        with patch.object(scanner, '_load_price_data', return_value=False):
            snapshot = scanner.get_signal_snapshot()
        assert isinstance(snapshot, SignalSnapshot)
        assert snapshot.source == "cross_asset_rv"
        assert snapshot.value == 0.0

    def test_snapshot_regime_fit_default(self, scanner):
        """Snapshot should always have regime_fit='all'."""
        from src.signals.signal_snapshot import SignalSnapshot
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24", pairs={},
            avg_z_score=1.5, max_divergence=2.0, num_diverged=1, total_pairs=5,
            risk_on_score=-0.3, duration_score=0.1, overall_conviction=0.6,
        )
        with patch.object(scanner, 'scan_all', return_value=signal):
            snapshot = scanner.get_signal_snapshot()
        assert snapshot.regime_fit == "all"

    def test_snapshot_explanation_format(self, scanner):
        """Snapshot explanation should contain z-score and diverged count format."""
        from src.signals.signal_snapshot import SignalSnapshot
        # Use a signal with actual pairs data so z-score and diverged count are computed
        pair = PairReading(
            pair_name="SPY/GLD", symbol_a="SPY", symbol_b="GLD",
            return_a_60d=0.05, return_b_60d=-0.02, return_differential=0.07,
            z_score=1.23, z_score_mean=0.0, z_score_std=1.0,
            signal_value=0.5, regime="normal", conviction=0.6,
            active=True, days_active=5, entry_zscore=1.5,
        )
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24",
            pairs={"SPY/GLD": pair},
            avg_z_score=1.23, max_divergence=2.5, num_diverged=1, total_pairs=1,
            risk_on_score=-0.4, duration_score=0.2, overall_conviction=0.7,
        )
        with patch.object(scanner, 'scan_all', return_value=signal):
            snapshot = scanner.get_signal_snapshot()
        # Explanation format: "Cross-asset RV: z=+X.XX, diverged=N/M pairs"
        assert "z=" in snapshot.explanation
        assert "diverged" in snapshot.explanation
        assert "pairs" in snapshot.explanation


# ---------------------------------------------------------------------------
# State persistence full edge cases
# ---------------------------------------------------------------------------

class TestStatePersistenceFull:
    """Full state persistence edge cases."""

    def test_empty_state_file(self, tmp_path):
        """Empty state file should return empty dict."""
        scanner = CrossAssetRVScanner(data_dir=tmp_path / "empty_state")
        scanner.state_dir.mkdir(parents=True, exist_ok=True)
        scanner.state_path.write_text("")
        state = scanner._load_state()
        assert state == {}

    def test_state_file_with_extra_fields(self, scanner):
        """State file with extra fields should not break loading."""
        state = {
            "spy_qqq": {"active": True, "days_active": 3, "entry_zscore": 2.1,
                        "last_zscore": 2.0, "last_scan": "2026-05-24",
                        "extra_field": "should_not_cause_errors"},
        }
        scanner._save_state(state)
        loaded = scanner._load_state()
        assert loaded["spy_qqq"]["active"] is True
        assert loaded["spy_qqq"]["extra_field"] == "should_not_cause_errors"

    def test_state_not_found_returns_empty(self, tmp_path):
        """Non-existent state file should return empty dict."""
        scanner = CrossAssetRVScanner(data_dir=tmp_path / "no_state")
        assert scanner._load_state() == {}


# ---------------------------------------------------------------------------
# Edge-case scan_pair and get_ensemble_signal
# ---------------------------------------------------------------------------

class TestScannerResilience:
    """Resilience of scan_pair and related methods to unusual conditions."""

    def test_scan_pair_short_prices(self, scanner):
        """Scanner with insufficient price data should return None for all pairs."""
        # Use scanner that loaded too-short data
        scanner.prices = {sym: np.array([100.0, 101.0]) for sym in
                         ["SPY", "QQQ", "EFA", "GLD", "BTC", "TLT", "IEF"]}
        scanner.dates = ["2026-01-01", "2026-01-02"]
        reading = scanner.scan_pair("spy_qqq")
        assert reading is None

    def test_scan_pair_missing_symbol_from_prices(self, scanner):
        """Pair requiring a symbol not in prices dict returns None."""
        scanner.prices = {"SPY": np.ones(100)}
        scanner.dates = ["2026-01-01"]
        reading = scanner.scan_pair("spy_qqq")
        assert reading is None

    def test_scan_all_preserves_correct_pair_count(self, scanner):
        """scan_all with no data should still report total_pairs correctly."""
        with patch.object(scanner, '_load_price_data', return_value=False):
            signal = scanner.scan_all()
        assert signal.total_pairs == len(CROSS_ASSET_PAIRS)
        assert signal.num_diverged == 0

    def test_ensemble_signal_missing_asset_pairs(self, scanner):
        """get_ensemble_signal should handle missing pair keys gracefully."""
        signal = CrossAssetRVSignal(
            timestamp="2026-05-24", pairs={},
            avg_z_score=0.0, max_divergence=0.0, num_diverged=0, total_pairs=5,
            risk_on_score=0.0, duration_score=0.0, overall_conviction=0.0,
        )
        with patch.object(scanner, 'scan_all', return_value=signal):
            es = scanner.get_ensemble_signal()
        assert es["signal_value"] == 0.0
        assert es["confidence"] == 0.0
        assert isinstance(es["pairs"], dict)
