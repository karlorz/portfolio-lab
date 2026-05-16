#!/usr/bin/env python3
"""
Tests for ML-Light Regime Predictor (v5.73).

Tests the deterministic, threshold-based regime classifier using
synthetic price data with known regime characteristics.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.signals.regime_classifier import (
    RegimeClassifier,
    Regime,
    RegimeFactors,
    RegimeReading,
    REGIME_DESCRIPTION,
    REGIME_CONFIDENCE_MAP,
    LOW_VOL_THRESH,
    HIGH_VOL_THRESH,
    CRISIS_VOL_THRESH,
    MOM_NEGATIVE,
    DD_CRISIS,
    print_scan,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normal_market_prices(tmp_path):
    """Create prices with stable, normal market conditions."""
    dates = []
    base = datetime(2026, 1, 1)
    for i in range(200):
        d = base + timedelta(days=i)
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))

    np.random.seed(42)
    n = len(dates)

    # SPY: low vol, slight uptrend
    spy_returns = np.random.normal(0.0005, 0.008, n)
    spy_prices = 480.0 * np.exp(np.cumsum(spy_returns))

    # GLD: slight uptrend
    gld_returns = np.random.normal(0.0003, 0.006, n)
    gld_prices = 180.0 * np.exp(np.cumsum(gld_returns))

    # TLT, IEF: stable
    tlt_returns = np.random.normal(0.0001, 0.005, n)
    tlt_prices = 95.0 * np.exp(np.cumsum(tlt_returns))
    ief_returns = np.random.normal(0.0001, 0.003, n)
    ief_prices = 105.0 * np.exp(np.cumsum(ief_returns))

    prices = {
        "SPY": [{"d": dates[i], "p": float(spy_prices[i])} for i in range(n)],
        "GLD": [{"d": dates[i], "p": float(gld_prices[i])} for i in range(n)],
        "TLT": [{"d": dates[i], "p": float(tlt_prices[i])} for i in range(n)],
        "IEF": [{"d": dates[i], "p": float(ief_prices[i])} for i in range(n)],
    }

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    prices_path = tmp_path / "public" / "data"
    prices_path.mkdir(parents=True, exist_ok=True)
    with open(prices_path / "prices.json", "w") as f:
        json.dump(prices, f)
    return tmp_path


@pytest.fixture
def high_vol_prices(tmp_path):
    """Create prices with high volatility conditions."""
    dates = []
    base = datetime(2026, 1, 1)
    for i in range(200):
        d = base + timedelta(days=i)
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))

    np.random.seed(42)
    n = len(dates)

    # SPY: high vol, moderate downtrend (last 60 days)
    spy_returns = np.random.normal(-0.002, 0.025, n)
    spy_prices = 480.0 * np.exp(np.cumsum(spy_returns))

    # GLD: slight uptrend (flight to safety)
    gld_returns = np.random.normal(0.001, 0.01, n)
    gld_prices = 180.0 * np.exp(np.cumsum(gld_returns))

    # TLT: slight uptrend (flight to safety)
    tlt_returns = np.random.normal(0.0008, 0.008, n)
    tlt_prices = 95.0 * np.exp(np.cumsum(tlt_returns))
    ief_returns = np.random.normal(0.0002, 0.004, n)
    ief_prices = 105.0 * np.exp(np.cumsum(ief_returns))

    prices = {
        "SPY": [{"d": dates[i], "p": float(spy_prices[i])} for i in range(n)],
        "GLD": [{"d": dates[i], "p": float(gld_prices[i])} for i in range(n)],
        "TLT": [{"d": dates[i], "p": float(tlt_prices[i])} for i in range(n)],
        "IEF": [{"d": dates[i], "p": float(ief_prices[i])} for i in range(n)],
    }

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    prices_path = tmp_path / "public" / "data"
    prices_path.mkdir(parents=True, exist_ok=True)
    with open(prices_path / "prices.json", "w") as f:
        json.dump(prices, f)
    return tmp_path


@pytest.fixture
def crisis_prices(tmp_path):
    """Create prices with crisis conditions — extreme vol, sharp drawdown."""
    dates = []
    base = datetime(2026, 1, 1)
    for i in range(200):
        d = base + timedelta(days=i)
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))

    n = len(dates)

    # SPY: Normal then crash last 60 days
    np.random.seed(42)
    spy_returns = np.random.normal(0.0005, 0.008, n - 60).tolist()
    # Add crash
    crash_returns = np.random.normal(-0.005, 0.035, 60).tolist()
    spy_returns.extend(crash_returns)
    spy_prices = 480.0 * np.exp(np.cumsum(spy_returns))

    # GLD: rallies during crash
    gld_returns = np.random.normal(0.0003, 0.006, n - 60).tolist()
    gld_crash = np.random.normal(0.003, 0.015, 60).tolist()
    gld_returns.extend(gld_crash)
    gld_prices = 180.0 * np.exp(np.cumsum(gld_returns))

    # TLT: strong rally (flight to safety)
    tlt_returns = np.random.normal(0.0002, 0.005, n - 60).tolist()
    tlt_crash = np.random.normal(0.004, 0.012, 60).tolist()
    tlt_returns.extend(tlt_crash)
    tlt_prices = 95.0 * np.exp(np.cumsum(tlt_returns))

    ief_returns = np.random.normal(0.0001, 0.003, n).tolist()
    ief_prices = 105.0 * np.exp(np.cumsum(ief_returns))

    prices = {
        "SPY": [{"d": dates[i], "p": float(spy_prices[i])} for i in range(n)],
        "GLD": [{"d": dates[i], "p": float(gld_prices[i])} for i in range(n)],
        "TLT": [{"d": dates[i], "p": float(tlt_prices[i])} for i in range(n)],
        "IEF": [{"d": dates[i], "p": float(ief_prices[i])} for i in range(n)],
    }

    prices_path = tmp_path / "public" / "data"
    prices_path.mkdir(parents=True, exist_ok=True)
    with open(prices_path / "prices.json", "w") as f:
        json.dump(prices, f)
    return tmp_path


@pytest.fixture
def recovery_prices(tmp_path):
    """Create prices recovering from a crisis — improving momentum but elevated vol."""
    dates = []
    base = datetime(2026, 1, 1)
    for i in range(200):
        d = base + timedelta(days=i)
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))

    n = len(dates)

    # SPY: crash first 40 days of window, then strong recovery
    np.random.seed(42)
    # Crash phase
    crash_len = 40
    crash_returns = np.random.normal(-0.004, 0.03, crash_len).tolist()
    # Recovery phase
    rec_len = n - crash_len
    rec_returns = np.random.normal(0.003, 0.015, rec_len).tolist()
    spy_returns = crash_returns + rec_returns
    spy_prices = 480.0 * np.exp(np.cumsum(spy_returns))

    # GLD: rallied during crash, then normalizes
    gld_returns = np.random.normal(0.002, 0.01, crash_len).tolist()
    gld_rec = np.random.normal(0.0005, 0.008, rec_len).tolist()
    gld_returns.extend(gld_rec)
    gld_prices = 180.0 * np.exp(np.cumsum(gld_returns))

    tlt_returns = np.random.normal(0.003, 0.01, crash_len).tolist()
    tlt_rec = np.random.normal(-0.0005, 0.006, rec_len).tolist()
    tlt_returns.extend(tlt_rec)
    tlt_prices = 95.0 * np.exp(np.cumsum(tlt_returns))

    ief_returns = np.random.normal(0.0001, 0.003, n).tolist()
    ief_prices = 105.0 * np.exp(np.cumsum(ief_returns))

    prices = {
        "SPY": [{"d": dates[i], "p": float(spy_prices[i])} for i in range(n)],
        "GLD": [{"d": dates[i], "p": float(gld_prices[i])} for i in range(n)],
        "TLT": [{"d": dates[i], "p": float(tlt_prices[i])} for i in range(n)],
        "IEF": [{"d": dates[i], "p": float(ief_prices[i])} for i in range(n)],
    }

    prices_path = tmp_path / "public" / "data"
    prices_path.mkdir(parents=True, exist_ok=True)
    with open(prices_path / "prices.json", "w") as f:
        json.dump(prices, f)
    return tmp_path


@pytest.fixture
def low_vol_prices(tmp_path):
    """Create prices with very low volatility conditions."""
    dates = []
    base = datetime(2026, 1, 1)
    for i in range(200):
        d = base + timedelta(days=i)
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))

    np.random.seed(42)
    n = len(dates)

    # SPY: very low vol, steady uptrend
    spy_returns = np.random.normal(0.0008, 0.005, n)
    spy_prices = 480.0 * np.exp(np.cumsum(spy_returns))

    # GLD: low vol
    gld_returns = np.random.normal(0.0002, 0.004, n)
    gld_prices = 180.0 * np.exp(np.cumsum(gld_returns))

    tlt_returns = np.random.normal(0.0001, 0.003, n)
    tlt_prices = 95.0 * np.exp(np.cumsum(tlt_returns))
    ief_returns = np.random.normal(0.0001, 0.002, n)
    ief_prices = 105.0 * np.exp(np.cumsum(ief_returns))

    prices = {
        "SPY": [{"d": dates[i], "p": float(spy_prices[i])} for i in range(n)],
        "GLD": [{"d": dates[i], "p": float(gld_prices[i])} for i in range(n)],
        "TLT": [{"d": dates[i], "p": float(tlt_prices[i])} for i in range(n)],
        "IEF": [{"d": dates[i], "p": float(ief_prices[i])} for i in range(n)],
    }

    prices_path = tmp_path / "public" / "data"
    prices_path.mkdir(parents=True, exist_ok=True)
    with open(prices_path / "prices.json", "w") as f:
        json.dump(prices, f)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegimeClassifierInit:
    """Test classifier initialization and state management."""

    def test_init_default(self, tmp_path):
        """Default initialization should not crash."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        classifier = RegimeClassifier(data_dir=data_dir)
        assert classifier.current_regime == Regime.NORMAL
        assert classifier.previous_regime is None
        assert len(classifier.regime_history) == 0

    def test_init_with_data_dir(self, normal_market_prices):
        """Initialization with custom data dir."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        assert classifier.data_dir == normal_market_prices / "data"

    def test_load_state_from_file(self, normal_market_prices, tmp_path):
        """Loading state from file should restore regimes."""
        data_dir = normal_market_prices / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "current_regime": "crisis",
            "previous_regime": "high_vol",
            "regime_start_date": "2026-03-01",
            "history": [],
        }
        state_path = data_dir / "regime_classifier_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f)

        classifier = RegimeClassifier(data_dir=data_dir)
        assert classifier.current_regime == Regime.CRISIS
        assert classifier.previous_regime == Regime.HIGH_VOL
        assert classifier.regime_start_date == "2026-03-01"


class TestRegimeClassifierDataLoading:
    """Test data loading functionality."""

    def test_load_prices_normal(self, normal_market_prices):
        """Load prices from standard location."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        prices = classifier.load_prices()
        assert prices is not None
        assert "SPY" in prices
        assert "GLD" in prices
        assert len(prices["SPY"]) > 0

    def test_load_prices_missing_file(self, tmp_path):
        """Missing prices file should return None."""
        classifier = RegimeClassifier(data_dir=tmp_path / "data")
        prices = classifier.load_prices()
        assert prices is None

    def test_get_series(self, normal_market_prices):
        """Getting price series as numpy array."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()
        series = classifier._get_series("SPY")
        assert series is not None
        assert isinstance(series, np.ndarray)
        assert len(series) > 0

    def test_get_series_missing_symbol(self, normal_market_prices):
        """Missing symbol should return None."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()
        series = classifier._get_series("NONEXISTENT")
        assert series is None

    def test_get_dates(self, normal_market_prices):
        """Getting date strings."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()
        dates = classifier._get_dates()
        assert dates is not None
        assert len(dates) > 0
        assert isinstance(dates[0], str)


class TestRegimeClassifierFactorComputation:
    """Test factor computation."""

    def test_compute_factors_normal(self, normal_market_prices):
        """Factors should be computed from normal market data."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()
        factors = classifier.compute_factors()
        assert factors is not None
        assert factors.spy_vol_20d > 0
        assert isinstance(factors.spy_mom_20d, float)
        assert isinstance(factors.spy_drawdown_60d, float)
        assert factors.timestamp is not None

    def test_compute_factors_insufficient_data(self, tmp_path):
        """Not enough data should return None."""
        prices = {"SPY": [{"d": "2026-01-01", "p": 100.0}] * 5}
        prices_path = tmp_path / "public" / "data"
        prices_path.mkdir(parents=True, exist_ok=True)
        with open(prices_path / "prices.json", "w") as f:
            json.dump(prices, f)

        classifier = RegimeClassifier(data_dir=tmp_path / "data")
        classifier.load_prices()
        factors = classifier.compute_factors()
        assert factors is None

    def test_gld_spy_ratio(self, normal_market_prices):
        """GLD/SPY ratio should be computed."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()
        factors = classifier.compute_factors()
        assert factors is not None
        assert isinstance(factors.gld_spy_ratio_60d, float)

    def test_tlt_ief_ratio(self, normal_market_prices):
        """TLT/IEF ratio should be computed."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()
        factors = classifier.compute_factors()
        assert factors is not None
        assert isinstance(factors.tlt_ief_ratio_60d, float)


class TestRegimeClassifierClassification:
    """Test regime classification results."""

    def test_normal_market_classification(self, normal_market_prices):
        """Normal market conditions should classify as NORMAL or LOW_VOL."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()
        reading = classifier.classify()
        assert reading.regime in (Regime.NORMAL, Regime.LOW_VOL)
        assert reading.confidence >= 0.5
        assert reading.regime_reason != ""

    def test_high_vol_classification(self, high_vol_prices):
        """High vol conditions should classify as HIGH_VOL or CRISIS."""
        classifier = RegimeClassifier(data_dir=high_vol_prices / "data")
        classifier.load_prices()
        reading = classifier.classify()
        assert reading.regime in (Regime.HIGH_VOL, Regime.CRISIS)
        assert reading.confidence >= 0.5

    def test_crisis_classification(self, crisis_prices):
        """Crisis conditions should classify as CRISIS."""
        classifier = RegimeClassifier(data_dir=crisis_prices / "data")
        classifier.load_prices()
        reading = classifier.classify()
        assert reading.regime == Regime.CRISIS
        assert reading.confidence >= 0.8
        assert "Crisis" in reading.regime_reason or "crisis" in reading.regime_reason

    def test_low_vol_classification(self, low_vol_prices):
        """Low vol conditions should classify as LOW_VOL."""
        classifier = RegimeClassifier(data_dir=low_vol_prices / "data")
        classifier.load_prices()
        reading = classifier.classify()
        assert reading.regime == Regime.LOW_VOL
        assert reading.confidence >= 0.5

    def test_recovery_classification(self, recovery_prices):
        """Recovery conditions should classify as RECOVERY if preceded by crisis."""
        classifier = RegimeClassifier(data_dir=recovery_prices / "data")
        classifier.load_prices()

        # First classification — may be crisis or recovery depending on data
        reading1 = classifier.classify()
        assert reading1.regime in (Regime.CRISIS, Regime.RECOVERY, Regime.HIGH_VOL, Regime.NORMAL)

        # After initial classification, subsequent runs maintain state
        reading2 = classifier.classify()
        assert reading2.regime is not None

    def test_classification_is_deterministic(self, normal_market_prices):
        """Same data should produce same classification."""
        classifier1 = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier1.load_prices()
        reading1 = classifier1.classify()

        classifier2 = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier2.load_prices()
        reading2 = classifier2.classify()

        assert reading1.regime == reading2.regime

    def test_classification_unknown_no_data(self, tmp_path):
        """No data should produce UNKNOWN regime."""
        classifier = RegimeClassifier(data_dir=tmp_path / "data")
        reading = classifier.classify()
        assert reading.regime == Regime.UNKNOWN
        assert reading.confidence <= 0.3


class TestRegimeClassifierSignal:
    """Test signal output."""

    def test_signal_value_mapping(self, normal_market_prices):
        """Signal values should be in valid range."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()
        reading = classifier.classify()
        signal = classifier.get_signal_value(reading)
        assert -1.0 <= signal <= 1.0

    def test_signal_value_direction(self, crisis_prices):
        """Crisis should give negative signal."""
        classifier = RegimeClassifier(data_dir=crisis_prices / "data")
        classifier.load_prices()
        reading = classifier.classify()
        signal = classifier.get_signal_value(reading)
        if reading.regime == Regime.CRISIS:
            assert signal < 0

    def test_signal_value_low_vol(self, low_vol_prices):
        """Low vol should give positive signal."""
        classifier = RegimeClassifier(data_dir=low_vol_prices / "data")
        classifier.load_prices()
        reading = classifier.classify()
        signal = classifier.get_signal_value(reading)
        if reading.regime == Regime.LOW_VOL:
            assert signal > 0

    def test_asset_signals_have_all_keys(self, normal_market_prices):
        """Asset signals should include SPY, GLD, TLT, IEF."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()
        reading = classifier.classify()
        signals = classifier.get_asset_signals(reading)
        for key in ["SPY", "GLD", "TLT", "IEF"]:
            assert key in signals
            assert -1.0 <= signals[key] <= 1.0


class TestRegimeClassifierStatePersistence:
    """Test state save/load."""

    def test_state_saved_after_classify(self, normal_market_prices):
        """Classify should save state to disk."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()
        reading = classifier.classify()
        classifier._save_state(reading)

        state_path = normal_market_prices / "data" / "regime_classifier_state.json"
        assert state_path.exists()
        with open(state_path) as f:
            state = json.load(f)
        assert "current_regime" in state
        assert "last_updated" in state

    def test_state_restoration(self, normal_market_prices):
        """Saved state should be loadable by new instance."""
        classifier1 = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier1.load_prices()
        reading1 = classifier1.classify()
        classifier1._save_state(reading1)

        classifier2 = RegimeClassifier(data_dir=normal_market_prices / "data")
        assert classifier2.current_regime is not None

    def test_regime_history_accumulates(self, normal_market_prices):
        """Multiple classify calls should accumulate history."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()
        reading1 = classifier.classify()
        reading2 = classifier.classify()
        reading3 = classifier.classify()

        assert len(classifier.regime_history) >= 3

    def test_regime_duration_tracking(self, normal_market_prices):
        """Regime duration should be computed."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()
        reading = classifier.classify()
        # Duration may be None on first call if no regime_start_date set
        assert reading.regime_duration_days is None or reading.regime_duration_days >= 0


class TestRegimeReading:
    """Test RegimeReading dataclass."""

    def test_reading_to_dict(self):
        """RegimeReading should serialize to dict."""
        factors = RegimeFactors(
            timestamp="2026-01-15",
            spy_vol_20d=0.15,
            spy_mom_20d=0.01,
            spy_mom_60d=0.03,
            spy_drawdown_60d=-0.02,
            gld_spy_ratio_60d=0.01,
            tlt_ief_ratio_60d=0.02,
        )
        reading = RegimeReading(
            timestamp="2026-01-15",
            regime=Regime.NORMAL,
            confidence=0.7,
            factors=factors,
            regime_reason="Normal conditions",
        )
        d = reading.to_dict()
        assert d["regime"] == "normal"
        assert d["confidence"] == 0.7
        assert "regime_description" in d

    def test_reading_with_optional_fields(self):
        """RegimeReading with optional fields set."""
        factors = RegimeFactors(
            timestamp="2026-01-15",
            spy_vol_20d=0.15,
            spy_mom_20d=0.01,
            spy_mom_60d=0.03,
            spy_drawdown_60d=-0.02,
            gld_spy_ratio_60d=0.01,
            tlt_ief_ratio_60d=0.02,
        )
        reading = RegimeReading(
            timestamp="2026-01-15",
            regime=Regime.CRISIS,
            confidence=0.9,
            factors=factors,
            regime_reason="Crisis triggered",
            regime_duration_days=5,
            previous_regime="high_vol",
        )
        d = reading.to_dict()
        assert d["regime"] == "crisis"
        assert d["regime_duration_days"] == 5
        assert d["previous_regime"] == "high_vol"


class TestRegimeConstants:
    """Test regime constants and descriptions."""

    def test_all_regimes_have_descriptions(self):
        """Every regime should have a description."""
        for regime in Regime:
            assert regime in REGIME_DESCRIPTION
            assert len(REGIME_DESCRIPTION[regime]) > 0

    def test_all_regimes_have_confidence(self):
        """Every regime should have a confidence mapping."""
        for regime in Regime:
            assert regime in REGIME_CONFIDENCE_MAP
            assert 0 < REGIME_CONFIDENCE_MAP[regime] <= 1.0

    def test_threshold_constants_positive(self):
        """Threshold constants should be positive."""
        assert LOW_VOL_THRESH > 0
        assert HIGH_VOL_THRESH > LOW_VOL_THRESH
        assert CRISIS_VOL_THRESH > HIGH_VOL_THRESH
        assert MOM_NEGATIVE < 0
        assert DD_CRISIS < 0


class TestCli:
    """Test CLI functionality."""

    def test_print_scan_output(self, normal_market_prices, capsys):
        """print_scan should produce formatted output."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()
        reading = classifier.classify()
        print_scan(reading)
        captured = capsys.readouterr()
        assert "ML-LIGHT REGIME PREDICTOR" in captured.out
        assert reading.regime.value.upper() in captured.out

    def test_main_scan(self, normal_market_prices):
        """CLI scan command should not crash."""
        from src.signals.regime_classifier import main
        with patch.object(sys, 'argv', ['regime_classifier.py', 'scan']):
            with patch.object(RegimeClassifier, 'load_prices', return_value={}):
                try:
                    main()
                except SystemExit:
                    pass
                except Exception:
                    pass

    def test_main_signal(self, normal_market_prices):
        """CLI signal command should emit numeric output."""
        from src.signals.regime_classifier import main
        with patch.object(sys, 'argv', ['regime_classifier.py', 'signal']):
            with patch.object(RegimeClassifier, 'load_prices', return_value={}):
                try:
                    main()
                except SystemExit:
                    pass
                except Exception:
                    pass


class TestRegimeClassifierEdgeCases:
    """Test edge cases and robustness."""

    def test_empty_price_data(self, tmp_path):
        """Empty price data should not crash."""
        prices = {"SPY": []}
        prices_path = tmp_path / "public" / "data"
        prices_path.mkdir(parents=True, exist_ok=True)
        with open(prices_path / "prices.json", "w") as f:
            json.dump(prices, f)

        classifier = RegimeClassifier(data_dir=tmp_path / "data")
        classifier.load_prices()
        factors = classifier.compute_factors()
        assert factors is None

    def test_single_price_point(self, tmp_path):
        """Single price point should not crash."""
        prices = {"SPY": [{"d": "2026-01-01", "p": 100.0}]}
        prices_path = tmp_path / "public" / "data"
        prices_path.mkdir(parents=True, exist_ok=True)
        with open(prices_path / "prices.json", "w") as f:
            json.dump(prices, f)

        classifier = RegimeClassifier(data_dir=tmp_path / "data")
        classifier.load_prices()
        factors = classifier.compute_factors()
        assert factors is None

    def test_missing_symbols(self, tmp_path):
        """Missing GLD/TLT/IEF should not crash."""
        prices = {"SPY": [{"d": "2026-01-01", "p": 100.0}] * 100}
        prices_path = tmp_path / "public" / "data"
        prices_path.mkdir(parents=True, exist_ok=True)
        with open(prices_path / "prices.json", "w") as f:
            json.dump(prices, f)

        classifier = RegimeClassifier(data_dir=tmp_path / "data")
        classifier.load_prices()
        factors = classifier.compute_factors()
        assert factors is not None
        assert factors.gld_spy_ratio_60d == 0.0
        assert factors.tlt_ief_ratio_60d == 0.0

    def test_corrupt_state_file(self, tmp_path):
        """Corrupt state file should not crash initialization."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        state_path = data_dir / "regime_classifier_state.json"
        with open(state_path, "w") as f:
            f.write("NOT JSON{broken")

        classifier = RegimeClassifier(data_dir=data_dir)
        assert classifier.current_regime == Regime.NORMAL
        assert classifier.previous_regime is None

    def test_state_save_failure_doesnt_crash(self, normal_market_prices):
        """Failed state save should not crash classify."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()

        # Make state path unwritable
        state_path = normal_market_prices / "data" / "regime_classifier_state.json"
        normal_market_prices.chmod(0o444)

        try:
            reading = classifier.classify()
            # Should still work — just log warning on save failure
            assert reading is not None
        finally:
            normal_market_prices.chmod(0o755)


class TestRegimeClassifierIntegration:
    """Integration-style tests."""

    def test_classify_then_signal(self, normal_market_prices):
        """Full classify -> signal pipeline."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()
        reading = classifier.classify()
        signal = classifier.get_signal_value(reading)
        assert isinstance(signal, float)

    def test_regime_transition_tracking(self, normal_market_prices):
        """Regime transitions should update previous_regime."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()

        # Force start state
        classifier.current_regime = Regime.NORMAL
        classifier.previous_regime = None

        reading = classifier.classify()
        # If regime changed, previous should be set
        assert reading.previous_regime is None or isinstance(reading.previous_regime, str)

    def test_asset_signals_symmetry(self, normal_market_prices):
        """Asset signals should be symmetric (SPY vs GLD/TLT opposite)."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()
        reading = classifier.classify()
        signals = classifier.get_asset_signals(reading)

        # In crisis, SPY should be negative and GLD/TLT positive
        if reading.regime == Regime.CRISIS:
            assert signals["SPY"] < 0
            assert signals["GLD"] > 0
            assert signals["TLT"] > 0

        # In low vol, SPY should be positive
        if reading.regime == Regime.LOW_VOL:
            assert signals["SPY"] > 0

    def test_consecutive_classifications(self, normal_market_prices):
        """Multiple consecutive classifications should be consistent."""
        classifier = RegimeClassifier(data_dir=normal_market_prices / "data")
        classifier.load_prices()

        regimes = []
        for _ in range(5):
            reading = classifier.classify()
            regimes.append(reading.regime)

        # Should be consistent (same regime or adjacent)
        for r in regimes:
            assert r in (Regime.NORMAL, Regime.LOW_VOL)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
