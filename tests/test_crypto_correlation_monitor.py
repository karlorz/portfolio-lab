#!/usr/bin/env python3
"""
Tests for crypto_correlation_monitor.py — constants, CorrelationMetrics dataclass,
correlation/volatility calculation, regime detection, allocation signal logic,
and signal explanation.
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.crypto_correlation_monitor import (
    CRYPTO_SYMBOLS,
    BENCHMARK,
    CORRELATION_WINDOW,
    ALERT_THRESHOLD_LOW,
    ALERT_THRESHOLD_HIGH,
    CorrelationMetrics,
    CryptoCorrelationMonitor,
)


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    def test_crypto_symbols(self):
        assert 'BTC' in CRYPTO_SYMBOLS
        assert 'ETH' in CRYPTO_SYMBOLS
        assert CRYPTO_SYMBOLS['BTC'] == 'IBIT'

    def test_benchmark(self):
        assert BENCHMARK == 'SPY'

    def test_correlation_window(self):
        assert CORRELATION_WINDOW == 30

    def test_thresholds(self):
        assert ALERT_THRESHOLD_LOW == 0.25
        assert ALERT_THRESHOLD_HIGH == 0.50


# ---------------------------------------------------------------------------
# CorrelationMetrics Tests
# ---------------------------------------------------------------------------

class TestCorrelationMetrics:

    def test_to_dict(self):
        m = CorrelationMetrics(
            timestamp="2026-05-14",
            symbol="IBIT",
            benchmark="SPY",
            correlation_30d=0.15,
            correlation_60d=0.20,
            correlation_90d=0.25,
            btc_price=60000,
            spy_price=450,
            btc_volatility_30d=0.45,
            spy_volatility_30d=0.15,
            regime="low_corr",
            allocation_signal="consider",
        )
        d = m.to_dict()
        assert d['symbol'] == 'IBIT'
        assert d['correlation_30d'] == 0.15
        assert d['regime'] == 'low_corr'


# ---------------------------------------------------------------------------
# calculate_correlation Tests
# ---------------------------------------------------------------------------

class TestCalculateCorrelation:

    def test_returns_float(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        prices1 = list(range(100, 150))
        prices2 = list(range(200, 250))
        corr = monitor.calculate_correlation(prices1, prices2, 30)
        assert isinstance(corr, float)

    def test_perfect_positive(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        prices = list(range(100, 200))
        corr = monitor.calculate_correlation(prices, prices, 30)
        assert corr == pytest.approx(1.0)

    def test_negative_correlation(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        rng = np.random.RandomState(42)
        returns = rng.normal(0, 0.01, 50)
        prices1 = (100 * np.cumprod(1 + returns)).tolist()
        prices2 = (100 * np.cumprod(1 - returns)).tolist()
        corr = monitor.calculate_correlation(prices1, prices2, 30)
        assert corr < -0.5

    def test_short_data_returns_zero(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        corr = monitor.calculate_correlation([1, 2], [3, 4], 30)
        assert corr == 0.0

    def test_constant_prices_returns_zero(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        prices = [100.0] * 50
        corr = monitor.calculate_correlation(prices, prices, 30)
        assert corr == 0.0  # std is 0

    def test_different_lengths(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        rng = np.random.RandomState(42)
        prices1 = (100 * np.cumprod(1 + rng.normal(0, 0.01, 50))).tolist()
        prices2 = (200 * np.cumprod(1 + rng.normal(0, 0.01, 60))).tolist()
        corr = monitor.calculate_correlation(prices1, prices2, 30)
        assert -1 <= corr <= 1


# ---------------------------------------------------------------------------
# calculate_volatility Tests
# ---------------------------------------------------------------------------

class TestCalculateVolatility:

    def test_returns_float(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        rng = np.random.RandomState(42)
        prices = (100 * np.cumprod(1 + rng.normal(0, 0.01, 50))).tolist()
        vol = monitor.calculate_volatility(prices, 30)
        assert isinstance(vol, float)

    def test_positive(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        rng = np.random.RandomState(42)
        prices = (100 * np.cumprod(1 + rng.normal(0, 0.01, 50))).tolist()
        vol = monitor.calculate_volatility(prices, 30)
        assert vol > 0

    def test_short_data_returns_zero(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        vol = monitor.calculate_volatility([100, 101], 30)
        assert vol == 0.0

    def test_annualized(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        rng = np.random.RandomState(42)
        daily_vol = 0.02
        prices = (100 * np.cumprod(1 + rng.normal(0, daily_vol, 100))).tolist()
        vol = monitor.calculate_volatility(prices, 50)
        expected = daily_vol * np.sqrt(252)
        assert vol == pytest.approx(expected, rel=0.3)


# ---------------------------------------------------------------------------
# determine_regime Tests
# ---------------------------------------------------------------------------

class TestDetermineRegime:

    def test_low_corr(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        assert monitor.determine_regime(0.10, 0.15, 0.20) == 'low_corr'

    def test_moderate(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        assert monitor.determine_regime(0.30, 0.35, 0.40) == 'moderate'

    def test_high_corr(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        assert monitor.determine_regime(0.55, 0.60, 0.65) == 'high_corr'

    def test_boundary_025(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        # avg = 0.25 → moderate (not < 0.25)
        assert monitor.determine_regime(0.25, 0.25, 0.25) == 'moderate'
        # avg = 0.24 → low_corr
        assert monitor.determine_regime(0.24, 0.24, 0.24) == 'low_corr'

    def test_boundary_050(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        # avg = 0.50 → high_corr (not < 0.50)
        assert monitor.determine_regime(0.50, 0.50, 0.50) == 'high_corr'
        # avg = 0.49 → moderate
        assert monitor.determine_regime(0.49, 0.49, 0.49) == 'moderate'


# ---------------------------------------------------------------------------
# allocation_signal Tests
# ---------------------------------------------------------------------------

class TestAllocationSignal:

    def test_consider(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        assert monitor.allocation_signal('low_corr', 0.20) == 'consider'

    def test_avoid_high_regime(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        assert monitor.allocation_signal('high_corr', 0.30) == 'avoid'

    def test_avoid_high_corr_30d(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        assert monitor.allocation_signal('moderate', 0.55) == 'avoid'

    def test_monitor(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        assert monitor.allocation_signal('moderate', 0.30) == 'monitor'

    def test_low_corr_but_above_threshold(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        # low_corr regime but corr_30d >= 0.25
        assert monitor.allocation_signal('low_corr', 0.26) == 'monitor'


# ---------------------------------------------------------------------------
# _get_signal_explanation Tests
# ---------------------------------------------------------------------------

class TestSignalExplanation:

    def test_consider_explanation(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        m = CorrelationMetrics(
            timestamp="2026-05-14", symbol="IBIT", benchmark="SPY",
            correlation_30d=0.15, correlation_60d=0.20, correlation_90d=0.25,
            btc_price=60000, spy_price=450,
            btc_volatility_30d=0.45, spy_volatility_30d=0.15,
            regime="low_corr", allocation_signal="consider",
        )
        explanation = monitor._get_signal_explanation(m)
        assert "low" in explanation.lower() or "diversification" in explanation.lower()

    def test_avoid_explanation(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        m = CorrelationMetrics(
            timestamp="2026-05-14", symbol="IBIT", benchmark="SPY",
            correlation_30d=0.55, correlation_60d=0.60, correlation_90d=0.65,
            btc_price=60000, spy_price=450,
            btc_volatility_30d=0.45, spy_volatility_30d=0.15,
            regime="high_corr", allocation_signal="avoid",
        )
        explanation = monitor._get_signal_explanation(m)
        assert "elevated" in explanation.lower() or "avoid" in explanation.lower()

    def test_monitor_explanation(self):
        monitor = CryptoCorrelationMonitor.__new__(CryptoCorrelationMonitor)
        m = CorrelationMetrics(
            timestamp="2026-05-14", symbol="IBIT", benchmark="SPY",
            correlation_30d=0.35, correlation_60d=0.40, correlation_90d=0.45,
            btc_price=60000, spy_price=450,
            btc_volatility_30d=0.45, spy_volatility_30d=0.15,
            regime="moderate", allocation_signal="monitor",
        )
        explanation = monitor._get_signal_explanation(m)
        assert "monitor" in explanation.lower() or "moderate" in explanation.lower()
