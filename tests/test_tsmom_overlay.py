#!/usr/bin/env python3
"""
Tests for TSMOM overlay — data classes, formation returns, volatility scaling,
signal computation, portfolio construction, and backtester.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.signals.tsmom_overlay import (
    TSMOMSignal, TSMOMPortfolio, TSMOMOverlay, TSMOMBacktester,
    LOOKBACK_DAYS, SKIP_DAYS, VOL_WINDOW, MAX_DEVIATION, MIN_WEIGHT,
    DEFAULT_BASE_ALLOCATION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices_series(n_days=400, drift=0.0004, vol=0.012, seed=42):
    """Create synthetic price series."""
    np.random.seed(seed)
    prices = [500.0]
    for _ in range(n_days - 1):
        ret = np.random.normal(drift, vol)
        prices.append(prices[-1] * (1 + ret))
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    return pd.Series(prices, index=dates, name='close')


def _make_overlay(tmp_path=None):
    """Create a TSMOMOverlay with mocked price cache."""
    overlay = TSMOMOverlay.__new__(TSMOMOverlay)
    overlay.lookback_days = LOOKBACK_DAYS
    overlay.skip_days = SKIP_DAYS
    overlay.vol_window = VOL_WINDOW
    overlay.max_deviation = MAX_DEVIATION
    overlay.min_weight = MIN_WEIGHT
    overlay.data_source = "test"
    overlay.price_cache = {}
    overlay.signal_history = []
    return overlay


def _inject_prices(overlay, tickers=None, n_days=400):
    """Inject synthetic prices into overlay cache."""
    if tickers is None:
        tickers = ['SPY', 'GLD', 'TLT']
    for i, ticker in enumerate(tickers):
        prices = _make_prices_series(n_days, drift=0.0003 + i * 0.0001, seed=42 + i)
        df = pd.DataFrame({'close': prices})
        overlay.price_cache[ticker] = df


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Test module constants."""

    def test_lookback_days(self):
        assert LOOKBACK_DAYS == 252

    def test_skip_days(self):
        assert SKIP_DAYS == 21

    def test_vol_window(self):
        assert VOL_WINDOW == 20

    def test_max_deviation(self):
        assert MAX_DEVIATION == 0.10

    def test_min_weight(self):
        assert MIN_WEIGHT == 0.05

    def test_default_allocation_sums_to_one(self):
        total = sum(DEFAULT_BASE_ALLOCATION.values())
        assert abs(total - 1.0) < 0.01

    def test_default_allocation_keys(self):
        assert 'SPY' in DEFAULT_BASE_ALLOCATION
        assert 'GLD' in DEFAULT_BASE_ALLOCATION
        assert 'TLT' in DEFAULT_BASE_ALLOCATION
        assert 'CASH' in DEFAULT_BASE_ALLOCATION


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestTSMOMSignal:
    """Test TSMOMSignal dataclass."""

    def test_creation(self):
        sig = TSMOMSignal(
            ticker='SPY', timestamp=datetime.now().isoformat(),
            lookback_return=0.12, recent_return=0.02, signal=1,
            realized_vol=0.16, vol_scaled_position=6.25,
            base_weight=0.46, adjustment=0.05, target_weight=0.51,
            lookback_start_price=450.0, lookback_end_price=504.0,
            formation_days=252,
        )
        assert sig.ticker == 'SPY'
        assert sig.signal == 1
        assert sig.target_weight == 0.51

    def test_to_dict(self):
        sig = TSMOMSignal(
            ticker='GLD', timestamp='2026-01-01',
            lookback_return=0.05, recent_return=-0.01, signal=1,
            realized_vol=0.14, vol_scaled_position=7.14,
            base_weight=0.38, adjustment=-0.02, target_weight=0.36,
            lookback_start_price=180.0, lookback_end_price=189.0,
            formation_days=252,
        )
        d = sig.to_dict()
        assert d['ticker'] == 'GLD'
        assert 'signal' in d
        assert 'realized_vol' in d


class TestTSMOMPortfolio:
    """Test TSMOMPortfolio dataclass."""

    def test_creation(self):
        sig = TSMOMSignal(
            ticker='SPY', timestamp='2026-01-01',
            lookback_return=0.12, recent_return=0.02, signal=1,
            realized_vol=0.16, vol_scaled_position=6.25,
            base_weight=0.46, adjustment=0.05, target_weight=0.51,
            lookback_start_price=450.0, lookback_end_price=504.0,
            formation_days=252,
        )
        port = TSMOMPortfolio(
            timestamp='2026-01-01',
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            tsmom_adjustments={'SPY': 0.05},
            target_allocation={'SPY': 0.51, 'GLD': 0.38, 'TLT': 0.16, 'CASH': -0.05},
            predicted_volatility=0.14,
            max_drawdown_estimate=-0.15,
            tsmom_signals={'SPY': sig},
            overall_confidence=0.75,
        )
        assert port.predicted_volatility == 0.14

    def test_to_dict(self):
        port = TSMOMPortfolio(
            timestamp='2026-01-01',
            base_allocation={'SPY': 0.46},
            tsmom_adjustments={'SPY': 0.05},
            target_allocation={'SPY': 0.51, 'CASH': 0.49},
            predicted_volatility=0.14,
            max_drawdown_estimate=-0.15,
            tsmom_signals={},
            overall_confidence=0.75,
        )
        d = port.to_dict()
        assert 'base_allocation' in d
        assert 'tsmom_adjustments' in d
        assert 'target_allocation' in d
        assert 'overall_confidence' in d


# ---------------------------------------------------------------------------
# Overlay init tests
# ---------------------------------------------------------------------------

class TestTSMOMOverlayInit:
    """Test TSMOMOverlay initialization."""

    def test_default_params(self):
        overlay = TSMOMOverlay()
        assert overlay.lookback_days == LOOKBACK_DAYS
        assert overlay.skip_days == SKIP_DAYS
        assert overlay.vol_window == VOL_WINDOW

    def test_custom_params(self):
        overlay = TSMOMOverlay(lookback_days=126, skip_days=10)
        assert overlay.lookback_days == 126
        assert overlay.skip_days == 10

    def test_empty_cache(self):
        overlay = TSMOMOverlay()
        assert overlay.price_cache == {}


# ---------------------------------------------------------------------------
# Formation return tests
# ---------------------------------------------------------------------------

class TestCalculateFormationReturn:
    """Test calculate_formation_return."""

    def test_sufficient_data(self):
        overlay = _make_overlay()
        prices = _make_prices_series(n_days=400)
        ret, start_p, end_p, days = overlay.calculate_formation_return(prices, 399)
        assert isinstance(ret, float)
        assert days > 0

    def test_insufficient_data_returns_zero(self):
        overlay = _make_overlay()
        prices = _make_prices_series(n_days=50)
        ret, start_p, end_p, days = overlay.calculate_formation_return(prices, 30)
        assert ret == 0.0

    def test_positive_return(self):
        """Monotonically increasing prices → positive formation return."""
        overlay = _make_overlay()
        prices = pd.Series([100 + i * 0.5 for i in range(400)],
                          index=pd.date_range(end=datetime.now(), periods=400, freq='B'))
        ret, _, _, _ = overlay.calculate_formation_return(prices, 399)
        assert ret > 0

    def test_returns_tuple(self):
        overlay = _make_overlay()
        prices = _make_prices_series(n_days=400)
        result = overlay.calculate_formation_return(prices, 399)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Realized volatility tests
# ---------------------------------------------------------------------------

class TestCalculateRealizedVolatility:
    """Test calculate_realized_volatility."""

    def test_returns_positive_float(self):
        overlay = _make_overlay()
        prices = _make_prices_series(n_days=400)
        vol = overlay.calculate_realized_volatility(prices, 399)
        assert vol > 0

    def test_insufficient_data_returns_default(self):
        overlay = _make_overlay()
        prices = _make_prices_series(n_days=10)
        vol = overlay.calculate_realized_volatility(prices, 5)
        assert vol == 0.15  # Default

    def test_minimum_vol(self):
        """Even constant prices → min 1% vol."""
        overlay = _make_overlay()
        prices = pd.Series([100.0] * 50,
                          index=pd.date_range(end=datetime.now(), periods=50, freq='B'))
        vol = overlay.calculate_realized_volatility(prices, 49)
        assert vol >= 0.01

    def test_high_vol_series(self):
        """High-return series → higher vol."""
        overlay = _make_overlay()
        np.random.seed(99)
        returns = np.random.normal(0, 0.05, 50)
        prices = pd.Series(
            100 * np.cumprod(1 + returns),
            index=pd.date_range(end=datetime.now(), periods=50, freq='B')
        )
        vol = overlay.calculate_realized_volatility(prices, 49)
        assert vol > 0.3  # ~5% daily * sqrt(252) ≈ 79%


# ---------------------------------------------------------------------------
# Load prices tests
# ---------------------------------------------------------------------------

class TestLoadPrices:
    """Test load_prices method."""

    def test_cache_hit(self):
        overlay = _make_overlay()
        df = pd.DataFrame({'close': [100, 101, 102]})
        overlay.price_cache['SPY'] = df
        result = overlay.load_prices('SPY')
        assert result is df

    def test_no_prices_file_returns_none(self):
        overlay = _make_overlay()
        with patch('src.signals.tsmom_overlay.PRICES_PATH') as mock_path:
            mock_path.exists.return_value = False
            result = overlay.load_prices('NONEXISTENT')
        assert result is None


# ---------------------------------------------------------------------------
# Compute signal tests
# ---------------------------------------------------------------------------

class TestComputeSignal:
    """Test compute_signal method."""

    def test_returns_tsmom_signal(self):
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY'])
        sig = overlay.compute_signal('SPY')
        assert isinstance(sig, TSMOMSignal)
        assert sig.ticker == 'SPY'

    def test_signal_is_bounded(self):
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY'])
        sig = overlay.compute_signal('SPY')
        assert sig.signal in [-1, 0, 1]

    def test_target_weight_bounded(self):
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY'])
        sig = overlay.compute_signal('SPY')
        assert MIN_WEIGHT <= sig.target_weight <= 0.95

    def test_realized_vol_positive(self):
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY'])
        sig = overlay.compute_signal('SPY')
        assert sig.realized_vol > 0

    def test_returns_none_for_missing_ticker(self):
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY'])
        sig = overlay.compute_signal('NONEXISTENT')
        assert sig is None

    def test_formation_days_positive(self):
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY'])
        sig = overlay.compute_signal('SPY')
        assert sig.formation_days > 0

    def test_base_weight_from_allocation(self):
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY'])
        sig = overlay.compute_signal('SPY')
        assert sig.base_weight == DEFAULT_BASE_ALLOCATION['SPY']

    def test_consistent_timestamp(self):
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY'])
        ts = '2026-01-15T10:00:00'
        sig = overlay.compute_signal('SPY', timestamp=ts)
        assert sig.timestamp == ts


# ---------------------------------------------------------------------------
# Compute portfolio tests
# ---------------------------------------------------------------------------

class TestComputePortfolio:
    """Test compute_portfolio method."""

    def test_returns_portfolio(self):
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        port = overlay.compute_portfolio()
        assert isinstance(port, TSMOMPortfolio)

    def test_target_allocation_keys(self):
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        port = overlay.compute_portfolio()
        assert 'SPY' in port.target_allocation
        assert 'GLD' in port.target_allocation
        assert 'TLT' in port.target_allocation

    def test_confidence_bounded(self):
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        port = overlay.compute_portfolio()
        assert 0.0 <= port.overall_confidence <= 1.0

    def test_returns_none_for_empty(self):
        overlay = _make_overlay()
        port = overlay.compute_portfolio(tickers=['NONEXISTENT'])
        assert port is None

    def test_custom_base_allocation(self):
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        custom = {'SPY': 0.60, 'GLD': 0.30, 'TLT': 0.10, 'CASH': 0.0}
        port = overlay.compute_portfolio(base_allocation=custom)
        assert port.base_allocation == custom

    def test_predicted_vol_positive(self):
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        port = overlay.compute_portfolio()
        assert port.predicted_volatility > 0


# ---------------------------------------------------------------------------
# Get current recommendation tests
# ---------------------------------------------------------------------------

class TestGetCurrentRecommendation:
    """Test get_current_recommendation method."""

    def test_returns_dict(self):
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        rec = overlay.get_current_recommendation()
        assert isinstance(rec, dict)
        assert 'strategy' in rec

    def test_has_deltas(self):
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        rec = overlay.get_current_recommendation()
        assert 'deltas' in rec

    def test_has_signals(self):
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        rec = overlay.get_current_recommendation()
        assert 'signals' in rec

    def test_error_on_failure(self):
        overlay = _make_overlay()
        rec = overlay.get_current_recommendation()
        # No prices loaded → error
        assert 'error' in rec or 'strategy' in rec


# ---------------------------------------------------------------------------
# Backtester tests
# ---------------------------------------------------------------------------

class TestTSMOMBacktester:
    """Test TSMOMBacktester."""

    def test_init(self):
        bt = TSMOMBacktester()
        assert bt.tickers == ['SPY', 'GLD', 'TLT']
        assert bt.transaction_cost == 0.001

    def test_custom_params(self):
        bt = TSMOMBacktester(
            tickers=['SPY', 'TLT'],
            transaction_cost=0.002,
        )
        assert bt.tickers == ['SPY', 'TLT']
        assert bt.transaction_cost == 0.002

    def test_insufficient_data_returns_error(self):
        bt = TSMOMBacktester()
        with patch.object(bt.overlay, 'load_prices', return_value=None):
            result = bt.run_backtest()
        assert 'error' in result

    def test_weights_from_signals(self):
        """_weights_from_signals converts signals to weights."""
        bt = TSMOMBacktester()
        sig = TSMOMSignal(
            ticker='SPY', timestamp='2026-01-01',
            lookback_return=0.12, recent_return=0.02, signal=1,
            realized_vol=0.16, vol_scaled_position=6.25,
            base_weight=0.46, adjustment=0.05, target_weight=0.51,
            lookback_start_price=450.0, lookback_end_price=504.0,
            formation_days=252,
        )
        weights = bt._weights_from_signals({'SPY': sig})
        assert 'SPY' in weights
        assert 'CASH' in weights
        assert abs(sum(weights.values()) - 1.0) < 0.05


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
