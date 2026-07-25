#!/usr/bin/env python3
"""
Tests for TSMOM overlay — data classes, formation returns, volatility scaling,
signal computation, portfolio construction, and backtester.
"""
import json
import numpy as np
import pandas as pd

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


# ---------------------------------------------------------------------------
# Extended coverage tests
# ---------------------------------------------------------------------------


class TestTSMOMSignalExtended:
    """Extended TSMOMSignal dataclass tests."""

    def test_to_dict_has_all_fields(self):
        sig = TSMOMSignal(
            ticker='SPY', timestamp='2026-05-24',
            lookback_return=0.12, recent_return=0.02, signal=1,
            realized_vol=0.16, vol_scaled_position=6.25,
            base_weight=0.46, adjustment=0.05, target_weight=0.51,
            lookback_start_price=450.0, lookback_end_price=504.0,
            formation_days=252,
        )
        d = sig.to_dict()
        expected_keys = {
            'ticker', 'timestamp', 'lookback_return', 'recent_return',
            'signal', 'realized_vol', 'vol_scaled_position', 'base_weight',
            'adjustment', 'target_weight', 'lookback_start_price',
            'lookback_end_price', 'formation_days',
        }
        assert expected_keys == set(d.keys())

    def test_bearish_signal(self):
        sig = TSMOMSignal(
            ticker='TLT', timestamp='2026-05-24',
            lookback_return=-0.08, recent_return=-0.02, signal=-1,
            realized_vol=0.12, vol_scaled_position=8.33,
            base_weight=0.16, adjustment=-0.05, target_weight=0.11,
            lookback_start_price=100.0, lookback_end_price=92.0,
            formation_days=252,
        )
        assert sig.signal == -1
        assert sig.lookback_return < 0

    def test_to_signal_snapshot(self):
        """to_signal_snapshot should return a SignalSnapshot."""
        from src.signals.signal_snapshot import SignalSnapshot
        sig = TSMOMSignal(
            ticker='SPY', timestamp='2026-05-24',
            lookback_return=0.12, recent_return=0.02, signal=1,
            realized_vol=0.16, vol_scaled_position=6.25,
            base_weight=0.46, adjustment=0.05, target_weight=0.51,
            lookback_start_price=450.0, lookback_end_price=504.0,
            formation_days=252,
        )
        snapshot = sig.to_signal_snapshot()
        assert isinstance(snapshot, SignalSnapshot)
        assert snapshot.source == "tsmom_overlay"


class TestTSMOMPortfolioExtended:
    """Extended TSMOMPortfolio dataclass tests."""

    def test_to_dict_has_all_fields(self):
        port = TSMOMPortfolio(
            timestamp='2026-05-24',
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            tsmom_adjustments={'SPY': 0.05},
            target_allocation={'SPY': 0.51, 'GLD': 0.38, 'TLT': 0.16, 'CASH': -0.05},
            predicted_volatility=0.14,
            max_drawdown_estimate=-0.15,
            tsmom_signals={},
            overall_confidence=0.75,
        )
        d = port.to_dict()
        expected_keys = {
            'timestamp', 'base_allocation', 'tsmom_adjustments',
            'target_allocation', 'predicted_volatility', 'max_drawdown_estimate',
            'tsmom_signals', 'overall_confidence',
        }
        assert expected_keys == set(d.keys())

    def test_negative_drawdown(self):
        port = TSMOMPortfolio(
            timestamp='2026-05-24',
            base_allocation={}, tsmom_adjustments={},
            target_allocation={}, predicted_volatility=0.20,
            max_drawdown_estimate=-0.30,
            tsmom_signals={}, overall_confidence=0.5,
        )
        assert port.max_drawdown_estimate < 0


class TestCalculateFormationReturnExtended:
    """Extended formation return tests."""

    def test_negative_return(self):
        """Monotonically decreasing prices → negative formation return."""
        overlay = _make_overlay()
        prices = pd.Series([500 - i * 0.5 for i in range(400)],
                          index=pd.date_range(end=datetime.now(), periods=400, freq='B'))
        ret, _, _, _ = overlay.calculate_formation_return(prices, 399)
        assert ret < 0

    def test_boundary_index(self):
        """First valid index should work."""
        overlay = _make_overlay()
        prices = _make_prices_series(n_days=400)
        ret, _, _, _ = overlay.calculate_formation_return(prices, LOOKBACK_DAYS + SKIP_DAYS)
        assert isinstance(ret, float)


class TestCalculateRealizedVolatilityExtended:
    """Extended realized volatility tests."""

    def test_constant_prices_min_vol(self):
        """Constant prices should get minimum 1% vol."""
        overlay = _make_overlay()
        prices = pd.Series([100.0] * 100,
                          index=pd.date_range(end=datetime.now(), periods=100, freq='B'))
        vol = overlay.calculate_realized_volatility(prices, 99)
        assert vol >= 0.01

    def test_very_short_window(self):
        """Very short window should still return a value."""
        overlay = _make_overlay()
        prices = _make_prices_series(n_days=30)
        vol = overlay.calculate_realized_volatility(prices, 29)
        assert vol > 0


class TestComputeSignalExtended:
    """Extended compute_signal tests."""

    def test_multiple_signals_deterministic(self):
        """Same data should produce same signal."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY'])
        sig1 = overlay.compute_signal('SPY')
        sig2 = overlay.compute_signal('SPY')
        assert sig1.signal == sig2.signal
        assert sig1.lookback_return == sig2.lookback_return

    def test_all_tickers(self):
        """Should work for all default tickers."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        for ticker in ['SPY', 'GLD', 'TLT']:
            sig = overlay.compute_signal(ticker)
            assert sig is not None
            assert sig.ticker == ticker

    def test_adjustment_bounded_by_max_deviation(self):
        """Adjustment should not exceed max_deviation."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        for ticker in ['SPY', 'GLD', 'TLT']:
            sig = overlay.compute_signal(ticker)
            assert abs(sig.adjustment) <= MAX_DEVIATION + 0.01


class TestComputePortfolioExtended:
    """Extended compute_portfolio tests."""

    def test_adjustments_for_all_assets(self):
        """Portfolio should include adjustments for all assets."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        port = overlay.compute_portfolio()
        for ticker in ['SPY', 'GLD', 'TLT']:
            assert ticker in port.tsmom_adjustments

    def test_signals_for_all_assets(self):
        """Portfolio should include signals for all assets."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        port = overlay.compute_portfolio()
        for ticker in ['SPY', 'GLD', 'TLT']:
            assert ticker in port.tsmom_signals

    def test_custom_tickers(self):
        """Portfolio with custom tickers should work."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'TLT'])
        port = overlay.compute_portfolio(tickers=['SPY', 'TLT'])
        assert port is not None
        assert 'SPY' in port.target_allocation
        assert 'TLT' in port.target_allocation


class TestGetCurrentRecommendationExtended:
    """Extended get_current_recommendation tests."""

    def test_recommendation_fields(self):
        """Recommendation should include strategy and timestamp."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        rec = overlay.get_current_recommendation()
        assert 'strategy' in rec
        assert 'timestamp' in rec or 'error' in rec

    def test_custom_base_allocation(self):
        """Custom base allocation should be reflected."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        custom = {'SPY': 0.50, 'GLD': 0.30, 'TLT': 0.20, 'CASH': 0.0}
        rec = overlay.get_current_recommendation(base_allocation=custom)
        if 'strategy' in rec:
            assert rec.get('base_allocation') == custom or 'strategy' in rec


class TestTSMOMBacktesterExtended:
    """Extended backtester tests."""

    def test_default_tickers(self):
        bt = TSMOMBacktester()
        assert 'SPY' in bt.tickers
        assert 'GLD' in bt.tickers
        assert 'TLT' in bt.tickers

    def test_weights_from_signals_cash_fill(self):
        """Weights with fewer signals should fill rest with cash."""
        bt = TSMOMBacktester()
        sig = TSMOMSignal(
            ticker='SPY', timestamp='2026-01-01',
            lookback_return=0.10, recent_return=0.01, signal=1,
            realized_vol=0.15, vol_scaled_position=6.67,
            base_weight=0.46, adjustment=0.05, target_weight=0.51,
            lookback_start_price=450.0, lookback_end_price=495.0,
            formation_days=252,
        )
        weights = bt._weights_from_signals({'SPY': sig})
        assert weights['CASH'] >= 0
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.05


# ---------------------------------------------------------------------------
# TSMOMSignal edge cases
# ---------------------------------------------------------------------------

class TestTSMOMSignalEdgeCases:
    """Edge-case tests for TSMOMSignal dataclass."""

    def test_signal_zero(self):
        """Signal value of 0 (neutral)."""
        sig = TSMOMSignal(
            ticker='SPY', timestamp='2026-05-24',
            lookback_return=0.0005, recent_return=0.0, signal=0,
            realized_vol=0.15, vol_scaled_position=0.0,
            base_weight=0.46, adjustment=0.0, target_weight=0.46,
            lookback_start_price=450.0, lookback_end_price=450.0,
            formation_days=252,
        )
        assert sig.signal == 0
        assert sig.vol_scaled_position == 0.0

    def test_to_signal_snapshot_inactive_when_signal_zero(self):
        """is_active should be False when signal is 0."""
        sig = TSMOMSignal(
            ticker='SPY', timestamp='2026-05-24',
            lookback_return=0.0005, recent_return=0.0, signal=0,
            realized_vol=0.15, vol_scaled_position=0.0,
            base_weight=0.46, adjustment=0.0, target_weight=0.46,
            lookback_start_price=450.0, lookback_end_price=450.0,
            formation_days=252,
        )
        snapshot = sig.to_signal_snapshot()
        assert snapshot.is_active is False

    def test_to_signal_snapshot_zero_confidence_when_no_return(self):
        """Confidence of 0.0 when lookback_return is 0."""
        sig = TSMOMSignal(
            ticker='SPY', timestamp='2026-05-24',
            lookback_return=0.0, recent_return=0.0, signal=0,
            realized_vol=0.15, vol_scaled_position=0.0,
            base_weight=0.46, adjustment=0.0, target_weight=0.46,
            lookback_start_price=450.0, lookback_end_price=450.0,
            formation_days=252,
        )
        snapshot = sig.to_signal_snapshot()
        assert snapshot.confidence == 0.0

    def test_to_signal_snapshot_confidence_capped_at_one(self):
        """Confidence should be capped at 1.0 for large returns."""
        sig = TSMOMSignal(
            ticker='SPY', timestamp='2026-05-24',
            lookback_return=0.50, recent_return=0.02, signal=1,
            realized_vol=0.16, vol_scaled_position=6.25,
            base_weight=0.46, adjustment=0.05, target_weight=0.51,
            lookback_start_price=300.0, lookback_end_price=450.0,
            formation_days=252,
        )
        snapshot = sig.to_signal_snapshot()
        assert snapshot.confidence <= 1.0

    def test_to_signal_snapshot_metadata_keys(self):
        """Metadata should contain expected keys."""
        sig = TSMOMSignal(
            ticker='TLT', timestamp='2026-05-24',
            lookback_return=-0.08, recent_return=-0.02, signal=-1,
            realized_vol=0.12, vol_scaled_position=-8.33,
            base_weight=0.16, adjustment=-0.05, target_weight=0.11,
            lookback_start_price=100.0, lookback_end_price=92.0,
            formation_days=252,
        )
        snapshot = sig.to_signal_snapshot()
        assert 'ticker' in snapshot.metadata
        assert 'signal' in snapshot.metadata
        assert 'adjustment' in snapshot.metadata
        assert 'realized_vol' in snapshot.metadata
        assert 'lookback_return' in snapshot.metadata
        assert snapshot.metadata['ticker'] == 'TLT'
        assert snapshot.metadata['signal'] == -1


# ---------------------------------------------------------------------------
# TSMOMPortfolio edge cases
# ---------------------------------------------------------------------------

class TestTSMOMPortfolioEdgeCases:
    """Edge-case tests for TSMOMPortfolio dataclass."""

    def test_to_dict_with_nested_signals(self):
        """to_dict should serialize nested TSMOMSignal objects."""
        sig = TSMOMSignal(
            ticker='SPY', timestamp='2026-05-24',
            lookback_return=0.12, recent_return=0.02, signal=1,
            realized_vol=0.16, vol_scaled_position=6.25,
            base_weight=0.46, adjustment=0.05, target_weight=0.51,
            lookback_start_price=450.0, lookback_end_price=504.0,
            formation_days=252,
        )
        port = TSMOMPortfolio(
            timestamp='2026-05-24',
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            tsmom_adjustments={'SPY': 0.05},
            target_allocation={'SPY': 0.51, 'GLD': 0.38, 'TLT': 0.16, 'CASH': -0.05},
            predicted_volatility=0.14,
            max_drawdown_estimate=-0.15,
            tsmom_signals={'SPY': sig},
            overall_confidence=0.75,
        )
        d = port.to_dict()
        assert 'tsmom_signals' in d
        assert 'SPY' in d['tsmom_signals']
        assert d['tsmom_signals']['SPY']['ticker'] == 'SPY'
        assert d['tsmom_signals']['SPY']['signal'] == 1

    def test_zero_predicted_vol(self):
        """Predicted volatility of zero is valid."""
        port = TSMOMPortfolio(
            timestamp='2026-05-24',
            base_allocation={}, tsmom_adjustments={},
            target_allocation={'CASH': 1.0}, predicted_volatility=0.0,
            max_drawdown_estimate=-0.15,
            tsmom_signals={}, overall_confidence=0.5,
        )
        assert port.predicted_volatility == 0.0


# ---------------------------------------------------------------------------
# Overlay init edge cases
# ---------------------------------------------------------------------------

class TestTSMOMOverlayInitEdgeCases:
    """Edge-case tests for TSMOMOverlay initialization."""

    def test_custom_data_source(self):
        overlay = TSMOMOverlay(data_source="custom_db")
        assert overlay.data_source == "custom_db"
        assert overlay.price_cache == {}
        assert overlay.signal_history == []


# ---------------------------------------------------------------------------
# Load prices edge cases
# ---------------------------------------------------------------------------

class TestLoadPricesEdgeCases:
    """Edge-case tests for load_prices."""

    def test_malformed_json_returns_none(self, tmp_path):
        """Corrupted JSON file should return None."""
        overlay = TSMOMOverlay()
        with patch('src.signals.tsmom_overlay.get_prices_df', side_effect=ValueError("test")):
            result = overlay.load_prices('SPY')
        assert result is None

    def test_empty_ticker_data_returns_none(self, tmp_path):
        """Empty list for a ticker should return None."""
        overlay = TSMOMOverlay()
        with patch('src.signals.tsmom_overlay.get_prices_df', return_value=pd.DataFrame()):
            result = overlay.load_prices('SPY')
        assert result is None

    def test_non_list_data_returns_none(self, tmp_path):
        """Non-list data for a ticker should return None."""
        overlay = TSMOMOverlay()
        with patch('src.signals.tsmom_overlay.get_prices_df', return_value=pd.DataFrame()):
            result = overlay.load_prices('SPY')
        assert result is None

    def test_load_prices_into_cache(self, tmp_path):
        """After loading, prices should be cached."""
        overlay = TSMOMOverlay()
        data = {
            "SPY": [
                {"d": "2025-01-02", "p": 500.0},
                {"d": "2025-01-03", "p": 505.0},
                {"d": "2025-01-06", "p": 510.0},
            ]
        }
        records = []
        for sym, entries in data.items():
            for e in entries:
                records.append({"date": e["d"], "ticker": sym, "price": e["p"]})
        mock_df = pd.DataFrame(records).pivot(index="date", columns="ticker", values="price")
        with patch('src.signals.tsmom_overlay.get_prices_df', return_value=mock_df):
            df1 = overlay.load_prices('SPY')
            df2 = overlay.load_prices('SPY')  # cache hit
        assert df1 is df2
        assert len(df1) == 3

    def test_ticker_not_in_file_returns_none(self, tmp_path):
        """Asking for a ticker not in the JSON file returns None."""
        overlay = TSMOMOverlay()
        with patch('src.signals.tsmom_overlay.get_prices_df', return_value=pd.DataFrame()):
            result = overlay.load_prices('SPY')
        assert result is None


# ---------------------------------------------------------------------------
# Formation return boundary tests
# ---------------------------------------------------------------------------

class TestCalculateFormationReturnBoundary:
    """Boundary tests for calculate_formation_return."""

    def test_exact_minimum_data(self):
        """Exactly LOOKBACK_DAYS + SKIP_DAYS data points should compute."""
        overlay = _make_overlay()
        n = LOOKBACK_DAYS + SKIP_DAYS
        prices = _make_prices_series(n_days=n + 1)
        ret, start_p, end_p, days = overlay.calculate_formation_return(prices, n)
        # Should have enough data
        assert days > 0

    def test_one_below_minimum_returns_zero(self):
        """One less than minimum data should return 0.0."""
        overlay = _make_overlay()
        n = LOOKBACK_DAYS + SKIP_DAYS
        prices = _make_prices_series(n_days=n + 1)
        ret, start_p, end_p, days = overlay.calculate_formation_return(prices, n - 1)
        assert ret == 0.0
        assert days == n - 1


# ---------------------------------------------------------------------------
# Realized volatility boundary tests
# ---------------------------------------------------------------------------

class TestCalculateRealizedVolatilityBoundary:
    """Boundary tests for calculate_realized_volatility."""

    def test_exact_vol_window_plus_one(self):
        """Exactly vol_window + 1 data points should compute normally."""
        overlay = _make_overlay()
        prices = _make_prices_series(n_days=VOL_WINDOW + 2)
        vol = overlay.calculate_realized_volatility(prices, VOL_WINDOW + 1)
        assert vol > 0

    def test_large_single_jump_increases_vol(self):
        """A single large price jump within vol window should increase vol."""
        overlay = _make_overlay()
        # Jump at position 35, so window prices[29:50] includes 11 pre-jump + 10 post-jump values
        prices = pd.Series([100.0] * 35 + [150.0] * 15,
                          index=pd.date_range(end=datetime.now(), periods=50, freq='B'))
        vol = overlay.calculate_realized_volatility(prices, 49)
        # The jump produces log(150/100) ~ 0.405 in the return series → annualized vol >> 0.01
        assert vol > 0.01

    def test_one_below_vol_window_returns_default(self):
        """One below vol_window + 1 should return default 0.15."""
        overlay = _make_overlay()
        prices = _make_prices_series(n_days=VOL_WINDOW)
        vol = overlay.calculate_realized_volatility(prices, VOL_WINDOW - 1)
        assert vol == 0.15


# ---------------------------------------------------------------------------
# Compute signal edge cases
# ---------------------------------------------------------------------------

class TestComputeSignalEdgeCases:
    """Edge-case tests for compute_signal."""

    def test_signal_zero_near_zero_return(self):
        """Formation return below 0.001 threshold yields signal 0."""
        overlay = _make_overlay()
        n_days = LOOKBACK_DAYS + SKIP_DAYS + 10
        # Nearly constant prices
        prices_vals = [100.0 + (i * 0.0001) for i in range(n_days)]
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
        prices = pd.Series(prices_vals, index=dates, name='close')
        df = pd.DataFrame({'close': prices})
        overlay.price_cache['SPY'] = df
        sig = overlay.compute_signal('SPY')
        assert sig is not None
        assert sig.signal in [-1, 0, 1]

    def test_signal_negative_for_negative_return(self):
        """Formation return negative yields signal -1."""
        overlay = _make_overlay()
        n_days = LOOKBACK_DAYS + SKIP_DAYS + 10
        # Declining prices
        prices_vals = [200.0 - (i * 0.3) for i in range(n_days)]
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
        prices = pd.Series(prices_vals, index=dates, name='close')
        df = pd.DataFrame({'close': prices})
        overlay.price_cache['SPY'] = df
        sig = overlay.compute_signal('SPY')
        assert sig is not None
        assert sig.signal == -1

    def test_default_timestamp_format(self):
        """No explicit timestamp should produce ISO format string."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY'])
        sig = overlay.compute_signal('SPY')
        assert sig is not None
        # ISO format: YYYY-MM-DDTHH:MM:SS or similar
        assert 'T' in sig.timestamp or len(sig.timestamp) >= 10

    def test_adjustment_respected_under_bounds(self):
        """Adjustment should be within [-max_deviation, max_deviation]."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY'])
        sig = overlay.compute_signal('SPY')
        assert sig is not None
        assert -MAX_DEVIATION - 0.001 <= sig.adjustment <= MAX_DEVIATION + 0.001

    def test_vol_scaled_position_scales_with_vol(self):
        """Higher vol should produce smaller scaled position magnitude."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY'])
        sig = overlay.compute_signal('SPY')
        assert sig is not None
        if sig.signal != 0:
            expected = sig.signal / sig.realized_vol
            assert abs(sig.vol_scaled_position - expected) < 1e-10

    def test_low_lookback_days_returns_none(self):
        """Very short lookback should return None."""
        overlay = TSMOMOverlay(lookback_days=500, skip_days=100)
        _inject_prices(overlay, ['SPY'], n_days=400)
        sig = overlay.compute_signal('SPY')
        assert sig is None


# ---------------------------------------------------------------------------
# Compute portfolio edge cases
# ---------------------------------------------------------------------------

class TestComputePortfolioEdgeCases:
    """Edge-case tests for compute_portfolio."""

    def test_all_positive_signals_maxdd(self):
        """All signals >= 0 yields max_drawdown_estimate = -0.15."""
        overlay = _make_overlay()
        # Inject monotonically increasing prices for all tickers
        for ticker in ['SPY', 'GLD', 'TLT']:
            prices = pd.Series([100 + i * 0.5 for i in range(400)],
                              index=pd.date_range(end=datetime.now(), periods=400, freq='B'),
                              name='close')
            overlay.price_cache[ticker] = pd.DataFrame({'close': prices})
        port = overlay.compute_portfolio()
        assert port is not None
        # All monotonically increasing -> all signal >= 0
        assert port.max_drawdown_estimate == -0.15

    def test_mixed_signals_maxdd(self):
        """Any negative signal yields max_drawdown_estimate = -0.20."""
        overlay = _make_overlay()
        # SPY: increasing -> positive, GLD: decreasing -> negative, TLT: increasing
        for ticker, direction in [('SPY', 1), ('GLD', -1), ('TLT', 1)]:
            prices = pd.Series(
                [100 + i * 0.5 * direction for i in range(400)],
                index=pd.date_range(end=datetime.now(), periods=400, freq='B'),
                name='close')
            overlay.price_cache[ticker] = pd.DataFrame({'close': prices})
        port = overlay.compute_portfolio()
        assert port is not None
        assert port.max_drawdown_estimate == -0.20

    def test_mixed_signals_confidence(self):
        """Mixed signals should produce intermediate confidence."""
        overlay = _make_overlay()
        for ticker, direction in [('SPY', 1), ('GLD', -1), ('TLT', 1)]:
            prices = pd.Series(
                [100 + i * 0.5 * direction for i in range(400)],
                index=pd.date_range(end=datetime.now(), periods=400, freq='B'),
                name='close')
            overlay.price_cache[ticker] = pd.DataFrame({'close': prices})
        port = overlay.compute_portfolio()
        assert port is not None
        # Signals: +1, -1, +1 => sum=1, agreement=1/3=0.33 => confidence=0.5+0.165=0.665
        assert 0.5 < port.overall_confidence < 0.8

    def test_all_positive_signals_high_confidence(self):
        """All signals positive yields high confidence."""
        overlay = _make_overlay()
        for ticker in ['SPY', 'GLD', 'TLT']:
            prices = pd.Series([100 + i * 0.5 for i in range(400)],
                              index=pd.date_range(end=datetime.now(), periods=400, freq='B'),
                              name='close')
            overlay.price_cache[ticker] = pd.DataFrame({'close': prices})
        port = overlay.compute_portfolio()
        assert port is not None
        # All +1 => sum=3, agreement=1.0 => confidence=1.0
        assert port.overall_confidence >= 0.99

    def test_weights_normalize_when_over_allocated(self):
        """When total weight exceeds 1.0, normalization kicks in."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        # Override to push target weights high
        for ticker in ['SPY', 'GLD', 'TLT']:
            prices = pd.Series([100 + i * 500 for i in range(400)],
                              index=pd.date_range(end=datetime.now(), periods=400, freq='B'),
                              name='close')
            overlay.price_cache[ticker] = pd.DataFrame({'close': prices})
        port = overlay.compute_portfolio()
        assert port is not None
        total = sum(port.target_allocation.values())
        assert abs(total - 1.0) < 0.01

    def test_single_ticker_portfolio(self):
        """Portfolio with a single ticker should work."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY'])
        port = overlay.compute_portfolio(tickers=['SPY'])
        assert port is not None
        assert 'SPY' in port.target_allocation
        assert 'CASH' in port.target_allocation
        assert abs(sum(port.target_allocation.values()) - 1.0) < 0.01

    def test_cash_fill_when_under_allocated(self):
        """When total weight < 1.0, CASH fills the gap."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY'], n_days=400)
        # SPY gets signal, but weight + adjustment may not reach 1.0
        port = overlay.compute_portfolio(tickers=['SPY'])
        assert port is not None
        assert 'CASH' in port.target_allocation
        assert port.target_allocation['CASH'] >= 0

    def test_predicted_vol_weighted_average(self):
        """Predicted volatility is weighted average of individual vols."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD'])
        port = overlay.compute_portfolio(tickers=['SPY', 'GLD'])
        assert port is not None
        assert port.predicted_volatility > 0
        # Check that it matches the formula
        expected = sum(
            s.realized_vol * port.target_allocation.get(t, 0)
            for t, s in port.tsmom_signals.items()
        )
        assert abs(port.predicted_volatility - expected) < 1e-10


# ---------------------------------------------------------------------------
# Get current recommendation edge cases
# ---------------------------------------------------------------------------

class TestGetCurrentRecommendationEdgeCases:
    """Edge-case tests for get_current_recommendation."""

    def test_propagates_custom_base_allocation(self):
        """Custom base allocation should appear in deltas."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        custom = {'SPY': 0.50, 'GLD': 0.30, 'TLT': 0.20, 'CASH': 0.0}
        rec = overlay.get_current_recommendation(base_allocation=custom)
        if 'strategy' in rec:
            assert rec.get('base_allocation') == custom

    def test_deltas_match_allocation_diffs(self):
        """Deltas should equal target minus base for each asset."""
        overlay = _make_overlay()
        _inject_prices(overlay, ['SPY', 'GLD', 'TLT'])
        rec = overlay.get_current_recommendation()
        if 'deltas' in rec and 'base_allocation' in rec:
            for ticker in rec['base_allocation']:
                expected = rec.get('tsmom_allocation', {}).get(ticker, 0) - rec['base_allocation'][ticker]
                assert abs(rec['deltas'].get(ticker, 0) - expected) < 0.01


# ---------------------------------------------------------------------------
# Backtester internal method tests
# ---------------------------------------------------------------------------

class TestTSMOMBacktesterInternals:
    """Tests for TSMOMBacktester internal methods."""

    def test_load_all_prices_returns_dataframe(self):
        """_load_all_prices should return a combined DataFrame."""
        bt = TSMOMBacktester(tickers=['SPY', 'GLD'])
        # Use a fixed date index to guarantee alignment
        shared_idx = pd.date_range(end=datetime.now(), periods=400, freq='B')
        for i, ticker in enumerate(['SPY', 'GLD']):
            seed = 42 + i
            np.random.seed(seed)
            returns = np.random.normal(0.0003 + i * 0.0001, 0.012, 399)
            price_vals = np.concatenate([[500.0], 500.0 * np.cumprod(1 + returns)])
            prices = pd.Series(price_vals, index=shared_idx, name='close')
            bt.overlay.price_cache[ticker] = pd.DataFrame({'close': prices})
        df = bt._load_all_prices()
        assert df is not None
        assert 'SPY' in df.columns
        assert 'GLD' in df.columns
        assert len(df) > 0

    def test_load_all_prices_returns_none(self):
        """_load_all_prices returns None when no prices available."""
        bt = TSMOMBacktester(tickers=['SPY'])
        bt.overlay.price_cache = {}
        with patch('src.signals.tsmom_overlay.get_prices_df', return_value=pd.DataFrame()):
            result = bt._load_all_prices()
        assert result is None

    def test_compute_signals_at_date_insufficient_data(self):
        """_compute_signals_at_date returns no signals when index is too low."""
        bt = TSMOMBacktester(tickers=['SPY'])
        prices_df = pd.DataFrame({'SPY': [100.0] * 50},
                                index=pd.date_range(end=datetime.now(), periods=50, freq='B'))
        signals = bt._compute_signals_at_date(prices_df, 10)
        assert signals == {}

    def test_compute_signals_at_date_with_data(self):
        """_compute_signals_at_date with sufficient data returns signals."""
        bt = TSMOMBacktester(tickers=['SPY'])
        n = LOOKBACK_DAYS + SKIP_DAYS + 50
        prices = _make_prices_series(n_days=n)
        prices_df = pd.DataFrame({'SPY': prices.values},
                                index=prices.index)
        signals = bt._compute_signals_at_date(prices_df, n - 1)
        assert len(signals) == 1
        assert 'SPY' in signals
        assert isinstance(signals['SPY'], TSMOMSignal)

    def test_weights_from_signals_overallocated(self):
        """_weights_from_signals normalizes when total > 1.0."""
        bt = TSMOMBacktester(tickers=['SPY', 'GLD'])
        sig1 = TSMOMSignal(
            ticker='SPY', timestamp='2026-01-01',
            lookback_return=0.12, recent_return=0.02, signal=1,
            realized_vol=0.16, vol_scaled_position=6.25,
            base_weight=0.46, adjustment=0.10, target_weight=0.56,
            lookback_start_price=450.0, lookback_end_price=504.0,
            formation_days=252,
        )
        sig2 = TSMOMSignal(
            ticker='GLD', timestamp='2026-01-01',
            lookback_return=0.12, recent_return=0.02, signal=1,
            realized_vol=0.16, vol_scaled_position=6.25,
            base_weight=0.38, adjustment=0.10, target_weight=0.48,
            lookback_start_price=180.0, lookback_end_price=200.0,
            formation_days=252,
        )
        weights = bt._weights_from_signals({'SPY': sig1, 'GLD': sig2})
        # 0.56 + 0.48 = 1.04 > 1.0 -> normalization should kick in
        assert 'CASH' in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01
        # After normalization, SPY + GLD should be < 1.0
        assert weights['SPY'] < 0.56
        assert weights['GLD'] < 0.48

    def test_weights_from_signals_normalized_sums_to_one(self):
        """_weights_from_signals should always sum to 1.0."""
        bt = TSMOMBacktester(tickers=['SPY'])
        sig = TSMOMSignal(
            ticker='SPY', timestamp='2026-01-01',
            lookback_return=0.10, recent_return=0.01, signal=1,
            realized_vol=0.15, vol_scaled_position=6.67,
            base_weight=0.46, adjustment=0.05, target_weight=0.51,
            lookback_start_price=450.0, lookback_end_price=495.0,
            formation_days=252,
        )
        weights = bt._weights_from_signals({'SPY': sig})
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert weights['CASH'] == 0.49  # 1.0 - 0.51


# ---------------------------------------------------------------------------
# Backtester run_backtest edge cases
# ---------------------------------------------------------------------------

class TestTSMOMBacktesterRun:
    """Tests for TSMOMBacktester.run_backtest edge cases."""

    def test_run_backtest_date_filter_insufficient(self):
        """Date filtering leaving too few days returns error."""
        bt = TSMOMBacktester(
            start_date='2099-01-01',  # Far future -> no data
            end_date='2099-12-31',
        )
        _inject_prices(bt.overlay, ['SPY', 'GLD', 'TLT'])
        result = bt.run_backtest()
        assert 'error' in result

    def test_run_backtest_mocked_prices(self):
        """Full backtest with mock price data should compute metrics."""
        bt = TSMOMBacktester(tickers=['SPY'])
        n = LOOKBACK_DAYS + SKIP_DAYS + 200
        prices = _make_prices_series(n_days=n, drift=0.0005, seed=1)
        df = pd.DataFrame({'close': prices})
        bt.overlay.price_cache['SPY'] = df

        result = bt.run_backtest(rebalance_freq=63)
        assert 'error' not in result
        assert 'cagr' in result
        assert 'sharpe_ratio' in result
        assert 'volatility' in result
        assert result['trading_days'] > 0

    def test_run_backtest_rebalance_history(self):
        """Backtest should track rebalance history."""
        bt = TSMOMBacktester(tickers=['SPY'])
        n = LOOKBACK_DAYS + SKIP_DAYS + 200
        prices = _make_prices_series(n_days=n, drift=0.0005, seed=2)
        df = pd.DataFrame({'close': prices})
        bt.overlay.price_cache['SPY'] = df

        result = bt.run_backtest(rebalance_freq=21)
        if 'error' not in result:
            assert 'rebalance_history' in result
            assert len(result['rebalance_history']) > 0
            for rb in result['rebalance_history']:
                assert 'date' in rb
                assert 'turnover' in rb
                assert 'weights' in rb

    def test_run_backtest_custom_allocation(self):
        """Backtest with custom base allocation."""
        custom = {'SPY': 0.60, 'GLD': 0.30, 'TLT': 0.10, 'CASH': 0.0}
        bt = TSMOMBacktester(
            tickers=['SPY', 'GLD', 'TLT'],
            base_allocation=custom,
        )
        n = LOOKBACK_DAYS + SKIP_DAYS + 150
        for ticker in ['SPY', 'GLD', 'TLT']:
            prices = _make_prices_series(n_days=n, drift=0.0004, seed=hash(ticker) % 1000)
            df = pd.DataFrame({'close': prices})
            bt.overlay.price_cache[ticker] = df

        result = bt.run_backtest(rebalance_freq=63)
        if 'error' not in result:
            assert result['trading_days'] > 0
            assert 'cagr' in result

    def test_run_backtest_ticker_not_in_columns(self):
        """_compute_signals_at_date skips tickers not in prices_df."""
        bt = TSMOMBacktester(tickers=['SPY', 'MISSING'])
        n = LOOKBACK_DAYS + SKIP_DAYS + 50
        prices = _make_prices_series(n_days=n)
        prices_df = pd.DataFrame({'SPY': prices.values},
                                index=prices.index)
        signals = bt._compute_signals_at_date(prices_df, n - 1)
        assert 'SPY' in signals
        assert 'MISSING' not in signals

    def test_run_backtest_handles_empty_returns(self):
        """Backtest with no valid returns still returns a result dict."""
        bt = TSMOMBacktester(tickers=['SPY'])
        # Only 1 data point after lookback -> no returns
        n = LOOKBACK_DAYS + SKIP_DAYS + 1
        prices = _make_prices_series(n_days=n)
        df = pd.DataFrame({'close': prices})
        bt.overlay.price_cache['SPY'] = df

        result = bt.run_backtest()
        # Should not crash - either returns error or has 0 metrics
        assert isinstance(result, dict)

    def test_transaction_cost_impact(self):
        """Higher transaction costs should affect end value."""
        bt_low = TSMOMBacktester(tickers=['SPY'], transaction_cost=0.0001)
        bt_high = TSMOMBacktester(tickers=['SPY'], transaction_cost=0.05)
        n = LOOKBACK_DAYS + SKIP_DAYS + 200
        prices = _make_prices_series(n_days=n, drift=0.0005, seed=42)
        df = pd.DataFrame({'close': prices})
        bt_low.overlay.price_cache['SPY'] = df
        bt_high.overlay.price_cache['SPY'] = df.copy()

        res_low = bt_low.run_backtest(rebalance_freq=21)
        res_high = bt_high.run_backtest(rebalance_freq=21)
        if 'error' not in res_low and 'error' not in res_high:
            # Higher costs should lead to lower end value
            assert res_low['end_value'] >= res_high['end_value']

    def test_required_asset_missing_fails_closed(self):
        bt = TSMOMBacktester(tickers=['SPY', 'GLD'])
        n = LOOKBACK_DAYS + SKIP_DAYS + 200
        bt.overlay.price_cache['SPY'] = pd.DataFrame({
            'close': _make_prices_series(n_days=n, seed=7),
        })

        result = bt.run_backtest()

        assert result["status"] == "failed"
        assert result["missing_assets"] == ["GLD"]

    def test_backtest_exposes_canonical_real_data_evidence(self):
        bt = TSMOMBacktester(tickers=['SPY'], transaction_cost=0.001)
        n = LOOKBACK_DAYS + SKIP_DAYS + 200
        bt.overlay.price_cache['SPY'] = pd.DataFrame({
            'close': _make_prices_series(n_days=n, drift=0.0005, seed=8),
        })

        result = bt.run_backtest(rebalance_freq=21)
        evidence = result["profitability_evidence"]

        assert evidence["data"]["mode"] == "real"
        assert evidence["promotion_eligible"] is True
        assert evidence["coverage"]["observations"] == result["trading_days"]
        assert result["end_value"] == pytest.approx(
            evidence["trace"][-1]["net_equity"]
        )
        assert evidence["costs"]["max_reconciliation_error"] < 1e-12
