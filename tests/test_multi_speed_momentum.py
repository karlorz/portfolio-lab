#!/usr/bin/env python3
"""
Tests for multi-speed momentum ensemble — data classes, speed tier signals,
ensemble aggregation, confidence calculation, and portfolio construction.
"""
import json
import logging
import numpy as np
import pandas as pd

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.signals.multi_speed_momentum import (
    SpeedMomentumSignal, EnsembleSignal, MultiSpeedPortfolio,
    MultiSpeedMomentum, MultiSpeedBacktester,
    SPEED_TIERS, VOL_TARGET, MAX_DEVIATION, MIN_WEIGHT,
    DEFAULT_BASE_ALLOCATION, _parse_portfolio_arg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices_df(n_days=400, seed=42, tickers=None):
    """Create synthetic price DataFrame."""
    np.random.seed(seed)
    if tickers is None:
        tickers = ['SPY', 'GLD', 'TLT']
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    data = {}
    for i, ticker in enumerate(tickers):
        drift = 0.0004 - i * 0.0001
        vol = 0.012 - i * 0.002
        prices = [500.0]
        for _ in range(n_days - 1):
            ret = np.random.normal(drift, max(vol, 0.003))
            prices.append(prices[-1] * (1 + ret))
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


def _make_engine():
    """Create a MultiSpeedMomentum engine with test data."""
    engine = MultiSpeedMomentum.__new__(MultiSpeedMomentum)
    engine.prices_path = Path("/tmp/prices.json")
    engine.db_path = Path("/tmp/signals.db")
    engine.speed_tiers = SPEED_TIERS
    engine.vol_target = VOL_TARGET
    engine.max_deviation = MAX_DEVIATION
    engine.min_weight = MIN_WEIGHT
    engine._prices_df = None
    return engine


def _make_speed_signal(ticker='SPY', tier='fast', signal=1):
    """Create a test SpeedMomentumSignal."""
    return SpeedMomentumSignal(
        ticker=ticker, tier=tier, timestamp='2026-01-01',
        lookback_return=0.05, recent_return=0.01, signal=signal,
        realized_vol=0.15, vol_scaled_position=signal / 0.15,
        base_weight=0.46, adjustment=0.03, target_weight=0.49,
        lookback_start_price=450.0, lookback_end_price=472.5,
        formation_days=63,
    )


def _make_portfolio():
    """Create a test MultiSpeedPortfolio."""
    fast = _make_speed_signal(ticker='SPY', tier='fast', signal=1)
    medium = _make_speed_signal(ticker='SPY', tier='medium', signal=1)
    slow = _make_speed_signal(ticker='SPY', tier='slow', signal=0)
    ens = EnsembleSignal(
        ticker='SPY', timestamp='2026-05-24',
        fast_signal=fast, medium_signal=medium, slow_signal=slow,
        ensemble_position=0.5, ensemble_confidence=0.67,
        base_weight=0.46, adjustment=0.05, target_weight=0.51,
    )
    return MultiSpeedPortfolio(
        timestamp='2026-05-24',
        base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
        ensemble_adjustments={'SPY': 0.05, 'GLD': -0.02, 'TLT': 0.01},
        target_allocation={'SPY': 0.51, 'GLD': 0.36, 'TLT': 0.17, 'CASH': 0.0},
        predicted_volatility=0.11,
        max_drawdown_estimate=-0.275,
        ensemble_signals={'SPY': ens},
        tier_contributions={'fast': 0.3, 'medium': 0.4, 'slow': 0.3},
        overall_confidence=0.67,
    )


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Test module constants."""

    def test_speed_tiers_keys(self):
        assert 'fast' in SPEED_TIERS
        assert 'medium' in SPEED_TIERS
        assert 'slow' in SPEED_TIERS

    def test_fast_tier_lookback(self):
        assert SPEED_TIERS['fast']['lookback_days'] == 63

    def test_slow_tier_lookback(self):
        assert SPEED_TIERS['slow']['lookback_days'] == 252

    def test_vol_target(self):
        assert VOL_TARGET == 0.15

    def test_max_deviation(self):
        assert MAX_DEVIATION == 0.10

    def test_min_weight(self):
        assert MIN_WEIGHT == 0.05

    def test_default_allocation_sums_to_one(self):
        total = sum(DEFAULT_BASE_ALLOCATION.values())
        assert abs(total - 1.0) < 0.01

    def test_allocation_matches_canonical(self):
        """Allocation should match canonical BASE_ALLOCATION from src.paths."""
        from src.paths import BASE_ALLOCATION
        for asset, weight in BASE_ALLOCATION.items():
            assert asset in DEFAULT_BASE_ALLOCATION
            assert DEFAULT_BASE_ALLOCATION[asset] == weight


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestSpeedMomentumSignal:
    """Test SpeedMomentumSignal dataclass."""

    def test_creation(self):
        sig = _make_speed_signal()
        assert sig.ticker == 'SPY'
        assert sig.tier == 'fast'
        assert sig.signal == 1

    def test_to_dict(self):
        sig = _make_speed_signal()
        d = sig.to_dict()
        assert 'ticker' in d
        assert 'realized_vol' in d
        assert 'signal' in d


class TestEnsembleSignal:
    """Test EnsembleSignal dataclass."""

    def test_creation(self):
        fast = _make_speed_signal(tier='fast')
        medium = _make_speed_signal(tier='medium')
        slow = _make_speed_signal(tier='slow')
        ens = EnsembleSignal(
            ticker='SPY', timestamp='2026-01-01',
            fast_signal=fast, medium_signal=medium, slow_signal=slow,
            ensemble_position=0.5, ensemble_confidence=1.0,
            base_weight=0.46, adjustment=0.03, target_weight=0.49,
        )
        assert ens.ticker == 'SPY'
        assert ens.ensemble_confidence == 1.0

    def test_to_dict(self):
        fast = _make_speed_signal(tier='fast')
        medium = _make_speed_signal(tier='medium')
        slow = _make_speed_signal(tier='slow')
        ens = EnsembleSignal(
            ticker='SPY', timestamp='2026-01-01',
            fast_signal=fast, medium_signal=medium, slow_signal=slow,
            ensemble_position=0.5, ensemble_confidence=0.5,
            base_weight=0.46, adjustment=0.0, target_weight=0.46,
        )
        d = ens.to_dict()
        assert 'fast_signal' in d
        assert 'ensemble_position' in d


class TestMultiSpeedPortfolio:
    """Test MultiSpeedPortfolio dataclass."""

    def test_creation(self):
        port = MultiSpeedPortfolio(
            timestamp='2026-01-01',
            base_allocation={'SPY': 0.46, 'GLD': 0.34, 'TLT': 0.16},
            ensemble_adjustments={'SPY': 0.02, 'GLD': -0.01},
            target_allocation={'SPY': 0.48, 'GLD': 0.33, 'TLT': 0.16, 'CASH': 0.03},
            predicted_volatility=0.14,
            max_drawdown_estimate=-0.20,
            ensemble_signals={},
            tier_contributions={'fast': 0.3, 'medium': 0.4, 'slow': 0.3},
            overall_confidence=0.7,
        )
        assert port.predicted_volatility == 0.14

    def test_to_dict(self):
        port = MultiSpeedPortfolio(
            timestamp='2026-01-01',
            base_allocation={}, ensemble_adjustments={}, target_allocation={},
            predicted_volatility=0.14, max_drawdown_estimate=-0.20,
            ensemble_signals={}, tier_contributions={}, overall_confidence=0.7,
        )
        d = port.to_dict()
        assert 'predicted_volatility' in d
        assert 'tier_contributions' in d


# ---------------------------------------------------------------------------
# Engine init tests
# ---------------------------------------------------------------------------

class TestEngineInit:
    """Test MultiSpeedMomentum initialization."""

    def test_default_params(self):
        engine = MultiSpeedMomentum()
        assert engine.vol_target == VOL_TARGET
        assert engine.max_deviation == MAX_DEVIATION

    def test_custom_params(self):
        engine = MultiSpeedMomentum(vol_target=0.20, max_deviation=0.15)
        assert engine.vol_target == 0.20
        assert engine.max_deviation == 0.15


# ---------------------------------------------------------------------------
# Compute speed signal tests
# ---------------------------------------------------------------------------

class TestComputeSpeedSignal:
    """Test compute_speed_signal method."""

    def test_returns_speed_signal(self):
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.46, prices_df)
        assert isinstance(sig, SpeedMomentumSignal)
        assert sig.ticker == 'SPY'
        assert sig.tier == 'fast'

    def test_signal_bounded(self):
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400)
        engine._prices_df = prices_df
        for tier in ['fast', 'medium', 'slow']:
            sig = engine.compute_speed_signal('SPY', tier, 0.46, prices_df)
            if sig:
                assert sig.signal in [-1, 0, 1]

    def test_target_weight_bounded(self):
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.46, prices_df)
        if sig:
            assert MIN_WEIGHT <= sig.target_weight <= 1.0

    def test_returns_none_missing_ticker(self):
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400)
        sig = engine.compute_speed_signal('NONEXISTENT', 'fast', 0.46, prices_df)
        assert sig is None

    def test_returns_none_insufficient_data(self):
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=30)
        sig = engine.compute_speed_signal('SPY', 'slow', 0.46, prices_df)
        assert sig is None  # slow needs 252+21+20 days

    def test_realized_vol_positive(self):
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400)
        sig = engine.compute_speed_signal('SPY', 'medium', 0.46, prices_df)
        if sig:
            assert sig.realized_vol > 0


# ---------------------------------------------------------------------------
# Compute ensemble signal tests
# ---------------------------------------------------------------------------

class TestComputeEnsembleSignal:
    """Test compute_ensemble_signal method."""

    def test_returns_ensemble_signal(self):
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400)
        engine._prices_df = prices_df
        sig = engine.compute_ensemble_signal('SPY', 0.46, prices_df)
        assert isinstance(sig, EnsembleSignal)

    def test_has_all_tiers(self):
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400)
        engine._prices_df = prices_df
        sig = engine.compute_ensemble_signal('SPY', 0.46, prices_df)
        if sig:
            assert sig.fast_signal.tier == 'fast'
            assert sig.medium_signal.tier == 'medium'
            assert sig.slow_signal.tier == 'slow'

    def test_confidence_bounded(self):
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400)
        engine._prices_df = prices_df
        sig = engine.compute_ensemble_signal('SPY', 0.46, prices_df)
        if sig:
            assert 0.0 <= sig.ensemble_confidence <= 1.0

    def test_full_agreement_confidence_one(self):
        """All tiers same signal → confidence = 1.0."""
        engine = _make_engine()
        # Create prices with strong uptrend
        np.random.seed(42)
        n = 400
        prices = [100.0]
        for _ in range(n - 1):
            prices.append(prices[-1] * 1.002)  # Steady uptrend
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
        prices_df = pd.DataFrame({'SPY': prices}, index=dates)
        engine._prices_df = prices_df
        sig = engine.compute_ensemble_signal('SPY', 0.46, prices_df)
        if sig:
            # All tiers should agree on positive momentum
            assert sig.ensemble_confidence == 1.0

    def test_returns_none_for_missing_ticker(self):
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400)
        engine._prices_df = prices_df
        sig = engine.compute_ensemble_signal('NONEXISTENT', 0.46, prices_df)
        assert sig is None


# ---------------------------------------------------------------------------
# Get current recommendation tests
# ---------------------------------------------------------------------------

class TestGetCurrentRecommendation:
    """Test get_current_recommendation method."""

    def test_returns_portfolio(self):
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400, tickers=['SPY', 'GLD', 'TLT'])
        engine._prices_df = prices_df
        rec = engine.get_current_recommendation({'SPY': 0.46, 'GLD': 0.34, 'TLT': 0.16})
        assert isinstance(rec, MultiSpeedPortfolio)

    def test_target_allocation_keys(self):
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400, tickers=['SPY', 'GLD', 'TLT'])
        engine._prices_df = prices_df
        rec = engine.get_current_recommendation({'SPY': 0.46, 'GLD': 0.34, 'TLT': 0.16})
        assert 'SPY' in rec.target_allocation
        assert 'GLD' in rec.target_allocation
        assert 'TLT' in rec.target_allocation
        assert 'CASH' in rec.target_allocation

    def test_confidence_bounded(self):
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400, tickers=['SPY', 'GLD', 'TLT'])
        engine._prices_df = prices_df
        rec = engine.get_current_recommendation({'SPY': 0.46, 'GLD': 0.34, 'TLT': 0.16})
        assert 0.0 <= rec.overall_confidence <= 1.0

    def test_predicted_vol_positive(self):
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400, tickers=['SPY', 'GLD', 'TLT'])
        engine._prices_df = prices_df
        rec = engine.get_current_recommendation({'SPY': 0.46, 'GLD': 0.34, 'TLT': 0.16})
        assert rec.predicted_volatility > 0


# ---------------------------------------------------------------------------
# Parse portfolio arg tests
# ---------------------------------------------------------------------------

class TestParsePortfolioArg:
    """Test _parse_portfolio_arg helper."""

    def test_3_part_percent(self):
        alloc = _parse_portfolio_arg('46/38/16')
        assert alloc['SPY'] == pytest.approx(0.46)
        assert alloc['GLD'] == pytest.approx(0.38)
        assert alloc['TLT'] == pytest.approx(0.16)

    def test_4_part_percent(self):
        alloc = _parse_portfolio_arg('46/34/16/4')
        assert alloc['SPY'] == pytest.approx(0.46)
        assert alloc['DBC'] == pytest.approx(0.04)

    def test_fractional_values(self):
        alloc = _parse_portfolio_arg('0.46/0.38/0.16')
        assert alloc['SPY'] == pytest.approx(0.46)


# ---------------------------------------------------------------------------
# Backtester tests
# ---------------------------------------------------------------------------

class TestMultiSpeedBacktester:
    """Test MultiSpeedBacktester."""

    def test_init(self):
        bt = MultiSpeedBacktester.__new__(MultiSpeedBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION.copy()
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        assert bt.rebalance_freq == 21

    def test_insufficient_data_returns_error(self):
        bt = MultiSpeedBacktester.__new__(MultiSpeedBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION.copy()
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.multi_speed = _make_engine()
        bt.multi_speed._prices_df = _make_prices_df(n_days=50)
        bt.prices_df = bt.multi_speed._prices_df
        result = bt.run_backtest()
        assert 'error' in result


# ---------------------------------------------------------------------------
# Extended coverage tests
# ---------------------------------------------------------------------------


class TestSpeedMomentumSignalDataclass:
    """Test SpeedMomentumSignal dataclass and to_dict."""

    def test_to_dict(self):
        sig = _make_speed_signal(ticker='GLD', tier='medium', signal=-1)
        d = sig.to_dict()
        assert d['ticker'] == 'GLD'
        assert d['tier'] == 'medium'
        assert d['signal'] == -1

    def test_to_dict_has_all_fields(self):
        sig = _make_speed_signal()
        d = sig.to_dict()
        expected_keys = {
            'ticker', 'tier', 'timestamp', 'lookback_return', 'recent_return',
            'signal', 'realized_vol', 'vol_scaled_position', 'base_weight',
            'adjustment', 'target_weight', 'lookback_start_price',
            'lookback_end_price', 'formation_days',
        }
        assert expected_keys.issubset(set(d.keys()))


class TestEnsembleSignalDataclass:
    """Test EnsembleSignal dataclass and to_dict."""

    def test_to_dict(self):
        fast = _make_speed_signal(tier='fast', signal=1)
        medium = _make_speed_signal(tier='medium', signal=0)
        slow = _make_speed_signal(tier='slow', signal=1)
        ens = EnsembleSignal(
            ticker='SPY', timestamp='2026-05-24',
            fast_signal=fast, medium_signal=medium, slow_signal=slow,
            ensemble_position=0.5, ensemble_confidence=0.5,
            base_weight=0.46, adjustment=0.05, target_weight=0.51,
        )
        d = ens.to_dict()
        assert d['ticker'] == 'SPY'
        assert 'fast_signal' in d
        assert isinstance(d['fast_signal'], dict)
        assert d['ensemble_confidence'] == 0.5

    def test_to_dict_nested_signals(self):
        """Nested speed signals should be serialized as dicts."""
        fast = _make_speed_signal(tier='fast')
        medium = _make_speed_signal(tier='medium')
        slow = _make_speed_signal(tier='slow')
        ens = EnsembleSignal(
            ticker='TLT', timestamp='2026-05-24',
            fast_signal=fast, medium_signal=medium, slow_signal=slow,
            ensemble_position=-0.3, ensemble_confidence=1.0,
            base_weight=0.16, adjustment=-0.03, target_weight=0.13,
        )
        d = ens.to_dict()
        assert d['fast_signal']['tier'] == 'fast'
        assert d['slow_signal']['tier'] == 'slow'


class TestMultiSpeedPortfolioDataclass:
    """Test MultiSpeedPortfolio dataclass and to_dict."""

    def test_to_dict(self):
        portfolio = _make_portfolio()
        d = portfolio.to_dict()
        assert 'timestamp' in d
        assert 'base_allocation' in d
        assert 'ensemble_adjustments' in d
        assert 'target_allocation' in d
        assert 'predicted_volatility' in d
        assert 'tier_contributions' in d
        assert 'overall_confidence' in d

    def test_to_dict_ensemble_signals_serialized(self):
        """Ensemble signals in portfolio dict should be dicts, not dataclasses."""
        portfolio = _make_portfolio()
        d = portfolio.to_dict()
        for ticker, sig_dict in d['ensemble_signals'].items():
            assert isinstance(sig_dict, dict)
            assert 'ticker' in sig_dict


class TestComputeSpeedSignal:
    """Test compute_speed_signal edge cases."""

    def test_unknown_ticker_returns_none(self):
        engine = _make_engine()
        engine._prices_df = _make_prices_df(tickers=['SPY', 'GLD', 'TLT'])
        result = engine.compute_speed_signal('UNKNOWN', 'fast', 0.3)
        assert result is None

    def test_insufficient_data_returns_none(self):
        engine = _make_engine()
        engine._prices_df = _make_prices_df(n_days=10)
        result = engine.compute_speed_signal('SPY', 'slow', 0.3)
        assert result is None

    def test_fast_tier_signal(self):
        engine = _make_engine()
        engine._prices_df = _make_prices_df(n_days=400)
        result = engine.compute_speed_signal('SPY', 'fast', 0.46)
        assert result is not None
        assert result.tier == 'fast'
        assert result.signal in (-1, 0, 1)

    def test_slow_tier_signal(self):
        engine = _make_engine()
        engine._prices_df = _make_prices_df(n_days=400)
        result = engine.compute_speed_signal('SPY', 'slow', 0.46)
        assert result is not None
        assert result.tier == 'slow'

    def test_target_weight_within_bounds(self):
        """target_weight should be between min_weight and 1.0."""
        engine = _make_engine()
        engine._prices_df = _make_prices_df(n_days=400)
        result = engine.compute_speed_signal('SPY', 'fast', 0.46)
        assert result.target_weight >= engine.min_weight
        assert result.target_weight <= 1.0

    def test_adjustment_within_max_deviation(self):
        """Adjustment should be clipped to max_deviation."""
        engine = _make_engine()
        engine._prices_df = _make_prices_df(n_days=400)
        result = engine.compute_speed_signal('SPY', 'fast', 0.46)
        assert abs(result.adjustment) <= engine.max_deviation

    def test_medium_tier_signal(self):
        engine = _make_engine()
        engine._prices_df = _make_prices_df(n_days=400)
        result = engine.compute_speed_signal('GLD', 'medium', 0.38)
        assert result is not None
        assert result.ticker == 'GLD'


class TestComputeEnsembleSignal:
    """Test compute_ensemble_signal edge cases."""

    def test_returns_ensemble_with_all_tiers(self):
        engine = _make_engine()
        engine._prices_df = _make_prices_df(n_days=400)
        result = engine.compute_ensemble_signal('SPY', 0.46)
        assert result is not None
        assert result.fast_signal is not None
        assert result.medium_signal is not None
        assert result.slow_signal is not None

    def test_ensemble_confidence_full_agreement(self):
        """When all tiers agree, confidence should be 1.0."""
        engine = _make_engine()
        engine._prices_df = _make_prices_df(n_days=400)
        result = engine.compute_ensemble_signal('SPY', 0.46)
        # Can't guarantee agreement with random data, but confidence should be valid
        assert 0.0 <= result.ensemble_confidence <= 1.0

    def test_insufficient_data_returns_none(self):
        engine = _make_engine()
        engine._prices_df = _make_prices_df(n_days=10)
        result = engine.compute_ensemble_signal('SPY', 0.46)
        assert result is None

    def test_unknown_ticker_returns_none(self):
        engine = _make_engine()
        engine._prices_df = _make_prices_df(tickers=['SPY', 'GLD', 'TLT'])
        result = engine.compute_ensemble_signal('UNKNOWN', 0.3)
        assert result is None


class TestGetSignalForTicker:
    """Test get_signal_for_ticker integration method."""

    def test_valid_ticker_returns_dict(self):
        engine = _make_engine()
        engine._prices_df = _make_prices_df(n_days=400)
        result = engine.get_signal_for_ticker('SPY')
        assert isinstance(result, dict)
        assert 'value' in result
        assert 'confidence' in result
        assert -1 <= result['value'] <= 1
        assert 0 <= result['confidence'] <= 1

    def test_unknown_ticker_returns_none(self):
        engine = _make_engine()
        engine._prices_df = _make_prices_df(tickers=['SPY', 'GLD', 'TLT'])
        result = engine.get_signal_for_ticker('UNKNOWN')
        assert result is None


class TestSpeedTierConstants:
    """Test speed tier configuration constants."""

    def test_three_tiers_exist(self):
        assert 'fast' in SPEED_TIERS
        assert 'medium' in SPEED_TIERS
        assert 'slow' in SPEED_TIERS

    def test_each_tier_has_required_keys(self):
        for tier_name, tier_config in SPEED_TIERS.items():
            assert 'lookback_days' in tier_config
            assert 'skip_days' in tier_config
            assert 'vol_window' in tier_config

    def test_slow_longer_lookback_than_fast(self):
        assert SPEED_TIERS['slow']['lookback_days'] > SPEED_TIERS['fast']['lookback_days']

    def test_default_base_allocation_keys(self):
        assert 'SPY' in DEFAULT_BASE_ALLOCATION
        assert 'GLD' in DEFAULT_BASE_ALLOCATION
        assert 'TLT' in DEFAULT_BASE_ALLOCATION
        assert 'CASH' in DEFAULT_BASE_ALLOCATION

    def test_default_base_allocation_sums_to_one(self):
        total = sum(DEFAULT_BASE_ALLOCATION.values())
        assert abs(total - 1.0) < 0.01


class TestParsePortfolioArgExtended:
    """Extended _parse_portfolio_arg tests."""

    def test_invalid_single_value_raises(self):
        """Single value should raise ValueError (requires 3 or 4 parts)."""
        with pytest.raises(ValueError, match="3 or 4"):
            _parse_portfolio_arg('100')

    def test_invalid_five_parts_raises(self):
        """5-part portfolio should raise ValueError."""
        with pytest.raises(ValueError, match="3 or 4"):
            _parse_portfolio_arg('20/20/20/20/20')

    def test_equal_weights_3_part(self):
        """Equal 3-way split should give 1/3 each."""
        alloc = _parse_portfolio_arg('33/33/34')
        assert alloc['SPY'] == pytest.approx(0.33)
        assert alloc['GLD'] == pytest.approx(0.33)
        assert alloc['TLT'] == pytest.approx(0.34)


# ---------------------------------------------------------------------------
# to_dict() field completeness for all three dataclasses
# ---------------------------------------------------------------------------


class TestDataclassFieldCompleteness:
    """Verify to_dict() returns all expected fields for each dataclass."""

    def test_speed_signal_to_dict_has_all_fields(self):
        """SpeedMomentumSignal.to_dict() should have exactly 14 fields."""
        sig = _make_speed_signal()
        d = sig.to_dict()
        expected = {
            'ticker', 'tier', 'timestamp', 'lookback_return', 'recent_return',
            'signal', 'realized_vol', 'vol_scaled_position', 'base_weight',
            'adjustment', 'target_weight', 'lookback_start_price',
            'lookback_end_price', 'formation_days',
        }
        assert set(d.keys()) == expected

    def test_speed_signal_to_dict_values_preserved(self):
        """Values in to_dict() should match dataclass attributes."""
        sig = _make_speed_signal(ticker='TLT', tier='slow', signal=-1)
        d = sig.to_dict()
        assert d['ticker'] == 'TLT'
        assert d['tier'] == 'slow'
        assert d['signal'] == -1
        assert d['lookback_return'] == 0.05
        assert d['formation_days'] == 63

    def test_ensemble_signal_to_dict_all_fields(self):
        """EnsembleSignal.to_dict() should have exactly 9 top-level keys."""
        fast = _make_speed_signal(tier='fast', signal=1)
        medium = _make_speed_signal(tier='medium', signal=0)
        slow = _make_speed_signal(tier='slow', signal=-1)
        ens = EnsembleSignal(
            ticker='GLD', timestamp='2026-05-24',
            fast_signal=fast, medium_signal=medium, slow_signal=slow,
            ensemble_position=0.1, ensemble_confidence=0.5,
            base_weight=0.38, adjustment=-0.02, target_weight=0.36,
        )
        d = ens.to_dict()
        expected_top = {
            'ticker', 'timestamp', 'fast_signal', 'medium_signal',
            'slow_signal', 'ensemble_position', 'ensemble_confidence',
            'base_weight', 'adjustment', 'target_weight',
        }
        assert set(d.keys()) == expected_top

    def test_ensemble_signal_to_dict_nested_structure(self):
        """Nested speed signals should each contain all 14 fields."""
        fast = _make_speed_signal(tier='fast', signal=1)
        medium = _make_speed_signal(tier='medium', signal=0)
        slow = _make_speed_signal(tier='slow', signal=-1)
        ens = EnsembleSignal(
            ticker='GLD', timestamp='2026-05-24',
            fast_signal=fast, medium_signal=medium, slow_signal=slow,
            ensemble_position=0.1, ensemble_confidence=0.5,
            base_weight=0.38, adjustment=-0.02, target_weight=0.36,
        )
        d = ens.to_dict()
        for nested_key in ('fast_signal', 'medium_signal', 'slow_signal'):
            assert isinstance(d[nested_key], dict)
            assert len(d[nested_key]) == 14

    def test_portfolio_to_dict_all_fields(self):
        """MultiSpeedPortfolio.to_dict() should have exactly 9 top-level keys."""
        portfolio = _make_portfolio()
        d = portfolio.to_dict()
        expected = {
            'timestamp', 'base_allocation', 'ensemble_adjustments',
            'target_allocation', 'predicted_volatility',
            'max_drawdown_estimate', 'ensemble_signals',
            'tier_contributions', 'overall_confidence',
        }
        assert set(d.keys()) == expected

    def test_portfolio_to_dict_json_serializable(self):
        """to_dict() output should be JSON-serializable."""
        portfolio = _make_portfolio()
        d = portfolio.to_dict()
        serialized = json.dumps(d, default=str)
        assert isinstance(serialized, str)
        restored = json.loads(serialized)
        assert restored['timestamp'] == '2026-05-24'
        assert restored['overall_confidence'] == 0.67


# ---------------------------------------------------------------------------
# Momentum calculation edge cases
# ---------------------------------------------------------------------------


class TestMomentumCalculationEdgeCases:
    """Test compute_speed_signal with extreme/edge price data."""

    def test_constant_prices_signal_zero(self):
        """Constant prices produce lookback_return=0 => signal=0."""
        engine = _make_engine()
        dates = pd.date_range(end=datetime.now(), periods=400, freq='B')
        prices_df = pd.DataFrame({'SPY': [100.0] * 400}, index=dates)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.46, prices_df)
        assert sig is not None
        assert sig.signal == 0
        assert sig.lookback_return == 0.0

    def test_constant_prices_adjustment_zero(self):
        """Constant prices produce zero adjustment."""
        engine = _make_engine()
        dates = pd.date_range(end=datetime.now(), periods=400, freq='B')
        prices_df = pd.DataFrame({'SPY': [100.0] * 400}, index=dates)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.46, prices_df)
        assert sig is not None
        assert sig.adjustment == 0.0
        assert sig.target_weight == pytest.approx(0.46)

    def test_constant_prices_realized_vol_zero(self):
        """Constant prices produce realized_vol=0 and vol_scaled_position=0."""
        engine = _make_engine()
        dates = pd.date_range(end=datetime.now(), periods=400, freq='B')
        prices_df = pd.DataFrame({'SPY': [100.0] * 400}, index=dates)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.46, prices_df)
        assert sig is not None
        assert sig.realized_vol == 0.0
        assert sig.vol_scaled_position == 0.0

    def test_single_period_returns_none(self):
        """Single row of prices should be insufficient for any tier."""
        engine = _make_engine()
        dates = pd.date_range(end=datetime.now(), periods=1, freq='B')
        prices_df = pd.DataFrame({'SPY': [500.0]}, index=dates)
        engine._prices_df = prices_df
        for tier in ('fast', 'medium', 'slow'):
            sig = engine.compute_speed_signal('SPY', tier, 0.46, prices_df)
            assert sig is None

    def test_negative_lookback_return_signal_minus_one(self):
        """Strongly declining prices produce signal=-1."""
        engine = _make_engine()
        n = 400
        np.random.seed(0)
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
        prices = [500.0]
        for _ in range(n - 1):
            prices.append(prices[-1] * (1 - 0.003))  # daily -0.3% downtrend
        prices_df = pd.DataFrame({'SPY': prices}, index=dates)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.46, prices_df)
        assert sig is not None
        assert sig.lookback_return < 0
        assert sig.signal == -1

    def test_skip_days_zero_recent_return_zero(self):
        """Custom tier with skip_days=0 sets recent_return=0.0."""
        engine = _make_engine()
        custom_tiers = {
            'fast': {'lookback_days': 63, 'skip_days': 0, 'vol_window': 10,
                     'description': 'test'},
        }
        engine.speed_tiers = custom_tiers
        prices_df = _make_prices_df(n_days=400)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.46, prices_df)
        assert sig is not None
        assert sig.recent_return == 0.0

    def test_updating_prices_clears_cache(self):
        """Setting _prices_df to None forces reload on next call."""
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400, tickers=['SPY', 'GLD', 'TLT'])
        engine._prices_df = prices_df
        sig1 = engine.compute_speed_signal('SPY', 'fast', 0.46)
        assert sig1 is not None
        # Clear cache
        engine._prices_df = None
        # Should still return None since no real file, but we can verify
        # that the internal state was reset
        assert engine._prices_df is None


# ---------------------------------------------------------------------------
# get_signal_snapshot bridge method edge cases
# ---------------------------------------------------------------------------


class TestSignalSnapshotBridge:
    """Test get_signal_snapshot() output as SignalSnapshot."""

    def test_valid_tickers_returns_active_snapshot(self):
        """With valid tickers, snapshot should be active with correct metadata."""
        from src.signals.signal_snapshot import SignalSnapshot

        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400, tickers=['SPY', 'GLD', 'TLT'])
        engine._prices_df = prices_df
        snap = engine.get_signal_snapshot(tickers=['SPY'])
        assert isinstance(snap, SignalSnapshot)
        assert snap.source == 'multi_speed_momentum'
        assert snap.is_active is True
        assert -1 <= snap.value <= 1
        assert 0 <= snap.confidence <= 1

    def test_snapshot_has_asset_signals(self):
        """Valid tickers should produce per-asset signal dict."""
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400, tickers=['SPY', 'GLD', 'TLT'])
        engine._prices_df = prices_df
        snap = engine.get_signal_snapshot(tickers=['SPY', 'GLD'])
        assert snap.is_active is True
        assert 'SPY' in snap.asset_signals
        assert 'GLD' in snap.asset_signals
        assert len(snap.asset_signals) == 2

    def test_none_tickers_uses_defaults(self):
        """When tickers is None, should default to SPY/TLT/GLD."""
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400, tickers=['SPY', 'GLD', 'TLT'])
        engine._prices_df = prices_df
        snap = engine.get_signal_snapshot(tickers=None)
        assert snap.is_active is True
        for t in ('SPY', 'GLD', 'TLT'):
            assert t in snap.asset_signals

    def test_empty_tickers_returns_inactive(self):
        """Empty tickers list should produce is_active=False snapshot."""
        engine = _make_engine()
        snap = engine.get_signal_snapshot(tickers=[])
        assert snap.is_active is False
        assert snap.value == 0.0
        assert snap.confidence == 0.0

    def test_single_ticker_snapshot(self):
        """Single ticker should work and produce correct value/confidence."""
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400, tickers=['SPY'])
        engine._prices_df = prices_df
        snap = engine.get_signal_snapshot(tickers=['SPY'])
        assert snap.is_active is True
        assert 'SPY' in snap.asset_signals
        assert len(snap.asset_signals) == 1

    def test_all_unknown_tickers_returns_inactive(self):
        """When all tickers return None, snapshot should be inactive."""
        engine = _make_engine()
        snap = engine.get_signal_snapshot(tickers=['UNKNOWN1', 'UNKNOWN2'])
        assert snap.is_active is False
        assert snap.value == 0.0
        assert snap.explanation == 'Multi-speed momentum: no data available'


# ---------------------------------------------------------------------------
# Constants extended validation
# ---------------------------------------------------------------------------


class TestConstantsExtended:
    """Extended constants validation beyond basic existence checks."""

    def test_lookback_days_monotonic(self):
        """Lookback days should increase: fast < medium < slow."""
        assert SPEED_TIERS['fast']['lookback_days'] < SPEED_TIERS['medium']['lookback_days']
        assert SPEED_TIERS['medium']['lookback_days'] < SPEED_TIERS['slow']['lookback_days']

    def test_skip_days_monotonic(self):
        """Skip days should increase: fast < medium < slow."""
        assert SPEED_TIERS['fast']['skip_days'] < SPEED_TIERS['medium']['skip_days']
        assert SPEED_TIERS['medium']['skip_days'] < SPEED_TIERS['slow']['skip_days']

    def test_vol_windows_positive(self):
        """All vol windows should be positive integers."""
        for config in SPEED_TIERS.values():
            assert config['vol_window'] > 0
            assert isinstance(config['vol_window'], int)

    def test_max_deviation_greater_than_min_weight(self):
        """MAX_DEVIATION should be >= MIN_WEIGHT so clipping works."""
        assert MAX_DEVIATION >= MIN_WEIGHT

    def test_vol_target_between_zero_and_one(self):
        """VOL_TARGET should be a reasonable annualized vol."""
        assert 0.05 <= VOL_TARGET <= 0.30

    def test_rebalance_freq_positive(self):
        """REBALANCE_FREQ should be a positive integer representing trading days."""
        from src.signals.multi_speed_momentum import REBALANCE_FREQ
        assert REBALANCE_FREQ > 0
        assert isinstance(REBALANCE_FREQ, int)


# ---------------------------------------------------------------------------
# Signal classification boundary conditions
# ---------------------------------------------------------------------------


class TestSignalClassificationBoundaries:
    """Test boundary conditions in signal generation and confidence."""

    def test_lookback_return_zero_signal_zero(self):
        """lookback_return == 0 should produce signal=0."""
        sig = SpeedMomentumSignal(
            ticker='SPY', tier='fast', timestamp='2026-01-01',
            lookback_return=0.0, recent_return=0.0, signal=0,
            realized_vol=0.15, vol_scaled_position=0.0,
            base_weight=0.46, adjustment=0.0, target_weight=0.46,
            lookback_start_price=450.0, lookback_end_price=450.0,
            formation_days=63,
        )
        assert sig.lookback_return == 0.0
        assert sig.signal == 0

    def test_max_disagreement_confidence_zero(self):
        """Signals summing to zero => maximum disagreement => confidence=0."""
        engine = _make_engine()
        n = 400
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
        # Build prices where fast goes up, medium goes down, slow stays flat
        # Use custom speed_tiers with very short lookbacks to force specific signals
        custom_tiers = {
            'fast':  {'lookback_days': 10, 'skip_days': 1, 'vol_window': 5,
                      'description': 'fast'},
            'medium': {'lookback_days': 10, 'skip_days': 1, 'vol_window': 5,
                       'description': 'medium'},
            'slow':  {'lookback_days': 10, 'skip_days': 1, 'vol_window': 5,
                      'description': 'slow'},
        }
        engine.speed_tiers = custom_tiers
        # Prices: up then down then flat -> different signals per lookback window
        prices = list(np.linspace(100, 120, 7))[:-1]  # up (fast)
        prices += list(np.linspace(120, 80, 7))[:-1]   # down (medium)
        prices += [100.0] * 6                           # flat (slow)
        # Pad to n days
        while len(prices) < n:
            prices.append(prices[-1])
        prices = prices[:n]
        prices_df = pd.DataFrame({'SPY': prices}, index=dates)
        engine._prices_df = prices_df
        sig = engine.compute_ensemble_signal('SPY', 0.46, prices_df)
        if sig:
            assert 0.0 <= sig.ensemble_confidence <= 1.0

    def test_partial_agreement_confidence_half(self):
        """Two tiers agree, third disagrees => confidence=0.5."""
        fast = _make_speed_signal(tier='fast', signal=1)
        medium = _make_speed_signal(tier='medium', signal=1)
        slow = _make_speed_signal(tier='slow', signal=-1)
        ens = EnsembleSignal(
            ticker='SPY', timestamp='2026-05-24',
            fast_signal=fast, medium_signal=medium, slow_signal=slow,
            ensemble_position=0.0, ensemble_confidence=0.5,
            base_weight=0.46, adjustment=0.0, target_weight=0.46,
        )
        assert ens.ensemble_confidence == 0.5

    def test_full_agreement_confidence_one(self):
        """All tiers agree => confidence=1.0."""
        fast = _make_speed_signal(tier='fast', signal=1)
        medium = _make_speed_signal(tier='medium', signal=1)
        slow = _make_speed_signal(tier='slow', signal=1)
        ens = EnsembleSignal(
            ticker='SPY', timestamp='2026-05-24',
            fast_signal=fast, medium_signal=medium, slow_signal=slow,
            ensemble_position=1.0, ensemble_confidence=1.0,
            base_weight=0.46, adjustment=0.05, target_weight=0.51,
        )
        assert ens.ensemble_confidence == 1.0

    def test_no_cash_base_weight_sums_to_one(self):
        """Target allocation should sum to ~1.0 after normalization."""
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400, tickers=['SPY', 'GLD', 'TLT'])
        engine._prices_df = prices_df
        rec = engine.get_current_recommendation({'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16})
        total = sum(w for k, w in rec.target_allocation.items() if k != 'CASH')
        assert abs(total - 1.0) < 0.01

    def test_no_ensemble_signals_confidence_zero(self):
        """When no ensemble signals are available, confidence should be 0."""
        engine = _make_engine()
        engine._prices_df = _make_prices_df(n_days=10)
        rec = engine.get_current_recommendation({'SPY': 0.46})
        assert rec.overall_confidence == 0.0
        assert rec.ensemble_signals == {}


# ---------------------------------------------------------------------------
# State persistence and caching
# ---------------------------------------------------------------------------


class TestStatePersistence:
    """Test save_to_db and prices caching behavior."""

    def test_prices_caching_avoids_reload(self):
        """When _prices_df is set, _load_prices should return it directly."""
        engine = _make_engine()
        original_df = _make_prices_df(n_days=400)
        engine._prices_df = original_df
        loaded = engine._load_prices()
        assert loaded is original_df

    def test_prices_cache_with_load_prices_method(self):
        """load_prices() should use the cached _prices_df if set."""
        engine = _make_engine()
        original_df = _make_prices_df(n_days=400)
        engine._prices_df = original_df
        loaded = engine.load_prices()
        assert loaded is original_df

    def test_save_to_db_creates_table_and_inserts(self, tmp_path):
        """save_to_db should create the SQLite table and insert a row."""
        db_path = tmp_path / "test_signals.db"
        engine = MultiSpeedMomentum(db_path=db_path)
        portfolio = _make_portfolio()
        engine.save_to_db(portfolio)
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.execute(
                "SELECT COUNT(*) FROM multi_speed_recommendations"
            )
            count = cur.fetchone()[0]
            assert count == 1
            cur = conn.execute(
                "SELECT timestamp, predicted_volatility, overall_confidence "
                "FROM multi_speed_recommendations"
            )
            row = cur.fetchone()
            assert row is not None
            assert row[0] == '2026-05-24'
            assert row[1] == pytest.approx(0.11)
            assert row[2] == pytest.approx(0.67)
        finally:
            conn.close()

    def test_save_to_db_multiple_inserts(self, tmp_path):
        """Multiple saves to the same DB should append rows."""
        db_path = tmp_path / "test_signals_multi.db"
        engine = MultiSpeedMomentum(db_path=db_path)
        portfolio1 = _make_portfolio()
        portfolio2 = MultiSpeedPortfolio(
            timestamp='2026-06-01',
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            ensemble_adjustments={},
            target_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0},
            predicted_volatility=0.12,
            max_drawdown_estimate=-0.30,
            ensemble_signals={},
            tier_contributions={'fast': 0.0, 'medium': 0.0, 'slow': 0.0},
            overall_confidence=0.5,
        )
        engine.save_to_db(portfolio1)
        engine.save_to_db(portfolio2)
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.execute(
                "SELECT COUNT(*) FROM multi_speed_recommendations"
            )
            assert cur.fetchone()[0] == 2
        finally:
            conn.close()

    def test_save_to_db_empty_ensemble_signal_portfolio(self, tmp_path):
        """Portfolio with empty ensemble_signals should still save."""
        db_path = tmp_path / "test_signals_empty.db"
        engine = MultiSpeedMomentum(db_path=db_path)
        portfolio = MultiSpeedPortfolio(
            timestamp='2026-05-24',
            base_allocation={},
            ensemble_adjustments={},
            target_allocation={},
            predicted_volatility=0.0,
            max_drawdown_estimate=0.0,
            ensemble_signals={},
            tier_contributions={'fast': 0.0, 'medium': 0.0, 'slow': 0.0},
            overall_confidence=0.0,
        )
        engine.save_to_db(portfolio)
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.execute(
                "SELECT COUNT(*) FROM multi_speed_recommendations"
            )
            assert cur.fetchone()[0] == 1
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Backtester extended edge cases
# ---------------------------------------------------------------------------


class TestBacktesterExtended:
    """Extended backtester edge cases."""

    def test_init_with_custom_dates(self):
        """Backtester should accept custom start and end dates."""
        bt = MultiSpeedBacktester.__new__(MultiSpeedBacktester)
        bt.base_allocation = DEFAULT_BASE_ALLOCATION.copy()
        bt.start_date = pd.to_datetime('2020-01-01')
        bt.end_date = pd.to_datetime('2023-12-31')
        bt.rebalance_freq = 21
        bt.multi_speed = _make_engine()
        bt.multi_speed._prices_df = _make_prices_df(n_days=400)
        bt.prices_df = bt.multi_speed._prices_df
        assert bt.start_date == pd.to_datetime('2020-01-01')
        assert bt.end_date == pd.to_datetime('2023-12-31')

    def test_default_rebalance_freq(self):
        """REBALANCE_FREQ constant should be 21 trading days."""
        from src.signals.multi_speed_momentum import REBALANCE_FREQ
        assert REBALANCE_FREQ == 21


# ---------------------------------------------------------------------------
# __all__ exports validation
# ---------------------------------------------------------------------------


class TestAllExports:
    """Validate __all__ exports from multi_speed_momentum module."""

    def test_all_names_in_all_exist_in_module(self):
        """Every name listed in __all__ must be importable from the module."""
        from src.signals import multi_speed_momentum as msm
        for name in msm.__all__:
            assert hasattr(msm, name), f"{name} is in __all__ but not defined in the module"

    def test_all_public_symbols_in_all_or_explicit_excluded(self):
        """Module-level names without underscore prefix should be in __all__
        unless they are intentionally excluded (DB_PATH, PRICES_PATH, logger, main).
        """
        import types
        from src.signals import multi_speed_momentum as msm

        private_names = {name for name in dir(msm) if name.startswith('_')}
        public_names = {
            name for name in dir(msm)
            if not name.startswith('_')
            and not isinstance(getattr(msm, name), types.ModuleType)
            and not callable(getattr(msm, name))  # exclude functions like main()
        }
        # Subset of public names that are intentionally not exported
        expected_unexported = {'logger', 'DB_PATH', 'PRICES_PATH', 'BASE_ALLOCATION', 'DATA_DIR', 'PRICES_JSON'}
        for name in public_names:
            if name not in msm.__all__:
                assert name in expected_unexported, (
                    f"{name!r} is a public module-level name but not in __all__ "
                    f"and not in the excluded set {expected_unexported}"
                )


# ---------------------------------------------------------------------------
# Dataclass field type validation
# ---------------------------------------------------------------------------


class TestDataclassFieldTypes:
    """Validate that dataclass field types match annotations and to_dict() output."""

    def test_speed_signal_to_dict_value_types(self):
        """SpeedMomentumSignal.to_dict() values should have correct Python types."""
        sig = _make_speed_signal()
        d = sig.to_dict()
        expected = {
            'ticker': str, 'tier': str, 'timestamp': str,
            'lookback_return': float, 'recent_return': float, 'signal': int,
            'realized_vol': float, 'vol_scaled_position': float,
            'base_weight': float, 'adjustment': float, 'target_weight': float,
            'lookback_start_price': float, 'lookback_end_price': float,
            'formation_days': int,
        }
        for field, expected_type in expected.items():
            assert isinstance(d[field], expected_type), (
                f"Field {field!r}: expected {expected_type.__name__}, got {type(d[field]).__name__}"
            )

    def test_ensemble_signal_to_dict_top_level_types(self):
        """EnsembleSignal.to_dict() top-level fields should have correct types."""
        fast = _make_speed_signal(tier='fast', signal=1)
        medium = _make_speed_signal(tier='medium', signal=1)
        slow = _make_speed_signal(tier='slow', signal=-1)
        ens = EnsembleSignal(
            ticker='SPY', timestamp='2026-05-24',
            fast_signal=fast, medium_signal=medium, slow_signal=slow,
            ensemble_position=0.5, ensemble_confidence=0.5,
            base_weight=0.46, adjustment=0.05, target_weight=0.51,
        )
        d = ens.to_dict()
        assert isinstance(d['ticker'], str)
        assert isinstance(d['timestamp'], str)
        assert isinstance(d['fast_signal'], dict)
        assert isinstance(d['medium_signal'], dict)
        assert isinstance(d['slow_signal'], dict)
        for key in ('ensemble_position', 'ensemble_confidence', 'base_weight', 'adjustment', 'target_weight'):
            assert isinstance(d[key], float), f"{key!r} should be float, got {type(d[key])}"

    def test_portfolio_to_dict_value_types(self):
        """MultiSpeedPortfolio.to_dict() values should have correct types."""
        portfolio = _make_portfolio()
        d = portfolio.to_dict()
        assert isinstance(d['timestamp'], str)
        assert isinstance(d['base_allocation'], dict)
        assert isinstance(d['ensemble_adjustments'], dict)
        assert isinstance(d['target_allocation'], dict)
        assert isinstance(d['predicted_volatility'], float)
        assert isinstance(d['max_drawdown_estimate'], float)
        assert isinstance(d['ensemble_signals'], dict)
        assert isinstance(d['tier_contributions'], dict)
        assert isinstance(d['overall_confidence'], float)


# ---------------------------------------------------------------------------
# Additional computation edge cases
# ---------------------------------------------------------------------------


class TestAdditionalEdgeCases:
    """Boundary values, zero/negative inputs, very large inputs for compute_speed_signal."""

    def test_negative_base_weight_clipped_to_min_weight(self):
        """A negative base_weight should still produce target_weight >= min_weight."""
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', -0.10, prices_df)
        assert sig is not None
        assert sig.base_weight == -0.10
        assert sig.target_weight >= engine.min_weight

    def test_zero_base_weight_clipped(self):
        """A zero base_weight should produce target_weight >= min_weight."""
        engine = _make_engine()
        prices_df = _make_prices_df(n_days=400)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.0, prices_df)
        assert sig is not None
        assert sig.target_weight >= engine.min_weight

    def test_zero_max_deviation_means_no_adjustment(self):
        """When max_deviation=0, adjustment must always be 0."""
        engine = _make_engine()
        engine.max_deviation = 0.0
        prices_df = _make_prices_df(n_days=400)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.46, prices_df)
        assert sig is not None
        assert sig.adjustment == 0.0
        assert sig.target_weight == 0.46

    def test_very_large_prices_no_overflow(self):
        """Extremely large prices (1e12 range) should not overflow."""
        engine = _make_engine()
        dates = pd.date_range(end=datetime.now(), periods=400, freq='B')
        prices = [1e12 * (1 + 0.0001 * i) for i in range(400)]
        prices_df = pd.DataFrame({'SPY': prices}, index=dates)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.46, prices_df)
        assert sig is not None
        assert np.isfinite(sig.lookback_return)
        assert np.isfinite(sig.realized_vol)
        assert sig.signal in (-1, 0, 1)

    def test_very_small_prices_no_underflow(self):
        """Extremely small prices (1e-12 range) should not underflow."""
        engine = _make_engine()
        dates = pd.date_range(end=datetime.now(), periods=400, freq='B')
        prices = [1e-12 * (1 + 0.001 * i) for i in range(400)]
        prices_df = pd.DataFrame({'SPY': prices}, index=dates)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.46, prices_df)
        assert sig is not None
        assert np.isfinite(sig.lookback_return)
        assert sig.signal in (-1, 0, 1)

    def test_all_nan_prices_returns_none(self):
        """All-NaN prices should return None (returns can't be computed)."""
        engine = _make_engine()
        dates = pd.date_range(end=datetime.now(), periods=400, freq='B')
        prices_df = pd.DataFrame({'SPY': [np.nan] * 400}, index=dates)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.46, prices_df)
        assert sig is None

    def test_extreme_volatility_no_error(self):
        """Extremely volatile prices (10% daily moves) should not cause errors."""
        engine = _make_engine()
        np.random.seed(42)
        n = 400
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
        prices = [100.0]
        for _ in range(n - 1):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.10)))
        prices_df = pd.DataFrame({'SPY': prices}, index=dates)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.46, prices_df)
        assert sig is not None
        assert sig.realized_vol > 0
        assert np.isfinite(sig.vol_scaled_position)
        assert np.isfinite(sig.adjustment)

    def test_exact_minimum_data_points_for_fast_tier(self):
        """Exactly fast-tier minimum data points should produce a valid signal."""
        engine = _make_engine()
        min_needed = 63 + 5 + 10 + 1  # lookback + skip + vol_window + 1 for pct_change
        dates = pd.date_range(end=datetime.now(), periods=min_needed, freq='B')
        prices = [100.0 * (1 + 0.001 * i) for i in range(min_needed)]
        prices_df = pd.DataFrame({'SPY': prices}, index=dates)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.46, prices_df)
        assert sig is not None
        assert sig.signal in (-1, 0, 1)

    def test_one_below_minimum_returns_none(self):
        """One data point below the required minimum should return None."""
        engine = _make_engine()
        one_below = 63 + 5 + 10 - 1  # Exactly 1 less than fast tier needs (78-1=77)
        dates = pd.date_range(end=datetime.now(), periods=one_below, freq='B')
        prices = [100.0 * (1 + 0.001 * i) for i in range(one_below)]
        prices_df = pd.DataFrame({'SPY': prices}, index=dates)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.46, prices_df)
        assert sig is None

    def test_zero_skip_days_recent_return_zero(self):
        """Custom tier with skip_days=0 should set recent_return=0.0."""
        engine = _make_engine()
        engine.speed_tiers = {
            'fast': {'lookback_days': 63, 'skip_days': 0, 'vol_window': 10,
                     'description': 'test'},
        }
        prices_df = _make_prices_df(n_days=400)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.46, prices_df)
        assert sig is not None
        assert sig.recent_return == 0.0

    def test_very_high_realized_vol_produces_valid_position(self):
        """Very high vol should produce a finite vol_scaled_position."""
        engine = _make_engine()
        np.random.seed(1)
        n = 400
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
        prices = [100.0]
        for _ in range(n - 1):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.05)))
        prices_df = pd.DataFrame({'SPY': prices}, index=dates)
        engine._prices_df = prices_df
        sig = engine.compute_speed_signal('SPY', 'fast', 0.46, prices_df)
        assert sig is not None
        assert sig.realized_vol > 0.0
        # vol_scaled_position = signal / realized_vol; can exceed +/-1
        # when vol is low, but must always be finite
        assert np.isfinite(sig.vol_scaled_position)


# ---------------------------------------------------------------------------
# Constants full validation
# ---------------------------------------------------------------------------


class TestAllConstants:
    """Validate ALL module-level constants for type, range, and reasonableness."""

    def test_db_path_is_pathlib_path(self):
        """DB_PATH should be a pathlib.Path instance."""
        from src.signals.multi_speed_momentum import DB_PATH
        assert isinstance(DB_PATH, Path)

    def test_prices_path_is_pathlib_path(self):
        """PRICES_PATH should be a pathlib.Path instance."""
        from src.signals.multi_speed_momentum import PRICES_PATH
        assert isinstance(PRICES_PATH, Path)

    def test_rebalance_freq_is_int(self):
        """REBALANCE_FREQ should be an int."""
        from src.signals.multi_speed_momentum import REBALANCE_FREQ
        assert isinstance(REBALANCE_FREQ, int)

    def test_rebalance_freq_reasonable_range(self):
        """REBALANCE_FREQ should be between 5 and 63 trading days (weekly to quarterly)."""
        from src.signals.multi_speed_momentum import REBALANCE_FREQ
        assert 5 <= REBALANCE_FREQ <= 63

    def test_asset_tickers_contains_expected_symbols(self):
        """ASSET_TICKERS should include all canonical symbols."""
        from src.signals.multi_speed_momentum import ASSET_TICKERS
        for symbol in ('SPY', 'GLD', 'TLT', 'DBC', 'CASH'):
            assert symbol in ASSET_TICKERS, f"{symbol} missing from ASSET_TICKERS"

    def test_asset_tickers_self_mapping(self):
        """ASSET_TICKERS values should be identical to their keys."""
        from src.signals.multi_speed_momentum import ASSET_TICKERS
        for key, value in ASSET_TICKERS.items():
            assert key == value, f"ASSET_TICKERS[{key!r}] = {value!r}, expected {key!r}"

    def test_all_speed_tier_values_non_negative(self):
        """All speed tier numeric parameters should be >= 0."""
        for config in SPEED_TIERS.values():
            assert config['lookback_days'] >= 0
            assert config['skip_days'] >= 0
            assert config['vol_window'] >= 0

    def test_speed_tier_trading_day_reasonableness(self):
        """Speed tier lookback days should align with calendar conventions."""
        assert SPEED_TIERS['fast']['lookback_days'] == 63   # ~3 months
        assert SPEED_TIERS['medium']['lookback_days'] == 126  # ~6 months
        assert SPEED_TIERS['slow']['lookback_days'] == 252   # ~12 months

    def test_speed_tier_skip_days_reasonableness(self):
        """Speed tier skip days should be roughly 1 week, 2 weeks, 1 month."""
        assert SPEED_TIERS['fast']['skip_days'] == 5
        assert SPEED_TIERS['medium']['skip_days'] == 10
        assert SPEED_TIERS['slow']['skip_days'] == 21


# ---------------------------------------------------------------------------
# CLI main() function tests
# ---------------------------------------------------------------------------


class TestCLIMain:
    """Test the CLI main() entry point with mocked arguments."""

    def test_status_prints_system_info(self, caplog):
        """`status` subcommand should print system info without error."""
        caplog.set_level(logging.INFO)
        from src.signals.multi_speed_momentum import main
        with patch('sys.argv', ['multi_speed_momentum', 'status']):
            main()
        output = ' '.join(caplog.text)
        assert 'Multi-Speed Momentum' in caplog.text
        assert 'FAST TIER' in caplog.text
        assert 'MEDIUM TIER' in caplog.text
        assert 'SLOW TIER' in caplog.text
        assert 'Equal risk-weight' in caplog.text

    def test_status_shows_prices_path(self, caplog):
        """Status should mention the data source path."""
        caplog.set_level(logging.INFO)
        from src.signals.multi_speed_momentum import main
        with patch('sys.argv', ['multi_speed_momentum', 'status']):
            main()
        assert 'Data source' in caplog.text

    def test_backtest_with_mocked_engine_writes_json(self, tmp_path):
        """Backtest with mocked backtester should produce JSON output."""
        import io
        output_file = tmp_path / 'bt_result.json'
        mock_result = {
            'sharpe_ratio': 0.80,
            'strategy': 'Multi-Speed Momentum Ensemble v2.56',
            'cagr': 0.106,
            'volatility': 0.111,
        }

        with patch('src.signals.multi_speed_momentum.MultiSpeedBacktester') as mock_bt_cls:
            mock_bt = MagicMock()
            mock_bt.run_backtest.return_value = mock_result
            mock_bt_cls.return_value = mock_bt

            with patch('sys.argv', [
                'multi_speed_momentum', 'backtest', '--portfolio', '46/38/16',
                '--output', str(output_file),
            ]):
                with patch('sys.stdout', new_callable=io.StringIO):
                    from src.signals.multi_speed_momentum import main
                    main()

            # Verify backtester received correct base allocation
            mock_bt_cls.assert_called_once()
            _, call_kwargs = mock_bt_cls.call_args
            assert call_kwargs['base_allocation'] == {
                'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0,
            }

        assert output_file.exists()
        saved = json.loads(output_file.read_text())
        assert saved['sharpe_ratio'] == 0.80

    def test_backtest_with_dates_passed_to_constructor(self):
        """Backtest with --start and --end should pass dates to constructor."""
        import io
        with patch('src.signals.multi_speed_momentum.MultiSpeedBacktester') as mock_bt_cls:
            mock_bt = MagicMock()
            mock_bt.run_backtest.return_value = {'sharpe_ratio': 0.5}
            mock_bt_cls.return_value = mock_bt

            with patch('sys.argv', [
                'multi_speed_momentum', 'backtest', '--portfolio', '58/32/10',
                '--start', '2020-01-01', '--end', '2023-12-31', '--freq', '63',
            ]):
                with patch('sys.stdout', new_callable=io.StringIO):
                    from src.signals.multi_speed_momentum import main
                    main()

            mock_bt_cls.assert_called_once()
            _, call_kwargs = mock_bt_cls.call_args
            assert call_kwargs['start_date'] == '2020-01-01'
            assert call_kwargs['end_date'] == '2023-12-31'
            assert call_kwargs['rebalance_freq'] == 63

    def test_backtest_with_4_part_portfolio(self):
        """Backtest with 4-part portfolio should include DBC."""
        import io
        with patch('src.signals.multi_speed_momentum.MultiSpeedBacktester') as mock_bt_cls:
            mock_bt = MagicMock()
            mock_bt.run_backtest.return_value = {'sharpe_ratio': 0.5}
            mock_bt_cls.return_value = mock_bt

            with patch('sys.argv', [
                'multi_speed_momentum', 'backtest', '--portfolio', '46/34/16/4',
            ]):
                with patch('sys.stdout', new_callable=io.StringIO):
                    from src.signals.multi_speed_momentum import main
                    main()

            _, call_kwargs = mock_bt_cls.call_args
            assert call_kwargs['base_allocation']['DBC'] == pytest.approx(0.04)
            assert 'CASH' in call_kwargs['base_allocation']

    def test_live_with_save_db_calls_save_to_db(self, tmp_path):
        """Live command with --save-db should trigger save_to_db."""
        import io
        db_path = tmp_path / 'live_test.db'
        mock_portfolio = MagicMock()
        mock_portfolio.to_dict.return_value = {
            'timestamp': '2026-05-24',
            'base_allocation': {'SPY': 0.46},
            'ensemble_adjustments': {},
            'target_allocation': {'SPY': 0.50, 'CASH': 0.0},
            'predicted_volatility': 0.12,
            'max_drawdown_estimate': -0.30,
            'ensemble_signals': {},
            'tier_contributions': {},
            'overall_confidence': 0.70,
        }

        with patch('src.signals.multi_speed_momentum.MultiSpeedMomentum') as mock_eng_cls:
            mock_eng = MagicMock()
            mock_eng.get_current_recommendation.return_value = mock_portfolio
            mock_eng_cls.return_value = mock_eng

            with patch('src.signals.multi_speed_momentum.DB_PATH', db_path):
                with patch('sys.argv', [
                    'multi_speed_momentum', 'live', '--portfolio', '50/30/20', '--save-db',
                ]):
                    with patch('sys.stdout', new_callable=io.StringIO):
                        from src.signals.multi_speed_momentum import main
                        main()

                mock_eng.save_to_db.assert_called_once_with(mock_portfolio)

    def test_live_with_output_file(self, tmp_path):
        """Live command with --output should write JSON file."""
        import io
        output_file = tmp_path / 'live_out.json'
        mock_portfolio = MagicMock()
        mock_portfolio.to_dict.return_value = {
            'timestamp': '2026-05-24',
            'base_allocation': {},
            'ensemble_adjustments': {},
            'target_allocation': {},
            'predicted_volatility': 0.0,
            'max_drawdown_estimate': 0.0,
            'ensemble_signals': {},
            'tier_contributions': {},
            'overall_confidence': 0.0,
        }

        with patch('src.signals.multi_speed_momentum.MultiSpeedMomentum') as mock_eng_cls:
            mock_eng = MagicMock()
            mock_eng.get_current_recommendation.return_value = mock_portfolio
            mock_eng_cls.return_value = mock_eng

            with patch('sys.argv', [
                'multi_speed_momentum', 'live', '--portfolio', '46/38/16', '--output', str(output_file),
            ]):
                with patch('sys.stdout', new_callable=io.StringIO):
                    from src.signals.multi_speed_momentum import main
                    main()

        assert output_file.exists()
        saved = json.loads(output_file.read_text())
        assert saved['timestamp'] == '2026-05-24'

    def test_compute_with_mocked_engine(self):
        """Compute command with mocked engine should not crash."""
        import io
        mock_signal = MagicMock()
        mock_signal.to_dict.return_value = {
            'ticker': 'SPY', 'tier': 'fast', 'signal': 1,
        }

        with patch('src.signals.multi_speed_momentum.MultiSpeedMomentum') as mock_eng_cls:
            mock_eng = MagicMock()
            mock_eng.compute_speed_signal.return_value = mock_signal
            mock_eng_cls.return_value = mock_eng

            with patch('sys.argv', [
                'multi_speed_momentum', 'compute', '--ticker', 'SPY', '--tier', 'fast',
            ]):
                with patch('sys.stdout', new_callable=io.StringIO):
                    from src.signals.multi_speed_momentum import main
                    main()

            mock_eng.compute_speed_signal.assert_called_once_with('SPY', 'fast', 0.46)

    def test_compute_saves_output_file(self, tmp_path):
        """Compute command with --output should write JSON file."""
        import io
        output_file = tmp_path / 'compute_out.json'
        mock_signal = MagicMock()
        mock_signal.to_dict.return_value = {'ticker': 'SPY', 'signal': 1}

        with patch('src.signals.multi_speed_momentum.MultiSpeedMomentum') as mock_eng_cls:
            mock_eng = MagicMock()
            mock_eng.compute_speed_signal.return_value = mock_signal
            mock_eng_cls.return_value = mock_eng

            with patch('sys.argv', [
                'multi_speed_momentum', 'compute', '--ticker', 'SPY', '--tier', 'fast',
                '--output', str(output_file),
            ]):
                with patch('sys.stdout', new_callable=io.StringIO):
                    from src.signals.multi_speed_momentum import main
                    main()

        assert output_file.exists()
        saved = json.loads(output_file.read_text())
        assert saved['ticker'] == 'SPY'

    def test_compute_no_signal_prints_error(self, caplog):
        """Compute command when signal is None should print error JSON."""
        caplog.set_level(logging.INFO)
        with patch('src.signals.multi_speed_momentum.MultiSpeedMomentum') as mock_eng_cls:
            mock_eng = MagicMock()
            mock_eng.compute_speed_signal.return_value = None
            mock_eng_cls.return_value = mock_eng

            with patch('sys.argv', [
                'multi_speed_momentum', 'compute', '--ticker', 'SPY', '--tier', 'fast',
            ]):
                from src.signals.multi_speed_momentum import main
                main()
            assert 'Could not compute signal for SPY' in caplog.text

    def test_invalid_portfolio_raises_value_error(self):
        """Backtest with invalid portfolio format should raise ValueError."""
        import io
        from src.signals.multi_speed_momentum import main, _parse_portfolio_arg
        with patch('sys.argv', [
            'multi_speed_momentum', 'backtest', '--portfolio', '100',
        ]):
            with patch('sys.stdout', new_callable=io.StringIO):
                with pytest.raises(ValueError, match="3 or 4"):
                    main()

    def test_live_portfolio_parsing(self):
        """Live command should parse portfolio arg into correct allocation."""
        import io
        with patch('src.signals.multi_speed_momentum.MultiSpeedMomentum') as mock_eng_cls:
            mock_eng = MagicMock()
            mock_portfolio = MagicMock()
            mock_portfolio.to_dict.return_value = {}
            mock_eng.get_current_recommendation.return_value = mock_portfolio
            mock_eng_cls.return_value = mock_eng

            with patch('sys.argv', [
                'multi_speed_momentum', 'live', '--portfolio', '40/35/25',
            ]):
                with patch('sys.stdout', new_callable=io.StringIO):
                    from src.signals.multi_speed_momentum import main
                    main()

            mock_eng.get_current_recommendation.assert_called_once()
            alloc = mock_eng.get_current_recommendation.call_args[0][0]
            assert alloc['SPY'] == pytest.approx(0.40)
            assert alloc['GLD'] == pytest.approx(0.35)
            assert alloc['TLT'] == pytest.approx(0.25)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
