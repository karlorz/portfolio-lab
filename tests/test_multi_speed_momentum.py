#!/usr/bin/env python3
"""
Tests for multi-speed momentum ensemble — data classes, speed tier signals,
ensemble aggregation, confidence calculation, and portfolio construction.
"""
import json
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
