#!/usr/bin/env python3
"""
Tests for trend integration module — data classes, trend signal generation,
CTA overlay allocation, vol regime detection, and replication ETF weighting.
"""
import sys
import os
import json
import sqlite3
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.strategy.trend_integration import (
    TrendSignal, CarrySignal, CTAAllocation, TrendBacktestResult,
    TrendSignalGenerator, TrendReplicationStrategy,
    TREND_LOOKBACKS, TREND_SIGNAL_THRESHOLD, REPLICATION_ETFS,
    VOL_REGIMES, LOOKBACK_WEIGHTS, CARRY_MARKETS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trend_signal(ticker='SPY', composite=0.05, strength=0.05, direction='neutral'):
    """Create a test TrendSignal."""
    return TrendSignal(
        ticker=ticker,
        timestamp='2026-01-01',
        trend_1m=0.01, trend_3m=0.02, trend_6m=0.03, trend_12m=0.04,
        composite_trend=composite,
        trend_strength=strength,
        trend_direction=direction,
        momentum_consistency=0.75,
        sharpe_of_trends=1.2,
    )


def _make_carry_signal(ticker='GLD', curve='backwardation', signal=0.5):
    """Create a test CarrySignal."""
    return CarrySignal(
        ticker=ticker,
        timestamp='2026-01-01',
        roll_yield_annual=0.03,
        curve_shape=curve,
        carry_signal=signal,
        carry_quality=0.6,
        volatility_annual=0.15,
        carry_per_vol=0.20,
    )


def _mock_price_data(n_days=300, base_price=500.0, drift=0.0004, seed=42):
    """Generate synthetic price rows for DB mock."""
    np.random.seed(seed)
    prices = [base_price]
    for _ in range(n_days - 1):
        ret = np.random.normal(drift, 0.012)
        prices.append(prices[-1] * (1 + ret))
    dates = [(datetime.now() - __import__('datetime').timedelta(days=n_days - i)).strftime('%Y-%m-%d')
             for i in range(n_days)]
    return list(zip(dates, prices))


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Test module constants."""

    def test_trend_lookbacks(self):
        assert 20 in TREND_LOOKBACKS
        assert 60 in TREND_LOOKBACKS
        assert 120 in TREND_LOOKBACKS
        assert 252 in TREND_LOOKBACKS

    def test_lookback_weights_sum_to_one(self):
        total = sum(LOOKBACK_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_replication_etfs(self):
        assert "DBMF" in REPLICATION_ETFS
        assert "CTA" in REPLICATION_ETFS
        assert "KMLM" in REPLICATION_ETFS
        assert "HFMF" in REPLICATION_ETFS

    def test_vol_regimes(self):
        assert "low" in VOL_REGIMES
        assert "normal" in VOL_REGIMES
        assert "high" in VOL_REGIMES
        assert "extreme" in VOL_REGIMES

    def test_low_regime_max_overlay(self):
        assert VOL_REGIMES["low"]["max_overlay"] > VOL_REGIMES["normal"]["max_overlay"]

    def test_extreme_regime_most_restrictive(self):
        assert VOL_REGIMES["extreme"]["max_overlay"] < VOL_REGIMES["high"]["max_overlay"]

    def test_carry_markets(self):
        assert "commodities" in CARRY_MARKETS
        assert "rates" in CARRY_MARKETS
        assert "GLD" in CARRY_MARKETS["commodities"]

    def test_dbmf_expense_ratio(self):
        assert REPLICATION_ETFS["DBMF"]["expense_ratio"] == 0.0085


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestTrendSignal:
    """Test TrendSignal dataclass."""

    def test_creation(self):
        sig = _make_trend_signal()
        assert sig.ticker == 'SPY'
        assert sig.composite_trend == 0.05

    def test_to_dict(self):
        sig = _make_trend_signal()
        d = sig.to_dict()
        assert 'ticker' in d
        assert 'composite_trend' in d
        assert 'momentum_consistency' in d


class TestCarrySignal:
    """Test CarrySignal dataclass."""

    def test_creation(self):
        sig = _make_carry_signal()
        assert sig.ticker == 'GLD'
        assert sig.curve_shape == 'backwardation'

    def test_to_dict(self):
        sig = _make_carry_signal()
        d = sig.to_dict()
        assert 'carry_signal' in d
        assert 'roll_yield_annual' in d


class TestCTAAllocation:
    """Test CTAAllocation dataclass."""

    def test_creation(self):
        alloc = CTAAllocation(
            portfolio_value=100000,
            timestamp='2026-01-01',
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            overlay_pct=0.10,
            overlay_usd=10000,
            replication_etfs={},
            expected_vol=0.10,
            expected_return=0.08,
            correlation_to_base=0.10,
            diversification_ratio=0.91,
            vol_regime='normal',
            trend_strength_avg=0.15,
            signal_confidence=0.30,
            rebalance_triggered=False,
        )
        assert alloc.overlay_pct == 0.10

    def test_to_dict(self):
        alloc = CTAAllocation(
            portfolio_value=100000, timestamp='2026-01-01',
            base_allocation={}, overlay_pct=0.0, overlay_usd=0.0,
            replication_etfs={}, expected_vol=0.0, expected_return=0.0,
            correlation_to_base=0.0, diversification_ratio=1.0,
            vol_regime='normal', trend_strength_avg=0.0,
            signal_confidence=0.0, rebalance_triggered=False,
        )
        d = alloc.to_dict()
        assert 'overlay_pct' in d


class TestTrendBacktestResult:
    """Test TrendBacktestResult dataclass."""

    def test_creation(self):
        result = TrendBacktestResult(
            start_date='2020-01-01', end_date='2026-01-01',
            total_return=0.50, annualized_return=0.08,
            annualized_vol=0.12, sharpe_ratio=0.67,
            max_drawdown=-0.20, spy_return=0.60, spy_vol=0.18,
            spy_sharpe=0.33, correlation_to_spy=0.05,
            correlation_to_bonds=0.02, correlation_to_gold=0.03,
            recovery_time_avg=120.0, drawdown_events=5,
            performance_by_regime={},
        )
        assert result.sharpe_ratio == 0.67

    def test_to_dict(self):
        result = TrendBacktestResult(
            start_date='2020-01-01', end_date='2026-01-01',
            total_return=0.0, annualized_return=0.0,
            annualized_vol=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, spy_return=0.0, spy_vol=0.0,
            spy_sharpe=0.0, correlation_to_spy=0.0,
            correlation_to_bonds=0.0, correlation_to_gold=0.0,
            recovery_time_avg=0.0, drawdown_events=0,
            performance_by_regime={},
        )
        d = result.to_dict()
        assert 'sharpe_ratio' in d


# ---------------------------------------------------------------------------
# TrendSignalGenerator tests
# ---------------------------------------------------------------------------

class TestTrendSignalGenerator:
    """Test TrendSignalGenerator."""

    def test_calculate_trend_signal_returns_none_no_db(self, tmp_path):
        gen = TrendSignalGenerator.__new__(TrendSignalGenerator)
        gen.market_db = tmp_path / "nonexistent.db"
        result = gen.calculate_trend_signal('SPY')
        assert result is None

    def test_calculate_trend_signal_insufficient_data(self, tmp_path):
        gen = TrendSignalGenerator.__new__(TrendSignalGenerator)
        db_path = tmp_path / "market.db"
        gen.market_db = db_path
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
        # Insert only 100 rows (need 252+)
        for i in range(100):
            conn.execute("INSERT INTO prices VALUES ('SPY', ?, ?)",
                         (f'2026-{i+1:03d}', 500.0 + i))
        conn.commit()
        conn.close()
        result = gen.calculate_trend_signal('SPY')
        assert result is None

    def test_calculate_trend_signal_with_data(self, tmp_path):
        gen = TrendSignalGenerator.__new__(TrendSignalGenerator)
        db_path = tmp_path / "market.db"
        gen.market_db = db_path
        rows = _mock_price_data(300)
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
        for date, price in rows:
            conn.execute("INSERT INTO prices VALUES ('SPY', ?, ?)", (date, price))
        conn.commit()
        conn.close()
        result = gen.calculate_trend_signal('SPY')
        assert result is not None
        assert isinstance(result, TrendSignal)
        assert result.ticker == 'SPY'

    def test_trend_direction_classification(self, tmp_path):
        gen = TrendSignalGenerator.__new__(TrendSignalGenerator)
        db_path = tmp_path / "market.db"
        gen.market_db = db_path
        # Create strong uptrend
        n = 300
        dates = [(datetime.now() - __import__('datetime').timedelta(days=n - i)).strftime('%Y-%m-%d')
                 for i in range(n)]
        prices = [500.0 * (1.002 ** i) for i in range(n)]  # Steady uptrend
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
        for date, price in zip(dates, prices):
            conn.execute("INSERT INTO prices VALUES ('SPY', ?, ?)", (date, price))
        conn.commit()
        conn.close()
        result = gen.calculate_trend_signal('SPY')
        if result:
            assert result.trend_direction == 'bullish'

    def test_momentum_consistency_bounded(self, tmp_path):
        gen = TrendSignalGenerator.__new__(TrendSignalGenerator)
        db_path = tmp_path / "market.db"
        gen.market_db = db_path
        rows = _mock_price_data(300)
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
        for date, price in rows:
            conn.execute("INSERT INTO prices VALUES ('SPY', ?, ?)", (date, price))
        conn.commit()
        conn.close()
        result = gen.calculate_trend_signal('SPY')
        if result:
            assert 0.0 <= result.momentum_consistency <= 1.0

    def test_calculate_carry_signal_no_trend(self, tmp_path):
        gen = TrendSignalGenerator.__new__(TrendSignalGenerator)
        gen.market_db = tmp_path / "nonexistent.db"
        result = gen.calculate_carry_signal('GLD')
        assert result is None

    def test_get_trend_regime_no_signals(self, tmp_path):
        gen = TrendSignalGenerator.__new__(TrendSignalGenerator)
        gen.market_db = tmp_path / "nonexistent.db"
        regime = gen.get_trend_regime()
        assert regime == 'normal'


# ---------------------------------------------------------------------------
# TrendReplicationStrategy tests
# ---------------------------------------------------------------------------

class TestTrendReplicationStrategy:
    """Test TrendReplicationStrategy."""

    def test_calculate_overlay_with_trend_strength(self, tmp_path):
        strategy = TrendReplicationStrategy.__new__(TrendReplicationStrategy)
        strategy.market_db = tmp_path / "market.db"
        strategy.generator = MagicMock()
        alloc = strategy.calculate_overlay(
            portfolio_value=100000,
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            vol_regime='normal',
            trend_strength=0.15,
        )
        assert isinstance(alloc, CTAAllocation)
        assert alloc.overlay_pct > 0

    def test_overlay_zero_in_weak_trend(self, tmp_path):
        strategy = TrendReplicationStrategy.__new__(TrendReplicationStrategy)
        strategy.market_db = tmp_path / "market.db"
        strategy.generator = MagicMock()
        alloc = strategy.calculate_overlay(
            portfolio_value=100000,
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            vol_regime='normal',
            trend_strength=0.05,  # Below 0.10 threshold
        )
        assert alloc.overlay_pct == 0.0

    def test_overlay_scales_by_regime(self, tmp_path):
        strategy = TrendReplicationStrategy.__new__(TrendReplicationStrategy)
        strategy.market_db = tmp_path / "market.db"
        strategy.generator = MagicMock()
        alloc_low = strategy.calculate_overlay(
            portfolio_value=100000,
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            vol_regime='low', trend_strength=0.20,
        )
        alloc_extreme = strategy.calculate_overlay(
            portfolio_value=100000,
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            vol_regime='extreme', trend_strength=0.20,
        )
        assert alloc_low.overlay_pct > alloc_extreme.overlay_pct

    def test_replication_etfs_populated(self, tmp_path):
        strategy = TrendReplicationStrategy.__new__(TrendReplicationStrategy)
        strategy.market_db = tmp_path / "market.db"
        strategy.generator = MagicMock()
        alloc = strategy.calculate_overlay(
            portfolio_value=100000,
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            vol_regime='normal', trend_strength=0.15,
        )
        assert len(alloc.replication_etfs) > 0
        for etf in alloc.replication_etfs:
            assert etf in REPLICATION_ETFS

    def test_expected_return_positive_with_trend(self, tmp_path):
        strategy = TrendReplicationStrategy.__new__(TrendReplicationStrategy)
        strategy.market_db = tmp_path / "market.db"
        strategy.generator = MagicMock()
        alloc = strategy.calculate_overlay(
            portfolio_value=100000,
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            vol_regime='normal', trend_strength=0.15,
        )
        assert alloc.expected_return > 0

    def test_expected_vol_positive(self, tmp_path):
        strategy = TrendReplicationStrategy.__new__(TrendReplicationStrategy)
        strategy.market_db = tmp_path / "market.db"
        strategy.generator = MagicMock()
        alloc = strategy.calculate_overlay(
            portfolio_value=100000,
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            vol_regime='normal', trend_strength=0.15,
        )
        assert alloc.expected_vol > 0

    def test_correlation_to_base(self, tmp_path):
        strategy = TrendReplicationStrategy.__new__(TrendReplicationStrategy)
        strategy.market_db = tmp_path / "market.db"
        strategy.generator = MagicMock()
        alloc = strategy.calculate_overlay(
            portfolio_value=100000,
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            vol_regime='normal', trend_strength=0.15,
        )
        assert 0 < alloc.correlation_to_base < 0.5  # CTA low correlation

    def test_determine_vol_regime_default(self, tmp_path):
        strategy = TrendReplicationStrategy.__new__(TrendReplicationStrategy)
        strategy.market_db = tmp_path / "nonexistent.db"
        regime = strategy.determine_vol_regime()
        assert regime == 'normal'

    def test_determine_vol_regime_extreme(self, tmp_path):
        strategy = TrendReplicationStrategy.__new__(TrendReplicationStrategy)
        db_path = tmp_path / "market.db"
        strategy.market_db = db_path
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
        conn.execute("INSERT INTO prices VALUES ('VIX', '2026-01-01', 40.0)")
        conn.commit()
        conn.close()
        regime = strategy.determine_vol_regime()
        assert regime == 'extreme'

    def test_determine_vol_regime_low(self, tmp_path):
        strategy = TrendReplicationStrategy.__new__(TrendReplicationStrategy)
        db_path = tmp_path / "market.db"
        strategy.market_db = db_path
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
        conn.execute("INSERT INTO prices VALUES ('VIX', '2026-01-01', 12.0)")
        conn.commit()
        conn.close()
        regime = strategy.determine_vol_regime()
        assert regime == 'low'

    def test_allocate_strong_trend_favors_dbmf(self, tmp_path):
        strategy = TrendReplicationStrategy.__new__(TrendReplicationStrategy)
        strategy.market_db = tmp_path / "market.db"
        strategy.generator = MagicMock()
        alloc = strategy.calculate_overlay(
            portfolio_value=100000,
            base_allocation={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            vol_regime='normal', trend_strength=0.20,  # Strong trend
        )
        if 'DBMF' in alloc.replication_etfs:
            assert alloc.replication_etfs['DBMF']['allocation_pct'] > 0.30

    def test_calculate_expected_return_empty(self, tmp_path):
        strategy = TrendReplicationStrategy.__new__(TrendReplicationStrategy)
        ret = strategy._calculate_expected_return({})
        assert ret == 0.0

    def test_calculate_expected_vol_empty(self, tmp_path):
        strategy = TrendReplicationStrategy.__new__(TrendReplicationStrategy)
        vol = strategy._calculate_expected_vol({}, VOL_REGIMES["normal"])
        assert vol == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
