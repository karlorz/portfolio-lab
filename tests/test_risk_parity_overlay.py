#!/usr/bin/env python3
"""
Tests for risk parity overlay — data classes, volatility regime detection,
risk parity allocation calculation, leverage targeting, and persistence.
"""
import sys
import os
import json
import sqlite3
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.strategy.risk_parity_overlay import (
    RiskParityAllocation, VolatilityTargetState,
    RiskParityOverlay, RiskParityBacktester,
    VOL_LOOKBACK, VOL_TARGET_DEFAULT, MAX_LEVERAGE, MIN_LEVERAGE,
    MIN_WEIGHT, DEFAULT_BASE_ALLOCATION, ASSETS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices_df(n_days=400, seed=42, assets=None):
    """Create synthetic price DataFrame."""
    np.random.seed(seed)
    if assets is None:
        assets = ['SPY', 'GLD', 'TLT']
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    data = {}
    for i, ticker in enumerate(assets):
        drift = 0.0004 - i * 0.0001
        vol = 0.012 - i * 0.002
        prices = [500.0]
        for _ in range(n_days - 1):
            ret = np.random.normal(drift, max(vol, 0.003))
            prices.append(prices[-1] * (1 + ret))
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


def _make_overlay(tmp_path=None):
    """Create a RiskParityOverlay with test paths."""
    overlay = RiskParityOverlay.__new__(RiskParityOverlay)
    overlay.prices_path = tmp_path / "prices.json" if tmp_path else Path("/tmp/prices.json")
    overlay.db_path = tmp_path / "signals.db" if tmp_path else Path("/tmp/signals.db")
    overlay.vol_lookback = VOL_LOOKBACK
    overlay.target_vol = VOL_TARGET_DEFAULT
    overlay.max_leverage = MAX_LEVERAGE
    overlay.min_leverage = MIN_LEVERAGE
    overlay._prices_df = None
    return overlay


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Test module constants."""

    def test_vol_lookback(self):
        assert VOL_LOOKBACK == 252

    def test_vol_target_default(self):
        assert VOL_TARGET_DEFAULT == 0.10

    def test_max_leverage(self):
        assert MAX_LEVERAGE == 2.0

    def test_min_leverage(self):
        assert MIN_LEVERAGE == 0.5

    def test_min_weight(self):
        assert MIN_WEIGHT == 0.05

    def test_default_allocation_sums_to_one(self):
        total = sum(DEFAULT_BASE_ALLOCATION.values())
        assert abs(total - 1.0) < 0.01

    def test_assets_defined(self):
        assert 'SPY' in ASSETS
        assert 'GLD' in ASSETS
        assert 'TLT' in ASSETS
        assert 'CASH' in ASSETS


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestRiskParityAllocation:
    """Test RiskParityAllocation dataclass."""

    def test_creation(self):
        alloc = RiskParityAllocation(
            timestamp='2026-01-01',
            asset_vols={'SPY': 0.15, 'GLD': 0.12, 'TLT': 0.10},
            inverse_vols={'SPY': 6.67, 'GLD': 8.33, 'TLT': 10.0},
            raw_rp_weights={'SPY': 0.27, 'GLD': 0.33, 'TLT': 0.40},
            portfolio_vol_unlevered=0.085,
            leverage=1.18,
            target_weights={'SPY': 0.32, 'GLD': 0.39, 'TLT': 0.47, 'CASH': 0.0},
            target_vol=0.10,
            actual_vol_estimated=0.10,
            risk_contribution={'SPY': 0.048, 'GLD': 0.047, 'TLT': 0.047},
            risk_parity_quality=0.95,
        )
        assert alloc.leverage == 1.18
        assert alloc.risk_parity_quality == 0.95

    def test_to_dict(self):
        alloc = RiskParityAllocation(
            timestamp='2026-01-01',
            asset_vols={'SPY': 0.15},
            inverse_vols={'SPY': 6.67},
            raw_rp_weights={'SPY': 1.0},
            portfolio_vol_unlevered=0.15,
            leverage=0.67,
            target_weights={'SPY': 0.67, 'CASH': 0.33},
            target_vol=0.10,
            actual_vol_estimated=0.10,
            risk_contribution={'SPY': 0.10},
            risk_parity_quality=1.0,
        )
        d = alloc.to_dict()
        assert 'leverage' in d
        assert 'target_weights' in d


class TestVolatilityTargetState:
    """Test VolatilityTargetState dataclass."""

    def test_creation(self):
        state = VolatilityTargetState(
            timestamp='2026-01-01',
            current_portfolio_vol=0.12,
            target_vol=0.10,
            vol_regime='high',
            leverage_adjustment=0.85,
            regime_thresholds={'low': 0.06, 'normal': 0.14, 'high': 0.20, 'crisis': 0.30},
        )
        assert state.vol_regime == 'high'
        assert state.leverage_adjustment == 0.85

    def test_to_dict(self):
        state = VolatilityTargetState(
            timestamp='2026-01-01',
            current_portfolio_vol=0.10,
            target_vol=0.10,
            vol_regime='normal',
            leverage_adjustment=1.0,
            regime_thresholds={},
        )
        d = state.to_dict()
        assert 'vol_regime' in d


# ---------------------------------------------------------------------------
# Overlay init tests
# ---------------------------------------------------------------------------

class TestOverlayInit:
    """Test RiskParityOverlay initialization."""

    def test_default_params(self):
        overlay = RiskParityOverlay()
        assert overlay.vol_lookback == VOL_LOOKBACK
        assert overlay.target_vol == VOL_TARGET_DEFAULT
        assert overlay.max_leverage == MAX_LEVERAGE

    def test_custom_params(self):
        overlay = RiskParityOverlay(target_vol=0.15, max_leverage=3.0)
        assert overlay.target_vol == 0.15
        assert overlay.max_leverage == 3.0


# ---------------------------------------------------------------------------
# Volatility regime detection tests
# ---------------------------------------------------------------------------

class TestDetectVolRegime:
    """Test detect_vol_regime method."""

    def test_low_regime(self):
        overlay = _make_overlay()
        regime, adj = overlay.detect_vol_regime(0.05, 0.10)
        assert regime == 'low'
        assert adj == 1.2

    def test_normal_regime(self):
        overlay = _make_overlay()
        regime, adj = overlay.detect_vol_regime(0.10, 0.10)
        assert regime == 'normal'
        assert adj == 1.0

    def test_high_regime(self):
        overlay = _make_overlay()
        regime, adj = overlay.detect_vol_regime(0.16, 0.10)
        assert regime == 'high'
        assert adj == 0.85

    def test_crisis_regime(self):
        overlay = _make_overlay()
        regime, adj = overlay.detect_vol_regime(0.25, 0.10)
        assert regime == 'crisis'
        assert adj == 0.7

    def test_zero_target_vol(self):
        overlay = _make_overlay()
        regime, adj = overlay.detect_vol_regime(0.10, 0.0)
        assert regime == 'normal'  # ratio defaults to 1.0


# ---------------------------------------------------------------------------
# Realized volatility tests
# ---------------------------------------------------------------------------

class TestCalculateRealizedVol:
    """Test calculate_realized_vol method."""

    def test_returns_positive_vol(self):
        overlay = _make_overlay()
        prices_df = _make_prices_df(n_days=400)
        vol = overlay.calculate_realized_vol('SPY', prices_df=prices_df)
        assert vol is not None
        assert vol > 0

    def test_returns_none_for_missing_ticker(self):
        overlay = _make_overlay()
        prices_df = _make_prices_df(n_days=400)
        vol = overlay.calculate_realized_vol('NONEXISTENT', prices_df=prices_df)
        assert vol is None

    def test_returns_none_insufficient_data(self):
        overlay = _make_overlay()
        prices_df = _make_prices_df(n_days=50)
        vol = overlay.calculate_realized_vol('SPY', prices_df=prices_df)
        assert vol is None

    def test_custom_lookback(self):
        overlay = _make_overlay()
        prices_df = _make_prices_df(n_days=400)
        vol = overlay.calculate_realized_vol('SPY', lookback_days=100, prices_df=prices_df)
        assert vol is not None


# ---------------------------------------------------------------------------
# Risk parity allocation tests
# ---------------------------------------------------------------------------

class TestCalculateRiskParityAllocation:
    """Test calculate_risk_parity_allocation method."""

    def test_returns_allocation(self):
        overlay = _make_overlay()
        prices_df = _make_prices_df(n_days=400)
        alloc = overlay.calculate_risk_parity_allocation(prices_df=prices_df)
        assert isinstance(alloc, RiskParityAllocation)

    def test_weights_sum_to_one(self):
        overlay = _make_overlay()
        prices_df = _make_prices_df(n_days=400)
        alloc = overlay.calculate_risk_parity_allocation(prices_df=prices_df)
        total = sum(alloc.target_weights.values())
        assert abs(total - 1.0) < 0.05

    def test_leverage_bounded(self):
        overlay = _make_overlay()
        prices_df = _make_prices_df(n_days=400)
        alloc = overlay.calculate_risk_parity_allocation(prices_df=prices_df)
        assert MIN_LEVERAGE <= alloc.leverage <= MAX_LEVERAGE

    def test_custom_target_vol(self):
        overlay = _make_overlay()
        prices_df = _make_prices_df(n_days=400)
        alloc = overlay.calculate_risk_parity_allocation(target_vol=0.15, prices_df=prices_df)
        assert alloc.target_vol == 0.15

    def test_returns_none_for_missing_data(self):
        overlay = _make_overlay()
        prices_df = _make_prices_df(n_days=400, assets=['SPY'])  # Missing GLD, TLT
        alloc = overlay.calculate_risk_parity_allocation(prices_df=prices_df)
        assert alloc is None

    def test_risk_parity_quality_bounded(self):
        overlay = _make_overlay()
        prices_df = _make_prices_df(n_days=400)
        alloc = overlay.calculate_risk_parity_allocation(prices_df=prices_df)
        assert 0.0 <= alloc.risk_parity_quality <= 1.0

    def test_has_all_assets(self):
        overlay = _make_overlay()
        prices_df = _make_prices_df(n_days=400)
        alloc = overlay.calculate_risk_parity_allocation(prices_df=prices_df)
        assert 'SPY' in alloc.target_weights
        assert 'GLD' in alloc.target_weights
        assert 'TLT' in alloc.target_weights
        assert 'CASH' in alloc.target_weights

    def test_risk_contribution_non_negative(self):
        overlay = _make_overlay()
        prices_df = _make_prices_df(n_days=400)
        alloc = overlay.calculate_risk_parity_allocation(prices_df=prices_df)
        for v in alloc.risk_contribution.values():
            assert v >= 0.0

    def test_actual_vol_estimated_positive(self):
        overlay = _make_overlay()
        prices_df = _make_prices_df(n_days=400)
        alloc = overlay.calculate_risk_parity_allocation(prices_df=prices_df)
        assert alloc.actual_vol_estimated > 0


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestSaveToDb:
    """Test save_to_db method."""

    def test_saves_to_database(self, tmp_path):
        overlay = _make_overlay(tmp_path)
        alloc = RiskParityAllocation(
            timestamp='2026-01-01',
            asset_vols={'SPY': 0.15, 'GLD': 0.12, 'TLT': 0.10},
            inverse_vols={'SPY': 6.67, 'GLD': 8.33, 'TLT': 10.0},
            raw_rp_weights={'SPY': 0.27, 'GLD': 0.33, 'TLT': 0.40},
            portfolio_vol_unlevered=0.085,
            leverage=1.18,
            target_weights={'SPY': 0.32, 'GLD': 0.39, 'TLT': 0.47, 'CASH': 0.0},
            target_vol=0.10,
            actual_vol_estimated=0.10,
            risk_contribution={'SPY': 0.048, 'GLD': 0.047, 'TLT': 0.047},
            risk_parity_quality=0.95,
        )
        overlay.save_to_db(alloc)

        conn = sqlite3.connect(str(overlay.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM risk_parity_allocations")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1


# ---------------------------------------------------------------------------
# Backtester tests
# ---------------------------------------------------------------------------

class TestRiskParityBacktester:
    """Test RiskParityBacktester."""

    def test_init(self):
        bt = RiskParityBacktester.__new__(RiskParityBacktester)
        bt.target_vol = VOL_TARGET_DEFAULT
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        assert bt.target_vol == 0.10

    def test_insufficient_data_returns_error(self):
        bt = RiskParityBacktester.__new__(RiskParityBacktester)
        bt.target_vol = 0.10
        bt.start_date = None
        bt.end_date = None
        bt.rebalance_freq = 21
        bt.rp_overlay = _make_overlay()
        bt.rp_overlay._prices_df = _make_prices_df(n_days=50)  # Too few
        bt.prices_df = bt.rp_overlay._prices_df
        result = bt.run_backtest()
        assert 'error' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
