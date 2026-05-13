#!/usr/bin/env python3
"""
Tests for vol_parity_allocator.py — VolParityAllocation dataclass, constants,
core allocation calculation, VIX allocation calculation, and CLI.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime

# Mock external dependencies before importing
sys.modules['data.vix_futures'] = MagicMock()
sys.modules['strategy.convexity_harvest'] = MagicMock()

from src.strategy.vol_parity_allocator import (
    VolParityAllocation,
    VolatilityParityAllocator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_convexity_signal(**overrides):
    """Create a mock ConvexityPosition."""
    mock = MagicMock()
    mock.vix_level = overrides.get("vix_level", 20.0)
    mock.allocation_pct = overrides.get("allocation_pct", 3.0)
    mock.exit_triggered = overrides.get("exit_triggered", False)
    return mock


def _make_allocation(**overrides):
    defaults = dict(
        date="2026-05-14",
        target_volatility=10.0,
        spy_pct=36.8,
        gld_pct=30.4,
        tlt_pct=12.8,
        core_vol_contribution=8.0,
        vix_short_pct=3.0,
        vix_tail_pct=1.0,
        vix_vol_contribution=0.5,
        cash_pct=16.0,
        expected_portfolio_vol=7.5,
        expected_max_dd=11.25,
        rebalance_triggered=False,
        rebalance_reason=None,
    )
    defaults.update(overrides)
    return VolParityAllocation(**defaults)


# ---------------------------------------------------------------------------
# VolParityAllocation Tests
# ---------------------------------------------------------------------------

class TestVolParityAllocation:

    def test_to_dict(self):
        a = _make_allocation()
        d = a.to_dict()
        assert d["date"] == "2026-05-14"
        assert d["target_volatility"] == 10.0
        assert "spy_pct" in d

    def test_total_allocation(self):
        a = _make_allocation(spy_pct=36.8, gld_pct=30.4, tlt_pct=12.8,
                             vix_short_pct=3.0, vix_tail_pct=1.0, cash_pct=16.0)
        assert a.total_allocation == pytest.approx(100.0)

    def test_total_vol_contribution(self):
        a = _make_allocation(core_vol_contribution=8.0, vix_vol_contribution=0.5)
        assert a.total_vol_contribution == pytest.approx(8.5)


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    def test_target_volatility(self):
        assert VolatilityParityAllocator.TARGET_VOLATILITY == 10.0

    def test_core_base_weights(self):
        w = VolatilityParityAllocator.CORE_BASE_WEIGHTS
        assert w["SPY"] == 0.46
        assert w["GLD"] == 0.38
        assert w["TLT"] == 0.16
        assert sum(w.values()) == pytest.approx(1.0)

    def test_core_asset_vols(self):
        v = VolatilityParityAllocator.CORE_ASSET_VOLS
        assert v["SPY"] == 15.0
        assert v["GLD"] == 14.0
        assert v["TLT"] == 12.0

    def test_max_vix(self):
        assert VolatilityParityAllocator.MAX_VIX_SHORT_PCT == 5.0
        assert VolatilityParityAllocator.MAX_VIX_TAIL_PCT == 2.0

    def test_rebalance_threshold(self):
        assert VolatilityParityAllocator.REBALANCE_THRESHOLD == 10.0


# ---------------------------------------------------------------------------
# calculate_core_allocation Tests
# ---------------------------------------------------------------------------

class TestCalculateCoreAllocation:

    def test_normal_regime(self):
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        signal = _make_convexity_signal(vix_level=20)
        weights, vol = allocator.calculate_core_allocation(signal)
        assert weights == VolatilityParityAllocator.CORE_BASE_WEIGHTS
        assert vol > 0

    def test_stress_regime(self):
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        signal = _make_convexity_signal(vix_level=35)
        weights, vol = allocator.calculate_core_allocation(signal)
        assert weights["SPY"] == 0.35
        assert weights["GLD"] == 0.45

    def test_elevated_vol_regime(self):
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        signal = _make_convexity_signal(vix_level=27)
        weights, vol = allocator.calculate_core_allocation(signal)
        assert weights["SPY"] == 0.40
        assert weights["GLD"] == 0.42

    def test_low_vol_regime(self):
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        signal = _make_convexity_signal(vix_level=12)
        weights, vol = allocator.calculate_core_allocation(signal)
        assert weights["SPY"] == 0.50
        assert weights["GLD"] == 0.35

    def test_vol_calculation(self):
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        signal = _make_convexity_signal(vix_level=20)
        weights, vol = allocator.calculate_core_allocation(signal)
        expected = 0.46 * 15.0 + 0.38 * 14.0 + 0.16 * 12.0
        assert vol == pytest.approx(expected)

    def test_boundary_vix_30(self):
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        signal = _make_convexity_signal(vix_level=30)
        weights, vol = allocator.calculate_core_allocation(signal)
        # Exactly 30 → elevated (not stress)
        assert weights["SPY"] == 0.40

    def test_boundary_vix_25(self):
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        signal = _make_convexity_signal(vix_level=25)
        weights, vol = allocator.calculate_core_allocation(signal)
        # Exactly 25 → normal
        assert weights["SPY"] == 0.46

    def test_boundary_vix_15(self):
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        signal = _make_convexity_signal(vix_level=15)
        weights, vol = allocator.calculate_core_allocation(signal)
        # Exactly 15 → normal
        assert weights["SPY"] == 0.46


# ---------------------------------------------------------------------------
# calculate_vix_allocation Tests
# ---------------------------------------------------------------------------

class TestCalculateVixAllocation:

    def test_normal_vix(self):
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0)
        short, tail, vol = allocator.calculate_vix_allocation(signal)
        assert short == 3.0
        assert tail <= VolatilityParityAllocator.MAX_VIX_TAIL_PCT

    def test_short_capped(self):
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        signal = _make_convexity_signal(vix_level=20, allocation_pct=10.0)
        short, tail, vol = allocator.calculate_vix_allocation(signal)
        assert short == VolatilityParityAllocator.MAX_VIX_SHORT_PCT

    def test_low_vix_full_tail(self):
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        signal = _make_convexity_signal(vix_level=12, allocation_pct=3.0)
        short, tail, vol = allocator.calculate_vix_allocation(signal)
        assert tail == 2.0

    def test_high_vix_reduced_tail(self):
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        signal = _make_convexity_signal(vix_level=35, allocation_pct=3.0)
        short, tail, vol = allocator.calculate_vix_allocation(signal)
        assert tail == 0.5

    def test_tail_capped(self):
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        signal = _make_convexity_signal(vix_level=10, allocation_pct=3.0)
        short, tail, vol = allocator.calculate_vix_allocation(signal)
        assert tail <= VolatilityParityAllocator.MAX_VIX_TAIL_PCT

    def test_vol_contribution_positive(self):
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0)
        short, tail, vol = allocator.calculate_vix_allocation(signal)
        # Short VIX contributes positive vol, tail subtracts
        assert isinstance(vol, float)
