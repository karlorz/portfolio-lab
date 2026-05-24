#!/usr/bin/env python3
"""
Tests for vol_parity_allocator.py — VolParityAllocation dataclass, constants,
core allocation calculation, VIX allocation calculation, and CLI.
"""
import sys

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


class TestTimedeltaImport:
    """Regression: vol_parity_allocator used datetime.timedelta instead of timedelta."""

    def test_timedelta_importable(self):
        """timedelta should be importable directly (was missing from import)."""
        from src.strategy.vol_parity_allocator import timedelta
        assert timedelta is not None

    def test_run_backtest_uses_timedelta(self):
        """run_backtest should iterate dates without AttributeError."""
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        # Mock generate_allocation to avoid DB dependency
        mock_alloc = VolParityAllocation(
            date='2026-01-01', target_volatility=10.0,
            spy_pct=0.50, gld_pct=0.30, tlt_pct=0.10,
            core_vol_contribution=8.0,
            vix_short_pct=0.05, vix_tail_pct=0.02, vix_vol_contribution=2.0,
            cash_pct=0.03,
            expected_portfolio_vol=10.0, expected_max_dd=15.0,
            rebalance_triggered=False, rebalance_reason=None,
        )
        with patch.object(allocator, 'generate_allocation', return_value=mock_alloc):
            result = allocator.run_backtest('2026-01-01', '2026-01-03')
        assert 'days' in result and result['days'] == 3


class TestVolParityAllocationExtended:
    """Additional VolParityAllocation edge cases."""

    def test_total_allocation_with_varied_weights(self):
        a = _make_allocation(spy_pct=50.0, gld_pct=25.0, tlt_pct=10.0,
                             vix_short_pct=5.0, vix_tail_pct=2.0, cash_pct=8.0)
        assert a.total_allocation == pytest.approx(100.0)

    def test_rebalance_triggered_flag(self):
        a = _make_allocation(rebalance_triggered=True, rebalance_reason="Drift exceeded 10%")
        assert a.rebalance_triggered is True
        assert "Drift" in a.rebalance_reason

    def test_to_dict_includes_all_fields(self):
        a = _make_allocation()
        d = a.to_dict()
        expected_keys = {
            "date", "target_volatility", "spy_pct", "gld_pct", "tlt_pct",
            "core_vol_contribution", "vix_short_pct", "vix_tail_pct",
            "vix_vol_contribution", "cash_pct", "expected_portfolio_vol",
            "expected_max_dd", "rebalance_triggered", "rebalance_reason",
        }
        assert expected_keys.issubset(set(d.keys()))


class TestCalculateCoreAllocationExtended:
    """Additional core allocation edge cases."""

    def _make_allocator(self):
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        return allocator

    def test_weights_sum_to_one(self):
        allocator = self._make_allocator()
        for vix in [10, 15, 20, 25, 30, 35, 40]:
            signal = _make_convexity_signal(vix_level=vix)
            weights, _ = allocator.calculate_core_allocation(signal)
            assert sum(weights.values()) == pytest.approx(1.0)

    def test_higher_vix_reduces_spy(self):
        allocator = self._make_allocator()
        signal_low = _make_convexity_signal(vix_level=15)
        signal_high = _make_convexity_signal(vix_level=35)
        _, _ = allocator.calculate_core_allocation(signal_low)
        weights_high, _ = allocator.calculate_core_allocation(signal_high)
        # Stress regime should have lower SPY than base
        assert weights_high["SPY"] < 0.46

    def test_higher_vix_increases_gld(self):
        allocator = self._make_allocator()
        signal_high = _make_convexity_signal(vix_level=35)
        weights, _ = allocator.calculate_core_allocation(signal_high)
        assert weights["GLD"] > 0.38  # More gold in stress

    def test_low_vix_increases_spy(self):
        allocator = self._make_allocator()
        signal = _make_convexity_signal(vix_level=12)
        weights, _ = allocator.calculate_core_allocation(signal)
        assert weights["SPY"] > 0.46  # More equity in low vol


class TestCalculateVixAllocationExtended:
    """Additional VIX allocation edge cases."""

    def _make_allocator(self):
        allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
        allocator.target_vol = 10.0
        return allocator

    def test_zero_allocation_pct(self):
        """Zero convexity allocation should give zero short."""
        allocator = self._make_allocator()
        signal = _make_convexity_signal(vix_level=20, allocation_pct=0.0)
        short, tail, vol = allocator.calculate_vix_allocation(signal)
        assert short == 0.0

    def test_mid_range_vix_tail(self):
        """VIX 20 should use proportional tail sizing."""
        allocator = self._make_allocator()
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0)
        short, tail, vol = allocator.calculate_vix_allocation(signal)
        # tail = min(2.0, 3.0 * 0.4) = 1.2
        assert tail == pytest.approx(1.2)

    def test_vol_contribution_type(self):
        """VIX vol contribution should be a float."""
        allocator = self._make_allocator()
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0)
        _, _, vol = allocator.calculate_vix_allocation(signal)
        assert isinstance(vol, float)

    def test_higher_short_increases_vol(self):
        """Higher allocation_pct should increase VIX vol contribution."""
        allocator = self._make_allocator()
        signal_low = _make_convexity_signal(vix_level=20, allocation_pct=1.0)
        signal_high = _make_convexity_signal(vix_level=20, allocation_pct=5.0)
        _, _, vol_low = allocator.calculate_vix_allocation(signal_low)
        _, _, vol_high = allocator.calculate_vix_allocation(signal_high)
        assert vol_high > vol_low
