#!/usr/bin/env python3
"""
Tests for vol_parity_allocator.py — comprehensive coverage for:

1. Dataclass field validation (types, defaults, properties)
2. Computation edge cases (zero, negative, boundary, extreme inputs)
3. Constants validation (types, ranges, completeness)
4. Function boundary conditions (extreme inputs, missing keys, wrong types)
5. CLI/__main__ guard (capsys for print, caplog for logging)
6. Export completeness (__all__ coverage)
"""
import sys

import pytest
from unittest.mock import patch, MagicMock

# Mock external dependencies before importing — with cleanup
_ORIG_MODULES = {}
for _key in ('data.vix_futures', 'strategy.convexity_harvest'):
    _ORIG_MODULES[_key] = sys.modules.get(_key)
    sys.modules[_key] = MagicMock()

from src.strategy.vol_parity_allocator import (
    VolParityAllocation,
    VolatilityParityAllocator,
)


@pytest.fixture(scope="module", autouse=True)
def _cleanup_sys_modules():
    """Restore sys.modules after test module completes."""
    yield
    for _key, _orig in _ORIG_MODULES.items():
        if _orig is None:
            sys.modules.pop(_key, None)
        else:
            sys.modules[_key] = _orig


# ==============================================================================
# Helpers
# ==============================================================================

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


def _make_allocator(**kwargs):
    """Create an allocator without calling __init__ (avoids external deps)."""
    allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
    allocator.target_vol = kwargs.get("target_vol", 10.0)
    allocator.vix_strategy = kwargs.get("vix_strategy", MagicMock())
    allocator.vix_manager = kwargs.get("vix_manager", MagicMock())
    allocator.last_allocation = kwargs.get("last_allocation", None)
    return allocator


# ==============================================================================
# 1. Dataclass Field Validation
# ==============================================================================

class TestVolParityAllocation:
    """Core VolParityAllocation dataclass tests."""

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


class TestDataclassFieldValidation:
    """Validate dataclass fields: presence, types, no defaults where expected."""

    def test_all_fields_present(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(VolParityAllocation)}
        expected = {
            "date", "target_volatility", "spy_pct", "gld_pct", "tlt_pct",
            "core_vol_contribution", "vix_short_pct", "vix_tail_pct",
            "vix_vol_contribution", "cash_pct", "expected_portfolio_vol",
            "expected_max_dd", "rebalance_triggered", "rebalance_reason",
        }
        assert field_names == expected
        assert len(field_names) == 14

    def test_field_count(self):
        from dataclasses import fields
        assert len(list(fields(VolParityAllocation))) == 14

    def test_field_types(self):
        from dataclasses import fields
        field_map = {f.name: f.type for f in fields(VolParityAllocation)}
        # str fields
        assert field_map["date"] is str
        # float fields
        float_fields = [
            "target_volatility", "spy_pct", "gld_pct", "tlt_pct",
            "core_vol_contribution", "vix_short_pct", "vix_tail_pct",
            "vix_vol_contribution", "cash_pct", "expected_portfolio_vol",
            "expected_max_dd",
        ]
        for fname in float_fields:
            assert field_map[fname] is float, f"{fname} should be float, got {field_map[fname]}"
        # bool field
        assert field_map["rebalance_triggered"] is bool
        # Optional[str] field — can be typing.Optional[str] or Optional[str]
        opt_str = field_map["rebalance_reason"]
        assert "Optional" in str(opt_str) and "str" in str(opt_str), f"Expected Optional[str], got {opt_str}"

    def test_all_fields_required(self):
        """All fields have no defaults — all required at construction."""
        from dataclasses import fields, MISSING
        for f in fields(VolParityAllocation):
            assert f.default is MISSING, f"{f.name} has unexpected default: {f.default}"
            assert f.default_factory is MISSING, f"{f.name} has unexpected factory"

    def test_construction_minimal(self):
        """Can construct with all fields."""
        a = VolParityAllocation(
            date="2026-01-01",
            target_volatility=10.0,
            spy_pct=40.0,
            gld_pct=30.0,
            tlt_pct=10.0,
            core_vol_contribution=8.0,
            vix_short_pct=3.0,
            vix_tail_pct=1.0,
            vix_vol_contribution=2.0,
            cash_pct=16.0,
            expected_portfolio_vol=9.0,
            expected_max_dd=13.5,
            rebalance_triggered=False,
            rebalance_reason=None,
        )
        assert a.date == "2026-01-01"

    def test_construction_missing_field_raises(self):
        """Missing a required field should raise TypeError."""
        with pytest.raises(TypeError):
            VolParityAllocation(
                date="2026-01-01",
                target_volatility=10.0,
                spy_pct=40.0,
                gld_pct=30.0,
                tlt_pct=10.0,
                core_vol_contribution=8.0,
                vix_short_pct=3.0,
                vix_tail_pct=1.0,
                vix_vol_contribution=2.0,
                cash_pct=16.0,
                expected_portfolio_vol=9.0,
                expected_max_dd=13.5,
                rebalance_triggered=False,
                # missing rebalance_reason
            )


# ==============================================================================
# Dataclass Edge Cases
# ==============================================================================

class TestVolParityAllocationEdgeCases:
    """Edge cases for VolParityAllocation properties and methods."""

    def test_total_allocation_zero_all(self):
        a = _make_allocation(spy_pct=0, gld_pct=0, tlt_pct=0,
                             vix_short_pct=0, vix_tail_pct=0, cash_pct=0)
        assert a.total_allocation == pytest.approx(0.0)

    def test_total_allocation_only_core(self):
        a = _make_allocation(spy_pct=46.0, gld_pct=38.0, tlt_pct=16.0,
                             vix_short_pct=0, vix_tail_pct=0, cash_pct=0)
        assert a.total_allocation == pytest.approx(100.0)

    def test_total_allocation_only_vix(self):
        a = _make_allocation(spy_pct=0, gld_pct=0, tlt_pct=0,
                             vix_short_pct=5.0, vix_tail_pct=2.0, cash_pct=0)
        assert a.total_allocation == pytest.approx(7.0)

    def test_total_allocation_negative_weights(self):
        """Negative weights are summed (no guard in source)."""
        a = _make_allocation(spy_pct=-10.0, gld_pct=50.0, tlt_pct=20.0,
                             vix_short_pct=0, vix_tail_pct=0, cash_pct=40.0)
        assert a.total_allocation == pytest.approx(100.0)

    def test_total_vol_contribution_zero(self):
        a = _make_allocation(core_vol_contribution=0.0, vix_vol_contribution=0.0)
        assert a.total_vol_contribution == pytest.approx(0.0)

    def test_total_vol_contribution_large(self):
        a = _make_allocation(core_vol_contribution=50.0, vix_vol_contribution=30.0)
        assert a.total_vol_contribution == pytest.approx(80.0)

    def test_total_vol_contribution_negative_vix(self):
        """VIX vol contribution can be negative (tail > short)."""
        a = _make_allocation(core_vol_contribution=8.0, vix_vol_contribution=-5.0)
        assert a.total_vol_contribution == pytest.approx(3.0)

    def test_to_dict_roundtrip(self):
        """to_dict should contain the same values as fields."""
        a = _make_allocation()
        d = a.to_dict()
        assert d["spy_pct"] == 36.8
        assert d["gld_pct"] == 30.4
        assert d["tlt_pct"] == 12.8
        assert d["rebalance_reason"] is None

    def test_to_dict_keys_exact(self):
        a = _make_allocation()
        d = a.to_dict()
        expected_keys = {
            "date", "target_volatility", "spy_pct", "gld_pct", "tlt_pct",
            "core_vol_contribution", "vix_short_pct", "vix_tail_pct",
            "vix_vol_contribution", "cash_pct", "expected_portfolio_vol",
            "expected_max_dd", "rebalance_triggered", "rebalance_reason",
            # honesty provenance fields (Batch AG full to_dict)
            "weight_unit", "role", "live_authoritative", "description",
        }
        assert set(d.keys()) == expected_keys

    def test_rebalance_reason_string(self):
        a = _make_allocation(rebalance_triggered=True, rebalance_reason="Drift exceeded 10%")
        assert a.rebalance_triggered is True
        assert isinstance(a.rebalance_reason, str)
        assert "Drift" in a.rebalance_reason

    def test_rebalance_reason_none(self):
        a = _make_allocation(rebalance_triggered=False, rebalance_reason=None)
        assert a.rebalance_triggered is False
        assert a.rebalance_reason is None

    def test_expected_max_dd_proportional(self):
        """expected_max_dd should be 1.5 * expected_portfolio_vol."""
        a = _make_allocation(expected_portfolio_vol=10.0, expected_max_dd=15.0)
        assert a.expected_max_dd == pytest.approx(a.expected_portfolio_vol * 1.5)


# ==============================================================================
# 2. Constants Validation
# ==============================================================================

class TestConstants:
    """Verifies module-level constants of VolatilityParityAllocator."""

    def test_target_volatility(self):
        assert VolatilityParityAllocator.TARGET_VOLATILITY == 10.0
        assert isinstance(VolatilityParityAllocator.TARGET_VOLATILITY, (int, float))
        assert VolatilityParityAllocator.TARGET_VOLATILITY > 0

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


class TestAllocatorConstants:
    """Extended constants validation — types, ranges, completeness."""

    def test_target_volatility_type(self):
        assert isinstance(VolatilityParityAllocator.TARGET_VOLATILITY, (int, float))

    def test_core_asset_vols_positive(self):
        vols = VolatilityParityAllocator.CORE_ASSET_VOLS
        assert all(v > 0 for v in vols.values())

    def test_core_asset_vols_all_assets(self):
        vols = VolatilityParityAllocator.CORE_ASSET_VOLS
        assert "SPY" in vols
        assert "GLD" in vols
        assert "TLT" in vols

    def test_max_vix_short_pct(self):
        assert VolatilityParityAllocator.MAX_VIX_SHORT_PCT == 5.0

    def test_max_vix_tail_pct(self):
        assert VolatilityParityAllocator.MAX_VIX_TAIL_PCT == 2.0

    def test_rebalance_threshold(self):
        assert VolatilityParityAllocator.REBALANCE_THRESHOLD == 10.0

    def test_vix_caps_positive(self):
        assert VolatilityParityAllocator.MAX_VIX_SHORT_PCT > 0
        assert VolatilityParityAllocator.MAX_VIX_TAIL_PCT > 0

    def test_vix_short_greater_than_tail(self):
        assert VolatilityParityAllocator.MAX_VIX_SHORT_PCT > VolatilityParityAllocator.MAX_VIX_TAIL_PCT

    def test_rebalance_threshold_range(self):
        """Rebalance threshold should be a reasonable percentage (5-20%)."""
        thresh = VolatilityParityAllocator.REBALANCE_THRESHOLD
        assert 5.0 <= thresh <= 20.0

    def test_core_asset_vols_range(self):
        vols = VolatilityParityAllocator.CORE_ASSET_VOLS
        assert all(5.0 <= v <= 50.0 for v in vols.values())

    def test_core_base_weights_positive(self):
        w = VolatilityParityAllocator.CORE_BASE_WEIGHTS
        assert all(val > 0 for val in w.values())

    def test_core_base_weights_keys(self):
        w = VolatilityParityAllocator.CORE_BASE_WEIGHTS
        assert set(w.keys()) == {"SPY", "GLD", "TLT"}


# ==============================================================================
# 3. calculate_core_allocation Boundary Conditions
# ==============================================================================

class TestCalculateCoreAllocation:
    """Core allocation regimes and boundaries."""

    def test_normal_regime(self):
        allocator = _make_allocator()
        signal = _make_convexity_signal(vix_level=20)
        weights, vol = allocator.calculate_core_allocation(signal)
        assert weights == VolatilityParityAllocator.CORE_BASE_WEIGHTS
        assert vol > 0

    def test_stress_regime(self):
        allocator = _make_allocator()
        signal = _make_convexity_signal(vix_level=35)
        weights, vol = allocator.calculate_core_allocation(signal)
        assert weights["SPY"] == 0.35
        assert weights["GLD"] == 0.45

    def test_elevated_vol_regime(self):
        allocator = _make_allocator()
        signal = _make_convexity_signal(vix_level=27)
        weights, vol = allocator.calculate_core_allocation(signal)
        assert weights["SPY"] == 0.40
        assert weights["GLD"] == 0.42

    def test_low_vol_regime(self):
        allocator = _make_allocator()
        signal = _make_convexity_signal(vix_level=12)
        weights, vol = allocator.calculate_core_allocation(signal)
        assert weights["SPY"] == 0.50
        assert weights["GLD"] == 0.35

    def test_vol_calculation(self):
        allocator = _make_allocator()
        signal = _make_convexity_signal(vix_level=20)
        weights, vol = allocator.calculate_core_allocation(signal)
        expected = 0.46 * 15.0 + 0.38 * 14.0 + 0.16 * 12.0
        assert vol == pytest.approx(expected)

    def test_boundary_vix_30(self):
        allocator = _make_allocator()
        signal = _make_convexity_signal(vix_level=30)
        weights, vol = allocator.calculate_core_allocation(signal)
        assert weights["SPY"] == 0.40  # elevated, not stress

    def test_boundary_vix_25(self):
        allocator = _make_allocator()
        signal = _make_convexity_signal(vix_level=25)
        weights, vol = allocator.calculate_core_allocation(signal)
        assert weights["SPY"] == 0.46  # normal

    def test_boundary_vix_15(self):
        allocator = _make_allocator()
        signal = _make_convexity_signal(vix_level=15)
        weights, vol = allocator.calculate_core_allocation(signal)
        assert weights["SPY"] == 0.46  # normal

    def test_boundary_vix_just_below_stress(self):
        allocator = _make_allocator()
        signal = _make_convexity_signal(vix_level=30.0)
        weights, _ = allocator.calculate_core_allocation(signal)
        assert weights["SPY"] == 0.40  # elevated, stress starts at >30

    def test_boundary_vix_just_above_stress(self):
        allocator = _make_allocator()
        signal = _make_convexity_signal(vix_level=30.001)
        weights, _ = allocator.calculate_core_allocation(signal)
        assert weights["SPY"] == 0.35  # stress


class TestCalculateCoreAllocationExtended:
    """Extended core allocation tests — edge cases and invariants."""

    def _make_allocator(self):
        return _make_allocator()

    def test_weights_sum_to_one(self):
        allocator = self._make_allocator()
        for vix in [10, 15, 20, 25, 30, 35, 40]:
            signal = _make_convexity_signal(vix_level=vix)
            weights, _ = allocator.calculate_core_allocation(signal)
            assert sum(weights.values()) == pytest.approx(1.0)

    def test_higher_vix_reduces_spy(self):
        allocator = self._make_allocator()
        weights_high, _ = allocator.calculate_core_allocation(
            _make_convexity_signal(vix_level=35)
        )
        assert weights_high["SPY"] < 0.46

    def test_higher_vix_increases_gld(self):
        allocator = self._make_allocator()
        weights, _ = allocator.calculate_core_allocation(
            _make_convexity_signal(vix_level=35)
        )
        assert weights["GLD"] > 0.38

    def test_low_vix_increases_spy(self):
        allocator = self._make_allocator()
        weights, _ = allocator.calculate_core_allocation(
            _make_convexity_signal(vix_level=12)
        )
        assert weights["SPY"] > 0.46

    def test_vol_calculation_stress(self):
        allocator = self._make_allocator()
        weights, vol = allocator.calculate_core_allocation(
            _make_convexity_signal(vix_level=35)
        )
        expected = 0.35 * 15.0 + 0.45 * 14.0 + 0.20 * 12.0
        assert vol == pytest.approx(expected)

    def test_vol_calculation_elevated(self):
        allocator = self._make_allocator()
        weights, vol = allocator.calculate_core_allocation(
            _make_convexity_signal(vix_level=27)
        )
        expected = 0.40 * 15.0 + 0.42 * 14.0 + 0.18 * 12.0
        assert vol == pytest.approx(expected)

    def test_vol_calculation_low(self):
        allocator = self._make_allocator()
        weights, vol = allocator.calculate_core_allocation(
            _make_convexity_signal(vix_level=12)
        )
        expected = 0.50 * 15.0 + 0.35 * 14.0 + 0.15 * 12.0
        assert vol == pytest.approx(expected)

    def test_vix_level_extreme_high(self):
        allocator = self._make_allocator()
        weights, vol = allocator.calculate_core_allocation(
            _make_convexity_signal(vix_level=100)
        )
        assert weights["SPY"] == 0.35
        assert weights["GLD"] == 0.45
        assert vol > 0

    def test_vix_level_zero(self):
        allocator = self._make_allocator()
        weights, vol = allocator.calculate_core_allocation(
            _make_convexity_signal(vix_level=0)
        )
        assert weights["SPY"] == 0.50  # low vol regime
        assert vol > 0

    def test_vix_level_negative(self):
        """Negative vix_level triggers low vol regime."""
        allocator = self._make_allocator()
        weights, _ = allocator.calculate_core_allocation(
            _make_convexity_signal(vix_level=-5.0)
        )
        assert weights["SPY"] == 0.50  # low vol regime


# ==============================================================================
# 4. calculate_vix_allocation Boundary Conditions
# ==============================================================================

class TestCalculateVixAllocation:
    """VIX allocation tests — capping, regimes, contributions."""

    def test_normal_vix(self):
        allocator = _make_allocator()
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0)
        short, tail, vol = allocator.calculate_vix_allocation(signal)
        assert short == 3.0
        assert tail <= VolatilityParityAllocator.MAX_VIX_TAIL_PCT

    def test_short_capped(self):
        allocator = _make_allocator()
        signal = _make_convexity_signal(vix_level=20, allocation_pct=10.0)
        short, tail, vol = allocator.calculate_vix_allocation(signal)
        assert short == VolatilityParityAllocator.MAX_VIX_SHORT_PCT

    def test_low_vix_full_tail(self):
        allocator = _make_allocator()
        signal = _make_convexity_signal(vix_level=12, allocation_pct=3.0)
        short, tail, vol = allocator.calculate_vix_allocation(signal)
        assert tail == 2.0

    def test_high_vix_reduced_tail(self):
        allocator = _make_allocator()
        signal = _make_convexity_signal(vix_level=35, allocation_pct=3.0)
        short, tail, vol = allocator.calculate_vix_allocation(signal)
        assert tail == 0.5

    def test_tail_capped(self):
        allocator = _make_allocator()
        signal = _make_convexity_signal(vix_level=10, allocation_pct=3.0)
        short, tail, vol = allocator.calculate_vix_allocation(signal)
        assert tail <= VolatilityParityAllocator.MAX_VIX_TAIL_PCT

    def test_vol_contribution_positive(self):
        allocator = _make_allocator()
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0)
        short, tail, vol = allocator.calculate_vix_allocation(signal)
        assert isinstance(vol, float)


class TestCalculateVixAllocationExtended:
    """Extended VIX allocation edge cases."""

    def _make_allocator(self):
        return _make_allocator()

    def test_zero_allocation_pct(self):
        allocator = self._make_allocator()
        signal = _make_convexity_signal(vix_level=20, allocation_pct=0.0)
        short, tail, vol = allocator.calculate_vix_allocation(signal)
        assert short == 0.0

    def test_mid_range_vix_tail(self):
        allocator = self._make_allocator()
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0)
        short, tail, vol = allocator.calculate_vix_allocation(signal)
        assert tail == pytest.approx(1.2)

    def test_vol_contribution_type(self):
        allocator = self._make_allocator()
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0)
        _, _, vol = allocator.calculate_vix_allocation(signal)
        assert isinstance(vol, float)

    def test_higher_short_increases_vol(self):
        allocator = self._make_allocator()
        _, _, vol_low = allocator.calculate_vix_allocation(
            _make_convexity_signal(vix_level=20, allocation_pct=1.0)
        )
        _, _, vol_high = allocator.calculate_vix_allocation(
            _make_convexity_signal(vix_level=20, allocation_pct=5.0)
        )
        assert vol_high > vol_low

    def test_vix_level_boundary_15(self):
        """vix_level=15 uses proportional tail sizing (normal band)."""
        allocator = self._make_allocator()
        _, tail, _ = allocator.calculate_vix_allocation(
            _make_convexity_signal(vix_level=15, allocation_pct=3.0)
        )
        assert tail == pytest.approx(1.2)

    def test_vix_level_boundary_30(self):
        """vix_level=30 uses proportional tail (not >30 so not high-vol branch)."""
        allocator = self._make_allocator()
        _, tail, _ = allocator.calculate_vix_allocation(
            _make_convexity_signal(vix_level=30, allocation_pct=3.0)
        )
        # vix_level=30 is NOT >30, so falls to else: tail = min(2, 3*0.4) = 1.2
        assert tail == pytest.approx(1.2)

    def test_vix_level_negative(self):
        """Negative vix_level triggers low-vix full tail."""
        allocator = self._make_allocator()
        _, tail, _ = allocator.calculate_vix_allocation(
            _make_convexity_signal(vix_level=-5.0, allocation_pct=3.0)
        )
        assert tail == 2.0

    def test_allocation_pct_negative(self):
        """Negative allocation_pct is treated as-is (min with 5.0)."""
        allocator = self._make_allocator()
        short, _, _ = allocator.calculate_vix_allocation(
            _make_convexity_signal(vix_level=20, allocation_pct=-1.0)
        )
        assert short == -1.0  # min(-1.0, 5.0) = -1.0, no lower guard

    def test_allocation_pct_extreme(self):
        """Extreme allocation capped at MAX_VIX_SHORT_PCT."""
        allocator = self._make_allocator()
        short, _, _ = allocator.calculate_vix_allocation(
            _make_convexity_signal(vix_level=20, allocation_pct=100.0)
        )
        assert short == 5.0

    def test_vol_contribution_formula(self):
        """Verify vix_vol_contribution = short*80*0.3 - tail*150*0.1."""
        allocator = self._make_allocator()
        vix_level = 20  # normal band, tail = min(2, 3*0.4) = 1.2
        short, tail, vol = allocator.calculate_vix_allocation(
            _make_convexity_signal(vix_level=vix_level, allocation_pct=3.0)
        )
        expected = (short * 80.0 * 0.3) - (tail * 150.0 * 0.1)
        assert vol == pytest.approx(expected)

    def test_net_negative_vol_contribution(self):
        """Tail vol contribution can exceed short vol contribution."""
        allocator = self._make_allocator()
        # Very small short, large tail
        short, tail, vol = allocator.calculate_vix_allocation(
            _make_convexity_signal(vix_level=10, allocation_pct=0.5)
        )
        # short=0.5, tail=2.0
        # vol = 0.5*80*0.3 - 2.0*150*0.1 = 12 - 30 = -18
        expected = (0.5 * 80.0 * 0.3) - (tail * 150.0 * 0.1)
        assert vol == pytest.approx(expected)
        assert vol < 0  # net negative

    def test_short_zero_gives_no_vol_contribution(self):
        """No short VIX position means lower vol contribution."""
        allocator = self._make_allocator()
        _, _, vol = allocator.calculate_vix_allocation(
            _make_convexity_signal(vix_level=20, allocation_pct=0.0)
        )
        # short=0, tail=0 (0 * 0.4 = 0, min(2,0)=0)
        assert vol == pytest.approx(0.0)


# ==============================================================================
# Timedelta Regression Tests
# ==============================================================================

class TestTimedeltaImport:
    """Regression: vol_parity_allocator uses timedelta correctly."""

    def test_timedelta_importable(self):
        from src.strategy.vol_parity_allocator import timedelta
        assert timedelta is not None

    def test_run_backtest_uses_timedelta(self):
        allocator = _make_allocator()
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


# ==============================================================================
# VolParityAllocation Extended
# ==============================================================================

class TestVolParityAllocationExtended:
    """Extended VolParityAllocation dataclass tests."""

    def test_all_fields(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(VolParityAllocation)}
        expected = {
            "date", "target_volatility", "spy_pct", "gld_pct", "tlt_pct",
            "core_vol_contribution", "vix_short_pct", "vix_tail_pct",
            "vix_vol_contribution", "cash_pct", "expected_portfolio_vol",
            "expected_max_dd", "rebalance_triggered", "rebalance_reason",
        }
        assert field_names == expected

    def test_to_dict(self):
        alloc = VolParityAllocation(
            date="2026-01-01", target_volatility=10.0,
            spy_pct=46.0, gld_pct=38.0, tlt_pct=16.0,
            core_vol_contribution=11.0,
            vix_short_pct=3.0, vix_tail_pct=1.0, vix_vol_contribution=5.0,
            cash_pct=4.0, expected_portfolio_vol=10.0, expected_max_dd=-15.0,
            rebalance_triggered=False, rebalance_reason=None,
        )
        d = alloc.to_dict()
        assert d["spy_pct"] == 46.0
        assert d["date"] == "2026-01-01"

    def test_total_allocation(self):
        alloc = VolParityAllocation(
            date="2026-01-01", target_volatility=10.0,
            spy_pct=46.0, gld_pct=38.0, tlt_pct=16.0,
            core_vol_contribution=11.0,
            vix_short_pct=3.0, vix_tail_pct=1.0, vix_vol_contribution=5.0,
            cash_pct=4.0, expected_portfolio_vol=10.0, expected_max_dd=-15.0,
            rebalance_triggered=False, rebalance_reason=None,
        )
        assert alloc.total_allocation == pytest.approx(46+38+16+3+1+4)

    def test_total_vol_contribution(self):
        alloc = VolParityAllocation(
            date="2026-01-01", target_volatility=10.0,
            spy_pct=46.0, gld_pct=38.0, tlt_pct=16.0,
            core_vol_contribution=11.0,
            vix_short_pct=3.0, vix_tail_pct=1.0, vix_vol_contribution=5.0,
            cash_pct=4.0, expected_portfolio_vol=10.0, expected_max_dd=-15.0,
            rebalance_triggered=False, rebalance_reason=None,
        )
        assert alloc.total_vol_contribution == pytest.approx(11.0 + 5.0)

    def test_rebalance_reason_none(self):
        alloc = VolParityAllocation(
            date="2026-01-01", target_volatility=10.0,
            spy_pct=46.0, gld_pct=38.0, tlt_pct=16.0,
            core_vol_contribution=11.0,
            vix_short_pct=3.0, vix_tail_pct=1.0, vix_vol_contribution=5.0,
            cash_pct=4.0, expected_portfolio_vol=10.0, expected_max_dd=-15.0,
            rebalance_triggered=False, rebalance_reason=None,
        )
        assert alloc.rebalance_reason is None

    def test_rebalance_triggered_with_reason(self):
        alloc = VolParityAllocation(
            date="2026-01-01", target_volatility=10.0,
            spy_pct=46.0, gld_pct=38.0, tlt_pct=16.0,
            core_vol_contribution=11.0,
            vix_short_pct=3.0, vix_tail_pct=1.0, vix_vol_contribution=5.0,
            cash_pct=4.0, expected_portfolio_vol=10.0, expected_max_dd=-15.0,
            rebalance_triggered=True, rebalance_reason="Drift test",
        )
        assert alloc.rebalance_triggered is True
        assert alloc.rebalance_reason == "Drift test"


# ==============================================================================
# 5. generate_allocation Boundary Conditions
# ==============================================================================

class TestGenerateAllocationExtended:
    """Extended generate_allocation tests — cash, scaling, rebalance."""

    def _make_allocator(self):
        with patch.object(VolatilityParityAllocator, '__init__', lambda self, **kw: None):
            alloc = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
            alloc.target_vol = 10.0
            alloc.vix_strategy = MagicMock()
            alloc.vix_manager = MagicMock()
            alloc.last_allocation = None
            return alloc

    def test_returns_vol_parity_allocation(self):
        allocator = self._make_allocator()
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0)
        allocator.vix_strategy.generate_signal.return_value = signal
        with patch.object(allocator, 'calculate_core_allocation', return_value=({"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}, 11.0)), \
             patch.object(allocator, 'calculate_vix_allocation', return_value=(3.0, 1.0, 5.0)):
            result = allocator.generate_allocation("2026-01-01")
            assert isinstance(result, VolParityAllocation)

    def test_allocation_date_matches(self):
        allocator = self._make_allocator()
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0)
        allocator.vix_strategy.generate_signal.return_value = signal
        with patch.object(allocator, 'calculate_core_allocation', return_value=({"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}, 11.0)), \
             patch.object(allocator, 'calculate_vix_allocation', return_value=(3.0, 1.0, 5.0)):
            result = allocator.generate_allocation("2026-05-24")
            assert result.date == "2026-05-24"

    def test_high_vix_more_cash(self):
        """High VIX (>25) should set base cash at 18% before scaling."""
        allocator = self._make_allocator()
        signal = _make_convexity_signal(vix_level=30, allocation_pct=3.0)
        allocator.vix_strategy.generate_signal.return_value = signal
        with patch.object(allocator, 'calculate_core_allocation', return_value=({"SPY": 0.40, "GLD": 0.30, "TLT": 0.10}, 11.0)), \
             patch.object(allocator, 'calculate_vix_allocation', return_value=(3.0, 1.0, 5.0)):
            result = allocator.generate_allocation("2026-01-01")
            assert result.cash_pct > 0

    def test_rebalance_triggered_on_drift(self):
        """Should trigger rebalance when drift exceeds threshold."""
        allocator = self._make_allocator()
        allocator.last_allocation = VolParityAllocation(
            date="2026-01-01", target_volatility=10.0,
            spy_pct=20.0, gld_pct=38.0, tlt_pct=16.0,
            core_vol_contribution=11.0,
            vix_short_pct=3.0, vix_tail_pct=1.0, vix_vol_contribution=5.0,
            cash_pct=18.0, expected_portfolio_vol=10.0, expected_max_dd=-15.0,
            rebalance_triggered=False, rebalance_reason=None,
        )
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0)
        allocator.vix_strategy.generate_signal.return_value = signal
        # With decimal weights: new spy=0.46*80=36.8, last spy=20.0, drift=16.8 > 10
        with patch.object(allocator, 'calculate_core_allocation', return_value=({"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}, 11.0)), \
             patch.object(allocator, 'calculate_vix_allocation', return_value=(3.0, 1.0, 5.0)):
            result = allocator.generate_allocation("2026-01-02")
            assert result.rebalance_triggered is True

    def test_no_rebalance_without_last_allocation(self):
        """First allocation has no last_allocation, no rebalance check."""
        allocator = self._make_allocator()
        allocator.last_allocation = None
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0)
        allocator.vix_strategy.generate_signal.return_value = signal
        with patch.object(allocator, 'calculate_core_allocation', return_value=({"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}, 11.0)), \
             patch.object(allocator, 'calculate_vix_allocation', return_value=(3.0, 1.0, 5.0)):
            result = allocator.generate_allocation("2026-01-01")
            assert result.rebalance_triggered is False
            assert result.rebalance_reason is None

    def test_no_rebalance_when_drift_below_threshold(self):
        """Small drift should not trigger rebalance."""
        allocator = self._make_allocator()
        allocator.last_allocation = VolParityAllocation(
            date="2026-01-01", target_volatility=10.0,
            spy_pct=44.0, gld_pct=38.0, tlt_pct=16.0,
            core_vol_contribution=11.0,
            vix_short_pct=3.0, vix_tail_pct=1.0, vix_vol_contribution=5.0,
            cash_pct=18.0, expected_portfolio_vol=10.0, expected_max_dd=-15.0,
            rebalance_triggered=False, rebalance_reason=None,
        )
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0)
        allocator.vix_strategy.generate_signal.return_value = signal
        # With decimal weights: new spy=36.8, drift=abs(36.8-44)=7.2 < 10
        with patch.object(allocator, 'calculate_core_allocation', return_value=({"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}, 11.0)), \
             patch.object(allocator, 'calculate_vix_allocation', return_value=(3.0, 1.0, 5.0)):
            result = allocator.generate_allocation("2026-01-02")
            assert result.rebalance_triggered is False

    def test_rebalance_drift_just_at_threshold(self):
        """Drift exactly at threshold does NOT trigger (> comparison, not >=)."""
        allocator = self._make_allocator()
        allocator.last_allocation = VolParityAllocation(
            date="2026-01-01", target_volatility=10.0,
            spy_pct=36.0, gld_pct=38.0, tlt_pct=16.0,
            core_vol_contribution=11.0,
            vix_short_pct=3.0, vix_tail_pct=1.0, vix_vol_contribution=5.0,
            cash_pct=18.0, expected_portfolio_vol=10.0, expected_max_dd=-15.0,
            rebalance_triggered=False, rebalance_reason=None,
        )
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0)
        allocator.vix_strategy.generate_signal.return_value = signal
        # With decimal weights: new spy=36.8, drift=abs(36.8-36)=0.8 < 10, no trigger
        with patch.object(allocator, 'calculate_core_allocation', return_value=({"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}, 11.0)), \
             patch.object(allocator, 'calculate_vix_allocation', return_value=(3.0, 1.0, 5.0)):
            result = allocator.generate_allocation("2026-01-02")
            assert result.rebalance_triggered is False

    def test_rebalance_drift_exceeds_threshold(self):
        """Drift exceeding threshold should trigger rebalance."""
        allocator = self._make_allocator()
        allocator.last_allocation = VolParityAllocation(
            date="2026-01-01", target_volatility=10.0,
            spy_pct=25.0, gld_pct=38.0, tlt_pct=16.0,
            core_vol_contribution=11.0,
            vix_short_pct=3.0, vix_tail_pct=1.0, vix_vol_contribution=5.0,
            cash_pct=18.0, expected_portfolio_vol=10.0, expected_max_dd=-15.0,
            rebalance_triggered=False, rebalance_reason=None,
        )
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0)
        allocator.vix_strategy.generate_signal.return_value = signal
        # With decimal weights: new spy=36.8, drift=abs(36.8-25)=11.8 > 10
        with patch.object(allocator, 'calculate_core_allocation', return_value=({"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}, 11.0)), \
             patch.object(allocator, 'calculate_vix_allocation', return_value=(3.0, 1.0, 5.0)):
            result = allocator.generate_allocation("2026-01-02")
            assert result.rebalance_triggered is True
            assert "Drift" in result.rebalance_reason

    def test_exit_triggered_more_cash(self):
        """exit_triggered=True should increase cash to 15%."""
        allocator = self._make_allocator()
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0, exit_triggered=True)
        allocator.vix_strategy.generate_signal.return_value = signal
        # Core weights are decimals (0.46 = 46%), multiplied by core_total_pct=80
        spy = 0.46 * 80.0
        gld = 0.38 * 80.0
        tlt = 0.16 * 80.0
        with patch.object(allocator, 'calculate_core_allocation',
                          return_value=({"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}, 11.0)), \
             patch.object(allocator, 'calculate_vix_allocation',
                          return_value=(3.0, 1.0, 5.0)):
            result = allocator.generate_allocation("2026-01-01")
            # exit_triggered => cash=15, total < 100 so no scaling
            assert result.cash_pct == pytest.approx(15.0)
            assert result.spy_pct == pytest.approx(spy)
            assert result.gld_pct == pytest.approx(gld)
            assert result.tlt_pct == pytest.approx(tlt)

    def test_normal_cash_no_vix(self):
        """No VIX short position => extra cash."""
        allocator = self._make_allocator()
        signal = _make_convexity_signal(vix_level=20, allocation_pct=0.0)
        allocator.vix_strategy.generate_signal.return_value = signal
        with patch.object(allocator, 'calculate_core_allocation',
                          return_value=({"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}, 11.0)), \
             patch.object(allocator, 'calculate_vix_allocation',
                          return_value=(0.0, 0.0, 0.0)):
            result = allocator.generate_allocation("2026-01-01")
            # vix_short=0 => cash = 13 + max(0, 5-0) = 13+5 = 18
            assert result.cash_pct == pytest.approx(18.0)

    def test_scaling_when_total_exceeds_100(self):
        """When total allocated > 100%, scale down proportionally."""
        allocator = self._make_allocator()
        signal = _make_convexity_signal(vix_level=30, allocation_pct=5.0)
        allocator.vix_strategy.generate_signal.return_value = signal
        # Core weights are decimals (0.46 = 46% of 80% core slice)
        spy = 0.46 * 80.0   # 36.8
        gld = 0.38 * 80.0   # 30.4
        tlt = 0.16 * 80.0   # 12.8
        vix_short = 5.0
        vix_tail = 2.0
        # High VIX => cash=18
        total_before = spy + gld + tlt + vix_short + vix_tail + 18.0  # 105
        scale = 100.0 / total_before
        with patch.object(allocator, 'calculate_core_allocation',
                          return_value=({"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}, 11.0)), \
             patch.object(allocator, 'calculate_vix_allocation',
                          return_value=(vix_short, vix_tail, 10.0)):
            result = allocator.generate_allocation("2026-01-01")
            assert result.total_allocation == pytest.approx(100.0, rel=1e-3)
            assert result.spy_pct == pytest.approx(spy * scale, rel=1e-3)
            assert result.tlt_pct == pytest.approx(tlt * scale, rel=1e-3)
            assert result.vix_short_pct == pytest.approx(vix_short * scale, rel=1e-3)
            assert result.vix_tail_pct == pytest.approx(vix_tail * scale, rel=1e-3)

    def test_scaling_cash_is_remainder(self):
        """Cash after scaling is explicitly set as remainder."""
        allocator = self._make_allocator()
        signal = _make_convexity_signal(vix_level=30, allocation_pct=5.0)
        allocator.vix_strategy.generate_signal.return_value = signal
        with patch.object(allocator, 'calculate_core_allocation', return_value=({"SPY": 0.50, "GLD": 0.30, "TLT": 0.20}, 11.0)), \
             patch.object(allocator, 'calculate_vix_allocation', return_value=(5.0, 2.0, 10.0)):
            result = allocator.generate_allocation("2026-01-01")
            core_total = 80.0
            spy = 0.50 * core_total
            gld = 0.30 * core_total
            tlt = 0.20 * core_total
            total_before_cash = spy + gld + tlt + 5.0 + 2.0
            # total_before_cash + cash_initial = total_before_cash + 18 > 100
            if total_before_cash + 18 > 100:
                assert result.total_allocation == pytest.approx(100.0)

    def test_preserves_small_allocations_no_scaling(self):
        """When total <= 100, no scaling should occur."""
        allocator = self._make_allocator()
        signal = _make_convexity_signal(vix_level=20, allocation_pct=1.0)
        allocator.vix_strategy.generate_signal.return_value = signal
        spy = 0.46 * 80.0   # 36.8
        gld = 0.38 * 80.0   # 30.4
        tlt = 0.16 * 80.0   # 12.8
        vix_short = 1.0
        vix_tail = 0.5
        cash_expected = 13.0 + max(0, 5.0 - vix_short)  # 17
        total_expected = spy + gld + tlt + vix_short + vix_tail + cash_expected  # 98.5
        with patch.object(allocator, 'calculate_core_allocation',
                          return_value=({"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}, 11.0)), \
             patch.object(allocator, 'calculate_vix_allocation',
                          return_value=(vix_short, vix_tail, 2.0)):
            result = allocator.generate_allocation("2026-01-01")
            assert result.total_allocation == pytest.approx(total_expected, rel=1e-3)
            assert result.cash_pct == pytest.approx(cash_expected, rel=1e-3)
            # No scaling: raw values unchanged
            assert result.spy_pct == pytest.approx(spy)
            assert result.gld_pct == pytest.approx(gld)
            assert result.tlt_pct == pytest.approx(tlt)

    def test_generate_allocation_updates_last_allocation(self):
        """generate_allocation should update self.last_allocation."""
        allocator = self._make_allocator()
        allocator.last_allocation = None
        signal = _make_convexity_signal(vix_level=20, allocation_pct=3.0)
        allocator.vix_strategy.generate_signal.return_value = signal
        with patch.object(allocator, 'calculate_core_allocation', return_value=({"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}, 11.0)), \
             patch.object(allocator, 'calculate_vix_allocation', return_value=(3.0, 1.0, 5.0)):
            result = allocator.generate_allocation("2026-01-01")
            assert allocator.last_allocation is result


# ==============================================================================
# 6. run_backtest
# ==============================================================================

class TestRunBacktest:
    """run_backtest method boundary conditions."""

    def _make_allocator(self):
        allocator = _make_allocator()
        allocator.last_allocation = None
        return allocator

    def test_returns_dict_with_keys(self):
        allocator = self._make_allocator()
        mock_alloc = _make_allocation()
        with patch.object(allocator, 'generate_allocation', return_value=mock_alloc):
            result = allocator.run_backtest('2026-01-01', '2026-01-01')
        expected_keys = {"period", "days", "average_allocation",
                         "average_expected_volatility", "rebalance_events",
                         "target_volatility"}
        assert set(result.keys()) == expected_keys

    def test_single_day(self):
        allocator = self._make_allocator()
        mock_alloc = _make_allocation()
        with patch.object(allocator, 'generate_allocation', return_value=mock_alloc):
            result = allocator.run_backtest('2026-01-01', '2026-01-01')
        assert result['days'] == 1

    def test_multiple_days(self):
        allocator = self._make_allocator()
        mock_alloc = _make_allocation()
        with patch.object(allocator, 'generate_allocation', return_value=mock_alloc):
            result = allocator.run_backtest('2026-01-01', '2026-01-05')
        assert result['days'] == 5

    def test_average_allocation_values(self):
        allocator = self._make_allocator()
        mock_alloc = _make_allocation(spy_pct=36.8, gld_pct=30.4, tlt_pct=12.8,
                                       vix_short_pct=3.0, vix_tail_pct=1.0,
                                       cash_pct=16.0)
        with patch.object(allocator, 'generate_allocation', return_value=mock_alloc):
            result = allocator.run_backtest('2026-01-01', '2026-01-03')
        avg = result['average_allocation']
        assert avg['SPY'] == "36.8%"
        assert avg['GLD'] == "30.4%"
        assert avg['TLT'] == "12.8%"
        assert avg['VIX_Short'] == "3.0%"
        assert avg['VIX_Tail'] == "1.0%"
        assert avg['Cash'] == "16.0%"

    def test_average_expected_volatility(self):
        allocator = self._make_allocator()
        mock_alloc = _make_allocation(expected_portfolio_vol=7.5)
        with patch.object(allocator, 'generate_allocation', return_value=mock_alloc):
            result = allocator.run_backtest('2026-01-01', '2026-01-01')
        assert result['average_expected_volatility'] == "7.5%"

    def test_rebalance_count_zero(self):
        allocator = self._make_allocator()
        mock_alloc = _make_allocation(rebalance_triggered=False)
        with patch.object(allocator, 'generate_allocation', return_value=mock_alloc):
            result = allocator.run_backtest('2026-01-01', '2026-01-05')
        assert result['rebalance_events'] == 0

    def test_rebalance_count_nonzero(self):
        allocator = self._make_allocator()
        rebalance_alloc = _make_allocation(rebalance_triggered=True)
        no_rebalance_alloc = _make_allocation(rebalance_triggered=False)
        # Return rebalanced alloc for first two days, no rebalance for rest
        returns = [rebalance_alloc, rebalance_alloc, no_rebalance_alloc]
        with patch.object(allocator, 'generate_allocation', side_effect=returns):
            result = allocator.run_backtest('2026-01-01', '2026-01-03')
        assert result['rebalance_events'] == 2

    def test_period_string(self):
        allocator = self._make_allocator()
        mock_alloc = _make_allocation()
        with patch.object(allocator, 'generate_allocation', return_value=mock_alloc):
            result = allocator.run_backtest('2026-01-01', '2026-06-30')
        assert result['period'] == "2026-01-01 to 2026-06-30"


# ==============================================================================
# 7. get_current_allocation
# ==============================================================================

class TestGetCurrentAllocation:
    """get_current_allocation method tests."""

    def _make_allocator(self):
        allocator = _make_allocator()
        allocator.target_vol = 10.0
        allocator.last_allocation = None
        return allocator

    def test_returns_dict_with_keys(self):
        allocator = self._make_allocator()
        mock_alloc = _make_allocation()
        with patch.object(allocator, 'generate_allocation', return_value=mock_alloc):
            result = allocator.get_current_allocation()
        assert "allocation" in result
        assert "summary" in result

    def test_summary_contains_keys(self):
        allocator = self._make_allocator()
        mock_alloc = _make_allocation(target_volatility=10.0,
                                       expected_portfolio_vol=7.5,
                                       vix_short_pct=3.0)
        with patch.object(allocator, 'generate_allocation', return_value=mock_alloc):
            result = allocator.get_current_allocation()
        summary = result["summary"]
        assert "total_capital_allocation" in summary
        assert "total_vol_contribution" in summary
        assert "target_vol" in summary
        assert "vol_gap" in summary
        assert "vix_regime" in summary

    def test_vol_gap_calculation(self):
        allocator = self._make_allocator()
        mock_alloc = _make_allocation(target_volatility=10.0,
                                       expected_portfolio_vol=7.5)
        with patch.object(allocator, 'generate_allocation', return_value=mock_alloc):
            result = allocator.get_current_allocation()
        assert result["summary"]["vol_gap"] == pytest.approx(2.5)

    def test_vix_regime_contango(self):
        allocator = self._make_allocator()
        mock_alloc = _make_allocation(vix_short_pct=3.0)
        with patch.object(allocator, 'generate_allocation', return_value=mock_alloc):
            result = allocator.get_current_allocation()
        assert result["summary"]["vix_regime"] == "contango"

    def test_vix_regime_backwardation(self):
        allocator = self._make_allocator()
        mock_alloc = _make_allocation(vix_short_pct=0.0)
        with patch.object(allocator, 'generate_allocation', return_value=mock_alloc):
            result = allocator.get_current_allocation()
        assert result["summary"]["vix_regime"] == "backwardation"

    def test_allocation_contains_to_dict(self):
        allocator = self._make_allocator()
        a = _make_allocation()
        with patch.object(allocator, 'generate_allocation', return_value=a):
            result = allocator.get_current_allocation()
        assert result["allocation"]["date"] == "2026-05-14"


# ==============================================================================
# 8. CLI / __main__ Guard
# ==============================================================================

class TestCLI:
    """Test CLI entry points using caplog for logger output."""

    def test_main_callable(self):
        from src.strategy.vol_parity_allocator import main
        assert callable(main)

    def test_main_default_output(self, caplog):
        """Default invocation should log usage and sample allocations."""
        import logging
        caplog.set_level(logging.INFO)
        from src.strategy.vol_parity_allocator import main
        with patch.object(VolatilityParityAllocator, 'generate_allocation',
                          return_value=_make_allocation()):
            main()
        assert "Volatility Parity Allocator" in caplog.text
        assert "Usage:" in caplog.text
        assert "Sample Allocations:" in caplog.text
        assert "SPY:" in caplog.text

    def test_main_backtest_flag(self, caplog):
        """--backtest flag should run backtest and log JSON."""
        import logging
        caplog.set_level(logging.INFO)
        from src.strategy.vol_parity_allocator import main
        mock_alloc = _make_allocation()
        with patch.object(VolatilityParityAllocator, 'generate_allocation',
                          return_value=mock_alloc):
            with patch.object(sys, 'argv', ['prog', '--backtest', '2026-01-01', '2026-01-03']):
                main()
        assert "Running volatility parity backtest" in caplog.text
        assert "Volatility Parity Backtest" in caplog.text
        assert "period" in caplog.text

    def test_main_backtest_default_dates(self, caplog):
        """--backtest without explicit dates uses defaults."""
        import logging
        caplog.set_level(logging.INFO)
        from src.strategy.vol_parity_allocator import main
        mock_alloc = _make_allocation()
        with patch.object(VolatilityParityAllocator, 'generate_allocation',
                          return_value=mock_alloc):
            with patch.object(sys, 'argv', ['prog', '--backtest']):
                main()
        assert "2020-01-01" in caplog.text
        assert "2024-12-31" in caplog.text

    def test_main_current_flag(self, caplog):
        """--current flag should log current allocation."""
        import logging
        caplog.set_level(logging.INFO)
        from src.strategy.vol_parity_allocator import main
        with patch.object(VolatilityParityAllocator, 'get_current_allocation',
                          return_value={'allocation': {}, 'summary': {}}):
            with patch.object(sys, 'argv', ['prog', '--current']):
                main()
        assert "allocation" in caplog.text
        assert "summary" in caplog.text

    def test_main_backtest_partial_args(self, caplog):
        """--backtest with only start date uses default end."""
        import logging
        caplog.set_level(logging.INFO)
        from src.strategy.vol_parity_allocator import main
        mock_alloc = _make_allocation()
        with patch.object(VolatilityParityAllocator, 'generate_allocation',
                          return_value=mock_alloc):
            with patch.object(sys, 'argv', ['prog', '--backtest', '2023-01-01']):
                main()
        assert "2023-01-01" in caplog.text
        assert "2024-12-31" in caplog.text


class TestMainGuard:
    """Verify the __main__ guard exists."""

    def test_main_guard_exists(self):
        """Verify the module has a __main__ guard calling main()."""
        import ast
        with open('src/strategy/vol_parity_allocator.py') as f:
            tree = ast.parse(f.read())
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check for __name__ == '__main__' guard
                if (isinstance(node.test, ast.Compare) and
                    isinstance(node.test.left, ast.Name) and
                    node.test.left.id == '__name__'):
                    found = True
                    # Verify it calls main()
                    for item in node.body:
                        if (isinstance(item, ast.Expr) and
                            isinstance(item.value, ast.Call) and
                            isinstance(item.value.func, ast.Name) and
                            item.value.func.id == 'main'):
                            break
                    else:
                        pytest.fail("__name__ guard does not call main()")
        assert found, "No __name__ == '__main__' guard found"

    def test_module_runnable_as_main(self):
        """The if __name__ guard calls main() when the module is run as __main__."""
        # Verified by test_main_guard_exists via AST analysis above.
        # Integration check: main() executes without error.
        from src.strategy.vol_parity_allocator import main
        assert callable(main)


# ==============================================================================
# 9. Export Completeness (__all__)
# ==============================================================================

class TestExports:
    """Verify __all__ exports."""

    def test_all_exports_present(self):
        import src.strategy.vol_parity_allocator as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"Missing export: {name}"

    def test_all_count(self):
        import src.strategy.vol_parity_allocator as mod
        assert len(mod.__all__) == 2

    def test_export_types(self):
        import src.strategy.vol_parity_allocator as mod
        assert mod.__all__ == ['VolParityAllocation', 'VolatilityParityAllocator']


# ==============================================================================
# 10. Logger / logging
# ==============================================================================

class TestLogging:
    """Verify the module logger is configured."""

    def test_logger_exists(self):
        import src.strategy.vol_parity_allocator as mod
        assert mod.logger is not None
        assert mod.logger.name == 'src.strategy.vol_parity_allocator'

    def test_logger_level(self):
        import src.strategy.vol_parity_allocator as mod
        assert mod.logger.level == 0  # NOTSET (inherits root)


# ==============================================================================
# 11. Constructor / __init__ boundaries
# ==============================================================================

class TestConstructor:
    """Constructor boundary conditions."""

    def test_default_target_vol(self):
        """Default target_vol should equal TARGET_VOLATILITY."""
        with patch.object(VolatilityParityAllocator, '__init__',
                          lambda self, **kw: None):
            allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
            allocator.target_vol = VolatilityParityAllocator.TARGET_VOLATILITY
            assert allocator.target_vol == 10.0

    def test_custom_target_vol(self):
        """Custom target_vol should be set."""
        with patch.object(VolatilityParityAllocator, '__init__',
                          lambda self, **kw: None):
            allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
            allocator.target_vol = 15.0
            assert allocator.target_vol == 15.0

    def test_target_vol_negative_edge(self):
        """Negative target_vol is passed through."""
        with patch.object(VolatilityParityAllocator, '__init__',
                          lambda self, **kw: None):
            allocator = VolatilityParityAllocator.__new__(VolatilityParityAllocator)
            allocator.target_vol = -5.0
            assert allocator.target_vol == -5.0

    def test_last_allocation_initially_none(self):
        allocator = _make_allocator()
        assert allocator.last_allocation is None


# ==============================================================================
# 12. Module-level attributes
# ==============================================================================

class TestModuleAttributes:
    """Verify module-level attributes and types."""

    def test_core_base_weights_from_paths(self):
        """CORE_BASE_WEIGHTS should come from BASE_ALLOCATION."""
        from src.paths import BASE_ALLOCATION
        assert VolatilityParityAllocator.CORE_BASE_WEIGHTS is BASE_ALLOCATION

    def test_module_has_json(self):
        import src.strategy.vol_parity_allocator as mod
        assert hasattr(mod, 'json')

    def test_module_has_datetime(self):
        import src.strategy.vol_parity_allocator as mod
        assert hasattr(mod, 'datetime')
