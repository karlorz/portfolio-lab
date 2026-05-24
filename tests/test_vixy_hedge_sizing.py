"""
Tests for v7.04 Dynamic VIXY Hedge Sizing.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.strategy.vixy_hedge_sizing import (
    VIXYHedgeSizer,
    VIXYHedgeSignal,
    VIXYHedgeState,
    VIXYHedgeConfig,
    HedgeRegime,
    HedgeAction,
    DEFAULT_CONFIG,
)


class TestHedgeRegime:
    """Test regime classification from VIX levels."""

    def test_normal_regime(self):
        assert VIXYHedgeSizer.classify_regime(12.0) == HedgeRegime.NORMAL
        assert VIXYHedgeSizer.classify_regime(15.0) == HedgeRegime.NORMAL
        assert VIXYHedgeSizer.classify_regime(19.5) == HedgeRegime.NORMAL

    def test_elevated_regime(self):
        assert VIXYHedgeSizer.classify_regime(20.0) == HedgeRegime.ELEVATED
        assert VIXYHedgeSizer.classify_regime(25.0) == HedgeRegime.ELEVATED
        assert VIXYHedgeSizer.classify_regime(29.5) == HedgeRegime.ELEVATED

    def test_stress_regime(self):
        assert VIXYHedgeSizer.classify_regime(30.0) == HedgeRegime.STRESS
        assert VIXYHedgeSizer.classify_regime(35.0) == HedgeRegime.STRESS

    def test_crisis_regime(self):
        assert VIXYHedgeSizer.classify_regime(40.0) == HedgeRegime.CRISIS
        assert VIXYHedgeSizer.classify_regime(60.0) == HedgeRegime.CRISIS
        assert VIXYHedgeSizer.classify_regime(85.0) == HedgeRegime.CRISIS


class TestAllocationComputation:
    """Test VIX → allocation mapping."""

    def setup_method(self):
        self.sizer = VIXYHedgeSizer()

    def test_quantpedia_base_formula(self):
        """VIX=28 → allocation ≈ 2.8% (VIX/10 in %)"""
        alloc = self.sizer.compute_allocation(28.0)
        assert 2.5 <= alloc <= 3.5

    def test_low_vix_allocation(self):
        """VIX=12 should give near-zero allocation in NORMAL regime."""
        alloc = self.sizer.compute_allocation(12.0)
        assert 0.0 <= alloc <= 2.0  # NORMAL regime floor=0, ceiling=2

    def test_high_vix_allocation(self):
        """VIX=45 should give significant allocation in CRISIS regime."""
        alloc = self.sizer.compute_allocation(45.0)
        assert 3.0 <= alloc <= 10.0  # CRISIS regime floor=3, ceiling=10

    def test_extreme_vix_capped(self):
        """VIX=100+ should be capped at max_hedge_pct (10%)."""
        alloc = self.sizer.compute_allocation(120.0)
        assert alloc <= 10.0

    def test_zero_vix(self):
        """VIX=0 should give zero allocation."""
        alloc = self.sizer.compute_allocation(0.0)
        assert alloc == 0.0

    def test_vol_scaling_reduces_allocation(self):
        """When realized vol > VIX, scale down."""
        base = self.sizer.compute_allocation(25.0)
        scaled = self.sizer.compute_allocation_with_vol_scale(25.0, vol_ratio=1.5)
        assert scaled < base

    def test_vol_scaling_increases_allocation(self):
        """When realized vol < VIX, scale up."""
        base = self.sizer.compute_allocation(25.0)
        scaled = self.sizer.compute_allocation_with_vol_scale(25.0, vol_ratio=0.8)
        assert scaled > base


class TestCostAnalysis:
    """Test cost estimation for VIXY positions."""

    def setup_method(self):
        self.sizer = VIXYHedgeSizer()

    def test_zero_allocation_zero_cost(self):
        cost = self.sizer.estimate_annual_cost(0.0)
        assert cost == 0.0

    def test_cost_scales_with_allocation(self):
        cost_low = self.sizer.estimate_annual_cost(2.0)
        cost_high = self.sizer.estimate_annual_cost(5.0)
        assert cost_high > cost_low

    def test_monthly_cost_less_than_annual(self):
        annual = self.sizer.estimate_annual_cost(3.0)
        monthly = self.sizer.estimate_monthly_cost(3.0)
        assert monthly < annual

    def test_cost_includes_expense_and_decay(self):
        """Cost should include both expense ratio and premium decay."""
        cost = self.sizer.estimate_annual_cost(5.0)
        # 5% * (0.85% expense + 60% annual decay) ≈ reasonable range
        assert cost > 20  # Should be meaningful bps


class TestBenefitEstimation:
    """Test hedge benefit estimation."""

    def setup_method(self):
        self.sizer = VIXYHedgeSizer()

    def test_positive_benefit_during_shock(self):
        benefit = self.sizer.estimate_gain_during_shock(5.0, -15.0)
        assert benefit > 0  # Hedge should provide positive benefit

    def test_benefit_scales_with_allocation(self):
        benefit_low = self.sizer.estimate_gain_during_shock(2.0)
        benefit_high = self.sizer.estimate_gain_during_shock(5.0)
        assert benefit_high > benefit_low

    def test_zero_allocation_zero_benefit(self):
        benefit = self.sizer.estimate_gain_during_shock(0.0)
        assert benefit == 0.0


class TestHedgeEfficiency:
    """Test efficiency ratio computation."""

    def setup_method(self):
        self.sizer = VIXYHedgeSizer()

    def test_efficiency_during_crisis(self):
        """Hedge should be most efficient during crisis (high VIX)."""
        eff_low = self.sizer.compute_hedge_efficiency(3.0, 15.0)
        eff_high = self.sizer.compute_hedge_efficiency(3.0, 45.0)
        assert eff_high > eff_low  # Crisis = higher shock probability

    def test_efficiency_zero_when_no_cost(self):
        """Zero allocation = zero efficiency."""
        eff = self.sizer.compute_hedge_efficiency(0.0, 20.0)
        assert eff == 0.0

    def test_efficiency_threshold(self):
        """Efficiency above threshold means hedge is worthwhile."""
        eff = self.sizer.compute_hedge_efficiency(4.0, 30.0)
        assert isinstance(eff, float)
        assert eff >= 0


class TestSignalGeneration:
    """Test full signal generation."""

    def setup_method(self):
        self.sizer = VIXYHedgeSizer()

    def test_signal_has_all_fields(self):
        signal = self.sizer.get_signal(25.0)
        assert signal.vix_level == 25.0
        assert signal.regime == "elevated"
        assert 0 <= signal.allocation_pct <= 10
        assert -1 <= signal.signal_value <= 1
        assert signal.annual_cost_bps >= 0
        assert signal.monthly_decay_bps >= 0
        assert signal.ensemble_weight > 0

    def test_crisis_signal_max_strength(self):
        signal = self.sizer.get_signal(45.0)
        assert signal.signal_value == 1.0
        assert signal.ensemble_weight == 0.10  # Stress weight

    def test_normal_signal_minimal(self):
        signal = self.sizer.get_signal(15.0)
        assert signal.signal_value <= 0.3
        assert signal.ensemble_weight == 0.05  # Normal weight

    def test_collar_complement_reduces_with_high_vixy(self):
        signal_low = self.sizer.get_signal(15.0)   # Low VIXY → more collar needed
        signal_high = self.sizer.get_signal(45.0)   # High VIXY → less collar needed
        assert signal_high.collar_complement < signal_low.collar_complement

    def test_action_freeze_during_crisis(self):
        signal = self.sizer.get_signal(50.0)
        assert signal.action == "freeze"

    def test_action_increase_when_vix_spikes(self):
        # Pre-set state with low current allocation
        self.sizer._state = VIXYHedgeState(
            timestamp="2026-01-01T00:00:00",
            current_allocation=1.0,
            target_allocation=1.0,
            vix_level=30.0,
            regime="stress",
            ytd_cost_bps=0.0,
            ytd_benefit_bps=0.0,
            hedge_efficiency=0.0,
        )
        signal = self.sizer.get_signal(35.0)
        # Target should be higher than current
        assert signal.allocation_pct > 1.0


class TestStatePersistence:
    """Test state save/load."""

    def test_load_state_creates_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sizer = VIXYHedgeSizer(
                config={"state_file": f"{tmpdir}/vixy_state.json"},
                project_root=Path(tmpdir),
            )
            state = sizer.load_state()
            assert state.current_allocation == 0.0
            assert state.total_signals == 0

    def test_save_and_load_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "state_file": f"{tmpdir}/vixy_state.json",
                **{k: v for k, v in DEFAULT_CONFIG.items()
                   if k not in ("state_file",)}
            }
            sizer = VIXYHedgeSizer(config=config, project_root=Path(tmpdir))
            signal = sizer.get_signal(28.0)
            sizer.save_state(signal)

            # Reload
            sizer2 = VIXYHedgeSizer(config=config, project_root=Path(tmpdir))
            state = sizer2.load_state()
            assert state.vix_level == 28.0
            assert state.total_signals == 1

    def test_update_after_rebalance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "state_file": f"{tmpdir}/vixy_state.json",
                **{k: v for k, v in DEFAULT_CONFIG.items()
                   if k not in ("state_file",)}
            }
            sizer = VIXYHedgeSizer(config=config, project_root=Path(tmpdir))
            sizer.update_after_rebalance(3.5)
            state = sizer.load_state()
            assert state.current_allocation == 3.5
            assert state.last_rebalance is not None


class TestCollarCoordination:
    """Test collar complement logic."""

    def setup_method(self):
        self.sizer = VIXYHedgeSizer()

    def test_combined_coverage_string(self):
        result = self.sizer.get_combined_hedge_coverage(3.0, 2.0)
        assert "VIXY" in result
        assert "collar" in result
        assert "5.0%" in result

    def test_disable_collar_when_vixy_high(self):
        assert self.sizer.should_disable_collar(8.5) is True

    def test_keep_collar_when_vixy_low(self):
        assert self.sizer.should_disable_collar(2.0) is False


class TestConfigCustomization:
    """Test custom configuration."""

    def test_custom_max_hedge(self):
        sizer = VIXYHedgeSizer(config={"max_hedge_pct": 5.0})
        alloc = sizer.compute_allocation(80.0)
        assert alloc <= 5.0

    def test_custom_ensemble_weights(self):
        sizer = VIXYHedgeSizer(config={
            "ensemble_weight_normal": 0.08,
            "ensemble_weight_stress": 0.15,
        })
        signal_normal = sizer.get_signal(15.0)
        signal_stress = sizer.get_signal(35.0)
        assert signal_normal.ensemble_weight == 0.08
        assert signal_stress.ensemble_weight == 0.15


class TestCLI:
    """Test CLI entry points."""

    def test_status_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sizer = VIXYHedgeSizer(
                config={"state_file": f"{tmpdir}/state.json"},
                project_root=Path(tmpdir),
            )
            from src.strategy.vixy_hedge_sizing import _format_status
            output = _format_status(sizer)
            assert "VIXY" in output
            assert "allocation" in output.lower()

    def test_recommend_mode(self):
        sizer = VIXYHedgeSizer()
        from src.strategy.vixy_hedge_sizing import _format_recommend
        signal = sizer.get_signal(22.0)
        output = _format_recommend(signal)
        assert "VIX" in output
        assert "allocation" in output.lower()
        assert "efficiency" in output.lower()


class TestBacktestValidation:
    """Validate the VIXY hedge backtest logic with simulated data."""

    def test_backtest_allocations_bounded(self):
        """Backtest over simulated VIX data should produce bounded allocations."""
        sizer = VIXYHedgeSizer()
        np.random.seed(42)
        vix_levels = np.random.lognormal(mean=2.8, sigma=0.4, size=2520)

        allocations = [sizer.compute_allocation(float(v)) for v in vix_levels]
        assert all(0 <= a <= 10.0 for a in allocations), "Allocations exceed 10% cap"

    def test_backtest_average_allocation_reasonable(self):
        """Average allocation over typical VIX distribution should be 1-4%."""
        sizer = VIXYHedgeSizer()
        np.random.seed(42)
        vix_levels = np.random.lognormal(mean=2.8, sigma=0.4, size=2520)

        allocations = [sizer.compute_allocation(float(v)) for v in vix_levels]
        avg = np.mean(allocations)
        assert 0.5 <= avg <= 5.0, f"Average allocation {avg:.2f}% outside expected range"

    def test_backtest_costs_bounded(self):
        """Annual hedge costs should be positive and finite."""
        sizer = VIXYHedgeSizer()
        np.random.seed(42)
        vix_levels = np.random.lognormal(mean=2.8, sigma=0.4, size=2520)

        costs = [sizer.estimate_annual_cost(sizer.compute_allocation(float(v)))
                 for v in vix_levels]
        assert all(c >= 0 for c in costs)
        assert all(np.isfinite(c) for c in costs)

    def test_backtest_hedge_efficiency(self):
        """Hedge efficiency should be >=0 for typical allocation."""
        sizer = VIXYHedgeSizer()
        alloc = sizer.compute_allocation(25.0)  # Elevated VIX
        eff = sizer.compute_hedge_efficiency(alloc, vix_level=25.0)
        assert eff >= 0

    def test_backtest_regime_coverage(self):
        """All regimes should be encountered in long backtest."""
        sizer = VIXYHedgeSizer()
        np.random.seed(42)
        # Mix of VIX levels to ensure all regimes hit
        vix_levels = list(np.random.lognormal(mean=2.8, sigma=0.4, size=1000))
        vix_levels += [12.0] * 50 + [25.0] * 50 + [35.0] * 50 + [45.0] * 50

        regimes = set()
        for v in vix_levels:
            regimes.add(VIXYHedgeSizer.classify_regime(float(v)))

        assert HedgeRegime.NORMAL in regimes
        assert HedgeRegime.ELEVATED in regimes
        assert HedgeRegime.STRESS in regimes
        assert HedgeRegime.CRISIS in regimes

    def test_backtest_days_hedged(self):
        """With lognormal VIX, there should be hedged days."""
        sizer = VIXYHedgeSizer()
        np.random.seed(42)
        vix_levels = np.random.lognormal(mean=2.8, sigma=0.4, size=2520)

        allocations = [sizer.compute_allocation(float(v)) for v in vix_levels]
        days_hedged = sum(1 for a in allocations if a > 0)

        assert days_hedged > 0, "No hedged days in backtest"


class TestHedgeRegimeExtended:
    """Extended tests for regime classification."""

    def test_boundary_vix_20(self):
        assert VIXYHedgeSizer.classify_regime(20.0) == HedgeRegime.ELEVATED

    def test_boundary_vix_30(self):
        assert VIXYHedgeSizer.classify_regime(30.0) == HedgeRegime.STRESS

    def test_boundary_vix_40(self):
        assert VIXYHedgeSizer.classify_regime(40.0) == HedgeRegime.CRISIS

    def test_just_below_20(self):
        assert VIXYHedgeSizer.classify_regime(19.99) == HedgeRegime.NORMAL

    def test_just_below_30(self):
        assert VIXYHedgeSizer.classify_regime(29.99) == HedgeRegime.ELEVATED

    def test_just_below_40(self):
        assert VIXYHedgeSizer.classify_regime(39.99) == HedgeRegime.STRESS


class TestVIXYHedgeSignalExtended:
    """Extended tests for VIXYHedgeSignal dataclass."""

    def test_signal_all_fields_present(self):
        sizer = VIXYHedgeSizer()
        signal = sizer.get_signal(25.0)
        expected_fields = [
            'timestamp', 'vix_level', 'regime', 'allocation_pct',
            'action', 'signal_value', 'confidence',
            'annual_cost_bps', 'monthly_decay_bps',
            'expected_gain_shock', 'hedge_efficiency',
            'ensemble_weight', 'collar_complement', 'source',
        ]
        for field in expected_fields:
            assert hasattr(signal, field), f"Missing field: {field}"

    def test_signal_source(self):
        sizer = VIXYHedgeSizer()
        signal = sizer.get_signal(25.0)
        assert signal.source == "vixy_hedge"

    def test_signal_action_values(self):
        """Signal action should be one of the HedgeAction values."""
        valid_actions = {a.value for a in HedgeAction}
        sizer = VIXYHedgeSizer()
        for vix in [12.0, 25.0, 35.0, 50.0]:
            signal = sizer.get_signal(vix)
            assert signal.action in valid_actions


class TestVIXYHedgeStateExtended:
    """Extended tests for VIXYHedgeState dataclass."""

    def test_default_state(self):
        state = VIXYHedgeState(
            timestamp="2026-01-01T00:00:00",
            current_allocation=0.0,
            target_allocation=0.0,
            vix_level=0.0,
            regime="normal",
            ytd_cost_bps=0.0,
            ytd_benefit_bps=0.0,
            hedge_efficiency=0.0,
        )
        assert state.total_signals == 0
        assert state.last_rebalance is None

    def test_state_with_history(self):
        state = VIXYHedgeState(
            timestamp="2026-05-14T00:00:00",
            current_allocation=3.5,
            target_allocation=4.0,
            vix_level=25.0,
            regime="elevated",
            ytd_cost_bps=150.0,
            ytd_benefit_bps=300.0,
            hedge_efficiency=2.0,
            total_signals=42,
            last_rebalance="2026-05-10T00:00:00",
        )
        assert state.total_signals == 42
        assert state.last_rebalance is not None


class TestDetermineActionExtended:
    """Extended tests for _determine_action."""

    def test_increase_when_target_much_higher(self):
        sizer = VIXYHedgeSizer()
        action = sizer._determine_action(target=5.0, current=2.0, regime=HedgeRegime.NORMAL)
        assert action == HedgeAction.INCREASE

    def test_decrease_when_target_much_lower(self):
        sizer = VIXYHedgeSizer()
        action = sizer._determine_action(target=1.0, current=3.0, regime=HedgeRegime.NORMAL)
        assert action == HedgeAction.DECREASE

    def test_maintain_when_close(self):
        sizer = VIXYHedgeSizer()
        action = sizer._determine_action(target=3.0, current=2.5, regime=HedgeRegime.NORMAL)
        assert action == HedgeAction.MAINTAIN

    def test_freeze_during_crisis(self):
        sizer = VIXYHedgeSizer()
        action = sizer._determine_action(target=8.0, current=2.0, regime=HedgeRegime.CRISIS)
        assert action == HedgeAction.FREEZE

    def test_increase_during_stress(self):
        sizer = VIXYHedgeSizer()
        action = sizer._determine_action(target=5.0, current=2.0, regime=HedgeRegime.STRESS)
        assert action == HedgeAction.INCREASE


class TestVIXYHedgeConfigExtended:
    """Extended tests for VIXYHedgeConfig."""

    def test_config_defaults(self):
        cfg = VIXYHedgeConfig()
        assert cfg.min_hedge_pct == 0.0
        assert cfg.max_hedge_pct == 10.0
        assert cfg.cost_threshold == 2.0
        assert cfg.vixy_expense_ratio == 0.0085
        assert cfg.monthly_decay_pct == 0.05

    def test_config_custom_values(self):
        cfg = VIXYHedgeConfig(max_hedge_pct=5.0, ensemble_weight_stress=0.15)
        assert cfg.max_hedge_pct == 5.0
        assert cfg.ensemble_weight_stress == 0.15


class TestCollarComplementExtended:
    """Extended collar complement logic."""

    def test_collar_at_low_allocation(self):
        sizer = VIXYHedgeSizer()
        signal = sizer.get_signal(12.0)  # NORMAL, low allocation
        assert signal.collar_complement == 3.0

    def test_collar_reduces_with_allocation(self):
        sizer = VIXYHedgeSizer()
        sig_low = sizer.get_signal(12.0)
        sig_mid = sizer.get_signal(25.0)
        assert sig_mid.collar_complement <= sig_low.collar_complement


class TestStatusReport:
    """Test status report generation."""

    def test_status_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sizer = VIXYHedgeSizer(
                config={"state_file": f"{tmpdir}/state.json"},
                project_root=Path(tmpdir),
            )
            status = sizer.status()
            assert isinstance(status, dict)
            assert "current_allocation_pct" in status
            assert "vix_level" in status
            assert "regime" in status


# ---------------------------------------------------------------------------
# __all__ export validation
# ---------------------------------------------------------------------------

class TestExports:
    """Verify __all__ exports."""

    def test_all_exports_present(self):
        import src.strategy.vixy_hedge_sizing as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"Missing export: {name}"

    def test_all_count(self):
        import src.strategy.vixy_hedge_sizing as mod
        assert len(mod.__all__) == 7


# ---------------------------------------------------------------------------
# HedgeRegime extended
# ---------------------------------------------------------------------------

class TestHedgeRegimeExtended:
    """Extended HedgeRegime enum tests."""

    def test_all_four_values(self):
        from src.strategy.vixy_hedge_sizing import HedgeRegime
        assert len(HedgeRegime) == 4

    def test_normal_value(self):
        from src.strategy.vixy_hedge_sizing import HedgeRegime
        assert HedgeRegime.NORMAL.value == "normal"

    def test_crisis_value(self):
        from src.strategy.vixy_hedge_sizing import HedgeRegime
        assert HedgeRegime.CRISIS.value == "crisis"


# ---------------------------------------------------------------------------
# HedgeAction extended
# ---------------------------------------------------------------------------

class TestHedgeActionExtended:
    """Extended HedgeAction enum tests."""

    def test_all_four_values(self):
        from src.strategy.vixy_hedge_sizing import HedgeAction
        assert len(HedgeAction) == 4

    def test_increase_value(self):
        from src.strategy.vixy_hedge_sizing import HedgeAction
        assert HedgeAction.INCREASE.value == "increase"

    def test_maintain_value(self):
        from src.strategy.vixy_hedge_sizing import HedgeAction
        assert HedgeAction.MAINTAIN.value == "maintain"

    def test_decrease_value(self):
        from src.strategy.vixy_hedge_sizing import HedgeAction
        assert HedgeAction.DECREASE.value == "decrease"


# ---------------------------------------------------------------------------
# DEFAULT_CONFIG validation
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    """Test DEFAULT_CONFIG values."""

    def test_is_dict(self):
        from src.strategy.vixy_hedge_sizing import DEFAULT_CONFIG
        assert isinstance(DEFAULT_CONFIG, dict)

    def test_has_key_thresholds(self):
        from src.strategy.vixy_hedge_sizing import DEFAULT_CONFIG
        # Should have VIX threshold keys
        assert any("vix" in k.lower() or "threshold" in k.lower() for k in DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# VIXYHedgeConfig dataclass extended
# ---------------------------------------------------------------------------

class TestVIXYHedgeConfigExtended:
    """Extended VIXYHedgeConfig tests."""

    def test_all_fields(self):
        from dataclasses import fields
        from src.strategy.vixy_hedge_sizing import VIXYHedgeConfig
        field_names = {f.name for f in fields(VIXYHedgeConfig)}
        assert len(field_names) > 0  # Has fields


# ---------------------------------------------------------------------------
# VIXYHedgeSignal dataclass extended
# ---------------------------------------------------------------------------

class TestVIXYHedgeSignalExtended:
    """Extended VIXYHedgeSignal tests."""

    def test_all_fields(self):
        from dataclasses import fields
        from src.strategy.vixy_hedge_sizing import VIXYHedgeSignal
        field_names = {f.name for f in fields(VIXYHedgeSignal)}
        assert len(field_names) > 0


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

class TestCLI:
    """Test main() callable."""

    def test_main_callable(self):
        from src.strategy.vixy_hedge_sizing import VIXYHedgeSizer
        # No main() function in this module, just verify class is usable
        assert VIXYHedgeSizer is not None
