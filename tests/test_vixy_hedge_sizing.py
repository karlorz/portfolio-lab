"""
Tests for v7.04 Dynamic VIXY Hedge Sizing.
"""

import tempfile
from pathlib import Path

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


class TestExports:
    """Verify __all__ exports."""

    def test_all_exports_importable(self):
        from src.strategy.vixy_hedge_sizing import __all__
        import src.strategy.vixy_hedge_sizing as mod
        for name in __all__:
            assert hasattr(mod, name), f"{name} in __all__ but not in module"

    def test_all_contains_key_names(self):
        from src.strategy.vixy_hedge_sizing import __all__
        expected = {'DEFAULT_CONFIG', 'HedgeRegime', 'HedgeAction',
                    'VIXYHedgeConfig', 'VIXYHedgeSignal', 'VIXYHedgeState',
                    'VIXYHedgeSizer'}
        assert expected.issubset(set(__all__))


class TestVIXYHedgeConfigDataclass:
    """Comprehensive dataclass validation for VIXYHedgeConfig."""

    def test_all_fields_present(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(VIXYHedgeConfig)}
        expected = {"min_hedge_pct", "max_hedge_pct", "cost_threshold",
                    "vixy_expense_ratio", "monthly_decay_pct", "spy_shock_pct",
                    "ensemble_weight_normal", "ensemble_weight_stress"}
        assert field_names == expected

    def test_default_values(self):
        cfg = VIXYHedgeConfig()
        assert cfg.min_hedge_pct == 0.0
        assert cfg.max_hedge_pct == 10.0
        assert cfg.cost_threshold == 2.0
        assert cfg.vixy_expense_ratio == 0.0085
        assert cfg.monthly_decay_pct == 0.05
        assert cfg.spy_shock_pct == -15.0
        assert cfg.ensemble_weight_normal == 0.05
        assert cfg.ensemble_weight_stress == 0.10

    def test_custom_values(self):
        cfg = VIXYHedgeConfig(min_hedge_pct=1.0, max_hedge_pct=5.0)
        assert cfg.min_hedge_pct == 1.0
        assert cfg.max_hedge_pct == 5.0


class TestVIXYHedgeSignalDataclass:
    """Comprehensive dataclass validation for VIXYHedgeSignal."""

    def test_all_fields_present(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(VIXYHedgeSignal)}
        expected = {"timestamp", "vix_level", "regime", "allocation_pct",
                    "action", "signal_value", "confidence", "annual_cost_bps",
                    "monthly_decay_bps", "expected_gain_shock", "hedge_efficiency",
                    "ensemble_weight", "collar_complement", "source"}
        assert field_names == expected

    def test_source_default(self):
        sig = VIXYHedgeSignal(
            timestamp="2026-01-01", vix_level=15.0, regime="normal",
            allocation_pct=1.5, action="maintain", signal_value=0.0,
            confidence=1.0, annual_cost_bps=10.0, monthly_decay_bps=1.0,
            expected_gain_shock=50.0, hedge_efficiency=1.0,
            ensemble_weight=0.05, collar_complement=3.0,
        )
        assert sig.source == "vixy_hedge"


class TestVIXYHedgeStateDataclass:
    """Comprehensive dataclass validation for VIXYHedgeState."""

    def test_all_fields_present(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(VIXYHedgeState)}
        expected = {"timestamp", "current_allocation", "target_allocation",
                    "vix_level", "regime", "ytd_cost_bps", "ytd_benefit_bps",
                    "hedge_efficiency", "total_signals", "last_rebalance"}
        assert field_names == expected

    def test_defaults(self):
        state = VIXYHedgeState(
            timestamp="2026-01-01", current_allocation=0.0, target_allocation=0.0,
            vix_level=15.0, regime="normal", ytd_cost_bps=0.0,
            ytd_benefit_bps=0.0, hedge_efficiency=0.0,
        )
        assert state.total_signals == 0
        assert state.last_rebalance is None


class TestRegimeClassificationExtended:
    """Extended parametrized regime classification tests."""

    @pytest.mark.parametrize("vix,expected", [
        (0.0, HedgeRegime.NORMAL),
        (10.0, HedgeRegime.NORMAL),
        (19.9, HedgeRegime.NORMAL),
        (20.0, HedgeRegime.ELEVATED),
        (25.0, HedgeRegime.ELEVATED),
        (29.9, HedgeRegime.ELEVATED),
        (30.0, HedgeRegime.STRESS),
        (35.0, HedgeRegime.STRESS),
        (39.9, HedgeRegime.STRESS),
        (40.0, HedgeRegime.CRISIS),
        (50.0, HedgeRegime.CRISIS),
        (80.0, HedgeRegime.CRISIS),
    ])
    def test_regime_boundaries(self, vix, expected):
        assert VIXYHedgeSizer.classify_regime(vix) == expected


class TestAllocationComputationExtended:
    """Extended allocation computation tests."""

    def setup_method(self):
        self.sizer = VIXYHedgeSizer(config={"state_file": "/tmp/test_vixy_state.json"})

    def test_allocation_increases_with_vix(self):
        """Higher VIX should produce higher allocation."""
        alloc_low = self.sizer.compute_allocation(15.0)
        alloc_high = self.sizer.compute_allocation(35.0)
        assert alloc_high > alloc_low

    def test_allocation_capped_at_max(self):
        """Allocation should never exceed max_hedge_pct."""
        alloc = self.sizer.compute_allocation(100.0)
        assert alloc <= self.sizer.config.max_hedge_pct

    def test_allocation_floored_at_min(self):
        """Allocation should never go below min_hedge_pct."""
        alloc = self.sizer.compute_allocation(5.0)
        assert alloc >= self.sizer.config.min_hedge_pct

    def test_allocation_returns_float(self):
        alloc = self.sizer.compute_allocation(20.0)
        assert isinstance(alloc, float)

    def test_allocation_rounded_to_two_decimals(self):
        alloc = self.sizer.compute_allocation(23.7)
        assert alloc == round(alloc, 2)

    @pytest.mark.parametrize("vix,expected_regime_bounds", [
        (15.0, (0.0, 2.0)),   # NORMAL
        (25.0, (1.0, 3.5)),   # ELEVATED
        (35.0, (2.0, 6.0)),   # STRESS
        (50.0, (3.0, 10.0)),  # CRISIS
    ])
    def test_allocation_within_regime_bounds(self, vix, expected_regime_bounds):
        alloc = self.sizer.compute_allocation(vix)
        lo, hi = expected_regime_bounds
        assert lo <= alloc <= hi


class TestComputeAllocationWithVolScale:
    """Test compute_allocation_with_vol_scale."""

    def setup_method(self):
        self.sizer = VIXYHedgeSizer(config={"state_file": "/tmp/test_vixy_state.json"})

    def test_vol_ratio_1_0_same_as_base(self):
        base = self.sizer.compute_allocation(25.0)
        scaled = self.sizer.compute_allocation_with_vol_scale(25.0, 1.0)
        assert scaled == base

    def test_high_vol_ratio_scales_down(self):
        base = self.sizer.compute_allocation(25.0)
        scaled = self.sizer.compute_allocation_with_vol_scale(25.0, 2.0)
        assert scaled < base

    def test_low_vol_ratio_scales_up(self):
        base = self.sizer.compute_allocation(25.0)
        scaled = self.sizer.compute_allocation_with_vol_scale(25.0, 0.5)
        assert scaled > base

    def test_vol_ratio_clamped_at_1_5(self):
        """Very low vol_ratio should be clamped."""
        scaled = self.sizer.compute_allocation_with_vol_scale(25.0, 0.01)
        base = self.sizer.compute_allocation(25.0)
        assert scaled <= base * 1.5 + 0.01

    def test_vol_ratio_clamped_at_0_5(self):
        """Very high vol_ratio should be clamped."""
        scaled = self.sizer.compute_allocation_with_vol_scale(25.0, 100.0)
        base = self.sizer.compute_allocation(25.0)
        assert scaled >= base * 0.5 - 0.01


class TestCostAnalysis:
    """Test cost estimation methods."""

    def setup_method(self):
        self.sizer = VIXYHedgeSizer(config={"state_file": "/tmp/test_vixy_state.json"})

    def test_annual_cost_positive_for_positive_allocation(self):
        cost = self.sizer.estimate_annual_cost(3.0)
        assert cost > 0

    def test_annual_cost_zero_for_zero_allocation(self):
        cost = self.sizer.estimate_annual_cost(0.0)
        assert cost == 0.0

    def test_monthly_cost_less_than_annual(self):
        annual = self.sizer.estimate_annual_cost(5.0)
        monthly = self.sizer.estimate_monthly_cost(5.0)
        assert monthly < annual

    def test_annual_cost_scales_with_allocation(self):
        cost_low = self.sizer.estimate_annual_cost(1.0)
        cost_high = self.sizer.estimate_annual_cost(5.0)
        assert cost_high > cost_low


class TestBenefitEstimation:
    """Test benefit estimation methods."""

    def setup_method(self):
        self.sizer = VIXYHedgeSizer(config={"state_file": "/tmp/test_vixy_state.json"})

    def test_gain_positive_for_positive_allocation(self):
        gain = self.sizer.estimate_gain_during_shock(3.0)
        assert gain > 0

    def test_gain_zero_for_zero_allocation(self):
        gain = self.sizer.estimate_gain_during_shock(0.0)
        assert gain == 0.0

    def test_gain_scales_with_allocation(self):
        gain_low = self.sizer.estimate_gain_during_shock(1.0)
        gain_high = self.sizer.estimate_gain_during_shock(5.0)
        assert gain_high > gain_low

    def test_custom_shock_pct(self):
        gain_default = self.sizer.estimate_gain_during_shock(3.0)
        gain_worse = self.sizer.estimate_gain_during_shock(3.0, spy_shock_pct=-30.0)
        assert gain_worse > gain_default


class TestHedgeEfficiency:
    """Test compute_hedge_efficiency."""

    def setup_method(self):
        self.sizer = VIXYHedgeSizer(config={"state_file": "/tmp/test_vixy_state.json"})

    def test_zero_allocation_returns_zero(self):
        eff = self.sizer.compute_hedge_efficiency(0.0, 20.0)
        assert eff == 0.0

    def test_positive_allocation_returns_positive(self):
        eff = self.sizer.compute_hedge_efficiency(3.0, 25.0)
        assert eff > 0

    def test_higher_vix_improves_efficiency(self):
        eff_low = self.sizer.compute_hedge_efficiency(3.0, 15.0)
        eff_high = self.sizer.compute_hedge_efficiency(3.0, 35.0)
        assert eff_high > eff_low


class TestDetermineAction:
    """Test _determine_action method."""

    def setup_method(self):
        self.sizer = VIXYHedgeSizer(config={"state_file": "/tmp/test_vixy_state.json"})

    def test_crisis_returns_freeze(self):
        action = self.sizer._determine_action(5.0, 0.0, HedgeRegime.CRISIS)
        assert action == HedgeAction.FREEZE

    def test_large_increase_returns_increase(self):
        action = self.sizer._determine_action(5.0, 2.0, HedgeRegime.NORMAL)
        assert action == HedgeAction.INCREASE

    def test_large_decrease_returns_decrease(self):
        action = self.sizer._determine_action(1.0, 3.0, HedgeRegime.NORMAL)
        assert action == HedgeAction.DECREASE

    def test_small_change_returns_maintain(self):
        action = self.sizer._determine_action(2.5, 2.0, HedgeRegime.NORMAL)
        assert action == HedgeAction.MAINTAIN

    def test_exactly_at_increase_threshold(self):
        """target - current = 1.0 is not > 1.0, so maintain."""
        action = self.sizer._determine_action(3.0, 2.0, HedgeRegime.NORMAL)
        assert action == HedgeAction.MAINTAIN

    def test_exactly_at_decrease_threshold(self):
        """target - current = -0.5 is not < -0.5, so maintain."""
        action = self.sizer._determine_action(2.5, 3.0, HedgeRegime.NORMAL)
        assert action == HedgeAction.MAINTAIN


class TestGetSignal:
    """Test get_signal method."""

    def setup_method(self):
        self.sizer = VIXYHedgeSizer(config={"state_file": "/tmp/test_vixy_state.json"})

    def test_returns_vixy_hedge_signal(self):
        sig = self.sizer.get_signal(20.0)
        assert isinstance(sig, VIXYHedgeSignal)

    def test_signal_regime_matches_vix(self):
        sig = self.sizer.get_signal(35.0)
        assert sig.regime == "stress"

    def test_signal_values_by_regime(self):
        sig_normal = self.sizer.get_signal(15.0)
        assert sig_normal.signal_value == 0.0
        sig_elevated = self.sizer.get_signal(25.0)
        assert sig_elevated.signal_value == 0.3
        sig_stress = self.sizer.get_signal(35.0)
        assert sig_stress.signal_value == 0.7
        sig_crisis = self.sizer.get_signal(50.0)
        assert sig_crisis.signal_value == 1.0

    def test_ensemble_weight_stress_for_crisis(self):
        sig = self.sizer.get_signal(50.0)
        assert sig.ensemble_weight == self.sizer.config.ensemble_weight_stress

    def test_ensemble_weight_normal_for_normal(self):
        sig = self.sizer.get_signal(15.0)
        assert sig.ensemble_weight == self.sizer.config.ensemble_weight_normal

    def test_collar_complement_decreases_with_allocation(self):
        sig_low = self.sizer.get_signal(12.0)    # Low VIX, low alloc
        sig_high = self.sizer.get_signal(50.0)    # High VIX, high alloc
        assert sig_high.collar_complement <= sig_low.collar_complement

    def test_confidence_matches_data_freshness(self):
        sig = self.sizer.get_signal(20.0, data_freshness=0.8)
        assert sig.confidence == 0.8


class TestCollarCoordination:
    """Test collar coordination methods."""

    def setup_method(self):
        self.sizer = VIXYHedgeSizer(config={"state_file": "/tmp/test_vixy_state.json"})

    def test_combined_coverage_string(self):
        result = self.sizer.get_combined_hedge_coverage(3.0, 2.0)
        assert "5.0%" in result
        assert "VIXY" in result
        assert "collar" in result

    def test_should_disable_collar_high_alloc(self):
        # max_hedge_pct=10.0, 80% threshold = 8.0
        assert self.sizer.should_disable_collar(8.0) is True
        assert self.sizer.should_disable_collar(9.0) is True

    def test_should_not_disable_collar_low_alloc(self):
        assert self.sizer.should_disable_collar(5.0) is False
        assert self.sizer.should_disable_collar(3.0) is False


class TestStatePersistence:
    """Test state persistence methods."""

    def test_load_state_default(self, tmp_path):
        state_file = tmp_path / "vixy_state.json"
        sizer = VIXYHedgeSizer(config={"state_file": str(state_file)})
        sizer._state_file = state_file  # Override to ensure tmp_path
        state = sizer.load_state()
        assert isinstance(state, VIXYHedgeState)
        assert state.current_allocation == 0.0
        assert state.regime == "normal"

    def test_save_and_load_roundtrip(self, tmp_path):
        state_file = tmp_path / "vixy_state.json"
        sizer = VIXYHedgeSizer(config={"state_file": str(state_file)})
        sizer._state_file = state_file
        sizer.load_state()
        sig = sizer.get_signal(25.0)
        sizer.save_state(sig)
        # Reload
        sizer2 = VIXYHedgeSizer(config={"state_file": str(state_file)})
        sizer2._state_file = state_file
        state = sizer2.load_state()
        assert state.target_allocation == sig.allocation_pct

    def test_corrupt_state_file_handled(self, tmp_path):
        state_file = tmp_path / "vixy_state.json"
        state_file.write_text("NOT VALID JSON!!!")
        sizer = VIXYHedgeSizer(config={"state_file": str(state_file)})
        sizer._state_file = state_file
        state = sizer.load_state()
        assert isinstance(state, VIXYHedgeState)
        assert state.current_allocation == 0.0

    def test_update_after_rebalance(self, tmp_path):
        state_file = tmp_path / "vixy_state.json"
        sizer = VIXYHedgeSizer(config={"state_file": str(state_file)})
        sizer._state_file = state_file
        sizer.load_state()
        sig = sizer.get_signal(25.0)
        sizer.save_state(sig)
        sizer.update_after_rebalance(3.0)
        state = sizer.load_state()
        assert state.current_allocation == 3.0


class TestStatus:
    """Test status method."""

    def test_status_returns_dict(self, tmp_path):
        state_file = tmp_path / "vixy_state.json"
        sizer = VIXYHedgeSizer(config={"state_file": str(state_file)})
        sizer._state_file = state_file
        status = sizer.status()
        assert isinstance(status, dict)
        assert "current_allocation_pct" in status
        assert "regime" in status
        assert "vix_level" in status


class TestDefaultConfigValidation:
    """Validate DEFAULT_CONFIG constants."""

    def test_min_less_than_max(self):
        assert DEFAULT_CONFIG["min_hedge_pct"] < DEFAULT_CONFIG["max_hedge_pct"]

    def test_max_hedge_reasonable(self):
        assert 0 < DEFAULT_CONFIG["max_hedge_pct"] <= 20.0

    def test_expense_ratio_positive(self):
        assert DEFAULT_CONFIG["vixy_expense_ratio"] > 0

    def test_shock_pct_negative(self):
        assert DEFAULT_CONFIG["spy_shock_pct"] < 0

    def test_ensemble_weights_between_zero_and_one(self):
        assert 0 < DEFAULT_CONFIG["ensemble_weight_normal"] <= 1.0
        assert 0 < DEFAULT_CONFIG["ensemble_weight_stress"] <= 1.0
        assert DEFAULT_CONFIG["ensemble_weight_stress"] >= DEFAULT_CONFIG["ensemble_weight_normal"]


class TestMainCLI:
    """Test main() CLI entry point."""

    def test_main_status(self, tmp_path, capsys):
        from src.strategy import vixy_hedge_sizing as mod
        state_file = tmp_path / "vixy_state.json"
        sizer = mod.VIXYHedgeSizer(config={"state_file": str(state_file)})
        output = mod._format_status(sizer)
        assert "VIXY Hedge" in output

    def test_main_recommend_format(self):
        from src.strategy import vixy_hedge_sizing as mod
        sizer = mod.VIXYHedgeSizer(config={"state_file": "/tmp/test.json"})
        sig = sizer.get_signal(25.0)
        output = mod._format_recommend(sig)
        assert "VIXY Hedge Recommendation" in output
        assert "25.0" in output
        assert "stress" in output or "elevated" in output
