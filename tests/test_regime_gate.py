"""Tests for RegimeGate — regime-adaptive signal gating."""
import pytest
from src.signals.regime_gate import RegimeGate


class TestRegimeGate:
    def test_default_gate_rules_msm_off_in_high_vol(self):
        gate = RegimeGate()
        assert not gate.is_active("multi_speed_momentum", "HIGH_VOL")

    def test_default_gate_rules_msm_off_in_crisis(self):
        gate = RegimeGate()
        assert not gate.is_active("multi_speed_momentum", "CRISIS")

    def test_default_gate_rules_msm_on_in_normal(self):
        gate = RegimeGate()
        assert gate.is_active("multi_speed_momentum", "NORMAL")

    def test_default_gate_rules_intl_mom_off_in_crisis(self):
        gate = RegimeGate()
        assert not gate.is_active("international_momentum", "CRISIS")

    def test_default_gate_rules_intl_mom_on_in_normal(self):
        gate = RegimeGate()
        assert gate.is_active("international_momentum", "NORMAL")

    def test_default_gate_rules_intl_mom_on_in_high_vol(self):
        gate = RegimeGate()
        assert gate.is_active("international_momentum", "HIGH_VOL")

    def test_unknown_signal_is_always_active(self):
        gate = RegimeGate()
        assert gate.is_active("unknown_signal", "NORMAL")
        assert gate.is_active("unknown_signal", "CRISIS")
        assert gate.is_active("unknown_signal", "HIGH_VOL")

    def test_custom_gate_rules(self):
        custom = {"my_signal": {"NORMAL", "RECOVERY"}}
        gate = RegimeGate(gate_rules=custom)
        assert not gate.is_active("my_signal", "NORMAL")
        assert gate.is_active("my_signal", "CRISIS")

    def test_gate_returns_active_signals_only(self):
        gate = RegimeGate()
        active = gate.gate("CRISIS")
        # MSM is OFF in CRISIS, INTL_MOM is OFF in CRISIS
        assert "multi_speed_momentum" not in active
        assert "international_momentum" not in active
        # MSM is in gate_rules and ON in NORMAL
        active_normal = gate.gate("NORMAL")
        assert "multi_speed_momentum" in active_normal
        assert "international_momentum" in active_normal


class TestHysteresis:
    def test_hysteresis_uses_prev_regime_when_below_dwell(self):
        gate = RegimeGate(min_dwell_days=20)
        # Just switched from NORMAL to CRISIS, only 5 days in CRISIS
        active = gate.gate_with_hysteresis("CRISIS", "NORMAL", days_in_regime=5)
        # Should use NORMAL gating (MSM and INTL_MOM both ON)
        assert "multi_speed_momentum" in active
        assert "international_momentum" in active

    def test_hysteresis_uses_current_regime_when_above_dwell(self):
        gate = RegimeGate(min_dwell_days=20)
        # In CRISIS for 25 days
        active = gate.gate_with_hysteresis("CRISIS", "NORMAL", days_in_regime=25)
        # Should use CRISIS gating (MSM OFF, INTL_MOM OFF)
        assert "multi_speed_momentum" not in active
        assert "international_momentum" not in active

    def test_hysteresis_no_prev_regime_uses_current(self):
        gate = RegimeGate(min_dwell_days=20)
        active = gate.gate_with_hysteresis("CRISIS", None, days_in_regime=0)
        # No prev regime, so use current CRISIS gating
        assert "multi_speed_momentum" not in active

    def test_custom_min_dwell_days(self):
        gate = RegimeGate(min_dwell_days=5)
        active = gate.gate_with_hysteresis("CRISIS", "NORMAL", days_in_regime=6)
        # 6 days > 5 min dwell, use CRISIS gating
        assert "multi_speed_momentum" not in active


class TestFilterWeights:
    def test_zeros_out_gated_signals(self):
        gate = RegimeGate()
        weights = {
            "multi_speed_momentum": 0.10,
            "cross_asset_rv": 0.15,
            "alternative_data": 0.25,
            "international_momentum": 0.20,
        }
        filtered = gate.filter_weights(weights, "CRISIS")
        assert filtered["multi_speed_momentum"] == 0.0
        assert filtered["international_momentum"] == 0.0
        assert filtered["cross_asset_rv"] == 0.15
        assert filtered["alternative_data"] == 0.25

    def test_normal_regime_no_zeroing(self):
        gate = RegimeGate()
        weights = {
            "multi_speed_momentum": 0.10,
            "international_momentum": 0.20,
        }
        filtered = gate.filter_weights(weights, "NORMAL")
        assert filtered["multi_speed_momentum"] == 0.10
        assert filtered["international_momentum"] == 0.20

    def test_filter_with_enum_keys(self):
        """Filter should handle Enum keys (as in REGIME_WEIGHTS)."""
        from src.strategy.ensemble_voter import SignalSource, Regime
        gate = RegimeGate()
        weights = {
            SignalSource.MULTI_SPEED_MOM: 0.05,
            SignalSource.CROSS_ASSET_RV: 0.15,
        }
        filtered = gate.filter_weights(weights, "CRISIS")
        assert filtered[SignalSource.MULTI_SPEED_MOM] == 0.0
        assert filtered[SignalSource.CROSS_ASSET_RV] == 0.15


class TestGetActiveSignalNames:
    def test_includes_signals_not_in_gate_rules(self):
        gate = RegimeGate()
        all_signals = [
            "multi_speed_momentum",
            "cross_asset_rv",
            "alternative_data",
        ]
        active = gate.get_active_signal_names(all_signals, "CRISIS")
        # MSM OFF, but cross_asset_rv and alternative_data ON
        assert "multi_speed_momentum" not in active
        assert "cross_asset_rv" in active
        assert "alternative_data" in active

    def test_normal_all_active(self):
        gate = RegimeGate()
        all_signals = [
            "multi_speed_momentum",
            "international_momentum",
            "cross_asset_rv",
        ]
        active = gate.get_active_signal_names(all_signals, "NORMAL")
        assert len(active) == 3


class TestBehavioralSentimentGating:
    """Tests for behavioral_sentiment gate rule: OFF in NORMAL/HIGH_VOL/CRISIS."""

    def test_off_in_normal(self):
        gate = RegimeGate()
        assert not gate.is_active("behavioral_sentiment", "NORMAL")

    def test_off_in_high_vol(self):
        gate = RegimeGate()
        assert not gate.is_active("behavioral_sentiment", "HIGH_VOL")

    def test_off_in_crisis(self):
        gate = RegimeGate()
        assert not gate.is_active("behavioral_sentiment", "CRISIS")

    def test_on_in_low_vol(self):
        gate = RegimeGate()
        assert gate.is_active("behavioral_sentiment", "LOW_VOL")


class TestCrossAssetRegimeArbGating:
    """Tests for cross_asset_regime_arb gate rule: OFF in LOW_VOL."""

    def test_off_in_low_vol(self):
        gate = RegimeGate()
        assert not gate.is_active("cross_asset_regime_arb", "LOW_VOL")

    def test_on_in_normal(self):
        gate = RegimeGate()
        assert gate.is_active("cross_asset_regime_arb", "NORMAL")

    def test_on_in_high_vol(self):
        gate = RegimeGate()
        assert gate.is_active("cross_asset_regime_arb", "HIGH_VOL")

    def test_on_in_crisis(self):
        gate = RegimeGate()
        assert gate.is_active("cross_asset_regime_arb", "CRISIS")


class TestUpdateFromPerformance:
    """Tests for data-driven gate rule updates from rolling Sharpe."""

    def test_gates_off_signal_with_negative_sharpe(self):
        gate = RegimeGate()
        # Start with no rules for 'my_signal'
        assert gate.is_active("my_signal", "NORMAL")
        gate.update_from_performance({"NORMAL": {"my_signal": -0.5}})
        assert not gate.is_active("my_signal", "NORMAL")

    def test_re_enables_signal_with_positive_sharpe(self):
        custom = {"my_signal": {"NORMAL", "HIGH_VOL"}}
        gate = RegimeGate(gate_rules=custom)
        gate.update_from_performance({
            "NORMAL": {"my_signal": 0.3},
            "HIGH_VOL": {"my_signal": -0.1},
        })
        assert gate.is_active("my_signal", "NORMAL")
        assert not gate.is_active("my_signal", "HIGH_VOL")

    def test_removes_empty_signal_entry_when_all_reenabled(self):
        custom = {"my_signal": {"NORMAL"}}
        gate = RegimeGate(gate_rules=custom)
        gate.update_from_performance({"NORMAL": {"my_signal": 0.5}})
        assert "my_signal" not in gate.gate_rules

    def test_sharpe_at_threshold_is_on(self):
        gate = RegimeGate()
        gate.update_from_performance({"NORMAL": {"my_signal": 0.0}})
        assert gate.is_active("my_signal", "NORMAL")

    def test_sharpe_below_custom_threshold_is_off(self):
        gate = RegimeGate()
        gate.update_from_performance(
            {"NORMAL": {"my_signal": 0.3}},
            sharpe_threshold=0.5,
        )
        assert not gate.is_active("my_signal", "NORMAL")

    def test_normalizes_signal_names(self):
        gate = RegimeGate()
        gate.update_from_performance({"NORMAL": {"Multi Speed Momentum": -0.5}})
        # MSM already off in HIGH_VOL/CRISIS; this adds NORMAL
        assert not gate.is_active("multi_speed_momentum", "NORMAL")

    def test_does_not_affect_other_regimes(self):
        gate = RegimeGate()
        gate.update_from_performance({"CRISIS": {"my_signal": -1.0}})
        assert not gate.is_active("my_signal", "CRISIS")
        assert gate.is_active("my_signal", "NORMAL")


class TestGetGateSummary:
    """Tests for gate summary diagnostic output."""

    def test_returns_sorted_regimes(self):
        gate = RegimeGate()
        summary = gate.get_gate_summary()
        assert summary["multi_speed_momentum"] == ["CRISIS", "HIGH_VOL"]

    def test_includes_behavioral_sentiment(self):
        gate = RegimeGate()
        summary = gate.get_gate_summary()
        assert "behavioral_sentiment" in summary
        assert summary["behavioral_sentiment"] == ["CRISIS", "HIGH_VOL", "NORMAL"]

    def test_includes_cross_asset_regime_arb(self):
        gate = RegimeGate()
        summary = gate.get_gate_summary()
        assert "cross_asset_regime_arb" in summary
        assert summary["cross_asset_regime_arb"] == ["LOW_VOL"]

    def test_empty_after_all_reenabled(self):
        custom = {"my_signal": {"NORMAL"}}
        gate = RegimeGate(gate_rules=custom)
        gate.update_from_performance({"NORMAL": {"my_signal": 0.5}})
        summary = gate.get_gate_summary()
        assert "my_signal" not in summary


class TestRegimeGateClassifiers:
    """Edge-case classification tests for is_active, gate, and gate_with_hysteresis."""

    def test_is_active_empty_signal_name(self):
        gate = RegimeGate()
        assert gate.is_active("", "NORMAL")

    def test_is_active_empty_regime_name(self):
        gate = RegimeGate()
        assert gate.is_active("multi_speed_momentum", "")
        assert gate.is_active("international_momentum", "")

    def test_is_active_unknown_regime_is_active(self):
        """Unknown regime names are not in any OFF set, so signal is active."""
        gate = RegimeGate()
        assert gate.is_active("multi_speed_momentum", "RECOVERY")

    def test_is_active_case_sensitive_regime(self):
        """Regime matching is case-sensitive; 'crisis' != 'CRISIS'."""
        gate = RegimeGate()
        assert gate.is_active("multi_speed_momentum", "crisis")
        assert not gate.is_active("multi_speed_momentum", "CRISIS")

    def test_gate_unknown_regime_returns_all_ruled_signals(self):
        """Unknown regime: signals with rules are all ON (regime not in OFF sets)."""
        gate = RegimeGate()
        active = gate.gate("RECOVERY")
        assert "multi_speed_momentum" in active
        assert "international_momentum" in active
        # behavioral_sentiment is OFF only in NORMAL/HIGH_VOL/CRISIS, not RECOVERY
        assert "behavioral_sentiment" in active
        assert "cross_asset_regime_arb" in active

    def test_gate_empty_gate_rules(self):
        """gate_rules={} is treated as falsy, so falls back to class defaults."""
        gate = RegimeGate(gate_rules={})
        active = gate.gate("CRISIS")
        # Falls back to class defaults, not empty
        assert len(active) >= 1

    def test_gate_all_signals_off_in_regime(self):
        """When all ruled signals are OFF, gate returns empty list."""
        gate = RegimeGate()
        active = gate.gate("CRISIS")
        # MSM off, INTL_MOM off, behavioral_sentiment off (NORMAL/HIGH_VOL/CRISIS)
        # Only cross_asset_regime_arb is ON in CRISIS
        assert "cross_asset_regime_arb" in active
        assert len(active) == 1

    def test_gate_gives_no_implicit_signals(self):
        """gate() only returns signals that have explicit rules, not all known signals."""
        gate = RegimeGate()
        active = gate.gate("NORMAL")
        # cross_asset_rv and alternative_data are NOT in gate_rules
        assert "cross_asset_rv" not in active
        assert "alternative_data" not in active

    def test_gate_with_hysteresis_exactly_at_boundary(self):
        """days_in_regime == min_dwell_days means we've met the dwell requirement."""
        gate = RegimeGate(min_dwell_days=20)
        active = gate.gate_with_hysteresis("CRISIS", "NORMAL", days_in_regime=20)
        assert "multi_speed_momentum" not in active  # CRISIS gating

    def test_gate_with_hysteresis_same_regime_no_change(self):
        """When prev and current regime are the same, no hysteresis needed."""
        gate = RegimeGate()
        active = gate.gate_with_hysteresis("CRISIS", "CRISIS", days_in_regime=5)
        assert "multi_speed_momentum" not in active  # CRISIS gating regardless of dwell

    def test_gate_with_hysteresis_zero_days_uses_prev(self):
        """days_in_regime=0 with diff regime triggers hysteresis."""
        gate = RegimeGate(min_dwell_days=20)
        active = gate.gate_with_hysteresis("CRISIS", "NORMAL", days_in_regime=0)
        assert "multi_speed_momentum" in active  # Uses NORMAL gating

    def test_gate_with_hysteresis_first_observation(self):
        """No prev_regime means no hysteresis possible; use current."""
        gate = RegimeGate()
        active = gate.gate_with_hysteresis("HIGH_VOL", days_in_regime=0)
        assert "multi_speed_momentum" not in active
        assert "behavioral_sentiment" not in active

    def test_gate_with_hysteresis_low_vol_no_prev(self):
        """LOW_VOL regime: cross_asset_regime_arb should be OFF."""
        gate = RegimeGate()
        active = gate.gate_with_hysteresis("LOW_VOL", days_in_regime=999)
        assert "cross_asset_regime_arb" not in active


class TestRegimeGateEdgeCases:
    """Boundary conditions, empty inputs, and extreme parameter values."""

    def test_init_empty_gate_rules(self):
        """gate_rules={} is falsy; __init__ falls back to class defaults."""
        gate = RegimeGate(gate_rules={})
        # Falls back to class defaults because {} is falsy
        assert len(gate.gate_rules) > 0
        assert gate.min_dwell_days == 20

    def test_init_custom_min_dwell_zero(self):
        gate = RegimeGate(min_dwell_days=0)
        assert gate.min_dwell_days == 0
        # With zero dwell, even days_in_regime=0 uses current regime
        active = gate.gate_with_hysteresis("CRISIS", "NORMAL", days_in_regime=0)
        assert "multi_speed_momentum" not in active

    def test_init_negative_min_dwell(self):
        """Negative dwell means condition always passes; current regime always used."""
        gate = RegimeGate(min_dwell_days=-5)
        active = gate.gate_with_hysteresis("CRISIS", "NORMAL", days_in_regime=0)
        assert "multi_speed_momentum" not in active  # 0 < -5 is False

    def test_init_large_min_dwell(self):
        gate = RegimeGate(min_dwell_days=1000)
        active = gate.gate_with_hysteresis("CRISIS", "NORMAL", days_in_regime=999)
        assert "multi_speed_momentum" in active  # hysteresis: still using NORMAL

    def test_filter_weights_empty_dict(self):
        gate = RegimeGate()
        filtered = gate.filter_weights({}, "CRISIS")
        assert filtered == {}

    def test_filter_weights_no_off_signals(self):
        gate = RegimeGate()
        weights = {"my_signal": 0.5, "other_signal": 0.5}
        filtered = gate.filter_weights(weights, "CRISIS")
        assert filtered["my_signal"] == 0.5
        assert filtered["other_signal"] == 0.5

    def test_filter_weights_all_gated_off(self):
        """When all gated signals in a regime are OFF in CRISIS, remaining survive."""
        gate = RegimeGate()
        weights = {
            "multi_speed_momentum": 0.2,
            "international_momentum": 0.2,
            "cross_asset_rv": 0.2,
        }
        filtered = gate.filter_weights(weights, "CRISIS")
        assert filtered["multi_speed_momentum"] == 0.0
        assert filtered["international_momentum"] == 0.0
        assert filtered["cross_asset_rv"] == 0.2

    def test_filter_weights_signal_name_with_spaces(self):
        """Signal names can contain spaces and still be matched."""
        custom = {"my signal": {"CRISIS"}}
        gate = RegimeGate(gate_rules=custom)
        weights = {"my signal": 0.3, "other_signal": 0.7}
        filtered = gate.filter_weights(weights, "CRISIS")
        assert filtered["my signal"] == 0.0
        assert filtered["other_signal"] == 0.7

    def test_filter_weights_with_none_regime_name(self):
        """None regime name should be treated as string 'None' by default."""
        gate = RegimeGate()
        weights = {"multi_speed_momentum": 0.5}
        # None regime name does not match any OFF set, so signal stays active
        filtered = gate.filter_weights(weights, None)
        assert filtered["multi_speed_momentum"] == 0.5

    def test_get_active_signal_names_empty_list(self):
        gate = RegimeGate()
        active = gate.get_active_signal_names([], "CRISIS")
        assert active == []

    def test_get_active_signal_names_all_off(self):
        gate = RegimeGate()
        all_sigs = ["multi_speed_momentum", "international_momentum"]
        active = gate.get_active_signal_names(all_sigs, "CRISIS")
        assert active == []

    def test_get_active_signal_names_duplicates(self):
        gate = RegimeGate()
        all_sigs = ["multi_speed_momentum", "multi_speed_momentum"]
        active = gate.get_active_signal_names(all_sigs, "NORMAL")
        # Returns both entries (list, not set)
        assert active == ["multi_speed_momentum", "multi_speed_momentum"]

    def test_update_from_performance_empty_input(self):
        gate = RegimeGate()
        gate.update_from_performance({})
        # Gate rules unchanged
        assert "multi_speed_momentum" in gate.gate_rules

    def test_update_from_performance_negative_threshold(self):
        """Negative threshold means signals with mildly negative Sharpe pass."""
        gate = RegimeGate()
        gate.update_from_performance(
            {"NORMAL": {"my_signal": -0.3}},
            sharpe_threshold=-0.5,
        )
        assert gate.is_active("my_signal", "NORMAL")

    def test_update_from_performance_nonexistent_regime(self):
        """Data for a regime not in any gate rules still creates an entry."""
        gate = RegimeGate()
        gate.update_from_performance({"FUTURE_REGIME": {"my_signal": -1.0}})
        assert not gate.is_active("my_signal", "FUTURE_REGIME")

    def test_get_gate_summary_empty_rules(self):
        """gate_rules={} is falsy, falls back to class defaults, not empty."""
        gate = RegimeGate(gate_rules={})
        summary = gate.get_gate_summary()
        assert len(summary) > 0  # falls back to class defaults

    def test_default_min_dwell_constant(self):
        assert RegimeGate.DEFAULT_MIN_DWELL_DAYS == 20

    def test_default_gate_rules_structure(self):
        assert "multi_speed_momentum" in RegimeGate.GATE_RULES
        assert "international_momentum" in RegimeGate.GATE_RULES
        assert "behavioral_sentiment" in RegimeGate.GATE_RULES
        assert "cross_asset_regime_arb" in RegimeGate.GATE_RULES

    def test_gate_rules_deep_copy_independence(self):
        """Modifying a gate instance's rules must NOT affect class defaults."""
        gate = RegimeGate()
        gate.gate_rules["multi_speed_momentum"].add("RECOVERY")
        # Class default should NOT have RECOVERY
        assert "RECOVERY" not in RegimeGate.GATE_RULES["multi_speed_momentum"]

    def test_gate_rules_deep_copy_independence_new_entry(self):
        """Adding a new signal to instance must NOT affect class defaults."""
        gate = RegimeGate()
        gate.gate_rules["custom_signal"] = set()
        assert "custom_signal" not in RegimeGate.GATE_RULES

    def test_repr_no_exceptions(self):
        """Verify get_gate_summary output format."""
        gate = RegimeGate()
        summary = gate.get_gate_summary()
        assert isinstance(summary, dict)
        for sig, regimes in summary.items():
            assert isinstance(sig, str)
            assert isinstance(regimes, list)
            for r in regimes:
                assert isinstance(r, str)

    def test_multiple_gate_instances_independent(self):
        """Two gate instances should not share state."""
        g1 = RegimeGate()
        g2 = RegimeGate()
        g1.gate_rules.clear()
        assert len(g2.gate_rules) > 0


class TestRegimeGateGatingRules:
    """Comprehensive matrix test: every signal x every regime = correct ON/OFF."""

    # Full truth table derived from GATE_RULES
    # signal                     | NORMAL | LOW_VOL | HIGH_VOL | CRISIS | RECOVERY
    # multi_speed_momentum       |  ON    |   ON    |   OFF    |  OFF   |   ON
    # international_momentum     |  ON    |   ON    |   ON     |  OFF   |   ON
    # behavioral_sentiment       |  OFF   |   ON    |   OFF    |  OFF   |   ON(implied)
    # cross_asset_regime_arb     |  ON    |   OFF   |   ON     |  ON    |   ON
    # cross_asset_rv (no rules)  |  ON    |   ON    |   ON     |  ON    |   ON

    def test_msm_on_in_low_vol(self):
        gate = RegimeGate()
        assert gate.is_active("multi_speed_momentum", "LOW_VOL")

    def test_msm_off_in_crisis(self):
        gate = RegimeGate()
        assert not gate.is_active("multi_speed_momentum", "CRISIS")

    def test_msm_on_in_recovery(self):
        gate = RegimeGate()
        assert gate.is_active("multi_speed_momentum", "RECOVERY")

    def test_intl_mom_on_in_low_vol(self):
        gate = RegimeGate()
        assert gate.is_active("international_momentum", "LOW_VOL")

    def test_intl_mom_on_in_high_vol(self):
        gate = RegimeGate()
        assert gate.is_active("international_momentum", "HIGH_VOL")

    def test_behavioral_sentiment_on_in_low_vol(self):
        gate = RegimeGate()
        assert gate.is_active("behavioral_sentiment", "LOW_VOL")

    def test_behavioral_sentiment_off_in_high_vol(self):
        gate = RegimeGate()
        assert not gate.is_active("behavioral_sentiment", "HIGH_VOL")

    def test_cross_regime_arb_on_in_normal(self):
        gate = RegimeGate()
        assert gate.is_active("cross_asset_regime_arb", "NORMAL")

    def test_cross_regime_arb_on_in_crisis(self):
        gate = RegimeGate()
        assert gate.is_active("cross_asset_regime_arb", "CRISIS")

    def test_cross_regime_arb_off_in_low_vol(self):
        gate = RegimeGate()
        assert not gate.is_active("cross_asset_regime_arb", "LOW_VOL")

    def test_unruled_signals_always_active(self):
        """Signals without rules (cross_asset_rv, alternative_data) are ON everywhere."""
        gate = RegimeGate()
        for regime in ("NORMAL", "LOW_VOL", "HIGH_VOL", "CRISIS", "RECOVERY"):
            assert gate.is_active("cross_asset_rv", regime)
            assert gate.is_active("alternative_data", regime)

    def test_gate_returns_low_vol_correctly(self):
        gate = RegimeGate()
        active = gate.gate("LOW_VOL")
        # MSM: ON, INTL_MOM: ON, BEHAVIORAL: ON, CROSS_ARB: OFF
        assert "multi_speed_momentum" in active
        assert "international_momentum" in active
        assert "behavioral_sentiment" in active
        assert "cross_asset_regime_arb" not in active

    def test_gate_returns_high_vol_correctly(self):
        gate = RegimeGate()
        active = gate.gate("HIGH_VOL")
        assert "multi_speed_momentum" not in active
        assert "international_momentum" in active
        assert "behavioral_sentiment" not in active
        assert "cross_asset_regime_arb" in active

    def test_gate_returns_normal_correctly(self):
        gate = RegimeGate()
        active = gate.gate("NORMAL")
        assert "multi_speed_momentum" in active
        assert "international_momentum" in active
        assert "behavioral_sentiment" not in active
        assert "cross_asset_regime_arb" in active

    def test_gate_every_regime_happy_path(self):
        """Verify gate() returns non-empty lists for all standard regimes."""
        gate = RegimeGate()
        for regime in ("NORMAL", "LOW_VOL", "HIGH_VOL", "CRISIS"):
            result = gate.gate(regime)
            assert isinstance(result, list)
            # Every regime should have at least one active signal
            assert len(result) >= 1

    def test_get_active_signal_names_unruled_included(self):
        """get_active_signal_names includes signals NOT in gate_rules."""
        gate = RegimeGate()
        sigs = ["cross_asset_rv", "multi_speed_momentum"]
        active = gate.get_active_signal_names(sigs, "CRISIS")
        assert "cross_asset_rv" in active  # no rules = always active
        assert "multi_speed_momentum" not in active  # OFF in CRISIS

    def test_update_from_performance_can_gate_unruled_signal(self):
        """update_from_performance can add rules for signals not in GATE_RULES."""
        gate = RegimeGate()
        assert gate.is_active("cross_asset_rv", "CRISIS")
        gate.update_from_performance({"CRISIS": {"cross_asset_rv": -0.5}})
        assert not gate.is_active("cross_asset_rv", "CRISIS")
        assert gate.is_active("cross_asset_rv", "NORMAL")


class TestRegimeGateSerialization:
    """Test dict-based representation, reconstruction, and state copying."""

    def test_get_gate_summary_structure(self):
        gate = RegimeGate()
        summary = gate.get_gate_summary()
        assert "multi_speed_momentum" in summary
        assert isinstance(summary["multi_speed_momentum"], list)
        assert all(isinstance(r, str) for r in summary["multi_speed_momentum"])

    def test_get_gate_summary_sorted(self):
        gate = RegimeGate()
        summary = gate.get_gate_summary()
        for regimes in summary.values():
            assert regimes == sorted(regimes)

    def test_reconstruct_from_summary(self):
        """Recreate a RegimeGate from get_gate_summary output (round-trip)."""
        gate1 = RegimeGate()
        summary = gate1.get_gate_summary()
        # Convert list values back to sets for gate_rules
        reconstructed_rules = {k: set(v) for k, v in summary.items()}
        gate2 = RegimeGate(gate_rules=reconstructed_rules)
        # Verify same behavior
        assert gate1.gate("CRISIS") == gate2.gate("CRISIS")
        assert gate1.gate("NORMAL") == gate2.gate("NORMAL")
        assert gate1.gate("LOW_VOL") == gate2.gate("LOW_VOL")
        assert gate1.gate("HIGH_VOL") == gate2.gate("HIGH_VOL")

    def test_reconstruct_empty_rules(self):
        """Round-trip with empty gate_rules (falls back to class defaults)."""
        gate = RegimeGate(gate_rules={})
        summary = gate.get_gate_summary()
        assert len(summary) > 0  # falls back to class defaults
        gate2 = RegimeGate(gate_rules={k: set(v) for k, v in summary.items()})
        assert len(gate2.gate("CRISIS")) > 0

    def test_get_gate_summary_keys_are_strings(self):
        gate = RegimeGate()
        summary = gate.get_gate_summary()
        for key in summary:
            assert isinstance(key, str)

    def test_update_then_summary_reflects_change(self):
        gate = RegimeGate()
        gate.update_from_performance({"NORMAL": {"my_signal": -0.3}})
        summary = gate.get_gate_summary()
        assert "my_signal" in summary
        assert "NORMAL" in summary["my_signal"]

    def test_clear_all_rules_then_summary_empty(self):
        gate = RegimeGate()
        gate.gate_rules.clear()
        summary = gate.get_gate_summary()
        assert summary == {}

    def test_gate_rules_are_dict_with_set_values(self):
        gate = RegimeGate()
        for sig, regimes in gate.gate_rules.items():
            assert isinstance(regimes, set)
            assert all(isinstance(r, str) for r in regimes)

    def test_custom_rules_are_not_mutated_by_get_gate_summary(self):
        """get_gate_summary should not modify internal state."""
        custom = {"sig_a": {"HIGH_VOL", "CRISIS"}}
        gate = RegimeGate(gate_rules=custom)
        _ = gate.get_gate_summary()
        assert gate.gate_rules["sig_a"] == {"HIGH_VOL", "CRISIS"}


class TestRegimeGateIntegration:
    """End-to-end pipeline tests combining multiple RegimeGate methods."""

    def test_update_then_gate_pipeline(self):
        """update_from_performance → gate: gating reflects new rules.

        gate() returns only signals that are in gate_rules AND active.
        After update, cross_asset_rv is added to gate_rules but it's OFF
        in CRISIS (negative Sharpe), so gate("CRISIS") does NOT include it.
        In NORMAL (no rule), it IS included.
        """
        gate = RegimeGate()
        gate.update_from_performance({"CRISIS": {"cross_asset_rv": -0.5}})
        active_crisis = gate.gate("CRISIS")
        assert "cross_asset_rv" not in active_crisis  # OFF in CRISIS
        active_normal = gate.gate("NORMAL")
        assert "cross_asset_rv" in active_normal  # ON in NORMAL

    def test_filter_weights_after_update(self):
        """update_from_performance → filter_weights: new gating reflected."""
        gate = RegimeGate()
        gate.update_from_performance({"CRISIS": {"cross_asset_rv": -0.5}})
        weights = {"cross_asset_rv": 0.3, "alternative_data": 0.7}
        filtered = gate.filter_weights(weights, "CRISIS")
        assert filtered["cross_asset_rv"] == 0.0
        assert filtered["alternative_data"] == 0.7

    def test_full_pipeline_all_regimes(self):
        """Run full pipeline for every regime and verify consistency."""
        gate = RegimeGate()
        all_signals = [
            "multi_speed_momentum",
            "international_momentum",
            "behavioral_sentiment",
            "cross_asset_regime_arb",
            "cross_asset_rv",
            "alternative_data",
        ]
        regimes = ["NORMAL", "LOW_VOL", "HIGH_VOL", "CRISIS"]
        for regime in regimes:
            active = gate.get_active_signal_names(all_signals, regime)
            weights = {s: 1.0 / len(all_signals) for s in all_signals}
            filtered = gate.filter_weights(weights, regime)
            # Signals in active should have non-zero weight
            for sig in active:
                assert filtered[sig] > 0.0
            # Signals not in active should have zero weight
            for sig in all_signals:
                if sig not in active:
                    assert filtered[sig] == 0.0

    def test_filter_weights_with_enum_sources(self):
        """Integration: filter_weights with SignalSource enum keys in all regimes."""
        from src.strategy.ensemble_voter import SignalSource, Regime
        gate = RegimeGate()
        weights = {
            SignalSource.MULTI_SPEED_MOM: 0.10,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.15,
            SignalSource.CROSS_ASSET_REGIME_ARB: 0.05,
            SignalSource.UNIFIED_OVERLAY: 0.20,
        }
        # In CRISIS: MSM OFF, INTL_MOM OFF, REGIME_ARB ON, UNIFIED always ON
        filtered_crisis = gate.filter_weights(weights, "CRISIS")
        assert filtered_crisis[SignalSource.MULTI_SPEED_MOM] == 0.0
        assert filtered_crisis[SignalSource.INTERNATIONAL_MOMENTUM] == 0.0
        assert filtered_crisis[SignalSource.CROSS_ASSET_REGIME_ARB] == 0.05
        assert filtered_crisis[SignalSource.UNIFIED_OVERLAY] == 0.20
        # In LOW_VOL: MSM ON, INTL_MOM ON, REGIME_ARB OFF
        # Use uppercase "LOW_VOL" to match gate rule keys (not Regime.LOW_VOL.value which is lowercase)
        filtered_low_vol = gate.filter_weights(weights, "LOW_VOL")
        assert filtered_low_vol[SignalSource.MULTI_SPEED_MOM] == 0.10
        assert filtered_low_vol[SignalSource.CROSS_ASSET_REGIME_ARB] == 0.0

    def test_gate_with_hysteresis_chain(self):
        """Simulate a sequence of regime transitions with hysteresis."""
        gate = RegimeGate(min_dwell_days=10)
        # Day 1: NORMAL -> CRISIS transition
        active_1 = gate.gate_with_hysteresis("CRISIS", "NORMAL", days_in_regime=1)
        assert "multi_speed_momentum" in active_1  # hysteresis: NORMAL rules
        # Day 10: Still in CRISIS, met dwell
        active_2 = gate.gate_with_hysteresis("CRISIS", "NORMAL", days_in_regime=10)
        assert "multi_speed_momentum" not in active_2  # CRISIS rules
        # Day 11: CRISIS -> NORMAL back
        active_3 = gate.gate_with_hysteresis("NORMAL", "CRISIS", days_in_regime=1)
        assert "multi_speed_momentum" not in active_3  # hysteresis: CRISIS rules
        # Day 21: back in NORMAL long enough
        active_4 = gate.gate_with_hysteresis("NORMAL", "CRISIS", days_in_regime=11)
        assert "multi_speed_momentum" in active_4  # NORMAL rules

    def test_update_re_enable_then_gate(self):
        """Update with negative Sharpe gating, then positive Sharpe re-enabling."""
        gate = RegimeGate()
        gate.update_from_performance({"NORMAL": {"test_sig": -0.5}})
        assert not gate.is_active("test_sig", "NORMAL")
        gate.update_from_performance({"NORMAL": {"test_sig": 0.5}})
        assert gate.is_active("test_sig", "NORMAL")

    def test_update_multiple_signals_same_regime(self):
        """Multiple signals updated in same regime simultaneously."""
        gate = RegimeGate()
        gate.update_from_performance({
            "HIGH_VOL": {"sig_a": -0.5, "sig_b": -0.3, "sig_c": 0.2},
        })
        assert not gate.is_active("sig_a", "HIGH_VOL")
        assert not gate.is_active("sig_b", "HIGH_VOL")
        assert gate.is_active("sig_c", "HIGH_VOL")

    def test_update_multiple_regimes_same_signal(self):
        """One signal updated across multiple regimes."""
        gate = RegimeGate()
        gate.update_from_performance({
            "NORMAL": {"test_sig": -0.5},
            "HIGH_VOL": {"test_sig": -0.3},
            "CRISIS": {"test_sig": 0.1},
        })
        assert not gate.is_active("test_sig", "NORMAL")
        assert not gate.is_active("test_sig", "HIGH_VOL")
        assert gate.is_active("test_sig", "CRISIS")

    def test_duplicate_is_active_calls_stable(self):
        """Calling is_active repeatedly should not mutate state."""
        gate = RegimeGate()
        for _ in range(100):
            gate.is_active("multi_speed_momentum", "HIGH_VOL")
        assert not gate.is_active("multi_speed_momentum", "HIGH_VOL")

    def test_get_active_signal_names_after_update(self):
        """get_active_signal_names reflects dynamic updates."""
        gate = RegimeGate()
        sigs = ["cross_asset_rv", "multi_speed_momentum"]
        assert "cross_asset_rv" in gate.get_active_signal_names(sigs, "CRISIS")
        gate.update_from_performance({"CRISIS": {"cross_asset_rv": -1.0}})
        assert "cross_asset_rv" not in gate.get_active_signal_names(sigs, "CRISIS")
