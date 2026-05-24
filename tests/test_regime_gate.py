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
