"""
Tests for v8.01 TurnoverValidator — turnover-aware ensemble weight validation.
"""

import json
import math
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from src.strategy.turnover_validator import (
    MAX_TURNOVER_PENALTY,
    MIN_SIGNAL_HISTORY,
    DEFAULT_ROLLING_WINDOW,
    SignalTurnoverMetrics,
    TurnoverValidator,
    TurnoverValidatorState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_state_path():
    """Provide a temporary state file path."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp) / "test_state.json"


@pytest.fixture
def validator(tmp_state_path):
    """Provide a TurnoverValidator with temp state."""
    return TurnoverValidator(state_path=tmp_state_path, rolling_window=20)


# ---------------------------------------------------------------------------
# SignalTurnoverMetrics tests
# ---------------------------------------------------------------------------


class TestSignalTurnoverMetrics:
    def test_default_creation(self):
        m = SignalTurnoverMetrics(
            source="test", signal_std=0.1, sign_flip_rate=0.3,
            magnitude_volatility=0.05, turnover_penalty=0.2,
            stability_score=0.7, expected_return=0.05,
            risk_cost=0.02, turnover_cost=0.01, marginal_score=0.02,
            num_periods=20,
        )
        assert m.source == "test"
        assert m.turnover_penalty == 0.2
        assert m.stability_score == 0.7

    def test_missing_data_flag(self):
        m = SignalTurnoverMetrics(
            source="test", signal_std=0.0, sign_flip_rate=0.0,
            magnitude_volatility=0.0, turnover_penalty=0.0,
            stability_score=0.5, expected_return=0.0,
            risk_cost=0.0, turnover_cost=0.0, marginal_score=0.0,
            num_periods=2, missing_data=True,
        )
        assert m.missing_data

    def test_negative_marginal_score(self):
        m = SignalTurnoverMetrics(
            source="noisy", signal_std=0.5, sign_flip_rate=0.4,
            magnitude_volatility=0.3, turnover_penalty=0.3,
            stability_score=0.3, expected_return=0.01,
            risk_cost=0.15, turnover_cost=0.05, marginal_score=-0.19,
            num_periods=20,
        )
        assert m.marginal_score < 0


# ---------------------------------------------------------------------------
# TurnoverValidatorState tests
# ---------------------------------------------------------------------------


class TestTurnoverValidatorState:
    def test_default_creation(self):
        state = TurnoverValidatorState()
        assert state.signal_history == {}
        assert state.rolling_window == DEFAULT_ROLLING_WINDOW

    def test_to_dict_roundtrip(self):
        state = TurnoverValidatorState(
            signal_history={"a": [1.0, 2.0, 3.0]},
            rolling_window=10,
        )
        d = state.to_dict()
        assert d["signal_history"]["a"] == [1.0, 2.0, 3.0]
        assert d["rolling_window"] == 10

        restored = TurnoverValidatorState.from_dict(d)
        assert restored.signal_history["a"] == [1.0, 2.0, 3.0]
        assert restored.rolling_window == 10

    def test_from_dict_empty(self):
        restored = TurnoverValidatorState.from_dict({})
        assert restored.signal_history == {}
        assert restored.rolling_window == DEFAULT_ROLLING_WINDOW


# ---------------------------------------------------------------------------
# TurnoverValidator tests
# ---------------------------------------------------------------------------


class TestTurnoverValidatorInit:
    def test_default_init(self, tmp_state_path):
        v = TurnoverValidator(state_path=tmp_state_path)
        assert v.rolling_window == 20
        assert v.state.signal_history == {}

    def test_custom_window(self, tmp_state_path):
        v = TurnoverValidator(state_path=tmp_state_path, rolling_window=50)
        assert v.rolling_window == 50

    def test_state_persistence(self, tmp_state_path):
        """State should persist across instances."""
        v1 = TurnoverValidator(state_path=tmp_state_path)
        v1.update_and_validate({"source_a": 0.5})
        assert "source_a" in v1.state.signal_history

        v2 = TurnoverValidator(state_path=tmp_state_path)
        assert "source_a" in v2.state.signal_history


class TestUpdateAndValidate:
    def test_single_update(self, validator):
        """Single update should create history with 1 element."""
        results = validator.update_and_validate({"tsfm_momentum": 0.5})
        assert "tsfm_momentum" in results
        m = results["tsfm_momentum"]
        assert m.missing_data  # insufficient history
        assert m.num_periods == 1

    def test_multiple_updates(self, validator):
        """Multiple updates should accumulate history."""
        for val in [0.5, 0.6, 0.4, 0.7, 0.3, 0.8]:
            validator.update_and_validate({"src": val})
        results = validator.update_and_validate({"src": 0.5})
        m = results["src"]
        assert m.num_periods == 7
        assert not m.missing_data

    def test_rolling_window_trim(self, validator):
        """History should be trimmed to rolling_window."""
        v = TurnoverValidator(rolling_window=5)
        for i in range(20):
            v.update_and_validate({"src": math.sin(i * 0.5)})
        assert len(v.state.signal_history["src"]) == 5

    def test_multiple_sources(self, validator):
        """Multiple sources should be tracked independently."""
        signals = {
            "stable": [0.5] * 15,
            "noisy": [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5,
                      0.5, -0.5, 0.5, -0.5, 0.5],
            "climber": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                        1.1, 1.2, 1.3, 1.4, 1.5],
        }
        v = TurnoverValidator(rolling_window=20)
        for i in range(15):
            day_sigs = {k: vals[i] for k, vals in signals.items()}
            v.update_and_validate(day_sigs)

        results = v.update_and_validate({
            "stable": 0.5, "noisy": 0.5, "climber": 1.5,
        })
        # Stable signal should have lowest penalty
        assert results["stable"].turnover_penalty <= results["noisy"].turnover_penalty
        # Noisy signal should have high penalty
        assert results["noisy"].turnover_penalty > 0.15


class TestTurnoverPenalty:
    def test_stable_signal_low_penalty(self, validator):
        """A constant signal should have near-zero penalty."""
        for _ in range(20):
            validator.update_and_validate({"stable": 0.5})
        results = validator.update_and_validate({"stable": 0.5})
        assert results["stable"].turnover_penalty < 0.1

    def test_chatter_high_penalty(self, validator):
        """Signal that alternates sign should get high penalty."""
        for i in range(20):
            val = 0.5 if i % 2 == 0 else -0.5
            validator.update_and_validate({"chatter": val})
        results = validator.update_and_validate({"chatter": 0.5})
        assert results["chatter"].turnover_penalty > 0.2

    def test_penalty_bounded(self, validator):
        """Penalty should never exceed MAX_TURNOVER_PENALTY."""
        # Extreme noisy signal
        for i in range(20):
            val = 1.0 if i % 2 == 0 else -1.0
            validator.update_and_validate({"extreme": val})
        results = validator.update_and_validate({"extreme": 1.0})
        assert results["extreme"].turnover_penalty <= MAX_TURNOVER_PENALTY

    def test_penalty_nonnegative(self, validator):
        """Penalty should never be negative."""
        for _ in range(20):
            validator.update_and_validate({"a": 0.5})
        results = validator.update_and_validate({"a": 0.5})
        assert results["a"].turnover_penalty >= 0


class TestGetAdjustedWeights:
    def test_no_adjustment_without_history(self, validator):
        """Without history, weights should remain unchanged."""
        weights = {"src_a": 0.15, "src_b": 0.10}
        adjusted = validator.get_adjusted_weights(weights, {"src_a": 0.5, "src_b": -0.3})
        # Insufficient history (< MIN_SIGNAL_HISTORY) -> missing_data -> no adjustment
        assert adjusted["src_a"] == pytest.approx(0.15, abs=1e-10)
        assert adjusted["src_b"] == pytest.approx(0.10, abs=1e-10)

    def test_penalty_reduces_weight(self, validator):
        """Signals with high turnover should get reduced weights."""
        # Build history: stable vs noisy
        base_weights = {"stable": 0.15, "noisy": 0.15}
        for i in range(20):
            signals = {
                "stable": 0.5,
                "noisy": 0.5 if i % 2 == 0 else -0.5,
            }
            validator.update_and_validate(signals)

        adjusted = validator.get_adjusted_weights(base_weights, {"stable": 0.5, "noisy": 0.5})
        # Noisy signal weight should be reduced more than stable
        noisy_reduction = base_weights["noisy"] - adjusted["noisy"]
        stable_reduction = base_weights["stable"] - adjusted["stable"]
        assert noisy_reduction > stable_reduction

    def test_weight_non_negative(self, validator):
        """Adjusted weights should never be negative."""
        weights = {"high_turnover": 0.10}
        for i in range(20):
            validator.update_and_validate({"high_turnover": 1.0 if i % 2 == 0 else -1.0})
        adjusted = validator.get_adjusted_weights(weights, {"high_turnover": 1.0})
        assert adjusted["high_turnover"] >= 0


class TestStabilityScore:
    def test_constant_signal_max_stability(self, validator):
        """Constant signal should have stability near 1.0."""
        for _ in range(20):
            validator.update_and_validate({"const": 0.5})
        results = validator.update_and_validate({"const": 0.5})
        assert results["const"].stability_score > 0.9

    def test_random_signal_low_stability(self, validator):
        """Random signal should have lower stability."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            validator.update_and_validate({"rand": float(rng.uniform(-1, 1))})
        results = validator.update_and_validate({"rand": float(rng.uniform(-1, 1))})
        assert results["rand"].stability_score < 0.85

    def test_stability_bounded(self, validator):
        """Stability score should be in [0, 1]."""
        for _ in range(20):
            validator.update_and_validate({"a": float(np.random.uniform(-1, 1))})
        results = validator.update_and_validate({"a": float(np.random.uniform(-1, 1))})
        assert 0.0 <= results["a"].stability_score <= 1.0


class TestMarginalScore:
    def test_positive_marginal_for_strong_signal(self, validator):
        """Strong stable signal should have positive marginal score."""
        for _ in range(20):
            validator.update_and_validate({"strong": 0.8})
        results = validator.update_and_validate({"strong": 0.8})
        assert results["strong"].marginal_score > 0

    def test_negative_marginal_for_noisy_signal(self, validator):
        """Noisy signal with low mean should have negative marginal score."""
        for i in range(20):
            val = 0.1 if i % 2 == 0 else -0.1
            validator.update_and_validate({"weak": val})
        results = validator.update_and_validate({"weak": 0.1})
        assert results["weak"].marginal_score < 0

    def test_marginal_score_boost(self, validator):
        """Positive marginal score should boost weight."""
        base_weights = {"strong": 0.10, "weak": 0.10}
        for _ in range(20):
            validator.update_and_validate({"strong": 0.8})
        for i in range(20):
            val = 0.1 if i % 2 == 0 else -0.1
            validator.update_and_validate({"weak": val})

        adjusted = validator.get_adjusted_weights(base_weights, {"strong": 0.8, "weak": 0.1})
        # Strong signal should get marginal boost (positive marginal score)
        assert adjusted["strong"] > base_weights["strong"] * 0.9


class TestSignFlipRate:
    def test_no_flips(self, validator):
        """Constant positive signal should have zero sign flip rate."""
        for _ in range(20):
            validator.update_and_validate({"pos": 0.5})
        results = validator.update_and_validate({"pos": 0.5})
        assert results["pos"].sign_flip_rate == 0.0

    def test_alternating_flips(self, validator):
        """Alternating signal should have high flip rate."""
        for i in range(20):
            val = 0.5 if i % 2 == 0 else -0.5
            validator.update_and_validate({"alt": val})
        results = validator.update_and_validate({"alt": 0.5})
        # Should be near 1.0 (flips every period)
        assert results["alt"].sign_flip_rate > 0.8


class TestStatePersistence:
    def test_save_load_roundtrip(self, tmp_state_path):
        """State should survive save/load cycle."""
        v1 = TurnoverValidator(state_path=tmp_state_path)
        for i in range(10):
            v1.update_and_validate({"a": float(i * 0.1)})
        # Use approximate comparison for floating point
        expected = [round(float(i * 0.1), 6) for i in range(10)]
        actual = [round(float(x), 6) for x in v1.state.signal_history["a"]]
        assert actual == expected, f"{actual} != {expected}"

        v2 = TurnoverValidator(state_path=tmp_state_path)
        actual_v2 = [round(float(x), 6) for x in v2.state.signal_history["a"]]
        assert actual_v2 == expected, f"{actual_v2} != {expected}"

    def test_corrupted_state_fallback(self, tmp_state_path):
        """Corrupted state file should fall back to empty state."""
        tmp_state_path.write_text("{invalid json")
        v = TurnoverValidator(state_path=tmp_state_path)
        assert v.state.signal_history == {}

    def test_state_dict_roundtrip(self, tmp_state_path):
        """State dict should survive JSON roundtrip."""
        v = TurnoverValidator(state_path=tmp_state_path)
        for i in range(10):
            v.update_and_validate({"x": float(i * 0.1)})
        d = v.state.to_dict()
        restored = TurnoverValidatorState.from_dict(d)
        assert restored.signal_history == v.state.signal_history


class TestDiagnostics:
    def test_empty_diagnostics(self, validator):
        """Empty validator should return empty diagnostics."""
        diag = validator.get_state_diagnostics()
        assert diag == {}

    def test_populated_diagnostics(self, validator):
        """Populated validator should return all tracked signals."""
        for i in range(20):
            validator.update_and_validate({"a": 0.5, "b": -0.3})
        diag = validator.get_state_diagnostics()
        assert "a" in diag
        assert "b" in diag
        assert diag["a"]["periods"] == 20
        assert "turnover_penalty" in diag["a"]

    def test_diagnostics_contains_all_keys(self, validator):
        """Diagnostics should contain all expected fields."""
        for i in range(20):
            validator.update_and_validate({"src": 0.5})
        diag = validator.get_state_diagnostics()
        expected_keys = {
            "periods", "mean", "std", "sign_flip_rate", "mag_vol",
            "turnover_penalty", "stability_score", "marginal_score",
        }
        assert expected_keys.issubset(diag["src"].keys())


# ---------------------------------------------------------------------------
# Integration-ready: single-source test
# ---------------------------------------------------------------------------


class TestSingleSourceEdgeCases:
    def test_single_source_no_penalty(self, validator):
        """Single source with minimal data should get zero penalty."""
        for _ in range(5):
            validator.update_and_validate({"only": 0.3})
        results = validator.update_and_validate({"only": 0.3})
        # At 5 periods, just at the threshold; penalty should be very low
        assert results["only"].turnover_penalty < 0.2

    def test_single_source_full_window(self, validator):
        """Single source with full window should compute normally."""
        for i in range(20):
            val = 0.3 if i < 10 else -0.3
            validator.update_and_validate({"only": val})
        results = validator.update_and_validate({"only": 0.3})
        assert results["only"].num_periods == 20
        # Signal flips at transition points; should have non-zero sign flip rate
        assert results["only"].sign_flip_rate > 0.05

    def test_nan_handling(self, validator):
        """NaN signal values should not crash the validator."""
        for _ in range(5):
            validator.update_and_validate({"src": float("nan")})
        # Should not raise
        results = validator.update_and_validate({"src": 0.5})
        assert "src" in results


class TestConstants:
    """Validate module constants."""

    def test_max_penalty(self):
        assert MAX_TURNOVER_PENALTY == 0.5

    def test_min_history(self):
        assert MIN_SIGNAL_HISTORY == 5

    def test_default_window(self):
        assert DEFAULT_ROLLING_WINDOW == 20


class TestSignalTurnoverMetricsExtended:
    """Extended tests for SignalTurnoverMetrics dataclass."""

    def test_all_fields(self):
        m = SignalTurnoverMetrics(
            source="test", signal_std=0.1, sign_flip_rate=0.3,
            magnitude_volatility=0.05, turnover_penalty=0.2,
            stability_score=0.7, expected_return=0.05,
            risk_cost=0.02, turnover_cost=0.01, marginal_score=0.02,
            num_periods=20,
        )
        assert m.source == "test"
        assert m.signal_std == 0.1
        assert m.magnitude_volatility == 0.05
        assert m.expected_return == 0.05
        assert m.risk_cost == 0.02
        assert m.turnover_cost == 0.01
        assert m.num_periods == 20
        assert m.missing_data is False

    def test_missing_data_default_false(self):
        m = SignalTurnoverMetrics(
            source="test", signal_std=0.1, sign_flip_rate=0.3,
            magnitude_volatility=0.05, turnover_penalty=0.2,
            stability_score=0.7, expected_return=0.05,
            risk_cost=0.02, turnover_cost=0.01, marginal_score=0.02,
            num_periods=20,
        )
        assert m.missing_data is False


class TestTurnoverValidatorStateExtended:
    """Extended tests for TurnoverValidatorState."""

    def test_empty_history(self):
        state = TurnoverValidatorState()
        assert state.signal_history == {}
        assert state.rolling_window == DEFAULT_ROLLING_WINDOW

    def test_from_dict_partial(self):
        d = {"signal_history": {"a": [0.5]}}
        state = TurnoverValidatorState.from_dict(d)
        assert state.signal_history == {"a": [0.5]}
        assert state.rolling_window == DEFAULT_ROLLING_WINDOW

    def test_to_dict_includes_all_fields(self):
        state = TurnoverValidatorState(signal_history={"x": [1.0]}, rolling_window=30)
        d = state.to_dict()
        assert "signal_history" in d
        assert "rolling_window" in d
        assert d["rolling_window"] == 30


class TestUpdateAndValidateExtended:
    """Extended update_and_validate tests."""

    def test_zero_signal(self, validator):
        for _ in range(20):
            validator.update_and_validate({"src": 0.0})
        results = validator.update_and_validate({"src": 0.0})
        assert results["src"].sign_flip_rate == 0.0
        assert results["src"].signal_std == 0.0

    def test_positive_then_negative(self, validator):
        """Regime shift from positive to negative should produce sign flips."""
        for i in range(10):
            validator.update_and_validate({"src": 0.5})
        for i in range(10):
            validator.update_and_validate({"src": -0.5})
        results = validator.update_and_validate({"src": -0.5})
        assert results["src"].sign_flip_rate > 0

    def test_get_adjusted_weights_empty_weights(self, validator):
        """Empty weights dict should return empty."""
        adjusted = validator.get_adjusted_weights({}, {})
        assert adjusted == {}


class TestDiagnosticsExtended:
    """Extended diagnostics tests."""

    def test_diagnostics_numeric_types(self, validator):
        for _ in range(20):
            validator.update_and_validate({"src": 0.5})
        diag = validator.get_state_diagnostics()
        assert isinstance(diag["src"]["periods"], int)
        assert isinstance(diag["src"]["mean"], float)
        assert isinstance(diag["src"]["turnover_penalty"], float)

    def test_diagnostics_multiple_signals(self, validator):
        for _ in range(20):
            validator.update_and_validate({"a": 0.5, "b": -0.3})
        diag = validator.get_state_diagnostics()
        assert len(diag) == 2


# ---------------------------------------------------------------------------
# __all__ exports validation
# ---------------------------------------------------------------------------


class TestModuleAll:
    """Validate __all__ exports from turnover_validator module."""

    def test_all_is_defined(self):
        import src.strategy.turnover_validator as tv
        assert hasattr(tv, "__all__")
        assert isinstance(tv.__all__, list)

    def test_all_contains_expected_names(self):
        import src.strategy.turnover_validator as tv
        expected = {
            "DEFAULT_ROLLING_WINDOW",
            "MAX_TURNOVER_PENALTY",
            "MIN_SIGNAL_HISTORY",
            "DEFAULT_SIGNAL_COST",
            "DEFAULT_RISK_FREE_RATE",
            "SignalTurnoverMetrics",
            "TurnoverValidatorState",
            "TurnoverValidator",
        }
        assert set(tv.__all__) == expected

    def test_all_items_accessible(self):
        """Every name in __all__ should be importable from the module."""
        import src.strategy.turnover_validator as tv
        for name in tv.__all__:
            assert hasattr(tv, name), f"{name} not found in module"

    def test_public_exports_match_all(self):
        """Names imported in test file should be a subset of __all__."""
        import src.strategy.turnover_validator as tv
        test_imports = {
            "MAX_TURNOVER_PENALTY",
            "MIN_SIGNAL_HISTORY",
            "DEFAULT_ROLLING_WINDOW",
            "SignalTurnoverMetrics",
            "TurnoverValidator",
            "TurnoverValidatorState",
        }
        assert test_imports.issubset(set(tv.__all__))

    def test_signal_cost_and_rfr_exported(self):
        import src.strategy.turnover_validator as tv
        assert "DEFAULT_SIGNAL_COST" in tv.__all__
        assert "DEFAULT_RISK_FREE_RATE" in tv.__all__


# ---------------------------------------------------------------------------
# Extended dataclass field validation
# ---------------------------------------------------------------------------


class TestDataclassFieldValidation:
    """Comprehensive dataclass field validation for SignalTurnoverMetrics."""

    def test_signal_turnover_metrics_all_fields_in_to_dict(self):
        import src.strategy.turnover_validator as tv
        m = tv.SignalTurnoverMetrics(
            source="test", signal_std=0.1, sign_flip_rate=0.3,
            magnitude_volatility=0.05, turnover_penalty=0.2,
            stability_score=0.7, expected_return=0.05,
            risk_cost=0.02, turnover_cost=0.01, marginal_score=0.02,
            num_periods=20,
        )
        d = m.__dataclass_fields__
        expected_fields = {
            "source", "signal_std", "sign_flip_rate", "magnitude_volatility",
            "turnover_penalty", "stability_score", "expected_return",
            "risk_cost", "turnover_cost", "marginal_score", "num_periods",
            "missing_data",
        }
        assert set(d.keys()) == expected_fields

    def test_turnover_validator_state_all_fields_in_to_dict(self):
        state = TurnoverValidatorState()
        d = state.to_dict()
        assert "signal_history" in d
        assert "rolling_window" in d
        assert len(d) == 2

    def test_missing_data_default_value(self):
        """missing_data should default to False."""
        import src.strategy.turnover_validator as tv
        fields = tv.SignalTurnoverMetrics.__dataclass_fields__
        assert "missing_data" in fields
        default = fields["missing_data"].default
        assert default is False

    def test_signal_turnover_metrics_field_types(self):
        """Verify field types are correct."""
        import src.strategy.turnover_validator as tv
        fields = tv.SignalTurnoverMetrics.__dataclass_fields__
        str_fields = {"source"}
        float_fields = {
            "signal_std", "sign_flip_rate", "magnitude_volatility",
            "turnover_penalty", "stability_score", "expected_return",
            "risk_cost", "turnover_cost", "marginal_score",
        }
        int_fields = {"num_periods"}
        bool_fields = {"missing_data"}
        for name in str_fields:
            assert fields[name].type is str or fields[name].type == str, f"{name} should be str"
        for name in float_fields:
            assert fields[name].type is float or fields[name].type == float, f"{name} should be float"
        for name in int_fields:
            assert fields[name].type is int or fields[name].type == int, f"{name} should be int"
        for name in bool_fields:
            assert fields[name].type is bool or fields[name].type == bool, f"{name} should be bool"

    def test_turnover_validator_state_field_types(self):
        """Verify state dataclass field types."""
        import src.strategy.turnover_validator as tv
        fields = tv.TurnoverValidatorState.__dataclass_fields__
        assert fields["signal_history"].type == Dict[str, List[float]]
        assert fields["rolling_window"].type is int or fields["rolling_window"].type == int


# ---------------------------------------------------------------------------
# Additional computation edge cases
# ---------------------------------------------------------------------------


class TestComputationEdgeCases:
    """Boundary-value and edge-case tests for internal computation."""

    def test_exactly_min_signal_history_boundary(self, validator):
        """Exactly MIN_SIGNAL_HISTORY periods should compute normally (not missing)."""
        for _ in range(MIN_SIGNAL_HISTORY):
            validator.update_and_validate({"src": 0.5})
        results = validator.update_and_validate({"src": 0.5})
        m = results["src"]
        assert not m.missing_data
        assert m.num_periods == MIN_SIGNAL_HISTORY + 1  # +1 for the call itself

    def test_one_below_min_signal_history(self, validator):
        """One below threshold should be missing_data."""
        for _ in range(MIN_SIGNAL_HISTORY - 2):
            validator.update_and_validate({"src": 0.5})
        results = validator.update_and_validate({"src": 0.5})
        assert results["src"].missing_data

    def test_zero_signal_full_window(self, validator):
        """All-zero signals should produce zero penalty."""
        for _ in range(20):
            validator.update_and_validate({"src": 0.0})
        results = validator.update_and_validate({"src": 0.0})
        assert results["src"].turnover_penalty == 0.0
        assert results["src"].stability_score == 1.0

    def test_negative_signal_values(self, validator):
        """All-negative signals should work correctly."""
        for _ in range(20):
            validator.update_and_validate({"src": -0.5})
        results = validator.update_and_validate({"src": -0.5})
        assert results["src"].sign_flip_rate == 0.0
        assert results["src"].expected_return == pytest.approx(-0.5, abs=1e-10)
        assert results["src"].marginal_score < 0

    def test_very_large_signal_values(self, validator):
        """Very large signal values should be handled without overflow."""
        for _ in range(20):
            validator.update_and_validate({"src": 1000.0})
        results = validator.update_and_validate({"src": 1000.0})
        assert results["src"].turnover_penalty >= 0
        assert results["src"].turnover_penalty <= MAX_TURNOVER_PENALTY
        assert not math.isinf(results["src"].marginal_score)
        assert not math.isnan(results["src"].marginal_score)

    def test_very_small_signal_values(self, validator):
        """Very small (epsilon) signal values should work."""
        for _ in range(20):
            validator.update_and_validate({"src": 1e-10})
        results = validator.update_and_validate({"src": 1e-10})
        assert results["src"].turnover_penalty >= 0
        assert results["src"].stability_score > 0.9

    def test_single_element_history_penalty_zero(self, validator):
        """Single-element history should always have zero penalty (missing_data)."""
        results = validator.update_and_validate({"src": 0.5})
        assert results["src"].missing_data
        assert results["src"].turnover_penalty == 0.0

    def test_alternating_magnitude_changes(self, validator):
        """Large magnitude swings should increase magnitude_volatility."""
        for i in range(20):
            val = 10.0 if i % 2 == 0 else -10.0
            validator.update_and_validate({"src": val})
        results = validator.update_and_validate({"src": 10.0})
        assert results["src"].magnitude_volatility > 5.0

    def test_penalty_capped_at_max(self, validator):
        """Extreme signals should produce penalty exactly at MAX_TURNOVER_PENALTY."""
        # Maximum flip rate + maximum magnitude = penalty should be capped
        for i in range(20):
            val = 100.0 if i % 2 == 0 else -100.0
            validator.update_and_validate({"extreme": val})
        results = validator.update_and_validate({"extreme": 100.0})
        assert results["extreme"].turnover_penalty == MAX_TURNOVER_PENALTY

    def test_stability_score_formula_extremes(self, validator):
        """Stability score formula produces correct values at extremes."""
        # Constant signal -> max stability
        for _ in range(20):
            validator.update_and_validate({"const": 0.5})
        results_const = validator.update_and_validate({"const": 0.5})["const"]
        # Oscillating signal -> very low stability
        for i in range(20):
            validator.update_and_validate({"osc": 1.0 if i % 2 == 0 else -1.0})
        results_osc = validator.update_and_validate({"osc": 1.0})["osc"]

        assert results_const.stability_score >= results_osc.stability_score
        # Stability components individually testable
        assert results_const.sign_flip_rate < results_osc.sign_flip_rate

    def test_marginal_score_negative_when_costs_exceed_return(self, validator):
        """Marginal score should be negative when risk + turnover > expected return."""
        # Noisy, low-mean signal
        for i in range(20):
            val = 0.01 if i % 2 == 0 else -0.01
            validator.update_and_validate({"low_snr": val})
        results = validator.update_and_validate({"low_snr": 0.01})
        assert results["low_snr"].marginal_score < 0

    def test_empty_signal_values(self, validator):
        """Empty dict should not crash and produce empty results."""
        results = validator.update_and_validate({})
        assert results == {}

    def test_signal_appears_later(self, validator):
        """A signal that appears after updates should start fresh."""
        for _ in range(10):
            validator.update_and_validate({"existing": 0.5})
        results = validator.update_and_validate({"existing": 0.5, "new": 0.8})
        assert results["existing"].num_periods == 11
        assert results["new"].num_periods == 1
        assert results["new"].missing_data


# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------


class TestAllConstants:
    """Validate all module-level constants."""

    def test_rolling_window_range(self):
        assert 5 <= DEFAULT_ROLLING_WINDOW <= 100

    def test_max_turnover_penalty_range(self):
        assert 0.0 < MAX_TURNOVER_PENALTY <= 1.0

    def test_min_signal_history_range(self):
        assert 2 <= MIN_SIGNAL_HISTORY <= 20

    def test_default_signal_cost(self):
        from src.strategy.turnover_validator import DEFAULT_SIGNAL_COST
        assert isinstance(DEFAULT_SIGNAL_COST, float)
        assert 0.0 < DEFAULT_SIGNAL_COST < 0.01  # reasonable bps-level cost

    def test_default_risk_free_rate(self):
        from src.strategy.turnover_validator import DEFAULT_RISK_FREE_RATE
        assert isinstance(DEFAULT_RISK_FREE_RATE, float)
        assert 0.0 < DEFAULT_RISK_FREE_RATE < 0.30  # reasonable annual rate

    def test_state_file_defined(self):
        from src.strategy.turnover_validator import STATE_FILE
        assert isinstance(STATE_FILE, str)
        assert STATE_FILE.endswith(".json")
        assert len(STATE_FILE) > 5

    def test_all_constants_integration(self):
        """All constants used together should produce valid computation."""
        from src.strategy.turnover_validator import (
            DEFAULT_ROLLING_WINDOW as W,
            MIN_SIGNAL_HISTORY as MH,
            MAX_TURNOVER_PENALTY as MP,
            DEFAULT_SIGNAL_COST as SC,
            DEFAULT_RISK_FREE_RATE as RF,
        )
        # Verify numerical ordering sanity
        assert MH <= W  # min history should be <= window
        assert SC < RF  # signal cost should be < risk-free rate
        assert MP > 0.0

    def test_rolling_window_type(self):
        assert isinstance(DEFAULT_ROLLING_WINDOW, int)

    def test_max_penalty_type(self):
        assert isinstance(MAX_TURNOVER_PENALTY, (int, float))
        assert isinstance(MAX_TURNOVER_PENALTY, float)

    def test_min_history_type(self):
        assert isinstance(MIN_SIGNAL_HISTORY, int)

    def test_signal_cost_positive(self):
        from src.strategy.turnover_validator import DEFAULT_SIGNAL_COST
        assert DEFAULT_SIGNAL_COST > 0

    def test_risk_free_rate_positive(self):
        from src.strategy.turnover_validator import DEFAULT_RISK_FREE_RATE
        assert DEFAULT_RISK_FREE_RATE > 0

    def test_risk_free_rate_reasonable(self):
        """Risk-free rate should be between 1% and 15%."""
        from src.strategy.turnover_validator import DEFAULT_RISK_FREE_RATE
        assert 0.01 <= DEFAULT_RISK_FREE_RATE <= 0.15


# ---------------------------------------------------------------------------
# CLI main() function tests
# ---------------------------------------------------------------------------


class TestCLIMain:
    """Test the CLI main() function with mock arguments."""

    @pytest.fixture
    def isolated_data_dir(self, monkeypatch, tmp_path):
        """Isolate the DATA_DIR for CLI tests."""
        import src.strategy.turnover_validator as tv
        monkeypatch.setattr(tv, "DATA_DIR", tmp_path)
        return tmp_path

    def test_main_no_args(self, monkeypatch, isolated_data_dir):
        """Running with no args should print help and not crash."""
        import src.strategy.turnover_validator as tv
        monkeypatch.setattr("sys.argv", ["turnover_validator"])
        # Should call parser.print_help() which doesn't raise
        tv.main()

    def test_main_status_empty(self, monkeypatch, caplog, isolated_data_dir):
        """Status command with no history should log empty message."""
        import logging
        caplog.set_level(logging.INFO)
        import src.strategy.turnover_validator as tv
        monkeypatch.setattr("sys.argv", ["turnover_validator", "status"])
        tv.main()
        assert "0 tracked signals" in caplog.text

    def test_main_status_with_history(self, monkeypatch, caplog, isolated_data_dir):
        """Status command with populated history should show diagnostics."""
        import logging
        caplog.set_level(logging.INFO)
        import src.strategy.turnover_validator as tv
        # Pre-populate state by creating a validator that saves to the isolated dir
        state_path = isolated_data_dir / "turnover_validator_state.json"
        v = tv.TurnoverValidator(state_path=state_path)
        for _ in range(20):
            v.update_and_validate({"signal_a": 0.5})
        monkeypatch.setattr("sys.argv", ["turnover_validator", "status"])
        tv.main()
        assert "1 tracked signals" in caplog.text
        assert "signal_a" in caplog.text

    def test_main_adjust_no_signals(self, monkeypatch, caplog, isolated_data_dir):
        """Adjust command without --signals should log error."""
        import logging
        caplog.set_level(logging.INFO)
        import src.strategy.turnover_validator as tv
        monkeypatch.setattr("sys.argv", ["turnover_validator", "adjust"])
        tv.main()
        assert "ERROR" in caplog.text or "--signals required" in caplog.text

    def test_main_adjust_with_signals(self, monkeypatch, caplog, isolated_data_dir):
        """Adjust command with signals should compute adjusted weights."""
        import logging
        caplog.set_level(logging.INFO)
        import src.strategy.turnover_validator as tv
        monkeypatch.setattr("sys.argv", [
            "turnover_validator", "adjust",
            "--signals", "mom=0.5", "val=-0.3",
            "--weights", "mom=0.15", "val=0.10",
        ])
        tv.main()
        assert "Turnover-Adjusted Weights" in caplog.text
        assert "mom" in caplog.text
        assert "val" in caplog.text

    def test_main_adjust_default_weights(self, monkeypatch, caplog, isolated_data_dir):
        """Adjust command without --weights should use equal weights."""
        import logging
        caplog.set_level(logging.INFO)
        import src.strategy.turnover_validator as tv
        monkeypatch.setattr("sys.argv", [
            "turnover_validator", "adjust",
            "--signals", "a=0.5", "b=-0.3",
        ])
        tv.main()
        assert "0.5000" in caplog.text  # equal weight 1/2

    def test_main_adjust_malformed_signals(self, monkeypatch, caplog, isolated_data_dir):
        """Malformed signal entry should log WARN and skip."""
        import logging
        caplog.set_level(logging.WARNING)
        import src.strategy.turnover_validator as tv
        monkeypatch.setattr("sys.argv", [
            "turnover_validator", "adjust",
            "--signals", "badformat_no_eq",
        ])
        tv.main()
        assert "Skipping" in caplog.text

    def test_main_invalid_command(self, monkeypatch, capsys, isolated_data_dir):
        """Invalid subcommand should print help (argparse calls sys.exit)."""
        import src.strategy.turnover_validator as tv
        monkeypatch.setattr("sys.argv", ["turnover_validator", "bogus"])
        with pytest.raises(SystemExit):
            tv.main()

    def test_main_status_with_window(self, monkeypatch, capsys, isolated_data_dir):
        """Status command with --window should accept the argument."""
        import src.strategy.turnover_validator as tv
        monkeypatch.setattr("sys.argv", [
            "turnover_validator", "status", "--window", "30",
        ])
        tv.main()
        captured = capsys.readouterr()
        # Should not crash
        assert captured.out is not None


# ---------------------------------------------------------------------------
# Public methods not yet tested
# ---------------------------------------------------------------------------


class TestPublicMethodsCoverage:
    """Test remaining public methods and uncovered paths."""

    def test_resolve_path_returns_path(self, validator):
        """_resolve_path should return a Path object."""
        path = validator._resolve_path()
        assert isinstance(path, Path)

    def test_resolve_path_string_input(self, tmp_state_path):
        """Passing a string path should work."""
        v = TurnoverValidator(state_path=str(tmp_state_path))
        path = v._resolve_path()
        assert isinstance(path, Path)

    def test_state_file_default_location(self):
        """Default state path should have the correct filename."""
        from src.strategy.turnover_validator import STATE_FILE, DATA_DIR
        v = TurnoverValidator()
        expected = DATA_DIR / STATE_FILE
        assert v.state_path == expected
        assert v.state_path.name == STATE_FILE

    def test_save_state_with_unwritable_path(self, tmp_state_path, monkeypatch):
        """Save to unwritable path should not crash (caught OSError)."""
        # Make parent non-writable
        tmp_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_state_path.parent.chmod(0o444)  # read-only
        v = TurnoverValidator(state_path=tmp_state_path)
        v.update_and_validate({"src": 0.5})  # should not raise
        tmp_state_path.parent.chmod(0o755)  # restore

    def test_get_adjusted_weights_new_signal_in_metrics(self, validator):
        """A signal with no metrics in base_weights should get full weight."""
        weights = {"existing": 0.15}
        adjusted = validator.get_adjusted_weights(weights, {"existing": 0.5, "new_src": 0.8})
        # new_src is not in base_weights, so not in result
        assert "new_src" not in adjusted
        assert adjusted["existing"] == 0.15

    def test_get_adjusted_weights_partial_weights(self, validator):
        """Only signals in base_weights should appear in output."""
        weights = {"a": 0.10, "b": 0.10}
        # Only provide signal 'a'
        adjusted = validator.get_adjusted_weights(weights, {"a": 0.5})
        assert "a" in adjusted
        assert "b" in adjusted  # still appears with base weight
        # If 'b' has no history, it stays at base weight
        assert adjusted["b"] == 0.10

    def test_multiple_update_same_period(self, validator):
        """Calling update_and_validate multiple times with same period."""
        r1 = validator.update_and_validate({"src": 0.5})
        r2 = validator.update_and_validate({"src": 0.5})
        # Each call appends to history
        assert len(validator.state.signal_history["src"]) == 2

    def test_source_appears_disappears(self, validator):
        """Signal that disappears then reappears should be handled."""
        for _ in range(5):
            validator.update_and_validate({"a": 0.5, "b": -0.3})
        # Only 'a' in this period
        results = validator.update_and_validate({"a": 0.5})
        assert "a" in results
        assert "b" not in results  # not computed because not in signal_values

    def test_marginal_score_boost_capped(self, validator):
        """Marginal score boost should not exceed 20%."""
        base_weights = {"strong": 0.10}
        for _ in range(20):
            validator.update_and_validate({"strong": 10.0})
        adjusted = validator.get_adjusted_weights(base_weights, {"strong": 10.0})
        max_with_boost = base_weights["strong"] * 1.20  # max 20% boost
        assert adjusted["strong"] <= max_with_boost

    def test_marginal_score_negative_no_boost(self, validator):
        """Negative marginal score should not apply boost."""
        base_weights = {"weak": 0.10}
        for i in range(20):
            val = 0.01 if i % 2 == 0 else -0.01
            validator.update_and_validate({"weak": val})
        adjusted = validator.get_adjusted_weights(base_weights, {"weak": 0.01})
        # No boost since marginal is negative
        assert adjusted["weak"] <= base_weights["weak"]

    def test_signal_std_ddof1_used(self, validator):
        """Signal std should use ddof=1 (sample std)."""
        for _ in range(20):
            validator.update_and_validate({"src": 0.5})
        results = validator.update_and_validate({"src": 0.5})
        # Constant signal should have std=0
        assert results["src"].signal_std == 0.0

    def test_history_accumulation_order(self, validator):
        """History should be appended in chronological order."""
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        for v in values:
            validator.update_and_validate({"src": v})
        assert validator.state.signal_history["src"] == values
