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
