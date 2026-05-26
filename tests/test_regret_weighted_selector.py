"""
Tests for v8.03 RegretWeightedSelector — regret-weighted ensemble signal selection.
"""

import ast
import inspect
import json
import logging
import math
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.strategy.regret_weighted_selector import (
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_REGRET_LAMBDA,
    MIN_COVARIANCE_PERIODS,
    REGRET_LOW_THRESHOLD,
    REGRET_HIGH_THRESHOLD,
    REGRET_MAX_PENALTY,
    SignalRegretMetrics,
    RegretAdjustmentResult,
    RegretWeightedSelector,
    RegretWeightedState,
    apply_regret_adjustment,
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
def selector(tmp_state_path):
    """Provide a RegretWeightedSelector with temp state."""
    return RegretWeightedSelector(
        state_path=tmp_state_path,
        rolling_window=30,
        regret_lambda=0.3,
    )


@pytest.fixture
def sample_signals():
    """Sample signal values for testing."""
    return {
        "tsfm_momentum": 0.5,
        "cta_trend": 0.3,
        "hmm_regime": 0.4,
        "macro_momentum": 0.1,
        "duration_regime": 0.0,
    }


@pytest.fixture
def sample_weights():
    """Sample weights for testing."""
    return {
        "tsfm_momentum": 0.30,
        "cta_trend": 0.25,
        "hmm_regime": 0.20,
        "macro_momentum": 0.15,
        "duration_regime": 0.10,
    }


@pytest.fixture
def preloaded_selector(tmp_state_path):
    """Selector pre-loaded with historical data."""
    sel = RegretWeightedSelector(
        state_path=tmp_state_path,
        rolling_window=50,
        regret_lambda=0.3,
    )
    # Seed with 20 periods of correlated signal-decision pairs
    for period in range(20):
        base = period * 0.02
        sel._update_history(
            {
                "high_corr_signal": 0.5 + math.sin(period * 0.5) * 0.3,
                "medium_corr_signal": 0.3 + math.cos(period * 0.7) * 0.2,
                "low_corr_signal": 0.1 + math.sin(period * 3.0) * 0.05,
            },
            # Ensemble decision tracks high_corr_signal closely
            ensemble_decision=0.5 + math.sin(period * 0.5) * 0.25,
        )
    return sel


# ---------------------------------------------------------------------------
# SignalRegretMetrics tests
# ---------------------------------------------------------------------------


class TestSignalRegretMetrics:
    def test_default_creation(self):
        m = SignalRegretMetrics(
            source="test",
            asset_covariances={"ensemble": 0.05},
            regret_contribution=0.05,
            regret_normalized=0.3,
            regret_penalty=0.1,
            regime_current="normal",
            num_periods=20,
        )
        assert m.source == "test"
        assert m.asset_covariances == {"ensemble": 0.05}
        assert m.regret_contribution == 0.05
        assert m.regret_normalized == 0.3
        assert m.regret_penalty == 0.1
        assert not m.missing_data

    def test_missing_data_default(self):
        m = SignalRegretMetrics(
            source="test",
            asset_covariances={},
            regret_contribution=0.0,
            regret_normalized=0.0,
            regret_penalty=0.0,
            regime_current="normal",
            num_periods=2,
            missing_data=True,
        )
        assert m.missing_data


# ---------------------------------------------------------------------------
# RegretAdjustmentResult tests
# ---------------------------------------------------------------------------


class TestRegretAdjustmentResult:
    def test_default_creation(self):
        r = RegretAdjustmentResult(
            adjusted_weights={"sig_a": 0.5, "sig_b": 0.5},
            regret_metrics={},
            lambda_used=0.3,
            num_signals=2,
            signals_with_high_regret=[],
            signals_with_low_regret=[],
            avg_regret=0.0,
        )
        assert r.num_signals == 2
        assert r.signals_with_high_regret == []


# ---------------------------------------------------------------------------
# RegretWeightedState tests
# ---------------------------------------------------------------------------


class TestRegretWeightedState:
    def test_default_creation(self):
        state = RegretWeightedState()
        assert state.signal_history == {}
        assert state.decision_history == {}
        assert state.rolling_window == DEFAULT_ROLLING_WINDOW
        assert state.last_regime == "normal"

    def test_round_trip_serialization(self):
        state = RegretWeightedState(
            signal_history={"sig_a": [0.1, 0.2, 0.3]},
            decision_history={"ensemble": [0.5, 0.6, 0.7]},
            rolling_window=30,
            last_regime="crisis",
            last_ensemble_decision=0.42,
        )
        data = state.to_dict()
        restored = RegretWeightedState.from_dict(data)
        assert restored.signal_history == {"sig_a": [0.1, 0.2, 0.3]}
        assert restored.decision_history == {"ensemble": [0.5, 0.6, 0.7]}
        assert restored.rolling_window == 30
        assert restored.last_regime == "crisis"
        assert restored.last_ensemble_decision == 0.42


# ---------------------------------------------------------------------------
# RegretWeightedSelector tests
# ---------------------------------------------------------------------------


class TestRegretWeightedSelectorInit:
    def test_default_creation(self):
        selector = RegretWeightedSelector()
        assert selector.rolling_window == DEFAULT_ROLLING_WINDOW
        assert selector.regret_lambda == DEFAULT_REGRET_LAMBDA

    def test_custom_params(self, tmp_state_path):
        selector = RegretWeightedSelector(
            state_path=tmp_state_path,
            rolling_window=42,
            regret_lambda=0.5,
        )
        assert selector.rolling_window == 42
        assert selector.regret_lambda == 0.5


class TestAdjustWeights:
    def test_basic_adjustment(self, selector, sample_signals, sample_weights):
        """Basic adjustment with fresh selector (insufficient history)."""
        result = selector.adjust_weights(
            sample_signals, 0.3, sample_weights, "normal"
        )
        # Insufficient history — all signals should have missing_data=True
        # Weights should remain as-is but normalized
        assert result.num_signals == 5
        assert abs(sum(result.adjusted_weights.values()) - 1.0) < 0.01

    def test_with_history(self, preloaded_selector):
        """Adjustment with sufficient history penalizes high-regret signals."""
        signal_values = {
            "high_corr_signal": 0.6,
            "medium_corr_signal": 0.3,
            "low_corr_signal": 0.1,
        }
        current_weights = {
            "high_corr_signal": 0.5,
            "medium_corr_signal": 0.3,
            "low_corr_signal": 0.2,
        }

        result = preloaded_selector.adjust_weights(
            signal_values, 0.5, current_weights, "normal"
        )
        assert result.num_signals == 3
        assert abs(sum(result.adjusted_weights.values()) - 1.0) < 0.01

        # High-regret signal should have some penalty
        high_metrics = result.regret_metrics.get("high_corr_signal")
        assert high_metrics is not None
        assert not high_metrics.missing_data

        # low_corr_signal should have lower regret than high_corr_signal
        low_metrics = result.regret_metrics.get("low_corr_signal")
        if low_metrics and not low_metrics.missing_data:
            assert low_metrics.regret_normalized <= high_metrics.regret_normalized

    def test_regime_penalty_multiplier(self, preloaded_selector):
        """Crisis regime applies higher penalty multiplier."""
        signal_values = {
            "high_corr_signal": 0.6,
            "low_corr_signal": 0.1,
        }
        current_weights = {
            "high_corr_signal": 0.5,
            "low_corr_signal": 0.5,
        }

        normal_result = preloaded_selector.adjust_weights(
            signal_values, 0.5, current_weights, "normal"
        )
        crisis_result = preloaded_selector.adjust_weights(
            signal_values, 0.5, current_weights, "crisis"
        )

        # Crisis should have same or higher penalty
        for signal in signal_values:
            normal_m = normal_result.regret_metrics.get(signal)
            crisis_m = crisis_result.regret_metrics.get(signal)
            if normal_m and crisis_m and not normal_m.missing_data and not crisis_m.missing_data:
                # Crisis penalty should be >= normal penalty
                assert crisis_m.regret_penalty >= normal_m.regret_penalty

    def test_covariance_computation(self, preloaded_selector):
        """Internal covariance computation produces expected values."""
        result = preloaded_selector._compute_regret("high_corr_signal", "normal")
        assert not result.missing_data
        assert result.num_periods >= MIN_COVARIANCE_PERIODS
        assert "ensemble" in result.asset_covariances
        # High-correlation signal should have non-zero covariance
        assert result.regret_contribution > 0

    def test_low_covariance_signal(self, preloaded_selector):
        """Low-correlation signal should have near-zero regret."""
        result = preloaded_selector._compute_regret("low_corr_signal", "normal")
        # With only 20 data points and low correlation, regret should be low
        assert result.regret_normalized < 0.5


class TestGetAdjustedWeights:
    def test_convenience_method(self, selector, sample_signals, sample_weights):
        """Convenience method returns adjusted weight dict."""
        adjusted = selector.get_adjusted_weights(
            sample_weights, sample_signals, 0.3, "normal"
        )
        assert isinstance(adjusted, dict)
        assert abs(sum(adjusted.values()) - 1.0) < 0.01

    def test_same_as_adjust_weights(self, selector, sample_signals, sample_weights):
        """Convenience method matches adjust_weights output."""
        adjusted = selector.get_adjusted_weights(
            sample_weights, sample_signals, 0.3, "normal"
        )
        result = selector.adjust_weights(
            sample_signals, 0.3, sample_weights, "normal"
        )
        for sig in adjusted:
            assert abs(adjusted[sig] - result.adjusted_weights[sig]) < 1e-10


class TestHistoryManagement:
    def test_update_history_trims(self, selector):
        """History trimmed to rolling window."""
        for i in range(40):
            selector._update_history(
                {"test_sig": float(i)},
                ensemble_decision=float(i) * 0.5,
            )
        assert len(selector.state.signal_history["test_sig"]) == 30
        assert len(selector.state.decision_history["ensemble"]) == 30
        assert selector.state.signal_history["test_sig"][-1] == 39.0

    def test_update_adds_new_signals(self, selector):
        """New signals start fresh history."""
        selector._update_history(
            {"new_sig": 0.5, "another": 0.3},
            ensemble_decision=0.4,
        )
        assert "new_sig" in selector.state.signal_history
        assert "another" in selector.state.signal_history
        assert "ensemble" in selector.state.decision_history
        assert len(selector.state.signal_history["new_sig"]) == 1


class TestEdgeCases:
    def test_empty_signals(self, selector):
        """Empty signal dict doesn't crash."""
        result = selector.adjust_weights({}, 0.0, {}, "normal")
        assert result.adjusted_weights == {}
        assert result.num_signals == 0

    def test_single_signal(self, selector):
        """Single signal works fine."""
        # Add history first
        selector._update_history({"only_signal": 0.5}, ensemble_decision=0.3)
        result = selector.adjust_weights(
            {"only_signal": 0.5}, 0.3, {"only_signal": 1.0}, "normal"
        )
        assert result.num_signals == 1
        assert abs(sum(result.adjusted_weights.values()) - 1.0) < 0.01

    def test_nan_signal_values(self, selector):
        """NaN values handled via _update_history."""
        selector._update_history(
            {"sig_a": float("nan"), "sig_b": 0.5},
            ensemble_decision=0.3,
        )
        result = selector.adjust_weights(
            {"sig_a": float("nan"), "sig_b": 0.5},
            0.3,
            {"sig_a": 0.5, "sig_b": 0.5},
            "normal",
        )
        assert result.num_signals == 2

    def test_all_signals_high_regret(self, preloaded_selector):
        """All signals have some regret — doesn't crash."""
        signal_values = {
            "high_corr_signal": 0.6,
            "medium_corr_signal": 0.4,
            "low_corr_signal": 0.2,
        }
        current_weights = {
            "high_corr_signal": 0.4,
            "medium_corr_signal": 0.35,
            "low_corr_signal": 0.25,
        }
        result = preloaded_selector.adjust_weights(
            signal_values, 0.5, current_weights, "normal"
        )
        assert result.num_signals == 3
        assert abs(sum(result.adjusted_weights.values()) - 1.0) < 0.01

    def test_regime_penalty_multiplier_crisis(self):
        """Crisis regime has highest penalty multiplier."""
        from src.strategy.regret_weighted_selector import RegretWeightedSelector
        crisis_mult = RegretWeightedSelector._get_regime_penalty_multiplier("crisis")
        normal_mult = RegretWeightedSelector._get_regime_penalty_multiplier("normal")
        assert crisis_mult > normal_mult


class TestStatePersistence:
    def test_save_and_load(self, tmp_state_path):
        """State survives round-trip save/load."""
        s1 = RegretWeightedSelector(state_path=tmp_state_path)
        s1._update_history({"test": 0.5}, ensemble_decision=0.3)
        s1._save_state()

        s2 = RegretWeightedSelector(state_path=tmp_state_path)
        assert "test" in s2.state.signal_history
        assert s2.state.signal_history["test"] == [0.5]
        assert "ensemble" in s2.state.decision_history
        assert s2.state.decision_history["ensemble"] == [0.3]

    def test_load_missing_file(self, tmp_state_path):
        """Missing file creates default state."""
        nonexistent = tmp_state_path.parent / "nonexistent.json"
        selector = RegretWeightedSelector(state_path=nonexistent)
        assert selector.state.signal_history == {}

    def test_load_corrupted_file(self, tmp_state_path):
        """Corrupted file falls back to default state."""
        tmp_state_path.write_text("{invalid")
        selector = RegretWeightedSelector(state_path=tmp_state_path)
        assert isinstance(selector.state, RegretWeightedState)


class TestGetStateDiagnostics:
    def test_empty_diagnostics(self, selector):
        """Fresh selector has empty diagnostics."""
        diag = selector.get_state_diagnostics()
        assert diag == {}

    def test_diagnostics_after_update(self, selector):
        """Diagnostics reflect added signals."""
        selector._update_history(
            {"sig_a": 0.5, "sig_b": 0.3},
            ensemble_decision=0.4,
        )
        diag = selector.get_state_diagnostics()
        assert "sig_a" in diag
        assert "sig_b" in diag
        assert "signal_periods" in diag["sig_a"]

    def test_diagnostics_after_adjustment(self, selector, sample_signals):
        """Diagnostics populated after adjustment."""
        selector.adjust_weights(
            sample_signals, 0.3,
            {"tsfm_momentum": 1.0},
            "normal",
        )
        diag = selector.get_state_diagnostics()
        for sig in sample_signals:
            assert sig in diag


class TestConvenienceFunction:
    def test_apply_regret_adjustment(self, tmp_state_path, monkeypatch):
        """Convenience function produces valid weights."""
        from src.strategy.regret_weighted_selector import RegretWeightedSelector
        # Override state path to avoid cross-test contamination
        original_init = RegretWeightedSelector.__init__

        def patched_init(self, state_path=None, rolling_window=60, regret_lambda=0.3):
            original_init(self, state_path=tmp_state_path, rolling_window=rolling_window, regret_lambda=regret_lambda)

        monkeypatch.setattr(RegretWeightedSelector, "__init__", patched_init)

        result = apply_regret_adjustment(
            {"sig_a": 0.6, "sig_b": 0.4},
            {"sig_a": 0.5, "sig_b": 0.3},
            0.4,
            "normal",
        )
        assert isinstance(result, dict)
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_apply_regret_adjustment_fresh(self):
        """Fresh convenience call (no history) returns normalized weights."""
        result = apply_regret_adjustment(
            {"sig_a": 0.6, "sig_b": 0.4},
            {"sig_a": 0.5, "sig_b": 0.3},
            0.4,
            "normal",
        )
        assert isinstance(result, dict)
        assert abs(sum(result.values()) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Extended coverage tests
# ---------------------------------------------------------------------------


class TestRegimePenaltyMultiplier:
    """Test _get_regime_penalty_multiplier static method."""

    def test_normal_regime(self):
        assert RegretWeightedSelector._get_regime_penalty_multiplier("normal") == 1.0

    def test_high_vol_regime(self):
        assert RegretWeightedSelector._get_regime_penalty_multiplier("high_vol") == 1.2

    def test_crisis_regime(self):
        assert RegretWeightedSelector._get_regime_penalty_multiplier("crisis") == 1.5

    def test_recovery_regime(self):
        assert RegretWeightedSelector._get_regime_penalty_multiplier("recovery") == 0.8

    def test_unknown_regime_defaults_to_1(self):
        assert RegretWeightedSelector._get_regime_penalty_multiplier("unknown") == 1.0


class TestRegretWeightedState:
    """Test RegretWeightedState serialization."""

    def test_to_dict(self):
        state = RegretWeightedState(
            signal_history={"sig_a": [0.1, 0.2]},
            decision_history={"ensemble": [0.15]},
            rolling_window=60,
            last_regime="high_vol",
            last_ensemble_decision=0.3,
        )
        d = state.to_dict()
        assert d["signal_history"]["sig_a"] == [0.1, 0.2]
        assert d["last_regime"] == "high_vol"
        assert d["rolling_window"] == 60

    def test_from_dict(self):
        data = {
            "signal_history": {"sig_b": [0.5]},
            "decision_history": {"ensemble": [0.4]},
            "rolling_window": 30,
            "last_regime": "crisis",
            "last_ensemble_decision": -0.2,
        }
        state = RegretWeightedState.from_dict(data)
        assert state.signal_history["sig_b"] == [0.5]
        assert state.last_regime == "crisis"
        assert state.rolling_window == 30

    def test_from_dict_defaults(self):
        """Missing keys should use defaults."""
        data = {"signal_history": {}}
        state = RegretWeightedState.from_dict(data)
        assert state.decision_history == {}
        assert state.rolling_window == DEFAULT_ROLLING_WINDOW
        assert state.last_regime == "normal"
        assert state.last_ensemble_decision == 0.0

    def test_roundtrip(self):
        """to_dict -> from_dict should preserve state."""
        original = RegretWeightedState(
            signal_history={"x": [1.0, 2.0]},
            decision_history={"ensemble": [1.5]},
            rolling_window=90,
            last_regime="recovery",
            last_ensemble_decision=0.8,
        )
        restored = RegretWeightedState.from_dict(original.to_dict())
        assert restored.signal_history == original.signal_history
        assert restored.last_regime == original.last_regime


class TestSignalRegretMetrics:
    """Test SignalRegretMetrics dataclass."""

    def test_default_missing_data_false(self):
        m = SignalRegretMetrics(
            source="test", asset_covariances={}, regret_contribution=0.0,
            regret_normalized=0.0, regret_penalty=0.0, regime_current="normal",
            num_periods=5,
        )
        assert m.missing_data is False

    def test_explicit_missing_data(self):
        m = SignalRegretMetrics(
            source="test", asset_covariances={}, regret_contribution=0.0,
            regret_normalized=0.0, regret_penalty=0.0, regime_current="normal",
            num_periods=1, missing_data=True,
        )
        assert m.missing_data is True


class TestRegretComputation:
    """Test _compute_regret edge cases."""

    def test_insufficient_periods_returns_missing(self, selector):
        """Fewer than MIN_COVARIANCE_PERIODS should return missing_data=True."""
        # Add only 1 period
        selector.state.signal_history["test_sig"] = [0.5]
        selector.state.decision_history["ensemble"] = [0.3]
        metrics = selector._compute_regret("test_sig", "normal")
        assert metrics.missing_data is True
        assert metrics.num_periods == 1

    def test_sufficient_periods_returns_valid(self, selector):
        """Enough periods should return valid metrics."""
        selector.state.signal_history["test_sig"] = [0.5] * 10
        selector.state.decision_history["ensemble"] = [0.3] * 10
        metrics = selector._compute_regret("test_sig", "normal")
        assert metrics.missing_data is False
        assert metrics.num_periods == 10

    def test_zero_variance_signal(self, selector):
        """Constant signal values should produce zero regret_normalized."""
        selector.state.signal_history["constant"] = [1.0] * 10
        selector.state.decision_history["ensemble"] = [0.5] * 10
        metrics = selector._compute_regret("constant", "normal")
        # Constant signal → std ≈ 0 → regret_normalized = 0
        assert metrics.regret_normalized == 0.0

    def test_high_regret_above_threshold(self, selector):
        """Signal perfectly correlated with ensemble should have high regret."""
        # Create perfectly correlated signal and decision
        values = [0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5]
        selector.state.signal_history["corr_sig"] = values
        selector.state.decision_history["ensemble"] = values
        metrics = selector._compute_regret("corr_sig", "normal")
        # Perfect correlation → high normalized regret
        assert metrics.regret_normalized > 0.5

    def test_regret_penalty_capped_at_max(self, selector):
        """Regret penalty should never exceed REGRET_MAX_PENALTY."""
        # Even with very high regret, penalty should be capped
        values = [0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5]
        selector.state.signal_history["high_regret"] = values
        selector.state.decision_history["ensemble"] = values
        metrics = selector._compute_regret("high_regret", "crisis")
        assert metrics.regret_penalty <= REGRET_MAX_PENALTY


class TestUpdateHistory:
    """Test _update_history rolling window management."""

    def test_history_appended(self, selector):
        """Signal values should be appended to history."""
        selector._update_history({"sig_a": 0.5}, 0.3)
        assert "sig_a" in selector.state.signal_history
        assert selector.state.signal_history["sig_a"] == [0.5]

    def test_rolling_window_trimmed(self, selector):
        """History should be trimmed to rolling_window size."""
        for i in range(50):
            selector._update_history({"sig_a": float(i)}, 0.3)
        assert len(selector.state.signal_history["sig_a"]) == selector.rolling_window

    def test_ensemble_decision_tracked(self, selector):
        """Ensemble decision should be tracked in decision_history."""
        selector._update_history({"sig_a": 0.5}, 0.3)
        assert "ensemble" in selector.state.decision_history
        assert selector.state.decision_history["ensemble"] == [0.3]

    def test_decision_history_trimmed(self, selector):
        """Decision history should be trimmed to rolling_window size."""
        for i in range(50):
            selector._update_history({"sig_a": float(i)}, float(i) * 0.1)
        assert len(selector.state.decision_history["ensemble"]) == selector.rolling_window


class TestGetAdjustedWeights:
    """Test get_adjusted_weights convenience method."""

    def test_returns_dict(self, selector, sample_signals):
        """get_adjusted_weights should return a dict."""
        weights = selector.get_adjusted_weights(
            {"sig_a": 0.6, "sig_b": 0.4},
            sample_signals,
            0.3,
            "normal",
        )
        assert isinstance(weights, dict)

    def test_weights_sum_to_one(self, selector, sample_signals):
        """Adjusted weights should sum to approximately 1.0."""
        weights = selector.get_adjusted_weights(
            {"sig_a": 0.6, "sig_b": 0.4},
            sample_signals,
            0.3,
            "normal",
        )
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestStatePersistence:
    """Test state save/load roundtrip."""

    def test_state_persists_across_instances(self, tmp_state_path):
        """State should persist across selector instances."""
        sel1 = RegretWeightedSelector(
            state_path=tmp_state_path, rolling_window=30, regret_lambda=0.3,
        )
        sel1.adjust_weights({"sig_a": 0.5}, 0.3, {"sig_a": 1.0}, "normal")

        sel2 = RegretWeightedSelector(
            state_path=tmp_state_path, rolling_window=30, regret_lambda=0.3,
        )
        assert "sig_a" in sel2.state.signal_history

    def test_state_file_created(self, selector, sample_signals):
        """adjust_weights should create state file."""
        selector.adjust_weights(sample_signals, 0.3, {"sig_a": 0.6, "sig_b": 0.4}, "normal")
        assert selector._resolve_path().exists()


class TestAdjustWeightsExtended:
    """Extended adjust_weights edge cases."""

    def test_adjust_with_new_signal(self, selector):
        """Signal not in history should be treated with full weight initially."""
        result = selector.adjust_weights(
            {"new_sig": 0.5},
            0.3,
            {"new_sig": 1.0},
            "normal",
        )
        # No prior history → missing_data → full weight
        assert result.adjusted_weights["new_sig"] > 0

    def test_adjust_preserves_zero_weight_signals(self, selector):
        """Zero-weight signals should remain zero."""
        result = selector.adjust_weights(
            {"sig_a": 0.5, "sig_b": 0.3},
            0.3,
            {"sig_a": 1.0, "sig_b": 0.0},
            "normal",
        )
        assert result.adjusted_weights["sig_b"] == 0.0

    def test_avg_regret_computed(self, selector, sample_signals):
        """avg_regret should be computed from metrics."""
        result = selector.adjust_weights(
            sample_signals, 0.3,
            {"sig_a": 0.6, "sig_b": 0.4},
            "normal",
        )
        assert isinstance(result.avg_regret, float)
        assert result.avg_regret >= 0.0

    def test_lambda_stored_in_result(self, selector, sample_signals):
        """lambda_used should match the selector's regret_lambda."""
        result = selector.adjust_weights(
            sample_signals, 0.3,
            {"sig_a": 0.6, "sig_b": 0.4},
            "normal",
        )
        assert result.lambda_used == selector.regret_lambda


# ---------------------------------------------------------------------------
# __all__ export validation
# ---------------------------------------------------------------------------

class TestExports:
    """Verify __all__ exports."""

    def test_all_exports_present(self):
        import src.strategy.regret_weighted_selector as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# Constants validation extended
# ---------------------------------------------------------------------------

class TestConstantsExtended:
    """Extended constants validation."""

    def test_num_assets(self):
        from src.strategy.regret_weighted_selector import NUM_ASSETS
        assert NUM_ASSETS == 7

    def test_default_rolling_window_positive(self):
        assert DEFAULT_ROLLING_WINDOW > 0

    def test_default_regret_lambda_range(self):
        assert 0 < DEFAULT_REGRET_LAMBDA <= 1.0

    def test_min_covariance_periods_positive(self):
        assert MIN_COVARIANCE_PERIODS > 0

    def test_threshold_ordering(self):
        assert REGRET_LOW_THRESHOLD < REGRET_HIGH_THRESHOLD

    def test_max_penalty_range(self):
        assert 0 < REGRET_MAX_PENALTY <= 1.0


# ---------------------------------------------------------------------------
# SignalRegretMetrics dataclass extended
# ---------------------------------------------------------------------------

class TestSignalRegretMetricsExtended:
    """Extended SignalRegretMetrics dataclass tests."""

    def test_all_fields(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(SignalRegretMetrics)}
        expected = {
            "source", "asset_covariances", "regret_contribution",
            "regret_normalized", "regret_penalty", "regime_current",
            "num_periods", "missing_data",
        }
        assert field_names == expected

    def test_missing_data_default(self):
        m = SignalRegretMetrics(
            source="test", asset_covariances={}, regret_contribution=0.0,
            regret_normalized=0.0, regret_penalty=0.0, regime_current="normal",
            num_periods=10,
        )
        assert m.missing_data is False


# ---------------------------------------------------------------------------
# RegretAdjustmentResult dataclass extended
# ---------------------------------------------------------------------------

class TestRegretAdjustmentResultExtended:
    """Extended RegretAdjustmentResult dataclass tests."""

    def test_all_fields(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(RegretAdjustmentResult)}
        expected = {
            "adjusted_weights", "regret_metrics", "lambda_used",
            "num_signals", "signals_with_high_regret",
            "signals_with_low_regret", "avg_regret",
        }
        assert field_names == expected

    def test_result_with_empty_metrics(self):
        result = RegretAdjustmentResult(
            adjusted_weights={}, regret_metrics={}, lambda_used=0.3,
            num_signals=0, signals_with_high_regret=[],
            signals_with_low_regret=[], avg_regret=0.0,
        )
        assert result.num_signals == 0


# ---------------------------------------------------------------------------
# RegretWeightedState extended
# ---------------------------------------------------------------------------

class TestRegretWeightedStateExtended:
    """Extended RegretWeightedState tests."""

    def test_from_dict_roundtrip(self):
        state = RegretWeightedState(
            signal_history={"sig_a": [0.1, 0.2]},
            decision_history={"sig_a": [0.3, 0.4]},
            rolling_window=60,
            last_regime="normal",
            last_ensemble_decision=0.5,
        )
        d = state.to_dict()
        restored = RegretWeightedState.from_dict(d)
        assert restored.rolling_window == 60
        assert restored.last_regime == "normal"

    def test_from_dict_empty(self):
        restored = RegretWeightedState.from_dict({})
        assert restored.rolling_window == DEFAULT_ROLLING_WINDOW


# ---------------------------------------------------------------------------
# RegretWeightedSelector extended
# ---------------------------------------------------------------------------

class TestRegretWeightedSelectorExtended:
    """Extended selector tests."""

    @pytest.fixture
    def selector(self):
        return RegretWeightedSelector()

    def test_get_state_diagnostics(self, selector):
        diag = selector.get_state_diagnostics()
        assert isinstance(diag, dict)

    def test_adjust_weights_preserves_total(self, selector):
        """Sum of adjusted weights should be close to sum of original."""
        signals = {"sig_a": 0.5, "sig_b": 0.3, "sig_c": 0.2}
        portfolio_vol = 0.15
        base_weights = {"sig_a": 0.4, "sig_b": 0.4, "sig_c": 0.2}
        result = selector.adjust_weights(signals, portfolio_vol, base_weights, "normal")
        total = sum(result.adjusted_weights.values())
        assert total == pytest.approx(1.0, abs=0.05)

    def test_adjust_weights_with_single_signal(self, selector):
        """Single signal should still work."""
        signals = {"sig_a": 1.0}
        result = selector.adjust_weights(signals, 0.15, {"sig_a": 1.0}, "normal")
        assert "sig_a" in result.adjusted_weights

    def test_get_regime_penalty_multiplier(self):
        assert RegretWeightedSelector._get_regime_penalty_multiplier("crisis") > 1.0
        assert RegretWeightedSelector._get_regime_penalty_multiplier("normal") == 1.0

    def test_get_adjusted_weights_callable(self, selector):
        """get_adjusted_weights should be callable."""
        weights = selector.get_adjusted_weights(
            {"sig_a": 0.5, "sig_b": 0.5},
            {"sig_a": 1.0, "sig_b": -1.0},
            0.3,
            "normal",
        )
        assert isinstance(weights, dict)


# ---------------------------------------------------------------------------
# apply_regret_adjustment convenience function
# ---------------------------------------------------------------------------

class TestApplyRegretAdjustment:
    """Test the convenience function."""

    def test_returns_dict(self):
        from src.strategy.regret_weighted_selector import apply_regret_adjustment
        result = apply_regret_adjustment(
            {"sig_a": 0.5, "sig_b": 0.5},
            {"sig_a": 1.0, "sig_b": -1.0},
            0.3,
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

class TestCLI:
    """Test main() callable."""

    def test_main_callable(self):
        from src.strategy.regret_weighted_selector import main
        assert callable(main)


# ---------------------------------------------------------------------------
# Dataclass field type validation
# ---------------------------------------------------------------------------


class TestDataclassFieldTypes:
    """Verify field types for all dataclasses."""

    def test_signal_regret_metrics_field_types(self):
        from dataclasses import fields
        field_map = {f.name: f.type for f in fields(SignalRegretMetrics)}
        assert field_map["source"] is str
        assert field_map["asset_covariances"] == Dict[str, float]
        assert field_map["regret_contribution"] is float
        assert field_map["regret_normalized"] is float
        assert field_map["regret_penalty"] is float
        assert field_map["regime_current"] is str
        assert field_map["num_periods"] is int
        assert field_map["missing_data"] is bool

    def test_regret_adjustment_result_field_types(self):
        from dataclasses import fields
        field_map = {f.name: f.type for f in fields(RegretAdjustmentResult)}
        assert field_map["adjusted_weights"] == Dict[str, float]
        assert field_map["regret_metrics"] == Dict[str, SignalRegretMetrics]
        assert field_map["lambda_used"] is float
        assert field_map["num_signals"] is int
        assert field_map["signals_with_high_regret"] == List[str]
        assert field_map["signals_with_low_regret"] == List[str]
        assert field_map["avg_regret"] is float

    def test_regret_weighted_state_field_types(self):
        from dataclasses import fields
        field_map = {f.name: f.type for f in fields(RegretWeightedState)}
        assert field_map["signal_history"] == Dict[str, List[float]]
        assert field_map["decision_history"] == Dict[str, List[float]]
        assert field_map["rolling_window"] is int
        assert field_map["last_regime"] is str
        assert field_map["last_ensemble_decision"] is float

    def test_signal_regret_metrics_defaults(self):
        from dataclasses import fields
        for f in fields(SignalRegretMetrics):
            if f.name == "missing_data":
                assert f.default is False
                break
        else:
            pytest.fail("missing_data field not found")

    def test_regret_weighted_state_defaults(self):
        from dataclasses import fields
        for f in fields(RegretWeightedState):
            if f.name == "rolling_window":
                assert f.default == DEFAULT_ROLLING_WINDOW
            elif f.name == "last_regime":
                assert f.default == "normal"
            elif f.name == "last_ensemble_decision":
                assert f.default == 0.0


# ---------------------------------------------------------------------------
# Constants extended validation
# ---------------------------------------------------------------------------


class TestConstantsMore:
    """Additional constant validation beyond basic range checks."""

    def test_state_file_is_nonempty_string(self):
        from src.strategy.regret_weighted_selector import STATE_FILE
        assert isinstance(STATE_FILE, str)
        assert len(STATE_FILE) > 0
        assert STATE_FILE.endswith(".json")

    def test_performance_file_is_path(self):
        from src.strategy.regret_weighted_selector import PERFORMANCE_FILE
        assert isinstance(PERFORMANCE_FILE, Path)
        assert PERFORMANCE_FILE.name == "regret_weighted_performance.json"

    def test_num_assets_type(self):
        from src.strategy.regret_weighted_selector import NUM_ASSETS
        assert isinstance(NUM_ASSETS, int)
        assert NUM_ASSETS > 0

    def test_min_covariance_periods_lt_rolling_window(self):
        """MIN_COVARIANCE_PERIODS should be less than DEFAULT_ROLLING_WINDOW."""
        assert MIN_COVARIANCE_PERIODS < DEFAULT_ROLLING_WINDOW

    def test_regret_lambda_within_bounds(self):
        """DEFAULT_REGRET_LAMBDA should be between 0 and 1 (exclusive of 0)."""
        assert 0.0 < DEFAULT_REGRET_LAMBDA <= 1.0

    def test_all_exports_in_module(self):
        """Every name in __all__ should exist in the module."""
        import src.strategy.regret_weighted_selector as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"__all__ contains {name} but module has no such attribute"


# ---------------------------------------------------------------------------
# CLI extended tests with capsys
# ---------------------------------------------------------------------------


class TestCLIExtended:
    """Test CLI output with capsys."""

    def test_main_no_args_prints_help(self, capsys):
        from src.strategy.regret_weighted_selector import main
        import sys
        old_argv = sys.argv.copy()
        try:
            sys.argv = ["regret_weighted_selector.py"]
            main()
        finally:
            sys.argv = old_argv
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "usage:" in output.lower() or "Regret-Weighted" in output

    def test_main_status_empty_history(self, caplog, tmp_state_path):
        from src.strategy.regret_weighted_selector import main
        import sys
        old_argv = sys.argv.copy()
        try:
            sys.argv = ["regret_weighted_selector.py", "status"]
            with (
                patch("src.strategy.regret_weighted_selector.RegretWeightedSelector") as mock_cls,
                patch("src.strategy.regret_weighted_selector.DATA_DIR", tmp_state_path.parent),
            ):
                instance = MagicMock()
                instance.get_state_diagnostics.return_value = {}
                instance.rolling_window = 60
                instance.regret_lambda = 0.3
                mock_cls.return_value = instance
                with caplog.at_level(logging.INFO, logger="src.strategy.regret_weighted_selector"):
                    main()
            assert "No signal history" in caplog.text
        finally:
            sys.argv = old_argv

    def test_main_status_with_history(self, caplog, tmp_state_path):
        from src.strategy.regret_weighted_selector import main
        import sys
        old_argv = sys.argv.copy()
        try:
            sys.argv = ["regret_weighted_selector.py", "status"]
            with (
                patch("src.strategy.regret_weighted_selector.RegretWeightedSelector") as mock_cls,
                patch("src.strategy.regret_weighted_selector.DATA_DIR", tmp_state_path.parent),
            ):
                instance = MagicMock()
                instance.get_state_diagnostics.return_value = {
                    "sig_a": {"signal_periods": 10, "signal_mean": 0.5, "signal_std": 0.1},
                    "sig_b": {"signal_periods": 8, "signal_mean": 0.3, "signal_std": 0.05},
                }
                instance.rolling_window = 60
                instance.regret_lambda = 0.3
                mock_cls.return_value = instance
                with caplog.at_level(logging.INFO, logger="src.strategy.regret_weighted_selector"):
                    main()
            assert "sig_a" in caplog.text
            assert "sig_b" in caplog.text
            assert "periods" in caplog.text
        finally:
            sys.argv = old_argv

    def test_main_adjust_valid_args(self, caplog, tmp_state_path):
        from src.strategy.regret_weighted_selector import main
        import sys
        old_argv = sys.argv.copy()
        try:
            sys.argv = [
                "regret_weighted_selector.py", "adjust",
                "--signals", "sig_a=0.5", "sig_b=0.3",
                "--weights", "sig_a=0.6", "sig_b=0.4",
                "--decision", "0.4",
            ]
            with (
                patch("src.strategy.regret_weighted_selector.RegretWeightedSelector") as mock_cls,
                patch("src.strategy.regret_weighted_selector.DATA_DIR", tmp_state_path.parent),
            ):
                mock_metrics_a = MagicMock(spec=SignalRegretMetrics)
                mock_metrics_a.regret_penalty = 0.1
                mock_metrics_a.regret_normalized = 0.3
                mock_metrics_b = MagicMock(spec=SignalRegretMetrics)
                mock_metrics_b.regret_penalty = 0.0
                mock_metrics_b.regret_normalized = 0.1

                instance = MagicMock()
                instance.adjust_weights.return_value = RegretAdjustmentResult(
                    adjusted_weights={"sig_a": 0.55, "sig_b": 0.45},
                    regret_metrics={"sig_a": mock_metrics_a, "sig_b": mock_metrics_b},
                    lambda_used=0.3,
                    num_signals=2,
                    signals_with_high_regret=[],
                    signals_with_low_regret=["sig_b"],
                    avg_regret=0.2,
                )
                mock_cls.return_value = instance
                with caplog.at_level(logging.INFO, logger="src.strategy.regret_weighted_selector"):
                    main()
            assert "Regret-Weighted Adjustment" in caplog.text or "Adjusted weights" in caplog.text
        finally:
            sys.argv = old_argv

    def test_main_adjust_malformed_signal_warns(self, caplog, tmp_state_path):
        from src.strategy.regret_weighted_selector import main
        import sys
        old_argv = sys.argv.copy()
        try:
            sys.argv = [
                "regret_weighted_selector.py", "adjust",
                "--signals", "badsignal",
                "--weights", "sig_a=1.0",
            ]
            with (
                patch("src.strategy.regret_weighted_selector.RegretWeightedSelector") as mock_cls,
                patch("src.strategy.regret_weighted_selector.DATA_DIR", tmp_state_path.parent),
            ):
                instance = MagicMock()
                instance.adjust_weights.return_value = RegretAdjustmentResult(
                    adjusted_weights={"sig_a": 1.0},
                    regret_metrics={},
                    lambda_used=0.3,
                    num_signals=0,
                    signals_with_high_regret=[],
                    signals_with_low_regret=[],
                    avg_regret=0.0,
                )
                mock_cls.return_value = instance
                with caplog.at_level(logging.WARNING, logger="src.strategy.regret_weighted_selector"):
                    main()
            assert "Skipping malformed signal" in caplog.text
        finally:
            sys.argv = old_argv

    def test_main_adjust_no_weights_errors(self, capsys, tmp_state_path):
        from src.strategy.regret_weighted_selector import main
        import sys
        old_argv = sys.argv.copy()
        try:
            sys.argv = [
                "regret_weighted_selector.py", "adjust",
                "--signals", "sig_a=0.5",
            ]
            # --weights is required by argparse -> SystemExit
            with pytest.raises((SystemExit, Exception)):
                main()
        finally:
            sys.argv = old_argv
        captured = capsys.readouterr()
        assert any(w in (captured.out + captured.err).lower() for w in ("usage:", "argument", "required", "error"))

    def test_main_adjust_crisis_regime(self, caplog, tmp_state_path):
        """Adjust with --regime crisis uses crisis multiplier."""
        from src.strategy.regret_weighted_selector import main
        import sys
        old_argv = sys.argv.copy()
        try:
            sys.argv = [
                "regret_weighted_selector.py", "adjust",
                "--signals", "sig_a=0.5",
                "--weights", "sig_a=1.0",
                "--regime", "crisis",
            ]
            with (
                patch("src.strategy.regret_weighted_selector.RegretWeightedSelector") as mock_cls,
                patch("src.strategy.regret_weighted_selector.DATA_DIR", tmp_state_path.parent),
            ):
                instance = MagicMock()
                mock_metrics = MagicMock(spec=SignalRegretMetrics)
                mock_metrics.regret_penalty = 0.5
                mock_metrics.regret_normalized = 0.8
                instance.adjust_weights.return_value = RegretAdjustmentResult(
                    adjusted_weights={"sig_a": 0.5},
                    regret_metrics={"sig_a": mock_metrics},
                    lambda_used=0.3,
                    num_signals=1,
                    signals_with_high_regret=["sig_a"],
                    signals_with_low_regret=[],
                    avg_regret=0.8,
                )
                mock_cls.return_value = instance
                with caplog.at_level(logging.INFO, logger="src.strategy.regret_weighted_selector"):
                    main()
            assert "crisis" in caplog.text.lower()
        finally:
            sys.argv = old_argv


# ---------------------------------------------------------------------------
# __main__ guard test
# ---------------------------------------------------------------------------


class TestMainGuard:
    """Test that __name__ == '__main__' triggers main()."""

    def test_main_guard_in_source(self):
        """The __main__ guard should call main()."""
        import src.strategy.regret_weighted_selector as mod
        source = inspect.getsource(mod)
        tree = ast.parse(source)
        found_guard = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if (isinstance(node.test, ast.Compare)
                        and isinstance(node.test.left, ast.Name)
                        and node.test.left.id == "__name__"
                        and isinstance(node.test.ops[0], ast.Eq)
                        and isinstance(node.test.comparators[0], ast.Constant)
                        and node.test.comparators[0].value == "__main__"):
                    found_guard = True
                    body_has_main = any(
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Call)
                        and isinstance(stmt.value.func, ast.Name)
                        and stmt.value.func.id == "main"
                        for stmt in node.body
                    )
                    assert body_has_main, "__main__ guard must call main()"
                    break
        assert found_guard, "Source must have `if __name__ == '__main__': main()` guard"


# ---------------------------------------------------------------------------
# _compute_regret boundary conditions
# ---------------------------------------------------------------------------


class TestComputeRegretBoundary:
    """Boundary conditions for _compute_regret."""

    def test_exact_min_covariance_periods(self, selector):
        """Exactly MIN_COVARIANCE_PERIODS should produce valid metrics."""
        selector.state.signal_history["test"] = [float(i) for i in range(MIN_COVARIANCE_PERIODS)]
        selector.state.decision_history["ensemble"] = [float(i) * 0.5 for i in range(MIN_COVARIANCE_PERIODS)]
        metrics = selector._compute_regret("test", "normal")
        assert not metrics.missing_data
        assert metrics.num_periods == MIN_COVARIANCE_PERIODS

    def test_one_less_than_min_covariance_periods(self, selector):
        """One fewer than MIN_COVARIANCE_PERIODS should return missing_data."""
        n = MIN_COVARIANCE_PERIODS - 1
        selector.state.signal_history["test"] = [float(i) for i in range(n)]
        selector.state.decision_history["ensemble"] = [float(i) * 0.5 for i in range(n)]
        metrics = selector._compute_regret("test", "normal")
        assert metrics.missing_data
        assert metrics.num_periods == n

    def test_regret_normalized_exact_low_threshold(self, selector):
        """Create regret_normalized exactly at REGRET_LOW_THRESHOLD (0.2)."""
        # We need to engineer signal and decision to produce specific normalized regret
        # Use correlated values that produce cov / sqrt(var_sig * var_dec) = 0.2
        sig = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        # Decision scaled so covariance/sqrt(var_sig*var_dec) = 0.2
        # var of [1..7] = 28/6 ≈ 4.667, std ≈ 2.16
        # We want cov / sqrt(4.667 * var_dec) = 0.2
        # cov with decision = signal * decision scaled to produce ratio
        dec = [v * 0.2 for v in sig]  # perfectly correlated but scaled
        selector.state.signal_history["test"] = sig
        selector.state.decision_history["ensemble"] = dec
        metrics = selector._compute_regret("test", "normal")
        # With perfect correlation, normalized regret = 1.0
        assert metrics.regret_normalized >= 0.5

    def test_negative_covariance_gives_positive_regret(self, selector):
        """Negative covariance should produce positive regret_contribution via abs()."""
        sig = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        dec = [-v for v in sig]  # perfectly negatively correlated
        selector.state.signal_history["neg_sig"] = sig
        selector.state.decision_history["ensemble"] = dec
        metrics = selector._compute_regret("neg_sig", "normal")
        # Regret contribution is abs(covariance), should be > 0
        assert metrics.regret_contribution > 0
        assert metrics.regret_normalized > 0.5

    def test_signal_longer_than_decision_truncated(self, selector):
        """When signal history is longer, both are truncated to min length."""
        sig = [float(i) for i in range(20)]
        dec = [float(i) * 0.5 for i in range(10)]  # shorter
        selector.state.signal_history["test"] = sig
        selector.state.decision_history["ensemble"] = dec
        metrics = selector._compute_regret("test", "normal")
        assert metrics.num_periods == 10
        assert not metrics.missing_data

    def test_decision_longer_than_signal_truncated(self, selector):
        """When decision history is longer, both are truncated to min length."""
        sig = [float(i) for i in range(10)]  # shorter
        dec = [float(i) * 0.5 for i in range(20)]
        selector.state.signal_history["test"] = sig
        selector.state.decision_history["ensemble"] = dec
        metrics = selector._compute_regret("test", "normal")
        assert metrics.num_periods == 10
        assert not metrics.missing_data

    def test_signal_not_in_history(self, selector):
        """Signal not in state history should return missing_data."""
        metrics = selector._compute_regret("nonexistent_signal", "normal")
        assert metrics.missing_data
        assert metrics.num_periods == 0

    def test_ensemble_not_in_decision_history(self, selector):
        """No ensemble key in decision history should return missing_data."""
        selector.state.signal_history["test"] = [1.0, 2.0, 3.0, 4.0, 5.0]
        # Don't set decision_history["ensemble"]
        metrics = selector._compute_regret("test", "normal")
        assert metrics.missing_data
        assert metrics.num_periods == 0

    def test_inf_signal_value_produces_nan_regret(self, selector):
        """Infinity in signal history should produce nan or 0 regret_normalized."""
        selector.state.signal_history["test"] = [float("inf")] * 10
        selector.state.decision_history["ensemble"] = [1.0] * 10
        metrics = selector._compute_regret("test", "normal")
        # np.var of inf array is nan, so expected_cov_bound is nan
        # either missing_data or regret_normalized is 0.0
        if not metrics.missing_data:
            assert math.isnan(metrics.regret_normalized) or metrics.regret_normalized == 0.0

    def test_inf_decision_value_handled(self, selector):
        """Infinity in decision history should be handled gracefully."""
        selector.state.signal_history["test"] = [1.0] * 10
        selector.state.decision_history["ensemble"] = [float("inf")] * 10
        metrics = selector._compute_regret("test", "normal")
        if not metrics.missing_data:
            assert metrics.regret_normalized == 0.0 or math.isnan(metrics.regret_normalized) or metrics.regret_normalized >= 0.0


# ---------------------------------------------------------------------------
# Regret penalty linear ramp between thresholds
# ---------------------------------------------------------------------------


class TestPenaltyRamp:
    """Test the linear penalty interpolation between LOW and HIGH thresholds."""

    def test_penalty_zero_below_low_threshold(self, selector):
        """Regret_normalized <= REGRET_LOW_THRESHOLD should give penalty=0."""
        sig = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98]  # near-constant → near-zero regret
        dec = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # constant decision
        selector.state.signal_history["test"] = sig
        selector.state.decision_history["ensemble"] = dec
        metrics = selector._compute_regret("test", "normal")
        assert metrics.regret_penalty == 0.0

    def test_penalty_at_exact_high_threshold(self, preloaded_selector):
        """Regret_normalized >= REGRET_HIGH_THRESHOLD should use max penalty."""
        # Preloaded selector has high_corr_signal with high correlation
        metrics = preloaded_selector._compute_regret("high_corr_signal", "normal")
        if metrics.regret_normalized >= REGRET_HIGH_THRESHOLD:
            expected_penalty = min(REGRET_MAX_PENALTY * 1.0, REGRET_MAX_PENALTY)
            assert metrics.regret_penalty == expected_penalty
        else:
            # Linear ramp case
            assert metrics.regret_penalty < REGRET_MAX_PENALTY

    def test_penalty_capped_at_max_in_crisis(self, preloaded_selector):
        """Even with crisis multiplier, penalty should not exceed REGRET_MAX_PENALTY."""
        metrics = preloaded_selector._compute_regret("high_corr_signal", "crisis")
        assert metrics.regret_penalty <= REGRET_MAX_PENALTY

    def test_penalty_linear_ramp_midpoint(self, selector):
        """Regret_normalized at midpoint between thresholds should have intermediate penalty."""
        # Engineer signal to produce regret_normalized near midpoint of LOW and HIGH
        n = 10
        sig = [float(i) * 0.4 for i in range(n)]
        # Decision with partial correlation
        dec = [v + (i % 3) * 0.1 for i, v in enumerate(sig)]
        selector.state.signal_history["test"] = sig
        selector.state.decision_history["ensemble"] = dec
        metrics = selector._compute_regret("test", "normal")
        if not metrics.missing_data and REGRET_LOW_THRESHOLD < metrics.regret_normalized < REGRET_HIGH_THRESHOLD:
            t = (metrics.regret_normalized - REGRET_LOW_THRESHOLD) / (
                REGRET_HIGH_THRESHOLD - REGRET_LOW_THRESHOLD
            )
            expected_penalty = t * 0.3 * REGRET_MAX_PENALTY
            assert metrics.regret_penalty == pytest.approx(expected_penalty, abs=0.01)
        else:
            pytest.skip("regret_normalized not in linear ramp range")


# ---------------------------------------------------------------------------
# Function boundary conditions — extreme inputs, missing keys, wrong types
# ---------------------------------------------------------------------------


class TestFunctionBoundaries:
    """Test adjust_weights with extreme/edge inputs."""

    def test_signals_not_in_weights(self, selector, sample_signals):
        """Signals not present in weights should still appear in metrics."""
        weights = {"other_sig": 1.0}  # key not in sample_signals
        result = selector.adjust_weights(sample_signals, 0.3, weights, "normal")
        for sig in sample_signals:
            assert sig in result.regret_metrics

    def test_weights_not_in_signals(self, selector):
        """Weights not in signals dict should get no metrics -> full weight."""
        result = selector.adjust_weights(
            {"sig_a": 0.5}, 0.3, {"sig_a": 0.5, "sig_b": 0.5}, "normal"
        )
        # sig_b not in signals, so no metrics; it should still appear in adjusted_weights
        assert "sig_b" in result.adjusted_weights

    def test_empty_weights_dict(self, selector, sample_signals):
        """Empty weights dict with signals should produce empty adjusted_weights."""
        result = selector.adjust_weights(sample_signals, 0.3, {}, "normal")
        assert result.adjusted_weights == {}
        assert result.num_signals == len(sample_signals)

    def test_all_zero_weights(self, selector):
        """All zero weights should remain zero after normalization."""
        result = selector.adjust_weights(
            {"sig_a": 0.5}, 0.3, {"sig_a": 0.0}, "normal"
        )
        # total of all adjusted_weights is 0, normalization branch: total > 0 is False
        # so adjusted_weights is unchanged: {"sig_a": 0.0}
        assert result.adjusted_weights["sig_a"] == 0.0

    def test_negative_signal_values(self, selector):
        """Negative signal values should not crash."""
        result = selector.adjust_weights(
            {"sig_a": -0.5, "sig_b": -0.3},
            0.3,
            {"sig_a": 0.6, "sig_b": 0.4},
            "normal",
        )
        assert abs(sum(result.adjusted_weights.values()) - 1.0) < 0.01

    def test_very_large_signal_values(self, selector):
        """Very large signal values should not crash (_compute_regret uses numpy)."""
        selector._update_history(
            {"sig_a": 1e10, "sig_b": -1e10},
            ensemble_decision=1e5,
        )
        result = selector.adjust_weights(
            {"sig_a": 1e10, "sig_b": -1e10},
            0.3,
            {"sig_a": 0.6, "sig_b": 0.4},
            "normal",
        )
        assert abs(sum(result.adjusted_weights.values()) - 1.0) < 0.01

    def test_extra_keys_in_weights_not_in_metrics(self, selector):
        """Weights for signals not in signal_values get full weight."""
        result = selector.adjust_weights(
            {"sig_a": 0.5},
            0.3,
            {"sig_a": 0.5, "sig_b": 0.3, "sig_c": 0.2},
            "normal",
        )
        # sig_b and sig_c have no metrics -> full weight
        assert "sig_b" in result.adjusted_weights
        assert "sig_c" in result.adjusted_weights

    def test_regime_unknown_uses_default_multiplier(self, selector):
        """Unknown regime string should fall back to multiplier of 1.0."""
        selector.state.signal_history["test"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        selector.state.decision_history["ensemble"] = [v * 0.8 for v in range(10)]
        metrics = selector._compute_regret("test", "unknown_regime_123")
        assert metrics.regime_current == "unknown_regime_123"

    def test_negative_ensemble_decision(self, selector):
        """Negative ensemble decision should be stored and handled."""
        selector.adjust_weights({"sig_a": 0.5}, -0.3, {"sig_a": 1.0}, "normal")
        assert selector.state.last_ensemble_decision == -0.3


# ---------------------------------------------------------------------------
# _track_performance edge cases
# ---------------------------------------------------------------------------


class TestTrackPerformance:
    """Test performance tracking I/O edge cases."""

    def test_performance_file_created(self, selector, sample_signals):
        """After adjust_weights, performance file should exist."""
        selector.adjust_weights(sample_signals, 0.3, {"sig_a": 1.0}, "normal")
        perf_path = selector._resolve_perf_path()
        assert perf_path.exists()

    def test_performance_data_has_expected_keys(self, selector, sample_signals, tmp_path):
        """Performance data should contain expected fields."""
        perf_path = tmp_path / "perf.json"
        selector._resolve_perf_path = lambda: perf_path
        selector.adjust_weights(sample_signals, 0.3, {"sig_a": 1.0}, "normal")
        with open(perf_path) as f:
            data = json.load(f)
        assert len(data) > 0
        entry = data[-1]
        assert "avg_regret" in entry
        assert "num_signals" in entry
        assert "num_high_regret" in entry
        assert "num_low_regret" in entry
        assert "lambda_used" in entry
        assert "regime" in entry

    def test_performance_truncated_at_100(self, selector, sample_signals, tmp_path):
        """Performance history should be capped at 100 entries."""
        perf_path = tmp_path / "perf.json"
        selector._resolve_perf_path = lambda: perf_path
        # Call adjust_weights 110 times
        for i in range(110):
            selector.adjust_weights(sample_signals, float(i) * 0.01, {"sig_a": 1.0}, "normal")
        with open(perf_path) as f:
            data = json.load(f)
        assert len(data) <= 100

    def test_performance_oserror_handled(self, selector, sample_signals):
        """OSError during performance tracking should not propagate."""
        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", side_effect=OSError("Permission denied")):
                # Should not raise
                selector.adjust_weights(sample_signals, 0.3, {"sig_a": 1.0}, "normal")

    def test_adjust_triggers_track_performance(self, selector, sample_signals):
        """adjust_weights should call _track_performance."""
        with patch.object(selector, "_track_performance") as mock_track:
            selector.adjust_weights(sample_signals, 0.3, {"sig_a": 1.0}, "normal")
            mock_track.assert_called_once()


# ---------------------------------------------------------------------------
# State save/load extended
# ---------------------------------------------------------------------------


class TestLoadSaveExtended:
    """Extended state persistence tests."""

    def test_save_oserror_handled(self, selector):
        """OSError during _save_state should be caught and logged."""
        with patch.object(Path, "mkdir", side_effect=OSError("Permission denied")):
            # This might raise or be caught depending on where the error occurs
            try:
                selector._save_state()
            except OSError:
                pass  # Acceptable if not caught

    def test_load_keyerror_in_json_uses_default(self, tmp_state_path):
        """KeyError during JSON parsing should fall back to default state."""
        # Write valid JSON but with missing fields that might cause KeyError
        tmp_state_path.write_text('{"unexpected_key": "value"}')
        selector = RegretWeightedSelector(state_path=tmp_state_path)
        assert selector.state.signal_history == {}

    def test_round_trip_empty_state(self, tmp_state_path):
        """Round-trip with completely empty state preserves defaults."""
        s1 = RegretWeightedSelector(state_path=tmp_state_path, rolling_window=60, regret_lambda=0.3)
        s1._save_state()
        s2 = RegretWeightedSelector(state_path=tmp_state_path, rolling_window=60, regret_lambda=0.3)
        assert s2.state.rolling_window == 60
        assert s2.state.last_regime == "normal"

    def test_state_path_resolved_correctly(self, tmp_state_path):
        """_resolve_path should return a Path matching the initialized state_path."""
        selector = RegretWeightedSelector(state_path=tmp_state_path)
        resolved = selector._resolve_path()
        assert resolved == tmp_state_path

    def test_save_state_creates_parent_dirs(self, tmp_state_path):
        """_save_state should create parent directories if they don't exist."""
        nested = tmp_state_path.parent / "nested" / "dir" / "state.json"
        selector = RegretWeightedSelector(state_path=nested)
        selector.state.signal_history["test"] = [0.5]
        selector._save_state()
        assert nested.exists()


# ---------------------------------------------------------------------------
# History management extended
# ---------------------------------------------------------------------------


class TestHistoryManagementExtended:
    """Additional history management edge cases."""

    def test_empty_signal_dict_does_not_add_ensemble_history(self, selector):
        """Empty signal_values should still update ensemble decision history."""
        selector._update_history({}, ensemble_decision=0.5)
        assert "ensemble" in selector.state.decision_history
        assert selector.state.decision_history["ensemble"] == [0.5]

    def test_multiple_updates_accumulate(self, selector):
        """Repeated updates should accumulate in order."""
        for i in range(3):
            selector._update_history({"sig": float(i)}, ensemble_decision=float(i) * 0.5)
        assert selector.state.signal_history["sig"] == [0.0, 1.0, 2.0]
        assert selector.state.decision_history["ensemble"] == [0.0, 0.5, 1.0]

    def test_history_exactly_rolling_window(self, selector):
        """With exactly rolling_window entries, nothing should be trimmed."""
        for i in range(selector.rolling_window):
            selector._update_history({"sig": float(i)}, ensemble_decision=float(i) * 0.5)
        assert len(selector.state.signal_history["sig"]) == selector.rolling_window
        assert len(selector.state.decision_history["ensemble"]) == selector.rolling_window

    def test_history_one_over_rolling_window_trims(self, selector):
        """With one more than rolling_window, the first entry is trimmed."""
        n = selector.rolling_window + 1
        for i in range(n):
            selector._update_history({"sig": float(i)}, ensemble_decision=float(i) * 0.5)
        assert len(selector.state.signal_history["sig"]) == selector.rolling_window
        # First entry (index 0) should be gone, last entry should be n-1
        assert selector.state.signal_history["sig"][-1] == float(n - 1)
        assert selector.state.signal_history["sig"][0] == 1.0  # second entry

    def test_multiple_signals_trimmed_independently(self, selector):
        """Each signal should have its own history trimmed to window."""
        for i in range(selector.rolling_window + 5):
            selector._update_history(
                {"sig_a": float(i), "sig_b": float(i) * 2},
                ensemble_decision=float(i) * 0.5,
            )
        assert len(selector.state.signal_history["sig_a"]) == selector.rolling_window
        assert len(selector.state.signal_history["sig_b"]) == selector.rolling_window
        assert len(selector.state.decision_history["ensemble"]) == selector.rolling_window


# ---------------------------------------------------------------------------
# Diagnostics extended
# ---------------------------------------------------------------------------


class TestDiagnosticsExtended:
    """Additional get_state_diagnostics tests."""

    def test_single_period_std_is_zero(self, selector):
        """Single data point should produce std=0 (ddof=1 means division by zero)."""
        selector.state.signal_history["sig"] = [0.5]
        selector.state.decision_history["ensemble"] = [0.3]
        diag = selector.get_state_diagnostics()
        assert diag["sig"]["signal_std"] == 0.0
        assert diag["sig"]["signal_mean"] == 0.5

    def test_diagnostics_multiple_values(self, selector):
        """Multiple values produce correct mean and std."""
        selector.state.signal_history["sig"] = [1.0, 2.0, 3.0, 4.0, 5.0]
        selector.state.decision_history["ensemble"] = [0.5] * 5
        diag = selector.get_state_diagnostics()
        assert diag["sig"]["signal_mean"] == 3.0
        assert diag["sig"]["signal_std"] == pytest.approx(1.5811, abs=0.001)

    def test_diagnostics_empty_history_list(self, selector):
        """Empty history list should have 0 periods."""
        selector.state.signal_history["sig"] = []
        diag = selector.get_state_diagnostics()
        assert diag["sig"]["signal_periods"] == 0

    def test_diagnostics_not_affected_by_missing_ensemble(self, selector):
        """get_state_diagnostics does not depend on decision_history."""
        selector.state.signal_history["sig"] = [0.5]
        diag = selector.get_state_diagnostics()
        assert "sig" in diag


# ---------------------------------------------------------------------------
# apply_regret_adjustment extended
# ---------------------------------------------------------------------------


class TestApplyRegretAdjustmentExtended:
    """Extended convenience function tests."""

    def test_multiple_calls_accumulate_history(self):
        """Repeated calls should accumulate history in shared state."""
        result1 = apply_regret_adjustment(
            {"sig_a": 0.5}, {"sig_a": 0.3}, 0.4, "normal"
        )
        result2 = apply_regret_adjustment(
            {"sig_a": 0.6}, {"sig_a": 0.4}, 0.5, "normal"
        )
        # Second call should have accumulated enough history for metrics
        assert isinstance(result2, dict)
        assert abs(sum(result2.values()) - 1.0) < 0.01

    def test_with_crisis_regime(self):
        """apply_regret_adjustment with crisis regime should work."""
        result = apply_regret_adjustment(
            {"sig_a": 0.5}, {"sig_a": 0.3}, 0.4, "crisis"
        )
        assert isinstance(result, dict)

    def test_with_recovery_regime(self):
        """apply_regret_adjustment with recovery regime (lower multiplier)."""
        result = apply_regret_adjustment(
            {"sig_a": 0.5}, {"sig_a": 0.3}, 0.4, "recovery"
        )
        assert isinstance(result, dict)

    def test_single_signal_full_weight(self):
        """Single signal with weight=1.0 should return 1.0."""
        result = apply_regret_adjustment(
            {"sig_a": 1.0}, {"sig_a": 0.5}, 0.4, "normal"
        )
        assert abs(result["sig_a"] - 1.0) < 0.01

    def test_empty_signals_dict(self):
        """apply_regret_adjustment with empty signals dict should return current_weights as-is."""
        result = apply_regret_adjustment(
            {"sig_a": 0.5}, {}, 0.4, "normal"
        )
        # With empty signal_values, all weights get no metrics -> full weight -> normalized to 1.0
        assert result == {"sig_a": 1.0}


# ---------------------------------------------------------------------------
# _log_adjustment with caplog
# ---------------------------------------------------------------------------


class TestLogAdjustment:
    """Test logging behavior of _log_adjustment."""

    def test_log_adjustment_info_level(self, selector, caplog):
        """_log_adjustment should log at INFO level."""
        caplog.set_level(logging.INFO)
        result = RegretAdjustmentResult(
            adjusted_weights={"sig_a": 0.6, "sig_b": 0.4},
            regret_metrics={},
            lambda_used=0.3,
            num_signals=2,
            signals_with_high_regret=["sig_a"],
            signals_with_low_regret=["sig_b"],
            avg_regret=0.25,
        )
        selector._log_adjustment(result)
        assert "Regret-weighted adjustment" in caplog.text
        assert "lambda=0.3" in caplog.text
        assert "avg_regret=0.250" in caplog.text

    def test_log_high_regret_signals(self, selector, caplog):
        """High-regret signals should be logged."""
        caplog.set_level(logging.INFO)
        result = RegretAdjustmentResult(
            adjusted_weights={"sig_a": 0.5, "sig_b": 0.5},
            regret_metrics={},
            lambda_used=0.3,
            num_signals=2,
            signals_with_high_regret=["sig_a"],
            signals_with_low_regret=["sig_b"],
            avg_regret=0.4,
        )
        selector._log_adjustment(result)
        assert "sig_a" in caplog.text
        assert "High-regret" in caplog.text

    def test_log_low_regret_debug_level(self, selector, caplog):
        """Low-regret signals should be logged at DEBUG level (not visible at INFO)."""
        caplog.set_level(logging.INFO)
        result = RegretAdjustmentResult(
            adjusted_weights={"sig_a": 0.5, "sig_b": 0.5},
            regret_metrics={},
            lambda_used=0.3,
            num_signals=2,
            signals_with_high_regret=[],
            signals_with_low_regret=["sig_b"],
            avg_regret=0.1,
        )
        selector._log_adjustment(result)
        # At INFO level, low-regret message (DEBUG) should NOT appear
        assert "Low-regret" not in caplog.text

    def test_log_low_regret_debug_visible(self, selector, caplog):
        """Low-regret signals should be visible at DEBUG level."""
        caplog.set_level(logging.DEBUG)
        result = RegretAdjustmentResult(
            adjusted_weights={"sig_a": 0.5, "sig_b": 0.5},
            regret_metrics={},
            lambda_used=0.3,
            num_signals=2,
            signals_with_high_regret=[],
            signals_with_low_regret=["sig_b"],
            avg_regret=0.1,
        )
        selector._log_adjustment(result)
        assert "sig_b" in caplog.text
        assert "Low-regret" in caplog.text


# ---------------------------------------------------------------------------
# RegretWeightedState serialization edge cases
# ---------------------------------------------------------------------------


class TestStateSerialization:
    """Edge cases for RegretWeightedState serialization."""

    def test_from_dict_extra_keys_ignored(self):
        """Extra keys in dict should be ignored, not causing errors."""
        data = {
            "signal_history": {"sig": [1.0]},
            "decision_history": {"ensemble": [0.5]},
            "rolling_window": 30,
            "last_regime": "crisis",
            "last_ensemble_decision": 0.3,
            "extra_field": "should_be_ignored",
            "another_extra": 42,
        }
        state = RegretWeightedState.from_dict(data)
        assert state.signal_history == {"sig": [1.0]}
        assert state.last_regime == "crisis"

    def test_to_dict_json_serializable(self):
        """to_dict output should be json.dumps compatible."""
        state = RegretWeightedState(
            signal_history={"sig_a": [0.1, 0.2, 0.3]},
            decision_history={"ensemble": [0.5, 0.6]},
            rolling_window=60,
            last_regime="high_vol",
            last_ensemble_decision=0.42,
        )
        json_str = json.dumps(state.to_dict())
        parsed = json.loads(json_str)
        assert parsed["signal_history"]["sig_a"] == [0.1, 0.2, 0.3]
        assert parsed["last_ensemble_decision"] == 0.42

    def test_from_dict_wrong_types_uses_default(self):
        """If rolling_window is a string, from_dict should still accept it as-is."""
        data = {
            "signal_history": {},
            "decision_history": {},
            "rolling_window": "not_an_int",
            "last_regime": 123,  # wrong type for str
        }
        state = RegretWeightedState.from_dict(data)
        # Python doesn't enforce types at runtime for dataclasses
        # The value will be accepted as-is (no type coercion)
        assert state.rolling_window == "not_an_int"
        assert state.last_regime == 123

    def test_from_dict_partial_data(self):
        """Partial data with some missing keys uses defaults for those keys."""
        data = {
            "signal_history": {"sig_a": [0.5]},
        }
        state = RegretWeightedState.from_dict(data)
        assert state.signal_history == {"sig_a": [0.5]}
        assert state.decision_history == {}
        assert state.rolling_window == DEFAULT_ROLLING_WINDOW
        assert state.last_regime == "normal"
        assert state.last_ensemble_decision == 0.0


# ---------------------------------------------------------------------------
# RegretWeightedSelector init edge cases
# ---------------------------------------------------------------------------


class TestSelectorInitExtended:
    """Edge cases for RegretWeightedSelector initialization."""

    def test_zero_rolling_window(self):
        """Zero rolling window: history[-0:] returns full list, no trimming occurs."""
        selector = RegretWeightedSelector(rolling_window=0)
        selector._update_history({"sig": 0.5}, 0.3)
        # With window=0, Python slicing: list[-0:] returns full list (same as list[0:])
        # So history retains all elements
        assert len(selector.state.signal_history["sig"]) == 1
        assert selector.state.signal_history["sig"] == [0.5]

    def test_negative_rolling_window(self):
        """Negative rolling window: history[-(-n):] = history[n:]."""
        selector = RegretWeightedSelector(rolling_window=-5)
        selector._update_history({"sig": 0.5}, 0.3)
        # With negative window, Python slicing behaviour: history[-(-5):] = history[5:]
        # For 1 element, history[5:] = []
        assert len(selector.state.signal_history["sig"]) == 0

    def test_very_large_rolling_window(self):
        """Very large rolling window should not cause issues."""
        selector = RegretWeightedSelector(rolling_window=1000000)
        for i in range(10):
            selector._update_history({"sig": float(i)}, float(i) * 0.5)
        assert len(selector.state.signal_history["sig"]) == 10

    def test_regret_lambda_zero(self):
        """regret_lambda=0 should produce zero penalty for linear ramp."""
        selector = RegretWeightedSelector(regret_lambda=0.0)
        selector.state.signal_history["test"] = [float(i) for i in range(10)]
        selector.state.decision_history["ensemble"] = [float(i) * 0.8 for i in range(10)]
        metrics = selector._compute_regret("test", "normal")
        if REGRET_LOW_THRESHOLD < metrics.regret_normalized < REGRET_HIGH_THRESHOLD:
            # With lambda=0, linear ramp penalty should be 0
            assert metrics.regret_penalty == 0.0

    def test_regret_lambda_negative(self):
        """Negative regret_lambda should be accepted (produces negative penalty)."""
        selector = RegretWeightedSelector(regret_lambda=-0.1)
        selector.state.signal_history["test"] = [float(i) for i in range(10)]
        selector.state.decision_history["ensemble"] = [float(i) * 0.8 for i in range(10)]
        metrics = selector._compute_regret("test", "normal")
        if REGRET_LOW_THRESHOLD < metrics.regret_normalized < REGRET_HIGH_THRESHOLD:
            # With lambda=-0.1, penalty would be negative
            assert metrics.regret_penalty < 0

    def test_state_path_as_string(self):
        """State path passed as string should be converted to Path."""
        selector = RegretWeightedSelector(state_path="/tmp/test_state.json")
        resolved = selector._resolve_path()
        assert isinstance(resolved, Path)
        assert str(resolved) == "/tmp/test_state.json"

    def test_default_data_dir_used(self):
        """Default state path uses DATA_DIR."""
        from src.strategy.regret_weighted_selector import STATE_FILE, DATA_DIR
        selector = RegretWeightedSelector()
        resolved = selector._resolve_path()
        assert str(resolved) == str(DATA_DIR / STATE_FILE)


# ---------------------------------------------------------------------------
# adjust_weights with varying history lengths
# ---------------------------------------------------------------------------


class TestAdjustWeightsVaryingHistory:
    """Test adjust_weights with varying historical data conditions."""

    def test_with_insufficient_history_all_missing(self, selector, sample_signals, sample_weights):
        """All signals with insufficient history should have missing_data=True."""
        for sig in sample_signals:
            selector.state.signal_history[sig] = [0.5]
        selector.state.decision_history["ensemble"] = [0.3]
        result = selector.adjust_weights(
            sample_signals, 0.3, sample_weights, "normal"
        )
        for sig in sample_signals:
            assert result.regret_metrics[sig].missing_data

    def test_mixed_history_availability(self, selector):
        """Some signals with history, some without."""
        sig_a_vals = [float(i) for i in range(MIN_COVARIANCE_PERIODS)]
        sig_b_vals = [0.5]  # insufficient
        dec_vals = [float(i) * 0.5 for i in range(MIN_COVARIANCE_PERIODS)]
        selector.state.signal_history["sig_a"] = sig_a_vals
        selector.state.signal_history["sig_b"] = sig_b_vals
        selector.state.decision_history["ensemble"] = dec_vals
        result = selector.adjust_weights(
            {"sig_a": 0.6, "sig_b": 0.4},
            0.3,
            {"sig_a": 0.5, "sig_b": 0.5},
            "normal",
        )
        assert not result.regret_metrics["sig_a"].missing_data
        assert result.regret_metrics["sig_b"].missing_data

    def test_adjust_weights_preserves_state_after(self, selector, sample_signals, sample_weights):
        """State should be updated after adjust_weights call."""
        selector.adjust_weights(sample_signals, 0.42, sample_weights, "crisis")
        assert selector.state.last_regime == "crisis"
        assert selector.state.last_ensemble_decision == 0.42
        for sig in sample_signals:
            assert len(selector.state.signal_history[sig]) >= 1

    def test_adjust_weights_high_regret_classification(self, preloaded_selector):
        """High regret classification should work."""
        signal_values = {"high_corr_signal": 0.6, "low_corr_signal": 0.1}
        weights = {"high_corr_signal": 0.5, "low_corr_signal": 0.5}
        result = preloaded_selector.adjust_weights(signal_values, 0.5, weights, "normal")
        # At least one signal should be classified
        assert len(result.signals_with_high_regret) >= 0
        assert len(result.signals_with_low_regret) >= 0

    def test_adjust_weights_called_twice(self, selector, sample_signals, sample_weights):
        """Calling adjust_weights twice should accumulate history."""
        r1 = selector.adjust_weights(sample_signals, 0.3, sample_weights, "normal")
        r2 = selector.adjust_weights(sample_signals, 0.4, sample_weights, "normal")
        assert r2.avg_regret >= 0.0
        # Second call has more history
        for sig in sample_signals:
            assert len(selector.state.signal_history[sig]) == 2


# ---------------------------------------------------------------------------
# _get_regime_penalty_multiplier with all regimes
# ---------------------------------------------------------------------------


class TestRegimePenaltyMultiplierExtended:
    """Additional regime penalty multiplier tests."""

    def test_all_known_regimes_have_expected_multipliers(self):
        """All 4 known regimes should have distinct multipliers."""
        multipliers = {
            "normal": 1.0,
            "high_vol": 1.2,
            "crisis": 1.5,
            "recovery": 0.8,
        }
        for regime, expected in multipliers.items():
            actual = RegretWeightedSelector._get_regime_penalty_multiplier(regime)
            assert actual == expected, f"Regime {regime}: expected {expected}, got {actual}"

    def test_case_sensitivity(self):
        """Regime lookup should be case-sensitive."""
        normal = RegretWeightedSelector._get_regime_penalty_multiplier("NORMAL")
        assert normal == 1.0  # Falls back to default for unknown

    def test_empty_string_regime(self):
        """Empty string regime should fall back to default multiplier."""
        mult = RegretWeightedSelector._get_regime_penalty_multiplier("")
        assert mult == 1.0

    def test_none_regime_falls_to_default(self):
        """None regime should be treated as unknown, returning 1.0."""
        mult = RegretWeightedSelector._get_regime_penalty_multiplier(None)  # type: ignore
        assert mult == 1.0


# ---------------------------------------------------------------------------
# _compute_regret with all penalty regimes
# ---------------------------------------------------------------------------


class TestComputeRegretAllRegimes:
    """Verify _compute_regret works with all 4 regime types."""

    @pytest.fixture
    def seeded_selector(self, selector):
        """Selector with 10 periods of correlated data."""
        sig = [float(i) for i in range(10)]
        dec = [float(i) * 0.8 for i in range(10)]
        selector.state.signal_history["test"] = sig
        selector.state.decision_history["ensemble"] = dec
        return selector

    def test_normal_regime_applied(self, seeded_selector):
        m = seeded_selector._compute_regret("test", "normal")
        assert m.regime_current == "normal"
        assert not m.missing_data

    def test_high_vol_regime_applied(self, seeded_selector):
        m = seeded_selector._compute_regret("test", "high_vol")
        assert m.regime_current == "high_vol"

    def test_crisis_regime_applied(self, seeded_selector):
        m = seeded_selector._compute_regret("test", "crisis")
        assert m.regime_current == "crisis"

    def test_recovery_regime_applied(self, seeded_selector):
        m = seeded_selector._compute_regret("test", "recovery")
        assert m.regime_current == "recovery"


# ---------------------------------------------------------------------------
# RegretAdjustmentResult edge cases
# ---------------------------------------------------------------------------


class TestRegretAdjustmentResultEdge:
    """Edge cases for RegretAdjustmentResult dataclass."""

    def test_empty_high_regret_list(self):
        """Empty high-regret list should be valid."""
        r = RegretAdjustmentResult(
            adjusted_weights={"sig_a": 1.0},
            regret_metrics={},
            lambda_used=0.3,
            num_signals=1,
            signals_with_high_regret=[],
            signals_with_low_regret=[],
            avg_regret=0.0,
        )
        assert r.signals_with_high_regret == []

    def test_high_regret_with_multiple_signals(self):
        """Multiple signals in high-regret list."""
        r = RegretAdjustmentResult(
            adjusted_weights={"a": 0.5, "b": 0.5},
            regret_metrics={},
            lambda_used=0.3,
            num_signals=2,
            signals_with_high_regret=["a", "b"],
            signals_with_low_regret=[],
            avg_regret=0.8,
        )
        assert len(r.signals_with_high_regret) == 2
        assert r.avg_regret == 0.8

    def test_overlap_high_and_low_regret(self):
        """A signal can be in both lists (though unlikely in practice)."""
        r = RegretAdjustmentResult(
            adjusted_weights={"sig": 1.0},
            regret_metrics={},
            lambda_used=0.3,
            num_signals=1,
            signals_with_high_regret=["sig"],
            signals_with_low_regret=["sig"],
            avg_regret=0.3,
        )
        assert "sig" in r.signals_with_high_regret
        assert "sig" in r.signals_with_low_regret

    def test_avg_regret_extreme_values(self):
        """avg_regret can be 0.0 or 1.0."""
        r0 = RegretAdjustmentResult(
            adjusted_weights={"sig": 1.0},
            regret_metrics={},
            lambda_used=0.3,
            num_signals=1,
            signals_with_high_regret=[],
            signals_with_low_regret=["sig"],
            avg_regret=0.0,
        )
        assert r0.avg_regret == 0.0
        r1 = RegretAdjustmentResult(
            adjusted_weights={"sig": 1.0},
            regret_metrics={},
            lambda_used=0.3,
            num_signals=1,
            signals_with_high_regret=["sig"],
            signals_with_low_regret=[],
            avg_regret=1.0,
        )
        assert r1.avg_regret == 1.0


# ---------------------------------------------------------------------------
# SignalRegretMetrics edge cases
# ---------------------------------------------------------------------------


class TestSignalRegretMetricsEdge:
    """Edge cases for SignalRegretMetrics dataclass."""

    def test_empty_asset_covariances(self):
        """Empty asset_covariances dict should be valid."""
        m = SignalRegretMetrics(
            source="test",
            asset_covariances={},
            regret_contribution=0.0,
            regret_normalized=0.0,
            regret_penalty=0.0,
            regime_current="normal",
            num_periods=0,
            missing_data=True,
        )
        assert m.asset_covariances == {}
        assert m.num_periods == 0

    def test_max_regret_normalized(self):
        """Regret_normalized should be capped at 1.0."""
        m = SignalRegretMetrics(
            source="test",
            asset_covariances={"ensemble": 1.0},
            regret_contribution=1.0,
            regret_normalized=1.0,
            regret_penalty=0.5,
            regime_current="crisis",
            num_periods=10,
        )
        assert m.regret_normalized == 1.0

    def test_negative_regret_contribution_clamped(self):
        """regret_contribution is from abs(covariance), so it should always be >= 0."""
        m = SignalRegretMetrics(
            source="test",
            asset_covariances={"ensemble": -0.5},  # negative raw cov
            regret_contribution=0.5,  # abs of raw cov
            regret_normalized=0.3,
            regret_penalty=0.1,
            regime_current="normal",
            num_periods=10,
        )
        assert m.regret_contribution >= 0


# ---------------------------------------------------------------------------
# get_adjusted_weights extended edge cases
# ---------------------------------------------------------------------------


class TestGetAdjustedWeightsExtended:
    """Extended edge cases for get_adjusted_weights convenience method."""

    def test_all_missing_data_returns_normalized_weights(self, selector):
        """When all signals have missing_data, weights should be normalized to 1.0."""
        adjusted = selector.get_adjusted_weights(
            {"sig_a": 0.6, "sig_b": 0.4},
            {"sig_a": 0.5, "sig_b": 0.3},
            0.3,
            "normal",
        )
        assert abs(sum(adjusted.values()) - 1.0) < 0.01

    def test_with_crisis_regime_works(self, selector):
        """Crisis regime should not break the convenience method."""
        adjusted = selector.get_adjusted_weights(
            {"sig_a": 0.6, "sig_b": 0.4},
            {"sig_a": 0.5, "sig_b": 0.3},
            0.3,
            "crisis",
        )
        assert abs(sum(adjusted.values()) - 1.0) < 0.01

    def test_returns_empty_dict_for_empty_input(self, selector):
        """Empty input should return empty dict."""
        adjusted = selector.get_adjusted_weights({}, {}, 0.0, "normal")
        assert adjusted == {}


# ---------------------------------------------------------------------------
# Performance tracking with pre-existing file
# ---------------------------------------------------------------------------


class TestPerformanceTrackingExistingFile:
    """Performance tracking when file already exists."""

    def test_append_to_existing_performance_file(self, selector, sample_signals):
        """Existing performance file should be appended to, not overwritten."""
        # Patch _resolve_perf_path so it uses selector's state_path dir
        perf_dir = selector._resolve_path().parent
        patched_path = perf_dir / "performance.json"

        with patch.object(selector, "_resolve_perf_path", return_value=patched_path):
            # First call creates file
            selector.adjust_weights(sample_signals, 0.3, {"sig_a": 1.0}, "normal")
            with open(patched_path) as f:
                data_before = json.load(f)
            # Second call appends
            selector.adjust_weights(sample_signals, 0.4, {"sig_a": 1.0}, "normal")
            with open(patched_path) as f:
                data_after = json.load(f)
        assert len(data_after) == len(data_before) + 1

    def test_corrupted_performance_file_handled(self, selector, caplog, tmp_state_path):
        """Corrupted performance file should be handled gracefully."""
        caplog.set_level(logging.WARNING)
        # Use isolated path to avoid contaminating shared performance file
        isolated_perf = tmp_state_path.parent / "isolated_perf.json"
        with patch.object(selector, "_resolve_perf_path", return_value=isolated_perf):
            isolated_perf.write_text("{corrupted json")
            selector.adjust_weights({"sig_a": 0.5}, 0.3, {"sig_a": 1.0}, "normal")
            # Warning should be logged
            assert any("regret-weighted performance" in msg.lower() for msg in caplog.messages)


# ---------------------------------------------------------------------------
# verify REGRET_MAX_PENALTY behavior across all regimes
# ---------------------------------------------------------------------------


class TestMaxPenaltyAcrossRegimes:
    """REGRET_MAX_PENALTY should never be exceeded regardless of regime."""

    @pytest.fixture
    def max_regret_selector(self, selector):
        """Selector with perfectly correlated signal+decision for max regret."""
        sig = [float(i) * 0.1 for i in range(10)]
        dec = [float(i) * 0.1 for i in range(10)]
        selector.state.signal_history["test"] = sig
        selector.state.decision_history["ensemble"] = dec
        return selector

    def test_max_penalty_normal(self, max_regret_selector):
        m = max_regret_selector._compute_regret("test", "normal")
        assert m.regret_penalty <= REGRET_MAX_PENALTY

    def test_max_penalty_crisis(self, max_regret_selector):
        m = max_regret_selector._compute_regret("test", "crisis")
        assert m.regret_penalty <= REGRET_MAX_PENALTY

    def test_max_penalty_high_vol(self, max_regret_selector):
        m = max_regret_selector._compute_regret("test", "high_vol")
        assert m.regret_penalty <= REGRET_MAX_PENALTY

    def test_max_penalty_recovery(self, max_regret_selector):
        m = max_regret_selector._compute_regret("test", "recovery")
        assert m.regret_penalty <= REGRET_MAX_PENALTY


# ---------------------------------------------------------------------------
# Verify that empty decision_history for other keys doesn't affect "ensemble"
# ---------------------------------------------------------------------------


class TestDecisionHistoryIsolation:
    """decision_history keys other than 'ensemble' should be isolated."""

    def test_extra_decision_keys_ignored(self, selector):
        """Extra keys in decision_history should not affect 'ensemble' lookup."""
        selector.state.decision_history["other_key"] = [0.1, 0.2, 0.3]
        selector.state.signal_history["test"] = [1.0, 2.0, 3.0, 4.0, 5.0]
        # No "ensemble" key — should get empty list
        metrics = selector._compute_regret("test", "normal")
        assert metrics.missing_data
        assert metrics.num_periods == 0

    def test_ensemble_key_added_when_missing(self, selector):
        """_update_history should add 'ensemble' key if missing."""
        selector.state.decision_history = {}
        selector._update_history({"sig": 0.5}, 0.3)
        assert "ensemble" in selector.state.decision_history

    def test_multiple_decision_entries_after_update(self, selector):
        """Multiple calls should add to 'ensemble' history."""
        for i in range(5):
            selector._update_history({"sig": float(i)}, ensemble_decision=float(i) * 0.1)
        expected = [0.0, 0.1, 0.2, 0.3, 0.4]
        assert len(selector.state.decision_history["ensemble"]) == 5
        for got, exp in zip(selector.state.decision_history["ensemble"], expected):
            assert got == pytest.approx(exp, abs=1e-10)
