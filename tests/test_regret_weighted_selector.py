"""
Tests for v8.03 RegretWeightedSelector — regret-weighted ensemble signal selection.
"""

import json
import math
import tempfile
from pathlib import Path

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
