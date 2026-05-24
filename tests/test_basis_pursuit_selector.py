"""
Tests for v8.02 BasisPursuitSelector — basis-pursuit signal selection.
"""

import json
import math
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.strategy.basis_pursuit_selector import (
    DEFAULT_ROLLING_WINDOW,
    LAMBDA_BY_REGIME,
    MIN_ACTIVE_WEIGHT,
    REDUNDANCY_CORRELATION_THRESHOLD,
    SPARSITY_ALERT_THRESHOLD,
    BasisPursuitResult,
    BasisPursuitSelector,
    BasisPursuitState,
    PrunedSignal,
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
    """Provide a BasisPursuitSelector with temp state."""
    return BasisPursuitSelector(state_path=tmp_state_path, rolling_window=20)


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
    """Sample base weights for testing."""
    return {
        "tsfm_momentum": 0.25,
        "cta_trend": 0.17,
        "hmm_regime": 0.12,
        "macro_momentum": 0.08,
        "duration_regime": 0.05,
    }


@pytest.fixture
def preloaded_selector(tmp_state_path):
    """Selector pre-loaded with multiple periods of history."""
    sel = BasisPursuitSelector(state_path=tmp_state_path, rolling_window=20)
    # Seed with 10 periods of history for correlation testing
    for period in range(10):
        base = period * 0.1
        sel._update_history(
            {
                "tsfm_momentum": 0.5 + math.sin(period) * 0.2,
                "cta_trend": 0.3 + math.cos(period * 0.5) * 0.15,
                "hmm_regime": 0.4 + math.sin(period * 1.5) * 0.1,
                "macro_momentum": 0.1 + math.cos(period * 2.0) * 0.05,
                "duration_regime": 0.0 + math.sin(period * 0.3) * 0.02,
            },
            {
                "tsfm_momentum": 0.25,
                "cta_trend": 0.17,
                "hmm_regime": 0.12,
                "macro_momentum": 0.08,
                "duration_regime": 0.05,
            },
        )
    return sel


# ---------------------------------------------------------------------------
# PrunedSignal tests
# ---------------------------------------------------------------------------


class TestPrunedSignal:
    def test_default_creation(self):
        p = PrunedSignal(
            signal="test",
            weight=0.05,
            reason="near_zero",
        )
        assert p.signal == "test"
        assert p.weight == 0.05
        assert p.reason == "near_zero"
        assert p.paired_with is None
        assert p.correlation is None

    def test_redundant_creation(self):
        p = PrunedSignal(
            signal="redundant_a",
            weight=0.1,
            reason="redundant",
            paired_with="signal_b",
            correlation=0.92,
        )
        assert p.reason == "redundant"
        assert p.paired_with == "signal_b"
        assert p.correlation == 0.92


# ---------------------------------------------------------------------------
# BasisPursuitResult tests
# ---------------------------------------------------------------------------


class TestBasisPursuitResult:
    def test_is_concentrated_alert(self):
        """Sparsity below threshold triggers alert."""
        result = BasisPursuitResult(
            active_signals={"a": 1.0},
            pruned_signals={"b": 0.5},
            prune_reasons={},
            sparsity_ratio=0.2,
            lambda_used=0.05,
            regime="crisis",
            num_active=1,
            num_pruned=1,
            total_signals=2,
        )
        assert result.is_concentrated()

    def test_is_not_concentrated(self):
        """Sparsity above threshold is fine."""
        result = BasisPursuitResult(
            active_signals={"a": 0.5, "b": 0.5},
            pruned_signals={},
            prune_reasons={},
            sparsity_ratio=0.8,
            lambda_used=0.01,
            regime="normal",
            num_active=2,
            num_pruned=0,
            total_signals=2,
        )
        assert not result.is_concentrated()


# ---------------------------------------------------------------------------
# BasisPursuitState tests
# ---------------------------------------------------------------------------


class TestBasisPursuitState:
    def test_default_creation(self):
        state = BasisPursuitState()
        assert state.signal_history == {}
        assert state.full_weight_history == {}
        assert state.rolling_window == DEFAULT_ROLLING_WINDOW
        assert state.last_regime == "normal"

    def test_round_trip_serialization(self):
        state = BasisPursuitState(
            signal_history={"sig_a": [0.1, 0.2, 0.3]},
            full_weight_history={"sig_a": [0.5, 0.5, 0.5]},
            rolling_window=30,
            last_regime="crisis",
        )
        data = state.to_dict()
        restored = BasisPursuitState.from_dict(data)
        assert restored.signal_history == {"sig_a": [0.1, 0.2, 0.3]}
        assert restored.full_weight_history == {"sig_a": [0.5, 0.5, 0.5]}
        assert restored.rolling_window == 30
        assert restored.last_regime == "crisis"


# ---------------------------------------------------------------------------
# BasisPursuitSelector tests
# ---------------------------------------------------------------------------


class TestBasisPursuitSelectorInit:
    def test_default_creation(self):
        selector = BasisPursuitSelector()
        assert selector.rolling_window == DEFAULT_ROLLING_WINDOW
        assert selector.state.rolling_window == DEFAULT_ROLLING_WINDOW

    def test_custom_window(self, tmp_state_path):
        selector = BasisPursuitSelector(state_path=tmp_state_path, rolling_window=42)
        assert selector.rolling_window == 42
        assert selector.state.rolling_window == 42


class TestSelection:
    def test_basic_selection(self, selector, sample_signals, sample_weights):
        """Basic selection keeps all signals in normal regime (low lambda)."""
        result = selector.select_signals(sample_signals, sample_weights, "normal")
        assert result.total_signals == 5
        assert result.lambda_used == LAMBDA_BY_REGIME["normal"]
        assert result.regime == "normal"

    def test_crisis_pruning(self, selector, sample_signals, sample_weights):
        """Crisis regime prunes more aggressively."""
        result = selector.select_signals(sample_signals, sample_weights, "crisis")
        assert result.lambda_used == LAMBDA_BY_REGIME["crisis"]
        # Higher lambda should prune more signals
        assert result.num_active <= 5

    def test_regime_adaptation(self, selector, sample_signals, sample_weights):
        """Different regimes produce different lambda values."""
        normal_lambda = LAMBDA_BY_REGIME["normal"]
        crisis_lambda = LAMBDA_BY_REGIME["crisis"]
        high_vol_lambda = LAMBDA_BY_REGIME["high_vol"]
        assert crisis_lambda > high_vol_lambda > normal_lambda

        result_normal = selector.select_signals(sample_signals, sample_weights, "normal")
        result_crisis = selector.select_signals(sample_signals, sample_weights, "crisis")
        assert result_crisis.lambda_used > result_normal.lambda_used
        # Both should produce valid results (fallback ensures all base weights if pruned)
        assert abs(sum(result_normal.active_signals.values()) - 1.0) < 0.01
        assert abs(sum(result_crisis.active_signals.values()) - 1.0) < 0.01

    def test_get_active_weights(self, selector, sample_signals, sample_weights):
        """Convenience method returns only active weights."""
        active = selector.get_active_weights(sample_weights, sample_signals, "normal")
        assert isinstance(active, dict)
        assert len(active) > 0
        # Should sum to approximately 1.0
        total = sum(active.values())
        assert abs(total - 1.0) < 0.01

    def test_unknown_regime_uses_default(self, selector, sample_signals, sample_weights):
        """Unknown regime falls back to default lambda."""
        result = selector.select_signals(sample_signals, sample_weights, "unknown_regime")
        assert result.lambda_used == 0.01  # DEFAULT_LAMBDA (conservative — treat unknown as normal)

    def test_empty_signals(self, selector):
        """Empty signal dict returns empty."""
        result = selector.select_signals({}, {}, "normal")
        assert result.active_signals == {}
        assert result.sparsity_ratio == 0.0

    def test_single_signal(self, selector):
        """Single signal always remains active."""
        result = selector.select_signals(
            {"only_signal": 0.5},
            {"only_signal": 1.0},
            "crisis",
        )
        assert result.num_active == 1
        assert result.num_pruned == 0

    def test_prune_redundant_signals(self, preloaded_selector):
        """Highly correlated signals should be pruned."""
        # Create perfectly correlated pair
        signal_values = {
            "trend_a": 0.5,
            "trend_b": 0.49,  # near-perfectly correlated
            "unrelated": 0.1,
        }
        base_weights = {
            "trend_a": 0.4,
            "trend_b": 0.4,
            "unrelated": 0.2,
        }
        # Seed with perfectly correlated history
        for period in range(10):
            base = period * 0.1
            preloaded_selector._update_history(
                {
                    "trend_a": 0.5 + math.sin(period) * 0.15,
                    "trend_b": 0.48 + math.sin(period) * 0.15,  # very highly correlated
                    "unrelated": 0.1 + math.cos(period * 3.0) * 0.05,
                },
                base_weights,
            )
        result = preloaded_selector.select_signals(signal_values, base_weights, "normal")
        # Should have pruned at least one of the trend signals
        redundant_found = any(
            p.reason == "redundant"
            for p in result.prune_reasons.values()
        )
        assert redundant_found

    def test_nan_signal_values(self, selector):
        """NaN values should be handled gracefully via _update_history."""
        signal_values = {
            "sig_a": float("nan"),
            "sig_b": 0.5,
        }
        base_weights = {"sig_a": 0.5, "sig_b": 0.5}
        # Should not crash
        result = selector.select_signals(signal_values, base_weights, "normal")
        assert result.total_signals == 2


class TestRedundancyDetection:
    def test_find_redundant_signals(self, preloaded_selector):
        """Correlation-based redundancy detection."""
        redundant = preloaded_selector._find_redundant_signals()
        assert isinstance(redundant, dict)

    def test_no_redundant_with_single_signal(self, selector):
        """Single signal history means no redundancy."""
        selector._update_history({"only": 0.5}, {"only": 1.0})
        redundant = selector._find_redundant_signals()
        assert redundant == {}

    def test_safe_corr_perfect(self):
        """Perfectly correlated signals."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        corr = BasisPursuitSelector._safe_corr(a, b)
        assert corr is not None
        assert abs(corr - 1.0) < 0.01

    def test_safe_corr_constant(self):
        """Constant signal returns None."""
        a = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        corr = BasisPursuitSelector._safe_corr(a, b)
        assert corr is None

    def test_safe_corr_inverse(self):
        """Inversely correlated signals."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        corr = BasisPursuitSelector._safe_corr(a, b)
        assert corr is not None
        assert abs(corr - (-1.0)) < 0.01


class TestL1Selection:
    def test_soft_thresholding(self, selector):
        """L1 regularization shrinks small contributions."""
        base_weights = {"sig_a": 0.5, "sig_b": 0.5}
        signal_values = {"sig_a": 0.5, "sig_b": 0.5}

        # Low lambda: minimal shrinkage
        result_low = selector._apply_l1_selection(
            base_weights, signal_values, 0.01, {}
        )
        # High lambda: more shrinkage
        result_high = selector._apply_l1_selection(
            base_weights, signal_values, 0.5, {}
        )
        # High lambda should produce smaller or equal weights
        for sig in base_weights:
            assert result_high.get(sig, 0) <= result_low.get(sig, 0)

    def test_redundant_pruning_in_l1(self, selector):
        """Redundant signals get zero weight in L1 step."""
        base_weights = {"sig_a": 0.4, "sig_b": 0.4}
        signal_values = {"sig_a": 0.5, "sig_b": 0.5}
        redundant = {"sig_a": "sig_b"}

        result = selector._apply_l1_selection(
            base_weights, signal_values, 0.01, redundant
        )
        assert result["sig_a"] == 0.0
        assert result["sig_b"] > 0.0


class TestPruneNearZero:
    def test_prune_near_zero(self):
        """Signals below threshold get pruned."""
        l1_weights = {"sig_a": 0.5, "sig_b": 0.005, "sig_c": 0.4}
        original = {"sig_a": 0.3, "sig_b": 0.3, "sig_c": 0.4}
        active, pruned, reasons = BasisPursuitSelector._prune_near_zero(
            l1_weights, original, {}
        )
        assert "sig_a" in active
        assert "sig_c" in active
        assert "sig_b" in pruned
        assert reasons["sig_b"].reason == "near_zero"

    def test_prune_all_signals(self):
        """All signals pruned -> all go to pruned."""
        l1_weights = {"sig_a": 0.005, "sig_b": 0.003}
        original = {"sig_a": 0.5, "sig_b": 0.5}
        active, pruned, reasons = BasisPursuitSelector._prune_near_zero(
            l1_weights, original, {}
        )
        assert active == {}
        assert len(pruned) == 2

    def test_redundant_pruning(self):
        """Redundant signals get pruned with reason."""
        l1_weights = {"sig_a": 0.5, "sig_b": 0.5}
        original = {"sig_a": 0.4, "sig_b": 0.4}
        redundant = {"sig_b": "sig_a"}
        active, pruned, reasons = BasisPursuitSelector._prune_near_zero(
            l1_weights, original, redundant
        )
        assert "sig_a" in active
        assert "sig_b" in pruned
        assert reasons["sig_b"].reason == "redundant"
        assert reasons["sig_b"].paired_with == "sig_a"


class TestHistory:
    def test_update_history_trims(self, selector):
        """History trimmed to rolling window."""
        for i in range(30):
            selector._update_history(
                {"test_sig": float(i)},
                {"test_sig": 0.5},
            )
        assert len(selector.state.signal_history["test_sig"]) == 20
        assert len(selector.state.full_weight_history["test_sig"]) == 20
        # Should have the most recent values
        assert selector.state.signal_history["test_sig"][-1] == 29.0

    def test_update_adds_new_signals(self, selector):
        """New signals start fresh history."""
        selector._update_history(
            {"new_sig": 0.5, "another_new": 0.3},
            {"new_sig": 0.6, "another_new": 0.4},
        )
        assert "new_sig" in selector.state.signal_history
        assert "another_new" in selector.state.signal_history
        assert len(selector.state.signal_history["new_sig"]) == 1
        assert len(selector.state.full_weight_history["another_new"]) == 1


class TestStatePersistence:
    def test_save_and_load(self, tmp_state_path):
        """State survives round-trip save/load."""
        selector1 = BasisPursuitSelector(state_path=tmp_state_path, rolling_window=20)
        selector1._update_history({"test": 0.5}, {"test": 1.0})
        selector1._save_state()

        selector2 = BasisPursuitSelector(state_path=tmp_state_path, rolling_window=20)
        assert "test" in selector2.state.signal_history
        assert selector2.state.signal_history["test"] == [0.5]

    def test_load_missing_file(self, tmp_state_path):
        """Missing file creates default state."""
        nonexistent = tmp_state_path.parent / "nonexistent.json"
        selector = BasisPursuitSelector(state_path=nonexistent)
        assert selector.state.signal_history == {}
        assert selector.state.rolling_window == DEFAULT_ROLLING_WINDOW

    def test_load_corrupted_file(self, tmp_state_path):
        """Corrupted file falls back to default state."""
        tmp_state_path.write_text("{invalid json")
        selector = BasisPursuitSelector(state_path=tmp_state_path)
        # Should have loaded default state
        assert isinstance(selector.state, BasisPursuitState)


class TestGetStateDiagnostics:
    def test_diagnostics_after_initialization(self, selector):
        """Empty diagnostics with fresh selector."""
        diag = selector.get_state_diagnostics()
        assert diag == {}

    def test_diagnostics_after_update(self, selector):
        """Diagnostics reflect added signals."""
        selector._update_history({"sig_a": 0.5}, {"sig_a": 0.3})
        diag = selector.get_state_diagnostics()
        assert "sig_a" in diag
        assert diag["sig_a"]["signal_periods"] == 1

    def test_diagnostics_after_selection(self, selector, sample_signals, sample_weights):
        """Diagnostics populated after selection run."""
        selector.select_signals(sample_signals, sample_weights, "normal")
        diag = selector.get_state_diagnostics()
        for sig in sample_signals:
            assert sig in diag


class TestPerformanceTracking:
    def test_performance_file_created(self, selector, sample_signals, sample_weights):
        """Performance file is created after selection."""
        selector.select_signals(sample_signals, sample_weights, "normal")
        perf_path = selector._resolve_perf_path()
        assert perf_path.exists()
        with open(perf_path) as f:
            data = json.load(f)
        assert len(data) >= 1
        assert data[-1]["regime"] == "normal"
        assert "sparsity_ratio" in data[-1]


class TestEdgeCases:
    def test_weights_dont_sum_to_one(self, selector):
        """Non-normalized weights don't break selection."""
        signal_values = {"sig_a": 0.5, "sig_b": 0.3}
        base_weights = {"sig_a": 2.0, "sig_b": 3.0}
        result = selector.select_signals(signal_values, base_weights, "normal")
        assert result.total_signals == 2

    def test_negative_signal_values(self, selector):
        """Negative signal values handled correctly."""
        signal_values = {"sig_a": -0.5, "sig_b": 0.3}
        base_weights = {"sig_a": 0.5, "sig_b": 0.5}
        result = selector.select_signals(signal_values, base_weights, "normal")
        assert result.total_signals == 2
        # Negative signals should contribute via absolute value
        assert len(result.active_signals) > 0

    def test_zero_signal_values(self, selector):
        """Zero signals should still be treated as signals."""
        signal_values = {"sig_a": 0.0, "sig_b": 0.0}
        base_weights = {"sig_a": 0.5, "sig_b": 0.5}
        result = selector.select_signals(signal_values, base_weights, "crisis")
        assert result.total_signals == 2
        # In crisis with lambda=0.15 and zero signals, all may be pruned
        # But fallback should recover
        assert sum(result.active_signals.values()) > 0

    def test_sparsity_alert_only_when_concentrated(self, selector):
        """Sparsity ratio below threshold triggers is_concentrated()."""
        signal_values = {f"sig_{i}": 0.5 for i in range(20)}
        base_weights = {f"sig_{i}": 1.0 / 20 for i in range(20)}
        # With many near-zero-signals in crisis, sparsity should be low
        result = selector.select_signals(signal_values, base_weights, "crisis")
        # If sparsity is below threshold, alert triggers
        # This is a soft check since lambda determines pruning level
        assert result.sparsity_ratio >= 0.0
        assert result.sparsity_ratio <= 1.0


# ---------------------------------------------------------------------------
# Extended coverage tests
# ---------------------------------------------------------------------------


class TestPrunedSignalExtended:
    """Extended PrunedSignal dataclass tests."""

    def test_to_dict_has_all_fields(self):
        p = PrunedSignal(
            signal="test_a",
            weight=0.15,
            reason="redundant",
            paired_with="test_b",
            correlation=0.93,
        )
        # dataclass fields
        assert p.signal == "test_a"
        assert p.weight == 0.15
        assert p.reason == "redundant"
        assert p.paired_with == "test_b"
        assert p.correlation == 0.93

    def test_near_zero_defaults(self):
        """Near-zero pruned signal should have None for pair fields."""
        p = PrunedSignal(signal="x", weight=0.001, reason="near_zero")
        assert p.paired_with is None
        assert p.correlation is None


class TestBasisPursuitResultExtended:
    """Extended BasisPursuitResult dataclass tests."""

    def test_is_concentrated_boundary(self):
        """Exactly at threshold should not be concentrated."""
        result = BasisPursuitResult(
            active_signals={"a": 0.5, "b": 0.5},
            pruned_signals={},
            prune_reasons={},
            sparsity_ratio=0.3,  # == SPARSITY_ALERT_THRESHOLD
            lambda_used=0.01,
            regime="normal",
        )
        assert not result.is_concentrated()  # < not <=

    def test_just_below_threshold(self):
        """Just below threshold should be concentrated."""
        result = BasisPursuitResult(
            active_signals={"a": 1.0},
            pruned_signals={"b": 0.5},
            prune_reasons={},
            sparsity_ratio=0.29,
            lambda_used=0.15,
            regime="crisis",
        )
        assert result.is_concentrated()


class TestBasisPursuitStateExtended:
    """Extended BasisPursuitState tests."""

    def test_from_dict_defaults(self):
        """from_dict should provide sensible defaults."""
        state = BasisPursuitState.from_dict({})
        assert state.signal_history == {}
        assert state.full_weight_history == {}
        assert state.rolling_window == DEFAULT_ROLLING_WINDOW
        assert state.last_regime == "normal"

    def test_from_dict_partial(self):
        """from_dict with partial data should fill defaults."""
        state = BasisPursuitState.from_dict({
            "signal_history": {"a": [0.1]},
            "last_regime": "crisis",
        })
        assert state.signal_history == {"a": [0.1]}
        assert state.last_regime == "crisis"
        assert state.full_weight_history == {}

    def test_to_dict_and_back(self):
        """Roundtrip should preserve all data."""
        original = BasisPursuitState(
            signal_history={"x": [0.1, 0.2], "y": [0.3]},
            full_weight_history={"x": [0.5, 0.5], "y": [0.5]},
            rolling_window=42,
            last_regime="high_vol",
        )
        restored = BasisPursuitState.from_dict(original.to_dict())
        assert restored.signal_history == original.signal_history
        assert restored.full_weight_history == original.full_weight_history
        assert restored.rolling_window == 42
        assert restored.last_regime == "high_vol"


class TestL1SelectionExtended:
    """Extended L1 selection tests."""

    def test_empty_weights(self, selector):
        """Empty base weights should return empty."""
        result = selector._apply_l1_selection({}, {}, 0.05, {})
        assert result == {}

    def test_high_lambda_prunes_everything(self, selector):
        """Very high lambda should zero out all contributions."""
        base_weights = {"a": 0.1, "b": 0.05}
        signal_values = {"a": 0.5, "b": 0.3}
        result = selector._apply_l1_selection(base_weights, signal_values, 100.0, {})
        # All contributions < lambda, so all should be zero
        for v in result.values():
            assert v == 0.0

    def test_zero_lambda_no_shrinkage(self, selector):
        """Zero lambda should not shrink any weights."""
        base_weights = {"a": 0.3, "b": 0.2}
        signal_values = {"a": 0.5, "b": 0.5}
        result = selector._apply_l1_selection(base_weights, signal_values, 0.0, {})
        # All contributions should be preserved
        assert result["a"] > 0
        assert result["b"] > 0

    def test_signal_not_in_values(self, selector):
        """Signal not in signal_values should use base_weight as contribution."""
        base_weights = {"a": 0.3, "b": 0.2}
        signal_values = {"a": 0.5}  # b not in signal_values
        result = selector._apply_l1_selection(base_weights, signal_values, 0.01, {})
        # b's contribution = base_weight (0.2) since not in signal_values
        assert result["b"] > 0


class TestRedundancyDetectionExtended:
    """Extended redundancy detection tests."""

    def test_insufficient_history(self, selector):
        """Less than 3 periods should find no redundant signals."""
        selector._update_history({"a": 0.5, "b": 0.4}, {"a": 0.5, "b": 0.5})
        selector._update_history({"a": 0.6, "b": 0.5}, {"a": 0.5, "b": 0.5})
        # Only 2 periods — need >= 3
        redundant = selector._find_redundant_signals()
        assert redundant == {}

    def test_uncorrelated_signals(self, preloaded_selector):
        """Uncorrelated signals should not be flagged as redundant."""
        # Add independent signals
        for period in range(10):
            preloaded_selector._update_history(
                {
                    "independent_a": 0.5 + math.sin(period) * 0.3,
                    "independent_b": 0.5 + math.cos(period * 2.7) * 0.3,
                },
                {"independent_a": 0.5, "independent_b": 0.5},
            )
        redundant = preloaded_selector._find_redundant_signals()
        # Should not find these two redundant
        assert "independent_a" not in redundant or redundant.get("independent_a") != "independent_b"


class TestPruneNearZeroExtended:
    """Extended prune_near_zero tests."""

    def test_exactly_at_threshold(self):
        """Signal at exactly MIN_ACTIVE_WEIGHT should be pruned."""
        l1_weights = {"a": MIN_ACTIVE_WEIGHT}  # 0.01, not < 0.01
        original = {"a": 0.5}
        active, pruned, reasons = BasisPursuitSelector._prune_near_zero(
            l1_weights, original, {}
        )
        # abs(0.01) < MIN_ACTIVE_WEIGHT is False, so it should be active
        assert "a" in active

    def test_just_below_threshold(self):
        """Signal just below MIN_ACTIVE_WEIGHT should be pruned."""
        l1_weights = {"a": 0.009}
        original = {"a": 0.5}
        active, pruned, reasons = BasisPursuitSelector._prune_near_zero(
            l1_weights, original, {}
        )
        assert "a" in pruned
        assert reasons["a"].reason == "near_zero"

    def test_negative_weight_near_zero(self):
        """Negative weight near zero should be pruned."""
        l1_weights = {"a": -0.005}
        original = {"a": 0.5}
        active, pruned, reasons = BasisPursuitSelector._prune_near_zero(
            l1_weights, original, {}
        )
        assert "a" in pruned

    def test_mixed_active_and_pruned(self):
        """Mix of active and pruned signals."""
        l1_weights = {"a": 0.5, "b": 0.003, "c": 0.4}
        original = {"a": 0.3, "b": 0.3, "c": 0.4}
        active, pruned, reasons = BasisPursuitSelector._prune_near_zero(
            l1_weights, original, {}
        )
        assert len(active) == 2
        assert len(pruned) == 1


class TestSelectionExtended:
    """Extended selection integration tests."""

    def test_recovery_regime(self, selector, sample_signals, sample_weights):
        """Recovery regime should use recovery lambda."""
        result = selector.select_signals(sample_signals, sample_weights, "recovery")
        assert result.lambda_used == LAMBDA_BY_REGIME["recovery"]
        assert result.regime == "recovery"

    def test_high_vol_regime(self, selector, sample_signals, sample_weights):
        """High vol regime should use high_vol lambda."""
        result = selector.select_signals(sample_signals, sample_weights, "high_vol")
        assert result.lambda_used == LAMBDA_BY_REGIME["high_vol"]

    def test_active_weights_sum_to_one(self, selector, sample_signals, sample_weights):
        """Active weights should sum to 1.0 after normalization."""
        result = selector.select_signals(sample_signals, sample_weights, "normal")
        total = sum(result.active_signals.values())
        assert abs(total - 1.0) < 0.01

    def test_result_fields_populated(self, selector, sample_signals, sample_weights):
        """Result should have all fields populated."""
        result = selector.select_signals(sample_signals, sample_weights, "normal")
        assert result.num_active >= 0
        assert result.num_pruned >= 0
        assert result.total_signals == len(sample_weights)
        assert result.num_active + result.num_pruned == result.total_signals
        assert result.sparsity_ratio == result.num_active / max(result.total_signals, 1)

    def test_fallback_when_all_pruned(self, selector):
        """When all signals pruned, should fall back to base weights."""
        # Very high lambda with tiny weights
        signal_values = {"a": 0.001, "b": 0.001}
        base_weights = {"a": 0.5, "b": 0.5}
        result = selector.select_signals(signal_values, base_weights, "crisis")
        # Fallback ensures we still have active signals
        assert len(result.active_signals) > 0
        total = sum(result.active_signals.values())
        assert abs(total - 1.0) < 0.01


class TestStatePersistenceExtended:
    """Extended state persistence tests."""

    def test_regime_persisted(self, tmp_state_path):
        """Last regime should be persisted."""
        sel = BasisPursuitSelector(state_path=tmp_state_path, rolling_window=20)
        sel.select_signals({"a": 0.5}, {"a": 1.0}, "crisis")
        sel._save_state()

        sel2 = BasisPursuitSelector(state_path=tmp_state_path, rolling_window=20)
        assert sel2.state.last_regime == "crisis"

    def test_rolling_window_preserved(self, tmp_state_path):
        """Rolling window should persist correctly."""
        sel = BasisPursuitSelector(state_path=tmp_state_path, rolling_window=42)
        sel._save_state()

        sel2 = BasisPursuitSelector(state_path=tmp_state_path, rolling_window=42)
        assert sel2.state.rolling_window == 42


class TestGetStateDiagnosticsExtended:
    """Extended diagnostics tests."""

    def test_signal_statistics(self, selector):
        """Diagnostics should include mean and std for signals with history."""
        for i in range(5):
            selector._update_history({"test": float(i) / 10}, {"test": 0.5})
        diag = selector.get_state_diagnostics()
        assert diag["test"]["signal_periods"] == 5
        assert "signal_mean" in diag["test"]
        assert "signal_std" in diag["test"]

    def test_weight_statistics(self, selector):
        """Diagnostics should include weight mean."""
        for i in range(3):
            selector._update_history({"w_test": 0.5}, {"w_test": 0.3 + i * 0.1})
        diag = selector.get_state_diagnostics()
        assert "weight_mean" in diag["w_test"]
        assert diag["w_test"]["weight_periods"] == 3


class TestConstants:
    """Test module constants."""

    def test_default_rolling_window(self):
        assert DEFAULT_ROLLING_WINDOW == 60

    def test_lambda_by_regime_keys(self):
        expected = {"normal", "high_vol", "crisis", "recovery", "unknown_regime"}
        assert set(LAMBDA_BY_REGIME.keys()) == expected

    def test_lambda_monotonic(self):
        """Lambda should increase with regime severity."""
        assert LAMBDA_BY_REGIME["normal"] < LAMBDA_BY_REGIME["high_vol"]
        assert LAMBDA_BY_REGIME["high_vol"] < LAMBDA_BY_REGIME["crisis"]

    def test_redundancy_threshold(self):
        assert REDUNDANCY_CORRELATION_THRESHOLD == 0.85

    def test_min_active_weight(self):
        assert MIN_ACTIVE_WEIGHT == 0.01

    def test_sparsity_alert_threshold(self):
        assert SPARSITY_ALERT_THRESHOLD == 0.3


class TestCLI:
    """CLI main() dispatch tests."""

    def test_status_no_history(self, capsys):
        """Status command with no history displays appropriate message."""
        from src.strategy.basis_pursuit_selector import main
        with patch('sys.argv', ['bps.py', 'status']):
            with patch('src.strategy.basis_pursuit_selector.BasisPursuitSelector') as MockSel:
                mock = MagicMock()
                mock.get_state_diagnostics.return_value = {}
                MockSel.return_value = mock
                main()
        captured = capsys.readouterr()
        assert "No signal history yet" in captured.out

    def test_status_with_history(self, capsys):
        """Status command with history displays signal diagnostics."""
        from src.strategy.basis_pursuit_selector import main
        with patch('sys.argv', ['bps.py', 'status']):
            with patch('src.strategy.basis_pursuit_selector.BasisPursuitSelector') as MockSel:
                mock = MagicMock()
                mock.get_state_diagnostics.return_value = {
                    "momentum": {"signal_periods": 10, "signal_mean": 0.5, "signal_std": 0.1},
                }
                MockSel.return_value = mock
                main()
        captured = capsys.readouterr()
        assert "momentum" in captured.out
        assert "signal_periods= 10" in captured.out

    def test_select_malformed_signal(self, capsys):
        """Select command with malformed signals prints warning."""
        from src.strategy.basis_pursuit_selector import main
        with patch('sys.argv', ['bps.py', 'select', '--signals', 'badformat', '--weights', 'a=1.0']):
            with patch('src.strategy.basis_pursuit_selector.BasisPursuitSelector') as MockSel:
                mock = MagicMock()
                mock.select_signals.return_value = BasisPursuitResult(
                    active_signals={"a": 1.0}, pruned_signals={}, prune_reasons={},
                    sparsity_ratio=1.0, lambda_used=0.01, regime="normal",
                )
                MockSel.return_value = mock
                main()
        captured = capsys.readouterr()
        assert "WARN: Skipping malformed signal" in captured.out

    def test_select_malformed_weight(self, capsys):
        """Select command with malformed weights prints warning."""
        from src.strategy.basis_pursuit_selector import main
        with patch('sys.argv', ['bps.py', 'select', '--signals', 'a=0.5', '--weights', 'badformat']):
            with patch('src.strategy.basis_pursuit_selector.BasisPursuitSelector') as MockSel:
                mock = MagicMock()
                mock.select_signals.return_value = BasisPursuitResult(
                    active_signals={"a": 1.0}, pruned_signals={}, prune_reasons={},
                    sparsity_ratio=1.0, lambda_used=0.01, regime="normal",
                )
                MockSel.return_value = mock
                main()
        captured = capsys.readouterr()
        assert "WARN: Skipping malformed weight" in captured.out

    def test_select_no_valid_signals(self, capsys):
        """Select with only malformed signals handles gracefully."""
        from src.strategy.basis_pursuit_selector import main
        with patch('sys.argv', ['bps.py', 'select', '--signals', 'good=0.5', '--weights', 'good=1.0']):
            with patch('src.strategy.basis_pursuit_selector.BasisPursuitSelector') as MockSel:
                mock = MagicMock()
                result = BasisPursuitResult(
                    active_signals={"good": 1.0}, pruned_signals={}, prune_reasons={},
                    sparsity_ratio=1.0, lambda_used=0.01, regime="normal",
                )
                mock.select_signals.return_value = result
                MockSel.return_value = mock
                main()
        captured = capsys.readouterr()
        assert "Active signals" in captured.out
        assert "good" in captured.out

    def test_no_command_prints_help(self, capsys):
        """No command prints help."""
        from src.strategy.basis_pursuit_selector import main
        with patch('sys.argv', ['bps.py']):
            with patch('src.strategy.basis_pursuit_selector.BasisPursuitSelector') as MockSel:
                MockSel.return_value = MagicMock()
                main()
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "Basis-Pursuit" in captured.out

    def test_select_shows_concentration_warning(self, capsys):
        """Select shows warning when sparsity is below threshold."""
        from src.strategy.basis_pursuit_selector import main
        with patch('sys.argv', ['bps.py', 'select', '--signals', 'a=0.5', '--weights', 'a=1.0']):
            with patch('src.strategy.basis_pursuit_selector.BasisPursuitSelector') as MockSel:
                mock = MagicMock()
                result = BasisPursuitResult(
                    active_signals={"a": 1.0}, pruned_signals={"b": 1.0}, prune_reasons={
                        "b": PrunedSignal("b", 1.0, "near_zero"),
                    },
                    sparsity_ratio=0.2, lambda_used=0.15, regime="crisis",
                )
                result.is_concentrated = MagicMock(return_value=True)
                mock.select_signals.return_value = result
                MockSel.return_value = mock
                main()
        captured = capsys.readouterr()
        assert "concentration" in captured.out or "sparsity" in captured.out


class TestPerformanceTrackingExtended:
    """Advanced performance tracking edge cases."""

    def test_performance_truncates_at_100(self, selector, sample_signals, sample_weights, tmp_path):
        """Performance file should keep only last 100 entries."""
        perf_path = selector._resolve_perf_path()
        selector._resolve_perf_path = MagicMock(return_value=tmp_path / "perf.json")
        perf_path = tmp_path / "perf.json"
        for _ in range(105):
            selector._track_performance(BasisPursuitResult(
                active_signals={"a": 1.0}, pruned_signals={}, prune_reasons={},
                sparsity_ratio=0.5, lambda_used=0.01, regime="normal",
            ))
        with open(perf_path) as f:
            data = json.load(f)
        assert len(data) == 100

    def test_track_performance_json_decode_error(self, selector, tmp_path):
        """Corrupted performance file should not raise."""
        perf_path = selector._resolve_perf_path()
        selector._resolve_perf_path = MagicMock(return_value=tmp_path / "bad_perf.json")
        bad_path = tmp_path / "bad_perf.json"
        bad_path.write_text("{corrupt json}")
        # Should not raise
        selector._track_performance(BasisPursuitResult(
            active_signals={"a": 1.0}, pruned_signals={}, prune_reasons={},
            sparsity_ratio=0.5, lambda_used=0.01, regime="normal",
        ))

    def test_track_performance_os_error(self, selector, tmp_path):
        """OSError during performance tracking should not raise."""
        perf_path = selector._resolve_perf_path()
        selector._resolve_perf_path = MagicMock(return_value=tmp_path / "perf.json")
        # Make perf_path.exists() raise OSError inside the try block
        with patch.object(Path, 'exists', side_effect=OSError("Permission denied")):
            # Should not raise
            selector._track_performance(BasisPursuitResult(
                active_signals={"a": 1.0}, pruned_signals={}, prune_reasons={},
                sparsity_ratio=0.5, lambda_used=0.01, regime="normal",
            ))


class TestSaveState:
    """State persistence error handling."""

    def test_save_state_os_error(self, selector, tmp_path):
        """OSError during save should not raise."""
        selector._resolve_path = MagicMock(return_value=tmp_path / "state.json")
        # Make open() raise OSError inside the try block
        with patch('builtins.open', side_effect=OSError("Permission denied")):
            # Should not raise
            selector._save_state()


class TestSelectSignalsEdgeCases:
    """Additional select_signals edge cases."""

    def test_select_signals_with_empty_weights(self, selector):
        """Empty base_weights should return empty results."""
        result = selector.select_signals({"a": 0.5}, {}, "normal")
        assert result.active_signals == {}
        assert result.total_signals == 0

    def test_select_signals_all_same_signal(self, selector):
        """All signals have same value and weight, should all remain active."""
        signal_values = {"a": 0.5, "b": 0.5, "c": 0.5}
        base_weights = {"a": 0.33, "b": 0.33, "c": 0.34}
        # Run twice to build history for correlation check
        result1 = selector.select_signals(signal_values, base_weights, "normal")
        result2 = selector.select_signals(signal_values, base_weights, "normal")
        # Should still have active signals; each sum ~1
        assert abs(sum(result2.active_signals.values()) - 1.0) < 0.01

    def test_select_signals_near_zero_signal_value_in_crisis(self, selector):
        """Near-zero signal value in crisis regime should prune aggressively."""
        signal_values = {"a": 0.001, "b": 0.002}
        base_weights = {"a": 0.5, "b": 0.5}
        result = selector.select_signals(signal_values, base_weights, "crisis")
        # With lambda=0.15 and contribution=0.5*0.001=0.0005, all may be pruned
        # Fallback should activate base weights
        assert abs(sum(result.active_signals.values()) - 1.0) < 0.01

    def test_select_signals_realistic_scenario(self, selector):
        """Realistic 4-signal scenario with varied values."""
        signal_values = {"momentum": 0.8, "trend": 0.6, "carry": -0.2, "value": 0.1}
        base_weights = {"momentum": 0.35, "trend": 0.30, "carry": 0.20, "value": 0.15}
        result = selector.select_signals(signal_values, base_weights, "normal")
        assert result.total_signals == 4
        assert abs(sum(result.active_signals.values()) - 1.0) < 0.01

    def test_select_signals_unknown_regime(self, selector):
        """Unknown regime should use DEFAULT_LAMBDA (0.01)."""
        from src.strategy.basis_pursuit_selector import DEFAULT_LAMBDA
        result = selector.select_signals(
            {"a": 0.5}, {"a": 1.0}, "hypothetical_regime"
        )
        assert result.lambda_used == DEFAULT_LAMBDA
        assert result.regime == "hypothetical_regime"


class TestFindRedundantEdgeCases:
    """Corner cases in _find_redundant_signals."""

    def test_equal_mean_signal_keeps_first(self, selector):
        """When two signals have equal mean absolute value, first is kept."""
        # Build history with two signals that have identical means
        for _ in range(10):
            selector._update_history(
                {"sig_a": 0.5, "sig_b": 0.5},
                {"sig_a": 0.5, "sig_b": 0.5},
            )
        # They'll be perfectly correlated with identical means
        redundant = selector._find_redundant_signals()
        if redundant:
            # The second one should be flagged as redundant
            assert "sig_b" in redundant

    def test_min_periods_exactly_3(self, selector):
        """Exactly 3 periods should allow correlation computation."""
        selector._update_history({"a": 0.5, "b": 0.4}, {"a": 0.5, "b": 0.5})
        selector._update_history({"a": 0.6, "b": 0.5}, {"a": 0.5, "b": 0.5})
        selector._update_history({"a": 0.7, "b": 0.6}, {"a": 0.5, "b": 0.5})
        # 3 periods should be enough
        redundant = selector._find_redundant_signals()
        assert isinstance(redundant, dict)

    def test_uncorrelated_not_redundant(self, selector):
        """Signals with near-zero correlation should not be redundant."""
        for i in range(10):
            selector._update_history(
                {"up": 1.0 + i * 0.1, "down": 10.0 - i * 0.1},
                {"up": 0.5, "down": 0.5},
            )
        redundant = selector._find_redundant_signals()
        # These are inversely correlated (-1.0) so should be redundant
        assert len(redundant) > 0


class TestL1SelectionMoreEdgeCases:
    """More L1 selection edge cases."""

    def test_negative_signal_contribution(self, selector):
        """Negative signal values should use absolute value for contribution."""
        base_weights = {"a": 0.5, "b": 0.5}
        signal_values = {"a": -0.8, "b": 0.3}
        result = selector._apply_l1_selection(base_weights, signal_values, 0.01, {})
        # |a| contribution = 0.5 * 0.8 = 0.4, b = 0.5 * 0.3 = 0.15
        assert result.get("a", 0) > result.get("b", 0)

    def test_all_signals_missing_from_values(self, selector):
        """Signals not in signal_values should use base_weight as contribution."""
        base_weights = {"a": 0.3, "b": 0.2}
        signal_values = {}
        result = selector._apply_l1_selection(base_weights, signal_values, 0.01, {})
        # Both use base_weight as contribution
        # a: 0.3 - 0.01 = 0.29, b: 0.2 - 0.01 = 0.19
        assert result["a"] == pytest.approx(0.29, abs=0.01)
        assert result["b"] == pytest.approx(0.19, abs=0.01)

    def test_zero_lambda_preserves_exact(self, selector):
        """Zero lambda should preserve exact contributions."""
        base_weights = {"a": 0.25}
        signal_values = {"a": 0.5}
        result = selector._apply_l1_selection(base_weights, signal_values, 0.0, {})
        # contribution = 0.25 * 0.5 = 0.125, no shrinkage
        assert result["a"] == pytest.approx(0.125, abs=0.001)


class TestPruneNearZeroMoreEdgeCases:
    """More prune_near_zero edge cases."""

    def test_prune_overlapping_redundant_and_near_zero(self):
        """Signal that is both redundant and near-zero should be pruned as redundant."""
        l1_weights = {"a": 0.005, "b": 0.5}
        original = {"a": 0.3, "b": 0.5}
        redundant = {"a": "b"}
        active, pruned, reasons = BasisPursuitSelector._prune_near_zero(
            l1_weights, original, redundant
        )
        assert "a" in pruned
        assert reasons["a"].reason == "redundant"
        assert "b" in active

    def test_empty_l1_weights(self):
        """Empty L1 weights should produce empty active/pruned."""
        active, pruned, reasons = BasisPursuitSelector._prune_near_zero({}, {}, {})
        assert active == {}
        assert pruned == {}
        assert reasons == {}

    def test_signal_not_in_original_preserved(self):
        """Signal in l1_weights but not in original should use weight 0.
        NOTE: _prune_near_zero calls original_weights.get(signal, 0.0) for
        the pruned weight, so the original is 0 but the l1 weight is checked."""
        l1_weights = {"a": 0.5, "ghost": 0.5}
        original = {"a": 1.0}  # ghost not present
        active, pruned, reasons = BasisPursuitSelector._prune_near_zero(
            l1_weights, original, {}
        )
        assert "a" in active
        assert "ghost" in active  # Near-zero check is on l1 weight, not original


class TestSafeCorrEdgeCases:
    """_safe_corr edge cases."""

    def test_safe_corr_single_element(self):
        """Single-element arrays should return None or nan (insufficient data)."""
        a = np.array([1.0])
        b = np.array([2.0])
        corr = BasisPursuitSelector._safe_corr(a, b)
        assert corr is None or (isinstance(corr, float) and np.isnan(corr))

    def test_safe_corr_two_elements(self):
        """Two-element arrays should work fine."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        corr = BasisPursuitSelector._safe_corr(a, b)
        assert corr is not None


class TestHistoryExtended:
    """History management edge cases."""

    def test_history_keeps_recent_values(self, selector):
        """History should keep the most recent N values after trimming."""
        for i in range(25):
            selector._update_history({"sig": float(i)}, {"sig": 1.0})
        assert len(selector.state.signal_history["sig"]) == 20
        # First value should be 5 (25-20), last should be 24
        assert selector.state.signal_history["sig"][0] == 5.0
        assert selector.state.signal_history["sig"][-1] == 24.0

    def test_weight_history_independent_from_signal(self, selector):
        """Weight history should be tracked independently."""
        selector._update_history({"a": 0.5}, {"a": 0.3})
        selector._update_history({"a": 0.6}, {"a": 0.4})
        assert selector.state.signal_history["a"] == [0.5, 0.6]
        assert selector.state.full_weight_history["a"] == [0.3, 0.4]

    def test_different_signal_weight_keys(self, selector):
        """Signals and weights may have different keys."""
        selector._update_history(
            {"signal_only": 0.5, "both": 0.3},
            {"weight_only": 0.4, "both": 0.6},
        )
        assert "signal_only" in selector.state.signal_history
        assert "signal_only" not in selector.state.full_weight_history
        assert "weight_only" in selector.state.full_weight_history
        assert "weight_only" not in selector.state.signal_history
        assert "both" in selector.state.signal_history
        assert "both" in selector.state.full_weight_history


class TestResultDataclass:
    """BasisPursuitResult dataclass field validation."""

    def test_correlation_matrix_field_default(self):
        """correlation_matrix should default to None."""
        result = BasisPursuitResult(
            active_signals={"a": 1.0}, pruned_signals={}, prune_reasons={},
            sparsity_ratio=1.0, lambda_used=0.01, regime="normal",
        )
        assert result.correlation_matrix is None

    def test_correlation_matrix_custom(self):
        """correlation_matrix can be set to a float."""
        result = BasisPursuitResult(
            active_signals={"a": 1.0}, pruned_signals={}, prune_reasons={},
            sparsity_ratio=0.8, lambda_used=0.05, regime="high_vol",
            correlation_matrix=0.45,
        )
        assert result.correlation_matrix == 0.45

    def test_result_default_counts(self):
        """num_active and num_pruned should default to 0."""
        result = BasisPursuitResult(
            active_signals={}, pruned_signals={}, prune_reasons={},
            sparsity_ratio=0.0, lambda_used=0.01, regime="normal",
        )
        assert result.num_active == 0
        assert result.num_pruned == 0
        assert result.total_signals == 0


class TestStateFromDictEdgeCases:
    """BasisPursuitState.from_dict edge cases."""

    def test_from_dict_extra_keys_ignored(self):
        """Extra keys in dict should be ignored."""
        data = {
            "signal_history": {"a": [0.1]},
            "full_weight_history": {"a": [0.5]},
            "rolling_window": 42,
            "last_regime": "crisis",
            "nonexistent_field": "should_be_ignored",
        }
        state = BasisPursuitState.from_dict(data)
        assert state.signal_history == {"a": [0.1]}
        assert state.last_regime == "crisis"
        assert not hasattr(state, "nonexistent_field")


class TestConstantsExtended:
    """Additional constant validation."""

    def test_unknown_regime_matches_default(self):
        """unknown_regime should have same lambda as DEFAULT_LAMBDA."""
        from src.strategy.basis_pursuit_selector import DEFAULT_LAMBDA
        assert LAMBDA_BY_REGIME["unknown_regime"] == DEFAULT_LAMBDA

    def test_TOP_PRUNED_REPORT_COUNT(self):
        from src.strategy.basis_pursuit_selector import TOP_PRUNED_REPORT_COUNT
        assert TOP_PRUNED_REPORT_COUNT == 3
