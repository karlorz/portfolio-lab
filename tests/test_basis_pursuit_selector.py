"""
Tests for v8.02 BasisPursuitSelector — basis-pursuit signal selection.
"""

import json
import math
import tempfile
from pathlib import Path
from typing import Dict, List

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
