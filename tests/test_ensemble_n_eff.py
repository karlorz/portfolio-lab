"""Tests for effective signal count (N_eff) and weight entropy in ensemble voter.

TDD red phase — defines behavior before implementation.
"""

import math
import pytest
import numpy as np


def _compute_n_eff(weights):
    """Compute effective number of signals: N_eff = exp(Shannon entropy).

    Args:
        weights: List/array of normalized weights (sum to 1).

    Returns:
        N_eff in [1, len(weights)]. 1 = concentrated, len(weights) = uniform.
    """
    w = np.array(weights, dtype=float)
    w = w[w > 0]  # Avoid log(0)
    entropy = -np.sum(w * np.log(w))
    return float(np.exp(entropy))


def _compute_weight_entropy(weights):
    """Compute Shannon entropy (nats) of weight distribution.

    Args:
        weights: List/array of normalized weights (sum to 1).

    Returns:
        Entropy in nats. 0 = concentrated, log(n) = uniform.
    """
    w = np.array(weights, dtype=float)
    w = w[w > 0]
    return float(-np.sum(w * np.log(w)))


class TestNEffComputation:
    """Test N_eff and entropy math."""

    def test_uniform_weights(self):
        """Uniform weights: N_eff = n (all signals equally weighted)."""
        weights = [0.25, 0.25, 0.25, 0.25]
        assert _compute_n_eff(weights) == pytest.approx(4.0, abs=0.01)

    def test_concentrated_weights(self):
        """Single dominant signal: N_eff ≈ 1."""
        weights = [0.99, 0.005, 0.005]
        assert _compute_n_eff(weights) < 1.1

    def test_two_signal_split(self):
        """Two equal signals: N_eff = 2."""
        weights = [0.5, 0.5]
        assert _compute_n_eff(weights) == pytest.approx(2.0, abs=0.01)

    def test_eight_signal_uniform(self):
        """8 uniform signals: N_eff = 8."""
        weights = [0.125] * 8
        assert _compute_n_eff(weights) == pytest.approx(8.0, abs=0.01)

    def test_realistic_ensemble(self):
        """Current ensemble weights: ALT_DATA 0.2245, INTL_MOM 0.2205, ..."""
        weights = [0.2245, 0.2205, 0.117, 0.117, 0.171, 0.10, 0.05]
        n_eff = _compute_n_eff(weights)
        assert 4 < n_eff < 7  # Should be between 1 and 7

    def test_entropy_zero_for_concentrated(self):
        """Single signal: entropy = 0."""
        assert _compute_weight_entropy([1.0]) == pytest.approx(0.0, abs=1e-6)

    def test_entropy_log_n_for_uniform(self):
        """Uniform n signals: entropy = log(n)."""
        n = 8
        weights = [1.0 / n] * n
        assert _compute_weight_entropy(weights) == pytest.approx(math.log(n), abs=0.01)

    def test_n_eff_range(self):
        """N_eff should always be in [1, len(weights)]."""
        for _ in range(20):
            raw = np.random.dirichlet(np.ones(8))
            n_eff = _compute_n_eff(raw)
            assert 1.0 <= n_eff <= 8.01

    def test_n_eff_monotonic(self):
        """More uniform weights → higher N_eff."""
        concentrated = [0.8, 0.1, 0.05, 0.05]
        balanced = [0.4, 0.3, 0.2, 0.1]
        uniform = [0.25, 0.25, 0.25, 0.25]
        assert _compute_n_eff(concentrated) < _compute_n_eff(balanced) < _compute_n_eff(uniform)

    def test_ensemble_vote_has_n_eff(self):
        """EnsembleVote dataclass should have n_eff and weight_entropy fields."""
        from src.strategy.ensemble_voter import EnsembleVote
        fields = [f.name for f in EnsembleVote.__dataclass_fields__.values()]
        assert "n_eff" in fields
        assert "weight_entropy" in fields


def test_n_eff_renormalizes_incomplete_mass():
    """Equal active weights with mass < 1 still yield n_eff ≈ k after renorm."""
    # Simulate 6 equal sources with total mass 0.386 (incomplete collection)
    raw = [0.386 / 6] * 6
    # Without renorm (bug): entropy uses unnormalized → understated n_eff
    w = np.array(raw)
    buggy = float(np.exp(-np.sum(w * np.log(w))))
    # With renorm (fix)
    w_norm = w / w.sum()
    fixed = float(np.exp(-np.sum(w_norm * np.log(w_norm))))
    assert fixed == pytest.approx(6.0, abs=0.05)
    assert buggy < 4.0  # documents the pre-fix understatement
    assert _compute_n_eff(list(w_norm)) == pytest.approx(6.0, abs=0.05)


def test_build_vote_n_eff_renorms_active_mass(tmp_path):
    """EnsembleVoter._build_vote reports n_eff on renormalized active weights."""
    from src.strategy.ensemble_voter import (
        EnsembleVoter,
        Regime,
        SignalReading,
        SignalSource,
    )
    from types import SimpleNamespace

    voter = EnsembleVoter(data_path=tmp_path)
    # three equal-mass readings summing to 0.3
    members = list(SignalSource)[:3]
    readings = [
        SignalReading(
            source=src,
            timestamp="2026-07-20T00:00:00",
            value=0.1,
            confidence=0.8,
            weight=0.1,
            regime_fit="normal",
        )
        for src in members
    ]
    consensus = SimpleNamespace(
        weighted_consensus=0.1,
        agreement=0.9,
        equity_bias=0.0,
        duration_bias=0.0,
        gold_bias=0.0,
        action="neutral",
        action_confidence=0.5,
    )
    vote = voter._build_vote(readings, consensus, Regime.NORMAL, 0.7)
    assert vote.n_eff == pytest.approx(3.0, abs=0.05)
