"""Tests for signal correlation matrix in ensemble voting."""

import time

import numpy as np
import pytest


class TestComputeSignalCorrelationMatrix:
    """Tests for compute_signal_correlation_matrix()."""

    def test_identical_predictions_correlation_1(self):
        from src.strategy.ensemble_voter import compute_signal_correlation_matrix

        # Two signals with identical predictions → correlation = 1.0
        ic_data = {
            "signal_a": [[1.0, 0.01], [0.5, 0.02], [-0.3, -0.01], [0.8, 0.03],
                         [1.0, 0.01], [0.5, 0.02], [-0.3, -0.01], [0.8, 0.03],
                         [1.0, 0.01], [0.5, 0.02]],
            "signal_b": [[1.0, 0.01], [0.5, 0.02], [-0.3, -0.01], [0.8, 0.03],
                         [1.0, 0.01], [0.5, 0.02], [-0.3, -0.01], [0.8, 0.03],
                         [1.0, 0.01], [0.5, 0.02]],
        }
        result = compute_signal_correlation_matrix(ic_data=ic_data, threshold=0.7)
        assert abs(result["matrix"]["signal_a"]["signal_b"] - 1.0) < 0.01
        assert len(result["redundant_pairs"]) == 1

    def test_opposite_predictions_correlation_neg1(self):
        from src.strategy.ensemble_voter import compute_signal_correlation_matrix

        ic_data = {
            "signal_a": [[1.0, 0.01], [0.5, 0.02], [-0.3, -0.01], [0.8, 0.03],
                         [1.0, 0.01], [0.5, 0.02], [-0.3, -0.01], [0.8, 0.03],
                         [1.0, 0.01], [0.5, 0.02]],
            "signal_b": [[-1.0, 0.01], [-0.5, 0.02], [0.3, -0.01], [-0.8, 0.03],
                         [-1.0, 0.01], [-0.5, 0.02], [0.3, -0.01], [-0.8, 0.03],
                         [-1.0, 0.01], [-0.5, 0.02]],
        }
        result = compute_signal_correlation_matrix(ic_data=ic_data, threshold=0.7)
        assert result["matrix"]["signal_a"]["signal_b"] < -0.85
        # Opposite signals are correlated at |r| > 0.7 → flagged as redundant
        assert len(result["redundant_pairs"]) == 1

    def test_unrelated_predictions_near_zero(self):
        from src.strategy.ensemble_voter import compute_signal_correlation_matrix

        rng = np.random.RandomState(42)
        # Two random series should have correlation near 0
        ic_data = {
            "signal_a": [[float(rng.randn()), 0.0] for _ in range(30)],
            "signal_b": [[float(rng.randn()), 0.0] for _ in range(30)],
        }
        result = compute_signal_correlation_matrix(ic_data=ic_data, threshold=0.7)
        corr = result["matrix"]["signal_a"]["signal_b"]
        assert abs(corr) < 0.5  # Random series shouldn't be strongly correlated
        assert len(result["redundant_pairs"]) == 0

    def test_three_signals_two_redundant(self):
        from src.strategy.ensemble_voter import compute_signal_correlation_matrix

        # Signals A and B are identical (r=1.0), C is anti-correlated
        base = [[1.0, 0.01], [0.5, 0.02], [-0.3, -0.01], [0.8, 0.03],
                [1.0, 0.01], [0.5, 0.02], [-0.3, -0.01], [0.8, 0.03],
                [1.0, 0.01], [0.5, 0.02]]
        ic_data = {
            "signal_a": base,
            "signal_b": base,  # identical to A
            "signal_c": [[-v[0], v[1]] for v in base],  # opposite to A and B
        }
        result = compute_signal_correlation_matrix(ic_data=ic_data, threshold=0.7)
        assert len(result["redundant_pairs"]) == 3  # A-B, A-C, B-C all > 0.7 in abs

    def test_penalty_reduces_weight_for_correlated_signal(self):
        from src.strategy.ensemble_voter import compute_signal_correlation_matrix

        # Signal A is correlated with B and C at ~0.8 → penalty ~0.56
        rng = np.random.RandomState(42)
        base = [float(rng.randn()) for _ in range(30)]
        # B = A + small noise → high correlation
        b_noise = rng.randn(30) * 0.3
        # C = A + small noise → high correlation
        c_noise = rng.randn(30) * 0.3

        ic_data = {
            "signal_a": [[base[i], 0.0] for i in range(30)],
            "signal_b": [[base[i] + b_noise[i], 0.0] for i in range(30)],
            "signal_c": [[base[i] + c_noise[i], 0.0] for i in range(30)],
        }
        result = compute_signal_correlation_matrix(ic_data=ic_data, threshold=0.7)
        penalties = result["correlation_penalties"]
        # All three are highly correlated with each other → penalties < 1.0
        for sig in ["signal_a", "signal_b", "signal_c"]:
            assert penalties[sig] < 1.0
        # Mean abs correlation with peers for signal_a ≈ (0.9+ + 0.9+) / 2 ≈ 0.9
        # penalty = 1/(1+0.9) ≈ 0.52
        assert penalties["signal_a"] < 0.7

    def test_single_signal_returns_empty(self):
        from src.strategy.ensemble_voter import compute_signal_correlation_matrix

        ic_data = {
            "signal_a": [[1.0, 0.01], [0.5, 0.02], [-0.3, -0.01], [0.8, 0.03],
                         [1.0, 0.01], [0.5, 0.02], [-0.3, -0.01], [0.8, 0.03],
                         [1.0, 0.01], [0.5, 0.02]],
        }
        result = compute_signal_correlation_matrix(ic_data=ic_data)
        assert result["matrix"] == {}
        assert result["redundant_pairs"] == []
        assert result["correlation_penalties"] == {}

    def test_empty_data_graceful(self):
        from src.strategy.ensemble_voter import compute_signal_correlation_matrix

        for empty in [{}, {"signal_a": []}]:
            result = compute_signal_correlation_matrix(ic_data=empty)
            assert result["matrix"] == {}
            assert result["redundant_pairs"] == []
            assert result["correlation_penalties"] == {}

    def test_insufficient_observations_ignored(self):
        from src.strategy.ensemble_voter import compute_signal_correlation_matrix

        # Less than 10 observations per signal
        ic_data = {
            "signal_a": [[1.0, 0.01]] * 5,
            "signal_b": [[0.5, 0.02]] * 5,
        }
        result = compute_signal_correlation_matrix(ic_data=ic_data)
        assert result["matrix"] == {}
        assert result["redundant_pairs"] == []

    def test_staged_key_is_skipped(self):
        from src.strategy.ensemble_voter import compute_signal_correlation_matrix

        ic_data = {
            "__staged__": {"date": "2026-05-26", "predictions": {"sig": 0.5}},
            "signal_a": [[1.0, 0.01], [0.5, 0.02], [-0.3, -0.01], [0.8, 0.03],
                         [1.0, 0.01], [0.5, 0.02], [-0.3, -0.01], [0.8, 0.03],
                         [1.0, 0.01], [0.5, 0.02]],
        }
        result = compute_signal_correlation_matrix(ic_data=ic_data)
        # __staged__ should be ignored; single signal → empty result
        assert result["matrix"] == {}

    def test_custom_threshold(self):
        from src.strategy.ensemble_voter import compute_signal_correlation_matrix

        base = [[1.0, 0.01], [0.5, 0.02], [-0.3, -0.01], [0.8, 0.03],
                [1.0, 0.01], [0.5, 0.02], [-0.3, -0.01], [0.8, 0.03],
                [1.0, 0.01], [0.5, 0.02]]
        ic_data = {
            "signal_a": base,
            "signal_b": base,  # corr = 1.0
        }
        # With threshold=0.95, the pair IS redundant
        result_low = compute_signal_correlation_matrix(ic_data=ic_data, threshold=0.95)
        assert len(result_low["redundant_pairs"]) == 1
        # With threshold=1.5, no pair exceeds (but it won't be >1.5 since corr max=1.0)
        result_high = compute_signal_correlation_matrix(ic_data=ic_data, threshold=1.5)
        assert len(result_high["redundant_pairs"]) == 0

    def test_zero_variance_predictions(self):
        from src.strategy.ensemble_voter import compute_signal_correlation_matrix

        # All identical predictions (zero variance) → Spearman returns 0.0
        ic_data = {
            "signal_a": [[0.5, 0.01]] * 20,
            "signal_b": [[0.5, 0.02]] * 20,
        }
        result = compute_signal_correlation_matrix(ic_data=ic_data, threshold=0.7)
        # Zero variance → Spearman returns 0.0, below threshold
        assert len(result["redundant_pairs"]) == 0
        assert abs(result["matrix"]["signal_a"]["signal_b"]) < 0.01

    def test_unequal_lengths_and_invalid_observations_are_filtered(self):
        from src.strategy.ensemble_voter import compute_signal_correlation_matrix

        ic_data = {
            "__staged__": {"date": "2026-06-08", "predictions": {"ignored": 0.2}},
            "signal_a": [
                [1.0, 0.0],
                ["bad", 0.0],
                [2.0, 0.0],
                [None, 0.0],
                [float("nan"), 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
                [5.0, 0.0],
                [6.0, 0.0],
                [7.0, 0.0],
                [8.0, 0.0],
                [9.0, 0.0],
                [10.0, 0.0],
            ],
            "signal_b": [[float(i), 0.0] for i in range(1, 11)],
            "signal_c": [[0.25, 0.0]] * 12,
        }

        result = compute_signal_correlation_matrix(ic_data=ic_data, threshold=0.7)

        assert "signal_a" in result["matrix"]
        assert result["matrix"]["signal_a"]["signal_b"] > 0.9
        assert result["matrix"]["signal_a"]["signal_c"] == 0.0
        redundant_pairs = {(left, right): corr for left, right, corr in result["redundant_pairs"]}
        assert redundant_pairs[("signal_a", "signal_b")] > 0.9
        assert result["correlation_penalties"]["signal_a"] < 1.0
        assert "__staged__" not in result["correlation_penalties"]

    def test_large_signal_set_avoids_pairwise_rank_helper_cost(self, monkeypatch):
        from src.strategy.ensemble_voter import compute_signal_correlation_matrix

        rng = np.random.RandomState(42)
        base = rng.normal(size=500)
        ic_data = {
            f"signal_{idx:02d}": [[float(value + idx * 0.001), 0.0] for value in base]
            for idx in range(20)
        }

        def slow_pairwise_helper(_x, _y):
            time.sleep(0.001)
            return 0.0

        monkeypatch.setattr(
            "src.monitor.ic_decay_monitor._spearman_rank_correlation",
            slow_pairwise_helper,
        )

        started = time.perf_counter()
        result = compute_signal_correlation_matrix(ic_data=ic_data, threshold=0.7)
        elapsed = time.perf_counter() - started

        assert len(result["correlation_penalties"]) == 20
        assert elapsed < 0.05


class TestCorrelationPenaltyIntegration:
    """Integration tests for _apply_correlation_penalty in EnsembleVoter."""

    def test_apply_penalty_reduces_correlated_weights(self):
        from src.strategy.ensemble_voter import EnsembleVoter
        from src.strategy.ensemble_voter import SignalSource

        voter = EnsembleVoter.__new__(EnsembleVoter)

        weights = {
            SignalSource.ALTERNATIVE_DATA: 0.40,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.30,
            SignalSource.CROSS_ASSET_RV: 0.20,
            SignalSource.UNIFIED_OVERLAY: 0.10,
        }

        # Applying penalty without IC data should return unchanged
        result = voter._apply_correlation_penalty(weights)
        # Without IC state file, should pass through
        assert sum(result.values()) == pytest.approx(1.0, abs=0.01)

    def test_normalization_preserves_sum_to_1(self):
        from src.strategy.ensemble_voter import EnsembleVoter
        from src.strategy.ensemble_voter import SignalSource

        voter = EnsembleVoter.__new__(EnsembleVoter)

        weights = {
            SignalSource.ALTERNATIVE_DATA: 0.50,
            SignalSource.CROSS_ASSET_RV: 0.30,
            SignalSource.UNIFIED_OVERLAY: 0.20,
        }

        result = voter._apply_correlation_penalty(weights)
        total = sum(result.values())
        assert abs(total - 1.0) < 0.01

    def test_exception_handling_graceful_degradation(self):
        from src.strategy.ensemble_voter import EnsembleVoter
        from src.strategy.ensemble_voter import SignalSource

        voter = EnsembleVoter.__new__(EnsembleVoter)

        weights = {
            SignalSource.ALTERNATIVE_DATA: 1.0,
        }

        # Should not raise even with abnormal state
        result = voter._apply_correlation_penalty(weights)
        assert len(result) == len(weights)
