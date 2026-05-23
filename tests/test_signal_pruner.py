"""Tests for signal pruning via correlation clustering and backward elimination."""
import numpy as np
import pytest
from src.signals.signal_pruner import correlation_cluster, backward_eliminate


class TestCorrelationCluster:
    def test_returns_list_of_clusters(self):
        returns = {
            "sig_a": np.array([0.01, 0.02, -0.01, 0.005, 0.01]),
            "sig_b": np.array([0.01, 0.02, -0.01, 0.005, 0.01]),  # perfect corr with a
            "sig_c": np.array([-0.01, -0.02, 0.01, -0.005, -0.01]),  # neg corr with a
        }
        sharpe = {"sig_a": 0.5, "sig_b": 0.4, "sig_c": 0.3}
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        assert isinstance(result, list)
        # sig_a and sig_b are perfectly correlated -> only one survives as cluster rep
        # sig_c is negatively correlated -> separate cluster
        assert len(result) == 2

    def test_keeps_highest_sharpe_per_cluster(self):
        returns = {
            "high": np.array([0.01, 0.02, 0.01]),
            "low": np.array([0.01, 0.02, 0.01]),   # identical to high
        }
        sharpe = {"high": 0.8, "low": 0.3}
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        survivors = [r[0] for r in result]  # first element of each cluster is best
        assert "high" in survivors
        assert "low" not in survivors

    def test_uncorrelated_signals_all_survive(self):
        n = 10
        np.random.seed(42)
        returns = {f"sig_{i}": np.random.randn(252) for i in range(n)}
        sharpe = {f"sig_{i}": np.random.uniform(0.1, 0.5) for i in range(n)}
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        # With random data, most signals should survive as separate clusters
        assert len(result) >= 5

    def test_empty_input(self):
        result = correlation_cluster({}, {}, threshold=0.6)
        assert result == []

    def test_single_signal(self):
        returns = {"only": np.array([0.01, 0.02, -0.01])}
        sharpe = {"only": 0.5}
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        assert result == [["only"]]

    def test_different_length_returns_raises(self):
        returns = {
            "a": np.array([0.01, 0.02]),
            "b": np.array([0.01, 0.02, 0.03]),
        }
        sharpe = {"a": 0.5, "b": 0.4}
        with pytest.raises(ValueError, match="same length"):
            correlation_cluster(returns, sharpe, threshold=0.6)

    def test_threshold_zero_means_cluster_identical(self):
        returns = {
            "a": np.array([0.01, 0.02, 0.01]),
            "b": np.array([0.01, 0.02, 0.01]),  # identical
        }
        sharpe = {"a": 0.5, "b": 0.4}
        result = correlation_cluster(returns, sharpe, threshold=0.0)
        # Threshold 0 means anything with corr > 0 is clustered -> identical corr=1.0 > 0
        assert len(result) == 1

    def test_threshold_one_means_no_clustering(self):
        returns = {
            "a": np.array([0.01, 0.02, 0.01]),
            "b": np.array([0.01, 0.02, 0.01]),  # identical
        }
        sharpe = {"a": 0.5, "b": 0.4}
        result = correlation_cluster(returns, sharpe, threshold=1.0)
        # Threshold 1.0 means only corr > 1.0 triggers clustering -> nothing clusters
        assert len(result) == 2


class TestBackwardEliminate:
    def test_eliminates_weakest_signals(self):
        # 3 signals: strong (positive mean), medium (smaller positive), noise (zero mean)
        rng = np.random.RandomState(42)
        returns = {
            "strong": rng.randn(252) * 0.01 + 0.001,
            "medium": rng.randn(252) * 0.01 + 0.0005,
            "noise": rng.randn(252) * 0.01,
        }
        result = backward_eliminate(returns, target_n=2)
        assert len(result) == 2
        assert "noise" not in result

    def test_target_n_equals_input_returns_all(self):
        rng = np.random.RandomState(42)
        returns = {f"s{i}": rng.randn(252) for i in range(3)}
        result = backward_eliminate(returns, target_n=3)
        assert len(result) == 3

    def test_target_n_greater_than_input_returns_all(self):
        rng = np.random.RandomState(42)
        returns = {f"s{i}": rng.randn(252) for i in range(3)}
        result = backward_eliminate(returns, target_n=10)
        assert len(result) == 3

    def test_empty_returns(self):
        result = backward_eliminate({}, target_n=5)
        assert result == []

    def test_single_signal_returns_it(self):
        rng = np.random.RandomState(42)
        returns = {"only": rng.randn(252)}
        result = backward_eliminate(returns, target_n=5)
        assert result == ["only"]

    def test_elimination_reduces_count(self):
        np.random.seed(42)
        returns = {f"s{i}": np.random.randn(252) * 0.001 + np.random.uniform(-0.0002, 0.0005)
                   for i in range(8)}
        result = backward_eliminate(returns, target_n=5)
        assert len(result) == 5
