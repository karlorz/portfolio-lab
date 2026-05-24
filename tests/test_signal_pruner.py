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


class TestCorrelationClusterExtended:
    """Extended edge cases for correlation clustering."""

    def test_negative_correlation_separate_clusters(self):
        """Negatively correlated signals should NOT be clustered together."""
        returns = {
            "pos": np.array([0.01, 0.02, -0.01, 0.005, 0.01]),
            "neg": np.array([-0.01, -0.02, 0.01, -0.005, -0.01]),  # perfect neg corr
        }
        sharpe = {"pos": 0.5, "neg": 0.3}
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        # Negatively correlated → separate clusters (diversifying, not redundant)
        assert len(result) == 2

    def test_constant_returns_nan_corr_handled(self):
        """Constant returns produce NaN correlation, should be handled as 0."""
        returns = {
            "const": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),  # zero variance
            "var": np.array([0.01, -0.01, 0.02, -0.005, 0.01]),
        }
        sharpe = {"const": 0.0, "var": 0.5}
        # Should not crash — NaN corr treated as 0
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        assert len(result) == 2  # Not correlated (NaN → 0), separate clusters

    def test_cluster_order_by_sharpe(self):
        """Within each cluster, highest Sharpe signal should be first."""
        returns = {
            "low": np.array([0.01, 0.02, 0.01]),
            "mid": np.array([0.01, 0.02, 0.01]),
            "high": np.array([0.01, 0.02, 0.01]),
        }
        sharpe = {"low": 0.1, "mid": 0.5, "high": 0.9}
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        # All three in one cluster (identical), highest Sharpe first
        assert len(result) == 1
        assert result[0][0] == "high"

    def test_threshold_near_one_keeps_most(self):
        """High threshold means only very highly correlated signals cluster."""
        rng = np.random.RandomState(42)
        returns = {f"s{i}": rng.randn(100) for i in range(5)}
        sharpe = {f"s{i}": 0.3 for i in range(5)}
        result_low = correlation_cluster(returns, sharpe, threshold=0.1)
        result_high = correlation_cluster(returns, sharpe, threshold=0.9)
        # Higher threshold → more clusters (less aggressive grouping)
        assert len(result_high) >= len(result_low)

    def test_three_way_cluster(self):
        """Three mutually correlated signals should end up in one cluster."""
        base = np.array([0.01, 0.02, -0.01, 0.005, 0.01])
        returns = {
            "a": base,
            "b": base + np.array([0.001, -0.001, 0.001, -0.001, 0.001]),  # near-identical
            "c": base + np.array([0.002, -0.002, 0.002, -0.002, 0.002]),  # near-identical
        }
        sharpe = {"a": 0.5, "b": 0.4, "c": 0.3}
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        assert len(result) == 1
        assert result[0][0] == "a"  # Highest Sharpe first


class TestBackwardEliminateExtended:
    """Extended edge cases for backward elimination."""

    def test_target_n_one(self):
        """Should keep only the best single signal."""
        rng = np.random.RandomState(42)
        returns = {
            "strong": rng.randn(252) * 0.01 + 0.002,
            "medium": rng.randn(252) * 0.01 + 0.001,
            "weak": rng.randn(252) * 0.01 - 0.001,
        }
        result = backward_eliminate(returns, target_n=1)
        assert len(result) == 1
        assert "strong" in result

    def test_constant_returns_handled(self):
        """Constant (zero-variance) returns should not crash ensemble_sharpe."""
        returns = {
            "const": np.zeros(252),
            "var": np.random.RandomState(42).randn(252) * 0.01,
        }
        result = backward_eliminate(returns, target_n=1)
        assert len(result) == 1  # Should not crash

    def test_preserves_signal_names(self):
        """Result should contain actual signal names from input."""
        rng = np.random.RandomState(42)
        returns = {"alpha": rng.randn(252), "beta": rng.randn(252)}
        result = backward_eliminate(returns, target_n=2)
        assert set(result) == {"alpha", "beta"}

    def test_target_n_zero_minimal_result(self):
        """target_n=0 — backward elimination can't go below 1 signal
        (removing the last signal gives empty list with -inf Sharpe)."""
        rng = np.random.RandomState(42)
        returns = {"a": rng.randn(252), "b": rng.randn(252)}
        result = backward_eliminate(returns, target_n=0)
        assert len(result) >= 1  # Can't eliminate to 0

    def test_deterministic_results(self):
        """Same input should always produce same output."""
        rng = np.random.RandomState(42)
        returns = {f"s{i}": rng.randn(252) for i in range(6)}
        result1 = backward_eliminate(returns, target_n=3)
        result2 = backward_eliminate(returns, target_n=3)
        assert result1 == result2


class TestCorrelationClusterExtended2:
    """Additional edge cases for correlation clustering — untested paths."""

    def test_missing_sharpe_key_uses_zero_default(self):
        """Signal not in sharpe dict gets Sharpe=0.0 default, sorted last."""
        returns = {
            "high": np.array([0.01, 0.02, -0.01]),
            "mid": np.array([0.01, 0.02, -0.01]),
            "orphan": np.array([0.01, 0.02, -0.01]),
        }
        sharpe = {"high": 0.8, "mid": 0.4}  # "orphan" missing
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        # All three identical, one cluster. Highest Sharpe should be first.
        assert len(result) == 1
        # "high" (0.8) sorts first, "mid" (0.4) second, "orphan" (0.0 default) last
        assert result[0] == ["high", "mid", "orphan"]

    def test_all_signals_constant_returns(self):
        """All signals have constant returns — handled without crash
        (FP precision may yield tiny non-zero std for non-zeros vals)."""
        returns = {
            "a": np.ones(10) * 0.01,
            "b": np.ones(10) * 0.02,
            "c": np.ones(10) * 0.03,
        }
        sharpe = {"a": 0.5, "b": 0.4, "c": 0.3}
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        assert isinstance(result, list)
        # All signals accounted for regardless of clustering pattern
        all_signals = {sig for c in result for sig in c}
        assert all_signals == {"a", "b", "c"}
        assert len(result) >= 1

    def test_sharpe_ties_alphabetical_sort(self):
        """Tied Sharpe values break alphabetically via stable sort."""
        rng = np.random.RandomState(42)
        returns = {
            "c": rng.randn(252),
            "a": rng.randn(252),
            "b": rng.randn(252),
        }
        sharpe = {"a": 0.3, "b": 0.3, "c": 0.3}
        result = correlation_cluster(returns, sharpe, threshold=0.9)
        # With high threshold, random data is very unlikely to cluster
        assert len(result) == 3
        reps = [c[0] for c in result]
        assert reps == ["a", "b", "c"]

    def test_negative_sharpe_values_handled(self):
        """Negative Sharpe values sort appropriately (worse than positive)."""
        returns = {
            "good": np.array([0.01, 0.02, 0.01]),
            "bad": np.array([0.01, 0.02, 0.01]),
            "worst": np.array([0.01, 0.02, 0.01]),
        }
        sharpe = {"good": 0.6, "bad": -0.2, "worst": -0.5}
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        # All identical, one cluster. Order by Sharpe descending.
        assert len(result) == 1
        assert result[0][0] == "good"

    def test_all_sharpe_values_zero(self):
        """All Sharpe values are 0.0 — still clusters normally."""
        returns = {
            "a": np.array([0.01, 0.02, 0.01]),
            "b": np.array([0.01, 0.02, 0.01]),
        }
        sharpe = {"a": 0.0, "b": 0.0}
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        assert len(result) == 1

    def test_input_dicts_not_mutated(self):
        """Function should not mutate input dicts or arrays."""
        returns = {
            "a": np.array([0.01, 0.02, -0.01]),
            "b": np.array([0.01, 0.02, -0.01]),
        }
        sharpe = {"a": 0.5, "b": 0.4}
        orig_returns = {k: v.copy() for k, v in returns.items()}
        orig_sharpe = dict(sharpe)
        correlation_cluster(returns, sharpe, threshold=0.6)
        for k in returns:
            np.testing.assert_array_equal(returns[k], orig_returns[k])
        assert sharpe == orig_sharpe

    def test_correlation_exactly_threshold_not_clustered(self):
        """When corr exactly equals threshold, strict > check prevents clustering."""
        a = np.array([0.01, 0.02, -0.01, 0.005, 0.01])
        b = np.array([0.015, 0.01, 0.0, 0.01, 0.005])
        returns = {"a": a, "b": b}
        data = np.column_stack([a, b])
        corr = float(np.corrcoef(data, rowvar=False)[0, 1])
        corr = np.nan_to_num(corr, nan=0.0)
        sharpe = {"a": 0.5, "b": 0.4}
        result = correlation_cluster(returns, sharpe, threshold=corr)
        # threshold == corr, but condition is strict > → not clustered
        assert len(result) == 2

    def test_cluster_members_correct_content(self):
        """Verify cluster membership contains correct signal names."""
        base = np.array([0.01, 0.02, -0.01, 0.005, 0.01])
        returns = {
            "x": base,
            "y": base + 0.0005,
            "z": np.array([-0.01, -0.02, 0.01, -0.005, -0.01]),
        }
        sharpe = {"x": 0.7, "y": 0.5, "z": 0.3}
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        # x and y are near-identical → same cluster; z is neg-corr → separate
        assert len(result) == 2
        cluster_reps = {c[0] for c in result}
        assert "x" in cluster_reps
        assert "z" in cluster_reps

    def test_many_signals_with_redundancy_pattern(self):
        """20 signals: 4 groups of 5 identical signals."""
        rng = np.random.RandomState(42)
        returns = {}
        sharpe = {}
        for group in range(4):
            base = rng.randn(100)
            for member in range(5):
                name = f"g{group}_m{member}"
                returns[name] = base + rng.randn(100) * 0.01
                sharpe[name] = 0.5 - group * 0.1
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        # Expect 4 clusters (one per group), each starting with highest Sharpe
        assert len(result) == 4
        for cluster in result:
            assert len(cluster) == 5

    def test_zero_returns_handled(self):
        """All zeros returns — no crash, nan_to_num handles NaN."""
        returns = {
            "a": np.zeros(10),
            "b": np.zeros(10),
        }
        sharpe = {"a": 0.3, "b": 0.2}
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        assert len(result) == 2  # NaN → 0, none > 0.6

    def test_cluster_representative_is_correct(self):
        """First element of each cluster is its highest-Sharpe signal."""
        rng = np.random.RandomState(42)
        base = rng.randn(50)
        returns = {
            "low": base + rng.randn(50) * 0.01,
            "high": base + rng.randn(50) * 0.01,
        }
        sharpe = {"low": 0.2, "high": 0.9}
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        assert len(result) == 1
        assert result[0][0] == "high"  # highest Sharpe is representative

    def test_highly_precise_returns_stable(self):
        """Very small return values should not cause numerical issues."""
        rng = np.random.RandomState(42)
        returns = {
            "a": rng.randn(252) * 1e-8,
            "b": rng.randn(252) * 1e-8,
        }
        sharpe = {"a": 0.5, "b": 0.4}
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        assert isinstance(result, list)

    def test_only_positive_correlation_clusters(self):
        """Only positively correlated signals should cluster (diversifying kept)."""
        base = np.array([0.01, 0.02, -0.01, 0.005, 0.01])
        returns = {
            "a": base,
            "b": base,  # perfect positive corr with a
            "c": -base,  # perfect negative corr with a
        }
        sharpe = {"a": 0.6, "b": 0.4, "c": 0.3}
        result = correlation_cluster(returns, sharpe, threshold=0.6)
        # a and b in one cluster, c in separate (negatively correlated)
        assert len(result) == 2
        assert "c" in {c[0] for c in result}


class TestBackwardEliminateExtended2:
    """Additional edge cases for backward elimination — untested paths."""

    def test_all_return_series_identical(self):
        """All signals have identical returns — algorithm still converges."""
        rng = np.random.RandomState(42)
        base = rng.randn(252)
        returns = {"a": base.copy(), "b": base.copy(), "c": base.copy()}
        result = backward_eliminate(returns, target_n=1)
        assert len(result) == 1

    def test_nan_in_returns_not_crash(self):
        """NaN values in returns should not crash — handled gracefully."""
        rng = np.random.RandomState(42)
        arr = rng.randn(252)
        arr[10] = np.nan
        returns = {"a": arr, "b": rng.randn(252)}
        result = backward_eliminate(returns, target_n=1)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_returns_with_very_small_std(self):
        """Returns with std near 1e-11 hits the <1e-10 guard, returns 0 Sharpe."""
        rng = np.random.RandomState(42)
        a = rng.randn(252) * 1e-11 + 0.001  # std ~1e-11, below epsilon guard
        b = rng.randn(252) * 0.01 + 0.0005  # normal variance, positive mean
        returns = {"a": a, "b": b}
        result = backward_eliminate(returns, target_n=1)
        assert isinstance(result, list)
        assert len(result) == 1  # Converges without crash

    def test_negative_mean_returns_handled(self):
        """All return series have negative mean — still converges."""
        rng = np.random.RandomState(42)
        returns = {
            "a": rng.randn(252) * 0.01 - 0.002,
            "b": rng.randn(252) * 0.01 - 0.001,
            "c": rng.randn(252) * 0.01 - 0.003,
        }
        result = backward_eliminate(returns, target_n=1)
        assert len(result) == 1

    def test_input_dict_is_not_mutated(self):
        """Function should not mutate the input returns dict."""
        rng = np.random.RandomState(42)
        returns = {
            "a": rng.randn(252),
            "b": rng.randn(252),
            "c": rng.randn(252),
        }
        orig_returns = {k: v.copy() for k, v in returns.items()}
        backward_eliminate(returns, target_n=1)
        for k in returns:
            np.testing.assert_array_equal(returns[k], orig_returns[k])

    def test_large_number_of_signals(self):
        """Backward elimination handles 12 signals without issue."""
        rng = np.random.RandomState(42)
        returns = {f"s{i}": rng.randn(252) * 0.01 + rng.uniform(-0.001, 0.002)
                   for i in range(12)}
        result = backward_eliminate(returns, target_n=4)
        assert len(result) == 4

    def test_all_zero_returns_not_crash(self):
        """All zero returns should not crash (zero variance path)."""
        returns = {"a": np.zeros(252), "b": np.zeros(252), "c": np.zeros(252)}
        result = backward_eliminate(returns, target_n=1)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_mixed_zero_and_nonzero_variance(self):
        """Mix of zero-variance and normal-variance signals."""
        rng = np.random.RandomState(42)
        returns = {
            "const": np.zeros(252),
            "var": rng.randn(252) * 0.01 + 0.0005,
        }
        result = backward_eliminate(returns, target_n=1)
        assert len(result) == 1

    def test_target_n_zero_with_many_signals(self):
        """target_n=0 with many signals still returns at least 1 signal."""
        rng = np.random.RandomState(42)
        returns = {f"s{i}": rng.randn(252) for i in range(6)}
        result = backward_eliminate(returns, target_n=0)
        assert len(result) >= 1

    def test_elimination_of_noise_signals(self):
        """Signals with near-zero mean are eliminated before strong signals."""
        rng = np.random.RandomState(42)
        returns = {
            "strong": rng.randn(252) * 0.01 + 0.003,
            "medium": rng.randn(252) * 0.01 + 0.001,
            "noise_a": rng.randn(252) * 0.01,
            "noise_b": rng.randn(252) * 0.01,
        }
        result = backward_eliminate(returns, target_n=2)
        assert len(result) == 2
        assert "strong" in result

    def test_elimination_with_equal_sharpe_candidates(self):
        """When all signals have similar quality, elimination still converges."""
        rng = np.random.RandomState(42)
        returns = {f"s{i}": rng.randn(252) * 0.01 + 0.0001
                   for i in range(7)}
        result = backward_eliminate(returns, target_n=3)
        assert len(result) == 3

    def test_ensemble_sharpe_zero_variance_edge(self):
        """Ensemble Sharpe returns 0.0 when post-removal ensemble has zero variance."""
        rng = np.random.RandomState(42)
        # Only signal "b" has real variance; removing any other gives zero-var ensemble
        const = np.ones(252) * 0.001
        var = rng.randn(252) * 0.01 + 0.0005
        returns = {"const_a": const, "const_b": const.copy(), "var": var}
        result = backward_eliminate(returns, target_n=1)
        assert len(result) == 1
        # The variable signal should survive
        assert "var" in result

    def test_signal_count_preserved_after_elimination(self):
        """Exact count from target_n is preserved when sufficient signals exist."""
        rng = np.random.RandomState(42)
        returns = {f"s{i}": rng.randn(252) * 0.01 + rng.uniform(-0.001, 0.003)
                   for i in range(10)}
        for target in [2, 3, 4, 5, 6]:
            result = backward_eliminate(returns, target_n=target)
            assert len(result) == target

    def test_returns_with_inf_values_not_crash(self):
        """Infinity in returns should not crash."""
        rng = np.random.RandomState(42)
        arr = rng.randn(252)
        arr[0] = np.inf
        returns = {"inf_sig": arr, "normal": rng.randn(252)}
        result = backward_eliminate(returns, target_n=1)
        assert isinstance(result, list)
        assert len(result) >= 1
