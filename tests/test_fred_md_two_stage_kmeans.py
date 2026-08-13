#!/usr/bin/env python3
"""Tests for src/regime/fred_md_two_stage_kmeans.py — numpy-only two-stage k-means."""

import numpy as np
import pytest


def _make_synthetic_data(
    n_samples: int = 500,
    n_features: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic PCA-reduced FRED-MD-like data with known structure.

    Creates data with 2 regimes:
    - ~5% outlier samples (crisis): large magnitude, distinctive direction
    - ~95% normal samples: 5 clusters with different directional patterns
    """
    rng = np.random.default_rng(seed)
    n_crisis = max(5, n_samples // 20)
    n_normal = n_samples - n_crisis

    # Crisis samples: large magnitude, one direction
    crisis = rng.normal(loc=3.0, scale=1.5, size=(n_crisis, n_features))

    # Normal samples: 5 clusters with different directions
    # Each cluster has a different "loading" on specific feature groups
    normal = np.zeros((n_normal, n_features))
    cluster_size = n_normal // 5
    directions = [
        [1.0, 0.0, 0.0, 0.0, 0.0],  # Cluster 0: strong on feature group 0
        [0.0, 1.0, 0.0, 0.0, 0.0],  # Cluster 1: strong on feature group 1
        [0.0, 0.0, 1.0, 0.0, 0.0],  # Cluster 2
        [0.0, 0.0, 0.0, 1.0, 0.0],  # Cluster 3
        [0.0, 0.0, 0.0, 0.0, 1.0],  # Cluster 4
    ]
    for j in range(5):
        start = j * cluster_size
        end = ((j + 1) * cluster_size) if j < 4 else n_normal
        n_j = end - start
        base = rng.normal(loc=0.0, scale=0.5, size=(n_j, n_features))
        # Add directional bias: features j*4:(j+1)*4 get positive loading
        feat_group = n_features // 5
        for f in range(feat_group):
            base[:, j * feat_group + f] += directions[j][0] * 2.0
        normal[start:end] = base

    X = np.vstack([crisis, normal])
    return X


class TestPCANumpy:
    """Tests for pca_numpy function."""

    def test_basic_pca(self):
        from src.regime.fred_md_two_stage_kmeans import pca_numpy

        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 10))
        Z, n_comp, var_ret, components = pca_numpy(X, variance_threshold=0.95)

        assert Z.shape[0] == 100
        assert 1 <= n_comp <= 10
        assert 0 < var_ret <= 1.0
        assert components.shape[0] == n_comp
        assert components.shape[1] == 10

    def test_variance_threshold_respected(self):
        from src.regime.fred_md_two_stage_kmeans import pca_numpy

        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 30))
        _, n_comp, var_ret, _ = pca_numpy(X, variance_threshold=0.95)
        assert var_ret >= 0.94  # close to target

    def test_zero_variance_handled(self):
        from src.regime.fred_md_two_stage_kmeans import pca_numpy

        X = np.ones((50, 5))
        Z, n_comp, var_ret, components = pca_numpy(X)
        assert Z.shape[0] == 50

    def test_max_components_cap(self):
        from src.regime.fred_md_two_stage_kmeans import pca_numpy

        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 20))
        _, n_comp, _, _ = pca_numpy(X, variance_threshold=0.99, max_components=3)
        assert n_comp <= 3


class TestKMeansL2:
    """Tests for kmeans_l2 function."""

    def test_basic_clustering(self):
        from src.regime.fred_md_two_stage_kmeans import kmeans_l2

        rng = np.random.default_rng(42)
        # Two well-separated clusters
        X1 = rng.normal(loc=[-5, -5], scale=1.0, size=(50, 2))
        X2 = rng.normal(loc=[5, 5], scale=1.0, size=(50, 2))
        X = np.vstack([X1, X2])

        labels, centroids, inertia = kmeans_l2(X, k=2)
        assert len(labels) == 100
        assert centroids.shape == (2, 2)
        assert inertia > 0

        # Check cluster separation: one centroid near (-5,-5), other near (5,5)
        c_means = centroids.mean(axis=0)
        assert abs(c_means[0]) < 2  # centroids should be on opposite sides, roughly centered

    def test_deterministic_with_seed(self):
        from src.regime.fred_md_two_stage_kmeans import kmeans_l2

        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 5))

        labels1, _, _ = kmeans_l2(X, k=3, seed=42)
        labels2, _, _ = kmeans_l2(X, k=3, seed=42)
        assert np.array_equal(labels1, labels2)

    def test_all_points_assigned(self):
        from src.regime.fred_md_two_stage_kmeans import kmeans_l2

        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 3))

        labels, _, _ = kmeans_l2(X, k=4, seed=42)
        assert len(labels) == 30
        assert set(labels) <= {0, 1, 2, 3}


class TestKMeansCosine:
    """Tests for kmeans_cosine function."""

    def test_directional_clustering(self):
        from src.regime.fred_md_two_stage_kmeans import kmeans_cosine

        rng = np.random.default_rng(42)
        # Two clusters with opposite directions but same magnitude
        dir1 = np.array([1.0, 0.0])
        dir2 = np.array([-1.0, 0.0])

        X1 = np.tile(dir1, (30, 1)) + rng.normal(scale=0.1, size=(30, 2))
        X2 = np.tile(dir2, (30, 1)) + rng.normal(scale=0.1, size=(30, 2))
        X = np.vstack([X1, X2])

        labels, centroids, _ = kmeans_cosine(X, k=2, seed=42)

        # Should find two clusters based on direction
        n_unique = len(set(labels))
        assert n_unique == 2

    def test_normalizes_inputs(self):
        from src.regime.fred_md_two_stage_kmeans import kmeans_cosine

        rng = np.random.default_rng(42)
        # Large magnitude differences should be ignored by cosine
        X1 = rng.normal(loc=[5, 0], scale=0.1, size=(20, 2))   # magnitude ~5
        X2 = rng.normal(loc=[0.1, 0], scale=0.01, size=(20, 2))  # magnitude ~0.1
        X = np.vstack([X1, X2])

        labels, _, _ = kmeans_cosine(X, k=2, seed=42)
        # Both point in same direction [1,0], should cluster together
        # regardless of magnitude (cosine distance ignores magnitude)
        # They should mostly be in the same cluster
        unique, counts = np.unique(labels, return_counts=True)
        # The bigger cluster should contain most points (~35+ of 40)
        assert counts.max() >= 30


class TestSoftProbabilities:
    """Tests for _soft_probabilities."""

    def test_sums_to_one(self):
        from src.regime.fred_md_two_stage_kmeans import _soft_probabilities

        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 10))
        centroids = X[:6]  # use first 6 points as centroids

        probs = _soft_probabilities(X, centroids, distance_type="l2")
        assert probs.shape == (50, 6)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_cosine_distance(self):
        from src.regime.fred_md_two_stage_kmeans import _soft_probabilities

        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 10))
        centroids = X[:6]

        probs = _soft_probabilities(X, centroids, distance_type="cosine")
        assert probs.shape == (50, 6)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


class TestTwoStageKMeansRegime:
    """Tests for TwoStageKMeansRegime class."""

    def test_fit_basic(self):
        from src.regime.fred_md_two_stage_kmeans import TwoStageKMeansRegime

        X = _make_synthetic_data(n_samples=300, n_features=15)
        model = TwoStageKMeansRegime(max_pca_components=10)
        model.fit(X)

        assert model.result_ is not None
        assert model.result_.n_regimes == 6
        assert 0 in model.result_.regime_map
        assert model.result_.regime_map[0] == "CRISIS"
        assert model.pca_components > 0
        assert model.variance_retained > 0

    def test_fit_predict(self):
        from src.regime.fred_md_two_stage_kmeans import TwoStageKMeansRegime

        X = _make_synthetic_data(n_samples=300, n_features=15)
        model = TwoStageKMeansRegime(max_pca_components=10)

        _ = model.fit_predict(X)
        labels = model.predict()
        assert len(labels) == 300

    def test_predict_before_fit_raises(self):
        from src.regime.fred_md_two_stage_kmeans import TwoStageKMeansRegime

        model = TwoStageKMeansRegime()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict()

    def test_predict_proba_before_fit_raises(self):
        from src.regime.fred_md_two_stage_kmeans import TwoStageKMeansRegime

        model = TwoStageKMeansRegime()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict_proba()

    def test_crisis_detected(self):
        from src.regime.fred_md_two_stage_kmeans import TwoStageKMeansRegime

        X = _make_synthetic_data(n_samples=300, n_features=15)
        model = TwoStageKMeansRegime(max_pca_components=10)
        model.fit(X)

        result = model.result_
        # Crisis cluster should be the smaller one
        n_crisis = (result.combined_labels == 0).sum()
        n_normal = (result.combined_labels != 0).sum()
        assert n_crisis < n_normal  # crisis is small
        assert 5 <= n_crisis <= 50  # reasonable range for 5% of 300

    def test_six_regimes_produced(self):
        from src.regime.fred_md_two_stage_kmeans import TwoStageKMeansRegime

        X = _make_synthetic_data(n_samples=300, n_features=15)
        model = TwoStageKMeansRegime(max_pca_components=10)
        model.fit(X)

        labels = model.predict()
        unique_labels = set(labels)
        # May not find all 6 regimes depending on data, but should find most
        assert len(unique_labels) >= 2  # at least crisis + some normal
        assert max(labels) <= 5

    def test_probabilities_shape(self):
        from src.regime.fred_md_two_stage_kmeans import TwoStageKMeansRegime

        X = _make_synthetic_data(n_samples=100, n_features=10)
        model = TwoStageKMeansRegime(max_pca_components=5)
        model.fit(X)

        probs = model.predict_proba()
        assert probs.shape == (100, 6)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_get_signal(self):
        from src.regime.fred_md_two_stage_kmeans import TwoStageKMeansRegime

        X = _make_synthetic_data(n_samples=100, n_features=10)
        model = TwoStageKMeansRegime(max_pca_components=5)
        model.fit(X)

        signal = model.get_signal(latest_index=-1)
        assert "regime" in signal
        assert "confidence" in signal
        assert "probabilities" in signal
        assert "crisis_probability" in signal
        assert 0 <= signal["confidence"] <= 1
        assert len(signal["probabilities"]) >= 2  # at least crisis + some normal

    def test_predict_regime_names(self):
        from src.regime.fred_md_two_stage_kmeans import TwoStageKMeansRegime

        X = _make_synthetic_data(n_samples=100, n_features=10)
        model = TwoStageKMeansRegime(max_pca_components=5)
        model.fit(X)

        names = model.predict_regime_names()
        assert len(names) == 100
        assert all(isinstance(n, str) for n in names)
        # Should include portfolio-lab regime names
        valid_regimes = {"CRISIS", "HIGH_VOL", "NORMAL", "LOW_VOL", "RECOVERY"}
        assert all(n in valid_regimes for n in names)

    def test_deterministic_with_seed(self):
        from src.regime.fred_md_two_stage_kmeans import TwoStageKMeansRegime

        X = _make_synthetic_data(n_samples=200, n_features=10)

        model1 = TwoStageKMeansRegime(random_state=42, max_pca_components=5)
        model1.fit(X)
        labels1 = model1.predict()

        model2 = TwoStageKMeansRegime(random_state=42, max_pca_components=5)
        model2.fit(X)
        labels2 = model2.predict()

        assert np.array_equal(labels1, labels2)
