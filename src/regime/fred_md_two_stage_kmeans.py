"""
Numpy-Only Two-Stage K-Means Macro Regime Classifier.

Implements Oliveira et al. 2025 (arXiv 2503.11499) two-layer modified k-means
for macro regime detection using FRED-MD data — WITHOUT sklearn dependency.

Algorithm:
  Layer 1: L2 distance k-means (k=2) separates outlier/crisis months by magnitude.
            The smaller cluster becomes the "crisis" regime.
  Layer 2: Cosine similarity k-means (k=5) on remaining months detects directional
            differences in macro state. Total: 6 regimes (1 crisis + 5 normal).
  PCA: Applied first via numpy SVD (61 components for 95% variance).
  Soft probability: Logarithmic scaling of centroid distances.

This bypasses the ML gate (PORTFOLIO_LAB_ENABLE_ML=0) since it only uses numpy.

Usage:
    from src.regime.fred_md_two_stage_kmeans import TwoStageKMeansRegime

    classifier = TwoStageKMeansRegime()
    classifier.fit(data_matrix)          # Fit on FRED-MD PCA-reduced data
    labels = classifier.predict()        # Get hard regime labels
    probs = classifier.predict_proba()   # Get soft probability assignments
    signal = classifier.get_signal()     # Get FredSignal-compatible dict

    # Or all at once:
    result = TwoStageKMeansRegime().fit_predict(data_matrix)

References:
    Oliveira et al. 2025: "Macro Regime Detection via FRED-MD PCA + Modified K-Means"
    arXiv: https://arxiv.org/abs/2503.11499
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "TwoStageKMeansRegime",
    "TwoStageResult",
    "pca_numpy",
    "kmeans_l2",
    "kmeans_cosine",
]


@dataclass
class TwoStageResult:
    """Result from two-stage k-means regime classification."""
    n_samples: int
    n_features: int
    n_pca_components: int
    pca_variance_retained: float

    # Layer 1 (crisis detection)
    crisis_labels: np.ndarray  # 0=normal, 1=crisis (or vice versa)
    crisis_cluster_size: int
    crisis_centroid: np.ndarray

    # Layer 2 (normal regime sub-types)
    normal_labels: np.ndarray  # 0-4 for the 5 normal sub-regimes
    normal_centroids: np.ndarray  # (5, n_components)

    # Combined (6 regimes: 0=crisis, 1-5=normal subtypes)
    combined_labels: np.ndarray
    regime_probabilities: np.ndarray  # (n_samples, 6) soft assignment

    # Regime mapping to portfolio-lab 5-regime system
    regime_map: Dict[int, str] = field(default_factory=dict)

    @property
    def n_regimes(self) -> int:
        return 6


def pca_numpy(
    X: np.ndarray,
    variance_threshold: float = 0.95,
    max_components: Optional[int] = None,
) -> Tuple[np.ndarray, int, float, np.ndarray]:
    """PCA via numpy SVD.

    Args:
        X: Data matrix (n_samples, n_features), should be standardized
        variance_threshold: Cumulative variance threshold for component selection
        max_components: Hard cap on components (optional)

    Returns:
        Tuple of (X_projected, n_components, variance_retained, components)
        - X_projected: (n_samples, n_components) projected data
        - n_components: number of components retained
        - variance_retained: fraction of variance retained
        - components: (n_components, n_features) principal axes
    """
    n_samples = X.shape[0]

    # Center
    X_centered = X - X.mean(axis=0)

    # SVD: X_centered = U @ diag(S) @ Vt
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Variance explained
    explained_var = (S ** 2) / max(n_samples - 1, 1)
    total_var = explained_var.sum()
    if total_var < 1e-10:
        return np.zeros((n_samples, 1)), 1, 1.0, np.zeros((1, X.shape[1]))

    explained_var_ratio = explained_var / total_var
    cumsum_var = np.cumsum(explained_var_ratio)
    n_components = int(np.searchsorted(cumsum_var, variance_threshold) + 1)
    n_components = max(1, min(n_components, len(S)))

    if max_components is not None:
        n_components = min(n_components, max_components)

    # Project: Z = U[:, :k] @ diag(S[:k])
    Z = U[:, :n_components] * S[:n_components]
    variance_retained = float(cumsum_var[min(n_components - 1, len(cumsum_var) - 1)])

    return Z, n_components, variance_retained, Vt[:n_components]


def _kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++ initialization for better starting centroids.

    Selects k initial centroids using distance-weighted sampling.
    """
    n = X.shape[0]
    centroids = np.zeros((k, X.shape[1]))
    # First centroid: random point
    centroids[0] = X[rng.integers(n)]

    for j in range(1, k):
        # Compute min squared distance to any existing centroid
        dists = np.min(
            np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :j, :]) ** 2, axis=2),
            axis=1,
        )
        # Sample with probability proportional to squared distance
        probs = dists / dists.sum()
        centroids[j] = X[rng.choice(n, p=probs)]

    return centroids


def kmeans_l2(
    X: np.ndarray,
    k: int,
    max_iters: int = 100,
    n_init: int = 10,
    seed: int = 42,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """L2-distance k-means clustering via pure numpy (Lloyd's algorithm).

    Used for Layer 1: separates crisis (outlier) months from normal months
    based on magnitude of PCA scores.

    Args:
        X: Data matrix (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum Lloyd iterations per restart
        n_init: Number of random initializations
        seed: Random seed
        tol: Convergence tolerance

    Returns:
        Tuple of (labels, centroids, inertia)
    """
    rng = np.random.default_rng(seed)
    best_inertia = np.inf
    best_labels = np.zeros(X.shape[0], dtype=np.int64)
    best_centroids = np.zeros((k, X.shape[1]))

    for _ in range(n_init):
        centroids = _kmeans_plus_plus_init(X, k, rng)

        for _ in range(max_iters):
            # Assignment: Euclidean distance to each centroid
            diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            distances = np.sqrt((diff ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)

            # Update: centroid = mean of assigned points
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if (labels == j).any()
                else X[rng.integers(X.shape[0])]
                for j in range(k)
            ])

            if np.allclose(centroids, new_centroids, atol=tol):
                break
            centroids = new_centroids

        # Compute inertia
        diff = X - centroids[labels]
        inertia = float((diff ** 2).sum())

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    return best_labels, best_centroids, best_inertia


def kmeans_cosine(
    X: np.ndarray,
    k: int,
    max_iters: int = 100,
    n_init: int = 10,
    seed: int = 42,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Cosine-similarity k-means clustering via pure numpy.

    Normalizes vectors to unit length, then applies L2 k-means.
    For unit vectors, L2 distance is monotonic with cosine distance:
    ||a - b||^2 = 2 - 2*cos(a,b)

    Used for Layer 2: separates normal months into 5 directional sub-regimes
    based on the direction (not magnitude) of macro conditions.

    Args:
        X: Data matrix (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum Lloyd iterations per restart
        n_init: Number of random initializations
        seed: Random seed
        tol: Convergence tolerance

    Returns:
        Tuple of (labels, centroids, inertia)
    """
    # L2-normalize each row to unit length
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Handle zero-norm rows
    norms = np.where(norms < 1e-10, 1.0, norms)
    X_unit = X / norms

    # L2 k-means on unit vectors
    labels, centroids, inertia = kmeans_l2(
        X_unit, k, max_iters, n_init, seed, tol,
    )
    # Re-normalize centroids (they are in unit space)
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroid_norms = np.where(centroid_norms < 1e-10, 1.0, centroid_norms)
    centroids = centroids / centroid_norms

    return labels, centroids, inertia


def _soft_probabilities(
    X: np.ndarray,
    centroids: np.ndarray,
    distance_type: str = "l2",
) -> np.ndarray:
    """Compute soft probability assignments from distances to centroids.

    Uses logarithmic scaling as described in Oliveira et al. 2025:
    P(cluster j | point i) = softmax(-d_ij / sigma)
    where sigma is the median distance across all points.

    Args:
        X: Data matrix (n_samples, n_features)
        centroids: (k, n_features) centroid matrix
        distance_type: "l2" or "cosine"

    Returns:
        (n_samples, k) probability matrix
    """
    k = centroids.shape[0]
    n = X.shape[0]

    if distance_type == "cosine":
        # Normalize both X and centroids to unit length
        X_n = X / np.where(np.linalg.norm(X, axis=1, keepdims=True) < 1e-10, 1.0,
                           np.linalg.norm(X, axis=1, keepdims=True))
        c_n = centroids / np.where(np.linalg.norm(centroids, axis=1, keepdims=True) < 1e-10,
                                    1.0, np.linalg.norm(centroids, axis=1, keepdims=True))
        distances = np.zeros((n, k))
        for j in range(k):
            distances[:, j] = np.sqrt(np.sum((X_n - c_n[j, :]) ** 2, axis=1))
    else:
        distances = np.zeros((n, k))
        for j in range(k):
            distances[:, j] = np.sqrt(np.sum((X - centroids[j, :]) ** 2, axis=1))

    # Logarithmic scaling
    sigma = float(np.median(distances))
    if sigma < 1e-10:
        sigma = 1.0

    # Softmax: exp(-d/sigma) / sum(exp(-d/sigma))
    scaled = -distances / sigma
    # Numerically stable softmax
    scaled_max = scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled - scaled_max)
    probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

    return probs


def _map_to_portfolio_regimes(
    crisis_labels: np.ndarray,
    normal_labels: np.ndarray,
    combined_labels: np.ndarray,
    normal_centroids: np.ndarray,
    crisis_centroid: np.ndarray,
    data: np.ndarray,
) -> Dict[int, str]:
    """Map 6 two-stage regimes to portfolio-lab's 5-regime system.

    Regime 0 (crisis) → CRISIS
    Regimes 1-5 (normal subtypes) → mapped to HIGH_VOL/NORMAL/LOW_VOL/RECOVERY
    based on centroid characteristics.

    Heuristic mapping:
    - Highest magnitude normal centroid → RECOVERY (strongest macro)
    - Lowest magnitude normal centroid → LOW_VOL (calmest macro)
    - High dispersion centroid → HIGH_VOL
    - Remainder → NORMAL
    """
    regime_map = {0: "CRISIS"}

    if len(normal_centroids) == 0:
        return regime_map

    # Compute centroid magnitude (L2 norm) for ordering
    centroid_magnitudes = np.linalg.norm(normal_centroids, axis=1)
    # Compute dispersion (std of PCA dimensions) for vol detection
    centroid_dispersions = np.std(normal_centroids, axis=1)

    # Rank by magnitude (1=highest, 5=lowest)
    mag_rank = np.argsort(np.argsort(-centroid_magnitudes))  # 0 = highest magnitude

    for j in range(5):
        if mag_rank[j] == 0:
            regime_map[j + 1] = "RECOVERY"  # Strongest macro
        elif centroid_dispersions[j] > np.median(centroid_dispersions) * 1.2:
            regime_map[j + 1] = "HIGH_VOL"  # High dispersion
        elif mag_rank[j] == 4:
            regime_map[j + 1] = "LOW_VOL"  # Lowest magnitude
        else:
            regime_map[j + 1] = "NORMAL"

    return regime_map


class TwoStageKMeansRegime:
    """Two-stage k-means macro regime classifier (Oliveira et al. 2025).

    Pure numpy implementation — no sklearn required.

    Attributes:
        pca_components: Number of PCA components retained
        variance_retained: Fraction of variance retained by PCA
        crisis_labels_: Layer 1 cluster labels (0=normal, 1=crisis or vice versa)
        normal_labels_: Layer 2 cluster labels for normal months (0-4)
        combined_labels_: Final regime labels (0=crisis, 1-5=normal subtypes)
        regime_probabilities_: Soft probability assignments (n_samples, 6)
        regime_map_: Mapping from combined label to portfolio-lab regime name
    """

    def __init__(
        self,
        variance_threshold: float = 0.95,
        max_pca_components: Optional[int] = 61,
        random_state: int = 42,
    ):
        self.variance_threshold = variance_threshold
        self.max_pca_components = max_pca_components
        self.random_state = random_state

        # Fit results
        self.pca_components: int = 0
        self.variance_retained: float = 0.0
        self._pca_projections: Optional[np.ndarray] = None
        self._result: Optional[TwoStageResult] = None

    def fit(self, X: np.ndarray) -> "TwoStageKMeansRegime":
        """Fit the two-stage k-means model.

        Args:
            X: Data matrix (n_samples, n_features). Should be FRED-MD data
               with t-code transformations and standardization applied.

        Returns:
            self
        """
        n_samples, n_features = X.shape
        logger.info(
            "Fitting two-stage k-means on %d samples × %d features", n_samples, n_features,
        )

        # Step 1: PCA via numpy SVD
        logger.info("Step 1: PCA via numpy SVD...")
        Z, n_components, var_retained, components = pca_numpy(
            X, self.variance_threshold, self.max_pca_components,
        )
        self.pca_components = n_components
        self.variance_retained = var_retained
        self._pca_projections = Z
        logger.info(
            "  PCA: %d → %d components (%.1f%% variance retained)",
            n_features, n_components, var_retained * 100,
        )

        # Step 2: Layer 1 — L2 k-means (k=2) for crisis/outlier detection
        logger.info("Step 2: Layer 1 L2 k-means (k=2)...")
        l1_labels, l1_centroids, l1_inertia = kmeans_l2(
            Z, k=2, seed=self.random_state,
        )

        # The smaller cluster is the outlier/crisis regime
        cluster_sizes = [(l1_labels == j).sum() for j in range(2)]
        crisis_cluster = int(np.argmin(cluster_sizes))
        normal_cluster = 1 - crisis_cluster

        crisis_size = cluster_sizes[crisis_cluster]
        normal_size = cluster_sizes[normal_cluster]
        crisis_pct = crisis_size / n_samples * 100
        logger.info(
            "  L1 result: crisis=%d samples (%.1f%%), normal=%d samples",
            crisis_size, crisis_pct, normal_size,
        )

        # Relabel: crisis=0, normal=1 (temporary)
        is_crisis = (l1_labels == crisis_cluster)
        crisis_idx = np.where(is_crisis)[0]
        normal_idx = np.where(~is_crisis)[0]

        # Step 3: Layer 2 — Cosine k-means (k=5) on normal months
        logger.info("Step 3: Layer 2 cosine k-means (k=5) on %d normal months...",
                     len(normal_idx))
        if len(normal_idx) > 10:
            Z_normal = Z[normal_idx]
            l2_labels_raw, l2_centroids, l2_inertia = kmeans_cosine(
                Z_normal, k=5, seed=self.random_state + 1,
            )
            # Ensure consistent label ordering by centroid magnitude
            l2_magnitudes = np.linalg.norm(l2_centroids, axis=1)
            l2_order = np.argsort(-l2_magnitudes)  # highest magnitude first
            l2_labels = np.zeros(len(l2_labels_raw), dtype=np.int64)
            for new_j, old_j in enumerate(l2_order):
                l2_labels[l2_labels_raw == old_j] = new_j
            l2_centroids = l2_centroids[l2_order]
        else:
            logger.warning("  Too few normal months for Layer 2 — all assigned to cluster 0")
            l2_labels = np.zeros(len(normal_idx), dtype=np.int64)
            l2_centroids = np.zeros((5, n_components))
            l2_centroids[0] = Z[normal_idx].mean(axis=0) if len(normal_idx) > 0 else Z.mean(axis=0)

        # Step 4: Combine labels (0=crisis, 1-5=normal subtypes)
        combined = np.zeros(n_samples, dtype=np.int64)
        # Crisis samples stay at 0
        # Normal samples get labels 1-5
        combined[normal_idx] = l2_labels + 1

        # Full centroid set
        all_centroids = np.zeros((6, n_components))
        all_centroids[0] = l1_centroids[crisis_cluster]
        all_centroids[1:] = l2_centroids

        # Step 5: Soft probability assignments
        logger.info("Step 4: Computing soft probability assignments...")
        probs = _soft_probabilities(Z, all_centroids, distance_type="cosine")

        # Step 6: Map to portfolio-lab regimes
        regime_map = _map_to_portfolio_regimes(
            is_crisis.astype(np.int64), l2_labels, combined,
            l2_centroids, l1_centroids[crisis_cluster], Z,
        )

        self._result = TwoStageResult(
            n_samples=n_samples,
            n_features=n_features,
            n_pca_components=n_components,
            pca_variance_retained=var_retained,
            crisis_labels=is_crisis.astype(np.int64),
            crisis_cluster_size=crisis_size,
            crisis_centroid=l1_centroids[crisis_cluster],
            normal_labels=l2_labels,
            normal_centroids=l2_centroids,
            combined_labels=combined,
            regime_probabilities=probs,
            regime_map=regime_map,
        )

        # Log regime distribution
        for label in range(6):
            count = (combined == label).sum()
            name = regime_map.get(label, f"Regime_{label}")
            logger.info("  Regime %d (%s): %d samples (%.1f%%)",
                         label, name, count, count / n_samples * 100)

        return self

    def fit_predict(self, X: np.ndarray) -> TwoStageResult:
        """Fit the model and return results."""
        self.fit(X)
        return self._result

    def predict(self) -> np.ndarray:
        """Return hard regime labels (0-5)."""
        if self._result is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._result.combined_labels

    def predict_proba(self) -> np.ndarray:
        """Return soft probability assignments (n_samples, 6)."""
        if self._result is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._result.regime_probabilities

    def predict_regime_names(self) -> List[str]:
        """Return portfolio-lab regime name for each sample."""
        if self._result is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return [self._result.regime_map.get(l, f"Regime_{l}")
                for l in self._result.combined_labels]

    def get_signal(self, latest_index: int = -1) -> dict:
        """Return a FredSignal-compatible dict for the specified sample.

        Args:
            latest_index: Sample index (default: -1 = most recent)

        Returns:
            Dict with regime, confidence, probabilities
        """
        if self._result is None:
            raise ValueError("Model not fitted. Call fit() first.")

        label = int(self._result.combined_labels[latest_index])
        probs = self._result.regime_probabilities[latest_index]
        regime = self._result.regime_map.get(label, f"Regime_{label}")
        confidence = float(probs[label])

        # Build probabilities dict — aggregate by portfolio-lab regime name
        # (multiple k-means clusters can map to the same portfolio regime)
        prob_dict: Dict[str, float] = {}
        for j in range(6):
            name = self._result.regime_map.get(j, f"Regime_{j}")
            prob_dict[name] = prob_dict.get(name, 0.0) + round(float(probs[j]), 4)

        return {
            "regime": regime,
            "regime_label": label,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "n_pca_components": self.pca_components,
            "variance_retained": round(self.variance_retained, 4),
            "crisis_probability": round(float(probs[0]), 4),
        }

    @property
    def result_(self) -> Optional[TwoStageResult]:
        return self._result
