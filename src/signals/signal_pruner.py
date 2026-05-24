"""One-shot signal pruning via correlation clustering and backward elimination."""
import numpy as np
from typing import Dict, List



__all__ = ['correlation_cluster', 'backward_eliminate']

def correlation_cluster(
    returns: Dict[str, np.ndarray],
    sharpe: Dict[str, float],
    threshold: float = 0.6,
) -> List[List[str]]:
    """Group signals into clusters where within-cluster correlation > threshold.

    Greedy algorithm: sort by Sharpe descending, then for each signal,
    if it correlates >threshold with any already-selected cluster
    representative, skip it. Otherwise, start a new cluster.

    Returns list of clusters, each cluster is a list of signal names.
    The first element of each cluster is the highest-Sharpe representative.
    """
    if not returns:
        return []

    lengths = {len(v) for v in returns.values()}
    if len(lengths) > 1:
        raise ValueError(f"All return arrays must have same length, got {lengths}")

    signal_names = sorted(returns.keys())

    # Compute correlation matrix
    data = np.column_stack([returns[name] for name in signal_names])
    corr_matrix = np.corrcoef(data, rowvar=False)

    # Handle NaN from constant returns
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Sort by Sharpe descending
    sorted_signals = sorted(signal_names, key=lambda s: sharpe.get(s, 0.0), reverse=True)
    name_to_idx = {name: i for i, name in enumerate(signal_names)}

    clusters: List[List[str]] = []
    clustered: set = set()

    for sig in sorted_signals:
        if sig in clustered:
            continue
        cluster = [sig]
        clustered.add(sig)
        for other in sorted_signals:
            if other in clustered:
                continue
            corr = corr_matrix[name_to_idx[sig]][name_to_idx[other]]
            # Only cluster positively correlated signals (redundant).
            # Negatively correlated signals are diversifying — keep separate.
            if corr > threshold:
                cluster.append(other)
                clustered.add(other)
        clusters.append(cluster)

    return clusters


def backward_eliminate(
    returns: Dict[str, np.ndarray],
    target_n: int = 5,
) -> List[str]:
    """Backward-eliminate signals to target_n by marginal Sharpe contribution.

    At each step, remove the signal whose removal improves (or least degrades)
    the equal-weight ensemble Sharpe. Stop when target_n remain.
    """
    if not returns:
        return []

    signal_names = list(returns)
    if len(signal_names) <= target_n:
        return signal_names[:]

    def ensemble_sharpe(names):
        if not names:
            return -float("inf")
        # Guard against missing keys in returns dict
        available = [n for n in names if n in returns]
        if not available:
            return -float("inf")
        data = np.column_stack([returns[n] for n in available])
        eq_weighted = np.mean(data, axis=1)
        mu = np.mean(eq_weighted)
        sigma = np.std(eq_weighted)
        if sigma < 1e-10:
            return 0.0
        return float(mu / sigma * np.sqrt(252))

    current = list(signal_names)
    while len(current) > target_n:
        # Find signal whose removal improves Sharpe the most
        best_removal = None
        best_sharpe = -float("inf")
        for name in current:
            without = [n for n in current if n != name]
            sh = ensemble_sharpe(without)
            if sh > best_sharpe:
                best_sharpe = sh
                best_removal = name
        if best_removal is not None:
            current.remove(best_removal)
        else:
            break

    return current
