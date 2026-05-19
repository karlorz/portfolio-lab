"""One-shot signal pruning via correlation clustering and backward elimination."""
from typing import Dict, List

import numpy as np


def correlation_cluster(
    returns: Dict[str, np.ndarray],
    sharpe: Dict[str, float],
    threshold: float = 0.6,
) -> List[List[str]]:
    """Group signals where within-cluster correlation > threshold.

    Greedy: sort by Sharpe desc, skip if correlates >threshold with any
    already-selected representative. Returns list of clusters.
    """
    if not returns:
        return []

    lengths = {len(v) for v in returns.values()}
    if len(lengths) > 1:
        raise ValueError(f"All return arrays must have same length, got {lengths}")

    signal_names = sorted(returns.keys())
    data = np.column_stack([returns[name] for name in signal_names])
    corr_matrix = np.corrcoef(data, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    sorted_signals = sorted(
        signal_names, key=lambda s: sharpe.get(s, 0.0), reverse=True
    )
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
            if corr > threshold:
                cluster.append(other)
                clustered.add(other)
        clusters.append(cluster)

    return clusters


def backward_eliminate(
    returns: Dict[str, np.ndarray],
    target_n: int = 5,
) -> List[str]:
    """Backward-eliminate signals by marginal Sharpe contribution."""
    if not returns:
        return []

    signal_names = list(returns.keys())
    if len(signal_names) <= target_n:
        return signal_names[:]

    def ensemble_sharpe(names: List[str]) -> float:
        if not names:
            return -float("inf")
        data = np.column_stack([returns[n] for n in names])
        eq_weighted = np.mean(data, axis=1)
        mu = np.mean(eq_weighted)
        sigma = np.std(eq_weighted, ddof=0)
        return float(mu / (sigma + 1e-12) * np.sqrt(252))

    current = list(signal_names)
    while len(current) > target_n:
        best_removal: str | None = None
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
