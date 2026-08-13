"""
PBO/CSCV Diagnostic — Proof of Concept

Implements Combinatorially Symmetric Cross-Validation (CSCV) on a synthetic
performance matrix. Demonstrates that:
- Pure noise strategies → PBO ~55%
- A genuine signal embedded in one column → PBO < 0.2

No external dependencies beyond numpy.

Usage: uv run python scripts/pbo_cscv_diagnostic.py
"""

import itertools
import math
from typing import List, Tuple

import numpy as np


def generate_performance_matrix(
    T: int = 1000,
    N: int = 20,
    signal_col: int = 0,
    signal_strength: float = 0.0003,
    seed: int = 42,
) -> Tuple[np.ndarray, str]:
    """
    Generate T x N performance matrix.
    Columns 1..N-1 are pure noise (true SR = 0).
    Column 0 has a genuine signal (true SR > 0) if signal_strength > 0.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.01, size=(T, N))
    if signal_strength > 0:
        noise[:, signal_col] += signal_strength
    return noise, "genuine" if signal_strength > 0 else "noise"


def compute_sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe ratio from daily returns."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252)


def cscv_pbo(
    M: np.ndarray,
    S: int = 8,
    metric_fn=None,
) -> Tuple[float, List[float]]:
    """
    Combinatorially Symmetric Cross-Validation.

    Args:
        M: T x N performance matrix (rows=time, cols=strategies)
        S: Number of blocks (must be even). 8 → C(8,4)=70 combos.
        metric_fn: Function to rank strategies (default: Sharpe)

    Returns:
        (PBO, logits) where PBO = fraction of logits <= 0
    """
    if metric_fn is None:
        metric_fn = compute_sharpe

    T, N = M.shape
    block_size = T // S
    if block_size < 10:
        raise ValueError(f"Block size {block_size} too small (T={T}, S={S})")

    # Partition rows into S blocks
    blocks = [M[i * block_size : (i + 1) * block_size] for i in range(S)]

    # Generate all C(S, S/2) combinations
    half = S // 2
    all_combos = list(itertools.combinations(range(S), half))

    logits = []
    for is_indices in all_combos:
        oos_indices = tuple(i for i in range(S) if i not in is_indices)

        # Concatenate blocks for IS and OOS (preserving temporal order)
        is_data = np.concatenate([blocks[i] for i in is_indices], axis=0)
        oos_data = np.concatenate([blocks[i] for i in oos_indices], axis=0)

        # Rank strategies by IS metric
        is_metrics = np.array([metric_fn(is_data[:, j]) for j in range(N)])
        is_winner = np.argmax(is_metrics)

        # Compute OOS rank percentile of IS winner.
        # Higher OOS performance must map to higher omega_bar so a winner that
        # stays above median produces a positive logit, not an overfit flag.
        oos_metrics = np.array([metric_fn(oos_data[:, j]) for j in range(N)])
        oos_rank = np.sum(oos_metrics <= oos_metrics[is_winner])

        # Normalized percentile (0 = worst, 1 = best)
        omega_bar = oos_rank / N

        # Logit (avoid log(0))
        omega_bar = np.clip(omega_bar, 1e-6, 1 - 1e-6)
        logit = math.log(omega_bar / (1 - omega_bar))
        logits.append(logit)

    pbo = sum(1 for item in logits if item <= 0) / len(logits)
    return pbo, logits


def main():
    print("=" * 70)
    print("PBO/CSCV DIAGNOSTIC — PROOF OF CONCEPT")
    print("=" * 70)
    print()
    print("Reference: Bailey, Borwein, López de Prado, Zhu (2017)")
    print("  'The Probability of Backtest Overfitting'")
    print("  J. Computational Finance 20(4):39-69")
    print()

    # --- Test 1: Pure noise (all strategies have true SR = 0) ---
    print("--- TEST 1: Pure Noise (N=20 random strategies, T=1000 days) ---")
    M_noise, label = generate_performance_matrix(
        T=1000, N=20, signal_strength=0.0, seed=0
    )
    pbo_noise, logits_noise = cscv_pbo(M_noise, S=8)
    print(f"  PBO = {pbo_noise:.2%}")
    print("  Expected: around 50-55% across repeated noise draws")
    print(f"  Combos tested: {len(logits_noise)}")
    print()

    # --- Test 2: Genuine signal (column 0 has true SR > 0) ---
    print("--- TEST 2: Genuine Signal (N=20, column 0 has signal_strength=0.0015) ---")
    M_signal, label = generate_performance_matrix(
        T=1000, N=20, signal_col=0, signal_strength=0.0015, seed=42
    )
    pbo_signal, logits_signal = cscv_pbo(M_signal, S=8)
    print(f"  PBO = {pbo_signal:.2%}")
    print("  Expected: <10% (persistent signal → IS-best likely to outperform OOS)")
    print(f"  Combos tested: {len(logits_signal)}")
    print()

    # --- Test 3: Stronger signal ---
    print("--- TEST 3: Stronger Signal (signal_strength=0.0020) ---")
    M_strong, label = generate_performance_matrix(
        T=1000, N=20, signal_col=0, signal_strength=0.0020, seed=42
    )
    pbo_strong, logits_strong = cscv_pbo(M_strong, S=8)
    print(f"  PBO = {pbo_strong:.2%}")
    print("  Expected: <10% (strong signal → high confidence)")
    print()

    # --- Test 4: More blocks (S=16 → 12,870 combos) ---
    print("--- TEST 4: More Blocks (S=16, pure noise) ---")
    # Need more data for S=16 to keep block size reasonable
    M_noise_16, _ = generate_performance_matrix(T=2000, N=20, signal_strength=0.0, seed=42)
    pbo_16, logits_16 = cscv_pbo(M_noise_16, S=16)
    print(f"  PBO = {pbo_16:.2%}")
    print(f"  Combos tested: {len(logits_16)} (C(16,8) = 12,870)")
    print()

    # --- Summary ---
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"  Pure noise PBO:    {pbo_noise:.2%} (representative draw near 50%)")
    print(f"  Genuine signal:    {pbo_signal:.2%} (expect <10%)")
    print(f"  Stronger signal:   {pbo_strong:.2%} (expect <10%)")
    print()
    print("  PBO < 0.05-0.10: likely robust")
    print("  PBO 0.10-0.30:   acceptable with caution")
    print("  PBO > 0.50:      strong overfitting evidence")
    print()
    print("  To apply to portfolio-lab: build T x 94 performance matrix from")
    print("  grid-search results, then run cscv_pbo(matrix, S=8).")


if __name__ == "__main__":
    main()
