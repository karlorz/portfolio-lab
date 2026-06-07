"""Shared rolling-volatility helpers for backtest hot loops."""

from __future__ import annotations

import math
from typing import Sequence


def precomputed_rolling_volatility(
    returns: Sequence[float],
    *,
    window: int,
    fallback_vol: float,
    warmup_std_min_index: int,
    annualization_factor: float = 252.0,
) -> list[float]:
    """Compute legacy rolling volatility from prefix sums.

    The older overlay backtests used ``np.std`` on a sliced window for every
    output day.  This keeps the same population-standard-deviation contract
    while making each window lookup O(1).
    """
    if len(returns) == 0:
        return []
    if window <= 0:
        raise ValueError("window must be positive")

    prefix_sum = [0.0]
    prefix_sum_sq = [0.0]
    for value in returns:
        value_f = float(value)
        prefix_sum.append(prefix_sum[-1] + value_f)
        prefix_sum_sq.append(prefix_sum_sq[-1] + value_f * value_f)

    annualizer = math.sqrt(annualization_factor)

    def window_std(start: int, end: int) -> float:
        count = end - start
        if count <= 0:
            return 0.0
        total = prefix_sum[end] - prefix_sum[start]
        total_sq = prefix_sum_sq[end] - prefix_sum_sq[start]
        mean = total / count
        variance = max((total_sq / count) - (mean * mean), 0.0)
        return math.sqrt(variance)

    vols: list[float] = []
    for i in range(len(returns)):
        if i < window:
            if i >= warmup_std_min_index:
                vols.append(window_std(0, i + 1) * annualizer)
            else:
                vols.append(fallback_vol)
        else:
            vols.append(window_std(i - window, i) * annualizer)

    return vols
