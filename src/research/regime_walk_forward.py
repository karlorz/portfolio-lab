#!/usr/bin/env python3
"""
Walk-Forward Validation for Market Regime Classifier.

Validates that the rule-based regime classifier (vol + drawdown + momentum)
produces stable and economically meaningful regime labels out-of-sample.

Methodology:
- Expanding window: Start with 504 days (2 years), expand by 126 days (6 months)
- For each window, classify all dates using the regime rules
- Compare IS (in-sample) vs OOS (out-of-sample) labels for overlapping dates
- Compute regime label consistency (ARI), regime duration stability,
  and economic coherence (do CRISIS labels align with known crisis periods?)

Usage:
    python -m src.research.regime_walk_forward
    python -m src.research.regime_walk_forward --n-windows 10 --save
"""

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.paths import DATA_DIR, PRICES_JSON
from src.backtest.metrics import save_results_json

logger = logging.getLogger(__name__)

__all__ = [
    "WindowResult", "WalkForwardResult",
    "classify_regime_series", "compute_ari",
    "run_walk_forward_validation",
]

# Regime classification thresholds (must match EnsembleVoter)
CRISIS_VOL_THRESHOLD = 0.30
CRISIS_DRAWDOWN_THRESHOLD = -0.10
HIGH_VOL_VOL_THRESHOLD = 0.20
HIGH_VOL_DRAWDOWN_THRESHOLD = -0.05
HIGH_VOL_MOM_THRESHOLD = 0.0
LOW_VOL_VOL_THRESHOLD = 0.12
LOW_VOL_MOM_THRESHOLD = 0.01
RECOVERY_DRAWDOWN_THRESHOLD = -0.03
RECOVERY_MOM_THRESHOLD = 0.01


@dataclass
class WindowResult:
    """Result from a single walk-forward window."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train_days: int
    n_test_days: int
    regime_distribution: Dict[str, int]  # regime -> count in test period
    dominant_regime: str
    regime_transitions: int


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation result."""
    analysis_date: str
    n_windows: int
    initial_window: int
    expansion_step: int
    overall_regime_stability: float  # ARI across windows
    regime_persistence: Dict[str, float]  # avg days per regime
    windows: List[Dict]
    economic_coherence: Dict[str, bool]  # crisis period -> correctly detected
    summary: str


def _load_spy_prices() -> pd.Series:
    """Load SPY prices from prices.json."""
    with open(PRICES_JSON) as f:
        raw = json.load(f)

    entries = raw.get("SPY", [])
    if isinstance(entries, list) and len(entries) > 0 and isinstance(entries[0], dict):
        dates = [e["d"] for e in entries]
        prices = [e["p"] for e in entries]
    else:
        raise ValueError("Unexpected prices.json format for SPY")

    return pd.Series(prices, index=pd.to_datetime(dates), name="SPY")


def classify_regime_series(prices: pd.Series, window: int = 20) -> pd.Series:
    """Classify regime for each date in the price series.

    Uses the same rules as EnsembleVoter.detect_regime() but applied
    sequentially across the full price history.

    Returns:
        Series of regime labels indexed by date
    """
    returns = prices.pct_change().dropna()
    cum_returns = (1 + returns).cumprod()

    regimes = []
    dates = []

    for i in range(window, len(returns)):
        date = returns.index[i]
        dates.append(date)

        # Realized vol (annualized)
        recent_returns = returns.iloc[max(0, i - window):i]
        vol = float(recent_returns.std() * np.sqrt(252))

        # Drawdown from peak
        cum = cum_returns.iloc[max(0, i - window):i + 1]
        peak = cum.max()
        current = cum.iloc[-1]
        drawdown = float((current / peak) - 1) if peak > 0 else 0.0

        # Momentum
        mom = float(recent_returns.sum())

        # Classify
        if vol > CRISIS_VOL_THRESHOLD or drawdown < CRISIS_DRAWDOWN_THRESHOLD:
            regime = "CRISIS"
        elif vol > HIGH_VOL_VOL_THRESHOLD or (drawdown < HIGH_VOL_DRAWDOWN_THRESHOLD and mom < HIGH_VOL_MOM_THRESHOLD):
            regime = "HIGH_VOL"
        elif drawdown < RECOVERY_DRAWDOWN_THRESHOLD and mom > RECOVERY_MOM_THRESHOLD:
            regime = "RECOVERY"
        elif vol < LOW_VOL_VOL_THRESHOLD and mom > LOW_VOL_MOM_THRESHOLD:
            regime = "LOW_VOL"
        else:
            regime = "NORMAL"

        regimes.append(regime)

    return pd.Series(regimes, index=dates, name="regime")


def compute_ari(labels1: List[str], labels2: List[str]) -> float:
    """Compute Adjusted Rand Index between two label sequences.

    ARI measures agreement between two clusterings, corrected for chance.
    ARI = 1.0 means perfect agreement, ARI ≈ 0 means random.
    """
    if len(labels1) != len(labels2) or len(labels1) == 0:
        return 0.0

    # Build contingency table
    all_labels = sorted(set(labels1 + labels2))
    n = len(labels1)

    # Count pairs
    from collections import Counter
    pairs1 = Counter(zip(labels1, labels2))
    row_sums = Counter(labels1)
    col_sums = Counter(labels2)

    # ARI computation
    sum_comb_c = sum(int(c * (c - 1) / 2) for c in row_sums.values())
    sum_comb_k = sum(int(c * (c - 1) / 2) for c in col_sums.values())
    sum_comb_n = int(n * (n - 1) / 2)
    sum_comb_ij = sum(int(v * (v - 1) / 2) for v in pairs1.values())

    expected = sum_comb_c * sum_comb_k / sum_comb_n if sum_comb_n > 0 else 0
    max_index = (sum_comb_c + sum_comb_k) / 2
    denominator = max_index - expected

    if denominator == 0:
        return 1.0 if sum_comb_ij == expected else 0.0

    ari = (sum_comb_ij - expected) / denominator
    return round(float(ari), 4)


def _check_economic_coherence(regime_series: pd.Series) -> Dict[str, bool]:
    """Check if known crisis periods were correctly classified as CRISIS or HIGH_VOL.

    Known crisis periods:
    - 2008-09-15 to 2009-03-09 (GFC)
    - 2020-02-19 to 2020-03-23 (COVID crash)
    - 2022-01-03 to 2022-06-16 (Rate hike selloff)
    """
    crisis_periods = {
        "GFC_2008": ("2008-09-15", "2009-03-09"),
        "COVID_2020": ("2020-02-19", "2020-03-23"),
        "RateHike_2022": ("2022-01-03", "2022-06-16"),
    }

    coherence = {}
    for name, (start, end) in crisis_periods.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        mask = (regime_series.index >= start_dt) & (regime_series.index <= end_dt)
        period_labels = regime_series[mask]

        if len(period_labels) == 0:
            coherence[name] = False
            continue

        # At least 30% of crisis period should be CRISIS or HIGH_VOL
        crisis_count = sum(1 for l in period_labels if l in ("CRISIS", "HIGH_VOL"))
        coherence[name] = crisis_count / len(period_labels) >= 0.3

    return coherence


def run_walk_forward_validation(
    initial_window: int = 504,
    expansion_step: int = 126,
    n_windows: int = 10,
    save: bool = False,
) -> WalkForwardResult:
    """Run walk-forward validation of the regime classifier.

    Args:
        initial_window: Initial training window in trading days (default: 504 = 2 years)
        expansion_step: Window expansion step in trading days (default: 126 = 6 months)
        n_windows: Number of expanding windows to test
        save: Whether to save results to JSON

    Returns:
        WalkForwardResult with complete validation results
    """
    logger.info("Loading SPY prices for regime walk-forward validation")
    prices = _load_spy_prices()

    # Classify the full series using all data
    full_regimes = classify_regime_series(prices)
    logger.info("Full series: %d days classified", len(full_regimes))

    # Check economic coherence of full classification
    coherence = _check_economic_coherence(full_regimes)

    # Walk-forward windows
    window_results = []
    prev_test_labels = None
    ari_scores = []

    for w in range(n_windows):
        train_end_idx = initial_window + w * expansion_step
        if train_end_idx >= len(prices):
            break

        train_prices = prices.iloc[:train_end_idx]
        test_end_idx = min(train_end_idx + expansion_step, len(prices))
        test_prices = prices.iloc[train_end_idx - 20:test_end_idx]  # need 20 days for vol

        # Classify using expanding window
        train_regimes = classify_regime_series(train_prices)

        # Get test period labels
        test_start = prices.index[train_end_idx]
        test_end = prices.index[min(test_end_idx - 1, len(prices) - 1)]
        test_labels = full_regimes[(full_regimes.index >= test_start) & (full_regimes.index <= test_end)]

        if len(test_labels) == 0:
            continue

        # Regime distribution in test period
        dist = test_labels.value_counts().to_dict()
        dominant = test_labels.mode().iloc[0] if len(test_labels) > 0 else "NORMAL"

        # Count transitions
        transitions = sum(1 for i in range(1, len(test_labels)) if test_labels.iloc[i] != test_labels.iloc[i - 1])

        window_result = WindowResult(
            window_id=w,
            train_start=str(prices.index[0].date()),
            train_end=str(prices.index[train_end_idx - 1].date()),
            test_start=str(test_start.date()),
            test_end=str(test_end.date()),
            n_train_days=train_end_idx,
            n_test_days=len(test_labels),
            regime_distribution={k: int(v) for k, v in dist.items()},
            dominant_regime=dominant,
            regime_transitions=int(transitions),
        )
        window_results.append(window_result)

        # Compute ARI between consecutive windows
        if prev_test_labels is not None and len(prev_test_labels) == len(test_labels):
            ari = compute_ari(list(prev_test_labels), list(test_labels))
            ari_scores.append(ari)

        prev_test_labels = test_labels

    # Regime persistence: average days per regime episode
    regime_changes = full_regimes != full_regimes.shift(1)
    regime_groups = regime_changes.cumsum()
    regime_persistence = {}
    for regime in full_regimes.unique():
        mask = full_regimes == regime
        lengths = full_regimes[mask].groupby(regime_groups[mask]).count()
        if len(lengths) > 0:
            regime_persistence[regime] = round(float(lengths.mean()), 1)

    # Overall stability
    overall_ari = round(float(np.mean(ari_scores)), 4) if ari_scores else 0.0

    # Summary
    summary_parts = [
        f"Walk-forward validation with {len(window_results)} windows.",
        f"Overall ARI (regime stability): {overall_ari:.4f}.",
        f"Economic coherence: {sum(coherence.values())}/{len(coherence)} crisis periods detected.",
    ]
    if overall_ari > 0.7:
        summary_parts.append("Regime classifier is STABLE — labels are consistent across expanding windows.")
    elif overall_ari > 0.4:
        summary_parts.append("Regime classifier is MODERATELY STABLE — some label drift between windows.")
    else:
        summary_parts.append("Regime classifier is UNSTABLE — significant label changes between windows. Consider recalibrating thresholds.")

    result = WalkForwardResult(
        analysis_date=datetime.now().isoformat(),
        n_windows=len(window_results),
        initial_window=initial_window,
        expansion_step=expansion_step,
        overall_regime_stability=overall_ari,
        regime_persistence=regime_persistence,
        windows=[asdict(w) for w in window_results],
        economic_coherence=coherence,
        summary=" ".join(summary_parts),
    )

    if save:
        output_path = DATA_DIR / "regime_walk_forward.json"
        save_results_json(asdict(result), output_path=str(output_path))
        logger.info("Saved walk-forward results to %s", output_path)

    logger.info("Walk-Forward Validation Summary:")
    logger.info("  Windows:       %d", len(window_results))
    logger.info("  ARI:           %.4f", overall_ari)
    logger.info("  Coherence:     %s", coherence)
    logger.info("  Persistence:   %s", regime_persistence)

    return result


def main():
    """CLI entry point."""
    import argparse
    from src.utils.log_config import configure_logging
    configure_logging()

    parser = argparse.ArgumentParser(description="Walk-Forward Validation for Regime Classifier")
    parser.add_argument("--initial-window", type=int, default=504,
                        help="Initial training window (trading days)")
    parser.add_argument("--expansion-step", type=int, default=126,
                        help="Window expansion step (trading days)")
    parser.add_argument("--n-windows", type=int, default=10,
                        help="Number of expanding windows")
    parser.add_argument("--save", action="store_true",
                        help="Save results to JSON")
    args = parser.parse_args()

    result = run_walk_forward_validation(
        initial_window=args.initial_window,
        expansion_step=args.expansion_step,
        n_windows=args.n_windows,
        save=args.save,
    )

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD REGIME VALIDATION")
    print(f"{'='*60}")
    print(f"  Windows:              {result.n_windows}")
    print(f"  Overall ARI:          {result.overall_regime_stability:.4f}")
    print(f"  Economic Coherence:   {result.economic_coherence}")
    print(f"  Regime Persistence:   {result.regime_persistence}")
    print(f"\n  SUMMARY: {result.summary}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
