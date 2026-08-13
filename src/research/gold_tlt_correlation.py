#!/usr/bin/env python3
"""
Gold-TLT Correlation Regime Analysis.

Analyzes the evolving correlation structure between gold (GLD) and long-term
treasuries (TLT) to assess whether the diversification benefit underlying
the 46/38/16 champion portfolio (SPY/GLD/TLT) is eroding.

Key findings from research (2024-2026):
- GLD-TLT correlation shifted from negative (-0.3 to -0.5) pre-2020 to
  near-zero or positive post-2022, driven by inflation regime changes
- Structural breaks detected around 2020-03 (COVID) and 2022-06 (rate hike cycle)
- This has direct implications for the 38% GLD allocation in the champion

Usage:
    python -m src.research.gold_tlt_correlation
    python -m src.research.gold_tlt_correlation --window 252 --save
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.paths import DATA_DIR, PRICES_JSON
from src.backtest.metrics import save_results_json

logger = logging.getLogger(__name__)

__all__ = [
    "CorrelationRegime", "StructuralBreak", "CorrelationAnalysis",
    "compute_rolling_correlation", "detect_structural_breaks",
    "analyze_correlation_regimes", "run_analysis",
]


@dataclass
class CorrelationRegime:
    """A correlation regime period."""
    start_date: str
    end_date: str
    mean_correlation: float
    std_correlation: float
    n_observations: int
    regime_label: str  # "diversifying", "neutral", "correlated"


@dataclass
class StructuralBreak:
    """A detected structural break in correlation."""
    date: str
    before_correlation: float
    after_correlation: float
    change: float
    significance: str  # "high", "medium", "low"


@dataclass
class CorrelationAnalysis:
    """Complete correlation analysis result."""
    symbol_pair: str
    analysis_date: str
    window_days: int
    current_correlation: float
    current_regime: str
    mean_correlation: float
    min_correlation: float
    max_correlation: float
    correlation_trend: str  # "increasing", "decreasing", "stable"
    structural_breaks: List[Dict]
    regimes: List[Dict]
    implications: str


def _load_prices(symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """Load price data from prices.json and pivot to DataFrame."""
    with open(PRICES_JSON) as f:
        raw = json.load(f)

    frames = {}
    target_symbols = symbols or ["GLD", "TLT", "SPY", "IEF"]

    for sym in target_symbols:
        if sym not in raw:
            logger.warning("Symbol %s not found in prices.json", sym)
            continue
        entries = raw[sym]
        if isinstance(entries, list) and len(entries) > 0 and isinstance(entries[0], dict):
            dates = [e["d"] for e in entries]
            prices = [e["p"] for e in entries]
        elif isinstance(entries, dict):
            dates = entries.get("d", entries.get("dates", []))
            prices = entries.get("p", entries.get("prices", []))
        else:
            continue

        frames[sym] = pd.Series(prices, index=pd.to_datetime(dates), name=sym)

    if not frames:
        raise ValueError("No price data loaded")

    df = pd.DataFrame(frames)
    df = df.dropna()
    df.index.name = "date"
    return df


def compute_rolling_correlation(
    prices: pd.DataFrame,
    sym_a: str = "GLD",
    sym_b: str = "TLT",
    window: int = 252,
) -> pd.Series:
    """Compute rolling correlation between two symbols.

    Args:
        prices: DataFrame with price columns
        sym_a: First symbol
        sym_b: Second symbol
        window: Rolling window in trading days (default: 252 = 1 year)

    Returns:
        Series of rolling correlations indexed by date
    """
    returns = prices[[sym_a, sym_b]].pct_change().dropna()
    rolling_corr = returns[sym_a].rolling(window).corr(returns[sym_b])
    return rolling_corr.dropna()


def detect_structural_breaks(
    rolling_corr: pd.Series,
    min_segment_days: int = 126,
    threshold: float = 0.3,
) -> List[StructuralBreak]:
    """Detect structural breaks in correlation using mean-shift detection.

    Scans for points where the rolling mean correlation shifts by more
    than `threshold` within a window. Returns breaks sorted by magnitude.

    Args:
        rolling_corr: Rolling correlation series
        min_segment_days: Minimum days between breaks
        threshold: Minimum correlation change to flag as a break

    Returns:
        List of StructuralBreak objects
    """
    if len(rolling_corr) < min_segment_days * 2:
        return []

    breaks = []
    last_break_idx = 0

    for i in range(min_segment_days, len(rolling_corr) - min_segment_days):
        if i - last_break_idx < min_segment_days:
            continue

        before = rolling_corr.iloc[max(0, i - min_segment_days):i]
        after = rolling_corr.iloc[i:i + min_segment_days]

        if len(before) < 20 or len(after) < 20:
            continue

        before_mean = float(before.mean())
        after_mean = float(after.mean())
        change = after_mean - before_mean

        if abs(change) >= threshold:
            significance = "high" if abs(change) >= 0.4 else "medium" if abs(change) >= 0.3 else "low"
            breaks.append(StructuralBreak(
                date=str(rolling_corr.index[i].date()),
                before_correlation=round(before_mean, 4),
                after_correlation=round(after_mean, 4),
                change=round(change, 4),
                significance=significance,
            ))
            last_break_idx = i

    # Sort by magnitude
    breaks.sort(key=lambda b: abs(b.change), reverse=True)
    return breaks


def analyze_correlation_regimes(
    rolling_corr: pd.Series,
    n_regimes: int = 3,
) -> List[CorrelationRegime]:
    """Segment rolling correlation into regimes using simple thresholding.

    Labels:
    - "diversifying": correlation < -0.15 (gold and bonds offset each other)
    - "neutral": -0.15 <= correlation <= 0.15 (no diversification benefit)
    - "correlated": correlation > 0.15 (gold and bonds move together)

    Args:
        rolling_corr: Rolling correlation series
        n_regimes: Minimum regime segments to identify

    Returns:
        List of CorrelationRegime objects
    """
    regimes = []
    current_label = None
    regime_start = None
    regime_corrs = []

    for date, corr in rolling_corr.items():
        prev_date = date
        if corr < -0.15:
            label = "diversifying"
        elif corr > 0.15:
            label = "correlated"
        else:
            label = "neutral"

        if label != current_label:
            if current_label is not None and regime_start is not None:
                regimes.append(CorrelationRegime(
                    start_date=str(regime_start.date()),
                    end_date=str(prev_date.date()),
                    mean_correlation=round(float(np.mean(regime_corrs)), 4),
                    std_correlation=round(float(np.std(regime_corrs)), 4),
                    n_observations=len(regime_corrs),
                    regime_label=current_label,
                ))
            current_label = label
            regime_start = date
            regime_corrs = [corr]
        else:
            regime_corrs.append(corr)

    # Close last regime
    if current_label is not None and regime_start is not None:
        regimes.append(CorrelationRegime(
            start_date=str(regime_start.date()),
            end_date=str(prev_date.date()),
            mean_correlation=round(float(np.mean(regime_corrs)), 4),
            std_correlation=round(float(np.std(regime_corrs)), 4),
            n_observations=len(regime_corrs),
            regime_label=current_label,
        ))

    return regimes


def _compute_implications(analysis: CorrelationAnalysis) -> str:
    """Generate implications text based on analysis results."""
    implications = []

    if analysis.current_correlation > 0.1:
        implications.append(
            f"GLD-TLT correlation is currently {analysis.current_correlation:.2f} (positive). "
            "The diversification benefit that makes 38% GLD allocation optimal may be eroding."
        )
    elif analysis.current_correlation < -0.15:
        implications.append(
            f"GLD-TLT correlation is currently {analysis.current_correlation:.2f} (negative). "
            "Diversification benefit is intact — 38% GLD allocation is well-supported."
        )
    else:
        implications.append(
            f"GLD-TLT correlation is currently {analysis.current_correlation:.2f} (near-zero). "
            "Diversification benefit is reduced but not eliminated."
        )

    if analysis.correlation_trend == "increasing":
        implications.append(
            "Correlation trend is INCREASING — if this continues, GLD and TLT will "
            "move together more often, reducing portfolio resilience in crisis periods."
        )
    elif analysis.correlation_trend == "decreasing":
        implications.append(
            "Correlation trend is DECREASING — diversification benefit is strengthening. "
            "Current allocation is well-positioned."
        )

    n_breaks = len(analysis.structural_breaks)
    high_breaks = [b for b in analysis.structural_breaks if b.get("significance") == "high"]
    if n_breaks > 0:
        implications.append(
            f"{n_breaks} structural break(s) detected ({len(high_breaks)} high significance). "
            "The correlation regime has shifted — historical backtest assumptions may not hold."
        )

    # Recent regime assessment
    recent_regimes = [r for r in analysis.regimes if "2024" in r.get("start_date", "") or "2025" in r.get("start_date", "") or "2026" in r.get("start_date", "")]
    if recent_regimes:
        latest = recent_regimes[-1]
        implications.append(
            f"Current regime ({latest['regime_label']}): correlation={latest['mean_correlation']:.2f} "
            f"since {latest['start_date']}."
        )

    return " ".join(implications)


def run_analysis(
    window: int = 252,
    save: bool = False,
) -> CorrelationAnalysis:
    """Run complete Gold-TLT correlation analysis.

    Args:
        window: Rolling correlation window in trading days
        save: Whether to save results to JSON

    Returns:
        CorrelationAnalysis with complete results
    """
    logger.info("Loading price data for GLD/TLT correlation analysis")
    prices = _load_prices(["GLD", "TLT", "SPY", "IEF"])

    logger.info("Computing rolling %d-day correlation (GLD-TLT)", window)
    rolling_corr = compute_rolling_correlation(prices, "GLD", "TLT", window)

    # Current state
    current_corr = float(rolling_corr.iloc[-1])
    mean_corr = float(rolling_corr.mean())
    min_corr = float(rolling_corr.min())
    max_corr = float(rolling_corr.max())

    # Trend: linear regression slope of last 500 observations
    tail = rolling_corr.tail(500)
    if len(tail) > 50:
        x = np.arange(len(tail))
        slope = np.polyfit(x, tail.values, 1)[0]
        if slope > 0.001:
            trend = "increasing"
        elif slope < -0.001:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "stable"

    # Detect structural breaks
    breaks = detect_structural_breaks(rolling_corr)
    breaks_dicts = [
        {
            "date": b.date,
            "before_correlation": b.before_correlation,
            "after_correlation": b.after_correlation,
            "change": b.change,
            "significance": b.significance,
        }
        for b in breaks
    ]

    # Analyze regimes
    regimes = analyze_correlation_regimes(rolling_corr)
    regimes_dicts = [asdict(r) for r in regimes]

    # Determine current regime
    current_label = "neutral"
    if current_corr < -0.15:
        current_label = "diversifying"
    elif current_corr > 0.15:
        current_label = "correlated"

    # Build result
    analysis = CorrelationAnalysis(
        symbol_pair="GLD/TLT",
        analysis_date=datetime.now().isoformat(),
        window_days=window,
        current_correlation=round(current_corr, 4),
        current_regime=current_label,
        mean_correlation=round(mean_corr, 4),
        min_correlation=round(min_corr, 4),
        max_correlation=round(max_corr, 4),
        correlation_trend=trend,
        structural_breaks=breaks_dicts,
        regimes=regimes_dicts,
        implications="",
    )

    analysis = CorrelationAnalysis(
        **{**asdict(analysis), "implications": _compute_implications(analysis)}
    )

    # Save results
    if save:
        output_path = DATA_DIR / "gold_tlt_correlation.json"
        save_results_json(asdict(analysis), output_path=str(output_path))
        logger.info("Saved analysis to %s", output_path)

    # Log summary
    logger.info("GLD/TLT Correlation Analysis:")
    logger.info("  Current:      %.4f (%s)", current_corr, current_label)
    logger.info("  Mean:         %.4f", mean_corr)
    logger.info("  Range:        [%.4f, %.4f]", min_corr, max_corr)
    logger.info("  Trend:        %s", trend)
    logger.info("  Breaks:       %d detected", len(breaks))
    logger.info("  Regimes:      %d identified", len(regimes))

    return analysis


def main():
    """CLI entry point."""
    import argparse
    from src.utils.log_config import configure_logging
    configure_logging()

    parser = argparse.ArgumentParser(description="Gold-TLT Correlation Regime Analysis")
    parser.add_argument("--window", type=int, default=252,
                        help="Rolling correlation window (trading days)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to JSON")
    args = parser.parse_args()

    analysis = run_analysis(window=args.window, save=args.save)

    # Print summary
    print(f"\n{'='*60}")
    print("GOLD-TLT CORRELATION REGIME ANALYSIS")
    print(f"{'='*60}")
    print(f"  Current Correlation:  {analysis.current_correlation:.4f} ({analysis.current_regime})")
    print(f"  Mean Correlation:     {analysis.mean_correlation:.4f}")
    print(f"  Range:                [{analysis.min_correlation:.4f}, {analysis.max_correlation:.4f}]")
    print(f"  Trend:                {analysis.correlation_trend}")
    print(f"  Structural Breaks:    {len(analysis.structural_breaks)}")
    print(f"  Regimes Identified:   {len(analysis.regimes)}")
    print("\nIMPLICATIONS:")
    print(f"  {analysis.implications}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
