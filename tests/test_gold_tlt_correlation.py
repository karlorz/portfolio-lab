"""Tests for src/research/gold_tlt_correlation.py.

Tests compute_rolling_correlation, detect_structural_breaks,
analyze_correlation_regimes with synthetic data (no disk/network).
"""

import numpy as np
import pandas as pd
import pytest

from src.research.gold_tlt_correlation import (
    CorrelationRegime,
    StructuralBreak,
    compute_rolling_correlation,
    detect_structural_breaks,
    analyze_correlation_regimes,
)


@pytest.fixture
def synthetic_prices():
    """Generate 500 days of synthetic GLD/TLT price data."""
    dates = pd.bdate_range("2020-01-02", periods=500)
    np.random.seed(42)
    gld = 180 + np.cumsum(np.random.normal(0.1, 1.5, 500))
    tlt = 140 + np.cumsum(np.random.normal(0.05, 1.0, 500))
    return pd.DataFrame({"GLD": gld, "TLT": tlt}, index=dates)


@pytest.fixture
def step_correlation_series():
    """Create a correlation series with a clear step change at midpoint."""
    dates = pd.bdate_range("2015-01-01", periods=400)
    # First half: low correlation (~0.0), second half: high correlation (~0.5)
    corr = np.concatenate([
        np.random.normal(0.0, 0.05, 200),
        np.random.normal(0.5, 0.05, 200),
    ])
    return pd.Series(corr, index=dates)


@pytest.fixture
def three_regime_series():
    """Create a correlation series with 3 distinct regimes."""
    dates = pd.bdate_range("2015-01-01", periods=300)
    corr = np.concatenate([
        np.random.normal(-0.3, 0.05, 100),  # diversifying
        np.random.normal(0.0, 0.05, 100),    # neutral
        np.random.normal(0.4, 0.05, 100),    # correlated
    ])
    return pd.Series(corr, index=dates)


# ── Rolling Correlation ────────────────────────────────────────────────────


class TestComputeRollingCorrelation:

    def test_returns_series(self, synthetic_prices):
        result = compute_rolling_correlation(synthetic_prices, window=63)
        assert isinstance(result, pd.Series)

    def test_values_bounded(self, synthetic_prices):
        result = compute_rolling_correlation(synthetic_prices, window=63)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_shorter_window_more_values(self, synthetic_prices):
        result_short = compute_rolling_correlation(synthetic_prices, window=21)
        result_long = compute_rolling_correlation(synthetic_prices, window=126)
        assert len(result_short) > len(result_long)

    def test_custom_symbols(self, synthetic_prices):
        prices = synthetic_prices.copy()
        prices["SPY"] = 400 + np.cumsum(np.random.normal(0.3, 2.0, 500))
        result = compute_rolling_correlation(prices, sym_a="GLD", sym_b="SPY", window=63)
        assert isinstance(result, pd.Series)
        assert len(result) > 0


# ── Structural Breaks ──────────────────────────────────────────────────────


class TestDetectStructuralBreaks:

    def test_returns_list(self, step_correlation_series):
        breaks = detect_structural_breaks(step_correlation_series, min_segment_days=60, threshold=0.2)
        assert isinstance(breaks, list)

    def test_detects_step_change(self, step_correlation_series):
        breaks = detect_structural_breaks(step_correlation_series, min_segment_days=60, threshold=0.2)
        assert len(breaks) >= 1
        # The break should have positive change (0.0 → 0.5)
        assert breaks[0].change > 0

    def test_sorted_by_magnitude(self, step_correlation_series):
        breaks = detect_structural_breaks(step_correlation_series, min_segment_days=60, threshold=0.2)
        if len(breaks) > 1:
            magnitudes = [abs(b.change) for b in breaks]
            assert magnitudes == sorted(magnitudes, reverse=True)

    def test_empty_for_short_series(self):
        short = pd.Series(np.random.normal(0, 0.1, 50),
                          index=pd.bdate_range("2020-01-01", periods=50))
        breaks = detect_structural_breaks(short, min_segment_days=126)
        assert breaks == []

    def test_break_fields(self, step_correlation_series):
        breaks = detect_structural_breaks(step_correlation_series, min_segment_days=60, threshold=0.2)
        if breaks:
            b = breaks[0]
            assert hasattr(b, 'date')
            assert hasattr(b, 'before_correlation')
            assert hasattr(b, 'after_correlation')
            assert hasattr(b, 'change')
            assert hasattr(b, 'significance')

    def test_significance_levels(self, step_correlation_series):
        breaks = detect_structural_breaks(step_correlation_series, min_segment_days=60, threshold=0.2)
        for b in breaks:
            assert b.significance in ("high", "medium", "low")

    def test_no_breaks_when_flat(self):
        flat = pd.Series(np.full(300, 0.1),
                         index=pd.bdate_range("2015-01-01", periods=300))
        breaks = detect_structural_breaks(flat, min_segment_days=60, threshold=0.3)
        assert len(breaks) == 0


# ── Correlation Regimes ────────────────────────────────────────────────────


class TestAnalyzeCorrelationRegimes:

    def test_returns_list(self, three_regime_series):
        regimes = analyze_correlation_regimes(three_regime_series)
        assert isinstance(regimes, list)
        assert len(regimes) > 0

    def test_regime_labels(self, three_regime_series):
        regimes = analyze_correlation_regimes(three_regime_series)
        labels = {r.regime_label for r in regimes}
        assert "diversifying" in labels
        assert "neutral" in labels
        assert "correlated" in labels

    def test_regime_fields(self, three_regime_series):
        regimes = analyze_correlation_regimes(three_regime_series)
        r = regimes[0]
        assert hasattr(r, 'start_date')
        assert hasattr(r, 'end_date')
        assert hasattr(r, 'mean_correlation')
        assert hasattr(r, 'std_correlation')
        assert hasattr(r, 'n_observations')
        assert hasattr(r, 'regime_label')

    def test_all_observations_accounted(self, three_regime_series):
        regimes = analyze_correlation_regimes(three_regime_series)
        total_obs = sum(r.n_observations for r in regimes)
        assert total_obs == len(three_regime_series)

    def test_single_regime_for_constant_series(self):
        const = pd.Series(np.full(100, 0.3),
                          index=pd.bdate_range("2020-01-01", periods=100))
        regimes = analyze_correlation_regimes(const)
        assert len(regimes) == 1
        assert regimes[0].regime_label == "correlated"

    def test_diversifying_threshold(self):
        vals = pd.Series([-0.2, -0.16, -0.15, -0.14],
                         index=pd.bdate_range("2020-01-01", periods=4))
        regimes = analyze_correlation_regimes(vals)
        # -0.2 and -0.16 are diversifying, -0.15 and -0.14 are neutral
        labels = [r.regime_label for r in regimes]
        assert "diversifying" in labels
        assert "neutral" in labels


# ── Dataclasses ────────────────────────────────────────────────────────────


class TestDataclasses:

    def test_correlation_regime(self):
        r = CorrelationRegime(
            start_date="2020-01-01", end_date="2020-12-31",
            mean_correlation=0.25, std_correlation=0.1,
            n_observations=252, regime_label="correlated",
        )
        assert r.regime_label == "correlated"
        assert r.n_observations == 252

    def test_structural_break(self):
        b = StructuralBreak(
            date="2020-03-15",
            before_correlation=0.1, after_correlation=-0.3,
            change=-0.4, significance="high",
        )
        assert b.significance == "high"
        assert b.change == -0.4
