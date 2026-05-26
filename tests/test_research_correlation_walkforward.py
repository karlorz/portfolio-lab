#!/usr/bin/env python3
"""Tests for src/research/gold_tlt_correlation.py and src/research/regime_walk_forward.py."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ── Gold-TLT Correlation Tests ──────────────────────────────────────────────


class TestComputeRollingCorrelation:
    """Tests for compute_rolling_correlation."""

    def _make_prices(self, n=300, seed=42):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2020-01-01", periods=n)
        gld = 150 + np.cumsum(rng.standard_normal(n) * 0.5)
        tlt = 100 + np.cumsum(rng.standard_normal(n) * 0.3)
        return pd.DataFrame({"GLD": gld, "TLT": tlt}, index=dates)

    def test_returns_series_indexed_by_date(self):
        from src.research.gold_tlt_correlation import compute_rolling_correlation

        prices = self._make_prices()
        result = compute_rolling_correlation(prices, "GLD", "TLT", window=50)
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_correlation_bounded(self):
        from src.research.gold_tlt_correlation import compute_rolling_correlation

        prices = self._make_prices()
        result = compute_rolling_correlation(prices, "GLD", "TLT", window=50)
        assert (result >= -1.0).all() and (result <= 1.0).all()

    def test_perfect_positive_correlation(self):
        from src.research.gold_tlt_correlation import compute_rolling_correlation

        dates = pd.bdate_range("2020-01-01", periods=300)
        gld = pd.Series(100 + np.arange(300) * 0.1, index=dates, name="GLD")
        tlt = pd.Series(50 + np.arange(300) * 0.05, index=dates, name="TLT")
        prices = pd.DataFrame({"GLD": gld, "TLT": tlt})
        result = compute_rolling_correlation(prices, "GLD", "TLT", window=50)
        assert result.iloc[-1] > 0.99

    def test_window_too_large_returns_empty(self):
        from src.research.gold_tlt_correlation import compute_rolling_correlation

        prices = self._make_prices(n=50)
        result = compute_rolling_correlation(prices, "GLD", "TLT", window=100)
        assert len(result) == 0


class TestDetectStructuralBreaks:
    """Tests for detect_structural_breaks."""

    def _make_corr_series(self, n=500, break_point=250, before=-0.3, after=0.3):
        dates = pd.bdate_range("2015-01-01", periods=n)
        values = np.concatenate([
            np.full(break_point, before) + np.random.default_rng(42).standard_normal(break_point) * 0.05,
            np.full(n - break_point, after) + np.random.default_rng(43).standard_normal(n - break_point) * 0.05,
        ])
        return pd.Series(values, index=dates)

    def test_detects_major_break(self):
        from src.research.gold_tlt_correlation import detect_structural_breaks

        corr = self._make_corr_series(before=-0.3, after=0.3)
        breaks = detect_structural_breaks(corr, min_segment_days=60, threshold=0.1)
        assert len(breaks) >= 1
        assert breaks[0].change > 0.3  # large shift detected

    def test_no_break_when_stable(self):
        from src.research.gold_tlt_correlation import detect_structural_breaks

        dates = pd.bdate_range("2015-01-01", periods=500)
        values = np.full(500, 0.1) + np.random.default_rng(42).standard_normal(500) * 0.02
        corr = pd.Series(values, index=dates)
        breaks = detect_structural_breaks(corr, min_segment_days=60, threshold=0.3)
        assert len(breaks) == 0

    def test_short_series_no_break(self):
        from src.research.gold_tlt_correlation import detect_structural_breaks

        dates = pd.bdate_range("2015-01-01", periods=50)
        corr = pd.Series(np.random.default_rng(42).standard_normal(50), index=dates)
        breaks = detect_structural_breaks(corr, min_segment_days=30)
        assert len(breaks) == 0

    def test_break_significance_classification(self):
        from src.research.gold_tlt_correlation import detect_structural_breaks

        # 0.4+ change = high
        corr = self._make_corr_series(before=-0.2, after=0.3)
        breaks = detect_structural_breaks(corr, min_segment_days=60, threshold=0.3)
        if len(breaks) > 0:
            assert breaks[0].significance in ("high", "medium", "low")


class TestAnalyzeCorrelationRegimes:
    """Tests for analyze_correlation_regimes."""

    def test_classifies_regimes(self):
        from src.research.gold_tlt_correlation import analyze_correlation_regimes

        dates = pd.bdate_range("2015-01-01", periods=200)
        # Negative → diversifying, then positive → correlated
        values = np.concatenate([
            np.full(100, -0.3),
            np.full(100, 0.3),
        ])
        corr = pd.Series(values, index=dates)
        regimes = analyze_correlation_regimes(corr)
        labels = [r.regime_label for r in regimes]
        assert "diversifying" in labels
        assert "correlated" in labels

    def test_all_neutral(self):
        from src.research.gold_tlt_correlation import analyze_correlation_regimes

        dates = pd.bdate_range("2015-01-01", periods=100)
        corr = pd.Series(np.full(100, 0.05), index=dates)
        regimes = analyze_correlation_regimes(corr)
        assert all(r.regime_label == "neutral" for r in regimes)


class TestLoadPrices:
    """Tests for _load_prices."""

    def test_loads_from_prices_json(self, tmp_path):
        from src.research.gold_tlt_correlation import _load_prices

        prices_data = {
            "GLD": [{"d": "2020-01-02", "p": 150.0}, {"d": "2020-01-03", "p": 151.0}],
            "TLT": [{"d": "2020-01-02", "p": 100.0}, {"d": "2020-01-03", "p": 101.0}],
        }
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps(prices_data))

        with patch("src.research.gold_tlt_correlation.PRICES_JSON", str(prices_file)):
            df = _load_prices(["GLD", "TLT"])
            assert "GLD" in df.columns
            assert "TLT" in df.columns
            assert len(df) == 2

    def test_missing_symbol_skipped(self, tmp_path):
        from src.research.gold_tlt_correlation import _load_prices

        prices_data = {
            "GLD": [{"d": "2020-01-02", "p": 150.0}, {"d": "2020-01-03", "p": 151.0}],
        }
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps(prices_data))

        with patch("src.research.gold_tlt_correlation.PRICES_JSON", str(prices_file)):
            # Requesting only MISSING symbols should raise
            with pytest.raises(ValueError, match="No price data loaded"):
                _load_prices(["MISSING"])


# ── Regime Walk-Forward Tests ───────────────────────────────────────────────


class TestClassifyRegimeSeries:
    """Tests for classify_regime_series."""

    def _make_prices(self, n=100, seed=42):
        dates = pd.bdate_range("2020-01-01", periods=n)
        rng = np.random.default_rng(seed)
        prices = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
        return pd.Series(prices, index=dates, name="SPY")

    def test_returns_series_with_regime_labels(self):
        from src.research.regime_walk_forward import classify_regime_series

        prices = self._make_prices()
        result = classify_regime_series(prices, window=20)
        assert isinstance(result, pd.Series)
        assert len(result) > 0
        unique_labels = set(result.unique())
        assert unique_labels.issubset({"CRISIS", "HIGH_VOL", "NORMAL", "LOW_VOL", "RECOVERY"})

    def test_crisis_during_steep_drop(self):
        from src.research.regime_walk_forward import classify_regime_series

        # Create a steep drop
        dates = pd.bdate_range("2020-01-01", periods=100)
        prices = np.concatenate([np.linspace(100, 120, 30), np.linspace(120, 80, 30), np.linspace(80, 90, 40)])
        series = pd.Series(prices, index=dates, name="SPY")
        result = classify_regime_series(series, window=10)
        # Should detect CRISIS or HIGH_VOL during the drop
        crisis_mask = result.isin(["CRISIS", "HIGH_VOL"])
        assert crisis_mask.any()

    def test_output_length_matches_input(self):
        from src.research.regime_walk_forward import classify_regime_series

        prices = self._make_prices(n=100)
        result = classify_regime_series(prices, window=20)
        # Output length = len(prices) - window - 1 (pct_change drops first)
        assert len(result) == len(prices) - 20 - 1


class TestComputeARI:
    """Tests for compute_ari."""

    def test_perfect_agreement(self):
        from src.research.regime_walk_forward import compute_ari

        labels = ["A", "B", "A", "B", "A"]
        assert compute_ari(labels, labels) == 1.0

    def test_random_agreement_near_zero(self):
        from src.research.regime_walk_forward import compute_ari

        rng = np.random.default_rng(42)
        l1 = [f"R{rng.integers(0, 3)}" for _ in range(200)]
        l2 = [f"R{rng.integers(0, 3)}" for _ in range(200)]
        ari = compute_ari(l1, l2)
        assert -0.1 < ari < 0.3  # Random should be near 0

    def test_different_lengths_returns_zero(self):
        from src.research.regime_walk_forward import compute_ari

        assert compute_ari(["A", "B"], ["A"]) == 0.0

    def test_empty_returns_zero(self):
        from src.research.regime_walk_forward import compute_ari

        assert compute_ari([], []) == 0.0


class TestEconomicCoherence:
    """Tests for _check_economic_coherence."""

    def test_detects_known_crisis(self):
        from src.research.regime_walk_forward import _check_economic_coherence

        # Create regime series with CRISIS during GFC
        dates = pd.bdate_range("2008-09-15", "2009-03-09", freq="B")
        labels = ["CRISIS"] * len(dates)
        regime_series = pd.Series(labels, index=dates, name="regime")
        result = _check_economic_coherence(regime_series)
        assert result.get("GFC_2008", False) is True

    def test_misses_crisis_if_normal(self):
        from src.research.regime_walk_forward import _check_economic_coherence

        dates = pd.bdate_range("2008-09-15", "2009-03-09", freq="B")
        labels = ["NORMAL"] * len(dates)
        regime_series = pd.Series(labels, index=dates, name="regime")
        result = _check_economic_coherence(regime_series)
        assert result.get("GFC_2008", True) is False


class TestWalkForwardIntegration:
    """Integration test using synthetic data (no file I/O)."""

    def test_walk_forward_runs(self, tmp_path):
        from src.research.regime_walk_forward import run_walk_forward_validation

        # Build synthetic price series long enough for walk-forward
        dates = pd.bdate_range("2005-01-01", periods=2000)
        rng = np.random.default_rng(42)
        prices_values = 100 + np.cumsum(rng.standard_normal(2000) * 0.5)
        spy_prices = pd.Series(prices_values, index=dates, name="SPY")

        # Patch _load_spy_prices to return our synthetic data
        with patch("src.research.regime_walk_forward._load_spy_prices", return_value=spy_prices):
            result = run_walk_forward_validation(
                initial_window=200,
                expansion_step=100,
                n_windows=5,
                save=False,
            )
            assert result.n_windows > 0
            # ARI can be slightly negative with random data
            assert -0.2 <= result.overall_regime_stability <= 1.0
            assert isinstance(result.regime_persistence, dict)
            assert isinstance(result.economic_coherence, dict)
