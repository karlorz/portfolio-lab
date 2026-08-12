"""Tests for src/research/gold_tlt_correlation.py.

Tests compute_rolling_correlation, detect_structural_breaks,
analyze_correlation_regimes with synthetic data (no disk/network).
"""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.research.gold_tlt_correlation import (
    CorrelationAnalysis,
    CorrelationRegime,
    StructuralBreak,
    _compute_implications,
    _load_prices,
    compute_rolling_correlation,
    detect_structural_breaks,
    analyze_correlation_regimes,
    run_analysis,
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


# ── _compute_implications ─────────────────────────────────────────────────


class TestComputeImplications:

    def _make_analysis(self, **overrides):
        defaults = dict(
            symbol_pair="GLD/TLT",
            analysis_date="2026-05-28",
            window_days=252,
            current_correlation=0.0,
            current_regime="neutral",
            mean_correlation=0.0,
            min_correlation=-0.5,
            max_correlation=0.5,
            correlation_trend="stable",
            structural_breaks=[],
            regimes=[],
            implications="",
        )
        defaults.update(overrides)
        return CorrelationAnalysis(**defaults)

    def test_positive_correlation(self):
        result = _compute_implications(self._make_analysis(current_correlation=0.3))
        assert "positive" in result
        assert "eroding" in result

    def test_negative_correlation(self):
        result = _compute_implications(self._make_analysis(current_correlation=-0.3))
        assert "negative" in result
        assert "intact" in result

    def test_near_zero_correlation(self):
        result = _compute_implications(self._make_analysis(current_correlation=0.0))
        assert "near-zero" in result

    def test_increasing_trend(self):
        result = _compute_implications(self._make_analysis(correlation_trend="increasing"))
        assert "INCREASING" in result
        assert "move together more" in result

    def test_decreasing_trend(self):
        result = _compute_implications(self._make_analysis(correlation_trend="decreasing"))
        assert "DECREASING" in result
        assert "strengthening" in result

    def test_stable_trend_no_trend_text(self):
        result = _compute_implications(self._make_analysis(correlation_trend="stable"))
        assert "INCREASING" not in result
        assert "DECREASING" not in result

    def test_structural_breaks(self):
        breaks = [
            {"date": "2020-03-15", "before_correlation": 0.1,
             "after_correlation": -0.3, "change": -0.4, "significance": "high"},
            {"date": "2022-06-10", "before_correlation": -0.1,
             "after_correlation": 0.2, "change": 0.3, "significance": "medium"},
        ]
        result = _compute_implications(self._make_analysis(structural_breaks=breaks))
        assert "2 structural break(s)" in result
        assert "1 high significance" in result
        assert "regime has shifted" in result

    def test_no_structural_breaks(self):
        result = _compute_implications(self._make_analysis(structural_breaks=[]))
        assert "structural break" not in result

    def test_recent_regime(self):
        regimes = [
            {"start_date": "2024-06-01", "regime_label": "correlated",
             "mean_correlation": 0.25},
        ]
        result = _compute_implications(self._make_analysis(regimes=regimes))
        assert "Current regime" in result
        assert "correlated" in result
        assert "2024" in result

    def test_no_recent_regime(self):
        regimes = [
            {"start_date": "2020-01-01", "regime_label": "diversifying",
             "mean_correlation": -0.3},
        ]
        result = _compute_implications(self._make_analysis(regimes=regimes))
        assert "Current regime" not in result


# ── _load_prices ──────────────────────────────────────────────────────────


class TestLoadPrices:

    def _make_prices_json(self, symbols, tmp_path, fmt="list_of_dicts"):
        """Create a synthetic prices.json and return its path."""
        data = {}
        for sym in symbols:
            dates = pd.bdate_range("2020-01-02", periods=20)
            if fmt == "list_of_dicts":
                data[sym] = [{"d": str(d.date()), "p": 100.0 + i}
                             for i, d in enumerate(dates)]
            else:
                data[sym] = {
                    "d": [str(d.date()) for d in dates],
                    "p": [100.0 + i for i in range(len(dates))],
                }
        path = tmp_path / "prices.json"
        path.write_text(json.dumps(data))
        return path

    def test_list_of_dicts_format(self, tmp_path):
        path = self._make_prices_json(["GLD", "TLT"], tmp_path, "list_of_dicts")
        with patch("src.research.gold_tlt_correlation.PRICES_JSON", path):
            df = _load_prices(["GLD", "TLT"])
        assert set(df.columns) == {"GLD", "TLT"}
        assert len(df) == 20

    def test_dict_of_lists_format(self, tmp_path):
        path = self._make_prices_json(["GLD", "TLT"], tmp_path, "dict_of_lists")
        with patch("src.research.gold_tlt_correlation.PRICES_JSON", path):
            df = _load_prices(["GLD", "TLT"])
        assert set(df.columns) == {"GLD", "TLT"}
        assert len(df) == 20

    def test_missing_symbols_logged(self, tmp_path, caplog):
        path = self._make_prices_json(["GLD"], tmp_path)
        with patch("src.research.gold_tlt_correlation.PRICES_JSON", path):
            with caplog.at_level("WARNING"):
                df = _load_prices(["GLD", "MISSING"])
        assert "MISSING" in caplog.text
        assert "GLD" in df.columns

    def test_nonexistent_file(self):
        with patch("src.research.gold_tlt_correlation.PRICES_JSON", "/nonexistent/path.json"):
            with pytest.raises(FileNotFoundError):
                _load_prices(["GLD"])

    def test_default_symbols(self, tmp_path):
        path = self._make_prices_json(["GLD", "TLT", "SPY", "IEF"], tmp_path)
        with patch("src.research.gold_tlt_correlation.PRICES_JSON", path):
            df = _load_prices()
        assert set(df.columns) == {"GLD", "TLT", "SPY", "IEF"}


# ── run_analysis ──────────────────────────────────────────────────────────


class TestRunAnalysis:

    def _make_prices_json(self, tmp_path):
        """Create synthetic prices.json with GLD/TLT/SPY/IEF."""
        symbols = ["GLD", "TLT", "SPY", "IEF"]
        np.random.seed(123)
        dates = pd.bdate_range("2010-01-04", periods=500)
        data = {}
        for i, sym in enumerate(symbols):
            data[sym] = [
                {"d": str(d.date()), "p": round(float(100 + i * 10 + np.random.normal(0, 1)), 2)}
                for d in dates
            ]
        path = tmp_path / "prices.json"
        path.write_text(json.dumps(data))
        return path

    def test_returns_correlation_analysis(self, tmp_path):
        path = self._make_prices_json(tmp_path)
        with patch("src.research.gold_tlt_correlation.PRICES_JSON", path):
            result = run_analysis(window=63)
        assert isinstance(result, CorrelationAnalysis)
        assert result.symbol_pair == "GLD/TLT"
        assert result.window_days == 63
        assert isinstance(result.current_correlation, float)
        assert isinstance(result.correlation_trend, str)
        assert isinstance(result.implications, str)
        assert isinstance(result.structural_breaks, list)
        assert isinstance(result.regimes, list)

    def test_save_writes_json(self, tmp_path):
        path = self._make_prices_json(tmp_path)
        with patch("src.research.gold_tlt_correlation.PRICES_JSON", path):
            with patch("src.research.gold_tlt_correlation.save_results_json") as mock_save:
                run_analysis(window=63, save=True)
        mock_save.assert_called_once()
        call_kwargs = mock_save.call_args
        assert "gold_tlt_correlation.json" in call_kwargs.kwargs.get("output_path", call_kwargs[1].get("output_path", ""))

    def test_no_save_skips_write(self, tmp_path):
        path = self._make_prices_json(tmp_path)
        with patch("src.research.gold_tlt_correlation.PRICES_JSON", path):
            with patch("src.research.gold_tlt_correlation.save_results_json") as mock_save:
                run_analysis(window=63, save=False)
        mock_save.assert_not_called()
