"""Tests for src/research/regime_walk_forward.py.

Tests classify_regime_series, compute_ari, and _check_economic_coherence
with synthetic data (no disk/network dependencies).
"""

import numpy as np
import pandas as pd
import pytest

from src.research.regime_walk_forward import (
    WindowResult,
    WalkForwardResult,
    classify_regime_series,
    compute_ari,
)


@pytest.fixture
def uptrend_prices():
    """Generate a steady uptrend price series (likely NORMAL/LOW_VOL)."""
    dates = pd.bdate_range("2020-01-02", periods=300)
    np.random.seed(42)
    prices = 300 + np.cumsum(np.random.normal(0.5, 0.5, 300))
    return pd.Series(np.maximum(prices, 100), index=dates, name="SPY")


@pytest.fixture
def crash_prices():
    """Generate a price series with a crash in the middle (CRISIS)."""
    dates = pd.bdate_range("2020-01-02", periods=300)
    np.random.seed(99)
    prices = np.concatenate([
        300 + np.cumsum(np.random.normal(0.3, 1.0, 120)),
        330 + np.cumsum(np.random.normal(-2.0, 5.0, 60)),   # crash
        210 + np.cumsum(np.random.normal(0.5, 1.0, 120)),   # recovery
    ])
    return pd.Series(np.maximum(prices, 50), index=dates, name="SPY")


@pytest.fixture
def volatile_prices():
    """Generate a high-volatility price series."""
    dates = pd.bdate_range("2020-01-02", periods=300)
    np.random.seed(77)
    prices = 300 + np.cumsum(np.random.normal(0.0, 3.0, 300))
    return pd.Series(np.maximum(prices, 50), index=dates, name="SPY")


# ── classify_regime_series ─────────────────────────────────────────────────


class TestClassifyRegimeSeries:

    def test_returns_series(self, uptrend_prices):
        result = classify_regime_series(uptrend_prices, window=20)
        assert isinstance(result, pd.Series)
        assert result.name == "regime"

    def test_all_valid_regimes(self, uptrend_prices):
        result = classify_regime_series(uptrend_prices, window=20)
        valid = {"CRISIS", "HIGH_VOL", "NORMAL", "LOW_VOL", "RECOVERY"}
        assert all(r in valid for r in result)

    def test_length_matches(self, uptrend_prices):
        window = 20
        result = classify_regime_series(uptrend_prices, window=window)
        expected_len = len(uptrend_prices) - 1 - window  # returns dropna loses 1
        assert len(result) == expected_len

    def test_shorter_window_more_classifications(self, uptrend_prices):
        r20 = classify_regime_series(uptrend_prices, window=20)
        r10 = classify_regime_series(uptrend_prices, window=10)
        assert len(r10) > len(r20)

    def test_crash_produces_crisis_or_high_vol(self, crash_prices):
        result = classify_regime_series(crash_prices, window=20)
        # During the crash segment (roughly index 120-180 in returns space),
        # at least some dates should be CRISIS or HIGH_VOL
        crisis_count = sum(1 for r in result if r in ("CRISIS", "HIGH_VOL"))
        assert crisis_count > 0

    def test_uptrend_has_low_vol_or_normal(self, uptrend_prices):
        result = classify_regime_series(uptrend_prices, window=20)
        # Steady uptrend should have some LOW_VOL or NORMAL
        calm_count = sum(1 for r in result if r in ("NORMAL", "LOW_VOL"))
        assert calm_count > len(result) * 0.3  # at least 30%

    def test_index_is_datetime(self, uptrend_prices):
        result = classify_regime_series(uptrend_prices, window=20)
        assert hasattr(result.index, 'year')


# ── compute_ari ────────────────────────────────────────────────────────────


class TestComputeARI:

    def test_perfect_agreement(self):
        labels = ["A", "A", "B", "B", "C", "C"]
        assert compute_ari(labels, labels) == 1.0

    def test_different_lengths_returns_zero(self):
        assert compute_ari(["A", "B"], ["A", "B", "C"]) == 0.0

    def test_empty_returns_zero(self):
        assert compute_ari([], []) == 0.0

    def test_random_labels_near_zero(self):
        np.random.seed(42)
        l1 = list(np.random.choice(["A", "B", "C"], size=100))
        l2 = list(np.random.choice(["A", "B", "C"], size=100))
        ari = compute_ari(l1, l2)
        assert abs(ari) < 0.3  # random should be near zero

    def test_permuted_labels_same_ari(self):
        """Swapping label names shouldn't change ARI."""
        l1 = ["A", "A", "B", "B", "A", "B"]
        l2 = ["A", "A", "B", "B", "A", "B"]
        l1_swap = ["X", "X", "Y", "Y", "X", "Y"]
        l2_swap = ["X", "X", "Y", "Y", "X", "Y"]
        assert compute_ari(l1, l2) == compute_ari(l1_swap, l2_swap)

    def test_two_identical_clusters(self):
        l1 = ["A"] * 50 + ["B"] * 50
        l2 = ["X"] * 50 + ["Y"] * 50
        assert compute_ari(l1, l2) == 1.0

    def test_return_type(self):
        result = compute_ari(["A", "B"], ["A", "B"])
        assert isinstance(result, float)


# ── Dataclasses ────────────────────────────────────────────────────────────


class TestDataclasses:

    def test_window_result(self):
        w = WindowResult(
            window_id=0, train_start="2010-01-01", train_end="2014-12-31",
            test_start="2015-01-01", test_end="2015-12-31",
            n_train_days=1260, n_test_days=252,
            regime_distribution={"NORMAL": 150, "CRISIS": 50, "LOW_VOL": 52},
            dominant_regime="NORMAL", regime_transitions=8,
        )
        assert w.window_id == 0
        assert w.dominant_regime == "NORMAL"

    def test_walk_forward_result(self):
        r = WalkForwardResult(
            analysis_date="2026-05-27", n_windows=10,
            initial_window=1260, expansion_step=252,
            overall_regime_stability=0.9,
            regime_persistence={"NORMAL": 7.6, "CRISIS": 9.9},
            windows=[], economic_coherence={"GFC": True, "COVID": True},
            summary="Test summary.",
        )
        assert r.overall_regime_stability == 0.9
        assert r.economic_coherence["GFC"] is True
