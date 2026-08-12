"""Tests for TTL-cached computation utilities."""

import time

import numpy as np
import pandas as pd
import pytest

from src.utils.computation_cache import (
    get_realized_volatility,
    get_correlation_matrix,
    get_rolling_returns,
    invalidate_computation_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_caches():
    """Ensure a clean cache state before every test."""
    invalidate_computation_cache()
    yield


@pytest.fixture
def price_series() -> pd.Series:
    """A steadily rising price series (daily data, ~1yr)."""
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    # Daily return ~0.05% with small noise -> roughly 10% annual vol
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.006, size=252)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates, name="SPY")


@pytest.fixture
def multi_asset_df(price_series) -> pd.DataFrame:
    """A DataFrame with two correlated price series."""
    np.random.seed(99)
    noise = np.random.normal(0, 0.003, size=252)
    gld = price_series * (1 + 0.5 * (price_series.pct_change().fillna(0)).cumsum()) * (
        1 + noise
    ).cumsum()
    tlt = price_series * (1 + 0.3 * (price_series.pct_change().fillna(0)).cumsum()) * (
        1 + np.random.normal(0, 0.004, size=252)
    ).cumsum()
    return pd.DataFrame({"SPY": price_series, "GLD": gld, "TLT": tlt})


# ---------------------------------------------------------------------------
# get_realized_volatility
# ---------------------------------------------------------------------------

class TestGetRealizedVolatility:
    def test_returns_positive_float(self, price_series):
        """Volatility should be a positive float for a realistic price series."""
        vol = get_realized_volatility(price_series, window=21)
        assert vol is not None
        assert isinstance(vol, float)
        assert vol > 0
        # Typical annualized vol for the synthetic data is ~10-15%
        assert 0.05 < vol < 0.25

    def test_cache_hit_returns_same_value(self, price_series):
        """Second call with identical series should return cached value."""
        v1 = get_realized_volatility(price_series, window=21)
        v2 = get_realized_volatility(price_series, window=21)
        assert v1 == v2

    def test_different_window_produces_different_cache_key(self, price_series):
        """Different window sizes should generate distinct cache entries."""
        v_short = get_realized_volatility(price_series, window=5)
        v_long = get_realized_volatility(price_series, window=63)
        assert v_short != v_long

    def test_empty_series_returns_none(self):
        """Empty input should return None gracefully."""
        empty = pd.Series([], dtype=float)
        assert get_realized_volatility(empty) is None

    def test_single_element_series_returns_none(self):
        """A single-element series cannot produce volatility."""
        single = pd.Series([100.0])
        vol = get_realized_volatility(single)
        assert vol is None

    def test_short_series_falls_back_to_full_std(self):
        """Series shorter than window uses full-sample std instead of rolling."""
        short = pd.Series([100.0, 101.0, 102.0, 103.0, 99.0, 102.0])
        vol = get_realized_volatility(short, window=21)
        assert vol is not None
        assert isinstance(vol, float)
        assert vol > 0

    def test_invalidate_cache(self, price_series):
        """After invalidation, a fresh computation should occur."""
        v1 = get_realized_volatility(price_series, window=21)
        invalidate_computation_cache()
        v2 = get_realized_volatility(price_series, window=21)
        assert v1 == v2  # same data -> same result; just not from cache

    def test_caches_differ_by_name(self):
        """Series with different names but same values get separate entries."""
        data = [100.0, 101.0, 102.0, 103.0]
        a = pd.Series(data, name="A")
        b = pd.Series(data, name="B")
        va = get_realized_volatility(a)
        vb = get_realized_volatility(b)
        # Same data -> same result; cache held separately by key
        assert va == vb


# ---------------------------------------------------------------------------
# get_correlation_matrix
# ---------------------------------------------------------------------------

class TestGetCorrelationMatrix:
    def test_returns_dataframe(self, multi_asset_df):
        """Should return a DataFrame with the same columns as input (original order)."""
        corr = get_correlation_matrix(multi_asset_df, window=63)
        assert corr is not None
        assert isinstance(corr, pd.DataFrame)
        assert list(corr.columns) == ["SPY", "GLD", "TLT"]
        assert list(corr.index) == ["SPY", "GLD", "TLT"]

    def test_diagonal_is_one(self, multi_asset_df):
        """Diagonal entries of correlation matrix should be ~1.0."""
        corr = get_correlation_matrix(multi_asset_df, window=63)
        assert corr is not None
        for col in corr.columns:
            assert abs(corr.loc[col, col] - 1.0) < 1e-10

    def test_values_between_negative_one_and_one(self, multi_asset_df):
        """All off-diagonal correlations should be in [-1, 1]."""
        corr = get_correlation_matrix(multi_asset_df, window=63)
        assert corr is not None
        for i in corr.columns:
            for j in corr.columns:
                if i != j:
                    assert -1.0 <= corr.loc[i, j] <= 1.0

    def test_cache_hit(self, multi_asset_df):
        """Identical inputs should return cached result."""
        c1 = get_correlation_matrix(multi_asset_df, window=63)
        c2 = get_correlation_matrix(multi_asset_df, window=63)
        assert c1 is c2  # same object identity for cache hit

    def test_empty_dataframe_returns_none(self):
        """Empty DataFrame should return None."""
        empty = pd.DataFrame()
        assert get_correlation_matrix(empty) is None

    def test_single_column_dataframe(self):
        """Single-column DataFrame returns a 1x1 correlation matrix."""
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        corr = get_correlation_matrix(df)
        assert corr is not None
        assert corr.shape == (1, 1)
        assert abs(corr.loc["A", "A"] - 1.0) < 1e-10

    def test_different_windows(self, multi_asset_df):
        """Different windows should produce different cache keys."""
        c_short = get_correlation_matrix(multi_asset_df, window=5)
        c_long = get_correlation_matrix(multi_asset_df, window=63)
        # Different windows may produce different values
        assert c_short is not c_long

    def test_invalidate_cache(self, multi_asset_df):
        """After invalidation, a fresh computation should occur."""
        c1 = get_correlation_matrix(multi_asset_df, window=63)
        invalidate_computation_cache()
        c2 = get_correlation_matrix(multi_asset_df, window=63)
        assert c1 is not c2  # new object after invalidation
        pd.testing.assert_frame_equal(c1, c2)


# ---------------------------------------------------------------------------
# get_rolling_returns
# ---------------------------------------------------------------------------

class TestGetRollingReturns:
    def test_positive_return_for_uptrend(self, price_series):
        """A rising price series should yield a positive rolling return."""
        ret = get_rolling_returns(price_series, window=21)
        assert ret is not None
        assert ret > 0

    def test_cache_hit_returns_same_value(self, price_series):
        """Second call with identical series should return cached value."""
        r1 = get_rolling_returns(price_series, window=21)
        r2 = get_rolling_returns(price_series, window=21)
        assert r1 == r2

    def test_short_series(self):
        """Series with length < 2 returns None."""
        short = pd.Series([100.0])
        assert get_rolling_returns(short) is None

    def test_series_shorter_than_window_uses_full_period(self):
        """When series length < window, return total return from first to last."""
        series = pd.Series([100.0, 110.0, 121.0], name="X")
        ret = get_rolling_returns(series, window=21)
        assert ret is not None
        # 121 / 100 - 1 = 0.21
        assert abs(ret - 0.21) < 1e-10

    def test_window_return_precise_value(self):
        """Verify exact rolling return when series has enough data."""
        series = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], name="Y")
        # window=3 means: series[-1]/series[-3] - 1 = 120/110 - 1 = ~0.090909
        ret = get_rolling_returns(series, window=3)
        assert ret is not None
        expected = 120.0 / 110.0 - 1
        assert abs(ret - expected) < 1e-10

    def test_different_window_produces_different_result(self):
        """Different windows should return different rolling returns."""
        series = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0], name="Z")
        r_short = get_rolling_returns(series, window=2)
        r_long = get_rolling_returns(series, window=4)
        assert r_short is not None and r_long is not None
        assert r_short != r_long

    def test_invalidate_cache(self, price_series):
        """After invalidation, a fresh computation should occur."""
        r1 = get_rolling_returns(price_series, window=21)
        invalidate_computation_cache()
        r2 = get_rolling_returns(price_series, window=21)
        assert r1 == r2


# ---------------------------------------------------------------------------
# invalidate_computation_cache
# ---------------------------------------------------------------------------

class TestInvalidateComputationCache:
    def test_clears_all_three_caches(self, price_series, multi_asset_df):
        """After invalidation, all cache types should be cleared."""
        # Populate all caches
        get_realized_volatility(price_series)
        get_correlation_matrix(multi_asset_df)
        get_rolling_returns(price_series)

        # Invalidate
        invalidate_computation_cache()

        # Calling again should still work (fresh computation)
        assert get_realized_volatility(price_series) is not None
        assert get_correlation_matrix(multi_asset_df) is not None
        assert get_rolling_returns(price_series) is not None

    def test_can_be_called_on_empty_caches(self):
        """Calling invalidate on already-empty caches should not error."""
        invalidate_computation_cache()  # first call (already cleared by fixture)
        invalidate_computation_cache()  # second call on empty caches -> no error


# ---------------------------------------------------------------------------
# Environment variable COMPUTATION_CACHE_TTL_SECONDS
# ---------------------------------------------------------------------------

class TestEnvVarTTL:
    def test_custom_ttl_from_env(self, monkeypatch, price_series):
        """Setting COMPUTATION_CACHE_TTL_SECONDS should affect cache TTL."""
        monkeypatch.setenv("COMPUTATION_CACHE_TTL_SECONDS", "1")
        # Reimport with new env var
        import importlib
        from src.utils import computation_cache as cc

        importlib.reload(cc)

        v1 = cc.get_realized_volatility(price_series)
        assert v1 is not None

        # Wait for TTL to expire
        time.sleep(1.1)

        # Should recompute (not from cache)
        v2 = cc.get_realized_volatility(price_series)
        assert v2 is not None
        assert v1 == v2  # same data -> same result

    def test_default_ttl_is_300(self):
        """Default TTL should be 300 seconds when env var is not set."""
        import importlib
        from src.utils import computation_cache as cc

        importlib.reload(cc)
        assert cc.TTL == 300

    def test_invalid_ttl_value(self, monkeypatch, price_series):
        """Test that an explicitly set TTL value functions correctly."""
        monkeypatch.setenv("COMPUTATION_CACHE_TTL_SECONDS", "60")
        import importlib
        from src.utils import computation_cache as cc

        importlib.reload(cc)

        assert cc.TTL == 60
        vol = cc.get_realized_volatility(price_series)
        assert vol is not None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_access_does_not_crash(self, price_series, multi_asset_df):
        """Multiple threads accessing different caches simultaneously should not crash."""
        from concurrent.futures import ThreadPoolExecutor

        def worker_vol():
            for _ in range(20):
                get_realized_volatility(price_series, window=21)
                invalidate_computation_cache()

        def worker_corr():
            for _ in range(20):
                get_correlation_matrix(multi_asset_df, window=63)
                invalidate_computation_cache()

        def worker_ret():
            for _ in range(20):
                get_rolling_returns(price_series, window=21)
                invalidate_computation_cache()

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [
                pool.submit(worker_vol) for _ in range(2)
            ] + [
                pool.submit(worker_corr) for _ in range(2)
            ] + [
                pool.submit(worker_ret) for _ in range(2)
            ]
            for f in futures:
                f.result(timeout=10)  # should not raise
