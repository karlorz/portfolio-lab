"""Tests for src/data/price_cache.py — TTL-cached prices.json accessor."""

import json
import time
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.price_cache import get_prices, get_prices_df, invalidate_price_cache


class TestGetPrices:
    """Test get_prices() returns parsed prices.json data."""

    def test_returns_dict(self, tmp_path):
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps({"SPY": [{"d": "2024-01-02", "p": 100.0}]}))

        with patch("src.data.price_cache.PRICES_JSON", prices_file):
            result = get_prices()

        assert isinstance(result, dict)
        assert "SPY" in result
        assert result["SPY"][0]["p"] == 100.0

    def test_cache_hit_avoids_reread(self, tmp_path):
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps({"SPY": [{"d": "2024-01-02", "p": 100.0}]}))

        with patch("src.data.price_cache.PRICES_JSON", prices_file):
            result1 = get_prices()
            # Overwrite file — cache should still return old data
            prices_file.write_text(json.dumps({"SPY": [{"d": "2024-01-02", "p": 200.0}]}))
            result2 = get_prices()

        assert result1["SPY"][0]["p"] == 100.0
        assert result2["SPY"][0]["p"] == 100.0  # cached, not re-read

    def test_invalidate_forces_reread(self, tmp_path):
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps({"SPY": [{"d": "2024-01-02", "p": 100.0}]}))

        with patch("src.data.price_cache.PRICES_JSON", prices_file):
            result1 = get_prices()
            prices_file.write_text(json.dumps({"SPY": [{"d": "2024-01-02", "p": 200.0}]}))
            invalidate_price_cache()
            result2 = get_prices()

        assert result1["SPY"][0]["p"] == 100.0
        assert result2["SPY"][0]["p"] == 200.0  # re-read after invalidation

    def test_ttl_expiry_forces_reread(self, tmp_path):
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps({"SPY": [{"d": "2024-01-02", "p": 100.0}]}))

        with patch("src.data.price_cache.PRICES_JSON", prices_file), \
             patch("src.data.price_cache._PRICE_CACHE_TTL", 1):
            # Rebuild cache with short TTL
            from cachetools import TTLCache
            import src.data.price_cache as mod
            old_cache = mod._PRICE_CACHE
            mod._PRICE_CACHE = TTLCache(maxsize=1, ttl=1)

            try:
                result1 = get_prices()
                prices_file.write_text(json.dumps({"SPY": [{"d": "2024-01-02", "p": 200.0}]}))
                time.sleep(1.1)  # wait for TTL to expire
                result2 = get_prices()
            finally:
                mod._PRICE_CACHE = old_cache

        assert result1["SPY"][0]["p"] == 100.0
        assert result2["SPY"][0]["p"] == 200.0  # re-read after TTL expired

    def test_missing_file_raises(self, tmp_path):
        prices_file = tmp_path / "nonexistent.json"

        with patch("src.data.price_cache.PRICES_JSON", prices_file):
            with pytest.raises(FileNotFoundError):
                get_prices()

    def test_thread_safety(self, tmp_path):
        """Multiple threads calling get_prices() concurrently should not crash."""
        import threading

        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps({"SPY": [{"d": "2024-01-02", "p": 100.0}]}))

        results = []
        errors = []

        def reader():
            try:
                r = get_prices()
                results.append(r)
            except Exception as e:
                errors.append(e)

        # Patch once around the concurrent operation. Nested overlapping
        # unittest.mock.patch contexts are not thread-safe: each context may
        # capture another thread's temporary value and restore that stale path.
        with patch("src.data.price_cache.PRICES_JSON", prices_file):
            threads = [threading.Thread(target=reader) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 10
        assert all(r["SPY"][0]["p"] == 100.0 for r in results)


class TestGetPricesDF:
    """Test get_prices_df() returns pivoted DataFrame."""

    SAMPLE_DATA = {
        "SPY": [
            {"d": "2024-01-02", "p": 100.0},
            {"d": "2024-01-03", "p": 101.0},
        ],
        "GLD": [
            {"d": "2024-01-02", "p": 180.0},
            {"d": "2024-01-03", "p": 181.0},
        ],
        "TLT": [
            {"d": "2024-01-02", "p": 95.0},
            {"d": "2024-01-03", "p": 94.5},
        ],
    }

    def _patch_prices(self, tmp_path):
        prices_file = tmp_path / "prices.json"
        prices_file.write_text(json.dumps(self.SAMPLE_DATA))
        return patch("src.data.price_cache.PRICES_JSON", prices_file)

    def test_returns_dataframe(self, tmp_path):
        with self._patch_prices(tmp_path):
            df = get_prices_df()
        assert isinstance(df, pd.DataFrame)
        assert "SPY" in df.columns
        assert "GLD" in df.columns

    def test_dates_as_index(self, tmp_path):
        with self._patch_prices(tmp_path):
            df = get_prices_df()
        assert df.index.name == "date"
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_symbol_subset(self, tmp_path):
        with self._patch_prices(tmp_path):
            df = get_prices_df(symbols=["SPY", "GLD"])
        assert set(df.columns) == {"SPY", "GLD"}
        assert "TLT" not in df.columns

    def test_returns_copy(self, tmp_path):
        """Mutating the returned DataFrame should not affect the cache."""
        with self._patch_prices(tmp_path):
            df1 = get_prices_df()
            df1["SPY"] = 0  # mutate
            df2 = get_prices_df()
        assert df2["SPY"].iloc[0] == 100.0  # cache unaffected

    def test_cache_hit(self, tmp_path):
        with self._patch_prices(tmp_path):
            df1 = get_prices_df()
            df2 = get_prices_df()
        # Both should have same shape (from cache)
        assert df1.shape == df2.shape

    def test_empty_on_no_data(self, tmp_path):
        prices_file = tmp_path / "empty.json"
        prices_file.write_text("{}")
        with patch("src.data.price_cache.PRICES_JSON", prices_file):
            df = get_prices_df()
        assert df.empty

    def test_invalidate_clears_df_cache(self, tmp_path):
        with self._patch_prices(tmp_path):
            df1 = get_prices_df()
            invalidate_price_cache()
            df2 = get_prices_df()
        # Both should still work (re-built from raw cache or file)
        assert not df2.empty
