"""Tests for src/data/price_cache.py — TTL-cached prices.json accessor."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.data.price_cache import get_prices, invalidate_price_cache


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
                with patch("src.data.price_cache.PRICES_JSON", prices_file):
                    r = get_prices()
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 10
        assert all(r["SPY"][0]["p"] == 100.0 for r in results)
