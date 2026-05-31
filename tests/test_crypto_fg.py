"""
Tests for Crypto Fear & Greed Index Fetcher.
"""
import json
import pytest
import time
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

from src.data.crypto_fg import CryptoFgFetcher, CryptoFgData, get_crypto_fg

@pytest.fixture
def fetcher(tmp_path):
    """Fixture for CryptoFgFetcher with a temporary cache directory."""
    # Patch DATA_DIR to use tmp_path for testing
    with patch('src.data.crypto_fg.DATA_DIR', tmp_path):
        # Patch CACHE_FILE to be inside the temporary path
        with patch('src.data.crypto_fg.CACHE_FILE', tmp_path / "crypto_fg_cache.json"):
            f = CryptoFgFetcher()
            yield f

def test_fetch_success(fetcher):
    """Test successful API fetch and caching."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "name": "Fear and Greed Index",
        "data": [
            {
                "value": "25",
                "value_classification": "Extreme Fear",
                "timestamp": "1234567890",
                "time_until_update": "..."
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch('src.data.crypto_fg.requests.get', return_value=mock_response) as mock_get:
        result = fetcher.fetch()
        
        assert result is not None
        assert result.value == 25
        assert result.classification == "Extreme Fear"
        mock_get.assert_called_once()
        
        # Verify cache was created
        assert fetcher.cache_path.exists()
        
        # Verify subsequent call uses cache
        result2 = fetcher.fetch()
        assert result2.value == 25
        # requests.get should still only be called once
        mock_get.assert_called_once()

def test_fetch_failure(fetcher):
    """Test API failure returns None."""
    with patch('src.data.crypto_fg.requests.get', side_effect=Exception("Network error")) as mock_get:
        result = fetcher.fetch()
        assert result is None
        mock_get.assert_called_once()

def test_cache_expiry(fetcher):
    """Test that expired cache triggers new API call."""
    # Create old cache
    old_data = {
        "value": 10,
        "classification": "Extreme Fear",
        "timestamp": (datetime.now() - timedelta(hours=25)).isoformat()
    }
    with open(fetcher.cache_path, 'w') as f:
        json.dump(old_data, f)
        
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [{"value": "75", "value_classification": "Greed"}]
    }
    mock_response.raise_for_status = MagicMock()
    
    with patch('src.data.crypto_fg.requests.get', return_value=mock_response):
        result = fetcher.fetch()
        assert result.value == 75

def test_global_fetcher():
    """Test the global get_crypto_fg function."""
    with patch('src.data.crypto_fg._fetcher') as mock_fetcher:
        mock_fetcher.fetch.return_value = CryptoFgData(50, "Neutral", datetime.now().isoformat())
        result = get_crypto_fg()
        assert result.value == 50
        mock_fetcher.fetch.assert_called_once()
