"""
Crypto Fear & Greed Index Fetcher
==================================
Fetches the Fear & Greed Index from Alternative.me API.
Uses caching to avoid excessive API calls.

API: https://alternative.me/crypto/fear-and-greed-index/
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional

import requests

from src.paths import DATA_DIR

logger = logging.getLogger(__name__)

API_URL = "https://api.alternative.me/fng/?limit=1&format=json"
CACHE_FILE = DATA_DIR / "crypto_fg_cache.json"
CACHE_TTL_HOURS = 24

@dataclass
class CryptoFgData:
    value: int       # 0-100
    classification: str  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    timestamp: str   # ISO format

class CryptoFgFetcher:
    def __init__(self):
        self.cache_path = CACHE_FILE
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_cache(self) -> Optional[CryptoFgData]:
        if not self.cache_path.exists():
            return None
        try:
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
            ts = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - ts < timedelta(hours=CACHE_TTL_HOURS):
                return CryptoFgData(
                    value=data['value'],
                    classification=data['classification'],
                    timestamp=data['timestamp']
                )
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_cache(self, data: CryptoFgData):
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(asdict(data), f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def fetch(self) -> Optional[CryptoFgData]:
        # Check cache first
        cached = self._load_cache()
        if cached:
            return cached

        try:
            response = requests.get(API_URL, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if 'data' in result and len(result['data']) > 0:
                entry = result['data'][0]
                data = CryptoFgData(
                    value=int(entry['value']),
                    classification=entry['value_classification'],
                    timestamp=datetime.now().isoformat()
                )
                self._save_cache(data)
                return data
        except Exception as e:
            logger.error(f"Failed to fetch Crypto F&G: {e}")
        
        return None

# Global instance
_fetcher = CryptoFgFetcher()

def get_crypto_fg() -> Optional[CryptoFgData]:
    return _fetcher.fetch()
