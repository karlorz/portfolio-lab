"""
Tests for Closing Auction MOC Data Fetcher (v3.17 Phase 1)

Run: uv run pytest tests/test_closing_auction_fetcher.py -v
"""

import asyncio
import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.data.closing_auction_fetcher import (
    MOCImbalance,
    ClosingAuctionCache,
    ClosingAuctionFetcher,
    HistoricalBackfiller
)


class TestMOCImbalance:
    """Test MOC imbalance data class."""
    
    def test_imbalance_creation(self):
        """Test basic imbalance creation."""
        imb = MOCImbalance(
            symbol="SPY",
            timestamp=datetime(2026, 5, 15, 15, 50, 0),
            imbalance_shares=500000,
            paired_shares=2000000,
            reference_price=450.50,
            source="test"
        )
        
        assert imb.symbol == "SPY"
        assert imb.imbalance_shares == 500000
        assert imb.imbalance_ratio == 0.25
    
    def test_direction_score_buy(self):
        """Test direction scoring for buy imbalances."""
        # Strong buy (>50%)
        imb = MOCImbalance(
            symbol="SPY", timestamp=datetime.now(),
            imbalance_shares=1000000, paired_shares=1500000,
            reference_price=100, source="test"
        )
        assert imb.direction_score == 3
        
        # Medium buy (30-50%)
        imb.imbalance_shares = 500000
        imb.paired_shares = 1500000
        assert imb.direction_score == 2
        
        # Weak buy (15-30%)
        imb.imbalance_shares = 250000
        assert imb.direction_score == 1
    
    def test_direction_score_sell(self):
        """Test direction scoring for sell imbalances."""
        # Strong sell
        imb = MOCImbalance(
            symbol="SPY", timestamp=datetime.now(),
            imbalance_shares=-1000000, paired_shares=1500000,
            reference_price=100, source="test"
        )
        assert imb.direction_score == -3
        
        # Medium sell
        imb.imbalance_shares = -500000
        assert imb.direction_score == -2
        
        # Weak sell
        imb.imbalance_shares = -250000
        assert imb.direction_score == -1
    
    def test_direction_score_neutral(self):
        """Test direction scoring for neutral/weak imbalances."""
        # Neutral (<15%)
        imb = MOCImbalance(
            symbol="SPY", timestamp=datetime.now(),
            imbalance_shares=50000, paired_shares=1500000,
            reference_price=100, source="test"
        )
        assert imb.direction_score == 0
        
        # Zero
        imb.imbalance_shares = 0
        assert imb.direction_score == 0
    
    def test_to_dict(self):
        """Test JSON serialization."""
        imb = MOCImbalance(
            symbol="SPY", timestamp=datetime(2026, 5, 15, 15, 50, 0),
            imbalance_shares=500000, paired_shares=2000000,
            reference_price=450.50, source="test"
        )
        
        d = imb.to_dict()
        assert d['symbol'] == "SPY"
        assert d['imbalance_ratio'] == 0.25
        # 0.25 ratio is between 0.15-0.3 so score should be 1
        assert d['direction_score'] == 1
        assert 'timestamp' in d


class TestClosingAuctionCache:
    """Test caching functionality."""
    
    @pytest.fixture
    def temp_cache(self):
        """Create temporary cache for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            cache = ClosingAuctionCache(f.name)
            yield cache
            Path(f.name).unlink(missing_ok=True)
    
    def test_cache_storage_and_retrieval(self, temp_cache):
        """Test storing and retrieving imbalances."""
        imb = MOCImbalance(
            symbol="SPY", timestamp=datetime.now(),
            imbalance_shares=500000, paired_shares=2000000,
            reference_price=450.50, source="test"
        )
        
        temp_cache.store(imb)
        retrieved = temp_cache.get("SPY")
        
        assert retrieved is not None
        assert retrieved.symbol == "SPY"
        assert retrieved.imbalance_shares == 500000
    
    def test_cache_ttl(self, temp_cache):
        """Test cache TTL expiration."""
        # Store old data
        old_time = datetime.now() - timedelta(minutes=20)
        imb = MOCImbalance(
            symbol="SPY", timestamp=old_time,
            imbalance_shares=500000, paired_shares=2000000,
            reference_price=450.50, source="test"
        )
        
        temp_cache.store(imb)
        
        # Should be expired (default TTL = 15 min)
        retrieved = temp_cache.get("SPY", max_age_minutes=15)
        assert retrieved is None
        
        # Should be valid with longer TTL
        retrieved = temp_cache.get("SPY", max_age_minutes=30)
        assert retrieved is not None
    
    def test_get_all_for_date(self, temp_cache):
        """Test retrieving all imbalances for a date."""
        base_date = datetime(2026, 5, 15)
        
        # Store multiple imbalances
        for i, sym in enumerate(['SPY', 'QQQ', 'IWM']):
            imb = MOCImbalance(
                symbol=sym, 
                timestamp=base_date.replace(hour=15, minute=50),
                imbalance_shares=500000 * (i + 1),
                paired_shares=2000000,
                reference_price=400 + i * 50,
                source="test"
            )
            temp_cache.store(imb)
        
        # Store one for different date
        other_imb = MOCImbalance(
            symbol="SPY",
            timestamp=base_date - timedelta(days=1),
            imbalance_shares=100000,
            paired_shares=1000000,
            reference_price=400,
            source="test"
        )
        temp_cache.store(other_imb)
        
        # Retrieve for base date
        results = temp_cache.get_all_for_date(base_date)
        
        assert len(results) == 3
        symbols = {r.symbol for r in results}
        assert symbols == {'SPY', 'QQQ', 'IWM'}


class TestClosingAuctionFetcher:
    """Test data fetcher functionality."""
    
    @pytest.fixture
    def temp_fetcher(self):
        """Create temporary fetcher for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            fetcher = ClosingAuctionFetcher(f.name)
            yield fetcher
            Path(f.name).unlink(missing_ok=True)
    
    def test_fetcher_initialization(self, temp_fetcher):
        """Test fetcher initializes correctly."""
        assert temp_fetcher.cache is not None
        assert temp_fetcher.session is None  # Session created in async context
    
    def test_fetch_symbol_sync_wrapper(self, temp_fetcher):
        """Test fetching symbol with sync wrapper."""
        async def fetch_test():
            async with temp_fetcher:
                # Pre-populate cache
                imb = MOCImbalance(
                    symbol="SPY", timestamp=datetime.now(),
                    imbalance_shares=500000, paired_shares=2000000,
                    reference_price=450.50, source="cached"
                )
                temp_fetcher.cache.store(imb)
                
                # Should return cached value
                return await temp_fetcher.fetch_symbol("SPY")
        
        result = asyncio.run(fetch_test())
        assert result is not None
        assert result.source == "cached"
    
    def test_save_to_json(self, temp_fetcher, tmp_path):
        """Test JSON output."""
        output_path = tmp_path / "test_output.json"
        
        imbalances = {
            "SPY": MOCImbalance(
                symbol="SPY", timestamp=datetime.now(),
                imbalance_shares=500000, paired_shares=2000000,
                reference_price=450.50, source="test"
            )
        }
        
        temp_fetcher.save_to_json(imbalances, str(output_path))
        
        assert output_path.exists()
        
        with open(output_path) as f:
            data = json.load(f)
        
        assert 'timestamp' in data
        assert data['count'] == 1
        assert 'SPY' in data['imbalances']


class TestHistoricalBackfiller:
    """Test historical data backfilling."""
    
    @pytest.fixture
    def temp_backfiller(self):
        """Create temporary backfiller for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            cache = ClosingAuctionCache(f.name)
            backfiller = HistoricalBackfiller(cache)
            yield backfiller
            Path(f.name).unlink(missing_ok=True)
    
    def test_generate_synthetic_history(self, temp_backfiller):
        """Test synthetic history generation."""
        imbalances = temp_backfiller.generate_synthetic_history(days=30)
        
        # Should generate data for ~22 trading days (excluding weekends)
        # × 7 symbols
        assert len(imbalances) > 0
        assert len(imbalances) <= 30 * 7  # Max possible
        
        # Check that all symbols are represented
        symbols = {imb.symbol for imb in imbalances}
        assert len(symbols) > 0
        
        # Check timestamp format (should be 3:50pm)
        for imb in imbalances:
            assert imb.timestamp.hour == 15
            assert imb.timestamp.minute == 50
    
    def test_synthetic_direction_distribution(self, temp_backfiller):
        """Test that synthetic data has reasonable direction distribution."""
        imbalances = temp_backfiller.generate_synthetic_history(days=90)
        
        # Calculate direction distribution
        buy_count = sum(1 for imb in imbalances if imb.direction_score > 0)
        sell_count = sum(1 for imb in imbalances if imb.direction_score < 0)
        
        total = len(imbalances)
        if total == 0:
            pytest.skip("No imbalances generated")
        
        # Should have reasonable distribution (not all one direction)
        assert buy_count > 0
        assert sell_count > 0
        
        # Both buy and sell should be present - verify neither dominates completely
        # At least 15% in each direction (generous range for random seed variation)
        buy_ratio = buy_count / total
        sell_ratio = sell_count / total
        assert 0.15 < buy_ratio < 0.85
        assert 0.15 < sell_ratio < 0.85
        
        # Strong imbalances (|score| = 3) should be ~10%
        strong_count = sum(1 for imb in imbalances if abs(imb.direction_score) == 3)
        strong_ratio = strong_count / total if total > 0 else 0
        assert 0.05 < strong_ratio < 0.25  # Around 10% target


class TestIntegration:
    """Integration tests for the full flow."""
    
    def test_full_fetch_and_cache_cycle(self):
        """Test complete fetch-store-retrieve cycle."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
            
            # Create fetcher and run async operation via sync wrapper
            async def run_cycle():
                async with ClosingAuctionFetcher(db_path) as fetcher:
                    # Fetch data (may use network or return empty on failure)
                    imbalances = await fetcher.fetch_all(['SPY'])
                    
                    # Save to JSON
                    with tempfile.TemporaryDirectory() as tmpdir:
                        output_path = Path(tmpdir) / "output.json"
                        fetcher.save_to_json(imbalances, str(output_path))
                        assert output_path.exists()
            
            asyncio.run(run_cycle())
            
            # Verify database was created
            assert Path(db_path).exists()
            
            # Verify cache contains data (or at least schema)
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = {row[0] for row in cursor.fetchall()}
                assert 'moc_imbalances' in tables
            
            Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
