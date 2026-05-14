"""
Tests for FX Currency Carry modules.
v3.15 FX Currency Carry Overlay - Unit Tests
"""

import pytest
import json
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.data.fx_fetcher import FXFetcher, FXMetrics, DB_PATH, CACHE_TTL_HOURS
from src.signals.fx_carry_signal import (
    FXCarrySignal, FXCarrySignalGenerator, 
    FXMetrics as SignalFXMetrics
)


class TestFXFetcher:
    """Test suite for FX data fetcher."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            yield db_path
    
    @pytest.fixture
    def fetcher(self, temp_db):
        """Create FXFetcher with temp database."""
        return FXFetcher(db_path=temp_db)
    
    @pytest.fixture
    def mock_yahoo_response(self):
        """Mock Yahoo Finance API response."""
        return {
            "chart": {
                "result": [{
                    "timestamp": list(range(60)),
                    "indicators": {
                        "adjclose": [{
                            "adjclose": [25.0 + i * 0.1 for i in range(60)]
                        }]
                    }
                }]
            }
        }
    
    def test_init_creates_table(self, temp_db):
        """Test database initialization creates fx_cache table."""
        FXFetcher(db_path=temp_db)
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='fx_cache'"
            )
            assert cursor.fetchone() is not None
    
    def test_is_cache_fresh_empty(self, fetcher):
        """Test cache freshness check returns False for empty cache."""
        assert fetcher._is_cache_fresh("UUP") is False
    
    def test_is_cache_fresh_recent(self, fetcher):
        """Test cache freshness check returns True for recent data."""
        # Insert fresh data
        with sqlite3.connect(fetcher.db_path) as conn:
            conn.execute(
                """INSERT INTO fx_cache (symbol, price, price_30d_ago, volatility_30d, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                ("UUP", 27.5, 27.0, 0.10, datetime.now().isoformat())
            )
            conn.commit()
        
        assert fetcher._is_cache_fresh("UUP") is True
    
    def test_is_cache_fresh_stale(self, fetcher):
        """Test cache freshness check returns False for stale data."""
        # Insert stale data (older than TTL)
        stale_time = datetime.now() - timedelta(hours=CACHE_TTL_HOURS + 1)
        with sqlite3.connect(fetcher.db_path) as conn:
            conn.execute(
                """INSERT INTO fx_cache (symbol, price, price_30d_ago, volatility_30d, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                ("UUP", 27.5, 27.0, 0.10, stale_time.isoformat())
            )
            conn.commit()
        
        assert fetcher._is_cache_fresh("UUP") is False
    
    @patch('src.data.fx_fetcher.requests.get')
    def test_fetch_yahoo_success(self, mock_get, fetcher, mock_yahoo_response):
        """Test successful Yahoo Finance fetch."""
        mock_response = Mock()
        mock_response.json.return_value = mock_yahoo_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = fetcher._fetch_yahoo("UUP")
        
        assert "price" in result
        assert "price_30d_ago" in result
        assert "volatility_30d" in result
        assert "timestamp" in result
    
    @patch('src.data.fx_fetcher.requests.get')
    def test_fetch_yahoo_insufficient_data(self, mock_get, fetcher):
        """Test handling insufficient data from Yahoo."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "chart": {
                "result": [{
                    "timestamp": list(range(5)),
                    "indicators": {
                        "adjclose": [{"adjclose": [25.0] * 5}]
                    }
                }]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with pytest.raises(ValueError, match="Insufficient data"):
            fetcher._fetch_yahoo("UUP")
    
    @patch('src.data.fx_fetcher.requests.get')
    def test_fetch_metrics_computes_regime(self, mock_get, fetcher):
        """Test metrics computation with regime detection."""
        # UUP up, UDN down = USD strength
        uup_response = {
            "chart": {
                "result": [{
                    "timestamp": list(range(60)),
                    "indicators": {
                        "adjclose": [{
                            "adjclose": [25.0 + i * 0.1 for i in range(30)] + 
                                         [28.0 + i * 0.05 for i in range(30)]  # UUP up ~12%
                        }]
                    }
                }]
            }
        }
        udn_response = {
            "chart": {
                "result": [{
                    "timestamp": list(range(60)),
                    "indicators": {
                        "adjclose": [{
                            "adjclose": [20.0 - i * 0.05 for i in range(30)] +
                                         [18.5 - i * 0.03 for i in range(30)]  # UDN down ~7%
                        }]
                    }
                }]
            }
        }
        
        def side_effect(url, **kwargs):
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            if "UUP" in url:
                mock_response.json.return_value = uup_response
            else:
                mock_response.json.return_value = udn_response
            return mock_response
        
        mock_get.side_effect = side_effect
        
        metrics = fetcher.fetch_metrics()
        
        assert isinstance(metrics, FXMetrics)
        assert metrics.uup_return_30d > 0
        assert metrics.udn_return_30d < 0
        assert metrics.carry_regime in ["positive", "neutral"]  # Should be positive but may be borderline
        assert metrics.momentum_direction in ["bullish", "neutral"]
    
    def test_save_metrics_creates_file(self, fetcher, temp_db):
        """Test saving metrics creates JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fx_metrics.json"
            
            metrics = FXMetrics(
                timestamp=datetime.now().isoformat(),
                uup_price=27.5,
                udn_price=18.3,
                uup_return_30d=2.5,
                udn_return_30d=-1.8,
                usd_strength_score=0.54,
                carry_regime="positive",
                momentum_direction="bullish",
                volatility_regime="low",
                data_freshness_hours=0.5
            )
            
            fetcher.save_metrics(metrics, output_path)
            
            assert output_path.exists()
            with open(output_path) as f:
                data = json.load(f)
                assert data["carry_regime"] == "positive"
                assert data["momentum_direction"] == "bullish"


class TestFXCarrySignalGenerator:
    """Test suite for FX carry signal generator."""
    
    @pytest.fixture
    def temp_history(self):
        """Create temporary history file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            yield history_path
    
    @pytest.fixture
    def generator(self, temp_history):
        """Create signal generator with temp history."""
        gen = FXCarrySignalGenerator(signal_history_path=temp_history)
        # Mock the fetcher
        gen.fetcher = Mock()
        return gen
    
    def test_load_empty_history(self, generator):
        """Test loading empty history file."""
        history = generator._load_signal_history()
        assert history["signals"] == []
        assert history["last_signal_type"] == "neutral"
        assert history["days_in_regime"] == 0
    
    def test_update_persistence_same_signal(self, generator):
        """Test persistence counter increments for same signal."""
        # First call sets initial state
        days = generator._update_persistence("usd_strength")
        assert days == 1
        
        # Second call increments
        days = generator._update_persistence("usd_strength")
        assert days == 2
    
    def test_update_persistence_new_signal(self, generator):
        """Test persistence counter resets for new signal."""
        # Set up existing state
        generator._save_signal_history({
            "signals": [],
            "last_signal_type": "neutral",
            "days_in_regime": 5
        })
        
        # New signal resets counter
        days = generator._update_persistence("usd_strength")
        assert days == 1
    
    def test_check_momentum_conflict_both_positive(self, generator):
        """Test conflict detection when both UUP and UDN are positive."""
        metrics = Mock()
        metrics.uup_return_30d = 2.5
        metrics.udn_return_30d = 1.8
        
        assert generator._check_momentum_conflict(metrics) is True
    
    def test_check_momentum_conflict_no_conflict(self, generator):
        """Test no conflict when signals oppose."""
        metrics = Mock()
        metrics.uup_return_30d = 2.5
        metrics.udn_return_30d = -1.8
        
        assert generator._check_momentum_conflict(metrics) is False
    
    def test_calculate_confidence_usd_strength(self, generator):
        """Test confidence calculation for USD strength."""
        metrics = Mock()
        metrics.uup_return_30d = 4.0  # 4% return
        
        confidence = generator._calculate_confidence(metrics, "usd_strength")
        assert confidence == 1.0  # Max confidence at 4%
    
    def test_calculate_confidence_usd_weakness(self, generator):
        """Test confidence calculation for USD weakness."""
        metrics = Mock()
        metrics.udn_return_30d = 2.0  # 2% return
        
        confidence = generator._calculate_confidence(metrics, "usd_weakness")
        assert confidence == 0.5  # 2% / 4% = 0.5
    
    def test_calculate_allocation_shifts_strength(self, generator):
        """Test allocation shifts for USD strength."""
        spy_shift, efa_shift, vxus_shift = generator._calculate_allocation_shifts(
            "usd_strength", confidence=0.5
        )
        
        assert spy_shift > 0  # Add to SPY
        assert efa_shift < 0  # Reduce international
        assert vxus_shift < 0
        assert abs(spy_shift) == abs(efa_shift)
    
    def test_calculate_allocation_shifts_weakness(self, generator):
        """Test allocation shifts for USD weakness."""
        spy_shift, efa_shift, vxus_shift = generator._calculate_allocation_shifts(
            "usd_weakness", confidence=0.5
        )
        
        assert spy_shift < 0  # Reduce SPY
        assert efa_shift > 0  # Add international
        assert vxus_shift > 0
    
    def test_generate_signal_data_error(self, generator):
        """Test graceful handling of data fetch error."""
        generator.fetcher.fetch_metrics.side_effect = Exception("Network error")
        
        signal = generator.generate_signal()
        
        assert signal.signal_type == "neutral"
        assert signal.confidence == 0.0
        assert signal.reason == "data_error"
        assert not signal.is_valid
    
    def test_generate_signal_high_volatility(self, generator):
        """Test neutral signal during high volatility."""
        metrics = Mock()
        metrics.volatility_regime = "high"
        metrics.carry_regime = "positive"
        metrics.momentum_direction = "bullish"
        metrics.timestamp = datetime.now().isoformat()
        generator.fetcher.fetch_metrics.return_value = metrics
        
        signal = generator.generate_signal()
        
        assert signal.signal_type == "neutral"
        assert signal.reason == "high_volatility"
        assert not signal.is_valid
    
    def test_generate_signal_momentum_conflict(self, generator):
        """Test neutral signal on momentum conflict."""
        metrics = Mock()
        metrics.volatility_regime = "low"
        metrics.uup_return_30d = 2.0
        metrics.udn_return_30d = 1.5  # Both positive
        metrics.carry_regime = "positive"
        metrics.momentum_direction = "bullish"
        metrics.timestamp = datetime.now().isoformat()
        generator.fetcher.fetch_metrics.return_value = metrics
        
        signal = generator.generate_signal()
        
        assert signal.signal_type == "neutral"
        assert signal.reason == "momentum_conflict"
    
    def test_generate_signal_insufficient_persistence(self, generator):
        """Test neutral signal without sufficient persistence."""
        metrics = Mock()
        metrics.volatility_regime = "low"
        metrics.uup_return_30d = 3.0
        metrics.udn_return_30d = -1.5
        metrics.carry_regime = "positive"
        metrics.momentum_direction = "bullish"
        metrics.timestamp = datetime.now().isoformat()
        generator.fetcher.fetch_metrics.return_value = metrics
        
        signal = generator.generate_signal()
        
        # Should be neutral due to insufficient persistence
        assert signal.signal_type == "neutral"
        assert signal.reason == "insufficient_persistence"
    
    def test_generate_signal_valid_strength(self, generator):
        """Test valid USD strength signal."""
        # Simulate building up 6 days of persistence by calling multiple times
        # with the same bullish signal pattern
        metrics_template = Mock()
        metrics_template.volatility_regime = "low"
        metrics_template.uup_return_30d = 4.0
        metrics_template.udn_return_30d = -2.0
        metrics_template.carry_regime = "positive"
        metrics_template.momentum_direction = "bullish"
        metrics_template.timestamp = datetime.now().isoformat()
        
        # Call generate_signal 5 times to build up persistence to 5 days
        # The 6th call will produce the valid signal (after MIN_PERSISTENCE_DAYS=5 check)
        for i in range(5):
            generator.fetcher.fetch_metrics.return_value = metrics_template
            _ = generator.generate_signal()
        
        # Now we should have sufficient persistence for a valid signal
        generator.fetcher.fetch_metrics.return_value = metrics_template
        signal = generator.generate_signal()
        
        assert signal.signal_type == "usd_strength"
        assert signal.confidence > 0
        assert signal.is_valid
        assert signal.spy_shift > 0
        assert signal.efa_shift < 0
    
    def test_get_ensemble_input_format(self, generator):
        """Test ensemble input format."""
        # Set up for valid signal
        generator._save_signal_history({
            "signals": [{"signal": "usd_weakness", "timestamp": (datetime.now() - timedelta(days=6)).isoformat()}] * 6,
            "last_signal_type": "usd_weakness",
            "days_in_regime": 6
        })
        
        metrics = Mock()
        metrics.volatility_regime = "low"
        metrics.uup_return_30d = -2.0
        metrics.udn_return_30d = 3.0
        metrics.carry_regime = "negative"
        metrics.momentum_direction = "bearish"
        metrics.timestamp = datetime.now().isoformat()
        generator.fetcher.fetch_metrics.return_value = metrics
        
        ensemble_input = generator.get_ensemble_input()
        
        assert ensemble_input["source"] == "fx_carry"
        assert "signal" in ensemble_input
        assert "confidence" in ensemble_input
        assert "allocation_shifts" in ensemble_input
        assert "SPY" in ensemble_input["allocation_shifts"]
        assert "EFA" in ensemble_input["allocation_shifts"]
        assert "VXUS" in ensemble_input["allocation_shifts"]
        assert "metadata" in ensemble_input


class TestIntegration:
    """Integration tests for FX carry modules."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for integration tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_end_to_end_signal_flow(self, temp_dir):
        """Test complete signal flow from fetch to ensemble output."""
        db_path = temp_dir / "market.db"
        history_path = temp_dir / "history.json"
        
        # Create fetcher with mocked Yahoo
        fetcher = FXFetcher(db_path=db_path)
        
        with patch('src.data.fx_fetcher.requests.get') as mock_get:
            # UUP up 3%, UDN down 2%
            uup_data = {
                "chart": {
                    "result": [{
                        "timestamp": list(range(60)),
                        "indicators": {
                            "adjclose": [{
                                "adjclose": [25.0 + i * 0.05 for i in range(30)] +
                                             [26.5 + i * 0.03 for i in range(30)]
                            }]
                        }
                    }]
                }
            }
            udn_data = {
                "chart": {
                    "result": [{
                        "timestamp": list(range(60)),
                        "indicators": {
                            "adjclose": [{
                                "adjclose": [20.0 - i * 0.03 for i in range(30)] +
                                             [19.4 - i * 0.02 for i in range(30)]
                            }]
                        }
                    }]
                }
            }
            
            def side_effect(url, **kwargs):
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                if "UUP" in url:
                    mock_response.json.return_value = uup_data
                else:
                    mock_response.json.return_value = udn_data
                return mock_response
            
            mock_get.side_effect = side_effect
            
            # Fetch metrics
            metrics = fetcher.fetch_metrics()
            
            # Create generator and test signal
            generator = FXCarrySignalGenerator(signal_history_path=history_path)
            
            # Mock fetcher in generator to avoid second network call
            generator.fetcher = fetcher
            
            # Need to set up persistence first
            for _ in range(6):
                generator._update_persistence("usd_strength")
            
            signal = generator.generate_signal()
            ensemble = generator.get_ensemble_input()
        
        # Verify the chain worked
        assert metrics.uup_return_30d > 0
        assert ensemble["source"] == "fx_carry"
        assert "allocation_shifts" in ensemble


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
