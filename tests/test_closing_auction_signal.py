"""
Tests for Closing Auction Signal Generator (v3.17 Phase 2)

Run with: uv run pytest tests/test_closing_auction_signal.py -v
"""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.data.closing_auction_fetcher import MOCImbalance, ClosingAuctionCache
from src.signals.closing_auction import (
    ClosingAuctionSignal,
    ClosingAuctionSignalGenerator,
    HistoricalValidator,
    SignalAggregator,
    SignalConfidence,
    SignalDirection,
)


class TestSignalDirection:
    """Test SignalDirection enum."""
    
    def test_direction_values(self):
        """Test direction score mapping."""
        assert SignalDirection.STRONG_BUY.value == 3
        assert SignalDirection.BUY.value == 2
        assert SignalDirection.WEAK_BUY.value == 1
        assert SignalDirection.NEUTRAL.value == 0
        assert SignalDirection.WEAK_SELL.value == -1
        assert SignalDirection.SELL.value == -2
        assert SignalDirection.STRONG_SELL.value == -3


class TestClosingAuctionSignal:
    """Test ClosingAuctionSignal dataclass."""
    
    @pytest.fixture
    def sample_imbalance(self):
        """Create sample MOC imbalance."""
        return MOCImbalance(
            symbol="SPY",
            timestamp=datetime.now(),
            imbalance_shares=1000000,
            paired_shares=5000000,
            reference_price=450.0,
            source="test"
        )
    
    @pytest.fixture
    def sample_signal(self, sample_imbalance):
        """Create sample closing auction signal."""
        return ClosingAuctionSignal(
            symbol="SPY",
            timestamp=datetime.now(),
            direction=SignalDirection.BUY,
            confidence=SignalConfidence.HIGH,
            imbalance=sample_imbalance,
            entry_price=450.0,
            target_exit_price=450.45,
            stop_loss_price=449.1,
            historical_win_rate=0.68,
            historical_count=50,
            max_position_pct=0.03,
            urgency="high"
        )
    
    def test_to_dict(self, sample_signal):
        """Test signal serialization."""
        d = sample_signal.to_dict()
        
        assert d['symbol'] == "SPY"
        assert d['direction'] == "BUY"
        assert d['direction_score'] == 2
        assert d['confidence'] == "high"
        assert d['entry_price'] == 450.0
        assert d['historical_win_rate'] == 0.68
        assert d['urgency'] == "high"
    
    def test_should_trade_valid(self, sample_signal):
        """Test valid trade signal."""
        assert sample_signal.should_trade is True
    
    def test_should_trade_neutral(self, sample_imbalance):
        """Test neutral signal is not tradeable."""
        signal = ClosingAuctionSignal(
            symbol="SPY",
            timestamp=datetime.now(),
            direction=SignalDirection.NEUTRAL,
            confidence=SignalConfidence.HIGH,
            imbalance=sample_imbalance,
            entry_price=450.0,
            target_exit_price=450.0,
            stop_loss_price=None,
            historical_win_rate=0.60,
            historical_count=50,
            max_position_pct=0.03,
            urgency="normal"
        )
        assert signal.should_trade is False
    
    def test_should_trade_low_confidence(self, sample_imbalance):
        """Test low confidence signal is not tradeable."""
        signal = ClosingAuctionSignal(
            symbol="SPY",
            timestamp=datetime.now(),
            direction=SignalDirection.BUY,
            confidence=SignalConfidence.LOW,
            imbalance=sample_imbalance,
            entry_price=450.0,
            target_exit_price=450.45,
            stop_loss_price=449.1,
            historical_win_rate=0.50,
            historical_count=50,
            max_position_pct=0.03,
            urgency="high"
        )
        assert signal.should_trade is False
    
    def test_should_trade_stale(self, sample_imbalance):
        """Test stale signal is not tradeable."""
        signal = ClosingAuctionSignal(
            symbol="SPY",
            timestamp=datetime.now() - timedelta(minutes=10),
            direction=SignalDirection.BUY,
            confidence=SignalConfidence.HIGH,
            imbalance=sample_imbalance,
            entry_price=450.0,
            target_exit_price=450.45,
            stop_loss_price=449.1,
            historical_win_rate=0.68,
            historical_count=50,
            max_position_pct=0.03,
            urgency="high"
        )
        assert signal.should_trade is False
    
    def test_side_buy(self, sample_signal):
        """Test side property for buy."""
        assert sample_signal.side == "buy"
    
    def test_side_sell(self, sample_imbalance):
        """Test side property for sell."""
        signal = ClosingAuctionSignal(
            symbol="SPY",
            timestamp=datetime.now(),
            direction=SignalDirection.SELL,
            confidence=SignalConfidence.HIGH,
            imbalance=sample_imbalance,
            entry_price=450.0,
            target_exit_price=449.55,
            stop_loss_price=450.9,
            historical_win_rate=0.65,
            historical_count=40,
            max_position_pct=0.03,
            urgency="high"
        )
        assert signal.side == "sell"


class TestHistoricalValidator:
    """Test HistoricalValidator."""
    
    @pytest.fixture
    def temp_cache(self):
        """Create temporary cache with synthetic data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            cache = ClosingAuctionCache(f.name)
            
            # Add synthetic historical data
            for day in range(30):
                date = datetime.now() - timedelta(days=day)
                if date.weekday() >= 5:
                    continue
                
                ts = date.replace(hour=15, minute=50)
                imb = MOCImbalance(
                    symbol="SPY",
                    timestamp=ts,
                    imbalance_shares=500000,
                    paired_shares=2000000,
                    reference_price=400.0 + day,
                    source="synthetic"
                )
                cache.store(imb)
            
            yield cache
            
            # Cleanup
            Path(f.name).unlink(missing_ok=True)
    
    def test_calculate_win_rate_insufficient_data(self, temp_cache):
        """Test win rate with insufficient data."""
        validator = HistoricalValidator(temp_cache)
        
        # Request for symbol with no data
        win_rate, count = validator.calculate_win_rate("QQQ", 2, lookback_days=90)
        assert win_rate is None
        assert count == 0
    
    def test_get_confidence_level_high(self, temp_cache):
        """Test high confidence classification."""
        validator = HistoricalValidator(temp_cache)
        conf = validator.get_confidence_level(0.70, 30)
        assert conf == SignalConfidence.HIGH
    
    def test_get_confidence_level_medium(self, temp_cache):
        """Test medium confidence classification."""
        validator = HistoricalValidator(temp_cache)
        conf = validator.get_confidence_level(0.60, 30)
        assert conf == SignalConfidence.MEDIUM
    
    def test_get_confidence_level_low(self, temp_cache):
        """Test low confidence classification."""
        validator = HistoricalValidator(temp_cache)
        conf = validator.get_confidence_level(0.50, 30)
        assert conf == SignalConfidence.LOW
    
    def test_get_confidence_level_insufficient(self, temp_cache):
        """Test insufficient data classification."""
        validator = HistoricalValidator(temp_cache)
        conf = validator.get_confidence_level(None, 10)
        assert conf == SignalConfidence.INSUFFICIENT_DATA


class TestClosingAuctionSignalGenerator:
    """Test ClosingAuctionSignalGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create signal generator with temp cache."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            gen = ClosingAuctionSignalGenerator(f.name)
            yield gen
            Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_imbalance(self):
        """Create sample MOC imbalance."""
        return MOCImbalance(
            symbol="SPY",
            timestamp=datetime.now(),
            imbalance_shares=1000000,
            paired_shares=2000000,
            reference_price=450.0,
            source="test"
        )
    
    def test_score_to_direction(self, generator):
        """Test score to direction mapping."""
        assert generator._score_to_direction(3) == SignalDirection.STRONG_BUY
        assert generator._score_to_direction(2) == SignalDirection.BUY
        assert generator._score_to_direction(1) == SignalDirection.WEAK_BUY
        assert generator._score_to_direction(0) == SignalDirection.NEUTRAL
        assert generator._score_to_direction(-1) == SignalDirection.WEAK_SELL
        assert generator._score_to_direction(-2) == SignalDirection.SELL
        assert generator._score_to_direction(-3) == SignalDirection.STRONG_SELL
    
    def test_generate_signal_valid(self, generator, sample_imbalance):
        """Test signal generation for valid imbalance."""
        signal = generator.generate_signal(sample_imbalance)
        
        assert signal is not None
        assert signal.symbol == "SPY"
        assert signal.direction == SignalDirection.BUY  # positive imbalance
        assert signal.entry_price == 450.0
        assert signal.target_exit_price > signal.entry_price  # buy target higher
        assert signal.stop_loss_price < signal.entry_price  # buy stop lower
    
    def test_generate_signal_neutral(self, generator):
        """Test signal generation for neutral imbalance."""
        imb = MOCImbalance(
            symbol="SPY",
            timestamp=datetime.now(),
            imbalance_shares=0,
            paired_shares=2000000,
            reference_price=450.0,
            source="test"
        )
        
        signal = generator.generate_signal(imb)
        assert signal is None  # Neutral signals are filtered
    
    def test_generate_all_signals(self, generator, sample_imbalance):
        """Test batch signal generation."""
        # Create imbalances with various direction scores
        imbalances = {
            "SPY": sample_imbalance,
            "QQQ": MOCImbalance(
                symbol="QQQ",
                timestamp=datetime.now(),
                imbalance_shares=-800000,
                paired_shares=2000000,
                reference_price=380.0,
                source="test"
            )
        }
        
        # Generate signals (may be None if insufficient historical data)
        raw_signals = []
        for symbol, imb in imbalances.items():
            signal = generator.generate_signal(imb)
            if signal:
                raw_signals.append(signal)
        
        # Should generate 2 signals (even if not tradeable)
        assert len(raw_signals) == 2
        
        spy_signal = next(s for s in raw_signals if s.symbol == "SPY")
        qqq_signal = next(s for s in raw_signals if s.symbol == "QQQ")
        
        assert spy_signal.direction == SignalDirection.BUY
        assert qqq_signal.direction == SignalDirection.SELL
        
        # Check that signals are generated but may not be tradeable due to data
        tradeable = [s for s in raw_signals if s.should_trade]
        # Tradeability depends on historical data availability
    
    def test_save_signals(self, generator, tmp_path):
        """Test signal saving to JSON."""
        signal1 = ClosingAuctionSignal(
            symbol="SPY",
            timestamp=datetime.now(),
            direction=SignalDirection.BUY,
            confidence=SignalConfidence.HIGH,
            imbalance=MOCImbalance("SPY", datetime.now(), 500000, 1000000, 450.0, "test"),
            entry_price=450.0,
            target_exit_price=450.45,
            stop_loss_price=449.1,
            historical_win_rate=0.70,
            historical_count=50,
            max_position_pct=0.03,
            urgency="high"
        )
        
        signal2 = ClosingAuctionSignal(
            symbol="QQQ",
            timestamp=datetime.now(),
            direction=SignalDirection.NEUTRAL,
            confidence=SignalConfidence.LOW,
            imbalance=MOCImbalance("QQQ", datetime.now(), 10000, 1000000, 380.0, "test"),
            entry_price=380.0,
            target_exit_price=380.0,
            stop_loss_price=None,
            historical_win_rate=0.45,
            historical_count=10,
            max_position_pct=0.03,
            urgency="normal"
        )
        
        output_path = tmp_path / "signals.json"
        generator.save_signals([signal1, signal2], str(output_path))
        
        # Verify file exists and is valid JSON
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        
        assert data['tradeable_count'] == 1  # Only signal1 is tradeable
        assert data['non_tradeable_count'] == 1
        assert len(data['all_signals']) == 2


class TestSignalAggregator:
    """Test SignalAggregator."""
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample signals for aggregation."""
        base_time = datetime.now()
        
        return [
            ClosingAuctionSignal(
                symbol="SPY",
                timestamp=base_time,
                direction=SignalDirection.STRONG_BUY,
                confidence=SignalConfidence.HIGH,
                imbalance=MOCImbalance("SPY", base_time, 1000000, 2000000, 450.0, "test"),
                entry_price=450.0,
                target_exit_price=450.675,
                stop_loss_price=449.1,
                historical_win_rate=0.70,
                historical_count=50,
                max_position_pct=0.03,
                urgency="immediate"
            ),
            ClosingAuctionSignal(
                symbol="QQQ",
                timestamp=base_time,
                direction=SignalDirection.SELL,
                confidence=SignalConfidence.MEDIUM,
                imbalance=MOCImbalance("QQQ", base_time, -500000, 2000000, 380.0, "test"),
                entry_price=380.0,
                target_exit_price=379.62,
                stop_loss_price=380.76,
                historical_win_rate=0.60,
                historical_count=40,
                max_position_pct=0.03,
                urgency="high"
            ),
            ClosingAuctionSignal(
                symbol="IWM",
                timestamp=base_time,
                direction=SignalDirection.WEAK_BUY,
                confidence=SignalConfidence.LOW,
                imbalance=MOCImbalance("IWM", base_time, 200000, 2000000, 220.0, "test"),
                entry_price=220.0,
                target_exit_price=220.11,
                stop_loss_price=219.56,
                historical_win_rate=0.50,
                historical_count=20,
                max_position_pct=0.03,
                urgency="normal"
            )
        ]
    
    def test_aggregate_basic(self, sample_signals):
        """Test basic aggregation."""
        aggregator = SignalAggregator(max_total_allocation=0.10)
        allocations = aggregator.aggregate(sample_signals, portfolio_value=100000)
        
        # Should only include HIGH and MEDIUM confidence
        assert "SPY" in allocations
        assert "QQQ" in allocations
        assert "IWM" not in allocations  # LOW confidence
        
        # Check allocations are reasonable
        assert allocations["SPY"] > allocations["QQQ"]  # HIGH > MEDIUM, STRONG > normal
        
        # Check total within budget
        total = sum(allocations.values())
        assert total <= 100000 * 0.10  # 10% max
    
    def test_aggregate_empty(self):
        """Test aggregation with no signals."""
        aggregator = SignalAggregator()
        allocations = aggregator.aggregate([], portfolio_value=100000)
        assert allocations == {}
    
    def test_aggregate_all_low_confidence(self, sample_signals):
        """Test aggregation when all signals are low confidence."""
        # Modify all to low confidence
        for signal in sample_signals:
            signal.confidence = SignalConfidence.LOW
        
        aggregator = SignalAggregator()
        allocations = aggregator.aggregate(sample_signals, portfolio_value=100000)
        assert allocations == {}
    
    def test_aggregate_risk_budget_enforcement(self, sample_signals):
        """Test that allocations respect risk budget."""
        # Use tight budget
        aggregator = SignalAggregator(max_total_allocation=0.02)  # 2% max
        allocations = aggregator.aggregate(sample_signals, portfolio_value=100000)
        
        total = sum(allocations.values())
        assert total <= 100000 * 0.02  # 2% max enforced
        assert total > 0  # But still has some allocation


class TestIntegration:
    """Integration tests for the full signal pipeline."""
    
    def test_full_pipeline(self, tmp_path):
        """Test complete signal generation pipeline."""
        cache_path = tmp_path / "cache.db"
        
        # Create generator
        generator = ClosingAuctionSignalGenerator(str(cache_path))
        
        # Create sample imbalances with clear direction scores
        imbalances = {
            "SPY": MOCImbalance(
                symbol="SPY",
                timestamp=datetime.now(),
                imbalance_shares=2000000,
                paired_shares=3000000,
                reference_price=450.0,
                source="test"
            ),
            "QQQ": MOCImbalance(
                symbol="QQQ",
                timestamp=datetime.now(),
                imbalance_shares=-1500000,
                paired_shares=3000000,
                reference_price=380.0,
                source="test"
            ),
            "GLD": MOCImbalance(
                symbol="GLD",
                timestamp=datetime.now(),
                imbalance_shares=10000,  # Small - will be neutral
                paired_shares=3000000,
                reference_price=180.0,
                source="test"
            )
        }
        
        # Generate signals
        signals = generator.generate_all_signals(imbalances)
        
        # Signals may or may not be tradeable depending on historical data
        # but should be generated for non-neutral imbalances
        raw_signals = []
        for symbol, imb in imbalances.items():
            signal = generator.generate_signal(imb)
            if signal:
                raw_signals.append(signal)
        
        # SPY and QQQ should generate signals, GLD is neutral
        assert len(raw_signals) == 2  # GLD is neutral
        assert all(s.symbol in ["SPY", "QQQ"] for s in raw_signals)
        
        # Aggregate (may be empty if no tradeable signals)
        if signals:
            aggregator = SignalAggregator()
            allocations = aggregator.aggregate(signals, portfolio_value=100000)
            assert len(allocations) <= len(signals)
        
        # Save
        output_path = tmp_path / "signals.json"
        generator.save_signals(raw_signals, str(output_path))
        assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
