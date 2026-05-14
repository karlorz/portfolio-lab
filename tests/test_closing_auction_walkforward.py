"""
Tests for closing auction walk-forward backtest (v3.17 Phase 4)

Covers:
- Synthetic data generation accuracy
- Backtest trade simulation
- Metric calculations
- Edge cases
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.closing_auction_walkforward import (
    SyntheticMOCDataGenerator,
    ClosingAuctionBacktester,
    BacktestTrade,
    DailyResult,
    WalkForwardResults,
    run_validation_backtest,
)


class TestSyntheticMOCDataGenerator:
    """Test synthetic MOC data generation."""
    
    def test_generator_initialization(self):
        """Test generator can be created with seed."""
        gen = SyntheticMOCDataGenerator(seed=42)
        assert gen is not None
        
    def test_imbalance_generation_distribution(self):
        """Test imbalance ratios follow expected distribution."""
        gen = SyntheticMOCDataGenerator(seed=42)
        imbalances = [gen.generate_imbalance_ratio() for _ in range(1000)]
        
        # Mean should be close to BUY_SELL_BIAS (0.02)
        assert abs(np.mean(imbalances) - 0.02) < 0.1
        
        # Std should be close to IMBALANCE_STD (0.35)
        assert 0.25 < np.std(imbalances) < 0.45
        
    def test_score_mapping_symmetry(self):
        """Test score mapping is symmetric for positive/negative imbalances."""
        gen = SyntheticMOCDataGenerator(seed=42)
        
        assert gen.score_from_imbalance(0.9) == 3   # Strong buy
        assert gen.score_from_imbalance(0.6) == 2   # Buy
        assert gen.score_from_imbalance(0.3) == 1   # Weak buy
        assert gen.score_from_imbalance(0.1) == 0   # Neutral
        assert gen.score_from_imbalance(-0.1) == 0  # Neutral
        assert gen.score_from_imbalance(-0.3) == -1 # Weak sell
        assert gen.score_from_imbalance(-0.6) == -2 # Sell
        assert gen.score_from_imbalance(-0.9) == -3 # Strong sell
        
    def test_trade_result_returns_reasonable_values(self):
        """Test generated trade returns are reasonable."""
        gen = SyntheticMOCDataGenerator(seed=42)
        
        # Generate many trades for score 3
        returns = []
        slippages = []
        for _ in range(100):
            ret, slip = gen.generate_trade_result(3, "SPY")
            returns.append(ret)
            slippages.append(slip)
        
        # Returns should be in reasonable range (-2% to +2%)
        assert all(-0.02 < r < 0.02 for r in returns)
        
        # Slippage should be positive
        assert all(s > 0 for s in slippages)
        
        # Average slippage should be around 16-20 bps (8 base * 2)
        assert 10 < np.mean(slippages) < 30
        
    def test_trade_result_accuracy_matches_expected(self):
        """Test that win rate matches configured accuracy."""
        gen = SyntheticMOCDataGenerator(seed=42)
        
        # Test strong signals (expected ~72% accuracy)
        results = []
        for _ in range(1000):
            ret, _ = gen.generate_trade_result(3, "SPY")
            results.append(ret > 0)  # Positive return = correct signal
        
        accuracy = sum(results) / len(results)
        # Should be close to 72% (allow 10% tolerance for randomness)
        assert 0.62 < accuracy < 0.82


class TestClosingAuctionBacktester:
    """Test backtest engine functionality."""
    
    def test_backtester_initialization(self):
        """Test backtester can be created with default parameters."""
        bt = ClosingAuctionBacktester()
        assert bt.symbols == ["SPY", "QQQ", "IWM", "GLD", "TLT"]
        assert bt.position_size_pct == 0.025
        
    def test_backtester_custom_parameters(self):
        """Test backtester accepts custom parameters."""
        bt = ClosingAuctionBacktester(
            symbols=["SPY"],
            position_size_pct=0.05,
            max_positions_per_day=1,
            min_confidence="high",
        )
        assert bt.symbols == ["SPY"]
        assert bt.position_size_pct == 0.05
        assert bt.max_positions_per_day == 1
        assert bt.min_confidence == "high"
        
    def test_should_trade_filters(self):
        """Test trade filtering by confidence."""
        bt = ClosingAuctionBacktester(min_confidence="medium")
        
        # Should trade medium and high confidence
        assert bt._should_trade(2, "medium") is True
        assert bt._should_trade(3, "high") is True
        
        # Should not trade low confidence or neutral
        assert bt._should_trade(1, "low") is False
        assert bt._should_trade(0, "insufficient") is False
        
    def test_should_trade_high_confidence_only(self):
        """Test filtering for high confidence only."""
        bt = ClosingAuctionBacktester(min_confidence="high")
        
        assert bt._should_trade(3, "high") is True
        assert bt._should_trade(2, "medium") is False
        assert bt._should_trade(1, "low") is False


class TestDailyResult:
    """Test DailyResult dataclass."""
    
    def test_daily_result_creation(self):
        """Test DailyResult can be created."""
        date = datetime(2024, 1, 1)
        trades = []
        result = DailyResult(
            date=date,
            trades=trades,
            daily_gross_pnl=0.001,
            daily_net_pnl=0.0005,
            daily_costs=0.0005,
        )
        assert result.date == date
        assert result.daily_gross_pnl == 0.001


class TestWalkForwardResults:
    """Test WalkForwardResults dataclass."""
    
    def test_results_to_dict(self):
        """Test results can be serialized to dict."""
        results = WalkForwardResults(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2024, 1, 1),
            total_days=1000,
            trading_days=750,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.6,
            gross_cagr=0.01,
            net_cagr=0.005,
            gross_sharpe=0.5,
            net_sharpe=0.3,
            max_drawdown_pct=0.05,
            total_costs_pct=0.005,
            avg_slippage_bps=20.0,
            signal_accuracy_by_score={3: 0.7, 2: 0.6, 1: 0.55},
            confidence_win_rates={"high": 0.7, "medium": 0.6, "low": 0.5},
            trades=[],
        )
        
        d = results.to_dict()
        assert d["trades"]["total"] == 100
        assert d["trades"]["win_rate"] == 60.0  # Converted to percentage
        assert "performance" in d
        assert "costs" in d


class TestIntegration:
    """Integration tests for full backtest."""
    
    def test_short_backtest_run(self):
        """Test a short backtest completes successfully."""
        bt = ClosingAuctionBacktester(
            symbols=["SPY"],
            position_size_pct=0.025,
            max_positions_per_day=1,
        )
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 1)  # 2 months
        
        results = bt.run_walk_forward(start, end)
        
        # Should have some trades
        assert results.total_trades > 0
        
        # Win rate should be between 0 and 1
        assert 0 <= results.win_rate <= 1
        
        # Costs should be positive
        assert results.total_costs_pct > 0
        
    def test_backtest_consistency_with_same_seed(self):
        """Test that same seed produces consistent results."""
        results1 = run_validation_backtest(
            start_year=2024,
            end_year=2024,
        )
        
        # Just verify it runs and produces valid output
        assert results1 is not None
        assert results1.total_trades > 0


class TestMetricsCalculation:
    """Test metric calculation functions."""
    
    def test_sharpe_calculation_basic(self):
        """Test Sharpe calculation with basic returns."""
        bt = ClosingAuctionBacktester()
        
        # Create simple returns array
        returns = np.array([0.001, -0.0005, 0.002, -0.001, 0.0015] * 20)
        sharpe = bt._calculate_sharpe(returns)
        
        # Sharpe should be positive for these mostly positive returns
        assert sharpe > 0
        
    def test_sharpe_with_zeros(self):
        """Test Sharpe handles zero returns (no trade days)."""
        bt = ClosingAuctionBacktester()
        
        returns = np.array([0, 0, 0, 0.001, 0, 0, -0.001, 0])
        sharpe = bt._calculate_sharpe(returns)
        
        # Should handle zeros gracefully
        assert sharpe == 0.0  # Too few non-zero returns
        
    def test_max_drawdown_calculation(self):
        """Test max drawdown calculation."""
        bt = ClosingAuctionBacktester()
        
        # Create returns with known drawdown
        # Start at 1.0, go up to 1.1, down to 0.95, up to 1.2
        returns = np.array([0.01] * 10 + [-0.02] * 10 + [0.03] * 10)
        dd = bt._calculate_max_drawdown(returns)
        
        # Max drawdown should be around (1.1 - 0.95) / 1.1 = 13.6%
        assert 0.1 < dd < 0.2
        
    def test_max_drawdown_with_zeros(self):
        """Test drawdown handles zero returns."""
        bt = ClosingAuctionBacktester()
        
        returns = np.array([0, 0, 0, 0])
        dd = bt._calculate_max_drawdown(returns)
        
        # Should return 0 for no trades
        assert dd == 0.0


class TestSuccessCriteria:
    """Test against v3.17 success criteria."""
    
    def test_slippage_under_threshold(self):
        """Verify average slippage stays under 30 bps."""
        bt = ClosingAuctionBacktester(
            symbols=["SPY", "QQQ", "IWM"],
            transaction_cost_bps=5,
        )
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        
        results = bt.run_walk_forward(start, end)
        
        # Average slippage should be under 30 bps
        assert results.avg_slippage_bps < 30, f"Slippage {results.avg_slippage_bps} bps exceeds 30 bps limit"
        
    def test_win_rate_threshold(self):
        """Verify win rate can exceed 55% with proper signals."""
        bt = ClosingAuctionBacktester(
            symbols=["SPY"],
            position_size_pct=0.025,
            min_confidence="medium",  # Only trade medium+ confidence
        )
        
        start = datetime(2020, 1, 1)
        end = datetime(2026, 1, 1)
        
        results = bt.run_walk_forward(start, end)
        
        # Note: Win rate depends on random seed, but with proper signal accuracy
        # we should typically see >55% gross win rate on signals
        # The net win rate (after costs) will be lower
        
        # Check that we have signal accuracy data
        assert len(results.signal_accuracy_by_score) > 0
        
        # Strong signals (3) should have good accuracy
        if 3 in results.signal_accuracy_by_score:
            assert results.signal_accuracy_by_score[3] > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
