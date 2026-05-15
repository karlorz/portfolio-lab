"""
Tests for VIX Term Structure Overlay Backtest
"""

import json
import pytest
from pathlib import Path
from datetime import datetime

from src.backtest.vix_overlay_backtest import (
    VIXOverlayBacktester,
    BacktestConfig,
    DailyReturn,
    VIXOverlayBacktester  # For signal calculation tests
)


class TestVIXSignalCalculation:
    """Test VIX signal calculation logic."""
    
    def test_extreme_contango_signal(self):
        """Test signal generation in extreme contango regime."""
        backtester = VIXOverlayBacktester()
        
        # VIX3M/VIX > 1.15 should be extreme contango
        regime, signal = backtester.calculate_vix_signal(15.0, 18.0)
        
        assert regime == "extreme_contango"
        assert signal > 0.3  # Positive signal (risk-on)
        assert signal <= 1.0
    
    def test_contango_signal(self):
        """Test signal generation in contango regime."""
        backtester = VIXOverlayBacktester()
        
        # VIX3M/VIX between 1.05 and 1.15
        regime, signal = backtester.calculate_vix_signal(16.0, 17.5)
        ratio = 17.5 / 16.0  # ~1.09
        
        regime, signal = backtester.calculate_vix_signal(16.0, 17.44)
        
        assert regime == "contango"
        assert signal > 0  # Mild positive
        assert signal < 0.5
    
    def test_flat_signal(self):
        """Test signal generation in flat regime."""
        backtester = VIXOverlayBacktester()
        
        # VIX3M/VIX around 1.0
        regime, signal = backtester.calculate_vix_signal(16.0, 15.6)
        
        assert regime == "flat"
        assert -0.3 < signal < 0.3  # Near neutral
    
    def test_backwardation_signal(self):
        """Test signal generation in backwardation regime."""
        backtester = VIXOverlayBacktester()
        
        # VIX3M/VIX < 0.95
        regime, signal = backtester.calculate_vix_signal(18.0, 16.0)
        
        assert regime == "backwardation"
        assert signal < 0  # Negative signal (risk-off)
        assert signal > -0.7
    
    def test_extreme_backwardation_signal(self):
        """Test signal generation in extreme backwardation."""
        backtester = VIXOverlayBacktester()
        
        # VIX3M/VIX < 0.85
        regime, signal = backtester.calculate_vix_signal(20.0, 16.0)
        
        assert regime == "extreme_backwardation"
        assert signal < -0.3  # Strong negative signal
        assert signal >= -1.0
    
    def test_edge_case_zero_vix(self):
        """Test handling of zero/negative VIX values."""
        backtester = VIXOverlayBacktester()
        
        regime, signal = backtester.calculate_vix_signal(0.0, 15.0)
        assert regime == "unknown"
        assert signal == 0.0
        
        regime, signal = backtester.calculate_vix_signal(-5.0, 15.0)
        assert regime == "unknown"
        assert signal == 0.0


class TestAllocationShifts:
    """Test allocation shift calculations."""
    
    def test_extreme_risk_on_shifts(self):
        """Test allocation shifts in extreme risk-on (+1.0 signal)."""
        backtester = VIXOverlayBacktester()
        
        spy, gld, tlt = backtester.get_allocation_shifts(1.0)
        
        assert spy > 0  # Increase equity
        assert gld < 0  # Decrease gold
        assert tlt < 0  # Decrease bonds
        
        # Max shift is 10% for SPY
        assert abs(spy) <= 0.10
    
    def test_extreme_risk_off_shifts(self):
        """Test allocation shifts in extreme risk-off (-1.0 signal)."""
        backtester = VIXOverlayBacktester()
        
        spy, gld, tlt = backtester.get_allocation_shifts(-1.0)
        
        assert spy < 0  # Decrease equity
        assert gld > 0  # Increase gold
        assert tlt > 0  # Increase bonds
        
        # Max shift is 10% for SPY
        assert abs(spy) <= 0.10
    
    def test_neutral_signal_shifts(self):
        """Test allocation shifts at neutral (0.0 signal)."""
        backtester = VIXOverlayBacktester()
        
        spy, gld, tlt = backtester.get_allocation_shifts(0.0)
        
        assert spy == 0
        assert gld == 0
        assert tlt == 0
    
    def test_partial_signal_shifts(self):
        """Test allocation shifts at partial signal strength."""
        backtester = VIXOverlayBacktester()
        
        # Half strength risk-off
        spy, gld, tlt = backtester.get_allocation_shifts(-0.5)
        
        assert spy < 0
        assert gld > 0
        assert tlt > 0
        
        # Should be about half the extreme values
        assert abs(spy) == pytest.approx(0.05, abs=0.001)


class TestBacktestConfig:
    """Test backtest configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig()
        
        assert config.start_date == "2010-01-01"
        assert config.end_date == "2026-05-15"
        assert config.initial_capital == 100000.0
        assert config.base_spy_weight == 0.46
        assert config.base_gld_weight == 0.38
        assert config.base_tlt_weight == 0.16
        assert config.transaction_cost_bps == 10.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            start_date="2020-01-01",
            end_date="2022-12-31",
            initial_capital=50000.0,
            base_spy_weight=0.50,
            base_gld_weight=0.30,
            base_tlt_weight=0.20
        )
        
        assert config.start_date == "2020-01-01"
        assert config.end_date == "2022-12-31"
        assert config.initial_capital == 50000.0
        assert config.base_spy_weight == 0.50


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_data_no_file(self, tmp_path):
        """Test handling when no data file exists."""
        backtester = VIXOverlayBacktester()
        
        # Temporarily change directory to tmp_path with no data
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        result = backtester.load_data()
        
        os.chdir(original_dir)
        
        assert result is False
    
    def test_process_price_data(self):
        """Test price data processing."""
        backtester = VIXOverlayBacktester()
        
        # Mock price data in list format (matches actual data structure)
        mock_data = {
            "SPY": [
                {"d": "2024-01-01", "p": 100.0},
                {"d": "2024-01-02", "p": 101.0},
                {"d": "2024-01-03", "p": 100.5},
            ],
            "GLD": [
                {"d": "2024-01-01", "p": 180.0},
                {"d": "2024-01-02", "p": 181.8},
                {"d": "2024-01-03", "p": 181.0},
            ],
            "TLT": [
                {"d": "2024-01-01", "p": 90.0},
                {"d": "2024-01-02", "p": 89.1},
                {"d": "2024-01-03", "p": 89.5},
            ]
        }
        
        backtester._process_price_data(mock_data)
        
        assert len(backtester.data) == 2  # 2 return periods from 3 price points
        
        # Check first day's returns
        day1 = backtester.data[0]
        assert day1.date == "2024-01-02"
        assert pytest.approx(day1.spy_return, abs=0.001) == 0.01  # 1% gain
        assert pytest.approx(day1.gld_return, abs=0.001) == 0.01  # 1% gain
        assert pytest.approx(day1.tlt_return, abs=0.001) == -0.01  # -1% loss


class TestBacktestExecution:
    """Test backtest execution."""
    
    def test_backtest_without_data(self):
        """Test backtest fails gracefully without data."""
        backtester = VIXOverlayBacktester()
        
        result = backtester.run_backtest()
        
        assert result is None
    
    def test_backtest_with_sample_data(self):
        """Test backtest runs with sample data."""
        backtester = VIXOverlayBacktester(
            config=BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-02-01"
            )
        )
        
        # Create sample data
        for i in range(22):  # ~1 month of trading days
            date = f"2024-01-{i+1:02d}"
            backtester.data.append(DailyReturn(
                date=date,
                spy_return=0.001,  # 0.1% daily
                gld_return=0.0005,
                tlt_return=0.0003
            ))
        
        result = backtester.run_backtest()
        
        assert result is not None
        assert result.total_return > 0  # Should have positive returns
        assert result.sharpe_ratio > 0


class TestMetricsCalculation:
    """Test performance metrics calculation."""
    
    def test_calculate_returns_from_equity(self):
        """Test returns calculation from equity curve."""
        backtester = VIXOverlayBacktester()
        
        equity = [100.0, 101.0, 99.0, 102.0]
        returns = backtester._calculate_returns_from_equity(equity)
        
        assert len(returns) == 3
        assert pytest.approx(returns[0], abs=0.001) == 0.01  # 1% gain
        assert pytest.approx(returns[1], abs=0.001) == -0.0198  # ~-2% loss
        assert pytest.approx(returns[2], abs=0.001) == 0.0303  # ~3% gain
    
    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        backtester = VIXOverlayBacktester()
        
        # 10% annual return, low volatility
        daily_ret = 0.10 / 252  # ~10% annual
        returns = [daily_ret] * 252
        
        metrics = backtester._calculate_metrics(returns)
        
        assert "cagr" in metrics
        assert "volatility" in metrics
        assert "sharpe" in metrics
        assert "max_dd" in metrics
        
        # Should have positive CAGR
        assert metrics["cagr"] > 0
    
    def test_calculate_metrics_with_drawdown(self):
        """Test max drawdown calculation."""
        backtester = VIXOverlayBacktester()
        
        # Create returns with a drawdown
        returns = [0.01] * 10 + [-0.05] * 3 + [0.01] * 10
        
        metrics = backtester._calculate_metrics(returns)
        
        assert metrics["max_dd"] < 0  # Negative drawdown


class TestResultsSaving:
    """Test results saving functionality."""
    
    def test_save_results_creates_file(self, tmp_path):
        """Test that save_results creates output file."""
        from src.backtest.vix_overlay_backtest import BacktestResult
        
        backtester = VIXOverlayBacktester()
        
        result = BacktestResult(
            total_return=100.0,
            cagr=10.0,
            volatility=12.0,
            sharpe_ratio=0.83,
            max_drawdown=-15.0,
            overlay_active_days=100,
            baseline_sharpe=0.79,
            sharpe_improvement=0.04,
            return_2008=-5.0,
            return_2020=8.0,
            return_2022=-3.0,
            total_rebalances=50,
            avg_rebalance_size=0.03,
            total_transaction_costs=100.0,
            regime_returns={"contango": 10.0, "backwardation": 5.0},
            equity_curve=[{"date": "2024-01-01", "baseline": 100, "overlay": 102}]
        )
        
        output_file = tmp_path / "test_results.json"
        backtester.save_results(result, str(output_file))
        
        assert output_file.exists()
        
        # Verify JSON content
        with open(output_file) as f:
            data = json.load(f)
            assert data["sharpe_ratio"] == 0.83
            assert data["sharpe_improvement"] == 0.04


class TestRegimeClassification:
    """Test VIX regime classification edge cases."""
    
    @pytest.mark.parametrize("vix,vix3m,expected_regime", [
        (15.0, 18.0, "extreme_contango"),  # 1.2 ratio
        (16.0, 17.5, "contango"),  # ~1.09 ratio
        (16.0, 16.0, "flat"),  # 1.0 ratio
        (16.0, 15.5, "flat"),  # ~0.97 ratio
        (18.0, 16.0, "backwardation"),  # ~0.89 ratio
        (20.0, 16.0, "extreme_backwardation"),  # 0.8 ratio
    ])
    def test_regime_boundaries(self, vix, vix3m, expected_regime):
        """Test regime classification at various ratio boundaries."""
        backtester = VIXOverlayBacktester()
        
        regime, _ = backtester.calculate_vix_signal(vix, vix3m)
        
        assert regime == expected_regime


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
