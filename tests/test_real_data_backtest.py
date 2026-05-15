"""
Tests for Real Data Combined Backtest (v4.90)
"""

import json
import pytest
import numpy as np
from pathlib import Path

from src.backtest.real_data_backtest import (
    RealDataBacktest,
    RealDataBacktestResult,
    run_real_data_backtest,
)


class TestRealDataBacktestResult:
    """Test result dataclass."""

    def test_serializable(self):
        result = RealDataBacktestResult(
            timestamp="2026-05-16", data_start="2021-05-10",
            data_end="2026-05-15", trading_days=1200,
            baseline_cagr=14.8, baseline_vol=12.3, baseline_sharpe=1.204,
            baseline_max_dd=-19.1, baseline_total_return=82.2,
            collar_sharpe=1.22, collar_dd=-16.5,
            crypto_sharpe=1.22, bond_dur_sharpe=1.22,
            combined_cagr=15.3, combined_vol=11.6, combined_sharpe=1.318,
            combined_max_dd=-16.6, combined_total_return=87.1,
            sharpe_delta=0.113, dd_improvement=2.5,
            collar_days_pct=16.0, crypto_days_pct=55.0,
            avg_tlt_sleeve_pct=16.0,
            meets_target=True,
            recommendation="Test recommendation",
        )
        d = result.to_dict()
        assert d["baseline_sharpe"] == 1.204
        assert d["meets_target"]


class TestRealDataBacktest:
    """Test real data backtest."""

    @pytest.fixture
    def bt(self):
        return RealDataBacktest()

    def test_compute_returns(self, bt):
        rets = bt._compute_returns([100, 110, 105])
        assert len(rets) == 2
        assert abs(rets[0] - 0.10) < 0.01

    def test_compute_rolling_vol(self, bt):
        rng = np.random.RandomState(42)
        rets = list(rng.normal(0, 0.01, 100))
        vols = bt._compute_rolling_vol(rets, 30)
        assert len(vols) == len(rets)

    def test_collar_signals(self, bt):
        assert bt._collar_signal(15.0) == 0.0
        assert bt._collar_signal(26.0) == -0.01
        assert bt._collar_signal(35.0) == -0.03
        assert bt._collar_signal(50.0) == -0.05

    def test_bond_duration_signals(self, bt):
        t, i, s = bt._bond_duration_signal(0.15, 0)
        assert t > i  # Strong TLT rally → heavy TLT

        t, i, s = bt._bond_duration_signal(-0.15, 0)
        assert s > t  # TLT decline → heavy SHY

    def test_bond_weights_sum_to_one(self, bt):
        for mom in [-0.20, -0.05, 0.0, 0.05, 0.20]:
            t, i, s = bt._bond_duration_signal(mom, 0)
            assert abs(t + i + s - 1.0) < 0.01

    def test_crypto_signal_bull(self, bt):
        w = bt._crypto_signal(0.5, 0.3, 0.6, 0.7)
        assert w > 0

    def test_crypto_signal_extreme_vol(self, bt):
        w = bt._crypto_signal(0.5, 0.3, 1.5, 0.7)
        assert w == 0.0

    def test_crypto_signal_bear(self, bt):
        w = bt._crypto_signal(-0.3, -0.2, 0.6, 0.7)
        assert w == 0.0

    def test_crypto_weight_capped(self, bt):
        w = bt._crypto_signal(3.0, 3.0, 0.3, 0.3)
        assert w <= 0.05

    def test_run_with_real_data(self, bt):
        """Should work when market.db is available."""
        result = bt.run()
        assert isinstance(result, RealDataBacktestResult)
        # If data loaded successfully
        if result.trading_days > 0:
            assert result.baseline_sharpe != 0
            assert result.combined_sharpe != 0
            assert result.recommendation is not None

    def test_convenience_function(self):
        result = run_real_data_backtest()
        assert isinstance(result, RealDataBacktestResult)


class TestEdgeCases:
    """Edge cases."""

    def test_no_data_returns_safe_result(self):
        bt = RealDataBacktest()
        # Point to non-existent path
        bt.DATA_DIR = Path("/nonexistent")
        result = bt.run()
        assert result.trading_days == 0
        assert "No data" in result.recommendation
