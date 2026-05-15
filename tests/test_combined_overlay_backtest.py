"""
Tests for Combined Overlay Backtest (v4.90)
"""

import json
import pytest
import numpy as np
from datetime import date
from pathlib import Path

from src.backtest.combined_overlay_backtest import (
    CombinedOverlayBacktest,
    BacktestResult,
    run_combined_backtest,
)


class TestBacktestResult:
    """Test backtest result dataclass."""

    def test_serializable(self):
        result = BacktestResult(
            timestamp="2026-05-16T00:00:00",
            start_date="2006-01-03", end_date="2026-05-15",
            trading_days=5000,
            baseline_cagr=10.6, baseline_vol=11.1, baseline_sharpe=0.79,
            baseline_max_dd=-26.2,
            baseline_crisis_2008=-12.3, baseline_crisis_2020=-7.1,
            baseline_crisis_2022=-13.0,
            combined_cagr=12.0, combined_vol=11.0, combined_sharpe=0.91,
            combined_max_dd=-21.0,
            combined_crisis_2008=-10.0, combined_crisis_2020=-5.0,
            combined_crisis_2022=-10.0,
            sharpe_delta=0.12, dd_improvement=5.2, cagr_delta=1.4,
            collar_active_pct=65.0, crypto_active_pct=40.0,
            bond_rotation_avg_tlt=45.0, avg_overlays_active=2.5,
            meets_sharpe_target=True, meets_dd_target=True,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["combined_sharpe"] == 0.91
        assert d["meets_sharpe_target"]


class TestCombinedOverlayBacktest:
    """Test backtest core functionality."""

    @pytest.fixture
    def bt(self):
        return CombinedOverlayBacktest()

    def test_collar_signal_normal(self, bt):
        delta = bt._collar_signal(16.0, 0.001)
        assert delta == 0.0

    def test_collar_signal_elevated(self, bt):
        delta = bt._collar_signal(25.0, 0.001)
        assert delta == -0.01

    def test_collar_signal_stress(self, bt):
        delta = bt._collar_signal(35.0, 0.001)
        assert delta == -0.03

    def test_collar_signal_crisis(self, bt):
        delta = bt._collar_signal(50.0, 0.001)
        assert delta == -0.05

    def test_bond_signal_steep_falling(self, bt):
        tlt, ief, shy = bt._bond_duration_signal(1.5, -0.5)
        assert tlt > ief
        assert tlt > shy

    def test_bond_signal_inverted_rising(self, bt):
        tlt, ief, shy = bt._bond_duration_signal(-0.5, 0.5)
        assert shy > tlt
        assert shy > ief

    def test_bond_signal_normal(self, bt):
        tlt, ief, shy = bt._bond_duration_signal(0.5, 0.0)
        assert ief > tlt  # Balanced leans intermediate

    def test_bond_weights_sum_to_one(self, bt):
        for spread in [-1.0, 0.0, 0.5, 1.5]:
            for change in [-1.0, 0.0, 1.0]:
                tlt, ief, shy = bt._bond_duration_signal(spread, change)
                assert abs(tlt + ief + shy - 1.0) < 0.01

    def test_crypto_signal_extreme_vol(self, bt):
        w = bt._crypto_signal(0.5, 1.5, 0.3, 0.9)
        assert w == 0.0  # BTC vol extreme

    def test_crypto_signal_bear(self, bt):
        w = bt._crypto_signal(-0.3, 0.6, -0.2, 0.7)
        assert w == 0.0  # Both negative momentum

    def test_crypto_signal_bull(self, bt):
        w = bt._crypto_signal(0.5, 0.6, 0.3, 0.7)
        assert w > 0

    def test_crypto_weight_capped(self, bt):
        w = bt._crypto_signal(2.0, 0.3, 2.0, 0.3)
        assert w <= 0.05

    def test_run_backtest(self, bt):
        result = bt.run_backtest()
        assert isinstance(result, BacktestResult)
        assert result.trading_days > 0
        assert result.baseline_sharpe != 0
        assert result.combined_sharpe != 0

    def test_run_backtest_has_crisis_data(self, bt):
        result = bt.run_backtest()
        # At least one crisis period should have data
        any_crisis = (
            result.baseline_crisis_2008 != 0 or
            result.baseline_crisis_2020 != 0 or
            result.baseline_crisis_2022 != 0
        )
        assert any_crisis, "At least one crisis period should be captured"

    def test_sharpe_delta_reasonable(self, bt):
        result = bt.run_backtest()
        # Combined should be better or within noise
        assert result.sharpe_delta > -0.05

    def test_overlay_activity_tracked(self, bt):
        result = bt.run_backtest()
        assert 0 <= result.collar_active_pct <= 100
        assert 0 <= result.crypto_active_pct <= 100
        assert 0 <= result.bond_rotation_avg_tlt <= 100

    def test_convenience_function(self):
        result = run_combined_backtest()
        assert isinstance(result, BacktestResult)


class TestEdgeCases:
    """Edge cases for backtest."""

    def test_compute_returns(self):
        bt = CombinedOverlayBacktest()
        rets = bt._compute_returns([100, 110, 105, 115])
        assert len(rets) == 3
        assert abs(rets[0] - 0.10) < 0.01
        assert rets[1] < 0

    def test_compute_rolling_vol(self):
        bt = CombinedOverlayBacktest()
        rng = np.random.RandomState(42)
        rets = list(rng.normal(0, 0.01, 100))
        vols = bt._compute_rolling_vol(rets, 30)
        assert len(vols) == len(rets)
        assert vols[-1] > 0
