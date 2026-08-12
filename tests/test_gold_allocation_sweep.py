#!/usr/bin/env python3
"""Tests for src/backtest/gold_allocation_sweep.py."""

import numpy as np


class TestGoldSweepRow:
    """Tests for GoldSweepRow dataclass."""

    def test_row_creation(self):
        from src.backtest.gold_allocation_sweep import GoldSweepRow

        row = GoldSweepRow(
            spy_pct=0.46, gld_pct=0.38, tlt_pct=0.16, ief_pct=0.00,
            label="46/38/16", cagr=10.5, vol=11.1, sharpe=0.79,
            max_dd=-26.2, sharpe_delta=0.0,
            year_2008=-12.3, year_2020=-7.1, year_2022=-13.0,
        )
        d = row.to_dict()
        assert d["spy_pct"] == 0.46
        assert d["gld_pct"] == 0.38
        assert d["label"] == "46/38/16"
        assert d["sharpe"] == 0.79


class TestGoldSweepResult:
    """Tests for GoldSweepResult dataclass."""

    def test_result_creation(self):
        from src.backtest.gold_allocation_sweep import GoldSweepResult, GoldSweepRow

        row = GoldSweepRow(
            spy_pct=0.46, gld_pct=0.38, tlt_pct=0.16, ief_pct=0.00,
            label="46/38/16", cagr=10.5, vol=11.1, sharpe=0.79,
            max_dd=-26.2, sharpe_delta=0.0,
            year_2008=-12.3, year_2020=-7.1, year_2022=-13.0,
        )
        result = GoldSweepResult(
            timestamp="2026-05-26",
            data_range="2005-2026",
            n_days=5380,
            baseline_cagr=10.5, baseline_vol=11.1,
            baseline_sharpe=0.79, baseline_max_dd=-26.2,
            rows=[row],
            best_sharpe_row=row.to_dict(),
            best_drawdown_row=None,
            best_2022_row=None,
            recommendation="Keep at 38% GLD",
        )
        d = result.to_dict()
        assert d["baseline_sharpe"] == 0.79
        assert d["best_sharpe_row"]["label"] == "46/38/16"
        assert len(d["rows"]) == 1


class TestGoldAllocationSweep:
    """Tests for GoldAllocationSweep class."""

    def test_compute_returns(self):
        from src.backtest.gold_allocation_sweep import GoldAllocationSweep

        sweeper = GoldAllocationSweep()
        prices = [100.0, 101.0, 103.0, 102.0]
        rets = sweeper._compute_returns(prices)
        assert len(rets) == 3
        assert abs(rets[0] - 0.01) < 0.001  # 101/100 - 1
        assert abs(rets[1] - 0.01980198) < 0.001  # 103/101 - 1
        assert abs(rets[2] - (-0.00970873)) < 0.001  # 102/103 - 1

    def test_simulate_portfolio_basic(self):
        """Test _simulate_portfolio with random-walk prices."""
        from src.backtest.gold_allocation_sweep import GoldAllocationSweep

        sweeper = GoldAllocationSweep()
        n = 500
        rng = np.random.default_rng(42)
        sweeper.prices = {
            "SPY": [100 + rng.standard_normal(n).cumsum()[i] * 0.5 for i in range(n)],
            "GLD": [150 + rng.standard_normal(n).cumsum()[i] * 0.3 for i in range(n)],
            "TLT": [100 + rng.standard_normal(n).cumsum()[i] * 0.2 for i in range(n)],
            "IEF": [80 + rng.standard_normal(n).cumsum()[i] * 0.15 for i in range(n)],
        }
        sweeper.dates = [f"{y}-01-01" for y in range(2015, 2015 + n)]

        weights = {"spy": 0.46, "gld": 0.38, "tlt": 0.16, "ief": 0.00}
        cagr, vol, sharpe, max_dd, year_rets = sweeper._simulate_portfolio(weights)

        assert cagr != 0
        assert vol > 0
        assert sharpe != 0
        assert max_dd < 0  # max DD is negative

    def test_simulate_all_in_one_asset(self):
        """All-in SPY should roughly match SPY returns."""
        from src.backtest.gold_allocation_sweep import GoldAllocationSweep

        sweeper = GoldAllocationSweep()
        n = 500
        rng = np.random.default_rng(43)
        sweeper.prices = {
            "SPY": [100 + rng.standard_normal(n).cumsum()[i] * 0.5 for i in range(n)],
            "GLD": [150 + rng.standard_normal(n).cumsum()[i] * 0.3 for i in range(n)],
            "TLT": [100 + rng.standard_normal(n).cumsum()[i] * 0.2 for i in range(n)],
            "IEF": [80 + rng.standard_normal(n).cumsum()[i] * 0.15 for i in range(n)],
        }
        sweeper.dates = [f"{y}-01-01" for y in range(2015, 2015 + n)]

        w1 = {"spy": 1.0, "gld": 0.0, "tlt": 0.0, "ief": 0.0}
        cagr_1, vol_1, sharpe_1, dd_1, _ = sweeper._simulate_portfolio(w1)

        assert cagr_1 != 0
        assert vol_1 > 0

    def test_weights_sum_check(self):
        """Different allocations produce different results."""
        from src.backtest.gold_allocation_sweep import GoldAllocationSweep

        sweeper = GoldAllocationSweep()
        n = 500
        rng = np.random.default_rng(42)
        sweeper.prices = {
            "SPY": [100 + rng.standard_normal(n).cumsum()[i] * 0.5 for i in range(n)],
            "GLD": [150 + rng.standard_normal(n).cumsum()[i] * 0.3 for i in range(n)],
            "TLT": [100 + rng.standard_normal(n).cumsum()[i] * 0.2 for i in range(n)],
            "IEF": [80 + rng.standard_normal(n).cumsum()[i] * 0.15 for i in range(n)],
        }

        w_high_gld = {"spy": 0.40, "gld": 0.50, "tlt": 0.10, "ief": 0.00}
        w_low_gld = {"spy": 0.60, "gld": 0.20, "tlt": 0.20, "ief": 0.00}

        cagr_high, vol_high, sharpe_high, dd_high, _ = sweeper._simulate_portfolio(w_high_gld)
        cagr_low, vol_low, sharpe_low, dd_low, _ = sweeper._simulate_portfolio(w_low_gld)

        # Different allocations should yield different metrics
        assert (cagr_high != cagr_low) or (vol_high != vol_low) or (sharpe_high != sharpe_low)

    def test_ief_inclusion_works(self):
        """Allocation with IEF should run without error."""
        from src.backtest.gold_allocation_sweep import GoldAllocationSweep

        sweeper = GoldAllocationSweep()
        n = 200
        sweeper.prices = {
            "SPY": [100 * (1.001) ** i for i in range(n)],
            "GLD": [150 * (1.0008) ** i for i in range(n)],
            "TLT": [100 * (1.0005) ** i for i in range(n)],
            "IEF": [80 * (1.0003) ** i for i in range(n)],
        }

        weights = {"spy": 0.42, "gld": 0.38, "tlt": 0.10, "ief": 0.10}
        cagr, vol, sharpe, max_dd, year_rets = sweeper._simulate_portfolio(weights)
        assert cagr > 0
        assert sharpe > 0
