"""
Tests for DBC Commodity Weight Sweep (v4.90)
"""

import json
import pytest
import numpy as np
from pathlib import Path

from src.backtest.dbc_weight_sweep import (
    DBCWeightSweep,
    DBCSweepResult,
    DBCSweepRow,
    run_dbc_sweep,
)


class TestDBCSweepRow:
    """Test sweep row dataclass."""

    def test_serializable(self):
        row = DBCSweepRow(
            dbc_weight=0.03, funded_from="gld",
            cagr=10.8, vol=11.0, sharpe=0.82, max_dd=-25.0,
            sharpe_delta=0.03, crisis_2008=-11.0,
            crisis_2020=-6.0, crisis_2022=-12.0,
            avg_dbc_return=5.5,
        )
        d = row.to_dict()
        assert d["dbc_weight"] == 0.03
        assert d["funded_from"] == "gld"
        assert d["sharpe_delta"] == 0.03


class TestDBCWeightSweep:
    """Test sweep core functionality."""

    @pytest.fixture
    def sweep(self):
        return DBCWeightSweep()

    def test_generates_data(self, sweep):
        data = sweep._generate_test_data()
        assert "SPY" in data
        assert "DBC" in data
        assert len(data["SPY"]) > 100

    def test_compute_returns(self, sweep):
        rets = sweep._compute_returns([100, 110, 105, 115])
        assert len(rets) == 3
        assert abs(rets[0] - 0.10) < 0.01

    def test_baseline_portfolio(self, sweep):
        data = sweep._generate_test_data()
        spy_r = sweep._compute_returns(data["SPY"])
        gld_r = sweep._compute_returns(data["GLD"])
        tlt_r = sweep._compute_returns(data["TLT"])
        dbc_r = sweep._compute_returns(data["DBC"])

        cagr, vol, sharpe, dd = sweep._simulate_portfolio(
            spy_r, gld_r, tlt_r, dbc_r, 0.0, "gld"
        )
        assert cagr != 0
        assert vol > 0
        assert sharpe > 0

    def test_dbc_reduces_gld(self, sweep):
        """DBC funded from GLD should reduce GLD by exact weight."""
        data = sweep._generate_test_data()
        spy_r = sweep._compute_returns(data["SPY"])
        gld_r = sweep._compute_returns(data["GLD"])
        tlt_r = sweep._compute_returns(data["TLT"])
        dbc_r = sweep._compute_returns(data["DBC"])

        # With 3% DBC from GLD, portfolio should differ from baseline
        c_no_dbc, _, s_no_dbc, _ = sweep._simulate_portfolio(
            spy_r, gld_r, tlt_r, dbc_r, 0.0, "gld"
        )
        c_with_dbc, _, s_with_dbc, _ = sweep._simulate_portfolio(
            spy_r, gld_r, tlt_r, dbc_r, 0.03, "gld"
        )
        # Results should differ
        assert c_no_dbc != c_with_dbc or s_no_dbc != s_with_dbc

    def test_run_sweep(self, sweep):
        result = sweep.run_sweep()
        assert isinstance(result, DBCSweepResult)
        assert len(result.rows) == 18  # 6 weights × 3 sources
        assert result.baseline_sharpe > 0

    def test_sweep_has_all_weights(self, sweep):
        result = sweep.run_sweep()
        weights = sorted(set(r.dbc_weight for r in result.rows))
        assert weights == [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

    def test_sweep_has_all_sources(self, sweep):
        result = sweep.run_sweep()
        sources = sorted(set(r.funded_from for r in result.rows))
        assert sources == ["gld", "spy", "tlt"]

    def test_best_weight_valid(self, sweep):
        result = sweep.run_sweep()
        assert 0.0 <= result.best_weight <= 0.06
        assert result.best_source in ("gld", "spy", "tlt", "none")

    def test_recommendation_non_empty(self, sweep):
        result = sweep.run_sweep()
        assert len(result.recommendation) > 0

    def test_result_serializable(self, sweep):
        result = sweep.run_sweep()
        d = result.to_dict()
        assert "rows" in d
        assert len(d["rows"]) == 18

    def test_convenience_function(self):
        result = run_dbc_sweep()
        assert isinstance(result, DBCSweepResult)


class TestEdgeCases:
    """Edge cases for sweep."""

    @pytest.fixture
    def sweep(self):
        return DBCWeightSweep()

    def test_zero_weight_is_baseline(self, sweep):
        data = sweep._generate_test_data()
        spy_r = sweep._compute_returns(data["SPY"])
        gld_r = sweep._compute_returns(data["GLD"])
        tlt_r = sweep._compute_returns(data["TLT"])
        dbc_r = sweep._compute_returns(data["DBC"])

        c1, v1, s1, d1 = sweep._simulate_portfolio(
            spy_r, gld_r, tlt_r, dbc_r, 0.0, "gld"
        )
        c2, v2, s2, d2 = sweep._simulate_portfolio(
            spy_r, gld_r, tlt_r, dbc_r, 0.0, "spy"
        )
        assert abs(c1 - c2) < 0.01

    def test_max_dd_negative(self, sweep):
        result = sweep.run_sweep()
        assert result.baseline_max_dd < 0  # Max DD should be negative
        for row in result.rows:
            assert row.max_dd < 0
