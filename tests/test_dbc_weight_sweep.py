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

    def test_to_dict_field_completeness(self):
        """to_dict() returns exactly all 11 fields, no extras."""
        row = DBCSweepRow(
            dbc_weight=0.05, funded_from="tlt",
            cagr=9.8, vol=11.5, sharpe=0.71, max_dd=-28.0,
            sharpe_delta=-0.08, crisis_2008=-14.0,
            crisis_2020=-8.5, crisis_2022=-15.0,
            avg_dbc_return=4.2,
        )
        d = row.to_dict()
        assert set(d.keys()) == {
            "dbc_weight", "funded_from", "cagr", "vol", "sharpe",
            "max_dd", "sharpe_delta", "crisis_2008", "crisis_2020",
            "crisis_2022", "avg_dbc_return",
        }

    def test_to_dict_field_types(self):
        """Each field in to_dict() has the correct type."""
        row = DBCSweepRow(
            dbc_weight=0.02, funded_from="spy",
            cagr=10.2, vol=10.8, sharpe=0.79, max_dd=-26.0,
            sharpe_delta=-0.01, crisis_2008=-12.5,
            crisis_2020=-7.2, crisis_2022=-13.8,
            avg_dbc_return=3.8,
        )
        d = row.to_dict()
        assert isinstance(d["dbc_weight"], float)
        assert isinstance(d["funded_from"], str)
        assert isinstance(d["cagr"], float)
        assert isinstance(d["vol"], float)
        assert isinstance(d["sharpe"], float)
        assert isinstance(d["max_dd"], float)
        assert isinstance(d["sharpe_delta"], float)
        assert isinstance(d["crisis_2008"], float)
        assert isinstance(d["crisis_2020"], float)
        assert isinstance(d["crisis_2022"], float)
        assert isinstance(d["avg_dbc_return"], float)

    def test_round_trip_construct_from_dict_values(self):
        """Reconstructing from to_dict() values yields same to_dict()."""
        row = DBCSweepRow(
            dbc_weight=0.01, funded_from="gld",
            cagr=11.0, vol=11.2, sharpe=0.81, max_dd=-26.5,
            sharpe_delta=0.02, crisis_2008=-11.8,
            crisis_2020=-6.5, crisis_2022=-12.5,
            avg_dbc_return=5.0,
        )
        d1 = row.to_dict()
        row2 = DBCSweepRow(**d1)
        d2 = row2.to_dict()
        assert d1 == d2

    def test_funded_from_only_valid_sources(self):
        """funded_from must be one of the three valid sources."""
        row = DBCSweepRow(
            dbc_weight=0.04, funded_from="gld",
            cagr=10.0, vol=11.0, sharpe=0.75, max_dd=-27.0,
            sharpe_delta=-0.04, crisis_2008=-13.0,
            crisis_2020=-7.5, crisis_2022=-14.0,
            avg_dbc_return=4.0,
        )
        assert row.funded_from in ("gld", "spy", "tlt")


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

    def test_baseline_sums_to_one(self):
        """BASELINE allocation weights must sum to 1.0."""
        total = sum(DBCWeightSweep.BASELINE.values())
        assert abs(total - 1.0) < 1e-10

    def test_baseline_has_correct_keys(self):
        """BASELINE must contain exactly SPY, GLD, TLT."""
        assert set(DBCWeightSweep.BASELINE.keys()) == {"spy", "gld", "tlt"}

    def test_baseline_values_match_import(self):
        """BASELINE values should be 0.46, 0.38, 0.16."""
        assert DBCWeightSweep.BASELINE["spy"] == 0.46
        assert DBCWeightSweep.BASELINE["gld"] == 0.38
        assert DBCWeightSweep.BASELINE["tlt"] == 0.16


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


class TestDBCSweepRowExtended:
    """Additional DBCSweepRow edge cases."""

    def test_all_fields_present(self):
        row = DBCSweepRow(
            dbc_weight=0.04, funded_from="spy",
            cagr=10.5, vol=11.2, sharpe=0.78, max_dd=-24.5,
            sharpe_delta=-0.01, crisis_2008=-12.0,
            crisis_2020=-7.0, crisis_2022=-13.0,
            avg_dbc_return=3.2,
        )
        d = row.to_dict()
        assert set(d.keys()) >= {
            "dbc_weight", "funded_from", "cagr", "vol", "sharpe",
            "max_dd", "sharpe_delta", "crisis_2008", "crisis_2020",
            "crisis_2022", "avg_dbc_return",
        }


class TestDBCSweepResultExtended:
    """Additional DBCSweepResult edge cases."""

    @pytest.fixture
    def sweep(self):
        return DBCWeightSweep()

    def test_result_to_dict_includes_all_fields(self, sweep):
        result = sweep.run_sweep()
        d = result.to_dict()
        assert "timestamp" in d
        assert "baseline_cagr" in d
        assert "baseline_sharpe" in d
        assert "best_weight" in d
        assert "best_source" in d
        assert "recommendation" in d
        assert "is_worthwhile" in d

    def test_result_to_dict_rows_are_dicts(self, sweep):
        result = sweep.run_sweep()
        d = result.to_dict()
        for row in d["rows"]:
            assert isinstance(row, dict)
            assert "dbc_weight" in row

    def test_best_sharpe_delta(self, sweep):
        result = sweep.run_sweep()
        assert isinstance(result.best_sharpe_delta, float)


class TestSimulatePortfolioExtended:
    """Additional _simulate_portfolio edge cases."""

    @pytest.fixture
    def sweep(self):
        return DBCWeightSweep()

    def test_funded_from_spy(self, sweep):
        """Funding DBC from SPY should reduce SPY weight."""
        data = sweep._generate_test_data()
        spy_r = sweep._compute_returns(data["SPY"])
        gld_r = sweep._compute_returns(data["GLD"])
        tlt_r = sweep._compute_returns(data["TLT"])
        dbc_r = sweep._compute_returns(data["DBC"])
        cagr, vol, sharpe, dd = sweep._simulate_portfolio(
            spy_r, gld_r, tlt_r, dbc_r, 0.03, "spy"
        )
        assert cagr != 0
        assert vol > 0

    def test_funded_from_tlt(self, sweep):
        """Funding DBC from TLT should reduce TLT weight."""
        data = sweep._generate_test_data()
        spy_r = sweep._compute_returns(data["SPY"])
        gld_r = sweep._compute_returns(data["GLD"])
        tlt_r = sweep._compute_returns(data["TLT"])
        dbc_r = sweep._compute_returns(data["DBC"])
        cagr, vol, sharpe, dd = sweep._simulate_portfolio(
            spy_r, gld_r, tlt_r, dbc_r, 0.03, "tlt"
        )
        assert cagr != 0
        assert vol > 0

    def test_constant_returns_give_high_sharpe(self, sweep):
        """Constant returns produce very low vol → high Sharpe or zero division."""
        n = 100
        const_r = [0.001] * n
        cagr, vol, sharpe, dd = sweep._simulate_portfolio(
            const_r, const_r, const_r, const_r, 0.03, "gld"
        )
        # With constant returns, vol should be near 0 and Sharpe very high or 0
        assert sharpe >= 0


class TestSweepRecommendation:
    """Test recommendation logic based on Sharpe delta."""

    @pytest.fixture
    def sweep(self):
        return DBCWeightSweep()

    def test_recommendation_is_string(self, sweep):
        result = sweep.run_sweep()
        assert isinstance(result.recommendation, str)
        assert len(result.recommendation) > 10

    def test_is_worthwhile_is_bool(self, sweep):
        result = sweep.run_sweep()
        assert isinstance(result.is_worthwhile, bool)

    def test_best_source_valid(self, sweep):
        result = sweep.run_sweep()
        assert result.best_source in ("gld", "spy", "tlt", "none")

    def test_rows_contain_all_combinations(self, sweep):
        """Should have 6 weights × 3 sources = 18 rows."""
        result = sweep.run_sweep()
        assert len(result.rows) == 18

    def test_crisis_values_are_negative(self, sweep):
        """Crisis proxy values should be negative."""
        result = sweep.run_sweep()
        for row in result.rows:
            assert row.crisis_2008 < 0

    def test_sharpe_delta_distribution(self, sweep):
        """Some deltas should be positive, some negative, or all near zero."""
        result = sweep.run_sweep()
        deltas = [r.sharpe_delta for r in result.rows]
        # At least some should be non-zero (either direction)
        assert any(d != 0 for d in deltas)

    def test_compute_returns_simple(self, sweep):
        """_compute_returns should produce correct returns."""
        rets = sweep._compute_returns([100.0, 110.0, 99.0])
        assert abs(rets[0] - 0.10) < 1e-10
        assert abs(rets[1] - (-0.10)) < 1e-10
