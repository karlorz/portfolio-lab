#!/usr/bin/env python3
"""Tests for src/backtest/walk_forward_champion.py."""

import numpy as np
import pytest


class TestWalkForwardWindow:
    """Tests for WalkForwardWindow dataclass."""

    def test_window_creation(self):
        from src.backtest.walk_forward_champion import WalkForwardWindow

        window = WalkForwardWindow(
            window_index=0,
            is_start="2006-01-01", is_end="2010-12-31",
            oos_start="2011-01-01", oos_end="2011-12-31",
            is_days=1258, oos_days=252,
            champion_is_sharpe=0.75, champion_oos_sharpe=0.82,
            champion_oos_cagr=10.5, champion_oos_max_dd=-12.3,
            challenger_is_sharpe=0.78, challenger_oos_sharpe=0.80,
            challenger_oos_cagr=9.8, challenger_oos_max_dd=-11.5,
            benchmark_spy_oos_sharpe=0.65,
            benchmark_6040_oos_sharpe=0.70,
        )
        d = window.to_dict()
        assert d["window_index"] == 0
        assert d["is_start"] == "2006-01-01"
        assert d["champion_oos_sharpe"] == 0.82
        assert d["challenger_oos_sharpe"] == 0.80
        assert d["benchmark_spy_oos_sharpe"] == 0.65

    def test_window_to_dict_has_all_fields(self):
        from src.backtest.walk_forward_champion import WalkForwardWindow

        window = WalkForwardWindow(
            window_index=1,
            is_start="2006-01-01", is_end="2011-12-31",
            oos_start="2012-01-01", oos_end="2012-12-31",
            is_days=1500, oos_days=250,
            champion_is_sharpe=0.80, champion_oos_sharpe=0.90,
            champion_oos_cagr=11.0, champion_oos_max_dd=-9.0,
            challenger_is_sharpe=0.82, challenger_oos_sharpe=0.88,
            challenger_oos_cagr=10.5, challenger_oos_max_dd=-8.5,
            benchmark_spy_oos_sharpe=0.88,
            benchmark_6040_oos_sharpe=0.78,
        )
        d = window.to_dict()
        expected_keys = {
            "window_index", "is_start", "is_end", "oos_start", "oos_end",
            "is_days", "oos_days",
            "champion_is_sharpe", "champion_oos_sharpe",
            "champion_oos_cagr", "champion_oos_max_dd",
            "challenger_is_sharpe", "challenger_oos_sharpe",
            "challenger_oos_cagr", "challenger_oos_max_dd",
            "benchmark_spy_oos_sharpe", "benchmark_6040_oos_sharpe",
        }
        assert expected_keys == set(d.keys())


class TestWalkForwardResult:
    """Tests for WalkForwardResult dataclass."""

    def test_result_creation(self):
        from src.backtest.walk_forward_champion import WalkForwardResult, WalkForwardWindow

        windows = [
            WalkForwardWindow(
                window_index=0,
                is_start="2006-01-01", is_end="2010-12-31",
                oos_start="2011-01-01", oos_end="2011-12-31",
                is_days=1258, oos_days=252,
                champion_is_sharpe=0.75, champion_oos_sharpe=0.82,
                champion_oos_cagr=10.5, champion_oos_max_dd=-12.3,
                challenger_is_sharpe=0.78, challenger_oos_sharpe=0.80,
                challenger_oos_cagr=9.8, challenger_oos_max_dd=-11.5,
                benchmark_spy_oos_sharpe=0.65,
                benchmark_6040_oos_sharpe=0.70,
            ),
        ]
        result = WalkForwardResult(
            analysis_date="2026-05-26",
            data_range="2005-2026",
            n_windows=1,
            windows=windows,
            allocation_label="Champion (46/38/16)",
            mean_oos_sharpe=0.82,
            mean_oos_cagr=10.5,
            mean_oos_max_dd=-12.3,
            is_sharpe=0.90,
            wfe=0.91,
            benchmark_spy_mean_oos_sharpe=0.65,
            benchmark_6040_mean_oos_sharpe=0.70,
            oos_sharpe_positive_pct=1.0,
            beats_spy=1,
            beats_6040=1,
            summary="WFE=0.91 — validated.",
        )
        d = result.to_dict()
        assert d["n_windows"] == 1
        assert d["wfe"] == 0.91
        assert d["is_sharpe"] == 0.90
        assert d["allocation_label"] == "Champion (46/38/16)"
        assert len(d["windows"]) == 1

    def test_result_empty_windows(self):
        from src.backtest.walk_forward_champion import WalkForwardResult

        result = WalkForwardResult(
            analysis_date="2026-05-26",
            data_range="2005-2026",
            n_windows=0,
            windows=[],
            allocation_label="Test",
            mean_oos_sharpe=0.0,
            mean_oos_cagr=0.0,
            mean_oos_max_dd=0.0,
            is_sharpe=0.90,
            wfe=0.0,
            benchmark_spy_mean_oos_sharpe=0.0,
            benchmark_6040_mean_oos_sharpe=0.0,
            oos_sharpe_positive_pct=0.0,
            beats_spy=0,
            beats_6040=0,
            summary="No windows to validate.",
        )
        d = result.to_dict()
        assert d["n_windows"] == 0


class TestChampionVsChallengerResult:
    """Tests for ChampionVsChallengerResult dataclass."""

    def test_comparison_creation(self):
        from src.backtest.walk_forward_champion import (
            ChampionVsChallengerResult, WalkForwardResult, WalkForwardWindow,
        )

        windows = [
            WalkForwardWindow(
                window_index=0,
                is_start="2006-01-01", is_end="2010-12-31",
                oos_start="2011-01-01", oos_end="2011-12-31",
                is_days=1258, oos_days=252,
                champion_is_sharpe=0.75, champion_oos_sharpe=0.82,
                champion_oos_cagr=10.5, champion_oos_max_dd=-12.3,
                challenger_is_sharpe=0.78, challenger_oos_sharpe=0.80,
                challenger_oos_cagr=9.8, challenger_oos_max_dd=-11.5,
                benchmark_spy_oos_sharpe=0.65,
                benchmark_6040_oos_sharpe=0.70,
            ),
        ]
        champion = WalkForwardResult(
            analysis_date="2026-05-26", data_range="2005-2026", n_windows=1,
            windows=windows, allocation_label="Champion (46/38/16)",
            mean_oos_sharpe=0.82, mean_oos_cagr=10.5, mean_oos_max_dd=-12.3,
            is_sharpe=0.90, wfe=0.91,
            benchmark_spy_mean_oos_sharpe=0.65, benchmark_6040_mean_oos_sharpe=0.70,
            oos_sharpe_positive_pct=1.0, beats_spy=1, beats_6040=1,
            summary="WFE=0.91.",
        )
        challenger = WalkForwardResult(
            analysis_date="2026-05-26", data_range="2005-2026", n_windows=1,
            windows=windows, allocation_label="Challenger (44/36/20)",
            mean_oos_sharpe=0.80, mean_oos_cagr=9.8, mean_oos_max_dd=-11.5,
            is_sharpe=0.92, wfe=0.87,
            benchmark_spy_mean_oos_sharpe=0.65, benchmark_6040_mean_oos_sharpe=0.70,
            oos_sharpe_positive_pct=1.0, beats_spy=0, beats_6040=1,
            summary="WFE=0.87.",
        )
        comp = ChampionVsChallengerResult(
            analysis_date="2026-05-26", data_range="2005-2026", n_windows=1,
            windows=windows, champion=champion, challenger=challenger,
            champion_beats_challenger=1, challenger_beats_champion=0,
            better_wfe="champion", wfe_delta=-0.04,
            recommendation="Champion wins.",
        )
        d = comp.to_dict()
        assert d["n_windows"] == 1
        assert d["champion_beats_challenger"] == 1
        assert d["challenger_beats_champion"] == 0
        assert d["better_wfe"] == "champion"
        assert d["champion"]["allocation_label"] == "Champion (46/38/16)"
        assert d["challenger"]["wfe"] == 0.87


class TestPortfolioReturns:
    """Tests for _compute_portfolio_returns helper."""

    def test_all_spy_matches_spy(self):
        from src.backtest.walk_forward_champion import _compute_portfolio_returns

        prices_data = {
            "SPY": {"dates": ["2020-01-02", "2020-01-03", "2020-01-04"],
                     "prices": [100.0, 101.0, 103.0]},
            "GLD": {"dates": ["2020-01-02", "2020-01-03", "2020-01-04"],
                    "prices": [150.0, 149.0, 151.0]},
            "TLT": {"dates": ["2020-01-02", "2020-01-03", "2020-01-04"],
                    "prices": [100.0, 101.0, 100.0]},
        }
        rets = _compute_portfolio_returns(prices_data, {"SPY": 1.0, "GLD": 0.0, "TLT": 0.0})
        assert len(rets) == 2
        assert abs(rets[0] - 0.01) < 0.001
        assert abs(rets[1] - 0.01980198) < 0.001

    def test_weighted_portfolio(self):
        from src.backtest.walk_forward_champion import _compute_portfolio_returns

        prices_data = {
            "SPY": {"dates": ["d1", "d2", "d3"],
                     "prices": [100.0, 102.0, 101.0]},
            "GLD": {"dates": ["d1", "d2", "d3"],
                    "prices": [150.0, 148.0, 152.0]},
            "TLT": {"dates": ["d1", "d2", "d3"],
                    "prices": [100.0, 101.0, 100.5]},
        }
        rets = _compute_portfolio_returns(prices_data, {"SPY": 0.5, "GLD": 0.5, "TLT": 0.0})
        assert abs(rets[0] - 0.003333) < 0.01
        assert abs(rets[1] - 0.0086) < 0.01

    def test_zero_weight_asset_ignored(self):
        from src.backtest.walk_forward_champion import _compute_portfolio_returns

        prices_data = {
            "SPY": {"dates": ["d1", "d2"],
                     "prices": [100.0, 101.0]},
            "GLD": {"dates": ["d1", "d2"],
                    "prices": [150.0, 149.0]},
            "TLT": {"dates": ["d1", "d2"],
                    "prices": [100.0, 101.0]},
        }
        rets = _compute_portfolio_returns(prices_data, {"SPY": 1.0, "GLD": 0.0, "TLT": 0.0})
        assert len(rets) == 1
        assert abs(rets[0] - 0.01) < 0.001


class TestReturnsToEquity:
    """Tests for _returns_to_equity helper."""

    def test_converts_returns(self):
        from src.backtest.walk_forward_champion import _returns_to_equity
        import numpy as np

        rets = np.array([0.01, -0.005, 0.02])
        equity = _returns_to_equity(rets, initial=100000.0)
        assert equity[0] == 100000.0
        assert abs(equity[1] - 101000.0) < 0.01
        assert abs(equity[2] - 100495.0) < 0.01
        assert abs(equity[3] - 102504.9) < 0.01

    def test_no_returns(self):
        from src.backtest.walk_forward_champion import _returns_to_equity
        import numpy as np

        equity = _returns_to_equity(np.array([]))
        assert equity == [100000.0]


class TestWalkForwardIntegration:
    """Integration tests for walk-forward validation."""

    def test_champion_runs_and_produces_windows(self):
        from src.backtest.walk_forward_champion import run_walk_forward_champion

        result = run_walk_forward_champion(is_years=5, oos_years=1)
        assert result.n_windows > 5
        assert result.is_sharpe > 0.5
        assert result.wfe > 0

        for w in result.windows:
            assert w.is_days > 500
            assert w.oos_days > 50

    def test_comparison_runs_and_has_both_allocations(self):
        from src.backtest.walk_forward_champion import run_walk_forward_comparison

        comp = run_walk_forward_comparison(is_years=5, oos_years=1)
        assert comp.n_windows > 5
        assert comp.champion.n_windows == comp.challenger.n_windows
        assert comp.champion.wfe > 0
        assert comp.challenger.wfe > 0
        assert comp.champion_beats_challenger + comp.challenger_beats_champion == comp.n_windows
        assert comp.better_wfe in ("champion", "challenger", "tie")

    def test_window_dates_are_contiguous(self):
        from src.backtest.walk_forward_champion import run_walk_forward_champion

        result = run_walk_forward_champion(is_years=5, oos_years=1)
        for i in range(1, len(result.windows)):
            prev = result.windows[i - 1]
            curr = result.windows[i]
            assert curr.is_start >= prev.is_start

    def test_sharpe_values_are_finite(self):
        from src.backtest.walk_forward_champion import run_walk_forward_comparison

        comp = run_walk_forward_comparison(is_years=5, oos_years=1)
        for w in comp.windows:
            for val in [w.champion_is_sharpe, w.champion_oos_sharpe,
                         w.challenger_is_sharpe, w.challenger_oos_sharpe,
                         w.benchmark_spy_oos_sharpe, w.benchmark_6040_oos_sharpe]:
                assert np.isfinite(val)

    def test_both_allocations_validated(self):
        """Both champion and challenger should have WFE close to or above 1.0."""
        from src.backtest.walk_forward_champion import run_walk_forward_comparison

        comp = run_walk_forward_comparison(is_years=5, oos_years=1)
        # Both should survive walk-forward (WFE > 0.80 for "mostly validated")
        assert comp.champion.wfe > 0.80, f"Champion WFE={comp.champion.wfe:.4f} too low"
        assert comp.challenger.wfe > 0.80, f"Challenger WFE={comp.challenger.wfe:.4f} too low"


class TestComputeAllocationResult:
    """Tests for _compute_allocation_result (previously untested)."""

    def _make_window(self, oos_sharpe=0.5, oos_cagr=0.08, oos_dd=-0.15,
                     spy_sharpe=0.3, sf_sharpe=0.4):
        from src.backtest.walk_forward_champion import WalkForwardWindow
        return WalkForwardWindow(
            window_index=0, is_start="2010-01-01", is_end="2014-12-31",
            oos_start="2015-01-01", oos_end="2015-12-31",
            is_days=1260, oos_days=252,
            champion_is_sharpe=0.9, champion_oos_sharpe=oos_sharpe,
            champion_oos_cagr=oos_cagr, champion_oos_max_dd=oos_dd,
            challenger_is_sharpe=0.85, challenger_oos_sharpe=oos_sharpe * 0.95,
            challenger_oos_cagr=oos_cagr * 0.95, challenger_oos_max_dd=oos_dd * 1.1,
            benchmark_spy_oos_sharpe=spy_sharpe,
            benchmark_6040_oos_sharpe=sf_sharpe,
        )

    def test_empty_windows_returns_zero(self):
        from src.backtest.walk_forward_champion import _compute_allocation_result
        result = _compute_allocation_result([], "test", 0.9, "champion_oos_sharpe", "champion_oos_cagr", "champion_oos_max_dd")
        assert result.n_windows == 0
        assert result.wfe == 0.0
        assert result.mean_oos_sharpe == 0.0

    def test_wfe_above_one_validated(self):
        from src.backtest.walk_forward_champion import _compute_allocation_result
        w = self._make_window(oos_sharpe=1.0)
        result = _compute_allocation_result([w], "champion", 0.8, "champion_oos_sharpe", "champion_oos_cagr", "champion_oos_max_dd")
        assert result.wfe >= 1.0
        assert "VALIDATED" in result.summary

    def test_wfe_below_one_not_validated(self):
        from src.backtest.walk_forward_champion import _compute_allocation_result
        w = self._make_window(oos_sharpe=0.3)
        result = _compute_allocation_result([w], "champion", 1.0, "champion_oos_sharpe", "champion_oos_cagr", "champion_oos_max_dd")
        assert result.wfe < 0.80
        assert "NOT VALIDATED" in result.summary

    def test_wfe_between_08_and_1_mostly_validated(self):
        from src.backtest.walk_forward_champion import _compute_allocation_result
        w = self._make_window(oos_sharpe=0.75)
        result = _compute_allocation_result([w], "champion", 0.9, "champion_oos_sharpe", "champion_oos_cagr", "champion_oos_max_dd")
        assert 0.80 <= result.wfe < 1.0
        assert "MOSTLY VALIDATED" in result.summary

    def test_beats_spy_count(self):
        from src.backtest.walk_forward_champion import _compute_allocation_result
        w = self._make_window(oos_sharpe=0.5, spy_sharpe=0.3)
        result = _compute_allocation_result([w], "champion", 0.9, "champion_oos_sharpe", "champion_oos_cagr", "champion_oos_max_dd")
        assert result.beats_spy == 1

    def test_positive_oos_ratio(self):
        from src.backtest.walk_forward_champion import _compute_allocation_result
        w1 = self._make_window(oos_sharpe=0.5)
        w2 = self._make_window(oos_sharpe=-0.2)
        result = _compute_allocation_result([w1, w2], "champion", 0.9, "champion_oos_sharpe", "champion_oos_cagr", "champion_oos_max_dd")
        assert result.oos_sharpe_positive_pct == 0.5

    def test_zero_is_sharpe_returns_zero_wfe(self):
        from src.backtest.walk_forward_champion import _compute_allocation_result
        w = self._make_window(oos_sharpe=0.5)
        result = _compute_allocation_result([w], "test", 0.0, "champion_oos_sharpe", "champion_oos_cagr", "champion_oos_max_dd")
        assert result.wfe == 0.0


class TestWalkForwardResultEdgeCases:
    """Additional integration edge case tests."""

    def test_champion_result_fields(self):
        from src.backtest.walk_forward_champion import run_walk_forward_champion
        result = run_walk_forward_champion(is_years=5, oos_years=1)
        assert hasattr(result, 'wfe')
        assert hasattr(result, 'mean_oos_sharpe')
        assert hasattr(result, 'windows')
        assert hasattr(result, 'beats_spy')
        assert hasattr(result, 'beats_6040')
        assert result.n_windows >= 10

    def test_comparison_has_summary(self):
        from src.backtest.walk_forward_champion import run_walk_forward_comparison
        comp = run_walk_forward_comparison(is_years=5, oos_years=1)
        assert len(comp.champion.summary) > 0
        assert len(comp.challenger.summary) > 0
        assert "WFE" in comp.champion.summary


# ── Edge-case tests (appended) ──────────────────────────────────────


def _make_synthetic_prices(n_days=2000, start_date="2006-01-01", seed=42):
    """Build synthetic price data for SPY/GLD/TLT with deterministic returns."""
    rng = np.random.RandomState(seed)
    dates = []
    from datetime import datetime, timedelta
    dt = datetime.strptime(start_date, "%Y-%m-%d")
    for _ in range(n_days):
        dt += timedelta(days=1)
        while dt.weekday() >= 5:
            dt += timedelta(days=1)
        dates.append(dt.strftime("%Y-%m-%d"))

    def _gen_prices(initial, daily_vol):
        prices = [initial]
        for _ in range(n_days - 1):
            ret = rng.normal(0.0003, daily_vol)
            prices.append(prices[-1] * (1 + ret))
        return prices

    return {
        "SPY": {"dates": dates, "prices": _gen_prices(100.0, 0.012)},
        "GLD": {"dates": dates, "prices": _gen_prices(50.0, 0.010)},
        "TLT": {"dates": dates, "prices": _gen_prices(80.0, 0.008)},
    }


class TestAllocationResultEdgeCases:
    """Edge cases for _compute_allocation_result."""

    def test_single_window_data_range_and_date(self):
        """Single window should produce correct data_range string and analysis_date."""
        from src.backtest.walk_forward_champion import (
            _compute_allocation_result, WalkForwardWindow,
        )
        w = WalkForwardWindow(
            window_index=0, is_start="2010-01-01", is_end="2014-12-31",
            oos_start="2015-01-01", oos_end="2015-12-31",
            is_days=1260, oos_days=252,
            champion_is_sharpe=0.8, champion_oos_sharpe=0.9,
            champion_oos_cagr=0.10, champion_oos_max_dd=-0.12,
            challenger_is_sharpe=0.75, challenger_oos_sharpe=0.85,
            challenger_oos_cagr=0.09, challenger_oos_max_dd=-0.11,
            benchmark_spy_oos_sharpe=0.6,
            benchmark_6040_oos_sharpe=0.7,
        )
        result = _compute_allocation_result(
            [w], "TestAlloc", 0.85,
            "champion_oos_sharpe", "champion_oos_cagr", "champion_oos_max_dd",
        )
        assert result.data_range == "2010-01-01 to 2015-12-31"
        assert result.analysis_date  # non-empty timestamp
        assert result.n_windows == 1
        assert result.windows[0] is w

    def test_all_negative_oos_sharpes(self):
        """All negative OOS sharpes → 0% positive, 0 beats benchmarks."""
        from src.backtest.walk_forward_champion import (
            _compute_allocation_result, WalkForwardWindow,
        )
        windows = []
        for i in range(3):
            windows.append(WalkForwardWindow(
                window_index=i, is_start="2010-01-01", is_end="2014-12-31",
                oos_start="2015-01-01", oos_end="2015-12-31",
                is_days=1260, oos_days=252,
                champion_is_sharpe=0.5, champion_oos_sharpe=-0.1 * (i + 1),
                champion_oos_cagr=-0.05, champion_oos_max_dd=-0.30,
                challenger_is_sharpe=0.5, challenger_oos_sharpe=-0.15,
                challenger_oos_cagr=-0.05, challenger_oos_max_dd=-0.30,
                benchmark_spy_oos_sharpe=0.1,
                benchmark_6040_oos_sharpe=0.2,
            ))
        result = _compute_allocation_result(
            windows, "BadAlloc", 0.5,
            "champion_oos_sharpe", "champion_oos_cagr", "champion_oos_max_dd",
        )
        assert result.oos_sharpe_positive_pct == 0.0
        assert result.beats_spy == 0
        assert result.beats_6040 == 0
        assert result.mean_oos_sharpe < 0

    def test_negative_is_sharpe_yields_zero_wfe(self):
        """Negative IS Sharpe should produce WFE=0 (not negative WFE)."""
        from src.backtest.walk_forward_champion import (
            _compute_allocation_result, WalkForwardWindow,
        )
        w = WalkForwardWindow(
            window_index=0, is_start="2010-01-01", is_end="2014-12-31",
            oos_start="2015-01-01", oos_end="2015-12-31",
            is_days=1260, oos_days=252,
            champion_is_sharpe=-0.5, champion_oos_sharpe=0.3,
            champion_oos_cagr=0.05, champion_oos_max_dd=-0.15,
            challenger_is_sharpe=-0.5, challenger_oos_sharpe=0.2,
            challenger_oos_cagr=0.04, challenger_oos_max_dd=-0.16,
            benchmark_spy_oos_sharpe=0.1,
            benchmark_6040_oos_sharpe=0.15,
        )
        result = _compute_allocation_result(
            [w], "NegSharpe", -0.5,
            "champion_oos_sharpe", "champion_oos_cagr", "champion_oos_max_dd",
        )
        assert result.wfe == 0.0
        assert result.is_sharpe == -0.5

    def test_beats_6040_count(self):
        """Verify beats_6040 counting when OOS Sharpe is between SPY and 60/40."""
        from src.backtest.walk_forward_champion import (
            _compute_allocation_result, WalkForwardWindow,
        )
        # OOS Sharpe=0.45 beats 60/40 (0.40) but not SPY (0.50)
        w = WalkForwardWindow(
            window_index=0, is_start="2010-01-01", is_end="2014-12-31",
            oos_start="2015-01-01", oos_end="2015-12-31",
            is_days=1260, oos_days=252,
            champion_is_sharpe=0.8, champion_oos_sharpe=0.45,
            champion_oos_cagr=0.08, champion_oos_max_dd=-0.14,
            challenger_is_sharpe=0.8, challenger_oos_sharpe=0.40,
            challenger_oos_cagr=0.07, challenger_oos_max_dd=-0.15,
            benchmark_spy_oos_sharpe=0.50,
            benchmark_6040_oos_sharpe=0.40,
        )
        result = _compute_allocation_result(
            [w], "MidAlloc", 0.8,
            "champion_oos_sharpe", "champion_oos_cagr", "champion_oos_max_dd",
        )
        assert result.beats_spy == 0
        assert result.beats_6040 == 1


class TestComputePortfolioReturnsEdgeCases:
    """Edge cases for _compute_portfolio_returns."""

    def test_single_asset_only(self):
        """Portfolio with only one asset in prices_data."""
        from src.backtest.walk_forward_champion import _compute_portfolio_returns

        prices = {
            "SPY": {
                "dates": ["d1", "d2", "d3", "d4"],
                "prices": [100.0, 102.0, 101.0, 105.0],
            },
        }
        rets = _compute_portfolio_returns(prices, {"SPY": 1.0})
        assert len(rets) == 3
        np.testing.assert_allclose(rets, [0.02, -0.0098039, 0.0396039], atol=1e-4)

    def test_empty_weights_raises(self):
        """Empty weights dict should raise ValueError from min() on empty sequence."""
        from src.backtest.walk_forward_champion import _compute_portfolio_returns

        prices = {
            "SPY": {"dates": ["d1", "d2"], "prices": [100.0, 101.0]},
        }
        with pytest.raises(ValueError):
            _compute_portfolio_returns(prices, {})

    def test_price_prev_zero_skips_day(self):
        """When prev price is 0, the return contribution is skipped (division guard)."""
        from src.backtest.walk_forward_champion import _compute_portfolio_returns

        prices = {
            "SPY": {
                "dates": ["d1", "d2", "d3"],
                "prices": [0.0, 100.0, 110.0],
            },
        }
        rets = _compute_portfolio_returns(prices, {"SPY": 1.0})
        assert len(rets) == 2
        # First day: prev=0 → skipped, ret=0.0
        assert rets[0] == 0.0
        # Second day: (110/100 - 1) = 0.10
        np.testing.assert_allclose(rets[1], 0.10, atol=1e-6)

    def test_equal_weights_split(self):
        """Two assets with equal 50/50 weights."""
        from src.backtest.walk_forward_champion import _compute_portfolio_returns

        prices = {
            "SPY": {"dates": ["d1", "d2"], "prices": [100.0, 110.0]},
            "GLD": {"dates": ["d1", "d2"], "prices": [200.0, 190.0]},
        }
        rets = _compute_portfolio_returns(prices, {"SPY": 0.5, "GLD": 0.5})
        # SPY: +10%, GLD: -5% → weighted: 0.5*0.10 + 0.5*(-0.05) = 0.025
        np.testing.assert_allclose(rets[0], 0.025, atol=1e-6)


class TestReturnsToEquityEdgeCases:
    """Edge cases for _returns_to_equity."""

    def test_all_zero_returns(self):
        """All zero returns → equity stays flat at initial."""
        from src.backtest.walk_forward_champion import _returns_to_equity

        equity = _returns_to_equity(np.zeros(5), initial=50000.0)
        assert len(equity) == 6
        assert all(v == 50000.0 for v in equity)

    def test_all_negative_returns(self):
        """All negative returns → equity decays monotonically."""
        from src.backtest.walk_forward_champion import _returns_to_equity

        rets = np.array([-0.01, -0.02, -0.03])
        equity = _returns_to_equity(rets, initial=100000.0)
        assert equity[0] == 100000.0
        # Each step is strictly less than the previous
        for i in range(1, len(equity)):
            assert equity[i] < equity[i - 1]
        # Final value should be 100000 * 0.99 * 0.98 * 0.97
        np.testing.assert_allclose(equity[-1], 100000.0 * 0.99 * 0.98 * 0.97, atol=0.01)

    def test_single_return(self):
        """Single return element → equity has exactly 2 entries."""
        from src.backtest.walk_forward_champion import _returns_to_equity

        equity = _returns_to_equity(np.array([0.05]), initial=10000.0)
        assert len(equity) == 2
        assert equity[0] == 10000.0
        np.testing.assert_allclose(equity[1], 10500.0, atol=0.01)

    def test_100_percent_loss(self):
        """-100% return → equity goes to zero."""
        from src.backtest.walk_forward_champion import _returns_to_equity

        equity = _returns_to_equity(np.array([-1.0]), initial=100000.0)
        assert equity[-1] == 0.0

    def test_large_positive_returns_compound(self):
        """Large positive returns compound correctly."""
        from src.backtest.walk_forward_champion import _returns_to_equity

        rets = np.array([0.50, 0.50])  # 50% + 50%
        equity = _returns_to_equity(rets, initial=100.0)
        # 100 * 1.5 * 1.5 = 225
        np.testing.assert_allclose(equity[-1], 225.0, atol=0.01)


class TestWalkForwardComparisonEdgeCases:
    """Edge cases for run_walk_forward_comparison with synthetic data."""

    def test_zero_valid_windows(self):
        """Tiny dataset produces zero walk-forward windows."""
        from src.backtest.walk_forward_champion import (
            run_walk_forward_comparison,
        )

        # 1500 days: is_end_idx=1260, while guard 1260+252=1512 < 1500 is false → 0 windows
        prices = _make_synthetic_prices(n_days=1500)

        def mock_load():
            return prices

        import src.backtest.walk_forward_champion as mod
        orig = mod._load_prices
        mod._load_prices = mock_load
        try:
            comp = run_walk_forward_comparison(is_years=5, oos_years=1)
            assert comp.n_windows == 0
            assert comp.champion.n_windows == 0
            assert comp.challenger.n_windows == 0
            assert comp.champion_beats_challenger == 0
            assert comp.challenger_beats_champion == 0
            assert "No windows" in comp.champion.summary
            assert "No windows" in comp.challenger.summary
        finally:
            mod._load_prices = orig

    def test_identical_wfe_yields_tie(self):
        """When champion and challenger produce identical WFEs, result is 'tie'."""
        from src.backtest.walk_forward_champion import (
            run_walk_forward_comparison, WalkForwardResult,
        )
        import src.backtest.walk_forward_champion as mod

        prices = _make_synthetic_prices(n_days=3000)

        # Build a single canonical result that both champion and challenger will share
        canonical = WalkForwardResult(
            analysis_date="2026-01-01", data_range="2006-2017",
            n_windows=6, windows=[],
            allocation_label="Identical", mean_oos_sharpe=1.0,
            mean_oos_cagr=0.10, mean_oos_max_dd=-0.15, is_sharpe=0.8,
            wfe=1.25,  # Same WFE for both
            benchmark_spy_mean_oos_sharpe=0.5,
            benchmark_6040_mean_oos_sharpe=0.6,
            oos_sharpe_positive_pct=1.0, beats_spy=5, beats_6040=4,
            summary="WFE=1.25 — IDENTICAL.",
        )

        # Patch _compute_allocation_result so both calls return the same result
        orig_alloc = mod._compute_allocation_result
        mod._compute_allocation_result = lambda *a, **kw: canonical
        orig_load = mod._load_prices
        mod._load_prices = lambda: prices
        try:
            comp = run_walk_forward_comparison(is_years=5, oos_years=1)
            assert comp.n_windows > 0
            assert comp.better_wfe == "tie"
            assert abs(comp.wfe_delta) < 0.02
            # Identical WFEs → tie; head-to-head depends on per-window floats
        finally:
            mod._compute_allocation_result = orig_alloc
            mod._load_prices = orig_load

    def test_champion_delegates_to_comparison(self):
        """run_walk_forward_champion returns exactly comparison.champion."""
        from src.backtest.walk_forward_champion import (
            run_walk_forward_champion, run_walk_forward_comparison,
        )

        prices = _make_synthetic_prices(n_days=3000)

        import src.backtest.walk_forward_champion as mod
        orig = mod._load_prices
        mod._load_prices = lambda: prices
        try:
            champ = run_walk_forward_champion(is_years=5, oos_years=1)
            comp = run_walk_forward_comparison(is_years=5, oos_years=1)
            # Same allocation label, same n_windows, same WFE
            assert champ.allocation_label == comp.champion.allocation_label
            assert champ.n_windows == comp.champion.n_windows
            assert champ.wfe == comp.champion.wfe
            assert champ.mean_oos_sharpe == comp.champion.mean_oos_sharpe
        finally:
            mod._load_prices = orig

    def test_comparison_recommendation_text(self):
        """Verify recommendation text changes based on challenger WFE."""
        from src.backtest.walk_forward_champion import run_walk_forward_comparison

        prices = _make_synthetic_prices(n_days=3000)

        import src.backtest.walk_forward_champion as mod
        orig = mod._load_prices
        mod._load_prices = lambda: prices
        try:
            comp = run_walk_forward_comparison(is_years=5, oos_years=1)
            # With synthetic data and default allocations, recommendation should exist
            assert len(comp.recommendation) > 0
            # Should mention one of the known recommendation patterns
            assert any(kw in comp.recommendation for kw in [
                "PROMOTABLE", "VALIDATED BUT INFERIOR",
                "ACCEPTABLE", "NOT VALIDATED",
            ])
        finally:
            mod._load_prices = orig
