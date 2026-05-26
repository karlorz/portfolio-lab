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
